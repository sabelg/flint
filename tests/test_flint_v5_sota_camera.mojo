"""Area D (sota:camera): physical camera, scatter-as-gather DoF, lens flare / dirt / CA / optical vignette,
sensor noise (`test_flint_v5_sota_camera.mojo`). Zero external assets.

1. SPIR-V validation of every new shader.
2. CPU references: EV100 / exposure triangle / FOV / thin-lens CoC / hyperfocal / sensor noise / thin-film
   coating / aperture polygon, and the full-res CPU DoF reference (energy conservation, disc radius,
   near-field spread, no far-field bleed), the VRS tile-rate reference and the procedural dirt texture.
3. Live GPU (skips without Vulkan): in-focus identity, bokeh disc radius vs the thin lens, energy
   conservation, image diff of the GPU DoF against the CPU reference on the same input, near-field spread,
   foreground protection, physical exposure (EV100), lateral CA, optical vignetting vs f-stop, ISO noise
   vs the photon model, lens ghosts at the mirrored position, procedural dirt texture, anamorphic streak,
   compute VRS (coarse rate on flat regions, full rate on edges, output vs full-rate shading), and GPU
   time budgets (1280x720).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, acos, log10, max, min, sqrt
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.camera.physical import (
    PHYS_EXPOSURE_APERTURE_PRIORITY,
    PHYS_EXPOSURE_MANUAL,
    PhysicalCameraModel,
)
from flint.camera.rig import CameraRig
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB, srgb_u8_to_linear
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.sota_camera_post import (
    SotaCameraPost,
    SotaDofLayout,
    aperture_poly_factor_cpu,
    eval_coc_px_cpu,
    eval_dof_cpu,
    ghost_tint_cpu,
    lateral_ca_scale_cpu,
    optical_vignette_cpu,
    sota_dof_composite_shader,
    sota_dof_downsample_shader,
    sota_dof_gather_shader,
    sota_dof_postfilter_shader,
    sota_dof_setup_shader,
    sota_dof_tile_shader,
    sota_flare_blur_shader,
    sota_flare_bright_shader,
    sota_flare_ghost_shader,
    sota_ca_shader,
    sota_lens_composite_shader,
    sota_lens_layout,
    sota_streak_shader,
    thin_film_reflectance_cpu,
)
from flint.render.sota_camera_dirt import (
    SOTA_DIRT_H,
    SOTA_DIRT_W,
    SOTA_DIRT_WORDS,
    sota_dirt_offset,
    sota_lens_dirt_texture_cpu,
    sota_pack_dirt_u8,
)
from flint.render.sota_camera_vrs import (
    SOTA_VRS_RATE_1X1,
    SOTA_VRS_RATE_2X2,
    SOTA_VRS_RATE_4X4,
    SotaVrsLayout,
    sota_vrs_classify_shader,
    sota_vrs_fill_shader,
    sota_vrs_rate_shader,
    vrs_block_error_cpu,
    vrs_perceptual_cpu,
    vrs_shaded_fraction,
    vrs_tile_rate_cpu,
)
from flint.render.spirv_validate import validate_spirv_module


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _near(a: Float32, b: Float32, tol: Float32, msg: String):
    if abs(a - b) > tol:
        _fail(msg + " (got " + String(a) + ", want " + String(b) + " +- " + String(tol) + ")")


def _validate(name: String, words: List[UInt32]):
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _fail(name + " SPIR-V invalid: " + (rep.errors[0] if len(rep.errors) > 0 else String("?")))


def _lin(argb: UInt32) -> Vec3:
    return Vec3(
        srgb_u8_to_linear(Int((argb >> 16) & 0xFF)),
        srgb_u8_to_linear(Int((argb >> 8) & 0xFF)),
        srgb_u8_to_linear(Int(argb & 0xFF)),
    )


def _luma(c: Vec3) -> Float32:
    return c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)


def _sphere(r: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(r, r, r))
    return mb^.finish()


def _box(bmin: Vec3, bmax: Vec3) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_beveled_box((bmin + bmax) * Float32(0.5), (bmax - bmin) * Float32(0.5), bevel=0.002)
    return mb^.finish()


def _emissive(c: Vec3) -> AaaMaterial:
    var m = AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
    m.emissive = c
    return m^


# ============================================================================
# 1. SPIR-V
# ============================================================================

def test_spirv():
    _validate("sota_dof_setup_shader", sota_dof_setup_shader())
    _validate("sota_dof_downsample_shader", sota_dof_downsample_shader())
    _validate("sota_dof_tile_shader", sota_dof_tile_shader())
    _validate("sota_dof_gather_shader", sota_dof_gather_shader())
    _validate("sota_dof_postfilter_shader", sota_dof_postfilter_shader())
    _validate("sota_dof_composite_shader", sota_dof_composite_shader())
    _validate("sota_flare_bright_shader", sota_flare_bright_shader())
    _validate("sota_flare_ghost_shader", sota_flare_ghost_shader())
    _validate("sota_flare_blur_shader", sota_flare_blur_shader())
    _validate("sota_lens_composite_shader", sota_lens_composite_shader())
    _validate("sota_ca_shader", sota_ca_shader())
    _validate("sota_streak_shader", sota_streak_shader())
    _validate("sota_vrs_rate_shader", sota_vrs_rate_shader())
    _validate("sota_vrs_classify_shader", sota_vrs_classify_shader())
    _validate("sota_vrs_fill_shader", sota_vrs_fill_shader())
    print("  [PASS] Suite 1: 15 sota:camera compute shaders pass SPIR-V validation")


# ============================================================================
# 2. CPU references
# ============================================================================

def test_cpu_camera_model():
    var cam = PhysicalCameraModel()  # 50 mm f/8 1/125 ISO 100, full frame
    _near(cam.ev100(), 12.9658, 0.001, "EV100 of f/8 1/125 ISO100")
    _near(cam.exposure_scale(), 1.0, 1e-4, "default camera is the unity exposure")
    var sunny = PhysicalCameraModel(fstop=16.0, shutter_s=0.01, iso=100.0)
    _near(sunny.ev100(), 14.6439, 0.001, "Sunny-16 EV100")
    var open = PhysicalCameraModel(fstop=5.6)
    _near(open.exposure_scale(), Float32(64.0 / 31.36), 0.01, "one stop wider = 2x exposure")
    var iso = PhysicalCameraModel(iso=400.0)
    _near(iso.exposure_scale(), 4.0, 1e-3, "ISO x4 = 2 stops")
    _near(cam.ev100_for_exposure(cam.exposure_scale()), cam.ev100(), 1e-4, "exposure <-> EV100 round trip")
    _near(cam.fov_y_deg(), 26.9915, 0.01, "50 mm vertical FOV on a 24 mm sensor")
    _near(cam.fov_x_deg(), 39.5978, 0.01, "50 mm horizontal FOV on a 36 mm sensor")
    var c2 = cam
    c2.set_focal_length_for_fov_y(cam.fov_y_deg())
    _near(c2.focal_length_mm, 50.0, 0.01, "fov -> focal length round trip")
    _assert(cam.fov_y_deg(account_for_focus=True) < cam.fov_y_deg(), "lens breathing: focus at 5 m narrows the FOV")

    # Thin lens: zero at the focus plane, analytic at 2x the focus distance, limit at infinity.
    _near(cam.coc_diameter_mm(5.0), 0.0, 1e-6, "CoC vanishes on the focus plane")
    var f = Float32(0.05)
    var a = f / Float32(8.0)
    var want = a * f * (Float32(10.0) - Float32(5.0)) / (Float32(10.0) * (Float32(5.0) - f)) * Float32(1000.0)
    _near(cam.coc_diameter_mm(10.0), want, 1e-5, "thin-lens CoC at 10 m")
    _assert(cam.coc_diameter_mm(2.0) < Float32(0.0), "near field has a negative CoC")
    _near(cam.coc_radius_px(1.0e6, 1080), cam.coc_infinity_radius_px(1080), 1e-3, "CoC at infinity == K")
    _near(eval_coc_px_cpu(cam.coc_infinity_radius_px(1080), 5.0, 10.0, 64.0), cam.coc_radius_px(10.0, 1080), 1e-3, "K (1 - S/z) form")

    # Hyperfocal: 50 mm f/8 full frame, c = 0.0288 mm -> ~10.9 m; focusing there makes the far limit infinite.
    var hf = cam.hyperfocal_distance_m()
    _near(hf, 10.88, 0.05, "hyperfocal distance of 50 mm f/8")
    var at_h = cam
    at_h.focus_distance_m = hf
    var lim = at_h.depth_of_field_limits_m()
    _near(lim[0], hf * 0.5, 0.05, "near limit at hyperfocal focus = H/2")
    _assert(lim[1] > Float32(1.0e8), "far limit at hyperfocal focus is infinity")

    # Exposure triangle, aperture priority: EV 10 at f/4 -> 1/64 s ISO 100; EV 6 -> shutter pinned at 1/60, auto-ISO.
    var ap = PhysicalCameraModel(fstop=4.0)
    ap.exposure_mode = PHYS_EXPOSURE_APERTURE_PRIORITY
    var t1 = ap.solve_exposure_triangle(10.0)
    _near(t1.shutter_s, Float32(1.0 / 64.0), 1e-5, "aperture priority shutter at EV10")
    _near(t1.iso, 100.0, 0.01, "aperture priority keeps base ISO in good light")
    _near(t1.ev100, 10.0, 1e-3, "solved triangle reproduces the metered EV (bright)")
    var t2 = ap.solve_exposure_triangle(6.0)
    _near(t2.shutter_s, Float32(1.0 / 60.0), 1e-6, "slow-shutter limit")
    _near(t2.iso, 1500.0, 1.0, "auto-ISO picks up the remaining stops")
    _near(t2.ev100, 6.0, 1e-3, "solved triangle reproduces the metered EV (dim)")
    _near(ap.shutter_fraction(Float32(1.0 / 60.0)), Float32(60.0 / 125.0), 1e-4, "shutter fraction")

    # Sensor noise: shot-noise limited -> sigma scales with sqrt(ISO).
    var lo = PhysicalCameraModel(iso=100.0)
    var hi = PhysicalCameraModel(iso=6400.0)
    var ratio = hi.sensor_noise_sigma(0.18) / lo.sensor_noise_sigma(0.18)
    _near(ratio, 8.0, 0.75, "ISO 6400 vs 100 noise ratio ~ sqrt(64) (+ read noise)")
    _assert(hi.sensor_noise_sigma(0.5) > hi.sensor_noise_sigma(0.05), "shot noise grows with signal")

    # Optical vignetting fades 2 stops below wide open.
    var wide = PhysicalCameraModel(fstop=1.4)
    var stopped = PhysicalCameraModel(fstop=2.8)
    wide.max_aperture_fstop = 1.4
    stopped.max_aperture_fstop = 1.4
    _near(wide.optical_vignette_strength(), 1.0, 1e-4, "optical vignette full wide open")
    _near(stopped.optical_vignette_strength(), 0.0, 1e-4, "optical vignette gone 2 stops down")

    # CameraRig carries the lens and can take its FOV from it.
    var rig = CameraRig()
    rig.lens = PhysicalCameraModel(focal_length_mm=85.0)
    rig.use_lens_fov()
    _near(rig.fov_deg, rig.lens.fov_y_deg(), 1e-4, "CameraRig FOV follows the lens")
    print("  [PASS] Suite 2a: EV100, exposure triangle, FOV, thin-lens CoC, hyperfocal, noise, CameraRig lens")


def test_cpu_optics():
    # MgF2 quarter-wave coating: Airy reflectance minimum at the design wavelength, ((n1^2 - n0 n2)/(n1^2 + n0 n2))^2.
    var r550 = thin_film_reflectance_cpu(99.64, 550.0)
    var want = ((Float32(1.38 * 1.38) - Float32(1.52)) / (Float32(1.38 * 1.38) + Float32(1.52)))
    _near(r550, want * want, 2e-4, "quarter-wave MgF2 reflectance at 550 nm")
    _assert(thin_film_reflectance_cpu(99.64, 450.0) > r550 and thin_film_reflectance_cpu(99.64, 680.0) > r550, "coating minimum at 550 nm")
    var t0 = ghost_tint_cpu(0, 99.64)
    _assert(t0.y < t0.x and t0.y < t0.z, "first ghost is magenta (green suppressed)")
    _assert(t0.x < Float32(1e-3), "two coated reflections ~1e-4 of the source")
    # Aperture polygon: vertices at 1, mid-edge at cos(pi/n).
    _near(aperture_poly_factor_cpu(0.0, 6, 0.0, 0.0), 1.0, 1e-4, "hexagon vertex radius (vertex at rotation 0)")
    _near(aperture_poly_factor_cpu(Float32(3.14159265 / 6.0), 6, 0.0, 0.0), Float32(0.8660254), 1e-4, "hexagon mid-edge radius")
    _near(aperture_poly_factor_cpu(0.0, 6, 0.0, 1.0), 1.0, 1e-5, "curvature 1 = circular iris")
    _near(lateral_ca_scale_cpu(0, 0.01), 1.01, 1e-6, "red magnified")
    _near(lateral_ca_scale_cpu(2, 0.01), 0.99, 1e-6, "blue minified")
    _near(optical_vignette_cpu(0.0, 0.0, 1.78, 1.0), 1.0, 1e-6, "no optical vignette on axis")
    _assert(optical_vignette_cpu(1.0, 1.0, 1.78, 1.0) < Float32(0.5), "corner optical vignette")

    # Scratch layouts stay inside the post buffer's per-pixel region (1024 + 10 max_pix words).
    var mp = 1920 * 1080
    var dl = SotaDofLayout(1920, 1080)
    _assert(dl.end_off < 1024 + 10 * mp, "DoF scratch fits")
    var ll = sota_lens_layout(1920, 1080, mp)
    _assert(ll[6] + ll[0] * ll[1] * 4 < 1024 + 10 * mp, "lens scratch fits")
    _assert(ll[3] >= 1024 + 8 * mp + mp // 4, "lens quarter-res buffers clear the saved bloom mip")
    print("  [PASS] Suite 2b: thin-film ghosts, aperture polygon, CA, optical vignette, scratch layout")


def test_cpu_vrs_reference():
    var w = 24
    var h = 16
    # Tiles 0-1: affine ramp (bilinear-exact; the anchor grid of a tile reaches one pixel into its right
    # and bottom neighbours); tile 2: checkerboard (nothing can be interpolated).
    var p = List[Float32](length=w * h, fill=0.0)
    for y in range(h):
        for x in range(w):
            if x < 16:
                p[y * w + x] = Float32(0.2) + Float32(0.01) * Float32(x) + Float32(0.02) * Float32(y)
            else:
                p[y * w + x] = Float32(0.8) if ((x + y) & 1) == 0 else Float32(0.1)
    var e_ramp = vrs_block_error_cpu(p, w, h, 0, 0, 4)
    _assert(e_ramp[1] < 1.0e-5, "bilinear reconstruction of an affine ramp is exact")
    _assert(vrs_tile_rate_cpu(p, w, h, 0, 0, 0.006, 0.0, 8.0, True) == SOTA_VRS_RATE_4X4, "smooth tile -> 4x4")
    _assert(vrs_tile_rate_cpu(p, w, h, 0, 0, 0.006, 0.0, 8.0, False) == SOTA_VRS_RATE_2X2, "4x4 disallowed -> 2x2")
    _assert(vrs_tile_rate_cpu(p, w, h, 2, 0, 0.006, 0.0, 8.0, True) == SOTA_VRS_RATE_1X1, "checkerboard -> full rate")
    # A soft texture: 2x2 fails at rest and passes once motion blur hides it (tolerance x (1 + v / motion_px)).
    var q = List[Float32](length=64, fill=0.0)
    for y in range(8):
        for x in range(8):
            q[y * 8 + x] = Float32(0.5) + Float32(0.02) * Float32(((x * 7 + y * 3) % 5) - 2)
    var rest = vrs_tile_rate_cpu(q, 8, 8, 0, 0, 0.006, 0.0, 8.0, True)
    var moving = vrs_tile_rate_cpu(q, 8, 8, 0, 0, 0.006, 64.0, 8.0, True)
    _assert(rest == SOTA_VRS_RATE_1X1, "fine texture at rest stays full rate")
    _assert(moving > rest, "motion relaxes the rate")
    _near(vrs_perceptual_cpu(1.0, 1.0), 0.5, 1.0e-6, "perceptual encoding l/(1+l)")
    var rates = List[Int]()
    rates.append(SOTA_VRS_RATE_1X1)
    rates.append(SOTA_VRS_RATE_2X2)
    rates.append(SOTA_VRS_RATE_4X4)
    rates.append(SOTA_VRS_RATE_4X4)
    _near(vrs_shaded_fraction(rates), (1.0 + 0.25 + 0.0625 * 2.0) / 4.0, 1.0e-6, "shaded fraction")
    # Layout: shade list and the persistent rate map + skip masks sit inside the post region, apart from
    # each other, below the UI prims at 1024 + 10 * max_pix.
    for res in [(1920, 1080), (1280, 720), (640, 360), (333, 197)]:
        var mp = res[0] * res[1]
        var lay = SotaVrsLayout(res[0], res[1], mp)
        var tiles = lay.tiles_x * lay.tiles_y
        _assert(lay.list_off >= 1024 + 8 * mp + mp // 4, "VRS list after the saved bloom mip")
        _assert(lay.list_end <= lay.rate_off, "VRS list below the rate map")
        _assert(lay.mask_off >= lay.rate_off + tiles, "skip masks after the rate map")
        _assert(lay.mask_off + 2 * tiles <= 1024 + 10 * mp, "skip masks below the UI prims")
    print("  [PASS] Suite 2d: CPU VRS rate reference (ramp exact, checkerboard full rate, motion relax), layout")


def test_cpu_dirt_texture():
    var img = sota_lens_dirt_texture_cpu()
    var again = sota_lens_dirt_texture_cpu()
    var other = sota_lens_dirt_texture_cpu(1234)
    _assert(len(img) == SOTA_DIRT_W * SOTA_DIRT_H, "dirt texture size")
    var mean = Float32(0.0)
    var mx = Float32(0.0)
    var n_film = 0
    var n_dense = 0
    var same = True
    var diff = 0
    for i in range(len(img)):
        var v = img[i]
        mean += v
        mx = max(mx, v)
        if v > 0.02:
            n_film += 1
        if v > 0.3:
            n_dense += 1
        if v != again[i]:
            same = False
        if abs(v - other[i]) > 0.05:
            diff += 1
    var n = Float32(len(img))
    mean /= n
    var f_film = Float32(n_film) / n
    var f_dense = Float32(n_dense) / n
    print("    dirt texture " + String(SOTA_DIRT_W) + "x" + String(SOTA_DIRT_H) + ": mean " + String(mean) + ", max " + String(mx) + ", film coverage " + String(f_film) + ", dense " + String(f_dense))
    _assert(same, "dirt texture is deterministic per seed")
    _assert(Float32(diff) / n > 0.1, "another seed gives another lens")
    _assert(mean > 0.02 and mean < 0.25, "dirt is a light film, not an opaque layer")
    _assert(mx > 0.9, "dust specks reach full density")
    _assert(f_film > 0.3, "haze / smudges cover a good part of the element")
    _assert(f_dense > 0.003 and f_dense < 0.2, "dense marks (specks, smudge cores) are sparse")
    # Spatial structure: smudges are low-frequency, so neighbouring texels correlate strongly.
    var num = Float32(0.0)
    var den = Float32(0.0)
    for y in range(SOTA_DIRT_H):
        for x in range(SOTA_DIRT_W - 1):
            var a = img[y * SOTA_DIRT_W + x] - mean
            var b = img[y * SOTA_DIRT_W + x + 1] - mean
            num += a * b
            den += a * a
    _assert(num / den > 0.5, "dirt has spatial structure (lag-1 autocorrelation > 0.5)")
    var packed = sota_pack_dirt_u8(img)
    _assert(len(packed) == SOTA_DIRT_WORDS, "packed 4 texels / word")
    var t = 12345
    var q = Int((packed[t >> 2] >> UInt32((t & 3) * 8)) & 0xFF)
    _near(Float32(q) / 255.0, img[t], 0.003, "8-bit packing")
    # Layout: the persistent texture sits above the lens buffers and the VRS shade list, below the rate map.
    for res in [(1920, 1080), (1280, 720), (640, 360), (320, 180)]:
        var mp = res[0] * res[1]
        var off = sota_dirt_offset(res[0], res[1], mp)
        var vrs = SotaVrsLayout(res[0], res[1], mp)
        var lens = sota_lens_layout(res[0], res[1], mp)
        _assert(off != 0, "dirt texture fits")
        _assert(off >= lens[6] + lens[0] * lens[1] * 8, "dirt after the lens buffers (incl. the streak)")
        _assert(off >= vrs.list_end, "dirt after the VRS shade list")
        _assert(off + SOTA_DIRT_WORDS <= vrs.rate_off, "dirt below the VRS rate map")
    _assert(sota_dirt_offset(64, 64, 64 * 64) == 0, "tiny frames fall back to inline dirt")
    var png = List[UInt32](length=len(img), fill=0)
    for i in range(len(img)):
        var g = UInt32(Int(img[i] * 255.0))
        png[i] = UInt32(0xFF000000) | (g << 16) | (g << 8) | g
    _save("build/sota_camera_dirt_texture.png", png, SOTA_DIRT_W, SOTA_DIRT_H)
    print("  [PASS] Suite 2e: procedural lens-dirt texture: deterministic, film + specks + structure, layout")


def test_cpu_dof_reference():
    var w = 48
    var h = 48
    var k = Float32(10.0)
    var s = Float32(2.0)
    # (a) point light at z = 8: disc radius = K (1 - S/z) = 7.5 px; energy conserved.
    var rgb = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 0.0))
    var dep = List[Float32](length=w * h, fill=8.0)
    rgb[24 * w + 24] = Vec3(100.0, 100.0, 100.0)
    var out = eval_dof_cpu(rgb, dep, w, h, k, s, 16.0, 0)
    var e_in = Float32(100.0)
    var e_out = Float32(0.0)
    var area = 0
    var peak = Float32(0.0)
    for i in range(w * h):
        e_out += out[i].x
        peak = max(peak, out[i].x)
    for i in range(w * h):
        if out[i].x > peak * Float32(0.5):
            area += 1
    var r_eq = sqrt(Float32(area) / Float32(3.14159265))
    _near(e_out / e_in, 1.0, 0.02, "CPU DoF conserves the energy of a point light")
    _near(r_eq, 7.5, 0.3, "CPU DoF disc radius = thin-lens CoC")
    _near(peak, Float32(100.0) / (Float32(3.14159265) * 7.5 * 7.5), Float32(0.06), "CPU DoF disc brightness = energy / (pi r^2)")

    # (b) near field: an out-of-focus near half-plane (z = 1) spreads over the in-focus background (z = S).
    var rgb2 = List[Vec3](length=w * h, fill=Vec3(0.2, 0.2, 0.2))
    var dep2 = List[Float32](length=w * h, fill=2.0)
    for y in range(h):
        for x in range(w // 2):
            rgb2[y * w + x] = Vec3(1.0, 0.0, 0.0)
            dep2[y * w + x] = 1.0
    var out2 = eval_dof_cpu(rgb2, dep2, w, h, k, s, 16.0, 0)
    _assert(out2[24 * w + 27].x > Float32(0.35), "near-field red spreads 3 px onto the sharp background")
    _assert(out2[24 * w + 40].x < Float32(0.21), "spread stops beyond the near CoC")

    # (c) far field: blurred background (z = 20) does not bleed onto a sharp foreground (z = S).
    var rgb3 = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 1.0))
    var dep3 = List[Float32](length=w * h, fill=20.0)
    for y in range(h):
        for x in range(w // 2):
            rgb3[y * w + x] = Vec3(1.0, 1.0, 0.0)
            dep3[y * w + x] = 2.0
    var out3 = eval_dof_cpu(rgb3, dep3, w, h, k, s, 16.0, 0)
    _near(out3[24 * w + 22].z, 0.0, 1e-4, "sharp foreground protected from background blur")
    print("  [PASS] Suite 2c: CPU scatter-as-gather DoF: energy, disc radius, near spread, far protection")


# ============================================================================
# 3. GPU
# ============================================================================

struct _Scene(Copyable, Movable):
    var wall: Int
    var strip_hi: Int
    var strip_lo: Int
    var dot: Int
    var red_box: Int
    var grey_wall: Int
    var white_bar: Int
    var black_wall: Int
    var bright_dot: Int

    def __init__(out self, mut ren: FlintAaaVulkanRenderer):
        var m_black = ren.add_material(_emissive(Vec3(0.0, 0.0, 0.0)))
        var m_hi = ren.add_material(_emissive(Vec3(0.6, 0.6, 0.6)))
        var m_lo = ren.add_material(_emissive(Vec3(0.08, 0.08, 0.08)))
        var m_dot = ren.add_material(_emissive(Vec3(0.9, 0.9, 0.9)))
        var m_red = ren.add_material(_emissive(Vec3(0.9, 0.04, 0.04)))
        var m_grey = ren.add_material(_emissive(Vec3(0.18, 0.18, 0.18)))
        var m_white = ren.add_material(_emissive(Vec3(0.95, 0.95, 0.95)))
        var m_bright = ren.add_material(_emissive(Vec3(400.0, 400.0, 400.0)))
        self.wall = ren.add_mesh(_box(Vec3(-20.0, -20.0, 0.0), Vec3(20.0, 20.0, 0.05)), m_black).mesh_id
        self.strip_hi = ren.add_mesh(_box(Vec3(0.0, -20.0, -0.01), Vec3(0.1, 20.0, 0.0)), m_hi).mesh_id
        self.strip_lo = ren.add_mesh(_box(Vec3(0.0, -20.0, -0.01), Vec3(0.1, 20.0, 0.0)), m_lo).mesh_id
        self.dot = ren.add_mesh(_sphere(0.05), m_dot).mesh_id
        self.red_box = ren.add_mesh(_box(Vec3(-3.0, -3.0, 0.0), Vec3(0.0, 3.0, 0.1)), m_red).mesh_id
        self.grey_wall = ren.add_mesh(_box(Vec3(-20.0, -20.0, 0.0), Vec3(20.0, 20.0, 0.05)), m_grey).mesh_id
        self.white_bar = ren.add_mesh(_box(Vec3(-0.04, -3.0, -0.01), Vec3(0.04, 3.0, 0.0)), m_white).mesh_id
        self.black_wall = self.wall
        self.bright_dot = ren.add_mesh(_sphere(0.04), m_bright).mesh_id


def _setup(mut ren: FlintAaaVulkanRenderer):
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)  # no jitter: frames are directly comparable
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_vsm = False
    ren.enable_motion_blur = False
    ren.enable_bokeh_dof = False
    var sky = SkyDomeParams()
    sky.zenith_rgb = Vec3(0.0, 0.0, 0.0)
    sky.horizon_rgb = Vec3(0.0, 0.0, 0.0)
    sky.ground_rgb = Vec3(0.0, 0.0, 0.0)
    sky.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky, base_dim=16, mip_levels=4))
    ren.ibl_intensity = Float32(0.0)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)


def _frame(mut ren: FlintAaaVulkanRenderer, cam: PhysicalCameraModel, sc: _Scene, kind: Int) -> List[UInt32]:
    """kind 0: striped wall at 4 m; 1: dot at 8 m on a black wall at 12 m; 2: red near box at 1 m over
    the striped wall at 4 m; 3: grey wall at 4 m; 4: white bar near the right edge on black at 4 m;
    5: very bright dot off-axis on black at 12 m."""
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, 0.0), cam_target=Vec3(0.0, 0.0, 1.0), fov_y_deg=cam.fov_y_deg(), near_z=0.05)
    if kind == 0 or kind == 2:
        ren.draw_instance(sc.wall, trs(Vec3(0.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        for i in range(-12, 12):
            var mid = sc.strip_hi if (i & 1) == 0 else sc.strip_lo
            ren.draw_instance(mid, trs(Vec3(Float32(i) * Float32(0.1), 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        if kind == 2:
            ren.draw_instance(sc.red_box, trs(Vec3(0.0, 0.0, 1.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    elif kind == 1:
        ren.draw_instance(sc.wall, trs(Vec3(0.0, 0.0, 12.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        ren.draw_instance(sc.dot, trs(Vec3(0.0, 0.0, 8.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    elif kind == 3:
        ren.draw_instance(sc.grey_wall, trs(Vec3(0.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    elif kind == 4:
        ren.draw_instance(sc.wall, trs(Vec3(0.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        ren.draw_instance(sc.white_bar, trs(Vec3(0.55, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    else:
        ren.draw_instance(sc.wall, trs(Vec3(0.0, 0.0, 12.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        ren.draw_instance(sc.bright_dot, trs(Vec3(0.9, 0.45, 12.0 - 0.1), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "rendered frame kind " + String(kind))
    return ren.readback_pixels()


def _circle_overlap(a: Float32, b: Float32, d: Float32) -> Float32:
    """Area of the intersection of circles of radii `a` and `b` whose centres are `d` apart."""
    if d >= a + b:
        return 0.0
    if d <= abs(a - b):
        var m = min(a, b)
        return Float32(3.14159265) * m * m
    var ca = (d * d + a * a - b * b) / (Float32(2.0) * d * a)
    var cb = (d * d + b * b - a * a) / (Float32(2.0) * d * b)
    var k = (-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b)
    return a * a * acos(ca) + b * b * acos(cb) - Float32(0.5) * sqrt(max(k, Float32(0.0)))


def _disc_conv_half_max_radius(a: Float32, b: Float32) -> Float32:
    """Radius at which disc(a) (*) disc(b) falls to half its peak (peak = area of the smaller disc)."""
    var peak = _circle_overlap(a, b, 0.0)
    var d = Float32(0.0)
    while d < a + b and _circle_overlap(a, b, d) > peak * Float32(0.5):
        d += Float32(0.005)
    return d


def _save(path: String, mut px: List[UInt32], w: Int, h: Int):
    _ = write_png(path, px, w, h)


def test_gpu_dof():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _assert(ren.sota_cam.pipelines_ready(), "sota:camera pipelines created")
    _setup(ren)
    var sc = _Scene(ren)
    var cam = PhysicalCameraModel(focal_length_mm=85.0, fstop=1.4, focus_distance_m=4.0, aperture_blades=0)
    ren.sota_cam.camera = cam
    ren.sota_cam.dof_max_radius_px = 32.0

    # (a) In focus: DoF on == DoF off, bit for bit (the composite takes the sharp pixel when CoC < 0.5 px).
    ren.sota_cam.enable_dof = False
    var sharp = _frame(ren, cam, sc, 0)
    ren.sota_cam.enable_dof = True
    var dof_in = _frame(ren, cam, sc, 0)
    var max_d = 0
    for i in range(w * h):
        for sh in [0, 8, 16]:
            max_d = max(max_d, abs(Int((sharp[i] >> UInt32(sh)) & 0xFF) - Int((dof_in[i] >> UInt32(sh)) & 0xFF)))
    _assert(max_d <= 1, "in-focus plane unchanged by DoF (max 8-bit diff " + String(max_d) + ")")

    # (b) Defocused stripes lose contrast: focus at 0.6 m, stripes (32 px period) at 4 m -> CoC radius 32 px.
    var cam_b = cam
    cam_b.focus_distance_m = 0.6
    ren.sota_cam.camera = cam_b
    var blur = _frame(ren, cam_b, sc, 0)
    var hf_sharp = Float32(0.0)
    var hf_blur = Float32(0.0)
    for x in range(40, w - 40):
        hf_sharp += abs(_luma(_lin(sharp[90 * w + x])) - _luma(_lin(sharp[90 * w + x + 1])))
        hf_blur += abs(_luma(_lin(blur[90 * w + x])) - _luma(_lin(blur[90 * w + x + 1])))
    print("    stripes edge energy sharp=" + String(hf_sharp) + " defocused=" + String(hf_blur))
    _assert(hf_blur < hf_sharp * Float32(0.25), "defocused stripes lose >75% of their edge energy")

    # (c) Bokeh disc of a dot at 8 m with focus at 2 m: radius = K (1 - S / z), energy conserved,
    # and the GPU image matches the CPU reference DoF run on the GPU's own sharp frame.
    var cam_c = cam
    cam_c.focus_distance_m = 2.0
    ren.sota_cam.camera = cam_c
    ren.sota_cam.enable_dof = False
    var dot_sharp = _frame(ren, cam_c, sc, 1)
    ren.sota_cam.enable_dof = True
    var dot_dof = _frame(ren, cam_c, sc, 1)
    var k = cam_c.coc_infinity_radius_px(h)
    var want_r = eval_coc_px_cpu(k, 2.0, 8.0 - 0.05, 32.0)
    var e_sharp = Float32(0.0)
    var e_dof = Float32(0.0)
    var peak = Float32(0.0)
    for i in range(w * h):
        e_sharp += _luma(_lin(dot_sharp[i]))
        var l = _luma(_lin(dot_dof[i]))
        e_dof += l
        peak = max(peak, l)
    var area = 0
    for i in range(w * h):
        if _luma(_lin(dot_dof[i])) > peak * Float32(0.5):
            area += 1
    var r_eq = sqrt(Float32(area) / Float32(3.14159265))
    # The sharp dot is itself a disc (radius from its energy / peak); its bokeh is that disc convolved with
    # the thin-lens CoC disc, whose half-maximum radius is solved analytically from the circle overlap.
    var peak_sharp = Float32(0.0)
    for i in range(w * h):
        peak_sharp = max(peak_sharp, _luma(_lin(dot_sharp[i])))
    var dot_r = sqrt(e_sharp / peak_sharp / Float32(3.14159265))
    var want_half = _disc_conv_half_max_radius(dot_r, want_r)
    print("    bokeh: K=" + String(k) + " px, thin-lens r=" + String(want_r) + " px, dot r=" + String(dot_r) + ", analytic half-max r=" + String(want_half) + ", GPU r_eq=" + String(r_eq) + ", energy ratio=" + String(e_dof / e_sharp))
    _near(r_eq, want_half, 0.45, "GPU bokeh disc radius = thin-lens CoC disc (*) sharp dot, at half maximum")
    _near(e_dof / e_sharp, 1.0, 0.04, "GPU bokeh conserves energy (+-4 %)")

    # Image diff vs the CPU reference on a 64x64 crop around the dot.
    var cw = 64
    var x0 = w // 2 - cw // 2
    var y0 = h // 2 - cw // 2
    var crop = List[Vec3](length=cw * cw, fill=Vec3(0.0, 0.0, 0.0))
    var dep = List[Float32](length=cw * cw, fill=12.0)
    for y in range(cw):
        for x in range(cw):
            var c = _lin(dot_sharp[(y0 + y) * w + x0 + x])
            crop[y * cw + x] = c
            if _luma(c) > Float32(0.02):
                dep[y * cw + x] = 7.95
    var cpu_ref = eval_dof_cpu(crop, dep, cw, cw, k, 2.0, 32.0, 0)
    var err = Float32(0.0)
    var norm = Float32(0.0)
    for y in range(cw):
        for x in range(cw):
            var g = _luma(_lin(dot_dof[(y0 + y) * w + x0 + x]))
            var c = _luma(cpu_ref[y * cw + x])
            err += abs(g - c)
            norm += c
    var rel = err / max(norm, Float32(1e-6))
    var cpu_peak = Float32(0.0)
    var cpu_e = Float32(0.0)
    for i in range(cw * cw):
        cpu_peak = max(cpu_peak, _luma(cpu_ref[i]))
        cpu_e += _luma(cpu_ref[i])
    var cpu_area = 0
    for i in range(cw * cw):
        if _luma(cpu_ref[i]) > cpu_peak * Float32(0.5):
            cpu_area += 1
    var cpu_r_eq = sqrt(Float32(cpu_area) / Float32(3.14159265))
    print("    GPU vs CPU-reference DoF relative L1 = " + String(rel) + ", CPU r_eq=" + String(cpu_r_eq) + " peak GPU/CPU=" + String(peak / cpu_peak))
    _assert(rel < Float32(0.2), "GPU DoF matches the CPU reference (rel L1 " + String(rel) + ")")
    _near(cpu_r_eq, want_half, 0.45, "CPU reference disc radius at half maximum")
    _near(r_eq, cpu_r_eq, 0.4, "GPU and CPU bokeh disc radius agree")
    _near(peak / cpu_peak, 1.0, 0.06, "GPU and CPU bokeh disc brightness agree (+-6 %)")
    _near(cpu_e / e_sharp, 1.0, 0.03, "CPU reference conserves energy")

    # (c2) Large CoC (focus 1 m -> r = 18.5 px, taps from the prefiltered quarter-res level): still a flat,
    # energy-conserving disc of the thin-lens size (no hot centre from a double-counted centre tap).
    var cam_c2 = cam
    cam_c2.focus_distance_m = 1.0
    ren.sota_cam.camera = cam_c2
    var big = _frame(ren, cam_c2, sc, 1)
    var want_r2 = eval_coc_px_cpu(cam_c2.coc_infinity_radius_px(h), 1.0, 8.0 - 0.05, 32.0)
    var want_half2 = _disc_conv_half_max_radius(dot_r, want_r2)
    var e_big = Float32(0.0)
    var peak_big = Float32(0.0)
    var ring_sum = Float32(0.0)
    var ring_n = 0
    for y in range(h):
        for x in range(w):
            var l = _luma(_lin(big[y * w + x]))
            e_big += l
            peak_big = max(peak_big, l)
            var dx = Float32(x - w // 2) + 0.5
            var dy = Float32(y - h // 2) + 0.5
            var rr = sqrt(dx * dx + dy * dy)
            if rr > want_r2 * Float32(0.3) and rr < want_r2 * Float32(0.6):
                ring_sum += l
                ring_n += 1
    var centre = _luma(_lin(big[(h // 2) * w + w // 2]))
    var area_big = 0
    for i in range(w * h):
        if _luma(_lin(big[i])) > peak_big * Float32(0.5):
            area_big += 1
    var r_eq2 = sqrt(Float32(area_big) / Float32(3.14159265))
    var flat = centre / (ring_sum / Float32(max(ring_n, 1)))
    print("    large bokeh: thin-lens r=" + String(want_r2) + " analytic half-max r=" + String(want_half2) + " GPU r_eq=" + String(r_eq2) + " energy=" + String(e_big / e_sharp) + " centre/mid-disc=" + String(flat))
    _near(r_eq2, want_half2, 0.8, "large GPU bokeh disc radius at half maximum")
    _near(e_big / e_sharp, 1.0, 0.04, "large GPU bokeh conserves energy (+-4 %)")
    _near(flat, 1.0, 0.08, "large GPU bokeh disc is flat (no hot centre)")

    # (d) Near field: red box at 1 m (focus 4 m) spreads over the sharp stripes beside it.
    ren.sota_cam.camera = cam
    ren.sota_cam.enable_dof = False
    var near_sharp = _frame(ren, cam, sc, 2)
    ren.sota_cam.enable_dof = True
    var near_dof = _frame(ren, cam, sc, 2)
    var edge = -1
    for x in range(w):
        if _lin(near_sharp[90 * w + x]).x > Float32(0.5) and _lin(near_sharp[90 * w + x]).y < Float32(0.1):
            edge = x
    _assert(edge > 10 and edge < w - 20, "found the near box silhouette (x=" + String(edge) + ")")
    var spread = _lin(near_dof[90 * w + edge + 4]).x - _lin(near_sharp[90 * w + edge + 4]).x
    var ncrop = List[Vec3](length=64 * 64, fill=Vec3(0.0, 0.0, 0.0))
    var ndep = List[Float32](length=64 * 64, fill=4.0)
    var nx0 = edge - 32
    for y in range(64):
        for x in range(64):
            var c = _lin(near_sharp[(58 + y) * w + nx0 + x])
            ncrop[y * 64 + x] = c
            if c.x > Float32(0.5) and c.y < Float32(0.1):
                ndep[y * 64 + x] = 1.0
    var nref = eval_dof_cpu(ncrop, ndep, 64, 64, cam.coc_infinity_radius_px(h), 4.0, 32.0, 0)
    var max_dev = Float32(0.0)
    for o in range(1, 16):
        max_dev = max(max_dev, abs(_lin(near_dof[90 * w + edge + o]).x - nref[32 * 64 + 32 + o].x))
    print("    near-field edge profile: max |GPU - CPU ref| over 15 px = " + String(max_dev))
    _assert(max_dev < Float32(0.05), "GPU near-field spread profile matches the CPU reference")
    _assert(spread > Float32(0.08), "near-field blur spreads over the in-focus background (+" + String(spread) + ")")

    # (e) Focus on the near box: the sharp box keeps its pixels while the stripes behind blur.
    var cam_e = cam
    cam_e.focus_distance_m = 1.0
    ren.sota_cam.camera = cam_e
    var fg = _frame(ren, cam_e, sc, 2)
    var inside = _lin(fg[90 * w + edge - 3])
    _near(inside.x, _lin(near_sharp[90 * w + edge - 3]).x, 0.03, "in-focus foreground protected from background blur")

    var out_dir = String("build/")
    _save(out_dir + "sota_camera_dof_dot_sharp.png", dot_sharp, w, h)
    _save(out_dir + "sota_camera_dof_dot_bokeh.png", dot_dof, w, h)
    _save(out_dir + "sota_camera_dof_near_field.png", near_dof, w, h)
    ren.shutdown()
    print("  [PASS] Suite 3a: GPU DoF in-focus identity, defocus, bokeh radius/energy, CPU-ref diff, near/far fields")


def test_gpu_lens_and_exposure():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    var sc = _Scene(ren)
    var cam = PhysicalCameraModel(focal_length_mm=50.0, fstop=8.0, focus_distance_m=4.0)
    cam.max_aperture_fstop = 1.4
    ren.sota_cam.camera = cam
    ren.sota_cam.drive_renderer = True

    # Physical exposure: f/5.6 lets in 64/31.36 = 2.04x the light of f/8 (grey 0.18 wall, linear readback).
    var g8 = _lin(_frame(ren, cam, sc, 3)[90 * w + 160]).y
    var cam56 = cam
    cam56.fstop = 5.6
    ren.sota_cam.camera = cam56
    var g56 = _lin(_frame(ren, cam56, sc, 3)[90 * w + 160]).y
    _near(g8, 0.18, 0.01, "unity EV100 calibration keeps the 0.18 wall at 0.18")
    _near(g56 / g8, Float32(64.0 / 31.36), 0.12, "EV100 exposure ratio f/5.6 vs f/8")
    _near(ren.exposure, cam56.exposure_scale(), 1e-5, "renderer exposure driven by the camera")
    _near(ren.motion_blur_shutter, cam56.shutter_fraction(Float32(1.0 / 60.0)), 1e-5, "motion-blur shutter driven by the camera")
    ren.sota_cam.camera = cam

    # Lateral CA on a white bar right of centre: R and B separate at the bar edges, centre stays neutral.
    var bar = _frame(ren, cam, sc, 4)
    ren.sota_cam.enable_chromatic_aberration = True
    ren.sota_cam.ca_strength = 0.01
    var bar_ca = _frame(ren, cam, sc, 4)
    var max_split = Float32(0.0)
    var max_split_ref = Float32(0.0)
    for x in range(160, w):
        var c = _lin(bar_ca[90 * w + x])
        var r = _lin(bar[90 * w + x])
        max_split = max(max_split, abs(c.x - c.z))
        max_split_ref = max(max_split_ref, abs(r.x - r.z))
    _assert(max_split_ref < Float32(0.02), "no colour split without CA")
    _assert(max_split > Float32(0.25), "lateral CA splits R and B at the off-axis bar (" + String(max_split) + ")")

    # Physical order: CA is part of the lens image, so it runs right after DoF and before bloom (lens
    # scatter) and motion blur (shutter integration) see the image.
    ren.sota_cam.enable_chromatic_aberration = True
    ren.sota_cam.enable_dof = True
    ren.bloom_intensity = 0.05
    ren.enable_motion_blur = True
    _ = _frame(ren, cam, sc, 4)
    var names = List[String]()
    for t in ren.get_gpu_pass_timings():
        names.append(t.pass_name)
    var i_dof = -1
    var i_ca = -1
    var i_bloom = -1
    var i_mb = -1
    for i in range(len(names)):
        if names[i] == "SotaBokehDoF":
            i_dof = i
        elif names[i] == "SotaChromaticAberration":
            i_ca = i
        elif names[i] == "DualKawaseBloom5Mip":
            i_bloom = i
        elif names[i] == "TileMaxMotionBlur":
            i_mb = i
    print("    pass order: DoF " + String(i_dof) + ", CA " + String(i_ca) + ", bloom " + String(i_bloom) + ", motion blur " + String(i_mb))
    _assert(i_dof >= 0 and i_ca > i_dof and i_bloom > i_ca and i_mb > i_ca, "CA runs after DoF and before bloom and motion blur")
    ren.sota_cam.enable_chromatic_aberration = False
    ren.sota_cam.enable_dof = False
    ren.bloom_intensity = 0.0
    ren.enable_motion_blur = False

    # Optical vignetting: strong wide open, none 2 stops down.
    ren.sota_cam.enable_optical_vignette = True
    var cam14 = cam
    cam14.fstop = 1.4
    cam14.exposure_compensation_ev = cam14.ev100() - cam14.ev100_unity  # keep exposure at 1
    ren.sota_cam.camera = cam14
    var v14 = _frame(ren, cam14, sc, 3)
    var corner14 = _lin(v14[4 * w + 4]).y / _lin(v14[90 * w + 160]).y
    var cam28 = cam14
    cam28.fstop = 2.8
    cam28.exposure_compensation_ev = cam28.ev100() - cam28.ev100_unity
    ren.sota_cam.camera = cam28
    var v28 = _frame(ren, cam28, sc, 3)
    var corner28 = _lin(v28[4 * w + 4]).y / _lin(v28[90 * w + 160]).y
    var aspect = Float32(w) / Float32(h)
    var want14 = optical_vignette_cpu(Float32(-1.0) + Float32(9.0) / Float32(w), Float32(-1.0) + Float32(9.0) / Float32(h), aspect, 1.0)
    _near(corner14, want14, 0.06, "optical vignette at f/1.4 matches the CPU model")
    _near(corner28, 1.0, 0.03, "no optical vignette at f/2.8")
    ren.sota_cam.enable_optical_vignette = False

    # Sensor noise: grey wall at ISO 6400 (exposure held at 1) vs the photon-noise model.
    ren.sota_cam.enable_grain = True
    var cam_hi = cam
    cam_hi.iso = 6400.0
    cam_hi.exposure_compensation_ev = cam_hi.ev100() - cam_hi.ev100_unity
    ren.sota_cam.camera = cam_hi
    var noisy = _frame(ren, cam_hi, sc, 3)
    var cam_lo = cam
    ren.sota_cam.camera = cam_lo
    var clean = _frame(ren, cam_lo, sc, 3)
    ren.sota_cam.enable_grain = False
    var sum_hi = Float32(0.0)
    var sq_hi = Float32(0.0)
    var sum_lo = Float32(0.0)
    var sq_lo = Float32(0.0)
    var n = 0
    for y in range(40, 140):
        for x in range(80, 240):
            var a = _lin(noisy[y * w + x]).y
            var b = _lin(clean[y * w + x]).y
            sum_hi += a
            sq_hi += a * a
            sum_lo += b
            sq_lo += b * b
            n += 1
    var mean_hi = sum_hi / Float32(n)
    var sd_hi = sqrt(max(Float32(0.0), sq_hi / Float32(n) - mean_hi * mean_hi))
    var mean_lo = sum_lo / Float32(n)
    var sd_lo = sqrt(max(Float32(0.0), sq_lo / Float32(n) - mean_lo * mean_lo))
    var want_sd = cam_hi.sensor_noise_sigma(0.18)
    print("    noise: ISO6400 sd=" + String(sd_hi) + " (model " + String(want_sd) + "), ISO100 sd=" + String(sd_lo) + ", mean " + String(mean_hi))
    _near(mean_hi, 0.18, 0.01, "noise is zero-mean")
    _near(sd_hi, want_sd, want_sd * Float32(0.3), "ISO 6400 noise matches the shot+read noise model")
    _assert(sd_hi > sd_lo * Float32(3.0), "noise grows with ISO")
    _save("build/sota_camera_grain_iso6400.png", noisy, w, h)

    # Lens ghosts: a very bright off-axis source puts light at the point mirrored through the centre.
    ren.sota_cam.camera = cam
    var src_only = _frame(ren, cam, sc, 5)
    ren.sota_cam.enable_flare = True
    ren.sota_cam.flare_intensity = 200.0
    var flare = _frame(ren, cam, sc, 5)
    var sx = 0
    var sy = 0
    var best = Float32(0.0)
    for i in range(w * h):
        var l = _luma(_lin(src_only[i]))
        if l > best:
            best = l
            sx = i % w
            sy = i // w
    var mx = w - 1 - sx
    var my = h - 1 - sy
    var ghost = Float32(0.0)
    var ghost_ref = Float32(0.0)
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            var xx = max(0, min(w - 1, mx + dx))
            var yy = max(0, min(h - 1, my + dy))
            ghost = max(ghost, _luma(_lin(flare[yy * w + xx])))
            ghost_ref = max(ghost_ref, _luma(_lin(src_only[yy * w + xx])))
    print("    ghost at mirrored point: " + String(ghost) + " (without flare " + String(ghost_ref) + ")")
    _assert(ghost_ref < Float32(0.004), "no light at the mirrored point without flare")
    _assert(ghost > Float32(0.01), "lens ghost appears at the mirrored point")
    ren.sota_cam.enable_dirt = True
    var dirty = _frame(ren, cam, sc, 5)
    var e_flare = Float32(0.0)
    var e_dirty = Float32(0.0)
    for i in range(w * h):
        e_flare += _luma(_lin(flare[i]))
        e_dirty += _luma(_lin(dirty[i]))
    _assert(e_dirty > e_flare, "lens dirt adds scattered light")
    _assert(ren.sota_cam.dirt_uploaded_off != 0, "procedural dirt texture uploaded")
    ren.sota_cam.dirt_use_texture = False
    var dirty_inline = _frame(ren, cam, sc, 5)
    var n_differ = 0
    for i in range(w * h):
        if dirty_inline[i] != dirty[i]:
            n_differ += 1
    _assert(n_differ > w * h // 50, "the dirt texture (not the inline fallback) shapes the scattered light")
    ren.sota_cam.dirt_use_texture = True
    _save("build/sota_camera_flare_dirt.png", dirty, w, h)
    # Anamorphic streak: the bright source spreads along the horizontal, not the vertical.
    ren.sota_cam.enable_dirt = False
    ren.sota_cam.enable_flare = False
    ren.sota_cam.enable_anamorphic_streak = True
    ren.sota_cam.anamorphic_streak_intensity = 0.5
    var streak = _frame(ren, cam, sc, 5)
    ren.sota_cam.enable_anamorphic_streak = False
    var off_h = Float32(0.0)
    var off_v = Float32(0.0)
    var off_h0 = Float32(0.0)
    for d in [-60, -40, 40, 60]:
        var xh = max(0, min(w - 1, sx + d))
        var yv = max(0, min(h - 1, sy + d // 2))
        off_h += _luma(_lin(streak[sy * w + xh]))
        off_h0 += _luma(_lin(src_only[sy * w + xh]))
        off_v += _luma(_lin(streak[yv * w + sx]))
    _save("build/sota_camera_anamorphic_streak.png", streak, w, h)
    print("    anamorphic streak: horizontal " + String(off_h) + " (without " + String(off_h0) + "), vertical " + String(off_v))
    _assert(off_h > off_h0 + Float32(0.01), "anamorphic streak spreads light horizontally")
    _assert(off_h > off_v * Float32(4.0), "anamorphic streak is horizontal")
    ren.shutdown()
    print("  [PASS] Suite 3b: GPU EV100 exposure, lateral CA, optical vignette, ISO noise model, ghosts, dirt texture, anamorphic streak")


struct _Paired(Copyable, Movable):
    """Paired timing samples: the measured pass and the contention probe (legacy 10-dispatch bloom,
    ~0.5-0.7 ms at 720p on an idle GB10) recorded in the same frame."""
    var pass_min: Float64   # best absolute pass time
    var ratio_min: Float64  # best pass / probe ratio of one frame
    var probe_min: Float64

    def __init__(out self):
        self.pass_min = 1.0e9
        self.ratio_min = 1.0e9
        self.probe_min = 1.0e9


def _paired_sample(mut ren: FlintAaaVulkanRenderer, name: String, frames: Int, cam: PhysicalCameraModel, sc: _Scene) -> _Paired:
    var r = _Paired()
    for _ in range(frames):
        _ = _frame(ren, cam, sc, 2)
        var p = Float64(-1.0)
        var q = Float64(-1.0)
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == name:
                p = t.gpu_ms
            elif t.pass_name == "DualKawaseBloom5Mip":
                q = t.gpu_ms
        if p < 0.0 or q <= 0.0:
            continue
        r.pass_min = min(r.pass_min, p)
        r.probe_min = min(r.probe_min, q)
        r.ratio_min = min(r.ratio_min, p / q)
    return r^


def _budget(mut ren: FlintAaaVulkanRenderer, name: String, budget_ms: Float64, cam: PhysicalCameraModel, sc: _Scene) -> Bool:
    """Shared-GPU budget check. Other clients time-slice the GPU in bursts; a burst inflates every
    multi-dispatch pass of the frame it lands in, and a separately measured probe misses it. So each
    sample frame carries the pass and the probe together, and a round passes when either the best absolute
    time is within budget or the best same-frame pass/probe ratio is within `budget / 0.5 ms` (the idle
    probe cost). Up to 4 rounds of 16 paired frames; only a miss that persists through all of them with
    a quiet probe fails. A persistent miss with a busy probe is reported as a contention skip."""
    var ratio_budget = budget_ms / 0.5
    var best = _Paired()
    for round in range(4):
        var r = _paired_sample(ren, name, 16, cam, sc)
        best.pass_min = min(best.pass_min, r.pass_min)
        best.ratio_min = min(best.ratio_min, r.ratio_min)
        best.probe_min = min(best.probe_min, r.probe_min)
        if r.pass_min < budget_ms or r.ratio_min < ratio_budget:
            print("    " + name + ": min " + String(best.pass_min) + " ms (budget " + String(budget_ms) + "), min pass/probe "
                + String(best.ratio_min) + " (budget " + String(ratio_budget) + "), probe min " + String(best.probe_min) + " ms, round " + String(round + 1))
            return True
    print("    " + name + ": min " + String(best.pass_min) + " ms, min pass/probe " + String(best.ratio_min) + ", probe min " + String(best.probe_min) + " ms over 4 rounds")
    if best.probe_min > 1.0:
        print("  [WARN] " + name + " over budget while the GPU was time-sliced (probe " + String(best.probe_min) + " ms): not enforced")
        return True
    return False


def test_gpu_budgets():
    var w = 1280
    var h = 720
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    var sc = _Scene(ren)
    var cam = PhysicalCameraModel(focal_length_mm=85.0, fstop=1.4, focus_distance_m=4.0, aperture_blades=7)
    ren.sota_cam.camera = cam
    ren.bloom_intensity = 0.05  # the contention probe runs in every sample frame
    ren.sota_cam.enable_dof = True
    _assert(_budget(ren, "SotaBokehDoF", 2.0, cam, sc), "DoF within 2.0 ms at 720p")
    ren.sota_cam.enable_dof = False
    ren.sota_cam.enable_flare = True
    ren.sota_cam.enable_dirt = True
    ren.sota_cam.enable_optical_vignette = True
    ren.sota_cam.enable_grain = True
    _assert(_budget(ren, "SotaLensSensor", 1.5, cam, sc), "lens + sensor within 1.5 ms at 720p")
    ren.sota_cam.enable_chromatic_aberration = True
    _assert(_budget(ren, "SotaChromaticAberration", 0.6, cam, sc), "lateral CA within 0.6 ms at 720p")
    ren.shutdown()
    print("  [PASS] Suite 3c: GPU time budgets at 1280x720 (paired with a same-frame contention probe)")


def _read_post_word(ren: FlintAaaVulkanRenderer, word: Int) -> UInt32:
    var out = List[UInt32](length=1, fill=0)
    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), ren.buf_post_scratch.cpu_address + word * 4, 4)
    return out[0]


def _psnr(a: List[UInt32], b: List[UInt32]) -> Float64:
    var se = Float64(0.0)
    for i in range(len(a)):
        for sh in [0, 8, 16]:
            var d = Float64(Int((a[i] >> UInt32(sh)) & 0xFF)) - Float64(Int((b[i] >> UInt32(sh)) & 0xFF))
            se += d * d
    var mse = se / Float64(max(1, len(a) * 3))
    if mse <= 1.0e-12:
        return 200.0
    return 10.0 * log10(255.0 * 255.0 / mse)


def test_gpu_vrs():
    var w = 640
    var h = 360
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    var sc = _Scene(ren)
    var cam = PhysicalCameraModel(focal_length_mm=35.0, fstop=8.0, focus_distance_m=4.0)
    var lay = SotaVrsLayout(ren.render_w, ren.render_h, max(ren.output_w * ren.output_h, ren.render_w * ren.render_h))
    var n_px = ren.render_w * ren.render_h
    for kind in [3, 0]:
        ren.sota_cam.enable_vrs = False
        var full = List[UInt32]()
        for _ in range(3):
            full = _frame(ren, cam, sc, kind)
        ren.sota_cam.enable_vrs = True
        var on = List[UInt32]()
        for _ in range(3):
            on = _frame(ren, cam, sc, kind)
        _assert(ren.sota_cam.vrs_active(ren.render_w, ren.render_h), "VRS pipelines active")
        var shaded = Int(_read_post_word(ren, lay.list_off))
        var rates = List[Int]()
        for t in range(lay.tiles_x * lay.tiles_y):
            rates.append(Int(_read_post_word(ren, lay.rate_off + t)))
        var frac_rate = vrs_shaded_fraction(rates)
        var frac = Float32(shaded) / Float32(n_px)
        var q = _psnr(full, on)
        print("    VRS " + ("flat wall" if kind == 3 else "stripes") + ": shaded " + String(frac) + " of pixels (rate map " + String(frac_rate) + "), PSNR vs full rate " + String(q) + " dB")
        _assert(shaded > 0 and shaded <= n_px, "shade-list count in range")
        _assert(frac >= frac_rate - Float32(0.01), "classify never shades fewer pixels than the rate map allows")
        _assert(q > 45.0, "VRS output matches full-rate shading (PSNR > 45 dB)")
        if kind == 3:
            _assert(frac < 0.15, "flat wall shades mostly at 4x4 (< 15 % of pixels)")
        else:
            _assert(frac > 0.15 and frac < 0.9, "stripes keep full rate on the edges only")
        if kind == 0:
            _save("build/sota_camera_vrs_stripes.png", on, w, h)
    ren.sota_cam.enable_vrs = False
    _assert(not ren.sota_cam.vrs_active(ren.render_w, ren.render_h), "VRS off by setting")
    ren.shutdown()
    print("  [PASS] Suite 3d: compute VRS (rate map, compaction, fill): flat wall coarse, edges full rate, PSNR")


def main():
    print("=== Flint v5 sota:camera (Area D) Physical Camera & Lens Suite ===")
    test_spirv()
    test_cpu_camera_model()
    test_cpu_optics()
    test_cpu_vrs_reference()
    test_cpu_dirt_texture()
    test_cpu_dof_reference()
    test_gpu_dof()
    test_gpu_lens_and_exposure()
    test_gpu_vrs()
    test_gpu_budgets()
    print("=== ALL sota:camera TESTS PASSED ===")
