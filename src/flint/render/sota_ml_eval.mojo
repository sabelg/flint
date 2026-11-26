"""Area C (`sota:ml`) evaluation kit: procedural GI scenes, a renderer-consistent 512-spp reference,
and image-quality metrics (PSNR, SSIM, FLIP) for the learned denoiser (`sota_ml_denoise.mojo`).

Everything here runs through Flint's own renderer, so the same scenes feed the regression tests, the
audit numbers in the notes, and the training-data dumper that lives outside the repo.

Reference: the "512-spp reference" of a frame is the renderer's own unbiased estimator, converged:
`gi_rays_per_pixel` rays per pixel for `512 / gi_rays_per_pixel` frames, no TSR history or jitter,
no spatial denoise, averaged in linear HDR. A denoiser can at best reach this image; the gap between it
and a true path tracer (surface-cache bias) belongs to Area B, not to the denoiser.

Metrics are computed on the display image: linear HDR times exposure, clamped, sRGB-encoded (the
`AAA_DISPLAY_LINEAR_SRGB` transfer). PSNR is over the three 8-bit sRGB channels, SSIM is the Wang et al.
(2004) index on Rec.709 luma with an 11-tap Gaussian window (sigma 1.5), FLIP is `flip.mojo` (LDR-FLIP).
"""
from std.math import exp, log, log10, max, min, sqrt, sin, cos
from std.memory import bitcast
from flint.assets.mesh_builder import MeshBuilder
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import flip_error_map, flip_mean

comptime SOTA_ML_SCENE_CORNELL: Int = 0
comptime SOTA_ML_SCENE_GALLERY: Int = 1
comptime SOTA_ML_SCENE_PILLARS: Int = 2


@fieldwise_init
struct SotaMlScene(Copyable, Movable):
    """Meshes of a procedural scene plus its default camera."""
    var mesh_ids: List[Int]
    var cam_pos: Vec3
    var cam_target: Vec3
    var fov_y_deg: Float32


@fieldwise_init
struct SotaMlQuality(Copyable, ImplicitlyCopyable, Movable):
    var psnr_db: Float64
    var ssim: Float64
    var flip: Float64


struct _Rng(Movable):
    var s: UInt32

    def __init__(out self, seed: UInt32):
        self.s = seed * UInt32(747796405) + UInt32(2891336453)

    def next_f(mut self) -> Float32:
        self.s = self.s * UInt32(747796405) + UInt32(2891336453)
        var w = ((self.s >> ((self.s >> 28) + 4)) ^ self.s) * UInt32(277803737)
        w = (w >> 22) ^ w
        return Float32(w & 0xFFFFFF) / Float32(16777216.0)


def _quad(mut mb: MeshBuilder, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: Vec3):
    var i0 = mb.add_vertex(p0, n, 0.0, 0.0)
    var i1 = mb.add_vertex(p1, n, 1.0, 0.0)
    var i2 = mb.add_vertex(p2, n, 1.0, 1.0)
    var i3 = mb.add_vertex(p3, n, 0.0, 1.0)
    mb.add_quad(i0, i1, i2, i3)


def _grid_quad(mut mb: MeshBuilder, origin: Vec3, du: Vec3, dv: Vec3, n: Vec3, cells: Int):
    """Planar patch `origin + u*du + v*dv` (`u, v` in `[0, 1]`) split into `cells x cells` quads, so the
    per-triangle surface cache resolves shadows and colour bleeding (winding is `du x dv` along `n`)."""
    var inv = Float32(1.0) / Float32(cells)
    for j in range(cells):
        for i in range(cells):
            var a = origin + du * (Float32(i) * inv) + dv * (Float32(j) * inv)
            var b = origin + du * (Float32(i + 1) * inv) + dv * (Float32(j) * inv)
            var c = origin + du * (Float32(i + 1) * inv) + dv * (Float32(j + 1) * inv)
            var d = origin + du * (Float32(i) * inv) + dv * (Float32(j + 1) * inv)
            _quad(mb, a, b, c, d, n)


def _diffuse(base: Vec3, rough: Float32 = 1.0, spec: Float32 = 0.0) -> AaaMaterial:
    var m = AaaMaterial(base_color=base, roughness=rough, metallic=0.0)
    m.specular_factor = spec
    return m^


def _box(mut mb: MeshBuilder, lo: Vec3, hi: Vec3, cells: Int):
    """Axis-aligned box with outward faces subdivided `cells x cells` (no bottom face)."""
    var sx = hi.x - lo.x
    var sy = hi.y - lo.y
    var sz = hi.z - lo.z
    # -Z, +Z, -X, +X, +Y faces (outward)
    _grid_quad(mb, Vec3(hi.x, lo.y, lo.z), Vec3(-sx, 0.0, 0.0), Vec3(0.0, sy, 0.0), Vec3(0.0, 0.0, -1.0), cells)
    _grid_quad(mb, Vec3(lo.x, lo.y, hi.z), Vec3(sx, 0.0, 0.0), Vec3(0.0, sy, 0.0), Vec3(0.0, 0.0, 1.0), cells)
    _grid_quad(mb, Vec3(lo.x, lo.y, lo.z), Vec3(0.0, 0.0, sz), Vec3(0.0, sy, 0.0), Vec3(-1.0, 0.0, 0.0), cells)
    _grid_quad(mb, Vec3(hi.x, lo.y, hi.z), Vec3(0.0, 0.0, -sz), Vec3(0.0, sy, 0.0), Vec3(1.0, 0.0, 0.0), cells)
    _grid_quad(mb, Vec3(lo.x, hi.y, hi.z), Vec3(sx, 0.0, 0.0), Vec3(0.0, 0.0, -sz), Vec3(0.0, 1.0, 0.0), cells)


def _room(
    mut ren: FlintAaaVulkanRenderer,
    mut ids: List[Int],
    half: Float32,
    depth: Float32,
    left: Vec3,
    right: Vec3,
    white: Vec3,
    cells: Int,
):
    """Open-front room `x, y in [-half, half]`, `z in [0, depth]` with coloured side walls."""
    var w = ren.add_material(_diffuse(white))
    var l = ren.add_material(_diffuse(left))
    var r = ren.add_material(_diffuse(right))
    var s = Float32(2.0) * half
    var b_flr = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    _grid_quad(b_flr, Vec3(-half, -half, depth), Vec3(s, 0.0, 0.0), Vec3(0.0, 0.0, -depth), Vec3(0.0, 1.0, 0.0), cells)
    var b_ceil = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    _grid_quad(b_ceil, Vec3(-half, half, 0.0), Vec3(s, 0.0, 0.0), Vec3(0.0, 0.0, depth), Vec3(0.0, -1.0, 0.0), cells)
    var b_back = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    _grid_quad(b_back, Vec3(-half, -half, depth), Vec3(0.0, s, 0.0), Vec3(s, 0.0, 0.0), Vec3(0.0, 0.0, -1.0), cells)
    var b_left = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    _grid_quad(b_left, Vec3(-half, -half, 0.0), Vec3(0.0, s, 0.0), Vec3(0.0, 0.0, depth), Vec3(1.0, 0.0, 0.0), cells)
    var b_right = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    _grid_quad(b_right, Vec3(half, -half, 0.0), Vec3(0.0, 0.0, depth), Vec3(0.0, s, 0.0), Vec3(-1.0, 0.0, 0.0), cells)
    ids.append(ren.add_mesh(b_flr^.finish(), w).mesh_id)
    ids.append(ren.add_mesh(b_ceil^.finish(), w).mesh_id)
    ids.append(ren.add_mesh(b_back^.finish(), w).mesh_id)
    ids.append(ren.add_mesh(b_left^.finish(), l).mesh_id)
    ids.append(ren.add_mesh(b_right^.finish(), r).mesh_id)


def sota_ml_configure_renderer(mut ren: FlintAaaVulkanRenderer, sun_dir: Vec3, sun_rad: Vec3):
    """Shared lighting/feature setup for every Area C scene: RT GI + RT sun shadows, no IBL, linear display."""
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.enable_rt_shadows = True
    ren.enable_rt_reflections = False
    ren.enable_rt_gi = True
    ren.sun_shadow_softness = Float32(0.0)
    ren.ibl_intensity = Float32(0.0)
    ren.exposure = Float32(1.0)
    ren.sun_dir = sun_dir.normalized()
    ren.sun_radiance = sun_rad


def sota_ml_build_scene(mut ren: FlintAaaVulkanRenderer, kind: Int, seed: UInt32) -> SotaMlScene:
    """Builds one procedural scene (`SOTA_ML_SCENE_*`); `seed` varies colours, sun and object layout
    (seed 0 of `SOTA_ML_SCENE_CORNELL` is exactly the `test_flint.mojo` Cornell box)."""
    var rng = _Rng(seed * UInt32(2654435761) + UInt32(kind * 97 + 13))
    var ids = List[Int]()
    if kind == SOTA_ML_SCENE_CORNELL:
        var sd = Vec3(-0.32, 0.42, -0.849)
        var red = Vec3(0.80, 0.08, 0.08)
        var green = Vec3(0.08, 0.78, 0.12)
        if seed != 0:
            sd = Vec3(rng.next_f() * 1.2 - 0.6, 0.25 + rng.next_f() * 0.5, -0.85)
            red = Vec3(0.3 + rng.next_f() * 0.6, 0.05 + rng.next_f() * 0.3, 0.05 + rng.next_f() * 0.5)
            green = Vec3(0.05 + rng.next_f() * 0.4, 0.3 + rng.next_f() * 0.6, 0.05 + rng.next_f() * 0.6)
        sota_ml_configure_renderer(ren, sd, Vec3(3.1, 3.0, 2.9))
        # Room `[-2, 2]^2 x [0, 4]` with `0.25 m` cells (the walls of the V14 Cornell test)
        _room(ren, ids, 2.0, 4.0, red, green, Vec3(0.76, 0.76, 0.76), 16)
        var m_blk = ren.add_material(_diffuse(Vec3(0.76, 0.76, 0.76)))
        var b_blk = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
        _box(b_blk, Vec3(-0.75, -2.0, 1.45), Vec3(0.75, -0.20, 2.95), 6)
        ids.append(ren.add_mesh(b_blk^.finish(), m_blk).mesh_id)
        return SotaMlScene(ids^, Vec3(0.0, 0.0, -3.5), Vec3(0.0, 0.0, 2.0), Float32(50.0))

    if kind == SOTA_ML_SCENE_GALLERY:
        # Wider room, several boxes and spheres, sun through the open front
        var sd = Vec3(rng.next_f() * 1.4 - 0.7, 0.2 + rng.next_f() * 0.6, -0.8)
        var lw = Vec3(0.2 + rng.next_f() * 0.7, 0.2 + rng.next_f() * 0.7, 0.2 + rng.next_f() * 0.7)
        var rw = Vec3(0.2 + rng.next_f() * 0.7, 0.2 + rng.next_f() * 0.7, 0.2 + rng.next_f() * 0.7)
        sota_ml_configure_renderer(ren, sd, Vec3(3.4, 3.2, 3.0))
        _room(ren, ids, 3.0, 6.0, lw, rw, Vec3(0.72, 0.70, 0.66), 20)
        for k in range(5):
            var cx = rng.next_f() * 4.0 - 2.0
            var cz = 1.5 + rng.next_f() * 3.8
            var hx = 0.25 + rng.next_f() * 0.45
            var hy = 0.3 + rng.next_f() * 1.6
            var col = Vec3(0.15 + rng.next_f() * 0.8, 0.15 + rng.next_f() * 0.8, 0.15 + rng.next_f() * 0.8)
            var mid = ren.add_material(_diffuse(col, 0.6 + rng.next_f() * 0.4, Float32(0.3) if (k & 1) == 1 else Float32(0.0)))
            var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
            if (k % 3) == 2:
                mb.add_ellipsoid(Vec3(cx, -3.0 + hx, cz), Vec3(hx, hx, hx), rings=16, segments=24)
            else:
                _box(mb, Vec3(cx - hx, -3.0, cz - hx), Vec3(cx + hx, -3.0 + hy, cz + hx), 4)
            ids.append(ren.add_mesh(mb^.finish(), mid).mesh_id)
        return SotaMlScene(ids^, Vec3(0.0, -0.4, -4.6), Vec3(0.0, -1.0, 3.0), Float32(55.0))

    # SOTA_ML_SCENE_PILLARS: colonnade of thin pillars casting fine shadow stripes (hard for spatial filters)
    var sd2 = Vec3(rng.next_f() * 1.6 - 0.8, 0.3 + rng.next_f() * 0.4, -0.7)
    sota_ml_configure_renderer(ren, sd2, Vec3(3.0, 3.0, 3.0))
    _room(ren, ids, 2.5, 5.0, Vec3(0.7, 0.5, 0.3), Vec3(0.3, 0.45, 0.7), Vec3(0.75, 0.75, 0.75), 20)
    var m_p = ren.add_material(_diffuse(Vec3(0.8, 0.78, 0.74)))
    var b_p = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var n_p = 5 + Int(rng.next_f() * 4.0)
    for k in range(n_p):
        var px = Float32(-2.0) + Float32(k) * (Float32(4.0) / Float32(n_p - 1))
        var pz = 1.2 + rng.next_f() * 2.5
        _box(b_p, Vec3(px - 0.12, -2.5, pz - 0.12), Vec3(px + 0.12, 2.5, pz + 0.12), 3)
    ids.append(ren.add_mesh(b_p^.finish(), m_p).mesh_id)
    return SotaMlScene(ids^, Vec3(0.3, 0.0, -4.0), Vec3(0.0, -0.6, 2.5), Float32(55.0))


def sota_ml_camera_at(scene: SotaMlScene, frame: Int, motion: Float32) -> Tuple[Vec3, Vec3]:
    """Camera of frame `frame` on a slow sideways orbit (`motion` metres of sway; 0 is static)."""
    var t = Float32(frame) * Float32(0.05)
    var off = Vec3(sin(t) * motion, sin(t * Float32(0.7)) * motion * Float32(0.3), Float32(0.0))
    return (scene.cam_pos + off, scene.cam_target)


def sota_ml_draw_scene(mut ren: FlintAaaVulkanRenderer, scene: SotaMlScene, frame: Int, motion: Float32):
    var cam = sota_ml_camera_at(scene, frame, motion)
    ren.begin_frame(cam[0], cam[1], fov_y_deg=scene.fov_y_deg)
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    for i in range(len(scene.mesh_ids)):
        ren.draw_instance(scene.mesh_ids[i], ident)


def sota_ml_render_reference(
    mut ren: FlintAaaVulkanRenderer,
    scene: SotaMlScene,
    frame: Int,
    motion: Float32,
    spp: Int = 512,
    rays_per_frame: Int = 32,
    antialiased: Bool = True,
) raises -> List[Vec3]:
    """Renderer-consistent `spp` reference at the camera of `frame`: `rays_per_frame` GI rays per pixel,
    `spp / rays_per_frame` decorrelated frames (the shading seed follows `frame_index`), no spatial
    filter, averaged in linear HDR. `antialiased` keeps the renderer's 16-phase Halton sub-pixel jitter
    (TSR history weight 0.01, i.e. effectively none) so the average is the pixel-area integral that a
    TSR frame converges to; otherwise pixel centres only, as the denoiser sees them before TSR.
    Restores the renderer's settings afterwards."""
    var keep_rays = ren.gi_rays_per_pixel
    var keep_svgf = ren.svgf_passes
    var keep_tsr = ren.tsr_history_weight
    var keep_reblur = ren.enable_spec_reblur
    ren.gi_rays_per_pixel = rays_per_frame
    ren.svgf_passes = 0
    ren.tsr_history_weight = Float32(0.011) if antialiased else Float32(0.0)
    ren.enable_spec_reblur = False
    var n = ren.output_w * ren.output_h
    var acc = List[Vec3](length=n, fill=Vec3(0.0, 0.0, 0.0))
    var n_frames = max(1, spp // rays_per_frame)
    for _ in range(n_frames):
        sota_ml_draw_scene(ren, scene, frame, motion)
        if not ren.render_frame():
            raise Error("sota_ml_render_reference: render_frame failed")
        var hdr = ren.readback_resolved_hdr()
        for i in range(n):
            acc[i] = acc[i] + hdr[i]
    var inv = Float32(1.0) / Float32(n_frames)
    for i in range(n):
        acc[i] = acc[i] * inv
    ren.gi_rays_per_pixel = keep_rays
    ren.svgf_passes = keep_svgf
    ren.tsr_history_weight = keep_tsr
    ren.enable_spec_reblur = keep_reblur
    return acc^


def sota_ml_srgb_encode(c: Float32) -> Float32:
    var cl = max(Float32(0.0), min(Float32(1.0), c))
    if cl <= Float32(0.0031308):
        return cl * Float32(12.92)
    return Float32(1.055) * exp(log(cl) / Float32(2.4)) - Float32(0.055)


def sota_ml_tonemap(hdr: List[Vec3], exposure: Float32 = 1.0) -> List[UInt32]:
    """Linear HDR -> `0xFFRRGGBB` sRGB (clamp, the `AAA_DISPLAY_LINEAR_SRGB` transfer)."""
    var out = List[UInt32](length=len(hdr), fill=0xFF000000)
    for i in range(len(hdr)):
        var c = hdr[i] * exposure
        var r = UInt32(max(0, min(255, Int(sota_ml_srgb_encode(c.x) * Float32(255.0) + Float32(0.5)))))
        var g = UInt32(max(0, min(255, Int(sota_ml_srgb_encode(c.y) * Float32(255.0) + Float32(0.5)))))
        var b = UInt32(max(0, min(255, Int(sota_ml_srgb_encode(c.z) * Float32(255.0) + Float32(0.5)))))
        out[i] = UInt32(0xFF000000) | (r << 16) | (g << 8) | b
    return out^


def sota_ml_psnr(ref_px: List[UInt32], test_px: List[UInt32]) -> Float64:
    var se = Float64(0.0)
    var n = min(len(ref_px), len(test_px))
    for i in range(n):
        for sh in [16, 8, 0]:
            var d = Float64(Int((ref_px[i] >> UInt32(sh)) & 0xFF) - Int((test_px[i] >> UInt32(sh)) & 0xFF))
            se += d * d
    var mse = se / Float64(max(1, n * 3))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * log10(255.0 * 255.0 / mse)


def _luma_plane(px: List[UInt32]) -> List[Float32]:
    var out = List[Float32](length=len(px), fill=0.0)
    for i in range(len(px)):
        var r = Float32(Int((px[i] >> 16) & 0xFF))
        var g = Float32(Int((px[i] >> 8) & 0xFF))
        var b = Float32(Int(px[i] & 0xFF))
        out[i] = Float32(0.2126) * r + Float32(0.7152) * g + Float32(0.0722) * b
    return out^


def _gauss_blur(src: List[Float32], w: Int, h: Int, k: List[Float32]) -> List[Float32]:
    var r = len(k) // 2
    var tmp = List[Float32](length=w * h, fill=0.0)
    for y in range(h):
        for x in range(w):
            var s = Float32(0.0)
            for t in range(-r, r + 1):
                var xx = max(0, min(w - 1, x + t))
                s += k[t + r] * src[y * w + xx]
            tmp[y * w + x] = s
    var out = List[Float32](length=w * h, fill=0.0)
    for y in range(h):
        for x in range(w):
            var s = Float32(0.0)
            for t in range(-r, r + 1):
                var yy = max(0, min(h - 1, y + t))
                s += k[t + r] * tmp[yy * w + x]
            out[y * w + x] = s
    return out^


def sota_ml_ssim(ref_px: List[UInt32], test_px: List[UInt32], w: Int, h: Int) -> Float64:
    """Mean SSIM on 8-bit Rec.709 luma, 11-tap Gaussian window (sigma 1.5), `K1 = 0.01`, `K2 = 0.03`."""
    var k = List[Float32](length=11, fill=0.0)
    var ks = Float32(0.0)
    for i in range(11):
        var d = Float32(i - 5)
        k[i] = exp(-(d * d) / Float32(2.0 * 1.5 * 1.5))
        ks += k[i]
    for i in range(11):
        k[i] = k[i] / ks
    var a = _luma_plane(ref_px)
    var b = _luma_plane(test_px)
    var aa = List[Float32](length=w * h, fill=0.0)
    var bb = List[Float32](length=w * h, fill=0.0)
    var ab = List[Float32](length=w * h, fill=0.0)
    for i in range(w * h):
        aa[i] = a[i] * a[i]
        bb[i] = b[i] * b[i]
        ab[i] = a[i] * b[i]
    var mu_a = _gauss_blur(a, w, h, k)
    var mu_b = _gauss_blur(b, w, h, k)
    var s_aa = _gauss_blur(aa, w, h, k)
    var s_bb = _gauss_blur(bb, w, h, k)
    var s_ab = _gauss_blur(ab, w, h, k)
    var c1 = Float64(0.01 * 255.0) * Float64(0.01 * 255.0)
    var c2 = Float64(0.03 * 255.0) * Float64(0.03 * 255.0)
    var acc = Float64(0.0)
    for i in range(w * h):
        var ma = Float64(mu_a[i])
        var mb = Float64(mu_b[i])
        var va = Float64(s_aa[i]) - ma * ma
        var vb = Float64(s_bb[i]) - mb * mb
        var cov = Float64(s_ab[i]) - ma * mb
        acc += ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return acc / Float64(max(1, w * h))


def sota_ml_quality(ref_hdr: List[Vec3], test_hdr: List[Vec3], w: Int, h: Int, with_flip: Bool = True) -> SotaMlQuality:
    var r = sota_ml_tonemap(ref_hdr)
    var t = sota_ml_tonemap(test_hdr)
    var fl = Float64(-1.0)
    if with_flip:
        fl = flip_mean(flip_error_map(r, t, w, h))
    return SotaMlQuality(sota_ml_psnr(r, t), sota_ml_ssim(r, t, w, h), fl)


struct SotaMlCapture(Movable):
    """CPU copy of the learned path's scratch after a frame: the network input (temporal output A and
    guide G), the remodulation terms (D, only with `sota_ml.capture`) and the last level's output."""
    var width: Int
    var height: Int
    var rgb: List[Float32]  # 3 per pixel, demodulated irradiance after temporal accumulation
    var variance: List[Float32]
    var z: List[Float32]
    var nx: List[Float32]
    var ny: List[Float32]
    var inv_len: List[Float32]
    var crisp: List[Float32]  # 3 per pixel
    var albedo: List[Float32]  # 3 per pixel
    var out_rgb: List[Float32]  # 3 per pixel, filtered demodulated output (f16)

    def __init__(out self, width: Int, height: Int):
        var n = width * height
        self.width = width
        self.height = height
        self.rgb = List[Float32](length=3 * n, fill=0.0)
        self.variance = List[Float32](length=n, fill=0.0)
        self.z = List[Float32](length=n, fill=0.0)
        self.nx = List[Float32](length=n, fill=0.0)
        self.ny = List[Float32](length=n, fill=0.0)
        self.inv_len = List[Float32](length=n, fill=0.0)
        self.crisp = List[Float32](length=3 * n, fill=0.0)
        self.albedo = List[Float32](length=3 * n, fill=0.0)
        self.out_rgb = List[Float32](length=3 * n, fill=0.0)


def _word(addr: Int, word: Int) -> UInt32:
    return Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=addr + word * 4)[unsafe_offset=0]


def _half2(u: UInt32) -> Tuple[Float32, Float32]:
    var lo = SIMD[DType.uint16, 1](UInt16(u & 0xFFFF))
    var hi = SIMD[DType.uint16, 1](UInt16((u >> 16) & 0xFFFF))
    return (bitcast[DType.float16](lo).cast[DType.float32](), bitcast[DType.float16](hi).cast[DType.float32]())


def sota_ml_read_capture(ren: FlintAaaVulkanRenderer) -> SotaMlCapture:
    """Reads the learned path's scratch (host-visible unified memory) after a synchronous frame."""
    var w = ren.render_w
    var h = ren.render_h
    var cap = SotaMlCapture(w, h)
    var base = ren.buf_history.cpu_address
    if base == 0 or ren.sota_ml.scratch_base < 0:
        return cap^
    var a = ren.sota_ml.a_offset()
    var g = ren.sota_ml.g_offset()
    var d = ren.sota_ml.d_offset()
    var o = ren.sota_ml.final_offset()
    for i in range(w * h):
        var a0 = _half2(_word(base, a + 2 * i))
        var a1 = _half2(_word(base, a + 2 * i + 1))
        cap.rgb[3 * i] = a0[0]
        cap.rgb[3 * i + 1] = a0[1]
        cap.rgb[3 * i + 2] = a1[0]
        cap.variance[i] = a1[1]
        cap.z[i] = bitcast[DType.float32](_word(base, g + 3 * i))
        var g1 = _half2(_word(base, g + 3 * i + 1))
        cap.nx[i] = g1[0]
        cap.ny[i] = g1[1]
        cap.inv_len[i] = bitcast[DType.float32](_word(base, g + 3 * i + 2))
        var d0 = _half2(_word(base, d + 3 * i))
        var d1 = _half2(_word(base, d + 3 * i + 1))
        var d2 = _half2(_word(base, d + 3 * i + 2))
        cap.crisp[3 * i] = d0[0]
        cap.crisp[3 * i + 1] = d0[1]
        cap.crisp[3 * i + 2] = d1[0]
        cap.albedo[3 * i] = d1[1]
        cap.albedo[3 * i + 1] = d2[0]
        cap.albedo[3 * i + 2] = d2[1]
        var o0 = _half2(_word(base, o + 2 * i))
        var o1 = _half2(_word(base, o + 2 * i + 1))
        cap.out_rgb[3 * i] = o0[0]
        cap.out_rgb[3 * i + 1] = o0[1]
        cap.out_rgb[3 * i + 2] = o1[0]
    return cap^
