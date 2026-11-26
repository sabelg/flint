"""Area H (sota:atmo): physically based sky, aerial perspective, height fog, sun / moon, time of day
(`test_flint_v5_sota_atmo.mojo`). Zero external assets.

1. SPIR-V validation of the five shaders.
2. CPU references: transmittance vs the closed form, LUT march vs a converged march, the Hillaire
   multiple-scattering model vs a brute-force Monte Carlo path tracer (all orders), sun / moon positions and
   the moon phase vs published ephemerides, the moon irradiance, the analytic height-fog integral vs numerical
   quadrature.
3. Live GPU (skips without Vulkan): GPU LUTs vs their CPU twins (transmittance, multi-scattering, sky-view,
   aerial-perspective volume), sky pixels vs the CPU sky, aerial perspective on a distant emissive wall, the
   resolved solar disc (limb darkening, irradiance), the moon phase orientation, the height-fog tail vs the
   analytic integral, the time-of-day light hand-over (sun -> moon), `enabled = False` leaves the v5 frame
   unchanged, and GPU time budgets at 1280x720.

`FLINT_SOTA_ATMO_VERBOSE=1` prints the per-case numbers.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, acos, asin, cos, exp, max, min, sin, sqrt
from std.os import getenv
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.sota_atmo_core import (
    RGB,
    SOTA_ATMO_AP_DIM,
    SOTA_ATMO_MS_DIM,
    SOTA_ATMO_SKY_H,
    SOTA_ATMO_SKY_W,
    SOTA_ATMO_TRANS_H,
    SOTA_ATMO_TRANS_W,
    SotaAtmoLights,
    SotaAtmoLuts,
    SotaAtmoParams,
    SotaTimeOfDay,
    ap_column_cpu,
    apply_ap_cpu,
    bake_sky_view_lut,
    height_fog_optical_depth,
    moon_irradiance,
    rgb_luma,
    sample_sky_view,
    sample_transmittance,
    sky_radiance,
    sota_atmo_mc_sky_radiance,
    sun_disc_radiance,
)
from flint.render.sota_atmo_gpu import (
    SOTA_ATMO_G,
    sota_atmo_aerial_shader,
    sota_atmo_apply_shader,
    sota_atmo_multiscatter_shader,
    sota_atmo_skyview_shader,
    sota_atmo_transmittance_shader,
)
from flint.render.spirv_validate import validate_spirv_module


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _verbose() -> Bool:
    return getenv("FLINT_SOTA_ATMO_VERBOSE") == "1"


def _validate(name: String, words: List[UInt32]):
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _fail(name + " SPIR-V invalid: " + (rep.errors[0] if len(rep.errors) > 0 else String("?")))


def _rel(a: Float32, b: Float32, floor_v: Float32) -> Float32:
    return abs(a - b) / max(abs(b), floor_v)


def _deg(x: Float32) -> Float32:
    return x * Float32(57.2957795)


# ============================================================================
# 1. SPIR-V
# ============================================================================

def test_spirv():
    _validate("sota_atmo_transmittance", sota_atmo_transmittance_shader())
    _validate("sota_atmo_multiscatter", sota_atmo_multiscatter_shader())
    _validate("sota_atmo_skyview", sota_atmo_skyview_shader())
    _validate("sota_atmo_aerial", sota_atmo_aerial_shader())
    _validate("sota_atmo_apply", sota_atmo_apply_shader())
    print("  [PASS] Suite 1: SPIR-V of the 5 atmosphere shaders validates")


# ============================================================================
# 2. CPU references
# ============================================================================

def test_cpu_transmittance(luts: SotaAtmoLuts):
    var p = luts.params
    var top = p.r_top_km - p.r_bottom_km
    var worst = Float32(0.0)
    for c in range(3):
        var tau = p.rayleigh_scat[c] * p.rayleigh_h_km * (Float32(1.0) - exp(-top / p.rayleigh_h_km))
        tau += p.mie_ext[c] * p.mie_h_km * (Float32(1.0) - exp(-top / p.mie_h_km)) + p.ozone_abs[c] * p.ozone_half_width_km
        var want = exp(-tau)
        worst = max(worst, _rel(sample_transmittance(luts.trans, p.r_bottom_km, 1.0, p)[c], want, 1e-6))
    _assert(worst < 0.002, "zenith transmittance vs closed form (" + String(worst) + ")")
    print("  [PASS] Suite 2a: zenith transmittance vs closed form, worst rel err", worst)


def test_cpu_sky_vs_mc(luts: SotaAtmoLuts):
    """The LUT model (single scattering + Hillaire multiple scattering) against an all-orders path tracer."""
    var p = luts.params
    var els = [Float32(60.0), Float32(20.0), Float32(5.0)]
    var views = [Vec3(0.0, 1.0, 0.0), Vec3(-0.9, 0.3, 0.3), Vec3(0.95, 0.1, 0.0), Vec3(0.0, 0.05, 1.0)]
    var err_ms = Float32(0.0)
    var err_ss = Float32(0.0)
    var err_steps = Float32(0.0)
    var n = 0
    for ei in range(3):
        var el = els[ei] * Float32(0.0174533)
        var sun = Vec3(cos(el), sin(el), 0.0)
        var lights = SotaAtmoLights(sun, p.solar_irradiance)
        for vi in range(4):
            var v = views[vi].normalized()
            var lut = sky_radiance(luts, lights, 0.2, v)
            var hi = sky_radiance(luts, lights, 0.2, v, 512)
            var ss = sky_radiance(luts, lights, 0.2, v, 32, False)
            var mc = sota_atmo_mc_sky_radiance(p, sun, p.solar_irradiance, 0.2, v, 12000, UInt64(1 + vi + 7 * ei))
            var e_ms = _rel(rgb_luma(lut), rgb_luma(mc), 1e-6)
            var e_ss = _rel(rgb_luma(ss), rgb_luma(mc), 1e-6)
            var e_st = _rel(rgb_luma(lut), rgb_luma(hi), 1e-6)
            if _verbose():
                print("    sun el", els[ei], "view", vi, "MC", rgb_luma(mc), "LUT", rgb_luma(lut), "single", rgb_luma(ss), "err", e_ms, e_ss)
            err_ms += e_ms
            err_ss += e_ss
            err_steps = max(err_steps, e_st)
            n += 1
    err_ms /= Float32(n)
    err_ss /= Float32(n)
    _assert(err_steps < 0.01, "32-step LUT march vs 512 steps (" + String(err_steps) + ")")
    _assert(err_ms < 0.08, "single + multiple scattering vs Monte Carlo, mean luminance error (" + String(err_ms) + ")")
    _assert(err_ss > Float32(2.0) * err_ms, "multiple scattering closes most of the single-scattering gap")
    print("  [PASS] Suite 2b: sky luminance vs all-orders Monte Carlo: mean rel err", err_ms, "(single scattering only:", err_ss,
          "); 32 vs 512 steps worst", err_steps)


def test_cpu_time_of_day(p: SotaAtmoParams):
    # June solstice 2024, Greenwich, solar noon (12:02 UTC): elevation 90 - 51.4769 + 23.4393 = 61.962 deg, due south.
    var t = SotaTimeOfDay(2024, 6, 20, 12.0 + 2.0 / 60.0, 51.4769, 0.0)
    var sd = t.sun_dir()
    var el = _deg(asin(sd.y))
    _assert(abs(el - Float32(61.962)) < 0.1, "solstice noon elevation " + String(el))
    _assert(sd.z < Float32(-0.45) and abs(sd.x) < Float32(0.01), "solstice noon sun due south (-Z)")
    # March equinox 2024 (03:06 UTC), sun over the equator at local noon 12:07 UTC at lon 0.
    var e = SotaTimeOfDay(2024, 3, 20, 12.0 + 7.0 / 60.0, 0.0, 0.0)
    _assert(_deg(asin(e.sun_dir().y)) > 89.5, "equinox noon on the equator is at the zenith")
    # Sunset: Greenwich 2024-06-20 sunset 21:21 BST = 20:21 UTC -> elevation ~ -0.83 deg (refraction + disc).
    var ss = SotaTimeOfDay(2024, 6, 20, 20.0 + 21.0 / 60.0, 51.4769, 0.0)
    var ss_el = _deg(asin(ss.sun_dir().y))
    _assert(abs(ss_el + Float32(0.83)) < 0.35, "sunset elevation " + String(ss_el))
    # Moon: full moon 2024-01-25 17:54 UTC, new moon 2024-01-11 11:57 UTC, first quarter 2024-01-18 03:53 UTC.
    var full = SotaTimeOfDay(2024, 1, 25, 17.9, 0.0, 0.0)
    var new_m = SotaTimeOfDay(2024, 1, 11, 11.95, 0.0, 0.0)
    var fq = SotaTimeOfDay(2024, 1, 18, 3.88, 0.0, 0.0)
    _assert(_deg(full.moon_elongation_rad()) > 170.0, "full moon elongation")
    _assert(_deg(new_m.moon_elongation_rad()) < 10.0, "new moon elongation")
    _assert(abs(full.moon_illuminated_fraction() - Float32(1.0)) < 0.01, "full moon fraction")
    _assert(abs(fq.moon_illuminated_fraction() - Float32(0.5)) < 0.04, "first quarter fraction " + String(fq.moon_illuminated_fraction()))
    var e_full = moon_irradiance(p.solar_irradiance, 0.0, p)[1] / p.solar_irradiance[1]
    var e_half = moon_irradiance(p.solar_irradiance, 1.5708, p)[1] / p.solar_irradiance[1]
    # Full moon ~0.25 lux vs 1.1e5 lux of sun -> ~2.3e-6; a quarter moon is ~1/10 of full (opposition surge).
    _assert(e_full > 1.0e-6 and e_full < 3.0e-6, "full moon / sun irradiance " + String(e_full))
    _assert(e_half < Float32(0.45) * e_full and e_half > Float32(0.15) * e_full, "quarter moon darker than half the full moon")
    print("  [PASS] Suite 2c: time of day: solstice el", el, "sunset el", ss_el, "full / new / first-quarter moon; E_moon/E_sun",
          e_full, "(quarter", e_half, ")")


def test_cpu_height_fog():
    var worst = Float32(0.0)
    var dys = [Float32(0.3), Float32(0.02), Float32(-0.1), Float32(0.0)]
    for i in range(4):
        var dy = dys[i]
        var dens = Float32(0.01)
        var k = Float32(0.05)
        var od = height_fog_optical_depth(dens, k, 1.0, 5.0, dy, 20.0, 600.0)
        var num = Float64(0.0)
        var n = 20000
        for j in range(n):
            var t = Float32(20.0) + (Float32(j) + Float32(0.5)) * Float32(580.0) / Float32(n)
            var y = Float32(5.0) + dy * t
            num += Float64(dens * exp(-k * max(Float32(0.0), y - Float32(1.0))) * Float32(580.0) / Float32(n))
        worst = max(worst, _rel(od, Float32(num), 1e-6))
    _assert(worst < 0.002, "analytic height fog vs quadrature (" + String(worst) + ")")
    print("  [PASS] Suite 2d: analytic height-fog optical depth vs quadrature, worst rel err", worst)


# ============================================================================
# 3. GPU
# ============================================================================

def _read_hist(ren: FlintAaaVulkanRenderer, word: Int, n: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=0.0)
    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), ren.buf_history.cpu_address + word * 4, n * 4)
    return out^


def _hdr_px(ren: FlintAaaVulkanRenderer, x: Int, y: Int) -> RGB:
    var v = List[Float32](length=4, fill=0.0)
    _ = external_call["memcpy", Int](Int(v.unsafe_ptr()), ren.buf_hdr_out.cpu_address + ((y * ren.render_w + x) * 8) * 4, 16)
    return RGB(v[0], v[1], v[2], v[3])


def _box(c: Vec3, he: Vec3) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_beveled_box(c, he, bevel=0.001)
    return mb^.finish()


def _setup(mut ren: FlintAaaVulkanRenderer):
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_motion_blur = False
    ren.enable_bokeh_dof = False
    ren.enable_froxel_fog = False


def _frame(mut ren: FlintAaaVulkanRenderer, cam: Vec3, target: Vec3, fov: Float32, mesh: Int, wall_at: Vec3, wall_scale: Vec3) -> Bool:
    ren.begin_frame(cam_pos=cam, cam_target=target, fov_y_deg=fov, near_z=0.1)
    if mesh >= 0:
        ren.draw_instance(mesh, trs(wall_at, Quat(), wall_scale))
    return ren.render_frame(capture_readback=True)


def _dir_of_pixel(ren: FlintAaaVulkanRenderer, x: Int, y: Int) -> Vec3:
    var fw = Float32(ren.render_w)
    var fh = Float32(ren.render_h)
    var ty = sin(ren.fov_y_deg * Float32(0.00872665)) / cos(ren.fov_y_deg * Float32(0.00872665))
    var tx = ty * fw / fh
    var nx = (Float32(x) + Float32(0.5)) / fw * Float32(2.0) - Float32(1.0)
    var ny = Float32(1.0) - (Float32(y) + Float32(0.5)) / fh * Float32(2.0)
    return (ren.cam_fwd + ren.cam_right * (nx * tx) + ren.cam_up * (ny * ty)).normalized()


def test_gpu_luts_and_sky():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _assert(ren.sota_atmo.pipelines_ready(), "sota:atmo pipelines created")
    _setup(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.update_ibl = False
    ren.sota_atmo.sun_dir = Vec3(0.6, 0.30, 0.25).normalized()
    ren.sota_atmo.moon_enabled = True
    ren.sota_atmo.moon_dir = Vec3(-0.5, 0.45, -0.3).normalized()
    var cam = Vec3(0.0, 2.0, 0.0)
    for _ in range(2):
        _assert(_frame(ren, cam, Vec3(10.0, 4.0, 3.0), 75.0, -1, Vec3(), Vec3()), "frame")
    var p = ren.sota_atmo.params
    var luts = SotaAtmoLuts(p)
    ren.sota_atmo.update_lights()
    var lights = ren.sota_atmo.lights
    var alt = ren.sota_atmo.camera_altitude_km(cam.y)

    # (a) transmittance + multi-scattering LUTs.
    var gt = _read_hist(ren, ren.sota_atmo.trans_off(), SOTA_ATMO_TRANS_W * SOTA_ATMO_TRANS_H * 4)
    var worst_t = Float32(0.0)
    for i in range(SOTA_ATMO_TRANS_W * SOTA_ATMO_TRANS_H):
        for c in range(3):
            worst_t = max(worst_t, _rel(gt[i * 4 + c], luts.trans[i][c], 1e-3))
    var gm = _read_hist(ren, ren.sota_atmo.ms_off(), SOTA_ATMO_MS_DIM * SOTA_ATMO_MS_DIM * 4)
    var worst_m = Float32(0.0)
    var ms_peak = Float32(0.0)
    for i in range(SOTA_ATMO_MS_DIM * SOTA_ATMO_MS_DIM):
        ms_peak = max(ms_peak, luts.ms[i][2])
    for i in range(SOTA_ATMO_MS_DIM * SOTA_ATMO_MS_DIM):
        for c in range(3):
            worst_m = max(worst_m, _rel(gm[i * 4 + c], luts.ms[i][c], ms_peak * Float32(0.01)))
    _assert(worst_t < 0.005, "GPU transmittance LUT vs CPU (" + String(worst_t) + ")")
    _assert(worst_m < 0.01, "GPU multi-scattering LUT vs CPU (" + String(worst_m) + ")")

    # (b) sky-view LUT (sun + moon) vs the CPU twin.
    var cpu_sky = bake_sky_view_lut(luts, lights, alt)
    var gs = _read_hist(ren, ren.sota_atmo.sky_off(), SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H * 4)
    var sky_peak = Float32(0.0)
    for i in range(SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H):
        sky_peak = max(sky_peak, rgb_luma(cpu_sky[i]))
    var worst_s = Float32(0.0)
    for i in range(SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H):
        for c in range(3):
            worst_s = max(worst_s, _rel(gs[i * 4 + c], cpu_sky[i][c], sky_peak * Float32(0.01)))
    _assert(worst_s < 0.01, "GPU sky-view LUT vs CPU (" + String(worst_s) + ")")

    # (c) aerial-perspective froxels: a few columns vs `ap_column_cpu`.
    var ga = _read_hist(ren, ren.sota_atmo.ap_off(), SOTA_ATMO_AP_DIM * SOTA_ATMO_AP_DIM * SOTA_ATMO_AP_DIM * 8)
    var worst_a = Float32(0.0)
    var ty = sin(ren.fov_y_deg * Float32(0.00872665)) / cos(ren.fov_y_deg * Float32(0.00872665))
    var tx = ty * Float32(w) / Float32(h)
    var cols = [(3, 5), (16, 16), (28, 30)]
    for ci in range(3):
        var cx = cols[ci][0]
        var cy = cols[ci][1]
        var nx = (Float32(cx) + Float32(0.5)) / Float32(SOTA_ATMO_AP_DIM) * Float32(2.0) - Float32(1.0)
        var ny = Float32(1.0) - (Float32(cy) + Float32(0.5)) / Float32(SOTA_ATMO_AP_DIM) * Float32(2.0)
        var d = (ren.cam_fwd + ren.cam_right * (nx * tx) + ren.cam_up * (ny * ty)).normalized()
        var col = ap_column_cpu(luts, lights, alt, d, ren.sota_atmo.ap_max_km)
        var s_peak = rgb_luma(col[2 * (SOTA_ATMO_AP_DIM - 1)])
        for s in range(SOTA_ATMO_AP_DIM):
            var o = ((s * SOTA_ATMO_AP_DIM + cy) * SOTA_ATMO_AP_DIM + cx) * 8
            for c in range(3):
                worst_a = max(worst_a, _rel(ga[o + c], col[2 * s][c], s_peak * Float32(0.01)))
                worst_a = max(worst_a, _rel(ga[o + 4 + c], col[2 * s + 1][c], 1e-3))
    _assert(worst_a < 0.01, "GPU aerial-perspective volume vs CPU (" + String(worst_a) + ")")

    # (d) sky pixels (HDR after the apply pass) vs the CPU sky through the CPU LUT.
    var worst_px = Float32(0.0)
    for k in range(40):
        var x = 4 + (k * 37) % (w - 8)
        var y = 4 + (k * 23) % (h // 2)
        var c = _hdr_px(ren, x, y)
        if c[3] < Float32(1.0e5):
            continue
        var want = sample_sky_view(cpu_sky, _dir_of_pixel(ren, x, y), alt, p)
        for ch in range(3):
            worst_px = max(worst_px, _rel(c[ch], want[ch], sky_peak * Float32(0.02)))
    _assert(worst_px < 0.03, "sky pixels vs CPU sky (" + String(worst_px) + ")")
    ren.shutdown()
    print("  [PASS] Suite 3a: GPU vs CPU: transmittance", worst_t, "multi-scattering", worst_m, "sky-view", worst_s,
          "aerial volume", worst_a, "sky pixels", worst_px)


def test_gpu_aerial_perspective_and_fog():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    ren.ibl_intensity = 0.0
    var m_emis = AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
    m_emis.emissive = Vec3(0.02, 0.02, 0.02)
    var mat = ren.add_material(m_emis)
    var wall = ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5)), mat).mesh_id
    ren.sota_atmo.enabled = True
    ren.sota_atmo.update_ibl = False
    ren.sota_atmo.drive_renderer_light = False
    ren.sota_atmo.sun_dir = Vec3(0.2, 0.5, 0.8).normalized()
    var cam = Vec3(0.0, 2.0, 0.0)
    var dist = Float32(6000.0)
    for _ in range(2):
        _assert(_frame(ren, cam, Vec3(0.0, 2.0, 100.0), 40.0, wall, Vec3(0.0, 2.0, dist), Vec3(4000.0, 4000.0, 10.0)), "frame")
    var p = ren.sota_atmo.params
    var luts = SotaAtmoLuts(p)
    ren.sota_atmo.update_lights()
    var alt = ren.sota_atmo.camera_altitude_km(cam.y)
    var cx = w // 2
    var cy = h // 2
    var got = _hdr_px(ren, cx, cy)
    var d = _dir_of_pixel(ren, cx, cy)
    var col = ap_column_cpu(luts, ren.sota_atmo.lights, alt, d, ren.sota_atmo.ap_max_km)
    var dist_km = (got[3] * Float32(sqrt(Float64(1.0) + Float64(0.0)))) * ren.sota_atmo.km_per_unit
    _ = dist_km
    var surf_d = (dist - Float32(5.0)) / d.z  # front face of the wall, along the ray
    var want = apply_ap_cpu(col, RGB(0.02, 0.02, 0.02, 0.0), surf_d * ren.sota_atmo.km_per_unit, ren.sota_atmo.ap_max_km)
    var err_ap = Float32(0.0)
    for c in range(3):
        err_ap = max(err_ap, _rel(got[c], want[c], 1e-3))
    _assert(err_ap < 0.03, "aerial perspective on a wall at 6 km vs CPU (" + String(err_ap) + ")")
    var blue_haze = got[2] > got[0] and got[2] > Float32(0.5) * want[2]
    _assert(blue_haze, "distant dark wall turns blue (" + String(got[0]) + ", " + String(got[2]) + ")")

    # Height-fog tail without froxels (standalone analytic fog), sun off -> only the SH ambient (0 here) -> pure extinction.
    ren.sota_atmo.aerial_perspective = False
    ren.sota_atmo.height_fog_density = 0.002
    ren.fog_height_falloff = 0.01
    ren.fog_base_height_y = 0.0
    ren.fog_albedo = Vec3(0.0, 0.0, 0.0)
    for _ in range(2):
        _assert(_frame(ren, cam, Vec3(0.0, 2.0, 100.0), 40.0, wall, Vec3(0.0, 2.0, dist), Vec3(4000.0, 4000.0, 10.0)), "frame")
    var fogged = _hdr_px(ren, cx, cy)
    var od = height_fog_optical_depth(0.002, 0.01, 0.0, cam.y, d.y, 0.0, min(surf_d, ren.sota_atmo.fog_tail_max_dist))
    var want_f = Float32(0.02) * exp(-od)
    var err_f = _rel(fogged[1], want_f, 1e-5)
    _assert(err_f < 0.02, "height-fog transmittance vs analytic (" + String(err_f) + ", got " + String(fogged[1]) + " want " + String(want_f) + ")")
    ren.shutdown()
    print("  [PASS] Suite 3b: aerial perspective at 6 km vs CPU", err_ap, "(rgb", got[0], got[1], got[2], "); height-fog tail vs analytic", err_f)


def test_gpu_sun_moon_discs():
    var w = 256
    var h = 256
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.update_ibl = False
    var sun = Vec3(0.0, 0.5, 0.866).normalized()
    ren.sota_atmo.sun_dir = sun
    var cam = Vec3(0.0, 2.0, 0.0)
    var fov = Float32(2.0)  # 0.0136 deg / px: the 0.533 deg sun covers ~39 px
    for _ in range(2):
        _assert(_frame(ren, cam, cam + sun * Float32(100.0), fov, -1, Vec3(), Vec3()), "frame")
    var p = ren.sota_atmo.params
    var luts = SotaAtmoLuts(p)
    ren.sota_atmo.update_lights()
    var lights = ren.sota_atmo.lights
    var alt = ren.sota_atmo.camera_altitude_km(cam.y)
    var centre = _hdr_px(ren, w // 2, h // 2)
    var sky_c = sky_radiance(luts, lights, alt, sun)
    var want_c = sun_disc_radiance(luts, lights, alt, _dir_of_pixel(ren, w // 2, h // 2)) + sky_c
    var err_c = _rel(centre[1], want_c[1], 1e-3)
    _assert(err_c < 0.03, "sun disc centre radiance vs CPU (" + String(err_c) + ")")
    # Disc irradiance: sum (L - sky) * pixel solid angle == E_sun * T(view).
    var ty = sin(fov * Float32(0.00872665)) / cos(fov * Float32(0.00872665))
    var px_sa = (Float32(2.0) * ty / Float32(h)) * (Float32(2.0) * ty / Float32(h))
    var e = Float64(0.0)
    for y in range(h):
        for x in range(w):
            var c = _hdr_px(ren, x, y)
            e += Float64(max(Float32(0.0), c[1] - sky_c[1]) * px_sa)
    var t_view = sample_transmittance(luts.trans, p.r_bottom_km + alt, sun.y, p)
    var want_e = p.solar_irradiance[1] * t_view[1]
    var err_e = _rel(Float32(e), want_e, 1e-6)
    _assert(err_e < 0.05, "resolved sun disc integrates to the transmitted irradiance (" + String(err_e) + ")")
    # Limb darkening: radiance at 0.9 R is below the centre.
    var edge = _hdr_px(ren, w // 2 + Int(Float32(0.9) * p.sun_angular_radius / (Float32(2.0) * ty / Float32(h))), h // 2)
    _assert(edge[1] < Float32(0.85) * centre[1], "limb darkening")

    # Moon at quarter phase: the sun at 90 deg to the right of the moon lights its right half.
    ren.sota_atmo.sun_dir = Vec3(1.0, -0.2, 0.0).normalized()  # below the horizon: no sun disc, dark sky
    var moon = Vec3(0.0, 0.5, 0.866).normalized()
    ren.sota_atmo.moon_enabled = True
    ren.sota_atmo.moon_dir = moon
    for _ in range(2):
        _assert(_frame(ren, cam, cam + moon * Float32(100.0), fov, -1, Vec3(), Vec3()), "frame")
    var r_m = Int(Float32(0.5) * p.moon_angular_radius / (Float32(2.0) * ty / Float32(h)))
    var right = _hdr_px(ren, w // 2 + r_m, h // 2)
    var left = _hdr_px(ren, w // 2 - r_m, h // 2)
    var bg = _hdr_px(ren, 4, 4)
    _assert(right[1] > Float32(5.0) * max(left[1], Float32(1e-12)) and right[1] > Float32(10.0) * bg[1], "quarter moon: lit limb faces the sun")
    ren.shutdown()
    print("  [PASS] Suite 3c: sun disc centre vs CPU", err_c, ", disc irradiance vs E*T", err_e, "; moon phase lit toward the sun (",
          right[1], "vs", left[1], ")")


def test_gpu_time_of_day_and_off():
    var w = 160
    var h = 90
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    var cam = Vec3(0.0, 2.0, 0.0)
    # Off: two identical frames, no area-H passes.
    for _ in range(2):
        _assert(_frame(ren, cam, Vec3(5.0, 3.0, 9.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var a = ren.readback_pixels()
    for t in ren.get_gpu_pass_timings():
        _assert(t.pass_name.find("SotaAtmo") < 0, "no area-H pass when disabled")
    _assert(_frame(ren, cam, Vec3(5.0, 3.0, 9.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var b = ren.readback_pixels()
    var same = True
    for i in range(len(a)):
        if a[i] != b[i]:
            same = False
    _assert(same, "disabled path is deterministic")
    # Time of day drives the renderer's light: day -> the sun, night -> the moon (full moon 2024-01-25 23:00 UTC at lon 0).
    ren.sota_atmo.enabled = True
    ren.sota_atmo.use_time_of_day = True
    ren.sota_atmo.time_of_day = SotaTimeOfDay(2024, 1, 25, 12.0, 30.0, 0.0)
    _assert(_frame(ren, cam, Vec3(5.0, 3.0, 9.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var day_sun = ren.sota_atmo.time_of_day.sun_dir()
    _assert(ren.sun_dir.dot(day_sun) > Float32(0.9999), "day: renderer light is the sun")
    _assert(ren.sun_radiance.y > Float32(1.0), "day: sun radiance")
    var found = False
    for t in ren.get_gpu_pass_timings():
        if t.pass_name == "SotaAtmoApply":
            found = True
    _assert(found, "area-H passes recorded when enabled")
    _assert(ren.sota_atmo.ibl_valid, "IBL baked from the physical sky")
    var day_sh0 = ren.sh9_irradiance[0].z
    ren.sota_atmo.time_of_day = SotaTimeOfDay(2024, 1, 25, 23.0, 30.0, 0.0)
    _assert(_frame(ren, cam, Vec3(5.0, 3.0, 9.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var moon = ren.sota_atmo.time_of_day.moon_dir()
    _assert(moon.y > Float32(0.2), "full moon is up at 23:00 UTC")
    _assert(ren.sun_dir.dot(moon) > Float32(0.9999), "night: renderer light is the moon")
    _assert(ren.sun_radiance.y < Float32(1e-4) and ren.sun_radiance.y > Float32(1e-7), "night: moonlight radiance " + String(ren.sun_radiance.y))
    _assert(ren.sh9_irradiance[0].z < Float32(1e-3) * day_sh0, "night IBL re-baked (" + String(ren.sh9_irradiance[0].z) + " vs day " + String(day_sh0) + ")")
    print("  [PASS] Suite 3d: disabled = v5 path (no passes, deterministic); time of day hands the light from sun to moon;",
          "IBL re-bake", ren.sota_atmo.last_ibl_ms, "ms")
    ren.shutdown()


def test_gpu_budgets():
    var w = 1280
    var h = 720
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.update_ibl = False
    ren.sota_atmo.moon_enabled = True
    ren.enable_froxel_fog = True
    ren.fog_density = 0.003
    var m = ren.add_material(AaaMaterial(base_color=Vec3(0.3, 0.3, 0.3), roughness=0.8, metallic=0.0))
    var wall = ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5)), m).mesh_id
    var ls = List[Float64]()
    var aps = List[Float64]()
    for i in range(48):
        _ = _frame(ren, Vec3(0.0, 2.0, 0.0), Vec3(3.0, 2.5, 20.0), 60.0, wall, Vec3(0.0, 0.0, 400.0), Vec3(800.0, 60.0, 10.0))
        if i < 8:
            continue  # warm-up
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == "SotaAtmoLuts":
                ls.append(t.gpu_ms)
            elif t.pass_name == "SotaAtmoApply":
                aps.append(t.gpu_ms)
    _assert(len(ls) > 0 and len(aps) > 0, "area-H GPU timings recorded")
    var med_l = _median(ls)
    var med_a = _median(aps)
    # The GPU is shared: gate on the median (robust to a co-tenant burst), report it.
    print("    SotaAtmoLuts median", med_l, "ms, SotaAtmoApply median", med_a, "ms at 1280x720")
    _assert(med_l < 0.5, "sky-view + aerial LUTs within 0.5 ms (median " + String(med_l) + ")")
    _assert(med_a < 0.6, "apply within 0.6 ms at 720p (median " + String(med_a) + ")")
    ren.shutdown()
    print("  [PASS] Suite 3e: GPU budgets at 1280x720, medians (LUTs", med_l, "ms, apply", med_a, "ms)")


def _median(v: List[Float64]) -> Float64:
    var s = v.copy()
    for i in range(1, len(s)):
        var x = s[i]
        var j = i - 1
        while j >= 0 and s[j] > x:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = x
    return s[len(s) // 2]


def main():
    print("=== Flint v5 sota:atmo (Area H) Physical Sky & Atmosphere Suite ===")
    test_spirv()
    var p = SotaAtmoParams()
    var luts = SotaAtmoLuts(p)
    test_cpu_transmittance(luts)
    test_cpu_sky_vs_mc(luts)
    test_cpu_time_of_day(p)
    test_cpu_height_fog()
    test_gpu_luts_and_sky()
    test_gpu_aerial_perspective_and_fog()
    test_gpu_sun_moon_discs()
    test_gpu_time_of_day_and_off()
    test_gpu_budgets()
    print("=== ALL sota:atmo TESTS PASSED ===")
