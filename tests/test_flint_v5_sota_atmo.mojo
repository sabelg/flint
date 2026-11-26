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
4. Round 3: SPIR-V of the cloud / IBL shaders; the star catalogue vs the N(<m) counts (no cube-corner crowding)
   and the celestial pole; the night sky (airglow, Milky Way, star PSFs) vs its CPU twin; the GPU IBL (SH9 vs
   CPU, BC6H cube vs the CPU prefilter, bit-exact BC6H vs `encode_bc6h_hdr`, time-lapse hitch r2 vs r3); the
   clouds (noise and march vs CPU twins, coverage 0 = off, temporal accumulation, yaw + wind reprojection,
   sunset / moonlight sanity) and the round-3 GPU budgets.

`FLINT_SOTA_ATMO_VERBOSE=1` prints the per-case numbers.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, acos, asin, cos, exp, max, min, sin, sqrt
from std.os import getenv
from flint.assets.bc_compress import decode_bc6h_hdr, encode_bc6h_hdr
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import get_time
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_ibl import eval_sh9_cpu
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
from flint.render.sota_atmo_clouds import (
    SOTA_CLOUD_BASE_N,
    SOTA_CLOUD_DETAIL_N,
    SOTA_CLOUD_WEATHER_N,
    CloudNoiseCpu,
    cloud_base_texel_cpu,
    cloud_detail_texel_cpu,
    cloud_march_cpu,
    cloud_weather_texel_cpu,
    sota_cloud_march_shader,
    sota_cloud_noise_shader,
    sota_cloud_weather_shader,
)
from flint.render.sota_atmo_ibl import (
    ibl_mip_block_offset,
    ibl_prefilter_texel_cpu,
    ibl_sh9_from_sky_view_cpu,
    ibl_test_pattern_texel,
    sota_atmo_ibl_cube_shader,
    sota_atmo_ibl_sh_shader,
)
from flint.render.sota_atmo_night import (
    SOTA_NIGHT_CELLS,
    galactic_pole_equ,
    night_sky_radiance_cpu,
    star_cell_cpu,
    star_count_brighter,
    surface_brightness_radiance,
    world_to_equatorial_rows,
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
    var day_sh0 = ren.sota_atmo.read_gpu_sh9(ren.buf_history.cpu_address)[0].z
    ren.sota_atmo.time_of_day = SotaTimeOfDay(2024, 1, 25, 23.0, 30.0, 0.0)
    _assert(_frame(ren, cam, Vec3(5.0, 3.0, 9.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var moon = ren.sota_atmo.time_of_day.moon_dir()
    _assert(moon.y > Float32(0.2), "full moon is up at 23:00 UTC")
    _assert(ren.sun_dir.dot(moon) > Float32(0.9999), "night: renderer light is the moon")
    _assert(ren.sun_radiance.y < Float32(1e-4) and ren.sun_radiance.y > Float32(1e-7), "night: moonlight radiance " + String(ren.sun_radiance.y))
    var night_sh0 = ren.sota_atmo.read_gpu_sh9(ren.buf_history.cpu_address)[0].z
    _assert(night_sh0 < Float32(1e-3) * day_sh0, "night IBL re-baked (" + String(night_sh0) + " vs day " + String(day_sh0) + ")")
    print("  [PASS] Suite 3d: disabled = v5 path (no passes, deterministic); time of day hands the light from sun to moon;",
          "IBL on the GPU (CPU re-bake", ren.sota_atmo.last_ibl_ms, "ms)")
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


# ============================================================================
# 4. Round 3: night sky, GPU IBL, volumetric clouds
# ============================================================================

def test_spirv_r3():
    _validate("sota_atmo_ibl_sh", sota_atmo_ibl_sh_shader())
    _validate("sota_atmo_ibl_cube", sota_atmo_ibl_cube_shader())
    _validate("sota_cloud_noise", sota_cloud_noise_shader())
    _validate("sota_cloud_weather", sota_cloud_weather_shader(7))
    _validate("sota_cloud_march", sota_cloud_march_shader())
    print("  [PASS] Suite 4a: SPIR-V of the 5 round-3 shaders validates (apply with night + cloud composite: Suite 1)")


def test_cpu_stars():
    """Catalogue counts vs N(<m), no cube-corner crowding, and the celestial pole stays put."""
    var m_lim = Float32(6.5)
    var n = SOTA_NIGHT_CELLS
    var gp = galactic_pole_equ()
    var n_all = 0
    var n_b = 0
    var w_ctr = Float64(0.0)
    var o_ctr = Float64(0.0)
    var w_cor = Float64(0.0)
    var o_cor = Float64(0.0)
    for face in range(6):
        for cy in range(n):
            var tc = (Float32(cy) + Float32(0.5)) * Float32(2.0) / Float32(n) - Float32(1.0)
            for cx in range(n):
                var sc = (Float32(cx) + Float32(0.5)) * Float32(2.0) / Float32(n) - Float32(1.0)
                var r2 = sc * sc + tc * tc
                var q = Float32(1.0) + r2
                var dom = Float64(Float32(4.0) / Float32(n * n) / (q * sqrt(q)))
                var region = 0 if r2 < Float32(0.15) else (1 if r2 > Float32(1.0) else -1)
                if region == 0:
                    o_ctr += dom
                elif region == 1:
                    o_cor += dom
                var cell = star_cell_cpu(face, cx, cy, m_lim, 1)
                if not cell.present:
                    continue
                n_all += 1
                if cell.mag < Float32(4.5):
                    n_b += 1
                if region >= 0:
                    var sb = cell.dir_equ.dot(gp)
                    var wgt = Float64(1.0) / Float64(Float32(0.6) + Float32(1.2) * exp(-(sb * sb) / Float32(0.06)))
                    if region == 0:
                        w_ctr += wgt
                    else:
                        w_cor += wgt
    var want = star_count_brighter(m_lim)
    var want_b = star_count_brighter(4.5)
    var e_all = _rel(Float32(n_all), want, 1.0)
    var e_b = _rel(Float32(n_b), want_b, 1.0)
    var crowd = Float32((w_cor / o_cor) / (w_ctr / o_ctr))
    _assert(e_all < 0.04, "whole-sky star count vs N(<6.5) (" + String(n_all) + " vs " + String(want) + ")")
    _assert(e_b < 0.10, "stars brighter than 4.5 vs N(<4.5) (" + String(n_b) + " vs " + String(want_b) + ")")
    _assert(crowd > Float32(0.85) and crowd < Float32(1.15), "star density per sr, cube corners vs face centres (" + String(crowd) + ")")
    # Celestial pole: fixed in the sky at elevation = latitude; the equator turns 15.04 deg per solar hour.
    var a = world_to_equatorial_rows(SotaTimeOfDay(2026, 3, 20, 0.0, 45.0, 0.0))
    var b = world_to_equatorial_rows(SotaTimeOfDay(2026, 3, 20, 6.0, 45.0, 0.0))
    var pole_move = _deg(acos(min(Float32(1.0), a[2].dot(b[2]))))
    var pole_el = _deg(asin(a[2].y))
    var turn = _deg(acos(max(Float32(-1.0), min(Float32(1.0), a[0].dot(b[0])))))
    _assert(pole_move < Float32(0.05), "celestial pole fixed over 6 h (float32 acos floor 0.02 deg)  (" + String(pole_move) + " deg)")
    _assert(abs(pole_el - Float32(45.0)) < Float32(0.05), "pole elevation = latitude (" + String(pole_el) + ")")
    _assert(abs(turn - Float32(90.246)) < Float32(0.1), "sidereal rotation over 6 h (" + String(turn) + " deg)")
    _assert(abs(a[0].dot(a[1])) < Float32(1e-5) and abs(a[1].dot(a[2])) < Float32(1e-5) and abs(a[0].length() - Float32(1.0)) < Float32(1e-5), "rotation is orthonormal")
    print("  [PASS] Suite 4b: stars", n_all, "vs N(<6.5)", want, "; brighter than 4.5:", n_b, "vs", want_b, "; corner/centre density", crowd,
          "; pole drift", pole_move, "deg, elevation", pole_el, ", 6 h turn", turn, "deg")


def _jit(ren: FlintAaaVulkanRenderer) -> Tuple[Float32, Float32]:
    var v = List[Float32](length=2, fill=0.0)
    _ = external_call["memcpy", Int](Int(v.unsafe_ptr()), ren.buf_globals.cpu_address + 101 * 4, 8)
    return (v[0], v[1])


def _dir_of_pixel_j(ren: FlintAaaVulkanRenderer, x: Int, y: Int, jx: Float32, jy: Float32) -> Vec3:
    """The apply pass's ray (with this frame's TAA jitter)."""
    var fw = Float32(ren.render_w)
    var fh = Float32(ren.render_h)
    var ty = sin(ren.fov_y_deg * Float32(0.00872665)) / cos(ren.fov_y_deg * Float32(0.00872665))
    var tx = ty * fw / fh
    var fx0 = Float32(x) + Float32(0.5) - jx * fw * Float32(0.5)
    var fy0 = Float32(y) + Float32(0.5) + jy * fh * Float32(0.5)
    var nx = fx0 / fw * Float32(2.0) - Float32(1.0)
    var ny = Float32(1.0) - fy0 / fh * Float32(2.0)
    return (ren.cam_fwd + ren.cam_right * (nx * tx) + ren.cam_up * (ny * ty)).normalized()


def _gpu_sky_lut(ren: FlintAaaVulkanRenderer) -> List[RGB]:
    var gs = _read_hist(ren, ren.sota_atmo.sky_off(), SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H * 4)
    var lut = List[RGB](capacity=SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H)
    for i in range(SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H):
        lut.append(RGB(gs[i * 4], gs[i * 4 + 1], gs[i * 4 + 2], 0.0))
    return lut^


def test_gpu_night_sky():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.use_time_of_day = True
    ren.sota_atmo.time_of_day = SotaTimeOfDay(2026, 8, 12, 23.5, 45.0, 0.0)  # new moon, the summer Milky Way
    ren.sota_atmo.night.star_psf_px = 1.5
    var cam = Vec3(0.0, 2.0, 0.0)
    for _ in range(2):
        _assert(_frame(ren, cam, cam + Vec3(0.3, 0.55, -0.75) * Float32(100.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var p = ren.sota_atmo.params
    var luts = SotaAtmoLuts(p)
    var alt = ren.sota_atmo.camera_altitude_km(cam.y)
    var lut = _gpu_sky_lut(ren)
    var nt = ren.sota_atmo.night.copy()
    var e0 = nt.e0(rgb_luma(p.solar_irradiance))
    var bg_l = surface_brightness_radiance(e0, nt.airglow_mag_arcsec2) * nt.airglow_scale
    var mw_l = surface_brightness_radiance(e0, nt.milky_way_mag_arcsec2) * nt.milky_way_scale
    var rows = world_to_equatorial_rows(ren.sota_atmo.time_of_day)
    var ty = sin(ren.fov_y_deg * Float32(0.00872665)) / cos(ren.fov_y_deg * Float32(0.00872665))
    var pix_ang = ty * Float32(2.0) / Float32(h)
    var j = _jit(ren)
    var moon = ren.sota_atmo.lights.moon_dir
    var sum_got = Float64(0.0)
    var sum_want = Float64(0.0)
    var err_bg = Float64(0.0)
    var n_bg = 0
    var worst_star = Float32(0.0)
    var err_star = Float64(0.0)
    var n_star = 0
    var st_got = Float64(0.0)
    var st_want = Float64(0.0)
    var bias_bg = Float64(0.0)
    for y in range(h):
        for x in range(w):
            var c = _hdr_px(ren, x, y)
            if c[3] < Float32(1.0e5):
                continue
            var d = _dir_of_pixel_j(ren, x, y, j[0], j[1])
            if d.y < Float32(0.03) or d.dot(moon) > Float32(0.9998):
                continue
            var sky = sample_sky_view(lut, d, alt, p)
            var t_view = sample_transmittance(luts.trans, p.r_bottom_km + alt, d.y, p)
            var bg = night_sky_radiance_cpu(d, rows, 0.0, nt.limiting_magnitude, bg_l, mw_l, nt.star_psf_px, nt.seed, e0, pix_ang)
            var full = night_sky_radiance_cpu(d, rows, nt.star_brightness, nt.limiting_magnitude, bg_l, mw_l, nt.star_psf_px, nt.seed, e0, pix_ang)
            var got = rgb_luma(c) - rgb_luma(sky)
            var want_bg = rgb_luma(bg * t_view)
            var want = rgb_luma(full * t_view)
            sum_got += Float64(got)
            sum_want += Float64(want)
            st_got += Float64(got - want_bg)
            st_want += Float64(want - want_bg)
            if want < Float32(1.01) * want_bg:
                bias_bg += Float64((got - want) / want)
                err_bg += Float64(_rel(got, want, 1e-30))
                n_bg += 1
            elif want > Float32(1.2) * want_bg:
                # The star part alone (got - background) vs the CPU PSF.
                var e = abs(got - want) / (want - want_bg)
                worst_star = max(worst_star, e)
                err_star += Float64(e)
                n_star += 1
    var e_bg = Float32(err_bg / Float64(max(1, n_bg)))
    var e_star = Float32(err_star / Float64(max(1, n_star)))
    var e_sum = _rel(Float32(sum_got), Float32(sum_want), 1e-30)
    if _verbose():
        print("    bg mean err", e_bg, "bg signed bias", bias_bg / Float64(max(1, n_bg)), "star energy got", st_got, "want", st_want, "frame", e_sum)
    _assert(n_bg > 1000 and n_star > 50, "night frame has background and star pixels (" + String(n_bg) + ", " + String(n_star) + ")")
    _assert(e_bg < 0.02, "airglow + Milky Way vs CPU twin, mean rel err " + String(e_bg))
    _assert(e_sum < 0.02, "night-sky energy in the frame vs CPU twin " + String(e_sum))
    # Star light: the frame's total star energy is exact to ~1 %; single PSF pixels differ by a few % (sub-pixel shape).
    var e_st = _rel(Float32(st_got), Float32(st_want), 1e-30)
    _assert(e_st < 0.03, "star energy in the frame vs CPU twin " + String(e_st))
    _assert(e_star < 0.10, "star PSF pixels vs CPU twin, mean rel err " + String(e_star) + " (worst " + String(worst_star) + ")")
    ren.shutdown()
    print("  [PASS] Suite 4c: night sky vs CPU twin: background mean err", e_bg, "(", n_bg, "px ), star energy", e_st,
          ", star PSF pixels mean", e_star, "worst", worst_star, "(", n_star, "px ), frame", e_sum)


def _read_hist_bytes(ren: FlintAaaVulkanRenderer, word: Int, nbytes: Int) -> List[UInt8]:
    var out = List[UInt8](length=nbytes, fill=0)
    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), ren.buf_history.cpu_address + word * 4, nbytes)
    return out^


def _p90(v: List[Float64]) -> Float64:
    var s = v.copy()
    for i in range(1, len(s)):
        var x = s[i]
        var j = i - 1
        while j >= 0 and s[j] > x:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = x
    return s[(len(s) * 9) // 10]


def test_gpu_ibl():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.sun_dir = Vec3(0.5, 0.35, 0.3).normalized()
    var cam = Vec3(0.0, 2.0, 0.0)
    var tgt = Vec3(10.0, 4.0, 3.0)
    for _ in range(3):
        _assert(_frame(ren, cam, tgt, 75.0, -1, Vec3(), Vec3()), "frame")
    _assert(ren.sota_atmo.gpu_ibl_active() and ren.sota_atmo.ibl.cube_updates >= 1, "GPU IBL ran")
    _assert(ren.sota_atmo.last_ibl_ms == 0.0, "no CPU IBL bake on the GPU path")
    var p = ren.sota_atmo.params
    var alt = ren.sota_atmo.camera_altitude_km(cam.y)
    var lut = _gpu_sky_lut(ren)
    # (a) SH9: GPU vs the CPU projection of the same (read-back) sky-view LUT; and what b_globals[72..99) holds.
    var sh_cpu = ibl_sh9_from_sky_view_cpu(lut, SOTA_ATMO_SKY_W, SOTA_ATMO_SKY_H, alt, p)
    var sh_gpu = ren.sota_atmo.read_gpu_sh9(ren.buf_history.cpu_address)
    var floor_sh = max(sh_cpu[0].x, max(sh_cpu[0].y, sh_cpu[0].z)) * Float32(0.01)
    var worst_sh = Float32(0.0)
    for k in range(9):
        worst_sh = max(worst_sh, _rel(sh_gpu[k].x, sh_cpu[k].x, floor_sh))
        worst_sh = max(worst_sh, _rel(sh_gpu[k].y, sh_cpu[k].y, floor_sh))
        worst_sh = max(worst_sh, _rel(sh_gpu[k].z, sh_cpu[k].z, floor_sh))
    _assert(worst_sh < 0.01, "GPU SH9 vs CPU projection (" + String(worst_sh) + ")")
    var g = List[Float32](length=27, fill=0.0)
    _ = external_call["memcpy", Int](Int(g.unsafe_ptr()), ren.buf_globals.cpu_address + 72 * 4, 27 * 4)
    _assert(g[0] == sh_gpu[0].x and g[26] == sh_gpu[8].z, "SH9 copied into b_globals[72..99)")
    _assert(ren.sh9_irradiance[0].z == sh_gpu[0].z, "CPU-side SH9 follows the GPU one frame late")

    # (b) the BC6H GGX cube vs the CPU prefilter of the same LUT, encoded by the CPU mode-11 encoder (the GPU path's
    # codec), every mip and face; `codec` = what BC6H mode 11 itself loses against the float prefilter.
    var base = ren.sota_atmo.ibl.base_dim
    var mips = ren.sota_atmo.ibl.mips
    var worst_c = Float32(0.0)
    var worst_codec = Float32(0.0)
    for m in range(mips):
        var d = max(4, base >> m)
        var nb = d // 4
        for f in range(6):
            var off = ren.sota_atmo.ibl.blocks_off() + ibl_mip_block_offset(base, m) + f * nb * nb * 4
            var dec = decode_bc6h_hdr(_read_hist_bytes(ren, off, nb * nb * 16), d, d)
            var ref_tex = List[Vec3](capacity=d * d)
            for y in range(d):
                for x in range(d):
                    ref_tex.append(ibl_prefilter_texel_cpu(lut, SOTA_ATMO_SKY_W, SOTA_ATMO_SKY_H, alt, p, f, x, y, d, m, mips))
            var ref_dec = decode_bc6h_hdr(encode_bc6h_hdr(ref_tex, d, d, partitioned=False), d, d)
            var e_sum = Float64(0.0)
            var c_sum = Float64(0.0)
            var v_sum = Float64(0.0)
            for i in range(d * d):
                var got = dec[i]
                var want = ref_dec[i]
                var fl = ref_tex[i]
                e_sum += Float64(abs(got.x - want.x) + abs(got.y - want.y) + abs(got.z - want.z))
                c_sum += Float64(abs(fl.x - want.x) + abs(fl.y - want.y) + abs(fl.z - want.z))
                v_sum += Float64(fl.x + fl.y + fl.z)
            worst_c = max(worst_c, Float32(e_sum / max(v_sum, 1e-20)))
            worst_codec = max(worst_codec, Float32(c_sum / max(v_sum, 1e-20)))
            if _verbose():
                print("    cube mip", m, "face", f, "GPU vs CPU-encoded", e_sum / max(v_sum, 1e-20), "codec loss", c_sum / max(v_sum, 1e-20))
    _assert(worst_c < 0.02, "GPU BC6H cube vs CPU prefilter + encoder, worst mip/face mean rel err " + String(worst_c))

    # (c) the GPU BC6H encoder is bit-exact with `encode_bc6h_hdr(partitioned=False)` on exactly representable texels.
    ren.sota_atmo.ibl.test_pattern = True
    ren.sota_atmo.ibl_valid = False
    _assert(_frame(ren, cam, tgt, 75.0, -1, Vec3(), Vec3()), "frame")
    var n_blocks = 0
    var bad = 0
    for m in range(mips):
        var d = max(4, base >> m)
        var nb = d // 4
        for f in range(6):
            var tex = List[Vec3](capacity=d * d)
            for y in range(d):
                for x in range(d):
                    tex.append(ibl_test_pattern_texel(f, x, y, d, m))
            var want = encode_bc6h_hdr(tex, d, d, partitioned=False)
            var off = ren.sota_atmo.ibl.blocks_off() + ibl_mip_block_offset(base, m) + f * nb * nb * 4
            var got = _read_hist_bytes(ren, off, nb * nb * 16)
            for bi in range(nb * nb):
                n_blocks += 1
                for k in range(16):
                    if got[bi * 16 + k] != want[bi * 16 + k]:
                        bad += 1
                        break
    _assert(bad == 0, "GPU BC6H blocks bit-exact with the CPU encoder (" + String(bad) + " of " + String(n_blocks) + " differ)")
    ren.sota_atmo.ibl.test_pattern = False

    # (d) time-lapse: the sun moves 0.5 deg a frame. r2 CPU bake (every 1 deg) vs the GPU path (every 0.05 deg).
    var med = List[Float64]()
    var p90 = List[Float64]()
    var bake_ms = Float64(0.0)
    var ibl_gpu_ms = List[Float64]()
    var cube0 = 0
    for mode in range(2):
        ren.sota_atmo.ibl_gpu = mode == 1
        ren.sota_atmo.ibl_valid = False
        cube0 = ren.sota_atmo.ibl.cube_updates
        var ts = List[Float64]()
        for i in range(44):
            var az = Float32(i) * Float32(0.5) * Float32(0.0174532925)
            ren.sota_atmo.sun_dir = Vec3(cos(az) * Float32(0.94), Float32(0.342), sin(az) * Float32(0.94)).normalized()
            var t0 = get_time()
            _assert(_frame(ren, cam, tgt, 75.0, -1, Vec3(), Vec3()), "frame")
            var dt = (get_time() - t0) * 1000.0
            if i < 4:
                continue
            ts.append(dt)
            if mode == 0:
                bake_ms = max(bake_ms, Float64(ren.sota_atmo.last_ibl_ms))
            else:
                for t in ren.get_gpu_pass_timings():
                    if t.pass_name == "SotaAtmoIbl":
                        ibl_gpu_ms.append(t.gpu_ms)
        med.append(_median(ts))
        p90.append(_p90(ts))
    var cube_n = ren.sota_atmo.ibl.cube_updates - cube0
    var ibl_med = _median(ibl_gpu_ms)
    print("    time-lapse CPU frame ms: r2 CPU bake median", med[0], "p90", p90[0], "(bake", bake_ms, "ms); GPU IBL median", med[1],
          "p90", p90[1], "; GPU cube refreshes", cube_n, "/ 44, SotaAtmoIbl median", ibl_med, "ms")
    _assert(cube_n >= 40, "GPU cube refreshes every frame of the time-lapse (" + String(cube_n) + ")")
    _assert(p90[1] < p90[0], "GPU IBL removes the time-lapse hitch (p90 " + String(p90[1]) + " vs " + String(p90[0]) + " ms)")
    _assert(ibl_med < 1.0, "SotaAtmoIbl with a cube refresh within 1 ms (median " + String(ibl_med) + ")")
    ren.shutdown()
    print("  [PASS] Suite 4d: GPU IBL: SH9 vs CPU", worst_sh, ", BC6H cube vs CPU prefilter+encoder", worst_c, "(mode-11 codec loss", worst_codec, "),", n_blocks,
          "test-pattern blocks bit-exact; time-lapse p90", p90[1], "ms vs r2", p90[0], "ms")


def _cloud_setup(mut ren: FlintAaaVulkanRenderer):
    _setup(ren)
    ren.cloud_coverage = 0.0
    ren.sota_atmo.enabled = True
    ren.sota_atmo.clouds.enabled = True
    ren.sota_atmo.clouds.res_div = 1
    ren.sota_atmo.clouds.temporal_alpha = 1.0
    ren.sota_atmo.clouds.jitter = False
    ren.sota_atmo.clouds.wind_mps = Vec3(0.0, 0.0, 0.0)
    ren.sota_atmo.clouds.params.steps = 32
    ren.sota_atmo.sun_dir = Vec3(0.4, 0.5, 0.6).normalized()


def _cloud_img(ren: FlintAaaVulkanRenderer) -> List[Float32]:
    return _read_hist(ren, ren.sota_atmo.cloud_cur_off(), ren.sota_atmo.cloud_cw * ren.sota_atmo.cloud_ch * 4)


def _cloud_t_err(a: List[Float32], b: List[Float32]) -> Float32:
    """Mean |delta transmittance| over the cloud pixels both images march (sentinel 2 = no sky)."""
    var e = Float64(0.0)
    var n = 0
    for i in range(len(a) // 4):
        if a[i * 4 + 3] < Float32(1.5) and b[i * 4 + 3] < Float32(1.5):
            e += Float64(abs(a[i * 4 + 3] - b[i * 4 + 3]))
            n += 1
    return Float32(e / Float64(max(1, n)))


def _cloud_cam_tgt(yaw_deg: Float32) -> Vec3:
    var a = yaw_deg * Float32(0.0174532925)
    return Vec3(0.0, 2.0, 0.0) + Vec3(sin(a), Float32(0.32), cos(a)) * Float32(100.0)


def test_gpu_clouds():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _cloud_setup(ren)
    ren.sota_atmo.clouds.aerial_perspective = False
    var cam = Vec3(0.0, 2.0, 0.0)
    for _ in range(3):
        _assert(_frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3()), "frame")
    _assert(ren.sota_atmo.clouds_active(), "clouds active")
    # (a) noise volumes + weather map vs the CPU twins.
    var nb = SOTA_CLOUD_BASE_N
    var nd = SOTA_CLOUD_DETAIL_N
    var nw = SOTA_CLOUD_WEATHER_N
    var base = _read_hist(ren, ren.sota_atmo.cloud_base_off(), nb * nb * nb)
    var det = _read_hist(ren, ren.sota_atmo.cloud_detail_off(), nd * nd * nd)
    var wea = _read_hist(ren, ren.sota_atmo.cloud_weather_off(), nw * nw * 2)
    var worst_n = Float32(0.0)
    for i in range(0, nb * nb * nb, 13):
        worst_n = max(worst_n, abs(base[i] - cloud_base_texel_cpu(i % nb, (i // nb) % nb, i // (nb * nb))))
    for i in range(0, nd * nd * nd, 7):
        worst_n = max(worst_n, abs(det[i] - cloud_detail_texel_cpu(i % nd, (i // nd) % nd, i // (nd * nd))))
    for i in range(0, nw * nw, 5):
        var c = cloud_weather_texel_cpu(i % nw, i // nw, ren.sota_atmo.clouds.weather_seed)
        worst_n = max(worst_n, max(abs(wea[i * 2] - c[0]), abs(wea[i * 2 + 1] - c[1])))
    _assert(worst_n < 2e-3, "cloud noise + weather map vs CPU twins (" + String(worst_n) + ")")

    # (b) the march (no history, no jitter, no aerial perspective) vs `cloud_march_cpu`.
    var img = _cloud_img(ren)
    var nz = CloudNoiseCpu(base^, det^, wea^)
    var cp = ren.sota_atmo.clouds.params
    var p = ren.sota_atmo.params
    var luts = SotaAtmoLuts(p)
    var lights = ren.sota_atmo.lights
    var alt = ren.sota_atmo.camera_altitude_km(cam.y)
    var kpu = ren.sota_atmo.km_per_unit
    var sh = ren.sota_atmo.read_gpu_sh9(ren.buf_history.cpu_address)
    var up = eval_sh9_cpu(sh, Vec3(0.0, 1.0, 0.0))
    var dn = eval_sh9_cpu(sh, Vec3(0.0, -1.0, 0.0))
    var cw = ren.sota_atmo.cloud_cw
    var ch = ren.sota_atmo.cloud_ch
    var dv = ren.sota_atmo.cloud_div_used
    var fw = Float32(w)
    var fh = Float32(h)
    var tyf = sin(ren.fov_y_deg * Float32(0.00872665)) / cos(ren.fov_y_deg * Float32(0.00872665))
    var txf = tyf * fw / fh
    var e_t = Float64(0.0)
    var e_s = Float64(0.0)
    var v_s = Float64(0.0)
    var n_px = 0
    var n_cloud = 0
    for cy in range(0, ch, 3):
        for cx in range(0, cw, 3):
            var nx = (Float32(cx) + Float32(0.5)) * Float32(dv) / fw * Float32(2.0) - Float32(1.0)
            var ny = Float32(1.0) - (Float32(cy) + Float32(0.5)) * Float32(dv) / fh * Float32(2.0)
            var d = (ren.cam_fwd + ren.cam_right * (nx * txf) + ren.cam_up * (ny * tyf)).normalized()
            if d.y < Float32(0.03):
                continue
            var s = cloud_march_cpu(nz, cp, luts, cam.x * kpu, cam.z * kpu, p.r_bottom_km + alt, d, 0.5, lights.sun_dir, lights.sun_e,
                                    lights.moon_dir, lights.moon_e, RGB(up.x, up.y, up.z, 0.0), RGB(dn.x, dn.y, dn.z, 0.0))
            var o = (cy * cw + cx) * 4
            e_t += Float64(abs(img[o + 3] - s.trans))
            var gl = rgb_luma(RGB(img[o], img[o + 1], img[o + 2], 0.0))
            e_s += Float64(abs(gl - rgb_luma(s.rgb)))
            v_s += Float64(rgb_luma(s.rgb))
            n_px += 1
            if s.trans < Float32(0.9):
                n_cloud += 1
    var err_t = Float32(e_t / Float64(max(1, n_px)))
    var err_s = Float32(e_s / max(v_s, 1e-20))
    _assert(n_cloud > n_px // 10, "the parity view sees clouds (" + String(n_cloud) + " of " + String(n_px) + ")")
    _assert(err_t < 0.01, "cloud transmittance vs CPU march, mean abs err " + String(err_t))
    _assert(err_s < 0.03, "cloud in-scattering vs CPU march, mean rel err " + String(err_s))

    # (c) coverage 0 == clouds off (same frame, LDR).
    ren.sota_atmo.clouds.aerial_perspective = True
    ren.sota_atmo.clouds.params.coverage = 0.0
    for _ in range(2):
        _assert(_frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var a0 = ren.readback_pixels()
    ren.sota_atmo.clouds.enabled = False
    for _ in range(2):
        _assert(_frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3()), "frame")
    var b0 = ren.readback_pixels()
    var max_d = 0
    for i in range(len(a0)):
        max_d = max(max_d, abs(Int(a0[i]) - Int(b0[i])))
    _assert(max_d <= 1, "coverage 0 leaves the sky unchanged (max LDR diff " + String(max_d) + ")")
    ren.sota_atmo.clouds.enabled = True
    ren.sota_atmo.clouds.params.coverage = 0.45

    # (d) temporal accumulation: jittered 24-step marches, alpha 1 vs 0.15, against a 128-step reference.
    ren.sota_atmo.clouds.params.steps = 128
    for _ in range(2):
        _ = _frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3())
    var ref_img = _cloud_img(ren)
    ren.sota_atmo.clouds.params.steps = 24
    ren.sota_atmo.clouds.jitter = True
    var noise = List[Float32]()
    var bias = List[Float32]()
    for mode in range(2):
        ren.sota_atmo.clouds.temporal_alpha = Float32(1.0) if mode == 0 else Float32(0.15)
        var prev = _cloud_img(ren)
        var fl = Float64(0.0)
        var nf = 0
        for i in range(24):
            _ = _frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3())
            var cur = _cloud_img(ren)
            if i >= 16:
                fl += Float64(_cloud_t_err(cur, prev))
                nf += 1
            prev = cur^
        noise.append(Float32(fl / Float64(nf)))
        bias.append(_cloud_t_err(prev, ref_img))
    _assert(noise[1] < Float32(0.4) * noise[0], "history cuts the frame-to-frame flicker (" + String(noise[1]) + " vs " + String(noise[0]) + ")")
    _assert(bias[1] < bias[0], "accumulated 24-step clouds closer to the 128-step reference than one frame (" + String(bias[1]) + " vs " + String(bias[0]) + ")")

    # (e) reprojection: yaw 1.5 deg / frame, then wind 0.2 km / frame: the accumulated image tracks a fresh march.
    ren.sota_atmo.clouds.jitter = False
    ren.sota_atmo.clouds.params.steps = 32
    ren.sota_atmo.clouds.temporal_alpha = 0.15
    for i in range(20):
        _ = _frame(ren, cam, _cloud_cam_tgt(10.0 + Float32(i) * 1.5), 60.0, -1, Vec3(), Vec3())
    var acc_y = _cloud_img(ren)
    ren.sota_atmo.clouds.temporal_alpha = 1.0
    _ = _frame(ren, cam, _cloud_cam_tgt(10.0 + 19.0 * 1.5), 60.0, -1, Vec3(), Vec3())
    var ref_y = _cloud_img(ren)
    _ = _frame(ren, cam, _cloud_cam_tgt(10.0 + 13.0 * 1.5), 60.0, -1, Vec3(), Vec3())
    var stale_y = _cloud_img(ren)
    var e_yaw = _cloud_t_err(acc_y, ref_y)
    var e_yaw_stale = _cloud_t_err(stale_y, ref_y)
    _assert(e_yaw < Float32(0.35) * e_yaw_stale, "yaw reprojection (" + String(e_yaw) + " vs 6-frame-stale " + String(e_yaw_stale) + ")")
    ren.sota_atmo.clouds.wind_mps = Vec3(12000.0, 0.0, 4000.0)
    ren.sota_atmo.clouds.temporal_alpha = 0.15
    ren.scene_time_sec = 0.0
    for _ in range(20):
        _ = _frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3())
    var acc_w = _cloud_img(ren)
    ren.sota_atmo.clouds.temporal_alpha = 1.0
    ren.scene_time_sec = Float32(19.0 / 60.0)
    _ = _frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3())
    var ref_w = _cloud_img(ren)
    ren.scene_time_sec = Float32(13.0 / 60.0)
    _ = _frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3())
    var stale_w = _cloud_img(ren)
    var e_wind = _cloud_t_err(acc_w, ref_w)
    var e_wind_stale = _cloud_t_err(stale_w, ref_w)
    _assert(e_wind < Float32(0.35) * e_wind_stale, "wind reprojection (" + String(e_wind) + " vs 6-frame-stale " + String(e_wind_stale) + ")")
    ren.sota_atmo.clouds.wind_mps = Vec3(0.0, 0.0, 0.0)

    # (f) lighting: sunset clouds redden; a full moon where the noon sun was gives the noon clouds scaled by E_moon / E_sun
    # (transport is linear: the moon lights the sky LUTs, hence the cloud ambient, the same way the sun does).
    var stats = List[Tuple[Float32, Float32]]()
    var dirs = [Vec3(0.2, 0.75, 0.6).normalized(), Vec3(0.17, 0.035, 0.985).normalized()]
    for li in range(3):
        if li < 2:
            ren.sota_atmo.sun_dir = dirs[li]
            ren.sota_atmo.moon_enabled = False
        else:
            ren.sota_atmo.sun_dir = dirs[0] * Float32(-1.0)
            ren.sota_atmo.moon_enabled = True
            ren.sota_atmo.moon_dir = dirs[0]
        for _ in range(3):
            _ = _frame(ren, cam, _cloud_cam_tgt(10.0), 60.0, -1, Vec3(), Vec3())
        var im = _cloud_img(ren)
        var l = Float64(0.0)
        var rr = Float64(0.0)
        var bb = Float64(0.0)
        for i in range(len(im) // 4):
            if im[i * 4 + 3] < Float32(0.5):
                l += Float64(rgb_luma(RGB(im[i * 4], im[i * 4 + 1], im[i * 4 + 2], 0.0)))
                rr += Float64(im[i * 4])
                bb += Float64(im[i * 4 + 2])
        var e_l = rgb_luma(ren.sota_atmo.lights.moon_e if li == 2 else ren.sota_atmo.lights.sun_e)
        if _verbose():
            print("    lighting case", li, "cloud luma sum", l, "E", e_l, "sun", ren.sota_atmo.lights.sun_dir.y, "moon", ren.sota_atmo.lights.moon_dir.y,
                  "frame", ren.sota_atmo.clouds.frame)
        stats.append((Float32(l) / e_l, Float32(rr / max(bb, 1e-30))))
    var moon_ratio = stats[2][0] / stats[0][0]
    _assert(stats[1][1] > Float32(1.5) * stats[0][1], "sunset clouds redden (R/B " + String(stats[1][1]) + " vs noon " + String(stats[0][1]) + ")")
    _assert(abs(moon_ratio - Float32(1.0)) < Float32(0.02), "moonlit clouds = noon clouds scaled by E_moon / E_sun (" + String(moon_ratio) + ")")
    ren.shutdown()
    print("  [PASS] Suite 4e: clouds: noise vs CPU", worst_n, "; march vs CPU: T", err_t, ", S", err_s, "(", n_px, "px );",
          "coverage 0 = off; flicker", noise[1], "vs", noise[0], ", ref err", bias[1], "vs", bias[0], "; reprojection yaw", e_yaw, "/",
          e_yaw_stale, ", wind", e_wind, "/", e_wind_stale, "; R/B noon", stats[0][1], "sunset", stats[1][1], "; moon/noon", moon_ratio)


def test_gpu_budgets_r3():
    var w = 1280
    var h = 720
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.clouds.enabled = True
    var m = ren.add_material(AaaMaterial(base_color=Vec3(0.3, 0.3, 0.3), roughness=0.8, metallic=0.0))
    var wall = ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5)), m).mesh_id
    var res = List[Float64]()
    for case_i in range(2):
        if case_i == 0:
            ren.sota_atmo.use_time_of_day = False
            ren.sota_atmo.sun_dir = Vec3(0.3, 0.6, 0.7).normalized()
        else:
            ren.sota_atmo.use_time_of_day = True
            ren.sota_atmo.time_of_day = SotaTimeOfDay(2026, 6, 30, 1.0, 48.85, 2.35)  # moonlit night
        var cl = List[Float64]()
        var ib = List[Float64]()
        var ap = List[Float64]()
        for i in range(40):
            _ = _frame(ren, Vec3(0.0, 2.0, 0.0), Vec3(3.0, 12.0, 20.0), 70.0, wall, Vec3(0.0, 0.0, 400.0), Vec3(800.0, 60.0, 10.0))
            if i < 8:
                continue
            for t in ren.get_gpu_pass_timings():
                if t.pass_name == "SotaAtmoClouds":
                    cl.append(t.gpu_ms)
                elif t.pass_name == "SotaAtmoIbl":
                    ib.append(t.gpu_ms)
                elif t.pass_name == "SotaAtmoApply":
                    ap.append(t.gpu_ms)
        _assert(len(cl) > 0 and len(ib) > 0 and len(ap) > 0, "round-3 GPU timings recorded")
        res.append(_median(cl))
        res.append(_median(ib))
        res.append(_median(ap))
    print("    day: SotaAtmoClouds", res[0], "ms, SotaAtmoIbl (SH only)", res[1], "ms, SotaAtmoApply", res[2], "ms; moonlit night:", res[3], res[4], res[5])
    _assert(res[0] < 1.0 and res[3] < 1.0, "clouds within 1 ms at 720p (half-res image)")
    _assert(res[1] < 0.1 and res[4] < 0.1, "per-frame SH9 within 0.1 ms")
    _assert(res[2] < 0.6 and res[5] < 0.6, "apply (night sky + composite) within 0.6 ms")
    ren.shutdown()
    print("  [PASS] Suite 4f: round-3 budgets at 1280x720, medians: clouds", res[0], "/", res[3], "ms, SH9", res[1], "ms, apply", res[2], "/", res[5], "ms")


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
    if getenv("FLINT_SOTA_ATMO_R3_ONLY") != "1":  # dev shortcut: only the round-3 suites
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
    test_spirv_r3()
    test_cpu_stars()
    test_gpu_night_sky()
    test_gpu_ibl()
    test_gpu_clouds()
    test_gpu_budgets_r3()
    print("=== ALL sota:atmo TESTS PASSED ===")
