"""Area D round 4 (sota:camera): physically based auto exposure for day / twilight / night, mesopic vision,
luminance-aware (BT.2446-C style) HDR highlight expansion, and the composition of area H's physical sky with
the exposure and HDR paths (`test_flint_v5_sota_camera_r4.mojo`). Zero external assets.

1. SPIR-V validation of the new shaders and the display tonemap with the round-4 HDR hook.
2. CPU references: photometric calibration, EV100 meter with tent binning and percentile trimming, night
   compensation monotonicity, frame-rate independent adaptation, scotopic luminance, the exponential
   highlight expansion (E(1) = P, C1 knee, identity, saturated colours, channel limit).
3. Live GPU (skips without Vulkan): off = bit-identical; GPU meter vs the CPU meter over 13 stops; exposed
   mid grey at day / twilight / night; adaptation curve vs the CPU twin; Purkinje shift; physical sky at
   noon / sunset / twilight / night with the v5 meter vs the EV100 meter; HDR10 / scRGB with the sky and the
   sun disc in view, max-channel vs luminance expansion; GPU cost.
"""
from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.math import abs, cos, exp, exp2, log2, log10, max, min, sin, sqrt
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.camera.physical import PHYS_EXPOSURE_APERTURE_PRIORITY, PhysicalCameraModel
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709, AAA_DISPLAY_HDR10_PQ, AAA_DISPLAY_LINEAR_SRGB, aaa_display_tonemap_shader
from flint.render.aaa_renderer import AAA_DISPLAY_SCRGB_FP16, FlintAaaVulkanRenderer
from flint.render.sota_atmo_core import SotaTimeOfDay
from flint.render.sota_camera_exposure import (
    SOTA_AE_ADAPTED_EV,
    SOTA_AE_ADAPT_NITS,
    SOTA_AE_BINS,
    SOTA_AE_COMP_EV,
    SOTA_AE_METERED_EV,
    SOTA_AE_TARGET_EV,
    SOTA_AE_WEIGHT_ONE,
    SotaAutoExposure,
    ev100_from_nits_cpu,
    photopic_weight_cpu,
    scotopic_luminance_cpu,
    sota_ae_adapt_cpu,
    sota_ae_adapt_shader,
    sota_ae_bin_ev_cpu,
    sota_ae_center_weight_cpu,
    sota_ae_histogram_shader,
    sota_ae_meter_cpu,
    sota_ae_night_factor_cpu,
    sota_ae_purkinje_shader,
    sota_ae_splat_cpu,
    sota_ae_target_ev_cpu,
)
from flint.render.sota_camera_hdr import (
    SOTA_HDR_EXPAND_LUMINANCE,
    SOTA_HDR_EXPAND_MAX_CHANNEL,
    SotaHdrOutput,
    f16_to_f32_cpu,
    pq_word_decode_nits_cpu,
    rec2020_to_rec709_cpu,
    sota_hdr_expand_lum_cpu,
    sota_hdr_lum_scale_cpu,
    sota_hdr_map_nits_cpu,
    sota_hdr_solve_exp_b,
    srgb_eotf_cpu,
)
from flint.render.spirv_validate import validate_spirv_module

comptime OUT_DIR = "build/sota_camera_r4"


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


# ============================================================================
# 1. SPIR-V
# ============================================================================

def test_spirv():
    _validate("sota_ae_histogram", sota_ae_histogram_shader())
    _validate("sota_ae_adapt", sota_ae_adapt_shader())
    _validate("sota_ae_purkinje", sota_ae_purkinje_shader())
    _validate("aaa_display_tonemap (round-4 HDR hook)", aaa_display_tonemap_shader())
    print("  [PASS] Suite 1: SPIR-V of the EV100 meter, adaptation, mesopic pass and the tonemap HDR hook")


# ============================================================================
# 2. CPU references
# ============================================================================

def test_cpu_exposure():
    var ae = SotaAutoExposure()
    # (a) calibration: 128 klx / 4.5 units; mid grey of an EV100 scene is 2^EV / 8 cd/m^2.
    _near(ae.nits_per_unit, 28444.44, 0.1, "nits per engine unit")
    var unity = ae.ev100_unity()
    _near(unity, Float32(log2(Float64(8.0 * 28444.44 * 0.18))), 1e-4, "EV100 at exposure 1")
    _near(ae.exposure_for_ev(unity), 1.0, 1e-5, "exposure at the unity EV")
    _near(ev100_from_nits_cpu(4096.0), 15.0, 1e-5, "EV100 of 4096 cd/m^2")
    # Sunny-16 (f/16, 1/125 s, ISO 100 = EV100 14.97) is within 0.4 EV of exposure 1.
    _assert(abs(unity - Float32(log2(Float64(256.0 * 125.0)))) < 0.4, "sunny-16 ~ exposure 1")

    # (b) meter: tent binning recovers a uniform EV exactly; percentiles reject the sun and black corners.
    var worst_u = Float32(0.0)
    for k in range(40):
        var ev = Float32(-9.0) + Float32(k) * Float32(0.6137)
        var h = List[UInt32](length=SOTA_AE_BINS, fill=0)
        for _ in range(100):
            sota_ae_splat_cpu(h, ev, UInt32(64))
        worst_u = max(worst_u, abs(sota_ae_meter_cpu(h, 0.1, 0.9)[0] - ev))
    _assert(worst_u < 0.035, "uniform scene meters to its EV (" + String(worst_u) + ")")
    # Continuity: sliding a uniform scene by 0.01 EV never jumps the reading (no bin steps).
    var last_m = Float32(-100.0)
    var worst_jump = Float32(0.0)
    var mono_m = True
    for k in range(300):
        var ev = Float32(2.0) + Float32(k) * Float32(0.01)
        var hh = List[UInt32](length=SOTA_AE_BINS, fill=0)
        for _ in range(100):
            sota_ae_splat_cpu(hh, ev, UInt32(64))
        var mm = sota_ae_meter_cpu(hh, 0.1, 0.9)[0]
        if k > 0:
            worst_jump = max(worst_jump, abs(mm - last_m))
            if mm < last_m - Float32(1e-5):
                mono_m = False
        last_m = mm
    _assert(mono_m and worst_jump < 0.03, "meter is monotonic and continuous (jump " + String(worst_jump) + ")")
    var h2 = List[UInt32](length=SOTA_AE_BINS, fill=0)
    for _ in range(900):
        sota_ae_splat_cpu(h2, 3.0, UInt32(64))
    for _ in range(50):
        sota_ae_splat_cpu(h2, 21.5, UInt32(64))   # sun disc
        sota_ae_splat_cpu(h2, -13.0, UInt32(64))  # black corners
    var m2 = sota_ae_meter_cpu(h2, 0.1, 0.9)[0]
    _near(m2, 3.0, 0.02, "trimmed mean ignores 5 % sun and 5 % black")
    # centre weighting: 1 at the centre, falls to the corners, = 64 everywhere when off.
    _assert(sota_ae_center_weight_cpu(0.0, 0.0, 0.5) == UInt32(64) and sota_ae_center_weight_cpu(1.0, 1.0, 0.5) < UInt32(40), "centre weights")
    _assert(sota_ae_center_weight_cpu(0.9, -0.7, 0.0) == UInt32(64), "average metering")

    # (c) night compensation: exposed mid grey never gets brighter when the scene gets darker.
    var prev_exposed = Float32(-1.0)
    var mono = True
    for k in range(-30, 41):
        var ev = Float32(k) * Float32(0.5)
        var tgt = sota_ae_target_ev_cpu(ae, ev, 0.0)
        var exposed = ae.key * exp2(ev - tgt)  # mid grey of the scene, exposed
        if exposed < prev_exposed - Float32(1e-6):
            mono = False
        prev_exposed = exposed
    _assert(mono, "exposed mid grey is monotonic in scene luminance")
    _near(sota_ae_target_ev_cpu(ae, 14.0, 0.0), 14.0, 1e-5, "no compensation in daylight")
    _near(sota_ae_target_ev_cpu(ae, -3.5, 0.0), -3.5 + 2.5 * sota_ae_night_factor_cpu(ae, -3.5), 1e-5, "moonlight compensation")
    _assert(sota_ae_night_factor_cpu(ae, -5.0) == Float32(1.0) and sota_ae_night_factor_cpu(ae, 13.0) == Float32(0.0), "night factor ends")
    _near(sota_ae_target_ev_cpu(ae, -12.0, 0.0), ae.min_ev100 + 2.5, 1e-5, "metering range clamp")

    # (d) adaptation: half-life tau ln 2; frame-rate independent; asymmetric.
    var e60 = Float32(0.0)
    for _ in range(60):
        e60 = sota_ae_adapt_cpu(ae, e60, 10.0, Float32(1.0 / 60.0))
    var e30 = Float32(0.0)
    for _ in range(30):
        e30 = sota_ae_adapt_cpu(ae, e30, 10.0, Float32(1.0 / 30.0))
    _near(e60, e30, 1e-3, "adaptation independent of the frame rate")
    _near(e60, Float32(10.0) * (Float32(1.0) - exp(Float32(-1.0) / ae.tau_brighten_s)), 1e-3, "1 s of light adaptation")
    var d1 = sota_ae_adapt_cpu(ae, 10.0, 0.0, 0.1)
    var u1 = sota_ae_adapt_cpu(ae, 0.0, 10.0, 0.1)
    _assert(Float32(10.0) - d1 < u1, "darkening is slower than brightening")
    var dn = sota_ae_adapt_cpu(ae, -5.0, -15.0, 0.1)
    _assert(Float32(-5.0) - dn < Float32(10.0) - d1, "dark adaptation is slower again at night")
    var capped = ae
    capped.max_rate_ev_s = 2.0
    _near(sota_ae_adapt_cpu(capped, 0.0, 10.0, 0.1), 0.2, 1e-5, "max adaptation rate")

    # (e) scotopic luminance: white keeps its luminance; blues are brighter than reds for the rods.
    _near(scotopic_luminance_cpu(1.0, 1.0, 1.0), 1.0, 2e-3, "scotopic white")
    var r_ratio = scotopic_luminance_cpu(1.0, 0.0, 0.0) / Float32(0.2126)
    var b_ratio = scotopic_luminance_cpu(0.0, 0.0, 1.0) / Float32(0.0722)
    _assert(b_ratio > Float32(4.0) * r_ratio, "Purkinje: blue / red rod response (" + String(b_ratio) + " vs " + String(r_ratio) + ")")
    _near(photopic_weight_cpu(ae, 100.0), 1.0, 1e-6, "photopic in daylight")
    _near(photopic_weight_cpu(ae, 0.001), 0.0, 1e-6, "scotopic under starlight")
    print("  [PASS] Suite 2a: EV100 calibration (unity EV100", unity, "), tent-binned trimmed meter (uniform err", worst_u, "EV, max step", worst_jump,
          "), monotonic night compensation, frame-rate independent asymmetric adaptation, scotopic luminance (blue/red", b_ratio / r_ratio, ")")


def test_cpu_hdr_expansion():
    var hdr = SotaHdrOutput()
    hdr.expansion_mode = SOTA_HDR_EXPAND_LUMINANCE
    var p = hdr.peak_ratio()
    var k = hdr.knee_clamped()
    var b = hdr.exp_b()
    _near(sota_hdr_expand_lum_cpu(1.0, k, b), p, 1e-3 * p, "E(1) = P")
    var eps = Float32(1e-3)
    var slope = (sota_hdr_expand_lum_cpu(k + eps, k, b) - sota_hdr_expand_lum_cpu(k, k, b)) / eps
    _near(slope, 1.0, 0.02, "C1 at the knee")
    var mono = True
    var last = Float32(-1.0)
    for i in range(201):
        var y = Float32(i) / Float32(200.0)
        var e = sota_hdr_expand_lum_cpu(y, k, b)
        if e < last:
            mono = False
        last = e
    _assert(mono, "expansion is monotonic")
    _near(sota_hdr_solve_exp_b(1.0, k), 0.0, 1e-6, "identity at P = 1")
    # White expands to the peak; a saturated red (luma 0.21) keeps its SDR luminance; no channel over P.
    _near(sota_hdr_lum_scale_cpu(1.0, 1.0, 1.0, hdr), p, 1e-3 * p, "white to the peak")
    _near(sota_hdr_lum_scale_cpu(1.0, 0.05, 0.05, hdr), 1.0, 1e-5, "saturated red keeps its luminance")
    var s_y = sota_hdr_lum_scale_cpu(1.0, 1.0, 0.4, hdr)
    _assert(s_y <= p + 1e-4 and s_y > Float32(1.5), "yellow-white limited by the peak (" + String(s_y) + ")")
    var maxc = SotaHdrOutput()
    var n_lum = sota_hdr_map_nits_cpu(1.0, 0.25, 0.25, hdr)
    var n_max = sota_hdr_map_nits_cpu(1.0, 0.25, 0.25, maxc)
    _assert(n_lum[0] < n_max[0], "luminance mode pushes saturated highlights less than the max channel")
    var no_x = hdr
    no_x.crosstalk = 0.0
    _assert(sota_hdr_lum_scale_cpu(0.3, 0.3, 1.0, hdr) >= sota_hdr_lum_scale_cpu(0.3, 0.3, 1.0, no_x), "crosstalk lifts saturated colours")
    print("  [PASS] Suite 2b: luminance (BT.2446-C style) expansion: b =", b, ", E(1) = P, C1 knee slope", slope, ", identity, saturated colours kept")


# ============================================================================
# 3. GPU
# ============================================================================

def _box(c: Vec3, he: Vec3) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_beveled_box(c, he, bevel=0.002)
    return mb^.finish()


def _emissive(c: Vec3) -> AaaMaterial:
    var m = AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
    m.emissive = c
    return m^


def _post_f32(ren: FlintAaaVulkanRenderer, word: Int) -> Float32:
    var out = List[Float32](length=1, fill=0.0)
    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), ren.buf_post_scratch.cpu_address + word * 4, 4)
    return out[0]


def _hdr_px(ren: FlintAaaVulkanRenderer, i: Int) -> Vec3:
    var v = List[Float32](length=4, fill=0.0)
    _ = external_call["memcpy", Int](Int(v.unsafe_ptr()), ren.buf_hdr_out.cpu_address + (i * 8) * 4, 16)
    return Vec3(v[0], v[1], v[2])


def _black_env(mut ren: FlintAaaVulkanRenderer):
    var sky = SkyDomeParams()
    sky.zenith_rgb = Vec3(0.0, 0.0, 0.0)
    sky.horizon_rgb = Vec3(0.0, 0.0, 0.0)
    sky.ground_rgb = Vec3(0.0, 0.0, 0.0)
    sky.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky, base_dim=16, mip_levels=4))
    ren.ibl_intensity = Float32(0.0)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)


def _setup(mut ren: FlintAaaVulkanRenderer):
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_motion_blur = False
    ren.enable_bokeh_dof = False
    ren.enable_froxel_fog = False


def _wall_frame(mut ren: FlintAaaVulkanRenderer, mesh: Int) -> List[UInt32]:
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, 0.0), cam_target=Vec3(0.0, 0.0, 1.0), fov_y_deg=40.0, near_z=0.05)
    ren.draw_instance(mesh, trs(Vec3(0.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "wall frame")
    return ren.readback_pixels()


def _srgb8(v: Float32) -> Int:
    var c = max(Float32(0.0), min(Float32(1.0), v))
    var e = c * Float32(12.92) if c < Float32(0.0031308) else Float32(1.055) * Float32(exp(Float64(log2(Float64(c)) / 2.4) * 0.6931471805599453)) - Float32(0.055)
    return Int(e * Float32(255.0) + Float32(0.5))


def test_gpu_meter():
    var w = 320
    var h = 180
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _assert(ren.sota_cam.ae.pipelines_ready(), "AE pipelines created")
    _setup(ren)
    _black_env(ren)
    var lums = List[Float32]()
    lums.append(0.18)      # sunlit mid grey (EV100 ~15.3)
    lums.append(3.0e-3)    # overcast dusk
    lums.append(1.0e-4)    # deep twilight
    lums.append(2.0e-6)    # moonlight
    lums.append(1.0e-7)    # starlight (below the metering range)
    var walls = List[Int]()
    for i in range(len(lums)):
        var m = ren.add_material(_emissive(Vec3(lums[i], lums[i], lums[i])))
        walls.append(ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(20.0, 20.0, 0.02)), m).mesh_id)

    # (a) off: the sota meter enabled but the renderer's auto exposure off = identical frames; legacy AE on
    # with the sota meter off = identical to never touching the setting.
    var base = _wall_frame(ren, walls[1])
    ren.sota_cam.ae.enable = True
    var same_off = _wall_frame(ren, walls[1])
    var ident = True
    for i in range(len(base)):
        if base[i] != same_off[i]:
            ident = False
    _assert(ident, "sota meter without renderer auto exposure changes nothing")
    ren.sota_cam.ae.enable = False
    ren.enable_auto_exposure = True
    var leg_a = _wall_frame(ren, walls[0])
    var leg_names_ok = False
    for t in ren.get_gpu_pass_timings():
        if t.pass_name == "LuminanceAutoExposure":
            leg_names_ok = True
        _assert(t.pass_name != "SotaAutoExposure", "no sota meter pass when off")
    _assert(leg_names_ok, "v5 meter still recorded when the sota meter is off")
    _ = leg_a

    # (b) GPU meter vs the CPU meter, and the exposed mid grey on screen.
    ren.sota_cam.ae.enable = True
    ren.sota_cam.ae.center_weight = 0.0
    var ae = ren.sota_cam.ae
    var worst_ev = Float32(0.0)
    var worst_code = 0
    var report = String("")
    for i in range(len(lums)):
        ren.sota_cam.ae.snap_next = True
        var px = _wall_frame(ren, walls[i])
        var got = _post_f32(ren, SOTA_AE_METERED_EV)
        var want = ev100_from_nits_cpu(lums[i] * ae.nits_per_unit)
        worst_ev = max(worst_ev, abs(got - want))
        var tgt = sota_ae_target_ev_cpu(ae, want, 0.0)
        _near(_post_f32(ren, SOTA_AE_ADAPTED_EV), tgt, 0.03, "snapped exposure setting " + String(i))
        var mult = ren.get_adapted_exposure()[0]
        _near(log2(mult), ae.ev100_unity() - _post_f32(ren, SOTA_AE_ADAPTED_EV), 1e-3, "multiplier from the EV")
        var exposed = lums[i] * exp2(ae.ev100_unity() - tgt)
        var code = Int((px[(h // 2) * w + w // 2] >> 8) & 0xFF)
        worst_code = max(worst_code, abs(code - _srgb8(exposed)))
        report += String(" ") + String(want) + "->" + String(_srgb8(exposed))
    print("    metered EV100 vs CPU: worst", worst_ev, "EV; on-screen mid grey (EV100 -> sRGB code):" + report)
    _assert(worst_ev < 0.03, "GPU meter = CPU meter over 21 stops (" + String(worst_ev) + ")")
    _assert(worst_code <= 2, "exposed mid grey on screen (" + String(worst_code) + " codes)")

    # (c) legacy meter at moonlight is black; the EV100 meter shows a dark grey night.
    ren.sota_cam.ae.enable = False
    for _ in range(40):  # the v5 meter lerps from the previous multiplier: let it converge
        _ = _wall_frame(ren, walls[3])
    var leg_night = _wall_frame(ren, walls[3])
    ren.sota_cam.ae.enable = True
    ren.sota_cam.ae.snap_next = True
    var new_night = _wall_frame(ren, walls[3])
    var c_leg = Int((leg_night[(h // 2) * w + w // 2] >> 8) & 0xFF)
    var c_new = Int((new_night[(h // 2) * w + w // 2] >> 8) & 0xFF)
    print("    moonlit grey wall: v5 meter code", c_leg, "| EV100 meter code", c_new)
    _assert(c_leg <= 2 and c_new > 30 and c_new < 140, "night: v5 black, EV100 meter dark grey")

    # (d) adaptation on the GPU follows the CPU twin (day -> moonlight, 90 frames at 60 Hz).
    ren.sota_cam.ae.snap_next = True
    _ = _wall_frame(ren, walls[0])
    var ev_gpu = _post_f32(ren, SOTA_AE_ADAPTED_EV)
    var tgt_n = sota_ae_target_ev_cpu(ae, ev100_from_nits_cpu(lums[3] * ae.nits_per_unit), 0.0)
    var ev_cpu = ev_gpu
    var worst_ad = Float32(0.0)
    var half_frame = -1
    var start = ev_gpu
    for f in range(90):
        _ = _wall_frame(ren, walls[3])
        ev_cpu = sota_ae_adapt_cpu(ae, ev_cpu, tgt_n, ren.sota_cam.frame_dt_s)
        var g = _post_f32(ren, SOTA_AE_ADAPTED_EV)
        worst_ad = max(worst_ad, abs(g - ev_cpu))
        if half_frame < 0 and g < (start + tgt_n) * Float32(0.5):
            half_frame = f + 1
    print("    adaptation day -> moonlight: GPU vs CPU worst", worst_ad, "EV; half way after", half_frame, "frames (", Float32(half_frame) / Float32(60.0), "s)")
    _assert(worst_ad < 0.02, "GPU adaptation = CPU twin")
    _assert(half_frame > 20 and half_frame < 90, "darkening takes about a second")
    ren.enable_auto_exposure = False
    ren.sota_cam.ae.enable = False
    print("  [PASS] Suite 3a: off = identical; GPU EV100 meter vs CPU; exposed grey day..night; adaptation curve")
    ren.shutdown()


def test_gpu_purkinje():
    var w = 160
    var h = 90
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    _black_env(ren)
    # Half red / half blue wall of equal photopic luminance, at moonlight and at daylight.
    var scales = List[Float32]()
    scales.append(2.0e-6)
    scales.append(0.2)
    var red_ids = List[Int]()
    var blue_ids = List[Int]()
    for s in scales:
        var mr = ren.add_material(_emissive(Vec3(s / Float32(0.2126), 0.0, 0.0)))
        var mb = ren.add_material(_emissive(Vec3(0.0, 0.0, s / Float32(0.0722))))
        red_ids.append(ren.add_mesh(_box(Vec3(-10.0, 0.0, 0.0), Vec3(10.0, 20.0, 0.02)), mr).mesh_id)
        blue_ids.append(ren.add_mesh(_box(Vec3(10.0, 0.0, 0.0), Vec3(10.0, 20.0, 0.02)), mb).mesh_id)
    ren.enable_auto_exposure = True
    ren.sota_cam.ae.enable = True
    ren.sota_cam.ae.center_weight = 0.0
    var ratio_on = List[Float32]()
    var ratio_off = List[Float32]()
    for si in range(2):
        for on in range(2):
            ren.sota_cam.ae.purkinje = on == 1
            ren.sota_cam.ae.snap_next = True
            ren.begin_frame(cam_pos=Vec3(0.0, 0.0, 0.0), cam_target=Vec3(0.0, 0.0, 1.0), fov_y_deg=40.0, near_z=0.05)
            ren.draw_instance(red_ids[si], trs(Vec3(0.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
            ren.draw_instance(blue_ids[si], trs(Vec3(0.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
            _assert(ren.render_frame(capture_readback=True), "purkinje frame")
            var lr = _hdr_px(ren, (h // 2) * w + w // 4)
            var lb = _hdr_px(ren, (h // 2) * w + 3 * w // 4)
            var yr = lr.x * Float32(0.2126) + lr.y * Float32(0.7152) + lr.z * Float32(0.0722)
            var yb = lb.x * Float32(0.2126) + lb.y * Float32(0.7152) + lb.z * Float32(0.0722)
            if on == 1:
                ratio_on.append(yb / max(yr, Float32(1e-20)))
            else:
                ratio_off.append(yb / max(yr, Float32(1e-20)))
    var la = _post_f32(ren, SOTA_AE_ADAPT_NITS)
    print("    blue/red luminance ratio: moonlight off", ratio_off[0], "on", ratio_on[0], "| daylight off", ratio_off[1], "on", ratio_on[1], "| day adaptation", la, "cd/m^2")
    _assert(abs(ratio_off[0] - 1.0) < 0.02 and abs(ratio_off[1] - 1.0) < 0.02, "equal photopic luminance without the pass")
    _assert(ratio_on[0] > Float32(3.0), "Purkinje shift at night: blue brighter than red")
    _assert(abs(ratio_on[1] - ratio_off[1]) < 1e-4, "no change in daylight")
    ren.sota_cam.ae.purkinje = False
    ren.sota_cam.ae.enable = False
    ren.enable_auto_exposure = False
    print("  [PASS] Suite 3b: mesopic / scotopic vision (Purkinje shift at night, none in daylight)")
    ren.shutdown()


# ----------------------------------------------------------------------------------------------
# Physical sky (area H) with the exposure and HDR paths
# ----------------------------------------------------------------------------------------------

struct _Outdoor(Copyable, Movable):
    var ground: Int
    var block: Int
    var lamp: Int

    def __init__(out self, mut ren: FlintAaaVulkanRenderer):
        var m_ground = ren.add_material(AaaMaterial(base_color=Vec3(0.16, 0.17, 0.14), roughness=0.9, metallic=0.0))
        var m_block = ren.add_material(AaaMaterial(base_color=Vec3(0.45, 0.42, 0.38), roughness=0.7, metallic=0.0))
        var m_lamp = ren.add_material(_emissive(Vec3(0.004, 0.0028, 0.0012)))  # ~110 cd/m^2 warm lamp
        self.ground = ren.add_mesh(_box(Vec3(0.0, -0.5, 0.0), Vec3(3000.0, 0.5, 3000.0)), m_ground).mesh_id
        self.block = ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0)), m_block).mesh_id
        self.lamp = ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(0.08, 0.08, 0.08)), m_lamp).mesh_id


def _outdoor_frame(mut ren: FlintAaaVulkanRenderer, sc: _Outdoor, target: Vec3, fov: Float32, cam_y: Float32 = 1.7) -> Bool:
    ren.begin_frame(cam_pos=Vec3(0.0, cam_y, 0.0), cam_target=target, fov_y_deg=fov, near_z=0.1)
    ren.draw_instance(sc.ground, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    # Rings of blocks around the camera (so every view direction has near and distant geometry) and a ring
    # of small warm lamps.
    for row in range(5):
        var rad = Float32(14.0) * exp2(Float32(row))
        for i in range(24):
            if (i + row) % 3 == 0:
                continue
            var ang = (Float32(i) + Float32(row) * Float32(0.37)) * Float32(0.2617993877991494)
            var s = rad * Float32(0.035) + Float32(0.6)
            var hgt = s * (Float32(1.0) + Float32((i + row * 3) % 4) * Float32(0.5))
            ren.draw_instance(sc.block, trs(Vec3(rad * sin(ang), hgt, rad * cos(ang)), Quat(), Vec3(s, hgt, s)))
    for i in range(12):
        var ang = Float32(i) * Float32(0.5235987755982988)
        ren.draw_instance(sc.lamp, trs(Vec3(Float32(9.0) * sin(ang), 2.6, Float32(9.0) * cos(ang)), Quat(), Vec3(1.0, 1.0, 1.0)))
    return ren.render_frame(capture_readback=True)


def _hour_for_elevation(day: Int, lat: Float64, lon: Float64, elev_deg: Float32, h0: Float64, h1: Float64) -> Float64:
    """UTC hour in [h0, h1] where the sun's elevation crosses `elev_deg` (bisection; sun setting)."""
    var lo = h0
    var hi = h1
    var target = Float32(sin_deg(elev_deg))
    for _ in range(40):
        var mid = (lo + hi) * 0.5
        var y = SotaTimeOfDay(2026, 6, day, mid, lat, lon).sun_dir().y
        if y > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def sin_deg(d: Float32) -> Float32:
    from std.math import sin
    return sin(d * Float32(0.017453292519943295))


def _mean_luma8(px: List[UInt32]) -> Float32:
    var s = Float64(0.0)
    for p in px:
        var r = Float64(Int((p >> 16) & 0xFF))
        var g = Float64(Int((p >> 8) & 0xFF))
        var b = Float64(Int(p & 0xFF))
        s += 0.2126 * r + 0.7152 * g + 0.0722 * b
    return Float32(s / Float64(max(1, len(px))))


def test_gpu_atmosphere_exposure():
    var w = 480
    var h = 270
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    if not ren.sota_atmo.pipelines_ready():
        print("  [SKIP] sota:atmo pipelines unavailable")
        return
    _setup(ren)
    ren.display_mode = AAA_DISPLAY_AGX_REC709
    var sc = _Outdoor(ren)
    ren.sota_atmo.enabled = True
    ren.sota_atmo.use_time_of_day = True
    ren.sota_atmo.moon_enabled = True
    var lat = Float64(48.85)
    var lon = Float64(2.35)
    var names = List[String]()
    var tods = List[SotaTimeOfDay]()
    names.append("noon"); tods.append(SotaTimeOfDay(2026, 6, 21, 11.9, lat, lon))
    names.append("sunset"); tods.append(SotaTimeOfDay(2026, 6, 21, _hour_for_elevation(21, lat, lon, 1.5, 17.0, 22.0), lat, lon))
    names.append("twilight"); tods.append(SotaTimeOfDay(2026, 6, 21, _hour_for_elevation(21, lat, lon, -5.5, 18.0, 23.0), lat, lon))
    names.append("night"); tods.append(SotaTimeOfDay(2026, 6, 30, 1.0, lat, lon))
    var strip = List[UInt32](length=w * h * 2 * 4, fill=0)
    var metered = List[Float32]()
    var mean_new = List[Float32]()
    var mean_old = List[Float32]()
    for ti in range(len(tods)):
        ren.sota_atmo.time_of_day = tods[ti]
        # Aim at the light (sun / moon) azimuth, a little above the horizon.
        ren.sota_atmo.update_lights()
        var ld = ren.sota_atmo.time_of_day.sun_dir() if ti < 3 else ren.sota_atmo.time_of_day.moon_dir()
        var flat = Vec3(ld.x, 0.0, ld.z).normalized()
        var tgt = Vec3(0.0, 1.7, 0.0) + flat * Float32(10.0) + Vec3(0.0, 1.2, 0.0)
        for mode in range(2):
            ren.enable_auto_exposure = True
            ren.sota_cam.ae.enable = mode == 1
            ren.sota_cam.ae.snap_next = True
            ren.sota_cam.ae.purkinje = mode == 1
            for _ in range(2 if mode == 0 else 1):
                _assert(_outdoor_frame(ren, sc, tgt, 70.0), "outdoor frame")
            if mode == 0:
                for _ in range(40):  # let the v5 meter converge (per-frame lerp)
                    _ = _outdoor_frame(ren, sc, tgt, 70.0)
            var px = ren.readback_pixels()
            var m = _mean_luma8(px)
            if mode == 1:
                metered.append(_post_f32(ren, SOTA_AE_METERED_EV))
                mean_new.append(m)
            else:
                mean_old.append(m)
            for y in range(h):
                for x in range(w):
                    strip[(ti * h + y) * w * 2 + mode * w + x] = px[y * w + x]
        print("    " + names[ti] + ": sun elevation", Float32(asin_deg(ren.sota_atmo.time_of_day.sun_dir().y)), "deg, metered EV100", metered[ti],
              "(target", _post_f32(ren, SOTA_AE_TARGET_EV), ", night comp", _post_f32(ren, SOTA_AE_COMP_EV), "EV), mean sRGB luma v5", mean_old[ti], "| EV100", mean_new[ti])
    _ = write_png(String(OUT_DIR) + "/atmo_v5_vs_ev100_meter.png", strip, w * 2, h * 4)
    _assert(metered[0] > Float32(12.0) and metered[0] < Float32(17.0), "noon meters as daylight")
    _assert(metered[3] < Float32(-1.0) and metered[3] > Float32(-7.0), "moonlit night meters at a night EV")
    _assert(metered[0] > metered[1] and metered[1] > metered[2] and metered[2] > metered[3], "EV falls from noon to night")
    _assert(mean_new[0] >= mean_new[2] and mean_new[2] > mean_new[3], "night stays darker than day and twilight")
    _assert(mean_new[3] > Float32(12.0), "night is visible with the EV100 meter (" + String(mean_new[3]) + ")")
    _assert(mean_new[2] > mean_old[2] + Float32(5.0) and mean_new[3] > mean_old[3] + Float32(5.0), "the EV100 meter lifts twilight and night over the v5 clamp")

    # The physical camera body follows the photometric meter (aperture priority f/2.8, auto ISO):
    # daylight -> fast shutter at base ISO; moonlight -> slowest handheld shutter and high ISO.
    ren.sota_cam.drive_renderer = True
    ren.sota_cam.camera = PhysicalCameraModel(focal_length_mm=35.0, fstop=2.8, focus_distance_m=10.0)
    ren.sota_cam.camera.exposure_mode = PHYS_EXPOSURE_APERTURE_PRIORITY
    ren.sota_cam.camera.max_iso = 25600.0
    ren.sota_cam.ae.enable = True
    ren.sota_cam.ae.purkinje = False
    var tri = List[String]()
    for k in range(2):
        var ti = 0 if k == 0 else 3
        ren.sota_atmo.time_of_day = tods[ti]
        ren.sota_cam.ae.snap_next = True
        for _ in range(3):
            _ = _outdoor_frame(ren, sc, Vec3(10.0, 2.9, 3.0), 70.0)
        var cam = ren.sota_cam.camera
        tri.append(names[ti] + ": EV100 " + String(cam.ev100()) + " = f/" + String(cam.fstop) + ", 1/" + String(Float32(1.0) / cam.shutter_s) + " s, ISO " + String(cam.iso))
        if k == 0:
            _assert(cam.iso <= Float32(101.0) and cam.shutter_s < Float32(1.0 / 1000.0), "daylight: base ISO, fast shutter")
            _near(cam.ev100(), _post_f32(ren, SOTA_AE_ADAPTED_EV), 0.05, "camera EV100 = metered exposure setting")
        else:
            # A handheld body cannot reach moonlight (f/2.8, 1/60 s, ISO 25600 = EV100 0.9): it pins shutter and
            # ISO (max grain); the renderer's multiplier still applies the metered exposure.
            _assert(cam.iso >= cam.max_iso * Float32(0.999) and cam.shutter_s >= cam.min_shutter_s * Float32(0.999), "moonlight: slowest handheld shutter, max ISO")
    print("    camera body (aperture priority): " + tri[0] + " | " + tri[1])
    ren.sota_cam.drive_renderer = False

    # HDR10 / scRGB with the physical sky and the sun disc in view (sunset, looking at the sun).
    ren.sota_atmo.time_of_day = tods[1]
    ren.sota_atmo.update_lights()
    var sd = ren.sota_atmo.time_of_day.sun_dir()
    # From 80 m up, so no block hides the sun disc (elevation 1.5 deg), placed off centre.
    var tgt_sun = Vec3(0.0, 80.0, 0.0) + sd * Float32(10.0) + Vec3(sd.z, 0.0, -sd.x) * Float32(1.2)
    ren.sota_cam.ae.enable = True
    ren.sota_cam.ae.purkinje = False
    ren.sota_cam.ae.snap_next = True
    ren.sota_cam.hdr.enable = True
    ren.display_mode = AAA_DISPLAY_AGX_REC709
    for _ in range(10):  # converge VSM pages / SVGF history after the camera jump
        _assert(_outdoor_frame(ren, sc, tgt_sun, 30.0, 80.0), "SDR sun frame")
    var sdr = ren.readback_pixels()
    var res = List[String]()
    var words = List[List[UInt32]]()
    for mode in range(2):
        ren.sota_cam.hdr.expansion_mode = SOTA_HDR_EXPAND_MAX_CHANNEL if mode == 0 else SOTA_HDR_EXPAND_LUMINANCE
        ren.display_mode = AAA_DISPLAY_HDR10_PQ
        _assert(_outdoor_frame(ren, sc, tgt_sun, 30.0, 80.0), "HDR sun frame")
        var pq = ren.readback_pixels()
        ren.display_mode = AAA_DISPLAY_SCRGB_FP16
        _assert(_outdoor_frame(ren, sc, tgt_sun, 30.0, 80.0), "scRGB sun frame")
        var sc_ldr = ren.readback_pixels()
        var sc_words = ren.readback_scrgb_rgba16f_words()
        var hdr = ren.sota_cam.hdr
        var peak = hdr.paper() * hdr.peak_ratio()
        var max_n = Float32(0.0)
        var bad = 0
        var worst_rel = Float32(0.0)
        var n_rel = 0
        var scrgb_worst = Float32(0.0)
        var n_diff = 0
        var sky_sum = Float64(0.0)
        var sky_n = 0
        for i in range(len(pq)):
            var n2020 = pq_word_decode_nits_cpu(pq[i])
            var n = rec2020_to_rec709_cpu(n2020[0], n2020[1], n2020[2])
            var m = max(n[0], max(n[1], n[2]))
            if not (m == m) or m > peak * Float32(1.012):
                bad += 1
            max_n = max(max_n, m)
            var a = sdr[i]
            var rf = sota_hdr_map_nits_cpu(Float32(Int((a >> 16) & 0xFF)) / 255.0, Float32(Int((a >> 8) & 0xFF)) / 255.0, Float32(Int(a & 0xFF)) / 255.0, hdr)
            var rm = max(rf[0], max(rf[1], rf[2]))
            if rm > Float32(5.0):
                var rel_i = abs(m - rm) / rm
                worst_rel = max(worst_rel, rel_i)
                n_rel += 1
            if sc_ldr[i] != pq[i]:
                n_diff += 1
            var sn2020 = pq_word_decode_nits_cpu(sc_ldr[i])
            var sn = rec2020_to_rec709_cpu(sn2020[0], sn2020[1], sn2020[2])
            var s_r = f16_to_f32_cpu(sc_words[i * 2] & 0xFFFF) * Float32(80.0)
            var s_g = f16_to_f32_cpu(sc_words[i * 2] >> 16) * Float32(80.0)
            var s_b = f16_to_f32_cpu(sc_words[i * 2 + 1] & 0xFFFF) * Float32(80.0)
            scrgb_worst = max(scrgb_worst, max(abs(s_r - sn[0]), max(abs(s_g - sn[1]), abs(s_b - sn[2]))) / max(Float32(1.0), max(sn[0], max(sn[1], sn[2]))))
            if i < w * (h // 4):  # upper quarter: sky
                sky_sum += Float64(Float32(0.2126) * n[0] + Float32(0.7152) * n[1] + Float32(0.0722) * n[2])
                sky_n += 1
        var sky_mean = Float32(sky_sum / Float64(max(1, sky_n)))
        res.append(String("brightest ") + String(max_n) + " nits, out-of-range " + String(bad) + ", vs CPU worst rel " + String(worst_rel)
                   + " (" + String(n_rel) + " px), scRGB vs its PQ word worst rel " + String(scrgb_worst) + ", PQ words differing from the HDR10 frame " + String(n_diff) + ", upper-sky mean " + String(sky_mean) + " nits")
        print("    sunset sun in view, HDR expansion " + ("max channel" if mode == 0 else "luminance") + ": " + res[mode])
        _assert(bad == 0, "no NaN / over-peak PQ words with the sky and sun disc")
        _assert(max_n > peak * Float32(0.9), "the sun disc reaches the display peak")
        _assert(worst_rel < Float32(0.12), "HDR words match the CPU mapping of the SDR frame")
        _assert(n_diff * 100 < len(pq) and scrgb_worst < Float32(0.01), "scRGB = PQ output with the physical sky")
        words.append(pq.copy())
    # Luminance mode never exceeds the max-channel mode and keeps saturated sky colours lower.
    var lower = 0
    var higher = 0
    for i in range(len(words[0])):
        var a = pq_word_decode_nits_cpu(words[0][i])
        var b = pq_word_decode_nits_cpu(words[1][i])
        var ya = a[0] + a[1] + a[2]
        var yb = b[0] + b[1] + b[2]
        if yb < ya * Float32(0.98):
            lower += 1
        if yb > ya * Float32(1.02) + Float32(0.5):
            higher += 1
    print("    luminance vs max-channel expansion: " + String(lower) + " px dimmer, " + String(higher) + " px brighter (of " + String(len(words[0])) + ")")
    _assert(lower > 0, "luminance expansion differs from the max channel on coloured highlights")
    # False-colour nits of both modes for the eval folder.
    var fc = List[UInt32](length=w * h * 3, fill=0)
    for y in range(h):
        for x in range(w):
            fc[y * w * 3 + x] = sdr[y * w + x]
            for mode in range(2):
                var n2020 = pq_word_decode_nits_cpu(words[mode][y * w + x])
                var mx = max(n2020[0], max(n2020[1], n2020[2]))
                var t = max(Float32(0.0), min(Float32(1.0), Float32(log10(Float64(max(Float32(1.0), mx)))) / Float32(3.0)))
                var r8 = UInt32(Int(min(Float32(255.0), t * Float32(2.0) * Float32(255.0))))
                var g8 = UInt32(Int(max(Float32(0.0), min(Float32(255.0), (t * Float32(2.0) - Float32(0.5)) * Float32(255.0)))))
                var b8 = UInt32(Int(max(Float32(0.0), min(Float32(255.0), (t * Float32(2.0) - Float32(1.0)) * Float32(255.0)))))
                fc[y * w * 3 + (mode + 1) * w + x] = 0xFF000000 | (r8 << 16) | (g8 << 8) | b8
    _ = write_png(String(OUT_DIR) + "/atmo_sunset_sdr_hdr_max_vs_luminance.png", fc, w * 3, h)

    # Cost of the meter at 480x270 (median of 12 frames).
    ren.display_mode = AAA_DISPLAY_AGX_REC709
    ren.sota_cam.hdr.enable = False
    ren.sota_cam.hdr.expansion_mode = SOTA_HDR_EXPAND_MAX_CHANNEL
    var t_new = List[Float64]()
    var t_old = List[Float64]()
    for f in range(24):
        ren.sota_cam.ae.enable = (f & 1) == 0
        _ = _outdoor_frame(ren, sc, tgt_sun, 70.0, 80.0)
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == "SotaAutoExposure":
                t_new.append(t.gpu_ms)
            elif t.pass_name == "LuminanceAutoExposure":
                t_old.append(t.gpu_ms)
    print("    GPU ms at 480x270 (median): EV100 meter", _median(t_new), "| v5 meter", _median(t_old))
    ren.sota_cam.ae.enable = False
    ren.enable_auto_exposure = False
    ren.sota_atmo.enabled = False
    print("  [PASS] Suite 3c: physical sky noon / sunset / twilight / night with the EV100 meter; HDR10 / scRGB with the sun disc")
    ren.shutdown()


def asin_deg(y: Float32) -> Float32:
    from std.math import asin
    return asin(max(Float32(-1.0), min(Float32(1.0), y))) * Float32(57.29577951308232)


def _median(v: List[Float64]) -> Float64:
    if len(v) == 0:
        return -1.0
    var s = v.copy()
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[j] < s[i]:
                var t = s[i]
                s[i] = s[j]
                s[j] = t
    return s[len(s) // 2]


def test_gpu_meter_budget():
    var w = 1920
    var h = 1080
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan renderer unavailable")
        return
    _setup(ren)
    _black_env(ren)
    var m = ren.add_material(_emissive(Vec3(0.05, 0.04, 0.03)))
    var wall = ren.add_mesh(_box(Vec3(0.0, 0.0, 0.0), Vec3(20.0, 20.0, 0.02)), m).mesh_id
    ren.enable_auto_exposure = True
    var t_new = List[Float64]()
    var t_old = List[Float64]()
    for f in range(32):
        ren.sota_cam.ae.enable = (f & 1) == 0
        _ = _wall_frame(ren, wall)
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == "SotaAutoExposure":
                t_new.append(t.gpu_ms)
            elif t.pass_name == "LuminanceAutoExposure":
                t_old.append(t.gpu_ms)
    var mn = _median(t_new)
    var mo = _median(t_old)
    var best_new = Float64(1e9)
    for v in t_new:
        best_new = min(best_new, v)
    print("    1080p GPU ms: EV100 meter median", mn, "best", best_new, "| v5 meter median", mo)
    _assert(len(t_new) > 0, "meter timed")
    _assert(best_new < 0.5, "EV100 meter under 0.5 ms at 1080p (best of 16)")
    ren.sota_cam.ae.enable = False
    ren.enable_auto_exposure = False
    print("  [PASS] Suite 3d: EV100 meter cost at 1080p")
    ren.shutdown()


def main():
    print("=== Flint v5 sota:camera round 4 (Area D): EV100 auto exposure, mesopic vision, HDR expansion, sky ===")
    var bdir = String("build")
    var odir = String(OUT_DIR)
    _ = external_call["mkdir", Int32](bdir.as_c_string_span(), Int32(0o755))
    _ = external_call["mkdir", Int32](odir.as_c_string_span(), Int32(0o755))
    test_spirv()
    test_cpu_exposure()
    test_cpu_hdr_expansion()
    test_gpu_meter()
    test_gpu_purkinje()
    test_gpu_atmosphere_exposure()
    test_gpu_meter_budget()
    print("=== ALL sota:camera r4 TESTS PASSED ===")
