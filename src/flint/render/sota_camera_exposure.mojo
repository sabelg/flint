"""Area D round 4 (sota:camera): physically based auto exposure that covers day, twilight and night.

Why a new meter. The v5 meter (`aaa_adapt_exposure_shader`) bins log2 of the raw scene luminance over
[-10, +14] EV of engine units, sets `exposure = 0.18 / avg` and clamps it to [0.15, 3.2] (about +-2.2 EV).
Under the physical sky (area H) the sun is 4.5 units, full-moon light is ~1e-6 of that and twilight sits in
between, so every scene darker than late afternoon hits the clamp or the bottom histogram bin and renders
black (H round 2 had to raise the clamp by hand or switch to manual exposure for twilight and night).

What this meter does.
  1. Calibration to photometric units: `nits_per_unit` converts engine radiance to cd/m^2. The default is
     128 000 lx (solar illuminance at the top of the atmosphere) / 4.5 (the engine's TOA solar irradiance),
     i.e. 28 444 cd/m^2 per unit; with area H's sky on it is re-derived from the sky's own solar irradiance
     every frame (`calibrate_from_sky`).
  2. EV100 histogram: 128 bins over EV100 [-14, +22) (0.28 EV per bin; starlight to sunlit snow), EV100 of a
     pixel = log2(L * S / K) with S = 100, K = 12.5 (ISO 2720 reflected-light meter constant), built with
     workgroup-shared atomics (one global atomic per non-empty bin per 16x16 tile), optional centre weighting.
  3. Metering: percentile-trimmed mean of the histogram (default 10 % .. 90 %, so the sun disc, the moon,
     lamps and black sky corners do not steer it), clamped to the camera's metering range [min_ev100, max_ev100].
  4. Night / twilight compensation (the photographer's and the eye's rule that night must stay darker than
     day): exposure compensation as a smooth function of the metered EV100, 0 above `day_ev100` and
     `night_comp_ev` (default -2.5 EV) below `night_ev100`. A moonlit landscape (metered about EV100 -3.5)
     therefore renders as a dark blue-grey night (mid grey at 0.18 * 2^-2.5), not as an overcast day.
  5. Adaptation in EV space with frame-rate independent exponential smoothing and asymmetric time constants:
     getting brighter is fast (cone light adaptation, `tau_brighten_s`), getting darker is slower
     (`tau_darken_s`) and slower again in the scotopic range (`tau_darken_night_s`, rod dark adaptation),
     with an optional maximum rate in EV/s.
  6. Optional mesopic / scotopic vision (`purkinje`): below ~5 cd/m^2 adaptation luminance the image blends
     towards the rod response (scotopic luminance from CIE XYZ, Larson et al. 1997), shown with a blue shift
     (Jensen et al. 2000), so moonlit scenes lose colour and blues brighten relative to reds (Purkinje shift).

Output contract (unchanged consumers). The adapted EV100 is converted back to the renderer's exposure
multiplier and written to post-scratch word 64 (and the mean luminance to word 65), exactly where the v5
meter wrote them, so the display tonemap, the lens / VRS passes, the CPU readback (`get_adapted_exposure`)
and `SotaCameraPost.frame_sync` keep working. With `calibrate_camera` the physical camera's `ev100_unity`
is set to log2(8 * nits_per_unit * key) (= 15.32 with the defaults: sunny-16 is exposure 1.0), so the
exposure triangle the camera body solves (shutter, ISO, grain, motion blur) is in real EV100.

Everything here is behind `SotaAutoExposure.enable` (default False) and only replaces the v5 meter when the
renderer's `enable_auto_exposure` is also on; with it off no pass changes and frames are bit-identical.

Post-scratch words used (free in the v5 layout, below the bloom / DoF base 1024):
  [768 .. 895]  EV100 histogram (128 bins, tent-split weights, 64 = one un-weighted pixel)
  [896 .. 903]  state: adapted EV (exposure setting), magic, metered EV100, target EV, total weight,
                compensation applied, adaptation luminance (cd/m^2), night factor

Sources: ISO 2720:1974 (K = 12.5); Lagarde & de Rousiers 2014, "Moving Frostbite to PBR" s. 4.3 (EV100 and
exposure); Krawczyk, Myszkowski & Seidel 2005, "Perceptual effects in real-time tone mapping" (key value that
falls with adaptation luminance, temporal adaptation); Pattanaik et al. 2000, "Time-dependent visual
adaptation for fast realistic image display" (cone / rod adaptation time scales); Larson, Rushmeier & Piatko
1997, "A visibility matching tone reproduction operator" (scotopic luminance from XYZ); Jensen et al. 2000,
"A physically-based night sky model" (blue shift); CIE 191:2010 (mesopic range 0.005 .. 5 cd/m^2).
No code was copied.
"""
from std.collections import List
from std.math import exp, exp2, floor, log2, max, min
from flint.core.math import Vec3
from std.memory import bitcast
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    PIPELINE_BOTTOM,
    PIPELINE_COMPUTE,
    PIPELINE_HOST,
    VulkanGpu,
)
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    GpuF32,
    ShaderDsl,
    clamp_f,
    dot,
    exp2_f,
    floor_f,
    fma_f,
    log2_f,
    max_f,
    min_f,
    min_u,
    select_f,
)
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH

comptime SOTA_AE_BINS: Int = 128
comptime SOTA_AE_HIST_WORD: Int = 768
comptime SOTA_AE_STATE_WORD: Int = 896
comptime SOTA_AE_ADAPTED_EV: Int = 896     # adapted exposure-setting EV100 (f32)
comptime SOTA_AE_MAGIC_WORD: Int = 897     # == SOTA_AE_MAGIC once the state is valid
comptime SOTA_AE_METERED_EV: Int = 898     # this frame's trimmed-mean EV100 (f32)
comptime SOTA_AE_TARGET_EV: Int = 899      # this frame's exposure-setting target (metered - compensation)
comptime SOTA_AE_TOTAL_W: Int = 900        # histogram weight sum (f32)
comptime SOTA_AE_COMP_EV: Int = 901        # night compensation applied this frame (EV, <= 0)
comptime SOTA_AE_ADAPT_NITS: Int = 902     # adaptation luminance for the mesopic blend (cd/m^2)
comptime SOTA_AE_NIGHT_F: Int = 903        # night factor 0 (day) .. 1 (night) of the adapted state
comptime SOTA_AE_MAGIC: UInt32 = 0xAE5E0D04
comptime SOTA_AE_EV_MIN: Float32 = -14.0
comptime SOTA_AE_EV_MAX: Float32 = 22.0
comptime SOTA_AE_RUN: Int = 4  # samples per histogram invocation (down a column)
comptime SOTA_AE_WEIGHT_ONE: Float32 = 64.0  # fixed-point weight of an un-weighted pixel
comptime _POST_ADAPTED_EXP_WORD: Int = 64      # == AAA_POST_ADAPTED_EXP_WORD
comptime _POST_AVG_LUMA_WORD: Int = 65         # == AAA_POST_AVG_LUMA_WORD
comptime _LOG2_E: Float32 = 1.4426950408889634
comptime SOTA_AE_SOLAR_LUX: Float32 = 128000.0  # solar illuminance at the top of the atmosphere
comptime SOTA_AE_ENGINE_SOLAR: Float32 = 4.5    # engine TOA solar irradiance (`SotaAtmoParams` default)


struct SotaAutoExposure(Copyable, ImplicitlyCopyable, Movable):
    """Physically based auto-exposure settings (cvars). `enable` is off by default."""
    var enable: Bool
    var nits_per_unit: Float32       # cd/m^2 per engine radiance unit
    var calibrate_from_sky: Bool     # with area H's sky on: nits_per_unit = SOLAR_LUX / luma(solar irradiance)
    var calibrate_camera: Bool       # set the physical camera's ev100_unity to the photometric calibration
    var key: Float32                 # exposed value of the metered mean (0.18 = mid grey)
    var min_ev100: Float32           # metering range of the camera (scenes outside it under / over expose)
    var max_ev100: Float32
    var low_percentile: Float32      # trimmed mean over [low, high] of the weighted histogram
    var high_percentile: Float32
    var center_weight: Float32       # 0 = average metering, 1 = strongly centre-weighted
    var sample_step: Int             # histogram sample spacing in pixels (1 or 2; 2 reads 1/4 of the frame)
    var night_ev100: Float32         # full night compensation at and below this metered EV100
    var day_ev100: Float32           # no compensation at and above this metered EV100
    var night_comp_ev: Float32       # compensation at night (EV, negative = darker than mid grey)
    var tau_brighten_s: Float32      # time constant when the scene gets brighter (exposure goes down)
    var tau_darken_s: Float32        # time constant when the scene gets darker (photopic)
    var tau_darken_night_s: Float32  # same in the scotopic range (rod dark adaptation, game-scaled)
    var max_rate_ev_s: Float32       # 0 = unlimited; else |dEV/dt| cap
    var snap_next: Bool              # jump straight to the target on the next frame (cuts, first frame)
    var purkinje: Bool               # mesopic / scotopic colour vision (Purkinje shift + desaturation)
    var purkinje_strength: Float32
    var mesopic_lo_nits: Float32     # fully scotopic at and below this adaptation luminance
    var mesopic_hi_nits: Float32     # fully photopic at and above
    var night_tint: Vec3            # rod-vision display tint (luminance 1)
    var pipe_hist: Int
    var pipe_adapt: Int
    var pipe_purkinje: Int

    def __init__(out self):
        self.enable = False
        self.nits_per_unit = SOTA_AE_SOLAR_LUX / SOTA_AE_ENGINE_SOLAR
        self.calibrate_from_sky = True
        self.calibrate_camera = True
        self.key = Float32(0.18)
        self.min_ev100 = Float32(-6.0)
        self.max_ev100 = Float32(18.0)
        self.low_percentile = Float32(0.10)
        self.high_percentile = Float32(0.90)
        self.center_weight = Float32(0.35)
        self.sample_step = 2
        self.night_ev100 = Float32(-4.0)
        self.day_ev100 = Float32(12.0)
        self.night_comp_ev = Float32(-2.5)
        self.tau_brighten_s = Float32(0.4)
        self.tau_darken_s = Float32(1.2)
        self.tau_darken_night_s = Float32(4.0)
        self.max_rate_ev_s = Float32(0.0)
        self.snap_next = True
        self.purkinje = False
        self.purkinje_strength = Float32(1.0)
        self.mesopic_lo_nits = Float32(0.005)
        self.mesopic_hi_nits = Float32(5.0)
        self.night_tint = Vec3(0.86, 1.0, 1.32)
        self.pipe_hist = 0
        self.pipe_adapt = 0
        self.pipe_purkinje = 0

    def create_pipelines(mut self, mut vk: VulkanGpu, pipe_layout: Int) -> Bool:
        self.pipe_hist = vk.create_aaa_compute_pipeline(pipe_layout, sota_ae_histogram_shader())
        self.pipe_adapt = vk.create_aaa_compute_pipeline(pipe_layout, sota_ae_adapt_shader())
        self.pipe_purkinje = vk.create_aaa_compute_pipeline(pipe_layout, sota_ae_purkinje_shader())
        return self.pipelines_ready()

    def pipelines_ready(self) -> Bool:
        return self.pipe_hist != 0 and self.pipe_adapt != 0 and self.pipe_purkinje != 0

    def active(self, renderer_auto_exposure: Bool) -> Bool:
        return self.enable and renderer_auto_exposure and self.pipelines_ready()

    def ev100_unity(self) -> Float32:
        """EV100 at which the renderer's exposure multiplier is 1: log2(8 * nits_per_unit * key)."""
        return log2(Float32(8.0) * max(Float32(1.0e-6), self.nits_per_unit) * max(Float32(1.0e-4), self.key))

    def calibrate_to_sky(mut self, solar_irradiance_luma: Float32):
        if self.calibrate_from_sky and solar_irradiance_luma > Float32(1.0e-6):
            self.nits_per_unit = SOTA_AE_SOLAR_LUX / solar_irradiance_luma

    def exposure_for_ev(self, ev100: Float32) -> Float32:
        """Renderer exposure multiplier for an exposure setting of `ev100`."""
        return exp2(self.ev100_unity() - ev100)


# ----------------------------------------------------------------------------------------------
# CPU references
# ----------------------------------------------------------------------------------------------

def ev100_from_nits_cpu(nits: Float32) -> Float32:
    """Reflected-light meter: EV100 = log2(L * 100 / 12.5)."""
    return log2(max(Float32(1.0e-12), nits) * Float32(8.0))


def sota_ae_splat_cpu(mut hist: List[UInt32], ev100: Float32, weight: UInt32):
    """CPU twin of the GPU tent binning: splits `weight` between the two bins whose centres bracket `ev100`."""
    var t = (ev100 - SOTA_AE_EV_MIN) * (Float32(SOTA_AE_BINS) / (SOTA_AE_EV_MAX - SOTA_AE_EV_MIN)) - Float32(0.5)
    var t0 = floor(t)
    var frac = max(Float32(0.0), min(Float32(1.0), t - t0))
    var b0 = max(0, min(SOTA_AE_BINS - 1, Int(t0)))
    var b1 = max(0, min(SOTA_AE_BINS - 1, Int(t0) + 1))
    var w1 = min(UInt32(Int(Float32(Int(weight)) * frac + Float32(0.5))), weight)
    hist[b0] += weight - w1
    hist[b1] += w1


def sota_ae_bin_ev_cpu(b: Int) -> Float32:
    """Bin centre in EV100."""
    return SOTA_AE_EV_MIN + (Float32(b) + Float32(0.5)) * ((SOTA_AE_EV_MAX - SOTA_AE_EV_MIN) / Float32(SOTA_AE_BINS))


def sota_ae_center_weight_cpu(nx: Float32, ny: Float32, cw: Float32) -> UInt32:
    """Fixed-point weight of a pixel at normalised position (nx, ny) in [-1, 1]^2."""
    var d2 = nx * nx + ny * ny
    var w = (Float32(1.0) - cw) + cw * exp2(Float32(-2.5) * d2)
    return UInt32(Int(w * SOTA_AE_WEIGHT_ONE + Float32(0.5)))


def sota_ae_meter_cpu(hist: List[UInt32], lo_pct: Float32, hi_pct: Float32) -> Tuple[Float32, Float32]:
    """Percentile-trimmed mean EV100 of a weighted histogram: `(metered_ev100, total_weight)`."""
    var tot = Float32(0.0)
    for i in range(SOTA_AE_BINS):
        tot += Float32(Int(hist[i]))
    var lo = max(Float32(0.0), min(Float32(1.0), lo_pct)) * tot
    var hi = max(Float32(0.0), min(Float32(1.0), hi_pct)) * tot
    var run = Float32(0.0)
    var ws = Float32(0.0)
    var es = Float32(0.0)
    for i in range(SOTA_AE_BINS):
        var c = Float32(Int(hist[i]))
        var r0 = run
        run += c
        var w = max(Float32(0.0), min(run, hi) - max(r0, lo))
        es += w * sota_ae_bin_ev_cpu(i)
        ws += w
    return (es / max(ws, Float32(1.0e-6)), tot)


def _smoothstep_cpu(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = max(Float32(0.0), min(Float32(1.0), (x - e0) / max(Float32(1.0e-6), e1 - e0)))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


def sota_ae_night_factor_cpu(ae: SotaAutoExposure, ev100: Float32) -> Float32:
    return Float32(1.0) - _smoothstep_cpu(ae.night_ev100, ae.day_ev100, ev100)


def sota_ae_target_ev_cpu(ae: SotaAutoExposure, metered_ev100: Float32, user_comp_ev: Float32) -> Float32:
    """Exposure setting for a metered scene: clamp to the metering range, then subtract the night
    compensation (negative) and the user's compensation."""
    var ev = max(ae.min_ev100, min(ae.max_ev100, metered_ev100))
    var comp = ae.night_comp_ev * sota_ae_night_factor_cpu(ae, metered_ev100) + user_comp_ev
    return ev - comp


def sota_ae_adapt_cpu(ae: SotaAutoExposure, prev_ev: Float32, target_ev: Float32, dt_s: Float32) -> Float32:
    """One adaptation step in EV space (frame-rate independent)."""
    var delta = target_ev - prev_ev
    var nf = sota_ae_night_factor_cpu(ae, prev_ev)
    var tau_dark = ae.tau_darken_s + (ae.tau_darken_night_s - ae.tau_darken_s) * nf
    # Scene brighter -> the exposure-setting EV rises (delta > 0) -> fast light adaptation.
    var tau = ae.tau_brighten_s if delta > Float32(0.0) else tau_dark
    var a = Float32(1.0) - exp(-max(Float32(0.0), dt_s) / max(Float32(1.0e-3), tau))
    var step = delta * a
    if ae.max_rate_ev_s > Float32(0.0):
        var cap = ae.max_rate_ev_s * dt_s
        step = max(-cap, min(cap, step))
    return prev_ev + step


def scotopic_luminance_cpu(r: Float32, g: Float32, b: Float32) -> Float32:
    """Rod (scotopic) luminance from linear Rec.709, normalised so D65 white keeps its photopic luminance
    (Larson et al. 1997: Ys = Y (1.33 (1 + (Y + Z) / X) - 1.68))."""
    var x = Float32(0.4124) * r + Float32(0.3576) * g + Float32(0.1805) * b
    var y = Float32(0.2126) * r + Float32(0.7152) * g + Float32(0.0722) * b
    var z = Float32(0.0193) * r + Float32(0.1192) * g + Float32(0.9505) * b
    var ys = y * (Float32(1.33) * (Float32(1.0) + (y + z) / max(x, Float32(1.0e-9))) - Float32(1.68))
    return max(Float32(0.0), ys) * Float32(1.0 / 2.5733)


def photopic_weight_cpu(ae: SotaAutoExposure, adapt_nits: Float32) -> Float32:
    var lo = log2(max(Float32(1.0e-9), ae.mesopic_lo_nits))
    var hi = log2(max(Float32(1.0e-9), ae.mesopic_hi_nits))
    var t = max(Float32(0.0), min(Float32(1.0), (log2(max(Float32(1.0e-12), adapt_nits)) - lo) / max(Float32(1.0e-6), hi - lo)))
    var sig = t * t * (Float32(3.0) - Float32(2.0) * t)
    return Float32(1.0) - max(Float32(0.0), min(Float32(1.0), ae.purkinje_strength)) * (Float32(1.0) - sig)


# ----------------------------------------------------------------------------------------------
# GPU shaders
# ----------------------------------------------------------------------------------------------

def sota_ae_histogram_shader() -> List[UInt32]:
    """EV100 histogram of the HDR frame (8 words / px). 16x16 groups; invocation (x, y) meters samples
    (x, RUN y .. RUN y + RUN - 1) of the step grid; dispatch ceil(ceil(w/step)/16) x ceil(ceil(h/step)/(16 RUN)). Push: 0 w, 1 h, 2 nits_per_unit, 3 centre weight (0..1),
    4 sample step (1: every pixel, 2: one pixel per 2x2; the pass is bound by the 32-byte HDR pixel stride)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=16, group_height=16, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var sh = dsl.shared_array_u32[SOTA_AE_BINS]()
    var gx = dsl.global_id_x()
    var gy = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var k_nits = dsl.push_f32(2)
    var cw = clamp_f(dsl.push_f32(3), 0.0, 1.0)
    var step_u = dsl.push_u32(4)
    var px = gx * step_u
    var local_i = (gy & UInt32(15)) * UInt32(16) + (gx & UInt32(15))

    dsl.if_then(local_i.lt(UInt32(SOTA_AE_BINS)))
    sh.store(local_i, dsl.u32(0))
    dsl.end_if()
    dsl.workgroup_barrier()

    # Each invocation meters SOTA_AE_RUN samples down a column and run-length merges equal bins before the
    # shared atomics (a uniform 16x16 tile would otherwise serialise 256 atomics on one bin).
    var cur_bin = dsl.var_u32(dsl.u32(0xFFFFFFFF))
    var acc0 = dsl.var_u32(dsl.u32(0))
    var acc1 = dsl.var_u32(dsl.u32(0))
    var top = dsl.u32(UInt32(SOTA_AE_BINS - 1))
    for r in range(SOTA_AE_RUN):
        var py = (gy * UInt32(SOTA_AE_RUN) + UInt32(r)) * step_u
        var inside = px.lt(w_u) & py.lt(h_u)
        dsl.if_then(inside)
        var rgb = b_hdr.load_vec3(b_hdr.base_word(py * w_u + px))
        var luma = max_f(dot(rgb, dsl.vec3_const(0.2126, 0.7152, 0.0722)), 0.0)
        var ev = log2_f(max_f(luma * k_nits * Float32(8.0), 1.0e-12))
        # Linear (tent) binning about the bin centres: the weight is split between the two nearest bins, so
        # the metered EV moves continuously with the scene instead of in 0.28 EV bin steps.
        var t = (ev - SOTA_AE_EV_MIN) * (Float32(SOTA_AE_BINS) / (SOTA_AE_EV_MAX - SOTA_AE_EV_MIN)) - Float32(0.5)
        var t0 = floor_f(t)
        var in_range = t0.ge(0.0) & t0.lt(Float32(SOTA_AE_BINS - 1))
        var frac = select_f(in_range, clamp_f(t - t0, 0.0, 1.0), dsl.f32(0.0))
        var bin0 = min_u(max_f(t0, 0.0).saturate_u32(), top)
        var nx = (px.to_f32() + Float32(0.5)) / w_u.to_f32() * Float32(2.0) - Float32(1.0)
        var ny = (py.to_f32() + Float32(0.5)) / h_u.to_f32() * Float32(2.0) - Float32(1.0)
        var d2 = nx * nx + ny * ny
        var wgt = (dsl.f32(1.0) - cw) + cw * exp2_f(d2 * Float32(-2.5))
        var wgt_f = wgt * SOTA_AE_WEIGHT_ONE
        var wgt_u = (wgt_f + Float32(0.5)).saturate_u32()
        var w1 = min_u((wgt_f * frac + Float32(0.5)).saturate_u32(), wgt_u)
        var cb = cur_bin.get()
        dsl.if_then(cb.ne(bin0) & cb.ne(UInt32(0xFFFFFFFF)))
        _ = sh.atomic_add(cb, acc0.get())
        _ = sh.atomic_add(min_u(cb + UInt32(1), top), acc1.get())
        acc0.set(dsl.u32(0))
        acc1.set(dsl.u32(0))
        dsl.end_if()
        cur_bin.set(bin0)
        acc0.set(acc0.get() + (wgt_u - w1))
        acc1.set(acc1.get() + w1)
        dsl.end_if()
    var cbf = cur_bin.get()
    dsl.if_then(cbf.ne(UInt32(0xFFFFFFFF)))
    _ = sh.atomic_add(cbf, acc0.get())
    _ = sh.atomic_add(min_u(cbf + UInt32(1), top), acc1.get())
    dsl.end_if()
    dsl.workgroup_barrier()

    dsl.if_then(local_i.lt(UInt32(SOTA_AE_BINS)))
    var v = sh.load(local_i)
    dsl.if_then(v.ne(UInt32(0)))
    _ = b_post.atomic_add(local_i + UInt32(SOTA_AE_HIST_WORD), v)
    dsl.end_if()
    dsl.end_if()
    return dsl.finish()


def sota_ae_adapt_shader() -> List[UInt32]:
    """Meter + night compensation + adaptation. One 128-wide workgroup stages (and clears) the bins in shared
    memory, then invocation 0 reduces them (shared loads instead of 256 dependent global loads). Push:
    0 dt_s, 1 tau_brighten, 2 tau_darken, 3 tau_darken_night, 4 low pct, 5 high pct, 6 min EV100,
    7 max EV100, 8 night EV100, 9 day EV100, 10 night comp EV, 11 user comp EV, 12 EV100 unity,
    13 flags (bit 0: snap), 14 max rate EV/s (0: off), 15 nits_per_unit."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=SOTA_AE_BINS, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var sh = dsl.shared_array_u32[SOTA_AE_BINS]()
    var lid = dsl.global_id_x()
    sh.store(lid, b_post.load_u32(lid + UInt32(SOTA_AE_HIST_WORD)))
    b_post.store_u32(lid + UInt32(SOTA_AE_HIST_WORD), dsl.u32(0))
    dsl.workgroup_barrier()
    dsl.return_if(lid.ne(0))
    var dt = max_f(dsl.push_f32(0), 0.0)
    var tau_b = max_f(dsl.push_f32(1), 1.0e-3)
    var tau_d = max_f(dsl.push_f32(2), 1.0e-3)
    var tau_dn = max_f(dsl.push_f32(3), 1.0e-3)
    var lo_p = clamp_f(dsl.push_f32(4), 0.0, 1.0)
    var hi_p = clamp_f(dsl.push_f32(5), 0.0, 1.0)
    var ev_lo = dsl.push_f32(6)
    var ev_hi = dsl.push_f32(7)
    var night_ev = dsl.push_f32(8)
    var day_ev = dsl.push_f32(9)
    var night_comp = dsl.push_f32(10)
    var user_comp = dsl.push_f32(11)
    var ev_unity = dsl.push_f32(12)
    var snap = (dsl.push_u32(13) & UInt32(1)).ne(UInt32(0))
    var max_rate = dsl.push_f32(14)
    var k_nits = max_f(dsl.push_f32(15), 1.0e-6)

    var tot = dsl.var_f32_const(0.0)
    var bi = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var b0 = bi.get()
    dsl.break_if(b0.ge(UInt32(SOTA_AE_BINS)))
    tot.set(tot.get() + sh.load(b0).to_f32())
    bi.set(b0 + UInt32(1))
    dsl.end_loop()

    var total = tot.get()
    var lo_cut = total * lo_p
    var hi_cut = total * hi_p
    var running = dsl.var_f32_const(0.0)
    var w_sum = dsl.var_f32_const(0.0)
    var ev_sum = dsl.var_f32_const(0.0)
    var bj = dsl.var_u32(dsl.u32(0))
    var bin_w = (SOTA_AE_EV_MAX - SOTA_AE_EV_MIN) / Float32(SOTA_AE_BINS)
    dsl.begin_loop()
    var b1 = bj.get()
    dsl.break_if(b1.ge(UInt32(SOTA_AE_BINS)))
    var cnt = sh.load(b1).to_f32()
    var r0 = running.get()
    var r1 = r0 + cnt
    running.set(r1)
    var wb = max_f(min_f(r1, hi_cut) - max_f(r0, lo_cut), 0.0)
    var bin_ev = fma_f(b1.to_f32() + Float32(0.5), dsl.f32(bin_w), dsl.f32(SOTA_AE_EV_MIN))
    ev_sum.set(fma_f(wb, bin_ev, ev_sum.get()))
    w_sum.set(w_sum.get() + wb)
    bj.set(b1 + UInt32(1))
    dsl.end_loop()

    var prev_ev = b_post.load_f32(dsl.u32(UInt32(SOTA_AE_ADAPTED_EV)))
    var valid = b_post.load_u32(dsl.u32(UInt32(SOTA_AE_MAGIC_WORD))).eq(SOTA_AE_MAGIC) & (~snap)
    var prev_metered = b_post.load_f32(dsl.u32(UInt32(SOTA_AE_METERED_EV)))
    var has_hist = w_sum.get().gt(0.5)
    var metered_now = ev_sum.get() / max_f(w_sum.get(), 1.0e-6)
    # An empty histogram (zero-size frame) keeps the previous reading.
    var metered = select_f(has_hist, metered_now, select_f(valid, prev_metered, dsl.f32(12.0)))

    var t_n = clamp_f((metered - night_ev) / max_f(day_ev - night_ev, 1.0e-3), 0.0, 1.0)
    var nf = dsl.f32(1.0) - t_n * t_n * (dsl.f32(3.0) - t_n * Float32(2.0))
    var comp = night_comp * nf
    var target = clamp_f(metered, ev_lo, ev_hi) - comp - user_comp

    var delta = target - prev_ev
    var t_p = clamp_f((prev_ev - night_ev) / max_f(day_ev - night_ev, 1.0e-3), 0.0, 1.0)
    var nf_prev = dsl.f32(1.0) - t_p * t_p * (dsl.f32(3.0) - t_p * Float32(2.0))
    var tau_dark = tau_d + (tau_dn - tau_d) * nf_prev
    var tau = select_f(delta.gt(0.0), tau_b, tau_dark)
    var a = dsl.f32(1.0) - exp2_f(dt * (-_LOG2_E) / tau)
    var step = delta * a
    var cap = max_rate * dt
    step = select_f(max_rate.gt(0.0), clamp_f(step, dsl.f32(0.0) - cap, cap), step)
    var next_ev = select_f(valid, prev_ev + step, target)

    # Adaptation luminance for the mesopic blend: the scene mean the adapted setting corresponds to.
    var adapt_nits = exp2_f(next_ev + comp + user_comp) * Float32(0.125)
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_ADAPTED_EV)), next_ev)
    b_post.store_u32(dsl.u32(UInt32(SOTA_AE_MAGIC_WORD)), dsl.u32(SOTA_AE_MAGIC))
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_METERED_EV)), metered)
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_TARGET_EV)), target)
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_TOTAL_W)), total)
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_COMP_EV)), comp)
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_ADAPT_NITS)), adapt_nits)
    b_post.store_f32(dsl.u32(UInt32(SOTA_AE_NIGHT_F)), nf_prev)
    b_post.store_f32(dsl.u32(UInt32(_POST_ADAPTED_EXP_WORD)), exp2_f(ev_unity - next_ev))
    b_post.store_f32(dsl.u32(UInt32(_POST_AVG_LUMA_WORD)), exp2_f(metered) * Float32(0.125) / k_nits)
    return dsl.finish()


def sota_ae_purkinje_shader() -> List[UInt32]:
    """Mesopic / scotopic vision on the scene-referred HDR frame, in place (8 words / px, rgb only).
    Push: 0 w, 1 h, 2 strength, 3 log2 mesopic lo nits, 4 log2 mesopic hi nits, 5..7 night tint."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))
    var strength = clamp_f(dsl.push_f32(2), 0.0, 1.0)
    var lo = dsl.push_f32(3)
    var hi = dsl.push_f32(4)
    var tint = dsl.vec3(dsl.push_f32(5), dsl.push_f32(6), dsl.push_f32(7))
    var la = log2_f(max_f(b_post.load_f32(dsl.u32(UInt32(SOTA_AE_ADAPT_NITS))), 1.0e-12))
    var t = clamp_f((la - lo) / max_f(hi - lo, 1.0e-6), 0.0, 1.0)
    var sig = t * t * (dsl.f32(3.0) - t * Float32(2.0))
    var photopic = dsl.f32(1.0) - strength * (dsl.f32(1.0) - sig)
    var base = b_hdr.base_word(py * w_u + px)
    var rgb = b_hdr.load_vec3(base)
    var x = dot(rgb, dsl.vec3_const(0.4124, 0.3576, 0.1805))
    var y = dot(rgb, dsl.vec3_const(0.2126, 0.7152, 0.0722))
    var z = dot(rgb, dsl.vec3_const(0.0193, 0.1192, 0.9505))
    var ys = max_f(y * (((y + z) / max_f(x, 1.0e-9) + Float32(1.0)) * Float32(1.33) - Float32(1.68)), 0.0) * Float32(1.0 / 2.5733)
    var out = rgb * photopic + tint * (ys * (dsl.f32(1.0) - photopic))
    b_hdr.store_vec3(base, out)
    return dsl.finish()


# ----------------------------------------------------------------------------------------------
# Recording (replaces the v5 histogram + adapt dispatches when active)
# ----------------------------------------------------------------------------------------------

def _fb(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def sota_ae_record(
    mut ae: SotaAutoExposure,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    out_w: Int,
    out_h: Int,
    dt_s: Float32,
    user_comp_ev: Float32,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((String("SotaAutoExposure"), q))
    var ph = List[UInt32](length=16, fill=0)
    ph[0] = UInt32(out_w)
    ph[1] = UInt32(out_h)
    ph[2] = _fb(ae.nits_per_unit)
    ph[3] = _fb(ae.center_weight)
    var step = max(1, min(4, ae.sample_step))
    ph[4] = UInt32(step)
    var sw = (out_w + step - 1) // step
    var shh = (out_h + step - 1) // step
    var rows = (shh + SOTA_AE_RUN - 1) // SOTA_AE_RUN
    vk.dispatch_aaa(slot, ae.pipe_hist, pipe_layout, desc_set, ph, (sw + 15) // 16, (rows + 15) // 16)
    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
    var pa = List[UInt32](length=16, fill=0)
    pa[0] = _fb(dt_s)
    pa[1] = _fb(ae.tau_brighten_s)
    pa[2] = _fb(ae.tau_darken_s)
    pa[3] = _fb(ae.tau_darken_night_s)
    pa[4] = _fb(ae.low_percentile)
    pa[5] = _fb(ae.high_percentile)
    pa[6] = _fb(ae.min_ev100)
    pa[7] = _fb(ae.max_ev100)
    pa[8] = _fb(ae.night_ev100)
    pa[9] = _fb(ae.day_ev100)
    pa[10] = _fb(ae.night_comp_ev)
    pa[11] = _fb(user_comp_ev)
    pa[12] = _fb(ae.ev100_unity())
    pa[13] = UInt32(1) if ae.snap_next else UInt32(0)
    pa[14] = _fb(ae.max_rate_ev_s)
    pa[15] = _fb(ae.nits_per_unit)
    vk.dispatch_aaa(slot, ae.pipe_adapt, pipe_layout, desc_set, pa, 1, 1)
    ae.snap_next = False
    if ae.purkinje:
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        var pp = List[UInt32](length=16, fill=0)
        pp[0] = UInt32(out_w)
        pp[1] = UInt32(out_h)
        pp[2] = _fb(ae.purkinje_strength)
        pp[3] = _fb(log2(max(Float32(1.0e-9), ae.mesopic_lo_nits)))
        pp[4] = _fb(log2(max(Float32(1.0e-9), ae.mesopic_hi_nits)))
        var tl = max(Float32(1.0e-3), Float32(0.2126) * ae.night_tint.x + Float32(0.7152) * ae.night_tint.y + Float32(0.0722) * ae.night_tint.z)
        pp[5] = _fb(ae.night_tint.x / tl)
        pp[6] = _fb(ae.night_tint.y / tl)
        pp[7] = _fb(ae.night_tint.z / tl)
        vk.dispatch_aaa(slot, ae.pipe_purkinje, pipe_layout, desc_set, pp, (out_w + 7) // 8, (out_h + 7) // 8)
    vk.memory_barrier2(
        slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE | PIPELINE_HOST, Int(ACCESS_SHADER_READ | ACCESS_HOST_READ)
    )
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)
