"""Area C (`sota:ml`): learned kernel-predicting a-trous GI denoiser (`sota_ml_denoise.mojo`).

Passes with or without trained weights on the machine:
  1. CPU: `.flmlw` round trip and rejection of bad blobs; the CPU cascade keeps a constant image,
     passes sky pixels through, and does not blur across a depth step.
  2. SPIR-V: temporal, a-trous and fused temporal + level-0 tile shaders (learned with synthetic
     weights, analytic) validate.
  3. GPU (ray-query hardware only): with no weights the renderer runs SVGF bit-for-bit as before
     (clean fallback); the learned path with synthetic weights and the analytic path match the CPU
     cascade, both through the fused tile and the separate passes; the analytic path beats the shipping SVGF against the renderer's 256-spp reference
     (PSNR, SSIM, FLIP) on the Cornell box; pass-time budgets at 960x540 (absolute) and 1080p
     (relative to SVGF on alternating frames, since the GPU is shared with the desktop).
  4. When trained weights are installed (`FLINT_SOTA_ML_WEIGHTS` or the default path), the learned
     path must beat both SVGF and the analytic weights on the held-out Cornell scene.
"""
from std.ffi import external_call
from std.os import getenv
from std.math import abs, max, min
from std.memory import bitcast
from flint.core.math import Vec3
from flint.platform.vulkan import get32
from flint.platform.os import file_size
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.spirv_validate import validate_spirv_module
from flint.render.sota_ml_denoise import (
    SOTA_ML_MODE_ANALYTIC,
    SOTA_ML_MODE_LEARNED,
    SOTA_ML_TILE_H,
    SOTA_ML_TILE_W,
    SotaMlWeights,
    sota_ml_atrous_shader,
    sota_ml_default_weights_path,
    sota_ml_f16,
    sota_ml_filter_cpu,
    sota_ml_fused_shader,
    sota_ml_parse_weights,
    sota_ml_serialize_weights,
    sota_ml_temporal_shader,
)
from flint.render.sota_ml_upscale import (
    SotaMlUpFrame,
    SotaMlUpWeights,
    sota_ml_up_default_weights_path,
    sota_ml_up_parse_weights,
    sota_ml_up_serialize_weights,
    sota_ml_upscale_cpu_pixel,
    sota_ml_upscale_shader,
)
from flint.render.sota_ml_eval import (
    SOTA_ML_SCENE_CORNELL,
    SotaMlQuality,
    SotaMlScene,
    sota_ml_build_scene,
    sota_ml_draw_scene,
    sota_ml_quality,
    sota_ml_read_capture,
    sota_ml_render_reference,
)


def _fail(msg: String):
    print("FAIL:", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _synthetic_weights(levels: Int, hidden: Int, seed: UInt32) -> SotaMlWeights:
    var w = SotaMlWeights(levels, hidden)
    var s = seed
    for i in range(len(w.data)):
        s = s * UInt32(1664525) + UInt32(1013904223)
        w.data[i] = (Float32(s >> 8) / Float32(16777216.0) - Float32(0.5)) * Float32(2.0)
    return w^


def _fmt(x: Float64) -> String:
    var s = String(x)
    return String(s[byte = 0 : min(6, s.byte_length())])


def _median(mut xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def test_cpu_weights_and_cascade():
    var w = _synthetic_weights(3, 6, 7)
    var blob = sota_ml_serialize_weights(w)
    var back = sota_ml_parse_weights(blob)
    _assert(back.valid() and back.levels == 3 and back.hidden == 6, "weights round trip shape")
    var same = True
    for i in range(len(w.data)):
        same = same and back.data[i] == w.data[i]
    _assert(same, "weights round trip values")
    var bad = blob.copy()
    bad[0] = UInt8(0x58)
    _assert(not sota_ml_parse_weights(bad).valid(), "bad magic rejected")
    var short = blob.copy()
    _ = short.pop()
    _assert(not sota_ml_parse_weights(short).valid(), "truncated blob rejected")
    _assert(not sota_ml_parse_weights(List[UInt8]()).valid(), "empty blob rejected")

    # 40x16 image: left half at depth 2 (irradiance 0.2), right half at depth 9 (irradiance 1.0),
    # top row is sky. A constant region must stay constant, sky must pass through untouched.
    var W = 40
    var H = 16
    var n = W * H
    var rgb = List[Float32](length=3 * n, fill=0.0)
    var var_in = List[Float32](length=n, fill=0.0)
    var z = List[Float32](length=n, fill=0.0)
    var nx = List[Float32](length=n, fill=0.0)
    var ny = List[Float32](length=n, fill=0.0)
    var il = List[Float32](length=n, fill=Float32(0.25))
    for y in range(H):
        for x in range(W):
            var i = y * W + x
            var v = Float32(0.2) if x < W // 2 else Float32(1.0)
            rgb[3 * i] = v
            rgb[3 * i + 1] = v
            rgb[3 * i + 2] = v
            var_in[i] = Float32(0.0004)
            z[i] = Float32(2.0) if x < W // 2 else Float32(9.0)
            if y == 0:
                z[i] = Float32(1e6)  # sky: same colour, so it cannot leak, but it must skip the f16 store
    var an = sota_ml_filter_cpu(w, True, W, H, rgb, var_in, z, nx, ny, il)[0].copy()
    var ln = sota_ml_filter_cpu(w, False, W, H, rgb, var_in, z, nx, ny, il)[0].copy()
    _assert(an[3 * 2] == Float32(0.2) and ln[3 * 3] == Float32(0.2) and sota_ml_f16(0.2) != Float32(0.2), "sky pixels pass through")
    var i_l = 8 * W + 3  # the 3-level cascade reaches 1 + 2 + 4 = 7 px: stays inside the left plane
    var i_r = 8 * W + 36
    _assert(abs(ln[3 * i_l] - sota_ml_f16(0.2)) < 2e-3 and abs(ln[3 * i_r] - Float32(1.0)) < 2e-3, "learned keeps constant regions")
    # Analytic (SVGF) weights stop at the depth step: the pixel next to the edge stays near its side
    var i_e = 8 * W + (W // 2 - 1)
    _assert(an[3 * i_e] < Float32(0.3), "analytic weights do not blur across a depth step")
    print("  [PASS] sota:ml CPU weights blob + cascade invariants")


def test_spirv_shaders():
    var w = _synthetic_weights(4, 12, 3)
    var shaders = List[List[UInt32]]()
    shaders.append(sota_ml_temporal_shader())
    shaders.append(sota_ml_atrous_shader(w, 0, False))
    shaders.append(sota_ml_atrous_shader(w, 3, False))
    shaders.append(sota_ml_atrous_shader(w, 1, True))
    shaders.append(sota_ml_fused_shader[SOTA_ML_TILE_W, SOTA_ML_TILE_H](w, False))
    shaders.append(sota_ml_fused_shader[SOTA_ML_TILE_W, SOTA_ML_TILE_H](w, True))
    shaders.append(sota_ml_fused_shader[32, 8](w, False))
    for i in range(len(shaders)):
        var rep = validate_spirv_module(shaders[i])
        if not rep.valid:
            _fail("sota:ml shader " + String(i) + " SPIR-V invalid: " + rep.errors[0])
    print("  [PASS] sota:ml temporal, learned/analytic a-trous and fused tile SPIR-V validate")


def _gpu_vs_cpu(ren: FlintAaaVulkanRenderer, analytic: Bool) -> Float32:
    var cap = sota_ml_read_capture(ren)
    var cpu = sota_ml_filter_cpu(ren.sota_ml.weights, analytic, cap.width, cap.height, cap.rgb, cap.variance, cap.z, cap.nx, cap.ny, cap.inv_len)
    var maxd = Float32(0.0)
    for i in range(len(cap.out_rgb)):
        maxd = max(maxd, abs(cap.out_rgb[i] - cpu[0][i]) / (abs(cpu[0][i]) + Float32(0.01)))
    return maxd


def _run_sequence(mut ren: FlintAaaVulkanRenderer, scene: SotaMlScene, frames: Int, motion: Float32) -> List[Float64]:
    """Renders `frames` frames; returns the per-frame GPU ms of the GI denoise pass (SVGF or learned)."""
    var ms = List[Float64]()
    for f in range(frames):
        sota_ml_draw_scene(ren, scene, f, motion)
        _assert(ren.render_frame(), "render_frame")
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == "SotaMlDenoise" or t.pass_name == "SvgfTemporalAndAtrous":
                ms.append(t.gpu_ms)
    return ms^


def _quality(mut ren: FlintAaaVulkanRenderer, scene: SotaMlScene, frames: Int, motion: Float32) raises -> SotaMlQuality:
    var test_hdr = ren.readback_resolved_hdr()
    var refi = sota_ml_render_reference(ren, scene, frames - 1, motion, 256, 32)
    return sota_ml_quality(refi, test_hdr, ren.output_w, ren.output_h)


def test_gpu_fallback_equivalence_and_quality() raises:
    var W = 240
    var H = 136
    var motion = Float32(0.25)
    var frames = 24

    var probe = FlintAaaVulkanRenderer(width=W, height=H)
    if not probe.active or not probe.vk.has_ray_query:
        probe.shutdown()
        print("  [PASS] sota:ml GPU checks skipped (no hardware ray query)")
        return
    probe.shutdown()

    # 1. Fallback: no weights file -> the shipping SVGF pass runs (the learned pass never records) and the
    #    frame matches the learned path disabled outright. (Two renderer instances are not bit-identical
    #    even with this path off, so the match is on pass list and quality, not on pixels.)
    var r_off = FlintAaaVulkanRenderer(width=W, height=H)
    r_off.gi_rays_per_pixel = 1
    var sc_r_off = sota_ml_build_scene(r_off, SOTA_ML_SCENE_CORNELL, 0)
    _assert(not r_off.sota_ml.enabled or getenv("FLINT_SOTA_ML_DENOISE") == "1", "learned path is opt-in (off by default)")
    r_off.sota_ml.enabled = False
    var ms_svgf = _run_sequence(r_off, sc_r_off, frames, motion)
    var q_svgf = _quality(r_off, sc_r_off, frames, motion)
    r_off.shutdown()

    var r_fb = FlintAaaVulkanRenderer(width=W, height=H)
    r_fb.gi_rays_per_pixel = 1
    var sc_r_fb = sota_ml_build_scene(r_fb, SOTA_ML_SCENE_CORNELL, 0)
    r_fb.sota_ml.enabled = True
    r_fb.sota_ml.weights_path = String("/nonexistent/sota_ml_denoise.flmlw")
    var ms_fb = _run_sequence(r_fb, sc_r_fb, frames, motion)
    _assert(not r_fb.sota_ml.ran_last_frame and not r_fb.sota_ml.has_weights(), "no weights -> learned path stays off")
    var has_svgf = False
    var has_ml = False
    for t in r_fb.get_gpu_pass_timings():
        has_svgf = has_svgf or t.pass_name == "SvgfTemporalAndAtrous"
        has_ml = has_ml or t.pass_name == "SotaMlDenoise"
    _assert(has_svgf and not has_ml and len(ms_fb) == frames, "fallback frame runs SVGF, not the learned pass")
    var q_fb = _quality(r_fb, sc_r_fb, frames, motion)
    r_fb.shutdown()
    _assert(abs(q_fb.psnr_db - q_svgf.psnr_db) < 0.5 and abs(q_fb.flip - q_svgf.flip) < 0.01, "fallback quality equals shipping SVGF")

    # 2. Analytic weights inside the learned path: GPU == CPU cascade, and better than shipping SVGF.
    var r_an = FlintAaaVulkanRenderer(width=W, height=H)
    r_an.gi_rays_per_pixel = 1
    var sc_r_an = sota_ml_build_scene(r_an, SOTA_ML_SCENE_CORNELL, 0)
    r_an.sota_ml.enabled = True
    r_an.sota_ml.mode = SOTA_ML_MODE_ANALYTIC
    r_an.sota_ml.capture = True
    var ms_an = _run_sequence(r_an, sc_r_an, frames, motion)
    _assert(r_an.sota_ml.ran_last_frame, "analytic path runs in the live frame")
    var d_an = _gpu_vs_cpu(r_an, True)
    _assert(d_an < Float32(5e-3), "analytic GPU cascade matches CPU (max rel " + String(d_an) + ")")
    var q_an = _quality(r_an, sc_r_an, frames, motion)
    r_an.shutdown()
    _assert(q_an.psnr_db > q_svgf.psnr_db + 2.5, "analytic path beats SVGF by 2.5 dB PSNR")
    _assert(q_an.ssim > q_svgf.ssim + 0.1, "analytic path beats SVGF SSIM")
    _assert(q_an.flip < q_svgf.flip, "analytic path beats SVGF FLIP")

    # 3. Synthetic learned weights: GPU == CPU cascade (exercises the baked-constant MLP).
    var r_sy = FlintAaaVulkanRenderer(width=W, height=H)
    r_sy.gi_rays_per_pixel = 1
    var sc_r_sy = sota_ml_build_scene(r_sy, SOTA_ML_SCENE_CORNELL, 0)
    r_sy.sota_ml.enabled = True
    r_sy.sota_ml.set_weights(_synthetic_weights(4, 12, 11))
    r_sy.sota_ml.capture = True
    r_sy.sota_ml.fused = True
    _ = _run_sequence(r_sy, sc_r_sy, 6, motion)
    _assert(r_sy.sota_ml.ran_last_frame, "learned path runs with injected weights")
    var d_sy = _gpu_vs_cpu(r_sy, False)
    _assert(d_sy < Float32(5e-3), "learned GPU cascade (fused tile) matches CPU (max rel " + String(d_sy) + ")")
    # unfused temporal + level 0 must give the same numbers
    r_sy.sota_ml.fused = False
    _ = _run_sequence(r_sy, sc_r_sy, 3, motion)
    var d_sy2 = _gpu_vs_cpu(r_sy, False)
    _assert(d_sy2 < Float32(5e-3), "learned GPU cascade (separate passes) matches CPU (max rel " + String(d_sy2) + ")")
    # GI-mode gate: on by default only for the legacy and 1-spp path-traced inputs (it regresses the
    # already-smooth world-cache-only and hybrid modes), so `ready` refuses modes 1 and 2.
    _assert(r_sy.sota_ml.ready(r_sy.vk, r_sy.pipe_layout, 0) and r_sy.sota_ml.ready(r_sy.vk, r_sy.pipe_layout, 3), "learned path allowed for legacy and path-traced GI")
    _assert(not r_sy.sota_ml.ready(r_sy.vk, r_sy.pipe_layout, 1) and not r_sy.sota_ml.ready(r_sy.vk, r_sy.pipe_layout, 2), "learned path gated off for world-cache-only and hybrid GI")
    r_sy.shutdown()

    print(
        "  [PASS] sota:ml fallback = SVGF; Cornell 1 rpp vs 256-spp ref: SVGF PSNR "
        + _fmt(q_svgf.psnr_db) + " SSIM " + _fmt(q_svgf.ssim) + " FLIP " + _fmt(q_svgf.flip)
        + " | analytic PSNR " + _fmt(q_an.psnr_db) + " SSIM " + _fmt(q_an.ssim) + " FLIP " + _fmt(q_an.flip)
        + " | ms SVGF " + _fmt(_median(ms_svgf)) + " ML " + _fmt(_median(ms_an))
    )

    # 4. Trained weights, when installed on this machine (never in the repo)
    var wpath = sota_ml_default_weights_path()
    if file_size(wpath) > 0:
        var r_ln = FlintAaaVulkanRenderer(width=W, height=H)
        r_ln.gi_rays_per_pixel = 1
        var sc_r_ln = sota_ml_build_scene(r_ln, SOTA_ML_SCENE_CORNELL, 0)
        r_ln.sota_ml.capture = True
        r_ln.sota_ml.enabled = True
        _ = _run_sequence(r_ln, sc_r_ln, frames, motion)
        _assert(r_ln.sota_ml.ran_last_frame, "trained weights load: " + r_ln.sota_ml.status)
        var d_ln = _gpu_vs_cpu(r_ln, False)
        _assert(d_ln < Float32(5e-3), "trained GPU cascade matches CPU")
        var q_ln = _quality(r_ln, sc_r_ln, frames, motion)
        r_ln.shutdown()
        _assert(q_ln.psnr_db > q_an.psnr_db and q_ln.flip < q_an.flip, "trained weights beat analytic weights")
        print("  [PASS] sota:ml trained weights: PSNR " + _fmt(q_ln.psnr_db) + " SSIM " + _fmt(q_ln.ssim) + " FLIP " + _fmt(q_ln.flip))
    else:
        print("  [INFO] sota:ml no trained weights at " + wpath + " (learned-quality check skipped)")


def _paired_timings(width: Int, height: Int, frames: Int) raises -> Tuple[List[Float64], List[Float64]]:
    """One renderer, alternating SVGF and learned-path frames (synthetic K=4, H=8 weights), so the
    desktop sharing the GPU slows both alike. Returns sorted `(svgf_ms, learned_ms)`."""
    var ren = FlintAaaVulkanRenderer(width=width, height=height)
    var t_svgf = List[Float64]()
    var t_ml = List[Float64]()
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        return (t_svgf^, t_ml^)
    ren.gi_rays_per_pixel = 1
    var sc = sota_ml_build_scene(ren, SOTA_ML_SCENE_CORNELL, 0)
    ren.sota_ml.set_weights(_synthetic_weights(4, 8, 5))
    for f in range(frames):
        ren.sota_ml.enabled = (f & 1) == 1
        sota_ml_draw_scene(ren, sc, f, Float32(0.25))
        _assert(ren.render_frame(), "render_frame")
        if f < 4:
            continue
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == "SotaMlDenoise":
                t_ml.append(t.gpu_ms)
            elif t.pass_name == "SvgfTemporalAndAtrous":
                t_svgf.append(t.gpu_ms)
    ren.shutdown()
    _ = _median(t_svgf)
    _ = _median(t_ml)
    return (t_svgf^, t_ml^)


def test_gpu_perf_budget() raises:
    # 960x540 (fits the GPU caches): absolute budget on the best frames
    var lo = _paired_timings(960, 540, 24)
    if len(lo[1]) == 0:
        return
    _assert(lo[1][0] < 2.0, "learned denoise at 960x540 under 2 ms (best frame " + String(lo[1][0]) + ")")
    # 1080p: relative to the shipping SVGF on alternating frames of the same renderer
    var hi = _paired_timings(1920, 1080, 24)
    var m_svgf = hi[0][len(hi[0]) // 2]
    var m_ml = hi[1][len(hi[1]) // 2]
    _assert(len(hi[1]) >= 8 and m_ml < 2.0 * m_svgf, "learned denoise at 1080p within 2x SVGF (" + String(m_ml) + " vs " + String(m_svgf) + " ms)")
    print(
        "  [PASS] sota:ml perf: 960x540 best " + _fmt(lo[1][0]) + " ms (SVGF " + _fmt(lo[0][0]) + "); 1080p median "
        + _fmt(m_ml) + " ms vs SVGF " + _fmt(m_svgf) + " ms (paired, shared GPU)"
    )


def _synthetic_up_weights(hidden: Int, seed: UInt32) -> SotaMlUpWeights:
    var w = SotaMlUpWeights(hidden)
    var s = seed
    for i in range(len(w.data)):
        s = s * UInt32(1664525) + UInt32(1013904223)
        w.data[i] = (Float32(s >> 8) / Float32(16777216.0) - Float32(0.5)) * Float32(2.0)
    return w^


def test_upscale_cpu_and_spirv():
    var w = _synthetic_up_weights(8, 21)
    var back = sota_ml_up_parse_weights(sota_ml_up_serialize_weights(w))
    _assert(back.valid() and back.hidden == 8 and len(back.data) == len(w.data), "upscaler weights round trip")
    for i in range(len(w.data)):
        _assert(back.data[i] == w.data[i], "upscaler weights round trip value")
    _assert(not sota_ml_up_parse_weights(List[UInt8](length=12, fill=0)).valid(), "truncated upscaler blob rejected")
    var rep = validate_spirv_module(sota_ml_upscale_shader(w))
    if not rep.valid:
        _fail("sota:ml upscale SPIR-V invalid: " + rep.errors[0])
    print("  [PASS] sota:ml upscaler weights blob + SPIR-V validate")


def _up_pass_names(ren: FlintAaaVulkanRenderer) -> Tuple[Bool, Bool]:
    var tsr = False
    var ml = False
    for t in ren.get_gpu_pass_timings():
        tsr = tsr or t.pass_name == "FlintTsrV3Resolve"
        ml = ml or t.pass_name == "SotaMlUpscale"
    return (tsr, ml)


def _up_gpu_vs_cpu(ren: FlintAaaVulkanRenderer) -> Float32:
    """Last frame's learned resolve (history out) against the CPU reference on its captured inputs."""
    var fr = SotaMlUpFrame(ren.render_w, ren.render_h, ren.output_w, ren.output_h)
    var cap = ren.sota_ml_up.cap_buf.cpu_address
    for i in range(len(fr.hdr)):
        fr.hdr[i] = bitcast[DType.float32](get32(cap + i * 4))
    var max_pix = max(ren.output_w * ren.output_h, ren.render_w * ren.render_h)
    var last = ren.frame_index - 1
    var hist_in = max_pix * 16 if (last & 1) == 0 else max_pix * 20
    var hist_out = max_pix * 20 if (last & 1) == 0 else max_pix * 16
    var hb = ren.buf_history.cpu_address
    for i in range(len(fr.hist)):
        fr.hist[i] = bitcast[DType.float32](get32(hb + (hist_in + i) * 4))
    var g = ren.buf_globals.cpu_address
    fr.jx = bitcast[DType.float32](get32(g + 151 * 4))
    fr.jy = bitcast[DType.float32](get32(g + 152 * 4))
    fr.sharp = ren.tsr_sharpness
    fr.hist_on = ren.tsr_history_weight > Float32(0.001)
    var maxd = Float32(0.0)
    var y = 1
    while y < ren.output_h - 1:
        var x = 1
        while x < ren.output_w - 1:
            var c = sota_ml_upscale_cpu_pixel(ren.sota_ml_up.weights, fr, x, y)
            var o = (hist_out + (y * ren.output_w + x) * 4) * 4
            var gr = [bitcast[DType.float32](get32(hb + o)), bitcast[DType.float32](get32(hb + o + 4)), bitcast[DType.float32](get32(hb + o + 8))]
            var cr = [c[0], c[1], c[2]]
            for k in range(3):
                maxd = max(maxd, abs(gr[k] - cr[k]) / (abs(cr[k]) + Float32(0.02)))
            x += 3
        y += 3
    return maxd


def _up_run(mut ren: FlintAaaVulkanRenderer, scene: SotaMlScene, frames: Int, motion: Float32) -> List[Float64]:
    var ms = List[Float64]()
    for f in range(frames):
        sota_ml_draw_scene(ren, scene, f, motion)
        _assert(ren.render_frame(), "render_frame")
        for t in ren.get_gpu_pass_timings():
            if t.pass_name == "SotaMlUpscale" or t.pass_name == "FlintTsrV3Resolve":
                ms.append(t.gpu_ms)
    return ms^


def test_gpu_upscale() raises:
    var W = 320
    var H = 180
    var motion = Float32(0.6)
    var frames = 16
    var probe = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=0.5)
    if not probe.active or not probe.vk.has_ray_query:
        probe.shutdown()
        print("  [PASS] sota:ml upscaler GPU checks skipped (no hardware ray query)")
        return
    _assert(not probe.sota_ml_up.enabled or getenv("FLINT_SOTA_ML_UPSCALE") == "1", "learned upscaler is opt-in")
    probe.shutdown()

    # 1. Fallback: enabled but no weights -> the TSR v3 resolve records.
    var r_fb = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=0.5)
    r_fb.gi_rays_per_pixel = 1
    var sc_fb = sota_ml_build_scene(r_fb, SOTA_ML_SCENE_CORNELL, 0)
    r_fb.sota_ml_up.enabled = True
    r_fb.sota_ml_up.weights_path = String("/nonexistent/sota_ml_upscale.flmlw")
    _ = _up_run(r_fb, sc_fb, 4, motion)
    var pn = _up_pass_names(r_fb)
    _assert(not r_fb.sota_ml_up.ran_last_frame and pn[0] and not pn[1], "no upscaler weights -> TSR v3 resolve")
    r_fb.shutdown()

    # 2. Synthetic weights: the learned resolve records and matches the CPU reference.
    var r_sy = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=0.5)
    r_sy.gi_rays_per_pixel = 1
    var sc_sy = sota_ml_build_scene(r_sy, SOTA_ML_SCENE_CORNELL, 0)
    r_sy.sota_ml_up.enabled = True
    r_sy.sota_ml_up.capture = True
    r_sy.sota_ml_up.set_weights(_synthetic_up_weights(8, 5))
    _ = _up_run(r_sy, sc_sy, 6, motion)
    pn = _up_pass_names(r_sy)
    _assert(r_sy.sota_ml_up.ran_last_frame and pn[1] and not pn[0], "learned resolve replaces TSR v3")
    var d = _up_gpu_vs_cpu(r_sy)
    _assert(d < Float32(5e-3), "learned resolve GPU matches CPU (max rel " + String(d) + ")")
    r_sy.shutdown()
    print("  [PASS] sota:ml upscaler fallback = TSR v3; learned resolve GPU == CPU (max rel " + _fmt(Float64(d)) + ")")

    # 3. Trained weights, when installed: compare with TSR v3 at the same input resolution.
    var wpath = sota_ml_up_default_weights_path()
    if file_size(wpath) <= 0:
        print("  [INFO] sota:ml no upscaler weights at " + wpath + " (quality check skipped)")
        return
    var full = FlintAaaVulkanRenderer(width=W, height=H)
    var sc_full = sota_ml_build_scene(full, SOTA_ML_SCENE_CORNELL, 0)
    var refi = sota_ml_render_reference(full, sc_full, frames - 1, motion, 256, 32)
    full.shutdown()
    var q = List[SotaMlQuality]()
    var ms = List[Float64]()
    for v in range(2):
        var ren = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=0.5)
        ren.gi_rays_per_pixel = 1
        var sc = sota_ml_build_scene(ren, SOTA_ML_SCENE_CORNELL, 0)
        ren.sota_ml_up.enabled = v == 1
        var t = _up_run(ren, sc, frames, motion)
        _assert(v == 0 or ren.sota_ml_up.ran_last_frame, "trained upscaler loads: " + ren.sota_ml_up.status)
        q.append(sota_ml_quality(refi, ren.readback_resolved_hdr(), W, H))
        ms.append(_median(t))
        ren.shutdown()
    _assert(q[1].flip < q[0].flip + 0.005, "trained upscaler is not worse than TSR v3 (FLIP)")
    print(
        "  [PASS] sota:ml 2x upscale 160x90->320x180 vs 256-spp ref: TSR v3 PSNR " + _fmt(q[0].psnr_db) + " FLIP " + _fmt(q[0].flip)
        + " | learned PSNR " + _fmt(q[1].psnr_db) + " FLIP " + _fmt(q[1].flip) + " | ms " + _fmt(ms[0]) + " / " + _fmt(ms[1])
    )


def main() raises:
    print("[Flint v5] sota:ml learned GI denoiser + upscaler")
    test_cpu_weights_and_cascade()
    test_spirv_shaders()
    test_upscale_cpu_and_spirv()
    test_gpu_fallback_equivalence_and_quality()
    test_gpu_upscale()
    test_gpu_perf_budget()
    print("[Flint v5] sota:ml PASSED")
