"""Area C (`sota:ml`): learned kernel-predicting a-trous denoiser for the RT diffuse GI signal.

What it replaces. When trained weights are present, this path replaces `SvgfTemporalAndAtrous` in the
live frame; everything before (deferred shade, which demodulates GI by albedo) and after (ReBLUR, TSR,
post) is unchanged. Without weights the renderer runs SVGF exactly as before (the fallback is the
shipping path, not a second implementation).

Network. A cross-bilateral a-trous cascade (Dammertz et al. 2010; SVGF, Schied et al. 2017) whose
per-tap edge-stopping weight is predicted by a tiny MLP instead of SVGF's hand-tuned
`n^32 * exp(-dz - dl)`. This is the kernel-predicting idea (Bako et al. 2017, Vogels et al. 2018) scaled
down to what fits a real-time frame: the kernel is factorised into `K` dilated 3x3 levels
(steps 1, 2, 4, ...), and each tap weight is

    w = B3(tap) * sigmoid(MLP_k(f)),  MLP_k: 7 -> H (ReLU) -> 1, one MLP per level

with per-tap features `f` (all scale-free; `c` = centre, `n` = neighbour, `l` = luma of the
demodulated irradiance, `var` = temporal luma variance of the centre, `len` = history length):

    f0 = min(|z_c - z_n| / max(0.08 z_c, 0.05), 8)      relative depth step
    f1 = 1 - max(n_c . n_n, 0)                         normal change
    f2 = min(|l_c - l_n| / (sqrt(var) + 0.01), 16)     luma step in noise standard deviations
    f3 = |l_c - l_n| / (l_c + l_n + 0.01)              relative luma step
    f4 = min(sqrt(var) / (l_c + 0.01), 4)              relative noise level (per pixel)
    f5 = 1 / len                                       history confidence (per pixel)
    f6 = |tap offset| / sqrt(2)                        0, 0.707 or 1

`f4..f6` are constant per pixel or per tap, so their first-layer products fold into the bias; a tap
costs `4H + 2H` FMAs. Variance is propagated through the levels as in SVGF
(`var' = sum w^2 var_n / (sum w)^2`). The temporal stage is SVGF-style reprojection and moments, but
with its own history ping-pong (see the audit: the shipping SVGF clobbers its history every other
frame) and a longer cap (32 frames).

Weights are baked into the SPIR-V as constants when the pipelines are built, so the inference needs no
new descriptor binding: it is plain FMA compute through Flint's `ShaderDsl` emitter. With `K = 4`,
`H = 12` that is about 7 k FLOP per pixel. (`VK_KHR_cooperative_matrix` is offered on the GB10 but Flint's
emitter has no cooperative-matrix types yet; a 7 -> 12 -> 1 MLP per tap does not need tensor cores.)

Training (outside the repo, `~/flint-models/_train/render/`): data is dumped from this very path in
capture mode (temporal output + guides + remodulation terms) and paired with the renderer's own
512-spp reference (`sota_ml_eval.mojo`); PyTorch replays the cascade exactly (`sota_ml_filter_cpu`
below is the Mojo twin, the GPU test checks all three agree).

Weights file (`.flmlw`, little-endian): `"FLMLDN01"`, `u32 K`, `u32 H`, `u32 NF (= 7)`, then per level
`w1[H][NF]`, `b1[H]`, `w2[H]`, `b2`. Located by `FLINT_SOTA_ML_WEIGHTS` or
`$HOME/flint-models/_train/render/weights/sota_ml_denoise_v1.flmlw`. Weights never live in the repo.

Scratch (words of `BIND_AAA_HISTORY`, allocated past every other region by the renderer hook, so the
offsets are recorded at allocation, not recomputed): colour A (temporal output, kept for capture),
colour B and C (level ping-pong), each `2` words/pixel `f16x4 (r, g, b, var)`, and guide G
`3` words/pixel `(z, f16x2 (nx, ny), 1/len)`, plus capture D `3` words/pixel (f16 crisp rgb + albedo rgb, written
only when `capture` is set: the remodulation terms in `BIND_AAA_POST_SCRATCH` are overwritten by TSR). History ping-pong reuses the SVGF history words
`[0, 16 * max_pix)`, which SVGF does not touch on frames where this path runs.
"""
from std.math import exp, max, min, sqrt
from std.memory import bitcast
from flint.platform.os import env_now, read_bytes
from flint.platform.vulkan import ACCESS_SHADER_READ, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, VulkanGpu
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_HDR_OUT, BIND_AAA_HISTORY, BIND_AAA_POST_SCRATCH
from flint.render.shader_dsl import (
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    dot,
    exp2_f,
    floor_f,
    fma_f,
    max_f,
    min_f,
    min_u,
    pow_f,
    select_f,
    select_u,
    sqrt_f,
)

comptime SOTA_ML_NF: Int = 7
# Workgroup shape. Wide rows keep each warp inside one 4 KiB page per buffer row: on the GB10 the large
# per-pixel records of this renderer thrash the GPU TLB at 1080p with 8x8 tiles (see the notes).
comptime SOTA_ML_GROUP_W: Int = 32
comptime SOTA_ML_GROUP_H: Int = 2
comptime SOTA_ML_MODE_LEARNED: Int = 0
comptime SOTA_ML_MODE_ANALYTIC: Int = 1  # SVGF weights inside this path (ablation, and GPU tests without weights)
comptime SOTA_ML_ANALYTIC_LEVELS: Int = 4
comptime SOTA_ML_SCRATCH_WORDS_PER_PIXEL: Int = 12  # A, B, C (2 each) + G (3) + capture D (3)
comptime SOTA_ML_POST_BASE: Int = 1024  # `BIND_AAA_POST_SCRATCH[1024 + 8 * pix]`: crisp rgb, albedo rgb, velocity
comptime SOTA_ML_DEFAULT_WEIGHTS: String = "/flint-models/_train/render/weights/sota_ml_denoise_v1.flmlw"


# ============================================================================
# Weights
# ============================================================================


struct SotaMlWeights(Copyable, Movable):
    """`levels` MLPs `7 -> hidden -> 1`; per level `w1[hidden][7], b1[hidden], w2[hidden], b2`."""
    var levels: Int
    var hidden: Int
    var data: List[Float32]

    def __init__(out self):
        self.levels = 0
        self.hidden = 0
        self.data = List[Float32]()

    def __init__(out self, levels: Int, hidden: Int):
        self.levels = levels
        self.hidden = hidden
        self.data = List[Float32](length=levels * (hidden * (SOTA_ML_NF + 2) + 1), fill=0.0)

    def valid(self) -> Bool:
        return (
            self.levels >= 1
            and self.levels <= 8
            and self.hidden >= 1
            and self.hidden <= 64
            and len(self.data) == self.levels * (self.hidden * (SOTA_ML_NF + 2) + 1)
        )

    def level_base(self, k: Int) -> Int:
        return k * (self.hidden * (SOTA_ML_NF + 2) + 1)

    def w1(self, k: Int, j: Int, f: Int) -> Float32:
        return self.data[self.level_base(k) + j * SOTA_ML_NF + f]

    def b1(self, k: Int, j: Int) -> Float32:
        return self.data[self.level_base(k) + self.hidden * SOTA_ML_NF + j]

    def w2(self, k: Int, j: Int) -> Float32:
        return self.data[self.level_base(k) + self.hidden * (SOTA_ML_NF + 1) + j]

    def b2(self, k: Int) -> Float32:
        return self.data[self.level_base(k) + self.hidden * (SOTA_ML_NF + 2)]


def _u32_at(data: List[UInt8], at: Int) -> UInt32:
    return UInt32(data[at]) | (UInt32(data[at + 1]) << 8) | (UInt32(data[at + 2]) << 16) | (UInt32(data[at + 3]) << 24)


def sota_ml_parse_weights(data: List[UInt8]) -> SotaMlWeights:
    """Parses a `.flmlw` blob; returns an invalid (`levels == 0`) set on any mismatch."""
    if len(data) < 20:
        return SotaMlWeights()
    var magic = "FLMLDN01"
    for i in range(8):
        if data[i] != magic.as_bytes()[i]:
            return SotaMlWeights()
    var k = Int(_u32_at(data, 8))
    var h = Int(_u32_at(data, 12))
    var nf = Int(_u32_at(data, 16))
    if nf != SOTA_ML_NF or k < 1 or k > 8 or h < 1 or h > 64:
        return SotaMlWeights()
    var w = SotaMlWeights(k, h)
    if len(data) != 20 + len(w.data) * 4:
        return SotaMlWeights()
    for i in range(len(w.data)):
        w.data[i] = bitcast[DType.float32](_u32_at(data, 20 + i * 4))
    return w^


def sota_ml_serialize_weights(w: SotaMlWeights) -> List[UInt8]:
    var out = List[UInt8]()
    for b in "FLMLDN01".as_bytes():
        out.append(b)
    for v in [UInt32(w.levels), UInt32(w.hidden), UInt32(SOTA_ML_NF)]:
        for s in range(4):
            out.append(UInt8((v >> UInt32(8 * s)) & 0xFF))
    for i in range(len(w.data)):
        var u = bitcast[DType.uint32](w.data[i])
        for s in range(4):
            out.append(UInt8((u >> UInt32(8 * s)) & 0xFF))
    return out^


def sota_ml_load_weights(path: String) -> SotaMlWeights:
    return sota_ml_parse_weights(read_bytes(path))


def sota_ml_default_weights_path() -> String:
    var p = env_now("FLINT_SOTA_ML_WEIGHTS")
    if p.byte_length() > 0:
        return p
    return env_now("HOME") + SOTA_ML_DEFAULT_WEIGHTS


# ============================================================================
# CPU reference of the spatial cascade (bit-for-bit structure of the GPU shader and the PyTorch trainer)
# ============================================================================


def sota_ml_f16(x: Float32) -> Float32:
    """Round through IEEE half (what `packHalf2x16` stores), clamped to the finite range like `pack_f16x2`."""
    var c = max(Float32(-65504.0), min(Float32(65504.0), x))
    return c.cast[DType.float16]().cast[DType.float32]()


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


def sota_ml_tap_weight_cpu(
    w: SotaMlWeights, k: Int, analytic: Bool, sigma_luma: Float32, f: Array[Float32, 7], ndot: Float32, dz: Float32, dl_raw: Float32, std_c: Float32
) -> Float32:
    """Edge-stopping factor of one tap (without the B3 spatial factor)."""
    if analytic:
        var s_scaled = sigma_luma * std_c + Float32(0.001)
        var dl = dl_raw / max(s_scaled, Float32(0.01))
        return exp(Float32(32.0) * _safe_log(max(ndot, Float32(0.0)))) * exp(-(dz + dl))
    var o = w.b2(k)
    for j in range(w.hidden):
        var pre = w.b1(k, j)
        for q in range(SOTA_ML_NF):
            pre += w.w1(k, j, q) * f[q]
        o += w.w2(k, j) * max(pre, Float32(0.0))
    return _sigmoid(o)


def _safe_log(x: Float32) -> Float32:
    from std.math import log

    if x <= Float32(0.0):
        return Float32(-1.0e30)
    return log(x)


def sota_ml_filter_cpu(
    w: SotaMlWeights,
    analytic: Bool,
    width: Int,
    height: Int,
    rgb_in: List[Float32],
    var_in: List[Float32],
    z: List[Float32],
    nx: List[Float32],
    ny: List[Float32],
    inv_len: List[Float32],
) -> Tuple[List[Float32], List[Float32]]:
    """Runs the `K`-level cascade on f16-rounded inputs (`rgb_in` is `3 * w * h`), storing each level's
    output through f16 like the GPU. Returns the filtered demodulated `(rgb, var)`."""
    var n = width * height
    var levels = SOTA_ML_ANALYTIC_LEVELS if analytic else w.levels
    var cur_rgb = rgb_in.copy()
    var cur_var = var_in.copy()
    var kw1 = [Float32(0.25), Float32(0.5), Float32(0.25)]
    for k in range(levels):
        var step = 1 << k
        var out_rgb = List[Float32](length=3 * n, fill=0.0)
        var out_var = List[Float32](length=n, fill=0.0)
        for py in range(height):
            for px in range(width):
                var i = py * width + px
                var cz = z[i]
                if cz > Float32(1e5):
                    out_rgb[3 * i] = cur_rgb[3 * i]
                    out_rgb[3 * i + 1] = cur_rgb[3 * i + 1]
                    out_rgb[3 * i + 2] = cur_rgb[3 * i + 2]
                    out_var[i] = cur_var[i]
                    continue
                var cnz = sqrt(max(Float32(1.0) - (nx[i] * nx[i] + ny[i] * ny[i]), Float32(0.0)))
                var cl = Float32(0.2126) * cur_rgb[3 * i] + Float32(0.7152) * cur_rgb[3 * i + 1] + Float32(0.0722) * cur_rgb[3 * i + 2]
                var cvar = cur_var[i]
                var std_c = sqrt(max(cvar, Float32(0.0)))
                var f4 = min(std_c / (cl + Float32(0.01)), Float32(4.0))
                var f5 = inv_len[i]
                var sr = Float32(0.0)
                var sg = Float32(0.0)
                var sb = Float32(0.0)
                var sv = Float32(0.0)
                var sw = Float32(0.0)
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        var sx = min(width - 1, max(0, px + kx * step))
                        var sy = min(height - 1, max(0, py + ky * step))
                        var j = sy * width + sx
                        var nnz = sqrt(max(Float32(1.0) - (nx[j] * nx[j] + ny[j] * ny[j]), Float32(0.0)))
                        var ndot = nx[i] * nx[j] + ny[i] * ny[j] + cnz * nnz
                        var nl = Float32(0.2126) * cur_rgb[3 * j] + Float32(0.7152) * cur_rgb[3 * j + 1] + Float32(0.0722) * cur_rgb[3 * j + 2]
                        var dl_raw = abs(cl - nl)
                        var dz = abs(cz - z[j]) / max(cz * Float32(0.08), Float32(0.05))
                        var f = Array[Float32, 7](fill=0.0)
                        f[0] = min(dz, Float32(8.0))
                        f[1] = Float32(1.0) - max(ndot, Float32(0.0))
                        f[2] = min(dl_raw / (std_c + Float32(0.01)), Float32(16.0))
                        f[3] = dl_raw / (cl + nl + Float32(0.01))
                        f[4] = f4
                        f[5] = f5
                        var dist2 = kx * kx + ky * ky
                        f[6] = Float32(0.0) if dist2 == 0 else (Float32(0.70710678) if dist2 == 1 else Float32(1.0))
                        var wt = kw1[ky + 1] * kw1[kx + 1] * sota_ml_tap_weight_cpu(w, k, analytic, Float32(0.65), f, ndot, dz, dl_raw, std_c)
                        sr += wt * cur_rgb[3 * j]
                        sg += wt * cur_rgb[3 * j + 1]
                        sb += wt * cur_rgb[3 * j + 2]
                        sv += wt * wt * cur_var[j]
                        sw += wt
                var inv = Float32(1.0) / max(sw, Float32(1e-5))
                out_rgb[3 * i] = sota_ml_f16(sr * inv)
                out_rgb[3 * i + 1] = sota_ml_f16(sg * inv)
                out_rgb[3 * i + 2] = sota_ml_f16(sb * inv)
                out_var[i] = sota_ml_f16(sv * (inv * inv))
        cur_rgb = out_rgb^
        cur_var = out_var^
    return (cur_rgb^, cur_var^)


# ============================================================================
# SPIR-V (ShaderDsl)
# ============================================================================


def _pack2(mut dsl: ShaderDsl, a: GpuF32, b: GpuF32) -> GpuU32:
    return GpuU32(dsl.mod_addr, dsl.raw_module()[].pack_f16x2(a.id, b.id))


def _unpack2(mut dsl: ShaderDsl, u: GpuU32) -> Tuple[GpuF32, GpuF32]:
    var t = dsl.raw_module()[].unpack_f16x2(u.id)
    return (GpuF32(dsl.mod_addr, t[0]), GpuF32(dsl.mod_addr, t[1]))


def sota_ml_temporal_shader(group_w: Int = SOTA_ML_GROUP_W, group_h: Int = SOTA_ML_GROUP_H) -> List[UInt32]:
    """Reprojection + luma moments with a private history (8 words/pixel: rgb, len, m1, m2, z, nx).
    Push: 0 w, 1 h, 2 hist_in, 3 hist_out, 4 reset, 5 A offset, 6 G offset, 7 max history (f32)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=group_w, group_height=group_h, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var hist_in = dsl.push_u32(2)
    var hist_out = dsl.push_u32(3)
    var reset = dsl.push_u32(4).ne(0)
    var a_off = dsl.push_u32(5)
    var g_off = dsl.push_u32(6)
    var max_len = dsl.push_f32(7)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var pix = py * w_u + px
    var c_base = b_hdr.base_word(pix)
    var cur_rgb = b_hdr.load_vec3(c_base)
    var cur_z = b_hdr.load_f32(c_base + 3)
    var vel_u = b_hdr.load_f32(c_base + 4)
    var vel_v = b_hdr.load_f32(c_base + 5)
    var cur_nx = b_hdr.load_f32(c_base + 6)
    var cur_ny = b_hdr.load_f32(c_base + 7)
    var luma_w = dsl.vec3_const(0.2126, 0.7152, 0.0722)
    var cur_l = dot(cur_rgb, luma_w)

    var fw = w_u.to_f32()
    var fh = h_u.to_f32()
    var prev_u = (px.to_f32() + dsl.f32(0.5)) / fw - vel_u
    var prev_v = (py.to_f32() + dsl.f32(0.5)) / fh - vel_v
    var in_bounds = prev_u.ge(0.0) & prev_u.lt(1.0) & prev_v.ge(0.0) & prev_v.lt(1.0) & reset.not_()
    var prev_x = clamp_f(floor_f(prev_u * fw), dsl.f32(0.0), fw - dsl.f32(1.0)).saturate_u32()
    var prev_y = clamp_f(floor_f(prev_v * fh), dsl.f32(0.0), fh - dsl.f32(1.0)).saturate_u32()
    var p_base = hist_in + (prev_y * w_u + prev_x) * dsl.u32(8)
    var prev_rgb = b_hist.load_vec3(p_base)
    var prev_len = b_hist.load_f32(p_base + 3)
    var prev_m1 = b_hist.load_f32(p_base + 4)
    var prev_m2 = b_hist.load_f32(p_base + 5)
    var prev_z = b_hist.load_f32(p_base + 6)
    var prev_nx = b_hist.load_f32(p_base + 7)
    var dz = abs_f(cur_z - prev_z) / max_f(cur_z * 0.1, 0.08)
    var ok = in_bounds & dz.lt(0.12) & abs_f(cur_nx - prev_nx).lt(0.25) & cur_z.lt(1e5) & prev_len.ge(1.0)

    var r_rgb = dsl.var_vec3(cur_rgb)
    var r_m1 = dsl.var_f32(cur_l)
    var r_m2 = dsl.var_f32(cur_l * cur_l)
    var r_len = dsl.var_f32_const(1.0)
    dsl.if_then(ok)
    var nl = min_f(prev_len + dsl.f32(1.0), max_len)
    var al = dsl.f32(1.0) / nl
    var om = dsl.f32(1.0) - al
    r_rgb.set(cur_rgb * al + prev_rgb * om)
    r_m1.set(cur_l * al + prev_m1 * om)
    r_m2.set(cur_l * cur_l * al + prev_m2 * om)
    r_len.set(nl)
    dsl.end_if()
    var f_rgb = r_rgb.get()
    var f_m1 = r_m1.get()
    var f_m2 = r_m2.get()
    var f_len = r_len.get()
    var raw_var = max_f(f_m2 - f_m1 * f_m1, 0.0)
    var var_l = select_f(f_len.lt(4.0), raw_var * (dsl.f32(4.0) / f_len), raw_var)

    var o_base = hist_out + pix * dsl.u32(8)
    b_hist.store_vec3(o_base, f_rgb)
    b_hist.store_f32(o_base + 3, f_len)
    b_hist.store_f32(o_base + 4, f_m1)
    b_hist.store_f32(o_base + 5, f_m2)
    b_hist.store_f32(o_base + 6, cur_z)
    b_hist.store_f32(o_base + 7, cur_nx)

    var a_base = a_off + pix * dsl.u32(2)
    b_hist.store_u32(a_base, _pack2(dsl, f_rgb.x(), f_rgb.y()))
    b_hist.store_u32(a_base + 1, _pack2(dsl, f_rgb.z(), var_l))
    var g_base = g_off + pix * dsl.u32(3)
    b_hist.store_f32(g_base, cur_z)
    b_hist.store_u32(g_base + 1, _pack2(dsl, cur_nx, cur_ny))
    b_hist.store_f32(g_base + 2, dsl.f32(1.0) / f_len)
    return dsl.finish()


def sota_ml_atrous_shader(
    w: SotaMlWeights, level: Int, analytic: Bool, sigma_luma: Float32 = 0.65, group_w: Int = SOTA_ML_GROUP_W, group_h: Int = SOTA_ML_GROUP_H
) -> List[UInt32]:
    """One dilated 3x3 level with learned (or, `analytic`, SVGF) tap weights baked in as constants.
    Push: 0 w, 1 h, 2 step, 3 src offset, 4 dst offset, 5 G offset, 6 final (remodulate into HDR_OUT),
    7 capture (final pass also stores crisp + albedo as f16 at push 8 = D offset)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=group_w, group_height=group_h, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var step_u = dsl.push_u32(2)
    var src_off = dsl.push_u32(3)
    var dst_off = dsl.push_u32(4)
    var g_off = dsl.push_u32(5)
    var final_pass = dsl.push_u32(6).ne(0)
    var capture = dsl.push_u32(7).ne(0)
    var d_off = dsl.push_u32(8)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var pix = py * w_u + px
    var s0 = _unpack2(dsl, b_hist.load_u32(src_off + pix * dsl.u32(2)))
    var s1 = _unpack2(dsl, b_hist.load_u32(src_off + pix * dsl.u32(2) + 1))
    var c_rgb = GpuVec3(s0[0], s0[1], s1[0])
    var c_var = s1[1]
    var g_base = g_off + pix * dsl.u32(3)
    var cz = b_hist.load_f32(g_base)
    var cn2 = _unpack2(dsl, b_hist.load_u32(g_base + 1))
    var c_inv_len = b_hist.load_f32(g_base + 2)
    var cnz = sqrt_f(max_f(dsl.f32(1.0) - fma_f(cn2[0], cn2[0], cn2[1] * cn2[1]), 0.0))
    var luma_w = dsl.vec3_const(0.2126, 0.7152, 0.0722)
    var cl = dot(c_rgb, luma_w)
    var std_c = sqrt_f(max_f(c_var, 0.0))
    var f4 = min_f(std_c / (cl + dsl.f32(0.01)), 4.0)
    var dz_den = max_f(cz * 0.08, 0.05)
    var inv_std = dsl.f32(1.0) / (std_c + dsl.f32(0.01))
    var s_scaled = max_f(fma_f(dsl.f32(sigma_luma), std_c, dsl.f32(0.001)), 0.01)

    # Per-pixel first-layer bias: b1 + w1[4] f4 + w1[5] f5 (learned mode only)
    var hidden = 0 if analytic else w.hidden
    var base = List[GpuF32]()
    for j in range(hidden):
        var b = fma_f(f4, dsl.f32(w.w1(level, j, 4)), dsl.f32(w.b1(level, j)))
        base.append(fma_f(c_inv_len, dsl.f32(w.w1(level, j, 5)), b))

    var sum_rgb = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var sum_var = dsl.var_f32_const(0.0)
    var sum_w = dsl.var_f32_const(0.0)
    dsl.if_then(cz.gt(1e5))
    sum_rgb.set(c_rgb)
    sum_var.set(c_var)
    sum_w.set(dsl.f32(1.0))
    dsl.else_branch()
    var kw1 = [Float32(0.25), Float32(0.5), Float32(0.25)]
    for ky in range(-1, 2):
        for kx in range(-1, 2):
            var sx = px
            if kx < 0:
                sx = select_u(px.ge(step_u), px - step_u, dsl.u32(0))
            elif kx > 0:
                sx = min_u(w_u - 1, px + step_u)
            var sy = py
            if ky < 0:
                sy = select_u(py.ge(step_u), py - step_u, dsl.u32(0))
            elif ky > 0:
                sy = min_u(h_u - 1, py + step_u)
            var npix = sy * w_u + sx
            var n0 = _unpack2(dsl, b_hist.load_u32(src_off + npix * dsl.u32(2)))
            var n1 = _unpack2(dsl, b_hist.load_u32(src_off + npix * dsl.u32(2) + 1))
            var n_rgb = GpuVec3(n0[0], n0[1], n1[0])
            var n_var = n1[1]
            var ng = g_off + npix * dsl.u32(3)
            var nz_l = b_hist.load_f32(ng)
            var nn2 = _unpack2(dsl, b_hist.load_u32(ng + 1))
            var nnz = sqrt_f(max_f(dsl.f32(1.0) - fma_f(nn2[0], nn2[0], nn2[1] * nn2[1]), 0.0))
            var ndot = fma_f(cn2[0], nn2[0], fma_f(cn2[1], nn2[1], cnz * nnz))
            var nl = dot(n_rgb, luma_w)
            var dl_raw = abs_f(cl - nl)
            var dz = abs_f(cz - nz_l) / dz_den
            var kw = kw1[ky + 1] * kw1[kx + 1]
            var w_tap: GpuF32
            if analytic:
                var w_n = pow_f(max_f(ndot, 0.0), 32.0)
                var dl = dl_raw / s_scaled
                w_tap = (w_n * exp2_f(-((dz + dl) * 1.442695))) * kw
            else:
                var f0 = min_f(dz, 8.0)
                var f1 = dsl.f32(1.0) - max_f(ndot, 0.0)
                var f2 = min_f(dl_raw * inv_std, 16.0)
                var f3 = dl_raw / (cl + nl + dsl.f32(0.01))
                var d2 = kx * kx + ky * ky
                var f6 = Float32(0.0) if d2 == 0 else (Float32(0.70710678) if d2 == 1 else Float32(1.0))
                var o = dsl.f32(w.b2(level))
                for j in range(w.hidden):
                    var pre = base[j]
                    if f6 != Float32(0.0):
                        pre = pre + w.w1(level, j, 6) * f6
                    pre = fma_f(f0, dsl.f32(w.w1(level, j, 0)), pre)
                    pre = fma_f(f1, dsl.f32(w.w1(level, j, 1)), pre)
                    pre = fma_f(f2, dsl.f32(w.w1(level, j, 2)), pre)
                    pre = fma_f(f3, dsl.f32(w.w1(level, j, 3)), pre)
                    o = fma_f(max_f(pre, 0.0), dsl.f32(w.w2(level, j)), o)
                var sig = dsl.f32(1.0) / (dsl.f32(1.0) + exp2_f(o * -1.442695))
                w_tap = sig * kw
            var acc = sum_rgb.get()
            sum_rgb.set(GpuVec3(fma_f(n_rgb.x(), w_tap, acc.x()), fma_f(n_rgb.y(), w_tap, acc.y()), fma_f(n_rgb.z(), w_tap, acc.z())))
            sum_var.set(fma_f(n_var, w_tap * w_tap, sum_var.get()))
            sum_w.set(sum_w.get() + w_tap)
    dsl.end_if()

    var inv_sw = dsl.f32(1.0) / max_f(sum_w.get(), 1e-5)
    var f_rgb = sum_rgb.get() * inv_sw
    var f_var = sum_var.get() * (inv_sw * inv_sw)
    var d_base = dst_off + pix * dsl.u32(2)
    b_hist.store_u32(d_base, _pack2(dsl, f_rgb.x(), f_rgb.y()))
    b_hist.store_u32(d_base + 1, _pack2(dsl, f_rgb.z(), f_var))

    dsl.if_then(final_pass)
    var c_base = b_hdr.base_word(pix)
    var p_base = dsl.u32(UInt32(SOTA_ML_POST_BASE)) + c_base
    var crisp = b_post.load_vec3(p_base)
    var alb = b_post.load_vec3(p_base + UInt32(3))
    b_hdr.store_vec3(c_base, GpuVec3(fma_f(alb.x(), f_rgb.x(), crisp.x()), fma_f(alb.y(), f_rgb.y(), crisp.y()), fma_f(alb.z(), f_rgb.z(), crisp.z())))
    b_hdr.store_f32(c_base + 4, b_post.load_f32(p_base + UInt32(6)))
    b_hdr.store_f32(c_base + 5, b_post.load_f32(p_base + UInt32(7)))
    dsl.if_then(capture)
    var dd = d_off + pix * dsl.u32(3)
    b_hist.store_u32(dd, _pack2(dsl, crisp.x(), crisp.y()))
    b_hist.store_u32(dd + 1, _pack2(dsl, crisp.z(), alb.x()))
    b_hist.store_u32(dd + 2, _pack2(dsl, alb.y(), alb.z()))
    dsl.end_if()
    dsl.end_if()
    return dsl.finish()


# ============================================================================
# Runtime (owned by `FlintAaaVulkanRenderer.sota_ml`, driven from the `sota:ml` hooks)
# ============================================================================


struct SotaMlDenoiser(Movable):
    """Renderer settings and GPU state of the learned denoiser (`FlintAaaVulkanRenderer.sota_ml`).

    - `enabled` (default True; `FLINT_SOTA_ML_DENOISE=0` turns it off at start): use the learned path
      whenever weights load; otherwise the renderer runs SVGF unchanged.
    - `mode`: `SOTA_ML_MODE_LEARNED` (default) or `SOTA_ML_MODE_ANALYTIC` (SVGF weights inside this
      path; for ablations and for GPU tests on machines without weights).
    - `weights_path`: `.flmlw` file (default from `sota_ml_default_weights_path()`).
    - `max_history` (default 32): temporal accumulation cap in frames.
    - `capture` (default False): keep the remodulation terms in scratch D for training-data dumps.
    """
    var enabled: Bool
    var mode: Int
    var weights_path: String
    var max_history: Float32
    var capture: Bool
    var group_w: Int
    var group_h: Int
    var weights: SotaMlWeights
    var weights_tried: Bool
    var weights_version: Int
    var pipe_temporal: Int
    var pipes_level: List[Int]
    var built_mode: Int
    var built_version: Int
    var scratch_base: Int
    var scratch_max_pix: Int
    var parity: Int
    var valid_frames: Int
    var ran_last_frame: Bool
    var status: String

    def __init__(out self):
        self.enabled = env_now("FLINT_SOTA_ML_DENOISE") != "0"
        self.mode = SOTA_ML_MODE_LEARNED
        self.weights_path = sota_ml_default_weights_path()
        self.max_history = Float32(32.0)
        self.capture = False
        self.group_w = SOTA_ML_GROUP_W
        self.group_h = SOTA_ML_GROUP_H
        self.weights = SotaMlWeights()
        self.weights_tried = False
        self.weights_version = 0
        self.pipe_temporal = 0
        self.pipes_level = List[Int]()
        self.built_mode = -1
        self.built_version = -1
        self.scratch_base = -1
        self.scratch_max_pix = 0
        self.parity = 0
        self.valid_frames = 0
        self.ran_last_frame = False
        self.status = String("idle")

    @staticmethod
    def scratch_words(max_pix: Int) -> Int:
        return max_pix * SOTA_ML_SCRATCH_WORDS_PER_PIXEL

    def a_offset(self) -> Int:
        return self.scratch_base

    def b_offset(self) -> Int:
        return self.scratch_base + 2 * self.scratch_max_pix

    def c_offset(self) -> Int:
        return self.scratch_base + 4 * self.scratch_max_pix

    def g_offset(self) -> Int:
        return self.scratch_base + 6 * self.scratch_max_pix

    def d_offset(self) -> Int:
        return self.scratch_base + 9 * self.scratch_max_pix

    def levels(self) -> Int:
        return SOTA_ML_ANALYTIC_LEVELS if self.mode == SOTA_ML_MODE_ANALYTIC else self.weights.levels

    def final_offset(self) -> Int:
        """Scratch region holding the last level's packed demodulated output (`B` for odd level counts)."""
        return self.b_offset() if ((self.levels() - 1) & 1) == 0 else self.c_offset()

    def history_offsets(self) -> Tuple[Int, Int]:
        """`(hist_in, hist_out)` of the next frame: ping-pong inside the SVGF history words."""
        var a = 0
        var b = self.scratch_max_pix * 8
        return (a, b) if (self.parity & 1) == 0 else (b, a)

    def set_weights(mut self, var w: SotaMlWeights):
        """Installs weights directly (tests, tools); pipelines rebuild on the next frame."""
        self.weights = w^
        self.weights_tried = True
        self.weights_version += 1

    def has_weights(self) -> Bool:
        return self.weights.valid()

    def _try_load(mut self):
        if self.weights_tried:
            return
        self.weights_tried = True
        var w = sota_ml_load_weights(self.weights_path)
        if w.valid():
            self.weights = w^
            self.weights_version += 1
            self.status = String("weights ") + self.weights_path
        else:
            self.status = String("no weights at ") + self.weights_path + " (SVGF fallback)"

    def _destroy_pipes(mut self, mut vk: VulkanGpu):
        if self.pipe_temporal != 0:
            vk.destroy_pipeline(self.pipe_temporal)
            self.pipe_temporal = 0
        for i in range(len(self.pipes_level)):
            if self.pipes_level[i] != 0:
                vk.destroy_pipeline(self.pipes_level[i])
        self.pipes_level.clear()
        self.built_mode = -1
        self.built_version = -1

    def ready(mut self, mut vk: VulkanGpu, pipe_layout: Int) -> Bool:
        """True when this frame should take the learned path (weights present and pipelines built)."""
        if not self.enabled or self.scratch_base < 0 or self.scratch_max_pix <= 0:
            return False
        if self.mode == SOTA_ML_MODE_LEARNED:
            self._try_load()
            if not self.weights.valid():
                return False
        var ver = (self.weights_version if self.mode == SOTA_ML_MODE_LEARNED else -7) * 4096 + self.group_w * 64 + self.group_h
        if self.built_mode != self.mode or self.built_version != ver:
            self._destroy_pipes(vk)
            self.pipe_temporal = vk.create_aaa_compute_pipeline(pipe_layout, sota_ml_temporal_shader(self.group_w, self.group_h))
            var analytic = self.mode == SOTA_ML_MODE_ANALYTIC
            for k in range(self.levels()):
                self.pipes_level.append(vk.create_aaa_compute_pipeline(pipe_layout, sota_ml_atrous_shader(self.weights, k, analytic, 0.65, self.group_w, self.group_h)))
            self.built_mode = self.mode
            self.built_version = ver
            var ok = self.pipe_temporal != 0
            for p in self.pipes_level:
                ok = ok and p != 0
            if not ok:
                self._destroy_pipes(vk)
                self.status = String("pipeline creation failed (SVGF fallback)")
                self.enabled = False
                return False
        return True

    def record(mut self, mut vk: VulkanGpu, slot: Int, pipe_layout: Int, desc_set: Int, w: Int, h: Int):
        """Records temporal + `K` levels; the last level remodulates into `BIND_AAA_HDR_OUT` like SVGF."""
        var gx = (w + self.group_w - 1) // self.group_w
        var gy = (h + self.group_h - 1) // self.group_h
        var hist = self.history_offsets()
        var push_t = List[UInt32](length=16, fill=0)
        push_t[0] = UInt32(w)
        push_t[1] = UInt32(h)
        push_t[2] = UInt32(hist[0])
        push_t[3] = UInt32(hist[1])
        push_t[4] = UInt32(1) if self.valid_frames == 0 else UInt32(0)
        push_t[5] = UInt32(self.a_offset())
        push_t[6] = UInt32(self.g_offset())
        push_t[7] = bitcast[DType.uint32](self.max_history)
        vk.dispatch_aaa(slot, self.pipe_temporal, pipe_layout, desc_set, push_t, gx, gy)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        var n = len(self.pipes_level)
        for k in range(n):
            var src = self.a_offset() if k == 0 else (self.b_offset() if ((k - 1) & 1) == 0 else self.c_offset())
            var dst = self.b_offset() if (k & 1) == 0 else self.c_offset()
            var push_a = List[UInt32](length=16, fill=0)
            push_a[0] = UInt32(w)
            push_a[1] = UInt32(h)
            push_a[2] = UInt32(1 << k)
            push_a[3] = UInt32(src)
            push_a[4] = UInt32(dst)
            push_a[5] = UInt32(self.g_offset())
            push_a[6] = UInt32(1) if k + 1 == n else UInt32(0)
            push_a[7] = UInt32(1) if self.capture else UInt32(0)
            push_a[8] = UInt32(self.d_offset())
            vk.dispatch_aaa(slot, self.pipes_level[k], pipe_layout, desc_set, push_a, gx, gy)
            vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        self.parity += 1
        self.valid_frames += 1
        self.ran_last_frame = True

    def skipped(mut self):
        """Called on frames where the path did not run, so the next run starts a fresh history."""
        self.valid_frames = 0
        self.ran_last_frame = False

    def destroy(mut self, mut vk: VulkanGpu):
        self._destroy_pipes(vk)
