"""Area C (`sota:ml`): learned temporal upscaler, a drop-in for the `FlintTSR v3` resolve.

The resolve keeps TSR v3's reconstruction (unjittered bilinear current sample with 3x3 Laplacian sharpening
and min/max clamp, closest-depth velocity dilation, 5-tap Catmull-Rom history, Karis-weighted 3x3 YCoCg
moments) and replaces its two hand-tuned decisions with a small MLP per output pixel:

- the variance clip box size `gamma` (TSR: `clamp(1.22 - 0.08 * vel_px, 0.85, 1.25)`), and
- the current-frame blend weight (TSR: Blackman-Harris proximity x velocity / reactive attenuation).

Features: squared distance to the nearest render sample, output-pixel velocity, relative depth change
against the history, history deviation outside the unit-sigma box (before clipping), local relative luma
contrast and the render/output scale. MLP `6 -> H (ReLU) -> 2`; `gamma = 0.5 + 1.5 sigmoid(o0)`,
`w_cur = 0.02 + 0.98 sigmoid(o1)`, reactive pixels still drop history as in TSR.

Trained only on Flint-rendered sequences (`~/flint-models/_train/render/train_up.py`, data from
`datagen_up.mojo`); the weights stay outside the repo. With no weights file the renderer keeps the
TSR v3 resolve. Output layout is identical to TSR v3 (history `rgb z` at `hist_out`, 8-word HDR record at
`AAA_POST_BLOOM_BASE`), so everything downstream is unchanged.

Weights `.flmlw` (little endian): magic `FLMLUP01`, u32 NF (= 6), u32 H, then f32 `w1[H][NF]`, `b1[H]`,
`w2[2][H]`, `b2[2]`.
"""
from std.math import exp, floor, max, min, sqrt
from std.memory import bitcast
from flint.platform.os import env_now, read_bytes
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    PIPELINE_COMPUTE,
    PIPELINE_TRANSFER,
    VulkanGpu,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_OIT_NODES,
    BIND_AAA_POST_SCRATCH,
)
from flint.render.aaa_oit_volumetrics import AAA_OIT_REACTIVE_BASE_WORDS
from flint.render.aaa_post_tsr import _emit_catmull_rom_5tap_hist

comptime SOTA_ML_UP_NF: Int = 6
comptime SOTA_ML_UP_DEFAULT_WEIGHTS: String = "/flint-models/_train/render/weights/sota_ml_upscale_v1.flmlw"


struct SotaMlUpWeights(Copyable, Movable):
    var hidden: Int
    var data: List[Float32]  # w1[H][NF], b1[H], w2[2][H], b2[2]

    def __init__(out self):
        self.hidden = 0
        self.data = List[Float32]()

    def __init__(out self, hidden: Int):
        self.hidden = hidden
        self.data = List[Float32](length=hidden * SOTA_ML_UP_NF + hidden + 2 * hidden + 2, fill=0.0)

    def valid(self) -> Bool:
        return self.hidden > 0 and len(self.data) == self.hidden * (SOTA_ML_UP_NF + 3) + 2

    def w1(self, j: Int, f: Int) -> Float32:
        return self.data[j * SOTA_ML_UP_NF + f]

    def b1(self, j: Int) -> Float32:
        return self.data[self.hidden * SOTA_ML_UP_NF + j]

    def w2(self, o: Int, j: Int) -> Float32:
        return self.data[self.hidden * (SOTA_ML_UP_NF + 1) + o * self.hidden + j]

    def b2(self, o: Int) -> Float32:
        return self.data[self.hidden * (SOTA_ML_UP_NF + 3) + o]

    def set(mut self, index: Int, v: Float32):
        self.data[index] = v


def _u32_at(data: List[UInt8], at: Int) -> UInt32:
    return UInt32(data[at]) | (UInt32(data[at + 1]) << 8) | (UInt32(data[at + 2]) << 16) | (UInt32(data[at + 3]) << 24)


def sota_ml_up_parse_weights(data: List[UInt8]) -> SotaMlUpWeights:
    """Parses a `FLMLUP01` blob; returns an invalid (empty) set on any mismatch."""
    var magic = "FLMLUP01"
    if len(data) < 16:
        return SotaMlUpWeights()
    for i in range(8):
        if data[i] != magic.as_bytes()[i]:
            return SotaMlUpWeights()
    var nf = Int(_u32_at(data, 8))
    var hid = Int(_u32_at(data, 12))
    if nf != SOTA_ML_UP_NF or hid < 1 or hid > 64:
        return SotaMlUpWeights()
    var w = SotaMlUpWeights(hid)
    if len(data) != 16 + len(w.data) * 4:
        return SotaMlUpWeights()
    for i in range(len(w.data)):
        w.data[i] = bitcast[DType.float32](_u32_at(data, 16 + 4 * i))
    return w^


def sota_ml_up_serialize_weights(w: SotaMlUpWeights) -> List[UInt8]:
    var out = List[UInt8]()
    for b in "FLMLUP01".as_bytes():
        out.append(b)
    var hdr = [UInt32(SOTA_ML_UP_NF), UInt32(w.hidden)]
    for v in hdr:
        for s in range(4):
            out.append(UInt8((v >> UInt32(8 * s)) & 0xFF))
    for x in w.data:
        var u = bitcast[DType.uint32](x)
        for s in range(4):
            out.append(UInt8((u >> UInt32(8 * s)) & 0xFF))
    return out^


def sota_ml_up_default_weights_path() -> String:
    var p = env_now("FLINT_SOTA_ML_UPSCALE_WEIGHTS")
    if p != "":
        return p
    return env_now("HOME") + SOTA_ML_UP_DEFAULT_WEIGHTS


def sota_ml_up_load_weights(path: String) -> SotaMlUpWeights:
    return sota_ml_up_parse_weights(read_bytes(path))


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


def sota_ml_up_mlp_cpu(w: SotaMlUpWeights, f: Array[Float32, 6]) -> Tuple[Float32, Float32]:
    """`(gamma, w_cur)` from the features (CPU reference of the baked-constant MLP)."""
    var o0 = w.b2(0)
    var o1 = w.b2(1)
    for j in range(w.hidden):
        var pre = w.b1(j)
        for q in range(SOTA_ML_UP_NF):
            pre += w.w1(j, q) * f[q]
        var hj = max(pre, Float32(0.0))
        o0 += w.w2(0, j) * hj
        o1 += w.w2(1, j) * hj
    return (Float32(0.5) + Float32(1.5) * _sigmoid(o0), Float32(0.02) + Float32(0.98) * _sigmoid(o1))


# ============================================================================
# SPIR-V
# ============================================================================


def sota_ml_upscale_shader(w: SotaMlUpWeights) -> List[UInt32]:
    """Learned TSR resolve. Push layout identical to `aaa_tsr_v3_resolve_shader`:
    0 src_w, 1 src_h, 2 dst_w, 3 dst_h, 4 history weight (> 0.001 enables history), 5 hist_in,
    6 hist_out, 7 dst scratch offset, 8 sharpening."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var dx = s.invocation_axis(0)
    var dy = s.invocation_axis(1)
    var src_w = s.push_word(0)
    var src_h = s.push_word(1)
    var dst_w = s.push_word(2)
    var dst_h = s.push_word(3)
    var base_hist_w = s.push_float(4)
    var hist_in_off = s.push_word(5)
    var hist_out_off = s.push_word(6)
    var dst_scratch_off = s.push_word(7)
    var sharp_amt = s.nclamp(s.push_float(8), s.float(0.0), s.float(1.0))
    s.return_if(s.logical_or(s.uge(dx, dst_w), s.uge(dy, dst_h)))

    var f_sw = s.float_from_uint(src_w)
    var f_sh = s.float_from_uint(src_h)
    var f_dw = s.float_from_uint(dst_w)
    var f_dh = s.float_from_uint(dst_h)
    var uv_u = s.fdiv(s.fadd(s.float_from_uint(dx), s.float(0.5)), f_dw)
    var uv_v = s.fdiv(s.fadd(s.float_from_uint(dy), s.float(0.5)), f_dh)
    var jx_px = s.load_float(b_globals, s.uint(151))
    var jy_px = s.load_float(b_globals, s.uint(152))
    var fx_raw = s.fsub(s.fadd(s.fmul(uv_u, f_sw), jx_px), s.float(0.5))
    var fy_raw = s.fsub(s.fsub(s.fmul(uv_v, f_sh), jy_px), s.float(0.5))
    var fx = s.nclamp(fx_raw, s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_sw, s.float(1.0))))
    var fy = s.nclamp(fy_raw, s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_sh, s.float(1.0))))
    var ix0 = s.umin(s.isub(src_w, s.uint(1)), s.saturate_uint(s.ffloor(fx)))
    var iy0 = s.umin(s.isub(src_h, s.uint(1)), s.saturate_uint(s.ffloor(fy)))
    var ix1 = s.umin(s.isub(src_w, s.uint(1)), s.iadd(ix0, s.uint(1)))
    var iy1 = s.umin(s.isub(src_h, s.uint(1)), s.iadd(iy0, s.uint(1)))
    var tx = s.nclamp(s.fsub(fx, s.float_from_uint(ix0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(fy, s.float_from_uint(iy0)), s.float(0.0), s.float(1.0))
    var near_x = s.ffloor(s.fadd(fx_raw, s.float(0.5)))
    var near_y = s.ffloor(s.fadd(fy_raw, s.float(0.5)))
    var sdx = s.fsub(fx_raw, near_x)
    var sdy = s.fsub(fy_raw, near_y)
    var sub_d2 = s.nmin(s.float(1.0), s.fma(sdx, sdx, s.fmul(sdy, sdy)))

    var p00 = s.imul(s.iadd(s.imul(iy0, src_w), ix0), s.uint(8))
    var p10 = s.imul(s.iadd(s.imul(iy0, src_w), ix1), s.uint(8))
    var p01 = s.imul(s.iadd(s.imul(iy1, src_w), ix0), s.uint(8))
    var p11 = s.imul(s.iadd(s.imul(iy1, src_w), ix1), s.uint(8))
    var om_tx = s.fsub(s.float(1.0), tx)
    var om_ty = s.fsub(s.float(1.0), ty)
    var w00 = s.fmul(om_tx, om_ty)
    var w10 = s.fmul(tx, om_ty)
    var w01 = s.fmul(om_tx, ty)
    var w11 = s.fmul(tx, ty)
    var cur = List[UInt32]()
    for c in range(3):
        var o = s.uint(UInt32(c))
        cur.append(
            s.fma(
                s.load_float(b_hdr, s.iadd(p00, o)),
                w00,
                s.fma(s.load_float(b_hdr, s.iadd(p10, o)), w10, s.fma(s.load_float(b_hdr, s.iadd(p01, o)), w01, s.fmul(s.load_float(b_hdr, s.iadd(p11, o)), w11))),
            )
        )

    var ix_c = s.umin(s.isub(src_w, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), near_x)))
    var iy_c = s.umin(s.isub(src_h, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), near_y)))
    var p_center = s.imul(s.iadd(s.imul(iy_c, src_w), ix_c), s.uint(8))
    var best_p = s.uvar(p_center)
    var best_z = s.fvar(s.load_float(b_hdr, s.iadd(p_center, s.uint(3))))
    var sum_wk = s.fvar(s.float(0.0))
    var m1_y = s.fvar(s.float(0.0))
    var m2_y = s.fvar(s.float(0.0))
    var m1_co = s.fvar(s.float(0.0))
    var m2_co = s.fvar(s.float(0.0))
    var m1_cg = s.fvar(s.float(0.0))
    var m2_cg = s.fvar(s.float(0.0))
    var mins = List[UInt32]()
    var maxs = List[UInt32]()
    var laps = List[UInt32]()
    for _ in range(3):
        mins.append(s.fvar(s.float(1e9)))
        maxs.append(s.fvar(s.float(-1e9)))
        laps.append(s.fvar(s.float(0.0)))
    for ny in range(-1, 2):
        for nx in range(-1, 2):
            var sx_n = ix_c
            if nx < 0:
                sx_n = s.select_u(s.uge(ix_c, s.uint(1)), s.isub(ix_c, s.uint(1)), s.uint(0))
            elif nx > 0:
                sx_n = s.umin(s.isub(src_w, s.uint(1)), s.iadd(ix_c, s.uint(1)))
            var sy_n = iy_c
            if ny < 0:
                sy_n = s.select_u(s.uge(iy_c, s.uint(1)), s.isub(iy_c, s.uint(1)), s.uint(0))
            elif ny > 0:
                sy_n = s.umin(s.isub(src_h, s.uint(1)), s.iadd(iy_c, s.uint(1)))
            var np = s.imul(s.iadd(s.imul(sy_n, src_w), sx_n), s.uint(8))
            var nrgb = [s.load_float(b_hdr, np), s.load_float(b_hdr, s.iadd(np, s.uint(1))), s.load_float(b_hdr, s.iadd(np, s.uint(2)))]
            var nz = s.load_float(b_hdr, s.iadd(np, s.uint(3)))
            s.begin_if(s.flt(nz, s.load_f(best_z)))
            s.store(best_z, nz)
            s.store(best_p, np)
            s.end_if()
            for c in range(3):
                if (nx == 0 and abs(ny) == 1) or (ny == 0 and abs(nx) == 1):
                    s.store(laps[c], s.fadd(s.load_f(laps[c]), nrgb[c]))
                s.store(mins[c], s.nmin(s.load_f(mins[c]), nrgb[c]))
                s.store(maxs[c], s.nmax(s.load_f(maxs[c]), nrgb[c]))
            var ny_c = s.fma(nrgb[0], s.float(0.25), s.fma(nrgb[1], s.float(0.5), s.fmul(nrgb[2], s.float(0.25))))
            var nco = s.fmul(s.fsub(nrgb[0], nrgb[2]), s.float(0.5))
            var ncg = s.fma(nrgb[1], s.float(0.5), s.fmul(s.fadd(nrgb[0], nrgb[2]), s.float(-0.25)))
            var wk = s.fdiv(s.float(1.0), s.fma(s.nmax(s.float(0.0), ny_c), s.float(0.25), s.float(1.0)))
            s.store(sum_wk, s.fadd(s.load_f(sum_wk), wk))
            s.store(m1_y, s.fma(ny_c, wk, s.load_f(m1_y)))
            s.store(m2_y, s.fma(s.fmul(ny_c, ny_c), wk, s.load_f(m2_y)))
            s.store(m1_co, s.fma(nco, wk, s.load_f(m1_co)))
            s.store(m2_co, s.fma(s.fmul(nco, nco), wk, s.load_f(m2_co)))
            s.store(m1_cg, s.fma(ncg, wk, s.load_f(m1_cg)))
            s.store(m2_cg, s.fma(s.fmul(ncg, ncg), wk, s.load_f(m2_cg)))

    var bp = s.load_u(best_p)
    var cur_z = s.load_f(best_z)
    var vel_u = s.load_float(b_hdr, s.iadd(bp, s.uint(4)))
    var vel_v = s.load_float(b_hdr, s.iadd(bp, s.uint(5)))
    var cur_nx = s.load_float(b_hdr, s.iadd(bp, s.uint(6)))
    var cur_ny = s.load_float(b_hdr, s.iadd(bp, s.uint(7)))
    var sharp = List[UInt32]()
    for c in range(3):
        var raw = s.fma(s.fsub(cur[c], s.fmul(s.load_f(laps[c]), s.float(0.25))), sharp_amt, cur[c])
        sharp.append(s.nmax(s.float(0.0), s.nclamp(raw, s.load_f(mins[c]), s.load_f(maxs[c]))))
    var res_r = s.fvar(sharp[0])
    var res_g = s.fvar(sharp[1])
    var res_b = s.fvar(sharp[2])

    var src_pix_idx = s.iadd(s.imul(iy_c, src_w), ix_c)
    var reactive = s.load_float(b_oit, s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), src_pix_idx))
    var prev_u = s.fsub(uv_u, vel_u)
    var prev_v = s.fsub(uv_v, vel_v)
    var valid_reproj = s.logical_and(
        s.logical_and(s.fgt(base_hist_w, s.float(0.001)), s.flt(reactive, s.float(0.55))),
        s.logical_and(
            s.logical_and(s.fge(prev_u, s.float(0.0)), s.flt(prev_u, s.float(1.0))),
            s.logical_and(s.fge(prev_v, s.float(0.0)), s.flt(prev_v, s.float(1.0))),
        ),
    )
    s.begin_if(valid_reproj)
    var h_cat = _emit_catmull_rom_5tap_hist(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, prev_u, prev_v)
    var hr = h_cat[0]
    var hg = h_cat[1]
    var hb = h_cat[2]
    var hz = h_cat[3]
    var rel_dz = s.fdiv(s.fabs(s.fsub(cur_z, hz)), s.nmax(s.float(0.1), cur_z))
    s.begin_if(s.flt(rel_dz, s.float(0.10)))
    var vpx_x = s.fmul(vel_u, f_dw)
    var vpx_y = s.fmul(vel_v, f_dh)
    var vel_px = s.fsqrt(s.fma(vpx_x, vpx_x, s.fmul(vpx_y, vpx_y)))
    var inv_swk = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.load_f(sum_wk)))
    var mu_y = s.fmul(s.load_f(m1_y), inv_swk)
    var mu_co = s.fmul(s.load_f(m1_co), inv_swk)
    var mu_cg = s.fmul(s.load_f(m1_cg), inv_swk)
    var sig_y = s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.fmul(s.load_f(m2_y), inv_swk), s.fmul(mu_y, mu_y))))
    var sig_co = s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.fmul(s.load_f(m2_co), inv_swk), s.fmul(mu_co, mu_co))))
    var sig_cg = s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.fmul(s.load_f(m2_cg), inv_swk), s.fmul(mu_cg, mu_cg))))
    var hy_raw = s.fma(hr, s.float(0.25), s.fma(hg, s.float(0.5), s.fmul(hb, s.float(0.25))))
    var hco_raw = s.fmul(s.fsub(hr, hb), s.float(0.5))
    var hcg_raw = s.fma(hg, s.float(0.5), s.fmul(s.fadd(hr, hb), s.float(-0.25)))
    var d_y = s.fsub(hy_raw, mu_y)
    var d_co = s.fsub(hco_raw, mu_co)
    var d_cg = s.fsub(hcg_raw, mu_cg)
    # history deviation in unit-sigma boxes (before the learned box size)
    var r_unit = s.nmax(
        s.fdiv(s.fabs(d_y), s.nmax(s.float(1e-4), sig_y)),
        s.nmax(s.fdiv(s.fabs(d_co), s.nmax(s.float(1e-4), sig_co)), s.fdiv(s.fabs(d_cg), s.nmax(s.float(1e-4), sig_cg))),
    )
    # features
    var feats = List[UInt32]()
    feats.append(sub_d2)
    feats.append(s.nmin(s.fmul(vel_px, s.float(0.25)), s.float(4.0)))
    feats.append(s.nmin(s.fmul(rel_dz, s.float(10.0)), s.float(4.0)))
    feats.append(s.fmul(s.nmin(s.nmax(s.fsub(r_unit, s.float(1.0)), s.float(0.0)), s.float(8.0)), s.float(0.25)))
    feats.append(s.fmul(s.nmin(s.fdiv(sig_y, s.fadd(s.nmax(s.float(0.0), mu_y), s.float(0.02))), s.float(4.0)), s.float(0.5)))
    feats.append(s.fdiv(f_sw, f_dw))
    var o0 = s.float(w.b2(0))
    var o1 = s.float(w.b2(1))
    for j in range(w.hidden):
        var pre = s.float(w.b1(j))
        for q in range(SOTA_ML_UP_NF):
            pre = s.fma(feats[q], s.float(w.w1(j, q)), pre)
        var hj = s.nmax(pre, s.float(0.0))
        o0 = s.fma(hj, s.float(w.w2(0, j)), o0)
        o1 = s.fma(hj, s.float(w.w2(1, j)), o1)
    var sg0 = s.fdiv(s.float(1.0), s.fadd(s.float(1.0), s.exp2(s.fmul(o0, s.float(-1.442695)))))
    var sg1 = s.fdiv(s.float(1.0), s.fadd(s.float(1.0), s.exp2(s.fmul(o1, s.float(-1.442695)))))
    var gamma = s.fma(sg0, s.float(1.5), s.float(0.5))
    var w_cur = s.fma(sg1, s.float(0.98), s.float(0.02))
    var max_rat = s.nmax(s.float(1.0), s.fdiv(r_unit, gamma))
    var clip_s = s.fdiv(s.float(1.0), max_rat)
    var hy_c = s.fma(d_y, clip_s, mu_y)
    var hco = s.fma(d_co, clip_s, mu_co)
    var hcg = s.fma(d_cg, clip_s, mu_cg)
    var tmp_yc = s.fsub(hy_c, hcg)
    var chr = s.nmax(s.float(0.0), s.fadd(tmp_yc, hco))
    var chg = s.nmax(s.float(0.0), s.fadd(hy_c, hcg))
    var chb = s.nmax(s.float(0.0), s.fsub(tmp_yc, hco))
    var react_atten = s.nclamp(s.fsub(s.float(1.0), s.fmul(reactive, s.float(1.85))), s.float(0.0), s.float(1.0))
    var hw = s.fmul(s.fsub(s.float(1.0), w_cur), react_atten)
    var cw = s.fsub(s.float(1.0), hw)
    s.store(res_r, s.fma(chr, hw, s.fmul(sharp[0], cw)))
    s.store(res_g, s.fma(chg, hw, s.fmul(sharp[1], cw)))
    s.store(res_b, s.fma(chb, hw, s.fmul(sharp[2], cw)))
    s.end_if()
    s.end_if()

    var fin_r = s.load_f(res_r)
    var fin_g = s.load_f(res_g)
    var fin_b = s.load_f(res_b)
    var dst_pix = s.iadd(s.imul(dy, dst_w), dx)
    var ho_idx = s.iadd(hist_out_off, s.imul(dst_pix, s.uint(4)))
    s.store_float(b_hist, ho_idx, fin_r)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(1)), fin_g)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(2)), fin_b)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(3)), cur_z)
    var post_out = s.iadd(dst_scratch_off, s.imul(dst_pix, s.uint(8)))
    s.store_float(b_post, post_out, fin_r)
    s.store_float(b_post, s.iadd(post_out, s.uint(1)), fin_g)
    s.store_float(b_post, s.iadd(post_out, s.uint(2)), fin_b)
    s.store_float(b_post, s.iadd(post_out, s.uint(3)), cur_z)
    s.store_float(b_post, s.iadd(post_out, s.uint(4)), vel_u)
    s.store_float(b_post, s.iadd(post_out, s.uint(5)), vel_v)
    s.store_float(b_post, s.iadd(post_out, s.uint(6)), cur_nx)
    s.store_float(b_post, s.iadd(post_out, s.uint(7)), cur_ny)
    return s.finish()


# ============================================================================
# CPU reference (one output pixel; for the GPU parity test)
# ============================================================================


struct SotaMlUpFrame(Movable):
    """CPU copies of what one resolve reads: render-res HDR_OUT (8 words/px), output-res history
    (`rgb z`, 4 words/px), jitter, sharpening and history enable. Reactive is taken as 0."""
    var src_w: Int
    var src_h: Int
    var dst_w: Int
    var dst_h: Int
    var hdr: List[Float32]
    var hist: List[Float32]
    var jx: Float32
    var jy: Float32
    var sharp: Float32
    var hist_on: Bool

    def __init__(out self, src_w: Int, src_h: Int, dst_w: Int, dst_h: Int):
        self.src_w = src_w
        self.src_h = src_h
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.hdr = List[Float32](length=src_w * src_h * 8, fill=0.0)
        self.hist = List[Float32](length=dst_w * dst_h * 4, fill=0.0)
        self.jx = 0.0
        self.jy = 0.0
        self.sharp = 0.0
        self.hist_on = True


def _hist_bilinear(fr: SotaMlUpFrame, px_x: Float32, px_y: Float32) -> Tuple[Float32, Float32, Float32, Float32]:
    var f_dw = Float32(fr.dst_w)
    var f_dh = Float32(fr.dst_h)
    var fx = max(Float32(0.0), min(px_x - Float32(0.5), max(Float32(0.0), f_dw - Float32(1.0))))
    var fy = max(Float32(0.0), min(px_y - Float32(0.5), max(Float32(0.0), f_dh - Float32(1.0))))
    var hx0 = min(fr.dst_w - 1, Int(floor(fx)))
    var hy0 = min(fr.dst_h - 1, Int(floor(fy)))
    var hx1 = min(fr.dst_w - 1, hx0 + 1)
    var hy1 = min(fr.dst_h - 1, hy0 + 1)
    var htx = max(Float32(0.0), min(Float32(1.0), fx - Float32(hx0)))
    var hty = max(Float32(0.0), min(Float32(1.0), fy - Float32(hy0)))
    var ws = [(Float32(1.0) - htx) * (Float32(1.0) - hty), htx * (Float32(1.0) - hty), (Float32(1.0) - htx) * hty, htx * hty]
    var idx = [(hy0 * fr.dst_w + hx0) * 4, (hy0 * fr.dst_w + hx1) * 4, (hy1 * fr.dst_w + hx0) * 4, (hy1 * fr.dst_w + hx1) * 4]
    var o = Array[Float32, 4](fill=0.0)
    for c in range(4):
        for t in range(4):
            o[c] += fr.hist[idx[t] + c] * ws[t]
    return (o[0], o[1], o[2], o[3])


def _catmull_rom_hist(fr: SotaMlUpFrame, prev_u: Float32, prev_v: Float32) -> Tuple[Float32, Float32, Float32, Float32]:
    var sx = prev_u * Float32(fr.dst_w)
    var sy = prev_v * Float32(fr.dst_h)
    var tc1_x = floor(sx - Float32(0.5)) + Float32(0.5)
    var tc1_y = floor(sy - Float32(0.5)) + Float32(0.5)
    var fx = sx - tc1_x
    var fy = sy - tc1_y
    var fx2 = fx * fx
    var fy2 = fy * fy
    var w0_x = fx * (fx * (fx * Float32(-0.5) + Float32(1.0)) + Float32(-0.5))
    var w1_x = fx2 * (fx * Float32(1.5) + Float32(-2.5)) + Float32(1.0)
    var w2_x = fx * (fx * (fx * Float32(-1.5) + Float32(2.0)) + Float32(0.5))
    var w3_x = fx2 * (fx * Float32(0.5) + Float32(-0.5))
    var w0_y = fy * (fy * (fy * Float32(-0.5) + Float32(1.0)) + Float32(-0.5))
    var w1_y = fy2 * (fy * Float32(1.5) + Float32(-2.5)) + Float32(1.0)
    var w2_y = fy * (fy * (fy * Float32(-1.5) + Float32(2.0)) + Float32(0.5))
    var w3_y = fy2 * (fy * Float32(0.5) + Float32(-0.5))
    var w12_x = w1_x + w2_x
    var w12_y = w1_y + w2_y
    var tc12_x = tc1_x + w2_x / max(Float32(1e-5), w12_x)
    var tc12_y = tc1_y + w2_y / max(Float32(1e-5), w12_y)
    var t_n = _hist_bilinear(fr, tc12_x, tc1_y - Float32(1.0))
    var t_w = _hist_bilinear(fr, tc1_x - Float32(1.0), tc12_y)
    var t_c = _hist_bilinear(fr, tc12_x, tc12_y)
    var t_e = _hist_bilinear(fr, tc1_x + Float32(2.0), tc12_y)
    var t_s = _hist_bilinear(fr, tc12_x, tc1_y + Float32(2.0))
    var wn = w12_x * w0_y
    var ww = w0_x * w12_y
    var wc = w12_x * w12_y
    var we = w3_x * w12_y
    var wsv = w12_x * w3_y
    var inv_tw = Float32(1.0) / max(Float32(1e-5), wn + ww + wc + we + wsv)
    var r = max(Float32(0.0), (t_n[0] * wn + t_w[0] * ww + t_c[0] * wc + t_e[0] * we + t_s[0] * wsv) * inv_tw)
    var g = max(Float32(0.0), (t_n[1] * wn + t_w[1] * ww + t_c[1] * wc + t_e[1] * we + t_s[1] * wsv) * inv_tw)
    var b = max(Float32(0.0), (t_n[2] * wn + t_w[2] * ww + t_c[2] * wc + t_e[2] * we + t_s[2] * wsv) * inv_tw)
    return (r, g, b, t_c[3])


def sota_ml_upscale_cpu_pixel(w: SotaMlUpWeights, fr: SotaMlUpFrame, dx: Int, dy: Int) -> Tuple[Float32, Float32, Float32]:
    """CPU reference of `sota_ml_upscale_shader` for one output pixel (reactive = 0)."""
    var sw = fr.src_w
    var shh = fr.src_h
    var f_sw = Float32(sw)
    var f_sh = Float32(shh)
    var f_dw = Float32(fr.dst_w)
    var f_dh = Float32(fr.dst_h)
    var uv_u = (Float32(dx) + Float32(0.5)) / f_dw
    var uv_v = (Float32(dy) + Float32(0.5)) / f_dh
    var fx_raw = uv_u * f_sw + fr.jx - Float32(0.5)
    var fy_raw = uv_v * f_sh - fr.jy - Float32(0.5)
    var fx = max(Float32(0.0), min(fx_raw, max(Float32(0.0), f_sw - Float32(1.0))))
    var fy = max(Float32(0.0), min(fy_raw, max(Float32(0.0), f_sh - Float32(1.0))))
    var ix0 = min(sw - 1, Int(floor(fx)))
    var iy0 = min(shh - 1, Int(floor(fy)))
    var ix1 = min(sw - 1, ix0 + 1)
    var iy1 = min(shh - 1, iy0 + 1)
    var tx = max(Float32(0.0), min(Float32(1.0), fx - Float32(ix0)))
    var ty = max(Float32(0.0), min(Float32(1.0), fy - Float32(iy0)))
    var near_x = floor(fx_raw + Float32(0.5))
    var near_y = floor(fy_raw + Float32(0.5))
    var sdx = fx_raw - near_x
    var sdy = fy_raw - near_y
    var sub_d2 = min(Float32(1.0), sdx * sdx + sdy * sdy)
    var ps = [(iy0 * sw + ix0) * 8, (iy0 * sw + ix1) * 8, (iy1 * sw + ix0) * 8, (iy1 * sw + ix1) * 8]
    var bw = [(Float32(1.0) - tx) * (Float32(1.0) - ty), tx * (Float32(1.0) - ty), (Float32(1.0) - tx) * ty, tx * ty]
    var cur = Array[Float32, 3](fill=0.0)
    for c in range(3):
        for t in range(4):
            cur[c] += fr.hdr[ps[t] + c] * bw[t]
    var ix_c = min(sw - 1, Int(max(Float32(0.0), near_x)))
    var iy_c = min(shh - 1, Int(max(Float32(0.0), near_y)))
    var best_p = (iy_c * sw + ix_c) * 8
    var best_z = fr.hdr[best_p + 3]
    var swk = Float32(0.0)
    var m1 = Array[Float32, 3](fill=0.0)
    var m2 = Array[Float32, 3](fill=0.0)
    var mn = Array[Float32, 3](fill=1e9)
    var mx = Array[Float32, 3](fill=-1e9)
    var lap = Array[Float32, 3](fill=0.0)
    for ny in range(-1, 2):
        for nx in range(-1, 2):
            var sx_n = max(0, min(sw - 1, ix_c + nx))
            var sy_n = max(0, min(shh - 1, iy_c + ny))
            var np = (sy_n * sw + sx_n) * 8
            var rgb = [fr.hdr[np], fr.hdr[np + 1], fr.hdr[np + 2]]
            var nz = fr.hdr[np + 3]
            if nz < best_z:
                best_z = nz
                best_p = np
            for c in range(3):
                if (nx == 0 and abs(ny) == 1) or (ny == 0 and abs(nx) == 1):
                    lap[c] += rgb[c]
                mn[c] = min(mn[c], rgb[c])
                mx[c] = max(mx[c], rgb[c])
            var yv = rgb[0] * Float32(0.25) + rgb[1] * Float32(0.5) + rgb[2] * Float32(0.25)
            var co = (rgb[0] - rgb[2]) * Float32(0.5)
            var cg = rgb[1] * Float32(0.5) + (rgb[0] + rgb[2]) * Float32(-0.25)
            var wk = Float32(1.0) / (max(Float32(0.0), yv) * Float32(0.25) + Float32(1.0))
            swk += wk
            m1[0] += yv * wk
            m2[0] += yv * yv * wk
            m1[1] += co * wk
            m2[1] += co * co * wk
            m1[2] += cg * wk
            m2[2] += cg * cg * wk
    var cur_z = best_z
    var vel_u = fr.hdr[best_p + 4]
    var vel_v = fr.hdr[best_p + 5]
    var sharp = Array[Float32, 3](fill=0.0)
    var amt = max(Float32(0.0), min(Float32(1.0), fr.sharp))
    for c in range(3):
        var raw = (cur[c] - lap[c] * Float32(0.25)) * amt + cur[c]
        sharp[c] = max(Float32(0.0), max(mn[c], min(raw, mx[c])))
    var prev_u = uv_u - vel_u
    var prev_v = uv_v - vel_v
    if not (fr.hist_on and prev_u >= 0.0 and prev_u < 1.0 and prev_v >= 0.0 and prev_v < 1.0):
        return (sharp[0], sharp[1], sharp[2])
    var h = _catmull_rom_hist(fr, prev_u, prev_v)
    var rel_dz = abs(cur_z - h[3]) / max(Float32(0.1), cur_z)
    if not (rel_dz < Float32(0.10)):
        return (sharp[0], sharp[1], sharp[2])
    var vpx_x = vel_u * f_dw
    var vpx_y = vel_v * f_dh
    var vel_px = sqrt(vpx_x * vpx_x + vpx_y * vpx_y)
    var inv_swk = Float32(1.0) / max(Float32(1e-5), swk)
    var mu = Array[Float32, 3](fill=0.0)
    var sg = Array[Float32, 3](fill=0.0)
    for c in range(3):
        mu[c] = m1[c] * inv_swk
        sg[c] = sqrt(max(Float32(1e-7), m2[c] * inv_swk - mu[c] * mu[c]))
    var hyc = [h[0] * Float32(0.25) + h[1] * Float32(0.5) + h[2] * Float32(0.25), (h[0] - h[2]) * Float32(0.5), h[1] * Float32(0.5) + (h[0] + h[2]) * Float32(-0.25)]
    var d = Array[Float32, 3](fill=0.0)
    var r_unit = Float32(0.0)
    for c in range(3):
        d[c] = hyc[c] - mu[c]
        r_unit = max(r_unit, abs(d[c]) / max(Float32(1e-4), sg[c]))
    var f = Array[Float32, 6](fill=0.0)
    f[0] = sub_d2
    f[1] = min(vel_px * Float32(0.25), Float32(4.0))
    f[2] = min(rel_dz * Float32(10.0), Float32(4.0))
    f[3] = min(max(r_unit - Float32(1.0), Float32(0.0)), Float32(8.0)) * Float32(0.25)
    f[4] = min(sg[0] / (max(Float32(0.0), mu[0]) + Float32(0.02)), Float32(4.0)) * Float32(0.5)
    f[5] = f_sw / f_dw
    var gw = sota_ml_up_mlp_cpu(w, f)
    var clip_s = Float32(1.0) / max(Float32(1.0), r_unit / gw[0])
    var yc = d[0] * clip_s + mu[0]
    var co2 = d[1] * clip_s + mu[1]
    var cg2 = d[2] * clip_s + mu[2]
    var tmp = yc - cg2
    var ch = [max(Float32(0.0), tmp + co2), max(Float32(0.0), yc + cg2), max(Float32(0.0), tmp - co2)]
    var hw = Float32(1.0) - gw[1]
    var cw = Float32(1.0) - hw
    return (ch[0] * hw + sharp[0] * cw, ch[1] * hw + sharp[1] * cw, ch[2] * hw + sharp[2] * cw)


# ============================================================================
# Runtime (owned by `FlintAaaVulkanRenderer.sota_ml_up`, driven from the `sota:ml` hook at the TSR resolve)
# ============================================================================


struct SotaMlUpscaler(Movable):
    """Renderer settings of the learned upscaler (`FlintAaaVulkanRenderer.sota_ml_up`).

    - `enabled` (default False, opt-in: the app sets it, or `FLINT_SOTA_ML_UPSCALE=1` at start): replace
      the TSR v3 resolve when weights load; otherwise TSR v3 runs unchanged. Weights on disk alone
      never change the image.
    - `only_upscaling` (default True): use it only when `render < output` (it is trained for 2x).
    - `weights_path`: `.flmlw` file (`FLMLUP01`; default from `sota_ml_up_default_weights_path()`).
    - `capture` (default False): copy the render-resolution HDR_OUT the resolve reads into `cap_buf`
      (8 words per pixel, CPU-readable after the frame) for training data and the parity test. It runs
      for the TSR v3 resolve too, so data does not depend on the upscaler.
    """
    var enabled: Bool
    var only_upscaling: Bool
    var weights_path: String
    var weights: SotaMlUpWeights
    var weights_tried: Bool
    var weights_version: Int
    var built_version: Int
    var pipe: Int
    var ran_last_frame: Bool
    var status: String
    var capture: Bool
    var cap_buf: GpuBuffer

    def __init__(out self):
        self.enabled = env_now("FLINT_SOTA_ML_UPSCALE") == "1"
        self.only_upscaling = True
        self.weights_path = sota_ml_up_default_weights_path()
        self.weights = SotaMlUpWeights()
        self.weights_tried = False
        self.weights_version = 0
        self.built_version = -1
        self.pipe = 0
        self.ran_last_frame = False
        self.status = String("idle")
        self.capture = False
        self.cap_buf = GpuBuffer()

    def record_capture(mut self, mut vk: VulkanGpu, slot: Int, hdr_out: GpuBuffer, max_pix: Int, w: Int, h: Int):
        """Copy the resolve's render-resolution input (when `capture`); before the TSR / learned resolve."""
        if not self.capture:
            return
        if self.cap_buf.size < max_pix * 8 * 4:
            vk.destroy_buffer(self.cap_buf)
            self.cap_buf = vk.create_buffer(max_pix * 8 * 4, local=False, cpu_reads=True)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ))
        vk.copy_buffer(slot, hdr_out, 0, self.cap_buf, 0, w * h * 8 * 4)
        vk.memory_barrier2(slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ), PIPELINE_TRANSFER | PIPELINE_COMPUTE, Int(ACCESS_TRANSFER_WRITE | ACCESS_SHADER_WRITE))

    def set_weights(mut self, var w: SotaMlUpWeights):
        self.weights = w^
        self.weights_tried = True
        self.weights_version += 1

    def has_weights(self) -> Bool:
        return self.weights.valid()

    def ready(mut self, mut vk: VulkanGpu, pipe_layout: Int, upscaling: Bool) -> Bool:
        """True when this frame's resolve should be the learned one (pipeline built)."""
        self.ran_last_frame = False
        if not self.enabled or (self.only_upscaling and not upscaling):
            return False
        if not self.weights_tried:
            self.weights_tried = True
            var w = sota_ml_up_load_weights(self.weights_path)
            if w.valid():
                self.weights = w^
                self.weights_version += 1
                self.status = String("weights ") + self.weights_path
            else:
                self.status = String("no weights at ") + self.weights_path + " (TSR v3 fallback)"
        if not self.weights.valid():
            return False
        if self.built_version != self.weights_version:
            if self.pipe != 0:
                vk.destroy_pipeline(self.pipe)
            self.pipe = vk.create_aaa_compute_pipeline(pipe_layout, sota_ml_upscale_shader(self.weights))
            self.built_version = self.weights_version
            if self.pipe == 0:
                self.status = String("pipeline creation failed (TSR v3 fallback)")
                self.enabled = False
                return False
        self.ran_last_frame = True
        return True

    def destroy(mut self, mut vk: VulkanGpu):
        vk.destroy_buffer(self.cap_buf)
        self.cap_buf = GpuBuffer()
        if self.pipe != 0:
            vk.destroy_pipeline(self.pipe)
            self.pipe = 0
