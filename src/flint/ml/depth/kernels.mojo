"""MAX GPU kernels for `flint.ml.depth` (channels-last [B, H, W, C] float32 activations).

The dense work (linear layers, implicit-GEMM convolutions, attention) runs on
`flint.ml.diffusion.nn.Runtime`; these cover the rest of the ViT + DPT / ConvStack heads:
ReLU, bilinear resampling (`align_corners` either way, PyTorch index maths in float32),
the depth-to-space scatter of a stride == kernel transposed convolution, replicate padding,
2-D rotary position embeddings and strided row adds.
"""

from std.math import sin, cos
from max.gpu import global_idx
from flint.ml.gpu import FP

comptime DP_EW: Int = 256


def _dp_relu_kernel(x: FP, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        var v = x.unsafe_offset(i).unsafe_load()
        x.unsafe_offset(i).unsafe_store(v if v > 0 else Float32(0))


@always_inline
def bilinear_src(o: Int, n_in: Int, n_out: Int, align: Bool) -> Tuple[Int, Int, Float32]:
    """(i0, i1, lambda) of output index `o` (PyTorch `upsample_bilinear2d`, float32 scale)."""
    var s: Float32
    if align:
        var sc: Float32 = 0.0
        if n_out > 1:
            sc = Float32(n_in - 1) / Float32(n_out - 1)
        s = sc * Float32(o)
    else:
        var sc = Float32(n_in) / Float32(n_out)
        s = sc * (Float32(o) + 0.5) - 0.5
        if s < 0:
            s = 0
    var i0 = Int(s)
    if i0 > n_in - 1:
        i0 = n_in - 1
    var i1 = i0 + 1 if i0 < n_in - 1 else i0
    var l = s - Float32(i0)
    return (i0, i1, l)


def _dp_bilinear_kernel(src: FP, dst: FP, B: Int32, H: Int32, W: Int32, C: Int32, Ho: Int32, Wo: Int32,
                        align: Int32):
    var i = Int(global_idx.x)
    var cc = Int(C)
    var ho = Int(Ho)
    var wo = Int(Wo)
    if i >= Int(B) * ho * wo * cc:
        return
    var c = i % cc
    var r = i // cc
    var ox = r % wo
    var r2 = r // wo
    var oy = r2 % ho
    var b = r2 // ho
    var hh = Int(H)
    var ww = Int(W)
    var yv = bilinear_src(oy, hh, ho, align != 0)
    var xv = bilinear_src(ox, ww, wo, align != 0)
    var base = b * hh * ww
    var v00 = src.unsafe_offset((base + yv[0] * ww + xv[0]) * cc + c).unsafe_load()
    var v01 = src.unsafe_offset((base + yv[0] * ww + xv[1]) * cc + c).unsafe_load()
    var v10 = src.unsafe_offset((base + yv[1] * ww + xv[0]) * cc + c).unsafe_load()
    var v11 = src.unsafe_offset((base + yv[1] * ww + xv[1]) * cc + c).unsafe_load()
    var ly = yv[2]
    var lx = xv[2]
    var v = (1 - ly) * ((1 - lx) * v00 + lx * v01) + ly * ((1 - lx) * v10 + lx * v11)
    dst.unsafe_offset(i).unsafe_store(v)


def _dp_d2s_kernel(src: FP, dst: FP, B: Int32, H: Int32, W: Int32, C: Int32, s: Int32):
    """src [B, H, W, s * s * C] (column (dy * s + dx) * C + c) -> dst [B, H * s, W * s, C]."""
    var i = Int(global_idx.x)
    var cc = Int(C)
    var ss = Int(s)
    var hh = Int(H)
    var ww = Int(W)
    var wo = ww * ss
    var ho = hh * ss
    if i >= Int(B) * ho * wo * cc:
        return
    var c = i % cc
    var r = i // cc
    var ox = r % wo
    var r2 = r // wo
    var oy = r2 % ho
    var b = r2 // ho
    var y = oy // ss
    var x = ox // ss
    var t = (oy % ss) * ss + ox % ss
    dst.unsafe_offset(i).unsafe_store(src.unsafe_offset(((b * hh + y) * ww + x) * ss * ss * cc + t * cc + c).unsafe_load())


def _dp_pad_rep_kernel(src: FP, dst: FP, B: Int32, H: Int32, W: Int32, C: Int32, p: Int32):
    var i = Int(global_idx.x)
    var cc = Int(C)
    var pp = Int(p)
    var hh = Int(H)
    var ww = Int(W)
    var wo = ww + 2 * pp
    var ho = hh + 2 * pp
    if i >= Int(B) * ho * wo * cc:
        return
    var c = i % cc
    var r = i // cc
    var ox = r % wo
    var r2 = r // wo
    var oy = r2 % ho
    var b = r2 // ho
    var y = min(max(oy - pp, 0), hh - 1)
    var x = min(max(ox - pp, 0), ww - 1)
    dst.unsafe_offset(i).unsafe_store(src.unsafe_offset(((b * hh + y) * ww + x) * cc + c).unsafe_load())


def _dp_rope_kernel(x: FP, pos: FP, freq: FP, rows: Int32, heads: Int32, hd: Int32, ld: Int32):
    """2-D RoPE in place on x [rows, ld] (heads x hd from column 0): the first half of each head
    rotates with the row position pos[2r], the second half with the column position pos[2r+1];
    within a half of F = hd / 2 features, (x[i], x[i + F/2]) rotate by pos * freq[i]."""
    var e = Int(global_idx.x)
    var D = Int(hd)
    var F = D // 2
    var Q = F // 2
    var H = Int(heads)
    if e >= Int(rows) * H * 2 * Q:
        return
    var i = e % Q
    var t = e // Q
    var axis = t % 2
    var t2 = t // 2
    var h = t2 % H
    var r = t2 // H
    var p = pos.unsafe_offset(2 * r + axis).unsafe_load()
    var a = p * freq.unsafe_offset(i).unsafe_load()
    var cs = cos(a)
    var sn = sin(a)
    var base = r * Int(ld) + h * D + axis * F
    var x1 = x.unsafe_offset(base + i).unsafe_load()
    var x2 = x.unsafe_offset(base + Q + i).unsafe_load()
    x.unsafe_offset(base + i).unsafe_store(x1 * cs - x2 * sn)
    x.unsafe_offset(base + Q + i).unsafe_store(x2 * cs + x1 * sn)


def _dp_add_rows_kernel(dst: FP, src: FP, total: Int32, chunk: Int32, d_stride: Int32, d_off: Int32,
                        s_stride: Int32):
    """dst[o * d_stride + d_off + c] += src[o * s_stride + c] for (o, c) in outer x chunk."""
    var e = Int(global_idx.x)
    if e >= Int(total):
        return
    var ch = Int(chunk)
    var o = e // ch
    var c = e % ch
    var p = dst.unsafe_offset(o * Int(d_stride) + Int(d_off) + c)
    p.unsafe_store(p.unsafe_load() + src.unsafe_offset(o * Int(s_stride) + c).unsafe_load())
