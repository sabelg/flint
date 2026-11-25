"""Parallel image-shaped ops for the ONNX interpreter (`flint.ml.ops_image`).

Image generators (GAN inpainters, super-resolution nets) spend most of their non-convolution time
in full-resolution point-wise ops and resizes. These are the same operators as the generic ones in
`ops` / `ops_ext`, with the same float32 arithmetic (results are bit-identical), spread over the
`JobPool` with 8-wide SIMD:

  * `par_leaky_relu` (ONNX LeakyRelu), `par_clip` (Clip with scalar bounds);
  * `resize2d`: Resize over the last two axes of an [N, C, H, W] float tensor, `nearest`
    (asymmetric / floor, as `ops.resize_nearest`) or `linear` (H pass then W pass, the coordinate
    modes of `ops_ext.resize_linear`);
  * `pad2d_constant`: constant Pad over the last two axes.

`OnnxModel.lower_image_ops()` routes matching nodes here (op codes 91..94); anything else keeps the
generic path.
"""

from std.math import floor
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import Tensor, fptr_at

comptime _V = SIMD[DType.float32, 8]
comptime _CH: Int = 1 << 15


struct _PwCtx(Movable):
    var x: Int
    var o: Int
    var n: Int
    var kind: Int       # 0 leaky relu, 1 clip
    var a: Float32
    var b: Float32

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.n = 0
        self.kind = 0
        self.a = 0.0
        self.b = 0.0


def _pw_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_PwCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var e0 = job * _CH
    var e1 = min(c[].n, e0 + _CH)
    var px = fptr_at(c[].x)
    var po = fptr_at(c[].o)
    var e = e0
    if c[].kind == 0:
        var al = _V(c[].a)
        var z = _V(0.0)
        while e + 8 <= e1:
            var v = px.unsafe_offset(e).unsafe_load[width=8, alignment=1]()
            po.unsafe_offset(e).unsafe_store[alignment=1](v.gt(z).select(v, v * al))
            e += 8
        while e < e1:
            var v = px.unsafe_offset(e).unsafe_load[alignment=1]()
            po.unsafe_offset(e).unsafe_store[alignment=1](v if v > 0 else v * c[].a)
            e += 1
    else:
        var lo = _V(c[].a)
        var hi = _V(c[].b)
        while e + 8 <= e1:
            var v = px.unsafe_offset(e).unsafe_load[width=8, alignment=1]()
            po.unsafe_offset(e).unsafe_store[alignment=1](min(max(v, lo), hi))
            e += 8
        while e < e1:
            var v = px.unsafe_offset(e).unsafe_load[alignment=1]()
            po.unsafe_offset(e).unsafe_store[alignment=1](min(max(v, c[].a), c[].b))
            e += 1


def _pointwise(mut pool: JobPool, x: Tensor, kind: Int, a: Float32, b: Float32) -> Tensor:
    var out = Tensor.zeros(x.shape.copy())
    var c = _PwCtx()
    c.x = x.fptr()
    c.o = out.fptr()
    c.n = x.numel()
    c.kind = kind
    c.a = a
    c.b = b
    run_parallel_ctx(pool, (c.n + _CH - 1) // _CH, c, _pw_worker)
    return out^


def par_leaky_relu(mut pool: JobPool, x: Tensor, alpha: Float32) -> Tensor:
    """LeakyRelu of a float tensor: x if x > 0 else alpha * x."""
    return _pointwise(pool, x, 0, alpha, 0.0)


def par_clip(mut pool: JobPool, x: Tensor, lo: Float32, hi: Float32) -> Tensor:
    """Clip of a float tensor to [lo, hi]."""
    return _pointwise(pool, x, 1, lo, hi)


# ----------------------------------------------------------------------------- resize


def _src_coord(mode: String, xo: Int, scale: Float64, n_in: Int, n_out: Int) -> Float64:
    """`ops_ext._src_coord` (the ONNX coordinate_transformation_mode formulas)."""
    var x = Float64(xo)
    if mode == "align_corners":
        return 0.0 if n_out <= 1 else x * Float64(n_in - 1) / Float64(n_out - 1)
    if mode == "asymmetric":
        return x / scale
    if mode == "pytorch_half_pixel":
        return (x + 0.5) / scale - 0.5 if n_out > 1 else 0.0
    if mode == "half_pixel_symmetric":
        var adj = Float64(n_out) / (scale * Float64(n_in))
        var center = Float64(n_in) * 0.5
        var off = center * (1.0 - adj)
        return off + (x + 0.5) / scale - 0.5
    return (x + 0.5) / scale - 0.5


struct _Taps(Movable):
    var i0: List[Int]
    var i1: List[Int]
    var lam: List[Float32]
    var identity: Bool

    def __init__(out self):
        self.i0 = List[Int]()
        self.i1 = List[Int]()
        self.lam = List[Float32]()
        self.identity = False


def _linear_taps(mode: String, n_in: Int, n_out: Int, scale_in: Float64) -> _Taps:
    var t = _Taps()
    t.identity = n_in == n_out and (scale_in == 0 or scale_in == 1)
    var sc = scale_in if scale_in > 0 else Float64(n_out) / Float64(n_in)
    for i in range(n_out):
        var s = _src_coord(mode, i, sc, n_in, n_out)
        s = min(max(s, 0.0), Float64(n_in - 1))
        var i0 = Int(floor(s))
        t.i0.append(i0)
        t.i1.append(min(i0 + 1, n_in - 1))
        t.lam.append(Float32(s - Float64(i0)))
    return t^


def _nearest_taps(n_in: Int, n_out: Int) -> List[Int]:
    var out = List[Int]()
    var scale = Float64(n_out) / Float64(n_in)
    for i in range(n_out):
        var s = Int(floor(Float64(i) / scale))
        out.append(min(max(s, 0), n_in - 1))
    return out^


struct _RsCtx(Movable):
    var x: Int
    var o: Int
    var h: Int
    var w: Int
    var oh: Int
    var ow: Int
    var linear: Bool
    var hy: _Taps
    var wx: _Taps
    var ny: List[Int]
    var nx: List[Int]

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.h = 0
        self.w = 0
        self.oh = 0
        self.ow = 0
        self.linear = True
        self.hy = _Taps()
        self.wx = _Taps()
        self.ny = List[Int]()
        self.nx = List[Int]()


def _rs_worker(ctx_addr: Int, job: Int):
    """One output plane (n, c)."""
    var c = Pointer[_RsCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var H = c[].h
    var W = c[].w
    var OH = c[].oh
    var OW = c[].ow
    var px = fptr_at(c[].x + job * H * W * 4)
    var po = fptr_at(c[].o + job * OH * OW * 4)
    if not c[].linear:
        for y in range(OH):
            var sr = px.unsafe_offset(c[].ny[y] * W)
            var orow = po.unsafe_offset(y * OW)
            for xo in range(OW):
                orow.unsafe_offset(xo).unsafe_store[alignment=1](sr.unsafe_offset(c[].nx[xo]).unsafe_load[alignment=1]())
        return
    var row = List[Float32](length=W, fill=0.0)
    var pr = row.unsafe_ptr()
    for y in range(OH):
        # H pass (exactly the generic op's first axis pass), then W pass
        if c[].hy.identity:
            var sr = px.unsafe_offset(y * W)
            for k in range(W):
                pr.unsafe_offset(k).unsafe_store(sr.unsafe_offset(k).unsafe_load[alignment=1]())
        else:
            var r0 = px.unsafe_offset(c[].hy.i0[y] * W)
            var r1 = px.unsafe_offset(c[].hy.i1[y] * W)
            var lam = c[].hy.lam[y]
            var om = Float32(1) - lam
            for k in range(W):
                pr.unsafe_offset(k).unsafe_store(r0.unsafe_offset(k).unsafe_load[alignment=1]() * om
                                                 + r1.unsafe_offset(k).unsafe_load[alignment=1]() * lam)
        var orow = po.unsafe_offset(y * OW)
        if c[].wx.identity:
            for k in range(OW):
                orow.unsafe_offset(k).unsafe_store[alignment=1](pr.unsafe_offset(k).unsafe_load())
        else:
            for xo in range(OW):
                var lam = c[].wx.lam[xo]
                orow.unsafe_offset(xo).unsafe_store[alignment=1](
                    pr.unsafe_offset(c[].wx.i0[xo]).unsafe_load() * (Float32(1) - lam)
                    + pr.unsafe_offset(c[].wx.i1[xo]).unsafe_load() * lam)
    _ = row^


def resize2d(mut pool: JobPool, x: Tensor, oh: Int, ow: Int, linear: Bool, mode: String, scale_h: Float64,
             scale_w: Float64) raises -> Tensor:
    """Resize the last two axes of a float [N, C, H, W] tensor to [oh, ow] (`scale_*` = 0: from sizes)."""
    if x.rank() != 4 or not x.is_float():
        raise Error("resize2d: needs a float [N, C, H, W] tensor")
    var c = _RsCtx()
    c.h = x.shape[2]
    c.w = x.shape[3]
    c.oh = oh
    c.ow = ow
    c.linear = linear
    if linear:
        c.hy = _linear_taps(mode, c.h, oh, scale_h)
        c.wx = _linear_taps(mode, c.w, ow, scale_w)
    else:
        c.ny = _nearest_taps(c.h, oh)
        c.nx = _nearest_taps(c.w, ow)
    var out = Tensor.zeros([x.shape[0], x.shape[1], oh, ow])
    c.x = x.fptr()
    c.o = out.fptr()
    run_parallel_ctx(pool, x.shape[0] * x.shape[1], c, _rs_worker)
    return out^


# ----------------------------------------------------------------------------- pad


struct _PadCtx(Movable):
    var x: Int
    var o: Int
    var h: Int
    var w: Int
    var oh: Int
    var ow: Int
    var top: Int
    var left: Int
    var value: Float32

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.h = 0
        self.w = 0
        self.oh = 0
        self.ow = 0
        self.top = 0
        self.left = 0
        self.value = 0.0


def _pad_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_PadCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var px = fptr_at(c[].x + job * c[].h * c[].w * 4)
    var po = fptr_at(c[].o + job * c[].oh * c[].ow * 4)
    for y in range(c[].oh):
        var sy = y - c[].top
        var orow = po.unsafe_offset(y * c[].ow)
        for xo in range(c[].ow):
            var sx = xo - c[].left
            var v = c[].value
            if sy >= 0 and sy < c[].h and sx >= 0 and sx < c[].w:
                v = px.unsafe_offset(sy * c[].w + sx).unsafe_load[alignment=1]()
            orow.unsafe_offset(xo).unsafe_store[alignment=1](v)


def pad2d_constant(mut pool: JobPool, x: Tensor, top: Int, left: Int, bottom: Int, right: Int,
                   value: Float32) raises -> Tensor:
    """Constant Pad of the last two axes of a float [N, C, H, W] tensor (negative pads crop)."""
    if x.rank() != 4 or not x.is_float():
        raise Error("pad2d_constant: needs a float [N, C, H, W] tensor")
    var c = _PadCtx()
    c.h = x.shape[2]
    c.w = x.shape[3]
    c.oh = c.h + top + bottom
    c.ow = c.w + left + right
    if c.oh <= 0 or c.ow <= 0:
        raise Error("pad2d_constant: empty output")
    c.top = top
    c.left = left
    c.value = value
    var out = Tensor.zeros([x.shape[0], x.shape[1], c.oh, c.ow])
    c.x = x.fptr()
    c.o = out.fptr()
    run_parallel_ctx(pool, x.shape[0] * x.shape[1], c, _pad_worker)
    return out^
