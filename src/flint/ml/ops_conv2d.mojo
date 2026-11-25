"""2-D convolution for the ONNX interpreter (`flint.ml.ops_conv2d`).

`conv2d_nchw` is ONNX `Conv` on [N, C, H, W] tensors with any kernel size, stride, dilation,
asymmetric padding and group count:

  * depthwise / grouped with one input channel per group (`cin / group == 1`, e.g. MobileNet-style
    3 x 3 and the 4 x 4 FIR up / down-sampling filters of StyleGAN-family generators): a direct
    loop per output channel, rows spread over the `JobPool`, 8-wide SIMD along the row for stride 1;
  * dense groups: 1 x 1 / stride 1 / no padding is a plain `W_g [cout_g, cin_g] @ X_g [cin_g, H*W]`
    product (no im2col); every other kernel builds im2col column blocks and multiplies them.
    `conv2d_nchw_gpu` runs the dense products on the MAX GPU (`GpuMatmul`, weights cached by key).

The interpreter lowers rank-4 `Conv` nodes with a static 4-D weight to `OP_CONV2D` (code 90): k x k
kernels always, and 1 x 1 / 1 x k kernels after `OnnxModel.lower_conv2d(True)`.
"""

from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.ops import matmul_2d
from flint.ml.gpu import GpuMatmul

comptime _V = SIMD[DType.float32, 8]


@fieldwise_init
struct Conv2dGeom(Copyable, Movable):
    """Shapes and hyper-parameters of one 2-D convolution."""
    var n: Int
    var cin: Int
    var h: Int
    var w: Int
    var cout: Int
    var cin_g: Int
    var cout_g: Int
    var group: Int
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var dh: Int
    var dw: Int
    var pt: Int
    var pl: Int
    var ho: Int
    var wo: Int


def conv2d_geom(x: Tensor, w: Tensor, pads: List[Int], strides: List[Int], dils: List[Int],
                group: Int) raises -> Conv2dGeom:
    """Validate an ONNX Conv on [N, C, H, W] and compute its output size.
    `pads` is ONNX order [top, left, bottom, right] (empty = 0)."""
    if x.rank() != 4 or w.rank() != 4:
        raise Error("Conv2d: expects a 4-D input and a 4-D weight")
    var g = max(group, 1)
    var cin = x.shape[1]
    var cout = w.shape[0]
    var cin_g = w.shape[1]
    if cin_g * g != cin or cout % g != 0:
        raise Error("Conv2d: channels do not match the group count")
    var pt = pads[0] if len(pads) == 4 else 0
    var pl = pads[1] if len(pads) == 4 else 0
    var pb = pads[2] if len(pads) == 4 else 0
    var pr = pads[3] if len(pads) == 4 else 0
    var sh = strides[0] if len(strides) == 2 else 1
    var sw = strides[1] if len(strides) == 2 else 1
    var dh = dils[0] if len(dils) == 2 else 1
    var dw = dils[1] if len(dils) == 2 else 1
    var kh = w.shape[2]
    var kw = w.shape[3]
    var ho = (x.shape[2] + pt + pb - dh * (kh - 1) - 1) // sh + 1
    var wo = (x.shape[3] + pl + pr - dw * (kw - 1) - 1) // sw + 1
    if ho <= 0 or wo <= 0:
        raise Error("Conv2d: empty output")
    return Conv2dGeom(x.shape[0], cin, x.shape[2], x.shape[3], cout, cin_g, cout // g, g, kh, kw, sh, sw, dh, dw,
                      pt, pl, ho, wo)


# ----------------------------------------------------------------------------- depthwise


struct _DwCtx(Movable):
    var x: Int
    var w: Int
    var b: Int
    var has_b: Bool
    var o: Int
    var g: Conv2dGeom
    var rows: Int      # output rows per job

    def __init__(out self, g: Conv2dGeom):
        self.x = 0
        self.w = 0
        self.b = 0
        self.has_b = False
        self.o = 0
        self.g = g.copy()
        self.rows = 1


def _dw_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_DwCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    ref g = c[].g
    var row_jobs = (g.ho + c[].rows - 1) // c[].rows
    var plane = job // row_jobs              # (batch, out channel)
    var r0 = (job % row_jobs) * c[].rows
    var r1 = min(g.ho, r0 + c[].rows)
    var bi = plane // g.cout
    var co = plane % g.cout
    var ci = co // g.cout_g                  # cin_g == 1: one input channel per group
    var px = fptr_at(c[].x + ((bi * g.cin + ci) * g.h * g.w) * 4)
    var pw = fptr_at(c[].w + co * g.kh * g.kw * 4)
    var po = fptr_at(c[].o + ((bi * g.cout + co) * g.ho * g.wo) * 4)
    var bias: Float32 = 0.0
    if c[].has_b:
        bias = fptr_at(c[].b).unsafe_offset(co).unsafe_load[alignment=1]()
    for oy in range(r0, r1):
        var orow = po.unsafe_offset(oy * g.wo)
        for ox in range(g.wo):
            orow.unsafe_offset(ox).unsafe_store[alignment=1](bias)
        for ky in range(g.kh):
            var iy = oy * g.sh - g.pt + ky * g.dh
            if iy < 0 or iy >= g.h:
                continue
            var irow = px.unsafe_offset(iy * g.w)
            for kx in range(g.kw):
                var wv = pw.unsafe_offset(ky * g.kw + kx).unsafe_load[alignment=1]()
                var off = kx * g.dw - g.pl          # ix = ox * sw + off
                # valid ox: 0 <= ox * sw + off < w
                var ox0 = 0
                if off < 0:
                    ox0 = (-off + g.sw - 1) // g.sw
                var ox1 = g.wo
                var lim = (g.w - 1 - off)
                if lim < 0:
                    continue
                ox1 = min(g.wo, lim // g.sw + 1)
                if g.sw == 1:
                    var wvv = _V(wv)
                    var ox = ox0
                    while ox + 8 <= ox1:
                        var a = orow.unsafe_offset(ox).unsafe_load[width=8, alignment=1]()
                        var v = irow.unsafe_offset(ox + off).unsafe_load[width=8, alignment=1]()
                        orow.unsafe_offset(ox).unsafe_store[alignment=1](a + v * wvv)
                        ox += 8
                    while ox < ox1:
                        var a = orow.unsafe_offset(ox).unsafe_load[alignment=1]()
                        orow.unsafe_offset(ox).unsafe_store[alignment=1](
                            a + irow.unsafe_offset(ox + off).unsafe_load[alignment=1]() * wv)
                        ox += 1
                else:
                    for ox in range(ox0, ox1):
                        var a = orow.unsafe_offset(ox).unsafe_load[alignment=1]()
                        orow.unsafe_offset(ox).unsafe_store[alignment=1](
                            a + irow.unsafe_offset(ox * g.sw + off).unsafe_load[alignment=1]() * wv)


def _depthwise(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, g: Conv2dGeom) raises -> Tensor:
    var out = Tensor.zeros([g.n, g.cout, g.ho, g.wo])
    var c = _DwCtx(g)
    c.x = x.fptr()
    c.w = w.fptr()
    c.b = b.fptr() if has_b else 0
    c.has_b = has_b
    c.o = out.fptr()
    var planes = g.n * g.cout
    # aim at >= 4 jobs per worker without making jobs smaller than ~16 K multiply-adds
    var per_row = max(1, g.wo * g.kh * g.kw)
    c.rows = max(1, min(g.ho, (1 << 14) // per_row))
    var row_jobs = (g.ho + c.rows - 1) // c.rows
    run_parallel_ctx(pool, planes * row_jobs, c, _dw_worker)
    return out^


# ----------------------------------------------------------------------------- dense


struct _ColCtx(Movable):
    var x: Int          # input of this (batch, group): [cin_g, h, w]
    var col: Int        # [K, cols] column block
    var p0: Int         # first output pixel of the block
    var cols: Int
    var g: Conv2dGeom

    def __init__(out self, g: Conv2dGeom):
        self.x = 0
        self.col = 0
        self.p0 = 0
        self.cols = 0
        self.g = g.copy()


def _col_worker(ctx_addr: Int, job: Int):
    """Row `job` of the im2col block: tap (ci, ky, kx) over the block's output pixels."""
    var c = Pointer[_ColCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    ref g = c[].g
    var taps = g.kh * g.kw
    var ci = job // taps
    var t = job % taps
    var ky = t // g.kw
    var kx = t % g.kw
    var px = fptr_at(c[].x + ci * g.h * g.w * 4)
    var pc = fptr_at(c[].col + job * c[].cols * 4)
    for j in range(c[].cols):
        var p = c[].p0 + j
        var oy = p // g.wo
        var ox = p % g.wo
        var iy = oy * g.sh - g.pt + ky * g.dh
        var ix = ox * g.sw - g.pl + kx * g.dw
        var v: Float32 = 0.0
        if iy >= 0 and iy < g.h and ix >= 0 and ix < g.w:
            v = px.unsafe_offset(iy * g.w + ix).unsafe_load[alignment=1]()
        pc.unsafe_offset(j).unsafe_store[alignment=1](v)


struct _BiasCtx(Movable):
    var o: Int
    var b: Int
    var plane: Int
    var cout: Int

    def __init__(out self):
        self.o = 0
        self.b = 0
        self.plane = 0
        self.cout = 0


def _bias_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_BiasCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var bv = fptr_at(c[].b).unsafe_offset(job % c[].cout).unsafe_load[alignment=1]()
    var po = fptr_at(c[].o + job * c[].plane * 4)
    var bb = _V(bv)
    var n = c[].plane
    var k = 0
    while k + 8 <= n:
        po.unsafe_offset(k).unsafe_store[alignment=1](po.unsafe_offset(k).unsafe_load[width=8, alignment=1]() + bb)
        k += 8
    while k < n:
        po.unsafe_offset(k).unsafe_store[alignment=1](po.unsafe_offset(k).unsafe_load[alignment=1]() + bv)
        k += 1


def _add_bias(mut pool: JobPool, o: Tensor, b: Tensor, g: Conv2dGeom):
    var c = _BiasCtx()
    c.o = o.fptr()
    c.b = b.fptr()
    c.plane = g.ho * g.wo
    c.cout = g.cout
    run_parallel_ctx(pool, g.n * g.cout, c, _bias_worker)


def _is_pointwise(g: Conv2dGeom) -> Bool:
    return g.kh == 1 and g.kw == 1 and g.sh == 1 and g.sw == 1 and g.pt == 0 and g.pl == 0 \
        and g.ho == g.h and g.wo == g.w


def _dense(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, g: Conv2dGeom) raises -> Tensor:
    var out = Tensor.zeros([g.n, g.cout, g.ho, g.wo])
    var P = g.ho * g.wo
    var K = g.cin_g * g.kh * g.kw
    if _is_pointwise(g):
        for bi in range(g.n):
            for gi in range(g.group):
                var xa = x.fptr() + ((bi * g.cin + gi * g.cin_g) * P) * 4
                var wa = w.fptr() + gi * g.cout_g * K * 4
                var oa = out.fptr() + ((bi * g.cout + gi * g.cout_g) * P) * 4
                matmul_2d(pool, wa, xa, oa, 0, g.cout_g, K, P, False)
    else:
        var cols = max(64, min(P, (1 << 22) // max(K, 1)))
        var col = List[Float32](length=K * cols, fill=0.0)
        var tmp = List[Float32](length=g.cout_g * cols, fill=0.0)
        var c = _ColCtx(g)
        c.col = Int(col.unsafe_ptr())
        for bi in range(g.n):
            for gi in range(g.group):
                c.x = x.fptr() + ((bi * g.cin + gi * g.cin_g) * g.h * g.w) * 4
                var wa = w.fptr() + gi * g.cout_g * K * 4
                var p0 = 0
                while p0 < P:
                    var n = min(cols, P - p0)
                    c.p0 = p0
                    c.cols = n
                    run_parallel_ctx(pool, K, c, _col_worker)
                    matmul_2d(pool, wa, Int(col.unsafe_ptr()), Int(tmp.unsafe_ptr()), 0, g.cout_g, K, n, False)
                    var po = out.fptr() + ((bi * g.cout + gi * g.cout_g) * P + p0) * 4
                    for co in range(g.cout_g):
                        var src = fptr_at(Int(tmp.unsafe_ptr()) + co * n * 4)
                        var dst = fptr_at(po + co * P * 4)
                        for j in range(n):
                            dst.unsafe_offset(j).unsafe_store[alignment=1](src.unsafe_offset(j).unsafe_load[alignment=1]())
                    p0 += n
        _ = col^
        _ = tmp^
    if has_b:
        _add_bias(pool, out, b, g)
    return out^


def conv2d_nchw(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pads: List[Int],
                strides: List[Int], dils: List[Int], group: Int) raises -> Tensor:
    """ONNX Conv on [N, C, H, W] (CPU). `pads` = [top, left, bottom, right]."""
    var g = conv2d_geom(x, w, pads, strides, dils, group)
    if g.cin_g == 1 and g.group > 1:
        return _depthwise(pool, x, w, b, has_b, g)
    return _dense(pool, x, w, b, has_b, g)


# ----------------------------------------------------------------------------- GPU


struct _TCtx(Movable):
    var src: Int
    var dst: Int
    var rows: Int
    var cols: Int
    var per: Int

    def __init__(out self):
        self.src = 0
        self.dst = 0
        self.rows = 0
        self.cols = 0
        self.per = 1


def _transpose_worker(ctx_addr: Int, job: Int):
    """dst [cols, rows] = src [rows, cols]^T for a band of source rows."""
    var c = Pointer[_TCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var r0 = job * c[].per
    var r1 = min(c[].rows, r0 + c[].per)
    var ps = fptr_at(c[].src)
    var pd = fptr_at(c[].dst)
    var R = c[].rows
    var C = c[].cols
    # 8 x 8-ish tiles along the columns keep both sides cache friendly
    var j0 = 0
    while j0 < C:
        var j1 = min(C, j0 + 64)
        for r in range(r0, r1):
            for j in range(j0, j1):
                pd.unsafe_offset(j * R + r).unsafe_store[alignment=1](ps.unsafe_offset(r * C + j).unsafe_load[alignment=1]())
        j0 = j1


def _transpose(mut pool: JobPool, src: Int, dst: Int, rows: Int, cols: Int):
    var c = _TCtx()
    c.src = src
    c.dst = dst
    c.rows = rows
    c.cols = cols
    c.per = max(1, min(rows, (1 << 15) // max(cols, 1)))
    run_parallel_ctx(pool, (rows + c.per - 1) // c.per, c, _transpose_worker)


def conv2d_nchw_gpu(mut gpu: GpuMatmul, mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool,
                    pads: List[Int], strides: List[Int], dils: List[Int], group: Int, key: Int) raises -> Tensor:
    """`conv2d_nchw` with the dense products on the GPU: per (batch, group) the transposed input
    (1 x 1) or im2col block [P, K] times the cached W_g^T [K, cout_g]. Depthwise stays on the CPU.
    `key` identifies the (static) weight in the device cache."""
    var g = conv2d_geom(x, w, pads, strides, dils, group)
    if g.cin_g == 1 and g.group > 1:
        return _depthwise(pool, x, w, b, has_b, g)
    var out = Tensor.zeros([g.n, g.cout, g.ho, g.wo])
    var P = g.ho * g.wo
    var K = g.cin_g * g.kh * g.kw
    var a = List[Float32](length=P * K, fill=0.0)
    var tmp = List[Float32](length=P * g.cout_g, fill=0.0)
    var pw = _is_pointwise(g)
    var c = _ColCtx(g)
    var colT = List[Float32]()
    if not pw:
        colT = List[Float32](length=K * P, fill=0.0)
    for gi in range(g.group):
        var widx = gpu.weight((1 << 40) + key * 64 + gi, w.fptr() + gi * g.cout_g * K * 4, g.cout_g, K, True)
        for bi in range(g.n):
            var xa = x.fptr() + ((bi * g.cin + gi * g.cin_g) * g.h * g.w) * 4
            if pw:
                _transpose(pool, xa, Int(a.unsafe_ptr()), K, P)          # [K, P] -> [P, K]
            else:
                c.x = xa
                c.col = Int(colT.unsafe_ptr())
                c.p0 = 0
                c.cols = P
                run_parallel_ctx(pool, K, c, _col_worker)              # [K, P]
                _transpose(pool, Int(colT.unsafe_ptr()), Int(a.unsafe_ptr()), K, P)
            gpu.matmul(Int(a.unsafe_ptr()), widx, Int(tmp.unsafe_ptr()), 0, 0, P, K, g.cout_g)
            var oa = out.fptr() + ((bi * g.cout + gi * g.cout_g) * P) * 4
            _transpose(pool, Int(tmp.unsafe_ptr()), oa, P, g.cout_g)     # [P, cout_g] -> [cout_g, P]
    _ = a^
    _ = tmp^
    _ = colT^
    if has_b:
        _add_bias(pool, out, b, g)
    return out^
