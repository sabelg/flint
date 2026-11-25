"""Layer runtime for `flint.ml.diffusion`: one API, CPU (SIMD + JobPool) or MAX GPU.

Activations (`Act`) are channels-last: `b` images of `h x w` pixels (or `h` tokens, `w = 1`) by
`c` channels, i.e. a row-major `[b * h * w, c]` matrix, held either in host memory or in a pooled
device buffer. Every dense layer is then a GEMM:

  * `linear`: rows x [k, n] (+ bias) — the weight is stored transposed once at load;
  * `conv`: implicit GEMM. A gather table maps each output pixel and tap to an input row (or to
    a zero row for padding), so stride, asymmetric padding, a fused nearest-neighbour 2x upsample
    (`up = 2`) and **wrap-around padding** (`PAD_WRAP`, tileable / seamless output — the same as
    rolling the feature maps before every convolution) are just different tables. The GPU runs
    the tiled SGEMM of `flint.ml.gpu` in gather mode; the CPU builds im2col blocks in parallel and
    calls `ops.matmul_2d`;
  * `attention`: per-head softmax(scale Q K^T) V (GEMM / softmax / GEMM on the GPU, the fused
    blocked kernel of `flint.ml.fast` on the CPU), optionally causal for CLIP text.

Weights are read from safetensors (F32 / F16 / BF16) and kept as float32, resident on the GPU
when one is used. Values are freed explicitly with `rt.free(act^)` so device buffers recycle.
"""

from std.collections import Dict
from std.ffi import external_call
from std.math import exp, erf, sqrt
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.safetensors import SafeTensors
from flint.ml.ops import matmul_2d
from flint.ml.fast import attention as cpu_attention
from flint.ml.gpu import GpuMatmul, FP, _mm_tiled, TBM, TBN
from flint.ml.diffusion.kernels import (
    _dk_act_kernel, _dk_add_kernel, _dk_scale_kernel, _dk_fill_kernel, _dk_bcast_add_kernel, _dk_geglu_kernel,
    _dk_group_norm_kernel, _dk_heads_kernel, _dk_merge_kernel, _dk_softmax_kernel, DK_EW, DK_RED,
    ACT_SILU, ACT_GELU, ACT_QUICK_GELU,
)

comptime PAD_ZERO: Int = 0
comptime PAD_WRAP: Int = 1

comptime V8 = SIMD[DType.float32, 8]


# ----------------------------------------------------------------------------- values

struct Act(Movable):
    """A channels-last activation: [b, h, w, c] in host memory or a device pool slot."""
    var b: Int
    var h: Int
    var w: Int
    var c: Int
    var host: List[Float32]
    var slot: Int

    def __init__(out self):
        self.b = 0
        self.h = 0
        self.w = 0
        self.c = 0
        self.host = List[Float32]()
        self.slot = -1

    def rows(self) -> Int:
        return self.b * self.h * self.w

    def numel(self) -> Int:
        return self.b * self.h * self.w * self.c


struct Linear(Copyable, Movable):
    var k: Int
    var n: Int
    var w: List[Float32]     # host [k, n] (CPU runtime only)
    var bias: List[Float32]
    var has_bias: Bool
    var wi: Int              # device buffer index ([k, n]) or -1
    var bi: Int

    def __init__(out self):
        self.k = 0
        self.n = 0
        self.w = List[Float32]()
        self.bias = List[Float32]()
        self.has_bias = False
        self.wi = -1
        self.bi = -1


struct Conv(Copyable, Movable):
    var cin: Int
    var cout: Int
    var kh: Int
    var kw: Int
    var stride: Int
    var pt: Int
    var pb: Int
    var pl: Int
    var pr: Int
    var lin: Linear          # GEMM weight [kh * kw * cin, cout], rows tap-major

    def __init__(out self):
        self.cin = 0
        self.cout = 0
        self.kh = 1
        self.kw = 1
        self.stride = 1
        self.pt = 0
        self.pb = 0
        self.pl = 0
        self.pr = 0
        self.lin = Linear()


struct Norm(Copyable, Movable):
    var c: Int
    var groups: Int
    var eps: Float32
    var gamma: List[Float32]
    var beta: List[Float32]
    var gi: Int
    var bi: Int

    def __init__(out self):
        self.c = 0
        self.groups = 1
        self.eps = 1e-5
        self.gamma = List[Float32]()
        self.beta = List[Float32]()
        self.gi = -1
        self.bi = -1


# ----------------------------------------------------------------------------- CPU workers

comptime EW_CHUNK: Int = 1 << 14

comptime EW_ACT: Int = 0
comptime EW_ADD: Int = 1
comptime EW_BCAST: Int = 2
comptime EW_GEGLU: Int = 3
comptime EW_SCALE: Int = 4


struct _EwCtx(Movable):
    var op: Int
    var mode: Int
    var a: Int
    var b: Int
    var o: Int
    var n: Int
    var per: Int
    var c: Int
    var alpha: Float32

    def __init__(out self):
        self.op = 0
        self.mode = 0
        self.a = 0
        self.b = 0
        self.o = 0
        self.n = 0
        self.per = 1
        self.c = 1
        self.alpha = 1.0


@always_inline
def _act_s(x: Float32, mode: Int) -> Float32:
    if mode == ACT_SILU:
        return x / (1.0 + exp(-x))
    if mode == ACT_GELU:
        return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))
    return x / (1.0 + exp(-1.702 * x))


@always_inline
def _act_v(x: V8, mode: Int) -> V8:
    if mode == ACT_SILU:
        return x / (1.0 + exp(-x))
    if mode == ACT_GELU:
        return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))
    return x / (1.0 + exp(-1.702 * x))


def _ew_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_EwCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var i0 = job * EW_CHUNK
    var i1 = min(c[].n, i0 + EW_CHUNK)
    var pa = fptr_at(c[].a)
    var pb = fptr_at(c[].b)
    var po = fptr_at(c[].o)
    var op = c[].op
    if op == EW_ACT:
        var i = i0
        while i + 8 <= i1:
            po.unsafe_offset(i).unsafe_store[alignment=1](_act_v(pa.unsafe_offset(i).unsafe_load[width=8, alignment=1](), c[].mode))
            i += 8
        while i < i1:
            po.unsafe_offset(i).unsafe_store(_act_s(pa.unsafe_offset(i).unsafe_load(), c[].mode))
            i += 1
    elif op == EW_ADD:
        var al = c[].alpha
        var i = i0
        while i + 8 <= i1:
            po.unsafe_offset(i).unsafe_store[alignment=1](pa.unsafe_offset(i).unsafe_load[width=8, alignment=1]()
                                                          + al * pb.unsafe_offset(i).unsafe_load[width=8, alignment=1]())
            i += 8
        while i < i1:
            po.unsafe_offset(i).unsafe_store(pa.unsafe_offset(i).unsafe_load() + al * pb.unsafe_offset(i).unsafe_load())
            i += 1
    elif op == EW_SCALE:
        for i in range(i0, i1):
            po.unsafe_offset(i).unsafe_store(pa.unsafe_offset(i).unsafe_load() * c[].alpha)
    elif op == EW_BCAST:
        var C = c[].c
        for i in range(i0, i1):
            var bi = i // c[].per
            po.unsafe_offset(i).unsafe_store(pa.unsafe_offset(i).unsafe_load() + pb.unsafe_offset(bi * C + i % C).unsafe_load())
    else:  # GEGLU: o[r, j] = a[r, j] * gelu(a[r, I + j]), I = c
        var I = c[].c
        for i in range(i0, i1):
            var r = i // I
            var j = i % I
            var g = pa.unsafe_offset(r * 2 * I + I + j).unsafe_load()
            po.unsafe_offset(i).unsafe_store(pa.unsafe_offset(r * 2 * I + j).unsafe_load() * _act_s(g, ACT_GELU))


struct _GnCtx(Movable):
    var x: Int
    var o: Int
    var g: Int
    var b: Int
    var hw: Int
    var C: Int
    var G: Int
    var eps: Float32
    var silu: Bool

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.g = 0
        self.b = 0
        self.hw = 0
        self.C = 0
        self.G = 1
        self.eps = 1e-5
        self.silu = False


def _gn_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_GnCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var G = c[].G
    var C = c[].C
    var img = job // G
    var grp = job % G
    var cg = C // G
    var hw = c[].hw
    var base = img * hw * C + grp * cg
    var px = fptr_at(c[].x)
    var po = fptr_at(c[].o)
    var s: Float64 = 0
    for p in range(hw):
        var row = px.unsafe_offset(base + p * C)
        var acc: Float32 = 0
        for k in range(cg):
            acc += row.unsafe_offset(k).unsafe_load()
        s += Float64(acc)
    var cnt = Float64(hw * cg)
    var mean = Float32(s / cnt)
    var q: Float64 = 0
    for p in range(hw):
        var row = px.unsafe_offset(base + p * C)
        var acc: Float32 = 0
        for k in range(cg):
            var d = row.unsafe_offset(k).unsafe_load() - mean
            acc += d * d
        q += Float64(acc)
    var inv = Float32(1.0 / sqrt(q / cnt + Float64(c[].eps)))
    var pg = fptr_at(c[].g)
    var pbb = fptr_at(c[].b)
    for p in range(hw):
        for k in range(cg):
            var ch = grp * cg + k
            var off = base + p * C + k
            var y = (px.unsafe_offset(off).unsafe_load() - mean) * inv * pg.unsafe_offset(ch).unsafe_load() \
                + pbb.unsafe_offset(ch).unsafe_load()
            if c[].silu:
                y = y / (1.0 + exp(-y))
            po.unsafe_offset(off).unsafe_store(y)


struct _LnCtx(Movable):
    var x: Int
    var o: Int
    var g: Int
    var b: Int
    var n: Int
    var rows: Int
    var eps: Float32

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.g = 0
        self.b = 0
        self.n = 0
        self.rows = 0
        self.eps = 1e-5


def _ln_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_LnCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var n = c[].n
    var r0 = job * 16
    var r1 = min(c[].rows, r0 + 16)
    var px = fptr_at(c[].x)
    var po = fptr_at(c[].o)
    var pg = fptr_at(c[].g)
    var pb = fptr_at(c[].b)
    for r in range(r0, r1):
        var row = px.unsafe_offset(r * n)
        var s: Float32 = 0
        for k in range(n):
            s += row.unsafe_offset(k).unsafe_load()
        var mean = s / Float32(n)
        var q: Float32 = 0
        for k in range(n):
            var d = row.unsafe_offset(k).unsafe_load() - mean
            q += d * d
        var inv = 1.0 / sqrt(q / Float32(n) + c[].eps)
        for k in range(n):
            po.unsafe_offset(r * n + k).unsafe_store((row.unsafe_offset(k).unsafe_load() - mean) * inv
                                                     * pg.unsafe_offset(k).unsafe_load() + pb.unsafe_offset(k).unsafe_load())


struct _ColCtx(Movable):
    var x: Int
    var col: Int
    var idx: Int       # gather table [m, slots] as Int32
    var cin: Int
    var slots: Int
    var r0: Int
    var rows: Int
    var zero_row: Int

    def __init__(out self):
        self.x = 0
        self.col = 0
        self.idx = 0
        self.cin = 0
        self.slots = 0
        self.r0 = 0
        self.rows = 0
        self.zero_row = 0


def _col_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_ColCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var per = 32
    var a = job * per
    var e = min(c[].rows, a + per)
    var cin = c[].cin
    var slots = c[].slots
    var px = fptr_at(c[].x)
    var pc = fptr_at(c[].col)
    var pi = Pointer[Int32, MutAnyOrigin](unsafe_from_address=c[].idx)
    for r in range(a, e):
        for j in range(slots):
            var src = Int(pi.unsafe_offset((c[].r0 + r) * slots + j).unsafe_load())
            var dst = pc.unsafe_offset((r * slots + j) * cin)
            if src == c[].zero_row:
                for k in range(cin):
                    dst.unsafe_offset(k).unsafe_store(0)
            else:
                var sp = px.unsafe_offset(src * cin)
                var k = 0
                while k + 8 <= cin:
                    dst.unsafe_offset(k).unsafe_store[alignment=1](sp.unsafe_offset(k).unsafe_load[width=8, alignment=1]())
                    k += 8
                while k < cin:
                    dst.unsafe_offset(k).unsafe_store(sp.unsafe_offset(k).unsafe_load())
                    k += 1


struct _CausalCtx(Movable):
    var q: Int
    var k: Int
    var v: Int
    var o: Int
    var L: Int
    var H: Int
    var D: Int
    var scale: Float32

    def __init__(out self):
        self.q = 0
        self.k = 0
        self.v = 0
        self.o = 0
        self.L = 0
        self.H = 0
        self.D = 0
        self.scale = 1.0


def _causal_worker(ctx_addr: Int, job: Int):
    """Job = (image, head, query row) of causal self-attention over [B, L, H, D] rows."""
    var c = Pointer[_CausalCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var L = c[].L
    var H = c[].H
    var D = c[].D
    var l = job % L
    var bh = job // L
    var h = bh % H
    var b = bh // H
    var pq = fptr_at(c[].q)
    var pk = fptr_at(c[].k)
    var pv = fptr_at(c[].v)
    var po = fptr_at(c[].o)
    var sc = List[Float32](length=l + 1, fill=0.0)
    var mx: Float32 = -3.0e38
    var qb = ((b * L + l) * H + h) * D
    for s in range(l + 1):
        var kb = ((b * L + s) * H + h) * D
        var acc: Float32 = 0
        for d in range(D):
            acc += pq.unsafe_offset(qb + d).unsafe_load() * pk.unsafe_offset(kb + d).unsafe_load()
        sc[s] = acc * c[].scale
        mx = max(mx, sc[s])
    var tot: Float32 = 0
    for s in range(l + 1):
        sc[s] = exp(sc[s] - mx)
        tot += sc[s]
    for d in range(D):
        var acc: Float32 = 0
        for s in range(l + 1):
            acc += sc[s] * pv.unsafe_offset(((b * L + s) * H + h) * D + d).unsafe_load()
        po.unsafe_offset(qb + d).unsafe_store(acc / tot)


# ----------------------------------------------------------------------------- runtime

def conv_out_size(n: Int, up: Int, p0: Int, p1: Int, k: Int, stride: Int) -> Int:
    return (n * up + p0 + p1 - k) // stride + 1


def conv_gather_table(B: Int, H: Int, W: Int, cv: Conv, up: Int, mode: Int) -> List[Int32]:
    """Row of the [B * H * W (+ zero row), cin] input feeding each (output pixel, tap) of `cv`
    applied to the input upsampled `up` times (nearest); B * H * W means "zero padding"."""
    var Hl = H * up
    var Wl = W * up
    var Ho = conv_out_size(H, up, cv.pt, cv.pb, cv.kh, cv.stride)
    var Wo = conv_out_size(W, up, cv.pl, cv.pr, cv.kw, cv.stride)
    var slots = cv.kh * cv.kw
    var zero = B * H * W
    var t = List[Int32](length=B * Ho * Wo * slots, fill=0)
    var e = 0
    for b in range(B):
        for oy in range(Ho):
            for ox in range(Wo):
                for ky in range(cv.kh):
                    for kx in range(cv.kw):
                        var iy = oy * cv.stride - cv.pt + ky
                        var ix = ox * cv.stride - cv.pl + kx
                        var src = zero
                        if mode == PAD_WRAP:
                            iy = ((iy % Hl) + Hl) % Hl
                            ix = ((ix % Wl) + Wl) % Wl
                            src = (b * H + iy // up) * W + ix // up
                        elif iy >= 0 and iy < Hl and ix >= 0 and ix < Wl:
                            src = (b * H + iy // up) * W + ix // up
                        t[e] = Int32(src)
                        e += 1
    return t^


struct Runtime(Movable):
    var gpu: Optional[GpuMatmul]
    var pool: JobPool
    var next_key: Int
    var tables: Dict[String, Int]
    var host_tables: Dict[String, List[Int32]]
    var weight_bytes: Int

    def __init__(out self, use_gpu: Bool) raises:
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())
        self.gpu = None
        if use_gpu:
            self.gpu = GpuMatmul()
        self.next_key = 1 << 44
        self.tables = Dict[String, Int]()
        self.host_tables = Dict[String, List[Int32]]()
        self.weight_bytes = 0

    def __deinit__(deinit self):
        self.pool.stop()

    def on_gpu(self) -> Bool:
        return Bool(self.gpu)

    def device_name(self) raises -> String:
        if self.gpu:
            return self.gpu.value().name()
        return String("CPU")

    def sync(mut self) raises:
        if self.gpu:
            self.gpu.value().sync()

    # ------------------------------------------------------------------ values

    def alloc(mut self, b: Int, h: Int, w: Int, c: Int) raises -> Act:
        var a = Act()
        a.b = b
        a.h = h
        a.w = w
        a.c = c
        var n = b * h * w * c
        if self.gpu:
            a.slot = self.gpu.value().alloc(n)
        else:
            a.host = List[Float32](length=n, fill=0.0)
        return a^

    def upload(mut self, b: Int, h: Int, w: Int, c: Int, data: List[Float32]) raises -> Act:
        var n = b * h * w * c
        if len(data) != n:
            raise Error("diffusion: upload of " + String(len(data)) + " values into " + String(n))
        var a = Act()
        a.b = b
        a.h = h
        a.w = w
        a.c = c
        if self.gpu:
            a.slot = self.gpu.value().alloc(n)
            self.gpu.value().upload(a.slot, Int(data.unsafe_ptr()), n)
        else:
            a.host = data.copy()
        return a^

    def download(mut self, a: Act) raises -> List[Float32]:
        if a.slot < 0:
            return a.host.copy()
        var out = List[Float32](length=a.numel(), fill=0.0)
        self.gpu.value().download(a.slot, Int(out.unsafe_ptr()), a.numel())
        return out^

    def free(mut self, var a: Act):
        if a.slot >= 0 and self.gpu:
            self.gpu.value().release(a.slot)
        a.slot = -1
        a.host = List[Float32]()

    def ptr(self, a: Act) -> FP:
        """Device pointer of a GPU value."""
        return self.gpu.value().ptr(a.slot)

    def addr(self, a: Act) -> Int:
        """Host address of a CPU value."""
        return Int(a.host.unsafe_ptr())

    def reshape(self, var a: Act, b: Int, h: Int, w: Int, c: Int) raises -> Act:
        if b * h * w * c != a.numel():
            raise Error("diffusion: bad reshape")
        a.b = b
        a.h = h
        a.w = w
        a.c = c
        return a^

    def copy(mut self, a: Act) raises -> Act:
        var o = self.alloc(a.b, a.h, a.w, a.c)
        if self.gpu:
            self.gpu.value().copy_dev(self.ptr(a), self.ptr(o), 1, a.numel(), 0, 0, 0, 0)
        else:
            o.host = a.host.copy()
        return o^

    # ------------------------------------------------------------------ weights

    def _vec(mut self, var data: List[Float32]) raises -> Int:
        var key = self.next_key
        self.next_key += 1
        self.weight_bytes += len(data) * 4
        var idx = self.gpu.value().weight(key, Int(data.unsafe_ptr()), 1, len(data), False)
        _ = data^   # keep the host copy alive until the upload has read it (ASAP destruction)
        return idx

    def _mat(mut self, src: Int, rows: Int, cols: Int, transpose: Bool) raises -> Int:
        var key = self.next_key
        self.next_key += 1
        self.weight_bytes += rows * cols * 4
        return self.gpu.value().weight(key, src, rows, cols, transpose)

    def linear_from(mut self, w: Tensor, b: Tensor, has_bias: Bool) raises -> Linear:
        """Linear from a torch weight [n, k] (any trailing 1 x 1 kernel dims) and bias [n]."""
        var L = Linear()
        L.n = w.shape[0]
        L.k = w.numel() // L.n
        L.has_bias = has_bias
        if self.gpu:
            L.wi = self._mat(w.fptr(), L.n, L.k, True)
            if has_bias:
                L.bi = self._vec(b.owned_floats())
        else:
            L.w = List[Float32](length=L.k * L.n, fill=0.0)
            var p = fptr_at(w.fptr())
            for r in range(L.n):
                for c in range(L.k):
                    L.w[c * L.n + r] = p.unsafe_offset(r * L.k + c).unsafe_load[alignment=1]()
            if has_bias:
                L.bias = b.owned_floats()
            self.weight_bytes += L.k * L.n * 4
        return L^

    def linear(mut self, st: SafeTensors, name: String, has_bias: Bool = True) raises -> Linear:
        var w = st.weight(name + ".weight")
        var b = Tensor()
        if has_bias:
            b = st.weight(name + ".bias")
        return self.linear_from(w, b, has_bias)

    def conv_from(mut self, w: Tensor, b: Tensor, has_bias: Bool, stride: Int, pt: Int, pb: Int, pl: Int,
                  pr: Int) raises -> Conv:
        """Conv2d from a torch weight [cout, cin, kh, kw]; the GEMM matrix rows are (tap, cin)."""
        var cv = Conv()
        cv.cout = w.shape[0]
        cv.cin = w.shape[1]
        cv.kh = w.shape[2]
        cv.kw = w.shape[3]
        cv.stride = stride
        cv.pt = pt
        cv.pb = pb
        cv.pl = pl
        cv.pr = pr
        var taps = cv.kh * cv.kw
        var K = taps * cv.cin
        var m = List[Float32](length=K * cv.cout, fill=0.0)
        var p = fptr_at(w.fptr())
        for co in range(cv.cout):
            for ci in range(cv.cin):
                for t in range(taps):
                    m[(t * cv.cin + ci) * cv.cout + co] = p.unsafe_offset((co * cv.cin + ci) * taps + t).unsafe_load[alignment=1]()
        cv.lin.k = K
        cv.lin.n = cv.cout
        cv.lin.has_bias = has_bias
        if self.gpu:
            cv.lin.wi = self._mat(Int(m.unsafe_ptr()), K, cv.cout, False)
            _ = m^
            if has_bias:
                cv.lin.bi = self._vec(b.owned_floats())
        else:
            cv.lin.w = m^
            if has_bias:
                cv.lin.bias = b.owned_floats()
            self.weight_bytes += K * cv.cout * 4
        return cv^

    def conv(mut self, st: SafeTensors, name: String, stride: Int = 1, pad: Int = -1, has_bias: Bool = True) raises -> Conv:
        var w = st.weight(name + ".weight")
        var b = Tensor()
        if has_bias:
            b = st.weight(name + ".bias")
        var p = pad
        if p < 0:
            p = w.shape[2] // 2
        return self.conv_from(w, b, has_bias, stride, p, p, p, p)

    def norm_from(mut self, var gamma: List[Float32], var beta: List[Float32], groups: Int, eps: Float32) raises -> Norm:
        var nm = Norm()
        nm.c = len(gamma)
        nm.groups = groups
        nm.eps = eps
        if self.gpu:
            nm.gi = self._vec(gamma^)
            nm.bi = self._vec(beta^)
        else:
            nm.gamma = gamma^
            nm.beta = beta^
        return nm^

    def norm(mut self, st: SafeTensors, name: String, groups: Int = 1, eps: Float32 = 1e-5) raises -> Norm:
        return self.norm_from(st.weight(name + ".weight").owned_floats(), st.weight(name + ".bias").owned_floats(),
                              groups, eps)

    # ------------------------------------------------------------------ element-wise

    def _ew(mut self, op: Int, mode: Int, a: Int, b: Int, o: Int, n: Int, per: Int = 1, c: Int = 1,
            alpha: Float32 = 1.0):
        var ctx = _EwCtx()
        ctx.op = op
        ctx.mode = mode
        ctx.a = a
        ctx.b = b
        ctx.o = o
        ctx.n = n
        ctx.per = per
        ctx.c = c
        ctx.alpha = alpha
        var jobs = (n + EW_CHUNK - 1) // EW_CHUNK
        if jobs <= 1 or not self.pool.ready():
            for j in range(jobs):
                _ew_worker(Int(Pointer(to=ctx)), j)
            return
        run_parallel_ctx(self.pool, jobs, ctx, _ew_worker)

    def activate(mut self, mut x: Act, mode: Int) raises:
        """In place: SiLU / GELU / quick-GELU."""
        var n = x.numel()
        if self.gpu:
            var p = self.ptr(x)
            self.gpu.value().ctx.enqueue_function[_dk_act_kernel](p, p, Int32(n), Int32(mode),
                                                                 grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
        else:
            self._ew(EW_ACT, mode, self.addr(x), 0, self.addr(x), n)

    def add_(mut self, mut x: Act, y: Act, alpha: Float32 = 1.0) raises:
        """In place: x += alpha * y."""
        var n = x.numel()
        if self.gpu:
            var p = self.ptr(x)
            self.gpu.value().ctx.enqueue_function[_dk_add_kernel](p, self.ptr(y), p, Int32(n), alpha,
                                                                 grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
        else:
            self._ew(EW_ADD, 0, self.addr(x), self.addr(y), self.addr(x), n, 1, 1, alpha)

    def scale_(mut self, mut x: Act, s: Float32) raises:
        var n = x.numel()
        if self.gpu:
            var p = self.ptr(x)
            self.gpu.value().ctx.enqueue_function[_dk_scale_kernel](p, p, Int32(n), s,
                                                                   grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
        else:
            self._ew(EW_SCALE, 0, self.addr(x), 0, self.addr(x), n, 1, 1, s)

    def bcast_add_(mut self, mut x: Act, t: Act) raises:
        """In place: x[b, p, :] += t[b, :] (t has b rows of c)."""
        var n = x.numel()
        var per = x.h * x.w * x.c
        if self.gpu:
            var p = self.ptr(x)
            self.gpu.value().ctx.enqueue_function[_dk_bcast_add_kernel](
                p, self.ptr(t), Int32(n), Int32(per), Int32(x.c), grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
        else:
            self._ew(EW_BCAST, 0, self.addr(x), self.addr(t), self.addr(x), n, per, x.c)

    def geglu(mut self, h: Act) raises -> Act:
        """[rows, 2I] -> [rows, I]: first half * gelu(second half)."""
        var I = h.c // 2
        var o = self.alloc(h.b, h.h, h.w, I)
        var n = o.numel()
        if self.gpu:
            self.gpu.value().ctx.enqueue_function[_dk_geglu_kernel](
                self.ptr(h), self.ptr(o), Int32(h.rows()), Int32(I), grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
        else:
            self._ew(EW_GEGLU, 0, self.addr(h), 0, self.addr(o), n, 1, I)
        return o^

    def concat_c(mut self, a: Act, b: Act) raises -> Act:
        """Channel concatenation [.., ca] ++ [.., cb]."""
        var o = self.alloc(a.b, a.h, a.w, a.c + b.c)
        var rows = a.rows()
        if self.gpu:
            self.gpu.value().copy_dev(self.ptr(a), self.ptr(o), rows, a.c, a.c, 0, a.c + b.c, 0)
            self.gpu.value().copy_dev(self.ptr(b), self.ptr(o), rows, b.c, b.c, 0, a.c + b.c, a.c)
        else:
            var C = a.c + b.c
            for r in range(rows):
                for k in range(a.c):
                    o.host[r * C + k] = a.host[r * a.c + k]
                for k in range(b.c):
                    o.host[r * C + a.c + k] = b.host[r * b.c + k]
        return o^

    def slice_c(mut self, a: Act, c0: Int, c1: Int) raises -> Act:
        var o = self.alloc(a.b, a.h, a.w, c1 - c0)
        var rows = a.rows()
        if self.gpu:
            self.gpu.value().copy_dev(self.ptr(a), self.ptr(o), rows, c1 - c0, a.c, c0, c1 - c0, 0)
        else:
            for r in range(rows):
                for k in range(c0, c1):
                    o.host[r * (c1 - c0) + k - c0] = a.host[r * a.c + k]
        return o^

    # ------------------------------------------------------------------ norms

    def group_norm(mut self, x: Act, nm: Norm, silu: Bool = False) raises -> Act:
        var o = self.alloc(x.b, x.h, x.w, x.c)
        var hw = x.h * x.w
        if self.gpu:
            self.gpu.value().ctx.enqueue_function[_dk_group_norm_kernel](
                self.ptr(x), self.gpu.value().wptr(nm.gi), self.gpu.value().wptr(nm.bi), self.ptr(o), Int32(hw),
                Int32(x.c), Int32(nm.groups), nm.eps, Int32(1 if silu else 0), grid_dim=x.b * nm.groups,
                block_dim=DK_RED)
        else:
            var ctx = _GnCtx()
            ctx.x = self.addr(x)
            ctx.o = self.addr(o)
            ctx.g = Int(nm.gamma.unsafe_ptr())
            ctx.b = Int(nm.beta.unsafe_ptr())
            ctx.hw = hw
            ctx.C = x.c
            ctx.G = nm.groups
            ctx.eps = nm.eps
            ctx.silu = silu
            run_parallel_ctx(self.pool, x.b * nm.groups, ctx, _gn_worker)
        return o^

    def layer_norm(mut self, x: Act, nm: Norm) raises -> Act:
        var o = self.alloc(x.b, x.h, x.w, x.c)
        var rows = x.rows()
        if self.gpu:
            self.gpu.value().layer_norm_dev(self.ptr(x), self.gpu.value().wptr(nm.gi), self.gpu.value().wptr(nm.bi),
                                            self.ptr(o), rows, x.c, nm.eps, True)
        else:
            var ctx = _LnCtx()
            ctx.x = self.addr(x)
            ctx.o = self.addr(o)
            ctx.g = Int(nm.gamma.unsafe_ptr())
            ctx.b = Int(nm.beta.unsafe_ptr())
            ctx.n = x.c
            ctx.rows = rows
            ctx.eps = nm.eps
            run_parallel_ctx(self.pool, (rows + 15) // 16, ctx, _ln_worker)
        return o^

    # ------------------------------------------------------------------ dense

    def _gemm(mut self, a: Int, L: Linear, o: Int, m: Int) raises:
        """CPU: o[m, n] = a[m, k] @ W (+ b)."""
        var bias = 0
        if L.has_bias:
            bias = Int(L.bias.unsafe_ptr())
        matmul_2d(self.pool, a, Int(L.w.unsafe_ptr()), o, bias, m, L.k, L.n, False)

    def linear(mut self, x: Act, L: Linear) raises -> Act:
        if x.c != L.k:
            raise Error("diffusion: linear expects " + String(L.k) + " inputs, got " + String(x.c))
        var o = self.alloc(x.b, x.h, x.w, L.n)
        var m = x.rows()
        if self.gpu:
            var pb = self.ptr(o)
            if L.has_bias:
                pb = self.gpu.value().wptr(L.bi)
            self.gpu.value()._launch_mm(self.ptr(x), self.gpu.value().wptr(L.wi), self.ptr(o), pb, L.has_bias, m,
                                        L.k, L.n)
        else:
            self._gemm(self.addr(x), L, self.addr(o), m)
        return o^

    def _table_key(self, B: Int, H: Int, W: Int, cv: Conv, up: Int, mode: Int) -> String:
        return (String(B) + "," + String(H) + "," + String(W) + "," + String(cv.kh) + "," + String(cv.kw) + ","
                + String(cv.stride) + "," + String(cv.pt) + "," + String(cv.pb) + "," + String(cv.pl) + ","
                + String(cv.pr) + "," + String(up) + "," + String(mode))

    def conv2d(mut self, x: Act, cv: Conv, mode: Int = PAD_ZERO, up: Int = 1) raises -> Act:
        """Conv2d (cross-correlation) of `x` (optionally nearest-upsampled by `up`) with zero or
        wrap-around padding."""
        if x.c != cv.cin:
            raise Error("diffusion: conv expects " + String(cv.cin) + " channels, got " + String(x.c))
        var Ho = conv_out_size(x.h, up, cv.pt, cv.pb, cv.kh, cv.stride)
        var Wo = conv_out_size(x.w, up, cv.pl, cv.pr, cv.kw, cv.stride)
        var plain = cv.kh == 1 and cv.kw == 1 and cv.stride == 1 and up == 1 and cv.pt == 0 and cv.pl == 0 \
            and cv.pb == 0 and cv.pr == 0
        if plain:
            return self.linear(x, cv.lin)
        var o = self.alloc(x.b, Ho, Wo, cv.cout)
        var m = x.b * Ho * Wo
        var slots = cv.kh * cv.kw
        var key = self._table_key(x.b, x.h, x.w, cv, up, mode)
        var rows_in = x.rows()
        if self.gpu:
            var found = self.tables.get(key)
            var ti: Int
            if found:
                ti = found.value()
            else:
                var t = conv_gather_table(x.b, x.h, x.w, cv, up, mode)
                var tf = List[Float32](length=len(t), fill=0.0)
                for i in range(len(t)):
                    tf[i] = Float32(t[i])
                ti = self._mat(Int(tf.unsafe_ptr()), 1, len(tf), False)
                self.weight_bytes -= len(tf) * 4
                _ = tf^
                self.tables[key] = ti
            # input + one zero row for the padding taps
            var xs = self.gpu.value().alloc((rows_in + 1) * x.c)
            var px = self.gpu.value().ptr(xs)
            self.gpu.value().copy_dev(self.ptr(x), px, 1, rows_in * x.c, 0, 0, 0, 0)
            self.gpu.value().ctx.enqueue_function[_dk_fill_kernel](px.unsafe_offset(rows_in * x.c), Int32(x.c),
                                                                  Float32(0), grid_dim=(x.c + DK_EW - 1) // DK_EW,
                                                                  block_dim=DK_EW)
            var tn = (cv.cout + TBN - 1) // TBN
            var tm = (m + TBM - 1) // TBM
            var pb = self.ptr(o)
            if cv.lin.has_bias:
                pb = self.gpu.value().wptr(cv.lin.bi)
            self.gpu.value().ctx.enqueue_function[_mm_tiled[True]](
                px, self.gpu.value().wptr(cv.lin.wi), self.ptr(o), pb, Int32(m), Int32(slots * x.c), Int32(cv.cout),
                Int32(1 if cv.lin.has_bias else 0), Int32(tn), self.gpu.value().wptr(ti), Int32(x.c), Int32(slots),
                grid_dim=tn * tm, block_dim=256)
            self.gpu.value().products += 1
            self.gpu.value().release(xs)
            return o^
        if key not in self.host_tables:
            self.host_tables[key] = conv_gather_table(x.b, x.h, x.w, cv, up, mode)
        var K = slots * x.c
        var chunk = max(64, min(m, (1 << 22) // max(K, 1)))
        var col = List[Float32](length=chunk * K, fill=0.0)
        var ctx = _ColCtx()
        ctx.x = self.addr(x)
        ctx.col = Int(col.unsafe_ptr())
        ctx.idx = Int(self.host_tables[key].unsafe_ptr())
        ctx.cin = x.c
        ctx.slots = slots
        ctx.zero_row = rows_in
        var r0 = 0
        while r0 < m:
            var rows = min(chunk, m - r0)
            ctx.r0 = r0
            ctx.rows = rows
            run_parallel_ctx(self.pool, (rows + 31) // 32, ctx, _col_worker)
            self._gemm(Int(col.unsafe_ptr()), cv.lin, self.addr(o) + r0 * cv.cout * 4, rows)
            r0 += rows
        _ = col^
        return o^

    # ------------------------------------------------------------------ attention

    def attention(mut self, q: Act, k: Act, v: Act, heads: Int, causal: Bool = False) raises -> Act:
        """Multi-head attention over rows: q [b, L, H*D], k / v [b, S, H*D] -> [b, L, H*D]
        (q / k / v may be any [b, h, w, c] values; L = h * w)."""
        var B = q.b
        var L = q.h * q.w
        var S = k.h * k.w
        var C = q.c
        var D = C // heads
        var scale = Float32(1.0 / sqrt(Float64(D)))
        var o = self.alloc(q.b, q.h, q.w, C)
        if self.gpu:
            var qh = self.gpu.value().alloc(B * L * C)
            var kt = self.gpu.value().alloc(B * S * C)
            var vh = self.gpu.value().alloc(B * S * C)
            var oh = self.gpu.value().alloc(B * L * C)
            var sc = self.gpu.value().alloc(L * S)
            var pq = self.gpu.value().ptr(qh)
            var pk = self.gpu.value().ptr(kt)
            var pv = self.gpu.value().ptr(vh)
            var po = self.gpu.value().ptr(oh)
            var ps = self.gpu.value().ptr(sc)
            var nq = B * L * C
            var nk = B * S * C
            self.gpu.value().ctx.enqueue_function[_dk_heads_kernel](
                self.ptr(q), pq, Int32(B), Int32(L), Int32(heads), Int32(D), Int32(C), Int32(0), Int32(0),
                grid_dim=(nq + DK_EW - 1) // DK_EW, block_dim=DK_EW)
            self.gpu.value().ctx.enqueue_function[_dk_heads_kernel](
                self.ptr(k), pk, Int32(B), Int32(S), Int32(heads), Int32(D), Int32(C), Int32(0), Int32(1),
                grid_dim=(nk + DK_EW - 1) // DK_EW, block_dim=DK_EW)
            self.gpu.value().ctx.enqueue_function[_dk_heads_kernel](
                self.ptr(v), pv, Int32(B), Int32(S), Int32(heads), Int32(D), Int32(C), Int32(0), Int32(0),
                grid_dim=(nk + DK_EW - 1) // DK_EW, block_dim=DK_EW)
            for bh in range(B * heads):
                self.gpu.value()._launch_mm(pq.unsafe_offset(bh * L * D), pk.unsafe_offset(bh * D * S), ps, ps, False,
                                            L, D, S)
                self.gpu.value().ctx.enqueue_function[_dk_softmax_kernel](ps, Int32(S), scale,
                                                                         Int32(1 if causal else 0), Int32(L),
                                                                         grid_dim=L, block_dim=DK_RED)
                self.gpu.value()._launch_mm(ps, pv.unsafe_offset(bh * S * D), po.unsafe_offset(bh * L * D), ps, False,
                                            L, S, D)
            self.gpu.value().ctx.enqueue_function[_dk_merge_kernel](
                po, self.ptr(o), Int32(B), Int32(L), Int32(heads), Int32(D), grid_dim=(nq + DK_EW - 1) // DK_EW,
                block_dim=DK_EW)
            self.gpu.value().release(qh)
            self.gpu.value().release(kt)
            self.gpu.value().release(vh)
            self.gpu.value().release(oh)
            self.gpu.value().release(sc)
            return o^
        if causal:
            var ctx = _CausalCtx()
            ctx.q = self.addr(q)
            ctx.k = self.addr(k)
            ctx.v = self.addr(v)
            ctx.o = self.addr(o)
            ctx.L = L
            ctx.H = heads
            ctx.D = D
            ctx.scale = scale
            run_parallel_ctx(self.pool, B * heads * L, ctx, _causal_worker)
            return o^
        var tq = Tensor.borrowed([B, L, heads, D], self.addr(q))
        var tk = Tensor.borrowed([B, S, heads, D], self.addr(k))
        var tv = Tensor.borrowed([B, S, heads, D], self.addr(v))
        var r = cpu_attention(self.pool, tq, tk, tv, scale)
        if r.rank() == 0:
            raise Error("diffusion: CPU attention does not apply")
        o.host = r.owned_floats()
        return o^
