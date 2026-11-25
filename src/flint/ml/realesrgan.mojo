"""Real-ESRGAN image super-resolution in Mojo (`flint.ml.realesrgan`), CPU or MAX GPU.

Two network families from https://github.com/xinntao/Real-ESRGAN (code BSD-3-Clause):

  * `RRDBNet` (ESRGAN's residual-in-residual dense blocks, from basicsr): `RealESRGAN_x4plus`
    (23 blocks) and `RealESRGAN_x4plus_anime_6B` (6 blocks). conv_first -> blocks -> conv_body
    (+ skip) -> two nearest 2x upsample + conv + LeakyReLU(0.2) -> conv_hr -> conv_last.
  * `SRVGGNetCompact`: `realesr-general-x4v3` (32 convs), its denoising twin
    `realesr-general-wdn-x4v3` and `realesr-animevideov3` (16 convs). A plain 3x3 conv + PReLU
    chain, then PixelShuffle(4) plus a nearest-upsampled copy of the input.

Layers run on `flint.ml.diffusion.nn.Runtime` (channels-last implicit-GEMM convolutions; the
nearest 2x upsample is folded into the conv's gather table). This module adds the activations
(LeakyReLU, per-channel PReLU) and a fused PixelShuffle + nearest-base add, on CPU (JobPool) and
GPU (MAX kernels). `RealEsrgan.upscale_rgb` follows Real-ESRGAN's `RealESRGANer`: optional
tiled inference with the same tile / tile_pad geometry as `tile_process`, then
clamp to [0, 1] and round(x * 255) (half to even, like numpy). `upscale_image` runs a NativeImage
(0xAARRGGBB) and upsamples a non-opaque alpha channel through the network as grey, like the
reference's `alpha_upsampler="realesrgan"`.

Weights are read from safetensors written by `tools/upscale/convert.py` (original parameter
names). Their licence is not stated upstream, so Flint never ships them: see licenses/README.md.
"""

from std.ffi import external_call
from std.math import floor
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from flint.core.ecs import run_parallel_ctx
from flint.assets.image import NativeImage
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.safetensors import SafeTensors
from flint.ml.gpu import FP
from flint.ml.diffusion.nn import Runtime, Act, Conv, conv_gather_table, conv_out_size, PAD_ZERO
from flint.ml.diffusion.kernels import DK_EW, _dk_fill_kernel

comptime ARCH_RRDB: Int = 0
comptime ARCH_COMPACT: Int = 1

comptime _OP_LRELU: Int = 0
comptime _OP_PRELU: Int = 1
comptime _OP_PS_ADD: Int = 2
comptime _CHUNK: Int = 1 << 14


# ----------------------------------------------------------------------------- GPU kernels

def _re_lrelu_kernel(x: FP, n: Int32, slope: Float32):
    var i = Int(global_idx.x)
    if i < Int(n):
        var v = x.unsafe_offset(i).unsafe_load()
        if v < 0:
            x.unsafe_offset(i).unsafe_store(v * slope)


def _re_prelu_kernel(x: FP, a: FP, n: Int32, C: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        var v = x.unsafe_offset(i).unsafe_load()
        if v < 0:
            x.unsafe_offset(i).unsafe_store(v * a.unsafe_offset(i % Int(C)).unsafe_load())


def _re_ps_add_kernel(src: FP, base: FP, o: FP, H: Int32, W: Int32, C: Int32, r: Int32):
    """o[y*r+i, x*r+j, c] = src[y, x, c*r*r + i*r + j] + base[y, x, c] (PixelShuffle + nearest skip)."""
    var t = Int(global_idx.x)
    var R = Int(r)
    var Wo = Int(W) * R
    var CC = Int(C)
    var total = Int(H) * R * Wo * CC
    if t >= total:
        return
    var c = t % CC
    var p = t // CC
    var ox = p % Wo
    var oy = p // Wo
    var y = oy // R
    var x = ox // R
    var i = oy - y * R
    var j = ox - x * R
    var pix = y * Int(W) + x
    o.unsafe_offset(t).unsafe_store(src.unsafe_offset(pix * CC * R * R + c * R * R + i * R + j).unsafe_load()
                                    + base.unsafe_offset(pix * CC + c).unsafe_load())


comptime _CK: Int = 16                      # K step of the conv kernel
comptime KRM: Int = 8                       # output rows per thread
comptime _ACT_NONE: Int = 0
comptime _ACT_LRELU: Int = 1
comptime _ACT_PRELU: Int = 2
comptime _V8 = SIMD[DType.float32, 8]


def _re_conv_kernel[BN: Int](x: FP, w: FP, bias: FP, o: FP, idx: FP, m: Int32, cin: Int32, n: Int32,
                                      slots: Int32, act: Int32, slopes: FP):
    """k x k convolution as an implicit GEMM tuned for Real-ESRGAN's narrow layers: a
    (RM * 2048 / BN) x BN output tile per 256-thread block, RM x 8 outputs per thread, K in steps
    of 16 that never straddle a tap (cin % 16 == 0), so one gather-table read serves 16 channels.
    x is [rows + 1, cin] with a trailing zero row for the padding taps; idx [m, slots] holds input
    rows as floats (`conv_gather_table`); the epilogue adds the bias and applies LeakyReLU(0.2)
    or PReLU."""
    comptime RM = KRM
    comptime TM = RM * 2048 // BN
    comptime GX = BN // 8
    var As = stack_allocation[_CK * TM, Float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[_CK * BN, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var tx = t % GX
    var ty = t // GX
    var nn = Int(n)
    var tiles_n = (nn + BN - 1) // BN
    var blk = Int(block_idx.x)
    var bn = blk % tiles_n
    var bm = blk // tiles_n
    var row0 = bm * TM
    var col0 = bn * BN
    var mm = Int(m)
    var ci = Int(cin)
    var sl = Int(slots)
    var kn = ci * sl
    var acc0 = _V8(0)
    var acc1 = _V8(0)
    var acc2 = _V8(0)
    var acc3 = _V8(0)
    var acc4 = _V8(0)
    var acc5 = _V8(0)
    var acc6 = _V8(0)
    var acc7 = _V8(0)
    var k0 = 0
    while k0 < kn:
        var tap = k0 // ci
        var ch0 = k0 - tap * ci
        comptime for q in range(TM * _CK // 256):
            var e = t + q * 256
            var r = e // _CK
            var kk = e % _CK
            var gr = row0 + r
            var v: Float32 = 0
            if gr < mm:
                var row = Int(idx.unsafe_offset(gr * sl + tap).unsafe_load())
                v = x.unsafe_offset(row * ci + ch0 + kk).unsafe_load()
            As[unsafe_offset=kk * TM + r] = v
        comptime for q in range(_CK * BN // 256):
            var e = t + q * 256
            var kk = e // BN
            var cc = e % BN
            var gc = col0 + cc
            var v: Float32 = 0
            if gc < nn:
                v = w.unsafe_offset((k0 + kk) * nn + gc).unsafe_load()
            Bs[unsafe_offset=kk * BN + cc] = v
        barrier()
        comptime for kk in range(_CK):
            var bv = _V8(0)
            comptime for j in range(8):
                bv[j] = Bs[unsafe_offset=kk * BN + tx * 8 + j]
            acc0 += As[unsafe_offset=kk * TM + ty * RM + 0] * bv
            acc1 += As[unsafe_offset=kk * TM + ty * RM + 1] * bv
            acc2 += As[unsafe_offset=kk * TM + ty * RM + 2] * bv
            acc3 += As[unsafe_offset=kk * TM + ty * RM + 3] * bv
            acc4 += As[unsafe_offset=kk * TM + ty * RM + 4] * bv
            acc5 += As[unsafe_offset=kk * TM + ty * RM + 5] * bv
            acc6 += As[unsafe_offset=kk * TM + ty * RM + 6] * bv
            acc7 += As[unsafe_offset=kk * TM + ty * RM + 7] * bv
        barrier()
        k0 += _CK
    comptime for i in range(RM):
        var gr = row0 + ty * RM + i
        if gr < mm:
            var accv: _V8
            comptime if i == 0:
                accv = acc0
            elif i == 1:
                accv = acc1
            elif i == 2:
                accv = acc2
            elif i == 3:
                accv = acc3
            elif i == 4:
                accv = acc4
            elif i == 5:
                accv = acc5
            elif i == 6:
                accv = acc6
            else:
                accv = acc7
            comptime for j in range(8):
                var gc = col0 + tx * 8 + j
                if gc < nn:
                    var y = accv[j] + bias.unsafe_offset(gc).unsafe_load()
                    if y < 0:
                        if act == Int32(_ACT_LRELU):
                            y = y * 0.2
                        elif act == Int32(_ACT_PRELU):
                            y = y * slopes.unsafe_offset(gc).unsafe_load()
                    o.unsafe_offset(gr * nn + gc).unsafe_store(y)


# ----------------------------------------------------------------------------- CPU workers

struct _ReCtx(Movable):
    var op: Int
    var x: Int
    var a: Int
    var b: Int
    var o: Int
    var n: Int
    var C: Int
    var H: Int
    var W: Int
    var r: Int
    var slope: Float32

    def __init__(out self):
        self.op = 0
        self.x = 0
        self.a = 0
        self.b = 0
        self.o = 0
        self.n = 0
        self.C = 1
        self.H = 0
        self.W = 0
        self.r = 1
        self.slope = 0.2


def _re_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_ReCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var i0 = job * _CHUNK
    var i1 = min(c[].n, i0 + _CHUNK)
    if c[].op == _OP_LRELU:
        var px = fptr_at(c[].x)
        var s = c[].slope
        for i in range(i0, i1):
            var v = px.unsafe_offset(i).unsafe_load()
            if v < 0:
                px.unsafe_offset(i).unsafe_store(v * s)
    elif c[].op == _OP_PRELU:
        var px = fptr_at(c[].x)
        var pa = fptr_at(c[].a)
        var C = c[].C
        for i in range(i0, i1):
            var v = px.unsafe_offset(i).unsafe_load()
            if v < 0:
                px.unsafe_offset(i).unsafe_store(v * pa.unsafe_offset(i % C).unsafe_load())
    else:
        var ps = fptr_at(c[].x)
        var pb = fptr_at(c[].b)
        var po = fptr_at(c[].o)
        var R = c[].r
        var W = c[].W
        var Wo = W * R
        var C = c[].C
        for t in range(i0, i1):
            var ch = t % C
            var p = t // C
            var ox = p % Wo
            var oy = p // Wo
            var y = oy // R
            var x = ox // R
            var pix = y * W + x
            po.unsafe_offset(t).unsafe_store(ps.unsafe_offset(pix * C * R * R + ch * R * R + (oy - y * R) * R + ox - x * R).unsafe_load()
                                             + pb.unsafe_offset(pix * C + ch).unsafe_load())


# ----------------------------------------------------------------------------- helpers

def round_half_even(v: Float32) -> Float32:
    """numpy's `round`: halves go to the even neighbour."""
    var f = floor(v)
    var d = v - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    if Int(f) % 2 == 0:
        return f
    return f + 1


def to_u8(v: Float32) -> UInt32:
    """Real-ESRGAN's output conversion: clamp to [0, 1], then round(x * 255)."""
    var c = min(max(v, Float32(0)), Float32(1))
    return UInt32(Int(round_half_even(c * 255.0)))


def tile_count(n: Int, tile: Int) -> Int:
    return (n + tile - 1) // tile


# ----------------------------------------------------------------------------- the network

struct RealEsrgan(Movable):
    var rt: Runtime
    var arch: Int
    var scale: Int
    var num_feat: Int
    var num_block: Int
    var convs: List[Conv]          # RRDB: first, blocks (15 per block), body, up1, up2, hr, last;
                                   # compact: the body convs in order
    var slopes: List[List[Float32]]  # compact PReLU slopes (host)
    var slope_dev: List[Int]         # their device buffers (GPU runtime)
    var name: String

    def __init__(out self, use_gpu: Bool) raises:
        self.rt = Runtime(use_gpu)
        self.arch = ARCH_COMPACT
        self.scale = 4
        self.num_feat = 0
        self.num_block = 0
        self.convs = List[Conv]()
        self.slopes = List[List[Float32]]()
        self.slope_dev = List[Int]()
        self.name = String()

    @staticmethod
    def load(path: String, use_gpu: Bool, blend_path: String = "", strength: Float32 = 1.0) raises -> RealEsrgan:
        """A converted checkpoint. With `blend_path` (same architecture), every parameter is
        `strength * path + (1 - strength) * blend_path`: Real-ESRGAN's deep network
        interpolation, which sets realesr-general-x4v3's denoise strength against
        realesr-general-wdn-x4v3."""
        var net = RealEsrgan(use_gpu)
        net.name = path
        var st = SafeTensors.open(path)
        var blend = SafeTensors()
        var has_blend = blend_path.byte_length() > 0
        if has_blend:
            blend = SafeTensors.open(blend_path)
        if st.has("conv_first.weight"):
            net.arch = ARCH_RRDB
            var cin = st.shape("conv_first.weight")[1]
            if cin != 3:
                raise Error("realesrgan: " + path + " pixel-unshuffles its input (x2 / x1 models); only x4 nets are supported")
            net.num_feat = st.shape("conv_first.weight")[0]
            var nb = 0
            while st.has("body." + String(nb) + ".rdb1.conv1.weight"):
                nb += 1
            net.num_block = nb
            net._add_conv(st, blend, has_blend, strength, "conv_first")
            for b in range(nb):
                for r in range(1, 4):
                    for k in range(1, 6):
                        net._add_conv(st, blend, has_blend, strength,
                                      "body." + String(b) + ".rdb" + String(r) + ".conv" + String(k))
            net._add_conv(st, blend, has_blend, strength, "conv_body")
            net._add_conv(st, blend, has_blend, strength, "conv_up1")
            net._add_conv(st, blend, has_blend, strength, "conv_up2")
            net._add_conv(st, blend, has_blend, strength, "conv_hr")
            net._add_conv(st, blend, has_blend, strength, "conv_last")
            net.scale = 4
        elif st.has("body.0.weight"):
            net.arch = ARCH_COMPACT
            net.num_feat = st.shape("body.0.weight")[0]
            var i = 0
            while st.has("body." + String(i) + ".weight"):
                var nm = "body." + String(i)
                if len(st.shape(nm + ".weight")) == 4:
                    net._add_conv(st, blend, has_blend, strength, nm)
                else:
                    var a = _param(st, blend, has_blend, strength, nm + ".weight").owned_floats()
                    if net.rt.on_gpu():
                        net.slope_dev.append(net.rt._vec(a.copy()))
                    net.slopes.append(a^)
                i += 1
            if len(net.slopes) != len(net.convs) - 1:
                raise Error("realesrgan: " + path + " is not a PReLU SRVGGNetCompact")
            var outc = net.convs[len(net.convs) - 1].cout
            var r = 1
            while r * r * 3 < outc:
                r += 1
            if r * r * 3 != outc:
                raise Error("realesrgan: last conv of " + path + " has " + String(outc) + " channels")
            net.scale = r
        else:
            raise Error("realesrgan: " + path + " is neither an RRDBNet nor an SRVGGNetCompact checkpoint")
        return net^

    def _add_conv(mut self, st: SafeTensors, blend: SafeTensors, has_blend: Bool, strength: Float32, name: String) raises:
        var w = _param(st, blend, has_blend, strength, name + ".weight")
        var b = _param(st, blend, has_blend, strength, name + ".bias")
        var p = w.shape[2] // 2
        self.convs.append(self.rt.conv_from(w, b, True, 1, p, p, p, p))

    def device_name(self) raises -> String:
        return self.rt.device_name()

    def arch_name(self) -> String:
        if self.arch == ARCH_RRDB:
            return "RRDBNet(" + String(self.num_block) + " blocks, " + String(self.num_feat) + " feat)"
        return "SRVGGNetCompact(" + String(len(self.convs) - 2) + " convs, " + String(self.num_feat) + " feat)"

    # ------------------------------------------------------------------ layer helpers

    def _conv(mut self, x: Act, ci: Int, act: Int, up: Int = 1, prelu_layer: Int = -1) raises -> Act:
        """Conv `ci` of x (nearest-upsampled `up` times first) with the activation fused on the GPU."""
        ref cv = self.convs[ci]
        if not self.rt.on_gpu() or x.c % _CK != 0:
            var y = self.rt.conv2d(x, cv, PAD_ZERO, up)
            if act == _ACT_LRELU:
                self._lrelu(y)
            elif act == _ACT_PRELU:
                self._prelu(y, prelu_layer)
            return y^
        var Ho = conv_out_size(x.h, up, cv.pt, cv.pb, cv.kh, cv.stride)
        var Wo = conv_out_size(x.w, up, cv.pl, cv.pr, cv.kw, cv.stride)
        var o = self.rt.alloc(x.b, Ho, Wo, cv.cout)
        var m = x.b * Ho * Wo
        var slots = cv.kh * cv.kw
        var key = self.rt._table_key(x.b, x.h, x.w, cv, up, PAD_ZERO)
        var found = self.rt.tables.get(key)
        var ti: Int
        if found:
            ti = found.value()
        else:
            var tbl = conv_gather_table(x.b, x.h, x.w, cv, up, PAD_ZERO)
            var tf = List[Float32](length=len(tbl), fill=0.0)
            for i in range(len(tbl)):
                tf[i] = Float32(tbl[i])
            ti = self.rt._mat(Int(tf.unsafe_ptr()), 1, len(tf), False)
            self.rt.weight_bytes -= len(tf) * 4
            _ = tf^
            self.rt.tables[key] = ti
        var rows_in = x.rows()
        var px_in = self.rt.ptr(x)
        var po = self.rt.ptr(o)
        ref g = self.rt.gpu.value()
        var xs = g.alloc((rows_in + 1) * x.c)
        var px = g.ptr(xs)
        g.copy_dev(px_in, px, 1, rows_in * x.c, 0, 0, 0, 0)
        g.ctx.enqueue_function[_dk_fill_kernel](px.unsafe_offset(rows_in * x.c), Int32(x.c), Float32(0),
                                                grid_dim=(x.c + DK_EW - 1) // DK_EW, block_dim=DK_EW)
        var pb = g.wptr(cv.lin.bi)
        var slopes = pb
        if act == _ACT_PRELU:
            slopes = g.wptr(self.slope_dev[prelu_layer])
        var pw = g.wptr(cv.lin.wi)
        var pt = g.wptr(ti)
        if cv.cout <= 32:
            var tiles = ((m + KRM * 64 - 1) // (KRM * 64)) * ((cv.cout + 31) // 32)
            g.ctx.enqueue_function[_re_conv_kernel[32]](px, pw, pb, po, pt, Int32(m), Int32(x.c), Int32(cv.cout),
                                                        Int32(slots), Int32(act), slopes, grid_dim=tiles, block_dim=256)
        else:
            var tiles = ((m + KRM * 32 - 1) // (KRM * 32)) * ((cv.cout + 63) // 64)
            g.ctx.enqueue_function[_re_conv_kernel[64]](px, pw, pb, po, pt, Int32(m), Int32(x.c), Int32(cv.cout),
                                                        Int32(slots), Int32(act), slopes, grid_dim=tiles, block_dim=256)
        g.release(xs)
        return o^

    def _lrelu(mut self, mut x: Act) raises:
        var n = x.numel()
        if self.rt.on_gpu():
            self.rt.gpu.value().ctx.enqueue_function[_re_lrelu_kernel](self.rt.ptr(x), Int32(n), Float32(0.2),
                                                                      grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
            return
        var ctx = _ReCtx()
        ctx.op = _OP_LRELU
        ctx.x = self.rt.addr(x)
        ctx.n = n
        run_parallel_ctx(self.rt.pool, (n + _CHUNK - 1) // _CHUNK, ctx, _re_worker)

    def _prelu(mut self, mut x: Act, layer: Int) raises:
        var n = x.numel()
        if self.rt.on_gpu():
            self.rt.gpu.value().ctx.enqueue_function[_re_prelu_kernel](
                self.rt.ptr(x), self.rt.gpu.value().wptr(self.slope_dev[layer]), Int32(n), Int32(x.c),
                grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
            return
        var ctx = _ReCtx()
        ctx.op = _OP_PRELU
        ctx.x = self.rt.addr(x)
        ctx.a = Int(self.slopes[layer].unsafe_ptr())
        ctx.n = n
        ctx.C = x.c
        run_parallel_ctx(self.rt.pool, (n + _CHUNK - 1) // _CHUNK, ctx, _re_worker)

    def _ps_add(mut self, h: Act, base: Act, r: Int) raises -> Act:
        var o = self.rt.alloc(1, h.h * r, h.w * r, base.c)
        var n = o.numel()
        if self.rt.on_gpu():
            self.rt.gpu.value().ctx.enqueue_function[_re_ps_add_kernel](
                self.rt.ptr(h), self.rt.ptr(base), self.rt.ptr(o), Int32(h.h), Int32(h.w), Int32(base.c), Int32(r),
                grid_dim=(n + DK_EW - 1) // DK_EW, block_dim=DK_EW)
            return o^
        var ctx = _ReCtx()
        ctx.op = _OP_PS_ADD
        ctx.x = self.rt.addr(h)
        ctx.b = self.rt.addr(base)
        ctx.o = self.rt.addr(o)
        ctx.n = n
        ctx.C = base.c
        ctx.H = h.h
        ctx.W = h.w
        ctx.r = r
        run_parallel_ctx(self.rt.pool, (n + _CHUNK - 1) // _CHUNK, ctx, _re_worker)
        return o^

    def _concat(mut self, a: Act, b: Act) raises -> Act:
        if self.rt.on_gpu():
            return self.rt.concat_c(a, b)
        var o = self.rt.alloc(a.b, a.h, a.w, a.c + b.c)
        var C = a.c + b.c
        var po = self.rt.addr(o)
        var pa = self.rt.addr(a)
        var pb = self.rt.addr(b)
        for r in range(a.rows()):
            _ = external_call["memcpy", Int](po + r * C * 4, pa + r * a.c * 4, a.c * 4)
            _ = external_call["memcpy", Int](po + (r * C + a.c) * 4, pb + r * b.c * 4, b.c * 4)
        return o^

    def _rdb(mut self, x: Act, ci: Int) raises -> Act:
        var x1 = self._conv(x, ci, _ACT_LRELU)
        var c1 = self._concat(x, x1)
        self.rt.free(x1^)
        var x2 = self._conv(c1, ci + 1, _ACT_LRELU)
        var c2 = self._concat(c1, x2)
        self.rt.free(c1^)
        self.rt.free(x2^)
        var x3 = self._conv(c2, ci + 2, _ACT_LRELU)
        var c3 = self._concat(c2, x3)
        self.rt.free(c2^)
        self.rt.free(x3^)
        var x4 = self._conv(c3, ci + 3, _ACT_LRELU)
        var c4 = self._concat(c3, x4)
        self.rt.free(c3^)
        self.rt.free(x4^)
        var x5 = self._conv(c4, ci + 4, _ACT_NONE)
        self.rt.free(c4^)
        self.rt.scale_(x5, 0.2)
        self.rt.add_(x5, x)
        return x5^

    def _rrdb(mut self, x: Act, ci: Int) raises -> Act:
        var o1 = self._rdb(x, ci)
        var o2 = self._rdb(o1, ci + 5)
        self.rt.free(o1^)
        var o3 = self._rdb(o2, ci + 10)
        self.rt.free(o2^)
        self.rt.scale_(o3, 0.2)
        self.rt.add_(o3, x)
        return o3^

    def forward(mut self, x: Act) raises -> Act:
        """[1, h, w, 3] in 0..1 -> [1, scale h, scale w, 3] (unclamped), on the runtime's device."""
        if self.arch == ARCH_COMPACT:
            var nconv = len(self.convs)
            var h = self._conv(x, 0, _ACT_PRELU, 1, 0)
            for i in range(1, nconv - 1):
                var nh = self._conv(h, i, _ACT_PRELU, 1, i)
                self.rt.free(h^)
                h = nh^
            var last = self._conv(h, nconv - 1, _ACT_NONE)
            self.rt.free(h^)
            var o = self._ps_add(last, x, self.scale)
            self.rt.free(last^)
            return o^
        var feat = self._conv(x, 0, _ACT_NONE)
        var cur = self.rt.copy(feat)
        for b in range(self.num_block):
            var nxt = self._rrdb(cur, 1 + b * 15)
            self.rt.free(cur^)
            cur = nxt^
        var k = 1 + self.num_block * 15
        var body = self._conv(cur, k, _ACT_NONE)
        self.rt.free(cur^)
        self.rt.add_(feat, body)
        self.rt.free(body^)
        var u1 = self._conv(feat, k + 1, _ACT_LRELU, 2)
        self.rt.free(feat^)
        var u2 = self._conv(u1, k + 2, _ACT_LRELU, 2)
        self.rt.free(u1^)
        var hr = self._conv(u2, k + 3, _ACT_LRELU)
        self.rt.free(u2^)
        var o = self._conv(hr, k + 4, _ACT_NONE)
        self.rt.free(hr^)
        return o^

    def run(mut self, rgb: List[Float32], w: Int, h: Int) raises -> List[Float32]:
        """One forward pass on a host [h, w, 3] image; returns [scale h, scale w, 3]."""
        var x = self.rt.upload(1, h, w, 3, rgb)
        var y = self.forward(x)
        self.rt.free(x^)
        var out = self.rt.download(y)
        self.rt.free(y^)
        return out^

    def upscale_rgb(mut self, rgb: List[Float32], w: Int, h: Int, tile: Int = 0, tile_pad: Int = 10) raises -> List[Float32]:
        """[h, w, 3] RGB in 0..1 -> [s h, s w, 3] raw network output. `tile > 0` splits the input
        into tile x tile pieces, each run with `tile_pad` pixels of context and cropped back
        (`RealESRGANer.tile_process`), which bounds memory for large images."""
        if tile <= 0 or (tile >= w and tile >= h):
            return self.run(rgb, w, h)
        var s = self.scale
        var W = w * s
        var out = List[Float32](length=h * s * W * 3, fill=0.0)
        for ty in range(tile_count(h, tile)):
            for tx in range(tile_count(w, tile)):
                var sx = tx * tile
                var ex = min(sx + tile, w)
                var sy = ty * tile
                var ey = min(sy + tile, h)
                var sxp = max(sx - tile_pad, 0)
                var exp = min(ex + tile_pad, w)
                var syp = max(sy - tile_pad, 0)
                var eyp = min(ey + tile_pad, h)
                var tw = exp - sxp
                var th = eyp - syp
                var t = List[Float32](length=tw * th * 3, fill=0.0)
                for y in range(th):
                    _ = external_call["memcpy", Int](Int(t.unsafe_ptr()) + y * tw * 12,
                                                     Int(rgb.unsafe_ptr()) + ((syp + y) * w + sxp) * 12, tw * 12)
                var o = self.run(t, tw, th)
                var otx = (sx - sxp) * s
                var oty = (sy - syp) * s
                var cw = (ex - sx) * s
                for y in range((ey - sy) * s):
                    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()) + ((sy * s + y) * W + sx * s) * 12,
                                                     Int(o.unsafe_ptr()) + ((oty + y) * tw * s + otx) * 12, cw * 12)
        return out^

    def upscale_image(mut self, img: NativeImage, tile: Int = 0, tile_pad: Int = 10) raises -> NativeImage:
        """A 0xAARRGGBB image upscaled `scale` times. A non-opaque alpha channel goes through the
        network as a grey image (then BT.601 grey, like cv2.COLOR_BGR2GRAY in the reference)."""
        var w = img.width
        var h = img.height
        var rgb = List[Float32](length=w * h * 3, fill=0.0)
        var opaque = True
        for i in range(w * h):
            var p = img.pixels[i]
            rgb[i * 3] = Float32(Int((p >> 16) & 255)) / 255.0
            rgb[i * 3 + 1] = Float32(Int((p >> 8) & 255)) / 255.0
            rgb[i * 3 + 2] = Float32(Int(p & 255)) / 255.0
            if (p >> 24) != 255:
                opaque = False
        var y = self.upscale_rgb(rgb, w, h, tile, tile_pad)
        var s = self.scale
        var out = NativeImage()
        out.width = w * s
        out.height = h * s
        out.valid = True
        var n = out.width * out.height
        out.pixels = List[UInt32](length=n, fill=0xFF000000)
        for i in range(n):
            out.pixels[i] = 0xFF000000 | (to_u8(y[i * 3]) << 16) | (to_u8(y[i * 3 + 1]) << 8) | to_u8(y[i * 3 + 2])
        if not opaque:
            var a = List[Float32](length=w * h * 3, fill=0.0)
            for i in range(w * h):
                var v = Float32(Int(img.pixels[i] >> 24)) / 255.0
                a[i * 3] = v
                a[i * 3 + 1] = v
                a[i * 3 + 2] = v
            var ya = self.upscale_rgb(a, w, h, tile, tile_pad)
            for i in range(n):
                var r = Float32(Int(to_u8(ya[i * 3])))
                var g = Float32(Int(to_u8(ya[i * 3 + 1])))
                var b = Float32(Int(to_u8(ya[i * 3 + 2])))
                var grey = UInt32(Int(round_half_even(0.299 * r + 0.587 * g + 0.114 * b)))
                out.pixels[i] = (out.pixels[i] & 0x00FFFFFF) | (min(grey, UInt32(255)) << 24)
        return out^


def _param(st: SafeTensors, blend: SafeTensors, has_blend: Bool, strength: Float32, name: String) raises -> Tensor:
    var a = st.weight(name)
    if not has_blend:
        return a^
    var b = blend.weight(name)
    if b.numel() != a.numel():
        raise Error("realesrgan: " + name + " differs in shape between the blended checkpoints")
    var v = a.owned_floats()
    var bv = b.owned_floats()
    for i in range(len(v)):
        v[i] = strength * v[i] + (1.0 - strength) * bv[i]
    return Tensor.floats(a.shape.copy(), v^)
