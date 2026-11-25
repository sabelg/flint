"""Vulkan compute path for the pixel-art magnifiers (`flint.render.pixel_scale_vk`).

The SPIR-V twins of `flint.render.pixel_scale` (Scale2x / EPX, Scale3x, Scale4x, MMPX, the three
Super-xBR passes and the palette lock), assembled with `flint.render.spirv.SpirvModule` and run
headless on `flint.platform.vulkan.VulkanGpu` storage buffers. One invocation handles one
source pixel (one 2x2 / 3x3 output block) like the paper's MMPX compute shader, or one output
pixel for Super-xBR's third pass and the palette lock.

The float arithmetic of Super-xBR is decorated `NoContraction` and issued in the CPU's order,
so the driver cannot fuse it into FMAs and the GPU gives the CPU's pixels bit for bit; the
other filters only compare and copy 32-bit words.

Bindings: 0 = the original image, 1 = the pass input, 2 = the pass output (all ARGB words).
Push words: source w, h, input w, h, output w, h.
"""

from flint.assets.image import NativeImage
from flint.platform.vulkan import (
    VulkanGpu, PIPELINE_COMPUTE, PIPELINE_HOST, ACCESS_SHADER_WRITE, ACCESS_HOST_READ,
)
from flint.platform.amdgpu import GpuBuffer
from flint.render.spirv import SpirvModule, emit, OP_DECORATE
from flint.render.pixel_scale import (
    FILTER_NEAREST, FILTER_SCALE2X, FILTER_SCALE3X, FILTER_SCALE4X, FILTER_MMPX, FILTER_SUPER_XBR,
    SXBR_WGT1, SXBR_WGT2, new_image, filter_factor,
)

comptime _GROUP: Int = 8
comptime _PUSH_WORDS: Int = 6
comptime _P_SRC_W: Int = 0
comptime _P_SRC_H: Int = 1
comptime _P_IN_W: Int = 2
comptime _P_IN_H: Int = 3
comptime _P_OUT_W: Int = 4
comptime _P_OUT_H: Int = 5

comptime _BOUND_INPUT: Int = 0
comptime _BOUND_OUTPUT: Int = 1
comptime _BOUND_SOURCE: Int = 2
comptime _DECORATION_NO_CONTRACTION: UInt32 = 42
comptime _GLSL_CEIL: UInt32 = 9
comptime _GLSL_FABS: UInt32 = 4
comptime _GLSL_FMIN: UInt32 = 37
comptime _GLSL_FMAX: UInt32 = 40

# Pipeline indices, in creation order.
comptime SH_NEAREST: Int = 0
comptime SH_SCALE2X: Int = 1
comptime SH_SCALE3X: Int = 2
comptime SH_MMPX: Int = 3
comptime SH_SXBR1: Int = 4
comptime SH_SXBR2: Int = 5
comptime SH_SXBR3: Int = 6
comptime SH_SNAP: Int = 7


# ----------------------------------------------------------------------------- shader helpers

struct _Ctx(Movable):
    var src: UInt32
    var inp: UInt32
    var out: UInt32
    var x: UInt32
    var y: UInt32
    var in_w: UInt32
    var in_h: UInt32
    var out_w: UInt32
    var max_x: UInt32
    var max_y: UInt32

    def __init__(out self, mut s: SpirvModule, bound: Int):
        """Common prologue: bindings, the invocation's pixel, bounds check."""
        self.src = s.buffer(0)
        self.inp = s.buffer(1)
        self.out = s.buffer(2)
        self.x = s.invocation_axis(0)
        self.y = s.invocation_axis(1)
        self.in_w = s.push_word(_P_IN_W)
        self.in_h = s.push_word(_P_IN_H)
        self.out_w = s.push_word(_P_OUT_W)
        var one = s.uint(1)
        self.max_x = s.isub(self.in_w, one)
        self.max_y = s.isub(self.in_h, one)
        if bound == _BOUND_OUTPUT:
            s.return_if(s.uge(self.x, self.out_w))
            s.return_if(s.uge(self.y, s.push_word(_P_OUT_H)))
        elif bound == _BOUND_SOURCE:
            s.return_if(s.uge(self.x, s.push_word(_P_SRC_W)))
            s.return_if(s.uge(self.y, s.push_word(_P_SRC_H)))
        else:
            s.return_if(s.uge(self.x, self.in_w))
            s.return_if(s.uge(self.y, self.in_h))


def _clamp_coord(mut s: SpirvModule, v: UInt32, hi: UInt32) -> UInt32:
    return s.smin(s.smax(v, s.uint(0)), hi)


def _offset(mut s: SpirvModule, v: UInt32, d: Int) -> UInt32:
    if d == 0:
        return v
    if d > 0:
        return s.iadd(v, s.uint(UInt32(d)))
    return s.isub(v, s.uint(UInt32(-d)))


def _fetch(mut s: SpirvModule, c: _Ctx, dx: Int, dy: Int) -> UInt32:
    """Input word at (x + dx, y + dy), clamped to the edge."""
    var xx = _clamp_coord(s, _offset(s, c.x, dx), c.max_x)
    var yy = _clamp_coord(s, _offset(s, c.y, dy), c.max_y)
    return s.load_word(c.inp, s.iadd(s.imul(yy, c.in_w), xx))


def _fetch_at(mut s: SpirvModule, buf: UInt32, w: UInt32, max_x: UInt32, max_y: UInt32, x: UInt32, y: UInt32,
              dx: Int, dy: Int) -> UInt32:
    var xx = _clamp_coord(s, _offset(s, x, dx), max_x)
    var yy = _clamp_coord(s, _offset(s, y, dy), max_y)
    return s.load_word(buf, s.iadd(s.imul(yy, w), xx))


def _store(mut s: SpirvModule, c: _Ctx, ox: UInt32, oy: UInt32, value: UInt32):
    s.store_word(c.out, s.iadd(s.imul(oy, c.out_w), ox), value)


def _and(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.logical_and(a, b)


def _or(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.logical_or(a, b)


def _eq(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.ieq(a, b)


def _ne(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.ine(a, b)


def _block_store(mut s: SpirvModule, c: _Ctx, k: Int, vals: List[UInt32]):
    """Write the k x k block of source pixel (x, y)."""
    var bx = s.imul(c.x, s.uint(UInt32(k)))
    var by = s.imul(c.y, s.uint(UInt32(k)))
    for j in range(k):
        for i in range(k):
            _store(s, c, _offset(s, bx, i), _offset(s, by, j), vals[j * k + i])


# ----------------------------------------------------------------------------- shaders

def nearest_shader() -> List[UInt32]:
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_INPUT)
    var e = _fetch(s, c, 0, 0)
    var v: List[UInt32] = [e, e, e, e]
    _block_store(s, c, 2, v)
    return s.finish()


def scale2x_shader() -> List[UInt32]:
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_INPUT)
    var B = _fetch(s, c, 0, -1)
    var D = _fetch(s, c, -1, 0)
    var E = _fetch(s, c, 0, 0)
    var F = _fetch(s, c, 1, 0)
    var H = _fetch(s, c, 0, 1)
    var go = _and(s, _ne(s, B, H), _ne(s, D, F))
    var e0 = s.select_u(_and(s, go, _eq(s, D, B)), D, E)
    var e1 = s.select_u(_and(s, go, _eq(s, B, F)), F, E)
    var e2 = s.select_u(_and(s, go, _eq(s, D, H)), D, E)
    var e3 = s.select_u(_and(s, go, _eq(s, H, F)), F, E)
    _block_store(s, c, 2, [e0, e1, e2, e3])
    return s.finish()


def scale3x_shader() -> List[UInt32]:
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_INPUT)
    var A = _fetch(s, c, -1, -1)
    var B = _fetch(s, c, 0, -1)
    var C = _fetch(s, c, 1, -1)
    var D = _fetch(s, c, -1, 0)
    var E = _fetch(s, c, 0, 0)
    var F = _fetch(s, c, 1, 0)
    var G = _fetch(s, c, -1, 1)
    var H = _fetch(s, c, 0, 1)
    var I = _fetch(s, c, 1, 1)
    var go = _and(s, _ne(s, B, H), _ne(s, D, F))
    var db = _eq(s, D, B)
    var bf = _eq(s, B, F)
    var dh = _eq(s, D, H)
    var hf = _eq(s, H, F)
    var v = List[UInt32]()
    v.append(s.select_u(_and(s, go, db), D, E))
    v.append(s.select_u(_and(s, go, _or(s, _and(s, db, _ne(s, E, C)), _and(s, bf, _ne(s, E, A)))), B, E))
    v.append(s.select_u(_and(s, go, bf), F, E))
    v.append(s.select_u(_and(s, go, _or(s, _and(s, db, _ne(s, E, G)), _and(s, dh, _ne(s, E, A)))), D, E))
    v.append(E)
    v.append(s.select_u(_and(s, go, _or(s, _and(s, bf, _ne(s, E, I)), _and(s, hf, _ne(s, E, C)))), F, E))
    v.append(s.select_u(_and(s, go, dh), D, E))
    v.append(s.select_u(_and(s, go, _or(s, _and(s, dh, _ne(s, E, I)), _and(s, hf, _ne(s, E, G)))), H, E))
    v.append(s.select_u(_and(s, go, hf), F, E))
    _block_store(s, c, 3, v)
    return s.finish()


def _luma(mut s: SpirvModule, p: UInt32) -> UInt32:
    """MMPX luma: (R + G + B + 1) * (256 - alpha)."""
    var m = s.uint(255)
    var a = s.shr(p, s.uint(24))
    var sum = s.iadd(s.iadd(s.iadd(s.bit_and(s.shr(p, s.uint(16)), m), s.bit_and(s.shr(p, s.uint(8)), m)),
                            s.bit_and(p, m)), s.uint(1))
    return s.imul(sum, s.isub(s.uint(256), a))


def _all_eq(mut s: SpirvModule, b: UInt32, vals: List[UInt32]) -> UInt32:
    var r = _eq(s, b, vals[0])
    for i in range(1, len(vals)):
        r = _and(s, r, _eq(s, b, vals[i]))
    return r


def _none_eq(mut s: SpirvModule, b: UInt32, vals: List[UInt32]) -> UInt32:
    var r = _ne(s, b, vals[0])
    for i in range(1, len(vals)):
        r = _and(s, r, _ne(s, b, vals[i]))
    return r


def _any_eq(mut s: SpirvModule, b: UInt32, vals: List[UInt32]) -> UInt32:
    var r = _eq(s, b, vals[0])
    for i in range(1, len(vals)):
        r = _or(s, r, _eq(s, b, vals[i]))
    return r


def _assign_if(mut s: SpirvModule, cond: UInt32, var_id: UInt32, value: UInt32):
    s.store(var_id, s.select_u(cond, value, s.load_u(var_id)))


def mmpx_shader() -> List[UInt32]:
    """MMPX (McGuire & Gagiu, JCGT 2021), rule for rule as the reference MMPX.glc."""
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_INPUT)
    var A = _fetch(s, c, -1, -1)
    var B = _fetch(s, c, 0, -1)
    var C = _fetch(s, c, 1, -1)
    var D = _fetch(s, c, -1, 0)
    var E = _fetch(s, c, 0, 0)
    var F = _fetch(s, c, 1, 0)
    var G = _fetch(s, c, -1, 1)
    var H = _fetch(s, c, 0, 1)
    var I = _fetch(s, c, 1, 1)
    var J = s.uvar(E)
    var K = s.uvar(E)
    var L = s.uvar(E)
    var M = s.uvar(E)
    var diff = s.bit_or(s.bit_or(s.bit_or(s.bit_xor(A, E), s.bit_xor(B, E)), s.bit_or(s.bit_xor(C, E), s.bit_xor(D, E))),
                        s.bit_or(s.bit_or(s.bit_xor(F, E), s.bit_xor(G, E)), s.bit_or(s.bit_xor(H, E), s.bit_xor(I, E))))
    s.begin_if(_ne(s, diff, s.uint(0)))
    var P = _fetch(s, c, 0, -2)
    var S = _fetch(s, c, 0, 2)
    var Q = _fetch(s, c, -2, 0)
    var R = _fetch(s, c, 2, 0)
    var Bl = _luma(s, B)
    var Dl = _luma(s, D)
    var El = _luma(s, E)
    var Fl = _luma(s, F)
    var Hl = _luma(s, H)
    # 1:1 slope rules
    var r1 = _and(s, _and(s, _and(s, _and(s, _eq(s, D, B), _ne(s, D, H)), _ne(s, D, F)),
                          _or(s, s.uge(El, Dl), _eq(s, E, A))),
                  _and(s, _any_eq(s, E, [A, C, G]),
                       _or(s, _or(s, s.ult(El, Dl), _ne(s, A, D)), _or(s, _ne(s, E, P), _ne(s, E, Q)))))
    _assign_if(s, r1, J, D)
    var r2 = _and(s, _and(s, _and(s, _and(s, _eq(s, B, F), _ne(s, B, D)), _ne(s, B, H)),
                          _or(s, s.uge(El, Bl), _eq(s, E, C))),
                  _and(s, _any_eq(s, E, [A, C, I]),
                       _or(s, _or(s, s.ult(El, Bl), _ne(s, C, B)), _or(s, _ne(s, E, P), _ne(s, E, R)))))
    _assign_if(s, r2, K, B)
    var r3 = _and(s, _and(s, _and(s, _and(s, _eq(s, H, D), _ne(s, H, F)), _ne(s, H, B)),
                          _or(s, s.uge(El, Hl), _eq(s, E, G))),
                  _and(s, _any_eq(s, E, [A, G, I]),
                       _or(s, _or(s, s.ult(El, Hl), _ne(s, G, H)), _or(s, _ne(s, E, S), _ne(s, E, Q)))))
    _assign_if(s, r3, L, H)
    var r4 = _and(s, _and(s, _and(s, _and(s, _eq(s, F, H), _ne(s, F, B)), _ne(s, F, D)),
                          _or(s, s.uge(El, Fl), _eq(s, E, I))),
                  _and(s, _any_eq(s, E, [C, G, I]),
                       _or(s, _or(s, s.ult(El, Fl), _ne(s, I, H)), _or(s, _ne(s, E, R), _ne(s, E, S)))))
    _assign_if(s, r4, M, F)
    # Intersection rules
    var i1 = _and(s, _and(s, _ne(s, E, F), _all_eq(s, E, [C, I, D, Q])),
                  _and(s, _all_eq(s, F, [B, H]), _ne(s, F, _fetch(s, c, 3, 0))))
    _assign_if(s, i1, K, F)
    _assign_if(s, i1, M, F)
    var i2 = _and(s, _and(s, _ne(s, E, D), _all_eq(s, E, [A, G, F, R])),
                  _and(s, _all_eq(s, D, [B, H]), _ne(s, D, _fetch(s, c, -3, 0))))
    _assign_if(s, i2, J, D)
    _assign_if(s, i2, L, D)
    var i3 = _and(s, _and(s, _ne(s, E, H), _all_eq(s, E, [G, I, B, P])),
                  _and(s, _all_eq(s, H, [D, F]), _ne(s, H, _fetch(s, c, 0, 3))))
    _assign_if(s, i3, L, H)
    _assign_if(s, i3, M, H)
    var i4 = _and(s, _and(s, _ne(s, E, B), _all_eq(s, E, [A, C, H, S])),
                  _and(s, _all_eq(s, B, [D, F]), _ne(s, B, _fetch(s, c, 0, -3))))
    _assign_if(s, i4, J, B)
    _assign_if(s, i4, K, B)
    var i5 = _and(s, _and(s, s.ult(Bl, El), _all_eq(s, E, [G, H, I, S])), _none_eq(s, E, [A, D, C, F]))
    _assign_if(s, i5, J, B)
    _assign_if(s, i5, K, B)
    var i6 = _and(s, _and(s, s.ult(Hl, El), _all_eq(s, E, [A, B, C, P])), _none_eq(s, E, [D, G, I, F]))
    _assign_if(s, i6, L, H)
    _assign_if(s, i6, M, H)
    var i7 = _and(s, _and(s, s.ult(Fl, El), _all_eq(s, E, [A, D, G, Q])), _none_eq(s, E, [B, C, I, H]))
    _assign_if(s, i7, K, F)
    _assign_if(s, i7, M, F)
    var i8 = _and(s, _and(s, s.ult(Dl, El), _all_eq(s, E, [C, F, I, R])), _none_eq(s, E, [B, A, G, H]))
    _assign_if(s, i8, J, D)
    _assign_if(s, i8, L, D)
    # 2:1 slope rules: sequential, each reading the current J / K / L / M.
    var hb = _ne(s, H, B)
    var h_ok = _and(s, hb, _none_eq(s, H, [A, E, C]))
    var t1 = _and(s, h_ok, _and(s, _all_eq(s, H, [G, F, R]), _none_eq(s, H, [D, _fetch(s, c, 2, -1)])))
    _assign_if(s, t1, L, s.load_u(M))
    var t2 = _and(s, h_ok, _and(s, _all_eq(s, H, [I, D, Q]), _none_eq(s, H, [F, _fetch(s, c, -2, -1)])))
    _assign_if(s, t2, M, s.load_u(L))
    var b_ok = _and(s, hb, _none_eq(s, B, [I, G, E]))
    var t3 = _and(s, b_ok, _and(s, _all_eq(s, B, [A, F, R]), _none_eq(s, B, [D, _fetch(s, c, 2, 1)])))
    _assign_if(s, t3, J, s.load_u(K))
    var t4 = _and(s, b_ok, _and(s, _all_eq(s, B, [C, D, Q]), _none_eq(s, B, [F, _fetch(s, c, -2, 1)])))
    _assign_if(s, t4, K, s.load_u(J))
    var fd = _ne(s, F, D)
    var d_ok = _and(s, fd, _none_eq(s, D, [I, E, C]))
    var t5 = _and(s, d_ok, _and(s, _all_eq(s, D, [A, H, S]), _none_eq(s, D, [B, _fetch(s, c, 1, 2)])))
    _assign_if(s, t5, J, s.load_u(L))
    var t6 = _and(s, d_ok, _and(s, _all_eq(s, D, [G, B, P]), _none_eq(s, D, [H, _fetch(s, c, 1, -2)])))
    _assign_if(s, t6, L, s.load_u(J))
    var f_ok = _and(s, fd, _none_eq(s, F, [E, A, G]))
    var t7 = _and(s, f_ok, _and(s, _all_eq(s, F, [C, H, S]), _none_eq(s, F, [B, _fetch(s, c, -1, 2)])))
    _assign_if(s, t7, K, s.load_u(M))
    var t8 = _and(s, f_ok, _and(s, _all_eq(s, F, [I, B, P]), _none_eq(s, F, [H, _fetch(s, c, -1, -2)])))
    _assign_if(s, t8, M, s.load_u(K))
    s.end_if()
    var v: List[UInt32] = [s.load_u(J), s.load_u(K), s.load_u(L), s.load_u(M)]
    _block_store(s, c, 2, v)
    return s.finish()


# Super-xBR ------------------------------------------------------------------

def _nc(mut s: SpirvModule, id: UInt32) -> UInt32:
    emit(s.decorations, OP_DECORATE, id, _DECORATION_NO_CONTRACTION)
    return id


def _fadd(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return _nc(s, s.fadd(a, b))


def _fsub(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return _nc(s, s.fsub(a, b))


def _fmul(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return _nc(s, s.fmul(a, b))


def _fmin(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.ext2(_GLSL_FMIN, s.type_float, a, b)


def _fmax(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.ext2(_GLSL_FMAX, s.type_float, a, b)


def _df(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    return s.ext1(_GLSL_FABS, s.type_float, _fsub(s, a, b))


def _chan(mut s: SpirvModule, p: UInt32, shift: Int) -> UInt32:
    return s.float_from_uint(s.bit_and(s.shr(p, s.uint(UInt32(shift))), s.uint(255)))


def _sxbr_sample(mut s: SpirvModule, words: List[UInt32], wp: List[Float32], wa: Float32, wb: Float32) -> UInt32:
    """The SPIR-V of `pixel_scale._sxbr_pixel` for 16 loaded words (index i * 4 + j)."""
    var r = List[UInt32]()
    var g = List[UInt32]()
    var b = List[UInt32]()
    var a = List[UInt32]()
    var y = List[UInt32]()
    var cr = s.float(0.2126)
    var cg = s.float(0.7152)
    var cb = s.float(0.0722)
    for k in range(16):
        var rr = _chan(s, words[k], 16)
        var gg = _chan(s, words[k], 8)
        var bb = _chan(s, words[k], 0)
        r.append(rr)
        g.append(gg)
        b.append(bb)
        a.append(_chan(s, words[k], 24))
        y.append(_fadd(s, _fadd(s, _fmul(s, cr, rr), _fmul(s, cg, gg)), _fmul(s, cb, bb)))
    var w0 = s.float(wp[0])
    var w1 = s.float(wp[1])
    var w2 = s.float(wp[2])
    var w3 = s.float(wp[3])
    var w4 = s.float(wp[4])
    var w5 = s.float(wp[5])
    var dw1 = _fmul(s, w0, _fadd(s, _fadd(s, _fadd(s, _df(s, y[2], y[5]), _df(s, y[5], y[8])), _df(s, y[7], y[10])), _df(s, y[10], y[13])))
    dw1 = _fadd(s, dw1, _fmul(s, w1, _fadd(s, _df(s, y[3], y[6]), _df(s, y[9], y[12]))))
    dw1 = _fadd(s, dw1, _fmul(s, w2, _fadd(s, _df(s, y[3], y[9]), _df(s, y[6], y[12]))))
    dw1 = _fadd(s, dw1, _fmul(s, w3, _df(s, y[6], y[9])))
    dw1 = _fadd(s, dw1, _fmul(s, w4, _fadd(s, _df(s, y[2], y[8]), _df(s, y[7], y[13]))))
    dw1 = _fadd(s, dw1, _fmul(s, w5, _fadd(s, _df(s, y[1], y[4]), _df(s, y[11], y[14]))))
    var dw2 = _fmul(s, w0, _fadd(s, _fadd(s, _fadd(s, _df(s, y[1], y[6]), _df(s, y[6], y[11])), _df(s, y[4], y[9])), _df(s, y[9], y[14])))
    dw2 = _fadd(s, dw2, _fmul(s, w1, _fadd(s, _df(s, y[0], y[5]), _df(s, y[10], y[15]))))
    dw2 = _fadd(s, dw2, _fmul(s, w2, _fadd(s, _df(s, y[0], y[10]), _df(s, y[5], y[15]))))
    dw2 = _fadd(s, dw2, _fmul(s, w3, _df(s, y[5], y[10])))
    dw2 = _fadd(s, dw2, _fmul(s, w4, _fadd(s, _df(s, y[4], y[14]), _df(s, y[1], y[11]))))
    dw2 = _fadd(s, dw2, _fmul(s, w5, _fadd(s, _df(s, y[2], y[7]), _df(s, y[8], y[13]))))
    var d = _fsub(s, dw1, dw2)
    var main_diag = s.fgt(d, s.float(0.0))
    var fa = s.float(wa)
    var fb = s.float(wb)
    var result = s.uint(0)
    for ch in range(4):
        var v: List[UInt32]
        if ch == 0:
            v = r.copy()
        elif ch == 1:
            v = g.copy()
        elif ch == 2:
            v = b.copy()
        else:
            v = a.copy()
        var lo = _fmin(s, _fmin(s, v[5], v[9]), _fmin(s, v[6], v[10]))
        var hi = _fmax(s, _fmax(s, v[5], v[9]), _fmax(s, v[6], v[10]))
        var f1 = _fadd(s, _fmul(s, fa, _fadd(s, v[3], v[12])), _fmul(s, fb, _fadd(s, v[6], v[9])))
        var f2 = _fadd(s, _fmul(s, fa, _fadd(s, v[0], v[15])), _fmul(s, fb, _fadd(s, v[5], v[10])))
        var f = s.select_f(main_diag, f2, f1)
        f = _fmin(s, _fmax(s, f, lo), hi)
        var q = s.uint_from_float(s.ext1(_GLSL_CEIL, s.type_float, f))
        q = s.umin(q, s.uint(255))
        var shift = 16 - ch * 8 if ch < 3 else 24
        result = s.bit_or(result, s.shl(q, s.uint(UInt32(shift))))
    return result


def _wp1() -> List[Float32]:
    return [Float32(2.0), Float32(1.0), Float32(-1.0), Float32(4.0), Float32(-1.0), Float32(1.0)]


def _wp2() -> List[Float32]:
    return [Float32(2.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]


def sxbr_pass1_shader() -> List[UInt32]:
    """Per source pixel: copy it to its 2x2 block, the odd/odd corner from the 4x4 window."""
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_INPUT)
    var words = List[UInt32]()
    for i in range(4):
        for j in range(4):
            words.append(_fetch(s, c, i - 1, j - 1))
    var e = words[5]
    var n = _sxbr_sample(s, words, _wp1(), -SXBR_WGT1, SXBR_WGT1 + 0.5)
    _block_store(s, c, 2, [e, e, e, n])
    return s.finish()


def sxbr_pass2_shader() -> List[UInt32]:
    """Per even/even output position (x, y) = 2 * invocation: (x + 1, y) and (x, y + 1) from
    the 45-degree lattice; (x, y) and (x + 1, y + 1) are copied through."""
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_SOURCE)   # one invocation per source pixel
    var two = s.uint(2)
    var x = s.imul(c.x, two)
    var y = s.imul(c.y, two)
    var w = c.in_w
    var wa = List[UInt32]()
    var wb = List[UInt32]()
    for i in range(4):
        for j in range(4):
            wa.append(_fetch_at(s, c.inp, w, c.max_x, c.max_y, x, y, (i - 1) + (j - 1), (i - 1) - (j - 1)))
    for i in range(4):
        for j in range(4):
            wb.append(_fetch_at(s, c.inp, w, c.max_x, c.max_y, x, y, (i - 1) + (j - 1) - 1, (i - 1) - (j - 1) + 1))
    var right = _sxbr_sample(s, wa, _wp2(), -SXBR_WGT2, SXBR_WGT2 + 0.5)
    var below = _sxbr_sample(s, wb, _wp2(), -SXBR_WGT2, SXBR_WGT2 + 0.5)
    var e = _fetch_at(s, c.inp, w, c.max_x, c.max_y, x, y, 0, 0)
    var d = _fetch_at(s, c.inp, w, c.max_x, c.max_y, x, y, 1, 1)
    var x1 = s.iadd(x, s.uint(1))
    var y1 = s.iadd(y, s.uint(1))
    _store(s, c, x, y, e)
    _store(s, c, x1, y, right)
    _store(s, c, x, y1, below)
    _store(s, c, x1, y1, d)
    return s.finish()


def sxbr_pass3_shader() -> List[UInt32]:
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_OUTPUT)
    var words = List[UInt32]()
    for i in range(4):
        for j in range(4):
            words.append(_fetch(s, c, i - 2, j - 2))
    _store(s, c, c.x, c.y, _sxbr_sample(s, words, _wp1(), -SXBR_WGT1, SXBR_WGT1 + 0.5))
    return s.finish()


def _cdist(mut s: SpirvModule, a: UInt32, b: UInt32) -> UInt32:
    var m = s.uint(255)
    var total = s.uint(0)
    for sh in range(4):
        var ca = s.bit_and(s.shr(a, s.uint(UInt32(sh * 8))), m)
        var cb = s.bit_and(s.shr(b, s.uint(UInt32(sh * 8))), m)
        var d = s.isub(ca, cb)
        total = s.iadd(total, s.imul(d, d))
    return total


def snap_shader() -> List[UInt32]:
    """Palette lock: each output pixel -> closest colour of its 3 x 3 source neighbourhood
    (centre first, then row-major, strict improvement), as `pixel_scale.palette_snap`."""
    var s = SpirvModule(_GROUP, _GROUP, _PUSH_WORDS)
    var c = _Ctx(s, _BOUND_OUTPUT)
    var sw = s.push_word(_P_SRC_W)
    var sh = s.push_word(_P_SRC_H)
    var one = s.uint(1)
    var smx = s.isub(sw, one)
    var smy = s.isub(sh, one)
    var p = s.load_word(c.inp, s.iadd(s.imul(c.y, c.out_w), c.x))
    var two = s.uint(2)
    var sx = s.shr(c.x, one)
    var sy = s.shr(c.y, one)
    _ = two
    var first = _fetch_at(s, c.src, sw, smx, smy, sx, sy, 0, 0)
    var best = s.uvar(first)
    var bd = s.uvar(_cdist(s, p, first))
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var col = _fetch_at(s, c.src, sw, smx, smy, sx, sy, dx, dy)
            var d = _cdist(s, p, col)
            var better = s.ult(d, s.load_u(bd))
            s.store(best, s.select_u(better, col, s.load_u(best)))
            s.store(bd, s.select_u(better, d, s.load_u(bd)))
    _store(s, c, c.x, c.y, s.load_u(best))
    return s.finish()


# ----------------------------------------------------------------------------- runner

struct PixelScaleVulkan(Movable):
    """Headless Vulkan compute runner. `open()` False -> `error` says why (no loader / GPU)."""
    var vk: VulkanGpu
    var ready: Bool
    var error: String
    var bufs: List[GpuBuffer]   # 0 = source, 1 = A, 2 = B
    var caps: List[Int]

    def __init__(out self):
        self.vk = VulkanGpu()
        self.ready = False
        self.error = String()
        self.bufs = List[GpuBuffer]()
        self.caps = List[Int]()
        for _ in range(3):
            self.bufs.append(GpuBuffer())
            self.caps.append(0)

    def __deinit__(deinit self):
        if self.ready:
            self.vk.wait_idle()
            for i in range(3):
                self.vk.destroy_buffer(self.bufs[i])
            self.vk.close()

    def open(mut self) -> Bool:
        if self.ready:
            return True
        if not self.vk.initialize():
            self.error = self.vk.error
            return False
        if not self.vk.create_layouts(3, _PUSH_WORDS * 4) or not self.vk.allocate_command_buffers(1):
            self.error = self.vk.error
            self.vk.close()
            return False
        var ok = self.vk.create_pipeline(nearest_shader())
        ok = ok and self.vk.create_pipeline(scale2x_shader())
        ok = ok and self.vk.create_pipeline(scale3x_shader())
        ok = ok and self.vk.create_pipeline(mmpx_shader())
        ok = ok and self.vk.create_pipeline(sxbr_pass1_shader())
        ok = ok and self.vk.create_pipeline(sxbr_pass2_shader())
        ok = ok and self.vk.create_pipeline(sxbr_pass3_shader())
        ok = ok and self.vk.create_pipeline(snap_shader())
        if not ok:
            self.error = self.vk.error
            self.vk.close()
            return False
        self.ready = True
        return True

    def device_name(self) -> String:
        return self.vk.device_name

    def _ensure(mut self, i: Int, words: Int) -> Bool:
        if self.caps[i] >= words:
            return True
        if self.caps[i] > 0:
            self.vk.wait_idle()
            self.vk.destroy_buffer(self.bufs[i])
        self.bufs[i] = self.vk.create_buffer(max(words, 1) * 4, True, False, True)
        if self.bufs[i].size == 0:
            self.caps[i] = 0
            self.error = self.vk.error
            return False
        self.caps[i] = words
        return True

    def run(mut self, img: NativeImage, filter: Int, palette_lock: Bool = False) raises -> NativeImage:
        """The same pixels as `pixel_scale.pixel_scale(img, filter, palette_lock)`."""
        if not self.open():
            raise Error("pixel_scale_vk: " + self.error)
        var w = img.width
        var h = img.height
        var k = filter_factor(filter)
        var ow = w * k
        var oh = h * k
        if not (self._ensure(0, w * h) and self._ensure(1, ow * oh) and self._ensure(2, ow * oh)):
            raise Error("pixel_scale_vk: " + self.error)
        var src = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=self.bufs[0].cpu_address)
        for i in range(w * h):
            src.unsafe_offset(i).unsafe_store(img.pixels[i])
        # descriptor sets: (in = S, out = A), (in = A, out = B), (in = B, out = A)
        var sets = self.vk.create_descriptor_sets(3, 3)
        if len(sets) != 3:
            raise Error("pixel_scale_vk: " + self.vk.error)
        var ins: List[Int] = [0, 1, 2]
        var outs: List[Int] = [1, 2, 1]
        for i in range(3):
            self.vk.write_descriptor(sets[i], 0, self.bufs[0])
            self.vk.write_descriptor(sets[i], 1, self.bufs[ins[i]])
            self.vk.write_descriptor(sets[i], 2, self.bufs[outs[i]])
        if not self.vk.begin_commands(0):
            raise Error("pixel_scale_vk: " + self.vk.error)
        var result_buf = 1
        if filter == FILTER_NEAREST:
            self._pass(sets[0], SH_NEAREST, w, h, w, h, ow, oh, w, h)
        elif filter == FILTER_SCALE2X:
            self._pass(sets[0], SH_SCALE2X, w, h, w, h, ow, oh, w, h)
        elif filter == FILTER_SCALE3X:
            self._pass(sets[0], SH_SCALE3X, w, h, w, h, ow, oh, w, h)
        elif filter == FILTER_MMPX:
            self._pass(sets[0], SH_MMPX, w, h, w, h, ow, oh, w, h)
        elif filter == FILTER_SCALE4X:
            self._pass(sets[0], SH_SCALE2X, w, h, w, h, 2 * w, 2 * h, w, h)
            self.vk.pass_barrier(0)
            self._pass(sets[1], SH_SCALE2X, w, h, 2 * w, 2 * h, ow, oh, 2 * w, 2 * h)
            result_buf = 2
        elif filter == FILTER_SUPER_XBR:
            self._pass(sets[0], SH_SXBR1, w, h, w, h, ow, oh, w, h)
            self.vk.pass_barrier(0)
            self._pass(sets[1], SH_SXBR2, w, h, ow, oh, ow, oh, w, h)
            self.vk.pass_barrier(0)
            self._pass(sets[2], SH_SXBR3, w, h, ow, oh, ow, oh, ow, oh)
            if palette_lock:
                self.vk.pass_barrier(0)
                self._pass(sets[1], SH_SNAP, w, h, ow, oh, ow, oh, ow, oh)
                result_buf = 2
        else:
            raise Error("pixel_scale_vk: unknown filter " + String(filter))
        self.vk.barrier(0, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
        var fence = self.vk.submit(0)
        if fence == 0 or not self.vk.wait(fence):
            raise Error("pixel_scale_vk: " + self.vk.error)
        var o = new_image(ow, oh)
        var res = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=self.bufs[result_buf].cpu_address)
        for i in range(ow * oh):
            o.pixels[i] = res.unsafe_offset(i).unsafe_load()
        return o^

    def _pass(mut self, dset: Int, pipeline: Int, sw: Int, sh: Int, iw: Int, ih: Int, ow: Int, oh: Int,
              gx: Int, gy: Int):
        """Dispatch over a gx x gy invocation grid (the input pixels, or the outputs)."""
        var push: List[UInt32] = [UInt32(sw), UInt32(sh), UInt32(iw), UInt32(ih), UInt32(ow), UInt32(oh)]
        self.vk.dispatch(0, pipeline, dset, push, (gx + _GROUP - 1) // _GROUP, (gy + _GROUP - 1) // _GROUP)
