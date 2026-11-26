"""Area E (`sota:E`): deep opacity shadow map (DOM) for strand hair.

A hair volume lit by the sun is not opaque: light reaching a strand deep inside the groom has crossed
many fibres. The VSM / RT sun shadow treats every ribbon as an opaque occluder, so everything below the
first layer goes black. A deep opacity map (Yuksel & Keyser, "Deep Opacity Maps", Eurographics 2008;
after Kim & Neumann, "Opacity Shadow Maps", EGSR 2001) stores, per light-space texel, the depth `d0` of the
first strand and the fibre coverage accumulated in `SOTA_E_DOM_LAYERS` slabs of `spacing` metres behind
it. A hair pixel at depth `d` then sees transmittance `T = exp(-k * coverage(d - d0))`.

GPU passes (compute, shared AAA layout, data in a region appended to `BIND_AAA_POST_SCRATCH`):
  clear    depth + layer words = 0
  depth    every strand segment, `SOTA_E_DOM_SAMPLES` samples: atomic max of the inverted 24-bit depth
           (max of inverted = nearest to the light)
  opacity  same samples: fixed-point fibre coverage (`width * step / texel^2`, x4096) added atomically to
           the slab `floor((d - d0) / spacing)` (the last slab takes everything behind)
  shade    `emit_sota_e_dom_transmittance` in the deferred pass: piecewise-linear cumulative coverage across
           the slabs, area-filtered over a 3x3-texel box (strands are thinner than a texel, so the local
           density sets the shadow); hair pixels inside the map use `T` as their sun visibility.

Integer atomics make the GPU map independent of thread order, so it matches `sota_e_dom_build_cpu`.
Limit: on hair pixels the map replaces the scene sun shadow (other occluders, e.g. the head, do not darken
the hair there). Strands are static in world space (`set_strands`); re-upload them when the groom moves.

cvar: `r.sota_e.hair_dom` = `SotaEHairShadow.enabled` (default on); env `FLINT_SOTA_E_HAIR_DOM=0` turns it off.
"""
from std.collections import List
from std.ffi import external_call
from std.math import exp, floor, max, min, sqrt
from flint.core.math import Vec3
from flint.platform.os import env_value
from flint.platform.vulkan import (
    VulkanGpu,
    put32,
    put_float,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    PIPELINE_COMPUTE,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_GLOBALS, BIND_AAA_POST_SCRATCH

comptime SOTA_E_DOM_G: Int = 3072          # globals header (words 3072 .. 3099 are free in the v5 layout)
comptime SOTA_E_DOM_G_ACTIVE: Int = 0
comptime SOTA_E_DOM_G_RES: Int = 1
comptime SOTA_E_DOM_G_BASE: Int = 2       # word offset of the map in post-scratch
comptime SOTA_E_DOM_G_NPTS: Int = 3
comptime SOTA_E_DOM_G_O: Int = 4          # light-frame origin (3)
comptime SOTA_E_DOM_G_R: Int = 7          # right axis (3)
comptime SOTA_E_DOM_G_U: Int = 10         # up axis (3)
comptime SOTA_E_DOM_G_F: Int = 13         # light travel direction (3)
comptime SOTA_E_DOM_G_HX: Int = 16
comptime SOTA_E_DOM_G_HY: Int = 17
comptime SOTA_E_DOM_G_DMIN: Int = 18
comptime SOTA_E_DOM_G_DRANGE: Int = 19
comptime SOTA_E_DOM_G_SPACING: Int = 20
comptime SOTA_E_DOM_G_K: Int = 21
comptime SOTA_E_DOM_G_WIDTH: Int = 22
comptime SOTA_E_DOM_G_TEXEL: Int = 23
comptime SOTA_E_DOM_G_BIAS: Int = 24

comptime SOTA_E_DOM_LAYERS: Int = 4
comptime SOTA_E_DOM_SAMPLES: Int = 8
comptime SOTA_E_DOM_FIXED: Float32 = 4096.0
comptime SOTA_E_DOM_DEPTH_MAX: Float32 = 16777215.0
comptime SOTA_E_DOM_DEFAULT_RES: Int = 128
comptime SOTA_E_DOM_MAX_POINTS: Int = 65536


def sota_e_dom_words(res: Int) -> Int:
    """Post-scratch words the map needs: depth + layers + point list."""
    return res * res * (1 + SOTA_E_DOM_LAYERS) + SOTA_E_DOM_MAX_POINTS * 4


# ---------------------------------------------------------------------------------------------
# Light frame + CPU reference
# ---------------------------------------------------------------------------------------------
@fieldwise_init
struct SotaEDomFrame(Copyable, Movable):
    var res: Int
    var o: Vec3
    var r: Vec3
    var u: Vec3
    var f: Vec3
    var hx: Float32
    var hy: Float32
    var dmin: Float32
    var drange: Float32
    var spacing: Float32
    var k: Float32
    var width: Float32
    var texel: Float32
    var bias: Float32


def sota_e_dom_frame(points: List[Float32], sun_dir: Vec3, res: Int, width: Float32, spacing: Float32, k: Float32) -> SotaEDomFrame:
    """Orthographic light frame fitted to the strand points (x, y, z, continues) along `-sun_dir`."""
    var f = (sun_dir * Float32(-1.0)).normalized()
    var helper = Vec3(0.0, 1.0, 0.0) if abs(f.y) < Float32(0.95) else Vec3(1.0, 0.0, 0.0)
    var r = helper.cross(f).normalized()
    var u = f.cross(r).normalized()
    var n = len(points) // 4
    var lo = Vec3(1e30, 1e30, 1e30)
    var hi = Vec3(-1e30, -1e30, -1e30)
    for i in range(n):
        var p = Vec3(points[i * 4], points[i * 4 + 1], points[i * 4 + 2])
        lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
        hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
    var o = (lo + hi) * Float32(0.5)
    var hx = Float32(1e-4)
    var hy = Float32(1e-4)
    var d0 = Float32(1e30)
    var d1 = Float32(-1e30)
    for c in range(8):
        var p = Vec3(hi.x if (c & 1) != 0 else lo.x, hi.y if (c & 2) != 0 else lo.y, hi.z if (c & 4) != 0 else lo.z)
        var q = p - o
        hx = max(hx, abs(q.dot(r)))
        hy = max(hy, abs(q.dot(u)))
        d0 = min(d0, q.dot(f))
        d1 = max(d1, q.dot(f))
    var h = max(hx, hy) * (Float32(1.0) + Float32(2.0) / Float32(res)) + width
    var texel = Float32(2.0) * h / Float32(res)
    return SotaEDomFrame(res, o, r, u, f, h, h, d0 - width, max(Float32(1e-4), d1 - d0 + Float32(2.0) * width), spacing, k, width, texel, spacing * Float32(0.25))


@always_inline
def _dom_texel(fr: SotaEDomFrame, p: Vec3) -> Tuple[Int, Float32]:
    """(texel index or -1, light depth) of a world point."""
    var q = p - fr.o
    var uu = (q.dot(fr.r) / fr.hx) * Float32(0.5) + Float32(0.5)
    var vv = (q.dot(fr.u) / fr.hy) * Float32(0.5) + Float32(0.5)
    var d = q.dot(fr.f)
    if uu < Float32(0.0) or uu >= Float32(1.0) or vv < Float32(0.0) or vv >= Float32(1.0):
        return (-1, d)
    var tx = min(fr.res - 1, Int(uu * Float32(fr.res)))
    var ty = min(fr.res - 1, Int(vv * Float32(fr.res)))
    return (ty * fr.res + tx, d)


@always_inline
def _dom_depth_q(fr: SotaEDomFrame, d: Float32) -> UInt32:
    var dn = max(Float32(0.0), min(Float32(1.0), (d - fr.dmin) / fr.drange))
    return UInt32(SOTA_E_DOM_DEPTH_MAX) - UInt32(dn * SOTA_E_DOM_DEPTH_MAX)


@always_inline
def _dom_depth_of(fr: SotaEDomFrame, stored: UInt32) -> Float32:
    return fr.dmin + (SOTA_E_DOM_DEPTH_MAX - Float32(stored)) / SOTA_E_DOM_DEPTH_MAX * fr.drange


def sota_e_dom_build_cpu(fr: SotaEDomFrame, points: List[Float32]) -> Tuple[List[UInt32], List[UInt32]]:
    """(depth words, layer words [layer * res^2 + texel]) exactly as the GPU passes build them."""
    var nt = fr.res * fr.res
    var depth = List[UInt32](length=nt, fill=0)
    var layers = List[UInt32](length=nt * SOTA_E_DOM_LAYERS, fill=0)
    var n = len(points) // 4
    for pass_i in range(2):
        for i in range(n - 1):
            if points[i * 4 + 3] < Float32(0.5):
                continue
            var a = Vec3(points[i * 4], points[i * 4 + 1], points[i * 4 + 2])
            var b = Vec3(points[i * 4 + 4], points[i * 4 + 5], points[i * 4 + 6])
            var seg = sqrt((b - a).length_squared())
            var cov = fr.width * (seg / Float32(SOTA_E_DOM_SAMPLES)) / (fr.texel * fr.texel)
            var add = UInt32(min(Float32(65535.0), cov * SOTA_E_DOM_FIXED + Float32(0.5)))
            for j in range(SOTA_E_DOM_SAMPLES):
                var t = (Float32(j) + Float32(0.5)) / Float32(SOTA_E_DOM_SAMPLES)
                var p = a + (b - a) * t
                var td = _dom_texel(fr, p)
                if td[0] < 0:
                    continue
                if pass_i == 0:
                    depth[td[0]] = max(depth[td[0]], _dom_depth_q(fr, td[1]))
                else:
                    var d0 = _dom_depth_of(fr, depth[td[0]])
                    var lf = max(Float32(0.0), min(Float32(SOTA_E_DOM_LAYERS - 1), floor((td[1] - d0) / fr.spacing)))
                    var li = Int(lf)
                    layers[li * nt + td[0]] += add
    return (depth^, layers^)


def _dom_texel_cum(fr: SotaEDomFrame, depth: List[UInt32], layers: List[UInt32], tex: Int, d: Float32) -> Float32:
    """Fixed-point fibre coverage between the light and depth `d` in one texel (0 for an empty texel)."""
    var s0 = depth[tex]
    if s0 == 0:
        return Float32(0.0)
    var drel = (d - _dom_depth_of(fr, s0) - fr.bias) / fr.spacing
    var cum = Float32(0.0)
    var nt = fr.res * fr.res
    for k in range(SOTA_E_DOM_LAYERS):
        var w = max(Float32(0.0), min(Float32(1.0), drel - Float32(k)))
        cum += Float32(layers[k * nt + tex]) * w
    return cum


@always_inline
def _tap_w(k: Int, a: Float32) -> Float32:
    """Weights of the 4 taps of a 3-texel box filter, bilinearly shifted by `a`."""
    if k == 0:
        return (Float32(1.0) - a) / Float32(3.0)
    if k == 3:
        return a / Float32(3.0)
    return Float32(1.0) / Float32(3.0)


def sota_e_dom_transmittance_cpu(fr: SotaEDomFrame, depth: List[UInt32], layers: List[UInt32], p: Vec3) -> Float32:
    """Sun transmittance at world point `p` (1 outside the map). The coverage (not T) is area-filtered
    over a 3x3-texel box (4x4 taps, bilinear), i.e. the opacity-map density model: strands are thinner
    than a texel, so the density around a point, not the single texel it falls in, sets its shadow."""
    var q = p - fr.o
    var uu = (q.dot(fr.r) / fr.hx) * Float32(0.5) + Float32(0.5)
    var vv = (q.dot(fr.u) / fr.hy) * Float32(0.5) + Float32(0.5)
    var d = q.dot(fr.f)
    if uu < Float32(0.0) or uu >= Float32(1.0) or vv < Float32(0.0) or vv >= Float32(1.0):
        return Float32(1.0)
    var fx = uu * Float32(fr.res) - Float32(0.5)
    var fy = vv * Float32(fr.res) - Float32(0.5)
    var ix = Int(floor(fx))
    var iy = Int(floor(fy))
    var ax = fx - Float32(ix)
    var ay = fy - Float32(iy)
    var cum = Float32(0.0)
    for j in range(4):
        var ty = max(0, min(fr.res - 1, iy - 1 + j))
        for i in range(4):
            var tx = max(0, min(fr.res - 1, ix - 1 + i))
            cum += _tap_w(i, ax) * _tap_w(j, ay) * _dom_texel_cum(fr, depth, layers, ty * fr.res + tx, d)
    return exp(-fr.k * cum / SOTA_E_DOM_FIXED)


def sota_e_strands_to_points(strands: List[List[Vec3]]) -> List[Float32]:
    """Packs strands as (x, y, z, continues) points; `continues` = 1 when a segment joins the next point."""
    var out = List[Float32]()
    for s in strands:
        for k in range(len(s)):
            out.append(s[k].x)
            out.append(s[k].y)
            out.append(s[k].z)
            out.append(Float32(1.0) if k + 1 < len(s) else Float32(0.0))
    return out^


# ---------------------------------------------------------------------------------------------
# SPIR-V
# ---------------------------------------------------------------------------------------------
def _g(mut s: AaaSpirvModule, b_globals: UInt32, w: Int) -> UInt32:
    return s.load_float(b_globals, s.uint(UInt32(SOTA_E_DOM_G + w)))


def _gw(mut s: AaaSpirvModule, b_globals: UInt32, w: Int) -> UInt32:
    return s.load_word(b_globals, s.uint(UInt32(SOTA_E_DOM_G + w)))


struct _DomProj(Copyable, Movable):
    var uu: UInt32
    var vv: UInt32
    var d: UInt32
    var inside: UInt32

    def __init__(out self, uu: UInt32, vv: UInt32, d: UInt32, inside: UInt32):
        self.uu = uu
        self.vv = vv
        self.d = d
        self.inside = inside


def _emit_dom_project(mut s: AaaSpirvModule, b_globals: UInt32, px: UInt32, py: UInt32, pz: UInt32) -> _DomProj:
    var qx = s.fsub(px, _g(s, b_globals, SOTA_E_DOM_G_O))
    var qy = s.fsub(py, _g(s, b_globals, SOTA_E_DOM_G_O + 1))
    var qz = s.fsub(pz, _g(s, b_globals, SOTA_E_DOM_G_O + 2))
    var dr = s.fma(qz, _g(s, b_globals, SOTA_E_DOM_G_R + 2), s.fma(qy, _g(s, b_globals, SOTA_E_DOM_G_R + 1), s.fmul(qx, _g(s, b_globals, SOTA_E_DOM_G_R))))
    var du = s.fma(qz, _g(s, b_globals, SOTA_E_DOM_G_U + 2), s.fma(qy, _g(s, b_globals, SOTA_E_DOM_G_U + 1), s.fmul(qx, _g(s, b_globals, SOTA_E_DOM_G_U))))
    var d = s.fma(qz, _g(s, b_globals, SOTA_E_DOM_G_F + 2), s.fma(qy, _g(s, b_globals, SOTA_E_DOM_G_F + 1), s.fmul(qx, _g(s, b_globals, SOTA_E_DOM_G_F))))
    var uu = s.fadd(s.fmul(s.fdiv(dr, _g(s, b_globals, SOTA_E_DOM_G_HX)), s.float(0.5)), s.float(0.5))
    var vv = s.fadd(s.fmul(s.fdiv(du, _g(s, b_globals, SOTA_E_DOM_G_HY)), s.float(0.5)), s.float(0.5))
    var inside = s.logical_and(
        s.logical_and(s.fge(uu, s.float(0.0)), s.flt(uu, s.float(1.0))),
        s.logical_and(s.fge(vv, s.float(0.0)), s.flt(vv, s.float(1.0))),
    )
    return _DomProj(uu, vv, d, inside)


def _emit_texel(mut s: AaaSpirvModule, uu: UInt32, vv: UInt32, res: UInt32, res_f: UInt32) -> UInt32:
    var rm1 = s.isub(res, s.uint(1))
    var tx = s.umin(rm1, s.uint_from_float(s.nmax(s.float(0.0), s.fmul(uu, res_f))))
    var ty = s.umin(rm1, s.uint_from_float(s.nmax(s.float(0.0), s.fmul(vv, res_f))))
    return s.iadd(s.imul(ty, res), tx)


def _emit_depth_of(mut s: AaaSpirvModule, b_globals: UInt32, stored: UInt32) -> UInt32:
    var dn = s.fdiv(s.fsub(s.float(SOTA_E_DOM_DEPTH_MAX), s.float_from_uint(stored)), s.float(SOTA_E_DOM_DEPTH_MAX))
    return s.fma(dn, _g(s, b_globals, SOTA_E_DOM_G_DRANGE), _g(s, b_globals, SOTA_E_DOM_G_DMIN))


def sota_e_dom_clear_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var i = s.invocation_axis(0)
    var res = _gw(s, b_globals, SOTA_E_DOM_G_RES)
    var total = s.imul(s.imul(res, res), s.uint(UInt32(1 + SOTA_E_DOM_LAYERS)))
    s.return_if(s.uge(i, total))
    s.store_word(b_post, s.iadd(_gw(s, b_globals, SOTA_E_DOM_G_BASE), i), s.uint(0))
    return s.finish()


def sota_e_dom_splat_shader(opacity_pass: Bool) -> List[UInt32]:
    """Grid: X = point / 64, Y = sample (`SOTA_E_DOM_SAMPLES`)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var i = s.invocation_axis(0)
    var j = s.invocation_axis(1)
    var npts = _gw(s, b_globals, SOTA_E_DOM_G_NPTS)
    s.return_if(s.uge(s.iadd(i, s.uint(1)), npts))
    var res = _gw(s, b_globals, SOTA_E_DOM_G_RES)
    var res_f = s.float_from_uint(res)
    var nt = s.imul(res, res)
    var base = _gw(s, b_globals, SOTA_E_DOM_G_BASE)
    var pbase = s.iadd(s.iadd(base, s.imul(nt, s.uint(UInt32(1 + SOTA_E_DOM_LAYERS)))), s.imul(i, s.uint(4)))
    s.return_if(s.flt(s.load_float(b_post, s.iadd(pbase, s.uint(3))), s.float(0.5)))
    var ax = s.load_float(b_post, pbase)
    var ay = s.load_float(b_post, s.iadd(pbase, s.uint(1)))
    var az = s.load_float(b_post, s.iadd(pbase, s.uint(2)))
    var bx = s.load_float(b_post, s.iadd(pbase, s.uint(4)))
    var by = s.load_float(b_post, s.iadd(pbase, s.uint(5)))
    var bz = s.load_float(b_post, s.iadd(pbase, s.uint(6)))
    var t = s.fdiv(s.fadd(s.float_from_uint(j), s.float(0.5)), s.float(Float32(SOTA_E_DOM_SAMPLES)))
    var px = s.fadd(ax, s.fmul(s.fsub(bx, ax), t))
    var py = s.fadd(ay, s.fmul(s.fsub(by, ay), t))
    var pz = s.fadd(az, s.fmul(s.fsub(bz, az), t))
    var pr = _emit_dom_project(s, b_globals, px, py, pz)
    s.return_if(s.logical_not(pr.inside))
    var tex = _emit_texel(s, pr.uu, pr.vv, res, res_f)
    if not opacity_pass:
        var dn = s.nclamp(
            s.fdiv(s.fsub(pr.d, _g(s, b_globals, SOTA_E_DOM_G_DMIN)), _g(s, b_globals, SOTA_E_DOM_G_DRANGE)), s.float(0.0), s.float(1.0)
        )
        var q = s.isub(s.uint(UInt32(SOTA_E_DOM_DEPTH_MAX)), s.uint_from_float(s.fmul(dn, s.float(SOTA_E_DOM_DEPTH_MAX))))
        _ = s.atomic_umax_word(b_post, s.iadd(base, tex), q)
    else:
        var dx = s.fsub(bx, ax)
        var dy = s.fsub(by, ay)
        var dz = s.fsub(bz, az)
        var seg = s.fsqrt(s.fma(dx, dx, s.fma(dy, dy, s.fmul(dz, dz))))
        var texel = _g(s, b_globals, SOTA_E_DOM_G_TEXEL)
        var cov = s.fdiv(
            s.fmul(_g(s, b_globals, SOTA_E_DOM_G_WIDTH), s.fdiv(seg, s.float(Float32(SOTA_E_DOM_SAMPLES)))), s.fmul(texel, texel)
        )
        var add = s.uint_from_float(s.nmin(s.float(65535.0), s.fadd(s.fmul(cov, s.float(SOTA_E_DOM_FIXED)), s.float(0.5))))
        var d0 = _emit_depth_of(s, b_globals, s.load_word(b_post, s.iadd(base, tex)))
        var lf = s.nclamp(
            s.ffloor(s.fdiv(s.fsub(pr.d, d0), _g(s, b_globals, SOTA_E_DOM_G_SPACING))), s.float(0.0), s.float(Float32(SOTA_E_DOM_LAYERS - 1))
        )
        var li = s.uint_from_float(lf)
        _ = s.atomic_add_word(b_post, s.iadd(s.iadd(base, nt), s.iadd(s.imul(li, nt), tex)), add)
    return s.finish()


def _emit_texel_cum(mut s: AaaSpirvModule, b_post: UInt32, b_globals: UInt32, base: UInt32, nt: UInt32, tex: UInt32, d: UInt32) -> UInt32:
    var s0 = s.load_word(b_post, s.iadd(base, tex))
    var drel = s.fdiv(
        s.fsub(s.fsub(d, _emit_depth_of(s, b_globals, s0)), _g(s, b_globals, SOTA_E_DOM_G_BIAS)), _g(s, b_globals, SOTA_E_DOM_G_SPACING)
    )
    var cum = s.float(0.0)
    var lbase = s.iadd(base, nt)
    for k in range(SOTA_E_DOM_LAYERS):
        var w = s.nclamp(s.fsub(drel, s.float(Float32(k))), s.float(0.0), s.float(1.0))
        var lv = s.float_from_uint(s.load_word(b_post, s.iadd(lbase, s.iadd(s.imul(s.uint(UInt32(k)), nt), tex))))
        cum = s.fma(lv, w, cum)
    return s.select_f(s.ieq(s0, s.uint(0)), s.float(0.0), cum)


def _emit_tap_w(mut s: AaaSpirvModule, k: Int, a: UInt32) -> UInt32:
    var third = s.float(Float32(1.0) / Float32(3.0))
    if k == 0:
        return s.fmul(s.fsub(s.float(1.0), a), third)
    if k == 3:
        return s.fmul(a, third)
    return third


def emit_sota_e_dom_transmittance(mut s: AaaSpirvModule, b_globals: UInt32, b_post: UInt32, wx: UInt32, wy: UInt32, wz: UInt32) -> Tuple[UInt32, UInt32]:
    """(transmittance, inside-the-map) at a world point; mirrors `sota_e_dom_transmittance_cpu`. The loads
    read clamped texels, so they are safe for any pixel; the caller selects on `inside`."""
    var pr = _emit_dom_project(s, b_globals, wx, wy, wz)
    var res = _gw(s, b_globals, SOTA_E_DOM_G_RES)
    var res_f = s.float_from_uint(res)
    var nt = s.imul(res, res)
    var base = _gw(s, b_globals, SOTA_E_DOM_G_BASE)
    var rm1_f = s.fsub(res_f, s.float(1.0))
    var fx = s.fsub(s.fmul(s.nclamp(pr.uu, s.float(0.0), s.float(1.0)), res_f), s.float(0.5))
    var fy = s.fsub(s.fmul(s.nclamp(pr.vv, s.float(0.0), s.float(1.0)), res_f), s.float(0.5))
    var flx = s.ffloor(fx)
    var fly = s.ffloor(fy)
    var ax = s.fsub(fx, flx)
    var ay = s.fsub(fy, fly)
    var cum = s.float(0.0)
    for j in range(4):
        var ty = s.uint_from_float(s.nclamp(s.fadd(fly, s.float(Float32(j - 1))), s.float(0.0), rm1_f))
        var wyj = _emit_tap_w(s, j, ay)
        for i in range(4):
            var tx = s.uint_from_float(s.nclamp(s.fadd(flx, s.float(Float32(i - 1))), s.float(0.0), rm1_f))
            var c = _emit_texel_cum(s, b_post, b_globals, base, nt, s.iadd(s.imul(ty, res), tx), pr.d)
            cum = s.fma(s.fmul(_emit_tap_w(s, i, ax), wyj), c, cum)
    # exp(-k * cum / FIXED) = exp2(-k * cum * log2(e) / FIXED)
    var t = s.exp2(s.fmul(s.fneg(_g(s, b_globals, SOTA_E_DOM_G_K)), s.fmul(cum, s.float(Float32(1.44269504) / SOTA_E_DOM_FIXED))))
    var active = s.ine(_gw(s, b_globals, SOTA_E_DOM_G_ACTIVE), s.uint(0))
    return (t, s.logical_and(active, pr.inside))


# ---------------------------------------------------------------------------------------------
# Renderer-side state
# ---------------------------------------------------------------------------------------------
struct SotaEHairShadow(Movable):
    var enabled: Bool          # cvar r.sota_e.hair_dom
    var ready: Bool            # pipelines + scratch region exist
    var res: Int
    var spacing: Float32       # slab thickness (m)
    var extinction: Float32    # optical depth per unit fibre coverage
    var width: Float32         # strand width (m) used for coverage
    var base_word: Int
    var points: List[Float32]
    var pipe_clear: Int
    var pipe_depth: Int
    var pipe_opacity: Int

    def __init__(out self):
        self.enabled = env_value("FLINT_SOTA_E_HAIR_DOM") != "0"
        self.ready = False
        self.res = SOTA_E_DOM_DEFAULT_RES
        self.spacing = Float32(0.008)
        self.extinction = Float32(1.0)
        self.width = Float32(0.0016)
        self.base_word = 0
        self.points = List[Float32]()
        self.pipe_clear = 0
        self.pipe_depth = 0
        self.pipe_opacity = 0

    def init(mut self, mut vk: VulkanGpu, pipe_layout: Int, base_word: Int) -> Bool:
        self.base_word = base_word
        self.pipe_clear = vk.create_aaa_compute_pipeline(pipe_layout, sota_e_dom_clear_shader())
        self.pipe_depth = vk.create_aaa_compute_pipeline(pipe_layout, sota_e_dom_splat_shader(False))
        self.pipe_opacity = vk.create_aaa_compute_pipeline(pipe_layout, sota_e_dom_splat_shader(True))
        self.ready = self.pipe_clear != 0 and self.pipe_depth != 0 and self.pipe_opacity != 0
        return self.ready

    def set_strands(mut self, strands: List[List[Vec3]], width: Float32, scratch_addr: Int):
        """World-space strands (root to tip). Uploads at once: call between frames (static grooms)."""
        self.points = sota_e_strands_to_points(strands)
        if len(self.points) // 4 > SOTA_E_DOM_MAX_POINTS:
            self.points.resize(SOTA_E_DOM_MAX_POINTS * 4, Float32(0.0))
            self.points[len(self.points) - 1] = Float32(0.0)
        self.width = width
        if scratch_addr != 0 and len(self.points) > 0:
            var dst = scratch_addr + (self.base_word + self.res * self.res * (1 + SOTA_E_DOM_LAYERS)) * 4
            _ = external_call["memcpy", Int](dst, Int(self.points.unsafe_ptr()), len(self.points) * 4)

    def clear_strands(mut self):
        self.points = List[Float32]()

    def active(self) -> Bool:
        return self.enabled and self.ready and len(self.points) >= 8

    def frame(self, sun_dir: Vec3) -> SotaEDomFrame:
        return sota_e_dom_frame(self.points, sun_dir, self.res, self.width, self.spacing, self.extinction)

    def write_globals(self, g_addr: Int, sun_dir: Vec3):
        var g = g_addr + SOTA_E_DOM_G * 4
        put32(g + SOTA_E_DOM_G_RES * 4, UInt32(self.res))        # keep the (unselected) shader loads in range
        put32(g + SOTA_E_DOM_G_BASE * 4, UInt32(self.base_word))
        if not self.active():
            put32(g + SOTA_E_DOM_G_ACTIVE * 4, 0)
            return
        var fr = self.frame(sun_dir)
        put32(g + SOTA_E_DOM_G_ACTIVE * 4, 1)
        put32(g + SOTA_E_DOM_G_RES * 4, UInt32(self.res))
        put32(g + SOTA_E_DOM_G_BASE * 4, UInt32(self.base_word))
        put32(g + SOTA_E_DOM_G_NPTS * 4, UInt32(len(self.points) // 4))
        var vs = [fr.o, fr.r, fr.u, fr.f]
        var offs = [SOTA_E_DOM_G_O, SOTA_E_DOM_G_R, SOTA_E_DOM_G_U, SOTA_E_DOM_G_F]
        for k in range(4):
            put_float(g + offs[k] * 4, vs[k].x)
            put_float(g + (offs[k] + 1) * 4, vs[k].y)
            put_float(g + (offs[k] + 2) * 4, vs[k].z)
        put_float(g + SOTA_E_DOM_G_HX * 4, fr.hx)
        put_float(g + SOTA_E_DOM_G_HY * 4, fr.hy)
        put_float(g + SOTA_E_DOM_G_DMIN * 4, fr.dmin)
        put_float(g + SOTA_E_DOM_G_DRANGE * 4, fr.drange)
        put_float(g + SOTA_E_DOM_G_SPACING * 4, fr.spacing)
        put_float(g + SOTA_E_DOM_G_K * 4, fr.k)
        put_float(g + SOTA_E_DOM_G_WIDTH * 4, fr.width)
        put_float(g + SOTA_E_DOM_G_TEXEL * 4, fr.texel)
        put_float(g + SOTA_E_DOM_G_BIAS * 4, fr.bias)

    def record(self, mut vk: VulkanGpu, slot: Int, pipe_layout: Int, desc_set: Int):
        """Clear + depth + opacity passes; leaves a compute -> compute barrier for the deferred pass."""
        if not self.active():
            return
        var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
        var n = len(self.points) // 4
        var words = self.res * self.res * (1 + SOTA_E_DOM_LAYERS)
        # last frame's deferred pass may still read the map (same queue, earlier submission)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ), PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE))
        vk.dispatch_aaa(slot, self.pipe_clear, pipe_layout, desc_set, push, (words + 63) // 64, 1)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        vk.dispatch_aaa(slot, self.pipe_depth, pipe_layout, desc_set, push, (n + 63) // 64, SOTA_E_DOM_SAMPLES)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        vk.dispatch_aaa(slot, self.pipe_opacity, pipe_layout, desc_set, push, (n + 63) // 64, SOTA_E_DOM_SAMPLES)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ))

    def shutdown(mut self, mut vk: VulkanGpu):
        if self.pipe_clear != 0:
            vk.destroy_pipeline(self.pipe_clear)
        if self.pipe_depth != 0:
            vk.destroy_pipeline(self.pipe_depth)
        if self.pipe_opacity != 0:
            vk.destroy_pipeline(self.pipe_opacity)
        self.pipe_clear = 0
        self.pipe_depth = 0
        self.pipe_opacity = 0
        self.ready = False
