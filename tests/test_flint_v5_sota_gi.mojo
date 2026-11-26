"""Area B render SOTA: world radiance cache, hybrid surface-cache GI and the real-time path tracer
(`test_flint_v5_sota_gi.mojo`).

1. CPU twins: SH9 projection / irradiance convolution, Chebyshev visibility, toroidal probe slots and
   SPIR-V validation of the new passes and of the hooked v5 shaders.
2. GPU (needs hardware ray queries, otherwise skipped): the Cornell box of `test_flint.mojo` rendered
   with every GI mode and compared by FLIP against two 512-spp CPU references (the v5 2-indirect-bounce
   one and a converged multi-bounce one), with GPU pass timings and before/after PNGs.

Environment: `FLINT_SOTA_GI_PNG_DIR` overrides where PNGs go (default `~/flint-models/_eval/render/B`);
`FLINT_SOTA_GI_QUICK=1` lowers the reference sample count for iteration;
`FLINT_SOTA_GI_ONLY=gi|mirror|di|svgf|lowres|ml` runs a single GPU suite.
"""
from std.ffi import external_call
from std.math import abs, cos, exp, log, max, min, sin, sqrt, tan
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import env_now, get_time, make_dirs, read_bytes
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_gi_svgf import aaa_deferred_shade_shader, surface_cache_update_shader
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.flip import flip_error_map, flip_heat_image, flip_mean, srgb_to_linear
from flint.render.spirv_validate import validate_spirv_module
from flint.render.sota_gi_world_cache import (
    SOTA_GI_MODE_HYBRID,
    SOTA_GI_MODE_LEGACY,
    SOTA_GI_MODE_PATH,
    SOTA_GI_MODE_WRC,
    sota_chebyshev_cpu,
    sota_fib_dir_cpu,
    sota_gi_resolve_shader,
    sota_gi_upsample_shader,
    sota_sh9_irradiance_over_pi_cpu,
    sota_sh9_project_cpu,
    sota_wrc_slot_cpu,
    sota_wrc_update_shader,
)
from flint.render.sota_gi_core import sota_gi_bayer_offset
from flint.render.sota_ml_denoise import SOTA_ML_GI_MODES_DEFAULT, sota_ml_default_weights_path


def _assert(cond: Bool, msg: String) raises:
    if not cond:
        print("  [FAIL] " + msg)
        raise Error(msg)


def _validate(words: List[UInt32], name: String) raises:
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _assert(False, name + " SPIR-V validation failed: " + rep.errors[0])


def _f3(x: Float64) -> String:
    if x < 0.0:
        return "-" + _f3(-x)
    var v = Int(x * 1000.0 + 0.5)
    var frac = String(v % 1000)
    while frac.byte_length() < 3:
        frac = "0" + frac
    return String(v // 1000) + "." + frac


# ---------------------------------------------------------------------------------------------
# 1. CPU twins
# ---------------------------------------------------------------------------------------------
def test_cpu_twins() raises:
    # Uniform radiance L = 0.7 -> E / pi = 0.7 for every normal (SH projection + convolution exact for band 0)
    var dirs = List[Vec3]()
    var rad = List[Vec3]()
    var n_dirs = 4096
    for i in range(n_dirs):
        dirs.append(sota_fib_dir_cpu(i, n_dirs, Float32(0.37), Float32(0.61)))
        rad.append(Vec3(0.7, 0.7, 0.7))
    var sh = sota_sh9_project_cpu(dirs, rad)
    for n in [Vec3(1, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0.577, 0.577, -0.577)]:
        var e = sota_sh9_irradiance_over_pi_cpu(sh, n.normalized())
        _assert(abs(e.x - Float32(0.7)) < Float32(0.01), "uniform SH irradiance " + String(e.x))

    # Upper-hemisphere sky L = 1 (y > 0): exact E/pi(up) = 1, E/pi(down) = 0, E/pi(horizontal) = 0.5
    var rad2 = List[Vec3]()
    for i in range(n_dirs):
        rad2.append(Vec3(1.0, 1.0, 1.0) if dirs[i].y > Float32(0.0) else Vec3(0.0, 0.0, 0.0))
    var sh2 = sota_sh9_project_cpu(dirs, rad2)
    var e_up = sota_sh9_irradiance_over_pi_cpu(sh2, Vec3(0.0, 1.0, 0.0))
    var e_dn = sota_sh9_irradiance_over_pi_cpu(sh2, Vec3(0.0, -1.0, 0.0))
    var e_hz = sota_sh9_irradiance_over_pi_cpu(sh2, Vec3(1.0, 0.0, 0.0))
    _assert(abs(e_up.x - Float32(1.0)) < Float32(0.07), "hemisphere up " + String(e_up.x))
    _assert(e_dn.x < Float32(0.07), "hemisphere down " + String(e_dn.x))
    _assert(abs(e_hz.x - Float32(0.5)) < Float32(0.03), "hemisphere horizon " + String(e_hz.x))

    # Chebyshev: in front of the mean = visible, far behind a tight distribution = occluded
    _assert(sota_chebyshev_cpu(1.0, 1.0, 0.5) == Float32(1.0), "chebyshev front")
    _assert(sota_chebyshev_cpu(1.0, 1.0001, 2.0) < Float32(0.001), "chebyshev behind")
    _assert(sota_chebyshev_cpu(1.0, 1.5, 1.2) > Float32(0.4), "chebyshev wide")

    # Toroidal slots: a window of 16 cells scrolled by one cell keeps 15 slots unchanged
    var base = 1 << 20
    var changed = 0
    for i in range(16):
        var old_cell = base + i
        var slot = sota_wrc_slot_cpu(old_cell, 16)
        var new_cell_for_slot = (base + 1) + ((slot + 16 - ((base + 1) % 16)) % 16)
        if new_cell_for_slot != old_cell:
            changed += 1
    _assert(changed == 1, "scroll should recycle exactly one slot, got " + String(changed))

    # SPIR-V of the new passes and of the hooked v5 shaders
    _validate(sota_wrc_update_shader(True), "sota_wrc_update_shader(rt)")
    _validate(sota_wrc_update_shader(False), "sota_wrc_update_shader(no rt)")
    _validate(sota_gi_resolve_shader(True), "sota_gi_resolve_shader(rt)")
    _validate(sota_gi_resolve_shader(False), "sota_gi_resolve_shader(no rt)")
    _validate(sota_gi_upsample_shader(), "sota_gi_upsample_shader")
    # Reduced-res resolve: the Bayer phases visit every pixel of a block once per scale^2 frames
    for sc in [1, 2, 4]:
        var seen = List[Int](length=sc * sc, fill=0)
        for k in range(sc * sc):
            var j = sota_gi_bayer_offset(sc, k)
            seen[j[1] * sc + j[0]] += 1
        for i in range(sc * sc):
            _assert(seen[i] == 1, "Bayer phase cover, scale " + String(sc))
    var j1 = sota_gi_bayer_offset(2, 1)
    _assert(j1[0] == 1 and j1[1] == 1, "Bayer phase 1 of 2x2 is the opposite corner")
    _validate(aaa_deferred_shade_shader(True), "aaa_deferred_shade_shader(rt) + sota hooks")
    _validate(surface_cache_update_shader(True), "surface_cache_update_shader(rt) + sota hooks")
    print("  [PASS] SOTA GI CPU twins (SH9 irradiance, Chebyshev, toroidal slots) + SPIR-V validation")


# ---------------------------------------------------------------------------------------------
# 2. Cornell box references (512 spp)
# ---------------------------------------------------------------------------------------------
def _cornell_intersect(ro: Vec3, rd: Vec3) -> Tuple[Bool, Float32, Vec3, Vec3, Int]:
    """Same analytic scene as `test_flint.mojo` (`mat: 0 white, 1 red left, 2 green right`)."""
    var best_t = Float32(1e9)
    var best_n = Vec3(0.0, 1.0, 0.0)
    var best_m = 0
    var hit = False
    if abs(rd.y) > Float32(1e-6):
        var t = (Float32(-2.0) - ro.y) / rd.y
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.x) <= Float32(2.0) and p.z >= Float32(0.0) and p.z <= Float32(4.0):
                best_t = t; best_n = Vec3(0.0, 1.0, 0.0); best_m = 0; hit = True
        var t2 = (Float32(2.0) - ro.y) / rd.y
        if t2 > Float32(0.002) and t2 < best_t:
            var p2 = ro + rd * t2
            if abs(p2.x) <= Float32(2.0) and p2.z >= Float32(0.0) and p2.z <= Float32(4.0):
                best_t = t2; best_n = Vec3(0.0, -1.0, 0.0); best_m = 0; hit = True
    if abs(rd.z) > Float32(1e-6):
        var t = (Float32(4.0) - ro.z) / rd.z
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.x) <= Float32(2.0) and abs(p.y) <= Float32(2.0):
                best_t = t; best_n = Vec3(0.0, 0.0, -1.0); best_m = 0; hit = True
    if abs(rd.x) > Float32(1e-6):
        var t = (Float32(-2.0) - ro.x) / rd.x
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.y) <= Float32(2.0) and p.z >= Float32(0.0) and p.z <= Float32(4.0):
                best_t = t; best_n = Vec3(1.0, 0.0, 0.0); best_m = 1; hit = True
        var t2 = (Float32(2.0) - ro.x) / rd.x
        if t2 > Float32(0.002) and t2 < best_t:
            var p2 = ro + rd * t2
            if abs(p2.y) <= Float32(2.0) and p2.z >= Float32(0.0) and p2.z <= Float32(4.0):
                best_t = t2; best_n = Vec3(-1.0, 0.0, 0.0); best_m = 2; hit = True
    var bmin = Vec3(-0.75, -2.0, 1.45)
    var bmax = Vec3(0.75, -0.20, 2.95)
    var inv_d = Vec3(
        Float32(1.0) / (rd.x if abs(rd.x) > Float32(1e-6) else Float32(1e-6)),
        Float32(1.0) / (rd.y if abs(rd.y) > Float32(1e-6) else Float32(1e-6)),
        Float32(1.0) / (rd.z if abs(rd.z) > Float32(1e-6) else Float32(1e-6)),
    )
    var tx1 = (bmin.x - ro.x) * inv_d.x; var tx2 = (bmax.x - ro.x) * inv_d.x
    var tmin = min(tx1, tx2); var tmax = max(tx1, tx2)
    var ty1 = (bmin.y - ro.y) * inv_d.y; var ty2 = (bmax.y - ro.y) * inv_d.y
    tmin = max(tmin, min(ty1, ty2)); tmax = min(tmax, max(ty1, ty2))
    var tz1 = (bmin.z - ro.z) * inv_d.z; var tz2 = (bmax.z - ro.z) * inv_d.z
    tmin = max(tmin, min(tz1, tz2)); tmax = min(tmax, max(tz1, tz2))
    if tmax >= max(Float32(0.002), tmin) and tmin < best_t and tmin > Float32(0.002):
        best_t = tmin
        var hp = ro + rd * tmin
        if abs(hp.x - bmin.x) < Float32(0.003): best_n = Vec3(-1.0, 0.0, 0.0)
        elif abs(hp.x - bmax.x) < Float32(0.003): best_n = Vec3(1.0, 0.0, 0.0)
        elif abs(hp.y - bmax.y) < Float32(0.003): best_n = Vec3(0.0, 1.0, 0.0)
        elif abs(hp.z - bmin.z) < Float32(0.003): best_n = Vec3(0.0, 0.0, -1.0)
        else: best_n = Vec3(0.0, 0.0, 1.0)
        best_m = 0
        hit = True
    return (hit, best_t, ro + rd * best_t, best_n, best_m)


def _albedo(m: Int) -> Vec3:
    if m == 0:
        return Vec3(0.76, 0.76, 0.76)
    if m == 1:
        return Vec3(0.80, 0.08, 0.08)
    return Vec3(0.08, 0.78, 0.12)


def _pcg(mut st: UInt32) -> Float32:
    st = st * UInt32(747796405) + UInt32(2891336453)
    var w = ((st >> ((st >> 28) + 4)) ^ st) * UInt32(277803737)
    w = (w >> 22) ^ w
    return Float32(w & 0xFFFFFF) / Float32(16777216.0)


def _cos_dir(n: Vec3, u1: Float32, u2: Float32) -> Vec3:
    var r = sqrt(u1)
    var phi = Float32(6.2831853) * u2
    var up = Vec3(1.0, 0.0, 0.0) if abs(n.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
    var t = up.cross(n).normalized()
    var b = n.cross(t)
    return (t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(Float32(0.0), Float32(1.0) - u1))).normalized()


def _direct(p: Vec3, n: Vec3, alb: Vec3, sun_dir: Vec3, sun_rad: Vec3) -> Vec3:
    var ndl = max(Float32(0.0), n.dot(sun_dir))
    if ndl <= Float32(0.0):
        return Vec3(0.0, 0.0, 0.0)
    var sh = _cornell_intersect(p + n * Float32(0.004), sun_dir)
    if sh[0]:
        return Vec3(0.0, 0.0, 0.0)
    return Vec3(alb.x * sun_rad.x, alb.y * sun_rad.y, alb.z * sun_rad.z) * (ndl * Float32(0.318309886))


def _to_srgb(c: Vec3) -> UInt32:
    var out = UInt32(0xFF000000)
    var ch = [c.x, c.y, c.z]
    for i in range(3):
        var cl = max(Float32(0.0), min(Float32(1.0), ch[i]))
        var sr = cl * Float32(12.92) if cl <= Float32(0.0031308) else (Float32(1.055) * exp(log(cl) / Float32(2.4)) - Float32(0.055))
        out |= UInt32(max(0, min(255, Int(sr * Float32(255.0) + Float32(0.5))))) << UInt32(16 - 8 * i)
    return out


comptime _MIRROR_F0: Float32 = 0.9


def _is_mirror(mirror_floor: Bool, hit: Tuple[Bool, Float32, Vec3, Vec3, Int]) -> Bool:
    return mirror_floor and hit[0] and hit[3].y > Float32(0.5) and hit[2].y < Float32(-1.99)


def _schlick(f0: Float32, cos_t: Float32) -> Float32:
    var m = Float32(1.0) - max(Float32(0.0), min(Float32(1.0), cos_t))
    return f0 + (Float32(1.0) - f0) * (m * m * m * m * m)


def _reflect(d: Vec3, n: Vec3) -> Vec3:
    return d - n * (Float32(2.0) * d.dot(n))


def _primary_dir(px: Int, py: Int, w: Int, h: Int) -> Vec3:
    var tan_y = tan(Float32(50.0 * 3.14159265 / 360.0))
    var tan_x = tan_y * (Float32(w) / Float32(h))
    var ndc_y = Float32(1.0) - ((Float32(py) + Float32(0.5)) / Float32(h)) * Float32(2.0)
    var ndc_x = ((Float32(px) + Float32(0.5)) / Float32(w)) * Float32(2.0) - Float32(1.0)
    return Vec3(ndc_x * tan_x, ndc_y * tan_y, Float32(1.0)).normalized()


def render_cornell_reference(
    w: Int, h: Int, spp: Int, sun_dir: Vec3, sun_rad: Vec3, indirect_bounces: Int, mirror_floor: Bool = False
) -> List[UInt32]:
    """Unbiased diffuse path tracer with NEE of the point sun, `indirect_bounces` bounces after the primary
    hit (2 = the v5 reference truncation; 12 = converged for these albedos to < 0.5% energy).
    `mirror_floor`: the floor is a perfect metal mirror (Schlick F0 0.9) for camera and indirect rays."""
    var out = List[UInt32](length=w * h, fill=0xFF000000)
    var cam = Vec3(0.0, 0.0, -3.5)
    for py in range(h):
        for px in range(w):
            var rd = _primary_dir(px, py, w, h)
            var ro = cam
            var tp0 = Vec3(1.0, 1.0, 1.0)
            var h0 = _cornell_intersect(ro, rd)
            var depth = 0
            while _is_mirror(mirror_floor, h0) and depth < 4:
                tp0 = tp0 * _schlick(_MIRROR_F0, -rd.dot(h0[3]))
                ro = h0[2] + h0[3] * Float32(0.004)
                rd = _reflect(rd, h0[3])
                h0 = _cornell_intersect(ro, rd)
                depth += 1
            if not h0[0] or _is_mirror(mirror_floor, h0):
                continue
            var alb0 = _albedo(h0[4])
            var l_dir = _direct(h0[2], h0[3], alb0, sun_dir, sun_rad)
            var st = UInt32((py * w + px) * 9781 + 17)
            var gi = Vec3(0.0, 0.0, 0.0)
            for s_i in range(spp):
                var p = h0[2]
                var n = h0[3]
                var tp = Vec3(1.0, 1.0, 1.0)
                var acc = Vec3(0.0, 0.0, 0.0)
                var spec_d = Vec3(0.0, 0.0, 0.0)
                var have_spec = False
                for b in range(indirect_bounces):
                    var d = spec_d
                    if not have_spec:
                        var u1 = _pcg(st)
                        if b == 0:
                            u1 = (Float32(s_i) + u1) / Float32(spp)
                        var u2 = _pcg(st)
                        d = _cos_dir(n, u1, u2)
                    have_spec = False
                    var hh = _cornell_intersect(p + n * Float32(0.005), d)
                    if not hh[0]:
                        break
                    if _is_mirror(mirror_floor, hh):
                        tp = tp * _schlick(_MIRROR_F0, -d.dot(hh[3]))
                        spec_d = _reflect(d, hh[3])
                        have_spec = True
                        p = hh[2]
                        n = hh[3]
                        continue
                    var a = _albedo(hh[4])
                    var dl = _direct(hh[2], hh[3], a, sun_dir, sun_rad)
                    acc = acc + Vec3(tp.x * dl.x, tp.y * dl.y, tp.z * dl.z)
                    tp = Vec3(tp.x * a.x, tp.y * a.y, tp.z * a.z)
                    p = hh[2]
                    n = hh[3]
                gi = gi + acc
            gi = gi * (Float32(1.0) / Float32(spp))
            var c = l_dir + Vec3(alb0.x * gi.x, alb0.y * gi.y, alb0.z * gi.z)
            out[py * w + px] = _to_srgb(Vec3(c.x * tp0.x, c.y * tp0.y, c.z * tp0.z))
    return out^


def _add_flat_quad(mut mb: MeshBuilder, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: Vec3):
    var i0 = mb.add_vertex(p0, n, 0.0, 0.0)
    var i1 = mb.add_vertex(p1, n, 1.0, 0.0)
    var i2 = mb.add_vertex(p2, n, 1.0, 1.0)
    var i3 = mb.add_vertex(p3, n, 0.0, 1.0)
    mb.add_quad(i0, i1, i2, i3)


def _build_cornell(mut ren: FlintAaaVulkanRenderer, mirror_floor: Bool = False) -> List[Int]:
    var m_white = AaaMaterial(base_color=Vec3(0.76, 0.76, 0.76), roughness=1.0, metallic=0.0)
    m_white.specular_factor = Float32(0.0)
    var m_red = AaaMaterial(base_color=Vec3(0.80, 0.08, 0.08), roughness=1.0, metallic=0.0)
    m_red.specular_factor = Float32(0.0)
    var m_green = AaaMaterial(base_color=Vec3(0.08, 0.78, 0.12), roughness=1.0, metallic=0.0)
    m_green.specular_factor = Float32(0.0)
    var id_w = ren.add_material(m_white)
    var id_r = ren.add_material(m_red)
    var id_g = ren.add_material(m_green)
    var m_mirror = AaaMaterial(base_color=Vec3(_MIRROR_F0, _MIRROR_F0, _MIRROR_F0), roughness=0.0, metallic=1.0)
    var id_m = ren.add_material(m_mirror)
    var b_flr = MeshBuilder(base_color=0xFFFFFFFF, roughness=Float32(0.0) if mirror_floor else Float32(1.0), metallic=Float32(1.0) if mirror_floor else Float32(0.0))
    var b_ceil = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_back = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_left = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_rght = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    for gy in range(16):
        var a0 = Float32(-2.0) + Float32(gy) * Float32(0.25)
        var a1 = a0 + Float32(0.25)
        var z0 = Float32(gy) * Float32(0.25)
        var z1 = z0 + Float32(0.25)
        for gx in range(16):
            var x0 = Float32(-2.0) + Float32(gx) * Float32(0.25)
            var x1 = x0 + Float32(0.25)
            var zx0 = Float32(gx) * Float32(0.25)
            var zx1 = zx0 + Float32(0.25)
            _add_flat_quad(b_flr, Vec3(x0, -2.0, z1), Vec3(x1, -2.0, z1), Vec3(x1, -2.0, z0), Vec3(x0, -2.0, z0), Vec3(0.0, 1.0, 0.0))
            _add_flat_quad(b_ceil, Vec3(x0, 2.0, z0), Vec3(x1, 2.0, z0), Vec3(x1, 2.0, z1), Vec3(x0, 2.0, z1), Vec3(0.0, -1.0, 0.0))
            _add_flat_quad(b_back, Vec3(x0, a0, 4.0), Vec3(x1, a0, 4.0), Vec3(x1, a1, 4.0), Vec3(x0, a1, 4.0), Vec3(0.0, 0.0, -1.0))
            _add_flat_quad(b_left, Vec3(-2.0, a0, zx0), Vec3(-2.0, a0, zx1), Vec3(-2.0, a1, zx1), Vec3(-2.0, a1, zx0), Vec3(1.0, 0.0, 0.0))
            _add_flat_quad(b_rght, Vec3(2.0, a0, zx1), Vec3(2.0, a0, zx0), Vec3(2.0, a1, zx0), Vec3(2.0, a1, zx1), Vec3(-1.0, 0.0, 0.0))
    var b_blk = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    for by in range(6):
        var fy0 = Float32(by) / Float32(6.0)
        var fy1 = Float32(by + 1) / Float32(6.0)
        var py0 = Float32(-2.0) + fy0 * Float32(1.80)
        var py1 = Float32(-2.0) + fy1 * Float32(1.80)
        var pz_t0 = Float32(1.45) + fy0 * Float32(1.50)
        var pz_t1 = Float32(1.45) + fy1 * Float32(1.50)
        for bx in range(6):
            var fx0 = Float32(bx) / Float32(6.0)
            var fx1 = Float32(bx + 1) / Float32(6.0)
            var px0 = Float32(-0.75) + fx0 * Float32(1.50)
            var px1 = Float32(-0.75) + fx1 * Float32(1.50)
            var pz0 = Float32(1.45) + fx0 * Float32(1.50)
            var pz1 = Float32(1.45) + fx1 * Float32(1.50)
            _add_flat_quad(b_blk, Vec3(px1, py0, 1.45), Vec3(px0, py0, 1.45), Vec3(px0, py1, 1.45), Vec3(px1, py1, 1.45), Vec3(0.0, 0.0, -1.0))
            _add_flat_quad(b_blk, Vec3(px0, py0, 2.95), Vec3(px1, py0, 2.95), Vec3(px1, py1, 2.95), Vec3(px0, py1, 2.95), Vec3(0.0, 0.0, 1.0))
            _add_flat_quad(b_blk, Vec3(-0.75, py0, pz0), Vec3(-0.75, py0, pz1), Vec3(-0.75, py1, pz1), Vec3(-0.75, py1, pz0), Vec3(-1.0, 0.0, 0.0))
            _add_flat_quad(b_blk, Vec3(0.75, py0, pz1), Vec3(0.75, py0, pz0), Vec3(0.75, py1, pz0), Vec3(0.75, py1, pz1), Vec3(1.0, 0.0, 0.0))
            _add_flat_quad(b_blk, Vec3(px0, -0.20, pz_t1), Vec3(px1, -0.20, pz_t1), Vec3(px1, -0.20, pz_t0), Vec3(px0, -0.20, pz_t0), Vec3(0.0, 1.0, 0.0))
    var ids = List[Int]()
    ids.append(ren.add_mesh(b_flr^.finish(), id_m if mirror_floor else id_w).mesh_id)
    ids.append(ren.add_mesh(b_ceil^.finish(), id_w).mesh_id)
    ids.append(ren.add_mesh(b_back^.finish(), id_w).mesh_id)
    ids.append(ren.add_mesh(b_left^.finish(), id_r).mesh_id)
    ids.append(ren.add_mesh(b_rght^.finish(), id_g).mesh_id)
    ids.append(ren.add_mesh(b_blk^.finish(), id_w).mesh_id)
    return ids^


struct GiRun(Copyable, Movable):
    var name: String
    var pixels: List[UInt32]
    var flip_v5: Float64
    var flip_full: Float64
    var gi_ms: Float64  # median of the GI passes (surface cache + shade + WRC + resolve + SVGF)
    var frame_ms: Float64  # median whole-frame GPU time
    var resolve_ms: Float64  # median of the area-B passes alone

    def __init__(out self, name: String):
        self.name = name
        self.pixels = List[UInt32]()
        self.flip_v5 = 1.0
        self.flip_full = 1.0
        self.gi_ms = 0.0
        self.frame_ms = 0.0
        self.resolve_ms = 0.0


def _median(mut v: List[Float64]) -> Float64:
    if len(v) == 0:
        return 0.0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


def _render_frames(mut ren: FlintAaaVulkanRenderer, ids: List[Int], frames: Int, mut run: GiRun) raises:
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    var gi_t = List[Float64]()
    var fr_t = List[Float64]()
    var rs_t = List[Float64]()
    for f in range(frames):
        ren.begin_frame(Vec3(0.0, 0.0, -3.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=50.0)
        for i in range(len(ids)):
            ren.draw_instance(ids[i], ident)
        _assert(ren.render_frame(), "render_frame failed in " + run.name)
        if f >= 1 or frames == 1:
            var tim = ren.get_gpu_pass_timings()
            var gi = 0.0
            var rs = 0.0
            for t in tim:
                var nm = t.pass_name
                if nm == "LumenSurfaceCache" or nm == "DeferredMultiLobeRtShade" or nm == "SvgfTemporalAndAtrous":
                    gi += t.gpu_ms
                if nm == "SotaWorldRadianceCache" or nm == "SotaGiResolve" or nm == "SotaGiUpsample":
                    gi += t.gpu_ms
                    rs += t.gpu_ms
            gi_t.append(gi)
            rs_t.append(rs)
            fr_t.append(ren.last_frame_gpu_ms)
    run.pixels = ren.readback_pixels()
    run.gi_ms = _median(gi_t)
    run.frame_ms = _median(fr_t)
    run.resolve_ms = _median(rs_t)


def _warm(mut ren: FlintAaaVulkanRenderer, ids: List[Int], frames: Int) raises:
    var w = GiRun("warmup")
    _render_frames(ren, ids, frames, w)


def _png_dir() -> String:
    var d = env_now("FLINT_SOTA_GI_PNG_DIR")
    if d.byte_length() == 0:
        d = env_now("HOME") + "/flint-models/_eval/render/B"
    _ = make_dirs(d)
    return d


def _find(runs: List[GiRun], name: String) raises -> Int:
    for i in range(len(runs)):
        if runs[i].name == name:
            return i
    raise Error("no run " + name)


def _masked_mean(fmap: List[Float32], mask: List[Bool]) -> Float64:
    var acc = 0.0
    var n = 0
    for i in range(len(fmap)):
        if mask[i]:
            acc += Float64(fmap[i])
            n += 1
    return acc / Float64(max(1, n))


def _new_cornell_renderer(cw: Int, ch: Int, sun_dir: Vec3, sun_rad: Vec3) -> FlintAaaVulkanRenderer:
    var ren = FlintAaaVulkanRenderer(width=cw, height=ch)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    # Pin the shipping SVGF denoiser: these thresholds measure the GI modes, and the
    # area-C learned denoiser switches itself on whenever its weights are on disk.
    ren.sota_ml.enabled = False
    ren.enable_vsm = False
    ren.enable_rt_shadows = True
    ren.enable_rt_reflections = False
    ren.enable_rt_gi = True
    ren.sun_shadow_softness = Float32(0.0)
    ren.ibl_intensity = Float32(0.0)
    ren.exposure = Float32(1.0)
    # v5 two-phase HiZ culling drops visible back-wall meshlets of this box from the 2nd frame on
    # (area A; see render-sota-gi.md), which would swamp every multi-frame GI measurement.
    ren.enable_hiz_cull = False
    ren.sun_dir = sun_dir
    ren.sun_radiance = sun_rad
    return ren^


def _wrc_grid_for_cornell(mut ren: FlintAaaVulkanRenderer):
    ren.sota_gi.wrc_center_fixed = True
    ren.sota_gi.wrc_center = Vec3(0.0, 0.0, 2.0)
    ren.sota_gi.wrc_spacing = Float32(0.5)
    ren.sota_gi.wrc_dims_x = 10
    ren.sota_gi.wrc_dims_y = 10
    ren.sota_gi.wrc_dims_z = 10
    ren.sota_gi.wrc_max_history = 64


def _print_row(r: GiRun):
    var pad = r.name
    while pad.byte_length() < 34:
        pad += " "
    print("    " + pad + _f3(r.flip_v5) + "         " + _f3(r.flip_full) + "          " + _f3(r.gi_ms) + "   " + _f3(r.resolve_ms) + "      " + _f3(r.frame_ms))


def test_gpu_cornell_gi_modes() raises:
    var cw = 128
    var ch = 128
    var sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var sun_rad = Vec3(3.1, 3.0, 2.9)
    var ren = _new_cornell_renderer(cw, ch, sun_dir, sun_rad)
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] SOTA GI GPU checks (no hardware ray queries)")
        return
    var ids = _build_cornell(ren)

    var quick = env_now("FLINT_SOTA_GI_QUICK").byte_length() > 0
    var spp = 128 if quick else 512
    var t_ref = get_time()
    var ref_v5 = render_cornell_reference(cw, ch, spp, sun_dir, sun_rad, 2)
    var ref_full = render_cornell_reference(cw, ch, spp, sun_dir, sun_rad, 12)
    print("    references: " + String(spp) + " spp, 2 and 12 indirect bounces in " + _f3(get_time() - t_ref) + " s")

    var runs = List[GiRun]()

    # (a) v5 baseline exactly as `test_flint.mojo` V14 (48 surface-cache rays + SVGF)
    var r_a = GiRun("v5_sc48_svgf")
    ren.sota_gi.mode = SOTA_GI_MODE_LEGACY
    ren.gi_rays_per_pixel = 48
    ren.svgf_passes = 2
    _render_frames(ren, ids, 1, r_a)
    runs.append(r_a^)

    # (b) v5 real-time budget: 4 rays + SVGF, 16 frames of temporal accumulation
    var r_b = GiRun("v5_sc4_svgf_16f")
    ren.gi_rays_per_pixel = 4
    _render_frames(ren, ids, 16, r_b)
    runs.append(r_b^)

    # A mode switch restarts the world radiance cache, so every with / without pair first warms the
    # mode up for 96 frames (probe hysteresis is 64 frames); the new feature then runs first, so the
    # baseline of the pair gets the slightly longer-converged cache (conservative comparisons).
    # (c2) world radiance cache (0 per-pixel GI rays, probes every 0.5 m) + 2 near-field contact rays
    _wrc_grid_for_cornell(ren)
    ren.sota_gi.mode = SOTA_GI_MODE_WRC
    ren.sota_gi.contact_rays = 2
    _warm(ren, ids, 96)
    var r_c2 = GiRun("sota_wrc_contact2_32f")
    _render_frames(ren, ids, 32, r_c2)
    runs.append(r_c2^)

    # (c) world radiance cache only, as in 91639b5
    var r_c = GiRun("sota_wrc_only_32f")
    ren.sota_gi.contact_rays = 0
    _render_frames(ren, ids, 32, r_c)
    runs.append(r_c^)

    # (d2) hybrid: 4 surface-cache rays per pixel, WRC behind them and as the surface-cache bounce, ReSTIR GI
    var r_d2 = GiRun("sota_hybrid_sc4_restirgi_32f")
    ren.sota_gi.mode = SOTA_GI_MODE_HYBRID
    ren.sota_gi.pixel_rays = 4
    ren.sota_gi.restir_gi_hybrid = True
    _warm(ren, ids, 96)
    _render_frames(ren, ids, 32, r_d2)
    runs.append(r_d2^)

    # (d) hybrid without ReSTIR GI (91639b5)
    var r_d = GiRun("sota_hybrid_sc4_wrc_32f")
    ren.sota_gi.restir_gi_hybrid = False
    _render_frames(ren, ids, 32, r_d)
    runs.append(r_d^)

    # (e2) real-time path tracer: 1 path / pixel, 2 bounces + WRC termination, ReSTIR GI, SVGF + TSR
    var r_e2 = GiRun("sota_pt1spp_restirgi_svgf_32f")
    # The ReSTIR GI pairs below are measured at a full-resolution resolve (their thresholds date from
    # round 2); half / quarter resolution of the path tracer is covered by `test_gpu_lowres_resolve`.
    ren.sota_gi.resolve_scale = 1
    ren.sota_gi.mode = SOTA_GI_MODE_PATH
    ren.sota_gi.pixel_rays = 1
    ren.sota_gi.pt_bounces = 2
    ren.sota_gi.pt_wrc_terminate = True
    ren.sota_gi.restir_gi = True
    _warm(ren, ids, 96)
    _render_frames(ren, ids, 32, r_e2)
    runs.append(r_e2^)

    # (e) the same without ReSTIR GI (91639b5)
    var r_e = GiRun("sota_pt1spp_wrc_svgf_32f")
    ren.sota_gi.restir_gi = False
    _render_frames(ren, ids, 32, r_e)
    runs.append(r_e^)

    # (e4, e3) 1 spp path tracer before the denoiser (SVGF off, TSR on): with / without ReSTIR GI
    ren.svgf_passes = 0
    var r_e4 = GiRun("sota_pt1spp_restirgi_raw_32f")
    ren.sota_gi.restir_gi = True
    _render_frames(ren, ids, 32, r_e4)
    runs.append(r_e4^)
    var r_e3 = GiRun("sota_pt1spp_raw_32f")
    ren.sota_gi.restir_gi = False
    _render_frames(ren, ids, 32, r_e3)
    runs.append(r_e3^)
    ren.sota_gi.restir_gi = True

    # (f) photo mode: 4 paths / pixel / frame, 8 bounces + WRC termination, progressive accumulation, no SVGF
    var r_f = GiRun("sota_pt_photo_8b_wrc_128f")
    ren.sota_gi.pixel_rays = 4
    ren.sota_gi.pt_bounces = 8
    ren.sota_gi.pt_wrc_terminate = True
    ren.sota_gi.accumulate = True
    ren.svgf_passes = 0
    _render_frames(ren, ids, 128, r_f)
    runs.append(r_f^)

    # (g) photo mode, 2 bounces, no cache: must converge to the v5 2-bounce reference (validates the tracer)
    var r_g = GiRun("sota_pt_photo_2b_nocache_128f")
    ren.sota_gi.pt_bounces = 2
    ren.sota_gi.pt_wrc_terminate = False
    _render_frames(ren, ids, 128, r_g)
    runs.append(r_g^)
    ren.sota_gi.accumulate = False
    ren.svgf_passes = 2

    var dir = _png_dir()
    _ = write_png(dir + "/ref_512spp_2bounce.png", ref_v5, cw, ch, stamped=False)
    _ = write_png(dir + "/ref_512spp_12bounce.png", ref_full, cw, ch, stamped=False)
    print("    mode                              FLIP(v5 ref)  FLIP(12b ref)  GI ms   area-B ms  frame ms")
    for i in range(len(runs)):
        runs[i].flip_v5 = flip_mean(flip_error_map(ref_v5, runs[i].pixels, cw, ch))
        var fmap = flip_error_map(ref_full, runs[i].pixels, cw, ch)
        runs[i].flip_full = flip_mean(fmap)
        var heat = flip_heat_image(fmap)
        _ = write_png(dir + "/" + runs[i].name + ".png", runs[i].pixels, cw, ch, stamped=False)
        _ = write_png(dir + "/" + runs[i].name + "_flip12b.png", heat, cw, ch, stamped=False)
        _print_row(runs[i])

    var i_a = _find(runs, "v5_sc48_svgf")
    var i_b = _find(runs, "v5_sc4_svgf_16f")
    var i_c = _find(runs, "sota_wrc_only_32f")
    var i_c2 = _find(runs, "sota_wrc_contact2_32f")
    var i_d = _find(runs, "sota_hybrid_sc4_wrc_32f")
    var i_d2 = _find(runs, "sota_hybrid_sc4_restirgi_32f")
    var i_e = _find(runs, "sota_pt1spp_wrc_svgf_32f")
    var i_e2 = _find(runs, "sota_pt1spp_restirgi_svgf_32f")
    var i_e3 = _find(runs, "sota_pt1spp_raw_32f")
    var i_e4 = _find(runs, "sota_pt1spp_restirgi_raw_32f")
    var i_f = _find(runs, "sota_pt_photo_8b_wrc_128f")
    var i_g = _find(runs, "sota_pt_photo_2b_nocache_128f")
    # Correctness: the 2-bounce no-cache path tracer converges to the v5 2-bounce reference
    var tol_ref = 0.065 if quick else 0.05
    _assert(runs[i_g].flip_v5 < tol_ref, "photo-mode 2-bounce PT vs 2-bounce reference FLIP " + _f3(runs[i_g].flip_v5))
    # The multi-bounce photo mode beats the v5 baseline against the converged reference
    _assert(runs[i_f].flip_full < runs[i_a].flip_full, "photo-mode PT should beat the v5 baseline")
    _assert(runs[i_f].flip_full < 0.07, "photo-mode PT vs converged reference FLIP " + _f3(runs[i_f].flip_full))
    # Real-time modes against the converged reference (v5 baseline 0.26). With a warmed-up WRC the
    # 91639b5 modes measure ~0.10; 91639b5's 0.150 / 0.124 were taken on an under-converged cache.
    _assert(runs[i_e].flip_full < runs[i_b].flip_full, "RT path tracer should beat v5 4-ray GI")
    _assert(runs[i_c].flip_full < 0.13, "WRC-only FLIP " + _f3(runs[i_c].flip_full))
    _assert(runs[i_d].flip_full < 0.13, "hybrid FLIP " + _f3(runs[i_d].flip_full))
    _assert(runs[i_c2].flip_full < 0.14, "WRC + contact rays FLIP " + _f3(runs[i_c2].flip_full))
    _assert(runs[i_d2].flip_full < 0.14, "hybrid + ReSTIR GI FLIP " + _f3(runs[i_d2].flip_full))
    _assert(runs[i_c].flip_full < runs[i_a].flip_full, "WRC-only should beat the v5 baseline")
    _assert(runs[i_d].flip_full < runs[i_a].flip_full, "hybrid should beat the v5 baseline")
    # ReSTIR GI on the 1-spp path tracer: less error before the denoiser and after it
    _assert(runs[i_e4].flip_full < runs[i_e3].flip_full * 0.85, "ReSTIR GI should cut raw 1-spp path-tracer error")
    _assert(runs[i_e2].flip_full < runs[i_e].flip_full * 0.85, "ReSTIR GI should cut denoised 1-spp path-tracer error")
    # Perf budgets at 128x128 (generous: the GPU is shared)
    _assert(runs[i_c2].resolve_ms < 2.0, "WRC + contact area-B passes " + _f3(runs[i_c2].resolve_ms) + " ms")
    _assert(runs[i_e2].resolve_ms < 3.0, "RT path tracer + ReSTIR GI area-B passes " + _f3(runs[i_e2].resolve_ms) + " ms")
    ren.shutdown()
    print("  [PASS] SOTA GI Cornell modes vs 512-spp references (PNGs in " + dir + ")")


# ---------------------------------------------------------------------------------------------
# 3. RT reflections: mirror floor, reflection hits lit by the WRC
# ---------------------------------------------------------------------------------------------
def test_gpu_mirror_reflections() raises:
    var cw = 128
    var ch = 128
    var sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var sun_rad = Vec3(3.1, 3.0, 2.9)
    var ren = _new_cornell_renderer(cw, ch, sun_dir, sun_rad)
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] SOTA GI mirror reflections (no hardware ray queries)")
        return
    ren.enable_rt_reflections = True
    var ids = _build_cornell(ren, mirror_floor=True)
    var quick = env_now("FLINT_SOTA_GI_QUICK").byte_length() > 0
    var spp = 128 if quick else 512
    var ref_img = render_cornell_reference(cw, ch, spp, sun_dir, sun_rad, 12, mirror_floor=True)
    var mask = List[Bool](length=cw * ch, fill=False)
    for py in range(ch):
        for px in range(cw):
            mask[py * cw + px] = _is_mirror(True, _cornell_intersect(Vec3(0.0, 0.0, -3.5), _primary_dir(px, py, cw, ch)))

    var runs = List[GiRun]()
    var r0 = GiRun("mirror_v5_sc48_svgf_8f")
    ren.gi_rays_per_pixel = 48
    _render_frames(ren, ids, 8, r0)
    runs.append(r0^)
    _wrc_grid_for_cornell(ren)
    ren.sota_gi.mode = SOTA_GI_MODE_WRC
    ren.sota_gi.refl_use_wrc = False
    var r1 = GiRun("mirror_wrc_hits_sh9sky_32f")
    _render_frames(ren, ids, 32, r1)
    runs.append(r1^)
    ren.sota_gi.refl_use_wrc = True
    var r2 = GiRun("mirror_wrc_hits_wrc_32f")
    _render_frames(ren, ids, 32, r2)
    runs.append(r2^)
    ren.sota_gi.mode = SOTA_GI_MODE_HYBRID
    ren.sota_gi.pixel_rays = 4
    var r3 = GiRun("mirror_hybrid_hits_wrc_32f")
    _render_frames(ren, ids, 32, r3)
    runs.append(r3^)

    var dir = _png_dir()
    _ = write_png(dir + "/ref_mirror_512spp_12bounce.png", ref_img, cw, ch, stamped=False)
    print("    mirror floor                      FLIP(full)    FLIP(mirror px) GI ms  frame ms")
    var masked = List[Float64]()
    for i in range(len(runs)):
        var fmap = flip_error_map(ref_img, runs[i].pixels, cw, ch)
        runs[i].flip_full = flip_mean(fmap)
        masked.append(_masked_mean(fmap, mask))
        _ = write_png(dir + "/" + runs[i].name + ".png", runs[i].pixels, cw, ch, stamped=False)
        var heat = flip_heat_image(fmap)
        _ = write_png(dir + "/" + runs[i].name + "_flip.png", heat, cw, ch, stamped=False)
        var pad = runs[i].name
        while pad.byte_length() < 34:
            pad += " "
        print("    " + pad + _f3(runs[i].flip_full) + "         " + _f3(masked[i]) + "          " + _f3(runs[i].gi_ms) + "  " + _f3(runs[i].frame_ms))
    _assert(masked[2] < masked[1], "WRC-lit reflection hits should beat SH9-sky-lit hits on the mirror")
    ren.shutdown()
    print("  [PASS] SOTA GI RT reflections with WRC-lit secondary hits")


# ---------------------------------------------------------------------------------------------
# 4. ReSTIR DI: 128 shadowed point lights, reuse vs the v5 4-candidate RIS
# ---------------------------------------------------------------------------------------------
comptime N_DI_LIGHTS: Int = 128


def _add_many_lights(mut ren: FlintAaaVulkanRenderer):
    var st = UInt32(12345)
    for i in range(N_DI_LIGHTS):
        var p = Vec3(_pcg(st) * Float32(3.4) - Float32(1.7), _pcg(st) * Float32(3.4) - Float32(1.7), _pcg(st) * Float32(3.2) + Float32(0.4))
        if abs(p.x) < Float32(0.95) and p.y < Float32(0.0) and p.z > Float32(1.25) and p.z < Float32(3.15):
            p.y = Float32(0.25) + _pcg(st) * Float32(1.4)  # keep lights out of the block
        var c = Vec3(Float32(0.3) + _pcg(st), Float32(0.3) + _pcg(st), Float32(0.3) + _pcg(st)) * (Float32(0.15) + _pcg(st) * Float32(0.6))
        ren.add_cluster_light(AaaClusterLight.point(p, c, Float32(1.6)))
        _ = i


def _render_di_frame(mut ren: FlintAaaVulkanRenderer, ids: List[Int]) raises -> Float64:
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    ren.begin_frame(Vec3(0.0, 0.0, -3.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=50.0)
    for i in range(len(ids)):
        ren.draw_instance(ids[i], ident)
    _add_many_lights(ren)
    _assert(ren.render_frame(), "render_frame failed (ReSTIR DI)")
    var ms = 0.0
    for t in ren.get_gpu_pass_timings():
        if t.pass_name == "DeferredMultiLobeRtShade":
            ms += t.gpu_ms
    return ms


def _di_run(mut ren: FlintAaaVulkanRenderer, ids: List[Int], frames: Int, mut run: GiRun) raises:
    var t = List[Float64]()
    for f in range(frames):
        var ms = _render_di_frame(ren, ids)
        if f >= 1:
            t.append(ms)
    run.pixels = ren.readback_pixels()
    run.gi_ms = _median(t)


def test_gpu_restir_di() raises:
    var cw = 128
    var ch = 128
    var ren = _new_cornell_renderer(cw, ch, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0))
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] SOTA ReSTIR DI (no hardware ray queries)")
        return
    ren.enable_rt_gi = False
    ren.enable_restir_di = True
    ren.tsr_history_weight = Float32(0.0)  # raw estimator noise: no TSR history, no jitter
    # Low exposure: 8-bit RIS frames must not clip, or the mean-of-RIS reference is biased dark
    # (at exposure 1, 6% of RIS channels saturate vs 2% for ReSTIR, which read as a +20% "bias").
    ren.exposure = Float32(0.25)
    var ids = _build_cornell(ren)
    var quick = env_now("FLINT_SOTA_GI_QUICK").byte_length() > 0
    var n_ref = 96 if quick else 384

    # Reference: mean of independent frames of the unbiased v5 RIS estimator (linear light)
    ren.sota_gi.restir_di_reuse = False
    var acc = List[Float64](length=cw * ch * 3, fill=0.0)
    var ris = GiRun("di_v5_ris4_1f")
    var ris_t = List[Float64]()
    for f in range(n_ref):
        var ms = _render_di_frame(ren, ids)
        if f >= 1:
            ris_t.append(ms)
        var px = ren.readback_pixels()
        if f == n_ref - 1:
            ris.pixels = px.copy()
        for i in range(cw * ch):
            for c in range(3):
                acc[i * 3 + c] += Float64(srgb_to_linear(Float32((px[i] >> UInt32(16 - 8 * c)) & 0xFF) / Float32(255.0)))
    ris.gi_ms = _median(ris_t)
    var ref_img = List[UInt32](length=cw * ch, fill=0xFF000000)
    for i in range(cw * ch):
        var inv = 1.0 / Float64(n_ref)
        ref_img[i] = _to_srgb(Vec3(Float32(acc[i * 3] * inv), Float32(acc[i * 3 + 1] * inv), Float32(acc[i * 3 + 2] * inv)))

    var ref_mean = 0.0
    for k in range(cw * ch * 3):
        ref_mean += acc[k] / Float64(n_ref)
    var runs = List[GiRun]()
    runs.append(ris^)
    ren.sota_gi.restir_di_reuse = True
    ren.sota_gi.restir_di_spatial = 0
    var r_t = GiRun("di_restir_temporal_16f")
    _di_run(ren, ids, 16, r_t)
    runs.append(r_t^)
    ren.sota_gi.restir_di_spatial = 2
    ren.sota_gi.restir_di_radius = Float32(6.0)
    var r_ts = GiRun("di_restir_temporal_spatial2_16f")
    _di_run(ren, ids, 16, r_ts)
    runs.append(r_ts^)

    var dir = _png_dir()
    _ = write_png(dir + "/ref_di_mean_ris.png", ref_img, cw, ch, stamped=False)
    print("    ReSTIR DI (128 point lights)      FLIP(vs mean)  shade ms")
    for i in range(len(runs)):
        var fmap = flip_error_map(ref_img, runs[i].pixels, cw, ch)
        runs[i].flip_full = flip_mean(fmap)
        _ = write_png(dir + "/" + runs[i].name + ".png", runs[i].pixels, cw, ch, stamped=False)
        var heat = flip_heat_image(fmap)
        _ = write_png(dir + "/" + runs[i].name + "_flip.png", heat, cw, ch, stamped=False)
        var pad = runs[i].name
        while pad.byte_length() < 34:
            pad += " "
        var lm = 0.0
        var sat = 0
        for k in range(cw * ch):
            for c in range(3):
                var b8 = (runs[i].pixels[k] >> UInt32(16 - 8 * c)) & 0xFF
                lm += Float64(srgb_to_linear(Float32(b8) / Float32(255.0)))
                if b8 >= 254:
                    sat += 1
        runs[i].flip_v5 = lm / ref_mean  # linear mean / reference mean
        print("    " + pad + _f3(runs[i].flip_full) + "          " + _f3(runs[i].gi_ms) + "     mean/ref " + _f3(runs[i].flip_v5) + "  clipped " + _f3(Float64(sat) / Float64(cw * ch * 3)))
    # Biased (1/M) ReSTIR stays within a few percent of the unbiased mean
    _assert(abs(runs[1].flip_v5 - 1.0) < 0.04, "temporal ReSTIR DI mean bias " + _f3(runs[1].flip_v5))
    _assert(abs(runs[2].flip_v5 - 1.0) < 0.04, "spatiotemporal ReSTIR DI mean bias " + _f3(runs[2].flip_v5))
    _assert(runs[1].flip_full < runs[0].flip_full, "temporal ReSTIR DI should beat 4-candidate RIS")
    # Spatial reuse (opt-in) must stay unbiased and well below RIS; on this static view it does not beat
    # temporal-only reuse (0.115 vs 0.107), so it is off by default.
    _assert(runs[2].flip_full < runs[0].flip_full * 0.7, "spatiotemporal ReSTIR DI vs RIS FLIP " + _f3(runs[2].flip_full))
    _assert(runs[1].flip_full < runs[0].flip_full * 0.7, "temporal ReSTIR DI vs RIS FLIP " + _f3(runs[1].flip_full))
    _assert(runs[2].gi_ms < runs[0].gi_ms + 1.0, "ReSTIR DI reuse cost " + _f3(runs[2].gi_ms - runs[0].gi_ms) + " ms")
    ren.shutdown()
    print("  [PASS] SOTA ReSTIR DI temporal + spatial reuse")


# ---------------------------------------------------------------------------------------------
# 5. SVGF temporal history regression (a-trous ping-pong used to clobber history bank A)
# ---------------------------------------------------------------------------------------------


struct SvgfHistStats(Copyable, Movable):
    var mean_len: Float64  # SVGF history length (slot 3 of the bank the reprojection just wrote)
    var mean_var: Float64  # variance the a-trous pass sees: Var(L) = m2 - m1^2, x 4 / len while len < 4
    var flicker: Float64  # mean |dG| of the final 8-bit output between the last two frames (temporal variance)
    var flip_v5: Float64
    var flip_full: Float64
    var frac_cap: Float64  # fraction of surface pixels at >= 20 frames
    var frac_one: Float64  # fraction of surface pixels whose history was rejected this frame

    def __init__(out self):
        self.frac_cap = 0.0
        self.frac_one = 0.0
        self.mean_len = 0.0
        self.mean_var = 0.0
        self.flicker = 0.0
        self.flip_v5 = 1.0
        self.flip_full = 1.0


def _svgf_hist_stats(
    mut ren: FlintAaaVulkanRenderer, ids: List[Int], frames: Int, ref_v5: List[UInt32], ref_full: List[UInt32], name: String
) raises -> SvgfHistStats:
    var r = GiRun(name)
    _render_frames(ren, ids, frames, r)
    # End on an odd frame: it writes history bank B ([8 max_pix, 16 max_pix)), which the a-trous pass
    # never touched, so the length read here is the real one with and without the fix.
    if ((ren.frame_index - 1) & 1) == 0:
        _render_frames(ren, ids, 1, r)
    var prev = r.pixels.copy()
    var r2 = GiRun(name)
    _render_frames(ren, ids, 2, r2)  # two more frames: ends on an odd frame again
    var st = SvgfHistStats()
    var n = ren.render_w * ren.render_h
    var max_pix = max(ren.output_w * ren.output_h, n)
    var bank = 0 if ((ren.frame_index - 1) & 1) == 0 else max_pix * 8
    var h = List[Float32](length=n * 8, fill=0.0)
    _ = external_call["memcpy", Int](Int(h.unsafe_ptr()), ren.buf_history.cpu_address + bank * 4, n * 32)
    var cnt = 0
    for i in range(n):
        var z = h[i * 8 + 6]
        if z > 0.0 and z < 1e5:
            var ln = Float64(h[i * 8 + 3])
            var m1 = Float64(h[i * 8 + 4])
            var v = max(Float64(h[i * 8 + 5]) - m1 * m1, 0.0)
            if ln < 4.0:
                v = v * 4.0 / max(ln, 1.0)
            st.mean_len += ln
            if ln >= 20.0:
                st.frac_cap += 1.0
            if ln < 1.5:
                st.frac_one += 1.0
            st.mean_var += v
            cnt += 1
    st.mean_len /= Float64(max(cnt, 1))
    st.mean_var /= Float64(max(cnt, 1))
    st.frac_cap /= Float64(max(cnt, 1))
    st.frac_one /= Float64(max(cnt, 1))
    var fl = 0.0
    for i in range(len(prev)):
        fl += abs(Float64((r2.pixels[i] >> 8) & 0xFF) - Float64((prev[i] >> 8) & 0xFF))
    st.flicker = fl / Float64(max(len(prev), 1))
    st.flip_v5 = flip_mean(flip_error_map(ref_v5, r2.pixels, ren.output_w, ren.output_h))
    st.flip_full = flip_mean(flip_error_map(ref_full, r2.pixels, ren.output_w, ren.output_h))
    _ = write_png(_png_dir() + "/" + name + ".png", r2.pixels, ren.output_w, ren.output_h, stamped=False)
    print(
        "    " + name + ": len " + _f3(st.mean_len) + " (>=20: " + _f3(st.frac_cap) + ", 1: " + _f3(st.frac_one) + ")  var " + String(st.mean_var) + "  flicker " + _f3(st.flicker)
        + "  FLIP(v5 ref) " + _f3(st.flip_v5) + "  FLIP(12b ref) " + _f3(st.flip_full)
    )
    return st^


def test_gpu_svgf_history() raises:
    var cw = 128
    var ch = 128
    var sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var sun_rad = Vec3(3.1, 3.0, 2.9)
    var ren = _new_cornell_renderer(cw, ch, sun_dir, sun_rad)
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] SVGF history GPU checks (no hardware ray queries)")
        return
    var ids = _build_cornell(ren)
    var quick = env_now("FLINT_SOTA_GI_QUICK").byte_length() > 0
    var spp = 128 if quick else 512
    var ref_v5 = render_cornell_reference(cw, ch, spp, sun_dir, sun_rad, 2)
    var ref_full = render_cornell_reference(cw, ch, spp, sun_dir, sun_rad, 12)
    var frames = 40  # > the 24-frame history cap
    # v5 shipping GI: 4 surface-cache rays + SVGF
    ren.sota_gi.mode = SOTA_GI_MODE_LEGACY
    ren.gi_rays_per_pixel = 4
    ren.svgf_passes = 2
    ren.sota_gi.svgf_history_fix = False
    var a0 = _svgf_hist_stats(ren, ids, frames, ref_v5, ref_full, "svgf_hist_v5_sc4_before")
    ren.sota_gi.svgf_history_fix = True
    var a1 = _svgf_hist_stats(ren, ids, frames, ref_v5, ref_full, "svgf_hist_v5_sc4_after")
    # Without the TSR sub-pixel jitter every static pixel passes the reprojection test, so the history
    # must sit at the cap; with jitter, depth/normal edges and grazing walls still reject now and then.
    var tsr_w = ren.tsr_history_weight
    ren.tsr_history_weight = Float32(0.0)
    var a2 = _svgf_hist_stats(ren, ids, frames, ref_v5, ref_full, "svgf_hist_v5_sc4_after_nojitter")
    ren.tsr_history_weight = tsr_w
    # Area-B 1-spp path tracer (ReSTIR GI on) + SVGF
    _wrc_grid_for_cornell(ren)
    ren.sota_gi.mode = SOTA_GI_MODE_PATH
    ren.sota_gi.pixel_rays = 1
    ren.sota_gi.pt_bounces = 3
    _warm(ren, ids, 96)
    ren.sota_gi.svgf_history_fix = False
    var b0 = _svgf_hist_stats(ren, ids, frames, ref_v5, ref_full, "svgf_hist_pt1spp_before")
    ren.sota_gi.svgf_history_fix = True
    var b1 = _svgf_hist_stats(ren, ids, frames, ref_v5, ref_full, "svgf_hist_pt1spp_after")
    ren.shutdown()
    # The bug: every surface pixel restarts at length 1 (bank A holds a-trous records, slot 3 = depth)
    _assert(a0.mean_len < 1.5 and b0.mean_len < 1.5, "SVGF history should be clobbered with the fix off")
    _assert(a1.mean_len > 10.0 and a1.frac_cap > 0.4, "SVGF history length after " + String(frames) + " static frames " + _f3(a1.mean_len))
    _assert(b1.mean_len > 10.0 and b1.frac_cap > 0.4, "PT SVGF history length after " + String(frames) + " static frames " + _f3(b1.mean_len))
    _assert(a2.mean_len > 20.0 and a2.frac_cap > 0.9, "unjittered SVGF history length " + _f3(a2.mean_len))
    # One-sample moments give Var(L) = 0, which collapses the a-trous luminance edge-stop; the fixed
    # history gives a real variance estimate, and the output flickers less between static frames.
    _assert(a0.mean_var < 1e-6 and b0.mean_var < 1e-6, "clobbered history should have degenerate variance")
    _assert(a1.mean_var > 1e-4 and b1.mean_var > 1e-4, "accumulated history should estimate the variance")
    _assert(a1.flicker < a0.flicker * 0.8, "temporal flicker should fall with the full history (v5 GI)")
    _assert(b1.flicker < b0.flicker * 0.8, "temporal flicker should fall with the full history (PT)")
    _assert(a1.flip_full <= a0.flip_full + 0.002, "SVGF fix must not raise FLIP (v5 GI)")
    _assert(b1.flip_full <= b0.flip_full + 0.002, "SVGF fix must not raise FLIP (PT)")
    print("  [PASS] SVGF temporal history accumulates to the cap after static frames")


# ---------------------------------------------------------------------------------------------
# 6. Round 3: reduced-resolution GI resolve + depth / normal-aware upsample
# ---------------------------------------------------------------------------------------------
def _lowres_run(
    mut ren: FlintAaaVulkanRenderer, ids: List[Int], name: String, scale: Int, frames: Int, mut runs: List[GiRun]
) raises:
    ren.sota_gi.resolve_scale = scale
    _warm(ren, ids, 96)
    var r = GiRun(name + "_s" + String(scale))
    _render_frames(ren, ids, frames, r)
    runs.append(r^)


def test_gpu_lowres_resolve() raises:
    var cw = 128
    var ch = 128
    var sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var sun_rad = Vec3(3.1, 3.0, 2.9)
    var ren = _new_cornell_renderer(cw, ch, sun_dir, sun_rad)
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] SOTA GI reduced-res resolve (no hardware ray queries)")
        return
    var ids = _build_cornell(ren)
    var quick = env_now("FLINT_SOTA_GI_QUICK").byte_length() > 0
    var ref_full = render_cornell_reference(cw, ch, 128 if quick else 512, sun_dir, sun_rad, 12)
    _wrc_grid_for_cornell(ren)
    var runs = List[GiRun]()
    # Hybrid (the `high` preset) and the 1-spp path tracer with ReSTIR GI, SVGF + TSR, at 1/1, 1/2, 1/4
    ren.sota_gi.mode = SOTA_GI_MODE_HYBRID
    ren.sota_gi.pixel_rays = 4
    for sc in [1, 2, 4]:
        _lowres_run(ren, ids, "lowres_hybrid_sc4", sc, 32, runs)
    ren.sota_gi.mode = SOTA_GI_MODE_PATH
    ren.sota_gi.pixel_rays = 1
    ren.sota_gi.pt_bounces = 2
    ren.sota_gi.restir_gi = True
    for sc in [1, 2, 4]:
        _lowres_run(ren, ids, "lowres_pt1spp_restirgi", sc, 32, runs)
    # Photo mode always resolves at full resolution (accumulation is per pixel)
    ren.sota_gi.resolve_scale = 2
    ren.sota_gi.accumulate = True
    ren.sota_gi.pixel_rays = 4
    var r_ph = GiRun("lowres_photo_2b_s2")
    ren.sota_gi.pt_wrc_terminate = False
    ren.svgf_passes = 0
    _render_frames(ren, ids, 64, r_ph)
    _assert(ren.sota_gi.cur_scale == 1, "photo mode must resolve at full resolution")
    ren.sota_gi.accumulate = False
    ren.sota_gi.pt_wrc_terminate = True
    ren.svgf_passes = 2
    ren.sota_gi.pixel_rays = 1

    var dir = _png_dir()
    print("    reduced-res resolve (Cornell 128x128, SVGF + TSR)   FLIP(12b ref)  area-B ms")
    for i in range(len(runs)):
        var fmap = flip_error_map(ref_full, runs[i].pixels, cw, ch)
        runs[i].flip_full = flip_mean(fmap)
        _ = write_png(dir + "/" + runs[i].name + ".png", runs[i].pixels, cw, ch, stamped=False)
        var heat = flip_heat_image(fmap)
        _ = write_png(dir + "/" + runs[i].name + "_flip12b.png", heat, cw, ch, stamped=False)
        var pad = runs[i].name
        while pad.byte_length() < 52:
            pad += " "
        print("    " + pad + _f3(runs[i].flip_full) + "          " + _f3(runs[i].resolve_ms))
    var h1 = runs[_find(runs, "lowres_hybrid_sc4_s1")].flip_full
    var h2 = runs[_find(runs, "lowres_hybrid_sc4_s2")].flip_full
    var p1 = runs[_find(runs, "lowres_pt1spp_restirgi_s1")].flip_full
    var p2 = runs[_find(runs, "lowres_pt1spp_restirgi_s2")].flip_full
    # Half resolution may cost a little quality (a quarter of the GI samples per frame), never much
    _assert(h2 < h1 + 0.015, "half-res hybrid FLIP " + _f3(h2) + " vs full " + _f3(h1))
    _assert(p2 < p1 + 0.025, "half-res path tracer FLIP " + _f3(p2) + " vs full " + _f3(p1))
    _assert(h2 < 0.13, "half-res hybrid FLIP " + _f3(h2))

    # Cost: the resolve at 512x384 must get cheaper at half resolution (resolve + upsample)
    ren.shutdown()
    var big = _new_cornell_renderer(512, 384, sun_dir, sun_rad)
    var ids_b = _build_cornell(big)
    _wrc_grid_for_cornell(big)
    big.sota_gi.mode = SOTA_GI_MODE_HYBRID
    big.sota_gi.pixel_rays = 4
    var t = List[Float64]()
    for sc in [1, 2, 1, 2]:
        big.sota_gi.resolve_scale = sc
        _warm(big, ids_b, 8)
        var r = GiRun("timing")
        _render_frames(big, ids_b, 24, r)
        t.append(r.resolve_ms)
    var t1 = min(t[0], t[2])
    var t2 = min(t[1], t[3])
    print("    512x384 hybrid sc4: area-B passes " + _f3(t1) + " ms full res -> " + _f3(t2) + " ms half res")
    _assert(t2 < t1 * 0.95, "half-res resolve should be cheaper: " + _f3(t2) + " vs " + _f3(t1) + " ms")
    big.shutdown()
    print("  [PASS] SOTA GI reduced-resolution resolve + depth/normal-aware upsample")


# ---------------------------------------------------------------------------------------------
# 7. Round 3: area-C learned denoiser vs the (fixed) SVGF per GI mode -> `SOTA_ML_GI_MODES_DEFAULT`
# ---------------------------------------------------------------------------------------------
def _ml_pair(
    mut ren: FlintAaaVulkanRenderer, ids: List[Int], name: String, frames: Int, refimg: List[UInt32], cw: Int, ch: Int
) raises -> Tuple[Float64, Float64]:
    """Learned first (the conservative order: SVGF gets the longer-converged caches), then SVGF."""
    var dir = _png_dir()
    _warm(ren, ids, 96)
    ren.sota_ml.enabled = True
    var r_ml = GiRun(name + "_learned")
    _render_frames(ren, ids, frames, r_ml)
    ren.sota_ml.enabled = False
    var r_sv = GiRun(name + "_svgf")
    _render_frames(ren, ids, frames, r_sv)
    var f_ml = flip_mean(flip_error_map(refimg, r_ml.pixels, cw, ch))
    var f_sv = flip_mean(flip_error_map(refimg, r_sv.pixels, cw, ch))
    _ = write_png(dir + "/ml_" + r_ml.name + ".png", r_ml.pixels, cw, ch, stamped=False)
    _ = write_png(dir + "/ml_" + r_sv.name + ".png", r_sv.pixels, cw, ch, stamped=False)
    var pad = name
    while pad.byte_length() < 40:
        pad += " "
    print("    " + pad + _f3(f_sv) + "     " + _f3(f_ml) + "      " + _f3(f_ml - f_sv))
    return (f_sv, f_ml)


def test_gpu_ml_gating() raises:
    var wpath = sota_ml_default_weights_path()
    if len(read_bytes(wpath)) == 0:
        print("  [SKIP] learned-denoiser gating (no weights at " + wpath + ")")
        return
    var cw = 128
    var ch = 128
    var sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var sun_rad = Vec3(3.1, 3.0, 2.9)
    var ren = _new_cornell_renderer(cw, ch, sun_dir, sun_rad)
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] learned-denoiser gating (no hardware ray queries)")
        return
    var ids = _build_cornell(ren)
    var quick = env_now("FLINT_SOTA_GI_QUICK").byte_length() > 0
    var ref_full = render_cornell_reference(cw, ch, 128 if quick else 512, sun_dir, sun_rad, 12)
    ren.sota_ml.gi_modes = 0xF  # measure every mode; the shipped mask is checked against the data below
    ren.svgf_passes = 2
    print("    GI mode (Cornell 128x128, FLIP vs 12b)    SVGF      learned    delta")
    var deltas = List[Float64]()  # legacy, WRC, hybrid, path (at the default resolve scale)
    ren.sota_gi.mode = SOTA_GI_MODE_LEGACY
    ren.gi_rays_per_pixel = 4
    var d0 = _ml_pair(ren, ids, "legacy_sc4_16f", 16, ref_full, cw, ch)
    deltas.append(d0[1] - d0[0])
    _wrc_grid_for_cornell(ren)
    ren.sota_gi.mode = SOTA_GI_MODE_WRC
    ren.sota_gi.contact_rays = 0
    var d1 = _ml_pair(ren, ids, "wrc_only_32f", 32, ref_full, cw, ch)
    deltas.append(d1[1] - d1[0])
    ren.sota_gi.mode = SOTA_GI_MODE_HYBRID
    ren.sota_gi.pixel_rays = 4
    var def_scale = ren.sota_gi.resolve_scale
    ren.sota_gi.resolve_scale = 1
    _ = _ml_pair(ren, ids, "hybrid_sc4_fullres_32f", 32, ref_full, cw, ch)
    ren.sota_gi.resolve_scale = def_scale
    var d2 = _ml_pair(ren, ids, "hybrid_sc4_32f", 32, ref_full, cw, ch)
    deltas.append(d2[1] - d2[0])
    ren.sota_gi.mode = SOTA_GI_MODE_PATH
    ren.sota_gi.pixel_rays = 1
    ren.sota_gi.pt_bounces = 2
    ren.sota_gi.restir_gi = True
    ren.sota_gi.resolve_scale = 1
    _ = _ml_pair(ren, ids, "pt1spp_restirgi_fullres_32f", 32, ref_full, cw, ch)
    ren.sota_gi.resolve_scale = def_scale
    var d3 = _ml_pair(ren, ids, "pt1spp_restirgi_32f", 32, ref_full, cw, ch)
    deltas.append(d3[1] - d3[0])
    ren.sota_ml.enabled = False
    ren.shutdown()
    # The shipped gating must follow the data: a mode where the learned path is clearly better
    # (FLIP lower by > 0.005) is enabled by default, one where it is clearly worse is not.
    for m in range(4):
        var on = ((SOTA_ML_GI_MODES_DEFAULT >> m) & 1) != 0
        if deltas[m] < -0.005:
            _assert(on, "learned denoiser is better in GI mode " + String(m) + " but gated off")
        if deltas[m] > 0.005:
            _assert(not on, "learned denoiser is worse in GI mode " + String(m) + " but gated on")
    print("  [PASS] learned-denoiser GI-mode gating matches the measurements (mask " + String(SOTA_ML_GI_MODES_DEFAULT) + ")")


def main() raises:
    var t0 = get_time()
    print("=== Flint v5 Area B (GI) SOTA Suite ===")
    var only = env_now("FLINT_SOTA_GI_ONLY")  # "gi", "mirror", "di", "svgf", "lowres" or "ml" runs one GPU suite
    test_cpu_twins()
    if only.byte_length() == 0 or only == "gi":
        test_gpu_cornell_gi_modes()
    if only.byte_length() == 0 or only == "mirror":
        test_gpu_mirror_reflections()
    if only.byte_length() == 0 or only == "di":
        test_gpu_restir_di()
    if only.byte_length() == 0 or only == "svgf":
        test_gpu_svgf_history()
    if only.byte_length() == 0 or only == "lowres":
        test_gpu_lowres_resolve()
    if only.byte_length() == 0 or only == "ml":
        test_gpu_ml_gating()
    print("=== ALL FLINT V5 SOTA GI TESTS PASSED in " + _f3(get_time() - t0) + "s ===")
