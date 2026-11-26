"""Area B render SOTA: world radiance cache, hybrid surface-cache GI and the real-time path tracer
(`test_flint_v5_sota_gi.mojo`).

1. CPU twins: SH9 projection / irradiance convolution, Chebyshev visibility, toroidal probe slots and
   SPIR-V validation of the new passes and of the hooked v5 shaders.
2. GPU (needs hardware ray queries, otherwise skipped): the Cornell box of `test_flint.mojo` rendered
   with every GI mode and compared by FLIP against two 512-spp CPU references (the v5 2-indirect-bounce
   one and a converged multi-bounce one), with GPU pass timings and before/after PNGs.

Environment: `FLINT_SOTA_GI_PNG_DIR` overrides where PNGs go (default `~/flint-models/_eval/render/B`);
`FLINT_SOTA_GI_QUICK=1` lowers the reference sample count for iteration.
"""
from std.math import abs, cos, exp, log, max, min, sin, sqrt, tan
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import env_now, get_time, make_dirs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_gi_svgf import aaa_deferred_shade_shader, surface_cache_update_shader
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import flip_error_map, flip_heat_image, flip_mean
from flint.render.spirv_validate import validate_spirv_module
from flint.render.sota_gi_world_cache import (
    SOTA_GI_MODE_HYBRID,
    SOTA_GI_MODE_LEGACY,
    SOTA_GI_MODE_PATH,
    SOTA_GI_MODE_WRC,
    sota_chebyshev_cpu,
    sota_fib_dir_cpu,
    sota_gi_resolve_shader,
    sota_sh9_irradiance_over_pi_cpu,
    sota_sh9_project_cpu,
    sota_wrc_slot_cpu,
    sota_wrc_update_shader,
)


def _assert(cond: Bool, msg: String) raises:
    if not cond:
        print("  [FAIL] " + msg)
        raise Error(msg)


def _validate(words: List[UInt32], name: String) raises:
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _assert(False, name + " SPIR-V validation failed: " + rep.errors[0])


def _f3(x: Float64) -> String:
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


def render_cornell_reference(w: Int, h: Int, spp: Int, sun_dir: Vec3, sun_rad: Vec3, indirect_bounces: Int) -> List[UInt32]:
    """Unbiased diffuse path tracer with NEE of the point sun, `indirect_bounces` bounces after the primary
    hit (2 = the v5 reference truncation; 12 = converged for these albedos to < 0.5% energy)."""
    var out = List[UInt32](length=w * h, fill=0xFF000000)
    var cam = Vec3(0.0, 0.0, -3.5)
    var tan_y = tan(Float32(50.0 * 3.14159265 / 360.0))
    var tan_x = tan_y * (Float32(w) / Float32(h))
    for py in range(h):
        var ndc_y = Float32(1.0) - ((Float32(py) + Float32(0.5)) / Float32(h)) * Float32(2.0)
        for px in range(w):
            var ndc_x = ((Float32(px) + Float32(0.5)) / Float32(w)) * Float32(2.0) - Float32(1.0)
            var rd0 = Vec3(ndc_x * tan_x, ndc_y * tan_y, Float32(1.0)).normalized()
            var h0 = _cornell_intersect(cam, rd0)
            if not h0[0]:
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
                for b in range(indirect_bounces):
                    var u1 = _pcg(st)
                    if b == 0:
                        u1 = (Float32(s_i) + u1) / Float32(spp)
                    var u2 = _pcg(st)
                    var d = _cos_dir(n, u1, u2)
                    var hh = _cornell_intersect(p + n * Float32(0.005), d)
                    if not hh[0]:
                        break
                    var a = _albedo(hh[4])
                    var dl = _direct(hh[2], hh[3], a, sun_dir, sun_rad)
                    acc = acc + Vec3(tp.x * dl.x, tp.y * dl.y, tp.z * dl.z)
                    tp = Vec3(tp.x * a.x, tp.y * a.y, tp.z * a.z)
                    p = hh[2]
                    n = hh[3]
                gi = gi + acc
            gi = gi * (Float32(1.0) / Float32(spp))
            out[py * w + px] = _to_srgb(l_dir + Vec3(alb0.x * gi.x, alb0.y * gi.y, alb0.z * gi.z))
    return out^


def _add_flat_quad(mut mb: MeshBuilder, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: Vec3):
    var i0 = mb.add_vertex(p0, n, 0.0, 0.0)
    var i1 = mb.add_vertex(p1, n, 1.0, 0.0)
    var i2 = mb.add_vertex(p2, n, 1.0, 1.0)
    var i3 = mb.add_vertex(p3, n, 0.0, 1.0)
    mb.add_quad(i0, i1, i2, i3)


def _build_cornell(mut ren: FlintAaaVulkanRenderer) -> List[Int]:
    var m_white = AaaMaterial(base_color=Vec3(0.76, 0.76, 0.76), roughness=1.0, metallic=0.0)
    m_white.specular_factor = Float32(0.0)
    var m_red = AaaMaterial(base_color=Vec3(0.80, 0.08, 0.08), roughness=1.0, metallic=0.0)
    m_red.specular_factor = Float32(0.0)
    var m_green = AaaMaterial(base_color=Vec3(0.08, 0.78, 0.12), roughness=1.0, metallic=0.0)
    m_green.specular_factor = Float32(0.0)
    var id_w = ren.add_material(m_white)
    var id_r = ren.add_material(m_red)
    var id_g = ren.add_material(m_green)
    var b_flr = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
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
    ids.append(ren.add_mesh(b_flr^.finish(), id_w).mesh_id)
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
                if nm == "SotaWorldRadianceCache" or nm == "SotaGiResolve":
                    gi += t.gpu_ms
                    rs += t.gpu_ms
            gi_t.append(gi)
            rs_t.append(rs)
            fr_t.append(ren.last_frame_gpu_ms)
    run.pixels = ren.readback_pixels()
    run.gi_ms = _median(gi_t)
    run.frame_ms = _median(fr_t)
    run.resolve_ms = _median(rs_t)


def _png_dir() -> String:
    var d = env_now("FLINT_SOTA_GI_PNG_DIR")
    if d.byte_length() == 0:
        d = env_now("HOME") + "/flint-models/_eval/render/B"
    _ = make_dirs(d)
    return d


def test_gpu_cornell_gi_modes() raises:
    var cw = 128
    var ch = 128
    var ren = FlintAaaVulkanRenderer(width=cw, height=ch)
    if not ren.active or not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [SKIP] SOTA GI GPU checks (no hardware ray queries)")
        return
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
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
    var sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var sun_rad = Vec3(3.1, 3.0, 2.9)
    ren.sun_dir = sun_dir
    ren.sun_radiance = sun_rad
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

    # (c) world radiance cache only (0 per-pixel GI rays), probes every 0.5 m
    ren.sota_gi.wrc_center_fixed = True
    ren.sota_gi.wrc_center = Vec3(0.0, 0.0, 2.0)
    ren.sota_gi.wrc_spacing = Float32(0.5)
    ren.sota_gi.wrc_dims_x = 10
    ren.sota_gi.wrc_dims_y = 10
    ren.sota_gi.wrc_dims_z = 10
    ren.sota_gi.wrc_max_history = 64
    var r_c = GiRun("sota_wrc_only_32f")
    ren.sota_gi.mode = SOTA_GI_MODE_WRC
    _render_frames(ren, ids, 32, r_c)
    runs.append(r_c^)

    # (d) hybrid: 4 surface-cache rays per pixel, WRC behind them and as the surface-cache bounce
    var r_d = GiRun("sota_hybrid_sc4_wrc_32f")
    ren.sota_gi.mode = SOTA_GI_MODE_HYBRID
    ren.sota_gi.pixel_rays = 4
    _render_frames(ren, ids, 32, r_d)
    runs.append(r_d^)

    # (e) real-time path tracer: 1 path / pixel, 2 bounces + WRC termination, SVGF + TSR
    var r_e = GiRun("sota_pt1spp_wrc_svgf_32f")
    ren.sota_gi.mode = SOTA_GI_MODE_PATH
    ren.sota_gi.pixel_rays = 1
    ren.sota_gi.pt_bounces = 2
    ren.sota_gi.pt_wrc_terminate = True
    _render_frames(ren, ids, 32, r_e)
    runs.append(r_e^)

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
        var pad = runs[i].name
        while pad.byte_length() < 34:
            pad += " "
        print("    " + pad + _f3(runs[i].flip_v5) + "         " + _f3(runs[i].flip_full) + "          " + _f3(runs[i].gi_ms) + "   " + _f3(runs[i].resolve_ms) + "      " + _f3(runs[i].frame_ms))

    # Correctness: the 2-bounce no-cache path tracer converges to the v5 2-bounce reference
    _assert(runs[6].flip_v5 < 0.05, "photo-mode 2-bounce PT vs 2-bounce reference FLIP " + _f3(runs[6].flip_v5))
    # The multi-bounce photo mode beats the v5 baseline against the converged reference
    _assert(runs[5].flip_full < runs[0].flip_full, "photo-mode PT should beat the v5 baseline")
    _assert(runs[5].flip_full < 0.06, "photo-mode PT vs converged reference FLIP " + _f3(runs[5].flip_full))
    # Real-time modes: better than the v5 real-time budget against the converged reference
    _assert(runs[4].flip_full < runs[1].flip_full, "RT path tracer should beat v5 4-ray GI")
    # Measured medians at 512 spp: WRC-only 0.150, hybrid 0.124 (v5 baseline 0.262).
    # The remaining WRC error is probe-resolution bias (missing contact occlusion),
    # not noise, so the budget is the measured value plus a margin.
    _assert(runs[2].flip_full < 0.17, "WRC-only FLIP " + _f3(runs[2].flip_full))
    _assert(runs[3].flip_full < 0.15, "hybrid FLIP " + _f3(runs[3].flip_full))
    _assert(runs[2].flip_full < runs[0].flip_full, "WRC-only should beat the v5 baseline")
    _assert(runs[3].flip_full < runs[0].flip_full, "hybrid should beat the v5 baseline")
    # Perf budgets at 128x128 (generous: the GPU is shared)
    _assert(runs[2].resolve_ms < 2.0, "WRC-only area-B passes " + _f3(runs[2].resolve_ms) + " ms")
    _assert(runs[4].resolve_ms < 3.0, "RT path tracer area-B passes " + _f3(runs[4].resolve_ms) + " ms")
    ren.shutdown()
    print("  [PASS] SOTA GI Cornell modes vs 512-spp references (PNGs in " + dir + ")")


def main() raises:
    var t0 = get_time()
    print("=== Flint v5 Area B (GI) SOTA Suite ===")
    test_cpu_twins()
    test_gpu_cornell_gi_modes()
    print("=== ALL FLINT V5 SOTA GI TESTS PASSED in " + _f3(get_time() - t0) + "s ===")
