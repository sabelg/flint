"""Flint v5 sota:mat round 3: the layered BSDF outside the sun / split-sum path.

  1. SPIR-V: every shader the hooks touch validates (deferred, surface cache, RT sun shadow, layered graph).
  2. CPU: coat normal (n_c == n reproduces round 2, energy bound with a tilted coat), hemispherical albedo
     (white = 1, == cosine-weighted directional albedo), shadow transmittance.
  3. GPU probe: lobes + eval_light with a separate coat normal, albedo, shadow T == CPU twins.
  4. Clustered lights: point light and LTC rect light, GPU deferred vs exact CPU integration of the BSDF.
  5. RT hit shading: spheres seen in a mirror (RT reflections) vs seen directly from the mirrored camera.
  6. Transmission: a thin pane in front of the spheres (ratio through / direct ~ w_trans), ball-lens inversion
     (volume refracts, thin-walled does not), shadows through a tinted pane (~ T, legacy opaque).
  7. Material graph: layer outputs and the three specialisation constants (VkSpecializationInfo), layered bake.
  8. Cost of the round-3 hooks (deferred pass, clustered lights + RT on), cvar-off identity.
"""

from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.math import abs, cos, max, min, sin, sqrt
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import env_value, get_time, make_dirs
from flint.platform.vulkan import get_float, get32, put_float, ACCESS_HOST_READ, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, PIPELINE_HOST
from flint.render.aaa_bsdf import AaaMaterial, AAA_MATERIAL_WORDS
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_gi_svgf import aaa_deferred_shade_shader, surface_cache_update_shader
from flint.render.aaa_rt_shadow import rt_sun_shadow_trace_shader
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_MATERIALS, BIND_AAA_POST_SCRATCH
from flint.render.material_graph import (
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_TWO_SIDED_FOLIAGE,
    MAT_DOMAIN_UNLIT_UI,
    StandardPbrMaterialNode,
    compile_surface_material_shader,
)
from flint.render.sota_mat_core import (
    SOTA_MAT_COAT_N_GEOMETRIC,
    SOTA_MAT_EXT_BASE,
    SOTA_MAT_EXT_WORDS,
    SOTA_MAT_MAGIC,
    SotaMatInputs,
    SotaMatLayers,
    SotaMatLut,
    SotaMatParams,
    SotaMatSurfaceExtra,
    emit_sota_mat_albedo_avg,
    emit_sota_mat_eval_light,
    emit_sota_mat_lobes,
    emit_sota_mat_shadow_transmittance,
    sota_mat_albedo_avg_cpu,
    sota_mat_ambient_cpu,
    sota_mat_eval_cpu,
    sota_mat_eval_nc_cpu,
    sota_mat_params_from_words,
    sota_mat_shadow_transmittance_cpu,
)
from flint.render.sota_mat_graph import (
    BrushedSpunMetalNode,
    SotaMatGraphSpec,
    compile_sota_layered_material_shader,
    sota_mat_bake_layered_graph,
    sota_mat_graph_eval,
)
from flint.render.sota_mat_reference import SotaMatScene, SotaMatSphere, SotaRng


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _lum(c: Vec3) -> Float32:
    return c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)


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


def _enc(x: Float32) -> UInt32:
    var c = max(Float32(0.0), min(Float32(1.0), x))
    return UInt32((c ** Float32(1.0 / 2.2)) * Float32(255.0) + Float32(0.5))


def _save_hdr(name: String, img: List[Vec3], w: Int, h: Int, scale: Float32):
    var d = env_value("FLINT_SOTA_MAT_PNG_DIR")
    if d.byte_length() == 0:
        return
    _ = make_dirs(d)
    var px = List[UInt32](length=w * h, fill=0)
    for i in range(w * h):
        var c = img[i] * scale
        px[i] = UInt32(0xFF000000) | (_enc(c.x) << 16) | (_enc(c.y) << 8) | _enc(c.z)
    if write_png(d + "/" + name, px, w, h):
        print("    wrote", d + "/" + name)


def _words(m: AaaMaterial) -> List[UInt32]:
    var w = List[UInt32]()
    m.append_words(w)
    return w^


def _ext(l: SotaMatLayers) -> List[UInt32]:
    var w = List[UInt32]()
    l.append_words(w)
    return w^


def _params(m: AaaMaterial, l: SotaMatLayers) -> SotaMatParams:
    return sota_mat_params_from_words(_words(m), _ext(l))


struct _MatSet(Movable):
    var names: List[String]
    var mats: List[AaaMaterial]
    var layers: List[SotaMatLayers]

    def __init__(out self):
        self.names = List[String]()
        self.mats = List[AaaMaterial]()
        self.layers = List[SotaMatLayers]()

    def add(mut self, name: String, var m: AaaMaterial, var l: SotaMatLayers):
        self.names.append(name)
        self.mats.append(m^)
        self.layers.append(l^)

    def params(self, i: Int) -> SotaMatParams:
        return _params(self.mats[i], self.layers[i])


def _set(white: Bool) -> _MatSet:
    """Round-2 showcase layers plus a volume and a geometric-coat material."""
    var s = _MatSet()
    var wv = Vec3(1.0, 1.0, 1.0)
    s.add("plastic", AaaMaterial(base_color=wv if white else Vec3(0.75, 0.18, 0.12), roughness=0.4, metallic=0.0), SotaMatLayers())
    s.add("gold", AaaMaterial(base_color=wv if white else Vec3(1.0, 0.71, 0.29), roughness=0.35, metallic=1.0), SotaMatLayers())
    var m2 = AaaMaterial(base_color=wv if white else Vec3(0.04, 0.12, 0.55), roughness=0.5, metallic=0.0)
    m2.clearcoat = Float32(1.0)
    m2.clearcoat_roughness = Float32(0.08)
    var l2 = SotaMatLayers()
    l2.coat_normal_mode = SOTA_MAT_COAT_N_GEOMETRIC
    s.add("carpaint_geo_coat", m2^, l2^)
    var m3 = AaaMaterial(base_color=wv if white else Vec3(0.30, 0.03, 0.06), roughness=0.8, metallic=0.0)
    m3.sheen_color = wv if white else Vec3(1.0, 0.35, 0.45)
    m3.sheen_roughness = Float32(0.4)
    s.add("velvet", m3^, SotaMatLayers())
    var l4 = SotaMatLayers()
    l4.anisotropy = Float32(0.8)
    s.add("brushed_alu", AaaMaterial(base_color=wv if white else Vec3(0.91, 0.92, 0.92), roughness=0.3, metallic=1.0), l4^)
    var l5 = SotaMatLayers()
    l5.transmission = Float32(1.0)
    l5.transmission_tint = wv if white else Vec3(0.65, 0.88, 1.0)
    s.add("thin_glass", AaaMaterial(base_color=wv, roughness=0.15, metallic=0.0), l5^)
    var l6 = SotaMatLayers()
    l6.transmission = Float32(1.0)
    l6.transmission_tint = wv if white else Vec3(0.9, 0.55, 0.3)
    l6.volume_distance = Float32(0.6)
    s.add("amber_volume", AaaMaterial(base_color=wv, roughness=0.05, metallic=0.0), l6^)
    var m7 = AaaMaterial(base_color=wv if white else Vec3(0.55, 0.56, 0.58), roughness=0.45, metallic=1.0)
    m7.clearcoat = Float32(1.0)
    m7.clearcoat_roughness = Float32(0.05)
    var l7 = SotaMatLayers()
    l7.anisotropy = Float32(0.6)
    l7.anisotropy_rotation = Float32(0.785)
    s.add("coat_over_aniso", m7^, l7^)
    return s^


def _rand_unit(seed: Int) -> Vec3:
    var a = Float32((seed * 7919 + 13) % 1000) / Float32(1000.0)
    var c = Float32((seed * 104729 + 71) % 1000) / Float32(1000.0)
    var z = Float32(1.0) - Float32(2.0) * a
    var r = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
    var ph = c * Float32(6.2831853)
    return Vec3(r * cos(ph), r * sin(ph), z).normalized()


def _tilt(n: Vec3, axis: Vec3, ang: Float32) -> Vec3:
    var a = axis - n * n.dot(axis)
    if a.length() < Float32(1e-4):
        a = Vec3(1.0, 0.0, 0.0) if abs(n.x) < Float32(0.9) else Vec3(0.0, 1.0, 0.0)
        a = a - n * n.dot(a)
    a = a.normalized()
    return (n * cos(ang) + a * sin(ang)).normalized()


# ---------------------------------------------------------------------------------------------
# 1. SPIR-V
# ---------------------------------------------------------------------------------------------
def test_spirv():
    for rq in range(2):
        for lay in range(2):
            var tag = " (ray query " + String(rq) + ", layered variant " + String(lay) + ")"
            var d = aaa_deferred_shade_shader(rq == 1, sota_mat_layered=lay == 1)
            var r1 = validate_spirv_module(d)
            _assert(r1.valid, "deferred shader validates" + tag)
            var r2 = validate_spirv_module(surface_cache_update_shader(rq == 1, sota_mat_layered=lay == 1))
            _assert(r2.valid, "surface cache shader validates" + tag)
        # The default pipelines carry none of the layered code: the variant is strictly larger
        var d_off = len(aaa_deferred_shade_shader(rq == 1))
        var d_on = len(aaa_deferred_shade_shader(rq == 1, sota_mat_layered=True))
        print("    deferred shader words: default", d_off, " layered variant", d_on)
        _assert(d_on > d_off + 1000, "layered code compiled out of the default deferred shader")
    for lay in range(2):
        var r3 = validate_spirv_module(rt_sun_shadow_trace_shader(sota_mat_layered=lay == 1))
        _assert(r3.valid, "RT sun shadow shader validates (layered variant " + String(lay) + ")")
    _assert(len(rt_sun_shadow_trace_shader(sota_mat_layered=True)) > len(rt_sun_shadow_trace_shader()), "shadow variant differs")
    var g = compile_sota_layered_material_shader(BrushedSpunMetalNode())
    var r4 = validate_spirv_module(g)
    _assert(r4.valid, "layered material graph validates")
    var r5 = validate_spirv_module(compile_surface_material_shader(StandardPbrMaterialNode()))
    _assert(r5.valid, "legacy material graph validates")
    # OpDecorate %x SpecId n (opcode 71, decoration 1): the three constants are decorated
    var found = List[Bool](length=3, fill=False)
    var i = 5
    while i < len(g):
        var wc = Int(g[i] >> 16)
        var op = g[i] & 0xFFFF
        if op == 71 and wc >= 4 and g[i + 2] == 1 and g[i + 3] < 3:
            found[Int(g[i + 3])] = True
        i += max(1, wc)
    _assert(found[0] and found[1] and found[2], "layered graph decorates SpecId 0, 1, 2")
    print("  [PASS] Suite 1: SPIR-V (deferred, surface cache, RT sun shadow, layered + legacy graph; SpecId 0..2)")


# ---------------------------------------------------------------------------------------------
# 2. CPU
# ---------------------------------------------------------------------------------------------
def test_cpu(lut: SotaMatLut):
    var col = _set(False)
    var wht = _set(True)
    # 2a. n_c == n reproduces round 2 exactly
    var worst = Float32(0.0)
    for c in range(200):
        var p = col.params(c % len(col.names))
        var n = _rand_unit(c * 3 + 1)
        var v = _rand_unit(c * 3 + 2)
        if v.dot(n) < Float32(0.05):
            v = (v + n * Float32(1.2)).normalized()
        var l = _rand_unit(c * 3 + 3)
        var t = Vec3(1.0, 0.0, 0.0) if abs(n.x) < Float32(0.9) else Vec3(0.0, 1.0, 0.0)
        var b = n.cross(t).normalized()
        var a = sota_mat_eval_cpu(p, lut, n, t, b, v, l)
        var e = sota_mat_eval_nc_cpu(p, lut, n, n, t, b, v, l)
        worst = max(worst, abs(a.x - e.x) + abs(a.y - e.y) + abs(a.z - e.z))
    print("    n_c == n vs round-2 eval, max |diff| =", worst)
    _assert(worst < Float32(1e-6), "coat normal == shading normal reproduces round 2")
    # 2b. energy with a tilted coat (white coat over white plastic, MC over the sphere of directions)
    var mc = SotaMatParams()
    mc.albedo = Vec3(1.0, 1.0, 1.0)
    mc.roughness = Float32(0.5)
    mc.clearcoat = Float32(1.0)
    mc.clearcoat_roughness = Float32(0.1)
    var worst_e = Float32(0.0)
    for k in range(4):
        var n = Vec3(0.0, 0.0, 1.0)
        var nc = _tilt(n, Vec3(1.0, 0.0, 0.0), Float32(0.1) * Float32(k + 1))
        var v = _tilt(n, Vec3(0.0, 1.0, 0.0), Float32(0.3) * Float32(k))
        var rng = SotaRng(UInt64(77 + k))
        var acc = Float64(0.0)
        var ns = 60000
        for _ in range(ns):
            var z = rng.next()
            var r = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
            var ph = rng.next() * Float32(6.2831853)
            var l = Vec3(r * cos(ph), r * sin(ph), z)
            var f = sota_mat_eval_nc_cpu(mc, lut, n, nc, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), v, l)
            acc += Float64(_lum(f)) * 6.2831853  # pdf = 1 / 2pi (upper hemisphere)
        var albedo = Float32(acc / Float64(ns))
        print("    tilted coat (", Float32(0.1) * Float32(k + 1), "rad ): directional albedo", albedo)
        worst_e = max(worst_e, albedo)
    _assert(worst_e < Float32(1.03), "a tilted coat normal does not create energy (white coat + white base <= 1)")
    # 2c. hemispherical albedo
    var worst_w = Float32(0.0)
    var worst_c = Float32(0.0)
    for i in range(len(wht.names)):
        var pw = wht.params(i)
        var aw = sota_mat_albedo_avg_cpu(pw, lut)
        if wht.layers[i].transmission == Float32(0.0):  # the bounce albedo is the reflected part only
            worst_w = max(worst_w, abs(_lum(aw) - Float32(1.0)))
        var pc = col.params(i)
        var ac = sota_mat_albedo_avg_cpu(pc, lut)
        # 2 int_0^1 E(mu) mu dmu with E(mu) = the split-sum weights' sum (white environment)
        var acc = Vec3(0.0, 0.0, 0.0)
        var nm = 64
        for k in range(nm):
            var mu = (Float32(k) + Float32(0.5)) / Float32(nm)
            var am = sota_mat_ambient_cpu(pc, lut, mu)
            var e = am.w_spec + am.w_diff + Vec3(am.w_cc, am.w_cc, am.w_cc)  # reflection only (no w_trans)
            acc = acc + e * (Float32(2.0) * mu / Float32(nm))
        var rel = abs(_lum(ac) - _lum(acc)) / max(Float32(0.02), _lum(acc))
        print("    albedo", col.names[i], ": hemispherical", _lum(ac), " cos-avg of reflected E(mu)", _lum(acc), " white", _lum(aw))
        worst_c = max(worst_c, rel)
    _assert(worst_w < Float32(0.01), "white opaque hemispherical albedo == 1 within 1%")
    _assert(worst_c < Float32(0.04), "hemispherical albedo == cosine-weighted directional albedo within 4%")
    # 2d. shadow transmittance
    var t_glass = sota_mat_shadow_transmittance_cpu(wht.params(5))
    var t_plastic = sota_mat_shadow_transmittance_cpu(wht.params(0))
    var t_amber = sota_mat_shadow_transmittance_cpu(col.params(6))
    print("    shadow T: white thin glass", t_glass, " plastic", t_plastic, " amber volume", t_amber)
    _assert(abs(t_glass - Float32(0.96)) < Float32(1e-3) and t_plastic == Float32(0.0), "shadow transmittance")
    print("  [PASS] Suite 2: CPU coat normal, hemispherical albedo, shadow transmittance")


# ---------------------------------------------------------------------------------------------
# 3. GPU probe
# ---------------------------------------------------------------------------------------------
comptime PROBE_IN: Int = 20
comptime PROBE_OUT: Int = 12
comptime PROBE_CASES: Int = 256


def _probe_shader(in_off: Int, out_off: Int) -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_POST_SCRATCH)
    var bm = s.buffer(BIND_AAA_MATERIALS)
    var i = s.invocation_axis(0)
    s.return_if(s.uge(i, s.uint(UInt32(PROBE_CASES))))
    var base = s.iadd(s.uint(UInt32(in_off)), s.imul(i, s.uint(UInt32(PROBE_IN))))
    var v = List[UInt32]()
    for k in range(PROBE_IN):
        v.append(s.load_float(b, s.iadd(base, s.uint(UInt32(k)))))
    var mat_id = s.uint_from_float(v[18])
    var mw = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var w = List[UInt32]()
    for k in range(AAA_MATERIAL_WORDS):
        w.append(s.load_float(bm, s.iadd(mw, s.uint(UInt32(k)))))
    var yes = s.ieq(s.uint(1), s.uint(1))
    var inp = SotaMatInputs(
        v[0], v[1], v[2], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17],
        w[0], w[1], w[2], w[5], s.fabs(w[3]), w[4], w[6], w[7],
        w[16], w[17], w[18], w[19], w[20], w[21], w[22], w[13], w[14], w[15],
    )
    var x = SotaMatSurfaceExtra(v[3], v[4], v[5], 0, 0, 0, 0)
    var lb = emit_sota_mat_lobes(s, bm, mat_id, yes, inp, x)
    var f = emit_sota_mat_eval_light(s, bm, lb, v[9], v[10], v[11], True)
    var fp = emit_sota_mat_eval_light(s, bm, lb, v[9], v[10], v[11], False)
    var a = emit_sota_mat_albedo_avg(s, bm, lb)
    var t = emit_sota_mat_shadow_transmittance(s, bm, mat_id)
    var ob = s.iadd(s.uint(UInt32(out_off)), s.imul(i, s.uint(UInt32(PROBE_OUT))))
    var outs = [f[0], f[1], f[2], fp[0], fp[1], fp[2], a[0], a[1], a[2], t]
    for k in range(len(outs)):
        s.store_float(b, s.iadd(ob, s.uint(UInt32(k))), outs[k])
    return s.finish()


def run_probe(mut ren: FlintAaaVulkanRenderer, s: _MatSet, ids: List[Int], lut: SotaMatLut) -> Float32:
    ren.flush_deferred_frame()
    ren.vk.wait_idle()
    var in_off = ren.sota_e_hair.base_word
    var out_off = in_off + PROBE_CASES * PROBE_IN
    var code = _probe_shader(in_off, out_off)
    _assert(validate_spirv_module(code).valid, "probe shader validates")
    var pipe = ren.vk.create_aaa_compute_pipeline(ren.pipe_layout, code)
    _assert(pipe != 0, "probe pipeline created")
    var addr = ren.buf_post_scratch.cpu_address
    var ins = List[List[Float32]]()
    for c in range(PROBE_CASES):
        var n = _rand_unit(c * 5 + 1)
        var nc = _tilt(n, _rand_unit(c * 5 + 9), Float32(0.05) + Float32(c % 7) * Float32(0.05))
        var vv = _rand_unit(c * 5 + 2)
        if vv.dot(n) < Float32(0.1) or vv.dot(nc) < Float32(0.1):
            vv = (n + nc + vv * Float32(0.3)).normalized()
        var l = _rand_unit(c * 5 + 3)
        var t = _rand_unit(c * 5 + 4)
        var bb = n.cross(t)
        if bb.length() < Float32(1e-3):
            t = Vec3(1.0, 0.0, 0.0) if abs(n.x) < Float32(0.9) else Vec3(0.0, 1.0, 0.0)
            bb = n.cross(t)
        if c % 2 == 1:
            bb = bb * Float32(-1.0)
        var row: List[Float32] = [n.x, n.y, n.z, nc.x, nc.y, nc.z, vv.x, vv.y, vv.z, l.x, l.y, l.z, t.x, t.y, t.z, bb.x, bb.y, bb.z,
                   Float32(ids[c % len(ids)]), Float32(0.0)]
        for k in range(PROBE_IN):
            put_float(addr + (in_off + c * PROBE_IN + k) * 4, row[k])
        ins.append(row^)
    for c in range(PROBE_CASES * PROBE_OUT):
        put_float(addr + (out_off + c) * 4, Float32(-12345.0))
    _ = ren.vk.begin_commands(0)
    var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
    ren.vk.dispatch_aaa(0, pipe, ren.pipe_layout, ren.desc_sets[0], push, (PROBE_CASES + 63) // 64, 1)
    ren.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
    _ = ren.vk.wait(ren.vk.submit(0))
    var worst = Float32(0.0)
    for c in range(PROBE_CASES):
        var row = ins[c].copy()
        var mi = c % len(ids)
        var p = s.params(mi)
        var n = Vec3(row[0], row[1], row[2])
        var nc = Vec3(row[3], row[4], row[5])
        var vv = Vec3(row[6], row[7], row[8])
        var l = Vec3(row[9], row[10], row[11])
        var t = Vec3(row[12], row[13], row[14])
        var bb = Vec3(row[15], row[16], row[17])
        var f = sota_mat_eval_nc_cpu(p, lut, n, nc, t, bb, vv, l, True)
        var fp = sota_mat_eval_nc_cpu(p, lut, n, nc, t, bb, vv, l, False)
        var a = sota_mat_albedo_avg_cpu(p, lut)
        var tr = sota_mat_shadow_transmittance_cpu(p)
        var refs = [f.x, f.y, f.z, fp.x, fp.y, fp.z, a.x, a.y, a.z, tr]
        for k in range(len(refs)):
            var o = get_float(addr + (out_off + c * PROBE_OUT + k) * 4)
            var e = abs(o - refs[k]) / max(Float32(0.02), abs(refs[k]))
            if e > worst:
                worst = e
                if e > Float32(2e-3):
                    print("      case", c, s.names[mi], "out", k, "gpu", o, "cpu", refs[k])
    ren.vk.destroy_pipeline(pipe)
    return worst


# ---------------------------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------------------------
def _sphere_mesh() -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=64, segments=96)
    var m = mb^.finish()
    for i in range(m.vertex_count()):
        m.texcoords[i * 2] = Float32(0.5) + Float32(0.5) * m.positions[i * 3]
        m.texcoords[i * 2 + 1] = Float32(0.5) - Float32(0.5) * m.positions[i * 3 + 1]
    return m^


def _quad_mesh(c: Vec3, u: Vec3, v: Vec3) -> NativeMesh:
    """One-sided quad centre `c`, half axes `u`, `v`; normal = u x v."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    var n = u.cross(v).normalized()
    var a = mb.add_vertex(c - u - v, n, 0.0, 1.0)
    var b = mb.add_vertex(c + u - v, n, 1.0, 1.0)
    var d = mb.add_vertex(c + u + v, n, 1.0, 0.0)
    var e = mb.add_vertex(c - u + v, n, 0.0, 0.0)
    mb.add_quad(a, b, d, e)
    return mb^.finish()


comptime GRID_R: Float32 = 0.42


def _grid_center(i: Int) -> Vec3:
    return Vec3(Float32(-1.5) + Float32(i % 4), Float32(0.5) if i < 4 else Float32(-0.5), Float32(0.0))


def _scene(s: _MatSet, cam: Vec3) -> SotaMatScene:
    var sc = SotaMatScene()
    for i in range(len(s.names)):
        sc.spheres.append(SotaMatSphere(_grid_center(i), GRID_R, s.params(i)))
    sc.cam_pos = cam
    sc.cam_target = Vec3(0.0, 0.0, 0.0)
    sc.fov_y_deg = Float32(30.0)
    return sc^


def _setup(mut ren: FlintAaaVulkanRenderer):
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_vsm = False
    ren.enable_rt_shadows = False
    ren.enable_rt_gi = False
    ren.enable_rt_reflections = False
    ren.enable_restir_di = False
    ren.enable_aerial_perspective = False


def _sky() -> SkyDomeParams:
    var sky_p = SkyDomeParams()
    sky_p.zenith_rgb = Vec3(0.30, 0.42, 0.62)
    sky_p.horizon_rgb = Vec3(0.55, 0.55, 0.52)
    sky_p.ground_rgb = Vec3(0.16, 0.13, 0.10)
    sky_p.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    return sky_p^


@fieldwise_init
struct _Inst(Copyable, Movable):
    var mesh: Int
    var center: Vec3
    var scale: Float32
    var mat: Int


def _render(mut ren: FlintAaaVulkanRenderer, insts: List[_Inst], cam: Vec3, target: Vec3, lights: List[AaaClusterLight]) -> List[Vec3]:
    var hdr = List[Vec3]()
    for _ in range(4):
        ren.begin_frame(cam_pos=cam, cam_target=target, fov_y_deg=30.0, near_z=0.05)
        for i in range(len(insts)):
            var k = insts[i].scale
            ren.draw_instance(insts[i].mesh, trs(insts[i].center, Quat(), Vec3(k, k, k)), material_override=insts[i].mat)
        for i in range(len(lights)):
            ren.add_cluster_light(lights[i])
        _ = ren.render_frame(capture_readback=True)
        hdr = ren.readback_resolved_hdr()
    return hdr^


def _grid(sph: Int, ids: List[Int]) -> List[_Inst]:
    var o = List[_Inst]()
    for i in range(len(ids)):
        o.append(_Inst(sph, _grid_center(i), GRID_R, ids[i]))
    return o^


def _sphere_means(sc: SotaMatScene, img: List[Vec3], w: Int, h: Int, flip_x: Bool = False, light_pos: Vec3 = Vec3(0.0, 0.0, 0.0), mask_core: Bool = False) -> List[Vec3]:
    """Mean radiance over each sphere's interior pixels (n . v > 0.3) as seen from `sc.cam_pos`; `flip_x`
    reads the image mirrored (a view through a mirror). `mask_core` skips the mirror core of a point light at
    `light_pos` (n . h > 0.995), where a 1-2 px highlight of the smooth coats is point-sampled on a tessellated
    normal on the GPU and an analytic one on the CPU (as round 2 masks the sun core)."""
    var n_s = len(sc.spheres)
    var acc = List[Vec3](length=n_s, fill=Vec3(0.0, 0.0, 0.0))
    var cnt = List[Int](length=n_s, fill=0)
    for py in range(h):
        for px in range(w):
            var d = sc.primary_dir(px, py, w, h, Float32(0.5), Float32(0.5))
            var hh = sc.hit(sc.cam_pos, d)
            if hh[0] < 0:
                continue
            var p = sc.cam_pos + d * hh[1]
            var n = ((p - sc.spheres[hh[0]].center) * (Float32(1.0) / sc.spheres[hh[0]].radius)).normalized()
            if n.dot(d * Float32(-1.0)) < Float32(0.3):
                continue
            if mask_core and n.dot(((light_pos - p).normalized() - d).normalized()) > Float32(0.995):
                continue
            var qx = (w - 1 - px) if flip_x else px
            acc[hh[0]] = acc[hh[0]] + img[py * w + qx]
            cnt[hh[0]] += 1
    var o = List[Vec3]()
    for i in range(n_s):
        o.append(acc[i] * (Float32(1.0) / Float32(max(1, cnt[i]))))
    return o^


def _cpu_direct(sc: SotaMatScene, lut: SotaMatLut, w: Int, h: Int, light: AaaClusterLight, rect_n: Int) -> List[Vec3]:
    """Exact direct light of one clustered light on the spheres (no occlusion, like the GPU with shadows off):
    point: L * window(d) * f cos; rect (rect_n^2 stratified points): sum L f cos cos_e / d^2 dA."""
    var img = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 0.0))
    for py in range(h):
        for px in range(w):
            var d = sc.primary_dir(px, py, w, h, Float32(0.5), Float32(0.5))
            var hh = sc.hit(sc.cam_pos, d)
            if hh[0] < 0:
                continue
            var p = sc.cam_pos + d * hh[1]
            var fr = sc.frame_at(hh[0], p)
            var pm = sc.spheres[hh[0]].params.copy()
            var v = d * Float32(-1.0)
            var acc = Vec3(0.0, 0.0, 0.0)
            if rect_n == 0:
                var rel = light.position - p
                var d2 = max(Float32(0.01), rel.dot(rel))
                var q = d2 / max(Float32(0.01), light.radius * light.radius)
                var q2 = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - q * q))
                var att = q2 * q2 / (d2 + Float32(0.25))
                var l = rel * (Float32(1.0) / sqrt(d2))
                if fr[0].dot(l) > Float32(0.0001):
                    var f = sota_mat_eval_nc_cpu(pm, lut, fr[0], fr[0], fr[1], fr[2], v, l, False)
                    acc = Vec3(light.radiance.x * f.x, light.radiance.y * f.y, light.radiance.z * f.z) * att
            else:
                var ln = light.axis_u.cross(light.axis_v).normalized()
                var da = Float32(4.0) * light.axis_u.length() * light.axis_v.length() / Float32(rect_n * rect_n)
                for iu in range(rect_n):
                    for iv in range(rect_n):
                        var su = (Float32(iu) + Float32(0.5)) / Float32(rect_n) * Float32(2.0) - Float32(1.0)
                        var sv = (Float32(iv) + Float32(0.5)) / Float32(rect_n) * Float32(2.0) - Float32(1.0)
                        var q = light.position + light.axis_u * su + light.axis_v * sv
                        var rel = q - p
                        var d2 = max(Float32(1e-4), rel.dot(rel))
                        var l = rel * (Float32(1.0) / sqrt(d2))
                        var ce = -l.dot(ln)
                        if ce <= Float32(0.0):
                            continue
                        var f = sota_mat_eval_nc_cpu(pm, lut, fr[0], fr[0], fr[1], fr[2], v, l, False)
                        acc = acc + f * (ce * da / d2)
                acc = Vec3(acc.x * light.radiance.x, acc.y * light.radiance.y, acc.z * light.radiance.z)
            img[py * w + px] = acc
    return img^


def _rel(a: Vec3, b: Vec3) -> Float32:
    return abs(_lum(a) - _lum(b)) / max(Float32(1e-4), _lum(b))


# ---------------------------------------------------------------------------------------------
# GPU suites
# ---------------------------------------------------------------------------------------------
def test_gpu(lut: SotaMatLut):
    var w = 192
    var h = 144
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren)
    var sky = _sky()
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky, base_dim=64, mip_levels=6))
    ren.begin_frame()
    _ = ren.render_frame(capture_readback=True)
    ren.sota_mat.enabled = True
    ren.sota_mat.prepare(ren.buf_materials.cpu_address)
    var col = _set(False)
    var wht = _set(True)
    var ids = List[Int]()
    var ids_w = List[Int]()
    for i in range(len(col.names)):
        ids.append(ren.add_material(col.mats[i]))
        ren.sota_mat_set_layers(ids[i], col.layers[i])
    for i in range(len(wht.names)):
        ids_w.append(ren.add_material(wht.mats[i]))
        ren.sota_mat_set_layers(ids_w[i], wht.layers[i])

    # 3. probe
    var e_c = run_probe(ren, col, ids, lut)
    var e_w = run_probe(ren, wht, ids_w, lut)
    print("    GPU vs CPU (eval with coat normal, back / front only, hemispherical albedo, shadow T) max rel err:", e_c, e_w)
    _assert(e_c < Float32(5e-3) and e_w < Float32(5e-3), "GPU lobes / eval_light / albedo / T match the CPU twins")
    print("  [PASS] Suite 3: GPU probe (coat normal, eval_light, albedo, shadow transmittance)")

    var sph = ren.add_mesh(_sphere_mesh(), ids[0]).mesh_id
    var none = List[AaaClusterLight]()
    var cam = Vec3(0.0, 0.0, -6.0)
    var org = Vec3(0.0, 0.0, 0.0)
    var sc = _scene(col, cam)

    # 4. clustered lights (black environment, no sun)
    ren.ibl_intensity = Float32(0.0)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    var pl = AaaClusterLight.point(Vec3(1.1, 1.4, -2.6), Vec3(9.0, 9.0, 9.0), 40.0)
    var rl = AaaClusterLight.rect_ltc(Vec3(-0.4, 2.0, -2.2), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.8, 0.6), 0.9, 0.45, Vec3(4.0, 4.0, 4.0), 40.0)
    var names = ["point", "ltc_rect"]
    var mats_ok = List[Int]()  # spheres the clustered comparison covers: all but the transmissive pair
    for i in range(len(col.names)):
        if col.layers[i].transmission == Float32(0.0):
            mats_ok.append(i)
    for li in range(2):
        var lights = List[AaaClusterLight]()
        lights.append(pl.copy() if li == 0 else rl.copy())
        var cpu = _cpu_direct(sc, lut, w, h, lights[0], 0 if li == 0 else 14)
        ren.sota_mat.enabled = True
        var g_on = _render(ren, _grid(sph, ids), cam, org, lights)
        ren.sota_mat.enabled = False
        var g_off = _render(ren, _grid(sph, ids), cam, org, lights)
        var mk = False  # masking the point-light core does not help: the GPU's Kaplanyan specular AA widens the
        # highlights of these small spheres (roughness up at high curvature), the CPU integral uses the material value
        var m_cpu = _sphere_means(sc, cpu, w, h, False, pl.position, mk)
        var m_on = _sphere_means(sc, g_on, w, h, False, pl.position, mk)
        var m_off = _sphere_means(sc, g_off, w, h, False, pl.position, mk)
        var worst_on = Float32(0.0)
        var worst_off = Float32(0.0)
        var regress = Float32(-1.0)
        for k in range(len(mats_ok)):
            var i = mats_ok[k]
            regress = max(regress, _rel(m_on[i], m_cpu[i]) - _rel(m_off[i], m_cpu[i]))
            print("    ", names[li], col.names[i], ": CPU exact", _lum(m_cpu[i]), " layered", _lum(m_on[i]), "(err", _rel(m_on[i], m_cpu[i]), ")  legacy", _lum(m_off[i]), "(err", _rel(m_off[i], m_cpu[i]), ")")
            worst_on = max(worst_on, _rel(m_on[i], m_cpu[i]))
            worst_off = max(worst_off, _rel(m_off[i], m_cpu[i]))
        print("    ", names[li], "worst per-sphere error: layered", worst_on, " legacy", worst_off)
        _save_hdr("clustered_" + String(names[li]) + "_layered.png", g_on, w, h, Float32(1.0))
        _save_hdr("clustered_" + String(names[li]) + "_legacy.png", g_off, w, h, Float32(1.0))
        _save_hdr("clustered_" + String(names[li]) + "_cpu.png", cpu, w, h, Float32(1.0))
        if li == 0:
            _assert(worst_on < Float32(0.08) and worst_on < worst_off, "point light: layered GPU within 8% of the exact CPU BSDF integral (closer than legacy)")
        else:
            # The LTC specular form factor (shared, pre-existing) returns ~15% of the exact polygon integral for these
            # glossy metals in legacy and layered alike (open item for the LTC owner); the layered weights must not
            # make any sphere worse and fix the sheen sphere (index 3).
            print("     LTC worst regression layered - legacy:", regress)
            _assert(regress < Float32(0.02) and _rel(m_on[3], m_cpu[3]) < Float32(0.06), "LTC rect: layered weights no worse than legacy anywhere, sheen within 6%")
    print("  [PASS] Suite 4: clustered point + LTC rect lights use the layered BSDF")

    ren.ibl_intensity = Float32(1.0)
    var sun_dir = Vec3(0.35, 0.75, -0.55).normalized()
    ren.sun_dir = sun_dir
    ren.sun_radiance = Vec3(2.5, 2.4, 2.2)

    if not ren.vk.has_ray_query:
        print("  [SKIP] Suites 5-6: no ray query on this device")
        ren.shutdown()
        return

    # 5. RT hit shading: the grid in a mirror vs the grid from the mirrored camera
    ren.enable_rt_reflections = True
    var mirror_mat = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.0, metallic=1.0))
    var mir = ren.add_mesh(_quad_mesh(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)), mirror_mat).mesh_id
    var cam_m = Vec3(0.0, 0.0, -10.0)
    var sc_m = _scene(col, cam_m)
    var errs = List[Float32]()
    for mode in range(2):
        ren.sota_mat.enabled = mode == 0
        var direct = _render(ren, _grid(sph, ids), cam_m, org, none)
        var insts = _grid(sph, ids)
        insts.append(_Inst(mir, Vec3(0.0, 0.0, -8.0), Float32(9.0), mirror_mat))
        var refl = _render(ren, insts, cam, Vec3(0.0, 0.0, -20.0), none)
        var md = _sphere_means(sc_m, direct, w, h)
        var mr = _sphere_means(sc_m, refl, w, h, True)
        var worst = Float32(0.0)
        for i in range(len(col.names)):
            if col.layers[i].transmission > Float32(0.0):
                continue  # the direct view of a transmissive sphere has a traced background the mirror view lacks
            worst = max(worst, _rel(mr[i], md[i]))
            print("    mirror", "layered" if mode == 0 else "legacy", col.names[i], ": direct", _lum(md[i]), " via RT hit", _lum(mr[i]), " err", _rel(mr[i], md[i]))
        errs.append(worst)
        _save_hdr("mirror_" + ("layered" if mode == 0 else "legacy") + "_direct.png", direct, w, h, Float32(1.0))
        _save_hdr("mirror_" + ("layered" if mode == 0 else "legacy") + "_reflected.png", refl, w, h, Float32(1.0))
    print("    RT hit vs primary, worst per-sphere error: layered", errs[0], " legacy", errs[1])
    _assert(errs[0] < Float32(0.12) and errs[0] < errs[1], "RT-hit shading matches the primary view (layered within 12%, closer than legacy)")
    print("  [PASS] Suite 5: RT-hit shading uses the layered model")

    # 6a. thin pane in front of the grid: through / direct ~ w_trans (black environment, sun not on the pane)
    ren.ibl_intensity = Float32(0.0)
    var pane_l = SotaMatLayers()
    pane_l.transmission = Float32(1.0)
    var pane_m = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.03, metallic=0.0))
    ren.sota_mat_set_layers(pane_m, pane_l)
    var pane = ren.add_mesh(_quad_mesh(Vec3(0.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)), pane_m).mesh_id
    ren.sun_dir = Vec3(0.3, 0.5, -0.8).normalized()
    var ratios = List[Float32]()
    for mode in range(2):
        ren.sota_mat.enabled = mode == 0
        var direct = _render(ren, _grid(sph, ids), cam, org, none)
        var insts = _grid(sph, ids)
        insts.append(_Inst(pane, Vec3(0.0, 0.0, -1.5), Float32(4.0), pane_m))
        var thru = _render(ren, insts, cam, org, none)
        var md = _sphere_means(sc, direct, w, h)
        var mt = _sphere_means(sc, thru, w, h)
        var num = Float32(0.0)
        var den = Float32(0.0)
        for i in range(len(col.names)):
            if col.layers[i].transmission > Float32(0.0):
                continue
            num += _lum(mt[i])
            den += _lum(md[i])
        ratios.append(num / max(Float32(1e-6), den))
        _save_hdr("pane_" + ("layered" if mode == 0 else "legacy") + ".png", thru, w, h, Float32(1.0))
        _save_hdr("pane_" + ("layered" if mode == 0 else "legacy") + "_none.png", direct, w, h, Float32(1.0))
    var pp = _params(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.03, metallic=0.0), pane_l)
    var wt = sota_mat_ambient_cpu(pp, lut, Float32(0.99)).w_trans
    print("    pane: through / direct luminance  layered", ratios[0], " (w_trans at normal incidence", _lum(wt), ")  legacy", ratios[1])
    _assert(abs(ratios[0] - _lum(wt)) < Float32(0.08), "layered pane transmits ~ w_trans of what is behind it")

    # 6b. ball lens: a volume sphere inverts the red | blue backdrop, a thin-walled one does not
    ren.sota_mat.enabled = True
    var red = AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
    red.emissive = Vec3(3.0, 0.15, 0.1)
    var blue = AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0)
    blue.emissive = Vec3(0.1, 0.2, 3.0)
    var red_m = ren.add_material(red)
    var blue_m = ren.add_material(blue)
    var lens_ids = List[Int]()
    for k in range(2):
        var lm = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.02, metallic=0.0))
        var ll = SotaMatLayers()
        ll.transmission = Float32(1.0)
        ll.volume_distance = Float32(20.0) if k == 0 else Float32(0.0)
        ren.sota_mat_set_layers(lm, ll)
        lens_ids.append(lm)
    var back = ren.add_mesh(_quad_mesh(Vec3(0.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)), red_m).mesh_id
    var lens_sc = SotaMatScene()
    lens_sc.spheres.append(SotaMatSphere(Vec3(0.0, 0.0, 0.0), Float32(1.0), SotaMatParams()))
    lens_sc.cam_pos = cam
    var lr = List[Float32]()
    for k in range(2):
        var insts = List[_Inst]()
        insts.append(_Inst(sph, Vec3(0.0, 0.0, 0.0), Float32(1.0), lens_ids[k]))
        insts.append(_Inst(back, Vec3(-2.0, 0.0, 3.0), Float32(2.0), red_m))
        insts.append(_Inst(back, Vec3(2.0, 0.0, 3.0), Float32(2.0), blue_m))
        var img = _render(ren, insts, cam, org, none)
        # left half of the ball (inner 60 % of the radius)
        var acc = Vec3(0.0, 0.0, 0.0)
        for py in range(h):
            for px in range(w):
                var d = lens_sc.primary_dir(px, py, w, h, Float32(0.5), Float32(0.5))
                var hh = lens_sc.hit(cam, d)
                if hh[0] < 0:
                    continue
                var p = cam + d * hh[1]
                if p.x < Float32(-0.15) and p.x > Float32(-0.6) and abs(p.y) < Float32(0.45):
                    acc = acc + img[py * w + px]
        lr.append(acc.x / max(Float32(1e-6), acc.z))
        _save_hdr("lens_" + ("volume" if k == 0 else "thin") + ".png", img, w, h, Float32(0.5))
    print("    ball lens, left half red / blue: volume", lr[0], " thin-walled", lr[1])
    _assert(lr[0] < Float32(0.5) and lr[1] > Float32(2.0), "volume transmission refracts (inverted image), thin-walled does not")

    # 6c. shadow through a tinted pane (RT sun shadows)
    ren.enable_rt_shadows = True
    var ground_m = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=1.0, metallic=0.0))
    var ground = ren.add_mesh(_quad_mesh(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0)), ground_m).mesh_id
    var tint_l = SotaMatLayers()
    tint_l.transmission = Float32(1.0)
    tint_l.transmission_tint = Vec3(0.5, 0.8, 1.0)
    var tint_m = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.05, metallic=0.0))
    ren.sota_mat_set_layers(tint_m, tint_l)
    var roof = ren.add_mesh(_quad_mesh(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)), tint_m).mesh_id
    ren.sun_dir = Vec3(0.0, 1.0, 0.0)
    var cam_g = Vec3(0.0, -0.3, -5.0)
    var tgt_g = Vec3(0.0, -1.0, 0.0)
    var gsc = SotaMatScene()
    gsc.cam_pos = cam_g
    gsc.cam_target = tgt_g
    var shadow_r = List[Float32]()
    for mode in range(2):
        ren.sota_mat.enabled = mode == 0
        var insts = List[_Inst]()
        insts.append(_Inst(ground, Vec3(0.0, -1.0, 0.0), Float32(4.0), ground_m))
        insts.append(_Inst(roof, Vec3(-2.0, 1.5, 0.0), Float32(2.0), tint_m))
        var img = _render(ren, insts, cam_g, tgt_g, none)
        var s_in = Float32(0.0)
        var n_in = 0
        var s_out = Float32(0.0)
        var n_out = 0
        for py in range(h):
            for px in range(w):
                var d = gsc.primary_dir(px, py, w, h, Float32(0.5), Float32(0.5))
                if d.y >= Float32(-1e-3):
                    continue
                var t = (Float32(-1.0) - cam_g.y) / d.y
                var p = cam_g + d * t
                if abs(p.z) > Float32(1.0):
                    continue
                if p.x < Float32(-0.6) and p.x > Float32(-1.6):
                    s_in += _lum(img[py * w + px])
                    n_in += 1
                elif p.x > Float32(0.6) and p.x < Float32(1.6):
                    s_out += _lum(img[py * w + px])
                    n_out += 1
        shadow_r.append((s_in / Float32(max(1, n_in))) / max(Float32(1e-6), s_out / Float32(max(1, n_out))))
        _save_hdr("shadow_pane_" + ("layered" if mode == 0 else "legacy") + ".png", img, w, h, Float32(1.0))
    var t_exp = sota_mat_shadow_transmittance_cpu(_params(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.05, metallic=0.0), tint_l))
    print("    shadow under a tinted pane, lit ratio: layered", shadow_r[0], " expected T", t_exp, "  legacy", shadow_r[1])
    _assert(abs(shadow_r[0] - t_exp) < Float32(0.06) and shadow_r[1] < Float32(0.1), "RT sun shadows pass T through layered transmissive surfaces")
    print("  [PASS] Suite 6: transmission (thin pane, ball-lens refraction, shadows through a pane)")
    ren.shutdown()


def test_graph():
    var ren = FlintAaaVulkanRenderer(width=64, height=64, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    var m = AaaMaterial(base_color=Vec3(0.8, 0.8, 0.82), roughness=0.3, metallic=1.0)
    m.clearcoat = Float32(0.5)
    var mid = ren.add_material(m)
    var node = BrushedSpunMetalNode(0.85, 0.42, 0.1)
    var d = 32
    # texel at (u, v) = (0.75 + .5/d, 0.5 + .5/d) -> du ~ 0.25, dv ~ 0: tangential rotation ~ pi/2
    var tx = (d * 3) // 4
    var ty = d // 2
    var a = sota_mat_graph_eval(ren, node, mid, d)
    _assert(a.dim == d, "layered graph evaluated")
    var o = (ty * d + tx) * 12
    print("    default spec: aniso", a.texels[o + 8], " rot/pi", a.texels[o + 9], " trans", a.texels[o + 10], " coat", a.texels[o + 5], " rough", a.texels[o + 3])
    _assert(abs(a.texels[o + 8] - Float32(0.85)) < Float32(1e-4) and abs(a.texels[o + 9] - Float32(0.5)) < Float32(0.03), "layer outputs")
    _assert(abs(a.texels[o + 5] - Float32(0.5)) < Float32(1e-4), "coat passes with SpecId 0 = true")
    var oc = ((d // 2) * d + d // 2) * 12
    _assert(a.texels[oc + 10] > Float32(0.99), "transmissive window at the centre")
    var b = sota_mat_graph_eval(ren, node, mid, d, SotaMatGraphSpec(False, 0, False))
    print("    clearcoat off + layers off: coat", b.texels[o + 5], " aniso", b.texels[o + 8], " max aniso", b.max_aniso)
    _assert(b.texels[o + 5] == Float32(0.0) and b.max_aniso == Float32(0.0) and b.max_trans == Float32(0.0), "SpecId 0 / 2 = false")
    var c = sota_mat_graph_eval(ren, node, mid, d, SotaMatGraphSpec(True, MAT_DOMAIN_UNLIT_UI, True))
    print("    domain UNLIT_UI: rough", c.texels[o + 3], " metal", c.texels[o + 4], " coat", c.texels[o + 5], " aniso", c.texels[o + 8])
    _assert(c.texels[o + 3] == Float32(1.0) and c.texels[o + 4] == Float32(0.0) and c.texels[o + 5] == Float32(0.0) and c.max_aniso == Float32(0.0), "SpecId 1 = UNLIT_UI")
    var e = sota_mat_graph_eval(ren, node, mid, d, SotaMatGraphSpec(True, MAT_DOMAIN_CLEARCOAT, True))
    var f = sota_mat_graph_eval(ren, node, mid, d, SotaMatGraphSpec(True, MAT_DOMAIN_TWO_SIDED_FOLIAGE, True))
    _assert(e.texels[o + 5] == Float32(1.0) and abs(f.texels[o + 10] - Float32(0.35)) < Float32(1e-4), "SpecId 1 = CLEARCOAT / FOLIAGE")
    # legacy graph bake now goes through VkSpecializationInfo too
    var lg = ren.evaluate_material_graph_to_bindless(StandardPbrMaterialNode(), mid, 16, domain=MAT_DOMAIN_UNLIT_UI)
    _assert(lg[0] >= 0 and lg[1] >= 0, "legacy graph bake with specialisation")
    # layered bake: albedo + ORM + layer mask, attached to the extension words
    var mid2 = ren.add_material(m)
    var bk = sota_mat_bake_layered_graph(ren, node, mid2, d)
    _assert(bk.albedo_slot >= 0 and bk.orm_slot >= 0 and bk.layer_slot >= 0, "layered bake uploads three textures")
    var eb = ren.buf_materials.cpu_address + (SOTA_MAT_EXT_BASE + mid2 * SOTA_MAT_EXT_WORDS) * 4
    _assert(get32(eb) == SOTA_MAT_MAGIC and Int(get32(eb + 9 * 4)) == bk.layer_slot, "layer mask slot in ext word 9")
    _assert(ren.sota_mat.transmissive[mid2], "baked window makes the material transmissive (TLAS any-hit)")
    print("  [PASS] Suite 7: material graph layer outputs + VkSpecializationInfo (SpecId 0 clearcoat, 1 domain, 2 layers)")
    ren.shutdown()


def test_perf():
    """Deferred pass at 1280x720, eight layered spheres, two point lights + one rect light, RT reflections and
    RT shadows on (when available): flag on vs off on alternating frames, medians (the GPU is shared)."""
    var w = 1280
    var h = 720
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(_sky(), base_dim=64, mip_levels=6))
    if ren.vk.has_ray_query:
        ren.enable_rt_reflections = True
        ren.enable_rt_shadows = True
    ren.sota_mat.enabled = True
    var col = _set(False)
    var ids = List[Int]()
    for i in range(len(col.names)):
        ids.append(ren.add_material(col.mats[i]))
        ren.sota_mat_set_layers(ids[i], col.layers[i])
    var sph = ren.add_mesh(_sphere_mesh(), ids[0]).mesh_id
    var lights = List[AaaClusterLight]()
    lights.append(AaaClusterLight.point(Vec3(1.1, 1.4, -2.6), Vec3(6.0, 6.0, 6.0), 30.0))
    lights.append(AaaClusterLight.point(Vec3(-1.5, -0.8, -2.0), Vec3(2.0, 3.0, 6.0), 30.0))
    lights.append(AaaClusterLight.rect_ltc(Vec3(-0.4, 2.0, -2.2), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.8, 0.6), 0.9, 0.45, Vec3(3.0, 3.0, 3.0), 30.0))
    var names = ["DeferredMultiLobeRtShade", "SurfaceCacheUpdate", "RtSunShadowTrace"]
    var on_ms = List[List[Float64]]()
    var off_ms = List[List[Float64]]()
    for _ in range(3):
        on_ms.append(List[Float64]())
        off_ms.append(List[Float64]())
    for it in range(44):
        ren.sota_mat.enabled = (it % 2) == 0
        ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -3.1), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=30.0, near_z=0.05)
        for i in range(len(ids)):
            ren.draw_instance(sph, trs(_grid_center(i), Quat(), Vec3(0.49, 0.49, 0.49)), material_override=ids[i])
        for i in range(len(lights)):
            ren.add_cluster_light(lights[i])
        _ = ren.render_frame(capture_readback=True)
        if it >= 4:
            var tim = ren.get_gpu_pass_timings()
            for k in range(len(tim)):
                for j in range(3):
                    if tim[k].pass_name == names[j]:
                        if (it % 2) == 0:
                            on_ms[j].append(tim[k].gpu_ms)
                        else:
                            off_ms[j].append(tim[k].gpu_ms)
    var d_on = Float64(0.0)
    var d_off = Float64(0.0)
    for j in range(3):
        if len(on_ms[j]) == 0:
            continue
        var a = _median(on_ms[j])
        var b = _median(off_ms[j])
        print("    ", names[j], "GPU ms (median,", len(on_ms[j]), "frames each) flag-on =", a, " flag-off =", b)
        if j == 0:
            d_on = a
            d_off = b
    _assert(d_on - d_off < 1.0, "round-3 layered paths cost < 1 ms in the deferred pass at 1280x720 (lights + RT)")
    print("  [PASS] Suite 8: cost")
    ren.shutdown()


def main():
    print("=== Flint v5 sota:mat round 3 (lights, RT hits, transmission, coat normal, graph) ===")
    test_spirv()
    var lut = SotaMatLut.bake()
    test_cpu(lut)
    test_gpu(lut)
    test_graph()
    test_perf()
    print("=== ALL FLINT v5 sota:mat r3 TESTS PASSED ===")
