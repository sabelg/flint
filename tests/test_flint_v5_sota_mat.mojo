# Flint v5 sota:mat (area I, round 2) - layered material model verification.
#
#   1. SPIR-V: the deferred shade pass with the sota:mat hooks validates (raster and ray-query variants).
#   2. CPU: LUT bake, Monte-Carlo white furnace of the layered model vs its split-sum weights, Heitz 2016
#      random-walk multiple scattering vs Kulla-Conty (and the legacy approximations), legacy direct albedo
#      audit, thin film (Belcour-Barla) vs an exact spectral Airy reference (and the legacy cosine fit).
#   3. GPU: SPIR-V emitter == CPU twin (direct + split-sum weights), white furnace of every layer on the
#      GPU, GPU vs CPU path-traced image diffs (furnace and sky + sun), cvar-off identity, deferred-pass cost.
#
# PNGs: FLINT_SOTA_MAT_PNG_DIR=<dir> (the round-2 evaluation uses ~/flint-models/_eval/render/I-r2).

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
from flint.platform.vulkan import get_float, put_float, ACCESS_HOST_READ, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, PIPELINE_HOST
from flint.render.aaa_bsdf import AaaMaterial, AAA_MATERIAL_WORDS
from flint.render.aaa_gi_svgf import aaa_deferred_shade_shader
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_MATERIALS, BIND_AAA_POST_SCRATCH
from flint.render.sota_mat_core import (
    SOTA_MAT_LUT_A,
    SOTA_MAT_LUT_B,
    SotaMatInputs,
    SotaMatLayers,
    SotaMatLut,
    SotaMatParams,
    emit_sota_mat_shade,
    sota_mat_ambient_cpu,
    sota_mat_clear_layers,
    sota_mat_direct_cpu,
    sota_mat_eval_cpu,
    sota_mat_frame_cpu,
    sota_mat_iridescence_cpu,
    sota_mat_params_from_words,
    sota_mat_view_phi,
)
from flint.render.sota_mat_reference import (
    SotaLegacyLut,
    SotaMatScene,
    SotaMatSphere,
    sota_mat_albedo_mc,
    sota_mat_legacy_ambient_total_cpu,
    sota_mat_legacy_direct_cpu,
    sota_mat_rw_albedo,
    sota_mat_thin_film_spectral,
)


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _lum(c: Vec3) -> Float32:
    return c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)


def _vmax(c: Vec3) -> Float32:
    return max(c.x, max(c.y, c.z))


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


def _png_dir() -> String:
    return env_value("FLINT_SOTA_MAT_PNG_DIR")


def _enc(x: Float32) -> UInt32:
    var c = max(Float32(0.0), min(Float32(1.0), x))
    # sRGB-ish display encode for the evaluation PNGs
    var e = c ** Float32(1.0 / 2.2)
    return UInt32(e * Float32(255.0) + Float32(0.5))


def _save_hdr(name: String, img: List[Vec3], w: Int, h: Int, scale: Float32):
    var d = _png_dir()
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


# ---------------------------------------------------------------------------------------------
# Material set (shared by the CPU, probe and image tests)
# ---------------------------------------------------------------------------------------------
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


def _showcase(white: Bool) -> _MatSet:
    """Eight materials, one per layer. `white`: every colour forced to 1 (white-furnace set, iridescence off)."""
    var s = _MatSet()
    var wv = Vec3(1.0, 1.0, 1.0)
    var m0 = AaaMaterial(base_color=wv if white else Vec3(0.75, 0.18, 0.12), roughness=0.4, metallic=0.0)
    s.add("plastic", m0^, SotaMatLayers())
    var m1 = AaaMaterial(base_color=wv if white else Vec3(1.0, 0.71, 0.29), roughness=0.35, metallic=1.0)
    s.add("gold_ms", m1^, SotaMatLayers())
    var m2 = AaaMaterial(base_color=wv if white else Vec3(0.04, 0.12, 0.55), roughness=0.5, metallic=0.0)
    m2.clearcoat = Float32(1.0)
    m2.clearcoat_roughness = Float32(0.08)
    s.add("carpaint_coat", m2^, SotaMatLayers())
    var m3 = AaaMaterial(base_color=wv if white else Vec3(0.30, 0.03, 0.06), roughness=0.8, metallic=0.0)
    m3.sheen_color = wv if white else Vec3(1.0, 0.35, 0.45)
    m3.sheen_roughness = Float32(0.4)
    s.add("velvet_sheen", m3^, SotaMatLayers())
    var m4 = AaaMaterial(base_color=wv if white else Vec3(0.91, 0.92, 0.92), roughness=0.3, metallic=1.0)
    var l4 = SotaMatLayers()
    l4.anisotropy = Float32(0.8)
    s.add("brushed_alu_aniso", m4^, l4^)
    var m5 = AaaMaterial(base_color=wv if white else Vec3(0.02, 0.02, 0.025), roughness=0.25, metallic=0.0)
    if not white:
        m5.iridescence = Float32(1.0)
        m5.iridescence_thickness_nm = Float32(420.0)
        m5.iridescence_ior = Float32(1.33)
    s.add("thinfilm_irid" if not white else "dielectric_r25", m5^, SotaMatLayers())
    var m6 = AaaMaterial(base_color=wv if white else Vec3(0.95, 0.97, 1.0), roughness=0.15, metallic=0.0)
    var l6 = SotaMatLayers()
    l6.transmission = Float32(1.0)
    l6.transmission_tint = wv if white else Vec3(0.65, 0.88, 1.0)
    s.add("thin_glass_trans", m6^, l6^)
    var m7 = AaaMaterial(base_color=wv if white else Vec3(0.55, 0.56, 0.58), roughness=0.45, metallic=1.0)
    m7.clearcoat = Float32(1.0)
    m7.clearcoat_roughness = Float32(0.05)
    var l7 = SotaMatLayers()
    l7.anisotropy = Float32(0.6)
    l7.anisotropy_rotation = Float32(0.785)
    s.add("coat_over_aniso", m7^, l7^)
    return s^


# ---------------------------------------------------------------------------------------------
# 1. SPIR-V
# ---------------------------------------------------------------------------------------------
def test_spirv():
    for rq in range(2):
        var words = aaa_deferred_shade_shader(rq == 1)
        var rep = validate_spirv_module(words)
        if not rep.valid:
            var err = rep.errors[0] if len(rep.errors) > 0 else String("unknown")
            _fail("deferred shade (ray_query=" + String(rq) + ") SPIR-V invalid: " + err)
        print("    deferred shade ray_query =", rq, ":", len(words), "words,", rep.instruction_count, "instructions")
    print("  [PASS] Suite 1: deferred shade pass with sota:mat hooks validates (RT and raster variants)")


# ---------------------------------------------------------------------------------------------
# 2. CPU references
# ---------------------------------------------------------------------------------------------
def test_cpu_furnace(lut: SotaMatLut):
    """MC estimate of the hemispherical (+ transmitted) albedo of the layered BSDF under a white furnace
    vs the split-sum weights the GPU applies; white materials must sum to 1 (energy conserved, none lost)."""
    var s = _showcase(True)
    var worst_mc = Float32(0.0)
    var worst_one = Float32(0.0)
    for i in range(len(s.names)):
        var p = s.params(i)
        var line = String("    ") + s.names[i] + ":"
        for mu in [Float32(0.15), Float32(0.5), Float32(0.95)]:
            var mc = sota_mat_albedo_mc(p, lut, mu, 40000, UInt64(17 + i * 3))
            var fr = sota_mat_frame_cpu(Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), p.aniso_rot)
            var st = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
            var v = Vec3(st * Float32(0.8), st * Float32(0.6), mu)
            var amb = sota_mat_ambient_cpu(p, lut, mu, sota_mat_view_phi(v.dot(fr.t), v.dot(fr.b)))
            var tot = amb.total()
            worst_mc = max(worst_mc, abs(_lum(mc) - _lum(tot)))
            worst_one = max(worst_one, abs(_lum(tot) - Float32(1.0)))
            line += " mu=" + String(mu) + " mc=" + String(_lum(mc)) + " ss=" + String(_lum(tot))
        print(line)
    print("    white furnace: worst |MC - split-sum| =", worst_mc, " worst |split-sum - 1| =", worst_one)
    _assert(worst_mc < Float32(0.012), "MC albedo of the layered BSDF matches its split-sum weights within 1.2%")
    _assert(worst_one < Float32(0.006), "white layered materials conserve energy in the furnace within 0.6%")
    print("  [PASS] Suite 2a: CPU white furnace of every layer (MC vs split-sum)")


def test_cpu_multiscatter(lut: SotaMatLut):
    """Heitz 2016 random walk (ground truth for multiple scattering on GGX) vs Kulla-Conty (new) and the
    legacy `1 + F0 (1/E - 1)` boost, for a white (F = 1) and a gold conductor."""
    var llut = SotaLegacyLut()
    var worst_white = Float32(0.0)
    var worst_gold_new = Float32(0.0)
    var worst_gold_old = Float32(0.0)
    var gold = Vec3(1.0, 0.71, 0.29)
    for r in [Float32(0.3), Float32(0.6), Float32(1.0)]:
        for mu in [Float32(0.3), Float32(0.8)]:
            var a = r * r
            var rw_w = sota_mat_rw_albedo(mu, 0.0, a, a, Vec3(1.0, 1.0, 1.0), 1.0, 60000, UInt64(5))
            var rw_g = sota_mat_rw_albedo(mu, 0.0, a, a, gold, 1.0, 60000, UInt64(9))
            var m = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=r, metallic=1.0)
            var pw = _params(m, SotaMatLayers())
            m.base_color = gold
            var pg = _params(m, SotaMatLayers())
            var new_w = sota_mat_ambient_cpu(pw, lut, mu).total()
            var new_g = sota_mat_ambient_cpu(pg, lut, mu).total()
            var old_g = sota_mat_legacy_ambient_total_cpu(pg, llut, mu)
            worst_white = max(worst_white, abs(new_w.y - rw_w.total.y))
            worst_gold_new = max(worst_gold_new, abs(new_g.y - rw_g.total.y) / rw_g.total.y)
            worst_gold_old = max(worst_gold_old, abs(old_g.y - rw_g.total.y) / rw_g.total.y)
            print("    r=", r, "mu=", mu, " white RW", rw_w.total.y, "(1st order", rw_w.single.y, ") KC", new_w.y,
                  "| gold G RW", rw_g.total.y, "KC", new_g.y, "legacy", old_g.y)
    print("    worst: white |KC - RW| =", worst_white, " gold rel KC =", worst_gold_new, " gold rel legacy =", worst_gold_old)
    _assert(worst_white < Float32(0.01), "Kulla-Conty restores the white-furnace energy of the random walk within 1%")
    _assert(worst_gold_new < Float32(0.10), "Kulla-Conty colored conductor within 10% of the random walk")
    print("  [PASS] Suite 2b: multiscatter GGX vs Heitz random walk")


def _albedo_direct(p: SotaMatParams, lut: SotaMatLut, llut: SotaLegacyLut, mu: Float32, legacy: Bool, n_s: Int) -> Vec3:
    """Directional albedo of the *direct* evaluators: integral of f |cos| over the upper hemisphere
    (stratified uniform), i.e. what a uniform white sky made of point lights would reflect."""
    var n = Vec3(0.0, 0.0, 1.0)
    var t = Vec3(1.0, 0.0, 0.0)
    var b = Vec3(0.0, 1.0, 0.0)
    var st = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
    var v = Vec3(st, 0.0, mu)
    var acc = Vec3(0.0, 0.0, 0.0)
    for i in range(n_s):
        for j in range(n_s):
            var z = (Float32(i) + Float32(0.5)) / Float32(n_s)
            var ph = (Float32(j) + Float32(0.5)) / Float32(n_s) * Float32(6.2831853)
            var sz = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
            var l = Vec3(sz * cos(ph), sz * sin(ph), z)
            var f = sota_mat_legacy_direct_cpu(p, llut, n, v, l) if legacy else sota_mat_eval_cpu(p, lut, n, t, b, v, l)
            acc = acc + f
    return acc * (Float32(6.2831853) / Float32(n_s * n_s))


def test_cpu_legacy_audit(lut: SotaMatLut):
    """Directional albedo of the legacy direct BSDF vs the layered one for white materials: the legacy
    sheen adds energy (> 1), the legacy clearcoat attenuates the base twice, the new model stays <= 1."""
    var llut = SotaLegacyLut()
    var s = _showcase(True)
    var worst_new = Float32(0.0)
    for i in [0, 1, 2, 3]:
        var p = s.params(i)
        if p.clearcoat > Float32(0.0):
            p.clearcoat_roughness = Float32(0.35)  # a lobe the uniform quadrature resolves (MC covers 0.08)
        var line = String("    ") + s.names[i] + ":"
        for mu in [Float32(0.3), Float32(0.9)]:
            var a_old = _albedo_direct(p, lut, llut, mu, True, 160)
            var a_new = _albedo_direct(p, lut, llut, mu, False, 160)
            worst_new = max(worst_new, abs(_lum(a_new) - Float32(1.0)))
            line += " mu=" + String(mu) + " legacy=" + String(_lum(a_old)) + " layered=" + String(_lum(a_new))
        print(line)
    print("    worst |layered direct albedo - 1| (white) =", worst_new)
    _assert(worst_new < Float32(0.02), "layered direct BSDF of white materials integrates to 1 within 2%")
    print("  [PASS] Suite 2c: legacy direct BSDF energy audit")


def test_cpu_thin_film():
    """Belcour-Barla 2017 analytic (new) vs exact spectral Airy summation, film 1.33 over a 1.5 base."""
    var worst = Float32(0.0)
    var f0 = Float32(0.04)
    for d in [Float32(250.0), Float32(400.0), Float32(600.0)]:
        for c in [Float32(1.0), Float32(0.7), Float32(0.4)]:
            var ref_s = sota_mat_thin_film_spectral(c, d, 1.33, 1.5)
            var bb = sota_mat_iridescence_cpu(c, d, 1.33, Vec3(f0, f0, f0))
            var e = abs(bb.x - ref_s.x) + abs(bb.y - ref_s.y) + abs(bb.z - ref_s.z)
            worst = max(worst, e / Float32(3.0))
            print("    d=", d, "cos=", c, " spectral", ref_s.x, ref_s.y, ref_s.z, " belcour", bb.x, bb.y, bb.z)
    print("    worst mean-abs RGB error =", worst)
    _assert(worst < Float32(0.03), "Belcour-Barla thin film within 0.03 (abs) of the spectral reference")
    print("  [PASS] Suite 2d: thin-film iridescence vs spectral Airy reference")


# ---------------------------------------------------------------------------------------------
# 3. GPU
# ---------------------------------------------------------------------------------------------
comptime PROBE_IN: Int = 16
comptime PROBE_OUT: Int = 16
comptime PROBE_CASES: Int = 256


def _probe_shader(in_off: Int, out_off: Int) -> List[UInt32]:
    """Case i: n, v, l, t, b, material id (from post-scratch); the material words come from the live
    material buffer. Writes direct rgb, w_cc, w_spec rgb, w_diff rgb, w_trans rgb, w_em."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_POST_SCRATCH)
    var bm = s.buffer(BIND_AAA_MATERIALS)
    var i = s.invocation_axis(0)
    s.return_if(s.uge(i, s.uint(UInt32(PROBE_CASES))))
    var base = s.iadd(s.uint(UInt32(in_off)), s.imul(i, s.uint(UInt32(PROBE_IN))))
    var v = List[UInt32]()
    for k in range(16):
        v.append(s.load_float(b, s.iadd(base, s.uint(UInt32(k)))))
    var mat_id = s.uint_from_float(v[15])
    var mw = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var w = List[UInt32]()
    for k in range(AAA_MATERIAL_WORDS):
        w.append(s.load_float(bm, s.iadd(mw, s.uint(UInt32(k)))))
    var yes = s.ieq(s.uint(1), s.uint(1))
    var z = s.float(0.0)
    var sh = emit_sota_mat_shade(
        s, bm, mat_id, yes,
        SotaMatInputs(
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14],
            w[0], w[1], w[2], w[5], s.fabs(w[3]), w[4], w[6], w[7],
            w[16], w[17], w[18], w[19], w[20], w[21], w[22], w[13], w[14], w[15],
        ),
        (z, z, z),
    )
    var ob = s.iadd(s.uint(UInt32(out_off)), s.imul(i, s.uint(UInt32(PROBE_OUT))))
    var outs = [sh.dir_r, sh.dir_g, sh.dir_b, sh.w_cc, sh.ws_r, sh.ws_g, sh.ws_b, sh.wd_r, sh.wd_g, sh.wd_b, sh.wt_r, sh.wt_g, sh.wt_b, sh.w_em]
    for k in range(len(outs)):
        s.store_float(b, s.iadd(ob, s.uint(UInt32(k))), s.load_f(outs[k]))
    return s.finish()


def _rand_unit(seed: Int) -> Vec3:
    var a = Float32((seed * 7919 + 13) % 1000) / Float32(1000.0)
    var c = Float32((seed * 104729 + 71) % 1000) / Float32(1000.0)
    var z = Float32(1.0) - Float32(2.0) * a
    var r = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
    var ph = c * Float32(6.2831853)
    return Vec3(r * cos(ph), r * sin(ph), z).normalized()


def run_emitter_probe(mut ren: FlintAaaVulkanRenderer, s: _MatSet, ids: List[Int], lut: SotaMatLut) -> Float32:
    """Max relative error of the GPU emitter vs the CPU twin over PROBE_CASES random cases."""
    ren.flush_deferred_frame()
    ren.vk.wait_idle()
    _assert(not ren.sota_e_hair.active(), "probe region is free (hair DOM inactive)")
    var in_off = ren.sota_e_hair.base_word
    var out_off = in_off + PROBE_CASES * PROBE_IN
    var code = _probe_shader(in_off, out_off)
    var rep = validate_spirv_module(code)
    _assert(rep.valid, "probe shader validates")
    var pipe = ren.vk.create_aaa_compute_pipeline(ren.pipe_layout, code)
    _assert(pipe != 0, "probe pipeline created")
    var addr = ren.buf_post_scratch.cpu_address
    var ins = List[List[Float32]]()
    for c in range(PROBE_CASES):
        var n = _rand_unit(c * 5 + 1)
        var vv = _rand_unit(c * 5 + 2)
        if vv.dot(n) < Float32(0.05):
            vv = (vv - n * (Float32(2.0) * vv.dot(n))).normalized()
            if vv.dot(n) < Float32(0.05):
                vv = (vv + n * Float32(0.3)).normalized()
        var l = _rand_unit(c * 5 + 3)
        var t = _rand_unit(c * 5 + 4)
        var bb = n.cross(t)
        if bb.length() < Float32(1e-3):
            t = Vec3(1.0, 0.0, 0.0) if abs(n.x) < Float32(0.9) else Vec3(0.0, 1.0, 0.0)
            bb = n.cross(t)
        if c % 2 == 1:
            bb = bb * Float32(-1.0)   # both handedness
        var row = List[Float32]()
        row.append(n.x); row.append(n.y); row.append(n.z)
        row.append(vv.x); row.append(vv.y); row.append(vv.z)
        row.append(l.x); row.append(l.y); row.append(l.z)
        row.append(t.x); row.append(t.y); row.append(t.z)
        row.append(bb.x); row.append(bb.y); row.append(bb.z)
        row.append(Float32(ids[c % len(ids)]))
        for k in range(PROBE_IN):
            put_float(addr + (in_off + c * PROBE_IN + k) * 4, row[k])
        ins.append(row^)
    for c in range(PROBE_CASES * PROBE_OUT):
        put_float(addr + (out_off + c) * 4, Float32(-12345.0))
    ren.vk.wait_idle()
    _ = ren.vk.begin_commands(0)
    var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
    ren.vk.dispatch_aaa(0, pipe, ren.pipe_layout, ren.desc_sets[0], push, (PROBE_CASES + 63) // 64, 1)
    ren.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
    var fence = ren.vk.submit(0)
    _ = ren.vk.wait(fence)
    var worst = Float32(0.0)
    for c in range(PROBE_CASES):
        var row = ins[c].copy()
        var mi = c % len(ids)
        var p = s.params(mi)
        var n = Vec3(row[0], row[1], row[2])
        var vv = Vec3(row[3], row[4], row[5])
        var l = Vec3(row[6], row[7], row[8])
        var t = Vec3(row[9], row[10], row[11])
        var bb = Vec3(row[12], row[13], row[14])
        var d = sota_mat_direct_cpu(p, lut, n, t, bb, vv, l)
        var fr = sota_mat_frame_cpu(n, t, bb, p.aniso_rot)
        var amb = sota_mat_ambient_cpu(p, lut, n.dot(vv), sota_mat_view_phi(vv.dot(fr.t), vv.dot(fr.b)))
        var refs = [d.x, d.y, d.z, amb.w_cc, amb.w_spec.x, amb.w_spec.y, amb.w_spec.z, amb.w_diff.x, amb.w_diff.y,
                    amb.w_diff.z, amb.w_trans.x, amb.w_trans.y, amb.w_trans.z, amb.w_em]
        for k in range(len(refs)):
            var o = get_float(addr + (out_off + c * PROBE_OUT + k) * 4)
            var e = abs(o - refs[k]) / max(Float32(0.02), abs(refs[k]))
            if e > worst:
                worst = e
                if e > Float32(2e-3):
                    print("      case", c, s.names[mi], "out", k, "gpu", o, "cpu", refs[k])
    ren.vk.destroy_pipeline(pipe)
    return worst


def _sphere_mesh() -> NativeMesh:
    """Unit sphere with the planar front projection UV = 0.5 + 0.5 (x, -y) of `SotaMatScene.frame_at`
    (tangent = dP/du along +x); built at unit radius, positioned by the instance transform."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=64, segments=96)
    var m = mb^.finish()
    for i in range(m.vertex_count()):
        m.texcoords[i * 2] = Float32(0.5) + Float32(0.5) * m.positions[i * 3]
        m.texcoords[i * 2 + 1] = Float32(0.5) - Float32(0.5) * m.positions[i * 3 + 1]
    return m^


comptime GRID_R: Float32 = 0.42


def _grid_center(i: Int) -> Vec3:
    return Vec3(Float32(-1.5) + Float32(i % 4), Float32(0.5) if i < 4 else Float32(-0.5), Float32(0.0))


def _cpu_scene(s: _MatSet, furnace: Bool, sky: SkyDomeParams, sun_dir: Vec3, sun_rad: Vec3, ibl: Float32) -> SotaMatScene:
    var sc = SotaMatScene()
    for i in range(len(s.names)):
        sc.spheres.append(SotaMatSphere(_grid_center(i), GRID_R, s.params(i)))
    sc.furnace = furnace
    sc.sky = sky.copy()
    sc.sun_dir = sun_dir
    sc.sun_rad = sun_rad
    sc.ibl_intensity = ibl
    sc.cam_pos = Vec3(0.0, 0.0, -6.0)
    sc.cam_target = Vec3(0.0, 0.0, 0.0)
    sc.fov_y_deg = Float32(30.0)
    return sc^


def _render_gpu(mut ren: FlintAaaVulkanRenderer, mesh_id: Int, ids: List[Int]) -> List[Vec3]:
    var hdr = List[Vec3]()
    for _ in range(3):
        ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -6.0), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=30.0, near_z=0.05)
        for i in range(len(ids)):
            ren.draw_instance(mesh_id, trs(_grid_center(i), Quat(), Vec3(GRID_R, GRID_R, GRID_R)), material_override=ids[i])
        _ = ren.render_frame(capture_readback=True)
        hdr = ren.readback_resolved_hdr()
    return hdr^


struct _Diff(Copyable, Movable):
    var per_sphere: List[Float32]   # |mean(gpu) - mean(cpu)| / mean(cpu), luminance, interior pixels
    var per_sphere_px: List[Float32]  # mean per-pixel relative luminance error
    var worst: Float32
    var worst_px: Float32

    def __init__(out self):
        self.per_sphere = List[Float32]()
        self.per_sphere_px = List[Float32]()
        self.worst = Float32(0.0)
        self.worst_px = Float32(0.0)


def _diff(sc: SotaMatScene, gpu: List[Vec3], cpu: List[Vec3], w: Int, h: Int, mut dimg: List[Vec3], mask_sun_core: Bool = False) -> _Diff:
    """Interior pixels only (n . v > 0.3): the tessellated silhouette and the planar UV's rim degenerate.
    `mask_sun_core`: also skip the sun's mirror core (n . h > 0.99), where a 1-2 px highlight of the
    sharp coat is point-sampled at the pixel centre on both sides (tessellated vs analytic normal)."""
    var r = _Diff()
    var n_s = len(sc.spheres)
    var sg = List[Float64](length=n_s, fill=0.0)
    var scp = List[Float64](length=n_s, fill=0.0)
    var spx = List[Float64](length=n_s, fill=0.0)
    var cnt = List[Int](length=n_s, fill=0)
    dimg = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 0.0))
    for py in range(h):
        for px in range(w):
            var d = sc.primary_dir(px, py, w, h, Float32(0.5), Float32(0.5))
            var hh = sc.hit(sc.cam_pos, d)
            if hh[0] < 0:
                continue
            var p = sc.cam_pos + d * hh[1]
            var n = ((p - sc.spheres[hh[0]].center) * (Float32(1.0) / sc.spheres[hh[0]].radius)).normalized()
            var g = gpu[py * w + px]
            var c = cpu[py * w + px]
            dimg[py * w + px] = Vec3(abs(g.x - c.x), abs(g.y - c.y), abs(g.z - c.z))
            if n.dot(d * Float32(-1.0)) < Float32(0.3):
                continue
            if mask_sun_core and n.dot((sc.sun_dir - d).normalized()) > Float32(0.99):
                continue
            var i = hh[0]
            sg[i] += Float64(_lum(g))
            scp[i] += Float64(_lum(c))
            spx[i] += Float64(abs(_lum(g) - _lum(c)) / max(Float32(0.02), _lum(c)))
            cnt[i] += 1
    for i in range(n_s):
        var k = Float64(max(1, cnt[i]))
        var e = Float32(abs(sg[i] - scp[i]) / max(1e-4, scp[i]))
        var epx = Float32(spx[i] / k)
        r.per_sphere.append(e)
        r.per_sphere_px.append(epx)
        r.worst = max(r.worst, e)
        r.worst_px = max(r.worst_px, epx)
    return r^


def _split_worst(d: _Diff, s: _MatSet) -> Tuple[Float32, Float32]:
    """(worst per-sphere error without iridescence, iridescent sphere's error): the thin-film split-sum
    weight uses F_irid(n . v) (open item), so it gets its own budget."""
    var w = Float32(0.0)
    var wi = Float32(0.0)
    for i in range(len(s.names)):
        if s.mats[i].iridescence > Float32(0.0):
            wi = max(wi, d.per_sphere[i])
        else:
            w = max(w, d.per_sphere[i])
    return (w, wi)


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


def test_gpu():
    var w = 192
    var h = 144
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren)
    var sky = _sky()
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky, base_dim=64, mip_levels=6))
    ren.ibl_intensity = Float32(1.0)
    ren.begin_frame()
    _ = ren.render_frame(capture_readback=True)

    # The LUT block the renderer uploads == the CPU bake the references use.
    _assert(not ren.sota_mat.enabled or env_value("FLINT_SOTA_MAT") == "1", "cvar defaults off")
    ren.sota_mat.enabled = True
    ren.sota_mat.prepare(ren.buf_materials.cpu_address)
    _assert(ren.sota_mat.lut_uploaded, "LUT block uploaded into the material buffer")
    print("    LUT bake + upload (renderer, first enabled frame):", ren.sota_mat.bake_s * 1000.0, "ms")
    var lut = SotaMatLut.bake()

    var col = _showcase(False)
    var wht = _showcase(True)
    var ids_c = List[Int]()
    var ids_w = List[Int]()
    for i in range(len(col.names)):
        ids_c.append(ren.add_material(col.mats[i]))
        ren.sota_mat_set_layers(ids_c[i], col.layers[i])
    for i in range(len(wht.names)):
        ids_w.append(ren.add_material(wht.mats[i]))
        ren.sota_mat_set_layers(ids_w[i], wht.layers[i])

    # 3a. emitter == CPU twin
    var e_c = run_emitter_probe(ren, col, ids_c, lut)
    var e_w = run_emitter_probe(ren, wht, ids_w, lut)
    print("    GPU emitter vs CPU twin, max rel err: coloured =", e_c, " white =", e_w)
    _assert(e_c < Float32(5e-3) and e_w < Float32(5e-3), "GPU layered emitter matches the CPU twin")
    print("  [PASS] Suite 3a: SPIR-V layered model == CPU twin (direct + split-sum weights, 2 x 256 cases)")

    var sph = ren.add_mesh(_sphere_mesh(), ids_c[0])
    var dimg = List[Vec3]()

    # 3b. GPU white furnace, layered model on vs off
    ren.furnace_white_dome = True
    var sc_w = _cpu_scene(wht, True, sky, Vec3(0.3, 0.8, -0.5).normalized(), Vec3(0.0, 0.0, 0.0), 1.0)
    var g_w = _render_gpu(ren, sph.mesh_id, ids_w)
    ren.sota_mat.enabled = False
    var g_w_off = _render_gpu(ren, sph.mesh_id, ids_w)
    ren.sota_mat.enabled = True
    var ones = List[Vec3](length=w * h, fill=Vec3(1.0, 1.0, 1.0))
    var fw = _diff(sc_w, g_w, ones, w, h, dimg)
    var fw_off = _diff(sc_w, g_w_off, ones, w, h, dimg)
    for i in range(len(wht.names)):
        print("    furnace", wht.names[i], ": layered", Float32(1.0) + fw.per_sphere[i], "(|err|)  legacy |err|", fw_off.per_sphere[i])
    print("    GPU white furnace worst |mean - 1|: layered =", fw.worst, " legacy =", fw_off.worst)
    _save_hdr("furnace_white_layered.png", g_w, w, h, Float32(0.5))
    _save_hdr("furnace_white_legacy.png", g_w_off, w, h, Float32(0.5))
    _assert(fw.worst < Float32(0.01), "GPU white furnace of every layer within 1%")
    print("  [PASS] Suite 3b: GPU white furnace (8 layered materials)")

    # 3c. GPU vs CPU path tracer, coloured materials in the furnace
    var sc_f = _cpu_scene(col, True, sky, Vec3(0.3, 0.8, -0.5).normalized(), Vec3(0.0, 0.0, 0.0), 1.0)
    var t0 = get_time()
    var pt_f = sc_f.render(w, h, 64, 1, False, lut, UInt64(11))
    var t_pt = get_time() - t0
    var g_f = _render_gpu(ren, sph.mesh_id, ids_c)
    ren.sota_mat.enabled = False
    var g_f_off = _render_gpu(ren, sph.mesh_id, ids_c)
    ren.sota_mat.enabled = True
    var df = _diff(sc_f, g_f, pt_f, w, h, dimg)
    _save_hdr("furnace_diff_x10.png", dimg, w, h, Float32(10.0))
    var df_off = _diff(sc_f, g_f_off, pt_f, w, h, dimg)
    for i in range(len(col.names)):
        print("    furnace PT diff", col.names[i], ": layered", df.per_sphere[i], "(px", df.per_sphere_px[i], ")  legacy", df_off.per_sphere[i], "(px", df_off.per_sphere_px[i], ")")
    print("    furnace GPU vs PT worst mean err: layered =", df.worst, " legacy =", df_off.worst, " (PT", t_pt, "s)")
    _save_hdr("furnace_pt.png", pt_f, w, h, Float32(1.0))
    _save_hdr("furnace_gpu_layered.png", g_f, w, h, Float32(1.0))
    _save_hdr("furnace_gpu_legacy.png", g_f_off, w, h, Float32(1.0))
    var sw_f = _split_worst(df, col)
    _assert(sw_f[0] < Float32(0.015) and sw_f[1] < Float32(0.05), "GPU layered furnace render within 1.5% (per-sphere mean; thin film 5%) of the CPU path tracer")
    print("  [PASS] Suite 3c: GPU vs CPU path tracer, furnace")

    # 3d. GPU vs CPU path tracer, sky + sun
    ren.furnace_white_dome = False
    var sun_dir = Vec3(0.35, 0.75, -0.55).normalized()
    ren.sun_dir = sun_dir
    ren.sun_radiance = Vec3(2.5, 2.4, 2.2)
    var sc_s = _cpu_scene(col, False, sky, sun_dir, Vec3(2.5, 2.4, 2.2), 1.0)
    var pt_s = sc_s.render(w, h, 64, 1, False, lut, UInt64(23))
    var g_s = _render_gpu(ren, sph.mesh_id, ids_c)
    ren.sota_mat.enabled = False
    var g_s_off = _render_gpu(ren, sph.mesh_id, ids_c)
    var ds_full = _diff(sc_s, g_s, pt_s, w, h, dimg)
    var ds = _diff(sc_s, g_s, pt_s, w, h, dimg, True)
    _save_hdr("sky_sun_diff_x10.png", dimg, w, h, Float32(10.0))
    var ds_off = _diff(sc_s, g_s_off, pt_s, w, h, dimg, True)
    for i in range(len(col.names)):
        print("    sky+sun PT diff", col.names[i], ": layered", ds.per_sphere[i], "(px", ds.per_sphere_px[i], ")  legacy", ds_off.per_sphere[i], "(px", ds_off.per_sphere_px[i], ")")
    print("    sky+sun GPU vs PT worst mean err (sun mirror core masked): layered =", ds.worst, " legacy =", ds_off.worst,
          " | unmasked layered =", ds_full.worst)
    _save_hdr("sky_sun_pt.png", pt_s, w, h, Float32(1.0))
    _save_hdr("sky_sun_gpu_layered.png", g_s, w, h, Float32(1.0))
    _save_hdr("sky_sun_gpu_legacy.png", g_s_off, w, h, Float32(1.0))
    var sw_s = _split_worst(ds, col)
    _assert(sw_s[0] < Float32(0.05) and sw_s[1] < Float32(0.08), "GPU layered sky + sun render within 5% (per-sphere mean; thin film 8%) of the CPU path tracer")
    print("  [PASS] Suite 3d: GPU vs CPU path tracer, sky + sun")

    # 3e. cvar off: the extension words are ignored, output bit-identical with and without them
    ren.sota_mat.enabled = False
    var off_a = _render_gpu(ren, sph.mesh_id, ids_c)
    sota_mat_clear_layers(ren.buf_materials.cpu_address)
    var off_b = _render_gpu(ren, sph.mesh_id, ids_c)
    var n_diff = 0
    for i in range(w * h):
        if off_a[i].x != off_b[i].x or off_a[i].y != off_b[i].y or off_a[i].z != off_b[i].z:
            n_diff += 1
    print("    cvar off: pixels differing with vs without layer words =", n_diff)
    _assert(n_diff == 0, "cvar off ignores the layer extension (bit-identical)")
    for i in range(len(col.names)):
        ren.sota_mat_set_layers(ids_c[i], col.layers[i])
    print("  [PASS] Suite 3e: cvar-off identity")

    ren.shutdown()


def test_gpu_perf():
    """Deferred-pass cost at 1280x720 with the eight layered spheres filling the frame, flag on vs off on
    alternating frames (medians; the GPU is shared)."""
    var w = 1280
    var h = 720
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(_sky(), base_dim=64, mip_levels=6))
    ren.sota_mat.enabled = True
    var col = _showcase(False)
    var ids = List[Int]()
    for i in range(len(col.names)):
        ids.append(ren.add_material(col.mats[i]))
        ren.sota_mat_set_layers(ids[i], col.layers[i])
    var sph = ren.add_mesh(_sphere_mesh(), ids[0])
    var on_ms = List[Float64]()
    var off_ms = List[Float64]()
    for it in range(44):
        ren.sota_mat.enabled = (it % 2) == 0
        ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -3.1), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=30.0, near_z=0.05)
        for i in range(len(ids)):
            ren.draw_instance(sph.mesh_id, trs(_grid_center(i), Quat(), Vec3(0.49, 0.49, 0.49)), material_override=ids[i])
        _ = ren.render_frame(capture_readback=True)
        if it >= 4:
            var tim = ren.get_gpu_pass_timings()
            for k in range(len(tim)):
                if tim[k].pass_name == "DeferredMultiLobeRtShade":
                    if (it % 2) == 0:
                        on_ms.append(tim[k].gpu_ms)
                    else:
                        off_ms.append(tim[k].gpu_ms)
    var m_on = _median(on_ms)
    var m_off = _median(off_ms)
    print("    deferred shade GPU ms (median,", len(on_ms), "frames each) flag-on =", m_on, " flag-off =", m_off, "at", w, "x", h)
    _assert(m_on - m_off < 0.5, "layered model costs < 0.5 ms in the deferred pass at 1280x720")
    print("  [PASS] Suite 3f: deferred-pass cost")
    ren.shutdown()


def main():
    print("=== Flint v5 sota:mat Layered Material Suite ===")
    test_spirv()
    var t0 = get_time()
    var lut = SotaMatLut.bake()
    print("    LUT bake (2-D + 4-D anisotropic tables):", (get_time() - t0) * 1000.0, "ms")
    test_cpu_furnace(lut)
    test_cpu_multiscatter(lut)
    test_cpu_legacy_audit(lut)
    test_cpu_thin_film()
    test_gpu()
    test_gpu_perf()
    print("=== ALL FLINT v5 sota:mat TESTS PASSED ===")
