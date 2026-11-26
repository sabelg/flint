"""Area I round 3: the layered material model in the local-light loops (clustered point / spot, LTC rect / tube,
ReSTIR single light). The deferred pass hands the per-pixel lobe state (`SotaMatLobesGpu`) to the loops; with the
switch off (`lb.on` false) every helper returns the legacy value it was given, bit for bit.

Point / spot: the exact BSDF * cos of `emit_sota_mat_eval_light` (front side) replaces the legacy GGX + Lambert.
LTC: the polygon form factors are split-sum weighted with the layered directional albedos (as the ambient
composition does): base specular `E_s(v) * att`, diffuse + sheen under the coat, and the coat lobe with its own
LTC fetch at the coat roughness (Heitz et al. 2016, LTC; weights after Kulla & Conty 2017).
"""

from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_vec3_cross, emit_vec3_dot, emit_vec3_normalize
from flint.render.sota_mat_core import (
    SOTA_MAT_LUT_SH,
    SotaMatLobesGpu,
    SotaMatSurfaceExtra,
    _e_lut2,
    emit_sota_mat_albedo_avg,
    emit_sota_mat_eval_light,
    emit_sota_mat_inputs_from_words,
    emit_sota_mat_lobes,
    emit_sota_mat_on_for_material,
    sota_mat_lb_index,
)


struct SotaMatLtcWeights(Copyable, Movable):
    """Per-pixel LTC weights of the layered model (fvars, valid when `on`)."""
    var on: UInt32
    var wd: List[UInt32]
    var ws: List[UInt32]
    var wc: UInt32
    var cc_uv_mu: UInt32
    var cc_uv_r: UInt32

    def __init__(out self):
        self.on = 0
        self.wd = List[UInt32]()
        self.ws = List[UInt32]()
        self.wc = 0
        self.cc_uv_mu = 0
        self.cc_uv_r = 0

    def __init__(out self, mut s: AaaSpirvModule, on: UInt32):
        self.on = on
        self.wd = [s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(0.0))]
        self.ws = [s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(0.0))]
        self.wc = s.fvar(s.float(0.0))
        self.cc_uv_mu = s.fvar(s.float(0.5))
        self.cc_uv_r = s.fvar(s.float(0.5))


def emit_sota_mat_ltc_weights(mut s: AaaSpirvModule, b_mats: UInt32, lb: SotaMatLobesGpu) -> SotaMatLtcWeights:
    var w = SotaMatLtcWeights(s, lb.on)
    s.begin_if(lb.on)
    var one = s.float(1.0)
    var ndotv = lb.get(s, sota_mat_lb_index("ndotv"))
    var m = lb.get(s, sota_mat_lb_index("metal"))
    var trans = lb.get(s, sota_mat_lb_index("trans"))
    var att_cc = lb.get(s, sota_mat_lb_index("att_cc"))
    var att_sh = lb.get(s, sota_mat_lb_index("att_sh"))
    var att = s.fmul(att_cc, att_sh)
    var es_v = lb.get3(s, sota_mat_lb_index("esv"))
    var alb = lb.get3(s, sota_mat_lb_index("alb"))
    var shc = lb.get3(s, sota_mat_lb_index("sh"))
    var esh_v = _e_lut2(s, b_mats, SOTA_MAT_LUT_SH, ndotv, lb.get(s, sota_mat_lb_index("sh_rough")))
    var kd = s.fmul(s.fmul(s.fsub(one, m), s.fsub(one, trans)), att_sh)
    for k in range(3):
        s.store(w.ws[k], s.fmul(es_v[k], att))
        s.store(w.wd[k], s.fmul(s.fma(shc[k], esh_v, s.fmul(s.fmul(alb[k], s.fsub(one, es_v[k])), kd)), att_cc))
    s.store(w.wc, s.fsub(one, att_cc))
    s.store(w.cc_uv_mu, s.nclamp(lb.get(s, sota_mat_lb_index("ndotv_c")), s.float(0.01), s.float(0.99)))
    s.store(w.cc_uv_r, s.nclamp(lb.get(s, sota_mat_lb_index("cc_rough")), s.float(0.02), s.float(0.99)))
    s.end_if()
    return w^


def emit_sota_mat_ltc_coat_cond(mut s: AaaSpirvModule, w: SotaMatLtcWeights) -> UInt32:
    """`on && coat weight > 0`: the caller evaluates the coat form factor inside this branch."""
    return s.logical_and(w.on, s.fgt(s.load_f(w.wc), s.float(0.0)))


def emit_sota_mat_ltc_coat_uv(mut s: AaaSpirvModule, w: SotaMatLtcWeights) -> UInt32:
    return s.vec2(s.load_f(w.cc_uv_mu), s.load_f(w.cc_uv_r))


def emit_sota_mat_ltc_combine(
    mut s: AaaSpirvModule,
    w: SotaMatLtcWeights,
    legacy: Tuple[UInt32, UInt32, UInt32],
    lw: Tuple[UInt32, UInt32, UInt32],
    ff_diff: UInt32,
    ff_spec: UInt32,
    ff_cc: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    var lws = [lw[0], lw[1], lw[2]]
    var leg = [legacy[0], legacy[1], legacy[2]]
    var o = List[UInt32]()
    var wc = s.load_f(w.wc)
    for k in range(3):
        var lay = s.fmul(lws[k], s.fma(s.load_f(w.wd[k]), ff_diff, s.fma(s.load_f(w.ws[k]), ff_spec, s.fmul(wc, ff_cc))))
        o.append(s.select_f(w.on, lay, leg[k]))
    return (o[0], o[1], o[2])


def emit_sota_mat_punctual(
    mut s: AaaSpirvModule,
    b_mats: UInt32,
    lb: SotaMatLobesGpu,
    lx: UInt32,
    ly: UInt32,
    lz: UInt32,
    atten: UInt32,
    lrgb: Tuple[UInt32, UInt32, UInt32],
    legacy: Tuple[UInt32, UInt32, UInt32],
) -> Tuple[UInt32, UInt32, UInt32]:
    """Point / spot radiance: `L * atten * f(v, l) cos` of the layered model when on, `legacy` otherwise."""
    var o_r = s.fvar(legacy[0])
    var o_g = s.fvar(legacy[1])
    var o_b = s.fvar(legacy[2])
    s.begin_if(lb.on)
    var f = emit_sota_mat_eval_light(s, b_mats, lb, lx, ly, lz, False)
    s.store(o_r, s.fmul(s.fmul(lrgb[0], atten), f[0]))
    s.store(o_g, s.fmul(s.fmul(lrgb[1], atten), f[1]))
    s.store(o_b, s.fmul(s.fmul(lrgb[2], atten), f[2]))
    s.end_if()
    return (s.load_f(o_r), s.load_f(o_g), s.load_f(o_b))


def emit_sota_mat_rt_hit(
    mut s: AaaSpirvModule,
    b_mats: UInt32,
    r_flags: UInt32,
    mat_id: UInt32,
    m_base: UInt32,
    n: Tuple[UInt32, UInt32, UInt32],
    d: Tuple[UInt32, UInt32, UInt32],
    t_raw: Tuple[UInt32, UInt32, UInt32],
    t_sign: UInt32,
    alb: Tuple[UInt32, UInt32, UInt32],
    rough: UInt32,
    metal: UInt32,
    em: Tuple[UInt32, UInt32, UInt32],
    sun_l: Tuple[UInt32, UInt32, UInt32],
    sun_rgb: Tuple[UInt32, UInt32, UInt32],
    sun_vis: UInt32,
    loc_ir: Tuple[UInt32, UInt32, UInt32],
    loc_sp: Tuple[UInt32, UInt32, UInt32],
    amb: Tuple[UInt32, UInt32, UInt32],
    env_cube: Tuple[UInt32, UInt32],
    max_mip: UInt32,
    ibl_int: UInt32,
    hit_ao: UInt32,
    sc_ok: UInt32,
    sc_rgb: Tuple[UInt32, UInt32, UInt32],
    b_weight: UInt32,
    out_r: UInt32,
    out_g: UInt32,
    out_b: UInt32,
):
    """Layered shading of a ray hit (RT reflections, transmission rays, any caller of `emit_rt_textured_hit_shade`),
    so indirect views of a material match its primary view: exact sun BSDF, split-sum weighted ambient / local /
    environment terms, transmission through the hit, and the surface-cache multi-bounce excess measured against the
    layered hemispherical albedo. Writes `out_*` only when the model is on for this material."""
    var on = emit_sota_mat_on_for_material(s, r_flags, b_mats, m_base)
    s.begin_if(on)
    var one = s.float(1.0)
    var zero = s.float(0.0)
    var v = (s.fneg(d[0]), s.fneg(d[1]), s.fneg(d[2]))
    # Tangent with a perpendicular fallback (meshes without tangents)
    var tdn = emit_vec3_dot(s, t_raw[0], t_raw[1], t_raw[2], n[0], n[1], n[2])
    var gx = s.fsub(t_raw[0], s.fmul(tdn, n[0]))
    var gy = s.fsub(t_raw[1], s.fmul(tdn, n[1]))
    var gz = s.fsub(t_raw[2], s.fmul(tdn, n[2]))
    var g_ok = s.fgt(emit_vec3_dot(s, gx, gy, gz, gx, gy, gz), s.float(1e-8))
    var up_y = s.flt(s.fabs(n[1]), s.float(0.92))
    var fb = emit_vec3_cross(s, s.select_f(up_y, zero, one), s.select_f(up_y, one, zero), zero, n[0], n[1], n[2])
    var t = emit_vec3_normalize(s, s.select_f(g_ok, gx, fb[0]), s.select_f(g_ok, gy, fb[1]), s.select_f(g_ok, gz, fb[2]))
    var bc = emit_vec3_cross(s, n[0], n[1], n[2], t[0], t[1], t[2])
    var b = (s.fmul(bc[0], t_sign), s.fmul(bc[1], t_sign), s.fmul(bc[2], t_sign))
    var inp = emit_sota_mat_inputs_from_words(s, b_mats, m_base, n, v, sun_l, t, b, alb, rough, metal)
    var lb = emit_sota_mat_lobes(s, b_mats, mat_id, on, inp, SotaMatSurfaceExtra())
    var f = emit_sota_mat_eval_light(s, b_mats, lb, sun_l[0], sun_l[1], sun_l[2], False)
    var w = emit_sota_mat_ltc_weights(s, b_mats, lb)
    var m = lb.get(s, sota_mat_lb_index("metal"))
    var trans = lb.get(s, sota_mat_lb_index("trans"))
    var att_cc = lb.get(s, sota_mat_lb_index("att_cc"))
    var att = s.fmul(att_cc, lb.get(s, sota_mat_lb_index("att_sh")))
    var es_v = lb.get3(s, sota_mat_lb_index("esv"))
    var tint = lb.get3(s, sota_mat_lb_index("tint"))
    var kt = s.fmul(s.fmul(s.fsub(one, m), trans), att)
    var r_eff = lb.get(s, sota_mat_lb_index("reff"))
    var two_nd = s.fmul(s.float(2.0), emit_vec3_dot(s, n[0], n[1], n[2], d[0], d[1], d[2]))
    var rv = s.vec3(s.fsub(d[0], s.fmul(n[0], two_nd)), s.fsub(d[1], s.fmul(n[1], two_nd)), s.fsub(d[2], s.fmul(n[2], two_nd)))
    var env_s = s.sample_cube_lod(env_cube, rv, s.fmul(r_eff, max_mip))
    var env_c = s.sample_cube_lod(env_cube, rv, s.fmul(lb.get(s, sota_mat_lb_index("cc_rough")), max_mip))
    var env_t = s.sample_cube_lod(env_cube, s.vec3(d[0], d[1], d[2]), s.fmul(r_eff, max_mip))
    var k = s.fmul(ibl_int, hit_ao)
    var a_avg = emit_sota_mat_albedo_avg(s, b_mats, lb)
    var a3 = [a_avg[0], a_avg[1], a_avg[2]]
    var nl = s.nmax(zero, emit_vec3_dot(s, n[0], n[1], n[2], sun_l[0], sun_l[1], sun_l[2]))
    var sun_ir_k = s.fmul(s.fmul(nl, sun_vis), s.float(0.318309886))
    var wc = s.load_f(w.wc)
    var f3 = [f[0], f[1], f[2]]
    var sun3 = [sun_rgb[0], sun_rgb[1], sun_rgb[2]]
    var li3 = [loc_ir[0], loc_ir[1], loc_ir[2]]
    var ls3 = [loc_sp[0], loc_sp[1], loc_sp[2]]
    var am3 = [amb[0], amb[1], amb[2]]
    var em3 = [em[0], em[1], em[2]]
    var sc3 = [sc_rgb[0], sc_rgb[1], sc_rgb[2]]
    var outs = [out_r, out_g, out_b]
    for c in range(3):
        var direct = s.fmul(s.fmul(sun3[c], sun_vis), f3[c])
        var irr = s.fadd(li3[c], am3[c])
        var diff = s.fmul(s.load_f(w.wd[c]), irr)
        var wt_c = s.fmul(s.fmul(tint[c], s.fsub(one, es_v[c])), kt)
        var spec = s.fma(
            s.fmul(s.load_f(w.ws[c]), s.extract_f(env_s, c)), k,
            s.fma(ls3[c], att, s.fmul(s.fma(wc, s.extract_f(env_c, c), s.fmul(wt_c, s.extract_f(env_t, c))), k)),
        )
        var exc = s.nmax(zero, s.fsub(s.fsub(sc3[c], em3[c]), s.fmul(a3[c], s.fma(sun3[c], sun_ir_k, irr))))
        var exc_w = s.select_f(sc_ok, s.fmul(exc, b_weight), zero)
        s.store(outs[c], s.fadd(s.fadd(direct, diff), s.fadd(s.fadd(spec, s.fmul(em3[c], att_cc)), exc_w)))
    s.end_if()


def emit_sota_mat_surface_cache_albedo(
    mut s: AaaSpirvModule,
    b_mats: UInt32,
    r_flags: UInt32,
    mat_id: UInt32,
    m_base: UInt32,
    legacy_kd: Tuple[UInt32, UInt32, UInt32],
    alb: Tuple[UInt32, UInt32, UInt32],
    metal: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Surface-cache bounce albedo: the layered model's hemispherical albedo (`sota_mat_albedo_avg_cpu`: base
    diffuse + specular + sheen + coat energy, so coated / metallic / sheen surfaces bounce what they reflect)
    instead of the legacy `albedo * (1 - metal)`. Legacy value when off for this material."""
    var o_r = s.fvar(legacy_kd[0])
    var o_g = s.fvar(legacy_kd[1])
    var o_b = s.fvar(legacy_kd[2])
    var on = emit_sota_mat_on_for_material(s, r_flags, b_mats, m_base)
    s.begin_if(on)
    var z = s.float(0.0)
    var o1 = s.float(1.0)
    var inp = emit_sota_mat_inputs_from_words(
        s, b_mats, m_base, (z, z, o1), (z, s.float(0.8), s.float(0.6)), (z, z, o1), (o1, z, z), (z, o1, z),
        alb, s.load_float(b_mats, s.iadd(m_base, s.uint(4))), metal,
    )
    var lb = emit_sota_mat_lobes(s, b_mats, mat_id, on, inp, SotaMatSurfaceExtra())
    var a = emit_sota_mat_albedo_avg(s, b_mats, lb)
    s.store(o_r, a[0])
    s.store(o_g, a[1])
    s.store(o_b, a[2])
    s.end_if()
    return (s.load_f(o_r), s.load_f(o_g), s.load_f(o_b))
