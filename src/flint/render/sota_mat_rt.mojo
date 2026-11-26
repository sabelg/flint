"""Area I round 3: ray-traced transmission for the layered model (deferred pass, RT reflections on).

Thin-walled (`volume_distance == 0`, KHR_materials_transmission): the surface has no thickness, light passes
straight through, so a ray continues along -v from just behind the surface.
Volume (`volume_distance > 0`, KHR_materials_volume): refract into the medium at the shading normal (IOR from
the dielectric F0, `n = (1 + sqrt(F0)) / (1 - sqrt(F0))`), find the exit with an opaque-forced ray query,
refract out at the interpolated exit normal (total internal reflection reflects), apply Beer-Lambert
`T = attenuation_colour ^ (d / attenuation_distance)` over the inside path, then shade what the exit ray hits
with `emit_rt_textured_hit_shade`.
Rough transmission fades to the pre-filtered environment along -v between r_eff 0.15 and 0.45 (the compose
fallback). The result goes into `SotaMatShade.tr_*`; compose multiplies it by the transmission weight.
"""

from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import AAA_INSTANCE_WORDS, emit_mat4_mul_dir, emit_vec3_dot, emit_vec3_normalize
from flint.render.aaa_raytracing import RAY_FLAG_OPAQUE_KHR, emit_rt_textured_hit_shade
from flint.render.sota_mat_core import SotaMatShade, sota_mat_lb_index


def _e_refract(
    mut s: AaaSpirvModule, i: Tuple[UInt32, UInt32, UInt32], n: Tuple[UInt32, UInt32, UInt32], eta: UInt32
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """GLSL refract(I, N, eta) plus a TIR flag (1.0 = total internal reflection; the direction is then reflect)."""
    var one = s.float(1.0)
    var ni = emit_vec3_dot(s, n[0], n[1], n[2], i[0], i[1], i[2])
    var k = s.fsub(one, s.fmul(s.fmul(eta, eta), s.fsub(one, s.fmul(ni, ni))))
    var tir = s.flt(k, s.float(0.0))
    var a = s.fadd(s.fmul(eta, ni), s.fsqrt(s.nmax(s.float(0.0), k)))
    var two_ni = s.fmul(s.float(2.0), ni)
    var o = List[UInt32]()
    var ii = [i[0], i[1], i[2]]
    var nn = [n[0], n[1], n[2]]
    for c in range(3):
        var rf = s.fsub(s.fmul(eta, ii[c]), s.fmul(a, nn[c]))
        var rl = s.fsub(ii[c], s.fmul(two_ni, nn[c]))
        o.append(s.select_f(tir, rl, rf))
    var d = emit_vec3_normalize(s, o[0], o[1], o[2])
    return (d[0], d[1], d[2], s.select_f(tir, one, s.float(0.0)))


def emit_sota_mat_transmission_trace(
    mut s: AaaSpirvModule,
    sh: SotaMatShade,
    rt_on: UInt32,
    tlas_var: UInt32,
    b_verts: UInt32,
    b_indices: UInt32,
    b_instances: UInt32,
    b_materials: UInt32,
    b_globals: UInt32,
    b_vsm: UInt32,
    b_surfcache: UInt32,
    b_clusters: UInt32,
    vsm_samp: Tuple[UInt32, UInt32],
    env_cube: Tuple[UInt32, UInt32],
    bindless_arr: Tuple[UInt32, UInt32],
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    gn: Tuple[UInt32, UInt32, UInt32],
    cam_dist: UInt32,
    ibl_int: UInt32,
    max_mip: UInt32,
):
    """Traces the transmitted radiance for a layered transmissive pixel into `sh.tr_*` (see module doc).
    `gn` is the geometric normal facing the viewer."""
    var lb = sh.lb.copy()
    var zero = s.float(0.0)
    var one = s.float(1.0)
    var wt_max = s.nmax(s.load_f(sh.wt_r), s.nmax(s.load_f(sh.wt_g), s.load_f(sh.wt_b)))
    var r_eff = s.load_f(sh.r_eff)
    var cond = s.logical_and(s.logical_and(sh.on, rt_on), s.logical_and(s.fgt(wt_max, zero), s.flt(r_eff, s.float(0.45))))
    s.begin_if(cond)
    var v = lb.get3(s, sota_mat_lb_index("v"))
    var n = lb.get3(s, sota_mat_lb_index("n"))
    var vol = lb.get(s, sota_mat_lb_index("vol"))
    var atten = lb.get3(s, sota_mat_lb_index("atten"))
    var mi = (s.fneg(v[0]), s.fneg(v[1]), s.fneg(v[2]))
    var eps = s.float(0.006)
    var o_x = s.fvar(s.fsub(wx, s.fmul(gn[0], eps)))
    var o_y = s.fvar(s.fsub(wy, s.fmul(gn[1], eps)))
    var o_z = s.fvar(s.fsub(wz, s.fmul(gn[2], eps)))
    var d_x = s.fvar(mi[0])
    var d_y = s.fvar(mi[1])
    var d_z = s.fvar(mi[2])
    var beer = [s.fvar(one), s.fvar(one), s.fvar(one)]
    s.begin_if(s.fgt(vol, zero))
    var f0 = s.fmul(s.float(0.04), lb.get(s, sota_mat_lb_index("spec")))
    var sq = s.fsqrt(s.nclamp(f0, s.float(1e-4), s.float(0.5)))
    var ior = s.nclamp(s.fdiv(s.fadd(one, sq), s.fsub(one, sq)), s.float(1.01), s.float(3.0))
    var d1 = _e_refract(s, mi, (n[0], n[1], n[2]), s.fdiv(one, ior))
    s.store(d_x, d1[0])
    s.store(d_y, d1[1])
    s.store(d_z, d1[2])
    var rq = s.ray_query_var()
    s.ray_query_init(
        rq, tlas_var, s.uint(RAY_FLAG_OPAQUE_KHR), s.uint(0xFF),
        s.vec3(s.load_f(o_x), s.load_f(o_y), s.load_f(o_z)), s.float(0.001), s.vec3(d1[0], d1[1], d1[2]), s.float(64.0),
    )
    s.begin_loop()
    s.break_if(s.logical_not(s.ray_query_proceed(rq)))
    s.end_loop()
    s.begin_if(s.ine(s.ray_query_committed_type(rq), s.uint(0)))
    var t_e = s.ray_query_committed_t(rq)
    var inst = s.ray_query_committed_instance_id(rq)
    var prim = s.ray_query_committed_prim_index(rq)
    var bary = s.ray_query_committed_barycentrics(rq)
    var b1 = s.extract_f(bary, 0)
    var b2 = s.extract_f(bary, 1)
    var b0 = s.fsub(s.fsub(one, b1), b2)
    var ibase = s.imul(inst, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var ioff = s.iadd(s.load_word(b_instances, s.iadd(ibase, s.uint(38))), s.imul(prim, s.uint(3)))
    var vo0 = s.imul(s.load_word(b_indices, ioff), s.uint(12))
    var vo1 = s.imul(s.load_word(b_indices, s.iadd(ioff, s.uint(1))), s.uint(12))
    var vo2 = s.imul(s.load_word(b_indices, s.iadd(ioff, s.uint(2))), s.uint(12))
    var ln = List[UInt32]()
    for c in range(3):
        var k = s.uint(UInt32(3 + c))
        ln.append(s.fma(
            s.load_float(b_verts, s.iadd(vo0, k)), b0,
            s.fma(s.load_float(b_verts, s.iadd(vo1, k)), b1, s.fmul(s.load_float(b_verts, s.iadd(vo2, k)), b2)),
        ))
    var wn = emit_mat4_mul_dir(s, b_instances, ibase, ln[0], ln[1], ln[2])
    var ne = emit_vec3_normalize(s, wn[0], wn[1], wn[2])
    # Outward exit normal (along the inside ray), refract out against it
    var out_s = s.select_f(s.flt(emit_vec3_dot(s, ne[0], ne[1], ne[2], d1[0], d1[1], d1[2]), zero), s.float(-1.0), one)
    var nox = s.fmul(ne[0], out_s)
    var noy = s.fmul(ne[1], out_s)
    var noz = s.fmul(ne[2], out_s)
    var d2 = _e_refract(s, (d1[0], d1[1], d1[2]), (s.fneg(nox), s.fneg(noy), s.fneg(noz)), ior)
    var px = s.fma(d1[0], t_e, s.load_f(o_x))
    var py = s.fma(d1[1], t_e, s.load_f(o_y))
    var pz = s.fma(d1[2], t_e, s.load_f(o_z))
    # TIR: the ray stays inside (start just inside the exit), otherwise just outside
    var side = s.select_f(s.fgt(d2[3], s.float(0.5)), s.fneg(eps), eps)
    s.store(o_x, s.fma(nox, side, px))
    s.store(o_y, s.fma(noy, side, py))
    s.store(o_z, s.fma(noz, side, pz))
    s.store(d_x, d2[0])
    s.store(d_y, d2[1])
    s.store(d_z, d2[2])
    var path = s.fdiv(t_e, s.nmax(s.float(1e-4), vol))
    for c in range(3):
        s.store(beer[c], s.exp2(s.fmul(s.log2(s.nmax(s.float(1e-4), atten[c])), path)))
    s.end_if()
    s.end_if()
    var dx = s.load_f(d_x)
    var dy = s.load_f(d_y)
    var dz = s.load_f(d_z)
    var mip = s.fmul(r_eff, max_mip)
    var env_m = s.sample_cube_lod(env_cube, s.vec3(dx, dy, dz), mip)
    var hit = emit_rt_textured_hit_shade(
        s, tlas_var, b_verts, b_indices, b_instances, b_materials, b_globals, b_vsm, b_surfcache, b_clusters,
        vsm_samp, env_cube, bindless_arr,
        s.load_f(o_x), s.load_f(o_y), s.load_f(o_z), dx, dy, dz,
        s.float(0.004), s.float(260.0), cam_dist, s.fmul(r_eff, r_eff),
        s.fmul(s.extract_f(env_m, 0), ibl_int), s.fmul(s.extract_f(env_m, 1), ibl_int), s.fmul(s.extract_f(env_m, 2), ibl_int),
    )
    var env_v = s.sample_cube_lod(env_cube, s.vec3(mi[0], mi[1], mi[2]), mip)
    var fade = s.nclamp(s.fdiv(s.fsub(s.float(0.45), r_eff), s.float(0.3)), zero, one)
    var trs = [sh.tr_r, sh.tr_g, sh.tr_b]
    var hit3 = [hit[0], hit[1], hit[2]]
    for c in range(3):
        var traced = s.fmul(hit3[c], s.load_f(beer[c]))
        var fb = s.fmul(s.extract_f(env_v, c), ibl_int)
        s.store(trs[c], s.fma(s.fsub(traced, fb), fade, fb))
    s.store(sh.tr_ok, one)
    s.end_if()
