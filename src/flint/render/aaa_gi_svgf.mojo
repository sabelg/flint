"""Milestone 3 (`V14`): Lumen-Class Multi-Bounce Mesh Surface Cache, Full Deferred
`VisBuffer` Multi-Lobe + VSM + `SPV_KHR_ray_query` Shader, `SVGF` Wavelet Denoiser &
512-spp Unbiased Monte-Carlo Reference Path Tracer (`aaa_gi_svgf.mojo`).
"""
from std.math import cos, exp, max, min, sin, sqrt
from flint.core.math import Vec2, Vec3
from flint.core.transform import Mat4
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
# --- sota:camera hook (import) ---
from flint.render.sota_camera_vrs import emit_sota_vrs_remap
# --- end sota:camera hook ---
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_BRDF_LUT,
    BIND_AAA_CLUSTERS,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_LTC_LUT,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_SKINNING,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_instance_point,
    emit_ray_barycentrics,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_bsdf import (
    AAA_MATERIAL_WORDS,
    AAA_MAT_DET_NONE,
    AAA_MAT_INTERIOR_PARALLAX,
    emit_direct_multilobe_bsdf,
    emit_iridescence_f0,
    emit_kaplanyan_specular_aa,
    emit_rnm_blend,
)
from flint.render.aaa_vsm import emit_vsm_pcss_shadow
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.aaa_raytracing import (
    RAY_FLAG_OPAQUE_KHR,
    emit_decode_vertex_color_rgb_spv,
    emit_froxel_local_lighting_at_hit,
    emit_rt_shadow_query,
    emit_rt_surface_cache_ray,
    emit_rt_textured_hit_shade,
    emit_window_suite_prng_spv,
)
from flint.render.aaa_clustered_lights import (
    AAA_CLUSTER_LIGHTS_BASE,
    AAA_CLUSTER_LIGHT_WORDS,
    emit_eval_clustered_and_ltc_lights,
    emit_eval_single_cluster_light,
    emit_restir_candidate_selection_spv,
)
from flint.render.aaa_skinning_rt import AAA_SKIN_PREV_POS_BASE
from flint.render.aaa_sky_atmosphere import (
    emit_aerial_perspective_spv,
    emit_nubis_sky_clouds_and_celestial_spv,
)

# Render feature flags in `BIND_AAA_GLOBALS[99]`
comptime AAA_FLAG_VSM: UInt32 = 1
comptime AAA_FLAG_RT_SHADOWS: UInt32 = 2
comptime AAA_FLAG_RT_REFLECTIONS: UInt32 = 4
comptime AAA_FLAG_RT_SURFACE_GI: UInt32 = 8
comptime AAA_FLAG_FURNACE_WHITE_DOME: UInt32 = 16
comptime AAA_FLAG_CLUSTERED_LIGHTS: UInt32 = 32
comptime AAA_FLAG_AERIAL_PERSPECTIVE: UInt32 = 64
comptime AAA_FLAG_RESTIR_DI: UInt32 = 128
comptime AAA_FLAG_GLASS: UInt32 = 256  # some instance this frame uses a glass material (`aaa_glass.mojo`)


@always_inline
def svgf_accumulate_moments_cpu(
    prev_m1: Float32,
    prev_m2: Float32,
    luma: Float32,
    history_len: Float32,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Evaluates SVGF temporal exponential moving average of luminance moments and variance.
    Returns: (m1, m2, variance, new_history_len).
    """
    var new_len = min(Float32(24.0), history_len + Float32(1.0))
    var alpha = Float32(1.0) / new_len
    var om_alpha = Float32(1.0) - alpha
    var m1 = om_alpha * prev_m1 + alpha * luma
    var m2 = om_alpha * prev_m2 + alpha * (luma * luma)
    var variance = max(Float32(0.0), m2 - m1 * m1)
    if new_len < Float32(4.0):
        variance *= (Float32(4.0) / new_len)
    return (m1, m2, variance, new_len)


def eval_virtual_depth_reprojection_cpu(
    cam_pos: Vec3,
    primary_pos: Vec3,
    t_hit: Float32,
    hit_valid: Bool,
    prev_view_proj: List[Float32],
    cur_uv: Vec2,
) -> Tuple[Float32, Vec2]:
    """Computes virtual depth z_virtual = ||P - C|| + t_hit and the reprojected specular UV.
    Returns: (z_virtual, prev_specular_uv).
    """
    var d_primary = (primary_pos - cam_pos).length()
    var z_virtual = (d_primary + t_hit) if hit_valid else Float32(100000.0)

    var cam_ray_dir = (primary_pos - cam_pos).normalized()
    var p_virtual = cam_pos + cam_ray_dir * z_virtual

    # Project with prev_view_proj (4x4 column-major matrix)
    var clip_x = prev_view_proj[0] * p_virtual.x + prev_view_proj[4] * p_virtual.y + prev_view_proj[8] * p_virtual.z + prev_view_proj[12]
    var clip_y = prev_view_proj[1] * p_virtual.x + prev_view_proj[5] * p_virtual.y + prev_view_proj[9] * p_virtual.z + prev_view_proj[13]
    var clip_w = prev_view_proj[3] * p_virtual.x + prev_view_proj[7] * p_virtual.y + prev_view_proj[11] * p_virtual.z + prev_view_proj[15]

    var inv_w = Float32(1.0) / (clip_w if abs(clip_w) > Float32(1e-5) else Float32(1e-5))
    var prev_uv = Vec2(
        (clip_x * inv_w) * Float32(0.5) + Float32(0.5),
        -(clip_y * inv_w) * Float32(0.5) + Float32(0.5),
    )
    return (z_virtual, prev_uv)


def eval_cofactor_gram_schmidt_tbn_cpu(
    m: Mat4,
    ln: Vec3,
    lt: Vec3,
    bitan_sign: Float32 = 1.0,
) -> Tuple[Vec3, Vec3, Vec3]:
    """CPU reference for Cofactor Normal Matrix + Gram-Schmidt Orthonormal TBN under non-uniform scale."""
    var c0 = Vec3(m.values[0], m.values[1], m.values[2])
    var c1 = Vec3(m.values[4], m.values[5], m.values[6])
    var c2 = Vec3(m.values[8], m.values[9], m.values[10])
    var cof0 = c1.cross(c2)
    var cof1 = c2.cross(c0)
    var cof2 = c0.cross(c1)
    var det = c0.dot(cof0)
    var d_sign = Float32(-1.0) if det < Float32(0.0) else Float32(1.0)
    var wn_raw = (cof0 * ln.x + cof1 * ln.y + cof2 * ln.z) * d_sign
    var wn = wn_raw.normalized() if wn_raw.length_squared() > Float32(1e-12) else Vec3(0.0, 1.0, 0.0)

    var wt_raw = c0 * lt.x + c1 * lt.y + c2 * lt.z
    var wt_gs = wt_raw - wn * wt_raw.dot(wn)
    var wt: Vec3
    if wt_gs.length_squared() > Float32(1e-8):
        wt = wt_gs.normalized()
    else:
        var h = Vec3(0.0, 1.0, 0.0) if abs(wn.y) < Float32(0.92) else Vec3(1.0, 0.0, 0.0)
        wt = h.cross(wn).normalized()
    var s_bit = Float32(-1.0) if bitan_sign < Float32(0.0) else Float32(1.0)
    var wb = wn.cross(wt) * (s_bit * d_sign)
    return (wn, wt, wb)



def _emit_pcg_hash_f01(mut s: AaaSpirvModule, seed_in: UInt32) -> Tuple[UInt32, UInt32]:
    """Fast GPU PCG hash returning `(next_seed_u32, uniform_float_01)`."""
    var st = s.iadd(s.imul(seed_in, s.uint(747796405)), s.uint(2891336453))
    var wd = s.imul(s.bit_xor(s.shr(st, s.iadd(s.shr(st, s.uint(28)), s.uint(4))), st), s.uint(277803737))
    var res = s.bit_xor(s.shr(wd, s.uint(22)), wd)
    var f01 = s.fmul(s.float_from_uint(s.bit_and(res, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))
    return (res, f01)


def _emit_cosine_hemisphere_dir(
    mut s: AaaSpirvModule,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    u1: UInt32,
    u2: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Builds an orthonormal basis around `N` and maps `(u1, u2)` to a cosine-weighted hemisphere ray!"""
    var r = s.fsqrt(s.nclamp(u1, s.float(0.0), s.float(1.0)))
    var phi = s.fmul(u2, s.float(6.283185307))
    var lx = s.fmul(r, s.fcos(phi))
    var ly = s.fmul(r, s.fsin(phi))
    var lz = s.fsqrt(s.nmax(s.float(0.0001), s.fsub(s.float(1.0), u1)))

    var up_is_y = s.flt(s.fabs(ny), s.float(0.92))
    var hx = s.select_f(up_is_y, s.float(0.0), s.float(1.0))
    var hy = s.select_f(up_is_y, s.float(1.0), s.float(0.0))
    var hz = s.float(0.0)
    var t_raw = emit_vec3_cross(s, hx, hy, hz, nx, ny, nz)
    var tang = emit_vec3_normalize(s, t_raw[0], t_raw[1], t_raw[2])
    var bitan = emit_vec3_cross(s, nx, ny, nz, tang[0], tang[1], tang[2])

    var wx = s.fma(tang[0], lx, s.fma(bitan[0], ly, s.fmul(nx, lz)))
    var wy = s.fma(tang[1], lx, s.fma(bitan[1], ly, s.fmul(ny, lz)))
    var wz = s.fma(tang[2], lx, s.fma(bitan[2], ly, s.fmul(nz, lz)))
    return emit_vec3_normalize(s, wx, wy, wz)


def _emit_decode_vertex_ao_spv(mut s: AaaSpirvModule, meta: UInt32) -> UInt32:
    """Decodes per-vertex AO supporting BOTH legacy `(ao_u8 << 8) | 0xFFFF0000` and new `bits 2..7` (`0..63`)."""
    var is_legacy = s.ieq(s.bit_and(meta, s.uint(0xFFFF0000)), s.uint(0xFFFF0000))
    var leg_ao = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta, s.uint(8)), s.uint(0xFF))), s.float(1.0 / 255.0))
    var new_ao = s.fmul(s.float_from_uint(s.bit_and(s.shr(meta, s.uint(2)), s.uint(0x3F))), s.float(1.0 / 63.0))
    return s.select_f(is_legacy, leg_ao, new_ao)


def _emit_jimenez_multibounce_ao_spv(
    mut s: AaaSpirvModule,
    vis_ao: UInt32,
    alb_r: UInt32,
    alb_g: UInt32,
    alb_b: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Jimenez et al. (SIGGRAPH 2016) cubic multi-bounce albedo-tinted ambient occlusion (`1.0` at `vis_ao=1.0`)."""
    var out = List[UInt32]()
    for rho in [alb_r, alb_g, alb_b]:
        var a = s.fma(s.float(2.0404), rho, s.float(-0.3324))
        var b = s.fma(s.float(-4.7951), rho, s.float(0.6417))
        var c = s.fma(s.float(2.7552), rho, s.float(0.6903))
        var poly = s.fmul(s.fma(s.fma(vis_ao, a, b), vis_ao, c), vis_ao)
        out.append(s.nclamp(s.nmax(vis_ao, poly), s.float(0.0), s.float(1.0)))
    return (out[0], out[1], out[2])


def _emit_gtso_spec_occlusion_spv(
    mut s: AaaSpirvModule,
    ndotv: UInt32,
    vis_ao: UInt32,
    roughness: UInt32,
) -> UInt32:
    """Lagarde & de Rousiers / Jimenez Ground-Truth Specular Occlusion (`GTSO`):
    `clamp((NdotV + ao)^exp2(-16*alpha - 1) - 1 + ao, 0, 1)`. Equals `1.0000` when `ao = 1.0`!"""
    var alpha = s.fmul(roughness, roughness)
    var expo = s.exp2(s.fma(s.float(-16.0), alpha, s.float(-1.0)))
    var base = s.nmax(s.float(1e-4), s.fadd(ndotv, vis_ao))
    return s.nclamp(s.fadd(s.fsub(s.fpow(base, expo), s.float(1.0)), vis_ao), s.float(0.0), s.float(1.0))


def _emit_heitz_bounded_vndf_dir(
    mut s: AaaSpirvModule,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    gnx: UInt32,
    gny: UInt32,
    gnz: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    eff_rough: UInt32,
    u1: UInt32,
    u2: UInt32,
    use_stochastic: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """SPIR-V Heitz 2018 Spherical-Cap Bounded `VNDF` reflection direction sampler.
    Falls back cleanly to deterministic mirror reflection when `eff_rough <= 0.048` or `not use_stochastic`."""
    var two_nv = s.fmul(s.float(2.0), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
    var mir_x = s.fsub(s.fmul(nx, two_nv), vx)
    var mir_y = s.fsub(s.fmul(ny, two_nv), vy)
    var mir_z = s.fsub(s.fmul(nz, two_nv), vz)

    var out_x = s.fvar(mir_x)
    var out_y = s.fvar(mir_y)
    var out_z = s.fvar(mir_z)

    s.begin_if(s.logical_and(use_stochastic, s.fgt(eff_rough, s.float(0.048))))
    var up_is_y = s.flt(s.fabs(ny), s.float(0.92))
    var hx_u = s.select_f(up_is_y, s.float(0.0), s.float(1.0))
    var hy_u = s.select_f(up_is_y, s.float(1.0), s.float(0.0))
    var t_raw = emit_vec3_cross(s, hx_u, hy_u, s.float(0.0), nx, ny, nz)
    var tang = emit_vec3_normalize(s, t_raw[0], t_raw[1], t_raw[2])
    var bitan = emit_vec3_cross(s, nx, ny, nz, tang[0], tang[1], tang[2])

    var vlx = emit_vec3_dot(s, vx, vy, vz, tang[0], tang[1], tang[2])
    var vly = emit_vec3_dot(s, vx, vy, vz, bitan[0], bitan[1], bitan[2])
    var vlz = s.nmax(s.float(1e-4), emit_vec3_dot(s, vx, vy, vz, nx, ny, nz))

    # Bounded alpha prevents extreme tail fireflies while preserving full anisotropic wet-road elongation
    var alpha = s.nclamp(s.fmul(s.fmul(eff_rough, eff_rough), s.float(0.82)), s.float(0.002), s.float(0.32))
    var vh = emit_vec3_normalize(s, s.fmul(alpha, vlx), s.fmul(alpha, vly), vlz)

    var lensq = s.fma(vh[0], vh[0], s.fmul(vh[1], vh[1]))
    var has_t1 = s.fgt(lensq, s.float(1e-6))
    var inv_l1 = s.inverse_sqrt(s.nmax(s.float(1e-8), lensq))
    var t1x = s.select_f(has_t1, s.fmul(s.fneg(vh[1]), inv_l1), s.float(1.0))
    var t1y = s.select_f(has_t1, s.fmul(vh[0], inv_l1), s.float(0.0))
    var t1z = s.float(0.0)
    var t2 = emit_vec3_cross(s, vh[0], vh[1], vh[2], t1x, t1y, t1z)

    var rad = s.fsqrt(s.nclamp(u1, s.float(0.0), s.float(0.999)))
    var phi = s.fmul(u2, s.float(6.283185307))
    var p1 = s.fmul(rad, s.fcos(phi))
    var p2_raw = s.fmul(rad, s.fsin(phi))
    var s_cap = s.fmul(s.float(0.5), s.fadd(s.float(1.0), vh[2]))
    var p2 = s.fma(
        s.fsub(s.float(1.0), s_cap),
        s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(p1, p1)))),
        s.fmul(s_cap, p2_raw),
    )
    var p3 = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(p1, p1, s.fmul(p2, p2)))))

    var nhx = s.fma(t1x, p1, s.fma(t2[0], p2, s.fmul(vh[0], p3)))
    var nhy = s.fma(t1y, p1, s.fma(t2[1], p2, s.fmul(vh[1], p3)))
    var nhz = s.fma(t1z, p1, s.fma(t2[2], p2, s.fmul(vh[2], p3)))
    var h_loc = emit_vec3_normalize(s, s.fmul(alpha, nhx), s.fmul(alpha, nhy), s.nmax(s.float(1e-4), nhz))

    var hwx = s.fma(tang[0], h_loc[0], s.fma(bitan[0], h_loc[1], s.fmul(nx, h_loc[2])))
    var hwy = s.fma(tang[1], h_loc[0], s.fma(bitan[1], h_loc[1], s.fmul(ny, h_loc[2])))
    var hwz = s.fma(tang[2], h_loc[0], s.fma(bitan[2], h_loc[1], s.fmul(nz, h_loc[2])))
    var hw = emit_vec3_normalize(s, hwx, hwy, hwz)

    var two_vh = s.fmul(s.float(2.0), s.nmax(s.float(0.0), emit_vec3_dot(s, vx, vy, vz, hw[0], hw[1], hw[2])))
    var l_cand = emit_vec3_normalize(
        s,
        s.fsub(s.fmul(hw[0], two_vh), vx),
        s.fsub(s.fmul(hw[1], two_vh), vy),
        s.fsub(s.fmul(hw[2], two_vh), vz),
    )
    var above_geom = s.fgt(emit_vec3_dot(s, gnx, gny, gnz, l_cand[0], l_cand[1], l_cand[2]), s.float(0.002))
    s.begin_if(above_geom)
    s.store(out_x, l_cand[0])
    s.store(out_y, l_cand[1])
    s.store(out_z, l_cand[2])
    s.end_if()
    s.end_if()

    return (s.load_f(out_x), s.load_f(out_y), s.load_f(out_z))


def _emit_parallax_windowbox_spv(
    mut s: AaaSpirvModule,
    bindless_arr: Tuple[UInt32, UInt32],
    t_det: UInt32,
    inst_id: UInt32,
    uv_u: UInt32,
    uv_v: UInt32,
    cam_dist: UInt32,
    ray_dx: UInt32,
    ray_dy: UInt32,
    ray_dz: UInt32,
    wt_x: UInt32,
    wt_y: UInt32,
    wt_z: UInt32,
    wb_x: UInt32,
    wb_y: UInt32,
    wb_z: UInt32,
    nx_var: UInt32,
    ny_var: UInt32,
    nz_var: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """GPU Tangent-Space 3D Ray-Box Parallax Interior Mapping (`[0,1]x[0,1]x[0,d_room]`):
    - Coherent `4`-bay office suites (`emit_window_suite_prng_spv`) + per-pane `±1.2 deg` glass warble
    - Strict surface separation: acoustic ceiling + `2x2` LED troffers, dark carpet floor, neutral side
      walls, and `4x4` `BC7` Room Atlas mapped EXCLUSIVELY onto the back wall (`tz < tx && tz < ty`)!
    - Distance-faded Venetian blinds (`20m..45m` -> zero distant tower Moiré!) + Kelvin tint!"""
    var suite_prng = emit_window_suite_prng_spv(s, inst_id, uv_u, uv_v)
    var suite_occ = suite_prng[0]
    var suite_kelvin = suite_prng[1]
    var pane_h0 = suite_prng[2]
    var pane_h1 = suite_prng[3]
    var suite_hash = suite_prng[4]

    var pu = s.fract(uv_u)
    var pv = s.fract(uv_v)

    # 1. Per-pane `±1.2 deg` (`sin(1.2 deg) = 0.021`) tempered glass warble + subtle pillow bow
    var wx_tilt = s.fma(s.fsub(pane_h0, s.float(0.5)), s.float(0.042), s.fmul(s.fsub(pu, s.float(0.5)), s.float(0.014)))
    var wy_tilt = s.fma(s.fsub(pane_h1, s.float(0.5)), s.float(0.042), s.fmul(s.fsub(pv, s.float(0.5)), s.float(0.014)))
    var cnx = s.load_f(nx_var); var cny = s.load_f(ny_var); var cnz = s.load_f(nz_var)
    var n_warb = emit_vec3_normalize(
        s,
        s.fma(wt_x, wx_tilt, s.fma(wb_x, wy_tilt, cnx)),
        s.fma(wt_y, wx_tilt, s.fma(wb_y, wy_tilt, cny)),
        s.fma(wt_z, wx_tilt, s.fma(wb_z, wy_tilt, cnz)),
    )
    s.store(nx_var, n_warb[0]); s.store(ny_var, n_warb[1]); s.store(nz_var, n_warb[2])

    # 2. Transform camera ray `D` into tangent-space room box `[0,1] x [0,1] x [0, d_room]`
    var rdu = emit_vec3_dot(s, ray_dx, ray_dy, ray_dz, wt_x, wt_y, wt_z)
    var rdv = emit_vec3_dot(s, ray_dx, ray_dy, ray_dz, wb_x, wb_y, wb_z)
    var rdz = s.nmax(s.float(0.05), s.fneg(emit_vec3_dot(s, ray_dx, ray_dy, ray_dz, cnx, cny, cnz)))

    var d_room = s.fma(suite_kelvin, s.float(0.35), s.float(0.68))
    var safe_u = s.select_f(s.fge(rdu, s.float(0.0)), s.nmax(rdu, s.float(1e-4)), s.nmin(rdu, s.float(-1e-4)))
    var safe_v = s.select_f(s.fge(rdv, s.float(0.0)), s.nmax(rdv, s.float(1e-4)), s.nmin(rdv, s.float(-1e-4)))
    var target_u = s.select_f(s.fge(rdu, s.float(0.0)), s.float(1.0), s.float(0.0))
    var target_v = s.select_f(s.fge(rdv, s.float(0.0)), s.float(1.0), s.float(0.0))
    var tx = s.fdiv(s.fsub(target_u, pu), safe_u)
    var ty = s.fdiv(s.fsub(target_v, pv), safe_v)
    var tz = s.fdiv(d_room, rdz)
    var t_hit = s.nmin(s.nmin(tx, ty), tz)

    var hu = s.nclamp(s.fma(rdu, t_hit, pu), s.float(0.0), s.float(1.0))
    var hv = s.nclamp(s.fma(rdv, t_hit, pv), s.float(0.0), s.float(1.0))
    var hz = s.nclamp(s.fmul(rdz, t_hit), s.float(0.0), d_room)
    var z_norm = s.fdiv(hz, d_room)

    # Strict architectural surface separation: ceiling vs floor vs back-wall vs side-walls!
    var hit_vert = s.logical_and(s.fle(ty, tx), s.fle(ty, tz))
    var up_sign_v = emit_vec3_dot(s, wb_x, wb_y, wb_z, s.float(0.0), s.float(1.0), s.float(0.0))
    var is_ceil = s.logical_and(hit_vert, s.fgt(s.fmul(rdv, up_sign_v), s.float(0.0)))
    var is_floor = s.logical_and(hit_vert, s.fle(s.fmul(rdv, up_sign_v), s.float(0.0)))
    var is_back_wall = s.logical_and(s.flt(tz, tx), s.flt(tz, ty))

    var wall_r = s.fvar(s.float(0.50))
    var wall_g = s.fvar(s.float(0.48))
    var wall_b = s.fvar(s.float(0.46))

    s.begin_if(is_ceil)
    # Acoustic ceiling grid (never maps wall bookshelves onto ceiling!)
    var cg_u = s.fabs(s.fsub(s.fract(s.fmul(hu, s.float(4.0))), s.float(0.5)))
    var cg_z = s.fabs(s.fsub(s.fract(s.fmul(z_norm, s.float(4.0))), s.float(0.5)))
    var tile_seam = s.select_f(s.logical_or(s.fgt(cg_u, s.float(0.46)), s.fgt(cg_z, s.float(0.46))), s.float(0.38), s.float(0.60))
    s.store(wall_r, tile_seam)
    s.store(wall_g, s.fmul(tile_seam, s.float(0.99)))
    s.store(wall_b, s.fmul(tile_seam, s.float(0.96)))
    s.begin_else()
    s.begin_if(is_floor)
    # Commercial dark carpet / concrete floor
    var flr_tex = s.fma(s.float(0.03), s.fsin(s.fma(hu, s.float(18.0), s.fmul(z_norm, s.float(18.0)))), s.float(0.15))
    s.store(wall_r, flr_tex)
    s.store(wall_g, s.fmul(flr_tex, s.float(0.97)))
    s.store(wall_b, s.fmul(flr_tex, s.float(0.94)))
    s.begin_else()
    s.begin_if(s.logical_and(is_back_wall, s.ult(t_det, s.uint(AAA_MAT_DET_NONE))))
    # Back wall ONLY: sample `4x4` `BC7` Interior Room Atlas (`0` shelf stretching onto ceiling/side-walls!)
    var acol = s.float_from_uint(s.bit_and(suite_hash, s.uint(3)))
    var arow = s.float_from_uint(s.bit_and(s.shr(suite_hash, s.uint(2)), s.uint(3)))
    var up_hv = s.select_f(s.fgt(up_sign_v, s.float(0.0)), hv, s.fsub(s.float(1.0), hv))
    var au = s.fmul(s.fadd(acol, s.nclamp(hu, s.float(0.04), s.float(0.96))), s.float(0.25))
    var av = s.fmul(s.fadd(arow, s.nclamp(up_hv, s.float(0.04), s.float(0.96))), s.float(0.25))
    var s_at = s.sample_bindless_lod(bindless_arr, t_det, s.vec2(au, av), s.float(1.0))
    s.store(wall_r, s.extract_f(s_at, 0))
    s.store(wall_g, s.extract_f(s_at, 1))
    s.store(wall_b, s.extract_f(s_at, 2))
    s.begin_else()
    # Architectural side-wall / fallback back-wall vertical paneling + wainscot trim
    var p_coord = s.select_f(is_back_wall, hu, z_norm)
    var grid_u = s.fabs(s.fsub(s.fract(s.fmul(p_coord, s.float(3.0))), s.float(0.5)))
    var up_w = s.select_f(s.fgt(up_sign_v, s.float(0.0)), hv, s.fsub(s.float(1.0), hv))
    var wainscot = s.select_f(s.flt(up_w, s.float(0.28)), s.float(0.30), s.fma(s.float(0.14), grid_u, s.float(0.46)))
    s.store(wall_r, s.fmul(wainscot, s.float(1.05)))
    s.store(wall_g, wainscot)
    s.store(wall_b, s.fmul(wainscot, s.float(0.93)))
    s.end_if()
    s.end_if()
    s.end_if()

    # Mid-depth furniture / cubicle partition plane at `z_mid = 0.52 * d_room`
    var t_mid = s.fdiv(s.fmul(d_room, s.float(0.52)), rdz)
    s.begin_if(s.flt(t_mid, t_hit))
    var mu = s.fma(rdu, t_mid, pu)
    var mv = s.fma(rdv, t_mid, pv)
    var height_up = s.select_f(s.fgt(up_sign_v, s.float(0.0)), mv, s.fsub(s.float(1.0), mv))
    var desk_h = s.fma(pane_h0, s.float(0.16), s.float(0.22))
    var in_desk = s.logical_and(
        s.flt(height_up, desk_h),
        s.logical_and(s.fgt(mu, s.float(0.16)), s.flt(mu, s.float(0.84))),
    )
    s.begin_if(in_desk)
    s.store(wall_r, s.fmul(s.load_f(wall_r), s.float(0.28)))
    s.store(wall_g, s.fmul(s.load_f(wall_g), s.float(0.28)))
    s.store(wall_b, s.fmul(s.load_f(wall_b), s.float(0.32)))
    s.end_if()
    s.end_if()

    # Recessed `2x2` Ceiling LED Troffers
    var troffer_boost = s.fvar(s.float(0.0))
    s.begin_if(is_ceil)
    var cu2 = s.fabs(s.fsub(s.fract(s.fmul(hu, s.float(2.0))), s.float(0.5)))
    var cv2 = s.fabs(s.fsub(s.fract(s.fmul(z_norm, s.float(2.0))), s.float(0.5)))
    var in_led = s.logical_and(s.flt(cu2, s.float(0.24)), s.flt(cv2, s.float(0.30)))
    s.store(troffer_boost, s.select_f(in_led, s.float(2.4), s.float(0.0)))
    s.end_if()

    # Distance-faded Venetian blind drop (`20m..45m` smooth fade to analytic mean `0.53` -> zero distant Moiré!)
    var top_dist = s.select_f(s.fgt(up_sign_v, s.float(0.0)), s.fsub(s.float(1.0), pv), pv)
    var blind_drop = s.fmul(pane_h1, s.float(0.46))
    var slats = s.nclamp(s.fma(s.float(0.55), s.fsin(s.fmul(top_dist, s.float(155.0))), s.float(0.45)), s.float(0.20), s.float(0.86))
    var bt_d = s.nclamp(s.fmul(s.fsub(cam_dist, s.float(20.0)), s.float(1.0 / 25.0)), s.float(0.0), s.float(1.0))
    var b_fade = s.fmul(s.fmul(bt_d, bt_d), s.fma(bt_d, s.float(-2.0), s.float(3.0)))
    var slat_eff = s.fma(s.fsub(s.float(0.53), slats), b_fade, slats)
    var blind_vis = s.select_f(s.flt(top_dist, blind_drop), slat_eff, s.float(1.0))

    # Window mullion border darkening + depth AO + coherent 4-bay suite occupancy & Kelvin temperature
    var edge_x = s.nmin(pu, s.fsub(s.float(1.0), pu))
    var edge_y = s.nmin(pv, s.fsub(s.float(1.0), pv))
    var frame_mask = s.nclamp(s.fmul(s.nmin(edge_x, edge_y), s.float(28.0)), s.float(0.0), s.float(1.0))
    var depth_ao = s.fma(s.float(-0.46), z_norm, s.float(1.0))

    var is_lit_room = s.flt(suite_occ, s.float(0.70))
    var pane_var = s.fma(pane_h0, s.float(0.24), s.float(0.88))
    var room_int = s.select_f(is_lit_room, s.fmul(s.fma(suite_kelvin, s.float(1.5), s.float(1.1)), pane_var), s.float(0.05))
    # Kelvin tint: 2700K warm amber -> 4200K neutral -> 6000K cool monitor glow
    var kr = s.fma(s.float(-0.22), suite_kelvin, s.float(1.02))
    var kg = s.fma(s.float(0.14), suite_kelvin, s.float(0.78))
    var kb = s.fma(s.float(0.52), suite_kelvin, s.float(0.48))

    # Exterior glass Fresnel transmittance `1 - F_glass(NdotV)` (`F0 = 0.04`)
    var ndotv_g = s.nclamp(s.fneg(emit_vec3_dot(s, ray_dx, ray_dy, ray_dz, n_warb[0], n_warb[1], n_warb[2])), s.float(0.0), s.float(1.0))
    var omg = s.fsub(s.float(1.0), ndotv_g)
    var omg2 = s.fmul(omg, omg)
    var f_glass = s.fma(s.float(0.96), s.fmul(s.fmul(omg2, omg2), omg), s.float(0.04))
    var trans_g = s.fmul(s.fsub(s.float(1.0), f_glass), s.fmul(frame_mask, blind_vis))

    var tot_scale = s.fmul(room_int, trans_g)
    var tb = s.load_f(troffer_boost)
    var out_ir = s.fmul(s.fma(s.load_f(wall_r), depth_ao, tb), s.fmul(kr, tot_scale))
    var out_ig = s.fmul(s.fma(s.load_f(wall_g), depth_ao, tb), s.fmul(kg, tot_scale))
    var out_ib = s.fmul(s.fma(s.load_f(wall_b), depth_ao, tb), s.fmul(kb, tot_scale))
    return (out_ir, out_ig, out_ib)


def _emit_forzatech_road_wetness_spv(
    mut s: AaaSpirvModule,
    wetness: UInt32,
    rain_int: UInt32,
    time_s: UInt32,
    wx: UInt32,
    wz: UInt32,
    uv_u: UInt32,
    cam_dist: UInt32,
    road_paint_orm: UInt32,
    alb_r: UInt32,
    alb_g: UInt32,
    alb_b: UInt32,
    rough: UInt32,
    metal: UInt32,
    spec_fac_var: UInt32,
    nx_var: UInt32,
    ny_var: UInt32,
    nz_var: UInt32,
) -> UInt32:
    """ForzaTech Physical Wet Asphalt & Weathering:
    - Non-periodic world-space street crown + twin compacted tire ruts + `[0.05..0.85]` subsidence FBM
    - Recalibrated `water_lvl = wetness * 0.56 - 0.14` (`~28%` localized ruts/puddles + `~72%` damp basalt!)
    - Explicit `ORM.b` (`road_paint_orm >= 0`) OR luma thermoplastic lane paint with `0.0` porosity darkening!
    - Bounded Lekner-Dorf porosity darkening (`max(dry * 0.52, 0.38 * dry^1.35)`) + GPU raindrop ripple rings.
    Returns `damp_unsubmerged` `[0..1]` for grazing wet-aggregate retro-backscatter!"""
    var dry_r = s.load_f(alb_r)
    var dry_g = s.load_f(alb_g)
    var dry_b = s.load_f(alb_b)
    var cur_r = s.load_f(rough)
    var dry_luma = s.fma(dry_r, s.float(0.2126), s.fma(dry_g, s.float(0.7152), s.fmul(dry_b, s.float(0.0722))))

    # Explicit `ORM.b` thermoplastic lane paint mask when present (`>= 0`), else bright paint luma (`> 0.38`)
    var luma_paint = s.nclamp(s.fmul(s.fsub(dry_luma, s.float(0.38)), s.float(6.0)), s.float(0.0), s.float(1.0))
    var paint_mask = s.select_f(
        s.fge(road_paint_orm, s.float(0.0)),
        s.nclamp(road_paint_orm, s.float(0.0), s.float(1.0)),
        luma_paint,
    )

    # Wide-dynamic-range world-space FBM subsidence (`4m..22m` basins) + street crown + twin tire ruts
    var n1 = s.fmul(s.fsin(s.fma(wx, s.float(0.043), s.fmul(wz, s.float(0.037)))), s.fcos(s.fma(wx, s.float(-0.031), s.fmul(wz, s.float(0.059)))))
    var n2 = s.fmul(s.fsin(s.fma(wx, s.float(0.127), s.fmul(wz, s.float(-0.109)))), s.fcos(s.fma(wx, s.float(0.089), s.fmul(wz, s.float(0.149)))))
    var n3 = s.fmul(s.fsin(s.fma(wx, s.float(0.370), s.fmul(wz, s.float(0.310)))), s.float(0.35))
    var fbm = s.nclamp(s.fma(n1, s.float(0.34), s.fma(n2, s.float(0.22), s.fma(n3, s.float(0.14), s.float(0.46)))), s.float(0.04), s.float(0.88))

    var u_lane = s.fract(uv_u)
    var crown_d = s.fsub(u_lane, s.float(0.5))
    var crown = s.fma(s.float(-0.95), s.fmul(crown_d, crown_d), s.float(0.22))
    var rut1 = s.fsub(u_lane, s.float(0.28))
    var rut2 = s.fsub(u_lane, s.float(0.72))
    var rut_dip = s.fadd(
        s.exp2(s.fmul(s.fmul(rut1, rut1), s.float(-85.0))),
        s.exp2(s.fmul(s.fmul(rut2, rut2), s.float(-85.0))),
    )
    var h_road = s.nclamp(
        s.fadd(
            s.fma(fbm, s.float(0.72), s.fma(crown, s.float(0.28), s.fmul(rut_dip, s.float(-0.22)))),
            s.fma(paint_mask, s.float(0.09), s.fmul(cur_r, s.float(0.05))),
        ),
        s.float(0.03),
        s.float(0.92),
    )

    var water_lvl = s.fma(wetness, s.float(0.56), s.float(-0.14))
    var delta_h = s.fsub(water_lvl, h_road)

    # 1. Bounded Lekner-Dorf TIR Porosity Darkening (`0.0` on non-porous polymer lane paint!)
    var damp_t = s.nclamp(s.fma(wetness, s.float(1.25), s.fmul(h_road, s.float(-0.20))), s.float(0.0), s.float(1.0))
    var om_paint = s.fsub(s.float(1.0), paint_mask)
    var porous_w = s.fmul(damp_t, om_paint)
    var tir_r = s.nmax(s.fmul(dry_r, s.float(0.45)), s.fmul(s.float(0.35), s.fpow(s.nmax(s.float(1e-4), dry_r), s.float(1.50))))
    var tir_g = s.nmax(s.fmul(dry_g, s.float(0.45)), s.fmul(s.float(0.35), s.fpow(s.nmax(s.float(1e-4), dry_g), s.float(1.50))))
    var tir_b = s.nmax(s.fmul(dry_b, s.float(0.45)), s.fmul(s.float(0.35), s.fpow(s.nmax(s.float(1e-4), dry_b), s.float(1.50))))
    s.store(alb_r, s.fma(s.fsub(tir_r, dry_r), porous_w, dry_r))
    s.store(alb_g, s.fma(s.fsub(tir_g, dry_g), porous_w, dry_g))
    s.store(alb_b, s.fma(s.fsub(tir_b, dry_b), porous_w, dry_b))

    # 2. Continuous Meniscus Roughness (`0.85 dry -> 0.28 damp aggregate -> 0.032 localized puddle`!)
    var r_damp = s.fma(s.fsub(s.nmin(cur_r, s.float(0.30)), cur_r), damp_t, cur_r)
    var film_lin = s.nclamp(s.fmul(s.fadd(delta_h, s.float(0.035)), s.float(8.5)), s.float(0.0), s.float(1.0))
    var film_s = s.fmul(film_lin, s.fmul(film_lin, s.fma(s.float(-2.0), film_lin, s.float(3.0))))
    var new_rough = s.fma(s.fsub(s.float(0.032), r_damp), film_s, r_damp)
    s.store(rough, s.nclamp(new_rough, s.float(0.032), s.float(1.0)))
    s.store(metal, s.fmul(s.load_f(metal), s.fsub(s.float(1.0), film_s)))

    # Water IOR = 1.333 (`F0 = 0.02 -> spec_fac = 0.5`), boosted on retroreflective glass-bead lane paint
    var base_sf = s.load_f(spec_fac_var)
    var retro_sf = s.fma(paint_mask, s.float(0.75), base_sf)
    s.store(spec_fac_var, s.fma(s.fsub(s.float(0.50), retro_sf), film_s, retro_sf))

    # Flatten standing puddle normal smoothly toward world up `(0, 1, 0)` + GPU concentric raindrop ripples!
    var cnx = s.load_f(nx_var); var cny = s.load_f(ny_var); var cnz = s.load_f(nz_var)
    var fnx = s.fvar(s.fmul(cnx, s.fsub(s.float(1.0), film_s)))
    var fny = s.fvar(s.fma(s.fsub(s.float(1.0), cny), film_s, cny))
    var fnz = s.fvar(s.fmul(cnz, s.fsub(s.float(1.0), film_s)))

    s.begin_if(s.logical_and(s.logical_and(s.fgt(rain_int, s.float(0.01)), s.fgt(film_s, s.float(0.15))), s.flt(cam_dist, s.float(45.0))))
    var dist_fade = s.nclamp(s.fmul(s.fsub(s.float(45.0), cam_dist), s.float(1.0 / 25.0)), s.float(0.0), s.float(1.0))
    for layer_i in range(2):
        var sc = s.float(Float32(3.1) if layer_i == 0 else Float32(4.3))
        var off_l = s.float(Float32(layer_i) * Float32(13.7))
        var gx = s.fadd(s.fmul(wx, sc), off_l)
        var gz = s.fadd(s.fmul(wz, sc), off_l)
        var ix = s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(s.fadd(gx, s.float(4096.0)))))
        var iz = s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(s.fadd(gz, s.float(4096.0)))))
        var fx_g = s.fract(gx)
        var fz_g = s.fract(gz)
        var ch0 = _emit_pcg_hash_f01(s, s.bit_xor(s.imul(ix, s.uint(73856093)), s.imul(iz, s.uint(19349663))))
        var t_cyc = s.fadd(s.fmul(time_s, s.float(2.4)), ch0[1])
        var cyc_id = s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(t_cyc)))
        var age = s.fract(t_cyc)
        var ch1 = _emit_pcg_hash_f01(s, s.iadd(ch0[0], s.imul(cyc_id, s.uint(2654435761))))
        var ch2 = _emit_pcg_hash_f01(s, ch1[0])
        var dx_r = s.fsub(fx_g, s.fma(ch1[1], s.float(0.6), s.float(0.2)))
        var dz_r = s.fsub(fz_g, s.fma(ch2[1], s.float(0.6), s.float(0.2)))
        var rad_d = s.fsqrt(s.nmax(s.float(1e-6), s.fma(dx_r, dx_r, s.fmul(dz_r, dz_r))))
        var ring_r = s.fmul(age, s.float(0.42))
        var delta_r = s.fsub(rad_d, ring_r)
        var env_r = s.fmul(
            s.exp2(s.fmul(s.fabs(delta_r), s.float(-26.0))),
            s.fmul(s.fsub(s.float(1.0), age), s.fmul(rain_int, s.fmul(film_s, dist_fade))),
        )
        var wave_s = s.fmul(s.fsin(s.fmul(delta_r, s.float(56.0))), s.fmul(env_r, s.float(0.14)))
        var inv_rd = s.fdiv(wave_s, rad_d)
        s.store(fnx, s.fma(dx_r, inv_rd, s.load_f(fnx)))
        s.store(fnz, s.fma(dz_r, inv_rd, s.load_f(fnz)))
    s.end_if()

    var w_nrm = emit_vec3_normalize(s, s.load_f(fnx), s.load_f(fny), s.load_f(fnz))
    s.store(nx_var, w_nrm[0]); s.store(ny_var, w_nrm[1]); s.store(nz_var, w_nrm[2])
    return s.fmul(damp_t, s.fmul(s.fsub(s.float(1.0), film_s), om_paint))


def surface_cache_update_shader(enable_ray_query: Bool) -> List[UInt32]:
    """Lumen-Class Direct + Multi-Bounce Mesh Surface-Cache Updater (`BIND_AAA_SURFCACHE`).
    Invocation X = instance_idx, Invocation Y = instance_local_tri_idx!
    Push[0] = instance_count, Push[1] = bounce_pass (0 = direct sun + emissive, 1 = +1 indirect bounce!).
    Note: each instance stores `surfcache_base` in `b_surfcache` header `[inst_idx]` (`0..255`),
    and Surface-Cache triangle radiance slots start at word `1024 + (sc_base + tri_idx) * 4`
    (OR `sc_base` already includes the `256`-slot = `1024`-word header offset so `TLAS` `custom_index`
    addresses `b_surfcache[(custom_index + prim_id) * 4]` directly with zero offset addition!!).
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=4,
        group_height=16,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_ray_query=enable_ray_query,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_surfcache = s.buffer(BIND_AAA_SURFCACHE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var inst_idx = s.iadd(s.push_word(2), s.invocation_axis(0))
    var local_tri = s.invocation_axis(1)
    var inst_count = s.push_word(0)
    var bounce_pass = s.push_word(1)
    s.return_if(s.uge(inst_idx, inst_count))

    # Per-instance Surface-Cache metadata header in `b_surfcache[inst_idx * 4 .. +3]`:
    # [0] = sc_slot_base (>= 256), [1] = mesh_first_index, [2] = mesh_tri_count, [3] = material_id
    var hdr_off = s.imul(inst_idx, s.uint(4))
    var sc_base = s.load_word(b_surfcache, hdr_off)
    var first_idx = s.load_word(b_surfcache, s.iadd(hdr_off, s.uint(1)))
    var tri_cnt = s.load_word(b_surfcache, s.iadd(hdr_off, s.uint(2)))
    var mat_id = s.load_word(b_surfcache, s.iadd(hdr_off, s.uint(3)))
    s.return_if(s.uge(local_tri, tri_cnt))

    var idx_off = s.iadd(first_idx, s.imul(local_tri, s.uint(3)))
    var i0 = s.imul(s.load_word(b_indices, idx_off), s.uint(12))
    var i1 = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(1))), s.uint(12))
    var i2 = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(2))), s.uint(12))

    var inst_base = s.imul(inst_idx, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var p0 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, i0), s.load_float(b_verts, s.iadd(i0, s.uint(1))), s.load_float(b_verts, s.iadd(i0, s.uint(2))))
    var p1 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, i1), s.load_float(b_verts, s.iadd(i1, s.uint(1))), s.load_float(b_verts, s.iadd(i1, s.uint(2))))
    var p2 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, i2), s.load_float(b_verts, s.iadd(i2, s.uint(1))), s.load_float(b_verts, s.iadd(i2, s.uint(2))))

    var third = s.float(0.33333334)
    var cx = s.fmul(s.fadd(s.fadd(p0[0], p1[0]), p2[0]), third)
    var cy = s.fmul(s.fadd(s.fadd(p0[1], p1[1]), p2[1]), third)
    var cz = s.fmul(s.fadd(s.fadd(p0[2], p1[2]), p2[2]), third)

    var n0 = emit_mat4_mul_dir(s, b_instances, inst_base, s.load_float(b_verts, s.iadd(i0, s.uint(3))), s.load_float(b_verts, s.iadd(i0, s.uint(4))), s.load_float(b_verts, s.iadd(i0, s.uint(5))))
    var wn = emit_vec3_normalize(s, n0[0], n0[1], n0[2])

    var r_flags_sc = s.load_word(b_globals, s.uint(99))
    var is_furnace_sc = s.ine(s.bit_and(r_flags_sc, s.uint(AAA_FLAG_FURNACE_WHITE_DOME)), s.uint(0))

    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var alb_r_v = s.fvar(s.load_float(b_materials, m_base))
    var alb_g_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(1))))
    var alb_b_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(2))))
    var met = s.load_float(b_materials, s.iadd(m_base, s.uint(5)))
    var em_r_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(8))))
    var em_g_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(9))))
    var em_b_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(10))))
    var tex_pk = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))
    var t_alb_sc = s.bit_and(tex_pk, s.uint(0xFF))

    var cu = s.fmul(s.fadd(s.fadd(s.load_float(b_verts, s.iadd(i0, s.uint(9))), s.load_float(b_verts, s.iadd(i1, s.uint(9)))), s.load_float(b_verts, s.iadd(i2, s.uint(9)))), third)
    var cv = s.fmul(s.fadd(s.fadd(s.load_float(b_verts, s.iadd(i0, s.uint(10))), s.load_float(b_verts, s.iadd(i1, s.uint(10)))), s.load_float(b_verts, s.iadd(i2, s.uint(10)))), third)

    s.begin_if(s.ult(t_alb_sc, s.uint(0xFF)))
    var s_tx = s.sample_bindless_lod(bindless_arr, t_alb_sc, s.vec2(cu, cv), s.float(1.0))
    var tx_r = s.extract_f(s_tx, 0)
    var tx_g = s.extract_f(s_tx, 1)
    var tx_b = s.extract_f(s_tx, 2)
    s.store(alb_r_v, s.fmul(s.load_f(alb_r_v), tx_r))
    s.store(alb_g_v, s.fmul(s.load_f(alb_g_v), tx_g))
    s.store(alb_b_v, s.fmul(s.load_f(alb_b_v), tx_b))
    s.store(em_r_v, s.fmul(s.load_f(em_r_v), tx_r))
    s.store(em_g_v, s.fmul(s.load_f(em_g_v), tx_g))
    s.store(em_b_v, s.fmul(s.load_f(em_b_v), tx_b))
    s.end_if()

    # Decode 24-bit sRGB vertex colors (`vertex_words[11][8..31]`) and apply to both albedo & emissive!
    var sc_m0 = s.load_word(b_verts, s.iadd(i0, s.uint(11)))
    var sc_m1 = s.load_word(b_verts, s.iadd(i1, s.uint(11)))
    var sc_m2 = s.load_word(b_verts, s.iadd(i2, s.uint(11)))
    var svc0 = emit_decode_vertex_color_rgb_spv(s, sc_m0, is_furnace_sc)
    var svc1 = emit_decode_vertex_color_rgb_spv(s, sc_m1, is_furnace_sc)
    var svc2 = emit_decode_vertex_color_rgb_spv(s, sc_m2, is_furnace_sc)
    var svcr = s.fmul(s.fadd(s.fadd(svc0[0], svc1[0]), svc2[0]), third)
    var svcg = s.fmul(s.fadd(s.fadd(svc0[1], svc1[1]), svc2[1]), third)
    var svcb = s.fmul(s.fadd(s.fadd(svc0[2], svc1[2]), svc2[2]), third)
    s.store(alb_r_v, s.fmul(s.load_f(alb_r_v), svcr))
    s.store(alb_g_v, s.fmul(s.load_f(alb_g_v), svcg))
    s.store(alb_b_v, s.fmul(s.load_f(alb_b_v), svcb))
    s.store(em_r_v, s.fmul(s.load_f(em_r_v), svcr))
    s.store(em_g_v, s.fmul(s.load_f(em_g_v), svcg))
    s.store(em_b_v, s.fmul(s.load_f(em_b_v), svcb))

    # 4-Bay Interior Window Suite Glow in `LumenSurfaceCache` so skyscraper windows cast GI onto streets/balconies!
    var sc_int_par = s.ine(s.bit_and(tex_pk, s.uint(AAA_MAT_INTERIOR_PARALLAX)), s.uint(0))
    s.begin_if(s.logical_and(s.logical_not(is_furnace_sc), s.logical_and(sc_int_par, s.flt(met, s.float(0.45)))))
    var sc_suite = emit_window_suite_prng_spv(s, inst_idx, cu, cv)
    var sc_lit = s.flt(sc_suite[0], s.float(0.70))
    var sc_wint = s.select_f(sc_lit, s.fma(sc_suite[1], s.float(0.55), s.float(0.42)), s.float(0.02))
    var sc_kr = s.fma(s.float(-0.22), sc_suite[1], s.float(1.02))
    var sc_kg = s.fma(s.float(0.14), sc_suite[1], s.float(0.78))
    var sc_kb = s.fma(s.float(0.52), sc_suite[1], s.float(0.48))
    s.store(em_r_v, s.fma(sc_kr, sc_wint, s.load_f(em_r_v)))
    s.store(em_g_v, s.fma(sc_kg, sc_wint, s.load_f(em_g_v)))
    s.store(em_b_v, s.fma(sc_kb, sc_wint, s.load_f(em_b_v)))
    s.end_if()

    var alb_r = s.load_f(alb_r_v)
    var alb_g = s.load_f(alb_g_v)
    var alb_b = s.load_f(alb_b_v)
    var em_r = s.load_f(em_r_v)
    var em_g = s.load_f(em_g_v)
    var em_b = s.load_f(em_b_v)

    var diff_Scale = s.fsub(s.float(1.0), met)
    var kd_r = s.fmul(alb_r, diff_Scale)
    var kd_g = s.fmul(alb_g, diff_Scale)
    var kd_b = s.fmul(alb_b, diff_Scale)

    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))

    var ndotl = s.nmax(s.float(0.0), emit_vec3_dot(s, wn[0], wn[1], wn[2], sun_x, sun_y, sun_z))
    var sh_vis = s.fvar(s.float(1.0))
    var sky_occ = s.fvar(s.float(1.0))
    if enable_ray_query:
        s.begin_if(s.fgt(ndotl, s.float(0.001)))
        var rt_s = emit_rt_shadow_query(
            s, tlas_var, cx, cy, cz, wn[0], wn[1], wn[2], sun_x, sun_y, sun_z, s.float(500.0),
            b_verts=b_verts, b_indices=b_indices, b_instances=b_instances, b_materials=b_materials, bindless_arr=bindless_arr,
        )
        s.store(sh_vis, rt_s)
        s.end_if()

        # Upward sky-occlusion probe prevents open-sky SH leaking into tunnels, interiors & ride soffits!
        s.begin_if(s.logical_not(is_furnace_sc))
        var sky_up = emit_vec3_normalize(
            s,
            s.fmul(wn[0], s.float(0.35)),
            s.nmax(s.float(0.25), s.fma(wn[1], s.float(0.35), s.float(0.85))),
            s.fmul(wn[2], s.float(0.35)),
        )
        var up_vis = emit_rt_shadow_query(
            s, tlas_var, cx, cy, cz, wn[0], wn[1], wn[2], sky_up[0], sky_up[1], sky_up[2], s.float(65.0),
            b_verts=b_verts, b_indices=b_indices, b_instances=b_instances, b_materials=b_materials, bindless_arr=bindless_arr,
        )
        s.store(sky_occ, s.fma(up_vis, s.float(0.84), s.float(0.16)))
        s.end_if()

    var inv_pi = s.float(0.318309886)
    var d_scale = s.fmul(s.fmul(ndotl, s.load_f(sh_vis)), inv_pi)
    var sky_dir_sh = emit_eval_sh9_irradiance(s, b_globals, 72, wn[0], wn[1], wn[2])
    var ibl_dir_int = s.fmul(s.load_float(b_globals, s.uint(55)), s.load_f(sky_occ))
    var night_bounce_r = s.fmul(sky_dir_sh[0], ibl_dir_int)
    var night_bounce_g = s.fmul(sky_dir_sh[1], ibl_dir_int)
    var night_bounce_b = s.fmul(sky_dir_sh[2], ibl_dir_int)
    var out_r = s.fvar(s.fma(kd_r, s.fadd(s.fmul(sun_r, d_scale), night_bounce_r), em_r))
    var out_g = s.fvar(s.fma(kd_g, s.fadd(s.fmul(sun_g, d_scale), night_bounce_g), em_g))
    var out_b = s.fvar(s.fma(kd_b, s.fadd(s.fmul(sun_b, d_scale), night_bounce_b), em_b))

    # Direct Clustered Local & `LTC` Area Light Injection into `LumenSurfaceCache`!
    s.begin_if(s.ine(s.bit_and(r_flags_sc, s.uint(AAA_FLAG_CLUSTERED_LIGHTS)), s.uint(0)))
    var loc_sc = emit_froxel_local_lighting_at_hit(s, b_clusters, b_globals, cx, cy, cz, wn[0], wn[1], wn[2])
    s.store(out_r, s.fma(kd_r, loc_sc[0], s.load_f(out_r)))
    s.store(out_g, s.fma(kd_g, loc_sc[1], s.load_f(out_g)))
    s.store(out_b, s.fma(kd_b, loc_sc[2], s.load_f(out_b)))
    s.end_if()

    if enable_ray_query:
        # Bounce 1+: gather multi-bounce indirect diffuse radiance from other Surface-Cache triangles!
        s.begin_if(s.ine(bounce_pass, s.uint(0)))
        var sky_sh = emit_eval_sh9_irradiance(s, b_globals, 72, wn[0], wn[1], wn[2])
        var ibl_int = s.load_float(b_globals, s.uint(55))
        var miss_r = s.fmul(sky_sh[0], ibl_int)
        var miss_g = s.fmul(sky_sh[1], ibl_int)
        var miss_b = s.fmul(sky_sh[2], ibl_int)

        var gi_r = s.fvar(s.float(0.0))
        var gi_g = s.fvar(s.float(0.0))
        var gi_b = s.fvar(s.float(0.0))
        var seed0 = s.iadd(s.imul(inst_idx, s.uint(9781)), s.imul(local_tri, s.uint(6271)))
        var ro_x = s.fma(wn[0], s.float(0.008), cx)
        var ro_y = s.fma(wn[1], s.float(0.008), cy)
        var ro_z = s.fma(wn[2], s.float(0.008), cz)

        for ray_k in range(16):
            var h1 = _emit_pcg_hash_f01(s, s.iadd(seed0, s.uint(UInt32(ray_k * 19 + 3))))
            var h2 = _emit_pcg_hash_f01(s, h1[0])
            var sx = s.float(Float32(ray_k & 3))
            var sy = s.float(Float32(ray_k >> 2))
            var u1_s = s.fmul(s.fadd(sx, h1[1]), s.float(0.25))
            var u2_s = s.fmul(s.fadd(sy, h2[1]), s.float(0.25))
            var rdir = _emit_cosine_hemisphere_dir(s, wn[0], wn[1], wn[2], u1_s, u2_s)
            var hit_rad = emit_rt_surface_cache_ray(
                s, tlas_var, b_surfcache, ro_x, ro_y, ro_z, rdir[0], rdir[1], rdir[2],
                s.float(0.005), s.float(200.0), miss_r, miss_g, miss_b,
                b_instances=b_instances,
                b_materials=b_materials,
                b_verts=b_verts,
                b_indices=b_indices,
                bindless_arr=bindless_arr,
            )
            s.store(gi_r, s.fadd(s.load_f(gi_r), hit_rad[0]))
            s.store(gi_g, s.fadd(s.load_f(gi_g), hit_rad[1]))
            s.store(gi_b, s.fadd(s.load_f(gi_b), hit_rad[2]))

        var inv16 = s.float(0.0625)
        s.store(out_r, s.fma(kd_r, s.fmul(s.load_f(gi_r), inv16), s.load_f(out_r)))
        s.store(out_g, s.fma(kd_g, s.fmul(s.load_f(gi_g), inv16), s.load_f(out_g)))
        s.store(out_b, s.fma(kd_b, s.fmul(s.load_f(gi_b), inv16), s.load_f(out_b)))
        s.end_if()

    var sc_word = s.imul(s.iadd(sc_base, local_tri), s.uint(4))
    s.store_float(b_surfcache, sc_word, s.load_f(out_r))
    s.store_float(b_surfcache, s.iadd(sc_word, s.uint(1)), s.load_f(out_g))
    s.store_float(b_surfcache, s.iadd(sc_word, s.uint(2)), s.load_f(out_b))
    s.store_float(b_surfcache, s.iadd(sc_word, s.uint(3)), s.float(1.0))

    return s.finish()


def aaa_deferred_shade_shader(enable_ray_query: Bool) -> List[UInt32]:
    """Unified AAA Deferred `VisBuffer` Compute Material & Lighting Shader:
    - Analytic 3-Ray Möller-Trumbore perspective-correct barycentrics + `ddx/ddy` UV gradients
    - Bindless `16x` anisotropic `BC7` Albedo/ORM + `BC5` Macro & Micro-Detail `RNM` Normals
    - Multi-Scatter Energy-Conserving GGX + Clearcoat + Thin-Film Iridescence + Sheen + SSS
    - `16k` Virtual Shadow Maps (`PCSS`) + `SPV_KHR_ray_query` Contact Shadows, Reflections &
      12-Ray Cosine-Hemisphere Lumen Surface-Cache Global Illumination!
    Push[0] = render_w, Push[1] = render_h, Push[2] = gi_ray_count (e.g. 8 or 16).
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_ray_query=enable_ray_query,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var b_surfcache = s.buffer(BIND_AAA_SURFCACHE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var b_skinning = s.buffer(BIND_AAA_SKINNING)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var b_hist = s.buffer(BIND_AAA_HISTORY)

    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var _depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var brdf_lut = s.sampled_image_2d(BIND_AAA_BRDF_LUT)
    var ltc_lut = s.sampled_image_2d(BIND_AAA_LTC_LUT)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    # --- sota:camera hook (compute VRS: pixel from the compacted shade list when push[7] != 0) ---
    var sota_vrs_xy = emit_sota_vrs_remap(s, b_post, px, py, s.push_word(7), s.push_word(8))
    px = sota_vrs_xy[0]
    py = sota_vrs_xy[1]
    # --- end sota:camera hook ---
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var gi_rays_u = s.push_word(2)
    var demod_gi = s.ine(s.push_word(3), s.uint(0))
    var demod_spec = s.ine(s.push_word(4), s.uint(0))
    var spec_in_off = s.push_word(5)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var r_flags = s.load_word(b_globals, s.uint(99))
    var frame_idx = s.load_word(b_globals, s.uint(100))
    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))
    var max_env_mip = s.load_float(b_globals, s.uint(103))

    # Build camera ray `D(px, py)` (accounting for sub-pixel TSR jitter!) + `+dx` and `+dy` rays
    var fx0 = s.fsub(s.fadd(s.float_from_uint(px), s.float(0.5)), s.fmul(jit_x, s.fmul(fw, s.float(0.5))))
    var fy0 = s.fadd(s.fadd(s.float_from_uint(py), s.float(0.5)), s.fmul(jit_y, s.fmul(fh, s.float(0.5))))
    var ndc_x = s.fsub(s.fmul(s.fdiv(fx0, fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(fy0, fh), s.float(2.0)))
    var step_nx = s.fdiv(s.float(2.0), fw)
    var step_ny = s.fneg(s.fdiv(s.float(2.0), fh))

    var ray_c = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var ray_dx = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(s.fadd(ndc_x, step_nx), tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(s.fadd(ndc_x, step_nx), tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(s.fadd(ndc_x, step_nx), tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var ray_dy = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(s.fadd(ndc_y, step_ny), tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(s.fadd(ndc_y, step_ny), tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(s.fadd(ndc_y, step_ny), tan_y), fwd_z)),
    )

    var vis_uv4 = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var vis0 = s.extract_u(vis_uv4, 0)
    var vis1 = s.extract_u(vis_uv4, 1)
    var pix_idx_u = s.iadd(s.imul(py, w_u), px)
    var out_pix_base = s.imul(pix_idx_u, s.uint(8))
    var post_pix_base = s.iadd(s.uint(1024), out_pix_base)
    var spec_pix_base = s.iadd(spec_in_off, s.imul(pix_idx_u, s.uint(4)))
    var is_furnace = s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_FURNACE_WHITE_DOME)), s.uint(0))

    # Sky background pixel (`vis0 == 0`)
    s.begin_if(s.ieq(vis0, s.uint(0)))
    var env_bg = s.sample_cube_lod(env_cube, s.vec3(ray_c[0], ray_c[1], ray_c[2]), s.float(0.0))
    var bg_r_v = s.fvar(s.select_f(is_furnace, s.float(1.0), s.extract_f(env_bg, 0)))
    var bg_g_v = s.fvar(s.select_f(is_furnace, s.float(1.0), s.extract_f(env_bg, 1)))
    var bg_b_v = s.fvar(s.select_f(is_furnace, s.float(1.0), s.extract_f(env_bg, 2)))
    s.begin_if(s.logical_not(is_furnace))
    var nubis_bg = emit_nubis_sky_clouds_and_celestial_spv(
        s, b_globals, env_cube,
        ray_c[0], ray_c[1], ray_c[2], fh,
        s.load_f(bg_r_v), s.load_f(bg_g_v), s.load_f(bg_b_v),
    )
    s.store(bg_r_v, nubis_bg[0])
    s.store(bg_g_v, nubis_bg[1])
    s.store(bg_b_v, nubis_bg[2])
    s.end_if()
    var bg_r = s.load_f(bg_r_v)
    var bg_g = s.load_f(bg_g_v)
    var bg_b = s.load_f(bg_b_v)
    s.begin_if(demod_gi)
    s.store_float(b_post, post_pix_base, bg_r)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(1)), bg_g)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(2)), bg_b)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(3)), s.float(0.0))
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(4)), s.float(0.0))
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(5)), s.float(0.0))
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(6)), s.float(0.0))
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(7)), s.float(0.0))
    s.end_if()
    s.begin_if(demod_spec)
    var zero_f16 = s.pack_f16x2(s.float(0.0), s.float(0.0))
    var sky_r_f16 = s.pack_f16x2(s.float(0.0), s.float(1.0))
    var sky_t_f16 = s.pack_f16x2(s.float(0.0), s.float(32768.0))
    s.store_word(b_hist, spec_pix_base, zero_f16)
    s.store_word(b_hist, s.iadd(spec_pix_base, s.uint(1)), sky_r_f16)
    s.store_word(b_hist, s.iadd(spec_pix_base, s.uint(2)), zero_f16)
    s.store_word(b_hist, s.iadd(spec_pix_base, s.uint(3)), sky_t_f16)
    s.end_if()
    s.store_float(b_hdr, out_pix_base, s.select_f(demod_gi, s.float(0.0), bg_r))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(1)), s.select_f(demod_gi, s.float(0.0), bg_g))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(2)), s.select_f(demod_gi, s.float(0.0), bg_b))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(3)), s.float(1e6))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(4)), s.float(0.0))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(5)), s.float(0.0))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(6)), s.float(0.0))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(7)), s.float(0.0))
    s.begin_else()

    # Decode 64-bit VisBuffer payload
    var inst_id = s.shr(vis0, s.uint(17))
    var ml_id = s.isub(s.bit_and(vis0, s.uint(0x1FFFF)), s.uint(1))
    var mat_id = s.shr(vis1, s.uint(16))
    var local_prim = s.bit_and(vis1, s.uint(0xFFFF))

    var ml_base = s.imul(ml_id, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var global_tri = s.iadd(first_tri, local_prim)
    var idx_off = s.imul(global_tri, s.uint(3))
    var vi0 = s.load_word(b_indices, idx_off)
    var vi1 = s.load_word(b_indices, s.iadd(idx_off, s.uint(1)))
    var vi2 = s.load_word(b_indices, s.iadd(idx_off, s.uint(2)))
    var v0_off = s.imul(vi0, s.uint(12))
    var v1_off = s.imul(vi1, s.uint(12))
    var v2_off = s.imul(vi2, s.uint(12))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var l0x = s.load_float(b_verts, v0_off); var l0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(1))); var l0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(2)))
    var l1x = s.load_float(b_verts, v1_off); var l1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(1))); var l1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(2)))
    var l2x = s.load_float(b_verts, v2_off); var l2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(1))); var l2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(2)))

    var p0 = emit_instance_point(s, b_instances, b_globals, b_verts, inst_base, v0_off, l0x, l0y, l0z)
    var p1 = emit_instance_point(s, b_instances, b_globals, b_verts, inst_base, v1_off, l1x, l1y, l1z)
    var p2 = emit_instance_point(s, b_instances, b_globals, b_verts, inst_base, v2_off, l2x, l2y, l2z)

    # Analytic 3-Ray Möller-Trumbore Differential Barycentrics
    var bary = emit_ray_barycentrics(
        s,
        cam_x, cam_y, cam_z,
        ray_c[0], ray_c[1], ray_c[2],
        ray_dx[0], ray_dx[1], ray_dx[2],
        ray_dy[0], ray_dy[1], ray_dy[2],
        p0[0], p0[1], p0[2],
        p1[0], p1[1], p1[2],
        p2[0], p2[1], p2[2],
    )
    var b1 = bary[0]
    var b2 = bary[1]
    var b0 = s.fsub(s.fsub(s.float(1.0), b1), b2)

    # Perspective-correct world position & local position (including skeletal previous-frame deformed vertices!)
    var wx = s.fma(p0[0], b0, s.fma(p1[0], b1, s.fmul(p2[0], b2)))
    var wy = s.fma(p0[1], b0, s.fma(p1[1], b1, s.fmul(p2[1], b2)))
    var wz = s.fma(p0[2], b0, s.fma(p1[2], b1, s.fmul(p2[2], b2)))
    var loc_x = s.fvar(s.fma(l0x, b0, s.fma(l1x, b1, s.fmul(l2x, b2))))
    var loc_y = s.fvar(s.fma(l0y, b0, s.fma(l1y, b1, s.fmul(l2y, b2))))
    var loc_z = s.fvar(s.fma(l0z, b0, s.fma(l1z, b1, s.fmul(l2z, b2))))

    var inst_flags_u = s.bit_and(s.shr(s.load_word(b_instances, s.iadd(inst_base, s.uint(34))), s.uint(16)), s.uint(1))
    s.begin_if(s.ieq(inst_flags_u, s.uint(1)))
    var skin_first_slot = s.load_word(b_instances, s.iadd(inst_base, s.uint(36)))
    var mesh_first_vtx = s.load_word(b_instances, s.iadd(inst_base, s.uint(37)))
    var lvi0 = s.iadd(skin_first_slot, s.isub(vi0, mesh_first_vtx))
    var lvi1 = s.iadd(skin_first_slot, s.isub(vi1, mesh_first_vtx))
    var lvi2 = s.iadd(skin_first_slot, s.isub(vi2, mesh_first_vtx))
    var sp0 = s.iadd(s.uint(UInt32(AAA_SKIN_PREV_POS_BASE)), s.imul(lvi0, s.uint(3)))
    var sp1 = s.iadd(s.uint(UInt32(AAA_SKIN_PREV_POS_BASE)), s.imul(lvi1, s.uint(3)))
    var sp2 = s.iadd(s.uint(UInt32(AAA_SKIN_PREV_POS_BASE)), s.imul(lvi2, s.uint(3)))
    s.store(loc_x, s.fma(s.load_float(b_skinning, sp0), b0, s.fma(s.load_float(b_skinning, sp1), b1, s.fmul(s.load_float(b_skinning, sp2), b2))))
    s.store(loc_y, s.fma(s.load_float(b_skinning, s.iadd(sp0, s.uint(1))), b0, s.fma(s.load_float(b_skinning, s.iadd(sp1, s.uint(1))), b1, s.fmul(s.load_float(b_skinning, s.iadd(sp2, s.uint(1))), b2))))
    s.store(loc_z, s.fma(s.load_float(b_skinning, s.iadd(sp0, s.uint(2))), b0, s.fma(s.load_float(b_skinning, s.iadd(sp1, s.uint(2))), b1, s.fmul(s.load_float(b_skinning, s.iadd(sp2, s.uint(2))), b2))))
    s.end_if()

    # Compute exact 3D per-pixel motion vector `(vel_u, vel_v)` from unjittered `(fx0/fw, fy0/fh)` & `prev_view_proj`!
    var prev_wp = emit_mat4_mul_point(s, b_instances, s.iadd(inst_base, s.uint(16)), s.load_f(loc_x), s.load_f(loc_y), s.load_f(loc_z))
    var prev_clip = emit_mat4_mul_point(s, b_globals, s.uint(16), prev_wp[0], prev_wp[1], prev_wp[2])
    var prev_inv_w = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), prev_clip[3]))
    var prev_u = s.fma(s.fmul(prev_clip[0], prev_inv_w), s.float(0.5), s.float(0.5))
    var prev_v = s.fma(s.fneg(s.fmul(prev_clip[1], prev_inv_w)), s.float(0.5), s.float(0.5))
    var cur_u = s.fdiv(fx0, fw)
    var cur_v = s.fdiv(fy0, fh)
    var vel_u = s.fsub(cur_u, prev_u)
    var vel_v = s.fsub(cur_v, prev_v)

    # Interpolate smooth vertex normal + tangent + UV + analytic `(ddx_uv, ddy_uv)`!
    # Load upper 3x3 columns `c0, c1, c2` of `model_to_world` and compute Cofactor Normal Matrix `cof = [c1xc2, c2xc0, c0xc1]`
    var c0x = s.load_float(b_instances, inst_base)
    var c0y = s.load_float(b_instances, s.iadd(inst_base, s.uint(1)))
    var c0z = s.load_float(b_instances, s.iadd(inst_base, s.uint(2)))
    var c1x = s.load_float(b_instances, s.iadd(inst_base, s.uint(4)))
    var c1y = s.load_float(b_instances, s.iadd(inst_base, s.uint(5)))
    var c1z = s.load_float(b_instances, s.iadd(inst_base, s.uint(6)))
    var c2x = s.load_float(b_instances, s.iadd(inst_base, s.uint(8)))
    var c2y = s.load_float(b_instances, s.iadd(inst_base, s.uint(9)))
    var c2z = s.load_float(b_instances, s.iadd(inst_base, s.uint(10)))
    var cof0 = emit_vec3_cross(s, c1x, c1y, c1z, c2x, c2y, c2z)
    var cof1 = emit_vec3_cross(s, c2x, c2y, c2z, c0x, c0y, c0z)
    var cof2 = emit_vec3_cross(s, c0x, c0y, c0z, c1x, c1y, c1z)
    var det_m = emit_vec3_dot(s, c0x, c0y, c0z, cof0[0], cof0[1], cof0[2])
    var det_sign = s.select_f(s.flt(det_m, s.float(0.0)), s.float(-1.0), s.float(1.0))

    var n0x = s.load_float(b_verts, s.iadd(v0_off, s.uint(3))); var n0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(4))); var n0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(5)))
    var n1x = s.load_float(b_verts, s.iadd(v1_off, s.uint(3))); var n1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(4))); var n1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(5)))
    var n2x = s.load_float(b_verts, s.iadd(v2_off, s.uint(3))); var n2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(4))); var n2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(5)))
    var ln_x = s.fma(n0x, b0, s.fma(n1x, b1, s.fmul(n2x, b2)))
    var ln_y = s.fma(n0y, b0, s.fma(n1y, b1, s.fmul(n2y, b2)))
    var ln_z = s.fma(n0z, b0, s.fma(n1z, b1, s.fmul(n2z, b2)))
    var wn_cx = s.fmul(s.fma(cof0[0], ln_x, s.fma(cof1[0], ln_y, s.fmul(cof2[0], ln_z))), det_sign)
    var wn_cy = s.fmul(s.fma(cof0[1], ln_x, s.fma(cof1[1], ln_y, s.fmul(cof2[1], ln_z))), det_sign)
    var wn_cz = s.fmul(s.fma(cof0[2], ln_x, s.fma(cof1[2], ln_y, s.fmul(cof2[2], ln_z))), det_sign)
    var wn_geom = emit_vec3_normalize(s, wn_cx, wn_cy, wn_cz)

    var t0x = s.load_float(b_verts, s.iadd(v0_off, s.uint(6))); var t0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(7))); var t0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(8)))
    var t1x = s.load_float(b_verts, s.iadd(v1_off, s.uint(6))); var t1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(7))); var t1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(8)))
    var t2x = s.load_float(b_verts, s.iadd(v2_off, s.uint(6))); var t2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(7))); var t2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(8)))
    var lt_x = s.fma(t0x, b0, s.fma(t1x, b1, s.fmul(t2x, b2)))
    var lt_y = s.fma(t0y, b0, s.fma(t1y, b1, s.fmul(t2y, b2)))
    var lt_z = s.fma(t0z, b0, s.fma(t1z, b1, s.fmul(t2z, b2)))
    var wt_rx = s.fma(c0x, lt_x, s.fma(c1x, lt_y, s.fmul(c2x, lt_z)))
    var wt_ry = s.fma(c0y, lt_x, s.fma(c1y, lt_y, s.fmul(c2y, lt_z)))
    var wt_rz = s.fma(c0z, lt_x, s.fma(c1z, lt_y, s.fmul(c2z, lt_z)))
    # Gram-Schmidt re-orthogonalization `t_hat = normalize(t - (t . n_hat) * n_hat)` with perpendicular fallback
    var td_n = emit_vec3_dot(s, wt_rx, wt_ry, wt_rz, wn_geom[0], wn_geom[1], wn_geom[2])
    var gs_tx = s.fsub(wt_rx, s.fmul(td_n, wn_geom[0]))
    var gs_ty = s.fsub(wt_ry, s.fmul(td_n, wn_geom[1]))
    var gs_tz = s.fsub(wt_rz, s.fmul(td_n, wn_geom[2]))
    var gs_l2 = emit_vec3_dot(s, gs_tx, gs_ty, gs_tz, gs_tx, gs_ty, gs_tz)
    var fb_up_y = s.flt(s.fabs(wn_geom[1]), s.float(0.92))
    var fb_hx = s.select_f(fb_up_y, s.float(0.0), s.float(1.0))
    var fb_hy = s.select_f(fb_up_y, s.float(1.0), s.float(0.0))
    var fb_t = emit_vec3_cross(s, fb_hx, fb_hy, s.float(0.0), wn_geom[0], wn_geom[1], wn_geom[2])
    var gs_ok = s.fgt(gs_l2, s.float(1e-8))
    var wt_u = emit_vec3_normalize(
        s,
        s.select_f(gs_ok, gs_tx, fb_t[0]),
        s.select_f(gs_ok, gs_ty, fb_t[1]),
        s.select_f(gs_ok, gs_tz, fb_t[2]),
    )
    var meta0 = s.load_word(b_verts, s.iadd(v0_off, s.uint(11)))
    var meta1 = s.load_word(b_verts, s.iadd(v1_off, s.uint(11)))
    var meta2 = s.load_word(b_verts, s.iadd(v2_off, s.uint(11)))
    var t_sign = s.fmul(s.select_f(s.ieq(s.bit_and(meta0, s.uint(1)), s.uint(0)), s.float(1.0), s.float(-1.0)), det_sign)
    var wb_raw = emit_vec3_cross(s, wn_geom[0], wn_geom[1], wn_geom[2], wt_u[0], wt_u[1], wt_u[2])
    var wb_u = (s.fmul(wb_raw[0], t_sign), s.fmul(wb_raw[1], t_sign), s.fmul(wb_raw[2], t_sign))

    var u0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(9))); var v0_t = s.load_float(b_verts, s.iadd(v0_off, s.uint(10)))
    var u1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(9))); var v1_t = s.load_float(b_verts, s.iadd(v1_off, s.uint(10)))
    var u2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(9))); var v2_t = s.load_float(b_verts, s.iadd(v2_off, s.uint(10)))
    var uv_u = s.fma(u0, b0, s.fma(u1, b1, s.fmul(u2, b2)))
    var uv_v = s.fma(v0_t, b0, s.fma(v1_t, b1, s.fmul(v2_t, b2)))

    var b0_dx = s.fsub(s.fsub(s.float(1.0), bary[2]), bary[3])
    var b0_dy = s.fsub(s.fsub(s.float(1.0), bary[4]), bary[5])

    # Neighbor ray geometric normals `(wn_dx, wn_dy)` via cofactor matrix for Kaplanyan Specular AA
    var lndx_x = s.fma(n0x, b0_dx, s.fma(n1x, bary[2], s.fmul(n2x, bary[3])))
    var lndx_y = s.fma(n0y, b0_dx, s.fma(n1y, bary[2], s.fmul(n2y, bary[3])))
    var lndx_z = s.fma(n0z, b0_dx, s.fma(n1z, bary[2], s.fmul(n2z, bary[3])))
    var wndx_cx = s.fmul(s.fma(cof0[0], lndx_x, s.fma(cof1[0], lndx_y, s.fmul(cof2[0], lndx_z))), det_sign)
    var wndx_cy = s.fmul(s.fma(cof0[1], lndx_x, s.fma(cof1[1], lndx_y, s.fmul(cof2[1], lndx_z))), det_sign)
    var wndx_cz = s.fmul(s.fma(cof0[2], lndx_x, s.fma(cof1[2], lndx_y, s.fmul(cof2[2], lndx_z))), det_sign)
    var wn_dx = emit_vec3_normalize(s, wndx_cx, wndx_cy, wndx_cz)

    var lndy_x = s.fma(n0x, b0_dy, s.fma(n1x, bary[4], s.fmul(n2x, bary[5])))
    var lndy_y = s.fma(n0y, b0_dy, s.fma(n1y, bary[4], s.fmul(n2y, bary[5])))
    var lndy_z = s.fma(n0z, b0_dy, s.fma(n1z, bary[4], s.fmul(n2z, bary[5])))
    var wndy_cx = s.fmul(s.fma(cof0[0], lndy_x, s.fma(cof1[0], lndy_y, s.fmul(cof2[0], lndy_z))), det_sign)
    var wndy_cy = s.fmul(s.fma(cof0[1], lndy_x, s.fma(cof1[1], lndy_y, s.fmul(cof2[1], lndy_z))), det_sign)
    var wndy_cz = s.fmul(s.fma(cof0[2], lndy_x, s.fma(cof1[2], lndy_y, s.fmul(cof2[2], lndy_z))), det_sign)
    var wn_dy = emit_vec3_normalize(s, wndy_cx, wndy_cy, wndy_cz)

    # Apply `-0.42` sub-pixel Halton jitter UV-gradient mip bias (`b_globals[127]`)
    var mip_bias_g = s.load_float(b_globals, s.uint(127))
    var grad_bias_scale = s.exp2(mip_bias_g)
    var ddx_u = s.fmul(s.fsub(s.fma(u0, b0_dx, s.fma(u1, bary[2], s.fmul(u2, bary[3]))), uv_u), grad_bias_scale)
    var ddx_v = s.fmul(s.fsub(s.fma(v0_t, b0_dx, s.fma(v1_t, bary[2], s.fmul(v2_t, bary[3]))), uv_v), grad_bias_scale)
    var ddy_u = s.fmul(s.fsub(s.fma(u0, b0_dy, s.fma(u1, bary[4], s.fmul(u2, bary[5]))), uv_u), grad_bias_scale)
    var ddy_v = s.fmul(s.fsub(s.fma(v0_t, b0_dy, s.fma(v1_t, bary[4], s.fmul(v2_t, bary[5]))), uv_v), grad_bias_scale)

    var rel_cam_x = s.fsub(wx, cam_x)
    var rel_cam_y = s.fsub(wy, cam_y)
    var rel_cam_z = s.fsub(wz, cam_z)
    var z_lin = s.nmax(s.float(0.01), emit_vec3_dot(s, rel_cam_x, rel_cam_y, rel_cam_z, fwd_x, fwd_y, fwd_z))
    var cam_dist = s.fsqrt(s.nmax(s.float(1e-6), emit_vec3_dot(s, rel_cam_x, rel_cam_y, rel_cam_z, rel_cam_x, rel_cam_y, rel_cam_z)))

    # Load Enterprise BSDF Material + Bindless `BC7` / `BC5` Hardware Anisotropic Textures!
    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var alb_r = s.fvar(s.load_float(b_materials, m_base))
    var alb_g = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(1))))
    var alb_b = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(2))))
    var spec_fac_v = s.fvar(s.fabs(s.load_float(b_materials, s.iadd(m_base, s.uint(3)))))
    var rough = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(4))))
    var metal = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(5))))
    var ccoat = s.load_float(b_materials, s.iadd(m_base, s.uint(6)))
    var ccoat_r_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(7))))
    var em_r_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(8))))
    var em_g_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(9))))
    var em_b_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(10))))
    var raw_det_scale = s.load_float(b_materials, s.iadd(m_base, s.uint(11)))
    var is_wet_road_mat = s.flt(raw_det_scale, s.float(0.0))
    var det_scale = s.fabs(raw_det_scale)
    var det_str = s.load_float(b_materials, s.iadd(m_base, s.uint(12)))
    var irid_w = s.load_float(b_materials, s.iadd(m_base, s.uint(13)))
    var irid_nm = s.load_float(b_materials, s.iadd(m_base, s.uint(14)))
    var irid_ior = s.load_float(b_materials, s.iadd(m_base, s.uint(15)))
    var sh_r = s.load_float(b_materials, s.iadd(m_base, s.uint(16)))
    var sh_g = s.load_float(b_materials, s.iadd(m_base, s.uint(17)))
    var sh_b = s.load_float(b_materials, s.iadd(m_base, s.uint(18)))
    var sh_rough = s.load_float(b_materials, s.iadd(m_base, s.uint(19)))
    var sss_r = s.load_float(b_materials, s.iadd(m_base, s.uint(20)))
    var sss_g = s.load_float(b_materials, s.iadd(m_base, s.uint(21)))
    var sss_b = s.load_float(b_materials, s.iadd(m_base, s.uint(22)))
    var tex_pack = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))

    var t_alb = s.bit_and(tex_pack, s.uint(0xFF))
    var t_nrm = s.bit_and(s.shr(tex_pack, s.uint(8)), s.uint(0xFF))
    var t_orm = s.bit_and(s.shr(tex_pack, s.uint(16)), s.uint(0xFF))
    var t_det = s.bit_and(s.shr(tex_pack, s.uint(24)), s.uint(AAA_MAT_DET_NONE))
    var has_int_parallax = s.ine(s.bit_and(tex_pack, s.uint(AAA_MAT_INTERIOR_PARALLAX)), s.uint(0))

    # Interpolate per-vertex AO supporting both legacy and new `vertex_words[11]` layouts
    var vao0 = _emit_decode_vertex_ao_spv(s, meta0)
    var vao1 = _emit_decode_vertex_ao_spv(s, meta1)
    var vao2 = _emit_decode_vertex_ao_spv(s, meta2)
    var vtx_ao = s.select_f(is_furnace, s.float(1.0), s.fma(vao0, b0, s.fma(vao1, b1, s.fmul(vao2, b2))))
    var mat_ao = s.fvar(vtx_ao)

    var uv_vec = s.vec2(uv_u, uv_v)
    var ddx_vec = s.vec2(ddx_u, ddx_v)
    var ddy_vec = s.vec2(ddy_u, ddy_v)

    var tn_x = s.fvar(s.float(0.0))
    var tn_y = s.fvar(s.float(0.0))
    var tn_z = s.fvar(s.float(1.0))
    var road_paint_orm_v = s.fvar(s.float(-1.0))

    # 4-Layer Frostbite Height-Modulated Terrain Splatting (`(meta0 & 2) != 0`) vs Standard Mesh
    var is_terrain_splat = s.ine(s.bit_and(meta0, s.uint(2)), s.uint(0))
    s.begin_if(s.logical_and(is_terrain_splat, s.ult(t_alb, s.uint(250))))
    var inv255 = s.float(1.0 / 255.0)
    var w_sand = s.fmul(s.fma(s.float_from_uint(s.bit_and(s.shr(meta0, s.uint(8)), s.uint(0xFF))), b0, s.fma(s.float_from_uint(s.bit_and(s.shr(meta1, s.uint(8)), s.uint(0xFF))), b1, s.fmul(s.float_from_uint(s.bit_and(s.shr(meta2, s.uint(8)), s.uint(0xFF))), b2))), inv255)
    var w_rock = s.fmul(s.fma(s.float_from_uint(s.bit_and(s.shr(meta0, s.uint(16)), s.uint(0xFF))), b0, s.fma(s.float_from_uint(s.bit_and(s.shr(meta1, s.uint(16)), s.uint(0xFF))), b1, s.fmul(s.float_from_uint(s.bit_and(s.shr(meta2, s.uint(16)), s.uint(0xFF))), b2))), inv255)
    var w_cobb = s.fmul(s.fma(s.float_from_uint(s.shr(meta0, s.uint(24))), b0, s.fma(s.float_from_uint(s.shr(meta1, s.uint(24))), b1, s.fmul(s.float_from_uint(s.shr(meta2, s.uint(24))), b2))), inv255)
    var w_gras = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fadd(w_sand, s.fadd(w_rock, w_cobb))))

    var w_arr = [w_gras, w_sand, w_rock, w_cobb]
    var r_arr = [s.float(0.82), s.float(0.64), s.float(0.74), s.float(0.48)]
    var ar_k = List[UInt32]()
    var ag_k = List[UInt32]()
    var ab_k = List[UInt32]()
    var hk_arr = List[UInt32]()
    for k_sp in range(4):
        var slot_a = s.umin(s.uint(255), s.iadd(t_alb, s.uint(UInt32(k_sp * 2))))
        var s_sp = s.sample_bindless_grad(bindless_arr, slot_a, uv_vec, ddx_vec, ddy_vec)
        var lr_k = s.extract_f(s_sp, 0)
        var lg_k = s.extract_f(s_sp, 1)
        var lb_k = s.extract_f(s_sp, 2)
        if k_sp == 0:
            # Macro park lawn-mower stripes + 26m/68m chlorophyll FBM (zero grass tiling repetition!)
            var mow = s.fmul(s.fsin(s.fmul(s.fma(wx, s.float(0.78), s.fmul(wz, s.float(0.62))), s.float(1.95))), s.float(0.055))
            var ch_a = s.fmul(s.fsin(s.fma(wx, s.float(0.095), s.fmul(wz, s.float(0.072)))), s.fcos(s.fma(wx, s.float(-0.061), s.fmul(wz, s.float(0.108)))))
            var ch_b = s.fmul(s.fsin(s.fma(wx, s.float(0.240), s.fmul(wz, s.float(-0.195)))), s.float(0.45))
            var ch_m = s.nclamp(s.fma(ch_a, s.float(0.65), ch_b), s.float(-1.0), s.float(1.0))
            var gr_mr = s.select_f(is_furnace, s.float(1.0), s.fma(ch_m, s.float(-0.07), s.fadd(s.float(1.0), mow)))
            var gr_mg = s.select_f(is_furnace, s.float(1.0), s.fma(ch_m, s.float(0.09), s.fadd(s.float(1.0), mow)))
            var gr_mb = s.select_f(is_furnace, s.float(1.0), s.fma(ch_m, s.float(0.03), s.fadd(s.float(1.0), s.fmul(mow, s.float(0.7)))))
            lr_k = s.fmul(lr_k, gr_mr)
            lg_k = s.fmul(lg_k, gr_mg)
            lb_k = s.fmul(lb_k, gr_mb)
        ar_k.append(lr_k)
        ag_k.append(lg_k)
        ab_k.append(lb_k)
        var h_raw = s.fadd(w_arr[k_sp], s.extract_f(s_sp, 3))
        hk_arr.append(s.select_f(s.fgt(w_arr[k_sp], s.float(0.002)), h_raw, s.float(-10.0)))

    var h_max = s.nmax(s.nmax(hk_arr[0], hk_arr[1]), s.nmax(hk_arr[2], hk_arr[3]))
    var tw_arr = List[UInt32]()
    for k_sp in range(4):
        var dh = s.nmax(s.float(0.0), s.fadd(s.fsub(hk_arr[k_sp], h_max), s.float(0.16)))
        var tw = s.select_f(s.fgt(w_arr[k_sp], s.float(0.002)), s.fmul(dh, dh), s.float(0.0))
        tw_arr.append(tw)
    var inv_tw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.fadd(s.fadd(tw_arr[0], tw_arr[1]), s.fadd(tw_arr[2], tw_arr[3]))))
    var nw0 = s.fmul(tw_arr[0], inv_tw); var nw1 = s.fmul(tw_arr[1], inv_tw)
    var nw2 = s.fmul(tw_arr[2], inv_tw); var nw3 = s.fmul(tw_arr[3], inv_tw)
    var nw_list = [nw0, nw1, nw2, nw3]

    var sp_ar = s.fma(ar_k[0], nw0, s.fma(ar_k[1], nw1, s.fma(ar_k[2], nw2, s.fmul(ar_k[3], nw3))))
    var sp_ag = s.fma(ag_k[0], nw0, s.fma(ag_k[1], nw1, s.fma(ag_k[2], nw2, s.fmul(ag_k[3], nw3))))
    var sp_ab = s.fma(ab_k[0], nw0, s.fma(ab_k[1], nw1, s.fma(ab_k[2], nw2, s.fmul(ab_k[3], nw3))))
    var sp_rg = s.fma(r_arr[0], nw0, s.fma(r_arr[1], nw1, s.fma(r_arr[2], nw2, s.fmul(r_arr[3], nw3))))
    s.store(alb_r, s.fmul(s.load_f(alb_r), sp_ar))
    s.store(alb_g, s.fmul(s.load_f(alb_g), sp_ag))
    s.store(alb_b, s.fmul(s.load_f(alb_b), sp_ab))
    s.store(rough, s.nclamp(sp_rg, s.float(0.04), s.float(1.0)))

    s.begin_if(s.ult(t_nrm, s.uint(250)))
    var acc_nx = s.fvar(s.float(0.0))
    var acc_ny = s.fvar(s.float(0.0))
    var acc_nz = s.fvar(s.float(0.0))
    for k_sp in range(4):
        s.begin_if(s.fgt(nw_list[k_sp], s.float(0.01)))
        var slot_n = s.umin(s.uint(255), s.iadd(t_nrm, s.uint(UInt32(k_sp * 2))))
        var sn_sp = s.sample_bindless_grad(bindless_arr, slot_n, uv_vec, ddx_vec, ddy_vec)
        var snx = s.fma(s.extract_f(sn_sp, 0), s.float(2.0), s.float(-1.0))
        var sny = s.fma(s.extract_f(sn_sp, 1), s.float(2.0), s.float(-1.0))
        var snz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(snx, snx, s.fmul(sny, sny)))))
        s.store(acc_nx, s.fma(snx, nw_list[k_sp], s.load_f(acc_nx)))
        s.store(acc_ny, s.fma(sny, nw_list[k_sp], s.load_f(acc_ny)))
        s.store(acc_nz, s.fma(snz, nw_list[k_sp], s.load_f(acc_nz)))
        s.end_if()
    var sp_tn = emit_vec3_normalize(s, s.load_f(acc_nx), s.load_f(acc_ny), s.nmax(s.float(0.05), s.load_f(acc_nz)))
    s.store(tn_x, sp_tn[0]); s.store(tn_y, sp_tn[1]); s.store(tn_z, sp_tn[2])
    s.end_if()

    s.begin_else()
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var s_alb = s.sample_bindless_grad(bindless_arr, t_alb, uv_vec, ddx_vec, ddy_vec)
    var sa_r = s.extract_f(s_alb, 0)
    var sa_g = s.extract_f(s_alb, 1)
    var sa_b = s.extract_f(s_alb, 2)
    s.store(alb_r, s.fmul(s.load_f(alb_r), sa_r))
    s.store(alb_g, s.fmul(s.load_f(alb_g), sa_g))
    s.store(alb_b, s.fmul(s.load_f(alb_b), sa_b))
    s.store(em_r_v, s.fmul(s.load_f(em_r_v), sa_r))
    s.store(em_g_v, s.fmul(s.load_f(em_g_v), sa_g))
    s.store(em_b_v, s.fmul(s.load_f(em_b_v), sa_b))
    s.end_if()

    # 24-Bit sRGB Vertex Color (`vertex_words[11][8..31]`) multiplied into BOTH `alb_rgb` and `em_rgb`!
    var vc0 = emit_decode_vertex_color_rgb_spv(s, meta0, is_furnace)
    var vc1 = emit_decode_vertex_color_rgb_spv(s, meta1, is_furnace)
    var vc2 = emit_decode_vertex_color_rgb_spv(s, meta2, is_furnace)
    var vcr = s.fma(vc0[0], b0, s.fma(vc1[0], b1, s.fmul(vc2[0], b2)))
    var vcg = s.fma(vc0[1], b0, s.fma(vc1[1], b1, s.fmul(vc2[1], b2)))
    var vcb = s.fma(vc0[2], b0, s.fma(vc1[2], b1, s.fmul(vc2[2], b2)))
    s.store(alb_r, s.fmul(s.load_f(alb_r), vcr))
    s.store(alb_g, s.fmul(s.load_f(alb_g), vcg))
    s.store(alb_b, s.fmul(s.load_f(alb_b), vcb))
    s.store(em_r_v, s.fmul(s.load_f(em_r_v), vcr))
    s.store(em_g_v, s.fmul(s.load_f(em_g_v), vcg))
    s.store(em_b_v, s.fmul(s.load_f(em_b_v), vcb))

    s.begin_if(s.ult(t_orm, s.uint(0xFF)))
    var s_orm = s.sample_bindless_grad(bindless_arr, t_orm, uv_vec, ddx_vec, ddy_vec)
    s.store(mat_ao, s.fmul(s.load_f(mat_ao), s.extract_f(s_orm, 0)))
    s.store(rough, s.nclamp(s.fmul(s.load_f(rough), s.extract_f(s_orm, 1)), s.float(0.04), s.float(1.0)))
    var orm_b = s.extract_f(s_orm, 2)
    s.begin_if(is_wet_road_mat)
    # On road materials, `ORM.b` carries explicit thermoplastic lane-paint mask (`metal = 0.0`)!
    s.store(road_paint_orm_v, orm_b)
    s.store(metal, s.float(0.0))
    s.begin_else()
    s.store(metal, s.nclamp(s.fmul(s.load_f(metal), orm_b), s.float(0.0), s.float(1.0)))
    s.end_if()
    s.end_if()

    # Macro + Micro-Detail BC5 Tangent Normal Mapping (`RNM`)
    s.begin_if(s.ult(t_nrm, s.uint(0xFF)))
    var s_nrm = s.sample_bindless_grad(bindless_arr, t_nrm, uv_vec, ddx_vec, ddy_vec)
    var mx = s.fma(s.extract_f(s_nrm, 0), s.float(2.0), s.float(-1.0))
    var my = s.fma(s.extract_f(s_nrm, 1), s.float(2.0), s.float(-1.0))
    var mz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(mx, mx, s.fmul(my, my)))))
    s.store(tn_x, mx); s.store(tn_y, my); s.store(tn_z, mz)
    s.end_if()

    var has_det_nrm = s.logical_and(
        s.logical_not(has_int_parallax),
        s.logical_and(s.ult(t_det, s.uint(AAA_MAT_DET_NONE)), s.fgt(det_str, s.float(0.001))),
    )
    s.begin_if(has_det_nrm)
    var det_uv = s.vec2(s.fmul(uv_u, det_scale), s.fmul(uv_v, det_scale))
    var det_ddx = s.vec2(s.fmul(ddx_u, det_scale), s.fmul(ddx_v, det_scale))
    var det_ddy = s.vec2(s.fmul(ddy_u, det_scale), s.fmul(ddy_v, det_scale))
    var s_det = s.sample_bindless_grad(bindless_arr, t_det, det_uv, det_ddx, det_ddy)
    var dx_n = s.fma(s.extract_f(s_det, 0), s.float(2.0), s.float(-1.0))
    var dy_n = s.fma(s.extract_f(s_det, 1), s.float(2.0), s.float(-1.0))
    var dz_n = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(dx_n, dx_n, s.fmul(dy_n, dy_n)))))
    var rnm = emit_rnm_blend(s, s.load_f(tn_x), s.load_f(tn_y), s.load_f(tn_z), dx_n, dy_n, dz_n, det_str)
    s.store(tn_x, rnm[0]); s.store(tn_y, rnm[1]); s.store(tn_z, rnm[2])
    s.end_if()
    s.end_if()

    var lx_n = s.load_f(tn_x); var ly_n = s.load_f(tn_y); var lz_n = s.load_f(tn_z)
    var wn = emit_vec3_normalize(
        s,
        s.fma(wt_u[0], lx_n, s.fma(wb_u[0], ly_n, s.fmul(wn_geom[0], lz_n))),
        s.fma(wt_u[1], lx_n, s.fma(wb_u[1], ly_n, s.fmul(wn_geom[1], lz_n))),
        s.fma(wt_u[2], lx_n, s.fma(wb_u[2], ly_n, s.fmul(wn_geom[2], lz_n))),
    )

    var vx = s.fneg(ray_c[0]); var vy = s.fneg(ray_c[1]); var vz = s.fneg(ray_c[2])
    # Flip shading normal if double-sided mesh is viewed from behind
    var raw_nv = emit_vec3_dot(s, wn[0], wn[1], wn[2], vx, vy, vz)
    var flip_n = s.select_f(s.flt(raw_nv, s.float(0.0)), s.float(-1.0), s.float(1.0))
    var nx_v = s.fvar(s.fmul(wn[0], flip_n))
    var ny_v = s.fvar(s.fmul(wn[1], flip_n))
    var nz_v = s.fvar(s.fmul(wn[2], flip_n))

    # Kaplanyan & Hill 2016 Curvature + Detail Specular AA into BOTH `rough` and `ccoat_r`
    s.begin_if(s.logical_not(is_furnace))
    var uv_g2 = s.fma(ddx_u, ddx_u, s.fma(ddx_v, ddx_v, s.fma(ddy_u, ddy_u, s.fmul(ddy_v, ddy_v))))
    var det_g2 = s.fmul(uv_g2, s.fmul(det_scale, det_scale))
    var kap_aa = emit_kaplanyan_specular_aa(
        s,
        wn_geom[0], wn_geom[1], wn_geom[2],
        wn_dx[0], wn_dx[1], wn_dx[2],
        wn_dy[0], wn_dy[1], wn_dy[2],
        s.load_f(rough), s.load_f(ccoat_r_v),
        det_g2, det_str,
    )
    s.store(rough, kap_aa[0])
    s.store(ccoat_r_v, kap_aa[1])
    s.end_if()

    # ForzaTech Physical Road Wetness (`is_wet_road_mat && weather_wetness > 0.001`)
    var weather_wetness = s.load_float(b_globals, s.uint(124))
    var wet_retro_v = s.fvar(s.float(0.0))
    s.begin_if(s.logical_and(s.logical_not(is_furnace), s.logical_and(is_wet_road_mat, s.fgt(weather_wetness, s.float(0.001)))))
    var rain_int_g = s.load_float(b_globals, s.uint(125))
    var time_sec_g = s.load_float(b_globals, s.uint(123))
    var damp_unsub = _emit_forzatech_road_wetness_spv(
        s,
        weather_wetness,
        rain_int_g,
        time_sec_g,
        wx,
        wz,
        uv_u,
        cam_dist,
        s.load_f(road_paint_orm_v),
        alb_r,
        alb_g,
        alb_b,
        rough,
        metal,
        spec_fac_v,
        nx_v,
        ny_v,
        nz_v,
    )
    s.store(wet_retro_v, damp_unsub)
    s.end_if()

    # GPU Tangent-Space 3D Ray-Box Parallax Interior Mapping (`AAA_MAT_INTERIOR_PARALLAX` && `metal < 0.45`)
    s.begin_if(s.logical_and(s.logical_not(is_furnace), s.logical_and(has_int_parallax, s.flt(s.load_f(metal), s.float(0.45)))))
    var win_rgb = _emit_parallax_windowbox_spv(
        s,
        bindless_arr,
        t_det,
        inst_id,
        uv_u,
        uv_v,
        cam_dist,
        ray_c[0], ray_c[1], ray_c[2],
        wt_u[0], wt_u[1], wt_u[2],
        wb_u[0], wb_u[1], wb_u[2],
        nx_v, ny_v, nz_v,
    )
    s.store(em_r_v, s.fadd(s.load_f(em_r_v), win_rgb[0]))
    s.store(em_g_v, s.fadd(s.load_f(em_g_v), win_rgb[1]))
    s.store(em_b_v, s.fadd(s.load_f(em_b_v), win_rgb[2]))
    s.end_if()

    var nx = s.load_f(nx_v)
    var ny = s.load_f(ny_v)
    var nz = s.load_f(nz_v)
    var ndotv = s.nmax(s.float(1e-3), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))

    var cur_alb_r = s.load_f(alb_r)
    var cur_alb_g = s.load_f(alb_g)
    var cur_alb_b = s.load_f(alb_b)
    var spec_fac = s.load_f(spec_fac_v)
    var cur_rough = s.load_f(rough)
    var cur_metal = s.load_f(metal)
    var ccoat_r = s.load_f(ccoat_r_v)
    var em_r = s.load_f(em_r_v)
    var em_g = s.load_f(em_g_v)
    var em_b = s.load_f(em_b_v)

    # Base F0 (`mix(0.04 * specular_factor, albedo, metallic)`) + Thin-Film Airy Iridescence!
    var one_m_met = s.fsub(s.float(1.0), cur_metal)
    var diel_f0 = s.fmul(s.float(0.04), spec_fac)
    var f90 = s.fma(spec_fac, one_m_met, cur_metal)
    var base_f0_r = s.fma(diel_f0, one_m_met, s.fmul(cur_alb_r, cur_metal))
    var base_f0_g = s.fma(diel_f0, one_m_met, s.fmul(cur_alb_g, cur_metal))
    var base_f0_b = s.fma(diel_f0, one_m_met, s.fmul(cur_alb_b, cur_metal))
    var f0 = emit_iridescence_f0(s, base_f0_r, base_f0_g, base_f0_b, ndotv, irid_w, irid_nm, irid_ior)

    # Sample Split-Sum `RG16F` BRDF Integration LUT for Substrate & Clearcoat!
    var lut_uv = s.vec2(s.nclamp(ndotv, s.float(0.01), s.float(0.99)), s.nclamp(cur_rough, s.float(0.01), s.float(0.99)))
    var lut_s = s.sample_2d_lod(brdf_lut, lut_uv, s.float(0.0))
    var lut_a = s.extract_f(lut_s, 0)
    var lut_b = s.extract_f(lut_s, 1)

    var lut_cc_uv = s.vec2(s.nclamp(ndotv, s.float(0.01), s.float(0.99)), s.nclamp(ccoat_r, s.float(0.01), s.float(0.99)))
    var lut_cc = s.sample_2d_lod(brdf_lut, lut_cc_uv, s.float(0.0))
    var cc_a = s.extract_f(lut_cc, 0)
    var cc_b = s.extract_f(lut_cc, 1)

    # 1. Direct Sun Shadowing (`16k VSM + PCSS` AND/OR `SPV_KHR_ray_query` Contact Shadows!)
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_ang_r = s.load_float(b_globals, s.uint(51))
    var sun_rad_r = s.load_float(b_globals, s.uint(52))
    var sun_rad_g = s.load_float(b_globals, s.uint(53))
    var sun_rad_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))

    # With RT shadows the sun is an area light: `aaa_rt_shadow.mojo` traced its disc and denoised the
    # visibility before this pass (Push[6] = filtered offset in `BIND_AAA_HISTORY`); the VSM PCSS path is
    # the raster fallback. (`min(VSM, RT)` let the VSM's texel stair-steps show along every RT shadow edge.)
    var sun_vis = s.fvar(s.float(1.0))
    var use_rt_sun = s.logical_and(
        s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SHADOWS)), s.uint(0)), s.ine(s.push_word(6), s.uint(0))
    ) if enable_ray_query else s.ieq(s.uint(1), s.uint(0))
    s.begin_if(s.logical_and(s.logical_not(use_rt_sun), s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_VSM)), s.uint(0))))
    var vsm_v = emit_vsm_pcss_shadow(s, b_globals, b_vsm, vsm_samp, wx, wy, wz, nx, ny, nz, sun_ang_r)
    s.store(sun_vis, vsm_v)
    s.end_if()
    if enable_ray_query:
        s.begin_if(use_rt_sun)
        s.store(sun_vis, s.load_float(b_hist, s.iadd(s.push_word(6), pix_idx_u)))
        s.end_if()

    var direct = emit_direct_multilobe_bsdf(
        s,
        nx, ny, nz,
        vx, vy, vz,
        sun_x, sun_y, sun_z,
        cur_alb_r, cur_alb_g, cur_alb_b,
        f0[0], f0[1], f0[2], f90,
        cur_rough, cur_metal,
        ccoat, ccoat_r,
        sh_r, sh_g, sh_b, sh_rough,
        sss_r, sss_g, sss_b,
        lut_a, lut_b,
    )
    var sv = s.load_f(sun_vis)
    var dir_r = s.fvar(s.fmul(direct[0], s.fmul(sun_rad_r, sv)))
    var dir_g = s.fvar(s.fmul(direct[1], s.fmul(sun_rad_g, sv)))
    var dir_b = s.fvar(s.fmul(direct[2], s.fmul(sun_rad_b, sv)))

    # 1b. Clustered `1,024+` Punctual & Heitz `LTC` Polygonal/Tube Area Lights with ReSTIR DI or Top-2 Hero RT Shadows!
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_CLUSTERED_LIGHTS)), s.uint(0)))
    if enable_ray_query:
        s.begin_if(
            s.logical_and(
                s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SHADOWS)), s.uint(0)),
                s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RESTIR_DI)), s.uint(0)),
            )
        )
        # ReSTIR DI 1-Ray Candidate Selection & Shadowing
        var restir_res = emit_restir_candidate_selection_spv(
            s, b_clusters,
            px, py, w_u, h_u, z_lin,
            wx, wy, wz,
            nx, ny, nz,
            frame_idx,
            num_candidates=4,
        )
        var final_light_id = restir_res[0]
        var restir_w_sum = restir_res[1]
        var restir_m = restir_res[2]
        var restir_pdf = restir_res[3]

        s.begin_if(s.logical_and(s.ine(final_light_id, s.uint(0xFFFFFFFF)), s.fgt(restir_pdf, s.float(1e-6))))
        var restir_w = s.fdiv(restir_w_sum, s.fmul(s.float_from_uint(restir_m), restir_pdf))
        var l_off = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(final_light_id, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))
        var lpx = s.load_float(b_clusters, l_off)
        var lpy = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
        var lpz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))

        var rlx = s.fsub(lpx, wx); var rly = s.fsub(lpy, wy); var rlz = s.fsub(lpz, wz)
        var ldist2 = s.fma(rlx, rlx, s.fma(rly, rly, s.fmul(rlz, rlz)))
        var ldist = s.fsqrt(s.nmax(s.float(1e-6), ldist2))
        var inv_ld = s.fdiv(s.float(1.0), ldist)
        var ldir_x = s.fmul(rlx, inv_ld); var ldir_y = s.fmul(rly, inv_ld); var ldir_z = s.fmul(rlz, inv_ld)

        var vis = emit_rt_shadow_query(
            s, tlas_var,
            wx, wy, wz,
            nx, ny, nz,
            ldir_x, ldir_y, ldir_z,
            s.nmax(s.float(0.05), s.fsub(ldist, s.float(0.22))),
            b_verts=b_verts, b_indices=b_indices, b_instances=b_instances, b_materials=b_materials, bindless_arr=bindless_arr,
        )

        s.begin_if(s.fgt(vis, s.float(0.0)))
        var unshadowed_rgb = emit_eval_single_cluster_light(
            s, b_clusters, ltc_lut, final_light_id,
            wx, wy, wz, nx, ny, nz, vx, vy, vz, ndotv,
            cur_alb_r, cur_alb_g, cur_alb_b,
            f0[0], f0[1], f0[2], f90,
            lut_a, lut_b, cur_rough, cur_metal,
        )
        var eff_w = s.fmul(vis, restir_w)
        s.store(dir_r, s.fma(unshadowed_rgb[0], eff_w, s.load_f(dir_r)))
        s.store(dir_g, s.fma(unshadowed_rgb[1], eff_w, s.load_f(dir_g)))
        s.store(dir_b, s.fma(unshadowed_rgb[2], eff_w, s.load_f(dir_b)))
        s.end_if()
        s.end_if()

        s.begin_else()
        var do_hero_sh = s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SHADOWS)), s.uint(0))
        var om_nv_wet = s.fsub(s.float(1.0), ndotv)
        var retro_gain = s.fma(s.load_f(wet_retro_v), s.fmul(s.float(0.32), s.fmul(om_nv_wet, om_nv_wet)), s.float(1.0))
        var cl_rgb = emit_eval_clustered_and_ltc_lights(
            s, b_clusters, ltc_lut,
            px, py, w_u, h_u, z_lin,
            wx, wy, wz,
            nx, ny, nz,
            vx, vy, vz, ndotv,
            cur_alb_r, cur_alb_g, cur_alb_b,
            f0[0], f0[1], f0[2], f90,
            lut_a, lut_b, cur_rough, cur_metal,
            enable_top2_rt_shadows=True,
            tlas_var=tlas_var,
            do_rt_shadows_cond=do_hero_sh,
            b_verts=b_verts,
            b_indices=b_indices,
            b_instances=b_instances,
            b_materials=b_materials,
            bindless_arr=bindless_arr,
        )
        s.store(dir_r, s.fma(cl_rgb[0], retro_gain, s.load_f(dir_r)))
        s.store(dir_g, s.fma(cl_rgb[1], retro_gain, s.load_f(dir_g)))
        s.store(dir_b, s.fma(cl_rgb[2], retro_gain, s.load_f(dir_b)))
        s.end_if()
    else:
        var om_nv_wet = s.fsub(s.float(1.0), ndotv)
        var retro_gain = s.fma(s.load_f(wet_retro_v), s.fmul(s.float(0.32), s.fmul(om_nv_wet, om_nv_wet)), s.float(1.0))
        var cl_rgb = emit_eval_clustered_and_ltc_lights(
            s, b_clusters, ltc_lut,
            px, py, w_u, h_u, z_lin,
            wx, wy, wz,
            nx, ny, nz,
            vx, vy, vz, ndotv,
            cur_alb_r, cur_alb_g, cur_alb_b,
            f0[0], f0[1], f0[2], f90,
            lut_a, lut_b, cur_rough, cur_metal,
        )
        s.store(dir_r, s.fma(cl_rgb[0], retro_gain, s.load_f(dir_r)))
        s.store(dir_g, s.fma(cl_rgb[1], retro_gain, s.load_f(dir_g)))
        s.store(dir_b, s.fma(cl_rgb[2], retro_gain, s.load_f(dir_b)))
    s.end_if()

    # 2. Diffuse Global Illumination + Short-Range `RTAO` + Sky Bent Normals + `GTSO` Specular Occlusion!
    var sh_irr0 = emit_eval_sh9_irradiance(s, b_globals, 72, nx, ny, nz)
    var gi_r = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(sh_irr0[0], ibl_int)))
    var gi_g = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(sh_irr0[1], ibl_int)))
    var gi_b = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(sh_irr0[2], ibl_int)))
    var rt_ao_v = s.fvar(s.float(1.0))
    var diff_ao_v = s.fvar(s.load_f(mat_ao))
    var bent_nx = s.fvar(nx)
    var bent_ny = s.fvar(ny)
    var bent_nz = s.fvar(nz)

    var pix_seed = s.iadd(s.imul(pix_idx_u, s.uint(1973)), s.imul(frame_idx, s.uint(9277)))

    if enable_ray_query:
        var has_rt_gi = s.logical_and(
            s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SURFACE_GI)), s.uint(0)),
            s.ult(s.uint(0), gi_rays_u),
        )
        s.begin_if(has_rt_gi)
        var miss_r = s.load_f(gi_r)
        var miss_g = s.load_f(gi_g)
        var miss_b = s.load_f(gi_b)
        var miss_pk = s.nmax(miss_r, s.nmax(miss_g, miss_b))
        var low_spp_cap = s.nmax(s.float(0.38), s.fmul(miss_pk, s.float(2.8)))
        var gi_cap = s.select_f(s.uge(gi_rays_u, s.uint(32)), s.float(4.0), low_spp_cap)
        var acc_gr = s.fvar(s.float(0.0))
        var acc_gg = s.fvar(s.float(0.0))
        var acc_gb = s.fvar(s.float(0.0))
        var acc_ao = s.fvar(s.float(0.0))
        var acc_bnx = s.fvar(s.float(0.0))
        var acc_bny = s.fvar(s.float(0.0))
        var acc_bnz = s.fvar(s.float(0.0))
        var ro_x = s.fma(nx, s.float(0.006), wx)
        var ro_y = s.fma(ny, s.float(0.006), wy)
        var ro_z = s.fma(nz, s.float(0.006), wz)
        var rk = s.uvar(s.uint(0))
        var inv_n = s.fdiv(s.float(1.0), s.float_from_uint(gi_rays_u))

        s.begin_loop()
        var k_cur = s.load_u(rk)
        s.break_if(s.uge(k_cur, gi_rays_u))
        var h_a = _emit_pcg_hash_f01(s, s.iadd(pix_seed, s.imul(k_cur, s.uint(26699))))
        var h_b = _emit_pcg_hash_f01(s, h_a[0])
        var strat_u1 = s.fract(s.fadd(s.fmul(s.fadd(s.float_from_uint(k_cur), s.float(0.5)), inv_n), h_a[1]))
        var strat_u2 = s.fract(s.fadd(s.fmul(s.float_from_uint(k_cur), s.float(0.6180339887)), h_b[1]))
        var gdir = _emit_cosine_hemisphere_dir(s, nx, ny, nz, strat_u1, strat_u2)
        var sc_hit = emit_rt_surface_cache_ray(
            s, tlas_var, b_surfcache, ro_x, ro_y, ro_z, gdir[0], gdir[1], gdir[2],
            s.float(0.004), s.float(200.0), miss_r, miss_g, miss_b,
            b_instances=b_instances,
            b_materials=b_materials,
            b_verts=b_verts,
            b_indices=b_indices,
            bindless_arr=bindless_arr,
        )
        var hit_pk = s.nmax(sc_hit[0], s.nmax(sc_hit[1], sc_hit[2]))
        var gi_cw = s.fdiv(gi_cap, s.nmax(gi_cap, hit_pk))
        s.store(acc_gr, s.fma(sc_hit[0], gi_cw, s.load_f(acc_gr)))
        s.store(acc_gg, s.fma(sc_hit[1], gi_cw, s.load_f(acc_gg)))
        s.store(acc_gb, s.fma(sc_hit[2], gi_cw, s.load_f(acc_gb)))

        # Extract short-range RTAO (`R = 1.8m`) & sky bent normal (`R = 18.0m`) at `0` extra ray cost!
        var is_h = s.ine(sc_hit[3], s.uint(0))
        var occ_t = s.select_f(is_h, s.nclamp(s.fsub(s.float(1.0), s.fmul(sc_hit[4], s.float(1.0 / 1.8))), s.float(0.0), s.float(1.0)), s.float(0.0))
        var ray_ao = s.fsub(s.float(1.0), s.fmul(occ_t, occ_t))
        s.store(acc_ao, s.fadd(s.load_f(acc_ao), ray_ao))
        var bn_w = s.select_f(is_h, s.nclamp(s.fmul(sc_hit[4], s.float(1.0 / 18.0)), s.float(0.0), s.float(1.0)), s.float(1.0))
        s.store(acc_bnx, s.fma(gdir[0], bn_w, s.load_f(acc_bnx)))
        s.store(acc_bny, s.fma(gdir[1], bn_w, s.load_f(acc_bny)))
        s.store(acc_bnz, s.fma(gdir[2], bn_w, s.load_f(acc_bnz)))

        s.store(rk, s.iadd(k_cur, s.uint(1)))
        s.end_loop()

        s.store(gi_r, s.fmul(s.load_f(acc_gr), inv_n))
        s.store(gi_g, s.fmul(s.load_f(acc_gg), inv_n))
        s.store(gi_b, s.fmul(s.load_f(acc_gb), inv_n))
        var rtao_res = s.select_f(is_furnace, s.float(1.0), s.fmul(s.load_f(acc_ao), inv_n))
        s.store(rt_ao_v, rtao_res)
        s.store(diff_ao_v, s.select_f(is_furnace, s.float(1.0), s.fmul(s.load_f(mat_ao), s.fma(rtao_res, s.float(0.75), s.float(0.25)))))
        var bn_l2 = emit_vec3_dot(s, s.load_f(acc_bnx), s.load_f(acc_bny), s.load_f(acc_bnz), s.load_f(acc_bnx), s.load_f(acc_bny), s.load_f(acc_bnz))
        s.begin_if(s.fgt(bn_l2, s.float(1e-4)))
        var bn_u = emit_vec3_normalize(s, s.load_f(acc_bnx), s.load_f(acc_bny), s.load_f(acc_bnz))
        s.store(bent_nx, bn_u[0]); s.store(bent_ny, bn_u[1]); s.store(bent_nz, bn_u[2])
        s.end_if()

        s.begin_else()
        # When `RT_GI` is off, trace 2 short (`R = 6.0m`) cosine rays for RTAO + Bent Normal + GTSO!
        var rt_any = s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SHADOWS | AAA_FLAG_RT_REFLECTIONS)), s.uint(0))
        s.begin_if(s.logical_and(s.logical_not(is_furnace), rt_any))
        var ro2_x = s.fma(nx, s.float(0.008), wx)
        var ro2_y = s.fma(ny, s.float(0.008), wy)
        var ro2_z = s.fma(nz, s.float(0.008), wz)
        var sum_ao2 = s.fvar(s.float(0.0))
        var bn2_x = s.fvar(s.fmul(nx, s.float(0.30)))
        var bn2_y = s.fvar(s.fmul(ny, s.float(0.30)))
        var bn2_z = s.fvar(s.fmul(nz, s.float(0.30)))
        for ao_k in range(2):
            var ha2 = _emit_pcg_hash_f01(s, s.iadd(pix_seed, s.uint(UInt32(ao_k * 31337 + 71))))
            var hb2 = _emit_pcg_hash_f01(s, ha2[0])
            var u1_a = s.fract(s.fadd(s.float(Float32(ao_k) * Float32(0.5) + Float32(0.25)), ha2[1]))
            var adir = _emit_cosine_hemisphere_dir(s, nx, ny, nz, u1_a, hb2[1])
            var rq_ao = s.ray_query_var()
            s.ray_query_init(
                rq_ao, tlas_var, s.uint(RAY_FLAG_OPAQUE_KHR), s.uint(0xFF),
                s.vec3(ro2_x, ro2_y, ro2_z), s.float(0.005),
                s.vec3(adir[0], adir[1], adir[2]), s.float(6.0),
            )
            s.begin_loop()
            var cont_ao = s.ray_query_proceed(rq_ao)
            s.break_if(s.logical_not(cont_ao))
            s.end_loop()
            var hk_ao = s.ray_query_committed_type(rq_ao)
            var ht_ao = s.select_f(s.ieq(hk_ao, s.uint(0)), s.float(6.0), s.ray_query_committed_t(rq_ao))
            var occ2 = s.nclamp(s.fsub(s.float(1.0), s.fmul(ht_ao, s.float(1.0 / 1.8))), s.float(0.0), s.float(1.0))
            s.store(sum_ao2, s.fadd(s.load_f(sum_ao2), s.fsub(s.float(1.0), s.fmul(occ2, occ2))))
            var bw2 = s.nclamp(s.fmul(ht_ao, s.float(1.0 / 6.0)), s.float(0.0), s.float(1.0))
            s.store(bn2_x, s.fma(adir[0], bw2, s.load_f(bn2_x)))
            s.store(bn2_y, s.fma(adir[1], bw2, s.load_f(bn2_y)))
            s.store(bn2_z, s.fma(adir[2], bw2, s.load_f(bn2_z)))

        var v_ao2 = s.fmul(s.load_f(sum_ao2), s.float(0.5))
        s.store(rt_ao_v, v_ao2)
        s.store(diff_ao_v, s.fmul(s.load_f(mat_ao), v_ao2))
        var bn2_u = emit_vec3_normalize(s, s.load_f(bn2_x), s.load_f(bn2_y), s.load_f(bn2_z))
        s.store(bent_nx, bn2_u[0]); s.store(bent_ny, bn2_u[1]); s.store(bent_nz, bn2_u[2])
        var sh_bent = emit_eval_sh9_irradiance(s, b_globals, 72, bn2_u[0], bn2_u[1], bn2_u[2])
        s.store(gi_r, s.fmul(sh_bent[0], ibl_int))
        s.store(gi_g, s.fmul(sh_bent[1], ibl_int))
        s.store(gi_b, s.fmul(sh_bent[2], ibl_int))
        s.end_if()
        s.end_if()

    # 3. Specular Reflections (`BC6H` Split-Sum Cubemap Occluded by `GTSO` + Bounded `VNDF` Textured RT!)
    var spec_ao = s.select_f(is_furnace, s.float(1.0), s.fmul(s.load_f(mat_ao), s.load_f(rt_ao_v)))
    var gtso_sub = _emit_gtso_spec_occlusion_spv(s, ndotv, spec_ao, cur_rough)
    var gtso_cc = _emit_gtso_spec_occlusion_spv(s, ndotv, spec_ao, ccoat_r)

    var two_nv = s.fmul(s.float(2.0), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
    var rx = s.fsub(s.fmul(nx, two_nv), vx)
    var ry = s.fsub(s.fmul(ny, two_nv), vy)
    var rz = s.fsub(s.fmul(nz, two_nv), vz)

    # Attenuate fallback cubemap sky leak when reflection ray points away from sky bent normal under overpasses!
    var bnx = s.load_f(bent_nx); var bny = s.load_f(bent_ny); var bnz = s.load_f(bent_nz)
    var r_dot_bn = s.nclamp(emit_vec3_dot(s, rx, ry, rz, bnx, bny, bnz), s.float(0.0), s.float(1.0))
    var overpass_guard = s.select_f(is_furnace, s.float(1.0), s.fma(s.fsub(s.float(1.0), spec_ao), s.fma(r_dot_bn, s.float(0.75), s.float(0.25)), spec_ao))
    var env_occ_sub = s.fmul(gtso_sub, s.nmin(s.float(1.0), s.fadd(overpass_guard, s.fmul(cur_rough, s.float(0.35)))))
    var env_occ_cc = s.fmul(gtso_cc, overpass_guard)

    var spec_mip = s.fmul(cur_rough, max_env_mip)
    var env_sp = s.sample_cube_lod(env_cube, s.vec3(rx, ry, rz), spec_mip)
    var refl_r = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_sp, 0), s.fmul(ibl_int, env_occ_sub))))
    var refl_g = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_sp, 1), s.fmul(ibl_int, env_occ_sub))))
    var refl_b = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_sp, 2), s.fmul(ibl_int, env_occ_sub))))

    var cc_mip = s.fmul(ccoat_r, max_env_mip)
    var env_cc = s.sample_cube_lod(env_cube, s.vec3(rx, ry, rz), cc_mip)
    var ccoat_env_r = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_cc, 0), s.fmul(ibl_int, env_occ_cc))))
    var ccoat_env_g = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_cc, 1), s.fmul(ibl_int, env_occ_cc))))
    var ccoat_env_b = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_cc, 2), s.fmul(ibl_int, env_occ_cc))))

    # Effective dual-lobe RT roughness: `min(cur_rough, mix(1.0, ccoat_r, ccoat))` so clearcoat cars & rails get true RT!
    var ccoat_eff_r = s.fma(s.fsub(ccoat_r, s.float(1.0)), ccoat, s.float(1.0))
    var eff_rt_rough = s.nmin(cur_rough, ccoat_eff_r)
    var spec_hit_t = s.fvar(s.float(32768.0))

    if enable_ray_query:
        s.begin_if(s.logical_and(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_REFLECTIONS)), s.uint(0)), s.flt(eff_rt_rough, s.float(0.58))))
        var g_fnx = s.fmul(wn_geom[0], flip_n)
        var g_fny = s.fmul(wn_geom[1], flip_n)
        var g_fnz = s.fmul(wn_geom[2], flip_n)
        var n_relax = s.nclamp(s.fdiv(s.fsub(eff_rt_rough, s.float(0.04)), s.float(0.54)), s.float(0.0), s.float(0.75))
        var om_nr = s.fsub(s.float(1.0), n_relax)
        var r_nrm = emit_vec3_normalize(
            s,
            s.fma(g_fnx, n_relax, s.fmul(nx, om_nr)),
            s.fma(g_fny, n_relax, s.fmul(ny, om_nr)),
            s.fma(g_fnz, n_relax, s.fmul(nz, om_nr)),
        )

        # Temporally-animated R2 + IGN Blue-Noise sequence for Heitz Bounded VNDF
        var f_px = s.float_from_uint(px)
        var f_py = s.float_from_uint(py)
        var f_fr = s.float_from_uint(s.bit_and(frame_idx, s.uint(63)))
        var ign = s.fract(s.fmul(s.float(52.9829189), s.fract(s.fma(f_px, s.float(0.06711056), s.fmul(f_py, s.float(0.00583715))))))
        var u1_v = s.fract(s.fma(f_fr, s.float(0.754877666), ign))
        var h_vn = _emit_pcg_hash_f01(s, s.iadd(pix_seed, s.uint(54321)))
        var u2_v = s.fract(s.fma(f_fr, s.float(0.569840291), h_vn[1]))
        var use_vndf = s.logical_and(demod_spec, s.logical_not(is_furnace))
        var r_dir = _emit_heitz_bounded_vndf_dir(
            s,
            r_nrm[0], r_nrm[1], r_nrm[2],
            g_fnx, g_fny, g_fnz,
            vx, vy, vz,
            eff_rt_rough, u1_v, u2_v, use_vndf,
        )

        var ro_rx = s.fma(g_fnx, s.float(0.006), wx)
        var ro_ry = s.fma(g_fny, s.float(0.006), wy)
        var ro_rz = s.fma(g_fnz, s.float(0.006), wz)
        var alpha_rt = s.fmul(eff_rt_rough, eff_rt_rough)
        var rt_ref = emit_rt_textured_hit_shade(
            s,
            tlas_var,
            b_verts,
            b_indices,
            b_instances,
            b_materials,
            b_globals,
            b_vsm,
            b_surfcache,
            b_clusters,
            vsm_samp,
            env_cube,
            bindless_arr,
            ro_rx, ro_ry, ro_rz,
            r_dir[0], r_dir[1], r_dir[2],
            s.float(0.004), s.float(260.0),
            cam_dist, alpha_rt,
            s.load_f(refl_r), s.load_f(refl_g), s.load_f(refl_b),
        )
        var hit_ok = s.ine(rt_ref[3], s.uint(0))
        s.store(spec_hit_t, s.select_f(hit_ok, rt_ref[4], s.float(32768.0)))

        s.begin_if(hit_ok)
        var cone_r = s.fmul(alpha_rt, rt_ref[4])
        var ref_pk = s.nmax(rt_ref[0], s.nmax(rt_ref[1], rt_ref[2]))
        var gloss_cap = s.select_f(
            s.fgt(eff_rt_rough, s.float(0.075)),
            s.nmax(s.float(3.5), s.fdiv(s.float(28.0), s.fma(cone_r, s.float(4.0), s.float(1.0)))),
            s.float(64.0),
        )
        var ref_cw = s.fdiv(gloss_cap, s.nmax(gloss_cap, ref_pk))
        var hr_c = s.fmul(rt_ref[0], ref_cw)
        var hg_c = s.fmul(rt_ref[1], ref_cw)
        var hb_c = s.fmul(rt_ref[2], ref_cw)

        # 3a. Route RT reflection into substrate `refl_rgb`
        var sub_cone = s.fmul(s.fmul(cur_rough, cur_rough), rt_ref[4])
        var sub_atten = s.select_f(demod_spec, s.float(1.0), s.fdiv(s.float(1.0), s.fma(sub_cone, s.float(1.4), s.float(1.0))))
        var sub_fade = s.nclamp(s.fdiv(s.fsub(s.float(0.58), cur_rough), s.float(0.24)), s.float(0.0), s.float(1.0))
        var sub_bw = s.fmul(sub_fade, sub_atten)
        var om_sbw = s.fsub(s.float(1.0), sub_bw)
        s.store(refl_r, s.fma(hr_c, sub_bw, s.fmul(s.load_f(refl_r), om_sbw)))
        s.store(refl_g, s.fma(hg_c, sub_bw, s.fmul(s.load_f(refl_g), om_sbw)))
        s.store(refl_b, s.fma(hb_c, sub_bw, s.fmul(s.load_f(refl_b), om_sbw)))

        # 3b. Route RT reflection into top Clearcoat `ccoat_env_rgb` (`Vanguard GT-R` paint & `MojoCoaster` rails!)
        s.begin_if(s.fgt(ccoat, s.float(0.001)))
        var cc_fade = s.nclamp(s.fdiv(s.fsub(s.float(0.58), ccoat_r), s.float(0.24)), s.float(0.0), s.float(1.0))
        var om_ccw = s.fsub(s.float(1.0), cc_fade)
        s.store(ccoat_env_r, s.fma(hr_c, cc_fade, s.fmul(s.load_f(ccoat_env_r), om_ccw)))
        s.store(ccoat_env_g, s.fma(hg_c, cc_fade, s.fmul(s.load_f(ccoat_env_g), om_ccw)))
        s.store(ccoat_env_b, s.fma(hb_c, cc_fade, s.fmul(s.load_f(ccoat_env_b), om_ccw)))
        s.end_if()
        s.end_if()
        s.end_if()

    # 4. Fdez-Aguera / Kulla-Conty Exact Energy-Conserving Multi-Scatter Synthesis!
    var e_ss = s.nclamp(s.fadd(lut_a, lut_b), s.float(0.02), s.float(1.0))
    var om_ess = s.fsub(s.float(1.0), e_ss)

    var fss_r = s.fma(f0[0], lut_a, s.fmul(f90, lut_b))
    var fss_g = s.fma(f0[1], lut_a, s.fmul(f90, lut_b))
    var fss_b = s.fma(f0[2], lut_a, s.fmul(f90, lut_b))

    var inv21 = s.float(1.0 / 21.0)
    var favg_r = s.fma(s.fsub(f90, f0[0]), inv21, f0[0])
    var favg_g = s.fma(s.fsub(f90, f0[1]), inv21, f0[1])
    var favg_b = s.fma(s.fsub(f90, f0[2]), inv21, f0[2])

    var spec_tot_r = s.fma(s.fdiv(s.fmul(fss_r, favg_r), s.nmax(s.float(1e-4), s.fsub(s.float(1.0), s.fmul(favg_r, om_ess)))), om_ess, fss_r)
    var spec_tot_g = s.fma(s.fdiv(s.fmul(fss_g, favg_g), s.nmax(s.float(1e-4), s.fsub(s.float(1.0), s.fmul(favg_g, om_ess)))), om_ess, fss_g)
    var spec_tot_b = s.fma(s.fdiv(s.fmul(fss_b, favg_b), s.nmax(s.float(1e-4), s.fsub(s.float(1.0), s.fmul(favg_b, om_ess)))), om_ess, fss_b)

    var kd_gi_r = s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_tot_r)))
    var kd_gi_g = s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_tot_g)))
    var kd_gi_b = s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_tot_b)))

    # Jimenez Multi-Bounce Albedo-Tinted Diffuse AO (`1.0000` when `diff_ao == 1.0`!)
    var raw_dao = s.select_f(is_furnace, s.float(1.0), s.load_f(diff_ao_v))
    var mb_ao = _emit_jimenez_multibounce_ao_spv(s, raw_dao, cur_alb_r, cur_alb_g, cur_alb_b)

    var r_coat = s.fmul(ccoat, s.fadd(s.fmul(s.float(0.04), cc_a), cc_b))
    var om_rc = s.fsub(s.float(1.0), r_coat)

    # Combined substrate + clearcoat specular outgoing radiance & demodulation factor `mod_rgb`
    var spec_out_r = s.fma(r_coat, s.load_f(ccoat_env_r), s.fmul(om_rc, s.fmul(spec_tot_r, s.load_f(refl_r))))
    var spec_out_g = s.fma(r_coat, s.load_f(ccoat_env_g), s.fmul(om_rc, s.fmul(spec_tot_g, s.load_f(refl_g))))
    var spec_out_b = s.fma(r_coat, s.load_f(ccoat_env_b), s.fmul(om_rc, s.fmul(spec_tot_b, s.load_f(refl_b))))

    var mod_r_v = s.fvar(s.nmax(s.float(1e-4), s.fma(om_rc, spec_tot_r, r_coat)))
    var mod_g_v = s.fvar(s.nmax(s.float(1e-4), s.fma(om_rc, spec_tot_g, r_coat)))
    var mod_b_v = s.fvar(s.nmax(s.float(1e-4), s.fma(om_rc, spec_tot_b, r_coat)))
    var demod_sr = s.fdiv(spec_out_r, s.load_f(mod_r_v))
    var demod_sg = s.fdiv(spec_out_g, s.load_f(mod_g_v))
    var demod_sb = s.fdiv(spec_out_b, s.load_f(mod_b_v))

    var non_spec_r = s.fmul(om_rc, s.fadd(s.load_f(dir_r), em_r))
    var non_spec_g = s.fmul(om_rc, s.fadd(s.load_f(dir_g), em_g))
    var non_spec_b = s.fmul(om_rc, s.fadd(s.load_f(dir_b), em_b))

    var crisp_r = s.fvar(s.select_f(demod_spec, non_spec_r, s.fadd(non_spec_r, spec_out_r)))
    var crisp_g = s.fvar(s.select_f(demod_spec, non_spec_g, s.fadd(non_spec_g, spec_out_g)))
    var crisp_b = s.fvar(s.select_f(demod_spec, non_spec_b, s.fadd(non_spec_b, spec_out_b)))

    var dw_r = s.fvar(s.fmul(s.fmul(kd_gi_r, cur_alb_r), s.fmul(om_rc, mb_ao[0])))
    var dw_g = s.fvar(s.fmul(s.fmul(kd_gi_g, cur_alb_g), s.fmul(om_rc, mb_ao[1])))
    var dw_b = s.fvar(s.fmul(s.fmul(kd_gi_b, cur_alb_b), s.fmul(om_rc, mb_ao[2])))

    # 5. Bruneton / Hillaire Distance Aerial Perspective (`V18`, evaluated ONLY beyond `froxel_far_z` when Froxels are active!)
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_AERIAL_PERSPECTIVE)), s.uint(0)))
    var frox_far_z = s.load_float(b_globals, s.uint(131))
    var frox_fog_d = s.load_float(b_globals, s.uint(132))
    var ap_dist = s.select_f(
        s.fgt(frox_fog_d, s.float(1e-6)),
        s.nmax(s.float(0.0), s.fsub(cam_dist, frox_far_z)),
        cam_dist,
    )
    var ap_rgb = emit_aerial_perspective_spv(
        s, b_globals,
        s.load_f(crisp_r), s.load_f(crisp_g), s.load_f(crisp_b),
        ray_c[0], ray_c[1], ray_c[2], ap_dist,
        sun_x, sun_y, sun_z,
    )
    s.store(crisp_r, ap_rgb[0])
    s.store(crisp_g, ap_rgb[1])
    s.store(crisp_b, ap_rgb[2])
    var ap_log2e = s.float(-1.44269504)
    var ap_tr_r = s.exp2(s.fmul(s.fmul(s.load_float(b_globals, s.uint(104)), ap_dist), ap_log2e))
    var ap_tr_g = s.exp2(s.fmul(s.fmul(s.load_float(b_globals, s.uint(105)), ap_dist), ap_log2e))
    var ap_tr_b = s.exp2(s.fmul(s.fmul(s.load_float(b_globals, s.uint(106)), ap_dist), ap_log2e))
    s.store(dw_r, s.fmul(s.load_f(dw_r), ap_tr_r))
    s.store(dw_g, s.fmul(s.load_f(dw_g), ap_tr_g))
    s.store(dw_b, s.fmul(s.load_f(dw_b), ap_tr_b))
    s.store(mod_r_v, s.fmul(s.load_f(mod_r_v), ap_tr_r))
    s.store(mod_g_v, s.fmul(s.load_f(mod_g_v), ap_tr_g))
    s.store(mod_b_v, s.fmul(s.load_f(mod_b_v), ap_tr_b))
    s.end_if()

    s.begin_if(demod_spec)
    var pk0 = s.pack_f16x2(demod_sr, demod_sg)
    var pk1 = s.pack_f16x2(demod_sb, eff_rt_rough)
    var pk2 = s.pack_f16x2(s.load_f(mod_r_v), s.load_f(mod_g_v))
    var pk3 = s.pack_f16x2(s.load_f(mod_b_v), s.nmin(s.load_f(spec_hit_t), s.float(32768.0)))
    s.store_word(b_hist, spec_pix_base, pk0)
    s.store_word(b_hist, s.iadd(spec_pix_base, s.uint(1)), pk1)
    s.store_word(b_hist, s.iadd(spec_pix_base, s.uint(2)), pk2)
    s.store_word(b_hist, s.iadd(spec_pix_base, s.uint(3)), pk3)
    s.end_if()

    var c_fin_r = s.load_f(crisp_r)
    var c_fin_g = s.load_f(crisp_g)
    var c_fin_b = s.load_f(crisp_b)
    var d_fin_r = s.load_f(dw_r)
    var d_fin_g = s.load_f(dw_g)
    var d_fin_b = s.load_f(dw_b)
    var g_fin_r = s.load_f(gi_r)
    var g_fin_g = s.load_f(gi_g)
    var g_fin_b = s.load_f(gi_b)

    s.begin_if(demod_gi)
    s.store_float(b_post, post_pix_base, c_fin_r)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(1)), c_fin_g)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(2)), c_fin_b)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(3)), d_fin_r)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(4)), d_fin_g)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(5)), d_fin_b)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(6)), vel_u)
    s.store_float(b_post, s.iadd(post_pix_base, s.uint(7)), vel_v)
    s.end_if()

    var hdr_out_r = s.select_f(demod_gi, g_fin_r, s.fma(d_fin_r, g_fin_r, c_fin_r))
    var hdr_out_g = s.select_f(demod_gi, g_fin_g, s.fma(d_fin_g, g_fin_g, c_fin_g))
    var hdr_out_b = s.select_f(demod_gi, g_fin_b, s.fma(d_fin_b, g_fin_b, c_fin_b))

    # Write 8 floats per pixel in `BIND_AAA_HDR_OUT`:
    # [0..2] = HDR RGB (or demodulated GI irradiance when `demod_gi != 0`), [3] = linear_z, [4..5] = (vel_u, vel_v), [6..7] = (nx, ny)!
    s.store_float(b_hdr, out_pix_base, hdr_out_r)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(1)), hdr_out_g)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(2)), hdr_out_b)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(3)), z_lin)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(4)), vel_u)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(5)), vel_v)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(6)), nx)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(7)), ny)
    s.end_if()

    return s.finish()


def svgf_temporal_reproject_shader() -> List[UInt32]:
    """Pass 1: SVGF Temporal Reprojection & Moments Accumulation authored in Pure-Mojo ShaderDsl.
    Push[0] = w_u, Push[1] = h_u, Push[2] = hist_in_words, Push[3] = hist_out_words, Push[4] = frame_index.
    """
    from flint.render.shader_dsl import (
        GpuVec3,
        ShaderDsl,
        abs_f,
        clamp_f,
        dot,
        floor_f,
        fma_f,
        max_f,
        min_f,
        select_f,
        select_vec3,
        sqrt_f,
    )

    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)

    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var hist_in_off = dsl.push_u32(2)
    var hist_out_off = dsl.push_u32(3)
    var frame_idx = dsl.push_u32(4)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var pix_idx = py * w_u + px
    var c_base = b_hdr.base_word(pix_idx)

    var cur_rgb = b_hdr.load_vec3(c_base)
    var cur_z = b_hdr.load_f32(c_base + 3)
    var vel_u = b_hdr.load_f32(c_base + 4)
    var vel_v = b_hdr.load_f32(c_base + 5)
    var cur_nx = b_hdr.load_f32(c_base + 6)
    var cur_ny = b_hdr.load_f32(c_base + 7)
    var cur_nz = sqrt_f(max_f(dsl.f32(1.0) - fma_f(cur_nx, cur_nx, cur_ny * cur_ny), 0.0))
    _ = cur_nz

    var luma_weights = dsl.vec3_const(0.2126, 0.7152, 0.0722)
    var cur_luma = dot(cur_rgb, luma_weights)
    var cur_luma_sq = cur_luma * cur_luma

    var fw = w_u.to_f32()
    var fh = h_u.to_f32()
    var cur_u = (px.to_f32() + dsl.f32(0.5)) / fw
    var cur_v = (py.to_f32() + dsl.f32(0.5)) / fh
    var prev_u = cur_u - vel_u
    var prev_v = cur_v - vel_v

    var in_bounds = prev_u.ge(0.0) & prev_u.lt(1.0) & prev_v.ge(0.0) & prev_v.lt(1.0) & frame_idx.ne(0)

    var prev_x = clamp_f(floor_f(prev_u * fw), dsl.f32(0.0), fw - dsl.f32(1.0)).saturate_u32()
    var prev_y = clamp_f(floor_f(prev_v * fh), dsl.f32(0.0), fh - dsl.f32(1.0)).saturate_u32()
    var prev_pix = prev_y * w_u + prev_x
    var prev_base = hist_in_off + prev_pix * dsl.u32(8)

    var prev_rgb = b_hist.load_vec3(prev_base)
    var prev_len = b_hist.load_f32(prev_base + 3)
    var prev_m1 = b_hist.load_f32(prev_base + 4)
    var prev_m2 = b_hist.load_f32(prev_base + 5)
    var prev_z = b_hist.load_f32(prev_base + 6)
    var prev_nx = b_hist.load_f32(prev_base + 7)

    var depth_diff = abs_f(cur_z - prev_z) / max_f(cur_z * 0.1, 0.08)
    var norm_diff = abs_f(cur_nx - prev_nx)
    var is_consistent = in_bounds & depth_diff.lt(0.12) & norm_diff.lt(0.25) & cur_z.lt(1e5)

    var res_rgb = dsl.var_vec3(cur_rgb)
    var res_m1 = dsl.var_f32(cur_luma)
    var res_m2 = dsl.var_f32(cur_luma_sq)
    var res_len = dsl.var_f32_const(1.0)

    dsl.if_then(is_consistent)
    var new_len = min_f(prev_len + dsl.f32(1.0), 24.0)
    var alpha = dsl.f32(1.0) / new_len
    var om_a = dsl.f32(1.0) - alpha
    res_rgb.set(cur_rgb * alpha + prev_rgb * om_a)
    res_m1.set(cur_luma * alpha + prev_m1 * om_a)
    res_m2.set(cur_luma_sq * alpha + prev_m2 * om_a)
    res_len.set(new_len)
    dsl.end_if()

    var fin_rgb = res_rgb.get()
    var fin_m1 = res_m1.get()
    var fin_m2 = res_m2.get()
    var fin_len = res_len.get()

    var raw_var_l = max_f(fin_m2 - fin_m1 * fin_m1, 0.0)
    var boosted_var_l = raw_var_l * (dsl.f32(4.0) / fin_len)
    var var_l = select_f(fin_len.lt(4.0), boosted_var_l, raw_var_l)

    # Write to b_hdr for Pass 2 (À-Trous Wavelet):
    b_hdr.store_vec3(c_base, fin_rgb)
    b_hdr.store_f32(c_base + 3, cur_z)
    b_hdr.store_f32(c_base + 4, var_l)      # Slot 4 stores True Var(L)
    b_hdr.store_f32(c_base + 5, fin_len)
    b_hdr.store_f32(c_base + 6, cur_nx)
    b_hdr.store_f32(c_base + 7, cur_ny)

    # Store into persistent history for Frame N+1:
    var out_base = hist_out_off + pix_idx * dsl.u32(8)
    b_hist.store_vec3(out_base, fin_rgb)
    b_hist.store_f32(out_base + 3, fin_len)
    b_hist.store_f32(out_base + 4, fin_m1)
    b_hist.store_f32(out_base + 5, fin_m2)
    b_hist.store_f32(out_base + 6, cur_z)
    b_hist.store_f32(out_base + 7, cur_nx)

    return dsl.finish()


def svgf_atrous_shader_specialized[ENABLE_EDGE_STOP: Bool = True]() -> List[UInt32]:
    """`SVGF` Edge-Stopping `À-Trous` Wavelet Denoiser authored in the Pure-Mojo `ShaderDsl` (`V16`),
    supporting compile-time `[ENABLE_EDGE_STOP: Bool]` permutation specialization!"""
    from flint.render.shader_dsl import (
        GpuVec3,
        ShaderDsl,
        abs_f,
        dot,
        exp2_f,
        fma_f,
        max_f,
        min_u,
        pow_f,
        select_f,
        select_u,
        select_vec3,
        sqrt_f,
    )

    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_hist = dsl.buffer[8](BIND_AAA_HISTORY)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)

    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var step_u = dsl.push_u32(2)
    var from_hist = dsl.push_u32(3).ne(0)
    var sigma_luma = dsl.push_f32(4)
    var remodulate = dsl.push_u32(5).ne(0)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var pix_idx = py * w_u + px
    var c_base = b_hdr.base_word(pix_idx)
    var c_rgb = select_vec3(from_hist, b_hist.load_vec3(c_base), b_hdr.load_vec3(c_base))
    var cz = select_f(from_hist, b_hist.load_f32(c_base + 3), b_hdr.load_f32(c_base + 3))
    var cvv = select_f(from_hist, b_hist.load_f32(c_base + 5), b_hdr.load_f32(c_base + 5))
    var cnx = select_f(from_hist, b_hist.load_f32(c_base + 6), b_hdr.load_f32(c_base + 6))
    var cny = select_f(from_hist, b_hist.load_f32(c_base + 7), b_hdr.load_f32(c_base + 7))
    var cnz = sqrt_f(max_f(dsl.f32(1.0) - fma_f(cnx, cnx, cny * cny), 0.0))
    var c_nrm = GpuVec3(cnx, cny, cnz)
    var luma_weights = dsl.vec3_const(0.2126, 0.7152, 0.0722)
    var c_luma = dot(c_rgb, luma_weights)

    var sum_rgb = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var sum_var = dsl.var_f32_const(0.0)
    var sum_w = dsl.var_f32_const(0.0)

    dsl.if_then(cz.gt(1e5))
    sum_rgb.set(c_rgb)
    var c_var_init = select_f(from_hist, b_hist.load_f32(c_base + 4), b_hdr.load_f32(c_base + 4))
    sum_var.set(c_var_init)
    sum_w.set(dsl.f32(1.0))
    dsl.else_branch()
    var k_weights = [Float32(0.25), Float32(0.5), Float32(0.25)]
    for ky in range(-1, 2):
        for kx in range(-1, 2):
            var kw = dsl.f32(k_weights[ky + 1] * k_weights[kx + 1])
            var sx_u = px
            if kx < 0:
                sx_u = select_u(px.ge(step_u), px - step_u, dsl.u32(0))
            elif kx > 0:
                sx_u = min_u(w_u - 1, px + step_u)
            var sy_u = py
            if ky < 0:
                sy_u = select_u(py.ge(step_u), py - step_u, dsl.u32(0))
            elif ky > 0:
                sy_u = min_u(h_u - 1, py + step_u)

            var n_base = b_hdr.base_word(sy_u * w_u + sx_u)
            var n_rgb = select_vec3(from_hist, b_hist.load_vec3(n_base), b_hdr.load_vec3(n_base))
            var w_tot = kw
            if ENABLE_EDGE_STOP:
                var nz_l = select_f(from_hist, b_hist.load_f32(n_base + 3), b_hdr.load_f32(n_base + 3))
                var nnx = select_f(from_hist, b_hist.load_f32(n_base + 6), b_hdr.load_f32(n_base + 6))
                var nny = select_f(from_hist, b_hist.load_f32(n_base + 7), b_hdr.load_f32(n_base + 7))
                var nnz = sqrt_f(max_f(dsl.f32(1.0) - fma_f(nnx, nnx, nny * nny), 0.0))
                var n_nrm = GpuVec3(nnx, nny, nnz)
                var ndot = max_f(dot(c_nrm, n_nrm), 0.0)
                var w_n = pow_f(ndot, 32.0)
                var dz = abs_f(cz - nz_l) / max_f(cz * 0.08, 0.05)
                var c_var = select_f(from_hist, b_hist.load_f32(c_base + 4), b_hdr.load_f32(c_base + 4))
                var n_luma = dot(n_rgb, luma_weights)
                var l_diff = abs_f(c_luma - n_luma)
                var s_scaled = fma_f(sigma_luma, sqrt_f(max_f(c_var, 1e-5)), dsl.f32(0.001))
                var dl = l_diff / max_f(s_scaled, 0.01)
                var w_zl = exp2_f(-((dz + dl) * 1.442695))
                w_tot = kw * (w_n * w_zl)

            var cur_acc = sum_rgb.get()
            sum_rgb.set(
                GpuVec3(
                    fma_f(n_rgb.x(), w_tot, cur_acc.x()),
                    fma_f(n_rgb.y(), w_tot, cur_acc.y()),
                    fma_f(n_rgb.z(), w_tot, cur_acc.z()),
                )
            )
            var n_var = select_f(from_hist, b_hist.load_f32(n_base + 4), b_hdr.load_f32(n_base + 4))
            sum_var.set(fma_f(n_var, w_tot * w_tot, sum_var.get()))
            sum_w.set(sum_w.get() + w_tot)
    dsl.end_if()

    var inv_sw = Float32(1.0) / max_f(sum_w.get(), 1e-5)
    var f_rgb = sum_rgb.get() * inv_sw
    var f_var = sum_var.get() * (inv_sw * inv_sw)
    var out_rgb = dsl.var_vec3(f_rgb)
    var out_s4 = dsl.var_f32(f_var)
    var out_s5 = dsl.var_f32(cvv)

    dsl.if_then(remodulate)
    var p_base = dsl.u32(1024) + c_base
    var crisp_rgb = b_post.load_vec3(p_base)
    var dw_rgb = b_post.load_vec3(p_base + UInt32(3))
    out_rgb.set(
        GpuVec3(
            fma_f(dw_rgb.x(), f_rgb.x(), crisp_rgb.x()),
            fma_f(dw_rgb.y(), f_rgb.y(), crisp_rgb.y()),
            fma_f(dw_rgb.z(), f_rgb.z(), crisp_rgb.z()),
        )
    )
    out_s4.set(b_post.load_f32(p_base + UInt32(6)))
    out_s5.set(b_post.load_f32(p_base + UInt32(7)))
    dsl.end_if()

    var w_rgb = out_rgb.get()
    var w_s4 = out_s4.get()
    var w_s5 = out_s5.get()
    dsl.if_then(from_hist)
    b_hdr.store_vec3(c_base, w_rgb)
    b_hdr.store_f32(c_base + 3, cz)
    b_hdr.store_f32(c_base + 4, w_s4)
    b_hdr.store_f32(c_base + 5, w_s5)
    b_hdr.store_f32(c_base + 6, cnx)
    b_hdr.store_f32(c_base + 7, cny)
    dsl.else_branch()
    b_hist.store_vec3(c_base, w_rgb)
    b_hist.store_f32(c_base + 3, cz)
    b_hist.store_f32(c_base + 4, w_s4)
    b_hist.store_f32(c_base + 5, w_s5)
    b_hist.store_f32(c_base + 6, cnx)
    b_hist.store_f32(c_base + 7, cny)
    dsl.end_if()

    return dsl.finish()


def svgf_atrous_shader() -> List[UInt32]:
    return svgf_atrous_shader_specialized[True]()


def svgf_atrous_shader_raw_spv() -> List[UInt32]:
    """Reference raw-SSA `SVGF` `À-Trous` Wavelet Spatial Denoiser kept for `Suite 20` bit-exact equivalence check."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var step_u = s.push_word(2)
    var from_hist = s.ine(s.push_word(3), s.uint(0))
    var sigma_luma = s.push_float(4)
    var remodulate = s.ine(s.push_word(5), s.uint(0))
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var c_base = s.imul(s.iadd(s.imul(py, w_u), px), s.uint(8))
    var cr = s.select_f(from_hist, s.load_float(b_hist, c_base), s.load_float(b_hdr, c_base))
    var cg = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(1))), s.load_float(b_hdr, s.iadd(c_base, s.uint(1))))
    var cb = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(2))), s.load_float(b_hdr, s.iadd(c_base, s.uint(2))))
    var cz = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(3))), s.load_float(b_hdr, s.iadd(c_base, s.uint(3))))
    var cvv = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(5))), s.load_float(b_hdr, s.iadd(c_base, s.uint(5))))
    var cnx = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(6))), s.load_float(b_hdr, s.iadd(c_base, s.uint(6))))
    var cny = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(7))), s.load_float(b_hdr, s.iadd(c_base, s.uint(7))))
    var cnz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(cnx, cnx, s.fmul(cny, cny)))))
    var c_luma = s.fma(cr, s.float(0.2126), s.fma(cg, s.float(0.7152), s.fmul(cb, s.float(0.0722))))

    var sum_r = s.fvar(s.float(0.0))
    var sum_g = s.fvar(s.float(0.0))
    var sum_b = s.fvar(s.float(0.0))
    var sum_var = s.fvar(s.float(0.0))
    var sum_w = s.fvar(s.float(0.0))

    s.begin_if(s.fgt(cz, s.float(1e5)))
    var c_var_init = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(4))), s.load_float(b_hdr, s.iadd(c_base, s.uint(4))))
    s.store(sum_r, cr); s.store(sum_g, cg); s.store(sum_b, cb); s.store(sum_var, c_var_init); s.store(sum_w, s.float(1.0))
    s.begin_else()
    var k_weights = [Float32(0.25), Float32(0.5), Float32(0.25)]
    for ky in range(-1, 2):
        for kx in range(-1, 2):
            var kw = s.float(k_weights[ky + 1] * k_weights[kx + 1])
            var sx_u = px
            if kx < 0:
                sx_u = s.select_u(s.uge(px, step_u), s.isub(px, step_u), s.uint(0))
            elif kx > 0:
                sx_u = s.umin(s.isub(w_u, s.uint(1)), s.iadd(px, step_u))
            var sy_u = py
            if ky < 0:
                sy_u = s.select_u(s.uge(py, step_u), s.isub(py, step_u), s.uint(0))
            elif ky > 0:
                sy_u = s.umin(s.isub(h_u, s.uint(1)), s.iadd(py, step_u))

            var n_base = s.imul(s.iadd(s.imul(sy_u, w_u), sx_u), s.uint(8))
            var nr = s.select_f(from_hist, s.load_float(b_hist, n_base), s.load_float(b_hdr, n_base))
            var ng = s.select_f(from_hist, s.load_float(b_hist, s.iadd(n_base, s.uint(1))), s.load_float(b_hdr, s.iadd(n_base, s.uint(1))))
            var nb = s.select_f(from_hist, s.load_float(b_hist, s.iadd(n_base, s.uint(2))), s.load_float(b_hdr, s.iadd(n_base, s.uint(2))))
            var nz_l = s.select_f(from_hist, s.load_float(b_hist, s.iadd(n_base, s.uint(3))), s.load_float(b_hdr, s.iadd(n_base, s.uint(3))))
            var nnx = s.select_f(from_hist, s.load_float(b_hist, s.iadd(n_base, s.uint(6))), s.load_float(b_hdr, s.iadd(n_base, s.uint(6))))
            var nny = s.select_f(from_hist, s.load_float(b_hist, s.iadd(n_base, s.uint(7))), s.load_float(b_hdr, s.iadd(n_base, s.uint(7))))
            var nnz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(nnx, nnx, s.fmul(nny, nny)))))

            var ndot = s.nmax(s.float(0.0), emit_vec3_dot(s, cnx, cny, cnz, nnx, nny, nnz))
            var w_n = s.fpow(ndot, s.float(32.0))
            var dz = s.fdiv(s.fabs(s.fsub(cz, nz_l)), s.nmax(s.float(0.05), s.fmul(cz, s.float(0.08))))
            var c_var = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(4))), s.load_float(b_hdr, s.iadd(c_base, s.uint(4))))
            var n_luma = s.fma(nr, s.float(0.2126), s.fma(ng, s.float(0.7152), s.fmul(nb, s.float(0.0722))))
            var l_diff = s.fabs(s.fsub(c_luma, n_luma))
            var s_scaled = s.fma(sigma_luma, s.fsqrt(s.nmax(s.float(1e-5), c_var)), s.float(0.001))
            var dl = s.fdiv(l_diff, s.nmax(s.float(0.01), s_scaled))
            var w_zl = s.exp2(s.fneg(s.fmul(s.fadd(dz, dl), s.float(1.442695))))
            var w_tot = s.fmul(kw, s.fmul(w_n, w_zl))

            s.store(sum_r, s.fma(nr, w_tot, s.load_f(sum_r)))
            s.store(sum_g, s.fma(ng, w_tot, s.load_f(sum_g)))
            s.store(sum_b, s.fma(nb, w_tot, s.load_f(sum_b)))
            var n_var = s.select_f(from_hist, s.load_float(b_hist, s.iadd(n_base, s.uint(4))), s.load_float(b_hdr, s.iadd(n_base, s.uint(4))))
            s.store(sum_var, s.fma(n_var, s.fmul(w_tot, w_tot), s.load_f(sum_var)))
            s.store(sum_w, s.fadd(s.load_f(sum_w), w_tot))
    s.end_if()

    var inv_sw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.load_f(sum_w)))
    var fr = s.fmul(s.load_f(sum_r), inv_sw)
    var fg = s.fmul(s.load_f(sum_g), inv_sw)
    var fb = s.fmul(s.load_f(sum_b), inv_sw)
    var f_var = s.fmul(s.load_f(sum_var), s.fmul(inv_sw, inv_sw))
    var out_r = s.fvar(fr)
    var out_g = s.fvar(fg)
    var out_b = s.fvar(fb)

    s.begin_if(remodulate)
    var p_base = s.iadd(s.uint(1024), c_base)
    var cr_p = s.load_float(b_post, p_base)
    var cg_p = s.load_float(b_post, s.iadd(p_base, s.uint(1)))
    var cb_p = s.load_float(b_post, s.iadd(p_base, s.uint(2)))
    var dw_r = s.load_float(b_post, s.iadd(p_base, s.uint(3)))
    var dw_g = s.load_float(b_post, s.iadd(p_base, s.uint(4)))
    var dw_b = s.load_float(b_post, s.iadd(p_base, s.uint(5)))
    s.store(out_r, s.fma(dw_r, fr, cr_p))
    s.store(out_g, s.fma(dw_g, fg, cg_p))
    s.store(out_b, s.fma(dw_b, fb, cb_p))
    s.end_if()

    var wr = s.load_f(out_r)
    var wg = s.load_f(out_g)
    var wb = s.load_f(out_b)

    s.begin_if(from_hist)
    s.store_float(b_hdr, c_base, wr)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(1)), wg)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(2)), wb)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(3)), cz)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(4)), f_var)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(5)), cvv)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(6)), cnx)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(7)), cny)
    s.begin_else()
    s.store_float(b_hist, c_base, wr)
    s.store_float(b_hist, s.iadd(c_base, s.uint(1)), wg)
    s.store_float(b_hist, s.iadd(c_base, s.uint(2)), wb)
    s.store_float(b_hist, s.iadd(c_base, s.uint(3)), cz)
    s.store_float(b_hist, s.iadd(c_base, s.uint(4)), f_var)
    s.store_float(b_hist, s.iadd(c_base, s.uint(5)), cvv)
    s.store_float(b_hist, s.iadd(c_base, s.uint(6)), cnx)
    s.store_float(b_hist, s.iadd(c_base, s.uint(7)), cny)
    s.end_if()

    return s.finish()
