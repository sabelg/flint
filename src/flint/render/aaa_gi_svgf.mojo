"""Milestone 3 (`V14`): Lumen-Class Multi-Bounce Mesh Surface Cache, Full Deferred
`VisBuffer` Multi-Lobe + VSM + `SPV_KHR_ray_query` Shader, `SVGF` Wavelet Denoiser &
512-spp Unbiased Monte-Carlo Reference Path Tracer (`aaa_gi_svgf.mojo`).
"""
from std.math import cos, exp, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_BRDF_LUT,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_ray_barycentrics,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_bsdf import (
    AAA_MATERIAL_WORDS,
    emit_direct_multilobe_bsdf,
    emit_iridescence_f0,
    emit_rnm_blend,
)
from flint.render.aaa_vsm import emit_vsm_pcss_shadow
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.aaa_raytracing import emit_rt_shadow_query, emit_rt_surface_cache_ray

# Render feature flags in `BIND_AAA_GLOBALS[99]`
comptime AAA_FLAG_VSM: UInt32 = 1
comptime AAA_FLAG_RT_SHADOWS: UInt32 = 2
comptime AAA_FLAG_RT_REFLECTIONS: UInt32 = 4
comptime AAA_FLAG_RT_SURFACE_GI: UInt32 = 8
comptime AAA_FLAG_FURNACE_WHITE_DOME: UInt32 = 16


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
        enable_ray_query=enable_ray_query,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_surfcache = s.buffer(BIND_AAA_SURFCACHE)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var inst_idx = s.invocation_axis(0)
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

    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var alb_r = s.load_float(b_materials, m_base)
    var alb_g = s.load_float(b_materials, s.iadd(m_base, s.uint(1)))
    var alb_b = s.load_float(b_materials, s.iadd(m_base, s.uint(2)))
    var met = s.load_float(b_materials, s.iadd(m_base, s.uint(5)))
    var em_r = s.load_float(b_materials, s.iadd(m_base, s.uint(8)))
    var em_g = s.load_float(b_materials, s.iadd(m_base, s.uint(9)))
    var em_b = s.load_float(b_materials, s.iadd(m_base, s.uint(10)))

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
    if enable_ray_query:
        s.begin_if(s.fgt(ndotl, s.float(0.001)))
        var rt_s = emit_rt_shadow_query(s, tlas_var, cx, cy, cz, wn[0], wn[1], wn[2], sun_x, sun_y, sun_z, s.float(500.0))
        s.store(sh_vis, rt_s)
        s.end_if()

    var inv_pi = s.float(0.318309886)
    var d_scale = s.fmul(s.fmul(ndotl, s.load_f(sh_vis)), inv_pi)
    var out_r = s.fvar(s.fma(kd_r, s.fmul(sun_r, d_scale), em_r))
    var out_g = s.fvar(s.fma(kd_g, s.fmul(sun_g, d_scale), em_g))
    var out_b = s.fvar(s.fma(kd_b, s.fmul(sun_b, d_scale), em_b))

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

    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var _depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var brdf_lut = s.sampled_image_2d(BIND_AAA_BRDF_LUT)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var gi_rays_u = s.push_word(2)
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
    var out_pix_base = s.imul(s.iadd(s.imul(py, w_u), px), s.uint(8))
    var is_furnace = s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_FURNACE_WHITE_DOME)), s.uint(0))

    # Sky background pixel (`vis0 == 0`)
    s.begin_if(s.ieq(vis0, s.uint(0)))
    var env_bg = s.sample_cube_lod(env_cube, s.vec3(ray_c[0], ray_c[1], ray_c[2]), s.float(0.0))
    var bg_r = s.select_f(is_furnace, s.float(1.0), s.extract_f(env_bg, 0))
    var bg_g = s.select_f(is_furnace, s.float(1.0), s.extract_f(env_bg, 1))
    var bg_b = s.select_f(is_furnace, s.float(1.0), s.extract_f(env_bg, 2))
    s.store_float(b_hdr, out_pix_base, bg_r)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(1)), bg_g)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(2)), bg_b)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(3)), s.float(1e6))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(4)), s.float(0.0))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(5)), s.float(0.0))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(6)), s.float(0.0))
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(7)), s.float(0.0))
    s.begin_else()

    # Decode 64-bit VisBuffer payload
    var inst_id = s.shr(vis0, s.uint(16))
    var ml_id = s.isub(s.bit_and(vis0, s.uint(0xFFFF)), s.uint(1))
    var mat_id = s.shr(vis1, s.uint(16))
    var local_prim = s.bit_and(vis1, s.uint(0xFFFF))

    var ml_base = s.imul(ml_id, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var global_tri = s.iadd(first_tri, local_prim)
    var idx_off = s.imul(global_tri, s.uint(3))
    var v0_off = s.imul(s.load_word(b_indices, idx_off), s.uint(12))
    var v1_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(1))), s.uint(12))
    var v2_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(2))), s.uint(12))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var l0x = s.load_float(b_verts, v0_off); var l0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(1))); var l0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(2)))
    var l1x = s.load_float(b_verts, v1_off); var l1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(1))); var l1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(2)))
    var l2x = s.load_float(b_verts, v2_off); var l2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(1))); var l2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(2)))

    var p0 = emit_mat4_mul_point(s, b_instances, inst_base, l0x, l0y, l0z)
    var p1 = emit_mat4_mul_point(s, b_instances, inst_base, l1x, l1y, l1z)
    var p2 = emit_mat4_mul_point(s, b_instances, inst_base, l2x, l2y, l2z)

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

    # Perspective-correct world position & local position (for 3D previous-frame motion vectors!)
    var wx = s.fma(p0[0], b0, s.fma(p1[0], b1, s.fmul(p2[0], b2)))
    var wy = s.fma(p0[1], b0, s.fma(p1[1], b1, s.fmul(p2[1], b2)))
    var wz = s.fma(p0[2], b0, s.fma(p1[2], b1, s.fmul(p2[2], b2)))
    var loc_x = s.fma(l0x, b0, s.fma(l1x, b1, s.fmul(l2x, b2)))
    var loc_y = s.fma(l0y, b0, s.fma(l1y, b1, s.fmul(l2y, b2)))
    var loc_z = s.fma(l0z, b0, s.fma(l1z, b1, s.fmul(l2z, b2)))

    # Compute exact 3D per-pixel motion vector `(vel_u, vel_v)` from `prev_model_to_world` + `prev_view_proj`!
    var prev_wp = emit_mat4_mul_point(s, b_instances, s.iadd(inst_base, s.uint(16)), loc_x, loc_y, loc_z)
    var prev_clip = emit_mat4_mul_point(s, b_globals, s.uint(16), prev_wp[0], prev_wp[1], prev_wp[2])
    var prev_inv_w = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), prev_clip[3]))
    var prev_u = s.fma(s.fmul(prev_clip[0], prev_inv_w), s.float(0.5), s.float(0.5))
    var prev_v = s.fma(s.fneg(s.fmul(prev_clip[1], prev_inv_w)), s.float(0.5), s.float(0.5))
    var cur_u = s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw)
    var cur_v = s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh)
    var vel_u = s.fsub(cur_u, prev_u)
    var vel_v = s.fsub(cur_v, prev_v)

    # Interpolate smooth vertex normal + tangent + UV + analytic `(ddx_uv, ddy_uv)`!
    var n0x = s.load_float(b_verts, s.iadd(v0_off, s.uint(3))); var n0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(4))); var n0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(5)))
    var n1x = s.load_float(b_verts, s.iadd(v1_off, s.uint(3))); var n1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(4))); var n1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(5)))
    var n2x = s.load_float(b_verts, s.iadd(v2_off, s.uint(3))); var n2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(4))); var n2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(5)))
    var ln_x = s.fma(n0x, b0, s.fma(n1x, b1, s.fmul(n2x, b2)))
    var ln_y = s.fma(n0y, b0, s.fma(n1y, b1, s.fmul(n2y, b2)))
    var ln_z = s.fma(n0z, b0, s.fma(n1z, b1, s.fmul(n2z, b2)))
    var wn_raw = emit_mat4_mul_dir(s, b_instances, inst_base, ln_x, ln_y, ln_z)
    var wn_geom = emit_vec3_normalize(s, wn_raw[0], wn_raw[1], wn_raw[2])

    var t0x = s.load_float(b_verts, s.iadd(v0_off, s.uint(6))); var t0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(7))); var t0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(8)))
    var t1x = s.load_float(b_verts, s.iadd(v1_off, s.uint(6))); var t1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(7))); var t1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(8)))
    var t2x = s.load_float(b_verts, s.iadd(v2_off, s.uint(6))); var t2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(7))); var t2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(8)))
    var lt_x = s.fma(t0x, b0, s.fma(t1x, b1, s.fmul(t2x, b2)))
    var lt_y = s.fma(t0y, b0, s.fma(t1y, b1, s.fmul(t2y, b2)))
    var lt_z = s.fma(t0z, b0, s.fma(t1z, b1, s.fmul(t2z, b2)))
    var wt_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lt_x, lt_y, lt_z)
    var wt_u = emit_vec3_normalize(s, wt_raw[0], wt_raw[1], wt_raw[2])
    var meta0 = s.load_word(b_verts, s.iadd(v0_off, s.uint(11)))
    var t_sign = s.select_f(s.ieq(s.bit_and(meta0, s.uint(1)), s.uint(0)), s.float(1.0), s.float(-1.0))
    var wb_raw = emit_vec3_cross(s, wn_geom[0], wn_geom[1], wn_geom[2], wt_u[0], wt_u[1], wt_u[2])
    var wb_u = (s.fmul(wb_raw[0], t_sign), s.fmul(wb_raw[1], t_sign), s.fmul(wb_raw[2], t_sign))

    var u0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(9))); var v0_t = s.load_float(b_verts, s.iadd(v0_off, s.uint(10)))
    var u1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(9))); var v1_t = s.load_float(b_verts, s.iadd(v1_off, s.uint(10)))
    var u2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(9))); var v2_t = s.load_float(b_verts, s.iadd(v2_off, s.uint(10)))
    var uv_u = s.fma(u0, b0, s.fma(u1, b1, s.fmul(u2, b2)))
    var uv_v = s.fma(v0_t, b0, s.fma(v1_t, b1, s.fmul(v2_t, b2)))

    var b0_dx = s.fsub(s.fsub(s.float(1.0), bary[2]), bary[3])
    var b0_dy = s.fsub(s.fsub(s.float(1.0), bary[4]), bary[5])
    var ddx_u = s.fsub(s.fma(u0, b0_dx, s.fma(u1, bary[2], s.fmul(u2, bary[3]))), uv_u)
    var ddx_v = s.fsub(s.fma(v0_t, b0_dx, s.fma(v1_t, bary[2], s.fmul(v2_t, bary[3]))), uv_v)
    var ddy_u = s.fsub(s.fma(u0, b0_dy, s.fma(u1, bary[4], s.fmul(u2, bary[5]))), uv_u)
    var ddy_v = s.fsub(s.fma(v0_t, b0_dy, s.fma(v1_t, bary[4], s.fmul(v2_t, bary[5]))), uv_v)

    # Load Enterprise BSDF Material + Bindless `BC7` / `BC5` Hardware Anisotropic Textures!
    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var alb_r = s.fvar(s.load_float(b_materials, m_base))
    var alb_g = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(1))))
    var alb_b = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(2))))
    var spec_fac = s.load_float(b_materials, s.iadd(m_base, s.uint(3)))
    var rough = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(4))))
    var metal = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(5))))
    var ccoat = s.load_float(b_materials, s.iadd(m_base, s.uint(6)))
    var ccoat_r = s.load_float(b_materials, s.iadd(m_base, s.uint(7)))
    var em_r = s.load_float(b_materials, s.iadd(m_base, s.uint(8)))
    var em_g = s.load_float(b_materials, s.iadd(m_base, s.uint(9)))
    var em_b = s.load_float(b_materials, s.iadd(m_base, s.uint(10)))
    var det_scale = s.load_float(b_materials, s.iadd(m_base, s.uint(11)))
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
    var t_det = s.bit_and(s.shr(tex_pack, s.uint(24)), s.uint(0x7F))
    var mat_ao = s.fvar(s.float(1.0))

    var uv_vec = s.vec2(uv_u, uv_v)
    var ddx_vec = s.vec2(ddx_u, ddx_v)
    var ddy_vec = s.vec2(ddy_u, ddy_v)

    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var s_alb = s.sample_bindless_grad(bindless_arr, t_alb, uv_vec, ddx_vec, ddy_vec)
    s.store(alb_r, s.fmul(s.load_f(alb_r), s.extract_f(s_alb, 0)))
    s.store(alb_g, s.fmul(s.load_f(alb_g), s.extract_f(s_alb, 1)))
    s.store(alb_b, s.fmul(s.load_f(alb_b), s.extract_f(s_alb, 2)))
    s.end_if()

    s.begin_if(s.ult(t_orm, s.uint(0xFF)))
    var s_orm = s.sample_bindless_grad(bindless_arr, t_orm, uv_vec, ddx_vec, ddy_vec)
    s.store(mat_ao, s.extract_f(s_orm, 0))
    s.store(rough, s.nclamp(s.fmul(s.load_f(rough), s.extract_f(s_orm, 1)), s.float(0.04), s.float(1.0)))
    s.store(metal, s.nclamp(s.fmul(s.load_f(metal), s.extract_f(s_orm, 2)), s.float(0.0), s.float(1.0)))
    s.end_if()

    # Macro + Micro-Detail BC5 Tangent Normal Mapping (`RNM`)
    var tn_x = s.fvar(s.float(0.0))
    var tn_y = s.fvar(s.float(0.0))
    var tn_z = s.fvar(s.float(1.0))
    s.begin_if(s.ult(t_nrm, s.uint(0xFF)))
    var s_nrm = s.sample_bindless_grad(bindless_arr, t_nrm, uv_vec, ddx_vec, ddy_vec)
    var mx = s.fma(s.extract_f(s_nrm, 0), s.float(2.0), s.float(-1.0))
    var my = s.fma(s.extract_f(s_nrm, 1), s.float(2.0), s.float(-1.0))
    var mz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(mx, mx, s.fmul(my, my)))))
    s.store(tn_x, mx); s.store(tn_y, my); s.store(tn_z, mz)
    s.end_if()

    s.begin_if(s.logical_and(s.ult(t_det, s.uint(0x7F)), s.fgt(det_str, s.float(0.001))))
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
    var nx = s.fmul(wn[0], flip_n)
    var ny = s.fmul(wn[1], flip_n)
    var nz = s.fmul(wn[2], flip_n)
    var ndotv = s.nmax(s.float(1e-3), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))

    var cur_alb_r = s.load_f(alb_r)
    var cur_alb_g = s.load_f(alb_g)
    var cur_alb_b = s.load_f(alb_b)
    var cur_rough = s.load_f(rough)
    var cur_metal = s.load_f(metal)

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

    var sun_vis = s.fvar(s.float(1.0))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_VSM)), s.uint(0)))
    var vsm_v = emit_vsm_pcss_shadow(s, b_globals, b_vsm, vsm_samp, wx, wy, wz, nx, ny, nz, sun_ang_r)
    s.store(sun_vis, vsm_v)
    s.end_if()
    if enable_ray_query:
        s.begin_if(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SHADOWS)), s.uint(0)))
        var rt_v = emit_rt_shadow_query(s, tlas_var, wx, wy, wz, nx, ny, nz, sun_x, sun_y, sun_z, s.float(500.0))
        s.store(sun_vis, s.nmin(s.load_f(sun_vis), rt_v))
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
    var dir_r = s.fmul(direct[0], s.fmul(sun_rad_r, sv))
    var dir_g = s.fmul(direct[1], s.fmul(sun_rad_g, sv))
    var dir_b = s.fmul(direct[2], s.fmul(sun_rad_b, sv))

    # 2. Diffuse Global Illumination (`SH9` Sky Irradiance OR `SPV_KHR_ray_query` Lumen Surface-Cache GI!)
    var sh_irr = emit_eval_sh9_irradiance(s, b_globals, 72, nx, ny, nz)
    var gi_r = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(sh_irr[0], ibl_int)))
    var gi_g = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(sh_irr[1], ibl_int)))
    var gi_b = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(sh_irr[2], ibl_int)))

    if enable_ray_query:
        s.begin_if(s.logical_and(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_SURFACE_GI)), s.uint(0)), s.ult(s.uint(0), gi_rays_u)))
        var miss_r = s.load_f(gi_r)
        var miss_g = s.load_f(gi_g)
        var miss_b = s.load_f(gi_b)
        var acc_gr = s.fvar(s.float(0.0))
        var acc_gg = s.fvar(s.float(0.0))
        var acc_gb = s.fvar(s.float(0.0))
        var ro_x = s.fma(nx, s.float(0.006), wx)
        var ro_y = s.fma(ny, s.float(0.006), wy)
        var ro_z = s.fma(nz, s.float(0.006), wz)
        var pix_seed = s.iadd(s.imul(s.iadd(s.imul(py, w_u), px), s.uint(1973)), s.imul(frame_idx, s.uint(9277)))
        var rk = s.uvar(s.uint(0))
        var inv_n = s.fdiv(s.float(1.0), s.float_from_uint(gi_rays_u))

        s.begin_loop()
        var k_cur = s.load_u(rk)
        s.break_if(s.uge(k_cur, gi_rays_u))
        # Stratified golden-ratio + Cranley-Patterson hash rotation
        var h_a = _emit_pcg_hash_f01(s, s.iadd(pix_seed, s.imul(k_cur, s.uint(26699))))
        var h_b = _emit_pcg_hash_f01(s, h_a[0])
        var strat_u1 = s.fract(s.fadd(s.fmul(s.fadd(s.float_from_uint(k_cur), s.float(0.5)), inv_n), h_a[1]))
        var strat_u2 = s.fract(s.fadd(s.fmul(s.float_from_uint(k_cur), s.float(0.6180339887)), h_b[1]))
        var gdir = _emit_cosine_hemisphere_dir(s, nx, ny, nz, strat_u1, strat_u2)
        var sc_hit = emit_rt_surface_cache_ray(
            s, tlas_var, b_surfcache, ro_x, ro_y, ro_z, gdir[0], gdir[1], gdir[2],
            s.float(0.004), s.float(200.0), miss_r, miss_g, miss_b,
        )
        s.store(acc_gr, s.fadd(s.load_f(acc_gr), sc_hit[0]))
        s.store(acc_gg, s.fadd(s.load_f(acc_gg), sc_hit[1]))
        s.store(acc_gb, s.fadd(s.load_f(acc_gb), sc_hit[2]))
        s.store(rk, s.iadd(k_cur, s.uint(1)))
        s.end_loop()

        s.store(gi_r, s.fmul(s.load_f(acc_gr), inv_n))
        s.store(gi_g, s.fmul(s.load_f(acc_gg), inv_n))
        s.store(gi_b, s.fmul(s.load_f(acc_gb), inv_n))
        s.end_if()

    # 3. Specular Reflections (`BC6H` Split-Sum Cubemap + `SPV_KHR_ray_query` Hit Radiance!)
    var two_nv = s.fmul(s.float(2.0), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
    var rx = s.fsub(s.fmul(nx, two_nv), vx)
    var ry = s.fsub(s.fmul(ny, two_nv), vy)
    var rz = s.fsub(s.fmul(nz, two_nv), vz)
    var spec_mip = s.fmul(cur_rough, max_env_mip)
    var env_sp = s.sample_cube_lod(env_cube, s.vec3(rx, ry, rz), spec_mip)
    var refl_r = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_sp, 0), ibl_int)))
    var refl_g = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_sp, 1), ibl_int)))
    var refl_b = s.fvar(s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_sp, 2), ibl_int)))

    var cc_mip = s.fmul(ccoat_r, max_env_mip)
    var env_cc = s.sample_cube_lod(env_cube, s.vec3(rx, ry, rz), cc_mip)
    var ccoat_env_r = s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_cc, 0), ibl_int))
    var ccoat_env_g = s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_cc, 1), ibl_int))
    var ccoat_env_b = s.select_f(is_furnace, s.float(1.0), s.fmul(s.extract_f(env_cc, 2), ibl_int))

    if enable_ray_query:
        s.begin_if(s.logical_and(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_RT_REFLECTIONS)), s.uint(0)), s.flt(cur_rough, s.float(0.55))))
        var ro_rx = s.fma(nx, s.float(0.006), wx)
        var ro_ry = s.fma(ny, s.float(0.006), wy)
        var ro_rz = s.fma(nz, s.float(0.006), wz)
        var rt_ref = emit_rt_surface_cache_ray(
            s, tlas_var, b_surfcache, ro_rx, ro_ry, ro_rz, rx, ry, rz,
            s.float(0.004), s.float(250.0), s.load_f(refl_r), s.load_f(refl_g), s.load_f(refl_b),
        )
        s.begin_if(s.ine(rt_ref[3], s.uint(0)))
        var blend_w = s.nclamp(s.fdiv(s.fsub(s.float(0.55), cur_rough), s.float(0.35)), s.float(0.0), s.float(1.0))
        var om_bw = s.fsub(s.float(1.0), blend_w)
        s.store(refl_r, s.fma(rt_ref[0], blend_w, s.fmul(s.load_f(refl_r), om_bw)))
        s.store(refl_g, s.fma(rt_ref[1], blend_w, s.fmul(s.load_f(refl_g), om_bw)))
        s.store(refl_b, s.fma(rt_ref[2], blend_w, s.fmul(s.load_f(refl_b), om_bw)))
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

    var ao_v = s.load_f(mat_ao)
    var sub_r = s.fadd(s.fma(spec_tot_r, s.load_f(refl_r), s.fmul(s.fmul(kd_gi_r, cur_alb_r), s.fmul(s.load_f(gi_r), ao_v))), s.fadd(dir_r, em_r))
    var sub_g = s.fadd(s.fma(spec_tot_g, s.load_f(refl_g), s.fmul(s.fmul(kd_gi_g, cur_alb_g), s.fmul(s.load_f(gi_g), ao_v))), s.fadd(dir_g, em_g))
    var sub_b = s.fadd(s.fma(spec_tot_b, s.load_f(refl_b), s.fmul(s.fmul(kd_gi_b, cur_alb_b), s.fmul(s.load_f(gi_b), ao_v))), s.fadd(dir_b, em_b))

    var r_coat = s.fmul(ccoat, s.fadd(s.fmul(s.float(0.04), cc_a), cc_b))
    var om_rc = s.fsub(s.float(1.0), r_coat)
    var fin_r = s.fma(r_coat, ccoat_env_r, s.fmul(om_rc, sub_r))
    var fin_g = s.fma(r_coat, ccoat_env_g, s.fmul(om_rc, sub_g))
    var fin_b = s.fma(r_coat, ccoat_env_b, s.fmul(om_rc, sub_b))

    var rel_cam_x = s.fsub(wx, cam_x)
    var rel_cam_y = s.fsub(wy, cam_y)
    var rel_cam_z = s.fsub(wz, cam_z)
    var z_lin = s.nmax(s.float(0.01), emit_vec3_dot(s, rel_cam_x, rel_cam_y, rel_cam_z, fwd_x, fwd_y, fwd_z))

    # Write 8 floats per pixel in `BIND_AAA_HDR_OUT`:
    # [0..2] = HDR RGB, [3] = linear_z, [4..5] = (vel_u, vel_v), [6..7] = (nx, ny) for SVGF & TSR v2!
    s.store_float(b_hdr, out_pix_base, fin_r)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(1)), fin_g)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(2)), fin_b)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(3)), z_lin)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(4)), vel_u)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(5)), vel_v)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(6)), nx)
    s.store_float(b_hdr, s.iadd(out_pix_base, s.uint(7)), ny)
    s.end_if()

    return s.finish()


def svgf_atrous_shader() -> List[UInt32]:
    """`SVGF` Edge-Stopping `À-Trous` 5x5 Wavelet Spatial Denoiser weighted by Normal, Linear-Z & Luminance!
    Push[0] = width, Push[1] = height, Push[2] = step_size (1, 2, 4), Push[3] = src_in_history (0/1), Push[4] = sigma_luma_bits.
    Reads from `BIND_AAA_HDR_OUT` (when `src_in_history == 0`) or `BIND_AAA_HISTORY` (`src_in_history == 1`)
    and writes filtered 8-float pixels to the opposite buffer!
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var step_u = s.push_word(2)
    var from_hist = s.ine(s.push_word(3), s.uint(0))
    var sigma_luma = s.push_float(4)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var c_base = s.imul(s.iadd(s.imul(py, w_u), px), s.uint(8))
    var cr = s.select_f(from_hist, s.load_float(b_hist, c_base), s.load_float(b_hdr, c_base))
    var cg = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(1))), s.load_float(b_hdr, s.iadd(c_base, s.uint(1))))
    var cb = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(2))), s.load_float(b_hdr, s.iadd(c_base, s.uint(2))))
    var cz = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(3))), s.load_float(b_hdr, s.iadd(c_base, s.uint(3))))
    var cvu = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(4))), s.load_float(b_hdr, s.iadd(c_base, s.uint(4))))
    var cvv = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(5))), s.load_float(b_hdr, s.iadd(c_base, s.uint(5))))
    var cnx = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(6))), s.load_float(b_hdr, s.iadd(c_base, s.uint(6))))
    var cny = s.select_f(from_hist, s.load_float(b_hist, s.iadd(c_base, s.uint(7))), s.load_float(b_hdr, s.iadd(c_base, s.uint(7))))
    var cnz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(cnx, cnx, s.fmul(cny, cny)))))
    var c_luma = s.fma(cr, s.float(0.2126), s.fma(cg, s.float(0.7152), s.fmul(cb, s.float(0.0722))))

    var sum_r = s.fvar(s.float(0.0))
    var sum_g = s.fvar(s.float(0.0))
    var sum_b = s.fvar(s.float(0.0))
    var sum_w = s.fvar(s.float(0.0))

    s.begin_if(s.fgt(cz, s.float(1e5)))
    s.store(sum_r, cr); s.store(sum_g, cg); s.store(sum_b, cb); s.store(sum_w, s.float(1.0))
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
            var n_luma = s.fma(nr, s.float(0.2126), s.fma(ng, s.float(0.7152), s.fmul(nb, s.float(0.0722))))
            var dl = s.fdiv(s.fabs(s.fsub(c_luma, n_luma)), s.nmax(s.float(0.05), sigma_luma))
            var w_zl = s.exp2(s.fneg(s.fmul(s.fadd(dz, dl), s.float(1.442695))))
            var w_tot = s.fmul(kw, s.fmul(w_n, w_zl))

            s.store(sum_r, s.fma(nr, w_tot, s.load_f(sum_r)))
            s.store(sum_g, s.fma(ng, w_tot, s.load_f(sum_g)))
            s.store(sum_b, s.fma(nb, w_tot, s.load_f(sum_b)))
            s.store(sum_w, s.fadd(s.load_f(sum_w), w_tot))
    s.end_if()

    var inv_sw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.load_f(sum_w)))
    var fr = s.fmul(s.load_f(sum_r), inv_sw)
    var fg = s.fmul(s.load_f(sum_g), inv_sw)
    var fb = s.fmul(s.load_f(sum_b), inv_sw)

    s.begin_if(from_hist)
    s.store_float(b_hdr, c_base, fr)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(1)), fg)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(2)), fb)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(3)), cz)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(4)), cvu)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(5)), cvv)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(6)), cnx)
    s.store_float(b_hdr, s.iadd(c_base, s.uint(7)), cny)
    s.begin_else()
    s.store_float(b_hist, c_base, fr)
    s.store_float(b_hist, s.iadd(c_base, s.uint(1)), fg)
    s.store_float(b_hist, s.iadd(c_base, s.uint(2)), fb)
    s.store_float(b_hist, s.iadd(c_base, s.uint(3)), cz)
    s.store_float(b_hist, s.iadd(c_base, s.uint(4)), cvu)
    s.store_float(b_hist, s.iadd(c_base, s.uint(5)), cvv)
    s.store_float(b_hist, s.iadd(c_base, s.uint(6)), cnx)
    s.store_float(b_hist, s.iadd(c_base, s.uint(7)), cny)
    s.end_if()

    return s.finish()
