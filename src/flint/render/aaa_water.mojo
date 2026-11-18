"""Commercial Engine v2 (Pillar III): Dielectric `Beer-Lambert` Refracting, Caustic &
Foaming Water Compute Pass (`aaa_water.mojo`).

Replaces opaque `metallic = 0.84` liquid-mercury water meshes with a physically based
post-`SVGF`/`ReBLUR` dielectric water pass (`IOR = 1.333`, `F0 = 0.020`):
  - 4-wave Gerstner swell + dual counter-scrolling capillary normals + raindrop ripple rings
  - Silhouette-guarded Snell screen-space refraction (`z_tap > z_water` prevents foreground bleeding!)
  - **Beer-Lambert Optical Depth Absorption** `exp(-sigma_water * d_opt)` (cyan-emerald shallow -> deep navy)
  - Animated Voronoi cellular shallow-water sun caustics on the sculpted lakebed (`shadowed by VSM`)
  - Hardware `SPV_KHR_ray_query` textured reflection (`emit_rt_textured_hit_shade`) + GGX sun glints
  - Breaking shoreline & bridge-piling lace foam (`d_perp < 0.35m`), writing TSR reactive mask to `b_oit`!
"""
from std.math import cos, exp, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_CLUSTERS,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_OIT_NODES,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_point,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_vsm import emit_sample_vsm_depth
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.aaa_raytracing import emit_rt_textured_hit_shade
from flint.render.aaa_oit_volumetrics import AAA_OIT_REACTIVE_BASE_WORDS


@always_inline
def eval_dielectric_water_fresnel_cpu(ndotv: Float32) -> Float32:
    """Exact Schlick dielectric Fresnel for pure water (`IOR = 1.333 -> F0 = ((1.333-1)/(1.333+1))^2 = 0.020`)."""
    var c = max(Float32(0.0), min(Float32(1.0), ndotv))
    var om = Float32(1.0) - c
    var om2 = om * om
    return Float32(0.020) + Float32(0.980) * (om2 * om2 * om)


def eval_beer_lambert_water_cpu(
    bed_rgb: Vec3,
    absorption_rgb: Vec3,
    optical_depth: Float32,
    deep_scatter_rgb: Vec3,
) -> Vec3:
    """Evaluates Beer-Lambert spectral water absorption `L_bed * exp(-sigma * d) + L_deep * (1 - exp(-sigma * d))`."""
    var d = max(Float32(0.0), optical_depth)
    var tx = exp(-max(Float32(0.0), absorption_rgb.x) * d)
    var ty = exp(-max(Float32(0.0), absorption_rgb.y) * d)
    var tz = exp(-max(Float32(0.0), absorption_rgb.z) * d)
    return Vec3(
        bed_rgb.x * tx + deep_scatter_rgb.x * (Float32(1.0) - tx),
        bed_rgb.y * ty + deep_scatter_rgb.y * (Float32(1.0) - ty),
        bed_rgb.z * tz + deep_scatter_rgb.z * (Float32(1.0) - tz),
    )


def eval_nyquist_lean_water_alpha2_cpu(dx_px: Float32, shallow_damp: Float32 = 1.0) -> Float32:
    """CPU reference for 8-wave Beaufort Nyquist bandlimiting + analytical LEAN slope-variance transfer (`alpha^2`)."""
    var k_arr = [
        Float32(0.42), Float32(0.71), Float32(1.18), Float32(1.95),
        Float32(3.15), Float32(4.85), Float32(6.90), Float32(9.20),
    ]
    var amp_arr = [
        Float32(0.052), Float32(0.038), Float32(0.026), Float32(0.017),
        Float32(0.011), Float32(0.0072), Float32(0.0048), Float32(0.0034),
    ]
    var sigma2 = Float32(0.0)
    for wi in range(8):
        var kf = k_arr[wi]
        var af = amp_arr[wi] * shallow_damp
        var t_nyq = max(Float32(0.0), min(Float32(1.0), (Float32(2.6) - kf * dx_px) / Float32(2.05)))
        var w_nyq = t_nyq * t_nyq * (Float32(3.0) - Float32(2.0) * t_nyq)
        var ak = af * kf
        sigma2 += (Float32(1.0) - w_nyq * w_nyq) * Float32(0.5) * ak * ak
    return max(Float32(0.0012), min(Float32(0.35), Float32(0.0012) + Float32(2.0) * sigma2))


def _emit_voronoi_caustics_spv(mut s: AaaSpirvModule, ux: UInt32, uz: UInt32, time_s: UInt32) -> UInt32:
    """Dual-layer animated cellular Voronoi caustic interference pattern for shallow sunlit lakebeds."""
    var acc = s.fvar(s.float(0.0))
    for layer in range(2):
        var sc = s.float(Float32(1.65) if layer == 0 else Float32(2.35))
        var spd = s.float(Float32(0.55) if layer == 0 else Float32(-0.42))
        var px_c = s.fma(ux, sc, s.fmul(time_s, spd))
        var pz_c = s.fma(uz, sc, s.fmul(time_s, s.fmul(spd, s.float(0.73))))
        var ix0 = s.ffloor(px_c)
        var iz0 = s.ffloor(pz_c)
        var min_d2 = s.fvar(s.float(4.0))
        for oz in range(-1, 2):
            for ox in range(-1, 2):
                var cx = s.fadd(ix0, s.float(Float32(ox)))
                var cz = s.fadd(iz0, s.float(Float32(oz)))
                var h1 = s.fract(s.fmul(s.fsin(s.fma(cx, s.float(127.1), s.fmul(cz, s.float(311.7)))), s.float(43758.5453)))
                var h2 = s.fract(s.fmul(s.fsin(s.fma(cx, s.float(269.5), s.fmul(cz, s.float(183.3)))), s.float(43758.5453)))
                var jx = s.fma(s.float(0.35), s.fsin(s.fma(time_s, s.float(1.7), s.fmul(h1, s.float(6.283185)))), s.float(0.5))
                var jz = s.fma(s.float(0.35), s.fcos(s.fma(time_s, s.float(1.5), s.fmul(h2, s.float(6.283185)))), s.float(0.5))
                var dx = s.fsub(s.fadd(cx, jx), px_c)
                var dz = s.fsub(s.fadd(cz, jz), pz_c)
                var d2 = s.fma(dx, dx, s.fmul(dz, dz))
                s.store(min_d2, s.nmin(s.load_f(min_d2), d2))
        var md = s.fsqrt(s.nmax(s.float(0.0), s.load_f(min_d2)))
        var ridge = s.fpow(s.nclamp(md, s.float(0.0), s.float(1.0)), s.float(3.2))
        s.store(acc, s.fma(ridge, s.float(0.95), s.load_f(acc)))
    return s.load_f(acc)


def aaa_water_refract_shader(enable_ray_query: Bool) -> List[UInt32]:
    """Post-`SVGF`/`ReBLUR` Dielectric Water Compute Shader (`group = 8 x 8`).
    Push[0] = render_w, Push[1] = render_h, Push[2] = src_hdr_copy_off (`1024` in `BIND_AAA_POST_SCRATCH`).
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
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var b_surfcache = s.buffer(BIND_AAA_SURFCACHE)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var copy_off = s.push_word(2)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

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

    var water_y = s.load_float(b_globals, s.uint(140))
    # Only active when camera is above water plane
    s.return_if(s.fle(cam_y, s.fadd(water_y, s.float(0.02))))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))
    var fx0 = s.fsub(s.fadd(s.float_from_uint(px), s.float(0.5)), s.fmul(jit_x, s.fmul(fw, s.float(0.5))))
    var fy0 = s.fadd(s.fadd(s.float_from_uint(py), s.float(0.5)), s.fmul(jit_y, s.fmul(fh, s.float(0.5))))
    var ndc_x = s.fsub(s.fmul(s.fdiv(fx0, fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(fy0, fh), s.float(2.0)))

    var rd = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    s.return_if(s.fge(rd[1], s.float(-1e-4)))

    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var c_src_base = s.iadd(copy_off, s.imul(pix_idx, s.uint(8)))
    var z_bed = s.load_float(b_post, s.iadd(c_src_base, s.uint(3)))

    var t_water = s.fdiv(s.fsub(water_y, cam_y), rd[1])
    var cos_fwd = s.nmax(s.float(0.02), emit_vec3_dot(s, rd[0], rd[1], rd[2], fwd_x, fwd_y, fwd_z))
    var z_water = s.fmul(t_water, cos_fwd)
    s.return_if(s.logical_or(s.fle(t_water, s.float(0.05)), s.fge(z_water, z_bed)))

    var wx = s.fma(rd[0], t_water, cam_x)
    var wz = s.fma(rd[2], t_water, cam_z)
    var bmin_x = s.load_float(b_globals, s.uint(144))
    var bmin_z = s.load_float(b_globals, s.uint(145))
    var bmax_x = s.load_float(b_globals, s.uint(146))
    var bmax_z = s.load_float(b_globals, s.uint(147))
    var out_bounds = s.logical_or(
        s.logical_or(s.flt(wx, bmin_x), s.fgt(wx, bmax_x)),
        s.logical_or(s.flt(wz, bmin_z), s.fgt(wz, bmax_z)),
    )
    s.return_if(out_bounds)

    # Underwater ray distance & vertical depth `d_perp` (`z_bed >= 1e5` = open-ocean deep water, no seabed hit)
    var is_sky_bed = s.fge(z_bed, s.float(1e5))
    var raw_t_under = s.nmax(s.float(0.0), s.fsub(s.fdiv(z_bed, cos_fwd), t_water))
    var d_perp = s.select_f(
        is_sky_bed,
        s.float(80.0),
        s.nmax(s.float(0.0), s.fmul(raw_t_under, s.fneg(rd[1]))),
    )

    var time_s = s.load_float(b_globals, s.uint(123))
    var rain_int = s.load_float(b_globals, s.uint(125))
    var shallow_damp = s.nclamp(s.fmul(d_perp, s.float(1.6)), s.float(0.15), s.float(1.0))

    # 1. 8-Wave Irrational Golden-Angle Beaufort Spectrum + Nyquist Bandlimiting & LEAN Variance Transfer!
    var dx_px = s.nmax(
        s.float(0.02),
        s.fdiv(
            s.fdiv(s.fmul(s.float(2.0), s.fmul(t_water, tan_y)), fh),
            s.nmax(s.float(0.06), s.fneg(rd[1])),
        ),
    )
    var dx_w = s.fvar(s.float(0.0))
    var dz_w = s.fvar(s.float(0.0))
    var crest_h = s.fvar(s.float(0.0))
    var sigma2_lean = s.fvar(s.float(0.0))
    var k_arr = [
        Float32(0.42), Float32(0.71), Float32(1.18), Float32(1.95),
        Float32(3.15), Float32(4.85), Float32(6.90), Float32(9.20),
    ]
    var amp_arr = [
        Float32(0.052), Float32(0.038), Float32(0.026), Float32(0.017),
        Float32(0.011), Float32(0.0072), Float32(0.0048), Float32(0.0034),
    ]
    for wi in range(8):
        var kf = k_arr[wi]
        var af = amp_arr[wi]
        var ang_f = Float32(wi) * Float32(2.39996323) + Float32(0.37)
        var kx_f = cos(ang_f) * kf
        var kz_f = sin(ang_f) * kf
        var om_f = sqrt(Float32(9.81) * kf)

        var t_nyq = s.nclamp(
            s.fmul(s.fsub(s.float(2.6), s.fmul(s.float(kf), dx_px)), s.float(1.0 / 2.05)),
            s.float(0.0),
            s.float(1.0),
        )
        var w_nyq = s.fmul(s.fmul(t_nyq, t_nyq), s.fma(t_nyq, s.float(-2.0), s.float(3.0)))
        var a_eff = s.fmul(s.float(af), shallow_damp)
        var a_res = s.fmul(a_eff, w_nyq)
        var ph = s.fma(wx, s.float(kx_f), s.fma(wz, s.float(kz_f), s.fmul(time_s, s.float(om_f))))
        var cs = s.fmul(s.fcos(ph), a_res)
        s.store(dx_w, s.fma(cs, s.float(kx_f), s.load_f(dx_w)))
        s.store(dz_w, s.fma(cs, s.float(kz_f), s.load_f(dz_w)))
        s.store(crest_h, s.fma(s.fsin(ph), a_res, s.load_f(crest_h)))

        # Unresolved sub-pixel wave slope energy transfers directly into analytical LEAN GGX roughness!
        var ak = s.fmul(a_eff, s.float(kf))
        var lost_w = s.fsub(s.float(1.0), s.fmul(w_nyq, w_nyq))
        s.store(sigma2_lean, s.fma(lost_w, s.fmul(s.float(0.5), s.fmul(ak, ak)), s.load_f(sigma2_lean)))

    s.begin_if(s.fgt(rain_int, s.float(0.01)))
    s.begin_if(s.flt(t_water, s.float(40.0)))
    var r_rip = s.fmul(
        s.fsin(s.fma(s.fsqrt(s.fma(s.fract(s.fmul(wx, s.float(2.7))), s.fract(s.fmul(wx, s.float(2.7))), s.float(0.01))), s.float(42.0), s.fmul(time_s, s.float(-16.0)))),
        s.fmul(rain_int, s.float(0.035)),
    )
    s.store(dx_w, s.fadd(s.load_f(dx_w), r_rip))
    s.store(dz_w, s.fsub(s.load_f(dz_w), r_rip))
    s.end_if()
    s.store(sigma2_lean, s.fma(rain_int, s.float(0.0035), s.load_f(sigma2_lean)))
    s.end_if()

    var a2_w = s.nclamp(s.fma(s.load_f(sigma2_lean), s.float(2.0), s.float(0.0012)), s.float(0.0012), s.float(0.35))
    var rough_w = s.fsqrt(a2_w)
    var nw = emit_vec3_normalize(s, s.fneg(s.load_f(dx_w)), s.float(1.0), s.fneg(s.load_f(dz_w)))

    # 2. Silhouette-Guarded Snell Screen-Space Refraction (`IOR = 1.333`)
    var cur_u = s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw)
    var cur_v = s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh)
    var refr_amp = s.fmul(s.nclamp(d_perp, s.float(0.0), s.float(1.2)), s.float(0.045))
    var ru = s.nclamp(s.fma(nw[0], refr_amp, cur_u), s.float(0.001), s.float(0.999))
    var rv = s.nclamp(s.fma(nw[2], refr_amp, cur_v), s.float(0.001), s.float(0.999))
    var rpx = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.fmul(ru, fw)))
    var rpy = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.fmul(rv, fh)))
    var r_src_base = s.iadd(copy_off, s.imul(s.iadd(s.imul(rpy, w_u), rpx), s.uint(8)))
    var z_tap = s.load_float(b_post, s.iadd(r_src_base, s.uint(3)))

    # Guard: if refracted pixel is in front of water (`z_tap <= z_water`), fall back to unperturbed center sample!
    var refr_valid = s.fgt(z_tap, s.fadd(z_water, s.float(0.02)))
    var use_src = s.select_u(refr_valid, r_src_base, c_src_base)
    var use_z_bed = s.select_f(refr_valid, z_tap, z_bed)
    var use_is_sky_bed = s.fge(use_z_bed, s.float(1e5))
    var eff_under = s.select_f(
        use_is_sky_bed,
        s.float(80.0),
        s.nmax(s.float(0.0), s.fsub(s.fdiv(use_z_bed, cos_fwd), t_water)),
    )

    var bed_r = s.fvar(s.load_float(b_post, use_src))
    var bed_g = s.fvar(s.load_float(b_post, s.iadd(use_src, s.uint(1))))
    var bed_b = s.fvar(s.load_float(b_post, s.iadd(use_src, s.uint(2))))

    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var r_flags = s.load_word(b_globals, s.uint(99))
    var max_env_mip = s.load_float(b_globals, s.uint(103))

    # VSM Sun Shadow on Water Surface (modulates both shallow caustics & specular sun glints!)
    var sun_vis_w = s.fvar(s.float(1.0))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(1)), s.uint(0)))
    var wclip = emit_mat4_mul_point(s, b_globals, s.uint(56), wx, water_y, wz)
    var wlu = s.fma(wclip[0], s.float(0.5), s.float(0.5))
    var wlv = s.fma(s.fneg(wclip[1]), s.float(0.5), s.float(0.5))
    var in_wv = s.logical_and(
        s.logical_and(s.fgt(wlu, s.float(0.002)), s.flt(wlu, s.float(0.998))),
        s.logical_and(s.fgt(wlv, s.float(0.002)), s.flt(wlv, s.float(0.998))),
    )
    s.begin_if(s.logical_and(in_wv, s.flt(wclip[2], s.float(0.999))))
    var wvz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, wlu, wlv)
    s.store(sun_vis_w, s.select_f(s.flt(wvz, s.fsub(wclip[2], s.float(0.002))), s.float(0.0), s.float(1.0)))
    s.end_if()
    s.end_if()

    # 3. Procedural Voronoi Shallow Sun Caustics on Lakebed (`z_bed < 1e5 && d_perp < 6.0m`)
    s.begin_if(s.logical_and(s.flt(use_z_bed, s.float(1e5)), s.flt(d_perp, s.float(6.0))))
    var bx_ca = s.fma(rd[0], s.fadd(t_water, eff_under), s.fma(nw[0], s.float(0.35), cam_x))
    var bz_ca = s.fma(rd[2], s.fadd(t_water, eff_under), s.fma(nw[2], s.float(0.35), cam_z))
    var caust = _emit_voronoi_caustics_spv(s, bx_ca, bz_ca, time_s)
    var ca_win = s.fmul(
        s.exp2(s.fmul(d_perp, s.float(-0.65))),
        s.fmul(s.nclamp(s.fmul(d_perp, s.float(4.0)), s.float(0.0), s.float(1.0)), s.load_f(sun_vis_w)),
    )
    var ca_str = s.fmul(caust, s.fmul(ca_win, s.float(0.28)))
    s.store(bed_r, s.fma(sun_r, ca_str, s.load_f(bed_r)))
    s.store(bed_g, s.fma(sun_g, ca_str, s.load_f(bed_g)))
    s.store(bed_b, s.fma(sun_b, ca_str, s.load_f(bed_b)))
    s.end_if()

    # 4. Beer-Lambert Optical Depth Absorption (`exp(-sigma_water * d_opt)`)
    var sig_r = s.nmax(s.float(0.01), s.load_float(b_globals, s.uint(141)))
    var sig_g = s.nmax(s.float(0.01), s.load_float(b_globals, s.uint(142)))
    var sig_b = s.nmax(s.float(0.01), s.load_float(b_globals, s.uint(143)))
    var d_opt = s.fmul(eff_under, s.float(1.15))
    var neg_log2e = s.float(-1.44269504)
    var tr_r = s.exp2(s.fmul(s.fmul(sig_r, d_opt), neg_log2e))
    var tr_g = s.exp2(s.fmul(s.fmul(sig_g, d_opt), neg_log2e))
    var tr_b = s.exp2(s.fmul(s.fmul(sig_b, d_opt), neg_log2e))

    var sh_up = emit_eval_sh9_irradiance(s, b_globals, 72, s.float(0.0), s.float(1.0), s.float(0.0))
    var sun_up = s.fmul(s.nmax(s.float(0.0), sun_y), s.float(0.18))
    var deep_r = s.fmul(s.fma(sh_up[0], ibl_int, s.fmul(sun_r, sun_up)), s.float(0.025))
    var deep_g = s.fmul(s.fma(sh_up[1], ibl_int, s.fmul(sun_g, sun_up)), s.float(0.105))
    var deep_b = s.fmul(s.fma(sh_up[2], ibl_int, s.fmul(sun_b, sun_up)), s.float(0.135))

    var sub_r = s.fma(s.load_f(bed_r), tr_r, s.fmul(deep_r, s.fsub(s.float(1.0), tr_r)))
    var sub_g = s.fma(s.load_f(bed_g), tr_g, s.fmul(deep_g, s.fsub(s.float(1.0), tr_g)))
    var sub_b = s.fma(s.load_f(bed_b), tr_b, s.fmul(deep_b, s.fsub(s.float(1.0), tr_b)))

    # 5. Pure Dielectric Fresnel (`IOR = 1.333 -> F0 = 0.020`) + Textured Hardware RT Reflection!
    var vx = s.fneg(rd[0]); var vy = s.fneg(rd[1]); var vz = s.fneg(rd[2])
    var ndotv = s.nclamp(emit_vec3_dot(s, nw[0], nw[1], nw[2], vx, vy, vz), s.float(0.001), s.float(1.0))
    var omv = s.fsub(s.float(1.0), ndotv)
    var omv2 = s.fmul(omv, omv)
    var f_water = s.fma(s.float(0.980), s.fmul(s.fmul(omv2, omv2), omv), s.float(0.020))

    var two_nv = s.fmul(s.float(2.0), emit_vec3_dot(s, nw[0], nw[1], nw[2], vx, vy, vz))
    var r_raw = emit_vec3_normalize(
        s,
        s.fsub(s.fmul(nw[0], two_nv), vx),
        s.nmax(s.float(0.008), s.fsub(s.fmul(nw[1], two_nv), vy)),
        s.fsub(s.fmul(nw[2], two_nv), vz),
    )

    var env_lod_w = s.fmul(s.nclamp(rough_w, s.float(0.05), s.float(0.65)), max_env_mip)
    var env_w = s.sample_cube_lod(env_cube, s.vec3(r_raw[0], r_raw[1], r_raw[2]), env_lod_w)
    var refl_r = s.fvar(s.fmul(s.extract_f(env_w, 0), ibl_int))
    var refl_g = s.fvar(s.fmul(s.extract_f(env_w, 1), ibl_int))
    var refl_b = s.fvar(s.fmul(s.extract_f(env_w, 2), ibl_int))

    if enable_ray_query:
        s.begin_if(s.ine(s.bit_and(r_flags, s.uint(4)), s.uint(0)))
        var rt_w = emit_rt_textured_hit_shade(
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
            wx, s.fadd(water_y, s.float(0.015)), wz,
            r_raw[0], r_raw[1], r_raw[2],
            s.float(0.02), s.float(280.0),
            t_water, s.nmax(s.float(0.003), a2_w),
            s.load_f(refl_r), s.load_f(refl_g), s.load_f(refl_b),
        )
        s.store(refl_r, rt_w[0])
        s.store(refl_g, rt_w[1])
        s.store(refl_b, rt_w[2])
        s.end_if()

    # Crisp LEAN-Filtered GGX Sun Specular Glint on Water Capillary Crests
    var h_sun = emit_vec3_normalize(s, s.fadd(vx, sun_x), s.fadd(vy, sun_y), s.fadd(vz, sun_z))
    var ndoth = s.nmax(s.float(0.0), emit_vec3_dot(s, nw[0], nw[1], nw[2], h_sun[0], h_sun[1], h_sun[2]))
    var ndotl_w = s.nmax(s.float(0.0), emit_vec3_dot(s, nw[0], nw[1], nw[2], sun_x, sun_y, sun_z))
    var d_den = s.fma(s.fmul(ndoth, ndoth), s.fsub(a2_w, s.float(1.0)), s.float(1.0))
    var d_ggx = s.fdiv(a2_w, s.nmax(s.float(1e-5), s.fmul(s.float(3.14159265), s.fmul(d_den, d_den))))
    var sun_glint = s.nmin(s.float(24.0), s.fmul(d_ggx, s.fmul(s.float(0.025), s.fmul(ndotl_w, s.load_f(sun_vis_w)))))

    var om_fw = s.fsub(s.float(1.0), f_water)
    var wat_r_v = s.fvar(s.fma(s.load_f(refl_r), f_water, s.fma(sub_r, om_fw, s.fmul(sun_r, sun_glint))))
    var wat_g_v = s.fvar(s.fma(s.load_f(refl_g), f_water, s.fma(sub_g, om_fw, s.fmul(sun_g, sun_glint))))
    var wat_b_v = s.fvar(s.fma(s.load_f(refl_b), f_water, s.fma(sub_b, om_fw, s.fmul(sun_b, sun_glint))))

    # 5b. Clustered Local / Boardwalk / Firework GGX Water Specular Glints (`BIND_AAA_CLUSTERS`)!
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(32)), s.uint(0)))
    var tot_wl = s.umin(s.load_word(b_clusters, s.uint(0)), s.uint(24))
    var li_v = s.uvar(s.uint(0))
    s.begin_loop()
    var li = s.load_u(li_v)
    s.break_if(s.uge(li, tot_wl))
    var l_off = s.iadd(s.uint(16), s.imul(li, s.uint(16)))
    var lpx = s.load_float(b_clusters, l_off)
    var lpy = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
    var lpz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))
    var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
    var lrr = s.load_float(b_clusters, s.iadd(l_off, s.uint(4)))
    var lrg = s.load_float(b_clusters, s.iadd(l_off, s.uint(5)))
    var lrb = s.load_float(b_clusters, s.iadd(l_off, s.uint(6)))
    var kind_l = s.load_word(b_clusters, s.iadd(l_off, s.uint(7)))
    var aux_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(8)))
    var aux_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(9)))
    var aux_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(10)))
    var p0_l = s.load_float(b_clusters, s.iadd(l_off, s.uint(11)))
    var avx_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(12)))
    var avx_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(13)))
    var avx_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(14)))
    var p1_l = s.load_float(b_clusters, s.iadd(l_off, s.uint(15)))

    var tx_l = s.fvar(lpx)
    var ty_l = s.fvar(lpy)
    var tz_l = s.fvar(lpz)
    s.begin_if(s.uge(kind_l, s.uint(2)))
    var cwx = s.fsub(wx, lpx)
    var cwy = s.fsub(water_y, lpy)
    var cwz = s.fsub(wz, lpz)
    var u_l2 = s.nmax(s.float(1e-6), s.fma(aux_x, aux_x, s.fma(aux_y, aux_y, s.fmul(aux_z, aux_z))))
    var tu_c = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx, cwy, cwz, aux_x, aux_y, aux_z), u_l2), s.float(-1.0), s.float(1.0))
    var tv_c = s.fvar(s.float(0.0))
    s.begin_if(s.ieq(kind_l, s.uint(2)))
    var v_l2 = s.nmax(s.float(1e-6), s.fma(avx_x, avx_x, s.fma(avx_y, avx_y, s.fmul(avx_z, avx_z))))
    s.store(tv_c, s.nclamp(s.fdiv(emit_vec3_dot(s, cwx, cwy, cwz, avx_x, avx_y, avx_z), v_l2), s.float(-1.0), s.float(1.0)))
    s.end_if()
    var ftv = s.load_f(tv_c)
    s.store(tx_l, s.fma(aux_x, tu_c, s.fma(avx_x, ftv, lpx)))
    s.store(ty_l, s.fma(aux_y, tu_c, s.fma(avx_y, ftv, lpy)))
    s.store(tz_l, s.fma(aux_z, tu_c, s.fma(avx_z, ftv, lpz)))
    s.end_if()

    var dlx = s.fsub(s.load_f(tx_l), wx)
    var dly = s.fsub(s.load_f(ty_l), water_y)
    var dlz = s.fsub(s.load_f(tz_l), wz)
    var d2_l = s.nmax(s.float(0.04), s.fma(dlx, dlx, s.fma(dly, dly, s.fmul(dlz, dlz))))
    var r2_l = s.nmax(s.float(0.04), s.fmul(lrad, lrad))
    s.begin_if(s.logical_and(s.fgt(dly, s.float(0.02)), s.flt(d2_l, r2_l)))
    var inv_dl = s.inverse_sqrt(d2_l)
    var lx_w = s.fmul(dlx, inv_dl); var ly_w = s.fmul(dly, inv_dl); var lz_w = s.fmul(dlz, inv_dl)
    var ndotl_l = s.nmax(s.float(0.0), emit_vec3_dot(s, nw[0], nw[1], nw[2], lx_w, ly_w, lz_w))
    var q_l = s.fdiv(d2_l, r2_l)
    var q2_l = s.nclamp(s.fsub(s.float(1.0), s.fmul(q_l, q_l)), s.float(0.0), s.float(1.0))
    var att_l = s.fvar(s.fdiv(s.fmul(q2_l, q2_l), s.fadd(d2_l, s.float(0.45))))
    s.begin_if(s.ieq(kind_l, s.uint(1)))
    var cos_sp = s.fneg(emit_vec3_dot(s, lx_w, ly_w, lz_w, aux_x, aux_y, aux_z))
    var sw_l = s.nclamp(s.fmul(s.fsub(cos_sp, p0_l), p1_l), s.float(0.0), s.float(1.0))
    s.store(att_l, s.fmul(s.load_f(att_l), s.fmul(sw_l, sw_l)))
    s.end_if()
    var hl_w = emit_vec3_normalize(s, s.fadd(vx, lx_w), s.fadd(vy, ly_w), s.fadd(vz, lz_w))
    var nh_l = s.nmax(s.float(0.0), emit_vec3_dot(s, nw[0], nw[1], nw[2], hl_w[0], hl_w[1], hl_w[2]))
    var a2_loc = s.nmax(s.float(0.0035), a2_w)
    var den_l = s.fma(s.fmul(nh_l, nh_l), s.fsub(a2_loc, s.float(1.0)), s.float(1.0))
    var dggx_l = s.fdiv(a2_loc, s.nmax(s.float(1e-4), s.fmul(s.float(3.14159265), s.fmul(den_l, den_l))))
    var gl_w = s.nmin(s.float(16.0), s.fmul(dggx_l, s.fmul(f_water, s.fmul(ndotl_l, s.load_f(att_l)))))
    s.store(wat_r_v, s.fma(lrr, gl_w, s.load_f(wat_r_v)))
    s.store(wat_g_v, s.fma(lrg, gl_w, s.load_f(wat_g_v)))
    s.store(wat_b_v, s.fma(lrb, gl_w, s.load_f(wat_b_v)))
    s.end_if()

    s.store(li_v, s.iadd(li, s.uint(1)))
    s.end_loop()
    s.end_if()

    var wat_r = s.load_f(wat_r_v)
    var wat_g = s.load_f(wat_g_v)
    var wat_b = s.load_f(wat_b_v)

    # 6. Domain-Warped Cellular Lace Shoreline (`d_perp < 0.48m`) & Wave-Crest Whitecap Foam
    var shore_t = s.nclamp(s.fmul(s.fsub(s.float(0.48), d_perp), s.float(1.0 / 0.45)), s.float(0.0), s.float(1.0))
    var warp_x = s.fma(nw[0], s.float(0.65), s.fma(s.fsin(s.fma(wz, s.float(1.7), time_s)), s.float(0.28), wx))
    var warp_z = s.fma(nw[2], s.float(0.65), s.fma(s.fcos(s.fma(wx, s.float(1.5), s.fmul(time_s, s.float(0.85)))), s.float(0.28), wz))
    var cell_f1 = _emit_voronoi_caustics_spv(s, s.fmul(warp_x, s.float(1.35)), s.fmul(warp_z, s.float(1.35)), s.fmul(time_s, s.float(0.45)))
    var lace = s.nclamp(s.fma(cell_f1, s.float(0.75), s.float(0.22)), s.float(0.0), s.float(1.0))
    var crest_foam = s.nclamp(s.fmul(s.fsub(s.load_f(crest_h), s.float(0.068)), s.float(11.0)), s.float(0.0), s.float(0.42))
    var foam_w = s.nclamp(s.fma(s.fmul(shore_t, shore_t), lace, s.fmul(crest_foam, lace)), s.float(0.0), s.float(0.88))
    var foam_lit_r = s.fmul(s.fma(sun_r, s.fmul(s.nmax(s.float(0.0), sun_y), s.fmul(s.load_f(sun_vis_w), s.float(0.3183))), s.fmul(sh_up[0], ibl_int)), s.float(0.90))
    var foam_lit_g = s.fmul(s.fma(sun_g, s.fmul(s.nmax(s.float(0.0), sun_y), s.fmul(s.load_f(sun_vis_w), s.float(0.3183))), s.fmul(sh_up[1], ibl_int)), s.float(0.92))
    var foam_lit_b = s.fmul(s.fma(sun_b, s.fmul(s.nmax(s.float(0.0), sun_y), s.fmul(s.load_f(sun_vis_w), s.float(0.3183))), s.fmul(sh_up[2], ibl_int)), s.float(0.95))

    var fin_r_raw = s.fma(s.fsub(foam_lit_r, wat_r), foam_w, wat_r)
    var fin_g_raw = s.fma(s.fsub(foam_lit_g, wat_g), foam_w, wat_g)
    var fin_b_raw = s.fma(s.fsub(foam_lit_b, wat_b), foam_w, wat_b)

    # Seamless open-ocean atmospheric horizon aerial-perspective blend (`is_sky_bed && t_water > 350m`)
    var hor_r_g = s.load_float(b_globals, s.uint(164))
    var hor_g_g = s.load_float(b_globals, s.uint(165))
    var hor_b_g = s.load_float(b_globals, s.uint(166))
    var has_hor_anc = s.fgt(s.fadd(hor_r_g, s.fadd(hor_g_g, hor_b_g)), s.float(1e-5))
    var cos_sun_h = s.fma(rd[0], sun_x, s.fmul(rd[2], sun_z))
    var mie_h = s.fmul(
        s.fpow(s.nclamp(s.fma(cos_sun_h, s.float(0.5), s.float(0.5)), s.float(0.0), s.float(1.0)), s.float(8.0)),
        s.float(0.41),
    )
    var env_hor = s.sample_cube_lod(env_cube, s.vec3(rd[0], s.float(0.02), rd[2]), s.float(0.0))
    var sky_hr = s.select_f(has_hor_anc, s.fma(sun_r, mie_h, hor_r_g), s.fmul(s.extract_f(env_hor, 0), ibl_int))
    var sky_hg = s.select_f(has_hor_anc, s.fma(sun_g, mie_h, hor_g_g), s.fmul(s.extract_f(env_hor, 1), ibl_int))
    var sky_hb = s.select_f(has_hor_anc, s.fma(sun_b, mie_h, hor_b_g), s.fmul(s.extract_f(env_hor, 2), ibl_int))
    var dist_far = s.nmax(s.float(0.0), s.fsub(t_water, s.float(350.0)))
    var hor_haze = s.select_f(
        is_sky_bed,
        s.fsub(s.float(1.0), s.exp2(s.fmul(dist_far, s.float(-0.00095)))),
        s.float(0.0),
    )
    var fin_r = s.fma(s.fsub(sky_hr, fin_r_raw), hor_haze, fin_r_raw)
    var fin_g = s.fma(s.fsub(sky_hg, fin_g_raw), hor_haze, fin_g_raw)
    var fin_b = s.fma(s.fsub(sky_hb, fin_b_raw), hor_haze, fin_b_raw)

    # Exact planar camera motion vector at water surface `(wx, water_y, wz)`
    var p_clip = emit_mat4_mul_point(s, b_globals, s.uint(16), wx, water_y, wz)
    var inv_pw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), p_clip[3]))
    var prev_u = s.fma(s.fmul(p_clip[0], inv_pw), s.float(0.5), s.float(0.5))
    var prev_v = s.fma(s.fneg(s.fmul(p_clip[1], inv_pw)), s.float(0.5), s.float(0.5))
    var vel_u = s.fsub(cur_u, prev_u)
    var vel_v = s.fsub(cur_v, prev_v)

    var out_base = s.imul(pix_idx, s.uint(8))
    s.store_float(b_hdr, out_base, fin_r)
    s.store_float(b_hdr, s.iadd(out_base, s.uint(1)), fin_g)
    s.store_float(b_hdr, s.iadd(out_base, s.uint(2)), fin_b)
    s.store_float(b_hdr, s.iadd(out_base, s.uint(3)), z_water)
    s.store_float(b_hdr, s.iadd(out_base, s.uint(4)), vel_u)
    s.store_float(b_hdr, s.iadd(out_base, s.uint(5)), vel_v)
    s.store_float(b_hdr, s.iadd(out_base, s.uint(6)), nw[0])
    s.store_float(b_hdr, s.iadd(out_base, s.uint(7)), nw[1])

    # Mark dynamic wave surface reactive mask for `FlintTSR v3`
    var react_slot = s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), pix_idx)
    s.store_float(b_oit, react_slot, s.float(0.28))
    return s.finish()
