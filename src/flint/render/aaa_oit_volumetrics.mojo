"""Milestone 4 (`V15`): Per-Pixel 4-Node Exact Short-Stack `OIT` & 6-Way Lightmapped
(`RLUDBF`) Self-Shadowed Beer-Powder Volumetric Plumes (`aaa_oit_volumetrics.mojo`).

Eliminates alpha-sorting popping when glass canopies, HUD holograms, clouds, and
missile smoke plumes intersect, and illuminates volumetric plumes directionally from
all 6 cardinal axes (`+X, -X, +Y, -Y, +Z, -Z`) with Henyey-Greenstein phase scattering
and Beer-Powder dark-edge self-shadowing.
"""
from std.math import exp, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_OIT_NODES,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)

# Per-pixel OIT layout in `BIND_AAA_OIT_NODES`:
# Header region [0 .. MAX_PLUMES * 12 - 1]: Volumetric Plume / Transparent Quad descriptors (`12` words each):
#   [0..2]: world_pos (x, y, z)
#   [3]:    radius
#   [4..6]: base_albedo_or_tint (r, g, b)
#   [7]:    optical_density (alpha / extinction)
#   [8..10]: core_emission_rgb (fireball blackbody glow)
#   [11]:   kind (0 = 6-Way Volumetric Smoke Plume, 1 = Optical Glass / Hologram Sheet)
# Per-pixel reactive mask output at offset `OIT_REACTIVE_OFFSET_WORDS + y * width + x`:
#   Float32 reactive weight `[0..1]` consumed by `FlintTSR v2` to prevent ghosting on moving particles!
comptime AAA_MAX_OIT_PRIMS: Int = 256
comptime AAA_OIT_PRIM_WORDS: Int = 12
comptime AAA_OIT_REACTIVE_BASE_WORDS: Int = AAA_MAX_OIT_PRIMS * AAA_OIT_PRIM_WORDS  # 3072


struct AaaVolumetricPlume(Copyable, Movable):
    var center: Vec3
    var radius: Float32
    var scatter_albedo: Vec3
    var density: Float32
    var emission: Vec3
    var is_glass: Bool
    var is_hologram: Bool

    def __init__(
        out self,
        center: Vec3,
        radius: Float32,
        scatter_albedo: Vec3 = Vec3(0.82, 0.80, 0.78),
        density: Float32 = 0.75,
        emission: Vec3 = Vec3(0.0, 0.0, 0.0),
        is_glass: Bool = False,
        is_hologram: Bool = False,
    ):
        self.center = center
        self.radius = radius
        self.scatter_albedo = scatter_albedo
        self.density = density
        self.emission = emission
        self.is_glass = is_glass
        self.is_hologram = is_hologram

    @staticmethod
    def hologram(
        center: Vec3,
        radius: Float32,
        tint: Vec3 = Vec3(0.15, 0.85, 1.0),
        density: Float32 = 0.55,
        emission: Vec3 = Vec3(0.35, 1.40, 2.20),
    ) -> AaaVolumetricPlume:
        return AaaVolumetricPlume(
            center=center,
            radius=radius,
            scatter_albedo=tint,
            density=density,
            emission=emission,
            is_glass=True,
            is_hologram=True,
        )

    def append_words(self, mut out: List[UInt32]):
        out.append(bitcast[DType.uint32](self.center.x))
        out.append(bitcast[DType.uint32](self.center.y))
        out.append(bitcast[DType.uint32](self.center.z))
        out.append(bitcast[DType.uint32](self.radius))
        out.append(bitcast[DType.uint32](self.scatter_albedo.x))
        out.append(bitcast[DType.uint32](self.scatter_albedo.y))
        out.append(bitcast[DType.uint32](self.scatter_albedo.z))
        out.append(bitcast[DType.uint32](self.density))
        out.append(bitcast[DType.uint32](self.emission.x))
        out.append(bitcast[DType.uint32](self.emission.y))
        out.append(bitcast[DType.uint32](self.emission.z))
        var kind_u = UInt32(2) if self.is_hologram else (UInt32(1) if self.is_glass else UInt32(0))
        out.append(kind_u)


def eval_hologram_fresnel_rim_cpu(chord_norm: Float32, world_y: Float32) -> Tuple[Float32, Float32]:
    """CPU reference for Fresnel Rim Hologram Grazing Modulation (`(alpha_scale, emission_scale)`).
    Translucent core (`chord_norm -> 1.0`) + glowing grazing Fresnel silhouette (`chord_norm -> 0.0`).
    """
    var cos_v = max(Float32(0.0), min(Float32(1.0), chord_norm))
    var omc = Float32(1.0) - cos_v
    var rim_pow = omc * omc * sqrt(max(Float32(1e-4), omc))
    var scan = Float32(0.82) + Float32(0.18) * sin(world_y * Float32(42.0))
    var alpha_scale = Float32(0.20) + Float32(0.80) * rim_pow
    var em_scale = (Float32(0.14) + Float32(0.86) * rim_pow) * scan
    return (alpha_scale, em_scale)


def eval_6way_lightmap_plume_cpu(
    local_uvw: Vec3,
    light_dir_local: Vec3,
    view_dir_local: Vec3,
    density: Float32,
) -> Tuple[Float32, Float32]:
    """CPU reference for 6-Way Lightmapped (`RLUDBF`) optical transmittance + Beer-Powder phase light!
    Returns `(alpha, directional_light_factor)`.
    """
    var r2 = local_uvw.length_squared()
    if r2 >= Float32(1.0):
        return (Float32(0.0), Float32(0.0))
    var radial = Float32(1.0) - sqrt(r2)
    var chord_tau = density * radial * Float32(2.4)
    var alpha = Float32(1.0) - exp(-chord_tau)

    # 6-Way directional optical depth along (+X, -X, +Y, -Y, +Z, -Z) through the billowing puff
    var tau_px = max(Float32(0.0), Float32(1.0) - local_uvw.x) * density * Float32(1.6)
    var tau_nx = max(Float32(0.0), Float32(1.0) + local_uvw.x) * density * Float32(1.6)
    var tau_py = max(Float32(0.0), Float32(1.0) - local_uvw.y) * density * Float32(1.6)
    var tau_ny = max(Float32(0.0), Float32(1.0) + local_uvw.y) * density * Float32(1.6)
    var tau_pz = max(Float32(0.0), Float32(1.0) - local_uvw.z) * density * Float32(1.6)
    var tau_nz = max(Float32(0.0), Float32(1.0) + local_uvw.z) * density * Float32(1.6)

    var wx = light_dir_local.x * light_dir_local.x
    var wy = light_dir_local.y * light_dir_local.y
    var wz = light_dir_local.z * light_dir_local.z
    var dir_tau = (
        wx * (tau_px if light_dir_local.x >= Float32(0.0) else tau_nx)
        + wy * (tau_py if light_dir_local.y >= Float32(0.0) else tau_ny)
        + wz * (tau_pz if light_dir_local.z >= Float32(0.0) else tau_nz)
    )
    # Horizon Zero Dawn / Guerrilla Beer-Powder law: `2 * exp(-tau) * (1 - exp(-2 * tau))`
    var beer = exp(-dir_tau)
    var powder = Float32(1.0) - exp(-dir_tau * Float32(2.0))
    var beer_powder = Float32(0.22) + Float32(1.78) * beer * powder

    # Dual-lobe Henyey-Greenstein forward silver-lining + back-scatter
    var cos_th = max(Float32(-1.0), min(Float32(1.0), (Vec3(0.0, 0.0, 0.0) - view_dir_local).dot(light_dir_local)))
    var g = Float32(0.55)
    var denom = Float32(1.0) + g * g - Float32(2.0) * g * cos_th
    var hg_fwd = (Float32(1.0) - g * g) / max(Float32(0.05), denom * sqrt(denom))
    var phase = Float32(0.65) + Float32(0.35) * min(Float32(3.5), hg_fwd)
    return (alpha, beer_powder * phase)


def oit_and_6way_volumetric_shader() -> List[UInt32]:
    """Compute Pass: Evaluates unsorted transparent sheets & 6-Way Lightmapped Volumetric Plumes
    into a Per-Pixel **4-Node Exact Depth-Sorted Short-Stack OIT** (`z0 <= z1 <= z2 <= z3` + tail),
    composites over `BIND_AAA_HDR_OUT` with zero sorting pop, and writes the `FlintTSR v2` reactive mask!
    Push[0] = render_w, Push[1] = render_h, Push[2] = prim_count.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var prim_cnt = s.push_word(2)
    var preserve_reactive = s.ine(s.push_word(3), s.uint(0))
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var react_off = s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), pix_idx)
    s.begin_if(s.ieq(prim_cnt, s.uint(0)))
    s.begin_if(s.logical_not(preserve_reactive))
    s.store_float(b_oit, react_off, s.float(0.0))
    s.end_if()
    s.end_if()
    s.return_if(s.ieq(prim_cnt, s.uint(0)))

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
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var amb_sky_r = s.fmul(s.load_float(b_globals, s.uint(72)), s.fmul(ibl_int, s.float(0.35)))
    var amb_sky_g = s.fmul(s.load_float(b_globals, s.uint(73)), s.fmul(ibl_int, s.float(0.35)))
    var amb_sky_b = s.fmul(s.load_float(b_globals, s.uint(74)), s.fmul(ibl_int, s.float(0.35)))

    var ndc_x = s.fsub(s.fmul(s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh), s.float(2.0)))
    var rd = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )

    var hdr_base = s.imul(pix_idx, s.uint(8))
    var opaque_z = s.load_float(b_hdr, s.iadd(hdr_base, s.uint(3)))

    # Per-pixel 4-node exact depth-sorted OIT registers (`z, premul_r, premul_g, premul_b, alpha`)
    var z0 = s.fvar(s.float(1e9)); var r0 = s.fvar(s.float(0.0)); var g0 = s.fvar(s.float(0.0)); var b0_v = s.fvar(s.float(0.0)); var a0 = s.fvar(s.float(0.0))
    var z1 = s.fvar(s.float(1e9)); var r1 = s.fvar(s.float(0.0)); var g1 = s.fvar(s.float(0.0)); var b1_v = s.fvar(s.float(0.0)); var a1 = s.fvar(s.float(0.0))
    var z2 = s.fvar(s.float(1e9)); var r2 = s.fvar(s.float(0.0)); var g2 = s.fvar(s.float(0.0)); var b2_v = s.fvar(s.float(0.0)); var a2 = s.fvar(s.float(0.0))
    var z3 = s.fvar(s.float(1e9)); var r3 = s.fvar(s.float(0.0)); var g3 = s.fvar(s.float(0.0)); var b3_v = s.fvar(s.float(0.0)); var a3 = s.fvar(s.float(0.0))

    var pi = s.uvar(s.uint(0))
    s.begin_loop()
    var k = s.load_u(pi)
    s.break_if(s.uge(k, prim_cnt))
    var p_off = s.imul(k, s.uint(UInt32(AAA_OIT_PRIM_WORDS)))
    var cx = s.load_float(b_oit, p_off)
    var cy = s.load_float(b_oit, s.iadd(p_off, s.uint(1)))
    var cz = s.load_float(b_oit, s.iadd(p_off, s.uint(2)))
    var rad = s.load_float(b_oit, s.iadd(p_off, s.uint(3)))

    # Ray-sphere intersection against billowing plume / glass lens
    var ocx = s.fsub(cx, cam_x)
    var ocy = s.fsub(cy, cam_y)
    var ocz = s.fsub(cz, cam_z)
    var t_ca = emit_vec3_dot(s, ocx, ocy, ocz, rd[0], rd[1], rd[2])
    var d2 = s.fsub(emit_vec3_dot(s, ocx, ocy, ocz, ocx, ocy, ocz), s.fmul(t_ca, t_ca))
    var rad2 = s.fmul(rad, rad)

    s.begin_if(s.logical_and(s.fgt(t_ca, s.float(0.05)), s.flt(d2, rad2)))
    var thc = s.fsqrt(s.nmax(s.float(0.0), s.fsub(rad2, d2)))
    var t_mid = t_ca
    var z_lin = emit_vec3_dot(s, s.fmul(rd[0], t_mid), s.fmul(rd[1], t_mid), s.fmul(rd[2], t_mid), fwd_x, fwd_y, fwd_z)

    s.begin_if(s.flt(z_lin, opaque_z))
    var soft_fade = s.nclamp(s.fdiv(s.fsub(opaque_z, z_lin), s.nmax(s.float(0.1), rad)), s.float(0.0), s.float(1.0))
    var dens = s.load_float(b_oit, s.iadd(p_off, s.uint(7)))
    var kind = s.load_word(b_oit, s.iadd(p_off, s.uint(11)))
    var sc_r = s.load_float(b_oit, s.iadd(p_off, s.uint(4)))
    var sc_g = s.load_float(b_oit, s.iadd(p_off, s.uint(5)))
    var sc_b = s.load_float(b_oit, s.iadd(p_off, s.uint(6)))
    var em_r = s.load_float(b_oit, s.iadd(p_off, s.uint(8)))
    var em_g = s.load_float(b_oit, s.iadd(p_off, s.uint(9)))
    var em_b = s.load_float(b_oit, s.iadd(p_off, s.uint(10)))

    var chord_norm = s.fdiv(thc, s.nmax(s.float(1e-4), rad))
    var chord_smooth = s.fmul(chord_norm, s.fmul(chord_norm, chord_norm))
    var tau = s.fmul(dens, s.fmul(chord_smooth, s.float(2.2)))
    var one_m_exp = s.fsub(s.float(1.0), s.exp2(s.fneg(s.fmul(tau, s.float(1.442695)))))
    var is_sheet = s.uge(kind, s.uint(1))
    var em_sum = s.fadd(em_r, s.fadd(em_g, em_b))
    var is_holo = s.logical_or(s.ieq(kind, s.uint(2)), s.logical_and(s.ieq(kind, s.uint(1)), s.fgt(em_sum, s.float(0.20))))

    var hit_wx = s.fma(rd[0], t_mid, cam_x)
    var hit_wy = s.fma(rd[1], t_mid, cam_y)
    var hit_wz = s.fma(rd[2], t_mid, cam_z)

    # Fresnel Rim Hologram Grazing Modulation: translucent core + glowing silhouette rim + scanlines
    var cos_v_h = s.nclamp(chord_norm, s.float(0.0), s.float(1.0))
    var omc_h = s.fsub(s.float(1.0), cos_v_h)
    var rim_pow = s.fmul(omc_h, s.fmul(omc_h, s.fsqrt(s.nmax(s.float(1e-4), omc_h))))
    var scan_h = s.fma(s.float(0.18), s.fsin(s.fmul(hit_wy, s.float(42.0))), s.float(0.82))
    var holo_a_mul = s.select_f(is_holo, s.fma(s.float(0.80), rim_pow, s.float(0.20)), s.float(1.0))
    var holo_e_mul = s.select_f(is_holo, s.fmul(s.fma(s.float(0.86), rim_pow, s.float(0.14)), scan_h), s.float(1.0))

    var base_frag_a = s.select_f(is_sheet, s.fmul(dens, soft_fade), s.fmul(one_m_exp, soft_fade))
    var frag_a = s.nclamp(
        s.fmul(base_frag_a, holo_a_mul),
        s.float(0.0),
        s.float(0.98),
    )

    # 6-Way Lightmap (`RLUDBF`) + Beer-Powder self-shadowing + Henyey-Greenstein phase!
    var hit_x = s.fsub(hit_wx, cx)
    var hit_y = s.fsub(hit_wy, cy)
    var hit_z = s.fsub(hit_wz, cz)
    var inv_r = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), rad))
    var ux = s.fmul(hit_x, inv_r); var uy = s.fmul(hit_y, inv_r); var uz = s.fmul(hit_z, inv_r)

    var tx_6 = s.select_f(s.fge(sun_x, s.float(0.0)), s.fsub(s.float(1.0), ux), s.fadd(s.float(1.0), ux))
    var ty_6 = s.select_f(s.fge(sun_y, s.float(0.0)), s.fsub(s.float(1.0), uy), s.fadd(s.float(1.0), uy))
    var tz_6 = s.select_f(s.fge(sun_z, s.float(0.0)), s.fsub(s.float(1.0), uz), s.fadd(s.float(1.0), uz))
    var dir_tau = s.fmul(
        s.fma(s.fmul(sun_x, sun_x), tx_6, s.fma(s.fmul(sun_y, sun_y), ty_6, s.fmul(s.fmul(sun_z, sun_z), tz_6))),
        s.fmul(dens, s.float(1.4)),
    )
    var beer = s.exp2(s.fneg(s.fmul(dir_tau, s.float(1.442695))))
    var powder = s.fsub(s.float(1.0), s.exp2(s.fneg(s.fmul(dir_tau, s.float(2.88539)))))
    var bp_light = s.select_f(is_sheet, s.float(1.0), s.fma(s.float(1.78), s.fmul(beer, powder), s.float(0.25)))

    var cos_th = emit_vec3_dot(s, rd[0], rd[1], rd[2], sun_x, sun_y, sun_z)
    var hg_boost = s.fma(s.float(0.45), s.nmax(s.float(0.0), cos_th), s.float(0.75))
    var lit_scale = s.select_f(is_holo, s.float(0.18), s.fmul(bp_light, hg_boost))

    # Soft-knee highlight compression on hologram core emission so `.flrp` ghosts never blow out solid white
    var em_peak = s.nmax(em_r, s.nmax(em_g, em_b))
    var holo_knee = s.select_f(is_holo, s.fdiv(holo_e_mul, s.fma(em_peak, s.float(0.22), s.float(1.0))), s.float(1.0))
    var eff_em_r = s.fmul(em_r, holo_knee)
    var eff_em_g = s.fmul(em_g, holo_knee)
    var eff_em_b = s.fmul(em_b, holo_knee)

    var col_r = s.fma(sc_r, s.fmul(sun_r, lit_scale), eff_em_r)
    var col_g = s.fma(sc_g, s.fmul(sun_g, lit_scale), eff_em_g)
    var col_b = s.fma(sc_b, s.fmul(sun_b, lit_scale), eff_em_b)
    var pm_r_v = s.fvar(s.fmul(col_r, frag_a))
    var pm_g_v = s.fvar(s.fmul(col_g, frag_a))
    var pm_b_v = s.fvar(s.fmul(col_b, frag_a))
    var frag_a_v = s.fvar(frag_a)

    # Pyroclastic 4-Step Stratified 3D Billow FBM Raymarch + Blackbody Core-to-Carbon-Soot (`kind == 0`)!
    s.begin_if(s.logical_not(is_sheet))
    var cos_fwd_p = s.nmax(s.float(0.05), emit_vec3_dot(s, rd[0], rd[1], rd[2], fwd_x, fwd_y, fwd_z))
    var t_opaq = s.fdiv(opaque_z, cos_fwd_p)
    var t_en = s.nmax(s.float(0.05), s.fsub(t_ca, thc))
    var t_ex = s.nmin(t_opaq, s.fadd(t_ca, thc))
    var seg_len = s.nmax(s.float(0.0), s.fsub(t_ex, t_en))
    var ds_p = s.fmul(seg_len, s.float(0.25))
    var ds_norm = s.fdiv(ds_p, s.nmax(s.float(1e-4), rad))
    var is_fire = s.fgt(em_peak, s.float(0.25))

    var tr_p = s.fvar(s.float(1.0))
    var pr_acc = s.fvar(s.float(0.0))
    var pg_acc = s.fvar(s.float(0.0))
    var pb_acc = s.fvar(s.float(0.0))

    for st_i in range(4):
        var t_s = s.fma(s.float(Float32(st_i) * Float32(0.25) + Float32(0.125)), seg_len, t_en)
        var sx_w = s.fma(rd[0], t_s, cam_x)
        var sy_w = s.fma(rd[1], t_s, cam_y)
        var sz_w = s.fma(rd[2], t_s, cam_z)
        var lx_p = s.fmul(s.fsub(sx_w, cx), inv_r)
        var ly_p = s.fmul(s.fsub(sy_w, cy), inv_r)
        var lz_p = s.fmul(s.fsub(sz_w, cz), inv_r)
        var r_p = s.fsqrt(s.nmax(s.float(0.0), s.fma(lx_p, lx_p, s.fma(ly_p, ly_p, s.fmul(lz_p, lz_p)))))

        # Buoyant 3-octave 3D Billow FBM (`1 - |sin * cos|`) anchored deterministically to `(cx, cy, cz, rad)`!
        var qx = s.fma(lx_p, s.float(2.95), s.fmul(cx, s.float(0.37)))
        var qy = s.fsub(s.fma(ly_p, s.float(2.95), s.fmul(cy, s.float(0.37))), s.fmul(rad, s.float(1.35)))
        var qz = s.fma(lz_p, s.float(2.95), s.fmul(cz, s.float(0.37)))
        var bl1 = s.fsub(s.float(1.0), s.fabs(s.fmul(s.fsin(s.fadd(qx, s.fmul(qy, s.float(0.7)))), s.fcos(s.fsub(qz, s.fmul(qx, s.float(0.6)))))))
        var bl2 = s.fsub(s.float(1.0), s.fabs(s.fmul(s.fsin(s.fma(qx, s.float(2.15), qz)), s.fcos(s.fma(qy, s.float(2.35), s.float(1.1))))))
        var bl3 = s.fsub(s.float(1.0), s.fabs(s.fsin(s.fma(qx, s.float(4.6), s.fma(qy, s.float(4.3), s.fmul(qz, s.float(4.1)))))))
        var billow = s.fma(bl1, s.float(0.54), s.fma(bl2, s.float(0.31), s.fmul(bl3, s.float(0.15))))

        var shell = s.nmax(s.float(0.0), s.fsub(s.float(1.0), r_p))
        var eroded = s.nmax(
            s.float(0.0),
            s.fsub(shell, s.fmul(s.fsub(s.float(1.0), billow), s.fma(r_p, s.float(0.34), s.float(0.14)))),
        )
        var step_sig = s.fmul(dens, s.fmul(eroded, s.select_f(is_fire, s.float(4.6), s.float(3.4))))
        var step_tau = s.fmul(step_sig, ds_norm)
        var step_tr = s.exp2(s.fmul(step_tau, s.float(-1.442695)))
        var step_w = s.fmul(s.load_f(tr_p), s.fsub(s.float(1.0), step_tr))

        # Directional 6-Way Beer-Powder self-shadowing at sample point
        var sx6 = s.select_f(s.fge(sun_x, s.float(0.0)), s.fsub(s.float(1.0), lx_p), s.fadd(s.float(1.0), lx_p))
        var sy6 = s.select_f(s.fge(sun_y, s.float(0.0)), s.fsub(s.float(1.0), ly_p), s.fadd(s.float(1.0), ly_p))
        var sz6 = s.select_f(s.fge(sun_z, s.float(0.0)), s.fsub(s.float(1.0), lz_p), s.fadd(s.float(1.0), lz_p))
        var dtau_s = s.fmul(
            s.fma(s.fmul(sun_x, sun_x), sx6, s.fma(s.fmul(sun_y, sun_y), sy6, s.fmul(s.fmul(sun_z, sun_z), sz6))),
            s.fmul(dens, s.float(1.35)),
        )
        var beer_s = s.exp2(s.fneg(s.fmul(dtau_s, s.float(1.442695))))
        var powd_s = s.fsub(s.float(1.0), s.exp2(s.fneg(s.fmul(dtau_s, s.float(2.88539)))))
        var bp_s = s.fmul(s.fma(s.float(1.78), s.fmul(beer_s, powd_s), s.float(0.24)), hg_boost)

        # Fireball Planckian blackbody core (`white-gold -> orange-crimson`) cooling into dark `0.035` carbon oil-soot!
        var temp = s.nclamp(s.fma(s.fsub(s.float(1.04), r_p), s.float(0.80), s.fma(billow, s.float(0.38), s.float(-0.12))), s.float(0.0), s.float(1.30))
        var soot_t = s.nclamp(s.fmul(s.fsub(s.float(0.58), temp), s.float(2.5)), s.float(0.0), s.float(1.0))
        var eff_sc_r = s.select_f(is_fire, s.fma(s.fsub(s.float(0.035), sc_r), soot_t, sc_r), sc_r)
        var eff_sc_g = s.select_f(is_fire, s.fma(s.fsub(s.float(0.033), sc_g), soot_t, sc_g), sc_g)
        var eff_sc_b = s.select_f(is_fire, s.fma(s.fsub(s.float(0.032), sc_b), soot_t, sc_b), sc_b)

        var fire_t2 = s.fmul(temp, temp)
        var fire_core = s.nmax(s.float(0.0), s.fsub(temp, s.float(0.75)))
        var f_em_r = s.select_f(is_fire, s.fma(em_r, s.fmul(fire_t2, s.float(1.45)), s.fmul(fire_core, s.fmul(em_peak, s.float(1.4)))), em_r)
        var f_em_g = s.select_f(is_fire, s.fma(em_g, s.fmul(s.fmul(fire_t2, temp), s.float(1.25)), s.fmul(fire_core, s.fmul(em_peak, s.float(1.1)))), em_g)
        var f_em_b = s.select_f(is_fire, s.fma(em_b, s.fmul(s.fmul(fire_t2, fire_t2), s.float(0.85)), s.fmul(fire_core, s.fmul(em_peak, s.float(0.65)))), em_b)

        var step_cr = s.fma(eff_sc_r, s.fma(sun_r, bp_s, amb_sky_r), f_em_r)
        var step_cg = s.fma(eff_sc_g, s.fma(sun_g, bp_s, amb_sky_g), f_em_g)
        var step_cb = s.fma(eff_sc_b, s.fma(sun_b, bp_s, amb_sky_b), f_em_b)

        s.store(pr_acc, s.fma(step_cr, step_w, s.load_f(pr_acc)))
        s.store(pg_acc, s.fma(step_cg, step_w, s.load_f(pg_acc)))
        s.store(pb_acc, s.fma(step_cb, step_w, s.load_f(pb_acc)))
        s.store(tr_p, s.fmul(s.load_f(tr_p), step_tr))

    var ray_a = s.nclamp(s.fmul(s.fsub(s.float(1.0), s.load_f(tr_p)), soft_fade), s.float(0.0), s.float(0.98))
    s.store(pm_r_v, s.fmul(s.load_f(pr_acc), soft_fade))
    s.store(pm_g_v, s.fmul(s.load_f(pg_acc), soft_fade))
    s.store(pm_b_v, s.fmul(s.load_f(pb_acc), soft_fade))
    s.store(frag_a_v, ray_a)
    s.end_if()

    var pm_r = s.load_f(pm_r_v)
    var pm_g = s.load_f(pm_g_v)
    var pm_b = s.load_f(pm_b_v)
    frag_a = s.load_f(frag_a_v)

    # Insertion-sort `(z_lin, pm_r, pm_g, pm_b, frag_a)` into the 4-node exact OIT stack!
    var iz = s.fvar(z_lin); var ir = s.fvar(pm_r); var ig = s.fvar(pm_g); var ib = s.fvar(pm_b); var ia = s.fvar(frag_a)

    s.begin_if(s.flt(s.load_f(iz), s.load_f(z0)))
    var tz = s.load_f(z0); var tr = s.load_f(r0); var tg = s.load_f(g0); var tb = s.load_f(b0_v); var ta = s.load_f(a0)
    s.store(z0, s.load_f(iz)); s.store(r0, s.load_f(ir)); s.store(g0, s.load_f(ig)); s.store(b0_v, s.load_f(ib)); s.store(a0, s.load_f(ia))
    s.store(iz, tz); s.store(ir, tr); s.store(ig, tg); s.store(ib, tb); s.store(ia, ta)
    s.end_if()

    s.begin_if(s.flt(s.load_f(iz), s.load_f(z1)))
    var tz1 = s.load_f(z1); var tr1 = s.load_f(r1); var tg1 = s.load_f(g1); var tb1 = s.load_f(b1_v); var ta1 = s.load_f(a1)
    s.store(z1, s.load_f(iz)); s.store(r1, s.load_f(ir)); s.store(g1, s.load_f(ig)); s.store(b1_v, s.load_f(ib)); s.store(a1, s.load_f(ia))
    s.store(iz, tz1); s.store(ir, tr1); s.store(ig, tg1); s.store(ib, tb1); s.store(ia, ta1)
    s.end_if()

    s.begin_if(s.flt(s.load_f(iz), s.load_f(z2)))
    var tz2 = s.load_f(z2); var tr2 = s.load_f(r2); var tg2 = s.load_f(g2); var tb2 = s.load_f(b2_v); var ta2 = s.load_f(a2)
    s.store(z2, s.load_f(iz)); s.store(r2, s.load_f(ir)); s.store(g2, s.load_f(ig)); s.store(b2_v, s.load_f(ib)); s.store(a2, s.load_f(ia))
    s.store(iz, tz2); s.store(ir, tr2); s.store(ig, tg2); s.store(ib, tb2); s.store(ia, ta2)
    s.end_if()

    s.begin_if(s.flt(s.load_f(iz), s.load_f(z3)))
    s.store(z3, s.load_f(iz)); s.store(r3, s.load_f(ir)); s.store(g3, s.load_f(ig)); s.store(b3_v, s.load_f(ib)); s.store(a3, s.load_f(ia))
    s.end_if()

    s.end_if()
    s.end_if()
    s.store(pi, s.iadd(k, s.uint(1)))
    s.end_loop()

    # Composite the 4 exact front-to-back OIT layers over opaque HDR color!
    var trans = s.fvar(s.float(1.0))
    var acc_r = s.fvar(s.float(0.0))
    var acc_g = s.fvar(s.float(0.0))
    var acc_b = s.fvar(s.float(0.0))

    var t0 = s.load_f(trans)
    s.store(acc_r, s.fma(s.load_f(r0), t0, s.load_f(acc_r)))
    s.store(acc_g, s.fma(s.load_f(g0), t0, s.load_f(acc_g)))
    s.store(acc_b, s.fma(s.load_f(b0_v), t0, s.load_f(acc_b)))
    s.store(trans, s.fmul(t0, s.fsub(s.float(1.0), s.load_f(a0))))

    var t1 = s.load_f(trans)
    s.store(acc_r, s.fma(s.load_f(r1), t1, s.load_f(acc_r)))
    s.store(acc_g, s.fma(s.load_f(g1), t1, s.load_f(acc_g)))
    s.store(acc_b, s.fma(s.load_f(b1_v), t1, s.load_f(acc_b)))
    s.store(trans, s.fmul(t1, s.fsub(s.float(1.0), s.load_f(a1))))

    var t2 = s.load_f(trans)
    s.store(acc_r, s.fma(s.load_f(r2), t2, s.load_f(acc_r)))
    s.store(acc_g, s.fma(s.load_f(g2), t2, s.load_f(acc_g)))
    s.store(acc_b, s.fma(s.load_f(b2_v), t2, s.load_f(acc_b)))
    s.store(trans, s.fmul(t2, s.fsub(s.float(1.0), s.load_f(a2))))

    var t3 = s.load_f(trans)
    s.store(acc_r, s.fma(s.load_f(r3), t3, s.load_f(acc_r)))
    s.store(acc_g, s.fma(s.load_f(g3), t3, s.load_f(acc_g)))
    s.store(acc_b, s.fma(s.load_f(b3_v), t3, s.load_f(acc_b)))
    s.store(trans, s.fmul(t3, s.fsub(s.float(1.0), s.load_f(a3))))

    var final_trans = s.load_f(trans)
    var bg_r = s.load_float(b_hdr, hdr_base)
    var bg_g = s.load_float(b_hdr, s.iadd(hdr_base, s.uint(1)))
    var bg_b = s.load_float(b_hdr, s.iadd(hdr_base, s.uint(2)))
    s.store_float(b_hdr, hdr_base, s.fma(bg_r, final_trans, s.load_f(acc_r)))
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(1)), s.fma(bg_g, final_trans, s.load_f(acc_g)))
    s.store_float(b_hdr, s.iadd(hdr_base, s.uint(2)), s.fma(bg_b, final_trans, s.load_f(acc_b)))

    # Write reactive mask `max(prev_reactive, 1.0 - transmittance)` for `FlintTSR v3`!
    var oit_react = s.fsub(s.float(1.0), final_trans)
    var prev_react = s.select_f(preserve_reactive, s.load_float(b_oit, react_off), s.float(0.0))
    s.store_float(b_oit, react_off, s.nmax(prev_react, oit_react))
    _ = emit_vec3_cross
    return s.finish()
