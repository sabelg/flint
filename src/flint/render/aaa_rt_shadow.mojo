"""Soft ray-traced sun shadows: trace + penumbra-aware spatial denoise (`aaa_rt_shadow.mojo`).

The sun is an area light. `rt_sun_shadow_trace_shader` sends `AAA_SUN_SHADOW_RAYS` occlusion rays per
pixel over the sun's disc (a Vogel spiral rotated per pixel by interleaved-gradient noise that advances
by the golden ratio each frame) and stores the visibility, the mean occluder distance and the surface
position and normal. `rt_sun_shadow_filter_shader` then estimates the penumbra width from the occluder
distance (`w = d_occluder * tan(cone)`, as PCSS does), and averages a rotated 16-tap disc of that radius
with plane-distance and normal edge-stopping, so contact shadows stay crisp and far penumbrae come out
soft and grain-free; TSR integrates what is left over frames. The deferred shade pass reads the result.

Glass instances (TLAS mask `AAA_RT_MASK_GLASS`) do not block shadow rays; when the frame has glass, a
second ray against glass only scales the light by `b_globals[AAA_GLOBALS_GLASS_SHADOW_WORD]`.

`BIND_AAA_HISTORY` layout (word offsets from `rt_shadow_raw_offset(max_pix)`):
  raw:      8 floats per pixel `(vis, occluder_t, wx, wy, wz, nx, ny, nz)` (`occluder_t <= 0`: none hit,
            `occluder_t < -0.5`: sky / no surface)
  filtered: 1 float per pixel at `rt_shadow_filtered_offset(max_pix)`
"""
from std.math import cos, sin, sqrt
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_PUSH_WORDS,
    AAA_MAX_BINDLESS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_GLOBALS,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_froxel_volumetrics import AAA_FROXEL_GRID_WORDS
from flint.render.aaa_gi_svgf import AAA_FLAG_GLASS
from flint.render.aaa_raytracing import (
    AAA_GLOBALS_GLASS_SHADOW_WORD,
    AAA_GLOBALS_SHADOW_CONE_WORD,
    AAA_RT_MASK_GLASS,
    AAA_RT_MASK_OPAQUE,
    RAY_FLAG_SHADOW_OCCLUSION,
    RAY_FLAG_TERMINATE_ON_FIRST_HIT_KHR,
    _emit_rt_candidate_alpha_loop,
)

comptime AAA_SUN_SHADOW_RAYS: Int = 2  # occlusion rays per pixel over the sun's disc
comptime AAA_RT_SHADOW_RAW_WORDS: Int = 8
comptime AAA_RT_SHADOW_FILTER_TAPS: Int = 16
comptime AAA_RT_SHADOW_MAX_RADIUS_PX: Float32 = 14.0


@always_inline
def rt_shadow_raw_offset(max_pix: Int) -> Int:
    """First word of the raw shadow region in `BIND_AAA_HISTORY` (past SVGF/TSR/ReBLUR and the froxel grids)."""
    return max_pix * 44 + AAA_FROXEL_GRID_WORDS * 3


@always_inline
def rt_shadow_filtered_offset(max_pix: Int) -> Int:
    return rt_shadow_raw_offset(max_pix) + max_pix * AAA_RT_SHADOW_RAW_WORDS


@always_inline
def rt_shadow_total_words(max_pix: Int) -> Int:
    """Words `BIND_AAA_HISTORY` needs past `rt_shadow_raw_offset` for both shadow regions."""
    return max_pix * (AAA_RT_SHADOW_RAW_WORDS + 1)


def penumbra_radius_px_cpu(occluder_t: Float32, cone_tan: Float32, view_z: Float32, tan_half_fov_y: Float32, height_px: Int) -> Float32:
    """CPU reference for the filter radius: penumbra width `d * tan(cone)` over the world size of a pixel."""
    var pix_world = max(Float32(1e-5), view_z * Float32(2.0) * tan_half_fov_y / Float32(max(1, height_px)))
    return max(Float32(0.0), min(AAA_RT_SHADOW_MAX_RADIUS_PX, occluder_t * cone_tan / pix_world))


def _emit_ign(mut s: AaaSpirvModule, px: UInt32, py: UInt32, frame_idx: UInt32) -> UInt32:
    """Interleaved gradient noise (Jimenez 2014) in `[0, 1)`, shifted each frame along the golden ratio."""
    var f = s.fmul(s.float_from_uint(s.bit_and(frame_idx, s.uint(63))), s.float(5.588238))
    var fx = s.fadd(s.float_from_uint(px), f)
    var fy = s.fadd(s.float_from_uint(py), f)
    return s.fract(s.fmul(s.float(52.9829189), s.fract(s.fma(fx, s.float(0.06711056), s.fmul(fy, s.float(0.00583715))))))


def emit_pixel_camera_ray(
    mut s: AaaSpirvModule, b_globals: UInt32, px: UInt32, py: UInt32, fw: UInt32, fh: UInt32
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Camera position and the unit ray through the TSR-jittered pixel center, exactly as the deferred
    shade pass builds it. Returns `(cam_x, cam_y, cam_z, dir_x, dir_y, dir_z)`."""
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
    return (cam_x, cam_y, cam_z, rd[0], rd[1], rd[2])


def emit_visbuffer_surface(
    mut s: AaaSpirvModule,
    b_verts: UInt32,
    b_indices: UInt32,
    b_meshlets: UInt32,
    b_instances: UInt32,
    vis0: UInt32,
    vis1: UInt32,
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Decodes a (non-sky) VisBuffer texel, intersects its triangle with the ray `o + t d` (Moller-Trumbore)
    and returns the world position and the interpolated vertex normal turned toward the ray origin:
    `(wx, wy, wz, nx, ny, nz)`."""
    var inst_id = s.bit_or(
        s.shr(vis0, s.uint(22)),
        s.shl(s.bit_and(s.shr(vis1, s.uint(8)), s.uint(0xFF)), s.uint(10)),
    )
    var ml_id = s.isub(s.bit_and(vis0, s.uint(0x3FFFFF)), s.uint(1))
    var local_prim = s.bit_and(vis1, s.uint(0xFF))
    var first_tri = s.load_word(b_meshlets, s.iadd(s.imul(ml_id, s.uint(12)), s.uint(8)))
    var idx_off = s.imul(s.iadd(first_tri, local_prim), s.uint(3))
    var v0_off = s.imul(s.load_word(b_indices, idx_off), s.uint(12))
    var v1_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(1))), s.uint(12))
    var v2_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(2))), s.uint(12))
    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var p0 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, v0_off), s.load_float(b_verts, s.iadd(v0_off, s.uint(1))), s.load_float(b_verts, s.iadd(v0_off, s.uint(2))))
    var p1 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, v1_off), s.load_float(b_verts, s.iadd(v1_off, s.uint(1))), s.load_float(b_verts, s.iadd(v1_off, s.uint(2))))
    var p2 = emit_mat4_mul_point(s, b_instances, inst_base, s.load_float(b_verts, v2_off), s.load_float(b_verts, s.iadd(v2_off, s.uint(1))), s.load_float(b_verts, s.iadd(v2_off, s.uint(2))))
    var e1 = (s.fsub(p1[0], p0[0]), s.fsub(p1[1], p0[1]), s.fsub(p1[2], p0[2]))
    var e2 = (s.fsub(p2[0], p0[0]), s.fsub(p2[1], p0[1]), s.fsub(p2[2], p0[2]))
    var tv = (s.fsub(ox, p0[0]), s.fsub(oy, p0[1]), s.fsub(oz, p0[2]))
    var pvec = emit_vec3_cross(s, dx, dy, dz, e2[0], e2[1], e2[2])
    var qvec = emit_vec3_cross(s, tv[0], tv[1], tv[2], e1[0], e1[1], e1[2])
    var det = emit_vec3_dot(s, e1[0], e1[1], e1[2], pvec[0], pvec[1], pvec[2])
    var inv_det = s.fdiv(s.float(1.0), s.select_f(s.fge(det, s.float(0.0)), s.nmax(det, s.float(1e-12)), s.nmin(det, s.float(-1e-12))))
    var b1 = s.nclamp(s.fmul(emit_vec3_dot(s, tv[0], tv[1], tv[2], pvec[0], pvec[1], pvec[2]), inv_det), s.float(0.0), s.float(1.0))
    var b2 = s.nclamp(s.fmul(emit_vec3_dot(s, dx, dy, dz, qvec[0], qvec[1], qvec[2]), inv_det), s.float(0.0), s.float(1.0))
    var b0 = s.fsub(s.fsub(s.float(1.0), b1), b2)
    var wx = s.fma(p0[0], b0, s.fma(p1[0], b1, s.fmul(p2[0], b2)))
    var wy = s.fma(p0[1], b0, s.fma(p1[1], b1, s.fmul(p2[1], b2)))
    var wz = s.fma(p0[2], b0, s.fma(p1[2], b1, s.fmul(p2[2], b2)))

    var lnx = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(3))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(3))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(3))), b2)))
    var lny = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(4))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(4))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(4))), b2)))
    var lnz = s.fma(s.load_float(b_verts, s.iadd(v0_off, s.uint(5))), b0, s.fma(s.load_float(b_verts, s.iadd(v1_off, s.uint(5))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2_off, s.uint(5))), b2)))
    var wn_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lnx, lny, lnz)
    var wn = emit_vec3_normalize(s, wn_raw[0], wn_raw[1], wn_raw[2])
    var flip = s.select_f(s.fgt(emit_vec3_dot(s, wn[0], wn[1], wn[2], dx, dy, dz), s.float(0.0)), s.float(-1.0), s.float(1.0))
    return (wx, wy, wz, s.fmul(wn[0], flip), s.fmul(wn[1], flip), s.fmul(wn[2], flip))


def rt_sun_shadow_trace_shader() -> List[UInt32]:
    """Per-pixel area-sun occlusion (`group = 8 x 8`). Push[0] = render_w, [1] = render_h, [2] = raw offset."""
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_ray_query=True,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var raw_off = s.push_word(2)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var o = s.iadd(raw_off, s.imul(pix, s.uint(UInt32(AAA_RT_SHADOW_RAW_WORDS))))

    var vis_uv4 = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var vis0 = s.extract_u(vis_uv4, 0)
    var vis1 = s.extract_u(vis_uv4, 1)

    s.begin_if(s.ieq(vis0, s.uint(0)))
    s.store_float(b_hist, o, s.float(1.0))
    s.store_float(b_hist, s.iadd(o, s.uint(1)), s.float(-1.0))
    s.begin_else()

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cam = emit_pixel_camera_ray(s, b_globals, px, py, fw, fh)
    var cam_x = cam[0]
    var cam_y = cam[1]
    var cam_z = cam[2]
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var r_flags = s.load_word(b_globals, s.uint(99))
    var frame_idx = s.load_word(b_globals, s.uint(100))
    var surf = emit_visbuffer_surface(s, b_verts, b_indices, b_meshlets, b_instances, vis0, vis1, cam_x, cam_y, cam_z, cam[3], cam[4], cam[5])
    var wx = surf[0]
    var wy = surf[1]
    var wz = surf[2]
    var nx = surf[3]
    var ny = surf[4]
    var nz = surf[5]

    # Orthonormal basis (tb, bb) around the sun direction
    var h_up = s.flt(s.fabs(sun_y), s.float(0.95))
    var hx = s.select_f(h_up, s.float(0.0), s.float(1.0))
    var hy = s.select_f(h_up, s.float(1.0), s.float(0.0))
    var tb = emit_vec3_normalize(s, s.fneg(s.fmul(hy, sun_z)), s.fmul(hx, sun_z), s.fsub(s.fmul(hy, sun_x), s.fmul(hx, sun_y)))
    var bb = emit_vec3_cross(s, sun_x, sun_y, sun_z, tb[0], tb[1], tb[2])

    var rot = s.fmul(_emit_ign(s, px, py, frame_idx), s.float(6.2831853))
    var rot_c = s.fcos(rot)
    var rot_s = s.fsin(rot)
    var cone_tan = s.nclamp(s.load_float(b_globals, s.uint(UInt32(AAA_GLOBALS_SHADOW_CONE_WORD))), s.float(0.0), s.float(0.2))
    var has_glass = s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_GLASS)), s.uint(0))
    var glass_tr = s.nclamp(s.load_float(b_globals, s.uint(UInt32(AAA_GLOBALS_GLASS_SHADOW_WORD))), s.float(0.0), s.float(1.0))

    # Offset along the normal, more at grazing light and with distance (keeps acne off curved low-poly surfaces)
    var cam_d = s.fsqrt(s.fma(s.fsub(wx, cam_x), s.fsub(wx, cam_x), s.fma(s.fsub(wy, cam_y), s.fsub(wy, cam_y), s.fmul(s.fsub(wz, cam_z), s.fsub(wz, cam_z)))))
    var bias = s.fma(cam_d, s.float(0.0006), s.float(0.004))
    var ro = s.vec3(s.fma(nx, bias, wx), s.fma(ny, bias, wy), s.fma(nz, bias, wz))
    var vis_sum = s.fvar(s.float(0.0))
    var occ_t_sum = s.fvar(s.float(0.0))
    var occ_n = s.fvar(s.float(0.0))
    for k in range(AAA_SUN_SHADOW_RAYS):
        var rk = sqrt((Float32(k) + Float32(0.5)) / Float32(AAA_SUN_SHADOW_RAYS))
        var ak = Float32(k) * Float32(2.39996323)
        var ck = s.float(rk * cos(ak))
        var sk = s.float(rk * sin(ak))
        var du = s.fmul(s.fsub(s.fmul(ck, rot_c), s.fmul(sk, rot_s)), cone_tan)
        var dv = s.fmul(s.fma(ck, rot_s, s.fmul(sk, rot_c)), cone_tan)
        var dir = emit_vec3_normalize(
            s,
            s.fma(tb[0], du, s.fma(bb[0], dv, sun_x)),
            s.fma(tb[1], du, s.fma(bb[1], dv, sun_y)),
            s.fma(tb[2], du, s.fma(bb[2], dv, sun_z)),
        )
        var rdir = s.vec3(dir[0], dir[1], dir[2])
        var rq = s.ray_query_var()
        s.ray_query_init(rq, tlas_var, s.uint(RAY_FLAG_TERMINATE_ON_FIRST_HIT_KHR), s.uint(AAA_RT_MASK_OPAQUE), ro, s.float(0.002), rdir, s.float(500.0))
        _emit_rt_candidate_alpha_loop(s, rq, b_verts, b_indices, b_instances, b_materials, bindless_arr)
        var hit = s.ine(s.ray_query_committed_type(rq), s.uint(0))
        var v_k = s.fvar(s.select_f(hit, s.float(0.0), s.float(1.0)))
        s.begin_if(hit)
        s.store(occ_t_sum, s.fadd(s.load_f(occ_t_sum), s.ray_query_committed_t(rq)))
        s.store(occ_n, s.fadd(s.load_f(occ_n), s.float(1.0)))
        s.end_if()

        s.begin_if(s.logical_and(has_glass, s.logical_not(hit)))
        var rq_g = s.ray_query_var()
        s.ray_query_init(rq_g, tlas_var, s.uint(RAY_FLAG_SHADOW_OCCLUSION), s.uint(AAA_RT_MASK_GLASS), ro, s.float(0.002), rdir, s.float(500.0))
        s.begin_loop()
        s.break_if(s.logical_not(s.ray_query_proceed(rq_g)))
        s.end_loop()
        var g_hit = s.ine(s.ray_query_committed_type(rq_g), s.uint(0))
        s.store(v_k, s.select_f(g_hit, glass_tr, s.float(1.0)))
        s.begin_if(g_hit)
        s.store(occ_t_sum, s.fadd(s.load_f(occ_t_sum), s.ray_query_committed_t(rq_g)))
        s.store(occ_n, s.fadd(s.load_f(occ_n), s.float(1.0)))
        s.end_if()
        s.end_if()
        s.store(vis_sum, s.fadd(s.load_f(vis_sum), s.load_f(v_k)))

    var n_occ = s.load_f(occ_n)
    var occ_t = s.select_f(s.fgt(n_occ, s.float(0.5)), s.fdiv(s.load_f(occ_t_sum), s.nmax(n_occ, s.float(1.0))), s.float(0.0))
    s.store_float(b_hist, o, s.fmul(s.load_f(vis_sum), s.float(1.0 / Float32(AAA_SUN_SHADOW_RAYS))))
    s.store_float(b_hist, s.iadd(o, s.uint(1)), occ_t)
    s.store_float(b_hist, s.iadd(o, s.uint(2)), wx)
    s.store_float(b_hist, s.iadd(o, s.uint(3)), wy)
    s.store_float(b_hist, s.iadd(o, s.uint(4)), wz)
    s.store_float(b_hist, s.iadd(o, s.uint(5)), nx)
    s.store_float(b_hist, s.iadd(o, s.uint(6)), ny)
    s.store_float(b_hist, s.iadd(o, s.uint(7)), nz)
    s.end_if()
    return s.finish()


def rt_sun_shadow_filter_shader() -> List[UInt32]:
    """Penumbra-sized, edge-stopped disc filter (`group = 8 x 8`).
    Push[0] = render_w, [1] = render_h, [2] = raw offset, [3] = filtered offset."""
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
    )
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hist = s.buffer(BIND_AAA_HISTORY)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var raw_off = s.push_word(2)
    var flt_off = s.push_word(3)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var o = s.iadd(raw_off, s.imul(pix, s.uint(UInt32(AAA_RT_SHADOW_RAW_WORDS))))
    var vis_c = s.load_float(b_hist, o)
    var t_c = s.load_float(b_hist, s.iadd(o, s.uint(1)))
    var out_v = s.fvar(vis_c)

    s.begin_if(s.fgt(t_c, s.float(-0.5)))
    var cwx = s.load_float(b_hist, s.iadd(o, s.uint(2)))
    var cwy = s.load_float(b_hist, s.iadd(o, s.uint(3)))
    var cwz = s.load_float(b_hist, s.iadd(o, s.uint(4)))
    var cnx = s.load_float(b_hist, s.iadd(o, s.uint(5)))
    var cny = s.load_float(b_hist, s.iadd(o, s.uint(6)))
    var cnz = s.load_float(b_hist, s.iadd(o, s.uint(7)))
    var fh = s.float_from_uint(h_u)
    var dxc = s.fsub(cwx, s.load_float(b_globals, s.uint(32)))
    var dyc = s.fsub(cwy, s.load_float(b_globals, s.uint(33)))
    var dzc = s.fsub(cwz, s.load_float(b_globals, s.uint(34)))
    var cam_d = s.fsqrt(s.fma(dxc, dxc, s.fma(dyc, dyc, s.fmul(dzc, dzc))))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var pix_world = s.nmax(s.float(1e-5), s.fdiv(s.fmul(cam_d, s.fmul(s.float(2.0), tan_y)), fh))
    var cone_tan = s.nclamp(s.load_float(b_globals, s.uint(UInt32(AAA_GLOBALS_SHADOW_CONE_WORD))), s.float(0.0), s.float(0.2))
    var frame_idx = s.load_word(b_globals, s.uint(100))
    var rot = s.fmul(_emit_ign(s, py, px, frame_idx), s.float(6.2831853))
    var rot_c = s.fcos(rot)
    var rot_s = s.fsin(rot)
    var plane_tol = s.fma(pix_world, s.float(3.0), s.float(0.004))

    # 1. Occluder distance: the center's own rays, else an 8-tap search over a fixed footprint
    var occ_sum = s.fvar(s.select_f(s.fgt(t_c, s.float(0.0)), s.fmul(t_c, s.float(2.0)), s.float(0.0)))
    var occ_w = s.fvar(s.select_f(s.fgt(t_c, s.float(0.0)), s.float(2.0), s.float(0.0)))
    var search_px = s.float(AAA_RT_SHADOW_MAX_RADIUS_PX * Float32(0.6))
    for k in range(8):
        var rk = sqrt((Float32(k) + Float32(0.5)) / Float32(8.0))
        var ak = Float32(k) * Float32(2.39996323)
        var ox = s.fmul(s.fsub(s.fmul(s.float(rk * cos(ak)), rot_c), s.fmul(s.float(rk * sin(ak)), rot_s)), search_px)
        var oy = s.fmul(s.fma(s.float(rk * cos(ak)), rot_s, s.fmul(s.float(rk * sin(ak)), rot_c)), search_px)
        var sx = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.fadd(s.float_from_uint(px), s.fadd(ox, s.float(0.5)))))
        var sy = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.fadd(s.float_from_uint(py), s.fadd(oy, s.float(0.5)))))
        var so = s.iadd(raw_off, s.imul(s.iadd(s.imul(sy, w_u), sx), s.uint(UInt32(AAA_RT_SHADOW_RAW_WORDS))))
        var t_s = s.load_float(b_hist, s.iadd(so, s.uint(1)))
        var ok = s.fgt(t_s, s.float(0.0))
        s.store(occ_sum, s.fadd(s.load_f(occ_sum), s.select_f(ok, t_s, s.float(0.0))))
        s.store(occ_w, s.fadd(s.load_f(occ_w), s.select_f(ok, s.float(1.0), s.float(0.0))))

    var ow = s.load_f(occ_w)
    s.begin_if(s.fgt(ow, s.float(0.5)))
    var occ_t = s.fdiv(s.load_f(occ_sum), ow)
    var radius = s.nclamp(s.fdiv(s.fmul(occ_t, cone_tan), pix_world), s.float(0.0), s.float(AAA_RT_SHADOW_MAX_RADIUS_PX))
    s.begin_if(s.fgt(radius, s.float(0.75)))
    # 2. Rotated 16-tap Vogel disc of the penumbra radius, plane-distance + normal weighted
    var acc = s.fvar(s.fmul(vis_c, s.float(1.0)))
    var wsum = s.fvar(s.float(1.0))
    for k in range(AAA_RT_SHADOW_FILTER_TAPS):
        var rk = sqrt((Float32(k) + Float32(0.5)) / Float32(AAA_RT_SHADOW_FILTER_TAPS))
        var ak = Float32(k) * Float32(2.39996323) + Float32(1.1)
        var ox = s.fmul(s.fsub(s.fmul(s.float(rk * cos(ak)), rot_c), s.fmul(s.float(rk * sin(ak)), rot_s)), radius)
        var oy = s.fmul(s.fma(s.float(rk * cos(ak)), rot_s, s.fmul(s.float(rk * sin(ak)), rot_c)), radius)
        var sx = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.fadd(s.float_from_uint(px), s.fadd(ox, s.float(0.5)))))
        var sy = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.fadd(s.float_from_uint(py), s.fadd(oy, s.float(0.5)))))
        var so = s.iadd(raw_off, s.imul(s.iadd(s.imul(sy, w_u), sx), s.uint(UInt32(AAA_RT_SHADOW_RAW_WORDS))))
        var t_s = s.load_float(b_hist, s.iadd(so, s.uint(1)))
        var pdx = s.fsub(s.load_float(b_hist, s.iadd(so, s.uint(2))), cwx)
        var pdy = s.fsub(s.load_float(b_hist, s.iadd(so, s.uint(3))), cwy)
        var pdz = s.fsub(s.load_float(b_hist, s.iadd(so, s.uint(4))), cwz)
        var plane_d = s.fabs(emit_vec3_dot(s, pdx, pdy, pdz, cnx, cny, cnz))
        var n_dot = emit_vec3_dot(
            s, s.load_float(b_hist, s.iadd(so, s.uint(5))), s.load_float(b_hist, s.iadd(so, s.uint(6))), s.load_float(b_hist, s.iadd(so, s.uint(7))), cnx, cny, cnz
        )
        var w_plane = s.nclamp(s.fsub(s.float(1.0), s.fdiv(plane_d, plane_tol)), s.float(0.0), s.float(1.0))
        var n_c = s.nclamp(n_dot, s.float(0.0), s.float(1.0))
        var n2 = s.fmul(n_c, n_c)
        var n4 = s.fmul(n2, n2)
        var w_k = s.select_f(s.fgt(t_s, s.float(-0.5)), s.fmul(w_plane, s.fmul(n4, n4)), s.float(0.0))
        s.store(acc, s.fma(s.load_float(b_hist, so), w_k, s.load_f(acc)))
        s.store(wsum, s.fadd(s.load_f(wsum), w_k))
    s.store(out_v, s.fdiv(s.load_f(acc), s.load_f(wsum)))
    s.end_if()
    s.end_if()
    s.end_if()
    s.store_float(b_hist, s.iadd(flt_off, pix), s.load_f(out_v))
    return s.finish()
