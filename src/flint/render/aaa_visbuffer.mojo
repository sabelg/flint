"""Milestone 1 (`V12`): Vulkan 1.3 Hardware Graphics `VK_KHR_dynamic_rendering`
64-Bit Visibility Buffer, GPU Meshlet Compute Culler & Bindless `BC7/BC5` Resolve.

Architecture:
  1. `meshlet_cull_shader()` (Compute): Evaluates `(Instance, Meshlet)` pairs against
     the view frustum, spherical backface normal cone, and Reverse-Z depth pyramid,
     emitting compact `VkDrawIndirectCommand` records + atomic count for
     `vkCmdDrawIndirectCount`.
  2. `visbuffer_vertex_shader()` + `visbuffer_fragment_shader()` (Hardware Graphics):
     Vertex-pulling Reverse-Z `[1 -> 0]` hardware rasterizer writing lossless 64-bit
     `( (inst_id << 16) | (meshlet_id + 1), (material_id << 16) | local_prim_id )`
     into `VK_FORMAT_R32G32_UINT` + `VK_FORMAT_D32_SFLOAT` (`EarlyFragmentTests`).
  3. Analytic Ray-Triangle Differential Helper (`emit_ray_barycentrics`):
     Computes exact perspective-correct barycentrics at `(x, y)`, `(x+1, y)`, and
     `(x, y+1)` to drive `OpImageSampleExplicitLod(..., Grad, ddx_uv, ddy_uv)` on
     bindless `16x` anisotropic `BC7` / `BC5` / `BC6H` `VkImage`s with zero quad overdraw!
"""
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_GL_COMPUTE,
    MODEL_VERTEX,
    MODEL_FRAGMENT,
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    OP_LOAD,
    OP_STORE,
    emit,
)

# Unified AAA Descriptor Set 0 Layout
comptime BIND_AAA_VERTICES: Int = 0      # AAA_VERTEX_WORDS (12) per vertex
comptime BIND_AAA_INDICES: Int = 1       # 3 UInt32s per triangle
comptime BIND_AAA_MESHLETS: Int = 2      # AAA_MESHLET_WORDS (12) per meshlet
comptime BIND_AAA_INSTANCES: Int = 3     # AAA_INSTANCE_WORDS (36) per instance (cur Mat4 + prev Mat4 + meshlet range)
comptime BIND_AAA_MATERIALS: Int = 4     # AAA_MATERIAL_WORDS (24) per enterprise BSDF material
comptime BIND_AAA_INDIRECT: Int = 5      # Word 0 = atomic draw count; Word 4+ = VkDrawIndirectCommand[N] (4 words each)
comptime BIND_AAA_GLOBALS: Int = 6       # Camera basis, ViewProj, Light, VSM matrix, SH9 irradiance, flags
comptime BIND_AAA_HDR_OUT: Int = 7       # Linear RGBA32F (4 floats/pixel) HDR resolve buffer + moments
comptime BIND_AAA_HISTORY: Int = 8       # TSR v2 / SVGF history RGBA32F + lock confidence
comptime BIND_AAA_VSM_TABLE: Int = 9     # 128x128 Virtual Shadow Map page table + request flags
comptime BIND_AAA_SURFCACHE: Int = 10    # Direct-lit Lumen-style per-triangle/meshlet Surface Cache RGBE/Float3
comptime BIND_AAA_OIT_NODES: Int = 11    # Per-pixel 4-layer exact short-stack OIT + 6-way volumetric plumes
comptime AAA_STORAGE_COUNT: Int = 12

comptime BIND_AAA_VIS_TEX: Int = 12      # usampler2D (VK_FORMAT_R32G32_UINT VisBuffer)
comptime BIND_AAA_DEPTH_TEX: Int = 13    # sampler2D  (VK_FORMAT_D32_SFLOAT Reverse-Z Depth)
comptime BIND_AAA_VSM_TEX: Int = 14      # sampler2D  (VK_FORMAT_R32_SFLOAT / D32_SFLOAT Physical VSM Page Pool)
comptime BIND_AAA_ENV_CUBE: Int = 15     # samplerCube (VK_FORMAT_BC6H_UFLOAT 6-Mip Split-Sum GGX Cubemap)
comptime BIND_AAA_BRDF_LUT: Int = 16     # sampler2D  (VK_FORMAT_R16G16_SFLOAT Split-Sum Multi-Scatter LUT)
comptime AAA_SINGLE_SAMPLERS: Int = 5

comptime BIND_AAA_BINDLESS_TEX: Int = 17 # sampler2D[AAA_MAX_BINDLESS] (BC7_UNORM, BC7_SRGB, BC5_UNORM, BC6H_UFLOAT)
comptime AAA_MAX_BINDLESS: Int = 256
comptime BIND_AAA_TLAS: Int = 18         # accelerationStructureKHR (Hardware BLAS/TLAS for SPV_KHR_ray_query)

comptime AAA_PUSH_WORDS: Int = 16
comptime AAA_PUSH_BYTES: Int = 64

# Instance SSBO layout: 36 words (144 bytes) per instance
# [0..15]:  model_to_world Mat4 (column-major Float32[16])
# [16..31]: prev_model_to_world Mat4 (column-major Float32[16], for 3D per-pixel TSR/SVGF motion vectors!)
# [32]:     first_meshlet (UInt32)
# [33]:     meshlet_count (UInt32)
# [34]:     material_override (UInt32: 0xFFFFFFFF = use meshlet.material_id)
# [35]:     uniform_scale_bits (Float32: max axis scale for bounding sphere radius)
comptime AAA_INSTANCE_WORDS: Int = 36

# Globals SSBO layout (`BIND_AAA_GLOBALS`, 128 words = 512 bytes):
# [0..15]:  view_proj Mat4 (Reverse-Z column-major Float32[16])
# [16..31]: prev_view_proj Mat4 (for 3D motion vectors)
# [32..34]: cam_pos (x, y, z)
# [35]:     near_z
# [36..38]: cam_fwd (x, y, z)
# [39]:     tan_half_fov_x
# [40..42]: cam_right (x, y, z)
# [43]:     tan_half_fov_y
# [44..46]: cam_up (x, y, z)
# [47]:     exposure
# [48..50]: sun_dir (toward light, unit Vec3: x, y, z)
# [51]:     sun_angular_radius (for PCSS blocker-search penumbra & RT soft shadow cone)
# [52..54]: sun_radiance (linear RGB)
# [55]:     ibl_intensity
# [56..71]: light_view_proj Mat4 (for 16k Virtual Shadow Map projection)
# [72..98]: SH9 diffuse irradiance (9 x Vec3 = 27 floats: [72..98])
# [99]:     render_flags (bit0=VSM, bit1=RT_Shadows, bit2=RT_Reflections, bit3=RT_SurfaceGI, bit4=Furnace_WhiteDome)
# [100]:    frame_index (UInt32)
# [101]:    jitter_x (Float32, pixels)
# [102]:    jitter_y (Float32, pixels)
# [103]:    env_mip_count (Float32, e.g. 5.0 for 6 mips 0..5)


def emit_vec3_dot(
    mut s: AaaSpirvModule,
    ax: UInt32,
    ay: UInt32,
    az: UInt32,
    bx: UInt32,
    by: UInt32,
    bz: UInt32,
) -> UInt32:
    return s.fma(ax, bx, s.fma(ay, by, s.fmul(az, bz)))


def emit_vec3_cross(
    mut s: AaaSpirvModule,
    ax: UInt32,
    ay: UInt32,
    az: UInt32,
    bx: UInt32,
    by: UInt32,
    bz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    var cx = s.fsub(s.fmul(ay, bz), s.fmul(az, by))
    var cy = s.fsub(s.fmul(az, bx), s.fmul(ax, bz))
    var cz = s.fsub(s.fmul(ax, by), s.fmul(ay, bx))
    return (cx, cy, cz)


def emit_vec3_normalize(
    mut s: AaaSpirvModule,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    var len2 = s.nmax(s.float(1e-12), emit_vec3_dot(s, vx, vy, vz, vx, vy, vz))
    var inv_l = s.inverse_sqrt(len2)
    return (s.fmul(vx, inv_l), s.fmul(vy, inv_l), s.fmul(vz, inv_l))


def emit_mat4_mul_point(
    mut s: AaaSpirvModule,
    buf: UInt32,
    mat_base: UInt32,
    px: UInt32,
    py: UInt32,
    pz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Column-major `Mat4 * vec4(px, py, pz, 1.0)`."""
    var m0 = s.load_float(buf, mat_base)
    var m1 = s.load_float(buf, s.iadd(mat_base, s.uint(1)))
    var m2 = s.load_float(buf, s.iadd(mat_base, s.uint(2)))
    var m3 = s.load_float(buf, s.iadd(mat_base, s.uint(3)))
    var m4 = s.load_float(buf, s.iadd(mat_base, s.uint(4)))
    var m5 = s.load_float(buf, s.iadd(mat_base, s.uint(5)))
    var m6 = s.load_float(buf, s.iadd(mat_base, s.uint(6)))
    var m7 = s.load_float(buf, s.iadd(mat_base, s.uint(7)))
    var m8 = s.load_float(buf, s.iadd(mat_base, s.uint(8)))
    var m9 = s.load_float(buf, s.iadd(mat_base, s.uint(9)))
    var m10 = s.load_float(buf, s.iadd(mat_base, s.uint(10)))
    var m11 = s.load_float(buf, s.iadd(mat_base, s.uint(11)))
    var m12 = s.load_float(buf, s.iadd(mat_base, s.uint(12)))
    var m13 = s.load_float(buf, s.iadd(mat_base, s.uint(13)))
    var m14 = s.load_float(buf, s.iadd(mat_base, s.uint(14)))
    var m15 = s.load_float(buf, s.iadd(mat_base, s.uint(15)))

    var rx = s.fma(m0, px, s.fma(m4, py, s.fma(m8, pz, m12)))
    var ry = s.fma(m1, px, s.fma(m5, py, s.fma(m9, pz, m13)))
    var rz = s.fma(m2, px, s.fma(m6, py, s.fma(m10, pz, m14)))
    var rw = s.fma(m3, px, s.fma(m7, py, s.fma(m11, pz, m15)))
    return (rx, ry, rz, rw)


def emit_mat4_mul_dir(
    mut s: AaaSpirvModule,
    buf: UInt32,
    mat_base: UInt32,
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Column-major upper `3x3 * vec3(dx, dy, dz)`."""
    var m0 = s.load_float(buf, mat_base)
    var m1 = s.load_float(buf, s.iadd(mat_base, s.uint(1)))
    var m2 = s.load_float(buf, s.iadd(mat_base, s.uint(2)))
    var m4 = s.load_float(buf, s.iadd(mat_base, s.uint(4)))
    var m5 = s.load_float(buf, s.iadd(mat_base, s.uint(5)))
    var m6 = s.load_float(buf, s.iadd(mat_base, s.uint(6)))
    var m8 = s.load_float(buf, s.iadd(mat_base, s.uint(8)))
    var m9 = s.load_float(buf, s.iadd(mat_base, s.uint(9)))
    var m10 = s.load_float(buf, s.iadd(mat_base, s.uint(10)))

    var rx = s.fma(m0, dx, s.fma(m4, dy, s.fmul(m8, dz)))
    var ry = s.fma(m1, dx, s.fma(m5, dy, s.fmul(m9, dz)))
    var rz = s.fma(m2, dx, s.fma(m6, dy, s.fmul(m10, dz)))
    return (rx, ry, rz)


def meshlet_cull_shader() -> List[UInt32]:
    """GPU Compute Meshlet Culler (`Frustum + Normal-Cone Backface + Reverse-Z Hi-Z` -> `VkDrawIndirectCommand`).
    Push[0] = instance_count, Push[1] = max_draws, Push[2] = enable_cone_cull (1/0), Push[3] = enable_hiz (1/0),
    Push[4] = screen_w, Push[5] = screen_h.
    Invocation X = instance_idx, Invocation Y = local_meshlet_idx!
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_indirect = s.buffer(BIND_AAA_INDIRECT)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)

    var inst_idx = s.invocation_axis(0)
    var local_ml = s.invocation_axis(1)
    var inst_count = s.push_word(0)
    var max_draws = s.push_word(1)
    var enable_cone = s.push_word(2)
    var enable_hiz = s.push_word(3)
    var scr_w = s.float_from_uint(s.push_word(4))
    var scr_h = s.float_from_uint(s.push_word(5))

    s.return_if(s.uge(inst_idx, inst_count))

    var inst_base = s.imul(inst_idx, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var first_ml = s.load_word(b_instances, s.iadd(inst_base, s.uint(32)))
    var ml_count = s.load_word(b_instances, s.iadd(inst_base, s.uint(33)))
    s.return_if(s.uge(local_ml, ml_count))

    var ml_id = s.iadd(first_ml, local_ml)
    var ml_base = s.imul(ml_id, s.uint(12))
    var lcx = s.load_float(b_meshlets, ml_base)
    var lcy = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(1)))
    var lcz = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(2)))
    var lrad = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(3)))
    var lax = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(4)))
    var lay = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(5)))
    var laz = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(6)))
    var cone_sin = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(7)))
    var tri_cnt = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(9)))

    var uscale = s.load_float(b_instances, s.iadd(inst_base, s.uint(35)))
    var wc = emit_mat4_mul_point(s, b_instances, inst_base, lcx, lcy, lcz)
    var wrad = s.fmul(lrad, uscale)

    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var near_z = s.load_float(b_globals, s.uint(35))
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

    var rx = s.fsub(wc[0], cam_x)
    var ry = s.fsub(wc[1], cam_y)
    var rz = s.fsub(wc[2], cam_z)
    var dist = s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, rx, ry, rz, rx, ry, rz)))

    # 1. Backface normal-cone cull
    s.begin_if(s.logical_and(s.ine(enable_cone, s.uint(0)), s.flt(cone_sin, s.float(1.5))))
    var waxis_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lax, lay, laz)
    var waxis = emit_vec3_normalize(s, waxis_raw[0], waxis_raw[1], waxis_raw[2])
    var cdot = emit_vec3_dot(s, rx, ry, rz, waxis[0], waxis[1], waxis[2])
    var thresh = s.fma(cone_sin, dist, wrad)
    s.return_if(s.fge(cdot, thresh))
    s.end_if()

    # 2. View-frustum bounding-sphere test
    var vz = emit_vec3_dot(s, rx, ry, rz, fwd_x, fwd_y, fwd_z)
    s.return_if(s.flt(s.fadd(vz, wrad), near_z))

    var vx = s.fabs(emit_vec3_dot(s, rx, ry, rz, rgt_x, rgt_y, rgt_z))
    var vy = s.fabs(emit_vec3_dot(s, rx, ry, rz, up_x, up_y, up_z))
    var sec_x = s.fsqrt(s.fma(tan_x, tan_x, s.float(1.0)))
    var sec_y = s.fsqrt(s.fma(tan_y, tan_y, s.float(1.0)))
    var vz_pos = s.nmax(s.float(0.0), vz)
    s.return_if(s.fgt(s.fsub(vx, s.fmul(wrad, sec_x)), s.fmul(vz_pos, tan_x)))
    s.return_if(s.fgt(s.fsub(vy, s.fmul(wrad, sec_y)), s.fmul(vz_pos, tan_y)))

    # 3. Optional Reverse-Z 5-Tap Depth Occlusion Test against previous pass/frame depth
    s.begin_if(s.logical_and(s.ine(enable_hiz, s.uint(0)), s.fgt(s.fsub(vz, wrad), near_z)))
    var clip = emit_mat4_mul_point(s, b_globals, s.uint(0), wc[0], wc[1], wc[2])
    s.begin_if(s.fgt(clip[3], near_z))
    var inv_w = s.fdiv(s.float(1.0), clip[3])
    var ndc_x = s.fmul(clip[0], inv_w)
    var ndc_y = s.fmul(clip[1], inv_w)
    var u_c = s.fma(ndc_x, s.float(0.5), s.float(0.5))
    var v_c = s.fma(s.fneg(ndc_y), s.float(0.5), s.float(0.5))
    var in_uv = s.logical_and(
        s.logical_and(s.fgt(u_c, s.float(0.02)), s.flt(u_c, s.float(0.98))),
        s.logical_and(s.fgt(v_c, s.float(0.02)), s.flt(v_c, s.float(0.98))),
    )
    s.begin_if(in_uv)
    var front_z = s.nmax(near_z, s.fsub(vz, wrad))
    var sphere_rev_z = s.fdiv(near_z, front_z)
    var uv_r_x = s.fdiv(wrad, s.nmax(s.float(0.001), s.fmul(vz, tan_x)))
    var uv_r_y = s.fdiv(wrad, s.nmax(s.float(0.001), s.fmul(vz, tan_y)))
    var d0 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(u_c, v_c), s.float(0.0)), 0)
    var d1 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fsub(u_c, uv_r_x), s.fsub(v_c, uv_r_y)), s.float(0.0)), 0)
    var d2 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fadd(u_c, uv_r_x), s.fsub(v_c, uv_r_y)), s.float(0.0)), 0)
    var d3 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fsub(u_c, uv_r_x), s.fadd(v_c, uv_r_y)), s.float(0.0)), 0)
    var d4 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fadd(u_c, uv_r_x), s.fadd(v_c, uv_r_y)), s.float(0.0)), 0)
    # In Reverse-Z, a closer occluder has LARGER depth; meshlet is occluded iff ALL 5 taps > sphere_rev_z!
    var min_occ_z = s.nmin(d0, s.nmin(s.nmin(d1, d2), s.nmin(d3, d4)))
    s.return_if(s.fgt(min_occ_z, s.fadd(sphere_rev_z, s.float(0.0005))))
    s.end_if()
    s.end_if()
    s.end_if()
    _ = scr_w
    _ = scr_h

    # 4. Append `VkDrawIndirectCommand` (`vertexCount, instanceCount, firstVertex, firstInstance`)
    var slot = s.atomic_add_word(b_indirect, s.uint(0), s.uint(1))
    s.begin_if(s.ult(slot, max_draws))
    var cmd_base = s.iadd(s.uint(4), s.imul(slot, s.uint(4)))
    var vtx_cnt = s.imul(tri_cnt, s.uint(3))
    var packed_first_inst = s.bit_or(s.shl(inst_idx, s.uint(16)), s.bit_and(ml_id, s.uint(0xFFFF)))
    s.store_word(b_indirect, cmd_base, vtx_cnt)
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(1)), s.uint(1))
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(2)), s.uint(0))
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(3)), packed_first_inst)
    s.end_if()

    return s.finish()


def visbuffer_vertex_shader() -> List[UInt32]:
    """Hardware Vertex Shader for 64-Bit Visibility Buffer (`VK_KHR_dynamic_rendering`).
    Pulls triangle indices & 48-byte vertices directly from SSBOs and outputs flat `uvec2` payload!
    Push[0] = view_proj word offset in `BIND_AAA_GLOBALS` (0 = main camera, 56 = VSM light matrix!).
    """
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var out_vis = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec2, location=0, flat=True)

    var v_idx = s.vertex_index()
    var packed_inst = s.instance_index()
    var inst_id = s.shr(packed_inst, s.uint(16))
    var ml_id = s.bit_and(packed_inst, s.uint(0xFFFF))

    var local_prim = s.udiv(v_idx, s.uint(3))
    var corner = s.umod(v_idx, s.uint(3))

    var ml_base = s.imul(ml_id, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var ml_mat = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(11)))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var mat_override = s.load_word(b_instances, s.iadd(inst_base, s.uint(34)))
    var final_mat = s.select_u(s.ieq(mat_override, s.uint(0xFFFFFFFF)), ml_mat, mat_override)

    var global_tri = s.iadd(first_tri, local_prim)
    var idx_word_off = s.iadd(s.imul(global_tri, s.uint(3)), corner)
    var real_vtx = s.load_word(b_indices, idx_word_off)
    var v_base = s.imul(real_vtx, s.uint(12))

    var lpx = s.load_float(b_verts, v_base)
    var lpy = s.load_float(b_verts, s.iadd(v_base, s.uint(1)))
    var lpz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))

    var wp = emit_mat4_mul_point(s, b_instances, inst_base, lpx, lpy, lpz)
    var vp_off = s.push_word(0)
    var clip = emit_mat4_mul_point(s, b_globals, vp_off, wp[0], wp[1], wp[2])

    # Apply sub-pixel TAA/TSR camera jitter when rendering main camera (vp_off == 0)
    var jx = s.push_float(1)
    var jy = s.push_float(2)
    var cx_jit = s.fma(jx, clip[3], clip[0])
    var cy_jit = s.fma(jy, clip[3], clip[1])
    s.write_clip_position(cx_jit, cy_jit, clip[2], clip[3])

    # VisBuffer payload: word0 = (inst_id << 16) | (ml_id + 1), word1 = (final_mat << 16) | local_prim
    var pay0 = s.bit_or(s.shl(inst_id, s.uint(16)), s.iadd(ml_id, s.uint(1)))
    var pay1 = s.bit_or(s.shl(final_mat, s.uint(16)), local_prim)
    emit(s.code, OP_STORE, out_vis[1], s.uvec2(pay0, pay1))

    return s.finish()


def visbuffer_fragment_shader() -> List[UInt32]:
    """Hardware Fragment Shader (`EarlyFragmentTests`) writing 64-bit `R32G32_UINT` VisBuffer IDs."""
    var s = AaaSpirvModule(MODEL_FRAGMENT, push_words=AAA_PUSH_WORDS, early_depth=True)
    var in_vis = s.add_stage_io(STORAGE_INPUT, s.type_uvec2, location=0, flat=True)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec4, location=0, flat=False)

    var vis_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_uvec2, vis_val, in_vis[1])
    var w0 = s.extract_u(vis_val, 0)
    var w1 = s.extract_u(vis_val, 1)
    emit(s.code, OP_STORE, out_col[1], s.uvec4(w0, w1, s.uint(0), s.uint(1)))
    return s.finish()


def emit_ray_barycentrics(
    mut s: AaaSpirvModule,
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
    dx_rx: UInt32,
    dx_ry: UInt32,
    dx_rz: UInt32,
    dy_rx: UInt32,
    dy_ry: UInt32,
    dy_rz: UInt32,
    p0x: UInt32,
    p0y: UInt32,
    p0z: UInt32,
    p1x: UInt32,
    p1y: UInt32,
    p1z: UInt32,
    p2x: UInt32,
    p2y: UInt32,
    p2z: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Analytic 3-Ray Möller-Trumbore Differential Solver!
    Returns `(b1, b2, b1_dx, b2_dx, b1_dy, b2_dy)` for exact perspective-correct
    barycentric interpolation AND screen-space UV gradients `(ddx_uv, ddy_uv)`.
    """
    var e1x = s.fsub(p1x, p0x)
    var e1y = s.fsub(p1y, p0y)
    var e1z = s.fsub(p1z, p0z)
    var e2x = s.fsub(p2x, p0x)
    var e2y = s.fsub(p2y, p0y)
    var e2z = s.fsub(p2z, p0z)
    var tx = s.fsub(ox, p0x)
    var ty = s.fsub(oy, p0y)
    var tz = s.fsub(oz, p0z)
    var qvec = emit_vec3_cross(s, tx, ty, tz, e1x, e1y, e1z)

    # Center pixel ray `D`
    var pvec = emit_vec3_cross(s, dx, dy, dz, e2x, e2y, e2z)
    var det = emit_vec3_dot(s, e1x, e1y, e1z, pvec[0], pvec[1], pvec[2])
    var safe_det = s.select_f(
        s.fge(det, s.float(0.0)),
        s.nmax(det, s.float(1e-12)),
        s.nmin(det, s.float(-1e-12)),
    )
    var inv_det = s.fdiv(s.float(1.0), safe_det)
    var b1 = s.nclamp(s.fmul(emit_vec3_dot(s, tx, ty, tz, pvec[0], pvec[1], pvec[2]), inv_det), s.float(0.0), s.float(1.0))
    var b2 = s.nclamp(s.fmul(emit_vec3_dot(s, dx, dy, dz, qvec[0], qvec[1], qvec[2]), inv_det), s.float(0.0), s.float(1.0))

    # Neighbor `+X` ray `D_dx`
    var pvec_x = emit_vec3_cross(s, dx_rx, dx_ry, dx_rz, e2x, e2y, e2z)
    var det_x = emit_vec3_dot(s, e1x, e1y, e1z, pvec_x[0], pvec_x[1], pvec_x[2])
    var safe_det_x = s.select_f(
        s.fge(det_x, s.float(0.0)),
        s.nmax(det_x, s.float(1e-12)),
        s.nmin(det_x, s.float(-1e-12)),
    )
    var inv_det_x = s.fdiv(s.float(1.0), safe_det_x)
    var b1_dx = s.fmul(emit_vec3_dot(s, tx, ty, tz, pvec_x[0], pvec_x[1], pvec_x[2]), inv_det_x)
    var b2_dx = s.fmul(emit_vec3_dot(s, dx_rx, dx_ry, dx_rz, qvec[0], qvec[1], qvec[2]), inv_det_x)

    # Neighbor `+Y` ray `D_dy`
    var pvec_y = emit_vec3_cross(s, dy_rx, dy_ry, dy_rz, e2x, e2y, e2z)
    var det_y = emit_vec3_dot(s, e1x, e1y, e1z, pvec_y[0], pvec_y[1], pvec_y[2])
    var safe_det_y = s.select_f(
        s.fge(det_y, s.float(0.0)),
        s.nmax(det_y, s.float(1e-12)),
        s.nmin(det_y, s.float(-1e-12)),
    )
    var inv_det_y = s.fdiv(s.float(1.0), safe_det_y)
    var b1_dy = s.fmul(emit_vec3_dot(s, tx, ty, tz, pvec_y[0], pvec_y[1], pvec_y[2]), inv_det_y)
    var b2_dy = s.fmul(emit_vec3_dot(s, dy_rx, dy_ry, dy_rz, qvec[0], qvec[1], qvec[2]), inv_det_y)

    return (b1, b2, b1_dx, b2_dx, b1_dy, b2_dy)


def make_reverse_z_perspective(
    fov_y_rad: Float32,
    aspect: Float32,
    near_z: Float32,
) -> List[Float32]:
    """Infinite Reverse-Z `[1.0 -> 0.0]` column-major projection matrix (`z_clip = near_z, w_clip = z_view`)."""
    from std.math import tan
    var f = Float32(1.0) / tan(fov_y_rad * Float32(0.5))
    var m = List[Float32](length=16, fill=0.0)
    m[0] = f / max(Float32(0.001), aspect)
    m[5] = f
    m[11] = Float32(1.0)     # w_clip = +z_view (left-handed forward +Z in view space!)
    m[14] = near_z           # z_clip = near_z -> z_ndc = near_z / z_view (1.0 at near_z, -> 0.0 at infinity!)
    return m^
