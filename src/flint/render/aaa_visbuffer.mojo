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
comptime BIND_AAA_INSTANCES: Int = 3     # AAA_INSTANCE_WORDS (40) per instance (cur Mat4 + prev Mat4 + meshlet/skin range)
comptime BIND_AAA_MATERIALS: Int = 4     # AAA_MATERIAL_WORDS (24) per enterprise BSDF material
comptime BIND_AAA_INDIRECT: Int = 5      # Word 0 = atomic draw count; Word 4+ = VkDrawIndirectCommand[N] (4 words each)
comptime BIND_AAA_GLOBALS: Int = 6       # Camera basis, ViewProj, Light, VSM matrix, SH9 irradiance, flags
comptime BIND_AAA_HDR_OUT: Int = 7       # Linear RGBA32F (4 floats/pixel) HDR resolve buffer + moments
comptime BIND_AAA_HISTORY: Int = 8       # TSR v2 / SVGF history RGBA32F + lock confidence
comptime BIND_AAA_VSM_TABLE: Int = 9     # 128x128 Virtual Shadow Map page table + request flags
comptime BIND_AAA_SURFCACHE: Int = 10    # Direct-lit Lumen-style per-triangle/meshlet Surface Cache RGBE/Float3
comptime BIND_AAA_OIT_NODES: Int = 11    # Per-pixel 4-layer exact short-stack OIT + 6-way volumetric plumes
comptime BIND_AAA_CLUSTERS: Int = 12     # 16x9x24 Log-Z Clustered Punctual & Heitz LTC Area Lights (V18)
comptime BIND_AAA_SKINNING: Int = 13     # Dual-Quaternion Bone Palettes + Bind-Pose + Prev Skinned Verts (V18)
comptime BIND_AAA_WORKLIST: Int = 13     # 1D Compacted Meshlet Cull Worklist (2 words/entry: `inst_idx, ml_id`)
comptime BIND_AAA_POST_SCRATCH: Int = 14 # Auto-Exposure Histogram + 5-Mip Dual-Kawase Bloom + Bokeh DoF (V19)
comptime BIND_AAA_CANDIDATES: Int = 14   # Two-Phase Hi-Z Occluded Candidates Queue (2 words/entry: `inst_idx, ml_id`)
comptime BIND_AAA_DRAW_PAIRS: Int = 15   # GPU Draw `(inst_id, ml_id)` pairs indexed by `gl_InstanceIndex`
comptime AAA_STORAGE_COUNT: Int = 16

comptime BIND_AAA_VIS_TEX: Int = 16      # usampler2D (VK_FORMAT_R32G32_UINT VisBuffer)
comptime BIND_AAA_DEPTH_TEX: Int = 17    # sampler2D  (VK_FORMAT_D32_SFLOAT Reverse-Z Depth)
comptime BIND_AAA_VSM_TEX: Int = 18      # sampler2D  (VK_FORMAT_R32_SFLOAT / D32_SFLOAT Physical VSM Page Pool)
comptime BIND_AAA_ENV_CUBE: Int = 19     # samplerCube (VK_FORMAT_BC6H_UFLOAT 6-Mip Split-Sum GGX Cubemap)
comptime BIND_AAA_BRDF_LUT: Int = 20     # sampler2D  (VK_FORMAT_R16G16_SFLOAT Split-Sum Multi-Scatter LUT)
comptime BIND_AAA_LTC_LUT: Int = 21      # sampler2D  (VK_FORMAT_R16G16_SFLOAT Heitz LTC 64x64 Inverse-Matrix LUT)
comptime AAA_SINGLE_SAMPLERS: Int = 6

comptime BIND_AAA_BINDLESS_TEX: Int = 22 # sampler2D[AAA_MAX_BINDLESS] (BC7_UNORM, BC7_SRGB, BC5_UNORM, BC6H_UFLOAT)
comptime AAA_MAX_BINDLESS: Int = 4096
comptime BIND_AAA_TLAS: Int = 23         # accelerationStructureKHR (Hardware BLAS/TLAS for SPV_KHR_ray_query)
comptime BIND_AAA_SWAPCHAIN_IMG: Int = 24 # writeonly/readwrite storage image2D bound to acquired swapchain image

comptime AAA_MAX_PACKED_MESHLET_ID: UInt32 = 0x3FFFFE  # 22-bit VisBuffer payload: 0 = sky, 1..0x3FFFFF = ml_id + 1
comptime AAA_MAX_PACKED_INSTANCES: UInt32 = 0x3FFFF    # 18-bit VisBuffer payload: inst_id in 0..262143
comptime AAA_HIZ_MAX_MIPS: Int = 8                    # Multi-mip Reverse-Z min-reduction Hi-Z pyramid (mips 0..7)

comptime AAA_PUSH_WORDS: Int = 16
comptime AAA_PUSH_BYTES: Int = 64

# Instance SSBO layout: 40 words (160 bytes) per instance
# [0..15]:  model_to_world Mat4 (column-major Float32[16])
# [16..31]: prev_model_to_world Mat4 (column-major Float32[16], for 3D per-pixel TSR/SVGF motion vectors!)
# [32]:     first_meshlet (UInt32)
# [33]:     meshlet_count (UInt32)
# [34]:     flags_and_material_override (UInt32: bit16 = is_skinned, bits[0..15] = 0xFFFF or material_override)
# [35]:     uniform_scale_bits (Float32: max axis scale for bounding sphere radius)
# [36]:     skin_bind_first_slot (UInt32: base vertex slot in BIND_AAA_SKINNING, 0 if static)
# [37]:     mesh_first_vertex (UInt32: CookedMeshEntry.first_vertex in BIND_AAA_VERTICES)
# [38..39]: reserved (0)
comptime AAA_INSTANCE_WORDS: Int = 40

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


def _emit_uf16_to_f32(mut s: AaaSpirvModule, h_u16: UInt32) -> UInt32:
    """Exact 5-instruction IEEE-754 unsigned `F16` (`0 .. 0x7BFF`) -> `Float32` decoder in SPIR-V."""
    var is_sub = s.ult(h_u16, s.uint(0x0400))
    var sub_f = s.fmul(s.float_from_uint(h_u16), s.float(5.96046448e-8))
    var norm_bits = s.shl(s.iadd(h_u16, s.uint(0x1C000)), s.uint(13))
    return s.select_f(is_sub, sub_f, s.bitcast_float(norm_bits))


def meshlet_cull_shader() -> List[UInt32]:
    """GPU Compute Meshlet Culler (`QEM LOD 0..3 Screen-Error Cut + Frustum + Normal-Cone + Reverse-Z Hi-Z`
    -> `VkDrawIndirectCommand`).
    Push[0] = instance_count, Push[1] = max_draws, Push[2] = enable_cone_cull (1/0), Push[3] = enable_hiz (1/0),
    Push[4] = screen_w, Push[5] = screen_h, Push[6] = max_lod_error_px_bits (Float32, e.g. 1.0 px),
    Push[7] = cull_phase (0 = Phase 1: queue occluded candidates, 1 = Phase 2: test candidates against fresh Hi-Z).
    Invocation X = instance_idx (or candidate workgroup thread in Phase 2), Invocation Y = local_meshlet_idx!
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_indirect = s.buffer(BIND_AAA_INDIRECT)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)
    var b_worklist = s.buffer(BIND_AAA_WORKLIST)
    var b_candidates = s.buffer(BIND_AAA_CANDIDATES)
    var b_draw_pairs = s.buffer(BIND_AAA_DRAW_PAIRS)

    var work_count = s.push_word(0)
    var max_draws = s.push_word(1)
    var enable_cone = s.push_word(2)
    var enable_hiz = s.push_word(3)
    var scr_w = s.float_from_uint(s.push_word(4))
    var scr_h = s.float_from_uint(s.push_word(5))
    var max_lod_px = s.push_float(6)
    var cull_phase = s.push_word(7)
    var draw_pairs_base = s.push_word(9)

    var inst_idx_var = s.uvar(s.uint(0))
    var ml_id_var = s.uvar(s.uint(0))
    var valid_var = s.uvar(s.uint(0))

    var flat_id = s.iadd(
        s.imul(s.workgroup_axis(0), s.uint(64)),
        s.iadd(s.imul(s.local_axis(1), s.uint(8)), s.local_axis(0)),
    )

    var is_p1 = s.ieq(cull_phase, s.uint(0))
    s.begin_if(is_p1)
    # Phase 1: 1D compacted 2-word `(inst_idx, ml_id)` meshlet worklist
    s.begin_if(s.ult(flat_id, work_count))
    var wl_off = s.imul(flat_id, s.uint(2))
    s.store(inst_idx_var, s.load_word(b_worklist, wl_off))
    s.store(ml_id_var, s.load_word(b_worklist, s.iadd(wl_off, s.uint(1))))
    s.store(valid_var, s.uint(1))
    s.end_if()
    s.begin_else()
    # Phase 2: process 2-word candidate meshlets queued in b_candidates
    var cand_total = s.load_word(b_candidates, s.uint(0))
    s.begin_if(s.logical_and(s.ult(flat_id, cand_total), s.ult(flat_id, max_draws)))
    var cand_off = s.iadd(s.uint(4), s.imul(flat_id, s.uint(2)))
    s.store(inst_idx_var, s.load_word(b_candidates, cand_off))
    s.store(ml_id_var, s.load_word(b_candidates, s.iadd(cand_off, s.uint(1))))
    s.store(valid_var, s.uint(1))
    s.end_if()
    s.end_if()

    s.return_if(s.ieq(s.load_u(valid_var), s.uint(0)))

    var inst_idx = s.load_u(inst_idx_var)
    var ml_id = s.load_u(ml_id_var)
    # Guard against 22-bit ml_id (0..0x3FFFFE) and 18-bit inst_idx (0..0x3FFFF) VisBuffer limits
    s.return_if(
        s.logical_or(
            s.ult(s.uint(AAA_MAX_PACKED_MESHLET_ID), ml_id),
            s.ult(s.uint(AAA_MAX_PACKED_INSTANCES), inst_idx),
        )
    )

    var inst_base = s.imul(inst_idx, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    # Skip alpha-masked instances (`bit 17` of `w39`); they draw in the dedicated Two-Bucket alpha-cutout pass!
    var inst_w39 = s.load_word(b_instances, s.iadd(inst_base, s.uint(39)))
    s.return_if(s.ine(s.bit_and(inst_w39, s.uint(0x00020000)), s.uint(0)))
    var ml_base = s.imul(ml_id, s.uint(12))
    var lcx = s.load_float(b_meshlets, ml_base)
    var lcy = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(1)))
    var lcz = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(2)))
    var lrad = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(3)))
    var lax = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(4)))
    var lay = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(5)))
    var laz = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(6)))
    var cone_sin = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(7)))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var tri_cnt = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(9)))
    var lod_err_pack = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(10)))

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
    var vz = emit_vec3_dot(s, rx, ry, rz, fwd_x, fwd_y, fwd_z)

    # 0. Continuous QEM Meshlet Hierarchy Cut & Cone/Frustum tests (Phase 1 only)
    s.begin_if(is_p1)
    var h_self = s.bit_and(lod_err_pack, s.uint(0xFFFF))
    var h_parent = s.shr(lod_err_pack, s.uint(16))
    var self_err = _emit_uf16_to_f32(s, h_self)
    var parent_err = _emit_uf16_to_f32(s, h_parent)
    var inst_ox = s.load_float(b_instances, s.iadd(inst_base, s.uint(12)))
    var inst_oy = s.load_float(b_instances, s.iadd(inst_base, s.uint(13)))
    var inst_oz = s.load_float(b_instances, s.iadd(inst_base, s.uint(14)))
    var io_dx = s.fsub(inst_ox, cam_x)
    var io_dy = s.fsub(inst_oy, cam_y)
    var io_dz = s.fsub(inst_oz, cam_z)
    var d_inst = s.nmax(near_z, s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, io_dx, io_dy, io_dz, io_dx, io_dy, io_dz))))
    var px_per_m = s.fdiv(s.fmul(uscale, scr_h), s.nmax(s.float(1e-5), s.fmul(s.float(2.0), s.fmul(d_inst, tan_y))))
    var e_max = s.select_f(
        s.fgt(max_lod_px, s.float(0.0)),
        s.fdiv(max_lod_px, s.nmax(s.float(1e-5), px_per_m)),
        s.float(0.0),
    )
    var too_coarse = s.fgt(self_err, e_max)
    var too_fine = s.logical_and(s.flt(parent_err, s.float(65000.0)), s.fge(e_max, parent_err))
    s.return_if(s.logical_or(too_coarse, too_fine))

    # 1. Backface normal-cone cull
    s.begin_if(s.logical_and(s.ine(enable_cone, s.uint(0)), s.flt(cone_sin, s.float(1.5))))
    var waxis_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lax, lay, laz)
    var waxis = emit_vec3_normalize(s, waxis_raw[0], waxis_raw[1], waxis_raw[2])
    var cdot = emit_vec3_dot(s, rx, ry, rz, waxis[0], waxis[1], waxis[2])
    var thresh = s.fma(cone_sin, dist, wrad)
    s.return_if(s.fge(cdot, thresh))
    s.end_if()

    # 2. View-frustum bounding-sphere test
    s.return_if(s.flt(s.fadd(vz, wrad), near_z))

    var vx = s.fabs(emit_vec3_dot(s, rx, ry, rz, rgt_x, rgt_y, rgt_z))
    var vy = s.fabs(emit_vec3_dot(s, rx, ry, rz, up_x, up_y, up_z))
    var sec_x = s.fsqrt(s.fma(tan_x, tan_x, s.float(1.0)))
    var sec_y = s.fsqrt(s.fma(tan_y, tan_y, s.float(1.0)))
    var vz_pos = s.nmax(s.float(0.0), vz)
    s.return_if(s.fgt(s.fsub(vx, s.fmul(wrad, sec_x)), s.fmul(vz_pos, tan_x)))
    s.return_if(s.fgt(s.fsub(vy, s.fmul(wrad, sec_y)), s.fmul(vz_pos, tan_y)))
    s.end_if()

    # 3. Reverse-Z 5-Tap Depth Occlusion Test (Phase 1 vs prev depth, Phase 2 vs fresh depth)
    s.begin_if(s.logical_and(s.ine(enable_hiz, s.uint(0)), s.fgt(s.fsub(vz, wrad), near_z)))
    var prev_wc = emit_mat4_mul_point(s, b_instances, s.iadd(inst_base, s.uint(16)), lcx, lcy, lcz)
    var clip_mat_offset = s.select_u(is_p1, s.uint(16), s.uint(0))
    var target_pt_x = s.select_f(is_p1, prev_wc[0], wc[0])
    var target_pt_y = s.select_f(is_p1, prev_wc[1], wc[1])
    var target_pt_z = s.select_f(is_p1, prev_wc[2], wc[2])
    var clip = emit_mat4_mul_point(s, b_globals, clip_mat_offset, target_pt_x, target_pt_y, target_pt_z)
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
    var footprint_px = s.fmul(s.nmax(s.fmul(uv_r_x, scr_w), s.fmul(uv_r_y, scr_h)), s.float(2.0))
    var hiz_mips_u = s.push_word(8)
    var max_hiz_lod = s.select_f(
        s.ult(s.uint(1), hiz_mips_u),
        s.nmin(s.float(Float32(AAA_HIZ_MAX_MIPS - 1)), s.float_from_uint(s.isub(hiz_mips_u, s.uint(1)))),
        s.float(0.0),
    )
    var hiz_lod = s.nclamp(s.log2(s.nmax(footprint_px, s.float(1.0))), s.float(0.0), max_hiz_lod)
    var d0 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(u_c, v_c), hiz_lod), 0)
    var d1 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fsub(u_c, uv_r_x), s.fsub(v_c, uv_r_y)), hiz_lod), 0)
    var d2 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fadd(u_c, uv_r_x), s.fsub(v_c, uv_r_y)), hiz_lod), 0)
    var d3 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fsub(u_c, uv_r_x), s.fadd(v_c, uv_r_y)), hiz_lod), 0)
    var d4 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(s.fadd(u_c, uv_r_x), s.fadd(v_c, uv_r_y)), hiz_lod), 0)
    # In Reverse-Z, a closer occluder has LARGER depth; meshlet is occluded iff ALL 5 taps > sphere_rev_z!
    var min_occ_z = s.nmin(d0, s.nmin(s.nmin(d1, d2), s.nmin(d3, d4)))
    var is_occluded = s.fgt(min_occ_z, s.fadd(sphere_rev_z, s.float(0.0005)))
    # In Phase 1: append 2-word occluded candidate pair `(inst_idx, ml_id)` into b_candidates
    s.begin_if(s.logical_and(is_occluded, is_p1))
    var cand_slot = s.atomic_add_word(b_candidates, s.uint(0), s.uint(1))
    s.begin_if(s.ult(cand_slot, max_draws))
    var c_dst = s.iadd(s.uint(4), s.imul(cand_slot, s.uint(2)))
    s.store_word(b_candidates, c_dst, inst_idx)
    s.store_word(b_candidates, s.iadd(c_dst, s.uint(1)), ml_id)
    s.end_if()
    s.end_if()
    s.return_if(is_occluded)
    s.end_if()
    s.end_if()
    s.end_if()

    # 4. Append `(inst_idx, ml_id)` to `b_draw_pairs[draw_slot]` + `VkDrawIndexedIndirectCommand`
    var slot = s.atomic_add_word(b_indirect, s.uint(0), s.uint(1))
    s.begin_if(s.ult(slot, max_draws))
    var draw_slot = s.iadd(draw_pairs_base, slot)
    var dp_off = s.imul(draw_slot, s.uint(2))
    s.store_word(b_draw_pairs, dp_off, inst_idx)
    s.store_word(b_draw_pairs, s.iadd(dp_off, s.uint(1)), ml_id)
    var cmd_base = s.iadd(s.uint(4), s.imul(slot, s.uint(5)))
    var idx_cnt = s.imul(tri_cnt, s.uint(3))
    var first_idx = s.imul(first_tri, s.uint(3))
    s.store_word(b_indirect, cmd_base, idx_cnt)
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(1)), s.uint(1))
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(2)), first_idx)
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(3)), s.uint(0))
    s.store_word(b_indirect, s.iadd(cmd_base, s.uint(4)), draw_slot)
    s.end_if()

    return s.finish()


def visbuffer_vertex_shader() -> List[UInt32]:
    """Hardware Vertex Shader for 64-Bit Visibility Buffer (`VK_KHR_dynamic_rendering`).
    Pulls hardware indexed vertices directly, reusing Forsyth post-transform cache!
    Push[0] = view_proj word offset in `BIND_AAA_GLOBALS` (0 = main camera, 56 = VSM light matrix!).
    """
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_draw_pairs = s.buffer(BIND_AAA_DRAW_PAIRS)
    _ = b_indices
    var out_vis = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec2, location=0, flat=True)

    var real_vtx = s.vertex_index()
    var draw_slot = s.instance_index()
    var dp_off = s.imul(draw_slot, s.uint(2))
    var inst_id = s.load_word(b_draw_pairs, dp_off)
    var ml_id = s.load_word(b_draw_pairs, s.iadd(dp_off, s.uint(1)))

    var ml_base = s.imul(ml_id, s.uint(12))
    var ml_mat = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(11)))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var mat_override = s.load_word(b_instances, s.iadd(inst_base, s.uint(34)))
    var mat_ov16 = s.bit_and(mat_override, s.uint(0xFFFF))
    var final_mat = s.select_u(s.ieq(mat_ov16, s.uint(0xFFFF)), ml_mat, mat_ov16)

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

    # 40-bit VisBuffer payload: word0 = ((inst_id & 0x3FF) << 22) | min(ml_id + 1, 0x3FFFFF),
    # word1_base = (final_mat << 16) | (((inst_id >> 10) & 0xFF) << 8)
    var pay0 = s.bit_or(
        s.shl(s.bit_and(inst_id, s.uint(0x3FF)), s.uint(22)),
        s.umin(s.iadd(ml_id, s.uint(1)), s.uint(0x3FFFFF)),
    )
    var pay1_base = s.bit_or(
        s.shl(final_mat, s.uint(16)),
        s.shl(s.bit_and(s.shr(inst_id, s.uint(10)), s.uint(0xFF)), s.uint(8)),
    )
    emit(s.code, OP_STORE, out_vis[1], s.uvec2(pay0, pay1_base))

    return s.finish()


def visbuffer_fragment_shader() -> List[UInt32]:
    """Hardware Fragment Shader (`EarlyFragmentTests`) writing 64-bit `R32G32_UINT` VisBuffer IDs."""
    var s = AaaSpirvModule(MODEL_FRAGMENT, push_words=AAA_PUSH_WORDS, early_depth=True)
    var prim_id = s.primitive_id()
    var in_vis = s.add_stage_io(STORAGE_INPUT, s.type_uvec2, location=0, flat=True)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec4, location=0, flat=False)

    var vis_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_uvec2, vis_val, in_vis[1])
    var w0 = s.extract_u(vis_val, 0)
    var w1_base = s.extract_u(vis_val, 1)
    var w1 = s.bit_or(w1_base, s.bit_and(prim_id, s.uint(0xFF)))
    emit(s.code, OP_STORE, out_col[1], s.uvec4(w0, w1, s.uint(0), s.uint(1)))
    return s.finish()


def visbuffer_alphamask_vertex_shader() -> List[UInt32]:
    """Two-Bucket Alpha-Masked Hardware Vertex Shader (`alpha_cutoff > 0.0`):
    Passes `flat uvec2(pay0, pay1_base)` (`location=0`), smooth `vec2 uv` (`location=1`),
    and `flat uint t_alb` (`location=2`) to `visbuffer_alphamask_fragment_shader`."""
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_draw_pairs = s.buffer(BIND_AAA_DRAW_PAIRS)
    _ = b_indices

    var out_vis = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec2, location=0, flat=True)
    var out_uv = s.add_stage_io(STORAGE_OUTPUT, s.type_vec2, location=1, flat=False)
    var out_talb = s.add_stage_io(STORAGE_OUTPUT, s.type_uint, location=2, flat=True)

    var real_vtx = s.vertex_index()
    var draw_slot = s.instance_index()
    var dp_off = s.imul(draw_slot, s.uint(2))
    var inst_id = s.load_word(b_draw_pairs, dp_off)
    var ml_id = s.load_word(b_draw_pairs, s.iadd(dp_off, s.uint(1)))

    var ml_base = s.imul(ml_id, s.uint(12))
    var ml_mat = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(11)))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var mat_override = s.load_word(b_instances, s.iadd(inst_base, s.uint(34)))
    var mat_ov16 = s.bit_and(mat_override, s.uint(0xFFFF))
    var final_mat = s.select_u(s.ieq(mat_ov16, s.uint(0xFFFF)), ml_mat, mat_ov16)

    var m_base = s.imul(final_mat, s.uint(24))
    var tex_pk = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))
    var t_alb = s.bit_and(tex_pk, s.uint(0xFF))

    var v_base = s.imul(real_vtx, s.uint(12))
    var lpx = s.load_float(b_verts, v_base)
    var lpy = s.load_float(b_verts, s.iadd(v_base, s.uint(1)))
    var lpz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))
    var uv_u = s.load_float(b_verts, s.iadd(v_base, s.uint(9)))
    var uv_v = s.load_float(b_verts, s.iadd(v_base, s.uint(10)))

    var wp = emit_mat4_mul_point(s, b_instances, inst_base, lpx, lpy, lpz)
    var vp_off = s.push_word(0)
    var clip = emit_mat4_mul_point(s, b_globals, vp_off, wp[0], wp[1], wp[2])

    var jx = s.push_float(1)
    var jy = s.push_float(2)
    var cx_jit = s.fma(jx, clip[3], clip[0])
    var cy_jit = s.fma(jy, clip[3], clip[1])
    s.write_clip_position(cx_jit, cy_jit, clip[2], clip[3])

    var pay0 = s.bit_or(
        s.shl(s.bit_and(inst_id, s.uint(0x3FF)), s.uint(22)),
        s.umin(s.iadd(ml_id, s.uint(1)), s.uint(0x3FFFFF)),
    )
    var pay1_base = s.bit_or(
        s.shl(final_mat, s.uint(16)),
        s.shl(s.bit_and(s.shr(inst_id, s.uint(10)), s.uint(0xFF)), s.uint(8)),
    )
    emit(s.code, OP_STORE, out_vis[1], s.uvec2(pay0, pay1_base))
    emit(s.code, OP_STORE, out_uv[1], s.vec2(uv_u, uv_v))
    emit(s.code, OP_STORE, out_talb[1], t_alb)
    return s.finish()


def visbuffer_alphamask_fragment_shader() -> List[UInt32]:
    """Two-Bucket Alpha-Cutout Fragment Shader (`early_depth = False`, `enable_bindless = True`):
    Samples `bindless_arr[t_alb]` alpha with mip-coverage preservation `a * (1.0 + 0.18 * mip_est)`
    and discards (`OpKill`) when `alpha < 0.48`, keeping `pipe_visbuffer_gfx` (`early_depth=True`) 100% untouched!"""
    var s = AaaSpirvModule(
        MODEL_FRAGMENT,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        early_depth=False,
    )
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var prim_id = s.primitive_id()
    var in_vis = s.add_stage_io(STORAGE_INPUT, s.type_uvec2, location=0, flat=True)
    var in_uv = s.add_stage_io(STORAGE_INPUT, s.type_vec2, location=1, flat=False)
    var in_talb = s.add_stage_io(STORAGE_INPUT, s.type_uint, location=2, flat=True)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec4, location=0, flat=False)

    var uv_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec2, uv_val, in_uv[1])
    var uv_x = s.extract_f(uv_val, 0)
    var uv_y = s.extract_f(uv_val, 1)

    # Compute screen-space UV derivatives in uniform quad control flow
    var ddx_u = s.dpdx(uv_x)
    var ddx_v = s.dpdx(uv_y)
    var ddy_u = s.dpdy(uv_x)
    var ddy_v = s.dpdy(uv_y)
    var gx2 = s.fma(ddx_u, ddx_u, s.fmul(ddx_v, ddx_v))
    var gy2 = s.fma(ddy_u, ddy_u, s.fmul(ddy_v, ddy_v))
    var px_foot = s.fmul(s.fsqrt(s.nmax(s.float(1e-8), s.nmax(gx2, gy2))), s.float(512.0))
    var mip_est = s.nclamp(s.log2(s.nmax(s.float(1.0), px_foot)), s.float(0.0), s.float(6.0))

    var t_alb = s.fresh()
    emit(s.code, OP_LOAD, s.type_uint, t_alb, in_talb[1])
    var alpha_v = s.fvar(s.float(1.0))
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var rgba = s.sample_bindless_grad(
        bindless_arr,
        t_alb,
        uv_val,
        s.vec2(ddx_u, ddx_v),
        s.vec2(ddy_u, ddy_v),
    )
    var raw_a = s.extract_f(rgba, 3)
    s.store(alpha_v, s.fmul(raw_a, s.fma(mip_est, s.float(0.18), s.float(1.0))))
    s.end_if()

    s.discard_if(s.flt(s.load_f(alpha_v), s.float(0.48)))

    var vis_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_uvec2, vis_val, in_vis[1])
    var w0 = s.extract_u(vis_val, 0)
    var w1_base = s.extract_u(vis_val, 1)
    var w1 = s.bit_or(w1_base, s.bit_and(prim_id, s.uint(0xFF)))
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


def compute_hiz_mip_levels(width: Int, height: Int) -> Int:
    """Computes the number of Hi-Z pyramid mip levels (clamped to `1..AAA_HIZ_MAX_MIPS`)."""
    var dim = width if width > height else height
    var mips = 1
    while dim > 1 and mips < AAA_HIZ_MAX_MIPS:
        dim = dim >> 1
        mips += 1
    return mips


def hiz_depth_downsample_shader() -> List[UInt32]:
    """Compute Shader (`8x8x1`) building a conservative Reverse-Z min-reduction Hi-Z pyramid (`R32_SFLOAT`).
    Uses the unified AAA descriptor set:
      - `BIND_AAA_DEPTH_TEX` (`D32_SFLOAT` depth buffer) for `mip_level == 0`
      - `BIND_AAA_POST_SCRATCH` (contiguous `R32_SFLOAT` mip chain starting at `dst_off`)
    Push constants:
      - `push[0]`: `dst_w`
      - `push[1]`: `dst_h`
      - `push[2]`: `src_w`
      - `push[3]`: `src_h`
      - `push[4]`: `src_word_offset` (unused when `mip_level == 0`)
      - `push[5]`: `dst_word_offset`
      - `push[6]`: `mip_level` (`0` = copy/sample depth -> mip 0; `> 0` = 2x2 conservative min-reduction)
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var dst_w = s.push_word(0)
    var dst_h = s.push_word(1)
    var src_w = s.push_word(2)
    var src_h = s.push_word(3)
    var src_off = s.push_word(4)
    var dst_off = s.push_word(5)
    var mip_level = s.push_word(6)

    s.return_if(s.logical_or(s.uge(px, dst_w), s.uge(py, dst_h)))
    var dst_idx = s.iadd(dst_off, s.iadd(s.imul(py, dst_w), px))

    s.begin_if(s.ieq(mip_level, s.uint(0)))
    var u = s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), s.float_from_uint(dst_w))
    var v = s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), s.float_from_uint(dst_h))
    var d0 = s.extract_f(s.sample_2d_lod(depth_samp, s.vec2(u, v), s.float(0.0)), 0)
    s.store_float(b_post, dst_idx, d0)
    s.begin_else()
    var max_sx = s.isub(src_w, s.uint(1))
    var max_sy = s.isub(src_h, s.uint(1))
    var sx0 = s.imul(px, s.uint(2))
    var sy0 = s.imul(py, s.uint(2))
    var sx1 = s.umin(s.iadd(sx0, s.uint(1)), max_sx)
    var sy1 = s.umin(s.iadd(sy0, s.uint(1)), max_sy)

    var row0 = s.iadd(src_off, s.imul(sy0, src_w))
    var row1 = s.iadd(src_off, s.imul(sy1, src_w))
    var t00 = s.load_float(b_post, s.iadd(row0, sx0))
    var t10 = s.load_float(b_post, s.iadd(row0, sx1))
    var t01 = s.load_float(b_post, s.iadd(row1, sx0))
    var t11 = s.load_float(b_post, s.iadd(row1, sx1))
    var min_d = s.fvar(s.nmin(s.nmin(t00, t10), s.nmin(t01, t11)))

    # Conservative odd-width / odd-height edge reduction for non-power-of-two resolutions
    var odd_w = s.logical_and(s.ine(s.bit_and(src_w, s.uint(1)), s.uint(0)), s.ieq(px, s.isub(dst_w, s.uint(1))))
    var odd_h = s.logical_and(s.ine(s.bit_and(src_h, s.uint(1)), s.uint(0)), s.ieq(py, s.isub(dst_h, s.uint(1))))
    s.begin_if(odd_w)
    var t20 = s.load_float(b_post, s.iadd(row0, max_sx))
    var t21 = s.load_float(b_post, s.iadd(row1, max_sx))
    s.store(min_d, s.nmin(s.load_f(min_d), s.nmin(t20, t21)))
    s.end_if()
    s.begin_if(odd_h)
    var row2 = s.iadd(src_off, s.imul(max_sy, src_w))
    var t02 = s.load_float(b_post, s.iadd(row2, sx0))
    var t12 = s.load_float(b_post, s.iadd(row2, sx1))
    s.store(min_d, s.nmin(s.load_f(min_d), s.nmin(t02, t12)))
    s.begin_if(odd_w)
    var t22 = s.load_float(b_post, s.iadd(row2, max_sx))
    s.store(min_d, s.nmin(s.load_f(min_d), t22))
    s.end_if()
    s.end_if()

    s.store_float(b_post, dst_idx, s.load_f(min_d))
    s.end_if()

    return s.finish()
