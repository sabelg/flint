"""Milestone 3 (`V14`) & Commercial Engine v2: Hardware `VK_KHR_acceleration_structure` (`BLAS` / `TLAS`),
`SPV_KHR_ray_query` Contact Shadows, Surface-Cache GI & Zero-Limit `O(1)` Barycentric Textured +
Froxel-Lit Ray-Hit Shader (`aaa_raytracing.mojo`).
"""
from std.math import max
from std.memory import bitcast
from flint.core.transform import Mat4
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    AS_TYPE_BOTTOM_LEVEL_KHR,
    AS_TYPE_TOP_LEVEL_KHR,
    BUFFER_USAGE_AS_STORAGE_KHR,
    BUFFER_USAGE_DEVICE_ADDRESS,
    BUILD_AS_ALLOW_UPDATE_KHR,
    BUILD_AS_MODE_BUILD_KHR,
    BUILD_AS_PREFER_FAST_TRACE_KHR,
    BUILD_AS_TYPE_DEVICE_KHR,
    FORMAT_R32G32B32_SFLOAT,
    GEOMETRY_OPAQUE_BIT_KHR,
    GEOMETRY_TYPE_INSTANCES_KHR,
    GEOMETRY_TYPE_TRIANGLES_KHR,
    INDEX_TYPE_UINT32,
    TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
    TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
    VulkanAccelStruct,
    VulkanGpu,
    call4,
    call5,
    get64,
    put32,
    put64,
    put_float,
)
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS, AAA_MAT_DET_NONE, AAA_MAT_INTERIOR_PARALLAX
from flint.render.aaa_vsm import emit_sample_vsm_depth
from flint.render.aaa_ibl import emit_eval_sh9_irradiance

# SPIR-V RayFlagsKHR constants
comptime RAY_FLAG_NONE_KHR: UInt32 = 0
comptime RAY_FLAG_OPAQUE_KHR: UInt32 = 1
comptime RAY_FLAG_TERMINATE_ON_FIRST_HIT_KHR: UInt32 = 4
comptime RAY_FLAG_SHADOW_OCCLUSION: UInt32 = 5  # OPAQUE | TERMINATE_ON_FIRST_HIT
comptime AAA_RT_MASK_GLASS: UInt32 = 0x80   # TLAS instance mask of transparent (glass) instances
comptime AAA_RT_MASK_OPAQUE: UInt32 = 0x7F  # ray cull mask that skips glass (shadow rays)
comptime AAA_GLOBALS_GLASS_SHADOW_WORD: Int = 256  # float: sun transmittance through glass occluders (0..1)
comptime AAA_GLOBALS_SHADOW_CONE_WORD: Int = 257  # float: tan(angular radius) of the RT sun-shadow cone

# Cluster constants matching `aaa_clustered_lights.mojo` (kept local to preserve acyclic import DAG)
comptime _RT_CLUSTER_X: UInt32 = 16
comptime _RT_CLUSTER_Y: UInt32 = 9
comptime _RT_CLUSTER_Z: UInt32 = 24
comptime _RT_CLUSTER_LIGHTS_BASE: UInt32 = 16
comptime _RT_CLUSTER_COUNTS_BASE: UInt32 = 16400
comptime _RT_CLUSTER_INDICES_BASE: UInt32 = 19856
comptime _RT_MAX_LIGHTS_PER_CLUSTER: UInt32 = 64
comptime _RT_CLUSTER_LIGHT_WORDS: UInt32 = 16


def write_vk_as_instance_64b(
    dst_addr: Int,
    world_mat: Mat4,
    surfcache_base_tri: Int,
    blas_device_address: UInt64,
    cull_disable: Bool = True,
    alpha_masked: Bool = False,
    glass: Bool = False,
):
    """Writes one 64-byte `VkAccelerationStructureInstanceKHR` record with row-major `3x4` transform
    and `instanceCustomIndex = surfcache_base_tri` for `O(1)` Lumen Surface-Cache lookup on ray hit!
    Sets `FORCE_NO_OPAQUE (0x8)` on alpha-masked foliage/fence instances and `FORCE_OPAQUE (0x4)` on opaque meshes.
    Glass instances get instance mask `AAA_RT_MASK_GLASS` only, so shadow rays (`AAA_RT_MASK_OPAQUE`) pass through them."""
    # Row 0: (m0, m4, m8, m12)
    put_float(dst_addr + 0, world_mat.values[0])
    put_float(dst_addr + 4, world_mat.values[4])
    put_float(dst_addr + 8, world_mat.values[8])
    put_float(dst_addr + 12, world_mat.values[12])
    # Row 1: (m1, m5, m9, m13)
    put_float(dst_addr + 16, world_mat.values[1])
    put_float(dst_addr + 20, world_mat.values[5])
    put_float(dst_addr + 24, world_mat.values[9])
    put_float(dst_addr + 28, world_mat.values[13])
    # Row 2: (m2, m6, m10, m14)
    put_float(dst_addr + 32, world_mat.values[2])
    put_float(dst_addr + 36, world_mat.values[6])
    put_float(dst_addr + 40, world_mat.values[10])
    put_float(dst_addr + 44, world_mat.values[14])

    var custom24 = UInt32(surfcache_base_tri & 0xFFFFFF)
    var mask8 = (UInt32(AAA_RT_MASK_GLASS) if glass else UInt32(0xFF)) << 24
    put32(dst_addr + 48, custom24 | mask8)

    var cull_bit = UInt32(1) if (cull_disable or alpha_masked) else UInt32(0)
    var opaq_bit = UInt32(8) if alpha_masked else UInt32(4)
    var flags8 = (cull_bit | opaq_bit) << 24
    put32(dst_addr + 52, flags8)
    put64(dst_addr + 56, Int(blas_device_address))


def build_blas_triangles_ex(
    mut vk: VulkanGpu,
    frame: Int,
    vertex_addr: UInt64,
    vertex_stride: Int,
    max_vertex: Int,
    index_addr: UInt64,
    triangle_count: Int,
    mut scratch_out: GpuBuffer,
    allow_update: Bool = False,
    opaque: Bool = True,
) -> VulkanAccelStruct:
    """Builds a hardware `BLAS` with `VK_GEOMETRY_OPAQUE_BIT_KHR` when `opaque=True` or `flags=0`
    (non-opaque candidate traversal) for alpha-masked foliage/hedges/fences (`opaque=False`)."""
    if opaque:
        return vk.build_blas_triangles(
            frame,
            vertex_addr,
            vertex_stride,
            max_vertex,
            index_addr,
            triangle_count,
            scratch_out,
            allow_update=allow_update,
        )
    var out = VulkanAccelStruct()
    if not vk.has_ray_query or triangle_count <= 0:
        return out^
    var as_flags = BUILD_AS_PREFER_FAST_TRACE_KHR | (BUILD_AS_ALLOW_UPDATE_KHR if allow_update else UInt32(0))
    var geom_flags = GEOMETRY_OPAQUE_BIT_KHR if opaque else UInt32(0)
    vk.rewind()
    var geom = vk.scratch(96)
    put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
    put32(geom + 16, GEOMETRY_TYPE_TRIANGLES_KHR)
    put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR)
    put32(geom + 24 + 16, FORMAT_R32G32B32_SFLOAT)
    put64(geom + 24 + 24, Int(vertex_addr))
    put64(geom + 24 + 32, vertex_stride)
    put32(geom + 24 + 40, UInt32(max_vertex))
    put32(geom + 24 + 44, INDEX_TYPE_UINT32)
    put64(geom + 24 + 48, Int(index_addr))
    put32(geom + 88, geom_flags)

    var build_info = vk.scratch(80)
    put32(build_info, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
    put32(build_info + 16, AS_TYPE_BOTTOM_LEVEL_KHR)
    put32(build_info + 20, as_flags)
    put32(build_info + 24, BUILD_AS_MODE_BUILD_KHR)
    put32(build_info + 48, 1)
    put64(build_info + 56, geom)

    var prim_cnt = vk.scratch(8)
    put32(prim_cnt, UInt32(triangle_count))
    var sizes = vk.scratch(40)
    put32(sizes, TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
    _ = call5(
        vk.device_call("vkGetAccelerationStructureBuildSizesKHR"),
        vk.device,
        BUILD_AS_TYPE_DEVICE_KHR,
        build_info,
        prim_cnt,
        sizes,
    )
    var as_bytes = max(4096, get64(sizes + 16))
    var scratch_bytes = max(4096, max(get64(sizes + 24), get64(sizes + 32)) + 256)

    var as_buf = vk.create_buffer_bda(
        as_bytes,
        extra_usage=BUFFER_USAGE_AS_STORAGE_KHR | BUFFER_USAGE_DEVICE_ADDRESS,
        host_visible=False,
    )
    if scratch_out.size < scratch_bytes:
        if scratch_out.size > 0:
            vk.destroy_buffer(scratch_out)
        scratch_out = vk.create_buffer_bda(
            scratch_bytes,
            extra_usage=BUFFER_USAGE_DEVICE_ADDRESS,
            host_visible=False,
        )
    var scratch_addr = (vk.buffer_device_address(scratch_out) + 255) & ~UInt64(255)

    vk.rewind()
    var create_info = vk.scratch(64)
    put32(create_info, TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
    put64(create_info + 24, as_buf.vk_buffer)
    put64(create_info + 40, as_bytes)
    put32(create_info + 48, AS_TYPE_BOTTOM_LEVEL_KHR)
    var slot = vk.scratch(8)
    if not vk.check(
        call4(vk.device_call("vkCreateAccelerationStructureKHR"), vk.device, create_info, 0, slot),
        "vkCreateAccelerationStructureKHR(BLAS-Alpha)",
    ):
        vk.destroy_buffer(as_buf)
        return out^
    var handle = get64(slot)

    vk.rewind()
    var geom2 = vk.scratch(96)
    put32(geom2, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
    put32(geom2 + 16, GEOMETRY_TYPE_TRIANGLES_KHR)
    put32(geom2 + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR)
    put32(geom2 + 24 + 16, FORMAT_R32G32B32_SFLOAT)
    put64(geom2 + 24 + 24, Int(vertex_addr))
    put64(geom2 + 24 + 32, vertex_stride)
    put32(geom2 + 24 + 40, UInt32(max_vertex))
    put32(geom2 + 24 + 44, INDEX_TYPE_UINT32)
    put64(geom2 + 24 + 48, Int(index_addr))
    put32(geom2 + 88, geom_flags)

    var binfo2 = vk.scratch(80)
    put32(binfo2, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
    put32(binfo2 + 16, AS_TYPE_BOTTOM_LEVEL_KHR)
    put32(binfo2 + 20, as_flags)
    put32(binfo2 + 24, BUILD_AS_MODE_BUILD_KHR)
    put64(binfo2 + 40, handle)
    put32(binfo2 + 48, 1)
    put64(binfo2 + 56, geom2)
    put64(binfo2 + 72, Int(scratch_addr))

    var range_info = vk.scratch(16)
    put32(range_info, UInt32(triangle_count))
    var pp_range = vk.scratch(8)
    put64(pp_range, range_info)
    _ = call4(vk.device_call("vkCmdBuildAccelerationStructuresKHR"), vk.command_buffers[frame], 1, binfo2, pp_range)

    vk.rewind()
    var addr_info = vk.scratch(24)
    put32(addr_info, TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR)
    put64(addr_info + 16, handle)
    comptime CallASDA = def(Int, Int) thin abi("C") -> UInt64
    var fn_asda = vk.device_call("vkGetAccelerationStructureDeviceAddressKHR")
    var as_da = Pointer(to=fn_asda).unsafe_bitcast[CallASDA]()[unsafe_offset=0](vk.device, addr_info)

    out.handle = handle
    out.buffer = as_buf^
    out.device_address = as_da
    return out^


def ensure_tlas_capacity_mixed_opaque(
    mut vk: VulkanGpu,
    mut tlas: VulkanAccelStruct,
    instances_addr: UInt64,
    max_instances_cap: Int,
    mut scratch_out: GpuBuffer,
) -> Bool:
    """Ensures `tlas` capacity with `geometry.flags = 0` so per-instance `FORCE_OPAQUE` / `FORCE_NO_OPAQUE`
    controls hardware RT auto-commit vs alpha-cutout candidate confirmation!"""
    if not vk.has_ray_query or max_instances_cap <= 0:
        return False
    vk.rewind()
    var geom = vk.scratch(96)
    put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
    put32(geom + 16, GEOMETRY_TYPE_INSTANCES_KHR)
    put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
    put64(geom + 24 + 24, Int(instances_addr))
    put32(geom + 88, UInt32(0))

    var build_info = vk.scratch(80)
    put32(build_info, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
    put32(build_info + 16, AS_TYPE_TOP_LEVEL_KHR)
    put32(build_info + 20, BUILD_AS_PREFER_FAST_TRACE_KHR)
    put32(build_info + 24, BUILD_AS_MODE_BUILD_KHR)
    put32(build_info + 48, 1)
    put64(build_info + 56, geom)

    var prim_cnt = vk.scratch(8)
    put32(prim_cnt, UInt32(max_instances_cap))
    var sizes = vk.scratch(40)
    put32(sizes, TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
    _ = call5(
        vk.device_call("vkGetAccelerationStructureBuildSizesKHR"),
        vk.device,
        BUILD_AS_TYPE_DEVICE_KHR,
        build_info,
        prim_cnt,
        sizes,
    )
    var as_bytes = max(4096, get64(sizes + 16))
    var scratch_bytes = max(4096, get64(sizes + 32) + 256)
    if scratch_out.size < scratch_bytes:
        if scratch_out.size > 0:
            vk.destroy_buffer(scratch_out)
        scratch_out = vk.create_buffer_bda(
            scratch_bytes,
            extra_usage=BUFFER_USAGE_DEVICE_ADDRESS,
            host_visible=False,
        )
    if tlas.handle != 0 and tlas.buffer.size >= as_bytes:
        return False
    if tlas.handle != 0:
        vk.destroy_accel_struct(tlas)
    var as_buf = vk.create_buffer_bda(
        as_bytes,
        extra_usage=BUFFER_USAGE_AS_STORAGE_KHR | BUFFER_USAGE_DEVICE_ADDRESS,
        host_visible=False,
    )
    vk.rewind()
    var create_info = vk.scratch(64)
    put32(create_info, TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
    put64(create_info + 24, as_buf.vk_buffer)
    put64(create_info + 40, as_bytes)
    put32(create_info + 48, AS_TYPE_TOP_LEVEL_KHR)
    var slot = vk.scratch(8)
    if not vk.check(
        call4(vk.device_call("vkCreateAccelerationStructureKHR"), vk.device, create_info, 0, slot),
        "vkCreateAccelerationStructureKHR(TLAS-Mixed)",
    ):
        vk.destroy_buffer(as_buf)
        return False
    tlas.handle = get64(slot)
    tlas.buffer = as_buf^
    return True


def record_build_tlas_mixed_opaque(
    mut vk: VulkanGpu,
    frame: Int,
    tlas: VulkanAccelStruct,
    instances_addr: UInt64,
    instance_count: Int,
    scratch_buf: GpuBuffer,
):
    """Records an inline `vkCmdBuildAccelerationStructuresKHR` into `tlas` with `geometry.flags = 0`."""
    if not vk.has_ray_query or tlas.handle == 0 or instance_count < 0 or scratch_buf.size == 0:
        return
    var scratch_addr = (vk.buffer_device_address(scratch_buf) + 255) & ~UInt64(255)
    vk.rewind()
    var geom = vk.scratch(96)
    put32(geom, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
    put32(geom + 16, GEOMETRY_TYPE_INSTANCES_KHR)
    put32(geom + 24 + 0, TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
    put64(geom + 24 + 24, Int(instances_addr))
    put32(geom + 88, UInt32(0))

    var binfo = vk.scratch(80)
    put32(binfo, TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
    put32(binfo + 16, AS_TYPE_TOP_LEVEL_KHR)
    put32(binfo + 20, BUILD_AS_PREFER_FAST_TRACE_KHR)
    put32(binfo + 24, BUILD_AS_MODE_BUILD_KHR)
    put64(binfo + 40, tlas.handle)
    put32(binfo + 48, 1)
    put64(binfo + 56, geom)
    put64(binfo + 72, Int(scratch_addr))

    var range_info = vk.scratch(16)
    put32(range_info, UInt32(instance_count))
    var pp_range = vk.scratch(8)
    put64(pp_range, range_info)
    _ = call4(vk.device_call("vkCmdBuildAccelerationStructuresKHR"), vk.command_buffers[frame], 1, binfo, pp_range)


def emit_decode_vertex_color_rgb_spv(
    mut s: AaaSpirvModule,
    meta: UInt32,
    is_furnace: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Decodes 24-bit sRGB vertex color from `vertex_words[11][8..31]`:
    - When `(meta & 2) != 0` (`is_terrain_splat`) OR `(meta >> 8) == 0` OR `(meta & 0xFFFF0000) == 0xFFFF0000`
      OR `is_furnace`: returns `(1.0, 1.0, 1.0)`.
    - Else: converts sRGB -> linear via `c * c * (0.75 + 0.25 * c)` (`1.0 -> 1.0000` exact!)."""
    var is_splat_m = s.ine(s.bit_and(meta, s.uint(2)), s.uint(0))
    var rgb24 = s.shr(meta, s.uint(8))
    var is_zero_c = s.ieq(rgb24, s.uint(0))
    var is_legacy_c = s.ieq(s.bit_and(meta, s.uint(0xFFFF0000)), s.uint(0xFFFF0000))
    var force_white = s.logical_or(
        s.logical_or(is_splat_m, is_furnace),
        s.logical_or(is_zero_c, is_legacy_c),
    )

    var inv255 = s.float(1.0 / 255.0)
    var rs = s.fmul(s.float_from_uint(s.bit_and(rgb24, s.uint(0xFF))), inv255)
    var gs = s.fmul(s.float_from_uint(s.bit_and(s.shr(rgb24, s.uint(8)), s.uint(0xFF))), inv255)
    var bs = s.fmul(s.float_from_uint(s.bit_and(s.shr(rgb24, s.uint(16)), s.uint(0xFF))), inv255)

    var rl = s.fmul(s.fmul(rs, rs), s.fma(rs, s.float(0.25), s.float(0.75)))
    var gl = s.fmul(s.fmul(gs, gs), s.fma(gs, s.float(0.25), s.float(0.75)))
    var bl = s.fmul(s.fmul(bs, bs), s.fma(bs, s.float(0.25), s.float(0.75)))
    return (
        s.select_f(force_white, s.float(1.0), rl),
        s.select_f(force_white, s.float(1.0), gl),
        s.select_f(force_white, s.float(1.0), bl),
    )


def emit_window_suite_prng_spv(
    mut s: AaaSpirvModule,
    inst_id: UInt32,
    uv_u: UInt32,
    uv_v: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Shared `4`-bay coherent office suite + per-pane PRNG across primary `_emit_parallax_windowbox_spv`,
    `surface_cache_update_shader`, and secondary `emit_rt_textured_hit_shade`!
    Returns `(suite_occ_f01, suite_kelvin_f01, pane_h0_f01, pane_h1_f01, suite_hash_u32)`."""
    var cell_x = s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(s.fadd(uv_u, s.float(512.0)))))
    var cell_y = s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(s.fadd(uv_v, s.float(512.0)))))
    var suite_x = s.shr(cell_x, s.uint(2))  # floor(cell_x * 0.25) -> 4-bay office suites!

    var st_seed = s.bit_xor(
        s.imul(suite_x, s.uint(73856093)),
        s.bit_xor(s.imul(cell_y, s.uint(19349663)), s.imul(inst_id, s.uint(83492791))),
    )
    var st0 = s.iadd(s.imul(st_seed, s.uint(747796405)), s.uint(2891336453))
    var stw0 = s.imul(s.bit_xor(s.shr(st0, s.iadd(s.shr(st0, s.uint(28)), s.uint(4))), st0), s.uint(277803737))
    var sh0 = s.bit_xor(s.shr(stw0, s.uint(22)), stw0)
    var suite_occ = s.fmul(s.float_from_uint(s.bit_and(sh0, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))

    var st1 = s.iadd(s.imul(sh0, s.uint(747796405)), s.uint(2891336453))
    var stw1 = s.imul(s.bit_xor(s.shr(st1, s.iadd(s.shr(st1, s.uint(28)), s.uint(4))), st1), s.uint(277803737))
    var sh1 = s.bit_xor(s.shr(stw1, s.uint(22)), stw1)
    var suite_kelvin = s.fmul(s.float_from_uint(s.bit_and(sh1, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))

    var pn_seed = s.bit_xor(
        s.imul(cell_x, s.uint(73856093)),
        s.bit_xor(s.imul(cell_y, s.uint(19349663)), s.imul(inst_id, s.uint(83492791))),
    )
    var pn0 = s.iadd(s.imul(pn_seed, s.uint(747796405)), s.uint(2891336453))
    var pnw0 = s.imul(s.bit_xor(s.shr(pn0, s.iadd(s.shr(pn0, s.uint(28)), s.uint(4))), pn0), s.uint(277803737))
    var ph0 = s.bit_xor(s.shr(pnw0, s.uint(22)), pnw0)
    var pane_h0 = s.fmul(s.float_from_uint(s.bit_and(ph0, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))

    var pn1 = s.iadd(s.imul(ph0, s.uint(747796405)), s.uint(2891336453))
    var pnw1 = s.imul(s.bit_xor(s.shr(pn1, s.iadd(s.shr(pn1, s.uint(28)), s.uint(4))), pn1), s.uint(277803737))
    var ph1 = s.bit_xor(s.shr(pnw1, s.uint(22)), pnw1)
    var pane_h1 = s.fmul(s.float_from_uint(s.bit_and(ph1, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))
    return (suite_occ, suite_kelvin, pane_h0, pane_h1, sh0)


def _emit_rt_candidate_alpha_loop(
    mut s: AaaSpirvModule,
    rq: UInt32,
    b_verts: UInt32 = 0,
    b_indices: UInt32 = 0,
    b_instances: UInt32 = 0,
    b_materials: UInt32 = 0,
    bindless_arr: Tuple[UInt32, UInt32] = (0, 0),
):
    """Hardware-accelerated `SPV_KHR_ray_query` traversal loop:
    99% of opaque BLAS/instances (`FORCE_OPAQUE`) auto-commit in RT silicon; non-opaque foliage/fence
    candidates (`FORCE_NO_OPAQUE`) test `bindless_arr[t_alb]` alpha >= `0.48` via candidate barycentrics!"""
    s.begin_loop()
    var cont = s.ray_query_proceed(rq)
    s.break_if(s.logical_not(cont))
    if b_verts != 0 and b_indices != 0 and b_instances != 0 and b_materials != 0 and bindless_arr[1] != 0:
        var cand_type = s.ray_query_candidate_type(rq)
        s.begin_if(s.ieq(cand_type, s.uint(0)))
        var c_inst = s.ray_query_candidate_instance_id(rq)
        var c_prim = s.ray_query_candidate_prim_index(rq)
        var c_bary = s.ray_query_candidate_barycentrics(rq)
        var cb1 = s.extract_f(c_bary, 0)
        var cb2 = s.extract_f(c_bary, 1)
        var cb0 = s.fsub(s.fsub(s.float(1.0), cb1), cb2)

        var c_ibase = s.imul(c_inst, s.uint(UInt32(AAA_INSTANCE_WORDS)))
        var c_fidx = s.load_word(b_instances, s.iadd(c_ibase, s.uint(38)))
        var c_w39 = s.load_word(b_instances, s.iadd(c_ibase, s.uint(39)))
        var c_mat = s.bit_and(c_w39, s.uint(0xFFFF))
        var c_mbase = s.imul(c_mat, s.uint(UInt32(AAA_MATERIAL_WORDS)))
        var c_raw_sf = s.load_float(b_materials, s.iadd(c_mbase, s.uint(3)))
        var c_tex_pk = s.load_word(b_materials, s.iadd(c_mbase, s.uint(23)))
        var c_talb = s.bit_and(c_tex_pk, s.uint(0xFF))

        var pass_alpha = s.uvar(s.uint(1))
        s.begin_if(s.logical_and(s.flt(c_raw_sf, s.float(0.0)), s.ult(c_talb, s.uint(0xFF))))
        var c_ioff = s.iadd(c_fidx, s.imul(c_prim, s.uint(3)))
        var c_v0 = s.imul(s.load_word(b_indices, c_ioff), s.uint(12))
        var c_v1 = s.imul(s.load_word(b_indices, s.iadd(c_ioff, s.uint(1))), s.uint(12))
        var c_v2 = s.imul(s.load_word(b_indices, s.iadd(c_ioff, s.uint(2))), s.uint(12))
        var cu = s.fma(
            s.load_float(b_verts, s.iadd(c_v0, s.uint(9))), cb0,
            s.fma(s.load_float(b_verts, s.iadd(c_v1, s.uint(9))), cb1, s.fmul(s.load_float(b_verts, s.iadd(c_v2, s.uint(9))), cb2)),
        )
        var cv = s.fma(
            s.load_float(b_verts, s.iadd(c_v0, s.uint(10))), cb0,
            s.fma(s.load_float(b_verts, s.iadd(c_v1, s.uint(10))), cb1, s.fmul(s.load_float(b_verts, s.iadd(c_v2, s.uint(10))), cb2)),
        )
        var s_ca = s.sample_bindless_lod(bindless_arr, c_talb, s.vec2(cu, cv), s.float(1.0))
        var ca_boost = s.fmul(s.extract_f(s_ca, 3), s.float(1.18))
        s.store(pass_alpha, s.select_u(s.fge(ca_boost, s.float(0.48)), s.uint(1), s.uint(0)))
        s.end_if()

        s.begin_if(s.ine(s.load_u(pass_alpha), s.uint(0)))
        s.ray_query_confirm_intersection(rq)
        s.end_if()
        s.end_if()
    else:
        s.ray_query_confirm_intersection(rq)
    s.end_loop()


def emit_rt_shadow_query(
    mut s: AaaSpirvModule,
    tlas_var: UInt32,
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    lx: UInt32,
    ly: UInt32,
    lz: UInt32,
    t_max: UInt32,
    b_verts: UInt32 = 0,
    b_indices: UInt32 = 0,
    b_instances: UInt32 = 0,
    b_materials: UInt32 = 0,
    bindless_arr: Tuple[UInt32, UInt32] = (0, 0),
) -> UInt32:
    """Traces an inline `SPV_KHR_ray_query` shadow occlusion ray (`1.0` = unoccluded, `0.0` = occluded).
    Supports candidate alpha-cutout confirmation when bindless buffers are supplied!"""
    var rq = s.ray_query_var()
    var bias = s.float(0.004)
    var ro = s.vec3(s.fma(nx, bias, ox), s.fma(ny, bias, oy), s.fma(nz, bias, oz))
    var rd = s.vec3(lx, ly, lz)
    var rflags = RAY_FLAG_TERMINATE_ON_FIRST_HIT_KHR if bindless_arr[1] != 0 else RAY_FLAG_SHADOW_OCCLUSION
    s.ray_query_init(
        rq,
        tlas_var,
        s.uint(rflags),
        s.uint(AAA_RT_MASK_OPAQUE),
        ro,
        s.float(0.002),
        rd,
        t_max,
    )
    _emit_rt_candidate_alpha_loop(s, rq, b_verts, b_indices, b_instances, b_materials, bindless_arr)

    var hit_kind = s.ray_query_committed_type(rq)
    return s.select_f(s.ieq(hit_kind, s.uint(0)), s.float(1.0), s.float(0.0))


def emit_rt_surface_cache_ray(
    mut s: AaaSpirvModule,
    tlas_var: UInt32,
    b_surfcache: UInt32,
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
    t_min: UInt32,
    t_max: UInt32,
    miss_r: UInt32,
    miss_g: UInt32,
    miss_b: UInt32,
    b_instances: UInt32 = 0,
    b_materials: UInt32 = 0,
    b_verts: UInt32 = 0,
    b_indices: UInt32 = 0,
    bindless_arr: Tuple[UInt32, UInt32] = (0, 0),
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Traces an inline `SPV_KHR_ray_query` closest-hit ray against `TLAS`.
    On hit: fetches `O(1)` Surface-Cache radiance `(Lr, Lg, Lb)` at `(custom_index + prim_index) * 4`,
    or falls back to material albedo/emissive via `b_instances[39]` for instances `>= 1,024` (`0xFFFFFF`)
    so sky radiance never leaks through solid geometry!
    On miss: returns `(miss_r, miss_g, miss_b)`.
    Returns `(rad_r, rad_g, rad_b, hit_flag_u32, hit_t_f32)`.
    """
    var rq = s.ray_query_var()
    var rflags = RAY_FLAG_NONE_KHR if bindless_arr[1] != 0 else RAY_FLAG_OPAQUE_KHR
    s.ray_query_init(
        rq,
        tlas_var,
        s.uint(rflags),
        s.uint(0xFF),
        s.vec3(ox, oy, oz),
        t_min,
        s.vec3(dx, dy, dz),
        t_max,
    )
    _emit_rt_candidate_alpha_loop(s, rq, b_verts, b_indices, b_instances, b_materials, bindless_arr)

    var out_r = s.fvar(miss_r)
    var out_g = s.fvar(miss_g)
    var out_b = s.fvar(miss_b)
    var out_hit = s.uvar(s.uint(0))
    var out_t = s.fvar(t_max)

    var hit_kind = s.ray_query_committed_type(rq)
    s.begin_if(s.ine(hit_kind, s.uint(0)))
    var sc_base = s.ray_query_committed_custom_index(rq)
    var prim_id = s.ray_query_committed_prim_index(rq)
    s.begin_if(s.ine(sc_base, s.uint(0xFFFFFF)))
    var sc_word = s.imul(s.iadd(sc_base, prim_id), s.uint(4))
    s.store(out_r, s.load_float(b_surfcache, sc_word))
    s.store(out_g, s.load_float(b_surfcache, s.iadd(sc_word, s.uint(1))))
    s.store(out_b, s.load_float(b_surfcache, s.iadd(sc_word, s.uint(2))))
    if b_instances != 0 and b_materials != 0:
        s.begin_else()
        var inst_id = s.ray_query_committed_instance_id(rq)
        var w39 = s.load_word(
            b_instances,
            s.iadd(s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS))), s.uint(39)),
        )
        var mat_id = s.bit_and(w39, s.uint(0xFFFF))
        var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
        var ar = s.load_float(b_materials, m_base)
        var ag = s.load_float(b_materials, s.iadd(m_base, s.uint(1)))
        var ab = s.load_float(b_materials, s.iadd(m_base, s.uint(2)))
        var met = s.load_float(b_materials, s.iadd(m_base, s.uint(5)))
        var er = s.load_float(b_materials, s.iadd(m_base, s.uint(8)))
        var eg = s.load_float(b_materials, s.iadd(m_base, s.uint(9)))
        var eb = s.load_float(b_materials, s.iadd(m_base, s.uint(10)))
        var kd_s = s.fmul(s.fsub(s.float(1.0), met), s.float(0.45))
        s.store(out_r, s.fma(ar, s.fmul(miss_r, kd_s), er))
        s.store(out_g, s.fma(ag, s.fmul(miss_g, kd_s), eg))
        s.store(out_b, s.fma(ab, s.fmul(miss_b, kd_s), eb))
    s.end_if()
    s.store(out_hit, s.uint(1))
    s.store(out_t, s.ray_query_committed_t(rq))
    s.end_if()

    return (
        s.load_f(out_r),
        s.load_f(out_g),
        s.load_f(out_b),
        s.load_u(out_hit),
        s.load_f(out_t),
    )


def emit_froxel_local_lighting_at_hit(
    mut s: AaaSpirvModule,
    b_clusters: UInt32,
    b_globals: UInt32,
    hx: UInt32,
    hy: UInt32,
    hz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    eval_specular: Bool = False,
    vx: UInt32 = 0,
    vy: UInt32 = 0,
    vz: UInt32 = 0,
    hit_rough: UInt32 = 0,
    f0_r: UInt32 = 0,
    f0_g: UInt32 = 0,
    f0_b: UInt32 = 0,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]:
    """`O(1)` Camera-Froxel Clustered Punctual & `LTC` Area Diffuse + Top-2 Closest-Point `LTC_TUBE`/`RECT`
    Specular Highlights at `P_hit` (with off-frustum fallback for `LumenSurfaceCache` GI triangles)!
    Returns `(diff_r, diff_g, diff_b, spec_r, spec_g, spec_b)`."""
    var acc_r = s.fvar(s.float(0.0))
    var acc_g = s.fvar(s.float(0.0))
    var acc_b = s.fvar(s.float(0.0))
    var sp0_lum = s.fvar(s.float(0.0))
    var sp0_r = s.fvar(s.float(0.0)); var sp0_g = s.fvar(s.float(0.0)); var sp0_b = s.fvar(s.float(0.0))
    var sp1_lum = s.fvar(s.float(0.0))
    var sp1_r = s.fvar(s.float(0.0)); var sp1_g = s.fvar(s.float(0.0)); var sp1_b = s.fvar(s.float(0.0))

    var total_l = s.load_word(b_clusters, s.uint(0))
    s.begin_if(s.ine(total_l, s.uint(0)))
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))

    var rx_c = s.fsub(hx, cam_x)
    var ry_c = s.fsub(hy, cam_y)
    var rz_c = s.fsub(hz, cam_z)
    var vz_h = emit_vec3_dot(s, rx_c, ry_c, rz_c, fwd_x, fwd_y, fwd_z)
    var near_cl = s.load_float(b_clusters, s.uint(1))
    var far_cl = s.load_float(b_clusters, s.uint(2))
    var log_ratio = s.load_float(b_clusters, s.uint(3))
    var tan_x = s.load_float(b_clusters, s.uint(4))
    var tan_y = s.load_float(b_clusters, s.uint(5))

    var use_cluster_list = s.uvar(s.uint(0))
    var c_cnt_var = s.uvar(s.umin(total_l, s.uint(20)))
    var list_base_var = s.uvar(s.uint(0))

    s.begin_if(s.logical_and(s.fgt(vz_h, near_cl), s.flt(vz_h, far_cl)))
    var vx_h = emit_vec3_dot(s, rx_c, ry_c, rz_c, rgt_x, rgt_y, rgt_z)
    var vy_h = emit_vec3_dot(s, rx_c, ry_c, rz_c, up_x, up_y, up_z)
    var ndc_x = s.fdiv(vx_h, s.nmax(s.float(1e-4), s.fmul(vz_h, tan_x)))
    var ndc_y = s.fdiv(vy_h, s.nmax(s.float(1e-4), s.fmul(vz_h, tan_y)))
    var in_xy = s.logical_and(
        s.flt(s.fabs(ndc_x), s.float(1.0)),
        s.flt(s.fabs(ndc_y), s.float(1.0)),
    )
    s.begin_if(in_xy)
    var u_h = s.nclamp(s.fma(ndc_x, s.float(0.5), s.float(0.5)), s.float(0.0), s.float(0.9999))
    var v_h = s.nclamp(s.fma(s.fneg(ndc_y), s.float(0.5), s.float(0.5)), s.float(0.0), s.float(0.9999))
    var cx = s.umin(s.uint(_RT_CLUSTER_X - 1), s.saturate_uint(s.fmul(u_h, s.float(16.0))))
    var cy = s.umin(s.uint(_RT_CLUSTER_Y - 1), s.saturate_uint(s.fmul(v_h, s.float(9.0))))
    var zn = s.nclamp(
        s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(vz_h, near_cl))), s.nmax(s.float(1e-4), log_ratio)),
        s.float(0.0),
        s.float(0.9999),
    )
    var cz = s.umin(s.uint(_RT_CLUSTER_Z - 1), s.saturate_uint(s.fmul(zn, s.float(24.0))))
    var cid = s.iadd(s.imul(cz, s.uint(144)), s.iadd(s.imul(cy, s.uint(16)), cx))
    s.store(use_cluster_list, s.uint(1))
    s.store(c_cnt_var, s.umin(s.load_word(b_clusters, s.iadd(s.uint(_RT_CLUSTER_COUNTS_BASE), cid)), s.uint(16)))
    s.store(list_base_var, s.iadd(s.uint(_RT_CLUSTER_INDICES_BASE), s.imul(cid, s.uint(_RT_MAX_LIGHTS_PER_CLUSTER))))
    s.end_if()
    s.end_if()

    var c_cnt = s.load_u(c_cnt_var)
    var list_base = s.load_u(list_base_var)
    var from_cl = s.ine(s.load_u(use_cluster_list), s.uint(0))
    var inv_pi = s.float(0.318309886)

    var rvx = s.float(0.0); var rvy = s.float(1.0); var rvz = s.float(0.0)
    if eval_specular:
        var two_nv_h = s.fmul(s.float(2.0), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
        rvx = s.fsub(s.fmul(nx, two_nv_h), vx)
        rvy = s.fsub(s.fmul(ny, two_nv_h), vy)
        rvz = s.fsub(s.fmul(nz, two_nv_h), vz)

    var ki_v = s.uvar(s.uint(0))
    s.begin_loop()
    var ki = s.load_u(ki_v)
    s.break_if(s.uge(ki, c_cnt))
    var li = s.select_u(from_cl, s.load_word(b_clusters, s.iadd(list_base, ki)), ki)
    var l_off = s.iadd(s.uint(_RT_CLUSTER_LIGHTS_BASE), s.imul(li, s.uint(_RT_CLUSTER_LIGHT_WORDS)))

    var lpx = s.load_float(b_clusters, l_off)
    var lpy = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
    var lpz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))
    var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
    var lrr = s.load_float(b_clusters, s.iadd(l_off, s.uint(4)))
    var lrg = s.load_float(b_clusters, s.iadd(l_off, s.uint(5)))
    var lrb = s.load_float(b_clusters, s.iadd(l_off, s.uint(6)))
    var kind = s.bit_and(s.load_word(b_clusters, s.iadd(l_off, s.uint(7))), s.uint(0xFF))
    var aux_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(8)))
    var aux_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(9)))
    var aux_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(10)))
    var p0_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(11)))
    var avx_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(12)))
    var avx_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(13)))
    var avx_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(14)))
    var p1_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(15)))

    # Closest point on Tube (kind==3) or Rect (kind==2) emitter for accurate near-field neon falloff
    var tx_e = s.fvar(lpx)
    var ty_e = s.fvar(lpy)
    var tz_e = s.fvar(lpz)
    var rpx_e = s.fvar(lpx)
    var rpy_e = s.fvar(lpy)
    var rpz_e = s.fvar(lpz)
    var area_gain = s.fvar(inv_pi)
    var src_rad = s.fvar(s.float(0.05))

    s.begin_if(s.ieq(kind, s.uint(3)))
    var cwx = s.fsub(hx, lpx); var cwy = s.fsub(hy, lpy); var cwz = s.fsub(hz, lpz)
    var u_l2 = s.nmax(s.float(1e-5), emit_vec3_dot(s, aux_x, aux_y, aux_z, aux_x, aux_y, aux_z))
    var t_seg = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx, cwy, cwz, aux_x, aux_y, aux_z), u_l2), s.float(-1.0), s.float(1.0))
    s.store(tx_e, s.fma(aux_x, t_seg, lpx))
    s.store(ty_e, s.fma(aux_y, t_seg, lpy))
    s.store(tz_e, s.fma(aux_z, t_seg, lpz))
    if eval_specular:
        # Karis closest point on tube segment to reflection ray `R = (rvx, rvy, rvz)`!
        var l0x = s.fsub(lpx, hx); var l0y = s.fsub(lpy, hy); var l0z = s.fsub(lpz, hz)
        var r_dot_u = emit_vec3_dot(s, rvx, rvy, rvz, aux_x, aux_y, aux_z)
        var r_dot_l0 = emit_vec3_dot(s, rvx, rvy, rvz, l0x, l0y, l0z)
        var l0_dot_u = emit_vec3_dot(s, l0x, l0y, l0z, aux_x, aux_y, aux_z)
        var denom_t = s.nmax(s.float(1e-4), s.fsub(u_l2, s.fmul(r_dot_u, r_dot_u)))
        var t_spec = s.nclamp(s.fdiv(s.fsub(s.fmul(r_dot_l0, r_dot_u), l0_dot_u), denom_t), s.float(-1.0), s.float(1.0))
        s.store(rpx_e, s.fma(aux_x, t_spec, lpx))
        s.store(rpy_e, s.fma(aux_y, t_spec, lpy))
        s.store(rpz_e, s.fma(aux_z, t_spec, lpz))
    var seg_half = s.fsqrt(u_l2)
    s.store(area_gain, s.fmul(s.float(0.63661977), s.nmax(s.float(0.08), s.fmul(seg_half, p1_f))))
    s.store(src_rad, s.nmax(s.float(0.04), p1_f))
    s.end_if()

    s.begin_if(s.ieq(kind, s.uint(2)))
    var cwx2 = s.fsub(hx, lpx); var cwy2 = s.fsub(hy, lpy); var cwz2 = s.fsub(hz, lpz)
    var u2_l2 = s.nmax(s.float(1e-5), emit_vec3_dot(s, aux_x, aux_y, aux_z, aux_x, aux_y, aux_z))
    var v2_l2 = s.nmax(s.float(1e-5), emit_vec3_dot(s, avx_x, avx_y, avx_z, avx_x, avx_y, avx_z))
    var tu = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx2, cwy2, cwz2, aux_x, aux_y, aux_z), u2_l2), s.float(-1.0), s.float(1.0))
    var tv = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx2, cwy2, cwz2, avx_x, avx_y, avx_z), v2_l2), s.float(-1.0), s.float(1.0))
    var clx_r = s.fma(aux_x, tu, s.fma(avx_x, tv, lpx))
    var cly_r = s.fma(aux_y, tu, s.fma(avx_y, tv, lpy))
    var clz_r = s.fma(aux_z, tu, s.fma(avx_z, tv, lpz))
    s.store(tx_e, clx_r); s.store(ty_e, cly_r); s.store(tz_e, clz_r)
    if eval_specular:
        var d_est = s.fsqrt(s.nmax(s.float(0.04), emit_vec3_dot(s, cwx2, cwy2, cwz2, cwx2, cwy2, cwz2)))
        var rhit_x = s.fsub(s.fma(rvx, d_est, hx), lpx)
        var rhit_y = s.fsub(s.fma(rvy, d_est, hy), lpy)
        var rhit_z = s.fsub(s.fma(rvz, d_est, hz), lpz)
        var su_r = s.nclamp(s.fdiv(emit_vec3_dot(s, rhit_x, rhit_y, rhit_z, aux_x, aux_y, aux_z), u2_l2), s.float(-1.0), s.float(1.0))
        var sv_r = s.nclamp(s.fdiv(emit_vec3_dot(s, rhit_x, rhit_y, rhit_z, avx_x, avx_y, avx_z), v2_l2), s.float(-1.0), s.float(1.0))
        s.store(rpx_e, s.fma(aux_x, su_r, s.fma(avx_x, sv_r, lpx)))
        s.store(rpy_e, s.fma(aux_y, su_r, s.fma(avx_y, sv_r, lpy)))
        s.store(rpz_e, s.fma(aux_z, su_r, s.fma(avx_z, sv_r, lpz)))
    var quad_area = s.fmul(s.float(4.0), s.fsqrt(s.fmul(u2_l2, v2_l2)))
    s.store(area_gain, s.fmul(inv_pi, s.nmin(s.float(16.0), quad_area)))
    s.store(src_rad, s.nmin(s.float(2.5), s.fmul(s.float(0.5), s.fsqrt(s.fadd(u2_l2, v2_l2)))))
    s.end_if()

    var dlx = s.fsub(s.load_f(tx_e), hx)
    var dly = s.fsub(s.load_f(ty_e), hy)
    var dlz = s.fsub(s.load_f(tz_e), hz)
    var d2 = s.nmax(s.float(0.04), emit_vec3_dot(s, dlx, dly, dlz, dlx, dly, dlz))
    var r2 = s.nmax(s.float(0.04), s.fmul(lrad, lrad))
    s.begin_if(s.flt(d2, r2))
    var inv_d = s.inverse_sqrt(d2)
    var lx = s.fmul(dlx, inv_d); var ly = s.fmul(dly, inv_d); var lz = s.fmul(dlz, inv_d)
    var ndotl = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, lx, ly, lz))

    var q = s.fdiv(d2, r2)
    var q2 = s.nclamp(s.fsub(s.float(1.0), s.fmul(q, q)), s.float(0.0), s.float(1.0))
    var atten = s.fvar(s.fdiv(s.fmul(q2, q2), s.fadd(d2, s.float(0.35))))
    s.begin_if(s.ieq(kind, s.uint(1)))
    var cos_c = s.fneg(emit_vec3_dot(s, lx, ly, lz, aux_x, aux_y, aux_z))
    var sw = s.nclamp(s.fmul(s.fsub(cos_c, p0_f), p1_f), s.float(0.0), s.float(1.0))
    s.store(atten, s.fmul(s.load_f(atten), s.fmul(sw, sw)))
    s.end_if()

    s.begin_if(s.fgt(ndotl, s.float(0.001)))
    var w_tot = s.fmul(ndotl, s.fmul(s.load_f(atten), s.load_f(area_gain)))
    s.store(acc_r, s.fma(lrr, w_tot, s.load_f(acc_r)))
    s.store(acc_g, s.fma(lrg, w_tot, s.load_f(acc_g)))
    s.store(acc_b, s.fma(lrb, w_tot, s.load_f(acc_b)))
    s.end_if()

    if eval_specular:
        var slx_r = s.fsub(s.load_f(rpx_e), hx)
        var sly_r = s.fsub(s.load_f(rpy_e), hy)
        var slz_r = s.fsub(s.load_f(rpz_e), hz)
        var sd2 = s.nmax(s.float(0.04), emit_vec3_dot(s, slx_r, sly_r, slz_r, slx_r, sly_r, slz_r))
        var sinv_d = s.inverse_sqrt(sd2)
        var slx = s.fmul(slx_r, sinv_d); var sly = s.fmul(sly_r, sinv_d); var slz = s.fmul(slz_r, sinv_d)
        var s_ndotl = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, slx, sly, slz))
        s.begin_if(s.fgt(s_ndotl, s.float(0.001)))
        var h_sp = emit_vec3_normalize(s, s.fadd(vx, slx), s.fadd(vy, sly), s.fadd(vz, slz))
        var ndoth_s = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, h_sp[0], h_sp[1], h_sp[2]))
        var a_hit = s.nmax(s.float(0.03), s.fmul(hit_rough, hit_rough))
        var a_rep = s.nclamp(s.fma(s.load_f(src_rad), s.fmul(sinv_d, s.float(0.5)), a_hit), s.float(0.03), s.float(1.0))
        var a2_s = s.fmul(a_hit, a_hit)
        var den_s = s.fma(s.fmul(ndoth_s, ndoth_s), s.fsub(a2_s, s.float(1.0)), s.float(1.0))
        var norm_rep = s.fdiv(a_hit, a_rep)
        var d_ggx_s = s.fdiv(
            s.fmul(a2_s, s.fmul(norm_rep, norm_rep)),
            s.nmax(s.float(1e-5), s.fmul(s.float(3.14159265), s.fmul(den_s, den_s))),
        )
        var sp_scale = s.nmin(s.float(12.0), s.fmul(d_ggx_s, s.fmul(s.float(0.25), s.fmul(s_ndotl, s.load_f(atten)))))
        var csr = s.fmul(lrr, s.fmul(f0_r, sp_scale))
        var csg = s.fmul(lrg, s.fmul(f0_g, sp_scale))
        var csb = s.fmul(lrb, s.fmul(f0_b, sp_scale))
        var c_lum = s.fma(csr, s.float(0.2126), s.fma(csg, s.float(0.7152), s.fmul(csb, s.float(0.0722))))
        # Retain Top-2 brightest area/local specular highlights at `P_hit`
        s.begin_if(s.fgt(c_lum, s.load_f(sp0_lum)))
        s.store(sp1_lum, s.load_f(sp0_lum))
        s.store(sp1_r, s.load_f(sp0_r)); s.store(sp1_g, s.load_f(sp0_g)); s.store(sp1_b, s.load_f(sp0_b))
        s.store(sp0_lum, c_lum)
        s.store(sp0_r, csr); s.store(sp0_g, csg); s.store(sp0_b, csb)
        s.begin_else()
        s.begin_if(s.fgt(c_lum, s.load_f(sp1_lum)))
        s.store(sp1_lum, c_lum)
        s.store(sp1_r, csr); s.store(sp1_g, csg); s.store(sp1_b, csb)
        s.end_if()
        s.end_if()
        s.end_if()
    s.end_if()

    s.store(ki_v, s.iadd(ki, s.uint(1)))
    s.end_loop()
    s.end_if()

    return (
        s.load_f(acc_r),
        s.load_f(acc_g),
        s.load_f(acc_b),
        s.fadd(s.load_f(sp0_r), s.load_f(sp1_r)),
        s.fadd(s.load_f(sp0_g), s.load_f(sp1_g)),
        s.fadd(s.load_f(sp0_b), s.load_f(sp1_b)),
    )


def emit_rt_textured_hit_shade(
    mut s: AaaSpirvModule,
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
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
    t_min: UInt32,
    t_max: UInt32,
    d_cam: UInt32,
    ray_rough_alpha: UInt32,
    miss_r: UInt32,
    miss_g: UInt32,
    miss_b: UInt32,
    trace_hit_sun_shadow: Bool = True,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Zero-Limit `O(1)` Barycentric Textured + Froxel-Lit Ray-Hit Shader across all `32,768` instances!
    Interpolates `(u, v)` and world normal via `OpRayQueryGetIntersectionBarycentricsKHR`, computes
    analytic ray-cone mip LOD, samples Bindless `BC7` Albedo/Emissive/ORM (+ 4-layer terrain splat &
    skyscraper interior glow), evaluates VSM/RT sun + `O(1)` camera-froxel local/LTC lighting at `P_hit`,
    and blends `SH9` + multi-bounce Surface-Cache diffuse irradiance!
    Returns `(rad_r, rad_g, rad_b, hit_flag_u32, hit_t_f32)`.
    """
    var rq = s.ray_query_var()
    s.ray_query_init(
        rq,
        tlas_var,
        s.uint(RAY_FLAG_NONE_KHR),
        s.uint(0xFF),
        s.vec3(ox, oy, oz),
        t_min,
        s.vec3(dx, dy, dz),
        t_max,
    )
    _emit_rt_candidate_alpha_loop(s, rq, b_verts, b_indices, b_instances, b_materials, bindless_arr)

    var out_r = s.fvar(miss_r)
    var out_g = s.fvar(miss_g)
    var out_b = s.fvar(miss_b)
    var out_hit = s.uvar(s.uint(0))
    var out_t = s.fvar(t_max)

    var hit_kind = s.ray_query_committed_type(rq)
    s.begin_if(s.ine(hit_kind, s.uint(0)))
    var t_hit = s.ray_query_committed_t(rq)
    var inst_id = s.ray_query_committed_instance_id(rq)
    var prim_id = s.ray_query_committed_prim_index(rq)
    var sc_base = s.ray_query_committed_custom_index(rq)
    var bary_v2 = s.ray_query_committed_barycentrics(rq)
    var is_ff = s.ray_query_committed_front_face(rq)

    var b1 = s.extract_f(bary_v2, 0)
    var b2 = s.extract_f(bary_v2, 1)
    var b0 = s.fsub(s.fsub(s.float(1.0), b1), b2)

    var hx = s.fma(dx, t_hit, ox)
    var hy = s.fma(dy, t_hit, oy)
    var hz = s.fma(dz, t_hit, oz)

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var uscale = s.nmax(s.float(0.01), s.load_float(b_instances, s.iadd(inst_base, s.uint(35))))
    var first_idx = s.load_word(b_instances, s.iadd(inst_base, s.uint(38)))
    var w39 = s.load_word(b_instances, s.iadd(inst_base, s.uint(39)))
    var mat_id = s.bit_and(w39, s.uint(0xFFFF))
    var double_sided = s.ine(s.bit_and(s.shr(w39, s.uint(16)), s.uint(3)), s.uint(0))

    var idx_off = s.iadd(first_idx, s.imul(prim_id, s.uint(3)))
    var vi0 = s.load_word(b_indices, idx_off)
    var vi1 = s.load_word(b_indices, s.iadd(idx_off, s.uint(1)))
    var vi2 = s.load_word(b_indices, s.iadd(idx_off, s.uint(2)))
    var v0_off = s.imul(vi0, s.uint(12))
    var v1_off = s.imul(vi1, s.uint(12))
    var v2_off = s.imul(vi2, s.uint(12))

    # Interpolate smooth world normal & orient toward incoming ray `-D`
    var n0x = s.load_float(b_verts, s.iadd(v0_off, s.uint(3))); var n0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(4))); var n0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(5)))
    var n1x = s.load_float(b_verts, s.iadd(v1_off, s.uint(3))); var n1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(4))); var n1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(5)))
    var n2x = s.load_float(b_verts, s.iadd(v2_off, s.uint(3))); var n2y = s.load_float(b_verts, s.iadd(v2_off, s.uint(4))); var n2z = s.load_float(b_verts, s.iadd(v2_off, s.uint(5)))
    var lnx = s.fma(n0x, b0, s.fma(n1x, b1, s.fmul(n2x, b2)))
    var lny = s.fma(n0y, b0, s.fma(n1y, b1, s.fmul(n2y, b2)))
    var lnz = s.fma(n0z, b0, s.fma(n1z, b1, s.fmul(n2z, b2)))
    var wn_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lnx, lny, lnz)
    var wn_u = emit_vec3_normalize(s, wn_raw[0], wn_raw[1], wn_raw[2])
    var n_dot_d = emit_vec3_dot(s, wn_u[0], wn_u[1], wn_u[2], dx, dy, dz)
    var need_flip = s.select_u(double_sided, s.select_u(s.fgt(n_dot_d, s.float(0.0)), s.uint(1), s.uint(0)), s.select_u(s.logical_or(s.logical_not(is_ff), s.fgt(n_dot_d, s.float(0.0))), s.uint(1), s.uint(0)))
    var flip_f = s.select_f(s.ine(need_flip, s.uint(0)), s.float(-1.0), s.float(1.0))
    var hnx = s.fmul(wn_u[0], flip_f)
    var hny = s.fmul(wn_u[1], flip_f)
    var hnz = s.fmul(wn_u[2], flip_f)

    # Interpolate `(hit_u, hit_v)` & compute analytic ray-cone mip LOD `hit_lod`
    var u0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(9))); var v0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(10)))
    var u1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(9))); var v1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(10)))
    var u2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(9))); var v2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(10)))
    var hit_u = s.fma(u0, b0, s.fma(u1, b1, s.fmul(u2, b2)))
    var hit_v = s.fma(v0, b0, s.fma(v1, b1, s.fmul(v2, b2)))

    var p0x = s.load_float(b_verts, v0_off); var p0y = s.load_float(b_verts, s.iadd(v0_off, s.uint(1))); var p0z = s.load_float(b_verts, s.iadd(v0_off, s.uint(2)))
    var p1x = s.load_float(b_verts, v1_off); var p1y = s.load_float(b_verts, s.iadd(v1_off, s.uint(1))); var p1z = s.load_float(b_verts, s.iadd(v1_off, s.uint(2)))
    var epx = s.fsub(p1x, p0x); var epy = s.fsub(p1y, p0y); var epz = s.fsub(p1z, p0z)
    var world_e = s.nmax(s.float(0.04), s.fmul(uscale, s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, epx, epy, epz, epx, epy, epz)))))
    var du_e = s.fsub(u1, u0); var dv_e = s.fsub(v1, v0)
    var uv_e = s.fsqrt(s.nmax(s.float(1e-8), s.fma(du_e, du_e, s.fmul(dv_e, dv_e))))
    var s_uv = s.nclamp(s.fdiv(uv_e, world_e), s.float(0.04), s.float(8.0))
    var tan_y_g = s.load_float(b_globals, s.uint(43))
    var theta_px = s.nmax(s.float(0.0005), s.fmul(tan_y_g, s.float(0.0028)))
    var cone_w = s.fma(s.fadd(d_cam, t_hit), theta_px, s.fmul(s.float(2.0), s.fmul(ray_rough_alpha, t_hit)))
    var hit_lod = s.nclamp(s.log2(s.nmax(s.float(1.0), s.fmul(s.fmul(cone_w, s_uv), s.float(256.0)))), s.float(0.0), s.float(6.0))

    # Sample Material + Bindless `BC7` Albedo/ORM (+ Terrain Splat & Interior Window Glow)
    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var alb_r_v = s.fvar(s.load_float(b_materials, m_base))
    var alb_g_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(1))))
    var alb_b_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(2))))
    var hit_rough_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(4))))
    var hit_metal_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(5))))
    var em_r_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(8))))
    var em_g_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(9))))
    var em_b_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(10))))
    var hit_ao_v = s.fvar(s.float(1.0))

    var tex_pk = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))
    var t_alb = s.bit_and(tex_pk, s.uint(0xFF))
    var t_orm = s.bit_and(s.shr(tex_pk, s.uint(16)), s.uint(0xFF))
    var t_det = s.bit_and(s.shr(tex_pk, s.uint(24)), s.uint(AAA_MAT_DET_NONE))
    var has_int_para = s.ine(s.bit_and(tex_pk, s.uint(AAA_MAT_INTERIOR_PARALLAX)), s.uint(0))
    var hit_uv_v2 = s.vec2(hit_u, hit_v)

    var r_flags = s.load_word(b_globals, s.uint(99))
    var is_furnace = s.ine(s.bit_and(r_flags, s.uint(16)), s.uint(0))

    var meta0 = s.load_word(b_verts, s.iadd(v0_off, s.uint(11)))
    var meta1 = s.load_word(b_verts, s.iadd(v1_off, s.uint(11)))
    var meta2 = s.load_word(b_verts, s.iadd(v2_off, s.uint(11)))
    var is_splat = s.ine(s.bit_and(meta0, s.uint(2)), s.uint(0))

    s.begin_if(s.logical_and(is_splat, s.ult(t_alb, s.uint(250))))
    var inv255 = s.float(1.0 / 255.0)
    var ws = s.fmul(s.fma(s.float_from_uint(s.bit_and(s.shr(meta0, s.uint(8)), s.uint(0xFF))), b0, s.fma(s.float_from_uint(s.bit_and(s.shr(meta1, s.uint(8)), s.uint(0xFF))), b1, s.fmul(s.float_from_uint(s.bit_and(s.shr(meta2, s.uint(8)), s.uint(0xFF))), b2))), inv255)
    var wr = s.fmul(s.fma(s.float_from_uint(s.bit_and(s.shr(meta0, s.uint(16)), s.uint(0xFF))), b0, s.fma(s.float_from_uint(s.bit_and(s.shr(meta1, s.uint(16)), s.uint(0xFF))), b1, s.fmul(s.float_from_uint(s.bit_and(s.shr(meta2, s.uint(16)), s.uint(0xFF))), b2))), inv255)
    var wc = s.fmul(s.fma(s.float_from_uint(s.shr(meta0, s.uint(24))), b0, s.fma(s.float_from_uint(s.shr(meta1, s.uint(24))), b1, s.fmul(s.float_from_uint(s.shr(meta2, s.uint(24))), b2))), inv255)
    var wg = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fadd(ws, s.fadd(wr, wc))))
    var s0_tx = s.sample_bindless_lod(bindless_arr, t_alb, hit_uv_v2, hit_lod)
    var s1_tx = s.sample_bindless_lod(bindless_arr, s.iadd(t_alb, s.uint(2)), hit_uv_v2, hit_lod)
    var s2_tx = s.sample_bindless_lod(bindless_arr, s.iadd(t_alb, s.uint(4)), hit_uv_v2, hit_lod)
    var s3_tx = s.sample_bindless_lod(bindless_arr, s.iadd(t_alb, s.uint(6)), hit_uv_v2, hit_lod)
    var w_sum_sp = s.nmax(s.float(1e-4), s.fadd(s.fadd(wg, ws), s.fadd(wr, wc)))
    var inv_wsp = s.fdiv(s.float(1.0), w_sum_sp)
    var sp_r = s.fmul(s.fma(s.extract_f(s0_tx, 0), wg, s.fma(s.extract_f(s1_tx, 0), ws, s.fma(s.extract_f(s2_tx, 0), wr, s.fmul(s.extract_f(s3_tx, 0), wc)))), inv_wsp)
    var sp_g = s.fmul(s.fma(s.extract_f(s0_tx, 1), wg, s.fma(s.extract_f(s1_tx, 1), ws, s.fma(s.extract_f(s2_tx, 1), wr, s.fmul(s.extract_f(s3_tx, 1), wc)))), inv_wsp)
    var sp_b = s.fmul(s.fma(s.extract_f(s0_tx, 2), wg, s.fma(s.extract_f(s1_tx, 2), ws, s.fma(s.extract_f(s2_tx, 2), wr, s.fmul(s.extract_f(s3_tx, 2), wc)))), inv_wsp)
    s.store(alb_r_v, s.fmul(s.load_f(alb_r_v), sp_r))
    s.store(alb_g_v, s.fmul(s.load_f(alb_g_v), sp_g))
    s.store(alb_b_v, s.fmul(s.load_f(alb_b_v), sp_b))
    s.begin_else()
    # 24-bit sRGB Vertex Color interpolation (`vertex_words[11][8..31]`) multiplied into `alb_rgb` & `em_rgb`!
    var vc0 = emit_decode_vertex_color_rgb_spv(s, meta0, is_furnace)
    var vc1 = emit_decode_vertex_color_rgb_spv(s, meta1, is_furnace)
    var vc2 = emit_decode_vertex_color_rgb_spv(s, meta2, is_furnace)
    var vc_r = s.fma(vc0[0], b0, s.fma(vc1[0], b1, s.fmul(vc2[0], b2)))
    var vc_g = s.fma(vc0[1], b0, s.fma(vc1[1], b1, s.fmul(vc2[1], b2)))
    var vc_b = s.fma(vc0[2], b0, s.fma(vc1[2], b1, s.fmul(vc2[2], b2)))
    s.store(alb_r_v, s.fmul(s.load_f(alb_r_v), vc_r))
    s.store(alb_g_v, s.fmul(s.load_f(alb_g_v), vc_g))
    s.store(alb_b_v, s.fmul(s.load_f(alb_b_v), vc_b))
    s.store(em_r_v, s.fmul(s.load_f(em_r_v), vc_r))
    s.store(em_g_v, s.fmul(s.load_f(em_g_v), vc_g))
    s.store(em_b_v, s.fmul(s.load_f(em_b_v), vc_b))

    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var s_alb = s.sample_bindless_lod(bindless_arr, t_alb, hit_uv_v2, hit_lod)
    var tx_r = s.extract_f(s_alb, 0)
    var tx_g = s.extract_f(s_alb, 1)
    var tx_b = s.extract_f(s_alb, 2)
    s.store(alb_r_v, s.fmul(s.load_f(alb_r_v), tx_r))
    s.store(alb_g_v, s.fmul(s.load_f(alb_g_v), tx_g))
    s.store(alb_b_v, s.fmul(s.load_f(alb_b_v), tx_b))
    s.store(em_r_v, s.fmul(s.load_f(em_r_v), tx_r))
    s.store(em_g_v, s.fmul(s.load_f(em_g_v), tx_g))
    s.store(em_b_v, s.fmul(s.load_f(em_b_v), tx_b))
    s.end_if()
    s.end_if()

    s.begin_if(s.ult(t_orm, s.uint(0xFF)))
    var s_orm = s.sample_bindless_lod(bindless_arr, t_orm, hit_uv_v2, hit_lod)
    s.store(hit_ao_v, s.extract_f(s_orm, 0))
    s.store(hit_rough_v, s.nclamp(s.fmul(s.load_f(hit_rough_v), s.extract_f(s_orm, 1)), s.float(0.04), s.float(1.0)))
    s.store(hit_metal_v, s.nclamp(s.fmul(s.load_f(hit_metal_v), s.extract_f(s_orm, 2)), s.float(0.0), s.float(1.0)))
    s.end_if()

    # Lit 4-bay office suites visible inside puddle & car-hood skyscraper reflections (exact PRNG match with primary!)
    s.begin_if(s.logical_and(s.logical_and(has_int_para, s.logical_not(is_furnace)), s.flt(s.load_f(hit_metal_v), s.float(0.45))))
    var wsuite = emit_window_suite_prng_spv(s, inst_id, hit_u, hit_v)
    s.begin_if(s.flt(wsuite[0], s.float(0.70)))
    var warm_t = wsuite[1]
    var kr = s.fma(s.float(-0.22), warm_t, s.float(1.02))
    var kg = s.fma(s.float(0.14), warm_t, s.float(0.78))
    var kb = s.fma(s.float(0.52), warm_t, s.float(0.48))
    var room_int_r = s.fma(wsuite[3], s.float(1.4), s.float(1.1))
    var room_str = s.fvar(room_int_r)
    s.begin_if(s.ult(t_det, s.uint(AAA_MAT_DET_NONE)))
    var acol = s.float_from_uint(s.bit_and(wsuite[4], s.uint(3)))
    var arow = s.float_from_uint(s.bit_and(s.shr(wsuite[4], s.uint(2)), s.uint(3)))
    var au = s.fmul(s.fadd(acol, s.nclamp(s.fract(hit_u), s.float(0.05), s.float(0.95))), s.float(0.25))
    var av = s.fmul(s.fadd(arow, s.nclamp(s.fract(hit_v), s.float(0.05), s.float(0.95))), s.float(0.25))
    var s_rm = s.sample_bindless_lod(bindless_arr, t_det, s.vec2(au, av), s.float(2.0))
    var r_sc = s.fmul(room_int_r, s.float(1.35))
    s.store(em_r_v, s.fma(s.extract_f(s_rm, 0), s.fmul(kr, r_sc), s.load_f(em_r_v)))
    s.store(em_g_v, s.fma(s.extract_f(s_rm, 1), s.fmul(kg, r_sc), s.load_f(em_g_v)))
    s.store(em_b_v, s.fma(s.extract_f(s_rm, 2), s.fmul(kb, r_sc), s.load_f(em_b_v)))
    s.store(room_str, s.float(0.0))
    s.end_if()
    var rs = s.load_f(room_str)
    s.store(em_r_v, s.fma(kr, rs, s.load_f(em_r_v)))
    s.store(em_g_v, s.fma(kg, rs, s.load_f(em_g_v)))
    s.store(em_b_v, s.fma(kb, rs, s.load_f(em_b_v)))
    s.end_if()
    s.end_if()

    var alb_r = s.load_f(alb_r_v)
    var alb_g = s.load_f(alb_g_v)
    var alb_b = s.load_f(alb_b_v)
    var hit_rough = s.load_f(hit_rough_v)
    var hit_metal = s.load_f(hit_metal_v)
    var hit_ao = s.load_f(hit_ao_v)
    var em_r = s.load_f(em_r_v)
    var em_g = s.load_f(em_g_v)
    var em_b = s.load_f(em_b_v)

    var one_m_met = s.fsub(s.float(1.0), hit_metal)
    var f0_r = s.fma(s.float(0.04), one_m_met, s.fmul(alb_r, hit_metal))
    var f0_g = s.fma(s.float(0.04), one_m_met, s.fmul(alb_g, hit_metal))
    var f0_b = s.fma(s.float(0.04), one_m_met, s.fmul(alb_b, hit_metal))
    var kd_r = s.fmul(alb_r, s.fmul(one_m_met, s.fsub(s.float(1.0), f0_r)))
    var kd_g = s.fmul(alb_g, s.fmul(one_m_met, s.fsub(s.float(1.0), f0_g)))
    var kd_b = s.fmul(alb_b, s.fmul(one_m_met, s.fsub(s.float(1.0), f0_b)))

    s.begin_if(is_furnace)
    s.store(out_r, s.fadd(s.fadd(kd_r, f0_r), em_r))
    s.store(out_g, s.fadd(s.fadd(kd_g, f0_g), em_g))
    s.store(out_b, s.fadd(s.fadd(kd_b, f0_b), em_b))
    s.begin_else()
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var max_env_mip = s.load_float(b_globals, s.uint(103))

    # 1. Direct Sun at `P_hit` (fast 1-tap VSM lookup or inline shadow ray)
    var ndotl_s = s.nmax(s.float(0.0), emit_vec3_dot(s, hnx, hny, hnz, sun_x, sun_y, sun_z))
    var sun_vis_h = s.fvar(s.float(1.0))
    s.begin_if(s.fgt(ndotl_s, s.float(0.001)))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(1)), s.uint(0)))
    var lclip = emit_mat4_mul_point(
        s, b_globals, s.uint(56),
        s.fma(hnx, s.float(0.02), hx),
        s.fma(hny, s.float(0.02), hy),
        s.fma(hnz, s.float(0.02), hz),
    )
    var lu = s.fma(lclip[0], s.float(0.5), s.float(0.5))
    var lv = s.fma(s.fneg(lclip[1]), s.float(0.5), s.float(0.5))
    var in_vsm = s.logical_and(
        s.logical_and(s.fgt(lu, s.float(0.002)), s.flt(lu, s.float(0.998))),
        s.logical_and(s.fgt(lv, s.float(0.002)), s.flt(lv, s.float(0.998))),
    )
    s.begin_if(s.logical_and(in_vsm, s.flt(lclip[2], s.float(0.999))))
    var vz_s = emit_sample_vsm_depth(s, b_vsm, vsm_samp, lu, lv)
    s.store(sun_vis_h, s.select_f(s.flt(vz_s, s.fsub(lclip[2], s.float(0.002))), s.float(0.0), s.float(1.0)))
    s.end_if()
    if trace_hit_sun_shadow:
        s.begin_else()
        s.begin_if(s.ine(s.bit_and(r_flags, s.uint(2)), s.uint(0)))
        var rts = emit_rt_shadow_query(
            s, tlas_var, hx, hy, hz, hnx, hny, hnz, sun_x, sun_y, sun_z, s.float(300.0),
            b_verts=b_verts, b_indices=b_indices, b_instances=b_instances, b_materials=b_materials, bindless_arr=bindless_arr,
        )
        s.store(sun_vis_h, rts)
        s.end_if()
    s.end_if()
    s.end_if()

    var d_sun_scale = s.fmul(s.fmul(ndotl_s, s.load_f(sun_vis_h)), s.float(0.318309886))
    var sun_ir_r = s.fmul(sun_r, d_sun_scale)
    var sun_ir_g = s.fmul(sun_g, d_sun_scale)
    var sun_ir_b = s.fmul(sun_b, d_sun_scale)

    # 2. `O(1)` Camera-Froxel Clustered Local + Top-2 `LTC_TUBE`/`RECT` Diffuse & Specular Lighting at `P_hit`
    var loc_ir_r = s.fvar(s.float(0.0))
    var loc_ir_g = s.fvar(s.float(0.0))
    var loc_ir_b = s.fvar(s.float(0.0))
    var loc_sp_r = s.fvar(s.float(0.0))
    var loc_sp_g = s.fvar(s.float(0.0))
    var loc_sp_b = s.fvar(s.float(0.0))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(32)), s.uint(0)))
    var fl = emit_froxel_local_lighting_at_hit(
        s, b_clusters, b_globals, hx, hy, hz, hnx, hny, hnz,
        eval_specular=True,
        vx=s.fneg(dx), vy=s.fneg(dy), vz=s.fneg(dz),
        hit_rough=hit_rough,
        f0_r=f0_r, f0_g=f0_g, f0_b=f0_b,
    )
    s.store(loc_ir_r, fl[0]); s.store(loc_ir_g, fl[1]); s.store(loc_ir_b, fl[2])
    s.store(loc_sp_r, fl[3]); s.store(loc_sp_g, fl[4]); s.store(loc_sp_b, fl[5])
    s.end_if()

    # 3. `SH9` Sky Irradiance + Multi-Bounce Excess from `b_surfcache` + Specular Env Bounce
    var sh_h = emit_eval_sh9_irradiance(s, b_globals, 72, hnx, hny, hnz)
    var sh_scale = s.fmul(ibl_int, hit_ao)
    var amb_r = s.fmul(sh_h[0], sh_scale)
    var amb_g = s.fmul(sh_h[1], sh_scale)
    var amb_b = s.fmul(sh_h[2], sh_scale)

    var tot_diff_r = s.fvar(s.fmul(kd_r, s.fadd(sun_ir_r, s.fadd(s.load_f(loc_ir_r), amb_r))))
    var tot_diff_g = s.fvar(s.fmul(kd_g, s.fadd(sun_ir_g, s.fadd(s.load_f(loc_ir_g), amb_g))))
    var tot_diff_b = s.fvar(s.fmul(kd_b, s.fadd(sun_ir_b, s.fadd(s.load_f(loc_ir_b), amb_b))))

    s.begin_if(s.ine(sc_base, s.uint(0xFFFFFF)))
    var sc_w = s.imul(s.iadd(sc_base, prim_id), s.uint(4))
    var scr = s.load_float(b_surfcache, sc_w)
    var scg = s.load_float(b_surfcache, s.iadd(sc_w, s.uint(1)))
    var scb = s.load_float(b_surfcache, s.iadd(sc_w, s.uint(2)))
    # Preserve multi-bounce indirect bleeding while keeping crisp per-pixel BC7 textures on mirrors
    var b_weight = s.select_f(s.ult(t_alb, s.uint(0xFF)), s.float(0.25), s.float(1.0))
    var exc_r = s.nmax(s.float(0.0), s.fsub(s.fsub(scr, em_r), s.load_f(tot_diff_r)))
    var exc_g = s.nmax(s.float(0.0), s.fsub(s.fsub(scg, em_g), s.load_f(tot_diff_g)))
    var exc_b = s.nmax(s.float(0.0), s.fsub(s.fsub(scb, em_b), s.load_f(tot_diff_b)))
    s.store(tot_diff_r, s.fma(exc_r, b_weight, s.load_f(tot_diff_r)))
    s.store(tot_diff_g, s.fma(exc_g, b_weight, s.load_f(tot_diff_g)))
    s.store(tot_diff_b, s.fma(exc_b, b_weight, s.load_f(tot_diff_b)))
    s.end_if()

    # Specular cubemap reflection + Top-2 LTC/local specular highlights on hit surface
    var two_nd = s.fmul(s.float(2.0), emit_vec3_dot(s, hnx, hny, hnz, dx, dy, dz))
    var rhx = s.fsub(dx, s.fmul(hnx, two_nd))
    var rhy = s.fsub(dy, s.fmul(hny, two_nd))
    var rhz = s.fsub(dz, s.fmul(hnz, two_nd))
    var env_h = s.sample_cube_lod(env_cube, s.vec3(rhx, rhy, rhz), s.fmul(hit_rough, max_env_mip))
    var env_sc = s.fmul(ibl_int, hit_ao)
    var sp_h_r = s.fadd(s.fmul(f0_r, s.fmul(s.extract_f(env_h, 0), env_sc)), s.load_f(loc_sp_r))
    var sp_h_g = s.fadd(s.fmul(f0_g, s.fmul(s.extract_f(env_h, 1), env_sc)), s.load_f(loc_sp_g))
    var sp_h_b = s.fadd(s.fmul(f0_b, s.fmul(s.extract_f(env_h, 2), env_sc)), s.load_f(loc_sp_b))

    s.store(out_r, s.fadd(s.load_f(tot_diff_r), s.fadd(sp_h_r, em_r)))
    s.store(out_g, s.fadd(s.load_f(tot_diff_g), s.fadd(sp_h_g, em_g)))
    s.store(out_b, s.fadd(s.load_f(tot_diff_b), s.fadd(sp_h_b, em_b)))
    s.end_if()

    s.store(out_hit, s.uint(1))
    s.store(out_t, t_hit)
    s.end_if()

    return (
        s.load_f(out_r),
        s.load_f(out_g),
        s.load_f(out_b),
        s.load_u(out_hit),
        s.load_f(out_t),
    )
