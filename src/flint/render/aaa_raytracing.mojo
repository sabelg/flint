"""Milestone 3 (`V14`): Hardware `VK_KHR_acceleration_structure` (`BLAS` / `TLAS`) &
`SPV_KHR_ray_query` Contact Shadows, Reflections & Surface-Cache Ray Tracing (`aaa_raytracing.mojo`).
"""
from std.memory import bitcast
from flint.core.transform import Mat4
from flint.platform.vulkan import VulkanAccelStruct, VulkanGpu, put32, put64, put_float
from flint.render.spirv import AaaSpirvModule

# SPIR-V RayFlagsKHR constants
comptime RAY_FLAG_NONE_KHR: UInt32 = 0
comptime RAY_FLAG_OPAQUE_KHR: UInt32 = 1
comptime RAY_FLAG_TERMINATE_ON_FIRST_HIT_KHR: UInt32 = 4
comptime RAY_FLAG_SHADOW_OCCLUSION: UInt32 = 5  # OPAQUE | TERMINATE_ON_FIRST_HIT


def write_vk_as_instance_64b(
    dst_addr: Int,
    world_mat: Mat4,
    surfcache_base_tri: Int,
    blas_device_address: UInt64,
    cull_disable: Bool = True,
):
    """Writes one 64-byte `VkAccelerationStructureInstanceKHR` record with row-major `3x4` transform
    and `instanceCustomIndex = surfcache_base_tri` for `O(1)` Lumen Surface-Cache lookup on ray hit!"""
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
    var mask8 = UInt32(0xFF) << 24
    put32(dst_addr + 48, custom24 | mask8)

    var flags8 = (UInt32(1) if cull_disable else UInt32(0)) << 24
    put32(dst_addr + 52, flags8)
    put64(dst_addr + 56, Int(blas_device_address))


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
) -> UInt32:
    """Traces an inline `SPV_KHR_ray_query` shadow occlusion ray (`1.0` = unoccluded, `0.0` = occluded)."""
    var rq = s.ray_query_var()
    var bias = s.float(0.004)
    var ro = s.vec3(s.fma(nx, bias, ox), s.fma(ny, bias, oy), s.fma(nz, bias, oz))
    var rd = s.vec3(lx, ly, lz)
    s.ray_query_init(
        rq,
        tlas_var,
        s.uint(RAY_FLAG_SHADOW_OCCLUSION),
        s.uint(0xFF),
        ro,
        s.float(0.002),
        rd,
        t_max,
    )
    s.begin_loop()
    var cont = s.ray_query_proceed(rq)
    s.break_if(s.logical_not(cont))
    s.end_loop()

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
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Traces an inline `SPV_KHR_ray_query` closest-hit ray against `TLAS`.
    On hit: fetches `O(1)` Surface-Cache radiance `(Lr, Lg, Lb)` at `(custom_index + prim_index) * 4`.
    On miss: returns `(miss_r, miss_g, miss_b)`.
    Returns `(rad_r, rad_g, rad_b, hit_flag_u32, hit_t_f32)`.
    """
    var rq = s.ray_query_var()
    s.ray_query_init(
        rq,
        tlas_var,
        s.uint(RAY_FLAG_OPAQUE_KHR),
        s.uint(0xFF),
        s.vec3(ox, oy, oz),
        t_min,
        s.vec3(dx, dy, dz),
        t_max,
    )
    s.begin_loop()
    var cont = s.ray_query_proceed(rq)
    s.break_if(s.logical_not(cont))
    s.end_loop()

    var out_r = s.fvar(miss_r)
    var out_g = s.fvar(miss_g)
    var out_b = s.fvar(miss_b)
    var out_hit = s.uvar(s.uint(0))
    var out_t = s.fvar(t_max)

    var hit_kind = s.ray_query_committed_type(rq)
    s.begin_if(s.ine(hit_kind, s.uint(0)))
    var sc_base = s.ray_query_committed_custom_index(rq)
    var prim_id = s.ray_query_committed_prim_index(rq)
    var sc_word = s.imul(s.iadd(sc_base, prim_id), s.uint(4))
    s.store(out_r, s.load_float(b_surfcache, sc_word))
    s.store(out_g, s.load_float(b_surfcache, s.iadd(sc_word, s.uint(1))))
    s.store(out_b, s.load_float(b_surfcache, s.iadd(sc_word, s.uint(2))))
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
