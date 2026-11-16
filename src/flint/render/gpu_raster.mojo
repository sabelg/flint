"""Tile-binned triangle rasterizer running as a compute shader.

The CPU keeps vertex transforms and lighting. Triangles become plane-equation
records, are binned into 32x32 tiles in submission order, and each wave of 32
pixels walks its tile's list with colour and depth held in registers. Keeping
the walk in order preserves the CPU renderer's transparency behaviour.
"""
from std.collections import Dict
from std.ffi import external_call
from std.math import floor
from flint.platform.amdgpu import AmdGpu, GpuBuffer, DOMAIN_GTT, DOMAIN_VRAM, PAGE_EXECUTE, PAGE_READ, PAGE_WRITE, WAIT_TIMEOUT_NS
from flint.core.anchor import A_GPU_HANGS, anchor_add, anchor_ready
from flint.core.log import LOG_GPU, log_error, log_info
from flint.core.memory import MemCharge, MEM_FIRST_GPU, MEM_TAGS, MEM_DOMAIN_VRAM, MEM_DOMAIN_GTT, MEM_GPU_TARGETS, MEM_GPU_FRAMES, MEM_GPU_TERRAIN, MEM_GPU_TEXTURES, MEM_GPU_SHADERS
from flint.platform.os import env_flag, env_value, get_time
from flint.core.profiler import (
    Profiler, profile_ticks, ZONE_NONE, ZONE_GPU_BIN, ZONE_GPU_SUBMIT, ZONE_GPU_WAIT, ZONE_GPU_FRAME,
    ZONE_GPU_RASTER, ZONE_GPU_BLOOM, ZONE_GPU_SMOOTH, ZONE_GPU_BACKDROP, ZONE_GPU_COMPOSE,
)
from flint.platform.amd_isa import ISA_GFX12, CommandBuilder, ShaderAssembler, SRC_EXEC_LO, SRC_FLOAT_HALF, SRC_FLOAT_MINUS_HALF, SRC_FLOAT_ONE, SRC_LITERAL, SRC_TTMP7, SRC_TTMP9, SRC_VCC_LO, INPUT_GROUP_X, INPUT_GROUP_Y, INPUT_THREAD_XY, src_int, src_vgpr
from flint.platform.vulkan import ACCESS_HOST_READ, ACCESS_SHADER_WRITE, NO_IMAGE, PIPELINE_COMPUTE, PIPELINE_HOST, VulkanGpu
from flint.render.occupancy_layout import OccupancyLayout, OCC_BYTES, write_occupancy_header
from flint.render.gpu_layout import (
    WIND_MIRROR_BYTES,
    BIND_BACKDROP, BIND_CELLS, BIND_CONSTANTS, BIND_COUNT, BIND_HUD, BIND_LISTS, BIND_RECORDS,
    BIND_SHADOW_MAP, BIND_SHOWN, BIND_TARGET, BIND_TEXTURES, BIND_TILES, BIND_VOXELS,
    BLOOM_BLURRED_OFFSET, BLOOM_BYTES, BLOOM_FINAL_OFFSET, BLOOM_MIP0_CRISP_OFFSET, BLOOM_MIP1_OFFSET,
    BLOOM_MIP1_UP_OFFSET, BLOOM_MIP2_OFFSET, BLOOM_SAMPLES_X, BLOOM_SAMPLES_Y, BLOOM_SOURCE_OFFSET,
    CLUSTER_WORDS_PER_SLICE, CLUSTER_WORDS_PER_TILE, CLUSTER_Z_SLICES,
    CONSTANT_BLOOM_DISTANCE, CONSTANT_BYTES, CONSTANT_CAMERA, CONSTANT_DITHER, CONSTANT_EFFECTS,
    CONSTANT_KEY, CONSTANT_LIGHTS, CONSTANT_SATURATION, CONSTANT_SPHERES, CONSTANT_WORLD,
    CSM_BYTES, CSM_DEPTH_BYTES, CSM_DEPTH_WORDS, CSM_FRAME_SLICE_BYTES, CSM_ITEMS_REL_BYTES,
    CSM_MAX_ITEMS, CSM_MAX_RECORDS, CSM_RECORD_DWORDS, CSM_RECORDS_REL_BYTES, CSM_RES, CSM_TILE_SIZE,
    EFFECT_AUTO_EXPOSURE, EFFECT_BILINEAR, EFFECT_FROXEL_VOL, EFFECT_HIZ_CULL, EFFECT_PHYSICAL_SKY,
    EFFECT_SHADOWS, EFFECT_VOLUMETRIC, DITHER_FLAG, FROXEL_INTEGRATED_OFFSET, FROXEL_SLICE_BYTES,
    FROXEL_VOL_OFFSET, FROXEL_X, FROXEL_Y, GPU_CULL_IN_STRIDE_WORDS, GPU_CULL_MAX_INSTANCES,
    GPU_CULL_REL_BYTES, GPU_PARTICLE_MAX, GPU_PARTICLE_REL_BYTES,
    GPU_PARTICLE_STRIDE_WORDS, GPU_SKIN_MAX_JOINTS, GPU_SKIN_MAX_VERTS, GPU_SKIN_REL_BYTES,
    GROUP_HEIGHT, GROUP_WIDTH, HIZ_BACKDROP_EXTRA_BYTES, HIZ_SLICE_WORDS, MAX_CLUSTER_LIGHTS,
    MAX_SHADOW_SPHERES, MIPMAPPED_FLAG, PBR_FLAG, PER_PIXEL_FLAG,
    RECORD_AMBIENT_OCCLUSION, RECORD_BYTES, RECORD_EMISSIVE, RECORD_FLAGS, RECORD_FLOATS, RECORD_FOG,
    RECORD_INVERSE_Z, RECORD_LIGHT, RECORD_METALLIC, RECORD_NORMAL, RECORD_ROUGHNESS, RECORD_TEXTURE,
    RECORD_TEXTURE_FLOAT_HEIGHT, RECORD_TEXTURE_FLOAT_WIDTH, RECORD_TEXTURE_MAX_X, RECORD_TEXTURE_MAX_Y,
    RECORD_TEXTURE_WIDTH, RECORD_TINT, RECORD_U, RECORD_V, RECORD_WEIGHT_A, RECORD_WEIGHT_B,
    SHADER_AUTO_EXPOSURE, SHADER_BLOOM_REDUCE, SHADER_BLUR_HORIZONTAL, SHADER_BLUR_VERTICAL,
    SHADER_CAPTURE, SHADER_COMPOSE, SHADER_COUNT, SHADER_CSM_CLEAR, SHADER_CSM_RASTER,
    SHADER_FROXEL_INTEGRATE, SHADER_GPU_CULL, SHADER_GPU_PARTICLES, SHADER_GPU_SKIN,
    SHADER_HISTOGRAM, SHADER_HIZ_PYRAMID, SHADER_KAWASE_DOWN, SHADER_KAWASE_UP, SHADER_RASTER,
    SHADER_SKY_LUT, SHADER_SMOOTH, SHADER_TAA_CAPTURE, SHADER_TAA_RESOLVE, SHADER_TSR_UPSCALE,
    SHADOW_COARSE_STEPS, SHADOW_FINE_STEPS, SKY_LUT_H,
    SKY_LUT_W, TEXTURED_FLAG, TILE_ENTRY_DWORDS, TILE_SIZE, TSR_SCRATCH_REL_BYTES,
)
from flint.render import vulkan_shaders
from flint.render.frame_passes import FramePasses
from flint.render.kernel import kernel_compute_state
from flint.render.particle_layout import TARGET_PLANES
from flint.core.profiler import ZONE_GPU_PARTICLES

# Large meshes take two slots per triangle (see draw_native_mesh), so this holds
# about 400,000 triangles of world. The 5 x 2 x 5 world peaks at 120,000.
comptime MAX_TRIANGLES: Int = 1000000
# The last slots are for characters, which draw after the world and must never lose out to it.
comptime CHARACTER_SLOTS: Int = 200000
comptime LIST_BYTES: Int = 96 * 1024 * 1024
comptime TEXTURE_BYTES: Int = 2 * 1024 * 1024 * 1024
comptime CLOCK_PAIR_FRAMES: Int = 600
# Breadcrumbs: before each pass the command processor writes the pass's number at
# MARKER_STARTED in the frame's stamp buffer, and after the pass's barrier at
# MARKER_FINISHED. After a hang they say which pass never finished.
comptime MARKER_STARTED: Int = 512  # dwords into the stamp buffer, past the stamps
comptime MARKER_FINISHED: Int = 513
comptime MARKER_FRAME: Int = 514
comptime MARKER_NONE: UInt32 = 0xFFFFFFFF
comptime PASS_RASTER: UInt32 = 1
comptime PASS_BLOOM: UInt32 = 2
comptime PASS_SMOOTH: UInt32 = 3
comptime PASS_BACKDROP: UInt32 = 4
comptime PASS_COMPOSE: UInt32 = 5
comptime PASS_EXTRA: UInt32 = 6  # 2.7's particle passes (render/frame_passes.mojo)
comptime PASS_HELD: UInt32 = 9  # the fault injection's wait
comptime RECOVERY_WAIT_NS: UInt64 = 250000000


def pass_name(index: UInt32) -> String:
    if index == PASS_RASTER:
        return "raster"
    if index == PASS_BLOOM:
        return "bloom"
    if index == PASS_SMOOTH:
        return "smooth"
    if index == PASS_BACKDROP:
        return "backdrop"
    if index == PASS_COMPOSE:
        return "compose"
    if index == PASS_EXTRA:
        return "particles"
    if index == PASS_HELD:
        return "held (fault injection)"
    if index == 0:
        return "the start of the frame"
    return "none"


def fence_timeout_from_environment() -> UInt64:
    """SD_GPU_FENCE_TIMEOUT_MS, else two seconds."""
    var text = env_value("SD_GPU_FENCE_TIMEOUT_MS")
    if text.byte_length() > 0:
        try:
            var milliseconds = Int(text)
            if milliseconds > 0:
                return UInt64(milliseconds) * 1000000
        except:
            print("SD_GPU_FENCE_TIMEOUT_MS must be a whole number of milliseconds")
    return WAIT_TIMEOUT_NS
comptime VGPR_COUNT: UInt32 = 104
# Shadow rays read one material byte per voxel in y, z, x order. Flat, because a
# table or packed bits cost 10 to 25% of the raster pass (see docs/RESEARCH.md).
comptime VOXEL_BYTES: Int = 128 * 1024 * 1024
# Registers that hold frame-wide values through the whole raster shader.
comptime S_EFFECTS: UInt32 = 104
comptime S_WORLD_MAX_X: UInt32 = 96
comptime S_WORLD_MAX_Y: UInt32 = 97
comptime S_WORLD_MAX_Z: UInt32 = 98
comptime S_SPHERE_COUNT: UInt32 = 99
comptime S_VOXELS: UInt32 = 100
comptime S_LIGHT_COUNT: UInt32 = 102
comptime S_WORLD_ROW: UInt32 = 103
comptime S_TILE_LIGHTS: UInt32 = 105
comptime V_RAY: UInt32 = 47
comptime V_CAMERA: UInt32 = 50
comptime V_TO_KEY: UInt32 = 53
comptime V_KEY_COLOR: UInt32 = 56
comptime V_WORLD: UInt32 = 59
comptime V_NORMAL: UInt32 = 62
comptime V_OCCLUSION: UInt32 = 65
comptime V_LIGHT: UInt32 = 66
comptime V_SHADOW: UInt32 = 69
# Package 3.2: the dither disc (centre x, y, scale, offset, near inverse depth),
# a never-dither value, and three scratch registers.
comptime V_DITHER: UInt32 = 87
comptime V_DITHER_OFF: UInt32 = 92
comptime V_DITHER_SCRATCH: UInt32 = 93
comptime V_SPEC: UInt32 = 96
# The screen holds one frame, the compositor may hold the next, and the CPU builds a third.
comptime FRAME_SETS: Int = 3
comptime SIDE_RING: Int = 1  # amdgpu scheduler entity for work beside the frames (GPU meshing)
comptime SHADER_BLOCK_DWORDS: Int = 4096
comptime QUERIES_PER_FRAME: Int = 16
comptime PUSH_BYTES: Int = 64

# The ways to draw a frame, and what SD_BACKEND calls them.
comptime BACKEND_CPU: Int = 0
comptime BACKEND_AMDGPU: Int = 1
comptime BACKEND_VULKAN: Int = 2
comptime BACKEND_AUTOMATIC: Int = 3


def requested_backend() -> Int:
    """SD_BACKEND=amdgpu|vulkan|cpu picks one and sticks to it. Unset means the direct
    backend, then Vulkan for GPUs it does not know, then the CPU. SD_CPU=1 still works."""
    return backend_from(env_value("SD_BACKEND"), env_flag("SD_CPU"))


def backend_from(name: String, cpu_only: Bool) -> Int:
    if name == "amdgpu":
        return BACKEND_AMDGPU
    if name == "vulkan":
        return BACKEND_VULKAN
    if name == "cpu" or cpu_only:
        return BACKEND_CPU
    if name.byte_length() > 0:
        print("SD_BACKEND must be amdgpu, vulkan or cpu, not " + name)
    return BACKEND_AUTOMATIC


def backend_name(backend: Int) -> String:
    return "vulkan" if backend == BACKEND_VULKAN else ("amdgpu" if backend == BACKEND_AMDGPU else "cpu")


def record_sgpr(dword: Int) -> UInt32:
    """Records load into s32 and upward, one register per dword."""
    return UInt32(32 + dword)


def evaluate_plane(mut s: ShaderAssembler, vdst: UInt32, dword: Int):
    s.v_fma_f32(vdst, record_sgpr(dword), src_vgpr(3), record_sgpr(dword + 2))
    s.v_fma_f32(vdst, record_sgpr(dword + 1), src_vgpr(4), src_vgpr(vdst))


def float_bits(value: Float32) -> UInt32:
    var copy = value
    return Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]


def divide_by_255(mut s: ShaderAssembler, register: UInt32):
    """Exact floor(n / 255) for n below 65535, matching the CPU's integer division."""
    s.v_lshrrev_b32(34, src_int(8), register)
    s.v_add_nc_u32(register, src_vgpr(34), register)
    s.v_add_nc_u32(register, src_int(1), register)
    s.v_lshrrev_b32(register, src_int(8), register)


def clamp_byte(mut s: ShaderAssembler, register: UInt32):
    s.v_max_i32(register, src_int(0), register)
    s.v_min_i32(register, src_vgpr(38), register)


def lerp_packed(mut s: ShaderAssembler, a: UInt32, b: UInt32, weight: UInt32):
    """a = a * (256 - weight) / 256 + b * weight / 256 on all four bytes at once.

    weight is a VGPR from 0 to 255. Clobbers v82-v85, and v86 must hold 0x00FF00FF.
    """
    s.v_sub_nc_u32(82, SRC_LITERAL, weight)
    s.literal(256)
    s.v_and_b32(83, src_vgpr(a), 86)
    s.v_and_b32(84, src_vgpr(b), 86)
    s.v_mul_u32_u24(83, src_vgpr(82), 83)
    s.v_mul_u32_u24(84, src_vgpr(weight), 84)
    s.v_add_nc_u32(83, src_vgpr(84), 83)
    s.v_lshrrev_b32(83, src_int(8), 83)
    s.v_and_b32(83, src_vgpr(83), 86)
    s.v_lshrrev_b32(a, src_int(8), a)
    s.v_lshrrev_b32(b, src_int(8), b)
    s.v_and_b32(a, src_vgpr(a), 86)
    s.v_and_b32(84, src_vgpr(b), 86)
    s.v_mul_u32_u24(a, src_vgpr(82), a)
    s.v_mul_u32_u24(84, src_vgpr(weight), 84)
    s.v_add_nc_u32(a, src_vgpr(84), a)
    s.v_lshrrev_b32(a, src_int(8), a)
    s.v_and_b32(a, src_vgpr(a), 86)
    s.v_lshlrev_b32(a, src_int(8), a)
    s.v_or_b32(a, src_vgpr(83), a)


def texel_address(mut s: ShaderAssembler, vdst: UInt32, x: UInt32, y: UInt32):
    s.v_mad_u32_u24(vdst, src_vgpr(y), record_sgpr(RECORD_TEXTURE_WIDTH), src_vgpr(x))
    s.v_lshl_add_u32(vdst, src_vgpr(vdst), src_int(2), record_sgpr(RECORD_TEXTURE))


def sample_bilinear(mut s: ShaderAssembler):
    """v13 and v14 hold wrapped texel coordinates. Leaves the filtered texel in v15."""
    s.v_add_f32(13, record_sgpr(RECORD_TEXTURE_FLOAT_WIDTH), 13)
    s.v_add_f32(13, SRC_FLOAT_MINUS_HALF, 13)
    s.v_add_f32(14, record_sgpr(RECORD_TEXTURE_FLOAT_HEIGHT), 14)
    s.v_add_f32(14, SRC_FLOAT_MINUS_HALF, 14)
    s.v_cvt_u32_f32(70, src_vgpr(13))
    s.v_cvt_u32_f32(71, src_vgpr(14))
    s.v_fract_f32(13, src_vgpr(13))
    s.v_fract_f32(14, src_vgpr(14))
    s.v_mul_f32(13, SRC_LITERAL, 13)
    s.literal(0x43800000)  # 256
    s.v_mul_f32(14, SRC_LITERAL, 14)
    s.literal(0x43800000)
    s.v_cvt_u32_f32(72, src_vgpr(13))
    s.v_cvt_u32_f32(73, src_vgpr(14))
    # Wrapping: an unsigned minimum with (value - size) folds value back into range.
    s.v_mov_b32(74, record_sgpr(RECORD_TEXTURE_WIDTH))
    s.v_mov_b32(75, record_sgpr(RECORD_TEXTURE_MAX_Y))
    s.v_add_nc_u32(75, src_int(1), 75)
    s.v_sub_nc_u32(76, src_vgpr(70), 74)
    s.v_min_u32(70, src_vgpr(76), 70)
    s.v_add_nc_u32(76, src_int(1), 70)
    s.v_sub_nc_u32(77, src_vgpr(76), 74)
    s.v_min_u32(76, src_vgpr(77), 76)
    s.v_sub_nc_u32(77, src_vgpr(71), 75)
    s.v_min_u32(71, src_vgpr(77), 71)
    s.v_add_nc_u32(77, src_int(1), 71)
    s.v_sub_nc_u32(74, src_vgpr(77), 75)
    s.v_min_u32(77, src_vgpr(74), 77)
    texel_address(s, 74, 70, 71)
    s.global_load_b32(78, 74, 8)
    texel_address(s, 74, 76, 71)
    s.global_load_b32(79, 74, 8)
    texel_address(s, 74, 70, 77)
    s.global_load_b32(80, 74, 8)
    texel_address(s, 74, 76, 77)
    s.global_load_b32(81, 74, 8)
    s.s_wait_loadcnt()
    s.v_mov_b32(86, SRC_LITERAL)
    s.literal(0x00FF00FF)
    lerp_packed(s, 78, 79, 72)
    lerp_packed(s, 80, 81, 72)
    lerp_packed(s, 78, 80, 73)
    s.v_mov_b32(15, src_vgpr(78))


def dot3(mut s: ShaderAssembler, vdst: UInt32, a: UInt32, b: UInt32):
    s.v_mul_f32(vdst, src_vgpr(a), b)
    s.v_fma_f32(vdst, src_vgpr(a + 1), src_vgpr(b + 1), src_vgpr(vdst))
    s.v_fma_f32(vdst, src_vgpr(a + 2), src_vgpr(b + 2), src_vgpr(vdst))


def load_lighting_constants(mut s: ShaderAssembler):
    """Once per wave: the view ray through this pixel and the key light, into VGPRs."""
    s.s_mov_b32(16, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_CAMERA))
    s.s_load_b512(88, 14, 16)
    s.s_wait_kmcnt()
    for axis in range(3):
        s.v_mov_b32(V_CAMERA + UInt32(axis), 88 + UInt32(axis))
    s.v_add_f32(59, SRC_FLOAT_HALF, 3)
    s.v_mov_b32(60, 100)
    s.v_sub_f32(59, src_vgpr(59), 60)
    s.v_mul_f32(59, 102, 59)
    s.v_add_f32(60, SRC_FLOAT_HALF, 4)
    s.v_sub_f32(60, 101, 60)
    s.v_mul_f32(60, 102, 60)
    for axis in range(3):
        var ray = V_RAY + UInt32(axis)
        s.v_mov_b32(ray, 97 + UInt32(axis))
        s.v_fma_f32(ray, 91 + UInt32(axis), src_vgpr(59), src_vgpr(ray))
        s.v_fma_f32(ray, 94 + UInt32(axis), src_vgpr(60), src_vgpr(ray))
    s.s_mov_b32(16, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_KEY))
    s.s_load_b256(88, 14, 16)
    s.s_mov_b32(16, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_WORLD))
    s.s_load_b256(96, 14, 16)
    s.s_wait_kmcnt()
    for axis in range(3):
        s.v_mov_b32(V_TO_KEY + UInt32(axis), 88 + UInt32(axis))
        s.v_mov_b32(V_KEY_COLOR + UInt32(axis), 91 + UInt32(axis))


def trace_shadow(mut s: ShaderAssembler):
    """Lanes in the current exec mask lose the key light if a sphere or voxel blocks it.

    Leaves V_SHADOW at 1 or 0. Clobbers v71-v78, s28-s31, and s92-s95, and restores exec.
    """
    var sphere_loop = s.new_label()
    var spheres_done = s.new_label()
    var march_loop = s.new_label()
    var march_done = s.new_label()
    var coarse_step = s.new_label()
    var stepped = s.new_label()
    s.s_mov_b32(30, SRC_EXEC_LO)
    s.v_mov_b32(V_SHADOW, SRC_FLOAT_ONE)
    s.v_mov_b32(77, src_int(0))
    s.s_mov_b32(28, src_int(0))
    s.s_mov_b32(29, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_SPHERES))
    s.bind(sphere_loop)
    s.s_cmp_ge_u32(28, S_SPHERE_COUNT)
    s.s_cbranch_scc1(spheres_done)
    s.s_load_b128(92, 14, 29)
    s.s_wait_kmcnt()
    for axis in range(3):
        s.v_sub_f32(71 + UInt32(axis), 92 + UInt32(axis), V_WORLD + UInt32(axis))
    dot3(s, 74, 71, V_TO_KEY)
    dot3(s, 75, 71, 71)
    s.v_mul_f32(76, src_vgpr(74), 74)
    s.v_sub_f32(76, src_vgpr(76), 75)
    s.v_add_f32(76, 95, 76)  # disc = b*b - (oc.oc - r*r)
    # Soft-penumbra analytic sphere shadow: 1.0 - clamp(disc / (0.35 * r2 + 1e-4), 0, 1) * 0.88
    s.v_mul_f32(78, SRC_LITERAL, 95)
    s.literal(0x3EB33333)  # 0.35
    s.v_add_f32(78, SRC_LITERAL, 78)
    s.literal(0x38D1B717)  # 0.0001
    s.v_rcp_f32(78, src_vgpr(78))
    s.s_nop(4)
    s.v_mul_f32(78, src_vgpr(76), 78)
    s.v_max_num_f32(78, src_int(0), 78)
    s.v_min_num_f32(78, SRC_FLOAT_ONE, 78)
    s.v_mul_f32(78, SRC_LITERAL, 78)
    s.literal(0x3F6147AE)  # 0.88
    s.v_sub_f32(78, SRC_FLOAT_ONE, 78)
    s.v_min_num_f32(78, src_vgpr(V_SHADOW), 78)
    s.v_cmp_lt_f32(src_int(0), 74)
    s.s_mov_b32(31, SRC_VCC_LO)
    s.v_cmp_lt_f32(src_int(0), 76)
    s.s_and_b32(SRC_VCC_LO, SRC_VCC_LO, 31)
    s.v_cndmask_b32(V_SHADOW, src_vgpr(V_SHADOW), 78)
    s.s_add_co_u32(28, 28, src_int(1))
    s.s_add_co_u32(29, 29, src_int(16))
    s.s_branch(sphere_loop)
    s.bind(spheres_done)
    s.s_cmp_eq_u32(S_WORLD_ROW, src_int(0))
    s.s_cbranch_scc1(march_done)
    # Start just off the surface, then walk toward the light half a voxel at a time.
    for axis in range(3):
        var offset = UInt32(axis)
        s.v_mul_f32(71 + offset, SRC_LITERAL, V_NORMAL + offset)
        s.literal(0x3F000000)  # 0.5
        s.v_add_f32(71 + offset, src_vgpr(V_WORLD + offset), 71 + offset)
    s.s_mov_b32(28, src_int(0))
    s.bind(march_loop)
    # Past the fine steps a voxel-sized stride still finds walls and costs half as much.
    s.s_cmp_ge_u32(28, src_int(UInt32(SHADOW_FINE_STEPS)))
    s.s_cbranch_scc1(coarse_step)
    for axis in range(3):
        var offset = UInt32(axis)
        s.v_fma_f32(71 + offset, src_vgpr(V_TO_KEY + offset), SRC_FLOAT_HALF, src_vgpr(71 + offset))
    s.s_branch(stepped)
    s.bind(coarse_step)
    for axis in range(3):
        s.v_add_f32(71 + UInt32(axis), src_vgpr(V_TO_KEY + UInt32(axis)), 71 + UInt32(axis))
    s.bind(stepped)
    s.v_cvt_u32_f32(74, src_vgpr(71))
    s.v_cvt_u32_f32(75, src_vgpr(72))
    s.v_cvt_u32_f32(76, src_vgpr(73))
    s.v_min_u32(74, S_WORLD_MAX_X, 74)
    s.v_min_u32(75, S_WORLD_MAX_Y, 75)
    s.v_min_u32(76, S_WORLD_MAX_Z, 76)
    s.v_mad_u32_u24(75, src_vgpr(75), S_WORLD_ROW, src_vgpr(76))
    s.v_mad_u32_u24(75, src_vgpr(75), S_WORLD_ROW, src_vgpr(74))
    s.global_load_u8(76, 75, S_VOXELS)
    s.s_wait_loadcnt()
    s.v_cmp_eq_u32(src_vgpr(76), 77)  # vcc: still in the open
    s.v_cndmask_b32(V_SHADOW, src_vgpr(77), V_SHADOW)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(march_done)
    s.s_add_co_u32(28, 28, src_int(1))
    s.s_cmp_ge_u32(28, SRC_LITERAL)
    s.literal(UInt32(SHADOW_FINE_STEPS + SHADOW_COARSE_STEPS))
    s.s_cbranch_scc0(march_loop)
    s.bind(march_done)
    s.s_mov_b32(SRC_EXEC_LO, 30)


def load_dither_constants(mut s: ShaderAssembler):
    """Once per wave: the frame's dither disc into V_DITHER..+4 (package 3.2).
    Uses s88-s95, free again once the lighting constants are in VGPRs."""
    s.s_mov_b32(16, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_DITHER))
    s.s_load_b256(88, 14, 16)
    s.s_wait_kmcnt()
    for index in range(5):
        s.v_mov_b32(V_DITHER + UInt32(index), 88 + UInt32(index))
    s.v_mov_b32(V_DITHER_OFF, SRC_LITERAL)
    s.literal(float_bits(1.0e9))


def occluder_dither(mut s: ShaderAssembler, next_triangle: Int):
    """Drop the lanes of a DITHER_FLAG triangle that render/dither.mojo's
    DitherView.drops would drop: in front of Maroon and below the disc's
    ordered-dither level. v1, v2: pixel x, y; v3, v4: the same as floats;
    v11: the pixel's inverse depth."""
    var done = s.new_label()
    var value = V_DITHER_SCRATCH
    var work = V_DITHER_SCRATCH + 1
    var level = V_DITHER_SCRATCH + 2
    s.s_and_b32(26, record_sgpr(RECORD_FLAGS), SRC_LITERAL)
    s.literal(DITHER_FLAG)
    s.s_cbranch_scc0(done)
    s.v_sub_f32(value, src_vgpr(3), V_DITHER)
    s.v_mul_f32(value, src_vgpr(value), value)
    s.v_sub_f32(work, src_vgpr(4), V_DITHER + 1)
    s.v_fma_f32(value, src_vgpr(work), src_vgpr(work), src_vgpr(value))
    s.v_fma_f32(value, src_vgpr(value), src_vgpr(V_DITHER + 2), src_vgpr(V_DITHER + 3))
    # At or behind Maroon (less the margin): never dithered.
    s.v_cmp_gt_f32(src_vgpr(11), V_DITHER + 4)
    s.v_cndmask_b32(value, src_vgpr(V_DITHER_OFF), value)
    # bayer4: ((x ^ y) & 1) << 3 | (y & 1) << 2 | (x ^ y) & 2 | (y >> 1) & 1, on x & 3, y & 3.
    s.v_xor_b32(work, src_vgpr(1), 2)
    s.v_and_b32(level, src_int(1), work)
    s.v_lshlrev_b32(level, src_int(3), level)
    s.v_and_b32(work, src_int(2), work)
    s.v_or_b32(level, src_vgpr(work), level)
    s.v_and_b32(work, src_int(1), 2)
    s.v_lshlrev_b32(work, src_int(2), work)
    s.v_or_b32(level, src_vgpr(work), level)
    s.v_lshrrev_b32(work, src_int(1), 2)
    s.v_and_b32(work, src_int(1), work)
    s.v_or_b32(level, src_vgpr(work), level)
    s.v_cvt_f32_u32(level, src_vgpr(level))
    # Keep the lanes whose value reaches the level.
    s.v_cmp_ge_f32(src_vgpr(value), level)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(next_triangle)
    s.bind(done)


def per_pixel_light(mut s: ShaderAssembler):
    """Key and local lights per pixel + Cook-Torrance GGX PBR specular & Schlick Fresnel."""
    var done = s.new_label()
    var no_shadows = s.new_label()
    var shadow_done = s.new_label()
    var light_loop = s.new_label()
    var next_light = s.new_label()
    var lights_done = s.new_label()
    var pbr_done = s.new_label()
    for channel in range(3):
        s.v_mov_b32(V_LIGHT + UInt32(channel), src_int(0))
        s.v_mov_b32(V_SPEC + UInt32(channel), src_int(0))
    s.s_and_b32(26, record_sgpr(RECORD_FLAGS), SRC_LITERAL)
    s.literal(PER_PIXEL_FLAG)
    s.s_cbranch_scc0(done)
    for axis in range(3):
        var offset = UInt32(axis)
        s.v_fma_f32(V_WORLD + offset, src_vgpr(V_RAY + offset), src_vgpr(12), src_vgpr(V_CAMERA + offset))
        evaluate_plane(s, V_NORMAL + offset, RECORD_NORMAL + axis * 3)
    evaluate_plane(s, V_OCCLUSION, RECORD_AMBIENT_OCCLUSION)
    dot3(s, 70, V_NORMAL, V_NORMAL)
    s.v_rsq_f32(70, src_vgpr(70))
    s.s_nop(4)
    for axis in range(3):
        s.v_mul_f32(V_NORMAL + UInt32(axis), src_vgpr(70), V_NORMAL + UInt32(axis))

    # Key light: 0.78 + 0.22 * ambient occlusion matches the vertex formula.
    dot3(s, 70, V_NORMAL, V_TO_KEY)
    s.v_max_num_f32(70, src_int(0), 70)
    s.v_mov_b32(71, SRC_LITERAL)
    s.literal(0x3F47AE14)  # 0.78
    s.v_mul_f32(72, SRC_LITERAL, V_OCCLUSION)
    s.literal(0x3E6147AE)  # 0.22
    s.v_add_f32(72, src_vgpr(71), 72)
    s.v_mul_f32(70, src_vgpr(72), 70)
    s.s_and_b32(26, S_EFFECTS, src_int(UInt32(EFFECT_SHADOWS)))
    s.s_cbranch_scc0(no_shadows)
    # Faces turned away from the key light are already dark and need no ray.
    s.s_mov_b32(91, SRC_EXEC_LO)
    s.v_cmp_lt_f32(src_int(0), 70)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(shadow_done)
    trace_shadow(s)
    s.v_mul_f32(70, src_vgpr(V_SHADOW), 70)
    s.bind(shadow_done)
    s.s_mov_b32(SRC_EXEC_LO, 91)
    s.bind(no_shadows)
    for channel in range(3):
        s.v_mul_f32(V_LIGHT + UInt32(channel), src_vgpr(V_KEY_COLOR + UInt32(channel)), 70)

    # Local lights, one per loop, loaded from the frame's constants.
    s.s_mov_b32(28, src_int(0))
    s.s_mov_b32(29, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_LIGHTS))
    s.bind(light_loop)
    s.s_cmp_ge_u32(28, S_LIGHT_COUNT)
    s.s_cbranch_scc1(lights_done)
    # The CPU marked which lights can reach this tile.
    s.s_lshr_b32(26, S_TILE_LIGHTS, 28)
    s.s_and_b32(26, 26, src_int(1))
    s.s_cbranch_scc0(next_light)
    s.s_load_b256(88, 14, 29)
    s.s_wait_kmcnt()
    for axis in range(3):
        s.v_sub_f32(71 + UInt32(axis), 88 + UInt32(axis), V_WORLD + UInt32(axis))
    dot3(s, 74, 71, 71)
    s.v_add_f32(74, SRC_LITERAL, 74)
    s.literal(0x38D1B717)  # 0.0001 keeps a light on the surface finite
    s.v_mul_f32(75, 95, 74)  # distance^2 / radius^2
    s.v_sub_f32(75, SRC_FLOAT_ONE, 75)
    s.v_max_num_f32(75, src_int(0), 75)
    s.v_mul_f32(75, src_vgpr(75), 75)
    s.v_rsq_f32(74, src_vgpr(74))
    s.s_nop(4)
    dot3(s, 76, V_NORMAL, 71)
    s.v_mul_f32(76, src_vgpr(74), 76)
    s.v_max_num_f32(76, src_int(0), 76)
    s.v_mul_f32(76, SRC_LITERAL, 76)
    s.literal(0x3F428F5C)  # 0.76
    s.v_add_f32(76, SRC_LITERAL, 76)
    s.literal(0x3E75C28F)  # 0.24
    s.v_mul_f32(75, src_vgpr(76), 75)
    s.v_mul_f32(75, src_vgpr(V_OCCLUSION), 75)
    for channel in range(3):
        s.v_fma_f32(V_LIGHT + UInt32(channel), 92 + UInt32(channel), src_vgpr(75), src_vgpr(V_LIGHT + UInt32(channel)))
    s.bind(next_light)
    s.s_add_co_u32(28, 28, src_int(1))
    s.s_add_co_u32(29, 29, src_int(32))
    s.s_branch(light_loop)
    s.bind(lights_done)

    # Cook-Torrance GGX specular + Schlick Fresnel + metallic diffuse attenuation (matching Vulkan SPIR-V).
    s.s_and_b32(26, record_sgpr(RECORD_FLAGS), SRC_LITERAL)
    s.literal(PBR_FLAG)
    s.s_cbranch_scc0(pbr_done)
    # View vector V = normalize(-ray) in v71..v73
    dot3(s, 74, V_RAY, V_RAY)
    s.v_rsq_f32(74, src_vgpr(74))
    s.s_nop(4)
    for axis in range(3):
        s.v_sub_f32(71 + UInt32(axis), src_int(0), V_RAY + UInt32(axis))
        s.v_mul_f32(71 + UInt32(axis), src_vgpr(74), 71 + UInt32(axis))
    # Half vector H = normalize(V + to_key) in v74..v76
    for axis in range(3):
        s.v_add_f32(74 + UInt32(axis), src_vgpr(71 + UInt32(axis)), V_TO_KEY + UInt32(axis))
    dot3(s, 77, 74, 74)
    s.v_add_f32(77, SRC_LITERAL, 77)
    s.literal(0x3727C5AC)  # 1e-5
    s.v_rsq_f32(77, src_vgpr(77))
    s.s_nop(4)
    for axis in range(3):
        s.v_mul_f32(74 + UInt32(axis), src_vgpr(77), 74 + UInt32(axis))
    # v77 = ndotv, v78 = ndoth, v79 = vdoth
    dot3(s, 77, V_NORMAL, 71)
    s.v_max_num_f32(77, SRC_LITERAL, 77)
    s.literal(0x3A83126F)  # 0.001
    dot3(s, 78, V_NORMAL, 74)
    s.v_max_num_f32(78, src_int(0), 78)
    dot3(s, 79, 71, 74)
    s.v_max_num_f32(79, src_int(0), 79)
    # Roughness -> a2 in v80, D_ggx * G_approx / 4 * 255 * sun_lit(v70) in v80
    s.v_mov_b32(80, record_sgpr(RECORD_ROUGHNESS))
    s.v_max_num_f32(80, SRC_LITERAL, 80)
    s.literal(0x3D75C28F)  # 0.06
    s.v_min_num_f32(80, SRC_FLOAT_ONE, 80)
    s.v_mul_f32(80, src_vgpr(80), 80)
    s.v_mul_f32(80, src_vgpr(80), 80)  # a2
    s.v_mul_f32(81, src_vgpr(78), 78)  # ndoth^2
    s.v_sub_f32(82, src_vgpr(80), SRC_FLOAT_ONE)
    s.v_fma_f32(81, src_vgpr(81), src_vgpr(82), SRC_FLOAT_ONE)
    s.v_mul_f32(81, src_vgpr(81), 81)
    s.v_mul_f32(81, SRC_LITERAL, 81)
    s.literal(0x40490FDB)  # pi
    s.v_add_f32(81, SRC_LITERAL, 81)
    s.literal(0x3727C5AC)  # 1e-5
    s.v_rcp_f32(81, src_vgpr(81))
    s.s_nop(4)
    s.v_mul_f32(80, src_vgpr(80), 81)  # D_ggx
    s.v_mul_f32(80, SRC_LITERAL, 80)
    s.literal(0x427F0000)  # 63.75 = 255 * 0.25
    s.v_mul_f32(80, src_vgpr(70), 80)  # * sun_lit (NdotL * shadow * AO)
    # Schlick (1 - vdoth)^5 in v79, grazing (1 - ndotv)^3 in v77
    s.v_sub_f32(79, SRC_FLOAT_ONE, 79)
    s.v_mul_f32(81, src_vgpr(79), 79)
    s.v_mul_f32(81, src_vgpr(81), 81)
    s.v_mul_f32(79, src_vgpr(81), 79)  # f_pow5
    s.v_sub_f32(77, SRC_FLOAT_ONE, 77)
    s.v_mul_f32(81, src_vgpr(77), 77)
    s.v_mul_f32(77, src_vgpr(81), 77)  # rim_f^3
    s.v_mov_b32(82, record_sgpr(RECORD_METALLIC))
    s.v_max_num_f32(82, src_int(0), 82)
    s.v_min_num_f32(82, SRC_FLOAT_ONE, 82)
    # Attenuate diffuse light for metals: V_LIGHT *= (1 - metallic * 0.82)
    s.v_mul_f32(83, SRC_LITERAL, 82)
    s.literal(0x3F51EB85)  # 0.82
    s.v_sub_f32(83, SRC_FLOAT_ONE, 83)
    for channel in range(3):
        var ch = UInt32(channel)
        s.v_mul_f32(V_LIGHT + ch, src_vgpr(83), V_LIGHT + ch)
        # F0 = mix(0.04, base/255, metallic)
        s.v_mul_f32(81, SRC_LITERAL, 44 + ch)
        s.literal(0x3B808081)  # 1/255
        s.v_sub_f32(81, src_vgpr(81), SRC_LITERAL)
        s.literal(0x3D23D70A)  # 0.04
        s.v_fma_f32(81, src_vgpr(81), src_vgpr(82), SRC_LITERAL)
        s.literal(0x3D23D70A)  # 0.04 -> F0
        s.v_sub_f32(84, SRC_FLOAT_ONE, 81)
        s.v_fma_f32(84, src_vgpr(84), src_vgpr(79), src_vgpr(81))  # F = F0 + (1 - F0) * f_pow5
        s.v_mul_f32(84, src_vgpr(80), 84)
        s.v_mul_f32(84, src_vgpr(V_KEY_COLOR + ch), 84)
        # Grazing rim sheen
        s.v_mul_f32(85, SRC_LITERAL, 81)
        s.literal(0x41B00000)  # 22.0
        s.v_fma_f32(V_SPEC + ch, src_vgpr(85), src_vgpr(77), src_vgpr(84))
    s.bind(pbr_done)
    s.bind(done)


def raster_shader(isa: Int = ISA_GFX12) -> ShaderAssembler:
    """User SGPRs: s0:1 records, s2:3 lists, s4:5 tiles, s6:7 frame, s8:9 textures,
    s10 row pixels, s11 tile columns, s12 clear colour, s13 width, s14:15 constants.
    v5 holds the running colour and v6 the running inverse depth."""
    var s = ShaderAssembler(isa)
    var loop_top = s.new_label()
    var next_triangle = s.new_label()
    var finished = s.new_label()
    var untextured = s.new_label()
    var shade_ready = s.new_label()
    var no_saturation = s.new_label()
    var nearest = s.new_label()

    # Pixel position from group id and thread id.
    pixel_position(s, 16)
    s.v_cvt_f32_u32(3, src_vgpr(1))
    s.v_cvt_f32_u32(4, src_vgpr(2))

    # Tile table entry: first list byte and triangle count.
    s.s_lshr_b32(16, 16, src_int(2))
    s.s_lshr_b32(18, 18, src_int(3))
    s.s_mul_i32(18, 18, 11)
    s.s_add_co_u32(16, 16, 18)
    s.s_lshl_b32(16, 16, src_int(4))
    s.s_load_b64(20, 4, 16)
    s.s_add_co_u32(16, 16, src_int(8))
    s.s_load_b32(S_TILE_LIGHTS, 4, 16)
    s.s_mov_b32(17, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_SATURATION))
    s.s_load_b32(19, 14, 17)
    s.s_mov_b32(16, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_BLOOM_DISTANCE))
    s.s_load_b32(17, 14, 16)  # byte distance from the colour image to the bloom image
    s.s_mov_b32(16, SRC_LITERAL)
    s.literal(UInt32(CONSTANT_EFFECTS))
    s.s_load_b32(S_EFFECTS, 14, 16)
    s.s_wait_kmcnt()
    load_lighting_constants(s)
    load_dither_constants(s)

    s.v_mov_b32(5, 12)
    s.v_mov_b32(6, SRC_LITERAL)
    s.literal(0x3089705F)  # 1e-9: the CPU clears depth to 1e9.
    s.v_mov_b32(10, SRC_LITERAL)
    s.literal(0xB8D1B717)  # -0.0001: the CPU's edge tolerance.
    s.v_mov_b32(38, SRC_LITERAL)
    s.literal(255)
    s.v_mov_b32(41, src_int(0))
    s.v_mov_b32(42, SRC_LITERAL)
    s.literal(0x3E4CCCCD)  # 0.2: the CPU's 0.16 emissive floor, times the 1.25 gain in v24.
    for channel in range(3):
        s.s_lshr_b32(16, 12, src_int(UInt32(16 - channel * 8)))
        s.s_and_b32(16, 16, SRC_LITERAL)
        s.literal(255)
        s.v_cvt_f32_u32(27 + UInt32(channel), 16)
    s.s_mov_b32(22, SRC_EXEC_LO)
    s.s_mov_b32(23, 20)
    s.s_lshl_b32(24, 21, src_int(2))
    s.s_add_co_u32(24, 23, 24)

    s.bind(loop_top)
    s.s_cmp_ge_u32(23, 24)
    s.s_cbranch_scc1(finished)
    s.s_load_b32(25, 2, 23)
    s.s_add_co_u32(23, 23, src_int(4))
    s.s_wait_kmcnt()
    s.s_load_b512(32, 0, 25)
    s.s_add_co_u32(26, 25, src_int(64))
    s.s_load_b512(48, 0, 26)
    s.s_add_co_u32(26, 25, SRC_LITERAL)
    s.literal(128)
    s.s_load_b512(64, 0, 26)
    s.s_add_co_u32(26, 25, SRC_LITERAL)
    s.literal(192)
    s.s_load_b256(80, 0, 26)
    s.s_wait_kmcnt()

    # Coverage from barycentric weights.
    evaluate_plane(s, 7, RECORD_WEIGHT_A)
    evaluate_plane(s, 8, RECORD_WEIGHT_B)
    s.v_sub_f32(9, SRC_FLOAT_ONE, 7)
    s.v_sub_f32(9, src_vgpr(9), 8)
    s.v_cmp_ge_f32(src_vgpr(7), 10)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.v_cmp_ge_f32(src_vgpr(8), 10)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.v_cmp_ge_f32(src_vgpr(9), 10)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(next_triangle)

    # Nearer means a larger inverse depth.
    evaluate_plane(s, 11, RECORD_INVERSE_Z)
    s.v_cmp_gt_f32(src_vgpr(11), 6)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(next_triangle)
    occluder_dither(s, next_triangle)

    # Texture sample, or opaque white.
    s.v_rcp_f32(12, src_vgpr(11))
    s.s_nop(4)
    s.s_and_b32(26, record_sgpr(RECORD_FLAGS), SRC_LITERAL)
    s.literal(TEXTURED_FLAG)
    s.s_cbranch_scc0(untextured)
    evaluate_plane(s, 13, RECORD_U)
    evaluate_plane(s, 14, RECORD_V)
    s.v_mul_f32(13, src_vgpr(12), 13)
    s.v_mul_f32(14, src_vgpr(12), 14)
    s.v_fract_f32(13, src_vgpr(13))
    s.v_fract_f32(14, src_vgpr(14))
    s.v_mul_f32(13, record_sgpr(RECORD_TEXTURE_FLOAT_WIDTH), 13)
    s.v_mul_f32(14, record_sgpr(RECORD_TEXTURE_FLOAT_HEIGHT), 14)
    s.s_and_b32(26, S_EFFECTS, src_int(UInt32(EFFECT_BILINEAR)))
    s.s_cbranch_scc0(nearest)
    sample_bilinear(s)
    s.s_branch(shade_ready)
    s.bind(nearest)
    s.v_cvt_u32_f32(13, src_vgpr(13))
    s.v_cvt_u32_f32(14, src_vgpr(14))
    s.v_min_u32(13, record_sgpr(RECORD_TEXTURE_MAX_X), 13)
    s.v_min_u32(14, record_sgpr(RECORD_TEXTURE_MAX_Y), 14)
    s.v_mad_u32_u24(14, src_vgpr(14), record_sgpr(RECORD_TEXTURE_WIDTH), src_vgpr(13))
    s.v_lshl_add_u32(14, src_vgpr(14), src_int(2), record_sgpr(RECORD_TEXTURE))
    s.global_load_b32(15, 14, 8)
    s.s_wait_loadcnt()
    s.s_branch(shade_ready)
    s.bind(untextured)
    s.v_mov_b32(15, SRC_LITERAL)
    s.literal(0xFFFFFFFF)
    s.bind(shade_ready)
    for channel in range(3):
        var shift = UInt32(16 - channel * 8)
        s.s_lshr_b32(26, record_sgpr(RECORD_TINT), src_int(shift))
        s.s_and_b32(26, 26, SRC_LITERAL)
        s.literal(255)
        s.v_lshrrev_b32(19, src_int(shift), 15)
        s.v_and_b32(19, src_vgpr(38), 19)
        s.v_mul_u32_u24(19, 26, 19)
        divide_by_255(s, 19)
        s.v_cvt_f32_u32(44 + UInt32(channel), src_vgpr(19))
    per_pixel_light(s)

    # Shared per-pixel inputs: emissive term, fog, and its inverse.
    evaluate_plane(s, 24, RECORD_EMISSIVE)
    s.v_mul_f32(24, SRC_LITERAL, 24)
    s.literal(0x3FA00000)  # 1.25
    evaluate_plane(s, 25, RECORD_FOG)
    s.v_sub_f32(26, SRC_FLOAT_ONE, 25)
    s.v_lshrrev_b32(16, src_int(24), 15)

    # Texture * tint, lighting + PBR specular, fog, then the region's tone table.
    for channel in range(3):
        s.v_mov_b32(22, src_vgpr(44 + UInt32(channel)))
        evaluate_plane(s, 23, RECORD_LIGHT + channel * 3)
        s.v_add_f32(23, src_vgpr(24), 23)
        s.v_add_f32(23, src_vgpr(V_LIGHT + UInt32(channel)), 23)
        s.v_fma_f32(22, src_vgpr(22), src_vgpr(23), src_vgpr(V_SPEC + UInt32(channel)))
        s.v_mul_f32(22, src_vgpr(26), 22)
        s.v_mul_f32(23, src_vgpr(27 + UInt32(channel)), 25)
        s.v_add_f32(22, src_vgpr(23), 22)
        s.v_cvt_i32_f32(22, src_vgpr(22))
        clamp_byte(s, 22)
        s.v_add_nc_u32(22, SRC_LITERAL, 22)
        s.literal(UInt32(channel * 256))
        s.global_load_u8(30 + UInt32(channel), 22, 14)
    s.s_wait_loadcnt()

    # Saturation about perceptual luminance (66, 150, 40) matching Vulkan & CPU PBR.
    s.s_cmp_eq_u32(19, src_int(0))
    s.s_cbranch_scc1(no_saturation)
    s.v_mul_u32_u24(36, SRC_LITERAL, 30)
    s.literal(66)
    s.v_mul_u32_u24(34, SRC_LITERAL, 31)
    s.literal(150)
    s.v_add_nc_u32(36, src_vgpr(34), 36)
    s.v_mul_u32_u24(34, src_int(40), 32)
    s.v_add_nc_u32(36, src_vgpr(34), 36)
    s.v_lshrrev_b32(36, src_int(8), 36)
    s.s_add_co_u32(27, 19, src_int(64))
    for channel in range(3):
        var register = 30 + UInt32(channel)
        s.v_sub_nc_u32(34, src_vgpr(register), 36)
        s.v_mul_i32_i24(34, 27, 34)
        s.v_ashrrev_i32(34, src_int(6), 34)
        s.v_add_nc_u32(register, src_vgpr(36), 34)
        clamp_byte(s, register)
    s.bind(no_saturation)

    # Effective alpha: triangle * tint * texture.
    s.s_and_b32(26, record_sgpr(RECORD_FLAGS), SRC_LITERAL)
    s.literal(255)
    s.s_lshr_b32(27, record_sgpr(RECORD_TINT), src_int(24))
    s.s_mul_i32(26, 26, 27)
    s.s_lshr_b32(27, 26, src_int(8))
    s.s_add_co_u32(26, 26, 27)
    s.s_add_co_u32(26, 26, src_int(1))
    s.s_lshr_b32(26, 26, src_int(8))
    s.v_mul_u32_u24(33, 26, 16)
    divide_by_255(s, 33)
    s.v_cmp_ne_u32(src_int(0), 33)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(next_triangle)

    # Blend over the running colour, keeping fully opaque pixels exact.
    s.v_cvt_f32_u32(39, src_vgpr(33))
    s.v_mul_f32(39, SRC_LITERAL, 39)
    s.literal(0x3B808081)  # 1 / 255
    s.v_sub_f32(40, SRC_FLOAT_ONE, 39)
    s.v_cmp_eq_u32(src_vgpr(33), 38)
    for channel in range(3):
        var register = 30 + UInt32(channel)
        s.v_lshrrev_b32(34, src_int(UInt32(16 - channel * 8)), 5)
        s.v_and_b32(34, src_vgpr(38), 34)
        s.v_cvt_f32_u32(34, src_vgpr(34))
        s.v_cvt_f32_u32(35, src_vgpr(register))
        s.v_mul_f32(35, src_vgpr(39), 35)
        s.v_mul_f32(34, src_vgpr(40), 34)
        s.v_add_f32(35, src_vgpr(34), 35)
        s.v_cvt_i32_f32(35, src_vgpr(35))
        clamp_byte(s, 35)
        s.v_cndmask_b32(register, src_vgpr(35), register)
    s.v_lshlrev_b32(30, src_int(16), 30)
    s.v_lshlrev_b32(31, src_int(8), 31)
    s.v_or_b32(5, src_vgpr(30), 31)
    s.v_or_b32(5, src_vgpr(32), 5)

    # Only mostly opaque pixels write depth, as on the CPU.
    s.v_cmp_lt_u32(SRC_LITERAL, 33)
    s.literal(127)
    s.v_cndmask_b32(6, src_vgpr(6), 11)

    # Bloom source: base colour * emissive * alpha for sufficiently bright pixels.
    s.v_cmp_ge_f32(src_vgpr(24), 42)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.v_cmp_lt_u32(SRC_LITERAL, 33)
    s.literal(23)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.s_cbranch_execz(next_triangle)
    s.v_mul_f32(43, SRC_LITERAL, 24)
    s.literal(0x3F4CCCCD)  # 0.8 removes the 1.25 gain.
    s.v_mul_f32(43, src_vgpr(39), 43)
    s.v_mov_b32(41, src_int(0))
    for channel in range(3):
        s.v_mul_f32(34, src_vgpr(43), 44 + UInt32(channel))
        s.v_cvt_i32_f32(34, src_vgpr(34))
        clamp_byte(s, 34)
        s.v_lshlrev_b32(34, src_int(UInt32(16 - channel * 8)), 34)
        s.v_or_b32(41, src_vgpr(34), 41)

    s.bind(next_triangle)
    s.s_mov_b32(SRC_EXEC_LO, 22)
    s.s_branch(loop_top)

    s.bind(finished)
    s.v_mov_b32(8, 13)
    s.v_cmp_lt_u32(src_vgpr(1), 8)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    s.v_mad_u32_u24(7, src_vgpr(2), 10, src_vgpr(1))
    s.v_lshlrev_b32(7, src_int(2), 7)
    s.v_or_b32(5, SRC_LITERAL, 5)
    s.literal(0xFF000000)
    s.global_store_b32(7, 5, 6)
    s.v_add_nc_u32(7, 17, 7)
    s.global_store_b32(7, 41, 6)
    # Package 2.7: the inverse depth, two planes on, for the soft particles.
    s.v_add_nc_u32(7, 17, 7)
    s.v_add_nc_u32(7, 17, 7)
    s.global_store_b32(7, 6, 6)
    s.s_endpgm()
    s.resolve()
    return s^


def pixel_position(mut s: ShaderAssembler, user_sgprs: UInt32):
    """v1 and v2 become x and y of a thread in an 8x4 group. Clobbers s16-s19."""
    s.read_thread_position(user_sgprs)
    s.s_lshl_b32(17, 16, src_int(3))
    s.v_add_nc_u32(1, 17, 1)
    if s.isa == ISA_GFX12:
        s.s_and_b32(18, SRC_TTMP7, SRC_LITERAL)
        s.literal(0xFFFF)
    s.s_lshl_b32(19, 18, src_int(2))
    s.v_add_nc_u32(2, 19, 2)


def keep_columns_below(mut s: ShaderAssembler, limit_sgpr: UInt32):
    """Group rounding can spill past the right edge, where a store would wrap a row."""
    s.v_mov_b32(8, limit_sgpr)
    s.v_cmp_lt_u32(src_vgpr(1), 8)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)


def bloom_reduce_shader(isa: Int = ISA_GFX12) -> ShaderAssembler:
    """Max-combine a grid of bright samples into each low-resolution bloom cell.

    User SGPRs: s0:1 bloom source, s2:3 cells, s4 row pixels, s5 cell columns,
    s6 and s7 pixels per cell as floats, s8 and s9 last pixel column and row.
    """
    var s = ShaderAssembler(isa)
    var outer = s.new_label()
    var inner = s.new_label()
    pixel_position(s, 10)
    keep_columns_below(s, 5)
    s.v_cvt_f32_u32(3, src_vgpr(1))
    s.v_cvt_f32_u32(4, src_vgpr(2))
    for channel in range(3):
        s.v_mov_b32(5 + UInt32(channel), src_int(0))
    s.v_mov_b32(38, SRC_LITERAL)
    s.literal(255)
    s.s_mov_b32(20, src_int(0))
    s.bind(outer)
    s.s_mov_b32(21, src_int(0))
    s.bind(inner)
    s.v_cvt_f32_u32(8, 21)
    s.v_add_f32(8, SRC_FLOAT_HALF, 8)
    s.v_mul_f32(8, SRC_LITERAL, 8)
    s.literal(0x3E000000)  # 1 / 8
    s.v_add_f32(8, src_vgpr(3), 8)
    s.v_mul_f32(8, 6, 8)
    s.v_cvt_u32_f32(8, src_vgpr(8))
    s.v_min_u32(8, 8, 8)
    s.v_cvt_f32_u32(9, 20)
    s.v_add_f32(9, SRC_FLOAT_HALF, 9)
    s.v_mul_f32(9, SRC_LITERAL, 9)
    s.literal(0x3E2AAAAB)  # 1 / 6
    s.v_add_f32(9, src_vgpr(4), 9)
    s.v_mul_f32(9, 7, 9)
    s.v_cvt_u32_f32(9, src_vgpr(9))
    s.v_min_u32(9, 9, 9)
    s.v_mad_u32_u24(10, src_vgpr(9), 4, src_vgpr(8))
    s.v_lshlrev_b32(10, src_int(2), 10)
    s.global_load_b32(11, 10, 0)
    s.s_wait_loadcnt()
    for channel in range(3):
        s.v_lshrrev_b32(12, src_int(UInt32(16 - channel * 8)), 11)
        s.v_and_b32(12, src_vgpr(38), 12)
        s.v_max_u32(5 + UInt32(channel), src_vgpr(12), 5 + UInt32(channel))
    s.s_add_co_u32(21, 21, src_int(1))
    s.s_cmp_ge_u32(21, src_int(UInt32(BLOOM_SAMPLES_X)))
    s.s_cbranch_scc0(inner)
    s.s_add_co_u32(20, 20, src_int(1))
    s.s_cmp_ge_u32(20, src_int(UInt32(BLOOM_SAMPLES_Y)))
    s.s_cbranch_scc0(outer)
    s.v_mad_u32_u24(8, src_vgpr(2), 5, src_vgpr(1))
    s.v_mul_u32_u24(8, src_int(12), 8)
    for channel in range(3):
        s.v_cvt_f32_u32(9, src_vgpr(5 + UInt32(channel)))
        s.global_store_b32(8, 9, 2, UInt32(channel * 4))
    s.s_endpgm()
    s.resolve()
    return s^


def bloom_blur_shader(horizontal: Bool, isa: Int = ISA_GFX12) -> ShaderAssembler:
    """One 5-tap [0.0625, 0.25, 0.375, 0.25, 0.0625] Gaussian axis of the bloom blur.

    The vertical pass is last, so it also applies the bloom strength and packs
    each cell into a byte per channel. User SGPRs: s0:1 source, s2:3 target,
    s4 cell columns, s5 last cell column, s6 last cell row, s7 inverse strength.
    """
    var s = ShaderAssembler(isa)
    pixel_position(s, 8)
    keep_columns_below(s, 4)
    s.v_mov_b32(38, SRC_LITERAL)
    s.literal(255)
    s.v_mad_u32_u24(3, src_vgpr(2), 4, src_vgpr(1))
    var moving = UInt32(1) if horizontal else UInt32(2)
    var limit = UInt32(5) if horizontal else UInt32(6)
    s.v_add_nc_u32(4, SRC_LITERAL, moving)
    s.literal(0xFFFFFFFF)
    s.v_max_i32(4, src_int(0), 4)
    s.v_add_nc_u32(5, src_int(1), moving)
    s.v_min_i32(5, limit, 5)
    s.v_add_nc_u32(24, SRC_LITERAL, moving)
    s.literal(0xFFFFFFFE)
    s.v_max_i32(24, src_int(0), 24)
    s.v_add_nc_u32(25, src_int(2), moving)
    s.v_min_i32(25, limit, 25)
    if horizontal:
        s.v_mad_u32_u24(4, src_vgpr(2), 4, src_vgpr(4))
        s.v_mad_u32_u24(5, src_vgpr(2), 4, src_vgpr(5))
        s.v_mad_u32_u24(24, src_vgpr(2), 4, src_vgpr(24))
        s.v_mad_u32_u24(25, src_vgpr(2), 4, src_vgpr(25))
    else:
        s.v_mad_u32_u24(4, src_vgpr(4), 4, src_vgpr(1))
        s.v_mad_u32_u24(5, src_vgpr(5), 4, src_vgpr(1))
        s.v_mad_u32_u24(24, src_vgpr(24), 4, src_vgpr(1))
        s.v_mad_u32_u24(25, src_vgpr(25), 4, src_vgpr(1))
    s.v_mul_u32_u24(6, src_int(12), 3)
    s.v_mul_u32_u24(4, src_int(12), 4)
    s.v_mul_u32_u24(5, src_int(12), 5)
    s.v_mul_u32_u24(24, src_int(12), 24)
    s.v_mul_u32_u24(25, src_int(12), 25)
    for channel in range(3):
        var offset = UInt32(channel * 4)
        var register = 9 + UInt32(channel) * 5
        s.global_load_b32(register, 4, 0, offset)
        s.global_load_b32(register + 1, 6, 0, offset)
        s.global_load_b32(register + 2, 5, 0, offset)
        s.global_load_b32(register + 3, 24, 0, offset)
        s.global_load_b32(register + 4, 25, 0, offset)
    s.s_wait_loadcnt()
    if not horizontal:
        s.v_lshlrev_b32(3, src_int(2), 3)
        s.v_mov_b32(7, SRC_LITERAL)
        s.literal(0xFF000000)
    for channel in range(3):
        var register = 9 + UInt32(channel) * 5
        s.v_add_f32(register, src_vgpr(register + 2), register)
        s.v_mul_f32(register, SRC_LITERAL, register)
        s.literal(0x3E800000)  # 0.25
        s.v_mul_f32(register + 1, SRC_LITERAL, register + 1)
        s.literal(0x3EC00000)  # 0.375
        s.v_add_f32(register, src_vgpr(register + 1), register)
        s.v_add_f32(register + 3, src_vgpr(register + 4), register + 3)
        s.v_fma_f32(register, SRC_LITERAL, src_vgpr(register + 3), src_vgpr(register))
        s.literal(0x3D800000)  # 0.0625
        if horizontal:
            s.global_store_b32(6, register, 2, UInt32(channel * 4))
        else:
            s.v_cvt_i32_f32(register, src_vgpr(register))
            clamp_byte(s, register)
            s.v_cvt_f32_u32(register, src_vgpr(register))
            s.v_mul_f32(register, 7, register)
            s.v_cvt_i32_f32(register, src_vgpr(register))
            s.v_lshlrev_b32(register, src_int(UInt32(16 - channel * 8)), register)
            s.v_or_b32(7, src_vgpr(register), 7)
    if not horizontal:
        s.global_store_b32(3, 7, 2)
    s.s_endpgm()
    return s^


def compose_shader(isa: Int = ISA_GFX12) -> ShaderAssembler:
    """Scene + bilinear-upsampled bloom, then the premultiplied HUD over it, into the shown image.

    User SGPRs: s0:1 scene, s2:3 shown image, s4:5 HUD, s6:7 packed bloom cells,
    s8 row pixels, s9 width, s10 cell columns, s11 and s12 cells per pixel as
    floats, s13 bloom enabled, s14 last cell column, s15 last cell row.
    """
    var s = ShaderAssembler(isa)
    var bloom_ready = s.new_label()
    pixel_position(s, 16)
    keep_columns_below(s, 9)
    s.v_mad_u32_u24(3, src_vgpr(2), 8, src_vgpr(1))
    s.v_lshlrev_b32(3, src_int(2), 3)
    s.global_load_b32(4, 3, 0)
    s.global_load_b32(5, 3, 4)
    s.v_mov_b32(7, src_int(0))
    s.v_mov_b32(38, SRC_LITERAL)
    s.literal(255)
    s.s_cmp_eq_u32(13, src_int(0))
    s.s_cbranch_scc1(bloom_ready)
    # Bilinear u = max(0, (x + 0.5) * scale_x - 0.5), v = max(0, (y + 0.5) * scale_y - 0.5)
    s.v_cvt_f32_u32(6, src_vgpr(1))
    s.v_add_f32(6, SRC_FLOAT_HALF, 6)
    s.v_mul_f32(6, 11, 6)
    s.v_add_f32(6, SRC_FLOAT_MINUS_HALF, 6)
    s.v_max_num_f32(6, src_int(0), 6)
    s.v_cvt_u32_f32(13, src_vgpr(6))
    s.v_min_u32(13, 14, 13)  # x0
    s.v_add_nc_u32(14, src_int(1), 13)
    s.v_min_u32(14, 14, 14)  # x1
    s.v_cvt_f32_u32(15, src_vgpr(13))
    s.v_sub_f32(6, src_vgpr(6), 15)
    s.v_max_num_f32(6, src_int(0), 6)
    s.v_mul_f32(6, SRC_LITERAL, 6)
    s.literal(0x43800000)  # 256.0
    s.v_cvt_u32_f32(15, src_vgpr(6))
    s.v_min_u32(15, SRC_LITERAL, 15)
    s.literal(256)  # fx

    s.v_cvt_f32_u32(8, src_vgpr(2))
    s.v_add_f32(8, SRC_FLOAT_HALF, 8)
    s.v_mul_f32(8, 12, 8)
    s.v_add_f32(8, SRC_FLOAT_MINUS_HALF, 8)
    s.v_max_num_f32(8, src_int(0), 8)
    s.v_cvt_u32_f32(16, src_vgpr(8))
    s.v_min_u32(16, 15, 16)  # y0
    s.v_add_nc_u32(17, src_int(1), 16)
    s.v_min_u32(17, 15, 17)  # y1
    s.v_cvt_f32_u32(18, src_vgpr(16))
    s.v_sub_f32(8, src_vgpr(8), 18)
    s.v_max_num_f32(8, src_int(0), 8)
    s.v_mul_f32(8, SRC_LITERAL, 8)
    s.literal(0x43800000)  # 256.0
    s.v_cvt_u32_f32(18, src_vgpr(8))
    s.v_min_u32(18, SRC_LITERAL, 18)
    s.literal(256)  # fy

    s.v_mad_u32_u24(19, src_vgpr(16), 10, src_vgpr(13))
    s.v_lshlrev_b32(19, src_int(2), 19)
    s.global_load_b32(20, 19, 6)
    s.v_mad_u32_u24(19, src_vgpr(16), 10, src_vgpr(14))
    s.v_lshlrev_b32(19, src_int(2), 19)
    s.global_load_b32(21, 19, 6)
    s.v_mad_u32_u24(19, src_vgpr(17), 10, src_vgpr(13))
    s.v_lshlrev_b32(19, src_int(2), 19)
    s.global_load_b32(22, 19, 6)
    s.v_mad_u32_u24(19, src_vgpr(17), 10, src_vgpr(14))
    s.v_lshlrev_b32(19, src_int(2), 19)
    s.global_load_b32(23, 19, 6)
    s.s_wait_loadcnt()
    s.v_mov_b32(86, SRC_LITERAL)
    s.literal(0x00FF00FF)
    lerp_packed(s, 20, 21, 15)
    lerp_packed(s, 22, 23, 15)
    lerp_packed(s, 20, 22, 18)
    s.v_mov_b32(7, src_vgpr(20))
    s.bind(bloom_ready)
    s.s_wait_loadcnt()
    s.v_lshrrev_b32(12, src_int(24), 5)
    s.v_sub_nc_u32(12, src_vgpr(38), 12)
    s.v_mov_b32(11, SRC_LITERAL)
    s.literal(0xFF000000)
    for channel in range(3):
        var shift = src_int(UInt32(16 - channel * 8))
        s.v_lshrrev_b32(9, shift, 4)
        s.v_and_b32(9, src_vgpr(38), 9)
        s.v_lshrrev_b32(10, shift, 7)
        s.v_and_b32(10, src_vgpr(38), 10)
        s.v_add_nc_u32(9, src_vgpr(10), 9)
        s.v_min_i32(9, src_vgpr(38), 9)
        s.v_mul_u32_u24(9, src_vgpr(12), 9)
        divide_by_255(s, 9)
        s.v_lshrrev_b32(10, shift, 5)
        s.v_and_b32(10, src_vgpr(38), 10)
        s.v_add_nc_u32(9, src_vgpr(10), 9)
        s.v_min_i32(9, src_vgpr(38), 9)
        s.v_lshlrev_b32(9, shift, 9)
        s.v_or_b32(11, src_vgpr(9), 11)
    s.global_store_b32(3, 11, 2)
    s.s_endpgm()
    s.resolve()
    return s^


def absolute_int(mut s: ShaderAssembler, register: UInt32):
    s.v_sub_nc_u32(34, src_int(0), register)
    s.v_max_i32(register, src_vgpr(34), register)


def luma(mut s: ShaderAssembler, vdst: UInt32, color: UInt32):
    """Rec. 709-style luma in 0..255. Clobbers v34."""
    s.v_lshrrev_b32(vdst, src_int(16), color)
    s.v_and_b32(vdst, src_vgpr(38), vdst)
    s.v_mul_u32_u24(vdst, src_int(54), vdst)
    s.v_lshrrev_b32(34, src_int(8), color)
    s.v_and_b32(34, src_vgpr(38), 34)
    s.v_mul_u32_u24(34, SRC_LITERAL, 34)
    s.literal(183)
    s.v_add_nc_u32(vdst, src_vgpr(34), vdst)
    s.v_and_b32(34, src_vgpr(38), color)
    s.v_mul_u32_u24(34, src_int(19), 34)
    s.v_add_nc_u32(vdst, src_vgpr(34), vdst)
    s.v_lshrrev_b32(vdst, src_int(8), vdst)


def smooth_shader(isa: Int = ISA_GFX12) -> ShaderAssembler:
    """Soften high-contrast edges by blending across them, in the manner of FXAA.

    User SGPRs: s0:1 sharp image, s2:3 smoothed image, s4 row pixels, s5 width,
    s6 last pixel column, s7 last pixel row.
    """
    var s = ShaderAssembler(isa)
    pixel_position(s, 8)
    keep_columns_below(s, 5)
    s.v_mov_b32(38, SRC_LITERAL)
    s.literal(255)
    s.v_mov_b32(86, SRC_LITERAL)
    s.literal(0x00FF00FF)
    s.v_mad_u32_u24(3, src_vgpr(2), 4, src_vgpr(1))
    s.v_lshlrev_b32(3, src_int(2), 3)
    s.v_add_nc_u32(9, SRC_LITERAL, 2)
    s.literal(0xFFFFFFFF)
    s.v_max_i32(9, src_int(0), 9)
    s.v_add_nc_u32(10, src_int(1), 2)
    s.v_min_i32(10, 7, 10)
    s.v_add_nc_u32(11, SRC_LITERAL, 1)
    s.literal(0xFFFFFFFF)
    s.v_max_i32(11, src_int(0), 11)
    s.v_add_nc_u32(12, src_int(1), 1)
    s.v_min_i32(12, 6, 12)
    s.v_mad_u32_u24(9, src_vgpr(9), 4, src_vgpr(1))
    s.v_mad_u32_u24(10, src_vgpr(10), 4, src_vgpr(1))
    s.v_mad_u32_u24(11, src_vgpr(2), 4, src_vgpr(11))
    s.v_mad_u32_u24(12, src_vgpr(2), 4, src_vgpr(12))
    for register in range(9, 13):
        s.v_lshlrev_b32(UInt32(register), src_int(2), UInt32(register))
    s.global_load_b32(4, 3, 0)
    s.global_load_b32(5, 9, 0)
    s.global_load_b32(6, 10, 0)
    s.global_load_b32(7, 11, 0)
    s.global_load_b32(8, 12, 0)
    s.s_wait_loadcnt()
    # Every pixel starts as itself; edge pixels overwrite that below.
    s.global_store_b32(3, 4, 2)
    luma(s, 13, 4)
    luma(s, 14, 5)
    luma(s, 15, 6)
    luma(s, 16, 7)
    luma(s, 17, 8)
    s.v_mov_b32(18, src_vgpr(13))
    s.v_mov_b32(19, src_vgpr(13))
    for register in range(14, 18):
        s.v_max_u32(18, src_vgpr(UInt32(register)), 18)
        s.v_min_u32(19, src_vgpr(UInt32(register)), 19)
    s.v_sub_nc_u32(20, src_vgpr(18), 19)  # contrast
    s.v_lshrrev_b32(21, src_int(3), 18)
    s.v_max_u32(21, src_int(16), 21)
    s.v_cmp_lt_u32(src_vgpr(21), 20)
    s.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
    # Across which axis does brightness change fastest?
    s.v_add_nc_u32(22, src_vgpr(14), 15)
    s.v_add_nc_u32(23, src_vgpr(16), 17)
    s.v_lshlrev_b32(24, src_int(1), 13)
    s.v_sub_nc_u32(25, src_vgpr(22), 24)
    s.v_sub_nc_u32(26, src_vgpr(23), 24)
    absolute_int(s, 25)
    absolute_int(s, 26)
    s.v_cmp_lt_u32(src_vgpr(25), 26)  # vcc: blend sideways
    s.v_cndmask_b32(27, src_vgpr(5), 7)
    s.v_cndmask_b32(28, src_vgpr(6), 8)
    s.v_cndmask_b32(29, src_vgpr(14), 16)
    s.v_cndmask_b32(30, src_vgpr(15), 17)
    s.v_sub_nc_u32(31, src_vgpr(29), 13)
    s.v_sub_nc_u32(32, src_vgpr(30), 13)
    absolute_int(s, 31)
    absolute_int(s, 32)
    s.v_cmp_lt_u32(src_vgpr(31), 32)  # vcc: the second neighbour differs more
    s.v_cndmask_b32(27, src_vgpr(27), 28)
    # Stronger blending the more this pixel stands out from its neighbours.
    s.v_add_nc_u32(22, src_vgpr(22), 23)
    s.v_lshrrev_b32(22, src_int(2), 22)
    s.v_sub_nc_u32(22, src_vgpr(22), 13)
    absolute_int(s, 22)
    s.v_cvt_f32_u32(22, src_vgpr(22))
    s.v_cvt_f32_u32(23, src_vgpr(20))
    s.v_rcp_f32(23, src_vgpr(23))
    s.s_nop(4)
    s.v_mul_f32(22, src_vgpr(23), 22)
    s.v_mul_f32(22, SRC_LITERAL, 22)
    s.literal(0x42C00000)  # 96 of 256 at full strength
    s.v_cvt_u32_f32(22, src_vgpr(22))
    s.v_max_u32(22, SRC_LITERAL, 22)
    s.literal(48)
    s.v_min_u32(22, SRC_LITERAL, 22)
    s.literal(128)
    lerp_packed(s, 4, 27, 22)
    s.global_store_b32(3, 4, 2)
    s.s_endpgm()
    return s^


def copy_bytes(destination: Int, source: Int, count: Int):
    _ = external_call["memcpy", Int](destination, source, count)


@fieldwise_init
struct FrameSettings(Copyable, Movable):
    """What `submit` was told about a frame, beyond what its buffers hold."""
    var clear_color: UInt32
    var bloom: Bool
    var bloom_width: Int
    var bloom_height: Int
    var bloom_strength: Int
    var smooth: Bool


struct FrameSet(Copyable, Movable):
    """Everything one frame in flight owns. Scratch images are shared, because the
    compute queue runs a frame's passes to completion before the next frame's."""
    var commands: GpuBuffer
    var records: GpuBuffer
    var lists: GpuBuffer
    var tiles: GpuBuffer
    var constants: GpuBuffer
    var shown: GpuBuffer
    var fence: UInt64
    var window_buffer: Int
    var stamps: GpuBuffer  # 64-bit GPU clock values, one per entry of stamp_zones
    var stamp_zones: List[Int]  # the zone that each stamp ends
    var stamp_frame: Int  # the profiler's frame number at submission
    var descriptor: Int  # Vulkan: this frame's buffers, bound together

    def __init__(out self):
        self.commands = GpuBuffer()
        self.records = GpuBuffer()
        self.lists = GpuBuffer()
        self.tiles = GpuBuffer()
        self.constants = GpuBuffer()
        self.shown = GpuBuffer()
        self.fence = 0
        self.window_buffer = -1
        self.stamps = GpuBuffer()
        self.stamp_zones = List[Int]()
        self.stamp_frame = 0
        self.descriptor = 0


struct GpuRaster(Copyable, Movable):
    var active: Bool
    var backend: Int
    var settings: FrameSettings
    var mirrored_texture_bytes: Int
    var gpu: AmdGpu
    var vk: VulkanGpu
    var validate: Bool  # ask for the Vulkan validation layers
    var sets: List[FrameSet]
    var current: Int
    var last_submitted: Int
    var commands: GpuBuffer
    var code: GpuBuffer
    var records: GpuBuffer
    var lists: GpuBuffer
    var tiles: GpuBuffer
    var constants: GpuBuffer
    var textures: GpuBuffer
    var target: GpuBuffer
    var bloom_cells: GpuBuffer
    var hud: GpuBuffer
    var backdrop: GpuBuffer  # the scene without its HUD, then a HUD image that stays empty
    var backdrop_wanted: Bool
    var backdrop_ready: Bool
    var backdrop_fence: UInt64
    var voxels: GpuBuffer
    var local_domain: UInt64
    var voxel_row: Int
    var voxel_rows: Int
    var voxel_layers: Int
    var voxel_stride: Int
    var voxel_bytes_used: Int
    var voxels_ready: Bool
    var occupancy: GpuBuffer  # the occupancy mip chain (package 1.6, render/occupancy.mojo)
    var occupancy_layout: OccupancyLayout
    var occupancy_bits: List[UInt32]  # the CPU copy of its chunk level
    var occupancy_ready: Bool
    var width: Int
    var height: Int
    var render_width: Int
    var render_height: Int
    var row_pixels: Int
    var columns: Int
    var rows: Int
    var triangle_count: Int
    var dropped_triangles: Int  # this frame
    var dropped_total: Int  # since the renderer started
    var character_pass: Bool
    var bounds_address: Int
    var live_triangles: Int
    var counts: List[UInt32]
    var tile_lights: List[UInt32]
    var cluster_lights: List[UInt32]
    var cursors: List[UInt32]
    var texture_offsets: Dict[Int, Int]
    var texture_used: Int
    var list_entries: Int
    var list_capacity: Int
    var truncated_entries: Int  # since the renderer started
    var bloom_offset: Int
    var bin_seconds: Float64
    var shader_seconds: Float64
    var copy_seconds: Float64
    var frames: Int
    var error: String
    var profiler: Profiler
    var clock_ticks: UInt64  # a CPU tick count and the GPU clock at the same moment
    var clock_gpu: UInt64
    var clock_age: Int
    var ticks_per_gpu_clock: Float64
    var fence_timeout_ns: UInt64
    var hung: Bool  # a fence timed out or a reset was seen; recover() before the next frame
    var hang_count: Int  # hangs this run, across recreations
    var hang_report: String
    var recoveries: Int
    var fault_gate: GpuBuffer  # fault injection: the dword a held frame waits on
    var wind: GpuBuffer        # the wind mirror (package 2.12), CPU-written each frame it changes
    var shadow_map: GpuBuffer  # 2-cascade 1024x1024 shadow map + per-frame CSM records/tiles
    var csm_record_count: Int
    var csm_item_count: Int
    var hiz_tiles_x: Int
    var hiz_tiles_y: Int
    var hiz_valid_mask: UInt32
    var hiz_synced_slot: Int
    var wind_uploads: Int
    var fault_pending: Int
    var fault_generation: UInt32
    var extra: FramePasses  # passes another module adds to the next frame (package 2.7)
    # GPU memory by tag (MEM_GPU_*), then VRAM and GTT. A new GpuRaster (as
    # recover makes) gives the old one's bytes back when the old one dies.
    var charges: List[MemCharge]

    def __init__(out self):
        self.active = False
        self.backend = BACKEND_AMDGPU
        self.settings = FrameSettings(0, False, 1, 1, 1, False)
        self.mirrored_texture_bytes = 0
        self.gpu = AmdGpu()
        self.vk = VulkanGpu()
        self.validate = False
        self.sets = List[FrameSet]()
        self.current = 0
        self.last_submitted = -1
        self.commands = GpuBuffer()
        self.code = GpuBuffer()
        self.records = GpuBuffer()
        self.lists = GpuBuffer()
        self.tiles = GpuBuffer()
        self.constants = GpuBuffer()
        self.textures = GpuBuffer()
        self.target = GpuBuffer()
        self.bloom_cells = GpuBuffer()
        self.hud = GpuBuffer()
        self.backdrop = GpuBuffer()
        self.backdrop_wanted = False
        self.backdrop_ready = False
        self.backdrop_fence = 0
        self.voxels = GpuBuffer()
        self.local_domain = DOMAIN_VRAM
        self.voxel_row = 0
        self.voxel_rows = 0
        self.voxel_layers = 0
        self.voxel_stride = 0
        self.voxel_bytes_used = 0
        self.voxels_ready = False
        self.occupancy = GpuBuffer()
        self.occupancy_layout = OccupancyLayout()
        self.occupancy_bits = List[UInt32]()
        self.occupancy_ready = False
        self.width = 0
        self.height = 0
        self.render_width = 0
        self.render_height = 0
        self.row_pixels = 0
        self.columns = 0
        self.rows = 0
        self.triangle_count = 0
        self.dropped_triangles = 0
        self.dropped_total = 0
        self.character_pass = False
        self.bounds_address = 0
        self.live_triangles = 0
        self.counts = List[UInt32]()
        self.tile_lights = List[UInt32]()
        self.cluster_lights = List[UInt32]()
        self.cursors = List[UInt32]()
        self.texture_offsets = Dict[Int, Int]()
        self.texture_used = 0
        self.list_entries = 0
        self.list_capacity = LIST_BYTES // 4
        self.truncated_entries = 0
        self.bloom_offset = 0
        self.bin_seconds = 0
        self.shader_seconds = 0
        self.copy_seconds = 0
        self.frames = 0
        self.error = ""
        self.profiler = Profiler()
        self.clock_ticks = 0
        self.clock_gpu = 0
        self.clock_age = 0
        self.ticks_per_gpu_clock = 0
        self.fence_timeout_ns = fence_timeout_from_environment()
        self.hung = False
        self.hang_count = 0
        self.hang_report = String()
        self.recoveries = 0
        self.fault_gate = GpuBuffer()
        self.wind = GpuBuffer()
        self.shadow_map = GpuBuffer()
        self.csm_record_count = 0
        self.csm_item_count = 0
        self.hiz_tiles_x = 0
        self.hiz_tiles_y = 0
        self.hiz_valid_mask = 0
        self.hiz_synced_slot = -1
        self.wind_uploads = 0
        self.fault_pending = 0
        self.fault_generation = 0
        self.extra = FramePasses()
        self.charges = List[MemCharge]()
        for tag in range(MEM_FIRST_GPU, MEM_DOMAIN_GTT + 1):
            self.charges.append(MemCharge(tag, 0, 0))

    def allocate(mut self, size: Int, domain: UInt64, flags: UInt32, tag: Int, shared: Bool = False, cpu_reads: Bool = False) -> GpuBuffer:
        """A buffer charged to `tag` (MEM_GPU_*) and to VRAM or GTT."""
        var buffer = self.vk.create_buffer(size, domain == DOMAIN_VRAM, shared, cpu_reads) if self.backend == BACKEND_VULKAN else self.gpu.create_buffer(size, domain, flags)
        if buffer.size > 0 and tag >= MEM_FIRST_GPU and tag < MEM_TAGS:
            buffer.tag = tag
            buffer.vram = domain == DOMAIN_VRAM
            self.charge_buffer(buffer, 1)
        return buffer^

    def upload_wind(mut self, cells: List[Int16]) -> Bool:
        """Copies the wind mirror's cells (physics/wind_mirror.mojo) into the
        GPU's wind buffer (WIND_MIRROR_* in render/gpu_layout.mojo), made on
        first use in GTT, mapped. Presentation only."""
        if not self.active:
            return False
        var size = len(cells) * 2
        if size != WIND_MIRROR_BYTES:
            return False
        if self.wind.size == 0:
            self.wind = self.allocate(WIND_MIRROR_BYTES, DOMAIN_GTT, PAGE_READ, MEM_GPU_FRAMES)
            if self.wind.size == 0 or self.wind.cpu_address == 0:
                return False
        _ = external_call["memcpy", Int](self.wind.cpu_address, Int(cells.unsafe_ptr()), size)
        self.wind_uploads += 1
        return True

    def read_wind(self, index: Int) -> Int16:
        """One int16 of the GPU's wind buffer as the GPU sees it (tests)."""
        if self.wind.size == 0 or index < 0 or index * 2 >= self.wind.size:
            return 0
        var word = self.wind.read32(index // 2)
        return Int16(Int32(word >> 16)) if index % 2 == 1 else Int16(Int32(word & 0xFFFF))

    def charge_buffer(mut self, buffer: GpuBuffer, sign: Int):
        self.charges[buffer.tag - MEM_FIRST_GPU].add(sign * buffer.size, sign)
        self.charges[(MEM_DOMAIN_VRAM if buffer.vram else MEM_DOMAIN_GTT) - MEM_FIRST_GPU].add(sign * buffer.size, sign)

    def release(mut self, buffer: GpuBuffer):
        if buffer.size > 0 and buffer.tag >= MEM_FIRST_GPU:
            self.charge_buffer(buffer, -1)
        if self.backend == BACKEND_VULKAN:
            self.vk.destroy_buffer(buffer)
        else:
            self.gpu.destroy_buffer(buffer)

    def backend_error(self) -> String:
        return self.vk.error if self.backend == BACKEND_VULKAN else self.gpu.error

    def wait_fence(mut self, fence: UInt64, side: Bool = False) -> Bool:
        """Wait for a frame. On amdgpu a fence that does not signal within
        `fence_timeout_ns`, or a context a reset made unusable, is a hang: it is
        reported with the frame's markers and `recover` runs before the next frame.
        `side` waits on the side queue instead (GPU meshing: amdgpu ring 1, the
        Vulkan side queue), whose fences count on their own."""
        if self.backend == BACKEND_VULKAN:
            return self.vk.wait_side(fence) if side else self.vk.wait(fence)
        self.gpu.timed_out = False
        if self.gpu.wait(fence, self.fence_timeout_ns, UInt32(SIDE_RING) if side else UInt32(0)):
            return True
        if self.gpu.timed_out or self.gpu.lost:
            self.note_hang(fence)
        return False

    def note_hang(mut self, fence: UInt64):
        """Say what was running when the GPU stopped answering."""
        var where = String("no frame matches the fence")
        for index in range(len(self.sets)):
            if self.sets[index].fence == fence and self.sets[index].stamps.size > 0:
                var started = self.sets[index].stamps.read32(MARKER_STARTED)
                var finished = self.sets[index].stamps.read32(MARKER_FINISHED)
                var frame = self.sets[index].stamps.read32(MARKER_FRAME)
                where = "frame " + String(frame) + " (set " + String(index) + "): last pass started " + pass_name(started) + ", last finished " + pass_name(finished)
                if started != MARKER_NONE and started != finished:
                    where += "; hung in " + pass_name(started)
                elif started == MARKER_NONE:
                    where += "; the frame never started"
        var state = self.gpu.query_state()
        self.hung = True
        self.hang_count += 1
        self.hang_report = ("GPU hang " + String(self.hang_count) + " on " + backend_name(self.backend) + ": " + self.gpu.error + "; " + where + "; kernel: " + state.describe())
        if anchor_ready():
            _ = anchor_add(A_GPU_HANGS)
        log_error(LOG_GPU, self.hang_report)

    def initialize(mut self, width: Int, height: Int, backend: Int = BACKEND_AMDGPU) -> Bool:
        """Open the GPU and prepare every buffer. False leaves the CPU path in charge."""
        self.backend = backend
        if backend == BACKEND_VULKAN:
            if not self.initialize_vulkan():
                return False
        elif not self.initialize_amdgpu():
            return False
        self.bounds_address = external_call["malloc", Int](MAX_TRIANGLES * 8)
        if not self.allocate_images(width, height):
            return False
        self.active = True
        return True

    def initialize_amdgpu(mut self) -> Bool:
        if not self.gpu.open_device() or not self.gpu.query_compute_alignment() or not self.gpu.create_context():
            self.error = self.gpu.error
            return False
        # Integrated GPUs have a small carved-out VRAM, and all their memory is local anyway.
        self.local_domain = DOMAIN_GTT if self.gpu.integrated() else DOMAIN_VRAM
        self.code = self.allocate(SHADER_BLOCK_DWORDS * 4 * SHADER_COUNT, DOMAIN_GTT, PAGE_READ | PAGE_EXECUTE, MEM_GPU_SHADERS)
        self.textures = self.allocate(TEXTURE_BYTES // 4 if self.gpu.integrated() else TEXTURE_BYTES, self.local_domain, PAGE_READ, MEM_GPU_TEXTURES)
        self.bloom_cells = self.allocate(BLOOM_BYTES, self.local_domain, PAGE_READ | PAGE_WRITE, MEM_GPU_SHADERS)
        self.voxels = self.allocate(VOXEL_BYTES, self.local_domain, PAGE_READ, MEM_GPU_TERRAIN)
        self.occupancy = self.allocate(OCC_BYTES, self.local_domain, PAGE_READ, MEM_GPU_TERRAIN)
        self.occupancy_layout = OccupancyLayout()
        if self.occupancy.size == 0 or self.voxels.size == 0 or self.code.size == 0 or self.textures.size == 0 or self.bloom_cells.size == 0:
            self.error = self.gpu.error
            return False
        var shaders = List[ShaderAssembler]()
        var isa = self.gpu.isa()
        shaders.append(raster_shader(isa))
        shaders.append(bloom_reduce_shader(isa))
        shaders.append(bloom_blur_shader(True, isa))
        shaders.append(bloom_blur_shader(False, isa))
        shaders.append(compose_shader(isa))
        shaders.append(smooth_shader(isa))
        for index in range(len(shaders)):
            if len(shaders[index].words) > SHADER_BLOCK_DWORDS:
                self.error = "shader too large for its block"
                return False
            _ = shaders[index].write_to(self.code, index * SHADER_BLOCK_DWORDS)
        return True

    def initialize_vulkan(mut self) -> Bool:
        self.local_domain = DOMAIN_VRAM
        if not self.vk.initialize(self.validate, self.vulkan_needs_discrete()):
            self.error = self.vk.error
            return False
        # A driver may cap one binding below what this engine would like for textures.
        self.textures = self.allocate(min(TEXTURE_BYTES, self.vk.max_buffer & ~4095), self.local_domain, PAGE_READ, MEM_GPU_TEXTURES)
        self.bloom_cells = self.allocate(BLOOM_BYTES, self.local_domain, PAGE_READ | PAGE_WRITE, MEM_GPU_SHADERS)
        self.voxels = self.allocate(VOXEL_BYTES, self.local_domain, PAGE_READ, MEM_GPU_TERRAIN)
        self.occupancy = self.allocate(OCC_BYTES, self.local_domain, PAGE_READ, MEM_GPU_TERRAIN)
        self.shadow_map = self.allocate(CSM_BYTES, self.local_domain, PAGE_READ | PAGE_WRITE, MEM_GPU_TARGETS)
        self.occupancy_layout = OccupancyLayout()
        if self.occupancy.size == 0 or self.voxels.size == 0 or self.textures.size == 0 or self.bloom_cells.size == 0 or self.shadow_map.size == 0:
            self.error = self.vk.error
            return False
        if not self.vk.create_layouts(BIND_COUNT, PUSH_BYTES) or not self.vk.allocate_command_buffers(FRAME_SETS + 1):
            self.error = self.vk.error
            return False
        if self.vk.timestamps and not self.vk.create_query_pool(FRAME_SETS * QUERIES_PER_FRAME):
            self.error = self.vk.error
            return False
        # In SHADER_ order, with CSM + TAA + V8 Atmosphere/Froxel/Hi-Z/Skinning pipelines.
        var ok = self.vk.create_pipeline(vulkan_shaders.raster_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.bloom_reduce_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.bloom_blur_shader(True))
        ok = ok and self.vk.create_pipeline(vulkan_shaders.bloom_blur_shader(False))
        ok = ok and self.vk.create_pipeline(vulkan_shaders.compose_shader(False))
        ok = ok and self.vk.create_pipeline(vulkan_shaders.smooth_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.compose_shader(True))
        ok = ok and self.vk.create_pipeline(vulkan_shaders.kawase_down_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.kawase_up_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.csm_clear_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.csm_raster_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.taa_compose_shader(False))
        ok = ok and self.vk.create_pipeline(vulkan_shaders.taa_compose_shader(True))
        ok = ok and self.vk.create_pipeline(vulkan_shaders.histogram_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.auto_exposure_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.sky_lut_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.froxel_integrate_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.hiz_pyramid_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.gpu_skin_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.gpu_particles_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.tsr_upscale_shader())
        ok = ok and self.vk.create_pipeline(vulkan_shaders.gpu_cull_shader())
        if not ok:
            self.error = self.vk.error
        return ok

    def vulkan_needs_discrete(self) -> Bool:
        """SD_VULKAN_DISCRETE_ONLY=1 refuses an integrated GPU rather than falling back
        to it, for machines where that GPU is driving the desktop."""
        return self.validate or env_value("SD_VULKAN_DISCRETE_ONLY").byte_length() > 0

    def allocate_images(mut self, width: Int, height: Int) -> Bool:
        """Create everything whose size depends on the render resolution."""
        self.width = width
        self.height = height
        self.render_width = width
        self.render_height = height
        self.row_pixels = ((width * 4 + 255) & ~255) // 4
        self.columns = (width + TILE_SIZE - 1) // TILE_SIZE
        self.rows = (height + TILE_SIZE - 1) // TILE_SIZE
        self.hiz_tiles_x = (width + 15) // 16
        self.hiz_tiles_y = (height + 15) // 16
        self.hiz_valid_mask = 0
        self.hiz_synced_slot = -1
        # Colour, bloom source, smoothed colour, inverse depth and a scratch
        # copy of the colour (the last two for 2.7's particles), all local to the GPU.
        self.bloom_offset = self.row_pixels * (height + GROUP_HEIGHT)
        self.target = self.allocate(self.bloom_offset * 4 * TARGET_PLANES, self.local_domain, PAGE_READ | PAGE_WRITE, MEM_GPU_TARGETS)
        self.hud = self.allocate(self.bloom_offset * 4, self.local_domain, PAGE_READ, MEM_GPU_TARGETS)
        self.backdrop = self.allocate(
            self.bloom_offset * 8 + HIZ_BACKDROP_EXTRA_BYTES,
            DOMAIN_GTT,
            PAGE_READ | PAGE_WRITE,
            MEM_GPU_TARGETS,
            False,
            True,
        )
        if self.target.size == 0 or self.hud.size == 0 or self.backdrop.size == 0:
            self.error = self.backend_error()
            return False
        if self.target.cpu_address != 0:
            _ = external_call["memset", Int](self.target.cpu_address, Int32(0), self.target.size)
        _ = external_call["memset", Int](self.hud.cpu_address, Int32(0), self.hud.size)
        _ = external_call["memset", Int](self.backdrop.cpu_address, Int32(0), self.backdrop.size)
        self.forget_backdrop()
        for _ in range(FRAME_SETS):
            var frame = FrameSet()
            if self.backend != BACKEND_VULKAN:
                frame.commands = self.allocate(16384, DOMAIN_GTT, PAGE_READ | PAGE_EXECUTE, MEM_GPU_FRAMES)
            frame.records = self.allocate(RECORD_BYTES * MAX_TRIANGLES, self.local_domain, PAGE_READ, MEM_GPU_FRAMES)
            frame.lists = self.allocate(LIST_BYTES, self.local_domain, PAGE_READ, MEM_GPU_FRAMES)
            frame.tiles = self.allocate(
                self.columns * self.rows * (TILE_ENTRY_DWORDS + CLUSTER_WORDS_PER_TILE) * 4,
                self.local_domain,
                PAGE_READ,
                MEM_GPU_FRAMES,
            )
            frame.constants = self.allocate(CONSTANT_BYTES, self.local_domain, PAGE_READ, MEM_GPU_FRAMES)
            # The shown image stays in shared memory so the display's GPU can read it.
            frame.shown = self.allocate(self.bloom_offset * 4, DOMAIN_GTT, PAGE_READ | PAGE_WRITE, MEM_GPU_FRAMES, True, True)
            frame.stamps = self.allocate(4096, DOMAIN_GTT, PAGE_READ | PAGE_WRITE, MEM_GPU_FRAMES)
            if frame.stamps.size == 0 or (frame.commands.size == 0 and self.backend != BACKEND_VULKAN) or frame.records.size == 0 or frame.lists.size == 0 or frame.tiles.size == 0 or frame.constants.size == 0 or frame.shown.size == 0:
                self.error = self.backend_error()
                return False
            self.sets.append(frame^)
        if self.backend == BACKEND_VULKAN and not self.bind_frame_sets():
            return False
        self.counts = List[UInt32](length=self.columns * self.rows, fill=0)
        self.tile_lights = List[UInt32](length=self.columns * self.rows, fill=0)
        self.cluster_lights = List[UInt32](length=self.columns * self.rows * CLUSTER_WORDS_PER_TILE, fill=0)
        self.cursors = List[UInt32](length=self.columns * self.rows, fill=0)
        self.current = 0
        self.last_submitted = -1
        return True

    def bind_frame_sets(mut self) -> Bool:
        """Point each frame's descriptor set at its buffers. Buffers change only on resize."""
        var descriptors = self.vk.create_descriptor_sets(FRAME_SETS, BIND_COUNT)
        if len(descriptors) != FRAME_SETS:
            self.error = self.vk.error
            return False
        for index in range(FRAME_SETS):
            var frame = descriptors[index]
            self.sets[index].descriptor = frame
            self.vk.write_descriptor(frame, BIND_RECORDS, self.sets[index].records)
            self.vk.write_descriptor(frame, BIND_LISTS, self.sets[index].lists)
            self.vk.write_descriptor(frame, BIND_TILES, self.sets[index].tiles)
            self.vk.write_descriptor(frame, BIND_CONSTANTS, self.sets[index].constants)
            self.vk.write_descriptor(frame, BIND_TEXTURES, self.textures)
            self.vk.write_descriptor(frame, BIND_TARGET, self.target)
            self.vk.write_descriptor(frame, BIND_VOXELS, self.voxels)
            self.vk.write_descriptor(frame, BIND_CELLS, self.bloom_cells)
            self.vk.write_descriptor(frame, BIND_HUD, self.hud)
            self.vk.write_descriptor(frame, BIND_SHOWN, self.sets[index].shown)
            self.vk.write_descriptor(frame, BIND_BACKDROP, self.backdrop)
            self.vk.write_descriptor(frame, BIND_SHADOW_MAP, self.shadow_map)
        return True

    def resize(mut self, width: Int, height: Int) -> Bool:
        """Change the render resolution between frames."""
        if not self.finish_frames():
            return False
        self.release_images()
        if self.backend == BACKEND_VULKAN and self.vk.swapchain != 0 and not self.vk.create_swapchain(width, height):
            self.error = self.vk.error
            self.active = False
            return False
        if not self.allocate_images(width, height):
            self.active = False
            return False
        return True

    def finish_frames(mut self) -> Bool:
        """Wait for everything in flight, collecting its stamps."""
        for index in range(len(self.sets)):
            if self.sets[index].fence != 0 and not self.wait_fence(self.sets[index].fence):
                self.error = self.backend_error()
                self.active = False
                return False
            self.sets[index].fence = 0
            self.collect_stamps(index)
        return True

    def release_images(mut self):
        self.release(self.target.copy())
        self.release(self.hud.copy())
        self.release(self.backdrop.copy())
        for index in range(len(self.sets)):
            self.release(self.sets[index].commands.copy())
            self.release(self.sets[index].records.copy())
            self.release(self.sets[index].lists.copy())
            self.release(self.sets[index].tiles.copy())
            self.release(self.sets[index].constants.copy())
            self.release(self.sets[index].shown.copy())
            self.release(self.sets[index].stamps.copy())
        self.sets.clear()

    def shutdown(mut self):
        """Close the Vulkan device, so the validation layers can report any leak.
        The amdgpu backend's objects die with the file descriptor."""
        if not self.active or self.backend != BACKEND_VULKAN:
            return
        self.active = False
        _ = self.finish_frames()
        self.release_images()
        self.release(self.textures.copy())
        self.release(self.bloom_cells.copy())
        self.release(self.voxels.copy())
        self.release(self.occupancy.copy())
        if self.shadow_map.size > 0:
            self.release(self.shadow_map.copy())
            self.shadow_map = GpuBuffer()
        self.occupancy = GpuBuffer()
        self.occupancy_ready = False
        if self.wind.size > 0:
            self.release(self.wind.copy())
            self.wind = GpuBuffer()
        self.vk.close()

    def inject_hang(mut self) -> Bool:
        """Fault injection (amdgpu): the next frame waits on the GPU for a dword only
        `recover` writes, so its fence times out exactly as a hung frame's would,
        and nothing on the GPU is actually stuck. Tests and SD_GPU_FAULT use it."""
        if not self.active or self.backend == BACKEND_VULKAN:
            return False
        if self.fault_gate.size == 0:
            self.fault_gate = self.allocate(4096, DOMAIN_GTT, PAGE_READ | PAGE_WRITE, MEM_GPU_SHADERS)
            if self.fault_gate.size == 0:
                return False
        self.fault_generation += 1
        self.fault_gate.write32(0, self.fault_generation - 1)
        self.fault_pending += 1
        return True

    def release_held_frames(self):
        """Let every frame the fault injection is holding finish."""
        if self.fault_gate.size > 0:
            self.fault_gate.write32(0, self.fault_generation)

    def recover(mut self, backend: Int) -> Bool:
        """After a hang: drop the context and every buffer, and build them again on
        `backend` (the same one, or Vulkan once amdgpu has hung too often). What the
        GPU held is uploaded again on use: textures through `texture_offset`, the
        shadow volume by the caller (NativeRuntime.recover_gpu). False leaves the
        CPU renderer in charge."""
        var width = self.width
        var height = self.height
        var profiler = self.profiler.copy()
        var validate = self.validate
        var timeout = self.fence_timeout_ns
        var hangs = self.hang_count
        var report = self.hang_report
        var recoveries = self.recoveries + 1
        var dropped = self.dropped_total
        var truncated = self.truncated_entries
        self.release_held_frames()
        if self.backend == BACKEND_VULKAN:
            self.shutdown()
        else:
            # Give a held or slow frame a moment, so the file closes on an idle queue.
            for index in range(len(self.sets)):
                if self.sets[index].fence != 0:
                    _ = self.gpu.wait(self.sets[index].fence, RECOVERY_WAIT_NS)
            self.gpu.close()
        if self.bounds_address != 0:
            external_call["free", NoneType](self.bounds_address)
        self = GpuRaster()
        self.validate = validate
        self.fence_timeout_ns = timeout
        self.hang_count = hangs
        self.hang_report = report
        self.recoveries = recoveries
        self.dropped_total = dropped
        self.truncated_entries = truncated
        if not self.initialize(width, height, backend):
            log_error(LOG_GPU, "GPU recovery on " + backend_name(backend) + " failed: " + self.error)
            return False
        if profiler.enabled():
            self.set_profiler(profiler)
        log_info(LOG_GPU, "GPU recovered on " + backend_name(backend) + " (recovery " + String(recoveries) + ", " + String(width) + "x" + String(height) + ")")
        return True

    def free_set(self, window_busy: List[Bool]) -> Int:
        """The oldest frame set the compositor is not holding, or -1."""
        for step in range(1, FRAME_SETS + 1):
            var index = (self.current + step) % FRAME_SETS
            if self.sets[index].window_buffer >= 0 and self.sets[index].window_buffer < len(window_busy) and window_busy[self.sets[index].window_buffer]:
                continue
            return index
        return -1

    def begin(mut self, selected: Int) -> Bool:
        """Start building into a frame set, waiting only if the GPU still needs it."""
        self.current = selected
        if self.sets[selected].fence != 0:
            var started = get_time()
            var wait_zone = self.profiler.now()
            if not self.wait_fence(self.sets[selected].fence):
                self.error = self.backend_error()
                self.active = False
                return False
            self.sets[selected].fence = 0
            if (self.hiz_valid_mask & (UInt32(1) << UInt32(selected))) != 0:
                self.hiz_synced_slot = selected
            self.shader_seconds += get_time() - started
            self.profiler.record(ZONE_GPU_WAIT, wait_zone, self.profiler.now())
        self.collect_stamps(selected)
        self.commands = self.sets[selected].commands.copy()
        self.records = self.sets[selected].records.copy()
        self.lists = self.sets[selected].lists.copy()
        self.tiles = self.sets[selected].tiles.copy()
        self.constants = self.sets[selected].constants.copy()
        self.triangle_count = 0
        self.dropped_triangles = 0
        self.character_pass = False
        self.csm_record_count = 0
        self.csm_item_count = 0
        return True

    @staticmethod
    def is_mipmapped(width: Int, height: Int) -> Bool:
        return width >= 16 and height >= 16 and (width & 15) == 0 and (height & 15) == 0

    def texture_offset(mut self, address: Int, width: Int, height: Int) -> Int:
        """Upload a texture the first time it is seen, appending a 5-level mip pyramid on Vulkan."""
        var known = self.texture_offsets.get(address)
        if known:
            return known.value()
        var area0 = width * height
        var has_mips = self.backend == BACKEND_VULKAN and Self.is_mipmapped(width, height)
        var total_words = (
            area0 + (area0 >> 2) + (area0 >> 4) + (area0 >> 6) + (area0 >> 8)
            if has_mips
            else area0
        )
        var size = total_words * 4
        if self.texture_used + size > self.textures.size:
            return -1
        var offset = self.texture_used
        var base_addr = self.textures.cpu_address + offset
        copy_bytes(base_addr, address, area0 * 4)
        if has_mips:
            var src_ptr = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=base_addr)
            var src_off = 0
            var src_w = width
            var src_h = height
            for _ in range(4):
                var dst_off = src_off + src_w * src_h
                var dst_w = src_w >> 1
                var dst_h = src_h >> 1
                for my in range(dst_h):
                    var r0 = src_off + (my * 2) * src_w
                    var r1 = r0 + src_w
                    var drow = dst_off + my * dst_w
                    for mx in range(dst_w):
                        var sx = mx * 2
                        var p00 = src_ptr[unsafe_offset=r0 + sx]
                        var p10 = src_ptr[unsafe_offset=r0 + sx + 1]
                        var p01 = src_ptr[unsafe_offset=r1 + sx]
                        var p11 = src_ptr[unsafe_offset=r1 + sx + 1]
                        var rb = (
                            ((p00 & 0x00FF00FF) + (p10 & 0x00FF00FF) + (p01 & 0x00FF00FF) + (p11 & 0x00FF00FF) + 0x00020002)
                            >> 2
                        ) & 0x00FF00FF
                        var ag = (
                            (((p00 >> 8) & 0x00FF00FF) + ((p10 >> 8) & 0x00FF00FF) + ((p01 >> 8) & 0x00FF00FF) + ((p11 >> 8) & 0x00FF00FF) + 0x00020002)
                            >> 2
                        ) & 0x00FF00FF
                        src_ptr[unsafe_offset=drow + mx] = rb | (ag << 8)
                src_off = dst_off
                src_w = dst_w
                src_h = dst_h
        self.texture_used += size
        self.texture_offsets[address] = offset
        return offset

    def add_csm_triangle(
        mut self,
        cascade: Int,
        ax: Float32,
        ay: Float32,
        az: Float32,
        bx: Float32,
        by: Float32,
        bz: Float32,
        cx: Float32,
        cy: Float32,
        cz: Float32,
    ):
        """Bin one shadow-casting triangle into 16x16 CSM tiles for Cascade 0 or 1."""
        if self.backend != BACKEND_VULKAN or self.shadow_map.cpu_address == 0:
            return
        if self.csm_record_count >= CSM_MAX_RECORDS:
            return
        var area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) <= 0.02:
            return
        var max_z = max(az, max(bz, cz))
        var min_z = min(az, min(bz, cz))
        if max_z <= 0.002 or min_z >= 0.998:
            return
        var min_px = max(0, Int(floor(min(ax, min(bx, cx)))))
        var max_px = min(CSM_RES - 1, Int(floor(max(ax, max(bx, cx)))))
        var min_py = max(0, Int(floor(min(ay, min(by, cy)))))
        var max_py = min(CSM_RES - 1, Int(floor(max(ay, max(by, cy)))))
        if min_px > max_px or min_py > max_py:
            return
        var tx0 = min_px >> 4
        var tx1 = max_px >> 4
        var ty0 = min_py >> 4
        var ty1 = max_py >> 4
        var num_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        if num_tiles > 1024 or self.csm_item_count + num_tiles > CSM_MAX_ITEMS:
            return

        var inv_area = Float32(1.0) / area
        var wa_a = (by - cy) * inv_area
        var wa_b = (cx - bx) * inv_area
        var wa_c = (bx * cy - by * cx) * inv_area
        var wb_a = (cy - ay) * inv_area
        var wb_b = (ax - cx) * inv_area
        var wb_c = (cx * ay - cy * ax) * inv_area
        var dz_a = az - cz
        var dz_b = bz - cz
        var z_a = dz_a * wa_a + dz_b * wb_a
        var z_b = dz_a * wa_b + dz_b * wb_b
        var z_c = cz + dz_a * wa_c + dz_b * wb_c

        var slice_base = self.shadow_map.cpu_address + CSM_DEPTH_BYTES + self.current * CSM_FRAME_SLICE_BYTES
        var rec_idx = self.csm_record_count
        self.csm_record_count += 1
        var rec_ptr = Pointer[Float32, origin=MutAnyOrigin](
            unsafe_from_address=slice_base + CSM_RECORDS_REL_BYTES + rec_idx * CSM_RECORD_DWORDS * 4
        )
        rec_ptr[unsafe_offset=0] = wa_a
        rec_ptr[unsafe_offset=1] = wa_b
        rec_ptr[unsafe_offset=2] = wa_c
        rec_ptr[unsafe_offset=3] = wb_a
        rec_ptr[unsafe_offset=4] = wb_b
        rec_ptr[unsafe_offset=5] = wb_c
        rec_ptr[unsafe_offset=6] = z_a
        rec_ptr[unsafe_offset=7] = z_b
        rec_ptr[unsafe_offset=8] = z_c
        rec_ptr[unsafe_offset=9] = 0.0
        rec_ptr[unsafe_offset=10] = 0.0
        rec_ptr[unsafe_offset=11] = 0.0

        var item_ptr = Pointer[UInt32, origin=MutAnyOrigin](
            unsafe_from_address=slice_base + CSM_ITEMS_REL_BYTES + self.csm_item_count * 8
        )
        var casc_bits = UInt32(cascade & 1) << 16
        var out_i = 0
        for ty in range(ty0, ty1 + 1):
            var ty_bits = casc_bits | (UInt32(ty) << 8)
            for tx in range(tx0, tx1 + 1):
                item_ptr[unsafe_offset=out_i * 2] = UInt32(rec_idx)
                item_ptr[unsafe_offset=out_i * 2 + 1] = ty_bits | UInt32(tx)
                out_i += 1
        self.csm_item_count += num_tiles

    def forget_texture(mut self, address: Int):
        """The memory at address is going away; a later texture may reuse it."""
        try:
            _ = self.texture_offsets.pop(address)
        except:
            pass

    def bounds_entry(self, slot: Int) -> Pointer[UInt16, origin=MutAnyOrigin]:
        return Pointer[UInt16, origin=MutAnyOrigin](unsafe_from_address=self.bounds_address + slot * 8)

    def reserve_slots(mut self, count: Int, slots_per_triangle: Int = 1) -> Int:
        """Claim record slots in draw order. Unfilled slots stay empty and are skipped."""
        var limit = MAX_TRIANGLES if self.character_pass else MAX_TRIANGLES - CHARACTER_SLOTS
        if self.triangle_count + count > limit:
            self.dropped_triangles += count // slots_per_triangle
            self.dropped_total += count // slots_per_triangle
            return -1
        var first = self.triangle_count
        for slot in range(first, first + count):
            var entry = self.bounds_entry(slot)
            entry[unsafe_offset=0] = 1
            entry[unsafe_offset=2] = 0
        self.triangle_count += count
        return first

    def store_bounds(self, slot: Int, minimum_x: Int, minimum_y: Int, maximum_x: Int, maximum_y: Int):
        """Safe from any thread that owns the slot."""
        var entry = self.bounds_entry(slot)
        entry[unsafe_offset=0] = UInt16(minimum_x // TILE_SIZE)
        entry[unsafe_offset=1] = UInt16(minimum_y // TILE_SIZE)
        entry[unsafe_offset=2] = UInt16(maximum_x // TILE_SIZE)
        entry[unsafe_offset=3] = UInt16(maximum_y // TILE_SIZE)

    def record_address(self, slot: Int) -> Int:
        return self.records.cpu_address + slot * RECORD_BYTES

    def bin_triangles(mut self):
        for index in range(len(self.counts)):
            self.counts[index] = 0
        var live = 0
        for triangle in range(self.triangle_count):
            var entry = self.bounds_entry(triangle)
            var left = Int(entry[unsafe_offset=0])
            var right = Int(entry[unsafe_offset=2])
            if left > right:
                continue
            live += 1
            for y in range(Int(entry[unsafe_offset=1]), Int(entry[unsafe_offset=3]) + 1):
                var row = y * self.columns
                for x in range(left, right + 1):
                    self.counts[row + x] += 1
        self.live_triangles = live
        var wanted = 0
        for index in range(len(self.counts)):
            wanted += Int(self.counts[index])
        var total_tiles = self.columns * self.rows
        var tile_flag = UInt32(total_tiles) if self.backend == BACKEND_VULKAN else UInt32(0)
        var first = 0
        for index in range(len(self.counts)):
            # Past capacity every tile gives up the same share of its list, and its
            # oldest entries: characters and water draw last.
            var count = Int(self.counts[index])
            var kept = count * self.list_capacity // wanted if wanted > self.list_capacity else count
            self.counts[index] = UInt32(count - kept)  # now the entries to skip
            count = kept
            self.cursors[index] = UInt32(first)
            self.tiles.write32(index * TILE_ENTRY_DWORDS, UInt32(first * 4))
            self.tiles.write32(index * TILE_ENTRY_DWORDS + 1, UInt32(count))
            self.tiles.write32(index * TILE_ENTRY_DWORDS + 2, self.tile_lights[index])
            self.tiles.write32(index * TILE_ENTRY_DWORDS + 3, tile_flag)
            first += count
        if len(self.cluster_lights) == total_tiles * CLUSTER_WORDS_PER_TILE and self.tiles.cpu_address != 0:
            copy_bytes(
                self.tiles.cpu_address + total_tiles * TILE_ENTRY_DWORDS * 4,
                Int(self.cluster_lights.unsafe_ptr()),
                total_tiles * CLUSTER_WORDS_PER_TILE * 4,
            )
        self.list_entries = first
        self.truncated_entries += wanted - first
        for triangle in range(self.triangle_count):
            var entry = self.bounds_entry(triangle)
            var left = Int(entry[unsafe_offset=0])
            var right = Int(entry[unsafe_offset=2])
            if left > right:
                continue
            var offset = UInt32(triangle * RECORD_BYTES)
            for y in range(Int(entry[unsafe_offset=1]), Int(entry[unsafe_offset=3]) + 1):
                var row = y * self.columns
                for x in range(left, right + 1):
                    if self.counts[row + x] > 0:
                        self.counts[row + x] -= 1
                    else:
                        self.lists.write32(Int(self.cursors[row + x]), offset)
                        self.cursors[row + x] += 1

    def prepare_voxels(mut self, world_width: Int, world_height: Int, world_depth: Int) -> Bool:
        """Size the shadow volume for a world. False if it does not fit, and then
        no voxels are read at all: none of an earlier layer's are left to cast
        shadows on this one."""
        var stride = max(world_width, world_depth)
        var used = world_height * stride * stride
        if used > VOXEL_BYTES:
            self.voxels_ready = False
            return False
        self.voxel_row = world_width
        self.voxel_rows = world_depth
        self.voxel_layers = world_height
        self.voxel_stride = stride
        self.voxel_bytes_used = used
        self.voxels_ready = True
        # The occupancy mip chain for the same world: a new size starts it over
        # (header, and the chunk level's CPU copy); chunks fill it as they upload.
        var layout = OccupancyLayout(world_width, world_height, world_depth)
        if self.occupancy.size > 0 and layout.bytes() <= self.occupancy.size:
            if not self.occupancy_ready or not (layout == self.occupancy_layout):
                self.occupancy_layout = layout
                self.occupancy_bits = List[UInt32](length=layout.l3_words, fill=0)
                write_occupancy_header(self.occupancy.cpu_address, layout)
                for word in range(layout.l3_words):
                    self.occupancy.write32(layout.l3 + word, 0)
                self.occupancy_ready = True
        else:
            self.occupancy_ready = False
        return True

    def voxel_address(self, x: Int, y: Int, z: Int) -> Int:
        return self.voxels.cpu_address + (y * self.voxel_stride + z) * self.voxel_stride + x

    def copy_voxel_row(mut self, x: Int, y: Int, z: Int, source: Int):
        """Eight material bytes from x along one row."""
        copy_bytes(self.voxel_address(x, y, z), source, 8)

    def fill_voxel_row(mut self, x: Int, y: Int, z: Int, material: UInt8):
        var row = UInt64(material) * 0x0101010101010101
        Pointer[UInt64, origin=MutAnyOrigin](unsafe_from_address=self.voxel_address(x, y, z))[unsafe_offset=0] = row

    def clear_tile_lights(mut self):
        for index in range(len(self.tile_lights)):
            self.tile_lights[index] = 0
        for index in range(len(self.cluster_lights)):
            self.cluster_lights[index] = 0

    def mark_tile_lights(mut self, light: Int, left: Int, top: Int, right: Int, bottom: Int):
        """Note that a light's bounds cover this pixel rectangle."""
        if light < 0 or light >= 32:
            return
        var first_column = max(0, left // TILE_SIZE)
        var last_column = min(self.columns - 1, right // TILE_SIZE)
        var first_row = max(0, top // TILE_SIZE)
        var last_row = min(self.rows - 1, bottom // TILE_SIZE)
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                self.tile_lights[row * self.columns + column] |= UInt32(1) << UInt32(light)

    def mark_cluster_light(
        mut self,
        light: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        min_vz: Float32,
        max_vz: Float32,
    ):
        """Mark a point or tube light in the 3D (tile_x, tile_y, z_slice) froxel bitmask."""
        if light < 0 or light >= MAX_CLUSTER_LIGHTS or max_vz <= 0.05:
            return
        if len(self.cluster_lights) != self.columns * self.rows * CLUSTER_WORDS_PER_TILE:
            return
        var first_column = max(0, left // TILE_SIZE)
        var last_column = min(self.columns - 1, right // TILE_SIZE)
        var first_row = max(0, top // TILE_SIZE)
        var last_row = min(self.rows - 1, bottom // TILE_SIZE)
        if first_column > last_column or first_row > last_row:
            return
        var s0 = 0 if min_vz < 18.0 else (1 if min_vz < 55.0 else (2 if min_vz < 145.0 else 3))
        var s1 = 0 if max_vz < 18.0 else (1 if max_vz < 55.0 else (2 if max_vz < 145.0 else 3))
        var word_off = light >> 5
        var bit = UInt32(1) << UInt32(light & 31)
        for row in range(first_row, last_row + 1):
            var row_base = row * self.columns
            for column in range(first_column, last_column + 1):
                var tile_idx = row_base + column
                for sl in range(s0, s1 + 1):
                    var froxel = (tile_idx * CLUSTER_Z_SLICES + sl) * CLUSTER_WORDS_PER_SLICE + word_off
                    self.cluster_lights[froxel] |= bit

    def write_float(self, byte_offset: Int, value: Float32):
        self.constants.write32(byte_offset // 4, float_bits(value))

    def address_values(self, addresses: List[UInt64]) -> List[UInt32]:
        var values = List[UInt32]()
        for address in addresses:
            values.append(UInt32(address & 0xFFFFFFFF))
            values.append(UInt32(address >> 32))
        return values^

    def shader_address(self, index: Int) -> UInt64:
        return self.code.gpu_address + UInt64(index * SHADER_BLOCK_DWORDS * 4)

    def add_bloom_passes(self, mut builder: CommandBuilder, bloom_width: Int, bloom_height: Int, strength: Int):
        var inputs = INPUT_GROUP_X | INPUT_GROUP_Y | INPUT_THREAD_XY
        var groups_x = UInt32((bloom_width + GROUP_WIDTH - 1) // GROUP_WIDTH)
        var groups_y = UInt32((bloom_height + GROUP_HEIGHT - 1) // GROUP_HEIGHT)
        var cells = self.bloom_cells.gpu_address
        var addresses = List[UInt64]()
        addresses.append(self.target.gpu_address + UInt64(self.bloom_offset * 4))
        addresses.append(cells + UInt64(BLOOM_SOURCE_OFFSET))
        var user = self.address_values(addresses)
        user.append(UInt32(self.row_pixels))
        user.append(UInt32(bloom_width))
        user.append(float_bits(Float32(self.width) / Float32(bloom_width)))
        user.append(float_bits(Float32(self.height) / Float32(bloom_height)))
        user.append(UInt32(self.width - 1))
        user.append(UInt32(self.height - 1))
        builder.compute_state(self.shader_address(SHADER_BLOOM_REDUCE), UInt32(GROUP_WIDTH), UInt32(GROUP_HEIGHT), 1, 10, VGPR_COUNT, inputs)
        builder.user_values(0, user)
        builder.dispatch(groups_x, groups_y, 1)
        builder.pass_barrier()

        for shader in [SHADER_BLUR_HORIZONTAL, SHADER_BLUR_VERTICAL]:
            var pass_addresses = List[UInt64]()
            if shader == SHADER_BLUR_HORIZONTAL:
                pass_addresses.append(cells + UInt64(BLOOM_SOURCE_OFFSET))
                pass_addresses.append(cells + UInt64(BLOOM_BLURRED_OFFSET))
            else:
                pass_addresses.append(cells + UInt64(BLOOM_BLURRED_OFFSET))
                pass_addresses.append(cells + UInt64(BLOOM_FINAL_OFFSET))
            var values = self.address_values(pass_addresses)
            values.append(UInt32(bloom_width))
            values.append(UInt32(bloom_width - 1))
            values.append(UInt32(bloom_height - 1))
            values.append(float_bits(1.0 / Float32(strength)))
            builder.compute_state(self.shader_address(shader), UInt32(GROUP_WIDTH), UInt32(GROUP_HEIGHT), 1, 8, VGPR_COUNT, inputs)
            builder.user_values(0, values)
            builder.dispatch(groups_x, groups_y, 1)
            builder.pass_barrier()

    def add_smooth_pass(self, mut builder: CommandBuilder):
        var addresses = List[UInt64]()
        addresses.append(self.target.gpu_address)
        addresses.append(self.target.gpu_address + UInt64(self.bloom_offset * 8))
        var user = self.address_values(addresses)
        user.append(UInt32(self.row_pixels))
        user.append(UInt32(self.width))
        user.append(UInt32(self.width - 1))
        user.append(UInt32(self.height - 1))
        builder.compute_state(self.shader_address(SHADER_SMOOTH), UInt32(GROUP_WIDTH), UInt32(GROUP_HEIGHT), 1, 8, VGPR_COUNT, INPUT_GROUP_X | INPUT_GROUP_Y | INPUT_THREAD_XY)
        builder.user_values(0, user)
        builder.dispatch(UInt32((self.width + GROUP_WIDTH - 1) // GROUP_WIDTH), UInt32((self.height + GROUP_HEIGHT - 1) // GROUP_HEIGHT), 1)
        builder.pass_barrier()

    def add_compose_pass(self, mut builder: CommandBuilder, bloom_width: Int, bloom_height: Int, bloom: Bool, smooth: Bool, output: UInt64, hud: UInt64):
        var addresses = List[UInt64]()
        addresses.append(self.target.gpu_address + (UInt64(self.bloom_offset * 8) if smooth else UInt64(0)))
        addresses.append(output)
        addresses.append(hud)
        addresses.append(self.bloom_cells.gpu_address + UInt64(BLOOM_FINAL_OFFSET))
        var user = self.address_values(addresses)
        user.append(UInt32(self.row_pixels))
        user.append(UInt32(self.width))
        user.append(UInt32(bloom_width))
        user.append(float_bits(Float32(bloom_width) / Float32(self.width)))
        user.append(float_bits(Float32(bloom_height) / Float32(self.height)))
        user.append(UInt32(1) if bloom else UInt32(0))
        user.append(UInt32(bloom_width - 1))
        user.append(UInt32(bloom_height - 1))
        builder.compute_state(self.shader_address(SHADER_COMPOSE), UInt32(GROUP_WIDTH), UInt32(GROUP_HEIGHT), 1, 16, VGPR_COUNT, INPUT_GROUP_X | INPUT_GROUP_Y | INPUT_THREAD_XY)
        builder.user_values(0, user)
        builder.dispatch(UInt32((self.width + GROUP_WIDTH - 1) // GROUP_WIDTH), UInt32((self.height + GROUP_HEIGHT - 1) // GROUP_HEIGHT), 1)
        builder.wait_for_shaders()

    def set_profiler(mut self, profiler: Profiler):
        """Time every pass from now on. Without this the command stream has no stamps."""
        if self.active and self.clock_hertz() == 0:
            return  # no rate to read the stamps with, so no stamps
        self.profiler = profiler.copy()
        if self.active and profiler.enabled():
            self.ticks_per_gpu_clock = profiler.ticks_per_second() / self.clock_hertz()
            self.measure_gpu_clock()

    def clock_hertz(self) -> Float64:
        """How fast timestamps count. 0 when this GPU cannot be timed."""
        if self.backend == BACKEND_VULKAN:
            if not self.vk.timestamps or self.vk.timestamp_period == 0 or not self.vk.has_calibration:
                return 0
            return 1000000000.0 / self.vk.timestamp_period
        return Float64(self.gpu.clock_khz) * 1000.0

    def read_gpu_clock(mut self) -> UInt64:
        return self.vk.read_clock() if self.backend == BACKEND_VULKAN else self.gpu.read_clock()

    def measure_gpu_clock(mut self):
        """Pair the two clocks. The ioctl's own time is the error, so keep the quickest try."""
        var best = UInt64(0xFFFFFFFFFFFFFFFF)
        for _ in range(4):
            var before = profile_ticks()
            var clock = self.read_gpu_clock()
            var after = profile_ticks()
            if clock != 0 and after - before < best:
                best = after - before
                self.clock_ticks = before + (after - before) // 2
                self.clock_gpu = clock
        self.clock_age = 0

    def stamp(mut self, mut builder: CommandBuilder, zone: Int):
        """End `zone` here. A value still zero afterwards means the write never landed."""
        var index = len(self.sets[self.current].stamp_zones)
        self.sets[self.current].stamps.write32(index * 2, 0)
        self.sets[self.current].stamps.write32(index * 2 + 1, 0)
        builder.write_timestamp(self.sets[self.current].stamps.gpu_address + UInt64(index * 8))
        self.sets[self.current].stamp_zones.append(zone)

    def stamp_ticks(self, value: UInt64) -> UInt64:
        """A GPU clock value on the CPU's clock."""
        var offset = (Float64(value) - Float64(self.clock_gpu)) * self.ticks_per_gpu_clock
        return UInt64(Int(self.clock_ticks) + Int(offset))

    def collect_stamps(mut self, index: Int):
        """Hand a finished frame's passes to the profiler. The GPU must be done with it."""
        var count = len(self.sets[index].stamp_zones)
        if count == 0:
            return
        if self.backend == BACKEND_VULKAN:
            _ = self.vk.read_queries(index * QUERIES_PER_FRAME, count, self.sets[index].stamps.cpu_address)
        var complete = count >= 2
        var values = List[UInt64](capacity=count)
        for stamp in range(count):
            var value = UInt64(self.sets[index].stamps.read32(stamp * 2)) | (UInt64(self.sets[index].stamps.read32(stamp * 2 + 1)) << 32)
            if value == 0 or (stamp > 0 and value < values[stamp - 1]):
                complete = False
            values.append(value)
        if complete:
            var frame = self.sets[index].stamp_frame
            self.profiler.record_gpu(ZONE_GPU_FRAME, frame, self.stamp_ticks(values[0]), self.stamp_ticks(values[count - 1]))
            for stamp in range(1, count):
                self.profiler.record_gpu(self.sets[index].stamp_zones[stamp], frame, self.stamp_ticks(values[stamp - 1]), self.stamp_ticks(values[stamp]))
        self.profiler.note_gpu_frame(complete)
        self.sets[index].stamp_zones.clear()
        # Clocks drift apart slowly, and the ioctl is not free: pair them a few times a minute.
        self.clock_age += 1
        if self.clock_age >= CLOCK_PAIR_FRAMES:
            self.measure_gpu_clock()

    def collect_all_stamps(mut self):
        """Wait for every frame in flight and collect it, so none goes unreported."""
        for index in range(len(self.sets)):
            if self.sets[index].fence != 0:
                if not self.wait_fence(self.sets[index].fence):
                    self.error = self.backend_error()
                    continue
                self.sets[index].fence = 0
            self.collect_stamps(index)

    def submit(mut self, mut tone: List[UInt8], saturation_steps: Int, clear_color: UInt32, bloom: Bool, bloom_width: Int, bloom_height: Int, bloom_strength: Int, effects: UInt32, smooth: Bool) -> Bool:
        """Queue every pass of the frame and return without waiting for the GPU."""
        self.settings = FrameSettings(clear_color, bloom, bloom_width, bloom_height, bloom_strength, smooth)
        var started = get_time()
        var bin_zone = self.profiler.now()
        self.bin_triangles()
        var binned = get_time()
        self.profiler.record(ZONE_GPU_BIN, bin_zone, self.profiler.now())
        var submit_zone = self.profiler.now()
        copy_bytes(self.constants.cpu_address, Int(tone.unsafe_ptr()), 768)
        self.constants.write32(CONSTANT_SATURATION // 4, UInt32(saturation_steps & 0xFFFFFFFF))
        self.constants.write32(CONSTANT_BLOOM_DISTANCE // 4, UInt32(self.bloom_offset * 4))
        self.constants.write32(CONSTANT_EFFECTS // 4, effects)

        if self.backend == BACKEND_VULKAN:
            if not self.record_vulkan_frame(clear_color, bloom, bloom_width, bloom_height, bloom_strength, smooth):
                return False
            self.finish_submit(started, binned, submit_zone, self.backdrop_wanted)
            return True

        var builder = CommandBuilder(self.gpu.isa())
        var timed = self.profiler.enabled()
        if timed:
            self.sets[self.current].stamp_frame = self.profiler.frame()
            self.stamp(builder, ZONE_NONE)
        var crumbs = self.sets[self.current].stamps.copy()
        var started_at = crumbs.gpu_address + UInt64(MARKER_STARTED * 4)
        var finished_at = crumbs.gpu_address + UInt64(MARKER_FINISHED * 4)
        crumbs.write32(MARKER_STARTED, MARKER_NONE)
        crumbs.write32(MARKER_FINISHED, MARKER_NONE)
        crumbs.write32(MARKER_FRAME, UInt32(self.frames & 0xFFFFFFFF))
        builder.write_marker(finished_at, 0)
        var held = self.fault_pending > 0 and self.fault_gate.size > 0
        if held:
            # Fault injection: hold the queue on a dword the CPU writes only in recover().
            self.fault_pending -= 1
            builder.write_marker(started_at, PASS_HELD)
            builder.wait_memory(self.fault_gate.gpu_address, self.fault_generation)
        builder.write_marker(started_at, PASS_RASTER)
        builder.compute_state(self.shader_address(SHADER_RASTER), UInt32(GROUP_WIDTH), UInt32(GROUP_HEIGHT), 1, 16, VGPR_COUNT, INPUT_GROUP_X | INPUT_GROUP_Y | INPUT_THREAD_XY)
        var user = List[UInt32]()
        for address in [self.records.gpu_address, self.lists.gpu_address, self.tiles.gpu_address, self.target.gpu_address, self.textures.gpu_address]:
            user.append(UInt32(address & 0xFFFFFFFF))
            user.append(UInt32(address >> 32))
        user.append(UInt32(self.row_pixels))
        user.append(UInt32(self.columns))
        user.append(clear_color)
        user.append(UInt32(self.width))
        user.append(UInt32(self.constants.gpu_address & 0xFFFFFFFF))
        user.append(UInt32(self.constants.gpu_address >> 32))
        builder.user_values(0, user)
        builder.dispatch(UInt32((self.width + GROUP_WIDTH - 1) // GROUP_WIDTH), UInt32((self.height + GROUP_HEIGHT - 1) // GROUP_HEIGHT), 1)
        builder.pass_barrier()
        builder.write_marker(finished_at, PASS_RASTER)
        if timed:
            self.stamp(builder, ZONE_GPU_RASTER)
        var extra = self.extra.has_work()
        if extra:
            builder.write_marker(started_at, PASS_EXTRA)
            self.add_extra_passes(builder)
            builder.write_marker(finished_at, PASS_EXTRA)
            if timed:
                self.stamp(builder, ZONE_GPU_PARTICLES)
        if bloom:
            builder.write_marker(started_at, PASS_BLOOM)
            self.add_bloom_passes(builder, bloom_width, bloom_height, bloom_strength)
            builder.write_marker(finished_at, PASS_BLOOM)
            if timed:
                self.stamp(builder, ZONE_GPU_BLOOM)
        if smooth:
            builder.write_marker(started_at, PASS_SMOOTH)
            self.add_smooth_pass(builder)
            builder.write_marker(finished_at, PASS_SMOOTH)
            if timed:
                self.stamp(builder, ZONE_GPU_SMOOTH)
        var capture = self.backdrop_wanted
        if capture:
            builder.write_marker(started_at, PASS_BACKDROP)
            self.add_compose_pass(builder, bloom_width, bloom_height, bloom, smooth, self.backdrop.gpu_address, self.backdrop.gpu_address + UInt64(self.bloom_offset * 4))
            builder.write_marker(finished_at, PASS_BACKDROP)
            if timed:
                self.stamp(builder, ZONE_GPU_BACKDROP)
        builder.write_marker(started_at, PASS_COMPOSE)
        self.add_compose_pass(builder, bloom_width, bloom_height, bloom, smooth, self.sets[self.current].shown.gpu_address, self.hud.gpu_address)
        builder.write_marker(finished_at, PASS_COMPOSE)
        if timed:
            self.stamp(builder, ZONE_GPU_COMPOSE)
        var dwords = builder.write_to(self.commands)

        var buffers = List[UInt32]()
        buffers.append(self.commands.handle)
        buffers.append(self.code.handle)
        buffers.append(self.records.handle)
        buffers.append(self.lists.handle)
        buffers.append(self.tiles.handle)
        buffers.append(self.constants.handle)
        buffers.append(self.textures.handle)
        buffers.append(self.target.handle)
        buffers.append(self.bloom_cells.handle)
        buffers.append(self.hud.handle)
        buffers.append(self.voxels.handle)
        buffers.append(self.sets[self.current].shown.handle)
        if capture:
            buffers.append(self.backdrop.handle)
        buffers.append(self.sets[self.current].stamps.handle)  # stamps and markers
        if held:
            buffers.append(self.fault_gate.handle)
        if extra:
            for handle in self.extra.handles:
                if handle not in buffers:
                    buffers.append(handle)
        var fence = self.gpu.submit(self.commands, self.gpu.pad_commands(self.commands, dwords), buffers)
        if fence == 0:
            self.error = self.gpu.error
            self.active = False
            if self.gpu.lost:
                self.note_hang(0)
            return False
        self.sets[self.current].fence = fence
        self.finish_submit(started, binned, submit_zone, capture)
        return True

    def add_extra_passes(self, mut builder: CommandBuilder):
        """The passes in `extra`, each followed by a barrier (amdgpu)."""
        for index in range(len(self.extra.passes)):
            ref p = self.extra.passes[index]
            kernel_compute_state(builder, p.code_address, p.vgprs, p.push_words)
            builder.user_address(0, p.table_address)
            if len(p.push) > 0:
                builder.user_values(2, p.push)
            if p.indirect:
                builder.dispatch_indirect(self.extra.indirect.gpu_address + UInt64(p.indirect_offset))
            else:
                builder.dispatch(UInt32(p.groups_x), UInt32(p.groups_y), 1)
            if p.feeds_indirect:
                builder.arguments_barrier()
            else:
                builder.pass_barrier()

    def record_extra_vulkan(mut self, frame: Int):
        """The passes in `extra` on Vulkan, each followed by a barrier that
        also covers the next indirect dispatch's counts."""
        for index in range(len(self.extra.passes)):
            ref p = self.extra.passes[index]
            if p.indirect:
                self.vk.dispatch_indirect(frame, p.pipeline, p.descriptor, p.push, self.extra.indirect, p.indirect_offset)
            else:
                self.vk.dispatch(frame, p.pipeline, p.descriptor, p.push, p.groups_x, p.groups_y)
            self.vk.indirect_barrier(frame)

    def finish_submit(mut self, started: Float64, binned: Float64, submit_zone: UInt64, capture: Bool):
        """The frame is queued, and sets[current].fence says when it is done."""
        if self.extra.has_work():
            self.extra.copies += 1
        self.extra.clear()
        self.last_submitted = self.current
        if capture:
            self.backdrop_wanted = False
            self.backdrop_ready = True
            self.backdrop_fence = self.sets[self.current].fence
        self.bin_seconds += binned - started
        self.copy_seconds += get_time() - binned
        self.frames += 1
        self.profiler.record(ZONE_GPU_SUBMIT, submit_zone, self.profiler.now())

    def stamp_vulkan(mut self, zone: Int):
        """End `zone` here, as `stamp` does through the query pool."""
        var frame = self.current
        var index = len(self.sets[frame].stamp_zones)
        self.sets[frame].stamps.write32(index * 2, 0)
        self.sets[frame].stamps.write32(index * 2 + 1, 0)
        self.vk.write_timestamp(frame, frame * QUERIES_PER_FRAME + index)
        self.sets[frame].stamp_zones.append(zone)

    def dispatch_vulkan(mut self, shader: Int, push: List[UInt32], width: Int, height: Int):
        var frame = self.current
        self.vk.dispatch(frame, shader, self.sets[frame].descriptor, push, (width + GROUP_WIDTH - 1) // GROUP_WIDTH, (height + GROUP_HEIGHT - 1) // GROUP_HEIGHT)

    def compose_push(
        self,
        bloom_width: Int,
        bloom_height: Int,
        bloom: Bool,
        clear_color: UInt32,
        rw: Int = 0,
        rh: Int = 0,
    ) -> List[UInt32]:
        var eff_w = rw if (rw > 0 and rw <= self.width) else self.width
        var eff_h = rh if (rh > 0 and rh <= self.height) else self.height
        var push = List[UInt32]()
        push.append(UInt32(self.row_pixels))
        push.append(UInt32(eff_w))
        push.append(UInt32(eff_h))
        push.append(UInt32(bloom_width))
        push.append(float_bits(Float32(bloom_width) / Float32(max(1, eff_w))))
        push.append(float_bits(Float32(bloom_height) / Float32(max(1, eff_h))))
        push.append(UInt32(1) if bloom else UInt32(0))
        push.append(clear_color)
        push.append(UInt32(bloom_height))
        push.append(UInt32(self.bloom_offset))
        return push^

    def taa_push(self, smooth: Bool) -> List[UInt32]:
        var push = List[UInt32]()
        push.append(UInt32(self.row_pixels))
        push.append(UInt32(self.width))
        push.append(UInt32(self.height))
        push.append(UInt32(self.bloom_offset * 2) if smooth else UInt32(0))
        push.append(UInt32(self.bloom_offset))
        return push^

    def record_vulkan_frame(mut self, clear_color: UInt32, bloom: Bool, bloom_width: Int, bloom_height: Int, bloom_strength: Int, smooth: Bool) -> Bool:
        """Record 2-Cascade CSM -> Linear HDR Raster -> Linear HDR Dual-Kawase Bloom ->
        SSR + Volumetrics + SSAO + AgX Compose -> Optional FXAA -> YCoCg TAA + Motion Blur + HUD."""
        var frame = self.current
        var presenting = self.vk.swapchain != 0
        var image = self.vk.acquire_image(frame) if presenting else NO_IMAGE
        if not self.vk.begin_commands(frame):
            self.error = self.vk.error
            self.active = False
            return False
        var timed = self.profiler.enabled()
        if timed:
            self.sets[frame].stamp_frame = self.profiler.frame()
            self.vk.reset_queries(frame, frame * QUERIES_PER_FRAME, QUERIES_PER_FRAME)
            self.stamp_vulkan(ZONE_NONE)

        # 1. 2-Cascade GPU Shadow Map Clear + Tile Rasterization
        var csm_quads = CSM_DEPTH_WORDS // 4
        var csm_clear_p = List[UInt32]()
        csm_clear_p.append(UInt32(csm_quads))
        self.vk.dispatch(frame, SHADER_CSM_CLEAR, self.sets[frame].descriptor, csm_clear_p, (csm_quads + 63) // 64, 1)
        self.vk.pass_barrier(frame)
        if self.csm_item_count > 0:
            var slice_words = (CSM_DEPTH_BYTES + frame * CSM_FRAME_SLICE_BYTES) // 4
            var rec_words = slice_words + CSM_RECORDS_REL_BYTES // 4
            var item_words = slice_words + CSM_ITEMS_REL_BYTES // 4
            var gx = min(self.csm_item_count, 4096)
            var gy = (self.csm_item_count + 4095) // 4096
            var csm_rast_p = List[UInt32]()
            csm_rast_p.append(UInt32(rec_words))
            csm_rast_p.append(UInt32(item_words))
            csm_rast_p.append(UInt32(self.csm_item_count))
            csm_rast_p.append(UInt32(gx))
            self.vk.dispatch(frame, SHADER_CSM_RASTER, self.sets[frame].descriptor, csm_rast_p, gx, gy)
            self.vk.pass_barrier(frame)

        # 2. Linear Float32 HDR + G-buffer Tile Rasterization (at eff_rw x eff_rh for Sub-Native TSR!)
        var eff_rw = self.render_width if (self.render_width > 0 and self.render_width <= self.width) else self.width
        var eff_rh = self.render_height if (self.render_height > 0 and self.render_height <= self.height) else self.height
        var push = List[UInt32]()
        push.append(UInt32(self.row_pixels))
        push.append(UInt32(self.columns))
        push.append(clear_color)
        push.append(UInt32(eff_rw))
        push.append(UInt32(eff_rh))
        self.dispatch_vulkan(SHADER_RASTER, push, eff_rw, eff_rh)
        self.vk.pass_barrier(frame)
        if timed:
            self.stamp_vulkan(ZONE_GPU_RASTER)
        if self.extra.has_work():
            self.record_extra_vulkan(frame)
            if timed:
                self.stamp_vulkan(ZONE_GPU_PARTICLES)

        # 3. Linear Float32 HDR Multi-Scale Dual-Kawase Bloom
        if bloom:
            var reduce = List[UInt32]()
            reduce.append(UInt32(self.row_pixels))
            reduce.append(UInt32(bloom_width))
            reduce.append(UInt32(bloom_height))
            reduce.append(float_bits(Float32(eff_rw) / Float32(bloom_width)))
            reduce.append(float_bits(Float32(eff_rh) / Float32(bloom_height)))
            reduce.append(UInt32(eff_rw - 1))
            reduce.append(UInt32(eff_rh - 1))
            reduce.append(UInt32(self.bloom_offset))
            self.dispatch_vulkan(SHADER_BLOOM_REDUCE, reduce, bloom_width, bloom_height)
            self.vk.pass_barrier(frame)
            var multi_scale = bloom_width >= 16 and bloom_height >= 12
            for shader in [SHADER_BLUR_HORIZONTAL, SHADER_BLUR_VERTICAL]:
                var horizontal = shader == SHADER_BLUR_HORIZONTAL
                var blur = List[UInt32]()
                blur.append(UInt32(bloom_width))
                blur.append(UInt32(bloom_height))
                blur.append(float_bits(1.0 / Float32(bloom_strength)))
                blur.append(UInt32((BLOOM_SOURCE_OFFSET if horizontal else BLOOM_BLURRED_OFFSET) // 4))
                blur.append(
                    UInt32(
                        (
                            BLOOM_BLURRED_OFFSET
                            if horizontal
                            else (BLOOM_MIP0_CRISP_OFFSET if multi_scale else BLOOM_FINAL_OFFSET)
                        )
                        // 4
                    )
                )
                blur.append(UInt32(0) if (not horizontal and multi_scale) else UInt32(1))
                self.dispatch_vulkan(shader, blur, bloom_width, bloom_height)
                self.vk.pass_barrier(frame)
            if multi_scale:
                var w1 = max(2, bloom_width // 2)
                var h1 = max(2, bloom_height // 2)
                var w2 = max(2, w1 // 2)
                var h2 = max(2, h1 // 2)
                var down1 = List[UInt32]()
                down1.append(UInt32(bloom_width))
                down1.append(UInt32(bloom_height))
                down1.append(UInt32(w1))
                down1.append(UInt32(h1))
                down1.append(UInt32(BLOOM_MIP0_CRISP_OFFSET // 4))
                down1.append(UInt32(BLOOM_MIP1_OFFSET // 4))
                self.dispatch_vulkan(SHADER_KAWASE_DOWN, down1, w1, h1)
                self.vk.pass_barrier(frame)

                var down2 = List[UInt32]()
                down2.append(UInt32(w1))
                down2.append(UInt32(h1))
                down2.append(UInt32(w2))
                down2.append(UInt32(h2))
                down2.append(UInt32(BLOOM_MIP1_OFFSET // 4))
                down2.append(UInt32(BLOOM_MIP2_OFFSET // 4))
                self.dispatch_vulkan(SHADER_KAWASE_DOWN, down2, w2, h2)
                self.vk.pass_barrier(frame)

                var up1 = List[UInt32]()
                up1.append(UInt32(w2))
                up1.append(UInt32(h2))
                up1.append(UInt32(w1))
                up1.append(UInt32(h1))
                up1.append(UInt32(BLOOM_MIP2_OFFSET // 4))
                up1.append(UInt32(BLOOM_MIP1_OFFSET // 4))
                up1.append(UInt32(BLOOM_MIP1_UP_OFFSET // 4))
                up1.append(float_bits(Float32(0.0)))
                self.dispatch_vulkan(SHADER_KAWASE_UP, up1, w1, h1)
                self.vk.pass_barrier(frame)

                var up0 = List[UInt32]()
                up0.append(UInt32(w1))
                up0.append(UInt32(h1))
                up0.append(UInt32(bloom_width))
                up0.append(UInt32(bloom_height))
                up0.append(UInt32(BLOOM_MIP1_UP_OFFSET // 4))
                up0.append(UInt32(BLOOM_MIP0_CRISP_OFFSET // 4))
                up0.append(UInt32(BLOOM_FINAL_OFFSET // 4))
                up0.append(float_bits(1.0 / Float32(bloom_strength)))
                self.dispatch_vulkan(SHADER_KAWASE_UP, up0, bloom_width, bloom_height)
                self.vk.pass_barrier(frame)
            if timed:
                self.stamp_vulkan(ZONE_GPU_BLOOM)

        # 3b. V8 Hillaire Sky-View LUT + Temporally Reprojected 3D Froxel Volumetrics + Coarse Hi-Z Pyramid + Auto-Exposure
        var frame_effects = self.sets[frame].constants.read32(CONSTANT_EFFECTS // 4)
        if (frame_effects & EFFECT_PHYSICAL_SKY) != 0:
            var sky_p = List[UInt32]()
            sky_p.append(UInt32(0))
            self.vk.dispatch(frame, SHADER_SKY_LUT, self.sets[frame].descriptor, sky_p, (SKY_LUT_W + 7) // 8, (SKY_LUT_H + 7) // 8)
            self.vk.pass_barrier(frame)
        if (frame_effects & EFFECT_VOLUMETRIC) != 0 and (frame_effects & EFFECT_FROXEL_VOL) != 0:
            var ping = self.frames & 1
            var frox_p = List[UInt32]()
            frox_p.append(UInt32((FROXEL_VOL_OFFSET + (1 - ping) * FROXEL_SLICE_BYTES) // 4))
            frox_p.append(UInt32((FROXEL_VOL_OFFSET + ping * FROXEL_SLICE_BYTES) // 4))
            frox_p.append(UInt32(FROXEL_INTEGRATED_OFFSET // 4))
            frox_p.append(UInt32(self.frames))
            self.vk.dispatch(
                frame,
                SHADER_FROXEL_INTEGRATE,
                self.sets[frame].descriptor,
                frox_p,
                (FROXEL_X + 7) // 8,
                (FROXEL_Y + 3) // 4,
            )
            self.vk.pass_barrier(frame)
        if (frame_effects & EFFECT_HIZ_CULL) != 0 and eff_rw == self.width and eff_rh == self.height and self.hiz_tiles_x <= 256 and self.hiz_tiles_y <= 144:
            var hiz_out_word = self.bloom_offset * 2 + frame * HIZ_SLICE_WORDS
            var hiz_p = List[UInt32]()
            hiz_p.append(UInt32(self.row_pixels))
            hiz_p.append(UInt32(eff_rw))
            hiz_p.append(UInt32(eff_rh))
            hiz_p.append(UInt32(self.hiz_tiles_x))
            hiz_p.append(UInt32(self.bloom_offset * 3))
            hiz_p.append(UInt32(hiz_out_word))
            self.vk.dispatch(
                frame,
                SHADER_HIZ_PYRAMID,
                self.sets[frame].descriptor,
                hiz_p,
                (self.hiz_tiles_x + 7) // 8,
                (self.hiz_tiles_y + 3) // 4,
            )
            self.vk.pass_barrier(frame)
            self.hiz_valid_mask |= UInt32(1) << UInt32(frame)
        else:
            self.hiz_valid_mask &= ~(UInt32(1) << UInt32(frame))
            if self.hiz_synced_slot == frame:
                self.hiz_synced_slot = -1
        if (frame_effects & EFFECT_AUTO_EXPOSURE) != 0:
            var hist_p = List[UInt32]()
            hist_p.append(UInt32(self.row_pixels))
            hist_p.append(UInt32(eff_rw))
            hist_p.append(UInt32(eff_rh))
            hist_p.append(UInt32(self.bloom_offset))
            var hist_w = (eff_rw + 3) // 4
            var hist_h = (eff_rh + 3) // 4
            self.vk.dispatch(frame, SHADER_HISTOGRAM, self.sets[frame].descriptor, hist_p, (hist_w + 15) // 16, (hist_h + 15) // 16)
            self.vk.pass_barrier(frame)
            var ae_p = List[UInt32]()
            self.vk.dispatch(frame, SHADER_AUTO_EXPOSURE, self.sets[frame].descriptor, ae_p, 1, 1)
            self.vk.pass_barrier(frame)

        # 4. Linear HDR Post-Compose (Refraction + SSR + Volumetrics + GTAO/Contact Shadows + Bloom + Auto-Exp + AgX -> PLANE_COLOR)
        var compose = self.compose_push(bloom_width, bloom_height, bloom, clear_color, eff_rw, eff_rh)
        self.dispatch_vulkan(SHADER_COMPOSE, compose, eff_rw, eff_rh)
        self.vk.pass_barrier(frame)

        # 5. Optional FXAA Edge Smoothing (PLANE_COLOR -> PLANE_SMOOTH)
        if smooth:
            var soften = List[UInt32]()
            soften.append(UInt32(self.row_pixels))
            soften.append(UInt32(eff_rw))
            soften.append(UInt32(eff_rh))
            soften.append(UInt32(self.bloom_offset * 2))
            self.dispatch_vulkan(SHADER_SMOOTH, soften, eff_rw, eff_rh)
            self.vk.pass_barrier(frame)
            if timed:
                self.stamp_vulkan(ZONE_GPU_SMOOTH)

        # 6. CAS + 3x3 YCoCg TAA + Supersonic Velocity Motion Blur + HUD Composite
        var taa = self.taa_push(smooth)
        if self.backdrop_wanted:
            self.dispatch_vulkan(SHADER_TAA_CAPTURE, taa, self.width, self.height)
            self.vk.pass_barrier(frame)
            if timed:
                self.stamp_vulkan(ZONE_GPU_BACKDROP)
        self.dispatch_vulkan(SHADER_TAA_RESOLVE, taa, self.width, self.height)
        self.vk.barrier(frame, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
        if timed:
            self.stamp_vulkan(ZONE_GPU_COMPOSE)
        if image != NO_IMAGE:
            self.vk.copy_to_image(frame, self.sets[frame].shown, self.row_pixels, image)
        var fence = self.vk.submit_and_present(frame, image) if image != NO_IMAGE else self.vk.submit(frame)
        if fence == 0:
            self.error = self.vk.error
            self.active = False
            return False
        self.sets[frame].fence = fence
        return True

    def read_back(mut self, mut pixels: List[UInt32]) -> Bool:
        """Wait for the newest frame and copy it to the CPU, for screenshots and shm."""
        if self.last_submitted < 0:
            return False
        var fence = self.sets[self.last_submitted].fence
        if fence != 0 and not self.wait_fence(fence):
            self.error = self.backend_error()
            self.active = False
            return False
        self.sets[self.last_submitted].fence = 0
        if (self.hiz_valid_mask & (UInt32(1) << UInt32(self.last_submitted))) != 0:
            self.hiz_synced_slot = self.last_submitted
        self.copy_image(self.sets[self.last_submitted].shown.cpu_address, pixels)
        return True

    def is_rect_occluded_hiz(
        self,
        min_px_x: Int,
        min_px_y: Int,
        max_px_x: Int,
        max_px_y: Int,
        nearest_iz: Float32,
    ) -> Bool:
        """Query the latest completed 16x16 Hi-Z pyramid: True iff every overlapping tile's
        minimum inverse_z (farthest depth) is strictly closer than `nearest_iz`."""
        if (
            self.backend != BACKEND_VULKAN
            or self.hiz_synced_slot < 0
            or self.backdrop.cpu_address == 0
            or self.hiz_tiles_x <= 0
            or self.hiz_tiles_y <= 0
        ):
            return False
        # Conservative 1-tile dilation to tolerate inter-frame camera/object motion.
        var tx0 = max(0, (min_px_x // 16) - 1)
        var ty0 = max(0, (min_px_y // 16) - 1)
        var tx1 = min(self.hiz_tiles_x - 1, (max_px_x // 16) + 1)
        var ty1 = min(self.hiz_tiles_y - 1, (max_px_y // 16) + 1)
        if tx0 > tx1 or ty0 > ty1:
            return False
        var base_addr = (
            self.backdrop.cpu_address
            + (self.bloom_offset * 2 + self.hiz_synced_slot * HIZ_SLICE_WORDS) * 4
        )
        var ptr = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=base_addr)
        var biased_iz = nearest_iz * Float32(1.03)
        for ty in range(ty0, ty1 + 1):
            var row_off = ty * self.hiz_tiles_x
            for tx in range(tx0, tx1 + 1):
                if ptr[unsafe_offset=row_off + tx] <= biased_iz:
                    return False
        return True

    def run_gpu_skin(
        mut self,
        joint_matrices_3x4: List[Float32],
        in_packed_words: List[UInt32],
        vert_count: Int,
        mut out_pos_norm: List[Float32],
    ) -> Bool:
        """Dispatch SHADER_GPU_SKIN on Vulkan using auxiliary command buffer `FRAME_SETS`."""
        if (
            not self.active
            or self.backend != BACKEND_VULKAN
            or self.backdrop.cpu_address == 0
            or len(self.sets) == 0
            or vert_count <= 0
        ):
            return False
        var joint_floats = min(len(joint_matrices_3x4), GPU_SKIN_MAX_JOINTS * 12)
        if joint_floats < 12:
            return False
        out_pos_norm.resize(vert_count * 6, 0.0)
        var skin_base_words = self.bloom_offset * 2 + GPU_SKIN_REL_BYTES // 4
        var skin_base_addr = self.backdrop.cpu_address + skin_base_words * 4
        copy_bytes(skin_base_addr, Int(joint_matrices_3x4.unsafe_ptr()), joint_floats * 4)

        var aux_cmd = FRAME_SETS
        var processed = 0
        while processed < vert_count:
            var batch = min(vert_count - processed, GPU_SKIN_MAX_VERTS)
            var in_words_off = skin_base_words + GPU_SKIN_MAX_JOINTS * 12
            var out_words_off = in_words_off + batch * 12
            copy_bytes(
                self.backdrop.cpu_address + in_words_off * 4,
                Int(in_packed_words.unsafe_ptr()) + processed * 12 * 4,
                batch * 12 * 4,
            )
            if not self.vk.begin_commands(aux_cmd):
                return False
            var push = List[UInt32]()
            push.append(UInt32(skin_base_words))
            push.append(UInt32(in_words_off))
            push.append(UInt32(out_words_off))
            push.append(UInt32(batch))
            self.vk.dispatch(
                aux_cmd,
                SHADER_GPU_SKIN,
                self.sets[0].descriptor,
                push,
                (batch + 63) // 64,
                1,
            )
            self.vk.barrier(aux_cmd, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
            var fence = self.vk.submit(aux_cmd)
            if fence == 0 or not self.vk.wait(fence):
                return False
            copy_bytes(
                Int(out_pos_norm.unsafe_ptr()) + processed * 6 * 4,
                self.backdrop.cpu_address + out_words_off * 4,
                batch * 6 * 4,
            )
            processed += batch
        return True

    def run_gpu_particles(
        mut self,
        mut packed_particles: List[UInt32],
        count: Int,
        dt: Float32,
        gravity: Float32 = -9.8,
        drag: Float32 = 0.85,
        depth_bounce: Bool = True,
        restitution: Float32 = 0.45,
    ) -> Bool:
        """Dispatch SHADER_GPU_PARTICLES (up to 65,536 particles) on Vulkan with screen-space depth bounce."""
        var n = min(count, min(GPU_PARTICLE_MAX, len(packed_particles) // GPU_PARTICLE_STRIDE_WORDS))
        if (
            not self.active
            or self.backend != BACKEND_VULKAN
            or self.backdrop.cpu_address == 0
            or len(self.sets) == 0
            or n <= 0
        ):
            return False
        if depth_bounce and self.last_submitted >= 0:
            if not self.finish_frames():
                return False
        var in_words_off = self.bloom_offset * 2 + GPU_PARTICLE_REL_BYTES // 4
        var out_words_off = in_words_off + GPU_PARTICLE_MAX * GPU_PARTICLE_STRIDE_WORDS
        var byte_count = n * GPU_PARTICLE_STRIDE_WORDS * 4
        copy_bytes(
            self.backdrop.cpu_address + in_words_off * 4,
            Int(packed_particles.unsafe_ptr()),
            byte_count,
        )
        var aux_cmd = FRAME_SETS
        if not self.vk.begin_commands(aux_cmd):
            return False
        var dt_cp = dt
        var grav_cp = gravity
        var drag_cp = drag
        var dt_bits = Pointer(to=dt_cp).unsafe_bitcast[UInt32]()[unsafe_offset=0]
        var grav_bits = Pointer(to=grav_cp).unsafe_bitcast[UInt32]()[unsafe_offset=0]
        var drag_bits = Pointer(to=drag_cp).unsafe_bitcast[UInt32]()[unsafe_offset=0]
        var depth_off = UInt32(self.bloom_offset * 3) if (depth_bounce and self.last_submitted >= 0) else UInt32(0)
        var rest_q = UInt32(max(0, min(255, Int(restitution * Float32(255.0) + Float32(0.5)))))
        var row_bounce = UInt32(self.row_pixels & 0xFFFF) | (rest_q << 16)
        var push = List[UInt32]()
        push.append(UInt32(in_words_off))
        push.append(UInt32(out_words_off))
        push.append(UInt32(n))
        push.append(dt_bits)
        push.append(grav_bits)
        push.append(drag_bits)
        push.append(depth_off)
        push.append(row_bounce)
        var set_idx = self.current if (self.current >= 0 and self.current < len(self.sets)) else 0
        self.vk.dispatch(
            aux_cmd,
            SHADER_GPU_PARTICLES,
            self.sets[set_idx].descriptor,
            push,
            (n + 63) // 64,
            1,
        )
        self.vk.barrier(aux_cmd, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
        var fence = self.vk.submit(aux_cmd)
        if fence == 0 or not self.vk.wait(fence):
            return False
        copy_bytes(
            Int(packed_particles.unsafe_ptr()),
            self.backdrop.cpu_address + out_words_off * 4,
            byte_count,
        )
        _ = len(packed_particles)
        return True

    def run_tsr_upscale(
        mut self,
        src_pixels: List[UInt32],
        src_w: Int,
        src_h: Int,
        dst_w: Int,
        dst_h: Int,
        mut out_pixels: List[UInt32],
        sharpness: Float32 = 0.42,
        history_blend: Float32 = 0.0,
        jitter_x: Float32 = 0.0,
        jitter_y: Float32 = 0.0,
        history_parity: Int = 0,
    ) -> Bool:
        """Dispatch V10 SHADER_TSR_UPSCALE (5-Tap Catmull-Rom + YCoCg history clamp) on Vulkan."""
        if (
            not self.active
            or self.backend != BACKEND_VULKAN
            or self.backdrop.cpu_address == 0
            or len(self.sets) == 0
            or src_w < 2
            or src_h < 2
            or dst_w < 2
            or dst_h < 2
            or len(src_pixels) < src_w * src_h
        ):
            return False
        var src_words = src_w * src_h
        var dst_words = dst_w * dst_h
        var max_words = (HIZ_BACKDROP_EXTRA_BYTES - TSR_SCRATCH_REL_BYTES) // 4
        if src_words + dst_words > max_words:
            return False
        if history_blend > Float32(0.005) and self.last_submitted >= 0:
            if not self.finish_frames():
                return False
        out_pixels.resize(dst_words, 0)
        var tsr_base_words = self.bloom_offset * 2 + TSR_SCRATCH_REL_BYTES // 4
        var dst_words_off = tsr_base_words + src_words
        copy_bytes(
            self.backdrop.cpu_address + tsr_base_words * 4,
            Int(src_pixels.unsafe_ptr()),
            src_words * 4,
        )
        var aux_cmd = FRAME_SETS
        if not self.vk.begin_commands(aux_cmd):
            return False
        var use_hist = history_blend > Float32(0.005) and dst_w == self.width and dst_h == self.height and self.last_submitted >= 0
        var hist_plane = 9 if history_parity == 0 else 10
        var hist_off = UInt32(self.bloom_offset * hist_plane) if use_hist else UInt32(0)
        var vel_off = UInt32(self.bloom_offset * 11) if use_hist else UInt32(0)
        var blend_q = UInt32(max(0, min(255, Int(history_blend * Float32(255.0) + Float32(0.5)))))
        var row_blend = UInt32(self.row_pixels & 0xFFFF) | (blend_q << 16)
        var sharp_q = UInt32(max(0, min(255, Int(sharpness * Float32(127.5) + Float32(0.5)))))
        var jx_q = UInt32(max(0, min(255, Int(jitter_x * Float32(128.0) + Float32(128.5)))))
        var jy_q = UInt32(max(0, min(255, Int(jitter_y * Float32(128.0) + Float32(128.5)))))
        var sharp_jit = sharp_q | (jx_q << 8) | (jy_q << 16)
        var push = List[UInt32]()
        push.append(UInt32(tsr_base_words))
        push.append(UInt32(dst_words_off))
        push.append(hist_off)
        push.append(vel_off)
        push.append(UInt32(src_w & 0xFFFF) | (UInt32(src_h & 0xFFFF) << 16))
        push.append(UInt32(dst_w & 0xFFFF) | (UInt32(dst_h & 0xFFFF) << 16))
        push.append(row_blend)
        push.append(sharp_jit)
        var set_idx = self.last_submitted if (self.last_submitted >= 0 and self.last_submitted < len(self.sets)) else 0
        self.vk.dispatch(
            aux_cmd,
            SHADER_TSR_UPSCALE,
            self.sets[set_idx].descriptor,
            push,
            (dst_w + GROUP_WIDTH - 1) // GROUP_WIDTH,
            (dst_h + GROUP_HEIGHT - 1) // GROUP_HEIGHT,
        )
        self.vk.barrier(aux_cmd, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
        var fence = self.vk.submit(aux_cmd)
        if fence == 0 or not self.vk.wait(fence):
            return False
        copy_bytes(
            Int(out_pixels.unsafe_ptr()),
            self.backdrop.cpu_address + dst_words_off * 4,
            dst_words * 4,
        )
        _ = len(src_pixels)
        _ = len(out_pixels)
        return True

    def run_gpu_cull(
        mut self,
        packed_instances: List[UInt32],
        instance_count: Int,
        mut out_visibility: List[UInt32],
        mut out_indirect_cmds: List[UInt32],
        use_hiz: Bool = True,
    ) -> Int:
        """Dispatch V10 SHADER_GPU_CULL (Frustum + 16x16 Hi-Z Occlusion -> Indirect Draw Buffer) on Vulkan.
        Returns visible count (>= 0) on success, or -1 if GPU path unavailable."""
        var n = min(instance_count, min(GPU_CULL_MAX_INSTANCES, len(packed_instances) // GPU_CULL_IN_STRIDE_WORDS))
        if (
            not self.active
            or self.backend != BACKEND_VULKAN
            or self.backdrop.cpu_address == 0
            or len(self.sets) == 0
            or n <= 0
        ):
            return -1
        if use_hiz and self.last_submitted >= 0:
            if self.finish_frames() and (self.hiz_valid_mask & (UInt32(1) << UInt32(self.last_submitted))) != 0:
                self.hiz_synced_slot = self.last_submitted
        var cull_base_words = self.bloom_offset * 2 + GPU_CULL_REL_BYTES // 4
        var vis_words_off = cull_base_words + GPU_CULL_MAX_INSTANCES * GPU_CULL_IN_STRIDE_WORDS
        var cnt_words_off = vis_words_off + GPU_CULL_MAX_INSTANCES
        var ind_words_off = cnt_words_off + 16
        copy_bytes(
            self.backdrop.cpu_address + cull_base_words * 4,
            Int(packed_instances.unsafe_ptr()),
            n * GPU_CULL_IN_STRIDE_WORDS * 4,
        )
        Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.backdrop.cpu_address + cnt_words_off * 4)[unsafe_offset=0] = 0
        var aux_cmd = FRAME_SETS
        if not self.vk.begin_commands(aux_cmd):
            return -1
        var hiz_ready = use_hiz and self.hiz_synced_slot >= 0 and self.hiz_tiles_x > 0 and self.hiz_tiles_y > 0
        var hiz_off = UInt32(self.bloom_offset * 2 + self.hiz_synced_slot * HIZ_SLICE_WORDS) if hiz_ready else UInt32(0)
        var hiz_dims = (UInt32(self.hiz_tiles_x & 0xFFFF) | (UInt32(self.hiz_tiles_y & 0xFFFF) << 16)) if hiz_ready else UInt32(0)
        var scr_wh = UInt32(self.width & 0xFFFF) | (UInt32(self.height & 0xFFFF) << 16)
        var push = List[UInt32]()
        push.append(UInt32(cull_base_words))
        push.append(UInt32(vis_words_off))
        push.append(UInt32(cnt_words_off))
        push.append(UInt32(ind_words_off))
        push.append(UInt32(n))
        push.append(hiz_off)
        push.append(hiz_dims)
        push.append(scr_wh)
        var set_idx = self.current if (self.current >= 0 and self.current < len(self.sets) and self.constants.cpu_address != 0) else (
            self.last_submitted if (self.last_submitted >= 0 and self.last_submitted < len(self.sets)) else 0
        )
        self.vk.dispatch(
            aux_cmd,
            SHADER_GPU_CULL,
            self.sets[set_idx].descriptor,
            push,
            (n + 63) // 64,
            1,
        )
        self.vk.barrier(aux_cmd, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
        var fence = self.vk.submit(aux_cmd)
        if fence == 0 or not self.vk.wait(fence):
            return -1
        var vis_cnt = Int(
            Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.backdrop.cpu_address + cnt_words_off * 4)[unsafe_offset=0]
        )
        vis_cnt = max(0, min(n, vis_cnt))
        out_visibility.resize(n, 0)
        copy_bytes(
            Int(out_visibility.unsafe_ptr()),
            self.backdrop.cpu_address + vis_words_off * 4,
            n * 4,
        )
        out_indirect_cmds.resize(vis_cnt * 5, 0)
        if vis_cnt > 0:
            copy_bytes(
                Int(out_indirect_cmds.unsafe_ptr()),
                self.backdrop.cpu_address + ind_words_off * 4,
                vis_cnt * 5 * 4,
            )
        _ = len(packed_instances)
        _ = len(out_visibility)
        _ = len(out_indirect_cmds)
        return vis_cnt

    def read_back_backdrop(mut self, mut pixels: List[UInt32]) -> Bool:
        """Wait for the captured frame and copy its scene, without the HUD, to the CPU."""
        if not self.wait_fence(self.backdrop_fence):
            self.error = self.backend_error()
            self.active = False
            return False
        self.copy_image(self.backdrop.cpu_address, pixels)
        return True

    def mirror(mut self, source: GpuRaster, slot: Int) -> Bool:
        """Draw the frame `source` just submitted, from a copy of its frame description.

        This is how the backends are compared on real frames: two runs of the game
        are not the same frame, because presentation runs on the wall clock, but one
        frame description through two sets of shaders is. Sizes must match.
        """
        if not self.begin(slot):
            return False
        ref frame = source.sets[source.last_submitted]
        copy_bytes(self.records.cpu_address, frame.records.cpu_address, source.triangle_count * RECORD_BYTES)
        copy_bytes(self.lists.cpu_address, frame.lists.cpu_address, source.list_entries * 4)
        copy_bytes(self.tiles.cpu_address, frame.tiles.cpu_address, frame.tiles.size)
        copy_bytes(self.constants.cpu_address, frame.constants.cpu_address, frame.constants.size)
        copy_bytes(self.hud.cpu_address, source.hud.cpu_address, source.hud.size)
        copy_bytes(self.voxels.cpu_address, source.voxels.cpu_address, source.voxel_bytes_used)
        if source.csm_item_count > 0 and source.shadow_map.cpu_address != 0 and self.shadow_map.cpu_address != 0:
            var src_slice = source.shadow_map.cpu_address + CSM_DEPTH_BYTES + source.last_submitted * CSM_FRAME_SLICE_BYTES
            var dst_slice = self.shadow_map.cpu_address + CSM_DEPTH_BYTES + self.current * CSM_FRAME_SLICE_BYTES
            copy_bytes(dst_slice, src_slice, source.csm_record_count * CSM_RECORD_DWORDS * 4)
            copy_bytes(dst_slice + CSM_ITEMS_REL_BYTES, src_slice + CSM_ITEMS_REL_BYTES, source.csm_item_count * 8)
            self.csm_record_count = source.csm_record_count
            self.csm_item_count = source.csm_item_count
        var new_texture_bytes = source.texture_used - self.mirrored_texture_bytes
        if new_texture_bytes > 0:
            copy_bytes(self.textures.cpu_address + self.mirrored_texture_bytes, source.textures.cpu_address + self.mirrored_texture_bytes, new_texture_bytes)
            self.mirrored_texture_bytes = source.texture_used
        var settings = source.settings.copy()
        var started = get_time()
        if not self.record_vulkan_frame(settings.clear_color, settings.bloom, settings.bloom_width, settings.bloom_height, settings.bloom_strength, settings.smooth):
            return False
        self.finish_submit(started, started, self.profiler.now(), False)
        return True

    def export_shown(mut self, index: Int) -> Int:
        """A frame's finished image as a dmabuf, or -1."""
        if self.backend == BACKEND_VULKAN:
            return self.vk.export_buffer(self.sets[index].shown)
        return self.gpu.export_buffer(self.sets[index].shown)

    def copy_image(self, source: Int, mut pixels: List[UInt32]):
        for y in range(self.height):
            copy_bytes(Int(pixels.unsafe_ptr()) + y * self.width * 4, source + y * self.row_pixels * 4, self.width * 4)

    def forget_backdrop(mut self):
        self.backdrop_wanted = False
        self.backdrop_ready = False
        self.backdrop_fence = 0
