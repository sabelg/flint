"""The memory both GPU backends agree on: one frame description, two sets of shaders.

The CPU fills these buffers the same way for either backend, so nothing here may
hold an address. Offsets count from the start of their own buffer.
"""

comptime TILE_SIZE: Int = 32
comptime RECORD_BYTES: Int = 256
comptime RECORD_FLOATS: Int = 64
comptime GROUP_WIDTH: Int = 8
comptime GROUP_HEIGHT: Int = 4
comptime TEXTURED_FLAG: UInt32 = 256
comptime PER_PIXEL_FLAG: UInt32 = 512
# Package 3.2: an occluder triangle, screen-doored around Maroon (render/dither.mojo).
comptime DITHER_FLAG: UInt32 = 1024
comptime PBR_FLAG: UInt32 = 2048
comptime NO_DEPTH_WRITE_FLAG: UInt32 = 4096
comptime ADDITIVE_FLAG: UInt32 = 8192
comptime SOFT_PARTICLE_FLAG: UInt32 = 16384
comptime ALPHA_ERODE_FLAG: UInt32 = 32768
comptime BLACKBODY_FLAG: UInt32 = 65536
comptime REFRACT_FLAG: UInt32 = 131072
comptime WATER_WAVE_FLAG: UInt32 = 262144
comptime MIPMAPPED_FLAG: UInt32 = 524288
comptime OBJECT_VELOCITY_FLAG: UInt32 = 1048576
comptime NORM_ORM_FLAG: UInt32 = 2097152
# Frame-wide effect switches, read by the raster and compose shaders from CONSTANT_EFFECTS.
comptime EFFECT_BILINEAR: UInt32 = 1
comptime EFFECT_SHADOWS: UInt32 = 2
comptime EFFECT_CSM: UInt32 = 4
comptime EFFECT_SSR: UInt32 = 8
comptime EFFECT_VOLUMETRIC: UInt32 = 16
comptime EFFECT_TAA: UInt32 = 32
comptime EFFECT_GTAO_CONTACT: UInt32 = 64
comptime EFFECT_AUTO_EXPOSURE: UInt32 = 128
comptime EFFECT_DOF: UInt32 = 256
comptime EFFECT_PHYSICAL_SKY: UInt32 = 512
comptime EFFECT_FROXEL_VOL: UInt32 = 1024
comptime EFFECT_HIZ_CULL: UInt32 = 2048
comptime EFFECT_VOL_CLOUDS: UInt32 = 4096
comptime EFFECT_SSGI_BENT: UInt32 = 8192
comptime EFFECT_CHROMATIC_REFRACT: UInt32 = 16384
comptime EFFECT_RADIANCE_GI: UInt32 = 32768
comptime EFFECT_DYRL_ANIME: UInt32 = 65536
comptime EFFECT_SDF_SHADOWS: UInt32 = 131072
comptime EFFECT_WEATHER_PRECIP: UInt32 = 262144
comptime EFFECT_DYN_REFLECT: UInt32 = 524288
comptime CONSTANT_SATURATION: Int = 768
comptime CONSTANT_BLOOM_DISTANCE: Int = 772
# Package 3.2, five floats: the dither disc's centre x, y, scale, offset, and
# the inverse depth that pixels must exceed to dither (render/dither.mojo).
comptime CONSTANT_DITHER: Int = 776
comptime CONSTANT_EFFECTS: Int = 800
comptime CONSTANT_CAMERA: Int = 816
comptime CONSTANT_KEY: Int = 880
comptime CONSTANT_WORLD: Int = 912
comptime CONSTANT_ENV_SKY: Int = 944
comptime CONSTANT_ENV_GROUND: Int = 960
comptime CONSTANT_ENV_HORIZON: Int = 976
comptime CONSTANT_LIGHTS: Int = 1024
comptime CONSTANT_SPHERES: Int = 2048
comptime CONSTANT_CSM: Int = 2176
comptime CONSTANT_PREV_CAMERA: Int = 2304
comptime CONSTANT_VOL_POST: Int = 2400
comptime CONSTANT_V7_POST: Int = 2464
comptime CONSTANT_V8_ATMOS: Int = 2512
comptime CONSTANT_PROBES: Int = 2576
comptime MAX_IRRADIANCE_PROBES: Int = 32
comptime CONSTANT_CLUSTER_LIGHTS: Int = 4096
comptime CONSTANT_DECALS: Int = 10240
comptime MAX_SURFACE_DECALS: Int = 32
comptime CONSTANT_V9_CLOUDS: Int = 12288
comptime CONSTANT_V10_POST: Int = 12416
comptime CONSTANT_RADIANCE_CASCADES: Int = 12544
comptime MAX_RADIANCE_CASCADES: Int = 32
comptime CONSTANT_V11_POST: Int = 13568
comptime CONSTANT_SDF_OCCLUDERS: Int = 13696
comptime MAX_SDF_OCCLUDERS: Int = 16
comptime CONSTANT_BYTES: Int = 16384
comptime MAX_CLUSTER_LIGHTS: Int = 128
comptime CLUSTER_Z_SLICES: Int = 4
comptime CLUSTER_WORDS_PER_SLICE: Int = 4
comptime CLUSTER_WORDS_PER_TILE: Int = CLUSTER_Z_SLICES * CLUSTER_WORDS_PER_SLICE
comptime MAX_SHADOW_SPHERES: Int = 8
# Shadow rays start fine and turn coarse: 8 units in fine steps, then 16 in full-voxel steps.
comptime SHADOW_FINE_STEPS: Int = 16
comptime SHADOW_COARSE_STEPS: Int = 16
comptime TILE_ENTRY_DWORDS: Int = 4
comptime BLOOM_SAMPLES_X: Int = 8
comptime BLOOM_SAMPLES_Y: Int = 6
comptime BLOOM_SOURCE_OFFSET: Int = 0
comptime BLOOM_BLURRED_OFFSET: Int = 1 * 1024 * 1024
comptime BLOOM_MIP0_CRISP_OFFSET: Int = 2 * 1024 * 1024
comptime BLOOM_MIP1_OFFSET: Int = 3 * 1024 * 1024
comptime BLOOM_MIP2_OFFSET: Int = 3 * 1024 * 1024 + 512 * 1024
comptime BLOOM_MIP1_UP_OFFSET: Int = 3 * 1024 * 1024 + 768 * 1024
comptime BLOOM_FINAL_OFFSET: Int = 4 * 1024 * 1024
comptime BLOOM_HIST_OFFSET: Int = 5 * 1024 * 1024
comptime SKY_LUT_OFFSET: Int = 5 * 1024 * 1024 + 16 * 1024
comptime SKY_LUT_W: Int = 128
comptime SKY_LUT_H: Int = 64
comptime FROXEL_VOL_OFFSET: Int = 5 * 1024 * 1024 + 160 * 1024
comptime FROXEL_X: Int = 64
comptime FROXEL_Y: Int = 36
comptime FROXEL_Z: Int = 16
comptime FROXEL_WORDS: Int = FROXEL_X * FROXEL_Y * FROXEL_Z * 4
comptime FROXEL_SLICE_BYTES: Int = FROXEL_WORDS * 4
comptime FROXEL_INTEGRATED_OFFSET: Int = FROXEL_VOL_OFFSET + 2 * FROXEL_SLICE_BYTES
comptime BLOOM_BYTES: Int = 8 * 1024 * 1024
comptime HIZ_BACKDROP_EXTRA_BYTES: Int = 16 * 1024 * 1024
comptime HIZ_MAX_TILES_X: Int = 256
comptime HIZ_MAX_TILES_Y: Int = 144
comptime HIZ_SLICE_WORDS: Int = HIZ_MAX_TILES_X * HIZ_MAX_TILES_Y
comptime GPU_SKIN_REL_BYTES: Int = 512 * 1024
comptime GPU_SKIN_MAX_JOINTS: Int = 64
comptime GPU_SKIN_MAX_VERTS: Int = 7000
comptime GPU_PARTICLE_REL_BYTES: Int = 1024 * 1024
comptime GPU_PARTICLE_MAX: Int = 65536
comptime GPU_PARTICLE_STRIDE_WORDS: Int = 8
comptime GPU_CULL_REL_BYTES: Int = 5 * 1024 * 1024
comptime GPU_CULL_MAX_INSTANCES: Int = 8192
comptime GPU_CULL_IN_STRIDE_WORDS: Int = 8
comptime TSR_SCRATCH_REL_BYTES: Int = 6 * 1024 * 1024

# 2-Cascade GPU Shadow Map (1024x1024x2 depth + 3 frame slices of records & tile items).
comptime CSM_RES: Int = 1024
comptime CSM_TILE_SIZE: Int = 16
comptime CSM_TILES_AXIS: Int = CSM_RES // CSM_TILE_SIZE
comptime CSM_CASCADE_WORDS: Int = CSM_RES * CSM_RES
comptime CSM_DEPTH_WORDS: Int = 2 * CSM_CASCADE_WORDS
comptime CSM_DEPTH_BYTES: Int = CSM_DEPTH_WORDS * 4
comptime CSM_RECORD_DWORDS: Int = 12
comptime CSM_MAX_RECORDS: Int = 65536
comptime CSM_RECORDS_REL_BYTES: Int = 0
comptime CSM_ITEMS_REL_BYTES: Int = CSM_MAX_RECORDS * CSM_RECORD_DWORDS * 4
comptime CSM_MAX_ITEMS: Int = 262144
comptime CSM_FRAME_SLICE_BYTES: Int = CSM_ITEMS_REL_BYTES + CSM_MAX_ITEMS * 8
comptime CSM_BYTES: Int = CSM_DEPTH_BYTES + 3 * CSM_FRAME_SLICE_BYTES

# Record layout in dwords. Planes are A, B, C for value = A*x + B*y + C.
comptime RECORD_WEIGHT_A: Int = 0
comptime RECORD_WEIGHT_B: Int = 3
comptime RECORD_INVERSE_Z: Int = 6
comptime RECORD_U: Int = 9
comptime RECORD_V: Int = 12
comptime RECORD_TINT: Int = 15
comptime RECORD_LIGHT: Int = 16
comptime RECORD_FOG: Int = 25
comptime RECORD_EMISSIVE: Int = 28
comptime RECORD_FLAGS: Int = 31
comptime RECORD_TEXTURE: Int = 32
comptime RECORD_TEXTURE_FLOAT_WIDTH: Int = 33
comptime RECORD_TEXTURE_FLOAT_HEIGHT: Int = 34
comptime RECORD_TEXTURE_WIDTH: Int = 35
comptime RECORD_TEXTURE_MAX_X: Int = 36
comptime RECORD_TEXTURE_MAX_Y: Int = 37
comptime RECORD_ROUGHNESS: Int = 38
comptime RECORD_METALLIC: Int = 39
comptime RECORD_NORMAL: Int = 40
comptime RECORD_AMBIENT_OCCLUSION: Int = 49
comptime RECORD_SOFT_INV_DIST: Int = 52
comptime RECORD_ERODE_THRESHOLD: Int = 53
comptime RECORD_BLACKBODY_HEAT: Int = 54
comptime RECORD_REFRACT_STRENGTH: Int = 55
comptime RECORD_VEL_X: Int = 56
comptime RECORD_VEL_Y: Int = 59
comptime RECORD_ANISO: Int = 62
comptime RECORD_NORM_ORM_TEX: Int = 63

# Vulkan bindings, all storage buffers in one descriptor set.
comptime BIND_RECORDS: Int = 0
comptime BIND_LISTS: Int = 1
comptime BIND_TILES: Int = 2
comptime BIND_CONSTANTS: Int = 3
comptime BIND_TEXTURES: Int = 4
comptime BIND_TARGET: Int = 5
comptime BIND_VOXELS: Int = 6
comptime BIND_CELLS: Int = 7
comptime BIND_HUD: Int = 8
comptime BIND_SHOWN: Int = 9
comptime BIND_BACKDROP: Int = 10
comptime BIND_SHADOW_MAP: Int = 11
comptime BIND_COUNT: Int = 12

comptime SHADER_RASTER: Int = 0
comptime SHADER_BLOOM_REDUCE: Int = 1
comptime SHADER_BLUR_HORIZONTAL: Int = 2
comptime SHADER_BLUR_VERTICAL: Int = 3
comptime SHADER_COMPOSE: Int = 4
comptime SHADER_SMOOTH: Int = 5
comptime SHADER_COUNT: Int = 6
comptime SHADER_CAPTURE: Int = 6  # Vulkan only: compose, into the backdrop, with no HUD
comptime SHADER_KAWASE_DOWN: Int = 7  # Vulkan multi-scale Dual-Kawase pyramid downsample
comptime SHADER_KAWASE_UP: Int = 8    # Vulkan multi-scale Dual-Kawase tent upsample + pack
comptime SHADER_CSM_CLEAR: Int = 9    # Vulkan 2-cascade shadow map clear
comptime SHADER_CSM_RASTER: Int = 10  # Vulkan 2-cascade shadow map tile rasterizer
comptime SHADER_TAA_RESOLVE: Int = 11 # Vulkan TAA + velocity motion blur + HUD output
comptime SHADER_TAA_CAPTURE: Int = 12 # Vulkan TAA + velocity motion blur backdrop capture
comptime SHADER_HISTOGRAM: Int = 13   # Vulkan 64-bin HDR log-luminance histogram
comptime SHADER_AUTO_EXPOSURE: Int = 14 # Vulkan temporal histogram auto-exposure solver
comptime SHADER_SKY_LUT: Int = 15       # Vulkan Hillaire physical sky & orbit LUT (128x64)
comptime SHADER_FROXEL_INTEGRATE: Int = 16 # Vulkan temporally reprojected 3D froxel volumetrics
comptime SHADER_HIZ_PYRAMID: Int = 17   # Vulkan 16x16 coarse Hi-Z depth occlusion pyramid
comptime SHADER_GPU_SKIN: Int = 18      # Vulkan compute 4-bone linear-blend skeletal skinning
comptime SHADER_GPU_PARTICLES: Int = 19 # Vulkan compute 65,536-particle ballistic + depth-bounce solver
comptime SHADER_TSR_UPSCALE: Int = 20   # Vulkan V10 Catmull-Rom 5-tap + YCoCg Temporal Super-Resolution upscaler
comptime SHADER_GPU_CULL: Int = 21      # Vulkan V10 compute frustum + 16x16 Hi-Z indirect instance culler
comptime VULKAN_SHADER_COUNT: Int = 22
# The wind mirror (package 2.12, physics/wind_mirror.mojo): one buffer, 16 x 8 x 16
# cells 4 m apart from WIND_MIRROR_ORIGIN (a cell centre, in the frame's
# constants of the pass that reads it), x fastest, then z, then y; each cell
# four int16: velocity x, y, z in 1/256 m/s and speed. Sample trilinear.
comptime WIND_MIRROR_X: Int = 16
comptime WIND_MIRROR_Y: Int = 8
comptime WIND_MIRROR_Z: Int = 16
comptime WIND_MIRROR_CELL_BYTES: Int = 8
comptime WIND_MIRROR_BYTES: Int = WIND_MIRROR_X * WIND_MIRROR_Y * WIND_MIRROR_Z * WIND_MIRROR_CELL_BYTES
