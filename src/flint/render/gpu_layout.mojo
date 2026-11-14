"""The memory both GPU backends agree on: one frame description, two sets of shaders.

The CPU fills these buffers the same way for either backend, so nothing here may
hold an address. Offsets count from the start of their own buffer.
"""

comptime TILE_SIZE: Int = 32
comptime RECORD_BYTES: Int = 224
comptime RECORD_FLOATS: Int = 56
comptime GROUP_WIDTH: Int = 8
comptime GROUP_HEIGHT: Int = 4
comptime TEXTURED_FLAG: UInt32 = 256
comptime PER_PIXEL_FLAG: UInt32 = 512
# Package 3.2: an occluder triangle, screen-doored around Maroon (render/dither.mojo).
comptime DITHER_FLAG: UInt32 = 1024
# Frame-wide effect switches, read by the raster shader from CONSTANT_EFFECTS.
comptime EFFECT_BILINEAR: UInt32 = 1
comptime EFFECT_SHADOWS: UInt32 = 2
comptime CONSTANT_SATURATION: Int = 768
comptime CONSTANT_BLOOM_DISTANCE: Int = 772
# Package 3.2, five floats: the dither disc's centre x, y, scale, offset, and
# the inverse depth that pixels must exceed to dither (render/dither.mojo).
comptime CONSTANT_DITHER: Int = 776
comptime CONSTANT_EFFECTS: Int = 800
comptime CONSTANT_CAMERA: Int = 816
comptime CONSTANT_KEY: Int = 880
comptime CONSTANT_WORLD: Int = 912
comptime CONSTANT_LIGHTS: Int = 1024
comptime CONSTANT_SPHERES: Int = 2048
comptime MAX_SHADOW_SPHERES: Int = 8
# Shadow rays start fine and turn coarse: 8 units in fine steps, then 16 in full-voxel steps.
comptime SHADOW_FINE_STEPS: Int = 16
comptime SHADOW_COARSE_STEPS: Int = 16
comptime TILE_ENTRY_DWORDS: Int = 4
comptime BLOOM_SAMPLES_X: Int = 8
comptime BLOOM_SAMPLES_Y: Int = 6
comptime BLOOM_SOURCE_OFFSET: Int = 0
comptime BLOOM_BLURRED_OFFSET: Int = 512 * 1024
comptime BLOOM_FINAL_OFFSET: Int = 1024 * 1024
comptime BLOOM_BYTES: Int = 2 * 1024 * 1024

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
comptime RECORD_NORMAL: Int = 40
comptime RECORD_AMBIENT_OCCLUSION: Int = 49

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
comptime BIND_COUNT: Int = 11

comptime SHADER_RASTER: Int = 0
comptime SHADER_BLOOM_REDUCE: Int = 1
comptime SHADER_BLUR_HORIZONTAL: Int = 2
comptime SHADER_BLUR_VERTICAL: Int = 3
comptime SHADER_COMPOSE: Int = 4
comptime SHADER_SMOOTH: Int = 5
comptime SHADER_COUNT: Int = 6
comptime SHADER_CAPTURE: Int = 6  # Vulkan only: compose, into the backdrop, with no HUD
comptime VULKAN_SHADER_COUNT: Int = 7
# The wind mirror (package 2.12, physics/wind_mirror.mojo): one buffer, 16 x 8 x 16
# cells 4 m apart from WIND_MIRROR_ORIGIN (a cell centre, in the frame's
# constants of the pass that reads it), x fastest, then z, then y; each cell
# four int16: velocity x, y, z in 1/256 m/s and speed. Sample trilinear.
comptime WIND_MIRROR_X: Int = 16
comptime WIND_MIRROR_Y: Int = 8
comptime WIND_MIRROR_Z: Int = 16
comptime WIND_MIRROR_CELL_BYTES: Int = 8
comptime WIND_MIRROR_BYTES: Int = WIND_MIRROR_X * WIND_MIRROR_Y * WIND_MIRROR_Z * WIND_MIRROR_CELL_BYTES
