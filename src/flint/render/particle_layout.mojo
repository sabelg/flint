"""What the GPU particle buffers hold (package 2.7): the records the CPU writes
and the kernels (render/particle_kernels.mojo) read, in 32-bit words.

Buffers, bound in this order to every particle kernel:

- PB_PARTICLES: PARTICLE_WORDS per slot.
- PB_STATE: STATE_HEADER_WORDS counters (below), then the dead list, then the
  two alive lists, each `capacity` words.
- PB_TABLE: the emitter records (EMITTER_WORDS each, MAX_EMITTERS), the ribbon
  table (RIBBON_WORDS per ribbon), then the flipbook atlas (bytes).
- PB_FRAME: FRAME_SETS regions of FRAME_REGION_WORDS, one per frame in flight:
  a header (FH_*), the spawn requests (REQUEST_WORDS each), then the spawn map
  (one request index per spawned particle).
- PB_FIELD: the per-brick distance field (world/brick_field.mojo).
- PB_WIND: the 2.12 wind mirror (render/gpu_layout.mojo).
- PB_ACCUM: the three accumulation levels (ACCUM_WORDS per pixel at half,
  quarter and eighth resolution), then the packed half-resolution image
  (PACKED_WORDS per pixel).
- PB_TARGET: the frame's target (render/gpu_raster.mojo): colour, bloom source,
  smoothed colour, inverse depth, scratch, each `plane` words apart.
"""

comptime PB_PARTICLES: Int = 0
comptime PB_STATE: Int = 1
comptime PB_TABLE: Int = 2
comptime PB_FRAME: Int = 3
comptime PB_FIELD: Int = 4
comptime PB_WIND: Int = 5
comptime PB_ACCUM: Int = 6
comptime PB_TARGET: Int = 7
comptime PARTICLE_BUFFERS: Int = 8

# A particle.
comptime PARTICLE_WORDS: Int = 16
comptime P_X: Int = 0
comptime P_Y: Int = 1
comptime P_Z: Int = 2
comptime P_VX: Int = 3
comptime P_VY: Int = 4
comptime P_VZ: Int = 5
comptime P_TIME: Int = 6      # simulated up to (s)
comptime P_DEATH: Int = 7     # dies at (s)
comptime P_BORN: Int = 8      # born at (s)
comptime P_EMITTER: Int = 9   # emitter id in the low 16 bits, PF_* flags above
comptime P_COLOR: Int = 10    # rgb8 (red low), alpha byte unused
comptime P_SIZE: Int = 11     # m at birth
comptime P_SEED: Int = 12
comptime P_SERIAL: Int = 13
comptime P_LINK: Int = 14     # ribbons: the previous point's slot
comptime P_LINK_SERIAL: Int = 15
comptime PF_RESTING: UInt32 = 0x10000
comptime PF_TOUCHED: UInt32 = 0x20000  # has touched terrain at least once

# State counters.
comptime STATE_HEADER_WORDS: Int = 64
comptime S_DEAD: Int = 0         # dead-list length
comptime S_ALIVE: Int = 1        # alive particles going into this frame
comptime S_ALIVE_IN: Int = 2     # after this frame's spawns
comptime S_ALIVE_OUT: Int = 3    # after this frame's update
comptime S_SERIAL: Int = 4       # serial of the next spawn
comptime S_UPDATE_ARGS: Int = 8  # x, y, z groups for the update
comptime S_SPLAT_ARGS: Int = 12  # x, y, z groups for the splat
comptime S_SPAWNED: Int = 16     # totals since the start
comptime S_DROPPED: Int = 17     # spawns the full pool refused
comptime S_DIED: Int = 18
comptime S_CONTACTS: Int = 19    # terrain contacts (all frames)
comptime S_RESTING: Int = 20     # particles resting after the last update
comptime S_SPLATS: Int = 21      # pixels written by the last splat
comptime S_CAPACITY: Int = 22
comptime S_PROJECT_ARGS: Int = 24  # x, y, z groups for the projection

# An emitter record.
comptime MAX_EMITTERS: Int = 256
comptime EMITTER_WORDS: Int = 32
comptime E_KIND: Int = 0
comptime E_FLAGS: Int = 1       # EF_*
comptime E_LIFE_LOW: Int = 2
comptime E_LIFE_HIGH: Int = 3
comptime E_SPEED_LOW: Int = 4
comptime E_SPEED_HIGH: Int = 5
comptime E_SPREAD: Int = 6
comptime E_UP: Int = 7
comptime E_GRAVITY: Int = 8
comptime E_DRAG: Int = 9
comptime E_SIZE_LOW: Int = 10
comptime E_SIZE_HIGH: Int = 11
comptime E_GROW: Int = 12
comptime E_COLOR: Int = 13       # rgb8 of the row (EMIT_COLOR_ROW)
comptime E_ALPHA: Int = 14
comptime E_END_ALPHA: Int = 15
comptime E_EMISSIVE: Int = 16
comptime E_RESTITUTION: Int = 17
comptime E_FRICTION: Int = 18
comptime E_WIND: Int = 19
comptime E_STRETCH: Int = 20
comptime E_SOFT: Int = 21        # 1 / soft distance (m), 0: hard
comptime E_FLIP_FIRST: Int = 22  # float frame
comptime E_FLIP_FRAMES: Int = 23 # float
comptime E_FLIP_RATE: Int = 24
comptime E_RING_WIDTH: Int = 25
comptime E_DISTORTION: Int = 26
comptime E_JITTER: Int = 27
comptime E_COLLIDE: Int = 28     # COLLIDE_* as a word
comptime EF_ROW_COLOR: UInt32 = 1
comptime EF_ADDITIVE: UInt32 = 2

comptime RIBBON_WORDS: Int = 2   # last slot, its serial
comptime RIBBON_SLOTS: Int = 64
comptime TABLE_RIBBON_WORD: Int = MAX_EMITTERS * EMITTER_WORDS
comptime TABLE_ATLAS_WORD: Int = TABLE_RIBBON_WORD + RIBBON_SLOTS * RIBBON_WORDS
comptime ATLAS_FRAME: Int = 16   # texels a side
comptime ATLAS_FRAMES: Int = 16
comptime ATLAS_BYTES: Int = ATLAS_FRAME * ATLAS_FRAME * ATLAS_FRAMES
comptime TABLE_WORDS: Int = TABLE_ATLAS_WORD + ATLAS_BYTES // 4

# A spawn request.
comptime MAX_REQUESTS: Int = 512
comptime MAX_SPAWNS: Int = 16384  # particles spawned in one frame, at most
comptime REQUEST_WORDS: Int = 20
comptime R_EMITTER: Int = 0
comptime R_FIRST: Int = 1        # its first index in the frame's spawn order
comptime R_COUNT: Int = 2
comptime R_SEED: Int = 3
comptime R_X: Int = 4
comptime R_VX: Int = 7
comptime R_DX: Int = 10          # spread axis, unit
comptime R_COLOR: Int = 13       # rgb8, or 0xFFFFFFFF for the row's
comptime R_SIZE: Int = 14        # size scale
comptime R_TIME: Int = 15
comptime R_RIBBON: Int = 16      # ribbon slot, or 0xFFFFFFFF
comptime R_RIBBON_FIRST: Int = 17  # 1: its first point this frame links to the ribbon table
comptime R_RIBBON_LAST: Int = 18   # 1: its last point this frame goes into the ribbon table
comptime NO_WORD: UInt32 = 0xFFFFFFFF

# A frame region's header.
comptime FRAME_HEADER_WORDS: Int = 64
comptime FH_TIME: Int = 0
comptime FH_SPAWNS: Int = 1
comptime FH_REQUESTS: Int = 2
comptime FH_PARITY: Int = 3       # 0: alive list A in, B out; 1: the other way
comptime FH_WIDTH: Int = 4
comptime FH_HEIGHT: Int = 5
comptime FH_ROW: Int = 6          # pixels per target row
comptime FH_PLANE: Int = 7        # words between target planes
comptime FH_CAMERA: Int = 8       # x, y, z
comptime FH_RIGHT: Int = 11
comptime FH_UP: Int = 14
comptime FH_FORWARD: Int = 17
comptime FH_FOCAL: Int = 20
comptime FH_HALF_W: Int = 21      # floats
comptime FH_HALF_H: Int = 22
comptime FH_WIND_ORIGIN: Int = 23 # x, y, z (a cell centre)
comptime FH_DISTORT: Int = 26     # 1: read the scene from the scratch plane, displaced
comptime FH_CAPACITY: Int = 27
comptime FH_NEAR_FADE: Int = 28   # m: particles closer than this fade out
comptime FH_WIND_ON: Int = 29     # 1: sample the wind mirror
comptime FH_LEVEL_W: Int = 30     # 30-32: level widths (half, quarter, eighth)
comptime FH_LEVEL_H: Int = 33     # 33-35
comptime FH_LEVEL_AT: Int = 36    # 36-38: first accumulation word of each level
comptime FH_PACKED_AT: Int = 39
comptime FH_FIELD_ON: Int = 40    # 1: the distance field is bound
comptime FH_STATS: Int = 41       # 1: count contacts and splatted pixels
comptime FH_BLOOM: Int = 42       # float: how much emissive goes into the bloom source
comptime FH_TILES_AT: Int = 43    # first accumulation word of the tile flags: two parities, then the grown flags
comptime FH_TILES_X: Int = 44     # 32 x 32 full-resolution tiles across
comptime FH_TILES_Y: Int = 45
comptime FH_TILE_COUNT: Int = 46
comptime FH_RECORDS_AT: Int = 47  # first accumulation word of the splat records
comptime FRAME_REQUEST_WORD: Int = FRAME_HEADER_WORDS
comptime FRAME_MAP_WORD: Int = FRAME_REQUEST_WORD + MAX_REQUESTS * REQUEST_WORDS
comptime FRAME_REGION_WORDS: Int = FRAME_MAP_WORD + MAX_SPAWNS

# Accumulation (integers, ACCUM_ONE per unit of weight).
comptime ACCUM_WORDS: Int = 8    # premultiplied r, g, b, coverage; additive r, g, b; distortion (dx | dy << 16)
comptime ACCUM_ONE: Float32 = 1024.0
comptime ACCUM_LEVELS: Int = 3
comptime PACKED_WORDS: Int = 3   # rgba8 premultiplied over, rgb8 additive, distortion int16 x2 (1/64 px)
comptime DISTORT_UNIT: Float32 = 64.0
comptime RECORD_WORDS: Int = 20  # a particle's splat record (render/particle_kernels.mojo project_kernel)
comptime TILE: Int = 32          # full-resolution pixels a side of a tile flag
comptime SPLAT_BLOCK: Int = 8    # level pixels a side per splat invocation
comptime SPLAT_BLOCKS_AXIS: Int = 8
comptime SPLAT_BLOCKS: Int = SPLAT_BLOCKS_AXIS * SPLAT_BLOCKS_AXIS  # so a sprite spans at most 64 level pixels
comptime SPLAT_SPAN: Float32 = 64.0

# Target planes, after the raster's three (render/gpu_raster.mojo).
comptime PLANE_COLOR: Int = 0
comptime PLANE_BLOOM: Int = 1
comptime PLANE_SMOOTH: Int = 2
comptime PLANE_DEPTH: Int = 3
comptime PLANE_SCRATCH: Int = 4
comptime PLANE_HDR_R: Int = 5
comptime PLANE_HDR_G: Int = 6
comptime PLANE_HDR_B: Int = 7
comptime PLANE_GBUF: Int = 8
comptime PLANE_HISTORY_A: Int = 9
comptime PLANE_HISTORY_B: Int = 10
comptime PLANE_VELOCITY: Int = 11
comptime TARGET_PLANES: Int = 12

# Simulation steps.
comptime MAX_STEP: Float32 = 0.05   # seconds a substep may cover
comptime SUBSTEPS: Int = 2          # per frame, unrolled in the kernel
comptime MAX_FRAME_STEP: Float32 = 0.1  # longer frames are simulated as this long


def tile_count(width: Int, height: Int) -> Int:
    return ((width + TILE - 1) // TILE) * ((height + TILE - 1) // TILE)


def accum_bytes(width: Int, height: Int, capacity: Int) -> Int:
    """The levels, the packed image, three tile flag sets, the splat records."""
    var total = 0
    for level in range(ACCUM_LEVELS):
        var shift = level + 1
        total += ((width + (1 << shift) - 1) >> shift) * ((height + (1 << shift) - 1) >> shift) * ACCUM_WORDS
    total += ((width + 1) >> 1) * ((height + 1) >> 1) * PACKED_WORDS
    total += 3 * tile_count(width, height)
    total += capacity * RECORD_WORDS
    return total * 4
