"""GPU occupancy mip-chain memory layout for Flint's compute rasterizer.

Provides the 4-level bitmask header layout expected by `flint.render.gpu_raster`
without requiring any voxel-world dependency when rendering standard 2D/3D scenes.
"""

comptime CHUNK_SIZE: Int = 32
comptime CHUNK_BRICKS_AXIS: Int = 4

comptime OCC_HEADER_WORDS: Int = 16
comptime OCC_L0_WORDS: Int = 16
comptime OCC_L1_WORDS: Int = 2
comptime OCC_L2_WORDS: Int = 2
comptime OCC_MAX_VOXELS: Int = 128 * 1024 * 1024
comptime OCC_MAX_BRICKS: Int = OCC_MAX_VOXELS // 512
comptime OCC_MAX_CHUNKS: Int = OCC_MAX_VOXELS // (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
comptime OCC_BYTES: Int = (
    OCC_HEADER_WORDS
    + OCC_MAX_BRICKS * (OCC_L0_WORDS + OCC_L1_WORDS)
    + OCC_MAX_CHUNKS * OCC_L2_WORDS
    + OCC_MAX_CHUNKS // 32
    + 64
) * 4


struct OccupancyLayout(Copyable, ImplicitlyCopyable, Movable):
    var bricks_x: Int
    var bricks_y: Int
    var bricks_z: Int
    var chunks_x: Int
    var chunks_y: Int
    var chunks_z: Int
    var l0: Int
    var l1: Int
    var l2: Int
    var l3: Int
    var l3_words: Int
    var total_words: Int

    def __init__(out self, width: Int = 0, height: Int = 0, depth: Int = 0):
        self.chunks_x = width // CHUNK_SIZE
        self.chunks_y = height // CHUNK_SIZE
        self.chunks_z = depth // CHUNK_SIZE
        self.bricks_x = self.chunks_x * CHUNK_BRICKS_AXIS
        self.bricks_y = self.chunks_y * CHUNK_BRICKS_AXIS
        self.bricks_z = self.chunks_z * CHUNK_BRICKS_AXIS
        var bricks = self.bricks_x * self.bricks_y * self.bricks_z
        var chunks = self.chunks_x * self.chunks_y * self.chunks_z
        self.l0 = OCC_HEADER_WORDS
        self.l1 = self.l0 + bricks * OCC_L0_WORDS
        self.l2 = self.l1 + bricks * OCC_L1_WORDS
        self.l3 = self.l2 + chunks * OCC_L2_WORDS
        self.l3_words = (chunks + 31) // 32
        self.total_words = self.l3 + self.l3_words

    def __eq__(self, other: Self) -> Bool:
        return (
            self.bricks_x == other.bricks_x
            and self.bricks_y == other.bricks_y
            and self.bricks_z == other.bricks_z
        )

    def bytes(self) -> Int:
        return self.total_words * 4

    def brick_index(self, bx: Int, by: Int, bz: Int) -> Int:
        return (by * self.bricks_z + bz) * self.bricks_x + bx

    def chunk_index(self, cx: Int, cy: Int, cz: Int) -> Int:
        return (cy * self.chunks_z + cz) * self.chunks_x + cx


@always_inline
def _put32(base: Int, word: Int, value: UInt32):
    Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=base + word * 4)[unsafe_offset=0] = value


def write_occupancy_header(base: Int, layout: OccupancyLayout):
    var values = [
        layout.bricks_x, layout.bricks_y, layout.bricks_z,
        layout.chunks_x, layout.chunks_y, layout.chunks_z,
        layout.l0, layout.l1, layout.l2, layout.l3, layout.total_words,
    ]
    for index in range(OCC_HEADER_WORDS):
        _put32(base, index, UInt32(values[index]) if index < len(values) else UInt32(0))
