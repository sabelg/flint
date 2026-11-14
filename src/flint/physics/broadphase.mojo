"""Broadphase: a hash grid on the chunk grid, pairs found in parallel and
merged by per-job bitsets (package 2.11).

Cells are the terrain's 8^3 bricks (four to a chunk side), so a cell is the
same box the terrain mid-phase reads from the brick table. Each body's box,
grown by how far it can move this tick, goes into every cell it covers (a body
covering more than MAX_CELLS_PER_BODY cells is *wide* and is tested against
every body instead). The table is open addressing over a fixed number of
buckets, filled in row order.

Pairs are found by row blocks on the job pool: the job for rows a..a+15 tests
each of its rows against the higher rows sharing a cell, and sets bit (a, b)
in its own bitset; nothing is shared while the jobs run. The bitsets are then
OR-ed together and read in bit order, so the pair list is sorted by (a, b),
has no duplicates (two bodies sharing several cells), and is the same at any
worker count. A pair is kept when at least one body is awake, their filters
let them touch, and their grown boxes overlap.
"""
from std.math import floor
from std.bit import count_trailing_zeros
from flint.core.math import Vec3
from flint.core.jobs import run_jobs
from flint.core.profiler import ZONE_BROADPHASE
from flint.physics.filter import CollisionFilter

comptime CELL_SIZE: Int = 8               # a brick
comptime HASH_BUCKETS: Int = 1024         # power of two
comptime MAX_CELLS_PER_BODY: Int = 64
comptime ROWS_PER_JOB: Int = 16


struct BroadphaseInput(Movable):
    """What the broadphase sees of each body row: its grown box, whether it
    takes part (active), whether it is awake, and its filter."""
    var lo: List[Vec3]
    var hi: List[Vec3]
    var live: List[Bool]
    var awake: List[Bool]
    var filters: List[CollisionFilter]

    def __init__(out self):
        self.lo = List[Vec3]()
        self.hi = List[Vec3]()
        self.live = List[Bool]()
        self.awake = List[Bool]()
        self.filters = List[CollisionFilter]()

    def add(mut self, lo: Vec3, hi: Vec3, live: Bool, awake: Bool, filter: CollisionFilter):
        self.lo.append(lo)
        self.hi.append(hi)
        self.live.append(live)
        self.awake.append(awake)
        self.filters.append(filter)

    def rows(self) -> Int:
        return len(self.lo)


@always_inline
def cell_key(x: Int, y: Int, z: Int) -> Int:
    # Offset so cells a little outside the world (negative) stay distinct.
    return ((y + 1024) << 22) | ((z + 1024) << 11) | (x + 1024)


@always_inline
def cell_coordinate(value: Float32) -> Int:
    if not (value > -1.0e6 and value < 1.0e6):
        return -1000000  # far outside or not a number: the body goes wide
    return Int(floor(value / Float32(CELL_SIZE)))


@always_inline
def bucket_of(key: Int) -> Int:
    var h = UInt64(key) * 0x9E3779B97F4A7C15
    return Int(h >> 54) & (HASH_BUCKETS - 1)


def boxes_overlap(alo: Vec3, ahi: Vec3, blo: Vec3, bhi: Vec3) -> Bool:
    return alo.x <= bhi.x and blo.x <= ahi.x and alo.y <= bhi.y and blo.y <= ahi.y and alo.z <= bhi.z and blo.z <= ahi.z


struct HashGrid(Movable):
    """Open addressing: bucket -> cell slot; each slot lists its rows."""
    var bucket_slot: List[Int]
    var slot_key: List[Int]
    var slot_rows: List[List[Int]]
    var wide: List[Int]
    var body_cells: List[List[Int]]  # per row: the slots it is in

    def __init__(out self, rows: Int):
        self.bucket_slot = List[Int](length=HASH_BUCKETS, fill=-1)
        self.slot_key = List[Int]()
        self.slot_rows = List[List[Int]]()
        self.wide = List[Int]()
        self.body_cells = List[List[Int]]()
        for _ in range(rows):
            self.body_cells.append(List[Int]())

    def slot_for(mut self, key: Int) -> Int:
        var bucket = bucket_of(key)
        for _ in range(HASH_BUCKETS):
            var slot = self.bucket_slot[bucket]
            if slot < 0:
                slot = len(self.slot_key)
                self.bucket_slot[bucket] = slot
                self.slot_key.append(key)
                self.slot_rows.append(List[Int]())
                return slot
            if self.slot_key[slot] == key:
                return slot
            bucket = (bucket + 1) & (HASH_BUCKETS - 1)
        return -1  # full: the caller treats the body as wide

    def insert(mut self, row: Int, lo: Vec3, hi: Vec3):
        var x0 = cell_coordinate(lo.x)
        var y0 = cell_coordinate(lo.y)
        var z0 = cell_coordinate(lo.z)
        var x1 = cell_coordinate(hi.x)
        var y1 = cell_coordinate(hi.y)
        var z1 = cell_coordinate(hi.z)
        var count = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
        if count > MAX_CELLS_PER_BODY or x0 < -1000 or y0 < -1000 or z0 < -1000 or x1 > 1000 or y1 > 1000 or z1 > 1000:
            self.wide.append(row)
            return
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                for x in range(x0, x1 + 1):
                    var slot = self.slot_for(cell_key(x, y, z))
                    if slot < 0:
                        self.wide.append(row)
                        return
                    self.slot_rows[slot].append(row)
                    self.body_cells[row].append(slot)


struct PairJobs:
    """What the pair jobs share (addresses); job j owns bitset j."""
    var input: Int
    var grid: Int
    var bitsets: Int
    var rows: Int

    def __init__(out self, input: Int, grid: Int, bitsets: Int, rows: Int):
        self.input = input
        self.grid = grid
        self.bitsets = bitsets
        self.rows = rows


@always_inline
def pair_wanted(input: BroadphaseInput, a: Int, b: Int) -> Bool:
    if not input.live[b] or not (input.awake[a] or input.awake[b]):
        return False
    if not input.filters[a].touches(input.filters[b]):
        return False
    return boxes_overlap(input.lo[a], input.hi[a], input.lo[b], input.hi[b])


def find_pairs_for_rows(input: BroadphaseInput, grid: HashGrid, first: Int, end: Int, rows: Int, mut bits: List[UInt64]):
    """Rows first..end-1 against every higher row they may touch."""
    for a in range(first, end):
        if not input.live[a]:
            continue
        for slot in grid.body_cells[a]:
            for b in grid.slot_rows[slot]:
                if b > a and pair_wanted(input, a, b):
                    var bit = a * rows + b
                    bits[bit >> 6] |= UInt64(1) << UInt64(bit & 63)
        # Wide bodies meet everyone; a wide row also checks every higher row.
        var a_wide = False
        for w in grid.wide:
            if w == a:
                a_wide = True
        if a_wide:
            for b in range(a + 1, rows):
                if pair_wanted(input, a, b):
                    var bit = a * rows + b
                    bits[bit >> 6] |= UInt64(1) << UInt64(bit & 63)
        else:
            for b in grid.wide:
                if b > a and pair_wanted(input, a, b):
                    var bit = a * rows + b
                    bits[bit >> 6] |= UInt64(1) << UInt64(bit & 63)


def run_pair_job(context: Int, job: Int):
    ref shared = Pointer[PairJobs, origin=MutAnyOrigin](unsafe_from_address=context)[unsafe_offset=0]
    ref input = Pointer[BroadphaseInput, origin=MutAnyOrigin](unsafe_from_address=shared.input)[unsafe_offset=0]
    ref grid = Pointer[HashGrid, origin=MutAnyOrigin](unsafe_from_address=shared.grid)[unsafe_offset=0]
    ref bitsets = Pointer[List[List[UInt64]], origin=MutAnyOrigin](unsafe_from_address=shared.bitsets)[unsafe_offset=0]
    var first = job * ROWS_PER_JOB
    find_pairs_for_rows(input, grid, first, min(shared.rows, first + ROWS_PER_JOB), shared.rows, bitsets[job])


struct PairResult(Movable):
    var pairs: List[Int]   # a * rows + b, a < b, sorted
    var cells: Int         # grid cells in use
    var wide: Int          # bodies tested against everyone
    var jobs: Int

    def __init__(out self):
        self.pairs = List[Int]()
        self.cells = 0
        self.wide = 0
        self.jobs = 0


def find_pairs(input: BroadphaseInput, jobs: Int = 0) -> PairResult:
    """Every candidate pair, sorted by (a, b). `jobs` is the job pool's
    address (0: this thread); the result does not depend on it."""
    var result = PairResult()
    var rows = input.rows()
    if rows < 2:
        return result^
    var grid = HashGrid(rows)
    for row in range(rows):
        if input.live[row]:
            grid.insert(row, input.lo[row], input.hi[row])
    var job_count = (rows + ROWS_PER_JOB - 1) // ROWS_PER_JOB
    var words = (rows * rows + 63) // 64
    var bitsets = List[List[UInt64]]()
    for _ in range(job_count):
        bitsets.append(List[UInt64](length=words, fill=0))
    var shared = PairJobs(Int(Pointer(to=input)), Int(Pointer(to=grid)), Int(Pointer(to=bitsets)), rows)
    run_jobs(jobs, run_pair_job, Int(Pointer(to=shared)), job_count, ZONE_BROADPHASE)
    _ = shared.rows  # the jobs read through the addresses: keep everything alive until here
    _ = len(grid.wide)
    var merged = List[UInt64](length=words, fill=0)
    for job in range(job_count):
        for word in range(words):
            merged[word] |= bitsets[job][word]
    for word in range(words):
        var bits = merged[word]
        while bits != 0:
            var low = Int(count_trailing_zeros(bits))
            result.pairs.append(word * 64 + low)
            bits &= bits - 1
    result.cells = len(grid.slot_key)
    result.wide = len(grid.wide)
    result.jobs = job_count
    return result^
