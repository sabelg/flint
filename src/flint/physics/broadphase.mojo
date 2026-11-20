"""Broadphase: Dynamic 8-wide SIMD SAH BVH with L2-affine parallel queries (v4).

Replaces O(N^3) per-job bitset allocations and fixed 1,024-bucket HashGrid spill
with a two-stage 16-bin SAH BVH collapsed into 8-wide SIMD `BroadphaseNode8`
packets:
  - Memory at N = 5,000 bodies is < 1 MiB (zero O(N^2) bitsets).
  - Arbitrary-scale bodies (e.g. 124 m capital ships alongside 0.3 m missiles)
    are bounded natively in O(log N) with zero `wide` O(N^2) spill.
  - Awake bodies query the BVH8 in parallel via `run_jobs_affine` into
    per-chunk `List[Int]` buffers, merged with deterministic bottom-up merge sort
    and adjacent deduplication for bit-identical `PairResult` across any worker count.
"""
from std.collections import List
from std.math import floor, min, max, abs
from flint.core.math import Vec3
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.physics.filter import CollisionFilter

comptime CELL_SIZE: Int = 8               # preserved for legacy spatial-hash callers
comptime HASH_BUCKETS: Int = 1024         # power of two
comptime MAX_CELLS_PER_BODY: Int = 64
comptime ROWS_PER_JOB: Int = 32
comptime _BP_LANES: Int = 8
comptime _BP_SAH_BINS: Int = 16
comptime _F8 = SIMD[DType.float32, _BP_LANES]
comptime _I8 = SIMD[DType.int32, _BP_LANES]


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
    return ((y + 1024) << 22) | ((z + 1024) << 11) | (x + 1024)


@always_inline
def cell_coordinate(value: Float32) -> Int:
    if not (value > -1.0e6 and value < 1.0e6):
        return -1000000
    return Int(floor(value / Float32(CELL_SIZE)))


@always_inline
def bucket_of(key: Int) -> Int:
    var h = UInt64(key) * 0x9E3779B97F4A7C15
    return Int(h >> 54) & (HASH_BUCKETS - 1)


@always_inline
def boxes_overlap(alo: Vec3, ahi: Vec3, blo: Vec3, bhi: Vec3) -> Bool:
    return alo.x <= bhi.x and blo.x <= ahi.x and alo.y <= bhi.y and blo.y <= ahi.y and alo.z <= bhi.z and blo.z <= ahi.z


struct HashGrid(Movable):
    """Legacy open-addressing spatial hash grid preserved for API compatibility."""
    var bucket_slot: List[Int]
    var slot_key: List[Int]
    var slot_rows: List[List[Int]]
    var wide: List[Int]
    var body_cells: List[List[Int]]

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
        return -1

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


@always_inline
def _half_area(lo: Vec3, hi: Vec3) -> Float32:
    var dx = max(Float32(0.0), hi.x - lo.x)
    var dy = max(Float32(0.0), hi.y - lo.y)
    var dz = max(Float32(0.0), hi.z - lo.z)
    return dx * dy + dy * dz + dz * dx


@fieldwise_init
struct _BpBinNode(Copyable, ImplicitlyCopyable, Movable):
    var lo: Vec3
    var hi: Vec3
    var left: Int32   # >= 0: left child; < 0: -(body_row + 1) for single-body leaf
    var right: Int32  # >= 0: right child; -1 for leaf
    var cat_or: UInt32
    var mask_or: UInt32


@fieldwise_init
struct BroadphaseNode8(Copyable, ImplicitlyCopyable, Movable):
    """8-wide SIMD BVH node.
    `child[c] > 0`: internal `BroadphaseNode8` index.
    `child[c] < 0`: leaf body row `-(child[c] + 1)`.
    `child[c] == 0`: empty lane (`min = +1e38, max = -1e38`).
    """
    var min_x: _F8
    var min_y: _F8
    var min_z: _F8
    var max_x: _F8
    var max_y: _F8
    var max_z: _F8
    var child: _I8
    var cat_or: UInt32
    var mask_or: UInt32

    def __init__(out self):
        self.min_x = _F8(1.0e38)
        self.min_y = _F8(1.0e38)
        self.min_z = _F8(1.0e38)
        self.max_x = _F8(-1.0e38)
        self.max_y = _F8(-1.0e38)
        self.max_z = _F8(-1.0e38)
        self.child = _I8(0)
        self.cat_or = 0
        self.mask_or = 0


def _build_bp_bvh2(
    input: BroadphaseInput,
    mut perm: List[Int],
    cx: List[Float32],
    cy: List[Float32],
    cz: List[Float32],
    start: Int,
    end: Int,
    mut out_nodes: List[_BpBinNode],
) -> Int:
    var node_idx = len(out_nodes)
    var count = end - start
    if count == 1:
        var r = perm[start]
        var filt = input.filters[r]
        out_nodes.append(_BpBinNode(input.lo[r], input.hi[r], Int32(-(r + 1)), Int32(-1), filt.category, filt.mask))
        return node_idx

    var n_lo = Vec3(1.0e38, 1.0e38, 1.0e38)
    var n_hi = Vec3(-1.0e38, -1.0e38, -1.0e38)
    var c_lo = Vec3(1.0e38, 1.0e38, 1.0e38)
    var c_hi = Vec3(-1.0e38, -1.0e38, -1.0e38)
    var cat_or = UInt32(0)
    var mask_or = UInt32(0)

    for i in range(start, end):
        var r = perm[i]
        var blo = input.lo[r]
        var bhi = input.hi[r]
        n_lo.x = min(n_lo.x, blo.x); n_lo.y = min(n_lo.y, blo.y); n_lo.z = min(n_lo.z, blo.z)
        n_hi.x = max(n_hi.x, bhi.x); n_hi.y = max(n_hi.y, bhi.y); n_hi.z = max(n_hi.z, bhi.z)
        c_lo.x = min(c_lo.x, cx[r]); c_lo.y = min(c_lo.y, cy[r]); c_lo.z = min(c_lo.z, cz[r])
        c_hi.x = max(c_hi.x, cx[r]); c_hi.y = max(c_hi.y, cy[r]); c_hi.z = max(c_hi.z, cz[r])
        cat_or |= input.filters[r].category
        mask_or |= input.filters[r].mask

    out_nodes.append(_BpBinNode(n_lo, n_hi, Int32(0), Int32(0), cat_or, mask_or))

    var span_x = c_hi.x - c_lo.x
    var span_y = c_hi.y - c_lo.y
    var span_z = c_hi.z - c_lo.z
    var axis = 0
    var span = span_x
    if span_y > span:
        axis = 1
        span = span_y
    if span_z > span:
        axis = 2
        span = span_z

    var mid = (start + end) // 2
    if count >= 6 and span > Float32(1.0e-5):
        var c_min = c_lo.x if axis == 0 else (c_lo.y if axis == 1 else c_lo.z)
        var inv_span = Float32(_BP_SAH_BINS) / span
        var bin_cnt = Array[Int, _BP_SAH_BINS](fill=0)
        var b_lox = Array[Float32, _BP_SAH_BINS](fill=1.0e38)
        var b_loy = Array[Float32, _BP_SAH_BINS](fill=1.0e38)
        var b_loz = Array[Float32, _BP_SAH_BINS](fill=1.0e38)
        var b_hix = Array[Float32, _BP_SAH_BINS](fill=-1.0e38)
        var b_hiy = Array[Float32, _BP_SAH_BINS](fill=-1.0e38)
        var b_hiz = Array[Float32, _BP_SAH_BINS](fill=-1.0e38)

        for i in range(start, end):
            var r = perm[i]
            var coord = cx[r] if axis == 0 else (cy[r] if axis == 1 else cz[r])
            var b = Int((coord - c_min) * inv_span)
            if b < 0:
                b = 0
            elif b >= _BP_SAH_BINS:
                b = _BP_SAH_BINS - 1
            bin_cnt[b] += 1
            var blo = input.lo[r]
            var bhi = input.hi[r]
            b_lox[b] = min(b_lox[b], blo.x); b_loy[b] = min(b_loy[b], blo.y); b_loz[b] = min(b_loz[b], blo.z)
            b_hix[b] = max(b_hix[b], bhi.x); b_hiy[b] = max(b_hiy[b], bhi.y); b_hiz[b] = max(b_hiz[b], bhi.z)

        var right_area = Array[Float32, _BP_SAH_BINS](fill=0.0)
        var right_cnt = Array[Int, _BP_SAH_BINS](fill=0)
        var r_lo = Vec3(1.0e38, 1.0e38, 1.0e38)
        var r_hi = Vec3(-1.0e38, -1.0e38, -1.0e38)
        var r_c = 0
        var b_rev = _BP_SAH_BINS - 1
        while b_rev >= 1:
            if bin_cnt[b_rev] > 0:
                r_c += bin_cnt[b_rev]
                r_lo.x = min(r_lo.x, b_lox[b_rev]); r_lo.y = min(r_lo.y, b_loy[b_rev]); r_lo.z = min(r_lo.z, b_loz[b_rev])
                r_hi.x = max(r_hi.x, b_hix[b_rev]); r_hi.y = max(r_hi.y, b_hiy[b_rev]); r_hi.z = max(r_hi.z, b_hiz[b_rev])
            right_cnt[b_rev] = r_c
            right_area[b_rev] = _half_area(r_lo, r_hi) if r_c > 0 else Float32(0.0)
            b_rev -= 1

        var l_lo = Vec3(1.0e38, 1.0e38, 1.0e38)
        var l_hi = Vec3(-1.0e38, -1.0e38, -1.0e38)
        var l_c = 0
        var best_cost = Float32(1.0e38)
        var best_split = -1
        for b in range(_BP_SAH_BINS - 1):
            if bin_cnt[b] > 0:
                l_c += bin_cnt[b]
                l_lo.x = min(l_lo.x, b_lox[b]); l_lo.y = min(l_lo.y, b_loy[b]); l_lo.z = min(l_lo.z, b_loz[b])
                l_hi.x = max(l_hi.x, b_hix[b]); l_hi.y = max(l_hi.y, b_hiy[b]); l_hi.z = max(l_hi.z, b_hiz[b])
            var rc = right_cnt[b + 1]
            if l_c > 0 and rc > 0:
                var cost = _half_area(l_lo, l_hi) * Float32(l_c) + right_area[b + 1] * Float32(rc)
                if cost < best_cost:
                    best_cost = cost
                    best_split = b

        if best_split >= 0:
            var i_left = start
            var i_right = end - 1
            while i_left <= i_right:
                var r = perm[i_left]
                var coord = cx[r] if axis == 0 else (cy[r] if axis == 1 else cz[r])
                var b = Int((coord - c_min) * inv_span)
                if b < 0:
                    b = 0
                elif b >= _BP_SAH_BINS:
                    b = _BP_SAH_BINS - 1
                if b <= best_split:
                    i_left += 1
                else:
                    perm[i_left] = perm[i_right]
                    perm[i_right] = r
                    i_right -= 1
            if i_left > start and i_left < end:
                mid = i_left
    elif span > Float32(1.0e-5):
        var c_mid = (c_lo.x + c_hi.x) * Float32(0.5) if axis == 0 else (
            (c_lo.y + c_hi.y) * Float32(0.5) if axis == 1 else (c_lo.z + c_hi.z) * Float32(0.5)
        )
        var i_left = start
        var i_right = end - 1
        while i_left <= i_right:
            var r = perm[i_left]
            var coord = cx[r] if axis == 0 else (cy[r] if axis == 1 else cz[r])
            if coord <= c_mid:
                i_left += 1
            else:
                perm[i_left] = perm[i_right]
                perm[i_right] = r
                i_right -= 1
        if i_left > start and i_left < end:
            mid = i_left

    var left_idx = _build_bp_bvh2(input, perm, cx, cy, cz, start, mid, out_nodes)
    var right_idx = _build_bp_bvh2(input, perm, cx, cy, cz, mid, end, out_nodes)
    out_nodes[node_idx].left = Int32(left_idx)
    out_nodes[node_idx].right = Int32(right_idx)
    return node_idx


def _collapse_bp_bvh8(bin_nodes: List[_BpBinNode], bin_idx: Int, mut out8: List[BroadphaseNode8]) -> Int:
    var out_idx = len(out8)
    out8.append(BroadphaseNode8())

    var frontier = Array[Int, _BP_LANES](fill=-1)
    var f_count: Int
    var root_bin = bin_nodes[bin_idx]
    if root_bin.right < 0:
        frontier[0] = bin_idx
        f_count = 1
    else:
        frontier[0] = Int(root_bin.left)
        frontier[1] = Int(root_bin.right)
        f_count = 2
        while f_count < _BP_LANES:
            var best_k = -1
            var best_area = Float32(-1.0)
            for k in range(f_count):
                var bn = bin_nodes[frontier[k]]
                if bn.right >= 0:
                    var a = _half_area(bn.lo, bn.hi)
                    if a > best_area:
                        best_area = a
                        best_k = k
            if best_k < 0:
                break
            var expand = bin_nodes[frontier[best_k]]
            frontier[best_k] = Int(expand.left)
            frontier[f_count] = Int(expand.right)
            f_count += 1

    var node8 = BroadphaseNode8()
    node8.cat_or = root_bin.cat_or
    node8.mask_or = root_bin.mask_or
    for c in range(f_count):
        var b_i = frontier[c]
        var bn = bin_nodes[b_i]
        node8.min_x[c] = bn.lo.x
        node8.min_y[c] = bn.lo.y
        node8.min_z[c] = bn.lo.z
        node8.max_x[c] = bn.hi.x
        node8.max_y[c] = bn.hi.y
        node8.max_z[c] = bn.hi.z
        if bn.right < 0:
            node8.child[c] = bn.left  # negative: -(body_row + 1)
        else:
            var child8_idx = _collapse_bp_bvh8(bin_nodes, b_i, out8)
            node8.child[c] = Int32(child8_idx)
    out8[out_idx] = node8
    return out_idx


struct _BpQueryShared(Movable):
    var input_addr: Int
    var nodes_addr: Int
    var awake_addr: Int
    var out_lists_addr: Int
    var rows: Int
    var awake_count: Int

    def __init__(out self, input_addr: Int, nodes_addr: Int, awake_addr: Int, out_lists_addr: Int, rows: Int, awake_count: Int):
        self.input_addr = input_addr
        self.nodes_addr = nodes_addr
        self.awake_addr = awake_addr
        self.out_lists_addr = out_lists_addr
        self.rows = rows
        self.awake_count = awake_count


def _query_awake_chunk(
    input: BroadphaseInput,
    nodes: List[BroadphaseNode8],
    awake_rows: List[Int],
    first_idx: Int,
    end_idx: Int,
    rows: Int,
    mut out_pairs: List[Int],
):
    var stack = Array[Int32, 96](fill=0)
    for idx in range(first_idx, end_idx):
        var q = awake_rows[idx]
        var qlo = input.lo[q]
        var qhi = input.hi[q]
        var qfilt = input.filters[q]
        var qlo_x = _F8(qlo.x)
        var qlo_y = _F8(qlo.y)
        var qlo_z = _F8(qlo.z)
        var qhi_x = _F8(qhi.x)
        var qhi_y = _F8(qhi.y)
        var qhi_z = _F8(qhi.z)

        var sp = 1
        stack[0] = 0
        while sp > 0:
            sp -= 1
            var n_idx = Int(stack[sp])
            ref node = nodes[n_idx]
            if qfilt.group == 0 and ((qfilt.category & node.mask_or) == 0 or (node.cat_or & qfilt.mask) == 0):
                continue

            var ov = (
                qlo_x.le(node.max_x)
                & qhi_x.ge(node.min_x)
                & qlo_y.le(node.max_y)
                & qhi_y.ge(node.min_y)
                & qlo_z.le(node.max_z)
                & qhi_z.ge(node.min_z)
            )
            if not ov.reduce_or():
                continue

            comptime for c in range(_BP_LANES):
                if ov[c]:
                    var ch = Int(node.child[c])
                    if ch > 0:
                        if sp < 96:
                            stack[sp] = Int32(ch)
                            sp += 1
                    elif ch < 0:
                        var other = -ch - 1
                        if other != q and (not input.awake[other] or q < other):
                            if qfilt.touches(input.filters[other]):
                                var a = min(q, other)
                                var b = max(q, other)
                                out_pairs.append(a * rows + b)


def _run_bp_query_job(context: Int, chunk: Int):
    ref shared = Pointer[_BpQueryShared, origin=MutAnyOrigin](unsafe_from_address=context)[unsafe_offset=0]
    ref input = Pointer[BroadphaseInput, origin=MutAnyOrigin](unsafe_from_address=shared.input_addr)[unsafe_offset=0]
    ref nodes = Pointer[List[BroadphaseNode8], origin=MutAnyOrigin](unsafe_from_address=shared.nodes_addr)[unsafe_offset=0]
    ref awake_rows = Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=shared.awake_addr)[unsafe_offset=0]
    ref out_lists = Pointer[List[List[Int]], origin=MutAnyOrigin](unsafe_from_address=shared.out_lists_addr)[unsafe_offset=0]
    var first = chunk * ROWS_PER_JOB
    var end = min(shared.awake_count, first + ROWS_PER_JOB)
    _query_awake_chunk(input, nodes, awake_rows, first, end, shared.rows, out_lists[chunk])


def _sort_and_dedup_pairs(mut pairs: List[Int]):
    """In-place bottom-up merge sort + adjacent deduplication (O(P log P), deterministic)."""
    var count = len(pairs)
    if count >= 2:
        var scratch = List[Int](length=count, fill=0)
        var width = 1
        while width < count:
            var start = 0
            while start < count:
                var mid = min(start + width, count)
                var end = min(start + 2 * width, count)
                var l = start
                var r = mid
                var o = start
                while l < mid and r < end:
                    if pairs[l] <= pairs[r]:
                        scratch[o] = pairs[l]
                        l += 1
                    else:
                        scratch[o] = pairs[r]
                        r += 1
                    o += 1
                while l < mid:
                    scratch[o] = pairs[l]
                    l += 1
                    o += 1
                while r < end:
                    scratch[o] = pairs[r]
                    r += 1
                    o += 1
                start = end
            for i in range(count):
                pairs[i] = scratch[i]
            width *= 2

        var write = 1
        for read in range(1, count):
            if pairs[read] != pairs[write - 1]:
                pairs[write] = pairs[read]
                write += 1
        if write < count:
            pairs.resize(write, 0)


struct PairResult(Movable):
    var pairs: List[Int]   # a * rows + b, a < b, sorted
    var cells: Int         # BVH8 nodes in use
    var wide: Int          # always 0 with BVH8 (arbitrary-scale bodies bounded natively)
    var jobs: Int

    def __init__(out self):
        self.pairs = List[Int]()
        self.cells = 0
        self.wide = 0
        self.jobs = 0


def find_pairs(input: BroadphaseInput, jobs: Int = 0) -> PairResult:
    """Every candidate pair, sorted by (a, b). `jobs` is the job pool's
    address (0: this thread); the result is bit-identical across worker counts."""
    var nodes8 = List[BroadphaseNode8]()
    return find_pairs_into_bvh8(input, nodes8, jobs)


def find_pairs_into_bvh8(
    input: BroadphaseInput,
    mut out_nodes8: List[BroadphaseNode8],
    jobs: Int = 0,
) -> PairResult:
    """Same as `find_pairs`, but writes the built 8-wide SAH BVH into `out_nodes8`
    so the caller can retain and refit it without rebuilding a second BVH8 per tick."""
    var result = PairResult()
    var rows = input.rows()
    if rows < 2:
        return result^

    var live_rows = List[Int](capacity=rows)
    var awake_rows = List[Int](capacity=rows)
    var cx = List[Float32](length=rows, fill=0.0)
    var cy = List[Float32](length=rows, fill=0.0)
    var cz = List[Float32](length=rows, fill=0.0)

    for r in range(rows):
        if not input.live[r]:
            continue
        var blo = input.lo[r]
        var bhi = input.hi[r]
        if not (blo.x <= bhi.x and blo.y <= bhi.y and blo.z <= bhi.z):
            continue
        live_rows.append(r)
        cx[r] = (blo.x + bhi.x) * Float32(0.5)
        cy[r] = (blo.y + bhi.y) * Float32(0.5)
        cz[r] = (blo.z + bhi.z) * Float32(0.5)
        if input.awake[r]:
            awake_rows.append(r)

    var live_count = len(live_rows)
    var awake_count = len(awake_rows)
    if live_count < 2 or awake_count == 0:
        return result^

    var bin_nodes = List[_BpBinNode](capacity=2 * live_count)
    var bin_root = _build_bp_bvh2(input, live_rows, cx, cy, cz, 0, live_count, bin_nodes)
    out_nodes8.clear()
    out_nodes8.reserve((live_count + _BP_LANES - 1) // _BP_LANES + 4)
    _ = _collapse_bp_bvh8(bin_nodes, bin_root, out_nodes8)
    _ = bin_nodes^

    var job_count = (awake_count + ROWS_PER_JOB - 1) // ROWS_PER_JOB
    if jobs == 0 or job_count <= 1:
        _query_awake_chunk(input, out_nodes8, awake_rows, 0, awake_count, rows, result.pairs)
        _sort_and_dedup_pairs(result.pairs)
        result.cells = len(out_nodes8)
        result.wide = 0
        result.jobs = 1
        return result^

    var out_lists = List[List[Int]](capacity=job_count)
    for _ in range(job_count):
        out_lists.append(List[Int]())

    var shared = _BpQueryShared(
        Int(Pointer(to=input)),
        Int(Pointer(to=out_nodes8)),
        Int(Pointer(to=awake_rows)),
        Int(Pointer(to=out_lists)),
        rows,
        awake_count,
    )
    ref pool = Pointer[JobPool, origin=MutAnyOrigin](unsafe_from_address=jobs)[unsafe_offset=0]
    run_parallel_ctx(pool, job_count, shared, _run_bp_query_job)
    _ = shared.rows
    _ = len(awake_rows)
    _ = input.rows()

    var total_pairs = 0
    for j in range(job_count):
        total_pairs += len(out_lists[j])

    if total_pairs > 0:
        result.pairs.reserve(total_pairs)
        for j in range(job_count):
            ref sub = out_lists[j]
            for k in range(len(sub)):
                result.pairs.append(sub[k])
        _sort_and_dedup_pairs(result.pairs)

    result.cells = len(out_nodes8)
    result.wide = 0
    result.jobs = job_count
    return result^
