"""Flint Data-Oriented ECS: Generational Handles, Dynamic Archetypes, 8-Wide SIMD SoA Columns & Safe Parallel Jobs.

Provides:
- `FLINT_MAGIC` (`"FLNT"` = `0x544E4C46`) container identifier
- `ArchetypeRegistry`: game-defined entity archetype names (no hardcoded player/enemy kinds)
- `EntityHandle` & `ArchetypeSlots`: deterministic lowest-free row allocator with deferred tick commit
- `SimdColumn[dtype]`: 8-lane SIMD-padded Structure-of-Arrays column with `MemCharge` & `BlobWriter` I/O
- `run_parallel_ctx`: lifetime-safe wrapper over `JobPool.run_jobs` preventing Mojo ASAP destruction
"""
from std.sys import size_of
from flint.core.jobs import JobPool, JobBody, run_jobs
from flint.core.memory import MemCharge, MEM_ENTITIES
from flint.core.serial import BlobWriter, BlobReader, fourcc

comptime FLINT_MAGIC: UInt32 = 0x544E4C46  # "FLNT"
comptime FLINT_ECS_VERSION: Int = 1

comptime ROW_FREE: UInt8 = 0
comptime ROW_SPAWNING: UInt8 = 1
comptime ROW_LIVE: UInt8 = 2
comptime ROW_DESPAWNING: UInt8 = 3
comptime GEN_MASK: UInt32 = 0x00FFFFFF


@fieldwise_init
struct EntityHandle(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """64-bit packable generational entity handle: archetype (8b) | row (32b) | generation (24b)."""
    var archetype: Int
    var row: Int
    var generation: UInt32

    def __init__(out self):
        self.archetype = 0
        self.row = -1
        self.generation = 0

    def __eq__(self, other: Self) -> Bool:
        return (
            self.archetype == other.archetype
            and self.row == other.row
            and self.generation == other.generation
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def valid(self) -> Bool:
        return self.archetype > 0 and self.row >= 0

    def packed(self) -> UInt64:
        if not self.valid():
            return 0
        return (
            (UInt64(self.archetype & 0xFF) << 56)
            | (UInt64(self.row & 0xFFFFFFFF) << 24)
            | UInt64(self.generation & GEN_MASK)
        )

    @staticmethod
    def unpack(bits: UInt64) -> Self:
        if bits == 0:
            return Self()
        return Self(
            Int((bits >> 56) & 0xFF),
            Int((bits >> 24) & 0xFFFFFFFF),
            UInt32(bits & UInt64(GEN_MASK)),
        )


struct ArchetypeRegistry(Copyable, Movable):
    """Maps game-defined archetype IDs (1..255) to human-readable debug names."""
    var names: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.names.append("none")

    def register(mut self, name: String) -> Int:
        var id = len(self.names)
        self.names.append(name)
        return id

    def name_of(self, archetype: Int) -> String:
        if archetype >= 0 and archetype < len(self.names):
            return self.names[archetype]
        return "archetype_" + String(archetype)


struct ArchetypeSlots(Copyable, Movable):
    """Deterministic lowest-free row pool with deferred tick-boundary spawn/despawn commit."""
    var archetype: Int
    var capacity: Int
    var state: List[UInt8]
    var generation: List[UInt32]
    var used: Int
    var pending: Int
    var live_count: Int

    def __init__(out self, archetype: Int, capacity: Int):
        var cap = max(8, capacity)
        self.archetype = archetype
        self.capacity = cap
        self.state = List[UInt8](length=cap, fill=ROW_FREE)
        self.generation = List[UInt32](length=cap, fill=0)
        self.used = 0
        self.pending = 0
        self.live_count = 0

    def spawn(mut self, deferred: Bool = False) -> EntityHandle:
        for idx in range(self.capacity):
            if self.state[idx] == ROW_FREE:
                if deferred:
                    self.state[idx] = ROW_SPAWNING
                    self.pending += 1
                else:
                    self.state[idx] = ROW_LIVE
                    self.live_count += 1
                if idx >= self.used:
                    self.used = idx + 1
                return EntityHandle(self.archetype, idx, self.generation[idx])
        return EntityHandle()

    def resolve(self, handle: EntityHandle) -> Int:
        if handle.archetype != self.archetype or handle.row < 0 or handle.row >= self.used:
            return -1
        var st = self.state[handle.row]
        if st != ROW_LIVE and st != ROW_DESPAWNING:
            return -1
        if self.generation[handle.row] != handle.generation:
            return -1
        return handle.row

    def despawn(mut self, handle: EntityHandle, deferred: Bool = False) -> Bool:
        var row = self.resolve(handle)
        if row < 0:
            return False
        if not deferred:
            if self.state[row] == ROW_LIVE:
                self.live_count -= 1
            self.state[row] = ROW_FREE
            self.generation[row] = (self.generation[row] + 1) & GEN_MASK
            while self.used > 0 and self.state[self.used - 1] == ROW_FREE:
                self.used -= 1
            return True
        if self.state[row] == ROW_LIVE:
            self.state[row] = ROW_DESPAWNING
            self.pending += 1
            return True
        return False

    def commit_tick(mut self):
        if self.pending == 0:
            return
        for idx in range(self.used):
            if self.state[idx] == ROW_SPAWNING:
                self.state[idx] = ROW_LIVE
                self.live_count += 1
            elif self.state[idx] == ROW_DESPAWNING:
                self.state[idx] = ROW_FREE
                self.generation[idx] = (self.generation[idx] + 1) & GEN_MASK
                self.live_count -= 1
        self.pending = 0
        while self.used > 0 and self.state[self.used - 1] == ROW_FREE:
            self.used -= 1

    def is_live(self, row: Int) -> Bool:
        if row < 0 or row >= self.used:
            return False
        var st = self.state[row]
        return st == ROW_LIVE or st == ROW_DESPAWNING

    def write_blob(self, mut w: BlobWriter):
        w.integer(FLINT_ECS_VERSION)
        w.integer(self.archetype)
        w.integer(self.capacity)
        w.integer(self.used)
        w.integer(self.pending)
        w.integer(self.live_count)
        for i in range(self.used):
            w.u8(self.state[i])
            w.u32(self.generation[i])

    def read_blob(mut self, mut r: BlobReader) -> Bool:
        var ver = r.integer()
        if ver != FLINT_ECS_VERSION:
            return False
        self.archetype = r.integer()
        var cap = max(8, r.integer())
        if cap != self.capacity:
            self.capacity = cap
            self.state = List[UInt8](length=cap, fill=ROW_FREE)
            self.generation = List[UInt32](length=cap, fill=0)
        else:
            for i in range(self.capacity):
                self.state[i] = ROW_FREE
                self.generation[i] = 0
        self.used = min(self.capacity, max(0, r.integer()))
        self.pending = max(0, r.integer())
        self.live_count = max(0, r.integer())
        for i in range(self.used):
            self.state[i] = r.u8()
            self.generation[i] = r.u32()
        return not r.failed


struct SimdColumn[dtype: DType](Copyable, Movable):
    """8-lane SIMD-padded SoA column with automatic MemCharge RAM tracking and binary persistence."""
    var data: List[Scalar[Self.dtype]]
    var capacity: Int
    var charge: MemCharge

    def __init__(out self, capacity: Int, mem_tag: Int = MEM_ENTITIES):
        var padded = ((max(8, capacity) + 7) // 8) * 8
        self.data = List[Scalar[Self.dtype]](length=padded, fill=0)
        self.capacity = padded
        self.charge = MemCharge(mem_tag)
        self.charge.set(padded * size_of[Scalar[Self.dtype]]())

    @always_inline
    def base_addr(self) -> Int:
        return Int(self.data.unsafe_ptr())

    @always_inline
    def keep_alive(self):
        """Anchors this column past a parallel `JobPool` barrier so Mojo ASAP destruction cannot free `data` early."""
        var anchor = Int(self.data.unsafe_ptr())
        _ = anchor

    @always_inline
    def get(self, idx: Int) -> Scalar[Self.dtype]:
        return self.data[idx]

    @always_inline
    def set(mut self, idx: Int, val: Scalar[Self.dtype]):
        self.data[idx] = val

    @always_inline
    def load8(self, idx: Int) -> SIMD[Self.dtype, 8]:
        return Pointer[Scalar[Self.dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(self.data.unsafe_ptr()) + idx * size_of[Scalar[Self.dtype]]()
        ).unsafe_load[width=8, alignment=4]()

    @always_inline
    def store8(mut self, idx: Int, val: SIMD[Self.dtype, 8]):
        Pointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=Int(self.data.unsafe_ptr()) + idx * size_of[Scalar[Self.dtype]]()
        ).unsafe_store[alignment=4](val)

    def write_raw(self, mut w: BlobWriter, used_rows: Int):
        var n = min(self.capacity, max(0, used_rows))
        var byte_len = n * size_of[Scalar[Self.dtype]]()
        w.integer(n)
        w.integer(byte_len)
        var ptr = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=Int(self.data.unsafe_ptr()))
        for b in range(byte_len):
            w.u8(ptr[unsafe_offset=b])

    def read_raw(mut self, mut r: BlobReader) -> Bool:
        var n = r.integer()
        var byte_len = r.integer()
        if n < 0 or n > self.capacity or byte_len != n * size_of[Scalar[Self.dtype]]():
            return False
        var ptr = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(self.data.unsafe_ptr()))
        for b in range(byte_len):
            ptr[unsafe_offset=b] = r.u8()
        return not r.failed


def run_parallel_ctx[CtxType: Movable](
    mut pool: JobPool,
    job_count: Int,
    mut ctx: CtxType,
    worker_fn: JobBody,
):
    """Dispatches `job_count` parallel jobs while anchoring `ctx` alive across the worker barrier.

    Prevents Mojo's eager ASAP value destruction from freeing stack context structs
    before worker threads finish reading `ctx_addr`.
    """
    var ctx_addr = Int(Pointer(to=ctx))
    run_jobs(Int(Pointer(to=pool)), worker_fn, ctx_addr, job_count)
    var keep_alive_anchor = Int(Pointer(to=ctx))
    _ = keep_alive_anchor
