"""Flint Data-Oriented ECS: Generational Handles, Dynamic Archetypes, 8-Wide SIMD SoA Columns & Safe Parallel Jobs.

Provides:
- `FLINT_MAGIC` (`"FLNT"` = `0x544E4C46`) container identifier
- `ArchetypeRegistry`: game-defined entity archetype names (no hardcoded player/enemy kinds)
- `EntityHandle` & `ArchetypeSlots`: deterministic lowest-free row allocator with deferred tick commit
- `SimdColumn[dtype]`: 8-lane SIMD-padded Structure-of-Arrays column with `MemCharge` & `BlobWriter` I/O
- `run_parallel_ctx`: lifetime-safe wrapper over `JobPool.run_jobs` preventing Mojo ASAP destruction
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.sys import size_of
from flint.core.jobs import JobPool, JobBody, run_jobs_affine
from flint.core.memory import MemCharge, MEM_ENTITIES
from flint.core.serial import BlobWriter, BlobReader, fourcc

comptime FLINT_MAGIC: UInt32 = 0x544E4C46  # "FLNT"
comptime FLINT_ECS_VERSION: Int = 1

comptime ROW_FREE: UInt8 = 0
comptime ROW_SPAWNING: UInt8 = 1
comptime ROW_LIVE: UInt8 = 2
comptime ROW_DESPAWNING: UInt8 = 3
comptime GEN_MASK: UInt32 = 0x00FFFFFF
comptime ECB_PROV_GEN: UInt32 = 0x00FFFFFF  # Reserved sentinel generation for EntityCommandBuffer provisional handles


@always_inline
def _next_slot_gen(g: UInt32) -> UInt32:
    var ng = (g + 1) & GEN_MASK
    return 0 if ng == ECB_PROV_GEN else ng


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


@always_inline
def _trailing_zeros_64(val: UInt64) -> Int:
    if val == 0:
        return 64
    var v = val
    var count = 0
    if (v & 0xFFFFFFFF) == 0:
        count += 32
        v >>= 32
    if (v & 0xFFFF) == 0:
        count += 16
        v >>= 16
    if (v & 0xFF) == 0:
        count += 8
        v >>= 8
    if (v & 0xF) == 0:
        count += 4
        v >>= 4
    if (v & 0x3) == 0:
        count += 2
        v >>= 2
    if (v & 0x1) == 0:
        count += 1
    return count


struct ArchetypeSlots(Copyable, Movable):
    """Deterministic lowest-free row pool with chunked bitset O(1) allocation and deferred tick commit."""
    var archetype: Int
    var capacity: Int
    var state: List[UInt8]
    var generation: List[UInt32]
    var free_mask: List[UInt64]
    var used: Int
    var pending: Int
    var live_count: Int

    def __init__(out self, archetype: Int, capacity: Int):
        var cap = max(8, capacity)
        self.archetype = archetype
        self.capacity = cap
        self.state = List[UInt8](length=cap, fill=ROW_FREE)
        self.generation = List[UInt32](length=cap, fill=0)
        var num_chunks = (cap + 63) // 64
        self.free_mask = List[UInt64](length=num_chunks, fill=0xFFFFFFFFFFFFFFFF)
        var rem = cap % 64
        if rem != 0:
            var valid_mask = (UInt64(1) << UInt64(rem)) - 1
            self.free_mask[num_chunks - 1] = valid_mask
        self.used = 0
        self.pending = 0
        self.live_count = 0

    def spawn(mut self, deferred: Bool = False) -> EntityHandle:
        for c in range(len(self.free_mask)):
            var m = self.free_mask[c]
            if m != 0:
                var bit_idx = _trailing_zeros_64(m)
                var idx = c * 64 + bit_idx
                if idx < self.capacity:
                    self.free_mask[c] = m & ~(UInt64(1) << UInt64(bit_idx))
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

    def resolve(self, handle: EntityHandle, allow_spawning: Bool = False) -> Int:
        if handle.archetype != self.archetype or handle.row < 0 or handle.row >= self.used:
            return -1
        var st = self.state[handle.row]
        if st != ROW_LIVE and st != ROW_DESPAWNING:
            if not (allow_spawning and st == ROW_SPAWNING):
                return -1
        if self.generation[handle.row] != handle.generation:
            return -1
        return handle.row

    def despawn(mut self, handle: EntityHandle, deferred: Bool = False) -> Bool:
        var row = self.resolve(handle, allow_spawning=True)
        if row < 0:
            return False
        if self.state[row] == ROW_SPAWNING:
            self.state[row] = ROW_FREE
            self.pending = max(0, self.pending - 1)
            var c = row // 64
            var b = row % 64
            self.free_mask[c] |= (UInt64(1) << UInt64(b))
            self.generation[row] = _next_slot_gen(self.generation[row])
            while self.used > 0 and self.state[self.used - 1] == ROW_FREE:
                self.used -= 1
            return True
        if not deferred:
            if self.state[row] == ROW_LIVE:
                self.live_count -= 1
            elif self.state[row] == ROW_DESPAWNING:
                self.live_count -= 1
                self.pending = max(0, self.pending - 1)
            self.state[row] = ROW_FREE
            var c = row // 64
            var b = row % 64
            self.free_mask[c] |= (UInt64(1) << UInt64(b))
            self.generation[row] = _next_slot_gen(self.generation[row])
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
                var c = idx // 64
                var b = idx % 64
                self.free_mask[c] |= (UInt64(1) << UInt64(b))
                self.generation[idx] = _next_slot_gen(self.generation[idx])
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
        var num_chunks = (self.capacity + 63) // 64
        self.free_mask = List[UInt64](length=num_chunks, fill=0)
        for i in range(self.capacity):
            if i < self.used:
                self.state[i] = r.u8()
                self.generation[i] = r.u32()
            else:
                self.state[i] = ROW_FREE
                self.generation[i] = 0
            if self.state[i] == ROW_FREE:
                self.free_mask[i // 64] |= (UInt64(1) << UInt64(i % 64))
        return not r.failed


comptime ECB_OP_SPAWN: Int = 1
comptime ECB_OP_DESPAWN: Int = 2


struct EcbCommand(Copyable, Movable):
    var op: Int
    var archetype: Int
    var handle: EntityHandle

    def __init__(out self, op: Int, archetype: Int, handle: EntityHandle):
        self.op = op
        self.archetype = archetype
        self.handle = handle


struct EntityCommandBuffer(Movable):
    """Thread-safe deferred command buffer for parallel worker entity mutation."""
    var commands: List[EcbCommand]
    var _lock_buf: List[Int64]
    var next_seq: Int

    def __init__(out self):
        self.commands = List[EcbCommand]()
        self._lock_buf = List[Int64](length=1, fill=0)
        self.next_seq = 0

    @always_inline
    def _acquire(self):
        var p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=Int(self._lock_buf.unsafe_ptr()))
        while True:
            var expected = Int64(0)
            if p[unsafe_offset=0].compare_exchange(expected, Int64(1)):
                return
            _ = external_call["sched_yield", Int32]()

    @always_inline
    def _release(self):
        var p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=Int(self._lock_buf.unsafe_ptr()))
        p[unsafe_offset=0].store(Int64(0))

    def spawn(mut self, archetype: Int) -> EntityHandle:
        self._acquire()
        var seq = self.next_seq
        self.next_seq += 1
        var h = EntityHandle(archetype, seq, ECB_PROV_GEN) if archetype > 0 else EntityHandle()
        self.commands.append(EcbCommand(ECB_OP_SPAWN, archetype, h))
        self._release()
        return h

    def despawn(mut self, handle: EntityHandle):
        self._acquire()
        self.commands.append(EcbCommand(ECB_OP_DESPAWN, handle.archetype, handle))
        self._release()

    def playback(mut self, mut slots: ArchetypeSlots):
        self._acquire()
        var prov_rows = List[Int]()
        var real_handles = List[EntityHandle]()
        for i in range(len(self.commands)):
            var cmd = self.commands[i].copy()
            if cmd.archetype == slots.archetype:
                if cmd.op == ECB_OP_SPAWN:
                    var real_h = slots.spawn(deferred=True)
                    prov_rows.append(cmd.handle.row)
                    real_handles.append(real_h)
                elif cmd.op == ECB_OP_DESPAWN:
                    var target_h = cmd.handle
                    if target_h.generation == ECB_PROV_GEN and target_h.row >= 0:
                        var mapped = False
                        for j in range(len(prov_rows)):
                            if prov_rows[j] == target_h.row:
                                target_h = real_handles[j]
                                mapped = True
                                break
                        if not mapped:
                            continue
                    _ = slots.despawn(target_h, deferred=True)
        self._release()

    def count(self) -> Int:
        self._acquire()
        var n = len(self.commands)
        self._release()
        return n

    def clear(mut self):
        self._acquire()
        self.commands.clear()
        self.next_seq = 0
        self._release()


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
        if idx < 0 or idx + 8 > self.capacity:
            return SIMD[Self.dtype, 8](0)
        return Pointer[Scalar[Self.dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(self.data.unsafe_ptr()) + idx * size_of[Scalar[Self.dtype]]()
        ).unsafe_load[width=8, alignment=4]()

    @always_inline
    def store8(mut self, idx: Int, val: SIMD[Self.dtype, 8]):
        if idx < 0 or idx + 8 > self.capacity:
            return
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
    run_jobs_affine(Int(Pointer(to=pool)), worker_fn, ctx_addr, job_count)
    var keep_alive_anchor = Int(Pointer(to=ctx))
    _ = keep_alive_anchor
