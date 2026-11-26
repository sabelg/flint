"""Flint v5 Hybrid ECS v2 (`flint.ecs.world`).

Provides:
- `comp_hash64(StaticString) -> UInt64`: deterministic compile-time/runtime component hash
- `Entity(index: UInt32, generation: UInt32)`: 64-bit generational entity handle (unbounded archetypes)
- `trait Component(Copyable, Movable, Deinitable)`: `COMPONENT_ID`, `STORAGE_MODE` (`STORAGE_TABLE` / `STORAGE_SPARSE`)
- `ComponentBitSet` (`BitSet256` alias): hybrid inline-256 + dynamic overflow component signature set for O(1) archetype queries & schedule conflict analysis
- `ArchetypeChunk`: 64-byte aligned, 8-SIMD-lane-multiple columnar storage with per-8-row `changed_ticks`
- `ArchetypeGraph`: cached `add_edges` / `remove_edges` for O(1) `insert[C]` / `remove[C]` transitions
- `SparseSet`: paged sparse-set storage for `STORAGE_SPARSE` tags and volatile components
- `ParallelCommands`: per-worker lock-free (`MAX_WORKERS + 1` lanes) command buffers with provisional
  `Entity` remapping and deterministic `detsort` `(order_key / lane_id, seq)` playback
- `Events[T]`: double-buffered 2-frame event channel (`send`, `swap_buffers`, `read_since`)
- `World`: full ECS runtime with lifetime-safe `par_for_each1` / `par_for_each2` over `JobPool`
  and `<0.2 ms` portable binary `snapshot_blob` / `restore_blob` / `state_checksum`.
"""
from std.atomic import Atomic, Ordering
from std.collections import List
from std.ffi import external_call, _Global
from std.sys import size_of, align_of
from flint.core.detsort import stable_order, mix64
from flint.core.ecs import EntityHandle, ECB_PROV_GEN, run_parallel_ctx
from flint.core.jobs import JobPool, MAX_WORKERS
from flint.core.memory import MemCharge, MEM_ENTITIES
from flint.core.serial import BlobWriter, BlobReader, fourcc, fast_hash64

comptime MAX_COMPONENTS: Int = 65536  # Soft legacy export; World imposes no fixed component limit
comptime _INITIAL_COMP_HASH_CAP: Int = 512
comptime STORAGE_TABLE: Int = 0
comptime STORAGE_SPARSE: Int = 1

comptime CHUNK_ALIGN: Int = 64
comptime SIMD_LANES: Int = 8
comptime DEFAULT_CHUNK_ROWS: Int = 64

comptime SPARSE_PAGE_BITS: Int = 10
comptime SPARSE_PAGE_SIZE: Int = 1 << SPARSE_PAGE_BITS
comptime SPARSE_PAGE_MASK: Int = SPARSE_PAGE_SIZE - 1

comptime INVALID_ENTITY_INDEX: UInt32 = 0xFFFFFFFF
comptime PROVISIONAL_GEN_BIT: UInt32 = 0x80000000
comptime MAX_LIVE_GENERATION: UInt32 = 0x7FFFFFFE

comptime MAX_COMMAND_LANES: Int = MAX_WORKERS + 1

comptime CMD_SPAWN: UInt8 = 1
comptime CMD_DESPAWN: UInt8 = 2
comptime CMD_INSERT: UInt8 = 3
comptime CMD_REMOVE: UInt8 = 4

comptime WORLD_BLOB_TAG: UInt32 = 0x32534345  # "ECS2"
comptime WORLD_BLOB_VERSION: Int = 3

# Dynamic process-local ECS runtime page (64x64B per-worker implicit order table)
comptime _ECS_RT_CTRL_KEY: StaticString = "FLINT_ECS_RT_CTRL"
comptime _ECS_RUNTIME_BYTES: Int = 8192
comptime _ECS_RUNTIME_MAGIC: Int = 0x31534345544E4C46  # "FLNTECS1"
comptime _RT_MAGIC_OFF: Int = 0
comptime _RT_PID_OFF: Int = 8
comptime _RT_WORKER_ORDER_OFF: Int = 256  # 64 slots * 64-byte cache line = 4096 B
comptime _WORKER_ORDER_SLOTS: Int = 64


def comp_hash64(name: StaticString) -> UInt64:
    """Deterministic 64-bit FNV-1a + SplitMix64 component identifier hash (never 0)."""
    var bytes = name.as_bytes()
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(bytes)):
        h = (h ^ UInt64(bytes[i])) * UInt64(0x100000001B3)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    h *= UInt64(0x94D049BB133111EB)
    h ^= h >> 31
    return h if h != 0 else UInt64(1)


def comp_hash64_str(name: String) -> UInt64:
    """Runtime `String` equivalent of `comp_hash64` (identical FNV-1a + SplitMix64 hash)."""
    var bytes = name.as_bytes()
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(bytes)):
        h = (h ^ UInt64(bytes[i])) * UInt64(0x100000001B3)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    h *= UInt64(0x94D049BB133111EB)
    h ^= h >> 31
    return h if h != 0 else UInt64(1)


@always_inline
def _next_entity_gen(gen: UInt32) -> UInt32:
    if gen >= MAX_LIVE_GENERATION or gen == 0:
        return 1
    return gen + 1


@fieldwise_init
struct Entity(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """64-bit generational entity handle decoupled from archetype ID."""
    var index: UInt32
    var generation: UInt32

    def __init__(out self):
        self.index = INVALID_ENTITY_INDEX
        self.generation = 0

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index and self.generation == other.generation

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    def valid(self) -> Bool:
        return self.index != INVALID_ENTITY_INDEX and self.generation != 0

    @always_inline
    def is_null(self) -> Bool:
        return self.index == INVALID_ENTITY_INDEX or self.generation == 0

    @always_inline
    def is_provisional(self) -> Bool:
        return self.index != INVALID_ENTITY_INDEX and (self.generation & PROVISIONAL_GEN_BIT) != 0

    @always_inline
    def provisional_lane(self) -> Int:
        return Int(self.generation & ~PROVISIONAL_GEN_BIT)

    @always_inline
    def to_bits(self) -> UInt64:
        return (UInt64(self.generation) << 32) | UInt64(self.index)

    @staticmethod
    @always_inline
    def from_bits(bits: UInt64) -> Self:
        return Self(UInt32(bits & 0xFFFFFFFF), UInt32(bits >> 32))

    @always_inline
    def to_entity_handle(self, archetype_hint: Int = 0) -> EntityHandle:
        """Losslessly converts `flint.ecs.Entity` into `flint.core.ecs.EntityHandle`."""
        if not self.valid():
            return EntityHandle()
        if self.is_provisional():
            var arch_p = archetype_hint if archetype_hint > 0 else (self.provisional_lane() + 1)
            return EntityHandle(arch_p, Int(self.index), ECB_PROV_GEN)
        var arch = archetype_hint if archetype_hint > 0 else 1
        return EntityHandle(arch, Int(self.index), self.generation - 1)

    @staticmethod
    @always_inline
    def from_entity_handle(handle: EntityHandle) -> Self:
        """Losslessly converts `flint.core.ecs.EntityHandle` into `flint.ecs.Entity`."""
        if not handle.valid():
            return Self()
        if handle.generation == ECB_PROV_GEN:
            var lane = max(0, handle.archetype - 1)
            return Self(UInt32(handle.row), PROVISIONAL_GEN_BIT | UInt32(lane))
        return Self(UInt32(handle.row), handle.generation + 1)


comptime NULL_ENTITY: Entity = Entity()


trait Component(Copyable, Movable, Deinitable):
    """Trait implemented by all Flint v5 ECS components."""
    comptime COMPONENT_ID: UInt64
    comptime STORAGE_MODE: Int


@always_inline
def _ctz64(val: UInt64) -> Int:
    if val == 0:
        return 64
    var v = val
    var n = 0
    if (v & 0xFFFFFFFF) == 0:
        n += 32
        v >>= 32
    if (v & 0xFFFF) == 0:
        n += 16
        v >>= 16
    if (v & 0xFF) == 0:
        n += 8
        v >>= 8
    if (v & 0xF) == 0:
        n += 4
        v >>= 4
    if (v & 0x3) == 0:
        n += 2
        v >>= 2
    if (v & 0x1) == 0:
        n += 1
    return n


@always_inline
def _popcnt64(val: UInt64) -> Int:
    var x = val
    x = x - ((x >> 1) & UInt64(0x5555555555555555))
    x = (x & UInt64(0x3333333333333333)) + ((x >> 2) & UInt64(0x3333333333333333))
    x = (x + (x >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
    return Int((x * UInt64(0x0101010101010101)) >> 56)


struct ComponentBitSet(Copyable, Movable, Equatable):
    """Hybrid inline-256 + dynamic overflow bitset for unbounded component signatures and schedule conflicts."""
    var w0: UInt64
    var w1: UInt64
    var w2: UInt64
    var w3: UInt64
    var overflow: List[UInt64]

    def __init__(out self):
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.overflow = List[UInt64]()

    def __init__(out self, w0: UInt64, w1: UInt64, w2: UInt64, w3: UInt64):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.overflow = List[UInt64]()

    def __init__(out self, *, copy: Self):
        self.w0 = copy.w0
        self.w1 = copy.w1
        self.w2 = copy.w2
        self.w3 = copy.w3
        self.overflow = copy.overflow.copy()

    @always_inline
    def _trim(mut self):
        while len(self.overflow) > 0 and self.overflow[len(self.overflow) - 1] == 0:
            _ = self.overflow.pop()

    @always_inline
    def trimmed_word_count(self) -> Int:
        var n = len(self.overflow)
        while n > 0:
            if self.overflow[n - 1] != 0:
                return 4 + n
            n -= 1
        if self.w3 != 0:
            return 4
        if self.w2 != 0:
            return 3
        if self.w1 != 0:
            return 2
        if self.w0 != 0:
            return 1
        return 0

    @always_inline
    def word_at(self, idx: Int) -> UInt64:
        if idx == 0:
            return self.w0
        if idx == 1:
            return self.w1
        if idx == 2:
            return self.w2
        if idx == 3:
            return self.w3
        var ov = idx - 4
        if ov >= 0 and ov < len(self.overflow):
            return self.overflow[ov]
        return 0

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        if (
            self.w0 != other.w0
            or self.w1 != other.w1
            or self.w2 != other.w2
            or self.w3 != other.w3
        ):
            return False
        var max_ov = max(len(self.overflow), len(other.overflow))
        for i in range(max_ov):
            var a = self.overflow[i] if i < len(self.overflow) else UInt64(0)
            var b = other.overflow[i] if i < len(other.overflow) else UInt64(0)
            if a != b:
                return False
        return True

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    def is_empty(self) -> Bool:
        if (self.w0 | self.w1 | self.w2 | self.w3) != 0:
            return False
        for i in range(len(self.overflow)):
            if self.overflow[i] != 0:
                return False
        return True

    @always_inline
    def set(mut self, bit: Int):
        if bit < 0:
            return
        var word = bit >> 6
        var shift = UInt64(bit & 63)
        var mask = UInt64(1) << shift
        if word == 0:
            self.w0 |= mask
        elif word == 1:
            self.w1 |= mask
        elif word == 2:
            self.w2 |= mask
        elif word == 3:
            self.w3 |= mask
        else:
            var ov = word - 4
            while len(self.overflow) <= ov:
                self.overflow.append(0)
            self.overflow[ov] |= mask

    @always_inline
    def clear(mut self, bit: Int):
        if bit < 0:
            return
        var word = bit >> 6
        var shift = UInt64(bit & 63)
        var mask = ~(UInt64(1) << shift)
        if word == 0:
            self.w0 &= mask
        elif word == 1:
            self.w1 &= mask
        elif word == 2:
            self.w2 &= mask
        elif word == 3:
            self.w3 &= mask
        else:
            var ov = word - 4
            if ov < len(self.overflow):
                self.overflow[ov] &= mask
                self._trim()

    @always_inline
    def clear_all(mut self):
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.overflow.clear()

    @always_inline
    def test(self, bit: Int) -> Bool:
        if bit < 0:
            return False
        var word = bit >> 6
        var shift = UInt64(bit & 63)
        if word == 0:
            return ((self.w0 >> shift) & 1) != 0
        if word == 1:
            return ((self.w1 >> shift) & 1) != 0
        if word == 2:
            return ((self.w2 >> shift) & 1) != 0
        if word == 3:
            return ((self.w3 >> shift) & 1) != 0
        var ov = word - 4
        if ov < len(self.overflow):
            return ((self.overflow[ov] >> shift) & 1) != 0
        return False

    @always_inline
    def has(self, bit: Int) -> Bool:
        return self.test(bit)

    @always_inline
    def with_bit(self, bit: Int) -> Self:
        var out = self.copy()
        out.set(bit)
        return out^

    @always_inline
    def without_bit(self, bit: Int) -> Self:
        var out = self.copy()
        out.clear(bit)
        return out^

    @always_inline
    def union(self, other: Self) -> Self:
        var out = Self(
            self.w0 | other.w0,
            self.w1 | other.w1,
            self.w2 | other.w2,
            self.w3 | other.w3,
        )
        var max_ov = max(len(self.overflow), len(other.overflow))
        if max_ov > 0:
            out.overflow = List[UInt64](capacity=max_ov)
            for i in range(max_ov):
                var a = self.overflow[i] if i < len(self.overflow) else UInt64(0)
                var b = other.overflow[i] if i < len(other.overflow) else UInt64(0)
                out.overflow.append(a | b)
            out._trim()
        return out^

    @always_inline
    def intersection(self, other: Self) -> Self:
        var out = Self(
            self.w0 & other.w0,
            self.w1 & other.w1,
            self.w2 & other.w2,
            self.w3 & other.w3,
        )
        var min_ov = min(len(self.overflow), len(other.overflow))
        if min_ov > 0:
            out.overflow = List[UInt64](capacity=min_ov)
            for i in range(min_ov):
                out.overflow.append(self.overflow[i] & other.overflow[i])
            out._trim()
        return out^

    @always_inline
    def without(self, other: Self) -> Self:
        var out = Self(
            self.w0 & ~other.w0,
            self.w1 & ~other.w1,
            self.w2 & ~other.w2,
            self.w3 & ~other.w3,
        )
        var n_ov = len(self.overflow)
        if n_ov > 0:
            out.overflow = List[UInt64](capacity=n_ov)
            for i in range(n_ov):
                var b = other.overflow[i] if i < len(other.overflow) else UInt64(0)
                out.overflow.append(self.overflow[i] & ~b)
            out._trim()
        return out^

    @always_inline
    def __or__(self, other: Self) -> Self:
        return self.union(other)

    @always_inline
    def __and__(self, other: Self) -> Self:
        return self.intersection(other)

    @always_inline
    def contains_all(self, required: Self) -> Bool:
        if (
            (self.w0 & required.w0) != required.w0
            or (self.w1 & required.w1) != required.w1
            or (self.w2 & required.w2) != required.w2
            or (self.w3 & required.w3) != required.w3
        ):
            return False
        for i in range(len(required.overflow)):
            var rw = required.overflow[i]
            if rw != 0:
                var sw = self.overflow[i] if i < len(self.overflow) else UInt64(0)
                if (sw & rw) != rw:
                    return False
        return True

    @always_inline
    def intersects(self, other: Self) -> Bool:
        if (
            (self.w0 & other.w0)
            | (self.w1 & other.w1)
            | (self.w2 & other.w2)
            | (self.w3 & other.w3)
        ) != 0:
            return True
        var min_ov = min(len(self.overflow), len(other.overflow))
        for i in range(min_ov):
            if (self.overflow[i] & other.overflow[i]) != 0:
                return True
        return False

    def popcount(self) -> Int:
        var cnt = _popcnt64(self.w0) + _popcnt64(self.w1) + _popcnt64(self.w2) + _popcnt64(self.w3)
        for i in range(len(self.overflow)):
            cnt += _popcnt64(self.overflow[i])
        return cnt

    def first_set(self) -> Int:
        if self.w0 != 0:
            return _ctz64(self.w0)
        if self.w1 != 0:
            return 64 + _ctz64(self.w1)
        if self.w2 != 0:
            return 128 + _ctz64(self.w2)
        if self.w3 != 0:
            return 192 + _ctz64(self.w3)
        for i in range(len(self.overflow)):
            if self.overflow[i] != 0:
                return (4 + i) * 64 + _ctz64(self.overflow[i])
        return -1

    def hash64(self) -> UInt64:
        var h = mix64(self.w0 ^ UInt64(0x9E3779B97F4A7C15))
        h = mix64(h ^ self.w1)
        h = mix64(h ^ self.w2)
        h = mix64(h ^ self.w3)
        var wc = self.trimmed_word_count()
        for i in range(4, wc):
            h = mix64(h ^ self.word_at(i))
        return h


comptime BitSet256 = ComponentBitSet


# ---------------------------------------------------------------------------
# Type-erased component lifecycle & dynamic per-worker implicit order
# ---------------------------------------------------------------------------

comptime _DropFn = def(Int) thin -> None
comptime _CopyInitFn = def(Int, Int) thin -> None
comptime _MoveInitFn = def(Int, Int) thin -> None
comptime _BinWriteFn = def(Int, Int) thin -> None
comptime _BinReadFn = def(Int, Int) thin -> Bool


def _raw_emplace[T: Movable](addr: Int, var val: T):
    var p = Pointer[T, MutAnyOrigin](unsafe_from_address=addr)
    __get_address_as_uninit_lvalue(p._mlir_value) = val^


def _raw_copy_init[T: Copyable](dst_addr: Int, src_addr: Int):
    """Initializes uninitialized `dst_addr` as a copy of live `src_addr`."""
    var src = Pointer[T, ImmutAnyOrigin](unsafe_from_address=src_addr)
    var dst = Pointer[T, MutAnyOrigin](unsafe_from_address=dst_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = src[unsafe_offset=0].copy()


def _raw_move_init[T: Movable](dst_addr: Int, src_addr: Int):
    """Relocates live `src_addr` into uninitialized `dst_addr` without running `T.__deinit__`."""
    var src = Pointer[T, MutAnyOrigin](unsafe_from_address=src_addr)
    var dst = Pointer[T, MutAnyOrigin](unsafe_from_address=dst_addr)
    var moved_val = __get_address_as_owned_value(src._mlir_value)
    __get_address_as_uninit_lvalue(dst._mlir_value) = moved_val^


def _raw_destroy[T: Movable & Deinitable](addr: Int):
    """Runs `T.__deinit__` in-place on live `addr`."""
    var p = Pointer[T, MutAnyOrigin](unsafe_from_address=addr)
    var owned = __get_address_as_owned_value(p._mlir_value)
    _ = owned^


def _noop_drop(_addr: Int):
    pass


def _noop_copy_init(_dst: Int, _src: Int):
    pass


def _noop_move_init(_dst: Int, _src: Int):
    pass


def _noop_bin_write(_c: Int, _w: Int):
    pass


def _noop_bin_read(_c: Int, _r: Int) -> Bool:
    return True


def _noop_cmd_apply(_w: Int, _e: Entity, _p: Int):
    pass


@fieldwise_init
struct ComponentInfo(Copyable, ImplicitlyCopyable, Movable):
    """Metadata and type-erased lifecycle hooks for a registered component type."""
    var type_id: Int
    var component_hash: UInt64
    var byte_size: Int
    var alignment: Int
    var storage_mode: Int
    var is_trivial_pod: Bool
    var copy_init_fn: _CopyInitFn
    var move_init_fn: _MoveInitFn
    var destroy_fn: _DropFn
    var has_custom_bin: Bool
    var bin_write_fn: _BinWriteFn
    var bin_read_fn: _BinReadFn

    @always_inline
    def drop_at(self, addr: Int):
        if self.byte_size > 0 and not self.is_trivial_pod and addr != 0:
            var dfn = self.destroy_fn
            dfn(addr)

    @always_inline
    def copy_at(self, dst_addr: Int, src_addr: Int):
        if self.byte_size > 0 and dst_addr != 0 and src_addr != 0:
            if self.is_trivial_pod:
                _ = external_call["memcpy", Int](dst_addr, src_addr, self.byte_size)
            else:
                var cfn = self.copy_init_fn
                cfn(dst_addr, src_addr)

    @always_inline
    def move_at(self, dst_addr: Int, src_addr: Int):
        if self.byte_size > 0 and dst_addr != 0 and src_addr != 0:
            if self.is_trivial_pod:
                _ = external_call["memcpy", Int](dst_addr, src_addr, self.byte_size)
            else:
                var mfn = self.move_init_fn
                mfn(dst_addr, src_addr)

    @always_inline
    def write_custom_bin(self, comp_addr: Int, writer_addr: Int):
        self.bin_write_fn(comp_addr, writer_addr)

    @always_inline
    def read_custom_bin(self, comp_addr: Int, reader_addr: Int) -> Bool:
        return self.bin_read_fn(comp_addr, reader_addr)


def _ecs_rt_on_exit(ctrl: Int) abi("C"):
    if ctrl == 0:
        return
    var base_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)
    while True:
        var cur = base_ptr[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
        if cur == 0:
            break
        var expected = cur
        if base_ptr[unsafe_offset=0].compare_exchange[
            failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
        ](expected, Int64(0)):
            external_call["free", NoneType](Int(cur))
            break
    external_call["free", NoneType](ctrl)


def _init_ecs_rt_ctrl() -> Int:
    var ctrl = external_call["aligned_alloc", Int](Int(64), Int(64))
    if ctrl == 0:
        return 0
    var p = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=ctrl)
    for i in range(8):
        p[unsafe_offset=i] = 0
    _ = external_call["__cxa_atexit", Int32](_ecs_rt_on_exit, ctrl, Int(0))
    return ctrl


comptime _ECS_RT_GLOBAL = _Global["FLINT_ECS_RT_CTRL", _init_ecs_rt_ctrl]


@always_inline
def _ecs_rt_ctrl_addr() -> Int:
    var slot = external_call["KGEN_CompilerRT_GetGlobalOrNull", Int](
        Int(_ECS_RT_CTRL_KEY.unsafe_ptr()), _ECS_RT_CTRL_KEY.byte_length()
    )
    if slot == 0:
        return 0
    return Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=slot)[unsafe_offset=0]


@always_inline
def _ecs_rt_ctrl_addr_or_create() -> Int:
    try:
        return _ECS_RT_GLOBAL.get_or_create_ptr()[unsafe_offset=0]
    except:
        return 0


@always_inline
def _ecs_runtime_base() -> Int:
    var ctrl = _ecs_rt_ctrl_addr()
    if ctrl == 0:
        return 0
    return Int(
        Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)[
            unsafe_offset=0
        ].load[ordering=Ordering.ACQUIRE]()
    )


@always_inline
def _rt_word_at(base: Int, offset: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
    return Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=base + offset)


@always_inline
def _ecs_runtime_ready() -> Bool:
    var base = _ecs_runtime_base()
    if base == 0:
        return False
    var magic = Int(
        _rt_word_at(base, _RT_MAGIC_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
    )
    if magic != _ECS_RUNTIME_MAGIC:
        return False
    var pid = Int(external_call["getpid", Int32]())
    return (
        Int(_rt_word_at(base, _RT_PID_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
        == pid
    )


def _ecs_runtime_ensure() -> Bool:
    var ctrl = _ecs_rt_ctrl_addr_or_create()
    if ctrl == 0:
        return False
    var cur_pid = Int(external_call["getpid", Int32]())
    var base_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)
    var existing = Int(base_ptr[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
    if existing != 0:
        var magic = Int(
            _rt_word_at(existing, _RT_MAGIC_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
        )
        if magic == _ECS_RUNTIME_MAGIC:
            var rec_pid = Int(
                _rt_word_at(existing, _RT_PID_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
            )
            if rec_pid == cur_pid:
                return True
            # Inherited across fork(): reset process-local state cleanly for child PID
            _ = external_call["memset", Int](existing, Int32(0), _ECS_RUNTIME_BYTES)
            _rt_word_at(existing, _RT_PID_OFF)[unsafe_offset=0].store[ordering=Ordering.RELEASE](
                Int64(cur_pid)
            )
            _rt_word_at(existing, _RT_MAGIC_OFF)[unsafe_offset=0].store[ordering=Ordering.RELEASE](
                Int64(_ECS_RUNTIME_MAGIC)
            )
            return True
    var blk = external_call["aligned_alloc", Int](Int(64), Int(_ECS_RUNTIME_BYTES))
    if blk == 0:
        return False
    _ = external_call["memset", Int](blk, Int32(0), _ECS_RUNTIME_BYTES)
    _rt_word_at(blk, _RT_PID_OFF)[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int64(cur_pid))
    _rt_word_at(blk, _RT_MAGIC_OFF)[unsafe_offset=0].store[ordering=Ordering.RELEASE](
        Int64(_ECS_RUNTIME_MAGIC)
    )
    var expected = Int64(0)
    if not base_ptr[unsafe_offset=0].compare_exchange[
        failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
    ](expected, Int64(blk)):
        external_call["free", NoneType](blk)
    return _ecs_runtime_ready()


def ecs_runtime_shutdown():
    """Frees the dynamic ECS runtime block early (idempotent with __cxa_atexit)."""
    var ctrl = _ecs_rt_ctrl_addr()
    if ctrl == 0:
        return
    var base_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)
    while True:
        var cur = base_ptr[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
        if cur == 0:
            return
        var expected = cur
        if base_ptr[unsafe_offset=0].compare_exchange[
            failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
        ](expected, Int64(0)):
            external_call["free", NoneType](Int(cur))
            return


@always_inline
def _lane_order_slot_from_base(base: Int, lane: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
    var slot = lane & (_WORKER_ORDER_SLOTS - 1)
    return _rt_word_at(base, _RT_WORKER_ORDER_OFF + slot * 64)


@always_inline
def _set_lane_implicit_order(lane: Int, implicit_order: UInt64):
    """Stores 1-based implicit entity order (`UInt64(order) + 1`, or `0` when inactive) in a
    dedicated 64-byte cache-line-padded slot for `lane`."""
    if not _ecs_runtime_ready():
        if not _ecs_runtime_ensure():
            return
    var base = _ecs_runtime_base()
    if base == 0:
        return
    _lane_order_slot_from_base(base, lane)[unsafe_offset=0].store(Int64(implicit_order))


@always_inline
def _get_lane_implicit_order(lane: Int) -> UInt64:
    if not _ecs_runtime_ready():
        return 0
    var base = _ecs_runtime_base()
    if base == 0:
        return 0
    return UInt64(_lane_order_slot_from_base(base, lane)[unsafe_offset=0].load())


# ---------------------------------------------------------------------------
# Hybrid Storage 1: 64-Byte Aligned, 8-SIMD-Lane-Multiple ArchetypeChunk
# ---------------------------------------------------------------------------

struct ArchetypeChunk(Movable):
    """64-byte aligned columnar table chunk with per-8-row SIMD changed_ticks."""
    var capacity: Int
    var count: Int
    var slab_ptr: Int
    var slab_bytes: Int
    var entities: List[Entity]
    var comp_type_ids: List[Int]
    var col_offsets: List[Int]
    var col_elem_sizes: List[Int]
    var changed_ticks: List[UInt32]
    var simd_groups: Int
    var charge: MemCharge

    def __init__(
        out self,
        comp_type_ids: List[Int],
        col_elem_sizes: List[Int],
        target_rows: Int = DEFAULT_CHUNK_ROWS,
    ):
        var cap = ((max(SIMD_LANES, target_rows) + (SIMD_LANES - 1)) // SIMD_LANES) * SIMD_LANES
        self.capacity = cap
        self.count = 0
        self.simd_groups = cap // SIMD_LANES
        self.entities = List[Entity](capacity=cap)
        self.comp_type_ids = comp_type_ids.copy()
        self.col_elem_sizes = col_elem_sizes.copy()
        var num_cols = len(comp_type_ids)
        self.col_offsets = List[Int](capacity=num_cols)

        var offset = 0
        for c in range(num_cols):
            offset = (offset + (CHUNK_ALIGN - 1)) & ~(CHUNK_ALIGN - 1)
            self.col_offsets.append(offset)
            offset += cap * col_elem_sizes[c]
        var total_bytes = max(CHUNK_ALIGN, (offset + (CHUNK_ALIGN - 1)) & ~(CHUNK_ALIGN - 1))
        self.slab_bytes = total_bytes
        self.slab_ptr = external_call["aligned_alloc", Int](CHUNK_ALIGN, total_bytes)
        if self.slab_ptr != 0:
            _ = external_call["memset", Int](self.slab_ptr, Int32(0), total_bytes)
        self.changed_ticks = List[UInt32](length=max(1, num_cols) * self.simd_groups, fill=0)
        self.charge = MemCharge(MEM_ENTITIES, self.slab_bytes)

    def __deinit__(deinit self):
        if self.slab_ptr != 0:
            external_call["free", NoneType](self.slab_ptr)

    @always_inline
    def keep_alive(self):
        """Anchors this chunk's slab and entity list past a parallel `JobPool` barrier."""
        var anchor = self.slab_ptr ^ Int(self.entities.unsafe_ptr())
        _ = anchor

    @always_inline
    def is_full(self) -> Bool:
        return self.count >= self.capacity

    @always_inline
    def column_ptr(self, col_idx: Int) -> Int:
        if self.slab_ptr == 0 or col_idx < 0 or col_idx >= len(self.col_offsets):
            return 0
        return self.slab_ptr + self.col_offsets[col_idx]

    @always_inline
    def cell_ptr(self, col_idx: Int, row: Int) -> Int:
        var base = self.column_ptr(col_idx)
        if base == 0:
            return 0
        return base + row * self.col_elem_sizes[col_idx]

    @always_inline
    def mark_changed(mut self, col_idx: Int, row: Int, tick: UInt32):
        if col_idx >= 0 and col_idx < len(self.comp_type_ids) and row >= 0 and row < self.capacity:
            self.changed_ticks[col_idx * self.simd_groups + (row >> 3)] = tick

    @always_inline
    def mark_all_groups_changed(mut self, col_idx: Int, tick: UInt32):
        if col_idx >= 0 and col_idx < len(self.comp_type_ids):
            var active_groups = (self.count + (SIMD_LANES - 1)) >> 3
            var base = col_idx * self.simd_groups
            for g in range(active_groups):
                self.changed_ticks[base + g] = tick

    @always_inline
    def group_changed_tick(self, col_idx: Int, simd_group: Int) -> UInt32:
        if col_idx < 0 or col_idx >= len(self.comp_type_ids) or simd_group < 0 or simd_group >= self.simd_groups:
            return 0
        return self.changed_ticks[col_idx * self.simd_groups + simd_group]

    @always_inline
    def row_changed_since(self, col_idx: Int, row: Int, since_tick: UInt32) -> Bool:
        if since_tick == 0:
            return True
        return self.group_changed_tick(col_idx, row >> 3) > since_tick

    @always_inline
    def any_group_changed_since(self, col_idx: Int, since_tick: UInt32) -> Bool:
        if since_tick == 0:
            return True
        if col_idx < 0 or col_idx >= len(self.comp_type_ids) or self.count <= 0:
            return False
        var active_groups = (self.count + (SIMD_LANES - 1)) >> 3
        var base = col_idx * self.simd_groups
        for g in range(active_groups):
            if self.changed_ticks[base + g] > since_tick:
                return True
        return False

    def push_uninit_row(mut self, entity: Entity, tick: UInt32) -> Int:
        var row = self.count
        self.entities.append(entity)
        self.count = row + 1
        var grp = row >> 3
        for c in range(len(self.comp_type_ids)):
            self.changed_ticks[c * self.simd_groups + grp] = tick
        return row

    def swap_remove_no_drop(
        mut self, row: Int, tick: UInt32, components: List[ComponentInfo]
    ) -> Entity:
        """Relocates last row into `row` via `move_at` (caller already dropped or moved `row`'s cells).
        Returns the entity that was moved into `row`, or `NULL_ENTITY` if `row` was last."""
        var last = self.count - 1
        var moved = NULL_ENTITY
        if row < last:
            moved = self.entities[last]
            self.entities[row] = moved
            var grp = row >> 3
            for c in range(len(self.comp_type_ids)):
                var tid = self.comp_type_ids[c]
                var info = components[tid]
                var esz = self.col_elem_sizes[c]
                if esz > 0:
                    var base = self.slab_ptr + self.col_offsets[c]
                    info.move_at(base + row * esz, base + last * esz)
                self.changed_ticks[c * self.simd_groups + grp] = tick
        _ = self.entities.pop()
        self.count = last
        return moved


struct Archetype(Movable):
    """Table archetype with cached `ArchetypeGraph` O(1) add/remove transition edges."""
    var id: Int
    var table_mask: ComponentBitSet
    var comp_type_ids: List[Int]
    var col_elem_sizes: List[Int]
    var type_to_col: List[Int32]
    var add_edges: List[Int32]
    var remove_edges: List[Int32]
    var chunks: List[ArchetypeChunk]
    var chunk_rows: Int

    def __init__(
        out self,
        id: Int,
        table_mask: ComponentBitSet,
        comp_type_ids: List[Int],
        col_elem_sizes: List[Int],
        chunk_rows: Int = DEFAULT_CHUNK_ROWS,
    ):
        self.id = id
        self.table_mask = table_mask.copy()
        self.comp_type_ids = comp_type_ids.copy()
        self.col_elem_sizes = col_elem_sizes.copy()
        var max_tid = -1
        for c in range(len(comp_type_ids)):
            if comp_type_ids[c] > max_tid:
                max_tid = comp_type_ids[c]
        self.type_to_col = List[Int32](length=max(0, max_tid + 1), fill=-1)
        for c in range(len(comp_type_ids)):
            self.type_to_col[comp_type_ids[c]] = Int32(c)
        self.add_edges = List[Int32]()
        self.remove_edges = List[Int32]()
        self.chunks = List[ArchetypeChunk]()
        self.chunk_rows = ((max(SIMD_LANES, chunk_rows) + (SIMD_LANES - 1)) // SIMD_LANES) * SIMD_LANES

    @always_inline
    def _ensure_add_edge_cap(mut self, tid: Int):
        while len(self.add_edges) <= tid:
            self.add_edges.append(-1)

    @always_inline
    def _ensure_remove_edge_cap(mut self, tid: Int):
        while len(self.remove_edges) <= tid:
            self.remove_edges.append(-1)

    @always_inline
    def col_index(self, type_id: Int) -> Int:
        if type_id < 0 or type_id >= len(self.type_to_col):
            return -1
        return Int(self.type_to_col[type_id])

    def ensure_chunk_with_space(mut self) -> Int:
        for i in range(len(self.chunks)):
            if not self.chunks[i].is_full():
                return i
        var idx = len(self.chunks)
        self.chunks.append(ArchetypeChunk(self.comp_type_ids, self.col_elem_sizes, self.chunk_rows))
        return idx

    def entity_count(self) -> Int:
        var n = 0
        for i in range(len(self.chunks)):
            n += self.chunks[i].count
        return n


# ---------------------------------------------------------------------------
# Hybrid Storage 2: Paged SparseSet for STORAGE_SPARSE tags & rare components
# ---------------------------------------------------------------------------

struct SparseSet(Movable):
    """Paged sparse-set storage (`1024`-entity pages) for O(1) tag/volatile components."""
    var type_id: Int
    var elem_size: Int
    var pages: List[List[Int32]]
    var dense_entities: List[Entity]
    var changed_ticks: List[UInt32]
    var data_ptr: Int
    var data_capacity: Int
    var charge: MemCharge

    def __init__(out self, type_id: Int, elem_size: Int):
        self.type_id = type_id
        self.elem_size = max(0, elem_size)
        self.pages = List[List[Int32]]()
        self.dense_entities = List[Entity]()
        self.changed_ticks = List[UInt32]()
        self.data_ptr = 0
        self.data_capacity = 0
        self.charge = MemCharge(MEM_ENTITIES, 0)

    def __deinit__(deinit self):
        if self.data_ptr != 0:
            external_call["free", NoneType](self.data_ptr)

    @always_inline
    def keep_alive(self):
        var anchor = self.data_ptr ^ Int(self.dense_entities.unsafe_ptr())
        _ = anchor

    def _ensure_data_capacity(mut self, needed: Int, info: ComponentInfo):
        if self.elem_size == 0 or needed <= self.data_capacity:
            return
        var new_cap = max(64, self.data_capacity * 2)
        while new_cap < needed:
            new_cap *= 2
        var raw_bytes = ((new_cap * self.elem_size) + (CHUNK_ALIGN - 1)) & ~(CHUNK_ALIGN - 1)
        var new_ptr = external_call["aligned_alloc", Int](CHUNK_ALIGN, raw_bytes)
        if new_ptr != 0:
            _ = external_call["memset", Int](new_ptr, Int32(0), raw_bytes)
            if self.data_ptr != 0:
                var used_count = len(self.dense_entities)
                if used_count > 0:
                    if info.is_trivial_pod:
                        _ = external_call["memcpy", Int](
                            new_ptr, self.data_ptr, used_count * self.elem_size
                        )
                    else:
                        for i in range(used_count):
                            info.move_at(
                                new_ptr + i * self.elem_size,
                                self.data_ptr + i * self.elem_size,
                            )
                external_call["free", NoneType](self.data_ptr)
            self.data_ptr = new_ptr
            self.data_capacity = new_cap
            self.charge.set(raw_bytes)

    def _ensure_page(mut self, page_idx: Int):
        while len(self.pages) <= page_idx:
            self.pages.append(List[Int32]())
        if len(self.pages[page_idx]) == 0:
            self.pages[page_idx] = List[Int32](length=SPARSE_PAGE_SIZE, fill=-1)

    @always_inline
    def dense_index_of(self, entity_index: UInt32) -> Int:
        if entity_index == INVALID_ENTITY_INDEX:
            return -1
        var idx = Int(entity_index)
        var page = idx >> SPARSE_PAGE_BITS
        if page < 0 or page >= len(self.pages):
            return -1
        if len(self.pages[page]) == 0:
            return -1
        return Int(self.pages[page][idx & SPARSE_PAGE_MASK])

    @always_inline
    def contains(self, entity: Entity) -> Bool:
        var di = self.dense_index_of(entity.index)
        return di >= 0 and di < len(self.dense_entities) and self.dense_entities[di] == entity

    @always_inline
    def value_ptr(self, dense_idx: Int) -> Int:
        if self.elem_size == 0:
            return Int(Pointer(to=self.dense_entities))
        if self.data_ptr == 0 or dense_idx < 0 or dense_idx >= len(self.dense_entities):
            return 0
        return self.data_ptr + dense_idx * self.elem_size

    def insert_uninit(mut self, entity: Entity, info: ComponentInfo, tick: UInt32) -> Int:
        """Allocates or overwrites slot for `entity`; returns address to emplace new `C`."""
        var idx = Int(entity.index)
        var page = idx >> SPARSE_PAGE_BITS
        self._ensure_page(page)
        var slot_in_page = idx & SPARSE_PAGE_MASK
        var existing = Int(self.pages[page][slot_in_page])
        if existing >= 0 and existing < len(self.dense_entities):
            if self.elem_size > 0:
                info.drop_at(self.data_ptr + existing * self.elem_size)
            self.dense_entities[existing] = entity
            self.changed_ticks[existing] = tick
            return self.value_ptr(existing)
        var di = len(self.dense_entities)
        self._ensure_data_capacity(di + 1, info)
        self.pages[page][slot_in_page] = Int32(di)
        self.dense_entities.append(entity)
        self.changed_ticks.append(tick)
        return self.value_ptr(di)

    def remove(mut self, entity_index: UInt32, info: ComponentInfo) -> Bool:
        var di = self.dense_index_of(entity_index)
        if di < 0:
            return False
        var idx = Int(entity_index)
        self.pages[idx >> SPARSE_PAGE_BITS][idx & SPARSE_PAGE_MASK] = -1
        var last = len(self.dense_entities) - 1
        if self.elem_size > 0:
            info.drop_at(self.data_ptr + di * self.elem_size)
        if di < last:
            var moved_ent = self.dense_entities[last]
            self.dense_entities[di] = moved_ent
            self.changed_ticks[di] = self.changed_ticks[last]
            if self.elem_size > 0:
                info.move_at(
                    self.data_ptr + di * self.elem_size,
                    self.data_ptr + last * self.elem_size,
                )
            var midx = Int(moved_ent.index)
            self.pages[midx >> SPARSE_PAGE_BITS][midx & SPARSE_PAGE_MASK] = Int32(di)
        _ = self.dense_entities.pop()
        _ = self.changed_ticks.pop()
        return True

    def clear_with_drop(mut self, info: ComponentInfo):
        if self.elem_size > 0 and self.data_ptr != 0:
            for i in range(len(self.dense_entities)):
                info.drop_at(self.data_ptr + i * self.elem_size)
        for p in range(len(self.pages)):
            if len(self.pages[p]) > 0:
                for k in range(SPARSE_PAGE_SIZE):
                    self.pages[p][k] = -1
        self.dense_entities.clear()
        self.changed_ticks.clear()


# ---------------------------------------------------------------------------
# Per-Worker Lock-Free ParallelCommands + Deterministic Playback
# ---------------------------------------------------------------------------

comptime _CmdApplyInsertFn = def(Int, Entity, Int) thin -> None


@fieldwise_init
struct _RecordedCmd(Copyable, ImplicitlyCopyable, Movable):
    var op: UInt8
    var lane_id: Int
    var seq: Int
    var sort_key: UInt64
    var entity: Entity
    var component_hash: UInt64
    var payload_offset: Int
    var payload_size: Int
    var is_trivial_pod: Bool
    var apply_fn: _CmdApplyInsertFn
    var move_fn: _MoveInitFn
    var drop_fn: _DropFn


struct CommandLane(Movable):
    """Single-worker lock-free command buffer lane with inline payload arena."""
    var lane_id: Int
    var next_seq: Int
    var next_prov_index: UInt32
    var last_implicit_order: UInt64
    var implicit_sub_seq: UInt64
    var last_explicit_order: Int
    var explicit_sub_seq: UInt64
    var cmds: List[_RecordedCmd]
    var payload_ptr: Int
    var payload_used: Int
    var payload_cap: Int

    def __init__(out self, lane_id: Int):
        self.lane_id = lane_id
        self.next_seq = 0
        self.next_prov_index = 0
        self.last_implicit_order = 0
        self.implicit_sub_seq = 0
        self.last_explicit_order = -2
        self.explicit_sub_seq = 0
        self.cmds = List[_RecordedCmd]()
        self.payload_ptr = 0
        self.payload_used = 0
        self.payload_cap = 0

    def __deinit__(deinit self):
        self._drop_unconsumed_payloads()
        if self.payload_ptr != 0:
            external_call["free", NoneType](self.payload_ptr)

    def _reset_counters(mut self):
        self.cmds.clear()
        self.payload_used = 0
        self.next_seq = 0
        self.next_prov_index = 0
        self.last_implicit_order = 0
        self.implicit_sub_seq = 0
        self.last_explicit_order = -2
        self.explicit_sub_seq = 0

    def _drop_unconsumed_payloads(mut self):
        if self.payload_ptr != 0:
            for i in range(len(self.cmds)):
                var cmd = self.cmds[i]
                if cmd.op == CMD_INSERT and cmd.payload_size > 0 and not cmd.is_trivial_pod:
                    var dfn = cmd.drop_fn
                    dfn(self.payload_ptr + cmd.payload_offset)
        self._reset_counters()

    @always_inline
    def _compute_sort_key(mut self, order_key: Int, seq: Int) -> UInt64:
        if order_key >= 0:
            if order_key == self.last_explicit_order:
                self.explicit_sub_seq += 1
            else:
                self.last_explicit_order = order_key
                self.explicit_sub_seq = 0
            return ((UInt64(order_key) + 1) << 24) | (self.explicit_sub_seq & UInt64(0xFFFFFF))
        var imp = _get_lane_implicit_order(self.lane_id)
        if imp > 0:
            if imp == self.last_implicit_order:
                self.implicit_sub_seq += 1
            else:
                self.last_implicit_order = imp
                self.implicit_sub_seq = 0
            return (imp << 24) | (self.implicit_sub_seq & UInt64(0xFFFFFF))
        return (UInt64(1) << 61) | (UInt64(self.lane_id) << 32) | UInt64(seq & 0xFFFFFFFF)

    def _alloc_payload(mut self, byte_size: Int, align: Int) -> Int:
        if byte_size <= 0:
            return 0
        var a = max(8, align)
        var aligned_off = (self.payload_used + (a - 1)) & ~(a - 1)
        var needed = aligned_off + byte_size
        if needed > self.payload_cap:
            var new_cap = max(1024, self.payload_cap * 2)
            while new_cap < needed:
                new_cap *= 2
            new_cap = (new_cap + 63) & ~63
            var new_ptr = external_call["aligned_alloc", Int](64, new_cap)
            if new_ptr != 0:
                _ = external_call["memset", Int](new_ptr, Int32(0), new_cap)
                if self.payload_ptr != 0:
                    for i in range(len(self.cmds)):
                        var cmd = self.cmds[i]
                        if cmd.op == CMD_INSERT and cmd.payload_size > 0:
                            if cmd.is_trivial_pod:
                                _ = external_call["memcpy", Int](
                                    new_ptr + cmd.payload_offset,
                                    self.payload_ptr + cmd.payload_offset,
                                    cmd.payload_size,
                                )
                            else:
                                var mfn = cmd.move_fn
                                mfn(
                                    new_ptr + cmd.payload_offset,
                                    self.payload_ptr + cmd.payload_offset,
                                )
                    external_call["free", NoneType](self.payload_ptr)
                self.payload_ptr = new_ptr
                self.payload_cap = new_cap
        self.payload_used = needed
        return aligned_off


def _apply_cmd_insert_typed[C: Component](world_addr: Int, entity: Entity, payload_addr: Int):
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var p = Pointer[C, MutAnyOrigin](unsafe_from_address=payload_addr)
    var val = __get_address_as_owned_value(p._mlir_value)
    _ = world.insert[C](entity, val^)


struct ParallelCommands(Movable):
    """Lock-free per-worker (`MAX_WORKERS + 1` lanes) deferred ECS command buffers."""
    var lanes: List[CommandLane]

    def __init__(out self):
        self.lanes = List[CommandLane](capacity=MAX_COMMAND_LANES)
        for lane in range(MAX_COMMAND_LANES):
            self.lanes.append(CommandLane(lane))

    @always_inline
    def _clamp_lane(self, lane: Int) -> Int:
        if lane < 0:
            return 0
        if lane >= MAX_COMMAND_LANES:
            return lane % MAX_COMMAND_LANES
        return lane

    def spawn_empty(mut self, lane: Int, order_key: Int = -1) -> Entity:
        """Allocates a thread-local provisional `Entity` without atomic contention."""
        var l = self._clamp_lane(lane)
        ref cl = self.lanes[l]
        var pidx = cl.next_prov_index
        cl.next_prov_index += 1
        var seq = cl.next_seq
        cl.next_seq += 1
        var skey = cl._compute_sort_key(order_key, seq)
        var prov = Entity(pidx, PROVISIONAL_GEN_BIT | UInt32(l))
        cl.cmds.append(
            _RecordedCmd(
                CMD_SPAWN, l, seq, skey, prov, 0, 0, 0, True, _noop_cmd_apply, _noop_move_init, _noop_drop
            )
        )
        return prov

    def despawn(mut self, lane: Int, entity: Entity, order_key: Int = -1):
        var l = self._clamp_lane(lane)
        ref cl = self.lanes[l]
        var seq = cl.next_seq
        cl.next_seq += 1
        var skey = cl._compute_sort_key(order_key, seq)
        cl.cmds.append(
            _RecordedCmd(
                CMD_DESPAWN, l, seq, skey, entity, 0, 0, 0, True, _noop_cmd_apply, _noop_move_init, _noop_drop
            )
        )

    def insert[C: Component](mut self, lane: Int, entity: Entity, var component: C, order_key: Int = -1):
        comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
        var l = self._clamp_lane(lane)
        ref cl = self.lanes[l]
        var seq = cl.next_seq
        cl.next_seq += 1
        var skey = cl._compute_sort_key(order_key, seq)
        var bsz = size_of[C]()
        var off = cl._alloc_payload(bsz, align_of[C]())
        if bsz > 0:
            _raw_emplace[C](cl.payload_ptr + off, component^)
        else:
            _ = component^
        var mfn: _MoveInitFn = _noop_move_init
        var dfn: _DropFn = _noop_drop
        if not is_triv:
            mfn = _raw_move_init[C]
            dfn = _raw_destroy[C]
        cl.cmds.append(
            _RecordedCmd(
                CMD_INSERT,
                l,
                seq,
                skey,
                entity,
                C.COMPONENT_ID,
                off,
                bsz,
                is_triv,
                _apply_cmd_insert_typed[C],
                mfn,
                dfn,
            )
        )

    def remove[C: Component](mut self, lane: Int, entity: Entity, order_key: Int = -1):
        var l = self._clamp_lane(lane)
        ref cl = self.lanes[l]
        var seq = cl.next_seq
        cl.next_seq += 1
        var skey = cl._compute_sort_key(order_key, seq)
        cl.cmds.append(
            _RecordedCmd(
                CMD_REMOVE,
                l,
                seq,
                skey,
                entity,
                C.COMPONENT_ID,
                0,
                0,
                True,
                _noop_cmd_apply,
                _noop_move_init,
                _noop_drop,
            )
        )

    def pending_count(self) -> Int:
        var total = 0
        for l in range(MAX_COMMAND_LANES):
            total += len(self.lanes[l].cmds)
        return total

    def clear(mut self):
        for l in range(MAX_COMMAND_LANES):
            self.lanes[l]._drop_unconsumed_payloads()


# ---------------------------------------------------------------------------
# Double-Buffered Events[T] (2-frame retention)
# ---------------------------------------------------------------------------

struct Events[T: Copyable & Deinitable](Movable):
    """Double-buffered ECS event channel keeping events live across 2 frames."""
    var older: List[Self.T]
    var current: List[Self.T]
    var older_start: Int
    var current_start: Int
    var total_sent: Int

    def __init__(out self):
        self.older = List[Self.T]()
        self.current = List[Self.T]()
        self.older_start = 0
        self.current_start = 0
        self.total_sent = 0

    def send(mut self, var event: Self.T) -> Int:
        var id = self.total_sent
        self.current.append(event^)
        self.total_sent += 1
        return id

    def swap_buffers(mut self):
        """Advances frame boundary: drops events from 2 frames ago, promotes current -> older."""
        self.older.clear()
        var recycled = self.older^
        self.older = self.current^
        self.current = recycled^
        self.older_start = self.current_start
        self.current_start = self.total_sent

    def read_since(self, mut cursor: Int) -> List[Self.T]:
        """Reads all retained events since `cursor` and advances `cursor` to `total_sent`."""
        var out = List[Self.T]()
        var start = max(cursor, self.older_start)
        if start < self.current_start:
            var idx_a = start - self.older_start
            for i in range(idx_a, len(self.older)):
                out.append(self.older[i].copy())
            start = self.current_start
        if start < self.total_sent:
            var idx_b = start - self.current_start
            for i in range(idx_b, len(self.current)):
                out.append(self.current[i].copy())
        cursor = self.total_sent
        return out^

    def active_count(self) -> Int:
        return len(self.older) + len(self.current)

    def clear(mut self):
        self.older.clear()
        self.current.clear()
        self.older_start = self.total_sent
        self.current_start = self.total_sent


# ---------------------------------------------------------------------------
# EntityRecord & Parallel Iteration Slices
# ---------------------------------------------------------------------------

@fieldwise_init
struct EntityRecord(Copyable, Movable):
    var generation: UInt32
    var alive: Bool
    var archetype_id: Int32
    var chunk_idx: Int32
    var row_in_chunk: Int32
    var sparse_mask: ComponentBitSet


def _blob_fits(reader: BlobReader, count: Int, elem_bytes: Int) -> Bool:
    """True when `count` records of `elem_bytes` each can still be read from `reader`
    (checked before allocating, so a corrupted count cannot request terabytes)."""
    if count < 0:
        return False
    var left = reader.limit() - reader.offset
    if left < 0:
        return False
    if elem_bytes <= 0:
        return count <= left
    return count <= left // elem_bytes


@fieldwise_init
struct _ParSlice1(Copyable, ImplicitlyCopyable, Movable):
    var entities_ptr: Int
    var col0_ptr: Int
    var elem0_size: Int
    var row_start: Int
    var row_count: Int
    var global_order_base: Int
    var is_sparse0: Bool


@fieldwise_init
struct _ParSlice2(Copyable, ImplicitlyCopyable, Movable):
    var entities_ptr: Int
    var col0_ptr: Int
    var elem0_size: Int
    var col1_ptr: Int
    var elem1_size: Int
    var row_start: Int
    var row_count: Int
    var global_order_base: Int
    var is_sparse0: Bool
    var is_sparse1: Bool
    var chunk_addr: Int
    var col0_idx: Int
    var col1_idx: Int


@fieldwise_init
struct _ParDispatch1[C0: Component](Copyable, Movable):
    var world_addr: Int
    var pool_addr: Int
    var slices_ptr: Int
    var kernel: def(Int, Int, Entity, Pointer[Self.C0, MutAnyOrigin], Int) thin -> None
    var user_ctx: Int
    var tid0: Int
    var ex_sparse: ComponentBitSet
    var has_ex_sparse: Bool
    var rt_base: Int
    var changed_since: UInt32
    var read_only: Bool
    var current_tick: UInt32


@fieldwise_init
struct _ParDispatch2[C0: Component, C1: Component](Copyable, Movable):
    var world_addr: Int
    var pool_addr: Int
    var slices_ptr: Int
    var kernel: def(Int, Int, Entity, Pointer[Self.C0, MutAnyOrigin], Pointer[Self.C1, MutAnyOrigin], Int) thin -> None
    var user_ctx: Int
    var tid0: Int
    var tid1: Int
    var ex_sparse: ComponentBitSet
    var has_ex_sparse: Bool
    var rt_base: Int
    var changed_since: UInt32
    var read_only: Bool
    var current_tick: UInt32


def _par_worker1[C0: Component](ctx_addr: Int, slice_idx: Int):
    ref disp = Pointer[_ParDispatch1[C0], ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var lane = 0
    if disp.pool_addr != 0:
        ref pool = Pointer[JobPool, ImmutAnyOrigin](unsafe_from_address=disp.pool_addr)[unsafe_offset=0]
        lane = pool.current_thread_index()
    var prev_imp = UInt64(0)
    if disp.rt_base != 0:
        prev_imp = UInt64(_lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].load())
    var sl = Pointer[_ParSlice1, ImmutAnyOrigin](unsafe_from_address=disp.slices_ptr)[unsafe_offset=slice_idx]
    var kfn = disp.kernel
    var ents = Pointer[Entity, ImmutAnyOrigin](unsafe_from_address=sl.entities_ptr)
    if not sl.is_sparse0 and not disp.has_ex_sparse:
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            var e = ents[unsafe_offset=r]
            var p0 = Pointer[C0, MutAnyOrigin](unsafe_from_address=sl.col0_ptr + r * sl.elem0_size)
            if disp.rt_base != 0:
                _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                    Int64(UInt64(ord) + 1)
                )
            kfn(lane, ord, e, p0, disp.user_ctx)
    elif not sl.is_sparse0:
        ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=disp.world_addr)[unsafe_offset=0]
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            var e = ents[unsafe_offset=r]
            if world.entities[Int(e.index)].sparse_mask.intersects(disp.ex_sparse):
                continue
            var p0 = Pointer[C0, MutAnyOrigin](unsafe_from_address=sl.col0_ptr + r * sl.elem0_size)
            if disp.rt_base != 0:
                _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                    Int64(UInt64(ord) + 1)
                )
            kfn(lane, ord, e, p0, disp.user_ctx)
    else:
        ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=disp.world_addr)[unsafe_offset=0]
        ref sset = world.sparse_sets[world.comp_to_sparse[disp.tid0]]
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            var e = ents[unsafe_offset=r]
            if disp.has_ex_sparse and world.entities[Int(e.index)].sparse_mask.intersects(disp.ex_sparse):
                continue
            var di = sset.dense_index_of(e.index)
            if di >= 0:
                if disp.changed_since > 0 and sset.changed_ticks[di] <= disp.changed_since:
                    continue
                if disp.changed_since > 0 and not disp.read_only:
                    sset.changed_ticks[di] = disp.current_tick
                var p0 = Pointer[C0, MutAnyOrigin](unsafe_from_address=sset.value_ptr(di))
                if disp.rt_base != 0:
                    _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                        Int64(UInt64(ord) + 1)
                    )
                kfn(lane, ord, e, p0, disp.user_ctx)
    if disp.rt_base != 0:
        _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(Int64(prev_imp))


def _par_worker2[C0: Component, C1: Component](ctx_addr: Int, slice_idx: Int):
    ref disp = Pointer[_ParDispatch2[C0, C1], ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var lane = 0
    if disp.pool_addr != 0:
        ref pool = Pointer[JobPool, ImmutAnyOrigin](unsafe_from_address=disp.pool_addr)[unsafe_offset=0]
        lane = pool.current_thread_index()
    var prev_imp = UInt64(0)
    if disp.rt_base != 0:
        prev_imp = UInt64(_lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].load())
    var sl = Pointer[_ParSlice2, ImmutAnyOrigin](unsafe_from_address=disp.slices_ptr)[unsafe_offset=slice_idx]
    var kfn = disp.kernel
    var ents = Pointer[Entity, ImmutAnyOrigin](unsafe_from_address=sl.entities_ptr)
    if not sl.is_sparse0 and not sl.is_sparse1 and not disp.has_ex_sparse:
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            var e = ents[unsafe_offset=r]
            var p0 = Pointer[C0, MutAnyOrigin](unsafe_from_address=sl.col0_ptr + r * sl.elem0_size)
            var p1 = Pointer[C1, MutAnyOrigin](unsafe_from_address=sl.col1_ptr + r * sl.elem1_size)
            if disp.rt_base != 0:
                _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                    Int64(UInt64(ord) + 1)
                )
            kfn(lane, ord, e, p0, p1, disp.user_ctx)
    else:
        ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=disp.world_addr)[unsafe_offset=0]
        var filter_sparse_ticks = disp.changed_since > 0 and (sl.is_sparse0 or sl.is_sparse1)
        var cur_grp = -1
        var grp_chg0 = False
        var grp_chg1 = False
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            if filter_sparse_ticks and (r >> 3) != cur_grp:
                cur_grp = r >> 3
                if sl.chunk_addr != 0:
                    ref ch_ref = Pointer[ArchetypeChunk, ImmutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0]
                    grp_chg0 = False if sl.is_sparse0 else (
                        ch_ref.group_changed_tick(sl.col0_idx, cur_grp) > disp.changed_since
                    )
                    grp_chg1 = False if sl.is_sparse1 else (
                        ch_ref.group_changed_tick(sl.col1_idx, cur_grp) > disp.changed_since
                    )
                else:
                    grp_chg0 = False
                    grp_chg1 = False
            var e = ents[unsafe_offset=r]
            if disp.has_ex_sparse and world.entities[Int(e.index)].sparse_mask.intersects(disp.ex_sparse):
                continue
            var a0: Int
            var d0 = -1
            if not sl.is_sparse0:
                a0 = sl.col0_ptr + r * sl.elem0_size
            else:
                ref s0 = world.sparse_sets[world.comp_to_sparse[disp.tid0]]
                d0 = s0.dense_index_of(e.index)
                if d0 < 0:
                    continue
                a0 = s0.value_ptr(d0)
            var a1: Int
            var d1 = -1
            if not sl.is_sparse1:
                a1 = sl.col1_ptr + r * sl.elem1_size
            else:
                ref s1 = world.sparse_sets[world.comp_to_sparse[disp.tid1]]
                d1 = s1.dense_index_of(e.index)
                if d1 < 0:
                    continue
                a1 = s1.value_ptr(d1)
            if filter_sparse_ticks:
                var any_chg = grp_chg0 or grp_chg1
                if not any_chg and sl.is_sparse0:
                    if world.sparse_sets[world.comp_to_sparse[disp.tid0]].changed_ticks[d0] > disp.changed_since:
                        any_chg = True
                if not any_chg and sl.is_sparse1:
                    if world.sparse_sets[world.comp_to_sparse[disp.tid1]].changed_ticks[d1] > disp.changed_since:
                        any_chg = True
                if not any_chg:
                    continue
                if not disp.read_only:
                    if sl.is_sparse0:
                        world.sparse_sets[world.comp_to_sparse[disp.tid0]].changed_ticks[d0] = disp.current_tick
                    elif sl.chunk_addr != 0:
                        Pointer[ArchetypeChunk, MutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0].mark_changed(
                            sl.col0_idx, r, disp.current_tick
                        )
                    if sl.is_sparse1:
                        world.sparse_sets[world.comp_to_sparse[disp.tid1]].changed_ticks[d1] = disp.current_tick
                    elif sl.chunk_addr != 0:
                        Pointer[ArchetypeChunk, MutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0].mark_changed(
                            sl.col1_idx, r, disp.current_tick
                        )
            if disp.rt_base != 0:
                _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                    Int64(UInt64(ord) + 1)
                )
            kfn(
                lane,
                ord,
                e,
                Pointer[C0, MutAnyOrigin](unsafe_from_address=a0),
                Pointer[C1, MutAnyOrigin](unsafe_from_address=a1),
                disp.user_ctx,
            )
    if disp.rt_base != 0:
        _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(Int64(prev_imp))


@fieldwise_init
struct Changed[C: Component](Copyable, ImplicitlyCopyable, Movable):
    """Query filter marker matching entities whose `C` component changed after `since_tick`."""
    comptime COMPONENT_ID: UInt64 = Self.C.COMPONENT_ID
    var since_tick: UInt32


@fieldwise_init
struct _ParSlice3(Copyable, ImplicitlyCopyable, Movable):
    var entities_ptr: Int
    var col0_ptr: Int
    var elem0_size: Int
    var col1_ptr: Int
    var elem1_size: Int
    var col2_ptr: Int
    var elem2_size: Int
    var row_start: Int
    var row_count: Int
    var global_order_base: Int
    var is_sparse0: Bool
    var is_sparse1: Bool
    var is_sparse2: Bool
    var chunk_addr: Int
    var col0_idx: Int
    var col1_idx: Int
    var col2_idx: Int


@fieldwise_init
struct _ParDispatch3[C0: Component, C1: Component, C2: Component](Copyable, Movable):
    var world_addr: Int
    var pool_addr: Int
    var slices_ptr: Int
    var kernel: def(
        Int,
        Int,
        Entity,
        Pointer[Self.C0, MutAnyOrigin],
        Pointer[Self.C1, MutAnyOrigin],
        Pointer[Self.C2, MutAnyOrigin],
        Int,
    ) thin -> None
    var user_ctx: Int
    var tid0: Int
    var tid1: Int
    var tid2: Int
    var ex_sparse: ComponentBitSet
    var has_ex_sparse: Bool
    var rt_base: Int
    var changed_since: UInt32
    var read_only: Bool
    var current_tick: UInt32


def _par_worker3[C0: Component, C1: Component, C2: Component](ctx_addr: Int, slice_idx: Int):
    ref disp = Pointer[_ParDispatch3[C0, C1, C2], ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var lane = 0
    if disp.pool_addr != 0:
        ref pool = Pointer[JobPool, ImmutAnyOrigin](unsafe_from_address=disp.pool_addr)[unsafe_offset=0]
        lane = pool.current_thread_index()
    var prev_imp = UInt64(0)
    if disp.rt_base != 0:
        prev_imp = UInt64(_lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].load())
    var sl = Pointer[_ParSlice3, ImmutAnyOrigin](unsafe_from_address=disp.slices_ptr)[unsafe_offset=slice_idx]
    var kfn = disp.kernel
    var ents = Pointer[Entity, ImmutAnyOrigin](unsafe_from_address=sl.entities_ptr)
    if not sl.is_sparse0 and not sl.is_sparse1 and not sl.is_sparse2 and not disp.has_ex_sparse:
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            var e = ents[unsafe_offset=r]
            var p0 = Pointer[C0, MutAnyOrigin](unsafe_from_address=sl.col0_ptr + r * sl.elem0_size)
            var p1 = Pointer[C1, MutAnyOrigin](unsafe_from_address=sl.col1_ptr + r * sl.elem1_size)
            var p2 = Pointer[C2, MutAnyOrigin](unsafe_from_address=sl.col2_ptr + r * sl.elem2_size)
            if disp.rt_base != 0:
                _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                    Int64(UInt64(ord) + 1)
                )
            kfn(lane, ord, e, p0, p1, p2, disp.user_ctx)
    else:
        ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=disp.world_addr)[unsafe_offset=0]
        var filter_sparse_ticks = disp.changed_since > 0 and (sl.is_sparse0 or sl.is_sparse1 or sl.is_sparse2)
        var cur_grp = -1
        var grp_chg0 = False
        var grp_chg1 = False
        var grp_chg2 = False
        for i in range(sl.row_count):
            var r = sl.row_start + i
            var ord = sl.global_order_base + i
            if filter_sparse_ticks and (r >> 3) != cur_grp:
                cur_grp = r >> 3
                if sl.chunk_addr != 0:
                    ref ch_ref = Pointer[ArchetypeChunk, ImmutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0]
                    grp_chg0 = False if sl.is_sparse0 else (
                        ch_ref.group_changed_tick(sl.col0_idx, cur_grp) > disp.changed_since
                    )
                    grp_chg1 = False if sl.is_sparse1 else (
                        ch_ref.group_changed_tick(sl.col1_idx, cur_grp) > disp.changed_since
                    )
                    grp_chg2 = False if sl.is_sparse2 else (
                        ch_ref.group_changed_tick(sl.col2_idx, cur_grp) > disp.changed_since
                    )
                else:
                    grp_chg0 = False
                    grp_chg1 = False
                    grp_chg2 = False
            var e = ents[unsafe_offset=r]
            if disp.has_ex_sparse and world.entities[Int(e.index)].sparse_mask.intersects(disp.ex_sparse):
                continue
            var a0: Int
            var d0 = -1
            if not sl.is_sparse0:
                a0 = sl.col0_ptr + r * sl.elem0_size
            else:
                ref s0 = world.sparse_sets[world.comp_to_sparse[disp.tid0]]
                d0 = s0.dense_index_of(e.index)
                if d0 < 0:
                    continue
                a0 = s0.value_ptr(d0)
            var a1: Int
            var d1 = -1
            if not sl.is_sparse1:
                a1 = sl.col1_ptr + r * sl.elem1_size
            else:
                ref s1 = world.sparse_sets[world.comp_to_sparse[disp.tid1]]
                d1 = s1.dense_index_of(e.index)
                if d1 < 0:
                    continue
                a1 = s1.value_ptr(d1)
            var a2: Int
            var d2 = -1
            if not sl.is_sparse2:
                a2 = sl.col2_ptr + r * sl.elem2_size
            else:
                ref s2 = world.sparse_sets[world.comp_to_sparse[disp.tid2]]
                d2 = s2.dense_index_of(e.index)
                if d2 < 0:
                    continue
                a2 = s2.value_ptr(d2)
            if filter_sparse_ticks:
                var any_chg = grp_chg0 or grp_chg1 or grp_chg2
                if not any_chg and sl.is_sparse0:
                    if world.sparse_sets[world.comp_to_sparse[disp.tid0]].changed_ticks[d0] > disp.changed_since:
                        any_chg = True
                if not any_chg and sl.is_sparse1:
                    if world.sparse_sets[world.comp_to_sparse[disp.tid1]].changed_ticks[d1] > disp.changed_since:
                        any_chg = True
                if not any_chg and sl.is_sparse2:
                    if world.sparse_sets[world.comp_to_sparse[disp.tid2]].changed_ticks[d2] > disp.changed_since:
                        any_chg = True
                if not any_chg:
                    continue
                if not disp.read_only:
                    if sl.is_sparse0:
                        world.sparse_sets[world.comp_to_sparse[disp.tid0]].changed_ticks[d0] = disp.current_tick
                    elif sl.chunk_addr != 0:
                        Pointer[ArchetypeChunk, MutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0].mark_changed(
                            sl.col0_idx, r, disp.current_tick
                        )
                    if sl.is_sparse1:
                        world.sparse_sets[world.comp_to_sparse[disp.tid1]].changed_ticks[d1] = disp.current_tick
                    elif sl.chunk_addr != 0:
                        Pointer[ArchetypeChunk, MutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0].mark_changed(
                            sl.col1_idx, r, disp.current_tick
                        )
                    if sl.is_sparse2:
                        world.sparse_sets[world.comp_to_sparse[disp.tid2]].changed_ticks[d2] = disp.current_tick
                    elif sl.chunk_addr != 0:
                        Pointer[ArchetypeChunk, MutAnyOrigin](unsafe_from_address=sl.chunk_addr)[unsafe_offset=0].mark_changed(
                            sl.col2_idx, r, disp.current_tick
                        )
            if disp.rt_base != 0:
                _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(
                    Int64(UInt64(ord) + 1)
                )
            kfn(
                lane,
                ord,
                e,
                Pointer[C0, MutAnyOrigin](unsafe_from_address=a0),
                Pointer[C1, MutAnyOrigin](unsafe_from_address=a1),
                Pointer[C2, MutAnyOrigin](unsafe_from_address=a2),
                disp.user_ctx,
            )
    if disp.rt_base != 0:
        _lane_order_slot_from_base(disp.rt_base, lane)[unsafe_offset=0].store(Int64(prev_imp))


# ---------------------------------------------------------------------------
# World — Hybrid Archetype + SparseSet ECS Runtime
# ---------------------------------------------------------------------------

struct World(Movable):
    """Flint v5 Hybrid ECS World supporting dynamic archetype transitions, sparse tags,
    parallel JobPool queries, lock-free deterministic commands, and `<0.2 ms` snapshots."""
    var entities: List[EntityRecord]
    var free_indices: List[UInt32]
    var live_entities: Int
    var current_tick: UInt32
    var components: List[ComponentInfo]
    var comp_hash_keys: List[UInt64]
    var comp_hash_vals: List[Int32]
    var table_comp_mask: ComponentBitSet
    var sparse_comp_mask: ComponentBitSet
    var comp_to_sparse: List[Int]
    var sparse_sets: List[SparseSet]
    var archetypes: List[Archetype]
    var chunk_rows: Int
    var restore_epoch: UInt64

    def __init__(out self, chunk_rows: Int = DEFAULT_CHUNK_ROWS):
        self.entities = List[EntityRecord]()
        self.free_indices = List[UInt32]()
        self.live_entities = 0
        self.current_tick = 1
        self.restore_epoch = 0
        self.components = List[ComponentInfo]()
        self.comp_hash_keys = List[UInt64](length=_INITIAL_COMP_HASH_CAP, fill=0)
        self.comp_hash_vals = List[Int32](length=_INITIAL_COMP_HASH_CAP, fill=-1)
        self.table_comp_mask = ComponentBitSet()
        self.sparse_comp_mask = ComponentBitSet()
        self.comp_to_sparse = List[Int]()
        self.sparse_sets = List[SparseSet]()
        self.chunk_rows = ((max(SIMD_LANES, chunk_rows) + (SIMD_LANES - 1)) // SIMD_LANES) * SIMD_LANES
        self.archetypes = List[Archetype]()
        var empty_ids = List[Int]()
        var empty_sizes = List[Int]()
        self.archetypes.append(Archetype(0, ComponentBitSet(), empty_ids, empty_sizes, self.chunk_rows))

    def __deinit__(deinit self):
        self._drop_all_live_components()

    @always_inline
    def keep_alive(self):
        """Anchors `World` and its chunk/sparse buffers past a parallel `JobPool` barrier."""
        var anchor = len(self.archetypes) ^ len(self.sparse_sets) ^ Int(Pointer(to=self))
        _ = anchor

    def _drop_all_live_components(mut self):
        for a in range(len(self.archetypes)):
            ref arch = self.archetypes[a]
            for ch in range(len(arch.chunks)):
                ref chunk = arch.chunks[ch]
                for c in range(len(arch.comp_type_ids)):
                    var tid = arch.comp_type_ids[c]
                    var info = self.components[tid]
                    if info.byte_size > 0 and not info.is_trivial_pod:
                        var base = chunk.column_ptr(c)
                        for r in range(chunk.count):
                            info.drop_at(base + r * info.byte_size)
                chunk.count = 0
                chunk.entities.clear()
        for s in range(len(self.sparse_sets)):
            var tid = self.sparse_sets[s].type_id
            self.sparse_sets[s].clear_with_drop(self.components[tid])

    def clear_entities(mut self):
        """Destroys all live entities and their components while preserving registered component types."""
        self._drop_all_live_components()
        for a in range(len(self.archetypes)):
            self.archetypes[a].chunks.clear()
        self.entities.clear()
        self.free_indices.clear()
        self.live_entities = 0
        self.current_tick += 1

    @always_inline
    def entity_count(self) -> Int:
        return self.live_entities

    @always_inline
    def archetype_count(self) -> Int:
        return len(self.archetypes)

    def total_chunk_count(self) -> Int:
        var n = 0
        for a in range(len(self.archetypes)):
            n += len(self.archetypes[a].chunks)
        return n

    def _reclaim_empty_chunk(mut self, arch_id: Int, ch_idx: Int):
        """Reclaims an empty `ArchetypeChunk` via swap-pop so slab memory does not leak on despawn."""
        if ch_idx < 0 or ch_idx >= len(self.archetypes[arch_id].chunks):
            return
        if self.archetypes[arch_id].chunks[ch_idx].count != 0:
            return
        var last_ch = len(self.archetypes[arch_id].chunks) - 1
        if ch_idx < last_ch:
            var moved_chunk = self.archetypes[arch_id].chunks.pop()
            for r in range(moved_chunk.count):
                var e = moved_chunk.entities[r]
                self.entities[Int(e.index)].chunk_idx = Int32(ch_idx)
            self.archetypes[arch_id].chunks[ch_idx] = moved_chunk^
        else:
            _ = self.archetypes[arch_id].chunks.pop()

    @always_inline
    def to_entity_handle(self, entity: Entity) -> EntityHandle:
        """Converts `entity` into a `flint.core.ecs.EntityHandle` carrying its 1-based archetype ID."""
        if not self.is_alive(entity):
            return entity.to_entity_handle(0)
        var arch_hint = Int(self.entities[Int(entity.index)].archetype_id) + 1
        return entity.to_entity_handle(arch_hint)

    @always_inline
    def advance_tick(mut self) -> UInt32:
        self.current_tick += 1
        if self.current_tick == 0:
            self.current_tick = 1
        return self.current_tick

    @always_inline
    def find_component_by_hash(self, comp_hash: UInt64) -> Int:
        var cap = len(self.comp_hash_vals)
        if cap == 0:
            return -1
        var mask = cap - 1
        var slot = Int(comp_hash ^ (comp_hash >> 33)) & mask
        for _ in range(cap):
            var tid = Int(self.comp_hash_vals[slot])
            if tid < 0:
                return -1
            if self.comp_hash_keys[slot] == comp_hash:
                return tid
            slot = (slot + 1) & mask
        return -1

    def _rehash_component_table(mut self, new_cap: Int):
        var cap = max(_INITIAL_COMP_HASH_CAP, new_cap)
        var new_keys = List[UInt64](length=cap, fill=0)
        var new_vals = List[Int32](length=cap, fill=-1)
        var mask = cap - 1
        for i in range(len(self.comp_hash_vals)):
            var tid = self.comp_hash_vals[i]
            if tid >= 0:
                var k = self.comp_hash_keys[i]
                var slot = Int(k ^ (k >> 33)) & mask
                while new_vals[slot] >= 0:
                    slot = (slot + 1) & mask
                new_keys[slot] = k
                new_vals[slot] = tid
        self.comp_hash_keys = new_keys^
        self.comp_hash_vals = new_vals^

    def _index_component_hash(mut self, comp_hash: UInt64, tid: Int):
        var cap = len(self.comp_hash_vals)
        if len(self.components) * 2 >= cap:
            self._rehash_component_table(max(_INITIAL_COMP_HASH_CAP, cap * 2))
            cap = len(self.comp_hash_vals)
        var mask = cap - 1
        var slot = Int(comp_hash ^ (comp_hash >> 33)) & mask
        for _ in range(cap):
            if self.comp_hash_vals[slot] < 0 or self.comp_hash_keys[slot] == comp_hash:
                self.comp_hash_keys[slot] = comp_hash
                self.comp_hash_vals[slot] = Int32(tid)
                return
            slot = (slot + 1) & mask

    def register_component[C: Component](mut self) -> Int:
        """Idempotently registers `C` and returns its dense component type ID."""
        var existing = self.find_component_by_hash(C.COMPONENT_ID)
        if existing >= 0:
            return existing
        var tid = len(self.components)
        comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
        var cfn: _CopyInitFn = _noop_copy_init
        var mfn: _MoveInitFn = _noop_move_init
        var dfn: _DropFn = _noop_drop
        if not is_triv:
            cfn = _raw_copy_init[C]
            mfn = _raw_move_init[C]
            dfn = _raw_destroy[C]
        var mode = C.STORAGE_MODE
        var bsz = size_of[C]()
        var aln = max(1, align_of[C]())
        self.components.append(
            ComponentInfo(
                tid,
                C.COMPONENT_ID,
                bsz,
                aln,
                mode,
                is_triv,
                cfn,
                mfn,
                dfn,
                False,
                _noop_bin_write,
                _noop_bin_read,
            )
        )
        if mode == STORAGE_SPARSE:
            self.sparse_comp_mask.set(tid)
            var sidx = len(self.sparse_sets)
            self.comp_to_sparse.append(sidx)
            self.sparse_sets.append(SparseSet(tid, bsz))
        else:
            self.comp_to_sparse.append(-1)
            self.table_comp_mask.set(tid)
        self._index_component_hash(C.COMPONENT_ID, tid)
        return tid

    def set_component_binary_hooks(mut self, type_id: Int, write_fn: _BinWriteFn, read_fn: _BinReadFn):
        """Registers custom binary hooks for components holding heap handles (`String`, `List`)."""
        if type_id >= 0 and type_id < len(self.components):
            self.components[type_id].has_custom_bin = True
            self.components[type_id].bin_write_fn = write_fn
            self.components[type_id].bin_read_fn = read_fn

    def register_pod_component(
        mut self,
        comp_hash: UInt64,
        byte_size: Int,
        alignment: Int = 8,
        storage_mode: Int = STORAGE_TABLE,
    ) -> Int:
        """Registers a POD component by runtime metadata (safe across `.so` plugin reloads)."""
        var bsz = max(0, byte_size)
        var existing = self.find_component_by_hash(comp_hash)
        if existing >= 0:
            if (
                self.components[existing].byte_size != bsz
                or self.components[existing].storage_mode != storage_mode
            ):
                return -1
            return existing
        var tid = len(self.components)
        var aln = max(1, alignment)
        self.components.append(
            ComponentInfo(
                tid,
                comp_hash,
                bsz,
                aln,
                storage_mode,
                True,
                _noop_copy_init,
                _noop_move_init,
                _noop_drop,
                False,
                _noop_bin_write,
                _noop_bin_read,
            )
        )
        if storage_mode == STORAGE_SPARSE:
            self.sparse_comp_mask.set(tid)
            var sidx = len(self.sparse_sets)
            self.comp_to_sparse.append(sidx)
            self.sparse_sets.append(SparseSet(tid, bsz))
        else:
            self.comp_to_sparse.append(-1)
            self.table_comp_mask.set(tid)
        self._index_component_hash(comp_hash, tid)
        return tid

    def _prepare_raw_pod_dst(mut self, entity: Entity, tid: Int) -> Tuple[Bool, Int]:
        """Allocates or prepares destination cell for POD component `tid` on `entity`, returning `(ok, dst_ptr)`."""
        if tid < 0 or tid >= len(self.components) or not self.is_alive(entity):
            return (False, 0)
        var info = self.components[tid]
        var idx = Int(entity.index)
        if info.storage_mode == STORAGE_SPARSE:
            var sidx = self.comp_to_sparse[tid]
            var dst_ptr = self.sparse_sets[sidx].insert_uninit(entity, info, self.current_tick)
            self.entities[idx].sparse_mask.set(tid)
            return (True, dst_ptr if info.byte_size > 0 else 0)
        var src_arch = Int(self.entities[idx].archetype_id)
        var src_ch = Int(self.entities[idx].chunk_idx)
        var src_row = Int(self.entities[idx].row_in_chunk)
        var existing_col = self.archetypes[src_arch].col_index(tid)
        if existing_col >= 0:
            var cell = self.archetypes[src_arch].chunks[src_ch].cell_ptr(existing_col, src_row)
            if info.byte_size > 0:
                info.drop_at(cell)
            self.archetypes[src_arch].chunks[src_ch].mark_changed(existing_col, src_row, self.current_tick)
            return (True, cell if info.byte_size > 0 else 0)
        var dst_arch = self._next_archetype_add(src_arch, tid)
        var dst_loc = self._move_entity_archetype(entity, src_arch, src_ch, src_row, dst_arch, -1)
        var dst_col = self.archetypes[dst_arch].col_index(tid)
        var cell = self.archetypes[dst_arch].chunks[dst_loc[0]].cell_ptr(dst_col, dst_loc[1]) if info.byte_size > 0 else 0
        self.archetypes[dst_arch].chunks[dst_loc[0]].mark_changed(dst_col, dst_loc[1], self.current_tick)
        return (True, cell)

    def insert_raw_pod(mut self, entity: Entity, tid: Int, src_ptr: Int) -> Bool:
        """Inserts or overwrites a POD component `tid` on `entity` from `src_ptr`."""
        var res = self._prepare_raw_pod_dst(entity, tid)
        if not res[0]:
            return False
        var byte_size = self.components[tid].byte_size
        if byte_size > 0:
            if src_ptr != 0:
                _ = external_call["memcpy", Int](res[1], src_ptr, byte_size)
            else:
                _ = external_call["memset", Int](res[1], Int32(0), byte_size)
        return True

    def insert_raw_pod_bytes(mut self, entity: Entity, tid: Int, data: Span[UInt8, _]) -> Bool:
        """Inserts or overwrites a POD component `tid` on `entity` from an origin-tracked `Span[UInt8]`."""
        if tid < 0 or tid >= len(self.components) or not self.is_alive(entity):
            return False
        var byte_size = self.components[tid].byte_size
        if byte_size > 0 and len(data) < byte_size:
            return False
        var res = self._prepare_raw_pod_dst(entity, tid)
        if not res[0]:
            return False
        if byte_size > 0:
            var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=res[1])
            for i in range(byte_size):
                dst[unsafe_offset=i] = data[i]
        return True

    def has_by_type_id(self, entity: Entity, tid: Int) -> Bool:
        if tid < 0 or tid >= len(self.components) or not self.is_alive(entity):
            return False
        ref rec = self.entities[Int(entity.index)]
        if self.components[tid].storage_mode == STORAGE_SPARSE:
            return rec.sparse_mask.test(tid)
        return self.archetypes[Int(rec.archetype_id)].table_mask.test(tid)

    def get_raw_ptr(self, entity: Entity, tid: Int) -> Int:
        if tid < 0 or tid >= len(self.components) or not self.is_alive(entity):
            return 0
        ref rec = self.entities[Int(entity.index)]
        var info = self.components[tid]
        if info.storage_mode == STORAGE_SPARSE:
            if not rec.sparse_mask.test(tid):
                return 0
            var sidx = self.comp_to_sparse[tid]
            var di = self.sparse_sets[sidx].dense_index_of(entity.index)
            return self.sparse_sets[sidx].value_ptr(di) if di >= 0 else 0
        var arch_id = Int(rec.archetype_id)
        var col = self.archetypes[arch_id].col_index(tid)
        if col < 0:
            return 0
        var addr = self.archetypes[arch_id].chunks[Int(rec.chunk_idx)].cell_ptr(col, Int(rec.row_in_chunk))
        if addr == 0 and info.byte_size == 0:
            addr = Int(Pointer(to=self.entities))
        return addr

    def get_raw_ptr_mut(mut self, entity: Entity, tid: Int) -> Int:
        if tid < 0 or tid >= len(self.components) or not self.is_alive(entity):
            return 0
        var idx = Int(entity.index)
        var info = self.components[tid]
        if info.storage_mode == STORAGE_SPARSE:
            if not self.entities[idx].sparse_mask.test(tid):
                return 0
            var sidx = self.comp_to_sparse[tid]
            var di = self.sparse_sets[sidx].dense_index_of(entity.index)
            if di < 0:
                return 0
            self.sparse_sets[sidx].changed_ticks[di] = self.current_tick
            return self.sparse_sets[sidx].value_ptr(di)
        var arch_id = Int(self.entities[idx].archetype_id)
        var ch = Int(self.entities[idx].chunk_idx)
        var row = Int(self.entities[idx].row_in_chunk)
        var col = self.archetypes[arch_id].col_index(tid)
        if col < 0:
            return 0
        self.archetypes[arch_id].chunks[ch].mark_changed(col, row, self.current_tick)
        var addr = self.archetypes[arch_id].chunks[ch].cell_ptr(col, row)
        if addr == 0 and info.byte_size == 0:
            addr = Int(Pointer(to=self.entities))
        return addr

    @always_inline
    def component_id[C: Component](self) -> Int:
        return self.find_component_by_hash(C.COMPONENT_ID)

    def bit_for[C: Component](mut self) -> ComponentBitSet:
        var tid = self.register_component[C]()
        return ComponentBitSet().with_bit(tid)

    @always_inline
    def is_alive(self, entity: Entity) -> Bool:
        if not entity.valid() or entity.is_provisional():
            return False
        var idx = Int(entity.index)
        if idx < 0 or idx >= len(self.entities):
            return False
        ref rec = self.entities[idx]
        return rec.alive and rec.generation == entity.generation

    def spawn_empty(mut self) -> Entity:
        """Spawns a live entity in the empty root archetype `0`."""
        var idx: UInt32
        var gen: UInt32
        if len(self.free_indices) > 0:
            # Pop lowest-index free slot if LIFO/sorted; deterministic across identical playback
            idx = self.free_indices.pop()
            gen = self.entities[Int(idx)].generation
        else:
            idx = UInt32(len(self.entities))
            gen = 1
            self.entities.append(EntityRecord(1, False, 0, 0, 0, ComponentBitSet()))
        var entity = Entity(idx, gen)
        var ch_idx = self.archetypes[0].ensure_chunk_with_space()
        var row = self.archetypes[0].chunks[ch_idx].push_uninit_row(entity, self.current_tick)
        self.entities[Int(idx)] = EntityRecord(
            gen, True, 0, Int32(ch_idx), Int32(row), ComponentBitSet()
        )
        self.live_entities += 1
        return entity

    def spawn(mut self) -> Entity:
        return self.spawn_empty()

    def despawn(mut self, entity: Entity) -> Bool:
        if not self.is_alive(entity):
            return False
        var idx = Int(entity.index)
        var rec_gen = self.entities[idx].generation
        var arch_id = Int(self.entities[idx].archetype_id)
        var ch_idx = Int(self.entities[idx].chunk_idx)
        var row = Int(self.entities[idx].row_in_chunk)
        var smask = self.entities[idx].sparse_mask.copy()

        # Drop table components at row, then swap-remove row in chunk
        for c in range(len(self.archetypes[arch_id].comp_type_ids)):
            var tid = self.archetypes[arch_id].comp_type_ids[c]
            var info = self.components[tid]
            if info.byte_size > 0:
                info.drop_at(self.archetypes[arch_id].chunks[ch_idx].cell_ptr(c, row))
        var moved = self.archetypes[arch_id].chunks[ch_idx].swap_remove_no_drop(
            row, self.current_tick, self.components
        )
        if moved.valid():
            self.entities[Int(moved.index)].row_in_chunk = Int32(row)
        self._reclaim_empty_chunk(arch_id, ch_idx)

        # Remove any sparse components attached to this entity
        var wc = smask.trimmed_word_count()
        for wi in range(wc):
            var w = smask.word_at(wi)
            while w != 0:
                var stid = wi * 64 + _ctz64(w)
                w &= w - 1
                var sidx = self.comp_to_sparse[stid]
                if sidx >= 0:
                    _ = self.sparse_sets[sidx].remove(entity.index, self.components[stid])

        var next_gen = _next_entity_gen(rec_gen)
        self.entities[idx] = EntityRecord(next_gen, False, -1, -1, -1, ComponentBitSet())
        self.free_indices.append(entity.index)
        self.live_entities -= 1
        return True

    # ---- ArchetypeGraph O(1) Transitions ----

    def _find_or_create_archetype(mut self, mask: ComponentBitSet) -> Int:
        for i in range(len(self.archetypes)):
            if self.archetypes[i].table_mask == mask:
                return i
        var ids = List[Int]()
        var sizes = List[Int]()
        var wc = mask.trimmed_word_count()
        for wi in range(wc):
            var w = mask.word_at(wi)
            while w != 0:
                var tid = wi * 64 + _ctz64(w)
                w &= w - 1
                ids.append(tid)
                sizes.append(self.components[tid].byte_size)
        var new_id = len(self.archetypes)
        self.archetypes.append(Archetype(new_id, mask, ids, sizes, self.chunk_rows))
        return new_id

    def _next_archetype_add(mut self, src_arch: Int, tid: Int) -> Int:
        if tid >= 0 and tid < len(self.archetypes[src_arch].add_edges):
            var cached = Int(self.archetypes[src_arch].add_edges[tid])
            if cached >= 0:
                return cached
        var target_mask = self.archetypes[src_arch].table_mask.with_bit(tid)
        var dst_arch = self._find_or_create_archetype(target_mask)
        self.archetypes[src_arch]._ensure_add_edge_cap(tid)
        self.archetypes[src_arch].add_edges[tid] = Int32(dst_arch)
        self.archetypes[dst_arch]._ensure_remove_edge_cap(tid)
        self.archetypes[dst_arch].remove_edges[tid] = Int32(src_arch)
        return dst_arch

    def _next_archetype_remove(mut self, src_arch: Int, tid: Int) -> Int:
        if tid >= 0 and tid < len(self.archetypes[src_arch].remove_edges):
            var cached = Int(self.archetypes[src_arch].remove_edges[tid])
            if cached >= 0:
                return cached
        var target_mask = self.archetypes[src_arch].table_mask.without_bit(tid)
        var dst_arch = self._find_or_create_archetype(target_mask)
        self.archetypes[src_arch]._ensure_remove_edge_cap(tid)
        self.archetypes[src_arch].remove_edges[tid] = Int32(dst_arch)
        self.archetypes[dst_arch]._ensure_add_edge_cap(tid)
        self.archetypes[dst_arch].add_edges[tid] = Int32(src_arch)
        return dst_arch

    def _move_entity_archetype(
        mut self,
        entity: Entity,
        src_arch: Int,
        src_ch: Int,
        src_row: Int,
        dst_arch: Int,
        dropped_tid: Int,
    ) -> Tuple[Int, Int]:
        """Relocates shared table components via `move_at` from `src` to `dst` and swap-removes `src_row`."""
        _ = dropped_tid
        var dst_ch = self.archetypes[dst_arch].ensure_chunk_with_space()
        var dst_row = self.archetypes[dst_arch].chunks[dst_ch].push_uninit_row(entity, self.current_tick)

        var src_n = len(self.archetypes[src_arch].comp_type_ids)
        for sc in range(src_n):
            var tid = self.archetypes[src_arch].comp_type_ids[sc]
            var info = self.components[tid]
            var esz = info.byte_size
            var s_ptr = self.archetypes[src_arch].chunks[src_ch].cell_ptr(sc, src_row)
            var dc = self.archetypes[dst_arch].col_index(tid)
            if dc >= 0:
                if esz > 0:
                    var d_ptr = self.archetypes[dst_arch].chunks[dst_ch].cell_ptr(dc, dst_row)
                    info.move_at(d_ptr, s_ptr)
            elif esz > 0:
                info.drop_at(s_ptr)

        var moved = self.archetypes[src_arch].chunks[src_ch].swap_remove_no_drop(
            src_row, self.current_tick, self.components
        )
        if moved.valid():
            self.entities[Int(moved.index)].row_in_chunk = Int32(src_row)

        var idx = Int(entity.index)
        self.entities[idx].archetype_id = Int32(dst_arch)
        self.entities[idx].chunk_idx = Int32(dst_ch)
        self.entities[idx].row_in_chunk = Int32(dst_row)
        self._reclaim_empty_chunk(src_arch, src_ch)
        return (dst_ch, dst_row)

    # ---- Component Insert / Remove / Has / Get / GetPtrMut ----

    def insert[C: Component](mut self, entity: Entity, var value: C) -> Bool:
        if not self.is_alive(entity):
            _ = value^
            return False
        var tid = self.register_component[C]()
        if tid < 0:
            _ = value^
            return False
        var idx = Int(entity.index)

        if C.STORAGE_MODE == STORAGE_SPARSE:
            var sidx = self.comp_to_sparse[tid]
            var info = self.components[tid]
            var dst_ptr = self.sparse_sets[sidx].insert_uninit(entity, info, self.current_tick)
            if info.byte_size > 0:
                _raw_emplace[C](dst_ptr, value^)
            else:
                _ = value^
            self.entities[idx].sparse_mask.set(tid)
            return True

        var src_arch = Int(self.entities[idx].archetype_id)
        var src_ch = Int(self.entities[idx].chunk_idx)
        var src_row = Int(self.entities[idx].row_in_chunk)
        var existing_col = self.archetypes[src_arch].col_index(tid)
        if existing_col >= 0:
            var cell = self.archetypes[src_arch].chunks[src_ch].cell_ptr(existing_col, src_row)
            if self.components[tid].byte_size > 0:
                self.components[tid].drop_at(cell)
                _raw_emplace[C](cell, value^)
            else:
                _ = value^
            self.archetypes[src_arch].chunks[src_ch].mark_changed(existing_col, src_row, self.current_tick)
            return True

        var dst_arch = self._next_archetype_add(src_arch, tid)
        var dst_loc = self._move_entity_archetype(entity, src_arch, src_ch, src_row, dst_arch, -1)
        var dst_col = self.archetypes[dst_arch].col_index(tid)
        if self.components[tid].byte_size > 0:
            var cell = self.archetypes[dst_arch].chunks[dst_loc[0]].cell_ptr(dst_col, dst_loc[1])
            _raw_emplace[C](cell, value^)
        else:
            _ = value^
        self.archetypes[dst_arch].chunks[dst_loc[0]].mark_changed(dst_col, dst_loc[1], self.current_tick)
        return True

    def remove_by_type_id(mut self, entity: Entity, tid: Int) -> Bool:
        if tid < 0 or tid >= len(self.components) or not self.is_alive(entity):
            return False
        var idx = Int(entity.index)
        var info = self.components[tid]
        if info.storage_mode == STORAGE_SPARSE:
            if not self.entities[idx].sparse_mask.test(tid):
                return False
            self.entities[idx].sparse_mask.clear(tid)
            var sidx = self.comp_to_sparse[tid]
            return self.sparse_sets[sidx].remove(entity.index, info)

        var src_arch = Int(self.entities[idx].archetype_id)
        var src_ch = Int(self.entities[idx].chunk_idx)
        var src_row = Int(self.entities[idx].row_in_chunk)
        if not self.archetypes[src_arch].table_mask.test(tid):
            return False
        var dst_arch = self._next_archetype_remove(src_arch, tid)
        _ = self._move_entity_archetype(
            entity, src_arch, src_ch, src_row, dst_arch, tid
        )
        return True

    def remove[C: Component](mut self, entity: Entity) -> Bool:
        var tid = self.find_component_by_hash(C.COMPONENT_ID)
        return self.remove_by_type_id(entity, tid)

    def has[C: Component](self, entity: Entity) -> Bool:
        if not self.is_alive(entity):
            return False
        var tid = self.find_component_by_hash(C.COMPONENT_ID)
        if tid < 0:
            return False
        ref rec = self.entities[Int(entity.index)]
        if C.STORAGE_MODE == STORAGE_SPARSE:
            return rec.sparse_mask.test(tid)
        return self.archetypes[Int(rec.archetype_id)].table_mask.test(tid)

    def get_ptr[C: Component](self, entity: Entity) -> Optional[Pointer[C, ImmutAnyOrigin]]:
        var tid = self.find_component_by_hash(C.COMPONENT_ID)
        var addr = self.get_raw_ptr(entity, tid)
        if addr == 0:
            return None
        return Pointer[C, ImmutAnyOrigin](unsafe_from_address=addr)

    def get[C: Component](self, entity: Entity) -> C:
        """Returns a copy of `C` on `entity` (caller must check `has[C]` or know `C` is present)."""
        var ptr = self.get_ptr[C](entity)
        return ptr.value()[unsafe_offset=0].copy()

    def get_ptr_mut[C: Component](mut self, entity: Entity) -> Optional[Pointer[C, MutAnyOrigin]]:
        """Returns a mutable pointer to `C` on `entity` and marks its 8-row SIMD `changed_ticks`."""
        var tid = self.find_component_by_hash(C.COMPONENT_ID)
        var addr = self.get_raw_ptr_mut(entity, tid)
        if addr == 0:
            return None
        return Pointer[C, MutAnyOrigin](unsafe_from_address=addr)

    # ---- Queries & Lifetime-Safe Parallel Iteration ----

    def query_entities(
        self,
        with_mask: ComponentBitSet,
        without_mask: ComponentBitSet = ComponentBitSet(),
    ) -> List[Entity]:
        """Collects all live entities matching `with_mask` and excluding `without_mask`."""
        var out = List[Entity]()
        var req_table = with_mask.intersection(self.table_comp_mask)
        var req_sparse = with_mask.intersection(self.sparse_comp_mask)
        var ex_table = without_mask.intersection(self.table_comp_mask)
        var ex_sparse = without_mask.intersection(self.sparse_comp_mask)
        # If with_mask requests an unregistered bit, nothing can match
        var known = self.table_comp_mask.union(self.sparse_comp_mask)
        if not known.contains_all(with_mask):
            return out^

        var check_sparse = (not req_sparse.is_empty()) or (not ex_sparse.is_empty())
        for a in range(len(self.archetypes)):
            ref arch = self.archetypes[a]
            if not arch.table_mask.contains_all(req_table):
                continue
            if arch.table_mask.intersects(ex_table):
                continue
            for ch in range(len(arch.chunks)):
                ref chunk = arch.chunks[ch]
                for r in range(chunk.count):
                    var e = chunk.entities[r]
                    if check_sparse:
                        ref sm = self.entities[Int(e.index)].sparse_mask
                        if not sm.contains_all(req_sparse) or sm.intersects(ex_sparse):
                            continue
                    out.append(e)
        return out^

    def query_changed_entities[C: Component](
        self,
        since_tick: UInt32,
        with_mask: ComponentBitSet = ComponentBitSet(),
        without_mask: ComponentBitSet = ComponentBitSet(),
    ) -> List[Entity]:
        """Collects all live entities with `C` (plus `with_mask`, excluding `without_mask`)
        whose `C` changed after `since_tick`, skipping unchanged chunks and 8-row SIMD groups."""
        var out = List[Entity]()
        var tid = self.find_component_by_hash(C.COMPONENT_ID)
        if tid < 0:
            return out^
        var full_with = with_mask.with_bit(tid)
        var req_table = full_with.intersection(self.table_comp_mask)
        var req_sparse = full_with.intersection(self.sparse_comp_mask)
        var ex_table = without_mask.intersection(self.table_comp_mask)
        var ex_sparse = without_mask.intersection(self.sparse_comp_mask)
        var known = self.table_comp_mask.union(self.sparse_comp_mask)
        if not known.contains_all(full_with):
            return out^

        var is_sparse = (C.STORAGE_MODE == STORAGE_SPARSE)
        var check_sparse = (not req_sparse.is_empty()) or (not ex_sparse.is_empty())
        for a in range(len(self.archetypes)):
            ref arch = self.archetypes[a]
            if not arch.table_mask.contains_all(req_table):
                continue
            if arch.table_mask.intersects(ex_table):
                continue
            var col_c = -1 if is_sparse else arch.col_index(tid)
            for ch in range(len(arch.chunks)):
                ref chunk = arch.chunks[ch]
                if chunk.count <= 0:
                    continue
                if not is_sparse and since_tick > 0 and not chunk.any_group_changed_since(col_c, since_tick):
                    continue
                if not is_sparse and since_tick > 0:
                    var active_groups = (chunk.count + (SIMD_LANES - 1)) >> 3
                    for g in range(active_groups):
                        if chunk.group_changed_tick(col_c, g) <= since_tick:
                            continue
                        var r_start = g << 3
                        var r_end = min(chunk.count, r_start + SIMD_LANES)
                        for r in range(r_start, r_end):
                            var e = chunk.entities[r]
                            if check_sparse:
                                ref sm = self.entities[Int(e.index)].sparse_mask
                                if not sm.contains_all(req_sparse) or sm.intersects(ex_sparse):
                                    continue
                            out.append(e)
                else:
                    for r in range(chunk.count):
                        var e = chunk.entities[r]
                        if check_sparse:
                            ref sm = self.entities[Int(e.index)].sparse_mask
                            if not sm.contains_all(req_sparse) or sm.intersects(ex_sparse):
                                continue
                        if is_sparse and since_tick > 0:
                            ref sset = self.sparse_sets[self.comp_to_sparse[tid]]
                            var di = sset.dense_index_of(e.index)
                            if di < 0 or sset.changed_ticks[di] <= since_tick:
                                continue
                        out.append(e)
        return out^

    def par_for_each1[C0: Component](
        mut self,
        mut pool: JobPool,
        kernel: def(Int, Int, Entity, Pointer[C0, MutAnyOrigin], Int) thin -> None,
        user_ctx: Int = 0,
        without_mask: ComponentBitSet = ComponentBitSet(),
        read_only: Bool = False,
        changed_since: UInt32 = 0,
    ):
        """Lifetime-safe parallel iteration over all entities with `C0` across `JobPool`.
        Set `read_only=True` to avoid dirtying `changed_ticks`; set `changed_since > 0`
        to skip unchanged chunks/SIMD groups."""
        var tid0 = self.find_component_by_hash(C0.COMPONENT_ID)
        if tid0 < 0:
            return
        var rt_base = _ecs_runtime_base() if _ecs_runtime_ensure() else 0
        var req_mask = ComponentBitSet().with_bit(tid0)
        var req_table = req_mask.intersection(self.table_comp_mask)
        var ex_table = without_mask.intersection(self.table_comp_mask)
        var ex_sparse = without_mask.intersection(self.sparse_comp_mask)
        var has_ex_sparse = not ex_sparse.is_empty()
        var is_sp0 = (C0.STORAGE_MODE == STORAGE_SPARSE)
        var esz0 = self.components[tid0].byte_size

        var slices = List[_ParSlice1]()
        var global_base = 0
        for a in range(len(self.archetypes)):
            if not self.archetypes[a].table_mask.contains_all(req_table):
                continue
            if self.archetypes[a].table_mask.intersects(ex_table):
                continue
            var col0 = -1 if is_sp0 else self.archetypes[a].col_index(tid0)
            for ch in range(len(self.archetypes[a].chunks)):
                var cnt = self.archetypes[a].chunks[ch].count
                if cnt <= 0:
                    continue
                if changed_since > 0 and not is_sp0:
                    if not self.archetypes[a].chunks[ch].any_group_changed_since(col0, changed_since):
                        continue
                    var ents_ptr = Int(self.archetypes[a].chunks[ch].entities.unsafe_ptr())
                    var c0_ptr = self.archetypes[a].chunks[ch].column_ptr(col0)
                    var active_groups = (cnt + (SIMD_LANES - 1)) >> 3
                    for g in range(active_groups):
                        if self.archetypes[a].chunks[ch].group_changed_tick(col0, g) <= changed_since:
                            continue
                        var rs = g << 3
                        var rc = min(cnt - rs, SIMD_LANES)
                        if not read_only:
                            self.archetypes[a].chunks[ch].mark_changed(col0, rs, self.current_tick)
                        slices.append(_ParSlice1(ents_ptr, c0_ptr, esz0, rs, rc, global_base, False))
                        global_base += rc
                else:
                    if not is_sp0 and not read_only:
                        self.archetypes[a].chunks[ch].mark_all_groups_changed(col0, self.current_tick)
                    elif is_sp0 and not read_only and changed_since == 0:
                        var sidx0 = self.comp_to_sparse[tid0]
                        ref sset0 = self.sparse_sets[sidx0]
                        for r in range(cnt):
                            var e = self.archetypes[a].chunks[ch].entities[r]
                            if has_ex_sparse and self.entities[Int(e.index)].sparse_mask.intersects(ex_sparse):
                                continue
                            var di0 = sset0.dense_index_of(e.index)
                            if di0 >= 0:
                                sset0.changed_ticks[di0] = self.current_tick
                    var ents_ptr = Int(self.archetypes[a].chunks[ch].entities.unsafe_ptr())
                    var c0_ptr = 0 if is_sp0 else self.archetypes[a].chunks[ch].column_ptr(col0)
                    slices.append(_ParSlice1(ents_ptr, c0_ptr, esz0, 0, cnt, global_base, is_sp0))
                    global_base += cnt

        if len(slices) > 0:
            var disp = _ParDispatch1[C0](
                Int(Pointer(to=self)),
                Int(Pointer(to=pool)),
                Int(slices.unsafe_ptr()),
                kernel,
                user_ctx,
                tid0,
                ex_sparse^,
                has_ex_sparse,
                rt_base,
                changed_since,
                read_only,
                self.current_tick,
            )
            run_parallel_ctx(pool, len(slices), disp, _par_worker1[C0])
        for a in range(len(self.archetypes)):
            for ch in range(len(self.archetypes[a].chunks)):
                self.archetypes[a].chunks[ch].keep_alive()
        for s in range(len(self.sparse_sets)):
            self.sparse_sets[s].keep_alive()
        self.keep_alive()
        var anchor_slices = len(slices)
        _ = anchor_slices

    def par_for_each2[C0: Component, C1: Component](
        mut self,
        mut pool: JobPool,
        kernel: def(Int, Int, Entity, Pointer[C0, MutAnyOrigin], Pointer[C1, MutAnyOrigin], Int) thin -> None,
        user_ctx: Int = 0,
        without_mask: ComponentBitSet = ComponentBitSet(),
        read_only: Bool = False,
        changed_since: UInt32 = 0,
    ):
        """Lifetime-safe parallel iteration over all entities with `(C0, C1)` across `JobPool`.
        Set `read_only=True` to avoid dirtying `changed_ticks`; set `changed_since > 0`
        to skip unchanged chunks/SIMD groups."""
        var tid0 = self.find_component_by_hash(C0.COMPONENT_ID)
        var tid1 = self.find_component_by_hash(C1.COMPONENT_ID)
        if tid0 < 0 or tid1 < 0:
            return
        var rt_base = _ecs_runtime_base() if _ecs_runtime_ensure() else 0
        var req_mask = ComponentBitSet().with_bit(tid0).with_bit(tid1)
        var req_table = req_mask.intersection(self.table_comp_mask)
        var ex_table = without_mask.intersection(self.table_comp_mask)
        var ex_sparse = without_mask.intersection(self.sparse_comp_mask)
        var has_ex_sparse = not ex_sparse.is_empty()
        var is_sp0 = (C0.STORAGE_MODE == STORAGE_SPARSE)
        var is_sp1 = (C1.STORAGE_MODE == STORAGE_SPARSE)
        var esz0 = self.components[tid0].byte_size
        var esz1 = self.components[tid1].byte_size

        var slices = List[_ParSlice2]()
        var global_base = 0
        for a in range(len(self.archetypes)):
            if not self.archetypes[a].table_mask.contains_all(req_table):
                continue
            if self.archetypes[a].table_mask.intersects(ex_table):
                continue
            var col0 = -1 if is_sp0 else self.archetypes[a].col_index(tid0)
            var col1 = -1 if is_sp1 else self.archetypes[a].col_index(tid1)
            for ch in range(len(self.archetypes[a].chunks)):
                var cnt = self.archetypes[a].chunks[ch].count
                if cnt <= 0:
                    continue
                if changed_since > 0 and not is_sp0 and not is_sp1:
                    var c0_chg = self.archetypes[a].chunks[ch].any_group_changed_since(col0, changed_since)
                    var c1_chg = self.archetypes[a].chunks[ch].any_group_changed_since(col1, changed_since)
                    if not c0_chg and not c1_chg:
                        continue
                    var ents_ptr = Int(self.archetypes[a].chunks[ch].entities.unsafe_ptr())
                    var c0_ptr = self.archetypes[a].chunks[ch].column_ptr(col0)
                    var c1_ptr = self.archetypes[a].chunks[ch].column_ptr(col1)
                    var active_groups = (cnt + (SIMD_LANES - 1)) >> 3
                    for g in range(active_groups):
                        if (
                            self.archetypes[a].chunks[ch].group_changed_tick(col0, g) <= changed_since
                            and self.archetypes[a].chunks[ch].group_changed_tick(col1, g) <= changed_since
                        ):
                            continue
                        var rs = g << 3
                        var rc = min(cnt - rs, SIMD_LANES)
                        if not read_only:
                            self.archetypes[a].chunks[ch].mark_changed(col0, rs, self.current_tick)
                            self.archetypes[a].chunks[ch].mark_changed(col1, rs, self.current_tick)
                        slices.append(
                            _ParSlice2(ents_ptr, c0_ptr, esz0, c1_ptr, esz1, rs, rc, global_base, False, False, 0, col0, col1)
                        )
                        global_base += rc
                else:
                    if changed_since == 0:
                        if not is_sp0 and not read_only:
                            self.archetypes[a].chunks[ch].mark_all_groups_changed(col0, self.current_tick)
                        if not is_sp1 and not read_only:
                            self.archetypes[a].chunks[ch].mark_all_groups_changed(col1, self.current_tick)
                        if (is_sp0 or is_sp1) and not read_only:
                            for r in range(cnt):
                                var e = self.archetypes[a].chunks[ch].entities[r]
                                ref sm = self.entities[Int(e.index)].sparse_mask
                                if has_ex_sparse and sm.intersects(ex_sparse):
                                    continue
                                if (is_sp0 and not sm.test(tid0)) or (is_sp1 and not sm.test(tid1)):
                                    continue
                                if is_sp0:
                                    var sidx0 = self.comp_to_sparse[tid0]
                                    var di0 = self.sparse_sets[sidx0].dense_index_of(e.index)
                                    if di0 >= 0:
                                        self.sparse_sets[sidx0].changed_ticks[di0] = self.current_tick
                                if is_sp1:
                                    var sidx1 = self.comp_to_sparse[tid1]
                                    var di1 = self.sparse_sets[sidx1].dense_index_of(e.index)
                                    if di1 >= 0:
                                        self.sparse_sets[sidx1].changed_ticks[di1] = self.current_tick
                    var ents_ptr = Int(self.archetypes[a].chunks[ch].entities.unsafe_ptr())
                    var c0_ptr = 0 if is_sp0 else self.archetypes[a].chunks[ch].column_ptr(col0)
                    var c1_ptr = 0 if is_sp1 else self.archetypes[a].chunks[ch].column_ptr(col1)
                    var chunk_addr = Int(Pointer(to=self.archetypes[a].chunks[ch]))
                    slices.append(
                        _ParSlice2(
                            ents_ptr, c0_ptr, esz0, c1_ptr, esz1, 0, cnt, global_base, is_sp0, is_sp1, chunk_addr, col0, col1
                        )
                    )
                    global_base += cnt

        if len(slices) > 0:
            var disp = _ParDispatch2[C0, C1](
                Int(Pointer(to=self)),
                Int(Pointer(to=pool)),
                Int(slices.unsafe_ptr()),
                kernel,
                user_ctx,
                tid0,
                tid1,
                ex_sparse^,
                has_ex_sparse,
                rt_base,
                changed_since,
                read_only,
                self.current_tick,
            )
            run_parallel_ctx(pool, len(slices), disp, _par_worker2[C0, C1])
        for a in range(len(self.archetypes)):
            for ch in range(len(self.archetypes[a].chunks)):
                self.archetypes[a].chunks[ch].keep_alive()
        for s in range(len(self.sparse_sets)):
            self.sparse_sets[s].keep_alive()
        self.keep_alive()
        var anchor_slices = len(slices)
        _ = anchor_slices

    def par_for_each3[C0: Component, C1: Component, C2: Component](
        mut self,
        mut pool: JobPool,
        kernel: def(
            Int,
            Int,
            Entity,
            Pointer[C0, MutAnyOrigin],
            Pointer[C1, MutAnyOrigin],
            Pointer[C2, MutAnyOrigin],
            Int,
        ) thin -> None,
        user_ctx: Int = 0,
        without_mask: ComponentBitSet = ComponentBitSet(),
        read_only: Bool = False,
        changed_since: UInt32 = 0,
    ):
        """Lifetime-safe parallel iteration over all entities with `(C0, C1, C2)` across `JobPool`.
        Set `read_only=True` to avoid dirtying `changed_ticks`; set `changed_since > 0`
        to skip unchanged chunks/SIMD groups."""
        var tid0 = self.find_component_by_hash(C0.COMPONENT_ID)
        var tid1 = self.find_component_by_hash(C1.COMPONENT_ID)
        var tid2 = self.find_component_by_hash(C2.COMPONENT_ID)
        if tid0 < 0 or tid1 < 0 or tid2 < 0:
            return
        var rt_base = _ecs_runtime_base() if _ecs_runtime_ensure() else 0
        var req_mask = ComponentBitSet().with_bit(tid0).with_bit(tid1).with_bit(tid2)
        var req_table = req_mask.intersection(self.table_comp_mask)
        var ex_table = without_mask.intersection(self.table_comp_mask)
        var ex_sparse = without_mask.intersection(self.sparse_comp_mask)
        var has_ex_sparse = not ex_sparse.is_empty()
        var is_sp0 = (C0.STORAGE_MODE == STORAGE_SPARSE)
        var is_sp1 = (C1.STORAGE_MODE == STORAGE_SPARSE)
        var is_sp2 = (C2.STORAGE_MODE == STORAGE_SPARSE)
        var esz0 = self.components[tid0].byte_size
        var esz1 = self.components[tid1].byte_size
        var esz2 = self.components[tid2].byte_size

        var slices = List[_ParSlice3]()
        var global_base = 0
        for a in range(len(self.archetypes)):
            if not self.archetypes[a].table_mask.contains_all(req_table):
                continue
            if self.archetypes[a].table_mask.intersects(ex_table):
                continue
            var col0 = -1 if is_sp0 else self.archetypes[a].col_index(tid0)
            var col1 = -1 if is_sp1 else self.archetypes[a].col_index(tid1)
            var col2 = -1 if is_sp2 else self.archetypes[a].col_index(tid2)
            for ch in range(len(self.archetypes[a].chunks)):
                var cnt = self.archetypes[a].chunks[ch].count
                if cnt <= 0:
                    continue
                if changed_since > 0 and not is_sp0 and not is_sp1 and not is_sp2:
                    var any_chg = (
                        self.archetypes[a].chunks[ch].any_group_changed_since(col0, changed_since)
                        or self.archetypes[a].chunks[ch].any_group_changed_since(col1, changed_since)
                        or self.archetypes[a].chunks[ch].any_group_changed_since(col2, changed_since)
                    )
                    if not any_chg:
                        continue
                    var ents_ptr = Int(self.archetypes[a].chunks[ch].entities.unsafe_ptr())
                    var c0_ptr = self.archetypes[a].chunks[ch].column_ptr(col0)
                    var c1_ptr = self.archetypes[a].chunks[ch].column_ptr(col1)
                    var c2_ptr = self.archetypes[a].chunks[ch].column_ptr(col2)
                    var active_groups = (cnt + (SIMD_LANES - 1)) >> 3
                    for g in range(active_groups):
                        if (
                            self.archetypes[a].chunks[ch].group_changed_tick(col0, g) <= changed_since
                            and self.archetypes[a].chunks[ch].group_changed_tick(col1, g) <= changed_since
                            and self.archetypes[a].chunks[ch].group_changed_tick(col2, g) <= changed_since
                        ):
                            continue
                        var rs = g << 3
                        var rc = min(cnt - rs, SIMD_LANES)
                        if not read_only:
                            self.archetypes[a].chunks[ch].mark_changed(col0, rs, self.current_tick)
                            self.archetypes[a].chunks[ch].mark_changed(col1, rs, self.current_tick)
                            self.archetypes[a].chunks[ch].mark_changed(col2, rs, self.current_tick)
                        slices.append(
                            _ParSlice3(
                                ents_ptr, c0_ptr, esz0, c1_ptr, esz1, c2_ptr, esz2, rs, rc, global_base, False, False, False, 0, col0, col1, col2
                            )
                        )
                        global_base += rc
                else:
                    if changed_since == 0:
                        if not is_sp0 and not read_only:
                            self.archetypes[a].chunks[ch].mark_all_groups_changed(col0, self.current_tick)
                        if not is_sp1 and not read_only:
                            self.archetypes[a].chunks[ch].mark_all_groups_changed(col1, self.current_tick)
                        if not is_sp2 and not read_only:
                            self.archetypes[a].chunks[ch].mark_all_groups_changed(col2, self.current_tick)
                        if (is_sp0 or is_sp1 or is_sp2) and not read_only:
                            for r in range(cnt):
                                var e = self.archetypes[a].chunks[ch].entities[r]
                                ref sm = self.entities[Int(e.index)].sparse_mask
                                if has_ex_sparse and sm.intersects(ex_sparse):
                                    continue
                                if (
                                    (is_sp0 and not sm.test(tid0))
                                    or (is_sp1 and not sm.test(tid1))
                                    or (is_sp2 and not sm.test(tid2))
                                ):
                                    continue
                                if is_sp0:
                                    var sidx0 = self.comp_to_sparse[tid0]
                                    var di0 = self.sparse_sets[sidx0].dense_index_of(e.index)
                                    if di0 >= 0:
                                        self.sparse_sets[sidx0].changed_ticks[di0] = self.current_tick
                                if is_sp1:
                                    var sidx1 = self.comp_to_sparse[tid1]
                                    var di1 = self.sparse_sets[sidx1].dense_index_of(e.index)
                                    if di1 >= 0:
                                        self.sparse_sets[sidx1].changed_ticks[di1] = self.current_tick
                                if is_sp2:
                                    var sidx2 = self.comp_to_sparse[tid2]
                                    var di2 = self.sparse_sets[sidx2].dense_index_of(e.index)
                                    if di2 >= 0:
                                        self.sparse_sets[sidx2].changed_ticks[di2] = self.current_tick
                    var ents_ptr = Int(self.archetypes[a].chunks[ch].entities.unsafe_ptr())
                    var c0_ptr = 0 if is_sp0 else self.archetypes[a].chunks[ch].column_ptr(col0)
                    var c1_ptr = 0 if is_sp1 else self.archetypes[a].chunks[ch].column_ptr(col1)
                    var c2_ptr = 0 if is_sp2 else self.archetypes[a].chunks[ch].column_ptr(col2)
                    var chunk_addr = Int(Pointer(to=self.archetypes[a].chunks[ch]))
                    slices.append(
                        _ParSlice3(
                            ents_ptr, c0_ptr, esz0, c1_ptr, esz1, c2_ptr, esz2, 0, cnt, global_base, is_sp0, is_sp1, is_sp2, chunk_addr, col0, col1, col2
                        )
                    )
                    global_base += cnt

        if len(slices) > 0:
            var disp = _ParDispatch3[C0, C1, C2](
                Int(Pointer(to=self)),
                Int(Pointer(to=pool)),
                Int(slices.unsafe_ptr()),
                kernel,
                user_ctx,
                tid0,
                tid1,
                tid2,
                ex_sparse^,
                has_ex_sparse,
                rt_base,
                changed_since,
                read_only,
                self.current_tick,
            )
            run_parallel_ctx(pool, len(slices), disp, _par_worker3[C0, C1, C2])
        for a in range(len(self.archetypes)):
            for ch in range(len(self.archetypes[a].chunks)):
                self.archetypes[a].chunks[ch].keep_alive()
        for s in range(len(self.sparse_sets)):
            self.sparse_sets[s].keep_alive()
        self.keep_alive()
        var anchor_slices = len(slices)
        _ = anchor_slices

    # ---- Deterministic Command Playback ----

    @always_inline
    def _resolve_provisional(self, ent: Entity, prov_remap: List[List[Entity]]) -> Entity:
        if not ent.is_provisional():
            return ent
        var l = ent.provisional_lane()
        var idx = Int(ent.index)
        if l >= 0 and l < len(prov_remap) and idx >= 0 and idx < len(prov_remap[l]):
            return prov_remap[l][idx]
        return NULL_ENTITY

    def flush_commands(mut self, mut cmds: ParallelCommands):
        """Replays all per-worker commands in deterministic `detsort` order (`order_key`, active
        `par_for_each` implicit entity order, or `(lane_id, seq)`), remapping provisional `Entity`
        handles cleanly regardless of OS worker thread scheduling."""
        var total = cmds.pending_count()
        if total == 0:
            return

        var flat = List[_RecordedCmd](capacity=total)
        var keys = List[Int](capacity=total)
        for l in range(MAX_COMMAND_LANES):
            ref cl = cmds.lanes[l]
            for i in range(len(cl.cmds)):
                var cmd = cl.cmds[i]
                flat.append(cmd)
                keys.append(Int(cmd.sort_key))

        var order = stable_order(keys)

        var prov_remap = List[List[Entity]](capacity=MAX_COMMAND_LANES)
        for l in range(MAX_COMMAND_LANES):
            var n_prov = Int(cmds.lanes[l].next_prov_index)
            prov_remap.append(List[Entity](length=n_prov, fill=NULL_ENTITY))

        var self_addr = Int(Pointer(to=self))
        for k in range(len(order)):
            var cmd = flat[order[k]]
            if cmd.op == CMD_SPAWN:
                var real_e = self.spawn_empty()
                prov_remap[cmd.lane_id][Int(cmd.entity.index)] = real_e
            elif cmd.op == CMD_DESPAWN:
                var target = self._resolve_provisional(cmd.entity, prov_remap)
                if target.valid():
                    _ = self.despawn(target)
            elif cmd.op == CMD_INSERT:
                var target = self._resolve_provisional(cmd.entity, prov_remap)
                var payload_addr = cmds.lanes[cmd.lane_id].payload_ptr + cmd.payload_offset
                if target.valid() and self.is_alive(target):
                    var afn = cmd.apply_fn
                    afn(self_addr, target, payload_addr)
                elif cmd.payload_size > 0 and not cmd.is_trivial_pod:
                    var dfn = cmd.drop_fn
                    dfn(payload_addr)
            elif cmd.op == CMD_REMOVE:
                var target = self._resolve_provisional(cmd.entity, prov_remap)
                if target.valid():
                    var tid = self.find_component_by_hash(cmd.component_hash)
                    if tid >= 0:
                        _ = self.remove_by_type_id(target, tid)

        # All payloads have been moved into World or dropped above; reset lanes without double-dropping
        for l in range(MAX_COMMAND_LANES):
            cmds.lanes[l]._reset_counters()

    # ---- Fast Checksum & <0.2 ms Binary World Snapshot / Restore ----

    def state_checksum(self) -> UInt64:
        """Computes a registration-order-independent deterministic 64-bit hash of the ECS World."""
        var h = mix64(UInt64(0xEC520005EC520005) ^ UInt64(self.live_entities))
        for i in range(len(self.entities)):
            ref rec = self.entities[i]
            if rec.alive:
                h = mix64(h ^ (UInt64(i) | (UInt64(rec.generation) << 32)))
                var sm_hash = UInt64(0)
                var wc = rec.sparse_mask.trimmed_word_count()
                for wi in range(wc):
                    var w = rec.sparse_mask.word_at(wi)
                    while w != 0:
                        var stid = wi * 64 + _ctz64(w)
                        w &= w - 1
                        if stid >= 0 and stid < len(self.components):
                            sm_hash ^= mix64(self.components[stid].component_hash)
                h = mix64(h ^ sm_hash)

        var arch_acc = UInt64(0)
        for a in range(len(self.archetypes)):
            ref arch = self.archetypes[a]
            var arch_live = arch.entity_count()
            if arch_live == 0:
                continue
            var sig_hash = UInt64(0xA2C47EA100000001)
            for c in range(len(arch.comp_type_ids)):
                sig_hash ^= mix64(self.components[arch.comp_type_ids[c]].component_hash)
            var ah = mix64(sig_hash ^ UInt64(arch_live))
            for ch in range(len(arch.chunks)):
                ref chunk = arch.chunks[ch]
                if chunk.count <= 0:
                    continue
                ah = mix64(
                    ah ^ fast_hash64(Int(chunk.entities.unsafe_ptr()), chunk.count * size_of[Entity]())
                )
                var cols_acc = UInt64(0)
                for c in range(len(arch.comp_type_ids)):
                    var tid = arch.comp_type_ids[c]
                    var info = self.components[tid]
                    var col_h = mix64(info.component_hash)
                    if info.byte_size > 0:
                        if info.has_custom_bin:
                            var tmp = BlobWriter()
                            var w_addr = Int(Pointer(to=tmp))
                            var base = chunk.column_ptr(c)
                            for r in range(chunk.count):
                                info.write_custom_bin(base + r * info.byte_size, w_addr)
                            col_h = mix64(col_h ^ fast_hash64(Int(tmp.data.unsafe_ptr()), len(tmp.data)))
                            var anchor_tmp = len(tmp.data)
                            _ = anchor_tmp
                        else:
                            col_h = mix64(
                                col_h ^ fast_hash64(chunk.column_ptr(c), chunk.count * info.byte_size)
                            )
                    cols_acc ^= mix64(col_h)
                ah = mix64(ah ^ cols_acc)
            arch_acc ^= mix64(ah)
        h = mix64(h ^ arch_acc)

        var sparse_acc = UInt64(0)
        for s in range(len(self.sparse_sets)):
            ref sset = self.sparse_sets[s]
            var scnt = len(sset.dense_entities)
            if scnt <= 0:
                continue
            var info = self.components[sset.type_id]
            var sh = mix64(info.component_hash ^ UInt64(scnt))
            sh = mix64(sh ^ fast_hash64(Int(sset.dense_entities.unsafe_ptr()), scnt * size_of[Entity]()))
            if info.byte_size > 0:
                if info.has_custom_bin:
                    var tmp = BlobWriter()
                    var w_addr = Int(Pointer(to=tmp))
                    for r in range(scnt):
                        info.write_custom_bin(sset.value_ptr(r), w_addr)
                    sh = mix64(sh ^ fast_hash64(Int(tmp.data.unsafe_ptr()), len(tmp.data)))
                    var anchor_tmp2 = len(tmp.data)
                    _ = anchor_tmp2
                else:
                    sh = mix64(sh ^ fast_hash64(sset.data_ptr, scnt * info.byte_size))
            sparse_acc ^= mix64(sh)
        return mix64(h ^ sparse_acc)

    def snapshot_blob(self) raises -> List[UInt8]:
        """Returns a portable binary snapshot of this `World` as `List[UInt8]`."""
        var writer = BlobWriter()
        self.snapshot_blob(writer)
        var out = List[UInt8]()
        swap(out, writer.data)
        return out^

    def snapshot_blob(self, mut out: BlobWriter) raises:
        """Writes a portable `<0.2 ms` binary snapshot of component metadata, entities, archetypes,
        and sparse sets. Raises `Error` if any live non-trivial component lacks binary hooks."""
        # Pre-validate before mutating `out` so a failing snapshot leaves `out` intact.
        for a in range(len(self.archetypes)):
            ref arch = self.archetypes[a]
            if arch.entity_count() > 0:
                for c in range(len(arch.comp_type_ids)):
                    var tid = arch.comp_type_ids[c]
                    var info = self.components[tid]
                    if info.byte_size > 0 and not info.has_custom_bin and not info.is_trivial_pod:
                        raise Error(
                            "World.snapshot_blob: non-trivial component "
                            + String(info.component_hash)
                            + " lacks binary serialization hooks"
                        )
        for s in range(len(self.sparse_sets)):
            ref sset = self.sparse_sets[s]
            if len(sset.dense_entities) > 0:
                var info = self.components[sset.type_id]
                if info.byte_size > 0 and not info.has_custom_bin and not info.is_trivial_pod:
                    raise Error(
                        "World.snapshot_blob: non-trivial sparse component "
                        + String(info.component_hash)
                        + " lacks binary serialization hooks"
                    )

        out.begin(WORLD_BLOB_TAG, WORLD_BLOB_VERSION)
        out.u32(self.current_tick)
        out.integer(self.live_entities)
        out.integer(len(self.entities))
        out.integer(len(self.free_indices))

        # Compute which registered components are actually referenced by non-empty storage
        var used_mask = ComponentBitSet()
        var non_empty_archs = 0
        for a in range(len(self.archetypes)):
            if self.archetypes[a].entity_count() > 0:
                non_empty_archs += 1
                used_mask = used_mask.union(self.archetypes[a].table_mask)
        var non_empty_sparse = 0
        for s in range(len(self.sparse_sets)):
            if len(self.sparse_sets[s].dense_entities) > 0:
                non_empty_sparse += 1
                used_mask.set(self.sparse_sets[s].type_id)

        # Registered component metadata table for saved_i -> local_i hash remapping
        out.integer(len(self.components))
        for i in range(len(self.components)):
            var info = self.components[i]
            out.u64(info.component_hash)
            out.integer(info.storage_mode)
            out.integer(info.byte_size)
            out.boolean(used_mask.test(i))

        for i in range(len(self.entities)):
            out.u32(self.entities[i].generation)
            out.boolean(self.entities[i].alive)
        if len(self.free_indices) > 0:
            out.write_bytes_raw(
                Int(self.free_indices.unsafe_ptr()),
                len(self.free_indices) * size_of[UInt32](),
            )

        out.integer(non_empty_archs)
        var out_addr = Int(Pointer(to=out))

        for a in range(len(self.archetypes)):
            ref arch = self.archetypes[a]
            if arch.entity_count() == 0:
                continue
            var wc = arch.table_mask.trimmed_word_count()
            out.u32(UInt32(wc))
            for wi in range(wc):
                out.u64(arch.table_mask.word_at(wi))
            out.integer(len(arch.chunks))
            for ch in range(len(arch.chunks)):
                ref chunk = arch.chunks[ch]
                out.integer(chunk.count)
                if chunk.count > 0:
                    out.write_bytes_raw(
                        Int(chunk.entities.unsafe_ptr()),
                        chunk.count * size_of[Entity](),
                    )
                    out.integer(len(arch.comp_type_ids))
                    for c in range(len(arch.comp_type_ids)):
                        var tid = arch.comp_type_ids[c]
                        var info = self.components[tid]
                        out.u64(info.component_hash)
                        if info.byte_size > 0:
                            var col_ptr = chunk.column_ptr(c)
                            if info.has_custom_bin:
                                out.u8(1)
                                for r in range(chunk.count):
                                    info.write_custom_bin(col_ptr + r * info.byte_size, out_addr)
                            else:
                                out.u8(0)
                                out.write_bytes_raw(col_ptr, chunk.count * info.byte_size)

        out.integer(non_empty_sparse)
        for s in range(len(self.sparse_sets)):
            ref sset = self.sparse_sets[s]
            var scnt = len(sset.dense_entities)
            if scnt == 0:
                continue
            var info = self.components[sset.type_id]
            out.u64(info.component_hash)
            out.integer(scnt)
            out.write_bytes_raw(
                Int(sset.dense_entities.unsafe_ptr()),
                scnt * size_of[Entity](),
            )
            if info.byte_size > 0:
                if info.has_custom_bin:
                    out.u8(1)
                    for r in range(scnt):
                        info.write_custom_bin(sset.value_ptr(r), out_addr)
                else:
                    out.u8(0)
                    out.write_bytes_raw(sset.data_ptr, scnt * info.byte_size)
        out.end()

    def _abort_partial_chunk(
        self,
        mut chunk: ArchetypeChunk,
        col_ready: List[Bool],
        partial_col: Int,
        partial_rows: Int,
    ):
        for c in range(len(chunk.comp_type_ids)):
            var tid = chunk.comp_type_ids[c]
            var info = self.components[tid]
            if info.byte_size > 0 and not info.is_trivial_pod:
                var base = chunk.column_ptr(c)
                if col_ready[c]:
                    for r in range(chunk.count):
                        info.drop_at(base + r * info.byte_size)
                elif c == partial_col:
                    for r in range(partial_rows):
                        info.drop_at(base + r * info.byte_size)
        chunk.count = 0
        chunk.entities.clear()

    def _find_or_create_staged_archetype(
        self,
        mut staged_archetypes: List[Archetype],
        mask: ComponentBitSet,
    ) -> Int:
        for i in range(len(staged_archetypes)):
            if staged_archetypes[i].table_mask == mask:
                return i
        var ids = List[Int]()
        var sizes = List[Int]()
        var wc = mask.trimmed_word_count()
        for wi in range(wc):
            var w = mask.word_at(wi)
            while w != 0:
                var tid = wi * 64 + _ctz64(w)
                w &= w - 1
                ids.append(tid)
                sizes.append(self.components[tid].byte_size)
        var new_id = len(staged_archetypes)
        staged_archetypes.append(Archetype(new_id, mask, ids, sizes, self.chunk_rows))
        return new_id

    def _drop_staged_state(
        self,
        mut staged_archetypes: List[Archetype],
        mut staged_sparse: List[SparseSet],
    ):
        for a in range(len(staged_archetypes)):
            ref arch = staged_archetypes[a]
            for ch in range(len(arch.chunks)):
                ref chunk = arch.chunks[ch]
                for c in range(len(arch.comp_type_ids)):
                    var tid = arch.comp_type_ids[c]
                    var info = self.components[tid]
                    if info.byte_size > 0 and not info.is_trivial_pod:
                        var base = chunk.column_ptr(c)
                        for r in range(chunk.count):
                            info.drop_at(base + r * info.byte_size)
                chunk.count = 0
                chunk.entities.clear()
        for s in range(len(staged_sparse)):
            var tid = staged_sparse[s].type_id
            staged_sparse[s].clear_with_drop(self.components[tid])

    def restore_blob(mut self, data: Span[UInt8, _]) raises:
        """Restores `World` from raw `snapshot_blob` bytes."""
        var buf = List[UInt8](capacity=max(1, len(data)))
        if len(data) > 0:
            buf.resize(unsafe_uninit_length=len(data))
            _ = external_call["memcpy", Int](
                Int(buf.unsafe_ptr()), Int(data.unsafe_ptr()), len(data)
            )
        var reader = BlobReader(buf^)
        self.restore_blob(reader)

    def restore_blob(mut self, mut reader: BlobReader) raises:
        """Restores `World` from a `snapshot_blob` buffer (`<0.2 ms`), remapping component IDs by
        `COMPONENT_ID` hash and staging all data before mutating `self` so any error leaves `self` untouched."""
        var blob_ver = reader.enter(WORLD_BLOB_TAG, WORLD_BLOB_VERSION)
        if blob_ver != WORLD_BLOB_VERSION:
            raise Error("World.restore_blob: unsupported ECS2 blob version " + String(blob_ver))
        var saved_tick = reader.u32()
        var saved_live = reader.integer()
        var ent_len = reader.integer()
        var free_len = reader.integer()
        var saved_comp_count = reader.integer()
        if (
            reader.failed
            or saved_live < 0
            or ent_len < 0
            or free_len < 0
            or saved_comp_count < 0
            or saved_live > ent_len
        ):
            raise Error("World.restore_blob: corrupted ECS2 header")
        if (
            not _blob_fits(reader, ent_len, 5)
            or not _blob_fits(reader, free_len, size_of[UInt32]())
            or not _blob_fits(reader, saved_comp_count, 25)
        ):
            raise Error("World.restore_blob: header counts exceed blob size")

        # Validate & remap saved_i -> local_i BEFORE staging/mutating self!
        var saved_to_local = List[Int](length=saved_comp_count, fill=-1)
        for si in range(saved_comp_count):
            var chash = reader.u64()
            var smode = reader.integer()
            var bsz = reader.integer()
            var is_used = reader.boolean()
            if reader.failed:
                raise Error("World.restore_blob: truncated component metadata table")
            var local_tid = self.find_component_by_hash(chash)
            if is_used:
                if local_tid < 0:
                    raise Error("World.restore_blob: unregistered component ID " + String(chash))
                if self.components[local_tid].storage_mode != smode:
                    raise Error(
                        "World.restore_blob: storage mode mismatch for component ID " + String(chash)
                    )
                if self.components[local_tid].byte_size != bsz:
                    raise Error(
                        "World.restore_blob: byte size mismatch for component ID " + String(chash)
                    )
                saved_to_local[si] = local_tid
            elif local_tid >= 0 and self.components[local_tid].storage_mode == smode and self.components[local_tid].byte_size == bsz:
                saved_to_local[si] = local_tid

        var staged_entities = List[EntityRecord](capacity=max(1, ent_len))
        var alive_count = 0
        for _ in range(ent_len):
            var gen = reader.u32()
            var is_alive = reader.boolean()
            if reader.failed:
                raise Error("World.restore_blob: truncated ECS2 entity records")
            if gen == 0 or (gen & PROVISIONAL_GEN_BIT) != 0 or gen > MAX_LIVE_GENERATION:
                raise Error("World.restore_blob: invalid or provisional entity generation")
            staged_entities.append(EntityRecord(gen, is_alive, -1, -1, -1, ComponentBitSet()))
            if is_alive:
                alive_count += 1
        if alive_count != saved_live:
            raise Error("World.restore_blob: live entity count does not match entity records")

        var staged_free = List[UInt32](capacity=max(1, free_len))
        staged_free.resize(unsafe_uninit_length=free_len)
        if free_len > 0:
            if not reader.read_bytes_raw(
                Int(staged_free.unsafe_ptr()), free_len * size_of[UInt32]()
            ):
                raise Error("World.restore_blob: truncated ECS2 free list")
        var seen_free = List[Bool](length=ent_len, fill=False)
        for i in range(free_len):
            var fidx = Int(staged_free[i])
            if fidx < 0 or fidx >= ent_len or staged_entities[fidx].alive or seen_free[fidx]:
                raise Error("World.restore_blob: free list names a live, duplicate, or out-of-range entity")
            seen_free[fidx] = True

        # Decode archetypes, chunks, and sparse sets into staged storage before mutating `self`.
        var staged_archetypes = List[Archetype]()
        var empty_ids = List[Int]()
        var empty_sizes = List[Int]()
        staged_archetypes.append(Archetype(0, ComponentBitSet(), empty_ids, empty_sizes, self.chunk_rows))
        var staged_sparse = List[SparseSet]()
        var staged_sparse_sidx = List[Int]()

        var reader_addr = Int(Pointer(to=reader))
        var num_archs = reader.integer()
        if num_archs < 0 or reader.failed or not _blob_fits(reader, num_archs, 12):
            raise Error("World.restore_blob: corrupted archetype count")

        for _ in range(num_archs):
            var wc = Int(reader.u32())
            if wc < 0 or reader.failed or not _blob_fits(reader, wc, 8):
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: corrupted archetype mask word count")
            var local_mask = ComponentBitSet()
            for wi in range(wc):
                var word = reader.u64()
                if reader.failed:
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: truncated archetype mask words")
                while word != 0:
                    var saved_tid = wi * 64 + _ctz64(word)
                    word &= word - 1
                    if saved_tid < 0 or saved_tid >= saved_comp_count or saved_to_local[saved_tid] < 0:
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: unregistered component ID in archetype mask")
                    var local_tid = saved_to_local[saved_tid]
                    if not self.table_comp_mask.test(local_tid):
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: non-table component in archetype table_mask")
                    local_mask.set(local_tid)

            var dst_arch = self._find_or_create_staged_archetype(staged_archetypes, local_mask)
            var num_chunks = reader.integer()
            if num_chunks < 0 or reader.failed or not _blob_fits(reader, num_chunks, 8):
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: corrupted chunk count")

            for _ in range(num_chunks):
                var cnt = reader.integer()
                if cnt < 0 or reader.failed or not _blob_fits(reader, cnt, size_of[Entity]()):
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: corrupted chunk row count")
                var dst_ch = len(staged_archetypes[dst_arch].chunks)
                var chunk = ArchetypeChunk(
                    staged_archetypes[dst_arch].comp_type_ids,
                    staged_archetypes[dst_arch].col_elem_sizes,
                    max(self.chunk_rows, cnt),
                )
                chunk.count = cnt
                if cnt > 0:
                    chunk.entities.resize(unsafe_uninit_length=cnt)
                    if not reader.read_bytes_raw(
                        Int(chunk.entities.unsafe_ptr()), cnt * size_of[Entity]()
                    ):
                        chunk.count = 0
                        chunk.entities.clear()
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: truncated ECS2 chunk entities")
                    for r in range(cnt):
                        var e = chunk.entities[r]
                        var eidx = Int(e.index)
                        if (
                            eidx < 0
                            or eidx >= len(staged_entities)
                            or not staged_entities[eidx].alive
                            or staged_entities[eidx].generation != e.generation
                            or staged_entities[eidx].archetype_id != -1
                        ):
                            chunk.count = 0
                            chunk.entities.clear()
                            self._drop_staged_state(staged_archetypes, staged_sparse)
                            raise Error(
                                "World.restore_blob: chunk names a dead, stale, duplicate or out-of-range entity"
                            )
                        staged_entities[eidx].archetype_id = Int32(dst_arch)
                        staged_entities[eidx].chunk_idx = Int32(dst_ch)
                        staged_entities[eidx].row_in_chunk = Int32(r)

                    var saved_cols = reader.integer()
                    if saved_cols != len(staged_archetypes[dst_arch].comp_type_ids) or reader.failed:
                        chunk.count = 0
                        chunk.entities.clear()
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: column count mismatch in archetype")

                    var col_ready = List[Bool](length=saved_cols, fill=False)
                    for _ in range(saved_cols):
                        var chash = reader.u64()
                        if reader.failed:
                            self._abort_partial_chunk(chunk, col_ready, -1, 0)
                            self._drop_staged_state(staged_archetypes, staged_sparse)
                            raise Error("World.restore_blob: truncated column component ID")
                        var tid = self.find_component_by_hash(chash)
                        if tid < 0:
                            self._abort_partial_chunk(chunk, col_ready, -1, 0)
                            self._drop_staged_state(staged_archetypes, staged_sparse)
                            raise Error("World.restore_blob: unregistered component ID " + String(chash))
                        var col_idx = staged_archetypes[dst_arch].col_index(tid)
                        if col_idx < 0:
                            self._abort_partial_chunk(chunk, col_ready, -1, 0)
                            self._drop_staged_state(staged_archetypes, staged_sparse)
                            raise Error("World.restore_blob: component column missing in archetype")
                        if col_ready[col_idx]:
                            self._abort_partial_chunk(chunk, col_ready, -1, 0)
                            self._drop_staged_state(staged_archetypes, staged_sparse)
                            raise Error("World.restore_blob: duplicate component column in archetype")
                        var info = self.components[tid]
                        chunk.mark_all_groups_changed(col_idx, saved_tick)
                        if info.byte_size > 0:
                            var col_ptr = chunk.column_ptr(col_idx)
                            if col_ptr == 0:
                                self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                self._drop_staged_state(staged_archetypes, staged_sparse)
                                raise Error("World.restore_blob: null column buffer")
                            var mode_tag = reader.u8()
                            if reader.failed:
                                self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                self._drop_staged_state(staged_archetypes, staged_sparse)
                                raise Error("World.restore_blob: truncated column encoding tag")
                            if mode_tag == 1:
                                if not info.has_custom_bin:
                                    self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                    self._drop_staged_state(staged_archetypes, staged_sparse)
                                    raise Error("World.restore_blob: missing custom binary codec")
                                for r in range(cnt):
                                    if not info.read_custom_bin(col_ptr + r * info.byte_size, reader_addr):
                                        self._abort_partial_chunk(chunk, col_ready, col_idx, r)
                                        self._drop_staged_state(staged_archetypes, staged_sparse)
                                        raise Error("World.restore_blob: custom component binary read failed")
                            elif mode_tag == 0:
                                if not info.is_trivial_pod:
                                    self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                    self._drop_staged_state(staged_archetypes, staged_sparse)
                                    raise Error("World.restore_blob: non-trivial component cannot raw-memcpy")
                                if not reader.read_bytes_raw(col_ptr, cnt * info.byte_size):
                                    self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                    self._drop_staged_state(staged_archetypes, staged_sparse)
                                    raise Error("World.restore_blob: truncated ECS2 column bytes")
                            elif mode_tag == 2:
                                self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                self._drop_staged_state(staged_archetypes, staged_sparse)
                                raise Error(
                                    "World.restore_blob: legacy Clone Vault tag 2 is unsupported for component ID "
                                    + String(chash)
                                )
                            else:
                                self._abort_partial_chunk(chunk, col_ready, -1, 0)
                                self._drop_staged_state(staged_archetypes, staged_sparse)
                                raise Error("World.restore_blob: unknown column encoding tag")
                        col_ready[col_idx] = True
                staged_archetypes[dst_arch].chunks.append(chunk^)

        for i in range(len(staged_entities)):
            if staged_entities[i].alive and staged_entities[i].archetype_id < 0:
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: live entity is missing from every archetype chunk")

        var num_sparse = reader.integer()
        if num_sparse < 0 or reader.failed or not _blob_fits(reader, num_sparse, 16):
            self._drop_staged_state(staged_archetypes, staged_sparse)
            raise Error("World.restore_blob: corrupted sparse set count")

        var seen_sparse = ComponentBitSet()
        for _ in range(num_sparse):
            var chash = reader.u64()
            if reader.failed:
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: truncated sparse component ID")
            var tid = self.find_component_by_hash(chash)
            if tid < 0:
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: unregistered component ID " + String(chash))
            var scnt = reader.integer()
            if scnt < 0 or reader.failed or not _blob_fits(reader, scnt, size_of[Entity]()):
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: corrupted sparse entity count")
            var sidx = self.comp_to_sparse[tid]
            if sidx < 0:
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: component is not STORAGE_SPARSE")
            if seen_sparse.test(tid):
                self._drop_staged_state(staged_archetypes, staged_sparse)
                raise Error("World.restore_blob: duplicate sparse component in blob")
            seen_sparse.set(tid)

            var info = self.components[tid]
            var sset = SparseSet(tid, info.byte_size)
            sset._ensure_data_capacity(scnt, info)
            var sp_ents = List[Entity](capacity=max(1, scnt))
            sp_ents.resize(unsafe_uninit_length=scnt)
            if scnt > 0:
                if not reader.read_bytes_raw(Int(sp_ents.unsafe_ptr()), scnt * size_of[Entity]()):
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: truncated ECS2 sparse entities")

            if info.byte_size > 0 and scnt > 0:
                if sset.data_ptr == 0:
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: null sparse buffer")
                var mode_tag = reader.u8()
                if reader.failed:
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: truncated sparse encoding tag")
                if mode_tag == 1:
                    if not info.has_custom_bin:
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: missing sparse custom binary codec")
                    for di in range(scnt):
                        if not info.read_custom_bin(sset.data_ptr + di * info.byte_size, reader_addr):
                            for prev_di in range(di):
                                info.drop_at(sset.data_ptr + prev_di * info.byte_size)
                            self._drop_staged_state(staged_archetypes, staged_sparse)
                            raise Error("World.restore_blob: sparse custom binary read failed")
                elif mode_tag == 0:
                    if not info.is_trivial_pod:
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: non-trivial sparse component cannot raw-memcpy")
                    if not reader.read_bytes_raw(sset.data_ptr, scnt * info.byte_size):
                        self._drop_staged_state(staged_archetypes, staged_sparse)
                        raise Error("World.restore_blob: truncated ECS2 sparse bytes")
                elif mode_tag == 2:
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error(
                        "World.restore_blob: legacy Clone Vault tag 2 is unsupported for sparse component ID "
                        + String(chash)
                    )
                else:
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: unknown sparse encoding tag")

            sset.dense_entities = sp_ents^
            sset.changed_ticks = List[UInt32](length=scnt, fill=saved_tick)
            for di in range(scnt):
                var eidx = Int(sset.dense_entities[di].index)
                if (
                    eidx < 0
                    or eidx >= len(staged_entities)
                    or not staged_entities[eidx].alive
                    or staged_entities[eidx].generation != sset.dense_entities[di].generation
                    or staged_entities[eidx].sparse_mask.test(tid)
                ):
                    sset.clear_with_drop(info)
                    self._drop_staged_state(staged_archetypes, staged_sparse)
                    raise Error("World.restore_blob: sparse set names a dead, stale, duplicate or out-of-range entity")
                var page = eidx >> SPARSE_PAGE_BITS
                sset._ensure_page(page)
                sset.pages[page][eidx & SPARSE_PAGE_MASK] = Int32(di)
                staged_entities[eidx].sparse_mask.set(tid)

            staged_sparse.append(sset^)
            staged_sparse_sidx.append(sidx)

        try:
            reader.leave()
        except err:
            self._drop_staged_state(staged_archetypes, staged_sparse)
            raise Error(String(err))

        # Commit staged state atomically after all validation & `reader.leave()` succeed.
        self._drop_all_live_components()
        self.current_tick = saved_tick
        self.restore_epoch += 1
        self.live_entities = saved_live
        self.entities = staged_entities^
        self.free_indices = staged_free^
        self.archetypes = staged_archetypes^
        for i in range(len(staged_sparse)):
            var sidx = staged_sparse_sidx[i]
            swap(self.sparse_sets[sidx], staged_sparse[i])
