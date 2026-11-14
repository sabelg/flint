"""Entity handles and the slot allocator every entity table shares.

An `EntityId` is a table kind, a row index and the row's generation. A row's
generation goes up each time the row is freed, so an id kept after its entity
despawned no longer resolves (`Slots.resolve` returns -1) even when the row
has been reused.

`Slots` is the allocator: a fixed number of rows, each FREE, SPAWNING, LIVE or
DESPAWNING. Allocation always takes the lowest free row, so the same sequence
of spawns and despawns gives the same rows and ids on every run and machine.
During a simulation tick, spawns and despawns are deferred: a spawned row is
SPAWNING (its data is written, but systems skip it) and a despawned row is
DESPAWNING (systems still see it); `commit` at the end of the tick makes them
LIVE and FREE. Outside a tick (layer load, tools, tests that set up a scene)
they apply at once. The row states, generations and the high-water mark are
serialised with the table, so they are part of snapshots and checksums.
"""
from flint.core.serial import BlobWriter, BlobReader, fourcc

comptime KIND_NONE: Int = 0
comptime KIND_PLAYER: Int = 1
comptime KIND_ENEMY: Int = 2
comptime KIND_COLLECTIBLE: Int = 3
comptime KIND_PROP: Int = 4
comptime KIND_BODY: Int = 5      # voxel bodies: rubble, debris, prop pieces
comptime KIND_TRIGGER: Int = 6
comptime KIND_LIGHT: Int = 7
comptime KIND_PROJECTILE: Int = 8
comptime KIND_SYSTEM: Int = 9    # a system or the world itself as an event source
comptime KIND_COUNT: Int = 10

comptime SLOT_FREE: UInt8 = 0
comptime SLOT_SPAWNING: UInt8 = 1
comptime SLOT_LIVE: UInt8 = 2
comptime SLOT_DESPAWNING: UInt8 = 3

comptime GENERATION_MASK: UInt32 = 0xFFFFFF  # 24 bits in the packed form
comptime SLOTS_VERSION: Int = 1


def kind_name(kind: Int) -> String:
    var names = ["none", "player", "enemy", "collectible", "prop", "body", "trigger", "light", "projectile", "system"]
    return String(names[kind]) if kind >= 0 and kind < KIND_COUNT else "unknown"


struct EntityId(Copyable, ImplicitlyCopyable, Movable, Equatable):
    var kind: Int
    var index: Int
    var generation: UInt32

    def __init__(out self, kind: Int = KIND_NONE, index: Int = -1, generation: UInt32 = 0):
        self.kind = kind
        self.index = index
        self.generation = generation

    def __eq__(self, other: Self) -> Bool:
        return self.kind == other.kind and self.index == other.index and self.generation == other.generation

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def valid(self) -> Bool:
        return self.kind != KIND_NONE and self.index >= 0

    def packed(self) -> UInt64:
        """kind (8 bits) | generation (24 bits) | index (32 bits). Sorting by
        this orders sources by kind, then row, then generation."""
        if not self.valid():
            return 0
        return (UInt64(self.kind) << 56) | (UInt64(self.index) << 24) | UInt64(self.generation & GENERATION_MASK)

    def describe(self) -> String:
        if not self.valid():
            return "none"
        return kind_name(self.kind) + "#" + String(self.index) + "g" + String(self.generation)


def no_entity() -> EntityId:
    return EntityId()


def system_source(system: Int) -> EntityId:
    """A system (gameplay/world.mojo SYSTEM_*) as the source of an event."""
    return EntityId(KIND_SYSTEM, system, 0)


def unpack_entity(bits: UInt64) -> EntityId:
    if bits == 0:
        return EntityId()
    return EntityId(Int(bits >> 56), Int((bits >> 24) & 0xFFFFFFFF), UInt32(bits & UInt64(GENERATION_MASK)))


struct Slots(Copyable, Movable, Sized):
    var kind: Int
    var capacity: Int
    var state: List[UInt8]
    var generation: List[UInt32]
    var used: Int        # rows [0, used) may be occupied: the iteration bound
    var pending: Int     # SPAWNING plus DESPAWNING rows waiting for commit
    var peak: Int        # the highest `used` ever: the pool's high-water mark (not saved)

    def __init__(out self, kind: Int, capacity: Int):
        self.kind = kind
        self.capacity = capacity
        self.state = List[UInt8](length=capacity, fill=SLOT_FREE)
        self.generation = List[UInt32](length=capacity, fill=0)
        self.used = 0
        self.pending = 0
        self.peak = 0

    def __len__(self) -> Int:
        return self.used

    def grow(mut self, capacity: Int):
        """More rows at the end, all free (a pool resized at configuration
        time, package 2.2's body cap); rows in use are kept."""
        while self.capacity < capacity:
            self.state.append(SLOT_FREE)
            self.generation.append(0)
            self.capacity += 1

    def allocate(mut self, deferred: Bool) -> Int:
        """The lowest free row, now SPAWNING (deferred) or LIVE; -1 when full."""
        for index in range(self.capacity):
            if self.state[index] == SLOT_FREE:
                if deferred:
                    self.state[index] = SLOT_SPAWNING
                    self.pending += 1
                else:
                    self.state[index] = SLOT_LIVE
                if index >= self.used:
                    self.used = index + 1
                    if self.used > self.peak:
                        self.peak = self.used
                return index
        return -1

    def free_now(mut self, index: Int):
        if index < 0 or index >= self.capacity or self.state[index] == SLOT_FREE:
            return
        if self.state[index] == SLOT_SPAWNING or self.state[index] == SLOT_DESPAWNING:
            self.pending -= 1
        self.state[index] = SLOT_FREE
        self.generation[index] = (self.generation[index] + 1) & GENERATION_MASK
        self.shrink()

    def release(mut self, index: Int, deferred: Bool) -> Bool:
        """Despawns a row: DESPAWNING until commit (deferred) or FREE now.
        False when the row was not occupied or is already leaving."""
        if index < 0 or index >= self.capacity:
            return False
        var current = self.state[index]
        if current == SLOT_FREE or current == SLOT_DESPAWNING:
            return False
        if not deferred:
            self.free_now(index)
            return True
        if current == SLOT_SPAWNING:
            # Spawned and despawned in the same tick: it never goes live.
            self.state[index] = SLOT_DESPAWNING
            return True
        self.state[index] = SLOT_DESPAWNING
        self.pending += 1
        return True

    def shrink(mut self):
        while self.used > 0 and self.state[self.used - 1] == SLOT_FREE:
            self.used -= 1

    def commit(mut self, mut spawned: List[Int], mut freed: List[Int]):
        """End of tick: SPAWNING rows go live, DESPAWNING rows are freed (their
        generation goes up). The rows are appended in row order."""
        if self.pending == 0:
            return
        for index in range(self.used):
            if self.state[index] == SLOT_SPAWNING:
                self.state[index] = SLOT_LIVE
                spawned.append(index)
            elif self.state[index] == SLOT_DESPAWNING:
                self.state[index] = SLOT_FREE
                self.generation[index] = (self.generation[index] + 1) & GENERATION_MASK
                freed.append(index)
        self.pending = 0
        self.shrink()

    def clear(mut self):
        """Frees every row (a layer change). Generations of occupied rows go
        up, so ids from before the clear stop resolving."""
        for index in range(self.used):
            if self.state[index] != SLOT_FREE:
                self.state[index] = SLOT_FREE
                self.generation[index] = (self.generation[index] + 1) & GENERATION_MASK
        self.used = 0
        self.pending = 0

    def occupied(self, index: Int) -> Bool:
        """Visible to systems: LIVE or DESPAWNING (a row leaving at the end of
        this tick still takes part in it)."""
        if index < 0 or index >= self.used:
            return False
        var current = self.state[index]
        return current == SLOT_LIVE or current == SLOT_DESPAWNING

    def held(self, index: Int) -> Bool:
        """Not free: occupied or still spawning."""
        return index >= 0 and index < self.used and self.state[index] != SLOT_FREE

    def bytes(self) -> Int:
        """Heap bytes of the state and generation columns."""
        return self.state.capacity() + self.generation.capacity() * 4

    def count(self) -> Int:
        var total = 0
        for index in range(self.used):
            if self.occupied(index):
                total += 1
        return total

    def id(self, index: Int) -> EntityId:
        if not self.held(index):
            return EntityId()
        return EntityId(self.kind, index, self.generation[index])

    def resolve(self, id: EntityId) -> Int:
        """The row of a live id, or -1 for a stale, foreign or empty one."""
        if id.kind != self.kind or not self.held(id.index):
            return -1
        if self.generation[id.index] != id.generation:
            return -1
        return id.index

    def serialize(self, mut out: BlobWriter):
        out.begin(fourcc("SLOT"), SLOTS_VERSION)
        out.integer(self.capacity)
        out.integer(self.used)
        out.integer(self.pending)
        # Generations of rows past `used` matter too: they are the next ids.
        var touched = self.used
        for index in range(self.used, self.capacity):
            if self.generation[index] != 0:
                touched = index + 1
        out.integer(touched)
        for index in range(touched):
            out.u8(self.state[index])
            out.u32(self.generation[index])
        out.end()

    def deserialize(mut self, mut input: BlobReader) raises:
        _ = input.enter(fourcc("SLOT"), SLOTS_VERSION)
        var capacity = input.integer()
        if capacity != self.capacity:
            raise Error(kind_name(self.kind) + " table holds " + String(self.capacity) + " rows; the blob has " + String(capacity))
        self.used = input.integer()
        self.pending = input.integer()
        self.peak = max(self.peak, self.used)
        var touched = input.integer()
        if self.used < 0 or self.used > capacity or touched < self.used or touched > capacity:
            raise Error(kind_name(self.kind) + " table blob has bad row counts")
        for index in range(capacity):
            self.state[index] = SLOT_FREE
            self.generation[index] = 0
        for index in range(touched):
            self.state[index] = input.u8()
            self.generation[index] = input.u32()
        input.leave()
