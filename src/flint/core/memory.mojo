"""Memory: tags, charges, arenas, budgets and the report (package 0.11).

Every large allocation the engine makes is charged to a *tag* (terrain, level,
meshes, assets, physics, entities, particles, journal, frame, render, audio,
and five GPU tags for the buffers the amdgpu and Vulkan backends create). Per
tag the tracker keeps the live bytes, the peak, the live allocation count, the
allocations ever made, a budget and how often the budget was crossed.

The counters live in one fixed-address mapping, `MEMORY_BASE`, like the safety
anchor (core/anchor.mojo): Mojo has no globals, and allocation sites (the brick
pool, the GPU backend, the renderer's model table) have no shared object to
reach. They are atomics, so jobs on any thread can charge. The mapping is made
by the first `MemCharge` or `Arena` a process creates (`memory_install`, safe to
repeat and to race); anything that charges holds one, so a charge never touches
an unmapped address. If the address is taken by something else the charges
turn themselves off (tag -1) and the engine runs untracked.

Two ways to charge:

- `MemCharge`: one owner's bytes under one tag. `set(bytes)` moves the tag by
  the difference; destroying the charge gives the bytes back, and copying it
  charges again (a copied list is a second allocation). Owners whose size
  changes often (the edit journal, CPU framebuffers) are re-measured once per
  frame instead of on every append.
- `Arena`: a bump allocator over one block, charged at its capacity. `reset`
  frees everything in it at once. The level arena (VoxelWorld) lives as long as
  a layer; the frame arenas (`FrameArenas`, one per GPU frame set) are reset
  when that frame set's fence has signalled, so memory written for a frame stays
  valid until the GPU is finished with it.

Fixed pools (bricks, bodies, particles, entity tables) are charged at their
capacity; their live and peak counts are read from the owners (see
gameplay/memory_report.mojo), so counting costs nothing on the hot path.

Budgets come from the `mem.budget.<tag>` console variables, in MiB (0 = none).
`memory_check_budgets` runs once per frame on the main thread, outside ticks:
memory is presentation and never changes the simulation.
"""
from std.ffi import external_call, _Global
from std.atomic import Atomic, Ordering

comptime MEMORY_BYTES: Int = 16384
comptime MEMORY_MAGIC: Int = 0x314D454D4453  # "SDMEM1"
comptime _MEMORY_CTRL_KEY: StaticString = "FLINT_MEMORY_CTRL"

# Tags. CPU tags first, then the GPU ones from MEM_FIRST_GPU.
comptime MEM_TERRAIN: Int = 0     # brick pool: mixed 8^3 bricks
comptime MEM_LEVEL: Int = 1       # the level arena: chunk brick tables
comptime MEM_MESH: Int = 2        # CPU copies of the voxel chunk meshes
comptime MEM_ASSETS: Int = 3      # models, textures and clips from glb files
comptime MEM_PHYSICS: Int = 4     # the voxel-body pool
comptime MEM_ENTITIES: Int = 5    # the world's entity tables
comptime MEM_PARTICLES: Int = 6   # the particle pool
comptime MEM_JOURNAL: Int = 7     # the voxel edit journal
comptime MEM_FRAME: Int = 8       # frame arenas
comptime MEM_RENDER: Int = 9      # CPU framebuffers, HUD and vertex caches
comptime MEM_AUDIO: Int = 10      # decoded sound clips
comptime MEM_GPU_TARGETS: Int = 11   # colour, HUD, backdrop images
comptime MEM_GPU_FRAMES: Int = 12    # per-frame-set buffers: records, lists, tiles, shown image
comptime MEM_GPU_TERRAIN: Int = 13   # the voxel shadow volume
comptime MEM_GPU_TEXTURES: Int = 14  # the texture atlas
comptime MEM_GPU_SHADERS: Int = 15   # shader code, bloom cells, fault gate
comptime MEM_GPU_MESHES: Int = 16    # the chunk-mesh heap (package 1.6)
comptime MEM_TAGS: Int = 17
comptime MEM_FIRST_GPU: Int = 11
# Not tags: where GPU bytes live, charged beside the GPU tag of each buffer.
comptime MEM_DOMAIN_VRAM: Int = MEM_TAGS
comptime MEM_DOMAIN_GTT: Int = MEM_TAGS + 1

comptime MIB: Int = 1 << 20

# Word offsets in the mapping.
comptime W_MAGIC: Int = 0
comptime W_CPU_LIVE: Int = 64
comptime W_CPU_PEAK: Int = 72
comptime W_GPU_LIVE: Int = 80
comptime W_GPU_PEAK: Int = 88
comptime W_VRAM_LIVE: Int = 96
comptime W_VRAM_PEAK: Int = 104
comptime W_GTT_LIVE: Int = 112
comptime W_GTT_PEAK: Int = 120
comptime W_BREACHES: Int = 128  # budget crossings, all tags
comptime W_TAGS: Int = 256
comptime TAG_STRIDE: Int = 64
comptime T_LIVE: Int = 0
comptime T_PEAK: Int = 8
comptime T_COUNT: Int = 16
comptime T_ALLOCS: Int = 24
comptime T_BUDGET: Int = 32
comptime T_OVER: Int = 40
comptime T_BREACHES: Int = 48


def mem_tag_key(tag: Int) -> String:
    """The tag's short name: the JSON key and the end of its budget variable."""
    var keys = [
        "terrain", "level", "mesh", "assets", "physics", "entities", "particles",
        "journal", "frame", "render", "audio",
        "gpu.targets", "gpu.frames", "gpu.terrain", "gpu.textures", "gpu.shaders", "gpu.meshes",
    ]
    return String(keys[tag]) if tag >= 0 and tag < MEM_TAGS else String("unknown")


def mem_tag_gpu(tag: Int) -> Bool:
    return tag >= MEM_FIRST_GPU and tag < MEM_TAGS


def default_budget_mib(tag: Int) -> Float32:
    """Default budgets in MiB (0 = none), from the peaks measured on rog: the
    profiled `destruction` replay at 3440x1440 on amdgpu and every layer in the
    soak (docs/RESEARCH.md, 0.11). Pools and arenas of fixed size get their
    size with room; what grows with resolution (render, gpu.targets,
    gpu.frames) gets about twice the 3440x1440 figure, enough for 4K."""
    var budgets: List[Float32] = [
        8,     # terrain: the brick pool; 1.5 MiB measured, a 5 x 2 x 5 layer reserves 4.7
        1,     # level: chunk tables, 8 KiB for the largest layer today
        64,    # mesh: 26 MiB for Glenwood's chunk meshes
        384,   # assets: 244 MiB of models, textures and clips
        4,     # physics: the 300-body pool (2.8: LOOSE_BODY_CAP), its grids and samples, 3.1 MiB
        1,     # entities: the tables, 26 KiB
        1,     # particles: the 240-particle pool, 11 KiB
        16,    # journal: 64 KiB after the destruction run; grows with digging
        4,     # frame: three 1 MiB frame arenas
        192,   # render: 97 MiB at 3440x1440 (the CPU framebuffer and HUD images)
        8,     # audio: under 1 MiB of clips and music
        256,   # gpu.targets: 114 MiB at 3440x1440
        2048,  # gpu.frames: 986 MiB, mostly three 224 MiB triangle-record buffers
        192,   # gpu.terrain: the 128 MiB shadow volume and 34 MiB of GPU meshing (1.3)
        2304,  # gpu.textures: the atlas reserves 2 GiB (21 MiB of it in use)
        8,     # gpu.shaders: 2 MiB
        64,    # gpu.meshes: the 32 MiB chunk-mesh heap (package 1.6), 16 MiB of skinning heaps and 8 more once a mesh has morph targets (3.4)
    ]
    return budgets[tag] if tag >= 0 and tag < MEM_TAGS else Float32(0)


# ---------------------------------------------------------------------------
# The mapping (dynamic OS allocation via lock-free global control block)
# ---------------------------------------------------------------------------

def _memory_on_exit(ctrl: Int) abi("C"):
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


def _init_memory_ctrl() -> Int:
    var ctrl = external_call["aligned_alloc", Int](Int(64), Int(64))
    if ctrl == 0:
        return 0
    var p = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=ctrl)
    for i in range(8):
        p[unsafe_offset=i] = 0
    _ = external_call["__cxa_atexit", Int32](_memory_on_exit, ctrl, Int(0))
    return ctrl


comptime _MEMORY_GLOBAL = _Global["FLINT_MEMORY_CTRL", _init_memory_ctrl]


@always_inline
def _memory_ctrl_addr() -> Int:
    var slot = external_call["KGEN_CompilerRT_GetGlobalOrNull", Int](
        Int(_MEMORY_CTRL_KEY.unsafe_ptr()), _MEMORY_CTRL_KEY.byte_length()
    )
    if slot == 0:
        return 0
    return Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=slot)[unsafe_offset=0]


@always_inline
def _memory_ctrl_addr_or_create() -> Int:
    try:
        return _MEMORY_GLOBAL.get_or_create_ptr()[unsafe_offset=0]
    except:
        return 0


@always_inline
def memory_base() -> Int:
    """Base address of the process memory tracker, or 0 if not installed."""
    var ctrl = _memory_ctrl_addr()
    if ctrl == 0:
        return 0
    return Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())


@always_inline
def mem_word(offset: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
    return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=memory_base() + offset)


@always_inline
def tag_word(tag: Int, field: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
    return mem_word(W_TAGS + tag * TAG_STRIDE + field)


def memory_ready() -> Bool:
    """True when this process has the tracker allocated (zero-alloc if uninstalled)."""
    var base = memory_base()
    if base == 0:
        return False
    var magic = Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base + W_MAGIC)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
    return magic == MEMORY_MAGIC or magic == 0


def memory_install() -> Bool:
    """Allocate the tracker once per process. Safe to repeat and to race."""
    var ctrl = _memory_ctrl_addr_or_create()
    if ctrl == 0:
        return False
    var base_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)
    var existing = Int(base_ptr[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
    if existing != 0:
        var magic = Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=existing + W_MAGIC)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
        return magic == MEMORY_MAGIC or magic == 0
    var blk = external_call["aligned_alloc", Int](Int(64), Int(MEMORY_BYTES))
    if blk == 0:
        return False
    var p = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=blk)
    for i in range(MEMORY_BYTES // 8):
        p[unsafe_offset=i] = 0
    Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=blk + W_MAGIC)[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int64(MEMORY_MAGIC))
    var expected = Int64(0)
    if not base_ptr[unsafe_offset=0].compare_exchange[
        failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
    ](expected, Int64(blk)):
        external_call["free", NoneType](blk)
    return memory_ready()


def memory_shutdown():
    """Free the process memory tracker early (idempotent with __cxa_atexit)."""
    var ctrl = _memory_ctrl_addr()
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
def _raise_peak_at(base_addr: Int, offset: Int, value: Int):
    """peak = max(peak, value), from any thread."""
    var w = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base_addr + offset)
    while True:
        var peak = w[unsafe_offset=0].load()
        if Int64(value) <= peak:
            return
        if w[unsafe_offset=0].compare_exchange(peak, Int64(value)):
            return


@always_inline
def raise_peak(offset: Int, value: Int):
    """peak = max(peak, value), from any thread."""
    var base_addr = memory_base()
    if base_addr != 0:
        _raise_peak_at(base_addr, offset, value)


def mem_add(tag: Int, delta: Int, count_delta: Int = 0):
    """Move a tag by `delta` bytes and `count_delta` live allocations. Any thread."""
    if tag == MEM_DOMAIN_VRAM or tag == MEM_DOMAIN_GTT:
        mem_add_domain(tag == MEM_DOMAIN_VRAM, delta)
        return
    if tag < 0 or tag >= MEM_TAGS:
        return
    var base_addr = memory_base()
    if base_addr == 0:
        return
    var base = W_TAGS + tag * TAG_STRIDE
    if delta != 0:
        var live = Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base_addr + base + T_LIVE)[unsafe_offset=0].fetch_add(Int64(delta))) + delta
        if delta > 0:
            _raise_peak_at(base_addr, base + T_PEAK, live)
        var total = W_GPU_LIVE if mem_tag_gpu(tag) else W_CPU_LIVE
        var sum = Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base_addr + total)[unsafe_offset=0].fetch_add(Int64(delta))) + delta
        if delta > 0:
            _raise_peak_at(base_addr, total + 8, sum)
    if count_delta != 0:
        _ = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base_addr + base + T_COUNT)[unsafe_offset=0].fetch_add(Int64(count_delta))
        if count_delta > 0:
            _ = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base_addr + base + T_ALLOCS)[unsafe_offset=0].fetch_add(Int64(count_delta))


def mem_add_domain(vram: Bool, delta: Int):
    """GPU bytes by where they live: VRAM, or GTT (system memory the GPU maps)."""
    var base_addr = memory_base()
    if base_addr == 0:
        return
    var live = W_VRAM_LIVE if vram else W_GTT_LIVE
    var now = Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base_addr + live)[unsafe_offset=0].fetch_add(Int64(delta))) + delta
    if delta > 0:
        _raise_peak_at(base_addr, live + 8, now)


def mem_get(tag: Int, field: Int) -> Int:
    if tag < 0 or tag >= MEM_TAGS or not memory_ready():
        return 0
    return Int(tag_word(tag, field)[unsafe_offset=0].load())


def mem_live(tag: Int) -> Int:
    return mem_get(tag, T_LIVE)


def mem_peak(tag: Int) -> Int:
    return mem_get(tag, T_PEAK)


def mem_count(tag: Int) -> Int:
    return mem_get(tag, T_COUNT)


def mem_total(offset: Int) -> Int:
    """W_CPU_LIVE, W_CPU_PEAK, W_GPU_LIVE, ... or 0 untracked."""
    if not memory_ready():
        return 0
    return Int(mem_word(offset)[unsafe_offset=0].load())


def mem_reset_peaks():
    """Every peak back to what is live now: the start of a measured stretch."""
    if not memory_ready():
        return
    for tag in range(MEM_TAGS):
        tag_word(tag, T_PEAK)[unsafe_offset=0].store(tag_word(tag, T_LIVE)[unsafe_offset=0].load())
    for offset in [W_CPU_LIVE, W_GPU_LIVE, W_VRAM_LIVE, W_GTT_LIVE]:
        mem_word(offset + 8)[unsafe_offset=0].store(mem_word(offset)[unsafe_offset=0].load())


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def mem_set_budget(tag: Int, bytes: Int):
    if tag >= 0 and tag < MEM_TAGS and memory_install():
        tag_word(tag, T_BUDGET)[unsafe_offset=0].store(Int64(max(0, bytes)))


def mem_budget(tag: Int) -> Int:
    return mem_get(tag, T_BUDGET)


def mem_over_budget(tag: Int) -> Bool:
    var budget = mem_budget(tag)
    return budget > 0 and mem_live(tag) > budget


def memory_check_budgets() -> List[Int]:
    """Tags that went over their budget since the last check (each crossing is
    reported once, until the tag is back under). Main thread, once per frame."""
    var crossed = List[Int]()
    if not memory_ready():
        return crossed^
    for tag in range(MEM_TAGS):
        var over = mem_over_budget(tag)
        var was = tag_word(tag, T_OVER)[unsafe_offset=0].load() != 0
        if over and not was:
            tag_word(tag, T_OVER)[unsafe_offset=0].store(1)
            _ = tag_word(tag, T_BREACHES)[unsafe_offset=0].fetch_add(1)
            _ = mem_word(W_BREACHES)[unsafe_offset=0].fetch_add(1)
            crossed.append(tag)
        elif not over and was:
            tag_word(tag, T_OVER)[unsafe_offset=0].store(0)
    return crossed^


def memory_breaches() -> Int:
    return mem_total(W_BREACHES)


def describe_crossing(tag: Int) -> String:
    return ("memory tag " + mem_tag_key(tag) + " is over its budget: " + mib_text(mem_live(tag))
        + " of " + mib_text(mem_budget(tag)) + " MiB (mem.budget." + mem_tag_key(tag) + ")")


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------

struct MemCharge(Movable):
    """One owner's bytes under one tag, given back when the charge dies."""
    var tag: Int    # -1: untracked
    var bytes: Int
    var count: Int  # live allocations this charge stands for

    def __init__(out self):
        self.tag = -1
        self.bytes = 0
        self.count = 0

    def __init__(out self, tag: Int, bytes: Int = 0, count: Int = 1):
        if tag >= 0 and tag <= MEM_DOMAIN_GTT and memory_install():
            self.tag = tag
            self.bytes = bytes
            self.count = count
            mem_add(tag, bytes, count)
        else:
            self.tag = -1
            self.bytes = 0
            self.count = 0

    def __init__(out self, *, deinit move: Self):
        self.tag = move.tag
        self.bytes = move.bytes
        self.count = move.count

    def release(mut self):
        """Idempotently release this charge early."""
        if self.tag >= 0:
            mem_add(self.tag, -self.bytes, -self.count)
            self.tag = -1
            self.bytes = 0
            self.count = 0

    def __deinit__(deinit self):
        if self.tag >= 0:
            mem_add(self.tag, -self.bytes, -self.count)

    @always_inline
    def set(mut self, bytes: Int):
        if self.tag >= 0 and bytes != self.bytes:
            mem_add(self.tag, bytes - self.bytes)
            self.bytes = bytes

    def add(mut self, bytes: Int, count: Int = 0):
        if self.tag >= 0:
            mem_add(self.tag, bytes, count)
            self.bytes += bytes
            self.count += count


# ---------------------------------------------------------------------------
# Arenas
# ---------------------------------------------------------------------------

struct Arena(Movable):
    """A bump allocator over one block, charged at its capacity. Allocation is
    for one thread (the owner's); `reset` frees everything at once. A request
    that does not fit returns 0 and counts as an overflow: the caller falls back
    to the heap, and the overlay shows the arena is too small."""
    var tag: Int
    var base: Int
    var capacity: Int
    var used: Int
    var peak: Int
    var allocations: Int  # since the last reset
    var overflows: Int
    var resets: Int
    var charge: MemCharge

    def __init__(out self, tag: Int, capacity: Int):
        self.tag = tag
        self.capacity = (max(0, capacity) + 63) & ~63
        self.base = external_call["aligned_alloc", Int](64, self.capacity) if self.capacity > 0 else 0
        if self.base == 0:
            self.capacity = 0
        self.used = 0
        self.peak = 0
        self.allocations = 0
        self.overflows = 0
        self.resets = 0
        self.charge = MemCharge(tag, self.capacity)

    def __init__(out self, *, deinit move: Self):
        self.tag = move.tag
        self.base = move.base
        self.capacity = move.capacity
        self.used = move.used
        self.peak = move.peak
        self.allocations = move.allocations
        self.overflows = move.overflows
        self.resets = move.resets
        self.charge = move.charge^

    def release(mut self):
        """Idempotently free the arena buffer and release its MemCharge."""
        if self.base != 0:
            external_call["free", NoneType](self.base)
            self.base = 0
        self.capacity = 0
        self.used = 0
        self.charge.release()

    def __deinit__(deinit self):
        if self.base != 0:
            external_call["free", NoneType](self.base)

    @always_inline
    def alloc(mut self, bytes: Int, align: Int = 16) -> Int:
        """The address of `bytes` bytes aligned to `align` (a power of two), or 0."""
        if bytes < 0 or self.base == 0:
            self.overflows += 1
            return 0
        var a = align if align > 1 else 1
        var aligned_addr = (self.base + self.used + a - 1) & ~(a - 1)
        var start = aligned_addr - self.base
        if start + bytes > self.capacity:
            self.overflows += 1
            return 0
        self.used = start + bytes
        if self.used > self.peak:
            self.peak = self.used
        self.allocations += 1
        return aligned_addr

    def reset(mut self):
        self.used = 0
        self.allocations = 0
        self.resets += 1

    def owns(self, address: Int) -> Bool:
        return self.base != 0 and address >= self.base and address < self.base + self.capacity


comptime FRAME_ARENA_BYTES: Int = 1 << 20


struct FrameArenas(Movable):
    """One arena per frame in flight. `begin(slot)` is called once the frame
    that last used the slot is finished on the GPU (GpuRaster.begin has waited
    for its fence), or at once on the CPU renderer; it resets that arena and
    makes it the one `alloc` uses until the next frame."""
    var arenas: List[Arena]
    var current: Int
    var frames: Int
    var frame_peak: Int  # most bytes one frame used

    def __init__(out self, slots: Int, capacity: Int = FRAME_ARENA_BYTES):
        self.arenas = List[Arena]()
        for _ in range(max(1, slots)):
            self.arenas.append(Arena(MEM_FRAME, capacity))
        self.current = 0
        self.frames = 0
        self.frame_peak = 0

    def __init__(out self, *, deinit move: Self):
        self.arenas = move.arenas^
        self.current = move.current
        self.frames = move.frames
        self.frame_peak = move.frame_peak

    def release(mut self):
        """Idempotently release all frame arenas and their memory charges."""
        self.arenas.clear()
        self.current = 0
        self.frames = 0
        self.frame_peak = 0

    def __deinit__(deinit self):
        self.arenas.clear()

    def begin(mut self, slot: Int):
        if len(self.arenas) == 0:
            return
        var index = slot % len(self.arenas) if slot >= 0 else 0
        var used = self.arenas[self.current].used
        if used > self.frame_peak:
            self.frame_peak = used
        self.current = index
        self.arenas[index].reset()
        self.frames += 1

    def next_slot(self) -> Int:
        if len(self.arenas) == 0:
            return 0
        return (self.current + 1) % len(self.arenas)

    @always_inline
    def alloc(mut self, bytes: Int, align: Int = 16) -> Int:
        if len(self.arenas) == 0:
            return 0
        return self.arenas[self.current].alloc(bytes, align)

    def used(self) -> Int:
        if len(self.arenas) == 0:
            return 0
        return self.arenas[self.current].used

    def peak(self) -> Int:
        if len(self.arenas) == 0:
            return self.frame_peak
        return max(self.frame_peak, self.arenas[self.current].used)

    def capacity(self) -> Int:
        var total = 0
        for i in range(len(self.arenas)):
            total += self.arenas[i].capacity
        return total

    def overflows(self) -> Int:
        var total = 0
        for i in range(len(self.arenas)):
            total += self.arenas[i].overflows
        return total


# ---------------------------------------------------------------------------
# The report: tags, pools, arenas, as a table and as JSON
# ---------------------------------------------------------------------------

@fieldwise_init
struct PoolStat(Copyable, Movable):
    """A fixed pool as its owner sees it."""
    var name: String
    var live: Int
    var peak: Int
    var capacity: Int
    var item_bytes: Int


@fieldwise_init
struct ArenaStat(Copyable, Movable):
    var name: String
    var used: Int
    var peak: Int
    var capacity: Int
    var overflows: Int


@fieldwise_init
struct TagStat(Copyable, Movable):
    var tag: Int
    var live: Int
    var peak: Int
    var count: Int
    var allocations: Int
    var budget: Int
    var breaches: Int

    def over(self) -> Bool:
        return self.budget > 0 and self.live > self.budget


def memory_tags() -> List[TagStat]:
    var out = List[TagStat]()
    for tag in range(MEM_TAGS):
        out.append(TagStat(tag, mem_live(tag), mem_peak(tag), mem_count(tag), mem_get(tag, T_ALLOCS), mem_budget(tag), mem_get(tag, T_BREACHES)))
    return out^


def mib_text(bytes: Int) -> String:
    """MiB with two decimals."""
    var hundredths = (bytes * 100 + MIB // 2) // MIB
    var fraction = hundredths % 100
    return String(hundredths // 100) + "." + ("0" if fraction < 10 else "") + String(fraction)


def json_name(text: String) -> String:
    return "\"" + text + "\""


def memory_json(pools: List[PoolStat], arenas: List[ArenaStat], rss_bytes: Int, peak_rss_bytes: Int, stamp: String) -> String:
    """Everything the overlay shows, for tools and for diffing two runs."""
    var out = String("{\n  \"build\": ") + json_name(stamp) + ",\n"
    out += "  \"process\": {\"rss_bytes\": " + String(rss_bytes) + ", \"peak_rss_bytes\": " + String(peak_rss_bytes) + "},\n"
    out += "  \"cpu\": {\"live_bytes\": " + String(mem_total(W_CPU_LIVE)) + ", \"peak_bytes\": " + String(mem_total(W_CPU_PEAK))
    out += ", \"untracked_bytes\": " + String(max(0, rss_bytes - mem_total(W_CPU_LIVE))) + "},\n"
    out += "  \"gpu\": {\"live_bytes\": " + String(mem_total(W_GPU_LIVE)) + ", \"peak_bytes\": " + String(mem_total(W_GPU_PEAK))
    out += ", \"vram_bytes\": " + String(mem_total(W_VRAM_LIVE)) + ", \"vram_peak_bytes\": " + String(mem_total(W_VRAM_PEAK))
    out += ", \"gtt_bytes\": " + String(mem_total(W_GTT_LIVE)) + ", \"gtt_peak_bytes\": " + String(mem_total(W_GTT_PEAK)) + "},\n"
    out += "  \"budget_breaches\": " + String(memory_breaches()) + ",\n"
    out += "  \"tags\": {\n"
    var tags = memory_tags()
    for index in range(len(tags)):
        ref t = tags[index]
        out += "    " + json_name(mem_tag_key(t.tag)) + ": {\"live_bytes\": " + String(t.live) + ", \"peak_bytes\": " + String(t.peak)
        out += ", \"count\": " + String(t.count) + ", \"allocations\": " + String(t.allocations)
        out += ", \"budget_bytes\": " + String(t.budget) + ", \"over\": " + ("true" if t.over() else "false")
        out += ", \"breaches\": " + String(t.breaches) + "}" + ("," if index + 1 < len(tags) else "") + "\n"
    out += "  },\n  \"pools\": {\n"
    for index in range(len(pools)):
        ref p = pools[index]
        out += "    " + json_name(p.name) + ": {\"live\": " + String(p.live) + ", \"peak\": " + String(p.peak)
        out += ", \"capacity\": " + String(p.capacity) + ", \"item_bytes\": " + String(p.item_bytes) + "}"
        out += ("," if index + 1 < len(pools) else "") + "\n"
    out += "  },\n  \"arenas\": {\n"
    for index in range(len(arenas)):
        ref a = arenas[index]
        out += "    " + json_name(a.name) + ": {\"used_bytes\": " + String(a.used) + ", \"peak_bytes\": " + String(a.peak)
        out += ", \"capacity_bytes\": " + String(a.capacity) + ", \"overflows\": " + String(a.overflows) + "}"
        out += ("," if index + 1 < len(arenas) else "") + "\n"
    out += "  }\n}\n"
    return out^


def memory_summary(most: Int) -> List[String]:
    """Totals, then the `most` largest tags, largest first: the console's `mem`."""
    var lines = List[String]()
    if not memory_ready():
        lines.append("memory is not tracked in this process")
        return lines^
    lines.append("cpu " + mib_text(mem_total(W_CPU_LIVE)) + " MiB (peak " + mib_text(mem_total(W_CPU_PEAK))
        + ")  gpu " + mib_text(mem_total(W_GPU_LIVE)) + " MiB (vram " + mib_text(mem_total(W_VRAM_LIVE))
        + ", gtt " + mib_text(mem_total(W_GTT_LIVE)) + ")  over budget " + String(memory_breaches()))
    var tags = memory_tags()
    var shown = List[Bool](length=len(tags), fill=False)
    for _ in range(min(most, len(tags))):
        var best = -1
        for index in range(len(tags)):
            if not shown[index] and (best < 0 or tags[index].live > tags[best].live):
                best = index
        if best < 0 or tags[best].live == 0:
            break
        shown[best] = True
        ref t = tags[best]
        lines.append(mem_tag_key(t.tag) + " " + mib_text(t.live) + " MiB, peak " + mib_text(t.peak) + ", "
            + String(t.count) + " live" + (", budget " + mib_text(t.budget) if t.budget > 0 else String("")) + (" OVER" if t.over() else String("")))
    return lines^
