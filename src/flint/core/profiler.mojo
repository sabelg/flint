"""A per-frame timeline: scoped CPU timers, GPU timestamps, a trace file.

Any thread can time a stretch of code through a `Profiler`, which is a small
copyable handle:

    with profiler.zone(ZONE_SYNC):
        renderer.sync(game, 2)

A handle with no block behind it (the default) reads no clock and records
nothing, so an unprofiled run pays one branch per zone. `SD_PROFILE=1` makes
`NativeRuntime` start a `ProfileTimeline`, which owns the block.

A zone is written once, when it ends, into one shared ring of fixed cells:
claim a cell with a compare-and-swap, fill it, then publish it by storing its
zone word. Nothing allocates, so a zone can sit on a path used a few thousand
times a frame, though not on one used per vertex. The ring is sized to hold a
whole frame; past that, events are counted as dropped and never overwrite.

`ProfileTimeline` runs on the main thread. `end_frame` takes every published
cell in order, stopping at the first one still being written, and adds each to
the per-zone history, the trace file, and the overlay's held frame. Times are
raw `rdtsc` ticks until then. The TSC on every CPU this runs on is invariant
and shared between cores, so ticks from different threads compare directly.

GPU spans arrive the same way, a few frames after the frame that submitted
them, with their times already mapped onto the CPU's clock by the backend. They
carry that frame's number, so a late span lands in the right place.

The file is the Trace Event format (Perfetto and chrome://tracing open it):
one `X` event per zone on a thread per lane, and `args.frame` on each frame and
each GPU pass. Lanes are named by `M` metadata events.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic
from std.time import monotonic

comptime ZONE_NONE: Int = -1
# Whole frame, on the main thread.
comptime ZONE_FRAME: Int = 0
comptime ZONE_PACE: Int = 1
comptime ZONE_INPUT: Int = 2
comptime ZONE_SIM: Int = 3
comptime ZONE_SYNC: Int = 4
comptime ZONE_UPLOAD: Int = 5
comptime ZONE_SCENE: Int = 6
comptime ZONE_POST: Int = 7
comptime ZONE_HUD: Int = 8
comptime ZONE_OVERLAY: Int = 9
comptime ZONE_PRESENT: Int = 10
comptime ZONE_GPU_SETUP: Int = 11
comptime ZONE_HUD_LAYER: Int = 12
comptime ZONE_GPU_BIN: Int = 13
comptime ZONE_GPU_SUBMIT: Int = 14
comptime ZONE_GPU_WAIT: Int = 15
comptime ZONE_READBACK: Int = 16
comptime ZONE_AUDIO_JOIN: Int = 17
comptime ZONE_PROFILER: Int = 18
# Inside a tick.
comptime ZONE_TICK: Int = 19
comptime ZONE_PLAYER: Int = 20
comptime ZONE_ENEMIES: Int = 21
comptime ZONE_BODIES: Int = 22
comptime ZONE_PROJECTILES: Int = 23
# Jobs, recorded per thread by the pool.
comptime ZONE_MESH: Int = 24
comptime ZONE_SKIN: Int = 25
comptime ZONE_TRIANGLES: Int = 26
comptime ZONE_AUDIO: Int = 27
# On the GPU's own lane. A frame is one span around its passes.
comptime ZONE_GPU_FRAME: Int = 28
comptime ZONE_GPU_RASTER: Int = 29
comptime ZONE_GPU_BLOOM: Int = 30
comptime ZONE_GPU_SMOOTH: Int = 31
comptime ZONE_GPU_BACKDROP: Int = 32
comptime ZONE_GPU_COMPOSE: Int = 33
# Back on the main thread, numbered last so no other zone moves.
comptime ZONE_DEBUG_DRAW: Int = 34
comptime ZONE_DETACH: Int = 35  # the detached-piece flood fill jobs (package 2.1)
comptime ZONE_BROADPHASE: Int = 36  # the body pair jobs (package 2.11)
comptime ZONE_SOLVER: Int = 37  # the solver tasks, one per worker (package 2.2)
comptime ZONE_GPU_PARTICLES: Int = 38  # the GPU particle passes, on the GPU's lane (package 2.7)
comptime ZONE_GROUND: Int = 39  # the granular settle jobs and the support pass (package 2.8)
comptime ZONE_FIELDS: Int = 40  # the field step, reactions and element states (package 2.9)
comptime ZONE_COUNT: Int = 41

comptime HEAD_WORD: Int = 0
comptime TAIL_WORD: Int = 8
comptime DROPPED_WORD: Int = 16
comptime GPU_FRAMES_WORD: Int = 24
comptime GPU_MISSING_WORD: Int = 25
comptime FRAME_WORD: Int = 32
comptime TICK_RATE_WORD: Int = 33
comptime EVENTS_WORD: Int = 40
comptime EVENT_WORDS: Int = 4
comptime EVENT_START: Int = 0
comptime EVENT_END: Int = 1
comptime EVENT_THREAD: Int = 2
comptime EVENT_ZONE: Int = 3  # published last, as zone + 1; zero means not ready
comptime EVENT_CAPACITY: Int = 1 << 16
comptime BLOCK_WORDS: Int = EVENTS_WORD + EVENT_CAPACITY * EVENT_WORDS

# A pthread_t is a user-space address, so it never has this bit.
comptime GPU_THREAD_TAG: UInt64 = 1 << 62
comptime GPU_FRAME_MASK: UInt64 = (1 << 62) - 1

comptime MAIN_LANE: Int = 0
comptime GPU_LANE: Int = 1
comptime FIRST_WORKER_LANE: Int = 2
comptime MAX_LANES: Int = 40

comptime HISTORY_FRAMES: Int = 4096
comptime RECENT_FRAMES: Int = 60
comptime HOLD_FRAMES: Int = 30  # the overlay holds the slowest frame of this many
comptime MAX_DISPLAY_EVENTS: Int = 4096
comptime MAX_DEPTH: Int = 8
comptime START_RING_FRAMES: Int = 64
comptime GPU_SPANS_PER_FRAME: Int = 8

comptime FILE_BUFFER_BYTES: Int = 1 << 20
comptime EVENT_TEXT_BYTES: Int = 256
comptime MAX_FILE_BYTES: Int = 1 << 31
comptime CALIBRATION_MICROSECONDS: UInt32 = 20000
comptime BUDGET_MS: Float64 = 1000.0 / 60.0


def zone_name(zone: Int) -> String:
    var names = [
        "frame", "pace", "input", "sim", "sync", "upload", "scene", "post", "hud", "overlay",
        "present", "gpu_setup", "hud_layer", "gpu_bin", "gpu_submit", "gpu_wait", "readback", "audio_join", "profiler",
        "tick", "player", "enemies", "bodies", "projectiles",
        "mesh", "skin", "triangles", "audio",
        "gpu_frame", "gpu_raster", "gpu_bloom", "gpu_smooth", "gpu_backdrop", "gpu_compose",
        "debug_draw", "detach", "broadphase", "solver", "gpu_particles", "ground", "fields",
    ]
    if zone < 0 or zone >= len(names):
        return "unknown"
    return String(names[zone])


def zone_color(zone: Int) -> Tuple[UInt8, UInt8, UInt8]:
    """Waiting is grey, so what is left on the timeline is work."""
    if zone == ZONE_PACE or zone == ZONE_GPU_WAIT or zone == ZONE_AUDIO_JOIN or zone == ZONE_SIM:
        return (UInt8(92), UInt8(98), UInt8(108))
    if zone == ZONE_DEBUG_DRAW:
        return (UInt8(120), UInt8(190), UInt8(246))
    if zone == ZONE_DETACH or zone == ZONE_BROADPHASE or zone == ZONE_SOLVER or zone == ZONE_GROUND or zone == ZONE_FIELDS:
        return (UInt8(150), UInt8(226), UInt8(150))
    if zone >= ZONE_GPU_FRAME:
        return (UInt8(238), UInt8(82), UInt8(112)) if zone % 2 == 0 else (UInt8(250), UInt8(150), UInt8(120))
    if zone == ZONE_AUDIO:
        return (UInt8(176), UInt8(120), UInt8(236))
    if zone >= ZONE_MESH:
        return (UInt8(246), UInt8(172), UInt8(62)) if zone % 2 == 0 else (UInt8(236), UInt8(208), UInt8(92))
    if zone >= ZONE_TICK:
        return (UInt8(92), UInt8(204), UInt8(120)) if zone % 2 == 0 else (UInt8(150), UInt8(226), UInt8(150))
    return (UInt8(80), UInt8(150), UInt8(236)) if zone % 2 == 0 else (UInt8(120), UInt8(190), UInt8(246))


@always_inline
def profile_ticks() -> UInt64:
    comptime if CompilationTarget.is_x86():
        return llvm_intrinsic["llvm.x86.rdtsc", UInt64]()
    else:
        return llvm_intrinsic["llvm.readcyclecounter", UInt64]()


struct ProfileScope(Movable):
    """Ends its zone when the `with` block is left, however it is left."""
    var address: Int
    var zone: Int
    var started: UInt64

    @always_inline
    def __init__(out self, address: Int, zone: Int):
        self.address = address
        self.zone = zone
        self.started = profile_ticks() if address != 0 else 0

    @always_inline
    def __enter__(self):
        pass

    @always_inline
    def __exit__(self):
        if self.address != 0:
            Profiler(self.address).record(self.zone, self.started, profile_ticks())


struct Profiler(Copyable, Movable):
    """Points at the shared block, or at nothing when profiling is off."""
    var address: Int

    @always_inline
    def __init__(out self, address: Int = 0):
        self.address = address

    @always_inline
    def enabled(self) -> Bool:
        return self.address != 0

    @always_inline
    def word(self, index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=self.address + index * 8)

    @always_inline
    def cell(self, position: Int, field: Int) -> Pointer[UInt64, origin=MutAnyOrigin]:
        return Pointer[UInt64, origin=MutAnyOrigin](unsafe_from_address=self.address + (EVENTS_WORD + (position & (EVENT_CAPACITY - 1)) * EVENT_WORDS + field) * 8)

    @always_inline
    def zone(self, zone: Int) -> ProfileScope:
        return ProfileScope(self.address, zone)

    @always_inline
    def now(self) -> UInt64:
        """A timestamp for `record`, or 0 when off, so that nothing reads the clock."""
        return profile_ticks() if self.address != 0 else 0

    @always_inline
    def record(self, zone: Int, started: UInt64, ended: UInt64):
        if self.address != 0:
            self.publish(zone, started, ended, UInt64(external_call["pthread_self", Int]()))

    def publish(self, zone: Int, started: UInt64, ended: UInt64, thread: UInt64):
        var head = self.word(HEAD_WORD)
        var position = head[unsafe_offset=0].load()
        while True:
            if Int(position) - Int(self.word(TAIL_WORD)[unsafe_offset=0].load()) >= EVENT_CAPACITY:
                _ = self.word(DROPPED_WORD)[unsafe_offset=0].fetch_add(1)
                return
            if head[unsafe_offset=0].compare_exchange(position, position + 1):
                break
        self.cell(Int(position), EVENT_START)[unsafe_offset=0] = started
        self.cell(Int(position), EVENT_END)[unsafe_offset=0] = ended
        self.cell(Int(position), EVENT_THREAD)[unsafe_offset=0] = thread
        Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(self.cell(Int(position), EVENT_ZONE)))[unsafe_offset=0].store(Int64(zone + 1))

    def record_gpu(self, zone: Int, frame: Int, started: UInt64, ended: UInt64):
        """A GPU span for frame number `frame`, in CPU ticks."""
        if self.address != 0:
            self.publish(zone, started, ended, GPU_THREAD_TAG | UInt64(frame))

    def ticks_per_second(self) -> Float64:
        return Float64(Int(self.word(TICK_RATE_WORD)[unsafe_offset=0].load())) if self.address != 0 else 0

    def frame(self) -> Int:
        """The number of the frame in progress; a backend tags its GPU work with it."""
        return Int(self.word(FRAME_WORD)[unsafe_offset=0].load()) if self.address != 0 else 0

    def note_gpu_frame(self, complete: Bool):
        """Every frame the backend timed reports here once, so a test can count them."""
        if self.address != 0:
            _ = self.word(GPU_FRAMES_WORD if complete else GPU_MISSING_WORD)[unsafe_offset=0].fetch_add(1)

    def dropped(self) -> Int:
        return Int(self.word(DROPPED_WORD)[unsafe_offset=0].load()) if self.address != 0 else 0

    def gpu_frames(self) -> Int:
        return Int(self.word(GPU_FRAMES_WORD)[unsafe_offset=0].load()) if self.address != 0 else 0

    def gpu_missing(self) -> Int:
        return Int(self.word(GPU_MISSING_WORD)[unsafe_offset=0].load()) if self.address != 0 else 0


@fieldwise_init
struct ProfileEvent(Copyable, Movable):
    """One zone of the frame being taken apart."""
    var zone: Int
    var lane: Int
    var started: UInt64
    var ended: UInt64


@fieldwise_init
struct DisplayEvent(Copyable, Movable):
    """One zone of the frame on screen: milliseconds from that frame's start."""
    var zone: Int
    var lane: Int
    var depth: Int
    var start_ms: Float32
    var end_ms: Float32


@fieldwise_init
struct ZoneRow(Copyable, Movable):
    """A zone's cost per frame over the last second, for the overlay's table."""
    var zone: Int
    var average_ms: Float32
    var worst_ms: Float32


def report_ms(value: Float32) -> String:
    """Milliseconds to the microsecond: enough, and always plain digits."""
    var scaled = Int(value * 1000.0 + 0.5)
    var fraction = String(scaled % 1000)
    return String(scaled // 1000) + "." + "0" * (3 - fraction.byte_length()) + fraction


def percentile_of(mut values: List[Float32], fraction: Float64) -> Float32:
    """`values` must be sorted."""
    if len(values) == 0:
        return 0
    return values[min(len(values) - 1, Int(fraction * Float64(len(values) - 1)))]


struct ProfileTimeline(Movable):
    """The main thread's side: turns cells into history, a file and a held frame."""
    var handle: Profiler
    var raw_block: Int
    var active: Bool
    var overlay_visible: Bool
    var frame: Int
    var frame_started: UInt64
    var origin: UInt64
    var nanoseconds_per_tick: Float64
    var events_total: Int
    # Lanes: a thread's lane is its index here; the first two are fixed.
    var lane_threads: List[UInt64]
    var lane_cursor: Int
    # Per-zone milliseconds per frame, a ring of HISTORY_FRAMES for each zone.
    var zone_ticks: List[UInt64]
    var history: List[Float32]
    var frame_events: List[ProfileEvent]
    # The held frame on screen, and the slowest of the frames since it was chosen.
    var display: List[DisplayEvent]
    var candidate: List[DisplayEvent]
    var candidate_ms: Float32
    var candidate_frame: Int
    var display_ms: Float32
    var display_frame: Int
    var display_rows: List[ZoneRow]
    var recent_frames_ms: List[Float32]
    var display_history: List[Float32]  # newest first, as of the last change on screen
    var display_events_in_frame: Int
    var frame_starts: List[UInt64]
    var gpu_spans: List[DisplayEvent]
    var gpu_span_frames: List[Int]
    var gpu_span_counts: List[Int]
    # The file.
    var path: String
    var fd: Int
    var buffer: Int
    var buffer_used: Int
    var file_bytes: Int
    var file_full: Bool
    var first_event: Bool
    var zone_text: List[String]

    def __init__(out self):
        self.handle = Profiler()
        self.raw_block = 0
        self.active = False
        self.overlay_visible = False
        self.frame = 0
        self.frame_started = 0
        self.origin = 0
        self.nanoseconds_per_tick = 0
        self.events_total = 0
        self.lane_threads = List[UInt64]()
        self.lane_cursor = 0
        self.zone_ticks = List[UInt64]()
        self.history = List[Float32]()
        self.frame_events = List[ProfileEvent]()
        self.display = List[DisplayEvent]()
        self.candidate = List[DisplayEvent]()
        self.candidate_ms = 0
        self.candidate_frame = -1
        self.display_ms = 0
        self.display_frame = -1
        self.display_rows = List[ZoneRow]()
        self.recent_frames_ms = List[Float32]()
        self.display_history = List[Float32]()
        self.display_events_in_frame = 0
        self.frame_starts = List[UInt64]()
        self.gpu_spans = List[DisplayEvent]()
        self.gpu_span_frames = List[Int]()
        self.gpu_span_counts = List[Int]()
        self.path = String()
        self.fd = -1
        self.buffer = 0
        self.buffer_used = 0
        self.file_bytes = 0
        self.file_full = False
        self.first_event = True
        self.zone_text = List[String]()

    def __deinit__(deinit self):
        if self.fd >= 0:
            _ = external_call["close", Int32](Int32(self.fd))
        if self.buffer != 0:
            external_call["free", NoneType](self.buffer)
        if self.raw_block != 0:
            external_call["free", NoneType](self.raw_block)

    def start(mut self, path: String, overlay: Bool = True) -> Bool:
        """Allocate everything, measure the clock, open the file.

        An empty path, or one that cannot be opened, still profiles: the
        overlay and the summary work with no file.
        """
        if self.active:
            return True
        if self.raw_block == 0:
            var memory = external_call["calloc", Int](Int(1), BLOCK_WORDS * 8 + 64)
            if memory == 0:
                return False
            self.raw_block = memory
            self.handle = Profiler((memory + 63) & ~63)
        self.overlay_visible = overlay
        self.lane_threads = List[UInt64](capacity=MAX_LANES)
        self.lane_threads.append(UInt64(external_call["pthread_self", Int]()))
        self.lane_threads.append(GPU_THREAD_TAG)
        self.lane_cursor = MAIN_LANE
        self.zone_ticks = List[UInt64](length=ZONE_COUNT, fill=0)
        self.history = List[Float32](length=ZONE_COUNT * HISTORY_FRAMES, fill=0)
        self.frame_events = List[ProfileEvent](capacity=EVENT_CAPACITY)
        self.display = List[DisplayEvent](capacity=MAX_DISPLAY_EVENTS)
        self.candidate = List[DisplayEvent](capacity=MAX_DISPLAY_EVENTS)
        self.display_rows = List[ZoneRow](capacity=ZONE_COUNT)
        self.recent_frames_ms = List[Float32](length=RECENT_FRAMES * 2, fill=0)
        self.display_history = List[Float32](length=RECENT_FRAMES * 2, fill=0)
        self.frame_starts = List[UInt64](length=START_RING_FRAMES, fill=0)
        self.gpu_spans = List[DisplayEvent](length=START_RING_FRAMES * GPU_SPANS_PER_FRAME, fill=DisplayEvent(0, 0, 0, 0, 0))
        self.gpu_span_frames = List[Int](length=START_RING_FRAMES, fill=-1)
        self.gpu_span_counts = List[Int](length=START_RING_FRAMES, fill=0)
        self.measure_clock()
        self.origin = profile_ticks()
        self.active = True
        self.open_file(path)
        return True

    def measure_clock(mut self):
        var wall = monotonic()
        var ticks = profile_ticks()
        # Non-blocking calibration over ~50 us window without synchronous usleep(20000)
        while monotonic() - wall < 50000:
            pass
        var spent_wall = monotonic() - wall
        var spent_ticks = profile_ticks() - ticks
        if spent_ticks > 100:
            self.nanoseconds_per_tick = Float64(spent_wall) / Float64(spent_ticks)
        else:
            self.nanoseconds_per_tick = 0.5
        self.handle.word(TICK_RATE_WORD)[unsafe_offset=0].store(Int64(Int(self.ticks_per_second())))

    def ticks_per_second(self) -> Float64:
        return 1000000000.0 / self.nanoseconds_per_tick

    def to_ms(self, ticks: UInt64) -> Float32:
        return Float32(Float64(ticks) * self.nanoseconds_per_tick * 0.000001)

    # ---- frames ----

    def begin_frame(mut self):
        if not self.active:
            return
        self.frame_started = profile_ticks()
        self.frame_starts[self.frame % START_RING_FRAMES] = self.frame_started
        self.handle.word(FRAME_WORD)[unsafe_offset=0].store(Int64(self.frame))

    def end_frame(mut self):
        """Close the frame. Every job of the frame must be finished."""
        if not self.active:
            return
        var ended = profile_ticks()
        # Empty the ring first: a frame that overflowed it still gets its frame event.
        self.take_events(self.frame)
        self.handle.record(ZONE_FRAME, self.frame_started, ended)
        self.take_events(self.frame)
        var total_ms = self.to_ms(ended - self.frame_started)
        for zone in range(ZONE_COUNT):
            self.history[zone * HISTORY_FRAMES + self.frame % HISTORY_FRAMES] = self.to_ms(self.zone_ticks[zone])
            self.zone_ticks[zone] = 0
        self.recent_frames_ms[self.frame % len(self.recent_frames_ms)] = total_ms
        if self.overlay_visible and (self.candidate_frame < 0 or total_ms > self.candidate_ms):
            self.hold_frame(self.frame, total_ms)
        if self.overlay_visible and self.frame % HOLD_FRAMES == HOLD_FRAMES - 1:
            self.show_held_frame()
        self.frame_events.clear()
        self.frame += 1
        self.handle.record(ZONE_PROFILER, ended, profile_ticks())

    def take_events(mut self, frame: Int):
        """Take every published cell, in order, up to the first unfinished one."""
        var profiler = self.handle.copy()
        var tail = Int(profiler.word(TAIL_WORD)[unsafe_offset=0].load())
        var head = Int(profiler.word(HEAD_WORD)[unsafe_offset=0].load())
        while tail < head:
            var published = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(profiler.cell(tail, EVENT_ZONE)))
            var zone = Int(published[unsafe_offset=0].load()) - 1
            if zone < 0:
                break
            var started = profiler.cell(tail, EVENT_START)[unsafe_offset=0]
            var ended = profiler.cell(tail, EVENT_END)[unsafe_offset=0]
            var thread = profiler.cell(tail, EVENT_THREAD)[unsafe_offset=0]
            published[unsafe_offset=0].store(0)
            tail += 1
            self.take_event(zone, started, ended, thread, frame)
        profiler.word(TAIL_WORD)[unsafe_offset=0].store(Int64(tail))

    def take_event(mut self, zone: Int, started: UInt64, ended: UInt64, thread: UInt64, frame: Int):
        self.events_total += 1
        var event_frame = frame
        var lane = GPU_LANE
        if (thread & GPU_THREAD_TAG) != 0:
            event_frame = Int(thread & GPU_FRAME_MASK)
            self.keep_gpu_span(zone, event_frame, started, ended)
        else:
            lane = self.lane_of(thread)
            self.frame_events.append(ProfileEvent(zone, lane, started, ended))
        self.zone_ticks[zone] += ended - started
        if self.fd >= 0:
            self.write_event(zone, lane, started, ended, event_frame)

    def lane_of(mut self, thread: UInt64) -> Int:
        if self.lane_threads[self.lane_cursor] == thread:
            return self.lane_cursor
        for lane in range(len(self.lane_threads)):
            if self.lane_threads[lane] == thread:
                self.lane_cursor = lane
                return lane
        if len(self.lane_threads) >= MAX_LANES:
            return len(self.lane_threads) - 1  # more threads than any pool here has
        self.lane_threads.append(thread)
        self.lane_cursor = len(self.lane_threads) - 1
        self.write_lane_name(self.lane_cursor)
        return self.lane_cursor

    def keep_gpu_span(mut self, zone: Int, frame: Int, started: UInt64, ended: UInt64):
        """Spans outlive their frame by a few frames, so the held frame can find its own."""
        var row = frame % START_RING_FRAMES
        if self.gpu_span_frames[row] != frame:
            self.gpu_span_frames[row] = frame
            self.gpu_span_counts[row] = 0
        var count = self.gpu_span_counts[row]
        var opened = self.frame_starts[row]
        if count >= GPU_SPANS_PER_FRAME or started < opened:
            return
        self.gpu_spans[row * GPU_SPANS_PER_FRAME + count] = DisplayEvent(zone, GPU_LANE, 1 if zone != ZONE_GPU_FRAME else 0, self.to_ms(started - opened), self.to_ms(ended - opened))
        self.gpu_span_counts[row] = count + 1

    def hold_frame(mut self, frame: Int, total_ms: Float32):
        """Copy this frame for the overlay. Events are in the order they ended, so
        walking them backwards meets each zone before the ones nested in it."""
        self.candidate_frame = frame
        self.candidate_ms = total_ms
        self.candidate.clear()
        self.display_events_in_frame = len(self.frame_events)
        var stacks = List[UInt64](length=MAX_LANES * MAX_DEPTH, fill=0)
        var depths = List[Int](length=MAX_LANES, fill=0)
        var index = len(self.frame_events)
        while index > 0 and len(self.candidate) < MAX_DISPLAY_EVENTS:
            index -= 1
            ref event = self.frame_events[index]
            if event.ended < self.frame_started or event.zone == ZONE_FRAME:
                continue
            var depth = depths[event.lane]
            # Going backwards in time, a zone that starts after this one ends is
            # a sibling or a sibling's child, not a parent.
            while depth > 0 and stacks[event.lane * MAX_DEPTH + depth - 1] >= event.ended:
                depth -= 1
            var start_ms = self.to_ms(max(event.started, self.frame_started) - self.frame_started)
            self.candidate.append(DisplayEvent(event.zone, event.lane, depth, start_ms, self.to_ms(event.ended - self.frame_started)))
            if depth < MAX_DEPTH:
                stacks[event.lane * MAX_DEPTH + depth] = event.started
                depth += 1
            depths[event.lane] = depth

    def show_held_frame(mut self):
        swap(self.display, self.candidate)
        self.display_ms = self.candidate_ms
        self.display_frame = self.candidate_frame
        self.candidate.clear()
        self.candidate_frame = -1
        self.candidate_ms = 0
        self.display_rows.clear()
        for back in range(min(self.frame + 1, len(self.display_history))):
            self.display_history[back] = self.recent_frames_ms[(self.frame - back) % len(self.recent_frames_ms)]
        var frames = min(self.frame + 1, RECENT_FRAMES)
        for zone in range(1, ZONE_COUNT):
            var total = Float32(0)
            var worst = Float32(0)
            for back in range(frames):
                var value = self.history[zone * HISTORY_FRAMES + (self.frame - back) % HISTORY_FRAMES]
                total += value
                worst = max(worst, value)
            # Waiting for the display is not a cost, and it would top the table.
            if worst > 0 and zone != ZONE_PACE:
                self.display_rows.append(ZoneRow(zone, total / Float32(frames), worst))

    def held_gpu_span_count(self) -> Int:
        if self.display_frame < 0:
            return 0
        var row = self.display_frame % START_RING_FRAMES
        return self.gpu_span_counts[row] if self.gpu_span_frames[row] == self.display_frame else 0

    def held_end_ms(self) -> Float32:
        """How far the frame on screen reaches, its GPU work included."""
        var reach = self.display_ms
        for index in range(self.held_gpu_span_count()):
            reach = max(reach, self.held_gpu_span(index).end_ms)
        return reach

    def held_gpu_span(self, index: Int) -> DisplayEvent:
        return self.gpu_spans[self.display_frame % START_RING_FRAMES * GPU_SPANS_PER_FRAME + index].copy()

    def lane_count(self) -> Int:
        return len(self.lane_threads)

    def finish(mut self):
        """Take what is left, end the file. GPU spans must already be recorded."""
        if not self.active:
            return
        self.take_events(self.frame)
        self.frame_events.clear()
        self.active = False
        if self.fd >= 0:
            self.append_text("\n]\n")
            self.flush_file()
            _ = external_call["close", Int32](Int32(self.fd))
            self.fd = -1
        if self.buffer != 0:
            external_call["free", NoneType](self.buffer)
            self.buffer = 0

    def report(mut self):
        """Print key=value lines: per-frame cost of every zone that ran."""
        print("profile_frames=" + String(self.frame))
        print("profile_events=" + String(self.events_total))
        print("profile_dropped=" + String(self.handle.dropped()))
        print("profile_gpu_frames=" + String(self.handle.gpu_frames()))
        print("profile_gpu_missing=" + String(self.handle.gpu_missing()))
        print("profile_threads=" + String(len(self.lane_threads)))
        print("profile_ticks_per_second=" + String(Int(self.ticks_per_second())))
        if self.path.byte_length() > 0:
            print("profile_file=" + self.path + " bytes=" + String(self.file_bytes) + (" FULL" if self.file_full else ""))
        var frames = min(self.frame, HISTORY_FRAMES)
        for zone in range(ZONE_COUNT):
            var values = List[Float32](capacity=frames)
            var worst = Float32(0)
            for index in range(frames):
                var value = self.history[zone * HISTORY_FRAMES + index]
                values.append(value)
                worst = max(worst, value)
            if worst <= 0:
                continue
            sort(values)
            print(
                "profile_zone_" + zone_name(zone)
                + " p50=" + report_ms(percentile_of(values, 0.50))
                + " p95=" + report_ms(percentile_of(values, 0.95))
                + " p99=" + report_ms(percentile_of(values, 0.99))
                + " max=" + report_ms(worst)
            )

    # ---- the file ----

    def open_file(mut self, path: String):
        if self.fd >= 0:
            _ = external_call["close", Int32](Int32(self.fd))
            self.fd = -1
        if self.buffer != 0:
            external_call["free", NoneType](self.buffer)
            self.buffer = 0
        self.buffer_used = 0
        self.path = path
        if path.byte_length() == 0:
            return
        var owned = path
        # O_WRONLY | O_CREAT | O_TRUNC, as write_bytes.
        self.fd = Int(external_call["open", Int32](owned.as_c_string_span(), Int32(1 | 64 | 512), Int32(438)))
        if self.fd < 0:
            print("PROFILE_FILE_FAILED " + path)
            self.path = String()
            return
        self.buffer = external_call["malloc", Int](FILE_BUFFER_BYTES)
        if self.buffer == 0:
            print("PROFILE_FILE_FAILED " + path)
            _ = external_call["close", Int32](Int32(self.fd))
            self.fd = -1
            self.path = String()
            return
        self.zone_text = List[String](capacity=ZONE_COUNT)
        for zone in range(ZONE_COUNT):
            self.zone_text.append("\"name\":\"" + zone_name(zone) + "\"")
        self.append_text("[")
        for lane in range(len(self.lane_threads)):
            self.write_lane_name(lane)

    def lane_name(self, lane: Int) -> String:
        if lane == MAIN_LANE:
            return "main"
        if lane == GPU_LANE:
            return "gpu"
        return "worker " + String(lane - FIRST_WORKER_LANE)

    def write_lane_name(mut self, lane: Int):
        if self.fd < 0:
            return
        self.start_entry()
        self.append_text("{\"ph\":\"M\",\"pid\":1,\"tid\":")
        self.append_number(lane)
        self.append_text(",\"name\":\"thread_name\",\"args\":{\"name\":\"" + self.lane_name(lane) + "\"}}")

    def start_entry(mut self):
        if self.buffer_used + EVENT_TEXT_BYTES > FILE_BUFFER_BYTES:
            self.flush_file()
        if not self.first_event:
            self.append_byte(44)  # comma
        self.append_byte(10)  # newline
        self.first_event = False

    def write_event(mut self, zone: Int, lane: Int, started: UInt64, ended: UInt64, frame: Int):
        self.start_entry()
        self.append_text("{\"ph\":\"X\",\"pid\":1,\"tid\":")
        self.append_number(lane)
        self.append_byte(44)
        self.append_string(zone)
        self.append_text(",\"ts\":")
        self.append_microseconds(started - self.origin if started > self.origin else 0)
        self.append_text(",\"dur\":")
        self.append_microseconds(ended - started)
        if zone == ZONE_FRAME or lane == GPU_LANE:
            self.append_text(",\"args\":{\"frame\":")
            self.append_number(frame)
            self.append_byte(125)
        self.append_byte(125)

    @always_inline
    def append_byte(mut self, value: UInt8):
        Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=self.buffer + self.buffer_used)[unsafe_offset=0] = value
        self.buffer_used += 1

    def append_text(mut self, text: String):
        var size = text.byte_length()
        _ = external_call["memcpy", Int](self.buffer + self.buffer_used, Int(text.as_bytes().unsafe_ptr()), size)
        self.buffer_used += size

    def append_string(mut self, zone: Int):
        ref text = self.zone_text[zone]
        var size = text.byte_length()
        _ = external_call["memcpy", Int](self.buffer + self.buffer_used, Int(text.as_bytes().unsafe_ptr()), size)
        self.buffer_used += size

    def append_number(mut self, value: Int):
        var digits = 1
        var rest = value // 10
        while rest > 0:
            digits += 1
            rest //= 10
        var position = self.buffer + self.buffer_used + digits
        rest = value
        for _ in range(digits):
            position -= 1
            Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=position)[unsafe_offset=0] = UInt8(48 + rest % 10)
            rest //= 10
        self.buffer_used += digits

    def append_microseconds(mut self, ticks: UInt64):
        """Trace timestamps are microseconds; three decimals keep the nanoseconds."""
        var nanoseconds = Int(Float64(ticks) * self.nanoseconds_per_tick)
        self.append_number(nanoseconds // 1000)
        var fraction = nanoseconds % 1000
        self.append_byte(46)
        self.append_byte(UInt8(48 + fraction // 100))
        self.append_byte(UInt8(48 + fraction // 10 % 10))
        self.append_byte(UInt8(48 + fraction % 10))

    def flush_file(mut self):
        if self.fd < 0 or self.buffer_used == 0:
            return
        if self.file_bytes + self.buffer_used > MAX_FILE_BYTES:
            # Stay valid JSON: the closing bracket still fits, and the rest is dropped.
            self.file_full = True
            print("PROFILE_FILE_FULL " + self.path)
            var closing = "\n]\n"
            _ = external_call["write", Int64](Int32(self.fd), closing.as_bytes().unsafe_ptr(), Int64(3))
            _ = external_call["close", Int32](Int32(self.fd))
            self.fd = -1
            self.buffer_used = 0
            return
        var offset = 0
        while offset < self.buffer_used:
            var count = Int(external_call["write", Int64](Int32(self.fd), Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=self.buffer + offset), Int64(self.buffer_used - offset)))
            if count <= 0:
                print("PROFILE_FILE_FAILED " + self.path)
                _ = external_call["close", Int32](Int32(self.fd))
                self.fd = -1
                break
            offset += count
        self.file_bytes += offset
        self.buffer_used = 0
