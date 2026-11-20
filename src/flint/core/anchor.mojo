"""The safety anchor: one fixed-address mapping the log, the crash handler and
the GPU watchdog share.

Mojo has no global variables, and a signal handler gets no pointer of its
own, so the safety net is dynamically mapped via anonymous `mmap(0)` and
published through a lock-free `FLINT_ANCHOR_CTRL` global control block;
everything else finds it with `anchor_base()` / `anchor_ready()`, which checks
the mapping, a magic word and the process id.

Everything here is safe to call from a signal handler: raw system calls
through `syscall`, plain loads and stores, no allocation. Numbers are written
into byte buffers by hand for the same reason.

Layout (byte offsets from ANCHOR):
    0 .. 1024        counters and lengths (the A_ constants)
    1024 .. 16384    text the crash report copies: stamp, paths, cvars, status
    16384 .. 32768   crash scratch: backtrace frames, the crash folder's path
    32768 .. 98304   a format buffer for the crash handler
    98304 .. 131072  the log's flush buffer
    131072 ..        the log ring (core/log.mojo)
    2 MiB .. 4 MiB   alternate signal stacks, 64 KiB per thread
"""
from std.ffi import external_call, _Global
from std.atomic import Atomic, Ordering
from std.sys.info import CompilationTarget
from std.time import monotonic

comptime ANCHOR_BYTES: Int = STACKS + STACK_SLOTS * STACK_BYTES
comptime ANCHOR_MAGIC: Int = 0x31454641534453  # "SDSAFE1"
comptime _ANCHOR_CTRL_KEY: StaticString = "FLINT_ANCHOR_CTRL"

# Words.
comptime A_MAGIC: Int = 0
comptime A_PID: Int = 8
comptime A_LOG_HEAD: Int = 16  # how many log entries were ever claimed
comptime A_LOG_FLUSHED: Int = 24  # the next entry to write to the file
comptime A_LOG_FD: Int = 32
comptime A_LOG_DROPPED: Int = 40
comptime A_LOG_ECHO: Int = 48  # entries at this level or above also go to stderr
comptime A_TICK: Int = 56  # the simulation tick now running, or -1
comptime A_CRASHED: Int = 64
comptime A_ENSURE_FAILURES: Int = 72
comptime A_CHECK_FAILURES: Int = 80
comptime A_GPU_HANGS: Int = 88
comptime A_REPLAY_COMMITTED: Int = 96  # bytes of whole ticks in the replay buffer
comptime A_REPLAY_PENDING: Int = 104  # plus the input of the tick now running
comptime A_REPLAY_BUFFER: Int = 112
comptime A_REPLAY_TICKS: Int = 120  # whole ticks recorded, in the file and the buffer
comptime A_START_NS: Int = 128
comptime A_NEXT_STACK: Int = 136
comptime A_HOT_RELOADS: Int = 144
comptime A_HANDLER_INSTALLED: Int = 152
comptime A_REPLAY_ACTIVE: Int = 160
comptime A_LOG_LEVEL: Int = 168  # the default level for categories set to follow it
comptime A_CATEGORY_LEVELS: Int = 256  # one byte per category
comptime A_TEXT_LENGTHS: Int = 512  # one word per text area

# Text areas: offset and capacity.
comptime T_STAMP: Int = 0
comptime T_STATE_DIR: Int = 1
comptime T_REPLAY_PATH: Int = 2
comptime T_LOG_PATH: Int = 3
comptime T_EXE_PATH: Int = 4
comptime T_REASON: Int = 5
comptime T_STATUS: Int = 6
comptime T_CVARS: Int = 7
comptime T_BUILD_ID: Int = 8
comptime TEXT_AREAS: Int = 9

comptime SCRATCH_FRAMES: Int = 16384  # 64 return addresses
comptime SCRATCH_SIGNAL: Int = 16896  # a struct sigaction and a timespec for the handler
comptime SCRATCH_PATH: Int = 17408  # 1 KiB
comptime SCRATCH_PATH2: Int = 18432  # 1 KiB
comptime SCRATCH_READ: Int = 19456  # 4 KiB
comptime FORMAT_BUFFER: Int = 32768
comptime FORMAT_BYTES: Int = 65536  # the log's flush buffer (core/log.mojo) follows it
comptime LOG_RING: Int = 131072
comptime STACKS: Int = 2 << 20
comptime STACK_BYTES: Int = 64 << 10
comptime STACK_SLOTS: Int = 33  # the main thread and up to 32 workers (MAX_WORKERS + 1) at once

# System call numbers (Linux x86-64 and aarch64).
comptime IS_X86: Bool = CompilationTarget.is_x86()
comptime SYS_READ: Int = 0 if IS_X86 else 63
comptime SYS_WRITE: Int = 1 if IS_X86 else 64
comptime SYS_CLOSE: Int = 3 if IS_X86 else 57
comptime SYS_MSYNC: Int = 26 if IS_X86 else 227
comptime SYS_NANOSLEEP: Int = 35 if IS_X86 else 101
comptime SYS_GETPID: Int = 39 if IS_X86 else 172
comptime SYS_KILL: Int = 62 if IS_X86 else 129
comptime SYS_GETTID: Int = 186 if IS_X86 else 178
comptime SYS_CLOCK_GETTIME: Int = 228 if IS_X86 else 113
comptime SYS_TGKILL: Int = 234 if IS_X86 else 131
comptime SYS_OPENAT: Int = 257 if IS_X86 else 56
comptime SYS_MKDIRAT: Int = 258 if IS_X86 else 34
comptime SYS_LINKAT: Int = 265 if IS_X86 else 37
comptime SYS_READLINKAT: Int = 267 if IS_X86 else 78
comptime AT_FDCWD: Int = -100
comptime O_WRONLY_CREATE_TRUNC: Int = 1 | 64 | 512 | 0x80000  # and close-on-exec
comptime O_WRONLY_CREATE_APPEND: Int = 1 | 64 | 1024 | 0x80000
comptime O_READ: Int = 0x80000


def _anchor_on_exit(ctrl: Int) abi("C"):
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
            _ = external_call["munmap", Int32](Int(cur), Int(ANCHOR_BYTES))
            break
    external_call["free", NoneType](ctrl)


def _init_anchor_ctrl() -> Int:
    var ctrl = external_call["aligned_alloc", Int](Int(64), Int(64))
    if ctrl == 0:
        return 0
    var p = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=ctrl)
    for i in range(8):
        p[unsafe_offset=i] = 0
    _ = external_call["__cxa_atexit", Int32](_anchor_on_exit, ctrl, Int(0))
    return ctrl


comptime _ANCHOR_GLOBAL = _Global["FLINT_ANCHOR_CTRL", _init_anchor_ctrl]


@always_inline
def _anchor_ctrl_addr() -> Int:
    var slot = external_call["KGEN_CompilerRT_GetGlobalOrNull", Int](
        Int(_ANCHOR_CTRL_KEY.unsafe_ptr()), _ANCHOR_CTRL_KEY.byte_length()
    )
    if slot == 0:
        return 0
    return Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=slot)[unsafe_offset=0]


@always_inline
def _anchor_ctrl_addr_or_create() -> Int:
    try:
        return _ANCHOR_GLOBAL.get_or_create_ptr()[unsafe_offset=0]
    except:
        return 0


@always_inline
def anchor_base() -> Int:
    """Async-signal-safe, zero-alloc base address of the safety anchor, or 0 if uninstalled."""
    var ctrl = _anchor_ctrl_addr()
    if ctrl == 0:
        return 0
    return Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())


def text_offset(area: Int) -> Int:
    """Where an area starts. Branches, not a list: a list would allocate."""
    var base = anchor_base()
    if base == 0:
        return 0
    if area == T_STAMP:
        return base + 1024
    if area <= T_STATUS:
        return base + 1152 + (area - 1) * 512
    if area == T_CVARS:
        return base + 4352
    return base + 12544


def text_capacity(area: Int) -> Int:
    if area == T_STAMP:
        return 128
    if area < T_STATUS:
        return 512
    if area == T_STATUS:
        return 640
    if area == T_CVARS:
        return 8192
    return 64


def sys(number: Int, a: Int = 0, b: Int = 0, c: Int = 0, d: Int = 0, e: Int = 0, f: Int = 0) -> Int:
    """A raw system call. Returns the result, negative errno on failure."""
    var result = external_call["syscall", Int](number, a, b, c, d, e, f)
    if result == -1:
        return -Int(external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()[unsafe_offset=0])
    return result


def c_string(text: String) -> List[UInt8]:
    """Text with a zero byte after it, for a system call. Keep it alive past the call."""
    var bytes = List[UInt8](text.as_bytes())
    bytes.append(0)
    return bytes^


def load(address: Int) -> Int:
    return Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0]


def store(address: Int, value: Int):
    Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0] = value


def load_byte(address: Int) -> Int:
    return Int(Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0])


def store_byte(address: Int, value: Int):
    Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=address)[unsafe_offset=0] = UInt8(value & 0xFF)


def atomic_word(offset: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
    return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=anchor_base() + offset)


def anchor_add(offset: Int, amount: Int = 1) -> Int:
    """Add to an anchor counter from any thread. Returns the value before."""
    var base = anchor_base()
    if base == 0:
        return 0
    return Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base + offset)[unsafe_offset=0].fetch_add(Int64(amount)))


def anchor_get(offset: Int) -> Int:
    var base = anchor_base()
    if base == 0:
        return 0
    return Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base + offset)[unsafe_offset=0].load())


def anchor_set(offset: Int, value: Int):
    var base = anchor_base()
    if base == 0:
        return
    Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=base + offset)[unsafe_offset=0].store(Int64(value))


def anchor_ready() -> Bool:
    """True when this process has installed the anchor (zero-alloc if uninstalled)."""
    var base = anchor_base()
    if base == 0:
        return False
    return load(base + A_MAGIC) == ANCHOR_MAGIC and load(base + A_PID) == sys(SYS_GETPID)


def monotonic_ns() -> Int:
    return Int(monotonic())


def install_anchor() -> Bool:
    """Map the anchor dynamically for this process. Safe to repeat and to race."""
    if anchor_ready():
        return True
    var ctrl = _anchor_ctrl_addr_or_create()
    if ctrl == 0:
        return False
    var base_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctrl)
    var existing = Int(base_ptr[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
    if existing != 0 and load(existing + A_MAGIC) == ANCHOR_MAGIC:
        store(existing + A_PID, sys(SYS_GETPID))
        return True
    # Dynamic OS anonymous mmap (no fixed address)
    var mapped = external_call["mmap", Int](Int(0), ANCHOR_BYTES, Int32(3), Int32(0x22), Int32(-1), Int(0))
    if mapped == 0 or mapped == -1:
        return False
    store(mapped + A_PID, sys(SYS_GETPID))
    store(mapped + A_LOG_FD, -1)
    store(mapped + A_TICK, -1)
    store(mapped + A_LOG_ECHO, 3)  # warnings and worse
    store(mapped + A_LOG_LEVEL, 2)  # info
    for category in range(32):
        store_byte(mapped + A_CATEGORY_LEVELS + category, 255)  # follow the default
    store(mapped + A_START_NS, monotonic_ns())
    store(mapped + A_MAGIC, ANCHOR_MAGIC)
    var expected = Int64(0)
    if not base_ptr[unsafe_offset=0].compare_exchange[
        failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
    ](expected, Int64(mapped)):
        _ = external_call["munmap", Int32](mapped, Int(ANCHOR_BYTES))
    return anchor_ready()


def anchor_shutdown():
    """Unmap the safety anchor early (idempotent with __cxa_atexit)."""
    var ctrl = _anchor_ctrl_addr()
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
            _ = external_call["munmap", Int32](Int(cur), Int(ANCHOR_BYTES))
            return


# ---------------------------------------------------------------------------
# Text areas
# ---------------------------------------------------------------------------

def set_text(area: Int, text: String):
    """Copy text into an area, cut to fit. The length is written last."""
    var anchor = anchor_base()
    if anchor == 0:
        return
    var bytes = text.as_bytes()
    var count = min(len(bytes), text_capacity(area))
    var base = text_offset(area)
    store(anchor + A_TEXT_LENGTHS + area * 8, 0)
    for index in range(count):
        store_byte(base + index, Int(bytes[index]))
    store(anchor + A_TEXT_LENGTHS + area * 8, count)


def text_length(area: Int) -> Int:
    var anchor = anchor_base()
    if anchor == 0:
        return 0
    return min(load(anchor + A_TEXT_LENGTHS + area * 8), text_capacity(area))


def get_text(area: Int) -> String:
    var anchor = anchor_base()
    if anchor == 0:
        return String()
    var bytes = List[UInt8]()
    var base = text_offset(area)
    for index in range(text_length(area)):
        bytes.append(UInt8(load_byte(base + index)))
    return String(unsafe_from_utf8=bytes)


# ---------------------------------------------------------------------------
# Writing without allocating, for the crash handler
# ---------------------------------------------------------------------------

struct RawText:
    """Bytes built up in a fixed buffer. Writing past the end is cut off."""
    var base: Int
    var capacity: Int
    var length: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.length = 0

    def byte(mut self, value: Int):
        if self.length < self.capacity:
            store_byte(self.base + self.length, value)
            self.length += 1

    def bytes_at(mut self, address: Int, count: Int):
        for index in range(count):
            self.byte(load_byte(address + index))

    def text(mut self, literal: StaticString):
        var bytes = literal.as_bytes()
        for index in range(len(bytes)):
            self.byte(Int(bytes[index]))

    def area(mut self, area: Int):
        self.bytes_at(text_offset(area), text_length(area))

    def decimal(mut self, value: Int):
        if value < 0:
            self.byte(45)
            if value == Int.MIN:
                self.text("9223372036854775808")
                return
            self.decimal(-value)
            return
        var digits = 1
        var scale = 1
        while value // scale >= 10:
            scale *= 10
            digits += 1
        while scale > 0:
            self.byte(48 + (value // scale) % 10)
            scale //= 10

    def padded(mut self, value: Int, width: Int):
        var scale = 1
        for _ in range(width - 1):
            scale *= 10
        while scale > 0:
            self.byte(48 + (value // scale) % 10)
            scale //= 10

    def hex(mut self, value: Int, digits: Int = 16):
        self.text("0x")
        var bits = UInt64(value)
        for position in range(digits - 1, -1, -1):
            var digit = Int((bits >> UInt64(position * 4)) & 15)
            self.byte(48 + digit if digit < 10 else 87 + digit)

    def write_to(self, fd: Int) -> Bool:
        var offset = 0
        while offset < self.length:
            var written = sys(SYS_WRITE, fd, self.base + offset, self.length - offset)
            if written <= 0:
                return False
            offset += written
        return True

    def terminate(mut self) -> Int:
        """End with a zero byte, as a path for the kernel. Returns the address."""
        if self.length >= self.capacity:
            self.length = self.capacity - 1
        store_byte(self.base + self.length, 0)
        return self.base


def write_all(fd: Int, address: Int, count: Int) -> Bool:
    var offset = 0
    while offset < count:
        var written = sys(SYS_WRITE, fd, address + offset, count - offset)
        if written <= 0:
            return False
        offset += written
    return True


def write_text(fd: Int, literal: StaticString):
    var bytes = literal.as_bytes()
    _ = write_all(fd, Int(bytes.unsafe_ptr()), len(bytes))


def civil_date(days: Int) -> Tuple[Int, Int, Int]:
    """Days since 1970-01-01 to (year, month, day), proleptic Gregorian."""
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    return (y + 1 if m <= 2 else y, m, d)


def write_utc(mut out: RawText, seconds: Int, compact: Bool):
    """2026-11-02T13:04:05Z, or 20261102-130405 for file names."""
    var date = civil_date(seconds // 86400)
    var of_day = seconds % 86400
    out.padded(date[0], 4)
    if not compact:
        out.byte(45)
    out.padded(date[1], 2)
    if not compact:
        out.byte(45)
    out.padded(date[2], 2)
    out.byte(45 if compact else 84)
    out.padded(of_day // 3600, 2)
    if not compact:
        out.byte(58)
    out.padded((of_day // 60) % 60, 2)
    if not compact:
        out.byte(58)
    out.padded(of_day % 60, 2)
    if not compact:
        out.byte(90)


def wall_seconds(scratch: Int = 0) -> Int:
    """Seconds since 1970. The crash handler passes anchor scratch memory to read
    the clock into; everyone else gets a small allocation."""
    if scratch != 0:
        _ = sys(SYS_CLOCK_GETTIME, 0, scratch)
        return load(scratch)
    var time = List[Int](length=2, fill=0)
    _ = sys(SYS_CLOCK_GETTIME, 0, Int(time.unsafe_ptr()))
    var value = time[0]
    _ = time^
    return value
