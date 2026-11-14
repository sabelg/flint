"""The crash handler: a report, the log tail and the replay up to the crash.

`install_crash_handler` (after `start_log`) catches SIGSEGV, SIGBUS, SIGFPE,
SIGILL, SIGTRAP and SIGABRT on an alternate signal stack (`sigaltstack`), so
even a stack overflow is reported. Worker threads take their own stack with
`install_thread_stack`. On a crash the handler writes, into
`$XDG_STATE_HOME/sharddepth/crashes/<UTC time>-<pid>/`:

    crash.txt    the signal and faulting address, the reason (for a failed
                 check), build stamp and build id, tick, counters, registers,
                 the raw backtrace, console variables, the last log entries,
                 and the executable mappings `tools/symbolize` needs
    replay.sdrp  the session's replay: every whole tick, then the input of the
                 tick that crashed with zero checksums (tools/replay.mojo
                 replays the whole ticks and then runs the crashing tick)

and prints `SHARDDEPTH_CRASH <folder>` to stderr. Then it restores the default
action and lets the signal kill the process as it would have, core dump
included.

The handler allocates nothing. It uses raw system calls and the anchor's fixed
buffers (core/anchor.mojo); everything it copies (stamp, paths, console
variables, status) was put into the anchor beforehand by ordinary code.
`backtrace` is called once at install so its unwinder is loaded before any
crash.
"""
from std.ffi import external_call
from std.atomic import Atomic
from flint.core.anchor import (
    ANCHOR, A_CHECK_FAILURES, A_CRASHED, A_ENSURE_FAILURES, A_GPU_HANGS, A_HANDLER_INSTALLED, A_HOT_RELOADS,
    A_LOG_DROPPED, A_LOG_FD, A_LOG_HEAD, A_NEXT_STACK, A_REPLAY_ACTIVE, A_REPLAY_BUFFER, A_REPLAY_COMMITTED,
    A_REPLAY_PENDING, A_REPLAY_TICKS, A_TICK, AT_FDCWD, FORMAT_BUFFER, FORMAT_BYTES, IS_X86, O_READ, O_WRONLY_CREATE_APPEND,
    O_WRONLY_CREATE_TRUNC, RawText, SCRATCH_FRAMES, SCRATCH_PATH, SCRATCH_SIGNAL, SCRATCH_PATH2, SCRATCH_READ, STACKS, STACK_BYTES,
    STACK_SLOTS, SYS_CLOSE, SYS_GETPID, SYS_GETTID, SYS_LINKAT, SYS_MKDIRAT, SYS_NANOSLEEP, SYS_OPENAT, SYS_READ, SYS_READLINKAT,
    SYS_TGKILL, SYS_WRITE, T_BUILD_ID, T_CVARS, T_EXE_PATH, T_REASON, T_REPLAY_PATH, T_STAMP, T_STATE_DIR, T_STATUS,
    anchor_add, anchor_ready, atomic_word, c_string, load, set_text, store, sys, text_length, wall_seconds,
    write_utc,
)
from flint.core.build_info import build_stamp
from flint.core.log import LOG_ENTRIES, flush_log, format_entry, start_log
from flint.platform.os import read_bytes

comptime CrashHandler = def(Int32, Int, Int) abi("C") thin -> None
comptime SIGILL: Int = 4
comptime SIGTRAP: Int = 5
comptime SIGABRT: Int = 6
comptime SIGBUS: Int = 7
comptime SIGFPE: Int = 8
comptime SIGSEGV: Int = 11
comptime SA_SIGINFO: Int = 4
comptime SA_ONSTACK: Int = 0x08000000
comptime TAIL_ENTRIES: Int = 64
comptime CHECK_BYTES: Int = 64  # a tick record's checksums (core/replay.mojo: CHECK_SECTIONS * 8)
comptime UCONTEXT_RSP: Int = 160 if IS_X86 else 432
comptime UCONTEXT_RIP: Int = 168 if IS_X86 else 440
comptime UCONTEXT_RBP: Int = 120 if IS_X86 else 416


def crash_signals() -> List[Int]:
    return [SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGABRT]


def write_signal_name(mut out: RawText, signal: Int):
    if signal == SIGSEGV:
        out.text("SIGSEGV")
    elif signal == SIGBUS:
        out.text("SIGBUS")
    elif signal == SIGFPE:
        out.text("SIGFPE")
    elif signal == SIGILL:
        out.text("SIGILL")
    elif signal == SIGTRAP:
        out.text("SIGTRAP")
    elif signal == SIGABRT:
        out.text("SIGABRT")
    else:
        out.text("signal")


def le_word(data: List[UInt8], offset: Int, size: Int) -> Int:
    var value = 0
    for index in range(size):
        value |= Int(data[offset + index]) << (index * 8)
    return value


def hex_digit(value: Int) -> String:
    return chr(48 + value if value < 10 else 87 + value)


def elf_build_id(data: List[UInt8]) -> String:
    """The GNU build id of an ELF file as hex, or empty. Read from its notes."""
    if len(data) < 64 or data[0] != 0x7F or data[1] != 0x45 or data[2] != 0x4C or data[3] != 0x46:
        return String()
    var program = le_word(data, 32, 8)
    var entry_size = le_word(data, 54, 2)
    var count = le_word(data, 56, 2)
    for header in range(count):
        var base = program + header * entry_size
        if base + 56 > len(data) or le_word(data, base, 4) != 4:  # PT_NOTE
            continue
        var offset = le_word(data, base + 8, 8)
        var end = offset + le_word(data, base + 32, 8)
        while offset + 12 <= end and end <= len(data):
            var name_size = le_word(data, offset, 4)
            var description_size = le_word(data, offset + 4, 4)
            var kind = le_word(data, offset + 8, 4)
            var description = offset + 12 + ((name_size + 3) & ~3)
            if kind == 3 and name_size == 4 and description + description_size <= len(data):
                var text = String()
                for index in range(description_size):
                    var byte = Int(data[description + index])
                    text += hex_digit(byte >> 4) + hex_digit(byte & 15)
                return text^
            offset = description + ((description_size + 3) & ~3)
    return String()


def executable_path() -> String:
    var buffer = List[UInt8](length=1024, fill=0)
    var link = c_string("/proc/self/exe")
    var count = sys(SYS_READLINKAT, AT_FDCWD, Int(link.unsafe_ptr()), Int(buffer.unsafe_ptr()), 1023)
    _ = link^
    if count <= 0:
        return String()
    var bytes = List[UInt8]()
    for index in range(count):
        bytes.append(buffer[index])
    _ = buffer^
    return String(unsafe_from_utf8=bytes)


def install_thread_stack() -> Bool:
    """Give this thread its own signal stack from the anchor. Every thread that may
    crash should have one; without it a stack overflow cannot be reported."""
    if not anchor_ready():
        return False
    # The first thread (main) keeps slot 0; others take 1..31 in turn, so pools that
    # stop and start again reuse the stacks of threads that are gone.
    var claim = anchor_add(A_NEXT_STACK)
    var slot = 0 if claim == 0 else 1 + (claim - 1) % (STACK_SLOTS - 1)
    var stack = List[Int](length=3, fill=0)  # stack_t: ss_sp, ss_flags, ss_size
    stack[0] = ANCHOR + STACKS + slot * STACK_BYTES
    stack[2] = STACK_BYTES
    var result = external_call["sigaltstack", Int32](Int(stack.unsafe_ptr()), Int(0))
    _ = stack^
    return result == 0


def install_crash_handler(status: String = "") -> Bool:
    """Catch fatal signals on an alternate stack and write a crash report.
    Starts the log if nobody has. Call once, early, from the main thread."""
    if not anchor_ready() and not start_log():
        if not anchor_ready():
            return False
    set_text(T_STAMP, build_stamp())
    var executable = executable_path()
    set_text(T_EXE_PATH, executable)
    set_text(T_BUILD_ID, elf_build_id(read_bytes(executable)) if executable.byte_length() > 0 else String())
    if status.byte_length() > 0:
        set_text(T_STATUS, status)
    # Load the unwinder now: the first call may allocate, which a handler must not.
    var warm = List[Int](length=4, fill=0)
    _ = external_call["backtrace", Int32](Int(warm.unsafe_ptr()), Int32(4))
    _ = warm^
    if not install_thread_stack():
        return False
    var action = List[Int](length=19, fill=0)  # struct sigaction: 152 bytes
    Pointer[CrashHandler, origin=MutAnyOrigin](unsafe_from_address=Int(action.unsafe_ptr()))[unsafe_offset=0] = on_crash
    action[17] = SA_SIGINFO | SA_ONSTACK
    var ok = True
    for signal in crash_signals():
        if external_call["sigaction", Int32](Int32(signal), Int(action.unsafe_ptr()), Int(0)) != 0:
            ok = False
    _ = action^
    store(ANCHOR + A_HANDLER_INSTALLED, 1 if ok else 0)
    return ok


def set_crash_status(text: String):
    """A line for the report about what the game is doing (layer, backend, mode)."""
    if anchor_ready():
        set_text(T_STATUS, text)


def set_crash_cvars(text: String):
    """The console variables as the report should list them, one per line."""
    if anchor_ready():
        set_text(T_CVARS, text)


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------

def make_folder(path: RawText) -> Bool:
    var result = sys(SYS_MKDIRAT, AT_FDCWD, path.base, 0o755)
    return result == 0 or result == -17  # EEXIST


def path_join(mut out: RawText, area: Int, tail: StaticString) -> Int:
    out.length = 0
    out.area(area)
    out.text(tail)
    return out.terminate()


def write_maps(fd: Int, mut out: RawText):
    """Copy the executable lines of /proc/self/maps: what symbolize needs."""
    var path = RawText(ANCHOR + SCRATCH_PATH2, 1024)
    path.text("/proc/self/maps")
    var maps = sys(SYS_OPENAT, AT_FDCWD, path.terminate(), O_READ)
    if maps < 0:
        out.text("(maps unreadable)\n")
        return
    var line_start = out.length
    var keep = False
    var field = 0
    while True:
        var count = sys(SYS_READ, maps, ANCHOR + SCRATCH_READ, 4096)
        if count <= 0:
            break
        for index in range(count):
            var byte = Int(Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=ANCHOR + SCRATCH_READ + index)[unsafe_offset=0])
            out.byte(byte)
            if byte == 32:
                field += 1
            elif byte == 120 and field == 1:  # an 'x' among the permissions
                keep = True
            elif byte == 10:
                if not keep:
                    out.length = line_start
                elif out.length > FORMAT_BYTES - 2048:
                    _ = out.write_to(fd)
                    out.length = 0
                line_start = out.length
                keep = False
                field = 0
    _ = sys(SYS_CLOSE, maps)


def copy_file(source: Int, destination: Int) -> Bool:
    var input = sys(SYS_OPENAT, AT_FDCWD, source, O_READ)
    if input < 0:
        return False
    var output = sys(SYS_OPENAT, AT_FDCWD, destination, O_WRONLY_CREATE_TRUNC, 0o644)
    if output < 0:
        _ = sys(SYS_CLOSE, input)
        return False
    var ok = True
    while True:
        var count = sys(SYS_READ, input, ANCHOR + SCRATCH_READ, 4096)
        if count == 0:
            break
        if count < 0:
            ok = False
            break
        var written = 0
        while written < count:
            var step = sys(SYS_WRITE, output, ANCHOR + SCRATCH_READ + written, count - written)
            if step <= 0:
                ok = False
                break
            written += step
        if not ok:
            break
    _ = sys(SYS_CLOSE, input)
    _ = sys(SYS_CLOSE, output)
    return ok


def save_replay(mut out: RawText, mut path: RawText, folder_length: Int) -> Int:
    """Append what the play loop had not written yet, then put the whole file in
    the crash folder. Returns the number of whole ticks, or -1 with no replay."""
    if load(ANCHOR + A_REPLAY_ACTIVE) == 0:
        return -1
    var source = RawText(ANCHOR + SCRATCH_PATH2, 1024)
    source.area(T_REPLAY_PATH)
    _ = source.terminate()
    var fd = sys(SYS_OPENAT, AT_FDCWD, source.base, O_WRONLY_CREATE_APPEND, 0o644)
    if fd < 0:
        return -1
    var buffer = load(ANCHOR + A_REPLAY_BUFFER)
    var committed = load(ANCHOR + A_REPLAY_COMMITTED)
    var pending = load(ANCHOR + A_REPLAY_PENDING)
    if buffer != 0 and committed > 0:
        _ = write_all_raw(fd, buffer, committed)
    if buffer != 0 and pending > committed:
        # The crashing tick: its input, then zero checksums, which no real tick has.
        _ = write_all_raw(fd, buffer + committed, pending - committed)
        for index in range(CHECK_BYTES):
            Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=ANCHOR + SCRATCH_READ + index)[unsafe_offset=0] = 0
        _ = write_all_raw(fd, ANCHOR + SCRATCH_READ, CHECK_BYTES)
    _ = sys(SYS_CLOSE, fd)
    path.length = folder_length
    path.text("/replay.sdrp")
    _ = path.terminate()
    if sys(SYS_LINKAT, AT_FDCWD, source.base, AT_FDCWD, path.base, 0) != 0:
        _ = copy_file(source.base, path.base)
    return load(ANCHOR + A_REPLAY_TICKS)


def write_all_raw(fd: Int, address: Int, count: Int) -> Bool:
    var offset = 0
    while offset < count:
        var written = sys(SYS_WRITE, fd, address + offset, count - offset)
        if written <= 0:
            return False
        offset += written
    return True


def on_crash(signal: Int32, info: Int, context: Int) abi("C"):
    var number = Int(signal)
    var tid = sys(SYS_GETTID)
    var expected = Int64(0)
    var first = 0
    if not atomic_word(A_CRASHED)[unsafe_offset=0].compare_exchange(expected, Int64(tid)):
        first = Int(expected)
    if first != 0:
        # Another thread is writing the report. Wait for it to end the process;
        # this same thread failing again falls through to the default action.
        if first != tid:
            store(ANCHOR + SCRATCH_SIGNAL + 256, 10)  # a timespec: ten seconds
            store(ANCHOR + SCRATCH_SIGNAL + 264, 0)
            _ = sys(SYS_NANOSLEEP, ANCHOR + SCRATCH_SIGNAL + 256, 0)
        restore_and_reraise(number, info)
        return
    var code = Int(Pointer[Int32, origin=MutAnyOrigin](unsafe_from_address=info + 8)[unsafe_offset=0])
    var address = load(info + 16)
    var pc = load(context + UCONTEXT_RIP) if context != 0 else 0
    var sp = load(context + UCONTEXT_RSP) if context != 0 else 0
    var fp = load(context + UCONTEXT_RBP) if context != 0 else 0
    var frames = Int(external_call["backtrace", Int32](ANCHOR + SCRATCH_FRAMES, Int32(64)))

    # The folder: <state>/crashes/<UTC>-<pid>
    var path = RawText(ANCHOR + SCRATCH_PATH, 1024)
    path.area(T_STATE_DIR)
    _ = path.terminate()
    _ = make_folder(path)
    path.text("/crashes")
    _ = path.terminate()
    _ = make_folder(path)
    path.byte(47)
    var seconds = wall_seconds(ANCHOR + SCRATCH_SIGNAL + 320)
    write_utc(path, seconds, True)
    path.byte(45)
    path.decimal(sys(SYS_GETPID))
    _ = path.terminate()
    _ = make_folder(path)
    var folder_length = path.length

    var out = RawText(ANCHOR + FORMAT_BUFFER, FORMAT_BYTES)
    path.text("/crash.txt")
    var fd = sys(SYS_OPENAT, AT_FDCWD, path.terminate(), O_WRONLY_CREATE_TRUNC, 0o644)
    var replay_ticks = save_replay(out, path, folder_length)

    out.text("Sharddepth crash report\nsignal: ")
    write_signal_name(out, number)
    out.text(" (")
    out.decimal(number)
    out.text(") code=")
    out.decimal(code)
    if code > 0:
        out.text(" address=")
        out.hex(address)
    else:
        out.text(" (sent, not a fault)")
    out.text("\nreason: ")
    if text_length(T_REASON) > 0:
        out.area(T_REASON)
    else:
        out.text("a fatal signal")
    out.text("\ntime: ")
    write_utc(out, seconds, False)
    out.text("\nbuild: ")
    out.area(T_STAMP)
    out.text("\nbuild_id: ")
    out.area(T_BUILD_ID)
    out.text("\nexecutable: ")
    out.area(T_EXE_PATH)
    out.text("\npid: ")
    out.decimal(sys(SYS_GETPID))
    out.text(" tid: ")
    out.decimal(tid)
    out.text("\ntick: ")
    out.decimal(load(ANCHOR + A_TICK))
    out.text("\nreplay: ")
    if replay_ticks >= 0:
        out.text("replay.sdrp (")
        out.decimal(replay_ticks)
        out.text(" whole ticks")
        if load(ANCHOR + A_REPLAY_PENDING) > load(ANCHOR + A_REPLAY_COMMITTED):
            out.text(", then the input of the tick that crashed")
        out.text(")")
    else:
        out.text("none (not recording)")
    out.text("\nstatus: ")
    out.area(T_STATUS)
    out.text("\ncounters: ensure_failures=")
    out.decimal(load(ANCHOR + A_ENSURE_FAILURES))
    out.text(" check_failures=")
    out.decimal(load(ANCHOR + A_CHECK_FAILURES))
    out.text(" gpu_hangs=")
    out.decimal(load(ANCHOR + A_GPU_HANGS))
    out.text(" hot_reloads=")
    out.decimal(load(ANCHOR + A_HOT_RELOADS))
    out.text(" log_entries=")
    out.decimal(load(ANCHOR + A_LOG_HEAD))
    out.text(" log_dropped=")
    out.decimal(load(ANCHOR + A_LOG_DROPPED))
    out.text("\nregisters: pc=")
    out.hex(pc)
    out.text(" sp=")
    out.hex(sp)
    out.text(" fp=")
    out.hex(fp)
    out.text("\nbacktrace:\n")
    for index in range(frames):
        out.text("  #")
        out.decimal(index)
        out.text(" ")
        out.hex(load(ANCHOR + SCRATCH_FRAMES + index * 8))
        out.byte(10)
    out.text("  pc ")
    out.hex(pc)
    out.text("\ncvars:\n")
    out.area(T_CVARS)
    out.text("\nlog:\n")
    if fd >= 0:
        _ = out.write_to(fd)
    out.length = 0
    var head = load(ANCHOR + A_LOG_HEAD)
    var first_entry = head - TAIL_ENTRIES if head > TAIL_ENTRIES else 0
    for index in range(first_entry, head):
        out.text("  ")
        if not format_entry(out, index):
            out.length -= 2
        if out.length > FORMAT_BYTES - 1024:
            if fd >= 0:
                _ = out.write_to(fd)
            out.length = 0
    out.text("maps:\n")
    if fd >= 0:
        _ = out.write_to(fd)
    out.length = 0
    write_maps(fd, out)
    out.text("end\n")
    if fd >= 0:
        _ = out.write_to(fd)
        _ = sys(SYS_CLOSE, fd)
    _ = flush_log()

    out.length = 0
    out.text("SHARDDEPTH_CRASH ")
    write_signal_name(out, number)
    out.text(" report=")
    out.bytes_at(path.base, folder_length)
    out.text("/crash.txt\n")
    _ = out.write_to(2)
    restore_and_reraise(number, info)


def restore_and_reraise(signal: Int, info: Int):
    """Put the default action back. A fault re-runs its instruction when the handler
    returns and dies of it; a signal that was sent is sent again."""
    for index in range(19):  # a zeroed struct sigaction: SIG_DFL
        store(ANCHOR + SCRATCH_SIGNAL + index * 8, 0)
    _ = external_call["sigaction", Int32](Int32(signal), ANCHOR + SCRATCH_SIGNAL, Int(0))
    var code = Int(Pointer[Int32, origin=MutAnyOrigin](unsafe_from_address=info + 8)[unsafe_offset=0])
    if code <= 0 or signal == SIGABRT:
        _ = sys(SYS_TGKILL, sys(SYS_GETPID), sys(SYS_GETTID), signal)
