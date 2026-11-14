"""Log levels and categories, in a lock-free ring flushed to a file.

    log_info(LOG_GPU, "context recreated")
    log_warn(LOG_ASSETS, "missing clip " + name)

Any thread may log. A writer claims an entry with one atomic add on the head
counter, fills it, and publishes it by writing the entry's sequence word
last; nobody waits for anybody. The ring holds the last `LOG_ENTRIES`
entries. `flush_log` (the main loop calls it once a frame, and at exit)
writes what is new to `$XDG_STATE_HOME/sharddepth/logs/sharddepth.log`; an
entry overwritten before it was flushed is counted as dropped. The crash
handler reads the same ring, so the last lines before a crash are in its
report even if they never reached the file.

Each category has its own level (`log.<category>` console variables, or
`SD_LOG=gpu=debug,assets=warn`); a category left at "follow" uses
`log.level`. Entries at `log.echo` or above are also printed to stderr.
Logging never touches the simulation: entries carry the tick only to read.

Without the anchor (a tool or test that did not call `start_log`), entries at
the echo level or above are printed and the rest are dropped.
"""
from std.atomic import Atomic
from flint.core.anchor import (
    ANCHOR, A_CATEGORY_LEVELS, A_LOG_DROPPED, A_LOG_ECHO, A_LOG_FD, A_LOG_FLUSHED, A_LOG_HEAD, A_LOG_LEVEL,
    A_START_NS, A_TICK, LOG_RING, O_WRONLY_CREATE_APPEND, AT_FDCWD, RawText, SYS_CLOSE, SYS_GETTID, SYS_OPENAT,
    T_LOG_PATH, T_STATE_DIR, anchor_get, anchor_ready, anchor_set, atomic_word, install_anchor, load, load_byte,
    monotonic_ns, set_text, store, store_byte, sys, text_offset, text_length, wall_seconds, write_all, write_utc,
    FORMAT_BUFFER, c_string, get_text,
)
from flint.platform.os import env_value, make_dirs, rename_file, file_size

comptime LOG_TRACE: Int = 0
comptime LOG_DEBUG: Int = 1
comptime LOG_INFO: Int = 2
comptime LOG_WARN: Int = 3
comptime LOG_ERROR: Int = 4
comptime LOG_FATAL: Int = 5
comptime LOG_OFF: Int = 6
comptime LOG_FOLLOW: Int = 255

comptime LOG_CORE: Int = 0
comptime LOG_SIM: Int = 1
comptime LOG_RENDER: Int = 2
comptime LOG_GPU: Int = 3
comptime LOG_VULKAN: Int = 4
comptime LOG_AUDIO: Int = 5
comptime LOG_ASSETS: Int = 6
comptime LOG_INPUT: Int = 7
comptime LOG_REPLAY: Int = 8
comptime LOG_PHYSICS: Int = 9
comptime LOG_TOOLS: Int = 10
comptime LOG_MEMORY: Int = 11
comptime LOG_CATEGORIES: Int = 12

comptime LOG_ENTRIES: Int = 4096  # a power of two
comptime ENTRY_BYTES: Int = 256
comptime ENTRY_TEXT: Int = 32  # where the text starts in an entry
comptime TEXT_BYTES: Int = ENTRY_BYTES - ENTRY_TEXT
comptime FLUSH_BUFFER: Int = FORMAT_BUFFER + 65536  # 32 KiB the crash handler does not use
comptime FLUSH_BYTES: Int = 32768
comptime LOG_FILES_KEPT: Int = 5


def level_name(level: Int) -> String:
    var names = ["trace", "debug", "info", "warn", "error", "fatal", "off"]
    return String(names[level]) if level >= 0 and level <= LOG_OFF else "follow"


def level_letter(level: Int) -> Int:
    """T D I W E F, as a byte. Branches, so the crash handler can use it."""
    if level == LOG_TRACE:
        return 84
    if level == LOG_DEBUG:
        return 68
    if level == LOG_INFO:
        return 73
    if level == LOG_WARN:
        return 87
    if level == LOG_ERROR:
        return 69
    return 70


def category_name(category: Int) -> String:
    var names = ["core", "sim", "render", "gpu", "vulkan", "audio", "assets", "input", "replay", "physics", "tools", "memory"]
    return String(names[category]) if category >= 0 and category < LOG_CATEGORIES else "unknown"


def write_category(mut out: RawText, category: Int):
    """The category's name without allocating, for the crash handler."""
    if category == LOG_CORE:
        out.text("core")
    elif category == LOG_SIM:
        out.text("sim")
    elif category == LOG_RENDER:
        out.text("render")
    elif category == LOG_GPU:
        out.text("gpu")
    elif category == LOG_VULKAN:
        out.text("vulkan")
    elif category == LOG_AUDIO:
        out.text("audio")
    elif category == LOG_ASSETS:
        out.text("assets")
    elif category == LOG_INPUT:
        out.text("input")
    elif category == LOG_REPLAY:
        out.text("replay")
    elif category == LOG_PHYSICS:
        out.text("physics")
    elif category == LOG_TOOLS:
        out.text("tools")
    elif category == LOG_MEMORY:
        out.text("memory")
    else:
        out.text("?")


def parse_level(text: String) -> Int:
    """A level by name or number; -1 if it is neither."""
    for level in range(LOG_OFF + 1):
        if text == level_name(level):
            return level
    if text == "follow":
        return LOG_FOLLOW
    if text == "warning":
        return LOG_WARN
    try:
        var number = Int(text)
        if number >= 0 and number <= LOG_OFF:
            return number
    except:
        pass
    return -1


def find_category(name: String) -> Int:
    for category in range(LOG_CATEGORIES):
        if category_name(category) == name:
            return category
    return -1


def entry_address(index: Int) -> Int:
    return ANCHOR + LOG_RING + (index & (LOG_ENTRIES - 1)) * ENTRY_BYTES


def entry_sequence(index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
    return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=entry_address(index))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def category_level(category: Int) -> Int:
    """The level a category keeps entries from (warn without the anchor)."""
    if not anchor_ready():
        return LOG_WARN
    return category_level_unchecked(category)


def category_level_unchecked(category: Int) -> Int:
    var own = load_byte(ANCHOR + A_CATEGORY_LEVELS + category) if category >= 0 and category < 32 else LOG_FOLLOW
    return load(ANCHOR + A_LOG_LEVEL) if own == LOG_FOLLOW else own


def log_enabled(category: Int, level: Int) -> Bool:
    """Would an entry be kept? Use it to skip building an expensive message."""
    if not anchor_ready():
        return level >= LOG_WARN
    return level >= category_level_unchecked(category) or level >= load(ANCHOR + A_LOG_ECHO)


def log_message(category: Int, level: Int, message: String):
    """Add one entry. Lock-free; any thread."""
    if not anchor_ready():
        if level >= LOG_WARN:
            print("[" + level_name(level) + " " + category_name(category) + "] " + message)
        return
    if level >= load(ANCHOR + A_LOG_ECHO):
        print("[" + level_name(level) + " " + category_name(category) + "] " + message)
    if level < category_level_unchecked(category):
        return
    var claim = anchor_add_head()
    var entry = entry_address(claim)
    # Odd while writing, so a reader never takes half an entry.
    entry_sequence(claim)[unsafe_offset=0].store(Int64(claim * 2 + 1))
    var bytes = message.as_bytes()
    var count = min(len(bytes), TEXT_BYTES)
    store_byte(entry + 8, level)
    store_byte(entry + 9, category)
    store_byte(entry + 10, count & 0xFF)
    store_byte(entry + 11, count >> 8)
    Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=entry + 12)[unsafe_offset=0] = UInt32(sys(SYS_GETTID) & 0xFFFFFFFF)
    store(entry + 16, load(ANCHOR + A_TICK))
    store(entry + 24, monotonic_ns() - load(ANCHOR + A_START_NS))
    for index in range(count):
        store_byte(entry + ENTRY_TEXT + index, Int(bytes[index]))
    entry_sequence(claim)[unsafe_offset=0].store(Int64(claim * 2 + 2))


def anchor_add_head() -> Int:
    return Int(atomic_word(A_LOG_HEAD)[unsafe_offset=0].fetch_add(1))


def log_trace(category: Int, message: String):
    log_message(category, LOG_TRACE, message)


def log_debug(category: Int, message: String):
    log_message(category, LOG_DEBUG, message)


def log_info(category: Int, message: String):
    log_message(category, LOG_INFO, message)


def log_warn(category: Int, message: String):
    log_message(category, LOG_WARN, message)


def log_error(category: Int, message: String):
    log_message(category, LOG_ERROR, message)


def log_fatal(category: Int, message: String):
    log_message(category, LOG_FATAL, message)


def set_log_level(level: Int):
    if anchor_ready() and level >= 0 and level <= LOG_OFF:
        store(ANCHOR + A_LOG_LEVEL, level)


def set_category_level(category: Int, level: Int):
    """LOG_FOLLOW makes the category use the default level again."""
    if anchor_ready() and category >= 0 and category < LOG_CATEGORIES and (level == LOG_FOLLOW or (level >= 0 and level <= LOG_OFF)):
        store_byte(ANCHOR + A_CATEGORY_LEVELS + category, level)


def set_echo_level(level: Int):
    if anchor_ready() and level >= 0 and level <= LOG_OFF:
        store(ANCHOR + A_LOG_ECHO, level)


def apply_log_settings(text: String) -> Int:
    """`gpu=debug,assets=warn,level=info,echo=error`. Returns how many it set."""
    var applied = 0
    for part in text.split(","):
        var pieces = String(part).strip().split("=")
        if len(pieces) != 2:
            continue
        var name = String(String(pieces[0]).strip())
        var level = parse_level(String(String(pieces[1]).strip()))
        if level < 0:
            print("LOG_BAD_LEVEL " + String(part))
            continue
        if name == "level" and level != LOG_FOLLOW:
            set_log_level(level)
        elif name == "echo" and level != LOG_FOLLOW:
            set_echo_level(level)
        else:
            var category = find_category(name)
            if category < 0:
                print("LOG_UNKNOWN_CATEGORY " + name)
                continue
            set_category_level(category, level)
        applied += 1
    return applied


def set_log_tick(tick: Int):
    """The tick now running, stamped on entries and the crash report. -1 between ticks."""
    if anchor_ready():
        store(ANCHOR + A_TICK, tick)


# ---------------------------------------------------------------------------
# Reading: the file, and the crash report's tail
# ---------------------------------------------------------------------------

def format_entry(mut out: RawText, index: Int) -> Bool:
    """One line for entry `index`, or False if it was overwritten or is still
    being written. Allocates nothing, so the crash handler can call it."""
    var entry = entry_address(index)
    var sequence = Int(entry_sequence(index)[unsafe_offset=0].load())
    if sequence != index * 2 + 2:
        return False
    var start = out.length
    var level = load_byte(entry + 8)
    var category = load_byte(entry + 9)
    var count = min(TEXT_BYTES, load_byte(entry + 10) | (load_byte(entry + 11) << 8))
    var thread = Int(Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=entry + 12)[unsafe_offset=0])
    var tick = load(entry + 16)
    var nanoseconds = load(entry + 24)
    var milliseconds = nanoseconds // 1000000
    out.decimal(milliseconds // 1000)
    out.byte(46)
    out.padded(milliseconds % 1000, 3)
    out.text(" ")
    out.byte(level_letter(level))
    out.text(" ")
    write_category(out, category)
    if tick >= 0:
        out.text(" tick=")
        out.decimal(tick)
    out.text(" tid=")
    out.decimal(thread)
    out.text(" | ")
    out.bytes_at(entry + ENTRY_TEXT, count)
    out.byte(10)
    # The writer may have reused the entry while it was copied.
    if Int(entry_sequence(index)[unsafe_offset=0].load()) != sequence:
        out.length = start
        return False
    return True


def state_directory() -> String:
    """SD_STATE_DIR, then $XDG_STATE_HOME/sharddepth, then ~/.local/state/sharddepth."""
    var chosen = env_value("SD_STATE_DIR")
    if chosen.byte_length() > 0:
        return chosen
    var state_home = env_value("XDG_STATE_HOME")
    if state_home.byte_length() > 0:
        return state_home + "/sharddepth"
    return env_value("HOME") + "/.local/state/sharddepth"


def rotate_logs(directory: String, name: String):
    """sharddepth.log becomes sharddepth.1.log, and so on; the oldest goes."""
    for index in range(LOG_FILES_KEPT - 1, 0, -1):
        var older = directory + "/" + name + "." + String(index) + ".log"
        if file_size(older) >= 0:
            _ = rename_file(older, directory + "/" + name + "." + String(index + 1) + ".log")
    var current = directory + "/" + name + ".log"
    if file_size(current) >= 0:
        _ = rename_file(current, directory + "/" + name + ".1.log")


def start_log(name: String = "sharddepth", directory: String = "") -> Bool:
    """Map the anchor, open the log file (rotating the old ones) and apply SD_LOG.
    False if the file cannot be opened; logging then still reaches the ring."""
    if not install_anchor():
        print("LOG_UNAVAILABLE the safety anchor could not be mapped")
        return False
    var state = directory if directory.byte_length() > 0 else state_directory()
    set_text(T_STATE_DIR, state)
    var settings = env_value("SD_LOG")
    if settings.byte_length() > 0:
        _ = apply_log_settings(settings)
    var folder = state + "/logs"
    if not make_dirs(folder):
        print("LOG_FAILED cannot make " + folder)
        return False
    rotate_logs(folder, name)
    var path = folder + "/" + name + ".log"
    var owned = c_string(path)
    var fd = sys(SYS_OPENAT, AT_FDCWD, Int(owned.unsafe_ptr()), O_WRONLY_CREATE_APPEND, 0o644)
    _ = owned^
    if fd < 0:
        print("LOG_FAILED cannot open " + path)
        return False
    var previous = load(ANCHOR + A_LOG_FD)
    if previous >= 0:
        _ = sys(SYS_CLOSE, previous)
    store(ANCHOR + A_LOG_FD, fd)
    set_text(T_LOG_PATH, path)
    var header = RawText(ANCHOR + FLUSH_BUFFER, FLUSH_BYTES)
    header.text("--- log opened ")
    write_utc(header, wall_seconds(), False)
    header.text(" (times below are seconds since start) ---\n")
    _ = header.write_to(fd)
    return True


def log_path() -> String:
    return get_text(T_LOG_PATH) if anchor_ready() else String()


def flush_log() -> Int:
    """Write every entry not yet in the file. One thread at a time: the main loop.
    Returns how many entries were written."""
    if not anchor_ready():
        return 0
    var fd = load(ANCHOR + A_LOG_FD)
    var head = anchor_get(A_LOG_HEAD)
    var next = anchor_get(A_LOG_FLUSHED)
    if head - next > LOG_ENTRIES:
        _ = atomic_word(A_LOG_DROPPED)[unsafe_offset=0].fetch_add(Int64(head - LOG_ENTRIES - next))
        next = head - LOG_ENTRIES
    var written = 0
    var out = RawText(ANCHOR + FLUSH_BUFFER, FLUSH_BYTES)
    while next < head:
        if out.length > FLUSH_BYTES - ENTRY_BYTES - 64:
            if fd >= 0:
                _ = out.write_to(fd)
            out.length = 0
        var sequence = Int(entry_sequence(next)[unsafe_offset=0].load())
        if sequence == next * 2 + 1:
            break  # still being written: the next flush takes it
        if format_entry(out, next):
            written += 1
        else:
            _ = atomic_word(A_LOG_DROPPED)[unsafe_offset=0].fetch_add(1)
        next += 1
    if fd >= 0 and out.length > 0:
        _ = out.write_to(fd)
    anchor_set(A_LOG_FLUSHED, next)
    return written


def log_dropped() -> Int:
    return anchor_get(A_LOG_DROPPED) if anchor_ready() else 0


def log_tail_lines(count: Int) -> List[String]:
    """The newest entries as text lines, oldest first (for tests and tools)."""
    var lines = List[String]()
    if not anchor_ready():
        return lines^
    var head = anchor_get(A_LOG_HEAD)
    var first = max(0, head - min(count, LOG_ENTRIES))
    var scratch = List[UInt8](length=ENTRY_BYTES + 128, fill=0)
    for index in range(first, head):
        var out = RawText(Int(scratch.unsafe_ptr()), len(scratch))
        if format_entry(out, index):
            var bytes = List[UInt8]()
            for position in range(out.length - 1):
                bytes.append(scratch[position])
            lines.append(String(unsafe_from_utf8=bytes))
    _ = scratch^
    return lines^
