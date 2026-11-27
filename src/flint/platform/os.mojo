"""Native Linux operating-system services used by Flint Engine.

Only system calls and the C ABI exported by the host C runtime are touched here.
No third-party compatibility library is linked into the game.
"""
from std.ffi import external_call
from std.math import isfinite
from std.sys.info import CompilationTarget
from std.time import monotonic


def get_time() -> Float64:
    return Float64(monotonic()) / 1000000000.0


def unix_time_sec() -> Int:
    """Wall-clock seconds since the Unix epoch for persisted timestamps."""
    return Int(external_call["time", Int64](Int(0)))


def env_value(name: String) -> String:
    return env_now(name)


def env_now(name: String) -> String:
    """`env_value`, but as the environment is now (getenv(3)), changes made
    in this process with setenv included: for switches a test flips."""
    var owned = name
    var address = external_call["getenv", Int](owned.as_c_string_span())
    if address == 0:
        return ""
    var bytes = List[UInt8]()
    var at = address
    while True:
        var byte = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=at)[unsafe_offset=0]
        if byte == 0:
            break
        bytes.append(byte)
        at += 1
    return String(unsafe_from_utf8=bytes)


def env_flag(name: String) -> Bool:
    var text = env_value(name)
    return text.byte_length() > 0 and text != "0" and text != "false" and text != "FALSE"


@always_inline
def set_signal_action(signal: Int32, action: Int, previous: Int = 0) -> Int32:
    return external_call["sigaction", Int32](
        signal, Pointer[UInt8, MutAnyOrigin](unsafe_from_address=action),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=previous),
    )


def open_path_fd(var path: String, flags: Int32, mode: Int32 = 0) -> Int:
    # openat(AT_FDCWD) also coexists with stdlib's variadic declaration of open.
    return Int(external_call["openat", Int32](Int32(-100), path.as_c_string_span(), flags, mode))


def open_read(path: String) -> Int:
    return open_path_fd(path, Int32(0))


def open_write(path: String) -> Int:
    return open_path_fd(path, Int32(1 | 64 | 512), 438)


def append_bytes(path: String, mut data: List[UInt8]) -> Bool:
    """Appends to a file, creating it if needed (O_WRONLY | O_CREAT | O_APPEND)."""
    var fd = open_path_fd(path, Int32(1 | 64 | 1024), 438)
    if fd < 0:
        return False
    var offset = 0
    while offset < len(data):
        var count = Int(external_call["write", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(offset), Int64(len(data) - offset)))
        if count <= 0:
            close_fd(fd)
            return False
        offset += count
    close_fd(fd)
    return True


def make_dirs(path: String) -> Bool:
    if path.byte_length() == 0:
        return False
    for end in range(1, path.byte_length() + 1):
        if end < path.byte_length() and path.as_bytes()[end] != 47:
            continue
        var prefix = List[UInt8]()
        for index in range(end):
            prefix.append(path.as_bytes()[index])
        var owned = String(unsafe_from_utf8=prefix)
        if owned == "/":
            continue
        var result = external_call["mkdirat", Int32](Int32(-100), owned.as_c_string_span(), Int32(511))
        if result != 0:
            var error_pointer = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
            if Int(error_pointer[unsafe_offset=0]) != 17:  # EEXIST
                return False
    return True


def rename_file(source: String, destination: String) -> Bool:
    var old_path = source
    var new_path = destination
    return external_call["rename", Int32](old_path.as_c_string_span(), new_path.as_c_string_span()) == 0


comptime DURABLE_OK: Int = 0
comptime FAULT_NONE: Int = 0        # write_file_durable fault injection, for tests
comptime FAULT_SHORT_WRITE: Int = 1 # the temporary file is cut off and left behind, as a crash mid-write
comptime FAULT_BEFORE_RENAME: Int = 2  # written and synced, then "the power fails" before the rename


def remove_file(path: String) -> Bool:
    var owned = path
    return external_call["unlink", Int32](owned.as_c_string_span()) == 0


def fsync_path(path: String, directory: Bool) -> Bool:
    """fsync(2) on a file or (O_DIRECTORY) a directory, so its entries are on disk."""
    comptime O_DIRECTORY_FLAG: Int32 = 65536 if CompilationTarget.is_x86() else 16384
    var flags = O_DIRECTORY_FLAG if directory else Int32(0)  # O_RDONLY | O_DIRECTORY
    var fd = open_path_fd(path, flags)
    if fd < 0:
        return False
    var ok = external_call["fsync", Int32](Int32(fd)) == 0
    close_fd(fd)
    return ok


def parent_directory(path: String) -> String:
    var bytes = path.as_bytes()
    var cut = -1
    for index in range(len(bytes)):
        if bytes[index] == 47:
            cut = index
    if cut < 0:
        return "."
    if cut == 0:
        return "/"
    return String(path[byte=0:cut])


def write_synced(path: String, mut data: List[UInt8], fault: Int = FAULT_NONE, sync: Bool = True) -> Bool:
    """Writes a whole file and fsyncs it: its bytes are on disk on return.
    `fault` stops part way, for tests (FAULT_*); `sync` False skips the fsync
    (scratch files nobody needs after a power cut: a replay's saves)."""
    var fd = open_write(path)
    if fd < 0:
        return False
    var ok = _write_synced_fd(fd, data, fault, sync)
    if not ok and fault == FAULT_NONE:
        _ = remove_file(path)
    return ok


def _write_synced_fd(fd: Int, mut data: List[UInt8], fault: Int, sync: Bool) -> Bool:
    var end = len(data) // 2 if fault == FAULT_SHORT_WRITE else len(data)
    var offset = 0
    while offset < end:
        var count = Int(external_call["write", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(offset), Int64(end - offset)))
        if count < 0 and last_errno() == EINTR:
            continue
        if count <= 0:
            close_fd(fd)
            return False
        offset += count
    if fault == FAULT_SHORT_WRITE:
        close_fd(fd)  # left behind cut off, as a crash mid-write would
        return False
    var synced = not sync or external_call["fsync", Int32](Int32(fd)) == 0
    var closed = external_call["close", Int32](Int32(fd)) == 0
    return synced and closed and fault != FAULT_BEFORE_RENAME


def durable_rename(source: String, destination: String, sync: Bool = True) -> Bool:
    """rename(2), then fsync the directory: the rename is only durable once
    the directory is. Source and destination share a directory."""
    if not rename_file(source, destination):
        _ = remove_file(source)
        return False
    return not sync or fsync_path(parent_directory(destination), True)


def write_file_durable(
    path: String, mut data: List[UInt8], fault: Int = FAULT_NONE, sync: Bool = True
) -> Bool:
    """Publish a complete file by renaming a private staging file in the same directory."""
    var owned = path
    var stat_buf = List[UInt8](length=256, fill=0)
    var mode = Int32(0o666)
    var exists = external_call["stat", Int32](owned.as_c_string_span(), stat_buf.unsafe_ptr()) == 0
    if exists:
        comptime MODE_OFFSET: Int = 24 if CompilationTarget.is_x86() else 16
        var bits = Pointer[UInt32, ImmutAnyOrigin](
            unsafe_from_address=Int(stat_buf.unsafe_ptr()) + MODE_OFFSET
        )[unsafe_offset=0]
        mode = Int32(bits & 0o777)
        _ = len(stat_buf)
    var prefix = parent_directory(path) + "/.flint-write-" + String(Int(external_call["getpid", Int32]())) + "-"
    var attempt = 0
    while True:
        var stage = prefix + String(attempt) + ".tmp"
        var fd = open_path_fd(stage, Int32(1 | 64 | 128), mode)  # O_WRONLY | O_CREAT | O_EXCL
        if fd >= 0:
            if exists and external_call["fchmod", Int32](Int32(fd), UInt32(mode)) != 0:
                close_fd(fd)
                _ = remove_file(stage)
                return False
            if not _write_synced_fd(fd, data, fault, sync):
                _ = remove_file(stage)
                return False
            return durable_rename(stage, path, sync)
        if last_errno() != 17:  # EEXIST means another writer owns this staging file
            return False
        attempt += 1


def player_save_directory() -> String:
    """FLINT_SAVE_DIR, then $XDG_DATA_HOME/flint, then ~/.local/share/flint."""
    var chosen = env_value("FLINT_SAVE_DIR")
    if chosen.byte_length() > 0:
        return chosen
    var data_home = env_value("XDG_DATA_HOME")
    if data_home.byte_length() > 0:
        return data_home + "/flint"
    return env_value("HOME") + "/.local/share/flint"


def player_save_directory_for(game_id: String) -> String:
    """`player_save_directory()/<game_id>`: each game gets its own folder (games used to share one).
    The id is reduced to `[A-Za-z0-9._-]`; an empty or unusable id becomes `default`."""
    var clean = List[UInt8]()
    for b in game_id.as_bytes():
        if (b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 45 or b == 95 or (b == 46 and len(clean) > 0):
            clean.append(b)
    var name = String(unsafe_from_utf8=clean) if len(clean) > 0 else String("default")
    return player_save_directory() + "/" + name


def close_fd(fd: Int):
    if fd >= 0:
        _ = external_call["close", Int32](Int32(fd))


def run_process_redirected(
    args: List[String],
    cwd: String = "",
    log_path: String = "",
) -> Int32:
    """Forks and executes `args[0]` via `execv(3)`, optionally changing to `cwd`,
    redirecting stdin to `/dev/null` and stdout/stderr to `log_path` (fallback `/dev/null`),
    and waiting synchronously. Returns exit code (0..255), 128+sig on signal, or -1 on error."""
    if len(args) == 0 or args[0].byte_length() == 0:
        return Int32(-1)

    var arg_buf = List[UInt8]()
    var offsets = List[Int](capacity=len(args))
    for i in range(len(args)):
        offsets.append(len(arg_buf))
        var sb = args[i].as_bytes()
        for j in range(len(sb)):
            arg_buf.append(sb[j])
        arg_buf.append(0)

    var base_addr = Int(arg_buf.unsafe_ptr())
    var ptrs = List[Int](capacity=len(args) + 1)
    for i in range(len(args)):
        ptrs.append(base_addr + offsets[i])
    ptrs.append(0)

    var cwd_buf = List[UInt8]()
    if cwd.byte_length() > 0:
        var cb = cwd.as_bytes()
        for i in range(len(cb)):
            cwd_buf.append(cb[i])
        cwd_buf.append(0)

    var log_buf = List[UInt8]()
    if log_path.byte_length() > 0:
        var lb = log_path.as_bytes()
        for i in range(len(lb)):
            log_buf.append(lb[i])
        log_buf.append(0)

    var devnull_buf = List[UInt8](capacity=10)
    var dn = String("/dev/null")
    var dnb = dn.as_bytes()
    for i in range(len(dnb)):
        devnull_buf.append(dnb[i])
    devnull_buf.append(0)

    var status_buf = List[Int32](length=1, fill=0)
    var pid = external_call["fork", Int32]()
    if pid == Int32(0):
        if len(cwd_buf) > 0:
            if external_call["chdir", Int32](cwd_buf.unsafe_ptr()) != Int32(0):
                external_call["_exit", NoneType](Int32(126))
        var in_fd = external_call["openat", Int32](Int32(-100), devnull_buf.unsafe_ptr(), Int32(0), Int32(0))
        if in_fd >= Int32(0):
            _ = external_call["dup2", Int32](in_fd, Int32(0))
            if in_fd > Int32(2):
                _ = external_call["close", Int32](in_fd)
        var out_fd = Int32(-1)
        if len(log_buf) > 0:
            out_fd = external_call["openat", Int32](
                Int32(-100), log_buf.unsafe_ptr(), Int32(1 | 64 | 512), Int32(420)
            )
        if out_fd < Int32(0):
            out_fd = external_call["openat", Int32](Int32(-100), devnull_buf.unsafe_ptr(), Int32(1), Int32(0))
        if out_fd >= Int32(0):
            _ = external_call["dup2", Int32](out_fd, Int32(1))
            _ = external_call["dup2", Int32](out_fd, Int32(2))
            if out_fd > Int32(2):
                _ = external_call["close", Int32](out_fd)
        _ = external_call["close_range", Int32](UInt32(3), UInt32(0xFFFFFFFF), UInt32(0))
        _ = external_call["execv", Int32](arg_buf.unsafe_ptr(), ptrs.unsafe_ptr())
        external_call["_exit", NoneType](Int32(127))

    if pid < Int32(0):
        _ = len(arg_buf)
        _ = len(offsets)
        _ = len(ptrs)
        _ = len(cwd_buf)
        _ = len(log_buf)
        _ = len(devnull_buf)
        _ = len(status_buf)
        return Int32(-1)

    while True:
        var w = external_call["waitpid", Int32](pid, status_buf.unsafe_ptr(), Int32(0))
        if w == pid:
            break
        if w < Int32(0) and last_errno() == EINTR:
            continue
        _ = len(arg_buf)
        _ = len(offsets)
        _ = len(ptrs)
        _ = len(cwd_buf)
        _ = len(log_buf)
        _ = len(devnull_buf)
        _ = len(status_buf)
        return Int32(-1)

    var st = status_buf[0]
    _ = len(arg_buf)
    _ = len(offsets)
    _ = len(ptrs)
    _ = len(cwd_buf)
    _ = len(log_buf)
    _ = len(devnull_buf)
    _ = len(status_buf)
    var sig = st & Int32(0x7F)
    if sig == Int32(0):
        return (st >> Int32(8)) & Int32(0xFF)
    return Int32(128) + sig


def _fstat_regular_size(fd: Int) -> Int:
    """Returns regular-file byte size via `fstat(2)`, or -1 if non-regular / invalid."""
    if fd < 0:
        return -1
    comptime STAT_MODE_OFF: Int = 24 if CompilationTarget.is_x86() else 16
    comptime S_IFMT: UInt32 = 0o170000
    comptime S_IFREG: UInt32 = 0o100000
    var buf = List[UInt8](length=256, fill=0)
    if external_call["fstat", Int32](Int32(fd), buf.unsafe_ptr()) != 0:
        return -1
    var mode = Pointer[UInt32, ImmutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr()) + STAT_MODE_OFF
    )[unsafe_offset=0]
    if (mode & S_IFMT) != S_IFREG:
        _ = len(buf)
        return -1
    var words = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))
    var size = Int(words[unsafe_offset=6])  # st_size at byte 48
    _ = len(buf)
    if size < 0 or size >= Int(0x7FFFFFFFFFFFFFFF):
        return -1
    return size


def file_size(path: String) -> Int:
    var fd = open_read(path)
    if fd < 0:
        return -1
    var size = _fstat_regular_size(fd)
    close_fd(fd)
    return size


comptime EINTR: Int32 = 4


def is_symlink(path: String) -> Bool:
    """Returns True if `path` is a symbolic link (`lstat` `S_IFMT == S_IFLNK`)."""
    if path.byte_length() == 0:
        return False
    comptime STAT_MODE_OFF: Int = 24 if CompilationTarget.is_x86() else 16
    comptime S_IFMT: UInt32 = 0o170000
    comptime S_IFLNK: UInt32 = 0o120000
    var owned = path
    var buf = List[UInt8](length=256, fill=0)
    if external_call["lstat", Int32](owned.as_c_string_span(), buf.unsafe_ptr()) != 0:
        return False
    var mode = Pointer[UInt32, ImmutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr()) + STAT_MODE_OFF
    )[unsafe_offset=0]
    _ = len(buf)
    return (mode & S_IFMT) == S_IFLNK


def is_directory(path: String, follow_symlinks: Bool = False) -> Bool:
    """Returns True if `path` is a directory (`S_IFMT == S_IFDIR`).
    By default uses `lstat` (symlink-safe: returns False for symlinks to avoid traversal cycles)."""
    if path.byte_length() == 0:
        return False
    var clean = path
    while clean.byte_length() > 1 and clean.as_bytes()[clean.byte_length() - 1] == 47:
        var trimmed = String(clean[byte=0 : clean.byte_length() - 1])
        clean = trimmed^
    comptime STAT_MODE_OFF: Int = 24 if CompilationTarget.is_x86() else 16
    comptime S_IFMT: UInt32 = 0o170000
    comptime S_IFDIR: UInt32 = 0o040000
    var buf = List[UInt8](length=256, fill=0)
    var rc = (
        external_call["stat", Int32](clean.as_c_string_span(), buf.unsafe_ptr())
        if follow_symlinks
        else external_call["lstat", Int32](clean.as_c_string_span(), buf.unsafe_ptr())
    )
    if rc != 0:
        return False
    var mode = Pointer[UInt32, ImmutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr()) + STAT_MODE_OFF
    )[unsafe_offset=0]
    _ = len(buf)
    return (mode & S_IFMT) == S_IFDIR


def real_path(path: String) -> Tuple[Bool, String]:
    """Resolves `path` via `realpath(3)` (requires `path` to exist). Returns `(ok, resolved)`."""
    if path.byte_length() == 0:
        return (False, String())
    var owned = path
    var resolved = List[UInt8](length=4096, fill=0)
    var got = external_call["realpath", Int](owned.as_c_string_span(), resolved.unsafe_ptr())
    if got == 0:
        return (False, String())
    var out = List[UInt8]()
    var p = resolved.unsafe_ptr()
    for i in range(4096):
        var b = p[unsafe_offset=i]
        if b == 0:
            break
        out.append(b)
    _ = len(resolved)
    return (True, String(unsafe_from_utf8=out))


def file_stamp(path: String) -> Tuple[Int, Int]:
    """A file's size and modification time in nanoseconds, from stat(2), or
    (-1, 0) if it is missing. The layout is x86-64 Linux's struct stat."""
    var owned = path
    var buffer = List[UInt8](length=256, fill=0)
    if external_call["stat", Int32](owned.as_c_string_span(), buffer.unsafe_ptr()) != 0:
        return (-1, 0)
    var words = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=Int(buffer.unsafe_ptr()))
    var size = Int(words[unsafe_offset=6])  # st_size at byte 48
    var mtime = Int(words[unsafe_offset=11]) * 1000000000 + Int(words[unsafe_offset=12])  # st_mtim at 88
    _ = buffer^
    return (size, mtime)


def list_directory(path: String) -> List[String]:
    """The names in a directory (not . or ..), sorted bytewise so every run
    sees the same order; empty if it cannot be read. readdir(3) on x86-64
    Linux: the name starts 19 bytes into each entry."""
    var names = List[String]()
    var owned = path
    var directory = external_call["opendir", Int](owned.as_c_string_span())
    if directory == 0:
        return names^
    var dir_ptr = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=directory)
    while True:
        var entry = external_call["readdir", Int](dir_ptr)
        if entry == 0:
            break
        var bytes = List[UInt8]()
        var at = entry + 19
        while True:
            var byte = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=at)[unsafe_offset=0]
            if byte == 0:
                break
            bytes.append(byte)
            at += 1
        var name = String(unsafe_from_utf8=bytes)
        if name != "." and name != "..":
            names.append(name)
    _ = external_call["closedir", Int32](dir_ptr)
    for i in range(1, len(names)):
        var j = i
        while j > 0 and names[j] < names[j - 1]:
            var t = names[j]
            names[j] = names[j - 1]
            names[j - 1] = t
            j -= 1
    return names^


def read_at(path: String, offset: Int, count: Int) -> List[UInt8]:
    """`count` bytes from `offset` with pread(2), or an empty list if the file
    is shorter or unreadable."""
    if count <= 0 or offset < 0:
        return List[UInt8]()
    var fd = open_read(path)
    if fd < 0:
        return List[UInt8]()
    var data = List[UInt8](capacity=count)
    data.resize(unsafe_uninit_length=count)
    var done = 0
    while done < count:
        var got = Int(external_call["pread", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(done), Int64(count - done), Int64(offset + done)))
        if got < 0 and last_errno() == EINTR:
            continue
        if got <= 0:
            close_fd(fd)
            return List[UInt8]()
        done += got
    close_fd(fd)
    return data^


def last_errno() -> Int32:
    """This thread's errno."""
    return external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()[unsafe_offset=0]


def read_bytes(path: String) -> List[UInt8]:
    var fd = open_read(path)
    if fd < 0:
        return List[UInt8]()
    var size = _fstat_regular_size(fd)
    if size <= 0:
        close_fd(fd)
        return List[UInt8]()
    var data = List[UInt8](length=size, fill=0)
    var offset = 0
    while offset < size:
        var count = Int(external_call["read", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(offset), Int64(size - offset)))
        if count < 0 and last_errno() == EINTR:
            continue  # a signal interrupted the read; the file itself is fine
        if count <= 0:
            close_fd(fd)
            return List[UInt8]()
        offset += count
    close_fd(fd)
    return data^


def read_stream(path: String, capacity: Int = 1048576) -> List[UInt8]:
    var fd = open_read(path)
    if fd < 0:
        return List[UInt8]()
    var data = List[UInt8]()
    var buffer = List[UInt8](length=4096, fill=0)
    while len(data) < capacity:
        var wanted = min(len(buffer), capacity - len(data))
        var count = Int(external_call["read", Int64](Int32(fd), buffer.unsafe_ptr(), Int64(wanted)))
        if count <= 0:
            break
        for index in range(count):
            data.append(buffer[index])
    close_fd(fd)
    return data^


def write_bytes(path: String, mut data: List[UInt8]) -> Bool:
    var fd = open_write(path)
    if fd < 0:
        return False
    var offset = 0
    while offset < len(data):
        var count = Int(external_call["write", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(offset), Int64(len(data) - offset)))
        if count <= 0:
            close_fd(fd)
            return False
        offset += count
    close_fd(fd)
    return True


def write_save(mut path: String, mut data: List[Int32]) -> Bool:
    var bytes = List[UInt8](length=len(data) * 4, fill=0)
    for index in range(len(data)):
        var value = UInt32(data[index])
        bytes[index * 4] = UInt8(value & 255)
        bytes[index * 4 + 1] = UInt8((value >> 8) & 255)
        bytes[index * 4 + 2] = UInt8((value >> 16) & 255)
        bytes[index * 4 + 3] = UInt8((value >> 24) & 255)
    return write_bytes(path, bytes)


def read_save_count(mut path: String) -> Int:
    var size = file_size(path)
    if size < 0:
        return 0
    return size // 4


def read_save(mut path: String, count: Int) -> List[Int32]:
    var bytes = read_bytes(path)
    if count < 0 or len(bytes) < count * 4:
        return List[Int32]()
    var data = List[Int32](length=count, fill=0)
    for index in range(count):
        var offset = index * 4
        var value = UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8) | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
        data[index] = Int32(value)
    return data^


def process_status_bytes(field: String) -> UInt64:
    """A "Name:  1234 kB" line of /proc/self/status, in bytes; 0 if missing."""
    var data = read_stream("/proc/self/status")
    var name = field.as_bytes()
    var length = len(name)
    var index = 0
    while index + length < len(data):
        var matches = index == 0 or data[index - 1] == 10
        if matches:
            for offset in range(length):
                if data[index + offset] != name[offset]:
                    matches = False
                    break
        if matches and data[index + length] == 58:
            index += length + 1
            var value = UInt64(0)
            while index < len(data) and (data[index] == 32 or data[index] == 9):
                index += 1
            while index < len(data) and data[index] >= 48 and data[index] <= 57:
                value = value * 10 + UInt64(data[index] - 48)
                index += 1
            return value * 1024
        index += 1
    return 0


def process_rss_bytes() -> UInt64:
    return process_status_bytes("VmRSS")


def process_peak_rss_bytes() -> UInt64:
    """The most resident memory this process has had (VmHWM)."""
    return process_status_bytes("VmHWM")


# ==============================================================================
# D3: Non-Blocking POSIX IPv4 UDP Datagram Sockets & Deterministic Lockstep Peer
# ==============================================================================

comptime _AF_INET: Int32 = 2
comptime _SOCK_DGRAM_NONBLOCK: Int32 = 2 | 2048  # SOCK_DGRAM (2) | SOCK_NONBLOCK (04000)


def _fill_sockaddr_in(mut addr: List[UInt8], ip_a: Int, ip_b: Int, ip_c: Int, ip_d: Int, port: Int):
    for i in range(16):
        addr[i] = 0
    addr[0] = UInt8(_AF_INET & 255)
    addr[1] = UInt8((_AF_INET >> 8) & 255)
    addr[2] = UInt8((port >> 8) & 255)
    addr[3] = UInt8(port & 255)
    addr[4] = UInt8(ip_a & 255)
    addr[5] = UInt8(ip_b & 255)
    addr[6] = UInt8(ip_c & 255)
    addr[7] = UInt8(ip_d & 255)


def udp_open_nonblocking(bind_port: Int = 0, loopback_only: Bool = False) -> Tuple[Int, Int]:
    """Opens a non-blocking IPv4 UDP socket bound to `bind_port` (`0` = ephemeral).
    Returns `(fd, bound_port)`, or `(-1, 0)` on failure."""
    var fd = Int(external_call["socket", Int32](_AF_INET, _SOCK_DGRAM_NONBLOCK, Int32(0)))
    if fd < 0:
        return (-1, 0)
    var opt_buf = List[Int32](length=1, fill=1)
    _ = external_call["setsockopt", Int32](Int32(fd), Int32(1), Int32(2), opt_buf.unsafe_ptr(), UInt32(4))
    _ = len(opt_buf)
    var addr = List[UInt8](length=16, fill=0)
    if loopback_only:
        _fill_sockaddr_in(addr, 127, 0, 0, 1, bind_port)
    else:
        _fill_sockaddr_in(addr, 0, 0, 0, 0, bind_port)
    var rc = external_call["bind", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(16))
    _ = len(addr)
    if rc < 0:
        close_fd(fd)
        return (-1, 0)
    var name_addr = List[UInt8](length=16, fill=0)
    var addr_len_buf = List[UInt32](length=1, fill=16)
    var grc = external_call["getsockname", Int32](Int32(fd), name_addr.unsafe_ptr(), addr_len_buf.unsafe_ptr())
    _ = len(addr_len_buf)
    var actual_port = bind_port
    if grc == 0:
        actual_port = (Int(name_addr[2]) << 8) | Int(name_addr[3])
    _ = len(name_addr)
    return (fd, actual_port)


def udp_send_ipv4(fd: Int, ip_a: Int, ip_b: Int, ip_c: Int, ip_d: Int, port: Int, payload: List[UInt8]) -> Int:
    """Sends a single non-blocking UDP datagram to `ip_a.ip_b.ip_c.ip_d:port`."""
    if fd < 0 or port <= 0 or len(payload) == 0:
        return -1
    var addr = List[UInt8](length=16, fill=0)
    _fill_sockaddr_in(addr, ip_a, ip_b, ip_c, ip_d, port)
    var n_bytes = len(payload)
    var sent = Int(
        external_call["sendto", Int64](
            Int32(fd),
            payload.unsafe_ptr(),
            Int64(n_bytes),
            Int32(0x4000),  # MSG_NOSIGNAL
            addr.unsafe_ptr(),
            UInt32(16),
        )
    )
    _ = len(payload)
    _ = len(addr)
    return sent


def udp_recv_nonblocking(fd: Int, max_bytes: Int = 512) -> List[UInt8]:
    """Polls `fd` for one incoming UDP datagram without blocking; returns empty list on `EAGAIN`."""
    if fd < 0 or max_bytes <= 0:
        return List[UInt8]()
    var buf = List[UInt8](length=max_bytes, fill=0)
    var src_addr = List[UInt8](length=16, fill=0)
    var addr_len_buf = List[UInt32](length=1, fill=16)
    var got = Int(
        external_call["recvfrom", Int64](
            Int32(fd),
            buf.unsafe_ptr(),
            Int64(max_bytes),
            Int32(0x40),  # MSG_DONTWAIT
            src_addr.unsafe_ptr(),
            addr_len_buf.unsafe_ptr(),
        )
    )
    _ = len(src_addr)
    _ = len(addr_len_buf)
    if got <= 0:
        _ = len(buf)
        return List[UInt8]()
    var out = List[UInt8](capacity=got)
    for i in range(got):
        out.append(buf[i])
    _ = len(buf)
    return out^


comptime LOCKSTEP_MAGIC: UInt32 = 0x50534B4C  # "LKSP"


struct LockstepFrame(Copyable, Movable):
    var tick: Int
    var state_hash: UInt64
    var input_bytes: List[UInt8]

    def __init__(out self, tick: Int = -1, state_hash: UInt64 = 0):
        self.tick = tick
        self.state_hash = state_hash
        self.input_bytes = List[UInt8]()


struct LockstepPeer(Movable):
    """Deterministic 2-player UDP lockstep transport exchanging `(tick, FNV-64 state_hash, input_bytes)`
    frames with automatic per-tick state-hash desynchronization verification."""
    var fd: Int
    var local_port: Int
    var peer_ip_a: Int
    var peer_ip_b: Int
    var peer_ip_c: Int
    var peer_ip_d: Int
    var peer_port: Int
    var packets_sent: Int
    var packets_received: Int
    var desync_detected: Bool
    var desync_tick: Int
    var local_ticks: List[Int]
    var local_hashes: List[UInt64]
    var remote_frames: List[LockstepFrame]

    def __init__(out self):
        self.fd = -1
        self.local_port = 0
        self.peer_ip_a = 127
        self.peer_ip_b = 0
        self.peer_ip_c = 0
        self.peer_ip_d = 1
        self.peer_port = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.desync_detected = False
        self.desync_tick = -1
        self.local_ticks = List[Int]()
        self.local_hashes = List[UInt64]()
        self.remote_frames = List[LockstepFrame]()

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.local_port = existing.local_port
        self.peer_ip_a = existing.peer_ip_a
        self.peer_ip_b = existing.peer_ip_b
        self.peer_ip_c = existing.peer_ip_c
        self.peer_ip_d = existing.peer_ip_d
        self.peer_port = existing.peer_port
        self.packets_sent = existing.packets_sent
        self.packets_received = existing.packets_received
        self.desync_detected = existing.desync_detected
        self.desync_tick = existing.desync_tick
        self.local_ticks = existing.local_ticks^
        self.local_hashes = existing.local_hashes^
        self.remote_frames = existing.remote_frames^
        existing.fd = -1
        existing.local_port = 0

    def __deinit__(deinit self):
        self.close()

    def open(mut self, bind_port: Int = 0, loopback_only: Bool = False) -> Bool:
        self.close()
        var res = udp_open_nonblocking(bind_port, loopback_only)
        self.fd = res[0]
        self.local_port = res[1]
        return self.fd >= 0

    def set_peer(mut self, ip_a: Int, ip_b: Int, ip_c: Int, ip_d: Int, port: Int):
        self.peer_ip_a = ip_a
        self.peer_ip_b = ip_b
        self.peer_ip_c = ip_c
        self.peer_ip_d = ip_d
        self.peer_port = port

    def close(mut self):
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
            self.local_port = 0

    def _verify_tick_hash(mut self, tick: Int, local_h: UInt64, remote_h: UInt64):
        if local_h != 0 and remote_h != 0 and local_h != remote_h:
            self.desync_detected = True
            if self.desync_tick < 0:
                self.desync_tick = tick

    def record_local_hash(mut self, tick: Int, state_hash: UInt64):
        for i in range(len(self.local_ticks)):
            if self.local_ticks[i] == tick:
                self.local_hashes[i] = state_hash
                for r in range(len(self.remote_frames)):
                    if self.remote_frames[r].tick == tick:
                        self._verify_tick_hash(tick, state_hash, self.remote_frames[r].state_hash)
                return
        if len(self.local_ticks) >= 256:
            _ = self.local_ticks.pop(0)
            _ = self.local_hashes.pop(0)
        self.local_ticks.append(tick)
        self.local_hashes.append(state_hash)
        for r in range(len(self.remote_frames)):
            if self.remote_frames[r].tick == tick:
                self._verify_tick_hash(tick, state_hash, self.remote_frames[r].state_hash)

    def send_tick(mut self, tick: Int, state_hash: UInt64, input_bytes: List[UInt8]) -> Bool:
        if self.fd < 0 or self.peer_port <= 0:
            return False
        self.record_local_hash(tick, state_hash)
        var pkt = List[UInt8](capacity=18 + len(input_bytes))
        var m = LOCKSTEP_MAGIC
        for s in range(4):
            pkt.append(UInt8((m >> UInt32(s * 8)) & 255))
        var t32 = UInt32(tick)
        for s in range(4):
            pkt.append(UInt8((t32 >> UInt32(s * 8)) & 255))
        for s in range(8):
            pkt.append(UInt8((state_hash >> UInt64(s * 8)) & 255))
        var ilen = min(240, len(input_bytes))
        pkt.append(UInt8(ilen & 255))
        pkt.append(UInt8((ilen >> 8) & 255))
        for i in range(ilen):
            pkt.append(input_bytes[i])
        var sent = udp_send_ipv4(
            self.fd, self.peer_ip_a, self.peer_ip_b, self.peer_ip_c, self.peer_ip_d, self.peer_port, pkt
        )
        if sent == len(pkt):
            self.packets_sent += 1
            return True
        return False

    def poll_incoming(mut self) -> Int:
        if self.fd < 0:
            return 0
        var count = 0
        for _ in range(32):
            var raw = udp_recv_nonblocking(self.fd, 512)
            if len(raw) < 18:
                break
            var magic = (
                UInt32(raw[0])
                | (UInt32(raw[1]) << 8)
                | (UInt32(raw[2]) << 16)
                | (UInt32(raw[3]) << 24)
            )
            if magic != LOCKSTEP_MAGIC:
                continue
            var tick = Int(
                UInt32(raw[4])
                | (UInt32(raw[5]) << 8)
                | (UInt32(raw[6]) << 16)
                | (UInt32(raw[7]) << 24)
            )
            var r_hash = UInt64(0)
            for s in range(8):
                r_hash |= UInt64(raw[8 + s]) << UInt64(s * 8)
            var ilen = Int(raw[16]) | (Int(raw[17]) << 8)
            if 18 + ilen > len(raw):
                continue
            var frame = LockstepFrame(tick, r_hash)
            frame.input_bytes.reserve(ilen)
            for i in range(ilen):
                frame.input_bytes.append(raw[18 + i])
            self.packets_received += 1
            count += 1
            var replaced = False
            for r in range(len(self.remote_frames)):
                if self.remote_frames[r].tick == tick:
                    self.remote_frames[r] = frame.copy()
                    replaced = True
                    break
            if not replaced:
                if len(self.remote_frames) >= 256:
                    _ = self.remote_frames.pop(0)
                self.remote_frames.append(frame^)
            for l_i in range(len(self.local_ticks)):
                if self.local_ticks[l_i] == tick:
                    self._verify_tick_hash(tick, self.local_hashes[l_i], r_hash)
                    break
        return count

    def try_consume_tick(mut self, tick: Int) -> Tuple[Bool, List[UInt8]]:
        _ = self.poll_incoming()
        for r in range(len(self.remote_frames)):
            if self.remote_frames[r].tick == tick:
                return (True, self.remote_frames[r].input_bytes.copy())
        return (False, List[UInt8]())


# ==============================================================================
# V9 C3: Zero-Dependency Linux Joystick / HOTAS Polling (/dev/input/js0..js3)
# ==============================================================================

comptime JS_EVENT_BUTTON: UInt8 = 0x01
comptime JS_EVENT_AXIS: UInt8 = 0x02
comptime JS_EVENT_INIT: UInt8 = 0x80
comptime GAMEPAD_MAX_AXES: Int = 16
comptime GAMEPAD_MAX_BUTTONS: Int = 32


struct LinuxGamepad(Movable):
    """Legacy joystick packet parser & deadzone/curve shaper; device polling retired in favor of `flint.input.evdev.EvdevHub`."""
    var fd: Int
    var device_index: Int
    var connected: Bool
    var deadzone: Float32
    var curve_power: Float32
    var axes: List[Float32]
    var buttons: List[Bool]
    var prev_buttons: List[Bool]
    var events_processed: Int

    def __init__(out self, deadzone: Float32 = 0.12, curve_power: Float32 = 1.55):
        self.fd = -1
        self.device_index = -1
        self.connected = False
        var safe_dz = deadzone if isfinite(deadzone) else Float32(0.12)
        var safe_cp = curve_power if isfinite(curve_power) else Float32(1.55)
        self.deadzone = max(Float32(0.0), min(Float32(0.65), safe_dz))
        self.curve_power = max(Float32(1.0), min(Float32(3.0), safe_cp))
        self.axes = List[Float32](length=GAMEPAD_MAX_AXES, fill=0.0)
        self.buttons = List[Bool](length=GAMEPAD_MAX_BUTTONS, fill=False)
        self.prev_buttons = List[Bool](length=GAMEPAD_MAX_BUTTONS, fill=False)
        self.events_processed = 0

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.device_index = existing.device_index
        self.connected = existing.connected
        self.deadzone = existing.deadzone
        self.curve_power = existing.curve_power
        self.axes = existing.axes^
        self.buttons = existing.buttons^
        self.prev_buttons = existing.prev_buttons^
        self.events_processed = existing.events_processed
        existing.fd = -1
        existing.device_index = -1
        existing.connected = False

    def __deinit__(deinit self):
        self.close()

    def open_first_available(mut self) -> Bool:
        """Legacy joystick device opening is retired; use `EvdevHub` (`/dev/input/event*`)."""
        self.close()
        return False

    def close(mut self):
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
        self.device_index = -1
        self.connected = False

    def begin_frame(mut self):
        for b in range(GAMEPAD_MAX_BUTTONS):
            self.prev_buttons[b] = self.buttons[b]

    def apply_js_event(mut self, ev_type: UInt8, number: UInt8, raw_value: Int16):
        var kind = ev_type & ~JS_EVENT_INIT
        var idx = Int(number)
        if kind == JS_EVENT_AXIS and idx >= 0 and idx < GAMEPAD_MAX_AXES:
            var norm = Float32(raw_value) / Float32(32767.0)
            self.axes[idx] = max(Float32(-1.0), min(Float32(1.0), norm))
            self.events_processed += 1
        elif kind == JS_EVENT_BUTTON and idx >= 0 and idx < GAMEPAD_MAX_BUTTONS:
            self.buttons[idx] = raw_value != 0
            self.events_processed += 1

    def parse_js_packet(mut self, pkt8: List[UInt8]) -> Bool:
        if len(pkt8) < 8:
            return False
        var raw_u16 = UInt16(pkt8[4]) | (UInt16(pkt8[5]) << 8)
        var raw_i16 = Int16(raw_u16)
        self.apply_js_event(pkt8[6], pkt8[7], raw_i16)
        return True

    def poll(mut self) -> Int:
        self.begin_frame()
        if self.fd < 0:
            return 0
        var buf = List[UInt8](length=64, fill=0)
        var got_events = 0
        for _ in range(16):
            var n = Int(external_call["read", Int64](Int32(self.fd), buf.unsafe_ptr(), Int64(64)))
            if n < 8:
                if n < 0:
                    var err = last_errno()
                    if err != Int32(11) and err != EINTR:  # EAGAIN = 11
                        self.close()
                break
            var off = 0
            while off + 8 <= n:
                var raw_u16 = UInt16(buf[off + 4]) | (UInt16(buf[off + 5]) << 8)
                self.apply_js_event(buf[off + 6], buf[off + 7], Int16(raw_u16))
                got_events += 1
                off += 8
        _ = len(buf)
        return got_events

    def axis_curved(self, axis_idx: Int) -> Float32:
        if axis_idx < 0 or axis_idx >= GAMEPAD_MAX_AXES:
            return 0.0
        var v = self.axes[axis_idx]
        var av = abs(v)
        if av <= self.deadzone:
            return 0.0
        var t = min(Float32(1.0), (av - self.deadzone) / max(Float32(0.01), Float32(1.0) - self.deadzone))
        # Fast smooth polynomial blend between linear (p=1) and quadratic/cubic stick precision
        var t2 = t * t
        var blend = max(Float32(0.0), min(Float32(1.0), (self.curve_power - Float32(1.0)) * Float32(0.5)))
        var shaped = t * (Float32(1.0) - blend) + (t2 * (Float32(0.65) + Float32(0.35) * t)) * blend
        return shaped if v >= Float32(0.0) else -shaped

    def button_down(self, btn_idx: Int) -> Bool:
        if btn_idx < 0 or btn_idx >= GAMEPAD_MAX_BUTTONS:
            return False
        return self.buttons[btn_idx]

    def button_pressed(self, btn_idx: Int) -> Bool:
        if btn_idx < 0 or btn_idx >= GAMEPAD_MAX_BUTTONS:
            return False
        return self.buttons[btn_idx] and not self.prev_buttons[btn_idx]


# ==============================================================================
# V9 D1: GGPO-Style 4-Player Deterministic Rollback Netcode (`RollbackPeer`)
# ==============================================================================

comptime ROLLBACK_MAGIC: UInt32 = 0x50424C52  # "RLBP"
comptime MAX_ROLLBACK_PLAYERS: Int = 4
comptime MAX_ROLLBACK_WINDOW: Int = 128


struct RollbackEndpoint(Copyable, Movable, ImplicitlyCopyable):
    var active: Bool
    var ip_a: Int
    var ip_b: Int
    var ip_c: Int
    var ip_d: Int
    var port: Int

    def __init__(out self, ip_a: Int = 127, ip_b: Int = 0, ip_c: Int = 0, ip_d: Int = 1, port: Int = 0, active: Bool = False):
        self.active = active
        self.ip_a = ip_a
        self.ip_b = ip_b
        self.ip_c = ip_c
        self.ip_d = ip_d
        self.port = port


struct RollbackSnapshot(Copyable, Movable):
    var tick: Int
    var state_hash: UInt64
    var state_bytes: List[UInt8]

    def __init__(out self, tick: Int = -1, state_hash: UInt64 = 0):
        self.tick = tick
        self.state_hash = state_hash
        self.state_bytes = List[UInt8]()


struct RollbackInputSlot(Copyable, Movable):
    var tick: Int
    var confirmed: Bool
    var state_hash: UInt64
    var data: List[UInt8]

    def __init__(out self, tick: Int = -1, confirmed: Bool = False, state_hash: UInt64 = 0):
        self.tick = tick
        self.confirmed = confirmed
        self.state_hash = state_hash
        self.data = List[UInt8]()


struct RollbackPeer(Movable):
    """GGPO-style up-to-4-player non-blocking UDP rollback peer:
    - Zero-input-delay local execution using last-confirmed remote input prediction
    - Automatic misprediction detection (`rollback_from_tick >= 0`) when late UDP packets arrive
    - Ring-buffer authoritative state snapshots (`save_snapshot` / `restore_snapshot_at_or_before`)
    - 4-way FNV-64 state-hash desync verification on confirmed ticks."""
    var fd: Int
    var local_port: Int
    var local_slot: Int
    var player_count: Int
    var endpoints: List[RollbackEndpoint]
    var history: List[List[RollbackInputSlot]]
    var last_confirmed_input: List[List[UInt8]]
    var snapshots: List[RollbackSnapshot]
    var packets_sent: Int
    var packets_received: Int
    var rollbacks_triggered: Int
    var rollback_from_tick: Int
    var desync_detected: Bool
    var desync_tick: Int

    def __init__(out self, local_slot: Int = 0, player_count: Int = 2, default_input_len: Int = 12):
        self.fd = -1
        self.local_port = 0
        self.local_slot = max(0, min(MAX_ROLLBACK_PLAYERS - 1, local_slot))
        self.player_count = max(1, min(MAX_ROLLBACK_PLAYERS, player_count))
        self.endpoints = List[RollbackEndpoint]()
        self.history = List[List[RollbackInputSlot]]()
        self.last_confirmed_input = List[List[UInt8]]()
        for _ in range(MAX_ROLLBACK_PLAYERS):
            self.endpoints.append(RollbackEndpoint())
            self.history.append(List[RollbackInputSlot]())
            self.last_confirmed_input.append(List[UInt8](length=max(1, default_input_len), fill=0))
        self.snapshots = List[RollbackSnapshot]()
        self.packets_sent = 0
        self.packets_received = 0
        self.rollbacks_triggered = 0
        self.rollback_from_tick = -1
        self.desync_detected = False
        self.desync_tick = -1

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.local_port = existing.local_port
        self.local_slot = existing.local_slot
        self.player_count = existing.player_count
        self.endpoints = existing.endpoints^
        self.history = existing.history^
        self.last_confirmed_input = existing.last_confirmed_input^
        self.snapshots = existing.snapshots^
        self.packets_sent = existing.packets_sent
        self.packets_received = existing.packets_received
        self.rollbacks_triggered = existing.rollbacks_triggered
        self.rollback_from_tick = existing.rollback_from_tick
        self.desync_detected = existing.desync_detected
        self.desync_tick = existing.desync_tick
        existing.fd = -1
        existing.local_port = 0

    def __deinit__(deinit self):
        self.close()

    def open(mut self, bind_port: Int = 0, loopback_only: Bool = False) -> Bool:
        self.close()
        var res = udp_open_nonblocking(bind_port, loopback_only)
        self.fd = res[0]
        self.local_port = res[1]
        return self.fd >= 0

    def close(mut self):
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
            self.local_port = 0

    def set_remote_endpoint(mut self, slot: Int, ip_a: Int, ip_b: Int, ip_c: Int, ip_d: Int, port: Int):
        if slot < 0 or slot >= MAX_ROLLBACK_PLAYERS or slot == self.local_slot:
            return
        self.endpoints[slot] = RollbackEndpoint(ip_a, ip_b, ip_c, ip_d, port, port > 0)
        if slot >= self.player_count:
            self.player_count = slot + 1

    @staticmethod
    def _inputs_equal(a: List[UInt8], b: List[UInt8]) -> Bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    def _find_slot_entry(self, player: Int, tick: Int) -> Int:
        for i in range(len(self.history[player])):
            if self.history[player][i].tick == tick:
                return i
        return -1

    def _upsert_slot_entry(mut self, player: Int, tick: Int, confirmed: Bool, state_hash: UInt64, data: List[UInt8]):
        var idx = self._find_slot_entry(player, tick)
        if idx >= 0:
            self.history[player][idx].confirmed = confirmed
            if state_hash != 0:
                self.history[player][idx].state_hash = state_hash
            self.history[player][idx].data = data.copy()
            return
        if len(self.history[player]) >= MAX_ROLLBACK_WINDOW * 2:
            _ = self.history[player].pop(0)
        var entry = RollbackInputSlot(tick, confirmed, state_hash)
        entry.data = data.copy()
        self.history[player].append(entry^)

    def _verify_confirmed_hashes(mut self, tick: Int):
        var local_idx = self._find_slot_entry(self.local_slot, tick)
        if local_idx < 0:
            return
        var lh = self.history[self.local_slot][local_idx].state_hash
        if lh == 0:
            return
        for p in range(self.player_count):
            if p == self.local_slot:
                continue
            var r_idx = self._find_slot_entry(p, tick)
            if r_idx >= 0 and self.history[p][r_idx].confirmed:
                var rh = self.history[p][r_idx].state_hash
                if rh != 0 and rh != lh:
                    self.desync_detected = True
                    if self.desync_tick < 0 or tick < self.desync_tick:
                        self.desync_tick = tick

    def save_snapshot(mut self, tick: Int, state_hash: UInt64, state_bytes: List[UInt8]):
        var local_idx = self._find_slot_entry(self.local_slot, tick)
        if local_idx >= 0 and state_hash != 0:
            self.history[self.local_slot][local_idx].state_hash = state_hash
            self._verify_confirmed_hashes(tick)
        for i in range(len(self.snapshots)):
            if self.snapshots[i].tick == tick:
                self.snapshots[i].state_hash = state_hash
                self.snapshots[i].state_bytes = state_bytes.copy()
                return
        if len(self.snapshots) >= MAX_ROLLBACK_WINDOW:
            _ = self.snapshots.pop(0)
        var snap = RollbackSnapshot(tick, state_hash)
        snap.state_bytes = state_bytes.copy()
        self.snapshots.append(snap^)

    def restore_snapshot_at_or_before(self, target_tick: Int) -> Tuple[Int, List[UInt8]]:
        var best_idx = -1
        var best_tick = -1
        for i in range(len(self.snapshots)):
            var st = self.snapshots[i].tick
            if st <= target_tick and st > best_tick:
                best_tick = st
                best_idx = i
        if best_idx >= 0:
            return (best_tick, self.snapshots[best_idx].state_bytes.copy())
        return (-1, List[UInt8]())

    def broadcast_local_input(mut self, tick: Int, state_hash: UInt64, local_input: List[UInt8]) -> Int:
        self._upsert_slot_entry(self.local_slot, tick, True, state_hash, local_input)
        self.last_confirmed_input[self.local_slot] = local_input.copy()
        self._verify_confirmed_hashes(tick)
        if self.fd < 0:
            return 0
        var ilen = min(220, len(local_input))
        var pkt = List[UInt8](capacity=20 + ilen)
        var m = ROLLBACK_MAGIC
        for s in range(4):
            pkt.append(UInt8((m >> UInt32(s * 8)) & 255))
        pkt.append(UInt8(self.local_slot & 255))
        pkt.append(UInt8(ilen & 255))
        pkt.append(0)
        pkt.append(0)
        var t32 = UInt32(tick)
        for s in range(4):
            pkt.append(UInt8((t32 >> UInt32(s * 8)) & 255))
        for s in range(8):
            pkt.append(UInt8((state_hash >> UInt64(s * 8)) & 255))
        for i in range(ilen):
            pkt.append(local_input[i])
        var sent_peers = 0
        for p in range(self.player_count):
            if p == self.local_slot or not self.endpoints[p].active:
                continue
            var ep = self.endpoints[p]
            if udp_send_ipv4(self.fd, ep.ip_a, ep.ip_b, ep.ip_c, ep.ip_d, ep.port, pkt) == len(pkt):
                self.packets_sent += 1
                sent_peers += 1
        return sent_peers

    def poll_incoming(mut self) -> Int:
        if self.fd < 0:
            return 0
        var count = 0
        for _ in range(64):
            var raw = udp_recv_nonblocking(self.fd, 512)
            if len(raw) < 20:
                break
            var magic = (
                UInt32(raw[0])
                | (UInt32(raw[1]) << 8)
                | (UInt32(raw[2]) << 16)
                | (UInt32(raw[3]) << 24)
            )
            if magic != ROLLBACK_MAGIC:
                continue
            var sender = Int(raw[4])
            var ilen = Int(raw[5])
            if sender < 0 or sender >= self.player_count or sender == self.local_slot or 20 + ilen > len(raw):
                continue
            var tick = Int(
                UInt32(raw[8])
                | (UInt32(raw[9]) << 8)
                | (UInt32(raw[10]) << 16)
                | (UInt32(raw[11]) << 24)
            )
            var r_hash = UInt64(0)
            for s in range(8):
                r_hash |= UInt64(raw[12 + s]) << UInt64(s * 8)
            var inp = List[UInt8](capacity=ilen)
            for i in range(ilen):
                inp.append(raw[20 + i])
            self.packets_received += 1
            count += 1
            self.receive_authoritative_input(sender, tick, r_hash, inp)
        return count

    def receive_authoritative_input(mut self, sender: Int, tick: Int, r_hash: UInt64, inp: List[UInt8]):
        if sender < 0 or sender >= self.player_count or sender == self.local_slot:
            return
        var existing_idx = self._find_slot_entry(sender, tick)
        if existing_idx >= 0 and not self.history[sender][existing_idx].confirmed:
            if not Self._inputs_equal(self.history[sender][existing_idx].data, inp):
                if self.rollback_from_tick < 0 or tick < self.rollback_from_tick:
                    self.rollback_from_tick = tick
                self.rollbacks_triggered += 1
        self._upsert_slot_entry(sender, tick, True, r_hash, inp)
        self.last_confirmed_input[sender] = inp.copy()
        self._verify_confirmed_hashes(tick)

    def gather_tick_inputs(mut self, tick: Int) -> List[List[UInt8]]:
        """Returns 1 input byte-vector per active player `[0 .. player_count-1]` for `tick`.
        Remote slots without a confirmed packet for `tick` predict from `last_confirmed_input`."""
        _ = self.poll_incoming()
        var out = List[List[UInt8]]()
        for p in range(self.player_count):
            var idx = self._find_slot_entry(p, tick)
            if idx >= 0:
                out.append(self.history[p][idx].data.copy())
            else:
                var pred = self.last_confirmed_input[p].copy()
                self._upsert_slot_entry(p, tick, p == self.local_slot, 0, pred)
                out.append(pred^)
        return out^

    def consume_rollback_tick(mut self) -> Int:
        var t = self.rollback_from_tick
        self.rollback_from_tick = -1
        return t
