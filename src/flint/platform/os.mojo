"""Native Linux operating-system services used by SHARDDEPTH.

Only system calls and the C ABI exported by the host C runtime are touched here.
No third-party compatibility library is linked into the game.
"""
from std.ffi import external_call
from std.time import monotonic


def get_time() -> Float64:
    return Float64(monotonic()) / 1000000000.0


def env_value(name: String) -> String:
    var data = read_stream("/proc/self/environ")
    var start = 0
    while start < len(data):
        var end = start
        while end < len(data) and data[end] != 0:
            end += 1
        if end > start + name.byte_length() and data[start + name.byte_length()] == 61:
            var matches = True
            for index in range(name.byte_length()):
                if data[start + index] != name.as_bytes()[index]:
                    matches = False
                    break
            if matches:
                var result = List[UInt8]()
                for index in range(start + name.byte_length() + 1, end):
                    result.append(data[index])
                return String(unsafe_from_utf8=result)
        start = end + 1
    return ""


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


def open_read(path: String) -> Int:
    var owned = path
    return Int(external_call["open", Int32](owned.as_c_string_span(), Int32(0), Int32(0)))


def open_write(path: String) -> Int:
    var owned = path
    return Int(external_call["open", Int32](owned.as_c_string_span(), Int32(1 | 64 | 512), Int32(438)))


def append_bytes(path: String, mut data: List[UInt8]) -> Bool:
    """Appends to a file, creating it if needed (O_WRONLY | O_CREAT | O_APPEND)."""
    var owned = path
    var fd = Int(external_call["open", Int32](owned.as_c_string_span(), Int32(1 | 64 | 1024), Int32(438)))
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
        var result = external_call["mkdir", Int32](owned.as_c_string_span(), Int32(511))
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
    var owned = path
    var flags = Int32(65536) if directory else Int32(0)  # O_RDONLY | O_DIRECTORY
    var fd = Int(external_call["open", Int32](owned.as_c_string_span(), flags, Int32(0)))
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
    var end = len(data) // 2 if fault == FAULT_SHORT_WRITE else len(data)
    var offset = 0
    while offset < end:
        var count = Int(external_call["write", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(offset), Int64(end - offset)))
        if count < 0 and last_errno() == EINTR:
            continue
        if count <= 0:
            close_fd(fd)
            _ = remove_file(path)
            return False
        offset += count
    if fault == FAULT_SHORT_WRITE:
        close_fd(fd)  # left behind cut off, as a crash mid-write would
        return False
    var synced = not sync or external_call["fsync", Int32](Int32(fd)) == 0
    close_fd(fd)
    if not synced:
        _ = remove_file(path)
        return False
    return fault != FAULT_BEFORE_RENAME


def durable_rename(source: String, destination: String, sync: Bool = True) -> Bool:
    """rename(2), then fsync the directory: the rename is only durable once
    the directory is. Source and destination share a directory."""
    if not rename_file(source, destination):
        _ = remove_file(source)
        return False
    return not sync or fsync_path(parent_directory(destination), True)


def write_file_durable(path: String, mut data: List[UInt8], fault: Int = FAULT_NONE) -> Bool:
    """Replaces `path` with `data` so that after a crash or power loss at any
    moment the file is either the old one or the new one, never a mix: write
    `path`.tmp and fsync it, rename it over `path`, fsync the directory."""
    var temporary = path + ".tmp"
    return write_synced(temporary, data, fault) and durable_rename(temporary, path)


def player_save_directory() -> String:
    """SD_SAVE_DIR, then $XDG_DATA_HOME/sharddepth, then ~/.local/share/sharddepth."""
    var chosen = env_value("SD_SAVE_DIR")
    if chosen.byte_length() > 0:
        return chosen
    var data_home = env_value("XDG_DATA_HOME")
    if data_home.byte_length() > 0:
        return data_home + "/sharddepth"
    return env_value("HOME") + "/.local/share/sharddepth"


def close_fd(fd: Int):
    if fd >= 0:
        _ = external_call["close", Int32](Int32(fd))


def file_size(path: String) -> Int:
    var fd = open_read(path)
    if fd < 0:
        return -1
    var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
    close_fd(fd)
    return size


comptime EINTR: Int32 = 4


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
    while True:
        var entry = external_call["readdir", Int](directory)
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
    _ = external_call["closedir", Int32](directory)
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
    var size = file_size(path)
    if size <= 0:
        return List[UInt8]()
    var fd = open_read(path)
    if fd < 0:
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
