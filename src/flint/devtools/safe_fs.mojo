"""Workspace-confined file access for agent-supplied paths (`flint.devtools.safe_fs`).

ACP agents (`fs/read_text_file`, `fs/write_text_file`) and MCP clients (`asset.import`) name files
by path. These helpers keep every such access inside an allowed root:

- `resolve_confined_path`: lexical checks (no `..`, no NUL/control/backslash bytes, no hidden
  tool-config components such as `.git` or `.claude`, absolute paths only below the root).
- `open_beneath`: opens the path one component at a time from a directory fd of the root, with
  `O_NOFOLLOW` on every component, so a symlink swapped in after a check can never redirect the
  open outside the root (no lstat -> open race).
- `read_confined_regular`: the above plus `O_NONBLOCK` (a FIFO cannot hang the UI thread) and a
  regular-file check on the opened fd.
- `prepare_confined_write` + `write_confined_durable`: confined atomic replacement through
  `write_file_durable` (temp file, fsync, rename, directory fsync).
"""
from std.collections import List
from std.ffi import external_call
from std.sys.info import CompilationTarget
from flint.platform.os import (
    _fstat_regular_size,
    close_fd,
    last_errno,
    write_file_durable,
)

comptime _O_RDONLY: Int32 = 0
comptime _O_NONBLOCK: Int32 = 0o4000
comptime _O_CLOEXEC: Int32 = 0o2000000
comptime _O_PATH: Int32 = 0o10000000
comptime _O_NOFOLLOW: Int32 = 0o400000 if CompilationTarget.is_x86() else 0o100000
comptime _O_DIRECTORY: Int32 = 0o200000 if CompilationTarget.is_x86() else 0o40000
comptime _AT_FDCWD: Int32 = -100
comptime _ENOENT: Int32 = 2
comptime _EINTR: Int32 = 4
comptime _STAT_MODE_OFF: Int = 24 if CompilationTarget.is_x86() else 16
comptime _S_IFMT: UInt32 = 0o170000
comptime _S_IFREG: UInt32 = 0o100000

comptime SAFE_FS_OK: Int = 0
comptime SAFE_FS_MISSING: Int = 1
comptime SAFE_FS_NOT_REGULAR: Int = 2
comptime SAFE_FS_TOO_LARGE: Int = 3
comptime SAFE_FS_IO_ERROR: Int = 4
comptime SAFE_FS_REJECTED: Int = 5

comptime MAX_AGENT_TEXT_FILE_BYTES: Int = 16 * 1024 * 1024


def _c_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length() + 1)
    for b in s.as_bytes():
        out.append(b)
    out.append(0)
    return out^


def fnv1a64(data: Span[UInt8, _]) -> UInt64:
    """64-bit FNV-1a content hash (file revert guards: "is the file still what we wrote?")."""
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(data)):
        h = (h ^ UInt64(data[i])) * UInt64(0x100000001B3)
    return h


def text_hash64(s: String) -> UInt64:
    return fnv1a64(s.as_bytes())


def _has_bad_bytes(s: String) -> Bool:
    for b in s.as_bytes():
        if b < 32 or b == 92 or b == 127:
            return True
    return False


def _split_segments(path: String) raises -> List[String]:
    var segs = List[String]()
    var b = path.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        while i < n and b[i] == 47:
            i += 1
        if i >= n:
            break
        var j = i
        while j < n and b[j] != 47:
            j += 1
        var seg = String(path[byte=i:j])
        if seg == "..":
            raise Error("path traversal '..' rejected")
        if seg != ".":
            segs.append(seg^)
        i = j
    return segs^


def _strip_slashes(path: String) -> String:
    var cur = path
    while cur.byte_length() > 1 and cur.endswith("/"):
        var nxt = String(cur[byte=0 : cur.byte_length() - 1])
        cur = nxt^
    return cur^


def _realpath(path: String) -> String:
    var owned = path
    var out = List[UInt8](length=4096, fill=0)
    var got = external_call["realpath", Int](owned.as_c_string_span(), out.unsafe_ptr())
    if got == 0:
        return String("")
    var n = 0
    while n < len(out) and out[n] != 0:
        n += 1
    var bytes = List[UInt8](capacity=n)
    for i in range(n):
        bytes.append(out[i])
    return String(unsafe_from_utf8=bytes)


def _cwd() -> String:
    var buf = List[UInt8](length=4096, fill=0)
    if external_call["getcwd", Int](buf.unsafe_ptr(), Int64(4096)) == 0:
        return String("/")
    var n = 0
    while n < len(buf) and buf[n] != 0:
        n += 1
    var bytes = List[UInt8](capacity=n)
    for i in range(n):
        bytes.append(buf[i])
    return String(unsafe_from_utf8=bytes)


def canonical_root(root: String) raises -> String:
    """Absolute, symlink-resolved form of an allowed root directory."""
    if root.byte_length() == 0 or _has_bad_bytes(root):
        raise Error("invalid root")
    var base = root
    if not base.startswith("/"):
        var c = _strip_slashes(_cwd())
        base = (c + "/" + root) if c != "/" else ("/" + root)
    var rp = _realpath(base)
    if rp.byte_length() == 0:
        raise Error("root does not exist: " + root)
    return _strip_slashes(rp)


struct ConfinedPath(Copyable, Movable):
    """A path proven (lexically) to lie below `root`; `segments` are relative to `root`."""
    var root: String
    var segments: List[String]

    def __init__(out self):
        self.root = String()
        self.segments = List[String]()

    def rel(self) -> String:
        var out = String()
        for i in range(len(self.segments)):
            if i > 0:
                out += "/"
            out += self.segments[i]
        return out^

    def abs(self) -> String:
        var r = self.root
        if r == "/":
            return "/" + self.rel()
        return r + "/" + self.rel()


def resolve_confined_path(root: String, user_path: String, allow_hidden: Bool = False) raises -> ConfinedPath:
    """Lexically confines `user_path` (relative to `root`, or absolute below it). Rejects `..`,
    control bytes, backslashes and hidden components (`.git`, `.claude`, ...; `.flint` is allowed).
    Symlinks are handled at open time by `open_beneath`."""
    if user_path.byte_length() == 0:
        raise Error("empty path rejected")
    if _has_bad_bytes(user_path):
        raise Error("forbidden NUL/control/backslash byte in path")
    var raw = _split_segments(user_path)
    var canon = canonical_root(root)
    var out = ConfinedPath()
    out.root = canon
    if user_path.startswith("/"):
        var clean = _strip_slashes(user_path)
        var lex_root = _strip_slashes(root) if root.startswith("/") else String("")
        var rel: String
        if canon != "/" and clean.startswith(canon + "/"):
            rel = String(clean[byte = canon.byte_length() + 1 : clean.byte_length()])
        elif lex_root.byte_length() > 1 and clean.startswith(lex_root + "/"):
            rel = String(clean[byte = lex_root.byte_length() + 1 : clean.byte_length()])
        elif canon == "/":
            rel = clean
        else:
            raise Error("absolute path outside the allowed root rejected: " + user_path)
        out.segments = _split_segments(rel)
    else:
        out.segments = raw^
    if len(out.segments) == 0:
        raise Error("path must name a file inside the allowed root")
    if not allow_hidden:
        for i in range(len(out.segments)):
            ref seg = out.segments[i]
            if seg.startswith(".") and seg != ".flint":
                raise Error("hidden or tool-config path rejected: " + seg)
    return out^


def _openat(dfd: Int32, name: String, flags: Int32, mode: Int32 = 0) -> Int32:
    var c = _c_bytes(name)
    while True:
        var fd = external_call["openat", Int32](dfd, c.unsafe_ptr(), flags, mode)
        if fd < 0 and last_errno() == _EINTR:
            continue
        _ = c[0]
        return fd


def open_beneath(cp: ConfinedPath, final_flags: Int32, create_dirs: Bool = False) -> Tuple[Int, Int]:
    """Opens `cp` walking from its root one component at a time with `O_NOFOLLOW`: no component,
    final one included, may be a symlink. Returns `(fd, status)`; `fd` is -1 unless status is OK.
    With `create_dirs`, missing parent directories are created (mode 0755) along the way."""
    var dfd = _openat(_AT_FDCWD, cp.root, _O_PATH | _O_DIRECTORY | _O_CLOEXEC)
    if dfd < 0:
        return (-1, SAFE_FS_IO_ERROR)
    var n = len(cp.segments)
    for i in range(n - 1):
        var nfd = _openat(dfd, cp.segments[i], _O_PATH | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
        if nfd < 0 and create_dirs and last_errno() == _ENOENT:
            var c = _c_bytes(cp.segments[i])
            _ = external_call["mkdirat", Int32](dfd, c.unsafe_ptr(), UInt32(0o755))
            _ = c[0]
            nfd = _openat(dfd, cp.segments[i], _O_PATH | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
        if nfd < 0:
            var err = last_errno()
            close_fd(Int(dfd))
            return (-1, SAFE_FS_MISSING if err == _ENOENT else SAFE_FS_REJECTED)
        close_fd(Int(dfd))
        dfd = nfd
    var fd = _openat(dfd, cp.segments[n - 1], final_flags | _O_NOFOLLOW | _O_CLOEXEC)
    var err2 = last_errno()
    close_fd(Int(dfd))
    if fd < 0:
        return (-1, SAFE_FS_MISSING if err2 == _ENOENT else SAFE_FS_REJECTED)
    return (Int(fd), SAFE_FS_OK)


def _fd_mode(fd: Int) -> UInt32:
    var buf = List[UInt8](length=256, fill=0)
    if external_call["fstat", Int32](Int32(fd), buf.unsafe_ptr()) != 0:
        return 0
    var mode = (
        UInt32(buf[_STAT_MODE_OFF])
        | (UInt32(buf[_STAT_MODE_OFF + 1]) << 8)
        | (UInt32(buf[_STAT_MODE_OFF + 2]) << 16)
        | (UInt32(buf[_STAT_MODE_OFF + 3]) << 24)
    )
    return mode


def read_fd_regular(fd: Int, max_bytes: Int) -> Tuple[Int, List[UInt8]]:
    """Reads a whole regular file from `fd` (closed by the caller)."""
    var size = _fstat_regular_size(fd)
    if size < 0:
        return (SAFE_FS_NOT_REGULAR, List[UInt8]())
    if size > max_bytes:
        return (SAFE_FS_TOO_LARGE, List[UInt8]())
    var data = List[UInt8](length=size, fill=0)
    var off = 0
    while off < size:
        var n = Int(external_call["read", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(off), Int64(size - off)))
        if n < 0 and last_errno() == _EINTR:
            continue
        if n <= 0:
            break
        off += n
    if off < size:
        data.resize(off, 0)
    return (SAFE_FS_OK, data^)


def read_confined_regular(cp: ConfinedPath, max_bytes: Int = MAX_AGENT_TEXT_FILE_BYTES) -> Tuple[Int, List[UInt8]]:
    """Reads a regular file below the root: per-component `O_NOFOLLOW`, `O_NONBLOCK` (a FIFO or
    device cannot block the caller) and an `fstat` regular-file check on the opened fd."""
    var o = open_beneath(cp, _O_RDONLY | _O_NONBLOCK)
    if o[1] != SAFE_FS_OK:
        return (o[1], List[UInt8]())
    var res = read_fd_regular(o[0], max_bytes)
    close_fd(o[0])
    return res^


def read_agent_file(root: String, user_path: String, max_bytes: Int = MAX_AGENT_TEXT_FILE_BYTES) -> Tuple[Int, List[UInt8], String]:
    """`resolve_confined_path` + `read_confined_regular`. Returns `(status, bytes, error_text)`."""
    var cp: ConfinedPath
    try:
        cp = resolve_confined_path(root, user_path)
    except e:
        return (SAFE_FS_REJECTED, List[UInt8](), String(e))
    var r = read_confined_regular(cp, max_bytes)
    var why = String("")
    if r[0] == SAFE_FS_MISSING:
        why = "file not found: " + user_path
    elif r[0] == SAFE_FS_NOT_REGULAR:
        why = "not a regular file: " + user_path
    elif r[0] == SAFE_FS_TOO_LARGE:
        why = "file too large: " + user_path
    elif r[0] == SAFE_FS_REJECTED:
        why = "symlink or non-directory component rejected: " + user_path
    elif r[0] != SAFE_FS_OK:
        why = "read failed: " + user_path
    return (r[0], r[1].copy(), why^)


def prepare_confined_write(cp: ConfinedPath) -> Tuple[Int, String]:
    """Checks that `cp` can be replaced in place: parents are real directories below the root
    (missing ones are created), and neither the target nor its `.tmp` sibling (used by
    `write_file_durable`) is a symlink or a non-regular file. Returns `(status, error_text)`."""
    var o = open_beneath(cp, _O_PATH, create_dirs=True)
    if o[1] == SAFE_FS_OK:
        var mode = _fd_mode(o[0]) & _S_IFMT
        close_fd(o[0])
        if mode != _S_IFREG:
            return (SAFE_FS_NOT_REGULAR, "target exists and is not a regular file (or is a symlink)")
    elif o[1] != SAFE_FS_MISSING:
        return (o[1], "symlink or non-directory component rejected")
    var tmp = cp.copy()
    tmp.segments[len(tmp.segments) - 1] = tmp.segments[len(tmp.segments) - 1] + ".tmp"
    var t = open_beneath(tmp, _O_PATH)
    if t[1] == SAFE_FS_OK:
        var tmode = _fd_mode(t[0]) & _S_IFMT
        close_fd(t[0])
        if tmode != _S_IFREG:
            return (SAFE_FS_REJECTED, "temporary sibling is a symlink or special file")
    elif t[1] != SAFE_FS_MISSING:
        return (t[1], "symlink or non-directory component rejected")
    return (SAFE_FS_OK, String(""))


def write_confined_durable(cp: ConfinedPath, text: String) -> Tuple[Bool, String]:
    """Confined atomic replace via `write_file_durable`. Returns `(ok, error_text)`."""
    var chk = prepare_confined_write(cp)
    if chk[0] != SAFE_FS_OK:
        return (False, chk[1])
    var data = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        data.append(b)
    if not write_file_durable(cp.abs(), data):
        return (False, "write failed: " + cp.rel())
    return (True, String(""))


def write_confined_durable_bytes(cp: ConfinedPath, mut data: List[UInt8]) -> Tuple[Bool, String]:
    """Binary twin of `write_confined_durable`: confined checks, then `write_file_durable`
    (`.tmp` + fsync + rename + directory fsync). Returns `(ok, error_text)`."""
    var chk = prepare_confined_write(cp)
    if chk[0] != SAFE_FS_OK:
        return (False, chk[1])
    if not write_file_durable(cp.abs(), data):
        return (False, "write failed: " + cp.rel())
    return (True, String(""))
