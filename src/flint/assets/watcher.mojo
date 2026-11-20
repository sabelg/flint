"""Generic non-blocking Linux `inotify` live asset watcher for Flint.

Watches any game-registered directories and file extensions (e.g., `.png`, `.glb`,
`.json`, `.ttf`), coalesces multi-file editor writes over a short quiet window,
and automatically disables itself during deterministic replays and CI runs.
"""
from std.ffi import external_call
from flint.platform.os import close_fd, get_time, list_directory

comptime IN_CLOSE_WRITE: UInt32 = 0x00000008
comptime IN_MOVED_TO: UInt32 = 0x00000080
comptime IN_CREATE: UInt32 = 0x00000100
comptime IN_Q_OVERFLOW: UInt32 = 0x00004000
comptime IN_IGNORED: UInt32 = 0x00008000
comptime IN_ISDIR: UInt32 = 0x40000000
comptime IN_NONBLOCK: Int32 = 0x800
comptime IN_CLOEXEC: Int32 = 0x80000
comptime EVENT_HEADER_BYTES: Int = 16
comptime MAX_WATCH_DEPTH: Int = 64
comptime QUIET_SECONDS: Float64 = 0.15
comptime MAX_COALESCE_SECONDS: Float64 = 0.60


def _is_symlink(path: String) -> Bool:
    """Returns True if `path` is a symbolic link (`readlink >= 0`), guarding against symlink loops."""
    if path.byte_length() == 0:
        return False
    var owned = path
    var probe = List[UInt8](length=64, fill=0)
    var r = external_call["readlink", Int64](owned.as_c_string_span(), probe.unsafe_ptr(), Int64(64))
    return r >= 0


def _strip_trailing_slash(path: String) -> String:
    var b = path.as_bytes()
    var end = len(b)
    while end > 1 and b[end - 1] == 47:
        end -= 1
    if end == len(b):
        return path
    return String(path[byte=0:end])


def _is_directory_path(path: String) -> Bool:
    if path.byte_length() == 0:
        return False
    var owned = path
    var d = external_call["opendir", Int](owned.as_c_string_span())
    if d == 0:
        return False
    _ = external_call["closedir", Int32](d)
    return True


struct AssetWatcher(Movable):
    var fd: Int32
    var watch_descriptors: List[Int32]
    var watch_folders: List[String]
    var watch_roots: List[String]
    var extensions: List[String]
    var pending_paths: List[String]
    var first_event_at: Float64
    var last_event_at: Float64
    var enabled: Bool

    def __init__(out self, enabled: Bool = True):
        self.enabled = enabled
        self.fd = (
            external_call["inotify_init1", Int32](IN_NONBLOCK | IN_CLOEXEC)
            if enabled
            else Int32(-1)
        )
        self.watch_descriptors = List[Int32]()
        self.watch_folders = List[String]()
        self.watch_roots = List[String]()
        self.extensions = List[String]()
        self.pending_paths = List[String]()
        self.first_event_at = 0.0
        self.last_event_at = 0.0

    def __init__(out self, root_dir: String, enabled: Bool = True):
        self.enabled = enabled
        self.fd = (
            external_call["inotify_init1", Int32](IN_NONBLOCK | IN_CLOEXEC)
            if enabled
            else Int32(-1)
        )
        self.watch_descriptors = List[Int32]()
        self.watch_folders = List[String]()
        self.watch_roots = List[String]()
        self.extensions = List[String]()
        self.pending_paths = List[String]()
        self.first_event_at = 0.0
        self.last_event_at = 0.0
        if enabled and root_dir.byte_length() > 0:
            _ = self.watch_recursive(root_dir)

    def __deinit__(deinit self):
        if self.fd >= 0:
            close_fd(Int(self.fd))

    def allow_extension(mut self, ext: String):
        """Registers an extension like `.png`, `.glb`, `.json`, or `.ttf` (empty list = allow all)."""
        self.extensions.append(ext)

    def _add_watch_with_root_status(mut self, folder: String, root_dir: String) -> Int:
        """Returns `1` if newly watched, `0` if already watched (same path or inode `wd`), `-1` on failure."""
        if not self.enabled or self.fd < 0:
            return -1
        var clean_folder = _strip_trailing_slash(folder)
        for i in range(len(self.watch_folders)):
            if self.watch_folders[i] == clean_folder:
                return 0
        var path_copy = clean_folder
        var wd = external_call["inotify_add_watch", Int32](
            self.fd,
            path_copy.as_c_string_span(),
            IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE,
        )
        if wd < 0:
            return -1
        for i in range(len(self.watch_descriptors)):
            if self.watch_descriptors[i] == wd:
                # Same kernel inode already watched (e.g. via symlink/bind mount) — do not re-descend!
                return 0
        self.watch_descriptors.append(wd)
        self.watch_folders.append(clean_folder)
        self.watch_roots.append(root_dir)
        return 1

    def _add_watch_with_root(mut self, folder: String, root_dir: String) -> Bool:
        return self._add_watch_with_root_status(folder, root_dir) >= 0

    def add_directory(mut self, folder: String) -> Bool:
        return self._add_watch_with_root(folder, "")

    def _watch_recursive_under(mut self, cur_dir: String, root_dir: String, depth: Int = 0) -> Int:
        if not self.enabled or self.fd < 0 or depth >= MAX_WATCH_DEPTH:
            return 0
        var status = self._add_watch_with_root_status(cur_dir, root_dir)
        if status <= 0:
            return 0
        var count = 1
        var entries = list_directory(cur_dir)
        for name in entries:
            var child = (cur_dir + "/" + name) if cur_dir != "/" else ("/" + name)
            if _is_symlink(child):
                continue
            if _is_directory_path(child):
                count += self._watch_recursive_under(child, root_dir, depth + 1)
        return count

    def watch_recursive(mut self, root_dir: String) -> Int:
        """Recursively registers `root_dir` and all existing subdirectories with `inotify`,
        returning relative paths from `poll_ready` and auto-watching newly created subdirectories."""
        var clean_root = _strip_trailing_slash(root_dir)
        if clean_root.byte_length() == 0:
            return 0
        return self._watch_recursive_under(clean_root, clean_root)

    def _matches_ext(self, name: String) -> Bool:
        if len(self.extensions) == 0:
            return True
        for ext in self.extensions:
            if name.endswith(ext):
                return True
        return False

    def _folder_for_wd(self, wd: Int32) -> String:
        for i in range(len(self.watch_descriptors)):
            if self.watch_descriptors[i] == wd:
                return self.watch_folders[i]
        return ""

    def _root_for_wd(self, wd: Int32) -> String:
        for i in range(len(self.watch_descriptors)):
            if self.watch_descriptors[i] == wd:
                return self.watch_roots[i]
        return ""

    def _format_event_path(self, dir_prefix: String, root_prefix: String, fname: String) -> String:
        if root_prefix.byte_length() > 0:
            if dir_prefix == root_prefix:
                return fname
            var rlen = root_prefix.byte_length()
            if dir_prefix.byte_length() > rlen and String(dir_prefix[byte=0:rlen]) == root_prefix:
                var sub = String(dir_prefix[byte=rlen + 1 : dir_prefix.byte_length()])
                return (sub + "/" + fname) if sub.byte_length() > 0 else fname
        return (dir_prefix + "/" + fname) if dir_prefix.byte_length() > 0 else fname

    def _enqueue_pending(mut self, rel_or_full: String, t: Float64):
        var seen = False
        for existing in self.pending_paths:
            if existing == rel_or_full:
                seen = True
                break
        if not seen:
            if len(self.pending_paths) == 0:
                self.first_event_at = t
            self.pending_paths.append(rel_or_full)
        self.last_event_at = t

    def _scan_existing_files_in_new_dir(mut self, dir_path: String, root_prefix: String, t: Float64, depth: Int = 0):
        if depth >= MAX_WATCH_DEPTH:
            return
        var entries = list_directory(dir_path)
        for name in entries:
            var child = (dir_path + "/" + name) if dir_path != "/" else ("/" + name)
            if _is_symlink(child):
                continue
            if _is_directory_path(child):
                self._scan_existing_files_in_new_dir(child, root_prefix, t, depth + 1)
            elif self._matches_ext(name):
                var formatted = self._format_event_path(dir_path, root_prefix, name)
                self._enqueue_pending(formatted, t)

    def process_inotify_buffer(mut self, buf: List[UInt8], n: Int, t: Float64) -> Int:
        """Parses a raw `inotify` event buffer (`n` bytes), skipping zero-length control events
        (`IN_Q_OVERFLOW`, `IN_IGNORED`) without dropping subsequent file events."""
        var parsed = 0
        var offset = 0
        while offset + EVENT_HEADER_BYTES <= n:
            var wd = Int32(
                Int(buf[offset])
                | (Int(buf[offset + 1]) << 8)
                | (Int(buf[offset + 2]) << 16)
                | (Int(buf[offset + 3]) << 24)
            )
            var mask = UInt32(
                UInt32(buf[offset + 4])
                | (UInt32(buf[offset + 5]) << 8)
                | (UInt32(buf[offset + 6]) << 16)
                | (UInt32(buf[offset + 7]) << 24)
            )
            var name_len = (
                Int(buf[offset + 12])
                | (Int(buf[offset + 13]) << 8)
                | (Int(buf[offset + 14]) << 16)
                | (Int(buf[offset + 15]) << 24)
            )
            var name_start = offset + EVENT_HEADER_BYTES
            offset = name_start + name_len
            if name_len < 0 or offset > n:
                break
            parsed += 1
            if name_len == 0:
                # Zero-length control events (`IN_Q_OVERFLOW`, `IN_IGNORED`, etc.): advance to next event!
                continue
            var end_idx = name_start
            while end_idx < name_start + name_len and buf[end_idx] != 0:
                end_idx += 1
            if end_idx > name_start:
                var name_bytes = List[UInt8]()
                for bi in range(name_start, end_idx):
                    name_bytes.append(buf[bi])
                var fname = String(unsafe_from_utf8=name_bytes)
                var dir_prefix = self._folder_for_wd(wd)
                var root_prefix = self._root_for_wd(wd)
                var is_dir_ev = (mask & IN_ISDIR) != 0
                if is_dir_ev and ((mask & IN_CREATE) != 0 or (mask & IN_MOVED_TO) != 0):
                    if dir_prefix.byte_length() > 0:
                        var new_sub = (dir_prefix + "/" + fname) if dir_prefix != "/" else ("/" + fname)
                        if not _is_symlink(new_sub):
                            _ = self._watch_recursive_under(new_sub, root_prefix)
                            self._scan_existing_files_in_new_dir(new_sub, root_prefix, t)
                elif (not is_dir_ev) and ((mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) != 0):
                    if self._matches_ext(fname):
                        var out_path = self._format_event_path(dir_prefix, root_prefix, fname)
                        self._enqueue_pending(out_path, t)
        return parsed

    def poll_ready(mut self, now: Float64 = -1.0) -> List[String]:
        """Returns coalesced modified file paths ready for hot-reload."""
        var ready = List[String]()
        if not self.enabled or self.fd < 0:
            return ready^
        var t = get_time() if now < 0.0 else now
        var buf = List[UInt8](length=4096, fill=0)
        while True:
            var n = Int(external_call["read", Int64](self.fd, buf.unsafe_ptr(), Int64(4096)))
            if n <= 0:
                break
            _ = self.process_inotify_buffer(buf, n, t)

        if len(self.pending_paths) > 0:
            if (t - self.last_event_at) >= QUIET_SECONDS or (t - self.first_event_at) >= MAX_COALESCE_SECONDS:
                for p in self.pending_paths:
                    ready.append(p)
                self.pending_paths.clear()
        return ready^
