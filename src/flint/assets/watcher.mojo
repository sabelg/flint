"""Generic non-blocking Linux `inotify` live asset watcher for Flint.

Watches any game-registered directories and file extensions (e.g., `.png`, `.glb`,
`.json`, `.ttf`), coalesces multi-file editor writes over a short quiet window,
and automatically disables itself during deterministic replays and CI runs.
"""
from std.ffi import external_call
from flint.platform.os import close_fd, get_time

comptime IN_CLOSE_WRITE: UInt32 = 0x00000008
comptime IN_MOVED_TO: UInt32 = 0x00000080
comptime IN_NONBLOCK: Int32 = 0x800
comptime IN_CLOEXEC: Int32 = 0x80000
comptime EVENT_HEADER_BYTES: Int = 16
comptime QUIET_SECONDS: Float64 = 0.15
comptime MAX_COALESCE_SECONDS: Float64 = 0.60


struct AssetWatcher(Movable):
    var fd: Int32
    var watch_descriptors: List[Int32]
    var watch_folders: List[String]
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
        self.extensions = List[String]()
        self.pending_paths = List[String]()
        self.first_event_at = 0.0
        self.last_event_at = 0.0

    def __deinit__(deinit self):
        if self.fd >= 0:
            close_fd(Int(self.fd))

    def allow_extension(mut self, ext: String):
        """Registers an extension like `.png`, `.glb`, `.json`, or `.ttf` (empty list = allow all)."""
        self.extensions.append(ext)

    def add_directory(mut self, folder: String) -> Bool:
        if not self.enabled or self.fd < 0:
            return False
        var path_copy = folder
        var wd = external_call["inotify_add_watch", Int32](
            self.fd, path_copy.as_c_string_span(), IN_CLOSE_WRITE | IN_MOVED_TO
        )
        if wd < 0:
            return False
        self.watch_descriptors.append(wd)
        self.watch_folders.append(folder)
        return True

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
            var offset = 0
            while offset + EVENT_HEADER_BYTES <= n:
                var wd = Int32(
                    Int(buf[offset])
                    | (Int(buf[offset + 1]) << 8)
                    | (Int(buf[offset + 2]) << 16)
                    | (Int(buf[offset + 3]) << 24)
                )
                var name_len = (
                    Int(buf[offset + 12])
                    | (Int(buf[offset + 13]) << 8)
                    | (Int(buf[offset + 14]) << 16)
                    | (Int(buf[offset + 15]) << 24)
                )
                var name_start = offset + EVENT_HEADER_BYTES
                offset = name_start + name_len
                if name_len <= 0 or offset > n:
                    break
                var end_idx = name_start
                while end_idx < name_start + name_len and buf[end_idx] != 0:
                    end_idx += 1
                if end_idx > name_start:
                    var name_bytes = List[UInt8]()
                    for bi in range(name_start, end_idx):
                        name_bytes.append(buf[bi])
                    var fname = String(unsafe_from_utf8=name_bytes)
                    if self._matches_ext(fname):
                        var dir_prefix = self._folder_for_wd(wd)
                        var full = (dir_prefix + "/" + fname) if dir_prefix.byte_length() > 0 else fname
                        var seen = False
                        for existing in self.pending_paths:
                            if existing == full:
                                seen = True
                                break
                        if not seen:
                            if len(self.pending_paths) == 0:
                                self.first_event_at = t
                            self.pending_paths.append(full)
                        self.last_event_at = t

        if len(self.pending_paths) > 0:
            if (t - self.last_event_at) >= QUIET_SECONDS or (t - self.first_event_at) >= MAX_COALESCE_SECONDS:
                for p in self.pending_paths:
                    ready.append(p)
                self.pending_paths.clear()
        return ready^
