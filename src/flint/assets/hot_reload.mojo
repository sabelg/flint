"""Watches assets/ for finished files, so the running game can reload them.

inotify over the export folders and the layer files (package 1.2: a changed
layer file rebuilds the current layer's terrain, the journal on top). The pipeline moves files in with rename, an
editor closes them after writing; both are the moment a file is complete. A
reload waits for a short quiet spell, because one export writes several files.
"""
from std.ffi import external_call
from flint.platform.os import close_fd, get_time

comptime IN_CLOSE_WRITE: UInt32 = 0x00000008
comptime IN_MOVED_TO: UInt32 = 0x00000080
comptime IN_NONBLOCK: Int32 = 0x800
comptime IN_CLOEXEC: Int32 = 0x80000
comptime EVENT_HEADER_BYTES: Int = 16
comptime QUIET_SECONDS: Float64 = 0.15
comptime LONGEST_WAIT: Float64 = 0.6


def asset_stem(file_name: String) -> String:
    """Bront_Base_LOD1.glb and Bront_Base.json both mean Bront_Base; anything
    that is not a .glb or .json means nothing."""
    var bytes = file_name.as_bytes()
    var count = len(bytes)
    if count < 6:
        return ""
    var glb = bytes[count - 4] == 46 and bytes[count - 3] == 103 and bytes[count - 2] == 108 and bytes[count - 1] == 98
    var json = bytes[count - 5] == 46 and bytes[count - 4] == 106 and bytes[count - 3] == 115 and bytes[count - 2] == 111 and bytes[count - 1] == 110
    if not glb and not json:
        return ""
    var end = count - (4 if glb else 5)
    # Strip _LOD<n>.
    if glb and end > 5 and bytes[end - 5] == 95 and bytes[end - 4] == 76 and bytes[end - 3] == 79 and bytes[end - 2] == 68:
        end -= 5
    var stem = List[UInt8]()
    for index in range(end):
        stem.append(bytes[index])
    return String(unsafe_from_utf8=stem)


def is_layer_stem(stem: String) -> Bool:
    """True for "layer5" (assets/layers/layer5.json) or "prefabs": a layer
    file, which reloads terrain rather than a model (package 1.2)."""
    if stem == "prefabs":
        return True
    var bytes = stem.as_bytes()
    if len(bytes) < 6 or not (bytes[0] == 108 and bytes[1] == 97 and bytes[2] == 121 and bytes[3] == 101 and bytes[4] == 114):
        return False
    for index in range(5, len(bytes)):
        if bytes[index] < 48 or bytes[index] > 57:
            return False
    return True


def hot_reload_allowed(automated: Bool, recording: Bool, disabled: Bool) -> Bool:
    """A scripted route or a recording must see the same files start to finish,
    so that what it produced can be compared. Replays never make a watcher."""
    return not automated and not recording and not disabled


struct AssetWatcher(Movable):
    var fd: Int32
    var pending: List[String]
    var pending_models: List[String]  # pending stems with a model among their files
    var ready_models: List[String]
    var first_event: Float64
    var last_event: Float64

    def __init__(out self):
        self.fd = external_call["inotify_init1", Int32](IN_NONBLOCK | IN_CLOEXEC)
        self.pending = List[String]()
        self.pending_models = List[String]()
        self.ready_models = List[String]()
        self.first_event = 0
        self.last_event = 0

    def __deinit__(deinit self):
        if self.fd >= 0:
            close_fd(Int(self.fd))

    def watch(mut self, var folder: String) -> Bool:
        if self.fd < 0:
            return False
        var path = folder^
        return external_call["inotify_add_watch", Int32](self.fd, path.as_c_string_span(), IN_CLOSE_WRITE | IN_MOVED_TO) >= 0

    def note(mut self, stem: String, model: Bool, now: Float64):
        if len(self.pending) == 0:
            self.first_event = now
        self.last_event = now
        if model and stem not in self.pending_models:
            self.pending_models.append(stem)
        for known in self.pending:
            if known == stem:
                return
        self.pending.append(stem)

    def drain(mut self, now: Float64):
        """Takes every waiting event without blocking."""
        if self.fd < 0:
            return
        var buffer = List[UInt8](length=16384, fill=0)
        while True:
            var count = external_call["read", Int](self.fd, buffer.unsafe_ptr(), len(buffer))
            if count <= 0:
                return
            var offset = 0
            while offset + EVENT_HEADER_BYTES <= count:
                var length = Int(buffer[offset + 12]) | (Int(buffer[offset + 13]) << 8) | (Int(buffer[offset + 14]) << 16) | (Int(buffer[offset + 15]) << 24)
                var name = List[UInt8]()
                for index in range(offset + EVENT_HEADER_BYTES, min(offset + EVENT_HEADER_BYTES + length, count)):
                    if buffer[index] == 0:
                        break
                    name.append(buffer[index])
                var file_name = String(unsafe_from_utf8=name)
                var stem = asset_stem(file_name)
                if stem.byte_length() > 0:
                    self.note(stem, file_name.endswith(".glb"), now)
                offset += EVENT_HEADER_BYTES + length

    def take_ready(mut self, now: Float64) -> List[String]:
        """Asset stems changed on disk and quiet for long enough to load."""
        self.drain(now)
        var ready = List[String]()
        if len(self.pending) == 0:
            return ready^
        if now - self.last_event >= QUIET_SECONDS or now - self.first_event >= LONGEST_WAIT:
            ready = self.pending^
            self.pending = List[String]()
            self.ready_models = self.pending_models^
            self.pending_models = List[String]()
        return ready^

    def model_changed(self, stem: String) -> Bool:
        """Whether a .glb moved for a stem take_ready just gave. When none did, only
        the manifest changed, and the pipeline does that after every .blend save."""
        return stem in self.ready_models
