"""The play loop's recorder: SD_RECORD files and the always-on session replay.

Play always records (unless SD_RECORD names another file, or SD_NO_SESSION_
REPLAY=1): every tick's input and checksums go to
`$XDG_STATE_HOME/sharddepth/last-session.sdrp`, replaced at each start. The
crash handler appends what is still in memory and links the file into the
crash folder, so a crash report always carries the replay up to the crash
tick.

The buffer is reserved once (1 MiB plus room for a tick) and never grows, so
its address stays valid for the handler. Around each tick the recorder
publishes, through the anchor, how many bytes hold whole ticks and how many
more hold the input of the tick now running.
"""
from std.ffi import external_call
from flint.core.anchor import (
    ANCHOR, A_REPLAY_ACTIVE, A_REPLAY_BUFFER, A_REPLAY_COMMITTED, A_REPLAY_PENDING, A_REPLAY_TICKS, T_REPLAY_PATH,
    anchor_ready, set_text, store,
)
from flint.core.replay import TICK_BYTES, encode_checks, write_input
from flint.platform.input import InputState
from flint.platform.os import append_bytes, write_bytes

comptime FLUSH_AT: Int = 1 << 20


struct SessionRecorder(Movable):
    var path: String
    var buffer: List[UInt8]
    var ticks: Int
    var active: Bool
    var published: Bool  # the crash handler knows about this recording

    def __init__(out self):
        self.path = String()
        self.buffer = List[UInt8](capacity=FLUSH_AT + 4 * TICK_BYTES)
        self.ticks = 0
        self.active = False
        self.published = False

    def start(mut self, path: String, var header: List[UInt8], for_crashes: Bool = True) -> Bool:
        """Begin a new file with this header. An older file there is removed first,
        so a crash folder's hard link to it keeps the old one intact."""
        var owned = path
        _ = external_call["unlink", Int32](owned.as_c_string_span())
        if not write_bytes(path, header):
            return False
        self.path = path
        self.active = True
        self.ticks = 0
        self.buffer.clear()
        if for_crashes and anchor_ready():
            set_text(T_REPLAY_PATH, path)
            self.publish()
            store(ANCHOR + A_REPLAY_ACTIVE, 1)
            self.published = True
        return True

    def publish(self, pending: Int = -1):
        if not self.published and not anchor_ready():
            return
        store(ANCHOR + A_REPLAY_BUFFER, Int(self.buffer.unsafe_ptr()))
        store(ANCHOR + A_REPLAY_COMMITTED, len(self.buffer))
        store(ANCHOR + A_REPLAY_PENDING, pending if pending >= 0 else len(self.buffer))
        store(ANCHOR + A_REPLAY_TICKS, self.ticks)

    def begin_tick(mut self, input: InputState):
        """The tick's input, before it runs."""
        if not self.active:
            return
        write_input(self.buffer, input)
        if self.published:
            store(ANCHOR + A_REPLAY_PENDING, len(self.buffer))

    def end_tick(mut self, checks: List[UInt64]):
        """The tick's checksums, after it ran. Whole ticks reach the file a MiB at a time."""
        if not self.active:
            return
        encode_checks(self.buffer, checks)
        self.ticks += 1
        if self.published:
            store(ANCHOR + A_REPLAY_COMMITTED, len(self.buffer))
            store(ANCHOR + A_REPLAY_PENDING, len(self.buffer))
            store(ANCHOR + A_REPLAY_TICKS, self.ticks)
        if len(self.buffer) >= FLUSH_AT:
            self.flush()

    def write_dev(mut self, records: List[UInt8]):
        """Dev records (core/replay.mojo, version 4) between two ticks: what the
        workbench changed before the next tick. Not ticks, so not counted."""
        if not self.active or len(records) == 0:
            return
        self.buffer.extend(records.copy())
        if self.published:
            store(ANCHOR + A_REPLAY_COMMITTED, len(self.buffer))
            store(ANCHOR + A_REPLAY_PENDING, len(self.buffer))

    def flush(mut self):
        if not self.active or len(self.buffer) == 0:
            return
        if self.published:
            # Told first: a crash during the write must not append these bytes twice.
            store(ANCHOR + A_REPLAY_COMMITTED, 0)
            store(ANCHOR + A_REPLAY_PENDING, 0)
        if not append_bytes(self.path, self.buffer):
            print("RECORDING_WRITE_FAILED " + self.path)
        self.buffer.clear()

    def finish(mut self):
        self.flush()
        if self.published:
            store(ANCHOR + A_REPLAY_ACTIVE, 0)
            self.published = False
        self.active = False
