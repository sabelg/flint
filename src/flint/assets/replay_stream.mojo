"""Deterministic Input Replay Recorder & Verifier (`flint.assets.replay_stream`).

Records an initial authoritative `BlobWriter` state snapshot, a sequence of per-tick
`(dt, input_bytes)` frames, and the `fnv64` checksum of the final authoritative state
into a single LZ4-compressed `.flrp` binary file so simulations can record, replay,
and bit-verify combat or gameplay sessions.
"""
from flint.core.serial import BlobWriter, BlobReader, fourcc, fnv64
from flint.assets.checkpoint import (
    save_checkpoint_lz4,
    load_checkpoint_lz4,
    pack_checkpoint_lz4,
    unpack_checkpoint_lz4,
)

comptime REPLAY_STREAM_TAG: UInt32 = fourcc("FLRP")
comptime REPLAY_STREAM_VERSION: Int = 1


struct FlintReplayStream(Movable):
    var app_name: String
    var initial_blob: List[UInt8]
    var tick_dts: List[Float32]
    var tick_inputs: List[List[UInt8]]
    var final_hash: UInt64

    def __init__(out self):
        self.app_name = String()
        self.initial_blob = List[UInt8]()
        self.tick_dts = List[Float32]()
        self.tick_inputs = List[List[UInt8]]()
        self.final_hash = 0

    def clear(mut self):
        self.app_name = String()
        self.initial_blob.clear()
        self.tick_dts.clear()
        self.tick_inputs.clear()
        self.final_hash = 0

    def tick_count(self) -> Int:
        return len(self.tick_dts)

    def begin_recording(mut self, app_name: String, initial_writer: BlobWriter):
        self.clear()
        self.app_name = app_name
        self.initial_blob = initial_writer.data.copy()

    def record_tick(mut self, dt: Float32, input_bytes: List[UInt8]):
        self.tick_dts.append(dt)
        self.tick_inputs.append(input_bytes.copy())

    def finish_recording(mut self, final_writer: BlobWriter) -> UInt64:
        self.final_hash = fnv64(final_writer.data)
        return self.final_hash

    def initial_reader(self) -> BlobReader:
        return BlobReader(self.initial_blob.copy())

    def to_writer(self) -> BlobWriter:
        var w = BlobWriter()
        w.begin(REPLAY_STREAM_TAG, REPLAY_STREAM_VERSION)
        w.text(self.app_name)
        w.u64(self.final_hash)
        w.u32(UInt32(len(self.initial_blob)))
        for b in self.initial_blob:
            w.u8(b)
        var n = len(self.tick_dts)
        w.u32(UInt32(n))
        for i in range(n):
            w.f32(self.tick_dts[i])
            var inp_len = len(self.tick_inputs[i])
            w.u32(UInt32(inp_len))
            for k in range(inp_len):
                w.u8(self.tick_inputs[i][k])
        w.end()
        return w^

    @staticmethod
    def from_reader(mut r: BlobReader) raises -> FlintReplayStream:
        var ver = r.enter(REPLAY_STREAM_TAG, REPLAY_STREAM_VERSION)
        _ = ver
        var out = FlintReplayStream()
        out.app_name = r.text()
        out.final_hash = r.u64()
        var init_len = Int(r.u32())
        out.initial_blob.reserve(init_len)
        for _ in range(init_len):
            out.initial_blob.append(r.u8())
        var count = Int(r.u32())
        out.tick_dts.reserve(count)
        out.tick_inputs.reserve(count)
        for _ in range(count):
            out.tick_dts.append(r.f32())
            var inp_len = Int(r.u32())
            var frame_bytes = List[UInt8](capacity=inp_len)
            for _ in range(inp_len):
                frame_bytes.append(r.u8())
            out.tick_inputs.append(frame_bytes^)
        r.leave()
        return out^

    def pack_bytes(self) -> List[UInt8]:
        var w = self.to_writer()
        return pack_checkpoint_lz4(w)

    @staticmethod
    def unpack_bytes(packed: List[UInt8]) raises -> FlintReplayStream:
        var r = unpack_checkpoint_lz4(packed)
        return FlintReplayStream.from_reader(r)

    def save(self, path: String) -> Bool:
        if len(self.initial_blob) == 0:
            return False
        var w = self.to_writer()
        return save_checkpoint_lz4(path, w)

    @staticmethod
    def load(path: String) raises -> FlintReplayStream:
        var r = load_checkpoint_lz4(path)
        return FlintReplayStream.from_reader(r)
