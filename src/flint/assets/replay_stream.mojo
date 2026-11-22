"""Deterministic Input Replay Recorder & Verifier (`flint.assets.replay_stream`).

Records an initial authoritative `BlobWriter` state snapshot, a sequence of per-tick
`(dt, input_bytes)` frames, and the `fnv64` checksum of the final authoritative state
into a single LZ4-compressed `.flrp` binary file so simulations can record, replay,
and bit-verify combat or gameplay sessions.
"""
from flint.core.math import Vec3
from flint.core.serial import BlobWriter, BlobReader, fourcc, fnv64
from flint.assets.checkpoint import (
    save_checkpoint_lz4,
    load_checkpoint_lz4,
    pack_checkpoint_lz4,
    unpack_checkpoint_lz4,
)

comptime REPLAY_STREAM_TAG: UInt32 = fourcc("FLRP")
comptime REPLAY_STREAM_VERSION: Int = 3


struct CameraPathKnot(Copyable, Movable, ImplicitlyCopyable):
    """Single keyframe on a `.flrp` V3 Catmull-Rom cinematic camera director spline."""
    var tick: Int
    var eye: Vec3
    var target: Vec3
    var fov_y: Float32
    var aperture: Float32
    var time_scale: Float32

    def __init__(
        out self,
        tick: Int = 0,
        eye: Vec3 = Vec3(0.0, 4.0, -14.0),
        target: Vec3 = Vec3(0.0, 2.0, 0.0),
        fov_y: Float32 = 58.0,
        aperture: Float32 = 0.0,
        time_scale: Float32 = 1.0,
    ):
        self.tick = tick
        self.eye = eye
        self.target = target
        self.fov_y = fov_y
        self.aperture = aperture
        self.time_scale = time_scale


@always_inline
def _catmull_rom_scalar(p0: Float32, p1: Float32, p2: Float32, p3: Float32, t: Float32) -> Float32:
    var t2 = t * t
    var t3 = t2 * t
    return Float32(0.5) * (
        (Float32(2.0) * p1)
        + (-p0 + p2) * t
        + (Float32(2.0) * p0 - Float32(5.0) * p1 + Float32(4.0) * p2 - p3) * t2
        + (-p0 + Float32(3.0) * p1 - Float32(3.0) * p2 + p3) * t3
    )


@always_inline
def _catmull_rom_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: Float32) -> Vec3:
    return Vec3(
        _catmull_rom_scalar(p0.x, p1.x, p2.x, p3.x, t),
        _catmull_rom_scalar(p0.y, p1.y, p2.y, p3.y, t),
        _catmull_rom_scalar(p0.z, p1.z, p2.z, p3.z, t),
    )


struct FlintReplayStream(Movable):
    var app_name: String
    var initial_blob: List[UInt8]
    var tick_dts: List[Float32]
    var tick_inputs: List[List[UInt8]]
    var keyframe_ticks: List[Int]
    var keyframe_blobs: List[List[UInt8]]
    var camera_knots: List[CameraPathKnot]
    var final_hash: UInt64

    def __init__(out self):
        self.app_name = String()
        self.initial_blob = List[UInt8]()
        self.tick_dts = List[Float32]()
        self.tick_inputs = List[List[UInt8]]()
        self.keyframe_ticks = List[Int]()
        self.keyframe_blobs = List[List[UInt8]]()
        self.camera_knots = List[CameraPathKnot]()
        self.final_hash = 0

    def clear(mut self):
        self.app_name = String()
        self.initial_blob.clear()
        self.tick_dts.clear()
        self.tick_inputs.clear()
        self.keyframe_ticks.clear()
        self.keyframe_blobs.clear()
        self.camera_knots.clear()
        self.final_hash = 0

    def tick_count(self) -> Int:
        return len(self.tick_dts)

    def keyframe_count(self) -> Int:
        return len(self.keyframe_ticks)

    def camera_knot_count(self) -> Int:
        return len(self.camera_knots)

    def clear_camera_knots(mut self):
        self.camera_knots.clear()

    def add_camera_knot(
        mut self,
        tick: Int,
        eye: Vec3,
        target: Vec3,
        fov_y: Float32 = 58.0,
        aperture: Float32 = 0.0,
        time_scale: Float32 = 1.0,
    ):
        """Inserts or replaces a Catmull-Rom camera director knot kept sorted by `tick`."""
        var knot = CameraPathKnot(tick, eye, target, fov_y, aperture, time_scale)
        for i in range(len(self.camera_knots)):
            if self.camera_knots[i].tick == tick:
                self.camera_knots[i] = knot^
                return
            if self.camera_knots[i].tick > tick:
                self.camera_knots.insert(i, knot^)
                return
        self.camera_knots.append(knot^)

    def evaluate_camera_spline(self, fractional_tick: Float32) -> Tuple[Bool, CameraPathKnot]:
        """Evaluates smooth C1 Catmull-Rom spline across `camera_knots` at `fractional_tick`."""
        var n = len(self.camera_knots)
        if n == 0:
            return (False, CameraPathKnot())
        if n == 1 or fractional_tick <= Float32(self.camera_knots[0].tick):
            return (True, self.camera_knots[0].copy())
        if fractional_tick >= Float32(self.camera_knots[n - 1].tick):
            return (True, self.camera_knots[n - 1].copy())
        var seg = 0
        for i in range(n - 1):
            if fractional_tick >= Float32(self.camera_knots[i].tick) and fractional_tick <= Float32(self.camera_knots[i + 1].tick):
                seg = i
                break
        var k0 = self.camera_knots[max(0, seg - 1)].copy()
        var k1 = self.camera_knots[seg].copy()
        var k2 = self.camera_knots[seg + 1].copy()
        var k3 = self.camera_knots[min(n - 1, seg + 2)].copy()
        var span = max(Float32(0.0001), Float32(k2.tick - k1.tick))
        var u = max(Float32(0.0), min(Float32(1.0), (fractional_tick - Float32(k1.tick)) / span))
        var out_eye = _catmull_rom_vec3(k0.eye, k1.eye, k2.eye, k3.eye, u)
        var out_tgt = _catmull_rom_vec3(k0.target, k1.target, k2.target, k3.target, u)
        var out_fov = max(Float32(18.0), min(Float32(115.0), _catmull_rom_scalar(k0.fov_y, k1.fov_y, k2.fov_y, k3.fov_y, u)))
        var out_ap = max(Float32(0.0), _catmull_rom_scalar(k0.aperture, k1.aperture, k2.aperture, k3.aperture, u))
        var out_ts = max(Float32(0.05), min(Float32(4.0), _catmull_rom_scalar(k0.time_scale, k1.time_scale, k2.time_scale, k3.time_scale, u)))
        return (True, CameraPathKnot(Int(fractional_tick), out_eye, out_tgt, out_fov, out_ap, out_ts))

    def begin_recording(mut self, app_name: String, initial_writer: BlobWriter):
        self.clear()
        self.app_name = app_name
        self.initial_blob = initial_writer.data.copy()

    def record_tick(mut self, dt: Float32, input_bytes: List[UInt8]):
        self.tick_dts.append(dt)
        self.tick_inputs.append(input_bytes.copy())

    def add_keyframe(mut self, tick: Int, state_writer: BlobWriter):
        """Records an authoritative mid-stream keyframe snapshot at `tick` (`0 <= tick <= tick_count()`)
        for interactive timeline scrubbing."""
        if tick <= 0 or len(state_writer.data) == 0:
            return
        for i in range(len(self.keyframe_ticks)):
            if self.keyframe_ticks[i] == tick:
                self.keyframe_blobs[i] = state_writer.data.copy()
                return
        self.keyframe_ticks.append(tick)
        self.keyframe_blobs.append(state_writer.data.copy())

    def nearest_keyframe_tick_at_or_before(self, target_tick: Int) -> Int:
        """Returns the closest recorded keyframe tick `<= target_tick` (or `0` if before the first keyframe)."""
        var best_tick = 0
        for i in range(len(self.keyframe_ticks)):
            var kt = self.keyframe_ticks[i]
            if kt <= target_tick and kt > best_tick:
                best_tick = kt
        return best_tick

    def keyframe_reader_at_or_before(self, target_tick: Int) -> BlobReader:
        """Returns a `BlobReader` for the closest recorded keyframe `<= target_tick`
        (falling back to `initial_reader()` if before the first keyframe)."""
        var best_idx = -1
        var best_tick = 0
        for i in range(len(self.keyframe_ticks)):
            var kt = self.keyframe_ticks[i]
            if kt <= target_tick and kt > best_tick:
                best_tick = kt
                best_idx = i
        if best_idx >= 0:
            return BlobReader(self.keyframe_blobs[best_idx].copy())
        return self.initial_reader()

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
        var kf_n = len(self.keyframe_ticks)
        w.u32(UInt32(kf_n))
        for i in range(kf_n):
            w.u32(UInt32(self.keyframe_ticks[i]))
            var kb_len = len(self.keyframe_blobs[i])
            w.u32(UInt32(kb_len))
            for k in range(kb_len):
                w.u8(self.keyframe_blobs[i][k])
        var ck_n = len(self.camera_knots)
        w.u32(UInt32(ck_n))
        for i in range(ck_n):
            var ck = self.camera_knots[i].copy()
            w.u32(UInt32(max(0, ck.tick)))
            w.f32(ck.eye.x)
            w.f32(ck.eye.y)
            w.f32(ck.eye.z)
            w.f32(ck.target.x)
            w.f32(ck.target.y)
            w.f32(ck.target.z)
            w.f32(ck.fov_y)
            w.f32(ck.aperture)
            w.f32(ck.time_scale)
        w.end()
        return w^

    @staticmethod
    def from_reader(mut r: BlobReader) raises -> FlintReplayStream:
        var ver = r.enter(REPLAY_STREAM_TAG, REPLAY_STREAM_VERSION)
        var out = FlintReplayStream()
        out.app_name = r.text()
        out.final_hash = r.u64()
        var init_len = Int(r.u32())
        if r.failed or init_len < 0 or init_len > max(0, r.limit() - r.offset):
            raise Error("replay stream blob: invalid initial_blob length")
        out.initial_blob.reserve(init_len)
        for _ in range(init_len):
            out.initial_blob.append(r.u8())
        var count = Int(r.u32())
        if r.failed or count < 0 or count > max(0, r.limit() - r.offset) // 8:
            raise Error("replay stream blob: invalid tick count")
        out.tick_dts.reserve(count)
        out.tick_inputs.reserve(count)
        for _ in range(count):
            out.tick_dts.append(r.f32())
            var inp_len = Int(r.u32())
            if r.failed or inp_len < 0 or inp_len > max(0, r.limit() - r.offset):
                raise Error("replay stream blob: invalid tick input length")
            var frame_bytes = List[UInt8](capacity=inp_len)
            for _ in range(inp_len):
                frame_bytes.append(r.u8())
            out.tick_inputs.append(frame_bytes^)
        if ver >= 2:
            var kf_n = Int(r.u32())
            if r.failed or kf_n < 0 or kf_n > max(0, r.limit() - r.offset) // 8:
                raise Error("replay stream blob: invalid keyframe count")
            out.keyframe_ticks.reserve(kf_n)
            out.keyframe_blobs.reserve(kf_n)
            for _ in range(kf_n):
                out.keyframe_ticks.append(Int(r.u32()))
                var kb_len = Int(r.u32())
                if r.failed or kb_len < 0 or kb_len > max(0, r.limit() - r.offset):
                    raise Error("replay stream blob: invalid keyframe length")
                var kb = List[UInt8](capacity=kb_len)
                for _ in range(kb_len):
                    kb.append(r.u8())
                out.keyframe_blobs.append(kb^)
        if ver >= 3:
            var ck_n = Int(r.u32())
            if r.failed or ck_n < 0 or ck_n > max(0, r.limit() - r.offset) // 40:
                raise Error("replay stream blob: invalid camera knot count")
            out.camera_knots.reserve(ck_n)
            for _ in range(ck_n):
                var kt = Int(r.u32())
                var ex = r.f32()
                var ey = r.f32()
                var ez = r.f32()
                var tx = r.f32()
                var ty = r.f32()
                var tz = r.f32()
                var fv = r.f32()
                var ap = r.f32()
                var ts = r.f32()
                out.camera_knots.append(CameraPathKnot(kt, Vec3(ex, ey, ez), Vec3(tx, ty, tz), fv, ap, ts))
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
