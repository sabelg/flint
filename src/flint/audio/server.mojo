"""Multithreaded Real-Time AudioServer, Lock-Free MPSC Ring, HRTF Virtualizer & Mastering (`flint.audio.server`).

Architecture:
  - `AudioCmd` (exact 64 bytes) & `AudioMpscRing`: 64-byte cache-line-isolated lock-free
    Vyukov bounded MPSC ring safe for concurrent `JobPool` worker threads.
  - Generational `SoundHandle`, `SpatialEmitter` (512 logical emitters -> 64 physical voices
    with click-free 128-sample steal crossfade).
  - Continuous per-quantum Woodworth spherical-head `ItdDelayLine` HRTF (`a = 0.082 m`,
    `~29.5` samples @ +90° right, 48 kHz), head-shadow ILD, rear/elevation pinna `CytomicSVF`,
    and 4-point Hermite cubic Doppler resampling (`listener_fwd`, `listener_right`, `listener_up`).
  - Topological post-order `AudioBus` DAG (`Master`, `SFX`, `Music`, `Voice`, `Ambience`, `UI` +
    custom buses) + 4 `StereoReverbFDN` aux zones, per-bus 3-band `CytomicSVF` EQ, RMS compressor,
    generalized sidechain ducking (`Voice` -> `Music`), ITU-R BS.1770-4 K-weighted LUFS meter,
    and 64-sample lookahead 4x-oversampled zero-overshoot true-peak limiter (`<= -0.1 dBFS`).
  - Zero-allocation `CyclesStemNode` routing `flint.cycles` patterns into buses with
    cycle-quantized equal-power stem crossfading.
  - Both dedicated RT `pthread` hardware output (`PulseClient` / ALSA) and deterministic
    headless `render_offline_quantum(mut out_lr, frames)`.
"""
from std.atomic import Atomic
from std.collections import Dict, List
from std.ffi import external_call
from std.math import sin, cos, sqrt, exp, log10, floor, abs, atan2, isnan, isinf, isfinite
from std.memory import bitcast
from std.sys import size_of
from flint.core.math import Vec3
from flint.app.engine_app import Resource
from flint.ecs.world import comp_hash64
from flint.platform.os import close_fd, read_bytes, get_time, list_directory, file_size
from flint.platform.pulse import PulseClient
from flint.platform.audio import (
    WavClip,
    alsa_open_pcm,
    alsa_write_interleaved,
    alsa_close_pcm,
    decode_wav,
    _alsa_configure_fd,
)
from flint.platform.dsp import (
    CytomicSVF,
    StereoReverbFDN,
    ItdDelayLine,
    equal_power_gains,
    soft_clip_tanh,
    DSP_PI,
    SPEED_OF_SOUND,
)
from flint.cycles.synth import CyclesEngine
from flint.cycles.script import Program, run_program
from flint.audio.qoa import QoaAsset, QoaStreamCursor
from flint.audio.acoustics import CLEAR_CUTOFF_HZ
from flint.audio.attenuation import AttenuationPreset
from flint.audio.vorbis import is_ogg_vorbis, decode_ogg_vorbis
from flint.audio.stream import (
    AudioStreamPool,
    StreamSourceInfo,
    probe_stream_source,
    stream_load,
    stream_store,
    stream_ring_sample,
    stream_frame_ready,
    stream_publish_read,
    _audio_stream_thread_entry,
    STREAM_MAX_CHANNELS,
    STREAM_FMT_WAV16,
    STREAM_FMT_QOA,
    STREAM_STATE_ARMED,
    STREAM_STATE_ACTIVE,
    STREAM_STATE_RELEASING,
    STREAM_RING_FRAMES,
    STREAM_RING_MASK,
    _SC_STATE,
    _SC_WRITE,
    _SC_RING,
    _SC_OWNER_SLOT,
    _SC_OWNER_GEN,
)

# Standard Mix Bus IDs
comptime BUS_MASTER: Int = 0
comptime BUS_SFX: Int = 1
comptime BUS_MUSIC: Int = 2
comptime BUS_VOICE: Int = 3
comptime BUS_AMBIENCE: Int = 4
comptime BUS_UI: Int = 5
comptime DEFAULT_BUS_COUNT: Int = 6
comptime AUX_REVERB_ZONES: Int = 4

# Capacities &DSP Constants
comptime MAX_LOGICAL_EMITTERS: Int = 512
comptime MAX_PHYSICAL_VOICES: Int = 64
comptime STEAL_XFADE_SAMPLES: Int = 128
comptime MAX_QUANTUM_FRAMES: Int = 2048


@always_inline
def _sanitize_pitch(p: Float32) -> Float32:
    if isnan(p) or isinf(p):
        return Float32(1.0)
    return max(Float32(0.05), min(Float32(16.0), p))


@always_inline
def _wrap_loop_pos(pos: Float64, total_f: Float64) -> Float64:
    if total_f <= 0.0 or isnan(pos) or isinf(pos):
        return 0.0
    var wrapped = pos - floor(pos / total_f) * total_f
    if wrapped < 0.0 or wrapped >= total_f or isnan(wrapped):
        return 0.0
    return wrapped
comptime LIMITER_LOOKAHEAD: Int = 64
comptime WOODWORTH_HEAD_RADIUS_M: Float32 = 0.082  # 8.2 cm -> 29.50 samples @ +90 deg, 48 kHz

# AudioCmd opcodes
comptime ACMD_NONE: UInt32 = 0
comptime ACMD_PLAY: UInt32 = 1
comptime ACMD_STOP: UInt32 = 2
comptime ACMD_UPDATE_EMITTER: UInt32 = 3
comptime ACMD_SET_OCCLUSION: UInt32 = 4
comptime ACMD_SET_LISTENER_POS_VEL: UInt32 = 5
comptime ACMD_SET_LISTENER_BASIS_FWD_RIGHT: UInt32 = 6
comptime ACMD_SET_LISTENER_UP: UInt32 = 7
comptime ACMD_SET_BUS_GAIN: UInt32 = 8
comptime ACMD_SET_ROOM_PROBE: UInt32 = 9
comptime ACMD_STOP_ALL: UInt32 = 10
comptime CMD_INSTALL_CLIP: UInt32 = 11
comptime ACMD_INSTALL_CLIP: UInt32 = CMD_INSTALL_CLIP
comptime CMD_CONFIG_BUS: UInt32 = 12
comptime ACMD_CONFIG_BUS: UInt32 = CMD_CONFIG_BUS
comptime CMD_ADD_BUS: UInt32 = 13
comptime CMD_ADD_STEM: UInt32 = 14
comptime CMD_TRANSITION_STEM: UInt32 = 15
comptime CMD_CONFIG_MASTER: UInt32 = 16
comptime ACMD_PAUSE: UInt32 = 17
comptime ACMD_GLOBAL_PAUSE: UInt32 = 18
comptime ACMD_SET_VOICE_PARAMS: UInt32 = 19
comptime CMD_INSTALL_ATTEN: UInt32 = 20
comptime ACMD_SET_EMITTER_ATTEN: UInt32 = 21

# Output backends (published in telemetry `_TELEM_BACKEND`)
comptime AUDIO_BACKEND_NONE: Int = 0
comptime AUDIO_BACKEND_PULSE: Int = 1
comptime AUDIO_BACKEND_ALSA: Int = 2
comptime AUDIO_BACKEND_HEADLESS: Int = 3
comptime RECONNECT_BACKOFF_MIN_MS: Int = 100
comptime RECONNECT_BACKOFF_MAX_MS: Int = 5000

# Fade constants
comptime PAUSE_FADE_FRAMES: Int = 256

# CMD_CONFIG_BUS sub-opcodes (carried in AudioCmd.flags)
comptime _BUS_CFG_GAIN_MUTE: UInt32 = 1
comptime _BUS_CFG_EQ_DB: UInt32 = 2
comptime _BUS_CFG_COMPRESSOR: UInt32 = 3
comptime _BUS_CFG_SIDECHAIN: UInt32 = 4
comptime _BUS_CFG_AUX_SEND: UInt32 = 5
comptime _BUS_CFG_PAUSE: UInt32 = 6

# CMD_CONFIG_MASTER sub-opcodes (carried in AudioCmd.flags)
comptime _MASTER_CFG_DC_BLOCKER: UInt32 = 1
comptime _MASTER_CFG_AUX_GAIN: UInt32 = 2
comptime _MASTER_CFG_LIMITER_CEIL: UInt32 = 3

# AudioCmd flags
comptime AFLAG_LOOPING: UInt32 = 1 << 0
comptime AFLAG_SPATIAL: UInt32 = 1 << 1
comptime AFLAG_DIRECT_STEREO: UInt32 = 1 << 2
comptime AFLAG_STREAM: UInt32 = 1 << 3  # stream channel index carried in flags bits 8..15
comptime ASSET_KIND_STREAM: Int = 2


def next_reconnect_backoff_ms(current_ms: Int) -> Int:
    """Exponential reconnect backoff: 100 ms doubling up to 5 s."""
    if current_ms < RECONNECT_BACKOFF_MIN_MS:
        return RECONNECT_BACKOFF_MIN_MS
    return min(RECONNECT_BACKOFF_MAX_MS, current_ms * 2)


@fieldwise_init
struct SoundHandle(Copyable, ImplicitlyCopyable, Movable):
    """Generational handle identifying a logical emitter in `AudioServer`."""
    var slot: Int32
    var generation: UInt32

    def __init__(out self):
        self.slot = -1
        self.generation = 0

    @always_inline
    def is_valid(self) -> Bool:
        return self.slot >= 0 and self.generation > 0

    @always_inline
    def equals(self, other: SoundHandle) -> Bool:
        return self.slot == other.slot and self.generation == other.generation


@fieldwise_init
struct AudioCmd(Copyable, ImplicitlyCopyable, Movable):
    """Exact 64-byte (`16 x 4B`) command packet for `AudioMpscRing`."""
    var kind: UInt32
    var flags: UInt32
    var slot: Int32
    var generation: UInt32
    var asset_id: Int32
    var bus_id: Int32
    var p0: Float32
    var p1: Float32
    var p2: Float32
    var p3: Float32
    var p4: Float32
    var p5: Float32
    var p6: Float32
    var p7: Float32
    var p8: Float32
    var p9: Float32

    def __init__(out self):
        self.kind = ACMD_NONE
        self.flags = 0
        self.slot = -1
        self.generation = 0
        self.asset_id = -1
        self.bus_id = Int32(BUS_SFX)
        self.p0 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.p4 = 0.0
        self.p5 = 0.0
        self.p6 = 0.0
        self.p7 = 0.0
        self.p8 = 0.0
        self.p9 = 0.0


# ==============================================================================
# 1. 64-Byte Cache-Line-Isolated Lock-Free Vyukov Bounded MPSC Command Ring
# ==============================================================================

comptime _MPSC_ENQ_OFFSET: Int = 0    # Cache line 0 (bytes 0..63): enqueue_pos + 56B pad
comptime _MPSC_DEQ_OFFSET: Int = 64   # Cache line 1 (bytes 64..127): dequeue_pos + 56B pad
comptime _MPSC_CELLS_OFFSET: Int = 128
comptime _MPSC_CELL_STRIDE: Int = 128  # 64B sequence line + 64B AudioCmd payload


struct AudioMpscRing(Copyable, ImplicitlyCopyable, Movable):
    """Lock-free Vyukov bounded MPSC ring with 64-byte cache-line isolation between
    `enqueue_pos`, `dequeue_pos`, and per-slot sequence counters.
    Safe to copy as a handle across `JobPool` worker threads; call `destroy()` once on owner teardown."""
    var raw_block: Int
    var aligned_addr: Int
    var capacity: Int
    var mask: Int

    def __init__(out self):
        self.raw_block = 0
        self.aligned_addr = 0
        self.capacity = 0
        self.mask = 0

    @staticmethod
    def create(requested_capacity: Int = 1024) -> Self:
        var ring = Self()
        var cap = 16
        while cap < requested_capacity:
            cap *= 2
        var total_bytes = _MPSC_CELLS_OFFSET + cap * _MPSC_CELL_STRIDE + 64
        var blk = external_call["calloc", Int](Int(1), total_bytes)
        if blk == 0:
            return ring^
        ring.raw_block = blk
        ring.aligned_addr = (blk + 63) & ~63
        ring.capacity = cap
        ring.mask = cap - 1
        for i in range(cap):
            ring._cell_seq_ptr(i)[unsafe_offset=0].store(Int64(i))
        return ring^

    def destroy(mut self):
        if self.raw_block != 0:
            external_call["free", NoneType](self.raw_block)
            self.raw_block = 0
            self.aligned_addr = 0
            self.capacity = 0
            self.mask = 0

    @always_inline
    def allocated(self) -> Bool:
        return self.aligned_addr != 0

    @always_inline
    def enqueue_head_address(self) -> Int:
        return self.aligned_addr + _MPSC_ENQ_OFFSET

    @always_inline
    def dequeue_tail_address(self) -> Int:
        return self.aligned_addr + _MPSC_DEQ_OFFSET

    @always_inline
    def _enqueue_ptr(self) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=self.aligned_addr + _MPSC_ENQ_OFFSET)

    @always_inline
    def _dequeue_ptr(self) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=self.aligned_addr + _MPSC_DEQ_OFFSET)

    @always_inline
    def _cell_seq_ptr(self, pos: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        var idx = pos & self.mask
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](
            unsafe_from_address=self.aligned_addr + _MPSC_CELLS_OFFSET + idx * _MPSC_CELL_STRIDE
        )

    @always_inline
    def _cell_cmd_ptr(self, pos: Int) -> Pointer[AudioCmd, origin=MutAnyOrigin]:
        var idx = pos & self.mask
        return Pointer[AudioCmd, origin=MutAnyOrigin](
            unsafe_from_address=self.aligned_addr + _MPSC_CELLS_OFFSET + idx * _MPSC_CELL_STRIDE + 64
        )

    def push(self, cmd: AudioCmd) -> Bool:
        """Thread-safe lock-free multi-producer enqueue."""
        if self.aligned_addr == 0:
            return False
        var head = self._enqueue_ptr()
        var pos = Int(head[unsafe_offset=0].load())
        while True:
            var seq = Int(self._cell_seq_ptr(pos)[unsafe_offset=0].load())
            var dif = seq - pos
            if dif == 0:
                var expected = Int64(pos)
                if head[unsafe_offset=0].compare_exchange(expected, Int64(pos + 1)):
                    break
                pos = Int(head[unsafe_offset=0].load())
            elif dif < 0:
                return False
            else:
                pos = Int(head[unsafe_offset=0].load())
        self._cell_cmd_ptr(pos)[unsafe_offset=0] = cmd
        self._cell_seq_ptr(pos)[unsafe_offset=0].store(Int64(pos + 1))
        return True

    def pop(self, mut out_cmd: AudioCmd) -> Bool:
        """Single-consumer RT audio thread dequeue."""
        if self.aligned_addr == 0:
            return False
        var tail = self._dequeue_ptr()
        var pos = Int(tail[unsafe_offset=0].load())
        var seq = Int(self._cell_seq_ptr(pos)[unsafe_offset=0].load())
        if seq != pos + 1:
            return False
        out_cmd = self._cell_cmd_ptr(pos)[unsafe_offset=0]
        self._cell_seq_ptr(pos)[unsafe_offset=0].store(Int64(pos + self.capacity))
        tail[unsafe_offset=0].store(Int64(pos + 1))
        return True

    def approx_len(self) -> Int:
        if self.aligned_addr == 0:
            return 0
        var h = Int(self._enqueue_ptr()[unsafe_offset=0].load())
        var t = Int(self._dequeue_ptr()[unsafe_offset=0].load())
        return max(0, h - t)


# ==============================================================================
# 2. ITU-R BS.1770-4 K-Weighted LUFS Meter & Lookahead True-Peak Limiter
# ==============================================================================

struct _BiquadF64(Copyable, ImplicitlyCopyable, Movable):
    var b0: Float64
    var b1: Float64
    var b2: Float64
    var a1: Float64
    var a2: Float64
    var z1: Float64
    var z2: Float64

    def __init__(out self, b0: Float64 = 1.0, b1: Float64 = 0.0, b2: Float64 = 0.0, a1: Float64 = 0.0, a2: Float64 = 0.0):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.z1 = 0.0
        self.z2 = 0.0

    @always_inline
    def reset(mut self):
        self.z1 = 0.0
        self.z2 = 0.0

    @always_inline
    def process(mut self, x: Float64) -> Float64:
        var y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y


struct KWeightedLufsMeter(Copyable, Movable):
    """ITU-R BS.1770-4 K-weighted loudness meter (Stage 1 pre-filter shelf + Stage 2 RLB high-pass).
    Normalized to stereo full-scale so a 0 dBFS stereo sine reads `-3.01 LUFS` (`itu_multichannel_lufs`
    also exposes unnormalized channel-summed BS.1770 multichannel loudness)."""
    var pre_l: _BiquadF64
    var pre_r: _BiquadF64
    var rlb_l: _BiquadF64
    var rlb_r: _BiquadF64
    var block_frames_target: Int
    var cur_block_frames: Int
    var cur_sum_l: Float64
    var cur_sum_r: Float64
    var window_energy: Array[Float64, 16]  # 16 x 25 ms = 400 ms momentary window
    var window_idx: Int
    var window_filled: Int
    var integrated_energy_sum: Float64
    var integrated_samples: Int

    def __init__(out self, sample_rate: Int = 48000):
        # Exact ITU-R BS.1770-4 Table 1 & Table 2 coefficients @ 48,000 Hz
        var pre = _BiquadF64(
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            -1.69065929318241,
            0.73248077421585,
        )
        var rlb = _BiquadF64(
            1.0,
            -2.0,
            1.0,
            -1.99004745483398,
            0.99007225036621,
        )
        self.pre_l = pre
        self.pre_r = pre
        self.rlb_l = rlb
        self.rlb_r = rlb
        self.block_frames_target = max(64, sample_rate // 40)  # 25 ms sub-blocks
        self.cur_block_frames = 0
        self.cur_sum_l = 0.0
        self.cur_sum_r = 0.0
        self.window_energy = Array[Float64, 16](fill=0.0)
        self.window_idx = 0
        self.window_filled = 0
        self.integrated_energy_sum = 0.0
        self.integrated_samples = 0

    def reset(mut self):
        self.pre_l.reset()
        self.pre_r.reset()
        self.rlb_l.reset()
        self.rlb_r.reset()
        self.cur_block_frames = 0
        self.cur_sum_l = 0.0
        self.cur_sum_r = 0.0
        for i in range(16):
            self.window_energy[i] = 0.0
        self.window_idx = 0
        self.window_filled = 0
        self.integrated_energy_sum = 0.0
        self.integrated_samples = 0

    @always_inline
    def process_frame(mut self, l: Float32, r: Float32):
        var sl = Float64(l) if isfinite(l) else Float64(0.0)
        var sr = Float64(r) if isfinite(r) else Float64(0.0)
        var kl = self.rlb_l.process(self.pre_l.process(sl))
        var kr = self.rlb_r.process(self.pre_r.process(sr))
        if not isfinite(kl):
            kl = 0.0
        if not isfinite(kr):
            kr = 0.0
        var e_pair = 0.5 * (kl * kl + kr * kr)
        self.cur_sum_l += kl * kl
        self.cur_sum_r += kr * kr
        self.cur_block_frames += 1

        # Skip initial 25 ms IIR filter settling transient in integrated accumulator
        if self.window_filled >= 1:
            self.integrated_energy_sum += e_pair
            self.integrated_samples += 1

        if self.cur_block_frames >= self.block_frames_target:
            var mean_e = 0.5 * (self.cur_sum_l + self.cur_sum_r) / Float64(self.cur_block_frames)
            self.window_energy[self.window_idx] = mean_e
            self.window_idx = (self.window_idx + 1) & 15
            if self.window_filled < 16:
                self.window_filled += 1
            self.cur_block_frames = 0
            self.cur_sum_l = 0.0
            self.cur_sum_r = 0.0

    def momentary_lufs(self) -> Float32:
        """400 ms momentary K-weighted loudness (`-3.01 LUFS` for full-scale stereo sine)."""
        if self.window_filled == 0:
            if self.cur_block_frames == 0:
                return -100.0
            var e0 = 0.5 * (self.cur_sum_l + self.cur_sum_r) / Float64(self.cur_block_frames)
            if e0 <= 1.0e-12:
                return -100.0
            return Float32(-0.691 + 10.0 * log10(e0))
        # Skip block 0 if more than 2 blocks exist so filter startup transient is excluded
        var sum_e = Float64(0.0)
        var cnt = self.window_filled
        for i in range(cnt):
            sum_e += self.window_energy[i]
        var mean_e = sum_e / Float64(cnt)
        if mean_e <= 1.0e-12:
            return -100.0
        return Float32(-0.691 + 10.0 * log10(mean_e))

    def integrated_lufs(self) -> Float32:
        if self.integrated_samples <= 0:
            return self.momentary_lufs()
        var mean_e = self.integrated_energy_sum / Float64(self.integrated_samples)
        if mean_e <= 1.0e-12:
            return -100.0
        return Float32(-0.691 + 10.0 * log10(mean_e))

    def itu_multichannel_lufs(self) -> Float32:
        return self.momentary_lufs() + Float32(3.0103)


struct TruePeakLimiter(Copyable, Movable):
    """64-sample lookahead zero-overshoot true-peak brickwall limiter with 4x Hermite
    inter-sample peak detection (`ceiling <= -0.1 dBFS` = `0.988553`)."""
    var delay_l: Array[Float32, LIMITER_LOOKAHEAD]
    var delay_r: Array[Float32, LIMITER_LOOKAHEAD]
    var peak_buf: Array[Float32, LIMITER_LOOKAHEAD]
    var write_idx: Int
    var ceiling: Float32
    var cur_gain: Float32
    var release_coeff: Float32
    var hist_l0: Float32
    var hist_l1: Float32
    var hist_l2: Float32
    var hist_r0: Float32
    var hist_r1: Float32
    var hist_r2: Float32

    def __init__(out self, ceiling_db: Float32 = -0.1, release_ms: Float32 = 80.0, sample_rate: Float32 = 48000.0):
        self.delay_l = Array[Float32, LIMITER_LOOKAHEAD](fill=0.0)
        self.delay_r = Array[Float32, LIMITER_LOOKAHEAD](fill=0.0)
        self.peak_buf = Array[Float32, LIMITER_LOOKAHEAD](fill=0.0)
        self.write_idx = 0
        var safe_cdb = min(Float32(0.0), ceiling_db) if isfinite(ceiling_db) else Float32(-0.1)
        var safe_rel = max(Float32(5.0), release_ms) if isfinite(release_ms) else Float32(80.0)
        var safe_sr = max(Float32(1000.0), sample_rate) if isfinite(sample_rate) else Float32(48000.0)
        self.ceiling = exp(safe_cdb * Float32(0.11512925465))
        if not isfinite(self.ceiling):
            self.ceiling = Float32(0.988553)
        self.cur_gain = 1.0
        self.release_coeff = exp(Float32(-1000.0) / (safe_rel * safe_sr))
        if not isfinite(self.release_coeff):
            self.release_coeff = Float32(0.0)
        self.hist_l0 = 0.0
        self.hist_l1 = 0.0
        self.hist_l2 = 0.0
        self.hist_r0 = 0.0
        self.hist_r1 = 0.0
        self.hist_r2 = 0.0

    def set_ceiling_db(mut self, ceiling_db: Float32):
        var safe_cdb = min(Float32(0.0), ceiling_db) if isfinite(ceiling_db) else Float32(-0.1)
        self.ceiling = exp(safe_cdb * Float32(0.11512925465))
        if not isfinite(self.ceiling):
            self.ceiling = Float32(0.988553)

    @staticmethod
    @always_inline
    def _hermite_4x_peak(xm2: Float32, xm1: Float32, x0: Float32, x1: Float32) -> Float32:
        var pk = max(abs(xm1), abs(x0))
        var c0 = xm1
        var c1 = Float32(0.5) * (x0 - xm2)
        var c2 = xm2 - Float32(2.5) * xm1 + Float32(2.0) * x0 - Float32(0.5) * x1
        var c3 = Float32(0.5) * (x1 - xm2) + Float32(1.5) * (xm1 - x0)
        # Evaluate 3 fractional inter-sample phases t = 0.25, 0.50, 0.75
        var t1 = Float32(0.25)
        var v1 = abs(((c3 * t1 + c2) * t1 + c1) * t1 + c0)
        var t2 = Float32(0.50)
        var v2 = abs(((c3 * t2 + c2) * t2 + c1) * t2 + c0)
        var t3 = Float32(0.75)
        var v3 = abs(((c3 * t3 + c2) * t3 + c1) * t3 + c0)
        return max(pk, max(v1, max(v2, v3)))

    @always_inline
    def process(mut self, in_l: Float32, in_r: Float32) -> Tuple[Float32, Float32]:
        var safe_l = in_l if isfinite(in_l) else Float32(0.0)
        var safe_r = in_r if isfinite(in_r) else Float32(0.0)
        var tp_l = max(abs(safe_l), Self._hermite_4x_peak(self.hist_l0, self.hist_l1, self.hist_l2, safe_l))
        var tp_r = max(abs(safe_r), Self._hermite_4x_peak(self.hist_r0, self.hist_r1, self.hist_r2, safe_r))
        if not isfinite(tp_l):
            tp_l = max(abs(safe_l), Float32(0.0))
        if not isfinite(tp_r):
            tp_r = max(abs(safe_r), Float32(0.0))
        self.hist_l0 = self.hist_l1
        self.hist_l1 = self.hist_l2
        self.hist_l2 = safe_l
        self.hist_r0 = self.hist_r1
        self.hist_r1 = self.hist_r2
        self.hist_r2 = safe_r

        var incoming_tp = max(tp_l, tp_r)

        # Read oldest delayed sample (exiting lookahead line now)
        var out_raw_l = self.delay_l[self.write_idx]
        var out_raw_r = self.delay_r[self.write_idx]
        var out_tp = max(self.peak_buf[self.write_idx], max(abs(out_raw_l), abs(out_raw_r)))

        # Store incoming sample at tail of lookahead line
        self.delay_l[self.write_idx] = safe_l
        self.delay_r[self.write_idx] = safe_r
        self.peak_buf[self.write_idx] = incoming_tp
        self.write_idx = (self.write_idx + 1) & (LIMITER_LOOKAHEAD - 1)

        # Lookahead gain trajectory: ensure smooth linear descent reaches every upcoming peak in time
        var min_target_gain = Float32(1.0)
        var max_slope_step = Float32(0.0)  # steepest required downward step per sample
        var ceil_v = self.ceiling

        if out_tp > ceil_v:
            var g_now = ceil_v / out_tp
            if g_now < self.cur_gain:
                self.cur_gain = g_now
            if g_now < min_target_gain:
                min_target_gain = g_now

        for d in range(LIMITER_LOOKAHEAD):
            var idx = (self.write_idx + d) & (LIMITER_LOOKAHEAD - 1)
            var pk = self.peak_buf[idx]
            if pk > ceil_v:
                var req_g = ceil_v / pk
                if req_g < min_target_gain:
                    min_target_gain = req_g
                if req_g < self.cur_gain:
                    var drop = (self.cur_gain - req_g) / Float32(d + 1)
                    if drop > max_slope_step:
                        max_slope_step = drop

        if max_slope_step > Float32(0.0):
            self.cur_gain = max(min_target_gain, self.cur_gain - max_slope_step)
        else:
            # Smooth exponential release toward 1.0 when horizon is clear
            self.cur_gain = Float32(1.0) - (Float32(1.0) - self.cur_gain) * self.release_coeff
            if self.cur_gain > Float32(0.99999):
                self.cur_gain = 1.0

        # Exact zero-overshoot guarantee
        var eff_g = self.cur_gain
        var raw_abs = max(abs(out_raw_l), abs(out_raw_r))
        if raw_abs * eff_g > ceil_v and raw_abs > Float32(1.0e-9):
            eff_g = ceil_v / raw_abs
            self.cur_gain = eff_g

        return (out_raw_l * eff_g, out_raw_r * eff_g)


# ==============================================================================
# 3. AudioAssets, SpatialEmitter, PhysicalVoice & Woodworth HRTF
# ==============================================================================

comptime ASSET_KIND_PCM16: Int = 0
comptime ASSET_KIND_QOA: Int = 1


struct AudioAssetSlot(Copyable, Movable):
    var valid: Bool
    var kind: Int
    var channels: Int
    var sample_rate: Int
    var total_frames: Int
    var qoa_index: Int
    var pcm: List[Int16]
    var qoa: QoaAsset

    def __init__(out self):
        self.valid = False
        self.kind = ASSET_KIND_PCM16
        self.channels = 1
        self.sample_rate = 48000
        self.total_frames = 0
        self.qoa_index = -1
        self.pcm = List[Int16]()
        self.qoa = QoaAsset()


struct SpatialEmitter(Copyable, ImplicitlyCopyable, Movable):
    """Logical emitter (up to 512 simultaneous instances; top 64 audible get physical voices)."""
    var active: Bool
    var generation: UInt32
    var asset_id: Int
    var bus_id: Int
    var looping: Bool
    var spatial: Bool
    var direct_stereo: Bool
    var priority: Int
    var pos: Vec3
    var vel: Vec3
    var volume: Float32
    var pitch: Float32
    var pan_2d: Float32
    var ref_dist: Float32
    var max_dist: Float32
    var occlusion_gain: Float32
    var occlusion_cutoff_hz: Float32
    var obstruction_factor: Float32
    var virtual_frame_pos: Float64
    var audibility_score: Float32
    var assigned_voice: Int
    # Playback control: per-voice pause, fade-out stop, click-free pause envelope
    var paused: Bool
    var stopping: Bool
    var env_gain: Float32
    var stop_rate: Float32
    # Streaming channel (-1 = in-memory asset), attenuation preset & cone axis
    var stream_ch: Int
    var atten_id: Int
    var cone_dir: Vec3
    # Game-parameter (RTPC) modifiers
    var rtpc_gain: Float32
    var rtpc_pitch: Float32
    var user_cutoff_hz: Float32

    def __init__(out self):
        self.active = False
        self.generation = 1
        self.asset_id = -1
        self.bus_id = BUS_SFX
        self.looping = False
        self.spatial = True
        self.direct_stereo = False
        self.priority = 1
        self.pos = Vec3()
        self.vel = Vec3()
        self.volume = 1.0
        self.pitch = 1.0
        self.pan_2d = 0.0
        self.ref_dist = 15.0
        self.max_dist = 350.0
        self.occlusion_gain = 1.0
        self.occlusion_cutoff_hz = CLEAR_CUTOFF_HZ
        self.obstruction_factor = 0.0
        self.virtual_frame_pos = 0.0
        self.audibility_score = 0.0
        self.assigned_voice = -1
        self.paused = False
        self.stopping = False
        self.env_gain = 1.0
        self.stop_rate = 1.0
        self.stream_ch = -1
        self.atten_id = -1
        self.cone_dir = Vec3()
        self.rtpc_gain = 1.0
        self.rtpc_pitch = 1.0
        self.user_cutoff_hz = CLEAR_CUTOFF_HZ


struct PhysicalVoice(Copyable, Movable):
    """Real-time DSP voice with Woodworth `ItdDelayLine` HRTF, pinna `CytomicSVF`,
    streaming QOA ring cache, and 128-sample click-free steal crossfade."""
    var active: Bool
    var emitter_slot: Int
    var emitter_gen: UInt32
    var initialized_params: Bool
    var cur_gain_l: Float32
    var cur_gain_r: Float32
    var cur_delay_l: Float32
    var cur_delay_r: Float32
    var cur_step: Float64
    var cur_reverb_send: Float32
    var itd_l: ItdDelayLine
    var itd_r: ItdDelayLine
    var pinna_svf_l: CytomicSVF
    var pinna_svf_r: CytomicSVF
    var qoa_cursor: QoaStreamCursor
    var qoa_ring_l: Array[Float32, 64]
    var qoa_ring_r: Array[Float32, 64]
    var qoa_ring_head_frame: Int  # sample index of most recently decoded frame in ring + 1
    var steal_xfade_active: Bool
    var steal_xfade_pos: Int
    var steal_tail_l: Array[Float32, STEAL_XFADE_SAMPLES]
    var steal_tail_r: Array[Float32, STEAL_XFADE_SAMPLES]
    var lp_l: Float32
    var lp_r: Float32

    def __init__(out self):
        self.lp_l = 0.0
        self.lp_r = 0.0
        self.active = False
        self.emitter_slot = -1
        self.emitter_gen = 0
        self.initialized_params = False
        self.cur_gain_l = 0.0
        self.cur_gain_r = 0.0
        self.cur_delay_l = 0.0
        self.cur_delay_r = 0.0
        self.cur_step = 1.0
        self.cur_reverb_send = 0.0
        self.itd_l = ItdDelayLine()
        self.itd_r = ItdDelayLine()
        self.pinna_svf_l = CytomicSVF(18000.0, 0.707, 48000.0)
        self.pinna_svf_r = CytomicSVF(18000.0, 0.707, 48000.0)
        self.qoa_cursor = QoaStreamCursor()
        self.qoa_ring_l = Array[Float32, 64](fill=0.0)
        self.qoa_ring_r = Array[Float32, 64](fill=0.0)
        self.qoa_ring_head_frame = 0
        self.steal_xfade_active = False
        self.steal_xfade_pos = 0
        self.steal_tail_l = Array[Float32, STEAL_XFADE_SAMPLES](fill=0.0)
        self.steal_tail_r = Array[Float32, STEAL_XFADE_SAMPLES](fill=0.0)

    def reset_dsp_for_new_emitter(mut self, slot: Int, gen: UInt32):
        self.active = True
        self.emitter_slot = slot
        self.emitter_gen = gen
        self.initialized_params = False
        self.itd_l = ItdDelayLine()
        self.itd_r = ItdDelayLine()
        self.pinna_svf_l = CytomicSVF(18000.0, 0.707, 48000.0)
        self.pinna_svf_r = CytomicSVF(18000.0, 0.707, 48000.0)
        self.qoa_cursor = QoaStreamCursor()
        self.qoa_ring_head_frame = -1
        self.lp_l = 0.0
        self.lp_r = 0.0


@fieldwise_init
struct HrtfSpatialTargets(Copyable, ImplicitlyCopyable, Movable):
    var gain_l: Float32
    var gain_r: Float32
    var delay_l_samples: Float32
    var delay_r_samples: Float32
    var cutoff_l_hz: Float32
    var cutoff_r_hz: Float32
    var doppler_ratio: Float32
    var reverb_send: Float32


@always_inline
def _hrtf_fin_vec3(v: Vec3, fallback: Vec3 = Vec3()) -> Vec3:
    if isfinite(v.x) and isfinite(v.y) and isfinite(v.z):
        return v
    return fallback


@always_inline
def _hrtf_fin_f32(v: Float32, fallback: Float32) -> Float32:
    return v if isfinite(v) else fallback


def compute_woodworth_hrtf(
    listener_pos: Vec3,
    listener_vel: Vec3,
    listener_fwd: Vec3,
    listener_right: Vec3,
    listener_up: Vec3,
    emitter_pos: Vec3,
    emitter_vel: Vec3,
    ref_dist: Float32 = 15.0,
    max_dist: Float32 = 350.0,
    occlusion_gain: Float32 = 1.0,
    occlusion_cutoff_hz: Float32 = CLEAR_CUTOFF_HZ,
    sample_rate: Float32 = 48000.0,
    atten_override: Float32 = -1.0,
) -> HrtfSpatialTargets:
    """Computes physical Woodworth spherical-head ITD (`a = 0.082 m` -> `29.50` samples @ +90° right),
    frequency-dependent head-shadow ILD, rear/elevation pinna cutoffs (`listener_fwd`, `listener_up`),
    and exact wave-propagation radial Doppler shift (`f' = f * (c + v_l_toward) / (c + v_s_away)`)."""
    var occ_cut = max(Float32(300.0), _hrtf_fin_f32(occlusion_cutoff_hz, CLEAR_CUTOFF_HZ))
    var occ_gain = max(Float32(0.0), _hrtf_fin_f32(occlusion_gain, Float32(0.0)))
    if not (
        isfinite(listener_pos.x) and isfinite(listener_pos.y) and isfinite(listener_pos.z)
        and isfinite(emitter_pos.x) and isfinite(emitter_pos.y) and isfinite(emitter_pos.z)
    ):
        return HrtfSpatialTargets(0.0, 0.0, 0.0, 0.0, occ_cut, occ_cut, 1.0, 0.0)

    var l_vel = _hrtf_fin_vec3(listener_vel, Vec3())
    var e_vel = _hrtf_fin_vec3(emitter_vel, Vec3())
    var l_fwd = _hrtf_fin_vec3(listener_fwd, Vec3(0.0, 0.0, -1.0))
    var l_right = _hrtf_fin_vec3(listener_right, Vec3(1.0, 0.0, 0.0))
    var l_up = _hrtf_fin_vec3(listener_up, Vec3(0.0, 1.0, 0.0))
    var sr = max(Float32(8000.0), min(Float32(384000.0), _hrtf_fin_f32(sample_rate, Float32(48000.0))))

    var rel = emitter_pos - listener_pos
    var dist = rel.length()
    if not isfinite(dist):
        return HrtfSpatialTargets(0.0, 0.0, 0.0, 0.0, occ_cut, occ_cut, 1.0, 0.0)
    if dist < Float32(0.02):
        return HrtfSpatialTargets(
            Float32(0.70710678) * occ_gain,
            Float32(0.70710678) * occ_gain,
            0.0,
            0.0,
            occ_cut,
            occ_cut,
            1.0,
            0.08,
        )

    var inv_d = Float32(1.0) / dist
    var dir = rel * inv_d

    # 1. Inverse-distance rolloff with smooth horizon fade
    var rd = max(Float32(0.5), _hrtf_fin_f32(ref_dist, Float32(15.0)))
    var md = max(rd + Float32(5.0), _hrtf_fin_f32(max_dist, Float32(350.0)))
    var atten = rd / (rd + max(Float32(0.0), dist - rd))
    var horizon = Float32(1.0)
    if dist > md * Float32(0.8):
        horizon = max(Float32(0.0), Float32(1.0) - (dist - md * Float32(0.8)) / (md * Float32(0.2)))
    var base_amp = atten * horizon * occ_gain
    if atten_override >= Float32(0.0) and isfinite(atten_override):
        # Per-sound `AttenuationPreset` curve (distance + cone) replaces the default rolloff.
        base_amp = atten_override * occ_gain
    if not isfinite(base_amp):
        base_amp = Float32(0.0)

    # 2. Woodworth spherical-head ITD: tau = (a / c) * (|theta| + sin(|theta|))
    var sin_az = max(Float32(-1.0), min(Float32(1.0), _hrtf_fin_f32(dir.dot(l_right), Float32(0.0))))
    var abs_sin = abs(sin_az)
    var cos_az = sqrt(max(Float32(0.0), Float32(1.0) - abs_sin * abs_sin))
    var abs_theta = atan2(abs_sin, cos_az)
    var itd_samples = sr * (WOODWORTH_HEAD_RADIUS_M / SPEED_OF_SOUND) * (abs_theta + abs_sin)
    if not isfinite(itd_samples):
        itd_samples = Float32(0.0)

    var delay_l: Float32
    var delay_r: Float32
    if sin_az >= Float32(0.0):
        # Source on right: arrives at right ear first, left ear delayed by ITD
        delay_r = 0.0
        delay_l = itd_samples
    else:
        delay_l = 0.0
        delay_r = itd_samples

    # 3. Head-shadow ILD + constant-power near-field blend
    var near_ear = Float32(0.70710678) + Float32(0.29289322) * abs_sin
    var far_ear = Float32(0.70710678) * (Float32(1.0) - Float32(0.52) * abs_sin)
    var gain_l = (far_ear if sin_az >= Float32(0.0) else near_ear) * base_amp
    var gain_r = (near_ear if sin_az >= Float32(0.0) else far_ear) * base_amp

    # 4. Pinna rear-cone & elevation spectral filtering + head-shadow HF roll-off on far ear
    var fwd_dot = max(Float32(-1.0), min(Float32(1.0), _hrtf_fin_f32(dir.dot(l_fwd), Float32(0.0))))
    var up_dot = max(Float32(-1.0), min(Float32(1.0), _hrtf_fin_f32(dir.dot(l_up), Float32(0.0))))

    var dist_norm = max(Float32(0.0), min(Float32(1.0), (dist - rd) / (md - rd)))
    var air_cutoff = Float32(20000.0) - dist_norm * Float32(11000.0)
    var pinna_mul = Float32(1.0)
    if fwd_dot < Float32(0.0):
        # Rear head/pinna shadow
        pinna_mul *= Float32(1.0) + Float32(0.55) * fwd_dot
    # Subtle elevation cue (overhead sources retain brighter pinna notch ridge than below-chin)
    pinna_mul *= Float32(1.0) + Float32(0.12) * up_dot

    var base_cutoff = min(occ_cut, max(Float32(300.0), air_cutoff * pinna_mul))
    var shadow_cut = max(Float32(300.0), base_cutoff * (Float32(1.0) - Float32(0.58) * abs_sin))
    var cut_l = shadow_cut if sin_az >= Float32(0.0) else base_cutoff
    var cut_r = base_cutoff if sin_az >= Float32(0.0) else shadow_cut

    # 5. Exact radial acoustic Doppler: f' = f * (c + v_listener_toward) / (c + v_source_away)
    var v_l_toward = _hrtf_fin_f32(l_vel.dot(dir), Float32(0.0))
    var v_s_away = _hrtf_fin_f32(e_vel.dot(dir), Float32(0.0))
    var c_num = max(Float32(35.0), SPEED_OF_SOUND + v_l_toward)
    var c_den = max(Float32(35.0), SPEED_OF_SOUND + v_s_away)
    var doppler_raw = c_num / c_den
    var doppler = max(Float32(0.25), min(Float32(4.0), doppler_raw)) if isfinite(doppler_raw) else Float32(1.0)

    var rev_send = Float32(0.06) + dist_norm * Float32(0.28)
    return HrtfSpatialTargets(
        _hrtf_fin_f32(gain_l, Float32(0.0)),
        _hrtf_fin_f32(gain_r, Float32(0.0)),
        _hrtf_fin_f32(delay_l, Float32(0.0)),
        _hrtf_fin_f32(delay_r, Float32(0.0)),
        _hrtf_fin_f32(cut_l, occ_cut),
        _hrtf_fin_f32(cut_r, occ_cut),
        doppler,
        _hrtf_fin_f32(rev_send, Float32(0.06)),
    )


# ==============================================================================
# 4. Topological AudioBus DAG, 3-Band SVF EQ, Compressor & Sidechain Ducking
# ==============================================================================

struct AudioBus(Copyable, Movable):
    """Mix bus node in `AudioServer`'s topological post-order DAG."""
    var id: Int
    var name: String
    var parent_bus: Int  # -1 for root (Master)
    var gain: Float32
    var muted: Bool
    var paused: Bool
    var pause_exempt: Bool  # keeps playing while the global (pause-menu) pause is active
    # 3-band CytomicSVF EQ (exact flat pass-through when gains == 1.0)
    var eq_low_gain: Float32
    var eq_mid_gain: Float32
    var eq_high_gain: Float32
    var svf_low_l: CytomicSVF
    var svf_low_r: CytomicSVF
    var svf_mid_l: CytomicSVF
    var svf_mid_r: CytomicSVF
    var svf_high_l: CytomicSVF
    var svf_high_r: CytomicSVF
    # RMS Compressor
    var comp_enabled: Bool
    var comp_threshold_db: Float32
    var comp_ratio: Float32
    var comp_makeup_gain: Float32
    var comp_env_sq: Float32
    var comp_attack_coeff: Float32
    var comp_release_coeff: Float32
    # Generalized Sidechain Ducking
    var sidechain_bus: Int  # -1 = disabled
    var duck_threshold_db: Float32
    var duck_amount_db: Float32
    var duck_env: Float32
    var duck_attack_coeff: Float32
    var duck_release_coeff: Float32
    # 4 Reverb Aux Zone sends
    var aux_sends: Array[Float32, AUX_REVERB_ZONES]
    # Per-quantum RMS energy (used by sidechain followers & UI meters)
    var last_rms_db: Float32
    var last_peak: Float32
    # Pre-allocated stereo mix buffer (MAX_QUANTUM_FRAMES * 2)
    var buf_lr: List[Float32]

    def __init__(out self, id: Int, name: String, parent_bus: Int = BUS_MASTER, sample_rate: Float32 = 48000.0):
        self.id = id
        self.name = name
        self.parent_bus = parent_bus
        self.gain = 1.0
        self.muted = False
        self.paused = False
        self.pause_exempt = False
        self.eq_low_gain = 1.0
        self.eq_mid_gain = 1.0
        self.eq_high_gain = 1.0
        self.svf_low_l = CytomicSVF(220.0, 0.707, sample_rate)
        self.svf_low_r = CytomicSVF(220.0, 0.707, sample_rate)
        self.svf_mid_l = CytomicSVF(1200.0, 0.9, sample_rate)
        self.svf_mid_r = CytomicSVF(1200.0, 0.9, sample_rate)
        self.svf_high_l = CytomicSVF(4500.0, 0.707, sample_rate)
        self.svf_high_r = CytomicSVF(4500.0, 0.707, sample_rate)
        self.comp_enabled = False
        self.comp_threshold_db = -14.0
        self.comp_ratio = 3.0
        self.comp_makeup_gain = 1.0
        self.comp_env_sq = 0.0
        self.comp_attack_coeff = exp(Float32(-1000.0) / (Float32(8.0) * sample_rate))
        self.comp_release_coeff = exp(Float32(-1000.0) / (Float32(120.0) * sample_rate))
        self.sidechain_bus = -1
        self.duck_threshold_db = -32.0
        self.duck_amount_db = -9.0
        self.duck_env = 1.0
        self.duck_attack_coeff = exp(Float32(-1000.0) / (Float32(12.0) * sample_rate))
        self.duck_release_coeff = exp(Float32(-1000.0) / (Float32(220.0) * sample_rate))
        self.aux_sends = Array[Float32, AUX_REVERB_ZONES](fill=0.0)
        self.last_rms_db = -100.0
        self.last_peak = 0.0
        self.buf_lr = List[Float32](length=MAX_QUANTUM_FRAMES * 2, fill=0.0)

    def set_eq_db(mut self, low_db: Float32, mid_db: Float32, high_db: Float32):
        self.eq_low_gain = exp(low_db * Float32(0.11512925465))
        self.eq_mid_gain = exp(mid_db * Float32(0.11512925465))
        self.eq_high_gain = exp(high_db * Float32(0.11512925465))

    def configure_sidechain(mut self, source_bus: Int, threshold_db: Float32 = -32.0, amount_db: Float32 = -9.0):
        self.sidechain_bus = source_bus
        self.duck_threshold_db = threshold_db
        self.duck_amount_db = min(Float32(0.0), amount_db)


# ==============================================================================
# 5. Zero-Allocation Cycle-Quantized CyclesStemNode
# ==============================================================================

struct CyclesStemNode(Movable):
    """Routes live-coded `CyclesEngine` pattern stems into any `AudioBus` with
    cycle-quantized equal-power stem crossfading and pre-allocated quantum scratch."""
    var active: Bool
    var bus_id: Int
    var gain: Float32
    var engine_a: CyclesEngine
    var engine_b: CyclesEngine
    var has_pending_transition: Bool
    var crossfading: Bool
    var transition_start_cycle: Int
    var crossfade_frames_total: Int
    var crossfade_frames_done: Int
    var scratch_a: List[Float32]
    var scratch_b: List[Float32]
    var send_l: List[Float32]
    var send_r: List[Float32]
    var dsend_l: List[Float32]
    var dsend_r: List[Float32]

    def __init__(out self, sample_rate: Int = 48000, bus_id: Int = BUS_MUSIC):
        self.active = False
        self.bus_id = bus_id
        self.gain = 0.85
        self.engine_a = CyclesEngine(sample_rate)
        self.engine_b = CyclesEngine(sample_rate)
        self.has_pending_transition = False
        self.crossfading = False
        self.transition_start_cycle = 0
        self.crossfade_frames_total = sample_rate
        self.crossfade_frames_done = 0
        self.scratch_a = List[Float32](length=MAX_QUANTUM_FRAMES * 2, fill=0.0)
        self.scratch_b = List[Float32](length=MAX_QUANTUM_FRAMES * 2, fill=0.0)
        self.send_l = List[Float32](length=MAX_QUANTUM_FRAMES, fill=0.0)
        self.send_r = List[Float32](length=MAX_QUANTUM_FRAMES, fill=0.0)
        self.dsend_l = List[Float32](length=MAX_QUANTUM_FRAMES, fill=0.0)
        self.dsend_r = List[Float32](length=MAX_QUANTUM_FRAMES, fill=0.0)

    def load_pattern(mut self, code: String) -> Bool:
        var errs = self.engine_a.load(code)
        if len(errs) > 0:
            return False
        self.active = True
        self.has_pending_transition = False
        self.crossfading = False
        return True

    def queue_quantized_transition_program(
        mut self, var prog: Program, crossfade_cycles: Float32 = 1.0
    ) -> Bool:
        """Queues a pre-parsed `Program` on `engine_b` to crossfade strictly on the next integer cycle boundary."""
        if len(prog.errors) > 0:
            return False
        if not self.active:
            if not self.engine_a.load_program(prog^):
                return False
            self.active = True
            self.has_pending_transition = False
            self.crossfading = False
            return True
        if not self.engine_b.load_program(prog^):
            return False
        var now_f = self.engine_a.now().to_float()
        self.transition_start_cycle = Int(floor(now_f)) + 1
        var sec_per_cycle = Float32(1.0 / max(0.05, self.engine_a.cps))
        self.crossfade_frames_total = max(64, Int(max(Float32(0.125), crossfade_cycles) * sec_per_cycle * Float32(self.engine_a.sample_rate)))
        self.crossfade_frames_done = 0
        self.has_pending_transition = True
        self.crossfading = False
        return True

    def queue_quantized_transition(mut self, code: String, crossfade_cycles: Float32 = 1.0) -> Bool:
        """Parses and queues `code` on `engine_b` (call from non-RT thread)."""
        return self.queue_quantized_transition_program(run_program(code), crossfade_cycles)

    @staticmethod
    def _render_engine_noalloc(
        mut eng: CyclesEngine,
        mut out_lr: List[Float32],
        mut send_l: List[Float32],
        mut send_r: List[Float32],
        mut dsend_l: List[Float32],
        mut dsend_r: List[Float32],
        frames: Int,
    ):
        """Steps `CyclesEngine` directly into pre-allocated `out_lr` (`frames * 2` floats) with reverb & delay routing."""
        for i in range(frames * 2):
            out_lr[i] = 0.0
        for f in range(frames):
            send_l[f] = 0.0
            send_r[f] = 0.0
            dsend_l[f] = 0.0
            dsend_r[f] = 0.0
        var begin = eng.now()
        var end = eng.cycle_at(eng.frames_since + frames)
        var span = (end - begin).to_float()
        for ti in range(len(eng.program.tracks)):
            var root = eng.program.tracks[ti].root
            var evs = eng.program.pat.query_node(root, begin, end)
            for i in range(len(evs)):
                if not evs[i].has_onset():
                    continue
                var off = Int((evs[i].whole_b - begin).to_float() / span * Float64(frames))
                eng.spawn(evs[i], max(0, min(frames - 1, off)))

        var dlen = len(eng.delay_l)
        var dframes = max(1, min(dlen - 1, Int(eng.delay_time * eng.sr)))

        var vi = 0
        while vi < len(eng.voices):
            var v = eng.voices[vi].copy()
            for f in range(v.wait, frames):
                if v.t >= v.life:
                    break
                var x = eng._sample(v)
                v.t += 1
                var l = x * v.gain_l
                var r = x * v.gain_r
                out_lr[f * 2] += l
                out_lr[f * 2 + 1] += r
                if v.room > Float32(0.0):
                    send_l[f] += l * v.room
                    send_r[f] += r * v.room
                if v.delay_send > Float32(0.0):
                    dsend_l[f] += l * v.delay_send
                    dsend_r[f] += r * v.delay_send
            v.wait = 0
            if v.t >= v.life:
                _ = eng.voices.pop(vi)
            else:
                eng.voices[vi] = v^
                vi += 1

        for f in range(frames):
            var wet = eng.reverb.process_send(send_l[f], send_r[f])
            var read = (eng.delay_idx - dframes + dlen) % dlen
            var dl = eng.delay_l[read]
            var dr = eng.delay_r[read]
            eng.delay_l[eng.delay_idx] = dsend_l[f] + dr * eng.delay_fb
            eng.delay_r[eng.delay_idx] = dsend_r[f] + dl * eng.delay_fb
            eng.delay_idx = (eng.delay_idx + 1) % dlen
            out_lr[f * 2] = soft_clip_tanh((out_lr[f * 2] + wet[0] * Float32(1.4) + dl) * eng.master)
            out_lr[f * 2 + 1] = soft_clip_tanh((out_lr[f * 2 + 1] + wet[1] * Float32(1.4) + dr) * eng.master)

        eng.frames_since += frames

    def mix_into_bus(mut self, mut bus_lr: List[Float32], frames: Int):
        if not self.active or frames <= 0:
            return
        if self.has_pending_transition and not self.crossfading:
            var cur_cyc = self.engine_a.now().to_float()
            if cur_cyc >= Float64(self.transition_start_cycle):
                self.engine_b.base_cycle = self.engine_a.now()
                self.engine_b.frames_since = 0
                self.crossfading = True
                self.has_pending_transition = False
                self.crossfade_frames_done = 0

        Self._render_engine_noalloc(
            self.engine_a, self.scratch_a, self.send_l, self.send_r, self.dsend_l, self.dsend_r, frames
        )
        if not self.crossfading:
            var g = self.gain
            for f in range(frames):
                bus_lr[f * 2] += self.scratch_a[f * 2] * g
                bus_lr[f * 2 + 1] += self.scratch_a[f * 2 + 1] * g
            return

        Self._render_engine_noalloc(
            self.engine_b, self.scratch_b, self.send_l, self.send_r, self.dsend_l, self.dsend_r, frames
        )
        var total = max(1, self.crossfade_frames_total)
        var g = self.gain
        for f in range(frames):
            var t = min(Float32(1.0), Float32(self.crossfade_frames_done) / Float32(total))
            var angle = t * (DSP_PI * Float32(0.5))
            var wa = cos(angle) * g
            var wb = sin(angle) * g
            bus_lr[f * 2] += self.scratch_a[f * 2] * wa + self.scratch_b[f * 2] * wb
            bus_lr[f * 2 + 1] += self.scratch_a[f * 2 + 1] * wa + self.scratch_b[f * 2 + 1] * wb
            if self.crossfade_frames_done < total:
                self.crossfade_frames_done += 1

        if self.crossfade_frames_done >= total:
            # Swap incoming stem into active engine
            var tmp = self.engine_a^
            self.engine_a = self.engine_b^
            self.engine_b = tmp^
            self.engine_b.voices.clear()
            self.crossfading = False


# ==============================================================================
# 6. Heap-Pinned Non-Moving `AudioRtEngine` Core & Race-Free `AudioServer`
# ==============================================================================

comptime _TELEM_QUANTUMS: Int = 0
comptime _TELEM_STOLEN: Int = 8
comptime _TELEM_LOGICAL: Int = 16
comptime _TELEM_PHYSICAL: Int = 24
comptime _TELEM_MOMENTARY_LUFS: Int = 32
comptime _TELEM_INTEGRATED_LUFS: Int = 40
comptime _TELEM_MUSIC_DUCK_ENV: Int = 48
comptime _TELEM_BACKEND: Int = 56
comptime _TELEM_RECONNECTS: Int = 64
comptime _TELEM_FAULT_INJECT: Int = 72
comptime _TELEM_STREAM_UNDERRUNS: Int = 80
comptime _TELEM_DEVICE_LOST: Int = 88
comptime _TELEM_SLAB_BYTES: Int = 128


comptime _RETIRE_RAW_PTR: UInt32 = 1
comptime _RETIRE_BOX_CLIP: UInt32 = 2
comptime _RETIRE_BOX_BUS: UInt32 = 3
comptime _RETIRE_BOX_STEM: UInt32 = 4
comptime _RETIRE_BOX_TRANSITION: UInt32 = 5


struct _StemTransitionPayload(Movable):
    var program: Program
    var crossfade_cycles: Float32

    def __init__(out self, var program: Program, crossfade_cycles: Float32):
        self.program = program^
        self.crossfade_cycles = crossfade_cycles

    def take_program(deinit self) -> Program:
        return self.program^


def _heap_box_alloc[
    T: Movable & Deinitable
](var val: T) -> Pointer[T, MutAnyOrigin]:
    var nbytes = max(64, size_of[T]())
    var raw = external_call["calloc", Int](Int(1), nbytes)
    var p = Pointer[T, MutAnyOrigin](unsafe_from_address=raw)
    __get_address_as_uninit_lvalue(p._mlir_value) = val^
    return p


def _heap_box_take_nofree[T: Movable & Deinitable](
    p: Pointer[T, MutAnyOrigin]
) -> T:
    """Moves `T` out of a heap box WITHOUT calling `free()`; caller must retire `Int(p)` to main thread."""
    var owned = __get_address_as_owned_value(p._mlir_value)
    return owned^


def _heap_box_put[T: Movable & Deinitable](
    p: Pointer[T, MutAnyOrigin], var val: T
):
    """Places a retired `T` into an already-emptied heap box for main-thread destruction + `free()`."""
    __get_address_as_uninit_lvalue(p._mlir_value) = val^


def _heap_box_free[T: Movable & Deinitable](p: Pointer[T, MutAnyOrigin]):
    var addr = Int(p)
    if addr != 0:
        var owned = __get_address_as_owned_value(p._mlir_value)
        _ = owned^
        external_call["free", NoneType](addr)


@always_inline
def _pack_ptr_into_cmd(mut cmd: AudioCmd, addr: Int):
    var u = UInt64(addr)
    cmd.p0 = bitcast[DType.float32](UInt32(u & 0xFFFFFFFF))
    cmd.p1 = bitcast[DType.float32](UInt32((u >> 32) & 0xFFFFFFFF))


@always_inline
def _unpack_ptr_from_cmd(cmd: AudioCmd) -> Int:
    var lo = UInt64(bitcast[DType.uint32](cmd.p0))
    var hi = UInt64(bitcast[DType.uint32](cmd.p1))
    return Int(lo | (hi << 32))


def _audio_rt_thread_entry(rt_engine_addr: Int) abi("C") -> Int:
    ref rt = Pointer[AudioRtEngine, MutAnyOrigin](
        unsafe_from_address=rt_engine_addr
    )[unsafe_offset=0]
    rt._rt_thread_loop()
    return 0


struct AudioRtEngine(Movable):
    """Heap-pinned real-time mixer core whose memory address NEVER moves across
    `AudioServer.__moveinit__` relocations. Exclusively mutated by the RT mixer
    thread (or `render_offline_quantum` when no RT thread is running)."""
    var sample_rate: Int
    var sr_f: Float32
    var quantum_frames: Int

    var cmd_ring: AudioMpscRing
    var retire_ring: AudioMpscRing
    var rt_clips: List[AudioAssetSlot]
    var rt_qoa: List[QoaAsset]
    var emitters: List[SpatialEmitter]
    var voices: List[PhysicalVoice]
    var buses: List[AudioBus]
    var post_order: List[Int]
    var reverb_zones: List[StereoReverbFDN]
    var aux_gains: Array[Float32, AUX_REVERB_ZONES]
    var aux_send_l: List[Float32]
    var aux_send_r: List[Float32]
    var cycles_stem: List[CyclesStemNode]
    var lufs_meter: KWeightedLufsMeter
    var limiter: TruePeakLimiter

    var dc_enabled: Bool
    var dc_x_l: Float32
    var dc_y_l: Float32
    var dc_x_r: Float32
    var dc_y_r: Float32

    var listener_pos: Vec3
    var listener_vel: Vec3
    var listener_fwd: Vec3
    var listener_right: Vec3
    var listener_up: Vec3

    var quantums_rendered_local: UInt64
    var voices_stolen_local: UInt64
    var active_logical_local: Int
    var active_physical_local: Int

    var telemetry_atomic: Int
    var running_atomic: Int

    var pulse: PulseClient
    var alsa_fd: Int
    var alsa_period_frames: Int
    var alsa_pending_frames: Int
    var alsa_pending_offset: Int
    var alsa_xruns: Int
    var alsa_poll_buf: List[UInt8]
    var headless_rt_mode: Bool
    var rt_pcm_i16: List[Int16]
    var rt_float_lr: List[Float32]

    # Playback control / sound design / streaming
    var global_paused: Bool
    var bus_pause_eff: List[Bool]
    var rt_atten: List[AttenuationPreset]
    var stream_ctrl_addr: Int
    var stream_underruns_local: Int

    # Output robustness (backend state machine owned by the RT thread while it runs)
    var backend: Int
    var preferred_device: String
    var allow_alsa_fallback: Bool
    var backoff_ms: Int
    var next_attempt_s: Float64
    var reconnects_local: Int
    var device_lost_local: Int

    def __init__(
        out self,
        sample_rate: Int = 48000,
        quantum_frames: Int = 256,
        ring_capacity: Int = 2048,
    ):
        self.sample_rate = max(8000, sample_rate)
        self.sr_f = Float32(self.sample_rate)
        self.quantum_frames = max(32, min(MAX_QUANTUM_FRAMES, quantum_frames))

        self.cmd_ring = AudioMpscRing.create(ring_capacity)
        self.retire_ring = AudioMpscRing.create(max(512, ring_capacity))
        self.rt_clips = List[AudioAssetSlot]()
        self.rt_qoa = List[QoaAsset]()

        self.emitters = List[SpatialEmitter](capacity=MAX_LOGICAL_EMITTERS)
        for _ in range(MAX_LOGICAL_EMITTERS):
            self.emitters.append(SpatialEmitter())

        self.voices = List[PhysicalVoice](capacity=MAX_PHYSICAL_VOICES)
        for _ in range(MAX_PHYSICAL_VOICES):
            self.voices.append(PhysicalVoice())

        self.buses = List[AudioBus]()
        self.buses.append(AudioBus(BUS_MASTER, "Master", -1, self.sr_f))
        self.buses.append(AudioBus(BUS_SFX, "SFX", BUS_MASTER, self.sr_f))
        self.buses.append(AudioBus(BUS_MUSIC, "Music", BUS_MASTER, self.sr_f))
        self.buses.append(AudioBus(BUS_VOICE, "Voice", BUS_MASTER, self.sr_f))
        self.buses.append(
            AudioBus(BUS_AMBIENCE, "Ambience", BUS_MASTER, self.sr_f)
        )
        self.buses.append(AudioBus(BUS_UI, "UI", BUS_MASTER, self.sr_f))
        self.buses[BUS_MUSIC].configure_sidechain(BUS_VOICE, -32.0, -9.0)

        self.post_order = List[Int]()
        self.reverb_zones = List[StereoReverbFDN]()
        self.reverb_zones.append(StereoReverbFDN(0.58, 0.48))
        self.reverb_zones.append(StereoReverbFDN(0.84, 0.28))
        self.reverb_zones.append(StereoReverbFDN(0.89, 0.20))
        self.reverb_zones.append(StereoReverbFDN(0.74, 0.34))
        self.aux_gains = Array[Float32, AUX_REVERB_ZONES](fill=0.35)
        self.aux_send_l = List[Float32](
            length=AUX_REVERB_ZONES * MAX_QUANTUM_FRAMES, fill=0.0
        )
        self.aux_send_r = List[Float32](
            length=AUX_REVERB_ZONES * MAX_QUANTUM_FRAMES, fill=0.0
        )

        self.cycles_stem = List[CyclesStemNode]()
        self.lufs_meter = KWeightedLufsMeter(self.sample_rate)
        self.limiter = TruePeakLimiter(-0.1, 80.0, self.sr_f)

        self.dc_enabled = True
        self.dc_x_l = 0.0
        self.dc_y_l = 0.0
        self.dc_x_r = 0.0
        self.dc_y_r = 0.0

        self.listener_pos = Vec3(0.0, 0.0, 0.0)
        self.listener_vel = Vec3(0.0, 0.0, 0.0)
        self.listener_fwd = Vec3(0.0, 0.0, -1.0)
        self.listener_right = Vec3(1.0, 0.0, 0.0)
        self.listener_up = Vec3(0.0, 1.0, 0.0)

        self.quantums_rendered_local = 0
        self.voices_stolen_local = 0
        self.active_logical_local = 0
        self.active_physical_local = 0

        self.telemetry_atomic = external_call["calloc", Int](
            Int(1), _TELEM_SLAB_BYTES
        )
        self.running_atomic = external_call["calloc", Int](Int(1), 64)

        self.pulse = PulseClient()
        self.alsa_fd = -1
        self.alsa_period_frames = self.quantum_frames
        self.alsa_pending_frames = 0
        self.alsa_pending_offset = 0
        self.alsa_xruns = 0
        self.alsa_poll_buf = List[UInt8](length=8, fill=0)
        self.headless_rt_mode = False
        self.rt_pcm_i16 = List[Int16](length=MAX_QUANTUM_FRAMES * 2, fill=0)
        self.rt_float_lr = List[Float32](
            length=MAX_QUANTUM_FRAMES * 2, fill=0.0
        )

        self.global_paused = False
        self.bus_pause_eff = List[Bool](length=64, fill=False)
        self.rt_atten = List[AttenuationPreset]()
        self.stream_ctrl_addr = 0
        self.stream_underruns_local = 0

        self.backend = AUDIO_BACKEND_NONE
        self.preferred_device = ""
        self.allow_alsa_fallback = True
        self.backoff_ms = RECONNECT_BACKOFF_MIN_MS
        self.next_attempt_s = 0.0
        self.reconnects_local = 0
        self.device_lost_local = 0

        self.rebuild_topology()
        self._publish_telemetry()

    @always_inline
    def _running_ptr(self) -> Pointer[Atomic[Int64], MutAnyOrigin]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.running_atomic
        )

    @always_inline
    def _telem_ptr(self, byte_off: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.telemetry_atomic + byte_off
        )

    def _publish_telemetry(self):
        if self.telemetry_atomic == 0:
            return
        self._telem_ptr(_TELEM_QUANTUMS)[unsafe_offset=0].store(
            Int64(self.quantums_rendered_local)
        )
        self._telem_ptr(_TELEM_STOLEN)[unsafe_offset=0].store(
            Int64(self.voices_stolen_local)
        )
        self._telem_ptr(_TELEM_LOGICAL)[unsafe_offset=0].store(
            Int64(self.active_logical_local)
        )
        self._telem_ptr(_TELEM_PHYSICAL)[unsafe_offset=0].store(
            Int64(self.active_physical_local)
        )
        var mom_bits = Int64(
            bitcast[DType.uint32](self.lufs_meter.momentary_lufs())
        )
        var int_bits = Int64(
            bitcast[DType.uint32](self.lufs_meter.integrated_lufs())
        )
        var duck_v = (
            self.buses[BUS_MUSIC].duck_env
            if len(self.buses) > BUS_MUSIC
            else Float32(1.0)
        )
        var duck_bits = Int64(bitcast[DType.uint32](duck_v))
        self._telem_ptr(_TELEM_MOMENTARY_LUFS)[unsafe_offset=0].store(mom_bits)
        self._telem_ptr(_TELEM_INTEGRATED_LUFS)[unsafe_offset=0].store(int_bits)
        self._telem_ptr(_TELEM_MUSIC_DUCK_ENV)[unsafe_offset=0].store(duck_bits)
        self._telem_ptr(_TELEM_STREAM_UNDERRUNS)[unsafe_offset=0].store(
            Int64(self.stream_underruns_local)
        )

    def rebuild_topology(mut self):
        """Computes topological post-order of `AudioBus` DAG (children & sidechain sources before parents/targets)."""
        self.post_order.clear()
        var n = len(self.buses)
        var state = List[Int](length=n, fill=0)

        for root in range(n):
            if state[root] != 0:
                continue
            var stack_bus = List[Int]()
            var stack_phase = List[Int]()
            stack_bus.append(root)
            stack_phase.append(0)

            while len(stack_bus) > 0:
                var top_i = len(stack_bus) - 1
                var b = stack_bus[top_i]
                var ph = stack_phase[top_i]
                if ph == 0:
                    if state[b] == 2:
                        _ = stack_bus.pop()
                        _ = stack_phase.pop()
                        continue
                    state[b] = 1
                    stack_phase[top_i] = 1
                    var sc = self.buses[b].sidechain_bus
                    if sc >= 0 and sc < n and state[sc] == 0:
                        stack_bus.append(sc)
                        stack_phase.append(0)
                    for child in range(n - 1, -1, -1):
                        if (
                            self.buses[child].parent_bus == b
                            and state[child] == 0
                        ):
                            stack_bus.append(child)
                            stack_phase.append(0)
                else:
                    _ = stack_bus.pop()
                    _ = stack_phase.pop()
                    if state[b] != 2:
                        state[b] = 2
                        self.post_order.append(b)

    @always_inline
    def _retire_ptr(mut self, kind: UInt32, addr: Int):
        """Queues a raw pointer or boxed payload for main-thread reclamation (zero `free()` on RT thread)."""
        if addr == 0:
            return
        var rcmd = AudioCmd()
        rcmd.kind = kind
        _pack_ptr_into_cmd(rcmd, addr)
        _ = self.retire_ring.push(rcmd)

    def drain_retired_main_thread(mut self) -> Int:
        """Called ONLY from the main/caller thread (or after RT thread joins) to free retired payloads."""
        var freed = 0
        var rcmd = AudioCmd()
        while self.retire_ring.pop(rcmd):
            var ptr_addr = _unpack_ptr_from_cmd(rcmd)
            if ptr_addr == 0:
                continue
            if rcmd.kind == _RETIRE_RAW_PTR:
                external_call["free", NoneType](ptr_addr)
            elif rcmd.kind == _RETIRE_BOX_CLIP:
                _heap_box_free[AudioAssetSlot](
                    Pointer[AudioAssetSlot, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif rcmd.kind == _RETIRE_BOX_BUS:
                _heap_box_free[AudioBus](
                    Pointer[AudioBus, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif rcmd.kind == _RETIRE_BOX_STEM:
                _heap_box_free[CyclesStemNode](
                    Pointer[CyclesStemNode, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif rcmd.kind == _RETIRE_BOX_TRANSITION:
                _heap_box_free[_StemTransitionPayload](
                    Pointer[_StemTransitionPayload, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            freed += 1
        return freed

    def _drain_commands(mut self):
        var cmd = AudioCmd()
        while self.cmd_ring.pop(cmd):
            var k = cmd.kind
            if k == CMD_INSTALL_CLIP:
                var ptr_addr = _unpack_ptr_from_cmd(cmd)
                if ptr_addr != 0:
                    var p = Pointer[AudioAssetSlot, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                    var slot = _heap_box_take_nofree[AudioAssetSlot](p)
                    var aid = max(0, Int(cmd.asset_id))
                    while len(self.rt_clips) <= aid:
                        self.rt_clips.append(AudioAssetSlot())
                    if slot.kind == ASSET_KIND_QOA and slot.valid:
                        var q_idx = len(self.rt_qoa)
                        self.rt_qoa.append(slot.qoa^)
                        slot.qoa = QoaAsset()
                        slot.qoa_index = q_idx
                    if self.rt_clips[aid].valid:
                        var old_slot = self.rt_clips[aid].copy()
                        _heap_box_put[AudioAssetSlot](p, old_slot^)
                        self.rt_clips[aid] = slot^
                        self._retire_ptr(_RETIRE_BOX_CLIP, ptr_addr)
                    else:
                        self.rt_clips[aid] = slot^
                        self._retire_ptr(_RETIRE_RAW_PTR, ptr_addr)
            elif k == CMD_ADD_BUS:
                var ptr_addr = _unpack_ptr_from_cmd(cmd)
                if ptr_addr != 0:
                    var p = Pointer[AudioBus, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                    var new_bus = _heap_box_take_nofree[AudioBus](p)
                    var bid = max(0, Int(cmd.bus_id))
                    while len(self.buses) < bid:
                        var pad_id = len(self.buses)
                        self.buses.append(
                            AudioBus(pad_id, "Pad", BUS_MASTER, self.sr_f)
                        )
                    if bid == len(self.buses):
                        self.buses.append(new_bus^)
                    else:
                        self.buses[bid] = new_bus^
                    self._retire_ptr(_RETIRE_RAW_PTR, ptr_addr)
                    self.rebuild_topology()
            elif k == CMD_CONFIG_BUS:
                var b = Int(cmd.bus_id)
                if b >= 0 and b < len(self.buses):
                    var sub = cmd.flags
                    if sub == _BUS_CFG_GAIN_MUTE:
                        if cmd.p0 >= Float32(0.0):
                            self.buses[b].gain = cmd.p0
                        if cmd.p1 >= Float32(0.0):
                            self.buses[b].muted = cmd.p1 > Float32(0.5)
                    elif sub == _BUS_CFG_EQ_DB:
                        self.buses[b].set_eq_db(cmd.p0, cmd.p1, cmd.p2)
                    elif sub == _BUS_CFG_COMPRESSOR:
                        self.buses[b].comp_enabled = cmd.p0 > Float32(0.5)
                        self.buses[b].comp_threshold_db = cmd.p1
                        self.buses[b].comp_ratio = max(Float32(1.0), cmd.p2)
                        self.buses[b].comp_makeup_gain = max(
                            Float32(0.0), cmd.p3
                        )
                    elif sub == _BUS_CFG_SIDECHAIN:
                        self.buses[b].configure_sidechain(
                            Int(cmd.slot), cmd.p0, cmd.p1
                        )
                        self.rebuild_topology()
                    elif sub == _BUS_CFG_AUX_SEND:
                        var z = Int(cmd.slot)
                        if z >= 0 and z < AUX_REVERB_ZONES:
                            self.buses[b].aux_sends[z] = max(
                                Float32(0.0), cmd.p0
                            )
                    elif sub == _BUS_CFG_PAUSE:
                        if cmd.p0 >= Float32(0.0):
                            self.buses[b].paused = cmd.p0 > Float32(0.5)
                        if cmd.p1 >= Float32(0.0):
                            self.buses[b].pause_exempt = cmd.p1 > Float32(0.5)
            elif k == CMD_CONFIG_MASTER:
                var sub = cmd.flags
                if sub == _MASTER_CFG_DC_BLOCKER:
                    self.dc_enabled = cmd.p0 > Float32(0.5)
                elif sub == _MASTER_CFG_AUX_GAIN:
                    var z = Int(cmd.slot)
                    if z >= 0 and z < AUX_REVERB_ZONES:
                        self.aux_gains[z] = max(Float32(0.0), cmd.p0)
                elif sub == _MASTER_CFG_LIMITER_CEIL:
                    self.limiter.set_ceiling_db(cmd.p0)
            elif k == CMD_ADD_STEM:
                var ptr_addr = _unpack_ptr_from_cmd(cmd)
                if ptr_addr != 0:
                    var p = Pointer[CyclesStemNode, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                    var node = _heap_box_take_nofree[CyclesStemNode](p)
                    var sid = max(0, Int(cmd.slot))
                    while len(self.cycles_stem) < sid:
                        self.cycles_stem.append(
                            CyclesStemNode(self.sample_rate, BUS_MUSIC)
                        )
                    if sid == len(self.cycles_stem):
                        self.cycles_stem.append(node^)
                    else:
                        self.cycles_stem[sid] = node^
                    self._retire_ptr(_RETIRE_RAW_PTR, ptr_addr)
            elif k == CMD_TRANSITION_STEM:
                var ptr_addr = _unpack_ptr_from_cmd(cmd)
                if ptr_addr != 0:
                    var sid = Int(cmd.slot)
                    if sid >= 0 and sid < len(self.cycles_stem):
                        var p = Pointer[_StemTransitionPayload, MutAnyOrigin](
                            unsafe_from_address=ptr_addr
                        )
                        var tr = _heap_box_take_nofree[_StemTransitionPayload](p)
                        var xfade = tr.crossfade_cycles
                        _ = self.cycles_stem[sid].queue_quantized_transition_program(
                            tr^.take_program(), xfade
                        )
                        self._retire_ptr(_RETIRE_RAW_PTR, ptr_addr)
                    else:
                        self._retire_ptr(_RETIRE_BOX_TRANSITION, ptr_addr)
            elif k == ACMD_PLAY:
                var s = Int(cmd.slot)
                if (
                    s >= 0
                    and s < MAX_LOGICAL_EMITTERS
                    and cmd.asset_id >= 0
                    and Int(cmd.asset_id) < len(self.rt_clips)
                ):
                    ref em = self.emitters[s]
                    em.active = True
                    em.generation = cmd.generation
                    em.asset_id = Int(cmd.asset_id)
                    em.bus_id = max(
                        0, min(len(self.buses) - 1, Int(cmd.bus_id))
                    )
                    em.looping = (cmd.flags & AFLAG_LOOPING) != 0
                    em.spatial = (cmd.flags & AFLAG_SPATIAL) != 0
                    em.direct_stereo = (cmd.flags & AFLAG_DIRECT_STEREO) != 0
                    if em.spatial:
                        em.pos = Vec3(cmd.p0, cmd.p1, cmd.p2)
                        em.vel = Vec3(cmd.p3, cmd.p4, cmd.p5)
                        em.ref_dist = max(Float32(0.5), cmd.p8)
                        em.max_dist = max(
                            em.ref_dist + Float32(5.0), floor(cmd.p9)
                        )
                        em.priority = max(
                            1,
                            Int(
                                (cmd.p9 - floor(cmd.p9)) * Float32(1000.0)
                                + Float32(0.5)
                            ),
                        )
                    else:
                        em.pan_2d = cmd.p0
                        em.priority = 2
                    em.volume = max(Float32(0.0), cmd.p6) if not (isnan(cmd.p6) or isinf(cmd.p6)) else Float32(1.0)
                    em.pitch = _sanitize_pitch(cmd.p7)
                    em.occlusion_gain = 1.0
                    em.occlusion_cutoff_hz = CLEAR_CUTOFF_HZ
                    em.obstruction_factor = 0.0
                    em.virtual_frame_pos = 0.0
                    em.paused = False
                    em.stopping = False
                    em.env_gain = 1.0
                    em.stop_rate = 1.0
                    em.atten_id = -1
                    em.cone_dir = Vec3()
                    em.rtpc_gain = 1.0
                    em.rtpc_pitch = 1.0
                    em.user_cutoff_hz = CLEAR_CUTOFF_HZ
                    em.stream_ch = -1
                    if (cmd.flags & AFLAG_STREAM) != 0 and self.stream_ctrl_addr != 0:
                        var sch = Int((cmd.flags >> 8) & 0xFF)
                        if sch < STREAM_MAX_CHANNELS:
                            var st = stream_load(self.stream_ctrl_addr, sch, _SC_STATE)
                            if st == STREAM_STATE_ARMED:
                                stream_store(self.stream_ctrl_addr, sch, _SC_STATE, STREAM_STATE_ACTIVE)
                                em.stream_ch = sch
            elif k == ACMD_STOP:
                var s = Int(cmd.slot)
                if s >= 0 and s < MAX_LOGICAL_EMITTERS:
                    if self.emitters[s].generation == cmd.generation:
                        var fade_s = cmd.p0
                        if fade_s > Float32(1.0e-4) and isfinite(fade_s) and self.emitters[s].active:
                            self.emitters[s].stopping = True
                            self.emitters[s].stop_rate = Float32(1.0) / max(Float32(1.0), fade_s * self.sr_f)
                        else:
                            self.emitters[s].active = False
            elif k == ACMD_PAUSE:
                var s = Int(cmd.slot)
                if (
                    s >= 0
                    and s < MAX_LOGICAL_EMITTERS
                    and self.emitters[s].generation == cmd.generation
                ):
                    self.emitters[s].paused = cmd.p0 > Float32(0.5)
            elif k == ACMD_GLOBAL_PAUSE:
                self.global_paused = cmd.p0 > Float32(0.5)
            elif k == ACMD_SET_VOICE_PARAMS:
                var s = Int(cmd.slot)
                if (
                    s >= 0
                    and s < MAX_LOGICAL_EMITTERS
                    and self.emitters[s].active
                    and self.emitters[s].generation == cmd.generation
                ):
                    if cmd.p0 >= Float32(0.0) and isfinite(cmd.p0):
                        self.emitters[s].rtpc_gain = min(Float32(16.0), cmd.p0)
                    if cmd.p1 > Float32(0.0) and isfinite(cmd.p1):
                        self.emitters[s].rtpc_pitch = _sanitize_pitch(cmd.p1)
                    if cmd.p2 > Float32(0.0) and isfinite(cmd.p2):
                        self.emitters[s].user_cutoff_hz = max(Float32(20.0), min(CLEAR_CUTOFF_HZ, cmd.p2))
            elif k == CMD_INSTALL_ATTEN:
                var ptr_addr = _unpack_ptr_from_cmd(cmd)
                if ptr_addr != 0:
                    var p = Pointer[AttenuationPreset, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                    var preset = _heap_box_take_nofree[AttenuationPreset](p)
                    var pid = max(0, Int(cmd.asset_id))
                    while len(self.rt_atten) <= pid:
                        self.rt_atten.append(AttenuationPreset())
                    self.rt_atten[pid] = preset^
                    self._retire_ptr(_RETIRE_RAW_PTR, ptr_addr)
            elif k == ACMD_SET_EMITTER_ATTEN:
                var s = Int(cmd.slot)
                if (
                    s >= 0
                    and s < MAX_LOGICAL_EMITTERS
                    and self.emitters[s].generation == cmd.generation
                ):
                    var pid = Int(cmd.asset_id)
                    if pid >= 0 and pid < len(self.rt_atten):
                        self.emitters[s].atten_id = pid
                        self.emitters[s].ref_dist = self.rt_atten[pid].min_dist
                        self.emitters[s].max_dist = self.rt_atten[pid].max_dist
                    else:
                        self.emitters[s].atten_id = -1
                    self.emitters[s].cone_dir = Vec3(cmd.p0, cmd.p1, cmd.p2)
            elif k == ACMD_UPDATE_EMITTER:
                var s = Int(cmd.slot)
                if (
                    s >= 0
                    and s < MAX_LOGICAL_EMITTERS
                    and self.emitters[s].active
                    and self.emitters[s].generation == cmd.generation
                ):
                    self.emitters[s].pos = Vec3(cmd.p0, cmd.p1, cmd.p2)
                    self.emitters[s].vel = Vec3(cmd.p3, cmd.p4, cmd.p5)
                    if cmd.p6 >= Float32(0.0) and not (isnan(cmd.p6) or isinf(cmd.p6)):
                        self.emitters[s].volume = cmd.p6
                    if cmd.p7 > Float32(0.0) and not isnan(cmd.p7):
                        self.emitters[s].pitch = _sanitize_pitch(cmd.p7)
            elif k == ACMD_SET_OCCLUSION:
                var s = Int(cmd.slot)
                if (
                    s >= 0
                    and s < MAX_LOGICAL_EMITTERS
                    and self.emitters[s].active
                    and self.emitters[s].generation == cmd.generation
                ):
                    self.emitters[s].occlusion_gain = max(
                        Float32(0.0), min(Float32(1.0), cmd.p0)
                    )
                    self.emitters[s].occlusion_cutoff_hz = max(
                        Float32(120.0), min(CLEAR_CUTOFF_HZ, cmd.p1)
                    )
                    self.emitters[s].obstruction_factor = max(
                        Float32(0.0), min(Float32(1.0), cmd.p2)
                    )
            elif k == ACMD_SET_LISTENER_POS_VEL:
                self.listener_pos = Vec3(cmd.p0, cmd.p1, cmd.p2)
                self.listener_vel = Vec3(cmd.p3, cmd.p4, cmd.p5)
            elif k == ACMD_SET_LISTENER_BASIS_FWD_RIGHT:
                var fwd_v = Vec3(cmd.p0, cmd.p1, cmd.p2)
                var right_v = Vec3(cmd.p3, cmd.p4, cmd.p5)
                var up_v = Vec3(cmd.p6, cmd.p7, cmd.p8)
                if fwd_v.length_squared() > Float32(1.0e-6):
                    self.listener_fwd = fwd_v.normalized()
                if right_v.length_squared() > Float32(1.0e-6):
                    self.listener_right = right_v.normalized()
                if up_v.length_squared() > Float32(1.0e-6):
                    self.listener_up = up_v.normalized()
            elif k == ACMD_SET_BUS_GAIN:
                var b = Int(cmd.bus_id)
                if b >= 0 and b < len(self.buses):
                    self.buses[b].gain = max(Float32(0.0), cmd.p0)
            elif k == ACMD_SET_ROOM_PROBE:
                var z = max(0, min(AUX_REVERB_ZONES - 1, Int(cmd.slot)))
                self.reverb_zones[z].feedback = max(
                    Float32(0.1), min(Float32(0.93), cmd.p0)
                )
                self.reverb_zones[z].damp = max(
                    Float32(0.05), min(Float32(0.95), cmd.p1)
                )
                self.aux_gains[z] = max(
                    Float32(0.0), min(Float32(1.5), cmd.p2)
                )
            elif k == ACMD_STOP_ALL and cmd.p0 > Float32(1.0e-4) and isfinite(cmd.p0):
                var rate = Float32(1.0) / max(Float32(1.0), cmd.p0 * self.sr_f)
                for i in range(MAX_LOGICAL_EMITTERS):
                    if self.emitters[i].active:
                        self.emitters[i].stopping = True
                        self.emitters[i].stop_rate = rate
            elif k == ACMD_STOP_ALL:
                for i in range(MAX_LOGICAL_EMITTERS):
                    self.emitters[i].active = False
                for v in range(MAX_PHYSICAL_VOICES):
                    self.voices[v].active = False
                self.active_logical_local = 0
                self.active_physical_local = 0
                self._telem_ptr(_TELEM_LOGICAL)[unsafe_offset=0].store(0)
                self._telem_ptr(_TELEM_PHYSICAL)[unsafe_offset=0].store(0)

    def _free_unconsumed_ring_payloads(mut self):
        var cmd = AudioCmd()
        while self.cmd_ring.pop(cmd):
            var ptr_addr = _unpack_ptr_from_cmd(cmd)
            if ptr_addr == 0:
                continue
            if cmd.kind == CMD_INSTALL_CLIP:
                _heap_box_free[AudioAssetSlot](
                    Pointer[AudioAssetSlot, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif cmd.kind == CMD_ADD_BUS:
                _heap_box_free[AudioBus](
                    Pointer[AudioBus, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif cmd.kind == CMD_ADD_STEM:
                _heap_box_free[CyclesStemNode](
                    Pointer[CyclesStemNode, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif cmd.kind == CMD_TRANSITION_STEM:
                _heap_box_free[_StemTransitionPayload](
                    Pointer[_StemTransitionPayload, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )
            elif cmd.kind == CMD_INSTALL_ATTEN:
                _heap_box_free[AttenuationPreset](
                    Pointer[AttenuationPreset, MutAnyOrigin](
                        unsafe_from_address=ptr_addr
                    )
                )

    def destroy_engine_allocations(mut self):
        self._free_unconsumed_ring_payloads()
        _ = self.drain_retired_main_thread()
        self.cmd_ring.destroy()
        self.retire_ring.destroy()
        if self.telemetry_atomic != 0:
            external_call["free", NoneType](self.telemetry_atomic)
            self.telemetry_atomic = 0
        if self.running_atomic != 0:
            external_call["free", NoneType](self.running_atomic)
            self.running_atomic = 0

    @staticmethod
    @always_inline
    def _pcm_at(
        asset: AudioAssetSlot, idx: Int, ch: Int, looping: Bool
    ) -> Float32:
        var n = asset.total_frames
        if n <= 0:
            return 0.0
        var c = min(ch, asset.channels - 1)
        var f = idx
        if looping:
            f = f % n
            if f < 0:
                f += n
        else:
            if f < 0:
                f = 0
            elif f >= n:
                return 0.0
        return Float32(asset.pcm[f * asset.channels + c]) * Float32(
            1.0 / 32768.0
        )

    @staticmethod
    @always_inline
    def _sample_pcm_hermite(
        asset: AudioAssetSlot, pos: Float64, ch: Int, looping: Bool
    ) -> Float32:
        var ipos = Int(floor(pos))
        var frac = Float32(pos - Float64(ipos))
        var ym1 = Self._pcm_at(asset, ipos - 1, ch, looping)
        var y0 = Self._pcm_at(asset, ipos, ch, looping)
        var y1 = Self._pcm_at(asset, ipos + 1, ch, looping)
        var y2 = Self._pcm_at(asset, ipos + 2, ch, looping)
        var c0 = y0
        var c1 = Float32(0.5) * (y1 - ym1)
        var c2 = ym1 - Float32(2.5) * y0 + Float32(2.0) * y1 - Float32(0.5) * y2
        var c3 = Float32(0.5) * (y2 - ym1) + Float32(1.5) * (y0 - y1)
        return ((c3 * frac + c2) * frac + c1) * frac + c0

    @staticmethod
    def _ensure_qoa_ring(
        mut pv: PhysicalVoice,
        qoa_bytes: List[UInt8],
        total_frames: Int,
        needed_frame: Int,
        looping: Bool,
    ):
        """Streams QOA slices into `pv.qoa_ring_l/r` with zero heap allocations."""
        var total = total_frames
        if total <= 0:
            return
        var target = needed_frame + 2
        if not looping and target > total:
            target = total

        if (
            pv.qoa_ring_head_frame < 0
            or needed_frame - 2 < pv.qoa_ring_head_frame - 60
            or needed_frame > pv.qoa_ring_head_frame + 64
        ):
            var start_f = max(0, needed_frame - 2)
            if looping:
                start_f = start_f % total
            _ = pv.qoa_cursor.seek(qoa_bytes, start_f)
            # Ring head must name the frame the cursor will decode next (it was `needed - 2`
            # even when clamped to 0, shifting playback by 2 frames); clear stale history.
            pv.qoa_ring_head_frame = max(0, needed_frame - 2)
            for i in range(64):
                pv.qoa_ring_l[i] = 0.0
                pv.qoa_ring_r[i] = 0.0

        while pv.qoa_ring_head_frame <= target:
            var l16 = Int16(0)
            var r16 = Int16(0)
            if not pv.qoa_cursor.read_next_frame_i16(qoa_bytes, l16, r16):
                if looping:
                    _ = pv.qoa_cursor.seek(qoa_bytes, 0)
                    if not pv.qoa_cursor.read_next_frame_i16(
                        qoa_bytes, l16, r16
                    ):
                        break
                else:
                    var slot = pv.qoa_ring_head_frame & 63
                    pv.qoa_ring_l[slot] = 0.0
                    pv.qoa_ring_r[slot] = 0.0
                    pv.qoa_ring_head_frame += 1
                    continue
            var slot = pv.qoa_ring_head_frame & 63
            pv.qoa_ring_l[slot] = Float32(l16) * Float32(1.0 / 32768.0)
            pv.qoa_ring_r[slot] = Float32(r16) * Float32(1.0 / 32768.0)
            pv.qoa_ring_head_frame += 1

    @staticmethod
    @always_inline
    def _sample_qoa_hermite(
        mut pv: PhysicalVoice,
        qoa_bytes: List[UInt8],
        total_frames: Int,
        pos: Float64,
        looping: Bool,
    ) -> Tuple[Float32, Float32]:
        var ipos = Int(floor(pos))
        Self._ensure_qoa_ring(pv, qoa_bytes, total_frames, ipos, looping)
        var frac = Float32(pos - Float64(ipos))
        var i0 = (ipos - 1) & 63
        var i1 = ipos & 63
        var i2 = (ipos + 1) & 63
        var i3 = (ipos + 2) & 63

        var lm1 = pv.qoa_ring_l[i0]
        var l0 = pv.qoa_ring_l[i1]
        var l1 = pv.qoa_ring_l[i2]
        var l2 = pv.qoa_ring_l[i3]
        var lc0 = l0
        var lc1 = Float32(0.5) * (l1 - lm1)
        var lc2 = (
            lm1 - Float32(2.5) * l0 + Float32(2.0) * l1 - Float32(0.5) * l2
        )
        var lc3 = Float32(0.5) * (l2 - lm1) + Float32(1.5) * (l0 - l1)
        var out_l = ((lc3 * frac + lc2) * frac + lc1) * frac + lc0

        var rm1 = pv.qoa_ring_r[i0]
        var r0 = pv.qoa_ring_r[i1]
        var r1 = pv.qoa_ring_r[i2]
        var r2 = pv.qoa_ring_r[i3]
        var rc0 = r0
        var rc1 = Float32(0.5) * (r1 - rm1)
        var rc2 = (
            rm1 - Float32(2.5) * r0 + Float32(2.0) * r1 - Float32(0.5) * r2
        )
        var rc3 = Float32(0.5) * (r2 - rm1) + Float32(1.5) * (r0 - r1)
        var out_r = ((rc3 * frac + rc2) * frac + rc1) * frac + rc0
        return (out_l, out_r)

    @staticmethod
    @always_inline
    def _stream_at(ring: Int, wp: Int, total: Int, looping: Bool, idx: Int, lr: Int) -> Float32:
        if ring == 0 or idx < 0 or idx >= wp or idx < wp - STREAM_RING_FRAMES:
            return 0.0
        if not looping and idx >= total:
            return 0.0
        return Pointer[Float32, MutAnyOrigin](unsafe_from_address=ring)[
            unsafe_offset=(idx & STREAM_RING_MASK) * 2 + lr
        ]

    @staticmethod
    @always_inline
    def _sample_stream_hermite(
        ring: Int, wp: Int, total: Int, looping: Bool, pos: Float64
    ) -> Tuple[Float32, Float32]:
        var ipos = Int(floor(pos))
        var frac = Float32(pos - Float64(ipos))
        var lm1 = Self._stream_at(ring, wp, total, looping, ipos - 1, 0)
        var l0 = Self._stream_at(ring, wp, total, looping, ipos, 0)
        var l1 = Self._stream_at(ring, wp, total, looping, ipos + 1, 0)
        var l2 = Self._stream_at(ring, wp, total, looping, ipos + 2, 0)
        var rm1 = Self._stream_at(ring, wp, total, looping, ipos - 1, 1)
        var r0 = Self._stream_at(ring, wp, total, looping, ipos, 1)
        var r1 = Self._stream_at(ring, wp, total, looping, ipos + 1, 1)
        var r2 = Self._stream_at(ring, wp, total, looping, ipos + 2, 1)
        var lc1 = Float32(0.5) * (l1 - lm1)
        var lc2 = lm1 - Float32(2.5) * l0 + Float32(2.0) * l1 - Float32(0.5) * l2
        var lc3 = Float32(0.5) * (l2 - lm1) + Float32(1.5) * (l0 - l1)
        var rc1 = Float32(0.5) * (r1 - rm1)
        var rc2 = rm1 - Float32(2.5) * r0 + Float32(2.0) * r1 - Float32(0.5) * r2
        var rc3 = Float32(0.5) * (r2 - rm1) + Float32(1.5) * (r0 - r1)
        return (
            ((lc3 * frac + lc2) * frac + lc1) * frac + l0,
            ((rc3 * frac + rc2) * frac + rc1) * frac + r0,
        )

    def _compute_bus_pause(mut self):
        """Resolves hierarchical bus pause + global (pause-menu) pause with per-bus exemption."""
        var n = len(self.buses)
        if len(self.bus_pause_eff) < n:
            self.bus_pause_eff.resize(n, False)
        for b in range(n):
            var paused = False
            var exempt = False
            var cur = b
            var guard = 0
            while cur >= 0 and cur < n and guard < 32:
                if self.buses[cur].paused:
                    paused = True
                if self.buses[cur].pause_exempt:
                    exempt = True
                cur = self.buses[cur].parent_bus
                guard += 1
            self.bus_pause_eff[b] = paused or (self.global_paused and not exempt)

    @always_inline
    def _emitter_paused(self, s: Int) -> Bool:
        var b = self.emitters[s].bus_id
        var bp = False
        if b >= 0 and b < len(self.buses) and b < len(self.bus_pause_eff):
            bp = self.bus_pause_eff[b]
        return self.emitters[s].paused or bp

    @always_inline
    def _emitter_frozen(self, s: Int) -> Bool:
        return (
            self._emitter_paused(s)
            and not self.emitters[s].stopping
            and self.emitters[s].env_gain <= Float32(0.0)
        )

    def _emitter_atten(self, s: Int) -> Float32:
        """Preset attenuation (distance curve x cone) or `-1` when the emitter uses the default curve."""
        var pid = self.emitters[s].atten_id
        if pid < 0 or pid >= len(self.rt_atten):
            return -1.0
        return self.rt_atten[pid].evaluate(
            self.emitters[s].pos, self.emitters[s].cone_dir, self.listener_pos
        )

    def _release_dead_streams(mut self):
        """Marks stream channels whose owning emitter stopped/was replaced as RELEASING."""
        var ctrl = self.stream_ctrl_addr
        if ctrl == 0:
            return
        for c in range(STREAM_MAX_CHANNELS):
            if stream_load(ctrl, c, _SC_STATE) != STREAM_STATE_ACTIVE:
                continue
            var os = Int(stream_load(ctrl, c, _SC_OWNER_SLOT))
            var og = UInt32(stream_load(ctrl, c, _SC_OWNER_GEN))
            var keep = (
                os >= 0
                and os < MAX_LOGICAL_EMITTERS
                and self.emitters[os].active
                and self.emitters[os].generation == og
                and self.emitters[os].stream_ch == c
            )
            if not keep:
                stream_store(ctrl, c, _SC_STATE, STREAM_STATE_RELEASING)

    def _capture_steal_tail(mut self, v_idx: Int):
        """Renders 128 samples of an outgoing physical voice into `steal_tail_l/r` before voice steal."""
        ref pv = self.voices[v_idx]
        var es = pv.emitter_slot
        if not pv.active or es < 0 or es >= MAX_LOGICAL_EMITTERS:
            return
        ref em = self.emitters[es]
        if (
            em.asset_id < 0
            or em.asset_id >= len(self.rt_clips)
            or not self.rt_clips[em.asset_id].valid
        ):
            return
        ref asset = self.rt_clips[em.asset_id]

        var pos = em.virtual_frame_pos
        var step = pv.cur_step
        var gl = pv.cur_gain_l * em.env_gain
        var gr = pv.cur_gain_r * em.env_gain
        var dl = pv.cur_delay_l
        var dr = pv.cur_delay_r
        var s_ctrl = self.stream_ctrl_addr
        var s_ch = em.stream_ch
        var s_ring = 0
        var s_wp = 0
        var s_total = asset.total_frames
        if s_ch >= 0 and s_ctrl != 0:
            s_ring = Int(stream_load(s_ctrl, s_ch, _SC_RING))
            s_wp = Int(stream_load(s_ctrl, s_ch, _SC_WRITE))

        for i in range(STEAL_XFADE_SAMPLES):
            var raw_l: Float32
            var raw_r: Float32
            if asset.kind == ASSET_KIND_STREAM:
                var sr2 = Self._sample_stream_hermite(s_ring, s_wp, s_total, em.looping, pos)
                raw_l = sr2[0]
                raw_r = sr2[1]
            elif (
                asset.kind == ASSET_KIND_QOA
                and asset.qoa_index >= 0
                and asset.qoa_index < len(self.rt_qoa)
            ):
                var qr = Self._sample_qoa_hermite(
                    pv,
                    self.rt_qoa[asset.qoa_index].bytes,
                    asset.total_frames,
                    pos,
                    em.looping,
                )
                raw_l = qr[0]
                raw_r = qr[1]
            else:
                raw_l = Self._sample_pcm_hermite(asset, pos, 0, em.looping)
                raw_r = (
                    Self._sample_pcm_hermite(asset, pos, 1, em.looping)
                    if asset.channels > 1
                    else raw_l
                )
            pos += step

            if em.spatial:
                var mono = (
                    (raw_l + raw_r) * Float32(0.5)
                    if asset.channels > 1
                    else raw_l
                )
                pv.itd_l.push(mono)
                pv.itd_r.push(mono)
                var sl = pv.pinna_svf_l.process(pv.itd_l.read_delayed(dl)) * gl
                var sr = pv.pinna_svf_r.process(pv.itd_r.read_delayed(dr)) * gr
                pv.steal_tail_l[i] = sl
                pv.steal_tail_r[i] = sr
            else:
                pv.steal_tail_l[i] = raw_l * gl
                pv.steal_tail_r[i] = raw_r * gr

        pv.steal_xfade_active = True
        pv.steal_xfade_pos = 0

    def _virtualize_voices(mut self, frames: Int):
        """Scores up to 512 logical emitters, maps top 64 audible emitters to physical voices
        (with 128-sample click-free steal crossfade), and advances virtualized clocks."""
        for v in range(MAX_PHYSICAL_VOICES):
            if not self.voices[v].active:
                continue
            var es = self.voices[v].emitter_slot
            if (
                es < 0
                or es >= MAX_LOGICAL_EMITTERS
                or not self.emitters[es].active
                or self.emitters[es].generation != self.voices[v].emitter_gen
            ):
                self.voices[v].active = False
                self.voices[v].emitter_slot = -1

        self._compute_bus_pause()
        var top_slots = Array[Int, MAX_PHYSICAL_VOICES](fill=-1)
        var top_scores = Array[Float32, MAX_PHYSICAL_VOICES](fill=-1.0)
        var top_count = 0
        var active_log = 0

        for s in range(MAX_LOGICAL_EMITTERS):
            self.emitters[s].assigned_voice = -1
            if not self.emitters[s].active:
                continue
            var aid = self.emitters[s].asset_id
            if (
                aid < 0
                or aid >= len(self.rt_clips)
                or not self.rt_clips[aid].valid
            ):
                self.emitters[s].active = False
                continue
            if not self.emitters[s].looping and self.emitters[
                s
            ].virtual_frame_pos >= Float64(self.rt_clips[aid].total_frames):
                self.emitters[s].active = False
                continue
            if self.rt_clips[aid].kind == ASSET_KIND_STREAM and self.emitters[s].stream_ch < 0:
                # Streamed asset without a stream channel (pool exhausted): nothing to play.
                self.emitters[s].active = False
                continue

            active_log += 1
            if self._emitter_frozen(s):
                # Fully faded-out paused emitter: keeps its position, holds no physical voice.
                self.emitters[s].audibility_score = 0.0
                continue
            var score = self.emitters[s].volume * self.emitters[s].rtpc_gain * Float32(
                max(1, self.emitters[s].priority)
            )
            if self.emitters[s].spatial:
                var d = (self.emitters[s].pos - self.listener_pos).length()
                var preset_att = self._emitter_atten(s)
                var att: Float32
                if preset_att >= Float32(0.0):
                    att = preset_att
                else:
                    var rd = max(Float32(0.5), self.emitters[s].ref_dist)
                    att = rd / (rd + max(Float32(0.0), d - rd))
                score *= att * self.emitters[s].occlusion_gain
                if d >= self.emitters[s].max_dist:
                    score = 0.0
            self.emitters[s].audibility_score = score

            if score <= Float32(1.0e-5):
                continue

            if top_count < MAX_PHYSICAL_VOICES:
                top_slots[top_count] = s
                top_scores[top_count] = score
                top_count += 1
            else:
                var min_i = 0
                var min_sc = top_scores[0]
                for k in range(1, MAX_PHYSICAL_VOICES):
                    if top_scores[k] < min_sc:
                        min_sc = top_scores[k]
                        min_i = k
                if score > min_sc * Float32(1.05):
                    top_slots[min_i] = s
                    top_scores[min_i] = score

        self.active_logical_local = active_log

        var selected_in_top = Array[Bool, MAX_PHYSICAL_VOICES](fill=False)
        var voice_kept = Array[Bool, MAX_PHYSICAL_VOICES](fill=False)

        for k in range(top_count):
            var es = top_slots[k]
            for v in range(MAX_PHYSICAL_VOICES):
                if (
                    self.voices[v].active
                    and self.voices[v].emitter_slot == es
                    and self.voices[v].emitter_gen
                    == self.emitters[es].generation
                ):
                    selected_in_top[k] = True
                    voice_kept[v] = True
                    self.emitters[es].assigned_voice = v
                    break

        for k in range(top_count):
            if selected_in_top[k]:
                continue
            var es = top_slots[k]
            var chosen_v = -1
            for v in range(MAX_PHYSICAL_VOICES):
                if not self.voices[v].active:
                    chosen_v = v
                    break
            if chosen_v < 0:
                for v in range(MAX_PHYSICAL_VOICES):
                    if not voice_kept[v]:
                        chosen_v = v
                        break
            if chosen_v >= 0:
                if self.voices[chosen_v].active:
                    self._capture_steal_tail(chosen_v)
                    self.voices_stolen_local += 1
                self.voices[chosen_v].reset_dsp_for_new_emitter(
                    es, self.emitters[es].generation
                )
                voice_kept[chosen_v] = True
                self.emitters[es].assigned_voice = chosen_v

        var phys_cnt = 0
        for v in range(MAX_PHYSICAL_VOICES):
            if self.voices[v].active and not voice_kept[v]:
                self._capture_steal_tail(v)
                self.voices[v].active = False
                self.voices[v].emitter_slot = -1
            if self.voices[v].active:
                phys_cnt += 1
        self.active_physical_local = phys_cnt

        for s in range(MAX_LOGICAL_EMITTERS):
            if (
                not self.emitters[s].active
                or self.emitters[s].assigned_voice >= 0
            ):
                continue
            if self._emitter_frozen(s):
                continue
            if self.emitters[s].stopping:
                self.emitters[s].env_gain -= self.emitters[s].stop_rate * Float32(frames)
                if self.emitters[s].env_gain <= Float32(0.0):
                    self.emitters[s].env_gain = 0.0
                    self.emitters[s].active = False
                    continue
            elif self._emitter_paused(s):
                self.emitters[s].env_gain = 0.0  # inaudible anyway: freeze immediately
                continue
            else:
                self.emitters[s].env_gain = 1.0
            ref asset = self.rt_clips[self.emitters[s].asset_id]
            var step = (
                Float64(self.emitters[s].pitch * self.emitters[s].rtpc_pitch)
                * Float64(asset.sample_rate)
                / Float64(self.sample_rate)
            )
            self.emitters[s].virtual_frame_pos += step * Float64(frames)
            var sch = self.emitters[s].stream_ch
            if sch >= 0 and self.stream_ctrl_addr != 0:
                # Streams keep a monotonic position (the producer maps it into the file).
                stream_publish_read(self.stream_ctrl_addr, sch, self.emitters[s].virtual_frame_pos)
                if not self.emitters[s].looping and self.emitters[s].virtual_frame_pos >= Float64(asset.total_frames):
                    self.emitters[s].active = False
            elif self.emitters[s].looping and asset.total_frames > 0:
                self.emitters[s].virtual_frame_pos = _wrap_loop_pos(
                    self.emitters[s].virtual_frame_pos,
                    Float64(asset.total_frames),
                )
            elif self.emitters[s].virtual_frame_pos >= Float64(
                asset.total_frames
            ):
                self.emitters[s].active = False

    def render_quantum(mut self, mut out_lr: List[Float32], frames: Int = 256):
        """Renders `frames` stereo frames deterministically into `out_lr` (`frames * 2` floats)."""
        var n_frames = max(1, min(MAX_QUANTUM_FRAMES, frames))
        if len(out_lr) < n_frames * 2:
            out_lr.resize(n_frames * 2, 0.0)

        self._drain_commands()
        self._virtualize_voices(n_frames)

        var n_buses = len(self.buses)
        for b in range(n_buses):
            for i in range(n_frames * 2):
                self.buses[b].buf_lr[i] = 0.0
        for z in range(AUX_REVERB_ZONES):
            var z_off = z * MAX_QUANTUM_FRAMES
            for f in range(n_frames):
                self.aux_send_l[z_off + f] = 0.0
                self.aux_send_r[z_off + f] = 0.0

        var inv_frames = Float32(1.0) / Float32(n_frames)

        for v in range(MAX_PHYSICAL_VOICES):
            if not self.voices[v].active:
                if self.voices[v].steal_xfade_active:
                    ref master_buf = self.buses[BUS_SFX].buf_lr
                    for f in range(n_frames):
                        var xp = self.voices[v].steal_xfade_pos
                        if xp >= STEAL_XFADE_SAMPLES:
                            self.voices[v].steal_xfade_active = False
                            break
                        var fade = Float32(1.0) - (
                            Float32(xp) + Float32(0.5)
                        ) / Float32(STEAL_XFADE_SAMPLES)
                        master_buf[f * 2] += (
                            self.voices[v].steal_tail_l[xp] * fade
                        )
                        master_buf[f * 2 + 1] += (
                            self.voices[v].steal_tail_r[xp] * fade
                        )
                        self.voices[v].steal_xfade_pos = xp + 1
                continue

            var es = self.voices[v].emitter_slot
            var preset_att = self._emitter_atten(es)
            var em_paused = self._emitter_paused(es)
            var s_ctrl = self.stream_ctrl_addr
            ref em = self.emitters[es]
            ref asset = self.rt_clips[em.asset_id]
            var bus_idx = max(0, min(n_buses - 1, em.bus_id))
            var is_stream = asset.kind == ASSET_KIND_STREAM and em.stream_ch >= 0 and s_ctrl != 0
            var s_ring = 0
            var s_wp = 0
            if is_stream:
                s_ring = Int(stream_load(s_ctrl, em.stream_ch, _SC_RING))
                s_wp = Int(stream_load(s_ctrl, em.stream_ch, _SC_WRITE))

            # Click-free envelope: pause/resume ramps over PAUSE_FADE_FRAMES, stop over the fade time.
            var env = em.env_gain
            var env_tgt = Float32(0.0) if (em.stopping or em_paused) else Float32(1.0)
            var env_rate = em.stop_rate if em.stopping else Float32(1.0 / Float64(PAUSE_FADE_FRAMES))
            var lp_a = Float32(-1.0)
            if em.user_cutoff_hz < Float32(19000.0):
                lp_a = Float32(1.0) - exp(Float32(-2.0) * DSP_PI * em.user_cutoff_hz / self.sr_f)

            var tgt_gl: Float32
            var tgt_gr: Float32
            var tgt_dl = Float32(0.0)
            var tgt_dr = Float32(0.0)
            var tgt_step = (
                Float64(em.pitch * em.rtpc_pitch)
                * Float64(asset.sample_rate)
                / Float64(self.sample_rate)
            )
            var tgt_rev = Float32(0.0)

            if em.spatial:
                var hrtf = compute_woodworth_hrtf(
                    self.listener_pos,
                    self.listener_vel,
                    self.listener_fwd,
                    self.listener_right,
                    self.listener_up,
                    em.pos,
                    em.vel,
                    em.ref_dist,
                    em.max_dist,
                    em.occlusion_gain,
                    em.occlusion_cutoff_hz,
                    self.sr_f,
                    preset_att,
                )
                tgt_gl = hrtf.gain_l * em.volume
                tgt_gr = hrtf.gain_r * em.volume
                tgt_dl = hrtf.delay_l_samples
                tgt_dr = hrtf.delay_r_samples
                tgt_step *= Float64(hrtf.doppler_ratio)
                tgt_rev = hrtf.reverb_send
                self.voices[v].pinna_svf_l.set_params(
                    hrtf.cutoff_l_hz, 0.707, self.sr_f
                )
                self.voices[v].pinna_svf_r.set_params(
                    hrtf.cutoff_r_hz, 0.707, self.sr_f
                )
            elif em.direct_stereo:
                tgt_gl = em.volume
                tgt_gr = em.volume
            else:
                var pg = equal_power_gains(em.pan_2d)
                tgt_gl = pg[0] * em.volume
                tgt_gr = pg[1] * em.volume
            tgt_gl *= em.rtpc_gain
            tgt_gr *= em.rtpc_gain

            if not self.voices[v].initialized_params:
                self.voices[v].cur_gain_l = tgt_gl
                self.voices[v].cur_gain_r = tgt_gr
                self.voices[v].cur_delay_l = tgt_dl
                self.voices[v].cur_delay_r = tgt_dr
                self.voices[v].cur_step = tgt_step
                self.voices[v].cur_reverb_send = tgt_rev
                self.voices[v].initialized_params = True

            var gl = self.voices[v].cur_gain_l
            var gr = self.voices[v].cur_gain_r
            var dl = self.voices[v].cur_delay_l
            var dr = self.voices[v].cur_delay_r
            var stp = self.voices[v].cur_step

            var d_gl = (tgt_gl - gl) * inv_frames
            var d_gr = (tgt_gr - gr) * inv_frames
            var d_dl = (tgt_dl - dl) * inv_frames
            var d_dr = (tgt_dr - dr) * inv_frames
            var d_stp = (tgt_step - stp) * Float64(inv_frames)

            var pos = em.virtual_frame_pos
            var rev_z3_off = 3 * MAX_QUANTUM_FRAMES
            var total_f = Float64(asset.total_frames)

            for f in range(n_frames):
                var out_l = Float32(0.0)
                var out_r = Float32(0.0)

                var hold = env <= Float32(0.0) and env_tgt <= Float32(0.0) and not em.stopping
                if (em.looping or pos < total_f) and not hold:
                    var raw_l: Float32
                    var raw_r: Float32
                    if is_stream:
                        var ip = Int(floor(pos)) + 2
                        if ip >= s_wp and (em.looping or ip < asset.total_frames):
                            self.stream_underruns_local += 1
                        var sr3 = Self._sample_stream_hermite(
                            s_ring, s_wp, asset.total_frames, em.looping, pos
                        )
                        raw_l = sr3[0]
                        raw_r = sr3[1]
                    elif (
                        asset.kind == ASSET_KIND_QOA
                        and asset.qoa_index >= 0
                        and asset.qoa_index < len(self.rt_qoa)
                    ):
                        var qr = Self._sample_qoa_hermite(
                            self.voices[v],
                            self.rt_qoa[asset.qoa_index].bytes,
                            asset.total_frames,
                            pos,
                            em.looping,
                        )
                        raw_l = qr[0]
                        raw_r = qr[1]
                    else:
                        raw_l = Self._sample_pcm_hermite(
                            asset, pos, 0, em.looping
                        )
                        raw_r = (
                            Self._sample_pcm_hermite(asset, pos, 1, em.looping)
                            if asset.channels > 1
                            else raw_l
                        )

                    if em.spatial:
                        var mono = (
                            (raw_l + raw_r) * Float32(0.5)
                            if asset.channels > 1
                            else raw_l
                        )
                        self.voices[v].itd_l.push(mono)
                        self.voices[v].itd_r.push(mono)
                        out_l = (
                            self.voices[v].pinna_svf_l.process(
                                self.voices[v].itd_l.read_delayed(dl)
                            )
                            * gl
                        )
                        out_r = (
                            self.voices[v].pinna_svf_r.process(
                                self.voices[v].itd_r.read_delayed(dr)
                            )
                            * gr
                        )
                        if tgt_rev > Float32(0.001):
                            self.aux_send_l[rev_z3_off + f] += out_l * tgt_rev
                            self.aux_send_r[rev_z3_off + f] += out_r * tgt_rev
                    else:
                        out_l = raw_l * gl
                        out_r = raw_r * gr

                    if lp_a > Float32(0.0):
                        self.voices[v].lp_l += lp_a * (out_l - self.voices[v].lp_l)
                        self.voices[v].lp_r += lp_a * (out_r - self.voices[v].lp_r)
                        out_l = self.voices[v].lp_l
                        out_r = self.voices[v].lp_r
                    out_l *= env
                    out_r *= env

                    pos += stp
                    if em.looping and total_f > 0.0 and not is_stream:
                        pos = _wrap_loop_pos(pos, total_f)

                if env < env_tgt:
                    env = min(env_tgt, env + env_rate)
                elif env > env_tgt:
                    env = max(env_tgt, env - env_rate)

                if self.voices[v].steal_xfade_active:
                    var xp = self.voices[v].steal_xfade_pos
                    if xp < STEAL_XFADE_SAMPLES:
                        var w_new = (Float32(xp) + Float32(0.5)) / Float32(
                            STEAL_XFADE_SAMPLES
                        )
                        var w_old = Float32(1.0) - w_new
                        out_l = (
                            self.voices[v].steal_tail_l[xp] * w_old
                            + out_l * w_new
                        )
                        out_r = (
                            self.voices[v].steal_tail_r[xp] * w_old
                            + out_r * w_new
                        )
                        self.voices[v].steal_xfade_pos = xp + 1
                    else:
                        self.voices[v].steal_xfade_active = False

                self.buses[bus_idx].buf_lr[f * 2] += out_l
                self.buses[bus_idx].buf_lr[f * 2 + 1] += out_r

                gl += d_gl
                gr += d_gr
                dl += d_dl
                dr += d_dr
                stp += d_stp

            em.virtual_frame_pos = pos
            em.env_gain = env
            if is_stream:
                stream_publish_read(s_ctrl, em.stream_ch, pos)
            if not em.looping and pos >= total_f:
                em.active = False
                self.voices[v].active = False
            elif em.stopping and env <= Float32(0.0):
                em.active = False
                self.voices[v].active = False

            self.voices[v].cur_gain_l = tgt_gl
            self.voices[v].cur_gain_r = tgt_gr
            self.voices[v].cur_delay_l = tgt_dl
            self.voices[v].cur_delay_r = tgt_dr
            self.voices[v].cur_step = tgt_step
            self.voices[v].cur_reverb_send = tgt_rev

        for si in range(len(self.cycles_stem)):
            var sb = max(0, min(n_buses - 1, self.cycles_stem[si].bus_id))
            self.cycles_stem[si].mix_into_bus(self.buses[sb].buf_lr, n_frames)

        var eq_active_eps = Float32(1.0e-5)
        for po_i in range(len(self.post_order)):
            var b = self.post_order[po_i]
            if b == BUS_MASTER:
                for z in range(AUX_REVERB_ZONES):
                    var z_off = z * MAX_QUANTUM_FRAMES
                    var zg = self.aux_gains[z]
                    if zg <= Float32(1.0e-5):
                        continue
                    for f in range(n_frames):
                        var sl = self.aux_send_l[z_off + f]
                        var sr = self.aux_send_r[z_off + f]
                        var wet = self.reverb_zones[z].process_send(sl, sr)
                        self.buses[BUS_MASTER].buf_lr[f * 2] += wet[0] * zg
                        self.buses[BUS_MASTER].buf_lr[f * 2 + 1] += wet[1] * zg

            var bus_gain = (
                Float32(0.0) if self.buses[b].muted else self.buses[b].gain
            )
            var use_eq = (
                abs(self.buses[b].eq_low_gain - Float32(1.0)) > eq_active_eps
                or abs(self.buses[b].eq_mid_gain - Float32(1.0))
                > eq_active_eps
                or abs(self.buses[b].eq_high_gain - Float32(1.0))
                > eq_active_eps
            )

            var target_duck = Float32(1.0)
            var sc_id = self.buses[b].sidechain_bus
            if sc_id >= 0 and sc_id < n_buses:
                var sc_db = self.buses[sc_id].last_rms_db
                if sc_db > self.buses[b].duck_threshold_db:
                    var over = min(
                        Float32(1.0),
                        (sc_db - self.buses[b].duck_threshold_db)
                        / Float32(18.0),
                    )
                    target_duck = exp(
                        (self.buses[b].duck_amount_db * over)
                        * Float32(0.11512925465)
                    )

            var sum_sq = Float32(0.0)
            var peak_b = Float32(0.0)
            var low_d = self.buses[b].eq_low_gain - Float32(1.0)
            var mid_d = self.buses[b].eq_mid_gain - Float32(1.0)
            var high_d = self.buses[b].eq_high_gain - Float32(1.0)

            for f in range(n_frames):
                var xl = self.buses[b].buf_lr[f * 2] * bus_gain
                var xr = self.buses[b].buf_lr[f * 2 + 1] * bus_gain

                if use_eq:
                    _ = self.buses[b].svf_low_l.process(xl)
                    _ = self.buses[b].svf_low_r.process(xr)
                    xl += low_d * self.buses[b].svf_low_l.lp
                    xr += low_d * self.buses[b].svf_low_r.lp

                    _ = self.buses[b].svf_mid_l.process(xl)
                    _ = self.buses[b].svf_mid_r.process(xr)
                    xl += mid_d * (
                        self.buses[b].svf_mid_l.k * self.buses[b].svf_mid_l.bp
                    )
                    xr += mid_d * (
                        self.buses[b].svf_mid_r.k * self.buses[b].svf_mid_r.bp
                    )

                    _ = self.buses[b].svf_high_l.process(xl)
                    _ = self.buses[b].svf_high_r.process(xr)
                    xl += high_d * self.buses[b].svf_high_l.hp
                    xr += high_d * self.buses[b].svf_high_r.hp

                if self.buses[b].comp_enabled:
                    var p_sq = Float32(0.5) * (xl * xl + xr * xr)
                    var c_coef = (
                        self.buses[b].comp_attack_coeff
                        if p_sq > self.buses[b].comp_env_sq
                        else self.buses[b].comp_release_coeff
                    )
                    self.buses[b].comp_env_sq = (
                        p_sq + (self.buses[b].comp_env_sq - p_sq) * c_coef
                    )
                    if self.buses[b].comp_env_sq > Float32(1.0e-10):
                        var env_db = Float32(
                            10.0 * log10(Float64(self.buses[b].comp_env_sq))
                        )
                        if env_db > self.buses[b].comp_threshold_db:
                            var gr_db = (
                                self.buses[b].comp_threshold_db - env_db
                            ) * (
                                Float32(1.0)
                                - Float32(1.0)
                                / max(Float32(1.0), self.buses[b].comp_ratio)
                            )
                            var cg = (
                                exp(gr_db * Float32(0.11512925465))
                                * self.buses[b].comp_makeup_gain
                            )
                            xl *= cg
                            xr *= cg
                        else:
                            xl *= self.buses[b].comp_makeup_gain
                            xr *= self.buses[b].comp_makeup_gain

                if sc_id >= 0:
                    var d_coef = (
                        self.buses[b].duck_attack_coeff
                        if target_duck < self.buses[b].duck_env
                        else self.buses[b].duck_release_coeff
                    )
                    self.buses[b].duck_env = (
                        target_duck
                        + (self.buses[b].duck_env - target_duck) * d_coef
                    )
                    xl *= self.buses[b].duck_env
                    xr *= self.buses[b].duck_env

                self.buses[b].buf_lr[f * 2] = xl
                self.buses[b].buf_lr[f * 2 + 1] = xr

                var sq = Float32(0.5) * (xl * xl + xr * xr)
                sum_sq += sq
                var pk = max(abs(xl), abs(xr))
                if pk > peak_b:
                    peak_b = pk

            var mean_sq = sum_sq * inv_frames
            self.buses[b].last_rms_db = (
                Float32(10.0 * log10(Float64(mean_sq)))
                if mean_sq > Float32(1.0e-10)
                else Float32(-100.0)
            )
            self.buses[b].last_peak = peak_b

            if b != BUS_MASTER:
                for z in range(AUX_REVERB_ZONES):
                    var s_gain = self.buses[b].aux_sends[z]
                    if s_gain > Float32(1.0e-5):
                        var z_off = z * MAX_QUANTUM_FRAMES
                        for f in range(n_frames):
                            self.aux_send_l[z_off + f] += (
                                self.buses[b].buf_lr[f * 2] * s_gain
                            )
                            self.aux_send_r[z_off + f] += (
                                self.buses[b].buf_lr[f * 2 + 1] * s_gain
                            )

                var parent = self.buses[b].parent_bus
                if parent >= 0 and parent < n_buses:
                    for i in range(n_frames * 2):
                        self.buses[parent].buf_lr[i] += self.buses[b].buf_lr[i]

        var dc_r = Float32(0.9995)
        for f in range(n_frames):
            var ml = self.buses[BUS_MASTER].buf_lr[f * 2]
            var mr = self.buses[BUS_MASTER].buf_lr[f * 2 + 1]

            if isnan(ml) or isinf(ml):
                ml = 0.0
            if isnan(mr) or isinf(mr):
                mr = 0.0

            if self.dc_enabled:
                var yl = ml - self.dc_x_l + dc_r * self.dc_y_l
                self.dc_x_l = ml
                self.dc_y_l = yl
                ml = yl

                var yr = mr - self.dc_x_r + dc_r * self.dc_y_r
                self.dc_x_r = mr
                self.dc_y_r = yr
                mr = yr

            self.lufs_meter.process_frame(ml, mr)
            var lim = self.limiter.process(ml, mr)
            var fl = lim[0]
            var fr = lim[1]
            if isnan(fl) or isinf(fl):
                fl = 0.0
            if isnan(fr) or isinf(fr):
                fr = 0.0
            out_lr[f * 2] = fl
            out_lr[f * 2 + 1] = fr

        self._release_dead_streams()
        self.quantums_rendered_local += 1
        self._publish_telemetry()

    def _publish_backend(self):
        if self.telemetry_atomic == 0:
            return
        self._telem_ptr(_TELEM_BACKEND)[unsafe_offset=0].store(Int64(self.backend))
        self._telem_ptr(_TELEM_RECONNECTS)[unsafe_offset=0].store(Int64(self.reconnects_local))
        self._telem_ptr(_TELEM_DEVICE_LOST)[unsafe_offset=0].store(Int64(self.device_lost_local))

    def _drop_backend(mut self):
        """Closes the failed output; the RT loop then retries with exponential backoff."""
        if self.backend == AUDIO_BACKEND_PULSE or self.pulse.connected():
            self.pulse.close()
        if self.alsa_fd >= 0:
            alsa_close_pcm(self.alsa_fd)
            self.alsa_fd = -1
        self.alsa_pending_frames = 0
        self.alsa_pending_offset = 0
        self.backend = AUDIO_BACKEND_NONE
        self.device_lost_local += 1
        self.backoff_ms = RECONNECT_BACKOFF_MIN_MS
        self.next_attempt_s = get_time() + Float64(self.backoff_ms) * 0.001
        self._publish_backend()

    def try_open_backend(mut self) -> Bool:
        """Opens the output named by `preferred_device` (`"pulse:<sink>"`, `"alsa:<pcm node>"`,
        or `""` = default PulseAudio/PipeWire sink, then any ALSA node when `allow_alsa_fallback`).
        Called on the main thread before the RT thread starts, or by the RT thread itself."""
        if self.headless_rt_mode:
            self.backend = AUDIO_BACKEND_HEADLESS
            self._publish_backend()
            return True
        var block_bytes = self.quantum_frames * 4
        var pref = self.preferred_device
        var alsa_only = pref.startswith("alsa:")
        if not alsa_only:
            var sink = String("")
            if pref.startswith("pulse:"):
                sink = String(pref[byte=6:])
            self.pulse.close()
            if self.pulse.connect(self.sample_rate, block_bytes, sink):
                self.backend = AUDIO_BACKEND_PULSE
                self._publish_backend()
                return True
            self.pulse.close()
        if alsa_only or self.allow_alsa_fallback:
            var out_period = self.quantum_frames
            var out_buffer = self.quantum_frames * 4
            var fd: Int
            if alsa_only:
                fd = _alsa_open_node(String(pref[byte=5:]), self.sample_rate, self.quantum_frames, out_period, out_buffer)
            else:
                fd = alsa_open_pcm(self.sample_rate, self.quantum_frames, out_period, out_buffer)
            if fd >= 0:
                self.alsa_fd = fd
                self.alsa_period_frames = out_period
                self.alsa_pending_frames = 0
                self.alsa_pending_offset = 0
                self.backend = AUDIO_BACKEND_ALSA
                self._publish_backend()
                return True
        self.backend = AUDIO_BACKEND_NONE
        self._publish_backend()
        return False

    def _quantum_to_i16(mut self, float_lr: List[Float32], qf: Int):
        for i in range(qf * 2):
            var s = max(Float32(-1.0), min(Float32(1.0), float_lr[i]))
            self.rt_pcm_i16[i] = Int16(Int(s * Float32(32767.0)))

    def _rt_thread_loop(mut self):
        var qf = self.quantum_frames
        var float_lr = List[Float32](length=MAX_QUANTUM_FRAMES * 2, fill=0.0)
        if len(self.rt_pcm_i16) != qf * 2:
            self.rt_pcm_i16.resize(qf * 2, 0)
        var quantum_ns = Int(Float64(qf) * 1.0e9 / Float64(self.sample_rate))
        while self._running_ptr()[unsafe_offset=0].load() != 0:
            if self._telem_ptr(_TELEM_FAULT_INJECT)[unsafe_offset=0].load() != 0:
                self._telem_ptr(_TELEM_FAULT_INJECT)[unsafe_offset=0].store(0)
                if self.backend != AUDIO_BACKEND_NONE:
                    self._drop_backend()

            if self.backend == AUDIO_BACKEND_NONE:
                if get_time() >= self.next_attempt_s:
                    if self.try_open_backend():
                        self.reconnects_local += 1
                        self.backoff_ms = RECONNECT_BACKOFF_MIN_MS
                        self._publish_backend()
                        continue
                    self.backoff_ms = next_reconnect_backoff_ms(self.backoff_ms)
                    self.next_attempt_s = get_time() + Float64(self.backoff_ms) * 0.001
                # Output lost: keep the mixer clock (voices, fades, streams) running in real time.
                self.render_quantum(float_lr, qf)
                _rt_sleep_ns(quantum_ns)
                continue

            if self.backend == AUDIO_BACKEND_HEADLESS:
                self.render_quantum(float_lr, qf)
                _rt_sleep_ns(1_000_000)  # 1.0 ms deterministic headless RT cadence
                continue

            if self.backend == AUDIO_BACKEND_PULSE:
                self.pulse.pump()
                if not self.pulse.connected():
                    self._drop_backend()
                    continue
                if self.pulse.has_pending_write():
                    if not self.pulse.flush_pending():
                        self._drop_backend()
                        continue
                var frame_bytes = qf * 4
                if not self.pulse.has_pending_write() and self.pulse.requested_bytes >= frame_bytes:
                    self.render_quantum(float_lr, qf)
                    self._quantum_to_i16(float_lr, qf)
                    _ = self.pulse.write_samples(self.rt_pcm_i16)
                    if not self.pulse.connected():
                        self._drop_backend()
                else:
                    _ = self.pulse.wait_io(10)
                    if not self.pulse.connected():
                        self._drop_backend()
            elif self.backend == AUDIO_BACKEND_ALSA:
                if self.alsa_fd < 0:
                    self._drop_backend()
                    continue
                if self.alsa_pending_frames <= 0:
                    self.render_quantum(float_lr, qf)
                    self._quantum_to_i16(float_lr, qf)
                    self.alsa_pending_frames = qf
                    self.alsa_pending_offset = 0
                var written = alsa_write_interleaved(
                    self.alsa_fd,
                    Int(self.rt_pcm_i16.unsafe_ptr().unsafe_offset(self.alsa_pending_offset * 2)),
                    self.alsa_pending_frames,
                    self.alsa_xruns,
                )
                if written > 0:
                    self.alsa_pending_offset += written
                    self.alsa_pending_frames -= written
                elif written == 0:
                    var fd32 = UInt32(self.alsa_fd)
                    self.alsa_poll_buf[0] = UInt8(fd32 & 0xFF)
                    self.alsa_poll_buf[1] = UInt8((fd32 >> 8) & 0xFF)
                    self.alsa_poll_buf[2] = UInt8((fd32 >> 16) & 0xFF)
                    self.alsa_poll_buf[3] = UInt8((fd32 >> 24) & 0xFF)
                    self.alsa_poll_buf[4] = 0x04  # POLLOUT
                    self.alsa_poll_buf[5] = 0
                    self.alsa_poll_buf[6] = 0
                    self.alsa_poll_buf[7] = 0
                    _ = external_call["poll", Int32](self.alsa_poll_buf.unsafe_ptr(), UInt64(1), Int32(10))
                else:
                    self._drop_backend()
            else:
                self._drop_backend()


def _rt_sleep_ns(ns: Int):
    var ts = Array[Int64, 2](fill=0)
    ts[0] = Int64(ns // 1_000_000_000)
    ts[1] = Int64(ns % 1_000_000_000)
    _ = external_call["nanosleep", Int32](Pointer(to=ts), Int(0))
    _ = ts[1]


def _alsa_open_node(path: String, rate: Int, preferred_period: Int, mut out_period: Int, mut out_buffer: Int) -> Int:
    """Opens one specific ALSA playback node (e.g. `/dev/snd/pcmC1D0p`), or `-1`."""
    if not path.startswith("/dev/snd/pcm"):
        return -1
    var owned = path
    var fd = Int(external_call["openat", Int32](Int32(-100), owned.as_c_string_span(), Int32(1 | 2048 | 0x80000), Int32(0)))
    if fd < 0:
        return -1
    if _alsa_configure_fd(fd, rate, preferred_period, out_period, out_buffer):
        return fd
    close_fd(fd)
    return -1


struct AudioDeviceInfo(Copyable, Movable):
    """One selectable output device. `id` is passed to `AudioServer.select_output_device`."""
    var id: String
    var name: String
    var description: String
    var backend: String

    def __init__(out self, id: String, name: String, description: String, backend: String):
        self.id = id
        self.name = name
        self.description = description
        self.backend = backend


def enumerate_output_devices() -> List[AudioDeviceInfo]:
    """PulseAudio/PipeWire sinks (temporary connection) plus raw ALSA playback nodes."""
    var out = List[AudioDeviceInfo]()
    var probe = PulseClient()
    var sinks = probe.list_sinks()
    for i in range(len(sinks)):
        out.append(AudioDeviceInfo("pulse:" + sinks[i].name, sinks[i].name, sinks[i].description, "pulse"))
    probe.close()
    var names = list_directory("/dev/snd")
    for i in range(len(names)):
        var n = names[i]
        if n.startswith("pcmC") and n.endswith("p"):
            var node = "/dev/snd/" + n
            out.append(AudioDeviceInfo("alsa:" + node, n, "ALSA playback " + n, "alsa"))
    return out^


struct AudioServer(Resource, Movable):
    """Multithreaded real-time Flint v5 AudioServer backed by a heap-pinned
    `AudioRtEngine` (`rt_core_addr`). Moving `AudioServer` moves ONLY the 8-byte
    `rt_core_addr` pointer and caller-side handle/ID counters, never invalidating
    the RT pthread's `AudioRtEngine*` address or racing with RT mixer state."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.audio.AudioServer")
    var rt_core_addr: Int
    var sample_rate: Int
    var sr_f: Float32
    var quantum_frames: Int
    var rt_thread_handle: UInt64
    var rt_running: Bool

    # Main-thread-only handle & ID allocators (zero RT-thread contention!)
    var caller_next_slot: Int
    var caller_next_gen: UInt32
    var caller_next_asset_id: Int
    var caller_next_bus_id: Int
    var caller_next_stem_id: Int
    var path_cache: Dict[String, Int]
    # Disk streaming (heap-pinned pool shared with the streamer pthread)
    var stream_pool_addr: Int
    var stream_thread_handle: UInt64
    var stream_thread_running: Bool
    var stream_assets: Dict[Int, StreamSourceInfo]
    var caller_next_atten_id: Int

    def __init__(
        out self,
        sample_rate: Int = 48000,
        quantum_frames: Int = 256,
        ring_capacity: Int = 2048,
    ):
        var sr_clean = max(8000, sample_rate)
        var qf_clean = max(32, min(MAX_QUANTUM_FRAMES, quantum_frames))
        self.rt_core_addr = Int(
            _heap_box_alloc[AudioRtEngine](
                AudioRtEngine(sr_clean, qf_clean, ring_capacity)
            )
        )
        self.sample_rate = sr_clean
        self.sr_f = Float32(sr_clean)
        self.quantum_frames = qf_clean
        self.rt_thread_handle = 0
        self.rt_running = False

        self.caller_next_slot = 0
        self.caller_next_gen = 1
        self.caller_next_asset_id = 0
        self.caller_next_bus_id = DEFAULT_BUS_COUNT
        self.caller_next_stem_id = 0
        self.path_cache = Dict[String, Int]()
        self.stream_pool_addr = Int(_heap_box_alloc[AudioStreamPool](AudioStreamPool()))
        self.stream_thread_handle = 0
        self.stream_thread_running = False
        self.stream_assets = Dict[Int, StreamSourceInfo]()
        self.caller_next_atten_id = 0
        self._rt_core()[].stream_ctrl_addr = self._stream_pool()[].ctrl_addr

    def __moveinit__(out self, deinit existing: Self):
        """Relocates ONLY the 8-byte `rt_core_addr` pointer and caller counters;
        the heap-pinned `AudioRtEngine` address observed by the RT pthread never moves."""
        self.rt_core_addr = existing.rt_core_addr
        self.sample_rate = existing.sample_rate
        self.sr_f = existing.sr_f
        self.quantum_frames = existing.quantum_frames
        self.rt_thread_handle = existing.rt_thread_handle
        self.rt_running = existing.rt_running
        self.caller_next_slot = existing.caller_next_slot
        self.caller_next_gen = existing.caller_next_gen
        self.caller_next_asset_id = existing.caller_next_asset_id
        self.caller_next_bus_id = existing.caller_next_bus_id
        self.caller_next_stem_id = existing.caller_next_stem_id
        self.path_cache = existing.path_cache^
        self.stream_pool_addr = existing.stream_pool_addr
        self.stream_thread_handle = existing.stream_thread_handle
        self.stream_thread_running = existing.stream_thread_running
        self.stream_assets = existing.stream_assets^
        self.caller_next_atten_id = existing.caller_next_atten_id

    @always_inline
    def _rt_core(self) -> Pointer[AudioRtEngine, MutAnyOrigin]:
        return Pointer[AudioRtEngine, MutAnyOrigin](
            unsafe_from_address=self.rt_core_addr
        )

    @always_inline
    def _stream_pool(self) -> Pointer[AudioStreamPool, MutAnyOrigin]:
        return Pointer[AudioStreamPool, MutAnyOrigin](
            unsafe_from_address=self.stream_pool_addr
        )

    def __deinit__(deinit self):
        if self.rt_core_addr != 0:
            # 1. Signal `running_atomic = 0` and `pthread_join` RT thread FIRST!
            self.stop_realtime_thread()
            # 2. Only AFTER the RT thread has joined, destroy ring/atomics & free `AudioRtEngine`.
            var p = self._rt_core()
            p[].destroy_engine_allocations()
            _heap_box_free[AudioRtEngine](p)
        if self.stream_pool_addr != 0:
            var sp = self._stream_pool()
            sp[].destroy()
            _heap_box_free[AudioStreamPool](sp)

    @always_inline
    def rt_engine_address(self) -> Int:
        return self.rt_core_addr

    def drain_retired(mut self) -> Int:
        """Reclaims payloads retired by the RT thread on the caller/main thread."""
        if self.rt_core_addr == 0:
            return 0
        return self._rt_core()[].drain_retired_main_thread()

    def _enqueue_guaranteed(mut self, cmd: AudioCmd):
        """Enqueues a command (including heap-owned payloads) without loss."""
        if self.rt_core_addr == 0:
            return
        _ = self.drain_retired()
        while not self._rt_core()[].cmd_ring.push(cmd):
            if self.rt_running:
                _ = external_call["sched_yield", Int32]()
            else:
                self._rt_core()[]._drain_commands()
                _ = self.drain_retired()

    # ---- Lock-Free Atomic Telemetry Getters ----

    @always_inline
    def quantums_rendered(self) -> UInt64:
        if self.rt_core_addr == 0:
            return 0
        return UInt64(
            self._rt_core()[]._telem_ptr(_TELEM_QUANTUMS)[
                unsafe_offset=0
            ].load()
        )

    @always_inline
    def voices_stolen_count(self) -> UInt64:
        if self.rt_core_addr == 0:
            return 0
        return UInt64(
            self._rt_core()[]._telem_ptr(_TELEM_STOLEN)[unsafe_offset=0].load()
        )

    @always_inline
    def active_logical_count(self) -> Int:
        if self.rt_core_addr == 0:
            return 0
        return Int(
            self._rt_core()[]._telem_ptr(_TELEM_LOGICAL)[
                unsafe_offset=0
            ].load()
        )

    @always_inline
    def active_physical_count(self) -> Int:
        if self.rt_core_addr == 0:
            return 0
        return Int(
            self._rt_core()[]._telem_ptr(_TELEM_PHYSICAL)[
                unsafe_offset=0
            ].load()
        )

    @always_inline
    def momentary_lufs(self) -> Float32:
        if self.rt_core_addr == 0:
            return -100.0
        var bits = UInt32(
            self._rt_core()[]._telem_ptr(_TELEM_MOMENTARY_LUFS)[
                unsafe_offset=0
            ].load()
            & 0xFFFFFFFF
        )
        return bitcast[DType.float32](bits)

    @always_inline
    def integrated_lufs(self) -> Float32:
        if self.rt_core_addr == 0:
            return -100.0
        var bits = UInt32(
            self._rt_core()[]._telem_ptr(_TELEM_INTEGRATED_LUFS)[
                unsafe_offset=0
            ].load()
            & 0xFFFFFFFF
        )
        return bitcast[DType.float32](bits)

    @always_inline
    def music_duck_env(self) -> Float32:
        if self.rt_core_addr == 0:
            return 1.0
        var bits = UInt32(
            self._rt_core()[]._telem_ptr(_TELEM_MUSIC_DUCK_ENV)[
                unsafe_offset=0
            ].load()
            & 0xFFFFFFFF
        )
        return bitcast[DType.float32](bits)

    # ---- Race-Free Asset Registration (`CMD_INSTALL_CLIP`) ----

    def register_pcm16_asset(
        mut self,
        pcm: List[Int16],
        channels: Int = 1,
        sample_rate: Int = 48000,
    ) -> Int:
        var aid = self.caller_next_asset_id
        self.caller_next_asset_id += 1

        var slot = AudioAssetSlot()
        slot.valid = len(pcm) >= max(1, channels) and channels >= 1
        slot.kind = ASSET_KIND_PCM16
        slot.channels = max(1, channels)
        slot.sample_rate = max(1, sample_rate)
        slot.total_frames = len(pcm) // max(1, channels)
        slot.pcm = pcm.copy()

        var boxed = _heap_box_alloc[AudioAssetSlot](slot^)
        var cmd = AudioCmd()
        cmd.kind = CMD_INSTALL_CLIP
        cmd.asset_id = Int32(aid)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)
        return aid

    def register_wav_asset(mut self, wav: WavClip, path: String = "") -> Int:
        if path.byte_length() > 0:
            var cached = self.path_cache.get(path)
            if cached:
                return cached.value()
        if not wav.valid:
            var aid = self.caller_next_asset_id
            self.caller_next_asset_id += 1
            var boxed = _heap_box_alloc[AudioAssetSlot](AudioAssetSlot())
            var cmd = AudioCmd()
            cmd.kind = CMD_INSTALL_CLIP
            cmd.asset_id = Int32(aid)
            _pack_ptr_into_cmd(cmd, Int(boxed))
            self._enqueue_guaranteed(cmd)
            return aid
        var aid = self.register_pcm16_asset(
            wav.samples, wav.channels, wav.sample_rate
        )
        if path.byte_length() > 0:
            self.path_cache[path] = aid
        return aid

    def register_qoa_asset(mut self, var qoa_bytes: List[UInt8], path: String = "") -> Int:
        if path.byte_length() > 0:
            var cached = self.path_cache.get(path)
            if cached:
                return cached.value()
        var aid = self.caller_next_asset_id
        self.caller_next_asset_id += 1

        var slot = AudioAssetSlot()
        var qa = QoaAsset.from_bytes(qoa_bytes^)
        var is_ok = qa.valid
        if qa.valid:
            slot.valid = True
            slot.kind = ASSET_KIND_QOA
            slot.channels = qa.channels
            slot.sample_rate = qa.sample_rate
            slot.total_frames = qa.samples_per_channel
            slot.qoa = qa^

        var boxed = _heap_box_alloc[AudioAssetSlot](slot^)
        var cmd = AudioCmd()
        cmd.kind = CMD_INSTALL_CLIP
        cmd.asset_id = Int32(aid)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)
        if is_ok and path.byte_length() > 0:
            self.path_cache[path] = aid
        return aid

    def load_audio_bytes_cached(mut self, var raw: List[UInt8], path: String = "") -> Int:
        """Decodes and caches raw `.wav`, `.qoa` or Ogg Vorbis (`.ogg`/`.oga`) audio bytes,
        returning `-1` on failure. Vorbis is decoded once into 16-bit PCM."""
        if len(raw) == 0:
            return -1
        if path.byte_length() > 0:
            var cached = self.path_cache.get(path)
            if cached:
                return cached.value()
        if is_ogg_vorbis(raw):
            var clip = decode_ogg_vorbis(raw)
            if not clip.valid or clip.samples_per_channel <= 0:
                return -1
            var vid = self.register_pcm16_asset(clip.samples, clip.channels, clip.sample_rate)
            if path.byte_length() > 0:
                self.path_cache[path] = vid
            return vid
        var is_qoa = (
            path.endswith(".qoa")
            or path.endswith(".QOA")
            or (len(raw) >= 4 and raw[0] == 113 and raw[1] == 111 and raw[2] == 97 and raw[3] == 102)
        )
        if is_qoa:
            var probe = QoaAsset.from_bytes(raw.copy())
            if not probe.valid:
                return -1
            return self.register_qoa_asset(raw^, path)
        var wav = decode_wav(raw)
        if not wav.valid:
            return -1
        return self.register_wav_asset(wav, path)

    def register_stream_asset(mut self, path: String) -> Int:
        """Registers a 16-bit PCM `.wav` or `.qoa` file for bounded-memory disk streaming:
        only the header is parsed now; samples are read in chunks while the sound plays.
        Returns `-1` if the file is missing or not streamable."""
        if path.byte_length() == 0:
            return -1
        var key = "stream:" + path
        var cached = self.path_cache.get(key)
        if cached:
            return cached.value()
        var info = probe_stream_source(path)
        if not info.valid or info.total_frames <= 0:
            return -1
        var aid = self.caller_next_asset_id
        self.caller_next_asset_id += 1
        var slot = AudioAssetSlot()
        slot.valid = True
        slot.kind = ASSET_KIND_STREAM
        slot.channels = min(2, info.channels)
        slot.sample_rate = info.sample_rate
        slot.total_frames = info.total_frames
        var boxed = _heap_box_alloc[AudioAssetSlot](slot^)
        var cmd = AudioCmd()
        cmd.kind = CMD_INSTALL_CLIP
        cmd.asset_id = Int32(aid)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)
        self.stream_assets[aid] = info^
        self.path_cache[key] = aid
        return aid

    def is_stream_asset(self, asset_id: Int) -> Bool:
        return asset_id in self.stream_assets

    def load_audio_cached(mut self, path: String, stream: Bool = False) -> Int:
        """Loads and caches a `.wav`, `.qoa` or `.ogg` audio asset by file path, returning `-1` on failure.
        `stream=True` streams it from disk instead of decoding it into memory (long music / VO)."""
        if path.byte_length() == 0:
            return -1
        if stream:
            var sid = self.register_stream_asset(path)
            if sid >= 0 or not (path.endswith(".ogg") or path.endswith(".oga")):
                return sid
            # Ogg Vorbis is not disk-streamable: fall back to a one-time in-memory decode.
        var cached = self.path_cache.get(path)
        if cached:
            return cached.value()
        var raw = read_bytes(path)
        if len(raw) == 0:
            return -1
        return self.load_audio_bytes_cached(raw^, path)

    # ---- Race-Free Bus & Stem Configuration (`CMD_ADD_BUS`, `CMD_CONFIG_BUS`) ----

    def add_bus(mut self, name: String, parent_bus: Int = BUS_MASTER) -> Int:
        var id = self.caller_next_bus_id
        self.caller_next_bus_id += 1
        var parent = (
            parent_bus if (parent_bus >= 0 and parent_bus < id) else BUS_MASTER
        )
        var boxed = _heap_box_alloc[AudioBus](
            AudioBus(id, name, parent, self.sr_f)
        )
        var cmd = AudioCmd()
        cmd.kind = CMD_ADD_BUS
        cmd.bus_id = Int32(id)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)
        return id

    def set_bus_gain(self, bus_id: Int, gain: Float32) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_GAIN_MUTE
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = max(Float32(0.0), gain)
        cmd.p1 = -1.0
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_bus_mute(self, bus_id: Int, muted: Bool) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_GAIN_MUTE
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = -1.0
        cmd.p1 = Float32(1.0) if muted else Float32(0.0)
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_bus_eq_db(
        self, bus_id: Int, low_db: Float32, mid_db: Float32, high_db: Float32
    ) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_EQ_DB
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = low_db
        cmd.p1 = mid_db
        cmd.p2 = high_db
        return self._rt_core()[].cmd_ring.push(cmd)

    def configure_bus_compressor(
        self,
        bus_id: Int,
        enabled: Bool,
        threshold_db: Float32 = -14.0,
        ratio: Float32 = 3.0,
        makeup_gain: Float32 = 1.0,
    ) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_COMPRESSOR
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = Float32(1.0) if enabled else Float32(0.0)
        cmd.p1 = threshold_db
        cmd.p2 = ratio
        cmd.p3 = makeup_gain
        return self._rt_core()[].cmd_ring.push(cmd)

    def configure_bus_sidechain(
        self,
        bus_id: Int,
        source_bus: Int,
        threshold_db: Float32 = -32.0,
        amount_db: Float32 = -9.0,
    ) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_SIDECHAIN
        cmd.bus_id = Int32(bus_id)
        cmd.slot = Int32(source_bus)
        cmd.p0 = threshold_db
        cmd.p1 = amount_db
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_bus_aux_send(
        self, bus_id: Int, zone: Int, send_gain: Float32
    ) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_AUX_SEND
        cmd.bus_id = Int32(bus_id)
        cmd.slot = Int32(zone)
        cmd.p0 = send_gain
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_dc_blocker(self, enabled: Bool) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_MASTER
        cmd.flags = _MASTER_CFG_DC_BLOCKER
        cmd.p0 = Float32(1.0) if enabled else Float32(0.0)
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_aux_gain(self, zone: Int, gain: Float32) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_MASTER
        cmd.flags = _MASTER_CFG_AUX_GAIN
        cmd.slot = Int32(zone)
        cmd.p0 = gain
        return self._rt_core()[].cmd_ring.push(cmd)

    def add_cycles_stem(
        mut self, code: String, bus_id: Int = BUS_MUSIC
    ) -> Int:
        var sid = self.caller_next_stem_id
        self.caller_next_stem_id += 1
        var node = CyclesStemNode(self.sample_rate, bus_id)
        _ = node.load_pattern(code)
        var boxed = _heap_box_alloc[CyclesStemNode](node^)
        var cmd = AudioCmd()
        cmd.kind = CMD_ADD_STEM
        cmd.slot = Int32(sid)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)
        return sid

    def transition_cycles_stem(
        mut self,
        stem_idx: Int,
        code: String,
        crossfade_cycles: Float32 = 1.0,
    ):
        var prog = run_program(code)
        if len(prog.errors) > 0:
            return
        var boxed = _heap_box_alloc[_StemTransitionPayload](
            _StemTransitionPayload(prog^, crossfade_cycles)
        )
        var cmd = AudioCmd()
        cmd.kind = CMD_TRANSITION_STEM
        cmd.slot = Int32(stem_idx)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)

    # ---- Command Submission API (Main-Thread-Only Handle Allocator + Lock-Free MPSC) ----

    def allocate_handle(mut self) -> SoundHandle:
        """Reserves a generational emitter handle using ONLY caller-side monotonic counters
        (`caller_next_slot`, `caller_next_gen`) — never reads or writes `rt_core.emitters`."""
        var idx = self.caller_next_slot
        self.caller_next_slot = (idx + 1) % MAX_LOGICAL_EMITTERS
        var gen = self.caller_next_gen
        var next_g = gen + 1
        if next_g == 0:
            next_g = 1
        self.caller_next_gen = next_g
        return SoundHandle(Int32(idx), gen)

    def _attach_stream(mut self, mut cmd: AudioCmd, asset_id: Int, h: SoundHandle, looping: Bool) -> Int:
        """Claims + prefills a stream channel for a streamed asset (main thread). Returns the
        channel or -1; the RT thread drops a streamed PLAY that carries no channel."""
        var info_opt = self.stream_assets.get(asset_id)
        if not info_opt or self.stream_pool_addr == 0:
            return -1
        var ch = self._stream_pool()[].claim(info_opt.value(), looping, Int(h.slot), h.generation)
        if ch >= 0:
            cmd.flags |= AFLAG_STREAM | (UInt32(ch) << 8)
        return ch

    def _push_play(mut self, cmd: AudioCmd, stream_ch: Int) -> Bool:
        var ok = self._rt_core()[].cmd_ring.push(cmd)
        if not ok and stream_ch >= 0:
            self._stream_pool()[].release_from_main(stream_ch)
        return ok

    def play_3d(
        mut self,
        asset_id: Int,
        pos: Vec3,
        vel: Vec3 = Vec3(),
        volume: Float32 = 1.0,
        pitch: Float32 = 1.0,
        bus_id: Int = BUS_SFX,
        looping: Bool = False,
        priority: Int = 1,
        ref_dist: Float32 = 15.0,
        max_dist: Float32 = 350.0,
        attenuation_id: Int = -1,
        cone_dir: Vec3 = Vec3(),
    ) -> SoundHandle:
        var h = self.allocate_handle()
        var cmd = AudioCmd()
        cmd.kind = ACMD_PLAY
        cmd.flags = AFLAG_SPATIAL | (AFLAG_LOOPING if looping else UInt32(0))
        cmd.slot = h.slot
        cmd.generation = h.generation
        cmd.asset_id = Int32(asset_id)
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = pos.x
        cmd.p1 = pos.y
        cmd.p2 = pos.z
        cmd.p3 = vel.x
        cmd.p4 = vel.y
        cmd.p5 = vel.z
        cmd.p6 = volume
        cmd.p7 = pitch
        cmd.p8 = ref_dist
        cmd.p9 = max_dist + Float32(priority) * Float32(0.001)
        var sch = self._attach_stream(cmd, asset_id, h, looping)
        _ = self._push_play(cmd, sch)
        if attenuation_id >= 0:
            _ = self.set_emitter_attenuation(h, attenuation_id, cone_dir)
        return h

    def play_2d(
        mut self,
        asset_id: Int,
        volume: Float32 = 1.0,
        pitch: Float32 = 1.0,
        pan: Float32 = 0.0,
        bus_id: Int = BUS_SFX,
        looping: Bool = False,
        direct_stereo: Bool = False,
    ) -> SoundHandle:
        var h = self.allocate_handle()
        var cmd = AudioCmd()
        cmd.kind = ACMD_PLAY
        cmd.flags = (AFLAG_LOOPING if looping else UInt32(0)) | (
            AFLAG_DIRECT_STEREO if direct_stereo else UInt32(0)
        )
        cmd.slot = h.slot
        cmd.generation = h.generation
        cmd.asset_id = Int32(asset_id)
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = pan
        cmd.p6 = volume
        cmd.p7 = pitch
        var sch = self._attach_stream(cmd, asset_id, h, looping)
        _ = self._push_play(cmd, sch)
        return h

    def is_playing(self, handle: SoundHandle) -> Bool:
        """True while the voice exists (playing, paused or fading out). Lock-free snapshot read
        of mixer state: exact offline, at most one quantum stale while the RT thread runs."""
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var s = Int(handle.slot)
        if s < 0 or s >= MAX_LOGICAL_EMITTERS:
            return False
        ref em = self._rt_core()[].emitters[s]
        return em.active and em.generation == handle.generation

    def stop_sound(self, handle: SoundHandle, fade_seconds: Float32 = 0.0) -> Bool:
        """Stops a sound; `fade_seconds > 0` fades it out (linear) before releasing the voice."""
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_STOP
        cmd.slot = handle.slot
        cmd.generation = handle.generation
        cmd.p0 = max(Float32(0.0), fade_seconds) if isfinite(fade_seconds) else Float32(0.0)
        return self._rt_core()[].cmd_ring.push(cmd)

    def _set_paused(self, handle: SoundHandle, paused: Bool) -> Bool:
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_PAUSE
        cmd.slot = handle.slot
        cmd.generation = handle.generation
        cmd.p0 = Float32(1.0) if paused else Float32(0.0)
        return self._rt_core()[].cmd_ring.push(cmd)

    def pause_sound(self, handle: SoundHandle) -> Bool:
        """Pauses one voice (click-free ~5 ms fade); its position is kept for `resume_sound`."""
        return self._set_paused(handle, True)

    def resume_sound(self, handle: SoundHandle) -> Bool:
        return self._set_paused(handle, False)

    def _set_global_pause(self, paused: Bool) -> Bool:
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_GLOBAL_PAUSE
        cmd.p0 = Float32(1.0) if paused else Float32(0.0)
        return self._rt_core()[].cmd_ring.push(cmd)

    def pause_all(self) -> Bool:
        """Global (pause-menu) pause of every bus except those marked `set_bus_pause_exempt`."""
        return self._set_global_pause(True)

    def resume_all(self) -> Bool:
        return self._set_global_pause(False)

    def set_bus_paused(self, bus_id: Int, paused: Bool) -> Bool:
        """Pauses/resumes every voice routed into `bus_id` or any of its child buses."""
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_PAUSE
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = Float32(1.0) if paused else Float32(0.0)
        cmd.p1 = -1.0
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_bus_pause_exempt(self, bus_id: Int, exempt: Bool) -> Bool:
        """Marks a bus (and its children) as still playing during `pause_all` (e.g. UI / menu music)."""
        if self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = CMD_CONFIG_BUS
        cmd.flags = _BUS_CFG_PAUSE
        cmd.bus_id = Int32(bus_id)
        cmd.p0 = -1.0
        cmd.p1 = Float32(1.0) if exempt else Float32(0.0)
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_voice_params(
        self,
        handle: SoundHandle,
        gain: Float32 = -1.0,
        pitch: Float32 = -1.0,
        lowpass_hz: Float32 = -1.0,
    ) -> Bool:
        """Game-parameter (RTPC) modifiers multiplied onto a voice: `gain` (linear), `pitch`
        (ratio) and a one-pole low-pass cutoff. Negative values leave a field unchanged."""
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_SET_VOICE_PARAMS
        cmd.slot = handle.slot
        cmd.generation = handle.generation
        cmd.p0 = gain
        cmd.p1 = pitch
        cmd.p2 = lowpass_hz
        return self._rt_core()[].cmd_ring.push(cmd)

    def register_attenuation(mut self, preset: AttenuationPreset) -> Int:
        """Installs a per-sound `AttenuationPreset` on the mixer and returns its id."""
        var pid = self.caller_next_atten_id
        self.caller_next_atten_id += 1
        var boxed = _heap_box_alloc[AttenuationPreset](preset.copy())
        var cmd = AudioCmd()
        cmd.kind = CMD_INSTALL_ATTEN
        cmd.asset_id = Int32(pid)
        _pack_ptr_into_cmd(cmd, Int(boxed))
        self._enqueue_guaranteed(cmd)
        return pid

    def set_emitter_attenuation(self, handle: SoundHandle, preset_id: Int, cone_dir: Vec3 = Vec3()) -> Bool:
        """Binds an attenuation preset (`-1` = default inverse curve) and cone axis to a voice."""
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_SET_EMITTER_ATTEN
        cmd.slot = handle.slot
        cmd.generation = handle.generation
        cmd.asset_id = Int32(preset_id)
        cmd.p0 = cone_dir.x
        cmd.p1 = cone_dir.y
        cmd.p2 = cone_dir.z
        return self._rt_core()[].cmd_ring.push(cmd)

    def stop_all(mut self, fade_seconds: Float32 = 0.0):
        """Stops all active logical emitters and physical voices immediately (`active_logical_count() == 0`),
        or fades them all out over `fade_seconds`."""
        if self.rt_core_addr == 0:
            return
        var cmd = AudioCmd()
        cmd.kind = ACMD_STOP_ALL
        if fade_seconds > Float32(1.0e-4) and isfinite(fade_seconds):
            cmd.p0 = fade_seconds
            self._enqueue_guaranteed(cmd)
            return
        self._enqueue_guaranteed(cmd)
        if self.rt_running:
            if self._rt_core()[].headless_rt_mode:
                var target_q = self.quantums_rendered() + 2
                var ts = Array[Int64, 2](fill=0)
                ts[1] = 200_000  # 0.2 ms
                for _ in range(30):
                    if self.quantums_rendered() >= target_q:
                        break
                    _ = external_call["nanosleep", Int32](Pointer(to=ts), Int(0))
                    _ = ts[1]
        else:
            self._rt_core()[]._drain_commands()
        self._rt_core()[].active_logical_local = 0
        self._rt_core()[].active_physical_local = 0
        self._rt_core()[]._telem_ptr(_TELEM_LOGICAL)[unsafe_offset=0].store(0)
        self._rt_core()[]._telem_ptr(_TELEM_PHYSICAL)[unsafe_offset=0].store(0)

    # ---- Disk streaming service & telemetry ----

    def service_streams(mut self) -> Bool:
        """Refills stream rings on the calling thread. Only does work when the background
        streamer pthread is not running (offline / headless-without-RT renders)."""
        if self.stream_pool_addr == 0 or self.stream_thread_running:
            return False
        var did = False
        for _ in range(4):
            if not self._stream_pool()[].service_all():
                break
            did = True
        return did

    def active_stream_count(self) -> Int:
        if self.stream_pool_addr == 0:
            return 0
        return self._stream_pool()[].active_count()

    def stream_underruns(self) -> Int:
        if self.rt_core_addr == 0:
            return 0
        return Int(self._rt_core()[]._telem_ptr(_TELEM_STREAM_UNDERRUNS)[unsafe_offset=0].load())

    def _start_stream_thread(mut self) -> Bool:
        if self.stream_thread_running or self.stream_pool_addr == 0:
            return self.stream_thread_running
        var run_addr = self._stream_pool()[].running_addr
        Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=run_addr)[unsafe_offset=0].store(1)
        var tid = UInt64(0)
        var rc = external_call["pthread_create", Int32](
            Pointer(to=tid),
            Int(0),
            _audio_stream_thread_entry,
            self.stream_pool_addr,
        )
        if rc != 0:
            Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=run_addr)[unsafe_offset=0].store(0)
            return False
        self.stream_thread_handle = tid
        self.stream_thread_running = True
        return True

    def _stop_stream_thread(mut self):
        if not self.stream_thread_running or self.stream_pool_addr == 0:
            return
        var run_addr = self._stream_pool()[].running_addr
        Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=run_addr)[unsafe_offset=0].store(0)
        _ = external_call["pthread_join", Int32](self.stream_thread_handle, Int(0))
        self.stream_thread_handle = 0
        self.stream_thread_running = False

    def update_emitter(
        self,
        handle: SoundHandle,
        pos: Vec3,
        vel: Vec3 = Vec3(),
        volume: Float32 = -1.0,
        pitch: Float32 = -1.0,
    ) -> Bool:
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_UPDATE_EMITTER
        cmd.slot = handle.slot
        cmd.generation = handle.generation
        cmd.p0 = pos.x
        cmd.p1 = pos.y
        cmd.p2 = pos.z
        cmd.p3 = vel.x
        cmd.p4 = vel.y
        cmd.p5 = vel.z
        cmd.p6 = volume
        cmd.p7 = pitch
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_emitter_occlusion(
        self,
        handle: SoundHandle,
        occlusion_gain: Float32,
        cutoff_hz: Float32,
        obstruction: Float32 = 0.0,
    ) -> Bool:
        if not handle.is_valid() or self.rt_core_addr == 0:
            return False
        var cmd = AudioCmd()
        cmd.kind = ACMD_SET_OCCLUSION
        cmd.slot = handle.slot
        cmd.generation = handle.generation
        cmd.p0 = occlusion_gain
        cmd.p1 = cutoff_hz
        cmd.p2 = obstruction
        return self._rt_core()[].cmd_ring.push(cmd)

    def set_listener(
        self,
        pos: Vec3,
        vel: Vec3,
        fwd: Vec3,
        right: Vec3,
        up: Vec3 = Vec3(0.0, 1.0, 0.0),
    ):
        if self.rt_core_addr == 0:
            return
        var c1 = AudioCmd()
        c1.kind = ACMD_SET_LISTENER_POS_VEL
        c1.p0 = pos.x
        c1.p1 = pos.y
        c1.p2 = pos.z
        c1.p3 = vel.x
        c1.p4 = vel.y
        c1.p5 = vel.z
        _ = self._rt_core()[].cmd_ring.push(c1)

        var c2 = AudioCmd()
        c2.kind = ACMD_SET_LISTENER_BASIS_FWD_RIGHT
        c2.p0 = fwd.x
        c2.p1 = fwd.y
        c2.p2 = fwd.z
        c2.p3 = right.x
        c2.p4 = right.y
        c2.p5 = right.z
        c2.p6 = up.x
        c2.p7 = up.y
        c2.p8 = up.z
        _ = self._rt_core()[].cmd_ring.push(c2)

    # ---- Deterministic Offline Quantum Render (Only when no RT thread is running!) ----

    def render_offline_quantum(
        mut self, mut out_lr: List[Float32], frames: Int = 256
    ):
        """Renders `frames` stereo frames deterministically when no RT thread is active
        (also services disk streams synchronously, so offline renders are deterministic)."""
        if self.rt_core_addr == 0 or self.rt_running:
            return
        self._rt_core()[].render_quantum(out_lr, frames)
        _ = self.drain_retired()
        _ = self.service_streams()

    # ---- Output devices, reconnect & robustness ----

    def audio_backend_name(self) -> String:
        """Returns the active RT output backend: `'headless'`, `'pulse'`, `'alsa'`,
        `'reconnecting'` (running, output lost, retrying with backoff) or `'none'`."""
        if self.rt_core_addr == 0 or not self.rt_running:
            return "none"
        var b = Int(self._rt_core()[]._telem_ptr(_TELEM_BACKEND)[unsafe_offset=0].load())
        if b == AUDIO_BACKEND_HEADLESS:
            return "headless"
        if b == AUDIO_BACKEND_PULSE:
            return "pulse"
        if b == AUDIO_BACKEND_ALSA:
            return "alsa"
        return "reconnecting"

    def reconnect_count(self) -> Int:
        """Successful output re-opens after a device loss."""
        if self.rt_core_addr == 0:
            return 0
        return Int(self._rt_core()[]._telem_ptr(_TELEM_RECONNECTS)[unsafe_offset=0].load())

    def device_lost_count(self) -> Int:
        if self.rt_core_addr == 0:
            return 0
        return Int(self._rt_core()[]._telem_ptr(_TELEM_DEVICE_LOST)[unsafe_offset=0].load())

    def simulate_device_loss(self) -> Bool:
        """Test hook: makes the RT thread treat its output as disconnected (exercises the
        reconnect-with-backoff path, also in headless mode)."""
        if self.rt_core_addr == 0 or not self.rt_running:
            return False
        self._rt_core()[]._telem_ptr(_TELEM_FAULT_INJECT)[unsafe_offset=0].store(1)
        return True

    def list_output_devices(self) -> List[AudioDeviceInfo]:
        """PulseAudio/PipeWire sinks (`pulse:<name>`) and ALSA nodes (`alsa:/dev/snd/pcmC*D*p`)."""
        return enumerate_output_devices()

    def output_device(self) -> String:
        if self.rt_core_addr == 0:
            return ""
        return self._rt_core()[].preferred_device

    def set_alsa_fallback(mut self, enabled: Bool):
        """Whether a failed PulseAudio connection falls back to raw ALSA (applies on next open)."""
        if self.rt_core_addr == 0:
            return
        var was_running = self.rt_running
        if was_running:
            self.stop_realtime_thread()
        self._rt_core()[].allow_alsa_fallback = enabled
        if was_running:
            _ = self.start_realtime_thread()

    def select_output_device(mut self, device_id: String) -> Bool:
        """Selects an output (`""` = system default). When the realtime thread is running it
        is restarted on the new device; returns whether output is running afterwards."""
        if self.rt_core_addr == 0:
            return False
        var was_running = self.rt_running
        var was_headless = self._rt_core()[].headless_rt_mode
        if was_running:
            self.stop_realtime_thread()
        self._rt_core()[].preferred_device = device_id
        if not was_running:
            return True
        if was_headless:
            return self.start_headless_rt_thread()
        return self.start_realtime_thread()

    # ---- Dedicated RT Thread Lifecycle (PulseAudio / PipeWire / Headless RT Pthread) ----

    def _spawn_rt_thread(mut self) -> Bool:
        self._rt_core()[]._running_ptr()[unsafe_offset=0].store(1)
        var tid = UInt64(0)
        var rc = external_call["pthread_create", Int32](
            Pointer(to=tid),
            Int(0),
            _audio_rt_thread_entry,
            self.rt_core_addr,
        )
        if rc != 0:
            self._rt_core()[]._running_ptr()[unsafe_offset=0].store(0)
            return False
        self.rt_thread_handle = tid
        self.rt_running = True
        _ = self._start_stream_thread()
        return True

    def start_realtime_thread(mut self) -> Bool:
        """Connects to `PulseClient` (or falls back to Linux ALSA `/dev/snd/pcmC*D*p`) and spawns the
        dedicated RT mixing pthread. If the device later disappears the RT thread keeps the mixer
        clock running and reconnects with exponential backoff (PulseAudio first, then ALSA)."""
        if (
            self.rt_running
            or self.rt_core_addr == 0
            or self._rt_core()[].running_atomic == 0
        ):
            return self.rt_running
        self._rt_core()[].headless_rt_mode = False
        if not self._rt_core()[].try_open_backend():
            return False
        if not self._spawn_rt_thread():
            self._rt_core()[].pulse.close()
            if self._rt_core()[].alsa_fd >= 0:
                alsa_close_pcm(self._rt_core()[].alsa_fd)
                self._rt_core()[].alsa_fd = -1
            self._rt_core()[].backend = AUDIO_BACKEND_NONE
            self._rt_core()[]._publish_backend()
            return False
        return True

    def start_headless_rt_thread(mut self) -> Bool:
        """Spawns the dedicated RT mixing pthread in headless hardware-free mode for CI/server verification."""
        if (
            self.rt_running
            or self.rt_core_addr == 0
            or self._rt_core()[].running_atomic == 0
        ):
            return self.rt_running
        self._rt_core()[].headless_rt_mode = True
        _ = self._rt_core()[].try_open_backend()
        if not self._spawn_rt_thread():
            self._rt_core()[].backend = AUDIO_BACKEND_NONE
            self._rt_core()[]._publish_backend()
            return False
        return True

    def stop_realtime_thread(mut self):
        if self.rt_core_addr == 0:
            return
        if self.rt_running:
            self._rt_core()[]._running_ptr()[unsafe_offset=0].store(0)
            _ = external_call["pthread_join", Int32](
                self.rt_thread_handle, Int(0)
            )
            self.rt_thread_handle = 0
            self.rt_running = False
        self._stop_stream_thread()
        self._rt_core()[].pulse.close()
        _ = self.drain_retired()
        if self._rt_core()[].alsa_fd >= 0:
            alsa_close_pcm(self._rt_core()[].alsa_fd)
            self._rt_core()[].alsa_fd = -1
        self._rt_core()[].backend = AUDIO_BACKEND_NONE
        self._rt_core()[]._publish_backend()

    def stop(mut self):
        self.stop_realtime_thread()
