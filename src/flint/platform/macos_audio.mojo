"""Pure-Mojo macOS CoreAudio / AudioToolbox (`AudioQueue`) audio backend (`flint.platform.macos_audio`).

Zero C/C++ wrapper libraries: implements `AudioBackend` with non-`Copyable` RAII
ownership, pure-Mojo `AudioStreamBasicDescription` (40 B `'lpcm'`) and
`AudioQueueBuffer` (56 B 64-bit) ABI packers, a 4-slot S16LE stereo PCM ring buffer,
and a deterministic polyphonic voice + ambient drone mixer.
"""
from std.atomic import Atomic, Ordering
from std.collections import List
from std.ffi import external_call
from std.math import sin
from std.memory import bitcast
from std.sys.info import CompilationTarget
from flint.platform.dsp import soft_clip_tanh
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.traits import AudioBackend

comptime MACOS_ASBD_BYTES: Int = 40
comptime MACOS_AQ_BUFFER_BYTES: Int = 56
comptime MACOS_AUDIO_RING_SLOTS: Int = 4
comptime MACOS_AUDIO_PERIOD_FRAMES: Int = 1024
comptime MACOS_AUDIO_SAMPLE_RATE: Int = 48000
comptime MACOS_AUDIO_MAX_VOICES: Int = 8
comptime _MACOS_AQ_STATE_BYTES: Int = 64  # [0..31]: 4x buf_ptr (Int), [32..47]: 4x Atomic[Int32] retired flags

comptime K_AUDIO_FORMAT_LINEAR_PCM: UInt32 = 0x6C70636D  # 'lpcm'
comptime K_AUDIO_FORMAT_FLAG_IS_FLOAT: UInt32 = 1
comptime K_AUDIO_FORMAT_FLAG_IS_SIGNED_INTEGER: UInt32 = 4
comptime K_AUDIO_FORMAT_FLAG_IS_PACKED: UInt32 = 8
comptime K_AUDIO_FORMAT_FLAGS_S16_PACKED: UInt32 = (
    K_AUDIO_FORMAT_FLAG_IS_SIGNED_INTEGER | K_AUDIO_FORMAT_FLAG_IS_PACKED
)  # 12
comptime K_AUDIO_FORMAT_FLAGS_F32_PACKED: UInt32 = (
    K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED
)  # 9


@always_inline
def _ma_put32(mut buf: List[UInt8], off: Int, val: UInt32):
    buf[off] = UInt8(val & 0xFF)
    buf[off + 1] = UInt8((val >> 8) & 0xFF)
    buf[off + 2] = UInt8((val >> 16) & 0xFF)
    buf[off + 3] = UInt8((val >> 24) & 0xFF)


@always_inline
def _ma_put64(mut buf: List[UInt8], off: Int, val: UInt64):
    for i in range(8):
        buf[off + i] = UInt8((val >> UInt64(i * 8)) & 0xFF)


@always_inline
def _ma_get32(buf: List[UInt8], off: Int) -> UInt32:
    return (
        UInt32(buf[off])
        | (UInt32(buf[off + 1]) << 8)
        | (UInt32(buf[off + 2]) << 16)
        | (UInt32(buf[off + 3]) << 24)
    )


@always_inline
def _ma_get64(buf: List[UInt8], off: Int) -> UInt64:
    var out = UInt64(0)
    for i in range(8):
        out |= UInt64(buf[off + i]) << UInt64(i * 8)
    return out


def macos_pack_asbd(
    sample_rate: Float64 = 48000.0,
    channels: UInt32 = 2,
    bits_per_channel: UInt32 = 16,
    is_float: Bool = False,
) -> List[UInt8]:
    """Packs a 40-byte CoreAudio `AudioStreamBasicDescription` (`'lpcm'`)."""
    var bytes_per_frame = channels * (bits_per_channel // 8)
    var flags = K_AUDIO_FORMAT_FLAGS_F32_PACKED if is_float else K_AUDIO_FORMAT_FLAGS_S16_PACKED
    var buf = List[UInt8](length=MACOS_ASBD_BYTES, fill=0)
    _ma_put64(buf, 0, bitcast[DType.uint64](sample_rate))  # mSampleRate
    _ma_put32(buf, 8, K_AUDIO_FORMAT_LINEAR_PCM)         # mFormatID ('lpcm')
    _ma_put32(buf, 12, flags)                            # mFormatFlags
    _ma_put32(buf, 16, bytes_per_frame)                  # mBytesPerPacket
    _ma_put32(buf, 20, 1)                                # mFramesPerPacket
    _ma_put32(buf, 24, bytes_per_frame)                  # mBytesPerFrame
    _ma_put32(buf, 28, channels)                         # mChannelsPerFrame
    _ma_put32(buf, 32, bits_per_channel)                 # mBitsPerChannel
    _ma_put32(buf, 36, 0)                                # mReserved
    return buf^


def macos_pack_audio_queue_buffer(
    capacity_bytes: UInt32,
    audio_data_ptr: UInt64,
    byte_size: UInt32,
    user_data: UInt64 = 0,
) -> List[UInt8]:
    """Packs a 56-byte 64-bit CoreAudio `AudioQueueBuffer` descriptor."""
    var buf = List[UInt8](length=MACOS_AQ_BUFFER_BYTES, fill=0)
    _ma_put32(buf, 0, capacity_bytes)  # mAudioDataBytesCapacity
    _ma_put64(buf, 8, audio_data_ptr)  # mAudioData
    _ma_put32(buf, 16, byte_size)      # mAudioDataByteSize
    _ma_put64(buf, 24, user_data)      # mUserData
    _ma_put32(buf, 32, 0)              # mPacketDescriptionCapacity
    _ma_put64(buf, 40, 0)              # mPacketDescriptions
    _ma_put32(buf, 48, 0)              # mPacketDescriptionCount
    return buf^


@fieldwise_init
struct _MacSynthVoice(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var clip_id: Int
    var phase: Float64
    var freq_hz: Float64
    var gain: Float32
    var age_samples: Int
    var duration_samples: Int

    def __init__(out self):
        self.active = False
        self.clip_id = 0
        self.phase = 0.0
        self.freq_hz = 220.0
        self.gain = 0.0
        self.age_samples = 0
        self.duration_samples = 0


def _macos_aq_output_callback(user_data: Int, queue: Int, aq_buf: Int) abi("C"):
    """CoreAudio `AudioQueueOutputCallback` (`abi("C")`): marks the matching buffer slot as retired."""
    _ = queue
    if user_data == 0 or aq_buf == 0:
        return
    for s in range(MACOS_AUDIO_RING_SLOTS):
        var slot_buf = Pointer[Int, origin=ImmutAnyOrigin](
            unsafe_from_address=user_data + s * 8
        )[unsafe_offset=0]
        if slot_buf == aq_buf:
            Pointer[Atomic[Int32], origin=MutAnyOrigin](
                unsafe_from_address=user_data + 32 + s * 4
            )[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int32(1))
            break


struct MacOsAudio(AudioBackend):
    """Pure-Mojo macOS `AudioBackend` (`AudioToolbox` `AudioQueue` + 4-slot PCM ring)."""
    var sample_rate: Int
    var channels: Int
    var period_frames: Int
    var ready: Bool
    var toolbox_lib: Int
    var audio_queue: Int
    var aq_state_ptr: Int
    var aq_buffers: List[Int]
    var ring_pcm: List[List[Int16]]
    var ring_cursor: Int
    var mixed_blocks: Int
    var frames_submitted: UInt64
    var drone_level: Float32
    var drone_gain: Float32
    var drone_phase: Float64
    var voices: List[_MacSynthVoice]
    var voice_cursor: Int

    def __init__(out self):
        self.sample_rate = MACOS_AUDIO_SAMPLE_RATE
        self.channels = 2
        self.period_frames = MACOS_AUDIO_PERIOD_FRAMES
        self.ready = False
        self.toolbox_lib = 0
        self.audio_queue = 0
        self.aq_state_ptr = 0
        self.aq_buffers = List[Int](length=MACOS_AUDIO_RING_SLOTS, fill=0)
        self.ring_pcm = List[List[Int16]](capacity=MACOS_AUDIO_RING_SLOTS)
        for _ in range(MACOS_AUDIO_RING_SLOTS):
            self.ring_pcm.append(List[Int16](length=MACOS_AUDIO_PERIOD_FRAMES * 2, fill=0))
        self.ring_cursor = 0
        self.mixed_blocks = 0
        self.frames_submitted = 0
        self.drone_level = 0.0
        self.drone_gain = 0.0
        self.drone_phase = 0.0
        self.voices = List[_MacSynthVoice](length=MACOS_AUDIO_MAX_VOICES, fill=_MacSynthVoice())
        self.voice_cursor = 0

    def __moveinit__(out self, deinit existing: Self):
        self.sample_rate = existing.sample_rate
        self.channels = existing.channels
        self.period_frames = existing.period_frames
        self.ready = existing.ready
        self.toolbox_lib = existing.toolbox_lib
        self.audio_queue = existing.audio_queue
        self.aq_state_ptr = existing.aq_state_ptr
        self.aq_buffers = existing.aq_buffers^
        self.ring_pcm = existing.ring_pcm^
        self.ring_cursor = existing.ring_cursor
        self.mixed_blocks = existing.mixed_blocks
        self.frames_submitted = existing.frames_submitted
        self.drone_level = existing.drone_level
        self.drone_gain = existing.drone_gain
        self.drone_phase = existing.drone_phase
        self.voices = existing.voices^
        self.voice_cursor = existing.voice_cursor
        existing.ready = False
        existing.toolbox_lib = 0
        existing.audio_queue = 0
        existing.aq_state_ptr = 0

    def __deinit__(deinit self):
        self.shutdown()

    def initialize(mut self) -> Bool:
        self.shutdown()
        comptime if CompilationTarget.is_macos():
            self.toolbox_lib = dl_open(
                "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"
            )
            if self.toolbox_lib == 0:
                return False
            var fn_new = dl_sym(self.toolbox_lib, "AudioQueueNewOutput")
            var fn_alloc = dl_sym(self.toolbox_lib, "AudioQueueAllocateBuffer")
            var fn_enq = dl_sym(self.toolbox_lib, "AudioQueueEnqueueBuffer")
            var fn_start = dl_sym(self.toolbox_lib, "AudioQueueStart")
            if fn_new == 0 or fn_alloc == 0 or fn_enq == 0 or fn_start == 0:
                self.shutdown()
                return False

            self.aq_state_ptr = external_call["calloc", Int](Int(1), Int(_MACOS_AQ_STATE_BYTES))
            if self.aq_state_ptr == 0:
                self.shutdown()
                return False

            var asbd = macos_pack_asbd(Float64(self.sample_rate), UInt32(self.channels), 16, False)
            comptime AqCbFn = def(Int, Int, Int) thin abi("C") -> None
            var cb: AqCbFn = _macos_aq_output_callback
            var cb_addr = Pointer(to=cb).unsafe_bitcast[Int]()[unsafe_offset=0]
            var q = 0
            comptime AqNewFn = def(Int, Int, Int, Int, Int, UInt32, Int) thin abi("C") -> Int32
            var status = Pointer(to=fn_new).unsafe_bitcast[AqNewFn]()[unsafe_offset=0](
                Int(asbd.unsafe_ptr()), cb_addr, self.aq_state_ptr, 0, 0, 0, Int(Pointer(to=q))
            )
            _ = len(asbd)
            if status != 0 or q == 0:
                self.shutdown()
                return False
            self.audio_queue = q

            var block_bytes = UInt32(self.period_frames * 4)
            comptime AqAllocFn = def(Int, UInt32, Int) thin abi("C") -> Int32
            comptime AqEnqFn = def(Int, Int, UInt32, Int) thin abi("C") -> Int32
            for s in range(MACOS_AUDIO_RING_SLOTS):
                var buf_ptr = 0
                if Pointer(to=fn_alloc).unsafe_bitcast[AqAllocFn]()[unsafe_offset=0](
                    self.audio_queue, block_bytes, Int(Pointer(to=buf_ptr))
                ) != 0 or buf_ptr == 0:
                    self.shutdown()
                    return False
                self.aq_buffers[s] = buf_ptr
                Pointer[Int, origin=MutAnyOrigin](
                    unsafe_from_address=self.aq_state_ptr + s * 8
                )[unsafe_offset=0] = buf_ptr
                Pointer[Atomic[Int32], origin=MutAnyOrigin](
                    unsafe_from_address=self.aq_state_ptr + 32 + s * 4
                )[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int32(0))
                var data_ptr = Int(
                    Pointer[UInt64, ImmutAnyOrigin](unsafe_from_address=buf_ptr + 8)[unsafe_offset=0]
                )
                if data_ptr != 0:
                    _ = external_call["memset", Int](data_ptr, Int32(0), Int64(block_bytes))
                Pointer[UInt32, MutAnyOrigin](unsafe_from_address=buf_ptr + 16)[unsafe_offset=0] = block_bytes
                if Pointer(to=fn_enq).unsafe_bitcast[AqEnqFn]()[unsafe_offset=0](
                    self.audio_queue, buf_ptr, 0, 0
                ) != 0:
                    Pointer[Atomic[Int32], origin=MutAnyOrigin](
                        unsafe_from_address=self.aq_state_ptr + 32 + s * 4
                    )[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int32(1))
                    self.shutdown()
                    return False

            comptime AqStartFn = def(Int, Int) thin abi("C") -> Int32
            if Pointer(to=fn_start).unsafe_bitcast[AqStartFn]()[unsafe_offset=0](
                self.audio_queue, 0
            ) != 0:
                self.shutdown()
                return False
            self.ready = True
            return True
        else:
            return False

    def trigger(mut self, clip_id: Int, pitch: Float32, gain: Float32):
        var slot = self.voice_cursor % MACOS_AUDIO_MAX_VOICES
        self.voice_cursor += 1
        var base_hz = 196.0 + Float64(abs(clip_id) % 12) * 32.0
        var p = max(Float64(0.25), min(Float64(4.0), Float64(pitch)))
        self.voices[slot] = _MacSynthVoice(
            True,
            clip_id,
            0.0,
            base_hz * p,
            max(Float32(0.0), min(Float32(2.0), gain)),
            0,
            self.sample_rate // 4,
        )

    def set_ambient_drone(mut self, level: Float32):
        self.drone_level = max(Float32(0.0), min(Float32(1.0), level))

    def mix_into_slot(mut self, slot_idx: Int, master_gain: Float32, frames: Int):
        var mg = max(Float32(0.0), min(Float32(2.0), master_gain))
        var count = max(1, min(self.period_frames, frames))
        var two_pi = 6.283185307179586
        var inv_sr = 1.0 / Float64(self.sample_rate)
        for f in range(count):
            var mix_l = Float32(0.0)
            var mix_r = Float32(0.0)
            self.drone_gain += (self.drone_level - self.drone_gain) * Float32(0.002)
            if self.drone_gain > Float32(1e-5):
                var d = Float32(sin(self.drone_phase * two_pi)) * self.drone_gain * Float32(0.22)
                mix_l += d
                mix_r += d * Float32(0.96)
                self.drone_phase += 55.0 * inv_sr
                if self.drone_phase >= 1.0:
                    self.drone_phase -= 1.0
            for v in range(MACOS_AUDIO_MAX_VOICES):
                if not self.voices[v].active:
                    continue
                var age = self.voices[v].age_samples
                var dur = max(1, self.voices[v].duration_samples)
                var env = Float32(1.0 - Float64(age) / Float64(dur))
                var sig = Float32(sin(self.voices[v].phase * two_pi)) * self.voices[v].gain * env * Float32(0.35)
                var pan = Float32(((self.voices[v].clip_id % 5) - 2)) * Float32(0.15)
                mix_l += sig * (Float32(1.0) - max(Float32(0.0), pan))
                mix_r += sig * (Float32(1.0) + min(Float32(0.0), pan))
                self.voices[v].phase += self.voices[v].freq_hz * inv_sr
                if self.voices[v].phase >= 1.0:
                    self.voices[v].phase -= 1.0
                self.voices[v].age_samples += 1
                if self.voices[v].age_samples >= dur:
                    self.voices[v].active = False

            mix_l = soft_clip_tanh(mix_l * mg)
            mix_r = soft_clip_tanh(mix_r * mg)
            self.ring_pcm[slot_idx][f * 2] = Int16(
                Int(max(Float32(-1.0), min(Float32(1.0), mix_l)) * Float32(32767.0))
            )
            self.ring_pcm[slot_idx][f * 2 + 1] = Int16(
                Int(max(Float32(-1.0), min(Float32(1.0), mix_r)) * Float32(32767.0))
            )

    def pump(mut self, master_gain: Float32):
        comptime if CompilationTarget.is_macos():
            if self.ready and self.audio_queue != 0 and self.toolbox_lib != 0 and self.aq_state_ptr != 0:
                var fn_enq = dl_sym(self.toolbox_lib, "AudioQueueEnqueueBuffer")
                var start = self.ring_cursor % MACOS_AUDIO_RING_SLOTS
                var byte_len = UInt32(self.period_frames * 4)
                comptime AqEnqFn = def(Int, Int, UInt32, Int) thin abi("C") -> Int32
                for off in range(MACOS_AUDIO_RING_SLOTS):
                    var s = (start + off) % MACOS_AUDIO_RING_SLOTS
                    var flag_ptr = Pointer[Atomic[Int32], origin=MutAnyOrigin](
                        unsafe_from_address=self.aq_state_ptr + 32 + s * 4
                    )
                    if flag_ptr[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]() == 0:
                        continue
                    var buf_ptr = self.aq_buffers[s]
                    if buf_ptr == 0:
                        continue
                    self.mix_into_slot(s, master_gain, self.period_frames)
                    self.ring_cursor = (s + 1) % MACOS_AUDIO_RING_SLOTS
                    self.mixed_blocks += 1
                    var data_ptr = Int(
                        Pointer[UInt64, ImmutAnyOrigin](unsafe_from_address=buf_ptr + 8)[unsafe_offset=0]
                    )
                    if data_ptr != 0 and fn_enq != 0:
                        _ = external_call["memcpy", Int](
                            data_ptr, Int(self.ring_pcm[s].unsafe_ptr()), Int64(byte_len)
                        )
                        Pointer[UInt32, MutAnyOrigin](unsafe_from_address=buf_ptr + 16)[unsafe_offset=0] = byte_len
                        flag_ptr[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int32(0))
                        if Pointer(to=fn_enq).unsafe_bitcast[AqEnqFn]()[unsafe_offset=0](
                            self.audio_queue, buf_ptr, 0, 0
                        ) == 0:
                            self.frames_submitted += UInt64(self.period_frames)
                        else:
                            flag_ptr[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int32(1))
                return
        var slot = self.ring_cursor % MACOS_AUDIO_RING_SLOTS
        self.mix_into_slot(slot, master_gain, self.period_frames)
        self.ring_cursor = (slot + 1) % MACOS_AUDIO_RING_SLOTS
        self.mixed_blocks += 1

    def last_mixed_slot(self) -> List[Int16]:
        if self.mixed_blocks == 0:
            return List[Int16]()
        var prev = (self.ring_cursor + MACOS_AUDIO_RING_SLOTS - 1) % MACOS_AUDIO_RING_SLOTS
        return self.ring_pcm[prev].copy()

    def shutdown(mut self):
        comptime if CompilationTarget.is_macos():
            if self.audio_queue != 0 and self.toolbox_lib != 0:
                var fn_stop = dl_sym(self.toolbox_lib, "AudioQueueStop")
                var fn_free = dl_sym(self.toolbox_lib, "AudioQueueFreeBuffer")
                var fn_disp = dl_sym(self.toolbox_lib, "AudioQueueDispose")
                if fn_stop != 0:
                    comptime AqStopFn = def(Int, UInt8) thin abi("C") -> Int32
                    _ = Pointer(to=fn_stop).unsafe_bitcast[AqStopFn]()[unsafe_offset=0](
                        self.audio_queue, 1
                    )
                if fn_free != 0:
                    comptime AqFreeFn = def(Int, Int) thin abi("C") -> Int32
                    for s in range(len(self.aq_buffers)):
                        if self.aq_buffers[s] != 0:
                            _ = Pointer(to=fn_free).unsafe_bitcast[AqFreeFn]()[unsafe_offset=0](
                                self.audio_queue, self.aq_buffers[s]
                            )
                            self.aq_buffers[s] = 0
                if fn_disp != 0:
                    comptime AqDispFn = def(Int, UInt8) thin abi("C") -> Int32
                    _ = Pointer(to=fn_disp).unsafe_bitcast[AqDispFn]()[unsafe_offset=0](
                        self.audio_queue, 1
                    )
            if self.aq_state_ptr != 0:
                external_call["free", NoneType](self.aq_state_ptr)
            _ = dl_close(self.toolbox_lib)
        for s in range(len(self.aq_buffers)):
            self.aq_buffers[s] = 0
        self.ready = False
        self.toolbox_lib = 0
        self.audio_queue = 0
        self.aq_state_ptr = 0
