"""Original WAV playback, CPU mixing, and direct Linux PCM output.

The decoder handles uncompressed RIFF/WAVE PCM without external codecs. Playback
uses the documented ALSA kernel interface directly; no audio client library is
loaded or linked. Failure to open or configure a device leaves the game silent.
"""
from std.ffi import external_call
from std.math import floor, cos, abs, sqrt
from flint.core.math import Vec3
from flint.platform.os import read_bytes, close_fd, env_value
from flint.platform.pulse import PulseClient
from flint.platform.dsp import (
    StereoReverbFDN,
    compute_spatial_3d,
    equal_power_gains,
    soft_clip_tanh,
)


def wav_u16(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 2 > len(data):
        return 0
    return Int(data[offset]) | (Int(data[offset + 1]) << 8)


def wav_u32(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)


def audio_put32(mut data: List[UInt8], offset: Int, value: Int):
    var bits = UInt32(value)
    data[offset] = UInt8(bits & 255)
    data[offset + 1] = UInt8((bits >> 8) & 255)
    data[offset + 2] = UInt8((bits >> 16) & 255)
    data[offset + 3] = UInt8((bits >> 24) & 255)


def audio_put64(mut data: List[UInt8], offset: Int, value: UInt64):
    audio_put32(data, offset, Int(value & 0xFFFFFFFF))
    audio_put32(data, offset + 4, Int((value >> 32) & 0xFFFFFFFF))


def audio_i64(data: List[UInt8], offset: Int) -> Int:
    var value = UInt64(wav_u32(data, offset)) | (UInt64(wav_u32(data, offset + 4)) << 32)
    return Int(value)


def riff_tag(data: List[UInt8], offset: Int, a: UInt8, b: UInt8, c: UInt8, d: UInt8) -> Bool:
    return offset + 4 <= len(data) and data[offset] == a and data[offset + 1] == b and data[offset + 2] == c and data[offset + 3] == d


struct WavClip:
    var valid: Bool
    var channels: Int
    var sample_rate: Int
    var samples: List[Int16]

    def __init__(out self):
        self.valid = False
        self.channels = 0
        self.sample_rate = 0
        self.samples = List[Int16]()

    def frame_count(self) -> Int:
        if self.channels <= 0:
            return 0
        return len(self.samples) // self.channels


def decode_wav(data: List[UInt8]) -> WavClip:
    var clip = WavClip()
    if len(data) < 12 or not riff_tag(data, 0, 82, 73, 70, 70) or not riff_tag(data, 8, 87, 65, 86, 69):
        return clip^
    var declared_size = wav_u32(data, 4)
    if declared_size + 8 > len(data) or declared_size < 4:
        return clip^
    var encoding = 0
    var bits = 0
    var block_align = 0
    var payload = List[UInt8]()
    var offset = 12
    var limit = min(len(data), declared_size + 8)
    while offset + 8 <= limit:
        var chunk_size = wav_u32(data, offset + 4)
        var chunk_start = offset + 8
        if chunk_size > limit - chunk_start:
            return clip^
        if riff_tag(data, offset, 102, 109, 116, 32):
            if chunk_size < 16:
                return clip^
            encoding = wav_u16(data, chunk_start)
            clip.channels = wav_u16(data, chunk_start + 2)
            clip.sample_rate = wav_u32(data, chunk_start + 4)
            block_align = wav_u16(data, chunk_start + 12)
            bits = wav_u16(data, chunk_start + 14)
        elif riff_tag(data, offset, 100, 97, 116, 97):
            if len(payload) == 0:
                for index in range(chunk_size):
                    payload.append(data[chunk_start + index])
        offset = chunk_start + chunk_size + (chunk_size & 1)
    if encoding != 1 or clip.channels < 1 or clip.channels > 2 or clip.sample_rate < 1:
        return WavClip()
    if bits != 8 and bits != 16 and bits != 24 and bits != 32:
        return WavClip()
    var bytes_per_sample = bits // 8
    var expected_align = bytes_per_sample * clip.channels
    if block_align != expected_align or len(payload) < expected_align:
        return WavClip()
    var frames = len(payload) // expected_align
    if frames < 1 or frames > 100000000:
        return WavClip()
    clip.samples.reserve(frames * clip.channels)
    for frame in range(frames):
        for channel in range(clip.channels):
            var at = (frame * clip.channels + channel) * bytes_per_sample
            var sample: Int
            if bits == 8:
                sample = (Int(payload[at]) - 128) << 8
            elif bits == 16:
                sample = wav_u16(payload, at)
                if sample >= 32768:
                    sample -= 65536
            elif bits == 24:
                sample = Int(payload[at]) | (Int(payload[at + 1]) << 8) | (Int(payload[at + 2]) << 16)
                if sample >= 8388608:
                    sample -= 16777216
                sample >>= 8
            else:
                var unsigned = wav_u32(payload, at)
                sample = unsigned - 4294967296 if unsigned >= 2147483648 else unsigned
                sample >>= 16
            clip.samples.append(Int16(max(-32768, min(32767, sample))))
    clip.valid = True
    return clip^


def load_wav(path: String) -> WavClip:
    return decode_wav(read_bytes(path))


def clip_sample(clip: WavClip, position: Float64, channel: Int, looping: Bool = False) -> Float32:
    var frames = clip.frame_count()
    if not clip.valid or frames == 0:
        return 0
    var base = Int(floor(position))
    if looping:
        base = base % frames
    else:
        base = max(0, min(frames - 1, base))
    var following = (base + 1) % frames if looping else min(base + 1, frames - 1)
    var source_channel = min(channel, clip.channels - 1)
    var first = Float32(clip.samples[base * clip.channels + source_channel]) / 32768.0
    var second = Float32(clip.samples[following * clip.channels + source_channel]) / 32768.0
    var fraction = Float32(position - floor(position))
    return first + (second - first) * fraction


# Package 3.15: the mixer's buses. Each has its own volume from the options;
# the gain is applied while mixing, so a change reaches sounds already playing.
comptime BUS_EFFECTS: Int = 0
comptime BUS_SPEECH: Int = 1
comptime BUS_MUSIC: Int = 2
comptime BUS_COUNT: Int = 3

# Package 3.24: the low hum. The sonar ping (clip 3: a 501 Hz tone that rings
# for a second) read slowly, so it sounds near 100 Hz, from the middle of its
# ring, by two read heads half a loop apart, each under a raised-cosine
# window: the windows add up to one, so the loop has no seam to click at.
comptime DRONE_CLIP: Int = 3
comptime DRONE_START: Float32 = 0.12   # of the clip: after the ping's attack
comptime DRONE_SPAN: Float32 = 0.55    # of the clip: the part that is looped
comptime DRONE_PITCH: Float32 = 0.19   # at level 0; 501 Hz becomes about 95 Hz
comptime DRONE_PITCH_RISE: Float32 = 0.025  # more at level 1 (about a tone higher)
comptime DRONE_GAIN: Float32 = 0.60    # at level 1, before the effects bus
comptime DRONE_EASE_SECONDS: Float32 = 0.4
comptime DRONE_LOW_PASS: Float32 = 0.08  # one-pole coefficient: takes the ping's edge off


struct AudioVoice:
    var active: Bool
    var clip: Int
    var position: Float64
    var pitch: Float32
    var volume: Float32
    var pan: Float32
    var lpf_coeff: Float32
    var lp_l: Float32
    var lp_r: Float32
    var reverb_send: Float32
    var priority: Int
    var order: UInt64
    var bank: Int  # the 2.5 sound bank it plays for, or -1
    var bus: Int   # BUS_EFFECTS or BUS_SPEECH (package 3.15)
    var elevation: Float32  # V10 Binaural HRTF pinna elevation (-1..+1)
    var hull_conducted: Bool  # V10 True for internal airframe/cockpit vibration (bypasses vacuum exterior muffle)

    def __init__(out self):
        self.bus = BUS_EFFECTS
        self.active = False
        self.clip = -1
        self.position = 0
        self.pitch = 1
        self.volume = 1
        self.pan = 0.0
        self.lpf_coeff = 1.0
        self.lp_l = 0.0
        self.lp_r = 0.0
        self.reverb_send = 0.08
        self.priority = 0
        self.order = 0
        self.bank = -1
        self.elevation = 0.0
        self.hull_conducted = False


struct NativeAudio:
    var clips: List[WavClip]
    var voices: List[AudioVoice]
    var music: WavClip
    var music_loaded: Bool
    var music_position: Float64
    var music_pitch: Float32
    var music_volume: Float32
    var music_pitch_follow: Bool
    var fd: Int
    var ready: Bool
    var output_rate: Int
    var period_frames: Int
    var buffer_frames: Int
    var pending: List[Int16]
    var pulse: PulseClient
    var frames_submitted: UInt64
    var xruns: Int
    var last_error: String
    var next_order: UInt64
    var events_accepted: UInt64
    var events_dropped: UInt64
    var voices_stolen: UInt64
    var bank_limited: UInt64  # plays a bank's voice limit dropped (2.5)
    var bus_volume: List[Float32]  # per BUS_* (package 3.15)
    var mono: Bool                 # both channels carry (L+R)/2 (package 3.15)
    var output_name: String        # the sound server sink in use; empty: the default
    # Package 3.24: the low hum, a looping drone made from an existing clip.
    var drone_clip: Int            # which clip it is made from (DRONE_CLIP)
    var drone_level: Float32       # what the game asked for, 0 to 1
    var drone_gain: Float32        # the gain now, eased toward the level's
    var drone_phase: Float64       # where the first read head is in the loop, 0 to 1
    var drone_low: Float32         # the low-pass filter's state (L)
    var drone_low_r: Float32       # the low-pass filter's state (R)
    var reverb: StereoReverbFDN    # live 4x4 Householder FDN spatial reverb bus
    var reverb_enabled: Bool
    var speech_duck: Float32       # smooth sidechain auto-ducking multiplier while BUS_SPEECH is active
    var hrtf_enabled: Bool         # V10 Woodworth ITD interaural delay + head-shadow + pinna elevation cue
    var vacuum_mix: Float32        # V10 0.0 = atmosphere, 1.0 = orbital space vacuum muffling for external SFX

    def __init__(out self):
        self.drone_clip = DRONE_CLIP
        self.drone_level = 0
        self.drone_gain = 0
        self.drone_phase = 0
        self.drone_low = 0
        self.drone_low_r = 0
        self.reverb = StereoReverbFDN(feedback=0.74, damp=0.36)
        self.reverb_enabled = True
        self.speech_duck = 1.0
        self.hrtf_enabled = True
        self.vacuum_mix = 0.0
        self.bus_volume = List[Float32](length=BUS_COUNT, fill=1)
        self.mono = False
        self.output_name = String()
        self.clips = List[WavClip]()
        self.voices = List[AudioVoice]()
        for _ in range(32):
            self.voices.append(AudioVoice())
        self.music = WavClip()
        self.music_loaded = False
        self.music_position = 0
        self.music_pitch = 1
        self.music_volume = 0.24
        self.music_pitch_follow = True
        self.fd = -1
        self.ready = False
        self.output_rate = 48000
        self.period_frames = 1024
        self.buffer_frames = 4096
        self.pending = List[Int16]()
        self.pulse = PulseClient()
        self.frames_submitted = 0
        self.xruns = 0
        self.last_error = ""
        self.next_order = 1
        self.events_accepted = 0
        self.events_dropped = 0
        self.voices_stolen = 0
        self.bank_limited = 0

    def load_assets(mut self):
        var paths = [
            "assets/audio/punch.wav", "assets/audio/stone.wav", "assets/audio/collect.wav",
            "assets/audio/sonar.wav", "assets/audio/transform.wav", "assets/audio/explosion.wav",
            "assets/audio/hurt.wav", "assets/audio/jump.wav",
            # Deepsong's own: 8 splash, 9 crumble, 10 launch; 11 is empty (it was
            # Echo's song, package 3.21) so the menu sounds keep their ids.
            "assets/deepsong/audio/splash.wav", "assets/deepsong/audio/crumble.wav",
            "assets/deepsong/audio/launch.wav", "",
            # 12-15: menu move, confirm, back, refused.
            "assets/deepsong/audio/menu_move.wav", "assets/deepsong/audio/menu_confirm.wav",
            "assets/deepsong/audio/menu_back.wav", "assets/deepsong/audio/menu_error.wav",
        ]
        self.clips.clear()
        for path in paths:
            self.clips.append(load_wav(path))
        self.music = load_wav("assets/audio/depth-theme.wav")
        self.music_loaded = self.music.valid
        self.music_position = 0

    def load_clips(mut self, paths: List[String]) -> Int:
        """Appends clips after the fixed sounds (package 2.5: the sound banks'
        variations) and returns the first one's index. A file that does not
        load stays an invalid clip, which plays as silence."""
        var first = len(self.clips)
        for path in paths:
            self.clips.append(load_wav(path))
        return first

    def errno(self) -> Int:
        var pointer = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
        return Int(pointer[unsafe_offset=0])

    def configure_device(mut self) -> Bool:
        # Linux UAPI struct snd_pcm_hw_params, 608 bytes on 64-bit Linux.
        var params = List[UInt8](length=608, fill=0)
        for mask in range(3):
            for word in range(8):
                audio_put32(params, 4 + mask * 32 + word * 4, Int(0xFFFFFFFF))
        for interval in range(12):
            var at = 260 + interval * 12
            audio_put32(params, at, 0)
            audio_put32(params, at + 4, Int(0xFFFFFFFF))
            # No integer flag here: time intervals are fractional. A 1024-frame
            # period at 48 kHz lasts 21333.33 microseconds.
        # RW interleaved, signed 16-bit little endian, standard subformat.
        for mask in range(3):
            for word in range(8):
                audio_put32(params, 4 + mask * 32 + word * 4, 0)
        audio_put32(params, 4, 1 << 3)
        audio_put32(params, 4 + 32, 1 << 2)
        audio_put32(params, 4 + 64, 1)
        # Exact stereo, 48 kHz, 1024-frame periods, and a four-period buffer.
        for index in range(4):
            var parameter = 10 if index == 0 else (11 if index == 1 else (13 if index == 2 else 17))
            var value = 2 if index == 0 else (48000 if index == 1 else (1024 if index == 2 else 4096))
            var at = 260 + (parameter - 8) * 12
            audio_put32(params, at, value)
            audio_put32(params, at + 4, value)
            audio_put32(params, at + 8, 4)
        audio_put32(params, 512, 0x000FFFFF)
        var result = external_call["ioctl", Int32](Int32(self.fd), UInt64(0xC2604110), params.unsafe_ptr())
        if result != 0:
            self.last_error = "PCM HW_REFINE errno " + String(self.errno())
            return False
        result = external_call["ioctl", Int32](Int32(self.fd), UInt64(0xC2604111), params.unsafe_ptr())
        if result != 0:
            self.last_error = "PCM HW_PARAMS errno " + String(self.errno())
            return False
        self.period_frames = wav_u32(params, 260 + (13 - 8) * 12)
        self.buffer_frames = wav_u32(params, 260 + (17 - 8) * 12)
        if self.period_frames < 64 or self.period_frames > 32768 or self.buffer_frames < self.period_frames:
            self.last_error = "PCM returned invalid period or buffer size"
            return False

        # Linux UAPI struct snd_pcm_sw_params, 136 bytes on 64-bit Linux.
        var software = List[UInt8](length=136, fill=0)
        audio_put32(software, 4, 1)  # period_step
        audio_put64(software, 16, UInt64(self.period_frames))  # avail_min
        audio_put64(software, 24, 1)  # obsolete xfer_align must be one
        audio_put64(software, 32, UInt64(self.period_frames))  # start_threshold
        audio_put64(software, 40, UInt64(self.buffer_frames))  # stop_threshold
        audio_put64(software, 64, UInt64(0x4000000000000000))  # boundary
        result = external_call["ioctl", Int32](Int32(self.fd), UInt64(0xC0884113), software.unsafe_ptr())
        if result != 0:
            self.last_error = "PCM SW_PARAMS errno " + String(self.errno())
            return False
        result = external_call["ioctl", Int32](Int32(self.fd), UInt64(0x4140), params.unsafe_ptr())
        if result != 0:
            self.last_error = "PCM PREPARE errno " + String(self.errno())
            return False
        return True

    def open_device(mut self, path: String) -> Bool:
        if self.fd >= 0:
            close_fd(self.fd)
        self.fd = -1
        self.ready = False
        self.pending.clear()
        var owned = path
        self.fd = Int(external_call["open", Int32](owned.as_c_string_span(), Int32(1 | 2048), Int32(0)))
        if self.fd < 0:
            self.last_error = "cannot open " + path + " errno " + String(self.errno())
            return False
        if not self.configure_device():
            close_fd(self.fd)
            self.fd = -1
            return False
        self.ready = True
        self.last_error = ""
        return True

    def initialize(mut self) -> Bool:
        self.load_assets()
        var requested = env_value("SD_AUDIO_DEVICE")
        if requested.byte_length() > 0:
            return self.open_device(requested)
        # The desktop's sound server comes first: it shares the output with other
        # programs and follows the user's chosen device.
        self.period_frames = 1024
        if self.pulse.connect(self.output_rate, self.period_frames * 4):
            self.ready = True
            self.last_error = ""
            return True
        self.last_error = self.pulse.error
        for card in range(4):
            if self.open_device("/dev/snd/pcmC" + String(card) + "D0p"):
                return True
        return False

    def close(mut self):
        self.pulse.close()
        if self.fd >= 0:
            var unused = List[UInt8](length=1, fill=0)
            _ = external_call["ioctl", Int32](Int32(self.fd), UInt64(0x4143), unused.unsafe_ptr())  # DROP
            close_fd(self.fd)
        self.fd = -1
        self.ready = False
        self.pending.clear()
        for index in range(len(self.voices)):
            self.voices[index].active = False

    def play_sound(mut self, sound_id: Int, pitch: Float32 = 1, volume: Float32 = 0.7, priority: Int = 1):
        self.play_voice(sound_id, pitch, volume, priority, -1, 0)

    def play_speech(mut self, sound_id: Int, pitch: Float32 = 1, volume: Float32 = 1, priority: Int = 2):
        """A clip on the speech bus (package 3.15; the narrator's voice, 3.28)."""
        self.play_voice(sound_id, pitch, volume, priority, -1, 0, BUS_SPEECH)

    def stop_speech(mut self):
        """Cleanly stops any currently playing voice on `BUS_SPEECH` so new tactical guidance never overlaps."""
        for index in range(len(self.voices)):
            if self.voices[index].active and self.voices[index].bus == BUS_SPEECH:
                self.voices[index].active = False

    def is_speech_playing(self) -> Bool:
        """True while any voice on `BUS_SPEECH` is actively playing."""
        for index in range(len(self.voices)):
            if self.voices[index].active and self.voices[index].bus == BUS_SPEECH:
                return True
        return False

    def speech_rms_level(self) -> Float32:
        """Returns real-time `[0.0, 1.0]` peak envelope at the active `BUS_SPEECH` read head for HUD voice oscilloscopes."""
        var best = Float32(0.0)
        for index in range(len(self.voices)):
            if not self.voices[index].active or self.voices[index].bus != BUS_SPEECH:
                continue
            var cid = self.voices[index].clip
            if cid < 0 or cid >= len(self.clips) or not self.clips[cid].valid:
                continue
            var pos = Int(floor(self.voices[index].position))
            var n_frames = self.clips[cid].frame_count()
            var ch = self.clips[cid].channels
            var span = min(256, max(0, n_frames - pos))
            var peak = Float32(0.0)
            for k in range(0, span, 4):
                var s = abs(Float32(self.clips[cid].samples[(pos + k) * ch]) / Float32(32768.0))
                if s > peak:
                    peak = s
            if peak > best:
                best = peak
        return min(Float32(1.0), best * Float32(1.35))

    def set_music_clip(
        mut self,
        var clip: WavClip,
        volume: Float32 = 0.24,
        pitch_follow: Bool = False,
    ):
        """Assigns an in-memory `WavClip` to the stereo looping `BUS_MUSIC` channel."""
        self.music = clip^
        self.music_loaded = self.music.valid
        self.music_position = 0
        self.music_pitch = 1.0
        self.music_volume = max(Float32(0.0), min(Float32(1.0), volume))
        self.music_pitch_follow = pitch_follow

    def set_bus_volumes(mut self, effects: Float32, speech: Float32, music: Float32):
        """The options' three volumes, each 0 to 1."""
        self.bus_volume[BUS_EFFECTS] = max(Float32(0), min(Float32(1), effects))
        self.bus_volume[BUS_SPEECH] = max(Float32(0), min(Float32(1), speech))
        self.bus_volume[BUS_MUSIC] = max(Float32(0), min(Float32(1), music))

    def set_drone(mut self, level: Float32):
        """The low hum's level, 0 (silent) to 1 (package 3.24). The gain
        follows it over DRONE_EASE_SECONDS while mixing."""
        self.drone_level = max(Float32(0), min(Float32(1), level))

    def mix_drone(mut self, mut mixed: List[Float32], count: Int):
        """Adds the stereo throttle-coupled turbine/ambient hum to a mixed block."""
        if self.drone_level <= 0 and self.drone_gain < 1.0e-5:
            self.drone_gain = 0
            self.drone_low = 0
            self.drone_low_r = 0
            return
        if self.drone_clip < 0 or self.drone_clip >= len(self.clips) or not self.clips[self.drone_clip].valid:
            return
        var frames = self.clips[self.drone_clip].frame_count()
        var start = Float64(Float32(frames) * DRONE_START)
        var span = Float64(Float32(frames) * DRONE_SPAN)
        if span < 16:
            return
        var pitch = DRONE_PITCH + DRONE_PITCH_RISE * self.drone_level
        var step = Float64(pitch) * Float64(self.clips[self.drone_clip].sample_rate) / Float64(self.output_rate) / span
        var ease = 1.0 / (DRONE_EASE_SECONDS * Float32(self.output_rate))
        var target = DRONE_GAIN * self.drone_level
        var lp_coeff = min(Float32(0.55), DRONE_LOW_PASS + self.drone_level * Float32(0.28))
        var bus = self.bus_volume[BUS_EFFECTS] * (Float32(0.45) + Float32(0.55) * self.speech_duck)
        var phase = self.drone_phase
        var gain = self.drone_gain
        var low_l = self.drone_low
        var low_r = self.drone_low_r
        for frame in range(count):
            var other = phase + 0.5
            if other >= 1:
                other -= 1
            var w = Float32(0.5 - 0.5 * cos(phase * 6.283185307179586))
            var al = clip_sample(self.clips[self.drone_clip], start + phase * span, 0)
            var bl = clip_sample(self.clips[self.drone_clip], start + other * span, 0)
            var ar = clip_sample(self.clips[self.drone_clip], start + phase * span, 1)
            var br = clip_sample(self.clips[self.drone_clip], start + other * span, 1)
            low_l += (al * w + bl * (1 - w) - low_l) * lp_coeff
            low_r += (ar * w + br * (1 - w) - low_r) * lp_coeff
            gain += (target - gain) * ease
            var g = gain * bus
            mixed[frame * 2] += low_l * g
            mixed[frame * 2 + 1] += low_r * g
            phase += step
            if phase >= 1:
                phase -= 1
        self.drone_phase = phase
        self.drone_gain = gain
        self.drone_low = low_l
        self.drone_low_r = low_r

    def set_mono(mut self, on: Bool):
        self.mono = on

    def list_outputs(mut self) -> List[String]:
        """The sound server's outputs by description, in its order (empty
        without a server). `output_names` gives the names to pass to
        `set_output`, in the same order."""
        var out = List[String]()
        for sink in self.pulse.list_sinks():
            out.append(sink.description)
        return out^

    def output_names(mut self) -> List[String]:
        var out = List[String]()
        for sink in self.pulse.list_sinks():
            out.append(sink.name)
        return out^

    def set_output(mut self, name: String) -> Bool:
        """Plays through the sink named `name` (empty: the default). Only the
        sound server path can choose; on failure it goes back to the default
        output and returns False."""
        if name == self.output_name and self.pulse.connected():
            return True
        if self.fd >= 0 and not self.pulse.connected():
            return name.byte_length() == 0  # a raw ALSA device has no choice
        self.pulse.close()
        self.pulse = PulseClient()
        self.pending.clear()
        if self.pulse.connect(self.output_rate, self.period_frames * 4, name):
            self.output_name = name
            self.ready = True
            self.last_error = ""
            return True
        var why = self.pulse.error
        self.pulse.close()
        self.pulse = PulseClient()
        self.output_name = String()
        self.ready = self.pulse.connect(self.output_rate, self.period_frames * 4)
        self.last_error = "output " + name + ": " + why
        return False

    def bank_voices(self, bank: Int) -> Int:
        var count = 0
        for index in range(len(self.voices)):
            if self.voices[index].active and self.voices[index].bank == bank:
                count += 1
        return count

    def set_hrtf(mut self, enabled: Bool):
        """Enable or disable V10 Binaural HRTF (Woodworth ITD sample delay + head-shadow + pinna elevation)."""
        self.hrtf_enabled = enabled

    def set_vacuum_acoustics(mut self, mix: Float32):
        """Set orbital vacuum acoustic muffling (0.0 = full atmosphere, 1.0 = deep space sub-bass conduction)."""
        self.vacuum_mix = max(Float32(0.0), min(Float32(1.0), mix))

    def play_voice(
        mut self,
        sound_id: Int,
        pitch: Float32,
        volume: Float32,
        priority: Int,
        bank: Int,
        limit: Int,
        bus: Int = BUS_EFFECTS,
        pan: Float32 = 0.0,
        lpf_coeff: Float32 = 1.0,
        reverb_send: Float32 = 0.08,
        elevation: Float32 = 0.0,
        hull_conducted: Bool = False,
    ):
        """Starts a clip. A bank play (bank >= 0) with `limit` of the bank's
        voices already sounding takes the bank's lowest-priority, oldest voice
        if that is of no higher priority, else it is dropped; otherwise a free
        voice, else the lowest-priority oldest voice of lower priority."""
        if sound_id < 0 or sound_id >= len(self.clips) or not self.clips[sound_id].valid:
            self.events_dropped += 1
            return
        var selected = -1
        if bank >= 0 and limit > 0 and self.bank_voices(bank) >= limit:
            var lowest = 2147483647
            var oldest = self.next_order
            for index in range(len(self.voices)):
                ref voice = self.voices[index]
                if not voice.active or voice.bank != bank:
                    continue
                if voice.priority < lowest or (voice.priority == lowest and voice.order < oldest):
                    lowest = voice.priority
                    oldest = voice.order
                    selected = index
            if selected < 0 or lowest > priority:
                self.events_dropped += 1
                self.bank_limited += 1
                return
            self.voices_stolen += 1
        if selected < 0:
            for index in range(len(self.voices)):
                if not self.voices[index].active:
                    selected = index
                    break
        if selected < 0:
            var lowest = 2147483647
            var oldest = self.next_order
            for index in range(len(self.voices)):
                if self.voices[index].priority < lowest or (self.voices[index].priority == lowest and self.voices[index].order < oldest):
                    lowest = self.voices[index].priority
                    oldest = self.voices[index].order
                    selected = index
            if lowest >= priority:
                self.events_dropped += 1
                return
            self.voices_stolen += 1
        self.voices[selected].active = True
        self.voices[selected].clip = sound_id
        self.voices[selected].position = 0
        self.voices[selected].pitch = max(Float32(0.25), min(Float32(4), pitch))
        self.voices[selected].volume = max(Float32(0), min(Float32(1.25), volume))
        self.voices[selected].pan = max(Float32(-1.0), min(Float32(1.0), pan))
        self.voices[selected].lpf_coeff = max(Float32(0.04), min(Float32(1.0), lpf_coeff))
        self.voices[selected].lp_l = 0.0
        self.voices[selected].lp_r = 0.0
        self.voices[selected].reverb_send = max(Float32(0.0), min(Float32(0.65), reverb_send))
        self.voices[selected].priority = priority
        self.voices[selected].order = self.next_order
        self.voices[selected].bank = bank
        self.voices[selected].bus = bus if bus == BUS_SPEECH else BUS_EFFECTS
        self.voices[selected].elevation = max(Float32(-1.0), min(Float32(1.0), elevation))
        self.voices[selected].hull_conducted = hull_conducted
        self.next_order += 1
        self.events_accepted += 1

    def play_spatial(
        mut self,
        sound_id: Int,
        pitch: Float32 = 1.0,
        volume: Float32 = 0.75,
        pan: Float32 = 0.0,
        lpf_coeff: Float32 = 1.0,
        reverb_send: Float32 = 0.14,
        priority: Int = 2,
        bank: Int = -1,
        limit: Int = 0,
        elevation: Float32 = 0.0,
        hull_conducted: Bool = False,
    ):
        self.play_voice(
            sound_id,
            pitch,
            volume,
            priority,
            bank,
            limit,
            BUS_EFFECTS,
            pan,
            lpf_coeff,
            reverb_send,
            elevation,
            hull_conducted,
        )

    def play_sound_3d(
        mut self,
        sound_id: Int,
        listener_pos: Vec3,
        listener_vel: Vec3,
        listener_right: Vec3,
        source_pos: Vec3,
        source_vel: Vec3 = Vec3(0.0, 0.0, 0.0),
        base_pitch: Float32 = 1.0,
        base_volume: Float32 = 0.85,
        ref_dist: Float32 = 24.0,
        max_dist: Float32 = 420.0,
        doppler_scale: Float32 = 0.65,
        priority: Int = 2,
        bank: Int = -1,
        limit: Int = 0,
        listener_up: Vec3 = Vec3(0.0, 1.0, 0.0),
        hull_conducted: Bool = False,
    ):
        """Plays a 3D spatialized sound with Binaural HRTF (ITD + pinna elevation), inverse-distance
        rolloff, radial-velocity Doppler pitch shift, HF air/vacuum absorption, and FDN reverb send."""
        var sp = compute_spatial_3d(
            listener_pos,
            listener_vel,
            listener_right,
            source_pos,
            source_vel,
            ref_dist,
            max_dist,
            doppler_scale,
        )
        var final_vol = base_volume * sp.gain
        if final_vol < Float32(0.012):
            return
        var rel = source_pos - listener_pos
        var dist = sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z)
        var up_dot = rel.x * listener_up.x + rel.y * listener_up.y + rel.z * listener_up.z
        var elev = max(Float32(-1.0), min(Float32(1.0), up_dot / max(Float32(0.5), dist)))
        self.play_voice(
            sound_id,
            base_pitch * sp.doppler_pitch,
            final_vol,
            priority,
            bank,
            limit,
            BUS_EFFECTS,
            sp.pan,
            sp.lpf_coeff,
            sp.reverb_send,
            elev,
            hull_conducted,
        )

    def mix_stereo(mut self, frames: Int) -> List[Int16]:
        var count = max(0, min(32768, frames))
        var mixed = List[Float32](length=count * 2, fill=0)
        var rev_send_buf = List[Float32](length=count * 2, fill=0)
        var has_rev_send = False
        var speech_active = self.is_speech_playing()
        var duck_goal = Float32(0.38) if speech_active else Float32(1.0)
        var duck_rate = Float32(0.0012) if speech_active else Float32(0.00018)
        if self.music_loaded:
            var position = self.music_position
            var step = Float64(self.music_pitch) * Float64(self.music.sample_rate) / Float64(self.output_rate)
            var music_frames = self.music.frame_count()
            var base_music_gain = self.music_volume * self.bus_volume[BUS_MUSIC]
            var duck = self.speech_duck
            for frame in range(count):
                duck += (duck_goal - duck) * duck_rate
                var music_gain = base_music_gain * duck
                for channel in range(2):
                    mixed[frame * 2 + channel] += clip_sample(self.music, position, channel, True) * music_gain
                position += step
                if position >= Float64(music_frames):
                    position -= Float64(music_frames)
            self.music_position = position
            self.speech_duck = duck
        else:
            for _ in range(count):
                self.speech_duck += (duck_goal - self.speech_duck) * duck_rate
        for voice_index in range(len(self.voices)):
            if not self.voices[voice_index].active:
                continue
            var clip_index = self.voices[voice_index].clip
            if clip_index < 0 or clip_index >= len(self.clips) or not self.clips[clip_index].valid:
                self.voices[voice_index].active = False
                continue
            var position = self.voices[voice_index].position
            var step = Float64(self.voices[voice_index].pitch) * Float64(self.clips[clip_index].sample_rate) / Float64(self.output_rate)
            var volume = self.voices[voice_index].volume * self.bus_volume[self.voices[voice_index].bus]
            var pan = self.voices[voice_index].pan
            var abs_pan = abs(pan)
            var elev = self.voices[voice_index].elevation
            var is_sfx = self.voices[voice_index].bus == BUS_EFFECTS
            var hull_cond = self.voices[voice_index].hull_conducted
            var vac = self.vacuum_mix if is_sfx else Float32(0.0)
            var vac_gain = Float32(1.0)
            var lpf = self.voices[voice_index].lpf_coeff
            if vac > Float32(0.005):
                if hull_cond:
                    # Internal airframe/cockpit vibration resonates cleanly with warm low-mid body
                    lpf = max(Float32(0.18), lpf * (Float32(1.0) - Float32(0.28) * vac))
                else:
                    # External space vacuum: air-borne HF vanishes, leaving deep shockwave sub-bass thump
                    lpf = max(Float32(0.032), lpf * (Float32(1.0) - Float32(0.88) * vac))
                    vac_gain = Float32(1.0) - Float32(0.56) * vac
            if elev < Float32(-0.05) and is_sfx:
                lpf = max(Float32(0.032), lpf * (Float32(1.0) + elev * Float32(0.24)))
            var ep = equal_power_gains(pan)
            var gain_l = ep[0] * Float32(1.41421356) * volume * vac_gain
            var gain_r = ep[1] * Float32(1.41421356) * volume * vac_gain
            var use_hrtf = self.hrtf_enabled and not self.mono and is_sfx and abs_pan > Float32(0.02)
            var itd_clip_frames = (
                Float64(abs_pan) * 0.00062 * Float64(self.voices[voice_index].pitch) * Float64(self.clips[clip_index].sample_rate)
                if use_hrtf
                else Float64(0.0)
            )
            var lpf_l = max(Float32(0.03), lpf * (Float32(1.0) - Float32(0.54) * pan)) if (use_hrtf and pan > Float32(0.0)) else lpf
            var lpf_r = max(Float32(0.03), lpf * (Float32(1.0) + Float32(0.54) * pan)) if (use_hrtf and pan < Float32(0.0)) else lpf
            var pinna_shelf = max(Float32(0.0), elev) * Float32(0.22) if (use_hrtf and vac < Float32(0.5)) else Float32(0.0)
            var lp_l = self.voices[voice_index].lp_l
            var lp_r = self.voices[voice_index].lp_r
            var rsend = (
                self.voices[voice_index].reverb_send * (Float32(1.0) - Float32(0.75) * vac)
                if is_sfx
                else Float32(0.03)
            )
            if rsend > Float32(0.005):
                has_rev_send = True
            var clip_ch = self.clips[clip_index].channels
            var clip_frames = self.clips[clip_index].frame_count()
            for frame in range(count):
                if position >= Float64(clip_frames):
                    break
                var pos_l = max(Float64(0.0), position - itd_clip_frames) if pan > Float32(0.02) else position
                var pos_r = max(Float64(0.0), position - itd_clip_frames) if pan < Float32(-0.02) else position
                var sl = clip_sample(self.clips[clip_index], pos_l, 0)
                var sr = clip_sample(self.clips[clip_index], pos_r, 0 if clip_ch == 1 else 1)
                # Fold opposite stereo energy slightly when panned hard so stereo SFX stay full
                if clip_ch == 2 and pan < Float32(-0.05):
                    sl += sr * (-pan) * Float32(0.35)
                elif clip_ch == 2 and pan > Float32(0.05):
                    sr += sl * pan * Float32(0.35)
                var raw_l = sl
                var raw_r = sr
                if lpf_l < Float32(0.995) or pinna_shelf > Float32(0.001):
                    lp_l += (sl - lp_l) * lpf_l
                    sl = lp_l + (raw_l - lp_l) * pinna_shelf
                if lpf_r < Float32(0.995) or pinna_shelf > Float32(0.001):
                    lp_r += (sr - lp_r) * lpf_r
                    sr = lp_r + (raw_r - lp_r) * pinna_shelf
                var out_l = sl * gain_l
                var out_r = sr * gain_r
                mixed[frame * 2] += out_l
                mixed[frame * 2 + 1] += out_r
                if rsend > Float32(0.005):
                    rev_send_buf[frame * 2] += out_l * rsend
                    rev_send_buf[frame * 2 + 1] += out_r * rsend
                position += step
            self.voices[voice_index].position = position
            self.voices[voice_index].lp_l = lp_l
            self.voices[voice_index].lp_r = lp_r
            if position >= Float64(clip_frames):
                self.voices[voice_index].active = False
        if self.reverb_enabled and has_rev_send:
            for frame in range(count):
                var wet = self.reverb.process_send(rev_send_buf[frame * 2], rev_send_buf[frame * 2 + 1])
                mixed[frame * 2] += wet[0]
                mixed[frame * 2 + 1] += wet[1]
        self.mix_drone(mixed, count)
        if self.mono:
            for frame in range(count):
                var both = (mixed[frame * 2] + mixed[frame * 2 + 1]) * 0.5
                mixed[frame * 2] = both
                mixed[frame * 2 + 1] = both
        var output = List[Int16](length=count * 2, fill=0)
        for index in range(len(output)):
            var x = mixed[index]
            # Transparent below 0.72, smooth Padé tanh studio soft-knee ceiling above 0.72
            var sample: Float32
            if x > Float32(0.72):
                sample = Float32(0.72) + Float32(0.28) * soft_clip_tanh((x - Float32(0.72)) * Float32(2.6))
            elif x < Float32(-0.72):
                sample = Float32(-0.72) + Float32(0.28) * soft_clip_tanh((x + Float32(0.72)) * Float32(2.6))
            else:
                sample = x
            output[index] = Int16(Int(max(Float32(-1.0), min(Float32(1.0), sample)) * Float32(32767.0)))
        return output^

    def submit_pending(mut self):
        if not self.ready or len(self.pending) == 0:
            return
        var frames = len(self.pending) // 2
        var transfer = List[UInt8](length=24, fill=0)
        audio_put64(transfer, 8, UInt64(self.pending.unsafe_ptr()))
        audio_put64(transfer, 16, UInt64(frames))
        var result = external_call["ioctl", Int32](Int32(self.fd), UInt64(0x40184150), transfer.unsafe_ptr())
        var written = audio_i64(transfer, 0)
        if result == 0 and written > 0:
            var consumed = min(frames, written)
            self.frames_submitted += UInt64(consumed)
            if consumed == frames:
                self.pending.clear()
            else:
                var remaining = List[Int16]()
                remaining.reserve((frames - consumed) * 2)
                for index in range(consumed * 2, len(self.pending)):
                    remaining.append(self.pending[index])
                self.pending = remaining^
            return
        var error = self.errno()
        if error == 11:  # EAGAIN: keep the exact mixed block for the next update.
            return
        if error == 32:  # EPIPE: recover from an underrun and resubmit later.
            self.xruns += 1
            _ = external_call["ioctl", Int32](Int32(self.fd), UInt64(0x4140), transfer.unsafe_ptr())
            return
        self.last_error = "PCM WRITEI_FRAMES errno " + String(error)
        self.ready = False

    def update_pulse(mut self):
        """Mix exactly what the sound server asked for, one period at a time."""
        self.pulse.pump()
        if not self.pulse.connected():
            self.last_error = self.pulse.error
            self.ready = False
            return
        var block_bytes = self.period_frames * 4
        while self.pulse.requested_bytes >= block_bytes:
            if len(self.pending) == 0:
                self.pending = self.mix_stereo(self.period_frames)
            if not self.pulse.write_samples(self.pending):
                return
            self.frames_submitted += UInt64(self.period_frames)
            self.pending.clear()

    def update(mut self, intensity: Float32):
        if not self.ready:
            return
        var amount = max(Float32(0), min(Float32(1), intensity))
        if self.music_pitch_follow:
            self.music_pitch = 0.96 + amount * 0.10
            self.music_volume = 0.20 + amount * 0.10
        if self.pulse.connected():
            self.update_pulse()
            return
        if len(self.pending) == 0:
            self.pending = self.mix_stereo(self.period_frames)
        self.submit_pending()
