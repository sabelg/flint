"""Sound engine for `flint.cycles`: turns pattern events into audio.

Built on Flint's own DSP (`flint.platform.dsp`): PolyBLEP oscillators, the
Cytomic SVF for `cutoff`/`hcutoff`/`resonance`, the FDN reverb for `room`,
modal resonators for bells/gongs/timpani, tube saturation for drums.

Sounds are looked up in this order:
  1. a sample folder `<root>/<s>/*.wav` (SuperDirt layout, `n` picks the file),
     with root from `$FLINT_CYCLES_SAMPLES`, else SuperCollider's
     `~/.local/share/SuperCollider/downloaded-quarks/Dirt-Samples`, else `./samples`;
  2. a built-in synthesized voice (drums: bd sn hh oh cp cb rim lt mt ht cr;
     synths: sine saw square tri supersaw bass acid pluck pad fm superpiano
     bell gong timpani noise — see `voice_kind`).
"""
from std.math import sin, exp, floor, sqrt
from flint.platform.dsp import (
    CytomicSVF, ModalResonator, StereoReverbFDN, hash_white_noise, poly_blep_saw, poly_blep_pulse,
    soft_clip_tanh, tube_saturate, equal_power_gains,
)
from flint.platform.audio import load_wav
from flint.platform.os import env_value, list_directory, file_stamp, write_bytes
from flint.cycles.pattern import (
    Frac, Params, Event, P_N, P_NOTE, P_GAIN, P_PAN, P_SPEED, P_CUTOFF, P_RESONANCE, P_HCUTOFF, P_ROOM, P_SIZE,
    P_SHAPE, P_CRUSH, P_LEGATO, P_ATTACK, P_RELEASE, P_DELAY, P_DELAYTIME, P_DELAYFB, P_COARSE, P_BEGIN, P_END,
    P_SUSTAIN,
)
from flint.cycles.script import Program, run_program

comptime TAU: Float32 = 6.283185307179586

comptime VK_NONE: Int = -1
comptime VK_SAMPLE: Int = 0
comptime VK_BD: Int = 1
comptime VK_SN: Int = 2
comptime VK_HH: Int = 3
comptime VK_OH: Int = 4
comptime VK_CP: Int = 5
comptime VK_CB: Int = 6
comptime VK_TOM: Int = 7
comptime VK_RIM: Int = 8
comptime VK_CRASH: Int = 9
comptime VK_BELL: Int = 10
comptime VK_TIMPANI: Int = 11
comptime VK_SINE: Int = 20
comptime VK_SAW: Int = 21
comptime VK_SQUARE: Int = 22
comptime VK_TRI: Int = 23
comptime VK_SUPERSAW: Int = 24
comptime VK_BASS: Int = 25
comptime VK_PLUCK: Int = 26
comptime VK_PAD: Int = 27
comptime VK_FM: Int = 28
comptime VK_NOISE: Int = 29


def voice_kind(name: String) -> Int:
    """Built-in voice for a sound name (SuperDirt-style aliases)."""
    if name == "bd" or name == "kick" or name == "808bd" or name == "drum":
        return VK_BD
    if name == "sn" or name == "snare" or name == "sd" or name == "808sd":
        return VK_SN
    if name == "hh" or name == "hc" or name == "hat" or name == "808hc" or name == "ch":
        return VK_HH
    if name == "oh" or name == "ho" or name == "808oh":
        return VK_OH
    if name == "cp" or name == "clap" or name == "realclaps":
        return VK_CP
    if name == "cb" or name == "cowbell":
        return VK_CB
    if name == "lt" or name == "mt" or name == "ht" or name == "tom":
        return VK_TOM
    if name == "rim" or name == "rs":
        return VK_RIM
    if name == "cr" or name == "crash" or name == "cy" or name == "ride":
        return VK_CRASH
    if name == "bell" or name == "gong" or name == "supergong" or name == "glock":
        return VK_BELL
    if name == "timpani" or name == "kettle":
        return VK_TIMPANI
    if name == "sine" or name == "supersine" or name == "beep":
        return VK_SINE
    if name == "saw" or name == "supersawsingle":
        return VK_SAW
    if name == "square" or name == "supersquare" or name == "superchip" or name == "pulse":
        return VK_SQUARE
    if name == "tri" or name == "triangle":
        return VK_TRI
    if name == "supersaw" or name == "lead":
        return VK_SUPERSAW
    if name == "bass" or name == "superbass" or name == "acid" or name == "tb303":
        return VK_BASS
    if name == "pluck" or name == "arpy" or name == "superpluck" or name == "karplus":
        return VK_PLUCK
    if name == "pad" or name == "superpad" or name == "strings":
        return VK_PAD
    if name == "fm" or name == "superfm" or name == "superpiano" or name == "rhodes" or name == "epiano":
        return VK_FM
    if name == "noise" or name == "white":
        return VK_NOISE
    return VK_NONE


# ---------------------------------------------------------------- samples


struct Clip(Copyable, Movable):
    var channels: Int
    var rate: Int
    var data: List[Int16]

    def __init__(out self, channels: Int, rate: Int, var data: List[Int16]):
        self.channels = channels
        self.rate = rate
        self.data = data^

    def frames(self) -> Int:
        return len(self.data) // max(1, self.channels)

    @always_inline
    def at(self, pos: Float64, ch: Int) -> Float32:
        var n = self.frames()
        if n == 0 or pos < 0 or pos >= Float64(n - 1):
            return 0.0
        var i = Int(pos)
        var c = min(ch, self.channels - 1)
        var a = Float32(self.data[i * self.channels + c]) / 32768.0
        var b = Float32(self.data[(i + 1) * self.channels + c]) / 32768.0
        return a + (b - a) * Float32(pos - Float64(i))


@fieldwise_init
struct Folder(Copyable, Movable):
    var name: String
    var files: List[String]
    var clip_ids: List[Int]  # -1 until loaded


struct SampleBank(Movable):
    var root: String
    var folders: List[Folder]
    var clips: List[Clip]

    def __init__(out self):
        self.root = SampleBank.find_root()
        self.folders = List[Folder]()
        self.clips = List[Clip]()

    @staticmethod
    def find_root() -> String:
        var env = env_value("FLINT_CYCLES_SAMPLES")
        if env.byte_length() > 0:
            return env
        var home = env_value("HOME")
        var dirt = home + "/.local/share/SuperCollider/downloaded-quarks/Dirt-Samples"
        if len(list_directory(dirt)) > 0:
            return dirt
        return "samples"

    def folder(mut self, name: String) -> Int:
        for i in range(len(self.folders)):
            if self.folders[i].name == name:
                return i
        var files = List[String]()
        var ids = List[Int]()
        var path = self.root + "/" + name
        for f in list_directory(path):
            if f.endswith(".wav") or f.endswith(".WAV"):
                files.append(path + "/" + f)
                ids.append(-1)
        self.folders.append(Folder(name, files^, ids^))
        return len(self.folders) - 1

    def clip(mut self, name: String, n: Int) -> Int:
        """Clip id for sound `name`, file `n` (wrapping); -1 if none."""
        var fi = self.folder(name)
        var count = len(self.folders[fi].files)
        if count == 0:
            return -1
        var idx = n % count
        if idx < 0:
            idx += count
        if self.folders[fi].clip_ids[idx] >= 0:
            return self.folders[fi].clip_ids[idx]
        var wav = load_wav(self.folders[fi].files[idx])
        if not wav.valid or wav.frame_count() == 0:
            return -1
        self.clips.append(Clip(wav.channels, wav.sample_rate, wav.samples.copy()))
        var id = len(self.clips) - 1
        self.folders[fi].clip_ids[idx] = id
        return id


# ---------------------------------------------------------------- voices


struct Voice(Copyable, Movable):
    var kind: Int
    var wait: Int          # frames until it starts (onset inside a block)
    var t: Int             # frames since start
    var gate: Int          # frames held before release
    var release: Int
    var life: Int          # frames until it is freed
    var freq: Float32
    var amp: Float32
    var gain_l: Float32
    var gain_r: Float32
    var attack: Float32    # seconds
    var room: Float32
    var delay_send: Float32
    var shape: Float32
    var crush: Float32
    var coarse: Int
    var held: Float32
    var use_lp: Bool
    var use_hp: Bool
    var lp: CytomicSVF
    var hp: CytomicSVF
    var f1: CytomicSVF     # voice-internal filter (drum colour, bass sweep)
    var cutoff: Float32
    var res_q: Float32
    var phases: List[Float32]
    var detune: List[Float32]
    var modes: List[ModalResonator]
    var ks: List[Float32]
    var ks_pos: Int
    var noise_seed: Int
    var clip: Int
    var pos: Float64
    var step: Float64
    var end_pos: Float64

    def __init__(out self, kind: Int):
        self.kind = kind
        self.wait = 0
        self.t = 0
        self.gate = 0
        self.release = 0
        self.life = 0
        self.freq = 261.6256
        self.amp = 0.5
        self.gain_l = 0.7071
        self.gain_r = 0.7071
        self.attack = 0.002
        self.room = 0.0
        self.delay_send = 0.0
        self.shape = 0.0
        self.crush = 0.0
        self.coarse = 0
        self.held = 0.0
        self.use_lp = False
        self.use_hp = False
        self.lp = CytomicSVF()
        self.hp = CytomicSVF()
        self.f1 = CytomicSVF()
        self.cutoff = 0.0
        self.res_q = 0.707
        self.phases = List[Float32]()
        self.detune = List[Float32]()
        self.modes = List[ModalResonator]()
        self.ks = List[Float32]()
        self.ks_pos = 0
        self.noise_seed = 0
        self.clip = -1
        self.pos = 0.0
        self.step = 1.0
        self.end_pos = 0.0


struct CyclesEngine(Movable):
    """Clock + scheduler + voice mixer. Feed it Tidal code with `load`, pull
    stereo audio with `render`."""

    var sample_rate: Int
    var sr: Float32
    var program: Program
    var cps: Float64
    var base_cycle: Frac       # cycle position at the last tempo change
    var frames_since: Int      # frames rendered since then
    var cps_frac: Frac
    var voices: List[Voice]
    var bank: SampleBank
    var reverb: StereoReverbFDN
    var delay_l: List[Float32]
    var delay_r: List[Float32]
    var delay_idx: Int
    var delay_time: Float32
    var delay_fb: Float32
    var master: Float32
    var missing: List[String]
    var events_played: Int
    var max_voices: Int

    def __init__(out self, sample_rate: Int = 48000):
        var sr_safe = 48000 if sample_rate <= 0 else min(sample_rate, 384000)
        self.sample_rate = sr_safe
        self.sr = Float32(sr_safe)
        self.program = Program()
        self.cps = 0.5625
        self.cps_frac = Frac.from_float(0.5625)
        self.base_cycle = Frac(0, 1)
        self.frames_since = 0
        self.voices = List[Voice]()
        self.bank = SampleBank()
        self.reverb = StereoReverbFDN(0.82, 0.32)
        self.delay_l = List[Float32](length=sr_safe * 2, fill=0.0)
        self.delay_r = List[Float32](length=sr_safe * 2, fill=0.0)
        self.delay_idx = 0
        self.delay_time = 0.375
        self.delay_fb = 0.45
        self.master = 0.8
        self.missing = List[String]()
        self.events_played = 0
        self.max_voices = 96

    # ---- program & clock

    def cycle_at(self, frames: Int) -> Frac:
        return self.base_cycle + Frac.make(frames * self.cps_frac.num, self.sample_rate * self.cps_frac.den)

    def now(self) -> Frac:
        return self.cycle_at(self.frames_since)

    def set_cps(mut self, cps: Float64):
        if not (cps > 0.0 and cps < 1.0e6):
            return
        var next_frac = Frac.from_float(cps)
        if next_frac.num <= 0 or next_frac.den <= 0:
            return
        # Re-anchor the clock so the cycle position is continuous.
        var here = self.now()
        self.base_cycle = Frac.make128(Int128(here.num) * Int128(1 << 20) // Int128(here.den), Int128(1 << 20))
        self.frames_since = 0
        self.cps = cps
        self.cps_frac = next_frac

    def load_program(mut self, var prog: Program) -> Bool:
        """Install an already-parsed `Program` (for RT-safe off-thread parsing)."""
        if len(prog.errors) > 0:
            return False
        if prog.cps_set:
            self.set_cps(prog.cps)
        self.program = prog^
        return True

    def load(mut self, text: String) -> List[String]:
        """Evaluate a whole Tidal file. On any error the previous program keeps
        playing (keep-last-good); the errors are returned either way."""
        var prog = run_program(text)
        var errs = prog.errors.copy()
        if len(errs) > 0:
            return errs^
        _ = self.load_program(prog^)
        return errs^

    # ---- scheduling

    def spawn(mut self, ev: Event, offset: Int):
        ref p = ev.value
        if p.raw:
            return
        var name = p.sound if p.has_sound else String("sine")  # `n "0 3"` alone plays a sine
        var n_val = p.get(P_N, 0.0)
        var speed = Float32(p.get(P_SPEED, 1.0))
        var dur_sec = Float32((ev.whole_e - ev.whole_b).to_float() / self.cps)
        var kind: Int
        var clip = self.bank.clip(name, Int(floor(n_val)))
        if clip >= 0:
            kind = VK_SAMPLE
        else:
            kind = voice_kind(name)
        if kind == VK_NONE:
            for m in self.missing:
                if m == name:
                    return
            self.missing.append(name)
            print("[cycles] no sound '" + name + "' (no sample folder in " + self.bank.root + " and no built-in voice)")
            return
        if len(self.voices) >= self.max_voices:
            # steal the oldest
            var oldest = 0
            for i in range(len(self.voices)):
                if self.voices[i].t > self.voices[oldest].t:
                    oldest = i
            _ = self.voices.pop(oldest)
        var v = Voice(kind)
        v.wait = offset
        v.noise_seed = Int(ev.whole_b.num * 7919 + ev.whole_b.den * 104729 + self.events_played * 31)
        var gain_raw = p.get(P_GAIN, 1.0)
        var gain = Float32(gain_raw) if gain_raw == gain_raw else Float32(1.0)
        gain = Float32(16.0) if gain > Float32(16.0) else max(Float32(0.0), gain)
        v.amp = gain * gain * 0.42
        var pan_raw = p.get(P_PAN, 0.5)
        var pan = Float32(pan_raw) if pan_raw == pan_raw else Float32(0.5)
        pan = Float32(1.0) if pan > Float32(1.0) else max(Float32(0.0), pan)
        var g = equal_power_gains(pan * 2.0 - 1.0)
        v.gain_l = g[0]
        v.gain_r = g[1]
        var room_raw = p.get(P_ROOM, 0.0)
        var room_f = Float32(room_raw) if room_raw == room_raw else Float32(0.0)
        v.room = Float32(4.0) if room_f > Float32(4.0) else max(Float32(0.0), room_f)
        var del_raw = p.get(P_DELAY, 0.0)
        var del_f = Float32(del_raw) if del_raw == del_raw else Float32(0.0)
        v.delay_send = Float32(4.0) if del_f > Float32(4.0) else max(Float32(0.0), del_f)
        var shape_raw = p.get(P_SHAPE, 0.0)
        var shape_f = shape_raw if shape_raw == shape_raw else 0.0
        v.shape = Float32(max(0.0, min(0.99, shape_f)))
        var crush_raw = p.get(P_CRUSH, 0.0)
        v.crush = Float32(crush_raw) if (crush_raw == crush_raw and crush_raw > 0.0) else Float32(0.0)
        if v.crush > Float32(32.0):
            v.crush = Float32(32.0)
        v.coarse = Int(p.get(P_COARSE, 0.0))
        if p.has(P_SIZE):
            self.reverb.feedback = Float32(0.55 + 0.38 * max(0.0, min(1.0, p.vals[P_SIZE])))
        if p.has(P_DELAYTIME):
            self.delay_time = Float32(max(0.001, min(1.99, p.vals[P_DELAYTIME])))
        if p.has(P_DELAYFB):
            self.delay_fb = Float32(max(0.0, min(0.95, p.vals[P_DELAYFB])))
        var q = Float32(0.707 + max(0.0, min(1.0, p.get(P_RESONANCE, 0.0))) * 14.0)
        v.res_q = q
        if p.has(P_CUTOFF):
            v.use_lp = True
            v.cutoff = Float32(p.vals[P_CUTOFF])
            v.lp.set_params(v.cutoff, q, self.sr)
        if p.has(P_HCUTOFF):
            v.use_hp = True
            v.hp.set_params(Float32(p.vals[P_HCUTOFF]), q, self.sr)
        # Pitch: note 0 = C5 (261.63 Hz); for synths `n` is a note too.
        var note = p.get(P_NOTE, 0.0)
        if kind >= VK_SINE or kind == VK_BELL or kind == VK_TIMPANI:
            note += n_val
        if note != note:
            note = 0.0
        note = max(-120.0, min(120.0, note))
        var ratio = Float32(2.0 ** (note / 12.0))
        v.freq = Float32(261.6256) * ratio
        var sr = self.sr
        var legato = Float32(p.get(P_LEGATO, 1.0))
        var gate_sec = dur_sec * legato
        if p.has(P_SUSTAIN):
            gate_sec = Float32(p.vals[P_SUSTAIN])
        v.attack = Float32(p.get(P_ATTACK, 0.002))
        var rel_sec = Float32(p.get(P_RELEASE, 0.06))
        if kind == VK_SAMPLE:
            ref c = self.bank.clips[clip]
            v.clip = clip
            var frames = Float64(c.frames())
            var b = max(0.0, min(1.0, p.get(P_BEGIN, 0.0)))
            var e = max(0.0, min(1.0, p.get(P_END, 1.0)))
            var rate = Float64(speed) * Float64(ratio) * Float64(c.rate) / Float64(self.sample_rate)
            v.step = rate
            if rate >= 0:
                v.pos = b * frames
                v.end_pos = e * frames
            else:
                v.pos = e * frames - 1.0
                v.end_pos = b * frames
            var span = (v.end_pos - v.pos) / rate if rate != 0 else 0.0
            var life = Int(span)
            if p.has(P_LEGATO) or p.has(P_SUSTAIN):
                life = min(life, Int(gate_sec * sr) + Int(0.01 * sr))
            v.gate = max(1, life - Int(0.005 * sr))
            v.release = Int(0.005 * sr)
            v.life = life
        elif kind < VK_SINE:
            self._init_drum(v, kind, Float32(n_val), speed * ratio)
        else:
            if kind == VK_PAD:
                v.attack = Float32(p.get(P_ATTACK, 0.25))
                rel_sec = Float32(p.get(P_RELEASE, 0.6))
            elif kind == VK_PLUCK:
                rel_sec = Float32(p.get(P_RELEASE, 0.25))
                gate_sec = max(gate_sec, Float32(1.2))
            elif kind == VK_FM:
                rel_sec = Float32(p.get(P_RELEASE, 0.4))
            v.gate = max(1, Int(gate_sec * sr))
            v.release = max(1, Int(rel_sec * sr))
            v.life = v.gate + v.release
            self._init_synth(v, kind)
        self.voices.append(v^)
        self.events_played += 1

    def _init_drum(mut self, mut v: Voice, kind: Int, n: Float32, pitch: Float32):
        var sr = self.sr
        var tune = pitch * (Float32(1.0) - Float32(0.05) * min(n, Float32(8)))
        v.freq = tune
        v.phases = [Float32(0), Float32(0)]
        var life_sec = Float32(0.5)
        if kind == VK_BD:
            life_sec = 0.7
        elif kind == VK_SN:
            life_sec = 0.35
            v.f1.set_params(Float32(1800) * tune, 0.9, sr)
        elif kind == VK_HH:
            life_sec = 0.12
            v.f1.set_params(Float32(7500) * tune, 0.8, sr)
        elif kind == VK_OH:
            life_sec = 0.7
            v.f1.set_params(Float32(7000) * tune, 0.8, sr)
        elif kind == VK_CP:
            life_sec = 0.35
            v.f1.set_params(Float32(1250) * tune, 2.2, sr)
        elif kind == VK_CB:
            life_sec = 0.4
            v.f1.set_params(Float32(1000) * tune, 1.6, sr)
        elif kind == VK_TOM:
            life_sec = 0.6
        elif kind == VK_RIM:
            life_sec = 0.1
            v.modes.append(ModalResonator(Float32(1700) * tune, 0.025, 1.0, sr))
            v.modes.append(ModalResonator(Float32(470) * tune, 0.04, 0.5, sr))
        elif kind == VK_CRASH:
            life_sec = 2.2
            v.f1.set_params(Float32(5200) * tune, 0.7, sr)
        elif kind == VK_BELL:
            life_sec = 3.5
            var ratios: List[Float32] = [1.0, 2.76, 5.40, 8.93, 1.583, 2.241]
            var gains: List[Float32] = [1.0, 0.55, 0.32, 0.18, 0.4, 0.3]
            for i in range(6):
                v.modes.append(ModalResonator(v.freq * ratios[i], Float32(1.6) / (Float32(1) + Float32(i) * 0.6), gains[i], sr))
        elif kind == VK_TIMPANI:
            life_sec = 2.0
            var base = v.freq * Float32(0.28)  # note 0 -> ~73 Hz (D2)
            var ratios: List[Float32] = [1.000, 1.504, 1.742, 2.000, 2.295]
            var gains: List[Float32] = [1.15, 0.72, 0.46, 0.38, 0.24]
            var decays: List[Float32] = [1.00, 0.86, 0.64, 0.72, 0.48]
            for i in range(5):
                v.modes.append(ModalResonator(base * ratios[i], Float32(0.62) * decays[i], gains[i], sr))
        v.life = Int(life_sec * sr)
        v.gate = v.life
        v.release = 1

    def _init_synth(mut self, mut v: Voice, kind: Int):
        var sr = self.sr
        if kind == VK_SUPERSAW:
            var spread: List[Float32] = [-0.30, -0.19, -0.09, 0.0, 0.09, 0.19, 0.30]
            for i in range(7):
                v.phases.append(Float32((hash_white_noise(v.noise_seed + i) + 1.0) * 0.5))
                v.detune.append(Float32(2.0) ** (spread[i] / Float32(12.0)))
        elif kind == VK_PAD:
            var spread: List[Float32] = [-0.12, 0.0, 0.11]
            for i in range(3):
                v.phases.append(Float32((hash_white_noise(v.noise_seed + i) + 1.0) * 0.5))
                v.detune.append(Float32(2.0) ** (spread[i] / Float32(12.0)))
            v.f1.set_params(1800, 0.8, sr)
        elif kind == VK_PLUCK:
            var length = max(2, Int(sr / max(Float32(20), v.freq)))
            for i in range(length):
                v.ks.append(hash_white_noise(v.noise_seed + i) * 0.8)
        else:
            v.phases = [Float32(0), Float32(0)]
        if kind == VK_BASS:
            v.f1.set_params(800, 5.0, sr)

    # ---- synthesis

    @always_inline
    def _env(self, v: Voice) -> Float32:
        var t = v.t
        var a = max(Float32(1), v.attack * self.sr)
        var level = Float32(1.0)
        if Float32(t) < a:
            level = Float32(t) / a
        if t >= v.gate:
            var r = Float32(t - v.gate) / Float32(max(1, v.release))
            level *= max(Float32(0), Float32(1) - r) * max(Float32(0), Float32(1) - r)
        return level

    def _sample(mut self, mut v: Voice) -> Float32:
        var sr = self.sr
        var ts = Float32(v.t) / sr
        var k = v.kind
        var x = Float32(0)
        var noise = hash_white_noise(v.noise_seed + v.t * 3 + 17)
        if k == VK_SAMPLE:
            ref c = self.bank.clips[v.clip]
            x = c.at(v.pos, 0)  # mono sum keeps pan meaningful
            if c.channels > 1:
                x = (x + c.at(v.pos, 1)) * 0.5
            v.pos += v.step
            x *= self._env(v) * 1.6
        elif k == VK_BD:
            var f = v.freq * (Float32(48) + Float32(120) * exp(-ts * 32))
            v.phases[0] += f / sr
            v.phases[0] -= floor(v.phases[0])
            x = sin(v.phases[0] * TAU) * exp(-ts * 6.5) + noise * exp(-ts * 380) * 0.35
            x = tube_saturate(x * 1.3, 1.4) * 1.6
        elif k == VK_SN:
            var f = v.freq * Float32(185) * (Float32(1) + Float32(0.4) * exp(-ts * 60))
            v.phases[0] += f / sr
            v.phases[0] -= floor(v.phases[0])
            var tone = sin(v.phases[0] * TAU) * exp(-ts * 22) * 0.55
            _ = v.f1.process(noise)
            var rattle = (v.f1.hp + v.f1.bp * 0.6) * exp(-ts * 13) * 0.9
            x = tube_saturate(tone + rattle, 1.2)
        elif k == VK_HH or k == VK_OH or k == VK_CRASH:
            # metallic: six square partials (808 ratios) plus noise, high-passed
            var metal = Float32(0)
            var hz: Array[Float32, 6] = [
                Float32(205.3),
                Float32(304.4),
                Float32(369.6),
                Float32(522.7),
                Float32(540.0),
                Float32(800.0),
            ]
            for i in range(6):
                var ph = (Float32(v.t) * hz[i] * v.freq * 2.0 / sr)
                metal += Float32(1.0) if ph - floor(ph) < 0.5 else Float32(-1.0)
            _ = v.f1.process(metal * 0.12 + noise * 0.5)
            var decay = Float32(48) if k == VK_HH else (Float32(6) if k == VK_OH else Float32(2.4))
            x = v.f1.hp * exp(-ts * decay) * 1.4
        elif k == VK_CP:
            var env = Float32(0)
            for i in range(3):
                var tt = ts - Float32(i) * 0.011
                if tt >= 0:
                    env += exp(-tt * 190)
            if ts > 0.022:
                env += exp(-(ts - 0.022) * 16) * 0.55
            _ = v.f1.process(noise)
            x = v.f1.bp * env * 1.6
        elif k == VK_CB:
            var a = poly_blep_pulse(wrap01(Float32(v.t) * 540 * v.freq / sr), 540 * v.freq / sr)
            var b = poly_blep_pulse(wrap01(Float32(v.t) * 800 * v.freq / sr), 800 * v.freq / sr)
            _ = v.f1.process((a + b) * 0.5)
            x = v.f1.bp * exp(-ts * 9) * 1.4
        elif k == VK_TOM:
            var f = v.freq * Float32(110) * (Float32(1) + Float32(0.5) * exp(-ts * 18))
            v.phases[0] += f / sr
            v.phases[0] -= floor(v.phases[0])
            x = sin(v.phases[0] * TAU) * exp(-ts * 7) + noise * exp(-ts * 200) * 0.2
        elif k == VK_RIM or k == VK_BELL or k == VK_TIMPANI:
            var exc = Float32(0)
            if v.t < 3:
                exc = Float32(1.0)
            if k == VK_TIMPANI and v.t < 48:
                exc = noise * Float32(0.25) * (Float32(1) - Float32(v.t) / 48)
            for i in range(len(v.modes)):
                x += v.modes[i].step(exc)
            if k == VK_BELL:
                x *= Float32(2.2)
            elif k == VK_TIMPANI:
                x *= Float32(4.0)
            else:
                x *= Float32(5.0)
        elif k == VK_SINE:
            v.phases[0] += v.freq / sr
            v.phases[0] -= floor(v.phases[0])
            x = sin(v.phases[0] * TAU) * self._env(v)
        elif k == VK_SAW or k == VK_BASS:
            var dt = v.freq / sr
            v.phases[0] = wrap01(v.phases[0] + dt)
            x = poly_blep_saw(v.phases[0], dt)
            if k == VK_BASS:
                var sweep = Float32(180) + Float32(2600) * exp(-ts * 11)
                if v.use_lp:
                    sweep = v.cutoff * (Float32(0.35) + Float32(1.3) * exp(-ts * 11))
                if v.t % 16 == 0:
                    v.f1.set_params(sweep, max(v.res_q, Float32(4.0)), sr)
                x = v.f1.process(x) * 1.2
            x *= self._env(v) * 0.7
        elif k == VK_SQUARE:
            var dt = v.freq / sr
            v.phases[0] = wrap01(v.phases[0] + dt)
            x = poly_blep_pulse(v.phases[0], dt, 0.5) * self._env(v) * 0.55
        elif k == VK_TRI:
            v.phases[0] = wrap01(v.phases[0] + v.freq / sr)
            x = (Float32(1) - Float32(4) * abs(v.phases[0] - 0.5)) * self._env(v)
        elif k == VK_SUPERSAW or k == VK_PAD:
            for i in range(len(v.phases)):
                var dt = v.freq * v.detune[i] / sr
                v.phases[i] = wrap01(v.phases[i] + dt)
                x += poly_blep_saw(v.phases[i], dt)
            x *= Float32(1.0) / sqrt(Float32(len(v.phases)))
            if k == VK_PAD:
                x = v.f1.process(x)
            x *= self._env(v) * 0.55
        elif k == VK_PLUCK:
            var n = len(v.ks)
            var i = v.ks_pos
            var j = (i + 1) % n
            x = v.ks[i]
            v.ks[i] = (v.ks[i] + v.ks[j]) * Float32(0.4985)
            v.ks_pos = j
            x *= self._env(v) * 0.9
        elif k == VK_FM:
            v.phases[0] = wrap01(v.phases[0] + v.freq / sr)
            v.phases[1] = wrap01(v.phases[1] + v.freq * 1.0 / sr)
            var index = Float32(0.6) + Float32(2.6) * exp(-ts * 5)
            var mod = sin(v.phases[1] * TAU) * index
            x = sin(v.phases[0] * TAU + mod) * exp(-ts * 1.2) * self._env(v) * 0.7
        elif k == VK_NOISE:
            x = noise * self._env(v) * 0.4
        # per-event effects (SuperDirt order: shape, crush, coarse, filters)
        if v.shape > 0:
            var drive = Float32(1) + v.shape * 10
            x = soft_clip_tanh(x * drive) / soft_clip_tanh(drive) * 0.9
        if v.crush > 0 and v.crush == v.crush:
            var crush_c = Float32(32.0) if v.crush > Float32(32.0) else max(Float32(1.0), v.crush)
            var steps = Float32(2.0) ** (crush_c - Float32(1.0))
            x = floor(x * steps + Float32(0.5)) / steps
        if v.coarse > 1:
            if v.t % v.coarse == 0:
                v.held = x
            x = v.held
        if v.use_lp and k != VK_BASS:
            x = v.lp.process(x)
        if v.use_hp:
            _ = v.hp.process(x)
            x = v.hp.hp
        var out_s = x * v.amp
        return out_s if out_s == out_s else Float32(0.0)

    def render(mut self, frames: Int) -> List[Float32]:
        """Schedule the events starting in the next `frames` and mix them.
        Returns interleaved stereo floats."""
        if frames <= 0:
            return List[Float32]()
        var begin = self.now()
        var end = self.cycle_at(self.frames_since + frames)
        var span = (end - begin).to_float()
        var roots = List[Int]()
        for t in self.program.tracks:
            roots.append(t.root)
        if span > 0.0:
            for root in roots:
                var evs = self.program.pat.query_node(root, begin, end)
                for i in range(len(evs)):
                    if not evs[i].has_onset():
                        continue
                    var off = Int((evs[i].whole_b - begin).to_float() / span * Float64(frames))
                    self.spawn(evs[i], max(0, min(frames - 1, off)))
        var out = List[Float32](length=frames * 2, fill=0.0)
        var send_l = List[Float32](length=frames, fill=0.0)
        var send_r = List[Float32](length=frames, fill=0.0)
        var dsend_l = List[Float32](length=frames, fill=0.0)
        var dsend_r = List[Float32](length=frames, fill=0.0)
        var vi = 0
        while vi < len(self.voices):
            var v = self.voices[vi].copy()
            for f in range(v.wait, frames):
                if v.t >= v.life:
                    break
                var x = self._sample(v)
                v.t += 1
                var l = x * v.gain_l
                var r = x * v.gain_r
                out[f * 2] += l
                out[f * 2 + 1] += r
                if v.room > 0:
                    send_l[f] += l * v.room
                    send_r[f] += r * v.room
                if v.delay_send > 0:
                    dsend_l[f] += l * v.delay_send
                    dsend_r[f] += r * v.delay_send
            v.wait = 0
            if v.t >= v.life:
                _ = self.voices.pop(vi)
            else:
                self.voices[vi] = v^
                vi += 1
        var dlen = len(self.delay_l)
        if dlen <= 0:
            return out^
        var dframes = max(1, min(dlen - 1, Int(self.delay_time * self.sr)))
        for f in range(frames):
            var wet = self.reverb.process_send(send_l[f], send_r[f])
            var read = (self.delay_idx - dframes + dlen) % dlen
            var dl = self.delay_l[read]
            var dr = self.delay_r[read]
            self.delay_l[self.delay_idx] = dsend_l[f] + dr * self.delay_fb  # ping-pong
            self.delay_r[self.delay_idx] = dsend_r[f] + dl * self.delay_fb
            self.delay_idx = (self.delay_idx + 1) % dlen
            out[f * 2] = soft_clip_tanh((out[f * 2] + wet[0] * 1.4 + dl) * self.master)
            out[f * 2 + 1] = soft_clip_tanh((out[f * 2 + 1] + wet[1] * 1.4 + dr) * self.master)
        self.frames_since += frames
        return out^

    def render_i16(mut self, frames: Int) -> List[Int16]:
        if frames <= 0:
            return List[Int16]()
        var f = self.render(frames)
        var out = List[Int16](capacity=len(f))
        for i in range(len(f)):
            out.append(Int16(Int(max(Float32(-1), min(Float32(1), f[i])) * 32767.0)))
        return out^


@always_inline
def wrap01(x: Float32) -> Float32:
    return x - floor(x)


def _put16(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))


def _put32(mut b: List[UInt8], v: Int):
    for i in range(4):
        b.append(UInt8((v >> (8 * i)) & 0xFF))


def encode_wav16(samples: List[Int16], sample_rate: Int, channels: Int = 2) -> List[UInt8]:
    var b = List[UInt8](capacity=44 + len(samples) * 2)
    for c in "RIFF".as_bytes():
        b.append(c)
    _put32(b, 36 + len(samples) * 2)
    for c in "WAVEfmt ".as_bytes():
        b.append(c)
    _put32(b, 16)
    _put16(b, 1)
    _put16(b, channels)
    _put32(b, sample_rate)
    _put32(b, sample_rate * channels * 2)
    _put16(b, channels * 2)
    _put16(b, 16)
    for c in "data".as_bytes():
        b.append(c)
    _put32(b, len(samples) * 2)
    for s in samples:
        _put16(b, Int(s) & 0xFFFF)
    return b^


def render_to_wav(code: String, seconds: Float64, path: String, sample_rate: Int = 48000) raises -> Int:
    """Offline render of a Tidal program; returns events played."""
    if not (seconds > 0.0) or sample_rate <= 0:
        return 0
    var engine = CyclesEngine(sample_rate)
    var errs = engine.load(code)
    if len(errs) > 0:
        var msg = String("program has errors:")
        for e in errs:
            msg += "\n  " + e
        raise Error(msg)
    var total = Int(seconds * Float64(sample_rate))
    if total <= 0:
        return 0
    var pcm = List[Int16](capacity=total * 2)
    var done = 0
    while done < total:
        var n = min(512, total - done)
        var block = engine.render_i16(n)
        for s in block:
            pcm.append(s)
        done += n
    var bytes = encode_wav16(pcm, sample_rate)
    if not write_bytes(path, bytes):
        raise Error("could not write " + path)
    return engine.events_played


def render_loop_to_wav(code: String, cycles: Int, path: String, sample_rate: Int = 48000) raises -> Int:
    """Render exactly `cycles` cycles as a seamless loop: one warm-up pass is
    rendered and thrown away, so the reverb/delay tails that wrap around the
    loop point are already in the captured pass. Returns the frame count."""
    if cycles <= 0 or sample_rate <= 0:
        return 0
    var engine = CyclesEngine(sample_rate)
    var errs = engine.load(code)
    if len(errs) > 0:
        var msg = String("program has errors:")
        for e in errs:
            msg += "\n  " + e
        raise Error(msg)
    var start = Frac.of(cycles)
    var stop = Frac.of(cycles * 2)
    # warm-up: render up to the frame where cycle `cycles` begins
    var cps = engine.cps_frac
    if cps.num <= 0 or cps.den <= 0:
        return 0
    var warm = Int((start / cps).to_float() * Float64(sample_rate) + 0.5)
    var loop_frames = Int(((stop - start) / cps).to_float() * Float64(sample_rate) + 0.5)
    if warm < 0 or loop_frames <= 0:
        return 0
    var done = 0
    while done < warm:
        var n = min(512, warm - done)
        _ = engine.render(n)
        done += n
    var pcm = List[Int16](capacity=loop_frames * 2)
    done = 0
    while done < loop_frames:
        var n = min(512, loop_frames - done)
        var block = engine.render_i16(n)
        for s in block:
            pcm.append(s)
        done += n
    var bytes = encode_wav16(pcm, sample_rate)
    if not write_bytes(path, bytes):
        raise Error("could not write " + path)
    return loop_frames
