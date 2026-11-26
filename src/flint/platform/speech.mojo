"""Engine-Generic Synchronous Speech & Giant-Robot AI OS DSP Synthesizer (`flint.platform.speech`).

Synthesizes intelligible spoken tactical guidance into 48 kHz Stereo `WavClip`s:
  1. Renders dry 24 kHz speech with one of two backends:
     - **Kokoro** (default when its weights load): `flint.ml.kokoro`, the 82M-parameter neural
       TTS, with one voice pack / speed / pitch per `VOICE_*` profile (`kokoro_voice_preset`).
       Moonshine word error ~2 % dry (Klatt: ~24 %). It runs at ~0.3-0.5 RTF, so finished clips
       are cached on disk (below) and can be generated ahead of time (`pregenerate_speech_lines`).
     - **Klatt** (fallback): Flint's pure-Mojo rule-based formant voice
       (`flint.audio.formant_tts`: misaki G2P, Klatt-style duration / F0 rules, cascade-parallel
       formant synthesizer with an LF glottal source). Each `VOICE_*` profile maps to one
       `formant_voice` preset (vocal-tract scale, F0 register, range, rate and voice quality).
       G2P is rules-only by default; set `FLINT_G2P_DIR` to a misaki lexicon directory
       (`us_gold.json` / `us_silver.json`, `gb_*` for British presets) for dictionary
       pronunciations.
  2. Masters dry speech through a 48 kHz Stereo Giant-Robot / Spaceship OS DSP chain:
     cubic Hermite upsampling, dual-head raised-cosine granular 3-voice harmonizer
     (`-26 cents` Left @ `6.5 ms` Haas, `+26 cents` Right @ `10.5 ms` Haas, and
     `-4.6 st` `0.765x` deep chest sub-voice), dual short-lag (`0.78 ms` & `2.38 ms`)
     titanium vocal-box comb resonance, warm `CytomicSVF` studio vocal-body sculpting,
     and dark-titanium `StereoReverbFDN` cockpit spatialization.

Backend selection (`active_speech_backend`):
  - `FLINT_SPEECH_BACKEND` = `auto` (default) | `kokoro` | `klatt`. `auto` and `kokoro` use Kokoro
    when its weights load and fall back to Klatt otherwise (a clip is never lost).
  - Kokoro weights: `$FLINT_KOKORO_DIR`, else `~/flint-models/kokoro` (`tools/kokoro/fetch.sh`:
    `model.onnx`, `voices/*.bin`, `lexicon/*.json`), else none. The model loads once per process
    (on the first Kokoro clip) and stays cached; it runs on the CPU (`FLINT_KOKORO_GPU=1` moves it
    to the MAX GPU: ~1 s slower to load, similar speed for short lines).

Clip cache (Kokoro clips only; Klatt renders far faster than real time): finished 48 kHz clips live
in `$FLINT_SPEECH_CACHE_DIR`, else `$XDG_CACHE_HOME/flint/speech`, else `~/.cache/flint/speech`, one
file per hash(backend, voice preset, text, profile / chain parameters, model). The full key is stored
in the file and checked on load. Entries are only ever added, never deleted.
`FLINT_SPEECH_CACHE_DIR=off` disables the cache.
"""
from std.math import sin, cos, exp, floor, abs, sqrt
from std.ffi import external_call, _get_global
from std.time import perf_counter_ns
from std.sys import size_of
from flint.platform.os import env_now, file_size, read_bytes, write_bytes, make_dirs, rename_file
from flint.platform.audio import WavClip
from flint.platform.dsp import (
    DSP_TAU,
    PinkNoise,
    CytomicSVF,
    StereoReverbFDN,
    hash_white_noise,
    soft_clip_tanh,
    tube_saturate,
    wrap_phase01,
)
from flint.audio.formant_tts.api import FormantTTS, FORMANT_TTS_SAMPLE_RATE
from flint.audio.formant_tts.voices import formant_voice, FV_DEFAULT
from flint.core.serial import fnv64
from flint.ml.kokoro import KokoroTTS, KOKORO_SAMPLE_RATE, KOKORO_STYLE_DIM
from flint.ml.phonemize import G2P

comptime SPEECH_OUT_RATE: Int = 48000
comptime SPEECH_OUT_SR_F: Float32 = 48000.0

comptime SPEECH_BACKEND_KLATT: Int = 0
comptime SPEECH_BACKEND_KOKORO: Int = 1
comptime SPEECH_BACKEND_AUTO: Int = 2

# Bump when a chain, gain or preset change alters cached clips (it is part of every cache key).
comptime SPEECH_CACHE_REVISION: Int = 1
comptime _CACHE_MAGIC = "FLSPCH01"

# Kokoro dry level into the chains. Kokoro's crest factor is ~6.5-7 against Klatt's ~3-4, so peak
# normalisation (Klatt) would leave it ~6 dB quieter and barely touch the saturation stages; Kokoro
# is RMS-normalised instead, with a soft knee on the sparse plosive peaks.
comptime _KOKORO_DRY_RMS: Float32 = 0.24
comptime _KOKORO_PEAK_CAP: Float32 = 1.6


def _formant_speech_mono(text: String, voice_id: Int, rate_wpm: Int = 0) -> List[Float32]:
    """Dry 24 kHz mono speech from the formant voice preset `voice_id` (`FV_*`).

    `rate_wpm > 0` overrides the preset's speaking rate. Uses the misaki lexicon in
    `FLINT_G2P_DIR` when it is set and loads, else the rules-only G2P. Never raises: a G2P
    failure returns an empty buffer."""
    var v = formant_voice(voice_id)
    if rate_wpm > 0:
        v.rate_wpm = Float32(rate_wpm)
    var lex = env_now("FLINT_G2P_DIR")
    try:
        if lex.byte_length() > 0:
            try:
                var tts_lex = FormantTTS.load(lex, v.british)
                return tts_lex.synthesize(text, v)
            except:
                pass
        var tts = FormantTTS(v.british)
        return tts.synthesize(text, v)
    except:
        return List[Float32]()


def _synth_klatt_fallback_mono(text: String, voice_id: Int = FV_DEFAULT) -> List[Float32]:
    """Dry 24 kHz mono formant speech (kept under its historical name for tools and evals)."""
    return _formant_speech_mono(text, voice_id)


# ==============================================================================
# Kokoro voice presets
# ==============================================================================

@fieldwise_init
struct KokoroVoicePreset(Copyable, Movable):
    """Kokoro rendering of one `VOICE_*` profile.

    `voice` is a pack name (`voices/<name>.bin`) or a style blend `"a*0.7+b*0.3"`. `speed` is
    Kokoro's duration scale (>1 faster), `pitch` the chain's granular pitch ratio (1 = none),
    `british` selects the misaki GB lexicon (the b* packs were trained on it)."""
    var voice: String
    var speed: Float64
    var pitch: Float32
    var british: Bool


def kokoro_voice_preset(profile_id: Int) -> KokoroVoicePreset:
    """Voice pack / speed / pitch per `VOICE_*` profile (packs from `tools/kokoro/fetch.sh`)."""
    if profile_id == VOICE_AI_OS:
        # calm synthetic RP male (the espeak preset was en-gb RP); unhurried, the harmonizer adds the machine
        return KokoroVoicePreset("bm_george", 0.94, 1.0, True)
    if profile_id == VOICE_BRIDGE_FEMALE:
        # composed US bridge officer, crisp and a touch brisk
        return KokoroVoicePreset("af_heart", 1.05, 1.0, False)
    if profile_id == VOICE_ACE_BARITONE:
        # relaxed veteran: Michael's natural register, slightly unhurried
        return KokoroVoicePreset("am_michael", 0.97, 1.0, False)
    if profile_id == VOICE_ACE_TENOR:
        # young, fast, bright: Michael +1.3 st through the chain's pitch stage, quicker delivery
        return KokoroVoicePreset("am_michael", 1.08, 1.08, False)
    if profile_id == VOICE_ACE_CRIMSON:
        # confident British female ace; the chain adds the alien chorus
        return KokoroVoicePreset("bf_emma", 1.02, 1.0, True)
    return KokoroVoicePreset("af_heart", 1.0, 1.0, False)


# ==============================================================================
# Backend selection & the process-wide Kokoro model
# ==============================================================================

struct _KokoroSlot(Movable):
    """Process-wide Kokoro state, behind `mutex` (a calloc'd, zero-initialised pthread mutex)."""
    var tts: Optional[KokoroTTS]
    var gb: Optional[G2P]
    var dir: String          # the directory `tts` was loaded from (or last failed to load)
    var tried: Bool
    var load_seconds: Float64
    var cache_hits: Int
    var cache_misses: Int
    var kokoro_clips: Int
    var mutex: Int

    def __init__(out self):
        self.tts = None
        self.gb = None
        self.dir = String()
        self.tried = False
        self.load_seconds = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.kokoro_clips = 0
        self.mutex = external_call["calloc", Int](Int(1), Int(128))


def _slot_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var raw = external_call["calloc", Int](Int(1), max(64, size_of[_KokoroSlot]()))
    var p = Pointer[_KokoroSlot, MutAnyOrigin](unsafe_from_address=raw)
    __get_address_as_uninit_lvalue(p._mlir_value) = _KokoroSlot()
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=raw)


def _slot_destroy(p: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    # The model lives until the process exits.
    pass


def _slot() -> Pointer[_KokoroSlot, MutAnyOrigin]:
    var g = _get_global["FLINT_SPEECH_KOKORO_SLOT", _slot_init, _slot_destroy]()
    return Pointer[_KokoroSlot, MutAnyOrigin](unsafe_from_address=Int(g.value()))


def _lock(s: Pointer[_KokoroSlot, MutAnyOrigin]):
    _ = external_call["pthread_mutex_lock", Int32](s[].mutex)


def _unlock(s: Pointer[_KokoroSlot, MutAnyOrigin]):
    _ = external_call["pthread_mutex_unlock", Int32](s[].mutex)


def speech_backend_request() -> Int:
    """`FLINT_SPEECH_BACKEND`: `klatt` / `kokoro` / anything else (default) = `auto`."""
    var v = env_now("FLINT_SPEECH_BACKEND").lower()
    if v == "klatt" or v == "formant":
        return SPEECH_BACKEND_KLATT
    if v == "kokoro":
        return SPEECH_BACKEND_KOKORO
    return SPEECH_BACKEND_AUTO


def speech_backend_name(backend: Int) -> String:
    if backend == SPEECH_BACKEND_KOKORO:
        return "kokoro"
    if backend == SPEECH_BACKEND_KLATT:
        return "klatt"
    return "auto"


def _kokoro_dir_ok(dir: String) -> Bool:
    """`dir` holds the Kokoro graph and at least one of the preset voice packs."""
    if dir.byte_length() == 0:
        return False
    if file_size(dir + "/model.onnx") <= 0 and file_size(dir + "/onnx/model.onnx") <= 0:
        return False
    var packs: List[String] = ["af_heart", "am_michael", "bm_george", "bf_emma"]
    for p in packs:
        if file_size(dir + "/voices/" + p + ".bin") > 0:
            return True
    return False


def kokoro_weights_dir() -> String:
    """`$FLINT_KOKORO_DIR`, then `~/flint-models/kokoro`, then "" (no weights)."""
    var e = env_now("FLINT_KOKORO_DIR")
    if _kokoro_dir_ok(e):
        return e
    var home = env_now("HOME")
    if home.byte_length() > 0 and _kokoro_dir_ok(home + "/flint-models/kokoro"):
        return home + "/flint-models/kokoro"
    return String()


def _ensure_kokoro_locked(s: Pointer[_KokoroSlot, MutAnyOrigin], dir: String) -> Bool:
    """Load (once per weights directory) the model into the slot. Caller holds the lock."""
    if s[].tried and s[].dir == dir:
        return True if s[].tts else False
    s[].tried = True
    s[].dir = dir
    s[].tts = None
    s[].gb = None
    var t0 = perf_counter_ns()
    try:
        var gpu = env_now("FLINT_KOKORO_GPU") == "1"
        var tts = KokoroTTS.load(dir, gpu)
        s[].tts = tts^
        if file_size(dir + "/lexicon/gb_gold.json") > 0:
            s[].gb = G2P.load(dir + "/lexicon", True)
    except e:
        print("[flint.speech] Kokoro failed to load from", dir, "(", e, "), using the Klatt voice")
        s[].tts = None
    s[].load_seconds = Float64(perf_counter_ns() - t0) / 1e9
    return True if s[].tts else False


def active_speech_backend() -> Int:
    """The backend the next clip will use (loads Kokoro on first use in `auto` / `kokoro`)."""
    if speech_backend_request() == SPEECH_BACKEND_KLATT:
        return SPEECH_BACKEND_KLATT
    var dir = kokoro_weights_dir()
    if dir.byte_length() == 0:
        return SPEECH_BACKEND_KLATT
    var s = _slot()
    _lock(s)
    var ok = _ensure_kokoro_locked(s, dir)
    _unlock(s)
    return SPEECH_BACKEND_KOKORO if ok else SPEECH_BACKEND_KLATT


def kokoro_load_seconds() -> Float64:
    """Wall time of the (single) Kokoro model load, 0 before it."""
    var s = _slot()
    return s[].load_seconds


def speech_cache_stats() -> Tuple[Int, Int]:
    """(hits, misses) of the on-disk clip cache in this process."""
    var s = _slot()
    return (s[].cache_hits, s[].cache_misses)


def kokoro_clips_rendered() -> Int:
    """Clips this process synthesised with Kokoro (cache hits excluded)."""
    var s = _slot()
    return s[].kokoro_clips


def _blend_voice(mut tts: KokoroTTS, spec: String) raises:
    """Register the style blend `"a*0.7+b*0.3"` under its own name in `tts.voices`."""
    if spec in tts.voices:
        return
    var acc = List[Float32]()
    var total = Float32(0.0)
    for part in spec.split("+"):
        var p = String(part)
        var w = Float32(1.0)
        var name = p
        var star = p.find("*")
        if star >= 0:
            name = String(p[byte=0:star])
            w = Float32(Float64(String(p[byte=star + 1:])))
        var pack = tts.voice(name)
        if len(acc) == 0:
            acc = List[Float32](length=len(pack), fill=0.0)
        for k in range(min(len(acc), len(pack))):
            acc[k] += pack[k] * w
        total += w
    if total > Float32(0.0) and total != Float32(1.0):
        for k in range(len(acc)):
            acc[k] /= total
    tts.voices[spec] = acc^


def _kokoro_dry_mono(text: String, preset: KokoroVoicePreset, speed: Float64) -> List[Float32]:
    """Dry 24 kHz Kokoro speech; empty when the model is unavailable or synthesis fails."""
    var dir = kokoro_weights_dir()
    if dir.byte_length() == 0:
        return List[Float32]()
    var s = _slot()
    _lock(s)
    var out = List[Float32]()
    if _ensure_kokoro_locked(s, dir):
        try:
            ref tts = s[].tts.value()
            if preset.voice.find("+") >= 0:
                _blend_voice(tts, preset.voice)
            var phonemes: String
            if preset.british and s[].gb:
                phonemes = s[].gb.value().phonemize(text).phonemes
            else:
                phonemes = tts.g2p.phonemize(text).phonemes
            var sp = tts.speak_phonemes(phonemes, preset.voice, speed)
            out = sp.samples.copy()
            s[].kokoro_clips += 1
        except e:
            print("[flint.speech] Kokoro synthesis failed (", e, "), using the Klatt voice")
            out = List[Float32]()
    _unlock(s)
    return out^


# ==============================================================================
# On-disk clip cache
# ==============================================================================

def speech_cache_dir() -> String:
    """`$FLINT_SPEECH_CACHE_DIR` (`off` / `0` disables), else `$XDG_CACHE_HOME/flint/speech`,
    else `~/.cache/flint/speech`; "" when none can be formed."""
    var o = env_now("FLINT_SPEECH_CACHE_DIR")
    if o.byte_length() > 0:
        if o == "off" or o == "0" or o == "none":
            return String()
        return o
    var x = env_now("XDG_CACHE_HOME")
    if x.byte_length() > 0:
        return x + "/flint/speech"
    var home = env_now("HOME")
    if home.byte_length() > 0:
        return home + "/.cache/flint/speech"
    return String()


def _hex64(v: UInt64) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & UInt64(15))
        out += String(digits[byte=nib])
    return out^


def _kokoro_model_tag(dir: String) -> String:
    var m = file_size(dir + "/model.onnx")
    if m <= 0:
        m = file_size(dir + "/onnx/model.onnx")
    return dir + "#" + String(m)


def _clip_cache_key(chain: String, text: String, preset: KokoroVoicePreset, speed: Float64,
                    profile_id: Int, lead_in_sec: Float32, radio_bandpass: Bool) -> String:
    var k = String("rev=") + String(SPEECH_CACHE_REVISION)
    k += "|backend=kokoro|model=" + _kokoro_model_tag(kokoro_weights_dir())
    k += "|chain=" + chain + "|profile=" + String(profile_id)
    k += "|voice=" + preset.voice + "|speed=" + String(speed) + "|pitch=" + String(preset.pitch)
    k += "|gb=" + ("1" if preset.british else "0")
    k += "|lead=" + String(lead_in_sec) + "|band=" + ("1" if radio_bandpass else "0")
    k += "|text=" + text
    return k^


def _cache_path(key: String) -> String:
    var dir = speech_cache_dir()
    if dir.byte_length() == 0:
        return String()
    var b = List[UInt8]()
    for c in key.as_bytes():
        b.append(c)
    return dir + "/" + _hex64(fnv64(b)) + ".flspeech"


def _u32_at(d: List[UInt8], o: Int) -> Int:
    return Int(d[o]) | (Int(d[o + 1]) << 8) | (Int(d[o + 2]) << 16) | (Int(d[o + 3]) << 24)


def _put_u32(mut d: List[UInt8], v: Int):
    d.append(UInt8(v & 255))
    d.append(UInt8((v >> 8) & 255))
    d.append(UInt8((v >> 16) & 255))
    d.append(UInt8((v >> 24) & 255))


def _cache_load(key: String) -> WavClip:
    var clip = WavClip()
    var path = _cache_path(key)
    if path.byte_length() == 0 or file_size(path) <= 0:
        return clip^
    var d = read_bytes(path)
    var magic = String(_CACHE_MAGIC)
    var mb = magic.as_bytes()
    if len(d) < 8 + 16:
        return clip^
    for i in range(8):
        if d[i] != mb[i]:
            return clip^
    var klen = _u32_at(d, 8)
    var kb = key.as_bytes()
    if klen != len(kb) or len(d) < 12 + klen + 12:
        return clip^
    for i in range(klen):
        if d[12 + i] != kb[i]:
            return clip^
    var o = 12 + klen
    var ch = _u32_at(d, o)
    var sr = _u32_at(d, o + 4)
    var n = _u32_at(d, o + 8)
    o += 12
    if ch < 1 or ch > 2 or sr <= 0 or len(d) < o + 2 * n:
        return clip^
    clip.samples.reserve(n)
    for i in range(n):
        var u = Int(d[o + 2 * i]) | (Int(d[o + 2 * i + 1]) << 8)
        clip.samples.append(Int16(u - 65536 if u >= 32768 else u))
    clip.channels = ch
    clip.sample_rate = sr
    clip.valid = True
    return clip^


def _cache_store(key: String, clip: WavClip):
    var path = _cache_path(key)
    if path.byte_length() == 0 or not clip.valid:
        return
    if not make_dirs(speech_cache_dir()):
        return
    var d = List[UInt8](capacity=12 + key.byte_length() + 12 + 2 * len(clip.samples))
    for c in String(_CACHE_MAGIC).as_bytes():
        d.append(c)
    _put_u32(d, key.byte_length())
    for c in key.as_bytes():
        d.append(c)
    _put_u32(d, clip.channels)
    _put_u32(d, clip.sample_rate)
    _put_u32(d, len(clip.samples))
    for v in clip.samples:
        var u = Int(v)
        if u < 0:
            u += 65536
        d.append(UInt8(u & 255))
        d.append(UInt8((u >> 8) & 255))
    # write a private temp file, then rename over: readers never see a partial clip
    var tmp = path + ".tmp" + String(Int(external_call["getpid", Int32]()))
    if write_bytes(tmp, d):
        _ = rename_file(tmp, path)


def _cache_count(hit: Bool):
    var s = _slot()
    _lock(s)
    if hit:
        s[].cache_hits += 1
    else:
        s[].cache_misses += 1
    _unlock(s)


# ==============================================================================
# Shared dry-voice staging
# ==============================================================================

struct _Dry(Movable):
    """Dry mono voice handed to a chain: samples, rate, backend and the chain pitch ratio."""
    var mono: List[Float32]
    var sample_rate: Int
    var backend: Int
    var pitch: Float32

    def __init__(out self, var mono: List[Float32], sample_rate: Int, backend: Int, pitch: Float32):
        self.mono = mono^
        self.sample_rate = sample_rate
        self.backend = backend
        self.pitch = pitch


def _kokoro_dry(text: String, profile_id: Int, rate_scale: Float64) -> _Dry:
    var preset = kokoro_voice_preset(profile_id)
    var mono = _kokoro_dry_mono(text, preset, preset.speed * rate_scale)
    return _Dry(mono^, KOKORO_SAMPLE_RATE, SPEECH_BACKEND_KOKORO, preset.pitch)


def _stage_dry48(dry: _Dry, peak_target: Float32, fade_in_len: Float32, fade_out_len: Float32) -> List[Float32]:
    """Trim, level and Hermite-upsample a dry voice to 48 kHz.

    Klatt: trailing-silence trim and peak normalisation to `peak_target` (unchanged).
    Kokoro: leading / trailing silence trim (the lead-in is the chain's job), RMS normalisation to
    `_KOKORO_DRY_RMS` with the gain capped at `_KOKORO_PEAK_CAP / peak` and a soft knee above 0.7."""
    var mono = dry.mono.copy()
    if len(mono) < 64:
        mono = List[Float32](length=64, fill=0.0)
    var kok = dry.backend == SPEECH_BACKEND_KOKORO
    var src_sr = dry.sample_rate

    var end_idx = len(mono)
    var thr = Float32(0.0025)
    var start_idx = 0
    if kok:
        var pk0 = Float32(0.0)
        for x in mono:
            pk0 = max(pk0, abs(x))
        thr = max(Float32(0.0025), pk0 * Float32(0.006))
        while start_idx < end_idx - 256 and abs(mono[start_idx]) < thr:
            start_idx += 1
        start_idx = max(0, start_idx - src_sr // 100)    # keep 10 ms before the onset
    while end_idx > start_idx + 256 and abs(mono[end_idx - 1]) < thr:
        end_idx -= 1
    if kok:
        end_idx = min(len(mono), end_idx + src_sr // 50)  # 20 ms release

    var peak = Float32(0.05)
    var e = Float64(0.0)
    var ne = 0
    for i in range(start_idx, end_idx):
        var a = abs(mono[i])
        if a > peak:
            peak = a
    var gain = peak_target / peak
    if kok:
        # RMS over the active samples (> -40 dB re peak) so pauses do not dilute it
        for i in range(start_idx, end_idx):
            var a = abs(mono[i])
            if a > peak * Float32(0.01):
                e += Float64(a) * Float64(a)
                ne += 1
        var rms = Float32(sqrt(e / Float64(max(1, ne)))) if ne > 0 else Float32(0.05)
        gain = min(_KOKORO_DRY_RMS / max(rms, Float32(1e-4)), _KOKORO_PEAK_CAP / peak)

    var n_src = end_idx - start_idx
    var voice_frames = max(64, Int(Float64(n_src) * Float64(SPEECH_OUT_RATE) / Float64(src_sr)))
    var step_src = Float64(src_sr) / Float64(SPEECH_OUT_RATE)
    var dry48 = List[Float32](capacity=voice_frames)
    for vf in range(voice_frames):
        var src_pos = Float64(start_idx) + Float64(vf) * step_src
        var fade_in = min(Float32(1.0), Float32(vf) / fade_in_len)
        var fade_out = min(Float32(1.0), Float32(voice_frames - vf) / fade_out_len)
        var x = _hermite_sample(mono, src_pos) * gain
        if kok:
            var ax = abs(x)
            if ax > Float32(0.7):
                # soft knee: 0.7 + 0.3 * tanh((|x| - 0.7) / 0.3), never above 1.0
                var over = soft_clip_tanh((ax - Float32(0.7)) / Float32(0.3)) * Float32(0.3)
                x = (Float32(0.7) + over) if x > 0 else -(Float32(0.7) + over)
        dry48.append(x * fade_in * fade_out)
    return dry48^


@always_inline
def _hermite_sample(buf: List[Float32], pos: Float64) -> Float32:
    var n = len(buf)
    if n == 0:
        return 0.0
    var i1 = Int(floor(pos))
    var frac = Float32(pos - floor(pos))
    var i0 = max(0, min(n - 1, i1 - 1))
    var ic = max(0, min(n - 1, i1))
    var i2 = max(0, min(n - 1, i1 + 1))
    var i3 = max(0, min(n - 1, i1 + 2))
    var y0 = buf[i0]
    var y1 = buf[ic]
    var y2 = buf[i2]
    var y3 = buf[i3]
    var c0 = y1
    var c1 = Float32(0.5) * (y2 - y0)
    var c2 = y0 - Float32(2.5) * y1 + Float32(2.0) * y2 - Float32(0.5) * y3
    var c3 = Float32(0.5) * (y3 - y0) + Float32(1.5) * (y1 - y2)
    return ((c3 * frac + c2) * frac + c1) * frac + c0


def _pitch_shift_ola(x: List[Float32], ratio: Float32, grain: Int) -> List[Float32]:
    """Dual-head raised-cosine (`Hann`) time-preserving granular pitch-shifter (`48 kHz`)."""
    var n = len(x)
    var out = List[Float32](capacity=max(1, n))
    if n < 4:
        for i in range(n):
            out.append(x[i])
        return out^

    var g_f = Float32(max(128, grain))
    var delta = (Float32(1.0) - ratio) / g_f
    var ph = Float32(0.0)
    var max_pos = Float32(n - 2)

    for i in range(n):
        ph = wrap_phase01(ph + delta)
        var p2 = wrap_phase01(ph + Float32(0.5))
        var w1 = Float32(0.5) - Float32(0.5) * cos(DSP_TAU * ph)
        var w2 = Float32(0.5) - Float32(0.5) * cos(DSP_TAU * p2)

        var pos1 = max(Float32(0.0), min(max_pos, Float32(i) - ph * g_f))
        var pos2 = max(Float32(0.0), min(max_pos, Float32(i) - p2 * g_f))
        var i1 = Int(pos1)
        var f1 = pos1 - Float32(i1)
        var i2 = Int(pos2)
        var f2 = pos2 - Float32(i2)

        var s1 = x[i1] * (Float32(1.0) - f1) + x[i1 + 1] * f1
        var s2 = x[i2] * (Float32(1.0) - f2) + x[i2 + 1] * f2
        out.append(s1 * w1 + s2 * w2)
    return out^


# ==============================================================================
# AI OS (Giant-Robot / Spaceship OS) chain
# ==============================================================================

def synth_ai_copilot_clip(
    text: String,
    lead_in_sec: Float32 = 0.24,
    voice_name: String = "en-gb-x-rp+m3",
    rate_wpm: Int = 148,
    pitch_0_99: Int = 26,
    range_0_99: Int = 20,
) -> WavClip:
    """Synthesizes a 48 kHz Stereo Giant-Robot / Spaceship OS AI Co-Pilot voice `WavClip`.
    Includes `lead_in_sec` pre-roll so the simultaneous Guidance Start Alert Chime rings
    cleanly while `NativeAudio` smoothly auto-ducks BGM before the first spoken word.
    With Kokoro, `rate_wpm` scales the preset speed (148 = preset)."""
    # `voice_name` / `pitch_0_99` / `range_0_99` were espeak-ng controls; the AI OS preset
    # (Kokoro voice pack, or the calm RP baritone formant voice) now carries voice, register and range.
    _ = voice_name
    _ = pitch_0_99
    _ = range_0_99
    if active_speech_backend() == SPEECH_BACKEND_KOKORO:
        var rate_scale = Float64(rate_wpm) / 148.0 if rate_wpm > 0 else 1.0
        var preset = kokoro_voice_preset(VOICE_AI_OS)
        var key = _clip_cache_key("ai_os", text, preset, preset.speed * rate_scale, VOICE_AI_OS, lead_in_sec, False)
        var hit = _cache_load(key)
        if hit.valid:
            _cache_count(True)
            return hit^
        var dry = _kokoro_dry(text, VOICE_AI_OS, rate_scale)
        if len(dry.mono) >= 64:
            _cache_count(False)
            var clip = _ai_os_chain(dry, lead_in_sec)
            _cache_store(key, clip)
            return clip^
    var mono = _formant_speech_mono(text, VOICE_AI_OS, rate_wpm)
    return _ai_os_chain(_Dry(mono^, FORMANT_TTS_SAMPLE_RATE, SPEECH_BACKEND_KLATT, 1.0), lead_in_sec)


def _ai_os_chain(dry: _Dry, lead_in_sec: Float32) -> WavClip:
    var kok = dry.backend == SPEECH_BACKEND_KOKORO
    # Peak-normalize dry mono buffer to consistent studio vocal level (`0.82`; Kokoro: RMS staging)
    var dry48 = _stage_dry48(dry, 0.82, 384.0, 480.0)
    var voice_frames = len(dry48)
    var lead_frames = max(0, Int(lead_in_sec * SPEECH_OUT_SR_F))
    var tail_frames = Int(Float32(0.095) * SPEECH_OUT_SR_F)
    var total_frames = lead_frames + voice_frames + tail_frames

    # 3-Voice Binaural Giant-Robot OS Harmonizer:
    #   - Left micro-pitch (`0.985x`, `-26 cents`) @ `6.5 ms` (`312` samples) Haas offset
    #   - Right micro-pitch (`1.015x`, `+26 cents`) @ `10.5 ms` (`504` samples) Haas offset
    #   - Deep chest sub-harmonic layer (`0.765x`, `-4.6 semitones`) anchored dead-center
    var v_l = _pitch_shift_ola(dry48, 0.985, 1680)
    var v_r = _pitch_shift_ola(dry48, 1.015, 1680)
    var v_sub = _pitch_shift_ola(dry48, 0.765, 1920)
    var haas_l = 312
    var haas_r = 504

    # Kokoro: natural voices are already bright in 2-4 kHz (Klatt's were dark), so the presence lift
    # is halved; the harmonizer / comb / reverb character is unchanged.
    var presence_amt = Float32(0.14) if kok else Float32(0.28)

    var clip = WavClip()
    clip.valid = True
    clip.channels = 2
    clip.sample_rate = SPEECH_OUT_RATE
    clip.samples.reserve(total_frames * 2)

    # Dual short-lag titanium vocal-box comb resonators (`0.77 ms` & `2.38 ms`) + Studio EQ
    var comb1_l = List[Float32](length=37, fill=0.0)
    var comb1_r = List[Float32](length=39, fill=0.0)
    var comb2_l = List[Float32](length=113, fill=0.0)
    var comb2_r = List[Float32](length=119, fill=0.0)
    var hp_clean = CytomicSVF(72.0, 0.78, SPEECH_OUT_SR_F)
    var presence_l = CytomicSVF(2550.0, 1.35, SPEECH_OUT_SR_F)
    var presence_r = CytomicSVF(2550.0, 1.35, SPEECH_OUT_SR_F)
    var warm_lp_l = CytomicSVF(6200.0, 0.72, SPEECH_OUT_SR_F)
    var warm_lp_r = CytomicSVF(6200.0, 0.72, SPEECH_OUT_SR_F)
    var chamber_rev = StereoReverbFDN(0.38, 0.56)

    for f in range(total_frames):
        var mix_l = Float32(0.0)
        var mix_r = Float32(0.0)
        if f >= lead_frames and f < lead_frames + voice_frames:
            var vf = f - lead_frames
            _ = hp_clean.process(dry48[vf])
            var c_dry = hp_clean.hp
            var vl = v_l[vf - haas_l] if vf >= haas_l else Float32(0.0)
            var vr = v_r[vf - haas_r] if vf >= haas_r else Float32(0.0)
            var vsub = v_sub[vf]
            mix_l = c_dry * Float32(0.56) + vl * Float32(0.54) - vr * Float32(0.15) + vsub * Float32(0.24)
            mix_r = c_dry * Float32(0.56) + vr * Float32(0.54) - vl * Float32(0.15) + vsub * Float32(0.24)

        # Subtle dark-titanium telemetry lock micro-tail (`880 -> 587.33 Hz`) right as transmission closes
        var pip = Float32(0.0)
        if f >= lead_frames + voice_frames and f < lead_frames + voice_frames + 1440:
            var pt = Float32(f - (lead_frames + voice_frames)) / SPEECH_OUT_SR_F
            var pf = Float32(880.0) if pt < Float32(0.014) else Float32(587.33)
            pip = sin(pt * pf * DSP_TAU) * exp(-pt * Float32(72.0)) * Float32(0.11)

        var c1_idx_l = f % 37
        var c1_idx_r = f % 39
        var c2_idx_l = f % 113
        var c2_idx_r = f % 119
        var res_l = mix_l + comb1_l[c1_idx_l] * Float32(0.23) + comb2_l[c2_idx_l] * Float32(0.15)
        var res_r = mix_r + comb1_r[c1_idx_r] * Float32(0.23) + comb2_r[c2_idx_r] * Float32(0.15)
        comb1_l[c1_idx_l] = res_l
        comb1_r[c1_idx_r] = res_r
        comb2_l[c2_idx_l] = res_l
        comb2_r[c2_idx_r] = res_r

        _ = presence_l.process(res_l)
        _ = presence_r.process(res_r)
        var eq_l = warm_lp_l.process(res_l + presence_l.bp * presence_amt) + pip
        var eq_r = warm_lp_r.process(res_r + presence_r.bp * presence_amt) + pip

        var sat_l = tube_saturate(eq_l, Float32(1.22))
        var sat_r = tube_saturate(eq_r, Float32(1.22))
        var wet = chamber_rev.process_send(sat_l * Float32(0.11), sat_r * Float32(0.11))

        var out_l = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh((sat_l + wet[0]) * Float32(1.08))))
        var out_r = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh((sat_r + wet[1]) * Float32(1.08))))
        clip.samples.append(Int16(Int(out_l * Float32(29500.0))))
        clip.samples.append(Int16(Int(out_r * Float32(29500.0))))

    return clip^


# ==============================================================================
# C3: Multi-Character Bridge & Squadron Radio Comms DSP
# ==============================================================================

comptime VOICE_AI_OS: Int = 0
comptime VOICE_BRIDGE_FEMALE: Int = 1  # Lisa Hayes (SDF-1 Bridge Tactical Officer)
comptime VOICE_ACE_BARITONE: Int = 2   # Roy Focker (Skull Leader)
comptime VOICE_ACE_TENOR: Int = 3      # Max Sterling (Blue Ace)
comptime VOICE_ACE_CRIMSON: Int = 4    # Miriya Parina (Quadrono Female Ace)


def synth_radio_comms_clip(
    text: String,
    profile_id: Int,
    lead_in_sec: Float32 = 0.14,
    radio_bandpass: Bool = True,
) -> WavClip:
    """Synthesizes a 48 kHz Stereo character radio voice clip with distinct vocal-tract formants,
    fundamental pitch and voice quality (one Kokoro voice preset or `formant_voice` preset per
    profile), military UHF `320–3350 Hz` bandpass, PTT click, and two-tone tactical squelch tail."""
    if profile_id == VOICE_AI_OS:
        return synth_ai_copilot_clip(text, lead_in_sec)

    if active_speech_backend() == SPEECH_BACKEND_KOKORO:
        var preset = kokoro_voice_preset(profile_id)
        var key = _clip_cache_key("radio", text, preset, preset.speed, profile_id, lead_in_sec, radio_bandpass)
        var hit = _cache_load(key)
        if hit.valid:
            _cache_count(True)
            return hit^
        var dry = _kokoro_dry(text, profile_id, 1.0)
        if len(dry.mono) >= 64:
            _cache_count(False)
            var clip = _radio_chain(dry, profile_id, lead_in_sec, radio_bandpass)
            _cache_store(key, clip)
            return clip^

    # Each profile's register, vocal-tract scale and speaking rate live in its formant preset
    # (`flint.audio.formant_tts.voices`), so the dry voice is not pitch-shifted afterwards.
    var voice_id = profile_id if (profile_id >= VOICE_BRIDGE_FEMALE and profile_id <= VOICE_ACE_CRIMSON) else FV_DEFAULT
    var mono = _formant_speech_mono(text, voice_id)
    return _radio_chain(_Dry(mono^, FORMANT_TTS_SAMPLE_RATE, SPEECH_BACKEND_KLATT, 1.0), profile_id, lead_in_sec, radio_bandpass)


def _radio_chain(dry: _Dry, profile_id: Int, lead_in_sec: Float32, radio_bandpass: Bool) -> WavClip:
    var kok = dry.backend == SPEECH_BACKEND_KOKORO
    var pitch_shift = dry.pitch
    var pan_bias = Float32(0.0)
    var roger_hi = Float32(1174.66)
    var roger_lo = Float32(880.0)

    if profile_id == VOICE_BRIDGE_FEMALE:
        pan_bias = -0.10
        roger_hi = 1318.51
        roger_lo = 987.77
    elif profile_id == VOICE_ACE_BARITONE:
        pan_bias = -0.16
        roger_hi = 1046.50
        roger_lo = 783.99
    elif profile_id == VOICE_ACE_TENOR:
        pan_bias = 0.16
        roger_hi = 1174.66
        roger_lo = 987.77
    elif profile_id == VOICE_ACE_CRIMSON:
        pan_bias = 0.20
        roger_hi = 1396.91
        roger_lo = 698.46

    var dry48 = _stage_dry48(dry, 0.84, 320.0, 384.0)
    var voice_frames = len(dry48)
    var lead_frames = max(0, Int(lead_in_sec * SPEECH_OUT_SR_F))
    var tail_frames = Int(Float32(0.11) * SPEECH_OUT_SR_F)
    var total_frames = lead_frames + voice_frames + tail_frames

    var voiced48 = _pitch_shift_ola(dry48, pitch_shift, 1536)
    var alien_chorus = _pitch_shift_ola(dry48, pitch_shift * Float32(0.985), 1440) if profile_id == VOICE_ACE_CRIMSON else List[Float32]()

    # Kokoro gain staging: the mid "bite" peak was there to lift Klatt's dark 2-4 kHz band and is
    # halved for Kokoro; the radio band, saturation, noise bed and squelch are unchanged.
    var bite_amt = Float32(0.55) if radio_bandpass else Float32(0.22)
    if kok:
        bite_amt = Float32(0.28) if radio_bandpass else Float32(0.12)

    var clip = WavClip()
    clip.valid = True
    clip.channels = 2
    clip.sample_rate = SPEECH_OUT_RATE
    clip.samples.reserve(total_frames * 2)

    var hp_uhf = CytomicSVF(Float32(320.0) if radio_bandpass else Float32(85.0), 0.85, SPEECH_OUT_SR_F)
    var mid_bite = CytomicSVF(2150.0, 1.55, SPEECH_OUT_SR_F)
    var lp_uhf = CytomicSVF(Float32(3350.0) if radio_bandpass else Float32(6800.0), 0.88, SPEECH_OUT_SR_F)
    var cabin_rev = StereoReverbFDN(0.26, 0.42)
    var pink = PinkNoise(313 + profile_id * 97)
    var gain_l = Float32(1.0) - max(Float32(0.0), pan_bias) * Float32(0.35)
    var gain_r = Float32(1.0) + min(Float32(0.0), pan_bias) * Float32(0.35)

    for f in range(total_frames):
        var sig = Float32(0.0)
        # PTT open carrier burst
        if radio_bandpass and f >= max(0, lead_frames - 720) and f < lead_frames:
            var kt = Float32(f - max(0, lead_frames - 720)) / SPEECH_OUT_SR_F
            sig += hash_white_noise(f * 17 + profile_id) * exp(-kt * Float32(90.0)) * Float32(0.16)

        if f >= lead_frames and f < lead_frames + voice_frames:
            var vf = f - lead_frames
            var v = voiced48[vf]
            if profile_id == VOICE_ACE_CRIMSON and vf < len(alien_chorus):
                v = v * Float32(0.76) + alien_chorus[vf] * Float32(0.30)
            _ = hp_uhf.process(v)
            _ = mid_bite.process(hp_uhf.hp)
            var shaped = lp_uhf.process(hp_uhf.hp + mid_bite.bp * bite_amt)
            if radio_bandpass:
                shaped = tube_saturate(shaped + pink.next() * Float32(0.012), Float32(1.48))
            sig += shaped

        # Two-tone tactical UHF squelch tail + carrier close click
        if f >= lead_frames + voice_frames and f < lead_frames + voice_frames + 2400:
            var pt = Float32(f - (lead_frames + voice_frames)) / SPEECH_OUT_SR_F
            var pf = roger_hi if pt < Float32(0.022) else roger_lo
            var beep = sin(pt * pf * DSP_TAU) * exp(-pt * Float32(48.0)) * Float32(0.12)
            var squelch = (hash_white_noise(f * 29 + 3) * exp(-max(Float32(0.0), pt - Float32(0.028)) * Float32(110.0)) * Float32(0.09)) if (radio_bandpass and pt >= Float32(0.028)) else Float32(0.0)
            sig += beep + squelch

        var wet = cabin_rev.process_send(sig * gain_l * Float32(0.07), sig * gain_r * Float32(0.07))
        var out_l = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh(sig * gain_l + wet[0])))
        var out_r = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh(sig * gain_r + wet[1])))
        clip.samples.append(Int16(Int(out_l * Float32(29000.0))))
        clip.samples.append(Int16(Int(out_r * Float32(29000.0))))

    return clip^


# ==============================================================================
# Ahead-of-time generation
# ==============================================================================

def pregenerate_speech_lines(
    lines: List[String],
    profile_id: Int,
    lead_in_sec: Float32 = 0.14,
    radio_bandpass: Bool = True,
) -> Int:
    """Render `lines` for `profile_id` into the on-disk clip cache (loading screens, level start),
    so the later `synth_radio_comms_clip(line, profile_id, lead_in_sec, radio_bandpass)` calls are
    cache hits. Returns how many lines were newly synthesised (0 when all were cached, or with the
    Klatt backend, which needs no cache)."""
    if active_speech_backend() != SPEECH_BACKEND_KOKORO:
        return 0
    var before = kokoro_clips_rendered()
    for l in lines:
        _ = synth_radio_comms_clip(l, profile_id, lead_in_sec, radio_bandpass)
    return kokoro_clips_rendered() - before


def pregenerate_copilot_lines(lines: List[String], lead_in_sec: Float32 = 0.24, rate_wpm: Int = 148) -> Int:
    """`pregenerate_speech_lines` for `synth_ai_copilot_clip(line, lead_in_sec, rate_wpm=rate_wpm)`."""
    if active_speech_backend() != SPEECH_BACKEND_KOKORO:
        return 0
    var before = kokoro_clips_rendered()
    for l in lines:
        _ = synth_ai_copilot_clip(l, lead_in_sec, rate_wpm=rate_wpm)
    return kokoro_clips_rendered() - before
