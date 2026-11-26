"""Voice presets for the Flint formant voice (`flint.audio.formant_tts.voices`).

The radio / AI-OS presets transcribe the prosody note's §3.2 table. F0 and formant scales are the
*final* targets (the old espeak path's `klatt_f0 x pitch_shift`), so the clip chains can skip the
granular pitch shifter. Per-formant female scales follow Peterson & Barney 1952 / Hillenbrand et
al. 1995 ratios (Fant 1966: female scaling is not uniform). Source settings use KLSYN88 names
(Klatt & Klatt 1990): OQ is mapped to the LF Rd parameter (Fant 1995) as Rd ~ Rd(Te/T0 = OQ + 0.12),
which puts KLSYN88's modal OQ 0.5 at Rd ~0.85 and the breathy female OQ 0.65 at Rd ~1.8.
"""

comptime FV_DEFAULT: Int = -1      # neutral US male, the stand-alone default
comptime FV_AI_OS: Int = 0
comptime FV_BRIDGE_FEMALE: Int = 1
comptime FV_ACE_BARITONE: Int = 2
comptime FV_ACE_TENOR: Int = 3
comptime FV_ACE_CRIMSON: Int = 4


from flint.core.detmath import det_exp, det_cos, det_sin


struct FormantVoice(Copyable, Movable):
    var id: Int
    var f0_base: Float32      # F_b: phrase baseline (Hz)
    var range_st: Float32     # accent range R (semitones)
    var decl: Float32         # declination multiplier on d_type
    var rate_wpm: Float32     # nominal rate (the old call-site numbers)
    var fs1: Float32          # per-formant scale F1..F5 (F6 / F7 / parallel use fs5)
    var fs2: Float32
    var fs3: Float32
    var fs4: Float32
    var fs5: Float32
    var bw1: Float32          # extra B1 factor (female +20 %)
    var rd: Float32           # LF Rd
    var tl: Float32           # spectral tilt dB at 3 kHz
    var ah_rel: Float32       # aspiration noise during voicing, dB re AV
    var jitter: Float32       # fraction of T0 (RMS)
    var shimmer: Float32      # dB RMS
    var flutter: Float32      # KLSYN88 FL 0..100
    var df1: Float32          # pitch-synchronous F1 / B1 excursion (Hz)
    var db1: Float32
    var creak: Bool           # phrase-final creak (baritone)
    var tracheal: Bool        # subglottal pole / zero pair (breathy voices)
    var british: Bool         # GB lexicon
    var f0_floor: Float32
    var shelf_db: Float32     # first-order high shelf (corner ~1.6 kHz), dB at the top of the band
    var presence_db: Float32  # peaking boost at 2.8 kHz (Q 1.2): consonant cues inside a 3.4 kHz radio band

    def __init__(out self):
        self.id = FV_DEFAULT
        self.f0_base = 100.0
        self.range_st = 7.5
        self.decl = 1.0
        self.rate_wpm = 165.0
        self.fs1 = 1.0
        self.fs2 = 1.0
        self.fs3 = 1.0
        self.fs4 = 1.0
        self.fs5 = 1.0
        self.bw1 = 1.0
        self.rd = 0.7
        self.tl = 0.0
        self.ah_rel = -24.0
        self.jitter = 0.005
        self.shimmer = 0.3
        self.flutter = 25.0
        self.df1 = 20.0
        self.db1 = 60.0
        self.creak = False
        self.tracheal = False
        self.british = False
        self.f0_floor = 65.0
        self.shelf_db = 0.0
        self.presence_db = 0.0

    def set_scales(mut self, a: Float32, b: Float32, c: Float32, d: Float32, e: Float32):
        self.fs1 = a
        self.fs2 = b
        self.fs3 = c
        self.fs4 = d
        self.fs5 = e


def formant_voice(id: Int) -> FormantVoice:
    """Preset by id: `FV_DEFAULT` or the `speech.mojo` VOICE_* ids 0..4.

    Rd / TL sit well below the KLSYN88-derived values the presets were designed with (AI OS 0.85 / 4,
    bridge 1.6 / 10, ...): with this LF source those were too dark for STT through the radio band
    (tuning log, preset tables); voice character now comes from F0, formant scale, rate and noise."""
    var v = FormantVoice()
    v.id = id
    if id == FV_AI_OS:
        v.shelf_db = 6.0
        v.presence_db = 6.0
        # calm machine RP baritone, steady (4.5 st: at 2.5 st words ran together, tuning log); mean 112 Hz
        v.f0_base = 104.0
        v.range_st = 4.5
        v.decl = 0.6
        v.rate_wpm = 148.0
        v.set_scales(0.97, 0.97, 0.96, 0.96, 0.96)
        v.rd = 0.65
        v.tl = 0.0
        v.ah_rel = -34.0
        v.jitter = 0.002
        v.shimmer = 0.10
        v.flutter = 8.0
        v.df1 = 10.0
        v.db1 = 30.0
        v.british = True
    elif id == FV_BRIDGE_FEMALE:
        v.shelf_db = 10.0
        v.presence_db = 9.0
        # composed officer: crisp, slightly breathy; mean 222 Hz
        v.f0_base = 200.0
        v.range_st = 6.5
        v.rate_wpm = 160.0
        v.set_scales(1.16, 1.17, 1.13, 1.10, 1.08)
        v.bw1 = 1.2
        v.rd = 0.9
        v.tl = 2.0
        v.ah_rel = -22.0
        v.jitter = 0.005
        v.shimmer = 0.25
        v.flutter = 20.0
        v.df1 = 40.0
        v.db1 = 150.0
        v.tracheal = True
        v.f0_floor = 120.0
    elif id == FV_ACE_BARITONE:
        v.shelf_db = 10.0
        v.presence_db = 9.0
        # relaxed veteran, phrase-final creak; mean 99 Hz
        v.f0_base = 90.0
        v.range_st = 5.5
        v.rate_wpm = 145.0
        v.set_scales(0.90, 0.90, 0.91, 0.92, 0.93)
        v.rd = 0.65
        v.tl = 0.0
        v.ah_rel = -30.0
        v.jitter = 0.006
        v.shimmer = 0.30
        v.flutter = 25.0
        v.creak = True
    elif id == FV_ACE_TENOR:
        v.shelf_db = 10.0
        v.presence_db = 9.0
        # young, fast, bright; mean 148 Hz
        v.f0_base = 136.0
        v.range_st = 6.5
        v.rate_wpm = 166.0
        v.set_scales(1.08, 1.09, 1.08, 1.07, 1.06)
        v.rd = 0.7
        v.tl = 0.0
        v.ah_rel = -28.0
        v.jitter = 0.004
        v.shimmer = 0.20
        v.flutter = 20.0
        v.f0_floor = 80.0
    elif id == FV_ACE_CRIMSON:
        v.shelf_db = 10.0
        v.presence_db = 9.0
        # pressed, confident warrior, GB; mean 196 Hz
        v.f0_base = 178.0
        v.range_st = 7.0
        v.rate_wpm = 156.0
        v.set_scales(1.15, 1.17, 1.13, 1.10, 1.08)
        v.bw1 = 1.15
        v.rd = 0.75
        v.tl = 0.0
        v.ah_rel = -26.0
        v.jitter = 0.004
        v.shimmer = 0.20
        v.flutter = 15.0
        v.df1 = 30.0
        v.db1 = 100.0
        v.british = True
        v.f0_floor = 110.0
    return v^


def voice_eq(var pcm: List[Float32], voice: FormantVoice) -> List[Float32]:
    """The preset's output EQ (identity when both gains are 0): a first-order high shelf and an RBJ
    peaking biquad at 2.8 kHz, then a gain trim so the peak stays <= 0.98."""
    if voice.shelf_db == Float32(0.0) and voice.presence_db == Float32(0.0):
        return pcm^
    var sr = Float64(24000.0)
    # high shelf: y = x + g * hp(x), hp = one-pole high-pass at 1.6 kHz
    var g = Float64(det_exp(Float32(voice.shelf_db) * Float32(0.11512925))) - 1.0
    var a = Float64(det_exp(Float32(-6.283185307 * 1600.0 / 24000.0)))
    var lp = Float64(0.0)
    # peaking biquad (RBJ cookbook)
    var A = Float64(det_exp(Float32(voice.presence_db) * Float32(0.057564627)))
    var w0 = 6.283185307179586 * 2800.0 / sr
    var cw = Float64(det_cos(Float32(w0)))
    var sw = Float64(det_sin(Float32(w0)))
    var alpha = sw / (2.0 * 1.2)
    var a0 = 1.0 + alpha / A
    var b0 = (1.0 + alpha * A) / a0
    var b1 = (-2.0 * cw) / a0
    var b2 = (1.0 - alpha * A) / a0
    var a1 = (-2.0 * cw) / a0
    var a2 = (1.0 - alpha / A) / a0
    var x1 = Float64(0.0)
    var x2 = Float64(0.0)
    var y1 = Float64(0.0)
    var y2 = Float64(0.0)
    var pk = Float32(0.0)
    for i in range(len(pcm)):
        var x = Float64(pcm[i])
        lp = (1.0 - a) * x + a * lp
        var xs = x + g * (x - lp)
        var y = b0 * xs + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = xs
        y2 = y1
        y1 = y
        pcm[i] = Float32(y)
        pk = max(pk, abs(Float32(y)))
    if pk > Float32(0.98):
        var k = Float32(0.98) / pk
        for i in range(len(pcm)):
            pcm[i] *= k
    return pcm^
