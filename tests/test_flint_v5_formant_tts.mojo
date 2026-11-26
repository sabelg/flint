"""Flint v5 formant TTS suite (`test_flint_v5_formant_tts.mojo`).

No model or lexicon files needed (rules-only G2P):
  1. Determinism: the same text renders bit-identical PCM twice.
  2. Sanity: no NaN / Inf, peak <= 1, a Harvard sentence lasts a plausible 1.5-4.5 s.
  3. Acoustics: /s/ carries far more high-band energy than a vowel.
  4. Intonation: a yes/no question ends rising, a declarative ends falling (F0 track).
  5. Coverage: every misaki phoneme symbol renders finite, non-silent audio in every preset.
  6. Preset EQ: `voice_eq` is the identity at 0 dB and lifts a 2.8 kHz tone far more than a 300 Hz one.
  7. `flint.platform.speech`: AI OS and the four radio profiles still produce 48 kHz stereo clips.
Prints the real-time factor of the dry synthesiser.
"""
from std.math import abs, isnan, isinf, sin
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from flint.audio.formant_tts.api import FormantTTS, word_flags, FORMANT_TTS_SAMPLE_RATE
from flint.audio.formant_tts.voices import (
    formant_voice, FV_DEFAULT, FV_AI_OS, FV_BRIDGE_FEMALE, FV_ACE_BARITONE, FV_ACE_TENOR, FV_ACE_CRIMSON,
    voice_eq,
)
from flint.audio.formant_tts.synth import KlattTrack, P_F0, P_AV
from flint.platform.speech import (
    synth_ai_copilot_clip,
    synth_radio_comms_clip,
    VOICE_AI_OS,
    VOICE_BRIDGE_FEMALE,
    VOICE_ACE_BARITONE,
    VOICE_ACE_TENOR,
    VOICE_ACE_CRIMSON,
)


def _hash(x: List[Float32]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(x)):
        var q = Int64(x[i] * Float32(1048576.0))
        h = (h ^ UInt64(q)) * UInt64(0x100000001B3)
    return h


def _check_finite(x: List[Float32]) raises -> Float32:
    var pk = Float32(0.0)
    for v in x:
        assert_true(not isnan(v) and not isinf(v), "non-finite sample")
        pk = max(pk, abs(v))
    assert_true(pk <= Float32(1.0), "peak above full scale")
    return pk


def _hf_fraction(x: List[Float32]) -> Float64:
    """Energy of the first difference over the signal energy: ~0 for a low vowel, ~2 for white noise."""
    var e = Float64(0.0)
    var d = Float64(0.0)
    for i in range(1, len(x)):
        e += Float64(x[i]) * Float64(x[i])
        var df = Float64(x[i] - x[i - 1])
        d += df * df
    return d / max(Float64(1e-20), e)


def _final_f0_move(tr: KlattTrack) -> Float32:
    """F0 at the last voiced frame minus the median-ish F0 of the 300 ms before the last 150 ms."""
    var last = -1
    for k in range(tr.nframes):
        if tr.get(k, P_AV) > Float32(0.0):
            last = k
    var f_end = tr.get(last, P_F0)
    var acc = Float32(0.0)
    var cnt = 0
    for k in range(max(0, last - 90), max(0, last - 30)):
        if tr.get(k, P_AV) > Float32(0.0):
            acc += tr.get(k, P_F0)
            cnt += 1
    return f_end - acc / Float32(max(1, cnt))


def test_determinism_and_sanity() raises:
    var tts = FormantTTS()
    var v = formant_voice(FV_DEFAULT)
    var text = String("The birch canoe slid on the smooth planks.")
    var a = tts.synthesize(text, v)
    var b = tts.synthesize(text, v)
    assert_equal(len(a), len(b))
    assert_equal(_hash(a), _hash(b))
    var pk = _check_finite(a)
    assert_true(pk > Float32(0.05), "near-silent output")
    var sec = Float64(len(a)) / Float64(FORMANT_TTS_SAMPLE_RATE)
    assert_true(sec > 1.5 and sec < 4.5, "implausible sentence duration")
    print("✓ determinism / finite / peak", pk, "/ duration", sec, "s")


def test_fricative_spectrum() raises:
    var tts = FormantTTS()
    var v = formant_voice(FV_DEFAULT)
    var s = tts.synthesize("/s/", v)
    var a = tts.synthesize("/ˈɑ/", v)
    var hs = _hf_fraction(s)
    var ha = _hf_fraction(a)
    assert_true(hs > 10.0 * ha, "/s/ is not brighter than /ɑ/")
    print("✓ /s/ high-band fraction", hs, "vs /ɑ/", ha)


def test_intonation() raises:
    var tts = FormantTTS()
    var v = formant_voice(FV_DEFAULT)
    var rq = tts.phonemize("Is the main gun charged?")
    var tq = tts.track(rq.phonemes, word_flags(rq), v)
    var rd = tts.phonemize("The main gun is charged.")
    var td = tts.track(rd.phonemes, word_flags(rd), v)
    var mq = _final_f0_move(tq)
    var md = _final_f0_move(td)
    assert_true(mq > Float32(5.0), "yes/no question does not rise")
    assert_true(md < Float32(-3.0), "declarative does not fall")
    print("✓ final F0 movement: question", mq, "Hz, declarative", md, "Hz")


def test_phoneme_coverage() raises:
    var tts = FormantTTS()
    var syms: List[String] = [
        "i", "ɪ", "ᵻ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "ʌ", "o", "ʊ", "u", "ɝ", "ɚ", "ɜ", "ə", "ɐ", "ᵊ",
        "A", "I", "O", "W", "Y", "Q", "w", "j", "ɹ", "r", "l", "h", "m", "n", "ŋ", "f", "v", "θ", "ð", "s",
        "z", "ʃ", "ʒ", "ʧ", "ʤ", "p", "b", "t", "d", "k", "ɡ", "g", "T", "ɾ", "ʔ",
    ]
    var ids: List[Int] = [FV_DEFAULT, FV_AI_OS, FV_BRIDGE_FEMALE, FV_ACE_BARITONE, FV_ACE_TENOR, FV_ACE_CRIMSON]
    var n = 0
    for vid in ids:
        var v = formant_voice(vid)
        for s in syms:
            var x = tts.synthesize("/ˈɑ" + s + "ə/", v)
            var pk = _check_finite(x)
            assert_true(pk > Float32(0.01), "silent phoneme " + s)
            n += 1
    print("✓", len(syms), "misaki phonemes x", len(ids), "presets render (", n, "clips )")


def _tone(hz: Float64) -> List[Float32]:
    var o = List[Float32]()
    for i in range(4800):
        o.append(Float32(0.1 * sin(6.283185307179586 * hz * Float64(i) / 24000.0)))
    return o^


def _rms_tail(x: List[Float32]) -> Float64:
    var e = Float64(0.0)
    for i in range(2400, len(x)):
        e += Float64(x[i]) * Float64(x[i])
    return e / Float64(len(x) - 2400)


def test_voice_eq() raises:
    var flat = formant_voice(FV_DEFAULT)
    var t = _tone(1000.0)
    var y = voice_eq(t.copy(), flat)
    for i in range(len(t)):
        assert_equal(y[i], t[i])
    var v = formant_voice(FV_ACE_TENOR)
    assert_true(v.presence_db > 0.0, "radio preset without presence boost")
    var lo = _rms_tail(voice_eq(_tone(300.0), v)) / _rms_tail(_tone(300.0))
    var hi = _rms_tail(voice_eq(_tone(2800.0), v)) / _rms_tail(_tone(2800.0))
    assert_true(hi > 4.0 * lo, "preset EQ does not lift the 2-4 kHz band")
    print("✓ preset EQ power gain 300 Hz", lo, "/ 2.8 kHz", hi)


def test_speech_clips() raises:
    var ai = synth_ai_copilot_clip("Guidance online. Turn left.")
    assert_true(ai.valid)
    assert_equal(ai.channels, 2)
    assert_equal(ai.sample_rate, 48000)
    assert_true(len(ai.samples) > 12000)
    var ids: List[Int] = [VOICE_AI_OS, VOICE_BRIDGE_FEMALE, VOICE_ACE_BARITONE, VOICE_ACE_TENOR, VOICE_ACE_CRIMSON]
    for p in ids:
        var c = synth_radio_comms_clip("Roger that, Skull Leader.", p)
        assert_true(c.valid)
        assert_equal(c.channels, 2)
        assert_equal(c.sample_rate, 48000)
        assert_true(len(c.samples) > 12000)
    print("✓ speech.mojo AI OS + 4 radio profiles")


def test_real_time_factor() raises:
    var tts = FormantTTS()
    var v = formant_voice(FV_DEFAULT)
    var t0 = perf_counter_ns()
    var total = 0
    var lines: List[String] = [
        "The birch canoe slid on the smooth planks.",
        "Glue the sheet to the dark blue background.",
        "Bridge to all squadrons, condition yellow.",
        "Vector two seven zero, angels fifteen.",
    ]
    for l in lines:
        total += len(tts.synthesize(l, v))
    var dt = Float64(perf_counter_ns() - t0) * 1e-9
    var rtf = Float64(total) / Float64(FORMANT_TTS_SAMPLE_RATE) / max(1e-9, dt)
    assert_true(rtf > 10.0, "formant TTS slower than 10x real time")
    print("✓ real-time factor", rtf, "x")


def main() raises:
    test_determinism_and_sanity()
    test_fricative_spectrum()
    test_intonation()
    test_phoneme_coverage()
    test_voice_eq()
    test_speech_clips()
    test_real_time_factor()
    print("test_flint_v5_formant_tts: all passed")
