"""Flint v5 Kokoro TTS suite (`test_flint_v5_kokoro.mojo`).

Always (no model or lexicon files):
  1. Number words: cardinals without "and", years (num2words `to_year`), ordinals.
  2. `apply_stress`: add / demote / promote / strip stress like misaki.
  3. G2P lexicon logic on a tiny in-memory lexicon: case variants, -s / -ed / -ing, "the" / "to"
     before a vowel, compounds, currency, inline `[word](/phonemes/)`, word spans.
  4. Letter-to-sound rules for words the lexicon lacks.
  5. The ONNX kernels Kokoro added: Round, Pad (reflect), ConvTranspose, STFT (exact DC /
     Nyquist zeros), linear Resize, ReduceMean, NonZero, LSTM, strided Conv.
  6. Visemes and the Kokoro vocabulary.
With `FLINT_KOKORO_DIR` (laid out by tools/kokoro/fetch.sh): a full synthesis with timings.
"""
from std.math import abs, sqrt, tanh, isnan, isinf
from std.testing import assert_true, assert_equal
from flint.ml.phonemize import G2P, apply_stress, cardinal_words, year_words, ordinal_words, NO_STRESS
from flint.ml.ops_ext import (
    round_even, pad, conv_transpose1d, stft, resize_linear, reduce_mean, nonzero, lstm, conv1d_ext, conv1d_par,
)
from flint.ml.tensor import Tensor
from flint.ml.kokoro import KokoroTTS, kokoro_vocab, viseme_of, to_pcm16, KOKORO_SAMPLE_RATE
from flint.audio.qoa import encode_qoa, decode_qoa
from flint.core.jobs import JobPool
from flint.platform.os import env_now


def _join(w: List[String]) -> String:
    var s = String()
    for i in range(len(w)):
        if i > 0:
            s += " "
        s += w[i]
    return s


def test_numbers() raises:
    assert_equal(_join(cardinal_words(0)), "zero")
    assert_equal(_join(cardinal_words(25)), "twenty five")
    assert_equal(_join(cardinal_words(1200)), "one thousand two hundred")
    assert_equal(_join(cardinal_words(1000005)), "one million five")
    assert_equal(_join(cardinal_words(-42)), "minus forty two")
    assert_equal(_join(year_words(1987)), "nineteen eighty seven")
    assert_equal(_join(year_words(1905)), "nineteen oh five")
    assert_equal(_join(year_words(1900)), "nineteen hundred")
    assert_equal(_join(year_words(2005)), "two thousand five")
    assert_equal(_join(year_words(2024)), "twenty twenty four")
    assert_equal(_join(ordinal_words(21)), "twenty first")
    assert_equal(_join(ordinal_words(12)), "twelfth")
    assert_equal(_join(ordinal_words(40)), "fortieth")
    print("✓ number words")


def test_stress() raises:
    assert_equal(apply_stress("ðɪs", 0.5), "ðˌɪs")
    assert_equal(apply_stress("kæt", 2.0), "kˈæt")
    assert_equal(apply_stress("həlˈO", -1.0), "həlˌO")
    assert_equal(apply_stress("nˈItfˌɔl", -2.5), "nItfɔl")
    assert_equal(apply_stress("dˌu", 1.0), "dˈu")
    assert_equal(apply_stress("həlˈO", NO_STRESS), "həlˈO")
    print("✓ apply_stress")


def _mini() -> G2P:
    var g = G2P()
    g.add_word("cat", "kˈæt")
    g.add_word("walk", "wˈɔk")
    g.add_word("hope", "hˈOp")
    g.add_word("the", "ði")
    g.add_word("to", "tu")
    g.add_word("apple", "ˈæpᵊl")
    g.add_word("night", "nˈIt")
    g.add_word("fall", "fˈɔl")
    g.add_word("hello", "həlˈO")
    g.add_word("three", "θɹˈi")
    g.add_word("five", "fˈIv")
    g.add_word("one", "wˈʌn")
    g.add_word("dollar", "dˈɑləɹ")
    g.add_word("and", "ænd")
    g.add_word("cent", "sˈɛnt")
    g.add_word("wait", "wˈAt")
    g.add_word("F", "ˈɛf")
    g.add_word("B", "bˈi")
    g.add_word("I", "ˈI")
    return g^


def test_g2p_lexicon() raises:
    var g = _mini()
    var r = g.phonemize("The cat walked to the apple.")
    assert_equal(r.phonemes, "ðə kˈæt wˈɔkt tə ði ˈæpᵊl.")
    assert_equal(len(r.words), 6)
    assert_equal(r.words[1].text, "cat")
    assert_equal(r.words[1].phonemes, "kˈæt")
    assert_equal(r.word_at(r.words[1].start), 1)
    assert_equal(r.word_at(2), -1)                       # the space after "ðə"
    assert_equal(g.phonemize("Hello, cats!").phonemes, "həlˈO, kˈæts!")
    assert_equal(g.phonemize("hoping waited").phonemes, "hˈOpɪŋ wˈATᵻd")   # flap: ɾ -> T
    assert_equal(g.phonemize("nightfall").phonemes, "nˈItfˌɔl")
    assert_equal(g.phonemize("FBI").phonemes, "ˌɛfbˌiˈI")
    assert_equal(g.phonemize("3 cats").phonemes, "θɹˈi kˈæts")
    assert_equal(g.phonemize("$5.01").phonemes, "fˈIv dˈɑləɹz ænd wˈʌn sˈɛnt")
    assert_equal(g.phonemize("[Zorblax](/zˈɔɹblæks/) walks").phonemes, "zˈɔɹblæks wˈɔks")
    assert_equal(g.phonemize("\"Hello\"").phonemes, "“həlˈO”")
    var u = g.phonemize("Zorblax")
    assert_true(not u.words[0].known and u.phonemes.byte_length() > 0, "unknown words are spelled by rule")
    print("✓ G2P lexicon logic")


def test_letter_rules() raises:
    var g = G2P()
    assert_equal(g.spell("cat"), "kˈæt")
    assert_equal(g.spell("make"), "mˈAk")
    assert_equal(g.spell("light"), "lˈIt")
    assert_equal(g.spell("brick"), "bɹˈɪk")
    assert_equal(g.spell("nation"), "nˈAʃən")
    var z = g.spell("Zorblax")
    assert_true(z.startswith("zˈɔɹbl"), "Zorblax -> " + z)
    print("✓ letter-to-sound rules")


def _close(a: Float32, b: Float64, tol: Float64 = 1e-5) -> Bool:
    return abs(Float64(a) - b) <= tol


def test_kernels() raises:
    var pool = JobPool()
    _ = pool.start(4)
    # Round: halves to even
    var rr = round_even(Tensor.floats([5], [Float32(0.5), 1.5, 2.5, -0.5, 2.6]))
    assert_true(_close(rr.f[0], 0) and _close(rr.f[1], 2) and _close(rr.f[2], 2) and _close(rr.f[3], 0) and _close(rr.f[4], 3))
    # Pad reflect: [1 2 3] by 2 on both sides -> 3 2 1 2 3 2 1
    var p = pad(Tensor.floats([1, 3], [Float32(1), 2, 3]), [0, 2, 0, 2], List[Int](), "reflect", 0.0)
    var want: List[Float32] = [3, 2, 1, 2, 3, 2, 1]
    assert_equal(p.shape[1], 7)
    for i in range(7):
        assert_true(_close(p.f[i], Float64(want[i])))
    # ConvTranspose: x [1 2], w [1 1 1], stride 2 -> [1 1 3 2 2] (+ bias 0.5)
    var ct = conv_transpose1d(pool, Tensor.floats([1, 1, 2], [Float32(1), 2]), Tensor.floats([1, 1, 3], [Float32(1), 1, 1]),
                              Tensor.floats([1], [Float32(0.5)]), True, 0, 0, 1, 2, 0, 1)
    var cw: List[Float32] = [1.5, 1.5, 3.5, 2.5, 2.5]
    assert_equal(ct.shape[2], 5)
    for i in range(5):
        assert_true(_close(ct.f[i], Float64(cw[i])))
    # strided Conv: x 0..7, w [1 1], stride 2 -> pairwise sums
    var xs = List[Float32]()
    for i in range(8):
        xs.append(Float32(i))
    var sc = conv1d_ext(pool, Tensor.floats([1, 1, 8], xs.copy()), Tensor.floats([1, 1, 2], [Float32(1), 1]), Tensor(), False, 0, 0, 1, 2, 1)
    var sp = conv1d_par(pool, Tensor.floats([1, 1, 8], xs.copy()), Tensor.floats([1, 1, 2], [Float32(1), 1]), Tensor(), False, 0, 0, 1, 2, 1)
    assert_equal(sc.shape[2], 4)
    for i in range(4):
        assert_true(_close(sc.f[i], Float64(4 * i + 1)) and _close(sp.f[i], Float64(4 * i + 1)))
    # STFT of a constant: DC = sum(window), the rest 0; imaginary DC / Nyquist exactly zero
    var sig = List[Float32](length=16, fill=1.0)
    var win = List[Float32]()
    for i in range(8):
        win.append(Float32(0.5) + Float32(i) * 0.1)
    var st = stft(Tensor.floats([1, 16, 1], sig^), 4, Tensor.floats([8], win^), True, 0, True)
    assert_equal(st.shape[1], 3)
    assert_equal(st.shape[2], 5)
    assert_true(_close(st.f[0], 6.8))
    for f in range(3):
        assert_true(st.f[(f * 5 + 0) * 2 + 1] == 0 and st.f[(f * 5 + 4) * 2 + 1] == 0, "exact zero imaginary parts")
    # linear Resize, half_pixel, x2: [0 1] -> [0 .25 .75 1]
    var rl = resize_linear(Tensor.floats([1, 1, 2], [Float32(0), 1]), [1, 1, 4], [1.0, 1.0, 2.0], "half_pixel")
    var rw: List[Float32] = [0, 0.25, 0.75, 1]
    for i in range(4):
        assert_true(_close(rl.f[i], Float64(rw[i])))
    # ReduceMean over the last axis
    var rm = reduce_mean(Tensor.floats([2, 2], [Float32(1), 3, 5, 9]), [-1], True)
    assert_true(rm.shape[1] == 1 and _close(rm.f[0], 2) and _close(rm.f[1], 7))
    # NonZero
    var nz = nonzero(Tensor.floats([2, 2], [Float32(0), 3, 5, 0]))
    assert_true(nz.shape[0] == 2 and nz.shape[1] == 2 and nz.i[0] == 0 and nz.i[1] == 1 and nz.i[2] == 1 and nz.i[3] == 0)
    # LSTM, hidden 1: only the cell-gate input weight is 1; x = 1, h0 = c0 = 0
    #   c = sigmoid(0) * tanh(1), h = sigmoid(0) * tanh(c)
    var w = Tensor.floats([1, 4, 1], [Float32(0), 0, 0, 1])
    var rz = Tensor.floats([1, 4, 1], [Float32(0), 0, 0, 0])
    var res = lstm(pool, Tensor.floats([1, 1, 1], [Float32(1)]), w, rz, Tensor(), False, Tensor(), False, Tensor(), False,
                   1, "forward", 0)
    var c = 0.5 * tanh(1.0)
    assert_true(_close(res[0].f[0], 0.5 * tanh(c)) and _close(res[2].f[0], c))
    var bi = lstm(pool, Tensor.floats([3, 1, 1], [Float32(1), 0, 0]), Tensor.floats([2, 4, 1], [Float32(0), 0, 0, 1, 0, 0, 0, 1]),
                  Tensor.floats([2, 4, 1], List[Float32](length=8, fill=0.0)), Tensor(), False, Tensor(), False, Tensor(), False,
                  1, "bidirectional", 0)
    assert_true(bi[0].shape[0] == 3 and bi[0].shape[1] == 2, "Y is [T, 2, B, H]")
    # the reverse direction sees x = 1 last, at t = 0
    assert_true(_close(bi[0].f[1], 0.5 * tanh(c)) and _close(bi[0].f[0], 0.5 * tanh(c)))
    pool.stop()
    print("✓ ONNX kernels (Round, Pad, ConvTranspose, Conv stride, STFT, Resize, ReduceMean, NonZero, LSTM)")


def test_vocab_visemes() raises:
    var v = kokoro_vocab()
    assert_equal(len(v), 114)
    assert_equal(v[0x2C8], 156)     # ˈ
    assert_equal(v[0x20], 16)       # space
    assert_equal(viseme_of(112), 1)        # p -> PP
    assert_equal(viseme_of(0x3B8), 3)      # θ -> TH
    assert_equal(viseme_of(117), 14)       # u -> ou
    assert_equal(viseme_of(44), 0)         # , -> sil
    var pcm = to_pcm16([Float32(0), 0.5, -1.0, 2.0])
    assert_true(pcm[1] == 16384 and pcm[2] == -32767 and pcm[3] == 32767)
    print("✓ Kokoro vocabulary and visemes")


def test_model() raises:
    var dir = env_now("FLINT_KOKORO_DIR")
    if dir.byte_length() == 0:
        print("- Kokoro model run skipped (set FLINT_KOKORO_DIR)")
        return
    var tts = KokoroTTS.load(dir, True)
    assert_true(tts.patched_atan2, "the DC / Nyquist atan2 fix-up is applied")
    assert_true(tts.durations_name.byte_length() > 0, "duration tensor found")
    var s = tts.say("Hello world, this is Flint speaking.", "af_heart")
    if tts.g2p.lexicon_size() > 0:
        assert_equal(s.phonemes, "həlˈO wˈɜɹld, ðɪs ɪz flˈɪnt spˈikɪŋ.")
    var n = len(s.samples)
    assert_true(n > KOKORO_SAMPLE_RATE and n < 6 * KOKORO_SAMPLE_RATE, "about two to three seconds of audio")
    var e = 0.0
    for x in s.samples:
        assert_true(not isnan(x) and not isinf(x))
        e += Float64(x) * Float64(x)
    var rms = sqrt(e / Float64(n))
    assert_true(rms > 0.01 and rms < 0.5, "speech-level RMS")
    assert_true(len(s.timings) > 20)
    for i in range(1, len(s.timings)):
        assert_true(s.timings[i].start >= s.timings[i - 1].start and s.timings[i].end >= s.timings[i].start)
    assert_true(s.timings[len(s.timings) - 1].end <= s.duration() + 1e-6)
    var q = encode_qoa(to_pcm16(s.samples), 1, KOKORO_SAMPLE_RATE)
    var clip = decode_qoa(q)
    assert_equal(clip.frame_count(), n)
    print("✓ Kokoro synthesis on", tts.device(), ":", s.duration(), "s audio, RTF", s.realtime_factor())


def main() raises:
    test_numbers()
    test_stress()
    test_g2p_lexicon()
    test_letter_rules()
    test_kernels()
    test_vocab_visemes()
    test_model()
    print("test_flint_v5_kokoro: all passed")
