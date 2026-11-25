"""Flint v5 Moonshine speech-to-text suite (`test_flint_v5_moonshine.mojo`).

Verifies, without model files:
  1. WAV decoding: 16-bit stereo is downmixed, IEEE float32 mono is read exactly.
  2. `Resampler`: 48 kHz and 24 kHz tones land at 16 kHz with the right amplitude and phase;
     a tone above the new Nyquist is suppressed; output lengths follow the rate ratio.
  3. `moonshine_frames`: per-frame CMVN + asinh(k x) on a known frame.
  4. `MoonshineTokenizer.decode`: word-boundary pieces, byte fallback, special tokens,
     invalid UTF-8 replaced.
  5. `EnergyVad` cuts one utterance out of silence / tone / silence; `VoiceCommands` matching.
  6. With `MOONSHINE_DIR` (a `moonshine-streaming-*` folder) and `MOONSHINE_WAV`: a transcript
     on the GPU when present and on the CPU, identical between the two; with `MOONSHINE_EXPECT`
     it must equal that text. With `MOONSHINE_REF` (prefix written by
     `tools/moonshine/reference.py`): encoder states and teacher-forced logits within 1e-3 of
     the `transformers` reference.
"""
from std.math import sin, sqrt, abs, asinh
from std.testing import assert_true, assert_equal
from flint.platform.os import env_now, read_bytes
from flint.assets.json import parse_json
from flint.ml.tensor import fptr_at
from flint.ml.moonshine_audio import (
    decode_wav_mono, Resampler, moonshine_frames, EnergyVad, to_model_rate, read_wav,
)
from flint.ml.moonshine_tokenizer import MoonshineTokenizer
from flint.ml.moonshine import Moonshine, VoiceCommands, command_words

comptime PI: Float64 = 3.141592653589793


def _put16(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 255))
    b.append(UInt8((v >> 8) & 255))


def _put32(mut b: List[UInt8], v: Int):
    for i in range(4):
        b.append(UInt8((v >> (8 * i)) & 255))


def _wav(fmt: Int, channels: Int, rate: Int, bits: Int, payload: List[UInt8]) -> List[UInt8]:
    var b = List[UInt8]()
    for c in "RIFF".as_bytes():
        b.append(c)
    _put32(b, 36 + len(payload))
    for c in "WAVEfmt ".as_bytes():
        b.append(c)
    _put32(b, 16)
    _put16(b, fmt)
    _put16(b, channels)
    _put32(b, rate)
    _put32(b, rate * channels * bits // 8)
    _put16(b, channels * bits // 8)
    _put16(b, bits)
    for c in "data".as_bytes():
        b.append(c)
    _put32(b, len(payload))
    for v in payload:
        b.append(v)
    return b^


def test_wav() raises:
    var p = List[UInt8]()
    _put16(p, 16384)       # L = 0.5
    _put16(p, 65536 - 16384)  # R = -0.5
    _put16(p, 32767)
    _put16(p, 0)
    var clip = decode_wav_mono(_wav(1, 2, 44100, 16, p))
    assert_true(clip.valid, "16-bit stereo WAV")
    assert_equal(clip.rate, 44100)
    assert_equal(len(clip.samples), 2)
    assert_true(abs(clip.samples[0]) < 1e-7, "stereo downmix")
    assert_true(abs(clip.samples[1] - 0.49998) < 1e-4)
    var f = List[UInt8]()
    var vals = List[Float32]()
    vals.append(0.25)
    vals.append(-0.75)
    for v in vals:
        var bits = Int(fptr_at(Int(Pointer(to=v))).unsafe_load().to_bits())
        _put32(f, bits)
    var fc = decode_wav_mono(_wav(3, 1, 16000, 32, f))
    assert_true(fc.valid, "float WAV")
    assert_equal(len(fc.samples), 2)
    assert_true(fc.samples[0] == 0.25 and fc.samples[1] == -0.75, "float samples exact")
    assert_true(not decode_wav_mono(List[UInt8](length=20, fill=0)).valid, "garbage rejected")
    print("  WAV decode: ok")


def _tone(rate: Int, hz: Float64, n: Int) -> List[Float32]:
    var x = List[Float32](length=n, fill=0.0)
    for i in range(n):
        x[i] = Float32(sin(2 * PI * hz * Float64(i) / Float64(rate)))
    return x^


def test_resampler() raises:
    for fin in [48000, 24000, 44100]:
        var r = Resampler(fin, 16000)
        var n = fin   # one second
        var y = r.process(_tone(fin, 440.0, n))
        assert_equal(len(y), 16000)
        var err: Float64 = 0
        for i in range(2000, 14000):   # away from the zero-padded ends
            err = max(err, abs(Float64(y[i]) - sin(2 * PI * 440.0 * Float64(i) / 16000.0)))
        assert_true(err < 2e-3, "440 Hz tone from " + String(fin) + " Hz, max error " + String(err))
    var r48 = Resampler(48000, 16000)
    var alias_y = r48.process(_tone(48000, 9500.0, 48000))
    var peak: Float64 = 0
    for i in range(2000, 14000):
        peak = max(peak, abs(Float64(alias_y[i])))
    assert_true(peak < 1e-3, "9.5 kHz must not alias into 16 kHz output, peak " + String(peak))
    var same = Resampler(16000, 16000)
    assert_equal(len(same.process(_tone(16000, 100.0, 1234))), 1234)
    print("  resampler: ok")


def test_frames() raises:
    var x = List[Float32](length=160, fill=0.0)
    for i in range(80):
        x[i] = Float32(i % 2) * 2.0 + 3.0   # 3, 5, 3, 5 ... -> CMVN gives -1, +1
    var f = moonshine_frames(x, 2, 0.75)
    var want = Float32(asinh(0.75 * (1.0 / sqrt(1.0 + 1e-6))))
    assert_true(abs(f[1] - want) < 1e-6 and abs(f[0] + want) < 1e-6, "CMVN + asinh")
    assert_true(abs(f[80]) < 1e-6, "silent frame stays zero")
    print("  frame features: ok")


def test_tokenizer() raises:
    var t = MoonshineTokenizer()
    for p in ["<unk>", "<s>", "</s>", "▁Open", "▁the", "▁gate", ".", "<0xC3>", "<0xA9>", "<0xFF>"]:
        t.pieces.append(String(p))
        t.byte_value.append(-1)
        t.special.append(False)
    for k in range(3):
        t.special[k] = True
    t.byte_value[7] = 0xC3
    t.byte_value[8] = 0xA9
    t.byte_value[9] = 0xFF
    var ids: List[Int] = [1, 3, 4, 5, 6, 2]
    assert_equal(t.decode(ids), "Open the gate.")
    var acc: List[Int] = [4, 7, 8]
    assert_equal(t.decode(acc), "theé")
    var bad: List[Int] = [3, 9]
    assert_equal(t.decode(bad), "Open�")
    print("  tokenizer decode: ok")


def test_vad_and_commands() raises:
    var vad = EnergyVad()
    var silence = List[Float32](length=16000, fill=0.0)
    for i in range(len(silence)):
        silence[i] = Float32(((i * 7919) % 200) - 100) * 1e-5   # faint noise floor
    var ended = vad.push(silence)
    assert_true(not ended and not vad.active)
    var speech = _tone(16000, 300.0, 16000)
    for i in range(len(speech)):
        speech[i] *= 0.3
    ended = vad.push(speech)
    assert_true(vad.active and not ended, "speech opens the gate")
    ended = vad.push(silence)
    assert_true(ended, "silence closes it")
    var secs = Float64(len(vad.utterance)) / 16000.0
    assert_true(secs > 1.0 and secs < 1.8, "utterance length " + String(secs))
    var cmds = VoiceCommands()
    _ = cmds.add("open the gate")
    _ = cmds.add("move the camera five meters left")
    _ = cmds.add("save the level")
    assert_equal(cmds.match("Open the gate!"), 0)
    assert_equal(cmds.match("Move the camera 5 meters left."), 1)
    assert_equal(cmds.match("Save level."), 2)
    assert_equal(cmds.match("What a lovely day."), -1)
    assert_equal(len(command_words("It's 12 o'clock.")), 3)
    print("  VAD + voice commands: ok")


def _read_f32(path: String) -> List[Float32]:
    var b = read_bytes(path)
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=0.0)
    var p = fptr_at(Int(b.unsafe_ptr()))
    for i in range(n):
        out[i] = p.unsafe_offset(i).unsafe_load[alignment=1]()
    return out^


def test_model() raises:
    var dir = env_now("MOONSHINE_DIR")
    if dir.byte_length() == 0:
        print("  model run: MOONSHINE_DIR not set, skipped")
        return
    var wav = env_now("MOONSHINE_WAV")
    if wav.byte_length() > 0:
        var clip = read_wav(wav)
        assert_true(clip.valid, "cannot read " + wav)
        var x = to_model_rate(clip)
        var cpu = Moonshine.load(dir, False)
        var rc = cpu.transcribe(x)
        print("  CPU:", rc.text, "(" + String(Int(rc.total_ms())) + " ms, RTF " + String(rc.rtf()) + ")")
        var gpu = Moonshine.load(dir, True)
        _ = gpu.transcribe(x)
        var rg = gpu.transcribe(x)
        print("  " + gpu.device() + ":", rg.text, "(" + String(Int(rg.total_ms())) + " ms, RTF " + String(rg.rtf()) + ")")
        assert_true(rc.ids == rg.ids, "CPU and GPU transcripts differ")
        assert_true(rc.text.byte_length() > 0, "empty transcript")
        var expect = env_now("MOONSHINE_EXPECT")
        if expect.byte_length() > 0:
            assert_equal(rc.text, expect)
    var refp = env_now("MOONSHINE_REF")
    if refp.byte_length() > 0:
        var meta = parse_json(read_bytes(refp + ".json"))
        var node = meta.object_get(meta.root, "ids")
        var ids = List[Int]()
        for i in range(meta.array_size(node)):
            ids.append(meta.integer(meta.at(node, i)))
        var x = _read_f32(refp + ".wav16.f32")
        var enc_ref = _read_f32(refp + ".enc.f32")
        var lg_ref = _read_f32(refp + ".logits.f32")
        for use_gpu in [False, True]:
            var m = Moonshine.load(dir, use_gpu)
            m.encode(x)
            assert_equal(len(m.enc), len(enc_ref))
            var de: Float32 = 0
            for i in range(len(enc_ref)):
                de = max(de, abs(m.enc[i] - enc_ref[i]))
            m.start_decoder(len(ids))
            var dl: Float32 = 0
            var V = m.cfg.vocab
            for s in range(len(ids) - 1):
                m.step(ids[s], s)
                for i in range(V):
                    dl = max(dl, abs(m.logits[i] - lg_ref[s * V + i]))
                assert_equal(m.argmax(), ids[s + 1])
            print("  reference parity on", m.device(), ": encoder", de, "logits", dl)
            assert_true(de < 1e-3 and dl < 1e-3, "drift from the transformers reference")
            var t = m.transcribe(x)
            assert_true(t.ids == ids, "greedy ids differ from the reference")


def main() raises:
    print("=== Flint v5 Moonshine Suite ===")
    test_wav()
    test_resampler()
    test_frames()
    test_tokenizer()
    test_vad_and_commands()
    test_model()
    print("=== ALL FLINT v5 MOONSHINE TESTS PASSED ===")
