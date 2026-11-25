"""Moonshine speech-to-text CLI: WAV files in, transcripts out.

usage: mojo run -I src examples/moonshine/main.mojo MODEL_DIR a.wav [b.wav ...] [--cpu] [--bench N]

MODEL_DIR is a Hugging Face `moonshine-ai/moonshine-streaming-{tiny,small,medium}` folder
(`config.json`, `model.safetensors`, `tokenizer.json`); weights are not part of Flint, see
`licenses/moonshine-LICENSE.txt`. Any WAV rate / channel count / PCM or float format works: the
audio is downmixed and resampled to 16 kHz in Mojo. `--bench N` repeats each file N times and
reports the median encoder / decoder time and real-time factor.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.ml.moonshine import Moonshine
from flint.ml.moonshine_audio import read_wav, to_model_rate


def _median(var xs: List[Float64]) -> Float64:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2] if len(xs) > 0 else 0.0


def main() raises:
    var a = argv()
    if len(a) < 3:
        print("usage: moonshine MODEL_DIR a.wav [b.wav ...] [--cpu] [--bench N]")
        return
    var gpu = True
    var bench = 0
    var files = List[String]()
    var i = 2
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--bench" and i + 1 < len(a):
            bench = Int(String(a[i + 1]))
            i += 1
        else:
            files.append(s)
        i += 1
    var t0 = perf_counter_ns()
    var m = Moonshine.load(String(a[1]), gpu)
    print("[moonshine] loaded", String(a[1]), "on", m.device(), "in", Float64(perf_counter_ns() - t0) / 1e6, "ms")
    for f in files:
        var clip = read_wav(f)
        if not clip.valid:
            print("[moonshine] cannot read", f)
            continue
        var x = to_model_rate(clip)
        var r = m.transcribe(x)   # the first run also uploads weights to the GPU
        print(f, "(" + String(clip.rate) + " Hz, " + String(r.audio_seconds) + " s):", r.text)
        if bench > 0:
            var enc = List[Float64]()
            var dec = List[Float64]()
            var tot = List[Float64]()
            for _ in range(bench):
                var b = m.transcribe(x)
                enc.append(b.encode_ms)
                dec.append(b.decode_ms)
                tot.append(b.total_ms())
            var med = _median(tot^)
            print("  bench x" + String(bench) + ": encode", _median(enc^), "ms, decode", _median(dec^), "ms (",
                  r.tokens, "tokens ), total", med, "ms, RTF", med / 1000.0 / r.audio_seconds)
