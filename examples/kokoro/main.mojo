"""Kokoro-82M text-to-speech CLI: text in, 24 kHz .wav or .qoa out.

usage: kokoro KOKORO_DIR "text" out.wav|out.qoa [--voice af_heart] [--speed 1.0] [--cpu]
              [--phonemes] [--timings] [--raw "misaki phonemes"]

KOKORO_DIR is laid out by `tools/kokoro/fetch.sh` (model.onnx, voices/*.bin, lexicon/*.json).
--phonemes prints the phonemiser output, --timings the per-phoneme start / end / viseme track,
--raw speaks a phoneme string as given (skips the phonemiser).
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.ml.kokoro import KokoroTTS, KokoroSpeech, write_wav_file, write_qoa_file, VISEME_NAMES
from flint.text.utf8 import encode_utf8


def main() raises:
    var a = argv()
    if len(a) < 4:
        print('usage: kokoro KOKORO_DIR "text" out.wav|out.qoa [--voice af_heart] [--speed 1.0] [--cpu] [--phonemes] [--timings] [--raw PHONEMES]')
        return
    var voice = String("af_heart")
    var speed = 1.0
    var gpu = True
    var show_ps = False
    var show_t = False
    var raw = String()
    var i = 4
    while i < len(a):
        var s = String(a[i])
        if s == "--voice" and i + 1 < len(a):
            voice = String(a[i + 1])
            i += 1
        elif s == "--speed" and i + 1 < len(a):
            speed = Float64(String(a[i + 1]))
            i += 1
        elif s == "--raw" and i + 1 < len(a):
            raw = String(a[i + 1])
            i += 1
        elif s == "--cpu":
            gpu = False
        elif s == "--phonemes":
            show_ps = True
        elif s == "--timings":
            show_t = True
        i += 1
    var t0 = perf_counter_ns()
    var tts = KokoroTTS.load(String(a[1]), gpu)
    print("[kokoro] loaded on", tts.device(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s; lexicon",
          tts.g2p.lexicon_size(), "entries")
    var speech = tts.speak_phonemes(raw, voice, speed) if raw.byte_length() > 0 else tts.say(String(a[2]), voice, speed)
    if show_ps:
        print("[kokoro] phonemes:", speech.phonemes)
    print("[kokoro]", speech.duration(), "s of audio in", speech.seconds, "s (RTF", speech.realtime_factor(), ")")
    if show_t:
        var names = String(VISEME_NAMES).split(" ")
        for t in speech.timings:
            var w = speech.words[t.word].text if t.word >= 0 and t.word < len(speech.words) else String("")
            print("  ", t.start, t.end, encode_utf8([t.phoneme]), String(names[t.viseme]), w)
    var out = String(a[3])
    var ok = write_qoa_file(out, speech.samples) if out.endswith(".qoa") else write_wav_file(out, speech.samples)
    if not ok:
        raise Error("could not write " + out)
    print("[kokoro] wrote", out)
