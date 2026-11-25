"""Kokoro-82M text-to-speech on `OnnxModel` (`flint.ml.kokoro`).

Kokoro (https://huggingface.co/hexgrad/Kokoro-82M, Apache-2.0 code / weights / voicepacks) is a
StyleTTS2-style model with an iSTFTNet vocoder: phoneme ids + a 256-d voice style -> 24 kHz audio.
Flint runs the onnx-community fp32 export (`onnx/model.onnx`) and the voicepacks as raw float32
`voices/<name>.bin` ([510, 1, 256]: one style row per input length). Text goes through the
pure-Mojo `flint.ml.phonemize.G2P` (misaki phoneme set), so no espeak-ng is involved.

    var tts = KokoroTTS.load("/path/to/kokoro")      # model.onnx, voices/, lexicon/ (tools/kokoro/fetch.sh)
    var out = tts.say("Welcome, traveller.", "af_heart")
    out.samples                                       # 24 kHz mono float32
    out.timings                                       # per phoneme: start / end seconds, word, viseme
    write_wav_file("line.wav", out.samples)          # or encode_qoa(to_pcm16(out.samples), 1, 24000)
    var asset = server.register_pcm16_asset(to_pcm16(out.samples), 1, KOKORO_SAMPLE_RATE)

Numerics: the export computes the source phase with atan(im / re) plus a sign fix-up that gives
-pi where im == 0 and re < 0 (the DC and Nyquist bins); PyTorch's `torch.atan2` (what Kokoro was
trained with) gives +pi there, and Kokoro's own `CustomSTFT` forces +pi for the same reason.
`KokoroTTS.load` therefore turns that one `Greater(im, 0)` into `GreaterOrEqual`, which restores
the PyTorch result (Flint's STFT gives those imaginary parts exactly +0).
"""

from std.collections import Dict
from std.memory import bitcast
from std.time import perf_counter_ns
from flint.ml.onnx import OnnxModel, OP_GREATER_EQ, OP_STFT, OP_ROUND, OP_CLIP
from flint.ml.tensor import Tensor, DT_I64
from flint.ml.phonemize import G2P, G2PResult, G2PWord
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.platform.os import read_bytes, write_bytes, file_size, list_directory
from flint.cycles.synth import encode_wav16
from flint.audio.qoa import encode_qoa

comptime KOKORO_SAMPLE_RATE: Int = 24000
comptime KOKORO_MAX_TOKENS: Int = 510
comptime KOKORO_STYLE_DIM: Int = 256


def kokoro_vocab() -> Dict[Int, Int]:
    """Phoneme codepoint -> Kokoro token id (the `vocab` of Kokoro-82M v1.0's config.json)."""
    var pairs: List[Int] = [
        0x3B, 1, 0x3A, 2, 0x2C, 3, 0x2E, 4, 0x21, 5, 0x3F, 6, 0x2014, 9, 0x2026, 10, 0x22, 11, 0x28, 12,
        0x29, 13, 0x201C, 14, 0x201D, 15, 0x20, 16, 0x303, 17, 0x2A3, 18, 0x2A5, 19, 0x2A6, 20, 0x2A8, 21,
        0x1D5D, 22, 0xAB67, 23, 0x41, 24, 0x49, 25, 0x4F, 31, 0x51, 33, 0x53, 35, 0x54, 36, 0x57, 39,
        0x59, 41, 0x1D4A, 42, 0x61, 43, 0x62, 44, 0x63, 45, 0x64, 46, 0x65, 47, 0x66, 48, 0x68, 50,
        0x69, 51, 0x6A, 52, 0x6B, 53, 0x6C, 54, 0x6D, 55, 0x6E, 56, 0x6F, 57, 0x70, 58, 0x71, 59,
        0x72, 60, 0x73, 61, 0x74, 62, 0x75, 63, 0x76, 64, 0x77, 65, 0x78, 66, 0x79, 67, 0x7A, 68,
        0x251, 69, 0x250, 70, 0x252, 71, 0xE6, 72, 0x3B2, 75, 0x254, 76, 0x255, 77, 0xE7, 78, 0x256, 80,
        0xF0, 81, 0x2A4, 82, 0x259, 83, 0x25A, 85, 0x25B, 86, 0x25C, 87, 0x25F, 90, 0x261, 92, 0x265, 99,
        0x268, 101, 0x26A, 102, 0x29D, 103, 0x26F, 110, 0x270, 111, 0x14B, 112, 0x273, 113, 0x272, 114,
        0x274, 115, 0xF8, 116, 0x278, 118, 0x3B8, 119, 0x153, 120, 0x279, 123, 0x27E, 125, 0x27B, 126,
        0x281, 128, 0x27D, 129, 0x282, 130, 0x283, 131, 0x288, 132, 0x2A7, 133, 0x28A, 135, 0x28B, 136,
        0x28C, 138, 0x263, 139, 0x264, 140, 0x3C7, 142, 0x28E, 143, 0x292, 147, 0x294, 148, 0x2C8, 156,
        0x2CC, 157, 0x2D0, 158, 0x2B0, 162, 0x2B2, 164, 0x2193, 169, 0x2192, 171, 0x2197, 172, 0x2198, 173,
        0x1D7B, 177,
    ]
    var d = Dict[Int, Int]()
    for i in range(0, len(pairs), 2):
        d[pairs[i]] = pairs[i + 1]
    return d^


# ----------------------------------------------------------------------------------- visemes

comptime VISEME_NAMES = "sil PP FF TH DD kk CH SS nn RR aa E ih oh ou"


def viseme_of(cp: Int) -> Int:
    """Phoneme codepoint -> one of 15 visemes (sil PP FF TH DD kk CH SS nn RR aa E ih oh ou, the
    common 15-shape lip-sync set). Stress marks and punctuation are silence (0)."""
    if cp == 112 or cp == 98 or cp == 109:                       # p b m
        return 1
    if cp == 102 or cp == 118:                                   # f v
        return 2
    if cp == 0x3B8 or cp == 0xF0:                                # θ ð
        return 3
    if cp == 116 or cp == 100 or cp == 84 or cp == 0x27E or cp == 108:   # t d T ɾ l
        return 4
    if cp == 107 or cp == 0x261 or cp == 0x14B or cp == 104 or cp == 0x294:  # k ɡ ŋ h ʔ
        return 5
    if cp == 0x2A7 or cp == 0x2A4 or cp == 0x283 or cp == 0x292 or cp == 106:  # ʧ ʤ ʃ ʒ j
        return 6
    if cp == 115 or cp == 122:                                   # s z
        return 7
    if cp == 110:                                                # n
        return 8
    if cp == 0x279 or cp == 0x25A or cp == 0x25C:                # ɹ ɚ ɜ
        return 9
    if cp == 0x251 or cp == 0xE6 or cp == 0x28C or cp == 0x250 or cp == 73 or cp == 87 or cp == 97:  # ɑ æ ʌ ɐ I W a
        return 10
    if cp == 0x25B or cp == 65 or cp == 101:                     # ɛ A e
        return 11
    if cp == 0x26A or cp == 105 or cp == 0x259 or cp == 0x1D4A or cp == 0x1D7B or cp == 0x268:  # ɪ i ə ᵊ ᵻ ɨ
        return 12
    if cp == 0x254 or cp == 79 or cp == 89 or cp == 0x252 or cp == 111:  # ɔ O Y ɒ o
        return 13
    if cp == 117 or cp == 0x28A or cp == 119:                    # u ʊ w
        return 14
    return 0


# ----------------------------------------------------------------------------------- results

struct PhonemeTiming(Copyable, Movable):
    var phoneme: Int     # codepoint
    var index: Int       # codepoint index in `KokoroSpeech.phonemes`
    var word: Int        # index into `KokoroSpeech.words`, -1 for spaces / punctuation
    var viseme: Int      # `viseme_of(phoneme)`
    var start: Float64   # seconds
    var end: Float64

    def __init__(out self, phoneme: Int, index: Int, word: Int, start: Float64, end: Float64):
        self.phoneme = phoneme
        self.index = index
        self.word = word
        self.viseme = viseme_of(phoneme)
        self.start = start
        self.end = end


struct KokoroSpeech(Movable):
    var samples: List[Float32]       # mono, KOKORO_SAMPLE_RATE
    var phonemes: String
    var words: List[G2PWord]
    var timings: List[PhonemeTiming]
    var seconds: Float64             # synthesis wall time (all chunks)

    def __init__(out self):
        self.samples = List[Float32]()
        self.phonemes = String()
        self.words = List[G2PWord]()
        self.timings = List[PhonemeTiming]()
        self.seconds = 0

    def duration(self) -> Float64:
        return Float64(len(self.samples)) / Float64(KOKORO_SAMPLE_RATE)

    def realtime_factor(self) -> Float64:
        """Synthesis time / audio duration (below 1 is faster than real time)."""
        var d = self.duration()
        return self.seconds / d if d > 0 else 0.0


def to_pcm16(samples: List[Float32]) -> List[Int16]:
    var out = List[Int16](capacity=len(samples))
    for s in samples:
        var v = min(max(Float64(s), -1.0), 1.0) * 32767.0
        out.append(Int16(Int(v + 0.5) if v >= 0 else Int(v - 0.5)))
    return out^


def write_wav_file(path: String, samples: List[Float32]) -> Bool:
    var bytes = encode_wav16(to_pcm16(samples), KOKORO_SAMPLE_RATE, 1)
    return write_bytes(path, bytes)


def write_qoa_file(path: String, samples: List[Float32]) -> Bool:
    var bytes = encode_qoa(to_pcm16(samples), 1, KOKORO_SAMPLE_RATE)
    return write_bytes(path, bytes)


def _f32_list(bytes: List[UInt8]) -> List[Float32]:
    var n = len(bytes) // 4
    var out = List[Float32](length=n, fill=0.0)
    for k in range(n):
        var u = UInt32(bytes[4 * k]) | (UInt32(bytes[4 * k + 1]) << 8) | (UInt32(bytes[4 * k + 2]) << 16) \
            | (UInt32(bytes[4 * k + 3]) << 24)
        out[k] = bitcast[DType.float32, 1](SIMD[DType.uint32, 1](u))[0]
    return out^


# --------------------------------------------------------------------------------------- TTS

struct KokoroTTS(Movable):
    var model: OnnxModel
    var g2p: G2P
    var vocab: Dict[Int, Int]
    var voice_dir: String
    var voices: Dict[String, List[Float32]]
    var durations_name: String       # the rounded / clipped per-token frame counts
    var patched_atan2: Bool
    var gpu: Bool

    def __init__(out self, var model: OnnxModel, var g2p: G2P, voice_dir: String):
        self.model = model^
        self.g2p = g2p^
        self.vocab = kokoro_vocab()
        self.voice_dir = voice_dir
        self.voices = Dict[String, List[Float32]]()
        self.durations_name = String()
        self.patched_atan2 = False
        self.gpu = False

    @staticmethod
    def load(dir: String, gpu: Bool = True, lexicon_dir: String = "") raises -> KokoroTTS:
        """`dir` holds `model.onnx` (or `onnx/model.onnx`), `voices/*.bin` and, unless
        `lexicon_dir` is given, `lexicon/us_gold.json` + `us_silver.json`. Without a lexicon the
        phonemiser falls back to its letter-to-sound rules."""
        var mpath = dir + "/model.onnx"
        if file_size(mpath) <= 0:
            mpath = dir + "/onnx/model.onnx"
        var model = OnnxModel.load(mpath)
        var lex = lexicon_dir if lexicon_dir.byte_length() > 0 else dir + "/lexicon"
        var g2p = G2P.load(lex) if file_size(lex + "/us_gold.json") > 0 else G2P()
        var t = KokoroTTS(model^, g2p^, dir + "/voices")
        t._prepare_graph()
        if gpu:
            t.gpu = t.model.use_gpu()
        return t^

    def device(self) raises -> String:
        return self.model.gpu_name() if self.gpu else "CPU"

    def _producer(self, value: Int) -> Int:
        for k in range(len(self.model.nodes)):
            for o in self.model.nodes[k].outputs:
                if o == value:
                    return k
        return -1

    def _prepare_graph(mut self) raises:
        """Find the duration tensor and restore PyTorch's atan2 at the DC / Nyquist bins."""
        ref nodes = self.model.nodes
        var stft_out = -1
        for k in range(len(nodes)):
            if nodes[k].op == OP_STFT:
                stft_out = nodes[k].outputs[0]
            if nodes[k].op == OP_CLIP and self.durations_name.byte_length() == 0:
                var p = self._producer(nodes[k].inputs[0])
                if p >= 0 and nodes[p].op == OP_ROUND:
                    self.durations_name = self.model.names[nodes[k].outputs[0]]
        if self.durations_name.byte_length() > 0:
            self.model.keep_value(self.durations_name)
        if stft_out < 0:
            return
        # Greater(Gather(Transpose(STFT), 1), 0): the "im > 0" of the atan2 emulation
        for k in range(len(self.model.nodes)):
            if self.model.nodes[k].op_type != "Greater":
                continue
            var g = self._producer(self.model.nodes[k].inputs[0])
            if g < 0 or self.model.nodes[g].op_type != "Gather":
                continue
            var tr = self._producer(self.model.nodes[g].inputs[0])
            if tr < 0 or self.model.nodes[tr].op_type != "Transpose" or self.model.nodes[tr].inputs[0] != stft_out:
                continue
            self.model.nodes[k].op = OP_GREATER_EQ
            self.patched_atan2 = True

    def voice(mut self, name: String) raises -> List[Float32]:
        if name in self.voices:
            return self.voices[name].copy()
        var bytes = read_bytes(self.voice_dir + "/" + name + ".bin")
        if len(bytes) < KOKORO_STYLE_DIM * 4:
            raise Error("Kokoro: no voice " + self.voice_dir + "/" + name + ".bin")
        var pack = _f32_list(bytes)
        self.voices[name] = pack.copy()
        return pack^

    def voice_names(self) -> List[String]:
        var out = List[String]()
        for f in list_directory(self.voice_dir):
            if f.endswith(".bin"):
                out.append(String(f[byte=0:f.byte_length() - 4]))
        return out^

    def _tokens(self, phonemes: String) raises -> Tuple[List[Int], List[Int]]:
        """Token ids for the phoneme string and, per id, its codepoint index (unknown symbols drop)."""
        var codes = decode_utf8(phonemes)
        var ids = List[Int]()
        var at = List[Int]()
        for i in range(len(codes)):
            if codes[i] in self.vocab:
                ids.append(self.vocab[codes[i]])
                at.append(i)
        return (ids^, at^)

    def _chunks(self, codes: List[Int], at: List[Int]) -> List[Int]:
        """End positions (into the token list) of chunks of at most KOKORO_MAX_TOKENS tokens,
        cut after sentence punctuation, else after a comma / pause, else at a space."""
        var ends = List[Int]()
        var start = 0
        var n = len(at)
        while n - start > KOKORO_MAX_TOKENS:
            var limit = start + KOKORO_MAX_TOKENS
            var cut = -1
            for rank in range(3):
                var j = limit - 1
                while j > start + KOKORO_MAX_TOKENS // 4:
                    var c = codes[at[j]]
                    var hit: Bool
                    if rank == 0:
                        hit = c == 46 or c == 33 or c == 63 or c == 0x2026
                    elif rank == 1:
                        hit = c == 44 or c == 59 or c == 58 or c == 0x2014
                    else:
                        hit = c == 32
                    if hit:
                        cut = j + 1
                        break
                    j -= 1
                if cut > 0:
                    break
            if cut < 0:
                cut = limit
            ends.append(cut)
            start = cut
        ends.append(n)
        return ends^

    def say(mut self, text: String, voice: String = "af_heart", speed: Float64 = 1.0) raises -> KokoroSpeech:
        """Phonemise `text` and synthesise it."""
        var r = self.g2p.phonemize(text)
        var out = self.speak_phonemes(r.phonemes, voice, speed)
        out.words = r.words.copy()
        for i in range(len(out.timings)):
            out.timings[i].word = r.word_at(out.timings[i].index)
        return out^

    def speak_phonemes(mut self, phonemes: String, voice: String = "af_heart", speed: Float64 = 1.0) raises -> KokoroSpeech:
        """Synthesise a misaki phoneme string (long input is split at punctuation)."""
        var out = KokoroSpeech()
        out.phonemes = phonemes
        var pack = self.voice(voice)
        var rows = len(pack) // KOKORO_STYLE_DIM
        var codes = decode_utf8(phonemes)
        var tk = self._tokens(phonemes)
        ref ids = tk[0]
        ref at = tk[1]
        var ends = self._chunks(codes, at)
        var t0 = perf_counter_ns()
        var start = 0
        for e in ends:
            # drop spaces at the chunk edges
            var a = start
            var b = e
            while a < b and codes[at[a]] == 32:
                a += 1
            while b > a and codes[at[b - 1]] == 32:
                b -= 1
            start = e
            if b <= a:
                continue
            var n = b - a
            var seq = List[Int]()
            seq.append(0)
            for j in range(a, b):
                seq.append(ids[j])
            seq.append(0)
            var row = min(n, rows - 1)
            var style = List[Float32](length=KOKORO_STYLE_DIM, fill=0.0)
            for c in range(KOKORO_STYLE_DIM):
                style[c] = pack[row * KOKORO_STYLE_DIM + c]
            self.model.set_input("input_ids", Tensor.ints(DT_I64, [1, n + 2], seq^))
            self.model.set_input("style", Tensor.floats([1, KOKORO_STYLE_DIM], style^))
            self.model.set_input("speed", Tensor.floats([1], [Float32(speed)]))
            self.model.run()
            var wav = self.model.output("waveform")
            var base = len(out.samples)
            for k in range(wav.numel()):
                out.samples.append(wav.fget(k))
            if self.durations_name.byte_length() > 0:
                var d = self.model.output(self.durations_name)
                var total = 0.0
                for k in range(d.numel()):
                    total += d.get(k)
                var spf = Float64(wav.numel()) / total if total > 0 else 0.0
                var frame = d.get(0)       # the leading pad token
                for j in range(n):
                    var dj = d.get(j + 1) if j + 1 < d.numel() else 0.0
                    var s0 = (Float64(base) + frame * spf) / Float64(KOKORO_SAMPLE_RATE)
                    frame += dj
                    var s1 = (Float64(base) + frame * spf) / Float64(KOKORO_SAMPLE_RATE)
                    out.timings.append(PhonemeTiming(codes[at[a + j]], at[a + j], -1, s0, s1))
        out.seconds = Float64(perf_counter_ns() - t0) / 1e9
        # a stress mark is spoken as part of the vowel after it
        var k = len(out.timings) - 2
        while k >= 0:
            var p = out.timings[k].phoneme
            if p == 0x2C8 or p == 0x2CC:
                out.timings[k].viseme = out.timings[k + 1].viseme
            k -= 1
        return out^
