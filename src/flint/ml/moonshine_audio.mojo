"""Audio front end for Moonshine speech-to-text (`flint.ml.moonshine_audio`).

  - `read_wav(path)`: any RIFF/WAVE file (PCM 8/16/24/32-bit, IEEE float 32/64, extensible
    headers, any channel count), downmixed to mono float32 in [-1, 1].
  - `Resampler(fin, fout)`: polyphase windowed-sinc resampler for any rational rate pair
    (Kaiser window, beta 8.6, 16 zero crossings, cutoff 0.945 x the lower Nyquist), used to bring
    44.1 / 48 / 24 kHz input to the model's 16 kHz.
  - `moonshine_frames`: the model's own feature step: 5 ms (80-sample) frames, per-frame
    CMVN (zero mean, unit RMS) and asinh(k x) compression.
  - `EnergyVad`: a small energy endpointer (adaptive noise floor, hangover) that cuts a live
    16 kHz stream into utterances for voice commands.
"""

from std.math import sqrt, sin, exp, log, ceil, asinh
from flint.platform.os import read_bytes

comptime MOONSHINE_RATE: Int = 16000
comptime MOONSHINE_FRAME: Int = 80   # 5 ms at 16 kHz


struct AudioClip(Movable):
    var samples: List[Float32]   # mono
    var rate: Int
    var valid: Bool

    def __init__(out self):
        self.samples = List[Float32]()
        self.rate = 0
        self.valid = False

    def seconds(self) -> Float64:
        if self.rate <= 0:
            return 0
        return Float64(len(self.samples)) / Float64(self.rate)


def _u16(d: List[UInt8], o: Int) -> Int:
    return Int(d[o]) | (Int(d[o + 1]) << 8)


def _u32(d: List[UInt8], o: Int) -> Int:
    return Int(d[o]) | (Int(d[o + 1]) << 8) | (Int(d[o + 2]) << 16) | (Int(d[o + 3]) << 24)


def _tag(d: List[UInt8], o: Int, t: String) -> Bool:
    for i in range(4):
        if d[o + i] != t.as_bytes()[i]:
            return False
    return True


def decode_wav_mono(data: List[UInt8]) -> AudioClip:
    var clip = AudioClip()
    if len(data) < 12 or not _tag(data, 0, "RIFF") or not _tag(data, 8, "WAVE"):
        return clip^
    var fmt = 0
    var channels = 0
    var bits = 0
    var align = 0
    var rate = 0
    var data_at = -1
    var data_len = 0
    var off = 12
    while off + 8 <= len(data):
        var size = _u32(data, off + 4)
        var start = off + 8
        if _tag(data, off, "fmt ") and size >= 16 and start + 16 <= len(data):
            fmt = _u16(data, start)
            channels = _u16(data, start + 2)
            rate = _u32(data, start + 4)
            align = _u16(data, start + 12)
            bits = _u16(data, start + 14)
            if fmt == 0xFFFE and size >= 26:
                fmt = _u16(data, start + 24)   # sub-format GUID's first two bytes
        elif _tag(data, off, "data"):
            data_at = start
            data_len = min(size, len(data) - start)   # streamed files may carry a bogus size
            break
        off = start + size + (size & 1)
    if data_at < 0 or channels < 1 or rate < 1 or align < 1:
        return clip^
    var bps = bits // 8
    if (fmt == 1 and (bits == 8 or bits == 16 or bits == 24 or bits == 32)) or (fmt == 3 and (bits == 32 or bits == 64)):
        pass
    else:
        return clip^
    var frames = data_len // align
    clip.samples = List[Float32](length=frames, fill=0.0)
    var inv_ch = 1.0 / Float64(channels)
    for f in range(frames):
        var acc: Float64 = 0
        for c in range(channels):
            var at = data_at + f * align + c * bps
            var v: Float64
            if fmt == 3:
                if bits == 32:
                    var u = UInt32(_u32(data, at))
                    v = Float64(bitcast_f32(u))
                else:
                    var lo = UInt64(_u32(data, at))
                    var hi = UInt64(_u32(data, at + 4))
                    v = bitcast_f64((hi << 32) | lo)
            elif bits == 8:
                v = Float64(Int(data[at]) - 128) / 128.0
            elif bits == 16:
                var s = _u16(data, at)
                if s >= 32768:
                    s -= 65536
                v = Float64(s) / 32768.0
            elif bits == 24:
                var s = Int(data[at]) | (Int(data[at + 1]) << 8) | (Int(data[at + 2]) << 16)
                if s >= 8388608:
                    s -= 16777216
                v = Float64(s) / 8388608.0
            else:
                var s = _u32(data, at)
                if s >= 2147483648:
                    s -= 4294967296
                v = Float64(s) / 2147483648.0
            acc += v
        clip.samples[f] = Float32(acc * inv_ch)
    clip.rate = rate
    clip.valid = True
    return clip^


def bitcast_f32(u: UInt32) -> Float32:
    from std.memory import bitcast
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](u))[0]


def bitcast_f64(u: UInt64) -> Float64:
    from std.memory import bitcast
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](u))[0]


def read_wav(path: String) -> AudioClip:
    return decode_wav_mono(read_bytes(path))


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


def bessel_i0(x: Float64) -> Float64:
    """Modified Bessel function of the first kind, order 0 (power series)."""
    var sum: Float64 = 1
    var term: Float64 = 1
    var q = x * x / 4.0
    var k = 1
    while k < 200:
        term *= q / Float64(k * k)
        sum += term
        if term < sum * 1e-17:
            break
        k += 1
    return sum


comptime PI64: Float64 = 3.141592653589793


struct Resampler(Movable):
    """y[n] = sum_k x[k] h(n M / L - k): one precomputed filter phase per output fraction."""
    var fin: Int
    var fout: Int
    var up: Int        # L
    var down: Int      # M
    var half: Int
    var taps: Int
    var table: List[Float64]   # [L, taps]

    def __init__(out self, fin: Int, fout: Int, zeros: Int = 16, beta: Float64 = 8.6, roll: Float64 = 0.945):
        self.fin = fin
        self.fout = fout
        var g = _gcd(fin, fout)
        self.up = fout // g
        self.down = fin // g
        var fc = min(1.0, Float64(fout) / Float64(fin)) * roll
        self.half = Int(ceil(Float64(zeros) / fc))
        self.taps = 2 * self.half
        self.table = List[Float64]()
        if fin == fout:
            return
        self.table = List[Float64](length=self.up * self.taps, fill=0.0)
        var i0b = bessel_i0(beta)
        for p in range(self.up):
            var frac = Float64(p) / Float64(self.up)
            for i in range(self.taps):
                var k = i - self.half + 1
                var d = frac - Float64(k)
                var r = d / Float64(self.half)
                var w = bessel_i0(beta * sqrt(max(0.0, 1.0 - r * r))) / i0b
                var a = fc * d
                var s: Float64 = 1.0
                if a != 0:
                    s = sin(PI64 * a) / (PI64 * a)
                self.table[p * self.taps + i] = fc * s * w

    def output_length(self, n: Int) -> Int:
        if self.fin == self.fout:
            return n
        return (n * self.up) // self.down

    def process(self, x: List[Float32]) -> List[Float32]:
        """Resamples a whole buffer (zero-padded at both ends)."""
        var n = len(x)
        if self.fin == self.fout:
            return x.copy()
        var n_out = self.output_length(n)
        var y = List[Float32](length=n_out, fill=0.0)
        for o in range(n_out):
            var base = (o * self.down) // self.up
            var p = (o * self.down) % self.up
            var first = base - self.half + 1
            var acc: Float64 = 0
            var row = p * self.taps
            var i0 = max(0, -first)
            var i1 = min(self.taps, n - first)
            for i in range(i0, i1):
                acc += self.table[row + i] * Float64(x[first + i])
            y[o] = Float32(acc)
        return y^


def to_model_rate(clip: AudioClip) -> List[Float32]:
    """The clip at 16 kHz mono."""
    if clip.rate == MOONSHINE_RATE:
        return clip.samples.copy()
    var r = Resampler(clip.rate, MOONSHINE_RATE)
    return r.process(clip.samples)


def moonshine_frames(x: List[Float32], frames: Int, k: Float32) -> List[Float32]:
    """[frames, 80]: per-frame CMVN (eps 1e-6) then asinh(k x). `x` must hold frames * 80
    samples (zero padded)."""
    var out = List[Float32](length=frames * MOONSHINE_FRAME, fill=0.0)
    for f in range(frames):
        var b = f * MOONSHINE_FRAME
        var s: Float32 = 0
        for i in range(MOONSHINE_FRAME):
            s += x[b + i]
        var mean = s / Float32(MOONSHINE_FRAME)
        var q: Float32 = 0
        for i in range(MOONSHINE_FRAME):
            var c = x[b + i] - mean
            q += c * c
        var inv = 1.0 / sqrt(q / Float32(MOONSHINE_FRAME) + 1e-6)
        for i in range(MOONSHINE_FRAME):
            out[b + i] = asinh(k * ((x[b + i] - mean) * inv))
    return out^


struct EnergyVad(Movable):
    """Energy endpointer on 16 kHz mono in 20 ms hops. Speech starts when a hop is `open_db`
    over the tracked noise floor and ends after `hang_ms` of hops under `close_db`; a pre-roll
    keeps the onset. `push` returns True when an utterance has just ended (read `utterance`)."""
    var hop: Int
    var floor_db: Float32
    var open_db: Float32
    var close_db: Float32
    var hang_hops: Int
    var max_hops: Int
    var min_hops: Int
    var preroll_hops: Int
    var active: Bool
    var quiet: Int
    var hops: Int
    var pending: List[Float32]
    var history: List[Float32]   # last preroll_hops hops of audio
    var utterance: List[Float32]
    var level_db: Float32

    def __init__(out self, open_db: Float32 = 12, close_db: Float32 = 6, hang_ms: Int = 450,
                 max_ms: Int = 12000, min_ms: Int = 200, preroll_ms: Int = 200):
        self.hop = 320
        self.floor_db = -60
        self.open_db = open_db
        self.close_db = close_db
        self.hang_hops = max(1, hang_ms // 20)
        self.max_hops = max(1, max_ms // 20)
        self.min_hops = max(1, min_ms // 20)
        self.preroll_hops = max(0, preroll_ms // 20)
        self.active = False
        self.quiet = 0
        self.hops = 0
        self.pending = List[Float32]()
        self.history = List[Float32]()
        self.utterance = List[Float32]()
        self.level_db = -100

    def push(mut self, samples: List[Float32]) -> Bool:
        var ended = False
        for s in samples:
            self.pending.append(s)
        var at = 0
        while at + self.hop <= len(self.pending):
            var e: Float32 = 0
            for i in range(self.hop):
                var v = self.pending[at + i]
                e += v * v
            var db = Float32(10.0 * log(Float64(e / Float32(self.hop)) + 1e-12) / log(10.0))
            self.level_db = db
            var hopbuf = List[Float32](capacity=self.hop)
            for i in range(self.hop):
                hopbuf.append(self.pending[at + i])
            if not self.active:
                # slow rise, fast fall: the floor follows the quietest recent level
                if db < self.floor_db:
                    self.floor_db = 0.7 * self.floor_db + 0.3 * db
                else:
                    self.floor_db = 0.995 * self.floor_db + 0.005 * db
                if db > self.floor_db + self.open_db:
                    self.active = True
                    self.quiet = 0
                    self.hops = 0
                    self.utterance = self.history.copy()
            if self.active:
                self.utterance.extend(hopbuf.copy())
                self.hops += 1
                if db < self.floor_db + self.close_db:
                    self.quiet += 1
                else:
                    self.quiet = 0
                if self.quiet >= self.hang_hops or self.hops >= self.max_hops:
                    self.active = False
                    if self.hops - self.quiet >= self.min_hops:
                        ended = True
                    else:
                        self.utterance = List[Float32]()
            self.history.extend(hopbuf^)
            var keep = self.preroll_hops * self.hop
            if len(self.history) > keep:
                var trimmed = List[Float32](capacity=keep)
                for i in range(len(self.history) - keep, len(self.history)):
                    trimmed.append(self.history[i])
                self.history = trimmed^
            at += self.hop
        var rest = List[Float32](capacity=len(self.pending) - at)
        for i in range(at, len(self.pending)):
            rest.append(self.pending[i])
        self.pending = rest^
        return ended
