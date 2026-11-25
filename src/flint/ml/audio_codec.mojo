"""Streaming neural audio codecs for Flint (`flint.ml.audio_codec`).

Shared pieces for ports that turn discrete audio tokens back into waveforms (music, SFX, voice):

  - `FBuf`: an owning malloc'd float32 buffer addressed by `Int`, the currency of the kernels.
  - `RvqCodebook`: residual vector quantiser tables `[levels, K, D]`: `decode` sums one centroid
    per level, `encode` does the greedy nearest-centroid residual search.
  - `StreamConv2D`: a causal-in-time 2-D convolution / transposed convolution over
    `[time, freq, channels]` activations that runs frame block by frame block with its own
    history, so streaming output equals whole-sequence output. Each output phase
    `(t mod st, f mod sf)` becomes one im2col + GEMM with only the taps that hit real inputs.
  - `InverseStft`: causal overlap-add iSTFT (Hann analysis window, tf.signal
    `inverse_stft_window_fn` synthesis window), the irfft folded into one GEMM per block.
  - `SpectroStreamDecoder`: Google DeepMind's SpectroStream decoder (Li et al. 2025,
    arXiv:2508.05207) as shipped with Magenta RealTime 2 (48 kHz stereo, 25 Hz frames, RVQ 64 x
    1024 x 256): codes -> RVQ embedding -> 2-D conv decoder over a 480-bin STFT -> waveform.
    `decode_frame` turns one 12..64-level code frame into 1920 stereo samples. Weights load
    from the official safetensors (`params/soundstream/decoder/...` inside `mrt2_*.safetensors`,
    or `resources/spectrostream/decoder.safetensors`) with no conversion.

Numerics follow `sequence_layers.jax` (the reference implementation): causal time padding
(`k - 1` frames on the left), "same"-style frequency padding, transposed convolutions as
input-dilated cross-correlations without kernel flips, Keras-style output sizes. Streaming
reproduces the reference's step mode, including the decoder's one-frame lookahead (the first
codec frame after `reset` decodes to silence).
"""

from std.ffi import external_call
from std.math import cos, sin, exp, sqrt
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.ops import matmul_2d
from flint.ml.llm.safetensors import SafeTensors, SafeTensor, widen_to_f32

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime PI64: Float64 = 3.141592653589793


@always_inline
def fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)


struct FBuf(Movable):
    """An owning float32 buffer (zero-filled); `addr` is the first element's address."""
    var addr: Int
    var n: Int

    def __init__(out self):
        self.addr = 0
        self.n = 0

    def __init__(out self, n: Int):
        self.n = n
        self.addr = external_call["calloc", Int](max(1, n), 4)

    def __deinit__(deinit self):
        if self.addr != 0:
            external_call["free", NoneType](self.addr)

    def resize(mut self, n: Int):
        """Grows to at least n floats (contents are not kept)."""
        if n <= self.n:
            return
        if self.addr != 0:
            external_call["free", NoneType](self.addr)
        self.n = n
        self.addr = external_call["calloc", Int](max(1, n), 4)

    def zero(mut self):
        _ = external_call["memset", Int](self.addr, Int32(0), self.n * 4)

    @always_inline
    def p(self) -> FP:
        return fp(self.addr)

    @always_inline
    def at(self, i: Int) -> Float32:
        return fp(self.addr + i * 4).unsafe_load()

    @always_inline
    def set(self, i: Int, v: Float32):
        fp(self.addr + i * 4).unsafe_store(v)


def copy_floats(dst: Int, src: Int, n: Int):
    _ = external_call["memcpy", Int](dst, src, n * 4)


def load_f32(st: SafeTensors, name: String) raises -> FBuf:
    var t = st.get(name)
    var b = FBuf(t.numel())
    widen_to_f32(t, b.addr, 0, t.numel())
    return b^


def tensor_shape(st: SafeTensors, name: String) raises -> List[Int]:
    return st.get(name).shape.copy()


# ------------------------------------------------------------------------------------------- RVQ

struct RvqCodebook(Movable):
    """`levels` codebooks of `size` centroids of dimension `dim`, stored [levels, size, dim]."""
    var levels: Int
    var size: Int
    var dim: Int
    var table: FBuf
    var norms: FBuf     # ||c||^2 per centroid, for `encode`

    def __init__(out self, levels: Int, size: Int, dim: Int, var table: FBuf):
        self.levels = levels
        self.size = size
        self.dim = dim
        self.table = table^
        self.norms = FBuf(levels * size)
        for i in range(levels * size):
            var acc: Float32 = 0
            for d in range(dim):
                var v = self.table.at(i * dim + d)
                acc += v * v
            self.norms.set(i, acc)

    @staticmethod
    def load(st: SafeTensors, name: String) raises -> RvqCodebook:
        var sh = tensor_shape(st, name)
        if len(sh) != 3:
            raise Error("RvqCodebook: expected [levels, K, D] for " + name)
        return RvqCodebook(sh[0], sh[1], sh[2], load_f32(st, name))

    def decode(self, codes: List[Int], out_addr: Int):
        """out[dim] = sum_l table[l, codes[l]] over the given levels (codes are 0..size-1)."""
        var o = fp(out_addr)
        for d in range(self.dim):
            o.unsafe_offset(d).unsafe_store(0)
        for l in range(min(len(codes), self.levels)):
            var row = self.table.addr + ((l * self.size + codes[l]) * self.dim) * 4
            for d in range(self.dim):
                var od = o.unsafe_offset(d)
                od.unsafe_store(od.unsafe_load() + fp(row + d * 4).unsafe_load())

    def encode(self, x_addr: Int, levels: Int) -> List[Int]:
        """Greedy residual quantisation of x[dim] with the first `levels` codebooks."""
        var r = FBuf(self.dim)
        copy_floats(r.addr, x_addr, self.dim)
        var codes = List[Int]()
        for l in range(min(levels, self.levels)):
            var best = 0
            var best_d: Float32 = 3.4e38
            for k in range(self.size):
                var row = self.table.addr + ((l * self.size + k) * self.dim) * 4
                var dot: Float32 = 0
                for d in range(self.dim):
                    dot += r.at(d) * fp(row + d * 4).unsafe_load()
                var dist = self.norms.at(l * self.size + k) - 2 * dot
                if dist < best_d:
                    best_d = dist
                    best = k
            codes.append(best)
            var row = self.table.addr + ((l * self.size + best) * self.dim) * 4
            for d in range(self.dim):
                r.set(d, r.at(d) - fp(row + d * 4).unsafe_load())
        return codes^


# ---------------------------------------------------------------------------- streaming conv

@always_inline
def elu(x: Float32) -> Float32:
    return x if x > 0 else exp(x) - 1


def elu_into(dst: Int, src: Int, n: Int):
    var s = fp(src)
    var d = fp(dst)
    for i in range(n):
        d.unsafe_offset(i).unsafe_store(elu(s.unsafe_offset(i).unsafe_load()))


@fieldwise_init
struct _Tap(Copyable, Movable, ImplicitlyCopyable):
    var a: Int     # time kernel index
    var b: Int     # freq kernel index


struct _Phase(Movable):
    var pt: Int
    var pf: Int
    var taps: List[_Tap]
    var w: FBuf    # [len(taps) * cin, cout]

    def __init__(out self, pt: Int, pf: Int, var taps: List[_Tap], var w: FBuf):
        self.pt = pt
        self.pf = pf
        self.taps = taps^
        self.w = w^


def floor_div(a: Int, b: Int) -> Int:
    var q = a // b
    return q


struct StreamConv2D(Movable):
    """Causal-in-time conv2d (time stride 1) or transposed conv2d over [time, freq, ch] blocks.

    `step(pool, x, t)` consumes `t` new input frames ([t, fin, cin] at address x) and writes
    `t * st` output frames ([t * st, fout, cout]) to `self.out`. Input frames are kept as history
    for the following steps (`hist` frames), so a sequence fed in any block sizes gives the same
    result as feeding it at once.
    """
    var kt: Int
    var kf: Int
    var st: Int
    var sf: Int
    var transposed: Bool
    var cin: Int
    var cout: Int
    var fin: Int
    var fout: Int
    var plt: Int
    var plf: Int
    var hist: Int
    var phases: List[_Phase]
    var bias: FBuf
    var inbuf: FBuf      # [hist + max_t, fin, cin]
    var col: FBuf
    var gout: FBuf
    var out: FBuf
    var max_t: Int

    def __init__(out self, kernel: FBuf, var bias: FBuf, kt: Int, kf: Int, cin: Int, cout: Int,
                 st: Int, sf: Int, transposed: Bool, fin: Int):
        self.kt = kt
        self.kf = kf
        self.st = st
        self.sf = sf
        self.transposed = transposed
        self.cin = cin
        self.cout = cout
        self.fin = fin
        self.bias = bias^
        self.plt = kt - 1
        if transposed:
            self.fout = fin * sf
            var pad_amount = kf + sf - 2
            if sf > kf - 1:
                self.plf = kf - 1
            else:
                self.plf = (pad_amount + 1) // 2
            self.hist = (kt - 1) // st
        else:
            if st != 1 or sf != 1:
                # Only stride-1 regular convolutions are needed by the decoders here.
                self.fout = fin
            else:
                self.fout = fin
            self.plf = (kf - 1) // 2
            self.hist = kt - 1
        self.phases = List[_Phase]()
        var nph_t = st if transposed else 1
        var nph_f = sf if transposed else 1
        for pt in range(nph_t):
            for pf in range(nph_f):
                var taps = List[_Tap]()
                for a in range(kt):
                    if (pt + a - self.plt) % nph_t != 0:
                        continue
                    for b in range(kf):
                        if (pf + b - self.plf) % nph_f != 0:
                            continue
                        taps.append(_Tap(a, b))
                var w = FBuf(len(taps) * cin * cout)
                for ti in range(len(taps)):
                    var src = kernel.addr + ((taps[ti].a * kf + taps[ti].b) * cin * cout) * 4
                    copy_floats(w.addr + ti * cin * cout * 4, src, cin * cout)
                self.phases.append(_Phase(pt, pf, taps^, w^))
        self.max_t = 0
        self.inbuf = FBuf()
        self.col = FBuf()
        self.gout = FBuf()
        self.out = FBuf()

    def reset(mut self):
        if self.inbuf.addr != 0:
            self.inbuf.zero()

    def _reserve(mut self, t: Int):
        if t <= self.max_t:
            return
        var old = self.max_t
        var keep = FBuf(self.hist * self.fin * self.cin)
        if old > 0 and self.hist > 0:
            copy_floats(keep.addr, self.inbuf.addr, self.hist * self.fin * self.cin)
        self.max_t = t
        var nin = (self.hist + t) * self.fin * self.cin
        var fresh = FBuf(nin)
        if old > 0 and self.hist > 0:
            copy_floats(fresh.addr, keep.addr, self.hist * self.fin * self.cin)
        self.inbuf = fresh^
        var tout = t * (self.st if self.transposed else 1)
        var max_taps = 0
        for pi in range(len(self.phases)):
            max_taps = max(max_taps, len(self.phases[pi].taps))
        self.col = FBuf(tout * self.fout * max_taps * self.cin)
        self.gout = FBuf(tout * self.fout * self.cout)
        self.out = FBuf(tout * self.fout * self.cout)

    def input_slot(mut self, t: Int) -> Int:
        """Address where the caller may write the t new (already activated) input frames."""
        self._reserve(t)
        return self.inbuf.addr + self.hist * self.fin * self.cin * 4

    def step(mut self, mut pool: JobPool, x: Int, t: Int) -> Int:
        """Runs t new frames (x may be `input_slot(t)` itself). Returns output frame count."""
        self._reserve(t)
        var row = self.fin * self.cin
        var slot = self.inbuf.addr + self.hist * row * 4
        if x != slot:
            copy_floats(slot, x, t * row)
        var tout = t * (self.st if self.transposed else 1)
        var nph_t = self.st if self.transposed else 1
        var nph_f = self.sf if self.transposed else 1
        for pi in range(len(self.phases)):
            ref ph = self.phases[pi]
            var ntap = len(ph.taps)
            if ntap == 0:
                continue
            var k = ntap * self.cin
            # rows of this phase: o_t = pt + nph_t * i, o_f = pf + nph_f * j
            var rt = (tout - ph.pt + nph_t - 1) // nph_t
            var rf = (self.fout - ph.pf + nph_f - 1) // nph_f
            var m = rt * rf
            if m == 0:
                continue
            for i in range(rt):
                var ot = ph.pt + nph_t * i
                for j in range(rf):
                    var of_ = ph.pf + nph_f * j
                    var r = i * rf + j
                    var dst = self.col.addr + (r * k) * 4
                    for ti in range(ntap):
                        var it_num = ot + ph.taps[ti].a - self.plt
                        var if_num = of_ + ph.taps[ti].b - self.plf
                        var it = floor_div(it_num, nph_t)
                        var if_ = floor_div(if_num, nph_f)
                        var d = dst + ti * self.cin * 4
                        if if_ < 0 or if_ >= self.fin or it < -self.hist or it >= t:
                            _ = external_call["memset", Int](d, Int32(0), self.cin * 4)
                        else:
                            var src = self.inbuf.addr + (((self.hist + it) * self.fin + if_) * self.cin) * 4
                            copy_floats(d, src, self.cin)
            matmul_2d(pool, self.col.addr, ph.w.addr, self.gout.addr, self.bias.addr, m, k, self.cout, False)
            for i in range(rt):
                var ot = ph.pt + nph_t * i
                for j in range(rf):
                    var of_ = ph.pf + nph_f * j
                    copy_floats(self.out.addr + ((ot * self.fout + of_) * self.cout) * 4,
                                self.gout.addr + ((i * rf + j) * self.cout) * 4, self.cout)
        # keep the last `hist` input frames
        if self.hist > 0:
            _ = external_call["memmove", Int](self.inbuf.addr, self.inbuf.addr + t * row * 4,
                                              self.hist * row * 4)
        return tout

    def flops(self, t: Int) -> Int:
        var tout = t * (self.st if self.transposed else 1)
        var total = 0
        for pi in range(len(self.phases)):
            ref ph = self.phases[pi]
            var nph_t = self.st if self.transposed else 1
            var nph_f = self.sf if self.transposed else 1
            var rt = (tout - ph.pt + nph_t - 1) // nph_t
            var rf = (self.fout - ph.pf + nph_f - 1) // nph_f
            total += 2 * rt * rf * len(ph.taps) * self.cin * self.cout
        return total


def load_conv(st: SafeTensors, name: String, sti: Int, sfi: Int, transposed: Bool, fin: Int) raises -> StreamConv2D:
    """A `StreamConv2D` from `name/kernel` ([kt, kf, cin, cout], HWIO) and `name/bias`."""
    var sh = tensor_shape(st, name + "/kernel")
    var k = load_f32(st, name + "/kernel")
    var b = load_f32(st, name + "/bias")
    return StreamConv2D(k, b^, sh[0], sh[1], sh[2], sh[3], sti, sfi, transposed, fin)


# ------------------------------------------------------------------------------------ iSTFT

def hann_periodic(n: Int) -> List[Float64]:
    var w = List[Float64]()
    for i in range(n):
        w.append(0.5 - 0.5 * cos(2.0 * PI64 * Float64(i) / Float64(n)))
    return w^


struct InverseStft(Movable):
    """Causal streaming inverse STFT: frames of `bins` complex values (bins 0..bins-1; the
    remaining rfft bins up to fft/2 are zero) -> `hop` samples per frame, overlap-added.

    Matches `sequence_layers` `InverseSTFT(time_padding='causal', fft_padding='right')` with a
    periodic Hann forward window and `inverse_stft_window_fn(hop)`: output sample block f is
    frame f's first hop samples plus the previous frame's overlap tail.
    """
    var frame: Int
    var hop: Int
    var bins: Int
    var basis: FBuf      # [2 * bins, frame]: rows re_k, then im_k; includes 1/N, x2 and window
    var tail: FBuf       # [channels, frame - hop]
    var spec: FBuf       # [rows, 2 * bins]
    var frames: FBuf     # [rows, frame]
    var channels: Int

    def __init__(out self, frame: Int, hop: Int, fft: Int, bins: Int, channels: Int):
        self.frame = frame
        self.hop = hop
        self.bins = bins
        self.channels = channels
        var fw = hann_periodic(frame)
        var overlaps = (frame + hop - 1) // hop
        var denom = List[Float64](length=hop, fill=0.0)
        for i in range(frame):
            denom[i % hop] += fw[i] * fw[i]
        var win = List[Float64]()
        for i in range(frame):
            var d = denom[i % hop]
            win.append(0.0 if d == 0.0 else fw[i] / d)
        _ = overlaps
        self.basis = FBuf(2 * bins * frame)
        for k in range(bins):
            var c: Float64 = 1.0 if (k == 0 or 2 * k == fft) else 2.0
            for n in range(frame):
                var ang = 2.0 * PI64 * Float64((k * n) % fft) / Float64(fft)
                var s = c / Float64(fft) * win[n]
                self.basis.set(k * frame + n, Float32(s * cos(ang)))
                if k == 0 or 2 * k == fft:
                    self.basis.set((bins + k) * frame + n, 0.0)
                else:
                    self.basis.set((bins + k) * frame + n, Float32(-s * sin(ang)))
        self.tail = FBuf(channels * (frame - hop))
        self.spec = FBuf()
        self.frames = FBuf()

    def reset(mut self):
        self.tail.zero()

    def step(mut self, mut pool: JobPool, spec: Int, t: Int, out_addr: Int):
        """spec: [t, channels, 2 * bins] (re block, im block). out: [t * hop, channels]."""
        var rows = t * self.channels
        self.frames.resize(rows * self.frame)
        matmul_2d(pool, spec, self.basis.addr, self.frames.addr, 0, rows, 2 * self.bins, self.frame, False)
        var o = fp(out_addr)
        var ov = self.frame - self.hop
        for f in range(t):
            for c in range(self.channels):
                var fr = self.frames.addr + ((f * self.channels + c) * self.frame) * 4
                var tl = self.tail.addr + c * ov * 4
                for n in range(self.hop):
                    var v = fp(fr + n * 4).unsafe_load()
                    if n < ov:
                        v += fp(tl + n * 4).unsafe_load()
                    o.unsafe_offset((f * self.hop + n) * self.channels + c).unsafe_store(v)
                # new tail = old tail shifted by hop + this frame's samples from hop on
                for n in range(ov):
                    var v = fp(fr + (self.hop + n) * 4).unsafe_load()
                    if n + self.hop < ov:
                        v += fp(tl + (n + self.hop) * 4).unsafe_load()
                    fp(tl + n * 4).unsafe_store(v)


# ------------------------------------------------------------------------- SpectroStream

comptime SS_SAMPLE_RATE: Int = 48000
comptime SS_FRAME_SAMPLES: Int = 1920     # 40 ms per codec frame
comptime SS_STFT_HOP: Int = 480
comptime SS_STFT_FRAMES: Int = 4          # STFT frames per codec frame
comptime SS_BINS: Int = 480
comptime SS_CHANNELS: Int = 2


struct _ResUnit(Movable):
    """SpectroStream decoder block: x -> conv_b(elu(conv_a(elu(x)))) + shortcut(x)."""
    var conv_a: StreamConv2D      # transposed (strided) or 3x3_a
    var conv_b: StreamConv2D      # 3x3
    var has_short: Bool
    var short: StreamConv2D       # 1x1 (only when channels change)
    var up_t: Int
    var up_f: Int
    var act: FBuf
    var sc: FBuf

    def __init__(out self, var conv_a: StreamConv2D, var conv_b: StreamConv2D, has_short: Bool,
                 var short: StreamConv2D, up_t: Int, up_f: Int):
        self.conv_a = conv_a^
        self.conv_b = conv_b^
        self.has_short = has_short
        self.short = short^
        self.up_t = up_t
        self.up_f = up_f
        self.act = FBuf()
        self.sc = FBuf()

    def reset(mut self):
        self.conv_a.reset()
        self.conv_b.reset()
        if self.has_short:
            self.short.reset()

    def step(mut self, mut pool: JobPool, x: Int, t: Int) -> Int:
        """Returns the output frame count; the result is in `self.conv_b.out` (residual added)."""
        var fin = self.conv_a.fin
        var cin = self.conv_a.cin
        var slot = self.conv_a.input_slot(t)
        elu_into(slot, x, t * fin * cin)
        var t1 = self.conv_a.step(pool, slot, t)
        var n1 = t1 * self.conv_a.fout * self.conv_a.cout
        var slot2 = self.conv_b.input_slot(t1)
        elu_into(slot2, self.conv_a.out.addr, n1)
        var t2 = self.conv_b.step(pool, slot2, t1)
        var fout = self.conv_b.fout
        var cout = self.conv_b.cout
        # shortcut at input resolution, then nearest upsampling
        var s_addr = x
        var s_c = cin
        if self.has_short:
            _ = self.short.step(pool, x, t)
            s_addr = self.short.out.addr
            s_c = self.short.cout
        var y = self.conv_b.out.addr
        for ot in range(t2):
            var it = ot // self.up_t
            for of_ in range(fout):
                var if_ = of_ // self.up_f
                var src = fp(s_addr + ((it * fin + if_) * s_c) * 4)
                var dst = fp(y + ((ot * fout + of_) * cout) * 4)
                for c in range(cout):
                    var d = dst.unsafe_offset(c)
                    d.unsafe_store(d.unsafe_load() + src.unsafe_offset(c).unsafe_load())
        return t2


def _no_conv() -> StreamConv2D:
    return StreamConv2D(FBuf(1), FBuf(1), 1, 1, 1, 1, 1, 1, False, 1)


struct SpectroStreamDecoder(Movable):
    """Streaming SpectroStream decoder (48 kHz stereo, 25 Hz codec frames)."""
    var rvq: RvqCodebook
    var first: StreamConv2D       # conv1x1_first: 256 -> 2560
    var b1: StreamConv2D
    var b2: StreamConv2D
    var iru: _ResUnit             # input_layers_residual_unit (512 ch, 5 bins)
    var d0: _ResUnit              # decoder_0 (shared input of both channel groups)
    var groups: List[_ResUnit]    # decoder_1 .. decoder_6 (weights shared by both groups)
    var last: StreamConv2D
    var group_state: List[List[_ResUnit]]
    var last_state: List[StreamConv2D]
    var istft: InverseStft
    var lookahead_left: Int
    var emb: FBuf
    var x0: FBuf
    var x1: FBuf
    var gin: FBuf
    var spec: FBuf
    var pcm: FBuf
    var levels: Int

    def __init__(out self, var rvq: RvqCodebook, var first: StreamConv2D, var b1: StreamConv2D,
                 var b2: StreamConv2D, var iru: _ResUnit, var d0: _ResUnit,
                 var group_state: List[List[_ResUnit]], var last_state: List[StreamConv2D], levels: Int):
        self.rvq = rvq^
        self.first = first^
        self.b1 = b1^
        self.b2 = b2^
        self.iru = iru^
        self.d0 = d0^
        self.groups = List[_ResUnit]()
        self.last = _no_conv()
        self.group_state = group_state^
        self.last_state = last_state^
        self.istft = InverseStft(960, SS_STFT_HOP, 960, SS_BINS, SS_CHANNELS)
        self.lookahead_left = SS_STFT_FRAMES
        self.emb = FBuf(256)
        self.x0 = FBuf(2560)
        self.x1 = FBuf(2560)
        self.gin = FBuf(2 * 5 * 512)
        self.spec = FBuf(SS_STFT_FRAMES * SS_CHANNELS * 2 * SS_BINS)
        self.pcm = FBuf(SS_FRAME_SAMPLES * SS_CHANNELS)
        self.levels = levels

    @staticmethod
    def load(st: SafeTensors, decoder_prefix: String, quantizer_name: String, levels: Int = 12) raises -> SpectroStreamDecoder:
        """`decoder_prefix` e.g. "params/soundstream/decoder/"; `quantizer_name` e.g.
        "params/soundstream/quantizer/embedding"; `levels` = RVQ levels decoded per frame."""
        var p = decoder_prefix
        var rvq = RvqCodebook.load(st, quantizer_name)
        var first = load_conv(st, p + "input_layer/conv1x1_first/conv", 1, 1, False, 1)
        var b1 = load_conv(st, p + "input_layer/shortcut_layer/conv1x1_b1/conv", 1, 1, False, 1)
        var b2 = load_conv(st, p + "input_layer/shortcut_layer/conv1x1_b2/conv", 1, 1, False, 1)
        var iru = _ResUnit(load_conv(st, p + "input_layers_residual_unit/conv2d_3x3_a/conv", 1, 1, False, 5),
                           load_conv(st, p + "input_layers_residual_unit/conv2d_3x3/conv", 1, 1, False, 5),
                           False, _no_conv(), 1, 1)
        var d0 = _ResUnit(load_conv(st, p + "decoder_0/conv2dtranspose_4x3/conv", 2, 1, True, 5),
                          load_conv(st, p + "decoder_0/conv2d_3x3/conv", 1, 1, False, 5),
                          True, load_conv(st, p + "decoder_0/shortcut_layer/conv1x1/conv", 1, 1, False, 5), 2, 1)
        # decoder_1..6 per channel group (separate streaming state, same weights)
        var names: List[String] = ["decoder_1/conv2dtranspose_4x4", "decoder_2/conv2dtranspose_3x4",
                                   "decoder_3/conv2dtranspose_3x4", "decoder_4/conv2dtranspose_3x6",
                                   "decoder_5/conv2dtranspose_3x4", "decoder_6/conv2dtranspose_3x4"]
        var strides: List[Int] = [2, 2, 1, 2, 1, 2, 1, 3, 1, 2, 1, 2]
        var shorts: List[Bool] = [True, False, False, True, False, True]
        var group_state = List[List[_ResUnit]]()
        var last_state = List[StreamConv2D]()
        for _g in range(2):
            var units = List[_ResUnit]()
            var fin = 5
            for i in range(6):
                var blk = "decoder_" + String(i + 1)
                var stt = strides[2 * i]
                var stf = strides[2 * i + 1]
                var ca = load_conv(st, p + names[i] + "/conv", stt, stf, True, fin)
                var cb = load_conv(st, p + blk + "/conv2d_3x3/conv", 1, 1, False, fin * stf)
                var sc = _no_conv()
                if shorts[i]:
                    sc = load_conv(st, p + blk + "/shortcut_layer/conv1x1/conv", 1, 1, False, fin)
                units.append(_ResUnit(ca^, cb^, shorts[i], sc^, stt, stf))
                fin *= stf
            group_state.append(units^)
            last_state.append(load_conv(st, p + "output_layer/base_conv_last/conv", 1, 1, False, fin))
        return SpectroStreamDecoder(rvq^, first^, b1^, b2^, iru^, d0^, group_state^, last_state^, levels)

    def reset(mut self):
        self.first.reset()
        self.b1.reset()
        self.b2.reset()
        self.iru.reset()
        self.d0.reset()
        for g in range(len(self.group_state)):
            for i in range(len(self.group_state[g])):
                self.group_state[g][i].reset()
            self.last_state[g].reset()
        self.istft.reset()
        self.lookahead_left = SS_STFT_FRAMES

    def embed(self, codes: List[Int], out_addr: Int):
        var c = List[Int]()
        for i in range(min(self.levels, len(codes))):
            c.append(codes[i])
        self.rvq.decode(c, out_addr)

    def decode_embedding(mut self, mut pool: JobPool, emb: Int) -> Int:
        """One 256-d RVQ embedding -> 1920 interleaved stereo samples at the returned address."""
        # input layer: conv_first(x) + b2(elu(b1(x)))  [1, 1, 2560] -> reshape [1, 5, 512]
        _ = self.first.step(pool, emb, 1)
        _ = self.b1.step(pool, emb, 1)
        var s2 = self.b2.input_slot(1)
        elu_into(s2, self.b1.out.addr, 2560)
        _ = self.b2.step(pool, s2, 1)
        for i in range(2560):
            self.x0.set(i, self.first.out.at(i) + self.b2.out.at(i))
        # input residual unit (identity shortcut)
        _ = self.iru.step(pool, self.x0.addr, 1)
        # decoder_0: [1, 5, 512] -> [2, 5, 1024]
        _ = self.d0.step(pool, self.iru.conv_b.out.addr, 1)
        var d0o = self.d0.conv_b.out.addr
        for g in range(2):
            # channel group g: channels [g * 512, (g + 1) * 512)
            for tf in range(2 * 5):
                copy_floats(self.gin.addr + tf * 512 * 4, d0o + (tf * 1024 + g * 512) * 4, 512)
            var x = self.gin.addr
            var t = 2
            for i in range(6):
                t = self.group_state[g][i].step(pool, x, t)
                x = self.group_state[g][i].conv_b.out.addr
            var slot = self.last_state[g].input_slot(t)
            elu_into(slot, x, t * 480 * 64)
            _ = self.last_state[g].step(pool, slot, t)
            # [4, 480, 2] (re, im) -> spec[f, g, re bins | im bins]
            var o = self.last_state[g].out.addr
            for f in range(SS_STFT_FRAMES):
                var base = self.spec.addr + ((f * SS_CHANNELS + g) * 2 * SS_BINS) * 4
                for k in range(SS_BINS):
                    fp(base + k * 4).unsafe_store(fp(o + ((f * SS_BINS + k) * 2) * 4).unsafe_load())
                    fp(base + (SS_BINS + k) * 4).unsafe_store(fp(o + ((f * SS_BINS + k) * 2 + 1) * 4).unsafe_load())
        # one-frame lookahead of the reference decoder: the first STFT frames are invalid (zero)
        while self.lookahead_left > 0:
            var f = SS_STFT_FRAMES - self.lookahead_left
            _ = external_call["memset", Int](self.spec.addr + (f * SS_CHANNELS * 2 * SS_BINS) * 4, Int32(0),
                                             SS_CHANNELS * 2 * SS_BINS * 4)
            self.lookahead_left -= 1
        self.istft.step(pool, self.spec.addr, SS_STFT_FRAMES, self.pcm.addr)
        return self.pcm.addr

    def decode_frame(mut self, mut pool: JobPool, codes: List[Int]) -> Int:
        """Codes (0..1023, first `levels` RVQ levels) -> 1920 stereo samples (interleaved L, R)."""
        self.embed(codes, self.emb.addr)
        return self.decode_embedding(pool, self.emb.addr)
