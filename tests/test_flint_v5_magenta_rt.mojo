"""Flint v5 Magenta RealTime 2 suite (`test_flint_v5_magenta_rt.mojo`).

Always (no model files):
  1. `flint.ml.jax_random`: threefry split / fold_in / bits / uniform / gumbel against JAX.
  2. `NumpyRandomState`: `numpy.random.RandomState(seed).randn` bit for bit.
  3. RVQ encode / decode round trip.
  4. `InverseStft`: perfect reconstruction (one hop of delay) of a causal periodic-Hann STFT,
     and identical output for any number of frames per step.
  5. `StreamConv2D`: causal and transposed (strided) convs against a naive reference, and
     streaming in 1-frame steps equals one 3-frame step.
  6. `SpmUnigram`: Viterbi segmentation, whitespace rules and byte fallback.
  7. Conditioning row layout and CFG discretisation.
  8. `PcmStreamRing` wrap-around / underruns and mixing through `AudioServer.add_pcm_stream`.
With `FLINT_MAGENTA_RT_DIR` (a google/magenta-realtime-2 download prepared by
`tools/magenta_rt/fetch.sh`) and `FLINT_MAGENTA_RT_REF` (dumps from
`tools/magenta_rt/reference.py` + `reference_musiccoca.py`): MusicCoCa ids / embeddings /
tokens, LM token parity over the reference frames (style switch included), SpectroStream audio
parity, and the GPU codec against the CPU one when a GPU is present.
"""
from std.math import abs, sqrt, sin, cos, log, isnan, isinf
from std.memory import bitcast
from std.testing import assert_true, assert_equal
from flint.ml.jax_random import JaxKey, prng_key, split_one, fold_in, random_bits32, uniform01, gumbel
from flint.ml.musiccoca import SpmUnigram, NumpyRandomState, MusicCoCaText
from flint.ml.audio_codec import (FBuf, fp, RvqCodebook, InverseStft, StreamConv2D, hann_periodic,
                                  SpectroStreamDecoder)
from flint.ml.magenta_rt import MagentaRT, MrtConditioning, discretize_cfg
from flint.ml.llm.safetensors import SafeTensors
from flint.audio.pcm_stream import PcmStreamRing
from flint.audio.server import AudioServer, BUS_MUSIC
from flint.core.jobs import JobPool
from flint.platform.os import env_now, read_bytes

comptime PI64 = 3.141592653589793


def _close(a: Float64, b: Float64, tol: Float64) -> Bool:
    return abs(a - b) <= tol


def test_jax_random() raises:
    var k = prng_key(42)
    assert_equal(Int(k.k0), 0)
    assert_equal(Int(k.k1), 42)
    var s0 = split_one(k, 3, 0)
    var s2 = split_one(k, 3, 2)
    assert_equal(Int(s0.k0), 1832780943)
    assert_equal(Int(s0.k1), 270669613)
    assert_equal(Int(s2.k0), 2465931498)
    assert_equal(Int(s2.k1), 255383827)
    var f = fold_in(k, 7)
    assert_equal(Int(f.k0), 2547012911)
    assert_equal(Int(f.k1), 1371500959)
    var bits: List[Int] = [2098992034, 2919706841, 2646866425, 2409546199]
    for i in range(4):
        assert_equal(Int(random_bits32(k, i)), bits[i])
    var u: List[Float64] = [0.48870956897735596, 0.6797971725463867, 0.6162714958190918]
    var g: List[Float64] = [0.3340933918952942, 0.952019453048706, 0.7255305647850037]
    for i in range(3):
        assert_equal(Float64(uniform01(k, i, 0.0)), u[i])
        assert_true(_close(Float64(gumbel(k, i)), g[i], 1e-6))
    print("  jax_random: threefry split / fold_in / bits / uniform / gumbel match JAX")


def test_numpy_random() raises:
    var r0 = NumpyRandomState(0)
    var w0: List[Float64] = [1.764052345967664, 0.4001572083672233, 0.9787379841057392, 2.240893199201458,
                             1.8675579901499675]
    for i in range(5):
        assert_true(_close(r0.randn(), w0[i], 1e-14))
    var r7 = NumpyRandomState(7)
    var w7: List[Float64] = [1.690525703800356, -0.4659373705408328, 0.0328201636785844, 0.40751628299650783,
                             -0.7889230286257386]
    for i in range(5):
        assert_true(_close(r7.randn(), w7[i], 1e-14))
    print("  NumpyRandomState: randn(seed 0, 7) bit-exact")


def test_rvq() raises:
    # 3 levels of 8 codewords in 4 dims, each level 10x finer than the one before
    var levels = 3
    var size = 8
    var dim = 4
    var tab = FBuf(levels * size * dim)
    var rs = NumpyRandomState(3)
    for l in range(levels):
        var scale = 1.0
        for _ in range(l):
            scale *= 0.1
        for i in range(size * dim):
            tab.set(l * size * dim + i, Float32(rs.randn() * scale))
    var cb = RvqCodebook(levels, size, dim, tab^)
    var codes: List[Int] = [5, 2, 7]
    var x = FBuf(dim)
    cb.decode(codes, x.addr)
    var back = cb.encode(x.addr, levels)
    _ = x.n   # keep x alive past the raw-address use (ASAP destruction)
    for l in range(levels):
        assert_equal(back[l], codes[l])
    print("  RVQ: decode -> encode round trip")


def _stft_frames(x: List[Float64], frames: Int, frame: Int, hop: Int) -> List[Float32]:
    """Causal periodic-Hann rfft frames (bins 0..frame/2 - 1), [frames, 2 * bins] re | im."""
    var bins = frame // 2
    var w = hann_periodic(frame)
    var out = List[Float32](length=frames * 2 * bins, fill=0.0)
    for f in range(frames):
        var start = f * hop - (frame - hop)
        for k in range(bins):
            var re: Float64 = 0
            var im: Float64 = 0
            for n in range(frame):
                var t = start + n
                if t < 0 or t >= len(x):
                    continue
                var v = x[t] * w[n]
                var ang = 2.0 * PI64 * Float64(k * n) / Float64(frame)
                re += v * cos(ang)
                im -= v * sin(ang)
            out[f * 2 * bins + k] = Float32(re)
            out[f * 2 * bins + bins + k] = Float32(im)
    return out^


def test_istft() raises:
    var pool = JobPool()
    var frame = 64
    var hop = 32
    var bins = 32
    var frames = 12
    var x = List[Float64]()
    for t in range(frames * hop):
        x.append(0.5 * sin(2.0 * PI64 * 3.0 * Float64(t) / 64.0) + 0.25 * cos(2.0 * PI64 * 5.0 * Float64(t) / 64.0 + 0.3))
    var spec = _stft_frames(x, frames, frame, hop)
    var a = InverseStft(frame, hop, frame, bins, 1)
    var ya = List[Float32](length=frames * hop, fill=0.0)
    a.step(pool, Int(spec.unsafe_ptr()), frames, Int(ya.unsafe_ptr()))
    var b = InverseStft(frame, hop, frame, bins, 1)
    var yb = List[Float32](length=frames * hop, fill=0.0)
    for f in range(frames):
        b.step(pool, Int(spec.unsafe_ptr()) + f * 2 * bins * 4, 1, Int(yb.unsafe_ptr()) + f * hop * 4)
    _ = len(spec)
    var md: Float64 = 0
    var ms: Float64 = 0
    for t in range(frames * hop):
        ms = max(ms, abs(Float64(ya[t] - yb[t])))
        # block 1 comes from the zero-padded first frame, whose leakage into the dropped
        # Nyquist bin is not reconstructed; from block 2 on every frame holds whole periods
        if t >= 2 * hop:
            md = max(md, abs(Float64(ya[t]) - x[t - hop]))
    assert_true(md < 1e-3, "iSTFT reconstruction error " + String(md))
    assert_true(ms < 1e-6, "iSTFT streaming mismatch " + String(ms))
    print("  InverseStft: reconstruction", md, "streaming", ms)


def _naive_conv(x: List[Float32], t: Int, fin: Int, cin: Int, w: List[Float32], bias: List[Float32], kt: Int, kf: Int,
                cout: Int, st: Int, sf: Int, transposed: Bool) -> List[Float32]:
    """Causal-in-time conv / transposed conv (cross-correlation of the dilated, padded input)."""
    var tout = t * st if transposed else t
    var fout = fin * sf if transposed else fin
    var plt = kt - 1
    var plf = (kf - 1) // 2
    if transposed:
        plf = kf - 1 if sf > kf - 1 else (kf + sf - 1) // 2
    var y = List[Float32](length=tout * fout * cout, fill=0.0)
    for ot in range(tout):
        for of_ in range(fout):
            for o in range(cout):
                var acc = Float64(bias[o])
                for a in range(kt):
                    for b in range(kf):
                        var dt = ot + a - plt
                        var df = of_ + b - plf
                        var it = dt
                        var if_ = df
                        if transposed:
                            if dt % st != 0 or df % sf != 0 or dt < 0 or df < 0:
                                continue
                            it = dt // st
                            if_ = df // sf
                        if it < 0 or it >= t or if_ < 0 or if_ >= fin:
                            continue
                        for c in range(cin):
                            acc += Float64(x[(it * fin + if_) * cin + c]) * Float64(w[((a * kf + b) * cin + c) * cout + o])
                y[(ot * fout + of_) * cout + o] = Float32(acc)
    return y^


def _conv_case(kt: Int, kf: Int, st: Int, sf: Int, transposed: Bool) raises:
    var pool = JobPool()
    var cin = 3
    var cout = 5
    var fin = 4
    var t = 3
    var rs = NumpyRandomState(kt * 7 + kf * 3 + st + sf)
    var w = List[Float32]()
    for _ in range(kt * kf * cin * cout):
        w.append(Float32(rs.randn()))
    var bias = List[Float32]()
    for _ in range(cout):
        bias.append(Float32(rs.randn()))
    var x = List[Float32]()
    for _ in range(t * fin * cin):
        x.append(Float32(rs.randn()))
    var ref_y = _naive_conv(x, t, fin, cin, w, bias, kt, kf, cout, st, sf, transposed)
    var kb = FBuf(len(w))
    for i in range(len(w)):
        kb.set(i, w[i])
    var bb = FBuf(cout)
    for i in range(cout):
        bb.set(i, bias[i])
    var whole = StreamConv2D(kb, bb^, kt, kf, cin, cout, st if transposed else 1, sf if transposed else 1, transposed, fin)
    var tout = whole.step(pool, Int(x.unsafe_ptr()), t)
    var n = tout * whole.fout * cout
    assert_equal(n, len(ref_y))
    var md: Float32 = 0
    for i in range(n):
        md = max(md, abs(whole.out.at(i) - ref_y[i]))
    var bb2 = FBuf(cout)
    for i in range(cout):
        bb2.set(i, bias[i])
    var stream = StreamConv2D(kb, bb2^, kt, kf, cin, cout, st if transposed else 1, sf if transposed else 1, transposed, fin)
    var ms: Float32 = 0
    var per = n // t
    for f in range(t):
        _ = stream.step(pool, Int(x.unsafe_ptr()) + f * fin * cin * 4, 1)
        for i in range(per):
            ms = max(ms, abs(stream.out.at(i) - ref_y[f * per + i]))
    _ = kb.n
    _ = len(x)
    assert_true(md < 1e-4, "conv vs naive " + String(md))
    assert_true(ms < 1e-4, "streaming conv vs naive " + String(ms))


def test_stream_conv() raises:
    _conv_case(3, 3, 1, 1, False)
    _conv_case(4, 4, 2, 2, True)
    _conv_case(3, 6, 1, 3, True)
    _conv_case(4, 3, 2, 1, True)
    print("  StreamConv2D: causal / transposed convs match the naive reference, streaming too")


def test_spm() raises:
    var pieces: List[String] = ["<unk>", "<s>", "</s>", "▁a", "▁ab", "b", "c", "▁", "abc", "▁c"]
    var scores: List[Float32] = [0, 0, 0, -1.0, -1.5, -2.0, -3.0, -4.0, -0.5, -2.5]
    var types: List[Int] = [2, 3, 3, 1, 1, 1, 1, 1, 1, 1]
    for v in range(256):
        var hex = String("0123456789ABCDEF")
        pieces.append("<0x" + String(hex[byte=v // 16]) + String(hex[byte=v % 16]) + ">")
        scores.append(0)
        types.append(6)
    var sp = SpmUnigram(pieces, scores, types)
    # "ab": "▁ab" (-1.5) beats "▁a" + "b" (-3)
    var ids = sp.encode("ab")
    assert_equal(len(ids), 1)
    assert_equal(ids[0], 4)
    # "  ab   c ": trimmed, collapsed -> "▁ab▁c"
    ids = sp.encode("  ab   c ")
    assert_equal(len(ids), 2)
    assert_equal(ids[0], 4)
    assert_equal(ids[1], 9)
    # unknown "z" -> byte fallback <0x7A> after "▁a"
    ids = sp.encode("az")
    assert_equal(len(ids), 2)
    assert_equal(ids[0], 3)
    assert_equal(ids[1], 10 + 0x7A)
    print("  SpmUnigram: Viterbi, whitespace, byte fallback")


def test_conditioning() raises:
    assert_equal(discretize_cfg(3.0, 0.2, 40), 20)
    assert_equal(discretize_cfg(-5.0, 0.2, 40), 0)
    assert_equal(discretize_cfg(9.0, 0.2, 40), 40)
    assert_equal(discretize_cfg(1.0, 1.0, 8), 2)
    var c = MrtConditioning()
    c.style[0] = 5
    var r = c.row()
    assert_equal(len(r), 144)
    assert_equal(r[0], 12)
    assert_equal(r[1], 6)          # masked style token -> the dropout token
    assert_equal(r[141], 27)       # cfg 3.0 -> bin 20
    assert_equal(r[143], 9)        # drum cfg 1.0 -> bin 2
    print("  conditioning: row layout and CFG bins")


def test_pcm_stream() raises:
    var ring = PcmStreamRing.create(300)
    assert_equal(ring.capacity, 512)
    var src = List[Float32]()
    for i in range(800):
        src.append(Float32(i))
    assert_equal(ring.write(Int(src.unsafe_ptr()), 400), 400)
    var dst = List[Float32](length=800, fill=-1.0)
    assert_equal(ring.read(Int(dst.unsafe_ptr()), 300), 300)
    assert_equal(ring.write(Int(src.unsafe_ptr()), 400), 400)   # wraps
    assert_equal(ring.buffered_frames(), 500)
    assert_equal(ring.write(Int(src.unsafe_ptr()), 400), 12)    # full
    assert_equal(ring.read(Int(dst.unsafe_ptr()), 100), 100)
    assert_equal(Int(dst[0]), 600)
    assert_equal(Int(dst[2 * 99 + 1]), 799)
    _ = len(src)
    ring.destroy()

    var srv = AudioServer(48000, 256)
    var r2 = PcmStreamRing.create(4096)
    srv.add_pcm_stream(r2, BUS_MUSIC, 0.5)
    var tone = List[Float32]()
    for i in range(1024):
        var v = Float32(0.2 * sin(2.0 * PI64 * 440.0 * Float64(i) / 48000.0))
        tone.append(v)
        tone.append(v)
    _ = r2.write(Int(tone.unsafe_ptr()), 1024)
    _ = len(tone)
    var outq = List[Float32](length=512, fill=0.0)
    var peak: Float32 = 0
    for _ in range(4):
        for j in range(512):
            outq[j] = 0
        srv.render_offline_quantum(outq, 256)
        for j in range(512):
            assert_true(not isnan(outq[j]) and not isinf(outq[j]))
            peak = max(peak, abs(outq[j]))
    assert_equal(r2.buffered_frames(), 0)
    assert_true(peak > 0.02 and peak < 0.2, "mixed stream peak " + String(peak))
    srv.render_offline_quantum(outq, 256)
    assert_equal(r2.underrun_frames(), 256)
    srv.stop()
    r2.destroy()
    print("  PcmStreamRing: wrap-around, back-pressure, AudioServer mixing, underruns")


def _u32(b: List[UInt8], i: Int) -> UInt32:
    return UInt32(b[4 * i]) | (UInt32(b[4 * i + 1]) << 8) | (UInt32(b[4 * i + 2]) << 16) | (UInt32(b[4 * i + 3]) << 24)


def test_model() raises:
    var dir = env_now("FLINT_MAGENTA_RT_DIR")
    var rdir = env_now("FLINT_MAGENTA_RT_REF")
    if dir.byte_length() == 0 or rdir.byte_length() == 0:
        print("  (set FLINT_MAGENTA_RT_DIR and FLINT_MAGENTA_RT_REF for the model parity checks)")
        return
    # --- MusicCoCa
    var coca = MusicCoCaText.load(dir + "/resources/musiccoca/musiccoca_text.safetensors")
    var lines = String(unsafe_from_utf8=read_bytes(rdir + "/musiccoca_ref.txt")).split("\n")
    var i = 0
    while i + 1 < len(lines) and lines[i].byte_length() > 0:
        var parts = lines[i].split("|")
        var mapped = lines[i].startswith("MAPPED")
        if lines[i].startswith("BLEND"):
            break
        var e: List[Float32]
        var want_ids = String("-")
        if mapped:
            e = coca.embed_style(String(parts[1]), True, Int(String(parts[0]).split(" ")[1]))
        else:
            var prompt = String(parts[0])
            var ids = coca.token_ids(prompt)
            var w = parts[1].split(" ")
            assert_equal(len(ids), len(w))
            for j in range(len(ids)):
                assert_equal(ids[j], Int(w[j]))
            e = coca.embed(prompt)
        var re = lines[i + 1].split(" ")
        var md: Float32 = 0
        for d in range(768):
            md = max(md, abs(e[d] - Float32(Float64(re[d]))))
        assert_true(md < 1e-3, "MusicCoCa embedding diff " + String(md))
        var tok = coca.tokenize(e)
        var wt = parts[2].split(" ")
        for j in range(12):
            assert_equal(tok[j], Int(wt[j]))
        _ = want_ids
        i += 2
    print("  MusicCoCa: ids, embeddings (mapper too) and RVQ style tokens match the TFLite reference")
    # --- LM token parity
    var mrt = MagentaRT.load(dir)
    var key = read_bytes(rdir + "/init_key.bin")
    mrt.reset_key(JaxKey(_u32(key, 0), _u32(key, 1)))
    var ca = read_bytes(rdir + "/cond_a.bin")
    var cb = read_bytes(rdir + "/cond_b.bin")
    var toks = read_bytes(rdir + "/tokens.bin")
    var frames = len(toks) // 48
    var switch_f = Int(env_now("FLINT_MAGENTA_RT_SWITCH")) if env_now("FLINT_MAGENTA_RT_SWITCH").byte_length() > 0 else 50
    var mism = 0
    var all_codes = List[List[Int]]()
    for f in range(frames):
        var row = List[Int]()
        for j in range(144):
            row.append(Int(_u32(ca, j)) if f < switch_f else Int(_u32(cb, j)))
        var codes = mrt.lm.step(row)
        for l in range(12):
            if codes[l] != Int(_u32(toks, f * 12 + l)):
                mism += 1
        all_codes.append(codes^)
    assert_equal(mism, 0)
    print("  LM:", frames, "frames, 0 token mismatches vs JAX (style switch at frame", switch_f, ")")
    # --- codec parity (reference codes)
    var audio = read_bytes(rdir + "/audio.bin")
    var pcm = FBuf(1920 * 2)
    var md: Float32 = 0
    for f in range(frames):
        var codes = List[Int]()
        for l in range(12):
            codes.append(Int(_u32(toks, f * 12 + l)))
        mrt.decode(codes, pcm.addr)
        for j in range(3840):
            md = max(md, abs(pcm.at(j) - bitcast[DType.float32](_u32(audio, f * 3840 + j))))
    assert_true(md < 5e-4, "SpectroStream diff " + String(md))
    print("  SpectroStream: max abs diff", md, "over", frames, "frames")
    if env_now("FLINT_MAGENTA_RT_GPU") == "1":
        var name = mrt.enable_gpu_codec()
        mrt.reset_key(JaxKey(_u32(key, 0), _u32(key, 1)))
        var gd: Float32 = 0
        for f in range(frames):
            var codes = List[Int]()
            for l in range(12):
                codes.append(Int(_u32(toks, f * 12 + l)))
            mrt.decode(codes, pcm.addr)
            for j in range(3840):
                gd = max(gd, abs(pcm.at(j) - bitcast[DType.float32](_u32(audio, f * 3840 + j))))
        assert_true(gd < 5e-4, "GPU SpectroStream diff " + String(gd))
        print("  SpectroStream on", name, ": max abs diff", gd)


def main() raises:
    test_jax_random()
    test_numpy_random()
    test_rvq()
    test_istft()
    test_stream_conv()
    test_spm()
    test_conditioning()
    test_pcm_stream()
    test_model()
    print("test_flint_v5_magenta_rt: all passed")
