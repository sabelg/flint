"""Flint v5 Audio2Face-3D suite (`test_flint_v5_audio2face.mojo`).

Verifies, without model files:
  1. Audio front end: closed-track frame counts and SDK windows (offset, zero padding, strength).
  2. `Npz`: a stored zip of `<f4`, `<i4`, `<f8` and `|S` members written here, read back.
  3. `bvls`: box-bounded least squares on separable and coupled problems against known optima.
  4. `rigid_fit`: a rotated + translated point set recovers its column-major 4x4.
  5. New ONNX kernels: InstanceNormalization, ReduceMax, a GRU step, per-channel broadcasts.
  6. ARKit morph-name matching (`blendShape1.jawOpen`, case) and `A2fClip` sampling.
  7. With `A2F_DIR` (a v2.3 / v2.3.1 model folder) and `A2F_REF` (prefix written by
     `tools/audio2face/reference.py`): network, ARKit, tongue, jaw and eye outputs against the
     reference; streaming in 0.37 s chunks gives the same clip as one closed-track call.
"""
from std.math import sqrt, exp, tanh, abs, sin, cos
from std.testing import assert_true, assert_equal
from flint.platform.os import env_now, read_bytes, write_bytes
from flint.core.jobs import JobPool
from flint.ml.tensor import Tensor, f32_from_bits
from flint.ml.npz import Npz
from flint.ml.ops import B_ADD, B_MUL
from flint.ml.ops_more import instance_norm, reduce_max, gru, chan_binary_applies, par_chan_binary
from flint.ml.audio2face import (
    a2f_frame_count, a2f_target_sample, a2f_window, bvls, rigid_fit, morph_name_matches, morph_map,
    A2fClip, Audio2Face,
)


def _env(name: String, fallback: String) -> String:
    var v = env_now(name)
    return v if v.byte_length() > 0 else fallback


def _near(a: Float32, b: Float32, tol: Float32) -> Bool:
    return abs(a - b) <= tol


# ----------------------------------------------------------------------------- 1. front end

def test_front_end() raises:
    assert_equal(a2f_frame_count(64000, 30), 120)
    assert_equal(a2f_frame_count(16000, 30), 30)
    assert_equal(a2f_frame_count(16001, 30), 31)
    assert_equal(a2f_frame_count(1, 30), 1)
    assert_equal(a2f_frame_count(0, 30), 0)
    assert_equal(a2f_frame_count(16000, 60), 60)
    assert_equal(a2f_target_sample(3, 30), 1600)
    assert_equal(a2f_target_sample(1, 60), 266)
    var audio = List[Float32]()
    for i in range(100):
        audio.append(Float32(i + 1))
    var out = List[Float32](length=20, fill=-1.0)
    # frame 0 at 30 fps targets sample 0: window [-4, 6) with offset 4
    a2f_window(audio, 0, 30, 10, 4, 2.0, out, 5)
    for j in range(5):
        assert_true(out[j] == -1.0)
    for j in range(4):
        assert_true(out[5 + j] == 0.0)
    for j in range(6):
        assert_true(out[9 + j] == Float32(2 * (j + 1)))
    assert_true(out[15] == -1.0)
    # past the end: zero padded
    var tail = List[Float32](length=10, fill=-1.0)
    a2f_window(audio, 0, 30, 10, -95, 1.0, tail, 0)
    for j in range(5):
        assert_true(tail[j] == Float32(96 + j))
    for j in range(5, 10):
        assert_true(tail[j] == 0.0)
    print("  front end: frame counts and windows ok")


# ----------------------------------------------------------------------------- 2. npz

def _u16(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 255))
    b.append(UInt8((v >> 8) & 255))


def _u32(mut b: List[UInt8], v: Int):
    for i in range(4):
        b.append(UInt8((v >> (8 * i)) & 255))


def _str(mut b: List[UInt8], s: String):
    for c in s.as_bytes():
        b.append(c)


def _npy(descr: String, shape: String, data: List[UInt8]) -> List[UInt8]:
    var h = String("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape + ", }"
    var hb = h.byte_length() + 1
    var pad = (64 - (10 + hb) % 64) % 64
    var b = List[UInt8]()
    b.append(0x93)
    _str(b, "NUMPY")
    b.append(1)
    b.append(0)
    _u16(b, hb + pad)
    _str(b, h)
    for _ in range(pad):
        b.append(0x20)
    b.append(0x0A)
    for v in data:
        b.append(v)
    return b^


def _zip(names: List[String], members: List[List[UInt8]]) -> List[UInt8]:
    var z = List[UInt8]()
    var offsets = List[Int]()
    for i in range(len(names)):
        offsets.append(len(z))
        _u32(z, 0x04034B50)
        _u16(z, 20)
        _u16(z, 0)
        _u16(z, 0)       # stored
        _u32(z, 0)       # time / date
        _u32(z, 0)       # crc (not checked)
        _u32(z, len(members[i]))
        _u32(z, len(members[i]))
        _u16(z, names[i].byte_length())
        _u16(z, 0)
        _str(z, names[i])
        for v in members[i]:
            z.append(v)
    var cd = len(z)
    for i in range(len(names)):
        _u32(z, 0x02014B50)
        _u16(z, 20)
        _u16(z, 20)
        _u16(z, 0)
        _u16(z, 0)
        _u32(z, 0)
        _u32(z, 0)
        _u32(z, len(members[i]))
        _u32(z, len(members[i]))
        _u16(z, names[i].byte_length())
        _u16(z, 0)
        _u16(z, 0)
        _u16(z, 0)
        _u16(z, 0)
        _u32(z, 0)
        _u32(z, offsets[i])
        _str(z, names[i])
    var cdlen = len(z) - cd
    _u32(z, 0x06054B50)
    _u16(z, 0)
    _u16(z, 0)
    _u16(z, len(names))
    _u16(z, len(names))
    _u32(z, cdlen)
    _u32(z, cd)
    _u16(z, 0)
    return z^


def _f32_bytes(vals: List[Float32]) -> List[UInt8]:
    var b = List[UInt8]()
    for v in vals:
        var u = Int(SIMD[DType.float32, 1](v).to_bits())
        _u32(b, u)
    return b^


def test_npz() raises:
    var names = List[String]()
    var members = List[List[UInt8]]()
    var fv: List[Float32] = [1.5, -2.0, 3.25, 0.0, 1e-3, 42.0]
    names.append("a.npy")
    members.append(_npy("<f4", "(2, 3)", _f32_bytes(fv)))
    var ib = List[UInt8]()
    _u32(ib, 7)
    _u32(ib, 0xFFFFFFFF)  # -1
    _u32(ib, 10085)
    names.append("idx.npy")
    members.append(_npy("<i4", "(3,)", ib))
    var db = List[UInt8]()
    # 0.5 and -3.0 as float64
    for bits in [UInt64(0x3FE0000000000000), UInt64(0xC008000000000000)]:
        for i in range(8):
            db.append(UInt8(Int((bits >> UInt64(8 * i)) & 255)))
    names.append("d.npy")
    members.append(_npy("<f8", "(2,)", db))
    var sb = List[UInt8]()
    _str(sb, "neutral")
    sb.append(0)
    _str(sb, "jawOpen")
    sb.append(0)
    names.append("poseNames.npy")
    members.append(_npy("|S8", "(2,)", sb))
    var path = _env("TMPDIR", "/tmp") + "/flint_a2f_test.npz"
    var z = _zip(names, members)
    assert_true(write_bytes(path, z))
    var npz = Npz.open(path)
    assert_true(npz.has("a") and npz.has("idx") and npz.has("d") and npz.has("poseNames"))
    assert_true(not npz.has("missing"))
    var sh = npz.shape("a")
    assert_equal(len(sh), 2)
    assert_equal(sh[0], 2)
    assert_equal(sh[1], 3)
    var a = npz.floats("a")
    for i in range(6):
        assert_true(a.fget(i) == fv[i])
    var idx = npz.ints("idx")
    assert_equal(idx[0], 7)
    assert_equal(idx[1], -1)
    assert_equal(idx[2], 10085)
    var d = npz.floats("d")
    assert_true(d.fget(0) == 0.5 and d.fget(1) == -3.0)
    var s = npz.strings("poseNames")
    assert_equal(len(s), 2)
    assert_true(s[0] == "neutral" and s[1] == "jawOpen")
    _ = a^
    print("  npz: stored zip, f4 / i4 / f8 / S members ok")


# ----------------------------------------------------------------------------- 3. bvls

def test_bvls() raises:
    # separable: A = 2 I, b = (4, -2, 1) -> unconstrained (2, -1, 0.5) -> clipped (1, 0, 0.5)
    var a: List[Float32] = [2, 0, 0, 0, 2, 0, 0, 0, 2]
    var b: List[Float32] = [4, -2, 1]
    var lo = List[Float32](length=3, fill=0.0)
    var hi = List[Float32](length=3, fill=1.0)
    var x = bvls(a, b, lo, hi, 3)
    assert_true(_near(x[0], 1, 1e-6) and _near(x[1], 0, 1e-6) and _near(x[2], 0.5, 1e-6))
    # coupled 2x2: min |A x - b|, A = [[1, 1], [0, 1]], b = (1.5, -1): with x1 = 0 -> x0 = 1
    var a2: List[Float32] = [1, 1, 0, 1]
    var b2: List[Float32] = [1.5, -1]
    var x2 = bvls(a2, b2, List[Float32](length=2, fill=0.0), List[Float32](length=2, fill=1.0), 2)
    assert_true(_near(x2[0], 1, 1e-6) and _near(x2[1], 0, 1e-6))
    # interior optimum is returned unchanged
    var b3: List[Float32] = [0.7, 0.2]
    var x3 = bvls(a2, b3, List[Float32](length=2, fill=0.0), List[Float32](length=2, fill=1.0), 2)
    assert_true(_near(x3[0], 0.5, 1e-6) and _near(x3[1], 0.2, 1e-6))
    # brute-force check of a coupled 3x3 against a fine grid
    var a4: List[Float32] = [3, 1, 0.5, 1, 2, -0.3, 0.5, -0.3, 1.5]
    var b4: List[Float32] = [2.0, -0.4, 1.9]
    var x4 = bvls(a4, b4, List[Float32](length=3, fill=0.0), List[Float32](length=3, fill=1.0), 3)
    var best: Float32 = 3.4e38
    var steps = 40
    for i in range(steps + 1):
        for j in range(steps + 1):
            for k in range(steps + 1):
                var v: List[Float32] = [Float32(i) / Float32(steps), Float32(j) / Float32(steps), Float32(k) / Float32(steps)]
                best = min(best, _res(a4, b4, v))
    assert_true(_res(a4, b4, x4) <= best + 1e-5)
    print("  bvls: separable, coupled, interior and grid-checked problems ok")


def _res(a: List[Float32], b: List[Float32], x: List[Float32]) -> Float32:
    var s: Float32 = 0
    for r in range(3):
        var t: Float32 = -b[r]
        for c in range(3):
            t += a[r * 3 + c] * x[c]
        s += t * t
    return s


# ----------------------------------------------------------------------------- 4. rigid fit

def test_rigid_fit() raises:
    var src: List[Float32] = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 1, 1, 1]
    # rotation 30 deg about (1, 2, 2)/3, then translation (0.5, -1, 2)
    var ax: List[Float64] = [1.0 / 3, 2.0 / 3, 2.0 / 3]
    var th = 0.5235987755982988
    var c = cos(th)
    var s = sin(th)
    var r = List[Float64](length=9, fill=0.0)
    for i in range(3):
        for j in range(3):
            r[i * 3 + j] = (1 - c) * ax[i] * ax[j] + (c if i == j else 0.0)
    r[1] -= s * ax[2]
    r[2] += s * ax[1]
    r[3] += s * ax[2]
    r[5] -= s * ax[0]
    r[6] -= s * ax[1]
    r[7] += s * ax[0]
    var t: List[Float64] = [0.5, -1, 2]
    var dst = List[Float32](length=15, fill=0.0)
    for p in range(5):
        for i in range(3):
            var v = t[i]
            for j in range(3):
                v += r[i * 3 + j] * Float64(src[p * 3 + j])
            dst[p * 3 + i] = Float32(v)
    var m = rigid_fit(src, dst, 5)
    for i in range(3):
        for j in range(3):
            assert_true(_near(m[j * 4 + i], Float32(r[i * 3 + j]), 1e-5))
        assert_true(_near(m[12 + i], Float32(t[i]), 1e-5))
        assert_true(m[i * 4 + 3] == 0)
    assert_true(m[15] == 1)
    # identity
    var e = rigid_fit(src, src, 5)
    for i in range(16):
        assert_true(_near(e[i], Float32(1.0) if i % 5 == 0 else Float32(0.0), 1e-6))
    print("  rigid fit: rotation + translation recovered ok")


# ----------------------------------------------------------------------------- 5. ops

def test_ops() raises:
    var pool = JobPool()
    _ = pool.start(2)
    # InstanceNorm on [1, 2, 4]
    var x = Tensor.floats([1, 2, 4], [1, 2, 3, 4, 10, 10, 10, 10])
    var y = instance_norm(x, Tensor.floats([2], [2, 1]), Tensor.floats([2], [0.5, -1]), 1e-5)
    var sd = Float32(sqrt(1.25 + 1e-5))
    assert_true(_near(y.fget(0), 2 * (1 - 2.5) / sd + 0.5, 1e-5))
    assert_true(_near(y.fget(3), 2 * (4 - 2.5) / sd + 0.5, 1e-5))
    assert_true(_near(y.fget(5), -1, 1e-5))
    # ReduceMax over axis 1 of [2, 3]
    var m = reduce_max(Tensor.floats([2, 3], [1, 5, -2, -7, -3, -9]), [1], False)
    assert_equal(m.numel(), 2)
    assert_true(m.fget(0) == 5 and m.fget(1) == -3)
    # one GRU step, I = 1, H = 1, gates z r h
    var w = Tensor.floats([1, 3, 1], [0.5, -0.4, 0.9])
    var rr = Tensor.floats([1, 3, 1], [0.3, 0.2, -0.6])
    var bb = Tensor.floats([1, 6], [0.1, 0.0, -0.2, 0.05, 0.1, 0.3])
    var h0 = Tensor.floats([1, 1, 1], [0.4])
    var xin: Float32 = 0.8
    var outs = gru(pool, Tensor.floats([1, 1, 1], [xin]), w, rr, bb, True, h0, True, 1, "forward", False, 0)
    var z = _sig(0.5 * xin + 0.3 * 0.4 + 0.1 + 0.05)
    var rg = _sig(-0.4 * xin + 0.2 * 0.4 + 0.0 + 0.1)
    var hh = Float32(tanh(Float64(0.9 * xin + -0.6 * (rg * 0.4) + 0.3 - 0.2)))
    var want = (1 - z) * hh + z * 0.4
    assert_true(_near(outs[0].fget(0), want, 1e-6))
    assert_true(_near(outs[1].fget(0), want, 1e-6))
    # per-channel broadcast [C, 1] against [B, C, T] (T large enough for the parallel path)
    var tl = 4096
    var big = Tensor.zeros([2, 3, tl])
    for i in range(2 * 3 * tl):
        big.f[i] = Float32(i % 1000)
    var ch = Tensor.floats([3, 1], [10, 20, 30])
    assert_true(chan_binary_applies(B_ADD, big, ch))
    var sum = par_chan_binary(pool, B_ADD, big, ch)
    var prod = par_chan_binary(pool, B_MUL, ch, big)
    for bi in range(2):
        for c in range(3):
            for t in range(tl):
                var k = (bi * 3 + c) * tl + t
                assert_true(sum.fget(k) == Float32(k % 1000) + Float32(10 * (c + 1)))
                assert_true(prod.fget(k) == Float32(k % 1000) * Float32(10 * (c + 1)))
    pool.stop()
    print("  ops: InstanceNorm, ReduceMax, GRU step, channel broadcast ok")


def _sig(v: Float32) -> Float32:
    return Float32(1.0 / (1.0 + exp(-Float64(v))))


# ----------------------------------------------------------------------------- 6. morphs + clip

def test_morphs_and_clip() raises:
    assert_true(morph_name_matches("jawOpen", "jawOpen"))
    assert_true(morph_name_matches("JAWOPEN", "jawOpen"))
    assert_true(morph_name_matches("blendShape1.jawOpen", "jawOpen"))
    assert_true(morph_name_matches("ARKit_mouthSmileLeft", "mouthSmileLeft"))
    assert_true(not morph_name_matches("myjawOpen", "jawOpen"))
    assert_true(not morph_name_matches("jawOpenWide", "jawOpen"))
    var map = morph_map(["Face.eyeBlinkLeft", "Basis", "jawopen"], ["eyeBlinkLeft", "jawOpen"])
    assert_true(map[0] == 0 and map[1] == -1 and map[2] == 1)
    var c = A2fClip()
    c.fps = 10
    c.frames = 2
    c.skin_names = ["a", "b"]
    c.skin = [0, 1, 1, 0]
    var w = c.sample_skin(0.05)
    assert_true(_near(w[0], 0.5, 1e-6) and _near(w[1], 0.5, 1e-6))
    var e = c.sample_skin(9.0)
    assert_true(e[0] == 1 and e[1] == 0)
    assert_true(c.pose_index("b") == 1 and c.pose_index("zz") == -1)
    print("  morph names and clip sampling ok")


# ----------------------------------------------------------------------------- 7. model

def _readf(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=0.0)
    for i in range(n):
        out[i] = f32_from_bits(UInt32(b[4 * i]) | (UInt32(b[4 * i + 1]) << 8) | (UInt32(b[4 * i + 2]) << 16) | (UInt32(b[4 * i + 3]) << 24))
    return out^


def _maxdiff(a: List[Float32], b: List[Float32]) -> Float32:
    if len(a) != len(b):
        return 3.4e38
    var m: Float32 = 0
    for i in range(len(a)):
        m = max(m, abs(a[i] - b[i]))
    return m


def test_model() raises:
    var dir = _env("A2F_DIR", "")
    var want = _env("A2F_REF", "")
    if dir.byte_length() == 0 or want.byte_length() == 0:
        print("  model: skipped (set A2F_DIR and A2F_REF to run)")
        return
    var a2f = Audio2Face.load(dir)
    var audio = _readf(want + ".wav16.f32")
    var clip = a2f.process(audio)
    var dn = _maxdiff(clip.net, _readf(want + ".net.f32"))
    var ds = _maxdiff(clip.skin, _readf(want + ".skin.f32"))
    var dt = _maxdiff(clip.tongue, _readf(want + ".tongue.f32"))
    var dj = _maxdiff(clip.jaw, _readf(want + ".jaw.f32"))
    var de = _maxdiff(clip.eyes, _readf(want + ".eyes.f32"))
    print("  model on", a2f.gpu_name(), ":", clip.frames, "frames; max diff net", dn, "skin", ds, "tongue", dt,
          "jaw", dj, "eyes", de, "; RTF", (Float64(a2f.t_net + a2f.t_post) / 1e9) / clip.seconds())
    assert_true(dn < 1e-3)
    assert_true(ds < 1e-2)   # near-ties in the active set; typical 1e-5
    assert_true(dt < 1e-4)
    assert_true(dj < 2e-3)
    assert_true(de < 1e-4)
    # streaming equals the closed track
    a2f.stream_reset()
    var acc = A2fClip()
    var chunk = 5920
    var at = 0
    while at < len(audio):
        var part = List[Float32]()
        for i in range(at, min(len(audio), at + chunk)):
            part.append(audio[i])
        acc.append(a2f.stream_push(part))
        at += chunk
    acc.append(a2f.stream_finish())
    assert_equal(acc.frames, clip.frames)
    var dsn = _maxdiff(acc.net, clip.net)
    var dss = _maxdiff(acc.skin, clip.skin)
    print("  model: streaming vs closed track max diff net", dsn, "skin", dss)
    assert_true(dsn < 1e-4)
    assert_true(dss < 1e-2)


def main() raises:
    print("=== Flint v5 Audio2Face Suite ===")
    test_front_end()
    test_npz()
    test_bvls()
    test_rigid_fit()
    test_ops()
    test_morphs_and_clip()
    test_model()
    print("=== ALL FLINT v5 AUDIO2FACE TESTS PASSED ===")
