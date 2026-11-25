"""Flint v5 2-D art tools suite: Marigold normals + MI-GAN inpainting (`test_flint_v5_marigold_migan.mojo`).

Always (no weights; expected values are PyTorch 2.15 / diffusers 0.43 numbers):
  1. `ops_conv2d.conv2d_nchw` (ONNX Conv on [N, C, H, W]) against a direct reference: depthwise
     3 x 3 and 4 x 4 / stride 2 (MI-GAN's FIR filters), depthwise with a channel multiplier,
     grouped dense 3 x 3 with stride 2, dilation 2 and asymmetric padding, point-wise 1 x 1 + bias.
  2. `ops_image` LeakyRelu / Clip / Resize (nearest, linear in three coordinate modes) / Pad are
     bit-identical to the generic interpreter ops; Cast to uint8 truncates and saturates.
  3. Zero-terminal-SNR DDIM (Marigold's scheduler config): trailing timesteps for 1 / 4 / 10
     steps and rescaled `alphas_cumprod` against diffusers; one v-prediction step at t = 999.
  4. `F.interpolate(mode="bilinear")` with and without antialias, and nearest to an odd size
     (the UNet's up-sampling for latents that are not a multiple of 8).
  5. Normals head: normalisation, ensembling (closest / mean, uncertainty), PNG encoding.
  6. Seamless tiling helpers: roll round trip, seam-cross mask, wrap-seam step ratio.
With FLINT_MIGAN=migan.onnx (tools/migan/fetch.sh): one inpainting pass and `make_tileable` on a
synthetic texture; with FLINT_MIGAN_REF (tools/migan/reference.py output) also parity with
onnxruntime. With FLINT_MARIGOLD_DIR (tools/marigold/fetch.sh) and FLINT_MARIGOLD_REF
(tools/marigold/reference.py ... --res 304 --steps 4 --name a): parity with diffusers.
FLINT_MARIGOLD_GPU=0 forces the CPU.
"""
from std.math import sqrt, sin, cos, acos
from std.os import getenv
from std.testing import assert_true, assert_equal
from flint.platform.os import read_bytes
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.tensor import Tensor, DT_U8, DT_I64, fptr_at
from flint.ml.ops import unary, clip, cast, resize_nearest, U_LEAKY
from flint.ml.ops_ext import resize_linear, pad
from flint.ml.ops_conv2d import conv2d_nchw
from flint.ml.ops_image import par_leaky_relu, par_clip, resize2d, pad2d_constant
from flint.ml.diffusion.schedulers import Scheduler, SCHED_DDIM
from flint.ml.diffusion.rng import TorchGenerator
from flint.ml.diffusion.nn import Runtime
from flint.ml.diffusion.unet import resize_nearest_act
from flint.ml.diffusion.pipeline import nhwc_to_nchw
from flint.ml.marigold import (
    resize_bilinear, normalize_normals, ensemble_normals, normals_to_pixels, MarigoldNormals,
    ENSEMBLE_CLOSEST, ENSEMBLE_MEAN,
)
from flint.ml.migan import MiGan, roll_pixels, seam_cross_mask, wrap_seam_error, make_tileable


def _close(a: Float64, b: Float64, tol: Float64, what: String) raises:
    if abs(a - b) > tol:
        raise Error(what + ": got " + String(a) + " expected " + String(b))


def _rand(n: Int, seed: Int) -> List[Float32]:
    var g = TorchGenerator(seed)
    return g.randn(n)


def _maxdiff(a: Tensor, b: Tensor) -> Float64:
    var m: Float64 = 0
    for i in range(a.numel()):
        m = max(m, abs(Float64(a.fget(i)) - Float64(b.fget(i))))
    return m


def _conv_ref(x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pads: List[Int], strides: List[Int], dils: List[Int],
              group: Int) -> Tensor:
    var N = x.shape[0]
    var C = x.shape[1]
    var H = x.shape[2]
    var W = x.shape[3]
    var CO = w.shape[0]
    var CG = w.shape[1]
    var KH = w.shape[2]
    var KW = w.shape[3]
    var HO = (H + pads[0] + pads[2] - dils[0] * (KH - 1) - 1) // strides[0] + 1
    var WO = (W + pads[1] + pads[3] - dils[1] * (KW - 1) - 1) // strides[1] + 1
    var out = Tensor.zeros([N, CO, HO, WO])
    var cog = CO // group
    for n in range(N):
        for co in range(CO):
            var g = co // cog
            for oy in range(HO):
                for ox in range(WO):
                    var acc: Float64 = Float64(b.fget(co)) if has_b else 0.0
                    for ci in range(CG):
                        for ky in range(KH):
                            for kx in range(KW):
                                var iy = oy * strides[0] - pads[0] + ky * dils[0]
                                var ix = ox * strides[1] - pads[1] + kx * dils[1]
                                if iy >= 0 and iy < H and ix >= 0 and ix < W:
                                    acc += Float64(x.fget(((n * C + g * CG + ci) * H + iy) * W + ix)) * Float64(
                                        w.fget(((co * CG + ci) * KH + ky) * KW + kx))
                    out.f[((n * CO + co) * HO + oy) * WO + ox] = Float32(acc)
    return out^


def _conv_case(mut pool: JobPool, n: Int, c: Int, h: Int, w: Int, co: Int, k: Int, group: Int, pads: List[Int],
               strides: List[Int], dils: List[Int], has_b: Bool, seed: Int, what: String) raises:
    var x = Tensor.floats([n, c, h, w], _rand(n * c * h * w, seed))
    var wt = Tensor.floats([co, c // group, k, k], _rand(co * (c // group) * k * k, seed + 1))
    var b = Tensor.floats([co], _rand(co, seed + 2))
    var got = conv2d_nchw(pool, x, wt, b, has_b, pads, strides, dils, group)
    var want = _conv_ref(x, wt, b, has_b, pads, strides, dils, group)
    assert_equal(len(got.shape), 4)
    for d in range(4):
        assert_equal(got.shape[d], want.shape[d])
    var d = _maxdiff(got, want)
    if d > 2e-4:
        raise Error("conv2d " + what + ": maxdiff " + String(d))


def test_conv2d() raises:
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    _conv_case(pool, 1, 8, 13, 17, 8, 3, 8, [1, 1, 1, 1], [1, 1], [1, 1], True, 1, "depthwise 3x3")
    _conv_case(pool, 2, 6, 16, 12, 6, 4, 6, [1, 1, 1, 1], [2, 2], [1, 1], False, 2, "depthwise 4x4 s2")
    _conv_case(pool, 1, 4, 11, 9, 4, 4, 4, [0, 0, 0, 0], [1, 1], [1, 1], False, 3, "depthwise 4x4 valid")
    _conv_case(pool, 1, 3, 10, 10, 6, 3, 3, [1, 1, 1, 1], [1, 1], [1, 1], True, 4, "depthwise x2 multiplier")
    _conv_case(pool, 1, 8, 15, 14, 6, 3, 2, [2, 1, 0, 3], [2, 2], [2, 2], True, 5, "grouped 3x3 s2 d2 asym")
    _conv_case(pool, 2, 16, 9, 7, 24, 1, 1, [0, 0, 0, 0], [1, 1], [1, 1], True, 6, "pointwise 1x1")
    _conv_case(pool, 1, 5, 8, 8, 7, 1, 1, [1, 1, 1, 1], [2, 2], [1, 1], True, 7, "1x1 padded s2")
    pool.stop()
    print("✓ conv2d (depthwise / grouped / dense / point-wise) vs direct reference")


def test_image_ops() raises:
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var n = 2 * 3 * 37 * 41
    var x = Tensor.floats([2, 3, 37, 41], _rand(n, 11))
    assert_true(_maxdiff(par_leaky_relu(pool, x, 0.2), unary(U_LEAKY, x, 0.2)) == 0.0)
    assert_true(_maxdiff(par_clip(pool, x, -0.5, 0.75), clip(x, -0.5, 0.75)) == 0.0)
    for m in range(3):
        var mode = String("pytorch_half_pixel") if m == 0 else (String("half_pixel") if m == 1 else String("align_corners"))
        var sc = List[Float64]()
        for _ in range(4):
            sc.append(0.0)
        var a = resize2d(pool, x, 64, 29, True, mode, 0.0, 0.0)
        var b = resize_linear(x, [2, 3, 64, 29], sc, mode)
        if _maxdiff(a, b) != 0.0:
            raise Error("resize2d linear " + mode + " differs from resize_linear")
    var nn = resize2d(pool, x, 74, 20, False, "asymmetric", 0.0, 0.0)
    assert_true(_maxdiff(nn, resize_nearest(x, [2, 3, 74, 20])) == 0.0)
    var p1 = pad2d_constant(pool, x, 2, 0, 3, 1, 0.5)
    var p2 = pad(x, [0, 0, 2, 0, 0, 0, 3, 1], List[Int](), "constant", 0.5)
    assert_true(_maxdiff(p1, p2) == 0.0)
    # Cast float -> uint8: truncation toward zero, saturation
    var f = Tensor.floats([5], [Float32(-3.0), 0.9, 127.99, 255.0, 300.5])
    var u = cast(f, DT_U8)
    assert_equal(u.dtype, DT_U8)
    assert_equal(u.i[0], 0)
    assert_equal(u.i[1], 0)
    assert_equal(u.i[2], 127)
    assert_equal(u.i[3], 255)
    assert_equal(u.i[4], 255)
    pool.stop()
    print("✓ parallel LeakyRelu / Clip / Resize / Pad bit-identical; Cast -> uint8")


comptime MARIGOLD_SCHED = '{"_class_name": "DDIMScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085, "clip_sample": false, "num_train_timesteps": 1000, "prediction_type": "v_prediction", "rescale_betas_zero_snr": true, "set_alpha_to_one": false, "steps_offset": 1, "timestep_spacing": "trailing"}'


def test_zero_snr_scheduler() raises:
    var s = Scheduler.from_config(List[UInt8](String(MARIGOLD_SCHED).as_bytes()), SCHED_DDIM)
    # diffusers 0.43 DDIMScheduler.from_pretrained(prs-eth/marigold-normals-v1-1)
    _close(Float64(s.alphas_cumprod[0]), 0.999149978, 1e-8, "ac[0]")
    _close(Float64(s.alphas_cumprod[1]), 0.998233497, 1e-8, "ac[1]")
    _close(Float64(s.alphas_cumprod[250]), 0.652456701, 2e-7, "ac[250]")
    _close(Float64(s.alphas_cumprod[500]), 0.241018534, 2e-7, "ac[500]")
    _close(Float64(s.alphas_cumprod[998]), 1.96796279e-07, 2e-9, "ac[998]")
    assert_true(s.alphas_cumprod[999] == 0.0)
    _close(Float64(s.final_alpha_cumprod), 0.999149978, 1e-8, "final alpha")
    s.set_timesteps(1)
    assert_equal(Int(s.timestep(0)), 999)
    s.set_timesteps(4)
    assert_equal(Int(s.timestep(0)), 999)
    assert_equal(Int(s.timestep(1)), 749)
    assert_equal(Int(s.timestep(2)), 499)
    assert_equal(Int(s.timestep(3)), 249)
    s.set_timesteps(10)
    assert_equal(Int(s.timestep(1)), 899)
    assert_equal(Int(s.timestep(9)), 99)
    # v-prediction at zero SNR: x0 = -v, eps = x -> x_prev = sqrt(a_prev) (-v) + sqrt(1 - a_prev) x
    s.set_timesteps(4)
    var x = List[Float32]()
    x.append(0.5)
    x.append(-1.25)
    var v = List[Float32]()
    v.append(0.25)
    v.append(2.0)
    var g = TorchGenerator(0)
    var o = s.step(v, 0, x, g)
    var ap = Float64(s.alphas_cumprod[749])
    _close(Float64(o[0]), -sqrt(ap) * 0.25 + sqrt(1.0 - ap) * 0.5, 1e-6, "v-step[0]")
    _close(Float64(o[1]), -sqrt(ap) * 2.0 - sqrt(1.0 - ap) * 1.25, 1e-6, "v-step[1]")
    print("✓ zero-terminal-SNR v-prediction DDIM (trailing) matches diffusers")


def test_resampling() raises:
    var x = List[Float32]()
    for i in range(35):
        x.append(Float32((i * 7) % 11) * 0.125)
    # torch F.interpolate(x[1, 1, 7, 5], (3, 4), mode="bilinear", antialias=True)
    var aa = resize_bilinear(x, 7, 5, 1, 3, 4, True)
    var e_aa = [0.3593749701976776, 0.7090773582458496, 0.5840773582458496, 0.8683034777641296, 0.6452205777168274,
                0.49325981736183167, 0.6648284196853638, 0.5533087849617004, 0.7924107313156128, 0.5773809552192688,
                0.5096726417541504, 0.662946343421936]
    for i in range(12):
        _close(Float64(aa[i]), e_aa[i], 1e-6, "bilinear aa " + String(i))
    # torch F.interpolate(x, (11, 9), mode="bilinear")
    var bl = resize_bilinear(x, 7, 5, 1, 11, 9, False)
    _close(Float64(bl[0]), 0.0, 1e-7, "bilinear [0]")
    _close(Float64(bl[1]), 0.2916666865348816, 1e-6, "bilinear [1]")
    _close(Float64(bl[10]), 0.40530306100845337, 1e-6, "bilinear [10]")
    _close(Float64(bl[37]), 0.42424237728118896, 1e-6, "bilinear [37]")
    _close(Float64(bl[50]), 0.847222089767456, 1e-6, "bilinear [50]")
    _close(Float64(bl[98]), 0.875, 1e-6, "bilinear [98]")
    # nearest 8 -> 15 (F.interpolate(size=15, mode="nearest")): the UNet's odd up-sampling
    var rt = Runtime(False)
    var row = List[Float32]()
    for i in range(8):
        row.append(Float32(i))
    var a = rt.upload(1, 1, 8, 1, row)
    var r = resize_nearest_act(rt, a, 1, 15)
    var got = rt.download(r)
    var e_nn = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    for i in range(15):
        assert_equal(Int(got[i]), e_nn[i])
    print("✓ bilinear (antialias / plain) and odd-size nearest match torch")


def test_normals_head() raises:
    var n = List[Float32]()
    n.append(3.0)
    n.append(0.0)
    n.append(4.0)
    n.append(0.0)
    n.append(0.0)
    n.append(0.0)
    normalize_normals(n)
    _close(Float64(n[0]), 0.6, 1e-6, "normalize x")
    _close(Float64(n[2]), 0.8, 1e-6, "normalize z")
    assert_true(n[3] == 0.0 and n[5] == 0.0)
    # three members, one pixel: two near +Z, one tilted; closest = member 1
    var members = List[List[Float32]]()
    var raw = [[Float32(0.1), 0.0, 1.0], [Float32(0.0), 0.0, 1.0], [Float32(0.8), 0.0, 0.6]]
    for k in range(3):
        var m = List[Float32]()
        for j in range(3):
            m.append(raw[k][j])
        normalize_normals(m)
        members.append(m^)
    var t = ensemble_normals(members, ENSEMBLE_CLOSEST, True)
    var mean = List[Float32]()
    for j in range(3):
        mean.append((members[0][j] + members[1][j] + members[2][j]) / 3.0)
    normalize_normals(mean)
    var best = 0
    var bs: Float32 = -2
    var acc: Float64 = 0
    for k in range(3):
        var s = mean[0] * members[k][0] + mean[1] * members[k][1] + mean[2] * members[k][2]
        acc += Float64(acos(min(max(s, Float32(-1)), Float32(1))))
        if s > bs:
            bs = s
            best = k
    for j in range(3):
        _close(Float64(t[0][j]), Float64(members[best][j]), 0.0, "closest member")
    _close(Float64(t[1][0]), acc / 3.0 / 3.141592653589793, 1e-6, "uncertainty")
    var tm = ensemble_normals(members, ENSEMBLE_MEAN, False)
    for j in range(3):
        _close(Float64(tm[0][j]), Float64(mean[j]), 1e-6, "mean reduction")
    assert_equal(len(tm[1]), 0)
    var up = List[Float32]()
    up.append(0.0)
    up.append(1.0)
    up.append(0.0)
    var px = normals_to_pixels(up, 1, False)
    assert_equal(Int(px[0]), 0xFF80FF80)
    var dx = normals_to_pixels(up, 1, True)
    assert_equal(Int(dx[0]), 0xFF800080)
    print("✓ normals: normalise, closest / mean ensembling, uncertainty, PNG encoding")


def _texture(w: Int, h: Int) -> List[UInt32]:
    """A smooth non-periodic test pattern (seams when tiled)."""
    var out = List[UInt32](length=w * h, fill=0)
    for y in range(h):
        for x in range(w):
            var r = Int(127.0 + 120.0 * sin(Float64(x) * 0.031 + Float64(y) * 0.011))
            var g = Int(127.0 + 120.0 * cos(Float64(y) * 0.023))
            var b = (x * 255) // max(w - 1, 1)
            out[y * w + x] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    return out^


def _noisy_texture(w: Int, h: Int) -> List[UInt32]:
    """Grain (hash noise) over non-periodic low-frequency colour: a photo-like texture that
    does not tile."""
    var out = List[UInt32](length=w * h, fill=0)
    for y in range(h):
        for x in range(w):
            var hsh = ((x * 73856093) ^ (y * 19349663)) % 1000
            var nz = Float64(hsh) / 1000.0 * 40.0 - 20.0
            var base = 110.0 + 60.0 * sin(Float64(x) * 0.021 + 0.7) + 30.0 * cos(Float64(y) * 0.029)
            var r = Int(min(max(base + nz, 0.0), 255.0))
            var g = Int(min(max(base * 0.8 + nz, 0.0), 255.0))
            var b = Int(min(max(base * 0.6 + nz, 0.0), 255.0))
            out[y * w + x] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    return out^


def test_tiling_helpers() raises:
    var w = 37
    var h = 22
    var t = _texture(w, h)
    var r = roll_pixels(t, w, h, w // 2, h // 2)
    assert_equal(Int(r[0]), Int(t[(h // 2) * w + w // 2]))
    var back = roll_pixels(r, w, h, -(w // 2), -(h // 2))
    for i in range(w * h):
        assert_equal(Int(back[i]), Int(t[i]))
    # the seams of the rolled image sit on column w - w // 2 and row h - h // 2
    var m = seam_cross_mask(w, h, 4)
    var cx = w - w // 2
    var cy = h - h // 2
    assert_true(m[3 * w + cx - 2] != 0 and m[3 * w + cx + 1] != 0 and m[3 * w + cx + 2] == 0 and m[3 * w + cx - 3] == 0)
    assert_true(m[(cy - 2) * w + 1] != 0 and m[(cy + 1) * w + 1] != 0 and m[(cy + 2) * w + 1] == 0)
    # a periodic pattern has no seam; the ramp has one
    var p = List[UInt32](length=64 * 32, fill=0)
    for y in range(32):
        for x in range(64):
            var v = Int(127.0 + 100.0 * sin(Float64(x) * 6.283185307179586 / 16.0) * cos(Float64(y) * 6.283185307179586 / 8.0))
            p[y * 64 + x] = UInt32(0xFF000000) | (UInt32(v) << 16) | (UInt32(v) << 8) | UInt32(v)
    var e_per = wrap_seam_error(p, 64, 32)
    var e_ramp = wrap_seam_error(_texture(64, 32), 64, 32)
    assert_true(e_per < 1.5)
    assert_true(e_ramp > 5.0)
    print("✓ tiling helpers: roll round trip, seam cross, seam metric (periodic", e_per, "/ ramp", e_ramp, ")")


def _u8(path: String, var shape: List[Int]) raises -> Tensor:
    var b = read_bytes(path)
    if len(b) == 0:
        raise Error("missing " + path)
    var v = List[Int](length=len(b), fill=0)
    for i in range(len(b)):
        v[i] = Int(b[i])
    return Tensor.ints(DT_U8, shape^, v^)


def _f32(path: String) -> List[Float32]:
    var b = read_bytes(path)
    var out = List[Float32](length=len(b) // 4, fill=0.0)
    var p = fptr_at(Int(b.unsafe_ptr()))
    for i in range(len(out)):
        out[i] = p.unsafe_offset(i).unsafe_load()
    _ = b^
    return out^


def test_migan(path: String, refdir: String) raises:
    var mg = MiGan.load(path)
    if refdir.byte_length() > 0:
        var cases = [("full512", 512, 512), ("crop320x384", 320, 384)]
        for c in cases:
            var name = String(c[0])
            var h = c[1]
            var w = c[2]
            mg.model.keep_value("1310")
            mg.model.set_input("image", _u8(refdir + "/" + name + "_image.u8", [1, 3, h, w]))
            mg.model.set_input("mask", _u8(refdir + "/" + name + "_mask.u8", [1, 1, h, w]))
            mg.model.run()
            var o = mg.model.output("result")
            var mid = mg.model.output("1310")
            var ro = read_bytes(refdir + "/" + name + "_out.u8")
            var rm = _f32(refdir + "/" + name + "_mid.f32")
            var md: Float64 = 0
            for i in range(mid.numel()):
                md = max(md, abs(Float64(mid.fget(i)) - Float64(rm[i])))
            var mu = 0
            var nd = 0
            for i in range(o.numel()):
                var d = abs(o.i[i] - Int(ro[i]))
                mu = max(mu, d)
                if d != 0:
                    nd += 1
            print("  MI-GAN", name, "vs onnxruntime: generator maxdiff", md, "| uint8 max", mu, "on", nd, "of", o.numel())
            assert_true(md < 1e-2)
            assert_true(mu <= 1 and nd * 1000 < o.numel())
    # seamless tile of a synthetic texture: the wrap seam step must drop to interior level
    var w = 192
    var h = 160
    var t = _noisy_texture(w, h)
    var before = wrap_seam_error(t, w, h)
    var tiled = make_tileable(mg, t, w, h, 24)
    var after = wrap_seam_error(tiled, w, h)
    print("  MI-GAN make_tileable: wrap-seam step ratio", before, "->", after)
    assert_true(after < before * 0.5 and after < 3.0)
    # pixels outside the repainted band are untouched
    assert_equal(Int(tiled[(h // 4) * w + w // 4]), Int(t[(h // 4) * w + w // 4]))
    print("✓ MI-GAN inpainting + seamless tiling")


def test_marigold(dir: String, refdir: String) raises:
    var gpu = getenv("FLINT_MARIGOLD_GPU", "1") != "0"
    var rt = Runtime(gpu)
    print("  device:", rt.device_name())
    var mg = MarigoldNormals.load(rt, dir)
    assert_equal(mg.core.unet.in_channels, 8)
    var refe = _f32(refdir + "/empty_text.f32")
    var de: Float64 = 0
    for i in range(len(refe)):
        de = max(de, abs(Float64(mg.core.text[i]) - Float64(refe[i])))
    print("  empty-prompt embedding maxdiff", de)
    assert_true(de < 1e-3)
    var rgb = _f32(refdir + "/a_rgb.f32")
    var r = mg.predict(rt, rgb, 300, 400, 4, 1, 304, 0)
    var refn = _f32(refdir + "/a_normals.f32")
    assert_equal(len(r.normals), len(refn))
    var mx: Float64 = 0
    var ang: Float64 = 0
    for p in range(len(refn) // 3):
        var d: Float64 = 0
        for k in range(3):
            mx = max(mx, abs(Float64(r.normals[p * 3 + k]) - Float64(refn[p * 3 + k])))
            d += Float64(r.normals[p * 3 + k]) * Float64(refn[p * 3 + k])
        ang += Float64(acos(Float32(min(max(d, -1.0), 1.0))))
    ang = ang / Float64(len(refn) // 3) * 57.29577951308232
    print("  Marigold normals vs diffusers: maxdiff", mx, "mean angle", ang, "deg")
    assert_true(mx < 1e-3 and ang < 0.05)
    print("✓ Marigold-Normals v1.1 parity")


def main() raises:
    test_conv2d()
    test_image_ops()
    test_zero_snr_scheduler()
    test_resampling()
    test_normals_head()
    test_tiling_helpers()
    var mp = getenv("FLINT_MIGAN", "")
    if mp.byte_length() > 0:
        test_migan(mp, getenv("FLINT_MIGAN_REF", ""))
    else:
        print("- MI-GAN runs skipped (set FLINT_MIGAN, see tools/migan/fetch.sh)")
    var md = getenv("FLINT_MARIGOLD_DIR", "")
    var mr = getenv("FLINT_MARIGOLD_REF", "")
    if md.byte_length() > 0 and mr.byte_length() > 0:
        test_marigold(md, mr)
    else:
        print("- Marigold runs skipped (set FLINT_MARIGOLD_DIR and FLINT_MARIGOLD_REF, see tools/marigold/)")
    print("marigold + MI-GAN suite passed")
