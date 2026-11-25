"""Flint v5 latent-diffusion suite (`test_flint_v5_diffusion.mojo`).

Always (no weights; expected values are diffusers 0.43 / torch dumps from tools/diffusion/reference.py):
  1. `TorchGenerator`: `torch.Generator().manual_seed(s); torch.randn(n)` bit-for-bit (both the
     n < 16 scalar path and the 16-wide `normal_fill` path, 64-bit seeds, continued streams).
  2. Schedulers: DDIM (leading + steps_offset, MatFuse's offset-0 schedule), Euler (sigmas, input
     scaling, step) and LCM (timesteps, step with re-noise, last step); LCM guidance embedding.
  3. `ClipTokenizer` on a synthetic vocab / merges pair: BPE ranks, `</w>`, lower-casing, punctuation,
     start / end tokens, padding to 77.
  4. Tiling: a wrap-mode convolution equals a zero-padded one on the 3 x 3 tiled image, commutes with
     circular shifts (stride 1 and 2) and `wrap_pad` / `crop` round-trip.
  5. Image + material helpers: antialiased bicubic resize (identity, constants), StableMaterials
     normal-z reconstruction, Poisson height-from-normal on a periodic bump field.
With `FLINT_DIFFUSION_DIR` (filled by `tools/diffusion/fetch.py`): StableMaterials and MatFuse runs
from seed; with `FLINT_DIFFUSION_REF` (reference.py output) also compared to diffusers images.
`FLINT_DIFFUSION_GPU=0` forces the CPU.
"""
from std.math import sqrt, sin, cos, pi
from std.os import getenv
from std.testing import assert_true, assert_equal
from flint.platform.os import read_bytes, write_bytes, make_dirs
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.diffusion.rng import TorchGenerator
from flint.ml.diffusion.schedulers import Scheduler, SCHED_DDIM, SCHED_EULER, SCHED_LCM, guidance_embedding
from flint.ml.diffusion.tokenizer import ClipTokenizer
from flint.ml.diffusion.nn import Runtime, PAD_ZERO, PAD_WRAP
from flint.ml.diffusion.vae import wrap_pad, crop
from flint.ml.diffusion.imageproc import resize_bicubic_aa
from flint.ml.diffusion.materials import PbrMaps, StableMaterials, MatFuse, height_from_normal
from flint.ml.diffusion.pipeline import nhwc_to_nchw


def _close(a: Float32, b: Float64, tol: Float64, what: String) raises:
    if abs(Float64(a) - b) > tol:
        raise Error(what + ": got " + String(a) + " expected " + String(b))


def _cfg(json: String) -> List[UInt8]:
    return List[UInt8](json.as_bytes())


comptime SD_CFG = '{"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear", "num_train_timesteps": 1000, "set_alpha_to_one": false, "steps_offset": 1, "prediction_type": "epsilon", "timestep_spacing": "leading"}'


def test_rng() raises:
    var g = TorchGenerator(0)
    var x = g.randn(64)
    _close(x[0], -1.1258398294448853, 0.0, "randn seed 0 [0]")
    _close(x[1], -1.152360200881958, 0.0, "randn seed 0 [1]")
    _close(x[2], -0.2505785822868347, 0.0, "randn seed 0 [2]")
    _close(x[3], -0.4338788390159607, 0.0, "randn seed 0 [3]")
    var e = g.randn(64)          # the stream continues (torch.randn twice after manual_seed(0))
    _close(e[0], -0.5692480206489563, 0.0, "continued stream [0]")
    _close(e[2], 1.1108161211013794, 0.0, "continued stream [2]")
    var g42 = TorchGenerator(42)
    var big = g42.randn(73728)          # StableMaterials' initial latent
    _close(big[73726], 0.017895115539431572, 0.0, "seed 42 tail")
    _close(big[73727], -2.1139986515045166, 0.0, "seed 42 last")
    var g64 = TorchGenerator((1 << 40) + 3)
    var s64 = g64.randn(17)
    _close(s64[0], -0.07664430886507034, 0.0, "64-bit seed [0]")
    _close(s64[16], 0.4851139783859253, 0.0, "64-bit seed [16] (scalar tail)")
    print("✓ torch-compatible RNG")


def test_schedulers() raises:
    var g = TorchGenerator(0)
    var x = g.randn(64)
    var e = g.randn(64)
    # DDIM, SD config: 25 leading steps with steps_offset 1
    var d = Scheduler.from_config(_cfg(SD_CFG), SCHED_DDIM)
    d.set_timesteps(25)
    assert_equal(d.num_steps(), 25)
    assert_equal(Int(d.timestep(0)), 961)
    assert_equal(Int(d.timestep(1)), 921)
    assert_equal(Int(d.timestep(24)), 1)
    _close(d.alphas_cumprod[0], 0.9991499781608582, 1e-7, "alphas_cumprod[0]")
    _close(d.alphas_cumprod[500], 0.27633246779441833, 1e-7, "alphas_cumprod[500]")
    _close(d.alphas_cumprod[999], 0.00466009508818388, 1e-8, "alphas_cumprod[999]")
    var ds = d.step(e, 3, x, g)
    _close(ds[0], -1.2353953123092651, 2e-6, "ddim step [0]")
    _close(ds[1], -1.569739818572998, 2e-6, "ddim step [1]")
    _close(ds[2], -0.5263504981994629, 2e-6, "ddim step [2]")
    # DDIM, MatFuse config
    var m = Scheduler.from_config(_cfg('{"beta_start": 0.0015, "beta_end": 0.0195, "beta_schedule": "scaled_linear", "steps_offset": 0}'), SCHED_DDIM)
    m.set_timesteps(20)
    assert_equal(Int(m.timestep(0)), 950)
    assert_equal(Int(m.timestep(19)), 0)
    var ms = m.step(e, 0, x, g)
    _close(ms[0], -1.4342139959335327, 2e-6, "matfuse ddim step [0]")
    _close(ms[1], -2.300987720489502, 2e-6, "matfuse ddim step [1]")
    # Euler
    var eu = Scheduler.from_config(_cfg(SD_CFG), SCHED_EULER)
    eu.set_timesteps(30)
    _close(Float32(eu.timestep(0)), 958.0, 1e-3, "euler t0")
    _close(Float32(eu.timestep(2)), 892.0, 1e-3, "euler t2")
    _close(eu.sigmas[0], 11.476850509643555, 2e-5, "sigma 0")
    _close(eu.sigmas[29], 0.04131447896361351, 1e-7, "sigma 29")
    _close(eu.sigmas[30], 0.0, 0.0, "final sigma")
    _close(eu.init_noise_sigma, 11.520334243774414, 2e-5, "init_noise_sigma")
    var sc = eu.scale_model_input(x, 5)
    _close(sc[0], -0.22289350628852844, 1e-6, "euler scaled [0]")
    var es = eu.step(e, 5, x, g)
    _close(es[0], -0.7432824373245239, 2e-6, "euler step [0]")
    _close(es[1], -1.7706177234649658, 2e-6, "euler step [1]")
    # LCM
    var lc = Scheduler.from_config(_cfg(SD_CFG), SCHED_LCM)
    lc.set_timesteps(2)
    assert_equal(Int(lc.timestep(0)), 999)
    assert_equal(Int(lc.timestep(1)), 499)
    lc.set_timesteps(8)
    var want8: List[Int] = [999, 879, 759, 639, 499, 379, 259, 139]
    for i in range(8):
        assert_equal(Int(lc.timestep(i)), want8[i])
    lc.set_timesteps(4)
    var g5 = TorchGenerator(5)
    var ls = lc.step(e, 1, x, g5)
    _close(ls[0], 0.2475188970565796, 1e-5, "lcm step [0]")
    _close(ls[1], -4.281840801239014, 1e-5, "lcm step [1]")
    var ll = lc.step(e, 3, x, g5)
    _close(ll[0], -0.9773845672607422, 1e-5, "lcm last [0]")
    _close(ll[2], -1.107779622077942, 1e-5, "lcm last [2]")
    var w = guidance_embedding(9.0, 256)
    _close(w[0], 0.6154466867446899, 1e-3, "w-embedding [0]")
    _close(w[128], -0.7881785035133362, 1e-3, "w-embedding [128]")
    _close(w[255], 0.6216100454330444, 1e-3, "w-embedding [255]")
    print("✓ DDIM / Euler / LCM schedulers")


def test_tokenizer() raises:
    var dir = getenv("TMPDIR", "/tmp") + "/flint_diffusion_tok"
    _ = make_dirs(dir)
    var vocab = String('{"<|startoftext|>": 0, "<|endoftext|>": 1, "l": 2, "o": 3, "w": 4, "e": 5, "r": 6, ')
    vocab += '"low</w>": 7, "lower</w>": 8, "!</w>": 9, "lo": 10, "er</w>": 11, "low": 12, "w</w>": 13, "r</w>": 14}'
    var merges = String("#version: 0.2\nl o\nlo w</w>\ne r</w>\nlo w\nlow er</w>\n")
    var vb = List[UInt8](vocab.as_bytes())
    var mb = List[UInt8](merges.as_bytes())
    assert_true(write_bytes(dir + "/vocab.json", vb))
    assert_true(write_bytes(dir + "/merges.txt", mb))
    var tok = ClipTokenizer.load(dir)
    var ids = tok.encode("Low   LOWER!")
    var want: List[Int] = [0, 7, 8, 9, 1]
    assert_equal(len(ids), len(want))
    for i in range(len(want)):
        assert_equal(ids[i], want[i])
    var w2 = tok.encode("w r")      # single letters at a word end carry </w>
    assert_equal(len(w2), 4)
    assert_equal(w2[1], 13)
    assert_equal(w2[2], 14)
    var padded = tok.encode("low", True)
    assert_equal(len(padded), 77)
    assert_equal(padded[1], 7)
    assert_equal(padded[2], 1)
    assert_equal(padded[76], 1)
    print("✓ CLIP BPE tokenizer")


def _conv_weights(cout: Int, cin: Int, k: Int) -> Tuple[Tensor, Tensor]:
    var w = List[Float32](length=cout * cin * k * k, fill=0.0)
    for i in range(len(w)):
        w[i] = Float32(sin(Float64(i) * 0.37)) * 0.5
    var b = List[Float32](length=cout, fill=0.0)
    for i in range(cout):
        b[i] = Float32(i) * 0.1 - 0.2
    return (Tensor.floats([cout, cin, k, k], w^), Tensor.floats([cout], b^))


def _roll(x: List[Float32], h: Int, w: Int, c: Int, dy: Int, dx: Int) -> List[Float32]:
    var o = List[Float32](length=len(x), fill=0.0)
    for y in range(h):
        for xx in range(w):
            for k in range(c):
                o[(((y + dy) % h) * w + (xx + dx) % w) * c + k] = x[(y * w + xx) * c + k]
    return o^


def _maxdiff(a: List[Float32], b: List[Float32]) -> Float64:
    var m: Float64 = 0
    for i in range(len(a)):
        m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return m


def test_tiling() raises:
    var rt = Runtime(False)
    var H = 8
    var W = 8
    var C = 3
    var x = List[Float32](length=H * W * C, fill=0.0)
    for i in range(len(x)):
        x[i] = Float32(cos(Float64(i) * 1.3))
    var wb = _conv_weights(4, C, 3)
    var cv = rt.conv_from(wb[0], wb[1], True, 1, 1, 1, 1, 1)
    var xa = rt.upload(1, H, W, C, x)
    var yw = rt.conv2d(xa, cv, PAD_WRAP)
    var wrapped = rt.download(yw)
    # wrap mode == zero padding on the 3 x 3 tiled image, centre tile
    var big = wrap_pad(x, 1, H, W, C, H)
    var ba = rt.upload(1, 3 * H, 3 * W, C, big)
    var yb = rt.conv2d(ba, cv, PAD_ZERO)
    var centre = crop(rt.download(yb), 1, 3 * H, 3 * W, 4, H, W, H, W)
    assert_true(_maxdiff(wrapped, centre) < 1e-5)
    # circular shifts commute with wrap convolutions (stride 1 and stride 2 by an even shift)
    var xr = rt.upload(1, H, W, C, _roll(x, H, W, C, 3, 5))
    var yr = rt.conv2d(xr, cv, PAD_WRAP)
    assert_true(_maxdiff(rt.download(yr), _roll(wrapped, H, W, 4, 3, 5)) < 1e-5)
    var cs = rt.conv_from(wb[0], wb[1], True, 2, 1, 1, 1, 1)
    var y2 = rt.download(rt.conv2d(xa, cs, PAD_WRAP))
    var xr2 = rt.upload(1, H, W, C, _roll(x, H, W, C, 2, 4))
    var y2r = rt.download(rt.conv2d(xr2, cs, PAD_WRAP))
    assert_true(_maxdiff(y2r, _roll(y2, H // 2, W // 2, 4, 1, 2)) < 1e-5)
    # a zero-padded conv does not commute: the test above would catch a missing wrap
    var yz = rt.download(rt.conv2d(xa, cv, PAD_ZERO))
    assert_true(_maxdiff(yz, wrapped) > 1e-2)
    # wrap_pad / crop round trip
    var back = crop(wrap_pad(x, 1, H, W, C, 2), 1, H + 4, W + 4, C, 2, 2, H, W)
    assert_true(_maxdiff(back, x) == 0.0)
    print("✓ wrap-mode convolutions (tileable output)")


def test_image_and_maps() raises:
    # bicubic AA: identity at equal size, constants stay constant when shrinking / growing
    var n = 16
    var img = List[Float32](length=n * n * 2, fill=0.0)
    for i in range(len(img)):
        img[i] = Float32(sin(Float64(i) * 0.7))
    assert_true(_maxdiff(resize_bicubic_aa(img, n, n, 2, n, n), img) < 1e-6)
    var flat = List[Float32](length=n * n, fill=0.25)
    var small = resize_bicubic_aa(flat, n, n, 1, 7, 5, True)
    var large = resize_bicubic_aa(flat, n, n, 1, 37, 41, False)
    for v in small:
        _close(v, 0.25, 1e-6, "resize constant (down)")
    for v in large:
        _close(v, 0.25, 1e-6, "resize constant (up)")
    # StableMaterials normal z: (0, 0) -> straight up; out-of-disc xy renormalised
    var raw2 = List[Float32](length=9, fill=0.0)
    raw2[3] = 1.0
    raw2[4] = 1.0
    var m2 = PbrMaps.from_stablematerials(raw2^, 1)
    _close(m2.normal[2], 0.5, 1e-6, "outside-disc normal z")
    _close(m2.normal[0], Float64(0.5 + 0.5 / sqrt(2.0)), 1e-6, "outside-disc normal x")
    var raw1 = List[Float32](length=9, fill=0.0)
    var m1 = PbrMaps.from_stablematerials(raw1^, 1)
    _close(m1.normal[2], 1.0, 1e-6, "flat normal z")
    _close(m1.normal[0], 0.5, 1e-6, "flat normal x")
    _close(m1.basecolor[0], 0.5, 1e-6, "basecolor mid grey")
    # height from normal: a periodic bump field is recovered up to scale / offset
    var S = 32
    var truth = List[Float64](length=S * S, fill=0.0)
    var nm = List[Float32](length=S * S * 3, fill=0.0)
    var amp = 0.6
    for r in range(S):
        for c in range(S):
            var u = 2.0 * pi * Float64(c) / Float64(S)
            var v = 2.0 * pi * Float64(r) / Float64(S)
            truth[r * S + c] = sin(u) + 0.5 * cos(2.0 * v)
            var dx = amp * cos(u) * 2.0 * pi / Float64(S)
            var drow = amp * -sin(2.0 * v) * 2.0 * pi / Float64(S)
            # +Y up: dh/dy = -dh/drow; normal = (-dx, -dy, 1) normalised
            var nx = -dx
            var ny = drow
            var l = sqrt(nx * nx + ny * ny + 1.0)
            nm[(r * S + c) * 3] = Float32(nx / l * 0.5 + 0.5)
            nm[(r * S + c) * 3 + 1] = Float32(ny / l * 0.5 + 0.5)
            nm[(r * S + c) * 3 + 2] = Float32(1.0 / l * 0.5 + 0.5)
    var h = height_from_normal(nm, S)
    var mh: Float64 = 0
    var mt: Float64 = 0
    for i in range(S * S):
        mh += Float64(h[i])
        mt += truth[i]
    mh /= Float64(S * S)
    mt /= Float64(S * S)
    var sab: Float64 = 0
    var saa: Float64 = 0
    var sbb: Float64 = 0
    for i in range(S * S):
        var a = Float64(h[i]) - mh
        var b = truth[i] - mt
        sab += a * b
        saa += a * a
        sbb += b * b
    var corr = sab / sqrt(saa * sbb)
    print("  height-from-normal correlation", corr)
    assert_true(corr > 0.99)
    print("✓ image resampling and material maps")


def _ref(dir: String, name: String) -> List[Float32]:
    var b = read_bytes(dir + "/" + name)
    var out = List[Float32](length=len(b) // 4, fill=0.0)
    var p = fptr_at(Int(b.unsafe_ptr()))
    for i in range(len(out)):
        out[i] = p.unsafe_offset(i).unsafe_load()
    _ = b^
    return out^


def test_models(dir: String, refdir: String) raises:
    var gpu = getenv("FLINT_DIFFUSION_GPU", "1") != "0"
    var rt = Runtime(gpu)
    print("  device:", rt.device_name())
    var sm = StableMaterials.load(rt, dir + "/stablematerials")
    var cond = sm.text_condition(rt, "Old rusty metal bars with peeling paint")
    var maps = sm.generate(rt, cond^, 4, 42, 10.0, True)
    assert_equal(maps.size, 512)
    for v in maps.roughness:
        assert_true(v >= 0.0 and v <= 1.0)
    if refdir.byte_length() > 0:
        var d = _maxdiff(nhwc_to_nchw(maps.raw, 1, 9, 512, 512), _ref(refdir, "sm_image.f32"))
        print("  StableMaterials image vs diffusers: maxdiff", d)
        assert_true(d < 2e-3)
    var mf = MatFuse.load(rt, dir + "/matfuse")
    var mm = mf.generate(rt, "red brick wall", List[Float32](), 0, 0, List[Float32](), List[Float32](), 20, 7, 4.0, False)
    assert_equal(mm.size, 256)
    if refdir.byte_length() > 0:
        var d = _maxdiff(nhwc_to_nchw(mm.raw, 1, 12, 256, 256), _ref(refdir, "mf_image.f32"))
        print("  MatFuse image vs diffusers: maxdiff", d)
        assert_true(d < 2e-3)
    print("✓ StableMaterials + MatFuse from seed")


def main() raises:
    test_rng()
    test_schedulers()
    test_tokenizer()
    test_tiling()
    test_image_and_maps()
    var dir = getenv("FLINT_DIFFUSION_DIR", "")
    if dir.byte_length() > 0:
        test_models(dir, getenv("FLINT_DIFFUSION_REF", ""))
    else:
        print("- model runs skipped (set FLINT_DIFFUSION_DIR, see tools/diffusion/fetch.py)")
    print("diffusion suite passed")
