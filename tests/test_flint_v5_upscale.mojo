"""Flint v5 image-upscaling suite (`test_flint_v5_upscale.mojo`).

Always (no downloads):
  1. Scale2x / EPX, Scale3x, Scale4x: the published rule tables on hand-made cases, palette
     exactness, flat images stay flat.
  2. MMPX: palette exactness, a hand-checked 1:1 diagonal, flat input.
  3. Super-xBR: flat input unchanged, anti-ringing (output inside the local min / max), the
     palette lock keeps the source palette.
  4. Golden images: every filter on a fixed synthetic sprite sheet must match the committed PNGs
     in tests/fixtures/upscale (regenerate with FLINT_UPSCALE_WRITE_GOLDEN=1).
  5. Vulkan compute path (when a Vulkan device opens; FLINT_UPSCALE_VK=0 skips): pixel-identical
     to the CPU for every filter.
  6. Real-ESRGAN networks on tiny closed-form-weight RRDBNet / SRVGGNetCompact fixtures against
     PyTorch outputs (tools/upscale/reference.py --fixtures), whole and tiled, CPU and (when a MAX
     GPU is present; FLINT_UPSCALE_GPU=0 skips) GPU; the u8 conversion's rounding.
With `FLINT_REALESRGAN_DIR` (safetensors from tools/upscale/fetch.sh) and `FLINT_REALESRGAN_REF`
(reference.py output): the released models against PyTorch on the reference input, whole and tiled.
"""
from std.os import getenv
from std.testing import assert_true, assert_equal
from flint.platform.os import read_bytes, write_bytes
from flint.assets.image import NativeImage, decode_png, encode_png
from flint.render.pixel_scale import (
    FILTER_NEAREST, FILTER_SCALE2X, FILTER_SCALE3X, FILTER_SCALE4X, FILTER_MMPX, FILTER_SUPER_XBR,
    new_image, scale2x, scale3x, scale4x, mmpx, super_xbr, pixel_scale, palette_subset, palette_of,
)
from flint.render.pixel_scale_vk import PixelScaleVulkan
from flint.ml.realesrgan import RealEsrgan, round_half_even, to_u8

comptime FIX = "tests/fixtures/upscale/"


def _img(w: Int, h: Int, px: List[UInt32]) -> NativeImage:
    var o = new_image(w, h)
    for i in range(w * h):
        o.pixels[i] = px[i]
    return o^


def sprite_sheet() -> NativeImage:
    """A 40 x 28 test sheet: transparent ground, a disc, 1:1 and 2:1 lines, a checker, single
    pixels, a semi-transparent block and a small gradient (for Super-xBR)."""
    var w = 40
    var h = 28
    var o = new_image(w, h)
    var ink: UInt32 = 0xFF202040
    var red: UInt32 = 0xFFD03030
    var yel: UInt32 = 0xFFF0D040
    var grn: UInt32 = 0xFF30A050
    var glass: UInt32 = 0x8060A0F0
    for y in range(h):
        for x in range(w):
            var p: UInt32 = 0x00000000
            var dx = x - 9
            var dy = y - 9
            if dx * dx + dy * dy <= 49:
                p = red
                if dx * dx + dy * dy <= 9:
                    p = yel
            if x >= 20 and x < 30 and y < 10 and x - 20 == y:
                p = ink
            if x >= 20 and x < 40 and y >= 12 and y < 22 and (x - 20) == 2 * (y - 12):
                p = ink
            if x >= 30 and y < 8 and ((x + y) % 2 == 0):
                p = grn
            if y >= 20 and x < 8:
                p = glass
            if y >= 22 and x >= 10 and x < 26:
                var v = UInt32((x - 10) * 16)
                p = 0xFF000000 | (v << 16) | (UInt32(255) - v) << 8 | UInt32(80)
            o.pixels[y * w + x] = p
    o.pixels[18 * w + 3] = yel
    o.pixels[25 * w + 34] = red
    return o^


def _same(a: NativeImage, b: NativeImage) -> Int:
    """Number of differing pixels (-1 if sizes differ)."""
    if a.width != b.width or a.height != b.height:
        return -1
    var n = 0
    for i in range(a.width * a.height):
        if a.pixels[i] != b.pixels[i]:
            n += 1
    return n


def test_scale2x_rules() raises:
    # A 1:1 diagonal: X on the anti-diagonal of a 3x3 -> the centre's corners fill in.
    var X: UInt32 = 0xFF000000
    var O: UInt32 = 0xFFFFFFFF
    var a = _img(3, 3, [O, O, X, O, X, O, X, O, O])
    var s = scale2x(a)
    assert_equal(s.width, 6)
    # source (0, 0) = O with B = O (edge), D = O (edge), F = O, H = O: B == H -> plain copy
    assert_equal(s.pixels[0], O)
    # source (1, 0) = O: B = O (edge), D = O, F = X, H = X -> B != H, D != F; B == F? no; H == F -> e3 = F = X
    assert_equal(s.pixels[1 * 6 + 3], X)
    assert_equal(s.pixels[0 * 6 + 2], O)
    # flat images stay flat
    var flat = _img(2, 2, [O, O, O, O])
    for f in [FILTER_SCALE2X, FILTER_SCALE3X, FILTER_SCALE4X, FILTER_MMPX, FILTER_SUPER_XBR]:
        var r = pixel_scale(flat, f)
        for p in r.pixels:
            assert_equal(p, O)
    var sheet = sprite_sheet()
    assert_true(palette_subset(scale2x(sheet), sheet), "scale2x palette")
    assert_true(palette_subset(scale3x(sheet), sheet), "scale3x palette")
    assert_true(palette_subset(scale4x(sheet), sheet), "scale4x palette")
    var s3 = scale3x(sheet)
    assert_equal(s3.width, 120)
    # the centre of every 3x3 block is the source pixel
    for y in range(sheet.height):
        for x in range(sheet.width):
            assert_equal(s3.pixels[(3 * y + 1) * 120 + 3 * x + 1], sheet.pixels[y * 40 + x])
    print("  scale2x / scale3x / scale4x rules + palette: ok")


def test_mmpx() raises:
    var sheet = sprite_sheet()
    var m = mmpx(sheet)
    assert_equal(m.width, 80)
    assert_true(palette_subset(m, sheet), "mmpx palette")
    # A thick 1:1 staircase in a 6 x 6 field: MMPX smooths the steps (fills inner corners).
    var X: UInt32 = 0xFF000000
    var O: UInt32 = 0xFFFFFFFF
    var px = List[UInt32](length=36, fill=O)
    for y in range(6):
        for x in range(6):
            if x <= y:
                px[y * 6 + x] = X
    var t = mmpx(_img(6, 6, px))
    # Source (3, 2) is O right of the step, with D = X (2, 2) and B = O: the upper-left quadrant
    # (J) takes D only where D == B; check the lower-left quadrant L (H = X at (3, 3), D == H).
    assert_equal(t.pixels[(2 * 2 + 1) * 12 + 2 * 3], X)
    assert_true(palette_subset(t, _img(6, 6, px)), "mmpx staircase palette")
    print("  mmpx: palette exact, staircase smoothed: ok")


def test_super_xbr() raises:
    var sheet = sprite_sheet()
    var s = super_xbr(sheet)
    assert_equal(s.width, 80)
    assert_equal(s.height, 56)
    var lock = super_xbr(sheet, True)
    assert_true(palette_subset(lock, sheet), "super-xbr palette lock")
    assert_true(len(palette_of(s)) > len(palette_of(sheet)), "super-xbr blends colours")
    # A hard vertical edge (20 | 200): the output stays within [min, max] of the source per channel.
    var g = new_image(8, 8)
    for y in range(8):
        for x in range(8):
            var v = UInt32(200) if x >= 4 else UInt32(20)
            g.pixels[y * 8 + x] = 0xFF000000 | (v << 16) | (v << 8) | v
    var o = super_xbr(g)
    for p in o.pixels:
        assert_true((p >> 24) == 255, "alpha preserved")
        var r = Int((p >> 16) & 255)
        assert_true(r >= 20 and r <= 200, "anti-ringing clamp")
    print("  super-xbr: size, anti-ringing, palette lock: ok")


def _write_png(path: String, img: NativeImage) raises:
    var px = img.pixels.copy()
    var data = encode_png(px, img.width, img.height, "", True)
    if not write_bytes(path, data):
        raise Error("cannot write " + path)


def _read_png(path: String) raises -> NativeImage:
    var img = decode_png(read_bytes(path), 1 << 20)
    if not img.valid:
        raise Error("cannot read golden " + path)
    return img^


def _filters() -> List[Int]:
    return [FILTER_NEAREST, FILTER_SCALE2X, FILTER_SCALE3X, FILTER_SCALE4X, FILTER_MMPX, FILTER_SUPER_XBR]


def _golden_names() -> List[String]:
    return ["nearest", "scale2x", "scale3x", "scale4x", "mmpx", "superxbr"]


def test_golden() raises:
    var sheet = sprite_sheet()
    var write = getenv("FLINT_UPSCALE_WRITE_GOLDEN", "0") == "1"
    var fs = _filters()
    var names = _golden_names()
    if write:
        _write_png(FIX + "sheet.png", sheet)
    var src = _read_png(FIX + "sheet.png")
    assert_equal(_same(src, sheet), 0)
    for i in range(len(fs)):
        var out = pixel_scale(sheet, fs[i])
        var path = FIX + "golden_" + names[i] + ".png"
        if write:
            _write_png(path, out)
        var g = _read_png(path)
        var d = _same(out, g)
        if d != 0:
            raise Error("golden " + names[i] + ": " + String(d) + " pixels differ")
    var lock = super_xbr(sheet, True)
    if write:
        _write_png(FIX + "golden_superxbr_lock.png", lock)
    assert_equal(_same(lock, _read_png(FIX + "golden_superxbr_lock.png")), 0)
    print("  golden images (7 filters): identical")


def test_vulkan() raises:
    if getenv("FLINT_UPSCALE_VK", "1") == "0":
        print("  vulkan: skipped (FLINT_UPSCALE_VK=0)")
        return
    var vk = PixelScaleVulkan()
    if not vk.open():
        print("  vulkan: skipped (" + vk.error + ")")
        return
    var sheet = sprite_sheet()
    var fs = _filters()
    var names = _golden_names()
    for i in range(len(fs)):
        var a = pixel_scale(sheet, fs[i])
        var b = vk.run(sheet, fs[i])
        var d = _same(a, b)
        if d != 0:
            raise Error("vulkan " + names[i] + ": " + String(d) + " pixels differ from the CPU")
    assert_equal(_same(super_xbr(sheet, True), vk.run(sheet, FILTER_SUPER_XBR, True)), 0)
    # odd sizes / 1-pixel images
    var tiny = new_image(1, 1)
    tiny.pixels[0] = 0xFF123456
    assert_equal(_same(mmpx(tiny), vk.run(tiny, FILTER_MMPX)), 0)
    print("  vulkan (" + vk.device_name() + "): all filters pixel-identical to the CPU")


def _read_f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) == 0 or len(b) % 4 != 0:
        raise Error("cannot read " + path)
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=0.0)
    var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr()))
    for i in range(n):
        out[i] = p.unsafe_offset(i).unsafe_load[alignment=1]()
    _ = b^
    return out^


def _max_err(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("length " + String(len(a)) + " vs " + String(len(b)))
    var m: Float64 = 0
    for i in range(len(a)):
        m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return m


def _u8_mismatch(a: List[Float32], b: List[Float32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if to_u8(a[i]) != to_u8(b[i]):
            n += 1
    return n


def _tiny(gpu: Bool) raises:
    var x = _read_f32(FIX + "tiny_input.f32")
    var rr = RealEsrgan.load(FIX + "tiny_rrdb.safetensors", gpu)
    assert_equal(rr.num_block, 2)
    var y = rr.upscale_rgb(x, 13, 10)
    var e1 = _max_err(y, _read_f32(FIX + "tiny_rrdb_out.f32"))
    var yt = rr.upscale_rgb(x, 13, 10, 5, 2)
    var e2 = _max_err(yt, _read_f32(FIX + "tiny_rrdb_tile.f32"))
    var cp = RealEsrgan.load(FIX + "tiny_compact.safetensors", gpu)
    assert_equal(cp.scale, 4)
    var yc = cp.upscale_rgb(x, 13, 10)
    var e3 = _max_err(yc, _read_f32(FIX + "tiny_compact_out.f32"))
    print("  tiny nets on " + rr.device_name() + " vs torch: rrdb " + String(e1) + ", rrdb tiled " + String(e2)
          + ", compact " + String(e3))
    assert_true(e1 < 2e-5 and e2 < 2e-5 and e3 < 2e-5, "tiny-net parity")


def test_tiny_nets() raises:
    _tiny(False)
    if getenv("FLINT_UPSCALE_GPU", "1") == "0":
        return
    try:
        _tiny(True)
    except e:
        var msg = String(e)
        if "parity" in msg:
            raise e^
        print("  tiny nets on GPU: skipped (" + msg + ")")
    # numpy-style rounding of the u8 conversion
    assert_equal(round_half_even(2.5), 2.0)
    assert_equal(round_half_even(3.5), 4.0)
    assert_equal(to_u8(1.5), UInt32(255))
    assert_equal(to_u8(-0.1), UInt32(0))
    assert_equal(to_u8(Float32(127.5) / 255.0), UInt32(128))


def test_released_models() raises:
    var d = getenv("FLINT_REALESRGAN_DIR", "")
    var ref_dir = getenv("FLINT_REALESRGAN_REF", "")
    if d == "" or ref_dir == "":
        print("  released models: skipped (set FLINT_REALESRGAN_DIR and FLINT_REALESRGAN_REF)")
        return
    var gpu = getenv("FLINT_UPSCALE_GPU", "1") != "0"
    var x = _read_f32(ref_dir + "/input.f32")
    var w = Int(getenv("FLINT_REALESRGAN_W", "64"))
    var h = Int(getenv("FLINT_REALESRGAN_H", "48"))
    var tile = Int(getenv("FLINT_REALESRGAN_TILE", "32"))
    var names: List[String] = ["realesr-general-x4v3", "realesr-animevideov3", "RealESRGAN_x4plus_anime_6B",
                             "RealESRGAN_x4plus"]
    for name in names:
        var path = d + "/" + name + ".safetensors"
        if len(read_bytes(ref_dir + "/" + name + ".f32")) == 0:
            continue
        var net = RealEsrgan.load(path, gpu)
        var want = _read_f32(ref_dir + "/" + name + ".f32")
        var y = net.upscale_rgb(x, w, h)
        var yt = net.upscale_rgb(x, w, h, tile, 10)
        var e = _max_err(y, want)
        var et = _max_err(yt, _read_f32(ref_dir + "/" + name + "_tile.f32"))
        print("  " + name + " on " + net.device_name() + ": max |diff| " + String(e) + ", tiled " + String(et)
              + ", u8 mismatches " + String(_u8_mismatch(y, want)) + " / " + String(len(want)))
        assert_true(e < 2e-3 and et < 2e-3, name + " parity")


def main() raises:
    print("[test_flint_v5_upscale]")
    test_scale2x_rules()
    test_mmpx()
    test_super_xbr()
    test_golden()
    test_vulkan()
    test_tiny_nets()
    test_released_models()
    print("[test_flint_v5_upscale] PASS")
