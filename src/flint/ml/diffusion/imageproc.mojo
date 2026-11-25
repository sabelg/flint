"""Image preprocessing for `flint.ml.diffusion`: float channels-last images, antialiased bicubic
resampling (bit-for-bit the separable filter of `torch.nn.functional.interpolate(mode="bicubic",
antialias=True)`, a = -0.5, which is also PIL's BICUBIC filter), centre crops, conversion from /
to `NativeImage` (0xAARRGGBB) and the CLIP / MatFuse input pipelines."""

from flint.assets.image import NativeImage


def _cubic(x0: Float64) -> Float64:
    var a: Float64 = -0.5
    var x = abs(x0)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


struct _Taps(Movable):
    var start: List[Int]
    var count: List[Int]
    var w: List[Float64]
    var stride: Int

    def __init__(out self):
        self.start = List[Int]()
        self.count = List[Int]()
        self.w = List[Float64]()
        self.stride = 0


def _taps(inn: Int, outn: Int, align_corners: Bool) -> _Taps:
    var scale = Float64(inn) / Float64(outn)
    if align_corners and outn > 1:
        scale = Float64(inn - 1) / Float64(outn - 1)
    var support = 2.0 * scale if scale >= 1.0 else 2.0
    var inv = 1.0 / scale if scale >= 1.0 else 1.0
    var t = _Taps()
    t.stride = Int(2.0 * support) + 3
    t.w = List[Float64](length=outn * t.stride, fill=0.0)
    for i in range(outn):
        var c = scale * (Float64(i) + 0.5)
        var lo = max(Int(c - support + 0.5), 0)
        var hi = min(Int(c + support + 0.5), inn)
        var n = min(hi - lo, t.stride)
        var total: Float64 = 0
        for j in range(n):
            var v = _cubic((Float64(j + lo) - c + 0.5) * inv)
            t.w[i * t.stride + j] = v
            total += v
        if total != 0.0:
            for j in range(n):
                t.w[i * t.stride + j] /= total
        t.start.append(lo)
        t.count.append(n)
    return t^


def resize_bicubic_aa(x: List[Float32], h: Int, w: Int, c: Int, oh: Int, ow: Int,
                      align_corners: Bool = False) -> List[Float32]:
    """Resize a channels-last [h, w, c] image to [oh, ow, c] (horizontal pass, then vertical)."""
    var tx = _taps(w, ow, align_corners)
    var tmp = List[Float32](length=h * ow * c, fill=0.0)
    for y in range(h):
        for i in range(ow):
            var s = tx.start[i]
            for k in range(c):
                var acc: Float64 = 0
                for j in range(tx.count[i]):
                    acc += tx.w[i * tx.stride + j] * Float64(x[(y * w + s + j) * c + k])
                tmp[(y * ow + i) * c + k] = Float32(acc)
    var ty = _taps(h, oh, align_corners)
    var out = List[Float32](length=oh * ow * c, fill=0.0)
    for i in range(oh):
        var s = ty.start[i]
        for xx in range(ow):
            for k in range(c):
                var acc: Float64 = 0
                for j in range(ty.count[i]):
                    acc += ty.w[i * ty.stride + j] * Float64(tmp[((s + j) * ow + xx) * c + k])
                out[(i * ow + xx) * c + k] = Float32(acc)
    return out^


def center_crop(x: List[Float32], h: Int, w: Int, c: Int, oh: Int, ow: Int) -> List[Float32]:
    """torchvision-style centre crop (top = round((h - oh) / 2))."""
    var top = Int((Float64(h - oh) / 2.0) + 0.5)
    var left = Int((Float64(w - ow) / 2.0) + 0.5)
    var out = List[Float32](length=oh * ow * c, fill=0.0)
    for y in range(oh):
        for xx in range(ow):
            for k in range(c):
                out[(y * ow + xx) * c + k] = x[((top + y) * w + left + xx) * c + k]
    return out^


def image_to_rgb(img: NativeImage) -> List[Float32]:
    """[h, w, 3] in [0, 1] from 0xAARRGGBB pixels."""
    var n = img.width * img.height
    var out = List[Float32](length=n * 3, fill=0.0)
    for i in range(n):
        var p = img.pixels[i]
        out[i * 3] = Float32((p >> 16) & 255) / 255.0
        out[i * 3 + 1] = Float32((p >> 8) & 255) / 255.0
        out[i * 3 + 2] = Float32(p & 255) / 255.0
    return out^


@always_inline
def _u8(v: Float32) -> UInt32:
    var c = min(max(v, Float32(0.0)), Float32(1.0))
    return UInt32(Int(c * 255.0 + 0.5))


def rgb_to_pixels(x: List[Float32], n: Int) -> List[UInt32]:
    """Opaque 0xAARRGGBB pixels from [n, 3] values in [0, 1]."""
    var out = List[UInt32](length=n, fill=0)
    for i in range(n):
        out[i] = UInt32(0xFF000000) | (_u8(x[i * 3]) << 16) | (_u8(x[i * 3 + 1]) << 8) | _u8(x[i * 3 + 2])
    return out^


def gray_to_pixels(x: List[Float32], n: Int) -> List[UInt32]:
    var out = List[UInt32](length=n, fill=0)
    for i in range(n):
        var g = _u8(x[i])
        out[i] = UInt32(0xFF000000) | (g << 16) | (g << 8) | g
    return out^


def clip_preprocess(rgb01: List[Float32], h: Int, w: Int, size: Int = 224) -> List[Float32]:
    """`CLIPImageProcessor`: bicubic resize of the shortest edge to `size`, centre crop
    `size` x `size` ([0, 1] values; normalise with `ClipVision.normalize`). PIL resamples 8-bit
    images and rounds between its two passes, so photos differ from transformers by ~1 / 255."""
    var oh = size
    var ow = size
    if h < w:
        ow = Int(Float64(size) * Float64(w) / Float64(h))
    else:
        oh = Int(Float64(size) * Float64(h) / Float64(w))
    var r = resize_bicubic_aa(rgb01, h, w, 3, oh, ow, False)
    for i in range(len(r)):
        r[i] = Float32(Int(min(max(r[i], Float32(0.0)), Float32(1.0)) * 255.0 + 0.5)) / 255.0
    return center_crop(r, oh, ow, 3, size, size)
