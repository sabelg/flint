"""Transparent-image preparation for TRELLIS.2's DINOv3 conditioning input."""

from std.math import sin
from flint.assets.image import NativeImage
from flint.ml.tensor import Tensor


def _lanczos3(x: Float64) -> Float64:
    if x == 0:
        return 1
    if abs(x) >= 3:
        return 0
    var a = x * Float64(3.141592653589793)
    return sin(a) * sin(a / 3) / (a * a / 3)


struct _Filter(Movable):
    var start: List[Int]
    var offsets: List[Int]
    var weights: List[Int]

    def __init__(out self, size: Int, target: Int):
        self.start = List[Int](capacity=target)
        self.offsets = List[Int](capacity=target + 1)
        self.weights = List[Int]()
        var ratio = Float64(size) / Float64(target)
        var scale = max(Float64(1), ratio)
        for out in range(target):
            var center = (Float64(out) + 0.5) * ratio
            var low = max(0, Int(center - 3 * scale + 0.5))
            var high = min(size, Int(center + 3 * scale + 0.5))
            self.start.append(low)
            self.offsets.append(len(self.weights))
            var taps = List[Float64](capacity=high - low)
            var total = Float64(0)
            for src in range(low, high):
                var tap = _lanczos3((Float64(src) + 0.5 - center) / scale)
                taps.append(tap)
                total += tap
            for value in taps:
                var fixed = value / total * Float64(1 << 22)
                self.weights.append(Int(fixed + 0.5) if fixed >= 0 else Int(fixed - 0.5))
        self.offsets.append(len(self.weights))


def _resample_axis(pixels: List[UInt32], width: Int, height: Int, target: Int, vertical: Bool) -> List[UInt32]:
    var filter = _Filter(height if vertical else width, target)
    var out_w = width if vertical else target
    var out_h = target if vertical else height
    var result = List[UInt32](length=out_w * out_h, fill=0)
    var stride = width if vertical else 1
    for y in range(out_h):
        for x in range(out_w):
            var q = y if vertical else x
            var base = filter.start[q] * width + x if vertical else y * width + filter.start[q]
            var sums = SIMD[DType.int64, 4](1 << 21)
            for k in range(filter.offsets[q], filter.offsets[q + 1]):
                var p = pixels[base]
                comptime for ch in range(4):
                    sums[ch] += Int64(Int((p >> UInt32(8 * ch)) & 255) * filter.weights[k])
                base += stride
            var packed = UInt32(0)
            comptime for ch in range(4):
                packed |= UInt32(min(255, max(0, Int(sums[ch]) >> 22))) << UInt32(8 * ch)
            result[y * out_w + x] = packed
    return result^


def _resize(var pixels: List[UInt32], w: Int, h: Int, nw: Int, nh: Int) -> List[UInt32]:
    if w != nw:
        pixels = _resample_axis(pixels, w, h, nw, False)
    if h != nh:
        pixels = _resample_axis(pixels, nw, h, nh, True)
    return pixels^


def _round_half_even(doubled: Int) -> Int:
    var floor = doubled >> 1
    return floor + ((doubled & 1) & (floor & 1))


def prepare_trellis_image(image: NativeImage, target: Int = 512, crop_scale: Float64 = 1.0) raises -> Tensor:
    """Crop an alpha-masked image onto black and normalize RGB for the 512/1024 encoder."""
    var w = image.width
    var h = image.height
    if (not image.valid or w < 1 or h < 1 or w > 40000000 // h or len(image.pixels) != w * h
            or not (crop_scale >= 1 and crop_scale <= 2) or (target != 512 and target != 1024)):
        raise Error("Invalid image or DINOv3 resolution (expected 512 or 1024)")
    var transparent = False
    for pixel in image.pixels:
        if pixel >> 24 != 255:
            transparent = True
            break
    if not transparent:
        raise Error("TRELLIS.2 input needs an alpha cutout")
    var pixels = image.pixels.copy()
    var maximum = max(w, h)
    if maximum > 1024:
        var nw = w * 1024 // maximum
        var nh = h * 1024 // maximum
        if nw == 0 or nh == 0:
            raise Error("Image too thin to resize")
        # Resample RGBA in premultiplied space to avoid fringes at cutout boundaries.
        for i in range(len(pixels)):
            var p = pixels[i]
            var a = p >> 24
            var v = a << 24
            comptime for ch in range(3):
                var color = (p >> UInt32(ch * 8)) & 255
                v |= ((color * a + 127) // 255) << UInt32(ch * 8)
            pixels[i] = v
        pixels = _resize(pixels^, w, h, nw, nh)
        for i in range(len(pixels)):
            var p = pixels[i]
            var a = p >> 24
            if a != 0 and a != 255:
                var v = a << 24
                comptime for ch in range(3):
                    var color = (p >> UInt32(ch * 8)) & 255
                    v |= min(UInt32(255), color * 255 // a) << UInt32(ch * 8)
                pixels[i] = v
        w = nw
        h = nh

    var left = w
    var top = h
    var right = -1
    var bottom = -1
    for i in range(len(pixels)):
        if pixels[i] >> 24 > 204:
            left = min(left, i % w)
            right = max(right, i % w)
            top = min(top, i // w)
            bottom = max(bottom, i // w)
    var half = Int(Float64(max(right - left, bottom - top)) * crop_scale) // 2
    if half < 1:
        raise Error("Too few opaque foreground pixels")
    var x0 = _round_half_even(left + right - 2 * half)
    var y0 = _round_half_even(top + bottom - 2 * half)
    var side = 2 * half
    var crop = List[UInt32](length=side * side, fill=0)
    for y in range(side):
        for x in range(side):
            if x + x0 < 0 or x + x0 >= w or y + y0 < 0 or y + y0 >= h:
                continue
            var p = pixels[(y + y0) * w + x + x0]
            var a = Float32(p >> 24) / 255
            var rgb = UInt32(0)
            comptime for ch in range(3):
                var color = Float32((p >> UInt32(ch * 8)) & 255) / 255
                rgb |= UInt32(color * a * 255) << UInt32(ch * 8)
            crop[y * side + x] = rgb
    var resized = _resize(crop^, side, side, target, target)
    var mean: List[Float32] = [0.485, 0.456, 0.406]
    var std: List[Float32] = [0.229, 0.224, 0.225]
    var out = Tensor.zeros([1, 3, target, target])
    for i in range(len(resized)):
        comptime for ch in range(3):
            var unit = Float32((resized[i] >> UInt32((2 - ch) * 8)) & 255) / 255
            out.f[ch * len(resized) + i] = (unit - mean[ch]) / std[ch]
    return out^
