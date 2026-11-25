"""Image loading and the per-model input resizers of `flint.ml.depth`.

Each model family is fed the way its reference pipeline prepares images:

  Depth Anything V2   float [0, 1] image, OpenCV `INTER_CUBIC` (float maths) to the lower-bound
                      518 size rounded to multiples of 14, ImageNet normalisation
  Depth Anything 3    8-bit image, longest side to 504 with OpenCV `INTER_AREA` (down) /
                      `INTER_CUBIC` (up, 11-bit fixed point), then each side to the nearest
                      multiple of 14 the same way, ImageNet normalisation
  MoGe-2              float [0, 1] image, PyTorch antialiased bilinear to the base token grid
                      times 14, the checkpoint's mean / std

Images are planar float [C, H, W] (`Planar`) or interleaved 8-bit RGB (`Rgb8`).
"""

from std.math import floor, ceil, sqrt
from flint.platform.os import read_bytes
from flint.assets.jpeg import decode_image_auto

comptime IMAGENET_MEAN_R: Float32 = 0.485
comptime IMAGENET_MEAN_G: Float32 = 0.456
comptime IMAGENET_MEAN_B: Float32 = 0.406
comptime IMAGENET_STD_R: Float32 = 0.229
comptime IMAGENET_STD_G: Float32 = 0.224
comptime IMAGENET_STD_B: Float32 = 0.225


struct Rgb8(Copyable, Movable):
    """8-bit RGB, rows top to bottom, interleaved."""
    var w: Int
    var h: Int
    var rgb: List[UInt8]

    def __init__(out self, w: Int, h: Int):
        self.w = w
        self.h = h
        self.rgb = List[UInt8](length=w * h * 3, fill=0)


struct Planar(Copyable, Movable):
    """Float image [c, h, w]."""
    var c: Int
    var h: Int
    var w: Int
    var data: List[Float32]

    def __init__(out self, c: Int, h: Int, w: Int):
        self.c = c
        self.h = h
        self.w = w
        self.data = List[Float32](length=c * h * w, fill=0.0)


def load_rgb(path: String) raises -> Rgb8:
    """A PNG or baseline JPEG as 8-bit RGB (alpha dropped)."""
    var data = read_bytes(path)
    if len(data) < 16:
        raise Error("depth: cannot read image " + path)
    var img = decode_image_auto(data, 1 << 15)
    if not img.valid:
        raise Error("depth: cannot decode " + path + " (PNG or baseline JPEG)")
    var out = Rgb8(img.width, img.height)
    for k in range(img.width * img.height):
        var p = img.pixels[k]
        out.rgb[3 * k] = UInt8((p >> 16) & 255)
        out.rgb[3 * k + 1] = UInt8((p >> 8) & 255)
        out.rgb[3 * k + 2] = UInt8(p & 255)
    return out^


def to_planar(img: Rgb8) -> Planar:
    """`ToTensor`: [3, h, w] in [0, 1]."""
    var p = Planar(3, img.h, img.w)
    var n = img.w * img.h
    for k in range(n):
        for c in range(3):
            p.data[c * n + k] = Float32(Int(img.rgb[3 * k + c])) / 255.0
    return p^


def normalize_(mut p: Planar, mean: List[Float32], std: List[Float32]):
    var n = p.h * p.w
    for c in range(p.c):
        var m = mean[c]
        var s = std[c]
        for k in range(n):
            p.data[c * n + k] = (p.data[c * n + k] - m) / s


def imagenet_mean() -> List[Float32]:
    return [IMAGENET_MEAN_R, IMAGENET_MEAN_G, IMAGENET_MEAN_B]


def imagenet_std() -> List[Float32]:
    return [IMAGENET_STD_R, IMAGENET_STD_G, IMAGENET_STD_B]


# ----------------------------------------------------------------------------- OpenCV

def _cv_round(x: Float32) -> Int:
    """`cvRound` (round half to even)."""
    var f = floor(x)
    var d = x - f
    var i = Int(f)
    if d > 0.5:
        return i + 1
    if d < 0.5:
        return i
    return i if i % 2 == 0 else i + 1


def _sat_u8(v: Int) -> UInt8:
    return UInt8(min(max(v, 0), 255))


def _cubic_coeffs(x: Float32) -> Tuple[Float32, Float32, Float32, Float32]:
    comptime A: Float32 = -0.75
    var c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
    var c1 = ((A + 2) * x - (A + 3)) * x * x + 1
    var c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1
    return (c0, c1, c2, 1 - c0 - c1 - c2)


struct _CubicTab(Movable):
    var idx: List[Int]       # [n_out, 4] clamped source indices
    var wf: List[Float32]    # [n_out, 4]
    var wi: List[Int]        # [n_out, 4] 11-bit fixed point

    def __init__(out self, n_in: Int, n_out: Int):
        self.idx = List[Int](length=n_out * 4, fill=0)
        self.wf = List[Float32](length=n_out * 4, fill=0.0)
        self.wi = List[Int](length=n_out * 4, fill=0)
        var scale = Float64(n_in) / Float64(n_out)
        for o in range(n_out):
            var f = Float32((Float64(o) + 0.5) * scale - 0.5)
            var s = Int(floor(f))
            f -= Float32(s)
            var c = _cubic_coeffs(f)
            var cs: List[Float32] = [c[0], c[1], c[2], c[3]]
            for k in range(4):
                self.idx[o * 4 + k] = min(max(s - 1 + k, 0), n_in - 1)
                self.wf[o * 4 + k] = cs[k]
                self.wi[o * 4 + k] = _cv_round(cs[k] * 2048.0)


def cv2_resize_cubic_f32(p: Planar, ow: Int, oh: Int) -> Planar:
    """`cv2.resize(float32 image, INTER_CUBIC)` on a planar image."""
    var tx = _CubicTab(p.w, ow)
    var ty = _CubicTab(p.h, oh)
    var out = Planar(p.c, oh, ow)
    var row = List[Float32](length=p.h * ow, fill=0.0)
    for c in range(p.c):
        var base = c * p.h * p.w
        for y in range(p.h):
            for x in range(ow):
                var acc: Float32 = 0
                for k in range(4):
                    acc += p.data[base + y * p.w + tx.idx[x * 4 + k]] * tx.wf[x * 4 + k]
                row[y * ow + x] = acc
        for y in range(oh):
            for x in range(ow):
                var acc: Float32 = 0
                for k in range(4):
                    acc += row[ty.idx[y * 4 + k] * ow + x] * ty.wf[y * 4 + k]
                out.data[(c * oh + y) * ow + x] = acc
    return out^


def cv2_resize_cubic_u8(img: Rgb8, ow: Int, oh: Int) -> Rgb8:
    """`cv2.resize(uint8 image, INTER_CUBIC)`: 11-bit fixed-point taps, 22-bit rounding."""
    var tx = _CubicTab(img.w, ow)
    var ty = _CubicTab(img.h, oh)
    var out = Rgb8(ow, oh)
    var row = List[Int](length=img.h * ow * 3, fill=0)
    for y in range(img.h):
        for x in range(ow):
            for c in range(3):
                var acc = 0
                for k in range(4):
                    acc += Int(img.rgb[(y * img.w + tx.idx[x * 4 + k]) * 3 + c]) * tx.wi[x * 4 + k]
                row[(y * ow + x) * 3 + c] = acc
    for y in range(oh):
        for x in range(ow):
            for c in range(3):
                var acc = 0
                for k in range(4):
                    acc += row[(ty.idx[y * 4 + k] * ow + x) * 3 + c] * ty.wi[y * 4 + k]
                out.rgb[(y * ow + x) * 3 + c] = _sat_u8((acc + (1 << 21)) >> 22)
    return out^


struct _AreaTab(Movable):
    var src: List[Int]
    var dst: List[Int]
    var w: List[Float32]

    def __init__(out self, n_in: Int, n_out: Int):
        self.src = List[Int]()
        self.dst = List[Int]()
        self.w = List[Float32]()
        var scale = Float64(n_in) / Float64(n_out)
        for dx in range(n_out):
            var fsx1 = Float64(dx) * scale
            var fsx2 = fsx1 + scale
            var cell = min(scale, Float64(n_in) - fsx1)
            var sx1 = Int(ceil(fsx1))
            var sx2 = Int(floor(fsx2))
            sx2 = min(sx2, n_in - 1)
            sx1 = min(sx1, sx2)
            if Float64(sx1) - fsx1 > 1e-3:
                self._add(dx, sx1 - 1, Float32((Float64(sx1) - fsx1) / cell))
            for sx in range(sx1, sx2):
                self._add(dx, sx, Float32(1.0 / cell))
            if fsx2 - Float64(sx2) > 1e-3:
                self._add(dx, sx2, Float32(min(min(fsx2 - Float64(sx2), 1.0), cell) / cell))

    def _add(mut self, d: Int, s: Int, w: Float32):
        self.dst.append(d)
        self.src.append(s)
        self.w.append(w)


def cv2_resize_area_u8(img: Rgb8, ow: Int, oh: Int) -> Rgb8:
    """`cv2.resize(uint8 image, INTER_AREA)` for downscaling (area-weighted box filter)."""
    var tx = _AreaTab(img.w, ow)
    var ty = _AreaTab(img.h, oh)
    var out = Rgb8(ow, oh)
    var hrow = List[Float32](length=ow * 3, fill=0.0)
    var acc = List[Float32](length=ow * 3, fill=0.0)
    var k = 0
    for dy in range(oh):
        for i in range(ow * 3):
            acc[i] = 0
        while k < len(ty.dst) and ty.dst[k] == dy:
            var sy = ty.src[k]
            var beta = ty.w[k]
            for i in range(ow * 3):
                hrow[i] = 0
            for j in range(len(tx.dst)):
                var d = tx.dst[j]
                var s = tx.src[j]
                for c in range(3):
                    hrow[d * 3 + c] += Float32(Int(img.rgb[(sy * img.w + s) * 3 + c])) * tx.w[j]
            for i in range(ow * 3):
                acc[i] += beta * hrow[i]
            k += 1
        for i in range(ow * 3):
            out.rgb[dy * ow * 3 + i] = _sat_u8(_cv_round(acc[i]))
    return out^


def cv2_resize_u8(img: Rgb8, ow: Int, oh: Int, upscale: Bool) -> Rgb8:
    if ow == img.w and oh == img.h:
        return img.copy()
    if upscale:
        return cv2_resize_cubic_u8(img, ow, oh)
    return cv2_resize_area_u8(img, ow, oh)


# ----------------------------------------------------------------------------- PyTorch

struct _AaTab(Movable):
    var start: List[Int]
    var size: List[Int]
    var w: List[Float32]     # [n_out, max_size]
    var stride: Int

    def __init__(out self, n_in: Int, n_out: Int):
        var scale = Float32(n_in) / Float32(n_out)
        var support: Float32 = scale if scale >= 1 else 1.0
        var inv: Float32 = Float32(1) / scale if scale >= 1 else 1.0
        self.stride = Int(ceil(support)) * 2 + 1
        self.start = List[Int](length=n_out, fill=0)
        self.size = List[Int](length=n_out, fill=0)
        self.w = List[Float32](length=n_out * self.stride, fill=0.0)
        for o in range(n_out):
            var center = scale * (Float32(o) + 0.5)
            var xmin = max(Int(center - support + 0.5), 0)
            var xsize = min(Int(center + support + 0.5), n_in) - xmin
            xsize = min(xsize, self.stride)
            var total: Float32 = 0
            for j in range(xsize):
                var t = abs((Float32(j + xmin) - center + 0.5) * inv)
                var v: Float32 = 1 - t if t < 1 else 0
                self.w[o * self.stride + j] = v
                total += v
            for j in range(xsize):
                if total != 0:
                    self.w[o * self.stride + j] /= total
            self.start[o] = xmin
            self.size[o] = xsize


def torch_resize_bilinear_aa(p: Planar, ow: Int, oh: Int) -> Planar:
    """`F.interpolate(mode="bilinear", align_corners=False, antialias=True)` (separable: width
    then height)."""
    var tx = _AaTab(p.w, ow)
    var ty = _AaTab(p.h, oh)
    var out = Planar(p.c, oh, ow)
    var row = List[Float32](length=p.h * ow, fill=0.0)
    for c in range(p.c):
        var base = c * p.h * p.w
        for y in range(p.h):
            for x in range(ow):
                var acc: Float32 = 0
                for j in range(tx.size[x]):
                    acc += p.data[base + y * p.w + tx.start[x] + j] * tx.w[x * tx.stride + j]
                row[y * ow + x] = acc
        for y in range(oh):
            for x in range(ow):
                var acc: Float32 = 0
                for j in range(ty.size[y]):
                    acc += row[(ty.start[y] + j) * ow + x] * ty.w[y * ty.stride + j]
                out.data[(c * oh + y) * ow + x] = acc
    return out^


# ----------------------------------------------------------------------------- per-model sizes

def _np_round(x: Float64) -> Int:
    """numpy / Python round (half to even)."""
    var f = floor(x)
    var d = x - f
    var i = Int(f)
    if d > 0.5:
        return i + 1
    if d < 0.5:
        return i
    return i if i % 2 == 0 else i + 1


def da2_size(w: Int, h: Int, target: Int = 518, mult: Int = 14) -> Tuple[Int, Int]:
    """Depth Anything V2 `Resize(keep_aspect_ratio, lower_bound, ensure_multiple_of=14)`: (w, h)."""
    var sw = Float64(target) / Float64(w)
    var sh = Float64(target) / Float64(h)
    var s = max(sw, sh)
    return (_fit_multiple(s * Float64(w), mult, target), _fit_multiple(s * Float64(h), mult, target))


def _fit_multiple(x: Float64, mult: Int, lo: Int) -> Int:
    var y = _np_round(x / Float64(mult)) * mult
    if y < lo:
        y = Int(ceil(x / Float64(mult))) * mult
    return y


def preprocess_da2(img: Rgb8, target: Int = 518) -> Planar:
    var sz = da2_size(img.w, img.h, target)
    var p = to_planar(img)
    var r = cv2_resize_cubic_f32(p, sz[0], sz[1])
    normalize_(r, imagenet_mean(), imagenet_std())
    return r^


def _nearest_multiple(x: Int, p: Int) -> Int:
    var down = (x // p) * p
    var up = down + p
    return up if abs(up - x) <= abs(x - down) else down


def da3_resize(img: Rgb8, target: Int = 504) -> Rgb8:
    """Depth Anything 3 `upper_bound_resize`: longest side to `target`, then each side to the
    nearest multiple of 14 (OpenCV area / cubic)."""
    var cur = img.copy()
    var longest = max(img.w, img.h)
    if longest != target:
        var scale = Float64(target) / Float64(longest)
        var nw = max(1, _np_round(Float64(img.w) * scale))
        var nh = max(1, _np_round(Float64(img.h) * scale))
        cur = cv2_resize_u8(img, nw, nh, scale > 1.0)
    var mw = max(1, _nearest_multiple(cur.w, 14))
    var mh = max(1, _nearest_multiple(cur.h, 14))
    if mw != cur.w or mh != cur.h:
        var up = mw > cur.w or mh > cur.h
        cur = cv2_resize_u8(cur, mw, mh, up)
    return cur^


def preprocess_da3(img: Rgb8, target: Int = 504) -> Planar:
    var r = da3_resize(img, target)
    var p = to_planar(r)
    normalize_(p, imagenet_mean(), imagenet_std())
    return p^


def preprocess_moge(img: Rgb8, bh: Int, bw: Int, mean: List[Float32], std: List[Float32]) -> Planar:
    var p = to_planar(img)
    var r = torch_resize_bilinear_aa(p, bw * 14, bh * 14)
    normalize_(r, mean, std)
    return r^
