"""Classical pixel-art magnifiers on the CPU (`flint.render.pixel_scale`).

Images are `NativeImage` (0xAARRGGBB words). Borders repeat the edge pixel. Each filter has a
SPIR-V compute twin in `flint.render.pixel_scale_vk` that gives the same pixels.

  * `scale2x` / `epx` (2x), `scale3x` (3x), `scale4x` (2x twice): Andrea Mazzoleni's Scale2x /
    Scale3x rules, the same as Eric Johnston's EPX, written from the published rule tables
    (https://www.scale2x.it/algorithm). Output colours are always input colours.
  * `mmpx` (2x): Morgan McGuire and Mara Gagiu, "MMPX Style-Preserving Pixel Art
    Magnification", JCGT 10(2), 2021. A port of the paper's MIT-licensed reference shader
    (copyright 2020 Morgan McGuire & Mara Gagiu; see licenses/mmpx-LICENSE.txt). Palette exact.
  * `super_xbr` (2x): Hyllian's Super-xBR (2015, MIT): three passes of a 4x4 diagonal-edge
    detector with a two-way edge-directed interpolation and anti-ringing clamp. Pass 1 fills the
    new diagonal-centre pixels from the source, pass 2 the remaining ones on the rotated lattice,
    pass 3 re-filters the whole 2x grid at a half-pixel offset. Each pass reads the previous
    pass's complete output (the multipass-shader formulation), so the CPU and GPU agree. It
    blends colours; `palette_lock=True` snaps every output pixel to the nearest colour of its
    3 x 3 source neighbourhood so the result stays inside the input palette.
  * `nearest(img, k)`.

Never derived from xBRZ (GPL) or the LGPL xBR / HQx code.
"""

from std.math import ceil

comptime FILTER_NEAREST: Int = 0
comptime FILTER_SCALE2X: Int = 1
comptime FILTER_SCALE3X: Int = 2
comptime FILTER_SCALE4X: Int = 3
comptime FILTER_MMPX: Int = 4
comptime FILTER_SUPER_XBR: Int = 5

from flint.assets.image import NativeImage


def filter_from_name(name: String) -> Int:
    """-1 when unknown."""
    if name == "nearest":
        return FILTER_NEAREST
    if name == "scale2x" or name == "epx":
        return FILTER_SCALE2X
    if name == "scale3x":
        return FILTER_SCALE3X
    if name == "scale4x":
        return FILTER_SCALE4X
    if name == "mmpx":
        return FILTER_MMPX
    if name == "superxbr" or name == "super-xbr" or name == "super_xbr":
        return FILTER_SUPER_XBR
    return -1


def filter_factor(f: Int) -> Int:
    if f == FILTER_SCALE3X:
        return 3
    if f == FILTER_SCALE4X:
        return 4
    return 2


def new_image(w: Int, h: Int) -> NativeImage:
    var o = NativeImage()
    o.width = w
    o.height = h
    o.valid = True
    o.pixels = List[UInt32](length=w * h, fill=0)
    return o^


@always_inline
def _px(img: NativeImage, x: Int, y: Int) -> UInt32:
    var cx = min(max(x, 0), img.width - 1)
    var cy = min(max(y, 0), img.height - 1)
    return img.pixels[cy * img.width + cx]


def nearest(img: NativeImage, k: Int) -> NativeImage:
    var o = new_image(img.width * k, img.height * k)
    for y in range(o.height):
        for x in range(o.width):
            o.pixels[y * o.width + x] = img.pixels[(y // k) * img.width + x // k]
    return o^


# ----------------------------------------------------------------------------- Scale2x / EPX

def scale2x(img: NativeImage) -> NativeImage:
    var w = img.width
    var o = new_image(w * 2, img.height * 2)
    var W = o.width
    for y in range(img.height):
        for x in range(w):
            var B = _px(img, x, y - 1)
            var D = _px(img, x - 1, y)
            var E = img.pixels[y * w + x]
            var F = _px(img, x + 1, y)
            var H = _px(img, x, y + 1)
            var e0 = E
            var e1 = E
            var e2 = E
            var e3 = E
            if B != H and D != F:
                if D == B:
                    e0 = D
                if B == F:
                    e1 = F
                if D == H:
                    e2 = D
                if H == F:
                    e3 = F
            var i = (2 * y) * W + 2 * x
            o.pixels[i] = e0
            o.pixels[i + 1] = e1
            o.pixels[i + W] = e2
            o.pixels[i + W + 1] = e3
    return o^


def epx(img: NativeImage) -> NativeImage:
    """EPX is Scale2x (the same rules, found independently)."""
    return scale2x(img)


def scale3x(img: NativeImage) -> NativeImage:
    var w = img.width
    var o = new_image(w * 3, img.height * 3)
    var W = o.width
    for y in range(img.height):
        for x in range(w):
            var A = _px(img, x - 1, y - 1)
            var B = _px(img, x, y - 1)
            var C = _px(img, x + 1, y - 1)
            var D = _px(img, x - 1, y)
            var E = img.pixels[y * w + x]
            var F = _px(img, x + 1, y)
            var G = _px(img, x - 1, y + 1)
            var H = _px(img, x, y + 1)
            var I = _px(img, x + 1, y + 1)
            var e = List[UInt32](length=9, fill=E)
            if B != H and D != F:
                if D == B:
                    e[0] = D
                if (D == B and E != C) or (B == F and E != A):
                    e[1] = B
                if B == F:
                    e[2] = F
                if (D == B and E != G) or (D == H and E != A):
                    e[3] = D
                if (B == F and E != I) or (H == F and E != C):
                    e[5] = F
                if D == H:
                    e[6] = D
                if (D == H and E != I) or (H == F and E != G):
                    e[7] = H
                if H == F:
                    e[8] = F
            for j in range(3):
                for i in range(3):
                    o.pixels[(3 * y + j) * W + 3 * x + i] = e[j * 3 + i]
    return o^


def scale4x(img: NativeImage) -> NativeImage:
    var t = scale2x(img)
    return scale2x(t)


# ----------------------------------------------------------------------------- MMPX

@always_inline
def mmpx_luma(c: UInt32) -> UInt32:
    var alpha = (c >> 24) & 0xFF
    return (((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF) + 1) * (256 - alpha)


@always_inline
def _all_eq2(b: UInt32, a0: UInt32, a1: UInt32) -> Bool:
    return ((b ^ a0) | (b ^ a1)) == 0


@always_inline
def _all_eq3(b: UInt32, a0: UInt32, a1: UInt32, a2: UInt32) -> Bool:
    return ((b ^ a0) | (b ^ a1) | (b ^ a2)) == 0


@always_inline
def _all_eq4(b: UInt32, a0: UInt32, a1: UInt32, a2: UInt32, a3: UInt32) -> Bool:
    return ((b ^ a0) | (b ^ a1) | (b ^ a2) | (b ^ a3)) == 0


@always_inline
def _any_eq3(b: UInt32, a0: UInt32, a1: UInt32, a2: UInt32) -> Bool:
    return b == a0 or b == a1 or b == a2


@always_inline
def _none_eq2(b: UInt32, a0: UInt32, a1: UInt32) -> Bool:
    return b != a0 and b != a1


@always_inline
def _none_eq4(b: UInt32, a0: UInt32, a1: UInt32, a2: UInt32, a3: UInt32) -> Bool:
    return b != a0 and b != a1 and b != a2 and b != a3


def mmpx(img: NativeImage) -> NativeImage:
    var w = img.width
    var o = new_image(w * 2, img.height * 2)
    var W = o.width
    for sy in range(img.height):
        for sx in range(w):
            var A = _px(img, sx - 1, sy - 1)
            var B = _px(img, sx, sy - 1)
            var C = _px(img, sx + 1, sy - 1)
            var D = _px(img, sx - 1, sy)
            var E = _px(img, sx, sy)
            var F = _px(img, sx + 1, sy)
            var G = _px(img, sx - 1, sy + 1)
            var H = _px(img, sx, sy + 1)
            var I = _px(img, sx + 1, sy + 1)
            var J = E
            var K = E
            var L = E
            var M = E
            if ((A ^ E) | (B ^ E) | (C ^ E) | (D ^ E) | (F ^ E) | (G ^ E) | (H ^ E) | (I ^ E)) != 0:
                var P = _px(img, sx, sy - 2)
                var S = _px(img, sx, sy + 2)
                var Q = _px(img, sx - 2, sy)
                var R = _px(img, sx + 2, sy)
                var Bl = mmpx_luma(B)
                var Dl = mmpx_luma(D)
                var El = mmpx_luma(E)
                var Fl = mmpx_luma(F)
                var Hl = mmpx_luma(H)
                # 1:1 slope rules
                if (D == B and D != H and D != F) and (El >= Dl or E == A) and _any_eq3(E, A, C, G) and ((El < Dl) or A != D or E != P or E != Q):
                    J = D
                if (B == F and B != D and B != H) and (El >= Bl or E == C) and _any_eq3(E, A, C, I) and ((El < Bl) or C != B or E != P or E != R):
                    K = B
                if (H == D and H != F and H != B) and (El >= Hl or E == G) and _any_eq3(E, A, G, I) and ((El < Hl) or G != H or E != S or E != Q):
                    L = H
                if (F == H and F != B and F != D) and (El >= Fl or E == I) and _any_eq3(E, C, G, I) and ((El < Fl) or I != H or E != R or E != S):
                    M = F
                # Intersection rules
                if (E != F and _all_eq4(E, C, I, D, Q) and _all_eq2(F, B, H)) and (F != _px(img, sx + 3, sy)):
                    K = F
                    M = F
                if (E != D and _all_eq4(E, A, G, F, R) and _all_eq2(D, B, H)) and (D != _px(img, sx - 3, sy)):
                    J = D
                    L = D
                if (E != H and _all_eq4(E, G, I, B, P) and _all_eq2(H, D, F)) and (H != _px(img, sx, sy + 3)):
                    L = H
                    M = H
                if (E != B and _all_eq4(E, A, C, H, S) and _all_eq2(B, D, F)) and (B != _px(img, sx, sy - 3)):
                    J = B
                    K = B
                if Bl < El and _all_eq4(E, G, H, I, S) and _none_eq4(E, A, D, C, F):
                    J = B
                    K = B
                if Hl < El and _all_eq4(E, A, B, C, P) and _none_eq4(E, D, G, I, F):
                    L = H
                    M = H
                if Fl < El and _all_eq4(E, A, D, G, Q) and _none_eq4(E, B, C, I, H):
                    K = F
                    M = F
                if Dl < El and _all_eq4(E, C, F, I, R) and _none_eq4(E, B, A, G, H):
                    J = D
                    L = D
                # 2:1 slope rules
                if H != B:
                    if H != A and H != E and H != C:
                        if _all_eq3(H, G, F, R) and _none_eq2(H, D, _px(img, sx + 2, sy - 1)):
                            L = M
                        if _all_eq3(H, I, D, Q) and _none_eq2(H, F, _px(img, sx - 2, sy - 1)):
                            M = L
                    if B != I and B != G and B != E:
                        if _all_eq3(B, A, F, R) and _none_eq2(B, D, _px(img, sx + 2, sy + 1)):
                            J = K
                        if _all_eq3(B, C, D, Q) and _none_eq2(B, F, _px(img, sx - 2, sy + 1)):
                            K = J
                if F != D:
                    if D != I and D != E and D != C:
                        if _all_eq3(D, A, H, S) and _none_eq2(D, B, _px(img, sx + 1, sy + 2)):
                            J = L
                        if _all_eq3(D, G, B, P) and _none_eq2(D, H, _px(img, sx + 1, sy - 2)):
                            L = J
                    if F != E and F != A and F != G:
                        if _all_eq3(F, C, H, S) and _none_eq2(F, B, _px(img, sx - 1, sy + 2)):
                            K = M
                        if _all_eq3(F, I, B, P) and _none_eq2(F, H, _px(img, sx - 1, sy - 2)):
                            M = K
            var i = 2 * sy * W + 2 * sx
            o.pixels[i] = J
            o.pixels[i + 1] = K
            o.pixels[i + W] = L
            o.pixels[i + W + 1] = M
    return o^


# ----------------------------------------------------------------------------- Super-xBR

comptime SXBR_WGT1: Float32 = 0.129633
comptime SXBR_WGT2: Float32 = 0.175068


struct _Planes(Movable):
    """Float RGBA planes plus luma of a working image."""
    var w: Int
    var h: Int
    var c: List[Float32]   # [h, w, 4]
    var y: List[Float32]   # [h, w]

    def __init__(out self, w: Int, h: Int):
        self.w = w
        self.h = h
        self.c = List[Float32](length=w * h * 4, fill=0.0)
        self.y = List[Float32](length=w * h, fill=0.0)

    def set(mut self, i: Int, p: UInt32):
        var a = Float32(Int((p >> 24) & 255))
        var r = Float32(Int((p >> 16) & 255))
        var g = Float32(Int((p >> 8) & 255))
        var b = Float32(Int(p & 255))
        self.c[i * 4] = r
        self.c[i * 4 + 1] = g
        self.c[i * 4 + 2] = b
        self.c[i * 4 + 3] = a
        self.y[i] = sxbr_luma(r, g, b)


@always_inline
def sxbr_luma(r: Float32, g: Float32, b: Float32) -> Float32:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@always_inline
def _df(a: Float32, b: Float32) -> Float32:
    return abs(a - b)


def sxbr_diagonal_edge(m: List[Float32], wp: List[Float32]) -> Float32:
    """`m` is the 4x4 luma window, m[i * 4 + j] = sample at (i - 1, j - 1) relative to the
    target's upper-left support pixel (i along x, j along y). Positive -> the main diagonal is
    the edge direction."""
    var dw1 = wp[0] * (_df(m[2], m[5]) + _df(m[5], m[8]) + _df(m[7], m[10]) + _df(m[10], m[13])) \
        + wp[1] * (_df(m[3], m[6]) + _df(m[9], m[12])) \
        + wp[2] * (_df(m[3], m[9]) + _df(m[6], m[12])) \
        + wp[3] * _df(m[6], m[9]) \
        + wp[4] * (_df(m[2], m[8]) + _df(m[7], m[13])) \
        + wp[5] * (_df(m[1], m[4]) + _df(m[11], m[14]))
    var dw2 = wp[0] * (_df(m[1], m[6]) + _df(m[6], m[11]) + _df(m[4], m[9]) + _df(m[9], m[14])) \
        + wp[1] * (_df(m[0], m[5]) + _df(m[10], m[15])) \
        + wp[2] * (_df(m[0], m[10]) + _df(m[5], m[15])) \
        + wp[3] * _df(m[5], m[10]) \
        + wp[4] * (_df(m[4], m[14]) + _df(m[1], m[11])) \
        + wp[5] * (_df(m[2], m[7]) + _df(m[8], m[13]))
    return dw1 - dw2


@always_inline
def _clampi(v: Int, lo: Int, hi: Int) -> Int:
    return min(max(v, lo), hi)


def _sxbr_pixel(src: _Planes, xs: List[Int], ys: List[Int], wp: List[Float32], wa: Float32, wb: Float32) -> UInt32:
    """One Super-xBR sample from the 16 support positions (xs[i * 4 + j], ys[i * 4 + j])."""
    var lum = List[Float32](length=16, fill=0.0)
    var idx = List[Int](length=16, fill=0)
    for k in range(16):
        idx[k] = ys[k] * src.w + xs[k]
        lum[k] = src.y[idx[k]]
    var d = sxbr_diagonal_edge(lum, wp)
    var out: UInt32 = 0
    for ch in range(4):
        var v0 = src.c[idx[0] * 4 + ch]
        var v3 = src.c[idx[3] * 4 + ch]
        var v5 = src.c[idx[5] * 4 + ch]
        var v6 = src.c[idx[6] * 4 + ch]
        var v9 = src.c[idx[9] * 4 + ch]
        var v10 = src.c[idx[10] * 4 + ch]
        var v12 = src.c[idx[12] * 4 + ch]
        var v15 = src.c[idx[15] * 4 + ch]
        var lo = min(min(v5, v9), min(v6, v10))
        var hi = max(max(v5, v9), max(v6, v10))
        var f: Float32
        if d <= 0.0:
            f = wa * (v3 + v12) + wb * (v6 + v9)       # along the anti-diagonal
        else:
            f = wa * (v0 + v15) + wb * (v5 + v10)      # along the main diagonal
        f = min(max(f, lo), hi)
        var q = _clampi(Int(ceil(f)), 0, 255)
        var shift = UInt32(16 - ch * 8) if ch < 3 else UInt32(24)
        out |= UInt32(q) << shift
    return out


def super_xbr(img: NativeImage, palette_lock: Bool = False) -> NativeImage:
    var w = img.width
    var h = img.height
    var W = w * 2
    var Hh = h * 2
    var wp1: List[Float32] = [Float32(2.0), Float32(1.0), Float32(-1.0), Float32(4.0), Float32(-1.0), Float32(1.0)]
    var wp2: List[Float32] = [Float32(2.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    var wa1 = -SXBR_WGT1
    var wb1 = SXBR_WGT1 + 0.5
    var wa2 = -SXBR_WGT2
    var wb2 = SXBR_WGT2 + 0.5
    var src = _Planes(w, h)
    for i in range(w * h):
        src.set(i, img.pixels[i])
    var xs = List[Int](length=16, fill=0)
    var ys = List[Int](length=16, fill=0)
    # Pass 1: originals on even/even, diagonal centres on odd/odd, the rest temporarily the source.
    var p1 = new_image(W, Hh)
    for cy in range(h):
        for cx in range(w):
            for i in range(4):
                for j in range(4):
                    xs[i * 4 + j] = _clampi(cx + i - 1, 0, w - 1)
                    ys[i * 4 + j] = _clampi(cy + j - 1, 0, h - 1)
            var e = img.pixels[cy * w + cx]
            p1.pixels[2 * cy * W + 2 * cx] = e
            p1.pixels[2 * cy * W + 2 * cx + 1] = e
            p1.pixels[(2 * cy + 1) * W + 2 * cx] = e
            p1.pixels[(2 * cy + 1) * W + 2 * cx + 1] = _sxbr_pixel(src, xs, ys, wp1, wa1, wb1)
    # Pass 2: the odd-parity pixels on the 45-degree lattice through the known ones.
    var s1 = _Planes(W, Hh)
    for i in range(W * Hh):
        s1.set(i, p1.pixels[i])
    var p2 = new_image(W, Hh)
    for i in range(W * Hh):
        p2.pixels[i] = p1.pixels[i]
    for y in range(0, Hh, 2):
        for x in range(0, W, 2):
            for i in range(4):
                for j in range(4):
                    xs[i * 4 + j] = _clampi(x + (i - 1) + (j - 1), 0, W - 1)
                    ys[i * 4 + j] = _clampi(y + (i - 1) - (j - 1), 0, Hh - 1)
            p2.pixels[y * W + x + 1] = _sxbr_pixel(s1, xs, ys, wp2, wa2, wb2)
            for i in range(4):
                for j in range(4):
                    xs[i * 4 + j] = _clampi(x + (i - 1) + (j - 1) - 1, 0, W - 1)
                    ys[i * 4 + j] = _clampi(y + (i - 1) - (j - 1) + 1, 0, Hh - 1)
            p2.pixels[(y + 1) * W + x] = _sxbr_pixel(s1, xs, ys, wp2, wa2, wb2)
    # Pass 3: every output pixel from the 4x4 window at (-2..1), the half-pixel shift back.
    var s2 = _Planes(W, Hh)
    for i in range(W * Hh):
        s2.set(i, p2.pixels[i])
    var o = new_image(W, Hh)
    for y in range(Hh):
        for x in range(W):
            for i in range(4):
                for j in range(4):
                    xs[i * 4 + j] = _clampi(x + i - 2, 0, W - 1)
                    ys[i * 4 + j] = _clampi(y + j - 2, 0, Hh - 1)
            o.pixels[y * W + x] = _sxbr_pixel(s2, xs, ys, wp1, wa1, wb1)
    if palette_lock:
        palette_snap(o, img, 2)
    return o^


def _cdist(a: UInt32, b: UInt32) -> Int:
    var s = 0
    for sh in range(4):
        var d = Int((a >> UInt32(sh * 8)) & 255) - Int((b >> UInt32(sh * 8)) & 255)
        s += d * d
    return s


def palette_snap(mut out: NativeImage, src: NativeImage, k: Int):
    """Replace each output pixel with the closest (RGBA squared distance, first wins on ties,
    centre first) colour of the 3 x 3 source neighbourhood of the pixel it came from."""
    for y in range(out.height):
        for x in range(out.width):
            var sx = x // k
            var sy = y // k
            var p = out.pixels[y * out.width + x]
            var best = _px(src, sx, sy)
            var bd = _cdist(p, best)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var c = _px(src, sx + dx, sy + dy)
                    var d = _cdist(p, c)
                    if d < bd:
                        bd = d
                        best = c
            out.pixels[y * out.width + x] = best


def pixel_scale(img: NativeImage, filter: Int, palette_lock: Bool = False) raises -> NativeImage:
    """One application of `filter` (factor `filter_factor(filter)`; nearest is 2x)."""
    if img.width <= 0 or img.height <= 0:
        raise Error("pixel_scale: empty image")
    if filter == FILTER_NEAREST:
        return nearest(img, 2)
    if filter == FILTER_SCALE2X:
        return scale2x(img)
    if filter == FILTER_SCALE3X:
        return scale3x(img)
    if filter == FILTER_SCALE4X:
        return scale4x(img)
    if filter == FILTER_MMPX:
        return mmpx(img)
    if filter == FILTER_SUPER_XBR:
        return super_xbr(img, palette_lock)
    raise Error("pixel_scale: unknown filter " + String(filter))


def palette_of(img: NativeImage) -> List[UInt32]:
    """The distinct colours of `img`, in first-seen order."""
    var seen = Dict[UInt32, Int]()
    var out = List[UInt32]()
    for p in img.pixels:
        if p not in seen:
            seen[p] = len(out)
            out.append(p)
    return out^


def palette_subset(img: NativeImage, src: NativeImage) -> Bool:
    """Every colour of `img` occurs in `src`."""
    var seen = Dict[UInt32, Int]()
    for p in src.pixels:
        seen[p] = 1
    for p in img.pixels:
        if p not in seen:
            return False
    return True
