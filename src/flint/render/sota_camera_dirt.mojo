"""Area D (sota:camera): procedural lens-dirt texture, generated on the CPU in pure Mojo (no asset).

The front element of a real lens carries four kinds of contamination, and each has its own look once it
is lit by a bright source in or near the frame (the lens is focused far away, so the dirt is heavily
defocused):

- haze: a thin, uneven film (low-frequency fBm);
- smudges: greasy patches with a mottled interior and soft edges (anisotropic Gaussian blobs times fBm);
- fingerprints / wipe marks: bands of near-parallel ridges (a few concentric ridge patterns, and long
  curved wipe strokes);
- dust: small particles, imaged as aperture-shaped discs with a brighter rim (the defocused image of a
  point is the pupil; diffraction and spherical aberration brighten the edge), plus a few hair-thin
  scratches.

The result is a single-channel 8-bit mask (`SOTA_DIRT_W` x `SOTA_DIRT_H`, 16:9), 4 texels per word,
uploaded once into a persistent region of the post scratch buffer. The lens composite samples it
bilinearly and multiplies the wide-angle bright energy by it (`sota_lens_composite_shader`).
"""
from std.collections import List
from std.ffi import external_call
from std.math import atan2, cos, exp, floor, max, min, sin, sqrt
from flint.platform.amdgpu import GpuBuffer

comptime SOTA_DIRT_W = 384
comptime SOTA_DIRT_H = 216
comptime SOTA_DIRT_WORDS = (SOTA_DIRT_W * SOTA_DIRT_H + 3) // 4
comptime _POST_BASE = 1024


def sota_dirt_offset(out_w: Int, out_h: Int, max_pix: Int) -> Int:
    """Post-scratch word offset of the persistent dirt texture, or 0 when it does not fit (tiny frames:
    the composite then falls back to its inline procedural dirt). It sits above the lens quarter-res
    buffers and the VRS shade list, below the VRS rate map (`SotaVrsLayout`)."""
    var qw = (out_w + 3) // 4
    var qh = (out_h + 3) // 4
    var lens_end = _POST_BASE + max_pix * 8 + max_pix // 4 + 64 + qw * qh * 20  # bright, ghost, blur, glow, streak
    var vrs_list_end = _POST_BASE + max_pix * 8 + max_pix // 4 + 64 + 2 + max_pix
    var off = ((max(lens_end, vrs_list_end) + 64 + 63) // 64) * 64
    var vrs_rate_off = _POST_BASE + max_pix * 10 - max_pix // 16 - 1024
    if off + SOTA_DIRT_WORDS + 64 > vrs_rate_off:
        return 0
    return off


# ============================================================================
# CPU noise
# ============================================================================

def _hash(x: UInt32) -> UInt32:
    var h = x * UInt32(747796405) + UInt32(2891336453)
    var w = ((h >> ((h >> UInt32(28)) + UInt32(4))) ^ h) * UInt32(277803737)
    return (w >> UInt32(22)) ^ w


def _hash2(ix: Int, iy: Int, seed: UInt32) -> Float32:
    var h = _hash(UInt32(ix & 0xFFFF) * UInt32(73856093) ^ UInt32(iy & 0xFFFF) * UInt32(19349663) ^ seed)
    return Float32(Int(h >> UInt32(8))) * Float32(1.0 / 16777216.0)


struct _Rng(Movable):
    var s: UInt32

    def __init__(out self, seed: UInt32):
        self.s = seed

    def next(mut self) -> Float32:
        self.s = _hash(self.s + UInt32(0x9E3779B9))
        return Float32(Int(self.s >> UInt32(8))) * Float32(1.0 / 16777216.0)


def _value_noise(x: Float32, y: Float32, seed: UInt32) -> Float32:
    var xi = Int(floor(x))
    var yi = Int(floor(y))
    var fx = x - Float32(xi)
    var fy = y - Float32(yi)
    var sx = fx * fx * (Float32(3.0) - Float32(2.0) * fx)
    var sy = fy * fy * (Float32(3.0) - Float32(2.0) * fy)
    var a = _hash2(xi, yi, seed)
    var b = _hash2(xi + 1, yi, seed)
    var c = _hash2(xi, yi + 1, seed)
    var d = _hash2(xi + 1, yi + 1, seed)
    return (a * (Float32(1.0) - sx) + b * sx) * (Float32(1.0) - sy) + (c * (Float32(1.0) - sx) + d * sx) * sy


def _fbm(x: Float32, y: Float32, octaves: Int, seed: UInt32) -> Float32:
    """Sum of `octaves` value-noise octaves (lacunarity 2, gain 0.5), normalised to [0, 1]."""
    var acc = Float32(0.0)
    var amp = Float32(0.5)
    var norm = Float32(0.0)
    var f = Float32(1.0)
    for o in range(octaves):
        acc += amp * _value_noise(x * f, y * f, seed + UInt32(o * 1013))
        norm += amp
        amp *= Float32(0.5)
        f *= Float32(2.0)
    return acc / norm


def _abs(x: Float32) -> Float32:
    return x if x >= 0.0 else -x


def _smoothstep(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = min(Float32(1.0), max(Float32(0.0), (x - e0) / (e1 - e0)))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


# ============================================================================
# Generator
# ============================================================================

def sota_lens_dirt_texture_cpu(seed: UInt32 = 0x5D17) -> List[Float32]:
    """The dirt mask in [0, 1], `SOTA_DIRT_W * SOTA_DIRT_H` texels, row-major. Deterministic per seed."""
    var w = SOTA_DIRT_W
    var h = SOTA_DIRT_H
    var img = List[Float32](length=w * h, fill=0.0)
    var aspect = Float32(w) / Float32(h)

    # 1. Haze: uneven thin film, stronger towards the rim of the element (where cleaning cloths miss).
    for y in range(h):
        for x in range(w):
            var u = Float32(x) / Float32(h)
            var v = Float32(y) / Float32(h)
            var n = _fbm(u * 3.0, v * 3.0, 5, seed)
            var cx = (Float32(x) / Float32(w) - 0.5) * aspect
            var cy = Float32(y) / Float32(h) - 0.5
            var rim = _smoothstep(0.35, 0.95, sqrt(cx * cx + cy * cy))
            img[y * w + x] = _smoothstep(0.42, 0.75, n) * (Float32(0.08) + Float32(0.10) * rim)

    var rng = _Rng(seed ^ UInt32(0xA511E9B3))

    # 2. Smudges: rotated anisotropic Gaussians with a mottled (fBm) interior.
    for i in range(34):
        var cx = rng.next() * Float32(w)
        var cy = rng.next() * Float32(h)
        var sa = (Float32(8.0) + rng.next() * Float32(34.0))
        var sb = sa * (Float32(0.35) + rng.next() * Float32(0.6))
        var ang = rng.next() * Float32(3.14159265)
        var strength = Float32(0.18) + rng.next() * Float32(0.42)
        var ca = cos(ang)
        var sn = sin(ang)
        var reach = Int(sa * 3.0) + 1
        var x0 = max(0, Int(cx) - reach)
        var x1 = min(w - 1, Int(cx) + reach)
        var y0 = max(0, Int(cy) - reach)
        var y1 = min(h - 1, Int(cy) + reach)
        var s_seed = seed + UInt32(7919 * (i + 3))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                var dx = Float32(x) - cx
                var dy = Float32(y) - cy
                var a = (dx * ca + dy * sn) / sa
                var b = (-dx * sn + dy * ca) / sb
                var g = exp(-(a * a + b * b) * Float32(1.5))
                if g < 0.003:
                    continue
                var mottle = _fbm(Float32(x) * 0.09, Float32(y) * 0.09, 4, s_seed)
                img[y * w + x] += strength * g * (Float32(0.35) + Float32(0.9) * _smoothstep(0.3, 0.8, mottle))

    # 3a. Fingerprints: concentric ridge patterns (elliptical, slightly warped) fading out at the edge.
    for i in range(3):
        var cx = (Float32(0.15) + rng.next() * Float32(0.7)) * Float32(w)
        var cy = (Float32(0.15) + rng.next() * Float32(0.7)) * Float32(h)
        var rad = Float32(22.0) + rng.next() * Float32(18.0)
        var ang = rng.next() * Float32(3.14159265)
        var ca = cos(ang)
        var sn = sin(ang)
        var period = Float32(2.6) + rng.next() * Float32(0.8)
        var reach = Int(rad * 1.3) + 1
        var f_seed = seed + UInt32(104729 * (i + 1))
        for y in range(max(0, Int(cy) - reach), min(h - 1, Int(cy) + reach) + 1):
            for x in range(max(0, Int(cx) - reach), min(w - 1, Int(cx) + reach) + 1):
                var dx = Float32(x) - cx
                var dy = Float32(y) - cy
                var a = dx * ca + dy * sn
                var b = (-dx * sn + dy * ca) * Float32(1.35)
                var r = sqrt(a * a + b * b)
                if r > rad:
                    continue
                var warp = _fbm(Float32(x) * 0.05, Float32(y) * 0.05, 3, f_seed) * Float32(6.0)
                var ridge = Float32(0.5) + Float32(0.5) * cos((r + warp) * Float32(6.283185) / period)
                var fade = Float32(1.0) - _smoothstep(rad * 0.55, rad, r)
                img[y * w + x] += Float32(0.22) * ridge * ridge * fade

    # 3b. Wipe strokes: long arcs of parallel streaks (a cloth dragged across the glass).
    for i in range(4):
        var cx = rng.next() * Float32(w)
        var cy = rng.next() * Float32(h) + Float32(h) * Float32(0.6)
        var r0 = Float32(120.0) + rng.next() * Float32(160.0)
        var band = Float32(10.0) + rng.next() * Float32(14.0)
        var a0 = Float32(3.4) + rng.next() * Float32(1.2)
        var span = Float32(0.6) + rng.next() * Float32(0.7)
        var w_seed = seed + UInt32(15485863 * (i + 1))
        for y in range(h):
            for x in range(w):
                var dx = Float32(x) - cx
                var dy = Float32(y) - cy
                var r = sqrt(dx * dx + dy * dy)
                var dr = r - r0
                if dr < -band or dr > band:
                    continue
                var t = atan2(dy, dx) + Float32(6.283185)
                var rel = (t - a0) - Float32(6.283185) * floor((t - a0) / Float32(6.283185))
                if rel > span:
                    continue
                var along = rel / span
                var env = _smoothstep(0.0, 0.15, along) * (Float32(1.0) - _smoothstep(0.7, 1.0, along))
                var across = Float32(1.0) - _abs(dr) / band
                var streak = _fbm(dr * 0.9, along * 3.0, 3, w_seed)
                img[y * w + x] += Float32(0.16) * env * across * _smoothstep(0.45, 0.8, streak)

    # 4a. Dust: defocused discs with a bright rim, a few large and soft, many small.
    for i in range(220):
        var cx = rng.next() * Float32(w)
        var cy = rng.next() * Float32(h)
        var big = rng.next() < 0.12
        var r = (Float32(3.0) + rng.next() * Float32(5.0)) if big else (Float32(0.7) + rng.next() * Float32(2.0))
        var strength = (Float32(0.25) + rng.next() * Float32(0.35)) if big else (Float32(0.25) + rng.next() * Float32(0.5))
        _ = i
        var reach = Int(r + 2.0)
        for y in range(max(0, Int(cy) - reach), min(h - 1, Int(cy) + reach) + 1):
            for x in range(max(0, Int(cx) - reach), min(w - 1, Int(cx) + reach) + 1):
                var dx = Float32(x) + 0.5 - cx
                var dy = Float32(y) + 0.5 - cy
                var d = sqrt(dx * dx + dy * dy)
                var disc = min(Float32(1.0), max(Float32(0.0), r - d + Float32(0.5)))
                if disc <= 0.0:
                    continue
                var edge = Float32(1.0)
                if big:
                    edge = Float32(0.6) + Float32(0.4) * _smoothstep(r * 0.4, r, d)
                img[y * w + x] += strength * disc * edge

    # 4b. Scratches: hair-thin, slightly curved lines.
    for i in range(7):
        var x = rng.next() * Float32(w)
        var y = rng.next() * Float32(h)
        var ang = rng.next() * Float32(6.283185)
        var bend = (rng.next() - Float32(0.5)) * Float32(0.02)
        var length = Int(Float32(30.0) + rng.next() * Float32(110.0))
        var strength = Float32(0.25) + rng.next() * Float32(0.3)
        _ = i
        for _s in range(length):
            var xi = Int(x)
            var yi = Int(y)
            if xi >= 0 and xi < w and yi >= 0 and yi < h:
                img[yi * w + xi] += strength
            x += cos(ang)
            y += sin(ang)
            ang += bend

    for i in range(w * h):
        img[i] = min(Float32(1.0), max(Float32(0.0), img[i]))
    return img^


def sota_pack_dirt_u8(img: List[Float32]) -> List[UInt32]:
    """4 texels per word (texel i in byte i & 3), rounded to 8 bits."""
    var out = List[UInt32](length=SOTA_DIRT_WORDS, fill=0)
    for i in range(min(len(img), SOTA_DIRT_W * SOTA_DIRT_H)):
        var q = UInt32(Int(min(Float32(255.0), max(Float32(0.0), img[i] * Float32(255.0) + Float32(0.5)))))
        out[i >> 2] |= q << UInt32((i & 3) * 8)
    return out^


def sota_upload_dirt(words: List[UInt32], buf_post: GpuBuffer, word_off: Int) -> Bool:
    """Host write of the packed texture into the (host-mapped) post scratch buffer."""
    if buf_post.cpu_address == 0 or word_off <= 0 or (word_off + len(words)) * 4 > buf_post.size:
        return False
    _ = external_call["memcpy", Int](buf_post.cpu_address + word_off * 4, Int(words.unsafe_ptr()), len(words) * 4)
    return True
