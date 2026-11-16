"""Pure-Mojo Hardware Texture Block Encoders & Reference Decoders (`BC7`, `BC5`, `BC6H`).

Implements Khronos Data Format Specification 1.3 compliant block compression so
Flint uploads 4:1 / 6:1 compressed `VkImage` mip chains directly to GPU hardware TMUs:
  - `BC7` (`VK_FORMAT_BC7_UNORM` / `BC7_SRGB`, Mode 6): Full 8-bit endpoint RGBA (`7+1` P-bit)
    with 4-bit (16-level) per-texel PCA line-fit weights (`16` bytes per `4x4` block).
  - `BC5` (`VK_FORMAT_BC5_UNORM`): Dual independent 8-bit endpoint + 3-bit weight channels for
    tangent-space `(Nx, Ny)` normals (`Nz = sqrt(max(0, 1 - Nx^2 - Ny^2))`, `< 0.35°` error).
  - `BC6H` (`VK_FORMAT_BC6H_UFLOAT`, Mode 11 `0x03`): Unsigned half-float HDR (`[0 .. 65504]`)
    with unquantized 10-bit RGB endpoints + 4-bit weights (`16` bytes per `4x4` block).
"""
from std.math import sqrt
from std.memory import bitcast
from flint.core.math import Vec3


@always_inline
def _bc7_weight4(idx: Int) -> Int:
    var w = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]
    return w[max(0, min(15, idx))]


struct _BitPacker128(Copyable, Movable):
    var lo: UInt64
    var hi: UInt64
    var bit_pos: Int

    def __init__(out self):
        self.lo = 0
        self.hi = 0
        self.bit_pos = 0

    @always_inline
    def put(mut self, val: Int, bits: Int):
        var mask = (UInt64(1) << UInt64(bits)) - UInt64(1)
        var v = UInt64(val) & mask
        if self.bit_pos < 64:
            self.lo |= v << UInt64(self.bit_pos)
            if self.bit_pos + bits > 64:
                self.hi |= v >> UInt64(64 - self.bit_pos)
        else:
            self.hi |= v << UInt64(self.bit_pos - 64)
        self.bit_pos += bits

    @always_inline
    def append_to(self, mut out_bytes: List[UInt8]):
        for i in range(8):
            out_bytes.append(UInt8((self.lo >> UInt64(i * 8)) & 0xFF))
        for i in range(8):
            out_bytes.append(UInt8((self.hi >> UInt64(i * 8)) & 0xFF))


struct _BitReader128(Copyable, Movable):
    var lo: UInt64
    var hi: UInt64
    var bit_pos: Int

    def __init__(out self, data: List[UInt8], offset: Int):
        self.lo = 0
        self.hi = 0
        self.bit_pos = 0
        for i in range(8):
            self.lo |= UInt64(data[offset + i]) << UInt64(i * 8)
        for i in range(8):
            self.hi |= UInt64(data[offset + 8 + i]) << UInt64(i * 8)

    @always_inline
    def get(mut self, bits: Int) -> Int:
        var mask = (UInt64(1) << UInt64(bits)) - UInt64(1)
        var res: UInt64
        if self.bit_pos < 64:
            res = self.lo >> UInt64(self.bit_pos)
            if self.bit_pos + bits > 64:
                res |= self.hi << UInt64(64 - self.bit_pos)
        else:
            res = self.hi >> UInt64(self.bit_pos - 64)
        self.bit_pos += bits
        return Int(res & mask)


# ============================================================================
# IEEE-754 Half-Float (`F16`) Exact Conversion Helpers
# ============================================================================

@always_inline
def f32_to_uf16_bits(val: Float32) -> Int:
    """Clamps `val` to `[0.0 .. 65504.0]` and returns IEEE-754 unsigned `F16` bits (`0 .. 0x7BFF`)."""
    var v = max(Float32(0.0), min(Float32(65504.0), val))
    var bits = bitcast[DType.uint32](v)
    var exp32 = Int((bits >> 23) & 0xFF)
    var mant32 = Int(bits & 0x7FFFFF)
    if exp32 == 0:
        return 0
    var exp16 = exp32 - 127 + 15
    if exp16 <= 0:
        if exp16 < -10:
            return 0
        var m = (mant32 | 0x800000) >> (1 - exp16)
        return (m + 0x1000) >> 13
    if exp16 >= 31:
        return 0x7BFF
    var rounded = mant32 + 0x1000
    if (rounded & 0x800000) != 0:
        exp16 += 1
        rounded = 0
        if exp16 >= 31:
            return 0x7BFF
    return (exp16 << 10) | (rounded >> 13)


@always_inline
def uf16_bits_to_f32(h: Int) -> Float32:
    var exp16 = (h >> 10) & 0x1F
    var mant16 = h & 0x3FF
    if exp16 == 0:
        if mant16 == 0:
            return Float32(0.0)
        return Float32(mant16) * Float32(5.960464477539063e-8)  # 2^-24
    if exp16 == 31:
        return Float32(65504.0)
    var bits32 = UInt32((exp16 - 15 + 127) << 23) | UInt32(mant16 << 13)
    return bitcast[DType.float32](bits32)


# ============================================================================
# 1. BC7 RGBA Encoder & Decoder (Mode 6: 7-bit RGBA + P-bits + 4-bit Indices)
# ============================================================================

def encode_bc7_rgba(pixels_argb: List[UInt32], width: Int, height: Int) -> List[UInt8]:
    """Compresses `0xAARRGGBB` pixels into `VK_FORMAT_BC7_UNORM` / `BC7_SRGB` (`R,G,B,A` order in Vulkan!)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt8](capacity=bw * bh * 16)

    var tr = Array[Int, 16](fill=0)
    var tg = Array[Int, 16](fill=0)
    var tb = Array[Int, 16](fill=0)
    var ta = Array[Int, 16](fill=0)
    var indices = Array[Int, 16](fill=0)

    for by in range(bh):
        for bx in range(bw):
            var min_r = 255; var max_r = 0
            var min_g = 255; var max_g = 0
            var min_b = 255; var max_b = 0
            var min_a = 255; var max_a = 0
            var mean_r = Float32(0.0)
            var mean_g = Float32(0.0)
            var mean_b = Float32(0.0)
            var mean_a = Float32(0.0)

            for ly in range(4):
                var py = min(height - 1, by * 4 + ly)
                for lx in range(4):
                    var px = min(width - 1, bx * 4 + lx)
                    var argb = pixels_argb[py * width + px]
                    var a = Int((argb >> 24) & 255)
                    var r = Int((argb >> 16) & 255)
                    var g = Int((argb >> 8) & 255)
                    var b = Int(argb & 255)
                    var t = ly * 4 + lx
                    tr[t] = r; tg[t] = g; tb[t] = b; ta[t] = a
                    if r < min_r: min_r = r
                    if r > max_r: max_r = r
                    if g < min_g: min_g = g
                    if g > max_g: max_g = g
                    if b < min_b: min_b = b
                    if b > max_b: max_b = b
                    if a < min_a: min_a = a
                    if a > max_a: max_a = a
                    mean_r += Float32(r)
                    mean_g += Float32(g)
                    mean_b += Float32(b)
                    mean_a += Float32(a)

            mean_r *= Float32(0.0625)
            mean_g *= Float32(0.0625)
            mean_b *= Float32(0.0625)
            mean_a *= Float32(0.0625)

            # 4-iteration power-method principal axis in 4D RGBA space for crisp diagonal gradients
            var ax_r = Float32(max_r - min_r)
            var ax_g = Float32(max_g - min_g)
            var ax_b = Float32(max_b - min_b)
            var ax_a = Float32(max_a - min_a)
            for _ in range(4):
                var nr = Float32(0.0); var ng = Float32(0.0); var nb = Float32(0.0); var na = Float32(0.0)
                for t in range(16):
                    var dr = Float32(tr[t]) - mean_r
                    var dg = Float32(tg[t]) - mean_g
                    var db = Float32(tb[t]) - mean_b
                    var da = Float32(ta[t]) - mean_a
                    var dot_v = dr * ax_r + dg * ax_g + db * ax_b + da * ax_a
                    nr += dr * dot_v
                    ng += dg * dot_v
                    nb += db * dot_v
                    na += da * dot_v
                var len2 = nr * nr + ng * ng + nb * nb + na * na
                if len2 > Float32(1e-6):
                    var inv_l = Float32(1.0) / sqrt(len2)
                    ax_r = nr * inv_l; ax_g = ng * inv_l; ax_b = nb * inv_l; ax_a = na * inv_l

            var r0: Int; var g0: Int; var b0: Int; var a0: Int
            var r1: Int; var g1: Int; var b1: Int; var a1: Int
            var alen2 = ax_r * ax_r + ax_g * ax_g + ax_b * ax_b + ax_a * ax_a
            if alen2 < Float32(1e-5):
                r0 = min_r; g0 = min_g; b0 = min_b; a0 = min_a
                r1 = max_r; g1 = max_g; b1 = max_b; a1 = max_a
            else:
                var min_p = Float32(1e9)
                var max_p = Float32(-1e9)
                for t in range(16):
                    var p = (Float32(tr[t]) - mean_r) * ax_r + (Float32(tg[t]) - mean_g) * ax_g + (Float32(tb[t]) - mean_b) * ax_b + (Float32(ta[t]) - mean_a) * ax_a
                    if p < min_p: min_p = p
                    if p > max_p: max_p = p
                r0 = max(0, min(255, Int(mean_r + ax_r * min_p + Float32(0.5))))
                g0 = max(0, min(255, Int(mean_g + ax_g * min_p + Float32(0.5))))
                b0 = max(0, min(255, Int(mean_b + ax_b * min_p + Float32(0.5))))
                a0 = max(0, min(255, Int(mean_a + ax_a * min_p + Float32(0.5))))
                r1 = max(0, min(255, Int(mean_r + ax_r * max_p + Float32(0.5))))
                g1 = max(0, min(255, Int(mean_g + ax_g * max_p + Float32(0.5))))
                b1 = max(0, min(255, Int(mean_b + ax_b * max_p + Float32(0.5))))
                a1 = max(0, min(255, Int(mean_a + ax_a * max_p + Float32(0.5))))

            # Snap each endpoint's shared P-bit (0 or 1) to minimize endpoint quantization error
            var best_p0 = 0
            var best_e0 = 1_000_000
            for p_cand in range(2):
                var qr = (((r0 - p_cand + 1) >> 1) << 1) | p_cand
                var qg = (((g0 - p_cand + 1) >> 1) << 1) | p_cand
                var qb = (((b0 - p_cand + 1) >> 1) << 1) | p_cand
                var qa = (((a0 - p_cand + 1) >> 1) << 1) | p_cand
                qr = max(p_cand, min(254 + p_cand, qr))
                qg = max(p_cand, min(254 + p_cand, qg))
                qb = max(p_cand, min(254 + p_cand, qb))
                qa = max(p_cand, min(254 + p_cand, qa))
                var err = (qr - r0) * (qr - r0) + (qg - g0) * (qg - g0) + (qb - b0) * (qb - b0) + (qa - a0) * (qa - a0)
                if err < best_e0:
                    best_e0 = err
                    best_p0 = p_cand
            var p0 = best_p0
            var cr0 = max(0, min(127, (r0 - p0 + 1) >> 1))
            var cg0 = max(0, min(127, (g0 - p0 + 1) >> 1))
            var cb0 = max(0, min(127, (b0 - p0 + 1) >> 1))
            var ca0 = max(0, min(127, (a0 - p0 + 1) >> 1))

            var best_p1 = 0
            var best_e1 = 1_000_000
            for p_cand in range(2):
                var qr = max(p_cand, min(254 + p_cand, (((r1 - p_cand + 1) >> 1) << 1) | p_cand))
                var qg = max(p_cand, min(254 + p_cand, (((g1 - p_cand + 1) >> 1) << 1) | p_cand))
                var qb = max(p_cand, min(254 + p_cand, (((b1 - p_cand + 1) >> 1) << 1) | p_cand))
                var qa = max(p_cand, min(254 + p_cand, (((a1 - p_cand + 1) >> 1) << 1) | p_cand))
                var err = (qr - r1) * (qr - r1) + (qg - g1) * (qg - g1) + (qb - b1) * (qb - b1) + (qa - a1) * (qa - a1)
                if err < best_e1:
                    best_e1 = err
                    best_p1 = p_cand
            var p1 = best_p1
            var cr1 = max(0, min(127, (r1 - p1 + 1) >> 1))
            var cg1 = max(0, min(127, (g1 - p1 + 1) >> 1))
            var cb1 = max(0, min(127, (b1 - p1 + 1) >> 1))
            var ca1 = max(0, min(127, (a1 - p1 + 1) >> 1))

            var er0 = (cr0 << 1) | p0; var er1 = (cr1 << 1) | p1
            var eg0 = (cg0 << 1) | p0; var eg1 = (cg1 << 1) | p1
            var eb0 = (cb0 << 1) | p0; var eb1 = (cb1 << 1) | p1
            var ea0 = (ca0 << 1) | p0; var ea1 = (ca1 << 1) | p1

            # Choose optimal 4-bit weight per texel
            for t in range(16):
                var best_idx = 0
                var best_d = 1_000_000_000
                for wi in range(16):
                    var w = _bc7_weight4(wi)
                    var ir = ((64 - w) * er0 + w * er1 + 32) >> 6
                    var ig = ((64 - w) * eg0 + w * eg1 + 32) >> 6
                    var ib = ((64 - w) * eb0 + w * eb1 + 32) >> 6
                    var ia = ((64 - w) * ea0 + w * ea1 + 32) >> 6
                    var d = (ir - tr[t]) * (ir - tr[t]) + (ig - tg[t]) * (ig - tg[t]) + (ib - tb[t]) * (ib - tb[t]) + (ia - ta[t]) * (ia - ta[t])
                    if d < best_d:
                        best_d = d
                        best_idx = wi
                indices[t] = best_idx

            # Ensure anchor texel 0 has MSB == 0 (indices[0] < 8)
            if indices[0] >= 8:
                var tmp = cr0; cr0 = cr1; cr1 = tmp
                tmp = cg0; cg0 = cg1; cg1 = tmp
                tmp = cb0; cb0 = cb1; cb1 = tmp
                tmp = ca0; ca0 = ca1; ca1 = tmp
                tmp = p0; p0 = p1; p1 = tmp
                for t in range(16):
                    indices[t] = 15 - indices[t]

            var bp = _BitPacker128()
            bp.put(0x40, 7)  # Mode 6 = 0b1000000
            bp.put(cr0, 7); bp.put(cr1, 7)
            bp.put(cg0, 7); bp.put(cg1, 7)
            bp.put(cb0, 7); bp.put(cb1, 7)
            bp.put(ca0, 7); bp.put(ca1, 7)
            bp.put(p0, 1); bp.put(p1, 1)
            bp.put(indices[0], 3)
            for t in range(1, 16):
                bp.put(indices[t], 4)
            bp.append_to(out)

    return out^


def decode_bc7_rgba(blocks: List[UInt8], width: Int, height: Int) -> List[UInt32]:
    """Reference Khronos Mode-6 `BC7` decoder returning `0xAARRGGBB` pixels."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt32](length=width * height, fill=0)

    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            var br = _BitReader128(blocks, off)
            var _mode = br.get(7)
            var cr0 = br.get(7); var cr1 = br.get(7)
            var cg0 = br.get(7); var cg1 = br.get(7)
            var cb0 = br.get(7); var cb1 = br.get(7)
            var ca0 = br.get(7); var ca1 = br.get(7)
            var p0 = br.get(1); var p1 = br.get(1)
            var er0 = (cr0 << 1) | p0; var er1 = (cr1 << 1) | p1
            var eg0 = (cg0 << 1) | p0; var eg1 = (cg1 << 1) | p1
            var eb0 = (cb0 << 1) | p0; var eb1 = (cb1 << 1) | p1
            var ea0 = (ca0 << 1) | p0; var ea1 = (ca1 << 1) | p1

            for t in range(16):
                var idx = br.get(3) if t == 0 else br.get(4)
                var w = _bc7_weight4(idx)
                var r = ((64 - w) * er0 + w * er1 + 32) >> 6
                var g = ((64 - w) * eg0 + w * eg1 + 32) >> 6
                var b = ((64 - w) * eb0 + w * eb1 + 32) >> 6
                var a = ((64 - w) * ea0 + w * ea1 + 32) >> 6
                var py = by * 4 + (t >> 2)
                var px = bx * 4 + (t & 3)
                if px < width and py < height:
                    out[py * width + px] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)

    return out^


# ============================================================================
# 2. BC5 Dual-Channel Tangent Normal Encoder & Decoder (`VK_FORMAT_BC5_UNORM`)
# ============================================================================

def _encode_bc4_channel(vals: Array[Int, 16], mut out: List[UInt8]):
    var lo = 255
    var hi = 0
    for t in range(16):
        if vals[t] < lo: lo = vals[t]
        if vals[t] > hi: hi = vals[t]

    if hi == lo:
        out.append(UInt8(hi))
        out.append(UInt8(lo))
        for _ in range(6):
            out.append(UInt8(0))
        return

    # e0 > e1 activates the 8-point linear interpolation palette
    var e0 = hi
    var e1 = lo
    out.append(UInt8(e0))
    out.append(UInt8(e1))

    var pal = Array[Int, 8](fill=0)
    pal[0] = e0
    pal[1] = e1
    pal[2] = (6 * e0 + 1 * e1 + 3) // 7
    pal[3] = (5 * e0 + 2 * e1 + 3) // 7
    pal[4] = (4 * e0 + 3 * e1 + 3) // 7
    pal[5] = (3 * e0 + 4 * e1 + 3) // 7
    pal[6] = (2 * e0 + 5 * e1 + 3) // 7
    pal[7] = (1 * e0 + 6 * e1 + 3) // 7

    var bits48: UInt64 = 0
    for t in range(16):
        var v = vals[t]
        var best_i = 0
        var best_d = 1_000_000
        for pi in range(8):
            var d = abs(pal[pi] - v)
            if d < best_d:
                best_d = d
                best_i = pi
        bits48 |= UInt64(best_i) << UInt64(t * 3)

    for bi in range(6):
        out.append(UInt8((bits48 >> UInt64(bi * 8)) & 0xFF))


def _decode_bc4_channel(blocks: List[UInt8], off: Int) -> Array[Int, 16]:
    var e0 = Int(blocks[off])
    var e1 = Int(blocks[off + 1])
    var pal = Array[Int, 8](fill=e0)
    pal[0] = e0
    pal[1] = e1
    if e0 > e1:
        pal[2] = (6 * e0 + 1 * e1 + 3) // 7
        pal[3] = (5 * e0 + 2 * e1 + 3) // 7
        pal[4] = (4 * e0 + 3 * e1 + 3) // 7
        pal[5] = (3 * e0 + 4 * e1 + 3) // 7
        pal[6] = (2 * e0 + 5 * e1 + 3) // 7
        pal[7] = (1 * e0 + 6 * e1 + 3) // 7
    else:
        pal[2] = (4 * e0 + 1 * e1 + 2) // 5
        pal[3] = (3 * e0 + 2 * e1 + 2) // 5
        pal[4] = (2 * e0 + 3 * e1 + 2) // 5
        pal[5] = (1 * e0 + 4 * e1 + 2) // 5
        pal[6] = 0
        pal[7] = 255

    var bits48: UInt64 = 0
    for bi in range(6):
        bits48 |= UInt64(blocks[off + 2 + bi]) << UInt64(bi * 8)
    var res = Array[Int, 16](fill=0)
    for t in range(16):
        var idx = Int((bits48 >> UInt64(t * 3)) & 7)
        res[t] = pal[idx]
    return res^


def encode_bc5_normals(normals: List[Vec3], width: Int, height: Int) -> List[UInt8]:
    """Compresses unit tangent-space normals into `VK_FORMAT_BC5_UNORM` (`16` bytes per `4x4` block)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt8](capacity=bw * bh * 16)
    var ch_x = Array[Int, 16](fill=128)
    var ch_y = Array[Int, 16](fill=128)

    for by in range(bh):
        for bx in range(bw):
            for ly in range(4):
                var py = min(height - 1, by * 4 + ly)
                for lx in range(4):
                    var px = min(width - 1, bx * 4 + lx)
                    var n = normals[py * width + px].normalized()
                    var t = ly * 4 + lx
                    ch_x[t] = max(0, min(255, Int((n.x * Float32(0.5) + Float32(0.5)) * Float32(255.0) + Float32(0.5))))
                    ch_y[t] = max(0, min(255, Int((n.y * Float32(0.5) + Float32(0.5)) * Float32(255.0) + Float32(0.5))))
            _encode_bc4_channel(ch_x, out)
            _encode_bc4_channel(ch_y, out)

    return out^


def decode_bc5_normals(blocks: List[UInt8], width: Int, height: Int) -> List[Vec3]:
    """Reference `BC5` decoder reconstructing exact unit `(Nx, Ny, Nz)` tangent normals."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[Vec3](length=width * height, fill=Vec3(0.0, 0.0, 1.0))

    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            var rx = _decode_bc4_channel(blocks, off)
            var ry = _decode_bc4_channel(blocks, off + 8)
            for t in range(16):
                var py = by * 4 + (t >> 2)
                var px = bx * 4 + (t & 3)
                if px < width and py < height:
                    var nx = (Float32(rx[t]) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                    var ny = (Float32(ry[t]) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                    var nz = sqrt(max(Float32(0.0), Float32(1.0) - nx * nx - ny * ny))
                    out[py * width + px] = Vec3(nx, ny, nz).normalized()

    return out^


# ============================================================================
# 3. BC6H Unsigned Half-Float HDR Encoder & Decoder (`VK_FORMAT_BC6H_UFLOAT`)
# ============================================================================

@always_inline
def _bc6h_unquantize_u10(q: Int) -> Int:
    if q <= 0:
        return 0
    if q >= 1023:
        return 0xFFFF
    return ((q << 16) + 0x8000) >> 10


@always_inline
def _bc6h_quantize_uf16_to_u10(f16_bits: Int) -> Int:
    if f16_bits <= 0:
        return 0
    if f16_bits >= 0x7BFF:
        return 1023
    var interp = (f16_bits << 6) // 31
    return max(0, min(1023, interp >> 6))


def encode_bc6h_hdr(hdr_rgb: List[Vec3], width: Int, height: Int) -> List[UInt8]:
    """Compresses linear HDR `Vec3` radiance (`[0 .. 65504]`) into `VK_FORMAT_BC6H_UFLOAT` (Mode 11 `0x03`)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt8](capacity=bw * bh * 16)

    var hr = Array[Int, 16](fill=0)
    var hg = Array[Int, 16](fill=0)
    var hb = Array[Int, 16](fill=0)
    var indices = Array[Int, 16](fill=0)

    for by in range(bh):
        for bx in range(bw):
            var min_r = 0x7BFF; var max_r = 0
            var min_g = 0x7BFF; var max_g = 0
            var min_b = 0x7BFF; var max_b = 0
            for ly in range(4):
                var py = min(height - 1, by * 4 + ly)
                for lx in range(4):
                    var px = min(width - 1, bx * 4 + lx)
                    var c = hdr_rgb[py * width + px]
                    var t = ly * 4 + lx
                    var fr = f32_to_uf16_bits(c.x)
                    var fg = f32_to_uf16_bits(c.y)
                    var fb = f32_to_uf16_bits(c.z)
                    hr[t] = fr; hg[t] = fg; hb[t] = fb
                    if fr < min_r: min_r = fr
                    if fr > max_r: max_r = fr
                    if fg < min_g: min_g = fg
                    if fg > max_g: max_g = fg
                    if fb < min_b: min_b = fb
                    if fb > max_b: max_b = fb

            var qr0 = _bc6h_quantize_uf16_to_u10(min_r)
            var qg0 = _bc6h_quantize_uf16_to_u10(min_g)
            var qb0 = _bc6h_quantize_uf16_to_u10(min_b)
            var qr1 = _bc6h_quantize_uf16_to_u10(max_r)
            var qg1 = _bc6h_quantize_uf16_to_u10(max_g)
            var qb1 = _bc6h_quantize_uf16_to_u10(max_b)

            var ur0 = _bc6h_unquantize_u10(qr0); var ur1 = _bc6h_unquantize_u10(qr1)
            var ug0 = _bc6h_unquantize_u10(qg0); var ug1 = _bc6h_unquantize_u10(qg1)
            var ub0 = _bc6h_unquantize_u10(qb0); var ub1 = _bc6h_unquantize_u10(qb1)

            for t in range(16):
                var best_idx = 0
                var best_d = 1_000_000_000
                for wi in range(16):
                    var w = _bc7_weight4(wi)
                    var ir = (((((64 - w) * ur0 + w * ur1 + 32) >> 6) * 31) >> 6)
                    var ig = (((((64 - w) * ug0 + w * ug1 + 32) >> 6) * 31) >> 6)
                    var ib = (((((64 - w) * ub0 + w * ub1 + 32) >> 6) * 31) >> 6)
                    var d = (ir - hr[t]) * (ir - hr[t]) + (ig - hg[t]) * (ig - hg[t]) + (ib - hb[t]) * (ib - hb[t])
                    if d < best_d:
                        best_d = d
                        best_idx = wi
                indices[t] = best_idx

            if indices[0] >= 8:
                var tmp = qr0; qr0 = qr1; qr1 = tmp
                tmp = qg0; qg0 = qg1; qg1 = tmp
                tmp = qb0; qb0 = qb1; qb1 = tmp
                for t in range(16):
                    indices[t] = 15 - indices[t]

            var bp = _BitPacker128()
            bp.put(0x03, 5)  # Mode 11 = 0b00011 (10-bit unsigned RGB endpoints)
            bp.put(qr0, 10); bp.put(qg0, 10); bp.put(qb0, 10)
            bp.put(qr1, 10); bp.put(qg1, 10); bp.put(qb1, 10)
            bp.put(indices[0], 3)
            for t in range(1, 16):
                bp.put(indices[t], 4)
            bp.append_to(out)

    return out^


def decode_bc6h_hdr(blocks: List[UInt8], width: Int, height: Int) -> List[Vec3]:
    """Reference Khronos `BC6H_UFLOAT` Mode-11 (`0x03`) decoder returning linear HDR `Vec3` radiance."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[Vec3](length=width * height, fill=Vec3(0.0, 0.0, 0.0))

    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            var br = _BitReader128(blocks, off)
            var _mode = br.get(5)
            var qr0 = br.get(10); var qg0 = br.get(10); var qb0 = br.get(10)
            var qr1 = br.get(10); var qg1 = br.get(10); var qb1 = br.get(10)
            var ur0 = _bc6h_unquantize_u10(qr0); var ur1 = _bc6h_unquantize_u10(qr1)
            var ug0 = _bc6h_unquantize_u10(qg0); var ug1 = _bc6h_unquantize_u10(qg1)
            var ub0 = _bc6h_unquantize_u10(qb0); var ub1 = _bc6h_unquantize_u10(qb1)

            for t in range(16):
                var idx = br.get(3) if t == 0 else br.get(4)
                var w = _bc7_weight4(idx)
                var hr = (((((64 - w) * ur0 + w * ur1 + 32) >> 6) * 31) >> 6)
                var hg = (((((64 - w) * ug0 + w * ug1 + 32) >> 6) * 31) >> 6)
                var hb = (((((64 - w) * ub0 + w * ub1 + 32) >> 6) * 31) >> 6)
                var py = by * 4 + (t >> 2)
                var px = bx * 4 + (t & 3)
                if px < width and py < height:
                    out[py * width + px] = Vec3(
                        uf16_bits_to_f32(hr),
                        uf16_bits_to_f32(hg),
                        uf16_bits_to_f32(hb),
                    )

    return out^


# ============================================================================
# Box-Filtered Mipmap Chain Builders for BC7 & BC5 Upload
# ============================================================================

def downsample_argb_2x(pixels: List[UInt32], w: Int, h: Int) -> Tuple[List[UInt32], Int, Int]:
    var nw = max(1, w >> 1)
    var nh = max(1, h >> 1)
    var out = List[UInt32](length=nw * nh, fill=0)
    for y in range(nh):
        var y0 = min(h - 1, y * 2)
        var y1 = min(h - 1, y * 2 + 1)
        for x in range(nw):
            var x0 = min(w - 1, x * 2)
            var x1 = min(w - 1, x * 2 + 1)
            var p00 = pixels[y0 * w + x0]
            var p10 = pixels[y0 * w + x1]
            var p01 = pixels[y1 * w + x0]
            var p11 = pixels[y1 * w + x1]
            var a = (Int((p00 >> 24) & 255) + Int((p10 >> 24) & 255) + Int((p01 >> 24) & 255) + Int((p11 >> 24) & 255) + 2) >> 2
            var r = (Int((p00 >> 16) & 255) + Int((p10 >> 16) & 255) + Int((p01 >> 16) & 255) + Int((p11 >> 16) & 255) + 2) >> 2
            var g = (Int((p00 >> 8) & 255) + Int((p10 >> 8) & 255) + Int((p01 >> 8) & 255) + Int((p11 >> 8) & 255) + 2) >> 2
            var b = (Int(p00 & 255) + Int(p10 & 255) + Int(p01 & 255) + Int(p11 & 255) + 2) >> 2
            out[y * nw + x] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    return (out^, nw, nh)


def downsample_normals_2x(normals: List[Vec3], w: Int, h: Int) -> Tuple[List[Vec3], Int, Int]:
    var nw = max(1, w >> 1)
    var nh = max(1, h >> 1)
    var out = List[Vec3](length=nw * nh, fill=Vec3(0.0, 0.0, 1.0))
    for y in range(nh):
        var y0 = min(h - 1, y * 2)
        var y1 = min(h - 1, y * 2 + 1)
        for x in range(nw):
            var x0 = min(w - 1, x * 2)
            var x1 = min(w - 1, x * 2 + 1)
            var sum_n = normals[y0 * w + x0] + normals[y0 * w + x1] + normals[y1 * w + x0] + normals[y1 * w + x1]
            out[y * nw + x] = sum_n.normalized()
    return (out^, nw, nh)
