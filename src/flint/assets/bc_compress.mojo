"""Pure-Mojo Hardware Texture Block Encoders, `SIMD` + `JobPool` Parallel Cookers & Decoders (`BC7`, `BC5`, `BC6H`).

Implements Khronos Data Format Specification 1.3 compliant block compression so
Flint uploads 4:1 / 6:1 compressed `VkImage` mip chains directly to GPU hardware TMUs:
  - `BC7` (`VK_FORMAT_BC7_UNORM` / `BC7_SRGB`, Multi-Mode `1, 3, 6`):
    - Mode 6 (`0b1000000`): 1-subset 7-bit RGBA + 2 P-bits + 4-bit weights.
    - Mode 3 (`0b0001000`): 2-subset (`64` partitions) 7-bit RGB + 4 P-bits + 2-bit weights.
    - Mode 1 (`0b0000010`): 2-subset (`64` partitions) 6-bit RGB + 2 shared P-bits + 3-bit weights.
    Vectorized across `SIMD[DType.float32, 8]` + `JobPool` (`PSNR > 46 dB`, `> 10x` speedup).
  - `BC5` (`VK_FORMAT_BC5_UNORM`): Dual independent 8-bit endpoint + 3-bit weight channels for
    tangent-space `(Nx, Ny)` normals (`Nz = sqrt(max(0, 1 - Nx^2 - Ny^2))`, `< 0.35°` error).
  - `BC6H` (`VK_FORMAT_BC6H_UFLOAT`, Multi-Mode `11` & `10`):
    - Mode 11 (`0x03`): 1-subset 10-bit unsigned half-float RGB + 4-bit weights.
    - Mode 10 (`0x1E`): 2-subset (`32` partitions) 6-bit unsigned half-float RGB + 3-bit weights.
  - Pre-cooked compressed `.flpk` mip-chunk packager & zero-cook runtime mip streamer.
"""
from std.ffi import external_call
from std.math import log2, max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.assets.vfs import encode_pack, pack_read
from flint.platform.os import write_bytes


@always_inline
def _bc7_weight2(idx: Int) -> Int:
    var i = max(0, min(3, idx))
    if i == 0:
        return 0
    if i == 1:
        return 21
    if i == 2:
        return 43
    return 64


@always_inline
def _bc7_weight3(idx: Int) -> Int:
    var i = max(0, min(7, idx))
    if i == 0:
        return 0
    if i == 1:
        return 9
    if i == 2:
        return 18
    if i == 3:
        return 27
    if i == 4:
        return 37
    if i == 5:
        return 46
    if i == 6:
        return 55
    return 64


@always_inline
def _bc7_weight4(idx: Int) -> Int:
    var i = max(0, min(15, idx))
    if i <= 7:
        if i == 0:
            return 0
        if i == 1:
            return 4
        if i == 2:
            return 9
        if i == 3:
            return 13
        if i == 4:
            return 17
        if i == 5:
            return 21
        if i == 6:
            return 26
        return 30
    else:
        if i == 8:
            return 34
        if i == 9:
            return 38
        if i == 10:
            return 43
        if i == 11:
            return 47
        if i == 12:
            return 51
        if i == 13:
            return 55
        if i == 14:
            return 60
        return 64


@always_inline
def _bc7_part2_mask(part: Int) -> Int:
    """Official Khronos BPTC Table.P2 64-entry 2-subset 16-bit partition masks (bit `t` = subset 0/1)."""
    var p2 = [
        0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
        0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
        0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
        0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
        0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A,
        0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
        0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C,
        0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE44,
    ]
    return p2[part & 63]


@always_inline
def _bc7_anchor2(part: Int) -> Int:
    """Official Khronos BPTC Table.A2 anchor texel index for subset 1 in 2-subset modes."""
    var a2 = [
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15,  2,  8,  2,  2,  8,  8, 15,  2,  8,  2,  2,  8,  8,  2,  2,
        15, 15,  6,  8,  2,  8, 15, 15,  2,  8,  2,  2,  2, 15, 15,  6,
         6,  2,  6,  8, 15, 15,  2,  2, 15, 15, 15, 15, 15,  2,  2,  2,
    ]
    return a2[part & 63]


struct _BitPacker128(Copyable, ImplicitlyCopyable, Movable):
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
    def write_at(self, dst_addr: Int):
        var p64 = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=dst_addr)
        p64[unsafe_offset=0] = self.lo
        p64[unsafe_offset=1] = self.hi

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
        return Float32(mant16) * Float32(5.960464477539063e-8)
    if exp16 == 31:
        return Float32(65504.0)
    var bits32 = UInt32((exp16 - 15 + 127) << 23) | UInt32(mant16 << 13)
    return bitcast[DType.float32](bits32)


# ============================================================================
# 1. 8-Wide SIMD + Multi-Mode BC7 RGBA Encoder & Decoder (`Modes 1, 3, 6`)
# ============================================================================

@always_inline
def _encode_bc7_mode6_simd_block(
    r_lo: SIMD[DType.float32, 8],
    r_hi: SIMD[DType.float32, 8],
    g_lo: SIMD[DType.float32, 8],
    g_hi: SIMD[DType.float32, 8],
    b_lo: SIMD[DType.float32, 8],
    b_hi: SIMD[DType.float32, 8],
    a_lo: SIMD[DType.float32, 8],
    a_hi: SIMD[DType.float32, 8],
    tr: Array[Int, 16],
    tg: Array[Int, 16],
    tb: Array[Int, 16],
    ta: Array[Int, 16],
) -> Tuple[_BitPacker128, Int]:
    """8-Wide SIMD PCA + Analytical Line-Projection Mode-6 BC7 block encoder returning `(packed_128b, total_sq_err)`."""
    var mean_r = (r_lo + r_hi).reduce_add() * Float32(0.0625)
    var mean_g = (g_lo + g_hi).reduce_add() * Float32(0.0625)
    var mean_b = (b_lo + b_hi).reduce_add() * Float32(0.0625)
    var mean_a = (a_lo + a_hi).reduce_add() * Float32(0.0625)

    var dr_lo = r_lo - mean_r; var dr_hi = r_hi - mean_r
    var dg_lo = g_lo - mean_g; var dg_hi = g_hi - mean_g
    var db_lo = b_lo - mean_b; var db_hi = b_hi - mean_b
    var da_lo = a_lo - mean_a; var da_hi = a_hi - mean_a

    var min_r = min(r_lo.reduce_min(), r_hi.reduce_min())
    var max_r = max(r_lo.reduce_max(), r_hi.reduce_max())
    var min_g = min(g_lo.reduce_min(), g_hi.reduce_min())
    var max_g = max(g_lo.reduce_max(), g_hi.reduce_max())
    var min_b = min(b_lo.reduce_min(), b_hi.reduce_min())
    var max_b = max(b_lo.reduce_max(), b_hi.reduce_max())
    var min_a = min(a_lo.reduce_min(), a_hi.reduce_min())
    var max_a = max(a_lo.reduce_max(), a_hi.reduce_max())

    var ax_r = max_r - min_r
    var ax_g = max_g - min_g
    var ax_b = max_b - min_b
    var ax_a = max_a - min_a

    # 3-iteration 8-wide SIMD power-method principal axis in 4D RGBA space
    for _ in range(3):
        var dot_lo = dr_lo * ax_r + dg_lo * ax_g + db_lo * ax_b + da_lo * ax_a
        var dot_hi = dr_hi * ax_r + dg_hi * ax_g + db_hi * ax_b + da_hi * ax_a
        var nr = (dr_lo * dot_lo + dr_hi * dot_hi).reduce_add()
        var ng = (dg_lo * dot_lo + dg_hi * dot_hi).reduce_add()
        var nb = (db_lo * dot_lo + db_hi * dot_hi).reduce_add()
        var na = (da_lo * dot_lo + da_hi * dot_hi).reduce_add()
        var len2 = nr * nr + ng * ng + nb * nb + na * na
        if len2 > Float32(1e-6):
            var inv_l = Float32(1.0) / sqrt(len2)
            ax_r = nr * inv_l; ax_g = ng * inv_l; ax_b = nb * inv_l; ax_a = na * inv_l

    var r0: Int; var g0: Int; var b0: Int; var a0: Int
    var r1: Int; var g1: Int; var b1: Int; var a1: Int
    var alen2 = ax_r * ax_r + ax_g * ax_g + ax_b * ax_b + ax_a * ax_a
    if alen2 < Float32(1e-5):
        r0 = Int(min_r); g0 = Int(min_g); b0 = Int(min_b); a0 = Int(min_a)
        r1 = Int(max_r); g1 = Int(max_g); b1 = Int(max_b); a1 = Int(max_a)
    else:
        var proj_lo = dr_lo * ax_r + dg_lo * ax_g + db_lo * ax_b + da_lo * ax_a
        var proj_hi = dr_hi * ax_r + dg_hi * ax_g + db_hi * ax_b + da_hi * ax_a
        var min_p = min(proj_lo.reduce_min(), proj_hi.reduce_min())
        var max_p = max(proj_lo.reduce_max(), proj_hi.reduce_max())
        r0 = max(0, min(255, Int(mean_r + ax_r * min_p + Float32(0.5))))
        g0 = max(0, min(255, Int(mean_g + ax_g * min_p + Float32(0.5))))
        b0 = max(0, min(255, Int(mean_b + ax_b * min_p + Float32(0.5))))
        a0 = max(0, min(255, Int(mean_a + ax_a * min_p + Float32(0.5))))
        r1 = max(0, min(255, Int(mean_r + ax_r * max_p + Float32(0.5))))
        g1 = max(0, min(255, Int(mean_g + ax_g * max_p + Float32(0.5))))
        b1 = max(0, min(255, Int(mean_b + ax_b * max_p + Float32(0.5))))
        a1 = max(0, min(255, Int(mean_a + ax_a * max_p + Float32(0.5))))

    var p0 = 0
    var e0_best = 1_000_000
    for pc in range(2):
        var qr = max(pc, min(254 + pc, (((r0 - pc + 1) >> 1) << 1) | pc))
        var qg = max(pc, min(254 + pc, (((g0 - pc + 1) >> 1) << 1) | pc))
        var qb = max(pc, min(254 + pc, (((b0 - pc + 1) >> 1) << 1) | pc))
        var qa = max(pc, min(254 + pc, (((a0 - pc + 1) >> 1) << 1) | pc))
        var err = (qr - r0) * (qr - r0) + (qg - g0) * (qg - g0) + (qb - b0) * (qb - b0) + (qa - a0) * (qa - a0)
        if err < e0_best:
            e0_best = err
            p0 = pc
    var cr0 = max(0, min(127, (r0 - p0 + 1) >> 1))
    var cg0 = max(0, min(127, (g0 - p0 + 1) >> 1))
    var cb0 = max(0, min(127, (b0 - p0 + 1) >> 1))
    var ca0 = max(0, min(127, (a0 - p0 + 1) >> 1))

    var p1 = 0
    var e1_best = 1_000_000
    for pc in range(2):
        var qr = max(pc, min(254 + pc, (((r1 - pc + 1) >> 1) << 1) | pc))
        var qg = max(pc, min(254 + pc, (((g1 - pc + 1) >> 1) << 1) | pc))
        var qb = max(pc, min(254 + pc, (((b1 - pc + 1) >> 1) << 1) << 0 | pc))
        var qa = max(pc, min(254 + pc, (((a1 - pc + 1) >> 1) << 1) | pc))
        var err = (qr - r1) * (qr - r1) + (qg - g1) * (qg - g1) + (qb - b1) * (qb - b1) + (qa - a1) * (qa - a1)
        if err < e1_best:
            e1_best = err
            p1 = pc
    var cr1 = max(0, min(127, (r1 - p1 + 1) >> 1))
    var cg1 = max(0, min(127, (g1 - p1 + 1) >> 1))
    var cb1 = max(0, min(127, (b1 - p1 + 1) >> 1))
    var ca1 = max(0, min(127, (a1 - p1 + 1) >> 1))

    var er0 = (cr0 << 1) | p0; var er1 = (cr1 << 1) | p1
    var eg0 = (cg0 << 1) | p0; var eg1 = (cg1 << 1) | p1
    var eb0 = (cb0 << 1) | p0; var eb1 = (cb1 << 1) | p1
    var ea0 = (ca0 << 1) | p0; var ea1 = (ca1 << 1) | p1

    # Analytical 8-wide SIMD projection onto (E0 -> E1) + +/-1 neighbor refinement (replaces 256 scalar checks!)
    var dx_r = Float32(er1 - er0)
    var dx_g = Float32(eg1 - eg0)
    var dx_b = Float32(eb1 - eb0)
    var dx_a = Float32(ea1 - ea0)
    var seg_len2 = dx_r * dx_r + dx_g * dx_g + dx_b * dx_b + dx_a * dx_a

    var indices = Array[Int, 16](fill=0)
    var total_err = 0

    if seg_len2 < Float32(0.5):
        for t in range(16):
            var d = (er0 - tr[t]) * (er0 - tr[t]) + (eg0 - tg[t]) * (eg0 - tg[t]) + (eb0 - tb[t]) * (eb0 - tb[t]) + (ea0 - ta[t]) * (ea0 - ta[t])
            total_err += d
    else:
        var scale15 = Float32(15.0) / seg_len2
        var t_lo = ((r_lo - Float32(er0)) * dx_r + (g_lo - Float32(eg0)) * dx_g + (b_lo - Float32(eb0)) * dx_b + (a_lo - Float32(ea0)) * dx_a) * scale15 + Float32(0.5)
        var t_hi = ((r_hi - Float32(er0)) * dx_r + (g_hi - Float32(eg0)) * dx_g + (b_hi - Float32(eb0)) * dx_b + (a_hi - Float32(ea0)) * dx_a) * scale15 + Float32(0.5)
        for t in range(16):
            var est = Int(t_lo[t] if t < 8 else t_hi[t - 8])
            var lo_w = max(0, min(15, est - 1))
            var hi_w = max(0, min(15, est + 1))
            var best_wi = lo_w
            var best_d = 1_000_000_000
            for wi in range(lo_w, hi_w + 1):
                var w = _bc7_weight4(wi)
                var ir = ((64 - w) * er0 + w * er1 + 32) >> 6
                var ig = ((64 - w) * eg0 + w * eg1 + 32) >> 6
                var ib = ((64 - w) * eb0 + w * eb1 + 32) >> 6
                var ia = ((64 - w) * ea0 + w * ea1 + 32) >> 6
                var d = (ir - tr[t]) * (ir - tr[t]) + (ig - tg[t]) * (ig - tg[t]) + (ib - tb[t]) * (ib - tb[t]) + (ia - ta[t]) * (ia - ta[t])
                if d < best_d:
                    best_d = d
                    best_wi = wi
            indices[t] = best_wi
            total_err += best_d

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
    return (bp, total_err)


def _select_best_2subset_partition(
    tr: Array[Int, 16],
    tg: Array[Int, 16],
    tb: Array[Int, 16],
) -> Int:
    """Fast variance-reduction search across Khronos 2-subset partitions to isolate sharp edges."""
    # Check 16 canonical geometric splits (vertical, horizontal, diagonal, corner, checker)
    var candidates = [0, 1, 2, 13, 14, 15, 3, 7, 16, 17, 19, 23, 26, 29, 34, 44]
    var best_part = 0
    var best_score = 1_000_000_000

    for c_idx in range(16):
        var part = candidates[c_idx]
        var mask = _bc7_part2_mask(part)
        var sr0 = 0; var sg0 = 0; var sb0 = 0; var n0 = 0
        var sr1 = 0; var sg1 = 0; var sb1 = 0; var n1 = 0
        for t in range(16):
            if ((mask >> t) & 1) == 0:
                sr0 += tr[t]; sg0 += tg[t]; sb0 += tb[t]; n0 += 1
            else:
                sr1 += tr[t]; sg1 += tg[t]; sb1 += tb[t]; n1 += 1
        if n0 == 0 or n1 == 0:
            continue
        var mr0 = sr0 // n0; var mg0 = sg0 // n0; var mb0 = sb0 // n0
        var mr1 = sr1 // n1; var mg1 = sg1 // n1; var mb1 = sb1 // n1
        var var_sum = 0
        for t in range(16):
            if ((mask >> t) & 1) == 0:
                var_sum += (tr[t] - mr0) * (tr[t] - mr0) + (tg[t] - mg0) * (tg[t] - mg0) + (tb[t] - mb0) * (tb[t] - mb0)
            else:
                var_sum += (tr[t] - mr1) * (tr[t] - mr1) + (tg[t] - mg1) * (tg[t] - mg1) + (tb[t] - mb1) * (tb[t] - mb1)
        if var_sum < best_score:
            best_score = var_sum
            best_part = part

    return best_part


def _encode_bc7_mode3_or_mode1_block(
    part: Int,
    tr: Array[Int, 16],
    tg: Array[Int, 16],
    tb: Array[Int, 16],
    use_mode1: Bool,
) -> Tuple[_BitPacker128, Int]:
    """Encodes an opaque (`A=255`) 2-subset `BC7` block in either Mode 3 (7-bit RGB, 4 P-bits, 2-bit weights)
    or Mode 1 (6-bit RGB, 2 shared P-bits, 3-bit weights)."""
    var mask = _bc7_part2_mask(part)
    var anchor2 = _bc7_anchor2(part)

    var e_r = Array[Int, 4](fill=0)
    var e_g = Array[Int, 4](fill=0)
    var e_b = Array[Int, 4](fill=0)

    # Fit bounding-box / principal diagonal endpoints independently for Subset 0 and Subset 1
    for sub in range(2):
        var min_r = 255; var max_r = 0
        var min_g = 255; var max_g = 0
        var min_b = 255; var max_b = 0
        var mr = Float32(0.0); var mg = Float32(0.0); var mb = Float32(0.0)
        var cnt = 0
        for t in range(16):
            if ((mask >> t) & 1) == sub:
                var r = tr[t]; var g = tg[t]; var b = tb[t]
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if g < min_g: min_g = g
                if g > max_g: max_g = g
                if b < min_b: min_b = b
                if b > max_b: max_b = b
                mr += Float32(r); mg += Float32(g); mb += Float32(b)
                cnt += 1
        var inv_c = Float32(1.0) / Float32(max(1, cnt))
        mr *= inv_c; mg *= inv_c; mb *= inv_c
        var ax_r = Float32(max_r - min_r)
        var ax_g = Float32(max_g - min_g)
        var ax_b = Float32(max_b - min_b)
        for _ in range(2):
            var nr = Float32(0.0); var ng = Float32(0.0); var nb = Float32(0.0)
            for t in range(16):
                if ((mask >> t) & 1) == sub:
                    var dr = Float32(tr[t]) - mr
                    var dg = Float32(tg[t]) - mg
                    var db = Float32(tb[t]) - mb
                    var dv = dr * ax_r + dg * ax_g + db * ax_b
                    nr += dr * dv; ng += dg * dv; nb += db * dv
            var l2 = nr * nr + ng * ng + nb * nb
            if l2 > Float32(1e-6):
                var il = Float32(1.0) / sqrt(l2)
                ax_r = nr * il; ax_g = ng * il; ax_b = nb * il

        if ax_r * ax_r + ax_g * ax_g + ax_b * ax_b < Float32(1e-5):
            e_r[sub * 2] = min_r; e_g[sub * 2] = min_g; e_b[sub * 2] = min_b
            e_r[sub * 2 + 1] = max_r; e_g[sub * 2 + 1] = max_g; e_b[sub * 2 + 1] = max_b
        else:
            var min_p = Float32(1e9); var max_p = Float32(-1e9)
            for t in range(16):
                if ((mask >> t) & 1) == sub:
                    var p = (Float32(tr[t]) - mr) * ax_r + (Float32(tg[t]) - mg) * ax_g + (Float32(tb[t]) - mb) * ax_b
                    if p < min_p: min_p = p
                    if p > max_p: max_p = p
            e_r[sub * 2] = max(0, min(255, Int(mr + ax_r * min_p + Float32(0.5))))
            e_g[sub * 2] = max(0, min(255, Int(mg + ax_g * min_p + Float32(0.5))))
            e_b[sub * 2] = max(0, min(255, Int(mb + ax_b * min_p + Float32(0.5))))
            e_r[sub * 2 + 1] = max(0, min(255, Int(mr + ax_r * max_p + Float32(0.5))))
            e_g[sub * 2 + 1] = max(0, min(255, Int(mg + ax_g * max_p + Float32(0.5))))
            e_b[sub * 2 + 1] = max(0, min(255, Int(mb + ax_b * max_p + Float32(0.5))))

    var bp = _BitPacker128()
    var indices = Array[Int, 16](fill=0)
    var total_err = 0

    if not use_mode1:
        # Mode 3 (`0b0001000`, 4 bits): 6-bit partition, 7-bit RGB * 4, 4 P-bits, 2-bit weights
        var cr = Array[Int, 4](fill=0)
        var cg = Array[Int, 4](fill=0)
        var cb = Array[Int, 4](fill=0)
        var pb = Array[Int, 4](fill=0)
        var ur = Array[Int, 4](fill=0)
        var ug = Array[Int, 4](fill=0)
        var ub = Array[Int, 4](fill=0)

        for ep in range(4):
            var best_p = 0
            var best_e = 1_000_000
            for pc in range(2):
                var c7r = max(0, min(127, (e_r[ep] - pc + 1) >> 1))
                var c7g = max(0, min(127, (e_g[ep] - pc + 1) >> 1))
                var c7b = max(0, min(127, (e_b[ep] - pc + 1) >> 1))
                var qr = (c7r << 1) | pc
                var qg = (c7g << 1) | pc
                var qb = (c7b << 1) | pc
                var d = (qr - e_r[ep]) * (qr - e_r[ep]) + (qg - e_g[ep]) * (qg - e_g[ep]) + (qb - e_b[ep]) * (qb - e_b[ep])
                if d < best_e:
                    best_e = d
                    best_p = pc
            pb[ep] = best_p
            cr[ep] = max(0, min(127, (e_r[ep] - best_p + 1) >> 1))
            cg[ep] = max(0, min(127, (e_g[ep] - best_p + 1) >> 1))
            cb[ep] = max(0, min(127, (e_b[ep] - best_p + 1) >> 1))
            ur[ep] = (cr[ep] << 1) | best_p
            ug[ep] = (cg[ep] << 1) | best_p
            ub[ep] = (cb[ep] << 1) | best_p

        for t in range(16):
            var sub = (mask >> t) & 1
            var e0 = sub * 2
            var e1 = e0 + 1
            var best_wi = 0
            var best_d = 1_000_000_000
            for wi in range(4):
                var w = _bc7_weight2(wi)
                var ir = ((64 - w) * ur[e0] + w * ur[e1] + 32) >> 6
                var ig = ((64 - w) * ug[e0] + w * ug[e1] + 32) >> 6
                var ib = ((64 - w) * ub[e0] + w * ub[e1] + 32) >> 6
                var d = (ir - tr[t]) * (ir - tr[t]) + (ig - tg[t]) * (ig - tg[t]) + (ib - tb[t]) * (ib - tb[t])
                if d < best_d:
                    best_d = d
                    best_wi = wi
            indices[t] = best_wi
            total_err += best_d

        if indices[0] >= 2:
            var tmp = cr[0]; cr[0] = cr[1]; cr[1] = tmp
            tmp = cg[0]; cg[0] = cg[1]; cg[1] = tmp
            tmp = cb[0]; cb[0] = cb[1]; cb[1] = tmp
            tmp = pb[0]; pb[0] = pb[1]; pb[1] = tmp
            for t in range(16):
                if ((mask >> t) & 1) == 0:
                    indices[t] = 3 - indices[t]
        if indices[anchor2] >= 2:
            var tmp = cr[2]; cr[2] = cr[3]; cr[3] = tmp
            tmp = cg[2]; cg[2] = cg[3]; cg[3] = tmp
            tmp = cb[2]; cb[2] = cb[3]; cb[3] = tmp
            tmp = pb[2]; pb[2] = pb[3]; pb[3] = tmp
            for t in range(16):
                if ((mask >> t) & 1) == 1:
                    indices[t] = 3 - indices[t]

        bp.put(0x08, 4)  # Mode 3 = 0b1000
        bp.put(part, 6)
        for ep in range(4): bp.put(cr[ep], 7)
        for ep in range(4): bp.put(cg[ep], 7)
        for ep in range(4): bp.put(cb[ep], 7)
        for ep in range(4): bp.put(pb[ep], 1)
        for t in range(16):
            if t == 0 or t == anchor2:
                bp.put(indices[t], 1)
            else:
                bp.put(indices[t], 2)
    else:
        # Mode 1 (`0b0000010`, 2 bits): 6-bit partition, 6-bit RGB * 4, 2 shared P-bits, 3-bit weights
        var cr = Array[Int, 4](fill=0)
        var cg = Array[Int, 4](fill=0)
        var cb = Array[Int, 4](fill=0)
        var ps = Array[Int, 2](fill=0)
        var ur = Array[Int, 4](fill=0)
        var ug = Array[Int, 4](fill=0)
        var ub = Array[Int, 4](fill=0)

        for sub in range(2):
            var best_p = 0
            var best_e = 1_000_000_000
            for pc in range(2):
                var sub_e = 0
                for k in range(2):
                    var ep = sub * 2 + k
                    var c6r = max(0, min(63, (e_r[ep] - (pc << 1) + 2) >> 2))
                    var c6g = max(0, min(63, (e_g[ep] - (pc << 1) + 2) >> 2))
                    var c6b = max(0, min(63, (e_b[ep] - (pc << 1) + 2) >> 2))
                    var qr = (c6r << 2) | (pc << 1) | (c6r >> 5)
                    var qg = (c6g << 2) | (pc << 1) | (c6g >> 5)
                    var qb = (c6b << 2) | (pc << 1) | (c6b >> 5)
                    sub_e += (qr - e_r[ep]) * (qr - e_r[ep]) + (qg - e_g[ep]) * (qg - e_g[ep]) + (qb - e_b[ep]) * (qb - e_b[ep])
                if sub_e < best_e:
                    best_e = sub_e
                    best_p = pc
            ps[sub] = best_p
            for k in range(2):
                var ep = sub * 2 + k
                cr[ep] = max(0, min(63, (e_r[ep] - (best_p << 1) + 2) >> 2))
                cg[ep] = max(0, min(63, (e_g[ep] - (best_p << 1) + 2) >> 2))
                cb[ep] = max(0, min(63, (e_b[ep] - (best_p << 1) + 2) >> 2))
                ur[ep] = (cr[ep] << 2) | (best_p << 1) | (cr[ep] >> 5)
                ug[ep] = (cg[ep] << 2) | (best_p << 1) | (cg[ep] >> 5)
                ub[ep] = (cb[ep] << 2) | (best_p << 1) | (cb[ep] >> 5)

        for t in range(16):
            var sub = (mask >> t) & 1
            var e0 = sub * 2
            var e1 = e0 + 1
            var best_wi = 0
            var best_d = 1_000_000_000
            for wi in range(8):
                var w = _bc7_weight3(wi)
                var ir = ((64 - w) * ur[e0] + w * ur[e1] + 32) >> 6
                var ig = ((64 - w) * ug[e0] + w * ug[e1] + 32) >> 6
                var ib = ((64 - w) * ub[e0] + w * ub[e1] + 32) >> 6
                var d = (ir - tr[t]) * (ir - tr[t]) + (ig - tg[t]) * (ig - tg[t]) + (ib - tb[t]) * (ib - tb[t])
                if d < best_d:
                    best_d = d
                    best_wi = wi
            indices[t] = best_wi
            total_err += best_d

        if indices[0] >= 4:
            var tmp = cr[0]; cr[0] = cr[1]; cr[1] = tmp
            tmp = cg[0]; cg[0] = cg[1]; cg[1] = tmp
            tmp = cb[0]; cb[0] = cb[1]; cb[1] = tmp
            for t in range(16):
                if ((mask >> t) & 1) == 0:
                    indices[t] = 7 - indices[t]
        if indices[anchor2] >= 4:
            var tmp = cr[2]; cr[2] = cr[3]; cr[3] = tmp
            tmp = cg[2]; cg[2] = cg[3]; cg[3] = tmp
            tmp = cb[2]; cb[2] = cb[3]; cb[3] = tmp
            for t in range(16):
                if ((mask >> t) & 1) == 1:
                    indices[t] = 7 - indices[t]

        bp.put(0x02, 2)  # Mode 1 = 0b10
        bp.put(part, 6)
        for ep in range(4): bp.put(cr[ep], 6)
        for ep in range(4): bp.put(cg[ep], 6)
        for ep in range(4): bp.put(cb[ep], 6)
        bp.put(ps[0], 1); bp.put(ps[1], 1)
        for t in range(16):
            if t == 0 or t == anchor2:
                bp.put(indices[t], 2)
            else:
                bp.put(indices[t], 3)

    return (bp, total_err)


@always_inline
def _encode_bc7_tile_at(
    src_addr: Int,
    width: Int,
    height: Int,
    bx: Int,
    by: Int,
    multi_mode: Bool,
) -> _BitPacker128:
    var src_p = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=src_addr)
    var tr = Array[Int, 16](fill=0)
    var tg = Array[Int, 16](fill=0)
    var tb = Array[Int, 16](fill=0)
    var ta = Array[Int, 16](fill=0)
    var r_lo = SIMD[DType.float32, 8](0.0); var r_hi = SIMD[DType.float32, 8](0.0)
    var g_lo = SIMD[DType.float32, 8](0.0); var g_hi = SIMD[DType.float32, 8](0.0)
    var b_lo = SIMD[DType.float32, 8](0.0); var b_hi = SIMD[DType.float32, 8](0.0)
    var a_lo = SIMD[DType.float32, 8](0.0); var a_hi = SIMD[DType.float32, 8](0.0)
    var all_opaque = True

    for ly in range(4):
        var py = min(height - 1, by * 4 + ly)
        var row_off = py * width
        for lx in range(4):
            var px = min(width - 1, bx * 4 + lx)
            var argb = src_p[unsafe_offset=row_off + px]
            var a = Int((argb >> 24) & 255)
            var r = Int((argb >> 16) & 255)
            var g = Int((argb >> 8) & 255)
            var b = Int(argb & 255)
            if a < 255:
                all_opaque = False
            var t = ly * 4 + lx
            tr[t] = r; tg[t] = g; tb[t] = b; ta[t] = a
            if t < 8:
                r_lo[t] = Float32(r); g_lo[t] = Float32(g); b_lo[t] = Float32(b); a_lo[t] = Float32(a)
            else:
                r_hi[t - 8] = Float32(r); g_hi[t - 8] = Float32(g); b_hi[t - 8] = Float32(b); a_hi[t - 8] = Float32(a)

    var m6 = _encode_bc7_mode6_simd_block(r_lo, r_hi, g_lo, g_hi, b_lo, b_hi, a_lo, a_hi, tr, tg, tb, ta)
    var best_bp = m6[0]
    var best_err = m6[1]

    # Only evaluate 2-subset Modes 1 & 3 on opaque blocks where Mode 6 left residual error (> 32 sq err)
    if multi_mode and all_opaque and best_err > 32:
        var part = _select_best_2subset_partition(tr, tg, tb)
        var m3 = _encode_bc7_mode3_or_mode1_block(part, tr, tg, tb, use_mode1=False)
        if m3[1] < best_err:
            best_err = m3[1]
            best_bp = m3[0]
        var m1 = _encode_bc7_mode3_or_mode1_block(part, tr, tg, tb, use_mode1=True)
        if m1[1] < best_err:
            best_bp = m1[0]

    return best_bp


struct _Bc7ParallelCtx(Movable):
    var src_addr: Int
    var dst_addr: Int
    var width: Int
    var height: Int
    var bw: Int
    var bh: Int
    var rows_per_job: Int
    var multi_mode: Bool

    def __init__(
        out self,
        src_addr: Int,
        dst_addr: Int,
        width: Int,
        height: Int,
        bw: Int,
        bh: Int,
        rows_per_job: Int,
        multi_mode: Bool,
    ):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.width = width
        self.height = height
        self.bw = bw
        self.bh = bh
        self.rows_per_job = rows_per_job
        self.multi_mode = multi_mode


def _bc7_parallel_worker(ctx_addr: Int, job_idx: Int):
    var ctx = Pointer[_Bc7ParallelCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var y0 = job_idx * ctx[].rows_per_job
    var y1 = min(ctx[].bh, y0 + ctx[].rows_per_job)
    for by in range(y0, y1):
        for bx in range(ctx[].bw):
            var bp = _encode_bc7_tile_at(ctx[].src_addr, ctx[].width, ctx[].height, bx, by, ctx[].multi_mode)
            bp.write_at(ctx[].dst_addr + (by * ctx[].bw + bx) * 16)


def encode_bc7_rgba_simd(
    pixels_argb: List[UInt32],
    width: Int,
    height: Int,
    mut pool: JobPool,
    multi_mode: Bool = True,
) -> List[UInt8]:
    """Multi-threaded `8-wide SIMD` + `JobPool` `BC7` encoder (`Modes 1, 3, 6`)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt8](length=bw * bh * 16, fill=0)
    if len(pixels_argb) < width * height or width <= 0 or height <= 0:
        return out^

    var jobs = min(bh, max(1, bh // 2))
    if jobs > 16:
        jobs = 16
    var rows_per_job = (bh + jobs - 1) // jobs
    var actual_jobs = (bh + rows_per_job - 1) // rows_per_job

    var ctx = _Bc7ParallelCtx(
        Int(pixels_argb.unsafe_ptr()),
        Int(out.unsafe_ptr()),
        width,
        height,
        bw,
        bh,
        rows_per_job,
        multi_mode,
    )
    run_parallel_ctx(pool, actual_jobs, ctx, _bc7_parallel_worker)
    var anchor_in = Int(pixels_argb.unsafe_ptr())
    var anchor_out = Int(out.unsafe_ptr())
    _ = anchor_in
    _ = anchor_out
    return out^


def encode_bc7_rgba(pixels_argb: List[UInt32], width: Int, height: Int, multi_mode: Bool = True) -> List[UInt8]:
    """Compresses `0xAARRGGBB` pixels into `VK_FORMAT_BC7_UNORM` / `BC7_SRGB` using 8-wide SIMD."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt8](length=bw * bh * 16, fill=0)
    if len(pixels_argb) < width * height or width <= 0 or height <= 0:
        return out^
    var src_addr = Int(pixels_argb.unsafe_ptr())
    var dst_addr = Int(out.unsafe_ptr())
    for by in range(bh):
        for bx in range(bw):
            var bp = _encode_bc7_tile_at(src_addr, width, height, bx, by, multi_mode)
            bp.write_at(dst_addr + (by * bw + bx) * 16)
    var anchor = src_addr
    _ = anchor
    return out^


def encode_bc7_rgba_scalar_reference(pixels_argb: List[UInt32], width: Int, height: Int) -> List[UInt8]:
    """Baseline scalar Mode-6 `BC7` encoder retained for `Suite 21` throughput & PSNR benchmarking."""
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

            var cr0 = min_r >> 1; var cr1 = max_r >> 1
            var cg0 = min_g >> 1; var cg1 = max_g >> 1
            var cb0 = min_b >> 1; var cb1 = max_b >> 1
            var ca0 = min_a >> 1; var ca1 = max_a >> 1
            var p0 = min_r & 1; var p1 = max_r & 1
            var er0 = (cr0 << 1) | p0; var er1 = (cr1 << 1) | p1
            var eg0 = (cg0 << 1) | p0; var eg1 = (cg1 << 1) | p1
            var eb0 = (cb0 << 1) | p0; var eb1 = (cb1 << 1) | p1
            var ea0 = (ca0 << 1) | p0; var ea1 = (ca1 << 1) | p1

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

            if indices[0] >= 8:
                var tmp = cr0; cr0 = cr1; cr1 = tmp
                tmp = cg0; cg0 = cg1; cg1 = tmp
                tmp = cb0; cb0 = cb1; cb1 = tmp
                tmp = ca0; ca0 = ca1; ca1 = tmp
                tmp = p0; p0 = p1; p1 = tmp
                for t in range(16):
                    indices[t] = 15 - indices[t]

            var bp = _BitPacker128()
            bp.put(0x40, 7)
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
    """Reference Khronos `BC7` decoder supporting Modes 1, 3, and 6 (`0xAARRGGBB`)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt32](length=width * height, fill=0)

    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            var br = _BitReader128(blocks, off)
            # Count trailing zeros of mode bit (0..7)
            var mode = 0
            while mode < 8:
                if br.get(1) == 1:
                    break
                mode += 1

            if mode == 6:
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
            elif mode == 3:
                var part = br.get(6)
                var mask = _bc7_part2_mask(part)
                var anchor2 = _bc7_anchor2(part)
                var cr = Array[Int, 4](fill=0)
                var cg = Array[Int, 4](fill=0)
                var cb = Array[Int, 4](fill=0)
                var pb = Array[Int, 4](fill=0)
                for ep in range(4): cr[ep] = br.get(7)
                for ep in range(4): cg[ep] = br.get(7)
                for ep in range(4): cb[ep] = br.get(7)
                for ep in range(4): pb[ep] = br.get(1)
                for t in range(16):
                    var idx = br.get(1) if (t == 0 or t == anchor2) else br.get(2)
                    var w = _bc7_weight2(idx)
                    var sub = (mask >> t) & 1
                    var e0 = sub * 2; var e1 = e0 + 1
                    var ur0 = (cr[e0] << 1) | pb[e0]; var ur1 = (cr[e1] << 1) | pb[e1]
                    var ug0 = (cg[e0] << 1) | pb[e0]; var ug1 = (cg[e1] << 1) | pb[e1]
                    var ub0 = (cb[e0] << 1) | pb[e0]; var ub1 = (cb[e1] << 1) | pb[e1]
                    var r = ((64 - w) * ur0 + w * ur1 + 32) >> 6
                    var g = ((64 - w) * ug0 + w * ug1 + 32) >> 6
                    var b = ((64 - w) * ub0 + w * ub1 + 32) >> 6
                    var py = by * 4 + (t >> 2)
                    var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
            elif mode == 1:
                var part = br.get(6)
                var mask = _bc7_part2_mask(part)
                var anchor2 = _bc7_anchor2(part)
                var cr = Array[Int, 4](fill=0)
                var cg = Array[Int, 4](fill=0)
                var cb = Array[Int, 4](fill=0)
                for ep in range(4): cr[ep] = br.get(6)
                for ep in range(4): cg[ep] = br.get(6)
                for ep in range(4): cb[ep] = br.get(6)
                var ps = Array[Int, 2](fill=0)
                ps[0] = br.get(1); ps[1] = br.get(1)
                for t in range(16):
                    var idx = br.get(2) if (t == 0 or t == anchor2) else br.get(3)
                    var w = _bc7_weight3(idx)
                    var sub = (mask >> t) & 1
                    var p = ps[sub]
                    var e0 = sub * 2; var e1 = e0 + 1
                    var ur0 = (cr[e0] << 2) | (p << 1) | (cr[e0] >> 5)
                    var ur1 = (cr[e1] << 2) | (p << 1) | (cr[e1] >> 5)
                    var ug0 = (cg[e0] << 2) | (p << 1) | (cg[e0] >> 5)
                    var ug1 = (cg[e1] << 2) | (p << 1) | (cg[e1] >> 5)
                    var ub0 = (cb[e0] << 2) | (p << 1) | (cb[e0] >> 5)
                    var ub1 = (cb[e1] << 2) | (p << 1) | (cb[e1] >> 5)
                    var r = ((64 - w) * ur0 + w * ur1 + 32) >> 6
                    var g = ((64 - w) * ug0 + w * ug1 + 32) >> 6
                    var b = ((64 - w) * ub0 + w * ub1 + 32) >> 6
                    var py = by * 4 + (t >> 2)
                    var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)

    return out^


def compute_argb_psnr_db(ref_px: List[UInt32], cmp_px: List[UInt32]) -> Float32:
    """Computes RGBA Peak Signal-to-Noise Ratio (`PSNR` in `dB`, capped at `99.0 dB` when lossless)."""
    var n = min(len(ref_px), len(cmp_px))
    if n <= 0:
        return Float32(0.0)
    var sq_sum = Float64(0.0)
    for i in range(n):
        var a = ref_px[i]
        var b = cmp_px[i]
        var dr = Float64(Int((a >> 16) & 255) - Int((b >> 16) & 255))
        var dg = Float64(Int((a >> 8) & 255) - Int((b >> 8) & 255))
        var db = Float64(Int(a & 255) - Int(b & 255))
        var da = Float64(Int((a >> 24) & 255) - Int((b >> 24) & 255))
        sq_sum += dr * dr + dg * dg + db * db + da * da
    var mse = sq_sum / Float64(n * 4)
    if mse < 1e-9:
        return Float32(99.0)
    # 10 * log10(255^2 / mse) = 3.01029995664 * log2(65025.0 / mse)
    return Float32(3.01029995664 * Float64(log2(Float32(65025.0 / mse))))


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
# 3. Multi-Mode BC6H Unsigned Half-Float HDR Encoder & Decoder (`VK_FORMAT_BC6H_UFLOAT`)
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
    return max(0, min(1023, (interp + 32) >> 6))


@always_inline
def _bc6h_unquantize_u6(q: Int) -> Int:
    if q <= 0:
        return 0
    if q >= 63:
        return 0xFFFF
    return ((q << 16) + 0x8000) >> 6


@always_inline
def _bc6h_quantize_uf16_to_u6(f16_bits: Int) -> Int:
    if f16_bits <= 0:
        return 0
    if f16_bits >= 0x7BFF:
        return 63
    var interp = (f16_bits << 6) // 31
    return max(0, min(63, (interp + 512) >> 10))


def _encode_bc6h_tile_at(
    hdr_rgb: List[Vec3],
    width: Int,
    height: Int,
    bx: Int,
    by: Int,
    partitioned: Bool = True,
) -> _BitPacker128:
    var hr = Array[Int, 16](fill=0)
    var hg = Array[Int, 16](fill=0)
    var hb = Array[Int, 16](fill=0)
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

    # Mode 11 (`0x03`): 1-subset 10-bit RGB + 4-bit weights
    var qr0 = _bc6h_quantize_uf16_to_u10(min_r)
    var qg0 = _bc6h_quantize_uf16_to_u10(min_g)
    var qb0 = _bc6h_quantize_uf16_to_u10(min_b)
    var qr1 = _bc6h_quantize_uf16_to_u10(max_r)
    var qg1 = _bc6h_quantize_uf16_to_u10(max_g)
    var qb1 = _bc6h_quantize_uf16_to_u10(max_b)

    var ur0 = _bc6h_unquantize_u10(qr0); var ur1 = _bc6h_unquantize_u10(qr1)
    var ug0 = _bc6h_unquantize_u10(qg0); var ug1 = _bc6h_unquantize_u10(qg1)
    var ub0 = _bc6h_unquantize_u10(qb0); var ub1 = _bc6h_unquantize_u10(qb1)

    var indices11 = Array[Int, 16](fill=0)
    var err11 = 0
    for t in range(16):
        var best_idx = 0
        var best_d = 1_000_000_000
        for wi in range(16):
            var w = _bc7_weight4(wi)
            var ir = ((((64 - w) * ur0 + w * ur1 + 32) >> 6) * 31) >> 6
            var ig = ((((64 - w) * ug0 + w * ug1 + 32) >> 6) * 31) >> 6
            var ib = ((((64 - w) * ub0 + w * ub1 + 32) >> 6) * 31) >> 6
            var d = (ir - hr[t]) * (ir - hr[t]) + (ig - hg[t]) * (ig - hg[t]) + (ib - hb[t]) * (ib - hb[t])
            if d < best_d:
                best_d = d
                best_idx = wi
        indices11[t] = best_idx
        err11 += best_d

    if indices11[0] >= 8:
        var tmp = qr0; qr0 = qr1; qr1 = tmp
        tmp = qg0; qg0 = qg1; qg1 = tmp
        tmp = qb0; qb0 = qb1; qb1 = tmp
        for t in range(16):
            indices11[t] = 15 - indices11[t]

    var bp11 = _BitPacker128()
    bp11.put(0x03, 5)
    bp11.put(qr0, 10); bp11.put(qg0, 10); bp11.put(qb0, 10)
    bp11.put(qr1, 10); bp11.put(qg1, 10); bp11.put(qb1, 10)
    bp11.put(indices11[0], 3)
    for t in range(1, 16):
        bp11.put(indices11[t], 4)

    # Mode 10 (`0x1E`): 2-subset (`0..31` partitions) 6-bit RGB + 3-bit weights for extreme multi-lobe HDR discontinuities
    # (Only switch away from 10-bit Mode 11 when Mode 11 residual is extreme and Mode 10 cuts error by > 4x,
    # avoiding 6-bit mantissa chromatic posterization on smooth sky gradients!)
    if partitioned and err11 > 4_000_000:
        var cand_parts = [0, 1, 2, 13, 14, 15, 16, 17, 19, 26, 29]
        var best_bp = bp11
        var best_err = err11
        for cp_i in range(len(cand_parts)):
            var part = cand_parts[cp_i]
            var mask = _bc7_part2_mask(part)
            var anchor2 = _bc7_anchor2(part)
            var smin_r = Array[Int, 2](fill=0x7BFF); var smax_r = Array[Int, 2](fill=0)
            var smin_g = Array[Int, 2](fill=0x7BFF); var smax_g = Array[Int, 2](fill=0)
            var smin_b = Array[Int, 2](fill=0x7BFF); var smax_b = Array[Int, 2](fill=0)
            for t in range(16):
                var s = (mask >> t) & 1
                if hr[t] < smin_r[s]: smin_r[s] = hr[t]
                if hr[t] > smax_r[s]: smax_r[s] = hr[t]
                if hg[t] < smin_g[s]: smin_g[s] = hg[t]
                if hg[t] > smax_g[s]: smax_g[s] = hg[t]
                if hb[t] < smin_b[s]: smin_b[s] = hb[t]
                if hb[t] > smax_b[s]: smax_b[s] = hb[t]

            var q6r = Array[Int, 4](fill=0)
            var q6g = Array[Int, 4](fill=0)
            var q6b = Array[Int, 4](fill=0)
            var u6r = Array[Int, 4](fill=0)
            var u6g = Array[Int, 4](fill=0)
            var u6b = Array[Int, 4](fill=0)
            for s in range(2):
                q6r[s * 2] = _bc6h_quantize_uf16_to_u6(smin_r[s])
                q6g[s * 2] = _bc6h_quantize_uf16_to_u6(smin_g[s])
                q6b[s * 2] = _bc6h_quantize_uf16_to_u6(smin_b[s])
                q6r[s * 2 + 1] = _bc6h_quantize_uf16_to_u6(smax_r[s])
                q6g[s * 2 + 1] = _bc6h_quantize_uf16_to_u6(smax_g[s])
                q6b[s * 2 + 1] = _bc6h_quantize_uf16_to_u6(smax_b[s])
                for k in range(2):
                    u6r[s * 2 + k] = _bc6h_unquantize_u6(q6r[s * 2 + k])
                    u6g[s * 2 + k] = _bc6h_unquantize_u6(q6g[s * 2 + k])
                    u6b[s * 2 + k] = _bc6h_unquantize_u6(q6b[s * 2 + k])

            var idx10 = Array[Int, 16](fill=0)
            var err10 = 0
            for t in range(16):
                var s = (mask >> t) & 1
                var e0 = s * 2; var e1 = e0 + 1
                var b_wi = 0; var b_d = 1_000_000_000
                for wi in range(8):
                    var w = _bc7_weight3(wi)
                    var ir = ((((64 - w) * u6r[e0] + w * u6r[e1] + 32) >> 6) * 31) >> 6
                    var ig = ((((64 - w) * u6g[e0] + w * u6g[e1] + 32) >> 6) * 31) >> 6
                    var ib = ((((64 - w) * u6b[e0] + w * u6b[e1] + 32) >> 6) * 31) >> 6
                    var d = (ir - hr[t]) * (ir - hr[t]) + (ig - hg[t]) * (ig - hg[t]) + (ib - hb[t]) * (ib - hb[t])
                    if d < b_d:
                        b_d = d
                        b_wi = wi
                idx10[t] = b_wi
                err10 += b_d

            if err10 * 4 < best_err:
                if idx10[0] >= 4:
                    var tmp = q6r[0]; q6r[0] = q6r[1]; q6r[1] = tmp
                    tmp = q6g[0]; q6g[0] = q6g[1]; q6g[1] = tmp
                    tmp = q6b[0]; q6b[0] = q6b[1]; q6b[1] = tmp
                    for t in range(16):
                        if ((mask >> t) & 1) == 0:
                            idx10[t] = 7 - idx10[t]
                if idx10[anchor2] >= 4:
                    var tmp = q6r[2]; q6r[2] = q6r[3]; q6r[3] = tmp
                    tmp = q6g[2]; q6g[2] = q6g[3]; q6g[3] = tmp
                    tmp = q6b[2]; q6b[2] = q6b[3]; q6b[3] = tmp
                    for t in range(16):
                        if ((mask >> t) & 1) == 1:
                            idx10[t] = 7 - idx10[t]

                var bp10 = _BitPacker128()
                bp10.put(0x1E, 5)  # Mode 10 = 0b11110 (5 bits, explicit 6:6:6 2-subset)
                # Official Khronos BPTC_FLOAT / Microsoft BC6H Mode 10 (`11110`) 82-bit header layout:
                # rw=q6r[0], gw=q6g[0], bw=q6b[0], rx=q6r[1], gx=q6g[1], bx=q6b[1],
                # ry=q6r[2], gy=q6g[2], by=q6b[2], rz=q6r[3], gz=q6g[3], bz=q6b[3]
                bp10.put(q6r[0], 6)                 # [10:5]  rw[5:0]
                bp10.put((q6g[3] >> 4) & 1, 1)      # [11]    gz[4]
                bp10.put(q6b[2] & 1, 1)             # [12]    by[0]
                bp10.put((q6b[2] >> 1) & 1, 1)      # [13]    by[1]
                bp10.put((q6g[2] >> 4) & 1, 1)      # [14]    gy[4]
                bp10.put(q6g[0], 6)                 # [20:15] gw[5:0]
                bp10.put((q6g[2] >> 5) & 1, 1)      # [21]    gy[5]
                bp10.put((q6b[2] >> 5) & 1, 1)      # [22]    by[5]
                bp10.put((q6b[3] >> 2) & 1, 1)      # [23]    bz[2]
                bp10.put(q6b[0], 6)                 # [29:24] bw[5:0]
                bp10.put((q6g[3] >> 5) & 1, 1)      # [30]    gz[5]
                bp10.put((q6b[3] >> 3) & 1, 1)      # [31]    bz[3]
                bp10.put((q6b[3] >> 5) & 1, 1)      # [32]    bz[5]
                bp10.put((q6b[3] >> 4) & 1, 1)      # [33]    bz[4]
                bp10.put(q6r[1], 6)                 # [39:34] rx[5:0]
                bp10.put(q6g[3] & 0xF, 4)           # [43:40] gz[3:0]
                bp10.put(q6g[1], 6)                 # [49:44] gx[5:0]
                bp10.put(q6g[2] & 0xF, 4)           # [53:50] gy[3:0]
                bp10.put(q6b[1], 6)                 # [59:54] bx[5:0]
                bp10.put(q6b[3] & 0x3, 2)           # [61:60] bz[1:0]
                bp10.put(q6r[2], 6)                 # [67:62] ry[5:0]
                bp10.put((q6b[2] >> 2) & 0x3, 2)    # [69:68] by[3:2]
                bp10.put(q6r[3], 6)                 # [75:70] rz[5:0]
                bp10.put((q6b[2] >> 4) & 1, 1)      # [76]    by[4]
                bp10.put(part, 5)                   # [81:77] d[4:0]
                for t in range(16):
                    if t == 0 or t == anchor2:
                        bp10.put(idx10[t], 2)
                    else:
                        bp10.put(idx10[t], 3)
                best_err = err10
                best_bp = bp10
        return best_bp

    return bp11


def encode_bc6h_hdr(hdr_rgb: List[Vec3], width: Int, height: Int, partitioned: Bool = True) -> List[UInt8]:
    """Compresses linear HDR `Vec3` radiance (`[0 .. 65504]`) into `VK_FORMAT_BC6H_UFLOAT` (Modes 11 & 10)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt8](length=bw * bh * 16, fill=0)
    if len(hdr_rgb) < width * height or width <= 0 or height <= 0:
        return out^
    var dst_addr = Int(out.unsafe_ptr())
    for by in range(bh):
        for bx in range(bw):
            var bp = _encode_bc6h_tile_at(hdr_rgb, width, height, bx, by, partitioned)
            bp.write_at(dst_addr + (by * bw + bx) * 16)
    return out^


def decode_bc6h_hdr(blocks: List[UInt8], width: Int, height: Int) -> List[Vec3]:
    """Reference Khronos `BC6H_UFLOAT` decoder supporting Mode 11 (`0x03`) and 2-Subset Mode 10 (`0x1E`)."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[Vec3](length=width * height, fill=Vec3(0.0, 0.0, 0.0))

    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            var br = _BitReader128(blocks, off)
            var mode = br.get(5)
            if mode == 0x1E:
                var rw = br.get(6)
                var gz4 = br.get(1)
                var by0 = br.get(1)
                var by1 = br.get(1)
                var gy4 = br.get(1)
                var gw = br.get(6)
                var gy5 = br.get(1)
                var by5 = br.get(1)
                var bz2 = br.get(1)
                var bw_c = br.get(6)
                var gz5 = br.get(1)
                var bz3 = br.get(1)
                var bz5 = br.get(1)
                var bz4 = br.get(1)
                var rx = br.get(6)
                var gz30 = br.get(4)
                var gx = br.get(6)
                var gy30 = br.get(4)
                var bx_c = br.get(6)
                var bz10 = br.get(2)
                var ry = br.get(6)
                var by32 = br.get(2)
                var rz = br.get(6)
                var by4 = br.get(1)
                var part = br.get(5)

                var gy = gy30 | (gy4 << 4) | (gy5 << 5)
                var gz = gz30 | (gz4 << 4) | (gz5 << 5)
                var by_c = by0 | (by1 << 1) | (by32 << 2) | (by4 << 4) | (by5 << 5)
                var bz = bz10 | (bz2 << 2) | (bz3 << 3) | (bz4 << 4) | (bz5 << 5)

                var u6r = Array[Int, 4](fill=0)
                var u6g = Array[Int, 4](fill=0)
                var u6b = Array[Int, 4](fill=0)
                u6r[0] = _bc6h_unquantize_u6(rw); u6g[0] = _bc6h_unquantize_u6(gw); u6b[0] = _bc6h_unquantize_u6(bw_c)
                u6r[1] = _bc6h_unquantize_u6(rx); u6g[1] = _bc6h_unquantize_u6(gx); u6b[1] = _bc6h_unquantize_u6(bx_c)
                u6r[2] = _bc6h_unquantize_u6(ry); u6g[2] = _bc6h_unquantize_u6(gy); u6b[2] = _bc6h_unquantize_u6(by_c)
                u6r[3] = _bc6h_unquantize_u6(rz); u6g[3] = _bc6h_unquantize_u6(gz); u6b[3] = _bc6h_unquantize_u6(bz)
                var mask = _bc7_part2_mask(part)
                var anchor2 = _bc7_anchor2(part)
                for t in range(16):
                    var idx = br.get(2) if (t == 0 or t == anchor2) else br.get(3)
                    var w = _bc7_weight3(idx)
                    var sub = (mask >> t) & 1
                    var e0 = sub * 2; var e1 = e0 + 1
                    var hr = ((((64 - w) * u6r[e0] + w * u6r[e1] + 32) >> 6) * 31) >> 6
                    var hg = ((((64 - w) * u6g[e0] + w * u6g[e1] + 32) >> 6) * 31) >> 6
                    var hb = ((((64 - w) * u6b[e0] + w * u6b[e1] + 32) >> 6) * 31) >> 6
                    var py = by * 4 + (t >> 2)
                    var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = Vec3(uf16_bits_to_f32(hr), uf16_bits_to_f32(hg), uf16_bits_to_f32(hb))
            else:
                var qr0 = br.get(10); var qg0 = br.get(10); var qb0 = br.get(10)
                var qr1 = br.get(10); var qg1 = br.get(10); var qb1 = br.get(10)
                var ur0 = _bc6h_unquantize_u10(qr0); var ur1 = _bc6h_unquantize_u10(qr1)
                var ug0 = _bc6h_unquantize_u10(qg0); var ug1 = _bc6h_unquantize_u10(qg1)
                var ub0 = _bc6h_unquantize_u10(qb0); var ub1 = _bc6h_unquantize_u10(qb1)
                for t in range(16):
                    var idx = br.get(3) if t == 0 else br.get(4)
                    var w = _bc7_weight4(idx)
                    var hr = ((((64 - w) * ur0 + w * ur1 + 32) >> 6) * 31) >> 6
                    var hg = ((((64 - w) * ug0 + w * ug1 + 32) >> 6) * 31) >> 6
                    var hb = ((((64 - w) * ub0 + w * ub1 + 32) >> 6) * 31) >> 6
                    var py = by * 4 + (t >> 2)
                    var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = Vec3(uf16_bits_to_f32(hr), uf16_bits_to_f32(hg), uf16_bits_to_f32(hb))

    return out^


# ============================================================================
# 4. Pre-Cooked `.flpk` Mip-Chunk Packager & Mipmap Downsamplers
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


def downsample_normals_and_orm_toksvig_2x(
    normals: List[Vec3],
    orm_argb: List[UInt32],
    w: Int,
    h: Int,
    toksvig_scale: Float32 = 1.0,
) -> Tuple[List[Vec3], List[UInt32], Int, Int]:
    """Coupled 2x normal + ORM mipmap downsampler implementing Toksvig normal-variance specular anti-aliasing.

    Averages unnormalized 2x2 unit normals `bar_n`, computes resultant length `r = clamp(||bar_n||, 0.05, 1.0)`,
    normalizes `bar_n / r` for `BC5`, and filters the `G` (roughness) byte of `orm_argb` (`0xAARRGGBB`) via:
      `sigma_n^2 = ((1 - r) / r) * toksvig_scale`
      `alpha'    = sqrt(min(1.0, alpha_0^2 + 2.0 * sigma_n^2))`   (`alpha_0 = rough_0^2`)
      `rough'    = sqrt(alpha')`
    """
    var nw = max(1, w >> 1)
    var nh = max(1, h >> 1)
    var out_normals = List[Vec3](length=nw * nh, fill=Vec3(0.0, 0.0, 1.0))
    var out_orm = List[UInt32](length=nw * nh, fill=UInt32(0xFFFF8000))
    var has_nrm = len(normals) >= w * h and w > 0 and h > 0
    var has_orm = len(orm_argb) >= w * h and w > 0 and h > 0
    var scale = max(Float32(0.0), toksvig_scale)

    for y in range(nh):
        var y0 = min(max(0, h - 1), y * 2)
        var y1 = min(max(0, h - 1), y * 2 + 1)
        for x in range(nw):
            var x0 = min(max(0, w - 1), x * 2)
            var x1 = min(max(0, w - 1), x * 2 + 1)

            var r_len = Float32(1.0)
            var unit_n = Vec3(0.0, 0.0, 1.0)
            if has_nrm:
                var n00 = normals[y0 * w + x0].normalized()
                var n10 = normals[y0 * w + x1].normalized()
                var n01 = normals[y1 * w + x0].normalized()
                var n11 = normals[y1 * w + x1].normalized()
                var bar_n = (n00 + n10 + n01 + n11) * Float32(0.25)
                var raw_len = bar_n.length()
                r_len = max(Float32(0.05), min(Float32(1.0), raw_len))
                if raw_len > Float32(1e-6):
                    unit_n = bar_n * (Float32(1.0) / raw_len)
                    if unit_n.z < Float32(0.02):
                        unit_n = Vec3(unit_n.x, unit_n.y, Float32(0.02)).normalized()
            out_normals[y * nw + x] = unit_n

            var a_avg = 255
            var ao_avg = 255
            var g_sum = 512  # default 128 * 4
            var m_avg = 0
            if has_orm:
                var p00 = orm_argb[y0 * w + x0]
                var p10 = orm_argb[y0 * w + x1]
                var p01 = orm_argb[y1 * w + x0]
                var p11 = orm_argb[y1 * w + x1]
                a_avg = (Int((p00 >> 24) & 255) + Int((p10 >> 24) & 255) + Int((p01 >> 24) & 255) + Int((p11 >> 24) & 255) + 2) >> 2
                ao_avg = (Int((p00 >> 16) & 255) + Int((p10 >> 16) & 255) + Int((p01 >> 16) & 255) + Int((p11 >> 16) & 255) + 2) >> 2
                g_sum = Int((p00 >> 8) & 255) + Int((p10 >> 8) & 255) + Int((p01 >> 8) & 255) + Int((p11 >> 8) & 255)
                m_avg = (Int(p00 & 255) + Int(p10 & 255) + Int(p01 & 255) + Int(p11 & 255) + 2) >> 2

            var rough0 = Float32(g_sum) * (Float32(0.25) / Float32(255.0))
            var alpha0 = rough0 * rough0
            var sigma_n2 = ((Float32(1.0) - r_len) / r_len) * scale
            var alpha_p = sqrt(min(Float32(1.0), alpha0 * alpha0 + Float32(2.0) * sigma_n2))
            var rough_p = sqrt(max(Float32(0.0), alpha_p))
            var rough_u8 = max(0, min(255, Int(rough_p * Float32(255.0) + Float32(0.5))))

            out_orm[y * nw + x] = (
                (UInt32(a_avg) << 24)
                | (UInt32(ao_avg) << 16)
                | (UInt32(rough_u8) << 8)
                | UInt32(m_avg)
            )

    return (out_normals^, out_orm^, nw, nh)


def bake_toksvig_normal_orm_pyramid(
    normals: List[Vec3],
    orm_argb: List[UInt32],
    w: Int,
    h: Int,
    toksvig_scale: Float32 = 1.0,
    min_dim: Int = 1,
) -> Tuple[List[List[Vec3]], List[List[UInt32]], List[Int], List[Int]]:
    """Builds coupled multi-mip `BC5` tangent-normal + `BC7` ORM pyramids with Toksvig roughness filtering."""
    var mip_normals = List[List[Vec3]]()
    var mip_orms = List[List[UInt32]]()
    var mip_w = List[Int]()
    var mip_h = List[Int]()
    if w <= 0 or h <= 0:
        return (mip_normals^, mip_orms^, mip_w^, mip_h^)

    var cur_n = normals.copy()
    var cur_o = orm_argb.copy()
    var cw = w
    var ch = h
    var stop_dim = max(1, min_dim)

    while True:
        mip_normals.append(cur_n.copy())
        mip_orms.append(cur_o.copy())
        mip_w.append(cw)
        mip_h.append(ch)
        if cw <= stop_dim and ch <= stop_dim:
            break
        var ds = downsample_normals_and_orm_toksvig_2x(cur_n, cur_o, cw, ch, toksvig_scale)
        cur_n = ds[0].copy()
        cur_o = ds[1].copy()
        cw = ds[2]
        ch = ds[3]

    return (mip_normals^, mip_orms^, mip_w^, mip_h^)


def write_cooked_bc7_flpk(
    pack_file: String,
    asset_key: String,
    pixels_argb: List[UInt32],
    width: Int,
    height: Int,
    mut pool: JobPool,
) -> Int:
    """Cooks all `BC7` mip levels in parallel via `SIMD` + `JobPool` and packs each mip as a streamable `.flpk` chunk."""
    var names = List[String]()
    var chunks = List[List[UInt8]]()
    var cur_px = pixels_argb.copy()
    var cw = width
    var ch = height
    var mip = 0

    while True:
        var bc7 = encode_bc7_rgba_simd(cur_px, cw, ch, pool, multi_mode=True)
        names.append(asset_key + ".mip" + String(mip) + ".bc7")
        chunks.append(bc7^)
        mip += 1
        if cw <= 4 and ch <= 4:
            break
        var ds = downsample_argb_2x(cur_px, cw, ch)
        cur_px = ds[0].copy()
        cw = ds[1]
        ch = ds[2]

    var archive = encode_pack(names, chunks)
    if not write_bytes(pack_file, archive):
        return 0
    return mip


def stream_bc7_mip_from_flpk(pack_file: String, asset_key: String, mip_level: Int) -> List[UInt8]:
    """Streams a single pre-cooked `BC7` mip chunk directly from `.flpk` with zero runtime re-compression."""
    return pack_read(pack_file, asset_key + ".mip" + String(mip_level) + ".bc7")
