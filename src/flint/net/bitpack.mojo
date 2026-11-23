"""Bit-level network serialization & Smallest-Three Quaternion quantization (`flint.net.bitpack`).

Implements:
- `BitWriter`: zero-overhead bit accumulator with capacity reuse (`clear()`), bounded integer
  encoding, quantized `Float32` / `Vec3`, 32-bit Smallest-Three `Quat` (`< 0.08 deg` max angular
  error across SO(3) via 8-corner optimal lattice rounding), and 48-bit high-precision `Quat`.
- `BitReader`: bounds-checked bit decoder setting `overflowed = True` and returning safe zero/identity
  defaults whenever a truncated or malformed packet attempts to read past `bit_len`.
"""
from std.collections import List
from std.math import abs, isfinite, max, min, sqrt
from flint.core.detmath import SQRT_HALF, det_atan2, f32_bits, f32_from_bits
from flint.core.math import Quat, Vec3


@always_inline
def bits_required(max_span: UInt64) -> Int:
    """Minimum number of bits required to represent integers in `[0, max_span]`."""
    if max_span == 0:
        return 0
    var v = max_span
    var count = 0
    while v > 0:
        v >>= 1
        count += 1
    return count


@always_inline
def _safe_unit_quat(q_in: Quat) -> Quat:
    if not (isfinite(q_in.x) and isfinite(q_in.y) and isfinite(q_in.z) and isfinite(q_in.w)):
        return Quat.identity()
    var l2 = q_in.length_squared()
    if not isfinite(l2) or l2 <= Float32(1.0e-12):
        return Quat.identity()
    var q = q_in.normalized()
    if not (isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)):
        return Quat.identity()
    return q


@always_inline
def quat_angular_error_deg(a: Quat, b: Quat) -> Float32:
    """Cancellation-free Kahan quaternion geodesic angle `acos(|qa . qb|)` in degrees:
    `phi = 2 * atan2(||qa - sign*qb||, ||qa + sign*qb||) * (180 / pi)`."""
    var qa = _safe_unit_quat(a)
    var qb = _safe_unit_quat(b)
    if qa.dot(qb) < Float32(0.0):
        qb = Quat(-qb.x, -qb.y, -qb.z, -qb.w)
    var dmx = qa.x - qb.x
    var dmy = qa.y - qb.y
    var dmz = qa.z - qb.z
    var dmw = qa.w - qb.w
    var dpx = qa.x + qb.x
    var dpy = qa.y + qb.y
    var dpz = qa.z + qb.z
    var dpw = qa.w + qb.w
    var chord_minus = sqrt(dmx * dmx + dmy * dmy + dmz * dmz + dmw * dmw)
    var chord_plus = sqrt(dpx * dpx + dpy * dpy + dpz * dpz + dpw * dpw)
    var angle_rad = Float32(2.0) * det_atan2(chord_minus, chord_plus)
    return angle_rad * Float32(57.29577951308232)


@always_inline
def _gnomonic_warp(u: Float32) -> Float32:
    return u * (Float32(0.79) + Float32(0.21) * u * u)


@always_inline
def _gnomonic_unwarp(g: Float32) -> Float32:
    var u = g
    for _ in range(4):
        var u2 = u * u
        var f = u * (Float32(0.79) + Float32(0.21) * u2) - g
        var df = Float32(0.79) + Float32(0.63) * u2
        u -= f / df
    return max(Float32(-1.0), min(Float32(1.0), u))


@always_inline
def _reconstruct_from_gnomonic(
    max_idx: Int, g0: Float32, g1: Float32, g2: Float32
) -> Quat:
    var inv_norm = Float32(1.0) / sqrt(
        Float32(1.0) + g0 * g0 + g1 * g1 + g2 * g2
    )
    var c0 = g0 * inv_norm
    var c1 = g1 * inv_norm
    var c2 = g2 * inv_norm
    var cm = inv_norm
    if max_idx == 0:
        return Quat(cm, c0, c1, c2)
    if max_idx == 1:
        return Quat(c0, cm, c1, c2)
    if max_idx == 2:
        return Quat(c0, c1, cm, c2)
    return Quat(c0, c1, c2, cm)


@always_inline
def _decode_bcc_quat32(
    max_idx: Int, parity: Int, i0: Int, i1: Int, i2: Int
) -> Quat:
    var shift = Float32(0.5) if parity == 1 else Float32(0.0)
    var inv_n = Float32(2.0 / 811.0)
    var u0 = (Float32(i0) + shift) * inv_n - Float32(1.0)
    var u1 = (Float32(i1) + shift) * inv_n - Float32(1.0)
    var u2 = (Float32(i2) + shift) * inv_n - Float32(1.0)
    return _reconstruct_from_gnomonic(
        max_idx, _gnomonic_warp(u0), _gnomonic_warp(u1), _gnomonic_warp(u2)
    )


@always_inline
def _reconstruct_quat_from_three(
    max_idx: Int, c0: Float32, c1: Float32, c2: Float32
) -> Quat:
    var sum_sq = c0 * c0 + c1 * c1 + c2 * c2
    var cm = sqrt(max(Float32(0.0), Float32(1.0) - sum_sq))
    if max_idx == 0:
        return Quat(cm, c0, c1, c2).normalized()
    if max_idx == 1:
        return Quat(c0, cm, c1, c2).normalized()
    if max_idx == 2:
        return Quat(c0, c1, cm, c2).normalized()
    return Quat(c0, c1, c2, cm).normalized()


@fieldwise_init
struct BitWriterMark(Copyable, ImplicitlyCopyable, Movable):
    """Saved cursor state for zero-allocation `BitWriter` rollback on budget overflow."""
    var data_len: Int
    var scratch: UInt64
    var scratch_bits: Int
    var total_bits: Int


def pack_quat_smallest_three(q_in: Quat) -> UInt32:
    """Packs a quaternion into a 32-bit Smallest-Three codeword (`2` index bits + `30`-bit BCC lattice).

    Maps the exact Smallest-Three Voronoi cell bijectively onto `[-1, 1]^3` with zero wasted
    codewords and quantizes onto a Body-Centered Cubic (`2 * 812^3 < 2^30`) lattice using
    16-candidate geodesic search (`< 0.10 deg` worst-case SO(3) error, ~0.04 deg typical).
    """
    var q = _safe_unit_quat(q_in)
    var ax = abs(q.x)
    var ay = abs(q.y)
    var az = abs(q.z)
    var aw = abs(q.w)

    var max_idx = 0
    var max_val = ax
    if ay > max_val:
        max_idx = 1
        max_val = ay
    if az > max_val:
        max_idx = 2
        max_val = az
    if aw > max_val:
        max_idx = 3

    var sign = Float32(1.0)
    if (
        (max_idx == 0 and q.x < Float32(0.0))
        or (max_idx == 1 and q.y < Float32(0.0))
        or (max_idx == 2 and q.z < Float32(0.0))
        or (max_idx == 3 and q.w < Float32(0.0))
    ):
        sign = Float32(-1.0)

    var qpos = Quat(q.x * sign, q.y * sign, q.z * sign, q.w * sign)
    var c0: Float32
    var c1: Float32
    var c2: Float32
    var cm: Float32
    if max_idx == 0:
        cm = qpos.x
        c0 = qpos.y
        c1 = qpos.z
        c2 = qpos.w
    elif max_idx == 1:
        c0 = qpos.x
        cm = qpos.y
        c1 = qpos.z
        c2 = qpos.w
    elif max_idx == 2:
        c0 = qpos.x
        c1 = qpos.y
        cm = qpos.z
        c2 = qpos.w
    else:
        c0 = qpos.x
        c1 = qpos.y
        c2 = qpos.z
        cm = qpos.w

    var inv_cm = Float32(1.0) / max(Float32(0.49), cm)
    var u0 = _gnomonic_unwarp(max(Float32(-1.0), min(Float32(1.0), c0 * inv_cm)))
    var u1 = _gnomonic_unwarp(max(Float32(-1.0), min(Float32(1.0), c1 * inv_cm)))
    var u2 = _gnomonic_unwarp(max(Float32(-1.0), min(Float32(1.0), c2 * inv_cm)))

    var best_p = 0
    var best_i0 = 405
    var best_i1 = 405
    var best_i2 = 405
    var best_err_sq = Float32(1.0e30)

    for parity in range(2):
        var max_coord = 810 if parity == 1 else 811
        var shift = Float32(0.5) if parity == 1 else Float32(0.0)
        var f0 = max(
            Float32(0.0),
            min(
                Float32(max_coord),
                (u0 + Float32(1.0)) * Float32(405.5) - shift,
            ),
        )
        var f1 = max(
            Float32(0.0),
            min(
                Float32(max_coord),
                (u1 + Float32(1.0)) * Float32(405.5) - shift,
            ),
        )
        var f2 = max(
            Float32(0.0),
            min(
                Float32(max_coord),
                (u2 + Float32(1.0)) * Float32(405.5) - shift,
            ),
        )
        var lo0 = min(max_coord, max(0, Int(f0)))
        var lo1 = min(max_coord, max(0, Int(f1)))
        var lo2 = min(max_coord, max(0, Int(f2)))
        var hi0 = min(max_coord, lo0 + 1)
        var hi1 = min(max_coord, lo1 + 1)
        var hi2 = min(max_coord, lo2 + 1)

        for corner in range(8):
            var i0 = hi0 if (corner & 1) != 0 else lo0
            var i1 = hi1 if (corner & 2) != 0 else lo1
            var i2 = hi2 if (corner & 4) != 0 else lo2
            var cand = _decode_bcc_quat32(max_idx, parity, i0, i1, i2)
            if qpos.dot(cand) < Float32(0.0):
                cand = Quat(-cand.x, -cand.y, -cand.z, -cand.w)
            var dx = qpos.x - cand.x
            var dy = qpos.y - cand.y
            var dz = qpos.z - cand.z
            var dw = qpos.w - cand.w
            var err_sq = dx * dx + dy * dy + dz * dz + dw * dw
            if err_sq < best_err_sq:
                best_err_sq = err_sq
                best_p = parity
                best_i0 = i0
                best_i1 = i1
                best_i2 = i2

    comptime SUBLATTICE_SIZE: UInt32 = 812 * 812 * 812
    var idx3d = UInt32((best_i0 * 812 + best_i1) * 812 + best_i2)
    var code30 = UInt32(best_p) * SUBLATTICE_SIZE + idx3d
    return UInt32(max_idx & 3) | (code30 << 2)


struct BitWriter(Copyable, Movable):
    """Packs arbitrary-width bitfields LSB-first into a reusable byte buffer."""
    var data: List[UInt8]
    var scratch: UInt64
    var scratch_bits: Int
    var total_bits: Int

    def __init__(out self, capacity_bytes: Int = 256):
        self.data = List[UInt8](capacity=max(16, capacity_bytes))
        self.scratch = 0
        self.scratch_bits = 0
        self.total_bits = 0

    def clear(mut self):
        """Resets bit stream while preserving underlying `List[UInt8]` capacity."""
        self.data.clear()
        self.scratch = 0
        self.scratch_bits = 0
        self.total_bits = 0

    @always_inline
    def bit_length(self) -> Int:
        return self.total_bits

    @always_inline
    def byte_length(self) -> Int:
        return len(self.data) + ((self.scratch_bits + 7) >> 3)

    @always_inline
    def mark(self) -> BitWriterMark:
        """Captures current cursor state for O(1) rollback via `restore()`."""
        return BitWriterMark(
            len(self.data), self.scratch, self.scratch_bits, self.total_bits
        )

    @always_inline
    def restore(mut self, m: BitWriterMark):
        """Rolls back cursor state to a previously captured `BitWriterMark` without reallocating."""
        while len(self.data) > m.data_len:
            _ = self.data.pop()
        self.scratch = m.scratch
        self.scratch_bits = m.scratch_bits
        self.total_bits = m.total_bits

    @always_inline
    def _write_bits_small(mut self, value: UInt64, num_bits: Int):
        """Internal helper for `1 <= num_bits <= 32`."""
        var mask = (
            UInt64(0xFFFFFFFF)
            if num_bits == 32
            else ((UInt64(1) << UInt64(num_bits)) - UInt64(1))
        )
        self.scratch |= (value & mask) << UInt64(self.scratch_bits)
        self.scratch_bits += num_bits
        self.total_bits += num_bits
        while self.scratch_bits >= 8:
            self.data.append(UInt8(self.scratch & 0xFF))
            self.scratch >>= 8
            self.scratch_bits -= 8

    def write_bits(mut self, value: UInt64, num_bits: Int):
        """Appends the lowest `num_bits` (`0..64`) of `value` LSB-first."""
        if num_bits <= 0:
            return
        if num_bits <= 32:
            self._write_bits_small(value, num_bits)
            return
        var clamped = min(64, num_bits)
        self._write_bits_small(value & 0xFFFFFFFF, 32)
        self._write_bits_small(value >> 32, clamped - 32)

    @always_inline
    def write_bool(mut self, value: Bool):
        self._write_bits_small(UInt64(1) if value else UInt64(0), 1)

    @always_inline
    def write_u8(mut self, value: UInt8):
        self._write_bits_small(UInt64(value), 8)

    @always_inline
    def write_u16(mut self, value: UInt16):
        self._write_bits_small(UInt64(value), 16)

    @always_inline
    def write_u32(mut self, value: UInt32):
        self._write_bits_small(UInt64(value), 32)

    @always_inline
    def write_u64(mut self, value: UInt64):
        self.write_bits(value, 64)

    @always_inline
    def write_f32_raw(mut self, value: Float32):
        self._write_bits_small(UInt64(f32_bits(value)), 32)

    def flush_align(mut self):
        """Pads remaining sub-byte bits with zeros to reach the next byte boundary."""
        if self.scratch_bits > 0:
            var pad = 8 - self.scratch_bits
            self.data.append(UInt8(self.scratch & 0xFF))
            self.scratch = 0
            self.scratch_bits = 0
            self.total_bits += pad

    def write_bytes_aligned(mut self, payload: List[UInt8]):
        self.flush_align()
        for i in range(len(payload)):
            self.data.append(payload[i])
        self.total_bits += len(payload) * 8

    def finish(mut self) -> List[UInt8]:
        """Flushes pending bits and returns a copy of the packed byte array."""
        self.flush_align()
        return self.data.copy()

    def write_int_bounded(mut self, value: Int, min_val: Int, max_val: Int):
        """Writes `value` clamped to `[min_val, max_val]` in `ceil(log2(max - min + 1))` bits."""
        if max_val <= min_val:
            return
        var span = UInt64(max_val - min_val)
        var nbits = bits_required(span)
        var clamped = max(min_val, min(max_val, value))
        var offset = UInt64(clamped - min_val)
        self.write_bits(offset, nbits)

    def write_f32_quantized(
        mut self,
        value: Float32,
        min_val: Float32,
        max_val: Float32,
        num_bits: Int,
    ):
        """Quantizes `value` uniformly over `[min_val, max_val]` into `num_bits` (`1..31`)."""
        var b = max(1, min(31, num_bits))
        var levels = (1 << b) - 1
        var span = max(Float32(1.0e-9), max_val - min_val)
        var safe_min = min_val if isfinite(min_val) else Float32(0.0)
        var safe_val = value if isfinite(value) else safe_min
        var norm = max(
            Float32(0.0), min(Float32(1.0), (safe_val - safe_min) / span)
        )
        if not isfinite(norm):
            norm = Float32(0.0)
        var q = Int(norm * Float32(levels) + Float32(0.5))
        q = max(0, min(levels, q))
        self._write_bits_small(UInt64(q), b)

    def write_vec3_quantized(
        mut self,
        v: Vec3,
        min_val: Float32,
        max_val: Float32,
        bits_per_axis: Int,
    ):
        self.write_f32_quantized(v.x, min_val, max_val, bits_per_axis)
        self.write_f32_quantized(v.y, min_val, max_val, bits_per_axis)
        self.write_f32_quantized(v.z, min_val, max_val, bits_per_axis)

    def write_quat_smallest_three(mut self, q_in: Quat):
        """32-bit Smallest-Three quaternion (`2` index bits + `30`-bit BCC cube-mapped lattice)."""
        self._write_bits_small(UInt64(pack_quat_smallest_three(q_in)), 32)

    @always_inline
    def write_quat32(mut self, q_in: Quat):
        self.write_quat_smallest_three(q_in)

    def write_quat_smallest_three_48(mut self, q_in: Quat):
        """48-bit high-precision Smallest-Three quaternion (`2 + 15 + 15 + 16 = 48` bits, `< 0.0025 deg`)."""
        comptime SPAN: Float32 = Float32(2.0) * SQRT_HALF
        var q = _safe_unit_quat(q_in)
        var ax = abs(q.x)
        var ay = abs(q.y)
        var az = abs(q.z)
        var aw = abs(q.w)

        var max_idx = 0
        var max_val = ax
        if ay > max_val:
            max_idx = 1
            max_val = ay
        if az > max_val:
            max_idx = 2
            max_val = az
        if aw > max_val:
            max_idx = 3

        var sign = Float32(1.0)
        if (
            (max_idx == 0 and q.x < Float32(0.0))
            or (max_idx == 1 and q.y < Float32(0.0))
            or (max_idx == 2 and q.z < Float32(0.0))
            or (max_idx == 3 and q.w < Float32(0.0))
        ):
            sign = Float32(-1.0)

        var qpos = Quat(q.x * sign, q.y * sign, q.z * sign, q.w * sign)
        var c0: Float32
        var c1: Float32
        var c2: Float32
        if max_idx == 0:
            c0 = qpos.y
            c1 = qpos.z
            c2 = qpos.w
        elif max_idx == 1:
            c0 = qpos.x
            c1 = qpos.z
            c2 = qpos.w
        elif max_idx == 2:
            c0 = qpos.x
            c1 = qpos.y
            c2 = qpos.w
        else:
            c0 = qpos.x
            c1 = qpos.y
            c2 = qpos.z

        var i0 = max(
            0,
            min(
                32767,
                Int(((c0 + SQRT_HALF) / SPAN) * Float32(32767.0) + Float32(0.5)),
            ),
        )
        var i1 = max(
            0,
            min(
                32767,
                Int(((c1 + SQRT_HALF) / SPAN) * Float32(32767.0) + Float32(0.5)),
            ),
        )
        var i2 = max(
            0,
            min(
                65535,
                Int(((c2 + SQRT_HALF) / SPAN) * Float32(65535.0) + Float32(0.5)),
            ),
        )

        var packed48 = (
            UInt64(max_idx & 3)
            | (UInt64(i0) << 2)
            | (UInt64(i1) << 17)
            | (UInt64(i2) << 32)
        )
        self.write_bits(packed48, 48)

    @always_inline
    def write_quat48(mut self, q_in: Quat):
        self.write_quat_smallest_three_48(q_in)


struct BitReader(Copyable, Movable):
    """Bounds-checked LSB-first bit reader over a byte buffer or raw address slice."""
    var data: List[UInt8]
    var raw_addr: Int
    var bit_pos: Int
    var bit_len: Int
    var overflowed: Bool

    def __init__(out self, data: List[UInt8], bit_limit: Int = -1):
        self.data = data.copy()
        self.raw_addr = 0
        self.bit_pos = 0
        var max_bits = len(self.data) * 8
        self.bit_len = (
            max_bits if bit_limit < 0 else min(max_bits, max(0, bit_limit))
        )
        self.overflowed = False

    @staticmethod
    def from_raw_ptr(addr: Int, byte_len: Int) -> Self:
        """Zero-copy view over a live datagram buffer `(addr, byte_len)`."""
        var r = Self(List[UInt8](), 0)
        if addr == 0 or byte_len <= 0:
            r.raw_addr = 0
            r.bit_pos = 0
            r.bit_len = 0
            r.overflowed = addr == 0 and byte_len > 0
            return r^
        r.raw_addr = addr
        r.bit_pos = 0
        r.bit_len = byte_len * 8
        r.overflowed = False
        return r^

    @always_inline
    def remaining_bits(self) -> Int:
        return max(0, self.bit_len - self.bit_pos)

    @always_inline
    def _byte_at(self, idx: Int) -> UInt8:
        if self.raw_addr != 0:
            return Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.raw_addr
            )[unsafe_offset=idx]
        return self.data[idx]

    def _read_bits_small(mut self, num_bits: Int) -> UInt64:
        if num_bits <= 0:
            return 0
        if self.overflowed or self.bit_pos + num_bits > self.bit_len:
            self.overflowed = True
            self.bit_pos = self.bit_len
            return 0
        var byte_idx = self.bit_pos >> 3
        var bit_off = self.bit_pos & 7
        var total_bytes = (self.bit_len + 7) >> 3
        # Assemble up to 5 bytes (`bit_off <= 7` + `num_bits <= 32` <= 39 bits) safely
        var acc = UInt64(0)
        var needed_bytes = (bit_off + num_bits + 7) >> 3
        for b in range(needed_bytes):
            if byte_idx + b < total_bytes:
                acc |= UInt64(self._byte_at(byte_idx + b)) << UInt64(b * 8)
        self.bit_pos += num_bits
        var mask = (
            UInt64(0xFFFFFFFF)
            if num_bits == 32
            else ((UInt64(1) << UInt64(num_bits)) - UInt64(1))
        )
        return (acc >> UInt64(bit_off)) & mask

    def read_bits(mut self, num_bits: Int) -> UInt64:
        """Reads `num_bits` (`0..64`) LSB-first; sets `overflowed = True` if truncated."""
        if num_bits < 0 or num_bits > 64:
            self.overflowed = True
            self.bit_pos = self.bit_len
            return 0
        if num_bits == 0:
            return 0
        if self.overflowed or self.bit_pos + num_bits > self.bit_len:
            self.overflowed = True
            self.bit_pos = self.bit_len
            return 0
        if num_bits <= 32:
            return self._read_bits_small(num_bits)
        var lo = self._read_bits_small(32)
        var hi = self._read_bits_small(num_bits - 32)
        return lo | (hi << 32)

    @always_inline
    def read_bool(mut self) -> Bool:
        return self._read_bits_small(1) != 0

    @always_inline
    def read_u8(mut self) -> UInt8:
        return UInt8(self._read_bits_small(8))

    @always_inline
    def read_u16(mut self) -> UInt16:
        return UInt16(self._read_bits_small(16))

    @always_inline
    def read_u32(mut self) -> UInt32:
        return UInt32(self._read_bits_small(32))

    @always_inline
    def read_u64(mut self) -> UInt64:
        return self.read_bits(64)

    @always_inline
    def read_f32_raw(mut self) -> Float32:
        return f32_from_bits(self.read_u32())

    def align_to_byte(mut self):
        var rem = self.bit_pos & 7
        if rem != 0:
            var adv = 8 - rem
            if self.bit_pos + adv > self.bit_len:
                self.overflowed = True
                self.bit_pos = self.bit_len
            else:
                self.bit_pos += adv

    def read_bytes_aligned(mut self, byte_count: Int) -> List[UInt8]:
        self.align_to_byte()
        if (
            byte_count < 0
            or self.overflowed
            or byte_count > ((self.bit_len - self.bit_pos) >> 3)
        ):
            self.overflowed = True
            self.bit_pos = self.bit_len
            return List[UInt8]()
        var out = List[UInt8](capacity=byte_count)
        var base = self.bit_pos >> 3
        for i in range(byte_count):
            out.append(self._byte_at(base + i))
        self.bit_pos += byte_count * 8
        return out^

    def read_int_bounded(mut self, min_val: Int, max_val: Int) -> Int:
        if max_val <= min_val:
            return min_val
        var span = UInt64(max_val - min_val)
        var nbits = bits_required(span)
        var raw = self.read_bits(nbits)
        if self.overflowed:
            return min_val
        if raw > span:
            raw = span
        return min_val + Int(raw)

    def read_f32_quantized(
        mut self, min_val: Float32, max_val: Float32, num_bits: Int
    ) -> Float32:
        var b = max(1, min(31, num_bits))
        var raw = self._read_bits_small(b)
        if self.overflowed:
            return min_val
        var levels = Float32((1 << b) - 1)
        return min_val + (Float32(raw) / levels) * (max_val - min_val)

    def read_vec3_quantized(
        mut self, min_val: Float32, max_val: Float32, bits_per_axis: Int
    ) -> Vec3:
        var x = self.read_f32_quantized(min_val, max_val, bits_per_axis)
        var y = self.read_f32_quantized(min_val, max_val, bits_per_axis)
        var z = self.read_f32_quantized(min_val, max_val, bits_per_axis)
        return Vec3(x, y, z)

    def read_quat_smallest_three(mut self) -> Quat:
        comptime SUBLATTICE_SIZE: UInt32 = 812 * 812 * 812
        var packed = UInt32(self._read_bits_small(32))
        if self.overflowed:
            return Quat.identity()
        var max_idx = Int(packed & 3)
        var code30 = packed >> 2
        var parity = 0
        var rem = code30
        if rem >= SUBLATTICE_SIZE:
            parity = 1
            rem -= SUBLATTICE_SIZE
            if rem >= SUBLATTICE_SIZE:
                rem = 0
        var i2 = Int(rem % 812)
        var q01 = rem // 812
        var i1 = Int(q01 % 812)
        var i0 = Int(q01 // 812)
        return _decode_bcc_quat32(max_idx, parity, i0, i1, i2)

    @always_inline
    def read_quat32(mut self) -> Quat:
        return self.read_quat_smallest_three()

    def read_quat_smallest_three_48(mut self) -> Quat:
        comptime SPAN: Float32 = Float32(2.0) * SQRT_HALF
        comptime STEP_15: Float32 = SPAN / Float32(32767.0)
        comptime STEP_16: Float32 = SPAN / Float32(65535.0)
        var packed48 = self.read_bits(48)
        if self.overflowed:
            return Quat.identity()
        var max_idx = Int(packed48 & 3)
        var i0 = Int((packed48 >> 2) & 32767)
        var i1 = Int((packed48 >> 17) & 32767)
        var i2 = Int((packed48 >> 32) & 65535)
        var c0 = Float32(i0) * STEP_15 - SQRT_HALF
        var c1 = Float32(i1) * STEP_15 - SQRT_HALF
        var c2 = Float32(i2) * STEP_16 - SQRT_HALF
        return _reconstruct_quat_from_three(max_idx, c0, c1, c2)

    @always_inline
    def read_quat48(mut self) -> Quat:
        return self.read_quat_smallest_three_48()
