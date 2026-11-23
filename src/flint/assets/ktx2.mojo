"""Pure-Mojo KTX2 container parser, RFC 8878 Zstandard, BasisLZ/ETC1S, UASTC & BCn texture decoder (`ktx2.mojo`)."""
from std.math import isfinite, isnan, max, min, sqrt
from std.memory import bitcast
from flint.assets.bc_compress import (
    _bc7_anchor2,
    _bc7_part2_mask,
    _bc7_weight2,
    _bc7_weight3,
    _bc7_weight4,
    _decode_bc4_channel,
    uf16_bits_to_f32,
)
from flint.assets.image import NativeImage, inflate_zlib

# KTX2 Supercompression schemes
comptime KTX2_SC_NONE: Int = 0
comptime KTX2_SC_BASIS_LZ: Int = 1
comptime KTX2_SC_ZSTD: Int = 2
comptime KTX2_SC_ZLIB: Int = 3

# Khronos DFD color models
comptime DFD_MODEL_RGBSDA: Int = 1
comptime DFD_MODEL_ETC1S: Int = 163
comptime DFD_MODEL_UASTC: Int = 166

# Common VkFormat values
comptime VK_FORMAT_R8_UNORM: Int = 9
comptime VK_FORMAT_R8_SRGB: Int = 15
comptime VK_FORMAT_R8G8_UNORM: Int = 16
comptime VK_FORMAT_R8G8_SRGB: Int = 22
comptime VK_FORMAT_R8G8B8_UNORM: Int = 23
comptime VK_FORMAT_R8G8B8_SRGB: Int = 29
comptime VK_FORMAT_B8G8R8_UNORM: Int = 30
comptime VK_FORMAT_B8G8R8_SRGB: Int = 36
comptime VK_FORMAT_R8G8B8A8_UNORM: Int = 37
comptime VK_FORMAT_R8G8B8A8_SRGB: Int = 43
comptime VK_FORMAT_B8G8R8A8_UNORM: Int = 44
comptime VK_FORMAT_B8G8R8A8_SRGB: Int = 50
comptime VK_FORMAT_R16G16B16A16_SFLOAT: Int = 97
comptime VK_FORMAT_R32G32B32A32_SFLOAT: Int = 109
comptime VK_FORMAT_BC1_RGB_UNORM: Int = 131
comptime VK_FORMAT_BC1_RGB_SRGB: Int = 132
comptime VK_FORMAT_BC1_RGBA_UNORM: Int = 133
comptime VK_FORMAT_BC1_RGBA_SRGB: Int = 134
comptime VK_FORMAT_BC2_UNORM: Int = 135
comptime VK_FORMAT_BC2_SRGB: Int = 136
comptime VK_FORMAT_BC3_UNORM: Int = 137
comptime VK_FORMAT_BC3_SRGB: Int = 138
comptime VK_FORMAT_BC4_UNORM: Int = 139
comptime VK_FORMAT_BC4_SNORM: Int = 140
comptime VK_FORMAT_BC5_UNORM: Int = 141
comptime VK_FORMAT_BC5_SNORM: Int = 142
comptime VK_FORMAT_BC7_UNORM: Int = 145
comptime VK_FORMAT_BC7_SRGB: Int = 146


@always_inline
def _sf16_bits_to_f32(h: Int) -> Float32:
    var sign = UInt32((h >> 15) & 1) << 31
    var exp16 = (h >> 10) & 0x1F
    var mant16 = h & 0x3FF
    if exp16 == 0:
        if mant16 == 0:
            return bitcast[DType.float32](sign)
        var mag = Float32(mant16) * Float32(5.960464477539063e-8)
        return -mag if sign != 0 else mag
    if exp16 == 31:
        if mant16 != 0:
            return bitcast[DType.float32](UInt32(0x7FC00000))
        return bitcast[DType.float32](sign | UInt32(0x7F800000))
    var bits32 = sign | UInt32((exp16 - 15 + 127) << 23) | UInt32(mant16 << 13)
    return bitcast[DType.float32](bits32)


@always_inline
def _f32_unit_to_u8(f: Float32) -> UInt32:
    if isnan(f) or f <= Float32(0.0):
        return UInt32(0)
    if f >= Float32(1.0):
        return UInt32(255)
    return UInt32(max(0, min(255, Int(f * Float32(255.0) + Float32(0.5)))))


@always_inline
def _u16_le(data: List[UInt8], off: Int) -> Int:
    if off < 0 or off + 2 > len(data):
        return 0
    return Int(data[off]) | (Int(data[off + 1]) << 8)


@always_inline
def _u32_le(data: List[UInt8], off: Int) -> Int:
    if off < 0 or off + 4 > len(data):
        return -1
    var b0 = Int(data[off])
    var b1 = Int(data[off + 1]) << 8
    var b2 = Int(data[off + 2]) << 16
    var b3 = Int(data[off + 3]) << 24
    return b0 | b1 | b2 | b3


@always_inline
def _u32_le_u(data: List[UInt8], off: Int) -> UInt32:
    if off < 0 or off + 4 > len(data):
        return 0
    return (
        UInt32(data[off])
        | (UInt32(data[off + 1]) << 8)
        | (UInt32(data[off + 2]) << 16)
        | (UInt32(data[off + 3]) << 24)
    )


@always_inline
def _u64_le(data: List[UInt8], off: Int) -> UInt64:
    if off < 0 or off + 8 > len(data):
        return 0
    var lo = UInt64(_u32_le_u(data, off))
    var hi = UInt64(_u32_le_u(data, off + 4))
    return lo | (hi << 32)


def is_ktx2_magic(data: List[UInt8]) -> Bool:
    if len(data) < 12:
        return False
    return (
        data[0] == 0xAB
        and data[1] == 0x4B
        and data[2] == 0x54
        and data[3] == 0x58
        and data[4] == 0x20
        and data[5] == 0x32
        and data[6] == 0x30
        and data[7] == 0xBB
        and data[8] == 0x0D
        and data[9] == 0x0A
        and data[10] == 0x1A
        and data[11] == 0x0A
    )


# ============================================================================
# 1. Pure-Mojo RFC 8878 Zstandard Decompressor
# ============================================================================

struct _ZstdFwdBits(Copyable, Movable):
    var ptr: Int
    var start: Int
    var end: Int
    var bit_pos: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int, end: Int):
        self.ptr = Int(data.unsafe_ptr())
        self.start = start
        self.end = end
        self.bit_pos = 0
        self.valid = start >= 0 and end >= start and end <= len(data)

    @always_inline
    def bytes_consumed(self) -> Int:
        return (self.bit_pos + 7) // 8

    def read_bits(mut self, count: Int) -> Int:
        if count <= 0:
            return 0
        if not self.valid or count > 31:
            self.valid = False
            return 0
        var total_bits = (self.end - self.start) * 8
        if self.bit_pos + count > total_bits:
            self.valid = False
            return 0
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        var res = 0
        for i in range(count):
            var bp = self.bit_pos + i
            var byte_i = self.start + (bp >> 3)
            var bit_i = bp & 7
            var b = (Int(p[unsafe_offset=byte_i]) >> bit_i) & 1
            res |= b << i
        self.bit_pos += count
        return res


struct _ZstdRevBits(Copyable, Movable):
    """Backward bit reader (sentinel `1` bit in last byte) per RFC 8878 §4.1.1 / §4.2.1."""
    var ptr: Int
    var start: Int
    var byte_pos: Int
    var container: UInt64
    var bits_consumed: Int
    var total_bits: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int, end: Int):
        self.ptr = Int(data.unsafe_ptr())
        self.start = start
        self.byte_pos = end - 1
        self.container = 0
        self.bits_consumed = 0
        self.total_bits = 0
        self.valid = False
        if start < 0 or end <= start or end > len(data):
            return
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        var last_b = Int(p[unsafe_offset=end - 1])
        if last_b == 0:
            return
        var high_bit = 7
        while high_bit > 0 and ((last_b >> high_bit) & 1) == 0:
            high_bit -= 1
        self.total_bits = (end - start - 1) * 8 + high_bit
        self.valid = True

    def read_bits(mut self, count: Int) -> Int:
        if count == 0:
            return 0
        if not self.valid or count < 0 or count > 31:
            self.valid = False
            return 0
        if self.bits_consumed + count > self.total_bits:
            self.valid = False
            return 0
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        var res = 0
        # Read `count` bits MSB-first from the end of the stream (after sentinel)
        var base_bit = self.total_bits - self.bits_consumed
        for i in range(count):
            var pos = base_bit - 1 - i
            var byte_i = self.start + (pos >> 3)
            var bit_i = pos & 7
            var b = (Int(p[unsafe_offset=byte_i]) >> bit_i) & 1
            res = (res << 1) | b
        self.bits_consumed += count
        return res

    @always_inline
    def bits_left(self) -> Int:
        return self.total_bits - self.bits_consumed


struct _ZstdFseTable(Copyable, Movable):
    var table_log: Int
    var symbol: List[UInt8]
    var num_bits: List[UInt8]
    var next_state: List[UInt16]
    var valid: Bool

    def __init__(out self):
        self.table_log = 0
        self.symbol = List[UInt8]()
        self.num_bits = List[UInt8]()
        self.next_state = List[UInt16]()
        self.valid = False

    def build_rle(mut self, sym: UInt8):
        self.table_log = 0
        self.symbol = List[UInt8](length=1, fill=sym)
        self.num_bits = List[UInt8](length=1, fill=0)
        self.next_state = List[UInt16](length=1, fill=0)
        self.valid = True

    def build(mut self, norm: List[Int], table_log: Int) -> Bool:
        if table_log < 0 or table_log > 9:
            return False
        var table_size = 1 << table_log
        self.table_log = table_log
        self.symbol = List[UInt8](length=table_size, fill=0)
        self.num_bits = List[UInt8](length=table_size, fill=0)
        self.next_state = List[UInt16](length=table_size, fill=0)

        var sym_next = List[Int](length=len(norm), fill=0)
        var high_threshold = table_size - 1
        for s in range(len(norm)):
            if norm[s] == -1:
                self.symbol[high_threshold] = UInt8(s)
                high_threshold -= 1
                sym_next[s] = 1
            elif norm[s] > 0:
                sym_next[s] = norm[s]

        var step = (table_size >> 1) + (table_size >> 3) + 3
        var mask = table_size - 1
        var pos = 0
        for s in range(len(norm)):
            var cnt = norm[s]
            if cnt <= 0:
                continue
            for _ in range(cnt):
                self.symbol[pos] = UInt8(s)
                pos = (pos + step) & mask
                while pos > high_threshold:
                    pos = (pos + step) & mask

        if pos != 0:
            self.valid = False
            return False

        for u in range(table_size):
            var s = Int(self.symbol[u])
            var next_s = sym_next[s]
            sym_next[s] += 1
            var nb = table_log
            var tmp = next_s
            while tmp > 1:
                tmp >>= 1
                nb -= 1
            self.num_bits[u] = UInt8(nb)
            self.next_state[u] = UInt16((next_s << nb) - table_size)

        self.valid = True
        return True


def _zstd_read_fse_table(
    data: List[UInt8],
    start: Int,
    end: Int,
    max_sym: Int,
    max_log: Int,
    mut table: _ZstdFseTable,
) -> Int:
    """Parses FSE NCount header from `data[start:end]` and builds `table`. Returns bytes consumed or -1."""
    if start >= end:
        return -1
    var br = _ZstdFwdBits(data, start, end)
    var accuracy_log = br.read_bits(4) + 5
    if not br.valid or accuracy_log > max_log:
        return -1
    var table_size = 1 << accuracy_log
    var remaining = table_size + 1
    var norm = List[Int](length=max_sym + 1, fill=0)
    var sym_idx = 0
    var previous0 = False

    while remaining > 1 and sym_idx <= max_sym and br.valid:
        if previous0:
            var n0 = sym_idx
            var repeat = br.read_bits(2)
            while repeat == 3 and br.valid:
                n0 += 3
                repeat = br.read_bits(2)
            n0 += repeat
            if n0 > max_sym + 1:
                return -1
            sym_idx = n0
            if sym_idx > max_sym:
                break
        var threshold = 1
        var nb_bits = 1
        while (threshold << 1) <= remaining:
            threshold <<= 1
            nb_bits += 1
        var thresh_lo = (threshold << 1) - 1 - remaining
        var v = br.read_bits(nb_bits - 1)
        if v >= thresh_lo:
            v += br.read_bits(1) << (nb_bits - 1)
            if v >= threshold:
                v -= thresh_lo
        if not br.valid:
            return -1
        var prob = v - 1
        norm[sym_idx] = prob
        sym_idx += 1
        if prob == -1:
            remaining -= 1
            previous0 = False
        elif prob == 0:
            previous0 = True
        else:
            remaining -= prob
            previous0 = False

    if remaining != 1 or not br.valid:
        return -1
    while len(norm) > sym_idx:
        _ = norm.pop()
    if not table.build(norm, accuracy_log):
        return -1
    return br.bytes_consumed()


def _zstd_predefined_ll(mut table: _ZstdFseTable):
    var norm: List[Int] = [
        4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
        -1, -1, -1, -1,
    ]
    _ = table.build(norm, 6)


def _zstd_predefined_ml(mut table: _ZstdFseTable):
    var norm: List[Int] = [
        1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
        -1, -1, -1, -1, -1,
    ]
    _ = table.build(norm, 6)


def _zstd_predefined_of(mut table: _ZstdFseTable):
    var norm: List[Int] = [
        1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    ]
    _ = table.build(norm, 5)


struct _ZstdHufTable(Copyable, Movable):
    var max_bits: Int
    var symbols: List[UInt8]
    var bit_lens: List[UInt8]
    var valid: Bool

    def __init__(out self):
        self.max_bits = 0
        self.symbols = List[UInt8]()
        self.bit_lens = List[UInt8]()
        self.valid = False

    def build_from_weights(mut self, weights: List[Int]) -> Bool:
        var sum_pow = 0
        var max_w = 0
        for i in range(len(weights)):
            var w = weights[i]
            if w > 0:
                sum_pow += 1 << (w - 1)
                if w > max_w:
                    max_w = w
        if sum_pow == 0 or max_w > 11:
            return False
        var total_pow = 1
        var max_bits = 0
        while total_pow <= sum_pow:
            total_pow <<= 1
            max_bits += 1
        var rem = total_pow - sum_pow
        if rem <= 0 or (rem & (rem - 1)) != 0:
            return False
        var last_w = 1
        var tmp_rem = rem
        while tmp_rem > 1:
            tmp_rem >>= 1
            last_w += 1
        if last_w > max_bits:
            return False

        var all_w = weights.copy()
        all_w.append(last_w)
        var nb_sym = len(all_w)
        var sym_bits = List[Int](length=nb_sym, fill=0)
        var counts = List[Int](length=max_bits + 2, fill=0)
        for s in range(nb_sym):
            if all_w[s] > 0:
                var bl = max_bits + 1 - all_w[s]
                sym_bits[s] = bl
                counts[bl] += 1

        var tbl_sz = 1 << max_bits
        self.max_bits = max_bits
        self.symbols = List[UInt8](length=tbl_sz, fill=0)
        self.bit_lens = List[UInt8](length=tbl_sz, fill=0)

        var rank_start = List[Int](length=max_bits + 2, fill=0)
        var cur = 0
        var bl = max_bits
        while bl >= 1:
            rank_start[bl] = cur
            cur += counts[bl] << (max_bits - bl)
            bl -= 1

        for s in range(nb_sym):
            var l = sym_bits[s]
            if l == 0:
                continue
            var span = 1 << (max_bits - l)
            var pos = rank_start[l]
            rank_start[l] += span
            for k in range(span):
                if pos + k < tbl_sz:
                    self.symbols[pos + k] = UInt8(s)
                    self.bit_lens[pos + k] = UInt8(l)
        self.valid = True
        return True

    @always_inline
    def decode_sym(self, mut br: _ZstdRevBits) -> UInt8:
        var left = br.bits_left()
        if left <= 0:
            br.valid = False
            return 0
        var mb = self.max_bits
        var peek_n = min(mb, left)
        var saved = br.bits_consumed
        var v = br.read_bits(peek_n)
        if peek_n < mb:
            v <<= mb - peek_n
        var sym = self.symbols[v]
        var bl = Int(self.bit_lens[v])
        if bl == 0 or bl > left:
            br.valid = False
            return 0
        br.bits_consumed = saved + bl
        return sym


def _zstd_decode_huf_stream(
    data: List[UInt8],
    start: Int,
    end: Int,
    huf: _ZstdHufTable,
    expected_bytes: Int,
    mut out: List[UInt8],
) -> Bool:
    if not huf.valid:
        return False
    var br = _ZstdRevBits(data, start, end)
    if not br.valid:
        return False
    for _ in range(expected_bytes):
        var s = huf.decode_sym(br)
        if not br.valid:
            return False
        out.append(s)
    return br.bits_left() == 0


def decompress_zstd(data: List[UInt8], start: Int, byte_len: Int, expected_size: Int = 0) -> Tuple[Bool, List[UInt8]]:
    """Pure-Mojo RFC 8878 Zstandard decompressor."""
    var empty = List[UInt8]()
    if start < 0 or byte_len < 6 or start > len(data) or byte_len > len(data) - start:
        return (False, empty^)
    var end = start + byte_len
    if _u32_le_u(data, start) != 0xFD2FB528:
        return (False, empty^)

    var p = start + 4
    var fhd = Int(data[p])
    p += 1
    var fcs_flag = (fhd >> 6) & 3
    var single_seg = ((fhd >> 5) & 1) != 0
    if ((fhd >> 3) & 1) != 0:
        return (False, empty^)
    var checksum_flag = ((fhd >> 2) & 1) != 0
    var dict_flag = fhd & 3

    if not single_seg:
        if p >= end:
            return (False, empty^)
        p += 1  # window descriptor

    var dict_bytes = 0
    if dict_flag == 1:
        dict_bytes = 1
    elif dict_flag == 2:
        dict_bytes = 2
    elif dict_flag == 3:
        dict_bytes = 4
    if dict_bytes > 0:
        if p + dict_bytes > end:
            return (False, empty^)
        var dict_id = 0
        for di in range(dict_bytes):
            dict_id |= Int(data[p + di]) << (di * 8)
        if dict_id != 0:
            return (False, empty^)
        p += dict_bytes

    var fcs_bytes: Int
    if fcs_flag == 0:
        fcs_bytes = 1 if single_seg else 0
    elif fcs_flag == 1:
        fcs_bytes = 2
    elif fcs_flag == 2:
        fcs_bytes = 4
    else:
        fcs_bytes = 8
    var frame_content_size = -1
    if fcs_bytes > 0:
        if p + fcs_bytes > end:
            return (False, empty^)
        var fcs_val: UInt64 = 0
        for fi in range(fcs_bytes):
            fcs_val |= UInt64(data[p + fi]) << UInt64(fi * 8)
        if fcs_flag == 1:
            fcs_val += 256
        if fcs_val > 268_435_456:
            return (False, empty^)
        frame_content_size = Int(fcs_val)
        p += fcs_bytes

    var max_out = 268_435_456
    if expected_size > 0:
        max_out = min(max_out, expected_size)
    if frame_content_size >= 0:
        if frame_content_size > max_out:
            return (False, empty^)
        max_out = min(max_out, frame_content_size)

    var cap = max(256, min(1 << 20, max_out))
    var out = List[UInt8](capacity=cap)
    var huf = _ZstdHufTable()
    var ll_tbl = _ZstdFseTable()
    var ml_tbl = _ZstdFseTable()
    var of_tbl = _ZstdFseTable()
    var rep = [1, 4, 8]
    var ll_base = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536,
    ]
    var ll_extra = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16,
    ]
    var ml_base = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051,
        4099, 8195, 16387, 32771, 65539,
    ]
    var ml_extra = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
    ]

    var last_block = False
    while not last_block:
        if p + 3 > end:
            return (False, empty^)
        var bh = Int(data[p]) | (Int(data[p + 1]) << 8) | (Int(data[p + 2]) << 16)
        p += 3
        last_block = (bh & 1) != 0
        var btype = (bh >> 1) & 3
        var bsize = bh >> 3
        if btype == 3:
            return (False, empty^)
        if btype == 0:  # Raw
            if p + bsize > end or bsize > max_out - len(out):
                return (False, empty^)
            for i in range(bsize):
                out.append(data[p + i])
            p += bsize
            continue
        if btype == 1:  # RLE
            if p + 1 > end or bsize > max_out - len(out):
                return (False, empty^)
            var b = data[p]
            p += 1
            for _ in range(bsize):
                out.append(b)
            continue

        # Compressed block
        var bend = p + bsize
        if bend > end:
            return (False, empty^)
        if p >= bend:
            return (False, empty^)

        # 1. Literals section
        var lhdr0 = Int(data[p])
        var ltype = lhdr0 & 3
        var sfmt = (lhdr0 >> 2) & 3
        var literals = List[UInt8]()
        if ltype == 0 or ltype == 1:  # Raw or RLE literals
            var regen_sz: Int
            if sfmt == 0 or sfmt == 2:
                regen_sz = lhdr0 >> 3
                p += 1
            elif sfmt == 1:
                if p + 2 > bend:
                    return (False, empty^)
                regen_sz = (lhdr0 >> 4) | (Int(data[p + 1]) << 4)
                p += 2
            else:
                if p + 3 > bend:
                    return (False, empty^)
                regen_sz = (lhdr0 >> 4) | (Int(data[p + 1]) << 4) | (Int(data[p + 2]) << 12)
                p += 3
            if regen_sz > max_out - len(out):
                return (False, empty^)
            if ltype == 0:
                if p + regen_sz > bend:
                    return (False, empty^)
                literals.reserve(regen_sz)
                for i in range(regen_sz):
                    literals.append(data[p + i])
                p += regen_sz
            else:
                if p + 1 > bend:
                    return (False, empty^)
                var lb = data[p]
                p += 1
                literals = List[UInt8](length=regen_sz, fill=lb)
        else:  # Huffman compressed (2) or Treeless (3)
            var regen_sz: Int
            var comp_sz: Int
            var n_streams = 1 if sfmt == 0 else 4
            if sfmt == 0 or sfmt == 1:
                if p + 3 > bend:
                    return (False, empty^)
                var hdr24 = Int(data[p]) | (Int(data[p + 1]) << 8) | (Int(data[p + 2]) << 16)
                regen_sz = (hdr24 >> 4) & 0x3FF
                comp_sz = (hdr24 >> 14) & 0x3FF
                p += 3
            elif sfmt == 2:
                if p + 4 > bend:
                    return (False, empty^)
                var hdr32 = UInt32(data[p]) | (UInt32(data[p + 1]) << 8) | (UInt32(data[p + 2]) << 16) | (UInt32(data[p + 3]) << 24)
                regen_sz = Int((hdr32 >> 4) & 0x3FFF)
                comp_sz = Int((hdr32 >> 18) & 0x3FFF)
                p += 4
            else:
                if p + 5 > bend:
                    return (False, empty^)
                var hdr40 = UInt64(data[p]) | (UInt64(data[p + 1]) << 8) | (UInt64(data[p + 2]) << 16) | (UInt64(data[p + 3]) << 24) | (UInt64(data[p + 4]) << 32)
                regen_sz = Int((hdr40 >> 4) & 0x3FFFF)
                comp_sz = Int((hdr40 >> 22) & 0x3FFFF)
                p += 5

            if n_streams == 4 and regen_sz < 6:
                return (False, empty^)
            if regen_sz > max_out - len(out):
                return (False, empty^)

            var lit_end = p + comp_sz
            if lit_end > bend:
                return (False, empty^)

            if ltype == 2:
                if p >= lit_end:
                    return (False, empty^)
                var huf_hdr = Int(data[p])
                p += 1
                var weights = List[Int]()
                if huf_hdr >= 128:
                    var num_syms = huf_hdr - 127
                    var n_bytes = (num_syms + 1) // 2
                    if p + n_bytes > lit_end:
                        return (False, empty^)
                    for bi in range(num_syms):
                        var bv = Int(data[p + (bi >> 1)])
                        var w = (bv >> 4) if (bi & 1) == 0 else (bv & 0xF)
                        weights.append(w)
                    p += n_bytes
                else:
                    var fse_sz = huf_hdr
                    if p + fse_sz > lit_end:
                        return (False, empty^)
                    var wt_tbl = _ZstdFseTable()
                    var hdr_used = _zstd_read_fse_table(data, p, p + fse_sz, 12, 6, wt_tbl)
                    if hdr_used < 0 or p + hdr_used > p + fse_sz:
                        return (False, empty^)
                    var rbr = _ZstdRevBits(data, p + hdr_used, p + fse_sz)
                    if not rbr.valid:
                        return (False, empty^)
                    var st1 = rbr.read_bits(wt_tbl.table_log)
                    var st2 = rbr.read_bits(wt_tbl.table_log)
                    if not rbr.valid:
                        return (False, empty^)
                    while len(weights) < 255:
                        weights.append(Int(wt_tbl.symbol[st1]))
                        var nb1 = Int(wt_tbl.num_bits[st1])
                        if rbr.bits_left() < nb1:
                            if rbr.bits_left() != 0:
                                return (False, empty^)
                            weights.append(Int(wt_tbl.symbol[st2]))
                            break
                        st1 = Int(wt_tbl.next_state[st1]) + rbr.read_bits(nb1)
                        weights.append(Int(wt_tbl.symbol[st2]))
                        var nb2 = Int(wt_tbl.num_bits[st2])
                        if rbr.bits_left() < nb2:
                            if rbr.bits_left() != 0:
                                return (False, empty^)
                            weights.append(Int(wt_tbl.symbol[st1]))
                            break
                        st2 = Int(wt_tbl.next_state[st2]) + rbr.read_bits(nb2)
                    p += fse_sz
                if not huf.build_from_weights(weights):
                    return (False, empty^)

            literals.reserve(regen_sz)
            if n_streams == 1:
                if not _zstd_decode_huf_stream(data, p, lit_end, huf, regen_sz, literals):
                    return (False, empty^)
            else:
                if p + 6 > lit_end:
                    return (False, empty^)
                var sz1 = _u16_le(data, p)
                var sz2 = _u16_le(data, p + 2)
                var sz3 = _u16_le(data, p + 4)
                p += 6
                if p + sz1 + sz2 + sz3 > lit_end:
                    return (False, empty^)
                var s1_end = p + sz1
                var s2_end = s1_end + sz2
                var s3_end = s2_end + sz3
                var seg = (regen_sz + 3) // 4
                var seg4 = regen_sz - 3 * seg
                if seg4 < 0:
                    return (False, empty^)
                if not _zstd_decode_huf_stream(data, p, s1_end, huf, seg, literals):
                    return (False, empty^)
                if not _zstd_decode_huf_stream(data, s1_end, s2_end, huf, seg, literals):
                    return (False, empty^)
                if not _zstd_decode_huf_stream(data, s2_end, s3_end, huf, seg, literals):
                    return (False, empty^)
                if not _zstd_decode_huf_stream(data, s3_end, lit_end, huf, seg4, literals):
                    return (False, empty^)
            p = lit_end

        # 2. Sequences section
        if p >= bend:
            return (False, empty^)
        var seq_b0 = Int(data[p])
        p += 1
        var num_seq: Int
        if seq_b0 == 0:
            num_seq = 0
        elif seq_b0 < 128:
            num_seq = seq_b0
        elif seq_b0 < 255:
            if p >= bend:
                return (False, empty^)
            num_seq = ((seq_b0 - 128) << 8) + Int(data[p])
            p += 1
        else:
            if p + 2 > bend:
                return (False, empty^)
            num_seq = Int(data[p]) + (Int(data[p + 1]) << 8) + 0x7F00
            p += 2

        var lit_cursor = 0
        if num_seq > 0:
            if p >= bend:
                return (False, empty^)
            var modes = Int(data[p])
            p += 1
            var ll_mode = (modes >> 6) & 3
            var of_mode = (modes >> 4) & 3
            var ml_mode = (modes >> 2) & 3

            if ll_mode == 0:
                _zstd_predefined_ll(ll_tbl)
            elif ll_mode == 1:
                if p >= bend:
                    return (False, empty^)
                ll_tbl.build_rle(data[p])
                p += 1
            elif ll_mode == 2:
                var used = _zstd_read_fse_table(data, p, bend, 35, 9, ll_tbl)
                if used < 0:
                    return (False, empty^)
                p += used
            elif not ll_tbl.valid:
                return (False, empty^)

            if of_mode == 0:
                _zstd_predefined_of(of_tbl)
            elif of_mode == 1:
                if p >= bend:
                    return (False, empty^)
                of_tbl.build_rle(data[p])
                p += 1
            elif of_mode == 2:
                var used = _zstd_read_fse_table(data, p, bend, 31, 8, of_tbl)
                if used < 0:
                    return (False, empty^)
                p += used
            elif not of_tbl.valid:
                return (False, empty^)

            if ml_mode == 0:
                _zstd_predefined_ml(ml_tbl)
            elif ml_mode == 1:
                if p >= bend:
                    return (False, empty^)
                ml_tbl.build_rle(data[p])
                p += 1
            elif ml_mode == 2:
                var used = _zstd_read_fse_table(data, p, bend, 52, 9, ml_tbl)
                if used < 0:
                    return (False, empty^)
                p += used
            elif not ml_tbl.valid:
                return (False, empty^)

            var sbr = _ZstdRevBits(data, p, bend)
            if not sbr.valid:
                return (False, empty^)
            var ll_st = sbr.read_bits(ll_tbl.table_log)
            var of_st = sbr.read_bits(of_tbl.table_log)
            var ml_st = sbr.read_bits(ml_tbl.table_log)
            if not sbr.valid:
                return (False, empty^)

            for si in range(num_seq):
                var ll_code = Int(ll_tbl.symbol[ll_st])
                var ml_code = Int(ml_tbl.symbol[ml_st])
                var of_code = Int(of_tbl.symbol[of_st])
                if ll_code > 35 or ml_code > 52 or of_code > 31:
                    return (False, empty^)

                var of_val = (1 << of_code) + sbr.read_bits(of_code)
                var ml_val = ml_base[ml_code] + sbr.read_bits(ml_extra[ml_code])
                var ll_val = ll_base[ll_code] + sbr.read_bits(ll_extra[ll_code])
                if not sbr.valid:
                    return (False, empty^)

                if si + 1 < num_seq:
                    ll_st = Int(ll_tbl.next_state[ll_st]) + sbr.read_bits(Int(ll_tbl.num_bits[ll_st]))
                    ml_st = Int(ml_tbl.next_state[ml_st]) + sbr.read_bits(Int(ml_tbl.num_bits[ml_st]))
                    of_st = Int(of_tbl.next_state[of_st]) + sbr.read_bits(Int(of_tbl.num_bits[of_st]))
                    if not sbr.valid:
                        return (False, empty^)

                var offset: Int
                if of_code > 1:
                    offset = of_val - 3
                    if offset <= 0:
                        return (False, empty^)
                    rep[2] = rep[1]
                    rep[1] = rep[0]
                    rep[0] = offset
                else:
                    var r_idx = of_val - 1 + (1 if ll_val == 0 else 0)
                    if r_idx == 3:
                        offset = rep[0] - 1
                        if offset <= 0:
                            return (False, empty^)
                        rep[2] = rep[1]
                        rep[1] = rep[0]
                        rep[0] = offset
                    else:
                        offset = rep[r_idx]
                        if r_idx == 1:
                            var tmp = rep[1]
                            rep[1] = rep[0]
                            rep[0] = tmp
                        elif r_idx == 2:
                            var tmp = rep[2]
                            rep[2] = rep[1]
                            rep[1] = rep[0]
                            rep[0] = tmp

                if lit_cursor + ll_val > len(literals) or ll_val > max_out - len(out):
                    return (False, empty^)
                for li in range(ll_val):
                    out.append(literals[lit_cursor + li])
                lit_cursor += ll_val

                if offset <= 0 or offset > len(out) or ml_val > max_out - len(out):
                    return (False, empty^)
                for _ in range(ml_val):
                    out.append(out[len(out) - offset])

            if sbr.bits_left() != 0:
                return (False, empty^)

        var rem_lit = len(literals) - lit_cursor
        if rem_lit > max_out - len(out):
            return (False, empty^)
        while lit_cursor < len(literals):
            out.append(literals[lit_cursor])
            lit_cursor += 1
        p = bend

    if checksum_flag and p + 4 > end:
        return (False, empty^)
    if frame_content_size >= 0 and len(out) != frame_content_size:
        return (False, empty^)
    return (True, out^)


# ============================================================================
# 2. BC1/BC2/BC3/BC4/BC5/BC7 (All 8 Modes) Decoders
# ============================================================================

@always_inline
def _rgb565_to_rgb(c: Int) -> Tuple[Int, Int, Int]:
    var r5 = (c >> 11) & 0x1F
    var g6 = (c >> 5) & 0x3F
    var b5 = c & 0x1F
    var r = (r5 << 3) | (r5 >> 2)
    var g = (g6 << 2) | (g6 >> 4)
    var b = (b5 << 3) | (b5 >> 2)
    return (r, g, b)


def _decode_bc1_block(data: List[UInt8], off: Int, allow_alpha: Bool, mut out: List[UInt32], bx: Int, by: Int, width: Int, height: Int):
    var c0 = _u16_le(data, off)
    var c1 = _u16_le(data, off + 2)
    var rgb0 = _rgb565_to_rgb(c0)
    var rgb1 = _rgb565_to_rgb(c1)
    var pr = Array[Int, 4](fill=0)
    var pg = Array[Int, 4](fill=0)
    var pb = Array[Int, 4](fill=0)
    var pa = Array[Int, 4](fill=255)
    pr[0] = rgb0[0]; pg[0] = rgb0[1]; pb[0] = rgb0[2]
    pr[1] = rgb1[0]; pg[1] = rgb1[1]; pb[1] = rgb1[2]
    if c0 > c1 or not allow_alpha:
        pr[2] = (2 * rgb0[0] + rgb1[0] + 1) // 3
        pg[2] = (2 * rgb0[1] + rgb1[1] + 1) // 3
        pb[2] = (2 * rgb0[2] + rgb1[2] + 1) // 3
        pr[3] = (rgb0[0] + 2 * rgb1[0] + 1) // 3
        pg[3] = (rgb0[1] + 2 * rgb1[1] + 1) // 3
        pb[3] = (rgb0[2] + 2 * rgb1[2] + 1) // 3
    else:
        pr[2] = (rgb0[0] + rgb1[0]) // 2
        pg[2] = (rgb0[1] + rgb1[1]) // 2
        pb[2] = (rgb0[2] + rgb1[2]) // 2
        pr[3] = 0; pg[3] = 0; pb[3] = 0; pa[3] = 0
    var idx_bits = _u32_le_u(data, off + 4)
    for t in range(16):
        var px = bx * 4 + (t & 3)
        var py = by * 4 + (t >> 2)
        if px < width and py < height:
            var sel = Int((idx_bits >> UInt32(t * 2)) & 3)
            out[py * width + px] = (UInt32(pa[sel]) << 24) | (UInt32(pr[sel]) << 16) | (UInt32(pg[sel]) << 8) | UInt32(pb[sel])


struct _BitReader128Local(Copyable, Movable):
    var lo: UInt64
    var hi: UInt64
    var pos: Int

    def __init__(out self, data: List[UInt8], off: Int):
        self.lo = _u64_le(data, off)
        self.hi = _u64_le(data, off + 8)
        self.pos = 0

    def get(mut self, count: Int) -> Int:
        if count <= 0:
            return 0
        var v = 0
        for i in range(count):
            var p = self.pos + i
            var b: Int
            if p < 64:
                b = Int((self.lo >> UInt64(p)) & 1)
            else:
                b = Int((self.hi >> UInt64(p - 64)) & 1)
            v |= b << i
        self.pos += count
        return v


@always_inline
def _bc7_part3_mask(part: Int) -> Tuple[Int, Int]:
    """Exact 64-entry Khronos BPTC Table.P3 3-subset 2-bit-per-texel packed partition masks."""
    var p3: List[Int] = [
        0xAA685050, 0x6A5A5040, 0x5A5A4200, 0x5450A0A8, 0xA5A50000, 0xA0A05050, 0x5555A0A0, 0x5A5A5050,
        0xAA550000, 0xAA555500, 0xAAAA5500, 0x90909090, 0x94949494, 0xA4A4A4A4, 0xA9A59450, 0x2A0A4250,
        0xA5945040, 0x0A425054, 0xA5A5A500, 0x55A0A0A0, 0xA8A85454, 0x6A6A4040, 0xA4A45000, 0x1A1A0500,
        0x0050A4A4, 0xAAA59090, 0x14696914, 0x69691400, 0xA08585A0, 0xAA821414, 0x50A4A450, 0x6A5A0200,
        0xA9A58000, 0x5090A0A8, 0xA8A09050, 0x24242424, 0x00AA5500, 0x24924924, 0x24499224, 0x50A50A50,
        0x500AA550, 0xAAAA4444, 0x66660000, 0xA5A0A5A0, 0x50A050A0, 0x69286928, 0x44AAAA44, 0x66666600,
        0xAA444444, 0x54A854A8, 0x95809580, 0x96969600, 0xA85454A8, 0x80959580, 0xAA141414, 0x96960000,
        0xAAAA1414, 0xA05050A0, 0xA0A5A5A0, 0x96000000, 0x40804080, 0xA9A8A9A8, 0xAAAAAA44, 0x2A4A5254,
    ]
    return (p3[part & 63], 0)


@always_inline
def _bc7_anchor3(part: Int) -> Tuple[Int, Int]:
    var tbl2: List[Int] = [
        3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
        3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
        8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
        3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
    ]
    var tbl3: List[Int] = [
        15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
        15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
        15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
        15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
    ]
    var p = part & 63
    return (tbl2[p], tbl3[p])


def _decode_bc7_full(blocks: List[UInt8], width: Int, height: Int) -> List[UInt32]:
    """Full 8-mode (`0..7`) Khronos BC7 decoder."""
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt32](length=width * height, fill=0xFF000000)
    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            if off + 16 > len(blocks):
                continue
            var br = _BitReader128Local(blocks, off)
            var mode = 0
            while mode < 8:
                if br.get(1) == 1:
                    break
                mode += 1
            if mode >= 8:
                continue

            if mode == 6:
                var cr = Array[Int, 2](fill=0); var cg = Array[Int, 2](fill=0)
                var cb = Array[Int, 2](fill=0); var ca = Array[Int, 2](fill=0)
                cr[0] = br.get(7); cr[1] = br.get(7)
                cg[0] = br.get(7); cg[1] = br.get(7)
                cb[0] = br.get(7); cb[1] = br.get(7)
                ca[0] = br.get(7); ca[1] = br.get(7)
                var p0 = br.get(1); var p1 = br.get(1)
                var er0 = (cr[0] << 1) | p0; var er1 = (cr[1] << 1) | p1
                var eg0 = (cg[0] << 1) | p0; var eg1 = (cg[1] << 1) | p1
                var eb0 = (cb[0] << 1) | p0; var eb1 = (cb[1] << 1) | p1
                var ea0 = (ca[0] << 1) | p0; var ea1 = (ca[1] << 1) | p1
                for t in range(16):
                    var idx = br.get(3) if t == 0 else br.get(4)
                    var w = _bc7_weight4(idx)
                    var r = ((64 - w) * er0 + w * er1 + 32) >> 6
                    var g = ((64 - w) * eg0 + w * eg1 + 32) >> 6
                    var b = ((64 - w) * eb0 + w * eb1 + 32) >> 6
                    var a = ((64 - w) * ea0 + w * ea1 + 32) >> 6
                    var py = by * 4 + (t >> 2); var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
            elif mode == 4 or mode == 5:
                var rot = br.get(2)
                var idx_mode = br.get(1) if mode == 4 else 0
                var cb_bits = 5 if mode == 4 else 7
                var ab_bits = 6 if mode == 4 else 8
                var cr0 = br.get(cb_bits); var cr1 = br.get(cb_bits)
                var cg0 = br.get(cb_bits); var cg1 = br.get(cb_bits)
                var cb0 = br.get(cb_bits); var cb1 = br.get(cb_bits)
                var ca0 = br.get(ab_bits); var ca1 = br.get(ab_bits)
                var er0 = (cr0 << (8 - cb_bits)) | (cr0 >> (2 * cb_bits - 8))
                var er1 = (cr1 << (8 - cb_bits)) | (cr1 >> (2 * cb_bits - 8))
                var eg0 = (cg0 << (8 - cb_bits)) | (cg0 >> (2 * cb_bits - 8))
                var eg1 = (cg1 << (8 - cb_bits)) | (cg1 >> (2 * cb_bits - 8))
                var eb0 = (cb0 << (8 - cb_bits)) | (cb0 >> (2 * cb_bits - 8))
                var eb1 = (cb1 << (8 - cb_bits)) | (cb1 >> (2 * cb_bits - 8))
                var ea0 = (ca0 << (8 - ab_bits)) | (ca0 >> (2 * ab_bits - 8)) if ab_bits < 8 else ca0
                var ea1 = (ca1 << (8 - ab_bits)) | (ca1 >> (2 * ab_bits - 8)) if ab_bits < 8 else ca1
                var c_idx = Array[Int, 16](fill=0)
                var a_idx = Array[Int, 16](fill=0)
                if mode == 4:
                    var idx2 = Array[Int, 16](fill=0)
                    var idx3 = Array[Int, 16](fill=0)
                    for t in range(16):
                        idx2[t] = br.get(1 if t == 0 else 2)
                    for t in range(16):
                        idx3[t] = br.get(2 if t == 0 else 3)
                    for t in range(16):
                        if idx_mode == 0:
                            c_idx[t] = idx2[t]
                            a_idx[t] = idx3[t]
                        else:
                            c_idx[t] = idx3[t]
                            a_idx[t] = idx2[t]
                else:
                    for t in range(16):
                        c_idx[t] = br.get(1 if t == 0 else 2)
                    for t in range(16):
                        a_idx[t] = br.get(1 if t == 0 else 2)
                for t in range(16):
                    var wc = _bc7_weight2(c_idx[t]) if (mode == 5 or idx_mode == 0) else _bc7_weight3(c_idx[t])
                    var wa = _bc7_weight2(a_idx[t]) if (mode == 5 or idx_mode == 1) else _bc7_weight3(a_idx[t])
                    var r = ((64 - wc) * er0 + wc * er1 + 32) >> 6
                    var g = ((64 - wc) * eg0 + wc * eg1 + 32) >> 6
                    var b = ((64 - wc) * eb0 + wc * eb1 + 32) >> 6
                    var a = ((64 - wa) * ea0 + wa * ea1 + 32) >> 6
                    if rot == 1:
                        var tmp = r; r = a; a = tmp
                    elif rot == 2:
                        var tmp = g; g = a; a = tmp
                    elif rot == 3:
                        var tmp = b; b = a; a = tmp
                    var py = by * 4 + (t >> 2); var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
            elif mode == 1 or mode == 3 or mode == 7:
                var part = br.get(6)
                var mask = _bc7_part2_mask(part)
                var anchor2 = _bc7_anchor2(part)
                var cr = Array[Int, 4](fill=0); var cg = Array[Int, 4](fill=0)
                var cb = Array[Int, 4](fill=0); var ca = Array[Int, 4](fill=255)
                if mode == 7:
                    for ep in range(4): cr[ep] = br.get(5)
                    for ep in range(4): cg[ep] = br.get(5)
                    for ep in range(4): cb[ep] = br.get(5)
                    for ep in range(4): ca[ep] = br.get(5)
                    var pb = Array[Int, 4](fill=0)
                    for ep in range(4): pb[ep] = br.get(1)
                    for t in range(16):
                        var idx = br.get(1) if (t == 0 or t == anchor2) else br.get(2)
                        var w = _bc7_weight2(idx)
                        var sub = (mask >> t) & 1
                        var e0 = sub * 2; var e1 = e0 + 1
                        var r0 = (cr[e0] << 3) | (pb[e0] << 2) | (cr[e0] >> 3)
                        var r1 = (cr[e1] << 3) | (pb[e1] << 2) | (cr[e1] >> 3)
                        var g0 = (cg[e0] << 3) | (pb[e0] << 2) | (cg[e0] >> 3)
                        var g1 = (cg[e1] << 3) | (pb[e1] << 2) | (cg[e1] >> 3)
                        var b0 = (cb[e0] << 3) | (pb[e0] << 2) | (cb[e0] >> 3)
                        var b1 = (cb[e1] << 3) | (pb[e1] << 2) | (cb[e1] >> 3)
                        var a0 = (ca[e0] << 3) | (pb[e0] << 2) | (ca[e0] >> 3)
                        var a1 = (ca[e1] << 3) | (pb[e1] << 2) | (ca[e1] >> 3)
                        var r = ((64 - w) * r0 + w * r1 + 32) >> 6
                        var g = ((64 - w) * g0 + w * g1 + 32) >> 6
                        var b = ((64 - w) * b0 + w * b1 + 32) >> 6
                        var a = ((64 - w) * a0 + w * a1 + 32) >> 6
                        var py = by * 4 + (t >> 2); var px = bx * 4 + (t & 3)
                        if px < width and py < height:
                            out[py * width + px] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                elif mode == 3:
                    for ep in range(4): cr[ep] = br.get(7)
                    for ep in range(4): cg[ep] = br.get(7)
                    for ep in range(4): cb[ep] = br.get(7)
                    var pb = Array[Int, 4](fill=0)
                    for ep in range(4): pb[ep] = br.get(1)
                    for t in range(16):
                        var idx = br.get(1) if (t == 0 or t == anchor2) else br.get(2)
                        var w = _bc7_weight2(idx)
                        var sub = (mask >> t) & 1
                        var e0 = sub * 2; var e1 = e0 + 1
                        var r = ((64 - w) * ((cr[e0] << 1) | pb[e0]) + w * ((cr[e1] << 1) | pb[e1]) + 32) >> 6
                        var g = ((64 - w) * ((cg[e0] << 1) | pb[e0]) + w * ((cg[e1] << 1) | pb[e1]) + 32) >> 6
                        var b = ((64 - w) * ((cb[e0] << 1) | pb[e0]) + w * ((cb[e1] << 1) | pb[e1]) + 32) >> 6
                        var py = by * 4 + (t >> 2); var px = bx * 4 + (t & 3)
                        if px < width and py < height:
                            out[py * width + px] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                else:
                    for ep in range(4): cr[ep] = br.get(6)
                    for ep in range(4): cg[ep] = br.get(6)
                    for ep in range(4): cb[ep] = br.get(6)
                    var ps = Array[Int, 2](fill=0)
                    ps[0] = br.get(1); ps[1] = br.get(1)
                    for t in range(16):
                        var idx = br.get(2) if (t == 0 or t == anchor2) else br.get(3)
                        var w = _bc7_weight3(idx)
                        var sub = (mask >> t) & 1
                        var p = ps[sub]; var e0 = sub * 2; var e1 = e0 + 1
                        var r0 = (cr[e0] << 2) | (p << 1) | (cr[e0] >> 5)
                        var r1 = (cr[e1] << 2) | (p << 1) | (cr[e1] >> 5)
                        var g0 = (cg[e0] << 2) | (p << 1) | (cg[e0] >> 5)
                        var g1 = (cg[e1] << 2) | (p << 1) | (cg[e1] >> 5)
                        var b0 = (cb[e0] << 2) | (p << 1) | (cb[e0] >> 5)
                        var b1 = (cb[e1] << 2) | (p << 1) | (cb[e1] >> 5)
                        var r = ((64 - w) * r0 + w * r1 + 32) >> 6
                        var g = ((64 - w) * g0 + w * g1 + 32) >> 6
                        var b = ((64 - w) * b0 + w * b1 + 32) >> 6
                        var py = by * 4 + (t >> 2); var px = bx * 4 + (t & 3)
                        if px < width and py < height:
                            out[py * width + px] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
            else:  # Mode 0 (3-subset 4-bit + P) or Mode 2 (3-subset 5-bit)
                var part = br.get(4 if mode == 0 else 6)
                var pbits = _bc7_part3_mask(part)[0]
                var anchors = _bc7_anchor3(part)
                var cb_bits = 4 if mode == 0 else 5
                var cr = Array[Int, 6](fill=0); var cg = Array[Int, 6](fill=0); var cb = Array[Int, 6](fill=0)
                var pb = Array[Int, 6](fill=0)
                for ep in range(6): cr[ep] = br.get(cb_bits)
                for ep in range(6): cg[ep] = br.get(cb_bits)
                for ep in range(6): cb[ep] = br.get(cb_bits)
                if mode == 0:
                    for ep in range(6): pb[ep] = br.get(1)
                for t in range(16):
                    var is_anc = t == 0 or t == anchors[0] or t == anchors[1]
                    var idx = br.get((2 if mode == 0 else 1) if is_anc else (3 if mode == 0 else 2))
                    var w = _bc7_weight3(idx) if mode == 0 else _bc7_weight2(idx)
                    var sub = (pbits >> (t * 2)) & 3
                    if sub > 2:
                        sub = 2
                    var e0 = sub * 2; var e1 = e0 + 1
                    var r0: Int; var r1: Int; var g0: Int; var g1: Int; var b0: Int; var b1: Int
                    if mode == 0:
                        r0 = (cr[e0] << 4) | (pb[e0] << 3) | (cr[e0] >> 1)
                        r1 = (cr[e1] << 4) | (pb[e1] << 3) | (cr[e1] >> 1)
                        g0 = (cg[e0] << 4) | (pb[e0] << 3) | (cg[e0] >> 1)
                        g1 = (cg[e1] << 4) | (pb[e1] << 3) | (cg[e1] >> 1)
                        b0 = (cb[e0] << 4) | (pb[e0] << 3) | (cb[e0] >> 1)
                        b1 = (cb[e1] << 4) | (pb[e1] << 3) | (cb[e1] >> 1)
                    else:
                        r0 = (cr[e0] << 3) | (cr[e0] >> 2); r1 = (cr[e1] << 3) | (cr[e1] >> 2)
                        g0 = (cg[e0] << 3) | (cg[e0] >> 2); g1 = (cg[e1] << 3) | (cg[e1] >> 2)
                        b0 = (cb[e0] << 3) | (cb[e0] >> 2); b1 = (cb[e1] << 3) | (cb[e1] >> 2)
                    var r = ((64 - w) * r0 + w * r1 + 32) >> 6
                    var g = ((64 - w) * g0 + w * g1 + 32) >> 6
                    var b = ((64 - w) * b0 + w * b1 + 32) >> 6
                    var py = by * 4 + (t >> 2); var px = bx * 4 + (t & 3)
                    if px < width and py < height:
                        out[py * width + px] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    return out^


# ============================================================================
# 3. Khronos UASTC (colorModel = 166) Decoder
# ============================================================================

@always_inline
def _uastc_read_bits(lo: UInt64, hi: UInt64, pos: Int, cnt: Int) -> Int:
    if cnt <= 0:
        return 0
    var v = 0
    for i in range(cnt):
        var p = pos + i
        var b = Int((lo >> UInt64(p)) & 1) if p < 64 else Int((hi >> UInt64(p - 64)) & 1)
        v |= b << i
    return v


@always_inline
def _uastc_mode_from_lo(lo: UInt64) -> Int:
    """Decodes UASTC Huffman mode prefix (Khronos DFS §23.2, 0..18 valid, 19 reserved)."""
    var tbl: Array[Int, 128] = [
        11, 0, 10, 3, 11, 15, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
        11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
        11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
        11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
        11, 0, 10, 3, 11, 19, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
        11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
        11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
        11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
    ]
    return tbl[Int(lo & 0x7F)]


@always_inline
def _astc_hash52(p_in: UInt32) -> UInt32:
    var p = p_in
    p ^= p >> 15
    p -= p << 17
    p += p << 7
    p += p << 4
    p ^= p >> 5
    p += p << 16
    p ^= p >> 7
    p ^= p >> 3
    p ^= p << 6
    p ^= p >> 17
    return p


@always_inline
def _astc_select_partition(seed_in: Int, x_in: Int, y_in: Int, partitioncount: Int) -> Int:
    """Khronos ASTC spec §C.2.21 `select_partition` for 4x4 2D small blocks (small_block=True)."""
    var x = x_in << 1
    var y = y_in << 1
    var seed = seed_in + (partitioncount - 1) * 1024
    var rnum = _astc_hash52(UInt32(seed))
    var s1 = Int(rnum & 0xF)
    var s2 = Int((rnum >> 4) & 0xF)
    var s3 = Int((rnum >> 8) & 0xF)
    var s4 = Int((rnum >> 12) & 0xF)
    var s5 = Int((rnum >> 16) & 0xF)
    var s6 = Int((rnum >> 20) & 0xF)
    var s7 = Int((rnum >> 24) & 0xF)
    var s8 = Int((rnum >> 28) & 0xF)
    s1 *= s1; s2 *= s2; s3 *= s3; s4 *= s4
    s5 *= s5; s6 *= s6; s7 *= s7; s8 *= s8
    var sh1: Int
    var sh2: Int
    if (seed & 1) != 0:
        sh1 = 4 if (seed & 2) != 0 else 5
        sh2 = 6 if partitioncount == 3 else 5
    else:
        sh1 = 6 if partitioncount == 3 else 5
        sh2 = 4 if (seed & 2) != 0 else 5
    s1 >>= sh1; s2 >>= sh2; s3 >>= sh1; s4 >>= sh2
    s5 >>= sh1; s6 >>= sh2; s7 >>= sh1; s8 >>= sh2
    var a = (s1 * x + s2 * y + Int(rnum >> 14)) & 0x3F
    var b = (s3 * x + s4 * y + Int(rnum >> 10)) & 0x3F
    var c = ((s5 * x + s6 * y + Int(rnum >> 6)) & 0x3F) if partitioncount >= 3 else 0
    if a >= b and a >= c:
        return 0
    elif b >= c:
        return 1
    return 2


@always_inline
def _uastc_unquant_endpoint_astc(val: Int, ep_range: Int) -> Int:
    """Exact Khronos ASTC §C.2.13 endpoint unquantization to [0, 255] for UASTC ranges."""
    if ep_range == 20:
        return val & 0xFF
    if ep_range == 11:
        var v5 = val & 31
        return (v5 << 3) | (v5 >> 2)
    if ep_range == 8:
        var v4 = val & 15
        return (v4 << 4) | v4
    var a_mask = 0x1FF if (val & 1) != 0 else 0
    var b_reg: Int
    var c_mul: Int
    var d_tq: Int
    if ep_range == 7:
        d_tq = val >> 2
        var b1 = (val >> 1) & 1
        b_reg = (b1 << 8) | (b1 << 4) | (b1 << 2) | (b1 << 1)
        c_mul = 93
    elif ep_range == 12:
        d_tq = val >> 3
        var cb = (val >> 1) & 3
        b_reg = (cb << 7) | (cb << 2) | cb
        c_mul = 26
    elif ep_range == 13:
        d_tq = val >> 4
        var dcb = (val >> 1) & 7
        b_reg = (dcb << 6) | dcb
        c_mul = 22
    elif ep_range == 18:
        d_tq = val >> 5
        var edcb = (val >> 1) & 15
        var e = (val >> 4) & 1
        b_reg = (edcb << 5) | e
        c_mul = 6
    else:
        # ep_range == 19
        d_tq = val >> 6
        var fedcb = (val >> 1) & 31
        var f = (val >> 5) & 1
        b_reg = (fedcb << 4) | f
        c_mul = 5
    var t = (d_tq * c_mul + b_reg) ^ a_mask
    return (a_mask & 0x80) | (t >> 2)


@always_inline
def _uastc_unquant_weight(wi: Int, w_bits: Int) -> Int:
    """Exact Khronos ASTC §C.2.15 weight unquantization to [0, 64] for pure-bit ranges."""
    var t: Int
    if w_bits == 1:
        return 64 if (wi & 1) != 0 else 0
    elif w_bits == 2:
        t = (wi & 3) * 21
    elif w_bits == 3:
        var v3 = wi & 7
        t = (v3 << 3) | v3
    elif w_bits == 4:
        var v4 = wi & 15
        t = (v4 << 2) | (v4 >> 2)
    else:
        var v5 = wi & 31
        t = (v5 << 1) | (v5 >> 4)
    if t > 32:
        t += 1
    return t


@always_inline
def _uastc_unpack_trits(code: Int, count: Int, mut dst: Array[Int, 18], dst_off: Int) -> Bool:
    var limit = 243 if count == 5 else (27 if count == 3 else 3)
    if code < 0 or code >= limit:
        return False
    var v = code
    for i in range(count):
        dst[dst_off + i] = v % 3
        v //= 3
    return True


@always_inline
def _uastc_unpack_quints(code: Int, count: Int, mut dst: Array[Int, 18], dst_off: Int) -> Bool:
    var limit = 125
    if code < 0 or code >= limit:
        return False
    var v = code
    for i in range(count):
        dst[dst_off + i] = v % 5
        v //= 5
    return True


def _decode_uastc_blocks(blocks: List[UInt8], width: Int, height: Int) -> Tuple[Bool, List[UInt32]]:
    """Decodes 128-bit 4x4 Khronos UASTC blocks (DFS §23, exact Modes 0..18)."""
    var empty = List[UInt32]()
    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var out = List[UInt32](length=width * height, fill=0xFF000000)

    var pat2_seeds: Array[Int, 30] = [
        28, 20, 16, 29, 91, 9, 107, 72, 149, 204, 50, 114, 496, 17, 78,
        39, 252, 828, 43, 156, 116, 210, 476, 273, 684, 359, 246, 195, 694, 524,
    ]
    var pat7_seeds: Array[Int, 19] = [
        36, 48, 61, 137, 161, 183, 226, 281, 302, 307, 479, 495, 593, 594, 605, 799, 812, 988, 993,
    ]
    var pat3_seeds: Array[Int, 11] = [
        260, 74, 32, 156, 183, 15, 745, 0, 335, 902, 254,
    ]

    for by in range(bh):
        for bx in range(bw):
            var off = (by * bw + bx) * 16
            if off + 16 > len(blocks):
                return (False, empty^)
            var lo = _u64_le(blocks, off)
            var hi = _u64_le(blocks, off + 8)
            var mode = _uastc_mode_from_lo(lo)
            if mode >= 19:
                return (False, empty^)

            if mode == 8:
                # Khronos DFS §23 Mode 8: Void-Extent constant RGBA8
                var r = _uastc_read_bits(lo, hi, 5, 8)
                var g = _uastc_read_bits(lo, hi, 13, 8)
                var b = _uastc_read_bits(lo, hi, 21, 8)
                var a = _uastc_read_bits(lo, hi, 29, 8)
                var px_col = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                for t in range(16):
                    var px = bx * 4 + (t & 3); var py = by * 4 + (t >> 2)
                    if px < width and py < height:
                        out[py * width + px] = px_col
                continue

            # Mode table properties (Khronos DFS §23)
            var n_sub = 1
            var planes = 1
            var is_la = False
            var has_alpha = False
            var ep_range: Int
            var ep_bits: Int
            var w_bits: Int
            var pat_pos = -1
            var pat_bits = 0
            var compset_pos = -1
            var etq_pos = -1
            var ebits_pos: Int
            var w_pos: Int

            if mode == 0:
                ep_range = 19; ep_bits = 6; w_bits = 4; etq_pos = 19; ebits_pos = 29; w_pos = 65
            elif mode == 1:
                ep_range = 20; ep_bits = 8; w_bits = 2; ebits_pos = 21; w_pos = 69
            elif mode == 2:
                n_sub = 2; ep_range = 8; ep_bits = 4; w_bits = 3; pat_pos = 20; pat_bits = 5; ebits_pos = 25; w_pos = 73
            elif mode == 3:
                n_sub = 3; ep_range = 7; ep_bits = 2; w_bits = 2; pat_pos = 20; pat_bits = 4; etq_pos = 24; ebits_pos = 53; w_pos = 89
            elif mode == 4:
                n_sub = 2; ep_range = 12; ep_bits = 3; w_bits = 2; pat_pos = 20; pat_bits = 5; etq_pos = 25; ebits_pos = 53; w_pos = 89
            elif mode == 5:
                ep_range = 20; ep_bits = 8; w_bits = 3; ebits_pos = 20; w_pos = 68
            elif mode == 6:
                planes = 2; ep_range = 18; ep_bits = 5; w_bits = 2; compset_pos = 20; etq_pos = 22; ebits_pos = 36; w_pos = 66
            elif mode == 7:
                n_sub = 2; ep_range = 12; ep_bits = 3; w_bits = 2; pat_pos = 20; pat_bits = 5; etq_pos = 25; ebits_pos = 53; w_pos = 89
            elif mode == 9:
                n_sub = 2; has_alpha = True; ep_range = 8; ep_bits = 4; w_bits = 2; pat_pos = 28; pat_bits = 5; ebits_pos = 33; w_pos = 97
            elif mode == 10:
                has_alpha = True; ep_range = 13; ep_bits = 4; w_bits = 4; etq_pos = 20; ebits_pos = 33; w_pos = 65
            elif mode == 11:
                planes = 2; has_alpha = True; ep_range = 13; ep_bits = 4; w_bits = 2; compset_pos = 19; etq_pos = 21; ebits_pos = 34; w_pos = 66
            elif mode == 12:
                has_alpha = True; ep_range = 19; ep_bits = 6; w_bits = 3; etq_pos = 20; ebits_pos = 33; w_pos = 81
            elif mode == 13:
                planes = 2; has_alpha = True; ep_range = 20; ep_bits = 8; w_bits = 1; compset_pos = 28; ebits_pos = 30; w_pos = 94
            elif mode == 14:
                has_alpha = True; ep_range = 20; ep_bits = 8; w_bits = 2; ebits_pos = 28; w_pos = 92
            elif mode == 15:
                is_la = True; has_alpha = True; ep_range = 20; ep_bits = 8; w_bits = 4; ebits_pos = 30; w_pos = 62
            elif mode == 16:
                n_sub = 2; is_la = True; has_alpha = True; ep_range = 20; ep_bits = 8; w_bits = 2; pat_pos = 29; pat_bits = 5; ebits_pos = 34; w_pos = 98
            elif mode == 17:
                planes = 2; is_la = True; has_alpha = True; ep_range = 20; ep_bits = 8; w_bits = 2; ebits_pos = 29; w_pos = 61
            else:
                # mode == 18
                ep_range = 11; ep_bits = 5; w_bits = 5; ebits_pos = 19; w_pos = 49

            var ccs = 3 if mode == 17 else 0
            if compset_pos >= 0:
                ccs = _uastc_read_bits(lo, hi, compset_pos, 2)
                if mode == 6 and ccs == 3:
                    return (False, empty^)

            var subsets = Array[Int, 16](fill=0)
            if n_sub > 1:
                var pat = _uastc_read_bits(lo, hi, pat_pos, pat_bits)
                var seed: Int
                if mode == 3:
                    if pat >= 11:
                        return (False, empty^)
                    seed = pat3_seeds[pat]
                elif mode == 7:
                    if pat >= 19:
                        return (False, empty^)
                    seed = pat7_seeds[pat]
                else:
                    if pat >= 30:
                        return (False, empty^)
                    seed = pat2_seeds[pat]
                for t in range(16):
                    subsets[t] = _astc_select_partition(seed, t & 3, t >> 2, n_sub)

            # Decode ETQ trits/quints if present
            var ch_per_ep = 2 if is_la else (4 if has_alpha else 3)
            var total_ep = n_sub * 2 * ch_per_ep
            var tq = Array[Int, 18](fill=0)
            if etq_pos >= 0:
                if mode == 0:
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, 19, 8), 5, tq, 0):
                        return (False, empty^)
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, 27, 2), 1, tq, 5):
                        return (False, empty^)
                elif mode == 3:
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, 24, 8), 5, tq, 0):
                        return (False, empty^)
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, 32, 8), 5, tq, 5):
                        return (False, empty^)
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, 40, 8), 5, tq, 10):
                        return (False, empty^)
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, 48, 5), 3, tq, 15):
                        return (False, empty^)
                elif mode == 4 or mode == 7:
                    for gi in range(4):
                        if not _uastc_unpack_quints(_uastc_read_bits(lo, hi, 25 + gi * 7, 7), 3, tq, gi * 3):
                            return (False, empty^)
                elif mode == 6:
                    for gi in range(2):
                        if not _uastc_unpack_quints(_uastc_read_bits(lo, hi, 22 + gi * 7, 7), 3, tq, gi * 3):
                            return (False, empty^)
                else:
                    # Modes 10, 11, 12 (8 trits)
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, etq_pos, 8), 5, tq, 0):
                        return (False, empty^)
                    if not _uastc_unpack_trits(_uastc_read_bits(lo, hi, etq_pos + 8, 5), 3, tq, 5):
                        return (False, empty^)

            var ep_vals = Array[Int, 24](fill=0)
            for ei in range(total_ep):
                var raw_bits = _uastc_read_bits(lo, hi, ebits_pos + ei * ep_bits, ep_bits)
                var q_val = (tq[ei] << ep_bits) | raw_bits
                ep_vals[ei] = _uastc_unquant_endpoint_astc(q_val, ep_range)

            # Determine anchor texels (first texel in raster order for each subset)
            var anc = Array[Int, 3](fill=-1)
            for t in range(16):
                var s = subsets[t]
                if anc[s] < 0:
                    anc[s] = t

            var cur_w = w_pos
            for t in range(16):
                var sub = subsets[t]
                var is_anc = t == anc[sub]
                var w0_raw: Int
                var w1_raw: Int
                if planes == 2:
                    var rb = w_bits - 1 if t == 0 else w_bits
                    w0_raw = _uastc_read_bits(lo, hi, cur_w, rb) if rb > 0 else 0
                    cur_w += rb
                    w1_raw = _uastc_read_bits(lo, hi, cur_w, rb) if rb > 0 else 0
                    cur_w += rb
                else:
                    var rb = w_bits - 1 if is_anc else w_bits
                    w0_raw = _uastc_read_bits(lo, hi, cur_w, rb) if rb > 0 else 0
                    cur_w += rb
                    w1_raw = w0_raw

                var w0 = _uastc_unquant_weight(w0_raw, w_bits)
                var w1 = _uastc_unquant_weight(w1_raw, w_bits)
                var base = sub * 2 * ch_per_ep
                var r: Int
                var g: Int
                var b: Int
                var a: Int
                if is_la:
                    var wl = w0
                    var wa = w1 if planes == 2 else w0
                    var l = ((64 - wl) * ep_vals[base] + wl * ep_vals[base + 1] + 32) >> 6
                    a = ((64 - wa) * ep_vals[base + 2] + wa * ep_vals[base + 3] + 32) >> 6
                    r = l; g = l; b = l
                else:
                    var wr = w1 if (planes == 2 and ccs == 0) else w0
                    var wg = w1 if (planes == 2 and ccs == 1) else w0
                    var wb = w1 if (planes == 2 and ccs == 2) else w0
                    r = ((64 - wr) * ep_vals[base] + wr * ep_vals[base + 1] + 32) >> 6
                    g = ((64 - wg) * ep_vals[base + 2] + wg * ep_vals[base + 3] + 32) >> 6
                    b = ((64 - wb) * ep_vals[base + 4] + wb * ep_vals[base + 5] + 32) >> 6
                    if has_alpha:
                        var wa = w1 if (planes == 2 and ccs == 3) else w0
                        a = ((64 - wa) * ep_vals[base + 6] + wa * ep_vals[base + 7] + 32) >> 6
                    else:
                        a = 255
                var px = bx * 4 + (t & 3); var py = by * 4 + (t >> 2)
                if px < width and py < height:
                    out[py * width + px] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    return (True, out^)


# ============================================================================
# 4. BasisLZ / ETC1S (`supercompressionScheme == 1`, `colorModel == 163`)
# ============================================================================

struct _BasisLzBits(Copyable, Movable):
    var ptr: Int
    var pos_bytes: Int
    var end_bytes: Int
    var buf: UInt64
    var bit_cnt: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int, end: Int):
        self.ptr = Int(data.unsafe_ptr())
        self.pos_bytes = start
        self.end_bytes = end
        self.buf = 0
        self.bit_cnt = 0
        self.valid = start >= 0 and end >= start and end <= len(data)

    @always_inline
    def _refill(mut self):
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        while self.bit_cnt <= 56 and self.pos_bytes < self.end_bytes:
            self.buf |= UInt64(p[unsafe_offset=self.pos_bytes]) << UInt64(self.bit_cnt)
            self.pos_bytes += 1
            self.bit_cnt += 8

    @always_inline
    def get_bits(mut self, count: Int) -> Int:
        if count <= 0:
            return 0
        self._refill()
        if self.bit_cnt < count:
            self.valid = False
            return 0
        var v = Int(self.buf & ((UInt64(1) << UInt64(count)) - 1))
        self.buf >>= UInt64(count)
        self.bit_cnt -= count
        return v


struct _BasisHuf(Copyable, Movable):
    var codes: List[Int]
    var lens: List[Int]
    var syms: List[Int]
    var max_len: Int
    var valid: Bool

    def __init__(out self):
        self.codes = List[Int]()
        self.lens = List[Int]()
        self.syms = List[Int]()
        self.max_len = 0
        self.valid = False

    def read_from(mut self, mut br: _BasisLzBits) -> Bool:
        var num_syms = br.get_bits(14)
        var codelen_bits = br.get_bits(5)
        if not br.valid or num_syms <= 0 or codelen_bits < 1 or codelen_bits > 16:
            return False
        var sym_lens = List[Int](length=num_syms, fill=0)
        var counts = List[Int](length=17, fill=0)
        var max_l = 0
        var kraft = 0
        for s in range(num_syms):
            var l = br.get_bits(codelen_bits)
            if not br.valid or l > 16:
                return False
            sym_lens[s] = l
            if l > 0:
                kraft += 1 << (16 - l)
                if kraft > 65536:
                    return False
                counts[l] += 1
                if l > max_l:
                    max_l = l
        var next_code = List[Int](length=17, fill=0)
        var code = 0
        for l in range(1, 17):
            code = (code + counts[l - 1]) << 1
            next_code[l] = code
        self.codes.clear(); self.lens.clear(); self.syms.clear()
        for s in range(num_syms):
            var l = sym_lens[s]
            if l > 0:
                var c = next_code[l]
                next_code[l] += 1
                if c >= (1 << l):
                    return False
                # Reverse `l` bits for LSB-first stream matching
                var rc = 0
                for b in range(l):
                    if ((c >> b) & 1) != 0:
                        rc |= 1 << (l - 1 - b)
                self.codes.append(rc)
                self.lens.append(l)
                self.syms.append(s)
        self.max_len = max_l
        self.valid = len(self.syms) > 0
        return self.valid

    def decode(self, mut br: _BasisLzBits) -> Int:
        br._refill()
        for i in range(len(self.syms)):
            var l = self.lens[i]
            if br.bit_cnt >= l and Int(br.buf & ((UInt64(1) << UInt64(l)) - 1)) == self.codes[i]:
                br.buf >>= UInt64(l)
                br.bit_cnt -= l
                return self.syms[i]
        br.valid = False
        return -1


def _decode_etc1s_block(color5: Int, inten_idx: Int, sel_u32: UInt32, mut out: List[UInt32], bx: Int, by: Int, width: Int, height: Int, alpha_val: Int = 255):
    var inten_table: List[List[Int]] = [
        [2, 8, -2, -8],
        [5, 17, -5, -17],
        [9, 29, -9, -29],
        [13, 42, -13, -42],
        [18, 60, -18, -60],
        [24, 80, -24, -80],
        [33, 106, -33, -106],
        [47, 183, -47, -183],
    ]
    var r5 = (color5 >> 10) & 0x1F
    var g5 = (color5 >> 5) & 0x1F
    var b5 = color5 & 0x1F
    var br = (r5 << 3) | (r5 >> 2)
    var bg = (g5 << 3) | (g5 >> 2)
    var bb = (b5 << 3) | (b5 >> 2)
    ref mods = inten_table[inten_idx & 7]
    for y in range(4):
        var py = by * 4 + y
        for x in range(4):
            var px = bx * 4 + x
            if px < width and py < height:
                var t = y * 4 + x
                var sel = Int((sel_u32 >> UInt32(t * 2)) & 3)
                var d = mods[sel]
                var r = max(0, min(255, br + d))
                var g = max(0, min(255, bg + d))
                var b = max(0, min(255, bb + d))
                out[py * width + px] = (UInt32(alpha_val) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)


def _decode_basislz_mip(
    data: List[UInt8],
    sgd_off: Int,
    sgd_len: Int,
    mip_level: Int,
    level_count: Int,
    slice_off: Int,
    slice_len: Int,
    width: Int,
    height: Int,
) -> Tuple[Bool, List[UInt32]]:
    """Decodes a KTX2 BasisLZ/ETC1S mip level using the SGD global endpoints/selectors/tables."""
    var empty = List[UInt32]()
    if width <= 0 or height <= 0 or width > 16384 or height > 16384 or width > 40_000_000 // height:
        return (False, empty^)
    if level_count <= 0 or level_count > 16 or mip_level < 0 or mip_level >= level_count:
        return (False, empty^)
    var hdr_base = 20 + level_count * 20
    if sgd_off < 0 or sgd_off > len(data) or sgd_len < hdr_base or sgd_len > len(data) - sgd_off:
        return (False, empty^)
    if slice_off < 0 or slice_len <= 0 or slice_off > len(data) or slice_len > len(data) - slice_off:
        return (False, empty^)
    var endpoint_cnt = _u16_le(data, sgd_off)
    var selector_cnt = _u16_le(data, sgd_off + 2)
    var ep_bytes = _u32_le(data, sgd_off + 4)
    var sel_bytes = _u32_le(data, sgd_off + 8)
    var tbl_bytes = _u32_le(data, sgd_off + 12)
    var ext_bytes = _u32_le(data, sgd_off + 16)
    if endpoint_cnt <= 0 or selector_cnt <= 0 or ep_bytes < 0 or sel_bytes < 0 or tbl_bytes < 0 or ext_bytes < 0:
        return (False, empty^)
    var rem_sgd = sgd_len - hdr_base
    if (
        ep_bytes > rem_sgd
        or sel_bytes > rem_sgd - ep_bytes
        or tbl_bytes > rem_sgd - ep_bytes - sel_bytes
        or ext_bytes > rem_sgd - ep_bytes - sel_bytes - tbl_bytes
    ):
        return (False, empty^)
    var hdr_end = sgd_off + hdr_base

    var desc_off = sgd_off + 20 + mip_level * 20
    var rgb_rel_off = _u32_le(data, desc_off + 4)
    var rgb_len = _u32_le(data, desc_off + 8)
    if rgb_rel_off < 0 or rgb_len <= 0 or rgb_rel_off > slice_len or rgb_len > slice_len - rgb_rel_off:
        return (False, empty^)

    var ep_off = hdr_end
    var sel_off = ep_off + ep_bytes
    var tbl_off = sel_off + sel_bytes

    var bw = max(1, (width + 3) // 4)
    var bh = max(1, (height + 3) // 4)
    var num_blocks = bw * bh

    # Raw/uncompressed BasisLZ fallback if tbl_bytes == 0 (compact synthetic fixtures) or Huffman-coded BasisLZ
    if tbl_bytes == 0:
        if ep_bytes < endpoint_cnt * 3 or sel_bytes < selector_cnt * 4:
            return (False, empty^)
        var ep_bits = 1
        while (1 << ep_bits) < endpoint_cnt:
            ep_bits += 1
        var sel_bits = 1
        while (1 << sel_bits) < selector_cnt:
            sel_bits += 1
        if rgb_len < (num_blocks * (ep_bits + sel_bits) + 7) // 8:
            return (False, empty^)

        var endpoints_c5 = List[Int](length=endpoint_cnt, fill=0)
        var endpoints_i = List[Int](length=endpoint_cnt, fill=0)
        var selectors = List[UInt32](length=selector_cnt, fill=0)
        for ei in range(endpoint_cnt):
            var eo = ep_off + ei * 3
            endpoints_c5[ei] = _u16_le(data, eo) & 0x7FFF
            endpoints_i[ei] = Int(data[eo + 2]) & 7
        for si in range(selector_cnt):
            selectors[si] = _u32_le_u(data, sel_off + si * 4)

        var sbr = _BasisLzBits(data, slice_off + rgb_rel_off, slice_off + rgb_rel_off + rgb_len)
        var out = List[UInt32](length=width * height, fill=0xFF000000)
        for by in range(bh):
            for bx in range(bw):
                var ei = sbr.get_bits(ep_bits)
                var si = sbr.get_bits(sel_bits)
                if not sbr.valid or ei >= endpoint_cnt or si >= selector_cnt:
                    return (False, empty^)
                _decode_etc1s_block(endpoints_c5[ei], endpoints_i[ei], selectors[si], out, bx, by, width, height)
        return (True, out^)

    # Standard BasisLZ Huffman tables — validate all tables/codebooks & slice budget BEFORE allocating pixels
    if rgb_len < (num_blocks * 2 + 7) // 8:
        return (False, empty^)
    var tbr = _BasisLzBits(data, tbl_off, tbl_off + tbl_bytes)
    var ep_huf = _BasisHuf()
    var sel_huf = _BasisHuf()
    if not ep_huf.read_from(tbr) or not sel_huf.read_from(tbr):
        return (False, empty^)

    var endpoints_c5 = List[Int](length=endpoint_cnt, fill=0)
    var endpoints_i = List[Int](length=endpoint_cnt, fill=0)
    var ebr = _BasisLzBits(data, ep_off, ep_off + ep_bytes)
    var prev_r = 0; var prev_g = 0; var prev_b = 0; var prev_i = 0
    for ei in range(endpoint_cnt):
        var dr = ep_huf.decode(ebr)
        var dg = ep_huf.decode(ebr)
        var db = ep_huf.decode(ebr)
        var di = ep_huf.decode(ebr)
        if not ebr.valid or dr < 0 or dg < 0 or db < 0 or di < 0:
            return (False, empty^)
        prev_r = (prev_r + dr) & 31
        prev_g = (prev_g + dg) & 31
        prev_b = (prev_b + db) & 31
        prev_i = (prev_i + di) & 7
        endpoints_c5[ei] = (prev_r << 10) | (prev_g << 5) | prev_b
        endpoints_i[ei] = prev_i

    var selectors = List[UInt32](length=selector_cnt, fill=0)
    var slbr = _BasisLzBits(data, sel_off, sel_off + sel_bytes)
    for si in range(selector_cnt):
        var s_lo = sel_huf.decode(slbr)
        var s_hi = sel_huf.decode(slbr)
        if not slbr.valid or s_lo < 0 or s_hi < 0:
            return (False, empty^)
        selectors[si] = UInt32(s_lo & 0xFFFF) | (UInt32(s_hi & 0xFFFF) << 16)

    var sbr = _BasisLzBits(data, slice_off + rgb_rel_off, slice_off + rgb_rel_off + rgb_len)
    var out = List[UInt32](length=width * height, fill=0xFF000000)
    var ep_idx = 0
    var sel_idx = 0
    for by in range(bh):
        for bx in range(bw):
            var de = ep_huf.decode(sbr)
            var ds = sel_huf.decode(sbr)
            if not sbr.valid or de < 0 or ds < 0:
                return (False, empty^)
            ep_idx = (ep_idx + de) % endpoint_cnt
            sel_idx = (sel_idx + ds) % selector_cnt
            _decode_etc1s_block(endpoints_c5[ep_idx], endpoints_i[ep_idx], selectors[sel_idx], out, bx, by, width, height)
    return (True, out^)


# ============================================================================
# 5. Public KTX2 Decoder Entry Point
# ============================================================================

def _downsample_2x_ktx2(pixels: List[UInt32], w: Int, h: Int) -> Tuple[List[UInt32], Int, Int]:
    var nw = max(1, w >> 1)
    var nh = max(1, h >> 1)
    var out = List[UInt32](length=nw * nh, fill=0)
    for y in range(nh):
        var sy0 = min(h - 1, y * 2)
        var sy1 = min(h - 1, y * 2 + 1)
        for x in range(nw):
            var sx0 = min(w - 1, x * 2)
            var sx1 = min(w - 1, x * 2 + 1)
            var p00 = pixels[sy0 * w + sx0]
            var p10 = pixels[sy0 * w + sx1]
            var p01 = pixels[sy1 * w + sx0]
            var p11 = pixels[sy1 * w + sx1]
            var a = (((p00 >> 24) & 0xFF) + ((p10 >> 24) & 0xFF) + ((p01 >> 24) & 0xFF) + ((p11 >> 24) & 0xFF) + 2) >> 2
            var r = (((p00 >> 16) & 0xFF) + ((p10 >> 16) & 0xFF) + ((p01 >> 16) & 0xFF) + ((p11 >> 16) & 0xFF) + 2) >> 2
            var g = (((p00 >> 8) & 0xFF) + ((p10 >> 8) & 0xFF) + ((p01 >> 8) & 0xFF) + ((p11 >> 8) & 0xFF) + 2) >> 2
            var b = ((p00 & 0xFF) + (p10 & 0xFF) + (p01 & 0xFF) + (p11 & 0xFF) + 2) >> 2
            out[y * nw + x] = (a << 24) | (r << 16) | (g << 8) | b
    return (out^, nw, nh)


def _ktx2_expected_mip_bytes(mw: Int, mh: Int, vk_format: Int, dfd_color_model: Int) -> Int:
    var bw = max(1, (mw + 3) // 4)
    var bh = max(1, (mh + 3) // 4)
    var blocks_n = bw * bh
    var bytes = -1
    if dfd_color_model == DFD_MODEL_UASTC and vk_format == 0:
        bytes = blocks_n * 16
    elif vk_format == 0:
        return -1
    elif (
        vk_format == VK_FORMAT_R8G8B8A8_UNORM
        or vk_format == VK_FORMAT_R8G8B8A8_SRGB
        or vk_format == VK_FORMAT_B8G8R8A8_UNORM
        or vk_format == VK_FORMAT_B8G8R8A8_SRGB
    ):
        bytes = mw * mh * 4
    elif (
        vk_format == VK_FORMAT_R8G8B8_UNORM
        or vk_format == VK_FORMAT_R8G8B8_SRGB
        or vk_format == VK_FORMAT_B8G8R8_UNORM
        or vk_format == VK_FORMAT_B8G8R8_SRGB
    ):
        bytes = mw * mh * 3
    elif vk_format == VK_FORMAT_R8G8_UNORM or vk_format == VK_FORMAT_R8G8_SRGB:
        bytes = mw * mh * 2
    elif vk_format == VK_FORMAT_R8_UNORM or vk_format == VK_FORMAT_R8_SRGB:
        bytes = mw * mh
    elif vk_format == VK_FORMAT_R16G16B16A16_SFLOAT:
        bytes = mw * mh * 8
    elif vk_format == VK_FORMAT_R32G32B32A32_SFLOAT:
        bytes = mw * mh * 16
    elif vk_format >= VK_FORMAT_BC1_RGB_UNORM and vk_format <= VK_FORMAT_BC1_RGBA_SRGB:
        bytes = blocks_n * 8
    elif (
        vk_format == VK_FORMAT_BC2_UNORM
        or vk_format == VK_FORMAT_BC2_SRGB
        or vk_format == VK_FORMAT_BC3_UNORM
        or vk_format == VK_FORMAT_BC3_SRGB
        or vk_format == VK_FORMAT_BC5_UNORM
        or vk_format == VK_FORMAT_BC5_SNORM
        or vk_format == VK_FORMAT_BC7_UNORM
        or vk_format == VK_FORMAT_BC7_SRGB
    ):
        bytes = blocks_n * 16
    elif vk_format == VK_FORMAT_BC4_UNORM or vk_format == VK_FORMAT_BC4_SNORM:
        bytes = blocks_n * 8
    if bytes <= 0 or bytes > 268_435_456:
        return -1
    return bytes


def decode_ktx2(data: List[UInt8], maximum_size: Int = 4096) -> NativeImage:
    """Decodes a KTX2 container (None/BasisLZ/Zstd/Zlib, Uncompressed/BC1-BC7/UASTC/ETC1S) to `NativeImage` (`0xAARRGGBB`)."""
    var img = NativeImage()
    if len(data) < 80 or not is_ktx2_magic(data):
        return img^

    var vk_format = _u32_le(data, 12)
    var pixel_width = _u32_le(data, 20)
    var pixel_height = _u32_le(data, 24)
    var pixel_depth = _u32_le(data, 28)
    var level_count = max(1, _u32_le(data, 40))
    var supercompression = _u32_le(data, 44)
    var dfd_off = _u32_le(data, 48)
    var dfd_len = _u32_le(data, 52)
    var sgd_off_u64 = _u64_le(data, 64)
    var sgd_len_u64 = _u64_le(data, 72)
    if sgd_off_u64 > UInt64(len(data)) or sgd_len_u64 > UInt64(len(data)) - sgd_off_u64:
        return img^
    var sgd_off = Int(sgd_off_u64)
    var sgd_len = Int(sgd_len_u64)

    if pixel_width <= 0 or pixel_height <= 0 or pixel_width > 16384 or pixel_height > 16384:
        return img^
    if pixel_depth > 1 or level_count > 16:
        return img^
    if 80 + level_count * 24 > len(data):
        return img^

    # Parse DFD colorModel if present (Khronos KTX2 / DFD v1.3: +0..3 totalSize, +4..7 vendorId/descriptorType,
    # +8..11 versionNumber/descriptorBlockSize, +12 colorModel)
    var dfd_color_model = 0
    if dfd_len >= 16 and dfd_off >= 80 and dfd_off + dfd_len <= len(data) and dfd_off + 16 <= len(data):
        dfd_color_model = Int(data[dfd_off + 12])

    # Select finest mip <= maximum_size
    var chosen_mip = 0
    while chosen_mip + 1 < level_count:
        var mw = max(1, pixel_width >> chosen_mip)
        var mh = max(1, pixel_height >> chosen_mip)
        if mw <= maximum_size and mh <= maximum_size:
            break
        chosen_mip += 1

    var lvl_idx_off = 80 + chosen_mip * 24
    var byte_offset_u64 = _u64_le(data, lvl_idx_off)
    var byte_length_u64 = _u64_le(data, lvl_idx_off + 8)
    var uncomp_len_u64 = _u64_le(data, lvl_idx_off + 16)
    if byte_offset_u64 > UInt64(len(data)) or byte_length_u64 > UInt64(len(data)) - byte_offset_u64:
        return img^
    if byte_length_u64 == 0 or uncomp_len_u64 > 268_435_456:
        return img^

    var mip_off = Int(byte_offset_u64)
    var mip_len = Int(byte_length_u64)
    var uncomp_len = Int(uncomp_len_u64)
    var mw = max(1, pixel_width >> chosen_mip)
    var mh = max(1, pixel_height >> chosen_mip)
    if mw <= 0 or mh <= 0 or mw > 40_000_000 // mh:
        return img^
    if level_count == 1 and maximum_size > 0 and (mw > max(maximum_size, 4096) or mh > max(maximum_size, 4096)):
        return img^

    var pixels: List[UInt32]
    if supercompression == KTX2_SC_BASIS_LZ or dfd_color_model == DFD_MODEL_ETC1S:
        var blz = _decode_basislz_mip(data, sgd_off, sgd_len, chosen_mip, level_count, mip_off, mip_len, mw, mh)
        if not blz[0]:
            return img^
        pixels = blz[1].copy()
    else:
        var expected_mip_bytes = _ktx2_expected_mip_bytes(mw, mh, vk_format, dfd_color_model)
        if expected_mip_bytes <= 0:
            return img^
        var payload = List[UInt8]()
        if supercompression == KTX2_SC_NONE:
            payload.reserve(mip_len)
            for i in range(mip_len):
                payload.append(data[mip_off + i])
        elif supercompression == KTX2_SC_ZLIB:
            if uncomp_len <= 0 or uncomp_len != expected_mip_bytes:
                return img^
            payload = inflate_zlib(data, mip_off, uncomp_len, mip_len)
            if len(payload) != uncomp_len:
                return img^
        elif supercompression == KTX2_SC_ZSTD:
            if uncomp_len <= 0 or uncomp_len != expected_mip_bytes:
                return img^
            var zres = decompress_zstd(data, mip_off, mip_len, uncomp_len)
            if not zres[0] or len(zres[1]) != uncomp_len:
                return img^
            payload = zres[1].copy()
        else:
            return img^

        var bw = max(1, (mw + 3) // 4)
        var bh = max(1, (mh + 3) // 4)
        var blocks_n = bw * bh

        if dfd_color_model == DFD_MODEL_UASTC and vk_format == 0:
            if len(payload) < blocks_n * 16:
                return img^
            var ures = _decode_uastc_blocks(payload, mw, mh)
            if not ures[0]:
                return img^
            pixels = ures[1].copy()
        elif vk_format == 0:
            return img^
        elif vk_format == VK_FORMAT_R8G8B8A8_UNORM or vk_format == VK_FORMAT_R8G8B8A8_SRGB:
            if len(payload) < mw * mh * 4:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 4
                pixels.append((UInt32(payload[o + 3]) << 24) | (UInt32(payload[o]) << 16) | (UInt32(payload[o + 1]) << 8) | UInt32(payload[o + 2]))
        elif vk_format == VK_FORMAT_B8G8R8A8_UNORM or vk_format == VK_FORMAT_B8G8R8A8_SRGB:
            if len(payload) < mw * mh * 4:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 4
                pixels.append((UInt32(payload[o + 3]) << 24) | (UInt32(payload[o + 2]) << 16) | (UInt32(payload[o + 1]) << 8) | UInt32(payload[o]))
        elif vk_format == VK_FORMAT_R8G8B8_UNORM or vk_format == VK_FORMAT_R8G8B8_SRGB:
            if len(payload) < mw * mh * 3:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 3
                pixels.append(UInt32(0xFF000000) | (UInt32(payload[o]) << 16) | (UInt32(payload[o + 1]) << 8) | UInt32(payload[o + 2]))
        elif vk_format == VK_FORMAT_B8G8R8_UNORM or vk_format == VK_FORMAT_B8G8R8_SRGB:
            if len(payload) < mw * mh * 3:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 3
                pixels.append(UInt32(0xFF000000) | (UInt32(payload[o + 2]) << 16) | (UInt32(payload[o + 1]) << 8) | UInt32(payload[o]))
        elif vk_format == VK_FORMAT_R8G8_UNORM or vk_format == VK_FORMAT_R8G8_SRGB:
            if len(payload) < mw * mh * 2:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 2
                pixels.append(UInt32(0xFF000000) | (UInt32(payload[o]) << 16) | (UInt32(payload[o + 1]) << 8))
        elif vk_format == VK_FORMAT_R8_UNORM or vk_format == VK_FORMAT_R8_SRGB:
            if len(payload) < mw * mh:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var g = UInt32(payload[i])
                pixels.append(UInt32(0xFF000000) | (g << 16) | (g << 8) | g)
        elif vk_format == VK_FORMAT_R16G16B16A16_SFLOAT:
            if len(payload) < mw * mh * 8:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 8
                var r = _f32_unit_to_u8(_sf16_bits_to_f32(_u16_le(payload, o)))
                var g = _f32_unit_to_u8(_sf16_bits_to_f32(_u16_le(payload, o + 2)))
                var b = _f32_unit_to_u8(_sf16_bits_to_f32(_u16_le(payload, o + 4)))
                var a = _f32_unit_to_u8(_sf16_bits_to_f32(_u16_le(payload, o + 6)))
                pixels.append((a << 24) | (r << 16) | (g << 8) | b)
        elif vk_format == VK_FORMAT_R32G32B32A32_SFLOAT:
            if len(payload) < mw * mh * 16:
                return img^
            pixels = List[UInt32](capacity=mw * mh)
            for i in range(mw * mh):
                var o = i * 16
                var r = _f32_unit_to_u8(bitcast[DType.float32](_u32_le_u(payload, o)))
                var g = _f32_unit_to_u8(bitcast[DType.float32](_u32_le_u(payload, o + 4)))
                var b = _f32_unit_to_u8(bitcast[DType.float32](_u32_le_u(payload, o + 8)))
                var a = _f32_unit_to_u8(bitcast[DType.float32](_u32_le_u(payload, o + 12)))
                pixels.append((a << 24) | (r << 16) | (g << 8) | b)
        elif vk_format >= VK_FORMAT_BC1_RGB_UNORM and vk_format <= VK_FORMAT_BC1_RGBA_SRGB:
            if len(payload) < blocks_n * 8:
                return img^
            var allow_a = vk_format == VK_FORMAT_BC1_RGBA_UNORM or vk_format == VK_FORMAT_BC1_RGBA_SRGB
            pixels = List[UInt32](length=mw * mh, fill=0xFF000000)
            for by in range(bh):
                for bx in range(bw):
                    _decode_bc1_block(payload, (by * bw + bx) * 8, allow_a, pixels, bx, by, mw, mh)
        elif vk_format == VK_FORMAT_BC2_UNORM or vk_format == VK_FORMAT_BC2_SRGB:
            if len(payload) < blocks_n * 16:
                return img^
            pixels = List[UInt32](length=mw * mh, fill=0xFF000000)
            for by in range(bh):
                for bx in range(bw):
                    var boff = (by * bw + bx) * 16
                    _decode_bc1_block(payload, boff + 8, False, pixels, bx, by, mw, mh)
                    var alpha64 = _u64_le(payload, boff)
                    for t in range(16):
                        var px = bx * 4 + (t & 3); var py = by * 4 + (t >> 2)
                        if px < mw and py < mh:
                            var a4 = Int((alpha64 >> UInt64(t * 4)) & 0xF)
                            var a8 = UInt32((a4 << 4) | a4)
                            pixels[py * mw + px] = (pixels[py * mw + px] & 0x00FFFFFF) | (a8 << 24)
        elif vk_format == VK_FORMAT_BC3_UNORM or vk_format == VK_FORMAT_BC3_SRGB:
            if len(payload) < blocks_n * 16:
                return img^
            pixels = List[UInt32](length=mw * mh, fill=0xFF000000)
            for by in range(bh):
                for bx in range(bw):
                    var boff = (by * bw + bx) * 16
                    _decode_bc1_block(payload, boff + 8, False, pixels, bx, by, mw, mh)
                    var av = _decode_bc4_channel(payload, boff)
                    for t in range(16):
                        var px = bx * 4 + (t & 3); var py = by * 4 + (t >> 2)
                        if px < mw and py < mh:
                            pixels[py * mw + px] = (pixels[py * mw + px] & 0x00FFFFFF) | (UInt32(av[t]) << 24)
        elif vk_format == VK_FORMAT_BC4_UNORM or vk_format == VK_FORMAT_BC4_SNORM:
            if len(payload) < blocks_n * 8:
                return img^
            pixels = List[UInt32](length=mw * mh, fill=0xFF000000)
            for by in range(bh):
                for bx in range(bw):
                    var rv = _decode_bc4_channel(payload, (by * bw + bx) * 8)
                    for t in range(16):
                        var px = bx * 4 + (t & 3); var py = by * 4 + (t >> 2)
                        if px < mw and py < mh:
                            var c = UInt32(rv[t])
                            pixels[py * mw + px] = UInt32(0xFF000000) | (c << 16) | (c << 8) | c
        elif vk_format == VK_FORMAT_BC5_UNORM or vk_format == VK_FORMAT_BC5_SNORM:
            if len(payload) < blocks_n * 16:
                return img^
            pixels = List[UInt32](length=mw * mh, fill=0xFF000000)
            for by in range(bh):
                for bx in range(bw):
                    var boff = (by * bw + bx) * 16
                    var rx = _decode_bc4_channel(payload, boff)
                    var ry = _decode_bc4_channel(payload, boff + 8)
                    for t in range(16):
                        var px = bx * 4 + (t & 3); var py = by * 4 + (t >> 2)
                        if px < mw and py < mh:
                            var nx = (Float32(rx[t]) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                            var ny = (Float32(ry[t]) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                            var nz = sqrt(max(Float32(0.0), Float32(1.0) - nx * nx - ny * ny))
                            var bz = UInt32(max(0, min(255, Int((nz * Float32(0.5) + Float32(0.5)) * Float32(255.0) + Float32(0.5)))))
                            pixels[py * mw + px] = UInt32(0xFF000000) | (UInt32(rx[t]) << 16) | (UInt32(ry[t]) << 8) | bz
        elif vk_format == VK_FORMAT_BC7_UNORM or vk_format == VK_FORMAT_BC7_SRGB:
            if len(payload) < blocks_n * 16:
                return img^
            pixels = _decode_bc7_full(payload, mw, mh)
        else:
            return img^

    while (mw > maximum_size or mh > maximum_size) and mw > 1 and mh > 1:
        var ds = _downsample_2x_ktx2(pixels, mw, mh)
        pixels = ds[0].copy()
        mw = ds[1]
        mh = ds[2]

    img.valid = True
    img.width = mw
    img.height = mh
    img.pixels = pixels^
    return img^
