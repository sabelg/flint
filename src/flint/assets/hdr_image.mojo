"""Phase 2 (`V17`): Pure-Mojo Radiance `.hdr` (`RGBE`) & OpenEXR `.exr` (`FP16`/`FP32`) Codec (`hdr_image.mojo`).

Replaces external C/C++ `OpenEXR`, `tinyexr`, and `stb_image` HDR loaders with 100%
pure Mojo 1.2 encoders and decoders for:
  1. Greg Ward Radiance `.hdr` (`#?RADIANCE` / `#?RGBE`, adaptive 4-channel RLE & flat scanlines).
  2. ILM OpenEXR v2 `.exr` (`magic = 0x01312F76`, `HALF` 16-bit & `FLOAT` 32-bit channels,
     `NO_COMPRESSION = 0`, `ZIPS_COMPRESSION = 2`, and `ZIP_COMPRESSION = 3` with OpenEXR
     predictor + even/odd byte-interleave reconstruction).
  3. Equirectangular `2:1` HDRI panorama bilinear sampler feeding `aaa_ibl.mojo` (`BC6H` + `SH9`).
"""
from std.math import atan2, cos, exp, exp2, floor, log, log2, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.assets.bc_compress import f32_to_uf16_bits, uf16_bits_to_f32
from flint.assets.image import inflate_zlib
from flint.core.detmath import det_acos


struct HdrImage(Copyable, Movable):
    var valid: Bool
    var width: Int
    var height: Int
    var pixels: List[Vec3]

    def __init__(out self):
        self.valid = False
        self.width = 0
        self.height = 0
        self.pixels = List[Vec3]()

    def sample_equirect(self, dir_unit: Vec3) -> Vec3:
        """Samples a `2:1` latitude-longitude HDRI environment map at unit world direction `dir_unit`."""
        if not self.valid or self.width <= 0 or self.height <= 0:
            return Vec3(0.0, 0.0, 0.0)
        var d = dir_unit.normalized()
        var phi = atan2(d.z, d.x)  # [-pi .. +pi]
        var theta = det_acos(max(Float32(-1.0), min(Float32(1.0), d.y)))  # [0 (zenith) .. pi (nadir)]
        var u = phi * Float32(0.159154943) + Float32(0.5)
        u = u - floor(u)
        var v = max(Float32(0.0), min(Float32(1.0), theta * Float32(0.318309886)))

        var fx = u * Float32(self.width) - Float32(0.5)
        var fy = v * Float32(self.height) - Float32(0.5)
        var x0_raw = Int(floor(fx))
        var y0 = max(0, min(self.height - 1, Int(floor(fy))))
        var y1 = max(0, min(self.height - 1, y0 + 1))
        var tx = max(Float32(0.0), min(Float32(1.0), fx - Float32(x0_raw)))
        var ty = max(Float32(0.0), min(Float32(1.0), fy - Float32(Int(floor(fy)))))

        var x0 = x0_raw % self.width
        if x0 < 0:
            x0 += self.width
        var x1 = (x0 + 1) % self.width

        var c00 = self.pixels[y0 * self.width + x0]
        var c10 = self.pixels[y0 * self.width + x1]
        var c01 = self.pixels[y1 * self.width + x0]
        var c11 = self.pixels[y1 * self.width + x1]
        var top = c00 * (Float32(1.0) - tx) + c10 * tx
        var bot = c01 * (Float32(1.0) - tx) + c11 * tx
        return top * (Float32(1.0) - ty) + bot * ty


# ============================================================================
# IEEE-754 Signed Half-Float (`F16`) Helpers for OpenEXR
# ============================================================================

@always_inline
def _f16_signed_to_f32(h: Int) -> Float32:
    var sign = Float32(-1.0) if ((h & 0x8000) != 0) else Float32(1.0)
    return sign * uf16_bits_to_f32(h & 0x7FFF)


@always_inline
def _f32_to_f16_signed(v: Float32) -> Int:
    if v < Float32(0.0):
        return 0x8000 | f32_to_uf16_bits(-v)
    return f32_to_uf16_bits(v)


@always_inline
def _le_i32(data: List[UInt8], off: Int) -> Int:
    var u = (
        UInt32(data[off])
        | (UInt32(data[off + 1]) << 8)
        | (UInt32(data[off + 2]) << 16)
        | (UInt32(data[off + 3]) << 24)
    )
    return Int(bitcast[DType.int32](u))


@always_inline
def _le_u64(data: List[UInt8], off: Int) -> Int:
    var u: UInt64 = 0
    for b in range(8):
        u |= UInt64(data[off + b]) << UInt64(b * 8)
    return Int(u)


@always_inline
def _put_le_u32(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


@always_inline
def _put_le_u64(mut out: List[UInt8], v: UInt64):
    for b in range(8):
        out.append(UInt8((v >> UInt64(b * 8)) & 0xFF))


def _put_cstr(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])
    out.append(0)


def _read_cstr(data: List[UInt8], start: Int, max_end: Int) -> Tuple[String, Int]:
    var bytes = List[UInt8]()
    var p = start
    while p < max_end:
        var ch = data[p]
        p += 1
        if ch == 0:
            return (String(unsafe_from_utf8=bytes), p)
        bytes.append(ch)
    return (String(unsafe_from_utf8=bytes), p)


# ============================================================================
# 1. Radiance `.hdr` (`RGBE`) Encoder & Decoder
# ============================================================================

@always_inline
def _float3_to_rgbe(c: Vec3) -> Tuple[UInt8, UInt8, UInt8, UInt8]:
    var r = max(Float32(0.0), c.x)
    var g = max(Float32(0.0), c.y)
    var b = max(Float32(0.0), c.z)
    var v = max(r, max(g, b))
    if v < Float32(1e-32):
        return (UInt8(0), UInt8(0), UInt8(0), UInt8(0))
    var exp_i = Int(floor(log2(v))) + 1
    if exp_i <= -128:
        return (UInt8(0), UInt8(0), UInt8(0), UInt8(0))
    if exp_i > 127:
        exp_i = 127
    var scale = Float32(256.0) / exp2(Float32(exp_i))
    var rb = UInt8(max(0, min(255, Int(r * scale))))
    var gb = UInt8(max(0, min(255, Int(g * scale))))
    var bb = UInt8(max(0, min(255, Int(b * scale))))
    return (rb, gb, bb, UInt8(exp_i + 128))


@always_inline
def _rgbe_to_float3(r: UInt8, g: UInt8, b: UInt8, e: UInt8) -> Vec3:
    if e == 0:
        return Vec3(0.0, 0.0, 0.0)
    var scale = exp2(Float32(Int(e) - 136))
    return Vec3(
        (Float32(r) + Float32(0.5)) * scale,
        (Float32(g) + Float32(0.5)) * scale,
        (Float32(b) + Float32(0.5)) * scale,
    )


def _encode_rle_channel(chan: List[UInt8], mut out: List[UInt8]):
    var n = len(chan)
    var cur = 0
    while cur < n:
        # Check for a run of >= 4 identical bytes
        var run_start = cur
        var run_len = 1
        while run_start + run_len < n and run_len < 127 and chan[run_start + run_len] == chan[run_start]:
            run_len += 1
        if run_len >= 4:
            out.append(UInt8(128 + run_len))
            out.append(chan[run_start])
            cur += run_len
        else:
            # Emit non-run literal span until a 4-byte run appears or 128 bytes reached
            var lit_end = cur + 1
            while lit_end < n and (lit_end - cur) < 128:
                var rl = 1
                while lit_end + rl < n and rl < 4 and chan[lit_end + rl] == chan[lit_end]:
                    rl += 1
                if rl >= 4:
                    break
                lit_end += 1
            var lit_cnt = lit_end - cur
            out.append(UInt8(lit_cnt))
            for k in range(lit_cnt):
                out.append(chan[cur + k])
            cur = lit_end


def encode_radiance_hdr(pixels: List[Vec3], width: Int, height: Int) -> List[UInt8]:
    """Encodes linear HDR `Vec3` pixels into a standard Radiance `.hdr` (`32-bit_rle_rgbe`) byte stream."""
    var out = List[UInt8]()
    var hdr_txt = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " + String(height) + " +X " + String(width) + "\n"
    var hb = hdr_txt.as_bytes()
    for i in range(len(hb)):
        out.append(hb[i])

    var use_rle = width >= 8 and width <= 0x7FFF
    var ch_r = List[UInt8](length=width, fill=0)
    var ch_g = List[UInt8](length=width, fill=0)
    var ch_b = List[UInt8](length=width, fill=0)
    var ch_e = List[UInt8](length=width, fill=0)

    for y in range(height):
        if not use_rle:
            for x in range(width):
                var q = _float3_to_rgbe(pixels[y * width + x])
                out.append(q[0]); out.append(q[1]); out.append(q[2]); out.append(q[3])
        else:
            out.append(2)
            out.append(2)
            out.append(UInt8((width >> 8) & 0xFF))
            out.append(UInt8(width & 0xFF))
            for x in range(width):
                var q = _float3_to_rgbe(pixels[y * width + x])
                ch_r[x] = q[0]; ch_g[x] = q[1]; ch_b[x] = q[2]; ch_e[x] = q[3]
            _encode_rle_channel(ch_r, out)
            _encode_rle_channel(ch_g, out)
            _encode_rle_channel(ch_b, out)
            _encode_rle_channel(ch_e, out)

    return out^


def _parse_int_decimal(s: String) -> Int:
    var v = 0
    var has_d = False
    var sb = s.as_bytes()
    for i in range(len(sb)):
        var b = Int(sb[i])
        if b >= 48 and b <= 57:
            v = v * 10 + (b - 48)
            has_d = True
    return v if has_d else -1


def decode_radiance_hdr(data: List[UInt8]) -> HdrImage:
    """Decodes Radiance `.hdr` (`#?RADIANCE` / `#?RGBE`) adaptive RLE & flat scanline HDRI files."""
    var img = HdrImage()
    var n = len(data)
    if n < 16 or data[0] != 35 or data[1] != 63:  # "#?"
        return img^

    # 1. Advance to empty line `\n\n` ending text header
    var pos = 0
    var found_blank = False
    while pos + 1 < n:
        if data[pos] == 10 and data[pos + 1] == 10:
            pos += 2
            found_blank = True
            break
        if pos + 3 < n and data[pos] == 13 and data[pos + 1] == 10 and data[pos + 2] == 13 and data[pos + 3] == 10:
            pos += 4
            found_blank = True
            break
        pos += 1
    if not found_blank or pos >= n:
        return img^

    # 2. Parse resolution line (`-Y <H> +X <W>\n`)
    var res_bytes = List[UInt8]()
    while pos < n and data[pos] != 10:
        if data[pos] != 13:
            res_bytes.append(data[pos])
        pos += 1
    if pos >= n:
        return img^
    pos += 1  # skip '\n'

    var tokens = List[String]()
    var cur_tok = List[UInt8]()
    for b in res_bytes:
        if b == 32 or b == 9:
            if len(cur_tok) > 0:
                tokens.append(String(unsafe_from_utf8=cur_tok))
                cur_tok.clear()
        else:
            cur_tok.append(b)
    if len(cur_tok) > 0:
        tokens.append(String(unsafe_from_utf8=cur_tok))

    if len(tokens) < 4:
        return img^
    var width = 0
    var height = 0
    for k in range(0, 3, 2):
        var axis_tok = tokens[k]
        var val = _parse_int_decimal(tokens[k + 1])
        var ab = axis_tok.as_bytes()
        if len(ab) >= 2 and (ab[1] == 89 or ab[1] == 121):  # 'Y' / 'y'
            height = val
        elif len(ab) >= 2 and (ab[1] == 88 or ab[1] == 120):  # 'X' / 'x'
            width = val
    if width <= 0 or height <= 0 or width * height > 33_554_432:
        return img^

    img.width = width
    img.height = height
    img.pixels = List[Vec3](length=width * height, fill=Vec3(0.0, 0.0, 0.0))

    var scan_ch = List[UInt8](length=width * 4, fill=0)

    for y in range(height):
        if pos + 4 > n:
            return HdrImage()
        var b0 = Int(data[pos])
        var b1 = Int(data[pos + 1])
        var b2 = Int(data[pos + 2])
        var b3 = Int(data[pos + 3])
        if b0 == 2 and b1 == 2 and ((b2 << 8) | b3) == width and width >= 8 and width <= 0x7FFF:
            pos += 4
            for ch in range(4):
                var x = 0
                while x < width:
                    if pos >= n:
                        return HdrImage()
                    var code = Int(data[pos])
                    pos += 1
                    if code > 128:
                        var count = code - 128
                        if pos >= n or x + count > width:
                            return HdrImage()
                        var val = data[pos]
                        pos += 1
                        for _ in range(count):
                            scan_ch[ch * width + x] = val
                            x += 1
                    else:
                        if code == 0 or pos + code > n or x + code > width:
                            return HdrImage()
                        for _ in range(code):
                            scan_ch[ch * width + x] = data[pos]
                            pos += 1
                            x += 1
            for x in range(width):
                img.pixels[y * width + x] = _rgbe_to_float3(
                    scan_ch[x],
                    scan_ch[width + x],
                    scan_ch[2 * width + x],
                    scan_ch[3 * width + x],
                )
        else:
            if pos + width * 4 > n:
                return HdrImage()
            for x in range(width):
                img.pixels[y * width + x] = _rgbe_to_float3(
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
                )
                pos += 4

    img.valid = True
    return img^


# ============================================================================
# 2. ILM OpenEXR v2 `.exr` (`FP16` / `FP32`) Encoder & Decoder
# ============================================================================

comptime EXR_MAGIC: Int = 0x01312F76
comptime EXR_PIXEL_UINT: Int = 0
comptime EXR_PIXEL_HALF: Int = 1
comptime EXR_PIXEL_FLOAT: Int = 2

comptime EXR_NO_COMPRESSION: Int = 0
comptime EXR_RLE_COMPRESSION: Int = 1
comptime EXR_ZIPS_COMPRESSION: Int = 2
comptime EXR_ZIP_COMPRESSION: Int = 3
comptime EXR_PIZ_COMPRESSION: Int = 4
comptime EXR_DWAA_COMPRESSION: Int = 8
comptime EXR_DWAB_COMPRESSION: Int = 9


@always_inline
def _le_u16(data: List[UInt8], off: Int) -> Int:
    return Int(data[off]) | (Int(data[off + 1]) << 8)


@always_inline
def _le_i64(data: List[UInt8], off: Int) -> Int:
    var u: UInt64 = 0
    for b in range(8):
        u |= UInt64(data[off + b]) << UInt64(b * 8)
    return Int(bitcast[DType.int64](u))


def _exr_rle_uncompress(data: List[UInt8], start: Int, packed_sz: Int, expected_out: Int) -> Tuple[Bool, List[UInt8]]:
    var empty = List[UInt8]()
    var end = start + packed_sz
    if start < 0 or packed_sz < 0 or end > len(data) or expected_out < 0:
        return (False, empty^)
    var out = List[UInt8](capacity=expected_out)
    var p = start
    while p < end:
        var c = Int(bitcast[DType.int8](data[p]))
        p += 1
        if c < 0:
            var cnt = -c
            if p + cnt > end or len(out) + cnt > expected_out:
                return (False, empty^)
            for i in range(cnt):
                out.append(data[p + i])
            p += cnt
        else:
            var cnt = c + 1
            if p >= end or len(out) + cnt > expected_out:
                return (False, empty^)
            var v = data[p]
            p += 1
            for _ in range(cnt):
                out.append(v)
    if len(out) != expected_out:
        return (False, empty^)
    return (True, out^)


struct _ExrMsbReader(Copyable, Movable):
    var ptr: Int
    var start: Int
    var end: Int
    var byte_pos: Int
    var bit_buf: UInt64
    var bit_cnt: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int, end: Int):
        self.ptr = Int(data.unsafe_ptr())
        self.start = start
        self.end = end
        self.byte_pos = start
        self.bit_buf = 0
        self.bit_cnt = 0
        self.valid = start >= 0 and end >= start and end <= len(data)

    @always_inline
    def _ensure(mut self, need: Int):
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        while self.bit_cnt < need and self.byte_pos < self.end:
            self.bit_buf = (self.bit_buf << 8) | UInt64(p[unsafe_offset=self.byte_pos])
            self.byte_pos += 1
            self.bit_cnt += 8

    @always_inline
    def read_msb(mut self, count: Int) -> Int:
        if count == 0:
            return 0
        if not self.valid or count < 0 or count > 58:
            self.valid = False
            return 0
        self._ensure(count)
        if self.bit_cnt < count:
            self.valid = False
            return 0
        self.bit_cnt -= count
        var mask = (UInt64(1) << UInt64(count)) - 1
        return Int((self.bit_buf >> UInt64(self.bit_cnt)) & mask)

    @always_inline
    def peek_msb(mut self, count: Int) -> Int:
        self._ensure(count)
        if self.bit_cnt >= count:
            var mask = (UInt64(1) << UInt64(count)) - 1
            return Int((self.bit_buf >> UInt64(self.bit_cnt - count)) & mask)
        if self.bit_cnt == 0:
            return 0
        var mask = (UInt64(1) << UInt64(self.bit_cnt)) - 1
        return Int((self.bit_buf & mask) << UInt64(count - self.bit_cnt))

    @always_inline
    def consume(mut self, count: Int):
        if count > self.bit_cnt:
            self.valid = False
            return
        self.bit_cnt -= count


def _exr_huf_uncompress(
    data: List[UInt8],
    huf_start: Int,
    huf_len: Int,
    n_raw_u16: Int,
) -> Tuple[Bool, List[UInt16]]:
    """OpenEXR canonical MSB-first Huffman decoder (PIZ & DWA `AC_STATIC_HUFFMAN`)."""
    var empty = List[UInt16]()
    if huf_len == 0:
        if n_raw_u16 == 0:
            return (True, empty^)
        return (False, empty^)
    if huf_len < 20 or huf_start < 0 or huf_start + huf_len > len(data):
        return (False, empty^)

    var im = _le_i32(data, huf_start + 0)
    var iM = _le_i32(data, huf_start + 4)
    var table_len = _le_i32(data, huf_start + 8)
    var n_bits = _le_i32(data, huf_start + 12)
    if im < 0 or iM < im or iM >= 65537:
        return (False, empty^)
    if table_len < 0 or 20 + table_len > huf_len:
        return (False, empty^)
    if n_bits < 0 or 20 + table_len + (n_bits + 7) // 8 > huf_len:
        return (False, empty^)

    var lengths = List[Int](length=iM + 1, fill=0)
    var len_counts = List[Int](length=59, fill=0)
    comptime HUF_KRAFT_MAX: UInt64 = UInt64(1) << 58
    var kraft_sum: UInt64 = 0
    var tbr = _ExrMsbReader(data, huf_start + 20, huf_start + 20 + table_len)
    var s = im
    while s <= iM and tbr.valid:
        var l = tbr.read_msb(6)
        if not tbr.valid:
            return (False, empty^)
        if l == 63:  # LONG_ZEROCODE_RUN
            var z = tbr.read_msb(8) + 6
            if not tbr.valid or s + z > iM + 1:
                return (False, empty^)
            s += z
        elif l >= 59:  # SHORT_ZEROCODE_RUN (59..62 -> 2..5)
            var z = l - 59 + 2
            if s + z > iM + 1:
                return (False, empty^)
            s += z
        else:
            lengths[s] = l
            if l > 0:
                var add_k = UInt64(1) << UInt64(58 - l)
                if add_k > HUF_KRAFT_MAX - kraft_sum:
                    return (False, empty^)
                kraft_sum += add_k
                len_counts[l] += 1
            s += 1
    if not tbr.valid or s != iM + 1:
        return (False, empty^)

    # OpenEXR canonical code assignment: reverse order 58 -> 1
    var first_code = List[UInt64](length=59, fill=0)
    var c: UInt64 = 0
    var l_rev = 58
    while l_rev >= 1:
        var nc = (c + UInt64(len_counts[l_rev])) >> 1
        first_code[l_rev] = c
        if len_counts[l_rev] > 0:
            var last_code = c + UInt64(len_counts[l_rev]) - 1
            if last_code >= (UInt64(1) << UInt64(l_rev)):
                return (False, empty^)
        c = nc
        l_rev -= 1
    if c > UInt64(1):
        return (False, empty^)

    # Build 14-bit fast lookup table + long-code list (15..58 bits)
    comptime HUF_FAST_BITS: Int = 14
    comptime HUF_FAST_SZ: Int = 1 << HUF_FAST_BITS
    var fast_sym = List[Int](length=HUF_FAST_SZ, fill=-1)
    var fast_len = List[UInt8](length=HUF_FAST_SZ, fill=0)
    var long_codes = List[UInt64]()
    var long_lens = List[Int]()
    var long_syms = List[Int]()

    for sym in range(im, iM + 1):
        var l = lengths[sym]
        if l == 0:
            continue
        var code = first_code[l]
        first_code[l] += 1
        if code >= (UInt64(1) << UInt64(l)):
            return (False, empty^)
        if l <= HUF_FAST_BITS:
            var shift = HUF_FAST_BITS - l
            var base = Int(code << UInt64(shift))
            var span = 1 << shift
            if base < 0 or base + span > HUF_FAST_SZ:
                return (False, empty^)
            for k in range(span):
                if fast_len[base + k] != 0:
                    return (False, empty^)
                fast_sym[base + k] = sym
                fast_len[base + k] = UInt8(l)
        else:
            long_codes.append(code)
            long_lens.append(l)
            long_syms.append(sym)

    var data_start = huf_start + 20 + table_len
    var data_end = data_start + (n_bits + 7) // 8
    var dbr = _ExrMsbReader(data, data_start, data_end)
    var bits_left = n_bits
    var out = List[UInt16](capacity=n_raw_u16)
    var rle_sym = iM

    while bits_left > 0 and len(out) < n_raw_u16 and dbr.valid:
        var sym = -1
        var slen = 0
        var peek14 = dbr.peek_msb(HUF_FAST_BITS)
        var fl = Int(fast_len[peek14])
        if fl > 0 and fl <= bits_left:
            sym = fast_sym[peek14]
            slen = fl
            dbr.consume(slen)
        else:
            # Slow path for >14-bit codes
            var matched = False
            var cur_code: UInt64 = 0
            for cl in range(1, min(58, bits_left) + 1):
                cur_code = (cur_code << 1) | UInt64(dbr.read_msb(1))
                if not dbr.valid:
                    return (False, empty^)
                for li in range(len(long_codes)):
                    if long_lens[li] == cl and long_codes[li] == cur_code:
                        sym = long_syms[li]
                        slen = cl
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return (False, empty^)
        bits_left -= slen
        if sym == rle_sym:
            if bits_left < 8 or len(out) == 0:
                return (False, empty^)
            var run_cnt = dbr.read_msb(8)
            bits_left -= 8
            if not dbr.valid or len(out) + run_cnt > n_raw_u16:
                return (False, empty^)
            var prev_v = out[len(out) - 1]
            for _ in range(run_cnt):
                out.append(prev_v)
        else:
            if sym < 0 or sym >= 65536:
                return (False, empty^)
            out.append(UInt16(sym))

    if not dbr.valid or len(out) != n_raw_u16:
        return (False, empty^)
    return (True, out^)


@always_inline
def _exr_wdec14(l: UInt16, h: UInt16) -> Tuple[UInt16, UInt16]:
    var hi = Int(bitcast[DType.int16](h))
    var li = Int(bitcast[DType.int16](l))
    var ai = li + (hi & 1) + (hi >> 1)
    return (UInt16(ai & 0xFFFF), UInt16((ai - hi) & 0xFFFF))


@always_inline
def _exr_wdec16(l: UInt16, h: UInt16) -> Tuple[UInt16, UInt16]:
    var bb = (Int(l) - (Int(h) >> 1)) & 0xFFFF
    var aa = (Int(h) + bb - 32768) & 0xFFFF
    return (UInt16(aa), UInt16(bb))


def _exr_wav2_decode(mut buf: List[UInt16], base: Int, nx: Int, ox: Int, ny: Int, oy: Int, max_value: Int):
    """Inverse 2D Haar wavelet transform (`wdec14` for `max_value < 16384`, `wdec16` otherwise)."""
    var n = nx if nx < ny else ny
    var p = 1
    while p <= n:
        p <<= 1
    p >>= 1
    var p2 = p
    p >>= 1
    var use14 = max_value < 16384

    while p >= 1:
        var y_end = oy * (ny - p2)
        var x_end = ox * (nx - p2)

        var iy = 0
        while iy <= y_end:
            var ix = 0
            while ix <= x_end:
                var i00 = base + iy + ix
                var i01 = i00 + ox * p
                var i10 = i00 + oy * p
                var i11 = i10 + ox * p
                if use14:
                    var v0 = _exr_wdec14(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
                    var v1 = _exr_wdec14(buf[i01], buf[i11]); buf[i01] = v1[0]; buf[i11] = v1[1]
                    var h0 = _exr_wdec14(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                    var h1 = _exr_wdec14(buf[i10], buf[i11]); buf[i10] = h1[0]; buf[i11] = h1[1]
                else:
                    var v0 = _exr_wdec16(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
                    var v1 = _exr_wdec16(buf[i01], buf[i11]); buf[i01] = v1[0]; buf[i11] = v1[1]
                    var h0 = _exr_wdec16(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                    var h1 = _exr_wdec16(buf[i10], buf[i11]); buf[i10] = h1[0]; buf[i11] = h1[1]
                ix += ox * p2
            if (nx & p) != 0:
                var i00 = base + iy + ix
                var i10 = i00 + oy * p
                if use14:
                    var v0 = _exr_wdec14(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
                else:
                    var v0 = _exr_wdec16(buf[i00], buf[i10]); buf[i00] = v0[0]; buf[i10] = v0[1]
            iy += oy * p2

        if (ny & p) != 0:
            var ix = 0
            while ix <= x_end:
                var i00 = base + iy + ix
                var i01 = i00 + ox * p
                if use14:
                    var h0 = _exr_wdec14(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                else:
                    var h0 = _exr_wdec16(buf[i00], buf[i01]); buf[i00] = h0[0]; buf[i01] = h0[1]
                ix += ox * p2

        p2 = p
        p >>= 1


@always_inline
def _dwa_to_linear(x: Float32) -> Float32:
    if x == Float32(0.0):
        return Float32(0.0)
    var s = Float32(-1.0) if x < Float32(0.0) else Float32(1.0)
    var ax = -x if x < Float32(0.0) else x
    if ax <= Float32(1.0):
        return s * exp(Float32(2.2) * log(ax))
    return s * exp(Float32(2.2) * (ax - Float32(1.0)))


def _exr_dwa_idct_8x8(coeff: List[Float32], mut out_spatial: List[Float32]):
    """Orthonormal 8x8 separable IDCT matching OpenEXR DWA (`DctCoder`)."""
    var tmp = List[Float32](length=64, fill=0.0)
    var inv_sqrt8 = Float32(0.3535533905932738)
    var c_norm = Float32(0.5)
    var pi_16 = Float32(0.19634954084936207)
    for y in range(8):
        for x in range(8):
            var acc = Float32(0.0)
            for u in range(8):
                var cu = inv_sqrt8 if u == 0 else c_norm
                acc += coeff[y * 8 + u] * cu * cos(Float32(2 * x + 1) * Float32(u) * pi_16)
            tmp[y * 8 + x] = acc
    for y in range(8):
        for x in range(8):
            var acc = Float32(0.0)
            for v in range(8):
                var cv = inv_sqrt8 if v == 0 else c_norm
                acc += tmp[v * 8 + x] * cv * cos(Float32(2 * y + 1) * Float32(v) * pi_16)
            out_spatial[y * 8 + x] = acc


def _exr_undo_predictor_and_reorder(raw_zip: List[UInt8]) -> List[UInt8]:
    """Reverses OpenEXR's ZIP byte delta predictor & even/odd split interleave."""
    var n = len(raw_zip)
    if n <= 1:
        return raw_zip.copy()
    var pred = raw_zip.copy()
    for i in range(1, n):
        pred[i] = UInt8((Int(pred[i - 1]) + Int(pred[i]) - 128) & 0xFF)

    var out = List[UInt8](length=n, fill=0)
    var half = (n + 1) // 2
    var i1 = 0
    var i2 = half
    var j = 0
    while True:
        if j < n:
            out[j] = pred[i1]
            j += 1
            i1 += 1
        else:
            break
        if j < n:
            out[j] = pred[i2]
            j += 1
            i2 += 1
        else:
            break
    return out^


def encode_exr_fp16(pixels: List[Vec3], width: Int, height: Int) -> List[UInt8]:
    """Writes a Khronos/ILM-compliant OpenEXR v2 single-part scanline `HALF` (`FP16`) `B, G, R` image."""
    var out = List[UInt8]()
    _put_le_u32(out, UInt32(EXR_MAGIC))
    _put_le_u32(out, UInt32(2))  # version 2, flags = 0 (single-part scanline)

    # 1. Attribute: channels (alphabetical: B, G, R, each 18 bytes + 1 trailing NUL = 55 bytes)
    _put_cstr(out, "channels")
    _put_cstr(out, "chlist")
    _put_le_u32(out, UInt32(3 * 18 + 1))
    for ch_name in ["B", "G", "R"]:
        _put_cstr(out, String(ch_name))       # 2 bytes
        _put_le_u32(out, UInt32(EXR_PIXEL_HALF))  # 4 bytes
        out.append(1)                         # pLinear = 1
        out.append(0); out.append(0); out.append(0)
        _put_le_u32(out, 1)                   # xSampling = 1
        _put_le_u32(out, 1)                   # ySampling = 1
    out.append(0)  # end of chlist

    # 2. Attribute: compression (NO_COMPRESSION = 0)
    _put_cstr(out, "compression")
    _put_cstr(out, "compression")
    _put_le_u32(out, 1)
    out.append(UInt8(EXR_NO_COMPRESSION))

    # 3. Attribute: dataWindow & displayWindow (`box2i`: 0, 0, width - 1, height - 1)
    for win_attr in ["dataWindow", "displayWindow"]:
        _put_cstr(out, String(win_attr))
        _put_cstr(out, "box2i")
        _put_le_u32(out, 16)
        _put_le_u32(out, 0)
        _put_le_u32(out, 0)
        _put_le_u32(out, UInt32(width - 1))
        _put_le_u32(out, UInt32(height - 1))

    # 4. Attribute: lineOrder (INCREASING_Y = 0)
    _put_cstr(out, "lineOrder")
    _put_cstr(out, "lineOrder")
    _put_le_u32(out, 1)
    out.append(0)

    # 5. Attribute: pixelAspectRatio = 1.0f
    _put_cstr(out, "pixelAspectRatio")
    _put_cstr(out, "float")
    _put_le_u32(out, 4)
    _put_le_u32(out, bitcast[DType.uint32](Float32(1.0)))

    # End of header
    out.append(0)

    # Scanline offset table: `height` UInt64 offsets
    var table_start = len(out)
    var scanline_bytes = width * 3 * 2  # 3 HALF channels * 2 bytes * width
    var chunk_bytes = 8 + scanline_bytes
    var first_chunk_off = table_start + height * 8
    for y in range(height):
        _put_le_u64(out, UInt64(first_chunk_off + y * chunk_bytes))

    # Scanline pixel chunks: [y: Int32][data_size: Int32][B row (F16)][G row (F16)][R row (F16)]
    for y in range(height):
        _put_le_u32(out, UInt32(y))
        _put_le_u32(out, UInt32(scanline_bytes))
        # Channel 0: B
        for x in range(width):
            var hb = _f32_to_f16_signed(pixels[y * width + x].z)
            out.append(UInt8(hb & 0xFF))
            out.append(UInt8((hb >> 8) & 0xFF))
        # Channel 1: G
        for x in range(width):
            var hg = _f32_to_f16_signed(pixels[y * width + x].y)
            out.append(UInt8(hg & 0xFF))
            out.append(UInt8((hg >> 8) & 0xFF))
        # Channel 2: R
        for x in range(width):
            var hr = _f32_to_f16_signed(pixels[y * width + x].x)
            out.append(UInt8(hr & 0xFF))
            out.append(UInt8((hr >> 8) & 0xFF))

    return out^


def _exr_write_channel_val(
    mut img: HdrImage,
    dst_x: Int,
    dst_y: Int,
    role: Int,
    val_f: Float32,
):
    var idx = dst_y * img.width + dst_x
    var cur_px = img.pixels[idx]
    var v = max(Float32(0.0), val_f)
    if role == 0:
        img.pixels[idx] = Vec3(v, cur_px.y, cur_px.z)
    elif role == 1:
        img.pixels[idx] = Vec3(cur_px.x, v, cur_px.z)
    elif role == 2:
        img.pixels[idx] = Vec3(cur_px.x, cur_px.y, v)
    elif role == 3:
        img.pixels[idx] = Vec3(v, v, v)


def _exr_decode_piz_block(
    data: List[UInt8],
    data_start: Int,
    packed_sz: Int,
    bw: Int,
    bh: Int,
    ch_types: List[Int],
    ch_byte_sizes: List[Int],
    ch_rgb_role: List[Int],
    expected_raw: Int,
    dst_x0: Int,
    dst_y0: Int,
    mut img: HdrImage,
) -> Bool:
    if packed_sz == expected_raw:
        var r_cursor = 0
        for ly in range(bh):
            var dst_y = dst_y0 + ly
            for ci in range(len(ch_types)):
                var role = ch_rgb_role[ci]
                var ptype = ch_types[ci]
                var bsz = ch_byte_sizes[ci]
                if role < 0:
                    r_cursor += bsz * bw
                    continue
                for x in range(bw):
                    var off = data_start + r_cursor
                    var val_f = Float32(0.0)
                    if ptype == EXR_PIXEL_HALF:
                        val_f = _f16_signed_to_f32(_le_u16(data, off))
                    elif ptype == EXR_PIXEL_FLOAT:
                        var u32_v = (
                            UInt32(data[off])
                            | (UInt32(data[off + 1]) << 8)
                            | (UInt32(data[off + 2]) << 16)
                            | (UInt32(data[off + 3]) << 24)
                        )
                        val_f = bitcast[DType.float32](u32_v)
                    r_cursor += bsz
                    _exr_write_channel_val(img, dst_x0 + x, dst_y, role, val_f)
        return True

    if packed_sz < 8:
        return False
    var data_end = data_start + packed_sz
    var p = data_start
    var min_nz = _le_u16(data, p)
    var max_nz = _le_u16(data, p + 2)
    p += 4
    var bitmap = List[UInt8](length=8192, fill=0)
    if min_nz <= max_nz:
        if max_nz >= 8192 or p + (max_nz - min_nz + 1) > data_end:
            return False
        for bi in range(min_nz, max_nz + 1):
            bitmap[bi] = data[p]
            p += 1

    var lut = List[UInt16](length=65536, fill=0)
    var k = 0
    for i in range(65536):
        if i == 0 or (bitmap[i >> 3] & UInt8(1 << (i & 7))) != 0:
            lut[k] = UInt16(i)
            k += 1
    var max_value = k - 1

    if p + 4 > data_end:
        return False
    var huf_len = _le_i32(data, p)
    p += 4
    if huf_len < 0 or p + huf_len > data_end:
        return False

    var n_u16 = expected_raw // 2
    var hres = _exr_huf_uncompress(data, p, huf_len, n_u16)
    if not hres[0]:
        return False
    var coeffs = hres[1].copy()

    var ch_base = 0
    for ci in range(len(ch_types)):
        var sz_u16 = 1 if ch_types[ci] == EXR_PIXEL_HALF else 2
        var cnt = bw * bh * sz_u16
        if ch_base + cnt > len(coeffs):
            return False
        for j in range(sz_u16):
            _exr_wav2_decode(coeffs, ch_base + j, bw, sz_u16, bh, bw * sz_u16, max_value)
        ch_base += cnt

    for i in range(len(coeffs)):
        var u = Int(coeffs[i])
        if u > max_value:
            return False
        coeffs[i] = lut[u]

    # Reorder channel-planar coeffs into pixels
    ch_base = 0
    for ci in range(len(ch_types)):
        var sz_u16 = 1 if ch_types[ci] == EXR_PIXEL_HALF else 2
        var role = ch_rgb_role[ci]
        var ptype = ch_types[ci]
        if role >= 0:
            for ly in range(bh):
                var dst_y = dst_y0 + ly
                for lx in range(bw):
                    var off = ch_base + (ly * bw + lx) * sz_u16
                    var val_f = Float32(0.0)
                    if ptype == EXR_PIXEL_HALF:
                        val_f = _f16_signed_to_f32(Int(coeffs[off]))
                    elif ptype == EXR_PIXEL_FLOAT:
                        var bits32 = UInt32(coeffs[off]) | (UInt32(coeffs[off + 1]) << 16)
                        val_f = bitcast[DType.float32](bits32)
                    _exr_write_channel_val(img, dst_x0 + lx, dst_y, role, val_f)
        ch_base += bw * bh * sz_u16
    return True


def _exr_suffix_match(name: String, suffix: String, case_insensitive: Bool) -> Bool:
    var nb = name.as_bytes()
    var sb = suffix.as_bytes()
    if len(sb) > len(nb):
        return False
    var off = len(nb) - len(sb)
    if off > 0 and nb[off - 1] != 46:  # '.'
        return False
    for i in range(len(sb)):
        var ca = Int(nb[off + i])
        var cb = Int(sb[i])
        if case_insensitive:
            if ca >= 65 and ca <= 90:
                ca += 32
            if cb >= 65 and cb <= 90:
                cb += 32
        if ca != cb:
            return False
    return True


def _exr_channel_layer_prefix(name: String) -> String:
    var nb = name.as_bytes()
    var last_dot = -1
    for i in range(len(nb)):
        if nb[i] == 46:
            last_dot = i
    if last_dot <= 0:
        return String()
    return String(name[byte=0 : last_dot])


def _exr_decode_dwa_block(
    data: List[UInt8],
    data_start: Int,
    packed_sz: Int,
    bw: Int,
    bh: Int,
    ch_names: List[String],
    ch_types: List[Int],
    ch_byte_sizes: List[Int],
    ch_rgb_role: List[Int],
    expected_raw: Int,
    dst_x0: Int,
    dst_y0: Int,
    mut img: HdrImage,
) -> Bool:
    if packed_sz == expected_raw:
        var r_cursor = 0
        for ly in range(bh):
            var dst_y = dst_y0 + ly
            for ci in range(len(ch_types)):
                var role = ch_rgb_role[ci]
                var ptype = ch_types[ci]
                var bsz = ch_byte_sizes[ci]
                if role < 0:
                    r_cursor += bsz * bw
                    continue
                for x in range(bw):
                    var off = data_start + r_cursor
                    var val_f = Float32(0.0)
                    if ptype == EXR_PIXEL_HALF:
                        val_f = _f16_signed_to_f32(_le_u16(data, off))
                    elif ptype == EXR_PIXEL_FLOAT:
                        var u32_v = (
                            UInt32(data[off])
                            | (UInt32(data[off + 1]) << 8)
                            | (UInt32(data[off + 2]) << 16)
                            | (UInt32(data[off + 3]) << 24)
                        )
                        val_f = bitcast[DType.float32](u32_v)
                    r_cursor += bsz
                    _exr_write_channel_val(img, dst_x0 + x, dst_y, role, val_f)
        return True

    if packed_sz < 88:
        return False
    var data_end = data_start + packed_sz
    var p = data_start
    var version = _le_i64(data, p + 0)
    var unknown_uncomp_sz = _le_i64(data, p + 8)
    var unknown_comp_sz = _le_i64(data, p + 16)
    var ac_comp_sz = _le_i64(data, p + 24)
    var dc_comp_sz = _le_i64(data, p + 32)
    var rle_comp_sz = _le_i64(data, p + 40)
    var rle_uncomp_sz = _le_i64(data, p + 48)
    var rle_raw_sz = _le_i64(data, p + 56)
    var total_ac_cnt = _le_i64(data, p + 64)
    var total_dc_cnt = _le_i64(data, p + 72)
    var ac_compression = _le_i64(data, p + 80)
    p += 88

    if (
        version < 0 or version > 2
        or unknown_uncomp_sz < 0 or unknown_comp_sz < 0
        or ac_comp_sz < 0 or dc_comp_sz < 0
        or rle_comp_sz < 0 or rle_uncomp_sz < 0 or rle_raw_sz < 0
        or total_ac_cnt < 0 or total_dc_cnt < 0
        or (ac_compression != 0 and ac_compression != 1)
    ):
        return False

    var n_ch = len(ch_names)
    var ch_scheme = List[Int](length=n_ch, fill=0)  # 0=UNKNOWN, 1=LOSSY_DCT, 2=RLE
    var ch_csc = List[Int](length=n_ch, fill=-1)    # -1=none, 0=R, 1=G, 2=B

    if version >= 2:
        if p + 2 > data_end:
            return False
        var rule_size = _le_u16(data, p)
        if rule_size < 2 or p + rule_size > data_end:
            return False
        var rule_end = p + rule_size
        p += 2
        while p < rule_end:
            var s_res = _read_cstr(data, p, rule_end)
            p = s_res[1]
            if p >= rule_end:
                return False
            var val_u8 = Int(data[p])
            p += 1
            var scheme = (val_u8 >> 2) & 0x03
            var csc_idx = ((val_u8 >> 4) & 0x03) - 1
            var ci_flag = (val_u8 & 0x40) != 0
            for ci in range(n_ch):
                if _exr_suffix_match(ch_names[ci], s_res[0], ci_flag):
                    ch_scheme[ci] = scheme
                    ch_csc[ci] = csc_idx
    else:
        for ci in range(n_ch):
            if _exr_suffix_match(ch_names[ci], "R", False):
                ch_scheme[ci] = 1; ch_csc[ci] = 0
            elif _exr_suffix_match(ch_names[ci], "G", False):
                ch_scheme[ci] = 1; ch_csc[ci] = 1
            elif _exr_suffix_match(ch_names[ci], "B", False):
                ch_scheme[ci] = 1; ch_csc[ci] = 2
            elif _exr_suffix_match(ch_names[ci], "Y", False):
                ch_scheme[ci] = 1; ch_csc[ci] = -1
            elif _exr_suffix_match(ch_names[ci], "A", False):
                ch_scheme[ci] = 2; ch_csc[ci] = -1
            else:
                ch_scheme[ci] = 0; ch_csc[ci] = -1

    if p + unknown_comp_sz + ac_comp_sz + dc_comp_sz + rle_comp_sz > data_end:
        return False
    if unknown_uncomp_sz > 0 and unknown_comp_sz <= 0:
        return False
    if total_ac_cnt > 0 and ac_comp_sz <= 0:
        return False
    if total_dc_cnt > 0 and dc_comp_sz <= 0:
        return False
    if (rle_raw_sz > 0 or rle_uncomp_sz > 0) and rle_comp_sz <= 0:
        return False

    var unknown_bytes = List[UInt8]()
    if unknown_comp_sz > 0:
        unknown_bytes = inflate_zlib(data, p, unknown_uncomp_sz)
        if len(unknown_bytes) != unknown_uncomp_sz:
            return False
        p += unknown_comp_sz

    var ac_u16 = List[UInt16]()
    if total_ac_cnt > 0:
        if ac_compression == 0:
            var hres = _exr_huf_uncompress(data, p, ac_comp_sz, total_ac_cnt)
            if not hres[0]:
                return False
            ac_u16 = hres[1].copy()
        else:
            var inf_ac = inflate_zlib(data, p, total_ac_cnt * 2)
            if len(inf_ac) != total_ac_cnt * 2:
                return False
            ac_u16 = List[UInt16](capacity=total_ac_cnt)
            for i in range(total_ac_cnt):
                ac_u16.append(UInt16(_le_u16(inf_ac, i * 2)))
    p += ac_comp_sz

    var dc_u16 = List[UInt16]()
    if total_dc_cnt > 0:
        var inf_dc = inflate_zlib(data, p, total_dc_cnt * 2)
        if len(inf_dc) != total_dc_cnt * 2:
            return False
        var raw_dc = _exr_undo_predictor_and_reorder(inf_dc)
        dc_u16 = List[UInt16](capacity=total_dc_cnt)
        for i in range(total_dc_cnt):
            dc_u16.append(UInt16(_le_u16(raw_dc, i * 2)))
    p += dc_comp_sz

    var rle_raw = List[UInt8]()
    if rle_comp_sz > 0:
        var inf_rle = inflate_zlib(data, p, rle_uncomp_sz)
        if len(inf_rle) != rle_uncomp_sz:
            return False
        var rres = _exr_rle_uncompress(inf_rle, 0, len(inf_rle), rle_raw_sz)
        if not rres[0]:
            return False
        rle_raw = rres[1].copy()
    p += rle_comp_sz

    # Build DCT channel block grids
    var nbx = (bw + 7) // 8
    var nby = (bh + 7) // 8
    var n_blocks = nbx * nby
    var dct_ch_indices = List[Int]()
    for ci in range(n_ch):
        if ch_scheme[ci] == 1:
            dct_ch_indices.append(ci)

    if len(dct_ch_indices) > 0:
        var n_dct = len(dct_ch_indices)
        var expected_dc = n_dct * n_blocks
        if len(dc_u16) != expected_dc:
            return False

        # Reconstruct spatial 8x8 blocks: DC stream is channel-major, AC stream is block-major
        comptime _DWA_ZZ = SIMD[DType.uint8, 64](
            0, 1, 8, 16, 9, 2, 3, 10,
            17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63,
        )
        var dct_planes = List[List[Float32]](capacity=n_dct)
        for _ in range(n_dct):
            dct_planes.append(List[Float32](length=bw * bh, fill=0.0))
        var ac_cursor = 0
        var coeff = List[Float32](length=64, fill=0.0)
        var spatial = List[Float32](length=64, fill=0.0)

        for by in range(nby):
            var th = min(8, bh - by * 8)
            for bx in range(nbx):
                var tw = min(8, bw - bx * 8)
                var blk_idx = by * nbx + bx
                for di in range(n_dct):
                    for i in range(64):
                        coeff[i] = 0.0
                    coeff[0] = _f16_signed_to_f32(Int(dc_u16[di * n_blocks + blk_idx]))
                    var k = 1
                    while k < 64:
                        if ac_cursor >= len(ac_u16):
                            return False
                        var v = Int(ac_u16[ac_cursor])
                        ac_cursor += 1
                        if (v & 0xFF00) == 0xFF00:
                            var run = v & 0xFF
                            if run == 0:
                                break
                            k += run
                            if k > 64:
                                return False
                        else:
                            coeff[Int(_DWA_ZZ[k])] = _f16_signed_to_f32(v)
                            k += 1
                    _exr_dwa_idct_8x8(coeff, spatial)
                    for ly in range(th):
                        for lx in range(tw):
                            dct_planes[di][(by * 8 + ly) * bw + bx * 8 + lx] = spatial[ly * 8 + lx]

        # Group CSC triplets (stream planes are [Y, Cb, Cr], output mapped by csc_idx 0=R, 1=G, 2=B)
        var used_dct = List[Bool](length=n_dct, fill=False)
        for di in range(n_dct):
            if used_dct[di]:
                continue
            var ci = dct_ch_indices[di]
            if ch_csc[ci] >= 0:
                var layer = _exr_channel_layer_prefix(ch_names[ci])
                var c_idx = [-1, -1, -1]
                var trip_planes = List[Int]()
                for dj in range(di, n_dct):
                    var cj = dct_ch_indices[dj]
                    if ch_csc[cj] >= 0 and ch_csc[cj] <= 2 and _exr_channel_layer_prefix(ch_names[cj]) == layer:
                        if c_idx[ch_csc[cj]] < 0:
                            c_idx[ch_csc[cj]] = dj
                            trip_planes.append(dj)
                if c_idx[0] >= 0 and c_idx[1] >= 0 and c_idx[2] >= 0 and len(trip_planes) == 3:
                    used_dct[c_idx[0]] = True
                    used_dct[c_idx[1]] = True
                    used_dct[c_idx[2]] = True
                    var role_r = ch_rgb_role[dct_ch_indices[c_idx[0]]]
                    var role_g = ch_rgb_role[dct_ch_indices[c_idx[1]]]
                    var role_b = ch_rgb_role[dct_ch_indices[c_idx[2]]]
                    var y_plane_idx = trip_planes[0]
                    var cb_plane_idx = trip_planes[1]
                    var cr_plane_idx = trip_planes[2]
                    for ly in range(bh):
                        for lx in range(bw):
                            var p_i = ly * bw + lx
                            var y_v = dct_planes[y_plane_idx][p_i]
                            var cb_v = dct_planes[cb_plane_idx][p_i]
                            var cr_v = dct_planes[cr_plane_idx][p_i]
                            var r_nl = y_v + Float32(1.5747) * cr_v
                            var g_nl = y_v - Float32(0.1873) * cb_v - Float32(0.4682) * cr_v
                            var b_nl = y_v + Float32(1.8556) * cb_v
                            if role_r >= 0:
                                _exr_write_channel_val(img, dst_x0 + lx, dst_y0 + ly, role_r, _dwa_to_linear(r_nl))
                            if role_g >= 0:
                                _exr_write_channel_val(img, dst_x0 + lx, dst_y0 + ly, role_g, _dwa_to_linear(g_nl))
                            if role_b >= 0:
                                _exr_write_channel_val(img, dst_x0 + lx, dst_y0 + ly, role_b, _dwa_to_linear(b_nl))
                    continue

            used_dct[di] = True
            var role = ch_rgb_role[ci]
            if role >= 0:
                for ly in range(bh):
                    for lx in range(bw):
                        _exr_write_channel_val(img, dst_x0 + lx, dst_y0 + ly, role, _dwa_to_linear(dct_planes[di][ly * bw + lx]))

    # Handle any RLE or UNKNOWN channels that carry RGB/Y
    var rle_cursor = 0
    var unk_cursor = 0
    for ci in range(n_ch):
        var bsz = ch_byte_sizes[ci]
        var ch_bytes = bw * bh * bsz
        var role = ch_rgb_role[ci]
        var ptype = ch_types[ci]
        if ch_scheme[ci] == 2:
            if rle_cursor + ch_bytes > len(rle_raw):
                return False
            if role >= 0:
                var px_cnt = bw * bh
                for pi in range(px_cnt):
                    var ly = pi // bw
                    var lx = pi % bw
                    var val_f = Float32(0.0)
                    if ptype == EXR_PIXEL_HALF:
                        var lo_b = Int(rle_raw[rle_cursor + px_cnt + pi])
                        var hi_b = Int(rle_raw[rle_cursor + pi])
                        val_f = _f16_signed_to_f32((hi_b << 8) | lo_b)
                    elif ptype == EXR_PIXEL_FLOAT:
                        var b0 = UInt32(rle_raw[rle_cursor + 3 * px_cnt + pi])
                        var b1 = UInt32(rle_raw[rle_cursor + 2 * px_cnt + pi])
                        var b2 = UInt32(rle_raw[rle_cursor + px_cnt + pi])
                        var b3 = UInt32(rle_raw[rle_cursor + pi])
                        val_f = bitcast[DType.float32](b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
                    _exr_write_channel_val(img, dst_x0 + lx, dst_y0 + ly, role, val_f)
            rle_cursor += ch_bytes
        elif ch_scheme[ci] == 0:
            if unk_cursor + ch_bytes > len(unknown_bytes):
                return False
            if role >= 0:
                for ly in range(bh):
                    for lx in range(bw):
                        var off = unk_cursor + (ly * bw + lx) * bsz
                        var val_f = Float32(0.0)
                        if ptype == EXR_PIXEL_HALF:
                            val_f = _f16_signed_to_f32(_le_u16(unknown_bytes, off))
                        elif ptype == EXR_PIXEL_FLOAT:
                            var u32_v = (
                                UInt32(unknown_bytes[off])
                                | (UInt32(unknown_bytes[off + 1]) << 8)
                                | (UInt32(unknown_bytes[off + 2]) << 16)
                                | (UInt32(unknown_bytes[off + 3]) << 24)
                            )
                            val_f = bitcast[DType.float32](u32_v)
                        _exr_write_channel_val(img, dst_x0 + lx, dst_y0 + ly, role, val_f)
            unk_cursor += ch_bytes

    return True


@always_inline
def _exr_level_size(dim: Int, level: Int, rmode: Int) -> Int:
    var b = 1 << level
    var sz = dim // b
    if rmode == 1 and sz * b < dim:
        sz += 1
    return max(1, sz)


@always_inline
def _exr_round_log2(x: Int, rmode: Int) -> Int:
    var v = max(1, x)
    var l = 0
    var p = 1
    while (p << 1) <= v:
        p <<= 1
        l += 1
    if rmode == 1 and p < v:
        l += 1
    return l


def _exr_decode_chunk_into(
    data: List[UInt8],
    data_start: Int,
    packed_sz: Int,
    compression: Int,
    bw: Int,
    bh: Int,
    ch_names: List[String],
    ch_types: List[Int],
    ch_byte_sizes: List[Int],
    ch_rgb_role: List[Int],
    dst_x0: Int,
    dst_y0: Int,
    mut img: HdrImage,
) -> Bool:
    var bytes_per_pixel_row = 0
    for ci in range(len(ch_types)):
        bytes_per_pixel_row += ch_byte_sizes[ci] * bw
    var expected_raw = bh * bytes_per_pixel_row

    if compression == EXR_PIZ_COMPRESSION:
        return _exr_decode_piz_block(
            data, data_start, packed_sz, bw, bh, ch_types, ch_byte_sizes, ch_rgb_role, expected_raw, dst_x0, dst_y0, img
        )
    if compression == EXR_DWAA_COMPRESSION or compression == EXR_DWAB_COMPRESSION:
        return _exr_decode_dwa_block(
            data, data_start, packed_sz, bw, bh, ch_names, ch_types, ch_byte_sizes, ch_rgb_role, expected_raw, dst_x0, dst_y0, img
        )

    var raw_block: List[UInt8]
    if compression == EXR_NO_COMPRESSION or packed_sz == expected_raw:
        if packed_sz < expected_raw:
            return False
        raw_block = List[UInt8](length=expected_raw, fill=0)
        for k in range(expected_raw):
            raw_block[k] = data[data_start + k]
    elif compression == EXR_RLE_COMPRESSION:
        var rres = _exr_rle_uncompress(data, data_start, packed_sz, expected_raw)
        if not rres[0]:
            return False
        raw_block = _exr_undo_predictor_and_reorder(rres[1])
    else:
        if packed_sz >= expected_raw:
            raw_block = List[UInt8](length=expected_raw, fill=0)
            for k in range(expected_raw):
                raw_block[k] = data[data_start + k]
        else:
            var inflated = inflate_zlib(data, data_start, expected_raw)
            if len(inflated) != expected_raw:
                return False
            raw_block = _exr_undo_predictor_and_reorder(inflated)

    var r_cursor = 0
    for ly in range(bh):
        var dst_y = dst_y0 + ly
        for ci in range(len(ch_names)):
            var role = ch_rgb_role[ci]
            var ptype = ch_types[ci]
            var bsz = ch_byte_sizes[ci]
            if role < 0:
                r_cursor += bsz * bw
                continue
            for x in range(bw):
                var val_f = Float32(0.0)
                if ptype == EXR_PIXEL_HALF:
                    var h16 = Int(raw_block[r_cursor]) | (Int(raw_block[r_cursor + 1]) << 8)
                    val_f = _f16_signed_to_f32(h16)
                elif ptype == EXR_PIXEL_FLOAT:
                    var u32_v = (
                        UInt32(raw_block[r_cursor])
                        | (UInt32(raw_block[r_cursor + 1]) << 8)
                        | (UInt32(raw_block[r_cursor + 2]) << 16)
                        | (UInt32(raw_block[r_cursor + 3]) << 24)
                    )
                    val_f = bitcast[DType.float32](u32_v)
                r_cursor += bsz
                _exr_write_channel_val(img, dst_x0 + x, dst_y, role, val_f)
    return True


def decode_exr_fp16(data: List[UInt8]) -> HdrImage:
    """Decodes OpenEXR v2 scanline & tiled `.exr` (`HALF`/`FLOAT`, `NONE`/`RLE`/`ZIPS`/`ZIP`/`PIZ`/`DWAA`/`DWAB`)."""
    var img = HdrImage()
    var n = len(data)
    if n < 32 or _le_i32(data, 0) != EXR_MAGIC:
        return img^
    var ver_word = _le_i32(data, 4)
    if (ver_word & 0xFF) != 2 or (ver_word & 0x800) != 0 or (ver_word & 0x1000) != 0:
        return img^
    var is_tiled = (ver_word & 0x200) != 0

    var pos = 8
    var x_min = 0; var y_min = 0; var x_max = -1; var y_max = -1
    var compression = EXR_NO_COMPRESSION
    var ch_names = List[String]()
    var ch_types = List[Int]()
    var has_tiles = False
    var tile_x = 0; var tile_y = 0; var level_mode = 0; var rounding_mode = 0

    while pos < n:
        if data[pos] == 0:
            pos += 1
            break
        var nm = _read_cstr(data, pos, n)
        pos = nm[1]
        var tp = _read_cstr(data, pos, n)
        pos = tp[1]
        if pos + 4 > n:
            return img^
        var attr_sz = _le_i32(data, pos)
        pos += 4
        if attr_sz < 0 or pos + attr_sz > n:
            return img^

        if nm[0] == "dataWindow" and attr_sz == 16:
            x_min = _le_i32(data, pos)
            y_min = _le_i32(data, pos + 4)
            x_max = _le_i32(data, pos + 8)
            y_max = _le_i32(data, pos + 12)
        elif nm[0] == "compression" and attr_sz == 1:
            compression = Int(data[pos])
        elif nm[0] == "tiles" and attr_sz == 9:
            has_tiles = True
            tile_x = _le_i32(data, pos)
            tile_y = _le_i32(data, pos + 4)
            var mode_b = Int(data[pos + 8])
            level_mode = mode_b & 0x0F
            rounding_mode = (mode_b >> 4) & 0x0F
        elif nm[0] == "channels":
            var cp = pos
            var c_end = pos + attr_sz
            while cp < c_end:
                if data[cp] == 0:
                    break
                var cn = _read_cstr(data, cp, c_end)
                cp = cn[1]
                if cp + 16 > c_end:
                    return img^
                var ptype = _le_i32(data, cp)
                var xs = _le_i32(data, cp + 8)
                var ys = _le_i32(data, cp + 12)
                if ptype < 0 or ptype > 2 or xs <= 0 or ys <= 0:
                    return img^
                ch_names.append(cn[0])
                ch_types.append(ptype)
                cp += 16

        pos += attr_sz

    var width = x_max - x_min + 1
    var height = y_max - y_min + 1
    if (
        width <= 0
        or height <= 0
        or len(ch_names) == 0
        or width > 33_554_432
        or height > 33_554_432
        or width * height > 33_554_432
    ):
        return img^
    if (
        compression != EXR_NO_COMPRESSION
        and compression != EXR_RLE_COMPRESSION
        and compression != EXR_ZIPS_COMPRESSION
        and compression != EXR_ZIP_COMPRESSION
        and compression != EXR_PIZ_COMPRESSION
        and compression != EXR_DWAA_COMPRESSION
        and compression != EXR_DWAB_COMPRESSION
    ):
        return img^

    if is_tiled:
        if not has_tiles or tile_x <= 0 or tile_y <= 0 or level_mode > 2 or rounding_mode > 1:
            return img^
        if tile_x > 33_554_432 or tile_y > 33_554_432 or tile_x * tile_y > 33_554_432:
            return img^

    var ch_byte_sizes = List[Int](capacity=len(ch_types))
    var ch_rgb_role = List[Int](capacity=len(ch_names))  # 0=R, 1=G, 2=B, 3=Y, -1=other
    var has_rgb = False
    for ci in range(len(ch_names)):
        if ch_names[ci] == "R" or ch_names[ci] == "G" or ch_names[ci] == "B":
            has_rgb = True
            break
    for ci in range(len(ch_names)):
        var bsz = 2 if ch_types[ci] == EXR_PIXEL_HALF else 4
        ch_byte_sizes.append(bsz)
        if ch_names[ci] == "R":
            ch_rgb_role.append(0)
        elif ch_names[ci] == "G":
            ch_rgb_role.append(1)
        elif ch_names[ci] == "B":
            ch_rgb_role.append(2)
        elif ch_names[ci] == "Y" and not has_rgb:
            ch_rgb_role.append(3)
        else:
            ch_rgb_role.append(-1)

    img.width = width
    img.height = height
    img.pixels = List[Vec3](length=width * height, fill=Vec3(0.0, 0.0, 0.0))

    if is_tiled:
        var ntx0 = (width + tile_x - 1) // tile_x
        var nty0 = (height + tile_y - 1) // tile_y
        var expected_l0_tiles = ntx0 * nty0
        var n_chunks = 0
        if level_mode == 0:
            n_chunks = expected_l0_tiles
        elif level_mode == 1:
            var nx_lv = _exr_round_log2(width, rounding_mode) + 1
            var ny_lv = _exr_round_log2(height, rounding_mode) + 1
            var n_lv = max(nx_lv, ny_lv)
            for lv in range(n_lv):
                var lw = _exr_level_size(width, lv, rounding_mode)
                var lh = _exr_level_size(height, lv, rounding_mode)
                n_chunks += ((lw + tile_x - 1) // tile_x) * ((lh + tile_y - 1) // tile_y)
        else:
            var nx_lv = _exr_round_log2(width, rounding_mode) + 1
            var ny_lv = _exr_round_log2(height, rounding_mode) + 1
            for ly in range(ny_lv):
                var lh = _exr_level_size(height, ly, rounding_mode)
                var nty = (lh + tile_y - 1) // tile_y
                for lx in range(nx_lv):
                    var lw = _exr_level_size(width, lx, rounding_mode)
                    n_chunks += ((lw + tile_x - 1) // tile_x) * nty

        if n_chunks <= 0 or pos + n_chunks * 8 > n:
            return HdrImage()
        var offsets = List[Int](capacity=n_chunks)
        for c_idx in range(n_chunks):
            offsets.append(_le_u64(data, pos + c_idx * 8))

        var seen_l0 = List[Bool](length=expected_l0_tiles, fill=False)
        var l0_decoded = 0
        for c_idx in range(n_chunks):
            var c_off = offsets[c_idx]
            if c_off < 0 or c_off + 20 > n:
                return HdrImage()
            var tx_idx = _le_i32(data, c_off)
            var ty_idx = _le_i32(data, c_off + 4)
            var lx = _le_i32(data, c_off + 8)
            var ly = _le_i32(data, c_off + 12)
            var packed_sz = _le_i32(data, c_off + 16)
            var data_start = c_off + 20
            if packed_sz < 0 or data_start + packed_sz > n:
                return HdrImage()
            if lx < 0 or ly < 0:
                return HdrImage()
            if lx != 0 or ly != 0:
                continue
            if tx_idx < 0 or ty_idx < 0 or tx_idx >= ntx0 or ty_idx >= nty0:
                return HdrImage()
            var tx0 = tx_idx * tile_x
            var ty0 = ty_idx * tile_y
            var tw = min(tile_x, width - tx0)
            var th = min(tile_y, height - ty0)
            if not _exr_decode_chunk_into(
                data, data_start, packed_sz, compression, tw, th, ch_names, ch_types, ch_byte_sizes, ch_rgb_role, tx0, ty0, img
            ):
                return HdrImage()
            var tid = ty_idx * ntx0 + tx_idx
            if not seen_l0[tid]:
                seen_l0[tid] = True
                l0_decoded += 1

        if l0_decoded != expected_l0_tiles:
            return HdrImage()

        img.valid = True
        return img^

    var lines_per_block = 1
    if compression == EXR_ZIP_COMPRESSION:
        lines_per_block = 16
    elif compression == EXR_PIZ_COMPRESSION or compression == EXR_DWAA_COMPRESSION:
        lines_per_block = 32
    elif compression == EXR_DWAB_COMPRESSION:
        lines_per_block = 256

    var num_blocks = (height + lines_per_block - 1) // lines_per_block
    if pos + num_blocks * 8 > n:
        return img^

    var offsets = List[Int](capacity=num_blocks)
    for b_idx in range(num_blocks):
        offsets.append(_le_u64(data, pos + b_idx * 8))

    var seen_rows = List[Bool](length=height, fill=False)
    var blocks_decoded = 0
    var rows_decoded = 0
    for b_idx in range(num_blocks):
        var c_off = offsets[b_idx]
        if c_off < 0 or c_off + 8 > n:
            return HdrImage()
        var y_coord = _le_i32(data, c_off)
        var packed_sz = _le_i32(data, c_off + 4)
        var data_start = c_off + 8
        if packed_sz < 0 or data_start + packed_sz > n:
            return HdrImage()

        var rel_y0 = y_coord - y_min
        var block_lines = min(lines_per_block, height - rel_y0)
        if rel_y0 < 0 or rel_y0 >= height or block_lines <= 0:
            return HdrImage()
        for ry in range(rel_y0, rel_y0 + block_lines):
            if seen_rows[ry]:
                return HdrImage()
            seen_rows[ry] = True
            rows_decoded += 1
        if not _exr_decode_chunk_into(
            data, data_start, packed_sz, compression, width, block_lines, ch_names, ch_types, ch_byte_sizes, ch_rgb_role, 0, rel_y0, img
        ):
            return HdrImage()
        blocks_decoded += 1

    if blocks_decoded != num_blocks or rows_decoded != height:
        return HdrImage()

    img.valid = True
    return img^


def decode_hdr_or_exr(data: List[UInt8]) -> HdrImage:
    """Auto-detects OpenEXR `.exr` (`0x01312F76`) vs Radiance `.hdr` (`#?`) from raw bytes."""
    if len(data) >= 4 and _le_i32(data, 0) == EXR_MAGIC:
        return decode_exr_fp16(data)
    return decode_radiance_hdr(data)
