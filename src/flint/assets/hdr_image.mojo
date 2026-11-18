"""Phase 2 (`V17`): Pure-Mojo Radiance `.hdr` (`RGBE`) & OpenEXR `.exr` (`FP16`/`FP32`) Codec (`hdr_image.mojo`).

Replaces external C/C++ `OpenEXR`, `tinyexr`, and `stb_image` HDR loaders with 100%
pure Mojo 1.2 encoders and decoders for:
  1. Greg Ward Radiance `.hdr` (`#?RADIANCE` / `#?RGBE`, adaptive 4-channel RLE & flat scanlines).
  2. ILM OpenEXR v2 `.exr` (`magic = 0x01312F76`, `HALF` 16-bit & `FLOAT` 32-bit channels,
     `NO_COMPRESSION = 0`, `ZIPS_COMPRESSION = 2`, and `ZIP_COMPRESSION = 3` with OpenEXR
     predictor + even/odd byte-interleave reconstruction).
  3. Equirectangular `2:1` HDRI panorama bilinear sampler feeding `aaa_ibl.mojo` (`BC6H` + `SH9`).
"""
from std.math import atan2, cos, exp2, floor, log2, max, min, sin, sqrt
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
comptime EXR_ZIPS_COMPRESSION: Int = 2
comptime EXR_ZIP_COMPRESSION: Int = 3


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


def decode_exr_fp16(data: List[UInt8]) -> HdrImage:
    """Decodes OpenEXR v2 scanline `.exr` (`HALF` & `FLOAT`, `NO_COMPRESSION` / `ZIPS` / `ZIP`)."""
    var img = HdrImage()
    var n = len(data)
    if n < 32 or _le_i32(data, 0) != EXR_MAGIC:
        return img^
    var ver_word = _le_i32(data, 4)
    if (ver_word & 0xFF) != 2 or (ver_word & 0x200) != 0:  # reject tiled bit 0x200
        return img^

    var pos = 8
    var x_min = 0; var y_min = 0; var x_max = -1; var y_max = -1
    var compression = EXR_NO_COMPRESSION
    var ch_names = List[String]()
    var ch_types = List[Int]()

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
                ch_names.append(cn[0])
                ch_types.append(ptype)
                cp += 16

        pos += attr_sz

    var width = x_max - x_min + 1
    var height = y_max - y_min + 1
    if width <= 0 or height <= 0 or len(ch_names) == 0 or width * height > 33_554_432:
        return img^
    if compression != EXR_NO_COMPRESSION and compression != EXR_ZIPS_COMPRESSION and compression != EXR_ZIP_COMPRESSION:
        return img^

    var lines_per_block = 16 if compression == EXR_ZIP_COMPRESSION else 1
    var num_blocks = (height + lines_per_block - 1) // lines_per_block
    if pos + num_blocks * 8 > n:
        return img^

    var offsets = List[Int](capacity=num_blocks)
    for b_idx in range(num_blocks):
        offsets.append(_le_u64(data, pos + b_idx * 8))

    var bytes_per_pixel_row = 0
    var ch_byte_sizes = List[Int](capacity=len(ch_types))
    var ch_rgb_role = List[Int](capacity=len(ch_names))  # 0=R, 1=G, 2=B, -1=other
    for ci in range(len(ch_names)):
        var bsz = 2 if ch_types[ci] == EXR_PIXEL_HALF else 4
        ch_byte_sizes.append(bsz)
        bytes_per_pixel_row += bsz * width
        if ch_names[ci] == "R":
            ch_rgb_role.append(0)
        elif ch_names[ci] == "G":
            ch_rgb_role.append(1)
        elif ch_names[ci] == "B":
            ch_rgb_role.append(2)
        else:
            ch_rgb_role.append(-1)

    img.width = width
    img.height = height
    img.pixels = List[Vec3](length=width * height, fill=Vec3(0.0, 0.0, 0.0))

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
        if rel_y0 < 0 or block_lines <= 0:
            continue
        var expected_raw = block_lines * bytes_per_pixel_row

        var raw_block: List[UInt8]
        if compression != EXR_NO_COMPRESSION and packed_sz < expected_raw:
            var inflated = inflate_zlib(data, data_start, expected_raw)
            if len(inflated) != expected_raw:
                return HdrImage()
            raw_block = _exr_undo_predictor_and_reorder(inflated)
        else:
            if packed_sz < expected_raw:
                return HdrImage()
            raw_block = List[UInt8](length=expected_raw, fill=0)
            for k in range(expected_raw):
                raw_block[k] = data[data_start + k]

        var r_cursor = 0
        for ly in range(block_lines):
            var dst_y = rel_y0 + ly
            for ci in range(len(ch_names)):
                var role = ch_rgb_role[ci]
                var ptype = ch_types[ci]
                var bsz = ch_byte_sizes[ci]
                if role < 0:
                    r_cursor += bsz * width
                    continue
                for x in range(width):
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
                    var cur_px = img.pixels[dst_y * width + x]
                    if role == 0:
                        img.pixels[dst_y * width + x] = Vec3(max(Float32(0.0), val_f), cur_px.y, cur_px.z)
                    elif role == 1:
                        img.pixels[dst_y * width + x] = Vec3(cur_px.x, max(Float32(0.0), val_f), cur_px.z)
                    else:
                        img.pixels[dst_y * width + x] = Vec3(cur_px.x, cur_px.y, max(Float32(0.0), val_f))

    img.valid = True
    return img^


def decode_hdr_or_exr(data: List[UInt8]) -> HdrImage:
    """Auto-detects OpenEXR `.exr` (`0x01312F76`) vs Radiance `.hdr` (`#?`) from raw bytes."""
    if len(data) >= 4 and _le_i32(data, 0) == EXR_MAGIC:
        return decode_exr_fp16(data)
    return decode_radiance_hdr(data)
