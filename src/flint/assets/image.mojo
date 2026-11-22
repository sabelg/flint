"""Native PNG/DEFLATE decoder and PNG screenshot encoder."""
from std.ffi import external_call
from flint.core.build_info import build_stamp
from flint.platform.os import write_bytes


struct NativeImage(Copyable, Movable):
    var valid: Bool
    var width: Int
    var height: Int
    var bit_depth: Int
    var pixels: List[UInt32]
    var pixels_u16: List[UInt16]

    def __init__(out self):
        self.valid = False
        self.width = 0
        self.height = 0
        self.bit_depth = 8
        self.pixels = List[UInt32]()
        self.pixels_u16 = List[UInt16]()


def be32(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 4 > len(data):
        return -1
    return (Int(data[offset]) << 24) | (Int(data[offset + 1]) << 16) | (Int(data[offset + 2]) << 8) | Int(data[offset + 3])


struct BitReader(Copyable, Movable):
    var ptr: Int
    var length: Int
    var position: Int
    var bit_buf: UInt64
    var bit_count: Int
    var valid: Bool

    def __init__(out self, ref data: List[UInt8], start: Int, end: Int = -1):
        self.ptr = Int(data.unsafe_ptr())
        var lim = len(data) if end < 0 else min(len(data), end)
        self.length = max(start, lim) if start >= 0 else 0
        self.position = max(0, start)
        self.bit_buf = 0
        self.bit_count = 0
        self.valid = True

    @always_inline
    def refill(mut self):
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        while self.bit_count <= 48 and self.position < self.length:
            self.bit_buf |= UInt64(p[unsafe_offset=self.position]) << UInt64(self.bit_count)
            self.position += 1
            self.bit_count += 8

    @always_inline
    def read_bit(mut self) -> Int:
        if self.bit_count == 0:
            self.refill()
            if self.bit_count == 0:
                self.valid = False
                return 0
        var result = Int(self.bit_buf & 1)
        self.bit_buf >>= 1
        self.bit_count -= 1
        return result

    @always_inline
    def read_bits(mut self, count: Int) -> Int:
        if count == 0:
            return 0
        if self.bit_count < count:
            self.refill()
            if self.bit_count < count:
                var fallback = 0
                for index in range(count):
                    fallback |= self.read_bit() << index
                return fallback
        var mask = (UInt64(1) << UInt64(count)) - 1
        var result = Int(self.bit_buf & mask)
        self.bit_buf >>= UInt64(count)
        self.bit_count -= count
        return result

    def align(mut self):
        var drop = self.bit_count & 7
        if drop != 0:
            self.bit_buf >>= UInt64(drop)
            self.bit_count -= drop
        var rewind = self.bit_count >> 3
        self.position -= rewind
        self.bit_buf = 0
        self.bit_count = 0


struct Huffman(Copyable, Movable):
    var counts: Array[Int, 16]
    var symbols: List[Int]
    var fast_sym: Array[Int16, 512]
    var fast_len: Array[UInt8, 512]
    var valid: Bool

    def __init__(out self):
        self.counts = Array[Int, 16](fill=0)
        self.symbols = List[Int]()
        self.fast_sym = Array[Int16, 512](fill=-1)
        self.fast_len = Array[UInt8, 512](fill=0)
        self.valid = True

    def build(mut self, lengths: List[UInt8]):
        self.counts = Array[Int, 16](fill=0)
        self.fast_sym = Array[Int16, 512](fill=-1)
        self.fast_len = Array[UInt8, 512](fill=0)
        for length in lengths:
            if length > 15:
                self.valid = False
                return
            self.counts[Int(length)] += 1
        self.counts[0] = 0
        var offsets = Array[Int, 17](fill=0)
        var first_code = Array[Int, 16](fill=0)
        var code_acc = 0
        for index in range(1, 16):
            offsets[index + 1] = offsets[index] + self.counts[index]
            code_acc = (code_acc + self.counts[index - 1]) << 1
            first_code[index] = code_acc
        var next_code = first_code^
        self.symbols = List[Int](length=len(lengths), fill=0)
        for symbol in range(len(lengths)):
            var length = Int(lengths[symbol])
            if length > 0:
                self.symbols[offsets[length]] = symbol
                offsets[length] += 1
                var c = next_code[length]
                next_code[length] = c + 1
                if length <= 9:
                    # Bit-reverse `c` of `length` bits and replicate across upper (9 - length) bits
                    var rev = 0
                    for b in range(length):
                        rev = (rev << 1) | ((c >> b) & 1)
                    var span = 1 << length
                    var entry = rev
                    while entry < 512:
                        self.fast_sym[entry] = Int16(symbol)
                        self.fast_len[entry] = UInt8(length)
                        entry += span

    @always_inline
    def decode(self, mut reader: BitReader) -> Int:
        if reader.bit_count < 15:
            reader.refill()
        if reader.bit_count >= 9:
            var idx9 = Int(reader.bit_buf & 511)
            var flen = Int(self.fast_len[idx9])
            if flen > 0:
                reader.bit_buf >>= UInt64(flen)
                reader.bit_count -= flen
                return Int(self.fast_sym[idx9])
        var code = 0
        var first = 0
        var index = 0
        for length in range(1, 16):
            code |= reader.read_bit()
            var count = self.counts[length]
            if code >= first and code - first < count:
                return self.symbols[index + code - first]
            index += count
            first = (first + count) << 1
            code <<= 1
        reader.valid = False
        return -1


def fixed_tables() -> Tuple[Huffman, Huffman]:
    var literal_lengths = List[UInt8](length=288, fill=8)
    for index in range(144, 256):
        literal_lengths[index] = 9
    for index in range(256, 280):
        literal_lengths[index] = 7
    var distances = List[UInt8](length=32, fill=5)
    var literal = Huffman()
    var distance = Huffman()
    literal.build(literal_lengths)
    distance.build(distances)
    return (literal^, distance^)


@always_inline
def length_base(code: Int) -> Int:
    var values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
    return values[max(0, min(28, code - 257))]


@always_inline
def length_extra(code: Int) -> Int:
    var values = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
    return values[max(0, min(28, code - 257))]


@always_inline
def distance_base(code: Int) -> Int:
    var values = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577]
    return values[max(0, min(29, code))]


@always_inline
def distance_extra(code: Int) -> Int:
    var values = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
    return values[max(0, min(29, code))]


def inflate_zlib(data: List[UInt8], start: Int, expected_bytes: Int = 0, max_in_bytes: Int = -1) -> List[UInt8]:
    if start < 0 or max_in_bytes < -1 or max_in_bytes == 0 or start > len(data):
        return List[UInt8]()
    if max_in_bytes >= 0 and max_in_bytes > len(data) - start:
        return List[UInt8]()
    var in_end = len(data) if max_in_bytes < 0 else start + max_in_bytes
    if in_end - start < 6:
        return List[UInt8]()
    var cmf = Int(data[start])
    var flg = Int(data[start + 1])
    if (cmf & 0x0F) != 8 or (cmf >> 4) > 7 or (flg & 0x20) != 0 or ((cmf * 256 + flg) % 31) != 0:
        return List[UInt8]()
    var max_out = min(expected_bytes, 536_870_912) if expected_bytes > 0 else 134_217_728
    var output = List[UInt8](capacity=max(256, min(1 << 20, max_out)))
    var reader = BitReader(data, start + 2, in_end)
    var final = 0
    while final == 0 and reader.valid:
        final = reader.read_bit()
        var kind = reader.read_bits(2)
        if not reader.valid:
            break
        if kind == 0:
            reader.align()
            if reader.position + 4 > in_end:
                reader.valid = False
                break
            var length = Int(data[reader.position]) | (Int(data[reader.position + 1]) << 8)
            var inverse = Int(data[reader.position + 2]) | (Int(data[reader.position + 3]) << 8)
            reader.position += 4
            if (length ^ 65535) != inverse or reader.position + length > in_end:
                reader.valid = False
                break
            if length > max_out - len(output):
                reader.valid = False
                break
            for _ in range(length):
                output.append(data[reader.position])
                reader.position += 1
            continue
        if kind == 3:
            reader.valid = False
            break
        var literal = Huffman()
        var distance = Huffman()
        if kind == 1:
            var tables = fixed_tables()
            literal = tables[0].copy()
            distance = tables[1].copy()
        else:
            var literal_count = reader.read_bits(5) + 257
            var distance_count = reader.read_bits(5) + 1
            var code_count = reader.read_bits(4) + 4
            var order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
            var code_lengths = List[UInt8](length=19, fill=0)
            for index in range(code_count):
                code_lengths[order[index]] = UInt8(reader.read_bits(3))
            var code_table = Huffman()
            code_table.build(code_lengths)
            var lengths = List[UInt8]()
            while len(lengths) < literal_count + distance_count and reader.valid:
                var symbol = code_table.decode(reader)
                if symbol >= 0 and symbol <= 15:
                    lengths.append(UInt8(symbol))
                elif symbol == 16:
                    if len(lengths) == 0:
                        reader.valid = False
                        break
                    var repeat = reader.read_bits(2) + 3
                    var previous = lengths[len(lengths) - 1]
                    for _ in range(repeat):
                        lengths.append(previous)
                elif symbol == 17:
                    var repeat = reader.read_bits(3) + 3
                    for _ in range(repeat):
                        lengths.append(0)
                elif symbol == 18:
                    var repeat = reader.read_bits(7) + 11
                    for _ in range(repeat):
                        lengths.append(0)
                else:
                    reader.valid = False
            if len(lengths) != literal_count + distance_count:
                reader.valid = False
                break
            var literal_lengths = List[UInt8]()
            var distance_lengths = List[UInt8]()
            for index in range(literal_count):
                literal_lengths.append(lengths[index])
            for index in range(distance_count):
                distance_lengths.append(lengths[literal_count + index])
            literal.build(literal_lengths)
            distance.build(distance_lengths)
        while reader.valid:
            var symbol = literal.decode(reader)
            if symbol < 0:
                break
            if symbol < 256:
                if len(output) >= max_out:
                    reader.valid = False
                    break
                output.append(UInt8(symbol))
            elif symbol == 256:
                break
            elif symbol <= 285:
                var length = length_base(symbol) + reader.read_bits(length_extra(symbol))
                var distance_symbol = distance.decode(reader)
                if distance_symbol < 0 or distance_symbol > 29:
                    reader.valid = False
                    break
                var backwards = distance_base(distance_symbol) + reader.read_bits(distance_extra(distance_symbol))
                if backwards <= 0 or backwards > len(output):
                    reader.valid = False
                    break
                if length > max_out - len(output):
                    reader.valid = False
                    break
                for _ in range(length):
                    output.append(output[len(output) - backwards])
            else:
                reader.valid = False
    if final == 0 or not reader.valid:
        return List[UInt8]()
    reader.align()
    if reader.position + 4 > in_end:
        return List[UInt8]()
    return output^


def paeth(a: Int, b: Int, c: Int) -> Int:
    var estimate = a + b - c
    var da = abs(estimate - a)
    var db = abs(estimate - b)
    var dc = abs(estimate - c)
    if da <= db and da <= dc:
        return a
    return b if db <= dc else c


@always_inline
def _png_unpack_subbyte(row_ptr: Pointer[UInt8, ImmutAnyOrigin], x: Int, bit_depth: Int) -> Int:
    """Unpacks 1-, 2-, or 4-bit MSB-first PNG sample at pixel index `x`."""
    if bit_depth == 8:
        return Int(row_ptr[unsafe_offset=x])
    var ppb = 8 // bit_depth
    var byte_val = Int(row_ptr[unsafe_offset=x // ppb])
    var shift = (ppb - 1 - (x % ppb)) * bit_depth
    var mask = (1 << bit_depth) - 1
    return (byte_val >> shift) & mask


@always_inline
def _png_scale_gray_to_u8(val: Int, bit_depth: Int) -> Int:
    if bit_depth == 8:
        return val
    if bit_depth == 1:
        return 255 if val != 0 else 0
    if bit_depth == 2:
        return val * 85
    if bit_depth == 4:
        return val * 17
    return (val + 128) >> 8 if val < 65408 else 255


def decode_png(data: List[UInt8], maximum_size: Int = 1024) -> NativeImage:
    """Decodes PNG images supporting Grayscale (0), RGB (2), Indexed/Palette (3 + PLTE/tRNS),
    Grayscale+Alpha (4), and RGBA (6) at 1/2/4/8/16-bit depths, with box-filter downscaling."""
    var image = NativeImage()
    if len(data) < 45:
        return image^
    var signature = [137, 80, 78, 71, 13, 10, 26, 10]
    for index in range(8):
        if Int(data[index]) != signature[index]:
            return image^
    var width = be32(data, 16)
    var height = be32(data, 20)
    var bit_depth = Int(data[24])
    var color_type = Int(data[25])
    var interlace = Int(data[28])
    if width <= 0 or height <= 0 or width > 40000000 or height > 40000000 or width > 40000000 // height or interlace != 0:
        return image^

    # Validate PNG spec (color_type, bit_depth) combinations
    var samples_per_pixel: Int
    if color_type == 0:
        if bit_depth != 1 and bit_depth != 2 and bit_depth != 4 and bit_depth != 8 and bit_depth != 16:
            return image^
        samples_per_pixel = 1
    elif color_type == 2:
        if bit_depth != 8 and bit_depth != 16:
            return image^
        samples_per_pixel = 3
    elif color_type == 3:
        if bit_depth != 1 and bit_depth != 2 and bit_depth != 4 and bit_depth != 8:
            return image^
        samples_per_pixel = 1
    elif color_type == 4:
        if bit_depth != 8 and bit_depth != 16:
            return image^
        samples_per_pixel = 2
    elif color_type == 6:
        if bit_depth != 8 and bit_depth != 16:
            return image^
        samples_per_pixel = 4
    else:
        return image^

    var compressed = List[UInt8](capacity=len(data))
    var palette = List[UInt8]()
    var trns = List[UInt8]()
    var offset = 8
    var finished = False
    while offset + 12 <= len(data):
        var length = be32(data, offset)
        if length < 0 or offset + length + 12 > len(data):
            return image^
        var kind = be32(data, offset + 4)
        if kind == 0x49444154:  # IDAT
            var cur = len(compressed)
            compressed.resize(cur + length, 0)
            _ = external_call["memcpy", Int](Int(compressed.unsafe_ptr()) + cur, Int(data.unsafe_ptr()) + offset + 8, length)
        elif kind == 0x504C5445:  # PLTE
            palette.resize(length, 0)
            if length > 0:
                _ = external_call["memcpy", Int](Int(palette.unsafe_ptr()), Int(data.unsafe_ptr()) + offset + 8, length)
        elif kind == 0x74524E53:  # tRNS
            trns.resize(length, 0)
            if length > 0:
                _ = external_call["memcpy", Int](Int(trns.unsafe_ptr()), Int(data.unsafe_ptr()) + offset + 8, length)
        elif kind == 0x49454E44:  # IEND
            finished = True
            break
        offset += length + 12
    if not finished:
        return image^
    if color_type == 3 and (len(palette) < 3 or (len(palette) % 3) != 0):
        return image^

    var bits_per_pixel = samples_per_pixel * bit_depth
    var stride = (width * bits_per_pixel + 7) // 8
    var bpp = max(1, (bits_per_pixel + 7) // 8)
    var raw = inflate_zlib(compressed, 0, (stride + 1) * height, len(compressed))
    if len(raw) != (stride + 1) * height:
        return image^

    var scanlines = List[UInt8](length=stride * height, fill=0)
    for y in range(height):
        var filter = Int(raw[y * (stride + 1)])
        var row_off = y * stride
        var raw_off = y * (stride + 1) + 1
        for x in range(stride):
            var encoded = Int(raw[raw_off + x])
            var left = Int(scanlines[row_off + x - bpp]) if x >= bpp else 0
            var above = Int(scanlines[(y - 1) * stride + x]) if y > 0 else 0
            var diagonal = Int(scanlines[(y - 1) * stride + x - bpp]) if y > 0 and x >= bpp else 0
            var predictor = 0
            if filter == 1:
                predictor = left
            elif filter == 2:
                predictor = above
            elif filter == 3:
                predictor = (left + above) // 2
            elif filter == 4:
                predictor = paeth(left, above, diagonal)
            elif filter != 0:
                return image^
            scanlines[row_off + x] = UInt8((encoded + predictor) & 255)

    # Decode full-resolution ARGB8888 pixels (+ 16-bit channel preservation for 16-bit PNGs)
    var full_pixels = List[UInt32](length=width * height, fill=0xFFFFFFFF)
    var full_u16 = List[UInt16]()
    var keep_u16 = bit_depth == 16
    if keep_u16:
        full_u16 = List[UInt16](length=width * height, fill=UInt16(0))

    var has_trns_gray = color_type == 0 and len(trns) >= 2
    var trns_gray_val = ((Int(trns[0]) << 8) | Int(trns[1])) if has_trns_gray else -1
    var has_trns_rgb = color_type == 2 and len(trns) >= 6
    var trns_r = ((Int(trns[0]) << 8) | Int(trns[1])) if has_trns_rgb else -1
    var trns_g = ((Int(trns[2]) << 8) | Int(trns[3])) if has_trns_rgb else -1
    var trns_b = ((Int(trns[4]) << 8) | Int(trns[5])) if has_trns_rgb else -1
    var pal_count = len(palette) // 3

    var scan_base = Int(scanlines.unsafe_ptr())
    for y in range(height):
        var row_ptr = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=scan_base + y * stride)
        var dst_row = y * width
        if color_type == 6 and bit_depth == 8:
            for x in range(width):
                var s = x * 4
                var r = UInt32(row_ptr[unsafe_offset=s])
                var g = UInt32(row_ptr[unsafe_offset=s + 1])
                var b = UInt32(row_ptr[unsafe_offset=s + 2])
                var a = UInt32(row_ptr[unsafe_offset=s + 3])
                full_pixels[dst_row + x] = (a << 24) | (r << 16) | (g << 8) | b
        elif color_type == 2 and bit_depth == 8:
            for x in range(width):
                var s = x * 3
                var r = Int(row_ptr[unsafe_offset=s])
                var g = Int(row_ptr[unsafe_offset=s + 1])
                var b = Int(row_ptr[unsafe_offset=s + 2])
                var a = 0 if (has_trns_rgb and r == trns_r and g == trns_g and b == trns_b) else 255
                full_pixels[dst_row + x] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
        elif color_type == 0:
            if bit_depth == 16:
                for x in range(width):
                    var s = x * 2
                    var g16 = (Int(row_ptr[unsafe_offset=s]) << 8) | Int(row_ptr[unsafe_offset=s + 1])
                    var g8 = _png_scale_gray_to_u8(g16, 16)
                    var a = 0 if (has_trns_gray and g16 == trns_gray_val) else 255
                    full_pixels[dst_row + x] = (UInt32(a) << 24) | (UInt32(g8) << 16) | (UInt32(g8) << 8) | UInt32(g8)
                    full_u16[dst_row + x] = UInt16(g16)
            else:
                for x in range(width):
                    var gv = _png_unpack_subbyte(row_ptr, x, bit_depth)
                    var g8 = _png_scale_gray_to_u8(gv, bit_depth)
                    var a = 0 if (has_trns_gray and gv == trns_gray_val) else 255
                    full_pixels[dst_row + x] = (UInt32(a) << 24) | (UInt32(g8) << 16) | (UInt32(g8) << 8) | UInt32(g8)
        elif color_type == 3:
            for x in range(width):
                var idx = _png_unpack_subbyte(row_ptr, x, bit_depth)
                if idx >= pal_count:
                    idx = max(0, pal_count - 1)
                var r = UInt32(palette[idx * 3])
                var g = UInt32(palette[idx * 3 + 1])
                var b = UInt32(palette[idx * 3 + 2])
                var a = UInt32(trns[idx]) if idx < len(trns) else UInt32(255)
                full_pixels[dst_row + x] = (a << 24) | (r << 16) | (g << 8) | b
        elif color_type == 4:
            if bit_depth == 16:
                for x in range(width):
                    var s = x * 4
                    var g16 = (Int(row_ptr[unsafe_offset=s]) << 8) | Int(row_ptr[unsafe_offset=s + 1])
                    var a16 = (Int(row_ptr[unsafe_offset=s + 2]) << 8) | Int(row_ptr[unsafe_offset=s + 3])
                    var g8 = _png_scale_gray_to_u8(g16, 16)
                    var a8 = _png_scale_gray_to_u8(a16, 16)
                    full_pixels[dst_row + x] = (UInt32(a8) << 24) | (UInt32(g8) << 16) | (UInt32(g8) << 8) | UInt32(g8)
                    full_u16[dst_row + x] = UInt16(g16)
            else:
                for x in range(width):
                    var s = x * 2
                    var g8 = UInt32(row_ptr[unsafe_offset=s])
                    var a8 = UInt32(row_ptr[unsafe_offset=s + 1])
                    full_pixels[dst_row + x] = (a8 << 24) | (g8 << 16) | (g8 << 8) | g8
        elif color_type == 2 and bit_depth == 16:
            for x in range(width):
                var s = x * 6
                var r16 = (Int(row_ptr[unsafe_offset=s]) << 8) | Int(row_ptr[unsafe_offset=s + 1])
                var g16 = (Int(row_ptr[unsafe_offset=s + 2]) << 8) | Int(row_ptr[unsafe_offset=s + 3])
                var b16 = (Int(row_ptr[unsafe_offset=s + 4]) << 8) | Int(row_ptr[unsafe_offset=s + 5])
                var r8 = _png_scale_gray_to_u8(r16, 16)
                var g8 = _png_scale_gray_to_u8(g16, 16)
                var b8 = _png_scale_gray_to_u8(b16, 16)
                var a8 = 0 if (has_trns_rgb and r16 == trns_r and g16 == trns_g and b16 == trns_b) else 255
                full_pixels[dst_row + x] = (UInt32(a8) << 24) | (UInt32(r8) << 16) | (UInt32(g8) << 8) | UInt32(b8)
                full_u16[dst_row + x] = UInt16(r16)
        elif color_type == 6 and bit_depth == 16:
            for x in range(width):
                var s = x * 8
                var r16 = (Int(row_ptr[unsafe_offset=s]) << 8) | Int(row_ptr[unsafe_offset=s + 1])
                var g16 = (Int(row_ptr[unsafe_offset=s + 2]) << 8) | Int(row_ptr[unsafe_offset=s + 3])
                var b16 = (Int(row_ptr[unsafe_offset=s + 4]) << 8) | Int(row_ptr[unsafe_offset=s + 5])
                var a16 = (Int(row_ptr[unsafe_offset=s + 6]) << 8) | Int(row_ptr[unsafe_offset=s + 7])
                var r8 = _png_scale_gray_to_u8(r16, 16)
                var g8 = _png_scale_gray_to_u8(g16, 16)
                var b8 = _png_scale_gray_to_u8(b16, 16)
                var a8 = _png_scale_gray_to_u8(a16, 16)
                full_pixels[dst_row + x] = (UInt32(a8) << 24) | (UInt32(r8) << 16) | (UInt32(g8) << 8) | UInt32(b8)
                full_u16[dst_row + x] = UInt16(r16)
    _ = len(scanlines)

    # Box-filter downscaling when image exceeds `maximum_size`
    var max_dim = max(1, maximum_size)
    var scale = 1
    while (width + scale - 1) // scale > max_dim or (height + scale - 1) // scale > max_dim:
        scale *= 2
    image.width = (width + scale - 1) // scale
    image.height = (height + scale - 1) // scale
    image.bit_depth = bit_depth
    if scale == 1:
        image.pixels = full_pixels^
        if keep_u16:
            image.pixels_u16 = full_u16^
    else:
        image.pixels = List[UInt32](length=image.width * image.height, fill=0xFFFFFFFF)
        if keep_u16:
            image.pixels_u16 = List[UInt16](length=image.width * image.height, fill=UInt16(0))
        for y in range(image.height):
            var sy0 = y * scale
            var sy1 = min(height, sy0 + scale)
            for x in range(image.width):
                var sx0 = x * scale
                var sx1 = min(width, sx0 + scale)
                var sa = 0
                var sr = 0
                var sg = 0
                var sb = 0
                var su16 = 0
                var cnt = 0
                for sy in range(sy0, sy1):
                    var srow = sy * width
                    for sx in range(sx0, sx1):
                        var px = full_pixels[srow + sx]
                        sa += Int((px >> 24) & 0xFF)
                        sr += Int((px >> 16) & 0xFF)
                        sg += Int((px >> 8) & 0xFF)
                        sb += Int(px & 0xFF)
                        if keep_u16:
                            su16 += Int(full_u16[srow + sx])
                        cnt += 1
                var half = cnt >> 1
                var da = UInt32((sa + half) // cnt)
                var dr = UInt32((sr + half) // cnt)
                var dg = UInt32((sg + half) // cnt)
                var db = UInt32((sb + half) // cnt)
                image.pixels[y * image.width + x] = (da << 24) | (dr << 16) | (dg << 8) | db
                if keep_u16:
                    image.pixels_u16[y * image.width + x] = UInt16((su16 + half) // cnt)
    image.valid = True
    return image^


def crc32(data: List[UInt8], start: Int, end: Int) -> UInt32:
    var lut = [
        UInt32(0x00000000), UInt32(0x1DB71064), UInt32(0x3B6E20C8), UInt32(0x26D930AC),
        UInt32(0x76DC4190), UInt32(0x6B6B51F4), UInt32(0x4DB26158), UInt32(0x5005713C),
        UInt32(0xEDB88320), UInt32(0xF00F9344), UInt32(0xD6D6A3E8), UInt32(0xCB61B38C),
        UInt32(0x9B64C2B0), UInt32(0x86D3D2D4), UInt32(0xA00AE278), UInt32(0xBDBDF21C),
    ]
    var crc = UInt32(0xFFFFFFFF)
    var p = data.unsafe_ptr()
    for index in range(start, end):
        var b = UInt32(p[unsafe_offset=index])
        crc = lut[Int((crc ^ b) & 15)] ^ (crc >> 4)
        crc = lut[Int((crc ^ (b >> 4)) & 15)] ^ (crc >> 4)
    return crc ^ 0xFFFFFFFF


def append_be32(mut data: List[UInt8], value: Int):
    data.append(UInt8((value >> 24) & 255))
    data.append(UInt8((value >> 16) & 255))
    data.append(UInt8((value >> 8) & 255))
    data.append(UInt8(value & 255))


def append_chunk(mut data: List[UInt8], kind: Int, payload: List[UInt8]):
    append_be32(data, len(payload))
    append_be32(data, kind)
    var start = len(data) - 4
    for value in payload:
        data.append(value)
    append_be32(data, Int(crc32(data, start, len(data))))


def encode_png(mut pixels: List[UInt32], width: Int, height: Int, software: String = "", alpha: Bool = False) -> List[UInt8]:
    var output = List[UInt8]()
    var signature = [137, 80, 78, 71, 13, 10, 26, 10]
    for value in signature:
        output.append(UInt8(value))
    var channels = 4 if alpha else 3
    var header = List[UInt8]()
    append_be32(header, width)
    append_be32(header, height)
    header.append(8)
    header.append(UInt8(6) if alpha else UInt8(2))
    header.append(0)
    header.append(0)
    header.append(0)
    append_chunk(output, 0x49484452, header)
    if software.byte_length() > 0:
        # tEXt "Software": which build drew the frame. Readers that only want pixels skip it.
        var text = List[UInt8]()
        for byte in "Software".as_bytes():
            text.append(byte)
        text.append(0)
        for byte in software.as_bytes():
            text.append(byte)
        append_chunk(output, 0x74455874, text)
    var zlib = List[UInt8]()
    zlib.append(0x78)
    zlib.append(0x01)
    var row_bytes = width * channels + 1
    var raw_size = row_bytes * height
    var position = 0
    var adler_a = 1
    var adler_b = 0
    while position < raw_size:
        var block = min(65535, raw_size - position)
        var final = 1 if position + block == raw_size else 0
        zlib.append(UInt8(final))
        zlib.append(UInt8(block & 255))
        zlib.append(UInt8((block >> 8) & 255))
        var inverse = block ^ 65535
        zlib.append(UInt8(inverse & 255))
        zlib.append(UInt8((inverse >> 8) & 255))
        for _ in range(block):
            var value = UInt8(0)
            if position % row_bytes != 0:
                var within = position % row_bytes - 1
                var pixel = pixels[position // row_bytes * width + within // channels]
                var channel = within % channels
                if channel == 3:
                    value = UInt8((pixel >> 24) & 255)
                else:
                    value = UInt8((pixel >> UInt32(16 - channel * 8)) & 255)
            zlib.append(value)
            adler_a = (adler_a + Int(value)) % 65521
            adler_b = (adler_b + adler_a) % 65521
            position += 1
    append_be32(zlib, (adler_b << 16) | adler_a)
    append_chunk(output, 0x49444154, zlib)
    append_chunk(output, 0x49454E44, List[UInt8]())
    return output^


def write_png(path: String, mut pixels: List[UInt32], width: Int, height: Int, stamped: Bool = True) -> Bool:
    """A PNG with the build stamp in a text chunk. References compared over time
    (tests/golden/shots) are written unstamped, so they change only with their pixels."""
    if width <= 0 or height <= 0 or len(pixels) < width * height:
        return False
    var data = encode_png(pixels, width, height, ("Flint " + build_stamp()) if stamped else String())
    return write_bytes(path, data)


def png_text(data: List[UInt8], keyword: String) -> String:
    """The value of a PNG tEXt chunk, or empty."""
    var offset = 8
    while offset + 12 <= len(data):
        var length = (Int(data[offset]) << 24) | (Int(data[offset + 1]) << 16) | (Int(data[offset + 2]) << 8) | Int(data[offset + 3])
        var kind = (Int(data[offset + 4]) << 24) | (Int(data[offset + 5]) << 16) | (Int(data[offset + 6]) << 8) | Int(data[offset + 7])
        if offset + 12 + length > len(data):
            break
        if kind == 0x74455874:
            var start = offset + 8
            var name = List[UInt8]()
            var index = start
            while index < start + length and data[index] != 0:
                name.append(data[index])
                index += 1
            if String(unsafe_from_utf8=name) == keyword and index < start + length:
                var value = List[UInt8]()
                for position in range(index + 1, start + length):
                    value.append(data[position])
                return String(unsafe_from_utf8=value)
        offset += 12 + length
    return String()
