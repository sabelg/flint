"""Native PNG/DEFLATE decoder and PNG screenshot encoder."""
from flint.core.build_info import build_stamp
from flint.platform.os import write_bytes


struct NativeImage(Copyable, Movable):
    var valid: Bool
    var width: Int
    var height: Int
    var pixels: List[UInt32]

    def __init__(out self):
        self.valid = False
        self.width = 0
        self.height = 0
        self.pixels = List[UInt32]()


def be32(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 4 > len(data):
        return -1
    return (Int(data[offset]) << 24) | (Int(data[offset + 1]) << 16) | (Int(data[offset + 2]) << 8) | Int(data[offset + 3])


struct BitReader(Copyable, Movable):
    var data: List[UInt8]
    var position: Int
    var bit: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int):
        self.data = data.copy()
        self.position = start
        self.bit = 0
        self.valid = True

    def read_bit(mut self) -> Int:
        if self.position >= len(self.data):
            self.valid = False
            return 0
        var result = (Int(self.data[self.position]) >> self.bit) & 1
        self.bit += 1
        if self.bit == 8:
            self.bit = 0
            self.position += 1
        return result

    def read_bits(mut self, count: Int) -> Int:
        var result = 0
        for index in range(count):
            result |= self.read_bit() << index
        return result

    def align(mut self):
        if self.bit != 0:
            self.bit = 0
            self.position += 1


struct Huffman(Copyable, Movable):
    var counts: List[Int]
    var symbols: List[Int]
    var valid: Bool

    def __init__(out self):
        self.counts = List[Int](length=16, fill=0)
        self.symbols = List[Int]()
        self.valid = True

    def build(mut self, lengths: List[UInt8]):
        self.counts = List[Int](length=16, fill=0)
        for length in lengths:
            if length > 15:
                self.valid = False
                return
            self.counts[Int(length)] += 1
        self.counts[0] = 0
        var offsets = List[Int](length=17, fill=0)
        for index in range(1, 16):
            offsets[index + 1] = offsets[index] + self.counts[index]
        self.symbols = List[Int](length=len(lengths), fill=0)
        for symbol in range(len(lengths)):
            var length = Int(lengths[symbol])
            if length > 0:
                self.symbols[offsets[length]] = symbol
                offsets[length] += 1

    def decode(self, mut reader: BitReader) -> Int:
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


def length_base(code: Int) -> Int:
    var values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
    return values[max(0, min(28, code - 257))]


def length_extra(code: Int) -> Int:
    var values = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
    return values[max(0, min(28, code - 257))]


def distance_base(code: Int) -> Int:
    var values = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577]
    return values[max(0, min(29, code))]


def distance_extra(code: Int) -> Int:
    var values = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
    return values[max(0, min(29, code))]


def inflate_zlib(data: List[UInt8], start: Int) -> List[UInt8]:
    var output = List[UInt8]()
    if start + 6 > len(data):
        return output^
    var reader = BitReader(data, start + 2)
    var final = 0
    while final == 0 and reader.valid:
        final = reader.read_bit()
        var kind = reader.read_bits(2)
        if kind == 0:
            reader.align()
            if reader.position + 4 > len(data):
                break
            var length = Int(data[reader.position]) | (Int(data[reader.position + 1]) << 8)
            var inverse = Int(data[reader.position + 2]) | (Int(data[reader.position + 3]) << 8)
            reader.position += 4
            if (length ^ 65535) != inverse or reader.position + length > len(data):
                reader.valid = False
                break
            for _ in range(length):
                output.append(data[reader.position])
                reader.position += 1
            continue
        if kind == 3:
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
                for _ in range(length):
                    output.append(output[len(output) - backwards])
            else:
                reader.valid = False
    if not reader.valid:
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


def decode_png(data: List[UInt8], maximum_size: Int = 1024) -> NativeImage:
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
    if width <= 0 or height <= 0 or width * height > 40000000 or bit_depth != 8 or (color_type != 2 and color_type != 6) or interlace != 0:
        return image^
    var channels = 4 if color_type == 6 else 3
    var compressed = List[UInt8]()
    var offset = 8
    var finished = False
    while offset + 12 <= len(data):
        var length = be32(data, offset)
        if length < 0 or offset + length + 12 > len(data):
            return image^
        var kind = be32(data, offset + 4)
        if kind == 0x49444154:
            for index in range(length):
                compressed.append(data[offset + 8 + index])
        elif kind == 0x49454E44:
            finished = True
            break
        offset += length + 12
    if not finished:
        return image^
    var raw = inflate_zlib(compressed, 0)
    var stride = width * channels
    if len(raw) != (stride + 1) * height:
        return image^
    var scanlines = List[UInt8](length=stride * height, fill=0)
    for y in range(height):
        var filter = Int(raw[y * (stride + 1)])
        for x in range(stride):
            var encoded = Int(raw[y * (stride + 1) + 1 + x])
            var left = Int(scanlines[y * stride + x - channels]) if x >= channels else 0
            var above = Int(scanlines[(y - 1) * stride + x]) if y > 0 else 0
            var diagonal = Int(scanlines[(y - 1) * stride + x - channels]) if y > 0 and x >= channels else 0
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
            scanlines[y * stride + x] = UInt8((encoded + predictor) & 255)
    var scale = 1
    while (width + scale - 1) // scale > maximum_size or (height + scale - 1) // scale > maximum_size:
        scale *= 2
    image.width = (width + scale - 1) // scale
    image.height = (height + scale - 1) // scale
    image.pixels = List[UInt32](length=image.width * image.height, fill=0xFFFFFFFF)
    for y in range(image.height):
        for x in range(image.width):
            var source = min(height - 1, y * scale) * stride + min(width - 1, x * scale) * channels
            var alpha = UInt32(scanlines[source + 3]) if channels == 4 else UInt32(255)
            image.pixels[y * image.width + x] = (alpha << 24) | UInt32(scanlines[source]) << 16 | UInt32(scanlines[source + 1]) << 8 | UInt32(scanlines[source + 2])
    image.valid = True
    return image^


def crc32(data: List[UInt8], start: Int, end: Int) -> UInt32:
    var crc = UInt32(0xFFFFFFFF)
    for index in range(start, end):
        crc = crc ^ UInt32(data[index])
        for _ in range(8):
            crc = (crc >> 1) ^ (UInt32(0xEDB88320) if (crc & 1) else UInt32(0))
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
    var data = encode_png(pixels, width, height, ("Sharddepth " + build_stamp()) if stamped else String())
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
