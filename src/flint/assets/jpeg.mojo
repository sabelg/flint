"""Native baseline sequential JPEG decoder for embedded SHARDDEPTH textures."""
from flint.assets.image import NativeImage


def jpeg_be16(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 2 > len(data):
        return -1
    return (Int(data[offset]) << 8) | Int(data[offset + 1])


def jpeg_zigzag(index: Int) -> Int:
    var order = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ]
    return order[index]


struct JpegBits(Copyable, Movable):
    var data: List[UInt8]
    var position: Int
    var current: Int
    var remaining: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int):
        self.data = data.copy()
        self.position = start
        self.current = 0
        self.remaining = 0
        self.valid = True

    def read_bit(mut self) -> Int:
        if self.remaining == 0:
            if self.position >= len(self.data):
                self.valid = False
                return 0
            var value = Int(self.data[self.position])
            self.position += 1
            if value == 255:
                if self.position >= len(self.data):
                    self.valid = False
                    return 0
                var following = Int(self.data[self.position])
                self.position += 1
                while following == 255 and self.position < len(self.data):
                    following = Int(self.data[self.position])
                    self.position += 1
                if following != 0:
                    self.valid = False
                    return 0
            self.current = value
            self.remaining = 8
        self.remaining -= 1
        return (self.current >> self.remaining) & 1

    def read_bits(mut self, count: Int) -> Int:
        var result = 0
        for _ in range(count):
            result = (result << 1) | self.read_bit()
        return result

    def restart(mut self) -> Bool:
        self.remaining = 0
        if self.position + 2 > len(self.data) or Int(self.data[self.position]) != 255:
            self.valid = False
            return False
        var marker = Int(self.data[self.position + 1])
        self.position += 2
        if marker < 208 or marker > 215:
            self.valid = False
            return False
        return True


struct JpegHuffman(Copyable, Movable):
    var minimum: List[Int]
    var maximum: List[Int]
    var offsets: List[Int]
    var symbols: List[Int]
    var valid: Bool

    def __init__(out self):
        self.minimum = List[Int](length=17, fill=0)
        self.maximum = List[Int](length=17, fill=-1)
        self.offsets = List[Int](length=17, fill=0)
        self.symbols = List[Int]()
        self.valid = False

    def build(mut self, counts: List[Int], mut symbols: List[Int]):
        self.minimum = List[Int](length=17, fill=0)
        self.maximum = List[Int](length=17, fill=-1)
        self.offsets = List[Int](length=17, fill=0)
        self.symbols = symbols.copy()
        var code = 0
        var offset = 0
        for length in range(1, 17):
            self.minimum[length] = code
            self.offsets[length] = offset
            if counts[length] > 0:
                self.maximum[length] = code + counts[length] - 1
            code = (code + counts[length]) << 1
            offset += counts[length]
        self.valid = offset == len(symbols)

    def decode(self, mut bits: JpegBits) -> Int:
        var code = 0
        for length in range(1, 17):
            code = (code << 1) | bits.read_bit()
            if not bits.valid:
                return -1
            if self.maximum[length] >= 0 and code >= self.minimum[length] and code <= self.maximum[length]:
                var index = self.offsets[length] + code - self.minimum[length]
                if index < 0 or index >= len(self.symbols):
                    bits.valid = False
                    return -1
                return self.symbols[index]
        bits.valid = False
        return -1


struct JpegComponent(Copyable, Movable):
    var identifier: Int
    var horizontal: Int
    var vertical: Int
    var quantization: Int
    var dc_table: Int
    var ac_table: Int
    var predictor: Int
    var stride: Int
    var rows: Int
    var samples: List[UInt8]

    def __init__(out self):
        self.identifier = 0
        self.horizontal = 1
        self.vertical = 1
        self.quantization = 0
        self.dc_table = 0
        self.ac_table = 0
        self.predictor = 0
        self.stride = 0
        self.rows = 0
        self.samples = List[UInt8]()


def jpeg_extend(mut bits: JpegBits, size: Int) -> Int:
    if size == 0:
        return 0
    var value = bits.read_bits(size)
    if value < (1 << (size - 1)):
        value -= (1 << size) - 1
    return value


def jpeg_level(value: Float32) -> UInt8:
    var shifted = value + 128.0
    if shifted <= 0:
        return 0
    if shifted >= 255:
        return 255
    return UInt8(Int(shifted + 0.5))


def jpeg_cosine_factor(index: Int) -> Float32:
    if index == 0:
        return 1.0
    if index == 1:
        return 0.9807852804032304
    if index == 2:
        return 0.9238795325112867
    if index == 3:
        return 0.8314696123025452
    if index == 4:
        return 0.7071067811865476
    if index == 5:
        return 0.5555702330196022
    if index == 6:
        return 0.3826834323650898
    return 0.1950903220161283


def jpeg_idct_store(mut coefficients: List[Float32], mut samples: List[UInt8], stride: Int, block_x: Int, block_y: Int, mut workspace: List[Float32], ac_count: Int):
    if ac_count == 0:
        var level = jpeg_level(coefficients[0] / 8.0)
        for y in range(8):
            for x in range(8):
                samples[(block_y + y) * stride + block_x + x] = level
        return
    # Fixed cosine factors keep the hot loop independent of transcendental
    # functions and external math libraries.
    for y in range(8):
        for x in range(8):
            var total = Float32(0)
            for u in range(8):
                var angle = ((2 * x + 1) * u) & 31
                var sign = Float32(1)
                var index: Int
                if angle < 8:
                    index = angle
                elif angle < 16:
                    index = 16 - angle
                    sign = -1
                elif angle < 24:
                    index = angle - 16
                    sign = -1
                else:
                    index = 32 - angle
                var weight = jpeg_cosine_factor(index) * sign
                if u == 0:
                    weight *= Float32(0.7071067811865476)
                total += coefficients[y * 8 + u] * weight
            workspace[y * 8 + x] = total
    for y in range(8):
        for x in range(8):
            var total = Float32(0)
            for v in range(8):
                var angle = ((2 * y + 1) * v) & 31
                var sign = Float32(1)
                var index: Int
                if angle < 8:
                    index = angle
                elif angle < 16:
                    index = 16 - angle
                    sign = -1
                elif angle < 24:
                    index = angle - 16
                    sign = -1
                else:
                    index = 32 - angle
                var weight = jpeg_cosine_factor(index) * sign
                if v == 0:
                    weight *= Float32(0.7071067811865476)
                total += workspace[v * 8 + x] * weight
            samples[(block_y + y) * stride + block_x + x] = jpeg_level(total * 0.25)


def jpeg_decode_block(mut bits: JpegBits, mut component: JpegComponent, quantization: List[Int], dc: JpegHuffman, ac: JpegHuffman, mut coefficients: List[Float32], mut workspace: List[Float32], block_x: Int, block_y: Int):
    for index in range(64):
        coefficients[index] = 0
    var dc_size = dc.decode(bits)
    if dc_size < 0 or dc_size > 16:
        bits.valid = False
        return
    component.predictor += jpeg_extend(bits, dc_size)
    coefficients[0] = Float32(component.predictor * quantization[0])
    var scan = 1
    var ac_count = 0
    while scan < 64 and bits.valid:
        var symbol = ac.decode(bits)
        if symbol < 0:
            return
        var run = symbol >> 4
        var size = symbol & 15
        if size == 0:
            if run == 15:
                scan += 16
                continue
            break
        scan += run
        if scan >= 64 or size > 16:
            bits.valid = False
            return
        var natural = jpeg_zigzag(scan)
        coefficients[natural] = Float32(jpeg_extend(bits, size) * quantization[natural])
        ac_count += 1
        scan += 1
    if bits.valid:
        jpeg_idct_store(coefficients, component.samples, component.stride, block_x, block_y, workspace, ac_count)


def jpeg_channel(value: Float32) -> UInt8:
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return UInt8(Int(value + 0.5))


def decode_jpeg(data: List[UInt8], maximum_size: Int = 1024) -> NativeImage:
    var image = NativeImage()
    if len(data) < 16 or Int(data[0]) != 255 or Int(data[1]) != 216:
        return image^
    var width = 0
    var height = 0
    var components = List[JpegComponent]()
    var quantization = List[List[Int]]()
    var dc_tables = List[JpegHuffman]()
    var ac_tables = List[JpegHuffman]()
    for _ in range(4):
        quantization.append(List[Int](length=64, fill=0))
        dc_tables.append(JpegHuffman())
        ac_tables.append(JpegHuffman())
    var maximum_horizontal = 1
    var maximum_vertical = 1
    var restart_interval = 0
    var offset = 2
    while offset + 4 <= len(data):
        while offset < len(data) and Int(data[offset]) == 255:
            offset += 1
        if offset >= len(data):
            break
        var marker = Int(data[offset])
        offset += 1
        if marker == 217:
            break
        if marker == 1 or (marker >= 208 and marker <= 215):
            continue
        var length = jpeg_be16(data, offset)
        if length < 2 or offset + length > len(data):
            return image^
        var start = offset + 2
        var end = offset + length
        if marker == 219:  # DQT
            var position = start
            while position < end:
                var information = Int(data[position])
                position += 1
                var precision = information >> 4
                var table = information & 15
                if table > 3 or precision > 1 or position + (128 if precision else 64) > end:
                    return image^
                for scan in range(64):
                    var natural = jpeg_zigzag(scan)
                    if precision:
                        quantization[table][natural] = jpeg_be16(data, position)
                        position += 2
                    else:
                        quantization[table][natural] = Int(data[position])
                        position += 1
        elif marker == 196:  # DHT
            var position = start
            while position + 17 <= end:
                var information = Int(data[position])
                position += 1
                var counts = List[Int](length=17, fill=0)
                var total = 0
                for index in range(1, 17):
                    counts[index] = Int(data[position + index - 1])
                    total += counts[index]
                position += 16
                if total <= 0 or position + total > end:
                    return image^
                var symbols = List[Int]()
                for index in range(total):
                    symbols.append(Int(data[position + index]))
                position += total
                var table = information & 15
                if table > 3:
                    return image^
                if information >> 4 == 0:
                    dc_tables[table].build(counts, symbols)
                elif information >> 4 == 1:
                    ac_tables[table].build(counts, symbols)
                else:
                    return image^
        elif marker == 221:  # DRI
            if end - start != 2:
                return image^
            restart_interval = jpeg_be16(data, start)
        elif marker == 192:  # SOF0, baseline sequential
            if len(components) > 0 or end - start < 6 or Int(data[start]) != 8:
                return image^
            height = jpeg_be16(data, start + 1)
            width = jpeg_be16(data, start + 3)
            var count = Int(data[start + 5])
            if width <= 0 or height <= 0 or width * height > 40000000 or (count != 1 and count != 3) or start + 6 + count * 3 != end:
                return image^
            for index in range(count):
                var component = JpegComponent()
                var position = start + 6 + index * 3
                component.identifier = Int(data[position])
                component.horizontal = Int(data[position + 1]) >> 4
                component.vertical = Int(data[position + 1]) & 15
                component.quantization = Int(data[position + 2])
                if component.horizontal <= 0 or component.vertical <= 0 or component.quantization > 3:
                    return image^
                maximum_horizontal = max(maximum_horizontal, component.horizontal)
                maximum_vertical = max(maximum_vertical, component.vertical)
                components.append(component^)
        elif marker == 218:  # SOS
            if width <= 0 or height <= 0 or len(components) == 0 or end <= start:
                return image^
            var count = Int(data[start])
            if count != len(components) or start + 1 + count * 2 + 3 != end:
                return image^
            for index in range(count):
                var identifier = Int(data[start + 1 + index * 2])
                var tables = Int(data[start + 2 + index * 2])
                if tables >> 4 > 3 or tables & 15 > 3:
                    return image^
                var found = False
                for component_index in range(len(components)):
                    if components[component_index].identifier == identifier:
                        components[component_index].dc_table = tables >> 4
                        components[component_index].ac_table = tables & 15
                        components[component_index].predictor = 0
                        found = True
                if not found:
                    return image^
            var mcu_width = (width + maximum_horizontal * 8 - 1) // (maximum_horizontal * 8)
            var mcu_height = (height + maximum_vertical * 8 - 1) // (maximum_vertical * 8)
            for component_index in range(len(components)):
                var horizontal_blocks = mcu_width * components[component_index].horizontal
                var vertical_blocks = mcu_height * components[component_index].vertical
                components[component_index].stride = horizontal_blocks * 8
                components[component_index].rows = vertical_blocks * 8
                components[component_index].samples = List[UInt8](length=components[component_index].stride * components[component_index].rows, fill=0)
            var bits = JpegBits(data, end)
            var coefficients = List[Float32](length=64, fill=0)
            var workspace = List[Float32](length=64, fill=0)
            for mcu_y in range(mcu_height):
                for mcu_x in range(mcu_width):
                    var mcu = mcu_y * mcu_width + mcu_x
                    if restart_interval > 0 and mcu > 0 and mcu % restart_interval == 0:
                        if not bits.restart():
                            return image^
                        for component_index in range(len(components)):
                            components[component_index].predictor = 0
                    for scan_index in range(count):
                        var identifier = Int(data[start + 1 + scan_index * 2])
                        var component_index = -1
                        for candidate in range(len(components)):
                            if components[candidate].identifier == identifier:
                                component_index = candidate
                        if component_index < 0:
                            return image^
                        for vertical in range(components[component_index].vertical):
                            for horizontal in range(components[component_index].horizontal):
                                var block_x = (mcu_x * components[component_index].horizontal + horizontal) * 8
                                var block_y = (mcu_y * components[component_index].vertical + vertical) * 8
                                var dc_index = components[component_index].dc_table
                                var ac_index = components[component_index].ac_table
                                var quantization_index = components[component_index].quantization
                                jpeg_decode_block(bits, components[component_index], quantization[quantization_index], dc_tables[dc_index], ac_tables[ac_index], coefficients, workspace, block_x, block_y)
                                if not bits.valid:
                                    return image^
            var scale = 1
            while (width + scale - 1) // scale > maximum_size or (height + scale - 1) // scale > maximum_size:
                scale *= 2
            image.width = (width + scale - 1) // scale
            image.height = (height + scale - 1) // scale
            image.pixels = List[UInt32](length=image.width * image.height, fill=0xFF000000)
            for y in range(image.height):
                var source_y = min(height - 1, y * scale)
                for x in range(image.width):
                    var source_x = min(width - 1, x * scale)
                    if len(components) == 1:
                        var sample = components[0].samples[(source_y * components[0].vertical // maximum_vertical) * components[0].stride + source_x * components[0].horizontal // maximum_horizontal]
                        image.pixels[y * image.width + x] = 0xFF000000 | UInt32(sample) << 16 | UInt32(sample) << 8 | UInt32(sample)
                    else:
                        var y_sample = Float32(components[0].samples[(source_y * components[0].vertical // maximum_vertical) * components[0].stride + source_x * components[0].horizontal // maximum_horizontal])
                        var cb = Float32(components[1].samples[(source_y * components[1].vertical // maximum_vertical) * components[1].stride + source_x * components[1].horizontal // maximum_horizontal]) - 128.0
                        var cr = Float32(components[2].samples[(source_y * components[2].vertical // maximum_vertical) * components[2].stride + source_x * components[2].horizontal // maximum_horizontal]) - 128.0
                        var red = jpeg_channel(y_sample + 1.402 * cr)
                        var green = jpeg_channel(y_sample - 0.344136 * cb - 0.714136 * cr)
                        var blue = jpeg_channel(y_sample + 1.772 * cb)
                        image.pixels[y * image.width + x] = 0xFF000000 | UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
            image.valid = True
            return image^
        offset = end
    return image^
