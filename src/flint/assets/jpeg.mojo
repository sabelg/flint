"""Native sequential (`SOF0`/`SOF1`) & progressive (`SOF2`) JPEG decoder/encoder & unified PNG/JPEG/KTX2 auto-decoder."""
from std.math import cos, max, min, sqrt
from flint.assets.image import NativeImage, decode_png
from flint.assets.ktx2 import decode_ktx2, is_ktx2_magic


def jpeg_be16(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 2 > len(data):
        return -1
    return (Int(data[offset]) << 8) | Int(data[offset + 1])


@always_inline
def jpeg_zigzag(index: Int) -> Int:
    comptime _ZIGZAG = SIMD[DType.uint8, 64](
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    )
    return Int(_ZIGZAG[index])


struct JpegBits(Copyable, Movable):
    var ptr: Int
    var length: Int
    var position: Int
    var current: Int
    var remaining: Int
    var valid: Bool

    def __init__(out self, data: List[UInt8], start: Int):
        self.ptr = Int(data.unsafe_ptr())
        self.length = len(data)
        self.position = start
        self.current = 0
        self.remaining = 0
        self.valid = start >= 0 and start <= self.length and self.ptr != 0

    @always_inline
    def _byte(self, idx: Int) -> Int:
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.ptr)
        return Int(p[unsafe_offset=idx])

    def read_bit(mut self) -> Int:
        if self.remaining == 0:
            if self.position >= self.length:
                self.valid = False
                return 0
            var value = self._byte(self.position)
            self.position += 1
            if value == 255:
                if self.position >= self.length:
                    self.valid = False
                    return 0
                var following = self._byte(self.position)
                self.position += 1
                while following == 255 and self.position < self.length:
                    following = self._byte(self.position)
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
        self.current = 0
        while self.position < self.length and self._byte(self.position) != 255:
            # No unaligned extra bytes allowed; fail unless right at 0xFF
            self.valid = False
            return False
        while self.position + 1 < self.length and self._byte(self.position) == 255 and self._byte(self.position + 1) == 255:
            self.position += 1
        if self.position + 2 > self.length or self._byte(self.position) != 255:
            self.valid = False
            return False
        var marker = self._byte(self.position + 1)
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
    var blocks_w: Int
    var blocks_h: Int
    var samples: List[UInt8]
    var coeffs: List[Int32]

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
        self.blocks_w = 0
        self.blocks_h = 0
        self.samples = List[UInt8]()
        self.coeffs = List[Int32]()


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


def _jpeg_decode_prog_block(
    mut bits: JpegBits,
    mut component: JpegComponent,
    dc_table: JpegHuffman,
    ac_table: JpegHuffman,
    bx: Int,
    by: Int,
    ss: Int,
    se: Int,
    ah: Int,
    al: Int,
    mut eobrun: Int,
):
    if bx < 0 or bx >= component.blocks_w or by < 0 or by >= component.blocks_h:
        bits.valid = False
        return
    var off = (by * component.blocks_w + bx) * 64
    if ss == 0:
        if ah == 0:
            if not dc_table.valid:
                bits.valid = False
                return
            var dc_size = dc_table.decode(bits)
            if not bits.valid or dc_size < 0 or dc_size > 16:
                bits.valid = False
                return
            component.predictor += jpeg_extend(bits, dc_size)
            if not bits.valid:
                return
            component.coeffs[off] = Int32(component.predictor) << Int32(al)
        else:
            var b = bits.read_bit()
            if not bits.valid:
                return
            if b != 0:
                component.coeffs[off] |= Int32(1 << al)
        return

    var p1 = Int32(1 << al)
    var m1 = -p1
    var k = ss
    if ah == 0:
        if eobrun > 0:
            eobrun -= 1
            return
        if not ac_table.valid:
            bits.valid = False
            return
        while k <= se and bits.valid:
            var sym = ac_table.decode(bits)
            if not bits.valid or sym < 0:
                bits.valid = False
                return
            var r = sym >> 4
            var s = sym & 15
            if s == 0:
                if r == 15:
                    k += 16
                    if k > se + 1:
                        bits.valid = False
                        return
                else:
                    eobrun = (1 << r) + bits.read_bits(r) - 1
                    if not bits.valid:
                        return
                    break
            else:
                k += r
                if k > se or s > 16:
                    bits.valid = False
                    return
                var val = jpeg_extend(bits, s)
                if not bits.valid:
                    return
                component.coeffs[off + jpeg_zigzag(k)] = Int32(val) * p1
                k += 1
        return

    # AC Refinement scan (Ss > 0, Ah > 0)
    if eobrun == 0:
        if not ac_table.valid:
            bits.valid = False
            return
        while k <= se and bits.valid:
            var sym = ac_table.decode(bits)
            if not bits.valid or sym < 0:
                bits.valid = False
                return
            var r = sym >> 4
            var s = sym & 15
            var new_val: Int32 = 0
            if s == 0:
                if r < 15:
                    eobrun = (1 << r) + bits.read_bits(r)
                    if not bits.valid:
                        return
                    break
            elif s == 1:
                var sign_b = bits.read_bit()
                if not bits.valid:
                    return
                new_val = p1 if sign_b == 1 else m1
            else:
                bits.valid = False
                return

            while k <= se and bits.valid:
                var nat = jpeg_zigzag(k)
                var cur = component.coeffs[off + nat]
                if cur != 0:
                    var ref_b = bits.read_bit()
                    if not bits.valid:
                        return
                    var abs_cur = Int(cur) if cur >= 0 else -Int(cur)
                    if ref_b != 0 and (abs_cur & Int(p1)) == 0:
                        component.coeffs[off + nat] = cur + (p1 if cur > 0 else m1)
                else:
                    if r == 0:
                        break
                    r -= 1
                k += 1
            if not bits.valid:
                return
            if k > se:
                if s != 0:
                    bits.valid = False
                return
            if new_val != 0:
                component.coeffs[off + jpeg_zigzag(k)] = new_val
            k += 1

    if eobrun > 0:
        while k <= se and bits.valid:
            var nat = jpeg_zigzag(k)
            var cur = component.coeffs[off + nat]
            if cur != 0:
                var ref_b = bits.read_bit()
                if not bits.valid:
                    return
                var abs_cur = Int(cur) if cur >= 0 else -Int(cur)
                if ref_b != 0 and (abs_cur & Int(p1)) == 0:
                    component.coeffs[off + nat] = cur + (p1 if cur > 0 else m1)
            k += 1
        eobrun -= 1


def _jpeg_assemble_image(
    mut components: List[JpegComponent],
    width: Int,
    height: Int,
    maximum_horizontal: Int,
    maximum_vertical: Int,
    maximum_size: Int,
) -> NativeImage:
    var image = NativeImage()
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
    var is_progressive = False
    var scan_count = 0
    var got_eoi = False
    var offset = 2
    while offset + 2 <= len(data):
        while offset < len(data) and Int(data[offset]) == 255:
            offset += 1
        if offset >= len(data):
            break
        var marker = Int(data[offset])
        offset += 1
        if marker == 217:
            got_eoi = True
            break
        if marker == 1 or (marker >= 208 and marker <= 215):
            continue
        if offset + 2 > len(data):
            return image^
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
            if position != end:
                return image^
        elif marker == 221:  # DRI
            if end - start != 2:
                return image^
            restart_interval = jpeg_be16(data, start)
        elif marker == 192 or marker == 193 or marker == 194:  # SOF0 / SOF1 / SOF2
            if len(components) > 0 or end - start < 6 or Int(data[start]) != 8:
                return image^
            is_progressive = marker == 194
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
                if component.horizontal <= 0 or component.horizontal > 4 or component.vertical <= 0 or component.vertical > 4 or component.quantization > 3:
                    return image^
                for existing_c in components:
                    if existing_c.identifier == component.identifier:
                        return image^
                maximum_horizontal = max(maximum_horizontal, component.horizontal)
                maximum_vertical = max(maximum_vertical, component.vertical)
                components.append(component^)
            var mcu_w0 = (width + maximum_horizontal * 8 - 1) // (maximum_horizontal * 8)
            var mcu_h0 = (height + maximum_vertical * 8 - 1) // (maximum_vertical * 8)
            for ci in range(len(components)):
                components[ci].blocks_w = mcu_w0 * components[ci].horizontal
                components[ci].blocks_h = mcu_h0 * components[ci].vertical
                components[ci].stride = components[ci].blocks_w * 8
                components[ci].rows = components[ci].blocks_h * 8
                if is_progressive:
                    components[ci].coeffs = List[Int32](length=components[ci].blocks_w * components[ci].blocks_h * 64, fill=0)
                else:
                    components[ci].samples = List[UInt8](length=components[ci].stride * components[ci].rows, fill=0)
        elif marker == 218:  # SOS
            if width <= 0 or height <= 0 or len(components) == 0 or end <= start:
                return image^
            var count = Int(data[start])
            if count <= 0 or count > len(components) or start + 1 + count * 2 + 3 != end:
                return image^
            var ss = Int(data[end - 3])
            var se = Int(data[end - 2])
            var ah = Int(data[end - 1]) >> 4
            var al = Int(data[end - 1]) & 15
            if not is_progressive:
                if count != len(components) or ss != 0 or se != 63 or ah != 0 or al != 0:
                    return image^
            else:
                if ss > se or se >= 64 or ah > 13 or al > 13:
                    return image^
                if ah != 0 and ah != al + 1:
                    return image^
                if ss == 0 and se != 0:
                    return image^
                if ss > 0 and count != 1:
                    return image^

            var scan_comp_indices = List[Int](capacity=count)
            for index in range(count):
                var identifier = Int(data[start + 1 + index * 2])
                var tables = Int(data[start + 2 + index * 2])
                if (tables >> 4) > 3 or (tables & 15) > 3:
                    return image^
                var found_idx = -1
                for component_index in range(len(components)):
                    if components[component_index].identifier == identifier:
                        found_idx = component_index
                        break
                if found_idx < 0:
                    return image^
                for prev_idx in scan_comp_indices:
                    if prev_idx == found_idx:
                        return image^
                components[found_idx].dc_table = tables >> 4
                components[found_idx].ac_table = tables & 15
                components[found_idx].predictor = 0
                scan_comp_indices.append(found_idx)

            var bits = JpegBits(data, end)
            if not is_progressive:
                var mcu_width = (width + maximum_horizontal * 8 - 1) // (maximum_horizontal * 8)
                var mcu_height = (height + maximum_vertical * 8 - 1) // (maximum_vertical * 8)
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
                            var component_index = scan_comp_indices[scan_index]
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
                _ = len(data)
                return _jpeg_assemble_image(components, width, height, maximum_horizontal, maximum_vertical, maximum_size)

            # Progressive SOF2 scan
            var eobrun = 0
            if count > 1:
                var mcu_width = (width + maximum_horizontal * 8 - 1) // (maximum_horizontal * 8)
                var mcu_height = (height + maximum_vertical * 8 - 1) // (maximum_vertical * 8)
                for mcu_y in range(mcu_height):
                    for mcu_x in range(mcu_width):
                        var mcu = mcu_y * mcu_width + mcu_x
                        if restart_interval > 0 and mcu > 0 and mcu % restart_interval == 0:
                            if not bits.restart():
                                return image^
                            for ci in scan_comp_indices:
                                components[ci].predictor = 0
                            eobrun = 0
                        for scan_index in range(count):
                            var ci = scan_comp_indices[scan_index]
                            var dc_i = components[ci].dc_table
                            var ac_i = components[ci].ac_table
                            for vertical in range(components[ci].vertical):
                                for horizontal in range(components[ci].horizontal):
                                    var bx = mcu_x * components[ci].horizontal + horizontal
                                    var by = mcu_y * components[ci].vertical + vertical
                                    _jpeg_decode_prog_block(bits, components[ci], dc_tables[dc_i], ac_tables[ac_i], bx, by, ss, se, ah, al, eobrun)
                                    if not bits.valid:
                                        return image^
            else:
                var ci = scan_comp_indices[0]
                var dc_i = components[ci].dc_table
                var ac_i = components[ci].ac_table
                var comp_w = (width * components[ci].horizontal + maximum_horizontal - 1) // maximum_horizontal
                var comp_h = (height * components[ci].vertical + maximum_vertical - 1) // maximum_vertical
                var scan_bx = max(1, (comp_w + 7) // 8)
                var scan_by = max(1, (comp_h + 7) // 8)
                for by in range(scan_by):
                    for bx in range(scan_bx):
                        var blk_i = by * scan_bx + bx
                        if restart_interval > 0 and blk_i > 0 and blk_i % restart_interval == 0:
                            if not bits.restart():
                                return image^
                            components[ci].predictor = 0
                            eobrun = 0
                        _jpeg_decode_prog_block(bits, components[ci], dc_tables[dc_i], ac_tables[ac_i], bx, by, ss, se, ah, al, eobrun)
                        if not bits.valid:
                            return image^
            if not bits.valid:
                return image^
            scan_count += 1
            var next_pos = bits.position
            while next_pos < len(data):
                if Int(data[next_pos]) == 255:
                    var j = next_pos + 1
                    while j < len(data) and Int(data[j]) == 255:
                        j += 1
                    if j < len(data):
                        var m2 = Int(data[j])
                        if m2 == 0 or (m2 >= 208 and m2 <= 215):
                            next_pos = j + 1
                            continue
                    break
                next_pos += 1
            offset = next_pos
            continue
        offset = end

    _ = len(data)
    if is_progressive and scan_count > 0 and got_eoi and width > 0 and height > 0:
        var coefficients = List[Float32](length=64, fill=0)
        var workspace = List[Float32](length=64, fill=0)
        for ci in range(len(components)):
            components[ci].samples = List[UInt8](length=components[ci].stride * components[ci].rows, fill=0)
            var q_idx = components[ci].quantization
            for by in range(components[ci].blocks_h):
                for bx in range(components[ci].blocks_w):
                    var off = (by * components[ci].blocks_w + bx) * 64
                    var ac_count = 0
                    for i in range(64):
                        var cv = Int(components[ci].coeffs[off + i])
                        if i > 0 and cv != 0:
                            ac_count += 1
                        coefficients[i] = Float32(cv * quantization[q_idx][i])
                    jpeg_idct_store(coefficients, components[ci].samples, components[ci].stride, bx * 8, by * 8, workspace, ac_count)
        return _jpeg_assemble_image(components, width, height, maximum_horizontal, maximum_vertical, maximum_size)

    return image^


def decode_image_auto(data: List[UInt8], maximum_size: Int = 4096) -> NativeImage:
    """Automatically decodes an embedded JPEG (`0xFF 0xD8`), KTX2 (`0xAB 'K' 'T' 'X' ...`), or PNG byte stream."""
    if len(data) >= 2 and Int(data[0]) == 0xFF and Int(data[1]) == 0xD8:
        return decode_jpeg(data, maximum_size)
    if is_ktx2_magic(data):
        return decode_ktx2(data, maximum_size)
    return decode_png(data, maximum_size)


struct _JpegBitWriter(Movable):
    var out: List[UInt8]
    var buf: UInt32
    var count: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.buf = 0
        self.count = 0

    @always_inline
    def write_bits(mut self, code: Int, nbits: Int):
        self.buf = (self.buf << UInt32(nbits)) | UInt32(code & ((1 << nbits) - 1))
        self.count += nbits
        while self.count >= 8:
            self.count -= 8
            var b = UInt8((self.buf >> UInt32(self.count)) & 0xFF)
            self.out.append(b)
            if b == 0xFF:
                self.out.append(0x00)

    def flush(mut self):
        if self.count > 0:
            var pad = 8 - self.count
            self.write_bits((1 << pad) - 1, pad)
        self.count = 0
        self.buf = 0


@always_inline
def _jpeg_bit_category(v: Int) -> Tuple[Int, Int]:
    if v == 0:
        return (0, 0)
    var av = abs(v)
    var cat = 0
    var tmp = av
    while tmp > 0:
        cat += 1
        tmp >>= 1
    var mag = v if v > 0 else (v + (1 << cat) - 1)
    return (cat, mag)


def _build_huff_enc_codes[N_SYMS: Int](counts: Array[Int, 16], syms: Array[Int, N_SYMS]) -> Tuple[List[Int], List[Int]]:
    var codes = List[Int](length=256, fill=0)
    var lens = List[Int](length=256, fill=0)
    var code = 0
    var idx = 0
    for length in range(1, 17):
        var c = counts[length - 1]
        for _ in range(c):
            var sym = syms[idx]
            codes[sym] = code
            lens[sym] = length
            code += 1
            idx += 1
        code <<= 1
    return (codes^, lens^)


def encode_jpeg_baseline(pixels: List[UInt32], width: Int, height: Int, quality: Int = 92) -> List[UInt8]:
    """Pure-Mojo ITU-T.81 Baseline Sequential (`SOF0`, `4:4:4` YCbCr) JPEG encoder with Annex K Huffman tables."""
    var res = List[UInt8]()
    if width <= 0 or height <= 0 or len(pixels) < width * height:
        return res^

    var q_clamped = max(10, min(100, quality))
    var q_scale = (5000 // q_clamped) if q_clamped < 50 else (200 - 2 * q_clamped)
    var std_lum_q = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]
    var qtbl_spatial = List[Float32](length=64, fill=1.0)
    var qtbl_zz = List[UInt8](length=64, fill=1)
    for k in range(64):
        var sp = jpeg_zigzag(k)
        var qv = max(1, min(255, (std_lum_q[sp] * q_scale + 50) // 100))
        qtbl_spatial[sp] = Float32(qv)
        qtbl_zz[k] = UInt8(qv)

    # ITU-T.81 Annex K.3 Standard Luminance DC & AC Huffman tables
    var dc_counts = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    var dc_syms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    var ac_counts = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
    var ac_syms = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA,
    ]
    var dc_enc = _build_huff_enc_codes(dc_counts, dc_syms)
    var dc_codes = dc_enc[0].copy()
    var dc_lens = dc_enc[1].copy()
    var ac_enc = _build_huff_enc_codes(ac_counts, ac_syms)
    var ac_codes = ac_enc[0].copy()
    var ac_lens = ac_enc[1].copy()

    # Precompute 8x8 separable FDCT cosine basis `C[u * 8 + x]`
    var cos_tbl = List[Float32](length=64, fill=0.0)
    var inv_sqrt2 = Float32(1.0) / sqrt(Float32(2.0))
    for u in range(8):
        var cu = inv_sqrt2 if u == 0 else Float32(1.0)
        for x in range(8):
            cos_tbl[u * 8 + x] = Float32(0.5) * cu * cos((Float32(2 * x + 1) * Float32(u) * Float32(3.1415926535)) / Float32(16.0))

    # 1. SOI + JFIF APP0
    res.append(0xFF); res.append(0xD8)
    var app0 = [0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]
    for b in app0:
        res.append(UInt8(b))

    # 2. DQT (table 0 shared across Y, Cb, Cr)
    res.append(0xFF); res.append(0xDB); res.append(0x00); res.append(67); res.append(0x00)
    for k in range(64):
        res.append(qtbl_zz[k])

    # 3. SOF0 (Baseline 8-bit 4:4:4 3-component YCbCr)
    res.append(0xFF); res.append(0xC0); res.append(0x00); res.append(17); res.append(8)
    res.append(UInt8((height >> 8) & 0xFF)); res.append(UInt8(height & 0xFF))
    res.append(UInt8((width >> 8) & 0xFF)); res.append(UInt8(width & 0xFF))
    res.append(3)
    res.append(1); res.append(0x11); res.append(0)
    res.append(2); res.append(0x11); res.append(0)
    res.append(3); res.append(0x11); res.append(0)

    # 4. DHT (DC table 0 + AC table 0)
    var dc_len = 2 + 1 + 16 + len(dc_syms)
    res.append(0xFF); res.append(0xC4); res.append(UInt8((dc_len >> 8) & 0xFF)); res.append(UInt8(dc_len & 0xFF))
    res.append(0x00)
    for c in dc_counts: res.append(UInt8(c))
    for s in dc_syms: res.append(UInt8(s))

    var ac_len = 2 + 1 + 16 + len(ac_syms)
    res.append(0xFF); res.append(0xC4); res.append(UInt8((ac_len >> 8) & 0xFF)); res.append(UInt8(ac_len & 0xFF))
    res.append(0x10)
    for c in ac_counts: res.append(UInt8(c))
    for s in ac_syms: res.append(UInt8(s))

    # 5. SOS
    var sos = [0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]
    for b in sos:
        res.append(UInt8(b))

    var bw = _JpegBitWriter()
    var prev_dc = [0, 0, 0]
    var blk = List[Float32](length=64, fill=0.0)
    var tmp = List[Float32](length=64, fill=0.0)
    var zz_q = List[Int](length=64, fill=0)

    var mcu_w = (width + 7) // 8
    var mcu_h = (height + 7) // 8

    for my in range(mcu_h):
        for mx in range(mcu_w):
            for ch in range(3):
                for py in range(8):
                    var sy = min(height - 1, my * 8 + py)
                    for px in range(8):
                        var sx = min(width - 1, mx * 8 + px)
                        var argb = pixels[sy * width + sx]
                        var r = Float32((argb >> 16) & 0xFF)
                        var g = Float32((argb >> 8) & 0xFF)
                        var b = Float32(argb & 0xFF)
                        var sample: Float32
                        if ch == 0:
                            sample = (Float32(0.299) * r + Float32(0.587) * g + Float32(0.114) * b) - Float32(128.0)
                        elif ch == 1:
                            sample = (Float32(-0.168736) * r - Float32(0.331264) * g + Float32(0.5) * b)
                        else:
                            sample = (Float32(0.5) * r - Float32(0.418688) * g - Float32(0.081312) * b)
                        blk[py * 8 + px] = sample

                # Separable 2D FDCT: rows then cols
                for row in range(8):
                    for u in range(8):
                        var acc = Float32(0.0)
                        for x in range(8):
                            acc += blk[row * 8 + x] * cos_tbl[u * 8 + x]
                        tmp[row * 8 + u] = acc

                for col in range(8):
                    for v in range(8):
                        var acc = Float32(0.0)
                        for y in range(8):
                            acc += tmp[y * 8 + col] * cos_tbl[v * 8 + y]
                        var sp = v * 8 + col
                        var qf = acc / qtbl_spatial[sp]
                        var qi = Int(qf + Float32(0.5)) if qf >= Float32(0.0) else -Int(-qf + Float32(0.5))
                        blk[sp] = Float32(qi)

                for k in range(64):
                    zz_q[k] = max(-1023, min(1023, Int(blk[jpeg_zigzag(k)])))

                # Encode DC diff
                var diff = zz_q[0] - prev_dc[ch]
                prev_dc[ch] = zz_q[0]
                var dc_cm = _jpeg_bit_category(diff)
                bw.write_bits(dc_codes[dc_cm[0]], dc_lens[dc_cm[0]])
                if dc_cm[0] > 0:
                    bw.write_bits(dc_cm[1], dc_cm[0])

                # Encode AC RLE
                var last_nz = 0
                for k in range(1, 64):
                    if zz_q[k] != 0:
                        last_nz = k
                var run = 0
                for k in range(1, last_nz + 1):
                    var val = zz_q[k]
                    if val == 0:
                        run += 1
                    else:
                        while run >= 16:
                            bw.write_bits(ac_codes[0xF0], ac_lens[0xF0])
                            run -= 16
                        var ac_cm = _jpeg_bit_category(val)
                        var rs = (run << 4) | ac_cm[0]
                        bw.write_bits(ac_codes[rs], ac_lens[rs])
                        bw.write_bits(ac_cm[1], ac_cm[0])
                        run = 0
                if last_nz < 63:
                    bw.write_bits(ac_codes[0x00], ac_lens[0x00])  # EOB

    bw.flush()
    for b in bw.out:
        res.append(b)
    res.append(0xFF); res.append(0xD9)
    return res^
