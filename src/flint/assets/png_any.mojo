"""PNG decoding for every non-interlaced colour type (`flint.assets.png_any`).

`flint.assets.image.decode_png` reads 8-bit RGB / RGBA only (and downsamples big images); pixel
art is often stored as palette or greyscale PNGs. `decode_png_any` reads colour types 0 (grey),
2 (RGB), 3 (palette, with tRNS transparency), 4 (grey + alpha) and 6 (RGBA) at bit depths
1 / 2 / 4 / 8 / 16 (16-bit samples keep their high byte) and returns full-size 0xAARRGGBB
pixels. Adam7-interlaced files are rejected. Inflate and the Paeth predictor are shared with
`flint.assets.image`.
"""

from std.ffi import external_call
from flint.assets.image import NativeImage, be32, inflate_zlib, paeth
from flint.platform.os import read_bytes


def _channels(color_type: Int) -> Int:
    if color_type == 0 or color_type == 3:
        return 1
    if color_type == 2:
        return 3
    if color_type == 4:
        return 2
    if color_type == 6:
        return 4
    return 0


def decode_png_any(data: List[UInt8]) raises -> NativeImage:
    if len(data) < 45:
        raise Error("png: file too short")
    var signature: List[Int] = [137, 80, 78, 71, 13, 10, 26, 10]
    for i in range(8):
        if Int(data[i]) != signature[i]:
            raise Error("png: not a PNG file")
    var width = be32(data, 16)
    var height = be32(data, 20)
    var depth = Int(data[24])
    var color_type = Int(data[25])
    var interlace = Int(data[28])
    var ch = _channels(color_type)
    if width <= 0 or height <= 0 or width * height > 268435456 or ch == 0:
        raise Error("png: unsupported header")
    if interlace != 0:
        raise Error("png: interlaced (Adam7) files are not supported")
    if not (depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16):
        raise Error("png: bad bit depth")
    if (color_type == 2 or color_type == 4 or color_type == 6) and depth < 8:
        raise Error("png: bad bit depth for colour type")
    var palette = List[UInt32]()
    var trns = List[UInt8]()
    var compressed = List[UInt8](capacity=len(data))
    var offset = 8
    var finished = False
    while offset + 12 <= len(data):
        var length = be32(data, offset)
        if length < 0 or offset + length + 12 > len(data):
            raise Error("png: truncated chunk")
        var kind = be32(data, offset + 4)
        var body = offset + 8
        if kind == 0x49444154:          # IDAT
            var cur = len(compressed)
            compressed.resize(cur + length, 0)
            _ = external_call["memcpy", Int](Int(compressed.unsafe_ptr()) + cur, Int(data.unsafe_ptr()) + body, length)
        elif kind == 0x504C5445:        # PLTE
            for k in range(length // 3):
                palette.append(0xFF000000 | (UInt32(data[body + 3 * k]) << 16) | (UInt32(data[body + 3 * k + 1]) << 8)
                               | UInt32(data[body + 3 * k + 2]))
        elif kind == 0x74524E53:        # tRNS
            for k in range(length):
                trns.append(data[body + k])
        elif kind == 0x49454E44:        # IEND
            finished = True
            break
        offset += length + 12
    if not finished:
        raise Error("png: no IEND")
    var bpp = max(1, (ch * depth) // 8)                 # filter unit in bytes
    var stride = (width * ch * depth + 7) // 8
    var raw = inflate_zlib(compressed, 0, (stride + 1) * height)
    if len(raw) < (stride + 1) * height:
        raise Error("png: image data too short")
    var lines = List[UInt8](length=stride * height, fill=0)
    for y in range(height):
        var filter = Int(raw[y * (stride + 1)])
        for x in range(stride):
            var enc = Int(raw[y * (stride + 1) + 1 + x])
            var left = Int(lines[y * stride + x - bpp]) if x >= bpp else 0
            var up = Int(lines[(y - 1) * stride + x]) if y > 0 else 0
            var diag = Int(lines[(y - 1) * stride + x - bpp]) if y > 0 and x >= bpp else 0
            var pred = 0
            if filter == 1:
                pred = left
            elif filter == 2:
                pred = up
            elif filter == 3:
                pred = (left + up) // 2
            elif filter == 4:
                pred = paeth(left, up, diag)
            elif filter != 0:
                raise Error("png: bad filter type")
            lines[y * stride + x] = UInt8((enc + pred) & 255)
    if color_type == 3:
        for k in range(min(len(trns), len(palette))):
            palette[k] = (palette[k] & 0x00FFFFFF) | (UInt32(trns[k]) << 24)
    # grey / RGB tRNS: one transparent colour (16-bit samples, compared at full precision)
    var key = -1
    if color_type == 0 and len(trns) >= 2:
        key = (Int(trns[0]) << 8) | Int(trns[1])
    var key_rgb = List[Int]()
    if color_type == 2 and len(trns) >= 6:
        for c in range(3):
            key_rgb.append((Int(trns[2 * c]) << 8) | Int(trns[2 * c + 1]))
    var img = NativeImage()
    img.width = width
    img.height = height
    img.pixels = List[UInt32](length=width * height, fill=0xFF000000)
    var bytes_per = depth // 8
    for y in range(height):
        var row = y * stride
        for x in range(width):
            var p: UInt32
            if depth < 8:
                var bit = x * depth
                var byte = Int(lines[row + bit // 8])
                var v = (byte >> (8 - depth - bit % 8)) & ((1 << depth) - 1)
                if color_type == 3:
                    if v >= len(palette):
                        raise Error("png: palette index out of range")
                    p = palette[v]
                else:
                    var g = UInt32(v * 255 // ((1 << depth) - 1))
                    p = 0xFF000000 | (g << 16) | (g << 8) | g
                    if key == v:
                        p &= 0x00FFFFFF
            else:
                var base = row + x * ch * bytes_per
                var s = List[Int](length=4, fill=0)
                var full = List[Int](length=4, fill=0)
                for c in range(ch):
                    s[c] = Int(lines[base + c * bytes_per])
                    full[c] = s[c] if bytes_per == 1 else (s[c] << 8) | Int(lines[base + c * bytes_per + 1])
                if color_type == 3:
                    if s[0] >= len(palette):
                        raise Error("png: palette index out of range")
                    p = palette[s[0]]
                elif color_type == 0:
                    p = 0xFF000000 | (UInt32(s[0]) << 16) | (UInt32(s[0]) << 8) | UInt32(s[0])
                    if key == full[0]:
                        p &= 0x00FFFFFF
                elif color_type == 4:
                    p = (UInt32(s[1]) << 24) | (UInt32(s[0]) << 16) | (UInt32(s[0]) << 8) | UInt32(s[0])
                elif color_type == 2:
                    p = 0xFF000000 | (UInt32(s[0]) << 16) | (UInt32(s[1]) << 8) | UInt32(s[2])
                    if len(key_rgb) == 3 and key_rgb[0] == full[0] and key_rgb[1] == full[1] and key_rgb[2] == full[2]:
                        p &= 0x00FFFFFF
                else:
                    p = (UInt32(s[3]) << 24) | (UInt32(s[0]) << 16) | (UInt32(s[1]) << 8) | UInt32(s[2])
            img.pixels[y * width + x] = p
    img.valid = True
    return img^


def load_png_any(path: String) raises -> NativeImage:
    var data = read_bytes(path)
    if len(data) == 0:
        raise Error("png: cannot read " + path)
    return decode_png_any(data)
