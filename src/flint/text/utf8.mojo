"""UTF-8 decoding and encoding (package 3.12).

Strict decoding per RFC 3629 / Unicode 15 chapter 3 (table 3-7): overlong
forms, surrogates (U+D800-U+DFFF), values above U+10FFFF and truncated
sequences are refused. Each refused lead byte becomes one U+FFFD and decoding
resumes at the next byte, so a broken string still shows where it broke.
"""

comptime REPLACEMENT: Int = 0xFFFD


def decode_utf8(text: String) -> List[Int]:
    """Codepoints of a UTF-8 string; malformed bytes become U+FFFD."""
    var codes = List[Int]()
    var bytes = text.as_bytes()
    var index = 0
    var count = len(bytes)
    while index < count:
        var lead = Int(bytes[index])
        var extra: Int
        var code: Int
        var minimum: Int
        if lead < 0x80:
            codes.append(lead)
            index += 1
            continue
        elif lead >= 0xC2 and lead < 0xE0:
            extra = 1
            code = lead & 0x1F
            minimum = 0x80
        elif lead >= 0xE0 and lead < 0xF0:
            extra = 2
            code = lead & 0x0F
            minimum = 0x800
        elif lead >= 0xF0 and lead < 0xF5:
            extra = 3
            code = lead & 0x07
            minimum = 0x10000
        else:
            # Continuation bytes on their own, C0/C1 (always overlong) and F5-FF.
            codes.append(REPLACEMENT)
            index += 1
            continue
        var valid = True
        for step in range(1, extra + 1):
            if index + step >= count:
                valid = False
                break
            var next = Int(bytes[index + step])
            if (next & 0xC0) != 0x80:
                valid = False
                break
            code = (code << 6) | (next & 0x3F)
        if valid and (code < minimum or code > 0x10FFFF or (code >= 0xD800 and code <= 0xDFFF)):
            valid = False
        if not valid:
            codes.append(REPLACEMENT)
            index += 1
            continue
        codes.append(code)
        index += extra + 1
    return codes^


def encode_codepoint(mut bytes: List[UInt8], code: Int):
    var c = code
    if c < 0 or c > 0x10FFFF or (c >= 0xD800 and c <= 0xDFFF):
        c = REPLACEMENT
    if c < 0x80:
        bytes.append(UInt8(c))
    elif c < 0x800:
        bytes.append(UInt8(0xC0 | (c >> 6)))
        bytes.append(UInt8(0x80 | (c & 0x3F)))
    elif c < 0x10000:
        bytes.append(UInt8(0xE0 | (c >> 12)))
        bytes.append(UInt8(0x80 | ((c >> 6) & 0x3F)))
        bytes.append(UInt8(0x80 | (c & 0x3F)))
    else:
        bytes.append(UInt8(0xF0 | (c >> 18)))
        bytes.append(UInt8(0x80 | ((c >> 12) & 0x3F)))
        bytes.append(UInt8(0x80 | ((c >> 6) & 0x3F)))
        bytes.append(UInt8(0x80 | (c & 0x3F)))


def encode_utf8(codes: List[Int], start: Int = 0, end: Int = -1) -> String:
    """UTF-8 for `codes[start:end]` (end -1 = to the end). Invalid scalar
    values are written as U+FFFD."""
    var bytes = List[UInt8]()
    var stop = len(codes) if end < 0 else min(end, len(codes))
    for index in range(max(0, start), stop):
        encode_codepoint(bytes, codes[index])
    return String(unsafe_from_utf8=bytes)
