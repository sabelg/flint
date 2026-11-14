"""CRC-32 (IEEE 802.3, the zlib and PNG polynomial 0xEDB88320, reflected).

Each section of a save carries one (gameplay/save_file.mojo), and so does the
save's header, so a flipped bit or a torn write is found before anything is
read from the file. `crc32(b"123456789") == 0xCBF43926`, the standard check
value (tests/test_saves.mojo).
"""


struct Crc32Table(Movable):
    var entries: List[UInt32]

    def __init__(out self):
        self.entries = List[UInt32](capacity=256)
        for n in range(256):
            var c = UInt32(n)
            for _ in range(8):
                if c & 1 == 1:
                    c = UInt32(0xEDB88320) ^ (c >> 1)
                else:
                    c = c >> 1
            self.entries.append(c)


def crc32_update(table: Crc32Table, crc: UInt32, data: List[UInt8], start: Int, end: Int) -> UInt32:
    """Continues a CRC (pass 0 to start) over data[start:end]."""
    var c = crc ^ 0xFFFFFFFF
    for index in range(max(0, start), min(end, len(data))):
        c = table.entries[Int((c ^ UInt32(data[index])) & 0xFF)] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def crc32(data: List[UInt8], start: Int = 0, end: Int = -1) -> UInt32:
    var table = Crc32Table()
    return crc32_update(table, 0, data, start, len(data) if end < 0 else end)
