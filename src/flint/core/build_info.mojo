"""The build stamp: which commit, whether the tree had changes, which Mojo.

`tools/build.sh` writes `core/build_stamp.mojo` before compiling anything, so
every binary knows the source it was built from. The stamp goes into
screenshots (a PNG text chunk), replay headers, saves, crash reports and the
performance history, so any of those can be traced back to its build.
"""
from flint.core.build_stamp import BUILD_COMMIT, BUILD_DIRTY, BUILD_MOJO

comptime STAMP_MAX_BYTES: Int = 96  # what saves and replay headers reserve for it


def build_commit() -> String:
    """The short commit, with "+dirty" when the tree had uncommitted changes."""
    comptime if BUILD_DIRTY:
        return String(BUILD_COMMIT) + "+dirty"
    return String(BUILD_COMMIT)


def build_stamp() -> String:
    """One line: "<commit>[+dirty] mojo <version>". No commas, so it fits a CSV cell."""
    var text = build_commit() + " mojo " + String(BUILD_MOJO)
    if text.byte_length() > STAMP_MAX_BYTES:
        return String(text[byte=0:STAMP_MAX_BYTES])
    return text^


def stamp_words(stamp: String) -> List[Int32]:
    """The stamp packed four bytes to a word, little-endian, length first, for saves."""
    var bytes = stamp.as_bytes()
    var count = min(len(bytes), STAMP_MAX_BYTES)
    var words = List[Int32]()
    words.append(Int32(count))
    var index = 0
    while index < count:
        var word = UInt32(0)
        for shift in range(4):
            if index + shift < count:
                word |= UInt32(bytes[index + shift]) << UInt32(shift * 8)
        words.append(Int32(Int(word)))
        index += 4
    return words^


def stamp_from_words(words: List[Int32], start: Int) -> String:
    """Reverses `stamp_words` from `start`. Empty if the words are not a stamp."""
    if start < 0 or start >= len(words):
        return String()
    var count = Int(words[start])
    if count < 0 or count > STAMP_MAX_BYTES or start + 1 + (count + 3) // 4 > len(words):
        return String()
    var bytes = List[UInt8]()
    for index in range(count):
        var word = UInt32(Int(words[start + 1 + index // 4]) & 0xFFFFFFFF)
        var byte = UInt8((word >> UInt32((index % 4) * 8)) & 0xFF)
        if byte < 32 or byte > 126:
            return String()
        bytes.append(byte)
    return String(unsafe_from_utf8=bytes)
