"""Versioned binary blobs: the one serialisation path for simulation state.

Every table of the simulation `World` (gameplay/world.mojo) writes itself as a
blob, and the same bytes serve the replay checksums (hash of the blobs), state
snapshots, golden-state diffs and, later, saves and rollback.

A blob is
    u32 tag (four ASCII letters)  u32 version  u32 payload_length  payload
little-endian. Blobs nest: a payload may hold child blobs. A reader enters a
blob by tag, learns its version, reads what that version wrote (older versions
fill the rest with defaults) and leaves it, skipping whatever a newer writer
appended, so a blob with fields added at the end still loads.

Floats are written as their exact bit pattern; nothing is rounded or
formatted, so a written-then-read value is the same value.
"""
from std.ffi import external_call
from flint.core.math import Vec3, Quat

comptime BLOB_HEADER_BYTES: Int = 12


def float32_bits(value: Float32) -> UInt32:
    var copy = value
    return Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]


def float32_from_bits(bits: UInt32) -> Float32:
    var copy = bits
    return Pointer(to=copy).unsafe_bitcast[Float32]()[unsafe_offset=0]


def fourcc(text: StaticString) -> UInt32:
    """Four ASCII letters as a little-endian tag ("PLYR")."""
    var bytes = text.as_bytes()
    var value = UInt32(0)
    for index in range(min(4, len(bytes))):
        value |= UInt32(bytes[index]) << UInt32(index * 8)
    return value


def tag_text(tag: UInt32) -> String:
    var out = String()
    for index in range(4):
        var byte = Int((tag >> UInt32(index * 8)) & 0xFF)
        out += chr(byte) if byte >= 32 and byte < 127 else "?"
    return out


struct BlobWriter(Movable):
    """Appends blobs to a byte list. `clear` keeps the capacity, so a writer
    reused every tick does not allocate once it has grown."""
    var data: List[UInt8]
    var open_at: List[Int]  # offsets of the length fields of open blobs

    def __init__(out self):
        self.data = List[UInt8]()
        self.open_at = List[Int]()

    def clear(mut self):
        self.data.clear()
        self.open_at.clear()

    def begin(mut self, tag: UInt32, version: Int):
        self.u32(tag)
        self.u32(UInt32(version))
        self.open_at.append(len(self.data))
        self.u32(0)

    def end(mut self):
        var at = self.open_at.pop()
        var length = UInt32(len(self.data) - at - 4)
        for shift in range(4):
            self.data[at + shift] = UInt8((length >> UInt32(shift * 8)) & 0xFF)

    def u8(mut self, value: UInt8):
        self.data.append(value)

    def boolean(mut self, value: Bool):
        self.data.append(UInt8(1) if value else UInt8(0))

    def u32(mut self, value: UInt32):
        for shift in range(4):
            self.data.append(UInt8((value >> UInt32(shift * 8)) & 0xFF))

    def u64(mut self, value: UInt64):
        for shift in range(8):
            self.data.append(UInt8((value >> UInt64(shift * 8)) & 0xFF))

    def integer(mut self, value: Int):
        """A signed 64-bit integer."""
        self.u64(UInt64(value))

    def f32(mut self, value: Float32):
        self.u32(float32_bits(value))

    def vec3(mut self, value: Vec3):
        self.f32(value.x)
        self.f32(value.y)
        self.f32(value.z)

    def quat(mut self, value: Quat):
        self.f32(value.x)
        self.f32(value.y)
        self.f32(value.z)
        self.f32(value.w)

    def text(mut self, value: String):
        var bytes = value.as_bytes()
        self.u32(UInt32(len(bytes)))
        for byte in bytes:
            self.data.append(byte)

    def write_bytes_raw(mut self, src_ptr: Int, byte_len: Int):
        """Appends `byte_len` raw bytes from `src_ptr` in a single memcpy."""
        if src_ptr == 0 or byte_len <= 0:
            return
        var old_len = len(self.data)
        self.data.resize(unsafe_uninit_length=old_len + byte_len)
        _ = external_call["memcpy", Int](
            Int(self.data.unsafe_ptr()) + old_len, src_ptr, byte_len
        )


struct BlobReader(Movable):
    """Reads what a BlobWriter wrote. A read past the end or a wrong tag sets
    `failed` and returns zeros; `enter` raises so a caller stops at once."""
    var data: List[UInt8]
    var offset: Int
    var failed: Bool
    var ends: List[Int]  # payload end offsets of entered blobs

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.offset = 0
        self.failed = False
        self.ends = List[Int]()

    def remaining(self) -> Int:
        return len(self.data) - self.offset

    def limit(self) -> Int:
        return self.ends[len(self.ends) - 1] if len(self.ends) > 0 else len(self.data)

    def peek_tag(self) -> UInt32:
        if self.offset + 4 > self.limit():
            return 0
        var value = UInt32(0)
        for shift in range(4):
            value |= UInt32(self.data[self.offset + shift]) << UInt32(shift * 8)
        return value

    def enter(mut self, tag: UInt32, newest: Int) raises -> Int:
        """Opens the next blob, which must carry `tag`; returns its version.
        A version newer than `newest` (what this build writes) is refused."""
        var found = self.u32()
        var version = Int(self.u32())
        var length = Int(self.u32())
        if self.failed or found != tag:
            self.failed = True
            raise Error("expected blob " + tag_text(tag) + ", found " + tag_text(found))
        if version < 1 or version > newest:
            self.failed = True
            raise Error("blob " + tag_text(tag) + " has version " + String(version) + "; this build reads 1 to " + String(newest))
        if length < 0 or self.offset + length > self.limit():
            self.failed = True
            raise Error("blob " + tag_text(tag) + " is truncated")
        self.ends.append(self.offset + length)
        return version

    def leave(mut self) raises:
        """Skips what a newer writer appended to the current blob."""
        if len(self.ends) == 0:
            raise Error("leave without enter")
        var end = self.ends.pop()
        if self.failed or self.offset > end:
            self.failed = True
            raise Error("blob payload overran its length")
        self.offset = end

    def u8(mut self) -> UInt8:
        if self.offset >= self.limit():
            self.failed = True
            return 0
        var value = self.data[self.offset]
        self.offset += 1
        return value

    def boolean(mut self) -> Bool:
        return self.u8() != 0

    def u32(mut self) -> UInt32:
        var value = UInt32(0)
        for shift in range(4):
            value |= UInt32(self.u8()) << UInt32(shift * 8)
        return value

    def u64(mut self) -> UInt64:
        var value = UInt64(0)
        for shift in range(8):
            value |= UInt64(self.u8()) << UInt64(shift * 8)
        return value

    def integer(mut self) -> Int:
        return Int(self.u64())

    def f32(mut self) -> Float32:
        return float32_from_bits(self.u32())

    def vec3(mut self) -> Vec3:
        var x = self.f32()
        var y = self.f32()
        var z = self.f32()
        return Vec3(x, y, z)

    def quat(mut self) -> Quat:
        var x = self.f32()
        var y = self.f32()
        var z = self.f32()
        var w = self.f32()
        return Quat(x, y, z, w)

    def text(mut self) -> String:
        var length = Int(self.u32())
        if length < 0 or self.offset + length > self.limit():
            self.failed = True
            return String()
        var bytes = List[UInt8]()
        for index in range(length):
            bytes.append(self.data[self.offset + index])
        self.offset += length
        return String(unsafe_from_utf8=bytes) if length > 0 else String()

    def read_bytes_raw(mut self, dst_ptr: Int, byte_len: Int) -> Bool:
        """Reads `byte_len` raw bytes into `dst_ptr` in a single memcpy."""
        if byte_len < 0 or self.offset + byte_len > self.limit():
            self.failed = True
            return False
        if byte_len > 0 and dst_ptr != 0:
            _ = external_call["memcpy", Int](
                dst_ptr, Int(self.data.unsafe_ptr()) + self.offset, byte_len
            )
        self.offset += byte_len
        return True


trait Serializable:
    """State that writes itself as one versioned blob and reads it back.

    `serialize` writes exactly the authoritative state (never presentation),
    in a fixed order with no hash-map iteration, so equal state gives equal
    bytes. `deserialize` reads every version from 1 to the current one."""

    def serialize(self, mut out: BlobWriter):
        ...

    def deserialize(mut self, mut input: BlobReader) raises:
        ...


def fnv64(data: List[UInt8], start: Int = 0, end: Int = -1) -> UInt64:
    """FNV-1a 64 over a byte range, the hash the replay checksums use."""
    var value = UInt64(0xCBF29CE484222325)
    var stop = len(data) if end < 0 else min(end, len(data))
    for index in range(start, stop):
        value = (value ^ UInt64(data[index])) * 0x100000001B3
    return value


@always_inline
def _fast_mix64(value: UInt64) -> UInt64:
    var h = value * UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 31
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> 29)


def fast_hash64(data_ptr: Int, byte_len: Int) -> UInt64:
    """Fast 8-byte word-strided 64-bit hash over `[data_ptr, data_ptr + byte_len)`."""
    if byte_len <= 0 or data_ptr == 0:
        var empty_h = UInt64(0x9E3779B97F4A7C15)
        return _fast_mix64(empty_h ^ (UInt64(0xA5) << 56))
    var h = UInt64(0x9E3779B97F4A7C15) ^ UInt64(byte_len)
    var at = 0
    while at + 8 <= byte_len:
        var word = Pointer[UInt64, origin=ImmutAnyOrigin](
            unsafe_from_address=data_ptr + at
        ).unsafe_load[alignment=1]()
        h = _fast_mix64(h ^ word)
        at += 8
    var tail = UInt64(0)
    var shift = UInt64(0)
    var bytes = Pointer[UInt8, origin=ImmutAnyOrigin](unsafe_from_address=data_ptr)
    while at < byte_len:
        tail |= UInt64(bytes[unsafe_offset=at]) << shift
        shift += 8
        at += 1
    return _fast_mix64(h ^ tail ^ (UInt64(0xA5) << 56))
