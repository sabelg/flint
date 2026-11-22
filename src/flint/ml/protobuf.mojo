"""A minimal protobuf wire-format reader over a memory range (an mmapped ONNX file)."""

comptime WIRE_VARINT: Int = 0
comptime WIRE_I64: Int = 1
comptime WIRE_LEN: Int = 2
comptime WIRE_I32: Int = 5


@always_inline
def byte_at(address: Int) -> Int:
    return Int(Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address).unsafe_load())


struct PbReader(Copyable, Movable):
    """Reads fields from [pos, end). `field` and `wire` hold the last key read."""
    var pos: Int
    var end: Int
    var field: Int
    var wire: Int

    def __init__(out self, start: Int, end: Int):
        self.pos = start
        self.end = end
        self.field = 0
        self.wire = 0

    def more(self) -> Bool:
        return self.pos < self.end

    def varint(mut self) raises -> Int:
        var result = 0
        var shift = 0
        while True:
            if self.pos >= self.end:
                raise Error("protobuf: truncated varint")
            var b = byte_at(self.pos)
            self.pos += 1
            result |= (b & 0x7F) << shift
            if b < 0x80:
                return result
            shift += 7
            if shift > 63:
                raise Error("protobuf: varint too long")

    def key(mut self) raises:
        var k = self.varint()
        self.field = k >> 3
        self.wire = k & 7

    def fixed32(mut self) raises -> UInt32:
        if self.end - self.pos < 4:
            raise Error("protobuf: truncated fixed32")
        var v = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=self.pos).unsafe_load[alignment=1]()
        self.pos += 4
        return v

    def length_delimited(mut self) raises -> Tuple[Int, Int]:
        """The (start, end) of a length-delimited field, skipping past it."""
        var n = self.varint()
        var start = self.pos
        if n < 0 or self.end < start or n > self.end - start:
            raise Error("protobuf: length out of range")
        self.pos = start + n
        return (start, start + n)

    def sub(mut self) raises -> PbReader:
        var r = self.length_delimited()
        return PbReader(r[0], r[1])

    def string(mut self) raises -> String:
        var r = self.length_delimited()
        var n = r[1] - r[0]
        var bytes = List[UInt8](capacity=n)
        for k in range(n):
            bytes.append(UInt8(byte_at(r[0] + k)))
        return String(unsafe_from_utf8=bytes)

    def skip(mut self) raises:
        if self.wire == WIRE_VARINT:
            _ = self.varint()
        elif self.wire == WIRE_I64:
            if self.end - self.pos < 8:
                raise Error("protobuf: truncated fixed64")
            self.pos += 8
        elif self.wire == WIRE_LEN:
            _ = self.length_delimited()
        elif self.wire == WIRE_I32:
            if self.end - self.pos < 4:
                raise Error("protobuf: truncated fixed32")
            self.pos += 4
        else:
            raise Error("protobuf: unsupported wire type " + String(self.wire))

    def append_ints(mut self, mut out: List[Int]) raises:
        """A repeated int64/int32 field, packed or not."""
        if self.wire == WIRE_LEN:
            var r = self.length_delimited()
            var p = PbReader(r[0], r[1])
            while p.more():
                out.append(_signed(p.varint()))
        else:
            out.append(_signed(self.varint()))

    def append_floats(mut self, mut out: List[Float32]) raises:
        if self.wire == WIRE_LEN:
            var r = self.length_delimited()
            var at = r[0]
            while r[1] - at >= 4:
                out.append(Pointer[Float32, MutAnyOrigin](unsafe_from_address=at).unsafe_load[alignment=1]())
                at += 4
        else:
            var bits = self.fixed32()
            out.append(Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=bits))).unsafe_load())


@always_inline
def _signed(v: Int) -> Int:
    """Int64 fields arrive as 64-bit two's complement varints; Int is 64-bit already."""
    return v
