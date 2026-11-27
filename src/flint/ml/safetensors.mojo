"""Read named float16, bfloat16 and float32 weights from a bounded SafeTensors archive."""

from std.memory import bitcast
from std.ffi import external_call
from flint.assets.json import JsonDocument, parse_json
from flint.ml.tensor import Tensor, numel_of
from flint.platform.os import close_fd, file_size, open_read, read_bytes


struct _PrivateCheckpointMap(Movable):
    """Copy-on-write Linux mapping for adapting weights without changing the checkpoint."""

    var address: Int
    var size: Int

    def __init__(out self):
        self.address = 0
        self.size = 0

    def open(mut self, path: String, max_bytes: Int) raises:
        var fd = open_read(path)
        if fd < 0:
            raise Error("Cannot open checkpoint " + path)
        var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
        if size < 10 or size > max_bytes:
            close_fd(fd)
            raise Error("Invalid SafeTensors file length")
        var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(3), Int32(2), Int32(fd), Int64(0))
        close_fd(fd)
        if addr == 0 or addr == -1:
            raise Error("Cannot map checkpoint " + path)
        self.address = addr
        self.size = size

    def __deinit__(deinit self):
        if self.address:
            _ = external_call["munmap", Int32](self.address, Int64(self.size))


struct SafeTensorWeights(Movable):
    var archive: List[UInt8]
    var mapping: _PrivateCheckpointMap
    var header: JsonDocument
    var data_offset: Int

    def __init__(out self, path: String, max_bytes: Int, private_mapping: Bool = False) raises:
        self.mapping = _PrivateCheckpointMap()
        self.archive = List[UInt8]()
        var addr: Int
        var size: Int
        if private_mapping:
            self.mapping.open(path, max_bytes)
            addr = self.mapping.address
            size = self.mapping.size
        else:
            size = file_size(path)
            if size < 10 or size > max_bytes:
                raise Error("Missing or oversized SafeTensors checkpoint")
            self.archive = read_bytes(path)
            if len(self.archive) != size:
                raise Error("Truncated SafeTensors checkpoint")
            addr = Int(self.archive.unsafe_ptr())
        var bytes = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
        var header_size = Int(0)
        for i in range(8):
            if i >= 4 and bytes.unsafe_load(i) != 0:
                raise Error("Invalid SafeTensors header length")
            if i < 4:
                header_size |= Int(bytes.unsafe_load(i)) << (8 * i)
        if header_size < 2 or header_size > min(1 << 20, size - 8):
            raise Error("Invalid SafeTensors header length")
        var header_bytes = List[UInt8](capacity=header_size)
        for i in range(header_size):
            header_bytes.append(bytes.unsafe_load(8 + i))
        self.header = parse_json(header_bytes)
        if not self.header.valid or self.header.nodes[self.header.root].kind != 5:
            raise Error("Invalid SafeTensors header")
        self.data_offset = 8 + header_size

    def address(self) -> Int:
        return self.mapping.address if self.mapping.address else Int(self.archive.unsafe_ptr())

    def discard_private_changes(mut self) raises:
        # MADV_DONTNEED restores the private mapping from its original file.
        if (
            self.mapping.address == 0
            or external_call["madvise", Int32](self.mapping.address, Int64(self.mapping.size), Int32(4)) != 0
        ):
            raise Error("Cannot restore private checkpoint mapping")

    def storage(self, key: String, shape: List[Int]) raises -> Tuple[Int, String]:
        """Validate a tensor and return its absolute archive offset and storage dtype."""
        var node = self.header.object_get(self.header.root, key)
        var dims = self.header.object_get(node, "shape")
        var dtype = self.header.text(self.header.object_get(node, "dtype"))
        if self.header.array_size(dims) != len(shape) or (
            dtype != "F16" and dtype != "BF16" and dtype != "F32"
        ):
            raise Error("Missing or incompatible tensor: " + key)
        for i in range(len(shape)):
            if self.header.integer(self.header.at(dims, i)) != shape[i]:
                raise Error("Unexpected shape for " + key)
        var pair = self.header.object_get(node, "data_offsets")
        var start = self.header.integer(self.header.at(pair, 0))
        var end = self.header.integer(self.header.at(pair, 1))
        var count = numel_of(shape)
        var stride = 4 if dtype == "F32" else 2
        if (
            self.header.array_size(pair) != 2
            or start < 0
            or end < start
            or end > (self.mapping.size if self.mapping.address else len(self.archive)) - self.data_offset
            or end - start != count * stride
        ):
            raise Error("Invalid byte offsets for " + key)
        return self.data_offset + start, dtype^

    def tensor(self, key: String, var shape: List[Int]) raises -> Tensor:
        var offset, dtype = self.storage(key, shape)
        var count = numel_of(shape)
        var result = Tensor.zeros(shape^)
        var address = self.address() + offset
        if dtype == "F16":
            var src = Pointer[Float16, ImmutAnyOrigin](unsafe_from_address=address)
            for i in range(count):
                result.f[i] = Float32(src.unsafe_load[alignment=1](i))
        elif dtype == "BF16":
            var src = Pointer[UInt16, ImmutAnyOrigin](unsafe_from_address=address)
            for i in range(count):
                var high = UInt32(src.unsafe_load[alignment=1](i))
                result.f[i] = bitcast[DType.float32](high << 16)
        else:
            var src = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=address)
            for i in range(count):
                result.f[i] = src.unsafe_load[alignment=1](i)
        return result^
