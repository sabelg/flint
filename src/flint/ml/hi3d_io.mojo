"""Little-endian tensor interchange for Hi3D inference stages."""
from std.memory import bitcast
from flint.platform.os import file_size, read_bytes, write_bytes
from flint.ml.tensor import Tensor, DT_I32


def hi3d_read_rows(path: String, cols: Int, integer: Bool = False) raises -> Tensor:
    var size = file_size(path)
    if cols < 1 or size <= 0 or size % (4 * cols) != 0:
        raise Error("Invalid tensor file dimensions: " + path)
    var raw = read_bytes(path)
    if len(raw) != size:
        raise Error("Incomplete tensor: " + path)
    var n = size // 4
    var out = Tensor.ints(
        DT_I32, [n // cols, cols], List[Int](length=n, fill=0)
    ) if integer else Tensor.zeros([n // cols, cols])
    for i in range(n):
        var bits = UInt32(0)
        comptime for j in range(4):
            bits |= UInt32(raw[4 * i + j]) << UInt32(8 * j)
        if integer:
            out.i[i] = Int(bitcast[DType.int32](bits))
        else:
            out.f[i] = bitcast[DType.float32](bits)
    return out^


def hi3d_write_rows(path: String, x: Tensor) raises:
    var raw = List[UInt8](capacity=x.numel() * 4)
    for i in range(x.numel()):
        var bits = UInt32(x.i[i]) if x.dtype == DT_I32 else bitcast[DType.uint32](x.f[i])
        comptime for j in range(4):
            raw.append(UInt8(bits >> UInt32(8 * j)))
    if not write_bytes(path, raw):
        raise Error("Could not write tensor: " + path)
