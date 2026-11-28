"""Write TRELLIS image conditioning as contiguous little-endian float32 tokens."""
from std.sys import argv
from std.memory import bitcast
from flint.assets.image import decode_png
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.trellis2_dino import TrellisImageEncoder
from flint.platform.os import read_bytes, write_bytes


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("Usage: encode_image MODEL.safetensors CUTOUT.png 512|1024 TOKENS.f32")
    var image = decode_png(read_bytes(String(args[2])), 40000000)
    var encoder = TrellisImageEncoder(String(args[1]))
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var tokens = encoder.encode(pool, image, Int(String(args[3])))
    var raw = List[UInt8](capacity=tokens.numel() * 4)
    for value in tokens.f:
        var bits = bitcast[DType.uint32](value)
        comptime for j in range(4):
            raw.append(UInt8(bits >> UInt32(8 * j)))
    if not write_bytes(String(args[4]), raw):
        raise Error("Could not write image conditioning")
    print("Encoded", tokens.shape[1], "tokens of width", tokens.shape[2])
