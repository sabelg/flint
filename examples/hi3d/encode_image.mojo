"""Write Hi3D image conditioning as contiguous little-endian float32 tokens."""
from std.sys import argv
from flint.assets.image import decode_png
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_image import hi3d_condition
from flint.platform.os import read_bytes
from flint.ml.hi3d_io import hi3d_write_rows


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("Usage: encode_image DINO.safetensors CLIP.safetensors INPUT.png TOKENS.f32")
    var image = decode_png(read_bytes(String(args[3])), 40000000)
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var tokens = hi3d_condition(pool, image, String(args[1]), String(args[2]))
    hi3d_write_rows(String(args[4]), tokens)
    print("Encoded", tokens.shape[1], "tokens of width", tokens.shape[2])
