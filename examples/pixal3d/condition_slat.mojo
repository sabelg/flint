"""Build sparse shape or texture conditioning using the released NAF upsampler."""
from std.sys import argv
from flint.assets.image import decode_png
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_io import hi3d_read_rows, hi3d_write_rows
from flint.ml.pixal3d import PixalCamera
from flint.ml.pixal3d_naf import sparse_condition
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.trellis2_dino import TrellisImageEncoder
from flint.ml.trellis2_image import prepare_trellis_image
from flint.platform.os import read_bytes


def main() raises:
    var args = argv()
    if len(args) != 11:
        raise Error(
            "Usage: condition_slat DINO NAF CUTOUT.png FOV IMAGE_SIZE NAF_SIZE GRID COORDS.i32 GLOBAL.f32"
            " PROJ.f32"
        )
    var size = Int(String(args[5]))
    var camera = PixalCamera.from_fov(Float32(Float64(String(args[4]))))
    var image = decode_png(read_bytes(String(args[3])), 40000000)
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var encoder = TrellisImageEncoder(String(args[1]))
    var tokens = encoder.encode(pool, image, size, 1.1)
    var weights = SafeTensorWeights(String(args[2]), 8 << 20)
    var projected = sparse_condition(
        tokens,
        prepare_trellis_image(image, size, 1.1),
        hi3d_read_rows(String(args[8]), 4, True),
        Int(String(args[7])),
        camera,
        weights,
        Int(String(args[6])),
    )
    var globals = Tensor.zeros([5, 1024])
    for i in range(globals.numel()):
        globals.f[i] = tokens.f[i]
    hi3d_write_rows(String(args[9]), globals)
    hi3d_write_rows(String(args[10]), projected)
