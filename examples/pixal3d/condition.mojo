"""Encode a reference cutout for Pixal3D's sparse-structure generator."""
from std.sys import argv
from flint.assets.image import decode_png
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_io import hi3d_write_rows
from flint.ml.pixal3d import PixalCamera, structure_condition
from flint.ml.trellis2_dino import TrellisImageEncoder
from flint.platform.os import read_bytes


def main() raises:
    var args = argv()
    if len(args) != 6:
        raise Error("Usage: condition DINO.safetensors CUTOUT.png FOV_RADIANS GLOBAL.f32 PROJECTED.f32")
    var camera = PixalCamera.from_fov(Float32(Float64(String(args[3]))))
    var image = decode_png(read_bytes(String(args[2])), 40000000)
    var encoder = TrellisImageEncoder(String(args[1]))
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var tokens = encoder.encode(pool, image, 512, 1.1)
    var conditions = structure_condition(tokens, camera)
    hi3d_write_rows(String(args[4]), conditions[0])
    hi3d_write_rows(String(args[5]), conditions[1])
    print("Encoded 5 global and 4096 projected DINOv3 tokens")
