"""Sample normalized Pixal3D shape or texture latent rows from supplied noise."""
from std.sys import argv
from flint.ml.hi3d_io import hi3d_read_rows, hi3d_write_rows
from flint.ml.pixal3d_flow import PixalFlow
from flint.ml.tensor import Tensor
from flint.ml.trellis2_flow import sample_pixal_slat


def main() raises:
    var args = argv()
    if len(args) != 8 and len(args) != 9:
        raise Error(
            "Usage: slat FLOW COORDS.i32 GLOBAL.f32 PROJECTED.f32 NOISE.f32 STEPS OUT.f32"
            " [NORMALIZED_SHAPE.f32]"
        )
    var coords = hi3d_read_rows(String(args[2]), 4, True)
    var texture = len(args) == 9
    var shape = hi3d_read_rows(String(args[8]), 32) if texture else Tensor()
    var model = PixalFlow(
        String(args[1]),
        hi3d_read_rows(String(args[3]), 1024),
        hi3d_read_rows(String(args[4]), 2048),
        coords,
        shape^,
    )
    var sampled = sample_pixal_slat(model, hi3d_read_rows(String(args[5]), 32), Int(String(args[6])), texture)
    hi3d_write_rows(String(args[7]), sampled)
    print("Sampled", sampled.shape[0], "sparse latent rows")
