"""Export the 512 refiner's mesh support as little-endian int32 xyz triples."""
from std.sys import argv
from flint.assets.glb_native import parse_native_glb
from flint.ml.hi3d_voxels import hi3d_model_voxels
from flint.platform.os import read_bytes
from flint.ml.hi3d_io import hi3d_write_rows


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("Usage: voxelize COARSE.glb COORDS.i32")
    var scene = parse_native_glb(read_bytes(String(args[1])))
    var xyz = hi3d_model_voxels(scene)
    hi3d_write_rows(String(args[2]), xyz)
    print("Voxelized", xyz.shape[0], "surface cells")
