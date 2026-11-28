"""Refine a coarse GLB with a reference image and the Hi3D 512 checkpoints."""
from std.sys import argv
from flint.assets.image import decode_png
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.ml.hi3d import refine_hi3d
from flint.platform.os import read_bytes


def main() raises:
    var args = argv()
    if len(args) < 7 or len(args) > 9:
        raise Error(
            "Usage: refine MODEL_DIR DINO.safetensors CLIP.safetensors IMAGE.png COARSE.glb OUTPUT.glb"
            " [steps=25] [seed=0]"
        )
    var steps = Int(String(args[7])) if len(args) > 7 else 25
    var seed = Int(String(args[8])) if len(args) > 8 else 0
    if seed < 0 or seed > 0xFFFFFFFF:
        raise Error("Seed must fit uint32")
    var coarse = load_native_glb(String(args[5]))
    var image = decode_png(read_bytes(String(args[4])), 40000000)
    var refined = refine_hi3d(
        String(args[1]), String(args[2]), String(args[3]), image, coarse, UInt32(seed), steps
    )
    if not write_native_glb(String(args[6]), refined):
        raise Error("Could not write refined GLB")
    print("Exported", len(refined.meshes[0].indices) // 3, "triangles")
