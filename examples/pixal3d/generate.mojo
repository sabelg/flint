"""Generate a static PBR GLB with Flint's native Pixal3D pipeline."""
from std.sys import argv
from flint.assets.glb_writer import write_native_glb
from flint.assets.image import decode_png
from flint.ml.pixal3d_pipeline import generate_pixal3d
from flint.platform.os import read_bytes


def main() raises:
    var args = argv()
    if len(args) != 10:
        raise Error(
            "Usage: generate CKPTS_DIR DINO NAF CUTOUT.png FOV_RADIANS 1024|1536 SEED STEPS OUTPUT.glb"
        )
    var resolution = Int(String(args[6]))
    var seed = Int(String(args[7]))
    var steps = Int(String(args[8]))
    if (resolution != 1024 and resolution != 1536) or seed < 0 or seed > 0xFFFFFFFF or steps < 1:
        raise Error("Invalid Pixal3D resolution, seed or steps")
    var image = decode_png(read_bytes(String(args[4])), 40000000)
    var model = generate_pixal3d(
        image,
        String(args[1]),
        String(args[2]),
        String(args[3]),
        Float32(Float64(String(args[5]))),
        resolution,
        UInt32(seed),
        steps,
        verbose=True,
    )
    if not write_native_glb(String(args[9]), model):
        raise Error("Could not write Pixal3D GLB")
    print("Exported", len(model.meshes[0].indices) // 3, "triangles")
