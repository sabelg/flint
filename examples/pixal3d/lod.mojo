"""Create a smaller Pixal3D asset with Flint's quadric-error mesh simplifier."""
from std.sys import argv
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.assets.meshlet import simplify_mesh_qem


def main() raises:
    var args = argv()
    if len(args) != 4 or args[1] == args[3]:
        raise Error("Usage: lod INPUT.glb TRIANGLE_BUDGET OUTPUT.glb (use a separate output path)")
    var budget = Int(String(args[2]))
    if budget < 1:
        raise Error("Expected a positive triangle budget")
    var model = load_native_glb(String(args[1]))
    if len(model.meshes) != 1:
        raise Error("Expected a single Pixal3D surface")
    var reduced = simplify_mesh_qem(model.meshes[0], budget)
    model.meshes[0] = reduced[0].copy()
    var triangles = len(model.meshes[0].indices) // 3
    if not triangles or not write_native_glb(String(args[3]), model):
        raise Error("Could not export simplified Pixal3D surface")
    print("Exported", triangles, "triangles")
