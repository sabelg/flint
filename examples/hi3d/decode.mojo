"""Decode [N,32] Hi3D latents and [N,3] coordinates into a geometry GLB."""
from std.sys import argv
from flint.assets.glb_writer import write_native_glb
from flint.ml.hi3d_io import hi3d_read_rows
from flint.ml.hi3d_decoder import Hi3DField
from flint.ml.hi3d_mesh import hi3d_surface, hi3d_model


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("Usage: decode VAE.safetensors COORDS.i32 LATENTS.f32 REFINED.glb")
    var field = Hi3DField(
        hi3d_read_rows(String(args[2]), 3, True), hi3d_read_rows(String(args[3]), 32), String(args[1])
    )
    var model = hi3d_model(hi3d_surface(field.grid, field.attributes))
    if not write_native_glb(String(args[4]), model):
        raise Error("Could not write refined mesh")
    print("Decoded", len(model.meshes[0].indices) // 3, "triangles")
