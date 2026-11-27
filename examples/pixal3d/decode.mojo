"""Decode normalized Pixal3D shape and optional material latents to GLB."""
from std.sys import argv
from flint.assets.glb_writer import write_native_glb
from flint.ml.hi3d_io import hi3d_read_rows
from flint.ml.pixal3d_decoder import decode_slat, denormalize_slat
from flint.ml.pixal3d_mesh import pixal_surface
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor


def main() raises:
    var args = argv()
    if len(args) != 6 and len(args) != 8:
        raise Error("Usage: decode SHAPE_DEC COORDS.i32 SHAPE.f32 GRID OUT.glb [TEX_DEC TEX.f32]")
    var edge = Int(String(args[4]))
    var coords = hi3d_read_rows(String(args[2]), 4, True)
    var masks = List[Tensor]()
    var shape = decode_slat(
        SafeTensorWeights(String(args[1]), 2 << 30, True),
        coords.copy(),
        denormalize_slat(hi3d_read_rows(String(args[3]), 32), False),
        edge,
        masks,
    )
    var materials = Tensor()
    if len(args) == 8:
        var tex = decode_slat(
            SafeTensorWeights(String(args[6]), 2 << 30, True),
            coords^,
            denormalize_slat(hi3d_read_rows(String(args[7]), 32), True),
            edge,
            masks,
            True,
        )
        materials = tex[1].copy()
    var model = pixal_surface(shape[0], shape[1], materials, edge * 16)
    if not write_native_glb(String(args[5]), model):
        raise Error("Failed to write Pixal3D GLB")
    print("Exported", len(model.meshes[0].indices) // 3, "triangles")
