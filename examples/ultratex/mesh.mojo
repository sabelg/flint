"""Prepare normal views or attach registered view textures to a native Hi3D mesh."""
from std.sys import argv
from flint.assets.image import decode_png, encode_png
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.ml.ultratex_mesh import UltraTexProjector
from flint.platform.os import read_bytes, write_bytes


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "Usage: mesh normals INPUT.glb TILE_SIZE NORMALS.png | mesh bake INPUT.glb ALBEDO.png OUTPUT.glb"
            " [glTF-MR.png]"
        )
    var model = load_native_glb(String(args[2]))
    if len(model.meshes) != 1 or len(model.skins) > 0 or model.meshes[0].morph_count > 0:
        raise Error("Expected a static single-mesh Hi3D GLB")
    if String(args[1]) == "normals" and len(args) == 5:
        var projector = UltraTexProjector(model.meshes[0], Int(String(args[3])))
        var png = encode_png(
            projector.normals.pixels, projector.normals.width, projector.normals.height, "", True
        )
        if not write_bytes(String(args[4]), png):
            raise Error("Failed to write normal atlas")
    elif String(args[1]) == "bake" and (len(args) == 5 or len(args) == 6):
        var image = decode_png(read_bytes(String(args[3])), 25165824)
        var projector = UltraTexProjector(model.meshes[0], image.width // 3)
        var mesh = projector.mesh_with_uvs()
        mesh.texture = len(model.textures)
        model.textures.append(projector.blend(image))
        if len(args) == 6:
            var mr = decode_png(read_bytes(String(args[5])), 25165824)
            mesh.orm_texture = len(model.textures)
            mesh.metallic = 1
            mesh.roughness = 1
            model.textures.append(projector.blend(mr, True))
        model.meshes[0] = mesh^
        if not write_native_glb(String(args[4]), model):
            raise Error("Failed to write textured mesh")
    else:
        raise Error("Invalid UltraTex mesh command or argument count")
