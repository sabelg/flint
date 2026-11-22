"""Bake a reference frond into three self-contained GLB assets."""
from std.sys import argv
from flint.assets.glb_writer import write_native_glb
from flint.assets.jpeg import decode_image_auto
from flint.assets.model import NativeModel
from flint.platform.os import read_bytes
from flint.procedural.reference_fern import FernReference, build_reference_fern


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("Usage: fern <upright-frond.png|jpg> <output-prefix>")
    var image = decode_image_auto(read_bytes(String(args[1])))
    if not image.valid:
        raise Error("Could not decode reference image " + String(args[1]))
    var reference = FernReference(image)
    for lod in range(3):
        var model = NativeModel()
        model.textures.append(reference.atlas.copy())
        model.meshes.append(build_reference_fern(reference, lod))
        var path = String(args[2]) + "-lod" + String(lod) + ".glb"
        if not write_native_glb(path, model):
            raise Error("Could not write " + path)
        print(path, len(model.meshes[0].indices) // 3, "triangles")
