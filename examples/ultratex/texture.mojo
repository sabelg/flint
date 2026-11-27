"""Bake both UltraTex materials into a static Hi3D GLB with a shared 9B backbone."""
from std.sys import argv
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.assets.image import decode_png
from flint.ml.hi3d_io import hi3d_read_rows
from flint.ml.tensor import Tensor
from flint.ml.ultratex import ultratex_noise, ultratex_views
from flint.ml.ultratex_dit import UltraTexDiT
from flint.ml.ultratex_mesh import UltraTexProjector
from flint.ml.ultratex_vae import UltraTexVAE, ultratex_encoder_path
from flint.platform.os import env_now, read_bytes


def main() raises:
    var args = argv()
    if len(args) < 5 or len(args) > 8:
        raise Error(
            "Usage: texture MODEL_DIR INPUT.glb REFERENCE.png OUTPUT.glb"
            " [view-size=2048] [steps=25] [seed=42]"
        )
    var root = String(args[1])
    var model = load_native_glb(String(args[2]))
    if (
        len(model.meshes) != 1
        or len(model.skins) > 0
        or model.meshes[0].morph_count > 0
    ):
        raise Error("Expected a static single-mesh Hi3D GLB")
    var side = Int(String(args[5])) if len(args) > 5 else 2048
    var steps = Int(String(args[6])) if len(args) > 6 else 25
    var seed = Int(String(args[7])) if len(args) > 7 else 42
    if steps < 1 or seed < 0 or seed > 0xFFFFFFFF:
        raise Error("Expected positive steps and an unsigned 32-bit seed")
    var projector = UltraTexProjector(model.meshes[0], side)
    var ref_image = decode_png(read_bytes(String(args[3])), 40000000)
    var noise = ultratex_noise(side, UInt32(seed))
    var dir = root + "/ultratex/"
    var ctx = hi3d_read_rows(dir + "empty-context.f32", 12288)
    if ctx.shape[0] != 512 or not ref_image.valid:
        raise Error(
            "Expected 512 cached Qwen3 tokens and a valid reference PNG"
        )
    var net = UltraTexDiT(
        root + "/flux-2-klein-base-9b.safetensors",
        dir + "flux2/lora/dit_lora.safetensors",
        use_gpu=env_now("FLINT_ULTRATEX_GPU") == "1",
    )
    var vae = UltraTexVAE(
        ultratex_encoder_path(root + "/ultratex"), dir + "decoder.safetensors"
    )
    var mesh = projector.mesh_with_uvs()
    var progress = env_now("FLINT_ULTRATEX_PROGRESS") == "1"
    for task in range(2):
        if progress:
            print(
                "Generating base color" if task
                == 0 else "Generating metallic-roughness"
            )
        if task == 1:
            net.load_adapter(dir + "flux2_mr/lora/dit_lora.safetensors")
        var views = ultratex_views(
            vae, net, projector.normals, ref_image, ctx, Tensor(), noise, steps
        )
        if task == 0:
            mesh.texture = len(model.textures)
        else:
            mesh.orm_texture = len(model.textures)
            mesh.metallic = 1
            mesh.roughness = 1
        model.textures.append(projector.blend(views, task == 1))
    model.meshes[0] = mesh^
    if not write_native_glb(String(args[4]), model):
        raise Error("Failed to write textured GLB")
