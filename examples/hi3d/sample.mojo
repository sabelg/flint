"""Sample Hi3D's 512 sparse flow from supplied float32 noise."""
from std.sys import argv
from flint.ml.hi3d_io import hi3d_read_rows, hi3d_write_rows
from flint.ml.hi3d_transformer import Hi3DTransformer
from flint.ml.hi3d import hi3d_latents


def main() raises:
    var args = argv()
    if len(args) != 7 and len(args) != 8:
        raise Error(
            "Usage: sample TRANSFORMER_512_DIR COORDS.i32 CONDITION.f32 NOISE.f32 LATENTS.f32 STEPS"
            " [velocity-at-model-time]"
        )
    var root = String(args[1])
    var xyz = hi3d_read_rows(String(args[2]), 3, True)
    var ctx = hi3d_read_rows(String(args[3]), 2048)
    ctx.shape = [1, ctx.shape[0], 2048]
    var x = hi3d_read_rows(String(args[4]), 32)
    var net = Hi3DTransformer(root + "/model.safetensors", xyz^, ctx^)
    if len(args) == 8:
        hi3d_write_rows(String(args[5]), net.velocity(x, Float32(Int(String(args[7]))), True))
        return
    var latent = hi3d_latents(net, x^, root, Int(String(args[6])))
    hi3d_write_rows(String(args[5]), latent)
