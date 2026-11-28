"""Generate UltraTex six-view PNGs with a FLUX.2 Klein Base 9B checkpoint."""
from std.sys import argv
from flint.assets.image import decode_png, write_png
from flint.ml.hi3d_io import hi3d_read_rows
from flint.ml.tensor import Tensor
from flint.ml.ultratex import ultratex_views, ultratex_noise
from flint.ml.ultratex_dit import UltraTexDiT
from flint.ml.ultratex_vae import UltraTexVAE
from flint.platform.os import env_now, read_bytes


def main() raises:
    var args = argv()
    if len(args) < 9 or len(args) > 14:
        raise Error(
            "Usage: views BASE.safetensors ADAPTER.safetensors"
            " ENCODER.safetensors DECODER.safetensors EMPTY_CONTEXT.f32"
            " NORMALS.png REFERENCE.png OUTPUT.png [steps=25] [seed=42]"
            " [positive-context.f32|-] [noise.f32|-] [view-size]"
        )
    var steps = Int(String(args[9])) if len(args) > 9 else 25
    var seed = Int(String(args[10])) if len(args) > 10 else 42
    if steps < 1 or seed < 0 or seed > 0xFFFFFFFF:
        raise Error("Expected positive steps and unsigned 32-bit seed")
    var atlas = decode_png(read_bytes(String(args[6])), 25165824)
    var ref_image = decode_png(read_bytes(String(args[7])), 40000000)
    var ctx = hi3d_read_rows(String(args[5]), 12288)
    if ctx.shape[0] != 512:
        raise Error("Expected 512 cached Qwen3 context tokens")
    var positive = Tensor()
    if len(args) > 11 and String(args[11]) != "-":
        positive = hi3d_read_rows(String(args[11]), 12288)
        if positive.shape != ctx.shape:
            raise Error(
                "Expected matching Qwen3 positive and empty context shapes"
            )
    var side = Int(String(args[13])) if len(args) > 13 else atlas.width // 3
    var noise = hi3d_read_rows(String(args[12]), 128) if len(
        args
    ) > 12 and String(args[12]) != "-" else ultratex_noise(side, UInt32(seed))
    var net = UltraTexDiT(
        String(args[1]),
        String(args[2]),
        use_gpu=env_now("FLINT_ULTRATEX_GPU") == "1",
    )
    var vae = UltraTexVAE(String(args[3]), String(args[4]))
    var img = ultratex_views(
        vae, net, atlas, ref_image, ctx, positive, noise, steps, view_size=side
    )
    if not write_png(String(args[8]), img.pixels, img.width, img.height, False):
        raise Error("Failed to write generated view atlas")
