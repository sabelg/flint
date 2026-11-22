from std.sys import argv
from editor import run_editor, save_sheet
from flint.ml.kimodo import Kimodo
from flint.ml.kimodo_text import KimodoTextEncoder, KimodoTokenizer
from flint.ml.kimodo_bvh import write_kimodo_bvh
from flint.platform.os import env_now


def main() raises:
    var args = argv()
    var dir = env_now("FLINT_KIMODO_DIR")
    if not dir:
        dir = env_now("HOME") + "/.cache/flint/kimodo-soma-rp-v1.1"
    if len(args) == 1:
        run_editor(String(args[0]), dir)
        return
    if len(args) == 4 and args[1] == "--sheet":
        save_sheet(String(args[2]), String(args[3]))
        return
    if len(args) < 3 or len(args) > 6:
        raise Error(
            'Usage: studio --kimodo "A person walks forward." output.bvh'
            " [frames] [steps] [seed]"
        )
    var frames = Int(String(args[3])) if len(args) > 3 else 60
    var steps = Int(String(args[4])) if len(args) > 4 else 20
    var seed = Int(String(args[5])) if len(args) > 5 else 1
    if (
        frames < 2
        or frames > 300
        or steps < 2
        or steps > 1000
        or seed < 0
        or seed > 4294967295
    ):
        raise Error(
            "Expected 2-300 frames, 2-1000 denoising steps and an unsigned"
            " 32-bit seed"
        )
    var tokenizer = KimodoTokenizer(dir + "/tokenizer.json")
    var ids = tokenizer.encode(String(args[1]))
    var encoder = KimodoTextEncoder(
        dir + "/model_int4_b128_fp32act_s64.onnx_data"
    )
    var vector = encoder.encode(ids)
    var kimodo = Kimodo(dir)
    var clip = kimodo.sample(vector, frames, steps, UInt32(seed))
    write_kimodo_bvh(clip, String(args[2]))
    print(
        "Wrote", clip.num_frames(), "SOMA frames at 30 fps to", String(args[2])
    )
