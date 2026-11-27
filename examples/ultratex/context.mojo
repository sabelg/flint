"""Build a reusable Qwen3-8B context cache natively for image-guided texturing."""
from std.sys import argv
from flint.ml.ultratex_text import ultratex_empty_tokens, ultratex_text
from flint.ml.hi3d_io import hi3d_read_rows, hi3d_write_rows
from flint.platform.os import env_now


def main() raises:
    var args = argv()
    if len(args) != 3 and len(args) != 4:
        raise Error("Usage: context QWEN3_8B_DIR OUTPUT.f32 [chat-template-token-ids.i32]")
    var ids = ultratex_empty_tokens()
    if len(args) == 4:
        var seq = hi3d_read_rows(String(args[3]), 1, True)
        ids = seq.i.copy()
    var ctx = ultratex_text(String(args[1]), ids, env_now("FLINT_ULTRATEX_GPU") == "1")
    hi3d_write_rows(String(args[2]), ctx)
