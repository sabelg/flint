"""Fetch the pinned single-view Pixal3D checkpoints and convert the NAF weight container."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


REVISION = "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f"
HUB = f"https://huggingface.co/TencentARC/Pixal3D/resolve/{REVISION}/ckpts"
CHECKPOINTS = {
    "shape_dec_next_dc_f16c32_fp16": "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
    "slat_flow_img2shape_dit_1_3B_1024_bf16": "3e6d1df0e62bc5bc863320d1d63d6b996235eef4abb8b851e878c88f316369f1",
    "slat_flow_img2shape_dit_1_3B_512_bf16": "63569838c1da93f010e2fa805b202ecf802cc758830230abac202a81b70e5b45",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16": "65df179ad02f9e92bc18335d7861fadd95f268e6249217352468dd83cb79d1d0",
    "ss_dec_conv3d_16l8_fp16": "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
    "ss_flow_img_dit_1_3B_64_bf16": "6d0147bb347eb61fa81c304cc68c7fd69d447e91870357a549b47e1d0fd77242",
    "tex_dec_next_dc_f16c32_fp16": "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
}
NAF = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_DIGEST = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"


def matches(path, digest):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def publish(temp, path, digest):
    try:
        os.link(temp, path)
    except FileExistsError:
        if not matches(path, digest):
            raise ValueError(f"Existing file differs from the release: {path}") from None


def fetch(url, path, digest):
    if matches(path, digest):
        return
    if path.exists() or path.is_symlink():
        raise ValueError(f"Existing file differs from the release: {path}")
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        name = Path(temp.name)
    try:
        print(f"Downloading {path.name}", flush=True)
        with urlopen(url, timeout=120) as response, name.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=4 << 20)
        if not matches(name, digest):
            raise ValueError(f"Incomplete download or checksum mismatch: {path.name}")
        publish(name, path, digest)
    finally:
        name.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trellis-root", type=Path, help="Reuse verified local TRELLIS decoder files")
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file

    root = args.directory.resolve()
    ckpts = root / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)
    for key, digest in CHECKPOINTS.items():
        name = key + ".safetensors"
        dest = ckpts / name
        if args.trellis_root and "_dec_" in key and not dest.exists():
            source = next((p for p in args.trellis_root.rglob(name) if matches(p, digest)), None)
            if source:
                try:
                    dest.symlink_to(source.resolve())
                except FileExistsError:
                    pass  # A second installer may already have published it. fetch verifies it below.
        fetch(f"{HUB}/{name}", dest, digest)
    original = root / "naf_release.pth"
    fetch(NAF, original, NAF_DIGEST)
    state = torch.load(original, map_location="cpu", weights_only=True)
    with NamedTemporaryFile(dir=root, delete=False) as temp:
        path = Path(temp.name)
    try:
        save_file(state, str(path))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        publish(path, root / "naf.safetensors", digest)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
