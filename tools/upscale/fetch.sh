#!/usr/bin/env bash
# Download the Real-ESRGAN checkpoints that `flint.ml.realesrgan` runs, into a folder outside the
# repository (default: ~/.cache/flint/realesrgan), then convert them to safetensors with
# tools/upscale/convert.py.
#
#     bash tools/upscale/fetch.sh [DEST] [--python /path/to/python-with-torch]
#
# The weights' licence is not stated by the upstream project (see licenses/README.md,
# "Real-ESRGAN"): they are a download you choose to make, never part of Flint.
set -euo pipefail

DEST="${1:-${HOME}/.cache/flint/realesrgan}"
PY="python3"
if [[ "${2:-}" == "--python" ]]; then
    PY="${3:?--python needs a path}"
fi
BASE="https://github.com/xinntao/Real-ESRGAN/releases/download"
mkdir -p "$DEST"

fetch() {
    local tag="$1" name="$2" sha="$3"
    local out="${DEST}/${name}"
    if [[ ! -f "$out" ]]; then
        echo "[upscale] downloading ${name} ..."
        curl -fsSL -o "${out}.part" "${BASE}/${tag}/${name}"
        mv "${out}.part" "$out"
    fi
    if [[ -n "$sha" ]]; then
        echo "${sha}  ${out}" | sha256sum -c --quiet - || { echo "[upscale] checksum mismatch: ${name}" >&2; exit 1; }
    fi
}

# name                               release    sha256 of the release assets (override with SHA_*=...)
fetch v0.1.0   RealESRGAN_x4plus.pth             "${SHA_X4PLUS:-4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1}"
fetch v0.2.2.4 RealESRGAN_x4plus_anime_6B.pth    "${SHA_ANIME6B:-f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da}"
fetch v0.2.5.0 realesr-general-x4v3.pth          "${SHA_GENERAL:-8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292}"
fetch v0.2.5.0 realesr-general-wdn-x4v3.pth      "${SHA_GENERAL_WDN:-1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca}"
fetch v0.2.5.0 realesr-animevideov3.pth          "${SHA_ANIMEVIDEO:-b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d}"

if "$PY" -c "import torch" >/dev/null 2>&1; then
    "$PY" "$(dirname "${BASH_SOURCE[0]}")/convert.py" "$DEST"
else
    echo "[upscale] $PY has no torch: run tools/upscale/convert.py $DEST with a Python that has torch"
fi
