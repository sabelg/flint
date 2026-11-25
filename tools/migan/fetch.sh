#!/usr/bin/env bash
# Download MI-GAN-512-Places2 (Picsart AI Research, MIT code and weights) for flint.ml.migan.
#   bash tools/migan/fetch.sh DIR
# Writes DIR/migan.onnx (29 MB: uint8 image + mask in, uint8 image out, 512 x 512 generator) and
# DIR/LICENSE. The upload is the authors' (Hugging Face andraniksargsyan/migan, linked from the
# MI-GAN README); the weights' licence is LICENSE-WEIGHTS in github.com/Picsart-AI-Research/MI-GAN
# (MIT). The file stays outside the repo; point examples/migan or FLINT_MIGAN at DIR/migan.onnx.
set -euo pipefail
DIR=${1:?usage: fetch.sh DIR}
HF=https://huggingface.co/andraniksargsyan/migan/resolve/main
mkdir -p "$DIR"
get() {
  if [ -s "$2" ]; then return 0; fi
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}
get "$HF/migan.onnx" "$DIR/migan.onnx"
get "$HF/LICENSE" "$DIR/LICENSE"
echo "MI-GAN ready in $DIR"
