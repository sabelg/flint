#!/usr/bin/env bash
# Download Magenta RealTime 2 (google/magenta-realtime-2: code Apache-2.0, weights CC-BY 4.0,
# Copyright 2026 Google LLC) for flint.ml.magenta_rt and convert MusicCoCa for flint.ml.musiccoca.
#   bash tools/magenta_rt/fetch.sh DIR [PYTHON]
# Layout:  DIR/checkpoints/mrt2_small.safetensors            (LM + SpectroStream decoder, ~0.9 GB)
#          DIR/resources/musiccoca/*.tflite, spm.model        (MusicCoCa, TFLite)
#          DIR/resources/musiccoca/musiccoca_text.safetensors (written by convert_musiccoca.py)
# PYTHON needs numpy, ai_edge_litert, tflite, sentencepiece and safetensors (pip install
# numpy ai-edge-litert tflite sentencepiece safetensors). The files stay outside the repo; point
# examples/magenta_rt or FLINT_MAGENTA_RT_DIR at DIR.
set -euo pipefail
DIR=${1:?usage: fetch.sh DIR [PYTHON]}
PY=${2:-python3}
HF=https://huggingface.co/google/magenta-realtime-2/resolve/main
HERE=$(cd "$(dirname "$0")" && pwd)
get() {
  if [ -s "$DIR/$1" ]; then return 0; fi
  mkdir -p "$(dirname "$DIR/$1")"
  curl -fL --retry 3 -o "$DIR/$1.part" "$HF/$1"
  mv "$DIR/$1.part" "$DIR/$1"
}
get checkpoints/mrt2_small.safetensors
for f in spm.model text_encoder.tflite pretrained_vector_quantizer.tflite mapper.tflite; do
  get "resources/musiccoca/$f"
done
if [ ! -s "$DIR/resources/musiccoca/musiccoca_text.safetensors" ]; then
  "$PY" -I "$HERE/convert_musiccoca.py" "$DIR"
fi
echo "Magenta RT 2 ready in $DIR"
