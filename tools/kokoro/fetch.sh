#!/usr/bin/env bash
# Download Kokoro-82M (Apache-2.0) and the misaki English lexicons (Apache-2.0) for flint.ml.kokoro.
#   bash tools/kokoro/fetch.sh DIR [voice ...]     (default voices: af_heart am_michael bf_emma bm_george)
# Layout:  DIR/model.onnx  DIR/voices/<name>.bin  DIR/lexicon/{us,gb}_{gold,silver}.json
# The files stay outside the repo; point examples/kokoro or FLINT_KOKORO_DIR at DIR.
set -euo pipefail
DIR=${1:?usage: fetch.sh DIR [voice ...]}
shift || true
VOICES=("$@")
[ ${#VOICES[@]} -eq 0 ] && VOICES=(af_heart am_michael bf_emma bm_george)
HF=https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main
LEX=https://raw.githubusercontent.com/hexgrad/misaki/main/misaki/data
mkdir -p "$DIR/voices" "$DIR/lexicon"
get() {
  if [ -s "$2" ]; then return 0; fi
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}
get "$HF/onnx/model.onnx" "$DIR/model.onnx"                  # fp32, ~326 MB
for v in "${VOICES[@]}"; do get "$HF/voices/$v.bin" "$DIR/voices/$v.bin"; done
for f in us_gold us_silver gb_gold gb_silver; do get "$LEX/$f.json" "$DIR/lexicon/$f.json"; done
echo "Kokoro ready in $DIR"
