#!/usr/bin/env bash
# Download Marigold-Normals v1.1 (prs-eth/marigold-normals-v1-1) for flint.ml.marigold.
#   bash tools/marigold/fetch.sh DIR [--fp32]
# Layout (diffusers): DIR/{model_index.json, scheduler/, unet/, vae/, text_encoder/, tokenizer/}
# The fp16 safetensors are stored under the standard names (Flint and the diffusers reference
# both upcast to float32); --fp32 fetches the full-precision files instead (~2x the size).
# Licence: CreativeML Open RAIL++-M (see licenses/marigold-LICENSE-MODEL.txt and
# licenses/README.md): commercial use is allowed, the Attachment A use restrictions apply.
# The files stay outside the repo; point examples/marigold or FLINT_MARIGOLD_DIR at DIR.
set -euo pipefail
DIR=${1:?usage: fetch.sh DIR [--fp32]}
VAR=.fp16
[ "${2:-}" = "--fp32" ] && VAR=
HF=https://huggingface.co/prs-eth/marigold-normals-v1-1/resolve/main
get() {
  if [ -s "$2" ]; then return 0; fi
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}
for f in model_index.json scheduler/scheduler_config.json unet/config.json vae/config.json \
         text_encoder/config.json tokenizer/vocab.json tokenizer/merges.txt \
         tokenizer/tokenizer_config.json tokenizer/special_tokens_map.json README.md; do
  get "$HF/$f" "$DIR/$f"
done
get "$HF/unet/diffusion_pytorch_model$VAR.safetensors" "$DIR/unet/diffusion_pytorch_model.safetensors"   # 1.7 GB fp16
get "$HF/vae/diffusion_pytorch_model$VAR.safetensors" "$DIR/vae/diffusion_pytorch_model.safetensors"     # 160 MB
get "$HF/text_encoder/model$VAR.safetensors" "$DIR/text_encoder/model.safetensors"                       # 680 MB
echo "Marigold-Normals v1.1 ready in $DIR"
