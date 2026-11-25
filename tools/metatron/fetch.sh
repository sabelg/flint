#!/usr/bin/env bash
# Download NVIDIA Metatron-v1 (NVIDIA Open Model License) and the Unitree G1 MJCF (BSD-3-Clause,
# via NVlabs/ProtoMotions) for flint.ml.metatron.
#   bash tools/metatron/fetch.sh DIR
# Layout:  DIR/model.onnx  DIR/weights.bin  DIR/NOTICE.txt  DIR/LICENSE.pdf  DIR/g1_holo_compat.xml
# The files stay outside the repo; point examples/metatron or METATRON_DIR at DIR.
set -euo pipefail
DIR=${1:?usage: fetch.sh DIR}
HF=https://huggingface.co/nvidia/Metatron-v1/resolve/main
PM=https://raw.githubusercontent.com/NVlabs/ProtoMotions/13de0cccd06f53ce5c9cf9ec7d1932c7da73610f
mkdir -p "$DIR"
get() {
  if [ -s "$2" ]; then return 0; fi
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}
get "$HF/demo/model.onnx" "$DIR/model.onnx"        # graph, external data in weights.bin
get "$HF/demo/weights.bin" "$DIR/weights.bin"      # fp32, ~403 MB
get "$HF/NOTICE.txt" "$DIR/NOTICE.txt"
get "$HF/LICENSE.pdf" "$DIR/LICENSE.pdf"
get "$PM/protomotions/data/assets/mjcf/g1_holo_compat.xml" "$DIR/g1_holo_compat.xml"
if command -v sha256sum >/dev/null; then
  echo "7180e82cae7ec04ff7766c262d430f57573d93678e025d8a71916885072a0cd4  $DIR/model.onnx" | sha256sum -c -
fi
echo "Metatron ready in $DIR"
