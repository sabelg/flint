#!/usr/bin/env bash
# Install the fixed SOMA RP v1.1 and LLM2Vec model assets outside the repository.
set -euo pipefail
dest="${FLINT_KIMODO_DIR:-$HOME/.cache/flint/kimodo-soma-rp-v1.1}"
mkdir -p "$dest"
stage=$(mktemp -d "$dest/.install-XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
verify() { printf '%s  %s\n' "$1" "$2" | sha256sum --status -c -; }
fetch() {
    local name=$1 hash=$2 url=$3
    if [[ -f "$dest/$name" ]] && verify "$hash" "$dest/$name"; then return; fi
    curl --fail --location --retry 3 "$url" -o "$stage/$name"
    verify "$hash" "$stage/$name" || { echo "Checksum mismatch for $name" >&2; return 1; }
    mv -- "$stage/$name" "$dest/$name"
}
encoder=https://huggingface.co/KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX/resolve/883615966398adc12e787d9b3a2dc4da26d4b44c
denoiser=https://huggingface.co/KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX/resolve/da022f2180b84d4a788564219364785de6cb2ce0
stats=https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1/resolve/6c9233af1180b8151e3c4703477104af5dce9dd5/stats/motion
fetch denoiser.onnx 07341f1d89bea7a2a392a63040f39c048f8b0638ee1f50a5a0988bb8a9d7109b "$denoiser/onnx/model_fp32.onnx"
text=model_int4_b128_fp32act_s64.onnx_data
fetch "$text" 97beb9d2a6b48076512c83cc8b7fac84418e5b4cef3b3ab7ac45fa6c5bce9c8b "$encoder/onnx/$text"
fetch tokenizer.json 52b583ee1699c6ca4468f70885bd5f96dd1f7517292362a67581038afc04f4d8 "$encoder/tokenizer.json"
fetch global_root-mean.npy 6c110980f9a0386e57a8c186f8eac7f55dc05bade92f44c4b391450ada598da1 "$stats/global_root/mean.npy"
fetch global_root-std.npy fb15aeed7143f5f0754611f0acf5fa5f55d1f6492a49c23e490fc62f7bf11fb0 "$stats/global_root/std.npy"
fetch body-mean.npy d42f35f8bb7ef52fdec66ae29c5cf590bd038f4f1aabd2d560d1d0f78663ed6f "$stats/body/mean.npy"
fetch body-std.npy 965bbad497726ef4e6c46b519827e2cbcb29f974214b25718d20a725b82279ed "$stats/body/std.npy"
fetch joints.p f8582822d4f7cdd7c9cbb5ee8003f07d3116155fc49be11ed125d373d13e4e8a https://raw.githubusercontent.com/nv-tlabs/kimodo/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/kimodo/assets/skeletons/somaskel30/joints.p
unzip -p "$dest/joints.p" joints/data/0 > "$stage/joints.f64"
verify 5f96b584f9af27b53fd124f2aa3d8a09244c1301122763c3bc07641f1fd9bba7 "$stage/joints.f64"
mv -- "$stage/joints.f64" "$dest/joints.f64"
echo "Kimodo model files verified in $dest"
