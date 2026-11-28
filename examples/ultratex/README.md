# Texture a Hi3D mesh with UltraTex

Flint takes a static Hi3D mesh and an RGBA reference image, renders six world-space normal views, generates base color and metallic-roughness with UltraTex’s FLUX.2 adapters, and writes a textured glTF binary. Checkpoint preparation, Qwen3 text conditioning, diffusion, VAE decoding and material projection run in Mojo. The output has new UVs, seam vertices and embedded PNG textures. Geometry and vertex normals retain their original coordinates.

## Build

Run these commands from the Flint root.

```bash
mkdir -p build/ultratex
for app in pack context texture mesh views; do
  pixi run mojo build -I src -I tests -O3 --fp-mode contract=off --Werror \
    "examples/ultratex/$app.mojo" -o "build/ultratex/$app"
done
```

## Prepare the model directory

Set `models` to the directory containing the following separately distributed files. The texture tool loads the original FP32 autoencoder directly and uses the released UltraTex decoder. The context tool creates a reusable empty-prompt cache.

| File under the model directory | Source |
| --- | --- |
| `flux-2-klein-base-9b.safetensors` | Original BF16 single-file Klein **Base 9B** denoiser. [Black Forest Labs checkpoint and terms](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| `Qwen3-8B/` | Standard five SafeTensors shards with their index. [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `ae.safetensors` | Original FP32 FLUX.2 autoencoder used by UltraTex. SHA-256 `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`. [Official file, approved access required](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) |
| `ultratex/flux2/decoder.pt`, `ultratex/flux2/lora/dit_lora.safetensors`, `ultratex/flux2_mr/lora/dit_lora.safetensors` | Released foreground decoder, albedo adapter and metallic-roughness adapter. [UltraTex weights](https://www.modelscope.ai/models/Yibo-Zhang/UltraTex) |

```bash
models=../flint-models
weights="$models/ultratex"
pixi run ./build/ultratex/pack decoder "$weights/flux2/decoder.pt" "$weights/decoder.safetensors"
pixi run ./build/ultratex/context "$models/Qwen3-8B" "$weights/empty-context.f32"
```

The tested 9B backbone has SHA-256 `4a54fad7f5f741b99eee217198daac20b8d8e515e2a1f5b064fd51cf074f95bd`. The decoder packer checks the released archive's SHA-256 before reading its storage descriptors. The optional third argument to `context` is a file of little-endian int32 Qwen3 chat-template token IDs for a custom prompt. The built-in empty prompt matches the upstream demo and needs no tokenizer.

An existing installation with a packed `ultratex/encoder.safetensors` can still use that encoder when the original file is absent. The checkpoint packer also accepts the [public BF16 Diffusers autoencoder](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/vae/diffusion_pytorch_model.safetensors).

## Generate the GLB

```bash
FLINT_ULTRATEX_GPU=1 FLINT_ULTRATEX_PROGRESS=1 pixi run ./build/ultratex/texture \
  "$models" refined.glb reference.png textured.glb 2048 25 42
```

The last three arguments select pixels per view, denoising steps per material and the seed. Their defaults match the original release at 2048, 25 and 42. Use 512 for a smaller six-view render. The input must have one static indexed triangle mesh with vertex normals. An alpha channel in the reference isolates the object. The six views occupy a single atlas three tiles wide and two high. The material stores roughness in green and metallic in blue. Hidden surfaces can inherit a neighboring projection.

Set `FLINT_ULTRATEX_GPU=1` on NVIDIA to use native Mojo tensor-core projections, sparse attention, VAE convolutions and dense VAE attention. The denoiser keeps BF16 activations on the device across transformer blocks. CPU inference remains available. Each material uses its own LoRA adapter and the same initial noise. The implementation supports tiles from 16 through 2048 pixels in multiples of 16. Larger tiles increase memory use and runtime.

## Inspect individual material views

```bash
pixi run ./build/ultratex/mesh normals refined.glb 128 normals.png
for task in flux2 flux2_mr; do
  FLINT_ULTRATEX_GPU=1 pixi run ./build/ultratex/views \
    "$models/flux-2-klein-base-9b.safetensors" "$weights/$task/lora/dit_lora.safetensors" \
    "$models/ae.safetensors" "$weights/decoder.safetensors" "$weights/empty-context.f32" \
    normals.png reference.png "$task.png" 4 42
done
pixi run ./build/ultratex/mesh bake refined.glb flux2.png textured.glb flux2_mr.png
```

After the seed, `views` accepts a positive context file, a float32 noise file, and a view size. Use `-` for either optional file. Each context has shape `[512, 12288]`. The noise has shape `[6 * (view_size / 16)^2, 128]` in atlas row order. NVIDIA inference uses the CUDA Philox seed and launch geometry of the current GPU. A saved noise tensor reproduces a trial across machines. Supply the full-resolution normal atlas when evaluating a smaller view size so Flint computes the foreground mask from the original alpha channel.

## Verify

```bash
pixi run mojo build -I src -I tests -O3 --fp-mode contract=off --Werror \
  tests/test_flint_v5_ultratex.mojo -o build/ultratex/test
pixi run ./build/ultratex/test
FLINT_ULTRATEX_WEIGHTS="$weights" pixi run ./build/ultratex/test
FLINT_ULTRATEX_GPU=1 FLINT_ULTRATEX_WEIGHTS="$weights" pixi run ./build/ultratex/test
```

The standalone regression exercises sparse attention with partial blocks, a synthetic FLUX.2 denoiser with an independently computed velocity, view projection, UVs and texture sampling. Setting the weights directory adds checks against released VAE and material-adapter values and a two-step, six-view PyTorch reference image. This port covers FLUX.2 inference. Training and the alternate FLUX.1 backbone are outside its scope. [Upstream implementation](https://github.com/yiboz2001/UltraTex/tree/a726678f3f3631fe2c5e3d1fb4597114784c7a8b)

### Real-weight evaluation

On an NVIDIA GB10, Flint and UltraTex revision `a726678f3f3631fe2c5e3d1fb4597114784c7a8b` textured the same one-million-triangle Hi3D mesh using the original FP32 autoencoder, the released foreground decoder and the respective albedo and metallic-roughness adapters. Each run used seed 42, 25 denoising steps and 512 pixels per view. Flint generated its empty-prompt embedding with its native Qwen3 encoder. The reference used the Hugging Face Qwen3 encoder.

Across the 300,150 foreground pixels in each six-view atlas, the mean absolute difference per channel was 2.03 for base color and 2.18 for the metallic-roughness map on a 0–255 scale. The 95th-percentile differences were 6 and 8 respectively. The native GLB embeds both 1536 × 1024 atlases and preserves all 1,000,000 input triangles. In a separate comparison with identical projection inputs, all 50,331,648 merged BF16 weights and all 7,114,752 output values in the first image-attention projection matched the upstream CUDA calculation.

At the release defaults of 2048 pixels per view, 25 steps and seed 42, both implementations also rendered the upstream demo's six-view base-color atlas (6144 × 4096). This comparison used the same cached Qwen3 empty-prompt embedding and independently generated initial noise. Across 2,935,823 foreground pixels, the mean absolute channel difference was 0.92 on a 0–255 scale. The 95th-percentile difference was 3, and 98.98% of channel values were within 5 of upstream.
