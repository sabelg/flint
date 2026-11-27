# TRELLIS.2 structure inference in Mojo

For a reference PNG with an alpha cutout, `flint.ml.trellis2_image.prepare_trellis_image` takes a decoded `NativeImage` and returns normalized [1,3,H,W] RGB input for the 512 or 1024 DINOv3 encoder. It crops to the foreground, composites on black and applies the upstream Lanczos resampling. Opaque images require a foreground mask.

The native decoder reads Microsoft's original SafeTensors weights and turns an eight-channel spatial latent into occupancy logits. It runs on the CPU with Flint's SIMD matrix multiplication and job pool. The structure Euler sampler implements the published 12-step schedule, interval guidance and rescaling. It takes a velocity predictor as an argument. The learned structure predictor, shape and texture networks, and O-Voxel mesh extraction are outstanding. This command takes a latent as input.

From the Flint checkout, fetch the pinned decoder outside source control and run it on the included reference latent.

```bash
mkdir -p build/trellis2
curl -fL --retry 2 'https://huggingface.co/microsoft/TRELLIS-image-large/resolve/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors' -o build/trellis2/structure.safetensors
echo '1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a  build/trellis2/structure.safetensors' | sha256sum --check
pixi run mojo build examples/trellis2/decode_structure.mojo -I src -I tests -O3 --fp-mode contract=off --Werror -o build/trellis2/decode
build/trellis2/decode build/trellis2/structure.safetensors tests/fixtures/trellis2/open_cube_latent.f32 16 build/trellis2/occupancy.f32
```

The input stores little-endian float32 values in channel, depth, height, width order. A native TRELLIS latent has dimensions 8 × 16 × 16 × 16. The output contains 64³ float32 logits in depth, height, width order. Positive values indicate occupied voxels. The library function `structure_coordinates(logits, resolution)` performs thresholding and max pooling, then emits batch, depth, height and width coordinates.

## Encode a reference image

The 24-layer native ViT-L/16 image encoder reads the original Meta SafeTensors checkpoint. Use an alpha-cutout PNG. The output is a little-endian float32 tensor with shape [1,1029,1024] at 512 pixels or [1,4101,1024] at 1024 pixels. It contains the class token, four register tokens and the spatial patch tokens, with TRELLIS's final normalization. A subsequent learned flow network consumes these image features to predict the structure latent.

```bash
export FLINT_TRELLIS2_DINO="$HOME/flint-models/model.safetensors"
mkdir -p build/trellis2
pixi run mojo build examples/trellis2/encode_image.mojo -I src -I tests -O3 --fp-mode contract=off --Werror -o build/trellis2/encode
build/trellis2/encode "$FLINT_TRELLIS2_DINO" tests/fixtures/trellis2/encoder_cutout.png 512 build/trellis2/features.f32
```

The expected encoder SHA-256 is `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`. The encoder checkpoint remains external and has its own [model license and download page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

## Verify against the reference

```bash
FLINT_TRELLIS2_DECODER="$PWD/build/trellis2/structure.safetensors" pixi run mojo run tests/test_trellis2.mojo -I src -I tests -O3 --fp-mode contract=off --Werror
```

The fixture starts with a 64³ empty grid. A one-voxel shell occupies the faces of the cube from 16 through 47 on each axis. On the face at width 16, the opening spans depth and height 22 through 41 inclusive. The reference uses the corresponding Microsoft encoder checkpoint at the same revision, the posterior mean as the latent, and the published decoder with PyTorch FP16 convolutions in the residual and upsample blocks. The mask stores successive width-first occupancy decisions starting at the low bit of each byte. The test checks every decision and samples signed logits within 0.2. The fixture produces 5,368 occupied voxels. The sampler's numerical test uses a deterministic stand-in predictor so the schedule and guidance can be tested independently.

[Microsoft's model configuration](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/af44b45f2e35a493886929c6d786e563ec68364d/pipeline.json) · [Decoder architecture](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/sparse_structure_vae.py) · [Flow sampler](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/samplers)
