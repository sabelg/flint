# Hi3D 512³ geometry refinement

[Powered by Hi3D 1.0](http://hi3d.ai/)

Give the refiner a reference PNG and a coarse triangle mesh of the same object. It runs the Hi3D v1.0 512³ pass on the CPU and writes a geometry-only GLB with vertex normals. The output is centered and scaled to unit radius. Preserve the input mesh if you need its original scene placement, materials, UVs, or rig.

## Build and run

Run these commands from the Flint repository root. The example assumes the model files are in a sibling `flint-models/hi3d` directory.

```bash
mkdir -p build/hi3d
pixi run mojo build -I src -I tests -O3 --fp-mode contract=off --Werror \
  examples/hi3d/refine.mojo -o build/hi3d/refine
weights=../flint-models/hi3d
pixi run ./build/hi3d/refine "$weights/model" "$weights/dino/model.safetensors" \
  "$weights/clip/model.safetensors" reference.png coarse.glb refined.glb
```

The optional arguments after the output path are the step count (default 25) and an unsigned 32-bit seed (default 0). One step is useful for checking the installation. Use 25 steps to evaluate refinement quality. Runtime and memory depend on the number of occupied mesh cells. The Mojo path uses CPU workers throughout.

## Model files

| CLI argument | Required files |
| --- | --- |
| Hi3D model directory | `transformer_512/config.json`, `transformer_512/model.safetensors`, `vae/model.safetensors` from [Hi3D v1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0) |
| DINOv2 checkpoint | `model.safetensors` from [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) |
| CLIP checkpoint | `model.safetensors` from [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |

The loader reads the original checkpoints from disk. See the [license and notices](../../licenses/README.md#hi3d-v10).

## Engine API

Import `refine_hi3d` from `flint.ml.hi3d`. It accepts the model directory, the two vision checkpoint paths, a decoded `NativeImage`, a mutable coarse `NativeModel`, and optional `seed` and `steps` arguments in that order. It returns a `NativeModel`. Mesh nodes are voxelized in world space. For predictable image framing, supply an RGBA PNG with a transparent background.

The implementation encodes the image with DINOv2 and CLIP, voxelizes the coarse surface, runs the 24-block sparse flow transformer with a Heun solver, and decodes the sparse field with the Hi3D VAE and FlexiCubes. Flint's quadric mesh simplifier targets a maximum of one million triangles. Floating-point arithmetic and simplification can produce different vertices and topology from the upstream GPU implementation.

## Inspect individual stages

Build these entry points with the same flags as `refine.mojo`. Intermediate arrays have no header and use little-endian values.

| Entry point | Arguments | Output |
| --- | --- | --- |
| `voxelize.mojo` | `COARSE.glb COORDS.i32` | Ordered xyz triples on the 64³ support grid |
| `encode_image.mojo` | `DINO.safetensors CLIP.safetensors INPUT.png CONDITION.f32` | 2,917 tokens of 2,048 floats |
| `sample.mojo` | `TRANSFORMER_512_DIR COORDS.i32 CONDITION.f32 NOISE.f32 LATENTS.f32 STEPS` | 32 denormalized latent channels per occupied cell |
| `decode.mojo` | `VAE.safetensors COORDS.i32 LATENTS.f32 OUTPUT.glb` | Refined mesh |

For a transformer parity check, append an integer model time to the sampler arguments. This writes one conditional velocity prediction in place of denormalized latents. The ordinary engine entry point generates the noise from the seed.

The model-independent Hi3D regressions run in the Flint v5 ML suite with `bash tools/flint.sh test v5`.

Set `FLINT_HI3D_TRANSFORMER` to the `transformer_512` checkpoint directory to include a numerical comparison against the upstream GPU implementation. The 1 KiB fixture records the 32 denormalized channels at eight synthetic sparse sites after one Heun step.
