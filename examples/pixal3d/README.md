# Pixal3D in Flint

Flint generates a static PBR GLB from one alpha-cutout PNG. The Mojo runtime uses the single-view structure, low-resolution shape, high-resolution shape, and texture checkpoints from Tencent ARC. DINOv3 and Neighborhood Attention Filtering supply camera-aligned features. All four flow stages and both sparse decoders run locally. The GPU kernels require an NVIDIA GPU with BF16 tensor cores. The reference run used a GB10 with 121 GiB of unified memory.

## Prepare the weights

The four single-view flow checkpoints occupy **22.0 GB**. The three decoders add **2.05 GB** and are byte-identical to the corresponding TRELLIS releases. Choose a model directory with sufficient space. The setup helper verifies SHA-256 before installing each checkpoint and can reuse existing decoder files. Python 3.11+, PyTorch, and SafeTensors are used only to prepare weights. The compiled generator runs entirely in Mojo.

```bash
python examples/pixal3d/prepare_weights.py "$HOME/flint-models/pixal3d" \
  --trellis-root "$HOME/flint-models/trellis2/microsoft"
```

Supply the original DINOv3 ViT-L/16 checkpoint separately. Its expected SHA-256 is `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`. [DINOv3 release](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

## Generate

```bash
mkdir -p build/pixal3d
pixi run mojo build examples/pixal3d/generate.mojo -I src -I tests \
  -O3 --fp-mode contract=off --Werror -o build/pixal3d/generate
build/pixal3d/generate "$HOME/flint-models/pixal3d/ckpts" \
  "$HOME/flint-models/model.safetensors" "$HOME/flint-models/pixal3d/naf.safetensors" \
  character-cutout.png 0.85 1024 42 12 build/pixal3d/character.glb
```

The last numeric arguments are the horizontal field of view in radians, requested cascade resolution (1024 or 1536), seed, and Euler step count per stage. A narrower field of view gives a more orthographic perspective. The 0.85 example uses the upstream manual-FOV camera fit. The released sampling schedule uses 12 steps. One step exercises the full pipeline for testing. At 1536, the upstream token budget can reduce the effective resolution down to 1024.

Engine callers can use `flint.ml.pixal3d_pipeline.generate_pixal3d` to obtain a `NativeModel`. The exporter extracts the learned flexible dual grid at native density, stores base color in vertex colors, and encodes learned metalness and roughness in a glTF material. This export differs from Tencent’s remeshed, decimated, UV-baked 4096-pixel texture atlas. Automatic background removal, MoGe camera estimation, multi-view checkpoints, and prompt-to-image generation are outside this single-view path.

To create a smaller mesh from the exported GLB, run Flint’s quadric-error simplifier with a triangle budget. It keeps the material and takes a separate output path.

```bash
pixi run mojo build examples/pixal3d/lod.mojo -I src -I tests \
  -O3 --fp-mode contract=off --Werror -o build/pixal3d/lod
build/pixal3d/lod build/pixal3d/character.glb 1000000 build/pixal3d/character-1m.glb
```

## Verify

```bash
pixi run mojo build tests/test_flint_v5_pixal3d.mojo -I src -I tests \
  -O3 --fp-mode contract=off --Werror -o build/pixal3d/test
build/pixal3d/test
```

The regression suite checks border-clamped projection, sparse coordinate ordering, cascade quantization, material packing, dual-grid topology, and continuity of the CPU noise stream. With the Sol reference, 512-pixel conditioning, and 0.85-radian FOV, the maximum absolute difference from the upstream PyTorch projection was 0.000035 across all 4,194,304 projected values. Seventeen sampled sparse sites agreed within 0.000064 on the high-resolution NAF branch. For the seed-42 one-step structure run, the native and upstream occupancy grids had an intersection-over-union of 0.9734. BF16 attention and subsequent thresholding can produce different occupied sites.

At 1024 resolution and 12 sampling steps per stage, the Sol Badguy cutout produced 2,604,901 vertices and 5,611,644 triangles. Flint reimported both the full GLB and a 1,000,000-triangle version with 376,379 vertices and the same metallic-roughness material. The untextured model shows the arm muscles and belt buckles. Small surface holes remain visible on the jacket, face, and boots. The 1536 mode has a coordinate regression test and still needs an end-to-end asset comparison.

[Pixal3D source revision](https://github.com/TencentARC/Pixal3D/tree/f7cf38429b0bd264f1995f0f8743a88b1c728b94) · [Single-view weights](https://huggingface.co/TencentARC/Pixal3D/tree/b0cb2e1b794cab9aa0ac38a95d794a4d9337437f/ckpts) · [NAF source and weights](https://github.com/valeoai/NAF)
