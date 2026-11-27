# Third-party notices

- `src/flint/ml/motionbricks.mojo` is a Mojo port of parts of NVIDIA MotionBricks
  (https://github.com/NVlabs/motionbricks): the SOMA pose contract, input encoding, output
  decoding and endpoint correction. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
  source code licensed under Apache-2.0 — see `motionbricks-LICENSE.txt`.
- Flint does not include the MotionBricks models or motion clips. The ONNX models are under the
  NVIDIA Open Model Agreement; the sample motions under the NVIDIA Sample Data Evaluation
  License, which allows internal evaluation only, with no redistribution.

## Kimodo

The native SOMA feature decoding and diffusion sampler follow NVIDIA Kimodo, copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. See `kimodo-LICENSE.txt` for its Apache-2.0 source license. The separate model installer downloads the [SOMA model conversion](https://huggingface.co/KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX) and the [LLM2Vec conversion built with Meta Llama 3](https://huggingface.co/KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX). The model cards link their respective terms and source revisions.

## Body capture

The offline RTMPose converter rewrites a user-supplied ONNX graph. The MotionBERT converter imports DSTformer from a user-supplied upstream checkout and reads the H36M fine-tuned checkpoint. Prepared weights live in the user's cache. Review the upstream terms for each checkpoint and dataset before distributing a model. [RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) · [MotionBERT project and license](https://github.com/Walter0807/MotionBERT)

## HY-Motion Lite

The preparation tools import the upstream Hunyuan model definitions and read locally obtained Lite, Qwen3-8B and CLIP-L weights. The generated inference graphs, packed text weights and prompt cache live outside the repository. Review each project's terms for the intended use. [Tencent HY-Motion source and license](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) · [HY-Motion weights](https://huggingface.co/tencent/HY-Motion-1.0) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [CLIP-L](https://huggingface.co/openai/clip-vit-large-patch14)

## TRELLIS.2

Flint's structure decoder and flow sampler implement the published Microsoft TRELLIS.2 algorithms. The Microsoft MIT notice is in [trellis2-LICENSE.txt](trellis2-LICENSE.txt). The decoder reads the original, separately distributed [TRELLIS-image-large structure checkpoint](https://huggingface.co/microsoft/TRELLIS-image-large). Tests include a latent and an occupancy mask for a synthetic hollow cube. The native DINOv3 encoder loads a separately supplied [Meta ViT-L/16 checkpoint under the DINOv3 license](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). The repository ships no TRELLIS or DINOv3 model weights.

## GEM-X & SOMA-X

- `src/flint/ml/gemx.mojo` ports the inference path of NVIDIA GEM-X
  (https://github.com/NVlabs/GEM-X, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
  Apache-2.0, see `gemx-LICENSE.txt`): the native77 crops and heatmap decoding, box smoothing,
  the Motion conditions and attention window, the SOMA output decoding, camera translation and
  world rollout. Its 6-D rotation decoding follows PyTorch3D (BSD-3-Clause, as credited in the
  GEM-X `ATTRIBUTIONS.md`). The crops reproduce the arithmetic of OpenCV's
  `getAffineTransform` / `warpAffine` (Apache-2.0); no OpenCV code is included.
  `tools/gemx/export.py` and `tools/gemx/reference.py` import a local GEM-X checkout.
- The SOMA joint names, hierarchy and T-pose joint orientations in `gemx.mojo` come from
  SOMA-X (`SOMA_neutral.npz`, https://huggingface.co/nvidia/soma-x, Apache-2.0, see
  `soma-x-LICENSE.txt`).
- Flint does not include the GEM-X weights or the converted graphs. The checkpoints
  (https://huggingface.co/nvidia/GEM-X) are under the NVIDIA Open Model License, which allows
  commercial use. Both image encoders inside `gemx_vitpose.onnx` and `gemx_pose.onnx` are Meta
  DINOv3 ViT-H/16+ backbones under the DINOv3 License (commercial use allowed; redistribution
  must carry that licence, and its trade-control and prohibited-use terms apply).
- GEM-X's YOLOX person detector is not ported and must not be added: its weights are trained on
  the Human-Art dataset, which is licensed for non-commercial use only. Person boxes come from
  the caller's detector or from `Gemx.track` (keypoint tracking from a first-frame box).

## Hi3D v1.0

Flint ports the 512³ inference pipeline from [Hi3D v1.0, revision 13657ad](https://github.com/hi3d-official/Hi3Dv1.0/tree/13657ad09afaa92e350b461a1cd66399e44a59c4) to Mojo. The sparse operators, vision encoders, flow solver, decoder and mesh extraction were adapted for Flint’s CPU runtime. The original [license](hi3d/LICENSE), [notice](hi3d/NOTICE), [author credits](hi3d/AUTHORS), [third-party notices](hi3d/THIRD_PARTY_LICENSES.md) and [FlexiCubes license](hi3d/FlexiCubes-LICENSE.txt) accompany the port. Users supply model weights separately.

## UltraTex

The FLUX.2 multi-view inference port follows UltraTex revision a726678f3f3631fe2c5e3d1fb4597114784c7a8b. The port adapts the SLA team's block-sparse attention and Black Forest Labs' FLUX.2 architecture. [UltraTex source and MIT notice](ultratex/LICENSE) · [SLA Apache-2.0 license](ultratex/SLA-LICENSE.txt) · [FLUX.2 source license](ultratex/FLUX2-LICENSE.md)

Model weights are obtained separately. The Klein Base 9B checkpoint is subject to Black Forest Labs' FLUX non-commercial license. The regression fixtures contain synthetic test weights. [Klein Base 9B model card and license](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) · [UltraTex adapters and decoder](https://www.modelscope.ai/models/Yibo-Zhang/UltraTex) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
