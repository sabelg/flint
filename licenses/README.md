# Third-party notices

- `src/flint/ml/motionbricks.mojo` is a Mojo port of parts of NVIDIA MotionBricks
  (https://github.com/NVlabs/motionbricks): the SOMA pose contract, input encoding, output
  decoding and endpoint correction. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
  source code licensed under Apache-2.0 — see `motionbricks-LICENSE.txt`.
- Flint does not include the MotionBricks models or motion clips. The ONNX models are under the
  NVIDIA Open Model Agreement; the sample motions under the NVIDIA Sample Data Evaluation
  License, which allows internal evaluation only, with no redistribution.
- `src/flint/ml/unirig.mojo`, `unirig_skin.mojo`, `unirig_rig.mojo` and `opt.mojo` port UniRig
  (https://github.com/VAST-AI-Research/UniRig, Copyright (c) 2025 VAST-AI-Research, MIT; see
  `unirig-LICENSE.txt`): the skeleton tokenizer and generation loop, the voxel-skin prior,
  mesh sampling, Point Transformer V3 serialisation (Pointcept, MIT, `pointcept-LICENSE.txt`) and the reskin step.
  `tools/unirig/export.py` converts the MIT-licensed checkpoints by importing a local UniRig
  checkout. The shape encoders inside those checkpoints come from Michelangelo (GPL-3.0), so the
  exported `unirig_skel_encoder.onnx` and `unirig_skin_global.onnx` are GPL-derived: Flint does
  not include them, and builds that redistribute Flint must not bundle them.
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
