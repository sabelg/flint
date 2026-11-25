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
- `src/flint/ml/moonshine.mojo`, `moonshine_gpu.mojo`, `moonshine_audio.mojo` and
  `moonshine_tokenizer.mojo` implement Moonshine v2 streaming speech-to-text
  (https://github.com/moonshine-ai/moonshine, Copyright (c) 2025 Useful Sensors, Inc. dba
  Moonshine AI, MIT; see `moonshine-LICENSE.txt`). The code and every streaming model are MIT;
  only the repository's legacy non-English *non-streaming* models are under the non-commercial
  Moonshine Community License, and Flint does not target them. Flint does not include the
  weights: `Moonshine.load` reads a downloaded `moonshine-ai/moonshine-streaming-*` folder. The
  Mojo code is written from the model's math and checked against Hugging Face `transformers`
  (Apache-2.0) with `tools/moonshine/reference.py`, which imports it at verification time; no
  `transformers` code is copied.
