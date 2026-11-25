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
- `src/flint/ml/kokoro.mojo` runs Kokoro-82M (https://huggingface.co/hexgrad/Kokoro-82M,
  StyleTTS2 + iSTFTNet, Copyright (c) hexgrad, Apache-2.0 code, weights and voicepacks; see
  `kokoro-LICENSE.txt`) from the onnx-community ONNX export, also Apache-2.0. Flint includes only
  the token vocabulary; `tools/kokoro/fetch.sh` downloads the model and voices outside the repo.
- `src/flint/ml/phonemize.mojo` ports the English G2P logic of misaki
  (https://github.com/hexgrad/misaki, Copyright (c) hexgrad, Apache-2.0; see `misaki-LICENSE.txt`)
  and reads its Apache-2.0 gold/silver lexicons, which `fetch.sh` also downloads. Flint does not
  use espeak-ng (GPL-3.0): the fallback for unknown words is Flint's own letter-to-sound rule set.
