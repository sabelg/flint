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
- `src/flint/ml/llm/` (`flint.ml.llm`) is an original pure-Mojo implementation of the Qwen3 and
  SmolLM3 decoder architectures, written from the published model configs and checked against
  Hugging Face Transformers outputs (`tools/llm/reference.py`); it contains no code from either
  project. The byte-level BPE alphabet and split regex it reproduces come from GPT-2 / tiktoken
  and the models' `tokenizer.json`. Flint does not include any weights or tokenizer files: users
  download them from Hugging Face. Qwen3-0.6B and Qwen3-1.7B are Apache-2.0 (Copyright 2024
  Alibaba Cloud, `qwen3-LICENSE.txt`); SmolLM3-3B is Apache-2.0 (Hugging Face,
  `smollm3-LICENSE.txt`). Both allow commercial use and redistribution with the notice.
- `src/flint/ml/kokoro.mojo` runs Kokoro-82M (https://huggingface.co/hexgrad/Kokoro-82M,
  StyleTTS2 + iSTFTNet, Copyright (c) hexgrad, Apache-2.0 code, weights and voicepacks; see
  `kokoro-LICENSE.txt`) from the onnx-community ONNX export, also Apache-2.0. Flint includes only
  the token vocabulary; `tools/kokoro/fetch.sh` downloads the model and voices outside the repo.
- `src/flint/ml/phonemize.mojo` ports the English G2P logic of misaki
  (https://github.com/hexgrad/misaki, Copyright (c) hexgrad, Apache-2.0; see `misaki-LICENSE.txt`)
  and reads its Apache-2.0 gold/silver lexicons, which `fetch.sh` also downloads. Flint does not
  use espeak-ng (GPL-3.0): the fallback for unknown words is Flint's own letter-to-sound rule set.
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
- `src/flint/ml/diffusion/` (`flint.ml.diffusion`) is an original pure-Mojo latent-diffusion
  stack: UNet, KL / VQ autoencoders, CLIP text and vision towers, CLIP BPE tokenizer, a
  `torch.Generator`-compatible RNG and DDIM / Euler / LCM schedulers. It is written from the model
  maths and checked against Hugging Face diffusers / transformers / PyTorch outputs
  (`tools/diffusion/reference.py`, Apache-2.0 / BSD libraries imported at verification time; no
  code is copied). Flint includes no weights: `tools/diffusion/fetch.py` downloads them outside
  the repository.
- StableMaterials (https://huggingface.co/gvecchio/StableMaterials) is published under an
  OpenRAIL licence, with no licence file in the repository. It allows commercial use, but its
  use-based restrictions must be passed on to anyone who runs it; see
  `stablematerials-LICENSE.txt`. Its CLIP ViT-L/14 text and vision encoders are OpenAI CLIP
  weights (MIT, `openai-clip-LICENSE.txt`).
- MatFuse (https://huggingface.co/gvecchio/MatFuse, Copyright (c) 2024 Giuseppe Vecchio, MIT; see
  `matfuse-LICENSE.txt`): UNet, VQ decoder and condition encoder. Its CLIP ViT-B/16 encoders are
  read from `sentence-transformers/clip-ViT-B-16`, a repackaging of OpenAI's CLIP ViT-B/16
  weights. That repository declares no licence of its own; the weights are OpenAI's
  (MIT via github.com/openai/CLIP, `openai-clip-LICENSE.txt`), the same ones MatFuse loads with
  `clip.load("ViT-B/16")`.
