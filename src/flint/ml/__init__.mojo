"""Flint machine-learning runtime (`flint.ml`): a pure-Mojo ONNX loader and CPU interpreter.

  - `tensor`: dense float32 / integer tensors, zero-copy views of mmapped weights
  - `protobuf`: the protobuf wire-format reader the loader uses
  - `ops`: operator kernels (SIMD + JobPool matrix products, 1-D convolution, reductions)
  - `fast`: parallel SIMD element-wise / gather / layer-norm kernels and fused attention
  - `gpu`: MAX GPU matrix products (split-K and 128x128 tiled SGEMM), sparse-conv gather-GEMM
    and fused attention
  - `onnx`: `OnnxModel.load(path)` (initializers inline or in external-data side files),
    `set_input`, `run`, `output`; plans attention and gather-GEMM fusions at load
  - `opt`: OPT decoder with a beam-aware KV cache
  - `motionbricks`: the MotionBricks motion-inbetweening runtime on top of `OnnxModel`
  - `unirig`, `unirig_skin`, `unirig_rig`: UniRig automatic rigging (skeleton generation, skin
    weights, skinned glTF output)
  - `safetensors`: memory-mapped `.safetensors` weights (zero-copy F32, F16 / BF16 converted)
  - `moonshine`, `moonshine_gpu`, `moonshine_audio`, `moonshine_tokenizer`: Moonshine v2
    streaming speech-to-text (encoder, KV-cached decoder, CPU / MAX GPU), its 16 kHz front end
    (WAV reading, resampling, CMVN frames, energy VAD), BPE decoding and voice-command matching
  - `llm`: small-LLM runtime for NPC dialogue and behaviour choice (Qwen3, SmolLM3 from
    safetensors; int8 / int4 weights; grammar-constrained BehaviorTree actions)
  - `ops_ext`: LSTM, ConvTranspose, Pad, STFT, linear Resize, ReduceMean, NonZero, Round and
    parallel / GPU 1-D convolution (op codes 60..79 in `onnx`)
  - `phonemize`: English G2P (misaki lexicons, morphology, numbers, letter-to-sound fallback)
  - `kokoro`: Kokoro-82M TTS, `KokoroTTS.load(dir).say(text, voice)` -> 24 kHz samples, timings
  - `gemx`: GEM-X monocular video to whole-body SOMA motion (keypoints, pose tokens, Motion
    transformer, BVH / `NativeAnimation` clips)
"""
