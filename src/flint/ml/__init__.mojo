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
  - `ops_ext`: LSTM, ConvTranspose, Pad, STFT, linear Resize, ReduceMean / Min / Max, GatherND, NonZero, Round and
    parallel / GPU 1-D convolution (op codes 60..79 in `onnx`)
  - `phonemize`: English G2P (misaki lexicons, morphology, numbers, letter-to-sound fallback)
  - `kokoro`: Kokoro-82M TTS, `KokoroTTS.load(dir).say(text, voice)` -> 24 kHz samples, timings
  - `gemx`: GEM-X monocular video to whole-body SOMA motion (keypoints, pose tokens, Motion
    transformer, BVH / `NativeAnimation` clips)
  - `diffusion`: latent diffusion (UNet, KL / VQ VAE, CLIP, DDIM / Euler / LCM, CFG, tileable
    wrap-mode convolutions) and `diffusion.materials`: StableMaterials / MatFuse text or photo ->
    PBR material sets
  - `npz`: memory-mapped NumPy `.npz` archives (stored members, zip64)
  - `ops_more`: InstanceNorm, ReduceMax, GRU, Softplus, Ceil, unit-kernel 2-D and batched GPU
    convolution, per-channel broadcasts (op codes 80..89 in `onnx`)
  - `audio2face`: NVIDIA Audio2Face-3D v2.3 speech -> ARKit-52 blend shapes, tongue, jaw, eyes
    (SDK post-processing and BVLS solve, streaming, morph-target GLB output)
  - `magenta_rt`, `musiccoca`, `audio_codec`, `audio_codec_gpu`, `jax_random`: Magenta RealTime 2
    streaming music (Depthformer LM, MusicCoCa text styles, SpectroStream 48 kHz decoder on CPU /
    MAX GPU, bit-exact `jax.random` sampling) and `MagentaRtStream` for the `AudioServer`
  - `metatron`: NVIDIA Metatron-v1 G1 humanoid controller (policy, ProtoMotions host port,
    `G1Sim` on `flint.physics.mjcf`)
  - `ops_conv2d`, `ops_image`: general 2-D Conv (depthwise / grouped / strided / dilated, CPU +
    GPU dense products) and parallel LeakyRelu / Clip / Resize / Pad (op codes 90..94 in `onnx`)
  - `marigold`: Marigold-Normals v1.1 image -> normal map on `diffusion` (shared `MarigoldCore`
    for later depth / intrinsics heads; 1..50 DDIM steps, ensembling, uncertainty, tileable mode)
  - `migan`: MI-GAN fast inpainting on `OnnxModel` (windowed full-resolution repaint, sprite
    clean-up, `make_tileable` seamless textures)
"""
