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
  - `gemx`: GEM-X monocular video to whole-body SOMA motion (keypoints, pose tokens, Motion
    transformer, BVH / `NativeAnimation` clips)
"""
