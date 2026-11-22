"""Flint machine-learning runtime (`flint.ml`): a pure-Mojo ONNX loader and CPU interpreter.

  - `tensor`: dense float32 / integer tensors, zero-copy views of mmapped weights
  - `protobuf`: the protobuf wire-format reader the loader uses
  - `ops`: operator kernels (SIMD + JobPool matrix products, 1-D convolution, reductions)
  - `onnx`: `OnnxModel.load(path)`, `set_input`, `run`, `output`
  - `kimodo`, `kimodo_text`, `kimodo_bvh`: text-conditioned SOMA motion generation and export
  - `motionbricks`: the MotionBricks motion-inbetweening runtime on top of `OnnxModel`
"""
