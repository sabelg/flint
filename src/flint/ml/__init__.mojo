"""Flint machine-learning runtime (`flint.ml`): a pure-Mojo ONNX loader and CPU interpreter.

  - `tensor`: dense float32 / integer tensors, zero-copy views of mmapped weights
  - `protobuf`: the protobuf wire-format reader the loader uses
  - `ops`: operator kernels (SIMD + JobPool matrix products, 1-D convolution, reductions)
  - `onnx`: `OnnxModel.load(path)`, `set_input`, `run`, `output`
  - `kimodo`, `kimodo_text`, `kimodo_bvh`: text-conditioned SOMA motion generation and export
  - `motionbricks`: the MotionBricks motion-inbetweening runtime on top of `OnnxModel`
  - `gemx`: GEM-X monocular video to whole-body SOMA motion (keypoints, pose tokens, Motion transformer, BVH / `NativeAnimation` clips)
  - `rgbmocap`: commercial-safe, backend-agnostic RGBMocap pipeline (multi-person IoU tracking, GEM-X bridge, demo SOMA-77 take generator, BVH & `NativeAnimation` export)
"""

from flint.ml.rgbmocap import (
    MOCAP_BACKEND_GEMX,
    MOCAP_BACKEND_RTMPOSE_MBERT,
    MocapBackendInfo,
    MocapBBox,
    MocapTrack,
    MocapCameraPose,
    MocapFrame,
    MocapTake,
    bbox_iou,
    probe_mocap_backends,
    track_person_boxes,
    run_gemx_image_sequence,
    mocap_take_to_mb_clip,
    export_mocap_take_bvh,
    mocap_take_to_native_animation,
    generate_demo_mocap_take,
)
