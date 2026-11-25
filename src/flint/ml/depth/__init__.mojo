"""Monocular depth and geometry (`flint.ml.depth`), pure Mojo on the CPU or the MAX GPU.

Models (Hugging Face folders, weights read in place; Apache-2.0 / MIT checkpoints only):

  depth-anything/Depth-Anything-V2-Small-hf      relative disparity
  depth-anything/DA3-SMALL, DA3-BASE              any-view depth + confidence + camera per view
                                                  (one or several views of a scene)
  depth-anything/DA3MONO-LARGE, DA3METRIC-LARGE   relative / metric depth with a sky mask
  Ruicheng/moge-2-vit{s,b,l}-normal               metric points, depth, normals, mask, FOV

Modules:

  `vit`      the shared DINOv2 trunk (+ Depth Anything 3 camera token, alternating attention,
             QK-norm, 2-D RoPE, reference view selection)
  `dpt`      the shared DPT reassemble-and-fuse head (DA2, DA3 DPT and DualDPT main branch)
  `moge`     MoGe-2 ConvStack neck / heads and focal-shift geometry recovery
  `layers`, `kernels`   ops on `flint.ml.diffusion.nn.Runtime` with CPU loops and GPU kernels
  `weights`  safetensors or PyTorch `.pt` (via `flint.ml.torch_pt`) under one name lookup
  `image`    image loading and the reference resizers (OpenCV area / cubic, PyTorch AA bilinear)
  `model`    `DepthModel.load(dir)`, `predict(img)`, `predict_views(imgs)` -> `DepthResult`
  `export`   depth / normal PNG, OpenEXR, point clouds and mesh blockout GLBs
  `parity`   comparisons with `tools/depth/reference.py` dumps

    var m = DepthModel.load("/models/DA3-SMALL")
    var r = m.predict(load_rgb("room.png"))
    write_depth_png("room_depth.png", r)
"""
