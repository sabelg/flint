# Third-party notices

- `src/flint/ml/motionbricks.mojo` is a Mojo port of parts of NVIDIA MotionBricks
  (https://github.com/NVlabs/motionbricks): the SOMA pose contract, input encoding, output
  decoding and endpoint correction. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
  source code licensed under Apache-2.0 — see `motionbricks-LICENSE.txt`.
- Flint does not include the MotionBricks models or motion clips. The ONNX models are under the
  NVIDIA Open Model Agreement; the sample motions under the NVIDIA Sample Data Evaluation
  License, which allows internal evaluation only, with no redistribution.

## Kimodo

The native SOMA feature decoding and diffusion sampler follow NVIDIA Kimodo, copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. See `kimodo-LICENSE.txt` for its Apache-2.0 source license. The separate model installer downloads the [SOMA model conversion](https://huggingface.co/KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX) and the [LLM2Vec conversion built with Meta Llama 3](https://huggingface.co/KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX). The model cards link their respective terms and source revisions.

## Body capture

The offline RTMPose converter rewrites a user-supplied ONNX graph. The MotionBERT converter imports DSTformer from a user-supplied upstream checkout and reads the H36M fine-tuned checkpoint. Prepared weights live in the user's cache. Review the upstream terms for each checkpoint and dataset before distributing a model. [RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) · [MotionBERT project and license](https://github.com/Walter0807/MotionBERT)

## HY-Motion Lite

The preparation tools import the upstream Hunyuan model definitions and read locally obtained Lite, Qwen3-8B and CLIP-L weights. The generated inference graphs, packed text weights and prompt cache live outside the repository. Review each project's terms for the intended use. [Tencent HY-Motion source and license](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) · [HY-Motion weights](https://huggingface.co/tencent/HY-Motion-1.0) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [CLIP-L](https://huggingface.co/openai/clip-vit-large-patch14)
