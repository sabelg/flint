"""Pure-Mojo latent diffusion (`flint.ml.diffusion`), Stable-Diffusion family, CPU or MAX GPU.

  - `nn`: `Runtime` (host lists or GPU buffers), channels-last activations, implicit-GEMM
    convolutions (zero or wrap-around padding, stride, fused nearest 2x upsampling), GroupNorm /
    LayerNorm, attention; `kernels`: the MAX GPU kernels behind it
  - `unet`: diffusers `UNet2DConditionModel` (cross-attention transformers, time / guidance
    embeddings, any block layout read from the weights)
  - `vae`: KL encoder / decoder and LDM-named (VQ) decoders, seam-free padded decoding
  - `clip`: CLIP text (causal, pooled + projected) and vision towers; `tokenizer`: CLIP BPE
  - `rng`: `torch.Generator`-exact MT19937 + `randn`; `schedulers`: DDIM, Euler, LCM
  - `pipeline`: `Denoiser` (classifier-free guidance, concat conditions, timestep conditions,
    tiling from step k) and the seeded sampling loop
  - `imageproc`: antialiased bicubic resize, CLIP preprocessing, pixel conversion
  - `materials`: StableMaterials and MatFuse -> `PbrMaps` -> PNGs, ORM, GLB, material JSON
"""
