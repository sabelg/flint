"""SigLIP 2 image-text embeddings in pure Mojo (`flint.ml.siglip`), CPU or MAX GPU.

  - `model`: `SiglipModel.load(dir)` (vision tower with attention pooling, text tower, sigmoid
    logit head), batched `embed_pixels` / `embed_texts`, `logits`; reuses the CLIP encoder layers
    and `Runtime` of `flint.ml.diffusion`
  - `image`: PIL-exact BILINEAR / BICUBIC resizing and SigLIP normalisation, RGBA over white, and
    software GLB thumbnails (`render_glb_thumbnail`)
  - `search`: `AssetIndex`, a text-to-image search over a folder of PNG and GLB files with an
    incremental embedding cache (`.flsg`), multilingual queries through the Gemma tokenizer

The tokenizer is `flint.ml.sentencepiece` (SentencePiece Unigram / BPE from `tokenizer.model`,
`spiece.model` or `tokenizer.json`). Weights are not part of Flint (Apache-2.0,
`tools/siglip/fetch.py`).
"""
