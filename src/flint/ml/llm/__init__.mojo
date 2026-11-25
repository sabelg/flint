"""In-engine small-LLM runtime for NPC dialogue and behaviour choice (`flint.ml.llm`).

  - `safetensors`: memory-mapped (sharded) safetensors checkpoints, BF16 / F16 / F32
  - `tokenizer`: byte-level BPE from a Hugging Face `tokenizer.json` (Qwen2/3, Llama 3)
  - `quant`: fp32 / int8 (per row) / int4 (groups of 32) weight matrices and CPU kernels
  - `kernels`: MAX GPU dequantising matvec, RMSNorm, QK-norm + RoPE, GQA attention, SwiGLU
  - `model`: `LlmModel` (Qwen3, SmolLM3) with a CPU or GPU KV cache and prefix reuse
  - `sample`: greedy / temperature / top-k / top-p sampling with a repetition penalty
  - `chat`: chat templates (ChatML for Qwen3, SmolLM3's metadata header), `generate`
  - `constrain`: byte-level grammar masks (JSON objects with enum and string fields)
  - `npc`: `NpcBrain`, an LLM that picks one of a BehaviorTree's actions as JSON
  - `embed`: `TextEmbedder` (Qwen3-Embedding: last-token pooling, L2 norm, MRL truncation,
    instruct queries) and `EmbeddingIndex` (cosine top-k over assets, scripts, quests, dialogue)
  - `speculative`: draft / verify / rollback speculative decoding (Qwen3-0.6B drafting for
    Qwen3-1.7B and larger), greedy-exact, standard acceptance rule when sampling
"""

from flint.ml.llm.safetensors import SafeTensors, SafeTensor
from flint.ml.llm.tokenizer import BpeTokenizer
from flint.ml.llm.quant import QMat, Q_F32, Q_INT8, Q_INT4, fmt_name
from flint.ml.llm.model import LlmModel, LlmConfig
from flint.ml.llm.sample import Sampler, Rng, argmax, argmax_allowed
from flint.ml.llm.sample import SparseDist
from flint.ml.llm.embed import TextEmbedder, EmbeddingIndex, EmbeddingHit, instruct_query, mrl_truncate, l2_normalize, cosine
from flint.ml.llm.speculative import speculative_tokens, generate_speculative, spec_accept, SpecGeneration, SpecStats
