# Small LLMs for NPCs (`flint.ml.llm`)

A pure-Mojo runtime for 0.6B–3B decoder LLMs, built for NPC dialogue and behaviour choice:
weights load straight from Hugging Face safetensors, decoding runs on the MAX GPU (or the CPU
job pool), and a grammar makes the model answer with one action of a fixed BehaviorTree action
set.

| Model | Licence | Architecture |
|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0 | 28 layers, d 1024, 16 q / 8 kv heads (hd 128), QK-norm, tied embeddings |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 | 28 layers, d 2048, 16 / 8 heads, QK-norm, tied |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Apache-2.0 | 36 layers, d 2048, 16 / 4 heads, NoPE every 4th layer, tied |

Weights are not part of the repository (see `licenses/`); download a checkpoint directory
(`config.json`, `*.safetensors`, `tokenizer.json`) from Hugging Face.

## Pieces

| Module | What it does |
|---|---|
| `safetensors` | mmaps (sharded) checkpoints, BF16 / F16 / F32 widening |
| `tokenizer` | byte-level BPE from `tokenizer.json`: added tokens, the Qwen / Llama 3 split regex, merges by rank; byte-identical to HF `tokenizers` on the test corpus for both vocabularies |
| `quant` | `QMat`: fp32, int8 (symmetric, groups of 128), int4 (symmetric, groups of 32, MSE-searched scale); q/k/v and gate/up stacked; CPU SIMD products |
| `kernels` | GPU: dequantising matvec (one warp per row, lane-interleaved weights), RMSNorm, QK-norm + RoPE + KV write, GQA attention with online softmax, SwiGLU |
| `model` | `LlmModel`: config, weight conversion, CPU and GPU forward, KV cache with longest-prefix reuse, chunked prefill |
| `sample` | greedy, temperature, top-k, top-p, repetition penalty, masks; seeded SplitMix64 |
| `chat` | Qwen3 (ChatML) and SmolLM3 templates in non-thinking mode, token-identical to `apply_chat_template`; `generate`, `generate_constrained` |
| `constrain` | `JsonGrammar` (literals, enums, bounded strings) and `GrammarMatcher.mask` over the vocabulary (0.7–1.5 ms for 151k tokens) |
| `npc` | `NpcBrain`: persona + actions + targets → `{"action", "target", "say"}`; `build_choice_tree` / `write_blackboard` hook into `flint.ai` |

Player text is encoded with `allow_special=False`, so typing `<|im_end|>` cannot end the turn.

```mojo
var model = LlmModel.load(dir, Q_INT8)              # GPU when present
var tok = BpeTokenizer.load(dir + "/tokenizer.json")
var tmpl = ChatTemplate.for_model(model.cfg, tok)
var brain = NpcBrain("You are Brakka, a dwarven blacksmith.", actions^, targets^)
var d = brain.decide(model, tok, tmpl, "A goblin charges at you.")
brain.write_blackboard(d, bb, k_choice)              # the tree's selector runs actions[d.index]
```

## Running

```bash
mojo build -I src -O3 --fp-mode contract=off --Werror examples/npc_llm/main.mojo -o build/npc_llm
build/npc_llm chat ~/models/Qwen3-1.7B            # streaming chat, int8 on the GPU
build/npc_llm npc  ~/models/Qwen3-1.7B            # scripted scenes -> LLM -> BehaviorTree
NPC_LLM_DIR=~/models/Qwen3-0.6B build/v5/test_flint_v5_llm   # suite incl. a model run
```

## Parity with Hugging Face Transformers

`tools/llm/reference.py MODEL OUT` runs Transformers (float32, CPU) on a fixed chat prompt and
writes the prompt ids, the logits of every prompt position and 32 greedy tokens;
`examples/npc_llm/parity.mojo MODEL OUT [--cpu] [--fmt ...]` compares. The token ids of the
templated prompt match exactly (60 tokens for Qwen3, 94 for SmolLM3).

| Model | Weights | max \|Δlogit\| (all positions) | argmax agrees | mean KL (nats) | 32 greedy tokens |
|---|---|---|---|---|---|
| Qwen3-0.6B | fp32 | 2.5e-4 | 60 / 60 | 7e-11 | identical |
| Qwen3-0.6B | int8 | 3.1 | 58 / 60 | 0.0074 | identical |
| Qwen3-0.6B | int4 | 14.8 | 38 / 60 | 0.53 | diverge at token 1 |
| Qwen3-1.7B | fp32 | 2.2e-4 | 60 / 60 | 9e-11 | identical (31, EOS) |
| Qwen3-1.7B | int8 | 2.4 | 58 / 60 | 0.0069 | identical |
| Qwen3-1.7B | int4 | 14.4 | 44 / 60 | 0.30 | diverge at token 1 |
| SmolLM3-3B | fp32 | 3.2e-4 | 94 / 94 | 6e-11 | identical |
| SmolLM3-3B | int8 | 2.4 | 93 / 94 | 0.0013 | identical |
| SmolLM3-3B | int4 | 20.5 | 78 / 94 | 0.47 | diverge at token 3 |

The CPU and GPU paths give the same figures to 4–5 significant digits in every row. fp32 is
exact up to summation order. The
int8 and int4 rows reproduce what PyTorch gives for the same round-to-nearest quantisation
(fake-quantised weights: int8/128 max Δ 3.1, int4/32 ≈ 14.6 on Qwen3-0.6B). int8 is
effectively lossless. int4 replies stay fluent and in character, but differ from fp32 from the
first token; calibrated int4 (GPTQ / AWQ) would be the next step if that matters. int8 with one
scale per row (instead of per 128) diverges after 23 tokens on Qwen3-0.6B, which is why groups
are used.

## Throughput

NVIDIA GB10 (DGX Spark, 20-core Grace CPU, unified LPDDR5X), 60–94 token prompt, decode of the
following 64 tokens. GPU decode is memory-bound: at int4 the 0.6B model streams about 360 MB per
token (weights and scales) in 2.9 ms of kernel time.

| Model | Weights | MiB | GPU prefill tok/s | GPU decode tok/s | CPU decode tok/s |
|---|---|---|---|---|---|
| Qwen3-0.6B | fp32 | 2273 | 620 | 84 | 28–41 |
| Qwen3-0.6B | int8 | 586 | 570–820 | 244 | 67–95 |
| Qwen3-0.6B | int4 | 355 | 750–890 | 335 | 20–121 |
| Qwen3-1.7B | fp32 | 6563 | 190 | 30 | 11 |
| Qwen3-1.7B | int8 | 1692 | 450 | 100 | 31 |
| Qwen3-1.7B | int4 | 1025 | 460 | 141 | 37 |
| SmolLM3-3B | fp32 | 11730 | 127 | 17.5 | 7.5 |
| SmolLM3-3B | int8 | 3024 | 255 | 60 | 15 |
| SmolLM3-3B | int4 | 1832 | 320 | 84 | 18 |

The CPU column was measured while sibling jobs kept the machine at a load average of ~30, so it
spans the spread seen across runs. The CPU job pool synchronises every product, and under
contention it waits on the slowest worker.

An NPC decision (`examples/npc_llm npc`, int8, GPU) re-evaluates only the ~35-token observation
(the 181-token system prompt stays cached) and generates ~30–45 constrained tokens: 190 ms
(Qwen3-0.6B), 490 ms (1.7B), 630 ms (SmolLM3-3B int4) on average. Qwen3-1.7B int8 gave the
most sensible action choices in the demo scenes; 0.6B tends to repeat one action.
