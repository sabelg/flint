"""sentence-transformers / Transformers reference for `flint.ml.llm.embed` parity checks.

usage: python tools/llm/embed_reference.py MODEL_DIR OUT_DIR [--transformers-only]

MODEL_DIR is a Qwen3-Embedding checkpoint (e.g. Qwen3-Embedding-0.6B). Writes to OUT_DIR:
  inputs.txt      one input per line, already in its final form (queries carry the
                  `Instruct: ...\\nQuery:` prefix); newlines inside a text are written as "\\n"
  ids.txt         the token ids of each input, one line per input (the tokenizer's
                  post-processor appends <|endoftext|>)
  st.f32          sentence-transformers embeddings, float32 [n, 1024], L2-normalised
  hf.f32          plain Transformers embeddings (last-token pooling + L2 norm), same shape
The model runs in float32 on the CPU, so the bf16 checkpoint weights are used exactly.
Both pipelines are run so the reference itself is cross-checked.
"""
import sys, os, argparse
import numpy as np
import torch

TASK = "Given a player's request, retrieve the game asset, script, quest or dialogue line it refers to"
QUERIES = [
    "where can I buy a better sword",
    "the quest about the missing miller's daughter",
    "a spooky sound for the crypt",
    "what does the blacksmith say when you have no money",
    "script that opens the drawbridge",
]
DOCS = [
    "asset: sfx/crypt_wind_howl.qoa - low howling wind with distant chains, for tombs and crypts",
    "asset: models/weapons/longsword_steel.glb - a steel longsword sold by Brakka the blacksmith",
    "asset: textures/grass_meadow_albedo.png - green meadow grass, tiling",
    "script: scripts/castle/drawbridge.mojo - lowers the castle drawbridge when the lever is pulled",
    "script: scripts/ai/goblin_patrol.mojo - goblins walk a patrol route and alert their friends",
    "quest: The Miller's Daughter - Elsa has not come home from the old mill; find her before nightfall",
    "quest: Ember Tithe - collect five ember stones for the village shrine",
    "dialogue: Brakka (no gold): \"No coin, no steel. Come back when your purse is heavier.\"",
    "dialogue: Elsa (rescued): \"I thought the wolves would find me first. Thank you!\"",
    "dialogue: Guard (night): \"Gates are shut after dark. Move along.\"",
]


def last_token_pool(h, mask):
    left = bool(mask[:, -1].sum() == mask.shape[0])
    if left:
        return h[:, -1]
    n = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0]), n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--transformers-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    inputs = ["Instruct: " + TASK + "\nQuery:" + q for q in QUERIES] + DOCS
    with open(os.path.join(a.out, "inputs.txt"), "w") as f:
        for t in inputs:
            f.write(t.replace("\n", "\\n") + "\n")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    with open(os.path.join(a.out, "ids.txt"), "w") as f:
        for t in inputs:
            f.write(" ".join(map(str, tok(t).input_ids)) + "\n")
    rows = []
    with torch.no_grad():
        for t in inputs:  # one at a time: no padding involved
            b = tok([t], return_tensors="pt")
            h = model(**b).last_hidden_state
            e = torch.nn.functional.normalize(last_token_pool(h, b["attention_mask"]), p=2, dim=1)
            rows.append(e[0].float().numpy())
    hf = np.stack(rows).astype(np.float32)
    hf.tofile(os.path.join(a.out, "hf.f32"))
    if not a.transformers_only:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(a.model, device="cpu", model_kwargs={"torch_dtype": torch.float32})
        e = st.encode(inputs, batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        e.tofile(os.path.join(a.out, "st.f32"))
        print("max |st - hf| =", float(np.abs(e - hf).max()), "min cos(st, hf) =", float((e * hf).sum(1).min()))
    else:
        hf.tofile(os.path.join(a.out, "st.f32"))
    nq = len(QUERIES)
    s = hf[:nq] @ hf[nq:].T
    for i, q in enumerate(QUERIES):
        j = int(s[i].argmax())
        print(f"{q!r} -> {DOCS[j][:60]!r} ({s[i, j]:.4f})")
    print("wrote", len(inputs), "embeddings of dim", hf.shape[1], "to", a.out)


if __name__ == "__main__":
    main()
