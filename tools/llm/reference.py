"""Hugging Face reference for `flint.ml.llm` parity checks (needs torch + transformers).

usage: python tools/llm/reference.py MODEL_DIR OUT_DIR [--steps N] [--date "25 November 2026"]

Writes to OUT_DIR:
  prompt.txt      the chat-templated prompt text (non-thinking mode)
  ids.txt         prompt token ids (space separated)
  logits.f32      float32 logits for every prompt position, [len(ids), vocab] row major
  greedy.txt      N greedy continuation ids (HF generate, do_sample=False)
The model runs in float32 on the CPU, so the bf16 checkpoint weights are used exactly.
"""
import sys, os, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = ("You are Brakka, a gruff dwarven blacksmith in the village of Emberfall. "
          "Answer in one or two short sentences.")
USER = "Hello there! Can you repair my sword before the goblins attack tonight?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    kw = {}
    if a.date:
        import jinja2  # noqa: F401  (strftime_now is resolved at render time)
        import transformers.utils.chat_template_utils as ctu
        orig = ctu._compile_jinja_template

        def patched(t):
            c = orig(t)
            c.globals["strftime_now"] = lambda fmt: a.date
            return c
        ctu._compile_jinja_template = patched
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)
    ids = tok(text, add_special_tokens=False).input_ids
    open(os.path.join(a.out, "prompt.txt"), "w").write(text)
    open(os.path.join(a.out, "ids.txt"), "w").write(" ".join(map(str, ids)))
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(os.cpu_count())
    with torch.no_grad():
        x = torch.tensor([ids])
        logits = model(x).logits[0].float().numpy()
        logits.tofile(os.path.join(a.out, "logits.f32"))
        g = model.generate(x, max_new_tokens=a.steps, do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = g[0, len(ids):].tolist()
    open(os.path.join(a.out, "greedy.txt"), "w").write(" ".join(map(str, gen)))
    print("prompt", len(ids), "tokens; greedy:", repr(tok.decode(gen)))


if __name__ == "__main__":
    main()
