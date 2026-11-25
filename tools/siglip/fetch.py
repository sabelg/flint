#!/usr/bin/env python3
"""Download SigLIP 2 checkpoints for flint.ml.siglip into DIR (never into the repository).

    python tools/siglip/fetch.py DIR [base] [so400m] [base-256] [so400m-384] ...

Each name becomes DIR/<repo name>/ with config.json, model.safetensors (float32, ~1.5 GB for
base, ~4.5 GB for so400m), preprocessor_config.json and the Gemma tokenizer files
(tokenizer.model, tokenizer.json, tokenizer_config.json, special_tokens_map.json). Only the
Python standard library is used. Files that already have the expected size are skipped.

Licence: the google/siglip2-* model cards state Apache-2.0 (see licenses/siglip2-LICENSE.txt).
Flint reads any fixed-resolution SigLIP / SigLIP 2 checkpoint in the `SiglipModel` layout; the
NaFlex variants (`google/siglip2-*-patch16-naflex`, `Siglip2Model`) are not supported.
"""
import os
import sys
import urllib.request

HF = "https://huggingface.co"
MODELS = {
    "base": "google/siglip2-base-patch16-224",
    "base-256": "google/siglip2-base-patch16-256",
    "large": "google/siglip2-large-patch16-256",
    "so400m": "google/siglip2-so400m-patch14-224",
    "so400m-384": "google/siglip2-so400m-patch14-384",
}
FILES = ["config.json", "preprocessor_config.json", "tokenizer.model", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "README.md", "model.safetensors"]


def fetch(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "flint-fetch/1.0"})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if total and os.path.exists(dst) and os.path.getsize(dst) == total:
            print("  have", os.path.basename(dst))
            return
        tmp = dst + ".part"
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total > 1 << 26:
                    print("\r  %s %5.1f%%" % (os.path.basename(dst), 100.0 * done / total), end="", flush=True)
        os.replace(tmp, dst)
        print("\r  got %s (%d bytes)" % (os.path.basename(dst), done))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    names = sys.argv[2:] or ["base"]
    for n in names:
        repo = MODELS.get(n, n)
        dst = os.path.join(root, repo.split("/")[-1])
        os.makedirs(dst, exist_ok=True)
        print(repo, "->", dst)
        for f in FILES:
            fetch("%s/%s/resolve/main/%s" % (HF, repo, f), os.path.join(dst, f))


if __name__ == "__main__":
    main()
