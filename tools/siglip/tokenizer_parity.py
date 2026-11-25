#!/usr/bin/env python3
"""Token-id parity of flint.ml.sentencepiece against sentencepiece / tokenizers.

    python tools/siglip/tokenizer_parity.py FLINT_SIGLIP_BIN LINES_FILE TOKENIZER_FILE [...]

FLINT_SIGLIP_BIN is a build of examples/siglip/main.mojo; LINES_FILE holds one text per line
(`\\t` / `\\n` escapes allowed). Every TOKENIZER_FILE (`*.model` for SentencePiece protobufs, read
by `sentencepiece`; `tokenizer.json`, read by `tokenizers`) is encoded without special tokens by
both, and the differing lines are printed. Exit status 1 if any line differs.

Tested files: SigLIP 2's Gemma `tokenizer.model` / `tokenizer.json` (BPE, byte fallback, user
defined pieces) and T5 / flan-t5 `spiece.model` / `tokenizer.json` (Unigram, nmt_nfkc charsmap).
"""
import subprocess
import sys


def reference(path, lines):
    if path.endswith(".json"):
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(path)
        t.no_padding()
        t.no_truncation()
        return [t.encode(l, add_special_tokens=False).ids for l in lines]
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=path)
    return [sp.encode(l) for l in lines]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    binary, lines_file = sys.argv[1], sys.argv[2]
    raw = open(lines_file, encoding="utf-8").read().split("\n")
    if raw and raw[-1] == "":
        raw = raw[:-1]
    lines = [l.replace("\\t", "\t").replace("\\n", "\n") for l in raw]
    bad_total = 0
    for path in sys.argv[3:]:
        want = reference(path, lines)
        out = subprocess.run([binary, path, "tokenize", lines_file], capture_output=True, text=True, check=True)
        got = [[int(x) for x in l.split()] for l in out.stdout.split("\n")[:len(lines)]]
        bad = [i for i in range(len(lines)) if got[i] != want[i]]
        print("%s: %d/%d lines identical" % (path, len(lines) - len(bad), len(lines)))
        for i in bad[:10]:
            print("  %r\n    flint %s\n    ref   %s" % (lines[i], got[i], want[i]))
        bad_total += len(bad)
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
