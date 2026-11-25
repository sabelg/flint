"""MusicCoCa text reference (TFLite) for `flint.ml.musiccoca` parity.

    MAGENTA_HOME=<magenta-realtime clone> python -I tools/magenta_rt/reference_musiccoca.py <mrt2 dir> <out dir>

Writes `<out>/musiccoca_ref.txt`: per prompt, one line `prompt|ids|tokens` (ids and RVQ tokens
space separated) followed by the 768 embedding values; `MAPPED seed|prompt|tokens` lines
(+ embedding) for `use_mapper=True`; and a 50/50 blend of the first two
prompts' embeddings and its tokens.
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.environ['MAGENTA_HOME'])
import types  # noqa: E402
sys.modules.setdefault('librosa', types.ModuleType('librosa'))  # only the audio path needs it
from magenta_rt import musiccoca  # noqa: E402

PROMPTS = ['upbeat synthwave with punchy drums', 'Dark ambient drone, 60 BPM!',
           'solo piano ballad', '  lo-fi   hip hop beats to study to ']


def main():
    m = musiccoca.MusicCoCa(pathlib.Path(sys.argv[1]) / 'resources' / 'musiccoca')
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    embs = []
    for p in PROMPTS:
        ids = [1] + m._vocab.EncodeAsIds(p.lower())[:127]
        e = m.embed(p)
        t = m.tokenize(e)
        embs.append(e)
        lines.append(f"{p}|{' '.join(map(str, ids))}|{' '.join(map(str, np.ravel(t)))}")
        lines.append(' '.join(f'{v:.9g}' for v in e))
    for p in PROMPTS[:2]:
        for seed in (0, 1):
            e = m.embed(p, use_mapper=True, seed=seed)
            lines.append(f"MAPPED {seed}|{p}|{' '.join(map(str, np.ravel(m.tokenize(e))))}")
            lines.append(' '.join(f'{v:.9g}' for v in e))
    b = np.mean([embs[0], embs[1]], axis=0)
    lines.append(f"BLEND|-|{' '.join(map(str, np.ravel(m.tokenize(b))))}")
    (out / 'musiccoca_ref.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(l for l in lines if '|' in l))


if __name__ == '__main__':
    main()
