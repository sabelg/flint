"""Convert the MusicCoCa text encoder + RVQ style quantizer (TFLite) to one safetensors file.

    python -I tools/magenta_rt/convert_musiccoca.py <mrt2 dir> [<out.safetensors>]

Reads `<dir>/resources/musiccoca/{text_encoder,pretrained_vector_quantizer,mapper}.tflite` and
`spm.model` from the `google/magenta-realtime-2` Hugging Face repo (CC-BY 4.0, Google LLC) and
writes `<dir>/resources/musiccoca/musiccoca_text.safetensors` (all fp32; the SentencePiece
unigram vocab is stored as `spm.bytes/offsets/scores/types` float rows), read by
`flint.ml.magenta_rt.MusicCoCaText`. Needs numpy, ai_edge_litert, sentencepiece, safetensors.

The text encoder is a 12-layer pre-LN transformer (d=768, 12x64 heads, tanh(x/50)*50 logit cap,
key padding mask, GELU FFN 3072 masked by (1-padding)), then LayerNorm, multi-head attention
pooling (12x256 heads with a learned query), an output projection and a final LayerNorm.
Weights are taken in graph order and checked against the expected shapes. The optional
text->audio-space mapper (`mapper.tflite`, used by Magenta RT's own text prompting) is stored
under `mapper.*`.
"""

import json
import pathlib
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from safetensors.numpy import save_file


def _interp(path):
    it = Interpreter(model_path=str(path),
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                     experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    return it


def _const(it, idx):
    return np.array(it.get_tensor(idx), dtype=np.float32)


def _consts_of(it, op, const_ids):
    return [i for i in op['inputs'] if i in const_ids]


def text_encoder(path):
    it = _interp(path)
    ops = it._get_ops_details()
    produced = {o for op in ops for o in op['outputs']}
    inputs = {d['index'] for d in it.get_input_details()}
    const_ids = {t['index'] for t in it.get_tensor_details()} - produced - inputs
    out = {}
    fcs, lns, pool_q = [], [], None
    emb = pos = None
    for op in ops:
        name = op['op_name']
        cs = _consts_of(it, op, const_ids)
        if name == 'EMBEDDING_LOOKUP':
            emb = _const(it, cs[0])
        elif name == 'ADD' and emb is not None and pos is None:
            pos = _const(it, cs[0]).reshape(128, 768)
        elif name == 'FULLY_CONNECTED':
            fcs.append((_const(it, op['inputs'][1]), _const(it, op['inputs'][2])))
        elif name in ('MUL', 'ADD') and cs:
            v = _const(it, cs[0])
            if v.shape == (1, 1, 768):
                lns.append(v.reshape(768))
        elif name == 'BATCH_MATMUL' and cs:
            pool_q = _const(it, cs[0]).reshape(12, 256)
    assert emb.shape == (16000, 768) and pos is not None and pool_q is not None
    assert len(fcs) == 12 * 6 + 3, len(fcs)
    assert len(lns) == 2 * (12 * 2 + 2), len(lns)
    out['tok_emb'] = emb
    out['pos_emb'] = pos
    for layer in range(12):
        f = fcs[layer * 6:(layer + 1) * 6]   # graph order: K, V, Q (scaled), O, FFN1, FFN2
        g = lns[layer * 4:(layer + 1) * 4]
        p = f'layers.{layer}.'
        for nm, (w, b) in zip(('wk', 'wv', 'wq', 'wo', 'w1', 'w2'), f):
            out[p + nm] = w
            out[p + 'b' + nm[1:]] = b
        out[p + 'ln1_g'], out[p + 'ln1_b'], out[p + 'ln2_g'], out[p + 'ln2_b'] = g
    out['ln_f_g'], out['ln_f_b'] = lns[48], lns[49]
    (pk, bk), (pv, bv), (po, bo) = fcs[72:75]
    out['pool.wk'], out['pool.bk'], out['pool.wv'], out['pool.bv'] = pk, bk, pv, bv
    out['pool.wo'], out['pool.bo'], out['pool.query'] = po, bo, pool_q
    out['ln_out_g'], out['ln_out_b'] = lns[50], lns[51]
    shapes = {'wq': (768, 768), 'wk': (768, 768), 'wv': (768, 768), 'wo': (768, 768),
              'w1': (3072, 768), 'w2': (768, 3072)}
    for layer in range(12):
        for nm, s in shapes.items():
            assert out[f'layers.{layer}.{nm}'].shape == s, (layer, nm)
    assert pk.shape == (3072, 768) and po.shape == (768, 3072)
    return out


def quantizer(path):
    it = _interp(path)
    ops = it._get_ops_details()
    books = [_const(it, op['inputs'][0]) for op in ops if op['op_name'] == 'GATHER']
    # The last level has no GATHER (its codeword is never subtracted): take it from the
    # FULLY_CONNECTED distance weights, which are -2 * codebook.
    fcw = [_const(it, op['inputs'][1]) for op in ops if op['op_name'] == 'FULLY_CONNECTED']
    assert len(fcw) == 12 and len(books) == 11
    for lvl in range(11):
        np.testing.assert_allclose(fcw[lvl], -2.0 * books[lvl], rtol=1e-6, atol=1e-6)
    books.append(fcw[11] / -2.0)
    return np.stack(books)


def mapper(path):
    """Text -> audio-space mapper: 8 AdaLN transformer layers (12 x 256 tokens) between two FCs.

    The layer weights live in the WHILE body subgraph (scanned by layer index); tensor indices
    are fixed by the exported graph and checked by shape.
    """
    import tflite
    m = tflite.Model.GetRootAsModel(pathlib.Path(path).read_bytes(), 0)

    def const(g, i, shape):
        t = g.Tensors(i)
        a = m.Buffers(t.Buffer()).DataAsNumpy().tobytes()
        a = np.frombuffer(a, dtype=np.float32).reshape(tuple(t.ShapeAsNumpy()) if t.ShapeLength() else ())
        assert a.shape == shape, (i, a.shape, shape)
        return np.array(a, dtype=np.float32)

    main, body = m.Subgraphs(0), m.Subgraphs(2)
    out = {
        'mapper.in_w': const(main, 6, (3072, 768)), 'mapper.in_b': const(main, 4, (3072,)),
        'mapper.out_w': const(main, 5, (768, 3072)), 'mapper.out_b': const(main, 3, (768,)),
        'mapper.ctx': const(main, 23, (1, 1, 128)).reshape(128),
        'mapper.rope_freq': const(body, 29, (1, 1, 1, 32)).reshape(32),
        'mapper.mod1_b': const(body, 22, (8, 2, 256)),
        'mapper.mod1_w': const(body, 21, (8, 1024, 2, 256)).reshape(8, 1024, 512),
        'mapper.norm1': const(body, 20, (8, 256)),
        'mapper.qkv': const(body, 19, (8, 256, 3, 8, 64)).reshape(8, 256, 1536),
        'mapper.sink_k': const(body, 18, (8, 1, 8, 64)).reshape(8, 512),
        'mapper.sink_v': const(body, 17, (8, 1, 8, 64)).reshape(8, 512),
        'mapper.wo': const(body, 16, (8, 256, 8, 64)).reshape(8, 256, 512),
        'mapper.mod2_b': const(body, 15, (8, 2, 256)),
        'mapper.mod2_w': const(body, 14, (8, 1024, 2, 256)).reshape(8, 1024, 512),
        'mapper.b1': const(body, 13, (8, 1024)),
        'mapper.w1': const(body, 12, (8, 256, 1024)),
        'mapper.b2': const(body, 11, (8, 256)),
        'mapper.w2': const(body, 10, (8, 1024, 256)),
        'mapper.norm2': const(body, 9, (8, 256)),
    }
    return out


def spm_vocab(path):
    from sentencepiece import sentencepiece_model_pb2 as pb
    m = pb.ModelProto()
    m.ParseFromString(pathlib.Path(path).read_bytes())
    assert m.trainer_spec.model_type == 1  # unigram
    blob = bytearray()
    offs = [0]
    for p in m.pieces:
        blob += p.piece.encode('utf-8')
        offs.append(len(blob))
    return {
        'spm.bytes': np.frombuffer(bytes(blob), dtype=np.uint8).astype(np.float32),
        'spm.offsets': np.array(offs, dtype=np.float32),
        'spm.scores': np.array([p.score for p in m.pieces], dtype=np.float32),
        'spm.types': np.array([p.type for p in m.pieces], dtype=np.float32),
    }


def main():
    root = pathlib.Path(sys.argv[1]) / 'resources' / 'musiccoca'
    dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'musiccoca_text.safetensors'
    t = text_encoder(root / 'text_encoder.tflite')
    t['rvq.codebooks'] = quantizer(root / 'pretrained_vector_quantizer.tflite')
    t.update(mapper(root / 'mapper.tflite'))
    t.update(spm_vocab(root / 'spm.model'))
    save_file({k: np.ascontiguousarray(v) for k, v in t.items()}, str(dst),
              metadata={'source': 'google/magenta-realtime-2 MusicCoCa (CC-BY 4.0)'})
    print(json.dumps({k: list(v.shape) for k, v in list(t.items())[:4]}), '...', len(t), 'tensors ->', dst)


if __name__ == '__main__':
    main()
