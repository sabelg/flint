#!/usr/bin/env python3
"""onnxruntime parity dumps for flint.ml.migan.

    python tools/migan/reference.py MIGAN_ONNX IMAGE OUT_DIR

Runs `migan.onnx` (tools/migan/fetch.sh) on two crops of IMAGE (a 512 x 512 one and a 384 x 320
one that exercises the graph's internal resizes) with a fixed rectangle + stroke hole mask, and
writes per case NAME_{image,mask,out}.u8 (NCHW uint8), NAME_mid.f32 (the float generator output
before the uint8 blend, graph value '1310') and cases.txt. tests/test_flint_v5_marigold_migan.mojo
reads them with FLINT_MIGAN_REF=OUT_DIR. Needs numpy, onnx, onnxruntime and Pillow.
"""
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


def mask_for(h, w, seed):
    """255 = keep, 0 = repaint: a rectangle and six thin strokes."""
    rng = np.random.default_rng(seed)
    mk = np.full((h, w), 255, np.uint8)
    mk[h // 3: h // 3 + h // 4, w // 4: w // 4 + w // 3] = 0
    for _ in range(6):
        y = rng.integers(0, h)
        x0 = rng.integers(0, w // 2)
        mk[max(0, y - 3): y + 3, x0: x0 + w // 3] = 0
    return mk


def main():
    src, img_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    m = onnx.load(src)
    m.graph.output.append(onnx.helper.make_tensor_value_info('1310', onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
    base = np.asarray(Image.open(img_path).convert('RGB').resize((512, 512), Image.BICUBIC))
    cases = [('full512', 0, 0, 512, 512), ('crop320x384', 40, 64, 320, 384)]
    lines = []
    for name, y0, x0, h, w in cases:
        im = np.ascontiguousarray(base[y0:y0 + h, x0:x0 + w].transpose(2, 0, 1)[None])
        mk = mask_for(h, w, 7)[None, None]
        res, mid = sess.run(['result', '1310'], {'image': im, 'mask': mk})
        t = time.time()
        for _ in range(3):
            sess.run(['result'], {'image': im, 'mask': mk})
        print('%s: onnxruntime CPU %.3f s / pass' % (name, (time.time() - t) / 3))
        im.tofile(os.path.join(out, name + '_image.u8'))
        mk.tofile(os.path.join(out, name + '_mask.u8'))
        res.tofile(os.path.join(out, name + '_out.u8'))
        mid.astype(np.float32).tofile(os.path.join(out, name + '_mid.f32'))
        lines.append('%s %d %d' % (name, h, w))
    open(os.path.join(out, 'cases.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
