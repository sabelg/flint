"""Synthetic voice-command test clips (no recorded voices) with Kokoro-82M (Apache-2.0).

usage: python tools/moonshine/make_clips.py KOKORO_ONNX_DIR OUT_DIR

KOKORO_ONNX_DIR is an onnx-community/Kokoro-82M-v1.0-ONNX download (onnx/model.onnx and
voices/*.bin). Writes four 24 kHz 16-bit WAVs of short game / editor commands; needs
onnxruntime, soundfile and kokoro-onnx (for its phonemizer).
"""
import sys, numpy as np, onnxruntime as ort, soundfile as sf
from kokoro_onnx.tokenizer import Tokenizer
if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)
D, out = sys.argv[1:3]
s=ort.InferenceSession(D+'/onnx/model.onnx'); t=Tokenizer()
clips=[('cmd_gate','af_heart','Open the north gate and light the torches.'),
       ('cmd_studio','am_michael','Select the camera and move it five meters to the left.'),
       ('cmd_squad','bf_emma','Squad, hold position. Fire on my signal.'),
       ('cmd_save','bm_george','Save the level, then start a new play session.')]
for name,voice,text in clips:
    ids=t.tokenize(t.phonemize(text,'en-us'))
    vs=np.fromfile(f'{D}/voices/{voice}.bin',dtype=np.float32).reshape(-1,1,256)
    style=vs[len(ids)]
    a=s.run(None,{'input_ids':np.array([[0,*ids,0]],dtype=np.int64),'style':style,'speed':np.ones(1,np.float32)})[0].reshape(-1)
    sf.write(f'{out}/{name}.wav',a,24000,subtype='PCM_16'); print(name,len(a)/24000,text)
