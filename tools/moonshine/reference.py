"""Reference dump for `flint.ml.moonshine` parity checks (`tests/test_flint_v5_moonshine.mojo`).

usage: python tools/moonshine/reference.py MODEL_DIR in.wav OUT_PREFIX [seconds]

  MODEL_DIR   a Hugging Face moonshine-ai/moonshine-streaming-{tiny,small,medium} folder
  in.wav      any rate / channel count; the first `seconds` are used when given
  OUT_PREFIX  receives
     OUT_PREFIX.wav16.f32   the 16 kHz mono input (float32), resampled with the same
                            windowed-sinc filter as `flint.ml.moonshine_audio.Resampler`
     OUT_PREFIX.enc.f32     encoder output [frames, encoder hidden]
     OUT_PREFIX.logits.f32  greedy-decoding logits [steps, vocab]
     OUT_PREFIX.json        token ids, transcript, sizes

Needs torch, transformers (>= 5, with `moonshine_streaming`), numpy and soundfile. Run the
test with MOONSHINE_DIR=MODEL_DIR MOONSHINE_REF=OUT_PREFIX.
"""
import sys, os, math, json
import numpy as np, torch, soundfile as sf
from transformers import MoonshineStreamingForConditionalGeneration, AutoTokenizer

def kaiser(x, beta):
    return np.i0(beta*np.sqrt(np.clip(1-x*x,0,1)))/np.i0(beta)
def resample(x, fin, fout, zeros=16, beta=8.6, roll=0.945):
    """Windowed-sinc polyphase resampler, the algorithm flint.ml.moonshine_audio implements."""
    if fin == fout: return x.astype(np.float32)
    g = math.gcd(fin, fout); L = fout//g; M = fin//g
    fc = min(1.0, fout/fin)*roll
    half = int(math.ceil(zeros/fc))
    taps = 2*half
    n_out = (len(x)*L)//M
    # phase p: output n has t = n*M/L = base + p/L, base = (n*M)//L, p = (n*M)%L
    table = np.zeros((L, taps), np.float64)
    for p in range(L):
        frac = p/L
        k = np.arange(-half+1, half+1)  # taps relative to base
        d = frac - k   # t - (base+k)
        w = kaiser(d/half, beta)
        table[p] = fc*np.sinc(fc*d)*w
    y = np.zeros(n_out, np.float64)
    xp = np.concatenate([np.zeros(half), x.astype(np.float64), np.zeros(half+1)])
    for n in range(n_out):
        base = (n*M)//L; p = (n*M)%L
        seg = xp[base+1: base+1+taps]   # x[base-half+1 .. base+half]
        y[n] = np.dot(table[p], seg)
    return y.astype(np.float32)


if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)
mdir, wav, out = sys.argv[1:4]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.set_num_threads(8)
a, sr = sf.read(wav, dtype='float32', always_2d=True)
a = a.mean(axis=1)
if secs: a = a[:int(secs*sr)]
x = resample(a, sr, 16000)
pad = (-len(x)) % 80
xp = np.concatenate([x, np.zeros(pad, np.float32)])
mask = np.concatenate([np.ones(len(x), np.int64), np.zeros(pad, np.int64)])
model = MoonshineStreamingForConditionalGeneration.from_pretrained(mdir, torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(mdir)
iv = torch.from_numpy(xp)[None]; am = torch.from_numpy(mask)[None]
with torch.no_grad():
    enc = model.model.encoder(iv, attention_mask=am)
    H = enc.last_hidden_state
    max_len = int(len(x) * 6.5 / 16000)
    g = model.generate(input_values=iv, attention_mask=am, max_length=max_len, do_sample=False, num_beams=1,
                       output_scores=True, return_dict_in_generate=True)
ids = g.sequences[0].tolist()
logits = torch.stack([s[0] for s in g.scores]).numpy().astype(np.float32)
text = tok.decode(ids, skip_special_tokens=True)
x.tofile(out + '.wav16.f32'); H[0].numpy().astype(np.float32).tofile(out + '.enc.f32')
logits.tofile(out + '.logits.f32')
json.dump({'samples': len(x), 'enc_frames': H.shape[1], 'hidden': H.shape[2], 'ids': ids, 'steps': logits.shape[0],
           'text': text, 'max_length': max_len}, open(out + '.json', 'w'))
print(os.path.basename(out), H.shape, len(ids), repr(text))
