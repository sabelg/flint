"""Fixed-seed JAX (fp32) reference dumps for flint.ml.magenta_rt parity tests.

Runs the official Magenta RealTime 2 JAX implementation
(github.com/magenta/magenta-realtime, Apache-2.0) with compute_dtype=float32 and
writes .npy files that tests/test_flint_v5_magenta_rt.mojo compares against.

Usage (from a venv with the magenta-realtime repo installed, see fetch.sh):
  python -I tools/magenta_rt/reference.py WEIGHTS_DIR OUT_DIR [--frames 100]
      [--switch 50] [--style-a 12 ints] [--style-b 12 ints]

WEIGHTS_DIR is the HF google/magenta-realtime-2 layout (checkpoints/, resources/).
Outputs (each as NAME.npy and raw little-endian NAME.bin):
  init_key.npy           uint32[2]   PRNG key the sampler starts from
  cond_a.npy / cond_b.npy int32[144] conditioning rows (already offset by +7)
  tokens.npy             int32[F,12] sampled SpectroStream codes (0..1023)
  unique_tokens.npy      int32[F,12] the decoder's unique-index tokens
  audio.npy              float32[F*1920,2] streaming decoder output in [-1,1]
  int16.npy              int16[F*1920,2] what MagentaRT2System returns
  temporal_inputs0.npy / temporal_outputs0.npy / depth_logits0.npy (step 0)
  temporal_outputs1.npy  (step 1)
  enc0.npy               encoder output of step 0 (256)
  codec_codes.npy / codec_audio_layer.npy  layer-mode decode of `tokens`
"""

import argparse
import functools
import pathlib
import sys

import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('weights')
  ap.add_argument('out')
  ap.add_argument('--frames', type=int, default=100)
  ap.add_argument('--switch', type=int, default=50)
  ap.add_argument('--style-a', type=str,
                  default='660,1016,295,206,857,841,391,857,619,70,401,22')
  ap.add_argument('--style-b', type=str, default='')
  ap.add_argument('--temperature', type=float, default=1.3)
  ap.add_argument('--top-k', type=int, default=40)
  ap.add_argument('--cfg', type=str, default='3.0,1.0,1.0')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()

  import jax
  jax.config.update('jax_platforms', 'cpu')
  import jax.numpy as jnp
  from jax import random
  import flax.traverse_util as flaxtu
  import safetensors.flax as safetensors_flax
  import types
  try:
    import librosa  # noqa: F401  (only used by magenta_rt.audio resampling)
  except ImportError:
    sys.modules['librosa'] = types.ModuleType('librosa')
  import magenta_rt  # noqa: F401  (activates vendored sequence_layers)
  import sequence_layers.jax as sl
  from magenta_rt.jax import model, system, spectrostream

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(out / (name + '.npy'), arr)
    arr.tofile(out / (name + '.bin'))   # raw little-endian for the Mojo tests

  exp = model.MagentaRT2ModelSmall()
  exp.compute_dtype = jnp.float32
  ss_cfg = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
      rvq_truncation_level=exp.spectrostream.rvq_truncation_level,
      use_unique_codes=False)
  sampler = system.MagentaRT2Sampler.Config(
      depthformer=exp.depthformer_config(), spectrostream=ss_cfg).make()

  ckpt = pathlib.Path(args.weights) / 'checkpoints' / 'mrt2_small.safetensors'
  flat = safetensors_flax.load_file(str(ckpt))
  params = flaxtu.unflatten_dict({tuple(k.split('/')): v for k, v in flat.items()})

  def cond_row(style):
    cfgs = [float(v) for v in args.cfg.split(',')]
    notes = [-1] * 128   # masked: model free to choose pitches
    drums = [-1]
    cfg_tok = [system.discretize_cfg(cfgs[0], 0.2, 40),
               system.discretize_cfg(cfgs[1], 0.2, 40),
               system.discretize_cfg(cfgs[2], 1.0, 8)]
    row = np.array(style + notes + drums + cfg_tok, np.int32) + 7
    return row

  style_a = [int(v) for v in args.style_a.split(',')]
  style_b = ([int(v) for v in args.style_b.split(',')] if args.style_b
             else list(reversed(style_a)))
  cond_a, cond_b = cond_row(style_a), cond_row(style_b)
  save('cond_a', cond_a)
  save('cond_b', cond_b)

  rngs = {'params': random.PRNGKey(42), 'random': random.PRNGKey(args.seed)}
  spec = jax.ShapeDtypeStruct([cond_a.shape[0]], jnp.int32)
  constants = {'temperature': jnp.array([args.temperature]),
               'top_k': jnp.array([args.top_k], jnp.int32)}

  init = jax.jit(lambda p: sampler.apply(
      p, 1, spec, constants=constants, training=False, rngs=rngs,
      method=sampler.get_initial_state))
  state = init(params)

  # Find the decoder PRNG key inside the state (uint32 [1, 2]).
  keys = [l for l in jax.tree.leaves(state)
          if getattr(l, 'dtype', None) == jnp.uint32 and l.shape == (1, 2)]
  assert len(keys) == 1, [k.shape for k in keys]
  save('init_key', np.asarray(keys[0][0], np.uint32))

  @jax.jit
  def step(p, x, s):
    return sampler.apply(p, x=x, state=s, constants=constants, training=False,
                         rngs=rngs, method=sampler.step_with_emits,
                         mutable=['intermediates'])

  # Float audio: intercept the int16 Lambda's input.
  captured = {}
  orig = system._float_samples_to_int16
  tokens, uniq, audio, pcm = [], [], [], []
  for f in range(args.frames):
    row = cond_a if f < args.switch else cond_b
    x = sl.Sequence.from_values(row.reshape(1, 1, -1))
    (y, state, _), inter = step(params, x, state)
    im = inter['intermediates']['depthformer']['decoder']
    u = np.stack([np.asarray(v)[0, 0] for v in im['depth_samples']], -1)
    uniq.append(u)
    tokens.append((u - 6) % 1024)
    pcm.append(np.asarray(y.values[0]))
    if f == 0:
      save('temporal_inputs0', np.asarray(im['temporal_inputs'][0]).reshape(-1))
      save('temporal_outputs0', np.asarray(im['temporal_outputs'][0]).reshape(-1))
      save('depth_logits0',
              np.stack([np.asarray(v).reshape(-1) for v in im['depth_logits']]))
    if f == 1:
      save('temporal_outputs1', np.asarray(im['temporal_outputs'][0]).reshape(-1))
    if f % 10 == 0:
      print('frame', f, tokens[-1][:4], flush=True)
  tokens = np.stack(tokens).astype(np.int32)
  save('tokens', tokens)
  save('unique_tokens', np.stack(uniq).astype(np.int32))
  save('int16', np.concatenate(pcm).astype(np.int16))

  # Stand-alone codec: stream the sampled codes through codes->waveform
  # (float output) to give the Mojo codec a deterministic target.
  ss = ss_cfg.make()
  ss_params = {'params': params['params']['soundstream']}
  codes = sl.Sequence.from_values(tokens[None].astype(np.int32))
  c2w = lambda mdl, c: mdl.codes_to_waveform(c, training=False)
  wav_layer = ss.apply(ss_params, codes, method=c2w)
  save('codec_audio_layer', np.asarray(wav_layer.values[0], np.float32))
  save('codec_mask_layer', np.asarray(wav_layer.mask[0]))

  def c2w_step(mdl, c):
    layer = mdl.codes_to_waveform_layer
    s0 = layer.get_initial_state(1, sl.ChannelSpec((12,), jnp.int32), training=False)
    ys, s = [], s0
    for t in range(c.shape[1]):
      y, s = layer.step(c[:, t:t + 1], s, training=False)
      ys.append(y.values)
    return jnp.concatenate(ys, 1)
  wav_step = ss.apply(ss_params, codes, method=c2w_step)
  save('audio', np.asarray(wav_step[0], np.float32))
  print('done', tokens.shape, np.asarray(wav_step).shape)


if __name__ == '__main__':
  main()
