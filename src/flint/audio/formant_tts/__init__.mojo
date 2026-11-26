"""Pure-Mojo rule-based formant text-to-speech (`flint.audio.formant_tts`).

A Klatt cascade/parallel formant synthesiser (KLSYN88 / MITalk / DECtalk class) driven by
synthesis-by-rule from misaki phonemes (`flint.ml.phonemize.G2P`): zero weights, zero C
dependencies, deterministic, far faster than real time on one core.

    from flint.audio.formant_tts import FormantTTS, formant_voice, FV_DEFAULT
    var tts = FormantTTS()                         # rules-only G2P; FormantTTS.load(dir) for the lexicon
    var pcm = tts.synthesize("Hello there.", formant_voice(FV_DEFAULT))   # 24 kHz mono Float32

Modules: `phonemes` (target table), `rules` (segments, durations, allophones, coarticulation,
F0 model -> 5 ms parameter tracks), `synth` (resonators, LF source, noise, output stage),
`voices` (presets), `api` (`FormantTTS`, `synthesize`).
"""
