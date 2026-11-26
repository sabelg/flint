"""Flint v5 speech backend suite (`test_flint_v5_speech_backend.mojo`).

Always (no model files needed):
  1. `FLINT_SPEECH_BACKEND=klatt` and "no Kokoro weights" both select the Klatt voice, and give
     bit-identical, finite, deterministic 48 kHz stereo clips for every profile.
  2. Every profile has a Kokoro voice preset; the cache directory honours `FLINT_SPEECH_CACHE_DIR`.
With Kokoro weights (`$FLINT_KOKORO_DIR` or `~/flint-models/kokoro`):
  3. `auto` selects Kokoro, the model loads once, clips are finite 48 kHz stereo, and the second
     identical request is an on-disk cache hit with the same samples; `pregenerate_speech_lines`
     fills the cache ahead of time.
The cache used here is a private temp directory (never the user's cache).
"""
from std.ffi import external_call
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from flint.platform.os import env_now, make_dirs
from flint.platform.audio import WavClip
from flint.platform.speech import (
    synth_ai_copilot_clip,
    synth_radio_comms_clip,
    active_speech_backend,
    speech_backend_request,
    kokoro_weights_dir,
    kokoro_voice_preset,
    kokoro_load_seconds,
    speech_cache_stats,
    speech_cache_dir,
    kokoro_clips_rendered,
    pregenerate_speech_lines,
    SPEECH_BACKEND_KLATT,
    SPEECH_BACKEND_KOKORO,
    SPEECH_BACKEND_AUTO,
    VOICE_AI_OS,
    VOICE_BRIDGE_FEMALE,
    VOICE_ACE_BARITONE,
    VOICE_ACE_TENOR,
    VOICE_ACE_CRIMSON,
)


def _setenv(name: String, value: String):
    var n = name
    var v = value
    _ = external_call["setenv", Int32](n.as_c_string_span(), v.as_c_string_span(), Int32(1))


def _unsetenv(name: String):
    var n = name
    _ = external_call["unsetenv", Int32](n.as_c_string_span())


def _check_clip(c: WavClip, what: String) raises:
    assert_true(c.valid, what + ": invalid clip")
    assert_equal(c.channels, 2)
    assert_equal(c.sample_rate, 48000)
    assert_true(len(c.samples) > 12000, what + ": too short")
    var peak = 0
    for s in c.samples:
        peak = max(peak, abs(Int(s)))
    assert_true(peak > 1000 and peak <= 32767, what + ": silent or out of range")


def _same(a: WavClip, b: WavClip) -> Bool:
    if len(a.samples) != len(b.samples) or a.channels != b.channels or a.sample_rate != b.sample_rate:
        return False
    for i in range(len(a.samples)):
        if a.samples[i] != b.samples[i]:
            return False
    return True


def _profiles() -> List[Int]:
    return [VOICE_AI_OS, VOICE_BRIDGE_FEMALE, VOICE_ACE_BARITONE, VOICE_ACE_TENOR, VOICE_ACE_CRIMSON]


def test_klatt_fallback() raises:
    var line = String("Roger that, Skull Leader.")
    # forced Klatt
    _setenv("FLINT_SPEECH_BACKEND", "klatt")
    assert_equal(speech_backend_request(), SPEECH_BACKEND_KLATT)
    assert_equal(active_speech_backend(), SPEECH_BACKEND_KLATT)
    var forced = List[WavClip]()
    for p in _profiles():
        var a = synth_radio_comms_clip(line, p)
        var b = synth_radio_comms_clip(line, p)
        _check_clip(a, "klatt profile " + String(p))
        assert_true(_same(a, b), "Klatt clip not deterministic")
        forced.append(a^)
    # auto, but no weights anywhere: same Klatt clips
    var home = env_now("HOME")
    var kdir = env_now("FLINT_KOKORO_DIR")
    _setenv("FLINT_SPEECH_BACKEND", "auto")
    _setenv("HOME", "/nonexistent-flint-home")
    _setenv("FLINT_KOKORO_DIR", "/nonexistent-flint-kokoro")
    assert_equal(speech_backend_request(), SPEECH_BACKEND_AUTO)
    assert_equal(kokoro_weights_dir(), "")
    assert_equal(active_speech_backend(), SPEECH_BACKEND_KLATT)
    var ps = _profiles()
    for i in range(len(ps)):
        var c = synth_radio_comms_clip(line, ps[i])
        assert_true(_same(c, forced[i]), "auto without weights is not the Klatt voice")
    var ai = synth_ai_copilot_clip("Guidance online. Turn left.")
    _check_clip(ai, "klatt ai os")
    assert_equal(kokoro_clips_rendered(), 0)
    _setenv("HOME", home)
    if kdir.byte_length() > 0:
        _setenv("FLINT_KOKORO_DIR", kdir)
    else:
        _unsetenv("FLINT_KOKORO_DIR")
    print("✓ Klatt fallback: forced and weightless auto agree, deterministic, finite")


def test_presets_and_cache_dir() raises:
    for p in _profiles():
        var v = kokoro_voice_preset(p)
        assert_true(v.voice.byte_length() > 0 and v.speed > 0.5 and v.speed < 1.5 and v.pitch > 0.8 and v.pitch < 1.25)
    assert_true(kokoro_voice_preset(VOICE_ACE_CRIMSON).voice.startswith("bf_"), "Crimson is British female")
    assert_true(kokoro_voice_preset(VOICE_BRIDGE_FEMALE).voice.startswith("af_"))
    var tmp = env_now("TMPDIR")
    if tmp.byte_length() == 0:
        tmp = "/tmp"
    var dir = tmp + "/flint-speech-test-" + String(Int(external_call["getpid", Int32]()))
    _setenv("FLINT_SPEECH_CACHE_DIR", dir)
    assert_equal(speech_cache_dir(), dir)
    print("✓ Kokoro voice presets; cache dir", dir)


def test_kokoro_path() raises:
    _setenv("FLINT_SPEECH_BACKEND", "auto")
    if kokoro_weights_dir().byte_length() == 0:
        print("- Kokoro path skipped (no weights in $FLINT_KOKORO_DIR or ~/flint-models/kokoro)")
        return
    var t0 = perf_counter_ns()
    assert_equal(active_speech_backend(), SPEECH_BACKEND_KOKORO)
    var load_s = Float64(perf_counter_ns() - t0) / 1e9
    var line = String("Bridge to Skull Leader, catapult cleared for launch.")
    var st0 = speech_cache_stats()
    var t1 = perf_counter_ns()
    var a = synth_radio_comms_clip(line, VOICE_BRIDGE_FEMALE)
    var synth_s = Float64(perf_counter_ns() - t1) / 1e9
    _check_clip(a, "kokoro bridge")
    var st1 = speech_cache_stats()
    assert_equal(st1[1], st0[1] + 1)
    var t2 = perf_counter_ns()
    var b = synth_radio_comms_clip(line, VOICE_BRIDGE_FEMALE)
    var hit_s = Float64(perf_counter_ns() - t2) / 1e9
    var st2 = speech_cache_stats()
    assert_equal(st2[0], st1[0] + 1)
    assert_true(_same(a, b), "cache hit returns a different clip")
    var n = kokoro_clips_rendered()
    # the Klatt clip of the same line differs (the backend really switched)
    _setenv("FLINT_SPEECH_BACKEND", "klatt")
    var k = synth_radio_comms_clip(line, VOICE_BRIDGE_FEMALE)
    assert_true(not _same(a, k))
    _setenv("FLINT_SPEECH_BACKEND", "auto")
    # every profile renders; pregeneration then hits the cache
    var lines: List[String] = ["Skull Leader, form on my wing."]
    for p in _profiles():
        var made = pregenerate_speech_lines(lines, p)
        assert_equal(made, 1)
        var again = pregenerate_speech_lines(lines, p)
        assert_equal(again, 0)
        _check_clip(synth_radio_comms_clip(lines[0], p), "kokoro profile " + String(p))
    assert_equal(kokoro_clips_rendered(), n + 5)
    var ai = synth_ai_copilot_clip("Guidance online. Turn left.")
    _check_clip(ai, "kokoro ai os")
    print("✓ Kokoro path: load", load_s, "s (model", kokoro_load_seconds(), "s), first clip", synth_s,
          "s, cache hit", hit_s, "s")


def main() raises:
    test_presets_and_cache_dir()
    test_klatt_fallback()
    test_kokoro_path()
    print("test_flint_v5_speech_backend: all passed")
