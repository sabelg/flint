"""Engine-Generic Synchronous Speech & Giant-Robot AI OS DSP Synthesizer (`flint.platform.speech`).

Synthesizes intelligible spoken tactical guidance into 48 kHz Stereo `WavClip`s:
  1. Renders dry 24 kHz speech with Flint's pure-Mojo rule-based formant voice
     (`flint.audio.formant_tts`: misaki G2P, Klatt-style duration / F0 rules, cascade-parallel
     formant synthesizer with an LF glottal source). Each `VOICE_*` profile maps to one
     `formant_voice` preset (vocal-tract scale, F0 register, range, rate and voice quality).
     G2P is rules-only by default; set `FLINT_G2P_DIR` to a misaki lexicon directory
     (`us_gold.json` / `us_silver.json`, `gb_*` for British presets) for dictionary
     pronunciations.
  2. Masters dry speech through a 48 kHz Stereo Giant-Robot / Spaceship OS DSP chain:
     cubic Hermite upsampling, dual-head raised-cosine granular 3-voice harmonizer
     (`-26 cents` Left @ `6.5 ms` Haas, `+26 cents` Right @ `10.5 ms` Haas, and
     `-4.6 st` `0.765x` deep chest sub-voice), dual short-lag (`0.78 ms` & `2.38 ms`)
     titanium vocal-box comb resonance, warm `CytomicSVF` studio vocal-body sculpting,
     and dark-titanium `StereoReverbFDN` cockpit spatialization.
"""
from std.math import sin, cos, exp, floor, abs
from flint.platform.os import env_value
from flint.platform.audio import WavClip
from flint.platform.dsp import (
    DSP_TAU,
    PinkNoise,
    CytomicSVF,
    StereoReverbFDN,
    hash_white_noise,
    soft_clip_tanh,
    tube_saturate,
    wrap_phase01,
)
from flint.audio.formant_tts.api import FormantTTS, FORMANT_TTS_SAMPLE_RATE
from flint.audio.formant_tts.voices import formant_voice, FV_DEFAULT

comptime SPEECH_OUT_RATE: Int = 48000
comptime SPEECH_OUT_SR_F: Float32 = 48000.0


def _formant_speech_mono(text: String, voice_id: Int, rate_wpm: Int = 0) -> List[Float32]:
    """Dry 24 kHz mono speech from the formant voice preset `voice_id` (`FV_*`).

    `rate_wpm > 0` overrides the preset's speaking rate. Uses the misaki lexicon in
    `FLINT_G2P_DIR` when it is set and loads, else the rules-only G2P. Never raises: a G2P
    failure returns an empty buffer."""
    var v = formant_voice(voice_id)
    if rate_wpm > 0:
        v.rate_wpm = Float32(rate_wpm)
    var lex = env_value("FLINT_G2P_DIR")
    try:
        if lex.byte_length() > 0:
            try:
                var tts_lex = FormantTTS.load(lex, v.british)
                return tts_lex.synthesize(text, v)
            except:
                pass
        var tts = FormantTTS(v.british)
        return tts.synthesize(text, v)
    except:
        return List[Float32]()


def _synth_klatt_fallback_mono(text: String, voice_id: Int = FV_DEFAULT) -> List[Float32]:
    """Dry 24 kHz mono formant speech (kept under its historical name for tools and evals)."""
    return _formant_speech_mono(text, voice_id)


@always_inline
def _hermite_sample(buf: List[Float32], pos: Float64) -> Float32:
    var n = len(buf)
    if n == 0:
        return 0.0
    var i1 = Int(floor(pos))
    var frac = Float32(pos - floor(pos))
    var i0 = max(0, min(n - 1, i1 - 1))
    var ic = max(0, min(n - 1, i1))
    var i2 = max(0, min(n - 1, i1 + 1))
    var i3 = max(0, min(n - 1, i1 + 2))
    var y0 = buf[i0]
    var y1 = buf[ic]
    var y2 = buf[i2]
    var y3 = buf[i3]
    var c0 = y1
    var c1 = Float32(0.5) * (y2 - y0)
    var c2 = y0 - Float32(2.5) * y1 + Float32(2.0) * y2 - Float32(0.5) * y3
    var c3 = Float32(0.5) * (y3 - y0) + Float32(1.5) * (y1 - y2)
    return ((c3 * frac + c2) * frac + c1) * frac + c0


def _pitch_shift_ola(x: List[Float32], ratio: Float32, grain: Int) -> List[Float32]:
    """Dual-head raised-cosine (`Hann`) time-preserving granular pitch-shifter (`48 kHz`)."""
    var n = len(x)
    var out = List[Float32](capacity=max(1, n))
    if n < 4:
        for i in range(n):
            out.append(x[i])
        return out^

    var g_f = Float32(max(128, grain))
    var delta = (Float32(1.0) - ratio) / g_f
    var ph = Float32(0.0)
    var max_pos = Float32(n - 2)

    for i in range(n):
        ph = wrap_phase01(ph + delta)
        var p2 = wrap_phase01(ph + Float32(0.5))
        var w1 = Float32(0.5) - Float32(0.5) * cos(DSP_TAU * ph)
        var w2 = Float32(0.5) - Float32(0.5) * cos(DSP_TAU * p2)

        var pos1 = max(Float32(0.0), min(max_pos, Float32(i) - ph * g_f))
        var pos2 = max(Float32(0.0), min(max_pos, Float32(i) - p2 * g_f))
        var i1 = Int(pos1)
        var f1 = pos1 - Float32(i1)
        var i2 = Int(pos2)
        var f2 = pos2 - Float32(i2)

        var s1 = x[i1] * (Float32(1.0) - f1) + x[i1 + 1] * f1
        var s2 = x[i2] * (Float32(1.0) - f2) + x[i2 + 1] * f2
        out.append(s1 * w1 + s2 * w2)
    return out^


def synth_ai_copilot_clip(
    text: String,
    lead_in_sec: Float32 = 0.24,
    voice_name: String = "en-gb-x-rp+m3",
    rate_wpm: Int = 148,
    pitch_0_99: Int = 26,
    range_0_99: Int = 20,
) -> WavClip:
    """Synthesizes a 48 kHz Stereo Giant-Robot / Spaceship OS AI Co-Pilot voice `WavClip`.
    Includes `lead_in_sec` pre-roll so the simultaneous Guidance Start Alert Chime rings
    cleanly while `NativeAudio` smoothly auto-ducks BGM before the first spoken word."""
    # `voice_name` / `pitch_0_99` / `range_0_99` were espeak-ng controls; the AI OS formant preset
    # (calm RP baritone, 104 Hz, 2.5 st range) now carries voice, register and range.
    _ = voice_name
    _ = pitch_0_99
    _ = range_0_99
    var mono = _formant_speech_mono(text, VOICE_AI_OS, rate_wpm)
    var src_sr = FORMANT_TTS_SAMPLE_RATE
    if len(mono) < 64:
        mono = List[Float32](length=64, fill=0.0)

    # Trim trailing digital silence (`|s| < 0.0025`) so `is_speech_playing()` releases promptly
    var end_idx = len(mono)
    while end_idx > 256 and abs(mono[end_idx - 1]) < Float32(0.0025):
        end_idx -= 1

    # Peak-normalize dry mono buffer to consistent studio vocal level (`0.82`)
    var peak = Float32(0.05)
    for i in range(end_idx):
        var a = abs(mono[i])
        if a > peak:
            peak = a
    var norm_gain = Float32(0.82) / peak

    var lead_frames = max(0, Int(lead_in_sec * SPEECH_OUT_SR_F))
    var voice_frames = max(64, Int(Float64(end_idx) * Float64(SPEECH_OUT_RATE) / Float64(src_sr)))
    var tail_frames = Int(Float32(0.095) * SPEECH_OUT_SR_F)
    var total_frames = lead_frames + voice_frames + tail_frames

    # Upsample dry baritone speech to 48 kHz mono for multi-voice granular pitch harmonization
    var step_src = Float64(src_sr) / Float64(SPEECH_OUT_RATE)
    var dry48 = List[Float32](capacity=voice_frames)
    for vf in range(voice_frames):
        var src_pos = Float64(vf) * step_src
        var fade_in = min(Float32(1.0), Float32(vf) / Float32(384.0))
        var fade_out = min(Float32(1.0), Float32(voice_frames - vf) / Float32(480.0))
        dry48.append(_hermite_sample(mono, src_pos) * norm_gain * fade_in * fade_out)

    # 3-Voice Binaural Giant-Robot OS Harmonizer:
    #   - Left micro-pitch (`0.985x`, `-26 cents`) @ `6.5 ms` (`312` samples) Haas offset
    #   - Right micro-pitch (`1.015x`, `+26 cents`) @ `10.5 ms` (`504` samples) Haas offset
    #   - Deep chest sub-harmonic layer (`0.765x`, `-4.6 semitones`) anchored dead-center
    var v_l = _pitch_shift_ola(dry48, 0.985, 1680)
    var v_r = _pitch_shift_ola(dry48, 1.015, 1680)
    var v_sub = _pitch_shift_ola(dry48, 0.765, 1920)
    var haas_l = 312
    var haas_r = 504

    var clip = WavClip()
    clip.valid = True
    clip.channels = 2
    clip.sample_rate = SPEECH_OUT_RATE
    clip.samples.reserve(total_frames * 2)

    # Dual short-lag titanium vocal-box comb resonators (`0.77 ms` & `2.38 ms`) + Studio EQ
    var comb1_l = List[Float32](length=37, fill=0.0)
    var comb1_r = List[Float32](length=39, fill=0.0)
    var comb2_l = List[Float32](length=113, fill=0.0)
    var comb2_r = List[Float32](length=119, fill=0.0)
    var hp_clean = CytomicSVF(72.0, 0.78, SPEECH_OUT_SR_F)
    var presence_l = CytomicSVF(2550.0, 1.35, SPEECH_OUT_SR_F)
    var presence_r = CytomicSVF(2550.0, 1.35, SPEECH_OUT_SR_F)
    var warm_lp_l = CytomicSVF(6200.0, 0.72, SPEECH_OUT_SR_F)
    var warm_lp_r = CytomicSVF(6200.0, 0.72, SPEECH_OUT_SR_F)
    var chamber_rev = StereoReverbFDN(0.38, 0.56)

    for f in range(total_frames):
        var mix_l = Float32(0.0)
        var mix_r = Float32(0.0)
        if f >= lead_frames and f < lead_frames + voice_frames:
            var vf = f - lead_frames
            _ = hp_clean.process(dry48[vf])
            var c_dry = hp_clean.hp
            var vl = v_l[vf - haas_l] if vf >= haas_l else Float32(0.0)
            var vr = v_r[vf - haas_r] if vf >= haas_r else Float32(0.0)
            var vsub = v_sub[vf]
            mix_l = c_dry * Float32(0.56) + vl * Float32(0.54) - vr * Float32(0.15) + vsub * Float32(0.24)
            mix_r = c_dry * Float32(0.56) + vr * Float32(0.54) - vl * Float32(0.15) + vsub * Float32(0.24)

        # Subtle dark-titanium telemetry lock micro-tail (`880 -> 587.33 Hz`) right as transmission closes
        var pip = Float32(0.0)
        if f >= lead_frames + voice_frames and f < lead_frames + voice_frames + 1440:
            var pt = Float32(f - (lead_frames + voice_frames)) / SPEECH_OUT_SR_F
            var pf = Float32(880.0) if pt < Float32(0.014) else Float32(587.33)
            pip = sin(pt * pf * DSP_TAU) * exp(-pt * Float32(72.0)) * Float32(0.11)

        var c1_idx_l = f % 37
        var c1_idx_r = f % 39
        var c2_idx_l = f % 113
        var c2_idx_r = f % 119
        var res_l = mix_l + comb1_l[c1_idx_l] * Float32(0.23) + comb2_l[c2_idx_l] * Float32(0.15)
        var res_r = mix_r + comb1_r[c1_idx_r] * Float32(0.23) + comb2_r[c2_idx_r] * Float32(0.15)
        comb1_l[c1_idx_l] = res_l
        comb1_r[c1_idx_r] = res_r
        comb2_l[c2_idx_l] = res_l
        comb2_r[c2_idx_r] = res_r

        _ = presence_l.process(res_l)
        _ = presence_r.process(res_r)
        var eq_l = warm_lp_l.process(res_l + presence_l.bp * Float32(0.28)) + pip
        var eq_r = warm_lp_r.process(res_r + presence_r.bp * Float32(0.28)) + pip

        var sat_l = tube_saturate(eq_l, Float32(1.22))
        var sat_r = tube_saturate(eq_r, Float32(1.22))
        var wet = chamber_rev.process_send(sat_l * Float32(0.11), sat_r * Float32(0.11))

        var out_l = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh((sat_l + wet[0]) * Float32(1.08))))
        var out_r = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh((sat_r + wet[1]) * Float32(1.08))))
        clip.samples.append(Int16(Int(out_l * Float32(29500.0))))
        clip.samples.append(Int16(Int(out_r * Float32(29500.0))))

    return clip^


# ==============================================================================
# C3: Multi-Character Bridge & Squadron Radio Comms DSP
# ==============================================================================

comptime VOICE_AI_OS: Int = 0
comptime VOICE_BRIDGE_FEMALE: Int = 1  # Lisa Hayes (SDF-1 Bridge Tactical Officer)
comptime VOICE_ACE_BARITONE: Int = 2   # Roy Focker (Skull Leader)
comptime VOICE_ACE_TENOR: Int = 3      # Max Sterling (Blue Ace)
comptime VOICE_ACE_CRIMSON: Int = 4    # Miriya Parina (Quadrono Female Ace)


def synth_radio_comms_clip(
    text: String,
    profile_id: Int,
    lead_in_sec: Float32 = 0.14,
    radio_bandpass: Bool = True,
) -> WavClip:
    """Synthesizes a 48 kHz Stereo character radio voice clip with distinct vocal-tract formants,
    fundamental pitch and voice quality (one `formant_voice` preset per profile), military UHF
    `320–3350 Hz` bandpass, PTT click, and two-tone tactical squelch tail."""
    if profile_id == VOICE_AI_OS:
        return synth_ai_copilot_clip(text, lead_in_sec)

    # Each profile's register, vocal-tract scale and speaking rate live in its formant preset
    # (`flint.audio.formant_tts.voices`), so the dry voice is not pitch-shifted afterwards.
    var pitch_shift = Float32(1.0)
    var pan_bias = Float32(0.0)
    var roger_hi = Float32(1174.66)
    var roger_lo = Float32(880.0)

    if profile_id == VOICE_BRIDGE_FEMALE:
        pan_bias = -0.10
        roger_hi = 1318.51
        roger_lo = 987.77
    elif profile_id == VOICE_ACE_BARITONE:
        pan_bias = -0.16
        roger_hi = 1046.50
        roger_lo = 783.99
    elif profile_id == VOICE_ACE_TENOR:
        pan_bias = 0.16
        roger_hi = 1174.66
        roger_lo = 987.77
    elif profile_id == VOICE_ACE_CRIMSON:
        pan_bias = 0.20
        roger_hi = 1396.91
        roger_lo = 698.46

    var voice_id = profile_id if (profile_id >= VOICE_BRIDGE_FEMALE and profile_id <= VOICE_ACE_CRIMSON) else FV_DEFAULT
    var mono = _formant_speech_mono(text, voice_id)
    var src_sr = FORMANT_TTS_SAMPLE_RATE
    if len(mono) < 64:
        mono = List[Float32](length=64, fill=0.0)

    var end_idx = len(mono)
    while end_idx > 256 and abs(mono[end_idx - 1]) < Float32(0.0025):
        end_idx -= 1

    var peak = Float32(0.05)
    for i in range(end_idx):
        var a = abs(mono[i])
        if a > peak:
            peak = a
    var norm_gain = Float32(0.84) / peak

    var lead_frames = max(0, Int(lead_in_sec * SPEECH_OUT_SR_F))
    var voice_frames = max(64, Int(Float64(end_idx) * Float64(SPEECH_OUT_RATE) / Float64(src_sr)))
    var tail_frames = Int(Float32(0.11) * SPEECH_OUT_SR_F)
    var total_frames = lead_frames + voice_frames + tail_frames

    var step_src = Float64(src_sr) / Float64(SPEECH_OUT_RATE)
    var dry48 = List[Float32](capacity=voice_frames)
    for vf in range(voice_frames):
        var src_pos = Float64(vf) * step_src
        var fade_in = min(Float32(1.0), Float32(vf) / Float32(320.0))
        var fade_out = min(Float32(1.0), Float32(voice_frames - vf) / Float32(384.0))
        dry48.append(_hermite_sample(mono, src_pos) * norm_gain * fade_in * fade_out)

    var voiced48 = _pitch_shift_ola(dry48, pitch_shift, 1536)
    var alien_chorus = _pitch_shift_ola(dry48, pitch_shift * Float32(0.985), 1440) if profile_id == VOICE_ACE_CRIMSON else List[Float32]()

    var clip = WavClip()
    clip.valid = True
    clip.channels = 2
    clip.sample_rate = SPEECH_OUT_RATE
    clip.samples.reserve(total_frames * 2)

    var hp_uhf = CytomicSVF(Float32(320.0) if radio_bandpass else Float32(85.0), 0.85, SPEECH_OUT_SR_F)
    var mid_bite = CytomicSVF(2150.0, 1.55, SPEECH_OUT_SR_F)
    var lp_uhf = CytomicSVF(Float32(3350.0) if radio_bandpass else Float32(6800.0), 0.88, SPEECH_OUT_SR_F)
    var cabin_rev = StereoReverbFDN(0.26, 0.42)
    var pink = PinkNoise(313 + profile_id * 97)
    var gain_l = Float32(1.0) - max(Float32(0.0), pan_bias) * Float32(0.35)
    var gain_r = Float32(1.0) + min(Float32(0.0), pan_bias) * Float32(0.35)

    for f in range(total_frames):
        var sig = Float32(0.0)
        # PTT open carrier burst
        if radio_bandpass and f >= max(0, lead_frames - 720) and f < lead_frames:
            var kt = Float32(f - max(0, lead_frames - 720)) / SPEECH_OUT_SR_F
            sig += hash_white_noise(f * 17 + profile_id) * exp(-kt * Float32(90.0)) * Float32(0.16)

        if f >= lead_frames and f < lead_frames + voice_frames:
            var vf = f - lead_frames
            var v = voiced48[vf]
            if profile_id == VOICE_ACE_CRIMSON and vf < len(alien_chorus):
                v = v * Float32(0.76) + alien_chorus[vf] * Float32(0.30)
            _ = hp_uhf.process(v)
            _ = mid_bite.process(hp_uhf.hp)
            var shaped = lp_uhf.process(hp_uhf.hp + mid_bite.bp * (Float32(0.55) if radio_bandpass else Float32(0.22)))
            if radio_bandpass:
                shaped = tube_saturate(shaped + pink.next() * Float32(0.012), Float32(1.48))
            sig += shaped

        # Two-tone tactical UHF squelch tail + carrier close click
        if f >= lead_frames + voice_frames and f < lead_frames + voice_frames + 2400:
            var pt = Float32(f - (lead_frames + voice_frames)) / SPEECH_OUT_SR_F
            var pf = roger_hi if pt < Float32(0.022) else roger_lo
            var beep = sin(pt * pf * DSP_TAU) * exp(-pt * Float32(48.0)) * Float32(0.12)
            var squelch = (hash_white_noise(f * 29 + 3) * exp(-max(Float32(0.0), pt - Float32(0.028)) * Float32(110.0)) * Float32(0.09)) if (radio_bandpass and pt >= Float32(0.028)) else Float32(0.0)
            sig += beep + squelch

        var wet = cabin_rev.process_send(sig * gain_l * Float32(0.07), sig * gain_r * Float32(0.07))
        var out_l = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh(sig * gain_l + wet[0])))
        var out_r = max(Float32(-1.0), min(Float32(1.0), soft_clip_tanh(sig * gain_r + wet[1])))
        clip.samples.append(Int16(Int(out_l * Float32(29000.0))))
        clip.samples.append(Int16(Int(out_r * Float32(29000.0))))

    return clip^
