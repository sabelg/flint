"""Engine-Generic Synchronous Speech & Giant-Robot AI OS DSP Synthesizer (`flint.platform.speech`).

Synthesizes intelligible spoken tactical guidance into 48 kHz Stereo `WavClip`s:
  1. Uses `libespeak-ng.so.1` via runtime `dlopen` in synchronous memory-only PCM mode
     (`AUDIO_OUTPUT_SYNCHRONOUS`, zero subprocesses, zero audio-daemon blocking) when
     available on Linux (`aarch64` / `x86_64`), tuned to a calm, composed British/RP
     male baritone (`en-gb-x-rp+m3`, `F0 ~ 105-118 Hz`, flat `range = 20` OS prosody).
  2. Automatically falls back to a pure-Mojo 3-formant (`F1`/`F2`/`F3` `CytomicSVF`)
     baritone (`112 Hz`) Klatt vocal-tract synthesizer if `libespeak-ng` is absent.
  3. Masters dry speech through a 48 kHz Stereo Giant-Robot / Spaceship OS DSP chain:
     cubic Hermite upsampling, dual-head raised-cosine granular 3-voice harmonizer
     (`-26 cents` Left @ `6.5 ms` Haas, `+26 cents` Right @ `10.5 ms` Haas, and
     `-4.6 st` `0.765x` deep chest sub-voice), dual short-lag (`0.78 ms` & `2.38 ms`)
     titanium vocal-box comb resonance, warm `CytomicSVF` studio vocal-body sculpting,
     and dark-titanium `StereoReverbFDN` cockpit spatialization.
"""
from std.ffi import external_call
from flint.platform.dynlib import dl_open, dl_sym
from std.math import sin, cos, exp, floor, abs
from flint.platform.os import env_value
from flint.platform.vulkan import get64, put64
from flint.platform.audio import WavClip
from flint.platform.dsp import (
    DSP_TAU,
    PinkNoise,
    CytomicSVF,
    StereoReverbFDN,
    hash_white_noise,
    poly_blep_saw,
    soft_clip_tanh,
    tube_saturate,
    wrap_phase01,
)

comptime SPEECH_OUT_RATE: Int = 48000
comptime SPEECH_OUT_SR_F: Float32 = 48000.0

# C Capture State Block layout (3 x 64-bit words):
#   word 0: data_ptr (Int16* malloc buffer)
#   word 1: count    (number of Int16 samples written)
#   word 2: capacity (allocated Int16 sample slots)
comptime CAP_PTR: Int = 0
comptime CAP_LEN: Int = 8
comptime CAP_CAP: Int = 16


def _espeak_pcm_callback(wav_ptr: Int, num_samples: Int32, events_ptr: Int) abi("C") -> Int32:
    """Synchronous `t_espeak_callback` appending 16-bit mono PCM chunks into `user_data`."""
    if wav_ptr == 0 or num_samples <= 0 or events_ptr == 0:
        return 0
    # In 64-bit Linux `espeak_EVENT`, `void* user_data` sits at byte offset 24.
    var state = get64(events_ptr + 24)
    if state == 0:
        return 0
    var n = Int(num_samples)
    var cur_ptr = get64(state + CAP_PTR)
    var cur_len = get64(state + CAP_LEN)
    var cur_cap = get64(state + CAP_CAP)
    if cur_len + n > cur_cap:
        var new_cap = max(65536, (cur_cap * 2) + n)
        var new_ptr = external_call["realloc", Int](cur_ptr, new_cap * 2)
        if new_ptr == 0:
            return 0
        cur_ptr = new_ptr
        cur_cap = new_cap
        put64(state + CAP_PTR, cur_ptr)
        put64(state + CAP_CAP, cur_cap)
    _ = external_call["memcpy", Int](cur_ptr + cur_len * 2, wav_ptr, n * 2)
    put64(state + CAP_LEN, cur_len + n)
    return 0


def _c_str(text: String) -> List[UInt8]:
    var out = List[UInt8](capacity=text.byte_length() + 1)
    for b in text.as_bytes():
        out.append(b)
    out.append(0)
    return out^


def _parse_pos_int(s: String) -> Int:
    var v = 0
    for i in range(s.byte_length()):
        var d = Int(s.as_bytes()[i]) - 48
        if d >= 0 and d <= 9:
            v = v * 10 + d
    return v


def _try_espeak_mono(
    text: String,
    voice_name: String,
    rate_wpm: Int,
    pitch_0_99: Int,
    range_0_99: Int = 20,
) -> Tuple[List[Float32], Int]:
    """Attempts synchronous in-memory phoneme synthesis via `libespeak-ng.so.1`."""
    var empty = List[Float32]()
    if text.byte_length() == 0:
        return (empty^, 0)

    var candidates = [
        String("libespeak-ng.so.1"),
        String("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
        String("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        String("libespeak-ng.so"),
    ]
    var lib = 0
    for i in range(len(candidates)):
        var cname = candidates[i]
        lib = dl_open(cname)
        _ = cname.byte_length()
        if lib != 0:
            break
    if lib == 0:
        return (empty^, 0)

    var s_init = String("espeak_Initialize")
    var s_cb = String("espeak_SetSynthCallback")
    var s_voice = String("espeak_SetVoiceByName")
    var s_param = String("espeak_SetParameter")
    var s_synth = String("espeak_Synth")
    var s_sync = String("espeak_Synchronize")

    var fn_init = dl_sym(lib, s_init)
    var fn_cb = dl_sym(lib, s_cb)
    var fn_voice = dl_sym(lib, s_voice)
    var fn_param = dl_sym(lib, s_param)
    var fn_synth = dl_sym(lib, s_synth)
    var fn_sync = dl_sym(lib, s_sync)
    _ = s_init.byte_length()
    _ = s_cb.byte_length()
    _ = s_voice.byte_length()
    _ = s_param.byte_length()
    _ = s_synth.byte_length()
    _ = s_sync.byte_length()

    if fn_init == 0 or fn_cb == 0 or fn_voice == 0 or fn_param == 0 or fn_synth == 0 or fn_sync == 0:
        return (empty^, 0)

    # Guard `espeak_Initialize` once per process via process env (`FLINT_ESPEAK_SR`)
    var cached_sr = _parse_pos_int(env_value("FLINT_ESPEAK_SR"))
    if cached_sr <= 0:
        comptime InitFn = def(Int32, Int32, Int, Int32) thin abi("C") -> Int32
        var slot_init = fn_init
        var sr_ret = Int(Pointer(to=slot_init).unsafe_bitcast[InitFn]()[unsafe_offset=0](Int32(2), Int32(0), Int(0), Int32(0)))
        if sr_ret <= 0:
            return (empty^, 0)
        cached_sr = sr_ret
        var env_k = String("FLINT_ESPEAK_SR")
        var env_v = String(cached_sr)
        _ = external_call["setenv", Int32](env_k.as_c_string_span(), env_v.as_c_string_span(), Int32(1))
        _ = env_k.byte_length()
        _ = env_v.byte_length()

    comptime SetCbFn = def(def(Int, Int32, Int) thin abi("C") -> Int32) thin abi("C") -> NoneType
    var slot_cb = fn_cb
    Pointer(to=slot_cb).unsafe_bitcast[SetCbFn]()[unsafe_offset=0](_espeak_pcm_callback)

    comptime SetVoiceFn = def(Int) thin abi("C") -> Int32
    var slot_v = fn_voice
    var v_primary_c = _c_str(voice_name)
    var v_rc = Pointer(to=slot_v).unsafe_bitcast[SetVoiceFn]()[unsafe_offset=0](Int(v_primary_c.unsafe_ptr()))
    _ = len(v_primary_c)
    if v_rc != 0:
        var v_fb1 = _c_str("en-gb+m3")
        v_rc = Pointer(to=slot_v).unsafe_bitcast[SetVoiceFn]()[unsafe_offset=0](Int(v_fb1.unsafe_ptr()))
        _ = len(v_fb1)
    if v_rc != 0:
        var v_fb2 = _c_str("en-us+m3")
        _ = Pointer(to=slot_v).unsafe_bitcast[SetVoiceFn]()[unsafe_offset=0](Int(v_fb2.unsafe_ptr()))
        _ = len(v_fb2)

    comptime SetParamFn = def(Int32, Int32, Int32) thin abi("C") -> Int32
    var slot_p = fn_param
    # speak_lib.h: espeakRATE=1, espeakVOLUME=2, espeakPITCH=3, espeakRANGE=4
    _ = Pointer(to=slot_p).unsafe_bitcast[SetParamFn]()[unsafe_offset=0](Int32(1), Int32(rate_wpm), Int32(0))
    _ = Pointer(to=slot_p).unsafe_bitcast[SetParamFn]()[unsafe_offset=0](Int32(2), Int32(125), Int32(0))
    _ = Pointer(to=slot_p).unsafe_bitcast[SetParamFn]()[unsafe_offset=0](Int32(3), Int32(pitch_0_99), Int32(0))
    _ = Pointer(to=slot_p).unsafe_bitcast[SetParamFn]()[unsafe_offset=0](Int32(4), Int32(range_0_99), Int32(0))

    var state = external_call["calloc", Int](Int(3), Int(8))
    if state == 0:
        return (empty^, 0)
    var init_cap = 65536
    var pcm_buf = external_call["malloc", Int](init_cap * 2)
    if pcm_buf == 0:
        external_call["free", NoneType](state)
        return (empty^, 0)
    put64(state + CAP_PTR, pcm_buf)
    put64(state + CAP_LEN, 0)
    put64(state + CAP_CAP, init_cap)

    comptime SynthFn = def(Int, Int, UInt32, Int32, UInt32, UInt32, Int, Int) thin abi("C") -> Int32
    comptime SyncFn = def() thin abi("C") -> Int32
    var slot_syn = fn_synth
    var slot_sync = fn_sync
    var txt_c = _c_str(text)
    # espeakCHARS_UTF8 = 1
    _ = Pointer(to=slot_syn).unsafe_bitcast[SynthFn]()[unsafe_offset=0](
        Int(txt_c.unsafe_ptr()),
        len(txt_c),
        UInt32(0),
        Int32(0),
        UInt32(0),
        UInt32(1),
        Int(0),
        state,
    )
    _ = len(txt_c)
    _ = Pointer(to=slot_sync).unsafe_bitcast[SyncFn]()[unsafe_offset=0]()

    var final_ptr = get64(state + CAP_PTR)
    var final_len = get64(state + CAP_LEN)
    var out = List[Float32](capacity=max(1, final_len))
    if final_ptr != 0 and final_len > 0:
        var i16_ptr = Pointer[Int16, origin=MutAnyOrigin](unsafe_from_address=final_ptr)
        for i in range(final_len):
            out.append(Float32(i16_ptr[unsafe_offset=i]) / Float32(32768.0))
    if final_ptr != 0:
        external_call["free", NoneType](final_ptr)
    external_call["free", NoneType](state)
    return (out^, cached_sr)


def _synth_klatt_fallback_mono(
    text: String,
    base_f0: Float32 = 112.0,
    formant_scale: Float32 = 1.0,
    char_dur_sec: Float32 = 0.056,
) -> List[Float32]:
    """Pure-Mojo 3-formant (`F1`/`F2`/`F3`) Klatt fallback synthesizer (`24 kHz` mono)
    with configurable vocal-tract `formant_scale` and `base_f0` pitch."""
    var sr = Float32(24000.0)
    var n_chars = max(4, min(96, text.byte_length()))
    var char_dur = max(Float32(0.038), min(Float32(0.075), char_dur_sec))
    var total = Int(Float32(n_chars) * char_dur * sr)
    var out = List[Float32](capacity=total)

    var f1 = CytomicSVF(460.0 * formant_scale, 4.5, sr)
    var f2 = CytomicSVF(1420.0 * formant_scale, 5.5, sr)
    var f3 = CytomicSVF(2420.0 * formant_scale, 6.0, sr)
    var fric_hp = CytomicSVF(3400.0, 2.2, sr)
    var pink = PinkNoise(719 + Int(base_f0 * 3.0))
    var ph = Float32(0.0)
    var prev_idx = -1
    var voiced = True

    for i in range(total):
        var t = Float32(i) / sr
        var idx = min(n_chars - 1, Int(t / char_dur))
        var local_t = t - Float32(idx) * char_dur
        if idx != prev_idx:
            prev_idx = idx
            var ch = Int(text.as_bytes()[idx]) if idx < text.byte_length() else 32
            if ch >= 97 and ch <= 122:
                ch -= 32
            voiced = ch != 32 and ch != 44 and ch != 46 and ch != 83 and ch != 70 and ch != 84 and ch != 75 and ch != 80
            var t_f1 = Float32(460.0)
            var t_f2 = Float32(1420.0)
            var t_f3 = Float32(2420.0)
            if ch == 65:  # A
                t_f1 = 680.0
                t_f2 = 1180.0
            elif ch == 69 or ch == 73 or ch == 89:  # E / I / Y
                t_f1 = 330.0
                t_f2 = 1980.0
            elif ch == 79 or ch == 85 or ch == 87:  # O / U / W
                t_f1 = 380.0
                t_f2 = 880.0
            f1.set_params(min(Float32(1150.0), t_f1 * formant_scale), Float32(4.8), sr)
            f2.set_params(min(Float32(3100.0), t_f2 * formant_scale), Float32(5.8), sr)
            f3.set_params(min(Float32(3900.0), t_f3 * formant_scale), Float32(6.2), sr)

        var ch_cur = Int(text.as_bytes()[idx]) if idx < text.byte_length() else 32
        if ch_cur == 32 or ch_cur == 44 or ch_cur == 46:
            out.append(Float32(0.0))
            continue

        var prosody = Float32(1.0) + Float32(0.035) * sin(t * Float32(4.2)) - (t / max(Float32(0.4), Float32(total) / sr)) * Float32(0.05)
        var f0 = base_f0 * prosody
        var dt_p = f0 / sr
        ph = wrap_phase01(ph + dt_p)
        var env = min(Float32(1.0), local_t * Float32(140.0)) * min(Float32(1.0), (char_dur - local_t) * Float32(95.0))
        if voiced:
            var glot = poly_blep_saw(ph, dt_p) * Float32(0.78) + pink.next() * Float32(0.06)
            _ = f1.process(glot)
            _ = f2.process(glot)
            _ = f3.process(glot)
            out.append((f1.bp * Float32(0.74) + f2.bp * Float32(0.48) + f3.bp * Float32(0.24)) * env)
        else:
            _ = fric_hp.process(hash_white_noise(i * 13 + 7))
            out.append(fric_hp.hp * Float32(0.25) * env)
    return out^


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
    var raw_res = _try_espeak_mono(text, voice_name, rate_wpm, pitch_0_99, range_0_99)
    var mono = raw_res[0].copy()
    var src_sr = raw_res[1]
    if len(mono) < 64 or src_sr <= 0:
        mono = _synth_klatt_fallback_mono(text)
        src_sr = 24000

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
    fundamental pitch, `espeak-ng` voice variant, military UHF `320–3350 Hz` bandpass, PTT click,
    and two-tone tactical squelch tail."""
    if profile_id == VOICE_AI_OS:
        return synth_ai_copilot_clip(text, lead_in_sec)

    var v_name = String("en-us+m2")
    var rate_wpm = 150
    var pitch_0_99 = 32
    var range_0_99 = 46
    var klatt_f0 = Float32(110.0)
    var klatt_formant = Float32(0.96)
    var klatt_dur = Float32(0.055)
    var pitch_shift = Float32(1.0)
    var pan_bias = Float32(0.0)
    var roger_hi = Float32(1174.66)
    var roger_lo = Float32(880.0)

    if profile_id == VOICE_BRIDGE_FEMALE:
        v_name = "en-us+f3"
        rate_wpm = 160
        pitch_0_99 = 66
        range_0_99 = 52
        klatt_f0 = 214.0
        klatt_formant = 1.20
        klatt_dur = 0.050
        pitch_shift = 1.04
        pan_bias = -0.10
        roger_hi = 1318.51
        roger_lo = 987.77
    elif profile_id == VOICE_ACE_BARITONE:
        v_name = "en-us+m2"
        rate_wpm = 145
        pitch_0_99 = 26
        range_0_99 = 44
        klatt_f0 = 104.0
        klatt_formant = 0.94
        klatt_dur = 0.058
        pitch_shift = 0.95
        pan_bias = -0.16
        roger_hi = 1046.50
        roger_lo = 783.99
    elif profile_id == VOICE_ACE_TENOR:
        v_name = "en-us+m4"
        rate_wpm = 166
        pitch_0_99 = 48
        range_0_99 = 55
        klatt_f0 = 144.0
        klatt_formant = 1.06
        klatt_dur = 0.048
        pitch_shift = 1.03
        pan_bias = 0.16
        roger_hi = 1174.66
        roger_lo = 987.77
    elif profile_id == VOICE_ACE_CRIMSON:
        v_name = "en-gb-x-rp+f2"
        rate_wpm = 156
        pitch_0_99 = 58
        range_0_99 = 58
        klatt_f0 = 192.0
        klatt_formant = 1.14
        klatt_dur = 0.052
        pitch_shift = 1.02
        pan_bias = 0.20
        roger_hi = 1396.91
        roger_lo = 698.46

    var raw_res = _try_espeak_mono(text, v_name, rate_wpm, pitch_0_99, range_0_99)
    var mono = raw_res[0].copy()
    var src_sr = raw_res[1]
    if len(mono) < 64 or src_sr <= 0:
        mono = _synth_klatt_fallback_mono(text, klatt_f0, klatt_formant, klatt_dur)
        src_sr = 24000

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
