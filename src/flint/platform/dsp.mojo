"""Pure-Mojo Studio Audio Synthesis, Physical Modeling, FDN Reverb & 3D Doppler DSP (`flint.platform.dsp`).

Engine-generic real-time and offline audio DSP building blocks with zero external
dependencies:
  - Band-limited `PolyBLEP` oscillators (saw, variable-width pulse, triangle)
  - Paul Kellet `-3 dB/oct` `PinkNoise` generator & deterministic hash white noise
  - Andrew Simper `CytomicSVF` zero-delay-feedback trapezoidal state-variable filter
  - Physical-modeling `ModalResonator` & `ModalBank6` for metallic/mechanical impacts
  - Padé `soft_clip_tanh` & asymmetric `tube_saturate` harmonic waveshapers
  - Quadrature-LFO `StereoChorus` & 4x4 Householder `StereoReverbFDN`
  - 3D Listener/Emitter equal-power panning, air-absorption LPF & radial Doppler shift
"""
from std.math import sin, cos, tan, exp, sqrt, floor, isfinite, isnan
from flint.core.math import Vec3

comptime DSP_PI: Float32 = 3.141592653589793
comptime DSP_TAU: Float32 = 6.283185307179586
comptime SPEED_OF_SOUND: Float32 = 343.0


@always_inline
def hash_white_noise(seed: Int) -> Float32:
    """Stateless PCG-style hash noise in `[-1.0, +1.0]`."""
    var x = UInt32(seed * 747796405 + 2891336453)
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737
    x = (x >> 22) ^ x
    return (Float32(x & 0xFFFF) / Float32(32767.5)) - Float32(1.0)


@always_inline
def soft_clip_tanh(x: Float32) -> Float32:
    """Fast C2-continuous rational Padé `tanh(x)` soft clipper in `[-1.0, +1.0]`."""
    if isnan(x):
        return Float32(0.0)
    if x <= Float32(-3.0):
        return Float32(-1.0)
    if x >= Float32(3.0):
        return Float32(1.0)
    var x2 = x * x
    return x * (Float32(27.0) + x2) / (Float32(27.0) + Float32(9.0) * x2)


@always_inline
def tube_saturate(x: Float32, drive: Float32 = 1.5) -> Float32:
    """Asymmetric warm even/odd-harmonic saturation for punchy drums, bass & explosions."""
    if not isfinite(x) or not isfinite(drive):
        return Float32(0.0)
    var d = x * drive
    var asym = d + Float32(0.12) * d * d
    return soft_clip_tanh(asym) - Float32(0.042)


@always_inline
def wrap_phase01(p: Float32) -> Float32:
    if not isfinite(p):
        return Float32(0.0)
    return p - floor(p)


@always_inline
def poly_blep(t: Float32, dt: Float32) -> Float32:
    """2-point polynomial band-limited step residual at `t == 0` / `t == 1`."""
    if not isfinite(t) or not isfinite(dt) or dt <= Float32(1.0e-7):
        return Float32(0.0)
    if t < dt:
        var u = t / dt
        return u + u - u * u - Float32(1.0)
    if t > Float32(1.0) - dt:
        var u = (t - Float32(1.0)) / dt
        return u * u + u + u + Float32(1.0)
    return Float32(0.0)


@always_inline
def poly_blep_saw(phase01: Float32, dt: Float32) -> Float32:
    """Alias-suppressed sawtooth wave in `[-1, +1]` for `phase01` in `[0, 1)`."""
    if not isfinite(phase01):
        return Float32(0.0)
    var t = wrap_phase01(phase01)
    var naive = Float32(2.0) * t - Float32(1.0)
    return naive - poly_blep(t, dt)


@always_inline
def poly_blep_pulse(phase01: Float32, dt: Float32, pulse_width: Float32 = 0.5) -> Float32:
    """Alias-suppressed variable-width pulse wave in `[-1, +1]`."""
    if not isfinite(phase01):
        return Float32(0.0)
    var t = wrap_phase01(phase01)
    var pw_in = pulse_width if isfinite(pulse_width) else Float32(0.5)
    var pw = max(Float32(0.05), min(Float32(0.95), pw_in))
    var naive = Float32(1.0) if t < pw else Float32(-1.0)
    var t2 = wrap_phase01(t - pw + Float32(1.0))
    return naive + poly_blep(t, dt) - poly_blep(t2, dt)


@always_inline
def equal_power_gains(pan: Float32) -> Tuple[Float32, Float32]:
    """Constant-power stereo pan law (`pan` in `[-1.0, +1.0]`, center -> `(0.7071, 0.7071)`)."""
    var p_in = pan if isfinite(pan) else Float32(0.0)
    var p = max(Float32(-1.0), min(Float32(1.0), p_in))
    var angle = (p + Float32(1.0)) * (DSP_PI * Float32(0.25))
    return (cos(angle), sin(angle))


struct PinkNoise(Copyable, ImplicitlyCopyable, Movable):
    """Paul Kellet's 7-state `-3 dB/octave` pink noise filter."""
    var b0: Float32
    var b1: Float32
    var b2: Float32
    var b3: Float32
    var b4: Float32
    var b5: Float32
    var b6: Float32
    var counter: Int

    def __init__(out self, seed: Int = 1337):
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.b3 = 0.0
        self.b4 = 0.0
        self.b5 = 0.0
        self.b6 = 0.0
        self.counter = seed

    @always_inline
    def next(mut self) -> Float32:
        var white = hash_white_noise(self.counter)
        self.counter += 1
        self.b0 = Float32(0.99886) * self.b0 + white * Float32(0.0555179)
        self.b1 = Float32(0.99332) * self.b1 + white * Float32(0.0750759)
        self.b2 = Float32(0.96900) * self.b2 + white * Float32(0.1538520)
        self.b3 = Float32(0.86650) * self.b3 + white * Float32(0.3104856)
        self.b4 = Float32(0.55000) * self.b4 + white * Float32(0.5329522)
        self.b5 = Float32(-0.7616) * self.b5 - white * Float32(0.0168980)
        var pink = self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6 + white * Float32(0.5362)
        self.b6 = white * Float32(0.115926)
        return max(Float32(-1.0), min(Float32(1.0), pink * Float32(0.18)))


struct CytomicSVF(Copyable, ImplicitlyCopyable, Movable):
    """Andrew Simper's zero-delay-feedback trapezoidal State Variable Filter.
    Simultaneously exposes `lp`, `bp`, `hp`, and `notch` with unconditional stability
    under rapid audio-rate cutoff & resonance modulation."""
    var ic1eq: Float32
    var ic2eq: Float32
    var a1: Float32
    var a2: Float32
    var a3: Float32
    var k: Float32
    var lp: Float32
    var bp: Float32
    var hp: Float32
    var notch: Float32

    def __init__(out self, cutoff_hz: Float32 = 1200.0, q: Float32 = 0.707, sample_rate: Float32 = 48000.0):
        self.ic1eq = 0.0
        self.ic2eq = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.k = 1.414
        self.lp = 0.0
        self.bp = 0.0
        self.hp = 0.0
        self.notch = 0.0
        self.set_params(cutoff_hz, q, sample_rate)

    @always_inline
    def set_params(mut self, cutoff_hz: Float32, q: Float32, sample_rate: Float32 = 48000.0):
        var sr_in = sample_rate if (isfinite(sample_rate) and sample_rate > Float32(0.0)) else Float32(48000.0)
        var sr = max(Float32(1000.0), sr_in)
        var fc_in = cutoff_hz if isfinite(cutoff_hz) else Float32(1200.0)
        var fc = max(Float32(16.0), min(sr * Float32(0.475), fc_in))
        var q_in = q if isfinite(q) else Float32(0.707)
        var qq = max(Float32(0.35), min(Float32(28.0), q_in))
        var g = tan(DSP_PI * fc / sr)
        self.k = Float32(1.0) / qq
        self.a1 = Float32(1.0) / (Float32(1.0) + g * (g + self.k))
        self.a2 = g * self.a1
        self.a3 = g * self.a2

    @always_inline
    def process(mut self, v0: Float32) -> Float32:
        var x = v0 if isfinite(v0) else Float32(0.0)
        var v3 = x - self.ic2eq
        var v1 = self.a1 * self.ic1eq + self.a2 * v3
        var v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3
        var next_ic1 = Float32(2.0) * v1 - self.ic1eq
        var next_ic2 = Float32(2.0) * v2 - self.ic2eq
        if not (isfinite(v1) and isfinite(v2) and isfinite(next_ic1) and isfinite(next_ic2)):
            self.ic1eq = Float32(0.0)
            self.ic2eq = Float32(0.0)
            self.lp = Float32(0.0)
            self.bp = Float32(0.0)
            self.hp = Float32(0.0)
            self.notch = Float32(0.0)
            return Float32(0.0)
        self.ic1eq = next_ic1
        self.ic2eq = next_ic2
        self.lp = v2
        self.bp = v1
        self.hp = x - self.k * v1 - v2
        self.notch = x - self.k * v1
        return v2


struct ModalResonator(Copyable, ImplicitlyCopyable, Movable):
    """2-pole damped harmonic oscillator for physical-modeling metallic & mechanical impacts."""
    var c1: Float32
    var c2: Float32
    var gain: Float32
    var y1: Float32
    var y2: Float32

    def __init__(out self, freq_hz: Float32 = 440.0, decay_sec: Float32 = 0.15, gain: Float32 = 1.0, sample_rate: Float32 = 48000.0):
        self.c1 = 0.0
        self.c2 = 0.0
        self.gain = gain
        self.y1 = 0.0
        self.y2 = 0.0
        self.configure(freq_hz, decay_sec, gain, sample_rate)

    @always_inline
    def configure(mut self, freq_hz: Float32, decay_sec: Float32, gain: Float32, sample_rate: Float32 = 48000.0):
        var sr_in = sample_rate if isfinite(sample_rate) else Float32(48000.0)
        var sr = max(Float32(1000.0), sr_in)
        var f_in = freq_hz if isfinite(freq_hz) else Float32(440.0)
        var f = max(Float32(24.0), min(sr * Float32(0.46), f_in))
        var tau_in = decay_sec if isfinite(decay_sec) else Float32(0.15)
        var tau = max(Float32(0.004), tau_in)
        var r = exp(Float32(-1.0) / (tau * sr))
        self.c1 = Float32(2.0) * r * cos(DSP_TAU * f / sr)
        self.c2 = r * r
        var g_in = gain if isfinite(gain) else Float32(0.0)
        var g_clamped = max(Float32(-100.0), min(Float32(100.0), g_in))
        self.gain = g_clamped * (Float32(1.0) - r) * Float32(6.5)

    @always_inline
    def step(mut self, excitation: Float32) -> Float32:
        var ex = excitation if isfinite(excitation) else Float32(0.0)
        var y0 = self.c1 * self.y1 - self.c2 * self.y2 + self.gain * ex
        if not isfinite(y0):
            y0 = Float32(0.0)
        self.y2 = self.y1
        self.y1 = y0
        return y0


struct ModalBank6(Movable):
    """6-mode inharmonic titanium/steel airframe plate & hydraulic latch resonator bank."""
    var modes: List[ModalResonator]

    def __init__(out self, base_hz: Float32, decay_sec: Float32, sample_rate: Float32 = 48000.0):
        self.modes = List[ModalResonator](capacity=6)
        var ratios = [Float32(1.000), Float32(1.583), Float32(2.241), Float32(2.917), Float32(3.762), Float32(4.614)]
        var gains = [Float32(1.00), Float32(0.76), Float32(0.58), Float32(0.42), Float32(0.30), Float32(0.22)]
        var decays = [Float32(1.00), Float32(0.84), Float32(0.68), Float32(0.54), Float32(0.42), Float32(0.33)]
        for i in range(6):
            self.modes.append(
                ModalResonator(
                    base_hz * ratios[i],
                    decay_sec * decays[i],
                    gains[i],
                    sample_rate,
                )
            )

    def retune(mut self, base_hz: Float32, decay_sec: Float32, sample_rate: Float32 = 48000.0):
        var ratios = [Float32(1.000), Float32(1.583), Float32(2.241), Float32(2.917), Float32(3.762), Float32(4.614)]
        var gains = [Float32(1.00), Float32(0.76), Float32(0.58), Float32(0.42), Float32(0.30), Float32(0.22)]
        var decays = [Float32(1.00), Float32(0.84), Float32(0.68), Float32(0.54), Float32(0.42), Float32(0.33)]
        for i in range(6):
            self.modes[i].configure(base_hz * ratios[i], decay_sec * decays[i], gains[i], sample_rate)

    @always_inline
    def process(mut self, excitation: Float32) -> Float32:
        var acc = Float32(0.0)
        for i in range(6):
            acc += self.modes[i].step(excitation)
        return acc


struct TimpaniDrum(Movable):
    """5-mode air-coupled circular Bessel kettle-drum membrane (`[1.000, 1.504, 1.742, 2.000, 2.295]`)
    for tuned orchestral concert timpani strikes & crescendo rolls (`D2`, `G1`, `A1`, `Bb1`, `C2`)."""
    var modes: List[ModalResonator]
    var mallet_lp: Float32

    def __init__(out self, fundamental_hz: Float32 = 73.42, decay_sec: Float32 = 0.62, sample_rate: Float32 = 48000.0):
        self.modes = List[ModalResonator](capacity=5)
        self.mallet_lp = 0.0
        var ratios = [Float32(1.000), Float32(1.504), Float32(1.742), Float32(2.000), Float32(2.295)]
        var gains = [Float32(1.15), Float32(0.72), Float32(0.46), Float32(0.38), Float32(0.24)]
        var decays = [Float32(1.00), Float32(0.86), Float32(0.64), Float32(0.72), Float32(0.48)]
        for i in range(5):
            self.modes.append(
                ModalResonator(
                    fundamental_hz * ratios[i],
                    decay_sec * decays[i],
                    gains[i],
                    sample_rate,
                )
            )

    def retune(mut self, fundamental_hz: Float32, decay_sec: Float32 = 0.62, sample_rate: Float32 = 48000.0):
        var ratios = [Float32(1.000), Float32(1.504), Float32(1.742), Float32(2.000), Float32(2.295)]
        var gains = [Float32(1.15), Float32(0.72), Float32(0.46), Float32(0.38), Float32(0.24)]
        var decays = [Float32(1.00), Float32(0.86), Float32(0.64), Float32(0.72), Float32(0.48)]
        for i in range(5):
            self.modes[i].configure(fundamental_hz * ratios[i], decay_sec * decays[i], gains[i], sample_rate)

    @always_inline
    def process(mut self, felt_strike: Float32) -> Float32:
        var strike = felt_strike if isfinite(felt_strike) else Float32(0.0)
        # Warm felt-mallet contact smoothing (~1.2 kHz one-pole) before exciting kettle membrane
        var next_lp = self.mallet_lp + (strike - self.mallet_lp) * Float32(0.145)
        self.mallet_lp = next_lp if isfinite(next_lp) else Float32(0.0)
        var acc = Float32(0.0)
        for i in range(5):
            acc += self.modes[i].step(self.mallet_lp)
        return acc if isfinite(acc) else Float32(0.0)


struct StereoChorus(Movable):
    """Dual quadrature-modulated fractional delay chorus for wide 1980s stereo brass & synths."""
    var buf_l: List[Float32]
    var buf_r: List[Float32]
    var mask: Int
    var write_idx: Int
    var lfo_phase: Float32
    var lfo_inc: Float32

    def __init__(out self, rate_hz: Float32 = 0.85, sample_rate: Float32 = 48000.0):
        self.buf_l = List[Float32](length=2048, fill=0.0)
        self.buf_r = List[Float32](length=2048, fill=0.0)
        self.mask = 2047
        self.write_idx = 0
        self.lfo_phase = 0.0
        var r_in = rate_hz if isfinite(rate_hz) else Float32(0.85)
        var sr_in = sample_rate if (isfinite(sample_rate) and sample_rate > Float32(0.0)) else Float32(48000.0)
        self.lfo_inc = max(Float32(0.0), r_in) / sr_in

    @always_inline
    def _read_interp(self, buf: List[Float32], delay_samples: Float32) -> Float32:
        var pos = Float32(self.write_idx) - delay_samples
        while pos < Float32(0.0):
            pos += Float32(2048.0)
        var idx0 = Int(floor(pos)) & self.mask
        var idx1 = (idx0 + 1) & self.mask
        var frac = pos - floor(pos)
        return buf[idx0] + (buf[idx1] - buf[idx0]) * frac

    @always_inline
    def process(mut self, in_l: Float32, in_r: Float32, mix: Float32 = 0.32) -> Tuple[Float32, Float32]:
        var sl = in_l if isfinite(in_l) else Float32(0.0)
        var sr = in_r if isfinite(in_r) else Float32(0.0)
        var m_in = mix if isfinite(mix) else Float32(0.0)
        var m = max(Float32(0.0), min(Float32(1.0), m_in))
        self.buf_l[self.write_idx] = sl
        self.buf_r[self.write_idx] = sr
        var dl = Float32(520.0) + sin(self.lfo_phase * DSP_TAU) * Float32(195.0)
        var dr = Float32(560.0) + cos(self.lfo_phase * DSP_TAU) * Float32(195.0)
        var wet_l = self._read_interp(self.buf_l, dl)
        var wet_r = self._read_interp(self.buf_r, dr)
        self.write_idx = (self.write_idx + 1) & self.mask
        self.lfo_phase = wrap_phase01(self.lfo_phase + self.lfo_inc)
        if not isfinite(self.lfo_phase):
            self.lfo_phase = Float32(0.0)
        var dry = Float32(1.0) - m * Float32(0.45)
        var out_l = sl * dry + wet_l * m
        var out_r = sr * dry + wet_r * m
        if not isfinite(out_l):
            out_l = Float32(0.0)
        if not isfinite(out_r):
            out_r = Float32(0.0)
        return (out_l, out_r)


struct StereoReverbFDN(Movable):
    """4x4 Unitary Householder Feedback Delay Network stereo reverb with mutually-prime
    delay lines (1499, 1877, 2237, 2683 frames @ 48 kHz) and HF air-absorption damping."""
    var line0: List[Float32]
    var line1: List[Float32]
    var line2: List[Float32]
    var line3: List[Float32]
    var idx0: Int
    var idx1: Int
    var idx2: Int
    var idx3: Int
    var lp0: Float32
    var lp1: Float32
    var lp2: Float32
    var lp3: Float32
    var feedback: Float32
    var damp: Float32

    def __init__(out self, feedback: Float32 = 0.76, damp: Float32 = 0.34):
        self.line0 = List[Float32](length=1499, fill=0.0)
        self.line1 = List[Float32](length=1877, fill=0.0)
        self.line2 = List[Float32](length=2237, fill=0.0)
        self.line3 = List[Float32](length=2683, fill=0.0)
        self.idx0 = 0
        self.idx1 = 0
        self.idx2 = 0
        self.idx3 = 0
        self.lp0 = 0.0
        self.lp1 = 0.0
        self.lp2 = 0.0
        self.lp3 = 0.0
        var fb_in = feedback if isfinite(feedback) else Float32(0.76)
        var dp_in = damp if isfinite(damp) else Float32(0.34)
        self.feedback = max(Float32(0.0), min(Float32(0.94), fb_in))
        self.damp = max(Float32(0.05), min(Float32(0.95), dp_in))

    def clear(mut self):
        for i in range(len(self.line0)):
            self.line0[i] = 0.0
        for i in range(len(self.line1)):
            self.line1[i] = 0.0
        for i in range(len(self.line2)):
            self.line2[i] = 0.0
        for i in range(len(self.line3)):
            self.line3[i] = 0.0
        self.lp0 = 0.0
        self.lp1 = 0.0
        self.lp2 = 0.0
        self.lp3 = 0.0

    @always_inline
    def process_send(mut self, send_l: Float32, send_r: Float32) -> Tuple[Float32, Float32]:
        var sl = send_l if isfinite(send_l) else Float32(0.0)
        var sr = send_r if isfinite(send_r) else Float32(0.0)
        var d0 = self.line0[self.idx0]
        var d1 = self.line1[self.idx1]
        var d2 = self.line2[self.idx2]
        var d3 = self.line3[self.idx3]

        # One-pole high-frequency absorption in feedback loop
        self.lp0 += (d0 - self.lp0) * self.damp
        self.lp1 += (d1 - self.lp1) * self.damp
        self.lp2 += (d2 - self.lp2) * self.damp
        self.lp3 += (d3 - self.lp3) * self.damp

        # Unitary 4x4 Householder reflection: H * x = x - 0.5 * sum(x)
        var half_sum = (self.lp0 + self.lp1 + self.lp2 + self.lp3) * Float32(0.5)
        var fb = self.feedback
        var n0 = sl * Float32(0.5) + (self.lp0 - half_sum) * fb
        var n1 = sr * Float32(0.5) + (self.lp1 - half_sum) * fb
        var n2 = (sl - sr) * Float32(0.35) + (self.lp2 - half_sum) * fb
        var n3 = (sl + sr) * Float32(0.35) + (self.lp3 - half_sum) * fb
        self.line0[self.idx0] = n0 if isfinite(n0) else Float32(0.0)
        self.line1[self.idx1] = n1 if isfinite(n1) else Float32(0.0)
        self.line2[self.idx2] = n2 if isfinite(n2) else Float32(0.0)
        self.line3[self.idx3] = n3 if isfinite(n3) else Float32(0.0)

        self.idx0 += 1
        if self.idx0 >= 1499:
            self.idx0 = 0
        self.idx1 += 1
        if self.idx1 >= 1877:
            self.idx1 = 0
        self.idx2 += 1
        if self.idx2 >= 2237:
            self.idx2 = 0
        self.idx3 += 1
        if self.idx3 >= 2683:
            self.idx3 = 0

        var wet_l = (d0 + d2 - d3 * Float32(0.35)) * Float32(0.45)
        var wet_r = (d1 - d2 * Float32(0.35) + d3) * Float32(0.45)
        if not isfinite(wet_l):
            wet_l = Float32(0.0)
        if not isfinite(wet_r):
            wet_r = Float32(0.0)
        return (wet_l, wet_r)


struct SpatialVoiceParams(Copyable, ImplicitlyCopyable, Movable):
    var gain: Float32
    var pan: Float32
    var doppler_pitch: Float32
    var lpf_coeff: Float32
    var reverb_send: Float32

    def __init__(out self):
        self.gain = 1.0
        self.pan = 0.0
        self.doppler_pitch = 1.0
        self.lpf_coeff = 1.0
        self.reverb_send = 0.14


def compute_spatial_3d(
    listener_pos: Vec3,
    listener_vel: Vec3,
    listener_right: Vec3,
    source_pos: Vec3,
    source_vel: Vec3,
    ref_dist: Float32 = 22.0,
    max_dist: Float32 = 420.0,
    doppler_scale: Float32 = 0.65,
    listener_fwd: Vec3 = Vec3(0, 0, 1),
) -> SpatialVoiceParams:
    """Computes 3D inverse-distance gain, equal-power stereo pan, radial-velocity Doppler
    pitch shift, atmospheric HF air-absorption LPF coefficient, and distance reverb send."""
    var out = SpatialVoiceParams()
    if not (
        isfinite(listener_pos.x) and isfinite(listener_pos.y) and isfinite(listener_pos.z)
        and isfinite(source_pos.x) and isfinite(source_pos.y) and isfinite(source_pos.z)
    ):
        out.gain = Float32(0.0)
        out.pan = Float32(0.0)
        out.doppler_pitch = Float32(1.0)
        out.lpf_coeff = Float32(1.0)
        out.reverb_send = Float32(0.0)
        return out^

    var rx = Float64(source_pos.x) - Float64(listener_pos.x)
    var ry = Float64(source_pos.y) - Float64(listener_pos.y)
    var rz = Float64(source_pos.z) - Float64(listener_pos.z)
    var dist64 = sqrt(rx * rx + ry * ry + rz * rz)
    if not isfinite(dist64):
        out.gain = Float32(0.0)
        out.pan = Float32(0.0)
        out.doppler_pitch = Float32(1.0)
        out.lpf_coeff = Float32(1.0)
        out.reverb_send = Float32(0.0)
        return out^
    if dist64 < 0.05:
        out.gain = Float32(1.0)
        out.pan = Float32(0.0)
        out.doppler_pitch = Float32(1.0)
        out.lpf_coeff = Float32(1.0)
        out.reverb_send = Float32(0.10)
        return out^

    var inv_d64 = 1.0 / dist64
    var dir = Vec3(Float32(rx * inv_d64), Float32(ry * inv_d64), Float32(rz * inv_d64))
    var dist = Float32(dist64) if isfinite(Float32(dist64)) else Float32(1.0e30)

    # 1. Smooth clamped inverse-distance rolloff + max_dist horizon fade
    var rd_in = ref_dist if isfinite(ref_dist) else Float32(22.0)
    var rd = max(Float32(1.0), rd_in)
    var md_in = max_dist if isfinite(max_dist) else Float32(420.0)
    var md = max(rd + Float32(10.0), md_in)
    var atten = rd / (rd + Float32(0.85) * max(Float32(0.0), dist - rd))
    var horizon = max(Float32(0.0), Float32(1.0) - max(Float32(0.0), dist - md * Float32(0.75)) / (md * Float32(0.25)))
    var gain_raw = atten * horizon
    out.gain = max(Float32(0.0), min(Float32(1.15), gain_raw)) if isfinite(gain_raw) else Float32(0.0)

    # 2. Azimuth stereo pan along listener's right ear vector (softened at point-blank range)
    var near_spread = min(Float32(1.0), dist / max(Float32(2.0), rd * Float32(0.45)))
    var r_ok = isfinite(listener_right.x) and isfinite(listener_right.y) and isfinite(listener_right.z)
    var right_v = listener_right if r_ok else Vec3(0.0, 0.0, 0.0)
    var right_dot = dir.x * right_v.x + dir.y * right_v.y + dir.z * right_v.z
    var pan_raw = right_dot * near_spread
    out.pan = max(Float32(-0.95), min(Float32(0.95), pan_raw)) if isfinite(pan_raw) else Float32(0.0)

    # 3. Radial-velocity acoustic Doppler shift: f' = f * (c + v_listener_toward) / (c + v_source_away)
    var lv_ok = isfinite(listener_vel.x) and isfinite(listener_vel.y) and isfinite(listener_vel.z)
    var sv_ok = isfinite(source_vel.x) and isfinite(source_vel.y) and isfinite(source_vel.z)
    var lv = listener_vel if lv_ok else Vec3(0.0, 0.0, 0.0)
    var sv = source_vel if sv_ok else Vec3(0.0, 0.0, 0.0)
    var dop_scale = doppler_scale if isfinite(doppler_scale) else Float32(0.65)
    var v_l_rad = lv.x * dir.x + lv.y * dir.y + lv.z * dir.z
    var v_s_rad = sv.x * dir.x + sv.y * dir.y + sv.z * dir.z
    var c = SPEED_OF_SOUND
    var num_raw = c + v_l_rad * dop_scale
    var den_raw = c + v_s_rad * dop_scale
    if isfinite(num_raw) and isfinite(den_raw):
        var num = max(Float32(60.0), num_raw)
        var den = max(Float32(60.0), den_raw)
        var ratio = num / den
        out.doppler_pitch = max(Float32(0.48), min(Float32(2.15), ratio)) if isfinite(ratio) else Float32(1.0)
    else:
        out.doppler_pitch = Float32(1.0)

    # 4. Atmospheric HF air-absorption LPF (1.0 close -> 0.22 far) & distance wet-reverb bloom
    var dist_norm_raw = (dist - rd) / (md - rd)
    var dist_norm = max(Float32(0.0), min(Float32(1.0), dist_norm_raw)) if isfinite(dist_norm_raw) else Float32(1.0)
    out.lpf_coeff = Float32(1.0) - dist_norm * Float32(0.76)
    out.reverb_send = Float32(0.10) + dist_norm * Float32(0.28)

    # 5. Rear-cone pinna/head-shadow occlusion filtering when listener_fwd is provided
    var fwd_ok = isfinite(listener_fwd.x) and isfinite(listener_fwd.y) and isfinite(listener_fwd.z)
    var fwd_v = listener_fwd if fwd_ok else Vec3(0.0, 0.0, 1.0)
    var fwd_dot = dir.x * fwd_v.x + dir.y * fwd_v.y + dir.z * fwd_v.z
    if isfinite(fwd_dot) and fwd_dot < Float32(0.0):
        var lpf_occ = out.lpf_coeff * (Float32(1.0) + fwd_dot * Float32(0.28))
        out.lpf_coeff = max(Float32(0.12), lpf_occ) if isfinite(lpf_occ) else Float32(0.12)

    return out^


struct ItdDelayLine(Copyable, Movable):
    """64-sample circular delay line with Hermite cubic fractional interpolation
    for transient-preserving continuous Woodworth interaural time difference (ITD)."""
    var buffer: List[Float32]
    var write_idx: Int

    def __init__(out self):
        self.buffer = List[Float32]()
        for _ in range(64):
            self.buffer.append(0.0)
        self.write_idx = 0

    def push(mut self, sample: Float32):
        self.buffer[self.write_idx] = sample if isfinite(sample) else Float32(0.0)
        self.write_idx = (self.write_idx + 1) & 63

    def read_delayed(self, delay_samples: Float32) -> Float32:
        var ds = delay_samples if isfinite(delay_samples) else Float32(0.0)
        var d = max(Float32(0.0), min(Float32(60.0), ds))
        var idelay = Int(floor(d))
        var frac = d - Float32(idelay)

        var idx1 = (self.write_idx - 1 - idelay) & 63
        var idx2 = (idx1 - 1) & 63
        var idx3 = (idx1 - 2) & 63

        var y1 = self.buffer[idx1]
        var y2 = self.buffer[idx2]
        var y3 = self.buffer[idx3]
        var y0 = (Float32(2.0) * y1 - y2) if idelay == 0 else self.buffer[(idx1 + 1) & 63]

        # Hermite cubic interpolation
        var c0 = y1
        var c1 = Float32(0.5) * (y2 - y0)
        var c2 = y0 - Float32(2.5) * y1 + Float32(2.0) * y2 - Float32(0.5) * y3
        var c3 = Float32(0.5) * (y3 - y0) + Float32(1.5) * (y1 - y2)
        var res = ((c3 * frac + c2) * frac + c1) * frac + c0
        return res if isfinite(res) else Float32(0.0)


struct VehicleHarmonicSynth(Movable):
    """Procedural multi-harmonic vehicle engine audio synthesizer using
    band-limited PolyBLEP oscillators, dual resonant SVF formants, and tube saturation."""
    var phase_cyl1: Float32
    var phase_cyl2: Float32
    var phase_cyl3: Float32
    var phase_cyl4: Float32
    var intake_svf: CytomicSVF
    var exhaust_svf: CytomicSVF
    var sample_rate: Float32

    def __init__(out self, sample_rate: Float32 = 48000.0):
        var sr_in = sample_rate if (isfinite(sample_rate) and sample_rate > Float32(0.0)) else Float32(48000.0)
        var sr = max(Float32(1000.0), sr_in)
        self.phase_cyl1 = 0.0
        self.phase_cyl2 = 0.25
        self.phase_cyl3 = 0.50
        self.phase_cyl4 = 0.75
        self.intake_svf = CytomicSVF(320.0, 1.4, sr)
        self.exhaust_svf = CytomicSVF(120.0, 1.8, sr)
        self.sample_rate = sr

    def step(mut self, rpm: Float32, load: Float32) -> Float32:
        var rpm_in = rpm if isfinite(rpm) else Float32(650.0)
        var load_in = load if isfinite(load) else Float32(0.0)
        var clamped_rpm = max(Float32(650.0), min(Float32(9500.0), rpm_in))
        var clamped_load = max(Float32(0.0), min(Float32(2.0), load_in))
        var f0 = (clamped_rpm / Float32(60.0)) * Float32(2.0)
        var dt = f0 / self.sample_rate

        var s1 = poly_blep_saw(self.phase_cyl1, dt)
        var s2 = poly_blep_saw(self.phase_cyl2, dt)
        var s3 = poly_blep_saw(self.phase_cyl3, dt)
        var s4 = poly_blep_saw(self.phase_cyl4, dt)

        self.phase_cyl1 = wrap_phase01(self.phase_cyl1 + dt)
        self.phase_cyl2 = wrap_phase01(self.phase_cyl2 + dt)
        self.phase_cyl3 = wrap_phase01(self.phase_cyl3 + dt)
        self.phase_cyl4 = wrap_phase01(self.phase_cyl4 + dt)

        var engine_raw = (s1 + s2 * Float32(0.8) + s3 * Float32(0.6) + s4 * Float32(0.4)) * Float32(0.35)
        var exh = self.exhaust_svf.process(engine_raw)
        var intk = self.intake_svf.process(engine_raw) * (clamped_load * Float32(0.5))
        var mixed = exh + intk
        return tube_saturate(mixed, Float32(1.2) + clamped_load * Float32(0.8))


struct GranularBiomeSynth(Movable):
    """Dynamic procedural granular synthesizer for biome ambient environmental textures (wind, foliage, atmosphere)."""
    var grain_phases: List[Float32]
    var grain_freqs: List[Float32]
    var grain_gains: List[Float32]
    var sample_rate: Float32
    var pink: PinkNoise

    def __init__(out self, sample_rate: Float32 = 48000.0):
        var sr_in = sample_rate if (isfinite(sample_rate) and sample_rate > Float32(0.0)) else Float32(48000.0)
        self.grain_phases = List[Float32]()
        self.grain_freqs = List[Float32]()
        self.grain_gains = List[Float32]()
        for i in range(8):
            self.grain_phases.append(Float32(i) * Float32(0.125))
            self.grain_freqs.append(Float32(120.0) + Float32(i) * Float32(45.0))
            self.grain_gains.append(Float32(0.12))
        self.sample_rate = max(Float32(1000.0), sr_in)
        self.pink = PinkNoise()

    def step(mut self, intensity: Float32) -> Float32:
        var inten_in = intensity if isfinite(intensity) else Float32(0.0)
        var inten = max(Float32(0.0), min(Float32(4.0), inten_in))
        var out_sample = Float32(0.0)
        var p_noise = self.pink.next() * inten
        for i in range(len(self.grain_phases)):
            var phase = self.grain_phases[i]
            var hann = Float32(0.5) * (Float32(1.0) - cos(DSP_TAU * phase))
            var osc = sin(DSP_TAU * phase * self.grain_freqs[i])
            out_sample += (osc * hann) * self.grain_gains[i]

            var dt = (Float32(2.0) + inten * Float32(6.0)) / self.sample_rate
            self.grain_phases[i] = wrap_phase01(phase + dt)

        return soft_clip_tanh(out_sample * Float32(0.8) + p_noise * Float32(0.25))
