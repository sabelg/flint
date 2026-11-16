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
from std.math import sin, cos, tan, exp, sqrt, floor
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
    if x <= Float32(-3.0):
        return Float32(-1.0)
    if x >= Float32(3.0):
        return Float32(1.0)
    var x2 = x * x
    return x * (Float32(27.0) + x2) / (Float32(27.0) + Float32(9.0) * x2)


@always_inline
def tube_saturate(x: Float32, drive: Float32 = 1.5) -> Float32:
    """Asymmetric warm even/odd-harmonic saturation for punchy drums, bass & explosions."""
    var d = x * drive
    var asym = d + Float32(0.12) * d * d
    return soft_clip_tanh(asym) - Float32(0.042)


@always_inline
def wrap_phase01(p: Float32) -> Float32:
    return p - floor(p)


@always_inline
def poly_blep(t: Float32, dt: Float32) -> Float32:
    """2-point polynomial band-limited step residual at `t == 0` / `t == 1`."""
    if dt <= Float32(1.0e-7):
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
    var t = wrap_phase01(phase01)
    var naive = Float32(2.0) * t - Float32(1.0)
    return naive - poly_blep(t, dt)


@always_inline
def poly_blep_pulse(phase01: Float32, dt: Float32, pulse_width: Float32 = 0.5) -> Float32:
    """Alias-suppressed variable-width pulse wave in `[-1, +1]`."""
    var t = wrap_phase01(phase01)
    var pw = max(Float32(0.05), min(Float32(0.95), pulse_width))
    var naive = Float32(1.0) if t < pw else Float32(-1.0)
    var t2 = wrap_phase01(t - pw + Float32(1.0))
    return naive + poly_blep(t, dt) - poly_blep(t2, dt)


@always_inline
def equal_power_gains(pan: Float32) -> Tuple[Float32, Float32]:
    """Constant-power stereo pan law (`pan` in `[-1.0, +1.0]`, center -> `(0.7071, 0.7071)`)."""
    var p = max(Float32(-1.0), min(Float32(1.0), pan))
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
        var fc = max(Float32(16.0), min(sample_rate * Float32(0.475), cutoff_hz))
        var qq = max(Float32(0.35), min(Float32(28.0), q))
        var g = tan(DSP_PI * fc / sample_rate)
        self.k = Float32(1.0) / qq
        self.a1 = Float32(1.0) / (Float32(1.0) + g * (g + self.k))
        self.a2 = g * self.a1
        self.a3 = g * self.a2

    @always_inline
    def process(mut self, v0: Float32) -> Float32:
        var v3 = v0 - self.ic2eq
        var v1 = self.a1 * self.ic1eq + self.a2 * v3
        var v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3
        self.ic1eq = Float32(2.0) * v1 - self.ic1eq
        self.ic2eq = Float32(2.0) * v2 - self.ic2eq
        self.lp = v2
        self.bp = v1
        self.hp = v0 - self.k * v1 - v2
        self.notch = v0 - self.k * v1
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
        var f = max(Float32(24.0), min(sample_rate * Float32(0.46), freq_hz))
        var tau = max(Float32(0.004), decay_sec)
        var r = exp(Float32(-1.0) / (tau * sample_rate))
        self.c1 = Float32(2.0) * r * cos(DSP_TAU * f / sample_rate)
        self.c2 = r * r
        self.gain = gain * (Float32(1.0) - r) * Float32(6.5)

    @always_inline
    def step(mut self, excitation: Float32) -> Float32:
        var y0 = self.c1 * self.y1 - self.c2 * self.y2 + self.gain * excitation
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
        # Warm felt-mallet contact smoothing (~1.2 kHz one-pole) before exciting kettle membrane
        self.mallet_lp += (felt_strike - self.mallet_lp) * Float32(0.145)
        var acc = Float32(0.0)
        for i in range(5):
            acc += self.modes[i].step(self.mallet_lp)
        return acc


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
        self.lfo_inc = rate_hz / sample_rate

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
        self.buf_l[self.write_idx] = in_l
        self.buf_r[self.write_idx] = in_r
        var dl = Float32(520.0) + sin(self.lfo_phase * DSP_TAU) * Float32(195.0)
        var dr = Float32(560.0) + cos(self.lfo_phase * DSP_TAU) * Float32(195.0)
        var wet_l = self._read_interp(self.buf_l, dl)
        var wet_r = self._read_interp(self.buf_r, dr)
        self.write_idx = (self.write_idx + 1) & self.mask
        self.lfo_phase = wrap_phase01(self.lfo_phase + self.lfo_inc)
        var dry = Float32(1.0) - mix * Float32(0.45)
        return (in_l * dry + wet_l * mix, in_r * dry + wet_r * mix)


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
        self.feedback = max(Float32(0.0), min(Float32(0.94), feedback))
        self.damp = max(Float32(0.05), min(Float32(0.95), damp))

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
        self.line0[self.idx0] = send_l * Float32(0.5) + (self.lp0 - half_sum) * fb
        self.line1[self.idx1] = send_r * Float32(0.5) + (self.lp1 - half_sum) * fb
        self.line2[self.idx2] = (send_l - send_r) * Float32(0.35) + (self.lp2 - half_sum) * fb
        self.line3[self.idx3] = (send_l + send_r) * Float32(0.35) + (self.lp3 - half_sum) * fb

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
) -> SpatialVoiceParams:
    """Computes 3D inverse-distance gain, equal-power stereo pan, radial-velocity Doppler
    pitch shift, atmospheric HF air-absorption LPF coefficient, and distance reverb send."""
    var out = SpatialVoiceParams()
    var rel = source_pos - listener_pos
    var dist = sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z)
    if dist < Float32(0.05):
        out.gain = Float32(1.0)
        out.pan = Float32(0.0)
        out.doppler_pitch = Float32(1.0)
        out.lpf_coeff = Float32(1.0)
        out.reverb_send = Float32(0.10)
        return out^

    var inv_d = Float32(1.0) / dist
    var dir = Vec3(rel.x * inv_d, rel.y * inv_d, rel.z * inv_d)

    # 1. Smooth clamped inverse-distance rolloff + max_dist horizon fade
    var rd = max(Float32(1.0), ref_dist)
    var md = max(rd + Float32(10.0), max_dist)
    var atten = rd / (rd + Float32(0.85) * max(Float32(0.0), dist - rd))
    var horizon = max(Float32(0.0), Float32(1.0) - max(Float32(0.0), dist - md * Float32(0.75)) / (md * Float32(0.25)))
    out.gain = max(Float32(0.0), min(Float32(1.15), atten * horizon))

    # 2. Azimuth stereo pan along listener's right ear vector (softened at point-blank range)
    var near_spread = min(Float32(1.0), dist / max(Float32(2.0), rd * Float32(0.45)))
    var right_dot = dir.x * listener_right.x + dir.y * listener_right.y + dir.z * listener_right.z
    out.pan = max(Float32(-0.95), min(Float32(0.95), right_dot * near_spread))

    # 3. Radial-velocity acoustic Doppler shift: f' = f * (c + v_listener_toward) / (c + v_source_away)
    var v_l_rad = listener_vel.x * dir.x + listener_vel.y * dir.y + listener_vel.z * dir.z
    var v_s_rad = source_vel.x * dir.x + source_vel.y * dir.y + source_vel.z * dir.z
    var c = SPEED_OF_SOUND
    var num = max(Float32(60.0), c + v_l_rad * doppler_scale)
    var den = max(Float32(60.0), c + v_s_rad * doppler_scale)
    out.doppler_pitch = max(Float32(0.48), min(Float32(2.15), num / den))

    # 4. Atmospheric HF air-absorption LPF (1.0 close -> 0.22 far) & distance wet-reverb bloom
    var dist_norm = max(Float32(0.0), min(Float32(1.0), (dist - rd) / (md - rd)))
    out.lpf_coeff = Float32(1.0) - dist_norm * Float32(0.76)
    out.reverb_send = Float32(0.10) + dist_norm * Float32(0.28)
    return out^
