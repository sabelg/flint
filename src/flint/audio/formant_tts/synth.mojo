"""Klatt cascade/parallel formant synthesiser at 24 kHz (`flint.audio.formant_tts.synth`).

Signal path (Klatt 1980 structure, KLSYN88 source controls, LF glottal flow):

    LF flow U (Rd table, pitch-synchronous latch, jitter / shimmer / flutter / diplophonia)
      -> TL one-pole tilt -> cascade  RTP RTZ RNP RNZ R6 R5 R4 R3 R2 R1(pitch-synchronous F1/B1)
      + AVS quasi-sinusoidal voicing (RGS low-pass of U)            (cascade input)
      + aspiration AH * LP(460 Hz) noise * glottal modulation       (cascade input)
    frication AF * noise (+ burst events) -> parallel R2..R7 (alternating signs) + AB bypass
    y = cascade + parallel -> radiation y[n] - y[n-1] -> 20 Hz DC block -> gain -> soft limiter

Parameters arrive as 5 ms frames (`KlattTrack`, 120 samples at 24 kHz) and are interpolated linearly
per sample; resonator coefficients are recomputed every 4 samples and ramped in between. Voicing
parameters latch at each glottal period start, where the LF flow is exactly zero, so voicing onsets,
offsets and F0 changes never click. Bursts come from a sample-accurate event list.

Sources: Klatt 1980 (JASA 67, resonator / anti-resonator equations, cascade/parallel split,
AVS / RGS, noise low-pass, 50 % noise modulation); Klatt & Klatt 1990 (JASA 87: TL, flutter,
diplophonia, DF1 / DB1, tracheal pair); Fant, Liljencrants & Lin 1985 and Fant 1995 (LF model and
the Rd regressions). Coefficients use `flint.core.detmath` so renders are bit-reproducible; the
LF table is built once at init in Float64.
"""
from std.math import exp, sin, cos, sqrt, floor
from flint.core.detmath import det_exp, det_cos, det_sin
from flint.platform.dsp import soft_clip_tanh

comptime KLATT_SR: Int = 24000
comptime KLATT_FRAME_SAMPLES: Int = 120     # 5 ms
comptime KLATT_SUB: Int = 4                 # coefficient update interval (6 kHz control rate)

# ---------------------------------------------------------------------------- frame layout
comptime P_F0: Int = 0
comptime P_AV: Int = 1
comptime P_AVS: Int = 2
comptime P_RD: Int = 3
comptime P_TL: Int = 4
comptime P_AH: Int = 5
comptime P_JIT: Int = 6
comptime P_SHIM: Int = 7
comptime P_FL: Int = 8
comptime P_DI: Int = 9
comptime P_F1: Int = 10
comptime P_B1: Int = 11
comptime P_DF1: Int = 12
comptime P_DB1: Int = 13
comptime P_F2: Int = 14
comptime P_B2: Int = 15
comptime P_F3: Int = 16
comptime P_B3: Int = 17
comptime P_F4: Int = 18
comptime P_B4: Int = 19
comptime P_F5: Int = 20
comptime P_B5: Int = 21
comptime P_F6: Int = 22
comptime P_B6: Int = 23
comptime P_FNP: Int = 24
comptime P_BNP: Int = 25
comptime P_FNZ: Int = 26
comptime P_BNZ: Int = 27
comptime P_FTP: Int = 28
comptime P_BTP: Int = 29
comptime P_FTZ: Int = 30
comptime P_BTZ: Int = 31
comptime P_AF: Int = 32
comptime P_A2: Int = 33
comptime P_A3: Int = 34
comptime P_A4: Int = 35
comptime P_A5: Int = 36
comptime P_A6: Int = 37
comptime P_A7: Int = 38
comptime P_AB: Int = 39
comptime P_B2P: Int = 40
comptime P_B3P: Int = 41
comptime P_B4P: Int = 42
comptime P_B5P: Int = 43
comptime P_B6P: Int = 44
comptime P_F7: Int = 45
comptime P_B7: Int = 46
comptime NPAR: Int = 47

# ---------------------------------------------------------------------------- calibration
# Branch scale factors on the dB -> linear map lin = K * 10^((dB - 60) / 20). They were set with the
# calibration render in tests/test_flint_v5_formant_tts.mojo (steady /ɑ/ at AV 60 and F0 110 Hz is
# -18 dBFS RMS; /h/ at AH 60 about 14 dB, /s/ at AF 60 about 10 dB below it).
comptime K_AV: Float32 = 1.0
comptime K_AVS: Float32 = 1.0
comptime K_AH: Float32 = 0.120
comptime K_AF: Float32 = 0.80
comptime K_AB: Float32 = 0.025
comptime K_OUT: Float32 = 0.25

# Return-phase scale on Fant's Ra: KLSYN88's modal source (KLGLOTT88 + TL 0..4 dB) is about
# -12 dB/oct in flow; Fant's Ra(Rd) adds a further first-order low-pass at 1/(2 pi Ta) that made the
# voice 10+ dB too dark at 2-3 kHz against natural speech (tuning log, iteration 2).
comptime LF_TA_SCALE: Float64 = 0.35
# Higher-pole correction (Fant 1960; Klatt 1980 §II.B): a uniform tract has a resonance about every
# 1 kHz, so a cascade truncated after F6 rolls off -12 dB/oct per missing pole and was 40-80 dB too
# dull above 5 kHz at 24 kHz against natural speech. Six fixed broad poles from 5.9 to 10.9 kHz
# (HPC_N = 6) restore that level, but cost 5-7 % WER on the STT eval (tuning log, iteration 4b), so
# the correction is off by default; HPC_N > 0 switches it on.
comptime HPC_N: Int = 0
comptime HPC_F0: Float32 = 5900.0     # first pole; spacing 1 kHz
comptime HPC_B0: Float32 = 1000.0     # first bandwidth; +100 Hz per pole
comptime LF_NRD: Int = 25        # Rd 0.3 .. 2.7, step 0.1
comptime LF_NPH: Int = 1024      # phase samples per period (+1 guard point)
comptime TWO_PI: Float64 = 6.283185307179586
comptime PI64: Float64 = 3.141592653589793


@always_inline
def db_lin(db: Float32) -> Float32:
    """Klatt dB (60 = strong vowel, <= 0 = off) to a linear gain of 1.0 at 60 dB."""
    if db <= Float32(0.5):
        return Float32(0.0)
    return det_exp((db - Float32(60.0)) * Float32(0.11512925464970229))


# ---------------------------------------------------------------------------- track
@fieldwise_init
struct KlattEvent(Copyable, Movable):
    """A sample-accurate stop burst on the parallel branch."""

    var sample: Int
    var amp_db: Float32
    var decay_ms: Float32
    var a2: Float32
    var a3: Float32
    var a4: Float32
    var a5: Float32
    var a6: Float32
    var ab: Float32


struct KlattTrack(Movable):
    """5 ms parameter frames (flat, `NPAR` values per frame; frame k is the value at sample k*120),
    sorted burst events and the noise seed."""

    var data: List[Float32]
    var nframes: Int
    var events: List[KlattEvent]
    var seed: UInt32

    def __init__(out self, nframes: Int, seed: UInt32):
        self.nframes = max(1, nframes)
        self.data = List[Float32](length=self.nframes * NPAR, fill=0.0)
        self.events = List[KlattEvent]()
        self.seed = seed
        for k in range(self.nframes):
            self.set_defaults(k)

    def set_defaults(mut self, k: Int):
        var o = k * NPAR
        self.data[o + P_F0] = 110.0
        self.data[o + P_RD] = 1.0
        self.data[o + P_F1] = 500.0
        self.data[o + P_B1] = 80.0
        self.data[o + P_F2] = 1500.0
        self.data[o + P_B2] = 90.0
        self.data[o + P_F3] = 2500.0
        self.data[o + P_B3] = 150.0
        self.data[o + P_F4] = 3300.0
        self.data[o + P_B4] = 250.0
        self.data[o + P_F5] = 3750.0
        self.data[o + P_B5] = 200.0
        self.data[o + P_F6] = 4900.0
        self.data[o + P_B6] = 1000.0
        self.data[o + P_FNP] = 270.0
        self.data[o + P_BNP] = 100.0
        self.data[o + P_FNZ] = 270.0
        self.data[o + P_BNZ] = 100.0
        self.data[o + P_FTP] = 2150.0
        self.data[o + P_BTP] = 180.0
        self.data[o + P_FTZ] = 2150.0
        self.data[o + P_BTZ] = 180.0
        self.data[o + P_B2P] = 200.0
        self.data[o + P_B3P] = 250.0
        self.data[o + P_B4P] = 320.0
        self.data[o + P_B5P] = 360.0
        self.data[o + P_B6P] = 1000.0
        self.data[o + P_F7] = 7500.0
        self.data[o + P_B7] = 2000.0

    @always_inline
    def get(self, k: Int, p: Int) -> Float32:
        return self.data[k * NPAR + p]

    @always_inline
    def set(mut self, k: Int, p: Int, v: Float32):
        self.data[k * NPAR + p] = v

    def total_samples(self) -> Int:
        return self.nframes * KLATT_FRAME_SAMPLES


# ---------------------------------------------------------------------------- resonators
@always_inline
def _clampf(x: Float32, lo: Float32, hi: Float32) -> Float32:
    return max(lo, min(hi, x))


struct Reso(Copyable, Movable):
    """Klatt 2-pole resonator y = A x + B y1 + C y2 with unity DC gain (Klatt 1980 eq. 2-4);
    coefficients ramp linearly over each 4-sample sub-block."""

    var a: Float32
    var b: Float32
    var c: Float32
    var da: Float32
    var db: Float32
    var dc: Float32
    var y1: Float32
    var y2: Float32
    var peak_inv: Float32   # 1 / |H| at the pole frequency (parallel-branch normalisation)

    def __init__(out self):
        self.a = 1.0
        self.b = 0.0
        self.c = 0.0
        self.da = 0.0
        self.db = 0.0
        self.dc = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        self.peak_inv = 1.0

    @always_inline
    def target(mut self, f_in: Float32, bw_in: Float32, n: Int, want_peak: Bool = False):
        var f = _clampf(f_in, 60.0, Float32(0.45) * Float32(KLATT_SR))
        var bw = _clampf(bw_in, 20.0, Float32(0.25) * Float32(KLATT_SR))
        var r = min(Float32(0.9995), det_exp(-Float32(3.141592653589793) * bw / Float32(KLATT_SR)))
        var cs = det_cos(Float32(6.283185307179586) * f / Float32(KLATT_SR))
        var b = Float64(2.0) * Float64(r) * Float64(cs)
        var c = -Float64(r) * Float64(r)
        var a = Float64(1.0) - b - c
        var inv_n = Float32(1.0) / Float32(n)
        self.da = (Float32(a) - self.a) * inv_n
        self.db = (Float32(b) - self.b) * inv_n
        self.dc = (Float32(c) - self.c) * inv_n
        if want_peak:
            # |H(e^jθ)| = A / ((1 - r) * |1 - r e^{-2jθ}|)
            var c2 = Float32(2.0) * cs * cs - Float32(1.0)
            var den = (Float32(1.0) - r) * sqrt(max(Float32(1e-12), Float32(1.0) - Float32(2.0) * r * c2 + r * r))
            self.peak_inv = den / max(Float32(1e-12), Float32(a))

    @always_inline
    def target_lp(mut self, bw: Float32, n: Int):
        """F = 0 low-pass (RGS): poles r, r on the real axis."""
        var r = det_exp(-Float32(3.141592653589793) * bw / Float32(KLATT_SR))
        var b = Float64(2.0) * Float64(r)
        var c = -Float64(r) * Float64(r)
        var a = Float64(1.0) - b - c
        var inv_n = Float32(1.0) / Float32(n)
        self.da = (Float32(a) - self.a) * inv_n
        self.db = (Float32(b) - self.b) * inv_n
        self.dc = (Float32(c) - self.c) * inv_n

    @always_inline
    def set_now(mut self, f_in: Float32, bw_in: Float32):
        """Immediate coefficients (pitch-synchronous R1)."""
        var f = _clampf(f_in, 60.0, Float32(0.45) * Float32(KLATT_SR))
        var bw = _clampf(bw_in, 20.0, Float32(0.25) * Float32(KLATT_SR))
        var r = min(Float32(0.9995), det_exp(-Float32(3.141592653589793) * bw / Float32(KLATT_SR)))
        var cs = det_cos(Float32(6.283185307179586) * f / Float32(KLATT_SR))
        var b = Float64(2.0) * Float64(r) * Float64(cs)
        var c = -Float64(r) * Float64(r)
        self.b = Float32(b)
        self.c = Float32(c)
        self.a = Float32(Float64(1.0) - b - c)
        self.da = 0.0
        self.db = 0.0
        self.dc = 0.0

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var y = self.a * x + self.b * self.y1 + self.c * self.y2
        self.y2 = self.y1
        self.y1 = y
        self.a += self.da
        self.b += self.db
        self.c += self.dc
        return y

    @always_inline
    def flush(mut self):
        if abs(self.y1) + abs(self.y2) < Float32(1e-20):
            self.y1 = 0.0
            self.y2 = 0.0


struct AntiReso(Copyable, Movable):
    """Klatt anti-resonator (zero pair, Klatt 1980 eq. 5): the exact inverse of `Reso` at the same
    F / BW, so an equal pole / zero pair cancels."""

    var a: Float32
    var b: Float32
    var c: Float32
    var da: Float32
    var db: Float32
    var dc: Float32
    var x1: Float32
    var x2: Float32

    def __init__(out self):
        self.a = 1.0
        self.b = 0.0
        self.c = 0.0
        self.da = 0.0
        self.db = 0.0
        self.dc = 0.0
        self.x1 = 0.0
        self.x2 = 0.0

    @always_inline
    def target(mut self, f_in: Float32, bw_in: Float32, n: Int):
        var f = _clampf(f_in, 150.0, Float32(0.45) * Float32(KLATT_SR))
        var bw = _clampf(bw_in, 40.0, Float32(0.25) * Float32(KLATT_SR))
        var r = min(Float32(0.9995), det_exp(-Float32(3.141592653589793) * bw / Float32(KLATT_SR)))
        var cs = det_cos(Float32(6.283185307179586) * f / Float32(KLATT_SR))
        var b = Float64(2.0) * Float64(r) * Float64(cs)
        var c = -Float64(r) * Float64(r)
        var a = Float64(1.0) - b - c
        var a2 = Float64(1.0) / a
        var b2 = -b / a
        var c2 = -c / a
        var inv_n = Float32(1.0) / Float32(n)
        self.da = (Float32(a2) - self.a) * inv_n
        self.db = (Float32(b2) - self.b) * inv_n
        self.dc = (Float32(c2) - self.c) * inv_n

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var y = self.a * x + self.b * self.x1 + self.c * self.x2
        self.x2 = self.x1
        self.x1 = x
        self.a += self.da
        self.b += self.db
        self.c += self.dc
        return y


# ---------------------------------------------------------------------------- LF glottal table
struct LfTable(Movable):
    """LF flow U(phase) for Ee = 1, T0 = 1 on a grid of Rd values (Fant 1995 Rd regressions,
    Fant et al. 1985 equations; zero net flow by bisection on alpha)."""

    var u: List[Float32]       # LF_NRD rows of LF_NPH + 1 points
    var upeak: List[Float32]
    var te: List[Float32]      # open quotient proxy per row (Te / T0)

    def __init__(out self, ta_scale: Float64 = LF_TA_SCALE):
        self.u = List[Float32](length=LF_NRD * (LF_NPH + 1), fill=0.0)
        self.upeak = List[Float32](length=LF_NRD, fill=1.0)
        self.te = List[Float32](length=LF_NRD, fill=0.6)
        for i in range(LF_NRD):
            self._build_row(i, 0.3 + 0.1 * Float64(i), ta_scale)

    def _build_row(mut self, row: Int, rd: Float64, ta_scale: Float64):
        var ra = (-1.0 + 4.8 * rd) / 100.0
        var rk = (22.4 + 11.8 * rd) / 100.0
        var rg = rk * (0.5 + 1.2 * rk) / (4.0 * (0.11 * rd - ra * (0.5 + 1.2 * rk)))
        var tp = 1.0 / (2.0 * rg)
        var te = tp * (1.0 + rk)
        var ta = max(ra * ta_scale, 0.002)
        if te > 0.97:
            te = 0.97
        var tc = 1.0
        var wg = PI64 / tp
        # epsilon: eps * ta = 1 - exp(-eps (tc - te))
        var eps = 1.0 / ta
        for _ in range(40):
            eps = (1.0 - exp(-eps * (tc - te))) / ta
        var ret = -(1.0 / (eps * ta)) * ((1.0 - exp(-eps * (tc - te))) / eps - (tc - te) * exp(-eps * (tc - te)))
        # alpha: open-phase flow + return-phase flow = 0
        var lo = -40.0
        var hi = 80.0
        var flo = _lf_open_area(lo, te, wg) + ret
        var found = False
        var a_prev = lo
        var f_prev = flo
        var steps = 960
        for s in range(1, steps + 1):
            var a = lo + (hi - lo) * Float64(s) / Float64(steps)
            var fa = _lf_open_area(a, te, wg) + ret
            if (fa > 0.0) != (f_prev > 0.0):
                lo = a_prev
                hi = a
                flo = f_prev
                found = True
                break
            a_prev = a
            f_prev = fa
        var alpha = 0.0
        if found:
            for _ in range(60):
                var mid = 0.5 * (lo + hi)
                var fm = _lf_open_area(mid, te, wg) + ret
                if (fm > 0.0) == (flo > 0.0):
                    lo = mid
                    flo = fm
                else:
                    hi = mid
            alpha = 0.5 * (lo + hi)
        var e0 = -1.0 / (exp(alpha * te) * sin(wg * te))
        var den = alpha * alpha + wg * wg
        var ute = e0 * (exp(alpha * te) * (alpha * sin(wg * te) - wg * cos(wg * te)) + wg) / den
        var ee = exp(-eps * (tc - te))
        var base = row * (LF_NPH + 1)
        var peak = 1e-9
        for k in range(LF_NPH + 1):
            var t = Float64(k) / Float64(LF_NPH)
            var u: Float64
            if t <= te:
                u = e0 * (exp(alpha * t) * (alpha * sin(wg * t) - wg * cos(wg * t)) + wg) / den
            else:
                var d = t - te
                u = ute - (1.0 / (eps * ta)) * ((1.0 - exp(-eps * d)) / eps - d * ee)
            if k == LF_NPH:
                u = 0.0
            u = max(0.0, u)
            self.u[base + k] = Float32(u)
            if u > peak:
                peak = u
        self.upeak[row] = Float32(peak)
        self.te[row] = Float32(te)

    def rd_for_oq(self, oq: Float32) -> Float32:
        """Rd whose Te/T0 is closest to `oq` (KLSYN88 OQ -> Rd)."""
        var best = 0
        var bd = Float32(1e9)
        for i in range(LF_NRD):
            var d = abs(self.te[i] - oq)
            if d < bd:
                bd = d
                best = i
        return Float32(0.3) + Float32(0.1) * Float32(best)


def _lf_open_area(alpha: Float64, te: Float64, wg: Float64) -> Float64:
    var e0 = -1.0 / (exp(alpha * te) * sin(wg * te))
    return e0 * (exp(alpha * te) * (alpha * sin(wg * te) - wg * cos(wg * te)) + wg) / (alpha * alpha + wg * wg)


# ---------------------------------------------------------------------------- noise
struct Lcg(Copyable, Movable):
    """32-bit LCG (Numerical Recipes constants); only the top 16 bits are used."""

    var s: UInt32

    def __init__(out self, seed: UInt32):
        self.s = seed

    @always_inline
    def uni(mut self) -> Float32:
        self.s = self.s * UInt32(1664525) + UInt32(1013904223)
        return Float32(self.s >> 16) * Float32(1.0 / 65536.0) - Float32(0.5)

    @always_inline
    def gauss(mut self) -> Float32:
        """Irwin-Hall sum of 4, unit variance."""
        var a = self.uni()
        var b = self.uni()
        var c = self.uni()
        var d = self.uni()
        return (a + b + c + d) * Float32(1.7320508075688772)


# ---------------------------------------------------------------------------- synthesiser
struct KlattSynth(Movable):
    var lf: LfTable

    def __init__(out self, ta_scale: Float64 = LF_TA_SCALE):
        self.lf = LfTable(ta_scale)

    def render(self, track: KlattTrack) -> List[Float32]:
        """Render a track to 24 kHz mono. Deterministic: the same track gives the same PCM."""
        var nfr = track.nframes
        var total = nfr * KLATT_FRAME_SAMPLES
        var out = List[Float32](capacity=total)
        var fs = Float32(KLATT_SR)

        # cascade
        var rtp = Reso()
        var rtz = AntiReso()
        var rnp = Reso()
        var rnz = AntiReso()
        var hpc = List[Reso]()
        for k in range(HPC_N):
            var rk = Reso()
            rk.set_now(HPC_F0 + Float32(1000.0) * Float32(k), HPC_B0 + Float32(100.0) * Float32(k))
            hpc.append(rk^)
        var r6 = Reso()
        var r5 = Reso()
        var r4 = Reso()
        var r3 = Reso()
        var r2 = Reso()
        var r1 = Reso()
        var rgs = Reso()
        # parallel
        var p2 = Reso()
        var p3 = Reso()
        var p4 = Reso()
        var p5 = Reso()
        var p6 = Reso()
        var p7 = Reso()

        var noise_rng = Lcg(track.seed ^ UInt32(0x9E3779B9))
        var voice_rng = Lcg(track.seed ^ UInt32(0x85EBCA6B))
        var n_lp = Float32(0.0)
        var p_noise = Float32(0.8867)   # exp(-2π·460/24000)

        # glottal clock (samples, Float64)
        var t_in = Float64(0.0)
        var t0 = Float64(fs / Float32(110.0))
        var ee_av = Float32(0.0)
        var ee_avs = Float32(0.0)
        var rd_row = 7
        var rd_frac = Float32(0.0)
        var upk = Float32(1.0)
        var tl_p = Float32(0.0)
        var tl_y = Float32(0.0)
        var period_k = 0
        var voiced = False
        var df1 = Float32(0.0)
        var db1 = Float32(0.0)

        # amplitude ramps
        var g_ah = Float32(0.0)
        var g_af = Float32(0.0)
        var g_ab = Float32(0.0)
        var g2 = Float32(0.0)
        var g3 = Float32(0.0)
        var g4 = Float32(0.0)
        var g5 = Float32(0.0)
        var g6 = Float32(0.0)
        var g7 = Float32(0.0)
        var d_ah: Float32
        var d_af: Float32
        var d_ab: Float32
        var d2: Float32
        var d3: Float32
        var d4: Float32
        var d5: Float32
        var d6: Float32
        var d7: Float32
        var f1_cur = Float32(500.0)
        var b1_cur = Float32(80.0)
        var df1_step: Float32
        var db1_step: Float32

        # burst events
        var ev_i = 0
        var b_on = False
        var b_t = 0
        var b_amp = Float32(0.0)
        var b_tau = Float32(1.0)
        var bg2 = Float32(0.0)
        var bg3 = Float32(0.0)
        var bg4 = Float32(0.0)
        var bg5 = Float32(0.0)
        var bg6 = Float32(0.0)
        var bgb = Float32(0.0)

        # output stage
        var y_prev = Float32(0.0)
        var rad_prev = Float32(0.0)
        var dc_y = Float32(0.0)
        var dc_r = Float32(1.0) - Float32(6.283185307179586 * 20.0 / 24000.0)

        var n_global = 0
        for k in range(nfr):
            var k1 = min(k + 1, nfr - 1)
            var o0 = k * NPAR
            var o1 = k1 * NPAR
            for sb in range(KLATT_FRAME_SAMPLES // KLATT_SUB):
                var al = Float32((sb + 1) * KLATT_SUB) / Float32(KLATT_FRAME_SAMPLES)

                var n = KLATT_SUB
                rtp.target(_lp(track.data, o0, o1, P_FTP, al), _lp(track.data, o0, o1, P_BTP, al), n)
                rtz.target(_lp(track.data, o0, o1, P_FTZ, al), _lp(track.data, o0, o1, P_BTZ, al), n)
                rnp.target(_lp(track.data, o0, o1, P_FNP, al), _lp(track.data, o0, o1, P_BNP, al), n)
                rnz.target(_lp(track.data, o0, o1, P_FNZ, al), _lp(track.data, o0, o1, P_BNZ, al), n)
                r6.target(_lp(track.data, o0, o1, P_F6, al), _lp(track.data, o0, o1, P_B6, al), n)
                r5.target(_lp(track.data, o0, o1, P_F5, al), _lp(track.data, o0, o1, P_B5, al), n)
                r4.target(_lp(track.data, o0, o1, P_F4, al), _lp(track.data, o0, o1, P_B4, al), n)
                r3.target(_lp(track.data, o0, o1, P_F3, al), _lp(track.data, o0, o1, P_B3, al), n)
                r2.target(_lp(track.data, o0, o1, P_F2, al), _lp(track.data, o0, o1, P_B2, al), n)
                rgs.target_lp(200.0, n)
                p2.target(_lp(track.data, o0, o1, P_F2, al), _lp(track.data, o0, o1, P_B2P, al), n, True)
                p3.target(_lp(track.data, o0, o1, P_F3, al), _lp(track.data, o0, o1, P_B3P, al), n, True)
                p4.target(_lp(track.data, o0, o1, P_F4, al), _lp(track.data, o0, o1, P_B4P, al), n, True)
                p5.target(_lp(track.data, o0, o1, P_F5, al), _lp(track.data, o0, o1, P_B5P, al), n, True)
                p6.target(_lp(track.data, o0, o1, P_F6, al), _lp(track.data, o0, o1, P_B6P, al), n, True)
                p7.target(_lp(track.data, o0, o1, P_F7, al), _lp(track.data, o0, o1, P_B7, al), n, True)
                var f1_t = _lp(track.data, o0, o1, P_F1, al)
                var b1_t = _lp(track.data, o0, o1, P_B1, al)
                df1_step = (f1_t - f1_cur) / Float32(n)
                db1_step = (b1_t - b1_cur) / Float32(n)
                var inv_n = Float32(1.0) / Float32(n)
                var af_t = db_lin(_lp(track.data, o0, o1, P_AF, al)) * K_AF
                d_af = (af_t - g_af) * inv_n
                d_ah = (db_lin(_lp(track.data, o0, o1, P_AH, al)) * K_AH - g_ah) * inv_n
                d_ab = (db_lin(_lp(track.data, o0, o1, P_AB, al)) * (K_AB / K_AF) - g_ab) * inv_n
                d2 = (db_lin(_lp(track.data, o0, o1, P_A2, al)) * p2.peak_inv - g2) * inv_n
                d3 = (db_lin(_lp(track.data, o0, o1, P_A3, al)) * p3.peak_inv - g3) * inv_n
                d4 = (db_lin(_lp(track.data, o0, o1, P_A4, al)) * p4.peak_inv - g4) * inv_n
                d5 = (db_lin(_lp(track.data, o0, o1, P_A5, al)) * p5.peak_inv - g5) * inv_n
                d6 = (db_lin(_lp(track.data, o0, o1, P_A6, al)) * p6.peak_inv - g6) * inv_n
                d7 = (db_lin(_lp(track.data, o0, o1, P_A7, al)) * p7.peak_inv - g7) * inv_n
                var bpk2 = p2.peak_inv
                var bpk3 = p3.peak_inv
                var bpk4 = p4.peak_inv
                var bpk5 = p5.peak_inv
                var bpk6 = p6.peak_inv

                for i in range(n):
                    var a_in = Float32(sb * KLATT_SUB + i) / Float32(KLATT_FRAME_SAMPLES)

                    # ---- burst events
                    while ev_i < len(track.events) and track.events[ev_i].sample <= n_global:
                        ref ev = track.events[ev_i]
                        b_on = True
                        b_t = 0
                        b_amp = db_lin(ev.amp_db) * K_AF
                        b_tau = max(Float32(0.5), ev.decay_ms) * Float32(0.001) * fs / Float32(2.302585)
                        bg2 = db_lin(ev.a2)
                        bg3 = db_lin(ev.a3)
                        bg4 = db_lin(ev.a4)
                        bg5 = db_lin(ev.a5)
                        bg6 = db_lin(ev.a6)
                        bgb = db_lin(ev.ab) * K_AB / K_AF
                        ev_i += 1
                    var benv = Float32(0.0)
                    if b_on:
                        if b_t < 6:
                            benv = Float32(0.5) - Float32(0.5) * det_cos(Float32(3.14159265) * Float32(b_t) / Float32(6.0))
                        else:
                            benv = det_exp(-Float32(b_t - 6) / b_tau)
                        benv *= b_amp
                        b_t += 1
                        if b_t > 6 and benv < b_amp * Float32(1e-3):
                            b_on = False

                    # ---- glottal clock and period latch
                    t_in += 1.0
                    if t_in >= t0:
                        t_in -= t0
                        period_k += 1

                        var f0 = max(Float32(40.0), _lp(track.data, o0, o1, P_F0, a_in))
                        var tsec = Float32(n_global) / fs
                        var fl = _lp(track.data, o0, o1, P_FL, a_in)
                        if fl > Float32(0.0):
                            var w = Float32(6.283185307179586) * tsec
                            f0 += (fl / Float32(50.0)) * (f0 / Float32(100.0)) * (
                                det_sin(w * Float32(12.7)) + det_sin(w * Float32(7.1)) + det_sin(w * Float32(4.7)))
                        var jit = _lp(track.data, o0, o1, P_JIT, a_in)
                        var gj = voice_rng.gauss()
                        var gs = voice_rng.gauss()
                        var per = Float64(fs / max(Float32(40.0), f0)) * Float64(Float32(1.0) + jit * gj)
                        var di = _lp(track.data, o0, o1, P_DI, a_in) * Float32(0.01)
                        var amp = Float32(1.0)
                        if di > Float32(0.0):
                            if (period_k & 1) == 0:
                                per *= Float64(Float32(1.0) + Float32(0.5) * di)
                            else:
                                per *= Float64(Float32(1.0) - Float32(0.5) * di)
                                amp = Float32(1.0) - di
                        t0 = max(Float64(24.0), per)
                        var shim = _lp(track.data, o0, o1, P_SHIM, a_in)
                        if shim > Float32(0.0):
                            amp *= det_exp(shim * gs * Float32(0.11512925464970229))
                        var av = _lp(track.data, o0, o1, P_AV, a_in)
                        var avs = _lp(track.data, o0, o1, P_AVS, a_in)
                        ee_av = db_lin(av) * K_AV * amp
                        ee_avs = db_lin(avs) * K_AVS * amp
                        voiced = ee_av > Float32(0.0) or ee_avs > Float32(0.0)
                        var rd = _clampf(_lp(track.data, o0, o1, P_RD, a_in), 0.3, 2.69)
                        var rf = (rd - Float32(0.3)) * Float32(10.0)
                        rd_row = min(LF_NRD - 2, Int(rf))
                        rd_frac = rf - Float32(rd_row)
                        upk = self.lf.upeak[rd_row] + rd_frac * (self.lf.upeak[rd_row + 1] - self.lf.upeak[rd_row])
                        var tl = _lp(track.data, o0, o1, P_TL, a_in)
                        tl_p = _tl_pole(tl)
                        df1 = _lp(track.data, o0, o1, P_DF1, a_in)
                        db1 = _lp(track.data, o0, o1, P_DB1, a_in)

                    # ---- LF flow (unit Ee, scaled by T0 in samples)
                    var u_unit = Float32(0.0)
                    var u_rel = Float32(0.0)
                    if voiced:
                        var ph = Float32(t_in / t0) * Float32(LF_NPH)
                        var ip = min(LF_NPH - 1, Int(ph))
                        var fr = ph - Float32(ip)
                        var b0 = rd_row * (LF_NPH + 1) + ip
                        var b1i = b0 + (LF_NPH + 1)
                        var ua = self.lf.u[b0] + fr * (self.lf.u[b0 + 1] - self.lf.u[b0])
                        var ub = self.lf.u[b1i] + fr * (self.lf.u[b1i + 1] - self.lf.u[b1i])
                        var un = ua + rd_frac * (ub - ua)
                        u_rel = un / upk
                        u_unit = un * Float32(t0)
                    tl_y = (Float32(1.0) - tl_p) * u_unit + tl_p * tl_y
                    var ug = tl_y * ee_av

                    # ---- noise
                    var g = noise_rng.gauss()
                    n_lp = g + p_noise * n_lp
                    var nz = n_lp   # about unit level above 460 Hz after radiation (§4.2)
                    g_ah += d_ah
                    g_af += d_af
                    g_ab += d_ab
                    g2 += d2
                    g3 += d3
                    g4 += d4
                    g5 += d5
                    g6 += d6
                    g7 += d7
                    var m_asp = Float32(1.0)
                    var m_fr = Float32(1.0)
                    if ee_av > Float32(0.0):
                        m_asp = Float32(0.25) + Float32(0.75) * u_rel
                        m_fr = Float32(0.5) + Float32(0.5) * u_rel
                    var asp = g_ah * nz * m_asp

                    # ---- cascade
                    var avs_sig = rgs.tick(u_unit) * ee_avs
                    var x = ug + avs_sig + asp
                    x = rtz.tick(rtp.tick(x))
                    x = rnz.tick(rnp.tick(x))
                    for k in range(HPC_N):
                        x = hpc[k].tick(x)
                    x = r6.tick(x)
                    x = r5.tick(x)
                    x = r4.tick(x)
                    x = r3.tick(x)
                    x = r2.tick(x)
                    f1_cur += df1_step
                    b1_cur += db1_step
                    r1.set_now(f1_cur + df1 * u_rel, b1_cur + db1 * u_rel)
                    var casc = r1.tick(x)

                    # ---- parallel (frication + bursts)
                    var src = nz * m_fr
                    var fa = g_af
                    var pin2 = src * (fa * g2 + benv * bg2 * bpk2)
                    var pin3 = src * (fa * g3 + benv * bg3 * bpk3)
                    var pin4 = src * (fa * g4 + benv * bg4 * bpk4)
                    var pin5 = src * (fa * g5 + benv * bg5 * bpk5)
                    var pin6 = src * (fa * g6 + benv * bg6 * bpk6)
                    var pin7 = src * (fa * g7)
                    var par = -p2.tick(pin2) + p3.tick(pin3) - p4.tick(pin4) + p5.tick(pin5) - p6.tick(pin6) + p7.tick(pin7)
                    par += src * (fa * g_ab + benv * bgb)

                    # ---- output stage
                    var y = casc + par
                    var rad = y - y_prev
                    y_prev = y
                    dc_y = rad - rad_prev + dc_r * dc_y
                    rad_prev = rad
                    out.append(_soft_limit(dc_y * K_OUT))
                    n_global += 1
                r1.flush()
                r2.flush()
                r3.flush()
                r4.flush()
                r5.flush()
                r6.flush()
                rnp.flush()
                rtp.flush()
                rgs.flush()
                p2.flush()
                p3.flush()
                p4.flush()
                p5.flush()
                p6.flush()
                p7.flush()
        # 2 ms fade-out
        var nf = min(len(out), 48)
        for i in range(nf):
            var idx = len(out) - nf + i
            out[idx] *= Float32(nf - 1 - i) / Float32(nf)
        return out^


@always_inline
def _lp(d: List[Float32], o0: Int, o1: Int, p: Int, al: Float32) -> Float32:
    var v0 = d[o0 + p]
    return v0 + al * (d[o1 + p] - v0)


@always_inline
def _soft_limit(x: Float32) -> Float32:
    """Unity below a 0.7 knee, C1-continuous tanh shoulder above (peak <= 1)."""
    var a = abs(x)
    if a <= Float32(0.7):
        return x
    var y = Float32(0.7) + Float32(0.3) * soft_clip_tanh((a - Float32(0.7)) / Float32(0.3))
    return y if x > 0 else -y


@always_inline
def _tl_pole(tl: Float32) -> Float32:
    """One-pole low-pass pole giving exactly TL dB attenuation at 3 kHz (KLSYN88 TL)."""
    if tl < Float32(0.1):
        return Float32(0.0)
    var g = det_exp(-tl * Float32(0.2302585093))   # 10^(-TL/10)
    var c = Float32(0.7071067811865476)             # cos(2π·3000/24000)
    var a = Float32(1.0) - g * c
    var d = a * a - (Float32(1.0) - g) * (Float32(1.0) - g)
    return (a - sqrt(max(Float32(0.0), d))) / (Float32(1.0) - g)
