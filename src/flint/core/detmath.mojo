"""Deterministic math for the simulation (package 0.12).

The simulation must give the same bits on every machine, build and worker
count (docs/DETERMINISM.md). The C library's `sinf`, `cosf`, `expf`, `logf` and
`powf` differ between glibc versions and pick FMA or non-FMA code at run time
by CPU, and the standard library's own polynomials are whatever its version
ships. So simulation code calls these instead: each is a fixed sequence of
IEEE-754 single-precision adds, multiplies, divides, `sqrt` and `floor` (all
correctly rounded, so exact on every x86-64 and ARM CPU), compiled with FMA
contraction off (`--fp-mode contract=off` in tools/build.sh).

Accuracy is a few units in the last place over the ranges the game uses
(tests/test_determinism.mojo measures it against Float64), which is all
gameplay needs; being the same everywhere is the point. The polynomials are
the classic single-precision minimax fits (Moshier's Cephes) with Cody-Waite
range reduction; see docs/RESEARCH.md, Test oracles.

The MXCSR helpers keep every thread that runs simulation code in one
floating-point mode: round to nearest, denormals kept (FTZ and DAZ off), all
exceptions masked. `core/jobs.mojo` sets it on every worker, and a tick
asserts it (`gameplay/invariants.mojo`).
"""
from std.math import floor, sqrt
from std.sys import llvm_intrinsic

comptime PI: Float32 = 3.14159265358979323846
comptime HALF_PI: Float32 = 1.57079632679489661923
comptime QUARTER_PI: Float32 = 0.78539816339744830962
comptime TWO_OVER_PI: Float32 = 0.63661977236758134308

# pi/2 in three parts; the first two have few enough bits that j * part is
# exact for every |j| < 2^15, which covers |x| below about 51000.
comptime PIO2_1: Float32 = 1.5703125
comptime PIO2_2: Float32 = 4.837512969970703125e-4
comptime PIO2_3: Float32 = 7.54978995489188216e-8

comptime LOG2E: Float32 = 1.44269504088896341
comptime LN2_HI: Float32 = 0.693359375
comptime LN2_LO: Float32 = -2.12194440e-4
comptime SQRT_HALF: Float32 = 0.707106781186547524

# MXCSR: the control bits only (6-15); 0-5 are sticky exception flags.
comptime MXCSR_CONTROL_MASK: UInt32 = 0xFFC0
# Round to nearest, every exception masked, FTZ (bit 15) and DAZ (bit 6) off:
# the IEEE default, which is also what Linux starts every thread with.
comptime MXCSR_SIM: UInt32 = 0x1F80


@always_inline
def f32_bits(value: Float32) -> UInt32:
    var copy = value
    return Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]


@always_inline
def f32_from_bits(bits: UInt32) -> Float32:
    var copy = bits
    return Pointer(to=copy).unsafe_bitcast[Float32]()[unsafe_offset=0]


@always_inline
def is_nan(x: Float32) -> Bool:
    return x != x


@always_inline
def is_finite32(x: Float32) -> Bool:
    return (f32_bits(x) & 0x7F800000) != 0x7F800000


def quiet_nan() -> Float32:
    return f32_from_bits(0x7FC00000)


def infinity() -> Float32:
    return f32_from_bits(0x7F800000)


def scale_by_pow2(x: Float32, n: Int) -> Float32:
    """x * 2^n with exact powers of two (ldexp). At most three steps, so a
    result that underflows rounds the same way everywhere."""
    var value = x
    var e = n
    while e > 127:
        value = value * f32_from_bits(UInt32(254) << 23)  # 2^127
        e -= 127
        if e > 127 and not is_finite32(value):
            return value
    while e < -126:
        value = value * f32_from_bits(UInt32(1) << 23)  # 2^-126
        e += 126
        if e < -126 and value == 0:
            return value
    return value * f32_from_bits(UInt32(e + 127) << 23)


# ---------------------------------------------------------------------------
# sin and cos
# ---------------------------------------------------------------------------

@always_inline
def _sin_poly(r: Float32, z: Float32) -> Float32:
    return ((Float32(-1.9515295891e-4) * z + Float32(8.3321608736e-3)) * z + Float32(-1.6666654611e-1)) * z * r + r


@always_inline
def _cos_poly(z: Float32) -> Float32:
    return ((Float32(2.443315711809948e-5) * z + Float32(-1.388731625493765e-3)) * z + Float32(4.166664568298827e-2)) * z * z - Float32(0.5) * z + Float32(1.0)


def _quadrant(x: Float32) -> Tuple[Float32, Int]:
    """x - j*pi/2 in [-pi/4, pi/4] and j mod 4."""
    var j = floor(x * TWO_OVER_PI + Float32(0.5))
    var r = ((x - j * PIO2_1) - j * PIO2_2) - j * PIO2_3
    return (r, Int(j) & 3)


def det_sin(x: Float32) -> Float32:
    if not (abs(x) < Float32(1.0e9)):
        return quiet_nan() if not is_finite32(x) else Float32(0)
    var reduced = _quadrant(x)
    var r = reduced[0]
    var z = r * r
    var q = reduced[1]
    if q == 0:
        return _sin_poly(r, z)
    if q == 1:
        return _cos_poly(z)
    if q == 2:
        return -_sin_poly(r, z)
    return -_cos_poly(z)


def det_cos(x: Float32) -> Float32:
    if not (abs(x) < Float32(1.0e9)):
        return quiet_nan() if not is_finite32(x) else Float32(1)
    var reduced = _quadrant(x)
    var r = reduced[0]
    var z = r * r
    var q = reduced[1]
    if q == 0:
        return _cos_poly(z)
    if q == 1:
        return -_sin_poly(r, z)
    if q == 2:
        return -_cos_poly(z)
    return _sin_poly(r, z)


# ---------------------------------------------------------------------------
# Inverse trigonometry
# ---------------------------------------------------------------------------

def det_atan(x: Float32) -> Float32:
    if is_nan(x):
        return x
    var negative = x < 0
    var a = abs(x)
    var base = Float32(0)
    if a > Float32(2.414213562373095):
        base = HALF_PI
        a = Float32(-1.0) / a
    elif a > Float32(0.4142135623730950):
        base = QUARTER_PI
        a = (a - Float32(1.0)) / (a + Float32(1.0))
    var z = a * a
    var result = base + ((((Float32(8.05374449538e-2) * z - Float32(1.38776856032e-1)) * z + Float32(1.99777106478e-1)) * z - Float32(3.33329491539e-1)) * z * a + a)
    return -result if negative else result


def det_atan2(y: Float32, x: Float32) -> Float32:
    """The angle of (x, y), in [-pi, pi], as std atan2 for every finite input
    (signed zeros: atan2(+0, -x) is pi, atan2(-0, -x) is -pi)."""
    if is_nan(x) or is_nan(y):
        return quiet_nan()
    var y_negative = (f32_bits(y) >> 31) != 0
    if x == 0:
        if y == 0:
            var x_negative = (f32_bits(x) >> 31) != 0
            if x_negative:
                return -PI if y_negative else PI
            return Float32(-0.0) if y_negative else Float32(0.0)
        return -HALF_PI if y < 0 else HALF_PI
    if y == 0:
        if x > 0:
            return y
        return -PI if y_negative else PI
    var angle = det_atan(y / x)
    if x < 0:
        angle = angle - PI if y < 0 else angle + PI
    return angle


def _asin_core(s: Float32, z: Float32) -> Float32:
    return ((((Float32(4.2163199048e-2) * z + Float32(2.4181311049e-2)) * z + Float32(4.5470025998e-2)) * z + Float32(7.4953002686e-2)) * z + Float32(1.6666752422e-1)) * z * s + s


def det_asin(x: Float32) -> Float32:
    var a = abs(x)
    if not (a <= Float32(1.0)):
        return quiet_nan()
    var result: Float32
    if a > Float32(0.5):
        var z = Float32(0.5) * (Float32(1.0) - a)
        var s = sqrt(z)
        var p = _asin_core(s, z)
        result = HALF_PI - (p + p)
    else:
        result = _asin_core(a, a * a)
    return -result if x < 0 else result


def det_acos(x: Float32) -> Float32:
    if not (abs(x) <= Float32(1.0)):
        return quiet_nan()
    if x < Float32(-0.5):
        return PI - Float32(2.0) * det_asin(sqrt(Float32(0.5) * (Float32(1.0) + x)))
    if x > Float32(0.5):
        return Float32(2.0) * det_asin(sqrt(Float32(0.5) * (Float32(1.0) - x)))
    return HALF_PI - det_asin(x)


# ---------------------------------------------------------------------------
# exp, log, pow
# ---------------------------------------------------------------------------

def det_exp(x: Float32) -> Float32:
    if is_nan(x):
        return x
    if x > Float32(88.72283905206835):
        return infinity()
    if x < Float32(-103.97207708):
        return Float32(0)
    var n = floor(LOG2E * x + Float32(0.5))
    var r = (x - n * LN2_HI) - n * LN2_LO
    var z = r * r
    var p = (((((Float32(1.9875691500e-4) * r + Float32(1.3981999507e-3)) * r + Float32(8.3334519073e-3)) * r + Float32(4.1665795894e-2)) * r + Float32(1.6666665459e-1)) * r + Float32(5.0000001201e-1)) * z + r + Float32(1.0)
    return scale_by_pow2(p, Int(n))


def det_log(x: Float32) -> Float32:
    """Natural logarithm: -inf at 0, NaN below."""
    if is_nan(x) or x < 0:
        return quiet_nan()
    if x == 0:
        return -infinity()
    if not is_finite32(x):
        return x
    var bits = f32_bits(x)
    var e = 0
    if (bits & 0x7F800000) == 0:  # subnormal: scale up by 2^25 first
        bits = f32_bits(x * f32_from_bits(UInt32(152) << 23))
        e = -25
    e += Int((bits >> 23) & 0xFF) - 126
    var m = f32_from_bits((bits & 0x007FFFFF) | 0x3F000000)  # in [0.5, 1)
    if m < SQRT_HALF:
        e -= 1
        m = m + m - Float32(1.0)
    else:
        m = m - Float32(1.0)
    var z = m * m
    var y = ((((((((Float32(7.0376836292e-2) * m - Float32(1.1514610310e-1)) * m + Float32(1.1676998740e-1)) * m - Float32(1.2420140846e-1)) * m + Float32(1.4249322787e-1)) * m - Float32(1.6668057665e-1)) * m + Float32(2.0000714765e-1)) * m - Float32(2.4999993993e-1)) * m + Float32(3.3333331174e-1)) * m * z
    var fe = Float32(e)
    y = y + fe * LN2_LO
    y = y - Float32(0.5) * z
    return (m + y) + fe * LN2_HI


def det_pow(x: Float32, y: Float32) -> Float32:
    """x^y as exp(y log |x|), with C's rules for zeros, ones and negative bases
    (an integer power keeps the sign; a fractional power of a negative is NaN)."""
    if y == 0 or x == 1:
        return Float32(1)
    if is_nan(x) or is_nan(y):
        return quiet_nan()
    if x == 0:
        return Float32(0) if y > 0 else infinity()
    if x < 0:
        if floor(y) != y:
            return quiet_nan()
        var magnitude = det_exp(y * det_log(-x))
        var half = y * Float32(0.5)
        return -magnitude if floor(half) != half else magnitude
    return det_exp(y * det_log(x))


# ---------------------------------------------------------------------------
# The floating-point mode of the thread
# ---------------------------------------------------------------------------

def read_mxcsr() -> UInt32:
    var value = UInt32(0)
    llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](Pointer(to=value))
    return value


def write_mxcsr(value: UInt32):
    var copy = value
    llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](Pointer(to=copy))


def set_sim_float_mode():
    """Round to nearest, FTZ and DAZ off, exceptions masked; flags cleared."""
    write_mxcsr(MXCSR_SIM)


def sim_float_mode_ok() -> Bool:
    return (read_mxcsr() & MXCSR_CONTROL_MASK) == MXCSR_SIM
