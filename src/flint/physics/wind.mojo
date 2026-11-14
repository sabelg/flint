"""The wind evaluator, in fixed point (package 2.12).

Air moves in the simulation as one field: every wind area's base vector
times hashed tick noise (gusts), every updraft's rising air, and the
vorticles gameplay drops into it (a roar, a clap, a slam, an explosion, a
cave-in blast). Bodies feel it as drag toward the air's velocity
(physics/forces.mojo); the presentation bakes the same field into the wind
mirror the GPU reads (render/wind_mirror.mojo).

Everything here is integer arithmetic on Q16.16 values in Int64: positions,
velocities, weights and noise. Integer adds, multiplies, shifts and floor
divisions are the same on every CPU, so the wind at a point and tick is the
same bits everywhere, whatever the float rules of the caller. Floats enter
once, when a volume or vorticle is made (`to_fx`, exact for the scale of the
world), and leave once, when the result is used (`FxVec.to_vec3`).

Gusts. A wind area's noise is value noise over (cell x, cell z, tick
period): each lattice value is splitmix64 of the area's seed and the three
integers, mapped to [-1, 1); between lattice points it is interpolated
linearly in x, z and time, so the wind changes smoothly in space and over
ticks. The wind is `base * (1 + gust * noise) * weight`.

Vorticles. A vorticle is a short-lived swirl and burst: at distance r < R
from its centre, with k = (1 - r/R)^2 and a linear decay over its life, it
adds `swirl * k * (axis x d/r)` (turning air) and `burst * k * d/r` (air
pushed out). They age one per tick, are dropped at the end of their life,
and at most MAX_VORTICLES live (the oldest goes first).
"""
from flint.core.math import Vec3
from flint.core.detsort import mix64
from flint.core.serial import BlobWriter, BlobReader

comptime FX_SHIFT: Int64 = 16
comptime FX_ONE: Int64 = 65536
comptime MAX_VORTICLES: Int = 32
comptime SHAPE_BOX: Int = 0
comptime SHAPE_SPHERE: Int = 1
comptime SHAPE_CAPSULE: Int = 2
comptime FALLOFF_NONE: Int = 0
comptime FALLOFF_LINEAR: Int = 1
comptime FALLOFF_SMOOTH: Int = 2


def to_fx(value: Float32) -> Int64:
    """A float in Q16.16, rounded to nearest (exact: the scale is a power of two)."""
    var scaled = Float64(value) * 65536.0  # det: exact (a Float32 times 2^16 fits a Float64); rounded to an integer below
    if scaled >= 0:
        return Int64(scaled + 0.5)
    return -Int64(-scaled + 0.5)


def fx_mul(a: Int64, b: Int64) -> Int64:
    return (a * b) >> FX_SHIFT


def fx_div(a: Int64, b: Int64) -> Int64:
    if b == 0:
        return 0
    return (a << FX_SHIFT) // b


def isqrt(value: Int64) -> Int64:
    """floor(sqrt(value)) for value >= 0, by the bit-by-bit method."""
    if value <= 0:
        return 0
    var n = value
    var result: Int64 = 0
    var bit: Int64 = Int64(1) << 62
    while bit > n:
        bit >>= 2
    while bit != 0:
        if n >= result + bit:
            n -= result + bit
            result = (result >> 1) + bit
        else:
            result >>= 1
        bit >>= 2
    return result


@fieldwise_init
struct FxVec(Copyable, ImplicitlyCopyable, Movable, Equatable):
    var x: Int64
    var y: Int64
    var z: Int64

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, 0)

    @staticmethod
    def of(v: Vec3) -> Self:
        return Self(to_fx(v.x), to_fx(v.y), to_fx(v.z))

    def to_vec3(self) -> Vec3:
        return Vec3(Float32(Float64(self.x) / 65536.0), Float32(Float64(self.y) / 65536.0), Float32(Float64(self.z) / 65536.0))  # det: one correctly rounded conversion of an exact quotient

    def __add__(self, o: Self) -> Self:
        return Self(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: Self) -> Self:
        return Self(self.x - o.x, self.y - o.y, self.z - o.z)

    def __eq__(self, o: Self) -> Bool:
        return self.x == o.x and self.y == o.y and self.z == o.z

    def __ne__(self, o: Self) -> Bool:
        return not self == o

    def scale(self, s: Int64) -> Self:
        """Times a Q16 scalar."""
        return Self(fx_mul(self.x, s), fx_mul(self.y, s), fx_mul(self.z, s))

    def dot_raw(self, o: Self) -> Int64:
        """The dot product in Q32 (no shift): exact for world-sized values."""
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: Self) -> Self:
        return Self(
            fx_mul(self.y, o.z) - fx_mul(self.z, o.y),
            fx_mul(self.z, o.x) - fx_mul(self.x, o.z),
            fx_mul(self.x, o.y) - fx_mul(self.y, o.x),
        )

    def length(self) -> Int64:
        return isqrt(self.dot_raw(self))

    def write(self, mut out: BlobWriter):
        out.u64(UInt64(self.x))
        out.u64(UInt64(self.y))
        out.u64(UInt64(self.z))

    @staticmethod
    def read(mut input: BlobReader) raises -> Self:
        var x = Int64(input.u64())
        var y = Int64(input.u64())
        return Self(x, y, Int64(input.u64()))


def falloff_fx(kind: Int, t: Int64) -> Int64:
    """The weight at normalized distance t (0 at the middle, ONE at the edge)."""
    if t >= FX_ONE or t < 0:
        return 0
    if kind == FALLOFF_LINEAR:
        return FX_ONE - t
    if kind == FALLOFF_SMOOTH:
        var t2 = fx_mul(t, t)
        return FX_ONE - (3 * t2 - 2 * fx_mul(t2, t))
    return FX_ONE


struct FxShape(Copyable, ImplicitlyCopyable, Movable):
    """A volume's shape in fixed point: a box (centre, half extents), a
    sphere (centre, radius) or a capsule (centre to `end`, radius)."""
    var shape: Int
    var falloff: Int
    var center: FxVec
    var half: FxVec
    var end: FxVec
    var radius: Int64

    def __init__(out self):
        self.shape = SHAPE_BOX
        self.falloff = FALLOFF_NONE
        self.center = FxVec.zero()
        self.half = FxVec.zero()
        self.end = FxVec.zero()
        self.radius = 0

    def distance_t(self, p: FxVec) -> Int64:
        """The normalized distance of p: below ONE inside."""
        var d = p - self.center
        if self.shape == SHAPE_SPHERE:
            if self.radius <= 0:
                return FX_ONE
            return fx_div(d.length(), self.radius)
        if self.shape == SHAPE_CAPSULE:
            if self.radius <= 0:
                return FX_ONE
            var axis = self.end - self.center
            var len2 = axis.dot_raw(axis)
            var s: Int64 = 0
            if len2 > 0:
                var along = d.dot_raw(axis)
                s = max(Int64(0), min(FX_ONE, (along // max(Int64(1), len2 >> FX_SHIFT))))
            var closest = self.center + axis.scale(s)
            return fx_div((p - closest).length(), self.radius)
        var t: Int64 = 0
        if self.half.x <= 0 or self.half.y <= 0 or self.half.z <= 0:
            return FX_ONE
        t = max(t, fx_div(abs(d.x), self.half.x))
        t = max(t, fx_div(abs(d.y), self.half.y))
        t = max(t, fx_div(abs(d.z), self.half.z))
        return t

    def weight(self, p: FxVec) -> Int64:
        return falloff_fx(self.falloff, self.distance_t(p))


struct WindSource(Copyable, ImplicitlyCopyable, Movable):
    """A wind area or an updraft in the field: its shape, the base air
    velocity, the gust share, the noise cell size and period, and the seed
    (the volume's id)."""
    var id: Int
    var shape: FxShape
    var base: FxVec
    var gust: Int64      # Q16 share of the base the noise adds or takes
    var cell: Int64      # Q16 metres between noise lattice points
    var period: Int      # ticks between noise lattice points
    var seed: UInt64

    def __init__(out self):
        self.id = 0
        self.shape = FxShape()
        self.base = FxVec.zero()
        self.gust = 0
        self.cell = 8 * FX_ONE
        self.period = 60
        self.seed = 0


def lattice(seed: UInt64, cx: Int64, cz: Int64, k: Int64) -> Int64:
    """A noise lattice value in [-ONE, ONE)."""
    var h = mix64(seed ^ mix64(UInt64(cx) ^ mix64(UInt64(cz) ^ mix64(UInt64(k)))))
    return Int64(h & 0x1FFFF) - FX_ONE


def lerp_fx(a: Int64, b: Int64, t: Int64) -> Int64:
    return a + fx_mul(b - a, t)


def gust_noise(source: WindSource, p: FxVec, tick: Int) -> Int64:
    """Value noise at p and tick, trilinear over (x cell, z cell, period)."""
    var cell = max(Int64(FX_ONE // 16), source.cell)
    var gx = fx_div(p.x, cell)
    var gz = fx_div(p.z, cell)
    var cx = gx >> FX_SHIFT
    var cz = gz >> FX_SHIFT
    var fx = gx - (cx << FX_SHIFT)
    var fz = gz - (cz << FX_SHIFT)
    var period = max(1, source.period)
    var k = Int64(tick // period)
    var ft = (Int64(tick % period) << FX_SHIFT) // Int64(period)
    var v = List[Int64]()
    for dt in range(2):
        var a = lerp_fx(lattice(source.seed, cx, cz, k + Int64(dt)), lattice(source.seed, cx + 1, cz, k + Int64(dt)), fx)
        var b = lerp_fx(lattice(source.seed, cx, cz + 1, k + Int64(dt)), lattice(source.seed, cx + 1, cz + 1, k + Int64(dt)), fx)
        v.append(lerp_fx(a, b, fz))
    return lerp_fx(v[0], v[1], ft)


struct Vorticle(Copyable, ImplicitlyCopyable, Movable):
    var id: Int
    var center: FxVec
    var axis: FxVec      # unit, Q16
    var radius: Int64
    var swirl: Int64     # m/s of turning air at full strength
    var burst: Int64     # m/s of air pushed out at full strength
    var life: Int        # ticks
    var age: Int

    def __init__(out self):
        self.id = 0
        self.center = FxVec.zero()
        self.axis = FxVec(0, FX_ONE, 0)
        self.radius = FX_ONE
        self.swirl = 0
        self.burst = 0
        self.life = 1
        self.age = 0

    def write(self, mut out: BlobWriter):
        out.integer(self.id)
        self.center.write(out)
        self.axis.write(out)
        out.u64(UInt64(self.radius))
        out.u64(UInt64(self.swirl))
        out.u64(UInt64(self.burst))
        out.integer(self.life)
        out.integer(self.age)

    @staticmethod
    def read(mut input: BlobReader) raises -> Self:
        var v = Self()
        v.id = input.integer()
        v.center = FxVec.read(input)
        v.axis = FxVec.read(input)
        v.radius = Int64(input.u64())
        v.swirl = Int64(input.u64())
        v.burst = Int64(input.u64())
        v.life = input.integer()
        v.age = input.integer()
        return v


def make_vorticle(id: Int, center: Vec3, axis: Vec3, radius: Float32, swirl: Float32, burst: Float32, life: Int) -> Vorticle:
    var v = Vorticle()
    v.id = id
    v.center = FxVec.of(center)
    var a = FxVec.of(axis)
    var length = a.length()
    v.axis = FxVec(fx_div(a.x, length), fx_div(a.y, length), fx_div(a.z, length)) if length > 0 else FxVec(0, FX_ONE, 0)
    v.radius = max(Int64(FX_ONE // 4), to_fx(radius))
    v.swirl = to_fx(swirl)
    v.burst = to_fx(burst)
    v.life = max(1, life)
    return v


def vorticle_wind(v: Vorticle, p: FxVec) -> FxVec:
    var d = p - v.center
    var r2 = d.dot_raw(d)
    var limit = v.radius * v.radius
    if r2 >= limit:
        return FxVec.zero()
    var r = isqrt(r2)
    var k = FX_ONE - fx_div(r, v.radius)
    k = fx_mul(k, k)
    var decay = (Int64(max(0, v.life - v.age)) << FX_SHIFT) // Int64(v.life)
    var s = fx_mul(k, decay)
    if r < FX_ONE // 256:
        return FxVec.zero()
    var dir = FxVec(fx_div(d.x, r), fx_div(d.y, r), fx_div(d.z, r))
    var radial = dir.scale(fx_mul(v.burst, s))
    var turning = v.axis.cross(dir).scale(fx_mul(v.swirl, s))
    return radial + turning


struct WindField(Copyable, Movable):
    """The sources and vorticles, ready to evaluate."""
    var sources: List[WindSource]
    var vorticles: List[Vorticle]

    def __init__(out self):
        self.sources = List[WindSource]()
        self.vorticles = List[Vorticle]()

    def at(self, p: FxVec, tick: Int) -> FxVec:
        """The air's velocity at p on this tick, in Q16 m/s."""
        var out = FxVec.zero()
        for source in self.sources:
            var w = source.shape.weight(p)
            if w == 0:
                continue
            var scale = w
            if source.gust != 0:
                scale = fx_mul(w, FX_ONE + fx_mul(source.gust, gust_noise(source, p, tick)))
            out = out + source.base.scale(scale)
        for v in self.vorticles:
            out = out + vorticle_wind(v, p)
        return out

    def at_vec(self, p: Vec3, tick: Int) -> Vec3:
        return self.at(FxVec.of(p), tick).to_vec3()

    def is_calm(self) -> Bool:
        return len(self.sources) == 0 and len(self.vorticles) == 0

    def age(mut self):
        """One tick: vorticles age; spent ones go (kept in order)."""
        var kept = List[Vorticle]()
        for v in self.vorticles:
            var next = v
            next.age += 1
            if next.age < next.life:
                kept.append(next)
        self.vorticles = kept^

    def add_vorticle(mut self, v: Vorticle):
        if len(self.vorticles) >= MAX_VORTICLES:
            var kept = List[Vorticle]()
            for index in range(1, len(self.vorticles)):
                kept.append(self.vorticles[index])
            self.vorticles = kept^
        self.vorticles.append(v)
