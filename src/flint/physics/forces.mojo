"""Field volumes: forces, wind, updrafts, currents, water (package 2.12).

One volume type covers everything that pushes bodies without touching them:
a shape (box, sphere or capsule) times a falloff (none, linear, smooth: the
weight from 1 in the middle to 0 at the edge) times a target:

    force     a constant push: `vector` in m/s2, the same for every mass
              (like gravity)
    velocity  drag toward `vector` at `rate` per second (a conveyor of air
              or water that moves anything inside)
    updraft   rising air: `strength` m/s up, felt like wind
    wind      moving air: `vector` m/s, with gusts (`gust` share, noise
              `cell` metres and `period` ticks) and the vorticles
    current   moves the water: `vector` m/s adds to the water's flow where
              it is (water drag pulls toward the flow), and drags at `rate`
              more on the part of a body under water (a river's pull)
    radial    a magnet-style pull toward the centre, `strength` m/s2
              (negative pushes out)
    wake      drag at `rate` toward `strength` m/s level and sideways out of
              the line through the centre along `vector` (what a moving hull
              leaves)
    kill      a body inside is taken out (a kill event); the game respawns
              the player from it
    water     water up to the volume's top: per-voxel buoyancy, linear and
              angular water drag, `vector` the water's own flow, `density`
              (the material table's scale; WATER_DENSITY by default)

Wind and updrafts are air: they go into the fixed-point wind field
(physics/wind.mojo) with the vorticles, and a body feels the air's velocity
as drag at WIND_DRAG times its mean projected area over its mass, on the part
not under water: a clay crumb flies, a stone block barely slides. Everything
else is float arithmetic under the simulation's float rules.

Volumes are authored in Blender (`ds_force`, `ds_wind_area`, `ds_updraft`,
`ds_water`) into the layer file's areas, or added by code (3.13 scripts).
They are simulation state: the FRCE blob holds them with the vorticles, the
ropes and which bodies are in water. The layer's flat water level (pre-2.12)
is one more water volume that is not stored: everything below the level.

Bodies are evaluated once a tick, before the solve, at their pose then: the
accelerations and drags are constant over the substeps (the field changes
slowly next to a substep; a floating body's bob is under 1 Hz). Buoyancy
sums every voxel of a voxel body (a plain box is cut into up to 3 x 3 x 3
cells): each cell's fill is how much of its height is below the surface,
its lift acts at its centre, so a body floats with its light side up and a
half-sunk plank tips. See docs/RESEARCH.md, Forces, wind and floating.
"""
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.serial import BlobWriter, BlobReader
from flint.physics.wind import (
    WindField, WindSource, Vorticle, FxShape, FxVec, to_fx, make_vorticle,
    SHAPE_BOX, SHAPE_SPHERE, SHAPE_CAPSULE, FALLOFF_NONE, FALLOFF_LINEAR, FALLOFF_SMOOTH,
)
from flint.physics.water import WATER_DENSITY, WATER_DRAG, NO_WATER
from flint.physics.filter import CAT_ALL
from flint.physics.defaults import WATER_ANGULAR_DRAG

comptime TARGET_FORCE: Int = 0
comptime TARGET_VELOCITY: Int = 1
comptime TARGET_UPDRAFT: Int = 2
comptime TARGET_WIND: Int = 3
comptime TARGET_CURRENT: Int = 4
comptime TARGET_RADIAL: Int = 5
comptime TARGET_WAKE: Int = 6
comptime TARGET_KILL: Int = 7
comptime TARGET_WATER: Int = 8
comptime FORCE_BLOB_VERSION: Int = 1
comptime MAX_FIELD_RATE: Float32 = 30.0      # no drag pulls faster than this (per second)
comptime FLAT_WATER_ID: Int = 0              # the layer's flat water level as a volume


def target_names() -> List[String]:
    return ["force", "velocity", "updraft", "wind", "current", "radial", "wake", "kill", "water"]


def shape_kind_names() -> List[String]:
    return ["box", "sphere", "capsule"]


def falloff_names() -> List[String]:
    return ["none", "linear", "smooth"]


def falloff_weight(kind: Int, t: Float32) -> Float32:
    if t >= 1 or t < 0:
        return 0
    if kind == FALLOFF_LINEAR:
        return 1 - t
    if kind == FALLOFF_SMOOTH:
        return 1 - t * t * (3 - 2 * t)
    return 1


struct FieldVolume(Copyable, ImplicitlyCopyable, Movable):
    var id: Int
    var target: Int
    var shape: Int
    var falloff: Int
    var center: Vec3
    var half: Vec3       # box
    var end: Vec3        # capsule: the other end of its segment
    var radius: Float32  # sphere, capsule
    var vector: Vec3
    var strength: Float32
    var rate: Float32
    var gust: Float32
    var cell: Float32
    var period: Int
    var density: Float32
    var drag: Float32
    var angular_drag: Float32
    var mask: UInt32     # the collision categories it acts on

    def __init__(out self):
        self.id = 0
        self.target = TARGET_FORCE
        self.shape = SHAPE_BOX
        self.falloff = FALLOFF_NONE
        self.center = Vec3()
        self.half = Vec3(1, 1, 1)
        self.end = Vec3()
        self.radius = 1
        self.vector = Vec3()
        self.strength = 0
        self.rate = 4
        self.gust = 0
        self.cell = 8
        self.period = 60
        self.density = WATER_DENSITY
        self.drag = WATER_DRAG
        self.angular_drag = WATER_ANGULAR_DRAG
        self.mask = CAT_ALL

    @staticmethod
    def box(id: Int, target: Int, lo: Vec3, hi: Vec3) -> Self:
        var v = Self()
        v.id = id
        v.target = target
        v.shape = SHAPE_BOX
        # Either two opposite corners will do.
        var a = Vec3(min(lo.x, hi.x), min(lo.y, hi.y), min(lo.z, hi.z))
        var b = Vec3(max(lo.x, hi.x), max(lo.y, hi.y), max(lo.z, hi.z))
        v.center = (a + b) * 0.5
        v.half = (b - a) * 0.5
        return v

    @staticmethod
    def sphere(id: Int, target: Int, center: Vec3, radius: Float32) -> Self:
        var v = Self()
        v.id = id
        v.target = target
        v.shape = SHAPE_SPHERE
        v.center = center
        v.radius = radius
        return v

    @staticmethod
    def capsule(id: Int, target: Int, a: Vec3, b: Vec3, radius: Float32) -> Self:
        var v = Self()
        v.id = id
        v.target = target
        v.shape = SHAPE_CAPSULE
        v.center = a
        v.end = b
        v.radius = radius
        return v

    def distance_t(self, p: Vec3) -> Float32:
        var d = p - self.center
        if self.shape == SHAPE_SPHERE:
            return d.length() / self.radius if self.radius > 0 else Float32(1)
        if self.shape == SHAPE_CAPSULE:
            if self.radius <= 0:
                return 1
            var axis = self.end - self.center
            var len2 = axis.length_squared()
            var s = Float32(0)
            if len2 > 0:
                s = max(Float32(0), min(Float32(1), dot(d, axis) / len2))
            return (p - (self.center + axis * s)).length() / self.radius
        if self.half.x <= 0 or self.half.y <= 0 or self.half.z <= 0:
            return 1
        return max(abs(d.x) / self.half.x, max(abs(d.y) / self.half.y, abs(d.z) / self.half.z))

    def weight(self, p: Vec3) -> Float32:
        return falloff_weight(self.falloff, self.distance_t(p))

    def inside(self, p: Vec3) -> Bool:
        return self.distance_t(p) < 1

    def bounds(self) -> Tuple[Vec3, Vec3]:
        if self.shape == SHAPE_SPHERE:
            var r = Vec3(self.radius, self.radius, self.radius)
            return (self.center - r, self.center + r)
        if self.shape == SHAPE_CAPSULE:
            var r = Vec3(self.radius, self.radius, self.radius)
            var lo = Vec3(min(self.center.x, self.end.x), min(self.center.y, self.end.y), min(self.center.z, self.end.z))
            var hi = Vec3(max(self.center.x, self.end.x), max(self.center.y, self.end.y), max(self.center.z, self.end.z))
            return (lo - r, hi + r)
        return (self.center - self.half, self.center + self.half)

    def surface(self) -> Float32:
        """A water volume's surface height: its top."""
        return self.bounds()[1].y

    def overlaps(self, lo: Vec3, hi: Vec3) -> Bool:
        var b = self.bounds()
        return lo.x <= b[1].x and hi.x >= b[0].x and lo.y <= b[1].y and hi.y >= b[0].y and lo.z <= b[1].z and hi.z >= b[0].z

    def is_air(self) -> Bool:
        return self.target == TARGET_WIND or self.target == TARGET_UPDRAFT

    def wind_source(self) -> WindSource:
        var s = WindSource()
        s.id = self.id
        s.shape.shape = self.shape
        s.shape.falloff = self.falloff
        s.shape.center = FxVec.of(self.center)
        s.shape.half = FxVec.of(self.half)
        s.shape.end = FxVec.of(self.end)
        s.shape.radius = to_fx(self.radius)
        s.base = FxVec.of(self.vector) if self.target == TARGET_WIND else FxVec(0, to_fx(self.strength), 0)
        s.gust = to_fx(self.gust)
        s.cell = to_fx(max(Float32(0.25), self.cell))
        s.period = max(1, self.period)
        s.seed = UInt64(self.id) * 0x9E3779B97F4A7C15
        return s

    def write(self, mut out: BlobWriter):
        out.integer(self.id)
        out.integer(self.target)
        out.integer(self.shape)
        out.integer(self.falloff)
        out.vec3(self.center)
        out.vec3(self.half)
        out.vec3(self.end)
        out.f32(self.radius)
        out.vec3(self.vector)
        out.f32(self.strength)
        out.f32(self.rate)
        out.f32(self.gust)
        out.f32(self.cell)
        out.integer(self.period)
        out.f32(self.density)
        out.f32(self.drag)
        out.f32(self.angular_drag)
        out.u32(self.mask)

    @staticmethod
    def read(mut input: BlobReader) raises -> Self:
        var v = Self()
        v.id = input.integer()
        v.target = input.integer()
        v.shape = input.integer()
        v.falloff = input.integer()
        v.center = input.vec3()
        v.half = input.vec3()
        v.end = input.vec3()
        v.radius = input.f32()
        v.vector = input.vec3()
        v.strength = input.f32()
        v.rate = input.f32()
        v.gust = input.f32()
        v.cell = input.f32()
        v.period = input.integer()
        v.density = input.f32()
        v.drag = input.f32()
        v.angular_drag = input.f32()
        v.mask = input.u32()
        if input.failed:
            raise Error("force blob: truncated FieldVolume")
        if v.target < 0 or v.target > TARGET_WATER or v.shape < 0 or v.shape > SHAPE_CAPSULE:
            raise Error("force blob: volume " + String(v.id) + " has an unknown target or shape")
        return v


def flat_water(level: Float32) -> FieldVolume:
    """The layer's flat water level as a volume: everything below it."""
    var v = FieldVolume.box(FLAT_WATER_ID, TARGET_WATER, Vec3(-1.0e6, level - 4096, -1.0e6), Vec3(1.0e6, level, 1.0e6))
    return v


def dot(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def cell_fill(volume: FieldVolume, c: Vec3, size: Float32) -> Float32:
    """How much of a cell of height `size` centred at c is under the water
    of `volume`: 0 when its centre is outside the water's extent."""
    var surface = volume.surface()
    var bottom = c.y - size * 0.5
    if bottom >= surface:
        return 0
    var probe = Vec3(c.x, min(c.y, surface - 0.001), c.z)
    if not volume.inside(probe):
        return 0
    return max(Float32(0), min(Float32(1), (surface - bottom) / size))


struct BodyEffect(Copyable, ImplicitlyCopyable, Movable):
    """What the fields do to one body this tick: an acceleration (and an
    angular one, from buoyancy's lever), and a drag toward a target velocity
    at `rate` per second (and an angular drag)."""
    var accel: Vec3
    var torque: Vec3       # buoyancy's torque about the centre of mass (world)
    var target_sum: Vec3   # sum of target * rate
    var rate: Float32
    var angular_rate: Float32
    var submerged: Float32 # share of the body under water, 0..1
    var water: Int         # the water volume's id, -1 when dry
    var surface: Float32
    var flow: Vec3         # the water's velocity there (its own flow and the currents)
    var kill: Bool
    var any: Bool

    def __init__(out self):
        self.accel = Vec3()
        self.torque = Vec3()
        self.target_sum = Vec3()
        self.rate = 0
        self.angular_rate = 0
        self.submerged = 0
        self.water = -1
        self.surface = NO_WATER
        self.flow = Vec3()
        self.kill = False
        self.any = False

    def drag(mut self, target: Vec3, rate: Float32):
        if rate <= 0:
            return
        self.target_sum = self.target_sum + target * rate
        self.rate += rate
        self.any = True

    def target(self) -> Vec3:
        if self.rate <= 0:
            return Vec3()
        return self.target_sum * (1.0 / self.rate)


struct ForceWorld(Copyable, Movable):
    """The field volumes, the wind (sources from the air volumes, and the
    vorticles), and their ids. `tick` is the game's tick (for gusts); it is
    set before each update and is not state."""
    var volumes: List[FieldVolume]
    var wind: WindField
    var next_vorticle_id: Int
    var tick: Int

    def __init__(out self):
        self.volumes = List[FieldVolume]()
        self.wind = WindField()
        self.next_vorticle_id = 1
        self.tick = 0

    def clear(mut self):
        self.volumes.clear()
        self.wind = WindField()
        self.next_vorticle_id = 1

    def rebuild_wind(mut self):
        var sources = List[WindSource]()
        for v in self.volumes:
            if v.is_air():
                sources.append(v.wind_source())
        self.wind.sources = sources^

    def add_volume(mut self, volume: FieldVolume) raises:
        """Adds a volume; ids are unique (a second volume with an id replaces it)."""
        if volume.id <= 0:
            raise Error("a field volume needs an id above 0")
        for index in range(len(self.volumes)):
            if self.volumes[index].id == volume.id:
                self.volumes[index] = volume
                self.rebuild_wind()
                return
        # Kept sorted by id: the order fields are summed in never depends on
        # the order they were added in.
        var at = len(self.volumes)
        for index in range(len(self.volumes)):
            if self.volumes[index].id > volume.id:
                at = index
                break
        self.volumes.insert(at, volume)
        self.rebuild_wind()

    def remove_volume(mut self, id: Int):
        var kept = List[FieldVolume]()
        for v in self.volumes:
            if v.id != id:
                kept.append(v)
        self.volumes = kept^
        self.rebuild_wind()

    def add_vorticle(mut self, center: Vec3, axis: Vec3, radius: Float32, swirl: Float32, burst: Float32, life: Int) -> Int:
        var id = self.next_vorticle_id
        self.next_vorticle_id += 1
        self.wind.add_vorticle(make_vorticle(id, center, axis, radius, swirl, burst, life))
        return id

    def wind_at(self, p: Vec3) -> Vec3:
        return self.wind.at_vec(p, self.tick)

    def is_default(self) -> Bool:
        return len(self.volumes) == 0 and len(self.wind.vorticles) == 0 and self.next_vorticle_id == 1

    def write(self, mut out: BlobWriter):
        out.integer(len(self.volumes))
        for v in self.volumes:
            v.write(out)
        out.integer(self.next_vorticle_id)
        out.integer(len(self.wind.vorticles))
        for v in self.wind.vorticles:
            v.write(out)

    def read(mut self, mut input: BlobReader) raises:
        self.clear()
        var count = input.integer()
        if input.failed or count < 0 or count > (1 << 16) or count > max(0, input.limit() - input.offset) // 120:
            raise Error("force blob lists " + String(count) + " volumes")
        for _ in range(count):
            self.volumes.append(FieldVolume.read(input))
        self.rebuild_wind()
        self.next_vorticle_id = input.integer()
        var vorticles = input.integer()
        if input.failed or vorticles < 0 or vorticles > (1 << 12) or vorticles > max(0, input.limit() - input.offset) // 96:
            raise Error("force blob lists " + String(vorticles) + " vorticles")
        for _ in range(vorticles):
            self.wind.vorticles.append(Vorticle.read(input))
        if input.failed:
            raise Error("force blob: truncated ForceWorld")
