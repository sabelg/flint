"""The wind mirror: the simulation's wind, baked for the presentation and the
GPU (package 2.12).

Every frame the presentation bakes the fixed-point wind field
(physics/wind.mojo) at the cells of a grid around the camera:
MIRROR_X x MIRROR_Y x MIRROR_Z cells MIRROR_CELL metres apart, its corner
snapped to whole cells so a still camera bakes the same cells. Each cell is
four Int16: the air's velocity in 1/256 m/s on x, y and z (+-128 m/s), and
its speed. The block is copied to a GPU buffer each frame on both backends
(GpuRaster.upload_wind; the layout is WIND_MIRROR_* in render/gpu_layout.mojo)
for the GPU passes that sway, blow and drift things (particles 2.7, spring
bones 3.3, vegetation 4.11 and 5.7); CPU presentation (the cosmetic ropes)
samples the same cells with `sample`, trilinear, as a shader would.

Presentation only: the simulation never reads the mirror, so nothing that
depends on it can change a replay. A calm world (no wind areas, updrafts or
vorticles) bakes nothing: the block stays zero and is not uploaded again.
"""
from std.math import floor
from flint.core.math import Vec3
from flint.physics.wind import WindField, FxVec

comptime MIRROR_X: Int = 16
comptime MIRROR_Y: Int = 8
comptime MIRROR_Z: Int = 16
comptime MIRROR_CELL: Float32 = 4.0
comptime MIRROR_CELLS: Int = MIRROR_X * MIRROR_Y * MIRROR_Z
comptime MIRROR_BYTES: Int = MIRROR_CELLS * 8
comptime MIRROR_SCALE: Float32 = 256.0  # Int16 units per m/s


struct WindMirror(Movable):
    var origin: Vec3        # the corner cell's centre
    var cells: List[Int16]  # 4 per cell, x fastest, then z, then y
    var tick: Int           # the tick it was baked for
    var calm: Bool          # all zero
    var dirty: Bool         # changed since the last upload
    var bakes: Int

    def __init__(out self):
        self.origin = Vec3()
        self.cells = List[Int16](length=MIRROR_CELLS * 4, fill=0)
        self.tick = -1
        self.calm = True
        self.dirty = True
        self.bakes = 0

    def bake(mut self, field: WindField, center: Vec3, tick: Int):
        """The field around `center` on `tick`; nothing when it is calm and
        was calm, or when nothing moved since the last bake."""
        var corner = Vec3(
            Float32(floor(center.x / MIRROR_CELL)) * MIRROR_CELL - MIRROR_CELL * Float32(MIRROR_X // 2),
            Float32(floor(center.y / MIRROR_CELL)) * MIRROR_CELL - MIRROR_CELL * Float32(MIRROR_Y // 2),
            Float32(floor(center.z / MIRROR_CELL)) * MIRROR_CELL - MIRROR_CELL * Float32(MIRROR_Z // 2),
        )
        if field.is_calm():
            if not self.calm:
                for k in range(len(self.cells)):
                    self.cells[k] = 0
                self.calm = True
                self.dirty = True
            self.origin = corner
            self.tick = tick
            return
        if tick == self.tick and corner.x == self.origin.x and corner.y == self.origin.y and corner.z == self.origin.z and not self.calm:
            return
        self.origin = corner
        self.tick = tick
        self.calm = False
        self.dirty = True
        self.bakes += 1
        var at = 0
        for y in range(MIRROR_Y):
            for z in range(MIRROR_Z):
                for x in range(MIRROR_X):
                    var p = corner + Vec3(Float32(x), Float32(y), Float32(z)) * MIRROR_CELL
                    var air = field.at(FxVec.of(p), tick).to_vec3()
                    self.cells[at] = pack(air.x)
                    self.cells[at + 1] = pack(air.y)
                    self.cells[at + 2] = pack(air.z)
                    self.cells[at + 3] = pack(air.length())
                    at += 4

    def cell_velocity(self, x: Int, y: Int, z: Int) -> Vec3:
        var cx = max(0, min(MIRROR_X - 1, x))
        var cy = max(0, min(MIRROR_Y - 1, y))
        var cz = max(0, min(MIRROR_Z - 1, z))
        var at = ((cy * MIRROR_Z + cz) * MIRROR_X + cx) * 4
        return Vec3(Float32(self.cells[at]), Float32(self.cells[at + 1]), Float32(self.cells[at + 2])) * (1.0 / MIRROR_SCALE)

    def sample(self, p: Vec3) -> Vec3:
        """The mirror's wind at p, trilinear between cell centres (clamped at
        the edges), as the GPU reads it."""
        if self.calm:
            return Vec3()
        var g = (p - self.origin) * (1.0 / MIRROR_CELL)
        var x0 = Int(floor(g.x))
        var y0 = Int(floor(g.y))
        var z0 = Int(floor(g.z))
        var fx = g.x - Float32(x0)
        var fy = g.y - Float32(y0)
        var fz = g.z - Float32(z0)
        var out = Vec3()
        for k in range(8):
            var dx = k & 1
            var dy = (k >> 1) & 1
            var dz = (k >> 2) & 1
            var w = (fx if dx == 1 else 1 - fx) * (fy if dy == 1 else 1 - fy) * (fz if dz == 1 else 1 - fz)
            out = out + self.cell_velocity(x0 + dx, y0 + dy, z0 + dz) * w
        return out


def pack(value: Float32) -> Int16:
    var scaled = value * MIRROR_SCALE
    return Int16(max(Float32(-32767), min(Float32(32767), scaled + (Float32(0.5) if scaled >= 0 else Float32(-0.5)))))
