"""Game-agnostic collision filter masks and contact probe primitives for Flint.

Decouples `flint.physics.broadphase` from voxel-specific collision shapes so
heightmap, triangle-mesh, SDF, and voxel worlds can share the parallel broadphase.
"""
from flint.core.math import Vec3

comptime CAT_TERRAIN: UInt32 = 1
comptime CAT_DEBRIS: UInt32 = 2
comptime CAT_PROP_PIECE: UInt32 = 4
comptime CAT_COSMETIC: UInt32 = 8
comptime CAT_STATIC: UInt32 = 16
comptime CAT_SENSOR: UInt32 = 32
comptime CAT_ALL: UInt32 = 0xFFFFFFFF


struct CollisionFilter(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """Box2D-style category/mask/group collision filter."""
    var category: UInt32
    var mask: UInt32
    var group: Int

    def __init__(out self, category: UInt32 = CAT_DEBRIS, mask: UInt32 = CAT_ALL, group: Int = 0):
        self.category = category
        self.mask = mask
        self.group = group

    def __eq__(self, other: Self) -> Bool:
        return self.category == other.category and self.mask == other.mask and self.group == other.group

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def touches(self, other: Self) -> Bool:
        if self.group != 0 and self.group == other.group:
            return self.group > 0
        return (self.category & other.mask) != 0 and (other.category & self.mask) != 0

    def touches_terrain(self) -> Bool:
        return (self.mask & CAT_TERRAIN) != 0

    def accepts(self, category: UInt32) -> Bool:
        return (self.mask & category) != 0


def cosmetic_filter() -> CollisionFilter:
    return CollisionFilter(CAT_COSMETIC, CAT_ALL & ~CAT_COSMETIC, 0)


struct Probe(Copyable, ImplicitlyCopyable, Movable):
    """A sample sphere query result against any signed-distance or collider surface."""
    var hit: Bool
    var depth: Float32
    var normal: Vec3
    var distance: Float32

    def __init__(
        out self,
        hit: Bool = False,
        depth: Float32 = 0.0,
        normal: Vec3 = Vec3(0.0, 1.0, 0.0),
        distance: Float32 = 0.0,
    ):
        self.hit = hit
        self.depth = depth
        self.normal = normal
        self.distance = distance
