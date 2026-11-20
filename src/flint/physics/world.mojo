"""Unified Rigid-Body World Coordinator (`PhysicsWorld`, v5).

Coordinates dynamic, kinematic, and static rigid bodies (`Sphere`, `Capsule`, `Box`,
`ConvexHull`, `Bvh8MeshCollider`, `HeightfieldCollider`, and `CompoundShape`),
persisted 8-wide SIMD `BroadphaseNode8` scene queries (`raycast`, `raycast_all`,
`shapecast_sphere`, `shapecast_capsule`, `shapecast_box`, `overlap_sphere`,
`overlap_aabb`), O(1) `id_to_slot` body lookup, discrete `Narrowphase` (explicitly
negating narrowphase `A -> B` normals into solver `B -> A` normals), `ContactCache`
warm starting, `IslandSet` union-find & fracture island splitting, `ConstraintGraph`
persistent 24-colouring, `Joint`s + `SixDofMotorJoint`s, Beaufort wave buoyancy &
`ForceWorld`, full-shape swept CCD (`swept_sphere_obb` / `swept_convex_cast`),
OBB/Sphere/Capsule `TriggerVolume` (`TRIGGER_ENTER`, `TRIGGER_STAY`, `TRIGGER_EXIT`),
and bit-exact `serialize` / `deserialize` snapshots for rollback & PIE.
"""
from std.collections import List
from std.math import sqrt, abs, min, max
from std.ffi import external_call
from flint.core.math import Vec3, Quat
from flint.core.detmath import set_sim_float_mode
from flint.core.detsort import stable_order, mix64
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.physics.defaults import JOINT_HERTZ, JOINT_DAMPING_RATIO
from flint.physics.filter import CollisionFilter, CAT_TERRAIN, CAT_STATIC, CAT_SENSOR, CAT_ALL
from flint.physics.broadphase import (
    BroadphaseInput,
    BroadphaseNode8,
    find_pairs,
    find_pairs_into_bvh8,
    boxes_overlap,
    _BpBinNode,
    _build_bp_bvh2,
    _collapse_bp_bvh8,
)
from flint.physics.narrowphase import (
    ContactManifold,
    ConvexHull,
    collide_sphere_sphere,
    collide_sphere_capsule,
    collide_sphere_box,
    collide_capsule_capsule,
    collide_capsule_box,
    collide_box_box,
    collide_convex_convex,
    reduce_manifold_candidates,
    ContactPoint as NarrowContactPoint,
    get_box_axes,
)
from flint.physics.bvh import Bvh8MeshCollider, MeshTriangle, EDGE_ACTIVE_ALL
from flint.physics.ccd import (
    CcdHit,
    gjk_closest_points,
    swept_sphere_moving_sphere,
    swept_sphere_obb,
)
from flint.physics.shapes_v5 import (
    CompoundChild,
    CompoundShape,
    HeightfieldCollider,
    SixDofMotorJoint,
    COMPOUND_CHILD_SPHERE,
    COMPOUND_CHILD_CAPSULE,
    COMPOUND_CHILD_BOX,
    COMPOUND_CHILD_CONVEX,
)
from flint.physics.solver_bodies import SolverBodies, F8, LANES, rotate_vec
from flint.physics.solver import (
    SolverSettings,
    Manifold,
    ContactPoint as SolverContactPoint,
    SolveResult,
    WideContacts,
    SolverStage,
    SolverRun,
    SPECULATIVE_DISTANCE,
    BODY_BATCHES_PER_BLOCK,
    CONTACT_BATCHES_PER_BLOCK,
    PARALLEL_MIN_BATCHES,
    SOLVER_WORD_STRIDE,
    STAGE_DONE,
    STAGE_INTEGRATE_VELOCITIES,
    STAGE_WARM,
    STAGE_SOLVE,
    STAGE_INTEGRATE_POSITIONS,
    STAGE_RELAX,
    STAGE_RESTITUTION,
    solver_softness,
    _condition_supported_mass_ratios,
    _restore_supported_mass_ratios,
    execute_block,
    pool_workers,
    count_cpus,
    solve,
    ccd_cpus,
)
from flint.physics.joints import (
    Joint,
    JointDef,
    JointRow,
    Softness,
    make_soft,
    JOINT_WELD,
    JOINT_DISTANCE,
    JOINT_WORLD,
    quat_conj,
    quat_product,
)
from flint.physics.islands import (
    IslandSet,
    ConstraintGraph,
    GRAPH_COLOURS,
    NO_ISLAND,
    ISLAND_SPLITS_PER_TICK,
    pair_key,
    manifold_key,
)
from flint.physics.events import (
    ContactCache,
    PhysicsEvent,
    Sensor,
    OTHER_TERRAIN,
    EVENT_SENSOR_BEGIN,
    EVENT_SENSOR_END,
    EVENT_SLEEP,
    EVENT_WAKE,
    EVENT_JOINT_BREAK,
    EVENT_WATER_ENTER,
    EVENT_WATER_EXIT,
    EVENT_KILL,
    static_other,
    sort_events,
)
from flint.physics.water import (
    NO_WATER,
    RHO_SEAWATER_SI,
    eval_sphere_buoyancy_drag,
    eval_box_buoyancy_drag,
)
from flint.physics.forces import (
    ForceWorld,
    TARGET_FORCE,
    TARGET_VELOCITY,
    TARGET_RADIAL,
    TARGET_KILL,
    TARGET_WATER,
    MAX_FIELD_RATE,
)

comptime BODY_SHAPE_SPHERE: Int = 0
comptime BODY_SHAPE_CAPSULE: Int = 1
comptime BODY_SHAPE_BOX: Int = 2
comptime BODY_SHAPE_CONVEX_HULL: Int = 3
comptime BODY_SHAPE_MESH_BVH8: Int = 4
comptime BODY_SHAPE_HEIGHTFIELD: Int = 5
comptime BODY_SHAPE_COMPOUND: Int = 6

comptime BODY_MOTION_STATIC: UInt8 = 0
comptime BODY_MOTION_KINEMATIC: UInt8 = 1
comptime BODY_MOTION_DYNAMIC: UInt8 = 2

comptime TRIGGER_SHAPE_BOX: Int = 0
comptime TRIGGER_SHAPE_SPHERE: Int = 1
comptime TRIGGER_SHAPE_CAPSULE: Int = 2

comptime TRIGGER_ENTER: Int = 20
comptime TRIGGER_STAY: Int = 21
comptime TRIGGER_EXIT: Int = 22

comptime SLEEP_TICKS_DEFAULT: Int = 60
comptime SLEEP_LIN_SPEED: Float32 = 0.05
comptime SLEEP_ANG_SPEED: Float32 = 0.08

comptime WORLD_BLOB_TAG: UInt32 = fourcc("PHYW")
comptime WORLD_BLOB_VERSION: Int = 1


@always_inline
def _safe_inv(d: Float32) -> Float32:
    if abs(d) > Float32(1.0e-8):
        return Float32(1.0) / d
    return Float32(1.0e30) if d >= Float32(0.0) else Float32(-1.0e30)


def compute_shape_inertia(shape: Int, mass: Float32, radius: Float32, half_height: Float32, half_extents: Vec3) -> Vec3:
    """Computes principal moments of inertia `(Ix, Iy, Iz)` for analytical shapes."""
    var m = max(Float32(1.0e-4), mass)
    if shape == BODY_SHAPE_SPHERE:
        var r = max(Float32(0.01), radius)
        var i_s = Float32(0.4) * m * r * r
        return Vec3(i_s, i_s, i_s)
    if shape == BODY_SHAPE_CAPSULE:
        var r = max(Float32(0.01), radius)
        var h_cyl = Float32(2.0) * max(Float32(0.0), half_height)
        var r2 = r * r
        var iy = Float32(0.5) * m * r2
        var i_perp = m * (Float32(0.25) * r2 + (Float32(1.0) / Float32(12.0)) * (h_cyl + Float32(1.5) * r) * (h_cyl + Float32(1.5) * r))
        return Vec3(i_perp, iy, i_perp)
    var hx = max(Float32(0.01), half_extents.x)
    var hy = max(Float32(0.01), half_extents.y)
    var hz = max(Float32(0.01), half_extents.z)
    var sx2 = Float32(4.0) * hx * hx
    var sy2 = Float32(4.0) * hy * hy
    var sz2 = Float32(4.0) * hz * hz
    var k = m * Float32(1.0 / 12.0)
    return Vec3(k * (sy2 + sz2), k * (sx2 + sz2), k * (sx2 + sy2))


def _hull_half_extents(hull: ConvexHull) -> Vec3:
    if len(hull.vertices) == 0:
        return Vec3(0.5, 0.5, 0.5)
    var mx = Float32(0.01)
    var my = Float32(0.01)
    var mz = Float32(0.01)
    for i in range(len(hull.vertices)):
        var v = hull.vertices[i]
        mx = max(mx, abs(v.x))
        my = max(my, abs(v.y))
        mz = max(mz, abs(v.z))
    return Vec3(mx, my, mz)


def _make_box_hull(half: Vec3) -> ConvexHull:
    var verts = List[Vec3](capacity=8)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append(Vec3(Float32(sx) * half.x, Float32(sy) * half.y, Float32(sz) * half.z))
    return ConvexHull(verts)


def _make_segment_hull(half_height: Float32) -> ConvexHull:
    var verts = List[Vec3](capacity=2)
    verts.append(Vec3(0.0, -half_height, 0.0))
    verts.append(Vec3(0.0, half_height, 0.0))
    return ConvexHull(verts)


def _make_point_hull() -> ConvexHull:
    var verts = List[Vec3](capacity=1)
    verts.append(Vec3())
    return ConvexHull(verts)


def _make_capsule_hull(radius: Float32, half_height: Float32) -> ConvexHull:
    var verts = List[Vec3](capacity=26)
    var dirs = [
        Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, -1),
        Vec3(0.70710678, 0, 0.70710678), Vec3(-0.70710678, 0, 0.70710678),
        Vec3(0.70710678, 0, -0.70710678), Vec3(-0.70710678, 0, -0.70710678),
        Vec3(0.57735027, 0.57735027, 0.57735027), Vec3(-0.57735027, 0.57735027, 0.57735027),
        Vec3(0.57735027, 0.57735027, -0.57735027), Vec3(-0.57735027, 0.57735027, -0.57735027),
    ]
    verts.append(Vec3(0, half_height + radius, 0))
    verts.append(Vec3(0, -half_height - radius, 0))
    for i in range(8):
        var d = dirs[i]
        verts.append(Vec3(d.x * radius, half_height, d.z * radius))
        verts.append(Vec3(d.x * radius, -half_height, d.z * radius))
    for i in range(8, 12):
        var d = dirs[i]
        verts.append(Vec3(d.x * radius, half_height + d.y * radius, d.z * radius))
        verts.append(Vec3(d.x * radius, -half_height - d.y * radius, d.z * radius))
    return ConvexHull(verts)


def _swept_inflated_hull_cast(
    hull_a: ConvexHull,
    inflate_r: Float32,
    pos_a0: Vec3,
    pos_a1: Vec3,
    rot_a: Quat,
    hull_b: ConvexHull,
    pos_b0: Vec3,
    pos_b1: Vec3,
    rot_b: Quat,
) -> CcdHit:
    """Exact GJK Conservative Advancement for a spherically-inflated `hull_a` (`radius = inflate_r`)
    sweeping `pos_a0 -> pos_a1` against `hull_b` (`pos_b0 -> pos_b1`).
    Returns outward normal pointing from `hull_b` toward `hull_a`."""
    if len(hull_a.vertices) == 0 or len(hull_b.vertices) == 0:
        return CcdHit()

    var dpa = pos_a1 - pos_a0
    var dpb = pos_b1 - pos_b0
    var rel_move = dpa - dpb
    var move_len = rel_move.length()

    var gjk0 = gjk_closest_points(hull_a, pos_a0, rot_a, hull_b, pos_b0, rot_b)
    if gjk0[0] or gjk0[3] <= inflate_r + Float32(1.0e-5):
        var n0: Vec3
        var pt0: Vec3
        var pen_depth: Float32
        if not gjk0[0] and gjk0[3] > Float32(1.0e-6):
            n0 = (gjk0[1] - gjk0[2]) * (Float32(1.0) / gjk0[3])
            pt0 = gjk0[2]
            pen_depth = max(Float32(0.0), inflate_r - gjk0[3])
        else:
            var m0 = collide_convex_convex(hull_a, pos_a0, rot_a, hull_b, pos_b0, rot_b)
            if m0.has_collision:
                n0 = (-m0.normal).normalized()
                pt0 = m0.p0.point
                pen_depth = m0.p0.depth + inflate_r
            else:
                var diff = pos_a0 - pos_b0
                n0 = diff.normalized() if diff.length_squared() > Float32(1.0e-8) else Vec3(0, 1, 0)
                pt0 = pos_a0 - n0 * inflate_r
                pen_depth = inflate_r
        if pen_depth <= Float32(5.0e-3) and move_len > Float32(1.0e-7) and rel_move.dot(n0) >= Float32(-1.0e-5):
            return CcdHit()
        return CcdHit(True, Float32(0.0), pt0, n0)

    if move_len < Float32(1.0e-7):
        return CcdHit()

    var lam = Float32(0.0)
    var hit_normal = (gjk0[1] - gjk0[2]) * (Float32(1.0) / gjk0[3])
    var hit_point = gjk0[2]
    var hit_tol = max(Float32(1.5e-4), inflate_r * Float32(5.0e-4))

    for _ in range(36):
        var cur_pa = pos_a0 + dpa * lam
        var cur_pb = pos_b0 + dpb * lam
        var gjk = gjk_closest_points(hull_a, cur_pa, rot_a, hull_b, cur_pb, rot_b)
        var overlapping = gjk[0]
        var ca = gjk[1]
        var cb = gjk[2]
        var dist = gjk[3]
        var gap = dist - inflate_r

        if overlapping or gap <= hit_tol:
            if not overlapping and dist > Float32(1.0e-6):
                hit_normal = (ca - cb) * (Float32(1.0) / dist)
                hit_point = cb
            return CcdHit(True, lam, hit_point, hit_normal)

        var n_ab = (cb - ca) * (Float32(1.0) / dist)
        var v_close = rel_move.dot(n_ab)
        if v_close <= Float32(1.0e-7):
            return CcdHit()

        var step = gap / v_close
        lam += step
        if lam > Float32(1.0):
            return CcdHit()

        hit_normal = -n_ab
        hit_point = cb + dpb * step

    return CcdHit()


@fieldwise_init
struct SceneCastHit(Copyable, ImplicitlyCopyable, Movable):
    """Unified result of a world raycast or shapecast query."""
    var hit: Bool
    var fraction: Float32
    var distance: Float32
    var point: Vec3
    var normal: Vec3
    var body_slot: Int
    var body_id: Int
    var sub_index: Int
    var material_id: Int

    def __init__(out self):
        self.hit = False
        self.fraction = Float32(1.0)
        self.distance = Float32(1.0e30)
        self.point = Vec3()
        self.normal = Vec3(0.0, 1.0, 0.0)
        self.body_slot = -1
        self.body_id = -1
        self.sub_index = -1
        self.material_id = 0


@fieldwise_init
struct TriggerEvent(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var trigger_id: Int
    var body_id: Int
    var point: Vec3


struct TriggerVolume(Copyable, ImplicitlyCopyable, Movable):
    """OBB, Sphere, or Capsule non-blocking trigger volume emitting
    `TRIGGER_ENTER`, `TRIGGER_STAY`, and `TRIGGER_EXIT` events."""
    var id: Int
    var shape: Int
    var pos: Vec3
    var rot: Quat
    var half_extents: Vec3
    var radius: Float32
    var half_height: Float32
    var filter: CollisionFilter
    var active: Bool

    def __init__(
        out self,
        id: Int = 1,
        shape: Int = TRIGGER_SHAPE_BOX,
        pos: Vec3 = Vec3(),
        rot: Quat = Quat(),
        half_extents: Vec3 = Vec3(1, 1, 1),
        radius: Float32 = 1.0,
        half_height: Float32 = 0.5,
        filter: CollisionFilter = CollisionFilter(CAT_SENSOR, CAT_ALL, 0),
        active: Bool = True,
    ):
        self.id = id
        self.shape = shape
        self.pos = pos
        self.rot = rot.normalized()
        self.half_extents = half_extents
        self.radius = radius
        self.half_height = half_height
        self.filter = filter
        self.active = active

    @staticmethod
    def obb(
        id: Int,
        pos: Vec3,
        half_extents: Vec3,
        rot: Quat = Quat(),
        filter: CollisionFilter = CollisionFilter(CAT_SENSOR, CAT_ALL, 0),
    ) -> Self:
        return Self(id, TRIGGER_SHAPE_BOX, pos, rot, half_extents, half_extents.length(), 0.0, filter, True)

    @staticmethod
    def sphere(
        id: Int,
        pos: Vec3,
        radius: Float32,
        filter: CollisionFilter = CollisionFilter(CAT_SENSOR, CAT_ALL, 0),
    ) -> Self:
        return Self(id, TRIGGER_SHAPE_SPHERE, pos, Quat(), Vec3(radius, radius, radius), radius, 0.0, filter, True)

    @staticmethod
    def capsule(
        id: Int,
        pos: Vec3,
        radius: Float32,
        half_height: Float32,
        rot: Quat = Quat(),
        filter: CollisionFilter = CollisionFilter(CAT_SENSOR, CAT_ALL, 0),
    ) -> Self:
        return Self(
            id, TRIGGER_SHAPE_CAPSULE, pos, rot,
            Vec3(radius, half_height + radius, radius), radius, half_height, filter, True,
        )

    def tight_aabb(self) -> Tuple[Vec3, Vec3]:
        if self.shape == TRIGGER_SHAPE_SPHERE:
            var rv = Vec3(self.radius, self.radius, self.radius)
            return (self.pos - rv, self.pos + rv)
        if self.shape == TRIGGER_SHAPE_CAPSULE:
            var ax = self.rot.rotate(Vec3(0, self.half_height, 0))
            var a = self.pos - ax
            var b = self.pos + ax
            var rv = Vec3(self.radius, self.radius, self.radius)
            return (
                Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)) - rv,
                Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)) + rv,
            )
        var axes = get_box_axes(self.rot)
        var wx = abs(axes[0].x) * self.half_extents.x + abs(axes[1].x) * self.half_extents.y + abs(axes[2].x) * self.half_extents.z
        var wy = abs(axes[0].y) * self.half_extents.x + abs(axes[1].y) * self.half_extents.y + abs(axes[2].y) * self.half_extents.z
        var wz = abs(axes[0].z) * self.half_extents.x + abs(axes[1].z) * self.half_extents.y + abs(axes[2].z) * self.half_extents.z
        var ext = Vec3(wx, wy, wz)
        return (self.pos - ext, self.pos + ext)


struct RigidBody(Copyable, Movable):
    var id: Int
    var shape: Int
    var shape_index: Int
    var motion_mode: UInt8
    var radius: Float32
    var half_height: Float32
    var half_extents: Vec3
    var pos: Vec3
    var rot: Quat
    var linear_vel: Vec3
    var angular_vel: Vec3
    var mass: Float32
    var inv_mass: Float32
    var local_inertia: Vec3
    var friction: Float32
    var restitution: Float32
    var rolling_resistance: Float32
    var linear_damping: Float32
    var angular_damping: Float32
    var gravity_scale: Float32
    var filter: CollisionFilter
    var is_static: Bool
    var active: Bool
    var awake: Bool
    var can_sleep: Bool
    var sleep_ticks: Int
    var in_water: Bool
    var water_volume_id: Int
    var material: Int
    var enable_ccd: Bool
    var wind_area_over_mass: Float32
    var force_accum: Vec3
    var torque_accum: Vec3

    def __init__(out self, id: Int = 1):
        self.id = id
        self.shape = BODY_SHAPE_BOX
        self.shape_index = -1
        self.motion_mode = BODY_MOTION_DYNAMIC
        self.radius = 0.5
        self.half_height = 0.5
        self.half_extents = Vec3(0.5, 0.5, 0.5)
        self.pos = Vec3()
        self.rot = Quat()
        self.linear_vel = Vec3()
        self.angular_vel = Vec3()
        self.mass = 1.0
        self.inv_mass = 1.0
        self.local_inertia = Vec3(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
        self.friction = 0.6
        self.restitution = 0.1
        self.rolling_resistance = 0.0
        self.linear_damping = 0.02
        self.angular_damping = 0.04
        self.gravity_scale = 1.0
        self.filter = CollisionFilter()
        self.is_static = False
        self.active = True
        self.awake = True
        self.can_sleep = True
        self.sleep_ticks = 0
        self.in_water = False
        self.water_volume_id = -1
        self.material = 0
        self.enable_ccd = False
        self.wind_area_over_mass = 0.0
        self.force_accum = Vec3()
        self.torque_accum = Vec3()

    def set_motion_mode(mut self, mode: UInt8, dynamic_mass: Float32 = -1.0):
        self.motion_mode = mode
        if mode == BODY_MOTION_STATIC:
            self.is_static = True
            self.inv_mass = 0.0
            self.local_inertia = Vec3()
            self.linear_vel = Vec3()
            self.angular_vel = Vec3()
            self.awake = False
        elif mode == BODY_MOTION_KINEMATIC:
            self.is_static = True
            self.inv_mass = 0.0
            self.local_inertia = Vec3()
            self.can_sleep = False
            self.awake = True
        else:
            self.is_static = False
            var m = dynamic_mass if dynamic_mass > Float32(0.0) else (self.mass if self.mass > Float32(0.0) else Float32(1.0))
            self.mass = m
            self.inv_mass = Float32(1.0) / m
            self.local_inertia = compute_shape_inertia(self.shape, m, self.radius, self.half_height, self.half_extents)
            self.awake = True

    def capsule_endpoints(self) -> Tuple[Vec3, Vec3]:
        var axis = self.rot.rotate(Vec3(0, self.half_height, 0))
        return (self.pos - axis, self.pos + axis)

    def tight_aabb(self, hulls: List[ConvexHull], meshes: List[Bvh8MeshCollider]) -> Tuple[Vec3, Vec3]:
        var empty_hf = List[HeightfieldCollider]()
        var empty_cmp = List[CompoundShape]()
        return self.tight_aabb_full(hulls, meshes, empty_hf, empty_cmp)

    def tight_aabb_full(
        self,
        hulls: List[ConvexHull],
        meshes: List[Bvh8MeshCollider],
        heightfields: List[HeightfieldCollider],
        compounds: List[CompoundShape],
    ) -> Tuple[Vec3, Vec3]:
        if self.shape == BODY_SHAPE_SPHERE:
            var r = Vec3(self.radius, self.radius, self.radius)
            return (self.pos - r, self.pos + r)
        if self.shape == BODY_SHAPE_CAPSULE:
            var ends = self.capsule_endpoints()
            var r = Vec3(self.radius, self.radius, self.radius)
            var lo = Vec3(min(ends[0].x, ends[1].x), min(ends[0].y, ends[1].y), min(ends[0].z, ends[1].z))
            var hi = Vec3(max(ends[0].x, ends[1].x), max(ends[0].y, ends[1].y), max(ends[0].z, ends[1].z))
            return (lo - r, hi + r)
        if self.shape == BODY_SHAPE_CONVEX_HULL and self.shape_index >= 0 and self.shape_index < len(hulls):
            ref h = hulls[self.shape_index]
            if len(h.vertices) > 0:
                var lo = Vec3(1.0e30, 1.0e30, 1.0e30)
                var hi = Vec3(-1.0e30, -1.0e30, -1.0e30)
                for i in range(len(h.vertices)):
                    var wp = self.pos + self.rot.rotate(h.vertices[i])
                    lo.x = min(lo.x, wp.x); lo.y = min(lo.y, wp.y); lo.z = min(lo.z, wp.z)
                    hi.x = max(hi.x, wp.x); hi.y = max(hi.y, wp.y); hi.z = max(hi.z, wp.z)
                return (lo, hi)
        if self.shape == BODY_SHAPE_MESH_BVH8 and self.shape_index >= 0 and self.shape_index < len(meshes):
            ref mesh = meshes[self.shape_index]
            if len(mesh.triangles) > 0:
                var mlo = Vec3(1.0e30, 1.0e30, 1.0e30)
                var mhi = Vec3(-1.0e30, -1.0e30, -1.0e30)
                if mesh.root_node >= 0 and mesh.root_node < len(mesh.nodes):
                    ref rn = mesh.nodes[mesh.root_node]
                    comptime for c in range(8):
                        if rn.child_indices[c] != 0 or rn.child_counts[c] != 0:
                            mlo.x = min(mlo.x, rn.min_x[c]); mlo.y = min(mlo.y, rn.min_y[c]); mlo.z = min(mlo.z, rn.min_z[c])
                            mhi.x = max(mhi.x, rn.max_x[c]); mhi.y = max(mhi.y, rn.max_y[c]); mhi.z = max(mhi.z, rn.max_z[c])
                else:
                    for ti in range(len(mesh.triangles)):
                        var tb = mesh.triangles[ti].bounds()
                        mlo.x = min(mlo.x, tb[0].x); mlo.y = min(mlo.y, tb[0].y); mlo.z = min(mlo.z, tb[0].z)
                        mhi.x = max(mhi.x, tb[1].x); mhi.y = max(mhi.y, tb[1].y); mhi.z = max(mhi.z, tb[1].z)
                var mc = (mlo + mhi) * Float32(0.5)
                var mh = (mhi - mlo) * Float32(0.5)
                var axes = get_box_axes(self.rot)
                var wx = abs(axes[0].x) * mh.x + abs(axes[1].x) * mh.y + abs(axes[2].x) * mh.z
                var wy = abs(axes[0].y) * mh.x + abs(axes[1].y) * mh.y + abs(axes[2].y) * mh.z
                var wz = abs(axes[0].z) * mh.x + abs(axes[1].z) * mh.y + abs(axes[2].z) * mh.z
                var wc = self.pos + self.rot.rotate(mc)
                var wext = Vec3(wx, wy, wz)
                return (wc - wext, wc + wext)
            return (self.pos - Vec3(1, 1, 1), self.pos + Vec3(1, 1, 1))
        if self.shape == BODY_SHAPE_HEIGHTFIELD and self.shape_index >= 0 and self.shape_index < len(heightfields):
            var hb = heightfields[self.shape_index].bounds()
            var mc = (hb[0] + hb[1]) * Float32(0.5)
            var mh = (hb[1] - hb[0]) * Float32(0.5)
            var axes = get_box_axes(self.rot)
            var wx = abs(axes[0].x) * mh.x + abs(axes[1].x) * mh.y + abs(axes[2].x) * mh.z
            var wy = abs(axes[0].y) * mh.x + abs(axes[1].y) * mh.y + abs(axes[2].y) * mh.z
            var wz = abs(axes[0].z) * mh.x + abs(axes[1].z) * mh.y + abs(axes[2].z) * mh.z
            var wc = self.pos + self.rot.rotate(mc)
            var wext = Vec3(wx, wy, wz)
            return (wc - wext, wc + wext)
        if self.shape == BODY_SHAPE_COMPOUND and self.shape_index >= 0 and self.shape_index < len(compounds):
            return compounds[self.shape_index].tight_aabb(self.pos, self.rot)
        var axes = get_box_axes(self.rot)
        var wx = abs(axes[0].x) * self.half_extents.x + abs(axes[1].x) * self.half_extents.y + abs(axes[2].x) * self.half_extents.z
        var wy = abs(axes[0].y) * self.half_extents.x + abs(axes[1].y) * self.half_extents.y + abs(axes[2].y) * self.half_extents.z
        var wz = abs(axes[0].z) * self.half_extents.x + abs(axes[1].z) * self.half_extents.y + abs(axes[2].z) * self.half_extents.z
        var ext = Vec3(wx, wy, wz)
        return (self.pos - ext, self.pos + ext)


def _negate_manifold_normals(mut m: ContactManifold):
    m.normal = -m.normal
    if m.count >= 1:
        m.p0.normal = -m.p0.normal
    if m.count >= 2:
        m.p1.normal = -m.p1.normal
    if m.count >= 3:
        m.p2.normal = -m.p2.normal
    if m.count >= 4:
        m.p3.normal = -m.p3.normal


struct _S6SolverCtx(Movable):
    """Sub-stepped solver context interleaving `SixDofMotorJoint` warm/solve/relax inside every substep."""
    var base: SolverRun
    var s6_ptr: Int
    var s6_indices: List[Int]
    var s6_sa: List[Int]
    var s6_sb: List[Int]
    var broken_s6_ptr: Int

    def __init__(out self):
        self.base = SolverRun()
        self.s6_ptr = 0
        self.s6_indices = List[Int]()
        self.s6_sa = List[Int]()
        self.s6_sb = List[Int]()
        self.broken_s6_ptr = 0


def _execute_s6_block(ctx: _S6SolverCtx, stage: SolverStage, block: Int):
    execute_block(ctx.base, stage, block)
    if stage.joints_only and ctx.s6_ptr != 0:
        ref s6_list = Pointer[List[SixDofMotorJoint], origin=MutAnyOrigin](unsafe_from_address=ctx.s6_ptr)[unsafe_offset=0]
        ref bodies = Pointer[SolverBodies, origin=MutAnyOrigin](unsafe_from_address=ctx.base.bodies)[unsafe_offset=0]
        for k in range(len(ctx.s6_indices)):
            var ji = ctx.s6_indices[k]
            if s6_list[ji].broken:
                continue
            var sa = ctx.s6_sa[k]
            var sb = ctx.s6_sb[k]
            if stage.kind == STAGE_WARM:
                s6_list[ji].warm_start(sa, sb, bodies)
            elif stage.kind == STAGE_SOLVE:
                s6_list[ji].solve_step(sa, sb, bodies, ctx.base.h, True)
                if s6_list[ji].broken and ctx.broken_s6_ptr != 0:
                    Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=ctx.broken_s6_ptr)[unsafe_offset=0].append(ji)
            elif stage.kind == STAGE_RELAX:
                s6_list[ji].solve_step(sa, sb, bodies, ctx.base.h, False)


def _work_s6_stage(ctx: _S6SolverCtx, index: Int):
    var stage = ctx.base.stages[index]
    var claimed = ctx.base.word(1 + index * 2)
    var done = ctx.base.word(2 + index * 2)
    while True:
        var block = Int(claimed[unsafe_offset=0].fetch_add(1))
        if block >= stage.blocks:
            return
        _execute_s6_block(ctx, stage, block)
        _ = done[unsafe_offset=0].fetch_add(1)


def _lead_s6(ctx: _S6SolverCtx):
    for index in range(len(ctx.base.stages)):
        ctx.base.word(0)[unsafe_offset=0].store(Int64(index + 1))
        _work_s6_stage(ctx, index)
        var done = ctx.base.word(2 + index * 2)
        var blocks = Int64(ctx.base.stages[index].blocks)
        var spins = 0
        while done[unsafe_offset=0].load() < blocks:
            spins += 1
            if spins > 2000:
                _ = external_call["sched_yield", Int32]()
    ctx.base.word(0)[unsafe_offset=0].store(Int64(STAGE_DONE))


def _help_s6(ctx: _S6SolverCtx):
    var cpu = Int(external_call["sched_getcpu", Int32]())
    if cpu >= 0 and cpu < len(ctx.base.cpus) and not ctx.base.cpus[cpu]:
        return
    var seen = Int64(0)
    var spins = 0
    while True:
        var current = ctx.base.word(0)[unsafe_offset=0].load()
        if current >= Int64(STAGE_DONE):
            return
        if current != seen and current > 0:
            seen = current
            _work_s6_stage(ctx, Int(current) - 1)
            spins = 0
            continue
        spins += 1
        if spins > 2000:
            _ = external_call["sched_yield", Int32]()
        if spins > 200000:
            return


def _run_s6_solver_task(context: Int, chunk: Int):
    ref ctx = Pointer[_S6SolverCtx, origin=MutAnyOrigin](unsafe_from_address=context)[unsafe_offset=0]
    if chunk == 0:
        _lead_s6(ctx)
    else:
        _help_s6(ctx)


def _solve_with_six_dof(
    mut bodies: SolverBodies,
    mut manifolds: List[Manifold],
    mut joints: List[Joint],
    joint_rows: List[JointRow],
    mut six_dof_joints: List[SixDofMotorJoint],
    s6_indices: List[Int],
    s6_sa: List[Int],
    s6_sb: List[Int],
    mut broken_s6_out: List[Int],
    settings: SolverSettings,
    dt: Float32,
    pool_address: Int,
    cpus: List[Bool],
) -> SolveResult:
    """Executes the full Soft-Step solver with `SixDofMotorJoint`s solved inside every `h = dt / substeps` substep."""
    set_sim_float_mode()
    var result = SolveResult()
    var substeps = max(1, settings.substeps)
    var h = dt / Float32(substeps)
    var inv_h = Float32(1.0) / h
    var soft = solver_softness(settings, h)

    var saved_masses = _condition_supported_mass_ratios(bodies, manifolds)

    var serial_list = List[Int]()
    var colour_lists = List[List[Int]]()
    for _ in range(GRAPH_COLOURS):
        colour_lists.append(List[Int]())
    for index in range(len(manifolds)):
        var colour = manifolds[index].colour
        if colour >= GRAPH_COLOURS or colour < 0:
            serial_list.append(index)
            result.overflow += 1
        else:
            colour_lists[colour].append(index)
    if settings.force_serial:
        for colour in range(GRAPH_COLOURS):
            for index in colour_lists[colour]:
                serial_list.append(index)
            colour_lists[colour].clear()

    var total = len(serial_list)
    for colour in range(GRAPH_COLOURS):
        total += (len(colour_lists[colour]) + LANES - 1) // LANES
    var contacts = WideContacts()
    contacts.reset(total, bodies.static_index)
    var batch = 0
    for index in serial_list:
        contacts.put(
            batch, 0, index, manifolds[index], manifolds[index].ia, manifolds[index].ib,
            bodies, soft[0], soft[1], settings.warm_starting, settings.rolling_resistance,
        )
        batch += 1
    var serial_batches = batch
    var colour_first = List[Int](length=GRAPH_COLOURS, fill=0)
    var colour_batches = List[Int](length=GRAPH_COLOURS, fill=0)
    for colour in range(GRAPH_COLOURS):
        colour_first[colour] = batch
        var lane = 0
        for index in colour_lists[colour]:
            contacts.put(
                batch, lane, index, manifolds[index], manifolds[index].ia, manifolds[index].ib,
                bodies, soft[0], soft[1], settings.warm_starting, settings.rolling_resistance,
            )
            lane += 1
            if lane == LANES:
                lane = 0
                batch += 1
        if lane > 0:
            batch += 1
        colour_batches[colour] = batch - colour_first[colour]
    result.batches = batch

    var rows = List[JointRow]()
    var nyquist = Float32(0.25) / h
    for row in joint_rows:
        if row.joint < 0 or row.joint >= len(joints):
            continue
        if joints[row.joint].broken:
            continue
        var j = joints[row.joint]
        var linear_hertz = min(j.linear_hertz if j.linear_hertz > 0 else JOINT_HERTZ, nyquist)
        var angular_hertz = min(j.angular_hertz if j.angular_hertz > 0 else JOINT_HERTZ, nyquist)
        var damping = j.damping if j.damping > 0 else JOINT_DAMPING_RATIO
        rows.append(
            JointRow(
                row.joint, row.a, row.b,
                make_soft(linear_hertz, damping, h),
                make_soft(angular_hertz, damping, h),
            )
        )
        joints[row.joint].force_impulse = Vec3()
        joints[row.joint].torque_impulse = Vec3()
        if not settings.warm_starting:
            joints[row.joint].linear_impulse = Vec3()
            joints[row.joint].angular_impulse = Vec3()
            joints[row.joint].axial_impulse = 0
            joints[row.joint].motor_impulse = 0
            joints[row.joint].lower_impulse = 0
            joints[row.joint].upper_impulse = 0
            joints[row.joint].swing_impulse = 0

    if not settings.warm_starting:
        for k in range(len(s6_indices)):
            six_dof_joints[s6_indices[k]].reset_warm_impulses()

    var broken_indices = List[Int]()
    var ctx = _S6SolverCtx()
    ctx.base.bodies = Int(Pointer(to=bodies))
    ctx.base.contacts = Int(Pointer(to=contacts))
    ctx.base.joints = Int(Pointer(to=joints)) if len(rows) > 0 else 0
    ctx.base.joint_rows = Int(Pointer(to=rows))
    ctx.base.broken_out = Int(Pointer(to=broken_indices))
    ctx.base.h = h
    ctx.base.inv_h = inv_h
    ctx.base.settings = settings
    ctx.base.cpus = cpus.copy()
    ctx.s6_ptr = Int(Pointer(to=six_dof_joints))
    ctx.s6_indices = s6_indices.copy()
    ctx.s6_sa = s6_sa.copy()
    ctx.s6_sb = s6_sb.copy()
    ctx.broken_s6_ptr = Int(Pointer(to=broken_s6_out))

    var body_batches = bodies.padded // LANES
    var body_blocks = (body_batches + BODY_BATCHES_PER_BLOCK - 1) // BODY_BATCHES_PER_BLOCK
    var has_any_joints = len(rows) > 0 or len(s6_indices) > 0
    var kinds = [STAGE_WARM, STAGE_SOLVE, STAGE_RELAX]
    for _ in range(substeps):
        if body_batches > 0:
            ctx.base.stages.append(SolverStage(STAGE_INTEGRATE_VELOCITIES, 0, body_batches, body_blocks, False, False))
        for pass_index in range(3):
            var kind = kinds[pass_index]
            if kind == STAGE_RELAX:
                if body_batches > 0:
                    ctx.base.stages.append(SolverStage(STAGE_INTEGRATE_POSITIONS, 0, body_batches, body_blocks, False, False))
                if not settings.relax:
                    continue
            if serial_batches > 0:
                ctx.base.stages.append(SolverStage(kind, 0, serial_batches, 1, True, False))
            for colour in range(GRAPH_COLOURS):
                if colour_batches[colour] > 0:
                    var blocks = (colour_batches[colour] + CONTACT_BATCHES_PER_BLOCK - 1) // CONTACT_BATCHES_PER_BLOCK
                    ctx.base.stages.append(SolverStage(kind, colour_first[colour], colour_batches[colour], blocks, False, False))
            if has_any_joints:
                ctx.base.stages.append(SolverStage(kind, 0, 0, 1, True, True))

    if serial_batches > 0:
        ctx.base.stages.append(SolverStage(STAGE_RESTITUTION, 0, serial_batches, 1, True, False))
    for colour in range(GRAPH_COLOURS):
        if colour_batches[colour] > 0:
            var blocks = (colour_batches[colour] + CONTACT_BATCHES_PER_BLOCK - 1) // CONTACT_BATCHES_PER_BLOCK
            ctx.base.stages.append(SolverStage(STAGE_RESTITUTION, colour_first[colour], colour_batches[colour], blocks, False, False))

    ctx.base.words = List[Int64](length=(1 + 2 * len(ctx.base.stages)) * SOLVER_WORD_STRIDE, fill=0)
    result.stages = len(ctx.base.stages)

    var tasks = 1
    var work = result.batches + body_batches
    if work >= PARALLEL_MIN_BATCHES and pool_address != 0:
        tasks = min(1 + pool_workers(pool_address), max(1, count_cpus(cpus) // 2))
        tasks = max(1, min(tasks, work // 4))
    result.tasks = tasks
    if tasks <= 1:
        _lead_s6(ctx)
    else:
        ref pool = Pointer[JobPool, origin=MutAnyOrigin](unsafe_from_address=pool_address)[unsafe_offset=0]
        run_parallel_ctx(pool, tasks, ctx, _run_s6_solver_task)
    _ = ctx.base.tasks_run

    for b in range(contacts.batches):
        for lane in range(LANES):
            var index = contacts.manifold[b * LANES + lane]
            if index >= 0:
                contacts.take(b, lane, manifolds[index])

    _restore_supported_mass_ratios(bodies, saved_masses)
    result.broken_joints = broken_indices^
    _ = len(rows)
    return result^


struct PhysicsWorld(Movable):
    var bodies: List[RigidBody]
    var id_to_slot: List[Int]
    var broadphase_nodes: List[BroadphaseNode8]
    var hulls: List[ConvexHull]
    var meshes: List[Bvh8MeshCollider]
    var heightfields: List[HeightfieldCollider]
    var compounds: List[CompoundShape]
    var joints: List[Joint]
    var six_dof_joints: List[SixDofMotorJoint]
    var sensors: List[Sensor]
    var sensor_inside_a: List[Int]
    var sensor_inside_b: List[Int]
    var triggers: List[TriggerVolume]
    var trigger_inside_a: List[Int]
    var trigger_inside_b: List[Int]
    var trigger_events: List[TriggerEvent]
    var cache: ContactCache
    var islands: IslandSet
    var graph: ConstraintGraph
    var force_world: ForceWorld
    var events: List[PhysicsEvent]
    var settings: SolverSettings
    var gravity: Float32
    var water_enabled: Bool
    var water_level: Float32
    var water_use_waves: Bool
    var fluid_density: Float32
    var enable_sleeping: Bool
    var sleep_threshold_ticks: Int
    var time_s: Float32
    var tick: Int
    var next_body_id: Int
    var next_joint_id: Int
    var next_trigger_id: Int
    var cpus: List[Bool]

    def __init__(out self, gravity: Float32 = 9.81):
        self.bodies = List[RigidBody]()
        self.id_to_slot = List[Int]()
        self.broadphase_nodes = List[BroadphaseNode8]()
        self.hulls = List[ConvexHull]()
        self.meshes = List[Bvh8MeshCollider]()
        self.heightfields = List[HeightfieldCollider]()
        self.compounds = List[CompoundShape]()
        self.joints = List[Joint]()
        self.six_dof_joints = List[SixDofMotorJoint]()
        self.sensors = List[Sensor]()
        self.sensor_inside_a = List[Int]()
        self.sensor_inside_b = List[Int]()
        self.triggers = List[TriggerVolume]()
        self.trigger_inside_a = List[Int]()
        self.trigger_inside_b = List[Int]()
        self.trigger_events = List[TriggerEvent]()
        self.cache = ContactCache()
        self.islands = IslandSet()
        self.graph = ConstraintGraph()
        self.force_world = ForceWorld()
        self.events = List[PhysicsEvent]()
        self.settings = SolverSettings()
        self.gravity = gravity
        self.water_enabled = False
        self.water_level = NO_WATER
        self.water_use_waves = True
        self.fluid_density = RHO_SEAWATER_SI
        self.enable_sleeping = True
        self.sleep_threshold_ticks = SLEEP_TICKS_DEFAULT
        self.time_s = 0.0
        self.tick = 0
        self.next_body_id = 1
        self.next_joint_id = 1
        self.next_trigger_id = 1
        self.cpus = ccd_cpus()

    @always_inline
    def _set_id_slot(mut self, id: Int, slot: Int):
        if id < 0:
            return
        if id >= len(self.id_to_slot):
            self.id_to_slot.resize(max(id + 1, len(self.id_to_slot) * 2 + 16), -1)
        self.id_to_slot[id] = slot

    def _rebuild_id_to_slot(mut self):
        var max_id = self.next_body_id
        for i in range(len(self.bodies)):
            if self.bodies[i].id >= max_id:
                max_id = self.bodies[i].id + 1
        self.id_to_slot = List[Int](length=max(16, max_id + 1), fill=-1)
        for i in range(len(self.bodies)):
            var bid = self.bodies[i].id
            if bid >= 0 and self.bodies[i].active:
                self.id_to_slot[bid] = i

    def rebuild_broadphase(mut self):
        """Rebuilds persisted 8-wide `broadphase_nodes` SAH BVH over all active bodies."""
        self.broadphase_nodes.clear()
        var n_rows = len(self.bodies)
        if n_rows == 0:
            return
        var bp_in = BroadphaseInput()
        var live_rows = List[Int](capacity=n_rows)
        var cx = List[Float32](length=n_rows, fill=0.0)
        var cy = List[Float32](length=n_rows, fill=0.0)
        var cz = List[Float32](length=n_rows, fill=0.0)

        for r in range(n_rows):
            ref b = self.bodies[r]
            if not b.active:
                bp_in.add(Vec3(), Vec3(), False, False, b.filter)
                continue
            var tb = b.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
            var blo = tb[0]
            var bhi = tb[1]
            bp_in.add(blo, bhi, True, b.awake, b.filter)
            if blo.x <= bhi.x and blo.y <= bhi.y and blo.z <= bhi.z:
                live_rows.append(r)
                cx[r] = (blo.x + bhi.x) * Float32(0.5)
                cy[r] = (blo.y + bhi.y) * Float32(0.5)
                cz[r] = (blo.z + bhi.z) * Float32(0.5)

        var live_count = len(live_rows)
        if live_count == 0:
            return
        var bin_nodes = List[_BpBinNode](capacity=2 * live_count)
        var bin_root = _build_bp_bvh2(bp_in, live_rows, cx, cy, cz, 0, live_count, bin_nodes)
        self.broadphase_nodes.reserve((live_count + 7) // 8 + 4)
        _ = _collapse_bp_bvh8(bin_nodes, bin_root, self.broadphase_nodes)

    def _refit_broadphase(mut self):
        """O(N) bottom-up refit of persisted `broadphase_nodes` to post-integration tight bounds,
        avoiding a redundant second 16-bin SAH BVH8 rebuild per tick."""
        var n_nodes = len(self.broadphase_nodes)
        if n_nodes == 0:
            return
        for ni in range(n_nodes - 1, -1, -1):
            var node = self.broadphase_nodes[ni]
            for c in range(8):
                var ch = Int(node.child[c])
                if ch == 0:
                    continue
                if ch < 0:
                    var slot = -(ch + 1)
                    if slot >= 0 and slot < len(self.bodies) and self.bodies[slot].active:
                        var tb = self.bodies[slot].tight_aabb_full(
                            self.hulls, self.meshes, self.heightfields, self.compounds
                        )
                        node.min_x[c] = tb[0].x
                        node.min_y[c] = tb[0].y
                        node.min_z[c] = tb[0].z
                        node.max_x[c] = tb[1].x
                        node.max_y[c] = tb[1].y
                        node.max_z[c] = tb[1].z
                    else:
                        node.min_x[c] = Float32(1.0e30)
                        node.min_y[c] = Float32(1.0e30)
                        node.min_z[c] = Float32(1.0e30)
                        node.max_x[c] = Float32(-1.0e30)
                        node.max_y[c] = Float32(-1.0e30)
                        node.max_z[c] = Float32(-1.0e30)
                else:
                    ref cn = self.broadphase_nodes[ch]
                    node.min_x[c] = cn.min_x.reduce_min()
                    node.min_y[c] = cn.min_y.reduce_min()
                    node.min_z[c] = cn.min_z.reduce_min()
                    node.max_x[c] = cn.max_x.reduce_max()
                    node.max_y[c] = cn.max_y.reduce_max()
                    node.max_z[c] = cn.max_z.reduce_max()
            self.broadphase_nodes[ni] = node

    def _insert_broadphase_leaf_fast(mut self, slot: Int):
        """Incrementally inserts `slot` into `broadphase_nodes` when root has a free lane,
        rebuilding full SAH BVH8 once the root packet fills."""
        ref b = self.bodies[slot]
        if not b.active:
            return
        if len(self.broadphase_nodes) == 1:
            var tb = b.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
            for c in range(8):
                if self.broadphase_nodes[0].child[c] == 0:
                    self.broadphase_nodes[0].min_x[c] = tb[0].x
                    self.broadphase_nodes[0].min_y[c] = tb[0].y
                    self.broadphase_nodes[0].min_z[c] = tb[0].z
                    self.broadphase_nodes[0].max_x[c] = tb[1].x
                    self.broadphase_nodes[0].max_y[c] = tb[1].y
                    self.broadphase_nodes[0].max_z[c] = tb[1].z
                    self.broadphase_nodes[0].child[c] = Int32(-(slot + 1))
                    self.broadphase_nodes[0].cat_or |= b.filter.category
                    self.broadphase_nodes[0].mask_or |= b.filter.mask
                    return
        elif len(self.broadphase_nodes) == 0:
            var tb = b.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
            var node = BroadphaseNode8()
            node.min_x[0] = tb[0].x
            node.min_y[0] = tb[0].y
            node.min_z[0] = tb[0].z
            node.max_x[0] = tb[1].x
            node.max_y[0] = tb[1].y
            node.max_z[0] = tb[1].z
            node.child[0] = Int32(-(slot + 1))
            node.cat_or = b.filter.category
            node.mask_or = b.filter.mask
            self.broadphase_nodes.append(node)
            return
        self.rebuild_broadphase()

    def _register_body(mut self, var body: RigidBody) -> Int:
        var slot = len(self.bodies)
        var bid = body.id
        var is_dyn = (body.motion_mode == BODY_MOTION_DYNAMIC) and (not body.is_static)
        self.bodies.append(body^)
        self._set_id_slot(bid, slot)
        self.islands.ensure_rows(slot + 1)
        if is_dyn:
            self.islands.set_single(slot, bid)
        else:
            self.islands.drop(slot)
        self._insert_broadphase_leaf_fast(slot)
        return slot

    def remove_body(mut self, slot: Int):
        """Deactivates `slot` in O(1) and removes its ID mapping."""
        if slot < 0 or slot >= len(self.bodies):
            return
        var bid = self.bodies[slot].id
        self.bodies[slot].active = False
        self.bodies[slot].awake = False
        if bid >= 0 and bid < len(self.id_to_slot) and self.id_to_slot[bid] == slot:
            self.id_to_slot[bid] = -1
        self.islands.drop(slot)
        self.rebuild_broadphase()

    def swap_remove_body(mut self, slot: Int):
        """Swap-removes `slot` with the final slot in O(1), maintaining `id_to_slot`."""
        var last = len(self.bodies) - 1
        if slot < 0 or slot > last:
            return
        var removed_id = self.bodies[slot].id
        if removed_id >= 0 and removed_id < len(self.id_to_slot):
            self.id_to_slot[removed_id] = -1
        if slot < last:
            var moved = self.bodies.pop()
            var moved_id = moved.id
            self.bodies[slot] = moved^
            self._set_id_slot(moved_id, slot)
            if last < len(self.islands.label):
                self.islands.label[slot] = self.islands.label[last]
                _ = self.islands.label.pop()
        else:
            _ = self.bodies.pop()
            if last < len(self.islands.label):
                _ = self.islands.label.pop()
        self.rebuild_broadphase()

    def set_body_motion_mode(mut self, slot: Int, mode: UInt8, dynamic_mass: Float32 = -1.0):
        if slot < 0 or slot >= len(self.bodies):
            return
        self.bodies[slot].set_motion_mode(mode, dynamic_mass)
        if mode == BODY_MOTION_DYNAMIC and self.bodies[slot].active:
            self.islands.set_single(slot, self.bodies[slot].id)
        else:
            self.islands.drop(slot)
        self.rebuild_broadphase()

    def add_sphere(
        mut self,
        pos: Vec3,
        radius: Float32,
        mass: Float32,
        rot: Quat = Quat(),
        friction: Float32 = 0.6,
        restitution: Float32 = 0.1,
        filter: CollisionFilter = CollisionFilter(),
    ) -> Int:
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_SPHERE
        b.radius = max(Float32(0.01), radius)
        b.half_extents = Vec3(b.radius, b.radius, b.radius)
        b.pos = pos
        b.rot = rot.normalized()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        if mass <= Float32(0.0):
            b.mass = 0.0
            b.inv_mass = 0.0
            b.local_inertia = Vec3()
            b.is_static = True
            b.motion_mode = BODY_MOTION_STATIC
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
            b.motion_mode = BODY_MOTION_DYNAMIC
            b.local_inertia = compute_shape_inertia(BODY_SHAPE_SPHERE, mass, b.radius, 0.0, b.half_extents)
            b.wind_area_over_mass = (Float32(3.14159265) * b.radius * b.radius) / mass
        return self._register_body(b^)

    def add_capsule(
        mut self,
        pos: Vec3,
        radius: Float32,
        half_height: Float32,
        mass: Float32,
        rot: Quat = Quat(),
        friction: Float32 = 0.6,
        restitution: Float32 = 0.1,
        filter: CollisionFilter = CollisionFilter(),
    ) -> Int:
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_CAPSULE
        b.radius = max(Float32(0.01), radius)
        b.half_height = max(Float32(0.0), half_height)
        b.half_extents = Vec3(b.radius, b.half_height + b.radius, b.radius)
        b.pos = pos
        b.rot = rot.normalized()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        if mass <= Float32(0.0):
            b.mass = 0.0
            b.inv_mass = 0.0
            b.local_inertia = Vec3()
            b.is_static = True
            b.motion_mode = BODY_MOTION_STATIC
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
            b.motion_mode = BODY_MOTION_DYNAMIC
            b.local_inertia = compute_shape_inertia(BODY_SHAPE_CAPSULE, mass, b.radius, b.half_height, b.half_extents)
            b.wind_area_over_mass = (Float32(2.0) * b.radius * (Float32(2.0) * b.half_height + Float32(2.0) * b.radius)) / mass
        return self._register_body(b^)

    def add_box(
        mut self,
        pos: Vec3,
        half_extents: Vec3,
        mass: Float32,
        rot: Quat = Quat(),
        friction: Float32 = 0.6,
        restitution: Float32 = 0.1,
        filter: CollisionFilter = CollisionFilter(),
    ) -> Int:
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_BOX
        b.half_extents = Vec3(
            max(Float32(0.01), half_extents.x),
            max(Float32(0.01), half_extents.y),
            max(Float32(0.01), half_extents.z),
        )
        b.radius = b.half_extents.length()
        b.pos = pos
        b.rot = rot.normalized()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        if mass <= Float32(0.0):
            b.mass = 0.0
            b.inv_mass = 0.0
            b.local_inertia = Vec3()
            b.is_static = True
            b.motion_mode = BODY_MOTION_STATIC
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
            b.motion_mode = BODY_MOTION_DYNAMIC
            b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, mass, 0.0, 0.0, b.half_extents)
            var area = Float32(4.0) * (
                b.half_extents.x * b.half_extents.y
                + b.half_extents.y * b.half_extents.z
                + b.half_extents.z * b.half_extents.x
            ) * Float32(1.0 / 3.0)
            b.wind_area_over_mass = area / mass
        return self._register_body(b^)

    def add_kinematic_box(
        mut self,
        pos: Vec3,
        half_extents: Vec3,
        rot: Quat = Quat(),
        linear_vel: Vec3 = Vec3(),
        angular_vel: Vec3 = Vec3(),
        friction: Float32 = 0.8,
        restitution: Float32 = 0.0,
        filter: CollisionFilter = CollisionFilter(CAT_STATIC, CAT_ALL, 0),
    ) -> Int:
        var slot = self.add_box(pos, half_extents, 0.0, rot, friction, restitution, filter)
        self.bodies[slot].set_motion_mode(BODY_MOTION_KINEMATIC)
        self.bodies[slot].linear_vel = linear_vel
        self.bodies[slot].angular_vel = angular_vel
        return slot

    def add_convex_hull(
        mut self,
        pos: Vec3,
        hull: ConvexHull,
        mass: Float32,
        rot: Quat = Quat(),
        friction: Float32 = 0.6,
        restitution: Float32 = 0.1,
        filter: CollisionFilter = CollisionFilter(),
    ) -> Int:
        var h_idx = len(self.hulls)
        self.hulls.append(hull.copy())
        var he = _hull_half_extents(hull)
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_CONVEX_HULL
        b.shape_index = h_idx
        b.half_extents = he
        b.radius = he.length()
        b.pos = pos
        b.rot = rot.normalized()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        if mass <= Float32(0.0):
            b.mass = 0.0
            b.inv_mass = 0.0
            b.local_inertia = Vec3()
            b.is_static = True
            b.motion_mode = BODY_MOTION_STATIC
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
            b.motion_mode = BODY_MOTION_DYNAMIC
            var mp = hull.compute_polyhedral_mass_properties(Float32(1.0))
            if mp[0] > Float32(1.0e-6):
                var scale_m = mass / mp[0]
                b.local_inertia = Vec3(
                    max(Float32(1.0e-5), mp[2].xx * scale_m),
                    max(Float32(1.0e-5), mp[2].yy * scale_m),
                    max(Float32(1.0e-5), mp[2].zz * scale_m),
                )
            else:
                b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, mass, 0.0, 0.0, he)
            b.wind_area_over_mass = (Float32(4.0) * he.x * he.y) / mass
        return self._register_body(b^)

    def add_static_mesh(
        mut self,
        bvh: Bvh8MeshCollider,
        pos: Vec3 = Vec3(),
        rot: Quat = Quat(),
        friction: Float32 = 0.8,
        restitution: Float32 = 0.0,
        filter: CollisionFilter = CollisionFilter(CAT_TERRAIN, CAT_ALL, 0),
    ) -> Int:
        var m_idx = len(self.meshes)
        self.meshes.append(bvh.copy())
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_MESH_BVH8
        b.shape_index = m_idx
        b.pos = pos
        b.rot = rot.normalized()
        b.mass = 0.0
        b.inv_mass = 0.0
        b.local_inertia = Vec3()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        b.is_static = True
        b.motion_mode = BODY_MOTION_STATIC
        b.awake = False
        return self._register_body(b^)

    def add_heightfield(
        mut self,
        hf: HeightfieldCollider,
        pos: Vec3 = Vec3(),
        rot: Quat = Quat(),
        friction: Float32 = 0.85,
        restitution: Float32 = 0.05,
        filter: CollisionFilter = CollisionFilter(CAT_TERRAIN, CAT_ALL, 0),
    ) -> Int:
        var hf_idx = len(self.heightfields)
        self.heightfields.append(hf.copy())
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_HEIGHTFIELD
        b.shape_index = hf_idx
        b.pos = pos
        b.rot = rot.normalized()
        b.mass = 0.0
        b.inv_mass = 0.0
        b.local_inertia = Vec3()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        b.is_static = True
        b.motion_mode = BODY_MOTION_STATIC
        b.awake = False
        return self._register_body(b^)

    def add_compound(
        mut self,
        pos: Vec3,
        compound: CompoundShape,
        mass: Float32,
        rot: Quat = Quat(),
        friction: Float32 = 0.6,
        restitution: Float32 = 0.1,
        filter: CollisionFilter = CollisionFilter(),
    ) -> Int:
        var c_idx = len(self.compounds)
        self.compounds.append(compound.copy())
        var aabb0 = compound.tight_aabb(Vec3(), Quat())
        var he = (aabb0[1] - aabb0[0]) * Float32(0.5)
        var b = RigidBody(self.next_body_id)
        self.next_body_id += 1
        b.shape = BODY_SHAPE_COMPOUND
        b.shape_index = c_idx
        b.half_extents = Vec3(max(Float32(0.02), he.x), max(Float32(0.02), he.y), max(Float32(0.02), he.z))
        b.radius = b.half_extents.length()
        b.pos = pos
        b.rot = rot.normalized()
        b.friction = friction
        b.restitution = restitution
        b.filter = filter
        if mass <= Float32(0.0):
            b.mass = 0.0
            b.inv_mass = 0.0
            b.local_inertia = Vec3()
            b.is_static = True
            b.motion_mode = BODY_MOTION_STATIC
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
            b.motion_mode = BODY_MOTION_DYNAMIC
            var mp = compound.compute_mass_properties(Float32(1.0))
            if mp[0] > Float32(1.0e-6):
                var scale_m = mass / mp[0]
                b.local_inertia = Vec3(
                    max(Float32(1.0e-5), mp[2].xx * scale_m),
                    max(Float32(1.0e-5), mp[2].yy * scale_m),
                    max(Float32(1.0e-5), mp[2].zz * scale_m),
                )
            else:
                b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, mass, 0.0, 0.0, b.half_extents)
            b.wind_area_over_mass = (Float32(4.0) * b.half_extents.x * b.half_extents.y) / mass
        return self._register_body(b^)

    def add_six_dof_joint(mut self, var j: SixDofMotorJoint) -> Int:
        var idx = len(self.six_dof_joints)
        if j.id <= 0:
            j.id = self.next_joint_id
            self.next_joint_id += 1
        var sa = self.find_slot_by_id(j.body_a) if j.body_a != JOINT_WORLD else -1
        var sb = self.find_slot_by_id(j.body_b)
        if sa >= 0:
            self.wake(sa)
        if sb >= 0:
            self.wake(sb)
        self.six_dof_joints.append(j^)
        return idx

    def add_sensor(mut self, sensor: Sensor):
        self.sensors.append(sensor)

    def add_trigger(mut self, var trigger: TriggerVolume) -> Int:
        if trigger.id <= 0:
            trigger.id = self.next_trigger_id
            self.next_trigger_id += 1
        elif trigger.id >= self.next_trigger_id:
            self.next_trigger_id = trigger.id + 1
        var idx = len(self.triggers)
        self.triggers.append(trigger^)
        return idx

    def set_water(mut self, level: Float32, use_waves: Bool = True, fluid_density: Float32 = RHO_SEAWATER_SI):
        self.water_enabled = level > NO_WATER + Float32(1.0)
        self.water_level = level
        self.water_use_waves = use_waves
        self.fluid_density = fluid_density

    def body_id(self, slot: Int) -> Int:
        if slot < 0 or slot >= len(self.bodies):
            return -1
        return self.bodies[slot].id

    @always_inline
    def find_slot_by_id(self, id: Int) -> Int:
        """O(1) direct table lookup from `body.id` to active `bodies` slot."""
        if id < 0 or id >= len(self.id_to_slot):
            return -1
        var slot = self.id_to_slot[id]
        if slot >= 0 and slot < len(self.bodies) and self.bodies[slot].active and self.bodies[slot].id == id:
            return slot
        return -1

    def wake(mut self, slot: Int):
        if slot < 0 or slot >= len(self.bodies):
            return
        if self.bodies[slot].motion_mode != BODY_MOTION_DYNAMIC or self.bodies[slot].is_static or not self.bodies[slot].active:
            return
        var lbl = self.islands.label[slot] if slot < len(self.islands.label) else NO_ISLAND
        if lbl == NO_ISLAND:
            if not self.bodies[slot].awake:
                self.bodies[slot].awake = True
                self.bodies[slot].sleep_ticks = 0
                self.events.append(PhysicsEvent(EVENT_WAKE, self.bodies[slot].id, 0))
            return
        for r in range(len(self.bodies)):
            if self.bodies[r].active and not self.bodies[r].is_static and r < len(self.islands.label) and self.islands.label[r] == lbl:
                if not self.bodies[r].awake:
                    self.bodies[r].awake = True
                    self.events.append(PhysicsEvent(EVENT_WAKE, self.bodies[r].id, 0))
                self.bodies[r].sleep_ticks = 0

    def set_velocity(mut self, slot: Int, linear: Vec3, angular: Vec3 = Vec3()):
        if slot < 0 or slot >= len(self.bodies):
            return
        if self.bodies[slot].motion_mode == BODY_MOTION_STATIC:
            return
        self.bodies[slot].linear_vel = linear
        self.bodies[slot].angular_vel = angular
        if self.bodies[slot].motion_mode == BODY_MOTION_KINEMATIC:
            self.bodies[slot].awake = True
        else:
            self.wake(slot)

    def apply_force(mut self, slot: Int, force: Vec3):
        if slot < 0 or slot >= len(self.bodies) or self.bodies[slot].is_static:
            return
        self.bodies[slot].force_accum = self.bodies[slot].force_accum + force
        self.wake(slot)

    def apply_impulse(mut self, slot: Int, impulse: Vec3):
        if slot < 0 or slot >= len(self.bodies) or self.bodies[slot].is_static:
            return
        self.bodies[slot].linear_vel = self.bodies[slot].linear_vel + impulse * self.bodies[slot].inv_mass
        self.wake(slot)

    def apply_impulse_at(mut self, slot: Int, impulse: Vec3, world_point: Vec3):
        if slot < 0 or slot >= len(self.bodies) or self.bodies[slot].is_static:
            return
        ref b = self.bodies[slot]
        b.linear_vel = b.linear_vel + impulse * b.inv_mass
        var r = world_point - b.pos
        var tau_local = b.rot.conjugate().rotate(r.cross(impulse))
        var dw_local = Vec3(
            tau_local.x / max(Float32(1.0e-8), b.local_inertia.x),
            tau_local.y / max(Float32(1.0e-8), b.local_inertia.y),
            tau_local.z / max(Float32(1.0e-8), b.local_inertia.z),
        )
        b.angular_vel = b.angular_vel + b.rot.rotate(dw_local)
        self.wake(slot)

    def add_joint(mut self, jdef: JointDef) -> Int:
        """Creates and registers a `Joint` from `JointDef`, returning its index in `self.joints`."""
        var slot_a = jdef.slot_a
        var slot_b = jdef.slot_b
        if slot_b < 0 or slot_b >= len(self.bodies):
            return -1

        var j = Joint()
        j.id = self.next_joint_id
        self.next_joint_id += 1
        j.kind = jdef.kind
        var ax_w = jdef.axis.normalized()
        if ax_w.length_squared() < Float32(0.25):
            ax_w = Vec3(0, 1, 0)

        var has_body_a = (
            slot_a >= 0
            and slot_a < len(self.bodies)
            and (not self.bodies[slot_a].is_static or self.bodies[slot_a].motion_mode == BODY_MOTION_KINEMATIC)
        )
        var qa = Quat()
        if has_body_a:
            j.body_a = self.bodies[slot_a].id
            var pa = self.bodies[slot_a].pos
            qa = self.bodies[slot_a].rot
            j.local_a = qa.conjugate().rotate(jdef.anchor - pa)
            j.axis_a = qa.conjugate().rotate(ax_w)
            self.wake(slot_a)
        else:
            j.body_a = JOINT_WORLD
            j.local_a = jdef.anchor
            j.axis_a = ax_w

        var pb = self.bodies[slot_b].pos
        var qb = self.bodies[slot_b].rot
        var anchor_b = jdef.anchor_b if jdef.has_anchor_b else jdef.anchor
        j.body_b = self.bodies[slot_b].id
        j.local_b = qb.conjugate().rotate(anchor_b - pb)
        j.axis_b = qb.conjugate().rotate(ax_w)
        j.reference = qb if not has_body_a else quat_product(quat_conj(qa), qb)
        j.lower = jdef.lower
        j.upper = jdef.upper
        j.cone = jdef.cone
        j.enable_limit = jdef.enable_limit
        j.enable_motor = jdef.enable_motor
        j.enable_spring = jdef.enable_spring
        j.motor_speed = jdef.motor_speed
        j.max_motor = jdef.max_motor
        j.spring_hertz = jdef.spring_hertz
        j.spring_damping = jdef.spring_damping
        j.break_force = jdef.break_force
        j.break_torque = jdef.break_torque
        j.linear_hertz = jdef.hertz
        j.angular_hertz = jdef.hertz
        j.damping = jdef.damping
        j.max_force = jdef.max_motor
        j.max_torque = jdef.max_motor

        if j.kind == JOINT_DISTANCE:
            if jdef.length >= Float32(0.0):
                j.rest = jdef.length
            elif jdef.enable_spring and jdef.rest > Float32(0.0):
                j.rest = jdef.rest
            else:
                j.rest = (anchor_b - jdef.anchor).length()
        else:
            j.rest = jdef.rest

        self.wake(slot_b)
        var j_idx = len(self.joints)
        self.joints.append(j)
        return j_idx

    def add_weld_joint(
        mut self,
        slot_a: Int,
        slot_b: Int,
        anchor: Vec3,
        break_force: Float32 = 0.0,
        break_torque: Float32 = 0.0,
        hertz: Float32 = 0.0,
        damping: Float32 = 0.0,
    ) -> Int:
        var jd = JointDef(JOINT_WELD, slot_a, slot_b, anchor)
        jd.break_force = break_force
        jd.break_torque = break_torque
        jd.hertz = hertz
        jd.damping = damping
        return self.add_joint(jd)

    def reset_body(
        mut self,
        slot: Int,
        pos: Vec3,
        rot: Quat = Quat(),
        linear_vel: Vec3 = Vec3(),
        angular_vel: Vec3 = Vec3(),
        active: Bool = True,
    ):
        """Resets a pooled rigid body's pose, velocities, accumulators, and island state in place."""
        if slot < 0 or slot >= len(self.bodies):
            return
        ref b = self.bodies[slot]
        b.pos = pos
        b.rot = rot.normalized()
        b.linear_vel = linear_vel
        b.angular_vel = angular_vel
        b.force_accum = Vec3()
        b.torque_accum = Vec3()
        b.active = active
        b.awake = active and (b.motion_mode != BODY_MOTION_STATIC)
        b.sleep_ticks = 0
        b.in_water = False
        b.water_volume_id = -1
        var bid = b.id
        if active:
            self._set_id_slot(bid, slot)
        elif bid >= 0 and bid < len(self.id_to_slot) and self.id_to_slot[bid] == slot:
            self.id_to_slot[bid] = -1
        self.islands.ensure_rows(len(self.bodies))
        if active and (not self.bodies[slot].is_static):
            self.islands.set_single(slot, bid)
        else:
            self.islands.drop(slot)
        self.rebuild_broadphase()

    def rearm_joint(
        mut self,
        joint_idx: Int,
        break_force: Float32 = -1.0,
        break_torque: Float32 = -1.0,
    ):
        """Re-arms a pooled joint, clearing fracture state and warm-start impulses."""
        if joint_idx < 0 or joint_idx >= len(self.joints):
            return
        ref j = self.joints[joint_idx]
        j.broken = False
        j.linear_impulse = Vec3()
        j.angular_impulse = Vec3()
        j.force_impulse = Vec3()
        j.torque_impulse = Vec3()
        j.axial_impulse = 0.0
        j.motor_impulse = 0.0
        j.lower_impulse = 0.0
        j.upper_impulse = 0.0
        j.swing_impulse = 0.0
        if break_force >= Float32(0.0):
            j.break_force = break_force
        if break_torque >= Float32(0.0):
            j.break_torque = break_torque
        var sa = self.find_slot_by_id(j.body_a) if j.body_a != JOINT_WORLD else -1
        var sb = self.find_slot_by_id(j.body_b)
        if sa >= 0:
            self.wake(sa)
        if sb >= 0:
            self.wake(sb)

    def rebuild_islands_from_joints(mut self):
        """Reconstructs `IslandSet` deterministically from active dynamic bodies and unbroken joints."""
        var n_rows = len(self.bodies)
        self.islands.clear()
        self.islands.ensure_rows(n_rows)
        var body_ids = List[Int](capacity=n_rows)
        for i in range(n_rows):
            if self.bodies[i].active and (not self.bodies[i].is_static):
                self.islands.set_single(i, self.bodies[i].id)
                body_ids.append(self.bodies[i].id)
            else:
                self.islands.drop(i)
                body_ids.append(NO_ISLAND)
        var link_a = List[Int]()
        var link_b = List[Int]()
        var link_ra = List[Int]()
        var link_rb = List[Int]()
        for ji in range(len(self.joints)):
            if self.joints[ji].broken or self.joints[ji].body_a == JOINT_WORLD:
                continue
            var sa = self.find_slot_by_id(self.joints[ji].body_a)
            var sb = self.find_slot_by_id(self.joints[ji].body_b)
            if sa >= 0 and sb >= 0 and self.bodies[sa].active and self.bodies[sb].active and (not self.bodies[sa].is_static) and (not self.bodies[sb].is_static):
                var id0 = self.bodies[sa].id
                var id1 = self.bodies[sb].id
                if id0 < id1:
                    link_a.append(id0); link_b.append(id1); link_ra.append(sa); link_rb.append(sb)
                else:
                    link_a.append(id1); link_b.append(id0); link_ra.append(sb); link_rb.append(sa)
        self.islands.update_links(link_a, link_b, link_ra, link_rb, body_ids)

    # =========================================================================
    # Unified O(log N) SIMD8 Scene Query APIs
    # =========================================================================

    def _query_bvh8_aabb_candidates(
        self,
        qlo: Vec3,
        qhi: Vec3,
        filter: CollisionFilter,
        ignore_body_id: Int,
        mut out_slots: List[Int],
    ):
        """Traverses persisted `broadphase_nodes` in 8-wide `F8` SIMD to gather candidate body slots."""
        if len(self.broadphase_nodes) == 0:
            for r in range(len(self.bodies)):
                ref b = self.bodies[r]
                if not b.active or b.id == ignore_body_id or not filter.touches(b.filter):
                    continue
                var tb = b.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
                if boxes_overlap(qlo, qhi, tb[0], tb[1]):
                    out_slots.append(r)
            return

        var qlo_x = F8(qlo.x)
        var qlo_y = F8(qlo.y)
        var qlo_z = F8(qlo.z)
        var qhi_x = F8(qhi.x)
        var qhi_y = F8(qhi.y)
        var qhi_z = F8(qhi.z)

        var stack = Array[Int32, 96](fill=0)
        var sp = 1
        stack[0] = 0

        while sp > 0:
            sp -= 1
            ref node = self.broadphase_nodes[Int(stack[sp])]
            if filter.group == 0 and ((filter.category & node.mask_or) == 0 or (node.cat_or & filter.mask) == 0):
                continue
            var ov = (
                qlo_x.le(node.max_x)
                & qhi_x.ge(node.min_x)
                & qlo_y.le(node.max_y)
                & qhi_y.ge(node.min_y)
                & qlo_z.le(node.max_z)
                & qhi_z.ge(node.min_z)
            )
            if not ov.reduce_or():
                continue

            comptime for c in range(LANES):
                if ov[c]:
                    var ch = Int(node.child[c])
                    if ch > 0:
                        if sp < 96:
                            stack[sp] = Int32(ch)
                            sp += 1
                    elif ch < 0:
                        var slot = -ch - 1
                        if slot >= 0 and slot < len(self.bodies):
                            ref b = self.bodies[slot]
                            if b.active and b.id != ignore_body_id and filter.touches(b.filter):
                                out_slots.append(slot)

    def _raycast_single_body(
        self,
        slot: Int,
        origin: Vec3,
        ndir: Vec3,
        max_dist: Float32,
    ) -> SceneCastHit:
        var out = SceneCastHit()
        out.distance = max_dist
        ref b = self.bodies[slot]
        var end = origin + ndir * max_dist

        if b.shape == BODY_SHAPE_SPHERE:
            var sh = swept_sphere_moving_sphere(origin, end, Float32(0.0), b.pos, b.pos, b.radius)
            if sh.hit and sh.toi * max_dist <= max_dist:
                out.hit = True
                out.fraction = sh.toi
                out.distance = sh.toi * max_dist
                out.point = sh.point
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_CAPSULE:
            var ends = b.capsule_endpoints()
            var seg = Bvh8MeshCollider._sweep_sphere_segment(origin, ndir, max_dist, b.radius, ends[0], ends[1])
            if seg[0] and seg[1] <= max_dist:
                out.hit = True
                out.distance = seg[1]
                out.fraction = seg[1] / max(Float32(1.0e-7), max_dist)
                out.point = origin + ndir * seg[1]
                out.normal = seg[3]
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_BOX:
            var sh = swept_sphere_obb(origin, end, Float32(0.0), b.pos, b.rot, b.half_extents)
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi
                out.distance = sh.toi * max_dist
                out.point = origin + ndir * out.distance
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_CONVEX_HULL and b.shape_index >= 0 and b.shape_index < len(self.hulls):
            var pt_hull = _make_point_hull()
            var sh = _swept_inflated_hull_cast(
                pt_hull, Float32(0.0), origin, end, Quat(),
                self.hulls[b.shape_index], b.pos, b.pos, b.rot,
            )
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi
                out.distance = sh.toi * max_dist
                out.point = origin + ndir * out.distance
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_MESH_BVH8 and b.shape_index >= 0 and b.shape_index < len(self.meshes):
            var inv_r = b.rot.conjugate()
            var l_orig = inv_r.rotate(origin - b.pos)
            var l_dir = inv_r.rotate(ndir)
            var rh = self.meshes[b.shape_index].raycast(l_orig, l_dir, max_dist)
            if rh.hit and rh.t <= max_dist:
                out.hit = True
                out.distance = rh.t
                out.fraction = rh.t / max(Float32(1.0e-7), max_dist)
                out.point = b.pos + b.rot.rotate(rh.point)
                var wn = b.rot.rotate(rh.normal).normalized()
                if wn.dot(ndir) > Float32(0.0):
                    wn = -wn
                out.normal = wn
                out.body_slot = slot
                out.body_id = b.id
                out.sub_index = rh.triangle_index
                out.material_id = Int(rh.material_id) if rh.material_id != 0 else b.material
            return out^

        if b.shape == BODY_SHAPE_HEIGHTFIELD and b.shape_index >= 0 and b.shape_index < len(self.heightfields):
            var inv_r = b.rot.conjugate()
            var l_orig = inv_r.rotate(origin - b.pos)
            var l_dir = inv_r.rotate(ndir)
            var rh = self.heightfields[b.shape_index].raycast_dda(l_orig, l_dir, max_dist)
            if rh.hit and rh.t <= max_dist:
                out.hit = True
                out.distance = rh.t
                out.fraction = rh.t / max(Float32(1.0e-7), max_dist)
                out.point = b.pos + b.rot.rotate(rh.point)
                var wn = b.rot.rotate(rh.normal).normalized()
                if wn.dot(ndir) > Float32(0.0):
                    wn = -wn
                out.normal = wn
                out.body_slot = slot
                out.body_id = b.id
                out.sub_index = rh.triangle_index
                out.material_id = Int(rh.material_id) if rh.material_id != 0 else b.material
            return out^

        if b.shape == BODY_SHAPE_COMPOUND and b.shape_index >= 0 and b.shape_index < len(self.compounds):
            ref cmp = self.compounds[b.shape_index]
            var best_d = max_dist
            for ci in range(len(cmp.children)):
                ref ch = cmp.children[ci]
                var wp_wr = ch.world_pose(b.pos, b.rot)
                var wp = wp_wr[0]
                var wr = wp_wr[1]
                var ch_end = origin + ndir * best_d
                var c_hit = CcdHit()
                if ch.shape == COMPOUND_CHILD_SPHERE:
                    c_hit = swept_sphere_moving_sphere(origin, ch_end, Float32(0.0), wp, wp, ch.radius)
                elif ch.shape == COMPOUND_CHILD_CAPSULE:
                    var ax = wr.rotate(Vec3(0, ch.half_height, 0))
                    var seg = Bvh8MeshCollider._sweep_sphere_segment(origin, ndir, best_d, ch.radius, wp - ax, wp + ax)
                    if seg[0]:
                        c_hit = CcdHit(True, seg[1] / max(Float32(1.0e-7), best_d), origin + ndir * seg[1], seg[3])
                elif ch.shape == COMPOUND_CHILD_BOX:
                    c_hit = swept_sphere_obb(origin, ch_end, Float32(0.0), wp, wr, ch.half_extents)
                elif ch.shape == COMPOUND_CHILD_CONVEX:
                    c_hit = _swept_inflated_hull_cast(
                        _make_point_hull(), Float32(0.0), origin, ch_end, Quat(),
                        ch.hull, wp, wp, wr,
                    )
                if c_hit.hit:
                    var d_hit = c_hit.toi * best_d
                    if d_hit <= best_d:
                        best_d = d_hit
                        out.hit = True
                        out.distance = d_hit
                        out.fraction = d_hit / max(Float32(1.0e-7), max_dist)
                        out.point = origin + ndir * d_hit
                        out.normal = c_hit.normal
                        out.body_slot = slot
                        out.body_id = b.id
                        out.sub_index = ci
                        out.material_id = b.material
            return out^

        return out^

    def raycast(
        self,
        origin: Vec3,
        dir: Vec3,
        max_dist: Float32 = 1.0e6,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> SceneCastHit:
        """O(log N) SIMD8 near-t ordered raycast across all active world colliders."""
        var best = SceneCastHit()
        best.distance = max_dist
        var d_len = dir.length()
        if d_len <= Float32(1.0e-8) or max_dist <= Float32(0.0):
            return best^
        var ndir = dir * (Float32(1.0) / d_len)

        if len(self.broadphase_nodes) == 0:
            for r in range(len(self.bodies)):
                ref b = self.bodies[r]
                if not b.active or b.id == ignore_body_id or not filter.touches(b.filter):
                    continue
                var h = self._raycast_single_body(r, origin, ndir, best.distance)
                if h.hit and (h.distance < best.distance or (h.distance == best.distance and h.body_id < best.body_id)):
                    best = h
                    best.fraction = h.distance / max_dist
            return best^

        var ox = F8(origin.x)
        var oy = F8(origin.y)
        var oz = F8(origin.z)
        var idx = F8(_safe_inv(ndir.x))
        var idy = F8(_safe_inv(ndir.y))
        var idz = F8(_safe_inv(ndir.z))

        var stack_node = Array[Int32, 96](fill=0)
        var stack_t = Array[Float32, 96](fill=0.0)
        var sp = 1
        stack_node[0] = 0
        stack_t[0] = 0.0

        var child_idx = Array[Int32, LANES](fill=0)
        var child_t = Array[Float32, LANES](fill=0.0)

        while sp > 0:
            sp -= 1
            if stack_t[sp] > best.distance:
                continue
            ref node = self.broadphase_nodes[Int(stack_node[sp])]
            if filter.group == 0 and ((filter.category & node.mask_or) == 0 or (node.cat_or & filter.mask) == 0):
                continue

            var tx1 = (node.min_x - ox) * idx
            var tx2 = (node.max_x - ox) * idx
            var tmin = min(tx1, tx2)
            var tmax = max(tx1, tx2)

            var ty1 = (node.min_y - oy) * idy
            var ty2 = (node.max_y - oy) * idy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

            var tz1 = (node.min_z - oz) * idz
            var tz2 = (node.max_z - oz) * idz
            tmin = max(tmin, min(tz1, tz2))
            tmax = min(tmax, max(tz1, tz2))

            var t_enter = max(F8(0.0), tmin)
            var mask = tmax.ge(t_enter) & t_enter.le(F8(best.distance))
            if not mask.reduce_or():
                continue

            var internal_cnt = 0
            comptime for c in range(LANES):
                if mask[c]:
                    var ch = Int(node.child[c])
                    if ch < 0:
                        var slot = -ch - 1
                        if slot >= 0 and slot < len(self.bodies):
                            ref b = self.bodies[slot]
                            if b.active and b.id != ignore_body_id and filter.touches(b.filter):
                                var h = self._raycast_single_body(slot, origin, ndir, best.distance)
                                if h.hit and (
                                    h.distance < best.distance
                                    or (h.distance == best.distance and h.body_id < best.body_id)
                                ):
                                    best = h
                                    best.fraction = h.distance / max_dist
                    elif ch > 0:
                        var te = t_enter[c]
                        if te <= best.distance:
                            child_idx[internal_cnt] = Int32(ch)
                            child_t[internal_cnt] = te
                            internal_cnt += 1

            for i in range(1, internal_cnt):
                var ki = child_idx[i]
                var kt = child_t[i]
                var j = i - 1
                while j >= 0 and child_t[j] < kt:
                    child_idx[j + 1] = child_idx[j]
                    child_t[j + 1] = child_t[j]
                    j -= 1
                child_idx[j + 1] = ki
                child_t[j + 1] = kt

            for i in range(internal_cnt):
                if child_t[i] <= best.distance and sp < 96:
                    stack_node[sp] = child_idx[i]
                    stack_t[sp] = child_t[i]
                    sp += 1

        return best^

    def raycast_all(
        self,
        origin: Vec3,
        dir: Vec3,
        max_dist: Float32 = 1.0e6,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> List[SceneCastHit]:
        """Returns all raycast hits along `[0, max_dist]` sorted ascending by `(distance, body_id)`."""
        var hits = List[SceneCastHit]()
        var d_len = dir.length()
        if d_len <= Float32(1.0e-8) or max_dist <= Float32(0.0):
            return hits^
        var ndir = dir * (Float32(1.0) / d_len)
        var end = origin + ndir * max_dist
        var rlo = Vec3(min(origin.x, end.x), min(origin.y, end.y), min(origin.z, end.z))
        var rhi = Vec3(max(origin.x, end.x), max(origin.y, end.y), max(origin.z, end.z))

        var cands = List[Int]()
        self._query_bvh8_aabb_candidates(rlo, rhi, filter, ignore_body_id, cands)
        for i in range(len(cands)):
            var h = self._raycast_single_body(cands[i], origin, ndir, max_dist)
            if h.hit:
                h.fraction = h.distance / max_dist
                hits.append(h)

        # Deterministic insertion sort by (distance, body_id)
        for i in range(1, len(hits)):
            var key = hits[i]
            var j = i - 1
            while j >= 0 and (
                hits[j].distance > key.distance
                or (hits[j].distance == key.distance and hits[j].body_id > key.body_id)
            ):
                hits[j + 1] = hits[j]
                j -= 1
            hits[j + 1] = key
        return hits^

    def _shapecast_hull_vs_body(
        self,
        query_hull: ConvexHull,
        inflate_r: Float32,
        center: Vec3,
        rot: Quat,
        delta: Vec3,
        slot: Int,
        max_frac: Float32,
    ) -> SceneCastHit:
        """Exact swept `query_hull` (inflated by `inflate_r`) against body `slot` on `[0, max_frac]`."""
        var out = SceneCastHit()
        out.fraction = max_frac
        var total_dist = delta.length()
        var sub_delta = delta * max_frac
        var p1 = center + sub_delta
        ref b = self.bodies[slot]

        var is_pure_sphere = len(query_hull.vertices) == 1
        if is_pure_sphere and b.shape == BODY_SHAPE_SPHERE:
            var sh = swept_sphere_moving_sphere(center, p1, inflate_r, b.pos, b.pos, b.radius)
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi * max_frac
                out.distance = out.fraction * total_dist
                out.point = sh.point
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if is_pure_sphere and b.shape == BODY_SHAPE_BOX:
            var sh = swept_sphere_obb(center, p1, inflate_r, b.pos, b.rot, b.half_extents)
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi * max_frac
                out.distance = out.fraction * total_dist
                out.point = sh.point
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if is_pure_sphere and b.shape == BODY_SHAPE_MESH_BVH8 and b.shape_index >= 0 and b.shape_index < len(self.meshes):
            var inv_r = b.rot.conjugate()
            var l_start = inv_r.rotate(center - b.pos)
            var l_delta = inv_r.rotate(delta)
            var sh = self.meshes[b.shape_index].sweep_sphere(l_start, l_delta, inflate_r, max_frac)
            if sh.hit and sh.toi <= max_frac:
                out.hit = True
                out.fraction = sh.toi
                out.distance = sh.toi * total_dist
                out.point = b.pos + b.rot.rotate(sh.point)
                out.normal = b.rot.rotate(sh.normal).normalized()
                out.body_slot = slot
                out.body_id = b.id
                out.sub_index = sh.triangle_index
                out.material_id = b.material
            return out^

        if is_pure_sphere and b.shape == BODY_SHAPE_HEIGHTFIELD and b.shape_index >= 0 and b.shape_index < len(self.heightfields):
            var inv_r = b.rot.conjugate()
            var l_start = inv_r.rotate(center - b.pos)
            var l_delta = inv_r.rotate(delta)
            var sh = self.heightfields[b.shape_index].sweep_sphere(l_start, l_delta, inflate_r, max_frac)
            if sh.hit and sh.toi <= max_frac:
                out.hit = True
                out.fraction = sh.toi
                out.distance = sh.toi * total_dist
                out.point = b.pos + b.rot.rotate(sh.point)
                out.normal = b.rot.rotate(sh.normal).normalized()
                out.body_slot = slot
                out.body_id = b.id
                out.sub_index = sh.triangle_index
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_SPHERE:
            var sh = _swept_inflated_hull_cast(
                query_hull, inflate_r + b.radius, center, p1, rot,
                _make_point_hull(), b.pos, b.pos, Quat(),
            )
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi * max_frac
                out.distance = out.fraction * total_dist
                out.point = b.pos + sh.normal * b.radius
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_CAPSULE:
            var sh = _swept_inflated_hull_cast(
                query_hull, inflate_r + b.radius, center, p1, rot,
                _make_segment_hull(b.half_height), b.pos, b.pos, b.rot,
            )
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi * max_frac
                out.distance = out.fraction * total_dist
                out.point = sh.point + sh.normal * b.radius
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_BOX:
            var sh = _swept_inflated_hull_cast(
                query_hull, inflate_r, center, p1, rot,
                _make_box_hull(b.half_extents), b.pos, b.pos, b.rot,
            )
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi * max_frac
                out.distance = out.fraction * total_dist
                out.point = sh.point
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if b.shape == BODY_SHAPE_CONVEX_HULL and b.shape_index >= 0 and b.shape_index < len(self.hulls):
            var sh = _swept_inflated_hull_cast(
                query_hull, inflate_r, center, p1, rot,
                self.hulls[b.shape_index], b.pos, b.pos, b.rot,
            )
            if sh.hit:
                out.hit = True
                out.fraction = sh.toi * max_frac
                out.distance = out.fraction * total_dist
                out.point = sh.point
                out.normal = sh.normal
                out.body_slot = slot
                out.body_id = b.id
                out.material_id = b.material
            return out^

        if (b.shape == BODY_SHAPE_MESH_BVH8 and b.shape_index >= 0 and b.shape_index < len(self.meshes)) or (
            b.shape == BODY_SHAPE_HEIGHTFIELD and b.shape_index >= 0 and b.shape_index < len(self.heightfields)
        ):
            var inv_r = b.rot.conjugate()
            var l_p0 = inv_r.rotate(center - b.pos)
            var l_p1 = inv_r.rotate(p1 - b.pos)
            var l_rot = inv_r * rot
            var q_he = _hull_half_extents(query_hull) + Vec3(inflate_r, inflate_r, inflate_r)
            var q_ax = get_box_axes(l_rot)
            var wx = abs(q_ax[0].x) * q_he.x + abs(q_ax[1].x) * q_he.y + abs(q_ax[2].x) * q_he.z
            var wy = abs(q_ax[0].y) * q_he.x + abs(q_ax[1].y) * q_he.y + abs(q_ax[2].y) * q_he.z
            var wz = abs(q_ax[0].z) * q_he.x + abs(q_ax[1].z) * q_he.y + abs(q_ax[2].z) * q_he.z
            var wext = Vec3(wx, wy, wz)
            var s_lo = Vec3(min(l_p0.x, l_p1.x), min(l_p0.y, l_p1.y), min(l_p0.z, l_p1.z)) - wext
            var s_hi = Vec3(max(l_p0.x, l_p1.x), max(l_p0.y, l_p1.y), max(l_p0.z, l_p1.z)) + wext

            var tris = List[MeshTriangle]()
            var tids = List[Int]()
            if b.shape == BODY_SHAPE_MESH_BVH8:
                ref bvh = self.meshes[b.shape_index]
                var cand_idx = List[Int]()
                bvh._query_aabb_triangles(s_lo, s_hi, cand_idx)
                for k in range(len(cand_idx)):
                    var ti = cand_idx[k]
                    tris.append(bvh.triangles[ti])
                    tids.append(ti)
            else:
                self.heightfields[b.shape_index]._query_aabb_triangles(s_lo, s_hi, tris, tids)

            var best_sub_toi = Float32(1.0)
            for k in range(len(tris)):
                ref tri = tris[k]
                var tv = List[Vec3](capacity=3)
                tv.append(tri.v0); tv.append(tri.v1); tv.append(tri.v2)
                var thull = ConvexHull(tv)
                var l_sub_p1 = l_p0 + (l_p1 - l_p0) * best_sub_toi
                var sh = _swept_inflated_hull_cast(
                    query_hull, inflate_r, l_p0, l_sub_p1, l_rot,
                    thull, Vec3(), Vec3(), Quat(),
                )
                if sh.hit:
                    var t_cand = sh.toi * best_sub_toi
                    if t_cand <= best_sub_toi:
                        best_sub_toi = t_cand
                        var l_n = sh.normal
                        # Smooth inactive triangle edges so swept capsules/boxes never snag internal seams
                        if tri.active_edges != EDGE_ACTIVE_ALL:
                            var cp_tri = tri.closest_point(sh.point + l_n * max(Float32(0.01), inflate_r))
                            l_n = cp_tri[1]
                        out.hit = True
                        out.fraction = t_cand * max_frac
                        out.distance = out.fraction * total_dist
                        out.point = b.pos + b.rot.rotate(sh.point)
                        out.normal = b.rot.rotate(l_n).normalized()
                        out.body_slot = slot
                        out.body_id = b.id
                        out.sub_index = tids[k]
                        out.material_id = Int(tri.material_id) if tri.material_id != 0 else b.material
            return out^

        if b.shape == BODY_SHAPE_COMPOUND and b.shape_index >= 0 and b.shape_index < len(self.compounds):
            ref cmp = self.compounds[b.shape_index]
            var best_sub = Float32(1.0)
            for ci in range(len(cmp.children)):
                ref ch = cmp.children[ci]
                var wp_wr = ch.world_pose(b.pos, b.rot)
                var wp = wp_wr[0]
                var wr = wp_wr[1]
                var c_p1 = center + sub_delta * best_sub
                var c_hit = CcdHit()
                if ch.shape == COMPOUND_CHILD_SPHERE:
                    c_hit = _swept_inflated_hull_cast(
                        query_hull, inflate_r + ch.radius, center, c_p1, rot,
                        _make_point_hull(), wp, wp, Quat(),
                    )
                elif ch.shape == COMPOUND_CHILD_CAPSULE:
                    c_hit = _swept_inflated_hull_cast(
                        query_hull, inflate_r + ch.radius, center, c_p1, rot,
                        _make_segment_hull(ch.half_height), wp, wp, wr,
                    )
                elif ch.shape == COMPOUND_CHILD_BOX:
                    c_hit = _swept_inflated_hull_cast(
                        query_hull, inflate_r, center, c_p1, rot,
                        _make_box_hull(ch.half_extents), wp, wp, wr,
                    )
                elif ch.shape == COMPOUND_CHILD_CONVEX:
                    c_hit = _swept_inflated_hull_cast(
                        query_hull, inflate_r, center, c_p1, rot,
                        ch.hull, wp, wp, wr,
                    )
                if c_hit.hit:
                    var t_cand = c_hit.toi * best_sub
                    if t_cand <= best_sub:
                        best_sub = t_cand
                        out.hit = True
                        out.fraction = t_cand * max_frac
                        out.distance = out.fraction * total_dist
                        out.point = c_hit.point
                        out.normal = c_hit.normal
                        out.body_slot = slot
                        out.body_id = b.id
                        out.sub_index = ci
                        out.material_id = b.material
            return out^

        return out^

    def _shapecast_hull_world(
        self,
        query_hull: ConvexHull,
        inflate_r: Float32,
        q_ext: Vec3,
        center: Vec3,
        rot: Quat,
        delta: Vec3,
        filter: CollisionFilter,
        ignore_body_id: Int,
    ) -> SceneCastHit:
        var best = SceneCastHit()
        var end = center + delta
        var qlo = Vec3(min(center.x, end.x), min(center.y, end.y), min(center.z, end.z)) - q_ext
        var qhi = Vec3(max(center.x, end.x), max(center.y, end.y), max(center.z, end.z)) + q_ext

        var cands = List[Int]()
        self._query_bvh8_aabb_candidates(qlo, qhi, filter, ignore_body_id, cands)
        for i in range(len(cands)):
            var h = self._shapecast_hull_vs_body(
                query_hull, inflate_r, center, rot, delta, cands[i], best.fraction,
            )
            if h.hit and (
                h.fraction < best.fraction
                or (h.fraction == best.fraction and h.body_id < best.body_id)
            ):
                best = h
        return best^

    def shapecast_sphere(
        self,
        center: Vec3,
        radius: Float32,
        delta: Vec3,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> SceneCastHit:
        var r = max(Float32(0.0), radius)
        return self._shapecast_hull_world(
            _make_point_hull(), r, Vec3(r, r, r), center, Quat(), delta, filter, ignore_body_id,
        )

    def shapecast_capsule(
        self,
        center: Vec3,
        radius: Float32,
        half_height: Float32,
        rot: Quat,
        delta: Vec3,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> SceneCastHit:
        var r = max(Float32(0.0), radius)
        var hh = max(Float32(0.0), half_height)
        var ax = rot.rotate(Vec3(0, hh, 0))
        var ext = Vec3(abs(ax.x) + r, abs(ax.y) + r, abs(ax.z) + r)
        return self._shapecast_hull_world(
            _make_segment_hull(hh), r, ext, center, rot, delta, filter, ignore_body_id,
        )

    def shapecast_capsule(
        self,
        center: Vec3,
        radius: Float32,
        half_height: Float32,
        delta: Vec3,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> SceneCastHit:
        return self.shapecast_capsule(center, radius, half_height, Quat(), delta, filter, ignore_body_id)

    def shapecast_box(
        self,
        center: Vec3,
        half_extents: Vec3,
        rot: Quat,
        delta: Vec3,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> SceneCastHit:
        var axes = get_box_axes(rot)
        var wx = abs(axes[0].x) * half_extents.x + abs(axes[1].x) * half_extents.y + abs(axes[2].x) * half_extents.z
        var wy = abs(axes[0].y) * half_extents.x + abs(axes[1].y) * half_extents.y + abs(axes[2].y) * half_extents.z
        var wz = abs(axes[0].z) * half_extents.x + abs(axes[1].z) * half_extents.y + abs(axes[2].z) * half_extents.z
        return self._shapecast_hull_world(
            _make_box_hull(half_extents), Float32(0.0), Vec3(wx, wy, wz),
            center, rot, delta, filter, ignore_body_id,
        )

    def _sphere_overlaps_body(self, center: Vec3, radius: Float32, slot: Int) -> Bool:
        ref b = self.bodies[slot]
        if b.shape == BODY_SHAPE_SPHERE:
            return collide_sphere_sphere(center, radius, b.pos, b.radius).has_collision
        if b.shape == BODY_SHAPE_CAPSULE:
            var ends = b.capsule_endpoints()
            return collide_sphere_capsule(center, radius, ends[0], ends[1], b.radius).has_collision
        if b.shape == BODY_SHAPE_BOX:
            return collide_sphere_box(center, radius, b.pos, b.rot, b.half_extents).has_collision
        if b.shape == BODY_SHAPE_CONVEX_HULL and b.shape_index >= 0 and b.shape_index < len(self.hulls):
            return self._collide_sphere_hull(center, radius, self.hulls[b.shape_index], b.pos, b.rot, 0, 1).has_collision
        if b.shape == BODY_SHAPE_MESH_BVH8 and b.shape_index >= 0 and b.shape_index < len(self.meshes):
            var l_pos = b.rot.conjugate().rotate(center - b.pos)
            return self.meshes[b.shape_index].collide_sphere(l_pos, radius).has_collision
        if b.shape == BODY_SHAPE_HEIGHTFIELD and b.shape_index >= 0 and b.shape_index < len(self.heightfields):
            var l_pos = b.rot.conjugate().rotate(center - b.pos)
            return self.heightfields[b.shape_index].collide_sphere(l_pos, radius).has_collision
        if b.shape == BODY_SHAPE_COMPOUND and b.shape_index >= 0 and b.shape_index < len(self.compounds):
            ref cmp = self.compounds[b.shape_index]
            for ci in range(len(cmp.children)):
                ref ch = cmp.children[ci]
                var wp_wr = ch.world_pose(b.pos, b.rot)
                if ch.shape == COMPOUND_CHILD_SPHERE and collide_sphere_sphere(center, radius, wp_wr[0], ch.radius).has_collision:
                    return True
                elif ch.shape == COMPOUND_CHILD_CAPSULE:
                    var ax = wp_wr[1].rotate(Vec3(0, ch.half_height, 0))
                    if collide_sphere_capsule(center, radius, wp_wr[0] - ax, wp_wr[0] + ax, ch.radius).has_collision:
                        return True
                elif ch.shape == COMPOUND_CHILD_BOX and collide_sphere_box(center, radius, wp_wr[0], wp_wr[1], ch.half_extents).has_collision:
                    return True
                elif ch.shape == COMPOUND_CHILD_CONVEX and self._collide_sphere_hull(center, radius, ch.hull, wp_wr[0], wp_wr[1], 0, 1).has_collision:
                    return True
        return False

    def overlap_sphere(
        self,
        center: Vec3,
        radius: Float32,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> List[Int]:
        """Returns sorted body slots whose geometry overlaps `Sphere(center, radius)`."""
        var rv = Vec3(radius, radius, radius)
        var cands = List[Int]()
        self._query_bvh8_aabb_candidates(center - rv, center + rv, filter, ignore_body_id, cands)
        var out = List[Int]()
        for i in range(len(cands)):
            if self._sphere_overlaps_body(center, radius, cands[i]):
                out.append(cands[i])
        for i in range(1, len(out)):
            var key = out[i]
            var j = i - 1
            while j >= 0 and out[j] > key:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = key
        return out^

    def overlap_aabb(
        self,
        lo: Vec3,
        hi: Vec3,
        filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
    ) -> List[Int]:
        """Returns sorted body slots overlapping `[lo, hi]`."""
        var out = List[Int]()
        self._query_bvh8_aabb_candidates(lo, hi, filter, ignore_body_id, out)
        for i in range(1, len(out)):
            var key = out[i]
            var j = i - 1
            while j >= 0 and out[j] > key:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = key
        return out^

    # =========================================================================
    # Narrowphase & Solver Stepping
    # =========================================================================

    def _body_as_hull(self, slot: Int) -> ConvexHull:
        ref b = self.bodies[slot]
        if b.shape == BODY_SHAPE_CONVEX_HULL and b.shape_index >= 0 and b.shape_index < len(self.hulls):
            return self.hulls[b.shape_index].copy()
        if b.shape == BODY_SHAPE_BOX:
            return _make_box_hull(b.half_extents)
        if b.shape == BODY_SHAPE_CAPSULE:
            return _make_capsule_hull(b.radius, b.half_height)
        return _make_capsule_hull(b.radius, Float32(0.0))

    def _collide_sphere_hull(
        self,
        s_pos: Vec3,
        s_rad: Float32,
        hull: ConvexHull,
        h_pos: Vec3,
        h_rot: Quat,
        row_a: Int,
        row_b: Int,
    ) -> ContactManifold:
        """Exact GJK closest-point sphere vs convex hull (normal points Sphere -> Hull)."""
        var m = ContactManifold(row_a, row_b)
        var pt_hull = _make_point_hull()
        var gjk = gjk_closest_points(pt_hull, s_pos, Quat(), hull, h_pos, h_rot)
        if not gjk[0]:
            var dist = gjk[3]
            if dist <= s_rad:
                var cb = gjk[2]
                var diff = cb - s_pos
                var d = diff.length()
                var n = diff * (Float32(1.0) / d) if d > Float32(1.0e-6) else Vec3(0, 1, 0)
                var depth = s_rad - d
                m.add_point(cb - n * (depth * Float32(0.5)), n, depth, 0)
            return m^
        var sph_hull = _make_capsule_hull(s_rad, Float32(0.0))
        return collide_convex_convex(sph_hull, s_pos, Quat(), hull, h_pos, h_rot, row_a, row_b)

    def _collide_dynamic_vs_mesh(
        self,
        dyn_slot: Int,
        mesh_slot: Int,
    ) -> ContactManifold:
        """Collides body `dyn_slot` against static BVH8 mesh `mesh_slot`.
        Returns manifold with normal pointing `Mesh -> Dynamic`."""
        ref db = self.bodies[dyn_slot]
        ref mb = self.bodies[mesh_slot]
        if mb.shape_index < 0 or mb.shape_index >= len(self.meshes):
            return ContactManifold(dyn_slot, mesh_slot)
        ref bvh = self.meshes[mb.shape_index]
        var inv_mr = mb.rot.conjugate()
        var l_pos = inv_mr.rotate(db.pos - mb.pos)
        var l_rot = inv_mr * db.rot

        var lm = ContactManifold()
        if db.shape == BODY_SHAPE_SPHERE:
            lm = bvh.collide_sphere(l_pos, db.radius)
        elif db.shape == BODY_SHAPE_CAPSULE:
            var l_axis = l_rot.rotate(Vec3(0, db.half_height, 0))
            lm = bvh.collide_capsule(l_pos - l_axis, l_pos + l_axis, db.radius)
        elif db.shape == BODY_SHAPE_BOX:
            lm = bvh.collide_box(l_pos, l_rot, db.half_extents)
        elif db.shape == BODY_SHAPE_CONVEX_HULL and db.shape_index >= 0 and db.shape_index < len(self.hulls):
            ref hull = self.hulls[db.shape_index]
            var axes = get_box_axes(l_rot)
            var he = db.half_extents
            var wx = abs(axes[0].x) * he.x + abs(axes[1].x) * he.y + abs(axes[2].x) * he.z
            var wy = abs(axes[0].y) * he.x + abs(axes[1].y) * he.y + abs(axes[2].y) * he.z
            var wz = abs(axes[0].z) * he.x + abs(axes[1].z) * he.y + abs(axes[2].z) * he.z
            var wext = Vec3(wx, wy, wz)
            var cand_tris = List[Int]()
            bvh._query_aabb_triangles(l_pos - wext, l_pos + wext, cand_tris)
            var cands = List[NarrowContactPoint]()
            for k in range(len(cand_tris)):
                var ti = cand_tris[k]
                ref tri = bvh.triangles[ti]
                var tverts = List[Vec3](capacity=3)
                tverts.append(tri.v0); tverts.append(tri.v1); tverts.append(tri.v2)
                var thull = ConvexHull(tverts)
                var tm = collide_convex_convex(hull, l_pos, l_rot, thull, Vec3(), Quat())
                if tm.has_collision:
                    for pi in range(tm.count):
                        var cp = tm.get_point(pi)
                        var n_out = -cp.normal
                        if (tri.active_edges == 0) and abs(n_out.dot(tri.normal)) > Float32(0.5):
                            n_out = tri.normal if n_out.dot(tri.normal) >= Float32(0.0) else -tri.normal
                        cands.append(NarrowContactPoint(cp.point, n_out, cp.depth, UInt32(ti * 8 + pi)))
            reduce_manifold_candidates(lm, cands)
        elif db.shape == BODY_SHAPE_COMPOUND and db.shape_index >= 0 and db.shape_index < len(self.compounds):
            ref cmp = self.compounds[db.shape_index]
            var cands = List[NarrowContactPoint]()
            for ci in range(len(cmp.children)):
                ref ch = cmp.children[ci]
                var ch_p = l_pos + l_rot.rotate(ch.local_pos)
                var ch_r = (l_rot * ch.local_rot).normalized()
                var cm = ContactManifold()
                if ch.shape == COMPOUND_CHILD_SPHERE:
                    cm = bvh.collide_sphere(ch_p, ch.radius)
                elif ch.shape == COMPOUND_CHILD_CAPSULE:
                    var ax = ch_r.rotate(Vec3(0, ch.half_height, 0))
                    cm = bvh.collide_capsule(ch_p - ax, ch_p + ax, ch.radius)
                elif ch.shape == COMPOUND_CHILD_BOX:
                    cm = bvh.collide_box(ch_p, ch_r, ch.half_extents)
                elif ch.shape == COMPOUND_CHILD_CONVEX:
                    cm = bvh.collide_box(ch_p, ch_r, ch.half_extents)
                for pi in range(cm.count):
                    var cp = cm.get_point(pi)
                    cands.append(NarrowContactPoint(cp.point, cp.normal, cp.depth, UInt32(ci * 256 + Int(cp.id))))
            reduce_manifold_candidates(lm, cands)

        var out_m = ContactManifold(dyn_slot, mesh_slot)
        if not lm.has_collision or lm.count == 0:
            return out_m^

        for pi in range(lm.count):
            var cp = lm.get_point(pi)
            var wp = mb.pos + mb.rot.rotate(cp.point)
            var wn = mb.rot.rotate(cp.normal).normalized()
            out_m.add_point(wp, wn, cp.depth, cp.id)
        return out_m^

    def _collide_dynamic_vs_heightfield(
        self,
        dyn_slot: Int,
        hf_slot: Int,
    ) -> ContactManifold:
        """Collides `dyn_slot` against `HeightfieldCollider` `hf_slot`.
        Returns manifold with normal pointing `Heightfield -> Dynamic`."""
        ref db = self.bodies[dyn_slot]
        ref hb = self.bodies[hf_slot]
        if hb.shape_index < 0 or hb.shape_index >= len(self.heightfields):
            return ContactManifold(dyn_slot, hf_slot)
        ref hf = self.heightfields[hb.shape_index]
        var inv_hr = hb.rot.conjugate()
        var l_pos = inv_hr.rotate(db.pos - hb.pos)
        var l_rot = inv_hr * db.rot

        var lm = ContactManifold()
        if db.shape == BODY_SHAPE_SPHERE:
            lm = hf.collide_sphere(l_pos, db.radius)
        elif db.shape == BODY_SHAPE_CAPSULE:
            var l_axis = l_rot.rotate(Vec3(0, db.half_height, 0))
            lm = hf.collide_capsule(l_pos - l_axis, l_pos + l_axis, db.radius)
        elif db.shape == BODY_SHAPE_BOX:
            lm = hf.collide_box(l_pos, l_rot, db.half_extents)
        elif db.shape == BODY_SHAPE_CONVEX_HULL and db.shape_index >= 0 and db.shape_index < len(self.hulls):
            lm = hf.collide_box(l_pos, l_rot, db.half_extents)
        elif db.shape == BODY_SHAPE_COMPOUND and db.shape_index >= 0 and db.shape_index < len(self.compounds):
            ref cmp = self.compounds[db.shape_index]
            var cands = List[NarrowContactPoint]()
            for ci in range(len(cmp.children)):
                ref ch = cmp.children[ci]
                var ch_p = l_pos + l_rot.rotate(ch.local_pos)
                var ch_r = (l_rot * ch.local_rot).normalized()
                var cm: ContactManifold
                if ch.shape == COMPOUND_CHILD_SPHERE:
                    cm = hf.collide_sphere(ch_p, ch.radius)
                elif ch.shape == COMPOUND_CHILD_CAPSULE:
                    var ax = ch_r.rotate(Vec3(0, ch.half_height, 0))
                    cm = hf.collide_capsule(ch_p - ax, ch_p + ax, ch.radius)
                else:
                    cm = hf.collide_box(ch_p, ch_r, ch.half_extents)
                for pi in range(cm.count):
                    var cp = cm.get_point(pi)
                    cands.append(NarrowContactPoint(cp.point, cp.normal, cp.depth, UInt32(ci * 256 + Int(cp.id))))
            reduce_manifold_candidates(lm, cands)

        var out_m = ContactManifold(dyn_slot, hf_slot)
        if not lm.has_collision or lm.count == 0:
            return out_m^
        for pi in range(lm.count):
            var cp = lm.get_point(pi)
            var wp = hb.pos + hb.rot.rotate(cp.point)
            var wn = hb.rot.rotate(cp.normal).normalized()
            out_m.add_point(wp, wn, cp.depth, cp.id)
        return out_m^

    def _child_as_hull(self, ch: CompoundChild) -> ConvexHull:
        if ch.shape == COMPOUND_CHILD_CONVEX:
            return ch.hull.copy()
        if ch.shape == COMPOUND_CHILD_BOX:
            return _make_box_hull(ch.half_extents)
        if ch.shape == COMPOUND_CHILD_CAPSULE:
            return _make_capsule_hull(ch.radius, ch.half_height)
        return _make_capsule_hull(ch.radius, Float32(0.0))

    def _stamp_manifold_feature_prefix(self, mut m: ContactManifold, prefix: UInt32):
        if m.count >= 1:
            m.p0.id = prefix | (m.p0.id & UInt32(15))
        if m.count >= 2:
            m.p1.id = prefix | (m.p1.id & UInt32(15))
        if m.count >= 3:
            m.p2.id = prefix | (m.p2.id & UInt32(15))
        if m.count >= 4:
            m.p3.id = prefix | (m.p3.id & UInt32(15))

    def _collide_pair_into(
        self,
        row_a: Int,
        row_b: Int,
        mut out_manifolds: List[ContactManifold],
        speculative_dist: Float32 = 0.0,
    ):
        """Generates one or more narrowphase contact manifolds for `(row_a, row_b)`.
        Compound shapes emit per-child manifolds (with child AABB culling) so distinct
        child face normals are never merged or corrupted into a single normal."""
        ref a = self.bodies[row_a]
        ref b = self.bodies[row_b]
        var sa = a.shape
        var sb = b.shape

        if sa == BODY_SHAPE_COMPOUND or sb == BODY_SHAPE_COMPOUND:
            if (
                sa == BODY_SHAPE_MESH_BVH8 or sa == BODY_SHAPE_HEIGHTFIELD
                or sb == BODY_SHAPE_MESH_BVH8 or sb == BODY_SHAPE_HEIGHTFIELD
            ):
                var m = self._collide_pair(row_a, row_b, speculative_dist)
                if m.has_collision and m.count > 0:
                    out_manifolds.append(m^)
                return

            var pad = Vec3(speculative_dist + Float32(1.0e-3), speculative_dist + Float32(1.0e-3), speculative_dist + Float32(1.0e-3))
            if sa == BODY_SHAPE_COMPOUND and sb == BODY_SHAPE_COMPOUND:
                ref ca_cmp = self.compounds[a.shape_index]
                ref cb_cmp = self.compounds[b.shape_index]
                var nb = len(cb_cmp.children)
                var b_lo = List[Vec3](capacity=nb)
                var b_hi = List[Vec3](capacity=nb)
                for ib in range(nb):
                    var bb = cb_cmp.children[ib].tight_aabb(b.pos, b.rot)
                    b_lo.append(bb[0] - pad)
                    b_hi.append(bb[1] + pad)
                for ia in range(len(ca_cmp.children)):
                    var ab = ca_cmp.children[ia].tight_aabb(a.pos, a.rot)
                    var alo = ab[0] - pad
                    var ahi = ab[1] + pad
                    var has_ha = False
                    var wa = (Vec3(), Quat())
                    var ha = ConvexHull()
                    for ib in range(nb):
                        if not boxes_overlap(alo, ahi, b_lo[ib], b_hi[ib]):
                            continue
                        if not has_ha:
                            wa = ca_cmp.children[ia].world_pose(a.pos, a.rot)
                            ha = self._child_as_hull(ca_cmp.children[ia])
                            has_ha = True
                        var wb = cb_cmp.children[ib].world_pose(b.pos, b.rot)
                        var hb = self._child_as_hull(cb_cmp.children[ib])
                        var cm = collide_convex_convex(ha, wa[0], wa[1], hb, wb[0], wb[1], row_a, row_b)
                        if cm.has_collision and cm.count > 0:
                            self._stamp_manifold_feature_prefix(cm, UInt32((ia * 32 + ib) << 4))
                            out_manifolds.append(cm^)
                return
            elif sa == BODY_SHAPE_COMPOUND:
                ref ca_cmp = self.compounds[a.shape_index]
                var bb = b.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
                var blo = bb[0] - pad
                var bhi = bb[1] + pad
                var has_hb = False
                var hb = ConvexHull()
                for ia in range(len(ca_cmp.children)):
                    var ab = ca_cmp.children[ia].tight_aabb(a.pos, a.rot)
                    if not boxes_overlap(ab[0] - pad, ab[1] + pad, blo, bhi):
                        continue
                    if not has_hb:
                        hb = self._body_as_hull(row_b)
                        has_hb = True
                    var wa = ca_cmp.children[ia].world_pose(a.pos, a.rot)
                    var ha = self._child_as_hull(ca_cmp.children[ia])
                    var cm = collide_convex_convex(ha, wa[0], wa[1], hb, b.pos, b.rot, row_a, row_b)
                    if cm.has_collision and cm.count > 0:
                        self._stamp_manifold_feature_prefix(cm, UInt32(ia << 4))
                        out_manifolds.append(cm^)
                return
            else:
                ref cb_cmp = self.compounds[b.shape_index]
                var ab = a.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
                var alo = ab[0] - pad
                var ahi = ab[1] + pad
                var has_ha = False
                var ha = ConvexHull()
                for ib in range(len(cb_cmp.children)):
                    var bb = cb_cmp.children[ib].tight_aabb(b.pos, b.rot)
                    if not boxes_overlap(alo, ahi, bb[0] - pad, bb[1] + pad):
                        continue
                    if not has_ha:
                        ha = self._body_as_hull(row_a)
                        has_ha = True
                    var wb = cb_cmp.children[ib].world_pose(b.pos, b.rot)
                    var hb = self._child_as_hull(cb_cmp.children[ib])
                    var cm = collide_convex_convex(ha, a.pos, a.rot, hb, wb[0], wb[1], row_a, row_b)
                    if cm.has_collision and cm.count > 0:
                        self._stamp_manifold_feature_prefix(cm, UInt32(ib << 4))
                        out_manifolds.append(cm^)
                return

        var m = self._collide_pair(row_a, row_b, speculative_dist)
        if m.has_collision and m.count > 0:
            out_manifolds.append(m^)

    def _collide_pair(self, row_a: Int, row_b: Int, speculative_dist: Float32 = 0.0) -> ContactManifold:
        """Discrete narrowphase between `row_a` and `row_b`.
        Invariant: returned `normal` ALWAYS points from `row_a` toward `row_b` (`A -> B`)."""
        ref a = self.bodies[row_a]
        ref b = self.bodies[row_b]
        var sa = a.shape
        var sb = b.shape

        if (sa == BODY_SHAPE_MESH_BVH8 or sa == BODY_SHAPE_HEIGHTFIELD) and (
            sb == BODY_SHAPE_MESH_BVH8 or sb == BODY_SHAPE_HEIGHTFIELD
        ):
            return ContactManifold(row_a, row_b)

        if sb == BODY_SHAPE_MESH_BVH8:
            var m = self._collide_dynamic_vs_mesh(row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^
        if sa == BODY_SHAPE_MESH_BVH8:
            var m = self._collide_dynamic_vs_mesh(row_b, row_a)
            m.body_a = row_a
            m.body_b = row_b
            return m^

        if sb == BODY_SHAPE_HEIGHTFIELD:
            var m = self._collide_dynamic_vs_heightfield(row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^
        if sa == BODY_SHAPE_HEIGHTFIELD:
            var m = self._collide_dynamic_vs_heightfield(row_b, row_a)
            m.body_a = row_a
            m.body_b = row_b
            return m^

        if sa == BODY_SHAPE_COMPOUND or sb == BODY_SHAPE_COMPOUND:
            var ms = List[ContactManifold]()
            self._collide_pair_into(row_a, row_b, ms, speculative_dist)
            if len(ms) > 0:
                return ms[0].copy()
            return ContactManifold(row_a, row_b)

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_SPHERE:
            return collide_sphere_sphere(a.pos, a.radius, b.pos, b.radius, row_a, row_b, speculative_dist)

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_CAPSULE:
            var eb = b.capsule_endpoints()
            return collide_sphere_capsule(a.pos, a.radius, eb[0], eb[1], b.radius, row_a, row_b, speculative_dist)
        if sa == BODY_SHAPE_CAPSULE and sb == BODY_SHAPE_SPHERE:
            var ea = a.capsule_endpoints()
            var m = collide_sphere_capsule(b.pos, b.radius, ea[0], ea[1], a.radius, row_a, row_b, speculative_dist)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_BOX:
            return collide_sphere_box(a.pos, a.radius, b.pos, b.rot, b.half_extents, row_a, row_b, speculative_dist)
        if sa == BODY_SHAPE_BOX and sb == BODY_SHAPE_SPHERE:
            var m = collide_sphere_box(b.pos, b.radius, a.pos, a.rot, a.half_extents, row_a, row_b, speculative_dist)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^

        if sa == BODY_SHAPE_CAPSULE and sb == BODY_SHAPE_CAPSULE:
            var ea = a.capsule_endpoints()
            var eb = b.capsule_endpoints()
            return collide_capsule_capsule(ea[0], ea[1], a.radius, eb[0], eb[1], b.radius, row_a, row_b, speculative_dist)

        if sa == BODY_SHAPE_CAPSULE and sb == BODY_SHAPE_BOX:
            var ea = a.capsule_endpoints()
            return collide_capsule_box(ea[0], ea[1], a.radius, b.pos, b.rot, b.half_extents, row_a, row_b)
        if sa == BODY_SHAPE_BOX and sb == BODY_SHAPE_CAPSULE:
            var eb = b.capsule_endpoints()
            var m = collide_capsule_box(eb[0], eb[1], b.radius, a.pos, a.rot, a.half_extents, row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^

        if sa == BODY_SHAPE_BOX and sb == BODY_SHAPE_BOX:
            return collide_box_box(a.pos, a.rot, a.half_extents, b.pos, b.rot, b.half_extents, row_a, row_b)

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_CONVEX_HULL:
            return self._collide_sphere_hull(a.pos, a.radius, self.hulls[b.shape_index], b.pos, b.rot, row_a, row_b)
        if sa == BODY_SHAPE_CONVEX_HULL and sb == BODY_SHAPE_SPHERE:
            var m = self._collide_sphere_hull(b.pos, b.radius, self.hulls[a.shape_index], a.pos, a.rot, row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^

        var ha = self._body_as_hull(row_a)
        var hb = self._body_as_hull(row_b)
        return collide_convex_convex(ha, a.pos, a.rot, hb, b.pos, b.rot, row_a, row_b)

    def _sync_motion_modes(mut self):
        for i in range(len(self.bodies)):
            ref b = self.bodies[i]
            if b.motion_mode == BODY_MOTION_KINEMATIC:
                b.is_static = True
                b.inv_mass = 0.0
                b.local_inertia = Vec3()
                b.can_sleep = False
                if b.linear_vel.length_squared() > Float32(0.0) or b.angular_vel.length_squared() > Float32(0.0):
                    b.awake = True
            elif b.motion_mode == BODY_MOTION_DYNAMIC and b.is_static:
                b.motion_mode = BODY_MOTION_STATIC
                b.inv_mass = 0.0
                b.local_inertia = Vec3()
            elif b.motion_mode == BODY_MOTION_STATIC and (not b.is_static) and b.inv_mass > Float32(0.0):
                b.motion_mode = BODY_MOTION_DYNAMIC

    def _run_ccd_clamp(
        mut self,
        dt: Float32,
        mut clamped_slots: List[Int],
        mut clamped_vpost: List[Vec3],
        mut clamped_rem: List[Float32],
    ):
        """Swept anti-tunneling clamp for bodies with `enable_ccd = True` against all collider shapes
        (`Sphere`, `Box`, `Capsule`, `ConvexHull`, `Bvh8MeshCollider`, `HeightfieldCollider`, `CompoundShape`).
        Applies a momentum-conserving bilateral impulse to any hit dynamic body and scales the
        remaining-tick solver velocity by `(1 - toi)` to prevent double-integration."""
        for i in range(len(self.bodies)):
            if not self.bodies[i].active or self.bodies[i].is_static or not self.bodies[i].awake or not self.bodies[i].enable_ccd:
                continue
            var step_vec = self.bodies[i].linear_vel * dt
            var r_core = self.bodies[i].radius if (
                self.bodies[i].shape == BODY_SHAPE_SPHERE or self.bodies[i].shape == BODY_SHAPE_CAPSULE
            ) else min(
                self.bodies[i].half_extents.x, min(self.bodies[i].half_extents.y, self.bodies[i].half_extents.z)
            )
            if step_vec.length_squared() <= r_core * r_core:
                continue

            var p0 = self.bodies[i].pos
            var p1 = p0 + step_vec
            var r_vec = Vec3(r_core, r_core, r_core)
            var s_lo = Vec3(min(p0.x, p1.x), min(p0.y, p1.y), min(p0.z, p1.z)) - r_vec
            var s_hi = Vec3(max(p0.x, p1.x), max(p0.y, p1.y), max(p0.z, p1.z)) + r_vec
            var cands = List[Int]()
            self._query_bvh8_aabb_candidates(s_lo, s_hi, self.bodies[i].filter, self.bodies[i].id, cands)

            var best_toi = Float32(1.0)
            var best_normal = Vec3(0.0, 1.0, 0.0)
            var best_slot = -1
            var hit_any = False
            var proj_hull = self._body_as_hull(i)
            var proj_rot = self.bodies[i].rot
            var proj_is_sphere = self.bodies[i].shape == BODY_SHAPE_SPHERE

            for ck in range(len(cands)):
                var m_slot = cands[ck]
                if m_slot == i or not self.bodies[m_slot].active:
                    continue
                ref mb = self.bodies[m_slot]
                var mb_p0 = mb.pos
                var mb_p1 = mb.pos + (mb.linear_vel * dt if mb.awake else Vec3())

                if mb.shape == BODY_SHAPE_MESH_BVH8 and mb.shape_index >= 0 and mb.shape_index < len(self.meshes):
                    var sh = self.meshes[mb.shape_index].swept_sphere_moving_bvh8(p0, p1, r_core, mb_p0, mb.rot, mb_p1, mb.rot)
                    if sh.hit and sh.toi < best_toi:
                        best_toi = sh.toi
                        best_normal = sh.normal
                        best_slot = m_slot
                        hit_any = True
                elif mb.shape == BODY_SHAPE_HEIGHTFIELD and mb.shape_index >= 0 and mb.shape_index < len(self.heightfields):
                    var inv_mr = mb.rot.conjugate()
                    var sh = self.heightfields[mb.shape_index].sweep_sphere(
                        inv_mr.rotate(p0 - mb_p0), inv_mr.rotate(step_vec), r_core, best_toi,
                    )
                    if sh.hit and sh.toi < best_toi:
                        best_toi = sh.toi
                        best_normal = mb.rot.rotate(sh.normal).normalized()
                        best_slot = m_slot
                        hit_any = True
                elif mb.shape == BODY_SHAPE_SPHERE:
                    var sh = swept_sphere_moving_sphere(p0, p1, r_core, mb_p0, mb_p1, mb.radius)
                    if sh.hit and sh.toi < best_toi:
                        best_toi = sh.toi
                        best_normal = sh.normal
                        best_slot = m_slot
                        hit_any = True
                elif mb.shape == BODY_SHAPE_BOX:
                    var rel_p1 = p0 + (step_vec - (mb_p1 - mb_p0))
                    var sh = swept_sphere_obb(p0, rel_p1, r_core, mb_p0, mb.rot, mb.half_extents) if proj_is_sphere else _swept_inflated_hull_cast(
                        proj_hull, Float32(0.0), p0, p1, proj_rot, _make_box_hull(mb.half_extents), mb_p0, mb_p1, mb.rot,
                    )
                    if sh.hit and sh.toi < best_toi:
                        best_toi = sh.toi
                        best_normal = sh.normal
                        best_slot = m_slot
                        hit_any = True
                elif mb.shape == BODY_SHAPE_CAPSULE or mb.shape == BODY_SHAPE_CONVEX_HULL:
                    var thull = self._body_as_hull(m_slot)
                    var sh = _swept_inflated_hull_cast(proj_hull, Float32(0.0), p0, p1, proj_rot, thull, mb_p0, mb_p1, mb.rot)
                    if sh.hit and sh.toi < best_toi:
                        best_toi = sh.toi
                        best_normal = sh.normal
                        best_slot = m_slot
                        hit_any = True
                elif mb.shape == BODY_SHAPE_COMPOUND and mb.shape_index >= 0 and mb.shape_index < len(self.compounds):
                    ref cmp = self.compounds[mb.shape_index]
                    for ci in range(len(cmp.children)):
                        var wp_wr = cmp.children[ci].world_pose(mb_p0, mb.rot)
                        var wp_wr1 = cmp.children[ci].world_pose(mb_p1, mb.rot)
                        var ch_hull = self._child_as_hull(cmp.children[ci])
                        var sh = _swept_inflated_hull_cast(proj_hull, Float32(0.0), p0, p1, proj_rot, ch_hull, wp_wr[0], wp_wr1[0], wp_wr[1])
                        if sh.hit and sh.toi < best_toi:
                            best_toi = sh.toi
                            best_normal = sh.normal
                            best_slot = m_slot
                            hit_any = True

            if hit_any and best_toi < Float32(1.0):
                var n = best_normal.normalized()
                var v_a = self.bodies[i].linear_vel
                var other_dyn = best_slot >= 0 and not self.bodies[best_slot].is_static
                var v_b = self.bodies[best_slot].linear_vel if best_slot >= 0 else Vec3()
                var inv_m_a = self.bodies[i].inv_mass
                var inv_m_b = self.bodies[best_slot].inv_mass if other_dyn else Float32(0.0)

                # Advance body i to exact impact point with 1 mm clearance along outward normal
                self.bodies[i].pos = p0 + step_vec * best_toi + n * Float32(0.001)

                var vn = (v_a - v_b).dot(n)
                var v_a_post = v_a
                if vn < Float32(0.0) and (inv_m_a + inv_m_b) > Float32(1.0e-10):
                    var e = max(self.bodies[i].restitution, self.bodies[best_slot].restitution) if best_slot >= 0 else self.bodies[i].restitution
                    var j_n = -(Float32(1.0) + e) * vn / (inv_m_a + inv_m_b)
                    v_a_post = v_a + n * (j_n * inv_m_a)
                    if other_dyn:
                        if not self.bodies[best_slot].awake:
                            self.wake(best_slot)
                        self.bodies[best_slot].linear_vel = v_b - n * (j_n * inv_m_b)

                var rem_frac = max(Float32(0.0), Float32(1.0) - best_toi)
                self.bodies[i].linear_vel = v_a_post * rem_frac
                clamped_slots.append(i)
                clamped_vpost.append(v_a_post)
                clamped_rem.append(rem_frac)

    def _update_sensors(mut self):
        if len(self.sensors) == 0:
            return
        var cur_a = List[Int]()
        var cur_b = List[Int]()
        var keys = List[Int]()
        for si in range(len(self.sensors)):
            ref s = self.sensors[si]
            for bi in range(len(self.bodies)):
                if not self.bodies[bi].active or self.bodies[bi].is_static:
                    continue
                if not s.filter.touches(self.bodies[bi].filter):
                    continue
                var tb = self.bodies[bi].tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
                if s.overlaps_box(tb[0], tb[1]):
                    cur_a.append(s.id)
                    cur_b.append(self.bodies[bi].id)
                    keys.append(pair_key(s.id, self.bodies[bi].id))

        var order = stable_order(keys)
        var sa = List[Int](capacity=len(order))
        var sb = List[Int](capacity=len(order))
        for idx in order:
            sa.append(cur_a[idx])
            sb.append(cur_b[idx])

        var old = 0
        for k in range(len(sa)):
            var a = sa[k]
            var b = sb[k]
            while old < len(self.sensor_inside_a) and (
                self.sensor_inside_a[old] < a or (self.sensor_inside_a[old] == a and self.sensor_inside_b[old] < b)
            ):
                self.events.append(PhysicsEvent(EVENT_SENSOR_END, self.sensor_inside_a[old], self.sensor_inside_b[old]))
                old += 1
            if old < len(self.sensor_inside_a) and self.sensor_inside_a[old] == a and self.sensor_inside_b[old] == b:
                old += 1
            else:
                self.events.append(PhysicsEvent(EVENT_SENSOR_BEGIN, a, b))
        while old < len(self.sensor_inside_a):
            self.events.append(PhysicsEvent(EVENT_SENSOR_END, self.sensor_inside_a[old], self.sensor_inside_b[old]))
            old += 1

        self.sensor_inside_a = sa^
        self.sensor_inside_b = sb^

    def _trigger_overlaps_body(self, tv: TriggerVolume, slot: Int) -> Tuple[Bool, Vec3]:
        ref b = self.bodies[slot]
        if tv.shape == TRIGGER_SHAPE_SPHERE:
            return (self._sphere_overlaps_body(tv.pos, tv.radius, slot), b.pos)
        if tv.shape == TRIGGER_SHAPE_CAPSULE:
            var ta = tv.pos - tv.rot.rotate(Vec3(0, tv.half_height, 0))
            var tb = tv.pos + tv.rot.rotate(Vec3(0, tv.half_height, 0))
            if b.shape == BODY_SHAPE_SPHERE:
                var m = collide_sphere_capsule(b.pos, b.radius, ta, tb, tv.radius)
                return (m.has_collision, m.p0.point if m.has_collision else b.pos)
            if b.shape == BODY_SHAPE_CAPSULE:
                var eb = b.capsule_endpoints()
                var m = collide_capsule_capsule(ta, tb, tv.radius, eb[0], eb[1], b.radius)
                return (m.has_collision, m.p0.point if m.has_collision else b.pos)
            if b.shape == BODY_SHAPE_BOX:
                var m = collide_capsule_box(ta, tb, tv.radius, b.pos, b.rot, b.half_extents)
                return (m.has_collision, m.p0.point if m.has_collision else b.pos)
            var th = _make_capsule_hull(tv.radius, tv.half_height)
            var bh = self._body_as_hull(slot)
            var m = collide_convex_convex(th, tv.pos, tv.rot, bh, b.pos, b.rot)
            return (m.has_collision, m.p0.point if m.has_collision else b.pos)

        # TRIGGER_SHAPE_BOX (OBB)
        if b.shape == BODY_SHAPE_SPHERE:
            var m = collide_sphere_box(b.pos, b.radius, tv.pos, tv.rot, tv.half_extents)
            return (m.has_collision, m.p0.point if m.has_collision else b.pos)
        if b.shape == BODY_SHAPE_CAPSULE:
            var eb = b.capsule_endpoints()
            var m = collide_capsule_box(eb[0], eb[1], b.radius, tv.pos, tv.rot, tv.half_extents)
            return (m.has_collision, m.p0.point if m.has_collision else b.pos)
        if b.shape == BODY_SHAPE_BOX:
            var m = collide_box_box(tv.pos, tv.rot, tv.half_extents, b.pos, b.rot, b.half_extents)
            return (m.has_collision, m.p0.point if m.has_collision else b.pos)
        var box_h = _make_box_hull(tv.half_extents)
        if b.shape == BODY_SHAPE_COMPOUND and b.shape_index >= 0 and b.shape_index < len(self.compounds):
            ref cmp = self.compounds[b.shape_index]
            for ci in range(len(cmp.children)):
                var wp_wr = cmp.children[ci].world_pose(b.pos, b.rot)
                var ch_h = self._child_as_hull(cmp.children[ci])
                var cm = collide_convex_convex(box_h, tv.pos, tv.rot, ch_h, wp_wr[0], wp_wr[1])
                if cm.has_collision:
                    return (True, cm.p0.point)
            return (False, b.pos)
        var bh = self._body_as_hull(slot)
        var m = collide_convex_convex(box_h, tv.pos, tv.rot, bh, b.pos, b.rot)
        return (m.has_collision, m.p0.point if m.has_collision else b.pos)

    def _update_triggers(mut self):
        self.trigger_events.clear()
        if len(self.triggers) == 0 and len(self.trigger_inside_a) == 0:
            return

        var cur_a = List[Int]()
        var cur_b = List[Int]()
        var cur_pt = List[Vec3]()
        var keys = List[Int]()

        for ti in range(len(self.triggers)):
            ref tv = self.triggers[ti]
            if not tv.active:
                continue
            var tb = tv.tight_aabb()
            var cands = List[Int]()
            self._query_bvh8_aabb_candidates(tb[0], tb[1], tv.filter, -1, cands)
            for ci in range(len(cands)):
                var slot = cands[ci]
                ref b = self.bodies[slot]
                if not b.active or b.motion_mode == BODY_MOTION_STATIC:
                    continue
                var ov = self._trigger_overlaps_body(tv, slot)
                if ov[0]:
                    cur_a.append(tv.id)
                    cur_b.append(b.id)
                    cur_pt.append(ov[1])
                    keys.append(pair_key(tv.id, b.id))

        var order = stable_order(keys)
        var ta = List[Int](capacity=len(order))
        var tb_ids = List[Int](capacity=len(order))
        var tpts = List[Vec3](capacity=len(order))
        var prev_k = -1
        for idx in order:
            if len(ta) > 0 and keys[idx] == prev_k:
                continue
            prev_k = keys[idx]
            ta.append(cur_a[idx])
            tb_ids.append(cur_b[idx])
            tpts.append(cur_pt[idx])

        var old = 0
        for k in range(len(ta)):
            var a = ta[k]
            var b_id = tb_ids[k]
            var pt = tpts[k]
            while old < len(self.trigger_inside_a) and (
                self.trigger_inside_a[old] < a or (self.trigger_inside_a[old] == a and self.trigger_inside_b[old] < b_id)
            ):
                var oa = self.trigger_inside_a[old]
                var ob = self.trigger_inside_b[old]
                self.trigger_events.append(TriggerEvent(TRIGGER_EXIT, oa, ob, Vec3()))
                self.events.append(PhysicsEvent(TRIGGER_EXIT, oa, ob))
                old += 1
            if old < len(self.trigger_inside_a) and self.trigger_inside_a[old] == a and self.trigger_inside_b[old] == b_id:
                self.trigger_events.append(TriggerEvent(TRIGGER_STAY, a, b_id, pt))
                self.events.append(PhysicsEvent(TRIGGER_STAY, a, b_id, pt))
                old += 1
            else:
                self.trigger_events.append(TriggerEvent(TRIGGER_ENTER, a, b_id, pt))
                self.events.append(PhysicsEvent(TRIGGER_ENTER, a, b_id, pt))
        while old < len(self.trigger_inside_a):
            var oa = self.trigger_inside_a[old]
            var ob = self.trigger_inside_b[old]
            self.trigger_events.append(TriggerEvent(TRIGGER_EXIT, oa, ob, Vec3()))
            self.events.append(PhysicsEvent(TRIGGER_EXIT, oa, ob))
            old += 1

        self.trigger_inside_a = ta^
        self.trigger_inside_b = tb_ids^

    def step(mut self, dt: Float32, pool_address: Int = 0):
        """Executes one complete rigid-body simulation tick of duration `dt`."""
        set_sim_float_mode()
        self.events.clear()
        if dt <= Float32(1.0e-7):
            return

        self._sync_motion_modes()
        self.time_s += dt
        self.tick += 1
        self.force_world.tick = self.tick
        self.force_world.wind.age()
        self.cache.begin_tick()

        # 1. Sensors, Triggers & Kill volumes
        self._update_sensors()
        self._update_triggers()
        var killed_any = False
        for i in range(len(self.bodies)):
            if not self.bodies[i].active or self.bodies[i].is_static:
                continue
            var p = self.bodies[i].pos
            var killed = False
            for vi in range(len(self.force_world.volumes)):
                ref vol = self.force_world.volumes[vi]
                if vol.target == TARGET_KILL and (vol.mask & self.bodies[i].filter.category) != 0 and vol.inside(p):
                    killed = True
                    break
            if killed:
                killed_any = True
                var ev = PhysicsEvent(EVENT_KILL, self.bodies[i].id, 0, p)
                ev.material = self.bodies[i].material
                self.events.append(ev)
                self.bodies[i].active = False
                if self.bodies[i].id >= 0 and self.bodies[i].id < len(self.id_to_slot):
                    self.id_to_slot[self.bodies[i].id] = -1
                self.islands.drop(i)

        # 2. Optional anti-tunneling CCD clamp across all collider types
        var ccd_slots = List[Int]()
        var ccd_vpost = List[Vec3]()
        var ccd_rem = List[Float32]()
        self._run_ccd_clamp(dt, ccd_slots, ccd_vpost, ccd_rem)

        # 3. Broadphase over all active bodies (retained into `self.broadphase_nodes` to avoid double rebuild)
        var n_rows = len(self.bodies)
        var bp_in = BroadphaseInput()
        var spec = Vec3(SPECULATIVE_DISTANCE, SPECULATIVE_DISTANCE, SPECULATIVE_DISTANCE)
        for i in range(n_rows):
            ref b = self.bodies[i]
            if not b.active:
                bp_in.add(Vec3(), Vec3(), False, False, b.filter)
                continue
            var tb = b.tight_aabb_full(self.hulls, self.meshes, self.heightfields, self.compounds)
            var lo = tb[0] - spec
            var hi = tb[1] + spec
            var kin_moving = b.motion_mode == BODY_MOTION_KINEMATIC and (
                b.linear_vel.length_squared() > Float32(0.0) or b.angular_vel.length_squared() > Float32(0.0)
            )
            var is_moving = ((not b.is_static) and b.awake) or kin_moving
            if is_moving:
                var move = b.linear_vel * dt
                lo = Vec3(min(lo.x, lo.x + move.x), min(lo.y, lo.y + move.y), min(lo.z, lo.z + move.z))
                hi = Vec3(max(hi.x, hi.x + move.x), max(hi.y, hi.y + move.y), max(hi.z, hi.z + move.z))
                if b.angular_vel.length_squared() > Float32(1.0e-8):
                    var rot_pad = b.angular_vel.length() * dt * b.radius
                    var rp = Vec3(rot_pad, rot_pad, rot_pad)
                    lo = lo - rp
                    hi = hi + rp
            bp_in.add(lo, hi, True, is_moving, b.filter)

        var bp_res = find_pairs_into_bvh8(bp_in, self.broadphase_nodes, pool_address)

        # 4. Discrete + Speculative Narrowphase & wake sleeping dynamic bodies hit by awake/moving bodies
        var raw_manifolds = List[ContactManifold]()
        for pk in range(len(bp_res.pairs)):
            var code = bp_res.pairs[pk]
            var row_a = code // n_rows
            var row_b = code % n_rows
            if self.bodies[row_a].is_static and self.bodies[row_b].is_static:
                continue
            var before_m = len(raw_manifolds)
            self._collide_pair_into(row_a, row_b, raw_manifolds, SPECULATIVE_DISTANCE)
            if len(raw_manifolds) > before_m:
                if not self.bodies[row_a].is_static and not self.bodies[row_a].awake:
                    self.wake(row_a)
                if not self.bodies[row_b].is_static and not self.bodies[row_b].awake:
                    self.wake(row_b)

        # Wake bodies linked to awake/moving bodies via unbroken joints & SixDofMotorJoints
        for ji in range(len(self.joints)):
            if self.joints[ji].broken:
                continue
            var sa = self.find_slot_by_id(self.joints[ji].body_a) if self.joints[ji].body_a != JOINT_WORLD else -1
            var sb = self.find_slot_by_id(self.joints[ji].body_b)
            if sb < 0:
                continue
            if sa >= 0 and self.bodies[sa].awake and not self.bodies[sb].awake:
                self.wake(sb)
            elif sa >= 0 and self.bodies[sb].awake and not self.bodies[sa].awake:
                self.wake(sa)

        for s6i in range(len(self.six_dof_joints)):
            if self.six_dof_joints[s6i].broken:
                continue
            var sa = self.find_slot_by_id(self.six_dof_joints[s6i].body_a) if self.six_dof_joints[s6i].body_a != JOINT_WORLD else -1
            var sb = self.find_slot_by_id(self.six_dof_joints[s6i].body_b)
            if sb < 0:
                continue
            var driven = self.six_dof_joints[s6i].enable_linear_drive or self.six_dof_joints[s6i].enable_angular_drive
            if driven and not self.bodies[sb].is_static and not self.bodies[sb].awake:
                self.wake(sb)
            if driven and sa >= 0 and not self.bodies[sa].is_static and not self.bodies[sa].awake:
                self.wake(sa)
            if sa >= 0 and self.bodies[sa].awake and not self.bodies[sb].awake:
                self.wake(sb)
            elif sa >= 0 and self.bodies[sb].awake and not self.bodies[sa].awake:
                self.wake(sa)

        # 5. Build SolverBodies for all active, awake dynamic bodies AND moving kinematic bodies
        var row_to_solver = List[Int](length=n_rows, fill=-1)
        var awake_slots = List[Int]()
        for i in range(n_rows):
            ref b = self.bodies[i]
            if not b.active:
                continue
            var kin_moving = b.motion_mode == BODY_MOTION_KINEMATIC and (
                b.linear_vel.length_squared() > Float32(0.0) or b.angular_vel.length_squared() > Float32(0.0)
            )
            if (not b.is_static and b.awake) or kin_moving:
                row_to_solver[i] = len(awake_slots)
                awake_slots.append(i)

        var sbodies = SolverBodies()
        sbodies.reset(len(awake_slots))
        for si in range(len(awake_slots)):
            var r = awake_slots[si]
            ref b = self.bodies[r]
            sbodies.row[si] = r
            sbodies.px[si] = b.pos.x
            sbodies.py[si] = b.pos.y
            sbodies.pz[si] = b.pos.z
            sbodies.qx[si] = b.rot.x
            sbodies.qy[si] = b.rot.y
            sbodies.qz[si] = b.rot.z
            sbodies.qw[si] = b.rot.w
            sbodies.vx[si] = b.linear_vel.x
            sbodies.vy[si] = b.linear_vel.y
            sbodies.vz[si] = b.linear_vel.z
            sbodies.wx[si] = b.angular_vel.x
            sbodies.wy[si] = b.angular_vel.y
            sbodies.wz[si] = b.angular_vel.z

            if b.motion_mode == BODY_MOTION_KINEMATIC or b.is_static:
                # Infinite mass in solver (`inv_mass = 0`, `inertia = 0`), zero gravity/damping,
                # while driving exact substepped normal & surface velocity `v + w x r`!
                sbodies.inv_mass[si] = Float32(0.0)
                sbodies.lin_damp[si] = Float32(0.0)
                sbodies.ang_damp[si] = Float32(0.0)
                sbodies.gravity[si] = Float32(0.0)
                continue

            sbodies.inv_mass[si] = b.inv_mass
            sbodies.lin_damp[si] = b.linear_damping
            sbodies.ang_damp[si] = b.angular_damping
            sbodies.gravity[si] = self.gravity * b.gravity_scale
            sbodies.set_body_inertia(si, b.local_inertia)

            # Evaluate external forces, Beaufort water buoyancy/drag, and ForceWorld wind/fields
            var ext_f = b.force_accum
            var ext_tau = b.torque_accum
            var drag_target_sum = Vec3()
            var drag_rate = Float32(0.0)
            var ang_drag_rate = Float32(0.0)
            var has_field = ext_f.length_squared() > Float32(0.0) or ext_tau.length_squared() > Float32(0.0)

            var cur_submerged = Float32(0.0)
            var cur_water_id = -1
            var cur_eta = NO_WATER

            if self.water_enabled:
                if b.shape == BODY_SHAPE_SPHERE:
                    var bw = eval_sphere_buoyancy_drag(
                        b.pos, b.radius, b.linear_vel, b.angular_vel,
                        self.time_s, self.water_level, self.fluid_density, self.gravity,
                        use_waves=self.water_use_waves,
                    )
                    if bw.submerged_fraction > Float32(0.0):
                        ext_f = ext_f + bw.force
                        ext_tau = ext_tau + bw.torque
                        cur_submerged = bw.submerged_fraction
                        cur_water_id = 0
                        cur_eta = bw.surface_eta
                        has_field = True
                else:
                    var he_b = Vec3(b.radius, b.half_height + b.radius, b.radius) if b.shape == BODY_SHAPE_CAPSULE else b.half_extents
                    var bw = eval_box_buoyancy_drag(
                        b.pos, b.rot, he_b, b.linear_vel, b.angular_vel,
                        self.time_s, self.water_level, self.fluid_density, self.gravity,
                        use_waves=self.water_use_waves,
                    )
                    if bw.submerged_fraction > Float32(0.0):
                        ext_f = ext_f + bw.force
                        ext_tau = ext_tau + bw.torque
                        cur_submerged = bw.submerged_fraction
                        cur_water_id = 0
                        cur_eta = bw.surface_eta
                        has_field = True

            for vi in range(len(self.force_world.volumes)):
                ref vol = self.force_world.volumes[vi]
                if (vol.mask & b.filter.category) == 0:
                    continue
                var w = vol.weight(b.pos)
                if w <= Float32(0.0):
                    continue
                if vol.target == TARGET_FORCE:
                    ext_f = ext_f + vol.vector * (w * b.mass)
                    has_field = True
                elif vol.target == TARGET_VELOCITY:
                    var r_v = min(MAX_FIELD_RATE, vol.rate * w)
                    drag_target_sum = drag_target_sum + vol.vector * r_v
                    drag_rate += r_v
                    has_field = True
                elif vol.target == TARGET_RADIAL:
                    var d_rad = vol.center - b.pos
                    var d_len = d_rad.length()
                    if d_len > Float32(1.0e-5):
                        ext_f = ext_f + d_rad * ((vol.strength * w * b.mass) / d_len)
                        has_field = True
                elif vol.target == TARGET_WATER:
                    var surf_y = vol.surface()
                    var bw = eval_box_buoyancy_drag(
                        b.pos, b.rot, b.half_extents, b.linear_vel, b.angular_vel,
                        self.time_s, surf_y, self.fluid_density * (vol.density / Float32(1.6)),
                        self.gravity, use_waves=False,
                    )
                    if bw.submerged_fraction > Float32(0.0):
                        ext_f = ext_f + bw.force
                        ext_tau = ext_tau + bw.torque
                        var r_w = min(MAX_FIELD_RATE, vol.drag * bw.submerged_fraction)
                        drag_target_sum = drag_target_sum + vol.vector * r_w
                        drag_rate += r_w
                        ang_drag_rate += vol.angular_drag * bw.submerged_fraction
                        cur_submerged = max(cur_submerged, bw.submerged_fraction)
                        cur_water_id = vol.id
                        cur_eta = surf_y
                        has_field = True

            if len(self.force_world.wind.sources) > 0 or len(self.force_world.wind.vorticles) > 0:
                var air_v = self.force_world.wind_at(b.pos)
                if air_v.length_squared() > Float32(1.0e-6) and cur_submerged < Float32(1.0):
                    var dry_share = Float32(1.0) - cur_submerged
                    var r_air = min(MAX_FIELD_RATE, Float32(0.75) * b.wind_area_over_mass * dry_share)
                    if r_air > Float32(0.0):
                        drag_target_sum = drag_target_sum + air_v * r_air
                        drag_rate += r_air
                        has_field = True

            var now_in_water = cur_submerged > Float32(0.0)
            if now_in_water and not b.in_water:
                var ev = PhysicsEvent(
                    EVENT_WATER_ENTER,
                    b.id,
                    cur_water_id,
                    Vec3(b.pos.x, cur_eta, b.pos.z),
                    Vec3(0, 1, 0),
                    abs(b.linear_vel.y),
                    b.mass * b.linear_vel.length(),
                )
                ev.material = b.material
                self.events.append(ev)
            elif (not now_in_water) and b.in_water:
                var ev = PhysicsEvent(
                    EVENT_WATER_EXIT,
                    b.id,
                    b.water_volume_id,
                    b.pos,
                    Vec3(0, 1, 0),
                    abs(b.linear_vel.y),
                    b.mass * b.linear_vel.length(),
                )
                ev.material = b.material
                self.events.append(ev)
            b.in_water = now_in_water
            if now_in_water:
                b.water_volume_id = cur_water_id

            if has_field:
                sbodies.field[si] = Float32(1.0)
                sbodies.ax[si] = ext_f.x * b.inv_mass
                sbodies.ay[si] = ext_f.y * b.inv_mass
                sbodies.az[si] = ext_f.z * b.inv_mass
                var alpha_w = sbodies.inertia(si).apply(ext_tau)
                sbodies.aax[si] = alpha_w.x
                sbodies.aay[si] = alpha_w.y
                sbodies.aaz[si] = alpha_w.z
                var capped_rate = min(MAX_FIELD_RATE, drag_rate)
                sbodies.rate[si] = capped_rate
                if drag_rate > Float32(1.0e-8):
                    var tv = drag_target_sum * (Float32(1.0) / drag_rate)
                    sbodies.tvx[si] = tv.x
                    sbodies.tvy[si] = tv.y
                    sbodies.tvz[si] = tv.z
                sbodies.arate[si] = min(MAX_FIELD_RATE, ang_drag_rate)

        # 6. Convert narrowphase ContactManifolds into solver Manifolds (explicit A->B to B->A normal negation!)
        var solver_manifolds = List[Manifold]()
        var graph_keys = List[Int]()
        var graph_ra = List[Int]()
        var graph_rb = List[Int]()

        for k in range(len(raw_manifolds)):
            var nm = raw_manifolds[k]
            var ra = nm.body_a
            var rb = nm.body_b
            var a_dyn = not self.bodies[ra].is_static and row_to_solver[ra] >= 0
            var b_dyn = not self.bodies[rb].is_static and row_to_solver[rb] >= 0
            if not a_dyn and not b_dyn:
                continue

            var a_solv = row_to_solver[ra] >= 0
            var b_solv = row_to_solver[rb] >= 0

            var m_a: Int
            var m_b: Int
            var b_moves: Bool
            var other_id: Int
            var solver_normal: Vec3
            var cache_a: Int
            var cache_b: Int

            if a_dyn and b_solv:
                m_a = ra
                m_b = rb
                b_moves = True
                other_id = self.bodies[rb].id
                solver_normal = -nm.normal
                cache_a = min(self.bodies[ra].id, self.bodies[rb].id)
                cache_b = max(self.bodies[ra].id, self.bodies[rb].id)
            elif b_dyn and a_solv:
                m_a = rb
                m_b = ra
                b_moves = True
                other_id = self.bodies[ra].id
                solver_normal = nm.normal
                cache_a = min(self.bodies[ra].id, self.bodies[rb].id)
                cache_b = max(self.bodies[ra].id, self.bodies[rb].id)
            elif a_dyn:
                m_a = ra
                m_b = -1
                b_moves = False
                other_id = OTHER_TERRAIN if (
                    self.bodies[rb].shape == BODY_SHAPE_MESH_BVH8 or self.bodies[rb].shape == BODY_SHAPE_HEIGHTFIELD
                ) else static_other(rb)
                solver_normal = -nm.normal
                cache_a = self.bodies[ra].id
                cache_b = other_id
            else:
                m_a = rb
                m_b = -1
                b_moves = False
                other_id = OTHER_TERRAIN if (
                    self.bodies[ra].shape == BODY_SHAPE_MESH_BVH8 or self.bodies[ra].shape == BODY_SHAPE_HEIGHTFIELD
                ) else static_other(ra)
                solver_normal = nm.normal
                cache_a = self.bodies[rb].id
                cache_b = other_id

            var sm = Manifold(m_a, m_b, b_moves, other_id, solver_normal)
            sm.ia = row_to_solver[m_a]
            sm.ib = row_to_solver[m_b] if b_moves else sbodies.static_index
            sm.friction = sqrt(max(Float32(0.0), self.bodies[ra].friction * self.bodies[rb].friction))
            sm.restitution = max(self.bodies[ra].restitution, self.bodies[rb].restitution)
            sm.rolling = max(self.bodies[ra].rolling_resistance, self.bodies[rb].rolling_resistance)

            var va = self.bodies[m_a].linear_vel
            var wa = self.bodies[m_a].angular_vel
            var pa = self.bodies[m_a].pos
            var vb = self.bodies[m_b].linear_vel if b_moves else Vec3()
            var wb = self.bodies[m_b].angular_vel if b_moves else Vec3()
            var pb = self.bodies[m_b].pos if b_moves else Vec3()

            for pi in range(nm.count):
                var cp = nm.get_point(pi)
                var feat = Int(cp.id)
                var anchor = self.bodies[m_b].rot.conjugate().rotate(cp.point - pb) if b_moves else cp.point
                var cached = self.cache.warm(cache_a, cache_b, feat, solver_normal, anchor)
                var scp = SolverContactPoint(cp.point, cp.depth, feat)
                scp.id = cached.id
                scp.normal_impulse = cached.impulse
                scp.tangent_impulse = cached.tangent

                var dv = (va + wa.cross(cp.point - pa)) - (vb + wb.cross(cp.point - pb))
                scp.speed = max(Float32(0.0), -dv.dot(solver_normal))
                sm.points.append(scp)

            var pkey = manifold_key(pair_key(cache_a, cache_b), 0)
            graph_keys.append(pkey)
            graph_ra.append(m_a)
            graph_rb.append(m_b if b_moves else -1)
            solver_manifolds.append(sm^)

        # Sort manifolds deterministically by graph key for ConstraintGraph.assign
        if len(solver_manifolds) > 0:
            var m_order = stable_order(graph_keys)
            var sorted_keys = List[Int](capacity=len(m_order))
            var sorted_ra = List[Int](capacity=len(m_order))
            var sorted_rb = List[Int](capacity=len(m_order))
            var sorted_ms = List[Manifold](capacity=len(m_order))
            var prev_key = -1
            var ord_sub = 0
            for idx in m_order:
                var k_val = graph_keys[idx]
                if len(sorted_keys) > 0 and (k_val >> 2) == (prev_key >> 2):
                    ord_sub = min(3, ord_sub + 1)
                else:
                    ord_sub = 0
                prev_key = k_val
                sorted_keys.append(((k_val >> 2) << 2) | ord_sub)
                sorted_ra.append(graph_ra[idx])
                sorted_rb.append(graph_rb[idx])
                sorted_ms.append(solver_manifolds[idx].copy())
            var colours = self.graph.assign(sorted_keys, sorted_ra, sorted_rb, n_rows)
            for mi in range(len(sorted_ms)):
                sorted_ms[mi].colour = colours[mi]
            solver_manifolds = sorted_ms^

        # 7. Build JointRow table for active unbroken joints & SixDofMotorJoints
        var j_rows = List[JointRow]()
        for ji in range(len(self.joints)):
            if self.joints[ji].broken:
                continue
            var slot_a = self.find_slot_by_id(self.joints[ji].body_a) if self.joints[ji].body_a != JOINT_WORLD else -1
            var slot_b = self.find_slot_by_id(self.joints[ji].body_b)
            if slot_b < 0 or not self.bodies[slot_b].active:
                continue
            var sa_idx = row_to_solver[slot_a] if (slot_a >= 0 and row_to_solver[slot_a] >= 0) else sbodies.static_index
            var sb_idx = row_to_solver[slot_b] if row_to_solver[slot_b] >= 0 else sbodies.static_index
            if sa_idx == sbodies.static_index and sb_idx == sbodies.static_index:
                continue
            j_rows.append(JointRow(ji, sa_idx, sb_idx, self.graph_soft_placeholder(), self.graph_soft_placeholder()))

        var s6_indices = List[Int]()
        var s6_sa = List[Int]()
        var s6_sb = List[Int]()
        var broken_s6 = List[Int]()
        for s6i in range(len(self.six_dof_joints)):
            if self.six_dof_joints[s6i].broken:
                continue
            var slot_a = self.find_slot_by_id(self.six_dof_joints[s6i].body_a) if self.six_dof_joints[s6i].body_a != JOINT_WORLD else -1
            var slot_b = self.find_slot_by_id(self.six_dof_joints[s6i].body_b)
            if slot_b < 0 or not self.bodies[slot_b].active:
                continue
            var sa_idx = row_to_solver[slot_a] if (slot_a >= 0 and row_to_solver[slot_a] >= 0) else sbodies.static_index
            var sb_idx = row_to_solver[slot_b] if row_to_solver[slot_b] >= 0 else sbodies.static_index
            if sa_idx != sbodies.static_index or sb_idx != sbodies.static_index:
                s6_indices.append(s6i)
                s6_sa.append(sa_idx)
                s6_sb.append(sb_idx)

        # 8. Solve soft contacts, joints & SixDofMotorJoints inside substeps (`h = dt / substeps`)
        var solve_res: SolveResult
        if len(s6_indices) == 0:
            solve_res = solve(
                sbodies,
                solver_manifolds,
                self.joints,
                j_rows,
                self.settings,
                dt,
                pool_address,
                self.cpus,
            )
        else:
            solve_res = _solve_with_six_dof(
                sbodies,
                solver_manifolds,
                self.joints,
                j_rows,
                self.six_dof_joints,
                s6_indices,
                s6_sa,
                s6_sb,
                broken_s6,
                self.settings,
                dt,
                pool_address,
                self.cpus,
            )

        # 9. Write back integrated poses & velocities to bodies
        for si in range(len(awake_slots)):
            var r = awake_slots[si]
            self.bodies[r].pos = sbodies.center(si)
            self.bodies[r].rot = sbodies.orientation(si)
            if not self.bodies[r].is_static:
                self.bodies[r].linear_vel = sbodies.velocity(si)
                self.bodies[r].angular_vel = sbodies.angular(si)
            self.bodies[r].force_accum = Vec3()
            self.bodies[r].torque_accum = Vec3()

        # Restore unscaled post-impact velocity for CCD-clamped bodies (preserving any solver delta)
        for ci in range(len(ccd_slots)):
            var cs = ccd_slots[ci]
            var vp = ccd_vpost[ci]
            var rf = ccd_rem[ci]
            if rf < Float32(0.999) and self.bodies[cs].active and not self.bodies[cs].is_static:
                var dv_solver = self.bodies[cs].linear_vel - vp * rf
                self.bodies[cs].linear_vel = vp + dv_solver

        # 10. Persist solved impulses to ContactCache and record PairTouch for events
        var manifold_touched = List[Bool](length=len(solver_manifolds), fill=False)
        for mi in range(len(solver_manifolds)):
            ref sm = solver_manifolds[mi]
            var cache_a = min(self.bodies[sm.a].id, sm.other) if sm.b_moves else self.bodies[sm.a].id
            var cache_b = max(self.bodies[sm.a].id, sm.other) if sm.b_moves else sm.other
            var max_spd = Float32(0.0)
            var sum_imp = Float32(0.0)
            var has_pen = False
            var best_pt = sm.points[0].point if len(sm.points) > 0 else self.bodies[sm.a].pos
            for pi in range(len(sm.points)):
                ref scp = sm.points[pi]
                self.cache.store(cache_a, cache_b, scp.feature, scp.normal_impulse, scp.tangent_impulse)
                sum_imp += scp.total_impulse
                if scp.depth >= Float32(0.0):
                    has_pen = True
                if scp.speed >= max_spd:
                    max_spd = scp.speed
                    best_pt = scp.point
            if has_pen or sum_imp > Float32(0.0):
                manifold_touched[mi] = True
                self.cache.touch(cache_a, cache_b, max_spd, sum_imp, best_pt, sm.normal)

        var sleeping_ids = List[Int]()
        for i in range(n_rows):
            if self.bodies[i].active and not self.bodies[i].is_static and not self.bodies[i].awake:
                sleeping_ids.append(self.bodies[i].id)
        self.cache.keep_resting(sleeping_ids)
        self.cache.end_tick(self.events)

        # 11. Emit joint break events and collect island links (dynamic-dynamic only!)
        var body_ids = List[Int](capacity=n_rows)
        for i in range(n_rows):
            body_ids.append(self.bodies[i].id if self.bodies[i].active and not self.bodies[i].is_static else NO_ISLAND)

        var fractured_labels = List[Int]()
        for bi in range(len(solve_res.broken_joints)):
            var ji = solve_res.broken_joints[bi]
            var j = self.joints[ji]
            var slot_b = self.find_slot_by_id(j.body_b)
            var break_pt = Vec3()
            if slot_b >= 0:
                break_pt = self.bodies[slot_b].pos + rotate_vec(self.bodies[slot_b].rot, j.local_b)
                self.wake(slot_b)
                if slot_b < len(self.islands.label) and self.islands.label[slot_b] != NO_ISLAND:
                    fractured_labels.append(self.islands.label[slot_b])
            var slot_a = self.find_slot_by_id(j.body_a) if j.body_a != JOINT_WORLD else -1
            if slot_a >= 0:
                self.wake(slot_a)
            self.events.append(PhysicsEvent(EVENT_JOINT_BREAK, j.id, j.body_b, break_pt))

        for bi in range(len(broken_s6)):
            var s6i = broken_s6[bi]
            var j6 = self.six_dof_joints[s6i]
            var slot_b = self.find_slot_by_id(j6.body_b)
            var break_pt = Vec3()
            if slot_b >= 0:
                break_pt = self.bodies[slot_b].pos + rotate_vec(self.bodies[slot_b].rot, j6.local_anchor_b)
                self.wake(slot_b)
                if slot_b < len(self.islands.label) and self.islands.label[slot_b] != NO_ISLAND:
                    fractured_labels.append(self.islands.label[slot_b])
            var slot_a = self.find_slot_by_id(j6.body_a) if j6.body_a != JOINT_WORLD else -1
            if slot_a >= 0:
                self.wake(slot_a)
            self.events.append(PhysicsEvent(EVENT_JOINT_BREAK, j6.id, j6.body_b, break_pt))

        var link_a = List[Int]()
        var link_b = List[Int]()
        var link_ra = List[Int]()
        var link_rb = List[Int]()
        for mi in range(len(solver_manifolds)):
            if not manifold_touched[mi]:
                continue
            ref sm = solver_manifolds[mi]
            if sm.b_moves and sm.b >= 0 and not self.bodies[sm.a].is_static and not self.bodies[sm.b].is_static:
                var id0 = self.bodies[sm.a].id
                var id1 = self.bodies[sm.b].id
                if id0 < id1:
                    link_a.append(id0); link_b.append(id1); link_ra.append(sm.a); link_rb.append(sm.b)
                else:
                    link_a.append(id1); link_b.append(id0); link_ra.append(sm.b); link_rb.append(sm.a)

        for ji in range(len(self.joints)):
            if self.joints[ji].broken or self.joints[ji].body_a == JOINT_WORLD:
                continue
            var sa = self.find_slot_by_id(self.joints[ji].body_a)
            var sb = self.find_slot_by_id(self.joints[ji].body_b)
            if sa >= 0 and sb >= 0 and not self.bodies[sa].is_static and not self.bodies[sb].is_static:
                var id0 = self.bodies[sa].id
                var id1 = self.bodies[sb].id
                if id0 < id1:
                    link_a.append(id0); link_b.append(id1); link_ra.append(sa); link_rb.append(sb)
                else:
                    link_a.append(id1); link_b.append(id0); link_ra.append(sb); link_rb.append(sa)

        for s6i in range(len(self.six_dof_joints)):
            if self.six_dof_joints[s6i].broken or self.six_dof_joints[s6i].body_a == JOINT_WORLD:
                continue
            var sa = self.find_slot_by_id(self.six_dof_joints[s6i].body_a)
            var sb = self.find_slot_by_id(self.six_dof_joints[s6i].body_b)
            if sa >= 0 and sb >= 0 and not self.bodies[sa].is_static and not self.bodies[sb].is_static:
                var id0 = self.bodies[sa].id
                var id1 = self.bodies[sb].id
                if id0 < id1:
                    link_a.append(id0); link_b.append(id1); link_ra.append(sa); link_rb.append(sb)
                else:
                    link_a.append(id1); link_b.append(id0); link_ra.append(sb); link_rb.append(sa)

        for k in range(len(self.islands.link_a)):
            var sa = self.find_slot_by_id(self.islands.link_a[k])
            var sb = self.find_slot_by_id(self.islands.link_b[k])
            if sa >= 0 and sb >= 0 and not self.bodies[sa].is_static and not self.bodies[sb].is_static and not self.bodies[sa].awake and not self.bodies[sb].awake:
                link_a.append(self.islands.link_a[k])
                link_b.append(self.islands.link_b[k])
                link_ra.append(sa)
                link_rb.append(sb)

        self.islands.update_links(link_a, link_b, link_ra, link_rb, body_ids)

        for fi in range(len(fractured_labels)):
            var fl = fractured_labels[fi]
            if self.islands.is_pending(fl):
                self.islands.split(fl, body_ids)

        var splits_done = 0
        while splits_done < ISLAND_SPLITS_PER_TICK and len(self.islands.pending) > 0:
            var lbl = self.islands.pending[0]
            self.islands.split(lbl, body_ids)
            splits_done += 1

        # 12. O(N) Island sleep evaluation via per-island open-addressed hash accumulators
        if self.enable_sleeping and n_rows > 0:
            var hcap = 16
            var htarget = (n_rows + len(self.islands.pending)) * 2 + 8
            while hcap < htarget:
                hcap *= 2
            var hmask = hcap - 1
            var isl_keys = List[Int](length=hcap, fill=NO_ISLAND)
            var isl_has_awake = List[Bool](length=hcap, fill=False)
            var isl_ready = List[Bool](length=hcap, fill=True)

            # Pending islands cannot sleep this tick
            for pi in range(len(self.islands.pending)):
                var plbl = self.islands.pending[pi]
                if plbl == NO_ISLAND:
                    continue
                var s = Int(mix64(UInt64(plbl))) & hmask
                while isl_keys[s] != NO_ISLAND and isl_keys[s] != plbl:
                    s = (s + 1) & hmask
                isl_keys[s] = plbl
                isl_ready[s] = False

            var lin_thresh2 = SLEEP_LIN_SPEED * SLEEP_LIN_SPEED
            var ang_thresh2 = SLEEP_ANG_SPEED * SLEEP_ANG_SPEED
            for i in range(n_rows):
                if not self.bodies[i].active or self.bodies[i].is_static:
                    continue
                if self.bodies[i].awake:
                    if not self.bodies[i].can_sleep:
                        self.bodies[i].sleep_ticks = 0
                    else:
                        var v2 = self.bodies[i].linear_vel.length_squared()
                        var w2 = self.bodies[i].angular_vel.length_squared()
                        if v2 <= lin_thresh2 and w2 <= ang_thresh2:
                            self.bodies[i].sleep_ticks += 1
                        else:
                            self.bodies[i].sleep_ticks = 0
                var lbl = self.islands.label[i]
                if lbl == NO_ISLAND:
                    continue
                var s = Int(mix64(UInt64(lbl))) & hmask
                while isl_keys[s] != NO_ISLAND and isl_keys[s] != lbl:
                    s = (s + 1) & hmask
                isl_keys[s] = lbl
                if self.bodies[i].awake:
                    isl_has_awake[s] = True
                if not self.bodies[i].can_sleep or self.bodies[i].sleep_ticks < self.sleep_threshold_ticks:
                    isl_ready[s] = False

            for i in range(n_rows):
                if not self.bodies[i].active or self.bodies[i].is_static:
                    continue
                var lbl = self.islands.label[i]
                if lbl == NO_ISLAND:
                    continue
                var s = Int(mix64(UInt64(lbl))) & hmask
                while isl_keys[s] != NO_ISLAND and isl_keys[s] != lbl:
                    s = (s + 1) & hmask
                if isl_keys[s] == lbl and isl_has_awake[s] and isl_ready[s]:
                    self.bodies[i].awake = False
                    self.bodies[i].linear_vel = Vec3()
                    self.bodies[i].angular_vel = Vec3()
                    self.events.append(PhysicsEvent(EVENT_SLEEP, self.bodies[i].id, 0))

        # 13. Refresh persisted `broadphase_nodes` to post-integration tight bounds for scene queries
        #     O(N) bottom-up refit reuses the BVH8 built in step 3 instead of rebuilding a second SAH BVH8.
        if bp_res.cells > 0:
            self._refit_broadphase()
        elif killed_any or len(awake_slots) > 0 or len(self.broadphase_nodes) == 0:
            self.rebuild_broadphase()
        sort_events(self.events)

    # =========================================================================
    # Bit-Exact Rollback & Play-In-Editor Snapshot Serialization
    # =========================================================================

    def serialize(self, mut w: BlobWriter):
        """Writes 100% of dynamic `PhysicsWorld` state (`bodies`, `sleep_ticks`, `joints`,
        `six_dof_joints`, `cache`, `graph`, `islands`, `sensors`, `triggers`, `force_world`,
        `settings`, `hulls`, `compounds`) as an IEEE-754 bit-exact versioned `'PHYW'` blob."""
        w.begin(WORLD_BLOB_TAG, WORLD_BLOB_VERSION)

        # 1. World scalars & SolverSettings
        w.f32(self.gravity)
        w.boolean(self.water_enabled)
        w.f32(self.water_level)
        w.boolean(self.water_use_waves)
        w.f32(self.fluid_density)
        w.boolean(self.enable_sleeping)
        w.integer(self.sleep_threshold_ticks)
        w.f32(self.time_s)
        w.integer(self.tick)
        w.integer(self.next_body_id)
        w.integer(self.next_joint_id)
        w.integer(self.next_trigger_id)

        w.integer(self.settings.substeps)
        w.f32(self.settings.contact_hertz)
        w.f32(self.settings.damping_ratio)
        w.f32(self.settings.push_max_speed)
        w.f32(self.settings.restitution_threshold)
        w.f32(self.settings.max_speed)
        w.boolean(self.settings.warm_starting)
        w.boolean(self.settings.relax)
        w.f32(self.settings.rolling_resistance)
        w.boolean(self.settings.force_serial)

        # 2. ConvexHulls & CompoundShapes
        w.integer(len(self.hulls))
        for hi in range(len(self.hulls)):
            ref h = self.hulls[hi]
            w.integer(len(h.vertices))
            for vi in range(len(h.vertices)):
                w.vec3(h.vertices[vi])
            w.integer(len(h.indices))
            for ii in range(len(h.indices)):
                w.integer(Int(h.indices[ii]))

        w.integer(len(self.compounds))
        for ci in range(len(self.compounds)):
            self.compounds[ci].serialize(w)

        # 3. RigidBodies (100% of fields)
        w.integer(len(self.bodies))
        for bi in range(len(self.bodies)):
            ref b = self.bodies[bi]
            w.integer(b.id)
            w.integer(b.shape)
            w.integer(b.shape_index)
            w.u8(b.motion_mode)
            w.f32(b.radius)
            w.f32(b.half_height)
            w.vec3(b.half_extents)
            w.vec3(b.pos)
            w.quat(b.rot)
            w.vec3(b.linear_vel)
            w.vec3(b.angular_vel)
            w.f32(b.mass)
            w.f32(b.inv_mass)
            w.vec3(b.local_inertia)
            w.f32(b.friction)
            w.f32(b.restitution)
            w.f32(b.rolling_resistance)
            w.f32(b.linear_damping)
            w.f32(b.angular_damping)
            w.f32(b.gravity_scale)
            w.u32(b.filter.category)
            w.u32(b.filter.mask)
            w.integer(b.filter.group)
            w.boolean(b.is_static)
            w.boolean(b.active)
            w.boolean(b.awake)
            w.boolean(b.can_sleep)
            w.integer(b.sleep_ticks)
            w.boolean(b.in_water)
            w.integer(b.water_volume_id)
            w.integer(b.material)
            w.boolean(b.enable_ccd)
            w.f32(b.wind_area_over_mass)
            w.vec3(b.force_accum)
            w.vec3(b.torque_accum)

        # 4. Joints & SixDofMotorJoints
        w.integer(len(self.joints))
        for ji in range(len(self.joints)):
            self.joints[ji].serialize(w)

        w.integer(len(self.six_dof_joints))
        for s6i in range(len(self.six_dof_joints)):
            self.six_dof_joints[s6i].serialize(w)

        # 5. Sensors & TriggerVolumes + persisted overlap state
        w.integer(len(self.sensors))
        for si in range(len(self.sensors)):
            ref s = self.sensors[si]
            w.integer(s.id)
            w.vec3(s.center)
            w.vec3(s.half)
            w.f32(s.radius)
            w.u32(s.filter.category)
            w.u32(s.filter.mask)
            w.integer(s.filter.group)
        w.integer(len(self.sensor_inside_a))
        for k in range(len(self.sensor_inside_a)):
            w.integer(self.sensor_inside_a[k])
            w.integer(self.sensor_inside_b[k])

        w.integer(len(self.triggers))
        for ti in range(len(self.triggers)):
            ref tv = self.triggers[ti]
            w.integer(tv.id)
            w.integer(tv.shape)
            w.vec3(tv.pos)
            w.quat(tv.rot)
            w.vec3(tv.half_extents)
            w.f32(tv.radius)
            w.f32(tv.half_height)
            w.u32(tv.filter.category)
            w.u32(tv.filter.mask)
            w.integer(tv.filter.group)
            w.boolean(tv.active)
        w.integer(len(self.trigger_inside_a))
        for k in range(len(self.trigger_inside_a)):
            w.integer(self.trigger_inside_a[k])
            w.integer(self.trigger_inside_b[k])

        # 6. ContactCache, IslandSet, ConstraintGraph & ForceWorld
        self.cache.serialize(w)
        self.islands.serialize(w)
        w.integer(self.islands.splits)
        w.integer(self.islands.merges)
        self.graph.serialize(w)
        w.integer(self.graph.overflow)
        w.integer(self.graph.used)
        self.force_world.write(w)

        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        """Restores 100% of dynamic `PhysicsWorld` state from a `'PHYW'` snapshot blob."""
        _ = r.enter(WORLD_BLOB_TAG, WORLD_BLOB_VERSION)

        self.gravity = r.f32()
        self.water_enabled = r.boolean()
        self.water_level = r.f32()
        self.water_use_waves = r.boolean()
        self.fluid_density = r.f32()
        self.enable_sleeping = r.boolean()
        self.sleep_threshold_ticks = r.integer()
        self.time_s = r.f32()
        self.tick = r.integer()
        self.next_body_id = r.integer()
        self.next_joint_id = r.integer()
        self.next_trigger_id = r.integer()

        self.settings.substeps = r.integer()
        self.settings.contact_hertz = r.f32()
        self.settings.damping_ratio = r.f32()
        self.settings.push_max_speed = r.f32()
        self.settings.restitution_threshold = r.f32()
        self.settings.max_speed = r.f32()
        self.settings.warm_starting = r.boolean()
        self.settings.relax = r.boolean()
        self.settings.rolling_resistance = r.f32()
        self.settings.force_serial = r.boolean()

        var n_hulls = r.integer()
        if n_hulls < 0 or n_hulls > 1 << 16:
            raise Error("physics world blob: invalid hull count")
        self.hulls.clear()
        for _ in range(n_hulls):
            var nv = r.integer()
            var verts = List[Vec3](capacity=max(0, nv))
            for _ in range(nv):
                verts.append(r.vec3())
            var ni = r.integer()
            var idxs = List[Int32](capacity=max(0, ni))
            for _ in range(ni):
                idxs.append(Int32(r.integer()))
            self.hulls.append(ConvexHull(verts, idxs))

        var n_cmp = r.integer()
        if n_cmp < 0 or n_cmp > 1 << 16:
            raise Error("physics world blob: invalid compound count")
        self.compounds.clear()
        for _ in range(n_cmp):
            self.compounds.append(CompoundShape.deserialize(r))

        var n_bodies = r.integer()
        if n_bodies < 0 or n_bodies > 1 << 20:
            raise Error("physics world blob: invalid body count")
        self.bodies.clear()
        self.bodies.reserve(n_bodies)
        for _ in range(n_bodies):
            var b = RigidBody()
            b.id = r.integer()
            b.shape = r.integer()
            b.shape_index = r.integer()
            b.motion_mode = r.u8()
            b.radius = r.f32()
            b.half_height = r.f32()
            b.half_extents = r.vec3()
            b.pos = r.vec3()
            b.rot = r.quat()
            b.linear_vel = r.vec3()
            b.angular_vel = r.vec3()
            b.mass = r.f32()
            b.inv_mass = r.f32()
            b.local_inertia = r.vec3()
            b.friction = r.f32()
            b.restitution = r.f32()
            b.rolling_resistance = r.f32()
            b.linear_damping = r.f32()
            b.angular_damping = r.f32()
            b.gravity_scale = r.f32()
            var f_cat = r.u32()
            var f_mask = r.u32()
            var f_grp = r.integer()
            b.filter = CollisionFilter(f_cat, f_mask, f_grp)
            b.is_static = r.boolean()
            b.active = r.boolean()
            b.awake = r.boolean()
            b.can_sleep = r.boolean()
            b.sleep_ticks = r.integer()
            b.in_water = r.boolean()
            b.water_volume_id = r.integer()
            b.material = r.integer()
            b.enable_ccd = r.boolean()
            b.wind_area_over_mass = r.f32()
            b.force_accum = r.vec3()
            b.torque_accum = r.vec3()
            self.bodies.append(b^)

        var n_joints = r.integer()
        if n_joints < 0 or n_joints > 1 << 20:
            raise Error("physics world blob: invalid joint count")
        self.joints.clear()
        for _ in range(n_joints):
            self.joints.append(Joint.deserialize(r))

        var n_s6 = r.integer()
        if n_s6 < 0 or n_s6 > 1 << 16:
            raise Error("physics world blob: invalid six-dof joint count")
        self.six_dof_joints.clear()
        for _ in range(n_s6):
            self.six_dof_joints.append(SixDofMotorJoint.deserialize(r))

        var n_sens = r.integer()
        if n_sens < 0 or n_sens > 1 << 16:
            raise Error("physics world blob: invalid sensor count")
        self.sensors.clear()
        for _ in range(n_sens):
            var s_id = r.integer()
            var s_c = r.vec3()
            var s_h = r.vec3()
            var s_rad = r.f32()
            var s_cat = r.u32()
            var s_mask = r.u32()
            var s_grp = r.integer()
            self.sensors.append(Sensor(s_id, s_c, s_h, s_rad, CollisionFilter(s_cat, s_mask, s_grp)))

        var n_sin = r.integer()
        self.sensor_inside_a.clear()
        self.sensor_inside_b.clear()
        for _ in range(n_sin):
            self.sensor_inside_a.append(r.integer())
            self.sensor_inside_b.append(r.integer())

        var n_trig = r.integer()
        if n_trig < 0 or n_trig > 1 << 16:
            raise Error("physics world blob: invalid trigger count")
        self.triggers.clear()
        for _ in range(n_trig):
            var tv_id = r.integer()
            var tv_sh = r.integer()
            var tv_p = r.vec3()
            var tv_q = r.quat()
            var tv_he = r.vec3()
            var tv_rad = r.f32()
            var tv_hh = r.f32()
            var tv_cat = r.u32()
            var tv_mask = r.u32()
            var tv_grp = r.integer()
            var tv_act = r.boolean()
            self.triggers.append(
                TriggerVolume(tv_id, tv_sh, tv_p, tv_q, tv_he, tv_rad, tv_hh, CollisionFilter(tv_cat, tv_mask, tv_grp), tv_act)
            )

        var n_tin = r.integer()
        self.trigger_inside_a.clear()
        self.trigger_inside_b.clear()
        for _ in range(n_tin):
            self.trigger_inside_a.append(r.integer())
            self.trigger_inside_b.append(r.integer())
        self.trigger_events.clear()
        self.events.clear()

        self.cache.deserialize(r)
        self.islands.deserialize(r)
        self.islands.splits = r.integer()
        self.islands.merges = r.integer()
        self.graph.deserialize(r)
        self.graph.overflow = r.integer()
        self.graph.used = r.integer()
        self.force_world.read(r)
        self.force_world.tick = self.tick

        r.leave()

        self._rebuild_id_to_slot()
        self.rebuild_broadphase()

    @staticmethod
    @always_inline
    def graph_soft_placeholder() -> Softness:
        return Softness()
