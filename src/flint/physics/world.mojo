"""Unified Rigid-Body World Coordinator (`PhysicsWorld`, v4).

Coordinates dynamic and static rigid bodies (`Sphere`, `Capsule`, `Box`, `ConvexHull`,
and static `Bvh8MeshCollider`), 8-wide SIMD `Broadphase`, discrete `Narrowphase` (explicitly
negating narrowphase `A -> B` normals into solver `B -> A` normals), `ContactCache` warm
starting, `IslandSet` union-find & fracture island splitting, `ConstraintGraph` persistent
24-colouring, `Joint`s (including breakable `JOINT_WELD` assemblies), Beaufort wave buoyancy
& `ForceWorld` wind/force volumes, optional swept-sphere CCD, and deterministic `PhysicsEvent`s.
"""
from std.collections import List
from std.math import sqrt, abs, min, max
from flint.core.math import Vec3, Quat
from flint.core.detmath import set_sim_float_mode
from flint.core.detsort import stable_order
from flint.physics.filter import CollisionFilter, CAT_TERRAIN, CAT_STATIC, CAT_ALL
from flint.physics.broadphase import BroadphaseInput, find_pairs
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
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.ccd import gjk_closest_points
from flint.physics.solver_bodies import SolverBodies, rotate_vec
from flint.physics.solver import (
    SolverSettings,
    Manifold,
    ContactPoint as SolverContactPoint,
    SPECULATIVE_DISTANCE,
    solve,
    ccd_cpus,
)
from flint.physics.joints import (
    Joint,
    JointDef,
    JointRow,
    Softness,
    JOINT_WELD,
    JOINT_DISTANCE,
    JOINT_WORLD,
    quat_conj,
    quat_product,
)
from flint.physics.islands import (
    IslandSet,
    ConstraintGraph,
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

comptime SLEEP_TICKS_DEFAULT: Int = 60
comptime SLEEP_LIN_SPEED: Float32 = 0.05
comptime SLEEP_ANG_SPEED: Float32 = 0.08


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


struct RigidBody(Copyable, Movable):
    var id: Int
    var shape: Int
    var shape_index: Int
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

    def capsule_endpoints(self) -> Tuple[Vec3, Vec3]:
        var axis = self.rot.rotate(Vec3(0, self.half_height, 0))
        return (self.pos - axis, self.pos + axis)

    def tight_aabb(self, hulls: List[ConvexHull], meshes: List[Bvh8MeshCollider]) -> Tuple[Vec3, Vec3]:
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


struct PhysicsWorld(Movable):
    var bodies: List[RigidBody]
    var hulls: List[ConvexHull]
    var meshes: List[Bvh8MeshCollider]
    var joints: List[Joint]
    var sensors: List[Sensor]
    var sensor_inside_a: List[Int]  # sorted (sensor_id, body_id) pairs inside last tick
    var sensor_inside_b: List[Int]
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
    var cpus: List[Bool]

    def __init__(out self, gravity: Float32 = 9.81):
        self.bodies = List[RigidBody]()
        self.hulls = List[ConvexHull]()
        self.meshes = List[Bvh8MeshCollider]()
        self.joints = List[Joint]()
        self.sensors = List[Sensor]()
        self.sensor_inside_a = List[Int]()
        self.sensor_inside_b = List[Int]()
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
        self.cpus = ccd_cpus()

    def _register_body(mut self, var body: RigidBody) -> Int:
        var slot = len(self.bodies)
        var bid = body.id
        var is_dyn = not body.is_static
        self.bodies.append(body^)
        self.islands.ensure_rows(slot + 1)
        if is_dyn:
            self.islands.set_single(slot, bid)
        else:
            self.islands.drop(slot)
        return slot

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
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
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
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
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
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
            b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, mass, 0.0, 0.0, b.half_extents)
            var area = Float32(4.0) * (
                b.half_extents.x * b.half_extents.y
                + b.half_extents.y * b.half_extents.z
                + b.half_extents.z * b.half_extents.x
            ) * Float32(1.0 / 3.0)
            b.wind_area_over_mass = area / mass
        return self._register_body(b^)

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
            b.awake = False
            if filter == CollisionFilter():
                b.filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        else:
            b.mass = mass
            b.inv_mass = Float32(1.0) / mass
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
        b.awake = False
        return self._register_body(b^)

    def add_sensor(mut self, sensor: Sensor):
        self.sensors.append(sensor)

    def set_water(mut self, level: Float32, use_waves: Bool = True, fluid_density: Float32 = RHO_SEAWATER_SI):
        self.water_enabled = level > NO_WATER + Float32(1.0)
        self.water_level = level
        self.water_use_waves = use_waves
        self.fluid_density = fluid_density

    def body_id(self, slot: Int) -> Int:
        if slot < 0 or slot >= len(self.bodies):
            return -1
        return self.bodies[slot].id

    def find_slot_by_id(self, id: Int) -> Int:
        for i in range(len(self.bodies)):
            if self.bodies[i].active and self.bodies[i].id == id:
                return i
        return -1

    def wake(mut self, slot: Int):
        if slot < 0 or slot >= len(self.bodies):
            return
        if self.bodies[slot].is_static or not self.bodies[slot].active:
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
        if slot < 0 or slot >= len(self.bodies) or self.bodies[slot].is_static:
            return
        self.bodies[slot].linear_vel = linear
        self.bodies[slot].angular_vel = angular
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

        var world_a = slot_a < 0 or slot_a >= len(self.bodies) or self.bodies[slot_a].is_static
        var qa = Quat()
        if not world_a:
            j.body_a = self.bodies[slot_a].id
            var pa = self.bodies[slot_a].pos
            qa = self.bodies[slot_a].rot
            j.local_a = qa.conjugate().rotate(jdef.anchor - pa)
            j.axis_a = qa.conjugate().rotate(ax_w)
            self.wake(slot_a)
        else:
            j.body_a = JOINT_WORLD
            if slot_a >= 0 and slot_a < len(self.bodies):
                # Static body anchor resolved in world coordinates
                j.local_a = jdef.anchor
            else:
                j.local_a = jdef.anchor
            j.axis_a = ax_w

        var pb = self.bodies[slot_b].pos
        var qb = self.bodies[slot_b].rot
        var anchor_b = jdef.anchor_b if jdef.has_anchor_b else jdef.anchor
        j.body_b = self.bodies[slot_b].id
        j.local_b = qb.conjugate().rotate(anchor_b - pb)
        j.axis_b = qb.conjugate().rotate(ax_w)
        j.reference = qb if world_a else quat_product(quat_conj(qa), qb)
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
        b.awake = active and (not b.is_static)
        b.sleep_ticks = 0
        b.in_water = False
        b.water_volume_id = -1
        self.islands.ensure_rows(len(self.bodies))
        if active and (not b.is_static):
            self.islands.set_single(slot, b.id)
        else:
            self.islands.drop(slot)

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
        var pt_verts = List[Vec3](capacity=1)
        pt_verts.append(Vec3(0, 0, 0))
        var pt_hull = ConvexHull(pt_verts)
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
        # Sphere center is inside hull: fall back to polyhedral EPA for deep push-out
        var sph_hull = _make_capsule_hull(s_rad, Float32(0.0))
        return collide_convex_convex(sph_hull, s_pos, Quat(), hull, h_pos, h_rot, row_a, row_b)

    def _collide_dynamic_vs_mesh(
        self,
        dyn_slot: Int,
        mesh_slot: Int,
    ) -> ContactManifold:
        """Collides dynamic body `dyn_slot` against static BVH8 mesh `mesh_slot`.
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
                tverts.append(tri.v0)
                tverts.append(tri.v1)
                tverts.append(tri.v2)
                var thull = ConvexHull(tverts)
                var tm = collide_convex_convex(hull, l_pos, l_rot, thull, Vec3(), Quat())
                if tm.has_collision:
                    for pi in range(tm.count):
                        var cp = tm.get_point(pi)
                        # tm normal points Hull -> Tri; negate to point Tri (Mesh) -> Hull (Dynamic)
                        var n_out = -cp.normal
                        if (tri.active_edges == 0) and abs(n_out.dot(tri.normal)) > Float32(0.5):
                            n_out = tri.normal if n_out.dot(tri.normal) >= Float32(0.0) else -tri.normal
                        cands.append(NarrowContactPoint(cp.point, n_out, cp.depth, UInt32(ti * 8 + pi)))
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

    def _collide_pair(self, row_a: Int, row_b: Int) -> ContactManifold:
        """Discrete narrowphase between `row_a` and `row_b`.
        Invariant: returned `normal` ALWAYS points from `row_a` toward `row_b` (`A -> B`)."""
        ref a = self.bodies[row_a]
        ref b = self.bodies[row_b]
        var sa = a.shape
        var sb = b.shape

        if sa == BODY_SHAPE_MESH_BVH8 and sb == BODY_SHAPE_MESH_BVH8:
            return ContactManifold(row_a, row_b)
        if sb == BODY_SHAPE_MESH_BVH8:
            # _collide_dynamic_vs_mesh returns Mesh (B) -> Dynamic (A); negate to A -> B!
            var m = self._collide_dynamic_vs_mesh(row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^
        if sa == BODY_SHAPE_MESH_BVH8:
            # _collide_dynamic_vs_mesh returns Mesh (A) -> Dynamic (B), which is ALREADY A -> B!
            var m = self._collide_dynamic_vs_mesh(row_b, row_a)
            m.body_a = row_a
            m.body_b = row_b
            return m^

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_SPHERE:
            return collide_sphere_sphere(a.pos, a.radius, b.pos, b.radius, row_a, row_b)

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_CAPSULE:
            var eb = b.capsule_endpoints()
            return collide_sphere_capsule(a.pos, a.radius, eb[0], eb[1], b.radius, row_a, row_b)
        if sa == BODY_SHAPE_CAPSULE and sb == BODY_SHAPE_SPHERE:
            var ea = a.capsule_endpoints()
            var m = collide_sphere_capsule(b.pos, b.radius, ea[0], ea[1], a.radius, row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^

        if sa == BODY_SHAPE_SPHERE and sb == BODY_SHAPE_BOX:
            return collide_sphere_box(a.pos, a.radius, b.pos, b.rot, b.half_extents, row_a, row_b)
        if sa == BODY_SHAPE_BOX and sb == BODY_SHAPE_SPHERE:
            var m = collide_sphere_box(b.pos, b.radius, a.pos, a.rot, a.half_extents, row_a, row_b)
            if m.has_collision:
                _negate_manifold_normals(m)
            return m^

        if sa == BODY_SHAPE_CAPSULE and sb == BODY_SHAPE_CAPSULE:
            var ea = a.capsule_endpoints()
            var eb = b.capsule_endpoints()
            return collide_capsule_capsule(ea[0], ea[1], a.radius, eb[0], eb[1], b.radius, row_a, row_b)

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

        # ConvexHull vs Sphere / Capsule / Box / ConvexHull
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

    def _run_ccd_clamp(mut self, dt: Float32):
        """Swept-sphere anti-tunneling clamp for bodies with `enable_ccd = True` against static BVH8 meshes."""
        for i in range(len(self.bodies)):
            if not self.bodies[i].active or self.bodies[i].is_static or not self.bodies[i].awake or not self.bodies[i].enable_ccd:
                continue
            var step_vec = self.bodies[i].linear_vel * dt
            var r_core = self.bodies[i].radius if self.bodies[i].shape == BODY_SHAPE_SPHERE or self.bodies[i].shape == BODY_SHAPE_CAPSULE else min(
                self.bodies[i].half_extents.x, min(self.bodies[i].half_extents.y, self.bodies[i].half_extents.z)
            )
            if step_vec.length_squared() <= r_core * r_core:
                continue
            var best_toi = Float32(1.0)
            var hit_any = False
            for m_slot in range(len(self.bodies)):
                if not self.bodies[m_slot].active or self.bodies[m_slot].shape != BODY_SHAPE_MESH_BVH8:
                    continue
                if not self.bodies[i].filter.touches(self.bodies[m_slot].filter):
                    continue
                ref mb = self.bodies[m_slot]
                ref bvh = self.meshes[mb.shape_index]
                var p0 = self.bodies[i].pos
                var p1 = p0 + step_vec
                var sh = bvh.swept_sphere_moving_bvh8(p0, p1, r_core, mb.pos, mb.rot, mb.pos, mb.rot)
                if sh.hit and sh.toi < best_toi:
                    best_toi = sh.toi
                    hit_any = True
            if hit_any and best_toi < Float32(1.0):
                # Advance to safe pre-impact contact envelope so discrete narrowphase engages cleanly
                var safe_t = max(Float32(0.0), best_toi - Float32(0.02))
                self.bodies[i].pos = self.bodies[i].pos + step_vec * safe_t
                self.bodies[i].linear_vel = self.bodies[i].linear_vel * max(Float32(0.05), Float32(1.0) - safe_t)

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
                var tb = self.bodies[bi].tight_aabb(self.hulls, self.meshes)
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

    def step(mut self, dt: Float32, pool_address: Int = 0):
        """Executes one complete rigid-body simulation tick of duration `dt`."""
        set_sim_float_mode()
        self.events.clear()
        if dt <= Float32(1.0e-7):
            return

        self.time_s += dt
        self.tick += 1
        self.force_world.tick = self.tick
        self.force_world.wind.age()
        self.cache.begin_tick()

        # 1. Sensors & kill volumes
        self._update_sensors()
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
                var ev = PhysicsEvent(EVENT_KILL, self.bodies[i].id, 0, p)
                ev.material = self.bodies[i].material
                self.events.append(ev)
                self.bodies[i].active = False
                self.islands.drop(i)

        # 2. Optional anti-tunneling CCD clamp
        self._run_ccd_clamp(dt)

        # 3. Broadphase over all active bodies
        var n_rows = len(self.bodies)
        var bp_in = BroadphaseInput()
        var spec = Vec3(SPECULATIVE_DISTANCE, SPECULATIVE_DISTANCE, SPECULATIVE_DISTANCE)
        for i in range(n_rows):
            ref b = self.bodies[i]
            if not b.active:
                bp_in.add(Vec3(), Vec3(), False, False, b.filter)
                continue
            var tb = b.tight_aabb(self.hulls, self.meshes)
            var lo = tb[0] - spec
            var hi = tb[1] + spec
            if not b.is_static and b.awake:
                var move = b.linear_vel * dt
                lo = Vec3(min(lo.x, lo.x + move.x), min(lo.y, lo.y + move.y), min(lo.z, lo.z + move.z))
                hi = Vec3(max(hi.x, hi.x + move.x), max(hi.y, hi.y + move.y), max(hi.z, hi.z + move.z))
            bp_in.add(lo, hi, True, (not b.is_static) and b.awake, b.filter)

        var bp_res = find_pairs(bp_in, pool_address)

        # 4. Discrete Narrowphase & wake sleeping dynamic bodies hit by awake bodies
        var raw_manifolds = List[ContactManifold]()
        for pk in range(len(bp_res.pairs)):
            var code = bp_res.pairs[pk]
            var row_a = code // n_rows
            var row_b = code % n_rows
            if self.bodies[row_a].is_static and self.bodies[row_b].is_static:
                continue
            var nm = self._collide_pair(row_a, row_b)
            if nm.has_collision and nm.count > 0:
                if not self.bodies[row_a].is_static and not self.bodies[row_a].awake:
                    self.wake(row_a)
                if not self.bodies[row_b].is_static and not self.bodies[row_b].awake:
                    self.wake(row_b)
                raw_manifolds.append(nm)

        # Wake bodies linked to awake bodies via unbroken joints
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

        # 5. Build SolverBodies for all active, awake dynamic bodies
        var row_to_solver = List[Int](length=n_rows, fill=-1)
        var awake_slots = List[Int]()
        for i in range(n_rows):
            if self.bodies[i].active and not self.bodies[i].is_static and self.bodies[i].awake:
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

            var m_a: Int
            var m_b: Int
            var b_moves: Bool
            var other_id: Int
            var solver_normal: Vec3
            var cache_a: Int
            var cache_b: Int

            if a_dyn and b_dyn:
                m_a = ra
                m_b = rb
                b_moves = True
                other_id = self.bodies[rb].id
                # nm.normal points A -> B; solver requires B -> A!
                solver_normal = -nm.normal
                cache_a = min(self.bodies[ra].id, self.bodies[rb].id)
                cache_b = max(self.bodies[ra].id, self.bodies[rb].id)
            elif a_dyn:
                m_a = ra
                m_b = -1
                b_moves = False
                other_id = OTHER_TERRAIN if self.bodies[rb].shape == BODY_SHAPE_MESH_BVH8 else static_other(rb)
                # nm.normal points A (dyn) -> B (static); negate to point B (static) -> A (dyn)!
                solver_normal = -nm.normal
                cache_a = self.bodies[ra].id
                cache_b = other_id
            else:
                m_a = rb
                m_b = -1
                b_moves = False
                other_id = OTHER_TERRAIN if self.bodies[ra].shape == BODY_SHAPE_MESH_BVH8 else static_other(ra)
                # nm.normal points row_a (static) -> row_b (dyn m_a), which is ALREADY B (static) -> A (dyn)!
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

        # 7. Build JointRow table for active unbroken joints
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

        # 8. Solve soft contacts & joints
        var solve_res = solve(
            sbodies,
            solver_manifolds,
            self.joints,
            j_rows,
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
            self.bodies[r].linear_vel = sbodies.velocity(si)
            self.bodies[r].angular_vel = sbodies.angular(si)
            self.bodies[r].force_accum = Vec3()
            self.bodies[r].torque_accum = Vec3()

        # 10. Persist solved impulses to ContactCache and record PairTouch for events
        for mi in range(len(solver_manifolds)):
            ref sm = solver_manifolds[mi]
            var cache_a = min(self.bodies[sm.a].id, sm.other) if sm.b_moves else self.bodies[sm.a].id
            var cache_b = max(self.bodies[sm.a].id, sm.other) if sm.b_moves else sm.other
            var max_spd = Float32(0.0)
            var sum_imp = Float32(0.0)
            var best_pt = sm.points[0].point if len(sm.points) > 0 else self.bodies[sm.a].pos
            for pi in range(len(sm.points)):
                ref scp = sm.points[pi]
                self.cache.store(cache_a, cache_b, scp.feature, scp.normal_impulse, scp.tangent_impulse)
                sum_imp += scp.total_impulse
                if scp.speed >= max_spd:
                    max_spd = scp.speed
                    best_pt = scp.point
            self.cache.touch(cache_a, cache_b, max_spd, sum_imp, best_pt, sm.normal)

        var sleeping_ids = List[Int]()
        for i in range(n_rows):
            if self.bodies[i].active and not self.bodies[i].is_static and not self.bodies[i].awake:
                sleeping_ids.append(self.bodies[i].id)
        self.cache.keep_resting(sleeping_ids)
        self.cache.end_tick(self.events)

        # 11. Emit joint break events and collect island links
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

        var link_a = List[Int]()
        var link_b = List[Int]()
        var link_ra = List[Int]()
        var link_rb = List[Int]()
        for mi in range(len(solver_manifolds)):
            ref sm = solver_manifolds[mi]
            if sm.b_moves and sm.b >= 0:
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

        # Preserve links between two sleeping dynamic bodies
        for k in range(len(self.islands.link_a)):
            var sa = self.find_slot_by_id(self.islands.link_a[k])
            var sb = self.find_slot_by_id(self.islands.link_b[k])
            if sa >= 0 and sb >= 0 and not self.bodies[sa].awake and not self.bodies[sb].awake:
                link_a.append(self.islands.link_a[k])
                link_b.append(self.islands.link_b[k])
                link_ra.append(sa)
                link_rb.append(sb)

        self.islands.update_links(link_a, link_b, link_ra, link_rb, body_ids)

        # Immediately split fractured joint islands + rate-limited pending islands
        for fi in range(len(fractured_labels)):
            var fl = fractured_labels[fi]
            if self.islands.is_pending(fl):
                self.islands.split(fl, body_ids)

        var splits_done = 0
        while splits_done < ISLAND_SPLITS_PER_TICK and len(self.islands.pending) > 0:
            var lbl = self.islands.pending[0]
            self.islands.split(lbl, body_ids)
            splits_done += 1

        # 12. Island sleep evaluation
        if self.enable_sleeping:
            for i in range(n_rows):
                if not self.bodies[i].active or self.bodies[i].is_static or not self.bodies[i].awake:
                    continue
                if not self.bodies[i].can_sleep:
                    self.bodies[i].sleep_ticks = 0
                    continue
                var v2 = self.bodies[i].linear_vel.length_squared()
                var w2 = self.bodies[i].angular_vel.length_squared()
                if v2 <= SLEEP_LIN_SPEED * SLEEP_LIN_SPEED and w2 <= SLEEP_ANG_SPEED * SLEEP_ANG_SPEED:
                    self.bodies[i].sleep_ticks += 1
                else:
                    self.bodies[i].sleep_ticks = 0

            var checked = List[Int]()
            for i in range(n_rows):
                if not self.bodies[i].active or self.bodies[i].is_static or not self.bodies[i].awake:
                    continue
                var lbl = self.islands.label[i]
                if lbl == NO_ISLAND or self.islands.is_pending(lbl):
                    continue
                var seen_lbl = False
                for c_lbl in checked:
                    if c_lbl == lbl:
                        seen_lbl = True
                        break
                if seen_lbl:
                    continue
                checked.append(lbl)

                var ready_to_sleep = True
                for j in range(n_rows):
                    if self.bodies[j].active and not self.bodies[j].is_static and self.islands.label[j] == lbl:
                        if not self.bodies[j].can_sleep or self.bodies[j].sleep_ticks < self.sleep_threshold_ticks:
                            ready_to_sleep = False
                            break
                if ready_to_sleep:
                    for j in range(n_rows):
                        if self.bodies[j].active and not self.bodies[j].is_static and self.islands.label[j] == lbl:
                            self.bodies[j].awake = False
                            self.bodies[j].linear_vel = Vec3()
                            self.bodies[j].angular_vel = Vec3()
                            self.events.append(PhysicsEvent(EVENT_SLEEP, self.bodies[j].id, 0))

        sort_events(self.events)

    @staticmethod
    @always_inline
    def graph_soft_placeholder() -> Softness:
        return Softness()
