"""Flint v5 Physics Plugin & ECS Component Schemas (`flint.physics.plugin`).

Provides:
- `PhysicsEventsResource` & `TriggerEventsResource`: `Resource` wrappers around
  `Events[PhysicsEvent]` and `Events[TriggerEvent]` swapped via `EngineApp.add_event_swap_hook`.
- `PhysicsSyncStateResource`: entity <-> body_id / slot tracking and last-synced
  transform cache for despawn cleanup and teleport detection.
- 7 `SerializableComponent` ECS schemas with explicit `world.set_component_binary_hooks`
  for every non-trivial struct (zero Clone Vault dependency):
  `RigidBodyComponent`, `ColliderComponent`, `JointComponent`,
  `CharacterControllerComponent`, `VehicleComponent`, `Aero6DofComponent`,
  `ClothMeshComponent`.
- `register_physics_components(mut world: World)` & `register_physics_codecs(mut reg: ComponentRegistry)`.
- `PhysicsPlugin(gravity=9.81)`: inserts `PhysicsWorld` directly as a `Resource`,
  materializes/despawns & syncs transforms in `STAGE_PRE_UPDATE`, steps
  `CharacterController`, `VehicleChassis`, `Aero6DofState`, `ParticleMesh`, and
  `PhysicsWorld.step(dt, pool_addr)` in `STAGE_FIXED_UPDATE`.
"""
from std.collections import Dict, List
from std.math import abs, max, min, sqrt
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FixedTime,
    Resource,
    Resources,
    STAGE_FIXED_UPDATE,
    STAGE_PRE_UPDATE,
)
from flint.assets.json import JsonDocument
from flint.core.detmath import det_atan2
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat
from flint.core.serial import BlobReader, BlobWriter, float32_bits, fourcc
from flint.ecs.world import (
    Entity,
    Events,
    NULL_ENTITY,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.physics.aero6dof import Aero6DofConfig, Aero6DofControls, Aero6DofState
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.events import PhysicsEvent
from flint.physics.filter import CAT_ALL, CAT_DEBRIS, CAT_PROP_PIECE, CAT_SENSOR, CollisionFilter
from flint.physics.joints import JOINT_WELD, JOINT_WORLD, JointDef
from flint.physics.narrowphase import ConvexHull
from flint.physics.shapes_v5 import SixDofMotorJoint
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import (
    BODY_MOTION_DYNAMIC,
    BODY_MOTION_KINEMATIC,
    BODY_MOTION_STATIC,
    BODY_SHAPE_BOX,
    BODY_SHAPE_CAPSULE,
    BODY_SHAPE_CONVEX_HULL,
    BODY_SHAPE_SPHERE,
    PhysicsWorld,
    TRIGGER_SHAPE_BOX,
    TRIGGER_SHAPE_CAPSULE,
    TRIGGER_SHAPE_SPHERE,
    TriggerEvent,
    TriggerVolume,
    compute_shape_inertia,
)
from flint.physics.xpbd import ParticleMesh, RigidAttachment
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    Parent,
    SerializableComponent,
)


# ---------------------------------------------------------------------------
# JSON & Binary Helpers
# ---------------------------------------------------------------------------

@always_inline
def _f32_s(v: Float32) -> String:
    return String(Float64(v))


def _vec3_json(v: Vec3) -> String:
    return "[" + _f32_s(v.x) + ", " + _f32_s(v.y) + ", " + _f32_s(v.z) + "]"


def _quat_json(q: Quat) -> String:
    return "[" + _f32_s(q.x) + ", " + _f32_s(q.y) + ", " + _f32_s(q.z) + ", " + _f32_s(q.w) + "]"


def _filter_json(f: CollisionFilter) -> String:
    return "[" + String(Int(f.category)) + ", " + String(Int(f.mask)) + ", " + String(f.group) + "]"


def _read_vec3_json(doc: JsonDocument, node: Int, fallback: Vec3 = Vec3()) -> Vec3:
    if doc.array_size(node) < 3:
        return fallback
    return Vec3(
        Float32(doc.number(doc.at(node, 0), Float64(fallback.x))),
        Float32(doc.number(doc.at(node, 1), Float64(fallback.y))),
        Float32(doc.number(doc.at(node, 2), Float64(fallback.z))),
    )


def _read_quat_json(doc: JsonDocument, node: Int) -> Quat:
    if doc.array_size(node) < 4:
        return Quat.identity()
    return Quat(
        Float32(doc.number(doc.at(node, 0), 0.0)),
        Float32(doc.number(doc.at(node, 1), 0.0)),
        Float32(doc.number(doc.at(node, 2), 0.0)),
        Float32(doc.number(doc.at(node, 3), 1.0)),
    )


def _read_filter_json(doc: JsonDocument, node: Int, fallback: CollisionFilter = CollisionFilter()) -> CollisionFilter:
    if doc.array_size(node) < 3:
        return fallback
    return CollisionFilter(
        UInt32(doc.integer(doc.at(node, 0), Int(fallback.category))),
        UInt32(doc.integer(doc.at(node, 1), Int(fallback.mask))),
        doc.integer(doc.at(node, 2), fallback.group),
    )


@always_inline
def _write_filter(mut out: BlobWriter, f: CollisionFilter):
    out.u32(f.category)
    out.u32(f.mask)
    out.integer(f.group)


@always_inline
def _read_filter(mut reader: BlobReader) raises -> CollisionFilter:
    var c = reader.u32()
    var m = reader.u32()
    var g = reader.integer()
    return CollisionFilter(c, m, g)


@always_inline
def _quat_to_scaled_angle_phys(q: Quat) -> Vec3:
    var qn = q.normalized()
    if qn.w < Float32(0.0):
        qn = Quat(-qn.x, -qn.y, -qn.z, -qn.w)
    var v = Vec3(qn.x, qn.y, qn.z)
    var s = v.length()
    if s <= Float32(1.0e-7):
        return v * Float32(2.0)
    var ang = Float32(2.0) * det_atan2(s, max(Float32(0.0), qn.w))
    return v * (ang / s)


def _phys_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _phys_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
        var val = C.read_bin(reader, ver)
        reader.leave()
        var dst = Pointer[C, MutAnyOrigin](unsafe_from_address=comp_addr)
        __get_address_as_uninit_lvalue(dst._mlir_value) = val^
        return True
    except:
        return False


def _register_non_trivial_phys[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
    if not is_triv and tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _phys_bin_write[C], _phys_bin_read[C]
        )


def _register_phys_bin_hooks[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    if tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _phys_bin_write[C], _phys_bin_read[C]
        )


# ---------------------------------------------------------------------------
# Resources & Event Buffers
# ---------------------------------------------------------------------------

struct PhysicsEventsResource(Resource, Movable):
    """Per-frame `Events[PhysicsEvent]` drained from `PhysicsWorld.step`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.physics.PhysicsEventsResource")
    var events: Events[PhysicsEvent]

    def __init__(out self):
        self.events = Events[PhysicsEvent]()


struct TriggerEventsResource(Resource, Movable):
    """Per-frame `Events[TriggerEvent]` drained from `PhysicsWorld.step`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.physics.TriggerEventsResource")
    var events: Events[TriggerEvent]

    def __init__(out self):
        self.events = Events[TriggerEvent]()


@always_inline
def _hash_vec3_list(pts: List[Vec3]) -> UInt64:
    var h = UInt64(14695981039346656037)
    for i in range(len(pts)):
        var p = pts[i]
        h = (h ^ UInt64(float32_bits(p.x))) * UInt64(1099511628211)
        h = (h ^ UInt64(float32_bits(p.y))) * UInt64(1099511628211)
        h = (h ^ UInt64(float32_bits(p.z))) * UInt64(1099511628211)
    return h


def _convex_hull_half_extents(hull: ConvexHull) -> Vec3:
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


struct _TrackedRigidSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var body_id: Int
    var last_pos: Vec3
    var last_rot: Quat
    var has_col: Bool
    var col_shape: Int
    var col_radius: Float32
    var col_half_height: Float32
    var col_half_extents: Vec3
    var col_local_offset: Vec3
    var col_local_rot: Quat
    var col_is_trigger: Bool
    var col_filter: CollisionFilter
    var col_points_len: Int
    var col_points_hash: UInt64

    def __init__(out self, entity_bits: UInt64 = 0, body_id: Int = -1, last_pos: Vec3 = Vec3(), last_rot: Quat = Quat()):
        self.entity_bits = entity_bits
        self.body_id = body_id
        self.last_pos = last_pos
        self.last_rot = last_rot
        self.has_col = False
        self.col_shape = BODY_SHAPE_BOX
        self.col_radius = 0.0
        self.col_half_height = 0.0
        self.col_half_extents = Vec3(0.5, 0.5, 0.5)
        self.col_local_offset = Vec3()
        self.col_local_rot = Quat()
        self.col_is_trigger = False
        self.col_filter = CollisionFilter()
        self.col_points_len = 0
        self.col_points_hash = 0


struct _TrackedVehicleSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var vehicle_slot: Int
    var body_id: Int

    def __init__(out self, entity_bits: UInt64 = 0, vehicle_slot: Int = -1, body_id: Int = -1):
        self.entity_bits = entity_bits
        self.vehicle_slot = vehicle_slot
        self.body_id = body_id


struct _TrackedAeroSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var aero_slot: Int
    var body_id: Int

    def __init__(out self, entity_bits: UInt64 = 0, aero_slot: Int = -1, body_id: Int = -1):
        self.entity_bits = entity_bits
        self.aero_slot = aero_slot
        self.body_id = body_id


struct _TrackedClothSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var cloth_slot: Int

    def __init__(out self, entity_bits: UInt64 = 0, cloth_slot: Int = -1):
        self.entity_bits = entity_bits
        self.cloth_slot = cloth_slot


struct _TrackedTriggerSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var trigger_id: Int

    def __init__(out self, entity_bits: UInt64 = 0, trigger_id: Int = -1):
        self.entity_bits = entity_bits
        self.trigger_id = trigger_id


struct _TrackedJointSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var runtime_index: Int
    var is_six_dof: Bool
    var body_id_a: Int
    var body_id_b: Int

    def __init__(
        out self,
        entity_bits: UInt64 = 0,
        runtime_index: Int = -1,
        is_six_dof: Bool = False,
        body_id_a: Int = -1,
        body_id_b: Int = -1,
    ):
        self.entity_bits = entity_bits
        self.runtime_index = runtime_index
        self.is_six_dof = is_six_dof
        self.body_id_a = body_id_a
        self.body_id_b = body_id_b


struct PhysicsSyncStateResource(Resource, Movable):
    """Tracks ECS entity <-> `PhysicsWorld` body/slot IDs and last-synced poses."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.physics.PhysicsSyncStateResource")
    var entity_to_slot: Dict[UInt64, Int]
    var tracked: List[_TrackedRigidSync]
    var body_to_entity: Dict[Int, UInt64]
    var tracked_vehicles: List[_TrackedVehicleSync]
    var tracked_aeros: List[_TrackedAeroSync]
    var tracked_cloths: List[_TrackedClothSync]
    var tracked_triggers: List[_TrackedTriggerSync]
    var tracked_joints: List[_TrackedJointSync]

    def __init__(out self):
        self.entity_to_slot = Dict[UInt64, Int]()
        self.tracked = List[_TrackedRigidSync]()
        self.body_to_entity = Dict[Int, UInt64]()
        self.tracked_vehicles = List[_TrackedVehicleSync]()
        self.tracked_aeros = List[_TrackedAeroSync]()
        self.tracked_cloths = List[_TrackedClothSync]()
        self.tracked_triggers = List[_TrackedTriggerSync]()
        self.tracked_joints = List[_TrackedJointSync]()

    def record_body(mut self, entity: Entity, body_id: Int, pos: Vec3, rot: Quat):
        var bits = entity.to_bits()
        self.body_to_entity[body_id] = bits
        var found = self.entity_to_slot.get(bits)
        if found:
            var idx = found.value()
            self.tracked[idx].body_id = body_id
            self.tracked[idx].last_pos = pos
            self.tracked[idx].last_rot = rot
        else:
            var idx = len(self.tracked)
            self.entity_to_slot[bits] = idx
            self.tracked.append(_TrackedRigidSync(bits, body_id, pos, rot))

    def record_body_collider(
        mut self,
        entity: Entity,
        col_opt: Optional[Pointer[ColliderComponent, ImmutAnyOrigin]],
    ):
        var found = self.entity_to_slot.get(entity.to_bits())
        if not found:
            return
        var idx = found.value()
        if col_opt:
            ref col = col_opt.value()[unsafe_offset=0]
            self.tracked[idx].has_col = True
            self.tracked[idx].col_shape = col.shape
            self.tracked[idx].col_radius = col.radius
            self.tracked[idx].col_half_height = col.half_height
            self.tracked[idx].col_half_extents = col.half_extents
            self.tracked[idx].col_local_offset = col.local_offset
            self.tracked[idx].col_local_rot = col.local_rot
            self.tracked[idx].col_is_trigger = col.is_trigger
            self.tracked[idx].col_filter = col.filter
            self.tracked[idx].col_points_len = len(col.points)
            self.tracked[idx].col_points_hash = (
                _hash_vec3_list(col.points)
                if (col.shape == BODY_SHAPE_CONVEX_HULL and len(col.points) >= 4)
                else UInt64(0)
            )
        else:
            self.tracked[idx].has_col = False
            self.tracked[idx].col_points_len = 0
            self.tracked[idx].col_points_hash = 0

    def collider_changed(
        self,
        entity: Entity,
        col_opt: Optional[Pointer[ColliderComponent, ImmutAnyOrigin]],
    ) -> Bool:
        var found = self.entity_to_slot.get(entity.to_bits())
        if not found:
            return True
        ref tr = self.tracked[found.value()]
        if not col_opt:
            return tr.has_col
        if not tr.has_col:
            return True
        ref col = col_opt.value()[unsafe_offset=0]
        if (
            tr.col_shape != col.shape
            or tr.col_radius != col.radius
            or tr.col_half_height != col.half_height
            or tr.col_half_extents.x != col.half_extents.x
            or tr.col_half_extents.y != col.half_extents.y
            or tr.col_half_extents.z != col.half_extents.z
            or tr.col_local_offset.x != col.local_offset.x
            or tr.col_local_offset.y != col.local_offset.y
            or tr.col_local_offset.z != col.local_offset.z
            or tr.col_local_rot.x != col.local_rot.x
            or tr.col_local_rot.y != col.local_rot.y
            or tr.col_local_rot.z != col.local_rot.z
            or tr.col_local_rot.w != col.local_rot.w
            or tr.col_is_trigger != col.is_trigger
            or tr.col_filter != col.filter
            or tr.col_points_len != len(col.points)
        ):
            return True
        if col.shape == BODY_SHAPE_CONVEX_HULL and len(col.points) >= 4:
            if tr.col_points_hash != _hash_vec3_list(col.points):
                return True
        return False

    def record_vehicle(mut self, entity: Entity, vehicle_slot: Int, body_id: Int):
        var bits = entity.to_bits()
        for i in range(len(self.tracked_vehicles)):
            if self.tracked_vehicles[i].entity_bits == bits:
                self.tracked_vehicles[i].vehicle_slot = vehicle_slot
                self.tracked_vehicles[i].body_id = body_id
                return
        self.tracked_vehicles.append(_TrackedVehicleSync(bits, vehicle_slot, body_id))

    def record_aero(mut self, entity: Entity, aero_slot: Int, body_id: Int):
        var bits = entity.to_bits()
        for i in range(len(self.tracked_aeros)):
            if self.tracked_aeros[i].entity_bits == bits:
                self.tracked_aeros[i].aero_slot = aero_slot
                self.tracked_aeros[i].body_id = body_id
                return
        self.tracked_aeros.append(_TrackedAeroSync(bits, aero_slot, body_id))

    def record_cloth(mut self, entity: Entity, cloth_slot: Int):
        var bits = entity.to_bits()
        for i in range(len(self.tracked_cloths)):
            if self.tracked_cloths[i].entity_bits == bits:
                self.tracked_cloths[i].cloth_slot = cloth_slot
                return
        self.tracked_cloths.append(_TrackedClothSync(bits, cloth_slot))

    def record_trigger(mut self, entity: Entity, trigger_id: Int):
        var bits = entity.to_bits()
        for i in range(len(self.tracked_triggers)):
            if self.tracked_triggers[i].entity_bits == bits:
                self.tracked_triggers[i].trigger_id = trigger_id
                return
        self.tracked_triggers.append(_TrackedTriggerSync(bits, trigger_id))

    def record_joint(
        mut self,
        entity: Entity,
        runtime_index: Int,
        is_six_dof: Bool,
        body_id_a: Int = -1,
        body_id_b: Int = -1,
    ):
        var bits = entity.to_bits()
        for i in range(len(self.tracked_joints)):
            if self.tracked_joints[i].entity_bits == bits:
                self.tracked_joints[i].runtime_index = runtime_index
                self.tracked_joints[i].is_six_dof = is_six_dof
                self.tracked_joints[i].body_id_a = body_id_a
                self.tracked_joints[i].body_id_b = body_id_b
                return
        self.tracked_joints.append(_TrackedJointSync(bits, runtime_index, is_six_dof, body_id_a, body_id_b))

    def tracked_joint(self, entity: Entity) -> Optional[_TrackedJointSync]:
        var bits = entity.to_bits()
        for i in range(len(self.tracked_joints)):
            if self.tracked_joints[i].entity_bits == bits:
                return self.tracked_joints[i]
        return None

    def lookup_entity(self, body_id: Int) -> Entity:
        var found = self.body_to_entity.get(body_id)
        if found:
            return Entity.from_bits(found.value())
        return NULL_ENTITY


# ---------------------------------------------------------------------------
# 7 Serializable ECS Physics Components
# ---------------------------------------------------------------------------

struct RigidBodyComponent(SerializableComponent, ImplicitlyCopyable):
    """Rigid-body simulation parameters & runtime `body_id` handle."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.RigidBodyComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "RigidBodyComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PRBD")
    comptime SCHEMA_VERSION: Int = 1

    var motion_mode: UInt8
    var mass: Float32
    var linear_vel: Vec3
    var angular_vel: Vec3
    var friction: Float32
    var restitution: Float32
    var rolling_resistance: Float32
    var linear_damping: Float32
    var angular_damping: Float32
    var gravity_scale: Float32
    var filter: CollisionFilter
    var can_sleep: Bool
    var enable_ccd: Bool
    var material: Int
    var active: Bool
    var body_id: Int

    def __init__(
        out self,
        motion_mode: UInt8 = BODY_MOTION_DYNAMIC,
        mass: Float32 = 1.0,
        linear_vel: Vec3 = Vec3(),
        angular_vel: Vec3 = Vec3(),
        friction: Float32 = 0.6,
        restitution: Float32 = 0.1,
        rolling_resistance: Float32 = 0.0,
        linear_damping: Float32 = 0.02,
        angular_damping: Float32 = 0.05,
        gravity_scale: Float32 = 1.0,
        filter: CollisionFilter = CollisionFilter(),
        can_sleep: Bool = True,
        enable_ccd: Bool = False,
        material: Int = 0,
        active: Bool = True,
        body_id: Int = -1,
    ):
        self.motion_mode = motion_mode
        self.mass = mass
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        self.friction = friction
        self.restitution = restitution
        self.rolling_resistance = rolling_resistance
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.gravity_scale = gravity_scale
        self.filter = filter
        self.can_sleep = can_sleep
        self.enable_ccd = enable_ccd
        self.material = material
        self.active = active
        self.body_id = body_id

    @staticmethod
    def dynamic(mass: Float32 = 1.0) -> Self:
        return Self(motion_mode=BODY_MOTION_DYNAMIC, mass=mass)

    @staticmethod
    def static_body() -> Self:
        return Self(motion_mode=BODY_MOTION_STATIC, mass=0.0, can_sleep=False)

    @staticmethod
    def kinematic() -> Self:
        return Self(motion_mode=BODY_MOTION_KINEMATIC, mass=0.0, can_sleep=False)

    def write_bin(self, mut out: BlobWriter):
        out.u8(self.motion_mode)
        out.f32(self.mass)
        out.vec3(self.linear_vel)
        out.vec3(self.angular_vel)
        out.f32(self.friction)
        out.f32(self.restitution)
        out.f32(self.rolling_resistance)
        out.f32(self.linear_damping)
        out.f32(self.angular_damping)
        out.f32(self.gravity_scale)
        _write_filter(out, self.filter)
        out.boolean(self.can_sleep)
        out.boolean(self.enable_ccd)
        out.integer(self.material)
        out.boolean(self.active)
        out.integer(self.body_id)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var mode = reader.u8()
        var mass = reader.f32()
        var lv = reader.vec3()
        var av = reader.vec3()
        var fric = reader.f32()
        var rest = reader.f32()
        var roll = reader.f32()
        var ld = reader.f32()
        var ad = reader.f32()
        var gs = reader.f32()
        var filt = _read_filter(reader)
        var sleep = reader.boolean()
        var ccd = reader.boolean()
        var mat = reader.integer()
        var act = reader.boolean()
        _ = reader.integer()
        if reader.failed:
            raise Error("RigidBodyComponent: truncated blob")
        return Self(mode, mass, lv, av, fric, rest, roll, ld, ad, gs, filt, sleep, ccd, mat, act, -1)

    def write_json(self) -> String:
        return (
            "{\"motion_mode\": " + String(Int(self.motion_mode))
            + ", \"mass\": " + _f32_s(self.mass)
            + ", \"linear_vel\": " + _vec3_json(self.linear_vel)
            + ", \"angular_vel\": " + _vec3_json(self.angular_vel)
            + ", \"friction\": " + _f32_s(self.friction)
            + ", \"restitution\": " + _f32_s(self.restitution)
            + ", \"rolling_resistance\": " + _f32_s(self.rolling_resistance)
            + ", \"linear_damping\": " + _f32_s(self.linear_damping)
            + ", \"angular_damping\": " + _f32_s(self.angular_damping)
            + ", \"gravity_scale\": " + _f32_s(self.gravity_scale)
            + ", \"filter\": " + _filter_json(self.filter)
            + ", \"can_sleep\": " + ("true" if self.can_sleep else "false")
            + ", \"enable_ccd\": " + ("true" if self.enable_ccd else "false")
            + ", \"material\": " + String(self.material)
            + ", \"active\": " + ("true" if self.active else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var out = Self()
        out.motion_mode = UInt8(doc.integer(doc.object_get(node, "motion_mode"), Int(BODY_MOTION_DYNAMIC)))
        out.mass = Float32(doc.number(doc.object_get(node, "mass"), 1.0))
        out.linear_vel = _read_vec3_json(doc, doc.object_get(node, "linear_vel"), Vec3())
        out.angular_vel = _read_vec3_json(doc, doc.object_get(node, "angular_vel"), Vec3())
        out.friction = Float32(doc.number(doc.object_get(node, "friction"), 0.6))
        out.restitution = Float32(doc.number(doc.object_get(node, "restitution"), 0.1))
        out.rolling_resistance = Float32(doc.number(doc.object_get(node, "rolling_resistance"), Float64(out.rolling_resistance)))
        out.linear_damping = Float32(doc.number(doc.object_get(node, "linear_damping"), Float64(out.linear_damping)))
        out.angular_damping = Float32(doc.number(doc.object_get(node, "angular_damping"), Float64(out.angular_damping)))
        out.gravity_scale = Float32(doc.number(doc.object_get(node, "gravity_scale"), 1.0))
        out.filter = _read_filter_json(doc, doc.object_get(node, "filter"), CollisionFilter())
        out.can_sleep = doc.boolean(doc.object_get(node, "can_sleep"), True)
        out.enable_ccd = doc.boolean(doc.object_get(node, "enable_ccd"), False)
        out.material = doc.integer(doc.object_get(node, "material"), out.material)
        out.active = doc.boolean(doc.object_get(node, "active"), True)
        return out


struct ColliderComponent(SerializableComponent):
    """Collider shape schema (`Sphere`, `Capsule`, `Box`, `ConvexHull`) or `TriggerVolume`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.ColliderComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "ColliderComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PCOL")
    comptime SCHEMA_VERSION: Int = 1

    var shape: Int
    var radius: Float32
    var half_height: Float32
    var half_extents: Vec3
    var local_offset: Vec3
    var local_rot: Quat
    var is_trigger: Bool
    var filter: CollisionFilter
    var points: List[Vec3]
    var trigger_id: Int

    def __init__(
        out self,
        shape: Int = BODY_SHAPE_BOX,
        radius: Float32 = 0.5,
        half_height: Float32 = 0.5,
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        local_offset: Vec3 = Vec3(),
        local_rot: Quat = Quat(),
        is_trigger: Bool = False,
        filter: CollisionFilter = CollisionFilter(),
        trigger_id: Int = -1,
    ):
        self.shape = shape
        self.radius = radius
        self.half_height = half_height
        self.half_extents = half_extents
        self.local_offset = local_offset
        self.local_rot = local_rot
        self.is_trigger = is_trigger
        self.filter = filter
        self.points = List[Vec3]()
        self.trigger_id = trigger_id

    @staticmethod
    def sphere(radius: Float32 = 0.5, is_trigger: Bool = False) -> Self:
        return Self(
            shape=BODY_SHAPE_SPHERE,
            radius=radius,
            half_extents=Vec3(radius, radius, radius),
            is_trigger=is_trigger,
        )

    @staticmethod
    def capsule(radius: Float32 = 0.35, half_height: Float32 = 0.5, is_trigger: Bool = False) -> Self:
        return Self(
            shape=BODY_SHAPE_CAPSULE,
            radius=radius,
            half_height=half_height,
            half_extents=Vec3(radius, half_height + radius, radius),
            is_trigger=is_trigger,
        )

    @staticmethod
    def box(half_extents: Vec3 = Vec3(0.5, 0.5, 0.5), is_trigger: Bool = False) -> Self:
        return Self(
            shape=BODY_SHAPE_BOX,
            radius=half_extents.length(),
            half_extents=half_extents,
            is_trigger=is_trigger,
        )

    @staticmethod
    def convex_hull(points: List[Vec3]) -> Self:
        var c = Self(shape=BODY_SHAPE_CONVEX_HULL)
        c.points = points.copy()
        return c^

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.shape)
        out.f32(self.radius)
        out.f32(self.half_height)
        out.vec3(self.half_extents)
        out.vec3(self.local_offset)
        out.quat(self.local_rot)
        out.boolean(self.is_trigger)
        _write_filter(out, self.filter)
        out.integer(len(self.points))
        for i in range(len(self.points)):
            out.vec3(self.points[i])
        out.integer(self.trigger_id)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var sh = reader.integer()
        var rad = reader.f32()
        var hh = reader.f32()
        var he = reader.vec3()
        var off = reader.vec3()
        var rot = reader.quat()
        var trig = reader.boolean()
        var filt = _read_filter(reader)
        var n_pts = reader.integer()
        if reader.failed or n_pts < 0 or n_pts > 1 << 20 or n_pts > max(0, reader.limit() - reader.offset) // 12:
            raise Error("ColliderComponent: invalid points count")
        var pts = List[Vec3](capacity=n_pts)
        for _ in range(n_pts):
            pts.append(reader.vec3())
        _ = reader.integer()
        if reader.failed:
            raise Error("ColliderComponent: truncated blob")
        var out = Self(sh, rad, hh, he, off, rot, trig, filt, -1)
        out.points = pts^
        return out^

    def write_json(self) -> String:
        var pts_s = String("[")
        for i in range(len(self.points)):
            if i > 0:
                pts_s += ", "
            pts_s += _vec3_json(self.points[i])
        pts_s += "]"
        return (
            "{\"shape\": " + String(self.shape)
            + ", \"radius\": " + _f32_s(self.radius)
            + ", \"half_height\": " + _f32_s(self.half_height)
            + ", \"half_extents\": " + _vec3_json(self.half_extents)
            + ", \"local_offset\": " + _vec3_json(self.local_offset)
            + ", \"local_rot\": " + _quat_json(self.local_rot)
            + ", \"is_trigger\": " + ("true" if self.is_trigger else "false")
            + ", \"filter\": " + _filter_json(self.filter)
            + ", \"points\": " + pts_s
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var out = Self()
        out.shape = doc.integer(doc.object_get(node, "shape"), BODY_SHAPE_BOX)
        out.radius = Float32(doc.number(doc.object_get(node, "radius"), 0.5))
        out.half_height = Float32(doc.number(doc.object_get(node, "half_height"), 0.5))
        out.half_extents = _read_vec3_json(doc, doc.object_get(node, "half_extents"), Vec3(0.5, 0.5, 0.5))
        out.local_offset = _read_vec3_json(doc, doc.object_get(node, "local_offset"), Vec3())
        out.local_rot = _read_quat_json(doc, doc.object_get(node, "local_rot"))
        out.is_trigger = doc.boolean(doc.object_get(node, "is_trigger"), False)
        out.filter = _read_filter_json(doc, doc.object_get(node, "filter"), CollisionFilter())
        var pts_node = doc.object_get(node, "points")
        var np = doc.array_size(pts_node)
        for i in range(np):
            out.points.append(_read_vec3_json(doc, doc.at(pts_node, i), Vec3()))
        return out^


struct JointComponent(SerializableComponent, ImplicitlyCopyable):
    """Joint constraint linking owner entity to `entity_b` (or world if null)."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.JointComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "JointComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PJNT")
    comptime SCHEMA_VERSION: Int = 1

    var joint_kind: Int
    var entity_b: Entity
    var anchor_a: Vec3
    var anchor_b: Vec3
    var axis: Vec3
    var lower: Float32
    var upper: Float32
    var cone: Float32
    var length: Float32
    var enable_limit: Bool
    var enable_motor: Bool
    var enable_spring: Bool
    var motor_speed: Float32
    var max_motor: Float32
    var spring_hertz: Float32
    var spring_damping: Float32
    var break_force: Float32
    var break_torque: Float32
    var hertz: Float32
    var damping: Float32
    var is_six_dof: Bool
    var runtime_index: Int

    def __init__(
        out self,
        joint_kind: Int = JOINT_WELD,
        entity_b: Entity = NULL_ENTITY,
        anchor_a: Vec3 = Vec3(),
        anchor_b: Vec3 = Vec3(),
        axis: Vec3 = Vec3(0.0, 1.0, 0.0),
        lower: Float32 = 0.0,
        upper: Float32 = 0.0,
        cone: Float32 = 0.0,
        length: Float32 = -1.0,
        enable_limit: Bool = False,
        enable_motor: Bool = False,
        enable_spring: Bool = False,
        motor_speed: Float32 = 0.0,
        max_motor: Float32 = 0.0,
        spring_hertz: Float32 = 0.0,
        spring_damping: Float32 = 0.0,
        break_force: Float32 = 0.0,
        break_torque: Float32 = 0.0,
        hertz: Float32 = 0.0,
        damping: Float32 = 0.0,
        is_six_dof: Bool = False,
        runtime_index: Int = -1,
    ):
        self.joint_kind = joint_kind
        self.entity_b = entity_b
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        self.axis = axis
        self.lower = lower
        self.upper = upper
        self.cone = cone
        self.length = length
        self.enable_limit = enable_limit
        self.enable_motor = enable_motor
        self.enable_spring = enable_spring
        self.motor_speed = motor_speed
        self.max_motor = max_motor
        self.spring_hertz = spring_hertz
        self.spring_damping = spring_damping
        self.break_force = break_force
        self.break_torque = break_torque
        self.hertz = hertz
        self.damping = damping
        self.is_six_dof = is_six_dof
        self.runtime_index = runtime_index

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.joint_kind)
        out.u64(self.entity_b.to_bits())
        out.vec3(self.anchor_a)
        out.vec3(self.anchor_b)
        out.vec3(self.axis)
        out.f32(self.lower)
        out.f32(self.upper)
        out.f32(self.cone)
        out.f32(self.length)
        out.boolean(self.enable_limit)
        out.boolean(self.enable_motor)
        out.boolean(self.enable_spring)
        out.f32(self.motor_speed)
        out.f32(self.max_motor)
        out.f32(self.spring_hertz)
        out.f32(self.spring_damping)
        out.f32(self.break_force)
        out.f32(self.break_torque)
        out.f32(self.hertz)
        out.f32(self.damping)
        out.boolean(self.is_six_dof)
        out.integer(self.runtime_index)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var jk = reader.integer()
        var eb = Entity.from_bits(reader.u64())
        var aa = reader.vec3()
        var ab = reader.vec3()
        var ax = reader.vec3()
        var lo = reader.f32()
        var hi = reader.f32()
        var cn = reader.f32()
        var ln = reader.f32()
        var el = reader.boolean()
        var em = reader.boolean()
        var es = reader.boolean()
        var ms = reader.f32()
        var mm = reader.f32()
        var sh = reader.f32()
        var sd = reader.f32()
        var bf = reader.f32()
        var bt = reader.f32()
        var hz = reader.f32()
        var dp = reader.f32()
        var s6 = reader.boolean()
        _ = reader.integer()
        if reader.failed:
            raise Error("JointComponent: truncated blob")
        return Self(jk, eb, aa, ab, ax, lo, hi, cn, ln, el, em, es, ms, mm, sh, sd, bf, bt, hz, dp, s6, -1)

    def write_json(self) -> String:
        return (
            "{\"joint_kind\": " + String(self.joint_kind)
            + ", \"anchor_a\": " + _vec3_json(self.anchor_a)
            + ", \"anchor_b\": " + _vec3_json(self.anchor_b)
            + ", \"axis\": " + _vec3_json(self.axis)
            + ", \"break_force\": " + _f32_s(self.break_force)
            + ", \"break_torque\": " + _f32_s(self.break_torque)
            + ", \"is_six_dof\": " + ("true" if self.is_six_dof else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var out = Self()
        out.joint_kind = doc.integer(doc.object_get(node, "joint_kind"), JOINT_WELD)
        out.anchor_a = _read_vec3_json(doc, doc.object_get(node, "anchor_a"), Vec3())
        out.anchor_b = _read_vec3_json(doc, doc.object_get(node, "anchor_b"), Vec3())
        out.axis = _read_vec3_json(doc, doc.object_get(node, "axis"), Vec3(0.0, 1.0, 0.0))
        out.break_force = Float32(doc.number(doc.object_get(node, "break_force"), 0.0))
        out.break_torque = Float32(doc.number(doc.object_get(node, "break_torque"), 0.0))
        out.is_six_dof = doc.boolean(doc.object_get(node, "is_six_dof"), False)
        return out


struct CharacterControllerComponent(SerializableComponent):
    """Kinematic character controller ECS component backed by `CharacterController`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.CharacterControllerComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "CharacterControllerComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PKCC")
    comptime SCHEMA_VERSION: Int = 1

    var config: CharacterControllerConfig
    var controller: CharacterController
    var desired_vel: Vec3
    var jump_speed: Float32
    var active: Bool

    def __init__(
        out self,
        pos: Vec3 = Vec3(0.0, 1.0, 0.0),
        config: CharacterControllerConfig = CharacterControllerConfig(),
        desired_vel: Vec3 = Vec3(),
        jump_speed: Float32 = 0.0,
        active: Bool = True,
    ):
        self.config = config
        self.controller = CharacterController(pos, config)
        self.desired_vel = desired_vel
        self.jump_speed = jump_speed
        self.active = active

    def write_bin(self, mut out: BlobWriter):
        out.f32(self.config.radius)
        out.f32(self.config.half_height)
        out.f32(self.config.skin_width)
        out.f32(self.config.max_slope_deg)
        out.f32(self.config.step_height)
        out.f32(self.config.snap_down_dist)
        out.integer(self.config.max_slides)
        out.integer(self.config.depenetration_iters)
        out.f32(self.config.mass)
        out.f32(self.config.push_force_scale)
        out.f32(self.config.max_push_mass_ratio)
        out.f32(self.config.gravity)
        out.vec3(self.config.up)
        _write_filter(out, self.config.filter)
        out.integer(self.config.ignore_body_id)
        self.controller.serialize(out)
        out.vec3(self.desired_vel)
        out.f32(self.jump_speed)
        out.boolean(self.active)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var cfg = CharacterControllerConfig()
        cfg.radius = reader.f32()
        cfg.half_height = reader.f32()
        cfg.skin_width = reader.f32()
        cfg.max_slope_deg = reader.f32()
        cfg.step_height = reader.f32()
        cfg.snap_down_dist = reader.f32()
        cfg.max_slides = reader.integer()
        cfg.depenetration_iters = reader.integer()
        cfg.mass = reader.f32()
        cfg.push_force_scale = reader.f32()
        cfg.max_push_mass_ratio = reader.f32()
        cfg.gravity = reader.f32()
        cfg.up = reader.vec3()
        cfg.filter = _read_filter(reader)
        cfg.ignore_body_id = reader.integer()
        var out = Self(Vec3(), cfg)
        out.controller.deserialize(reader)
        out.desired_vel = reader.vec3()
        out.jump_speed = reader.f32()
        out.active = reader.boolean()
        if reader.failed:
            raise Error("CharacterControllerComponent: truncated blob")
        return out^

    def write_json(self) -> String:
        return (
            "{\"pos\": " + _vec3_json(self.controller.pos)
            + ", \"radius\": " + _f32_s(self.config.radius)
            + ", \"half_height\": " + _f32_s(self.config.half_height)
            + ", \"mass\": " + _f32_s(self.config.mass)
            + ", \"desired_vel\": " + _vec3_json(self.desired_vel)
            + ", \"active\": " + ("true" if self.active else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var cfg = CharacterControllerConfig()
        cfg.radius = Float32(doc.number(doc.object_get(node, "radius"), 0.35))
        cfg.half_height = Float32(doc.number(doc.object_get(node, "half_height"), 0.55))
        cfg.mass = Float32(doc.number(doc.object_get(node, "mass"), 80.0))
        var pos = _read_vec3_json(doc, doc.object_get(node, "pos"), Vec3(0.0, 1.0, 0.0))
        var dv = _read_vec3_json(doc, doc.object_get(node, "desired_vel"), Vec3())
        var act = doc.boolean(doc.object_get(node, "active"), True)
        return Self(pos, cfg, dv, Float32(0.0), act)


struct VehicleComponent(SerializableComponent):
    """Raycast/Pacejka multi-wheel vehicle component coupled into `PhysicsWorld`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.VehicleComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "VehicleComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PVEH")
    comptime SCHEMA_VERSION: Int = 1

    var chassis: VehicleChassis
    var chassis_half_extents: Vec3
    var wheel_filter: CollisionFilter
    var body_filter: CollisionFilter
    var throttle: Float32
    var brake: Float32
    var steer: Float32
    var friction: Float32
    var restitution: Float32
    var vehicle_slot: Int

    def __init__(
        out self,
        var chassis: VehicleChassis,
        chassis_half_extents: Vec3 = Vec3(),
        wheel_filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        body_filter: CollisionFilter = CollisionFilter(),
        throttle: Float32 = 0.0,
        brake: Float32 = 0.0,
        steer: Float32 = 0.0,
        friction: Float32 = 0.4,
        restitution: Float32 = 0.1,
        vehicle_slot: Int = -1,
    ):
        self.chassis = chassis^
        self.chassis_half_extents = chassis_half_extents
        self.wheel_filter = wheel_filter
        self.body_filter = body_filter
        self.throttle = throttle
        self.brake = brake
        self.steer = steer
        self.friction = friction
        self.restitution = restitution
        self.vehicle_slot = vehicle_slot

    def __init__(out self):
        var ch = VehicleChassis()
        ch.setup_four_wheel_car()
        self.chassis = ch^
        self.chassis_half_extents = Vec3()
        self.wheel_filter = CollisionFilter(CAT_ALL, CAT_ALL, 0)
        self.body_filter = CollisionFilter()
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.friction = 0.4
        self.restitution = 0.1
        self.vehicle_slot = -1

    def write_bin(self, mut out: BlobWriter):
        self.chassis.serialize(out)
        out.vec3(self.chassis_half_extents)
        _write_filter(out, self.wheel_filter)
        _write_filter(out, self.body_filter)
        out.f32(self.throttle)
        out.f32(self.brake)
        out.f32(self.steer)
        out.f32(self.friction)
        out.f32(self.restitution)
        out.integer(self.vehicle_slot)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var ch = VehicleChassis()
        ch.deserialize(reader)
        var he = reader.vec3()
        var wf = _read_filter(reader)
        var bf = _read_filter(reader)
        var th = reader.f32()
        var br = reader.f32()
        var st = reader.f32()
        var fr = reader.f32()
        var re = reader.f32()
        _ = reader.integer()
        if reader.failed:
            raise Error("VehicleComponent: truncated blob")
        return Self(ch^, he, wf, bf, th, br, st, fr, re, -1)

    def write_json(self) -> String:
        return (
            "{\"mass\": " + _f32_s(self.chassis.mass)
            + ", \"pos\": " + _vec3_json(self.chassis.pos)
            + ", \"throttle\": " + _f32_s(self.throttle)
            + ", \"brake\": " + _f32_s(self.brake)
            + ", \"steer\": " + _f32_s(self.steer)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var mass = Float32(doc.number(doc.object_get(node, "mass"), 1450.0))
        var ch = VehicleChassis(mass)
        ch.setup_four_wheel_car()
        ch.pos = _read_vec3_json(doc, doc.object_get(node, "pos"), Vec3(0.0, 0.5, 0.0))
        var out = Self(ch^)
        out.throttle = Float32(doc.number(doc.object_get(node, "throttle"), 0.0))
        out.brake = Float32(doc.number(doc.object_get(node, "brake"), 0.0))
        out.steer = Float32(doc.number(doc.object_get(node, "steer"), 0.0))
        return out^


struct Aero6DofComponent(SerializableComponent, ImplicitlyCopyable):
    """6-DOF Aerospace flight dynamics component coupled into `PhysicsWorld`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.Aero6DofComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "Aero6DofComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PAER")
    comptime SCHEMA_VERSION: Int = 1

    var config: Aero6DofConfig
    var controls: Aero6DofControls
    var state: Aero6DofState
    var half_extents: Vec3
    var filter: CollisionFilter
    var enable_ccd: Bool
    var aero_slot: Int

    def __init__(
        out self,
        state: Aero6DofState = Aero6DofState(),
        config: Aero6DofConfig = Aero6DofConfig(),
        controls: Aero6DofControls = Aero6DofControls(),
        half_extents: Vec3 = Vec3(-1.0, -1.0, -1.0),
        filter: CollisionFilter = CollisionFilter(),
        enable_ccd: Bool = True,
        aero_slot: Int = -1,
    ):
        self.config = config
        self.controls = controls
        self.state = state
        self.half_extents = half_extents
        self.filter = filter
        self.enable_ccd = enable_ccd
        self.aero_slot = aero_slot

    def write_bin(self, mut out: BlobWriter):
        self.config.serialize(out)
        self.controls.serialize(out)
        self.state.serialize(out)
        out.vec3(self.half_extents)
        _write_filter(out, self.filter)
        out.boolean(self.enable_ccd)
        out.integer(self.aero_slot)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var cfg = Aero6DofConfig()
        var ctrl = Aero6DofControls()
        var st = Aero6DofState()
        cfg.deserialize(reader)
        ctrl.deserialize(reader)
        st.deserialize(reader)
        var he = reader.vec3()
        var filt = _read_filter(reader)
        var ccd = reader.boolean()
        _ = reader.integer()
        if reader.failed:
            raise Error("Aero6DofComponent: truncated blob")
        return Self(st, cfg, ctrl, he, filt, ccd, -1)

    def write_json(self) -> String:
        return (
            "{\"pos\": " + _vec3_json(self.state.pos)
            + ", \"vel\": " + _vec3_json(self.state.linear_vel)
            + ", \"throttle\": " + _f32_s(self.controls.throttle)
            + ", \"enable_ccd\": " + ("true" if self.enable_ccd else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var pos = _read_vec3_json(doc, doc.object_get(node, "pos"), Vec3())
        var vel = _read_vec3_json(doc, doc.object_get(node, "vel"), Vec3())
        var st = Aero6DofState(pos, vel)
        var ctrl = Aero6DofControls()
        ctrl.throttle = Float32(doc.number(doc.object_get(node, "throttle"), 0.0))
        var ccd = doc.boolean(doc.object_get(node, "enable_ccd"), True)
        return Self(st, Aero6DofConfig(), ctrl, Vec3(-1.0, -1.0, -1.0), CollisionFilter(), ccd, -1)


struct ClothMeshComponent(SerializableComponent):
    """XPBD `ParticleMesh` + `RigidAttachment` cloth component coupled into `PhysicsWorld`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.physics.ClothMeshComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "ClothMeshComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PCLT")
    comptime SCHEMA_VERSION: Int = 1

    var mesh: ParticleMesh
    var attachments: List[RigidAttachment]
    var substeps: Int
    var iterations: Int
    var filter: CollisionFilter
    var wind: Vec3
    var cloth_slot: Int

    def __init__(
        out self,
        var mesh: ParticleMesh,
        var attachments: List[RigidAttachment] = List[RigidAttachment](),
        substeps: Int = 4,
        iterations: Int = 2,
        filter: CollisionFilter = CollisionFilter(),
        wind: Vec3 = Vec3(),
        cloth_slot: Int = -1,
    ):
        self.mesh = mesh^
        self.attachments = attachments^
        self.substeps = substeps
        self.iterations = iterations
        self.filter = filter
        self.wind = wind
        self.cloth_slot = cloth_slot

    def __init__(out self):
        self.mesh = ParticleMesh()
        self.attachments = List[RigidAttachment]()
        self.substeps = 4
        self.iterations = 2
        self.filter = CollisionFilter()
        self.wind = Vec3()
        self.cloth_slot = -1

    def write_bin(self, mut out: BlobWriter):
        self.mesh.serialize(out)
        out.integer(len(self.attachments))
        for i in range(len(self.attachments)):
            self.attachments[i].serialize(out)
        out.integer(self.substeps)
        out.integer(self.iterations)
        _write_filter(out, self.filter)
        out.vec3(self.wind)
        out.integer(self.cloth_slot)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var m = ParticleMesh()
        m.deserialize(reader)
        var na = reader.integer()
        if reader.failed or na < 0 or na > 1 << 16 or na > max(0, reader.limit() - reader.offset) // 52:
            raise Error("ClothMeshComponent: invalid attachment count")
        var atts = List[RigidAttachment](capacity=na)
        for _ in range(na):
            var a = RigidAttachment(0, -1, Vec3())
            a.deserialize(reader)
            atts.append(a)
        var sub = reader.integer()
        var iters = reader.integer()
        var filt = _read_filter(reader)
        var w = reader.vec3()
        _ = reader.integer()
        if reader.failed:
            raise Error("ClothMeshComponent: truncated blob")
        return Self(m^, atts^, sub, iters, filt, w, -1)

    def write_json(self) -> String:
        return (
            "{\"substeps\": " + String(self.substeps)
            + ", \"iterations\": " + String(self.iterations)
            + ", \"wind\": " + _vec3_json(self.wind)
            + ", \"particles\": " + String(self.mesh.count())
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var out = Self()
        out.substeps = doc.integer(doc.object_get(node, "substeps"), 4)
        out.iterations = doc.integer(doc.object_get(node, "iterations"), 2)
        out.wind = _read_vec3_json(doc, doc.object_get(node, "wind"), Vec3())
        return out^


# ---------------------------------------------------------------------------
# Component & Codec Registration
# ---------------------------------------------------------------------------

def register_physics_components(mut world: World):
    """Idempotently registers all 7 physics components + binary hooks for non-trivial/handle-bearing structs."""
    _register_phys_bin_hooks[RigidBodyComponent](world)
    _register_non_trivial_phys[ColliderComponent](world)
    _register_phys_bin_hooks[JointComponent](world)
    _register_non_trivial_phys[CharacterControllerComponent](world)
    _register_non_trivial_phys[VehicleComponent](world)
    _register_phys_bin_hooks[Aero6DofComponent](world)
    _register_non_trivial_phys[ClothMeshComponent](world)


def register_physics_codecs(mut reg: ComponentRegistry):
    """Registers all 7 physics `SerializableComponent` codecs on `ComponentRegistry`."""
    reg.register[RigidBodyComponent]()
    reg.register[ColliderComponent]()
    reg.register[JointComponent]()
    reg.register[CharacterControllerComponent]()
    reg.register[VehicleComponent]()
    reg.register[Aero6DofComponent]()
    reg.register[ClothMeshComponent]()


# ---------------------------------------------------------------------------
# Transform Helpers & Systems
# ---------------------------------------------------------------------------

def _entity_world_pose(world: World, entity: Entity) -> Tuple[Vec3, Quat]:
    var lt_opt = world.get_ptr[LocalTransform](entity)
    if not lt_opt:
        var gt_opt = world.get_ptr[GlobalTransform](entity)
        if gt_opt:
            return (gt_opt.value()[unsafe_offset=0].translation, gt_opt.value()[unsafe_offset=0].rotation)
        return (Vec3(), Quat())
    var lt = lt_opt.value()[unsafe_offset=0]
    var par_opt = world.get_ptr[Parent](entity)
    if par_opt:
        var par_e = par_opt.value()[unsafe_offset=0].entity
        if world.is_alive(par_e):
            var p_gt = world.get_ptr[GlobalTransform](par_e)
            if p_gt:
                var comp = p_gt.value()[unsafe_offset=0].compose_child(lt)
                return (comp.translation, comp.rotation)
    return (lt.translation, lt.rotation.normalized())


def _write_world_pose_to_local(mut world: World, entity: Entity, world_pos: Vec3, world_rot: Quat):
    var lt_opt = world.get_ptr_mut[LocalTransform](entity)
    if not lt_opt:
        return
    var par_opt = world.get_ptr[Parent](entity)
    if par_opt:
        var par_e = par_opt.value()[unsafe_offset=0].entity
        if world.is_alive(par_e):
            var p_gt_opt = world.get_ptr[GlobalTransform](par_e)
            if p_gt_opt:
                ref p_gt = p_gt_opt.value()[unsafe_offset=0]
                var inv_q = p_gt.rotation.conjugate()
                var rel = inv_q.rotate(world_pos - p_gt.translation)
                var sx = p_gt.scale.x if abs(p_gt.scale.x) > Float32(1.0e-6) else Float32(1.0)
                var sy = p_gt.scale.y if abs(p_gt.scale.y) > Float32(1.0e-6) else Float32(1.0)
                var sz = p_gt.scale.z if abs(p_gt.scale.z) > Float32(1.0e-6) else Float32(1.0)
                lt_opt.value()[unsafe_offset=0].translation = Vec3(rel.x / sx, rel.y / sy, rel.z / sz)
                lt_opt.value()[unsafe_offset=0].rotation = (inv_q * world_rot).normalized()
                return
    lt_opt.value()[unsafe_offset=0].translation = world_pos
    lt_opt.value()[unsafe_offset=0].rotation = world_rot.normalized()


@always_inline
def _effective_body_filter(
    rb: RigidBodyComponent,
    col_opt: Optional[Pointer[ColliderComponent, ImmutAnyOrigin]],
) -> CollisionFilter:
    if col_opt and col_opt.value()[unsafe_offset=0].is_trigger:
        return CollisionFilter(0, 0, 0)
    if rb.filter != CollisionFilter():
        return rb.filter
    if col_opt and col_opt.value()[unsafe_offset=0].filter != CollisionFilter():
        return col_opt.value()[unsafe_offset=0].filter
    return rb.filter


def _sync_body_collider(
    mut physics: PhysicsWorld,
    slot: Int,
    rb: RigidBodyComponent,
    col_opt: Optional[Pointer[ColliderComponent, ImmutAnyOrigin]],
    target_pos: Vec3,
    target_rot: Quat,
):
    if slot < 0 or slot >= len(physics.bodies):
        return
    var is_dyn = (rb.motion_mode == BODY_MOTION_DYNAMIC) and (not physics.bodies[slot].is_static)
    var eff_mass = max(Float32(0.001), rb.mass) if is_dyn else Float32(0.0)
    if col_opt:
        ref col = col_opt.value()[unsafe_offset=0]
        if col.shape == BODY_SHAPE_SPHERE:
            ref b = physics.bodies[slot]
            b.shape = BODY_SHAPE_SPHERE
            b.shape_index = -1
            b.radius = max(Float32(0.01), col.radius)
            b.half_extents = Vec3(b.radius, b.radius, b.radius)
            if is_dyn:
                b.local_inertia = compute_shape_inertia(BODY_SHAPE_SPHERE, eff_mass, b.radius, 0.0, b.half_extents)
                b.wind_area_over_mass = (Float32(3.14159265) * b.radius * b.radius) / eff_mass
        elif col.shape == BODY_SHAPE_CAPSULE:
            ref b = physics.bodies[slot]
            b.shape = BODY_SHAPE_CAPSULE
            b.shape_index = -1
            b.radius = max(Float32(0.01), col.radius)
            b.half_height = max(Float32(0.0), col.half_height)
            b.half_extents = Vec3(b.radius, b.half_height + b.radius, b.radius)
            if is_dyn:
                b.local_inertia = compute_shape_inertia(BODY_SHAPE_CAPSULE, eff_mass, b.radius, b.half_height, b.half_extents)
                b.wind_area_over_mass = (Float32(2.0) * b.radius * (Float32(2.0) * b.half_height + Float32(2.0) * b.radius)) / eff_mass
        elif col.shape == BODY_SHAPE_CONVEX_HULL and len(col.points) >= 4:
            var hull = ConvexHull.build_quickhull(col.points)
            var h_idx = len(physics.hulls)
            physics.hulls.append(hull.copy())
            var he = _convex_hull_half_extents(hull)
            ref b = physics.bodies[slot]
            b.shape = BODY_SHAPE_CONVEX_HULL
            b.shape_index = h_idx
            b.half_extents = he
            b.radius = he.length()
            if is_dyn:
                var mp = hull.compute_polyhedral_mass_properties(Float32(1.0))
                if mp[0] > Float32(1.0e-6):
                    var scale_m = eff_mass / mp[0]
                    b.local_inertia = Vec3(
                        max(Float32(1.0e-5), mp[2].xx * scale_m),
                        max(Float32(1.0e-5), mp[2].yy * scale_m),
                        max(Float32(1.0e-5), mp[2].zz * scale_m),
                    )
                else:
                    b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, eff_mass, 0.0, 0.0, he)
                b.wind_area_over_mass = (Float32(4.0) * he.x * he.y) / eff_mass
        else:
            ref b = physics.bodies[slot]
            b.shape = BODY_SHAPE_BOX
            b.shape_index = -1
            b.half_extents = Vec3(
                max(Float32(0.01), col.half_extents.x),
                max(Float32(0.01), col.half_extents.y),
                max(Float32(0.01), col.half_extents.z),
            )
            b.radius = b.half_extents.length()
            if is_dyn:
                b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, eff_mass, 0.0, 0.0, b.half_extents)
                var area = Float32(4.0) * (
                    b.half_extents.x * b.half_extents.y
                    + b.half_extents.y * b.half_extents.z
                    + b.half_extents.z * b.half_extents.x
                ) * Float32(1.0 / 3.0)
                b.wind_area_over_mass = area / eff_mass
    else:
        ref b = physics.bodies[slot]
        b.shape = BODY_SHAPE_BOX
        b.shape_index = -1
        b.half_extents = Vec3(0.5, 0.5, 0.5)
        b.radius = b.half_extents.length()
        if is_dyn:
            b.local_inertia = compute_shape_inertia(BODY_SHAPE_BOX, eff_mass, 0.0, 0.0, b.half_extents)
            b.wind_area_over_mass = Float32(1.0) / eff_mass
    physics.bodies[slot].pos = target_pos
    physics.bodies[slot].rot = target_rot
    physics.bodies[slot].filter = _effective_body_filter(rb, col_opt)
    if is_dyn:
        physics.wake(slot)


def _recompute_wind_area_over_mass(mut physics: PhysicsWorld, slot: Int, eff_mass: Float32):
    if slot < 0 or slot >= len(physics.bodies):
        return
    ref b = physics.bodies[slot]
    if b.motion_mode != BODY_MOTION_DYNAMIC or b.is_static or eff_mass <= Float32(0.0):
        b.wind_area_over_mass = Float32(0.0)
        return
    if b.shape == BODY_SHAPE_SPHERE:
        b.wind_area_over_mass = (Float32(3.14159265) * b.radius * b.radius) / eff_mass
    elif b.shape == BODY_SHAPE_CAPSULE:
        b.wind_area_over_mass = (Float32(2.0) * b.radius * (Float32(2.0) * b.half_height + Float32(2.0) * b.radius)) / eff_mass
    elif b.shape == BODY_SHAPE_CONVEX_HULL:
        b.wind_area_over_mass = (Float32(4.0) * b.half_extents.x * b.half_extents.y) / eff_mass
    else:
        var area = Float32(4.0) * (
            b.half_extents.x * b.half_extents.y
            + b.half_extents.y * b.half_extents.z
            + b.half_extents.z * b.half_extents.x
        ) * Float32(1.0 / 3.0)
        b.wind_area_over_mass = area / eff_mass


def _sync_body_rb_params(
    mut physics: PhysicsWorld,
    slot: Int,
    rb: RigidBodyComponent,
    eff_filter: CollisionFilter,
):
    if slot < 0 or slot >= len(physics.bodies):
        return
    if physics.bodies[slot].motion_mode != rb.motion_mode:
        var dyn_m = max(Float32(0.001), rb.mass) if rb.motion_mode == BODY_MOTION_DYNAMIC else Float32(-1.0)
        physics.set_body_motion_mode(slot, rb.motion_mode, dyn_m)
        _recompute_wind_area_over_mass(physics, slot, dyn_m)
    elif rb.motion_mode == BODY_MOTION_DYNAMIC:
        var eff_m = max(Float32(0.001), rb.mass)
        if abs(physics.bodies[slot].mass - eff_m) > Float32(1.0e-6):
            physics.bodies[slot].mass = eff_m
            physics.bodies[slot].inv_mass = Float32(1.0) / eff_m
            physics.bodies[slot].local_inertia = compute_shape_inertia(
                physics.bodies[slot].shape,
                eff_m,
                physics.bodies[slot].radius,
                physics.bodies[slot].half_height,
                physics.bodies[slot].half_extents,
            )
            _recompute_wind_area_over_mass(physics, slot, eff_m)
    physics.bodies[slot].friction = rb.friction
    physics.bodies[slot].restitution = rb.restitution
    physics.bodies[slot].rolling_resistance = rb.rolling_resistance
    physics.bodies[slot].linear_damping = rb.linear_damping
    physics.bodies[slot].angular_damping = rb.angular_damping
    physics.bodies[slot].gravity_scale = rb.gravity_scale
    physics.bodies[slot].filter = eff_filter
    physics.bodies[slot].can_sleep = rb.can_sleep
    physics.bodies[slot].enable_ccd = rb.enable_ccd
    physics.bodies[slot].material = rb.material
    physics.bodies[slot].active = rb.active
    if rb.active and rb.motion_mode == BODY_MOTION_DYNAMIC:
        physics.wake(slot)


def _materialize_rigid_body(
    mut physics: PhysicsWorld,
    rb: RigidBodyComponent,
    col_opt: Optional[Pointer[ColliderComponent, ImmutAnyOrigin]],
    pos: Vec3,
    rot: Quat,
) -> Int:
    var eff_mass = Float32(0.0) if (rb.motion_mode == BODY_MOTION_STATIC or rb.motion_mode == BODY_MOTION_KINEMATIC) else max(Float32(0.001), rb.mass)
    var eff_filter = _effective_body_filter(rb, col_opt)
    var slot: Int
    if col_opt:
        ref col = col_opt.value()[unsafe_offset=0]
        var w_pos = pos + rot.rotate(col.local_offset)
        var w_rot = (rot * col.local_rot).normalized()
        if col.shape == BODY_SHAPE_SPHERE:
            slot = physics.add_sphere(w_pos, col.radius, eff_mass, w_rot, rb.friction, rb.restitution, eff_filter)
        elif col.shape == BODY_SHAPE_CAPSULE:
            slot = physics.add_capsule(w_pos, col.radius, col.half_height, eff_mass, w_rot, rb.friction, rb.restitution, eff_filter)
        elif col.shape == BODY_SHAPE_CONVEX_HULL and len(col.points) >= 4:
            var hull = ConvexHull.build_quickhull(col.points)
            slot = physics.add_convex_hull(w_pos, hull, eff_mass, w_rot, rb.friction, rb.restitution, eff_filter)
        else:
            slot = physics.add_box(w_pos, col.half_extents, eff_mass, w_rot, rb.friction, rb.restitution, eff_filter)
    else:
        slot = physics.add_box(pos, Vec3(0.5, 0.5, 0.5), eff_mass, rot, rb.friction, rb.restitution, eff_filter)

    if rb.motion_mode == BODY_MOTION_KINEMATIC:
        physics.set_body_motion_mode(slot, BODY_MOTION_KINEMATIC)
    ref b = physics.bodies[slot]
    b.linear_vel = rb.linear_vel
    b.angular_vel = rb.angular_vel
    b.rolling_resistance = rb.rolling_resistance
    b.linear_damping = rb.linear_damping
    b.angular_damping = rb.angular_damping
    b.gravity_scale = rb.gravity_scale
    b.filter = eff_filter
    b.can_sleep = rb.can_sleep
    b.enable_ccd = rb.enable_ccd
    b.material = rb.material
    b.active = rb.active
    return physics.body_id(slot)


def _sys_physics_pre_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var phys_opt = resources.get_ptr_mut[PhysicsWorld]()
    var sync_opt = resources.get_ptr_mut[PhysicsSyncStateResource]()
    if not phys_opt or not sync_opt:
        return
    ref physics = phys_opt.value()[unsafe_offset=0]
    ref sync = sync_opt.value()[unsafe_offset=0]

    var fixed_dt = Float32(1.0 / 60.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt:
        fixed_dt = max(Float32(1.0e-5), fix_opt.value()[unsafe_offset=0].dt)

    # 1. Remove physics bodies & non-RB physics objects for despawned entities or removed components
    var keep = List[_TrackedRigidSync]()
    for i in range(len(sync.tracked)):
        var tr = sync.tracked[i]
        var e = Entity.from_bits(tr.entity_bits)
        var rb_opt = world.get_ptr[RigidBodyComponent](e)
        var alive_rb = (
            world.is_alive(e)
            and Bool(rb_opt)
            and rb_opt.value()[unsafe_offset=0].body_id == tr.body_id
        )
        if not alive_rb:
            var slot = physics.find_slot_by_id_any(tr.body_id)
            if slot >= 0:
                physics.swap_remove_body(slot)
            _ = sync.body_to_entity.pop(tr.body_id, 0)
        else:
            keep.append(tr)
    sync.tracked = keep^
    sync.entity_to_slot.clear()
    for i in range(len(sync.tracked)):
        sync.entity_to_slot[sync.tracked[i].entity_bits] = i

    var keep_trigs = List[_TrackedTriggerSync]()
    for i in range(len(sync.tracked_triggers)):
        var tt = sync.tracked_triggers[i]
        var e = Entity.from_bits(tt.entity_bits)
        var col_opt = world.get_ptr[ColliderComponent](e)
        var alive_trig = (
            world.is_alive(e)
            and Bool(col_opt)
            and col_opt.value()[unsafe_offset=0].is_trigger
            and col_opt.value()[unsafe_offset=0].trigger_id == tt.trigger_id
        )
        if not alive_trig:
            for ti in range(len(physics.triggers)):
                if physics.triggers[ti].id == tt.trigger_id:
                    physics.triggers[ti].active = False
                    break
        else:
            keep_trigs.append(tt)
    sync.tracked_triggers = keep_trigs^

    var keep_joints = List[_TrackedJointSync]()
    for i in range(len(sync.tracked_joints)):
        var tj = sync.tracked_joints[i]
        var e = Entity.from_bits(tj.entity_bits)
        var jc_opt = world.get_ptr[JointComponent](e)
        var alive_j = (
            world.is_alive(e)
            and Bool(jc_opt)
            and jc_opt.value()[unsafe_offset=0].runtime_index == tj.runtime_index
            and jc_opt.value()[unsafe_offset=0].is_six_dof == tj.is_six_dof
        )
        if not alive_j:
            if tj.is_six_dof:
                if tj.runtime_index >= 0 and tj.runtime_index < len(physics.six_dof_joints):
                    physics.six_dof_joints[tj.runtime_index].broken = True
            else:
                if tj.runtime_index >= 0 and tj.runtime_index < len(physics.joints):
                    physics.joints[tj.runtime_index].broken = True
        else:
            keep_joints.append(tj)
    sync.tracked_joints = keep_joints^

    var keep_vehs = List[_TrackedVehicleSync]()
    for i in range(len(sync.tracked_vehicles)):
        var tv = sync.tracked_vehicles[i]
        var e = Entity.from_bits(tv.entity_bits)
        var vc_opt = world.get_ptr[VehicleComponent](e)
        var alive_v = (
            world.is_alive(e)
            and Bool(vc_opt)
            and vc_opt.value()[unsafe_offset=0].vehicle_slot == tv.vehicle_slot
        )
        if not alive_v:
            physics.remove_vehicle(tv.vehicle_slot, True)
            if tv.body_id >= 0:
                _ = sync.body_to_entity.pop(tv.body_id, 0)
        else:
            keep_vehs.append(tv)
    sync.tracked_vehicles = keep_vehs^

    var keep_aeros = List[_TrackedAeroSync]()
    for i in range(len(sync.tracked_aeros)):
        var ta = sync.tracked_aeros[i]
        var e = Entity.from_bits(ta.entity_bits)
        var ac_opt = world.get_ptr[Aero6DofComponent](e)
        var alive_a = (
            world.is_alive(e)
            and Bool(ac_opt)
            and ac_opt.value()[unsafe_offset=0].aero_slot == ta.aero_slot
        )
        if not alive_a:
            physics.remove_aero_body(ta.aero_slot, True)
            if ta.body_id >= 0:
                _ = sync.body_to_entity.pop(ta.body_id, 0)
        else:
            keep_aeros.append(ta)
    sync.tracked_aeros = keep_aeros^

    var keep_cloths = List[_TrackedClothSync]()
    for i in range(len(sync.tracked_cloths)):
        var tc = sync.tracked_cloths[i]
        var e = Entity.from_bits(tc.entity_bits)
        var cc_opt = world.get_ptr[ClothMeshComponent](e)
        var alive_c = (
            world.is_alive(e)
            and Bool(cc_opt)
            and cc_opt.value()[unsafe_offset=0].cloth_slot == tc.cloth_slot
        )
        if not alive_c:
            if tc.cloth_slot >= 0 and tc.cloth_slot < len(physics.xpbd_meshes):
                physics.xpbd_meshes[tc.cloth_slot].active = False
        else:
            keep_cloths.append(tc)
    sync.tracked_cloths = keep_cloths^

    # 2. Materialize & pre-sync RigidBodyComponent entities
    var need_broadphase_rebuild = False
    var rb_mask = world.bit_for[RigidBodyComponent]()
    var rb_entities = world.query_entities(rb_mask)
    for i in range(len(rb_entities)):
        var e = rb_entities[i]
        var rb_opt = world.get_ptr_mut[RigidBodyComponent](e)
        if not rb_opt:
            continue
        ref rb = rb_opt.value()[unsafe_offset=0]
        var pose = _entity_world_pose(world, e)
        var w_pos = pose[0]
        var w_rot = pose[1]

        var slot = physics.find_slot_by_id_any(rb.body_id) if rb.body_id >= 0 else -1
        if slot < 0:
            var col_opt = world.get_ptr[ColliderComponent](e)
            rb.body_id = _materialize_rigid_body(physics, rb, col_opt, w_pos, w_rot)
            sync.record_body(e, rb.body_id, w_pos, w_rot)
            sync.record_body_collider(e, col_opt)
        else:
            var col_off = Vec3()
            var col_rot = Quat()
            var col_opt = world.get_ptr[ColliderComponent](e)
            if col_opt:
                col_off = col_opt.value()[unsafe_offset=0].local_offset
                col_rot = col_opt.value()[unsafe_offset=0].local_rot
            var target_pos = w_pos + w_rot.rotate(col_off)
            var target_rot = (w_rot * col_rot).normalized()

            var eff_filter = _effective_body_filter(rb, col_opt)
            ref b_chk = physics.bodies[slot]
            var rb_params_changed = (
                b_chk.motion_mode != rb.motion_mode
                or b_chk.friction != rb.friction
                or b_chk.restitution != rb.restitution
                or b_chk.rolling_resistance != rb.rolling_resistance
                or b_chk.linear_damping != rb.linear_damping
                or b_chk.angular_damping != rb.angular_damping
                or b_chk.gravity_scale != rb.gravity_scale
                or b_chk.filter != eff_filter
                or b_chk.can_sleep != rb.can_sleep
                or b_chk.enable_ccd != rb.enable_ccd
                or b_chk.material != rb.material
                or b_chk.active != rb.active
                or (
                    rb.motion_mode == BODY_MOTION_DYNAMIC
                    and abs(b_chk.mass - max(Float32(0.001), rb.mass)) > Float32(1.0e-6)
                )
            )
            if rb_params_changed:
                _sync_body_rb_params(physics, slot, rb, eff_filter)
                need_broadphase_rebuild = True

            if sync.collider_changed(e, col_opt):
                _sync_body_collider(physics, slot, rb, col_opt, target_pos, target_rot)
                sync.record_body_collider(e, col_opt)
                need_broadphase_rebuild = True

            if rb.motion_mode == BODY_MOTION_KINEMATIC:
                ref b = physics.bodies[slot]
                var inv_dt = Float32(1.0) / fixed_dt
                b.linear_vel = (target_pos - b.pos) * inv_dt
                var dq = target_rot.multiply(b.rot.conjugate()).normalized()
                b.angular_vel = _quat_to_scaled_angle_phys(dq) * inv_dt
                b.awake = True
                sync.record_body(e, rb.body_id, w_pos, w_rot)
            elif rb.motion_mode == BODY_MOTION_DYNAMIC:
                var tr_slot = sync.entity_to_slot.get(e.to_bits())
                if tr_slot:
                    var prev = sync.tracked[tr_slot.value()]
                    var pos_d2 = (w_pos - prev.last_pos).length_squared()
                    var rot_dot = abs(
                        w_rot.x * prev.last_rot.x
                        + w_rot.y * prev.last_rot.y
                        + w_rot.z * prev.last_rot.z
                        + w_rot.w * prev.last_rot.w
                    )
                    if pos_d2 > Float32(1.0e-8) or rot_dot < Float32(0.99999):
                        physics.reset_body(slot, target_pos, target_rot, rb.linear_vel, rb.angular_vel, rb.active)
                        sync.record_body(e, rb.body_id, w_pos, w_rot)
                else:
                    sync.record_body(e, rb.body_id, w_pos, w_rot)
    if need_broadphase_rebuild:
        physics.rebuild_broadphase()

    # 3. Materialize & sync trigger ColliderComponents (both standalone and RB-attached)
    var col_mask = world.bit_for[ColliderComponent]()
    var trig_entities = world.query_entities(col_mask)
    for i in range(len(trig_entities)):
        var e = trig_entities[i]
        var col_opt = world.get_ptr_mut[ColliderComponent](e)
        if not col_opt:
            continue
        ref col = col_opt.value()[unsafe_offset=0]
        if not col.is_trigger:
            if col.trigger_id >= 0:
                for ti in range(len(physics.triggers)):
                    if physics.triggers[ti].id == col.trigger_id:
                        physics.triggers[ti].active = False
                        break
                col.trigger_id = -1
            continue
        var pose = _entity_world_pose(world, e)
        var t_pos = pose[0] + pose[1].rotate(col.local_offset)
        var t_rot = (pose[1] * col.local_rot).normalized()
        var filt = col.filter
        if filt == CollisionFilter():
            filt = CollisionFilter(CAT_SENSOR, CAT_ALL, 0)
        if col.trigger_id >= 0:
            var found_trig = False
            for ti in range(len(physics.triggers)):
                if physics.triggers[ti].id == col.trigger_id and physics.triggers[ti].active:
                    physics.triggers[ti].pos = t_pos
                    physics.triggers[ti].rot = t_rot
                    physics.triggers[ti].filter = filt
                    if col.shape == BODY_SHAPE_SPHERE:
                        physics.triggers[ti].shape = TRIGGER_SHAPE_SPHERE
                        physics.triggers[ti].radius = col.radius
                        physics.triggers[ti].half_extents = Vec3(col.radius, col.radius, col.radius)
                        physics.triggers[ti].half_height = Float32(0.0)
                    elif col.shape == BODY_SHAPE_CAPSULE:
                        physics.triggers[ti].shape = TRIGGER_SHAPE_CAPSULE
                        physics.triggers[ti].radius = col.radius
                        physics.triggers[ti].half_height = col.half_height
                        physics.triggers[ti].half_extents = Vec3(col.radius, col.half_height + col.radius, col.radius)
                    else:
                        physics.triggers[ti].shape = TRIGGER_SHAPE_BOX
                        physics.triggers[ti].half_extents = col.half_extents
                        physics.triggers[ti].radius = col.half_extents.length()
                        physics.triggers[ti].half_height = Float32(0.0)
                    found_trig = True
                    break
            if not found_trig:
                col.trigger_id = -1
        if col.trigger_id < 0:
            var tv: TriggerVolume
            if col.shape == BODY_SHAPE_SPHERE:
                tv = TriggerVolume.sphere(0, t_pos, col.radius, filt)
            elif col.shape == BODY_SHAPE_CAPSULE:
                tv = TriggerVolume.capsule(0, t_pos, col.radius, col.half_height, t_rot, filt)
            else:
                tv = TriggerVolume.obb(0, t_pos, col.half_extents, t_rot, filt)
            var tidx = physics.add_trigger(tv^)
            col.trigger_id = physics.triggers[tidx].id
            sync.record_trigger(e, col.trigger_id)

    # 4. Materialize & rebind JointComponents
    var j_mask = world.bit_for[JointComponent]()
    var j_entities = world.query_entities(j_mask)
    for i in range(len(j_entities)):
        var e = j_entities[i]
        var jc_opt = world.get_ptr_mut[JointComponent](e)
        if not jc_opt:
            continue
        ref jc = jc_opt.value()[unsafe_offset=0]

        var rb_a_opt = world.get_ptr[RigidBodyComponent](e)
        var bid_a = rb_a_opt.value()[unsafe_offset=0].body_id if rb_a_opt else -1
        var slot_a = physics.find_slot_by_id(bid_a) if bid_a >= 0 else -1

        var slot_b = -1
        var bid_b = -1
        var has_target_b = not jc.entity_b.is_null()
        if has_target_b and world.is_alive(jc.entity_b):
            var rb_b_opt = world.get_ptr[RigidBodyComponent](jc.entity_b)
            if rb_b_opt:
                bid_b = rb_b_opt.value()[unsafe_offset=0].body_id
                slot_b = physics.find_slot_by_id(bid_b) if bid_b >= 0 else -1

        var valid_endpoints = (slot_a >= 0) and ((not has_target_b) or (slot_b >= 0))

        if jc.runtime_index >= 0:
            var stale = not valid_endpoints
            if not stale:
                var tj_opt = sync.tracked_joint(e)
                if tj_opt:
                    var tj = tj_opt.value()
                    if tj.is_six_dof != jc.is_six_dof or tj.body_id_a != bid_a or tj.body_id_b != bid_b:
                        stale = True
            if not stale:
                var exp_b = bid_b if slot_b >= 0 else bid_a
                if jc.is_six_dof:
                    var exp_a = bid_a if slot_b >= 0 else JOINT_WORLD
                    if jc.runtime_index >= len(physics.six_dof_joints):
                        stale = True
                    else:
                        ref j6 = physics.six_dof_joints[jc.runtime_index]
                        if j6.body_a != exp_a or j6.body_b != exp_b:
                            stale = True
                else:
                    var has_dyn_or_kin_a = (
                        (not physics.bodies[slot_a].is_static)
                        or (physics.bodies[slot_a].motion_mode == BODY_MOTION_KINEMATIC)
                    )
                    var exp_a = bid_a if (slot_b >= 0 and has_dyn_or_kin_a) else JOINT_WORLD
                    if jc.runtime_index >= len(physics.joints):
                        stale = True
                    else:
                        ref j = physics.joints[jc.runtime_index]
                        if j.body_a != exp_a or j.body_b != exp_b:
                            stale = True
            if stale:
                if jc.is_six_dof:
                    if jc.runtime_index < len(physics.six_dof_joints):
                        physics.six_dof_joints[jc.runtime_index].broken = True
                else:
                    if jc.runtime_index < len(physics.joints):
                        physics.joints[jc.runtime_index].broken = True
                jc.runtime_index = -1

        if jc.runtime_index >= 0 or not valid_endpoints:
            continue
        if jc.is_six_dof:
            var j6 = SixDofMotorJoint()
            if slot_b >= 0:
                j6.body_a = bid_a
                j6.body_b = bid_b
            else:
                j6.body_a = JOINT_WORLD
                j6.body_b = bid_a
            j6.local_anchor_a = jc.anchor_a
            j6.local_anchor_b = jc.anchor_b
            j6.break_force = jc.break_force
            j6.break_torque = jc.break_torque
            jc.runtime_index = physics.add_six_dof_joint(j6^)
        else:
            var jdef: JointDef
            if slot_b >= 0:
                jdef = JointDef(jc.joint_kind, slot_a, slot_b, jc.anchor_a, jc.axis)
                jdef.second_anchor(jc.anchor_b)
            else:
                jdef = JointDef(jc.joint_kind, -1, slot_a, jc.anchor_a, jc.axis)
            jdef.lower = jc.lower
            jdef.upper = jc.upper
            jdef.cone = jc.cone
            jdef.length = jc.length
            jdef.enable_limit = jc.enable_limit
            jdef.enable_motor = jc.enable_motor
            jdef.enable_spring = jc.enable_spring
            jdef.motor_speed = jc.motor_speed
            jdef.max_motor = jc.max_motor
            jdef.spring_hertz = jc.spring_hertz
            jdef.spring_damping = jc.spring_damping
            jdef.break_force = jc.break_force
            jdef.break_torque = jc.break_torque
            jdef.hertz = jc.hertz
            jdef.damping = jc.damping
            jc.runtime_index = physics.add_joint(jdef)
        if jc.runtime_index >= 0:
            sync.record_joint(e, jc.runtime_index, jc.is_six_dof, bid_a, bid_b)

    # 5. Materialize VehicleComponents
    var veh_mask = world.bit_for[VehicleComponent]()
    var veh_entities = world.query_entities(veh_mask)
    for i in range(len(veh_entities)):
        var e = veh_entities[i]
        var vc_opt = world.get_ptr_mut[VehicleComponent](e)
        if not vc_opt:
            continue
        ref vc = vc_opt.value()[unsafe_offset=0]
        if (
            vc.vehicle_slot < 0
            or vc.vehicle_slot >= len(physics.vehicles)
            or (not physics.vehicles[vc.vehicle_slot].active)
            or physics.vehicle_body_slot(vc.vehicle_slot) < 0
        ):
            var pose = _entity_world_pose(world, e)
            var ch = vc.chassis.copy()
            if world.has[LocalTransform](e):
                ch.pos = pose[0]
                ch.rot = pose[1]
            vc.vehicle_slot = physics.add_vehicle(
                ch^,
                vc.chassis_half_extents,
                vc.wheel_filter,
                vc.body_filter,
                vc.friction,
                vc.restitution,
            )
            var bid = physics.vehicle_body_id(vc.vehicle_slot)
            if bid >= 0:
                sync.body_to_entity[bid] = e.to_bits()
            if vc.vehicle_slot >= 0:
                sync.record_vehicle(e, vc.vehicle_slot, bid)

    # 6. Materialize Aero6DofComponents
    var aero_mask = world.bit_for[Aero6DofComponent]()
    var aero_entities = world.query_entities(aero_mask)
    for i in range(len(aero_entities)):
        var e = aero_entities[i]
        var ac_opt = world.get_ptr_mut[Aero6DofComponent](e)
        if not ac_opt:
            continue
        ref ac = ac_opt.value()[unsafe_offset=0]
        if (
            ac.aero_slot < 0
            or ac.aero_slot >= len(physics.aero_bodies)
            or (not physics.aero_bodies[ac.aero_slot].active)
            or physics.aero_body_slot(ac.aero_slot) < 0
        ):
            if world.has[LocalTransform](e):
                var pose = _entity_world_pose(world, e)
                ac.state.pos = pose[0]
                ac.state.rot = pose[1]
            ac.aero_slot = physics.add_aero_body(
                ac.state,
                ac.config,
                ac.controls,
                ac.half_extents,
                filter=ac.filter,
                enable_ccd=ac.enable_ccd,
            )
            var bid = -1
            if ac.aero_slot >= 0 and ac.aero_slot < len(physics.aero_bodies):
                bid = physics.aero_bodies[ac.aero_slot].body_id
                sync.body_to_entity[bid] = e.to_bits()
            if ac.aero_slot >= 0:
                sync.record_aero(e, ac.aero_slot, bid)

    # 7. Materialize ClothMeshComponents
    var cloth_mask = world.bit_for[ClothMeshComponent]()
    var cloth_entities = world.query_entities(cloth_mask)
    for i in range(len(cloth_entities)):
        var e = cloth_entities[i]
        var cc_opt = world.get_ptr_mut[ClothMeshComponent](e)
        if not cc_opt:
            continue
        ref cc = cc_opt.value()[unsafe_offset=0]
        if (
            cc.cloth_slot < 0
            or cc.cloth_slot >= len(physics.xpbd_meshes)
            or (not physics.xpbd_meshes[cc.cloth_slot].active)
        ):
            cc.cloth_slot = physics.add_cloth_mesh(
                cc.mesh.copy(),
                cc.attachments,
                cc.substeps,
                cc.iterations,
                cc.filter,
            )
            if cc.cloth_slot >= 0:
                sync.record_cloth(e, cc.cloth_slot)


def _sys_physics_fixed_step(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var phys_opt = resources.get_ptr_mut[PhysicsWorld]()
    if not phys_opt:
        return
    ref physics = phys_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt:
        dt = fix_opt.value()[unsafe_offset=0].dt
    if dt <= Float32(1.0e-7):
        return

    # 1. Step CharacterControllers
    var kcc_mask = world.bit_for[CharacterControllerComponent]()
    var kcc_entities = world.query_entities(kcc_mask)
    for i in range(len(kcc_entities)):
        var e = kcc_entities[i]
        var kc_opt = world.get_ptr_mut[CharacterControllerComponent](e)
        if not kc_opt:
            continue
        ref kc = kc_opt.value()[unsafe_offset=0]
        if not kc.active:
            continue
        var new_pos = kc.controller.move_and_slide(physics, kc.desired_vel, dt, kc.jump_speed)
        kc.jump_speed = Float32(0.0)
        var rot = Quat()
        var lt_opt = world.get_ptr[LocalTransform](e)
        if lt_opt:
            rot = lt_opt.value()[unsafe_offset=0].rotation
        _write_world_pose_to_local(world, e, new_pos, rot)

    # 2. Push controls for Vehicles, Aero6Dof, and Cloth wind
    var veh_mask = world.bit_for[VehicleComponent]()
    var veh_entities = world.query_entities(veh_mask)
    for i in range(len(veh_entities)):
        var vc_opt = world.get_ptr[VehicleComponent](veh_entities[i])
        if vc_opt and vc_opt.value()[unsafe_offset=0].vehicle_slot >= 0:
            ref vc = vc_opt.value()[unsafe_offset=0]
            physics.set_vehicle_controls(vc.vehicle_slot, vc.throttle, vc.brake, vc.steer)

    var aero_mask = world.bit_for[Aero6DofComponent]()
    var aero_entities = world.query_entities(aero_mask)
    for i in range(len(aero_entities)):
        var ac_opt = world.get_ptr[Aero6DofComponent](aero_entities[i])
        if ac_opt and ac_opt.value()[unsafe_offset=0].aero_slot >= 0:
            ref ac = ac_opt.value()[unsafe_offset=0]
            physics.set_aero_config(ac.aero_slot, ac.config)
            physics.set_aero_controls(ac.aero_slot, ac.controls)

    var cloth_mask = world.bit_for[ClothMeshComponent]()
    var cloth_entities = world.query_entities(cloth_mask)
    for i in range(len(cloth_entities)):
        var cc_opt = world.get_ptr[ClothMeshComponent](cloth_entities[i])
        if cc_opt and cc_opt.value()[unsafe_offset=0].cloth_slot >= 0:
            ref cc = cc_opt.value()[unsafe_offset=0]
            var n_p = cc.mesh.count()
            if n_p > 0:
                var wind_list = List[Vec3](length=n_p, fill=cc.wind)
                physics.set_xpbd_mesh_wind(cc.cloth_slot, wind_list)

    # 3. Step PhysicsWorld (with JobPool acceleration)
    physics.step(dt, Int(pool_ptr))

    # 4. Sync solved poses & velocities back to ECS components & LocalTransform
    var sync_opt = resources.get_ptr_mut[PhysicsSyncStateResource]()

    var rb_mask = world.bit_for[RigidBodyComponent]()
    var rb_entities = world.query_entities(rb_mask)
    for i in range(len(rb_entities)):
        var e = rb_entities[i]
        var rb_opt = world.get_ptr_mut[RigidBodyComponent](e)
        if not rb_opt:
            continue
        ref rb = rb_opt.value()[unsafe_offset=0]
        if rb.body_id < 0:
            continue
        var slot = physics.find_slot_by_id(rb.body_id)
        if slot < 0:
            continue
        ref b = physics.bodies[slot]
        rb.linear_vel = b.linear_vel
        rb.angular_vel = b.angular_vel
        if rb.motion_mode == BODY_MOTION_DYNAMIC or rb.motion_mode == BODY_MOTION_KINEMATIC:
            var col_off = Vec3()
            var col_rot = Quat()
            var col_opt = world.get_ptr[ColliderComponent](e)
            if col_opt:
                col_off = col_opt.value()[unsafe_offset=0].local_offset
                col_rot = col_opt.value()[unsafe_offset=0].local_rot
            var ent_rot = (b.rot * col_rot.conjugate()).normalized()
            var ent_pos = b.pos - ent_rot.rotate(col_off)
            _write_world_pose_to_local(world, e, ent_pos, ent_rot)
            if sync_opt:
                sync_opt.value()[unsafe_offset=0].record_body(e, rb.body_id, ent_pos, ent_rot)

    for i in range(len(veh_entities)):
        var e = veh_entities[i]
        var vc_opt = world.get_ptr_mut[VehicleComponent](e)
        if vc_opt and vc_opt.value()[unsafe_offset=0].vehicle_slot >= 0:
            ref vc = vc_opt.value()[unsafe_offset=0]
            vc.chassis = physics.vehicle_state(vc.vehicle_slot)
            _write_world_pose_to_local(world, e, vc.chassis.pos, vc.chassis.rot)

    for i in range(len(aero_entities)):
        var e = aero_entities[i]
        var ac_opt = world.get_ptr_mut[Aero6DofComponent](e)
        if ac_opt and ac_opt.value()[unsafe_offset=0].aero_slot >= 0:
            ref ac = ac_opt.value()[unsafe_offset=0]
            ac.state = physics.aero_state(ac.aero_slot)
            _write_world_pose_to_local(world, e, ac.state.pos, ac.state.rot)

    for i in range(len(cloth_entities)):
        var e = cloth_entities[i]
        var cc_opt = world.get_ptr_mut[ClothMeshComponent](e)
        if cc_opt and cc_opt.value()[unsafe_offset=0].cloth_slot >= 0:
            ref cc = cc_opt.value()[unsafe_offset=0]
            cc.mesh = physics.cloth_mesh(cc.cloth_slot)

    # 5. Drain PhysicsEvents & TriggerEvents into ECS Events buffers
    var pev_opt = resources.get_ptr_mut[PhysicsEventsResource]()
    if pev_opt:
        ref pev = pev_opt.value()[unsafe_offset=0].events
        for ei in range(len(physics.events)):
            _ = pev.send(physics.events[ei])

    var tev_opt = resources.get_ptr_mut[TriggerEventsResource]()
    if tev_opt:
        ref tev = tev_opt.value()[unsafe_offset=0].events
        for ti in range(len(physics.trigger_events)):
            _ = tev.send(physics.trigger_events[ti])


def _swap_physics_events(res_ptr: Pointer[Resources, MutAnyOrigin]):
    ref resources = res_ptr[unsafe_offset=0]
    var pev_opt = resources.get_ptr_mut[PhysicsEventsResource]()
    if pev_opt:
        pev_opt.value()[unsafe_offset=0].events.swap_buffers()
    var tev_opt = resources.get_ptr_mut[TriggerEventsResource]()
    if tev_opt:
        tev_opt.value()[unsafe_offset=0].events.swap_buffers()


def _has_physics_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct PhysicsPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `PhysicsWorld`, 7 ECS physics components, and fixed-step systems in `EngineApp`."""
    var gravity: Float32

    def __init__(out self, gravity: Float32 = 9.81):
        self.gravity = gravity

    def build(self, mut app: EngineApp):
        register_physics_components(app.world)
        if not app.resources.has[PhysicsWorld]():
            app.insert_resource[PhysicsWorld](PhysicsWorld(gravity=self.gravity))
        if not app.resources.has[PhysicsEventsResource]():
            app.insert_resource[PhysicsEventsResource](PhysicsEventsResource())
        if not app.resources.has[TriggerEventsResource]():
            app.insert_resource[TriggerEventsResource](TriggerEventsResource())
        if not app.resources.has[PhysicsSyncStateResource]():
            app.insert_resource[PhysicsSyncStateResource](PhysicsSyncStateResource())

        if _has_physics_system(app, STAGE_PRE_UPDATE, "flint.physics.pre_sync"):
            return

        app.add_event_swap_hook(_swap_physics_events)

        var b_lt = app.world.bit_for[LocalTransform]()
        var b_rb = app.world.bit_for[RigidBodyComponent]()
        var b_col = app.world.bit_for[ColliderComponent]()
        var b_jnt = app.world.bit_for[JointComponent]()
        var b_kcc = app.world.bit_for[CharacterControllerComponent]()
        var b_veh = app.world.bit_for[VehicleComponent]()
        var b_aero = app.world.bit_for[Aero6DofComponent]()
        var b_cloth = app.world.bit_for[ClothMeshComponent]()
        var all_phys = (
            b_lt.union(b_rb).union(b_col).union(b_jnt)
            .union(b_kcc).union(b_veh).union(b_aero).union(b_cloth)
        )

        _ = app.add_system(
            STAGE_PRE_UPDATE,
            "flint.physics.pre_sync",
            _sys_physics_pre_update,
            reads=all_phys,
            writes=all_phys,
            exclusive=True,
        )
        _ = app.add_system(
            STAGE_FIXED_UPDATE,
            "flint.physics.fixed_step",
            _sys_physics_fixed_step,
            reads=all_phys,
            writes=all_phys,
            exclusive=True,
        )
