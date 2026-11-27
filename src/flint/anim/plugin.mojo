"""Flint v5 Animation Plugin & ECS Component Schemas (`flint.anim.plugin`).

Provides:
- `AnimAssetCache`: `Resource` storing `NativeModel` and `AnimGraphDef` + `ClipInfo`
  assets keyed by `UInt64` IDs for runtime ECS animation evaluation.
- `AnimEventsResource`: `Resource` wrapping `Events[AnimEventRecord]` swapped via
  `EngineApp.add_event_swap_hook`.
- 4 `SerializableComponent` ECS schemas with explicit `world.set_component_binary_hooks`
  (zero Clone Vault dependency):
  `SkeletonComponent`, `AnimatorComponent`, `RagdollComponent`, `BoneSocketComponent`.
- `register_anim_components(mut world: World)` & `register_anim_codecs(mut reg: ComponentRegistry)`.
- `AnimPlugin`: samples controller root motion before each fixed physics step. It
  schedules pose evaluation, ragdoll writeback, foot planting, and
  `BoneSocketAttachment.sync_to_physics` in `STAGE_UPDATE` before `TransformPlugin`
  propagates `GlobalTransform` in `STAGE_POST_UPDATE`.
"""
from std.collections import List
from std.math import abs, isfinite, max, min
from flint.anim.animator import Animator
from flint.anim.flanim_ecs import (
    BoneSocketAttachment,
    RAGDOLL_SM_ANIMATED,
    RagdollStateMachine,
    solve_foot_plant_world,
)
from flint.anim.graph import AnimGraphDef, Condition
from flint.anim.pose import Pose, SkeletonInstance, sample_pose
from flint.anim.procedural import DEG, FootPlantSolver, Placement
from flint.anim.sync import ClipEventInfo, ClipInfo, FiredEvent
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.assets.asset_ref import AssetRef
from flint.assets.json import JsonDocument
from flint.assets.json_write import json_quote
from flint.assets.model import NativeModel
from flint.core.detmath import det_atan2
from flint.core.ik import angle_axis
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import (
    Entity,
    Events,
    NULL_ENTITY,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.physics.plugin import CharacterControllerComponent, CharacterRootMotionSystem, PhysicsSyncStateResource
from flint.physics.world import PhysicsWorld
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    Parent,
    SerializableComponent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@always_inline
def _f32_s(v: Float32) -> String:
    return String(Float64(v))


def _vec3_json(v: Vec3) -> String:
    return "[" + _f32_s(v.x) + ", " + _f32_s(v.y) + ", " + _f32_s(v.z) + "]"


def _quat_json(q: Quat) -> String:
    return "[" + _f32_s(q.x) + ", " + _f32_s(q.y) + ", " + _f32_s(q.z) + ", " + _f32_s(q.w) + "]"


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


@always_inline
def _quat_yaw_deg(q: Quat) -> Float32:
    var qn = q.normalized()
    var fwd = qn.rotate(Vec3(0.0, 0.0, 1.0))
    return det_atan2(fwd.x, fwd.z) / DEG


def _anim_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _anim_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
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


def _register_non_trivial_anim[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
    if not is_triv and tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _anim_bin_write[C], _anim_bin_read[C]
        )


# ---------------------------------------------------------------------------
# Resources & Event Buffers
# ---------------------------------------------------------------------------

struct AnimAssetCache(Resource, Movable):
    """Runtime cache mapping `model_id` / `graph_id` to `NativeModel` & `AnimGraphDef` assets."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.anim.AnimAssetCache")
    var model_ids: List[UInt64]
    var models: List[NativeModel]
    var graph_ids: List[UInt64]
    var graphs: List[AnimGraphDef]
    var graph_clips: List[List[ClipInfo]]
    var model_versions: List[Int]
    var graph_versions: List[Int]

    def __init__(out self):
        self.model_ids = List[UInt64]()
        self.models = List[NativeModel]()
        self.graph_ids = List[UInt64]()
        self.graphs = List[AnimGraphDef]()
        self.graph_clips = List[List[ClipInfo]]()
        self.model_versions = List[Int]()
        self.graph_versions = List[Int]()

    def register_model(mut self, model_id: UInt64, var model: NativeModel):
        var idx = self.find_model(model_id)
        if idx >= 0:
            self.models[idx] = model^
            self.model_versions[idx] += 1
            return
        self.model_ids.append(model_id)
        self.models.append(model^)
        self.model_versions.append(1)

    def find_model(self, model_id: UInt64) -> Int:
        for i in range(len(self.model_ids)):
            if self.model_ids[i] == model_id:
                return i
        return -1

    def register_graph(mut self, graph_id: UInt64, var graph: AnimGraphDef, var clips: List[ClipInfo]):
        var idx = self.find_graph(graph_id)
        if idx >= 0:
            self.graphs[idx] = graph^
            self.graph_clips[idx] = clips^
            self.graph_versions[idx] += 1
            return
        self.graph_ids.append(graph_id)
        self.graphs.append(graph^)
        self.graph_clips.append(clips^)
        self.graph_versions.append(1)

    def find_graph(self, graph_id: UInt64) -> Int:
        for i in range(len(self.graph_ids)):
            if self.graph_ids[i] == graph_id:
                return i
        return -1


@fieldwise_init
struct AnimEventRecord(Copyable, Movable):
    """Animation clip event tagged with the emitting ECS `Entity`."""
    var entity: Entity
    var event: FiredEvent


struct AnimEventsResource(Resource, Movable):
    """Per-frame `Events[AnimEventRecord]` emitted by `AnimatorComponent`s."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.anim.AnimEventsResource")
    var events: Events[AnimEventRecord]

    def __init__(out self):
        self.events = Events[AnimEventRecord]()


struct _TrackedRagdollSync(Copyable, ImplicitlyCopyable, Movable):
    var entity_bits: UInt64
    var slot: Int

    def __init__(out self, entity_bits: UInt64 = 0, slot: Int = -1):
        self.entity_bits = entity_bits
        self.slot = slot


struct AnimSyncStateResource(Resource, Movable):
    """Tracks active ECS entity <-> `PhysicsWorld` ragdoll slot bindings for cleanup."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.anim.AnimSyncStateResource")
    var tracked_ragdolls: List[_TrackedRagdollSync]

    def __init__(out self):
        self.tracked_ragdolls = List[_TrackedRagdollSync]()

    def record_ragdoll(mut self, entity: Entity, slot: Int):
        var bits = entity.to_bits()
        for i in range(len(self.tracked_ragdolls)):
            if self.tracked_ragdolls[i].entity_bits == bits:
                self.tracked_ragdolls[i].slot = slot
                return
        self.tracked_ragdolls.append(_TrackedRagdollSync(bits, slot))


# ---------------------------------------------------------------------------
# 4 Serializable ECS Animation Components
# ---------------------------------------------------------------------------

struct SkeletonComponent(SerializableComponent):
    """Per-entity decoupled `SkeletonInstance` referencing a `NativeModel` in `AnimAssetCache`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.anim.SkeletonComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "SkeletonComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ASKL")
    comptime SCHEMA_VERSION: Int = 3

    var model_id: UInt64
    var skeleton: SkeletonInstance
    var dirty_topology: Bool
    var rig_ref: AssetRef

    def __init__(
        out self,
        model_id: UInt64 = 0,
        var skeleton: SkeletonInstance = SkeletonInstance(),
        dirty_topology: Bool = True,
        var rig_ref: AssetRef = AssetRef(),
    ):
        self.model_id = model_id
        self.skeleton = skeleton^
        self.dirty_topology = dirty_topology
        self.rig_ref = rig_ref^

    def write_bin(self, mut out: BlobWriter):
        out.u64(self.model_id)
        out.boolean(self.dirty_topology)
        var n = self.skeleton.pose.count()
        out.integer(n)
        for i in range(n):
            out.vec3(self.skeleton.pose.t[i])
            out.quat(self.skeleton.pose.r[i])
            out.vec3(self.skeleton.pose.s[i])
        self.rig_ref.write_bin(out)
        var par_n = min(n, len(self.skeleton.parents))
        for i in range(n):
            out.integer(self.skeleton.parents[i] if i < par_n else -1)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var mid = reader.u64()
        var dirty = reader.boolean()
        var n = reader.integer()
        if reader.failed or n < 0 or n > 1 << 16 or n > max(0, reader.limit() - reader.offset) // 40:
            raise Error("SkeletonComponent: invalid pose node count")
        var skel = SkeletonInstance(n)
        for i in range(n):
            skel.pose.t[i] = reader.vec3()
            skel.pose.r[i] = reader.quat()
            skel.pose.s[i] = reader.vec3()
        var rref = AssetRef.read_bin(reader, 1) if version >= 2 else AssetRef()
        if version >= 3:
            for i in range(n):
                skel.parents[i] = reader.integer()
        if reader.failed:
            raise Error("SkeletonComponent: truncated blob")
        return Self(mid, skel^, dirty, rref^)

    def write_json(self) -> String:
        var nc = self.skeleton.node_count
        var par_n = min(nc, len(self.skeleton.parents))
        var par_str = String("[")
        for i in range(nc):
            if i > 0:
                par_str += ", "
            var p = self.skeleton.parents[i] if i < par_n else -1
            par_str += String(p)
        par_str += "]"
        var pose_n = self.skeleton.pose.count()
        var pose_str = String("[")
        for i in range(pose_n):
            if i > 0:
                pose_str += ", "
            pose_str += (
                "{\"t\": " + _vec3_json(self.skeleton.pose.t[i])
                + ", \"r\": " + _quat_json(self.skeleton.pose.r[i])
                + ", \"s\": " + _vec3_json(self.skeleton.pose.s[i])
                + "}"
            )
        pose_str += "]"
        return (
            "{\"model_id\": " + String(self.model_id)
            + ", \"node_count\": " + String(nc)
            + ", \"dirty_topology\": " + ("true" if self.dirty_topology else "false")
            + ", \"rig_ref\": " + self.rig_ref.write_json()
            + ", \"parents\": " + par_str
            + ", \"pose\": " + pose_str
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var mid = doc.u64(doc.object_get(node, "model_id"), 0)
        var nc_node = doc.object_get(node, "node_count")
        var pnode = doc.object_get(node, "pose")
        var parnode = doc.object_get(node, "parents")
        var pose_len = doc.array_size(pnode) if pnode >= 0 else 0
        var nc = max(0, doc.integer(nc_node, pose_len if nc_node < 0 else 0))
        if nc > 1 << 16:
            raise Error("SkeletonComponent: invalid node_count")
        var dirty = doc.boolean(doc.object_get(node, "dirty_topology"), True)
        var rnode = doc.object_get(node, "rig_ref")
        var rref = AssetRef.read_json(doc, rnode) if rnode >= 0 else AssetRef()
        var skel = SkeletonInstance(nc)
        if parnode >= 0 and doc.array_size(parnode) == nc:
            for i in range(nc):
                skel.parents[i] = doc.integer(doc.at(parnode, i), -1)
        if pnode >= 0 and doc.array_size(pnode) == nc:
            for i in range(nc):
                var jnode = doc.at(pnode, i)
                var tn = doc.object_get(jnode, "t")
                var rn = doc.object_get(jnode, "r")
                var sn = doc.object_get(jnode, "s")
                if tn >= 0:
                    skel.pose.t[i] = _read_vec3_json(doc, tn, Vec3())
                if rn >= 0:
                    skel.pose.r[i] = _read_quat_json(doc, rn)
                if sn >= 0:
                    skel.pose.s[i] = _read_vec3_json(doc, sn, Vec3(1.0, 1.0, 1.0))
        return Self(mid, skel^, dirty, rref^)


struct AnimatorComponent(SerializableComponent):
    """Per-entity `Animator` state machine + optional `FootPlantSolver` + direct-clip playback."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.anim.AnimatorComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "AnimatorComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("AANM")
    comptime SCHEMA_VERSION: Int = 4

    var graph_id: UInt64
    # Derived cache keys are rebuilt on scene load. Shared graphs can bind to
    # different GLB clip orders without changing the source asset.
    var graph_binding_error: String
    var restore_graph_clocks: Bool
    var bound_graph_slot: Int
    var bound_model_slot: Int
    var bound_graph_version: Int
    var bound_model_version: Int
    var animator: Animator
    var apply_root_motion: Bool
    # Collision-resolved world displacement from the last controller tick.
    var actual_displacement: Vec3
    var bound: Bool
    var foot_plant: FootPlantSolver
    var foot_in_stance: List[Bool]
    var enable_foot_plant: Bool
    var ignore_body_id: Int
    var graph_ref: AssetRef
    var clip_index: Int
    var clip_name: String
    var playing: Bool
    var loop: Bool
    var time: Float32
    var speed: Float32

    def __init__(
        out self,
        graph_id: UInt64 = 0,
        apply_root_motion: Bool = False,
        enable_foot_plant: Bool = False,
        ignore_body_id: Int = -1,
        var graph_ref: AssetRef = AssetRef(),
        clip_index: Int = -1,
        clip_name: String = "",
        playing: Bool = True,
        loop: Bool = True,
        time: Float32 = 0.0,
        speed: Float32 = 1.0,
    ):
        self.graph_id = graph_id
        self.graph_binding_error = String()
        self.restore_graph_clocks = False
        self.bound_graph_slot = -1
        self.bound_model_slot = -1
        self.bound_graph_version = 0
        self.bound_model_version = 0
        self.animator = Animator()
        self.apply_root_motion = apply_root_motion
        self.actual_displacement = Vec3()
        self.bound = False
        self.foot_plant = FootPlantSolver()
        self.foot_in_stance = List[Bool]()
        self.enable_foot_plant = enable_foot_plant
        self.ignore_body_id = ignore_body_id
        self.graph_ref = graph_ref^
        self.clip_index = clip_index
        self.clip_name = clip_name
        self.playing = playing
        self.loop = loop
        self.time = time
        self.speed = speed

    def write_bin(self, mut out: BlobWriter):
        out.u64(self.graph_id)
        out.boolean(self.apply_root_motion)
        out.boolean(self.bound)
        out.boolean(self.enable_foot_plant)
        out.integer(self.ignore_body_id)
        out.integer(self.animator.state)
        out.integer(self.animator.previous)
        out.f32(self.animator.state_time)
        out.f32(self.animator.clip_time)
        out.f32(self.animator.prev_clip_time)
        out.f32(self.animator.phase)
        out.f32(self.animator.prev_phase)
        out.integer(len(self.animator.params))
        for i in range(len(self.animator.params)):
            out.f32(self.animator.params[i])
        out.integer(self.foot_plant.pelvis)
        out.integer(len(self.foot_plant.legs))
        for i in range(len(self.foot_plant.legs)):
            ref lg = self.foot_plant.legs[i]
            out.integer(lg.hip)
            out.integer(lg.knee)
            out.integer(lg.ankle)
            out.f32(lg.ankle_height)
            out.vec3(lg.pole_local)
        out.integer(len(self.foot_in_stance))
        for i in range(len(self.foot_in_stance)):
            out.boolean(self.foot_in_stance[i])
        self.graph_ref.write_bin(out)
        out.integer(self.clip_index)
        out.text(self.clip_name)
        out.boolean(self.playing)
        out.boolean(self.loop)
        out.f32(self.time)
        out.f32(self.speed)
        out.integer(self.animator.manual_state)
        out.vec3(self.actual_displacement)
        out.f32(self.foot_plant.pelvis_offset_y.value)
        out.f32(self.foot_plant.pelvis_offset_y.velocity)
        for leg in self.foot_plant.legs:
            out.boolean(leg.locked)
            out.vec3(leg.locked_world_pos)
            out.integer(leg.support_body_id)
            out.vec3(leg.support_local_point)
            out.vec3(leg.release_offset)
            out.vec3(leg.release_vel)
            out.quat(leg.smoothed_tilt)
            out.f32(leg.stance_blend.value)
            out.f32(leg.stance_blend.velocity)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var gid = reader.u64()
        var rm = reader.boolean()
        var bnd = reader.boolean()
        var efp = reader.boolean()
        var ign = reader.integer()
        var out = Self(gid, rm, efp, ign)
        out.bound = bnd
        out.restore_graph_clocks = bnd
        out.animator.state = reader.integer()
        out.animator.previous = reader.integer()
        out.animator.state_time = reader.f32()
        out.animator.clip_time = reader.f32()
        out.animator.prev_clip_time = reader.f32()
        out.animator.phase = reader.f32()
        out.animator.prev_phase = reader.f32()
        var n_p = reader.integer()
        if reader.failed or n_p < 0 or n_p > 4096 or n_p > max(0, reader.limit() - reader.offset) // 4:
            raise Error("AnimatorComponent: invalid param count")
        for _ in range(n_p):
            out.animator.params.append(reader.f32())
        out.foot_plant.pelvis = reader.integer()
        var n_legs = reader.integer()
        if reader.failed or n_legs < 0 or n_legs > 64 or n_legs > max(0, reader.limit() - reader.offset) // 40:
            raise Error("AnimatorComponent: invalid foot_plant leg count")
        for _ in range(n_legs):
            var hip = reader.integer()
            var knee = reader.integer()
            var ankle = reader.integer()
            var ah = reader.f32()
            var pole = reader.vec3()
            out.foot_plant.add_leg(hip, knee, ankle, ah, pole)
        var n_st = reader.integer()
        if reader.failed or n_st < 0 or n_st > 64 or n_st > max(0, reader.limit() - reader.offset):
            raise Error("AnimatorComponent: invalid stance count")
        for _ in range(n_st):
            out.foot_in_stance.append(reader.boolean())
        if version >= 2:
            out.graph_ref = AssetRef.read_bin(reader, 1)
            out.clip_index = reader.integer()
            out.clip_name = reader.text()
            out.playing = reader.boolean()
            out.loop = reader.boolean()
            out.time = reader.f32()
            out.speed = reader.f32()
        if version >= 3:
            out.animator.manual_state = reader.integer()
        if version >= 4:
            out.actual_displacement = reader.vec3()
            out.foot_plant.pelvis_offset_y.value = reader.f32()
            out.foot_plant.pelvis_offset_y.velocity = reader.f32()
            for i in range(n_legs):
                ref leg = out.foot_plant.legs[i]
                leg.locked = reader.boolean()
                leg.locked_world_pos = reader.vec3()
                leg.support_body_id = reader.integer()
                leg.support_local_point = reader.vec3()
                leg.release_offset = reader.vec3()
                leg.release_vel = reader.vec3()
                leg.smoothed_tilt = reader.quat()
                leg.stance_blend.value = reader.f32()
                leg.stance_blend.velocity = reader.f32()
        if reader.failed:
            raise Error("AnimatorComponent: truncated blob")
        return out^

    def write_json(self) -> String:
        var params_str = String("[")
        for i in range(len(self.animator.params)):
            if i > 0:
                params_str += ", "
            params_str += _f32_s(self.animator.params[i])
        params_str += "]"

        var legs_str = String("[")
        for i in range(len(self.foot_plant.legs)):
            if i > 0:
                legs_str += ", "
            ref lg = self.foot_plant.legs[i]
            legs_str += (
                "{\"hip\": " + String(lg.hip)
                + ", \"knee\": " + String(lg.knee)
                + ", \"ankle\": " + String(lg.ankle)
                + ", \"ankle_height\": " + _f32_s(lg.ankle_height)
                + ", \"pole_local\": " + _vec3_json(lg.pole_local)
                + "}"
            )
        legs_str += "]"

        var stance_str = String("[")
        for i in range(len(self.foot_in_stance)):
            if i > 0:
                stance_str += ", "
            stance_str += ("true" if self.foot_in_stance[i] else "false")
        stance_str += "]"

        return (
            "{\"graph_id\": " + String(self.graph_id)
            + ", \"apply_root_motion\": " + ("true" if self.apply_root_motion else "false")
            + ", \"bound\": " + ("true" if self.bound else "false")
            + ", \"enable_foot_plant\": " + ("true" if self.enable_foot_plant else "false")
            + ", \"ignore_body_id\": " + String(self.ignore_body_id)
            + ", \"state\": " + String(self.animator.state)
            + ", \"previous\": " + String(self.animator.previous)
            + ", \"state_time\": " + _f32_s(self.animator.state_time)
            + ", \"clip_time\": " + _f32_s(self.animator.clip_time)
            + ", \"prev_clip_time\": " + _f32_s(self.animator.prev_clip_time)
            + ", \"phase\": " + _f32_s(self.animator.phase)
            + ", \"prev_phase\": " + _f32_s(self.animator.prev_phase)
            + ", \"params\": " + params_str
            + ", \"foot_plant_pelvis\": " + String(self.foot_plant.pelvis)
            + ", \"foot_plant_legs\": " + legs_str
            + ", \"foot_in_stance\": " + stance_str
            + ", \"graph_ref\": " + self.graph_ref.write_json()
            + ", \"clip_index\": " + String(self.clip_index)
            + ", \"clip_name\": " + json_quote(self.clip_name)
            + ", \"playing\": " + ("true" if self.playing else "false")
            + ", \"loop\": " + ("true" if self.loop else "false")
            + ", \"time\": " + _f32_s(self.time)
            + ", \"speed\": " + _f32_s(self.speed)
            + ", \"manual_state\": " + String(self.animator.manual_state)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var gid = doc.u64(doc.object_get(node, "graph_id"), 0)
        var rm = doc.boolean(doc.object_get(node, "apply_root_motion"), False)
        var efp = doc.boolean(doc.object_get(node, "enable_foot_plant"), False)
        var ign = doc.integer(doc.object_get(node, "ignore_body_id"), -1)
        var gnode = doc.object_get(node, "graph_ref")
        var gref = AssetRef.read_json(doc, gnode) if gnode >= 0 else AssetRef()
        var ci = doc.integer(doc.object_get(node, "clip_index"), -1)
        var cn = doc.text(doc.object_get(node, "clip_name"), "")
        var pl = doc.boolean(doc.object_get(node, "playing"), True)
        var lp = doc.boolean(doc.object_get(node, "loop"), True)
        var tm = Float32(doc.number(doc.object_get(node, "time"), 0.0))
        var sp = Float32(doc.number(doc.object_get(node, "speed"), 1.0))
        var out = Self(gid, rm, efp, ign, gref^, ci, cn, pl, lp, tm, sp)
        out.bound = doc.boolean(doc.object_get(node, "bound"), False)
        out.animator.state = doc.integer(doc.object_get(node, "state"), -1)
        out.animator.manual_state = doc.integer(doc.object_get(node, "manual_state"), -1)
        out.animator.previous = doc.integer(doc.object_get(node, "previous"), -1)
        out.animator.state_time = Float32(doc.number(doc.object_get(node, "state_time"), 0.0))
        out.animator.clip_time = Float32(doc.number(doc.object_get(node, "clip_time"), 0.0))
        out.animator.prev_clip_time = Float32(doc.number(doc.object_get(node, "prev_clip_time"), 0.0))
        out.animator.phase = Float32(doc.number(doc.object_get(node, "phase"), 0.0))
        out.animator.prev_phase = Float32(doc.number(doc.object_get(node, "prev_phase"), 0.0))
        var params_node = doc.object_get(node, "params")
        if params_node >= 0:
            var n_p = min(4096, doc.array_size(params_node))
            for i in range(n_p):
                out.animator.params.append(Float32(doc.number(doc.at(params_node, i), 0.0)))
        out.foot_plant.pelvis = doc.integer(doc.object_get(node, "foot_plant_pelvis"), 0)
        var legs_node = doc.object_get(node, "foot_plant_legs")
        if legs_node >= 0:
            var n_legs = min(64, doc.array_size(legs_node))
            for i in range(n_legs):
                var ln = doc.at(legs_node, i)
                var hip = doc.integer(doc.object_get(ln, "hip"), -1)
                var knee = doc.integer(doc.object_get(ln, "knee"), -1)
                var ankle = doc.integer(doc.object_get(ln, "ankle"), -1)
                var ah = Float32(doc.number(doc.object_get(ln, "ankle_height"), 0.1))
                var pole = _read_vec3_json(doc, doc.object_get(ln, "pole_local"), Vec3(0.0, 0.5, 1.0))
                out.foot_plant.add_leg(hip, knee, ankle, ah, pole)
        var stance_node = doc.object_get(node, "foot_in_stance")
        if stance_node >= 0:
            var n_st = min(64, doc.array_size(stance_node))
            for i in range(n_st):
                out.foot_in_stance.append(doc.boolean(doc.at(stance_node, i), False))
        return out^


struct RagdollComponent(SerializableComponent):
    """Per-entity `RagdollStateMachine` coupled two-way with `PhysicsWorld`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.anim.RagdollComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "RagdollComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ARAG")
    comptime SCHEMA_VERSION: Int = 3

    var state_machine: RagdollStateMachine
    var auto_recover: Bool
    var pelvis_syncs_transform: Bool

    def __init__(
        out self,
        var state_machine: RagdollStateMachine = RagdollStateMachine(),
        auto_recover: Bool = True,
        pelvis_syncs_transform: Bool = True,
    ):
        self.state_machine = state_machine^
        self.auto_recover = auto_recover
        self.pelvis_syncs_transform = pelvis_syncs_transform

    def write_bin(self, mut out: BlobWriter):
        out.boolean(self.auto_recover)
        out.boolean(self.pelvis_syncs_transform)
        out.integer(self.state_machine.state)
        self.state_machine.ragdoll.serialize(out)
        out.integer(self.state_machine.supine_getup_clip)
        out.integer(self.state_machine.prone_getup_clip)
        out.f32(self.state_machine.getup_duration)
        out.f32(self.state_machine.getup_time)
        out.f32(self.state_machine.getup_halflife)
        out.f32(self.state_machine.active_pd_duration)
        out.f32(self.state_machine.active_pd_timer)
        out.f32(self.state_machine.min_limp_time)
        out.f32(self.state_machine.limp_timer)
        out.integer(self.state_machine.ignore_body_id)
        out.integer(self.state_machine.chest_limb_index)
        out.f32(self.state_machine.last_pelvis_snap_m)
        out.integer(self.state_machine.recovery_count)
        out.integer(self.state_machine.world_ragdoll_slot)
        out.f32(self.state_machine.query_padding)
        out.u32(self.state_machine.filter.category)
        out.u32(self.state_machine.filter.mask)
        out.integer(Int(self.state_machine.filter.group))
        self.state_machine.inertial.write_history(out)
        out.f32(self.state_machine.authored_floor_y)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var auto_rec = reader.boolean()
        var sync_t = reader.boolean()
        var sm = RagdollStateMachine()
        sm.state = reader.integer()
        sm.ragdoll.deserialize(reader)
        sm.supine_getup_clip = reader.integer()
        sm.prone_getup_clip = reader.integer()
        sm.getup_duration = reader.f32()
        sm.getup_time = reader.f32()
        sm.getup_halflife = reader.f32()
        sm.active_pd_duration = reader.f32()
        sm.active_pd_timer = reader.f32()
        sm.min_limp_time = reader.f32()
        sm.limp_timer = reader.f32()
        sm.ignore_body_id = reader.integer()
        sm.chest_limb_index = reader.integer()
        sm.last_pelvis_snap_m = reader.f32()
        sm.recovery_count = reader.integer()
        _ = reader.integer()
        sm.world_ragdoll_slot = -1
        sm._world_dirty = True
        if version >= 2:
            sm.query_padding = reader.f32()
            sm.filter.category = reader.u32()
            sm.filter.mask = reader.u32()
            sm.filter.group = reader.integer()
            sm.inertial.read_history(reader)
        if version >= 3:
            sm.authored_floor_y = reader.f32()
        if reader.failed:
            raise Error("RagdollComponent: truncated blob")
        return Self(sm^, auto_rec, sync_t)

    def write_json(self) -> String:
        return (
            "{\"state\": " + String(self.state_machine.state)
            + ", \"auto_recover\": " + ("true" if self.auto_recover else "false")
            + ", \"pelvis_syncs_transform\": " + ("true" if self.pelvis_syncs_transform else "false")
            + ", \"authored_floor_y\": " + _f32_s(self.state_machine.authored_floor_y)
            + ", \"limbs\": " + String(len(self.state_machine.ragdoll.limbs))
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var sm = RagdollStateMachine()
        sm.state = doc.integer(doc.object_get(node, "state"), RAGDOLL_SM_ANIMATED)
        sm.authored_floor_y = Float32(doc.number(doc.object_get(node, "authored_floor_y"), 0.0))
        var ar = doc.boolean(doc.object_get(node, "auto_recover"), True)
        var ps = doc.boolean(doc.object_get(node, "pelvis_syncs_transform"), True)
        return Self(sm^, ar, ps)


def _ascii_lower(s: String) -> String:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        var c = b[i]
        if c >= 65 and c <= 90:
            out.append(c + 32)
        else:
            out.append(c)
    return String(unsafe_from_utf8=out)


struct RigComponent(SerializableComponent):
    """Links an entity to a `.flrig` rig asset (`rig_ref`) and tracks bound joint count."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.anim.RigComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "RigComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ARIG")
    comptime SCHEMA_VERSION: Int = 1

    var rig_ref: AssetRef
    var bound: Bool
    var joint_count: Int

    def __init__(
        out self,
        var rig_ref: AssetRef = AssetRef(),
        bound: Bool = False,
        joint_count: Int = 0,
    ):
        self.rig_ref = rig_ref^
        self.bound = bound
        self.joint_count = joint_count

    def __init__(out self, *, copy: Self):
        self.rig_ref = copy.rig_ref.copy()
        self.bound = copy.bound
        self.joint_count = copy.joint_count

    def write_bin(self, mut out: BlobWriter):
        self.rig_ref.write_bin(out)
        out.boolean(self.bound)
        out.integer(self.joint_count)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var rref = AssetRef.read_bin(reader, 1)
        var bnd = reader.boolean()
        var jc = reader.integer()
        if reader.failed:
            raise Error("RigComponent: truncated blob")
        return Self(rref^, bnd, jc)

    def write_json(self) -> String:
        return (
            "{\"rig_ref\": " + self.rig_ref.write_json()
            + ", \"bound\": " + ("true" if self.bound else "false")
            + ", \"joint_count\": " + String(self.joint_count)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var rnode = doc.object_get(node, "rig_ref")
        var rref = AssetRef.read_json(doc, rnode) if rnode >= 0 else AssetRef()
        var bnd = doc.boolean(doc.object_get(node, "bound"), False)
        var jc = doc.integer(doc.object_get(node, "joint_count"), 0)
        return Self(rref^, bnd, jc)


struct BoneSocketComponent(SerializableComponent):
    """Attaches this entity (and optional kinematic body) to a skeleton bone."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.anim.BoneSocketComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "BoneSocketComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ABSK")
    comptime SCHEMA_VERSION: Int = 2

    var skeleton_entity: Entity
    var attachment: BoneSocketAttachment
    var skeleton_slot: Int

    def __init__(
        out self,
        skeleton_entity: Entity = NULL_ENTITY,
        var attachment: BoneSocketAttachment = BoneSocketAttachment(),
        skeleton_slot: Int = -1,
    ):
        self.skeleton_entity = skeleton_entity
        self.attachment = attachment^
        self.skeleton_slot = skeleton_slot

    def write_bin(self, mut out: BlobWriter):
        out.u64(self.skeleton_entity.to_bits())
        out.text(self.attachment.name)
        out.integer(self.attachment.bone_node)
        out.integer(self.attachment.entity_id)
        out.integer(self.attachment.kinematic_body_id)
        out.vec3(self.attachment.local_pos)
        out.quat(self.attachment.local_rot)
        out.vec3(self.attachment.local_scale)
        out.vec3(self.attachment.world_pos)
        out.quat(self.attachment.world_rot)
        out.vec3(self.attachment.world_scale)
        out.integer(self.skeleton_slot)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var sk_e = Entity.from_bits(reader.u64())
        var name = reader.text()
        var bone_node = reader.integer()
        var eid = reader.integer()
        var kbid = reader.integer()
        var lp = reader.vec3()
        var lr = reader.quat()
        var ls = reader.vec3()
        var wp = reader.vec3()
        var wr = reader.quat()
        var ws = reader.vec3()
        var sk_slot = reader.integer() if version >= 2 else -1
        var att = BoneSocketAttachment(name, bone_node, eid, lp, lr, ls, kbid)
        att.world_pos = wp
        att.world_rot = wr
        att.world_scale = ws
        if reader.failed:
            raise Error("BoneSocketComponent: truncated blob")
        return Self(sk_e, att^, sk_slot)

    def write_json_with_slot(self, skeleton_slot: Int) -> String:
        return (
            "{\"name\": " + json_quote(self.attachment.name)
            + ", \"bone_node\": " + String(self.attachment.bone_node)
            + ", \"entity_id\": " + String(self.attachment.entity_id)
            + ", \"kinematic_body_id\": " + String(self.attachment.kinematic_body_id)
            + ", \"skeleton_slot\": " + String(skeleton_slot)
            + ", \"local_pos\": " + _vec3_json(self.attachment.local_pos)
            + ", \"local_rot\": " + _quat_json(self.attachment.local_rot)
            + ", \"local_scale\": " + _vec3_json(self.attachment.local_scale)
            + ", \"world_pos\": " + _vec3_json(self.attachment.world_pos)
            + ", \"world_rot\": " + _quat_json(self.attachment.world_rot)
            + ", \"world_scale\": " + _vec3_json(self.attachment.world_scale)
            + "}"
        )

    def write_json(self) -> String:
        return self.write_json_with_slot(self.skeleton_slot)

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var name = doc.text(doc.object_get(node, "name"), "")
        var bnode = doc.integer(doc.object_get(node, "bone_node"), -1)
        var eid = doc.integer(doc.object_get(node, "entity_id"), -1)
        var kbid = doc.integer(doc.object_get(node, "kinematic_body_id"), -1)
        var sk_slot = doc.integer(doc.object_get(node, "skeleton_slot"), -1)
        var lp = _read_vec3_json(doc, doc.object_get(node, "local_pos"), Vec3())
        var lr = _read_quat_json(doc, doc.object_get(node, "local_rot"))
        var ls = _read_vec3_json(doc, doc.object_get(node, "local_scale"), Vec3(1.0, 1.0, 1.0))
        var wp = _read_vec3_json(doc, doc.object_get(node, "world_pos"), Vec3())
        var wr = _read_quat_json(doc, doc.object_get(node, "world_rot"))
        var ws = _read_vec3_json(doc, doc.object_get(node, "world_scale"), Vec3(1.0, 1.0, 1.0))
        var att = BoneSocketAttachment(name, bnode, eid, lp, lr, ls, kbid)
        att.world_pos = wp
        att.world_rot = wr
        att.world_scale = ws
        return Self(NULL_ENTITY, att^, sk_slot)


def _bone_socket_write_bin_slots(
    world_addr: Int, entity: Entity, ent_to_slot_addr: Int, writer_addr: Int
) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref ent_to_slot = Pointer[List[Int], ImmutAnyOrigin](unsafe_from_address=ent_to_slot_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    var ptr_opt = world.get_ptr[BoneSocketComponent](entity)
    if not ptr_opt:
        return False
    ref bs = ptr_opt.value()[unsafe_offset=0]
    var sk_slot = -1
    if world.is_alive(bs.skeleton_entity) and Int(bs.skeleton_entity.index) < len(ent_to_slot):
        sk_slot = ent_to_slot[Int(bs.skeleton_entity.index)]
    var tmp = BoneSocketComponent(NULL_ENTITY, bs.attachment.copy(), sk_slot)
    out.begin(BoneSocketComponent.SCHEMA_FOURCC, BoneSocketComponent.SCHEMA_VERSION)
    tmp.write_bin(out)
    out.end()
    return True


def _bone_socket_write_json_slots(
    world_addr: Int, entity: Entity, ent_to_slot_addr: Int, str_out_addr: Int
) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref ent_to_slot = Pointer[List[Int], ImmutAnyOrigin](unsafe_from_address=ent_to_slot_addr)[unsafe_offset=0]
    ref s_out = Pointer[String, MutAnyOrigin](unsafe_from_address=str_out_addr)[unsafe_offset=0]
    var ptr_opt = world.get_ptr[BoneSocketComponent](entity)
    if not ptr_opt:
        return False
    ref bs = ptr_opt.value()[unsafe_offset=0]
    var sk_slot = -1
    if world.is_alive(bs.skeleton_entity) and Int(bs.skeleton_entity.index) < len(ent_to_slot):
        sk_slot = ent_to_slot[Int(bs.skeleton_entity.index)]
    s_out = bs.write_json_with_slot(sk_slot)
    return True


def _bone_socket_resolve_slots(world_addr: Int, entity: Entity, spawned_addr: Int):
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref spawned = Pointer[List[Entity], ImmutAnyOrigin](unsafe_from_address=spawned_addr)[unsafe_offset=0]
    var bs_opt = world.get_ptr_mut[BoneSocketComponent](entity)
    if not bs_opt:
        return
    ref bs = bs_opt.value()[unsafe_offset=0]
    if bs.skeleton_slot >= 0 and bs.skeleton_slot < len(spawned):
        bs.skeleton_entity = spawned[bs.skeleton_slot]
        bs.skeleton_slot = -1
    elif not world.is_alive(bs.skeleton_entity):
        var par_opt = world.get_ptr[Parent](entity)
        if par_opt and world.is_alive(par_opt.value()[unsafe_offset=0].entity):
            var pe = par_opt.value()[unsafe_offset=0].entity
            if world.has[SkeletonComponent](pe):
                bs.skeleton_entity = pe


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_anim_components(mut world: World):
    """Idempotently registers all 5 animation components + custom binary snapshot hooks."""
    _register_non_trivial_anim[SkeletonComponent](world)
    _register_non_trivial_anim[AnimatorComponent](world)
    _register_non_trivial_anim[RagdollComponent](world)
    _register_non_trivial_anim[BoneSocketComponent](world)
    _register_non_trivial_anim[RigComponent](world)


def register_anim_codecs(mut reg: ComponentRegistry):
    """Registers all 5 animation `SerializableComponent` codecs on `ComponentRegistry`."""
    reg.register[SkeletonComponent]()
    reg.register[AnimatorComponent]()
    reg.register[RagdollComponent]()
    reg.register[BoneSocketComponent]()
    reg.set_slot_hooks[BoneSocketComponent](
        _bone_socket_write_bin_slots,
        _bone_socket_write_json_slots,
        _bone_socket_resolve_slots,
    )
    reg.register[RigComponent]()


# ---------------------------------------------------------------------------
# Systems (`STAGE_UPDATE`, ordered before `STAGE_POST_UPDATE` transforms)
# ---------------------------------------------------------------------------

def _current_world_global_transform(world: World, entity: Entity) -> Tuple[Bool, GlobalTransform]:
    var chain = List[Entity]()
    var cur = entity
    var steps = 0
    while world.is_alive(cur) and steps < 64:
        chain.append(cur)
        var par_opt = world.get_ptr[Parent](cur)
        if not par_opt or not world.is_alive(par_opt.value()[unsafe_offset=0].entity):
            break
        cur = par_opt.value()[unsafe_offset=0].entity
        steps += 1
    if len(chain) == 0:
        return (False, GlobalTransform())
    var acc = GlobalTransform()
    var has_acc = False
    var idx = len(chain) - 1
    while idx >= 0:
        var node = chain[idx]
        var lt_opt = world.get_ptr[LocalTransform](node)
        if lt_opt:
            if not has_acc:
                acc = GlobalTransform.from_local(lt_opt.value()[unsafe_offset=0])
                has_acc = True
            else:
                acc = acc.compose_child(lt_opt.value()[unsafe_offset=0])
        else:
            var gt_opt = world.get_ptr[GlobalTransform](node)
            if gt_opt:
                acc = gt_opt.value()[unsafe_offset=0].copy()
                has_acc = True
        idx -= 1
    return (has_acc, acc^)


def _placement_from_world(world: World, entity: Entity) -> Placement:
    var par_opt = world.get_ptr[Parent](entity)
    if par_opt and world.is_alive(par_opt.value()[unsafe_offset=0].entity):
        var gt_res = _current_world_global_transform(world, entity)
        if gt_res[0]:
            ref gt = gt_res[1]
            return Placement(gt.translation, gt.scale, _quat_yaw_deg(gt.rotation))
    var lt_opt = world.get_ptr[LocalTransform](entity)
    if lt_opt:
        ref lt = lt_opt.value()[unsafe_offset=0]
        return Placement(lt.translation, lt.scale, _quat_yaw_deg(lt.rotation))
    var gt_opt = world.get_ptr[GlobalTransform](entity)
    if gt_opt:
        ref gt = gt_opt.value()[unsafe_offset=0]
        return Placement(gt.translation, gt.scale, _quat_yaw_deg(gt.rotation))
    return Placement()


def _resolve_clip_index(authored: Int, mapping: List[Int], names: List[String]) raises -> Int:
    if authored < 0 or authored >= len(mapping):
        raise Error("Clip index " + String(authored) + " is outside the clip table")
    if mapping[authored] < 0:
        raise Error("Model has no unique clip named " + names[authored])
    return mapping[authored]


def _remap_clip_conditions(mut conditions: List[Condition], mapping: List[Int], names: List[String]) raises:
    for i in range(len(conditions)):
        ref condition = conditions[i]
        if condition.is_clip:
            if not isfinite(condition.value) or condition.value < 0 or condition.value >= Float32(len(mapping)):
                raise Error("Condition references an invalid clip index")
            condition.value = Float32(_resolve_clip_index(Int(condition.value), mapping, names))


def _bind_model_clips(mut graph: AnimGraphDef, model: NativeModel) raises -> List[ClipInfo]:
    """Resolve named FLAN references against this model. Legacy graphs use GLB order."""
    var clips = List[ClipInfo]()
    for anim in model.animations:
        var clip = ClipInfo(anim.name, anim.duration, anim.loop, anim.ground_speed)
        for ev in anim.events:
            clip.events.append(ClipEventInfo(ev.kind, ev.arg, ev.time))
        clips.append(clip^)
    var mapping = List[Int]()
    if len(graph.clip_names) == 0:
        for i in range(len(clips)):
            mapping.append(i)
    else:
        for name in graph.clip_names:
            var idx = -1
            for i in range(len(clips)):
                if clips[i].name == name:
                    if idx >= 0:
                        idx = -1
                        break
                    idx = i
            mapping.append(idx)
    for i in range(len(graph.states)):
        ref st = graph.states[i]
        _remap_clip_conditions(st.when, mapping, graph.clip_names)
        _remap_clip_conditions(st.hold_while, mapping, graph.clip_names)
        if st.clip >= 0:
            st.clip = _resolve_clip_index(st.clip, mapping, graph.clip_names)
        elif st.blend_2d >= 0:
            if st.blend_2d >= len(graph.blend_spaces_2d) or len(graph.blend_spaces_2d[st.blend_2d].samples) == 0:
                raise Error("Invalid 2D blend space in state " + st.name)
        elif st.blend < 0 or st.blend >= len(graph.blend_spaces) or len(graph.blend_spaces[st.blend].samples) == 0:
            raise Error("Invalid 1D blend space in state " + st.name)
    for i in range(len(graph.blend_spaces)):
        ref space = graph.blend_spaces[i]
        for j in range(len(space.samples)):
            ref sample = space.samples[j]
            sample.clip = _resolve_clip_index(sample.clip, mapping, graph.clip_names)
    for i in range(len(graph.blend_spaces_2d)):
        ref space = graph.blend_spaces_2d[i]
        for j in range(len(space.samples)):
            ref sample = space.samples[j]
            sample.clip = _resolve_clip_index(sample.clip, mapping, graph.clip_names)
    for i in range(len(graph.layers)):
        ref layer = graph.layers[i]
        _remap_clip_conditions(layer.when, mapping, graph.clip_names)
        layer.clip = _resolve_clip_index(layer.clip, mapping, graph.clip_names)
    return clips^


def _bind_cached_graph(mut ac: AnimatorComponent, cache: AnimAssetCache, model_slot: Int):
    var g = cache.find_graph(ac.graph_id)
    if g < 0:
        if ac.bound_graph_slot >= 0:
            ac.animator = Animator()
            ac.bound_graph_slot = -1
            ac.bound = False
        return
    if (ac.bound_graph_slot == g and ac.bound_model_slot == model_slot
        and ac.bound_graph_version == cache.graph_versions[g]
        and ac.bound_model_version == cache.model_versions[model_slot]):
        return
    var manual = ac.animator.manual_state
    var state = ac.animator.state
    var previous = ac.animator.previous
    var clocks = (ac.animator.state_time, ac.animator.clip_time, ac.animator.prev_clip_time,
                  ac.animator.phase, ac.animator.prev_phase)
    ac.graph_binding_error = String()
    if len(cache.graph_clips[g]) > 0:
        ac.animator.configure(cache.graphs[g], cache.graph_clips[g])
    else:
        var graph = cache.graphs[g].copy()
        try:
            var clips = _bind_model_clips(graph, cache.models[model_slot])
            ac.animator.configure(graph, clips)
        except err:
            ac.graph_binding_error = ac.graph_ref.rel_path + ": " + String(err)
            ac.animator.configure(graph, List[ClipInfo]())
    if manual >= 0 and manual < len(ac.animator.graph.states):
        ac.animator.set_manual_state(manual)
    if ac.restore_graph_clocks and ac.animator.configured() and state >= 0 and state < len(ac.animator.graph.states):
        ac.animator.state = state
        ac.animator.previous = previous if previous >= 0 and previous < len(ac.animator.graph.states) else -1
        ac.animator.state_time = clocks[0]
        ac.animator.clip_time = clocks[1]
        ac.animator.prev_clip_time = clocks[2]
        ac.animator.phase = clocks[3]
        ac.animator.prev_phase = clocks[4]
    ac.restore_graph_clocks = False
    ac.bound_graph_slot = g
    ac.bound_model_slot = model_slot
    ac.bound_graph_version = cache.graph_versions[g]
    ac.bound_model_version = cache.model_versions[model_slot]
    ac.bound = False


def _sys_anim_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    fixed_pass: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var cache_opt = resources.get_ptr[AnimAssetCache]()
    if not cache_opt:
        return
    ref cache = cache_opt.value()[unsafe_offset=0]
    var ev_opt = resources.get_ptr_mut[AnimEventsResource]()

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        dt = max(Float32(0.0), ft_opt.value()[unsafe_offset=0].delta_seconds)

    if fixed_pass != 0:
        dt = resources.get_ptr[FixedTime]().value()[].dt

    var controller_system = resources.has[PhysicsSyncStateResource]()
    var skel_mask = world.bit_for[SkeletonComponent]()
    var anim_mask = world.bit_for[AnimatorComponent]()

    var animated = skel_mask.union(anim_mask)
    if fixed_pass != 0:
        animated = animated.union(world.bit_for[CharacterControllerComponent]())
    var anim_entities = world.query_entities(animated)
    for i in range(len(anim_entities)):
        var e = anim_entities[i]
        var sk_opt = world.get_ptr_mut[SkeletonComponent](e)
        var ac_opt = world.get_ptr_mut[AnimatorComponent](e)
        if not sk_opt or not ac_opt:
            continue
        ref sk = sk_opt.value()[unsafe_offset=0]
        ref ac = ac_opt.value()[unsafe_offset=0]

        var kcc = world.get_ptr_mut[CharacterControllerComponent](e)
        var fixed_root = controller_system and ac.apply_root_motion and kcc and kcc.value()[].active
        if fixed_pass != 0 and not fixed_root:
            continue
        if fixed_root:
            ac.actual_displacement = kcc.value()[].actual_displacement
        var m_idx = cache.find_model(sk.model_id)
        if m_idx < 0:
            continue
        ref model = cache.models[m_idx]

        if ac.graph_id != 0:
            _bind_cached_graph(ac, cache, m_idx)
        elif ac.bound_graph_slot >= 0:
            ac.animator = Animator()
            ac.bound_graph_slot = -1
            ac.bound = False

        if sk.dirty_topology:
            sk.skeleton.ensure_topology(model)
            sk.dirty_topology = False

        if fixed_pass == 0:
            sk.skeleton.snapshot_prev_matrices()
        if ac.animator.configured():
            if fixed_root and fixed_pass == 0:
                # A display frame can contain zero or several simulation ticks.
                # Rebuild the base pose before applying this frame's procedural layers.
                if ac.animator.pose.count() == 0:
                    ac.animator.evaluate_paused(model, m_idx)
                ac.animator.apply_to_skeleton(sk.skeleton, model)
                continue
            var eff_dt = Float32(0.0)
            if ac.playing and isfinite(ac.speed) and ac.speed > Float32(0.0):
                eff_dt = dt * ac.speed
            if eff_dt > Float32(0.0):
                ac.animator.update(model, m_idx, eff_dt)
            else:
                ac.animator.evaluate_paused(model, m_idx)
            if fixed_pass == 0:
                ac.animator.apply_to_skeleton(sk.skeleton, model)
            ac.bound = True
            if ev_opt:
                ref ev_res = ev_opt.value()[unsafe_offset=0].events
                for ei in range(len(ac.animator.events)):
                    _ = ev_res.send(AnimEventRecord(e, ac.animator.events[ei]))
            if fixed_pass != 0:
                var rag = world.get_ptr[RagdollComponent](e)
                if not rag or rag.value()[].state_machine.state == RAGDOLL_SM_ANIMATED:
                    kcc.value()[].root_motion_local = ac.animator.root_delta
                    kcc.value()[].root_motion_yaw = ac.animator.root_yaw_delta
            elif ac.apply_root_motion:
                var lt_opt = world.get_ptr_mut[LocalTransform](e)
                if lt_opt:
                    ref lt = lt_opt.value()[unsafe_offset=0]
                    lt.translation = lt.translation + lt.rotation.rotate(ac.animator.root_delta)
                    if abs(ac.animator.root_yaw_delta) > Float32(1.0e-6):
                        var dq = angle_axis(ac.animator.root_yaw_delta, Vec3(0.0, 1.0, 0.0))
                        lt.rotation = (dq * lt.rotation).normalized()
        else:
            # A scene graph that failed name resolution must never silently
            # animate an unrelated numbered clip from the replacement GLB.
            if fixed_pass != 0 or ac.bound_graph_slot >= 0:
                continue
            var direct_cidx = -1
            if ac.clip_index >= 0 and ac.clip_index < len(model.animations):
                direct_cidx = ac.clip_index
            elif ac.clip_name.byte_length() > 0 and len(model.animations) > 0:
                var target_lower = _ascii_lower(ac.clip_name)
                for ci in range(len(model.animations)):
                    if model.animations[ci].name == ac.clip_name or _ascii_lower(model.animations[ci].name) == target_lower:
                        direct_cidx = ci
                        break
            if direct_cidx >= 0:
                if ac.playing:
                    ac.time += dt * ac.speed
                var dur = model.animations[direct_cidx].duration
                var sample_t = ac.time
                if ac.loop and dur > Float32(1.0e-5):
                    sample_t = ac.time - Float32(Int(ac.time / dur)) * dur
                    if sample_t < Float32(0.0):
                        sample_t += dur
                sample_pose(model, direct_cidx, sample_t, ac.animator.pose)
                sk.skeleton.apply_pose(ac.animator.pose, model)
                ac.bound = True
            sk.skeleton.update_matrices(model)

    if fixed_pass != 0:
        return

    # Skeleton-only entities (no AnimatorComponent)
    var skel_only = world.query_entities(skel_mask, anim_mask)
    for i in range(len(skel_only)):
        var e = skel_only[i]
        var sk_opt = world.get_ptr_mut[SkeletonComponent](e)
        if not sk_opt:
            continue
        ref sk = sk_opt.value()[unsafe_offset=0]
        var m_idx = cache.find_model(sk.model_id)
        if m_idx < 0:
            continue
        ref model = cache.models[m_idx]
        sk.skeleton.snapshot_prev_matrices()
        if sk.dirty_topology:
            sk.skeleton.ensure_topology(model)
            sk.dirty_topology = False
        sk.skeleton.update_matrices(model)


def _sys_anim_foot_plant(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var cache_opt = resources.get_ptr[AnimAssetCache]()
    var phys_opt = resources.get_ptr[PhysicsWorld]()
    if not cache_opt or not phys_opt:
        return
    ref cache = cache_opt.value()[unsafe_offset=0]
    ref physics = phys_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        dt = max(Float32(1.0e-5), ft_opt.value()[unsafe_offset=0].delta_seconds)

    var mask = world.bit_for[SkeletonComponent]().union(world.bit_for[AnimatorComponent]())
    var entities = world.query_entities(mask)
    for i in range(len(entities)):
        var e = entities[i]
        var ac_opt = world.get_ptr_mut[AnimatorComponent](e)
        var sk_opt = world.get_ptr_mut[SkeletonComponent](e)
        if not ac_opt or not sk_opt:
            continue
        ref ac = ac_opt.value()[unsafe_offset=0]
        if not ac.enable_foot_plant or len(ac.foot_plant.legs) == 0:
            continue
        ref sk = sk_opt.value()[unsafe_offset=0]
        var m_idx = cache.find_model(sk.model_id)
        if m_idx < 0:
            continue
        var rag = world.get_ptr[RagdollComponent](e)
        if rag and rag.value()[].state_machine.state != RAGDOLL_SM_ANIMATED:
            ac.foot_plant.reset()
            continue
        var placement = _placement_from_world(world, e)
        _ = solve_foot_plant_world(
            ac.foot_plant,
            physics,
            sk.skeleton,
            cache.models[m_idx],
            placement,
            ac.foot_in_stance,
            dt,
            ignore_body_id=ac.ignore_body_id,
        )


def _sys_anim_ragdoll_cleanup(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var phys_opt = resources.get_ptr_mut[PhysicsWorld]()
    var sync_opt = resources.get_ptr_mut[AnimSyncStateResource]()
    if not phys_opt or not sync_opt:
        return
    ref physics = phys_opt.value()[unsafe_offset=0]
    ref sync = sync_opt.value()[unsafe_offset=0]

    var keep_rags = List[_TrackedRagdollSync]()
    for i in range(len(sync.tracked_ragdolls)):
        var rec = sync.tracked_ragdolls[i]
        var e = Entity.from_bits(rec.entity_bits)
        var rc_opt = world.get_ptr[RagdollComponent](e)
        var alive_rag = (
            world.is_alive(e)
            and Bool(rc_opt)
            and world.has[SkeletonComponent](e)
            and rc_opt.value()[unsafe_offset=0].state_machine.world_ragdoll_slot == rec.slot
        )
        if not alive_rag:
            if rec.slot >= 0 and rec.slot < len(physics.ragdolls):
                physics.set_ragdoll_active(rec.slot, False)
        else:
            keep_rags.append(rec)
    sync.tracked_ragdolls = keep_rags^


def _sys_anim_ragdoll(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var phys_opt = resources.get_ptr_mut[PhysicsWorld]()
    if not phys_opt:
        return
    ref physics = phys_opt.value()[unsafe_offset=0]

    var sync_opt = resources.get_ptr_mut[AnimSyncStateResource]()

    var cache_opt = resources.get_ptr[AnimAssetCache]()
    if not cache_opt:
        return
    ref cache = cache_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        dt = max(Float32(1.0e-5), ft_opt.value()[unsafe_offset=0].delta_seconds)

    var mask = world.bit_for[RagdollComponent]().union(world.bit_for[SkeletonComponent]())
    var entities = world.query_entities(mask)
    for i in range(len(entities)):
        var e = entities[i]
        var rc_opt = world.get_ptr_mut[RagdollComponent](e)
        var sk_opt = world.get_ptr_mut[SkeletonComponent](e)
        if not rc_opt or not sk_opt:
            continue
        ref rc = rc_opt.value()[unsafe_offset=0]
        ref sk = sk_opt.value()[unsafe_offset=0]
        if len(rc.state_machine.ragdoll.limbs) == 0:
            continue
        var m_idx = cache.find_model(sk.model_id)
        if m_idx < 0:
            continue
        ref model = cache.models[m_idx]

        var anim_pose = Pose()
        var ac_opt = world.get_ptr[AnimatorComponent](e)
        if ac_opt:
            anim_pose = ac_opt.value()[unsafe_offset=0].animator.pose.copy()
        else:
            anim_pose.rest(model)

        var placement = _placement_from_world(world, e)
        # CRITICAL: step_world=False because PhysicsPlugin already stepped PhysicsWorld in STAGE_FIXED_UPDATE!
        rc.state_machine.update_world(
            physics,
            sk.skeleton,
            placement,
            model,
            anim_pose,
            dt,
            auto_recover=rc.auto_recover,
            step_world=False,
            snapshot_previous=False,
        )
        if sync_opt and rc.state_machine.world_ragdoll_slot >= 0:
            sync_opt.value()[unsafe_offset=0].record_ragdoll(
                e, rc.state_machine.world_ragdoll_slot
            )
        if rc.pelvis_syncs_transform and rc.state_machine.state != RAGDOLL_SM_ANIMATED:
            var lt_opt = world.get_ptr_mut[LocalTransform](e)
            if lt_opt:
                lt_opt.value()[unsafe_offset=0].translation = placement.position
                lt_opt.value()[unsafe_offset=0].rotation = placement.world_yaw_quat()


def _sys_anim_bone_sockets(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var phys_opt = resources.get_ptr_mut[PhysicsWorld]()

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        dt = max(Float32(1.0e-5), ft_opt.value()[unsafe_offset=0].delta_seconds)

    var sock_mask = world.bit_for[BoneSocketComponent]()
    var entities = world.query_entities(sock_mask)
    for i in range(len(entities)):
        var e = entities[i]
        var bs_opt = world.get_ptr_mut[BoneSocketComponent](e)
        if not bs_opt:
            continue
        ref bs = bs_opt.value()[unsafe_offset=0]
        if not world.is_alive(bs.skeleton_entity):
            continue
        var sk_opt = world.get_ptr[SkeletonComponent](bs.skeleton_entity)
        if not sk_opt:
            continue
        var placement = _placement_from_world(world, bs.skeleton_entity)
        ref skel = sk_opt.value()[unsafe_offset=0].skeleton
        if bs.attachment.kinematic_body_id >= 0 and phys_opt:
            bs.attachment.sync_to_physics(skel, placement, phys_opt.value()[unsafe_offset=0], dt)
        else:
            bs.attachment.update(skel, placement)

        var lt_opt = world.get_ptr_mut[LocalTransform](e)
        if lt_opt:
            var wp = bs.attachment.world_pos
            var wr = bs.attachment.world_rot
            var ws = bs.attachment.world_scale
            var par_opt = world.get_ptr[Parent](e)
            if par_opt and world.is_alive(par_opt.value()[unsafe_offset=0].entity):
                var pg_res = _current_world_global_transform(world, par_opt.value()[unsafe_offset=0].entity)
                if pg_res[0]:
                    ref pg = pg_res[1]
                    var inv_q = pg.rotation.conjugate()
                    var rel = inv_q.rotate(wp - pg.translation)
                    var sx = pg.scale.x if abs(pg.scale.x) > Float32(1.0e-6) else Float32(1.0)
                    var sy = pg.scale.y if abs(pg.scale.y) > Float32(1.0e-6) else Float32(1.0)
                    var sz = pg.scale.z if abs(pg.scale.z) > Float32(1.0e-6) else Float32(1.0)
                    lt_opt.value()[unsafe_offset=0].translation = Vec3(rel.x / sx, rel.y / sy, rel.z / sz)
                    lt_opt.value()[unsafe_offset=0].rotation = (inv_q * wr).normalized()
                    lt_opt.value()[unsafe_offset=0].scale = Vec3(ws.x / sx, ws.y / sy, ws.z / sz)
                    continue
            lt_opt.value()[unsafe_offset=0].translation = wp
            lt_opt.value()[unsafe_offset=0].rotation = wr
            lt_opt.value()[unsafe_offset=0].scale = ws


def _swap_anim_events(res_ptr: Pointer[Resources, MutAnyOrigin]):
    ref resources = res_ptr[unsafe_offset=0]
    var ev_opt = resources.get_ptr_mut[AnimEventsResource]()
    if ev_opt:
        ev_opt.value()[unsafe_offset=0].events.swap_buffers()


def _has_anim_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct AnimPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers animation components, `AnimAssetCache`, and `STAGE_UPDATE` systems."""

    def __init__(out self):
        pass

    def build(self, mut app: EngineApp):
        register_anim_components(app.world)
        if not app.resources.has[AnimAssetCache]():
            app.insert_resource[AnimAssetCache](AnimAssetCache())
        if not app.resources.has[AnimEventsResource]():
            app.insert_resource[AnimEventsResource](AnimEventsResource())
        if not app.resources.has[AnimSyncStateResource]():
            app.insert_resource[AnimSyncStateResource](AnimSyncStateResource())

        var b_lt = app.world.bit_for[LocalTransform]()
        var b_skel = app.world.bit_for[SkeletonComponent]()
        var b_anim = app.world.bit_for[AnimatorComponent]()
        var b_rag = app.world.bit_for[RagdollComponent]()
        var b_sock = app.world.bit_for[BoneSocketComponent]()
        var all_anim = b_lt.union(b_skel).union(b_anim).union(b_rag).union(b_sock)

        if not _has_anim_system(app, STAGE_PRE_UPDATE, "flint.anim.ragdoll_cleanup"):
            _ = app.add_system(
                STAGE_PRE_UPDATE,
                "flint.anim.ragdoll_cleanup",
                _sys_anim_ragdoll_cleanup,
                reads=b_skel.union(b_rag),
                exclusive=True,
            )

        if _has_anim_system(app, STAGE_UPDATE, "flint.anim.update"):
            return

        app.add_event_swap_hook(_swap_anim_events)
        app.insert_resource[CharacterRootMotionSystem](CharacterRootMotionSystem(_sys_anim_update))

        var id_anim = app.add_system(
            STAGE_UPDATE,
            "flint.anim.update",
            _sys_anim_update,
            reads=all_anim,
            writes=all_anim,
            exclusive=True,
        )
        var id_rag = app.add_system(
            STAGE_UPDATE,
            "flint.anim.ragdoll",
            _sys_anim_ragdoll,
            reads=all_anim,
            writes=b_skel.union(b_rag).union(b_lt),
            exclusive=True,
            after_system_id=id_anim,
        )
        var id_ik = app.add_system(
            STAGE_UPDATE,
            "flint.anim.foot_plant",
            _sys_anim_foot_plant,
            reads=all_anim,
            writes=b_skel.union(b_anim),
            exclusive=True,
            after_system_id=id_rag,
        )
        _ = app.add_system(
            STAGE_UPDATE,
            "flint.anim.bone_sockets",
            _sys_anim_bone_sockets,
            reads=all_anim,
            writes=b_sock.union(b_lt),
            exclusive=True,
            after_system_id=id_ik,
        )
