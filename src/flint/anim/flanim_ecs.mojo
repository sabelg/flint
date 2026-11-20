"""Flint v5 Animation Assets & Physics-World ECS Integration (`flint.anim.flanim_ecs`).

Implements:
1. Binary `.flanim` (`FLAN` FourCC) state-machine & blend-tree serializer and loader
   (`serialize_flanim`, `deserialize_flanim`, `save_flanim_bytes`, `load_flanim_bytes`),
   preserving 100% of `AnimGraphDef` states, 1D/2D blend spaces, bone masks, layers,
   transitions, settings, and form retarget offsets.
2. `BoneSocketAttachment` world transform propagator (`update`, `sync_to_physics`,
   `update_socket_attachments`) driving attached entities and kinematic `PhysicsWorld`
   colliders with exact linear & angular surface velocities.
3. `solve_foot_plant_world`: auto-raycasting `PhysicsWorld` per leg to gather terrain/deck
   hit heights & surface normals and solving `FootPlantSolver` directly on `SkeletonInstance`.
4. `RagdollStateMachine`: unified `Animated <-> ActivePD <-> Limp <-> Supine/Prone GetUp`
   controller querying `PhysicsWorld` (`raycast` floor + `overlap_aabb` OBB colliders)
   and re-rooting settled poses with `0.0 mm` world pelvis snap via `Inertializer`.
"""
from std.collections import List
from std.math import abs, floor, max, min, sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4, trs
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.ik import angle_axis
from flint.assets.model import NativeModel
from flint.anim.graph import (
    Condition,
    MaskDef,
    BlendSampleDef,
    BlendSample2DDef,
    BlendSpaceDef,
    BlendSpace2DDef,
    StateDef,
    LayerDef,
    TransitionDef,
    FormOffsetDef,
    AnimGraphDef,
)
from flint.anim.pose import (
    Pose,
    SkeletonInstance,
    Inertializer,
    sample_pose,
    decompose_trs,
    det_quat_to_scaled_angle,
)
from flint.anim.procedural import Placement, FootPlantSolver, DEG
from flint.anim.ragdoll import ArticulatedRagdoll, RagdollObbCollider
from flint.physics.filter import CollisionFilter, CAT_TERRAIN, CAT_STATIC, CAT_ALL
from flint.physics.world import (
    PhysicsWorld,
    BODY_SHAPE_SPHERE,
    BODY_SHAPE_CAPSULE,
    BODY_SHAPE_BOX,
    BODY_SHAPE_CONVEX_HULL,
    BODY_SHAPE_COMPOUND,
    _hull_half_extents,
)
from flint.physics.shapes_v5 import (
    COMPOUND_CHILD_SPHERE,
    COMPOUND_CHILD_CAPSULE,
    COMPOUND_CHILD_BOX,
    COMPOUND_CHILD_CONVEX,
)

comptime FLANIM_FOURCC: UInt32 = fourcc("FLAN")
comptime FLANIM_VERSION: Int = 1

comptime RAGDOLL_SM_ANIMATED: Int = 0
comptime RAGDOLL_SM_ACTIVE_PD: Int = 1
comptime RAGDOLL_SM_LIMP: Int = 2
comptime RAGDOLL_SM_GETUP_SUPINE: Int = 3
comptime RAGDOLL_SM_GETUP_PRONE: Int = 4


# =========================================================================
# 1. Binary `.flanim` (`FLAN` FourCC) Serializer & Loader
# =========================================================================

def _write_conditions(conds: List[Condition], mut w: BlobWriter):
    w.u32(UInt32(len(conds)))
    for i in range(len(conds)):
        w.integer(conds[i].param)
        w.integer(conds[i].op)
        w.f32(conds[i].value)


def _read_conditions(mut r: BlobReader) -> List[Condition]:
    var n = Int(r.u32())
    var out = List[Condition](capacity=max(0, n))
    for _ in range(n):
        var param = r.integer()
        var op = r.integer()
        var val = r.f32()
        out.append(Condition(param, op, val))
    return out^


def serialize_flanim(graph: AnimGraphDef, mut w: BlobWriter):
    """Writes a complete `AnimGraphDef` as a versioned `'FLAN'` binary `.flanim` blob."""
    w.begin(FLANIM_FOURCC, FLANIM_VERSION)
    w.text(graph.name)
    w.f32(graph.default_blend)

    # Parameters
    w.u32(UInt32(len(graph.params)))
    for i in range(len(graph.params)):
        w.text(graph.params[i])

    # Bone masks
    w.u32(UInt32(len(graph.masks)))
    for i in range(len(graph.masks)):
        ref m = graph.masks[i]
        w.text(m.name)
        w.boolean(m.children)
        w.u32(UInt32(len(m.bones)))
        for b in range(len(m.bones)):
            w.text(m.bones[b])
        w.u32(UInt32(len(m.weights)))
        for wt in range(len(m.weights)):
            w.f32(m.weights[wt])

    # 1D Blend Spaces
    w.u32(UInt32(len(graph.blend_spaces)))
    for i in range(len(graph.blend_spaces)):
        ref bs = graph.blend_spaces[i]
        w.text(bs.name)
        w.integer(bs.param)
        w.boolean(bs.sync)
        w.boolean(bs.ground_rate)
        w.f32(bs.min_rate)
        w.f32(bs.max_rate)
        w.u32(UInt32(len(bs.samples)))
        for s in range(len(bs.samples)):
            w.integer(bs.samples[s].clip)
            w.f32(bs.samples[s].at)

    # 2D Blend Spaces
    w.u32(UInt32(len(graph.blend_spaces_2d)))
    for i in range(len(graph.blend_spaces_2d)):
        ref bs2 = graph.blend_spaces_2d[i]
        w.text(bs2.name)
        w.integer(bs2.param_x)
        w.integer(bs2.param_y)
        w.boolean(bs2.is_polar)
        w.boolean(bs2.sync)
        w.boolean(bs2.ground_rate)
        w.f32(bs2.min_rate)
        w.f32(bs2.max_rate)
        w.u32(UInt32(len(bs2.samples)))
        for s in range(len(bs2.samples)):
            w.integer(bs2.samples[s].clip)
            w.f32(bs2.samples[s].x)
            w.f32(bs2.samples[s].y)

    # States
    w.u32(UInt32(len(graph.states)))
    for i in range(len(graph.states)):
        ref st = graph.states[i]
        w.text(st.name)
        w.integer(st.clip)
        w.integer(st.blend)
        w.integer(st.blend_2d)
        w.f32(st.rate)
        w.integer(st.time_param)
        w.integer(st.rate_param)
        w.text(st.hold_event)
        w.boolean(st.holds)
        w.integer(st.root_motion)
        w.integer(st.warp)
        w.boolean(st.reset_time)
        _write_conditions(st.when, w)
        _write_conditions(st.hold_while, w)

    # Layers
    w.u32(UInt32(len(graph.layers)))
    for i in range(len(graph.layers)):
        ref ly = graph.layers[i]
        w.text(ly.name)
        w.integer(ly.clip)
        w.integer(ly.mask)
        w.boolean(ly.additive)
        w.f32(ly.reference_time)
        w.f32(ly.fade)
        w.f32(ly.weight)
        w.integer(ly.weight_param)
        w.f32(ly.rate)
        w.boolean(ly.loop)
        _write_conditions(ly.when, w)
        w.u32(UInt32(len(ly.over)))
        for o in range(len(ly.over)):
            w.integer(ly.over[o])

    # Transitions
    w.u32(UInt32(len(graph.transitions)))
    for i in range(len(graph.transitions)):
        w.integer(graph.transitions[i].source)
        w.integer(graph.transitions[i].target)
        w.f32(graph.transitions[i].blend)

    # Runtime settings
    var n_set = min(len(graph.setting_names), len(graph.setting_values))
    w.u32(UInt32(n_set))
    for i in range(n_set):
        w.text(graph.setting_names[i])
        w.f32(graph.setting_values[i])

    # Form retarget offsets
    w.u32(UInt32(len(graph.forms)))
    for i in range(len(graph.forms)):
        ref fm = graph.forms[i]
        w.text(fm.name)
        w.f32(fm.scale)
        w.f32(fm.stride)
        w.f32(fm.hand_plant)
        w.f32(fm.half_length)
        var nb = min(len(fm.bones), len(fm.bone_scales))
        w.u32(UInt32(nb))
        for b in range(nb):
            w.text(fm.bones[b])
            w.f32(fm.bone_scales[b])

    w.end()


def deserialize_flanim(mut r: BlobReader) raises -> AnimGraphDef:
    """Loads an `AnimGraphDef` from a `'FLAN'` binary `.flanim` blob."""
    _ = r.enter(FLANIM_FOURCC, FLANIM_VERSION)
    var out = AnimGraphDef()
    out.name = r.text()
    out.default_blend = r.f32()

    var n_params = Int(r.u32())
    for _ in range(n_params):
        out.params.append(r.text())

    var n_masks = Int(r.u32())
    for _ in range(n_masks):
        var m = MaskDef()
        m.name = r.text()
        m.children = r.boolean()
        var nb = Int(r.u32())
        for _ in range(nb):
            m.bones.append(r.text())
        var nw = Int(r.u32())
        for _ in range(nw):
            m.weights.append(r.f32())
        out.masks.append(m^)

    var n_bs = Int(r.u32())
    for _ in range(n_bs):
        var bs = BlendSpaceDef()
        bs.name = r.text()
        bs.param = r.integer()
        bs.sync = r.boolean()
        bs.ground_rate = r.boolean()
        bs.min_rate = r.f32()
        bs.max_rate = r.f32()
        var ns = Int(r.u32())
        for _ in range(ns):
            var clip = r.integer()
            var at = r.f32()
            bs.samples.append(BlendSampleDef(clip, at))
        out.blend_spaces.append(bs^)

    var n_bs2 = Int(r.u32())
    for _ in range(n_bs2):
        var bs2 = BlendSpace2DDef()
        bs2.name = r.text()
        bs2.param_x = r.integer()
        bs2.param_y = r.integer()
        bs2.is_polar = r.boolean()
        bs2.sync = r.boolean()
        bs2.ground_rate = r.boolean()
        bs2.min_rate = r.f32()
        bs2.max_rate = r.f32()
        var ns = Int(r.u32())
        for _ in range(ns):
            var clip = r.integer()
            var sx = r.f32()
            var sy = r.f32()
            bs2.samples.append(BlendSample2DDef(clip, sx, sy))
        out.blend_spaces_2d.append(bs2^)

    var n_states = Int(r.u32())
    for _ in range(n_states):
        var st = StateDef()
        st.name = r.text()
        st.clip = r.integer()
        st.blend = r.integer()
        st.blend_2d = r.integer()
        st.rate = r.f32()
        st.time_param = r.integer()
        st.rate_param = r.integer()
        st.hold_event = r.text()
        st.holds = r.boolean()
        st.root_motion = r.integer()
        st.warp = r.integer()
        st.reset_time = r.boolean()
        st.when = _read_conditions(r)
        st.hold_while = _read_conditions(r)
        out.states.append(st^)

    var n_layers = Int(r.u32())
    for _ in range(n_layers):
        var ly = LayerDef()
        ly.name = r.text()
        ly.clip = r.integer()
        ly.mask = r.integer()
        ly.additive = r.boolean()
        ly.reference_time = r.f32()
        ly.fade = r.f32()
        ly.weight = r.f32()
        ly.weight_param = r.integer()
        ly.rate = r.f32()
        ly.loop = r.boolean()
        ly.when = _read_conditions(r)
        var no = Int(r.u32())
        for _ in range(no):
            ly.over.append(r.integer())
        out.layers.append(ly^)

    var n_trans = Int(r.u32())
    for _ in range(n_trans):
        var src = r.integer()
        var dst = r.integer()
        var bl = r.f32()
        out.transitions.append(TransitionDef(src, dst, bl))

    var n_set = Int(r.u32())
    for _ in range(n_set):
        out.setting_names.append(r.text())
        out.setting_values.append(r.f32())

    var n_forms = Int(r.u32())
    for _ in range(n_forms):
        var fm = FormOffsetDef()
        fm.name = r.text()
        fm.scale = r.f32()
        fm.stride = r.f32()
        fm.hand_plant = r.f32()
        fm.half_length = r.f32()
        var nb = Int(r.u32())
        for _ in range(nb):
            fm.bones.append(r.text())
            fm.bone_scales.append(r.f32())
        out.forms.append(fm^)

    r.leave()
    if r.failed:
        raise Error("corrupted .flanim payload")
    return out^


def save_flanim_bytes(graph: AnimGraphDef) -> List[UInt8]:
    """Serializes `graph` into a standalone `.flanim` byte buffer."""
    var w = BlobWriter()
    serialize_flanim(graph, w)
    return w.data.copy()


def load_flanim_bytes(data: List[UInt8]) raises -> AnimGraphDef:
    """Deserializes an `AnimGraphDef` from a `.flanim` byte buffer."""
    var r = BlobReader(data.copy())
    return deserialize_flanim(r)


# =========================================================================
# 2. `BoneSocketAttachment` World Transform Propagator
# =========================================================================

struct BoneSocketAttachment(Copyable, Movable):
    """Propagates an animated skeleton joint's world transform + local socket offset
    to an attached ECS entity (`entity_id`) and optional kinematic `PhysicsWorld`
    body (`kinematic_body_id`), including exact finite-difference linear & angular
    surface velocities."""
    var name: String
    var bone_node: Int
    var entity_id: Int
    var kinematic_body_id: Int
    var local_pos: Vec3
    var local_rot: Quat
    var local_scale: Vec3
    var world_pos: Vec3
    var world_rot: Quat
    var world_scale: Vec3
    var world_matrix: Mat4
    var prev_world_pos: Vec3
    var prev_world_rot: Quat
    var has_prev: Bool

    def __init__(
        out self,
        name: String = "",
        bone_node: Int = -1,
        entity_id: Int = -1,
        local_pos: Vec3 = Vec3(),
        local_rot: Quat = Quat(),
        local_scale: Vec3 = Vec3(1, 1, 1),
        kinematic_body_id: Int = -1,
    ):
        self.name = name
        self.bone_node = bone_node
        self.entity_id = entity_id
        self.kinematic_body_id = kinematic_body_id
        self.local_pos = local_pos
        self.local_rot = local_rot.normalized()
        self.local_scale = local_scale
        self.world_pos = Vec3()
        self.world_rot = Quat()
        self.world_scale = Vec3(1, 1, 1)
        self.world_matrix = Mat4()
        self.prev_world_pos = Vec3()
        self.prev_world_rot = Quat()
        self.has_prev = False

    def reset_history(mut self):
        self.has_prev = False

    def update(mut self, skel: SkeletonInstance, placement: Placement):
        """Computes exact world TRS and 4x4 matrix from `placement * skel[bone_node] * local`."""
        if self.has_prev:
            self.prev_world_pos = self.world_pos
            self.prev_world_rot = self.world_rot
        var place_rot = placement.world_yaw_quat()
        var place_mat = trs(placement.position, place_rot, placement.scale)
        var local_mat = trs(self.local_pos, self.local_rot, self.local_scale)

        if self.bone_node >= 0 and self.bone_node < len(skel.world_matrices):
            self.world_matrix = place_mat.multiply(skel.world_matrices[self.bone_node]).multiply(local_mat)
        else:
            self.world_matrix = place_mat.multiply(local_mat)

        var dec = decompose_trs(self.world_matrix)
        self.world_pos = dec[0]
        self.world_rot = dec[1]
        self.world_scale = dec[2]
        if not self.has_prev:
            self.prev_world_pos = self.world_pos
            self.prev_world_rot = self.world_rot
            self.has_prev = True

    def sync_to_physics(
        mut self,
        skel: SkeletonInstance,
        placement: Placement,
        mut world: PhysicsWorld,
        dt: Float32,
    ):
        """Updates world transform and drives any attached kinematic `RigidBody` (`kinematic_body_id`)
        with exact `(pos, rot, linear_vel, angular_vel)`."""
        var had_prev = self.has_prev
        self.update(skel, placement)

        if self.kinematic_body_id < 0:
            return
        var slot = world.find_slot_by_id(self.kinematic_body_id)
        if slot < 0 or slot >= len(world.bodies):
            return
        if had_prev and dt > Float32(1.0e-6):
            # Drive kinematic collider via velocity from its current physics pose to `self.world_pos/rot`
            # so `world.step(dt)` integrates it smoothly to the exact target without double-advancing.
            var inv_dt = Float32(1.0) / dt
            var cur_pos = world.bodies[slot].pos
            var cur_rot = world.bodies[slot].rot
            world.bodies[slot].linear_vel = (self.world_pos - cur_pos) * inv_dt
            var dq = self.world_rot.multiply(cur_rot.conjugate()).normalized()
            world.bodies[slot].angular_vel = det_quat_to_scaled_angle(dq) * inv_dt
        else:
            world.bodies[slot].pos = self.world_pos
            world.bodies[slot].rot = self.world_rot
            world.bodies[slot].linear_vel = Vec3()
            world.bodies[slot].angular_vel = Vec3()
        world.bodies[slot].awake = True


def update_socket_attachments(
    mut sockets: List[BoneSocketAttachment],
    skel: SkeletonInstance,
    placement: Placement,
):
    """Batch-propagates all `BoneSocketAttachment` world transforms for a skeleton."""
    for i in range(len(sockets)):
        sockets[i].update(skel, placement)


# =========================================================================
# 3. `solve_foot_plant_world` Auto-Raycasting `PhysicsWorld`
# =========================================================================

def solve_foot_plant_world(
    mut solver: FootPlantSolver,
    world: PhysicsWorld,
    mut skel: SkeletonInstance,
    model: NativeModel,
    placement: Placement,
    in_stance: List[Bool],
    dt: Float32,
    ray_up: Float32 = 0.65,
    ray_down: Float32 = 1.25,
    filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
    ignore_body_id: Int = -1,
) -> Int:
    """Auto-raycasts `PhysicsWorld` downward for every leg in `solver.legs` to gather
    terrain/deck heights & normals, then applies `FootPlantSolver.apply_to_skeleton`.
    Returns the number of legs whose downward ray hit world geometry."""
    var leg_count = len(solver.legs)
    if leg_count == 0:
        return 0
    skel.ensure_topology(model)
    skel.update_matrices(model)

    var scale_y = max(Float32(0.01), abs(placement.scale.y))
    var hit_heights = List[Float32](capacity=leg_count)
    var hit_normals = List[Vec3](capacity=leg_count)
    var total_dist = max(Float32(0.05), ray_up + ray_down)
    var hits_found = 0

    for f in range(leg_count):
        var ankle_node = solver.legs[f].ankle
        var p_anim = placement.to_world(skel.global_position(ankle_node))
        var stance = in_stance[f] if f < len(in_stance) else False
        var cast_x = p_anim.x
        var cast_z = p_anim.z
        if stance and solver.legs[f].locked:
            cast_x = solver.legs[f].locked_world_pos.x
            cast_z = solver.legs[f].locked_world_pos.z

        var origin = Vec3(cast_x, p_anim.y + ray_up, cast_z)
        var hit = world.raycast(origin, Vec3(0.0, -1.0, 0.0), total_dist, filter, ignore_body_id)
        if hit.hit:
            hit_heights.append(hit.point.y)
            hit_normals.append(hit.normal)
            hits_found += 1
        else:
            hit_heights.append(p_anim.y - solver.legs[f].ankle_height * scale_y)
            hit_normals.append(Vec3(0.0, 1.0, 0.0))

    solver.apply_to_skeleton(skel, model, placement, in_stance, hit_heights, hit_normals, dt)
    return hits_found


# =========================================================================
# 4. `RagdollStateMachine` (`Animated <-> ActivePD <-> Limp <-> Supine/Prone GetUp`)
# =========================================================================

struct RagdollStateMachine(Copyable, Movable):
    """Unified character ragdoll & recovery state machine integrated with `PhysicsWorld`:
    - `RAGDOLL_SM_ANIMATED` (0): Pure animation graph pose, snapshots matrices for 6-DOF handoff.
    - `RAGDOLL_SM_ACTIVE_PD` (1): Physical stagger / active balance with joint PD motors tracking `anim_pose`.
    - `RAGDOLL_SM_LIMP` (2): Full passive 6-DOF ragdoll colliding against `PhysicsWorld` floor + OBB colliders.
    - `RAGDOLL_SM_GETUP_SUPINE` (3) / `RAGDOLL_SM_GETUP_PRONE` (4): Face-up vs face-down recovery
      re-rooting `(placement.position, placement.yaw)` to the settled pelvis with `0.0 mm` snap
      and blending seamlessly via `Inertializer` back to `RAGDOLL_SM_ANIMATED`."""
    var state: Int
    var ragdoll: ArticulatedRagdoll
    var inertial: Inertializer
    var supine_getup_clip: Int
    var prone_getup_clip: Int
    var getup_duration: Float32
    var getup_time: Float32
    var getup_halflife: Float32
    var active_pd_duration: Float32
    var active_pd_timer: Float32
    var min_limp_time: Float32
    var limp_timer: Float32
    var query_padding: Float32
    var filter: CollisionFilter
    var ignore_body_id: Int
    var chest_limb_index: Int
    var last_pelvis_snap_m: Float32
    var recovery_count: Int
    var world_ragdoll_slot: Int
    var _world_dirty: Bool
    var scratch_pose: Pose
    var scratch_ahead: Pose

    def __init__(out self):
        self.state = RAGDOLL_SM_ANIMATED
        self.ragdoll = ArticulatedRagdoll()
        self.inertial = Inertializer()
        self.supine_getup_clip = -1
        self.prone_getup_clip = -1
        self.getup_duration = Float32(0.75)
        self.getup_time = Float32(0.0)
        self.getup_halflife = Float32(0.14)
        self.active_pd_duration = Float32(0.45)
        self.active_pd_timer = Float32(0.0)
        self.min_limp_time = Float32(0.25)
        self.limp_timer = Float32(0.0)
        self.query_padding = Float32(0.65)
        self.filter = CollisionFilter(CAT_ALL, CAT_ALL, 0)
        self.ignore_body_id = -1
        self.chest_limb_index = 1
        self.last_pelvis_snap_m = Float32(0.0)
        self.recovery_count = 0
        self.world_ragdoll_slot = -1
        self._world_dirty = False
        self.scratch_pose = Pose()
        self.scratch_ahead = Pose()

    def is_supine(self) -> Bool:
        """Returns `True` when the settled ragdoll is face-up (`+Z` chest/pelvis forward
        has world `y >= 0`), or `False` when face-down (prone)."""
        if len(self.ragdoll.states) == 0:
            return True
        var idx = self.chest_limb_index
        if idx < 0 or idx >= len(self.ragdoll.states):
            idx = 0
        var rest_inv = self.ragdoll.pelvis_rest_global_rot.conjugate()
        var local_fwd = rest_inv.rotate(Vec3(0.0, 0.0, 1.0))
        var world_fwd = self.ragdoll.states[idx].rotation.rotate(local_fwd)
        return world_fwd.y >= Float32(0.0)

    def sync_physics_environment(mut self, world: PhysicsWorld):
        """Queries `PhysicsWorld` via `raycast` (floor plane under pelvis) and `overlap_aabb`
        (nearby static/kinematic/dynamic box/sphere/capsule/convex/compound colliders)."""
        if len(self.ragdoll.limbs) == 0:
            return
        var pelvis_w = self.ragdoll.pelvis_world_position()
        var floor_hit = world.raycast(
            pelvis_w + Vec3(0.0, 0.85, 0.0),
            Vec3(0.0, -1.0, 0.0),
            Float32(12.0),
            self.filter,
            self.ignore_body_id,
        )
        if floor_hit.hit and floor_hit.normal.y > Float32(0.2):
            var plane_off = floor_hit.normal.dot(floor_hit.point)
            var fric = Float32(0.6)
            if floor_hit.body_slot >= 0 and floor_hit.body_slot < len(world.bodies):
                fric = world.bodies[floor_hit.body_slot].friction
            self.ragdoll.set_surface_plane(floor_hit.normal, plane_off, fric)

        # Gather world AABB around all ragdoll capsules
        var lo = pelvis_w
        var hi = pelvis_w
        for i in range(len(self.ragdoll.limbs)):
            var c = self.ragdoll.states[i].com_pos
            var r = (self.ragdoll.limbs[i].radius + Float32(0.5) * self.ragdoll.limbs[i].length) * self.ragdoll.scale
            var rv = Vec3(r, r, r)
            var b_lo = c - rv
            var b_hi = c + rv
            lo = Vec3(min(lo.x, b_lo.x), min(lo.y, b_lo.y), min(lo.z, b_lo.z))
            hi = Vec3(max(hi.x, b_hi.x), max(hi.y, b_hi.y), max(hi.z, b_hi.z))
        var pad = Vec3(self.query_padding, self.query_padding, self.query_padding)
        var slots = world.overlap_aabb(lo - pad, hi + pad, self.filter, self.ignore_body_id)

        self.ragdoll.clear_obb_colliders()
        var floor_slot = floor_hit.body_slot if floor_hit.hit else -1
        for k in range(len(slots)):
            var s = slots[k]
            if s < 0 or s >= len(world.bodies):
                continue
            ref b = world.bodies[s]
            # Skip large flat floor box already represented by the exact floor plane
            if s == floor_slot and b.is_static and b.half_extents.x > Float32(8.0) and b.half_extents.z > Float32(8.0):
                continue
            if b.shape == BODY_SHAPE_BOX:
                self.ragdoll.add_obb_collider(
                    RagdollObbCollider(b.pos, b.rot, b.half_extents, b.friction, b.restitution)
                )
            elif b.shape == BODY_SHAPE_SPHERE:
                var sr = Vec3(b.radius, b.radius, b.radius)
                self.ragdoll.add_obb_collider(
                    RagdollObbCollider(b.pos, b.rot, sr, b.friction, b.restitution)
                )
            elif b.shape == BODY_SHAPE_CAPSULE:
                var ce = Vec3(b.radius, b.half_height + b.radius, b.radius)
                self.ragdoll.add_obb_collider(
                    RagdollObbCollider(b.pos, b.rot, ce, b.friction, b.restitution)
                )
            elif b.shape == BODY_SHAPE_CONVEX_HULL and b.shape_index >= 0 and b.shape_index < len(world.hulls):
                var he = _hull_half_extents(world.hulls[b.shape_index])
                self.ragdoll.add_obb_collider(
                    RagdollObbCollider(b.pos, b.rot, he, b.friction, b.restitution)
                )
            elif b.shape == BODY_SHAPE_COMPOUND and b.shape_index >= 0 and b.shape_index < len(world.compounds):
                ref cmp = world.compounds[b.shape_index]
                for ci in range(len(cmp.children)):
                    ref ch = cmp.children[ci]
                    var wp_wr = ch.world_pose(b.pos, b.rot)
                    if ch.shape == COMPOUND_CHILD_BOX:
                        self.ragdoll.add_obb_collider(
                            RagdollObbCollider(wp_wr[0], wp_wr[1], ch.half_extents, b.friction, b.restitution)
                        )
                    elif ch.shape == COMPOUND_CHILD_SPHERE:
                        var cr = Vec3(ch.radius, ch.radius, ch.radius)
                        self.ragdoll.add_obb_collider(
                            RagdollObbCollider(wp_wr[0], wp_wr[1], cr, b.friction, b.restitution)
                        )
                    elif ch.shape == COMPOUND_CHILD_CAPSULE:
                        var ce2 = Vec3(ch.radius, ch.half_height + ch.radius, ch.radius)
                        self.ragdoll.add_obb_collider(
                            RagdollObbCollider(wp_wr[0], wp_wr[1], ce2, b.friction, b.restitution)
                        )
                    elif ch.shape == COMPOUND_CHILD_CONVEX:
                        var che = _hull_half_extents(ch.hull)
                        self.ragdoll.add_obb_collider(
                            RagdollObbCollider(wp_wr[0], wp_wr[1], che, b.friction, b.restitution)
                        )

    def enter_active_pd(
        mut self,
        skel: SkeletonInstance,
        placement: Placement,
        model: NativeModel,
        dt: Float32,
        motor_scale: Float32 = 1.0,
        duration: Float32 = 0.45,
    ):
        """Transitions into Active-PD physical stagger/tracking, preserving 6-DOF skeleton velocity."""
        if self.state != RAGDOLL_SM_ACTIVE_PD and self.state != RAGDOLL_SM_LIMP:
            self.ragdoll.handoff_from_skeleton(skel, placement, model, dt)
        self.ragdoll.set_active_pd(motor_scale)
        self.state = RAGDOLL_SM_ACTIVE_PD
        self.active_pd_duration = max(Float32(0.05), duration)
        self.active_pd_timer = Float32(0.0)
        self._world_dirty = True

    def enter_limp(
        mut self,
        skel: SkeletonInstance,
        placement: Placement,
        model: NativeModel,
        dt: Float32,
    ):
        """Transitions into full passive 6-DOF limp ragdoll."""
        if self.state != RAGDOLL_SM_ACTIVE_PD and self.state != RAGDOLL_SM_LIMP:
            self.ragdoll.handoff_from_skeleton(skel, placement, model, dt)
        self.ragdoll.go_limp()
        self.state = RAGDOLL_SM_LIMP
        self.limp_timer = Float32(0.0)
        self._world_dirty = True

    def apply_hit_impulse(
        mut self,
        skel: SkeletonInstance,
        placement: Placement,
        model: NativeModel,
        bone_node: Int,
        world_impulse: Vec3,
        world_point: Vec3,
        go_limp_threshold: Float32 = 120.0,
        dt: Float32 = 1.0 / 60.0,
    ):
        """Applies a ballistic/melee impulse: enters `ACTIVE_PD` below `go_limp_threshold`,
        or knocks down into `LIMP` at or above `go_limp_threshold`."""
        if world_impulse.length() >= go_limp_threshold:
            self.enter_limp(skel, placement, model, dt)
        elif self.state != RAGDOLL_SM_LIMP:
            self.enter_active_pd(skel, placement, model, dt, Float32(0.85), self.active_pd_duration)
        self.ragdoll.apply_impulse_at_bone(bone_node, world_impulse, world_point)
        self._world_dirty = True

    def begin_getup(
        mut self,
        mut skel: SkeletonInstance,
        mut placement: Placement,
        model: NativeModel,
        fallback_pose: Pose,
    ):
        """Classifies settled ragdoll as supine vs prone, re-roots `placement` to the settled
        pelvis with `0.0 mm` world pelvis snap, and seeds `Inertializer` into the getup clip."""
        if len(self.ragdoll.limbs) == 0:
            self.state = RAGDOLL_SM_ANIMATED
            return

        var pelvis_before = self.ragdoll.pelvis_world_position()
        var supine = self.is_supine()
        var chosen_clip = self.supine_getup_clip if supine else self.prone_getup_clip

        if chosen_clip >= 0 and chosen_clip < len(model.animations):
            sample_pose(model, chosen_clip, Float32(0.0), self.scratch_pose)
            self.getup_duration = max(Float32(0.25), model.animations[chosen_clip].duration)
        elif fallback_pose.count() == len(model.nodes):
            self.scratch_pose.assign(fallback_pose)
            self.getup_duration = max(Float32(0.25), self.getup_duration)
        else:
            self.scratch_pose.rest(model)
            self.getup_duration = max(Float32(0.25), self.getup_duration)

        var root_pos = placement.position
        var root_yaw_rad = placement.yaw * DEG
        self.ragdoll.begin_recovery(root_pos, root_yaw_rad, self.inertial, self.scratch_pose)
        placement.position = root_pos
        placement.yaw = root_yaw_rad / DEG

        # Write exact continuity pose (`inertial.out_t/r/s`) into `skel` for frame-0 zero-snap verification
        skel.ensure_topology(model)
        skel.pose.resize(len(model.nodes))
        for idx in range(min(skel.pose.count(), len(self.inertial.out_t))):
            skel.pose.t[idx] = self.inertial.out_t[idx]
            skel.pose.r[idx] = self.inertial.out_r[idx]
            skel.pose.s[idx] = self.inertial.out_s[idx]
        skel.update_matrices(model)

        var pelvis_node = self.ragdoll.limbs[0].bone_node
        var pelvis_after = placement.to_world(skel.global_position(pelvis_node))
        self.last_pelvis_snap_m = (pelvis_after - pelvis_before).length()

        self.getup_time = Float32(0.0)
        self.recovery_count += 1
        self.state = RAGDOLL_SM_GETUP_SUPINE if supine else RAGDOLL_SM_GETUP_PRONE

    def update(
        mut self,
        world: PhysicsWorld,
        mut skel: SkeletonInstance,
        mut placement: Placement,
        model: NativeModel,
        anim_pose: Pose,
        dt: Float32,
        auto_recover: Bool = True,
        substeps: Int = 4,
    ):
        """Advances the ragdoll state machine one tick and writes the resulting pose to `skel`."""
        skel.ensure_topology(model)
        if skel.node_count > 0:
            skel.snapshot_prev_matrices()

        if self.state == RAGDOLL_SM_ANIMATED:
            if anim_pose.count() == len(model.nodes):
                self.scratch_pose.assign(anim_pose)
            else:
                self.scratch_pose.rest(model)
            self.scratch_ahead.assign(self.scratch_pose)
            self.inertial.apply(self.scratch_pose, self.scratch_ahead, False, self.getup_halflife, dt)
            skel.apply_pose(self.scratch_pose, model)
            return

        if self.state == RAGDOLL_SM_ACTIVE_PD:
            self.sync_physics_environment(world)
            if anim_pose.count() == len(model.nodes):
                self.ragdoll.set_target_pose(anim_pose, placement.yaw * DEG)
            self.ragdoll.step(dt, substeps)
            self.ragdoll.write_to_skeleton(skel, model, placement)
            self.active_pd_timer += max(Float32(0.0), dt)
            if auto_recover and self.active_pd_timer >= self.active_pd_duration:
                # Re-root smoothly from Active-PD pose back onto `anim_pose` via Inertializer
                if anim_pose.count() == len(model.nodes):
                    self.scratch_pose.assign(anim_pose)
                else:
                    self.scratch_pose.assign(self.ragdoll.rest_pose)
                var root_pos = placement.position
                var root_yaw_rad = placement.yaw * DEG
                var pelvis_before = self.ragdoll.pelvis_world_position()
                self.ragdoll.begin_recovery(root_pos, root_yaw_rad, self.inertial, self.scratch_pose)
                placement.position = root_pos
                placement.yaw = root_yaw_rad / DEG
                skel.pose.resize(len(model.nodes))
                for idx in range(min(skel.pose.count(), len(self.inertial.out_t))):
                    skel.pose.t[idx] = self.inertial.out_t[idx]
                    skel.pose.r[idx] = self.inertial.out_r[idx]
                    skel.pose.s[idx] = self.inertial.out_s[idx]
                skel.update_matrices(model)
                var pelvis_node = self.ragdoll.limbs[0].bone_node
                self.last_pelvis_snap_m = (placement.to_world(skel.global_position(pelvis_node)) - pelvis_before).length()
                self.state = RAGDOLL_SM_ANIMATED
            return

        if self.state == RAGDOLL_SM_LIMP:
            self.sync_physics_environment(world)
            self.ragdoll.step(dt, substeps)
            self.ragdoll.write_to_skeleton(skel, model, placement)
            self.limp_timer += max(Float32(0.0), dt)
            if auto_recover and self.limp_timer >= self.min_limp_time and self.ragdoll.settled():
                self.begin_getup(skel, placement, model, anim_pose)
            return

        # RAGDOLL_SM_GETUP_SUPINE or RAGDOLL_SM_GETUP_PRONE
        var is_sup = self.state == RAGDOLL_SM_GETUP_SUPINE
        var clip_idx = self.supine_getup_clip if is_sup else self.prone_getup_clip
        self.getup_time += max(Float32(0.0), dt)

        if clip_idx >= 0 and clip_idx < len(model.animations):
            var dur = max(Float32(0.01), model.animations[clip_idx].duration)
            var t_clamped = min(dur, self.getup_time)
            var t_ahead = min(dur, t_clamped + max(Float32(0.001), dt))
            sample_pose(model, clip_idx, t_clamped, self.scratch_pose)
            sample_pose(model, clip_idx, t_ahead, self.scratch_ahead)
        elif anim_pose.count() == len(model.nodes):
            self.scratch_pose.assign(anim_pose)
            self.scratch_ahead.assign(anim_pose)
        else:
            self.scratch_pose.rest(model)
            self.scratch_ahead.rest(model)

        self.inertial.apply(self.scratch_pose, self.scratch_ahead, False, self.getup_halflife, dt)
        skel.apply_pose(self.scratch_pose, model)

        if self.getup_time >= self.getup_duration:
            if anim_pose.count() == len(model.nodes):
                self.scratch_pose.assign(anim_pose)
                self.scratch_ahead.assign(anim_pose)
                self.inertial.apply(self.scratch_pose, self.scratch_ahead, True, self.getup_halflife, max(Float32(0.001), dt))
            self.state = RAGDOLL_SM_ANIMATED

    def attach_to_world(mut self, mut world: PhysicsWorld) -> Int:
        """Registers or syncs `self.ragdoll` into `world` as hosted capsule bodies + `SixDofMotorJoint`s."""
        if self.world_ragdoll_slot >= 0 and self.world_ragdoll_slot < len(world.ragdolls):
            world.sync_ragdoll_from_state(self.world_ragdoll_slot, self.ragdoll)
        else:
            self.world_ragdoll_slot = world.add_ragdoll(self.ragdoll.copy())
            if self.state == RAGDOLL_SM_ANIMATED:
                world.set_ragdoll_active(self.world_ragdoll_slot, False)
        self._world_dirty = False
        return self.world_ragdoll_slot

    def _sync_floor_plane_from_world(mut self, world: PhysicsWorld):
        if len(self.ragdoll.limbs) == 0:
            return
        var pelvis_w = self.ragdoll.pelvis_world_position()
        var floor_hit = world.raycast(
            pelvis_w + Vec3(0.0, 0.85, 0.0),
            Vec3(0.0, -1.0, 0.0),
            Float32(12.0),
            CollisionFilter(CAT_ALL, CAT_TERRAIN | CAT_STATIC, 0),
            self.ignore_body_id,
        )
        if floor_hit.hit and floor_hit.normal.y > Float32(0.2):
            var plane_off = floor_hit.normal.dot(floor_hit.point)
            var fric = Float32(0.6)
            if floor_hit.body_slot >= 0 and floor_hit.body_slot < len(world.bodies):
                fric = world.bodies[floor_hit.body_slot].friction
            self.ragdoll.set_surface_plane(floor_hit.normal, plane_off, fric)

    def update_world(
        mut self,
        mut world: PhysicsWorld,
        mut skel: SkeletonInstance,
        mut placement: Placement,
        model: NativeModel,
        anim_pose: Pose,
        dt: Float32,
        auto_recover: Bool = True,
        step_world: Bool = True,
    ):
        """Advances the ragdoll state machine using two-way `PhysicsWorld`-hosted capsules & joints."""
        skel.ensure_topology(model)
        if skel.node_count > 0:
            skel.snapshot_prev_matrices()

        if self.state == RAGDOLL_SM_ANIMATED:
            if self.world_ragdoll_slot >= 0:
                world.set_ragdoll_active(self.world_ragdoll_slot, False)
            if anim_pose.count() == len(model.nodes):
                self.scratch_pose.assign(anim_pose)
            else:
                self.scratch_pose.rest(model)
            self.scratch_ahead.assign(self.scratch_pose)
            self.inertial.apply(self.scratch_pose, self.scratch_ahead, False, self.getup_halflife, dt)
            skel.apply_pose(self.scratch_pose, model)
            return

        if self.state == RAGDOLL_SM_ACTIVE_PD or self.state == RAGDOLL_SM_LIMP:
            if self.world_ragdoll_slot < 0 or self._world_dirty:
                _ = self.attach_to_world(world)
            world.set_ragdoll_active(self.world_ragdoll_slot, True)

            if self.state == RAGDOLL_SM_ACTIVE_PD and anim_pose.count() == len(model.nodes):
                self.ragdoll.set_target_pose(anim_pose, placement.yaw * DEG)
                var targets = List[Quat](capacity=len(self.ragdoll.states))
                for i in range(len(self.ragdoll.states)):
                    targets.append(self.ragdoll.states[i].target_local_rot)
                world.set_ragdoll_targets(
                    self.world_ragdoll_slot,
                    targets,
                    self.ragdoll.target_root_world_rot,
                    self.ragdoll.motor_scale,
                    False,
                )

            if step_world:
                world.step(dt)

            self.ragdoll = world.ragdoll_state(self.world_ragdoll_slot)
            self.ragdoll.write_to_skeleton(skel, model, placement)

            if self.state == RAGDOLL_SM_ACTIVE_PD:
                self.active_pd_timer += max(Float32(0.0), dt)
                if auto_recover and self.active_pd_timer >= self.active_pd_duration:
                    self._sync_floor_plane_from_world(world)
                    if anim_pose.count() == len(model.nodes):
                        self.scratch_pose.assign(anim_pose)
                    else:
                        self.scratch_pose.assign(self.ragdoll.rest_pose)
                    var root_pos = placement.position
                    var root_yaw_rad = placement.yaw * DEG
                    var pelvis_before = self.ragdoll.pelvis_world_position()
                    self.ragdoll.begin_recovery(root_pos, root_yaw_rad, self.inertial, self.scratch_pose)
                    placement.position = root_pos
                    placement.yaw = root_yaw_rad / DEG
                    skel.pose.resize(len(model.nodes))
                    for idx in range(min(skel.pose.count(), len(self.inertial.out_t))):
                        skel.pose.t[idx] = self.inertial.out_t[idx]
                        skel.pose.r[idx] = self.inertial.out_r[idx]
                        skel.pose.s[idx] = self.inertial.out_s[idx]
                    skel.update_matrices(model)
                    var pelvis_node = self.ragdoll.limbs[0].bone_node
                    self.last_pelvis_snap_m = (placement.to_world(skel.global_position(pelvis_node)) - pelvis_before).length()
                    world.set_ragdoll_active(self.world_ragdoll_slot, False)
                    self.state = RAGDOLL_SM_ANIMATED
            else:
                self.limp_timer += max(Float32(0.0), dt)
                if auto_recover and self.limp_timer >= self.min_limp_time and self.ragdoll.settled():
                    self._sync_floor_plane_from_world(world)
                    self.begin_getup(skel, placement, model, anim_pose)
                    world.set_ragdoll_active(self.world_ragdoll_slot, False)
            return

        if self.world_ragdoll_slot >= 0:
            world.set_ragdoll_active(self.world_ragdoll_slot, False)
        var is_sup = self.state == RAGDOLL_SM_GETUP_SUPINE
        var clip_idx = self.supine_getup_clip if is_sup else self.prone_getup_clip
        self.getup_time += max(Float32(0.0), dt)

        if clip_idx >= 0 and clip_idx < len(model.animations):
            var dur = max(Float32(0.01), model.animations[clip_idx].duration)
            var t_clamped = min(dur, self.getup_time)
            var t_ahead = min(dur, t_clamped + max(Float32(0.001), dt))
            sample_pose(model, clip_idx, t_clamped, self.scratch_pose)
            sample_pose(model, clip_idx, t_ahead, self.scratch_ahead)
        elif anim_pose.count() == len(model.nodes):
            self.scratch_pose.assign(anim_pose)
            self.scratch_ahead.assign(anim_pose)
        else:
            self.scratch_pose.rest(model)
            self.scratch_ahead.rest(model)

        self.inertial.apply(self.scratch_pose, self.scratch_ahead, False, self.getup_halflife, dt)
        skel.apply_pose(self.scratch_pose, model)

        if self.getup_time >= self.getup_duration:
            if anim_pose.count() == len(model.nodes):
                self.scratch_pose.assign(anim_pose)
                self.scratch_ahead.assign(anim_pose)
                self.inertial.apply(self.scratch_pose, self.scratch_ahead, True, self.getup_halflife, max(Float32(0.001), dt))
            self.state = RAGDOLL_SM_ANIMATED


# =========================================================================
# 5. Stepped Limited-Animation Timing (`12/20/24 Hz`) & Smear-Pose Helpers
# =========================================================================

@always_inline
def quantize_stepped_anim_time(time_sec: Float32, step_hz: Float32) -> Float32:
    """Snaps continuous `time_sec` onto discrete limited-animation keyframes (`step_hz > 0`, e.g. `12.0` or `24.0` Hz)
    so character poses hold crisp anime key drawings while cameras & VFX continue running at smooth 60–240 Hz!
    When `step_hz <= 0.0`, returns `time_sec` unquantized."""
    if step_hz <= Float32(0.0):
        return time_sec
    return floor(max(Float32(0.0), time_sec) * step_hz + Float32(1e-5)) / step_hz


struct SteppedAnimClock(Copyable, ImplicitlyCopyable, Movable):
    """Frame-accurate anime limited-animation clock with hitstop freeze frames and `step_hz` quantization."""
    var raw_time_sec: Float32
    var stepped_time_sec: Float32
    var step_hz: Float32
    var hitstop_remaining_sec: Float32
    var pose_changed_this_tick: Bool

    def __init__(out self, step_hz: Float32 = 20.0):
        self.raw_time_sec = Float32(0.0)
        self.stepped_time_sec = Float32(0.0)
        self.step_hz = max(Float32(0.0), step_hz)
        self.hitstop_remaining_sec = Float32(0.0)
        self.pose_changed_this_tick = True

    def trigger_hitstop(mut self, duration_sec: Float32):
        if duration_sec > self.hitstop_remaining_sec:
            self.hitstop_remaining_sec = duration_sec

    def advance(mut self, dt_sec: Float32) -> Float32:
        var dt = max(Float32(0.0), dt_sec)
        if self.hitstop_remaining_sec > Float32(0.0):
            self.hitstop_remaining_sec = max(Float32(0.0), self.hitstop_remaining_sec - dt)
            self.pose_changed_this_tick = False
            return self.stepped_time_sec

        self.raw_time_sec += dt
        var next_stepped = quantize_stepped_anim_time(self.raw_time_sec, self.step_hz)
        self.pose_changed_this_tick = abs(next_stepped - self.stepped_time_sec) > Float32(1e-5)
        self.stepped_time_sec = next_stepped
        return self.stepped_time_sec


def sample_pose_stepped(
    model: NativeModel,
    clip_index: Int,
    time_sec: Float32,
    step_hz: Float32,
    mut out_pose: Pose,
):
    """Samples `clip_index` at quantized limited-animation keyframe time `quantize_stepped_anim_time(time_sec, step_hz)`."""
    var q_time = quantize_stepped_anim_time(time_sec, step_hz)
    sample_pose(model, clip_index, q_time, out_pose)


def compute_volume_preserving_smear_scale(
    stretch_x: Float32,
    min_squash: Float32 = 0.55,
    max_stretch: Float32 = 3.2,
) -> Vec3:
    """Computes volume-preserving `(sx, 1/sqrt(sx), 1/sqrt(sx))` stretch/squash along a bone's primary axis."""
    var sx = max(min_squash, min(max_stretch, stretch_x))
    var inv_perp = max(min_squash, Float32(1.0) / sqrt(max(Float32(0.05), sx)))
    return Vec3(sx, inv_perp, inv_perp)

