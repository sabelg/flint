"""Flint v5 Animation Assets, Ragdoll State Machine, Gameplay AI & Camera Rig Lab (`labs/lab_anim.mojo`).

Executes and benchmarks:
  1. Binary `.flanim` (`FLAN` FourCC) state-machine & 1D/2D blend-tree serializer/loader.
  2. `BoneSocketAttachment` world transform & kinematic `PhysicsWorld` surface-velocity propagation.
  3. `solve_foot_plant_world` auto-raycasting `PhysicsWorld` stepped terrain.
  4. `RagdollStateMachine` (`Animated -> ActivePD -> Limp -> Supine/Prone GetUp`) with `PhysicsWorld`
     collider queries and `0.0 mm` world pelvis snap.
  5. Flat DFS `BehaviorTree` (`BT_ABORT_LOWER_PRIORITY`), `UtilityAiScorer`, SIMD8 `SpatialPerception`
     wall LOS occlusion, and `CameraRig` `SpringArm` anti-clipping + deterministic `trauma^2` shake.
"""
from std.collections import List
from std.time import monotonic
from flint.core.math import Vec3, Quat
from flint.core.ik import angle_axis
from flint.assets.model import NativeModel, NativeNode, NativeAnimation, NativeChannel
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.graph import (
    AnimGraphDef,
    BlendSpaceDef,
    BlendSampleDef,
    BlendSpace2DDef,
    BlendSample2DDef,
    StateDef,
    Condition,
    OP_GT,
)
from flint.anim.procedural import Placement, FootPlantSolver
from flint.anim.flanim_ecs import (
    save_flanim_bytes,
    load_flanim_bytes,
    BoneSocketAttachment,
    solve_foot_plant_world,
    RagdollStateMachine,
    RAGDOLL_SM_ANIMATED,
)
from flint.physics.world import PhysicsWorld
from flint.ai import (
    Blackboard,
    UtilityConsideration,
    UtilityOption,
    UtilityAiScorer,
    BehaviorTree,
    BtExecutionState,
    SpatialPerception,
    BT_RUNNING,
    BT_ABORT_LOWER_PRIORITY,
    BT_CMP_GT,
    CURVE_LINEAR,
    CURVE_LOGISTIC,
)
from flint.camera import CameraRig, CAM_MODE_ORBIT_FOLLOW


def _add_node(mut model: NativeModel, name: String, parent: Int, base_t: Vec3) -> Int:
    var idx = len(model.nodes)
    var node = NativeNode()
    node.name = name
    node.translation = base_t
    node.base_translation = base_t
    node.rotation = Quat()
    node.base_rotation = Quat()
    node.scale = Vec3(1, 1, 1)
    node.base_scale = Vec3(1, 1, 1)
    model.nodes.append(node^)
    if parent >= 0:
        model.nodes[parent].children.append(idx)
    else:
        model.roots.append(idx)
    return idx


def _build_biped_model() -> NativeModel:
    var m = NativeModel()
    var pelvis = _add_node(m, "Pelvis", -1, Vec3(0.0, 0.95, 0.0))
    var chest = _add_node(m, "Chest", pelvis, Vec3(0.0, 0.45, 0.0))
    _ = _add_node(m, "Head", chest, Vec3(0.0, 0.35, 0.0))
    var r_arm = _add_node(m, "UpperArm.R", chest, Vec3(0.28, 0.25, 0.0))
    _ = _add_node(m, "Hand.R", r_arm, Vec3(0.0, -0.50, 0.0))
    var l_hip = _add_node(m, "Thigh.L", pelvis, Vec3(-0.14, 0.0, 0.0))
    var l_knee = _add_node(m, "Shin.L", l_hip, Vec3(0.0, -0.45, 0.0))
    _ = _add_node(m, "Foot.L", l_knee, Vec3(0.0, -0.42, 0.0))
    var r_hip = _add_node(m, "Thigh.R", pelvis, Vec3(0.14, 0.0, 0.0))
    var r_knee = _add_node(m, "Shin.R", r_hip, Vec3(0.0, -0.45, 0.0))
    _ = _add_node(m, "Foot.R", r_knee, Vec3(0.0, -0.42, 0.0))

    for clip_name in [String("GetupSupine"), String("GetupProne")]:
        var anim = NativeAnimation()
        anim.name = clip_name
        anim.duration = 0.45
        var ch = NativeChannel()
        ch.node = pelvis
        ch.path = 0
        ch.times = [Float32(0.0), Float32(0.45)]
        ch.values = [Float32(0.0), Float32(0.95), Float32(0.0), Float32(0.0), Float32(0.95), Float32(0.0)]
        anim.channels.append(ch^)
        m.animations.append(anim^)
    return m^


def main() raises:
    var t_start_ns = monotonic()
    var dt = Float32(1.0 / 60.0)

    # 1. Binary `.flanim` round-trip
    var graph = AnimGraphDef()
    graph.name = "HeroLocomotion"
    graph.params.append("speed")
    graph.params.append("dir_rad")
    var bs1 = BlendSpaceDef()
    bs1.name = "Loco1D"
    bs1.param = 0
    bs1.samples.append(BlendSampleDef(0, 0.0))
    bs1.samples.append(BlendSampleDef(1, 5.5))
    graph.blend_spaces.append(bs1^)
    var bs2 = BlendSpace2DDef()
    bs2.name = "Strafe2D"
    bs2.param_x = 1
    bs2.param_y = 0
    bs2.is_polar = True
    bs2.samples.append(BlendSample2DDef(0, 0.0, 3.0))
    bs2.samples.append(BlendSample2DDef(1, 1.57, 3.0))
    graph.blend_spaces_2d.append(bs2^)
    var st = StateDef()
    st.name = "Move"
    st.blend_2d = 0
    st.when.append(Condition(0, OP_GT, 0.1))
    graph.states.append(st^)

    var flanim_bytes = save_flanim_bytes(graph)
    var loaded_graph = load_flanim_bytes(flanim_bytes)
    print(
        "[lab_anim] 1. .flanim FLAN bytes =", len(flanim_bytes),
        "loaded_name =", loaded_graph.name,
        "states =", len(loaded_graph.states),
        "bs2d =", len(loaded_graph.blend_spaces_2d),
    )

    # 2. BoneSocketAttachment + solve_foot_plant_world + RagdollStateMachine
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(0.0, -0.5, 0.0), Vec3(25.0, 0.5, 25.0), mass=0.0)
    _ = world.add_box(Vec3(0.20, 0.10, 0.0), Vec3(0.35, 0.10, 0.50), mass=0.0)  # raised right-foot step
    var sword_slot = world.add_kinematic_box(Vec3(), Vec3(0.06, 0.45, 0.06))
    var sword_id = world.bodies[sword_slot].id

    var model = _build_biped_model()
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1, 1, 1), 0.0)

    var socket = BoneSocketAttachment("RightHandSword", 4, 1001, Vec3(0.0, -0.25, 0.15), Quat(), Vec3(1, 1, 1), sword_id)
    socket.sync_to_physics(skel, place, world, dt)
    skel.pose.r[3] = angle_axis(-0.7, Vec3(1.0, 0.0, 0.0))
    skel.update_matrices(model)
    socket.sync_to_physics(skel, place, world, dt)

    var fp = FootPlantSolver(0)
    fp.add_leg(5, 6, 7, 0.08)
    fp.add_leg(8, 9, 10, 0.08)
    var stance: List[Bool] = [True, True]
    var hits = 0
    for _ in range(15):
        skel.reset_to_rest(model)
        hits = solve_foot_plant_world(fp, world, skel, model, place, stance, dt)

    var rsm = RagdollStateMachine()
    rsm.supine_getup_clip = 0
    rsm.prone_getup_clip = 1
    _ = rsm.ragdoll.add_limb_from_model(model, 0, -1, 14.0, 0.16, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(model, 1, 0, 12.0, 0.16, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(model, 5, 0, 6.0, 0.10, 0.42, is_hinge=True)
    _ = rsm.ragdoll.add_limb_from_model(model, 8, 0, 6.0, 0.10, 0.42, is_hinge=True)

    var rest_pose = Pose()
    rest_pose.rest(model)
    skel.reset_to_rest(model)
    rsm.apply_hit_impulse(skel, place, model, 1, Vec3(0.0, 20.0, -180.0), Vec3(0.0, 1.35, 0.0), go_limp_threshold=100.0, dt=dt)
    for _ in range(150):
        rsm.update(world, skel, place, model, rest_pose, dt)
        if rsm.state == RAGDOLL_SM_ANIMATED:
            break

    print(
        "[lab_anim] 2. Socket sword vel =", world.bodies[sword_slot].linear_vel.length(),
        "FootIK hits =", hits,
        "RagdollSM state =", rsm.state,
        "pelvis_snap_mm =", rsm.last_pelvis_snap_m * Float32(1000.0),
    )

    # 3. BehaviorTree Reactive Abort + UtilityAI + SpatialPerception + CameraRig
    var bb = Blackboard()
    var k_threat = bb.def_key("threat", 0.0)
    var k_ammo = bb.def_key("ammo", 0.8)
    var k_dist = bb.def_key("dist", 6.0)

    var scorer = UtilityAiScorer(use_geometric_mean=True)
    var opt_melee = UtilityOption(10, 1.0)
    opt_melee.add_consideration(UtilityConsideration(k_dist, 0.0, 20.0, CURVE_LINEAR, 1.0, True))
    _ = scorer.add_option(opt_melee^)
    var opt_ranged = UtilityOption(20, 1.0)
    opt_ranged.add_consideration(UtilityConsideration(k_ammo, 0.0, 1.0, CURVE_LOGISTIC, 1.2, False))
    _ = scorer.add_option(opt_ranged^)

    var bt = BehaviorTree()
    _ = bt.begin_selector()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_threat, BT_CMP_GT, 0.5, BT_ABORT_LOWER_PRIORITY)
    _ = bt.add_action(1)  # Combat evade/engage
    bt.end_composite()
    _ = bt.add_action(2)  # Long-running patrol
    bt.end_composite()

    var bt_state = BtExecutionState()
    bt_state.set_action_status(2, BT_RUNNING)
    _ = bt.tick(bb, bt_state, dt)
    bb.set_float(k_threat, 0.95)
    _ = bt.tick(bb, bt_state, dt)

    _ = world.add_box(Vec3(0.0, 1.5, 5.0), Vec3(3.0, 1.5, 0.3), mass=0.0)  # occluding wall
    var perc = SpatialPerception()
    _ = perc.add_stimulus(Vec3(0.0, 1.5, 9.0), loudness=25.0, body_id=500, team_id=2)  # behind wall, loud
    _ = perc.add_stimulus(Vec3(4.5, 1.5, 6.0), loudness=0.0, body_id=501, team_id=2)   # unoccluded in FOV
    var sensed = perc.query_perceiver(world, Vec3(0.0, 1.5, 0.0), Vec3(0.0, 0.0, 1.0), fov_deg=110.0)

    var cam = CameraRig(CAM_MODE_ORBIT_FOLLOW)
    cam.spring_arm.target_length = 6.0
    cam.yaw_deg = 180.0  # boom extends toward +Z where the wall at z=5 stands!
    cam.pitch_deg = 0.0
    cam.add_trauma(0.75)
    cam.update(world, Vec3(0.0, 0.0, 2.5), dt)

    var total_ms = Float64(monotonic() - t_start_ns) / 1.0e6
    print(
        "[lab_anim] 3. BT aborts =", bt_state.aborts_triggered,
        "utility_best =", scorer.select_best(bb)[0],
        "sensed =", len(sensed),
        "cam_arm_len =", cam.spring_arm.current_length.value,
        "occluded =", cam.spring_arm.is_occluded,
        "total_ms =", total_ms,
    )
