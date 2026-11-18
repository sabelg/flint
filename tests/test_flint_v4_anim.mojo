"""Comprehensive Verification Suite for Flint v4 Animation, IK, Motion Matching,
RBF Pose-Space Correctives, Helper Joints, Cofactor Skinning, Inertia-Clamped
Spring Bones, and 6-DOF Active-PD `ArticulatedRagdoll` (`-Werror` zero-warning clean).
"""
from std.collections import List
from std.math import abs
from flint.core.detmath import det_sin, det_cos, PI
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4, trs
from flint.core.springs import SpringFloat, SpringVec3, SpringAngle, fast_negexp, halflife_to_damping
from flint.core.ik import (
    angle_axis,
    fabrik,
    solve_two_bone_pole,
    solve_four_bar_linkage,
    three_bone_digitigrade_ik,
    fabrik_constrained,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.model import (
    NativeModel,
    NativeNode,
    NativeAnimation,
    NativeChannel,
    INTERP_STEP,
    INTERP_CUBICSPLINE,
)
from flint.assets.manifest import (
    ManifestBone,
    ManifestVolume,
    HELPER_TWIST,
)
from flint.anim.pose import (
    Pose,
    SkeletonInstance,
    Inertializer,
    sample_pose,
    reparent_socket_inertial,
    det_quat_to_scaled_angle,
)
from flint.anim.sync import ClipInfo
from flint.anim.graph import (
    AnimGraphDef,
    BlendSpace2DDef,
    BlendSample2DDef,
    StateDef,
    ROOT_EXTRACT,
)
from flint.anim.animator import Animator, extract_root_motion
from flint.anim.procedural import Placement, LookAt, Recoil, FootPlantSolver, pose_position
from flint.anim.mecha_ik import FourBarLinkageDriver, DigitigradeLegDriver, make_biped_foot_solver, solve_two_bone_mecha_limb
from flint.anim.motion_matching import (
    MotionFeature,
    MotionDatabase,
    STANCE_GROUND,
    STANCE_AIR_SPACE,
)
from flint.anim.correctives import RbfPoseDriver, swing_twist_y
from flint.anim.helper_joints import HelperJoints
from flint.anim.spring_bones import SpringBones
from flint.anim.skinning import (
    SKIN_LBS,
    PALETTE_FLOATS,
    SkinDraw,
    write_palette_entry,
    skin_vertex,
)
from flint.anim.ragdoll import (
    Ragdoll,
    ragdoll_joints,
    ragdoll_parents,
    ArticulatedRagdoll,
    RagdollObbCollider,
    compute_capsule_inertia,
    _pose_global_rot,
)
from e2e.common.harness import TestContext


def _add_node(
    mut model: NativeModel,
    name: String,
    parent: Int,
    base_t: Vec3,
    base_r: Quat = Quat(),
) -> Int:
    var idx = len(model.nodes)
    var node = NativeNode()
    node.name = name
    node.translation = base_t
    node.base_translation = base_t
    node.rotation = base_r
    node.base_rotation = base_r
    node.scale = Vec3(1, 1, 1)
    node.base_scale = Vec3(1, 1, 1)
    model.nodes.append(node^)
    if parent >= 0:
        model.nodes[parent].children.append(idx)
    else:
        model.roots.append(idx)
    return idx


# =========================================================================
# 1. CUBICSPLINE Hermite Sampling, STEP Final Keyframe, & Exact Spring Halflife
# =========================================================================
def test_1_cubicspline_step_and_springs(mut ctx: TestContext) raises:
    # Exact Holden critical damping: after 1.0 * halflife from rest, 50% of offset remains
    var sp = SpringFloat(10.0)
    sp.update(0.0, 0.25, 0.25)
    ctx.assert_approx(sp.value, 5.0, 0.002, "SpringFloat reaches exact 0.5x at t = halflife")
    ctx.assert_true(halflife_to_damping(0.1) > 33.5, "halflife_to_damping uses 3.356694 / halflife")

    var model = NativeModel()
    var n0 = _add_node(model, "Root", -1, Vec3(0, 0, 0))
    var n1 = _add_node(model, "BoneStep", n0, Vec3(0, 1, 0))

    var anim = NativeAnimation()
    anim.name = "SplineAndStep"
    anim.duration = 2.0

    # CUBICSPLINE translation on n0: t in [0.0, 2.0], p(0) = (0,0,0), v_out(0) = (0,2,4),
    # p(2) = (4,0,8), v_in(2) = (0,-2,4).
    var ch_t = NativeChannel()
    ch_t.node = n0
    ch_t.path = 0
    ch_t.interp_mode = INTERP_CUBICSPLINE
    ch_t.times = [Float32(0.0), Float32(2.0)]
    ch_t.values = [
        Float32(0.0), Float32(0.0), Float32(0.0),   # in_tan 0
        Float32(0.0), Float32(0.0), Float32(0.0),   # v0
        Float32(0.0), Float32(2.0), Float32(4.0),   # out_tan 0
        Float32(0.0), Float32(-2.0), Float32(4.0),  # in_tan 1
        Float32(4.0), Float32(0.0), Float32(8.0),   # v1
        Float32(0.0), Float32(0.0), Float32(0.0),   # out_tan 1
    ]
    anim.channels.append(ch_t^)

    # CUBICSPLINE rotation on n0: 0 deg -> 90 deg about +Y
    var q0 = Quat.identity()
    var q1 = angle_axis(PI * Float32(0.5), Vec3(0, 1, 0))
    var ch_r = NativeChannel()
    ch_r.node = n0
    ch_r.path = 1
    ch_r.interp_mode = INTERP_CUBICSPLINE
    ch_r.times = [Float32(0.0), Float32(2.0)]
    ch_r.values = [
        Float32(0), Float32(0), Float32(0), Float32(0),
        q0.x, q0.y, q0.z, q0.w,
        Float32(0), Float32(0.4), Float32(0), Float32(0),
        Float32(0), Float32(0.4), Float32(0), Float32(0),
        q1.x, q1.y, q1.z, q1.w,
        Float32(0), Float32(0), Float32(0), Float32(0),
    ]
    anim.channels.append(ch_r^)

    # STEP translation on n1: keys at [0.0, 1.0, 2.0] -> values [10, 20, 30] on X
    var ch_step = NativeChannel()
    ch_step.node = n1
    ch_step.path = 0
    ch_step.interp_mode = INTERP_STEP
    ch_step.times = [Float32(0.0), Float32(1.0), Float32(2.0)]
    ch_step.values = [
        Float32(10.0), Float32(1.0), Float32(0.0),
        Float32(20.0), Float32(1.0), Float32(0.0),
        Float32(30.0), Float32(1.0), Float32(0.0),
    ]
    anim.channels.append(ch_step^)
    model.animations.append(anim^)

    var pose = Pose()
    # At s = 0.5 (t = 1.0, dt_seg = 2.0):
    # h00 = 0.5, h10 = 0.125 * 2 = 0.25, h01 = 0.5, h11 = -0.125 * 2 = -0.25
    # x = 0.5*0 + 0.5*4 = 2.0; y = 0.25*2 + (-0.25)*(-2) = 1.0; z = 0.25*4 + 0.5*8 - 0.25*4 = 4.0
    sample_pose(model, 0, Float32(1.0), pose)
    ctx.assert_vec3_approx(pose.t[n0], Vec3(2.0, 1.0, 4.0), 0.001, "CUBICSPLINE Hermite midpoint translation")
    ctx.assert_approx(pose.r[n0].length(), 1.0, 0.0005, "CUBICSPLINE Hermite quaternion normalized")
    ctx.assert_approx(pose.t[n1].x, 20.0, 0.0005, "STEP channel hits middle keyframe at t == 1.0")

    sample_pose(model, 0, Float32(0.99), pose)
    ctx.assert_approx(pose.t[n1].x, 10.0, 0.0005, "STEP channel holds key 0 prior to key 1")

    sample_pose(model, 0, Float32(2.0), pose)
    ctx.assert_approx(pose.t[n1].x, 30.0, 0.0005, "STEP channel reaches final keyframe at t == duration")

    # Exercise SpringVec3, SpringAngle, fast_negexp, and parse_native_glb
    var sv3 = SpringVec3(Vec3(4, 0, 0))
    sv3.update(Vec3(), 0.25, 0.25)
    ctx.assert_approx(sv3.value.x, 2.0, 0.01, "SpringVec3 reaches 0.5x at halflife")
    var sa = SpringAngle(170.0)
    sa.update(-170.0, 0.15, 0.05)
    ctx.assert_true(fast_negexp(1.0) > 0.35 and fast_negexp(1.0) < 0.38, "fast_negexp Padé accuracy")
    var empty_glb = parse_native_glb(List[UInt8]())
    ctx.assert_false(empty_glb.loaded(), "parse_native_glb handles empty bytes safely")


# =========================================================================
# 2. Zero-Alloc Topological SkeletonInstance & snapshot_prev_matrices()
# =========================================================================
def test_2_skeleton_instance_topo_and_prev_matrices(mut ctx: TestContext) raises:
    # Construct out-of-order glTF node array: child index 0 has parent index 2, parent 2 has root index 1!
    var model = NativeModel()
    for _ in range(3):
        var nd = NativeNode()
        model.nodes.append(nd^)
    model.nodes[0].name = "LeafHand"
    model.nodes[0].base_translation = Vec3(0, 0, 0.5)
    model.nodes[0].translation = Vec3(0, 0, 0.5)

    model.nodes[1].name = "RootPelvis"
    model.nodes[1].base_translation = Vec3(0, 1.0, 0)
    model.nodes[1].translation = Vec3(0, 1.0, 0)
    model.nodes[1].children.append(2)
    model.roots.append(1)

    model.nodes[2].name = "UpperArm"
    model.nodes[2].base_translation = Vec3(0.4, 0.3, 0)
    model.nodes[2].translation = Vec3(0.4, 0.3, 0)
    model.nodes[2].children.append(0)

    var rev_before = model.transforms_revision
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)

    # Verify topological order puts 1 before 2 and 2 before 0
    var pos_1 = -1
    var pos_2 = -1
    var pos_0 = -1
    for i in range(len(skel.topo_order)):
        if skel.topo_order[i] == 1:
            pos_1 = i
        elif skel.topo_order[i] == 2:
            pos_2 = i
        elif skel.topo_order[i] == 0:
            pos_0 = i
    ctx.assert_true(pos_1 < pos_2 and pos_2 < pos_0, "topo_order sorts parent before child even when child index < parent")
    ctx.assert_vec3_approx(skel.global_position(0), Vec3(0.4, 1.3, 0.5), 0.001, "FK leaf position in rest pose")

    # Snapshot prev matrices, then rotate UpperArm 90 deg about +Y on skel without mutating model
    skel.snapshot_prev_matrices()
    ctx.assert_true(skel.has_prev, "snapshot_prev_matrices sets has_prev = True")
    skel.pose.r[2] = angle_axis(PI * Float32(0.5), Vec3(0, 1, 0))
    skel.update_matrices(model)

    var prev_leaf = Vec3(
        skel.prev_world_matrices[0].at(0, 3),
        skel.prev_world_matrices[0].at(1, 3),
        skel.prev_world_matrices[0].at(2, 3),
    )
    ctx.assert_vec3_approx(prev_leaf, Vec3(0.4, 1.3, 0.5), 0.001, "prev_world_matrices preserves frame n-1 leaf")
    ctx.assert_vec3_approx(skel.global_position(0), Vec3(0.9, 1.3, 0.0), 0.001, "world_matrices reflects frame n rotated leaf")
    ctx.assert_equal(model.transforms_revision, rev_before, "SkeletonInstance never mutates shared NativeModel")


# =========================================================================
# 3. Bollo Scalar-Axis Inertializer, Early-Out, & reparent_socket_inertial
# =========================================================================
def test_3_bollo_inertializer_and_socket_reparent(mut ctx: TestContext) raises:
    var model = NativeModel()
    var root = _add_node(model, "Pelvis", -1, Vec3(0, 1, 0))
    var belly = _add_node(model, "BellyHardpoint", root, Vec3(0, -0.2, 0.25))
    var r_hand = _add_node(model, "Hand.R", root, Vec3(-0.45, 0.15, 0.35), angle_axis(0.4, Vec3(1, 0, 0)))
    var gunpod = _add_node(model, "Gunpod", belly, Vec3(0, 0, 0.1))

    # 3A. Bollo scalar-axis monotonic angle decay on near-170-deg rotation jump with opposing velocity
    var inert = Inertializer()
    var p_a = Pose(4)
    var p_b = Pose(4)
    var p_ahead = Pose(4)
    p_a.r[0] = angle_axis(2.95, Vec3(0, 1, 0))
    p_b.r[0] = Quat.identity()
    # Ahead pose moving AWAY in opposite direction to stress velocity clamping
    p_ahead.r[0] = angle_axis(-0.25, Vec3(0, 1, 0))

    var work = Pose()
    work.assign(p_a)
    inert.apply(work, p_a, False, 0.08, Float32(1.0 / 60.0))
    # Advance one frame with aggressive angular velocity before switching
    p_a.r[0] = angle_axis(3.02, Vec3(0, 1, 0))
    work.assign(p_a)
    inert.apply(work, p_a, False, 0.08, Float32(1.0 / 60.0))

    # Trigger switch to p_b
    work.assign(p_b)
    inert.apply(work, p_ahead, True, 0.08, Float32(1.0 / 60.0))
    ctx.assert_false(inert.is_settled, "inertializer active immediately after switch")

    var prev_ang = inert.rot_ang[0]
    ctx.assert_true(prev_ang > 2.0 and prev_ang <= PI, "initial Bollo scalar angle bounded in (2.0, pi]")
    for _ in range(90):
        work.assign(p_b)
        inert.apply(work, p_b, False, 0.08, Float32(1.0 / 60.0))
        var cur_ang = inert.rot_ang[0]
        ctx.assert_true(cur_ang <= prev_ang + 1.0e-5, "Bollo scalar angle decays monotonically without overshoot")
        ctx.assert_true(cur_ang >= 0.0, "Bollo scalar angle never crosses negative")
        prev_ang = cur_ang

    ctx.assert_true(inert.is_settled, "inertializer enters zero-trig is_settled early-out once converged")
    ctx.assert_approx(inert.largest_offset(), 0.0, 1.0e-5, "settled largest_offset is 0")

    # 3B. Weapon socket re-parenting (VF-1 Gunpod ventral belly -> Hand.R)
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var old_gun_world = skel.world_matrices[gunpod].copy()
    var old_gun_pos = Vec3(old_gun_world.at(0, 3), old_gun_world.at(1, 3), old_gun_world.at(2, 3))

    # Re-parent topology in model/skeleton from belly to r_hand
    model.nodes[belly].children.clear()
    model.nodes[r_hand].children.append(gunpod)
    skel.parents[gunpod] = r_hand

    var hand_world = skel.world_matrices[r_hand].copy()
    var grip_offset = trs(Vec3(0.0, 0.0, 0.20), angle_axis(-0.2, Vec3(1, 0, 0)), Vec3(1, 1, 1))
    var sock_inert = Inertializer()
    reparent_socket_inertial(skel, sock_inert, gunpod, hand_world, grip_offset)

    var new_gun_world_t0 = skel.world_matrices[gunpod].copy()
    var new_gun_pos_t0 = Vec3(new_gun_world_t0.at(0, 3), new_gun_world_t0.at(1, 3), new_gun_world_t0.at(2, 3))
    ctx.assert_vec3_approx(new_gun_pos_t0, old_gun_pos, 0.0005, "reparent_socket_inertial has 0.0mm world pop at handoff")
    ctx.assert_false(sock_inert.is_settled, "socket inertializer seeded with continuity offset")

    # Decay socket inertializer toward target grip_offset
    var tgt_pose = Pose()
    tgt_pose.assign(skel.pose)
    tgt_pose.t[gunpod] = Vec3(0.0, 0.0, 0.20)
    tgt_pose.r[gunpod] = angle_axis(-0.2, Vec3(1, 0, 0))
    for _ in range(60):
        var step_pose = Pose()
        step_pose.assign(tgt_pose)
        sock_inert.apply(step_pose, tgt_pose, False, 0.06, Float32(1.0 / 60.0))
        skel.apply_pose(step_pose, model)

    var expected_final = hand_world.multiply(grip_offset)
    var exp_pos = Vec3(expected_final.at(0, 3), expected_final.at(1, 3), expected_final.at(2, 3))
    ctx.assert_vec3_approx(skel.global_position(gunpod), exp_pos, 0.002, "reparented gunpod converges smoothly onto hand grip")


# =========================================================================
# 4. BlendSpace2DDef Johansen Gradient-Band Weights & Animator Root Motion
# =========================================================================
def test_4_blend_space_2d_and_loop_root_motion(mut ctx: TestContext) raises:
    # 4A. Cartesian Gradient-Band 2D Blend Space
    var bs_cart = BlendSpace2DDef()
    bs_cart.name = "Strafe2D"
    bs_cart.param_x = 0
    bs_cart.param_y = 1
    bs_cart.is_polar = False
    bs_cart.samples.append(BlendSample2DDef(0, 0.0, 0.0))    # Idle
    bs_cart.samples.append(BlendSample2DDef(1, 0.0, 4.0))    # RunFwd
    bs_cart.samples.append(BlendSample2DDef(2, 4.0, 0.0))    # StrafeR
    bs_cart.samples.append(BlendSample2DDef(3, -4.0, 0.0))   # StrafeL

    var w_exact = bs_cart.weights(0.0, 4.0)
    ctx.assert_approx(w_exact[1], 1.0, 0.0005, "Johansen 2D hits exact 1.0 at authored sample")
    ctx.assert_approx(w_exact[0] + w_exact[2] + w_exact[3], 0.0, 0.0005, "other samples 0.0 at authored sample")

    var w_mid = bs_cart.weights(1.5, 2.0)
    var sum_mid = w_mid[0] + w_mid[1] + w_mid[2] + w_mid[3]
    ctx.assert_approx(sum_mid, 1.0, 0.0005, "Johansen 2D weights partition unity")
    ctx.assert_true(w_mid[0] >= 0.0 and w_mid[1] > 0.1 and w_mid[2] > 0.1, "interior query blends active neighbours")

    # 4B. Polar Gradient-Band 2D Blend Space across +-pi branch cut
    var bs_pol = BlendSpace2DDef()
    bs_pol.is_polar = True
    bs_pol.samples.append(BlendSample2DDef(0, 0.0, 5.0))
    bs_pol.samples.append(BlendSample2DDef(1, PI * Float32(0.5), 5.0))
    bs_pol.samples.append(BlendSample2DDef(2, PI, 5.0))
    bs_pol.samples.append(BlendSample2DDef(3, -PI * Float32(0.5), 5.0))
    var wp_plus = bs_pol.weights(PI - Float32(0.01), 5.0)
    var wp_minus = bs_pol.weights(-PI + Float32(0.01), 5.0)
    ctx.assert_approx(wp_plus[2], wp_minus[2], 0.01, "polar 2D blend space has zero discontinuity across +-pi")
    ctx.assert_true(wp_plus[2] > 0.98, "polar backward sample dominates near +-pi")

    # 4C. Animator evaluating 2D Blend Space + Loop-Wrap Root Motion Extraction
    var model = NativeModel()
    var root = _add_node(model, "Root", -1, Vec3(0, 0, 0))
    var arm = _add_node(model, "Arm", root, Vec3(0, 1, 0))

    # Clip 0: looping sprint moving Root +6.0m along +Z over 1.0s, and turning +0.5 rad yaw
    var sprint = NativeAnimation()
    sprint.name = "Sprint"
    sprint.duration = 1.0
    var ch_rt = NativeChannel()
    ch_rt.node = root
    ch_rt.path = 0
    ch_rt.times = [Float32(0.0), Float32(1.0)]
    ch_rt.values = [Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(6.0)]
    sprint.channels.append(ch_rt^)
    var ch_rr = NativeChannel()
    ch_rr.node = root
    ch_rr.path = 1
    ch_rr.times = [Float32(0.0), Float32(1.0)]
    var qr_end = angle_axis(0.5, Vec3(0, 1, 0))
    ch_rr.values = [Float32(0), Float32(0), Float32(0), Float32(1), qr_end.x, qr_end.y, qr_end.z, qr_end.w]
    sprint.channels.append(ch_rr^)
    model.animations.append(sprint^)

    # Clip 1: Strafe moving Arm
    var strafe = NativeAnimation()
    strafe.name = "Strafe"
    strafe.duration = 1.0
    var ch_arm = NativeChannel()
    ch_arm.node = arm
    ch_arm.path = 0
    ch_arm.times = [Float32(0.0), Float32(1.0)]
    ch_arm.values = [Float32(1.0), Float32(1.0), Float32(0.0), Float32(1.0), Float32(1.0), Float32(0.0)]
    strafe.channels.append(ch_arm^)
    model.animations.append(strafe^)

    # Direct loop-wrap test: from 0.90s -> 0.15s (0.25s elapsed across wrap => 0.25 * 6.0m = 1.50m!)
    var rm = extract_root_motion(model, 0, Float32(0.90), Float32(0.15), root)
    ctx.assert_vec3_approx(rm[0], Vec3(0.0, 0.0, 1.5), 0.002, "extract_root_motion wraps cleanly across loop boundary (>0.5m)")
    ctx.assert_approx(rm[1], 0.125, 0.005, "extract_root_motion extracts loop-wrapped yaw delta")

    # Animator 2D state evaluation test
    var graph = AnimGraphDef()
    graph.params = [String("vx"), String("vz")]
    var space2d = BlendSpace2DDef()
    space2d.name = "Move2D"
    space2d.param_x = 0
    space2d.param_y = 1
    space2d.samples.append(BlendSample2DDef(0, 0.0, 1.0))
    space2d.samples.append(BlendSample2DDef(1, 1.0, 0.0))
    graph.blend_spaces_2d.append(space2d^)
    var st = StateDef()
    st.name = "Locomotion2D"
    st.blend_2d = 0
    st.root_motion = ROOT_EXTRACT
    graph.states.append(st^)

    var clips = List[ClipInfo]()
    var ci0 = ClipInfo()
    ci0.name = "Sprint"
    ci0.duration = 1.0
    ci0.loop = True
    clips.append(ci0^)
    var ci1 = ClipInfo()
    ci1.name = "Strafe"
    ci1.duration = 1.0
    ci1.loop = True
    clips.append(ci1^)

    var animator = Animator()
    animator.configure(graph, clips)
    animator.set(0, 0.5)
    animator.set(1, 0.5)
    animator.update(model, 1, Float32(0.1))
    ctx.assert_approx(animator.pose.t[arm].x, 0.5, 0.02, "Animator evaluates 50/50 2D blend space state")


# =========================================================================
# 5. Pole IK, Kawamori 4-Bar Linkage, Digitigrade Walker IK, & FootPlantSolver
# =========================================================================
def test_5_ik_pole_fourbar_digitigrade_and_footplant(mut ctx: TestContext) raises:
    # 5A. Pre-bent knee tracking a moving pole vector
    var model = NativeModel()
    var pelvis = _add_node(model, "Pelvis", -1, Vec3(0, 1.0, 0))
    var hip_l = _add_node(model, "Hip.L", pelvis, Vec3(0.2, 0.0, 0.0), angle_axis(-0.35, Vec3(1, 0, 0)))
    var knee_l = _add_node(model, "Knee.L", hip_l, Vec3(0, -0.45, 0), angle_axis(0.70, Vec3(1, 0, 0)))
    var ankle_l = _add_node(model, "Ankle.L", knee_l, Vec3(0, -0.45, 0), angle_axis(-0.35, Vec3(1, 0, 0)))
    var hip_r = _add_node(model, "Hip.R", pelvis, Vec3(-0.2, 0.0, 0.0))
    var knee_r = _add_node(model, "Knee.R", hip_r, Vec3(0, -0.45, 0))
    var ankle_r = _add_node(model, "Ankle.R", knee_r, Vec3(0, -0.45, 0))

    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var foot_goal = Vec3(0.2, 0.28, 0.05)

    # Pole out to +X (lateral knee splay) vs +Z (forward knee)
    var pole_side = Vec3(1.2, 0.60, 0.0)
    skel.solve_two_bone(hip_l, knee_l, ankle_l, foot_goal, pole_side, model)
    ctx.assert_vec3_approx(skel.global_position(ankle_l), foot_goal, 0.002, "solve_two_bone_pole reaches foot goal (side pole)")
    ctx.assert_true(skel.global_position(knee_l).x > 0.35, "pre-bent knee tracks lateral +X pole")

    skel.reset_to_rest(model)
    var pole_fwd = Vec3(0.2, 0.60, 1.2)
    skel.solve_two_bone(hip_l, knee_l, ankle_l, foot_goal, pole_fwd, model)
    ctx.assert_vec3_approx(skel.global_position(ankle_l), foot_goal, 0.002, "solve_two_bone_pole reaches foot goal (forward pole)")
    ctx.assert_true(skel.global_position(knee_l).z > 0.25, "pre-bent knee tracks forward +Z pole")

    # 5B. Kawamori closed-loop 4-bar linkage invariance (< 1e-4 across full sweep)
    var a_piv = Vec3(0.0, 1.2, 0.0)
    var d_piv = Vec3(0.0, 1.2, 0.50)
    var l_ab = Float32(0.22)
    var l_bc = Float32(0.42)
    var l_cd = Float32(0.36)
    var driver_4b = FourBarLinkageDriver(1, 2, 3, a_piv, d_piv, l_ab, l_bc, l_cd, Vec3(1, 0, 0), 1.0)
    for step_i in range(15):
        var crank_ang = Float32(0.15) + Float32(step_i) * Float32(0.12)
        var res4 = solve_four_bar_linkage(a_piv, d_piv, l_ab, l_bc, l_cd, crank_ang, Vec3(1, 0, 0), 1.0)
        var jb = res4[0]
        var jc = res4[1]
        ctx.assert_approx((jb - a_piv).length(), l_ab, 0.0001, "4-bar crank length invariant < 1e-4")
        ctx.assert_approx((jc - jb).length(), l_bc, 0.0001, "4-bar coupler length invariant < 1e-4")
        ctx.assert_approx((jc - d_piv).length(), l_cd, 0.0001, "4-bar rocker length invariant < 1e-4")
    _ = driver_4b.evaluate(0.5)

    # 5C. 3-bone Digitigrade Reverse-Knee Walker IK (Regult / GERWALK)
    var regult = NativeModel()
    var pod = _add_node(regult, "Pod", -1, Vec3(0, 1.8, 0))
    var dg_hip = _add_node(regult, "Hip", pod, Vec3(0.4, -0.2, 0))
    var dg_knee = _add_node(regult, "Knee1", dg_hip, Vec3(0, -0.55, 0.25))
    var dg_ank = _add_node(regult, "AnkleRev", dg_knee, Vec3(0, -0.55, -0.45))
    var dg_foot = _add_node(regult, "Foot", dg_ank, Vec3(0, -0.50, 0.25))

    var dg_skel = SkeletonInstance()
    dg_skel.reset_to_rest(regult)
    var p0 = dg_skel.global_position(dg_hip)
    var p1 = dg_skel.global_position(dg_knee)
    var p2 = dg_skel.global_position(dg_ank)
    var p3 = dg_skel.global_position(dg_foot)
    var dg_tgt = Vec3(0.4, 0.08, 0.15)
    var dg_q = three_bone_digitigrade_ik(p0, p1, p2, p3, dg_tgt, Vec3(0, 0, 1), 0.55)
    dg_skel.pose.r[dg_hip] = dg_q[0].multiply(dg_skel.pose.r[dg_hip]).normalized()
    dg_skel.pose.r[dg_knee] = dg_q[1].multiply(dg_skel.pose.r[dg_knee]).normalized()
    dg_skel.pose.r[dg_ank] = dg_q[2].multiply(dg_skel.pose.r[dg_ank]).normalized()
    dg_skel.update_matrices(regult)

    ctx.assert_vec3_approx(dg_skel.global_position(dg_foot), dg_tgt, 0.003, "digitigrade 3-bone IK reaches ground target")
    ctx.assert_true(dg_skel.global_position(dg_knee).z > dg_skel.global_position(dg_hip).z, "digitigrade knee1 bends forward")
    ctx.assert_true(dg_skel.global_position(dg_ank).z < dg_skel.global_position(dg_knee).z, "digitigrade reverse ankle bends backward")

    var dg_driver = DigitigradeLegDriver(dg_hip, dg_knee, dg_ank, dg_foot, 0.08, 0.5)
    dg_driver.apply_to_skeleton(dg_skel, regult, Placement(), True, 0.0, Vec3(0, 1, 0), Float32(1.0 / 60.0))

    # Also verify zero-alloc FABRIK & constrained FABRIK cone clamp
    var chain: List[Vec3] = [Vec3(0, 0, 0), Vec3(0, 0.5, 0), Vec3(0, 1.0, 0), Vec3(0, 1.5, 0)]
    _ = fabrik(chain, Vec3(0.4, 1.1, 0.0))
    var cones: List[Float32] = [Float32(0.35), Float32(0.35), Float32(0.35)]
    var fab_rots = List[Quat]()
    _ = fabrik_constrained(chain, Vec3(0.4, 1.2, 0.2), cones, fab_rots)
    ctx.assert_equal(len(fab_rots), 4, "fabrik_constrained outputs minimal-twist joint rotations")

    _ = driver_4b.apply_to_skeleton(dg_skel, regult, Float32(0.5))
    var biped_fp = make_biped_foot_solver(pelvis, hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r)
    solve_two_bone_mecha_limb(skel, model, hip_l, knee_l, ankle_l, foot_goal, pole_fwd)
    _ = len(biped_fp.legs)

    var look = LookAt()
    look.apply_to_skeleton(skel, model, hip_l, pelvis, Vec3(1.0, 1.0, 2.0), True, 60.0, 45.0, 0.3, 0.05, Float32(1.0 / 60.0))
    var recoil = Recoil()
    recoil.kick(Vec3(0.0, 0.0, -1.0), 4.0)
    recoil.apply_to_skeleton(skel, model, pelvis, hip_l, 120.0, 14.0, Float32(1.0 / 60.0))
    ctx.assert_true(recoil.offset.length() > 0.0, "analytical closed-form Recoil spring integrates impact kick")

    # 5D. FootPlantSolver: zero stance skate, sloped ankle normal alignment, & pelvis drop
    skel.reset_to_rest(model)
    var fp = FootPlantSolver(pelvis)
    fp.add_leg(hip_l, knee_l, ankle_l, 0.10, Vec3(0, 0.5, 1.0))
    fp.add_leg(hip_r, knee_r, ankle_r, 0.10, Vec3(0, 0.5, 1.0))

    var slope_n = Vec3(0.0, det_cos(0.30), -det_sin(0.30)).normalized()
    var stance_flags: List[Bool] = [True, True]
    var heights: List[Float32] = [Float32(0.0), Float32(-0.12)]  # right foot on 12cm lower step -> pelvis drops!
    var normals: List[Vec3] = [slope_n, Vec3(0, 1, 0)]

    var place0 = Placement(Vec3(0, 0, 0), Vec3(1, 1, 1), 0.0)
    fp.apply_to_skeleton(skel, model, place0, stance_flags, heights, normals, Float32(1.0 / 60.0))
    var locked_l_world = fp.foot_world_position(0, skel, place0)

    # Move character body +0.18m forward while left foot remains in stance -> zero XZ foot skate!
    for step_k in range(12):
        var place_k = Placement(Vec3(0.0, 0.0, Float32(step_k + 1) * 0.015), Vec3(1, 1, 1), 0.0)
        skel.reset_to_rest(model)
        fp.apply_to_skeleton(skel, model, place_k, stance_flags, heights, normals, Float32(1.0 / 60.0))
        var cur_l_world = fp.foot_world_position(0, skel, place_k)
        ctx.assert_approx(cur_l_world.x, locked_l_world.x, 0.002, "stance foot X strictly locked (zero skate)")
        ctx.assert_approx(cur_l_world.z, locked_l_world.z, 0.002, "stance foot Z strictly locked (zero skate)")

    ctx.assert_true(fp.pelvis_offset_y.value < -0.05, "pelvis lowers smoothly to reach lower step")
    var ankle_up = skel.global_rotation(ankle_l).rotate(Vec3(0, 1, 0))
    ctx.assert_true(ankle_up.dot(slope_n) > 0.97, "ankle pitch/roll aligns with 17-deg sloped deck normal")


# =========================================================================
# 6. MotionDatabase Z-Score + SIMD8 AABB Chunk Parity & Anti-Thrash
# =========================================================================
def test_6_motion_matching_zscore_simd8_aabb(mut ctx: TestContext) raises:
    var db = MotionDatabase()
    # Populate 260 frames (> 4 chunks of 64 frames + partial tail SIMD group)
    for i in range(260):
        var fi = Float32(i)
        var feat = MotionFeature()
        var clip_id = i // 65
        var t_s = Float32(i % 65) * Float32(1.0 / 30.0)
        var angle = (fi * Float32(0.07)) - PI
        feat.traj_pos_200 = Vec3(det_sin(angle) * 0.6, 0.0, det_cos(angle) * 0.8)
        feat.traj_pos_400 = feat.traj_pos_200 * 2.0
        feat.traj_pos_800 = feat.traj_pos_200 * 4.0
        feat.traj_facing_200 = angle * Float32(0.25)
        feat.traj_facing_400 = angle * Float32(0.50)
        feat.traj_facing_800 = angle
        feat.left_foot_pos = Vec3(-0.18, 0.05 * abs(det_sin(fi * 0.2)), 0.25 * det_sin(fi * 0.2))
        feat.right_foot_pos = Vec3(0.18, 0.05 * abs(det_cos(fi * 0.2)), -0.25 * det_sin(fi * 0.2))
        feat.hip_vel = Vec3(det_sin(angle) * 3.5, 0.0, det_cos(angle) * 3.5)
        feat.ang_vel = Vec3(0.0, det_sin(fi * 0.05) * 1.2, 0.0)
        feat.stance_tag = STANCE_AIR_SPACE if clip_id == 3 else STANCE_GROUND
        db.add_frame(clip_id, i % 65, t_s, feat)

    db.finalize_and_normalize()
    ctx.assert_true(db.is_finalized and db.chunk_count() == 5, "MotionDatabase builds 5 AABB chunks for 260 frames")

    # Verify exact parity between SIMD8 + AABB branch-and-bound and full unpruned scan across 16 queries
    var total_pruned = 0
    for q_idx in range(16):
        var q = db.frames[q_idx * 15 + 3].feature
        q.traj_pos_400 = q.traj_pos_400 + Vec3(0.05, 0.0, -0.04)
        q.hip_vel = q.hip_vel + Vec3(0.1, 0.0, 0.1)

        var fast_res = db.find_best_match(q)
        total_pruned += db.count_pruned_chunks(q)

        # Brute-force reference over ALL chunks via _eval_chunk_simd8 without AABB pruning
        var q_norm = Array[Float32, 33](fill=0.0)
        db._normalize_query(q, q_norm)
        var brute_idx = 0
        var brute_cost = Float32(1.0e30)
        for c in range(db.chunk_count()):
            db._eval_chunk_simd8(c, q_norm, q.stance_tag, -1, 0.0, -1, 0.0, 0.0, brute_idx, brute_cost)

        ctx.assert_equal(fast_res[0], db.frames[brute_idx].clip_index, "AABB search clip matches brute-force")
        ctx.assert_approx(fast_res[1], db.frames[brute_idx].time_seconds, 1.0e-5, "AABB search time matches brute-force")
        ctx.assert_approx(fast_res[2], brute_cost, 1.0e-4, "AABB search cost matches brute-force")

    ctx.assert_true(total_pruned > 10, "64-frame AABB branch-and-bound prunes chunks cleanly")

    # Verify query_inertial continuing-pose anti-thrash & exclude_window anti-freeze
    var cur_frame = 40
    var q_tie = db.frames[cur_frame].feature
    # Make another clip's frame nearly identical in cost, then verify continuing_bias keeps cur_frame
    var chosen_cont = db.query_inertial(
        q_tie,
        db.frames[cur_frame].clip_index,
        db.frames[cur_frame].time_seconds,
        continuing_bias=0.20,
        exclude_window=0.15,
    )
    ctx.assert_equal(chosen_cont, cur_frame, "query_inertial keeps natural continuation frame via continuing_bias")

    # Also test bake_clip_features on a NativeModel
    var m_bake = NativeModel()
    var r_b = _add_node(m_bake, "Pelvis", -1, Vec3(0, 1, 0))
    var lf_b = _add_node(m_bake, "Foot.L", r_b, Vec3(0.15, -1.0, 0))
    var rf_b = _add_node(m_bake, "Foot.R", r_b, Vec3(-0.15, -1.0, 0))
    var clip_b = NativeAnimation()
    clip_b.name = "Walk"
    clip_b.duration = 0.5
    var ch_b = NativeChannel()
    ch_b.node = r_b
    ch_b.path = 0
    ch_b.times = [Float32(0.0), Float32(0.5)]
    ch_b.values = [Float32(0), Float32(1), Float32(0), Float32(0), Float32(1), Float32(1.5)]
    clip_b.channels.append(ch_b^)
    m_bake.animations.append(clip_b^)
    var db_baked = MotionDatabase()
    db_baked.bake_clip_features(m_bake, 0, lf_b, rf_b, r_b, 20.0, True, STANCE_GROUND)
    ctx.assert_true(db_baked.count() >= 10, "bake_clip_features extracts clip frames")
    ctx.assert_true(db_baked.frames[0].feature.hip_vel.z > 2.5, "baked hip_vel captures +Z forward speed")


# =========================================================================
# 7. RBF Correctives, Proximal/Distal Helpers, Cofactor Normals, & SpringBones
# =========================================================================
def test_7_rbf_helpers_skinning_and_springbones(mut ctx: TestContext) raises:
    # 7A. Gaussian RBF Pose-Space Deformer: exact Kronecker delta at sculpted targets & bounded diagonal
    var rbf = RbfPoseDriver(0, False)
    rbf.add_sculpted_pose(0.85, 0.0, 0)
    rbf.add_sculpted_pose(0.0, 0.85, 1)
    rbf.finalize()

    var w_rest = rbf.evaluate(0.0, 0.0)
    ctx.assert_approx(w_rest[0], 0.0, 0.01, "RBF rest pose morph 0 == 0")
    ctx.assert_approx(w_rest[1], 0.0, 0.01, "RBF rest pose morph 1 == 0")

    var w_t1 = rbf.evaluate(0.85, 0.0)
    ctx.assert_approx(w_t1[0], 1.0, 0.01, "RBF target 1 hits w_0 ~ 1.0")
    ctx.assert_approx(w_t1[1], 0.0, 0.01, "RBF target 1 suppresses w_1 ~ 0.0")

    var w_t2 = rbf.evaluate(0.0, 0.85)
    ctx.assert_approx(w_t2[0], 0.0, 0.01, "RBF target 2 suppresses w_0 ~ 0.0")
    ctx.assert_approx(w_t2[1], 1.0, 0.01, "RBF target 2 hits w_1 ~ 1.0")

    var w_diag = rbf.evaluate(0.60, 0.60)
    var diag_sum = w_diag[0] + w_diag[1]
    ctx.assert_true(diag_sum > 0.65 and diag_sum < 1.25, "RBF diagonal sum stays near 1.0 without 2x linear bulge")

    # 7B. Proximal Counter-Twist + Distal Twist HelperJoints
    var arm_model = NativeModel()
    var clavicle = _add_node(arm_model, "Clavicle", -1, Vec3(0, 1.4, 0))
    var upper_arm = _add_node(arm_model, "UpperArm", clavicle, Vec3(0.2, 0, 0))
    var upper_twist = _add_node(arm_model, "UpperArm_Twist", upper_arm, Vec3(0.05, 0, 0))
    var forearm = _add_node(arm_model, "Forearm", upper_arm, Vec3(0.32, 0, 0))
    var hand = _add_node(arm_model, "Hand", forearm, Vec3(0.28, 0, 0))
    var forearm_twist = _add_node(arm_model, "Forearm_Twist", forearm, Vec3(0.14, 0, 0))

    var manifest_bones = List[ManifestBone]()
    var mb_prox = ManifestBone("UpperArm_Twist", "UpperArm", Vec3(0.25, 1.4, 0))
    mb_prox.helper_kind = HELPER_TWIST
    mb_prox.helper_source = "UpperArm"
    mb_prox.helper_amount = 0.5
    manifest_bones.append(mb_prox^)

    var mb_dist = ManifestBone("Forearm_Twist", "Forearm", Vec3(0.66, 1.4, 0))
    mb_dist.helper_kind = HELPER_TWIST
    mb_dist.helper_source = "Hand"
    mb_dist.helper_amount = 0.5
    manifest_bones.append(mb_dist^)

    var helpers = HelperJoints()
    helpers.bind(arm_model, 1, manifest_bones)
    ctx.assert_equal(len(helpers.problems), 0, "proximal child & distal sibling twist helpers bind with 0 problems")
    ctx.assert_equal(helpers.count(), 2, "both proximal and distal twist helpers bound")

    # Apply combined swing (45 deg about Z) + axial twist (60 deg about shaft +X) to UpperArm, and 80 deg +X roll to Hand
    var swing_z = angle_axis(PI * Float32(0.25), Vec3(0, 0, 1))
    var roll_x = angle_axis(PI / Float32(3.0), Vec3(1, 0, 0))
    arm_model.nodes[upper_arm].rotation = swing_z.multiply(roll_x).normalized()
    arm_model.nodes[hand].rotation = angle_axis(Float32(1.4), Vec3(1, 0, 0))
    helpers.apply(arm_model)

    # UpperArm_Twist world rotation should have 100% of swing_z and (1 - 0.5) = 50% of roll_x!
    var expected_prox_world = swing_z.multiply(angle_axis(PI / Float32(6.0), Vec3(1, 0, 0))).normalized()
    var actual_prox_world = arm_model.global_rotation(upper_twist)
    var prox_err = det_quat_to_scaled_angle(expected_prox_world.multiply(actual_prox_world.conjugate())).length()
    ctx.assert_approx(prox_err, 0.0, 0.002, "proximal counter-twist cancels 50% axial roll while keeping 100% swing")

    var expected_dist_local = angle_axis(Float32(0.7), Vec3(1, 0, 0))
    var dist_err = det_quat_to_scaled_angle(expected_dist_local.multiply(arm_model.nodes[forearm_twist].rotation.conjugate())).length()
    ctx.assert_approx(dist_err, 0.0, 0.002, "distal sibling twist helper takes +50% of hand roll")
    _ = swing_twist_y(arm_model.nodes[upper_arm].rotation, Quat())

    # 7C. Cofactor SKIN_LBS normal transform under non-uniform scale diag(2.0, 0.5, 1.0)
    var palette = List[Float32](length=PALETTE_FLOATS, fill=0.0)
    var m_nonunif = Mat4()
    m_nonunif.set(0, 0, 2.0)
    m_nonunif.set(1, 1, 0.5)
    m_nonunif.set(2, 2, 1.0)
    _ = write_palette_entry(Int(palette.unsafe_ptr()), 0, m_nonunif)

    var draw = SkinDraw()
    draw.palette = Int(palette.unsafe_ptr())
    draw.joint_count = 1
    draw.method = SKIN_LBS
    var joints_v: List[UInt16] = [UInt16(0), UInt16(0), UInt16(0), UInt16(0)]
    var weights_v: List[Float32] = [Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0)]

    # Surface tangent in bind space t = (1, -1, 0), normal n = normalize(1, 1, 0) (t . n == 0).
    # Deformed tangent is M * t = (2.0, -0.5, 0). True normal must be perpendicular to M * t!
    var n_bind = Vec3(1.0, 1.0, 0.0).normalized()
    var skinned_vn = skin_vertex(
        draw,
        Vec3(1.0, 1.0, 0.0),
        n_bind,
        Vec3(),
        Int(joints_v.unsafe_ptr()),
        Int(weights_v.unsafe_ptr()),
    )
    var deformed_tangent = Vec3(2.0, -0.5, 0.0)
    ctx.assert_approx(skinned_vn[1].length(), 1.0, 0.0005, "cofactor skinned normal is unit length")
    ctx.assert_approx(skinned_vn[1].dot(deformed_tangent), 0.0, 0.001, "cofactor normal stays perpendicular under non-uniform scale")
    _ = len(palette) + len(joints_v) + len(weights_v)

    # 7D. Inertia-Clamped SpringBones under a 100m single-frame teleport
    var cape_model = NativeModel()
    var spine_c = _add_node(cape_model, "Spine", -1, Vec3(0, 1.5, 0))
    var cape1 = _add_node(cape_model, "Cape1", spine_c, Vec3(0, 0, -0.15))
    _ = _add_node(cape_model, "Cape2", cape1, Vec3(0, -0.40, 0))

    var sb_bones = List[ManifestBone]()
    var b1 = ManifestBone("Cape1", "Spine", Vec3(0, 1.5, -0.15))
    b1.spring = True
    b1.tail = Vec3(0, 1.1, -0.15)
    b1.stiffness = 12.0
    b1.drag = 0.25
    b1.gravity = 9.81
    sb_bones.append(b1^)

    var b2 = ManifestBone("Cape2", "Cape1", Vec3(0, 1.1, -0.15))
    b2.spring = True
    b2.tail = Vec3(0, 0.7, -0.15)
    b2.stiffness = 10.0
    b2.drag = 0.25
    b2.gravity = 9.81
    sb_bones.append(b2^)

    var vols = List[ManifestVolume]()
    var springs = SpringBones()
    springs.bind(cape_model, 1, sb_bones, vols)
    ctx.assert_equal(springs.count(), 2, "2-link spring chain bound")
    ctx.assert_equal(springs.bones[1].parent_spring, 0, "Cape2 linked to parent spring Cape1 for intra-substep propagation")

    springs.update(cape_model, Placement(Vec3(0, 0, 0)), Vec3(), Float32(1.0 / 60.0))
    # Teleport 100m in 1/60s (6000 m/s apparent velocity!)
    springs.update(cape_model, Placement(Vec3(100.0, 0.0, 0.0)), Vec3(), Float32(1.0 / 60.0))

    var head0_w = Placement(Vec3(100.0, 0.0, 0.0)).to_world(cape_model.global_position(cape1))
    var dist0 = (springs.tails[0] - head0_w).length()
    var vel0_apparent = (springs.tails[0] - springs.previous[0]).length() * Float32(120.0)
    ctx.assert_true(dist0 < springs.bones[0].length * 1.5, "spring bone tail stays within < 1.5x bone length on 100m teleport")
    ctx.assert_approx(dist0, springs.bones[0].length, 0.01, "spring bone preserves exact bone length after teleport")
    ctx.assert_true(vel0_apparent < 60.0, "root inertia clamp prevents 1200+ m/s Verlet launch")

    var cape_skel = SkeletonInstance()
    cape_skel.reset_to_rest(cape_model)
    springs.update_skeleton(cape_skel, cape_model, Placement(Vec3(100.2, 0.0, 0.0)), Vec3(2.0, 0.0, 0.0), Float32(1.0 / 60.0))
    var h_pose = Pose()
    h_pose.read(arm_model)
    helpers.apply_to_pose(h_pose, arm_model)
    var morph_out = List[Float32]()
    rbf.apply_to_pose(h_pose, arm_model, morph_out)


# =========================================================================
# 8. ArticulatedRagdoll (6-DOF Handoff, Active-PD 1G & Zero-G, Knee Hinge,
#    World Pelvis Re-Rooting) + Legacy Ragdoll 120-Deg Getup Heading Fix
# =========================================================================
def _build_battloid_rig() -> NativeModel:
    var m = NativeModel()
    var pelvis = _add_node(m, "Pelvis", -1, Vec3(0, 1.0, 0))
    var chest = _add_node(m, "Chest", pelvis, Vec3(0, 0.45, 0))
    var head = _add_node(m, "Head", chest, Vec3(0, 0.30, 0))
    var uarm_r = _add_node(m, "UpperArm.R", chest, Vec3(-0.28, 0.22, 0))
    var farm_r = _add_node(m, "Forearm.R", uarm_r, Vec3(0, -0.30, 0))
    var hand_r = _add_node(m, "Hand.R", farm_r, Vec3(0, -0.26, 0))
    var thigh_l = _add_node(m, "Thigh.L", pelvis, Vec3(0.18, 0, 0))
    var shin_l = _add_node(m, "Shin.L", thigh_l, Vec3(0, -0.44, 0))
    var foot_l = _add_node(m, "Foot.L", shin_l, Vec3(0, -0.44, 0))
    var thigh_r = _add_node(m, "Thigh.R", pelvis, Vec3(-0.18, 0, 0))
    var shin_r = _add_node(m, "Shin.R", thigh_r, Vec3(0, -0.44, 0))
    var foot_r = _add_node(m, "Foot.R", shin_r, Vec3(0, -0.44, 0))
    _ = head
    _ = hand_r
    _ = foot_l
    _ = foot_r
    return m^


def _configure_battloid_ragdoll(model: NativeModel) -> ArticulatedRagdoll:
    var rag = ArticulatedRagdoll()
    rag.bind_skeleton(model)
    var p_limb = rag.add_limb_from_model(model, 0, -1, 14.0, 0.14, 0.45, Vec3(0, 1, 0))
    var c_limb = rag.add_limb_from_model(
        model, 1, p_limb, 16.0, 0.16, 0.35, Vec3(0, 1, 0),
        is_hinge=False, swing_cone_rad=0.55, active_stiffness=340.0, active_damping=36.0, max_torque=600.0,
    )
    var ua_r = rag.add_limb_from_model(
        model, 3, c_limb, 4.0, 0.08, 0.30, Vec3(0, -1, 0),
        is_hinge=False, swing_cone_rad=1.2, active_stiffness=180.0, active_damping=20.0, max_torque=280.0,
    )
    _ = rag.add_limb_from_model(
        model, 4, ua_r, 3.0, 0.07, 0.26, Vec3(0, -1, 0),
        is_hinge=True, hinge_axis_local=Vec3(-1, 0, 0), hinge_min_rad=0.0, hinge_max_rad=2.40,
        swing_cone_rad=0.01, active_stiffness=150.0, active_damping=16.0, max_torque=220.0,
    )
    var th_l = rag.add_limb_from_model(
        model, 6, p_limb, 8.0, 0.10, 0.44, Vec3(0, -1, 0),
        is_hinge=False, swing_cone_rad=0.95, active_stiffness=320.0, active_damping=34.0, max_torque=520.0,
    )
    _ = rag.add_limb_from_model(
        model, 7, th_l, 5.5, 0.085, 0.44, Vec3(0, -1, 0),
        is_hinge=True, hinge_axis_local=Vec3(1, 0, 0), hinge_min_rad=0.0, hinge_max_rad=2.35,
        swing_cone_rad=0.0, active_stiffness=280.0, active_damping=30.0, max_torque=450.0,
    )
    var th_r = rag.add_limb_from_model(
        model, 9, p_limb, 8.0, 0.10, 0.44, Vec3(0, -1, 0),
        is_hinge=False, swing_cone_rad=0.95, active_stiffness=320.0, active_damping=34.0, max_torque=520.0,
    )
    _ = rag.add_limb_from_model(
        model, 10, th_r, 5.5, 0.085, 0.44, Vec3(0, -1, 0),
        is_hinge=True, hinge_axis_local=Vec3(1, 0, 0), hinge_min_rad=0.0, hinge_max_rad=2.35,
        swing_cone_rad=0.0, active_stiffness=280.0, active_damping=30.0, max_torque=450.0,
    )
    return rag^


def test_8_articulated_ragdoll_pd_zerog_limits_and_rerooting(mut ctx: TestContext) raises:
    var inertia_pair = compute_capsule_inertia(10.0, 0.10, 0.50)
    ctx.assert_true(inertia_pair[1] > inertia_pair[0] and inertia_pair[0] > 0.0, "capsule transverse inertia > axial inertia > 0")

    var model = _build_battloid_rig()
    var rag = _configure_battloid_ragdoll(model)

    # 8A. Exact 6-DOF (v_i, omega_i) velocity handoff from consecutive SkeletonInstance poses
    var dt_h = Float32(1.0 / 60.0)
    var skel_prev = SkeletonInstance()
    skel_prev.reset_to_rest(model)
    var skel_curr = SkeletonInstance()
    skel_curr.reset_to_rest(model)

    var v_root = Vec3(4.0, 0.0, 6.0)
    var omega_y = Float32(3.0)  # 3 rad/s spin about +Y
    skel_curr.pose.r[0] = angle_axis(omega_y * dt_h, Vec3(0, 1, 0)).multiply(skel_prev.pose.r[0]).normalized()
    skel_curr.update_matrices(model)

    var pl_prev = Placement(Vec3(0, 0, 0))
    var pl_curr = Placement(v_root * dt_h)
    rag.handoff_from_poses(skel_prev, pl_prev, skel_curr, pl_curr, model, dt_h)

    # Pelvis local_com_offset = (0, 0.225, 0) lies along +Y spin axis, so CoM linear vel == v_root!
    ctx.assert_vec3_approx(rag.states[0].linear_vel, v_root, 0.02, "6-DOF handoff captures exact linear velocity v_i")
    ctx.assert_vec3_approx(rag.states[0].angular_vel, Vec3(0, omega_y, 0), 0.02, "6-DOF handoff captures exact angular velocity omega_i")

    # Also verify snapshot_prev_matrices() single-skeleton handoff overload
    var skel_single = SkeletonInstance()
    skel_single.reset_to_rest(model)
    skel_single.snapshot_prev_matrices()
    skel_single.pose.r[0] = angle_axis(omega_y * dt_h, Vec3(0, 1, 0))
    skel_single.update_matrices(model)
    rag.handoff_from_skeleton(skel_single, pl_curr, model, dt_h, pl_prev)
    ctx.assert_vec3_approx(rag.states[0].angular_vel, Vec3(0, omega_y, 0), 0.02, "snapshot_prev_matrices handoff captures omega_i")

    # 8B. 1G Active-PD Ballistic Stagger, OBB Collision, & Knee Anti-Hyperextension
    skel_curr.reset_to_rest(model)
    rag.handoff_from_poses(skel_curr, pl_prev, skel_curr, pl_prev, model, dt_h)
    rag.set_active_pd(1.0)
    rag.add_obb_collider(RagdollObbCollider(Vec3(0.0, 0.5, -0.85), Quat(), Vec3(1.0, 0.5, 0.25)))

    # Deliver heavy ballistic hit to Chest (-Z knockback) AND brutal backward hyperextension kick to Shin.L (+Z)!
    rag.apply_impulse_at_bone(1, Vec3(0.0, 0.0, -95.0))
    # Shin.L hinge_axis_local = +X, hinge_min = 0.0 (bending knee moves foot toward -Z => +X rotation;
    # kicking foot toward +Z attempts negative -X hyperextension!)
    rag.apply_impulse_at_bone(7, Vec3(0.0, 0.0, 80.0), Vec3(0.18, 0.15, 0.0))

    rag.step(dt_h, 4)
    var chest_stagger_vel = rag.states[1].linear_vel.z
    ctx.assert_true(chest_stagger_vel < -1.0, "chest staggers backward under ballistic impulse in 1G")

    var max_stagger_pitch = Float32(0.0)
    for _ in range(45):
        rag.step(dt_h, 4)
        var knee_angle = rag.joint_hinge_angle(5)  # limb 5 = Shin.L
        ctx.assert_true(knee_angle >= -0.001 and knee_angle <= 2.36, "knee hinge strictly respects [0, 2.35] without backward hyperextension")
        var chest_rel = det_quat_to_scaled_angle(
            rag.states[0].rotation.conjugate().multiply(rag.states[1].rotation)
        ).length()
        max_stagger_pitch = max(max_stagger_pitch, chest_rel)

    ctx.assert_true(max_stagger_pitch > 0.03, "Active-PD joints flex visibly during stagger")
    var recovered_chest_err = det_quat_to_scaled_angle(
        rag.states[0].rotation.conjugate().multiply(rag.states[1].rotation)
    ).length()
    ctx.assert_true(recovered_chest_err < max_stagger_pitch * 0.45, "Active-PD motors recover chest toward animated posture")
    ctx.assert_true(rag.pelvis_world_position().y > 0.55, "Active-PD + floor contact keeps character above ground")
    rag.clear_obb_colliders()

    # 8C. Zero-G Orbital Active-PD Stagger & Linear Momentum Conservation
    var rag_zg = _configure_battloid_ragdoll(model)
    skel_curr.reset_to_rest(model)
    rag_zg.handoff_from_poses(skel_curr, pl_prev, skel_curr, pl_prev, model, dt_h)
    rag_zg.set_zero_g(True)
    rag_zg.linear_damping = 0.0
    rag_zg.angular_damping = 0.2
    rag_zg.root_attitude_stiffness = 0.0  # pure internal action-reaction in free orbital float
    rag_zg.set_active_pd(1.0)

    var j_orbital = Vec3(-30.0, 12.0, -45.0)
    rag_zg.apply_impulse_at_bone(4, j_orbital)  # hit Forearm.R in Zero-G!
    var p_after_hit = rag_zg.total_linear_momentum()
    ctx.assert_vec3_approx(p_after_hit, j_orbital, 0.05, "Zero-G impulse imparts exact total linear momentum J")

    for _ in range(30):
        rag_zg.step(dt_h, 4)
        var elbow_ang = rag_zg.joint_hinge_angle(3)  # limb 3 = Forearm.R
        ctx.assert_true(elbow_ang >= -0.001 and elbow_ang <= 2.41, "elbow hinge never hyperextends in Zero-G")

    var p_drifted = rag_zg.total_linear_momentum()
    ctx.assert_vec3_approx(p_drifted, j_orbital, 0.25, "Zero-G internal PD joint torques conserve linear momentum")
    ctx.assert_true(rag_zg.total_center_of_mass().y > 1.0, "Zero-G Battloid floats upward with +Y impulse (zero gravity drop)")

    # 8D. Limp Collapse + World Pelvis Re-Rooting (`begin_recovery`) with 0.0 mm Snap!
    var rag_limp = _configure_battloid_ragdoll(model)
    skel_curr.reset_to_rest(model)
    var start_place = Placement(Vec3(2.0, 0.0, -1.0), Vec3(1, 1, 1), 120.0)  # rotated 120 deg!
    rag_limp.handoff_from_poses(skel_curr, start_place, skel_curr, start_place, model, dt_h)
    rag_limp.go_limp()
    rag_limp.apply_impulse_at_bone(0, Vec3(35.0, 0.0, -25.0))
    rag_limp.states[0].angular_vel = Vec3(0.4, 2.2, -0.3)

    for _ in range(90):
        rag_limp.step(dt_h, 4)

    var settled_pelvis_w = rag_limp.pelvis_world_position()
    var settled_yaw_w = rag_limp.pelvis_world_yaw()
    rag_limp.write_to_skeleton(skel_curr, model, start_place)

    var entity_pos = Vec3(0.0, 0.0, 0.0)
    var entity_yaw = Float32(0.0)
    var rec_inert = Inertializer()
    var target_stand = Pose()
    target_stand.rest(model)

    rag_limp.begin_recovery(entity_pos, entity_yaw, rec_inert, target_stand)
    ctx.assert_approx(entity_yaw, settled_yaw_w, 0.0005, "begin_recovery updates entity root_yaw to settled pelvis yaw")
    ctx.assert_approx(entity_pos.y, 0.0, 0.0005, "begin_recovery places 1G entity root_pos.y on floor plane")

    # Reconstruct world pelvis at t = 0+ immediately from seeded inertializer (out_t, out_r)
    var seeded_pose = Pose(target_stand.count())
    for idx in range(target_stand.count()):
        seeded_pose.t[idx] = rec_inert.out_t[idx]
        seeded_pose.r[idx] = rec_inert.out_r[idx]
        seeded_pose.s[idx] = rec_inert.out_s[idx]
    var pelvis_m_t0 = pose_position(seeded_pose, rag_limp.skel_parents, 0)
    var pelvis_w_t0 = entity_pos + angle_axis(entity_yaw, Vec3(0, 1, 0)).rotate(pelvis_m_t0)
    ctx.assert_vec3_approx(pelvis_w_t0, settled_pelvis_w, 0.0005, "begin_recovery achieves 0.0mm (<0.5mm) world pelvis continuity")

    # Verify every simulated limb's world rotation matches settled ragdoll at t = 0+
    for l_idx in range(rag_limp.limb_count()):
        var b_node = rag_limp.limbs[l_idx].bone_node
        var q_m = _pose_global_rot(seeded_pose, rag_limp.skel_parents, b_node)
        var q_w_t0 = angle_axis(entity_yaw, Vec3(0, 1, 0)).multiply(q_m).normalized()
        var rot_diff = det_quat_to_scaled_angle(q_w_t0.multiply(rag_limp.states[l_idx].rotation.conjugate())).length()
        ctx.assert_approx(rot_diff, 0.0, 0.002, "begin_recovery preserves exact world limb orientation at handoff")

    # Advance recovery inertializer and verify smooth convergence to standing target_pose
    for _ in range(60):
        var live_p = Pose()
        live_p.assign(target_stand)
        rec_inert.apply(live_p, target_stand, False, 0.10, dt_h)
    ctx.assert_true(rec_inert.largest_offset() < 0.01, "getup inertializer converges smoothly to standing pose")

    # 8E. Legacy `Ragdoll` 120-deg settled getup heading preservation test
    var bear = NativeModel()
    var b_root = _add_node(bear, "Root", -1, Vec3(0, 0, 0))
    var j_names = ragdoll_joints()
    var j_parents = ragdoll_parents()
    var node_map = List[Int](length=len(j_names), fill=-1)
    for j in range(len(j_names)):
        var p_node = b_root if j_parents[j] < 0 else node_map[j_parents[j]]
        var off = Vec3(0, 0.25, 0)
        if j == 0:
            off = Vec3(0, 0.9, 0)
        elif j == 12:
            off = Vec3(0.25, -0.1, 0)   # Thigh.L (+X)
        elif j == 15:
            off = Vec3(-0.25, -0.1, 0)  # Thigh.R (-X)
        node_map[j] = _add_node(bear, j_names[j], p_node, off)

    var rec_clip = NativeAnimation()
    rec_clip.name = "Recovery"
    rec_clip.duration = 1.0
    var ch_rec = NativeChannel()
    ch_rec.node = node_map[0]
    ch_rec.path = 0
    ch_rec.times = [Float32(0.0), Float32(1.0)]
    ch_rec.values = [Float32(0), Float32(0.35), Float32(0), Float32(0), Float32(0.90), Float32(0)]
    rec_clip.channels.append(ch_rec^)
    bear.animations.append(rec_clip^)

    var legacy_rag = Ragdoll()
    var bear_place = Placement()
    legacy_rag.update(bear, 1, bear_place, False, dt_h)
    legacy_rag.go_limp()
    # Rotate all ragdoll particles by +120 deg (+2*pi/3) about +Y and shift +1.5m X, -0.8m Z
    var yaw_120 = PI * Float32(2.0 / 3.0)
    var q_120 = angle_axis(yaw_120, Vec3(0, 1, 0))
    var shift_xz = Vec3(1.5, -0.55, -0.8)
    for j in range(len(j_names)):
        var p_idx = legacy_rag.particle[j]
        if p_idx >= 0:
            var rotated_p = q_120.rotate(legacy_rag.mesh.position[p_idx]) + shift_xz
            legacy_rag.mesh.place(p_idx, rotated_p)

    ctx.assert_approx(legacy_rag.settled_pelvis_yaw(), yaw_120, 0.01, "legacy Ragdoll.settled_pelvis_yaw measures +120 deg")
    ctx.assert_approx(legacy_rag.settled_root_offset(bear_place).x, 1.5, 0.01, "legacy Ragdoll.settled_root_offset X == 1.5m")

    legacy_rag.start_getup(bear)
    # Advance past GETUP_IN (0.26s) so ragdoll blend r == 0 and clip weight g == 1
    legacy_rag.play_getup(bear, bear_place, Float32(0.26))
    var root_fwd_after = bear.global_rotation(b_root).rotate(Vec3(0, 0, 1))
    var expected_fwd_120 = q_120.rotate(Vec3(0, 0, 1))
    ctx.assert_vec3_approx(root_fwd_after, expected_fwd_120, 0.02, "Ragdoll.play_getup aligns clip root to +120 deg settled heading (zero snap)")


def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("Flint_v4_Anim_IK_Ragdoll")
    try:
        test_1_cubicspline_step_and_springs(ctx)
        ctx.record_pass("test_1_cubicspline_step_and_springs")
    except e:
        ctx.record_fail("test_1_cubicspline_step_and_springs", String(e))

    try:
        test_2_skeleton_instance_topo_and_prev_matrices(ctx)
        ctx.record_pass("test_2_skeleton_instance_topo_and_prev_matrices")
    except e:
        ctx.record_fail("test_2_skeleton_instance_topo_and_prev_matrices", String(e))

    try:
        test_3_bollo_inertializer_and_socket_reparent(ctx)
        ctx.record_pass("test_3_bollo_inertializer_and_socket_reparent")
    except e:
        ctx.record_fail("test_3_bollo_inertializer_and_socket_reparent", String(e))

    try:
        test_4_blend_space_2d_and_loop_root_motion(ctx)
        ctx.record_pass("test_4_blend_space_2d_and_loop_root_motion")
    except e:
        ctx.record_fail("test_4_blend_space_2d_and_loop_root_motion", String(e))

    try:
        test_5_ik_pole_fourbar_digitigrade_and_footplant(ctx)
        ctx.record_pass("test_5_ik_pole_fourbar_digitigrade_and_footplant")
    except e:
        ctx.record_fail("test_5_ik_pole_fourbar_digitigrade_and_footplant", String(e))

    try:
        test_6_motion_matching_zscore_simd8_aabb(ctx)
        ctx.record_pass("test_6_motion_matching_zscore_simd8_aabb")
    except e:
        ctx.record_fail("test_6_motion_matching_zscore_simd8_aabb", String(e))

    try:
        test_7_rbf_helpers_skinning_and_springbones(ctx)
        ctx.record_pass("test_7_rbf_helpers_skinning_and_springbones")
    except e:
        ctx.record_fail("test_7_rbf_helpers_skinning_and_springbones", String(e))

    try:
        test_8_articulated_ragdoll_pd_zerog_limits_and_rerooting(ctx)
        ctx.record_pass("test_8_articulated_ragdoll_pd_zerog_limits_and_rerooting")
    except e:
        ctx.record_fail("test_8_articulated_ragdoll_pd_zerog_limits_and_rerooting", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Flint v4 Anim/IK/Ragdoll verification failed")
