"""Comprehensive Verification Suite for Flint v5 Track 3:
  (a) `PhysicsWorld` bit-exact rollback snapshot (`serialize` @ tick 120 -> `deserialize` -> step to 300)
      on a 200-body pile + joints + `VehicleChassis` + XPBD `ParticleChain`/`ParticleMesh`, kinematic
      surface velocity, O(1) `id_to_slot`, `TriggerVolume` (`ENTER/STAY/EXIT`), full-shape CCD, and
      O(log N) SIMD8 Scene Queries across all 7 collider types.
  (b) `CharacterController` (`KCC`) stair climb, `45°` walkable slope vs `65°` steep cliff slide (zero
      wall-surfing!), `45°` acute V-corner 2-plane crease slide & 3-plane lock without jitter, rotating
      kinematic turntable inheritance, and mass-clamped dynamic crate pushing.
  (c) 3D Quickhull + exact polyhedral mass/CoM/inertia + V-HACD-lite L-shape compound + `HeightfieldCollider`
      Amanatides-Woo 2D DDA raycast + `SixDofMotorJoint`.
  (d) Recast `NavMesh.build_from_physics` over multi-room + ramp geometry, `< 0.25 ms` dynamic `NavObstacle`
      door block/unblock, `OffMeshLink`, SSFA taut path, and 256-agent antipodal/bottleneck SIMD8 ORCA
      (`0` agent overlaps).
  (e) Flat DFS `BehaviorTree` reactive `BT_ABORT_LOWER_PRIORITY`, Dave Mark `UtilityAiScorer`, SIMD8
      `SpatialPerception` wall LOS block + hearing, `CameraRig` wall `SpringArm` clip + `trauma^2` shake,
      `.flanim` (`FLAN`) serializer/loader, `BoneSocketAttachment`, `solve_foot_plant_world`, and
      `RagdollStateMachine` limp-to-getup recovery with `0.0 mm` pelvis snap.
"""
from std.collections import List
from std.math import abs, isnan, isinf, max, min, sqrt
from std.testing import assert_true, assert_equal
from std.time import monotonic
from flint.core.math import Vec3, Quat
from flint.core.transform import Mat4, trs
from flint.core.detmath import det_sin, det_cos, PI
from flint.core.serial import BlobWriter, BlobReader, fnv64, float32_bits
from flint.core.jobs import JobPool
from flint.core.springs import SpringFloat, SpringVec3, SpringAngle, fast_negexp, halflife_to_damping
from flint.core.ik import (
    angle_axis,
    fabrik,
    solve_two_bone_pole,
    solve_four_bar_linkage,
    three_bone_digitigrade_ik,
    fabrik_constrained,
)
from flint.physics.filter import CollisionFilter, CAT_TERRAIN, CAT_STATIC, CAT_PROP_PIECE, CAT_ALL
from flint.physics.islands import NO_ISLAND
from flint.physics.joints import JointDef, JOINT_REVOLUTE
from flint.physics.water import (
    eval_beaufort_wave,
    eval_sphere_buoyancy_drag,
    eval_box_buoyancy_drag,
    RHO_SEAWATER_SI,
)
from flint.physics.events import (
    Sensor,
    sensor_filter,
    EVENT_BEGIN,
    EVENT_JOINT_BREAK,
    EVENT_SENSOR_BEGIN,
    EVENT_WATER_ENTER,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.narrowphase import ConvexHull, ContactManifold
from flint.physics.shapes_v5 import (
    CompoundChild,
    CompoundShape,
    HeightfieldCollider,
    SixDofMotorJoint,
)
from flint.physics.tires import PacejkaTire, PacejkaParams
from flint.physics.vehicle import VehicleChassis
from flint.physics.aero6dof import (
    Aero6DofConfig,
    Aero6DofControls,
    Aero6DofState,
    MissileBatch8,
    eval_isa_atmosphere,
    eval_ground_effect,
    eval_transonic_wave_drag,
    eval_post_stall_polar,
    step_aero6dof,
    step_missile_batch8,
)
from flint.physics.solver_bodies import SolverBodies, F8, V8, LANES
from flint.physics.solver import (
    SOLVER_WORD_STRIDE,
    SolverSettings,
    SolverRun,
    SolverStage,
    STAGE_SOLVE,
    quat_delta_displacement,
    rodrigues_displacement,
    solver_softness,
    tangent_effective_mass_2x2,
)
from flint.physics.xpbd import ParticleChain, ParticleMesh, ChainCollider, RigidAttachment
from flint.anim.ragdoll import (
    ArticulatedRagdoll,
    RagdollObbCollider,
    compute_capsule_inertia,
    _pose_global_rot,
    Ragdoll,
    ragdoll_joints,
    ragdoll_parents,
)
from flint.physics.world import (
    PhysicsWorld,
    TriggerVolume,
    BODY_MOTION_STATIC,
    BODY_MOTION_KINEMATIC,
    BODY_MOTION_DYNAMIC,
    TRIGGER_ENTER,
    TRIGGER_STAY,
    TRIGGER_EXIT,
)
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.app.engine_app import AppConfig, EngineApp
from flint.physics.plugin import CharacterControllerComponent, PhysicsPlugin
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from flint.nav import (
    NavMesh,
    NavMeshConfig,
    NavObstacle,
    OffMeshLink,
    CrowdSimulator,
    CrowdNavBridge,
    OFFMESH_JUMP,
)
from flint.ai import (
    Blackboard,
    UtilityConsideration,
    UtilityOption,
    UtilityAiScorer,
    BehaviorTree,
    BtExecutionState,
    SpatialPerception,
    BT_SUCCESS,
    BT_FAILURE,
    BT_RUNNING,
    BT_ABORT_LOWER_PRIORITY,
    BT_ABORT_SELF,
    BT_CMP_GT,
    CURVE_LINEAR,
    CURVE_POLYNOMIAL,
    CURVE_LOGISTIC,
)
from flint.camera import (
    CameraRig,
    CAM_MODE_FIRST_PERSON,
    CAM_MODE_ORBIT_FOLLOW,
    CAM_MODE_RAIL_DOLLY,
    CAM_MODE_FREE_FLY,
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
from flint.anim.pose import (
    Pose,
    SkeletonInstance,
    Inertializer,
    sample_pose,
    reparent_socket_inertial,
    det_quat_to_scaled_angle,
)
from flint.anim.sync import ClipInfo
from flint.anim.animator import Animator, extract_root_motion
from flint.anim.graph import (
    AnimGraphDef,
    MaskDef,
    BlendSpaceDef,
    BlendSampleDef,
    BlendSpace2DDef,
    BlendSample2DDef,
    StateDef,
    LayerDef,
    TransitionDef,
    FormOffsetDef,
    Condition,
    OP_GT,
    ROOT_EXTRACT,
)
from flint.anim.procedural import Placement, LookAt, Recoil, FootPlantSolver, pose_position
from flint.anim.mecha_ik import FourBarLinkageDriver, DigitigradeLegDriver, make_biped_foot_solver, solve_two_bone_mecha_limb
from flint.anim.motion_matching import MotionFeature, MotionDatabase, STANCE_GROUND, STANCE_AIR_SPACE
from flint.anim.flanim_ecs import (
    FLANIM_FOURCC,
    save_flanim_bytes,
    load_flanim_bytes,
    BoneSocketAttachment,
    update_socket_attachments,
    solve_foot_plant_world,
    RagdollStateMachine,
    RAGDOLL_SM_ANIMATED,
    RAGDOLL_SM_ACTIVE_PD,
    RAGDOLL_SM_LIMP,
    RAGDOLL_SM_GETUP_SUPINE,
    RAGDOLL_SM_GETUP_PRONE,
)


def _append_box_tris(mut vs: List[Vec3], mut is_: List[Int32], lo: Vec3, hi: Vec3):
    var b = Int32(len(vs))
    vs.append(Vec3(lo.x, lo.y, lo.z))
    vs.append(Vec3(hi.x, lo.y, lo.z))
    vs.append(Vec3(hi.x, hi.y, lo.z))
    vs.append(Vec3(lo.x, hi.y, lo.z))
    vs.append(Vec3(lo.x, lo.y, hi.z))
    vs.append(Vec3(hi.x, lo.y, hi.z))
    vs.append(Vec3(hi.x, hi.y, hi.z))
    vs.append(Vec3(lo.x, hi.y, hi.z))
    var faces = [
        Int32(0), Int32(2), Int32(1), Int32(0), Int32(3), Int32(2),
        Int32(4), Int32(5), Int32(6), Int32(4), Int32(6), Int32(7),
        Int32(0), Int32(1), Int32(5), Int32(0), Int32(5), Int32(4),
        Int32(3), Int32(6), Int32(2), Int32(3), Int32(7), Int32(6),
        Int32(0), Int32(4), Int32(7), Int32(0), Int32(7), Int32(3),
        Int32(1), Int32(2), Int32(6), Int32(1), Int32(6), Int32(5),
    ]
    for k in range(len(faces)):
        is_.append(b + faces[k])


# =========================================================================
# (a) PhysicsWorld 200-Body + Joints + Vehicle Rollback Snapshot, O(1) Lookup,
#     Kinematic Motion, Triggers, Full-Shape CCD & Unified Scene Queries
# =========================================================================

def _populate_200_body_pile_and_vehicle(mut w: PhysicsWorld, mut car: VehicleChassis):
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(35.0, 0.5, 35.0), mass=0.0)
    _ = w.add_trigger(TriggerVolume.obb(900, Vec3(0.0, 1.5, 0.0), Vec3(4.0, 2.0, 4.0)))

    var prev_slot = -1
    for i in range(200):
        var gx = Float32((i % 5) - 2) * Float32(0.82) + Float32((i // 25) % 2) * Float32(0.07)
        var gz = Float32(((i // 5) % 5) - 2) * Float32(0.82)
        var gy = Float32(0.55) + Float32(i // 25) * Float32(0.74)
        var kind = i % 4
        var slot: Int
        if kind == 0:
            slot = w.add_box(Vec3(gx, gy, gz), Vec3(0.30, 0.30, 0.30), mass=2.5)
        elif kind == 1:
            slot = w.add_sphere(Vec3(gx, gy, gz), 0.30, mass=2.0)
        elif kind == 2:
            slot = w.add_capsule(Vec3(gx, gy, gz), 0.20, 0.18, mass=2.2)
        else:
            var pts = List[Vec3]()
            pts.append(Vec3(-0.28, -0.25, -0.28))
            pts.append(Vec3(0.28, -0.25, -0.28))
            pts.append(Vec3(0.0, -0.25, 0.32))
            pts.append(Vec3(0.0, 0.32, 0.0))
            slot = w.add_convex_hull(Vec3(gx, gy, gz), ConvexHull.build_quickhull(pts), mass=2.4)

        if i % 20 == 1 and prev_slot >= 0:
            var mid = (w.bodies[prev_slot].pos + w.bodies[slot].pos) * Float32(0.5)
            _ = w.add_weld_joint(prev_slot, slot, mid, break_force=8000.0)
        elif i % 25 == 3 and prev_slot >= 0:
            var j6 = SixDofMotorJoint()
            j6.body_a = w.bodies[prev_slot].id
            j6.body_b = w.bodies[slot].id
            j6.local_anchor_a = Vec3(0.32, 0.0, 0.0)
            j6.local_anchor_b = Vec3(-0.32, 0.0, 0.0)
            j6.enable_linear_limit = True
            j6.linear_lower = Vec3(-0.08, -0.08, -0.08)
            j6.linear_upper = Vec3(0.08, 0.08, 0.08)
            j6.enable_angular_drive = True
            j6.angular_hertz = 6.0
            _ = w.add_six_dof_joint(j6^)
        prev_slot = slot

    car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(10.0, 0.54, -8.0)
    car.linear_vel = Vec3(0.0, 0.0, 14.0)


def test_a_snapshot_determinism_queries_triggers_ccd() raises:
    var dt = Float32(1.0 / 60.0)
    var world_ref = PhysicsWorld()
    var car_ref = VehicleChassis(1450.0)
    _populate_200_body_pile_and_vehicle(world_ref, car_ref)

    # Step 0 -> 120, take snapshot
    for _ in range(120):
        world_ref.step(dt)
        car_ref.substep_suspension(dt, throttle=0.7, brake=0.0, steer=0.25)

    var snap_writer = BlobWriter()
    world_ref.serialize(snap_writer)
    car_ref.serialize(snap_writer)

    # Step uninterrupted 120 -> 300
    for t in range(180):
        var brk = Float32(0.4) if t > 90 else Float32(0.0)
        var thr = Float32(0.0) if t > 90 else Float32(0.5)
        world_ref.step(dt)
        car_ref.substep_suspension(dt, throttle=thr, brake=brk, steer=-0.18)

    var out_ref = BlobWriter()
    world_ref.serialize(out_ref)
    car_ref.serialize(out_ref)
    var hash_uninterrupted = fnv64(out_ref.data)

    # Restore snapshot at tick 120 into brand-new instances and step 120 -> 300
    var world_restored = PhysicsWorld()
    var car_restored = VehicleChassis(900.0)
    var snap_reader = BlobReader(snap_writer.data.copy())
    world_restored.deserialize(snap_reader)
    car_restored.deserialize(snap_reader)

    for t in range(180):
        var brk = Float32(0.4) if t > 90 else Float32(0.0)
        var thr = Float32(0.0) if t > 90 else Float32(0.5)
        world_restored.step(dt)
        car_restored.substep_suspension(dt, throttle=thr, brake=brk, steer=-0.18)

    var out_restored = BlobWriter()
    world_restored.serialize(out_restored)
    car_restored.serialize(out_restored)
    var hash_restored = fnv64(out_restored.data)

    assert_equal(hash_uninterrupted, hash_restored)
    assert_equal(len(world_ref.bodies), len(world_restored.bodies))
    for i in range(len(world_ref.bodies)):
        assert_equal(float32_bits(world_ref.bodies[i].pos.x), float32_bits(world_restored.bodies[i].pos.x))
        assert_equal(float32_bits(world_ref.bodies[i].pos.y), float32_bits(world_restored.bodies[i].pos.y))
        assert_equal(float32_bits(world_ref.bodies[i].pos.z), float32_bits(world_restored.bodies[i].pos.z))
        assert_equal(float32_bits(world_ref.bodies[i].linear_vel.x), float32_bits(world_restored.bodies[i].linear_vel.x))
    assert_equal(float32_bits(car_ref.pos.x), float32_bits(car_restored.pos.x))
    assert_equal(float32_bits(car_ref.wheel_states[0].bristle_x), float32_bits(car_restored.wheel_states[0].bristle_x))

    # XPBD ParticleChain & ParticleMesh snapshot round-trip (including `lambdas`)
    var chain_a = ParticleChain.line(Vec3(0, 3, 0), Vec3(2, 3, 0), 6, mass=0.25)
    chain_a.compliance = Float32(1.0e-4)
    chain_a.pin(0, Vec3(0, 3, 0))
    var pm_a = ParticleMesh()
    var p0 = pm_a.add_particle(Vec3(0, 2, 0), 1.0, 0.1)
    var p1 = pm_a.add_particle(Vec3(0.5, 2, 0), 1.0, 0.1)
    pm_a.add_link(p0, p1, 1.0e-4)
    for _ in range(10):
        chain_a.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
        pm_a.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
    var xp_w = BlobWriter()
    chain_a.serialize(xp_w)
    pm_a.serialize(xp_w)
    var chain_b = ParticleChain()
    var pm_b = ParticleMesh()
    var xp_r = BlobReader(xp_w.data.copy())
    chain_b.deserialize(xp_r)
    pm_b.deserialize(xp_r)
    for _ in range(10):
        chain_a.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
        pm_a.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
        chain_b.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
        pm_b.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
    assert_equal(float32_bits(chain_a.position[5].y), float32_bits(chain_b.position[5].y))
    assert_equal(float32_bits(pm_a.position[1].y), float32_bits(pm_b.position[1].y))

    # O(1) id_to_slot maintenance across swap_remove_body + Kinematic conveyor belt surface velocity
    var wk = PhysicsWorld()
    var belt = wk.add_kinematic_box(Vec3(0.0, 0.0, 0.0), Vec3(10.0, 0.5, 4.0), linear_vel=Vec3(3.0, 0.0, 0.0), friction=0.95)
    var dummy = wk.add_sphere(Vec3(0.0, 10.0, 0.0), 0.25, mass=1.0)
    var crate = wk.add_box(Vec3(0.0, 0.9, 0.0), Vec3(0.4, 0.4, 0.4), mass=4.0, friction=0.95)
    var crate_id = wk.bodies[crate].id
    var dummy_id = wk.bodies[dummy].id
    wk.swap_remove_body(dummy)
    assert_equal(wk.find_slot_by_id(dummy_id), -1)
    var crate_slot = wk.find_slot_by_id(crate_id)
    assert_true(crate_slot >= 0)
    assert_equal(wk.bodies[belt].motion_mode, BODY_MOTION_KINEMATIC)
    assert_equal(wk.bodies[belt].inv_mass, Float32(0.0))
    assert_equal(wk.bodies[crate_slot].motion_mode, BODY_MOTION_DYNAMIC)
    for _ in range(30):
        wk.step(dt)
    crate_slot = wk.find_slot_by_id(crate_id)
    # Kinematic conveyor belt advances +X and accelerates resting crate via contact friction without sagging!
    assert_true(wk.bodies[belt].pos.x > Float32(1.4))
    assert_equal(float32_bits(wk.bodies[belt].pos.y), float32_bits(Float32(0.0)))
    assert_true(wk.bodies[crate_slot].linear_vel.x > Float32(1.5))

    # TriggerVolume ENTER -> STAY -> EXIT lifecycle + Full-shape CCD against thin Box wall
    var wt = PhysicsWorld(gravity=0.0)
    _ = wt.add_trigger(TriggerVolume.sphere(55, Vec3(0.0, 0.0, 0.0), 1.0))
    var mover = wt.add_sphere(Vec3(-2.0, 0.0, 0.0), 0.25, mass=1.0)
    wt.set_velocity(mover, Vec3(60.0, 0.0, 0.0))  # 1.0 m per tick at 60 Hz
    var saw_enter = False
    var saw_stay = False
    var saw_exit = False
    for _ in range(6):
        wt.step(dt)
        for tev in wt.trigger_events:
            if tev.trigger_id == 55:
                if tev.kind == TRIGGER_ENTER: saw_enter = True
                elif tev.kind == TRIGGER_STAY: saw_stay = True
                elif tev.kind == TRIGGER_EXIT: saw_exit = True
    assert_true(saw_enter and saw_stay and saw_exit)

    var wccd = PhysicsWorld(gravity=0.0)
    _ = wccd.add_box(Vec3(2.0, 0.0, 0.0), Vec3(0.05, 4.0, 4.0), mass=0.0)  # thin 10 cm wall at x = 2.0
    var fast_box = wccd.add_box(Vec3(0.0, 0.0, 0.0), Vec3(0.2, 0.2, 0.2), mass=1.0)
    wccd.bodies[fast_box].enable_ccd = True
    wccd.set_velocity(fast_box, Vec3(300.0, 0.0, 0.0))  # 5 m in a single tick!
    wccd.step(dt)
    assert_true(wccd.bodies[fast_box].pos.x < Float32(2.0))

    # Unified Scene Queries across all 7 collider types + filter + ignore_body_id
    var wq = PhysicsWorld()
    var s_id = wq.bodies[wq.add_sphere(Vec3(0.0, 1.0, 2.0), 0.5, mass=0.0, filter=CollisionFilter(CAT_PROP_PIECE, CAT_ALL, 0))].id
    _ = wq.add_capsule(Vec3(0.0, 1.0, 5.0), 0.4, 0.5, mass=0.0)
    _ = wq.add_box(Vec3(0.0, 1.0, 8.0), Vec3(0.6, 0.6, 0.6), mass=0.0)
    var qh_pts: List[Vec3] = [Vec3(-0.5, -0.5, -0.5), Vec3(0.5, -0.5, -0.5), Vec3(0.0, 0.7, -0.5), Vec3(0.0, 0.0, 0.6)]
    _ = wq.add_convex_hull(Vec3(0.0, 1.0, 11.0), ConvexHull.build_quickhull(qh_pts), mass=0.0)
    var bvh_v: List[Vec3] = [Vec3(-1.0, 0.0, 14.0), Vec3(1.0, 0.0, 14.0), Vec3(0.0, 2.0, 14.0)]
    var bvh_i: List[Int32] = [Int32(0), Int32(1), Int32(2)]
    _ = wq.add_static_mesh(Bvh8MeshCollider.build(bvh_v, bvh_i))
    var cmp = CompoundShape()
    cmp.add_child(CompoundChild.box(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(0.5, 0.5, 0.5)))
    _ = wq.add_compound(Vec3(0.0, 1.0, 17.0), cmp, mass=0.0)
    var hf_flat: List[Float32] = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    _ = wq.add_heightfield(HeightfieldCollider.build(hf_flat, 2, 2, Vec3(-2.0, 0.0, 19.0), 4.0))
    wq.rebuild_broadphase()

    var h_first = wq.raycast(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), 25.0)
    assert_true(h_first.hit and h_first.body_id == s_id)

    # Ignore first sphere -> hits capsule at z = 5!
    var h_ign = wq.raycast(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), 25.0, ignore_body_id=s_id)
    assert_true(h_ign.hit and abs(h_ign.distance - Float32(4.6)) < Float32(0.05))

    # Filter mask excluding CAT_DEBRIS skips sphere as well
    var h_flt = wq.raycast(
        Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), 25.0,
        filter=CollisionFilter(CAT_ALL, CAT_STATIC | CAT_TERRAIN, 0),
    )
    assert_true(h_flt.hit and h_flt.body_id != s_id)

    var all_hits = wq.raycast_all(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_true(len(all_hits) >= 6)
    for k in range(len(all_hits) - 1):
        assert_true(all_hits[k].distance <= all_hits[k + 1].distance)

    var sc_sph = wq.shapecast_sphere(Vec3(0.0, 1.0, 0.0), 0.25, Vec3(0.0, 0.0, 10.0), ignore_body_id=s_id)
    assert_true(sc_sph.hit and abs(sc_sph.distance - Float32(4.35)) < Float32(0.05))
    var sc_cap = wq.shapecast_capsule(Vec3(0.0, 1.0, 6.2), 0.2, 0.3, Vec3(0.0, 0.0, 5.0))
    assert_true(sc_cap.hit and sc_cap.distance < Float32(1.5))
    var sc_box = wq.shapecast_box(Vec3(0.0, 1.0, 6.2), Vec3(0.2, 0.2, 0.2), Quat(), Vec3(0.0, 0.0, 5.0))
    assert_true(sc_box.hit and sc_box.distance < Float32(1.5))
    assert_true(len(wq.overlap_sphere(Vec3(0.0, 1.0, 8.0), 0.8)) >= 1)
    assert_true(len(wq.overlap_aabb(Vec3(-1.0, 0.0, 1.0), Vec3(1.0, 2.0, 18.0))) >= 6)


# =========================================================================
# (b) Kinematic Character Controller (`KCC`): Stairs, 45° vs 65° Slopes,
#     45° Acute V-Corner Crease & 3-Plane Lock, Turntable & Crate Push
# =========================================================================

def test_b_kcc_stairs_slopes_acute_corner_turntable_and_push() raises:
    var dt = Float32(1.0 / 60.0)
    var cfg = CharacterControllerConfig()
    cfg.radius = 0.35
    cfg.half_height = 0.55
    cfg.max_slope_deg = 50.0
    cfg.step_height = 0.35

    # 1. Stair auto-climb (5 steps of 0.25 m each -> total 1.25 m rise)
    var ws = PhysicsWorld()
    _ = ws.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    for i in range(5):
        var top_y = Float32(i + 1) * Float32(0.25)
        _ = ws.add_box(Vec3(0.0, top_y * Float32(0.5), Float32(2.0) + Float32(i) * Float32(0.65)), Vec3(1.5, top_y * Float32(0.5), 0.325), mass=0.0)
    _ = ws.add_box(Vec3(0.0, 0.625, 7.5), Vec3(1.5, 0.625, 2.6), mass=0.0)  # upstairs landing at y = 1.25
    ws.rebuild_broadphase()
    var kcc_stair = CharacterController(Vec3(0.0, 0.90, 0.5), cfg)
    var climbed_any = False
    for _ in range(95):
        _ = kcc_stair.move_and_slide(ws, Vec3(0.0, 0.0, 4.8), dt)
        if kcc_stair.stepped_up:
            climbed_any = True
    assert_true(climbed_any)
    assert_true(kcc_stair.grounded)
    assert_true(kcc_stair.foot_position().y > Float32(1.20))

    # 2. 45° walkable ramp (climbs smoothly, grounded = True) vs 65° steep cliff (zero wall-surfing!)
    var wr45 = PhysicsWorld()
    _ = wr45.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var ang45 = Float32(45.0 * 0.01745329252)
    _ = wr45.add_box(Vec3(0.0, 1.5, 5.0), Vec3(3.0, 0.25, 4.0), mass=0.0, rot=angle_axis(-ang45, Vec3(1, 0, 0)))
    wr45.rebuild_broadphase()
    var kcc45 = CharacterController(Vec3(0.0, 0.90, 1.2), cfg)
    for _ in range(75):
        _ = kcc45.move_and_slide(wr45, Vec3(0.0, 0.0, 4.0), dt)
    assert_true(kcc45.grounded)
    assert_true(kcc45.foot_position().y > Float32(1.5))

    var wr65 = PhysicsWorld()
    _ = wr65.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var ang65 = Float32(65.0 * 0.01745329252)
    _ = wr65.add_box(Vec3(0.0, 2.5, 4.0), Vec3(3.0, 0.25, 5.0), mass=0.0, rot=angle_axis(-ang65, Vec3(1, 0, 0)))
    wr65.rebuild_broadphase()
    var kcc65 = CharacterController(Vec3(0.0, 0.90, 0.8), cfg)
    for _ in range(75):
        _ = kcc65.move_and_slide(wr65, Vec3(0.0, 0.0, 5.0), dt)
    # Must NEVER surf upward up the 65° cliff!
    assert_true(kcc65.foot_position().y < Float32(0.15))

    # Also drop onto 65° slope from above -> slides downhill (`on_steep_slope == True`, `grounded == False`)
    var kcc_drop65 = CharacterController(Vec3(0.0, 3.8, 3.8), cfg)
    var saw_steep = False
    for _ in range(45):
        _ = kcc_drop65.move_and_slide(wr65, Vec3(0.0, 0.0, 0.0), dt)
        if kcc_drop65.on_steep_slope:
            saw_steep = True
    assert_true(saw_steep)
    assert_true(kcc_drop65.pos.y < Float32(3.2) and kcc_drop65.pos.z < Float32(3.7))

    # 3. 45° acute V-corner without jitter: 2-plane `n0 x n1` crease projection & 3-plane corner lock!
    var wv = PhysicsWorld()
    _ = wv.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    # Two vertical walls meeting at an acute 45° angle at apex z = 6.0 (+22.5° and -22.5° about Y)
    var half_wedge = Float32(22.5 * 0.01745329252)
    var s_w = det_sin(half_wedge)
    var c_w = det_cos(half_wedge)
    # Left wall passes through (-s_w*L, z_apex - c_w*L) with inward normal (+c_w, 0, -s_w)
    _ = wv.add_box(Vec3(-s_w * Float32(4.0) - c_w * Float32(0.25), 1.5, Float32(6.0) - c_w * Float32(4.0) + s_w * Float32(0.25)), Vec3(0.25, 1.5, 4.5), mass=0.0, rot=angle_axis(half_wedge, Vec3(0, 1, 0)))
    _ = wv.add_box(Vec3(s_w * Float32(4.0) + c_w * Float32(0.25), 1.5, Float32(6.0) - c_w * Float32(4.0) + s_w * Float32(0.25)), Vec3(0.25, 1.5, 4.5), mass=0.0, rot=angle_axis(-half_wedge, Vec3(0, 1, 0)))
    wv.rebuild_broadphase()

    var kcc_v = CharacterController(Vec3(-0.25, 0.90, 2.5), cfg)
    var prev_z = kcc_v.pos.z
    for _ in range(90):
        _ = kcc_v.move_and_slide(wv, Vec3(0.4, 0.0, 4.0), dt)
        # Monotonic forward progress into the V-wedge until locked smoothly at the acute crease: zero backward bounce/jitter!
        assert_true(kcc_v.pos.z >= prev_z - Float32(1.0e-4))
        prev_z = kcc_v.pos.z
    # Once wedged into the acute 45° vertical crease (ground + 2 walls = 3 planes!), velocity locks cleanly to 0 without oscillation
    var lock_p0 = kcc_v.pos
    _ = kcc_v.move_and_slide(wv, Vec3(0.0, 0.0, 4.0), dt)
    assert_true((kcc_v.pos - lock_p0).length() < Float32(2.0e-3))

    # 4. Rotating kinematic turntable inheritance (orbital translation + heading yaw)
    var wt = PhysicsWorld()
    var tt_omega = Float32(1.5)  # rad/s about +Y
    var tt_slot = wt.add_kinematic_box(
        Vec3(0.0, -0.25, 0.0), Vec3(5.0, 0.25, 5.0),
        linear_vel=Vec3(1.0, 0.0, 0.0),
        angular_vel=Vec3(0.0, tt_omega, 0.0),
    )
    var kcc_tt = CharacterController(Vec3(0.0, 0.90, 2.0), cfg)
    _ = kcc_tt.move_and_slide(wt, Vec3(), dt)  # latch onto turntable
    assert_true(kcc_tt.grounded and kcc_tt.ground_body_id == wt.bodies[tt_slot].id)
    for _ in range(30):  # 0.5 s -> turntable rotates +0.75 rad and translates +0.5 m in X
        wt.step(dt)
        _ = kcc_tt.move_and_slide(wt, Vec3(), dt)
    assert_true(abs(kcc_tt.yaw - Float32(0.75)) < Float32(0.05))
    assert_true(kcc_tt.pos.x > Float32(1.6))  # 0.5 m linear + 2.0 * sin(0.75) ~= 1.86 m!

    # 5. Dynamic RigidBody crate pushing + KCC serialize/deserialize
    var wp = PhysicsWorld()
    _ = wp.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0, friction=0.3)
    var crate_slot = wp.add_box(Vec3(0.0, 0.5, 2.0), Vec3(0.5, 0.5, 0.5), mass=25.0, friction=0.3)
    var kcc_push = CharacterController(Vec3(0.0, 0.90, 0.8), cfg)
    for _ in range(45):
        _ = kcc_push.move_and_slide(wp, Vec3(0.0, 0.0, 4.0), dt)
        wp.step(dt)
    assert_true(wp.bodies[crate_slot].pos.z > Float32(2.4))

    var kw = BlobWriter()
    kcc_push.serialize(kw)
    var kcc_rest = CharacterController()
    var kr = BlobReader(kw.data.copy())
    kcc_rest.deserialize(kr)
    assert_equal(float32_bits(kcc_push.pos.z), float32_bits(kcc_rest.pos.z))


# =========================================================================
# (c) Quickhull 3D + Polyhedral Mass + V-HACD-lite + Heightfield DDA + 6DOF Joint
# =========================================================================

def test_c_quickhull_mass_vhacd_heightfield_dda_and_6dof() raises:
    # 1. Quickhull 3D on a 2x4x6 box with interior clutter -> exact 8 corners & exact volume/CoM/inertia!
    var pts = List[Vec3]()
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                pts.append(Vec3(Float32(sx) * 1.0 + 2.0, Float32(sy) * 2.0 - 1.0, Float32(sz) * 3.0 + 0.5))
    # Add 40 interior points that Quickhull must discard
    for i in range(40):
        var f = Float32(i) * Float32(0.02)
        pts.append(Vec3(Float32(2.0) + det_cos(f) * Float32(0.5), Float32(-1.0) + det_sin(f) * Float32(0.8), Float32(0.5)))

    var hull = ConvexHull.build_quickhull(pts, max_verts=64)
    assert_equal(len(hull.vertices), 8)
    assert_true(len(hull.indices) >= 36)

    var rho = Float32(2.0)
    var mp = hull.compute_polyhedral_mass_properties(rho)
    # Volume of 2 x 4 x 6 box = 48.0 -> mass at density 2.0 = 96.0
    assert_true(abs(mp[0] - Float32(96.0)) < Float32(0.05))
    assert_true((mp[1] - Vec3(2.0, -1.0, 0.5)).length() < Float32(1.0e-3))
    # Exact box inertia about CoM: Ix = m/12 * (4^2 + 6^2) = 8 * 52 = 416; Iy = 8 * (4 + 36) = 320; Iz = 8 * (4 + 16) = 160
    assert_true(abs(mp[2].xx - Float32(416.0)) < Float32(0.5))
    assert_true(abs(mp[2].yy - Float32(320.0)) < Float32(0.5))
    assert_true(abs(mp[2].zz - Float32(160.0)) < Float32(0.5))

    # Vertex capping (`max_verts = 12` on a 64-point sphere cloud)
    var sph_pts = List[Vec3]()
    for i in range(64):
        var a = Float32(i) * Float32(0.41)
        var b = Float32(i) * Float32(0.83)
        sph_pts.append(Vec3(det_cos(a) * det_cos(b), det_sin(b), det_sin(a) * det_cos(b)))
    var capped_hull = ConvexHull.build_quickhull(sph_pts, max_verts=12)
    assert_true(len(capped_hull.vertices) <= 12 and len(capped_hull.vertices) >= 4)

    # 2. V-HACD-lite concave L-shape decomposition preserves empty concave notch!
    var lv = List[Vec3]()
    var li = List[Int32]()
    _append_box_tris(lv, li, Vec3(-1.0, -0.3, -0.3), Vec3(1.0, 0.3, 0.3))
    _append_box_tris(lv, li, Vec3(0.4, 0.3, -0.3), Vec3(1.0, 1.7, 0.3))
    var l_comp = CompoundShape.decompose_vhacd_lite(lv, li, max_hulls=4, voxel_res=16, concavity_tol=0.01)
    assert_true(l_comp.child_count() >= 2)

    var wc = PhysicsWorld()
    _ = wc.add_compound(Vec3(0.0, 0.0, 0.0), l_comp, mass=10.0)
    wc.rebuild_broadphase()
    # Ray along Z through the concave notch `(-0.4, 1.1, -3 -> +3)` MUST miss the L-shape,
    # whereas a single un-decomposed convex hull would falsely block it!
    var notch_ray = wc.raycast(Vec3(-0.4, 1.1, -3.0), Vec3(0.0, 0.0, 1.0), 6.0)
    assert_true(not notch_ray.hit)
    # Rays through the horizontal base `(0.0, 0.0, -3 -> +3)` and vertical arm `(0.7, 1.1, -3 -> +3)` MUST hit!
    assert_true(wc.raycast(Vec3(0.0, 0.0, -3.0), Vec3(0.0, 0.0, 1.0), 6.0).hit)
    assert_true(wc.raycast(Vec3(0.7, 1.1, -3.0), Vec3(0.0, 0.0, 1.0), 6.0).hit)

    # 3. HeightfieldCollider 2D Amanatides-Woo DDA raycast accuracy vs analytical plane height
    var nx = 33
    var nz = 33
    var h_vals = List[Float32](capacity=nx * nz)
    for iz in range(nz):
        for ix in range(nx):
            # Planar tilted ramp y = 0.25 * x + 0.15 * z
            var wx = Float32(ix) * Float32(0.5)
            var wz = Float32(iz) * Float32(0.5)
            h_vals.append(Float32(0.25) * wx + Float32(0.15) * wz)
    var hf = HeightfieldCollider.build(h_vals, nx, nz, Vec3(0.0, 0.0, 0.0), 0.5)
    var ro = Vec3(1.0, 6.0, 1.0)
    var rd = Vec3(0.6, -0.5, 0.5).normalized()
    var dda = hf.raycast_dda(ro, rd, 30.0)
    assert_true(dda.hit)
    var hp = ro + rd * dda.t
    var exact_y = Float32(0.25) * hp.x + Float32(0.15) * hp.z
    assert_true(abs(hp.y - exact_y) < Float32(2.0e-3))

    # 4. SixDofMotorJoint sub-stepped soft PD linear & angular drive: immediate 1-step response (0 frame lag!)
    var w6 = PhysicsWorld(gravity=0.0)
    var b_driven = w6.add_box(Vec3(0.0, 0.0, 0.0), Vec3(0.3, 0.3, 0.3), mass=2.0)
    var j6 = SixDofMotorJoint()
    j6.body_b = w6.bodies[b_driven].id
    j6.enable_linear_limit = False
    j6.enable_linear_drive = True
    j6.linear_target = Vec3(1.5, 0.0, 0.0)
    j6.linear_hertz = 6.0
    j6.linear_damping = 1.0
    j6.max_linear_force = 500.0
    j6.enable_angular_drive = True
    j6.angular_target = angle_axis(0.5, Vec3(0.0, 1.0, 0.0))
    j6.angular_hertz = 6.0
    j6.angular_damping = 1.0
    _ = w6.add_six_dof_joint(j6^)

    # Step 1 alone must integrate SixDofMotorJoint velocity into position & orientation within substeps (zero 1-frame lag)
    w6.step(Float32(1.0 / 60.0))
    assert_true(w6.bodies[b_driven].pos.x > Float32(1.0e-3))
    assert_true(abs(w6.bodies[b_driven].rot.y) > Float32(1.0e-4))

    for _ in range(59):
        w6.step(Float32(1.0 / 60.0))
    assert_true(abs(w6.bodies[b_driven].pos.x - Float32(1.5)) < Float32(0.08))


# =========================================================================
# (d) Recast NavMesh + <0.25 ms Dynamic Obstacle Carve + OffMeshLink + SSFA
#     + 256-Agent Antipodal & Bottleneck SIMD8 ORCA (`0` Overlaps!)
# =========================================================================

def test_d_recast_navmesh_obstacle_ssfa_and_256_orca() raises:
    # Multi-room + sloped ramp geometry in PhysicsWorld:
    # Lower Room A (x in [-7, -1], y = 0) -> doorway at x = -1 -> Hallway -> sloped ramp up to Upper Room B (y = 1.5)
    var wn = PhysicsWorld()
    _ = wn.add_box(Vec3(-4.0, -0.25, 0.0), Vec3(3.5, 0.25, 4.0), mass=0.0)  # Room A floor (y=0)
    # Wall at x = -1 with doorway gap at z in [-1.1, 1.1]
    _ = wn.add_box(Vec3(-1.0, 1.2, -2.6), Vec3(0.25, 1.2, 1.5), mass=0.0)
    _ = wn.add_box(Vec3(-1.0, 1.2, 2.6), Vec3(0.25, 1.2, 1.5), mass=0.0)
    # Ramp rising from (x = -0.8, y = 0) to (x = 3.2, y = 1.5) -> slope ~20.5 deg (< 46 deg walkable!)
    var ramp_pitch = Float32(0.35877)  # atan2(1.5, 4.0)
    _ = wn.add_box(Vec3(1.2, 0.52, 0.0), Vec3(2.3, 0.25, 2.2), mass=0.0, rot=angle_axis(ramp_pitch, Vec3(0, 0, 1)))
    # Upper Room B platform (x in [3.0, 7.5], y = 1.5)
    _ = wn.add_box(Vec3(5.2, 1.25, 0.0), Vec3(2.3, 0.25, 3.5), mass=0.0)
    # Isolated balcony across a chasm (x in [9.5, 12.5], y = 1.5) reachable only via OffMeshLink!
    _ = wn.add_box(Vec3(11.0, 1.25, 0.0), Vec3(1.5, 0.25, 2.0), mass=0.0)

    var cfg = NavMeshConfig()
    cfg.cell_size = 0.35
    cfg.cell_height = 0.12
    cfg.walkable_radius = 0.35
    cfg.walkable_climb = 0.45
    var nav = NavMesh.build_from_physics(wn, cfg)
    assert_true(len(nav.polys) >= 3)

    # 1. Path from Lower Room A (-5, 0, -2.5) through doorway & up ramp to Upper Room B (5.5, 1.5, 2.0)
    var p_start = Vec3(-5.0, 0.0, -2.5)
    var p_goal = Vec3(5.5, 1.5, 2.0)
    var path_open = nav.find_path(p_start, p_goal)
    assert_true(path_open.found)
    assert_true(len(path_open.waypoints) >= 2)
    # SSFA tautness check: path passes tightly past doorway corner without zig-zagging across polygon centers
    assert_true(path_open.total_length < Float32(14.5))

    # 2. Dynamic NavObstacle door block & unblock (< 0.25 ms tiled incremental carve!)
    var door_obs = NavObstacle.box(42, Vec3(-1.0, 1.0, 0.0), Vec3(0.45, 1.2, 1.3))
    var t_carve0 = monotonic()
    _ = nav.add_obstacle(door_obs)
    var carve_ms = Float64(monotonic() - t_carve0) / 1.0e6
    assert_true(carve_ms < 0.25)
    var path_blocked = nav.find_path(p_start, p_goal)
    assert_true(not path_blocked.found)

    nav.remove_obstacle(42)
    var path_unblocked = nav.find_path(p_start, p_goal)
    assert_true(path_unblocked.found)

    # 3. OffMeshLink jump across chasm to isolated balcony (11.0, 1.5, 0.0)
    var p_balcony = Vec3(11.0, 1.5, 0.0)
    assert_true(not nav.find_path(p_start, p_balcony).found)
    _ = nav.add_off_mesh_link(OffMeshLink(7, Vec3(6.8, 1.5, 0.0), Vec3(9.8, 1.5, 0.0), radius=0.8, kind=OFFMESH_JUMP))
    var path_jump = nav.find_path(p_start, p_balcony)
    assert_true(path_jump.found and len(path_jump.used_off_mesh_links) == 1 and path_jump.used_off_mesh_links[0] == 7)

    # 4. Large 60m x 60m 36-pillar tiled NavMesh: < 0.25 ms dynamic carve, agent_radius clearance & ~80m taut diagonal path
    var w60 = PhysicsWorld()
    _ = w60.add_box(Vec3(0.0, -0.25, 0.0), Vec3(30.0, 0.25, 30.0), mass=0.0)
    for pi in range(6):
        for pj in range(6):
            _ = w60.add_box(Vec3(Float32(pi * 10 - 25), 1.0, Float32(pj * 10 - 25)), Vec3(1.5, 1.0, 1.5), mass=0.0)
    var cfg60 = NavMeshConfig()
    cfg60.cell_size = 0.30
    cfg60.cell_height = 0.10
    cfg60.walkable_radius = 0.35
    var nav60 = NavMesh.build_from_physics(w60, cfg60)
    var t_c60 = monotonic()
    _ = nav60.add_obstacle(NavObstacle.box(1, Vec3(0.0, 0.5, 0.0), Vec3(1.0, 1.0, 1.0)))
    var carve60_ms = Float64(monotonic() - t_c60) / 1.0e6
    assert_true(carve60_ms < 0.25)
    var diag60 = nav60.find_path(Vec3(-28.0, 0.0, -28.0), Vec3(28.0, 0.0, 28.0))
    assert_true(diag60.found and diag60.total_length < Float32(82.0))

    # 5. 1,000-Tick 256-Agent Antipodal Convergence & Bottleneck Doorway SIMD8 ORCA (`0` overlaps across ALL 1,000 ticks!)
    var dt = Float32(1.0 / 60.0)
    var crowd_open = CrowdSimulator(time_horizon=1.8, neighbor_dist=4.5, max_neighbors=16)
    for i in range(256):
        var ang = Float32(i) * (Float32(2.0) * PI / Float32(256.0))
        var rad = Float32(20.0) + Float32(i % 4) * Float32(1.1)
        var sp = Vec3(det_cos(ang) * rad, 0.0, det_sin(ang) * rad)
        _ = crowd_open.add_agent(sp, radius=0.30, max_speed=3.6, goal=Vec3(-sp.x, 0.0, -sp.z))

    var max_ov_open = 0
    var min_center_r = Float32(1.0e9)
    for _ in range(1400):
        crowd_open.step_orca_simd8(dt)
        var ov = crowd_open.count_overlaps()
        if ov > max_ov_open:
            max_ov_open = ov
        for i in range(256):
            var p = crowd_open.agent_position(i)
            var r_c = sqrt(p.x * p.x + p.z * p.z)
            if r_c < min_center_r:
                min_center_r = r_c
    assert_equal(max_ov_open, 0)
    assert_true(min_center_r < Float32(0.5))

    var arrived_open = 0
    for i in range(256):
        var ang = Float32(i) * (Float32(2.0) * PI / Float32(256.0))
        var rad = Float32(20.0) + Float32(i % 4) * Float32(1.1)
        var gp = Vec3(-det_cos(ang) * rad, 0.0, -det_sin(ang) * rad)
        if (crowd_open.agent_position(i) - gp).length() < Float32(1.5):
            arrived_open += 1
    assert_true(arrived_open >= 245)

    # Also verify 1,000-tick doorway bottleneck crowd maintains 0 overlaps and funnels legally through central gap
    var crowd = CrowdSimulator(time_horizon=1.8, neighbor_dist=4.5, max_neighbors=16)
    crowd.add_wall_segment(Vec3(0.0, 0.0, -30.0), Vec3(0.0, 0.0, -3.5))
    crowd.add_wall_segment(Vec3(0.0, 0.0, 3.5), Vec3(0.0, 0.0, 30.0))

    for i in range(256):
        var ang = Float32(i) * (Float32(2.0) * PI / Float32(256.0))
        var rad = Float32(20.0) + Float32(i % 4) * Float32(1.1)
        var sp = Vec3(det_cos(ang) * rad, 0.0, det_sin(ang) * rad)
        _ = crowd.add_agent(sp, radius=0.30, max_speed=3.6, goal=Vec3(-sp.x, 0.0, -sp.z))

    assert_equal(crowd.count_overlaps(), 0)
    var max_ov_wall = 0
    for _ in range(1000):
        crowd.step_orca_simd8(dt)
        var ov = crowd.count_overlaps()
        if ov > max_ov_wall:
            max_ov_wall = ov
    assert_equal(max_ov_wall, 0)


# =========================================================================
# (e) BehaviorTree Reactive Abort, UtilityAI, SpatialPerception, CameraRig,
#     `.flanim`, BoneSocketAttachment, World Foot-IK & RagdollStateMachine
# =========================================================================

def _make_test_biped() -> NativeModel:
    var m = NativeModel()
    var names = [
        String("Pelvis"), String("Chest"), String("Head"),
        String("UpperArm.R"), String("Hand.R"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
    ]
    var parents = [-1, 0, 1, 1, 3, 0, 5, 6, 0, 8, 9]
    var offsets = [
        Vec3(0.0, 0.92, 0.0), Vec3(0.0, 0.42, 0.0), Vec3(0.0, 0.32, 0.0),
        Vec3(0.26, 0.24, 0.0), Vec3(0.0, -0.48, 0.0),
        Vec3(-0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
        Vec3(0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
    ]
    for i in range(len(names)):
        var nd = NativeNode()
        nd.name = names[i]
        nd.translation = offsets[i]
        nd.base_translation = offsets[i]
        nd.rotation = Quat()
        nd.base_rotation = Quat()
        nd.scale = Vec3(1, 1, 1)
        nd.base_scale = Vec3(1, 1, 1)
        m.nodes.append(nd^)
        var p = parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)

    for cname in [String("GetupSupine"), String("GetupProne")]:
        var anim = NativeAnimation()
        anim.name = cname
        anim.duration = 0.40
        var ch = NativeChannel()
        ch.node = 0
        ch.path = 0
        ch.times = [Float32(0.0), Float32(0.40)]
        ch.values = [Float32(0.0), Float32(0.92), Float32(0.0), Float32(0.0), Float32(0.92), Float32(0.0)]
        anim.channels.append(ch^)
        m.animations.append(anim^)
    return m^


def test_e_ai_perception_camera_flanim_and_ragdoll_sm() raises:
    var dt = Float32(1.0 / 60.0)

    # 1. BehaviorTree O(1) subtree_skip, decorators, UtilitySelect & reactive BT_ABORT_LOWER_PRIORITY + BT_ABORT_SELF
    var bb = Blackboard()
    var k_hp = bb.def_key("hp", 0.9)
    var k_enemy_near = bb.def_key("enemy_near", 0.0)
    var k_ammo = bb.def_key("ammo", 0.85)
    var k_cover = bb.def_key("cover", 0.80)

    # Verify Dave Mark UtilityAiScorer compensation prevents multi-consideration product penalty!
    var scorer = UtilityAiScorer(use_geometric_mean=True)
    var opt_single = UtilityOption(100, 1.0)
    opt_single.add_consideration(UtilityConsideration(k_cover, 0.0, 1.0, CURVE_LINEAR))  # 0.80
    _ = scorer.add_option(opt_single^)
    var opt_multi = UtilityOption(200, 1.0)
    # Three strong considerations (~0.85 each): uncompensated product = 0.614 < 0.80, but geometric mean = 0.85 > 0.80!
    opt_multi.add_consideration(UtilityConsideration(k_hp, 0.0, 1.0, CURVE_LINEAR))
    opt_multi.add_consideration(UtilityConsideration(k_ammo, 0.0, 1.0, CURVE_LINEAR))
    opt_multi.add_consideration(UtilityConsideration(k_ammo, 0.0, 1.0, CURVE_POLYNOMIAL, exponent=1.0))
    _ = scorer.add_option(opt_multi^)
    var best_u = scorer.select_best(bb)
    assert_equal(best_u[0], 200)
    assert_true(UtilityAiScorer.compensate_dave_mark(Float32(0.512), 3) > Float32(0.65))

    var bt = BehaviorTree()
    _ = bt.begin_selector()
    # High-priority reactive branch: if enemy_near > 0.5, immediately abort lower-priority patrol!
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_enemy_near, BT_CMP_GT, 0.5, BT_ABORT_LOWER_PRIORITY)
    _ = bt.add_action(10)
    bt.end_composite()
    # Mid-priority self-aborting branch + Cooldown + Inverter + Parallel coverage
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_hp, BT_CMP_GT, 2.0, BT_ABORT_SELF)
    _ = bt.begin_cooldown(1.0)
    _ = bt.begin_inverter()
    _ = bt.add_action(11)
    bt.end_composite()
    bt.end_composite()
    bt.end_composite()
    # Low-priority UtilitySelect + Patrol running action
    _ = bt.begin_parallel(1)
    _ = bt.add_action(20)
    bt.end_composite()
    bt.end_composite()

    var bts = BtExecutionState()
    bts.set_action_status(10, BT_RUNNING)
    bts.set_action_status(11, BT_FAILURE)
    bts.set_action_status(20, BT_RUNNING)

    # Tick 1: enemy_near == 0.0 -> falls through to low-priority action 20 (BT_RUNNING)
    var st1 = bt.tick(bb, bts, dt)
    assert_equal(st1, BT_RUNNING)
    assert_equal(bts.active_action_id, 20)
    assert_equal(bts.aborts_triggered, 0)

    # Tick 2: enemy_near spikes to 1.0 -> reactive BT_ABORT_LOWER_PRIORITY preempts action 20 and activates action 10!
    bb.set_float(k_enemy_near, 1.0)
    var st2 = bt.tick(bb, bts, dt)
    assert_equal(st2, BT_RUNNING)
    assert_equal(bts.aborts_triggered, 1)
    assert_equal(bts.last_aborted_action_id, 20)
    assert_equal(bts.active_action_id, 10)

    # 2. SIMD8 SpatialPerception: vision cone + PhysicsWorld wall LOS block + acoustic hearing
    var wp = PhysicsWorld()
    _ = wp.add_box(Vec3(0.0, 1.5, 5.0), Vec3(3.0, 1.5, 0.3), mass=0.0)  # solid wall at z = 5
    wp.rebuild_broadphase()

    var perc = SpatialPerception()
    # 0: Directly ahead behind wall (z = 8), silent -> occluded (`visible == False`, `audible == False`, omitted!)
    _ = perc.add_stimulus(Vec3(0.0, 1.5, 8.0), loudness=0.0, body_id=101, team_id=2)
    # 1: Directly ahead behind wall (z = 8), loud gunshot -> LOS blocked (`visible == False`), but `audible == True`!
    _ = perc.add_stimulus(Vec3(0.0, 1.5, 8.0), loudness=40.0, body_id=102, team_id=2)
    # 2: Clear LOS to the right within 110° FOV `(4.5, 1.5, 6.0)` -> `visible == True`!
    _ = perc.add_stimulus(Vec3(4.5, 1.5, 6.0), loudness=0.0, body_id=103, team_id=2)
    # Add 8 more far/behind stimuli to exercise multi-batch SIMD8 lanes
    for extra in range(8):
        _ = perc.add_stimulus(Vec3(0.0, 1.5, -15.0 - Float32(extra)), loudness=0.0, body_id=200 + extra, team_id=2)

    var sensed = perc.query_perceiver(wp, Vec3(0.0, 1.5, 0.0), Vec3(0.0, 0.0, 1.0), fov_deg=110.0, max_vision_dist=30.0)
    assert_equal(len(sensed), 2)
    assert_true(sensed[0].body_id == 102 and (not sensed[0].visible) and sensed[0].audible)
    assert_true(sensed[1].body_id == 103 and sensed[1].visible)

    # 3. Cinemachine CameraRig: Orbit SpringArm wall anti-clipping, Rail Dolly, FP, FreeFly & trauma^2 shake
    var cam = CameraRig(CAM_MODE_ORBIT_FOLLOW)
    cam.spring_arm.target_length = 6.0
    cam.spring_arm.probe_radius = 0.25
    cam.yaw_deg = 180.0  # look toward -Z so camera boom extends toward +Z into the wall at z = 5.0 (front face z = 4.7)!
    cam.pitch_deg = 0.0
    cam.look_offset = Vec3(0.0, 1.5, 0.0)
    cam.update(wp, Vec3(0.0, 0.0, 2.0), dt)
    assert_true(cam.spring_arm.is_occluded)
    # Pivot is at z = 2.0, wall front face is at z = 4.7, probe_radius = 0.25 -> camera z strictly < 4.5!
    assert_true(cam.world_pos.z < Float32(4.55) and cam.spring_arm.current_length.value < Float32(2.6))

    # Move target into open space (z = -10) -> boom smoothly springs back out toward 6.0 m
    for _ in range(45):
        cam.update(wp, Vec3(0.0, 0.0, -10.0), dt)
    assert_true(not cam.spring_arm.is_occluded and cam.spring_arm.current_length.value > Float32(5.2))

    # Deterministic trauma^2 shake scales quadratically
    var cam_lo = CameraRig(CAM_MODE_FIRST_PERSON)
    var cam_hi = CameraRig(CAM_MODE_FIRST_PERSON)
    cam_lo.add_trauma(0.4)
    cam_hi.add_trauma(0.8)
    cam_lo.update(wp, Vec3(), dt)
    cam_hi.update(wp, Vec3(), dt)
    var off_lo = (cam_lo.world_pos - cam_lo.unshaken_pos).length()
    var off_hi = (cam_hi.world_pos - cam_hi.unshaken_pos).length()
    assert_true(off_lo > Float32(1.0e-4) and abs(off_hi / off_lo - Float32(4.0)) < Float32(0.05))

    # Rail Dolly & Free-Fly modes
    var cam_rail = CameraRig(CAM_MODE_RAIL_DOLLY)
    var rail_pts: List[Vec3] = [Vec3(-5.0, 3.0, -5.0), Vec3(0.0, 3.0, -5.0), Vec3(5.0, 3.0, -5.0)]
    cam_rail.set_rail_waypoints(rail_pts)
    for _ in range(30):
        cam_rail.update(wp, Vec3(4.5, 0.0, 0.0), dt)
    assert_true(cam_rail.world_pos.x > Float32(2.5))

    var cam_fly = CameraRig(CAM_MODE_FREE_FLY)
    var fly0 = cam_fly.fly_pos
    cam_fly.update(wp, Vec3(), dt, fly_move_local=Vec3(0.0, 0.0, 1.0), fly_speed=12.0)
    assert_true((cam_fly.world_pos - fly0).length() > Float32(0.15))

    # 4. Binary `.flanim` (`FLAN` FourCC) complete round-trip
    var g = AnimGraphDef()
    g.name = "CombatGraphV5"
    g.default_blend = 0.12
    g.params.append("speed")
    g.params.append("aim")
    var msk = MaskDef()
    msk.name = "UpperBody"
    msk.bones.append("Chest")
    msk.weights.append(1.0)
    g.masks.append(msk^)
    var bs = BlendSpaceDef()
    bs.name = "RunBlend"
    bs.param = 0
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 6.0))
    g.blend_spaces.append(bs^)
    var bs2 = BlendSpace2DDef()
    bs2.name = "OmniBlend"
    bs2.param_x = 0
    bs2.param_y = 1
    bs2.is_polar = False
    bs2.samples.append(BlendSample2DDef(0, 0.0, 0.0))
    bs2.samples.append(BlendSample2DDef(1, 4.0, 1.0))
    g.blend_spaces_2d.append(bs2^)
    var st = StateDef()
    st.name = "Sprint"
    st.blend = 0
    st.root_motion = ROOT_EXTRACT
    st.when.append(Condition(0, OP_GT, 3.5))
    g.states.append(st^)
    var ly = LayerDef()
    ly.name = "AimLayer"
    ly.clip = 1
    ly.mask = 0
    ly.additive = True
    ly.over.append(0)
    g.layers.append(ly^)
    g.transitions.append(TransitionDef(0, -1, 0.15))
    g.setting_names.append("ik_weight")
    g.setting_values.append(1.0)
    var fm = FormOffsetDef()
    fm.name = "HeavyArmor"
    fm.scale = 1.15
    fm.bones.append("Chest")
    fm.bone_scales.append(1.1)
    g.forms.append(fm^)

    var fl_bytes = save_flanim_bytes(g)
    var fl_tag = UInt32(fl_bytes[0]) | (UInt32(fl_bytes[1]) << 8) | (UInt32(fl_bytes[2]) << 16) | (UInt32(fl_bytes[3]) << 24)
    assert_equal(fl_tag, FLANIM_FOURCC)
    var g2 = load_flanim_bytes(fl_bytes)
    assert_equal(g2.name, "CombatGraphV5")
    assert_equal(len(g2.masks), 1)
    assert_equal(len(g2.blend_spaces), 1)
    assert_equal(len(g2.blend_spaces_2d), 1)
    assert_equal(len(g2.states), 1)
    assert_equal(g2.states[0].root_motion, ROOT_EXTRACT)
    assert_equal(len(g2.layers), 1)
    assert_equal(len(g2.transitions), 1)
    assert_equal(len(g2.forms), 1)

    # 5. BoneSocketAttachment + solve_foot_plant_world + RagdollStateMachine (0.0 mm pelvis snap!)
    var wa = PhysicsWorld()
    _ = wa.add_box(Vec3(0.0, -0.5, 0.0), Vec3(30.0, 0.5, 30.0), mass=0.0)
    _ = wa.add_box(Vec3(0.14, 0.125, 0.0), Vec3(0.25, 0.125, 0.40), mass=0.0)  # 0.25 m step under right foot!
    _ = wa.add_box(Vec3(1.2, 0.25, -0.8), Vec3(0.4, 0.25, 0.4), mass=0.0)       # OBB crate near ragdoll fall zone
    var shield_slot = wa.add_kinematic_box(Vec3(), Vec3(0.2, 0.3, 0.05))
    var shield_id = wa.bodies[shield_slot].id
    wa.rebuild_broadphase()

    var biped = _make_test_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1, 1, 1), 0.0)

    var sock = BoneSocketAttachment("Shield", 4, 50, Vec3(0.05, 0.0, 0.1), Quat(), Vec3(1, 1, 1), shield_id)
    sock.sync_to_physics(skel, place, wa, dt)
    skel.pose.r[3] = angle_axis(0.8, Vec3(0, 1, 0))
    skel.update_matrices(biped)
    sock.sync_to_physics(skel, place, wa, dt)
    assert_true(wa.bodies[shield_slot].linear_vel.length() > Float32(1.0))
    var socks = List[BoneSocketAttachment]()
    socks.append(sock^)
    update_socket_attachments(socks, skel, place)

    var fp = FootPlantSolver(0)
    fp.add_leg(5, 6, 7, 0.04)
    fp.add_leg(8, 9, 10, 0.04)
    var stance: List[Bool] = [True, True]
    var ray_hits = 0
    for _ in range(20):
        skel.reset_to_rest(biped)
        ray_hits = solve_foot_plant_world(fp, wa, skel, biped, place, stance, dt)
    assert_equal(ray_hits, 2)
    var foot_l_y = fp.foot_world_position(0, skel, place).y
    var foot_r_y = fp.foot_world_position(1, skel, place).y
    # Right foot stands on 0.25 m step (`y ~= 0.29`), left foot on ground (`y ~= 0.04`)!
    assert_true(foot_r_y > foot_l_y + Float32(0.18))

    # RagdollStateMachine: Animated -> ActivePD -> Limp -> Supine/Prone GetUp -> Animated (`0.0 mm` pelvis snap!)
    var rsm = RagdollStateMachine()
    rsm.supine_getup_clip = 0
    rsm.prone_getup_clip = 1
    _ = rsm.ragdoll.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
    _ = rsm.ragdoll.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)

    var rest_pose = Pose()
    rest_pose.rest(biped)
    skel.reset_to_rest(biped)

    # Light hit -> enters ACTIVE_PD stagger
    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(15.0, 0.0, 0.0), Vec3(0.0, 1.3, 0.0), go_limp_threshold=100.0, dt=dt)
    assert_equal(rsm.state, RAGDOLL_SM_ACTIVE_PD)
    rsm.update(wa, skel, place, biped, rest_pose, dt)
    assert_true(len(rsm.ragdoll.obb_colliders) >= 1)  # PhysicsWorld OBB collider query populated nearby step/crate!

    # Heavy backward hit -> knocks into LIMP, settles on back (supine), re-roots with 0.0 mm snap, and recovers to ANIMATED!
    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(0.0, 15.0, -160.0), Vec3(0.0, 1.35, 0.0), go_limp_threshold=100.0, dt=dt)
    var saw_getup = False
    var getup_mode = -1
    for _ in range(180):
        rsm.update(wa, skel, place, biped, rest_pose, dt)
        if rsm.state == RAGDOLL_SM_GETUP_SUPINE or rsm.state == RAGDOLL_SM_GETUP_PRONE:
            saw_getup = True
            getup_mode = rsm.state
        if saw_getup and rsm.state == RAGDOLL_SM_ANIMATED:
            break

    assert_true(saw_getup)
    assert_true(getup_mode == RAGDOLL_SM_GETUP_SUPINE or getup_mode == RAGDOLL_SM_GETUP_PRONE)
    assert_equal(rsm.state, RAGDOLL_SM_ANIMATED)
    assert_equal(rsm.recovery_count, 1)
    # Pelvis snap across LIMP -> GETUP re-rooting MUST be 0.0 mm (< 0.1 mm / 1e-4 m)!
    assert_true(rsm.last_pelvis_snap_m < Float32(1.0e-4))


def test_f_phase0_phase1_regressions() raises:
    var dt = Float32(1.0 / 60.0)

    # 1. CompoundShape multi-normal per-child manifolds (no normal corruption!)
    var wc = PhysicsWorld(gravity=-9.81)
    var l_cmp = CompoundShape()
    l_cmp.add_child(CompoundChild.box(Vec3(0.0, -0.25, 0.0), Quat(), Vec3(2.0, 0.25, 2.0)))  # floor top at y = 0.0
    l_cmp.add_child(CompoundChild.box(Vec3(1.5, 1.0, 0.0), Quat(), Vec3(0.25, 1.25, 2.0)))   # wall inner face at x = 1.25
    var cmp_slot = wc.add_compound(Vec3(0.0, 0.0, 0.0), l_cmp, mass=0.0)
    # Box overlapping both floor (y < 0.0) and wall (x > 1.25) simultaneously
    var probe_slot = wc.add_box(Vec3(1.0, 0.20, 0.0), Vec3(0.35, 0.30, 0.35), mass=1.0)
    wc.rebuild_broadphase()

    var ms = List[ContactManifold]()
    wc._collide_pair_into(cmp_slot, probe_slot, ms)
    assert_true(len(ms) >= 2)
    var saw_y_normal = False
    var saw_x_normal = False
    for mi in range(len(ms)):
        if abs(ms[mi].normal.y) > Float32(0.8):
            saw_y_normal = True
        if abs(ms[mi].normal.x) > Float32(0.8):
            saw_x_normal = True
    assert_true(saw_y_normal and saw_x_normal)
    for _ in range(15):
        wc.step(dt)
    assert_true(wc.bodies[probe_slot].pos.y > Float32(0.25))
    assert_true(wc.bodies[probe_slot].pos.x < Float32(1.0))

    # 2. KCC vs CompoundShape: stands cleanly inside concave L-pocket AND depenetrates from wall child
    var kcc_cfg = CharacterControllerConfig()
    kcc_cfg.radius = 0.35
    kcc_cfg.half_height = 0.55
    var kcc = CharacterController(Vec3(0.0, 1.05, 0.0), kcc_cfg)
    for _ in range(20):
        _ = kcc.move_and_slide(wc, Vec3(0.0, 0.0, 0.0), dt)
    # Must rest on the floor inside the concave pocket (x ~= 0, foot_y ~= 0.0)
    assert_true(abs(kcc.pos.x) < Float32(0.15))
    assert_true(abs(kcc.foot_position().y) < Float32(0.12))
    # Now place KCC overlapping the vertical wall child (wall inner face x = 1.25) and verify depenetration pushes it out
    kcc.pos = Vec3(1.05, 1.05, 0.0)
    _ = kcc.move_and_slide(wc, Vec3(0.0, 0.0, 0.0), dt)
    assert_true(kcc.pos.x <= Float32(0.92))

    # 3. CCD bilateral momentum conservation against a dynamic body
    var wccd = PhysicsWorld(gravity=0.0)
    var proj = wccd.add_sphere(Vec3(0.0, 0.0, 0.0), 0.25, mass=1.0, restitution=1.0)
    var targ = wccd.add_sphere(Vec3(1.5, 0.0, 0.0), 0.25, mass=1.0, restitution=1.0)
    wccd.bodies[proj].enable_ccd = True
    wccd.set_velocity(proj, Vec3(120.0, 0.0, 0.0))  # 2.0 m/tick -> hits target at toi = 0.5
    wccd.rebuild_broadphase()
    wccd.step(dt)
    # Target dynamic sphere must wake and receive forward momentum; total momentum conserved (~120 kg*m/s)
    assert_true(wccd.bodies[targ].linear_vel.x > Float32(90.0))
    var p_total = wccd.bodies[proj].linear_vel.x + wccd.bodies[targ].linear_vel.x
    assert_true(abs(p_total - Float32(120.0)) < Float32(5.0))

    # 4. BoneSocketAttachment.sync_to_physics does not double-advance kinematic body during world.step(dt)
    var ws = PhysicsWorld(gravity=0.0)
    var sh_slot = ws.add_kinematic_box(Vec3(), Vec3(0.2, 0.3, 0.05))
    var sh_id = ws.bodies[sh_slot].id
    var biped = _make_test_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1, 1, 1), 0.0)
    var sock = BoneSocketAttachment("Shield", 4, 50, Vec3(0.05, 0.0, 0.1), Quat(), Vec3(1, 1, 1), sh_id)
    sock.sync_to_physics(skel, place, ws, dt)
    ws.step(dt)
    skel.pose.r[3] = angle_axis(0.5, Vec3(0, 1, 0))
    skel.update_matrices(biped)
    sock.sync_to_physics(skel, place, ws, dt)
    var target_pos = sock.world_pos
    ws.step(dt)
    assert_true((ws.bodies[sh_slot].pos - target_pos).length() < Float32(0.01))

    # 5. Tile-indexed NavMesh find_nearest_poly & _find_poly_containing_probe
    var wn = PhysicsWorld(gravity=0.0)
    _ = wn.add_box(Vec3(0.0, -0.25, 0.0), Vec3(12.0, 0.25, 12.0), mass=0.0)
    wn.rebuild_broadphase()
    var ncfg = NavMeshConfig()
    ncfg.cell_size = 0.5
    ncfg.tile_size = 8  # 4 m tiles -> 6x6 tile grid
    var nm = NavMesh.build_from_physics(wn, ncfg)
    assert_true(nm.tiles_x >= 4 and nm.tiles_z >= 4)
    var q_pts: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(5.5, 0.0, -4.5), Vec3(-6.0, 0.0, 6.0)]
    for qi in range(len(q_pts)):
        var qp = q_pts[qi]
        var res = nm.find_nearest_poly(qp, 4.0)
        assert_true(res[0] >= 0)
        assert_true((res[1] - qp).length() < Float32(0.6))
        var probe_pid = nm._find_poly_containing_probe(res[1], res[1].y, Float32(1.0))
        assert_true(probe_pid >= 0)

    # 6. CameraRig trauma^2 shake clamped against walls (never clips through wall!)
    var wcam = PhysicsWorld(gravity=0.0)
    # Thick static wall behind target (face at z = -1.5)
    _ = wcam.add_box(Vec3(0.0, 2.0, -2.0), Vec3(8.0, 4.0, 0.5), mass=0.0, filter=CollisionFilter(CAT_STATIC, CAT_ALL, 0))
    wcam.rebuild_broadphase()
    var rig = CameraRig(CAM_MODE_ORBIT_FOLLOW)
    rig.yaw_deg = 0.0
    rig.pitch_deg = 0.0
    rig.spring_arm.target_length = 4.0
    rig.spring_arm.probe_radius = 0.20
    rig.spring_arm.wall_margin = 0.05
    rig.max_shake_offset = Vec3(0.4, 0.4, 0.4)
    rig.add_trauma(1.0)
    for _ in range(30):
        rig.update(wcam, Vec3(0.0, 0.0, 0.0), dt)
        # Wall surface is at z = -1.5; camera center must NEVER penetrate closer than -1.5 + probe_radius!
        assert_true(rig.world_pos.z >= Float32(-1.5) + rig.spring_arm.probe_radius - Float32(0.01))

    # 7. BT_NODE_PARALLEL preserves independent per-node running state without clobbering
    var bt = BehaviorTree()
    _ = bt.begin_parallel(2)
    _ = bt.begin_sequence()
    _ = bt.add_action(0)
    _ = bt.add_action(1)
    bt.end_composite()
    _ = bt.begin_sequence()
    _ = bt.add_action(2)
    _ = bt.add_action(3)
    bt.end_composite()
    bt.end_composite()

    var bb = Blackboard()
    var st = BtExecutionState(8)
    st.set_action_status(0, BT_SUCCESS)
    st.set_action_status(1, BT_RUNNING)
    st.set_action_status(2, BT_SUCCESS)
    st.set_action_status(3, BT_RUNNING)

    assert_equal(bt.tick(bb, st, dt), BT_RUNNING)
    assert_equal(st.action_exec_counts[0], 1)
    assert_equal(st.action_exec_counts[1], 1)
    assert_equal(st.action_exec_counts[2], 1)
    assert_equal(st.action_exec_counts[3], 1)

    # Tick 2: both child Sequences MUST resume at their running actions (1 and 3) without re-running 0 or 2!
    assert_equal(bt.tick(bb, st, dt), BT_RUNNING)
    assert_equal(st.action_exec_counts[0], 1)
    assert_equal(st.action_exec_counts[1], 2)
    assert_equal(st.action_exec_counts[2], 1)
    assert_equal(st.action_exec_counts[3], 2)

    # Complete both running actions -> Parallel succeeds and clears running state
    st.set_action_status(1, BT_SUCCESS)
    st.set_action_status(3, BT_SUCCESS)
    assert_equal(bt.tick(bb, st, dt), BT_SUCCESS)
    assert_equal(st.running_node, -1)


def test_g_wave2_unification_and_parallel_regressions() raises:
    var dt = Float32(1.0 / 60.0)
    var pool = JobPool()
    _ = pool.start(4)
    var pool_addr = Int(Pointer(to=pool))

    # 1. Unified SixDofMotorJoint solver determinism (serial vs JobPool)
    var ws6_a = PhysicsWorld(gravity=-9.81)
    var ws6_b = PhysicsWorld(gravity=-9.81)
    for k in range(2):
        ref w = ws6_a if k == 0 else ws6_b
        _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(10.0, 0.5, 10.0), mass=0.0)
        _ = w.add_box(Vec3(0.0, 3.0, 0.0), Vec3(0.3, 0.3, 0.3), mass=0.0)
        var arm_s = w.add_box(Vec3(1.0, 3.0, 0.0), Vec3(0.5, 0.15, 0.15), mass=2.0)
        var j6 = SixDofMotorJoint()
        j6.body_b = w.bodies[arm_s].id
        j6.local_anchor_a = Vec3(0.5, 3.0, 0.0)
        j6.local_anchor_b = Vec3(-0.5, 0.0, 0.0)
        j6.enable_linear_limit = True
        j6.linear_lower = Vec3(-0.02, -0.02, -0.02)
        j6.linear_upper = Vec3(0.02, 0.02, 0.02)
        j6.linear_hertz = 12.0
        j6.linear_damping = 1.0
        j6.enable_angular_drive = True
        j6.angular_target = angle_axis(0.6, Vec3(0.0, 0.0, 1.0))
        j6.angular_hertz = 6.0
        j6.angular_damping = 1.0
        _ = w.add_six_dof_joint(j6^)
        w.rebuild_broadphase()
    for _ in range(40):
        ws6_a.step(dt, 0)
        ws6_b.step(dt, pool_addr)
    var bw_s6a = BlobWriter(); ws6_a.serialize(bw_s6a)
    var bw_s6b = BlobWriter(); ws6_b.serialize(bw_s6b)
    assert_equal(fnv64(bw_s6a.data), fnv64(bw_s6b.data))
    assert_true(ws6_a.bodies[2].pos.y > Float32(1.8))

    # 2. Parallel narrowphase determinism (>= 64 broadphase pairs, serial vs 4-worker JobPool)
    var wp_ser = PhysicsWorld(gravity=-9.81)
    var wp_par = PhysicsWorld(gravity=-9.81)
    for k in range(2):
        ref w = wp_ser if k == 0 else wp_par
        _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(12.0, 0.5, 12.0), mass=0.0)
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    var px = (Float32(ix) - Float32(1.5)) * Float32(0.85)
                    var py = Float32(0.55) + Float32(iy) * Float32(0.85)
                    var pz = (Float32(iz) - Float32(1.5)) * Float32(0.85)
                    if ((ix + iy + iz) & 1) == 0:
                        _ = w.add_box(Vec3(px, py, pz), Vec3(0.38, 0.38, 0.38), mass=1.0)
                    else:
                        _ = w.add_sphere(Vec3(px, py, pz), 0.38, mass=1.0)
        w.rebuild_broadphase()
    for _ in range(30):
        wp_ser.step(dt, 0)
        wp_par.step(dt, pool_addr)
    var bw_ser = BlobWriter(); wp_ser.serialize(bw_ser)
    var bw_par = BlobWriter(); wp_par.serialize(bw_par)
    assert_equal(fnv64(bw_ser.data), fnv64(bw_par.data))

    pool.stop()

    # 3. Water buoyancy AABB early-out for dry bodies vs submerged buoyancy
    var dry_w = eval_box_buoyancy_drag(
        Vec3(0.0, 5.0, 0.0), Quat(), Vec3(0.5, 0.5, 0.5),
        Vec3(1.0, 0.0, 0.0), Vec3(), Float32(1.23), water_y=0.0, use_waves=True,
    )
    assert_equal(dry_w.submerged_fraction, Float32(0.0))
    assert_equal(dry_w.force.length_squared(), Float32(0.0))
    var wet_w = eval_box_buoyancy_drag(
        Vec3(0.0, -0.5, 0.0), Quat(), Vec3(0.5, 0.5, 0.5),
        Vec3(), Vec3(), Float32(1.23), water_y=0.0, use_waves=True,
    )
    assert_true(wet_w.submerged_fraction > Float32(0.5) and wet_w.force.y > Float32(1000.0))

    # 4. NavMesh binary serialize / deserialize bit-exact round-trip
    var wn = PhysicsWorld(gravity=0.0)
    _ = wn.add_box(Vec3(0.0, -0.25, 0.0), Vec3(10.0, 0.25, 10.0), mass=0.0)
    _ = wn.add_box(Vec3(0.0, 1.0, 0.0), Vec3(1.0, 1.0, 3.0), mass=0.0)  # central divider wall
    wn.rebuild_broadphase()
    var ncfg = NavMeshConfig()
    ncfg.cell_size = 0.5
    ncfg.tile_size = 8
    var nav1 = NavMesh.build_from_physics(wn, ncfg)
    _ = nav1.add_off_mesh_link(OffMeshLink(77, Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0), 0.6, OFFMESH_JUMP))
    var nav_bw1 = BlobWriter()
    nav1.serialize(nav_bw1)
    var nav_br = BlobReader(nav_bw1.data.copy())
    var nav2 = NavMesh()
    nav2.deserialize(nav_br)
    var nav_bw2 = BlobWriter()
    nav2.serialize(nav_bw2)
    assert_equal(fnv64(nav_bw1.data), fnv64(nav_bw2.data))
    assert_equal(len(nav2.polys), len(nav1.polys))
    assert_equal(len(nav2.tiles), len(nav1.tiles))
    var p1 = nav1.find_path(Vec3(-6.0, 0.0, -5.0), Vec3(6.0, 0.0, 5.0))
    var p2 = nav2.find_path(Vec3(-6.0, 0.0, -5.0), Vec3(6.0, 0.0, 5.0))
    assert_true(p1.found and p2.found)
    assert_equal(len(p1.waypoints), len(p2.waypoints))
    assert_true(abs(p1.total_length - p2.total_length) < Float32(1.0e-4))

    # 5a. Multi-level ORCA Y-band vertical separation (stacked floors do not collide)
    var crowd_y = CrowdSimulator()
    crowd_y.agent_height = 2.0
    var a_low = crowd_y.add_agent(Vec3(-3.0, 0.0, 0.0), 0.4, 3.0, Vec3(3.0, 0.0, 0.0))
    var a_high = crowd_y.add_agent(Vec3(3.0, 3.0, 0.0), 0.4, 3.0, Vec3(-3.0, 3.0, 0.0))
    for _ in range(120):
        crowd_y.step_orca_simd8(dt)
    assert_equal(crowd_y.count_overlaps(), 0)
    # Agents on different floors must cross straight past each other with near-zero Z deflection!
    assert_true(crowd_y.agent_position(a_low).x > Float32(2.0))
    assert_true(crowd_y.agent_position(a_high).x < Float32(-2.0))
    assert_true(abs(crowd_y.agent_position(a_low).z) < Float32(0.25))
    assert_true(abs(crowd_y.agent_position(a_high).z) < Float32(0.25))

    # 5b. NavMesh -> ORCA CrowdNavBridge corridor following around central wall
    var crowd_nav = CrowdSimulator()
    var bridge = CrowdNavBridge()
    var b0 = bridge.add_agent(crowd_nav, nav2, Vec3(-6.0, 0.0, 0.0), Vec3(6.0, 0.0, 0.0), 0.35, 4.0)
    var b1 = bridge.add_agent(crowd_nav, nav2, Vec3(-6.0, 0.0, 1.5), Vec3(6.0, 0.0, -1.5), 0.35, 4.0)
    for _ in range(360):
        bridge.update_and_step(crowd_nav, nav2, dt)
    assert_equal(crowd_nav.count_overlaps(), 0)
    assert_true(bridge.is_arrived(b0, crowd_nav) and bridge.is_arrived(b1, crowd_nav))


def test_h_m2_two_way_physics_world_unification() raises:
    var dt = Float32(1.0 / 60.0)

    # 1. F10: VehicleChassis <-> PhysicsWorld two-way crate collision + BVH8 ramp + Heightfield + PHYW v2 rollback
    var w_free = PhysicsWorld(gravity=9.81)
    _ = w_free.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    var car_free = VehicleChassis(1450.0)
    car_free.setup_four_wheel_car()
    car_free.pos = Vec3(0.0, 0.55, 0.0)
    car_free.set_controls(throttle=1.0, brake=0.0, steer=0.0)
    var v_free = w_free.add_vehicle(car_free^)
    w_free.rebuild_broadphase()
    for _ in range(120):
        w_free.step(dt)
    var free_z = w_free.vehicle_state(v_free).pos.z
    assert_true(free_z > Float32(2.0))

    var w_hit = PhysicsWorld(gravity=9.81)
    _ = w_hit.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    var crate_slot = w_hit.add_box(Vec3(0.0, 0.5, 4.0), Vec3(0.6, 0.5, 0.6), mass=500.0)
    var car_hit = VehicleChassis(1450.0)
    car_hit.setup_four_wheel_car()
    car_hit.pos = Vec3(0.0, 0.55, 0.0)
    car_hit.set_controls(throttle=1.0, brake=0.0, steer=0.0)
    var v_hit = w_hit.add_vehicle(car_hit^)
    w_hit.rebuild_broadphase()
    for _ in range(60):
        w_hit.step(dt)

    # Mid-run PHYW v2 serialize / deserialize determinism check
    var v_snap_w = BlobWriter()
    w_hit.serialize(v_snap_w)
    var w_hit_restored = PhysicsWorld(gravity=9.81)
    var v_snap_r = BlobReader(v_snap_w.data.copy())
    w_hit_restored.deserialize(v_snap_r)

    for _ in range(60):
        w_hit.step(dt)
        w_hit_restored.step(dt)

    var vw_a = BlobWriter(); w_hit.serialize(vw_a)
    var vw_b = BlobWriter(); w_hit_restored.serialize(vw_b)
    assert_equal(fnv64(vw_a.data), fnv64(vw_b.data))
    assert_equal(float32_bits(w_hit.vehicle_state(v_hit).pos.z), float32_bits(w_hit_restored.vehicle_state(v_hit).pos.z))
    assert_equal(float32_bits(w_hit.vehicle_state(v_hit).wheel_states[0].bristle_x), float32_bits(w_hit_restored.vehicle_state(v_hit).wheel_states[0].bristle_x))

    var hit_z = w_hit.vehicle_state(v_hit).pos.z
    assert_true(w_hit.bodies[crate_slot].pos.z > Float32(4.2))
    assert_true(w_hit.bodies[crate_slot].linear_vel.length() > Float32(0.05))
    assert_true(hit_z < free_z - Float32(0.1))

    # Vehicle driving onto a Bvh8MeshCollider sloped ramp + dynamic wheel reaction plank
    var w_ramp = PhysicsWorld(gravity=9.81)
    var r_verts: List[Vec3] = [
        Vec3(-6.0, 0.0, -5.0), Vec3(6.0, 0.0, -5.0), Vec3(6.0, 0.0, 5.0), Vec3(-6.0, 0.0, 5.0),
        Vec3(-6.0, 1.4, 15.0), Vec3(6.0, 1.4, 15.0),
    ]
    var r_tris: List[Int32] = [
        Int32(0), Int32(2), Int32(1), Int32(0), Int32(3), Int32(2),
        Int32(3), Int32(4), Int32(5), Int32(3), Int32(5), Int32(2),
    ]
    _ = w_ramp.add_static_mesh(Bvh8MeshCollider.build(r_verts, r_tris))
    var plank_slot = w_ramp.add_box(Vec3(0.0, 0.08, 2.5), Vec3(1.5, 0.08, 0.6), mass=35.0)
    var car_ramp = VehicleChassis(1450.0)
    car_ramp.setup_four_wheel_car()
    car_ramp.pos = Vec3(0.0, 0.65, -2.0)
    car_ramp.set_controls(throttle=1.0, brake=0.0, steer=0.0)
    var v_ramp = w_ramp.add_vehicle(car_ramp^)
    w_ramp.rebuild_broadphase()
    for _ in range(180):
        w_ramp.step(dt)
    assert_true(w_ramp.vehicle_state(v_ramp).pos.z > Float32(6.5))
    assert_true(w_ramp.vehicle_state(v_ramp).pos.y > Float32(0.85))
    assert_true(
        w_ramp.bodies[plank_slot].linear_vel.length() > Float32(0.05)
        or (w_ramp.bodies[plank_slot].pos - Vec3(0.0, 0.08, 2.5)).length() > Float32(0.05)
    )

    # Vehicle on HeightfieldCollider terrain
    var w_hf = PhysicsWorld(gravity=9.81)
    var hf_heights = List[Float32](length=25, fill=0.0)
    for hz in range(5):
        for hx in range(5):
            hf_heights[hz * 5 + hx] = Float32(0.25) * Float32(hz)
    var hf = HeightfieldCollider.build(hf_heights, 5, 5, Vec3(-8.0, 0.0, -4.0), Float32(4.0))
    _ = w_hf.add_heightfield(hf)
    var car_hf = VehicleChassis(1450.0)
    car_hf.setup_four_wheel_car()
    car_hf.pos = Vec3(0.0, 0.90, 0.0)
    car_hf.set_controls(throttle=1.0, brake=0.0, steer=0.0)
    var v_hf = w_hf.add_vehicle(car_hf^)
    w_hf.rebuild_broadphase()
    for _ in range(60):
        w_hf.step(dt)
    assert_true(w_hf.vehicle_state(v_hf).pos.z > Float32(1.5))

    # 2. F11: Aero6DofState <-> PhysicsWorld two-way collision + terrain ground effect + MissileBatch8 detmath + rollback
    var w_afree = PhysicsWorld(gravity=9.81)
    var aero_init = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 60.0))
    var aero_cfg = Aero6DofConfig()
    var aero_ctrl = Aero6DofControls()
    aero_ctrl.throttle = Float32(1.0)
    var af_slot = w_afree.add_aero_body(aero_init, aero_cfg, aero_ctrl)
    w_afree.rebuild_broadphase()
    for _ in range(30):
        w_afree.step(dt)
    var aero_free_vz = w_afree.aero_state(af_slot).linear_vel.z

    var w_ahit = PhysicsWorld(gravity=9.81)
    var target_slot = w_ahit.add_box(Vec3(0.0, 500.0, 12.0), Vec3(1.5, 1.5, 1.0), mass=250.0)
    var ah_slot = w_ahit.add_aero_body(aero_init, aero_cfg, aero_ctrl)
    w_ahit.rebuild_broadphase()
    for _ in range(15):
        w_ahit.step(dt)
    var a_snap_w = BlobWriter()
    w_ahit.serialize(a_snap_w)
    var w_ahit_restored = PhysicsWorld(gravity=9.81)
    var a_snap_r = BlobReader(a_snap_w.data.copy())
    w_ahit_restored.deserialize(a_snap_r)
    for _ in range(15):
        w_ahit.step(dt)
        w_ahit_restored.step(dt)
    var aw_a = BlobWriter(); w_ahit.serialize(aw_a)
    var aw_b = BlobWriter(); w_ahit_restored.serialize(aw_b)
    assert_equal(fnv64(aw_a.data), fnv64(aw_b.data))

    var aero_hit_st = w_ahit.aero_state(ah_slot)
    assert_true(aero_hit_st.pos.x == aero_hit_st.pos.x and aero_hit_st.pos.y == aero_hit_st.pos.y and aero_hit_st.pos.z == aero_hit_st.pos.z)
    assert_true(w_ahit.bodies[target_slot].pos.z > Float32(12.2))
    assert_true(w_ahit.bodies[target_slot].linear_vel.z > Float32(0.5))
    assert_true(aero_hit_st.linear_vel.z < aero_free_vz - Float32(0.5))

    # Aero ground-effect height raycast over BVH runway + MissileBatch8 SIMD8 det_sin/det_cos
    var w_arunway = PhysicsWorld(gravity=9.81)
    _ = w_arunway.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 80.0), mass=0.0)
    var low_st = Aero6DofState(Vec3(0.0, 4.0, 0.0), Vec3(0.0, 0.0, 75.0))
    var low_slot = w_arunway.add_aero_body(low_st, aero_cfg, aero_ctrl)
    w_arunway.rebuild_broadphase()
    for _ in range(10):
        w_arunway.step(dt)
    assert_true(w_arunway.aero_state(low_slot).ge_lift_factor > Float32(1.15))

    var mb = MissileBatch8()
    mb.pos = V8(F8(-20.0), F8(120.0), F8(-400.0))
    mb.vel = V8(F8(0.0), F8(0.0), F8(380.0))
    var tgt_p = V8(F8(0.0), F8(120.0), F8(0.0))
    var tgt_v = V8.zero()
    var tgt_a = V8.zero()
    var r_init = step_missile_batch8(mb, tgt_p, tgt_v, tgt_a, dt)
    for _ in range(70):
        _ = step_missile_batch8(mb, tgt_p, tgt_v, tgt_a, dt)
    assert_true(mb.min_range[0] < r_init[0] * Float32(0.25))

    # 3. F12: XPBD ParticleChain & ParticleMesh two-way inside PhysicsWorld + swap_remove_body + rollback
    var g = Float32(9.81)
    var w_xp = PhysicsWorld(gravity=g)
    var dummy_xp = w_xp.add_sphere(Vec3(50.0, 50.0, 50.0), 0.2, mass=1.0)
    var weight_y0 = Float32(3.0)
    var weight_slot = w_xp.add_box(Vec3(0.0, weight_y0, 0.0), Vec3(0.25, 0.25, 0.25), mass=4.0)
    var weight_id = w_xp.body_id(weight_slot)
    var chain = ParticleChain.line(Vec3(0.0, 5.0, 0.0), Vec3(0.0, weight_y0, 0.0), 6, mass=0.5)
    chain.pin(0, Vec3(0.0, 5.0, 0.0))
    var att = RigidAttachment(5, weight_slot, Vec3(0.0, 0.0, 0.0), compliance=0.0)
    var c_slot = w_xp.add_xpbd_chain(chain^, att)
    # Off-center attachment starting at exact anchor_w (zero initial snap) to verify two-way torque
    var torque_box_slot = w_xp.add_box(Vec3(5.0, weight_y0, 0.0), Vec3(0.25, 0.25, 0.25), mass=4.0)
    var torque_box_id = w_xp.body_id(torque_box_slot)
    var chain_t = ParticleChain.line(Vec3(5.0, 5.0, 0.0), Vec3(5.15, weight_y0 + 0.20, 0.0), 6, mass=0.5)
    chain_t.pin(0, Vec3(5.0, 5.0, 0.0))
    var att_t = RigidAttachment(5, torque_box_slot, Vec3(0.15, 0.20, 0.0), compliance=0.0)
    _ = w_xp.add_xpbd_chain(chain_t^, att_t)
    # Swap-remove dummy body so weight_slot shifts, verifying persistent body_id resolution
    w_xp.swap_remove_body(dummy_xp)
    weight_slot = w_xp.find_slot_by_id(weight_id)
    torque_box_slot = w_xp.find_slot_by_id(torque_box_id)
    assert_true(weight_slot >= 0 and torque_box_slot >= 0)

    # Add suspended cloth sheet under a falling 1.5 kg dynamic sphere (`add_cloth_mesh`)
    var sphere_slot = w_xp.add_sphere(Vec3(10.0, 4.0, 0.0), 0.35, mass=1.5)
    var sphere_id = w_xp.body_id(sphere_slot)
    var cloth = ParticleMesh()
    var nx_c = 5
    var nz_c = 5
    for iz in range(nz_c):
        for ix in range(nx_c):
            var px = Float32(10.0) + (Float32(ix) - Float32(2.0)) * Float32(0.45)
            var pz = (Float32(iz) - Float32(2.0)) * Float32(0.45)
            var p_m = Float32(0.0) if (ix == 0 or ix == nx_c - 1) else Float32(0.25)
            _ = cloth.add_particle(Vec3(px, Float32(2.5), pz), p_m, Float32(0.08))
    for iz in range(nz_c):
        for ix in range(nx_c):
            var idx = iz * nx_c + ix
            if ix + 1 < nx_c:
                cloth.add_link(idx, idx + 1, Float32(1.0e-5))
            if iz + 1 < nz_c:
                cloth.add_link(idx, idx + nx_c, Float32(1.0e-5))
            if ix + 1 < nx_c and iz + 1 < nz_c:
                cloth.add_link(idx, idx + nx_c + 1, Float32(5.0e-5))
                cloth.add_link(idx + 1, idx + nx_c, Float32(5.0e-5))
    var m_slot = w_xp.add_cloth_mesh(cloth^, substeps=4, iterations=4)
    w_xp.rebuild_broadphase()

    for _ in range(30):
        w_xp.step(dt)
    var xp_snap_w = BlobWriter()
    w_xp.serialize(xp_snap_w)
    var w_xp_restored = PhysicsWorld(gravity=g)
    var xp_snap_r = BlobReader(xp_snap_w.data.copy())
    w_xp_restored.deserialize(xp_snap_r)
    for _ in range(30):
        w_xp.step(dt)
        w_xp_restored.step(dt)
    var xw_a = BlobWriter(); w_xp.serialize(xw_a)
    var xw_b = BlobWriter(); w_xp_restored.serialize(xw_b)
    assert_equal(fnv64(xw_a.data), fnv64(xw_b.data))
    assert_equal(float32_bits(w_xp.xpbd_chain(c_slot).lambdas[0]), float32_bits(w_xp_restored.xpbd_chain(c_slot).lambdas[0]))

    var t_60 = Float32(60.0) * dt
    var uncoupled_y = weight_y0 - Float32(0.5) * g * t_60 * t_60
    weight_slot = w_xp.find_slot_by_id(weight_id)
    torque_box_slot = w_xp.find_slot_by_id(torque_box_id)
    sphere_slot = w_xp.find_slot_by_id(sphere_id)
    var actual_y = w_xp.bodies[weight_slot].pos.y
    assert_true(actual_y < weight_y0)
    assert_true(actual_y > uncoupled_y + Float32(1.0))
    assert_true(w_xp.bodies[torque_box_slot].angular_vel.length() > Float32(0.1))
    assert_true(w_xp.cloth_chain(c_slot).position[5].y < Float32(4.9))

    # Sphere falling onto cloth mesh must be decelerated/deflected well above free-fall (`4.0 - 0.5*g*1^2 = -0.905`)
    var sphere_free_y = Float32(4.0) - Float32(0.5) * g * t_60 * t_60
    assert_true(w_xp.bodies[sphere_slot].pos.y > sphere_free_y + Float32(0.6))
    assert_true(w_xp.cloth_mesh(m_slot).position[12].y < Float32(2.45))

    # 4. F13: ArticulatedRagdoll & RagdollStateMachine <-> PhysicsWorld (BVH8 + dynamic RigidBody + rollback)
    var w_rag = PhysicsWorld(gravity=9.81)
    var bvh_v: List[Vec3] = [
        Vec3(-12.0, 0.0, -12.0), Vec3(12.0, 0.0, -12.0),
        Vec3(12.0, 0.0, 12.0), Vec3(-12.0, 0.0, 12.0),
    ]
    var bvh_i: List[Int32] = [Int32(0), Int32(2), Int32(1), Int32(0), Int32(3), Int32(2)]
    _ = w_rag.add_static_mesh(Bvh8MeshCollider.build(bvh_v, bvh_i))

    var hit_crate = w_rag.add_box(Vec3(0.0, 1.2, -1.0), Vec3(0.35, 0.35, 0.35), mass=12.0)
    w_rag.set_velocity(hit_crate, Vec3(0.0, 0.0, 8.0))
    var prop_crate = w_rag.add_box(Vec3(0.0, 0.50, 0.90), Vec3(0.45, 0.50, 0.45), mass=4.0)

    var biped = _make_test_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var rag = ArticulatedRagdoll()
    _ = rag.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
    _ = rag.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)
    rag.handoff_from_skeleton(skel, Placement(), biped, dt)
    rag.go_limp()
    var r_slot = w_rag.add_ragdoll(rag^)
    w_rag.rebuild_broadphase()

    for _ in range(30):
        w_rag.step(dt)
    var r_snap_w = BlobWriter()
    w_rag.serialize(r_snap_w)
    var w_rag_restored = PhysicsWorld(gravity=9.81)
    w_rag_restored.meshes.append(Bvh8MeshCollider.build(bvh_v, bvh_i))
    var r_snap_r = BlobReader(r_snap_w.data.copy())
    w_rag_restored.deserialize(r_snap_r)
    for _ in range(60):
        w_rag.step(dt)
        w_rag_restored.step(dt)

    var rw_a = BlobWriter(); w_rag.serialize(rw_a)
    var rw_b = BlobWriter(); w_rag_restored.serialize(rw_b)
    assert_equal(fnv64(rw_a.data), fnv64(rw_b.data))
    assert_equal(
        float32_bits(w_rag.ragdoll_state(r_slot).pelvis_world_position().y),
        float32_bits(w_rag_restored.ragdoll_state(r_slot).pelvis_world_position().y),
    )

    var rag_res = w_rag.ragdoll_state(r_slot)
    assert_true(rag_res.pelvis_world_position().y >= Float32(-0.1))
    assert_true(rag_res.total_linear_momentum().z > Float32(0.0) or rag_res.pelvis_world_position().z > Float32(0.05))
    assert_true(w_rag.bodies[hit_crate].linear_vel.z < Float32(7.5))
    assert_true(
        w_rag.bodies[prop_crate].linear_vel.length() > Float32(0.15)
        or (w_rag.bodies[prop_crate].pos - Vec3(0.0, 0.50, 0.90)).length() > Float32(0.1)
    )

    # RagdollStateMachine.attach_to_world + update_world lifecycle on Bvh8MeshCollider floor
    var w_rsm = PhysicsWorld(gravity=9.81)
    _ = w_rsm.add_static_mesh(Bvh8MeshCollider.build(bvh_v, bvh_i))
    w_rsm.rebuild_broadphase()

    var rsm = RagdollStateMachine()
    rsm.supine_getup_clip = 0
    rsm.prone_getup_clip = 1
    _ = rsm.ragdoll.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
    _ = rsm.ragdoll.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)

    var rest_pose = Pose()
    rest_pose.rest(biped)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1, 1, 1), 0.0)
    skel.reset_to_rest(biped)
    _ = rsm.attach_to_world(w_rsm)

    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(15.0, 0.0, 0.0), Vec3(0.0, 1.3, 0.0), go_limp_threshold=100.0, dt=dt)
    assert_equal(rsm.state, RAGDOLL_SM_ACTIVE_PD)
    rsm.update_world(w_rsm, skel, place, biped, rest_pose, dt)

    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(0.0, 15.0, -160.0), Vec3(0.0, 1.35, 0.0), go_limp_threshold=100.0, dt=dt)
    assert_equal(rsm.state, RAGDOLL_SM_LIMP)
    var saw_getup_w = False
    for _ in range(240):
        rsm.update_world(w_rsm, skel, place, biped, rest_pose, dt)
        if rsm.state == RAGDOLL_SM_GETUP_SUPINE or rsm.state == RAGDOLL_SM_GETUP_PRONE:
            saw_getup_w = True
        if saw_getup_w and rsm.state == RAGDOLL_SM_ANIMATED:
            break
    assert_true(saw_getup_w)
    assert_equal(rsm.state, RAGDOLL_SM_ANIMATED)
    assert_equal(rsm.recovery_count, 1)
    assert_true(rsm.last_pelvis_snap_m < Float32(1.0e-4))

    # Cycle 2 RagdollStateMachine island hygiene & two-way collision after recovery
    var rsm_slot = rsm.world_ragdoll_slot
    assert_true(rsm_slot >= 0 and rsm_slot < len(w_rsm.ragdolls))
    var rsm_limb0_id = w_rsm.ragdolls[rsm_slot].limb_body_ids[0]
    var rsm_limb0_slot = -1
    for k in range(len(w_rsm.bodies)):
        if w_rsm.bodies[k].id == rsm_limb0_id:
            rsm_limb0_slot = k
            break
    assert_true(rsm_limb0_slot >= 0)
    assert_equal(w_rsm.islands.label[rsm_limb0_slot], NO_ISLAND)

    var cyc2_crate_pos = place.position + Vec3(0.0, 0.55, 0.55)
    var cyc2_crate = w_rsm.add_box(cyc2_crate_pos, Vec3(0.35, 0.35, 0.35), mass=3.0)
    w_rsm.rebuild_broadphase()

    rsm.enter_active_pd(skel, place, biped, dt)
    assert_equal(rsm.state, RAGDOLL_SM_ACTIVE_PD)
    rsm.update_world(w_rsm, skel, place, biped, rest_pose, dt)
    rsm_limb0_slot = w_rsm.find_slot_by_id(rsm_limb0_id)
    assert_true(rsm_limb0_slot >= 0)
    assert_true(w_rsm.islands.label[rsm_limb0_slot] != NO_ISLAND)

    rsm.apply_hit_impulse(
        skel,
        place,
        biped,
        1,
        Vec3(0.0, 10.0, 180.0),
        place.position + Vec3(0.0, 1.35, 0.0),
        go_limp_threshold=100.0,
        dt=dt,
    )
    assert_equal(rsm.state, RAGDOLL_SM_LIMP)
    var saw_getup_2 = False
    for step2 in range(240):
        rsm.update_world(w_rsm, skel, place, biped, rest_pose, dt)
        if step2 == 0:
            var s0_now = w_rsm.find_slot_by_id(rsm_limb0_id)
            assert_true(s0_now >= 0 and w_rsm.islands.label[s0_now] != NO_ISLAND)
        if rsm.state == RAGDOLL_SM_GETUP_SUPINE or rsm.state == RAGDOLL_SM_GETUP_PRONE:
            saw_getup_2 = True
        if saw_getup_2 and rsm.state == RAGDOLL_SM_ANIMATED:
            break
    assert_true(saw_getup_2)
    assert_equal(rsm.state, RAGDOLL_SM_ANIMATED)
    assert_equal(rsm.recovery_count, 2)
    assert_true(
        w_rsm.bodies[cyc2_crate].linear_vel.length() > Float32(0.05)
        or (w_rsm.bodies[cyc2_crate].pos - cyc2_crate_pos).length() > Float32(0.05)
    )

    # Orphaned joint swap_remove_body safety (ragdoll root limb0 & regular weld Joint)
    var w_orph = PhysicsWorld(gravity=9.81)
    var rag_o = ArticulatedRagdoll()
    _ = rag_o.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rag_o.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rag_o.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
    _ = rag_o.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)
    skel.reset_to_rest(biped)
    rag_o.handoff_from_skeleton(skel, Placement(), biped, dt)
    var or_slot = w_orph.add_ragdoll(rag_o^)
    var or_l0_id = w_orph.ragdolls[or_slot].limb_body_ids[0]
    var or_l1_id = w_orph.ragdolls[or_slot].limb_body_ids[1]
    var or_l0_slot = w_orph.find_slot_by_id(or_l0_id)
    var or_l1_slot = w_orph.find_slot_by_id(or_l1_id)
    var or_l1_y0 = w_orph.bodies[or_l1_slot].pos.y
    assert_true(or_l1_y0 > Float32(1.0))

    var wb_a = w_orph.add_box(Vec3(0.0, 5.0, 0.0), Vec3(0.25, 0.25, 0.25), mass=2.0)
    var wb_a_id = w_orph.body_id(wb_a)
    var wb_b = w_orph.add_box(Vec3(0.6, 5.0, 0.0), Vec3(0.25, 0.25, 0.25), mass=2.0)
    var wb_b_id = w_orph.body_id(wb_b)
    _ = w_orph.add_weld_joint(wb_a, wb_b, Vec3(0.3, 5.0, 0.0))

    w_orph.swap_remove_body(or_l0_slot)
    var wb_a_slot_now = w_orph.find_slot_by_id(wb_a_id)
    if wb_a_slot_now >= 0:
        w_orph.swap_remove_body(wb_a_slot_now)
    for _ in range(5):
        w_orph.step(dt)
    var or_l1_after = w_orph.find_slot_by_id(or_l1_id)
    var wb_b_after = w_orph.find_slot_by_id(wb_b_id)
    assert_true(or_l1_after >= 0 and wb_b_after >= 0)
    assert_true(w_orph.bodies[or_l1_after].pos.y > Float32(0.8))
    assert_true(abs(w_orph.bodies[or_l1_after].pos.y - or_l1_y0) < Float32(0.25))
    assert_true(w_orph.bodies[wb_b_after].pos.y > Float32(4.5))

    # < 4-wheel VehicleChassis gravity fallback (must not float in zero gravity)
    var w_v0 = PhysicsWorld(gravity=9.81)
    var vc0 = VehicleChassis(1000.0)
    vc0.pos = Vec3(0.0, 5.0, 0.0)
    var v0_slot = w_v0.add_vehicle(vc0^)
    w_v0.rebuild_broadphase()
    for _ in range(30):
        w_v0.step(dt)
    assert_true(w_v0.vehicle_state(v0_slot).pos.y < Float32(4.0))

    # Iter 3 Regression 1: Non-origin static slot_b, SixDofMotorJoint(body_a=static), & stationary-kinematic joint anchors
    var w_janch = PhysicsWorld(gravity=9.81)
    var st_b = w_janch.add_box(Vec3(15.0, 5.0, 0.0), Vec3(0.4, 0.4, 0.4), mass=0.0)
    var dyn_a = w_janch.add_box(Vec3(16.0, 5.0, 0.0), Vec3(0.3, 0.3, 0.3), mass=2.0)
    _ = w_janch.add_weld_joint(dyn_a, st_b, Vec3(15.5, 5.0, 0.0))

    var st_rev_b = w_janch.add_box(Vec3(15.0, 5.0, 8.0), Vec3(0.4, 0.4, 0.4), mass=0.0)
    var dyn_rev_a = w_janch.add_box(Vec3(16.0, 5.0, 8.0), Vec3(0.3, 0.3, 0.3), mass=2.0)
    _ = w_janch.add_joint(JointDef(JOINT_REVOLUTE, dyn_rev_a, st_rev_b, Vec3(15.5, 5.0, 8.0), Vec3(0, 0, 1)))

    var st_6 = w_janch.add_box(Vec3(20.0, 5.0, 0.0), Vec3(0.4, 0.4, 0.4), mass=0.0)
    var dyn_6 = w_janch.add_box(Vec3(21.0, 5.0, 0.0), Vec3(0.3, 0.3, 0.3), mass=2.0)
    var j6_st = SixDofMotorJoint()
    j6_st.body_a = w_janch.body_id(st_6)
    j6_st.body_b = w_janch.body_id(dyn_6)
    j6_st.local_anchor_a = Vec3(0.5, 0.0, 0.0)
    j6_st.local_anchor_b = Vec3(-0.5, 0.0, 0.0)
    j6_st.enable_linear_limit = True
    j6_st.linear_lower = Vec3()
    j6_st.linear_upper = Vec3()
    _ = w_janch.add_six_dof_joint(j6_st^)

    var kin_a = w_janch.add_kinematic_box(Vec3(25.0, 5.0, 0.0), Vec3(0.4, 0.4, 0.4))
    var dyn_k = w_janch.add_box(Vec3(26.0, 5.0, 0.0), Vec3(0.3, 0.3, 0.3), mass=2.0)
    _ = w_janch.add_weld_joint(kin_a, dyn_k, Vec3(25.5, 5.0, 0.0))

    for _ in range(30):
        w_janch.step(dt)
    assert_true((w_janch.bodies[dyn_a].pos - Vec3(16.0, 5.0, 0.0)).length() < Float32(0.25))
    assert_true(w_janch.bodies[dyn_rev_a].pos.x > Float32(15.0) and w_janch.bodies[dyn_rev_a].pos.z > Float32(7.5))
    assert_true(abs((w_janch.bodies[dyn_6].pos - Vec3(20.5, 5.0, 0.0)).length() - Float32(0.5)) < Float32(0.15))
    assert_true(w_janch.bodies[dyn_6].pos.x > Float32(19.8) and w_janch.bodies[dyn_6].pos.y > Float32(4.2))
    assert_true((w_janch.bodies[dyn_k].pos - Vec3(26.0, 5.0, 0.0)).length() < Float32(0.25))

    # Iter 3 Regression 2: ParticleChain (low & high drop) & high-drop ParticleMesh draping over rotated BVH8 wedge + Heightfield
    var w_drape = PhysicsWorld(gravity=9.81)
    var wedge_v: List[Vec3] = [
        Vec3(-2.0, 0.0, -2.0), Vec3(2.0, 0.0, -2.0), Vec3(0.0, 1.5, -2.0),
        Vec3(-2.0, 0.0, 2.0), Vec3(2.0, 0.0, 2.0), Vec3(0.0, 1.5, 2.0),
    ]
    var wedge_i: List[Int32] = [
        Int32(0), Int32(3), Int32(2), Int32(3), Int32(5), Int32(2),
        Int32(2), Int32(4), Int32(1), Int32(2), Int32(5), Int32(4),
    ]
    var wedge_slot = w_drape.add_static_mesh(
        Bvh8MeshCollider.build(wedge_v, wedge_i),
        pos=Vec3(5.0, 0.0, 0.0),
        rot=Quat.from_axis_angle(Vec3(0, 1, 0), Float32(0.35)),
    )
    _ = wedge_slot
    var ch_low = ParticleChain.line(Vec3(5.0, 1.65, -0.6), Vec3(5.0, 1.65, 0.6), 7, mass=0.2)
    var ch_high = ParticleChain.line(Vec3(5.0, 2.20, -0.6), Vec3(5.0, 2.20, 0.6), 7, mass=0.2)
    var cs_low = w_drape.add_cloth_chain(ch_low^, substeps=4, iterations=4)
    var cs_high = w_drape.add_cloth_chain(ch_high^, substeps=4, iterations=4)

    var pm_drop = ParticleMesh()
    for iz in range(5):
        for ix in range(5):
            var px = Float32(5.0) + (Float32(ix) - Float32(2.0)) * Float32(0.3)
            var pz = (Float32(iz) - Float32(2.0)) * Float32(0.3)
            _ = pm_drop.add_particle(Vec3(px, Float32(6.5), pz), Float32(0.2), Float32(0.05))
    for iz in range(5):
        for ix in range(5):
            var idx = iz * 5 + ix
            if ix + 1 < 5:
                pm_drop.add_link(idx, idx + 1, Float32(1.0e-5))
            if iz + 1 < 5:
                pm_drop.add_link(idx, idx + 5, Float32(1.0e-5))
    var ms_drop = w_drape.add_cloth_mesh(pm_drop^, substeps=4, iterations=4)

    var hf_hill_h = List[Float32](length=25, fill=0.0)
    for hz in range(5):
        for hx in range(5):
            var dx = Float32(hx) - Float32(2.0)
            var dz = Float32(hz) - Float32(2.0)
            hf_hill_h[hz * 5 + hx] = max(Float32(0.0), Float32(1.5) - Float32(0.35) * (dx * dx + dz * dz))
    _ = w_drape.add_heightfield(HeightfieldCollider.build(hf_hill_h, 5, 5, Vec3(13.0, 0.0, -2.0), Float32(1.0)))
    var ch_hf = ParticleChain.line(Vec3(15.0, 5.5, -0.6), Vec3(15.0, 5.5, 0.6), 7, mass=0.2)
    var cs_hf = w_drape.add_cloth_chain(ch_hf^, substeps=4, iterations=4)

    w_drape.rebuild_broadphase()
    for _ in range(45):
        w_drape.step(dt)
    var dr_snap_w = BlobWriter()
    w_drape.serialize(dr_snap_w)
    var w_drape_b = PhysicsWorld(gravity=9.81)
    w_drape_b.meshes.append(Bvh8MeshCollider.build(wedge_v, wedge_i))
    w_drape_b.heightfields.append(HeightfieldCollider.build(hf_hill_h, 5, 5, Vec3(13.0, 0.0, -2.0), Float32(1.0)))
    var dr_snap_r = BlobReader(dr_snap_w.data.copy())
    w_drape_b.deserialize(dr_snap_r)
    for _ in range(45):
        w_drape.step(dt)
        w_drape_b.step(dt)
    var drw_a = BlobWriter(); w_drape.serialize(drw_a)
    var drw_b = BlobWriter(); w_drape_b.serialize(drw_b)
    assert_equal(fnv64(drw_a.data), fnv64(drw_b.data))
    assert_true(w_drape.cloth_chain(cs_low).position[3].y > Float32(0.3))
    assert_true(w_drape.cloth_chain(cs_high).position[3].y > Float32(0.3))
    assert_true(w_drape.cloth_mesh(ms_drop).position[12].y > Float32(0.3))
    assert_true(w_drape.cloth_chain(cs_hf).position[3].y > Float32(0.3))

    # Iter 3 Regression 3: Aero6DofState fuel exhaustion thrust cutoff (main afterburner & hover nozzles)
    var w_fuel = PhysicsWorld(gravity=0.0)
    var f_st = Aero6DofState(Vec3(0.0, 1000.0, 0.0), Vec3(0.0, 0.0, 0.0))
    f_st.fuel_mass = Float32(100.0)
    var f_cfg = Aero6DofConfig()
    f_cfg.sfc_kg_per_n_s = Float32(0.001)
    f_cfg.cd0_0 = Float32(0.0)
    f_cfg.cl0 = Float32(0.0)
    var f_ctrl = Aero6DofControls()
    f_ctrl.gravity = Float32(0.0)
    f_ctrl.throttle = Float32(1.0)
    var f_slot = w_fuel.add_aero_body(f_st, f_cfg, f_ctrl)

    # Also verify hover nozzles produce zero thrust when fuel_mass == 0 and sfc > 0
    var f_st_hov = Aero6DofState(Vec3(100.0, 1000.0, 0.0), Vec3(0.0, 0.0, 0.0))
    f_st_hov.fuel_mass = Float32(0.0)
    var f_ctrl_hov = Aero6DofControls()
    f_ctrl_hov.gravity = Float32(0.0)
    f_ctrl_hov.hover_left_throttle = Float32(1.0)
    f_ctrl_hov.hover_right_throttle = Float32(1.0)
    var fh_slot = w_fuel.add_aero_body(f_st_hov, f_cfg, f_ctrl_hov)

    for _ in range(60):
        w_fuel.step(dt)
    var st_dry = w_fuel.aero_state(f_slot)
    assert_equal(st_dry.fuel_mass, Float32(0.0))
    var vz_dry = st_dry.linear_vel.z
    assert_true(vz_dry > Float32(1.0))
    for _ in range(30):
        w_fuel.step(dt)
    var st_after = w_fuel.aero_state(f_slot)
    assert_equal(st_after.fuel_mass, Float32(0.0))
    # With cd0_0=0, cl0=0, gravity=0, alpha=0 and empty fuel, thrust must be 0 so velocity cannot increase
    assert_true(st_after.linear_vel.z <= vz_dry + Float32(1.0e-4))
    assert_true(w_fuel.aero_state(fh_slot).linear_vel.length() < Float32(1.0e-4))


def _v4_assert_approx(actual: Float32, expected: Float32, eps: Float32 = 0.001, msg: String = "") raises:
    if not (abs(actual - expected) <= eps):
        raise Error("Assertion failed (" + msg + "): expected " + String(expected) + ", got " + String(actual))


def _v4_assert_vec3_approx(actual: Vec3, expected: Vec3, eps: Float32 = 0.001, msg: String = "") raises:
    if (
        not (abs(actual.x - expected.x) <= eps)
        or not (abs(actual.y - expected.y) <= eps)
        or not (abs(actual.z - expected.z) <= eps)
    ):
        raise Error(
            "Vec3 assertion failed ("
            + msg
            + "): expected ("
            + String(expected.x)
            + ","
            + String(expected.y)
            + ","
            + String(expected.z)
            + "), got ("
            + String(actual.x)
            + ","
            + String(actual.y)
            + ","
            + String(actual.z)
            + ")"
        )


def _v4_add_node(
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


def _v4_build_battloid_rig() -> NativeModel:
    var m = NativeModel()
    var pelvis = _v4_add_node(m, "Pelvis", -1, Vec3(0, 1.0, 0))
    var chest = _v4_add_node(m, "Chest", pelvis, Vec3(0, 0.45, 0))
    var head = _v4_add_node(m, "Head", chest, Vec3(0, 0.30, 0))
    var uarm_r = _v4_add_node(m, "UpperArm.R", chest, Vec3(-0.28, 0.22, 0))
    var farm_r = _v4_add_node(m, "Forearm.R", uarm_r, Vec3(0, -0.30, 0))
    var hand_r = _v4_add_node(m, "Hand.R", farm_r, Vec3(0, -0.26, 0))
    var thigh_l = _v4_add_node(m, "Thigh.L", pelvis, Vec3(0.18, 0, 0))
    var shin_l = _v4_add_node(m, "Shin.L", thigh_l, Vec3(0, -0.44, 0))
    var foot_l = _v4_add_node(m, "Foot.L", shin_l, Vec3(0, -0.44, 0))
    var thigh_r = _v4_add_node(m, "Thigh.R", pelvis, Vec3(-0.18, 0, 0))
    var shin_r = _v4_add_node(m, "Shin.R", thigh_r, Vec3(0, -0.44, 0))
    var foot_r = _v4_add_node(m, "Foot.R", shin_r, Vec3(0, -0.44, 0))
    _ = head
    _ = hand_r
    _ = foot_l
    _ = foot_r
    return m^


def _v4_configure_battloid_ragdoll(model: NativeModel) -> ArticulatedRagdoll:
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


def _v4_make_flat_bvh_quad(half_size: Float32 = 50.0, y: Float32 = 0.0) -> Bvh8MeshCollider:
    var verts = List[Vec3]()
    verts.append(Vec3(-half_size, y, -half_size))
    verts.append(Vec3(-half_size, y, half_size))
    verts.append(Vec3(half_size, y, -half_size))
    verts.append(Vec3(half_size, y, half_size))
    var idxs = List[Int32]()
    idxs.append(0); idxs.append(1); idxs.append(2)
    idxs.append(2); idxs.append(1); idxs.append(3)
    return Bvh8MeshCollider.build(verts, idxs)


def _v4_make_ramp_bvh(slope_rad: Float32, half_len: Float32 = 30.0) -> Bvh8MeshCollider:
    var c = det_cos(slope_rad)
    var s = det_sin(slope_rad)
    var verts = List[Vec3]()
    verts.append(Vec3(-15.0, -half_len * s, -half_len * c))
    verts.append(Vec3(-15.0, half_len * s, half_len * c))
    verts.append(Vec3(15.0, -half_len * s, -half_len * c))
    verts.append(Vec3(15.0, half_len * s, half_len * c))
    var idxs = List[Int32]()
    idxs.append(0); idxs.append(1); idxs.append(2)
    idxs.append(2); idxs.append(1); idxs.append(3)
    return Bvh8MeshCollider.build(verts, idxs)


def test_i_v4_anim_1_cubicspline_step_and_springs() raises:
    var sp = SpringFloat(10.0)
    sp.update(0.0, 0.25, 0.25)
    _v4_assert_approx(sp.value, 5.0, 0.002, "SpringFloat reaches exact 0.5x at t = halflife")
    assert_true(halflife_to_damping(0.1) > 33.5)

    var model = NativeModel()
    var n0 = _v4_add_node(model, "Root", -1, Vec3(0, 0, 0))
    var n1 = _v4_add_node(model, "BoneStep", n0, Vec3(0, 1, 0))

    var anim = NativeAnimation()
    anim.name = "SplineAndStep"
    anim.duration = 2.0

    var ch_t = NativeChannel()
    ch_t.node = n0
    ch_t.path = 0
    ch_t.interp_mode = INTERP_CUBICSPLINE
    ch_t.times = [Float32(0.0), Float32(2.0)]
    ch_t.values = [
        Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(2.0), Float32(4.0),
        Float32(0.0), Float32(-2.0), Float32(4.0),
        Float32(4.0), Float32(0.0), Float32(8.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
    ]
    anim.channels.append(ch_t^)

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
    sample_pose(model, 0, Float32(1.0), pose)
    _v4_assert_vec3_approx(pose.t[n0], Vec3(2.0, 1.0, 4.0), 0.001, "CUBICSPLINE Hermite midpoint translation")
    _v4_assert_approx(pose.r[n0].length(), 1.0, 0.0005, "CUBICSPLINE Hermite quaternion normalized")
    _v4_assert_approx(pose.t[n1].x, 20.0, 0.0005, "STEP channel hits middle keyframe at t == 1.0")

    sample_pose(model, 0, Float32(0.99), pose)
    _v4_assert_approx(pose.t[n1].x, 10.0, 0.0005, "STEP channel holds key 0 prior to key 1")

    sample_pose(model, 0, Float32(2.0), pose)
    _v4_assert_approx(pose.t[n1].x, 30.0, 0.0005, "STEP channel reaches final keyframe at t == duration")

    var sv3 = SpringVec3(Vec3(4, 0, 0))
    sv3.update(Vec3(), 0.25, 0.25)
    _v4_assert_approx(sv3.value.x, 2.0, 0.01, "SpringVec3 reaches 0.5x at halflife")
    var sa = SpringAngle(170.0)
    sa.update(-170.0, 0.15, 0.05)
    assert_true(fast_negexp(1.0) > 0.35 and fast_negexp(1.0) < 0.38)
    var empty_glb = parse_native_glb(List[UInt8]())
    assert_true(not empty_glb.loaded())


def test_j_v4_anim_2_skeleton_instance_topo_and_prev_matrices() raises:
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
    assert_true(pos_1 < pos_2 and pos_2 < pos_0)
    _v4_assert_vec3_approx(skel.global_position(0), Vec3(0.4, 1.3, 0.5), 0.001, "FK leaf position in rest pose")

    skel.snapshot_prev_matrices()
    assert_true(skel.has_prev)
    skel.pose.r[2] = angle_axis(PI * Float32(0.5), Vec3(0, 1, 0))
    skel.update_matrices(model)

    var prev_leaf = Vec3(
        skel.prev_world_matrices[0].at(0, 3),
        skel.prev_world_matrices[0].at(1, 3),
        skel.prev_world_matrices[0].at(2, 3),
    )
    _v4_assert_vec3_approx(prev_leaf, Vec3(0.4, 1.3, 0.5), 0.001, "prev_world_matrices preserves frame n-1 leaf")
    _v4_assert_vec3_approx(skel.global_position(0), Vec3(0.9, 1.3, 0.0), 0.001, "world_matrices reflects frame n rotated leaf")
    assert_equal(model.transforms_revision, rev_before)


def test_k_v4_anim_3_bollo_inertializer_and_socket_reparent() raises:
    var model = NativeModel()
    var root = _v4_add_node(model, "Pelvis", -1, Vec3(0, 1, 0))
    var belly = _v4_add_node(model, "BellyHardpoint", root, Vec3(0, -0.2, 0.25))
    var r_hand = _v4_add_node(model, "Hand.R", root, Vec3(-0.45, 0.15, 0.35), angle_axis(0.4, Vec3(1, 0, 0)))
    var gunpod = _v4_add_node(model, "Gunpod", belly, Vec3(0, 0, 0.1))

    var inert = Inertializer()
    var p_a = Pose(4)
    var p_b = Pose(4)
    var p_ahead = Pose(4)
    p_a.r[0] = angle_axis(2.95, Vec3(0, 1, 0))
    p_b.r[0] = Quat.identity()
    p_ahead.r[0] = angle_axis(-0.25, Vec3(0, 1, 0))

    var work = Pose()
    work.assign(p_a)
    inert.apply(work, p_a, False, 0.08, Float32(1.0 / 60.0))
    p_a.r[0] = angle_axis(3.02, Vec3(0, 1, 0))
    work.assign(p_a)
    inert.apply(work, p_a, False, 0.08, Float32(1.0 / 60.0))

    work.assign(p_b)
    inert.apply(work, p_ahead, True, 0.08, Float32(1.0 / 60.0))
    assert_true(not inert.is_settled)

    var prev_ang = inert.rot_ang[0]
    assert_true(prev_ang > 2.0 and prev_ang <= PI)
    for _ in range(90):
        work.assign(p_b)
        inert.apply(work, p_b, False, 0.08, Float32(1.0 / 60.0))
        var cur_ang = inert.rot_ang[0]
        assert_true(cur_ang <= prev_ang + 1.0e-5)
        assert_true(cur_ang >= 0.0)
        prev_ang = cur_ang

    assert_true(inert.is_settled)
    _v4_assert_approx(inert.largest_offset(), 0.0, 1.0e-5, "settled largest_offset is 0")

    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var old_gun_world = skel.world_matrices[gunpod].copy()
    var old_gun_pos = Vec3(old_gun_world.at(0, 3), old_gun_world.at(1, 3), old_gun_world.at(2, 3))

    model.nodes[belly].children.clear()
    model.nodes[r_hand].children.append(gunpod)
    skel.parents[gunpod] = r_hand

    var hand_world = skel.world_matrices[r_hand].copy()
    var grip_offset = trs(Vec3(0.0, 0.0, 0.20), angle_axis(-0.2, Vec3(1, 0, 0)), Vec3(1, 1, 1))
    var sock_inert = Inertializer()
    reparent_socket_inertial(skel, sock_inert, gunpod, hand_world, grip_offset)

    var new_gun_world_t0 = skel.world_matrices[gunpod].copy()
    var new_gun_pos_t0 = Vec3(new_gun_world_t0.at(0, 3), new_gun_world_t0.at(1, 3), new_gun_world_t0.at(2, 3))
    _v4_assert_vec3_approx(new_gun_pos_t0, old_gun_pos, 0.0005, "reparent_socket_inertial has 0.0mm world pop at handoff")
    assert_true(not sock_inert.is_settled)

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
    _v4_assert_vec3_approx(skel.global_position(gunpod), exp_pos, 0.002, "reparented gunpod converges smoothly onto hand grip")


def test_l_v4_anim_4_blend_space_2d_and_loop_root_motion() raises:
    var bs_cart = BlendSpace2DDef()
    bs_cart.name = "Strafe2D"
    bs_cart.param_x = 0
    bs_cart.param_y = 1
    bs_cart.is_polar = False
    bs_cart.samples.append(BlendSample2DDef(0, 0.0, 0.0))
    bs_cart.samples.append(BlendSample2DDef(1, 0.0, 4.0))
    bs_cart.samples.append(BlendSample2DDef(2, 4.0, 0.0))
    bs_cart.samples.append(BlendSample2DDef(3, -4.0, 0.0))

    var w_exact = bs_cart.weights(0.0, 4.0)
    _v4_assert_approx(w_exact[1], 1.0, 0.0005, "Johansen 2D hits exact 1.0 at authored sample")
    _v4_assert_approx(w_exact[0] + w_exact[2] + w_exact[3], 0.0, 0.0005, "other samples 0.0 at authored sample")

    var w_mid = bs_cart.weights(1.5, 2.0)
    var sum_mid = w_mid[0] + w_mid[1] + w_mid[2] + w_mid[3]
    _v4_assert_approx(sum_mid, 1.0, 0.0005, "Johansen 2D weights partition unity")
    assert_true(w_mid[0] >= 0.0 and w_mid[1] > 0.1 and w_mid[2] > 0.1)

    var bs_pol = BlendSpace2DDef()
    bs_pol.is_polar = True
    bs_pol.samples.append(BlendSample2DDef(0, 0.0, 5.0))
    bs_pol.samples.append(BlendSample2DDef(1, PI * Float32(0.5), 5.0))
    bs_pol.samples.append(BlendSample2DDef(2, PI, 5.0))
    bs_pol.samples.append(BlendSample2DDef(3, -PI * Float32(0.5), 5.0))
    var wp_plus = bs_pol.weights(PI - Float32(0.01), 5.0)
    var wp_minus = bs_pol.weights(-PI + Float32(0.01), 5.0)
    _v4_assert_approx(wp_plus[2], wp_minus[2], 0.01, "polar 2D blend space has zero discontinuity across +-pi")
    assert_true(wp_plus[2] > 0.98)

    var model = NativeModel()
    var root = _v4_add_node(model, "Root", -1, Vec3(0, 0, 0))
    var arm = _v4_add_node(model, "Arm", root, Vec3(0, 1, 0))

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

    var rm = extract_root_motion(model, 0, Float32(0.90), Float32(0.15), root)
    _v4_assert_vec3_approx(rm[0], Vec3(0.0, 0.0, 1.5), 0.002, "extract_root_motion wraps cleanly across loop boundary (>0.5m)")
    _v4_assert_approx(rm[1], 0.125, 0.005, "extract_root_motion extracts loop-wrapped yaw delta")

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
    _v4_assert_approx(animator.pose.t[arm].x, 0.5, 0.02, "Animator evaluates 50/50 2D blend space state")


def test_m_v4_anim_5_ik_pole_fourbar_digitigrade_and_footplant() raises:
    var model = NativeModel()
    var pelvis = _v4_add_node(model, "Pelvis", -1, Vec3(0, 1.0, 0))
    var hip_l = _v4_add_node(model, "Hip.L", pelvis, Vec3(0.2, 0.0, 0.0), angle_axis(-0.35, Vec3(1, 0, 0)))
    var knee_l = _v4_add_node(model, "Knee.L", hip_l, Vec3(0, -0.45, 0), angle_axis(0.70, Vec3(1, 0, 0)))
    var ankle_l = _v4_add_node(model, "Ankle.L", knee_l, Vec3(0, -0.45, 0), angle_axis(-0.35, Vec3(1, 0, 0)))
    var hip_r = _v4_add_node(model, "Hip.R", pelvis, Vec3(-0.2, 0.0, 0.0))
    var knee_r = _v4_add_node(model, "Knee.R", hip_r, Vec3(0, -0.45, 0))
    var ankle_r = _v4_add_node(model, "Ankle.R", knee_r, Vec3(0, -0.45, 0))

    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var foot_goal = Vec3(0.2, 0.28, 0.05)

    var pole_side = Vec3(1.2, 0.60, 0.0)
    skel.solve_two_bone(hip_l, knee_l, ankle_l, foot_goal, pole_side, model)
    _v4_assert_vec3_approx(skel.global_position(ankle_l), foot_goal, 0.002, "solve_two_bone_pole reaches foot goal (side pole)")
    assert_true(skel.global_position(knee_l).x > 0.35)

    skel.reset_to_rest(model)
    var pole_fwd = Vec3(0.2, 0.60, 1.2)
    skel.solve_two_bone(hip_l, knee_l, ankle_l, foot_goal, pole_fwd, model)
    _v4_assert_vec3_approx(skel.global_position(ankle_l), foot_goal, 0.002, "solve_two_bone_pole reaches foot goal (forward pole)")
    assert_true(skel.global_position(knee_l).z > 0.25)

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
        _v4_assert_approx((jb - a_piv).length(), l_ab, 0.0001, "4-bar crank length invariant < 1e-4")
        _v4_assert_approx((jc - jb).length(), l_bc, 0.0001, "4-bar coupler length invariant < 1e-4")
        _v4_assert_approx((jc - d_piv).length(), l_cd, 0.0001, "4-bar rocker length invariant < 1e-4")
    _ = driver_4b.evaluate(0.5)

    var regult = NativeModel()
    var pod = _v4_add_node(regult, "Pod", -1, Vec3(0, 1.8, 0))
    var dg_hip = _v4_add_node(regult, "Hip", pod, Vec3(0.4, -0.2, 0))
    var dg_knee = _v4_add_node(regult, "Knee1", dg_hip, Vec3(0, -0.55, 0.25))
    var dg_ank = _v4_add_node(regult, "AnkleRev", dg_knee, Vec3(0, -0.55, -0.45))
    var dg_foot = _v4_add_node(regult, "Foot", dg_ank, Vec3(0, -0.50, 0.25))

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

    _v4_assert_vec3_approx(dg_skel.global_position(dg_foot), dg_tgt, 0.003, "digitigrade 3-bone IK reaches ground target")
    assert_true(dg_skel.global_position(dg_knee).z > dg_skel.global_position(dg_hip).z)
    assert_true(dg_skel.global_position(dg_ank).z < dg_skel.global_position(dg_knee).z)

    var dg_driver = DigitigradeLegDriver(dg_hip, dg_knee, dg_ank, dg_foot, 0.08, 0.5)
    dg_driver.apply_to_skeleton(dg_skel, regult, Placement(), True, 0.0, Vec3(0, 1, 0), Float32(1.0 / 60.0))

    var chain: List[Vec3] = [Vec3(0, 0, 0), Vec3(0, 0.5, 0), Vec3(0, 1.0, 0), Vec3(0, 1.5, 0)]
    _ = fabrik(chain, Vec3(0.4, 1.1, 0.0))
    var cones: List[Float32] = [Float32(0.35), Float32(0.35), Float32(0.35)]
    var fab_rots = List[Quat]()
    _ = fabrik_constrained(chain, Vec3(0.4, 1.2, 0.2), cones, fab_rots)
    assert_equal(len(fab_rots), 4)

    _ = driver_4b.apply_to_skeleton(dg_skel, regult, Float32(0.5))
    var biped_fp = make_biped_foot_solver(pelvis, hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r)
    solve_two_bone_mecha_limb(skel, model, hip_l, knee_l, ankle_l, foot_goal, pole_fwd)
    _ = len(biped_fp.legs)

    var look = LookAt()
    look.apply_to_skeleton(skel, model, hip_l, pelvis, Vec3(1.0, 1.0, 2.0), True, 60.0, 45.0, 0.3, 0.05, Float32(1.0 / 60.0))
    var recoil = Recoil()
    recoil.kick(Vec3(0.0, 0.0, -1.0), 4.0)
    recoil.apply_to_skeleton(skel, model, pelvis, hip_l, 120.0, 14.0, Float32(1.0 / 60.0))
    assert_true(recoil.offset.length() > 0.0)

    skel.reset_to_rest(model)
    var fp = FootPlantSolver(pelvis)
    fp.add_leg(hip_l, knee_l, ankle_l, 0.10, Vec3(0, 0.5, 1.0))
    fp.add_leg(hip_r, knee_r, ankle_r, 0.10, Vec3(0, 0.5, 1.0))

    var slope_n = Vec3(0.0, det_cos(Float32(0.30)), -det_sin(Float32(0.30))).normalized()
    var stance_flags: List[Bool] = [True, True]
    var heights: List[Float32] = [Float32(0.0), Float32(-0.12)]
    var normals: List[Vec3] = [slope_n, Vec3(0, 1, 0)]

    var place0 = Placement(Vec3(0, 0, 0), Vec3(1, 1, 1), 0.0)
    fp.apply_to_skeleton(skel, model, place0, stance_flags, heights, normals, Float32(1.0 / 60.0))
    var locked_l_world = fp.foot_world_position(0, skel, place0)

    for step_k in range(12):
        var place_k = Placement(Vec3(0.0, 0.0, Float32(step_k + 1) * 0.015), Vec3(1, 1, 1), 0.0)
        skel.reset_to_rest(model)
        fp.apply_to_skeleton(skel, model, place_k, stance_flags, heights, normals, Float32(1.0 / 60.0))
        var cur_l_world = fp.foot_world_position(0, skel, place_k)
        _v4_assert_approx(cur_l_world.x, locked_l_world.x, 0.002, "stance foot X strictly locked (zero skate)")
        _v4_assert_approx(cur_l_world.z, locked_l_world.z, 0.002, "stance foot Z strictly locked (zero skate)")

    assert_true(fp.pelvis_offset_y.value < -0.05)
    var ankle_up = skel.global_rotation(ankle_l).rotate(Vec3(0, 1, 0))
    assert_true(ankle_up.dot(slope_n) > 0.97)


def test_n_v4_anim_6_motion_matching_zscore_simd8_aabb() raises:
    var db = MotionDatabase()
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
    assert_true(db.is_finalized and db.chunk_count() == 5)

    var total_pruned = 0
    for q_idx in range(16):
        var q = db.frames[q_idx * 15 + 3].feature
        q.traj_pos_400 = q.traj_pos_400 + Vec3(0.05, 0.0, -0.04)
        q.hip_vel = q.hip_vel + Vec3(0.1, 0.0, 0.1)

        var fast_res = db.find_best_match(q)
        total_pruned += db.count_pruned_chunks(q)

        var q_norm = Array[Float32, 33](fill=0.0)
        db._normalize_query(q, q_norm)
        var brute_idx = 0
        var brute_cost = Float32(1.0e30)
        for c in range(db.chunk_count()):
            db._eval_chunk_simd8(c, q_norm, q.stance_tag, -1, 0.0, -1, 0.0, 0.0, brute_idx, brute_cost)

        assert_equal(fast_res[0], db.frames[brute_idx].clip_index)
        _v4_assert_approx(fast_res[1], db.frames[brute_idx].time_seconds, 1.0e-5, "AABB search time matches brute-force")
        _v4_assert_approx(fast_res[2], brute_cost, 1.0e-4, "AABB search cost matches brute-force")

    assert_true(total_pruned > 10)

    var cur_frame = 40
    var q_tie = db.frames[cur_frame].feature
    var chosen_cont = db.query_inertial(
        q_tie,
        db.frames[cur_frame].clip_index,
        db.frames[cur_frame].time_seconds,
        continuing_bias=0.20,
        exclude_window=0.15,
    )
    assert_equal(chosen_cont, cur_frame)

    var m_bake = NativeModel()
    var r_b = _v4_add_node(m_bake, "Pelvis", -1, Vec3(0, 1, 0))
    var lf_b = _v4_add_node(m_bake, "Foot.L", r_b, Vec3(0.15, -1.0, 0))
    var rf_b = _v4_add_node(m_bake, "Foot.R", r_b, Vec3(-0.15, -1.0, 0))
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
    assert_true(db_baked.count() >= 10)
    assert_true(db_baked.frames[0].feature.hip_vel.z > 2.5)


def test_o_v4_anim_8_articulated_ragdoll_pd_zerog_limits_and_rerooting() raises:
    var inertia_pair = compute_capsule_inertia(10.0, 0.10, 0.50)
    assert_true(inertia_pair[1] > inertia_pair[0] and inertia_pair[0] > 0.0)

    var model = _v4_build_battloid_rig()
    var rag = _v4_configure_battloid_ragdoll(model)

    var dt_h = Float32(1.0 / 60.0)
    var skel_prev = SkeletonInstance()
    skel_prev.reset_to_rest(model)
    var skel_curr = SkeletonInstance()
    skel_curr.reset_to_rest(model)

    var v_root = Vec3(4.0, 0.0, 6.0)
    var omega_y = Float32(3.0)
    skel_curr.pose.r[0] = angle_axis(omega_y * dt_h, Vec3(0, 1, 0)).multiply(skel_prev.pose.r[0]).normalized()
    skel_curr.update_matrices(model)

    var pl_prev = Placement(Vec3(0, 0, 0))
    var pl_curr = Placement(v_root * dt_h)
    rag.handoff_from_poses(skel_prev, pl_prev, skel_curr, pl_curr, model, dt_h)

    _v4_assert_vec3_approx(rag.states[0].linear_vel, v_root, 0.02, "6-DOF handoff captures exact linear velocity v_i")
    _v4_assert_vec3_approx(rag.states[0].angular_vel, Vec3(0, omega_y, 0), 0.02, "6-DOF handoff captures exact angular velocity omega_i")

    var skel_single = SkeletonInstance()
    skel_single.reset_to_rest(model)
    skel_single.snapshot_prev_matrices()
    skel_single.pose.r[0] = angle_axis(omega_y * dt_h, Vec3(0, 1, 0))
    skel_single.update_matrices(model)
    rag.handoff_from_skeleton(skel_single, pl_curr, model, dt_h, pl_prev)
    _v4_assert_vec3_approx(rag.states[0].angular_vel, Vec3(0, omega_y, 0), 0.02, "snapshot_prev_matrices handoff captures omega_i")

    skel_curr.reset_to_rest(model)
    rag.handoff_from_poses(skel_curr, pl_prev, skel_curr, pl_prev, model, dt_h)
    rag.set_active_pd(1.0)
    rag.add_obb_collider(RagdollObbCollider(Vec3(0.0, 0.5, -0.85), Quat(), Vec3(1.0, 0.5, 0.25)))

    rag.apply_impulse_at_bone(1, Vec3(0.0, 0.0, -95.0))
    rag.apply_impulse_at_bone(7, Vec3(0.0, 0.0, 80.0), Vec3(0.18, 0.15, 0.0))

    rag.step(dt_h, 4)
    var chest_stagger_vel = rag.states[1].linear_vel.z
    assert_true(chest_stagger_vel < -1.0)

    var max_stagger_pitch = Float32(0.0)
    for _ in range(45):
        rag.step(dt_h, 4)
        var knee_angle = rag.joint_hinge_angle(5)
        assert_true(knee_angle >= -0.001 and knee_angle <= 2.36)
        var chest_rel = det_quat_to_scaled_angle(
            rag.states[0].rotation.conjugate().multiply(rag.states[1].rotation)
        ).length()
        max_stagger_pitch = max(max_stagger_pitch, chest_rel)

    assert_true(max_stagger_pitch > 0.03)
    var recovered_chest_err = det_quat_to_scaled_angle(
        rag.states[0].rotation.conjugate().multiply(rag.states[1].rotation)
    ).length()
    assert_true(recovered_chest_err < max_stagger_pitch * 0.45)
    assert_true(rag.pelvis_world_position().y > 0.55)
    rag.clear_obb_colliders()

    var rag_zg = _v4_configure_battloid_ragdoll(model)
    skel_curr.reset_to_rest(model)
    rag_zg.handoff_from_poses(skel_curr, pl_prev, skel_curr, pl_prev, model, dt_h)
    rag_zg.set_zero_g(True)
    rag_zg.linear_damping = 0.0
    rag_zg.angular_damping = 0.2
    rag_zg.root_attitude_stiffness = 0.0
    rag_zg.set_active_pd(1.0)

    var j_orbital = Vec3(-30.0, 12.0, -45.0)
    rag_zg.apply_impulse_at_bone(4, j_orbital)
    var p_after_hit = rag_zg.total_linear_momentum()
    _v4_assert_vec3_approx(p_after_hit, j_orbital, 0.05, "Zero-G impulse imparts exact total linear momentum J")

    for _ in range(30):
        rag_zg.step(dt_h, 4)
        var elbow_ang = rag_zg.joint_hinge_angle(3)
        assert_true(elbow_ang >= -0.001 and elbow_ang <= 2.41)

    var p_drifted = rag_zg.total_linear_momentum()
    _v4_assert_vec3_approx(p_drifted, j_orbital, 0.25, "Zero-G internal PD joint torques conserve linear momentum")
    assert_true(rag_zg.total_center_of_mass().y > 1.0)

    var rag_limp = _v4_configure_battloid_ragdoll(model)
    skel_curr.reset_to_rest(model)
    var start_place = Placement(Vec3(2.0, 0.0, -1.0), Vec3(1, 1, 1), 120.0)
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
    _v4_assert_approx(entity_yaw, settled_yaw_w, 0.0005, "begin_recovery updates entity root_yaw to settled pelvis yaw")
    _v4_assert_approx(entity_pos.y, 0.0, 0.0005, "begin_recovery places 1G entity root_pos.y on floor plane")

    var seeded_pose = Pose(target_stand.count())
    for idx in range(target_stand.count()):
        seeded_pose.t[idx] = rec_inert.out_t[idx]
        seeded_pose.r[idx] = rec_inert.out_r[idx]
        seeded_pose.s[idx] = rec_inert.out_s[idx]
    var pelvis_m_t0 = pose_position(seeded_pose, rag_limp.skel_parents, 0)
    var pelvis_w_t0 = entity_pos + angle_axis(entity_yaw, Vec3(0, 1, 0)).rotate(pelvis_m_t0)
    _v4_assert_vec3_approx(pelvis_w_t0, settled_pelvis_w, 0.0005, "begin_recovery achieves 0.0mm (<0.5mm) world pelvis continuity")

    for l_idx in range(rag_limp.limb_count()):
        var b_node = rag_limp.limbs[l_idx].bone_node
        var q_m = _pose_global_rot(seeded_pose, rag_limp.skel_parents, b_node)
        var q_w_t0 = angle_axis(entity_yaw, Vec3(0, 1, 0)).multiply(q_m).normalized()
        var rot_diff = det_quat_to_scaled_angle(q_w_t0.multiply(rag_limp.states[l_idx].rotation.conjugate())).length()
        _v4_assert_approx(rot_diff, 0.0, 0.002, "begin_recovery preserves exact world limb orientation at handoff")

    for _ in range(60):
        var live_p = Pose()
        live_p.assign(target_stand)
        rec_inert.apply(live_p, target_stand, False, 0.10, dt_h)
    assert_true(rec_inert.largest_offset() < 0.01)

    var bear = NativeModel()
    var b_root = _v4_add_node(bear, "Root", -1, Vec3(0, 0, 0))
    var j_names = ragdoll_joints()
    var j_parents = ragdoll_parents()
    var node_map = List[Int](length=len(j_names), fill=-1)
    for j in range(len(j_names)):
        var p_node = b_root if j_parents[j] < 0 else node_map[j_parents[j]]
        var off = Vec3(0, 0.25, 0)
        if j == 0:
            off = Vec3(0, 0.9, 0)
        elif j == 12:
            off = Vec3(0.25, -0.1, 0)
        elif j == 15:
            off = Vec3(-0.25, -0.1, 0)
        node_map[j] = _v4_add_node(bear, j_names[j], p_node, off)

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
    var yaw_120 = PI * Float32(2.0 / 3.0)
    var q_120 = angle_axis(yaw_120, Vec3(0, 1, 0))
    var shift_xz = Vec3(1.5, -0.55, -0.8)
    for j in range(len(j_names)):
        var p_idx = legacy_rag.particle[j]
        if p_idx >= 0:
            var rotated_p = q_120.rotate(legacy_rag.mesh.position[p_idx]) + shift_xz
            legacy_rag.mesh.place(p_idx, rotated_p)

    _v4_assert_approx(legacy_rag.settled_pelvis_yaw(), yaw_120, 0.01, "legacy Ragdoll.settled_pelvis_yaw measures +120 deg")
    _v4_assert_approx(legacy_rag.settled_root_offset(bear_place).x, 1.5, 0.01, "legacy Ragdoll.settled_root_offset X == 1.5m")

    legacy_rag.start_getup(bear)
    legacy_rag.play_getup(bear, bear_place, Float32(0.26))
    var root_fwd_after = bear.global_rotation(b_root).rotate(Vec3(0, 0, 1))
    var expected_fwd_120 = q_120.rotate(Vec3(0, 0, 1))
    _v4_assert_vec3_approx(root_fwd_after, expected_fwd_120, 0.02, "Ragdoll.play_getup aligns clip root to +120 deg settled heading (zero snap)")


def test_p_v4_phys_1_solver_delta_quat_nyquist_coupled_tangent_and_gyro() raises:
    var q_large = Quat.from_axis_angle(Vec3(0.3, 0.8, -0.5).normalized(), Float32(2.4))
    var dq_xyz = V8.splat(Vec3(q_large.x, q_large.y, q_large.z))
    var dqw = F8(q_large.w)
    var r0 = V8.splat(Vec3(3.5, -1.2, 2.1))
    var dr = quat_delta_displacement(dq_xyz, dqw, r0)
    var r_rot = Vec3(r0.x[0] + dr.x[0], r0.y[0] + dr.y[0], r0.z[0] + dr.z[0])
    var r_ref = q_large.rotate(Vec3(3.5, -1.2, 2.1))
    assert_true((r_rot - r_ref).length() < Float32(1.0e-5))
    assert_true(abs(r_rot.length() - Vec3(3.5, -1.2, 2.1).length()) < Float32(1.0e-5))

    var disp_pade = rodrigues_displacement(V8.splat(Vec3(0.4, -0.7, 1.1)), r0)
    var r_pade = Vec3(r0.x[0] + disp_pade.x[0], r0.y[0] + disp_pade.y[0], r0.z[0] + disp_pade.z[0])
    assert_true(abs(r_pade.length() - Vec3(3.5, -1.2, 2.1).length()) < Float32(1.0e-5))

    var settings = SolverSettings()
    settings.contact_hertz = 50.0
    var h = Float32(1.0 / 240.0)
    var soft_pair = solver_softness(settings, h)
    settings.contact_hertz = 1000.0
    var soft_capped = solver_softness(settings, h)
    assert_true(abs(soft_pair[1].bias_rate - soft_capped[1].bias_rate) < Float32(1.0e-5))

    var sb = SolverBodies()
    sb.reset(1)
    sb.inv_mass[0] = 1.0
    sb.set_body_inertia(0, Vec3(0.2, 1.5, 3.0))
    var tm = tangent_effective_mass_2x2(
        Float32(1.0), Float32(0.0), sb.inertia(0), sb.inertia(sb.static_index),
        Vec3(0.4, -0.5, 0.3), Vec3(), Vec3(1, 0, 0), Vec3(0, 0, 1),
    )
    assert_true(tm[0] > Float32(0.0) and tm[1] > Float32(0.0))
    assert_true(abs(tm[2]) > Float32(1.0e-3))

    assert_equal(SOLVER_WORD_STRIDE, 8)
    var run = SolverRun()
    run.stages.append(SolverStage(STAGE_SOLVE, 0, 1, 1, False))
    run.words = List[Int64](length=(1 + 2 * len(run.stages)) * SOLVER_WORD_STRIDE, fill=0)
    var addr0 = Int(run.word(0))
    var addr1 = Int(run.word(1))
    var addr2 = Int(run.word(2))
    assert_equal(addr1 - addr0, 64)
    assert_equal(addr2 - addr1, 64)

    var gyro_bodies = SolverBodies()
    var batching = SolverBodies()
    for pair in [(0, 0), (1, 1), (5, 1), (8, 1), (9, 2), (17, 3)]:
        batching.reset(pair[0])
        assert_equal(batching.batch_count(), pair[1])
    gyro_bodies.reset(1)
    gyro_bodies.inv_mass[0] = 1.0
    gyro_bodies.set_body_inertia(0, Vec3(1.0, 2.5, 4.0))
    gyro_bodies.wx[0] = 0.08
    gyro_bodies.wy[0] = 12.0
    gyro_bodies.wz[0] = 0.05
    var e0 = Float32(0.5) * (Float32(1.0) * 0.08 * 0.08 + Float32(2.5) * 12.0 * 12.0 + Float32(4.0) * 0.05 * 0.05)
    var flipped_wy = False
    for _ in range(360):
        gyro_bodies.integrate_velocities(0, Float32(1.0 / 240.0), Float32(400.0), Float32(1.57))
        gyro_bodies.integrate_positions(0, Float32(1.0 / 240.0))
        var wb = gyro_bodies.orientation(0).conjugate().rotate(gyro_bodies.angular(0))
        if wb.y < Float32(-5.0):
            flipped_wy = True
    var wb_end = gyro_bodies.orientation(0).conjugate().rotate(gyro_bodies.angular(0))
    var e1 = Float32(0.5) * (Float32(1.0) * wb_end.x * wb_end.x + Float32(2.5) * wb_end.y * wb_end.y + Float32(4.0) * wb_end.z * wb_end.z)
    assert_true(flipped_wy)
    assert_true(e1 <= e0 * Float32(1.01))


def test_q_v4_phys_2_high_mass_ratio_1000_to_1_stack_and_fracture_world() raises:
    var world = PhysicsWorld()
    world.enable_sleeping = False
    _ = world.add_static_mesh(_v4_make_flat_bvh_quad(50.0, 0.0))
    var b_light = world.add_box(Vec3(0.0, 0.5, 0.0), Vec3(0.5, 0.5, 0.5), mass=1.0)
    var b_heavy = world.add_box(Vec3(0.0, 1.5, 0.0), Vec3(0.5, 0.5, 0.5), mass=1000.0)

    for _ in range(120):
        world.step(Float32(1.0 / 60.0))

    var y_light = world.bodies[b_light].pos.y
    var y_heavy = world.bodies[b_heavy].pos.y
    var sag_light = Float32(0.5) - y_light
    var sag_heavy = Float32(1.5) - y_heavy
    assert_true(sag_light < Float32(0.005) and sag_light > Float32(-0.002))
    assert_true(sag_heavy < Float32(0.005) and sag_heavy > Float32(-0.002))
    assert_equal(world.bodies[b_light].inv_mass, Float32(1.0))

    var w2 = PhysicsWorld()
    _ = w2.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    w2.add_sensor(Sensor(77, Vec3(0.0, 1.0, 0.0), Vec3(2.0, 2.0, 2.0), 0.0, sensor_filter()))
    var s_slot = w2.add_sphere(Vec3(-2.0, 0.8, 0.0), 0.4, mass=2.0)
    var c_slot = w2.add_capsule(Vec3(0.0, 1.2, 0.0), 0.3, 0.4, mass=3.0)
    var hull_verts = List[Vec3]()
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                hull_verts.append(Vec3(Float32(sx) * 0.4, Float32(sy) * 0.4, Float32(sz) * 0.4))
    var h_slot = w2.add_convex_hull(Vec3(2.0, 0.6, 0.0), ConvexHull(hull_verts), mass=4.0)

    var saw_sensor = False
    var saw_begin = False
    for _ in range(60):
        w2.step(Float32(1.0 / 60.0))
        for ev in w2.events:
            if ev.kind == EVENT_SENSOR_BEGIN and ev.a == 77:
                saw_sensor = True
            if ev.kind == EVENT_BEGIN:
                saw_begin = True
    assert_true(saw_sensor)
    assert_true(saw_begin)
    assert_true(w2.bodies[s_slot].pos.y > Float32(0.35))
    assert_true(w2.bodies[c_slot].pos.y > Float32(0.65))
    assert_true(w2.bodies[h_slot].pos.y > Float32(0.35))

    var wf = PhysicsWorld(gravity=0.0)
    var piece_a = wf.add_box(Vec3(-0.6, 2.0, 0.0), Vec3(0.4, 0.4, 0.4), mass=5.0)
    var piece_b = wf.add_box(Vec3(0.6, 2.0, 0.0), Vec3(0.4, 0.4, 0.4), mass=5.0)
    var weld_idx = wf.add_weld_joint(piece_a, piece_b, Vec3(0.0, 2.0, 0.0), break_force=150.0)
    wf.step(Float32(1.0 / 60.0))
    assert_true(not wf.joints[weld_idx].broken)
    assert_equal(wf.islands.label[piece_a], wf.islands.label[piece_b])

    wf.apply_impulse(piece_b, Vec3(120.0, 0.0, 0.0))
    wf.apply_impulse(piece_a, Vec3(-120.0, 0.0, 0.0))
    var saw_joint_break = False
    for _ in range(5):
        wf.step(Float32(1.0 / 60.0))
        for ev in wf.events:
            if ev.kind == EVENT_JOINT_BREAK:
                saw_joint_break = True
    assert_true(wf.joints[weld_idx].broken)
    assert_true(saw_joint_break)
    assert_true(wf.islands.label[piece_a] != wf.islands.label[piece_b])


def test_r_v4_phys_3_vehicle_and_tires() raises:
    var tire = PacejkaTire(PacejkaParams())
    var f_98 = tire.calc_forces(Float32(0.14 * 0.98), Float32(0.06), Float32(4000.0), Float32(25.0))
    var f_99 = tire.calc_forces(Float32(0.14 * 0.99), Float32(0.06), Float32(4000.0), Float32(25.0))
    var f_100 = tire.calc_forces(Float32(0.14 * 1.00), Float32(0.06), Float32(4000.0), Float32(25.0))
    var f_101 = tire.calc_forces(Float32(0.14 * 1.01), Float32(0.06), Float32(4000.0), Float32(25.0))
    var slope_left = f_99[0] - f_98[0]
    var slope_mid = f_100[0] - f_99[0]
    var slope_right = f_101[0] - f_100[0]
    assert_true(abs(slope_mid - slope_left) < Float32(5.0))
    assert_true(abs(slope_right - slope_mid) < Float32(5.0))

    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.52, 0.0)
    car.linear_vel = Vec3(0.0, 0.0, 18.0)
    for _ in range(60):
        car.substep_suspension(Float32(1.0 / 60.0), throttle=0.75, brake=0.0, steer=0.35)

    assert_true(car.wheel_states[1].steer_angle > car.wheel_states[0].steer_angle)
    assert_true(car.pos.x > Float32(1.0))
    assert_true(car.engine_rpm > Float32(1500.0))

    var car_arb = VehicleChassis(1450.0)
    car_arb.setup_four_wheel_car()
    car_arb.pos = Vec3(0.0, 1.00, 0.0)
    car_arb.rot = Quat.from_axis_angle(Vec3(0, 0, 1), Float32(0.04))
    car_arb.substep_suspension(Float32(1.0 / 240.0), 0.0, 0.0, 0.0)
    assert_true(car_arb.wheel_states[0].compression > car_arb.wheel_states[1].compression)
    assert_true(car_arb.wheel_states[0].normal_load > car_arb.wheel_states[1].normal_load + Float32(500.0))

    var slope = Float32(0.1745329)
    var ramp_bvh = _v4_make_ramp_bvh(slope)
    var hill_car = VehicleChassis(1450.0)
    hill_car.setup_four_wheel_car()
    hill_car.rot = Quat.from_axis_angle(Vec3(-1, 0, 0), slope)
    var n_ramp = Vec3(0.0, det_cos(slope), -det_sin(slope))
    hill_car.pos = n_ramp * Float32(1.00)
    hill_car.set_controls(0.0, 1.0, 0.0)

    for _ in range(30):
        hill_car.substep_suspension_bvh(Float32(1.0 / 60.0), ramp_bvh)
    var z_settled = hill_car.pos.z
    for _ in range(90):
        hill_car.substep_suspension_bvh(Float32(1.0 / 60.0), ramp_bvh)
    var hill_drift = abs(hill_car.pos.z - z_settled)
    assert_true(hill_car.wheel_states[0].grounded and hill_car.wheel_states[2].grounded)
    assert_true(abs(hill_car.wheel_states[0].bristle_x) > Float32(1.0e-4))
    assert_true(hill_drift < Float32(0.03))


def test_s_v4_phys_4_xpbd_and_beaufort_water() raises:
    var chain = ParticleChain.line(Vec3(0.0, 2.0, 0.0), Vec3(2.0, 2.0, 0.0), 9, mass=0.2)
    chain.compliance = Float32(1.0e-5)
    chain.pin(0, Vec3(0.0, 2.0, 0.0))
    var colliders = List[ChainCollider]()
    colliders.append(ChainCollider.capsule(Vec3(-1.0, 1.2, 0.0), Vec3(3.0, 1.2, 0.0), 0.35))
    for _ in range(30):
        chain.step(Float32(1.0 / 60.0), 4, 9.81, List[Vec3](), colliders, iterations=2)
    assert_equal(len(chain.lambdas), 8)
    for i in range(1, chain.count()):
        var cp = colliders[0].closest_axis_point(chain.position[i])
        assert_true((chain.position[i] - cp).length() >= Float32(0.349))

    var pmesh = ParticleMesh()
    var p0 = pmesh.add_particle(Vec3(0.0, 2.0, 0.0), mass=0.0, radius=0.05)
    var p1 = pmesh.add_particle(Vec3(0.5, 2.0, 0.0), mass=0.4, radius=0.05)
    pmesh.add_link(p0, p1, compliance=0.0)
    pmesh.step(Float32(1.0 / 60.0), 4, 9.81, List[Vec3](), colliders, iterations=2)
    assert_true(pmesh.link_error() < Float32(0.01))

    var sb = SolverBodies()
    sb.reset(1)
    sb.px[0] = 0.0; sb.py[0] = 5.0; sb.pz[0] = 0.0
    sb.inv_mass[0] = 0.2
    sb.set_body_inertia(0, Vec3(1.0, 1.0, 1.0))

    var tether = ParticleChain.line(Vec3(1.0, 5.0, 0.0), Vec3(1.0, 3.0, 0.0), 5, mass=1.0)
    var atts = List[RigidAttachment]()
    atts.append(RigidAttachment(0, 0, Vec3(1.0, 0.0, 0.0), compliance=0.0))
    tether.step_coupled(Float32(1.0 / 60.0), 4, 9.81, List[Vec3](), List[ChainCollider](), atts, sb, iterations=2)
    assert_true(sb.vy[0] < Float32(-0.01))
    assert_true(sb.wz[0] < Float32(-0.01))

    var w0 = eval_beaufort_wave(12.5, -7.3, 3.2, water_y=0.0)
    var w1 = eval_beaufort_wave(12.5, -7.3, 3.2, water_y=0.0)
    assert_equal(w0[0], w1[0])
    assert_true(abs(w0[1].length() - Float32(1.0)) < Float32(1.0e-5))
    assert_true(w0[2].length() > Float32(0.01))

    var r_sph = Float32(0.5)
    var sp_buoy = eval_sphere_buoyancy_drag(
        Vec3(0, 0, 0), r_sph, Vec3(), Vec3(), 0.0, water_y=0.0,
        fluid_density=RHO_SEAWATER_SI, gravity=9.81, use_waves=False,
    )
    var expected_half_vol = (Float32(2.0) / Float32(3.0)) * PI * r_sph * r_sph * r_sph
    assert_true(abs(sp_buoy.submerged_volume - expected_half_vol) < Float32(1.0e-4))
    assert_true(abs(sp_buoy.force.y - RHO_SEAWATER_SI * expected_half_vol * Float32(9.81)) < Float32(0.5))

    var tilt_q = Quat.from_axis_angle(Vec3(0, 0, 1), Float32(0.2))
    var box_buoy = eval_box_buoyancy_drag(
        Vec3(0, 0.0, 0), tilt_q, Vec3(1.5, 0.4, 2.0), Vec3(), Vec3(), 0.0,
        water_y=0.0, use_waves=False,
    )
    assert_true(box_buoy.torque.z < Float32(-100.0))

    var ww = PhysicsWorld()
    ww.set_water(0.0, use_waves=True)
    var crate = ww.add_box(Vec3(0.0, 0.6, 0.0), Vec3(0.5, 0.5, 0.5), mass=400.0)
    var saw_splash = False
    for _ in range(90):
        ww.step(Float32(1.0 / 60.0))
        for ev in ww.events:
            if ev.kind == EVENT_WATER_ENTER:
                saw_splash = True
    assert_true(saw_splash)
    assert_true(ww.bodies[crate].pos.y > Float32(-0.3) and ww.bodies[crate].pos.y < Float32(0.5))


def test_t_v4_phys_5_aero6dof_mecha_cobra_gerwalk_and_zem_apn_simd8() raises:
    var isa_0 = eval_isa_atmosphere(0.0)
    var isa_10k = eval_isa_atmosphere(10000.0)
    var isa_100k = eval_isa_atmosphere(100000.0)
    assert_true(abs(isa_0[0] - Float32(1.225)) < Float32(1.0e-3))
    assert_true(isa_10k[0] < Float32(0.42) and isa_10k[0] > Float32(0.30))
    assert_true(isa_100k[0] < Float32(1.0e-4))

    var ge_low = eval_ground_effect(1.2, 14.78)
    var ge_high = eval_ground_effect(100.0, 14.78)
    assert_true(ge_low[0] > Float32(1.2) and ge_low[1] < Float32(0.2))
    assert_true(abs(ge_high[0] - Float32(1.0)) < Float32(0.01) and ge_high[1] > Float32(0.99))

    var tr_sub = eval_transonic_wave_drag(0.60)
    var tr_sonic = eval_transonic_wave_drag(1.05)
    var tr_sup = eval_transonic_wave_drag(2.20)
    assert_equal(tr_sub[1], Float32(0.0))
    assert_true(tr_sonic[1] > Float32(0.04))
    assert_true(tr_sup[1] < tr_sonic[1] and tr_sup[1] > Float32(0.01))

    var p_90 = eval_post_stall_polar(Float32(PI * 0.5), 5.2, 0.08, 0.02, 0.30, 7.6, 1.0, 1.0, 1.0, 0.0)
    assert_true(abs(p_90[0]) < Float32(0.05))
    assert_true(p_90[1] > Float32(1.8))

    var cfg = Aero6DofConfig()
    var cobra = Aero6DofState(Vec3(0.0, 1200.0, 0.0), Vec3(0.0, 0.0, 140.0))
    var peak_alpha = Float32(0.0)
    for _ in range(90):
        var ctrl = Aero6DofControls()
        ctrl.throttle = 0.15
        var err_a = Float32(1.66) - cobra.alpha_rad
        var q_rate = cobra.rot.conjugate().rotate(cobra.angular_vel).x
        var cmd_p = max(Float32(-1.0), min(Float32(1.0), err_a * Float32(2.5) + q_rate * Float32(0.75)))
        ctrl.pitch_cmd = cmd_p
        ctrl.tvc_pitch = cmd_p * cfg.max_tvc_angle_rad
        ctrl.rcs_cmd = Vec3(-cmd_p, 0.0, 0.0)
        step_aero6dof(cobra, cfg, ctrl, Float32(1.0 / 60.0))
        if cobra.alpha_rad > peak_alpha:
            peak_alpha = cobra.alpha_rad
    assert_true(not (isnan(cobra.pos.y) or isinf(cobra.linear_vel.z)))
    assert_true(peak_alpha > Float32(1.45))
    assert_true(cobra.linear_vel.length() < Float32(115.0))

    var gerwalk = Aero6DofState(Vec3(0.0, 50.0, 0.0), Vec3(0.0, 0.0, 0.0))
    gerwalk.morph_u = 1.0
    var g_ctrl = Aero6DofControls()
    g_ctrl.morph_u_target = 1.0
    g_ctrl.hover_left_throttle = 0.88
    g_ctrl.hover_right_throttle = 0.78
    g_ctrl.hover_tilt_rad = Float32(PI * 0.5)
    for _ in range(30):
        step_aero6dof(gerwalk, cfg, g_ctrl, Float32(1.0 / 60.0))
    assert_equal(gerwalk.current_inertia.x, cfg.inertia_1.x)
    assert_true(gerwalk.pos.y > Float32(50.5))
    assert_true(abs(gerwalk.angular_vel.z) > Float32(0.05))
    assert_true(gerwalk.angular_vel.x != Float32(0.0))

    var space = Aero6DofState(Vec3(0.0, 200000.0, 0.0), Vec3(0.0, 0.0, 7800.0))
    var s_ctrl = Aero6DofControls()
    s_ctrl.gravity = 0.0
    s_ctrl.rcs_cmd = Vec3(0.0, 1.0, 0.0)
    for _ in range(30):
        step_aero6dof(space, cfg, s_ctrl, Float32(1.0 / 60.0))
    assert_true(space.dynamic_pressure < Float32(1.0))
    assert_true(space.angular_vel.y > Float32(0.3))

    var batch = MissileBatch8()
    batch.pos = V8(
        F8(-40, -28, -16, -5, 5, 16, 28, 40),
        F8(200, 210, 190, 205, 195, 210, 190, 200),
        F8(0.0),
    )
    batch.vel = V8(
        F8(-30, -20, -10, -4, 4, 10, 20, 30),
        F8(25, 15, -15, 20, -20, 15, -15, 25),
        F8(280.0),
    )
    var t_pos = Vec3(0.0, 260.0, 1400.0)
    var t_vel = Vec3(45.0, 8.0, -90.0)
    var t_acc = Vec3(-12.0, 18.0, 0.0)

    var dt_m = Float32(1.0 / 120.0)
    for _ in range(600):
        t_vel = t_vel + t_acc * dt_m
        t_pos = t_pos + t_vel * dt_m
        _ = step_missile_batch8(
            batch,
            V8.splat(t_pos),
            V8.splat(t_vel),
            V8.splat(t_acc),
            dt_m,
        )

    comptime for lane in range(LANES):
        assert_true(batch.min_range[lane] < Float32(4.0))


def test_t5_kcc_local_transform_sync_no_teleport() raises:
    var app = EngineApp(AppConfig(title="t5_kcc_lt_sync", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var cfg = CharacterControllerConfig()
    cfg.gravity = 0.0

    # 1. Spawn with LocalTransform=(10, 1, 0) & default CharacterControllerComponent() -> must NOT teleport to (0, 1, 0)
    var e1 = app.world.spawn()
    _ = app.world.insert[LocalTransform](e1, LocalTransform(Vec3(10.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e1, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](e1, PrevGlobalTransform())
    var kc1_init = CharacterControllerComponent()
    kc1_init.controller.cfg.gravity = 0.0
    _ = app.world.insert[CharacterControllerComponent](e1, kc1_init^)

    # 2. Spawn with LocalTransform=(0, 0, 0) & default CharacterControllerComponent() -> must stay at (0, 0, 0)
    var e2 = app.world.spawn()
    _ = app.world.insert[LocalTransform](e2, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e2, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](e2, PrevGlobalTransform())
    var kc2_init = CharacterControllerComponent()
    kc2_init.controller.cfg.gravity = 0.0
    _ = app.world.insert[CharacterControllerComponent](e2, kc2_init^)

    # 3. Spawn with default LocalTransform=(0, 0, 0) & explicit non-default controller.pos=(5, 2, 0) -> preserves (5, 2, 0)
    var e3 = app.world.spawn()
    _ = app.world.insert[LocalTransform](e3, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e3, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](e3, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        e3, CharacterControllerComponent(Vec3(5.0, 2.0, 0.0), cfg, Vec3())
    )

    app.step_one_frame(Float32(1.0 / 60.0))

    var lt1 = app.world.get[LocalTransform](e1)
    var kc1 = app.world.get_ptr[CharacterControllerComponent](e1).value()[unsafe_offset=0].controller.pos
    assert_true(abs(lt1.translation.x - Float32(10.0)) < Float32(1.0e-4), "KCC teleported away from LocalTransform x=10")
    assert_true(abs(lt1.translation.y - Float32(1.0)) < Float32(1.0e-2))
    assert_true(abs(kc1.x - Float32(10.0)) < Float32(1.0e-4))

    var lt2 = app.world.get[LocalTransform](e2)
    assert_true(abs(lt2.translation.y) < Float32(1.0e-2), "KCC at origin teleported to default y=1.0")

    var lt3 = app.world.get[LocalTransform](e3)
    assert_true(abs(lt3.translation.x - Float32(5.0)) < Float32(1.0e-4), "Explicit controller.pos=(5,2,0) lost")

    # 4. External LocalTransform teleport between frames propagates into KCC
    app.world.get_ptr_mut[LocalTransform](e1).value()[unsafe_offset=0].translation = Vec3(25.0, 2.0, 0.0)
    app.step_one_frame(Float32(1.0 / 60.0))
    var lt1_tp = app.world.get[LocalTransform](e1)
    assert_true(abs(lt1_tp.translation.x - Float32(25.0)) < Float32(1.0e-4), "External LocalTransform move failed to sync KCC")

    app.shutdown()


def main() raises:
    var t0 = monotonic()
    test_a_snapshot_determinism_queries_triggers_ccd()
    print("[PASS] (a) PhysicsWorld 200-body+joints+vehicle rollback snapshot, O(1) id_to_slot, kinematic, triggers, CCD & scene queries")
    test_b_kcc_stairs_slopes_acute_corner_turntable_and_push()
    print("[PASS] (b) KCC stair climb, 45 vs 65 deg slopes, 45 deg acute V-corner crease/lock, turntable & crate push")
    test_c_quickhull_mass_vhacd_heightfield_dda_and_6dof()
    print("[PASS] (c) Quickhull 3D + polyhedral mass/CoM/inertia + V-HACD-lite L-shape + Heightfield DDA + SixDofMotorJoint")
    test_d_recast_navmesh_obstacle_ssfa_and_256_orca()
    print("[PASS] (d) Recast NavMesh + dynamic obstacle carve + OffMeshLink + SSFA + 256-agent SIMD8 ORCA (0 overlaps)")
    test_e_ai_perception_camera_flanim_and_ragdoll_sm()
    print("[PASS] (e) BehaviorTree reactive abort + UtilityAI + SpatialPerception + CameraRig + .flanim + Socket + FootIK + RagdollSM")
    test_f_phase0_phase1_regressions()
    print("[PASS] (f) Phase 0+1 regressions: Compound multi-normal, KCC vs Compound, CCD bilateral momentum, Socket no double-step, Tile-indexed NavMesh, Camera shake wall clamp, BT_NODE_PARALLEL")
    test_g_wave2_unification_and_parallel_regressions()
    print("[PASS] (g) Wave 2 regressions: Unified SixDofMotorJoint determinism, Parallel narrowphase determinism, Water early-out, NavMesh binary round-trip, Multi-level ORCA Y-band & CrowdNavBridge")
    test_h_m2_two_way_physics_world_unification()
    print("[PASS] (h) M2a two-way PhysicsWorld unification: VehicleChassis, Aero6DofState + detmath SIMD8, XPBD Chain/Mesh, ArticulatedRagdoll + RagdollSM + PHYW v2 rollback")
    test_i_v4_anim_1_cubicspline_step_and_springs()
    print("[PASS] (i) v4 anim 1: CUBICSPLINE Hermite, STEP final keyframe, SpringFloat/Vec3/Angle & empty GLB guard")
    test_j_v4_anim_2_skeleton_instance_topo_and_prev_matrices()
    print("[PASS] (j) v4 anim 2: SkeletonInstance out-of-order topo_order, snapshot_prev_matrices & NativeModel immutability")
    test_k_v4_anim_3_bollo_inertializer_and_socket_reparent()
    print("[PASS] (k) v4 anim 3: Bollo scalar-axis Inertializer decay, is_settled & reparent_socket_inertial 0mm handoff")
    test_l_v4_anim_4_blend_space_2d_and_loop_root_motion()
    print("[PASS] (l) v4 anim 4: Cartesian/Polar BlendSpace2DDef, loop-wrap extract_root_motion & Animator 2D state")
    test_m_v4_anim_5_ik_pole_fourbar_digitigrade_and_footplant()
    print("[PASS] (m) v4 anim 5: solve_two_bone pole IK, FourBarLinkage, Digitigrade IK, FABRIK, LookAt, Recoil & FootPlantSolver")
    test_n_v4_anim_6_motion_matching_zscore_simd8_aabb()
    print("[PASS] (n) v4 anim 6: MotionDatabase z-score + SIMD8 AABB chunk pruning parity, query_inertial & bake_clip_features")
    test_o_v4_anim_8_articulated_ragdoll_pd_zerog_limits_and_rerooting()
    print("[PASS] (o) v4 anim 8: ArticulatedRagdoll 6-DOF handoff, 1G/Zero-G Active-PD, hinge limits, begin_recovery & legacy Ragdoll")
    test_p_v4_phys_1_solver_delta_quat_nyquist_coupled_tangent_and_gyro()
    print("[PASS] (p) v4 phys 1: Solver delta-quat, Nyquist softness, 2x2 tangent mass, 64B atomics & SolverBodies Dzhanibekov gyro")
    test_q_v4_phys_2_high_mass_ratio_1000_to_1_stack_and_fracture_world()
    print("[PASS] (q) v4 phys 2: 1000:1 stack (<5mm sag), Hull/Capsule/Sensor events & breakable weld fracture islands")
    test_r_v4_phys_3_vehicle_and_tires()
    print("[PASS] (r) v4 phys 3: PacejkaTire C1 ellipse, Ackermann steer, ARB sign & BVH8 ramp bristle hill-hold")
    test_s_v4_phys_4_xpbd_and_beaufort_water()
    print("[PASS] (s) v4 phys 4: XPBD capsule push-out, step_coupled SolverBodies & Beaufort Archimedes/metacentric water")
    test_t_v4_phys_5_aero6dof_mecha_cobra_gerwalk_and_zem_apn_simd8()
    print("[PASS] (t) v4 phys 5: Aero6DOF ISA/GE/transonic, Cobra, GERWALK hover, 200km LEO RCS & 8-lane MissileBatch8")
    test_t5_kcc_local_transform_sync_no_teleport()
    print("[PASS] (u) Tier 5 PhysicsPlugin KCC LocalTransform Registration & External Sync (No Teleport)")
    var ms = Float64(monotonic() - t0) / 1.0e6
    print("ALL FLINT V5 PHYSICS / KCC / NAVMESH / ORCA / AI / CAMERA / FLANIM_ECS TESTS PASSED in", ms, "ms!")
