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
from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from std.time import monotonic
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos, PI
from flint.core.serial import BlobWriter, BlobReader, fnv64, float32_bits
from flint.core.jobs import JobPool
from flint.core.ik import angle_axis
from flint.physics.filter import CollisionFilter, CAT_TERRAIN, CAT_STATIC, CAT_PROP_PIECE, CAT_ALL
from flint.physics.water import eval_box_buoyancy_drag
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.narrowphase import ConvexHull, ContactManifold
from flint.physics.shapes_v5 import (
    CompoundChild,
    CompoundShape,
    HeightfieldCollider,
    SixDofMotorJoint,
)
from flint.physics.vehicle import VehicleChassis
from flint.physics.xpbd import ParticleChain, ParticleMesh, ChainCollider
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
from flint.assets.model import NativeModel, NativeNode, NativeAnimation, NativeChannel
from flint.anim.pose import Pose, SkeletonInstance
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
from flint.anim.procedural import Placement, FootPlantSolver
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
    var ms = Float64(monotonic() - t0) / 1.0e6
    print("ALL FLINT V5 PHYSICS / KCC / NAVMESH / ORCA / AI / CAMERA / FLANIM_ECS TESTS PASSED in", ms, "ms!")
