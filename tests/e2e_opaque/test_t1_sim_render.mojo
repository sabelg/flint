"""Opaque-box E2E Tier 1 (Happy Path) suite for FA3 (Simulation) and FA4 (Render).

Covers >= 6 happy-path tests for FA3 (`PhysicsWorld`, `CharacterController`,
`VehicleChassis`, `Aero6DofState`, `ParticleChain`/`ParticleMesh`/`SolverBodies`,
`ArticulatedRagdoll`, `flanim_ecs`) and >= 6 happy-path tests for FA4
(`ScalableRenderer`, `AaaRenderGraph`, `FlintAaaVulkanRenderer`, `flip`, `image_diff`).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs
from std.testing import assert_true, assert_false, assert_equal

from flint.core.math import Vec3, Quat, Mat4
from flint.core.transform import trs
from flint.core.ik import angle_axis
from flint.core.serial import BlobWriter, BlobReader, fnv64, float32_bits

from flint.physics.world import (
    PhysicsWorld,
    TriggerVolume,
    TRIGGER_ENTER,
    TRIGGER_STAY,
    TRIGGER_EXIT,
)
from flint.physics.joints import JointDef, JOINT_DISTANCE
from flint.physics.filter import CollisionFilter, CAT_ALL
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.shapes_v5 import HeightfieldCollider
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.vehicle import VehicleChassis
from flint.physics.aero6dof import (
    Aero6DofConfig,
    Aero6DofControls,
    Aero6DofState,
    step_aero6dof,
    eval_isa_atmosphere,
)
from flint.physics.xpbd import ParticleChain, ParticleMesh, ChainCollider, RigidAttachment
from flint.physics.solver_bodies import SolverBodies

from flint.assets.model import NativeModel, NativeNode, NativeAnimation, NativeChannel
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.image import write_png
from flint.platform.os import make_dirs, remove_file

from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import Placement, FootPlantSolver
from flint.anim.ragdoll import ArticulatedRagdoll, SETTLE_SPEED
from flint.anim.graph import AnimGraphDef, BlendSpaceDef, BlendSampleDef, StateDef, Condition, OP_GT
from flint.anim.flanim_ecs import (
    RagdollStateMachine,
    BoneSocketAttachment,
    serialize_flanim,
    deserialize_flanim,
    save_flanim_bytes,
    load_flanim_bytes,
    solve_foot_plant_world,
    FLANIM_FOURCC,
    RAGDOLL_SM_ANIMATED,
    RAGDOLL_SM_ACTIVE_PD,
    RAGDOLL_SM_GETUP_SUPINE,
    RAGDOLL_SM_GETUP_PRONE,
)

from flint.render.render_world import (
    ScalableRenderer,
    RenderWorld,
    DoubleBufferedRenderWorld,
    RenderExtractor,
    RenderCameraPacket,
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
)
from flint.render.material_graph import MaterialAsset
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    RgResourceAccess,
    RG_QUEUE_GRAPHICS,
    RG_QUEUE_ASYNC_COMPUTE,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RG_USAGE_COLOR_ATT_WRITE,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.flip import flip_error_map, flip_mean, flip_heat_image
from flint.render.image_diff import compare_pixels, compare_with_png
from flint.ui.draw_list import UiDrawList


def _make_procedural_biped() -> NativeModel:
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


# ============================================================================
# FA3 Tier 1: Simulation (Physics, Character, Vehicle, Aero, XPBD, Ragdoll, Anim)
# ============================================================================

def test_fa3_t1_physics_world_bodies_step_and_queries() raises:
    var w = PhysicsWorld(gravity=9.81)
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var sphere_slot = w.add_sphere(Vec3(0.0, 2.5, 0.0), 0.5, mass=1.0)
    var sphere_id = w.body_id(sphere_slot)
    _ = w.add_capsule(Vec3(2.0, 2.5, 0.0), 0.3, 0.4, mass=2.0)

    var bvh_v: List[Vec3] = [Vec3(-1.5, 0.0, 6.0), Vec3(1.5, 0.0, 6.0), Vec3(0.0, 2.5, 6.0)]
    var bvh_i: List[Int32] = [Int32(0), Int32(1), Int32(2)]
    _ = w.add_static_mesh(Bvh8MeshCollider.build(bvh_v, bvh_i))

    var hf_heights: List[Float32] = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    _ = w.add_heightfield(HeightfieldCollider.build(hf_heights, 2, 2, Vec3(-2.0, 0.0, 10.0), 4.0))
    w.rebuild_broadphase()

    var dt = Float32(1.0 / 60.0)
    for _ in range(60):
        w.step(dt)

    assert_equal(w.tick, 60)
    assert_true(w.bodies[sphere_slot].pos.y > Float32(0.42) and w.bodies[sphere_slot].pos.y < Float32(0.85))

    var hit_down = w.raycast(Vec3(0.0, 5.0, 0.0), Vec3(0.0, -1.0, 0.0), 20.0)
    assert_true(hit_down.hit)
    assert_equal(hit_down.body_id, sphere_id)

    var hits_all = w.raycast_all(Vec3(0.0, 0.6, -2.0), Vec3(0.0, 0.0, 1.0), 15.0)
    assert_true(len(hits_all) >= 2)

    var sc_sphere = w.shapecast_sphere(Vec3(0.0, 0.6, -3.0), 0.3, Vec3(0.0, 0.0, 6.0))
    assert_true(sc_sphere.hit)

    var sc_cap = w.shapecast_capsule(Vec3(2.0, 0.7, -3.0), 0.25, 0.3, Quat(), Vec3(0.0, 0.0, 6.0))
    assert_true(sc_cap.hit)

    var sc_box = w.shapecast_box(Vec3(0.0, 5.0, 0.0), Vec3(0.3, 0.3, 0.3), Quat(), Vec3(0.0, -10.0, 0.0))
    assert_true(sc_box.hit)

    var ov_sph = w.overlap_sphere(Vec3(0.0, 0.5, 0.0), 1.2)
    assert_true(len(ov_sph) >= 1)
    var ov_box = w.overlap_aabb(Vec3(-1.0, -0.2, -1.0), Vec3(1.0, 1.5, 1.0))
    assert_true(len(ov_box) >= 1)


def test_fa3_t1_physics_joints_triggers_and_serialize() raises:
    var w = PhysicsWorld(gravity=9.81)
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var anchor_slot = w.add_box(Vec3(0.0, 4.0, 0.0), Vec3(0.25, 0.25, 0.25), mass=0.0)
    var bob_slot = w.add_box(Vec3(1.5, 4.0, 0.0), Vec3(0.25, 0.25, 0.25), mass=2.0)

    var jd = JointDef(JOINT_DISTANCE, anchor_slot, bob_slot, Vec3(0.0, 4.0, 0.0))
    jd.length = Float32(1.5)
    var j_idx = w.add_joint(jd)
    assert_true(j_idx >= 0)

    var dt = Float32(1.0 / 60.0)
    for _ in range(30):
        w.step(dt)
    var dist_from_anchor = (w.bodies[bob_slot].pos - Vec3(0.0, 4.0, 0.0)).length()
    assert_true(dist_from_anchor < Float32(2.2))

    # Trigger ENTER -> STAY -> EXIT lifecycle
    var wt = PhysicsWorld(gravity=0.0)
    _ = wt.add_trigger(TriggerVolume.sphere(77, Vec3(0.0, 0.0, 0.0), 1.0))
    var mover = wt.add_sphere(Vec3(-2.0, 0.0, 0.0), 0.25, mass=1.0)
    wt.set_velocity(mover, Vec3(60.0, 0.0, 0.0))
    var saw_enter = False
    var saw_stay = False
    var saw_exit = False
    for _ in range(6):
        wt.step(dt)
        for tev in wt.trigger_events:
            if tev.trigger_id == 77:
                if tev.kind == TRIGGER_ENTER:
                    saw_enter = True
                elif tev.kind == TRIGGER_STAY:
                    saw_stay = True
                elif tev.kind == TRIGGER_EXIT:
                    saw_exit = True
    assert_true(saw_enter and saw_stay and saw_exit)

    # Bit-exact snapshot serialize -> deserialize -> deterministic step
    var bw1 = BlobWriter()
    w.serialize(bw1)
    var w2 = PhysicsWorld()
    var br = BlobReader(bw1.data.copy())
    w2.deserialize(br)

    for _ in range(30):
        w.step(dt)
        w2.step(dt)

    var bw_a = BlobWriter()
    var bw_b = BlobWriter()
    w.serialize(bw_a)
    w2.serialize(bw_b)
    assert_equal(fnv64(bw_a.data), fnv64(bw_b.data))
    assert_equal(float32_bits(w.bodies[bob_slot].pos.y), float32_bits(w2.bodies[bob_slot].pos.y))


def test_fa3_t1_character_controller_move_and_slide() raises:
    var w = PhysicsWorld(gravity=9.81)
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0, friction=0.3)
    # 0.22 m step riser (< default step_height=0.35 m), platform spanning z=[1.5, 6.5]
    _ = w.add_box(Vec3(0.0, 0.11, 4.0), Vec3(3.0, 0.11, 2.5), mass=0.0, friction=0.5)
    var crate_slot = w.add_box(Vec3(0.0, 0.62, 3.8), Vec3(0.4, 0.4, 0.4), mass=20.0, friction=0.3)
    w.rebuild_broadphase()

    var cfg = CharacterControllerConfig()
    var kcc = CharacterController(Vec3(0.0, 0.90, 0.6), cfg)
    var dt = Float32(1.0 / 60.0)
    var saw_step_up = False

    for _ in range(65):
        _ = kcc.move_and_slide(w, Vec3(0.0, 0.0, 4.0), dt)
        w.step(dt)
        if kcc.stepped_up:
            saw_step_up = True

    assert_true(kcc.grounded)
    assert_true(saw_step_up)
    assert_true(kcc.foot_position().y > Float32(0.18))
    assert_true(w.bodies[crate_slot].pos.z > Float32(3.85))

    # Verify jump impulse raises vertical velocity and lifts off ground
    var y_before_jump = kcc.pos.y
    _ = kcc.move_and_slide(w, Vec3(0.0, 0.0, 0.0), dt, jump_speed=5.0)
    for _ in range(5):
        _ = kcc.move_and_slide(w, Vec3(0.0, 0.0, 0.0), dt)
    assert_true(kcc.pos.y > y_before_jump + Float32(0.15))


def test_fa3_t1_vehicle_chassis_and_aero6dof() raises:
    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.54, 0.0)
    car.set_controls(throttle=0.8, brake=0.0, steer=0.15)

    var dt = Float32(1.0 / 60.0)
    for _ in range(60):
        car.substep_suspension(dt)

    assert_true(car.pos.z > Float32(1.0))
    assert_true(abs(car.pos.x) > Float32(0.02))
    assert_true(car.wheel_states[0].normal_load > Float32(100.0))

    # ISA atmosphere + Aero6DofState flight dynamics
    var isa0 = eval_isa_atmosphere(0.0)
    var isa10k = eval_isa_atmosphere(10000.0)
    assert_true(abs(isa0[0] - Float32(1.225)) < Float32(0.01))
    assert_true(isa10k[0] < Float32(0.45) and isa10k[0] > Float32(0.35))

    var st = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 120.0))
    var acfg = Aero6DofConfig()
    var actrl = Aero6DofControls()
    actrl.throttle = Float32(1.0)
    actrl.pitch_cmd = Float32(0.3)

    for _ in range(30):
        step_aero6dof(st, acfg, actrl, dt, substeps=4)

    assert_true(st.pos.z > Float32(50.0))
    assert_true(st.dynamic_pressure > Float32(100.0))
    assert_true(abs(st.cl) > Float32(1.0e-4))


def test_fa3_t1_xpbd_chain_mesh_and_coupled() raises:
    var dt = Float32(1.0 / 60.0)
    var chain = ParticleChain.line(Vec3(0.0, 3.0, 0.0), Vec3(2.0, 3.0, 0.0), 6, mass=0.5)
    chain.pin(0, Vec3(0.0, 3.0, 0.0))
    for _ in range(20):
        chain.step(dt, 4, 9.81, List[Vec3](), List[ChainCollider](), iterations=4)

    assert_true((chain.position[0] - Vec3(0.0, 3.0, 0.0)).length() < Float32(1.0e-5))
    assert_true(chain.position[5].y < Float32(2.90))
    assert_true(abs(chain.length() - chain.rest_length()) < Float32(0.15))

    # 2x2 cloth mesh with top corners pinned
    var pmesh = ParticleMesh()
    var p0 = pmesh.add_particle(Vec3(0.0, 2.0, 0.0), 0.0, 0.05)
    var p1 = pmesh.add_particle(Vec3(1.0, 2.0, 0.0), 0.0, 0.05)
    var p2 = pmesh.add_particle(Vec3(0.0, 1.0, 0.0), 0.5, 0.05)
    var p3 = pmesh.add_particle(Vec3(1.0, 1.0, 0.0), 0.5, 0.05)
    pmesh.add_link(p0, p1, 0.0)
    pmesh.add_link(p0, p2, 0.0)
    pmesh.add_link(p1, p3, 0.0)
    pmesh.add_link(p2, p3, 0.0)
    pmesh.add_link(p0, p3, 0.0)
    for _ in range(20):
        pmesh.step(dt, 4, 9.81, List[Vec3](), List[ChainCollider](), iterations=4)
    assert_true(pmesh.link_error() < Float32(0.02))

    # Two-way RigidAttachment impulse feedback into SolverBodies
    var sb = SolverBodies()
    sb.reset(1)
    sb.px[0] = 0.0
    sb.py[0] = 5.0
    sb.pz[0] = 0.0
    sb.inv_mass[0] = 0.2
    sb.set_body_inertia(0, Vec3(1.0, 1.0, 1.0))

    var tether = ParticleChain.line(Vec3(1.0, 5.0, 0.0), Vec3(1.0, 3.0, 0.0), 5, mass=1.0)
    var atts = List[RigidAttachment]()
    atts.append(RigidAttachment(0, 0, Vec3(1.0, 0.0, 0.0), compliance=0.0))
    tether.step_coupled(dt, 4, 9.81, List[Vec3](), List[ChainCollider](), atts, sb, iterations=2)
    assert_true(sb.vy[0] < Float32(-0.01))
    assert_true(sb.wz[0] < Float32(-0.01))


def test_fa3_t1_articulated_ragdoll_modes_and_impulse() raises:
    var biped = _make_procedural_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1, 1, 1), 0.0)

    var rag = ArticulatedRagdoll()
    _ = rag.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
    _ = rag.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)

    var dt = Float32(1.0 / 60.0)
    rag.handoff_from_skeleton(skel, place, biped, dt)
    rag.set_active_pd(1.0)
    rag.apply_impulse_at_bone(1, Vec3(0.0, 0.0, 45.0), Vec3(0.0, 1.3, 0.0))
    for _ in range(10):
        rag.step(dt, substeps=4)
    assert_true(rag.mean_speed() > Float32(0.05))

    rag.go_limp()
    for _ in range(240):
        rag.step(dt, substeps=4)
        if rag.settled():
            break
    assert_true(rag.settled())
    assert_true(rag.mean_speed() < SETTLE_SPEED)

    # Zero-G mode disables gravity and floor plane
    rag.set_zero_g(True)
    assert_equal(float32_bits(rag.gravity.length()), float32_bits(Float32(0.0)))
    assert_false(rag.has_floor)
    var pelvis_y0 = rag.pelvis_world_position().y
    rag.apply_impulse_at_bone(0, Vec3(0.0, 20.0, 0.0))
    for _ in range(20):
        rag.step(dt, substeps=4)
    assert_true(rag.pelvis_world_position().y > pelvis_y0 + Float32(0.05))


def test_fa3_t1_flanim_ecs_roundtrip_socket_footplant_ragdoll_sm() raises:
    var g = AnimGraphDef()
    g.name = "HeroLocomotion"
    g.params.append("speed")
    var bs = BlendSpaceDef()
    bs.name = "Jog"
    bs.param = 0
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 5.0))
    g.blend_spaces.append(bs^)
    var st = StateDef()
    st.name = "Run"
    st.blend = 0
    st.when.append(Condition(0, OP_GT, 0.5))
    g.states.append(st^)

    var bytes = save_flanim_bytes(g)
    var tag = UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
    assert_equal(tag, FLANIM_FOURCC)
    var g2 = load_flanim_bytes(bytes)
    assert_equal(g2.name, "HeroLocomotion")
    assert_equal(len(g2.states), 1)

    var bw = BlobWriter()
    serialize_flanim(g, bw)
    var br = BlobReader(bw.data.copy())
    var g3 = deserialize_flanim(br)
    assert_equal(g3.name, "HeroLocomotion")

    var dt = Float32(1.0 / 60.0)
    var wa = PhysicsWorld()
    _ = wa.add_box(Vec3(0.0, -0.5, 0.0), Vec3(30.0, 0.5, 30.0), mass=0.0)
    _ = wa.add_box(Vec3(0.14, 0.125, 0.0), Vec3(0.25, 0.125, 0.40), mass=0.0)
    var shield_slot = wa.add_kinematic_box(Vec3(), Vec3(0.2, 0.3, 0.05))
    var shield_id = wa.body_id(shield_slot)
    wa.rebuild_broadphase()

    var biped = _make_procedural_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1, 1, 1), 0.0)

    var sock = BoneSocketAttachment("Shield", 4, 50, Vec3(0.05, 0.0, 0.1), Quat(), Vec3(1, 1, 1), shield_id)
    sock.sync_to_physics(skel, place, wa, dt)
    skel.pose.r[3] = angle_axis(0.8, Vec3(0, 1, 0))
    skel.update_matrices(biped)
    sock.sync_to_physics(skel, place, wa, dt)
    assert_true(wa.bodies[shield_slot].linear_vel.length() > Float32(1.0))

    var fp = FootPlantSolver(0)
    fp.add_leg(5, 6, 7, 0.04)
    fp.add_leg(8, 9, 10, 0.04)
    var stance: List[Bool] = [True, True]
    var ray_hits = 0
    for _ in range(15):
        skel.reset_to_rest(biped)
        ray_hits = solve_foot_plant_world(fp, wa, skel, biped, place, stance, dt)
    assert_equal(ray_hits, 2)
    assert_true(fp.foot_world_position(1, skel, place).y > fp.foot_world_position(0, skel, place).y + Float32(0.18))

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

    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(15.0, 0.0, 0.0), Vec3(0.0, 1.3, 0.0), go_limp_threshold=100.0, dt=dt)
    assert_equal(rsm.state, RAGDOLL_SM_ACTIVE_PD)
    rsm.update(wa, skel, place, biped, rest_pose, dt)

    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(0.0, 15.0, -160.0), Vec3(0.0, 1.35, 0.0), go_limp_threshold=100.0, dt=dt)
    var saw_getup = False
    for _ in range(180):
        rsm.update(wa, skel, place, biped, rest_pose, dt)
        if rsm.state == RAGDOLL_SM_GETUP_SUPINE or rsm.state == RAGDOLL_SM_GETUP_PRONE:
            saw_getup = True
        if saw_getup and rsm.state == RAGDOLL_SM_ANIMATED:
            break
    assert_true(saw_getup)
    assert_equal(rsm.state, RAGDOLL_SM_ANIMATED)
    assert_true(rsm.recovery_count >= 1)
    assert_true(rsm.last_pelvis_snap_m < Float32(1.0e-4))


# ============================================================================
# FA4 Tier 1: Rendering, RenderGraph, Headless Vulkan Probe, FLIP & ImageDiff
# ============================================================================

def test_fa4_t1_scalable_renderer_tier0_mesh_material_render_readback() raises:
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.95, 0.25, 0.15)))
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.6, 0.6, 0.6))
    var mesh_id = ren.register_mesh(mb^.finish(), mat_id)

    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.0), target=Vec3(0.0, 0.5, 0.0))
    var ext = RenderExtractor()
    ext.begin_extraction(rw, 1)
    ext.extract_entity(rw, UInt64(101), mesh_id, Mat4(), mat_id)
    ext.end_extraction()

    var blank_rw = RenderWorld()
    var empty_ui = UiDrawList()
    assert_true(ren.render_world(blank_rw, empty_ui, capture_readback=True))
    var blank_px = ren.readback_pixels()
    assert_equal(len(blank_px), 128 * 96)

    assert_true(ren.render_world(rw, empty_ui, capture_readback=True))
    var mesh_px = ren.readback_pixels()
    assert_equal(len(mesh_px), 128 * 96)
    var mesh_diff = 0
    for i in range(len(mesh_px)):
        if mesh_px[i] != blank_px[i]:
            mesh_diff += 1
    assert_true(mesh_diff > 0)

    var ui = UiDrawList()
    ui.add_rect(4.0, 4.0, 20.0, 12.0, 0xFFFF2020)
    assert_true(ren.render_world(rw, ui, capture_readback=True))
    var px = ren.readback_pixels()
    assert_equal(len(px), 128 * 96)
    assert_true(ren.cpu_canvas.triangles_drawn > 0)
    assert_equal(px[8 * 128 + 8], UInt32(0xFFFF2020))
    var diff_count = 0
    for i in range(len(px)):
        if px[i] != blank_px[i]:
            diff_count += 1
    assert_true(diff_count > 0)
    ren.shutdown()


def test_fa4_t1_scalable_renderer_resize_and_multi_frame() raises:
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.2, 0.8, 0.3)))
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.5, 0.5, 0.5))
    var mesh_id = ren.register_mesh(mb^.finish(), mat_id)

    var db = DoubleBufferedRenderWorld()
    var ext = RenderExtractor()
    var ui = UiDrawList()

    ext.begin_extraction(db.write_world(), 1)
    db.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.2), target=Vec3(0.0, 0.5, 0.0))
    ext.extract_entity(db.write_world(), UInt64(500), mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1, 1, 1)), mat_id)
    ext.end_extraction()
    db.swap_buffers()
    assert_true(ren.render_world(db.read_world(), ui, capture_readback=True))
    assert_equal(len(ren.readback_pixels()), 128 * 96)

    assert_true(ren.resize(160, 120))
    assert_equal(ren.width, 160)
    assert_equal(ren.height, 120)

    ext.begin_extraction(db.write_world(), 2)
    db.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.2), target=Vec3(0.0, 0.5, 0.0))
    ext.extract_entity(db.write_world(), UInt64(500), mesh_id, trs(Vec3(0.8, 0.0, 0.0), Quat(), Vec3(1, 1, 1)), mat_id)
    ext.end_extraction()
    db.swap_buffers()
    assert_true(db.read_world().instances[0].has_prev)
    assert_true(ren.render_world(db.read_world(), ui, capture_readback=True))
    assert_equal(len(ren.readback_pixels()), 160 * 120)
    ren.shutdown()


def test_fa4_t1_aaa_render_graph_aliasing_and_barriers() raises:
    var rg = AaaRenderGraph()
    rg.begin_frame()
    var r0 = rg.add_resource("vis_buf", 4 * 1024 * 1024, transient=True, is_image=True)
    var r1 = rg.add_resource("lighting_buf", 4 * 1024 * 1024, transient=True, is_image=False)
    var r2 = rg.add_resource("post_buf", 2 * 1024 * 1024, transient=True, is_image=True)

    var p0_acc = List[RgResourceAccess]()
    p0_acc.append(RgResourceAccess(r0, RG_USAGE_COLOR_ATT_WRITE))
    _ = rg.add_pass("vis_pass", p0_acc, RG_QUEUE_GRAPHICS)

    var p1_acc = List[RgResourceAccess]()
    p1_acc.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_READ))
    p1_acc.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("async_shade", p1_acc, RG_QUEUE_ASYNC_COMPUTE)

    var p2_acc = List[RgResourceAccess]()
    p2_acc.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_READ))
    p2_acc.append(RgResourceAccess(r2, RG_USAGE_COLOR_ATT_WRITE))
    _ = rg.add_pass("post_pass", p2_acc, RG_QUEUE_GRAPHICS)

    assert_true(rg.compile(heap_capacity_bytes=16 * 1024 * 1024))
    var m = rg.aliasing_metrics()
    assert_equal(m.transient_resource_count, 3)
    assert_true(m.aliased_peak_bytes < m.unaliased_bytes)
    assert_true(m.saved_bytes > 0)
    assert_true(m.compiled_barrier_count >= 2)
    assert_equal(rg.async_compute_pass_count, 1)


def test_fa4_t1_aaa_renderer_headless_probe_and_fallback() raises:
    var ren = FlintAaaVulkanRenderer(
        width=128,
        height=96,
        tsr_scale=1.0,
        max_draws=1024,
        max_instances=256,
        env_cube_dim=32,
    )
    var metrics = ren.get_render_graph_aliasing_metrics()
    assert_true(metrics.compiled_barrier_count >= 7)

    if ren.active:
        var mat_id = ren.add_material(AaaMaterial(Vec3(0.8, 0.4, 0.2), roughness=0.4, metallic=0.1))
        assert_true(mat_id >= 0)
        var mb = MeshBuilder()
        mb.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.6, 0.6, 0.6))
        var entry = ren.add_mesh(mb^.finish(), mat_id)

        ren.begin_frame(cam_pos=Vec3(0.0, 1.5, 4.0), cam_target=Vec3(0.0, 0.5, 0.0))
        assert_true(ren.render_frame(capture_readback=True))
        var blank_px = ren.readback_pixels()
        assert_equal(len(blank_px), 128 * 96)

        ren.begin_frame(cam_pos=Vec3(0.0, 1.5, 4.0), cam_target=Vec3(0.0, 0.5, 0.0))
        ren.draw_instance(entry.mesh_id, Mat4())
        assert_true(ren.render_frame(capture_readback=True))
        var px = ren.readback_pixels()
        assert_equal(len(px), 128 * 96)
        var mesh_diff = compare_pixels(px, blank_px, 128, 96, tolerance=4)
        assert_true(mesh_diff.over_tolerance >= 500)
        assert_true(mesh_diff.worst >= 32)
        assert_true(mesh_diff.flip_mean > 0.01)
        var hdr = ren.readback_resolved_hdr()
        assert_equal(len(hdr), 128 * 96)
        for i in range(0, len(hdr), 64):
            var v = hdr[i]
            assert_true(v.x == v.x and v.y == v.y and v.z == v.z)
    ren.shutdown()

    # ScalableRenderer Tier 2 must either run Vulkan or cleanly fall back to Tier 0 SIMD
    var sr = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=128, height=96)
    assert_true(sr.uses_vulkan() or sr.tier == RENDER_TIER_0_HEADLESS_SIMD)
    var rw = RenderWorld()
    var ui = UiDrawList()
    assert_true(sr.render_world(rw, ui, capture_readback=True))
    assert_equal(len(sr.readback_pixels()), 128 * 96)
    sr.shutdown()


def test_fa4_t1_flip_identical_and_different_images() raises:
    var n = 64 * 64
    var img_a = List[UInt32](length=n, fill=0xFF4080C0)
    var img_b = List[UInt32](length=n, fill=0xFF4080C0)
    var emap_same = flip_error_map(img_a, img_b, 64, 64)
    assert_equal(len(emap_same), n)
    assert_true(flip_mean(emap_same) < 1.0e-6)

    var black = List[UInt32](length=n, fill=0xFF000000)
    var white = List[UInt32](length=n, fill=0xFFFFFFFF)
    var emap_diff = flip_error_map(black, white, 64, 64)
    assert_equal(len(emap_diff), n)
    assert_true(flip_mean(emap_diff) > 0.2)

    var heat = flip_heat_image(emap_diff)
    assert_equal(len(heat), n)
    assert_equal(heat[0] & UInt32(0xFF000000), UInt32(0xFF000000))


def test_fa4_t1_image_diff_compare_pixels_pass_and_fail() raises:
    var n = 64 * 64
    var ref_px = List[UInt32](length=n, fill=0xFF336699)
    var act_same = List[UInt32](length=n, fill=0xFF336699)
    var d_ok = compare_pixels(act_same, ref_px, 64, 64, tolerance=2)
    assert_true(d_ok.same_size)
    assert_equal(d_ok.differing, 0)
    assert_true(d_ok.flip_mean < 1.0e-6)
    assert_true(d_ok.passes(0, 0.01))

    var act_bad = ref_px.copy()
    for i in range(n // 2):
        act_bad[i] = 0xFF5386B9  # +32 per channel
    var d_bad = compare_pixels(act_bad, ref_px, 64, 64, tolerance=2)
    assert_true(d_bad.same_size)
    assert_true(d_bad.differing > 0)
    assert_true(d_bad.over_tolerance > 0)
    assert_true(d_bad.worst >= 32)
    assert_false(d_bad.passes(100, 0.01))


def test_fa4_t1_image_diff_compare_with_png_roundtrip() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_dir = "/tmp/flint-e2e-opaque-t1-" + String(pid)
    assert_true(make_dirs(tmp_dir))
    var png_path = tmp_dir + "/ref_64.png"

    var w = 64
    var h = 64
    var px = List[UInt32](capacity=w * h)
    for y in range(h):
        for x in range(w):
            var r = UInt32((x * 4) & 255)
            var g = UInt32((y * 4) & 255)
            var b = UInt32(128)
            px.append(UInt32(0xFF000000) | (r << 16) | (g << 8) | b)

    var wrote = write_png(png_path, px, w, h, stamped=False)
    assert_true(wrote)

    var d = compare_with_png(px, w, h, png_path, tolerance=2)
    var ok = d.same_size and d.passes(0, 0.01)

    _ = remove_file(png_path)
    var owned_dir = tmp_dir
    _ = external_call["rmdir", Int32](owned_dir.as_c_string_span())
    assert_true(ok)


def main() raises:
    var passed = 0
    var total = 14

    test_fa3_t1_physics_world_bodies_step_and_queries()
    print("[PASS] test_fa3_t1_physics_world_bodies_step_and_queries")
    passed += 1

    test_fa3_t1_physics_joints_triggers_and_serialize()
    print("[PASS] test_fa3_t1_physics_joints_triggers_and_serialize")
    passed += 1

    test_fa3_t1_character_controller_move_and_slide()
    print("[PASS] test_fa3_t1_character_controller_move_and_slide")
    passed += 1

    test_fa3_t1_vehicle_chassis_and_aero6dof()
    print("[PASS] test_fa3_t1_vehicle_chassis_and_aero6dof")
    passed += 1

    test_fa3_t1_xpbd_chain_mesh_and_coupled()
    print("[PASS] test_fa3_t1_xpbd_chain_mesh_and_coupled")
    passed += 1

    test_fa3_t1_articulated_ragdoll_modes_and_impulse()
    print("[PASS] test_fa3_t1_articulated_ragdoll_modes_and_impulse")
    passed += 1

    test_fa3_t1_flanim_ecs_roundtrip_socket_footplant_ragdoll_sm()
    print("[PASS] test_fa3_t1_flanim_ecs_roundtrip_socket_footplant_ragdoll_sm")
    passed += 1

    test_fa4_t1_scalable_renderer_tier0_mesh_material_render_readback()
    print("[PASS] test_fa4_t1_scalable_renderer_tier0_mesh_material_render_readback")
    passed += 1

    test_fa4_t1_scalable_renderer_resize_and_multi_frame()
    print("[PASS] test_fa4_t1_scalable_renderer_resize_and_multi_frame")
    passed += 1

    test_fa4_t1_aaa_render_graph_aliasing_and_barriers()
    print("[PASS] test_fa4_t1_aaa_render_graph_aliasing_and_barriers")
    passed += 1

    test_fa4_t1_aaa_renderer_headless_probe_and_fallback()
    print("[PASS] test_fa4_t1_aaa_renderer_headless_probe_and_fallback")
    passed += 1

    test_fa4_t1_flip_identical_and_different_images()
    print("[PASS] test_fa4_t1_flip_identical_and_different_images")
    passed += 1

    test_fa4_t1_image_diff_compare_pixels_pass_and_fail()
    print("[PASS] test_fa4_t1_image_diff_compare_pixels_pass_and_fail")
    passed += 1

    test_fa4_t1_image_diff_compare_with_png_roundtrip()
    print("[PASS] test_fa4_t1_image_diff_compare_with_png_roundtrip")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
