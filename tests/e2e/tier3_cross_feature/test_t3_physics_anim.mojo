"""Tier 3 Cross-Feature Combinations: Physics & Animation Interactions (15 tests).
Genuine multi-frame cross-subsystem integration across `EngineApp`, `flint.ecs.World`,
`PhysicsWorld` (`VehicleChassis`, `ArticulatedRagdoll`, `Aero6DofState`, `Bvh8MeshCollider`),
`AnimPlugin` (`SkeletonInstance`, `Animator`, `FootPlantSolver`, `RbfPoseDriver`, `FourBarLinkageDriver`),
`AudioServer`, `CytomicSVF`, `ModalResonator`, and `FlintReplayStream`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, max, sqrt
from flint.anim.animator import Animator
from flint.anim.correctives import RbfPoseDriver
from flint.anim.flanim_ecs import solve_foot_plant_world
from flint.anim.graph import AnimGraphDef, Condition, OP_GT, StateDef
from flint.anim.mecha_ik import FourBarLinkageDriver
from flint.anim.plugin import AnimAssetCache, AnimPlugin
from flint.anim.pose import Pose, SkeletonInstance, blend_into
from flint.anim.procedural import FootPlantSolver, Placement
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.anim.sync import ClipInfo
from flint.app import AppConfig, EngineApp
from flint.assets.model import NativeAnimation, NativeChannel, NativeModel, NativeNode
from flint.assets.replay_stream import FlintReplayStream
from flint.audio import AudioServer, BUS_SFX, compute_woodworth_hrtf
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.core.ik import aim_rotation
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.core.transform import Quat
from flint.physics.aero6dof import Aero6DofConfig, Aero6DofControls, Aero6DofState
from flint.physics.events import EVENT_BEGIN, EVENT_HIT
from flint.physics.filter import CollisionFilter
from flint.physics.plugin import ColliderComponent, PhysicsPlugin, RigidBodyComponent
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld
from flint.platform.dsp import CytomicSVF, ModalResonator, compute_spatial_3d
from flint.scene import GlobalTransform, LocalTransform, Parent, TransformPlugin, set_parent
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import create_mock_bvh8_floor, create_mock_pcm16_sine


def _quat_norm_sq(q: Quat) -> Float32:
    return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w


def _make_t3_biped_model() -> NativeModel:
    var m = NativeModel()
    # 0: Pelvis, 1: Thigh.L, 2: Shin.L, 3: Foot.L, 4: Chest, 5: Neck, 6: Head
    var names = [
        String("Pelvis"),
        String("Thigh.L"),
        String("Shin.L"),
        String("Foot.L"),
        String("Chest"),
        String("Neck"),
        String("Head"),
    ]
    var parents = [-1, 0, 1, 2, 0, 4, 5]
    var offsets = [
        Vec3(0.0, 1.0, 0.0),
        Vec3(-0.15, -0.05, 0.0),
        Vec3(0.0, -0.45, 0.0),
        Vec3(0.0, -0.42, 0.0),
        Vec3(0.0, 0.45, 0.0),
        Vec3(0.0, 0.22, 0.0),
        Vec3(0.0, 0.14, 0.0),
    ]
    for i in range(len(names)):
        var nd = NativeNode()
        nd.name = names[i]
        nd.translation = offsets[i]
        nd.base_translation = offsets[i]
        nd.rotation = Quat()
        nd.base_rotation = Quat()
        nd.scale = Vec3(1.0, 1.0, 1.0)
        nd.base_scale = Vec3(1.0, 1.0, 1.0)
        m.nodes.append(nd^)
        if parents[i] >= 0:
            m.nodes[parents[i]].children.append(i)
        else:
            m.roots.append(i)

    # Clip 0: Idle (slight chest breath)
    var idle = NativeAnimation()
    idle.name = "Idle"
    idle.duration = Float32(1.0)
    var ch0 = NativeChannel()
    ch0.node = 4
    ch0.path = 0
    ch0.times.append(Float32(0.0)); ch0.times.append(Float32(1.0))
    ch0.values.append(Float32(0.0)); ch0.values.append(Float32(0.45)); ch0.values.append(Float32(0.0))
    ch0.values.append(Float32(0.0)); ch0.values.append(Float32(0.47)); ch0.values.append(Float32(0.0))
    idle.channels.append(ch0^)
    m.animations.append(idle^)

    # Clip 1: Run (forward pelvic lean + rotation on Pelvis)
    var run = NativeAnimation()
    run.name = "Run"
    run.duration = Float32(1.0)
    var ch1 = NativeChannel()
    ch1.node = 0
    ch1.path = 1
    ch1.times.append(Float32(0.0)); ch1.times.append(Float32(1.0))
    ch1.values.append(Float32(0.0)); ch1.values.append(Float32(0.3826834)); ch1.values.append(Float32(0.0)); ch1.values.append(Float32(0.9238795))
    ch1.values.append(Float32(0.0)); ch1.values.append(Float32(0.3826834)); ch1.values.append(Float32(0.0)); ch1.values.append(Float32(0.9238795))
    run.channels.append(ch1^)
    var ch2 = NativeChannel()
    ch2.node = 0
    ch2.path = 0
    ch2.times.append(Float32(0.0)); ch2.times.append(Float32(1.0))
    ch2.values.append(Float32(0.0)); ch2.values.append(Float32(1.0)); ch2.values.append(Float32(1.2))
    ch2.values.append(Float32(0.0)); ch2.values.append(Float32(1.0)); ch2.values.append(Float32(2.4))
    run.channels.append(ch2^)
    m.animations.append(run^)
    return m^


# 1. Broadphase + PhysicsWorld collision -> resting contact & raycast query
def test_t3_broadphase_to_solver_candidates(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var box_slot = pw.add_box(Vec3(0.0, 2.0, 0.0), Vec3(0.5, 0.5, 0.5), mass=2.0)
    var box_id = pw.body_id(box_slot)
    pw.rebuild_broadphase()

    var saw_hit = False
    for _ in range(60):
        pw.step(Float32(1.0 / 60.0))
        for ei in range(len(pw.events)):
            if pw.events[ei].kind == EVENT_HIT or pw.events[ei].kind == EVENT_BEGIN:
                saw_hit = True

    var final_pos = pw.bodies[box_slot].pos
    ctx.assert_true(saw_hit, "PhysicsWorld broadphase + solver generates contact event")
    ctx.assert_true(final_pos.y > Float32(0.35) and final_pos.y < Float32(0.70), "dynamic box settles onto static floor")
    var rc = pw.raycast(Vec3(0.0, 5.0, 0.0), Vec3(0.0, -1.0, 0.0), Float32(10.0))
    ctx.assert_true(rc.hit and rc.body_id == box_id, "PhysicsWorld raycast hits settled dynamic box")


# 2. Character IK foot placement onto Bvh8MeshCollider + stepped box in PhysicsWorld
def test_t3_ik_foot_placement_surface_normal(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_static_mesh(create_mock_bvh8_floor(20.0, 0.0))
    _ = pw.add_box(Vec3(0.0, 0.125, 0.8), Vec3(2.0, 0.125, 0.6), mass=0.0)
    pw.rebuild_broadphase()

    var model = _make_t3_biped_model()
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)

    var solver = FootPlantSolver(0)
    solver.add_leg(1, 2, 3, ankle_height=Float32(0.08), pole_local=Vec3(0.0, 0.4, 1.0))

    var in_stance = List[Bool]()
    in_stance.append(True)
    var ankle_y_flat = Float32(0.0)
    var ankle_y_step = Float32(0.0)
    var hits_total = 0
    for f in range(12):
        var z_pos = Float32(0.0) if f < 6 else Float32(0.8)
        var place = Placement(Vec3(0.0, 0.0, z_pos), Vec3(1.0, 1.0, 1.0), Float32(0.0))
        skel.reset_to_rest(model)
        hits_total += solve_foot_plant_world(solver, pw, skel, model, place, in_stance, Float32(1.0 / 60.0))
        var wy = place.to_world(skel.global_position(3)).y
        if f == 5:
            ankle_y_flat = wy
        elif f == 11:
            ankle_y_step = wy

    ctx.assert_true(hits_total >= 10, "solve_foot_plant_world raycasts PhysicsWorld terrain/step hits")
    ctx.assert_true(ankle_y_step > ankle_y_flat + Float32(0.10), "foot IK adapts ankle height onto raised 0.25m step")
    ctx.assert_approx(_quat_norm_sq(skel.pose.r[1]), 1.0, 0.001, "thigh IK rotation normalized")
    ctx.assert_approx(_quat_norm_sq(skel.pose.r[2]), 1.0, 0.001, "shin IK rotation normalized")


# 3. Hierarchical entity transforms + aim constraint head tracking across frames
def test_t3_aim_head_tracking(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="T3AimTrack", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.startup()

    var char_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](char_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](char_e, GlobalTransform())

    var neck_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](neck_e, LocalTransform(Vec3(0.0, 1.6, 0.0)))
    _ = app.world.insert[GlobalTransform](neck_e, GlobalTransform())
    _ = set_parent(app.world, neck_e, char_e)

    var head_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](head_e, LocalTransform(Vec3(0.0, 0.12, 0.1)))
    _ = app.world.insert[GlobalTransform](head_e, GlobalTransform())
    _ = set_parent(app.world, head_e, neck_e)

    var target_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](target_e, LocalTransform(Vec3(4.0, 1.72, 4.0)))
    _ = app.world.insert[GlobalTransform](target_e, GlobalTransform())

    app.step_one_frame(Float32(1.0 / 60.0))
    var dt = Float32(1.0 / 60.0)
    var fwd_right = Vec3()
    var fwd_left = Vec3()
    for f in range(10):
        var tx = Float32(4.0) if f < 5 else Float32(-4.0)
        app.world.get_ptr_mut[LocalTransform](target_e).value()[unsafe_offset=0].translation = Vec3(tx, 1.72, 4.0)
        app.step_one_frame(dt)
        var neck_w = app.world.get[GlobalTransform](neck_e).translation
        var head_w = app.world.get[GlobalTransform](head_e).translation
        var tgt_w = app.world.get[GlobalTransform](target_e).translation
        var q = aim_rotation(neck_w, head_w, tgt_w, Quat())
        app.world.get_ptr_mut[LocalTransform](head_e).value()[unsafe_offset=0].rotation = q
        app.step_one_frame(dt)
        var head_rot = app.world.get[GlobalTransform](head_e).rotation
        if f == 4:
            fwd_right = head_rot.rotate(Vec3(0.0, 0.0, 1.0))
        elif f == 9:
            fwd_left = head_rot.rotate(Vec3(0.0, 0.0, 1.0))

    app.shutdown()
    ctx.assert_true(fwd_right.x > 0.2 and fwd_right.z > 0.2, "head world forward aims toward (+X, +Z) target")
    ctx.assert_true(fwd_left.x < -0.2 and fwd_left.z > 0.2, "head world forward tracks target crossing to (-X, +Z)")


# 4. Pacejka tire forces + VehicleChassis two-way rigid-body collision inside PhysicsWorld
def test_t3_pacejka_to_wheel_impulse(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.72, 0.0)
    var v_slot = pw.add_vehicle(car^)
    var crate_slot = pw.add_box(Vec3(0.0, 0.5, 2.5), Vec3(0.5, 0.5, 0.5), mass=25.0)
    pw.rebuild_broadphase()

    pw.set_vehicle_controls(v_slot, Float32(1.0), Float32(0.0), Float32(0.22))
    var saw_lat = False
    var saw_lon = False
    for _ in range(75):
        pw.step(Float32(1.0 / 60.0))
        var st = pw.vehicle_state(v_slot)
        if abs(st.wheel_states[0].lateral_force) > Float32(50.0):
            saw_lat = True
        if abs(st.wheel_states[2].longitudinal_force) > Float32(100.0):
            saw_lon = True

    var vst = pw.vehicle_state(v_slot)
    var crate_pos = pw.bodies[crate_slot].pos
    ctx.assert_true(vst.wheel_states[0].grounded, "vehicle front wheel grounded in PhysicsWorld")
    ctx.assert_true(saw_lat and saw_lon, "Pacejka tire produces lateral & longitudinal forces in PhysicsWorld")
    ctx.assert_true(vst.pos.z > Float32(1.5), "vehicle drives forward under Pacejka traction")
    ctx.assert_true(crate_pos.z > Float32(2.8), "vehicle imparts two-way impact impulse onto dynamic crate")


# 5. Physics impact velocity -> AudioServer spatial 3D mix + ModalResonator excitation
def test_t3_impact_velocity_to_audio_resonance(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    _ = pw.add_sphere(Vec3(0.0, 4.0, 0.0), Float32(0.5), mass=5.0)
    pw.rebuild_broadphase()

    var hit_speed = Float32(0.0)
    var hit_point = Vec3()
    for _ in range(70):
        pw.step(Float32(1.0 / 60.0))
        for ei in range(len(pw.events)):
            if (pw.events[ei].kind == EVENT_HIT or pw.events[ei].kind == EVENT_BEGIN) and pw.events[ei].speed > hit_speed:
                hit_speed = pw.events[ei].speed
                hit_point = pw.events[ei].point

    ctx.assert_true(hit_speed > Float32(4.0), "falling sphere produces real >4 m/s impact event in PhysicsWorld")

    var impact_gain = min(Float32(1.0), hit_speed / Float32(12.0))
    var res = ModalResonator(220.0, 0.05, impact_gain, 48000.0)
    var pcm = List[Int16](capacity=512)
    for i in range(512):
        var excit = impact_gain if i == 0 else Float32(0.0)
        var s = res.step(excit)
        pcm.append(Int16(Int(max(Float32(-1.0), min(Float32(1.0), s)) * Float32(32767.0))))

    var srv = AudioServer(48000, 256)
    var aid = srv.register_pcm16_asset(pcm, 1, 48000)
    srv.set_listener(Vec3(0.0, 1.0, 4.0), Vec3(), Vec3(0.0, 0.0, -1.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    _ = srv.play_3d(aid, hit_point, Vec3(), volume=impact_gain, bus_id=BUS_SFX, looping=False)
    var out_lr = List[Float32](length=512, fill=0.0)
    srv.render_offline_quantum(out_lr, 256)
    var rms_sum = Float32(0.0)
    for i in range(len(out_lr)):
        rms_sum += abs(out_lr[i])
    ctx.assert_true(rms_sum > Float32(0.5), "physics impact excites ModalResonator mixed into AudioServer output")


# 6. Moving ECS RigidBody + AudioEmitterComponent -> AudioServer Doppler + Woodworth HRTF
def test_t3_moving_body_doppler_audio(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="T3DopplerAudio", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.add_plugin(AudioPlugin(headless=True))
    app.startup()

    var aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(
        create_mock_pcm16_sine(440.0, 4096, 48000), 1, 48000
    )

    var lis_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](lis_e, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_e, AudioListenerComponent(active=True, priority=1))

    var emt_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](emt_e, LocalTransform(Vec3(2.0, 0.0, 30.0)))
    _ = app.world.insert[GlobalTransform](emt_e, GlobalTransform())
    var rb = RigidBodyComponent.dynamic(2.0)
    rb.linear_vel = Vec3(0.0, 0.0, -25.0)
    rb.linear_damping = 0.0
    _ = app.world.insert[RigidBodyComponent](emt_e, rb)
    _ = app.world.insert[ColliderComponent](emt_e, ColliderComponent.sphere(0.3))
    _ = app.world.insert[AudioEmitterComponent](
        emt_e, AudioEmitterComponent(asset_id=aid, asset_path="siren.wav", looping=True, spatial=True, play_on_spawn=True)
    )

    var sp_far = compute_spatial_3d(Vec3(), Vec3(), Vec3(1, 0, 0), Vec3(2.0, 0.0, 30.0), Vec3(0.0, 0.0, -25.0))
    for _ in range(20):
        app.step_one_frame(Float32(1.0 / 60.0))
    for _ in range(15):
        if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() >= 1:
            break
        _ = external_call["usleep", Int32](UInt32(2000))

    var pos_now = app.world.get[GlobalTransform](emt_e).translation
    var active_voices = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count()
    var emt_valid = app.world.get_ptr[AudioEmitterComponent](emt_e).value()[unsafe_offset=0].handle.is_valid()
    var sp_near = compute_spatial_3d(Vec3(), Vec3(), Vec3(1, 0, 0), pos_now, Vec3(0.0, 0.0, -25.0))
    var hrtf = compute_woodworth_hrtf(Vec3(), Vec3(), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0), pos_now, Vec3(0.0, 0.0, -25.0))
    app.shutdown()

    ctx.assert_true(pos_now.z < Float32(23.0), "rigid body moved closer to listener across 20 frames")
    ctx.assert_true(emt_valid and active_voices >= 1, "AudioPlugin activated spatial voice in AudioServer")
    ctx.assert_true(sp_near.doppler_pitch > Float32(1.02) and sp_near.gain > sp_far.gain, "approaching body increases gain and Doppler pitch")
    ctx.assert_true(hrtf.delay_l_samples > hrtf.delay_r_samples, "Woodworth HRTF computes positive +X lateral ITD")


# 7. ArticulatedRagdoll self-collision isolation + two-way external projectile momentum transfer
def test_t3_collision_filtering_ragdoll(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=0.0)
    var model = _make_t3_biped_model()
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var rag = ArticulatedRagdoll()
    rag.gravity = Vec3()
    rag.bind_skeleton(model)
    var pelvis = rag.add_limb_from_model(model, 0, -1, 18.0, 0.18, 0.30)
    var chest = rag.add_limb_from_model(model, 4, pelvis, 16.0, 0.18, 0.35)
    rag.handoff_from_skeleton(skel, Placement(), model, Float32(1.0 / 60.0))
    rag.go_limp()
    var r_slot = pw.add_ragdoll(rag^)

    for _ in range(15):
        pw.step(Float32(1.0 / 60.0))
    var mom_idle = pw.ragdoll_state(r_slot).total_linear_momentum()
    ctx.assert_true(mom_idle.length() < Float32(0.5), "ragdoll internal limbs remain stable without self-collision explosion")

    var proj_slot = pw.add_sphere(Vec3(-1.2, 1.45, 0.0), Float32(0.28), mass=4.0)
    pw.bodies[proj_slot].linear_vel = Vec3(14.0, 0.0, 0.0)
    pw.rebuild_broadphase()

    for _ in range(20):
        pw.step(Float32(1.0 / 60.0))

    var mom_after = pw.ragdoll_state(r_slot).total_linear_momentum()
    var proj_vx = pw.bodies[proj_slot].linear_vel.x
    ctx.assert_true(mom_after.x > Float32(5.0), "external projectile transfers two-way linear momentum to ArticulatedRagdoll")
    ctx.assert_true(proj_vx < Float32(13.5), "projectile slows down upon impacting ragdoll limb")
    _ = chest


# 8. ArticulatedRagdoll 6-DOF joint pivot invariance under high angular velocity
def test_t3_rodrigues_joint_lever_arm(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    var model = _make_t3_biped_model()
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var rag = ArticulatedRagdoll()
    rag.bind_skeleton(model)
    var limb0 = rag.add_limb_from_model(model, 0, -1, 15.0, 0.14, 0.30)
    var limb1 = rag.add_limb_from_model(model, 4, limb0, 8.0, 0.12, 0.35, is_hinge=True)
    rag.handoff_from_skeleton(skel, Placement(Vec3(0.0, 1.5, 0.0), Vec3(1, 1, 1), 0.0), model, Float32(1.0 / 60.0))
    rag.states[limb1].angular_vel = Vec3(0.0, 8.0, 0.0)
    rag.go_limp()
    var r_slot = pw.add_ragdoll(rag^)

    for _ in range(40):
        pw.step(Float32(1.0 / 60.0))

    var st = pw.ragdoll_state(r_slot)
    var p0_top = st.states[limb0].com_pos + st.states[limb0].rotation.rotate(st.limbs[limb1].parent_anchor_local - st.limbs[limb0].local_com_offset)
    var p1_root = st.states[limb1].com_pos + st.states[limb1].rotation.rotate(-st.limbs[limb1].local_com_offset)
    var gap = (p1_root - p0_top).length()
    ctx.assert_true(gap < Float32(0.08), "6-DOF ragdoll joint preserves anchor pivot distance across 40 rotated frames")


# 9. VehicleChassis suspension compression + normal load response over ramp bump
def test_t3_suspension_compression_load(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    _ = pw.add_box(Vec3(0.0, 0.075, 2.6), Vec3(4.0, 0.075, 0.5), mass=0.0)
    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.95, 0.0)
    var v_slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()

    pw.set_vehicle_controls(v_slot, Float32(0.0), Float32(0.0), Float32(0.0))
    for _ in range(30):
        pw.step(Float32(1.0 / 60.0))
    var eq = pw.vehicle_state(v_slot)
    var eq_comp = eq.wheel_states[0].compression
    var eq_load = eq.wheel_states[0].normal_load
    ctx.assert_true(eq.wheel_states[0].grounded and eq_load > Float32(1000.0), "vehicle settles to positive front normal load")

    pw.set_vehicle_controls(v_slot, Float32(1.0), Float32(0.0), Float32(0.0))
    var peak_comp = eq_comp
    var peak_load = eq_load
    for _ in range(75):
        pw.step(Float32(1.0 / 60.0))
        var st = pw.vehicle_state(v_slot)
        if st.wheel_states[0].compression > peak_comp:
            peak_comp = st.wheel_states[0].compression
        if st.wheel_states[0].normal_load > peak_load:
            peak_load = st.wheel_states[0].normal_load

    ctx.assert_true(peak_comp > eq_comp + Float32(0.015), "driving onto bump increases front suspension compression")
    ctx.assert_true(peak_load > eq_load * Float32(1.10), "suspension bump spikes wheel normal load")


# 10. FlintReplayStream + World.snapshot_blob + PhysicsWorld deterministic record & replay
def test_t3_replay_physics_snapshot(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="T3Replay", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.startup()

    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(20.0, 0.5, 20.0)))

    for i in range(3):
        var e = app.world.spawn()
        _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(Float32(i - 1) * 1.4, Float32(2.0 + Float32(i) * 0.6), 0.0)))
        _ = app.world.insert[GlobalTransform](e, GlobalTransform())
        _ = app.world.insert[RigidBodyComponent](e, RigidBodyComponent.dynamic(2.0))
        _ = app.world.insert[ColliderComponent](e, ColliderComponent.box(Vec3(0.4, 0.4, 0.4)))

    var cksum_init = app.world.state_checksum()
    var w_init = BlobWriter()
    app.world.snapshot_blob(w_init)
    app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].serialize(w_init)

    var replay = FlintReplayStream()
    replay.begin_recording("T3Replay", w_init)
    var dt = Float32(1.0 / 60.0)
    for t in range(20):
        app.step_one_frame(dt)
        var ib = List[UInt8]()
        ib.append(UInt8(t & 0xFF))
        replay.record_tick(dt, ib)
        if t == 9:
            var w_kf = BlobWriter()
            app.world.snapshot_blob(w_kf)
            replay.add_keyframe(10, w_kf)

    var w_end1 = BlobWriter()
    app.world.snapshot_blob(w_end1)
    app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].serialize(w_end1)
    var hash1 = replay.finish_recording(w_end1)
    var cksum1 = app.world.state_checksum()

    # Restore exact initial state and replay 20 recorded ticks
    var r_init = replay.initial_reader()
    app.world.restore_blob(r_init)
    app.resources.get_ptr_mut[PhysicsWorld]().value()[unsafe_offset=0].deserialize(r_init)
    for t in range(replay.tick_count()):
        app.step_one_frame(replay.tick_dts[t])

    var w_end2 = BlobWriter()
    app.world.snapshot_blob(w_end2)
    app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].serialize(w_end2)
    var hash2 = replay.finish_recording(w_end2)
    var cksum2 = app.world.state_checksum()
    app.shutdown()

    ctx.assert_equal(replay.tick_count(), 20, "replay recorded 20 fixed ticks")
    ctx.assert_equal(replay.nearest_keyframe_tick_at_or_before(15), 10, "keyframe lookup finds tick 10")
    ctx.assert_true(cksum1 != cksum_init, "20 simulation ticks advance ECS World state_checksum from initial snapshot")
    ctx.assert_equal(cksum1, cksum2, "ECS World state_checksum matches bit-for-bit after replay")
    ctx.assert_equal(hash1, hash2, "combined ECS + PhysicsWorld replay hash matches bit-for-bit")


# 11. CytomicSVF smoothing real bouncing sphere impact acceleration spikes
def test_t3_acceleration_filtering_svf(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0, restitution=0.75)
    var s_slot = pw.add_sphere(Vec3(0.0, 3.0, 0.0), Float32(0.4), mass=1.5, restitution=0.75)
    pw.rebuild_broadphase()

    var svf = CytomicSVF(8.0, 0.707, 60.0)
    var dt = Float32(1.0 / 60.0)
    var prev_vy = pw.bodies[s_slot].linear_vel.y
    var max_raw_acc = Float32(0.0)
    var max_filt_acc = Float32(0.0)
    for _ in range(60):
        pw.step(dt)
        var vy = pw.bodies[s_slot].linear_vel.y
        var raw_acc = (vy - prev_vy) / dt
        prev_vy = vy
        var filt = svf.process(raw_acc)
        if abs(raw_acc) > max_raw_acc:
            max_raw_acc = abs(raw_acc)
        if abs(filt) > max_filt_acc:
            max_filt_acc = abs(filt)

    ctx.assert_true(max_raw_acc > Float32(50.0), "bouncing sphere generates sharp real bounce acceleration spike")
    ctx.assert_true(max_filt_acc < max_raw_acc * Float32(0.8), "CytomicSVF attenuates physical bounce acceleration spike")


# 12. Multi-frame Animator state crossfade + Pose quaternion slerp on SkeletonInstance
def test_t3_anim_pose_slerp(mut ctx: TestContext) raises:
    var model = _make_t3_biped_model()
    var g = AnimGraphDef()
    g.name = "Locomotion"
    g.default_blend = Float32(0.25)
    g.params.append("speed")
    var st_run = StateDef()
    st_run.name = "Run"
    st_run.clip = 1
    st_run.when.append(Condition(0, OP_GT, Float32(0.5)))
    var st_idle = StateDef()
    st_idle.name = "Idle"
    st_idle.clip = 0
    g.states.append(st_run^)
    g.states.append(st_idle^)

    var clips = List[ClipInfo]()
    clips.append(ClipInfo("Idle", Float32(1.0), True, Float32(0.0)))
    clips.append(ClipInfo("Run", Float32(1.0), True, Float32(0.0)))

    var anim = Animator()
    anim.configure(g, clips)
    anim.bind(model, 0)
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)

    var dt = Float32(1.0 / 60.0)
    anim.set(0, Float32(0.0))
    for _ in range(4):
        anim.update(model, 0, dt)
    ctx.assert_equal(anim.state, 1, "Animator starts in Idle state")

    anim.set(0, Float32(1.0))
    var mid_y = Float32(0.0)
    for f in range(18):
        anim.update(model, 0, dt)
        skel.apply_pose(anim.pose, model)
        if f == 5:
            mid_y = skel.pose.r[0].y

    var p_check = Pose(2)
    p_check.assign(anim.pose)
    blend_into(p_check, anim.pose, Float32(0.5))
    ctx.assert_equal(anim.state, 0, "Animator completes transition to Run")
    ctx.assert_true(mid_y > Float32(0.02) and mid_y < Float32(0.38), "crossfade blends intermediate Pelvis rotation")
    ctx.assert_approx(_quat_norm_sq(skel.pose.r[0]), 1.0, 0.001, "blended skeleton rotation stays normalized")


# 13. Simultaneous FootPlantSolver + aim_rotation + RbfPoseDriver + FourBarLinkageDriver
def test_t3_simultaneous_ik_and_aim(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_static_mesh(create_mock_bvh8_floor(20.0, 0.0))
    pw.rebuild_broadphase()

    var model = _make_t3_biped_model()
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)

    var foot_solver = FootPlantSolver(0)
    foot_solver.add_leg(1, 2, 3, ankle_height=Float32(0.08))
    var in_stance = List[Bool](); in_stance.append(True)

    var rbf = RbfPoseDriver(node=0)
    rbf.add_sculpted_pose(Float32(0.75), Float32(0.0), 0)
    rbf.finalize()

    var linkage = FourBarLinkageDriver(
        crank_node=4, coupler_node=5, rocker_node=6,
        ground_a=Vec3(0.0, 1.4, 0.0), ground_d=Vec3(0.0, 1.6, 0.2),
        len_crank_ab=Float32(0.35), len_coupler_bc=Float32(0.45), len_rocker_cd=Float32(0.35),
    )

    var rbf_out = Float32(0.0)
    var hits = 0
    for f in range(15):
        var place = Placement(Vec3(Float32(f) * 0.05, 0.0, 0.0), Vec3(1, 1, 1), Float32(0.0))
        skel.reset_to_rest(model)
        hits += solve_foot_plant_world(foot_solver, pw, skel, model, place, in_stance, Float32(1.0 / 60.0))
        var neck_w = place.to_world(skel.global_position(5))
        var head_w = place.to_world(skel.global_position(6))
        var q_aim = aim_rotation(neck_w, head_w, Vec3(4.0, 1.7, 4.0), Quat())
        skel.pose.r[6] = q_aim
        skel.update_matrices(model)
        _ = linkage.apply_to_skeleton(skel, model, Float32(0.2 + Float32(f) * 0.04))
        var eval_w = rbf.evaluate(Float32(f) * Float32(0.05), Float32(0.0))
        rbf_out = eval_w[0]

    ctx.assert_true(hits >= 15, "foot plant hits floor across all 15 walking frames")
    ctx.assert_true(rbf_out > Float32(0.5) and rbf_out <= Float32(1.05), "RbfPoseDriver activates corrective weight along pose sweep")
    ctx.assert_approx(_quat_norm_sq(skel.pose.r[6]), 1.0, 0.001, "aim + linkage + foot IK preserve normalized quaternions")


# 14. Aero6DofState airspeed drag decay inside PhysicsWorld
def test_t3_aero_drag_decay(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=0.0)
    var st = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 120.0))
    var cfg = Aero6DofConfig()
    var ctrl = Aero6DofControls()
    ctrl.throttle = Float32(0.0)
    ctrl.gravity = Float32(0.0)
    var a_slot = pw.add_aero_body(st, cfg, ctrl)

    var v_start = pw.aero_state(a_slot).linear_vel.z
    for _ in range(120):
        pw.step(Float32(1.0 / 60.0))
    var st_end = pw.aero_state(a_slot)
    var v_end = st_end.linear_vel.z

    ctx.assert_true(v_end < v_start - Float32(0.5), "zero-throttle Aero6DofState airspeed strictly decays under PhysicsWorld drag")
    ctx.assert_true(st_end.pos.z > Float32(100.0), "Aero6DofState advances forward across 120 frames")


# 15. VehicleChassis trail-braking forward weight transfer & yaw inside PhysicsWorld
def test_t3_trail_braking_weight_transfer(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(120.0, 0.5, 120.0), mass=0.0)
    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.85, 0.0)
    car.linear_vel = Vec3(0.0, 0.0, 18.0)
    var v_slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()

    pw.set_vehicle_controls(v_slot, Float32(0.0), Float32(0.0), Float32(0.0))
    for _ in range(6):
        pw.step(Float32(1.0 / 60.0))

    pw.set_vehicle_controls(v_slot, Float32(0.0), Float32(0.9), Float32(0.35))
    for _ in range(25):
        pw.step(Float32(1.0 / 60.0))

    var st = pw.vehicle_state(v_slot)
    var front_load = st.wheel_states[0].normal_load + st.wheel_states[1].normal_load
    var rear_load = st.wheel_states[2].normal_load + st.wheel_states[3].normal_load
    ctx.assert_true(st.linear_vel.z < Float32(17.5), "trail braking decelerates VehicleChassis in PhysicsWorld")
    ctx.assert_true(front_load > rear_load, "trail braking shifts normal load onto front axle")
    ctx.assert_true(abs(st.angular_vel.y) > Float32(0.02) or abs(st.linear_vel.x) > Float32(0.1), "trail braking induces real cornering yaw/lateral velocity")


# =========================================================================
# Aggregator for Tier 3 Physics & Animation
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T3_CrossFeature_PhysicsAnim")

    try: test_t3_broadphase_to_solver_candidates(ctx); ctx.record_pass("test_t3_broadphase_to_solver_candidates") except e: ctx.record_fail("test_t3_broadphase_to_solver_candidates", String(e))
    try: test_t3_ik_foot_placement_surface_normal(ctx); ctx.record_pass("test_t3_ik_foot_placement_surface_normal") except e: ctx.record_fail("test_t3_ik_foot_placement_surface_normal", String(e))
    try: test_t3_aim_head_tracking(ctx); ctx.record_pass("test_t3_aim_head_tracking") except e: ctx.record_fail("test_t3_aim_head_tracking", String(e))
    try: test_t3_pacejka_to_wheel_impulse(ctx); ctx.record_pass("test_t3_pacejka_to_wheel_impulse") except e: ctx.record_fail("test_t3_pacejka_to_wheel_impulse", String(e))
    try: test_t3_impact_velocity_to_audio_resonance(ctx); ctx.record_pass("test_t3_impact_velocity_to_audio_resonance") except e: ctx.record_fail("test_t3_impact_velocity_to_audio_resonance", String(e))
    try: test_t3_moving_body_doppler_audio(ctx); ctx.record_pass("test_t3_moving_body_doppler_audio") except e: ctx.record_fail("test_t3_moving_body_doppler_audio", String(e))
    try: test_t3_collision_filtering_ragdoll(ctx); ctx.record_pass("test_t3_collision_filtering_ragdoll") except e: ctx.record_fail("test_t3_collision_filtering_ragdoll", String(e))
    try: test_t3_rodrigues_joint_lever_arm(ctx); ctx.record_pass("test_t3_rodrigues_joint_lever_arm") except e: ctx.record_fail("test_t3_rodrigues_joint_lever_arm", String(e))
    try: test_t3_suspension_compression_load(ctx); ctx.record_pass("test_t3_suspension_compression_load") except e: ctx.record_fail("test_t3_suspension_compression_load", String(e))
    try: test_t3_replay_physics_snapshot(ctx); ctx.record_pass("test_t3_replay_physics_snapshot") except e: ctx.record_fail("test_t3_replay_physics_snapshot", String(e))
    try: test_t3_acceleration_filtering_svf(ctx); ctx.record_pass("test_t3_acceleration_filtering_svf") except e: ctx.record_fail("test_t3_acceleration_filtering_svf", String(e))
    try: test_t3_anim_pose_slerp(ctx); ctx.record_pass("test_t3_anim_pose_slerp") except e: ctx.record_fail("test_t3_anim_pose_slerp", String(e))
    try: test_t3_simultaneous_ik_and_aim(ctx); ctx.record_pass("test_t3_simultaneous_ik_and_aim") except e: ctx.record_fail("test_t3_simultaneous_ik_and_aim", String(e))
    try: test_t3_aero_drag_decay(ctx); ctx.record_pass("test_t3_aero_drag_decay") except e: ctx.record_fail("test_t3_aero_drag_decay", String(e))
    try: test_t3_trail_braking_weight_transfer(ctx); ctx.record_pass("test_t3_trail_braking_weight_transfer") except e: ctx.record_fail("test_t3_trail_braking_weight_transfer", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
