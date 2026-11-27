"""Final-pose contacts, fixed root motion and recovery snapshot regressions."""
from std.collections import List
from std.math import abs
from std.testing import assert_true
from e2e.tier3_cross_feature.test_t3_physics_anim import _make_t3_biped_model
from flint.anim.flanim_ecs import (
    RAGDOLL_SM_GETUP_PRONE, RAGDOLL_SM_GETUP_SUPINE, RagdollStateMachine,
)
from flint.anim.graph import ROOT_EXTRACT, AnimGraphDef, StateDef
from flint.anim.plugin import (
    AnimAssetCache, AnimEventsResource, AnimPlugin, AnimatorComponent, RagdollComponent, SkeletonComponent,
)
from flint.anim.pose import Pose, sample_pose
from flint.anim.procedural import Placement
from flint.anim.sync import ClipEventInfo, ClipInfo
from flint.app import AppConfig, EngineApp
from flint.assets.glb_native import load_native_glb
from flint.core.ik import angle_axis
from flint.core.math import Vec3
from flint.ecs.world import Entity
from flint.physics.plugin import CharacterControllerComponent, PhysicsPlugin
from flint.physics.world import PhysicsWorld
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin


def _actor(
    mut app: EngineApp,
    attached: Bool,
    walking: Bool,
    pelvis_offset: Float32 = 0.0,
    performer: Bool = False,
) raises -> Entity:
    var model = (
        load_native_glb("examples/character_course/assets/performer.glb")
        if performer else _make_t3_biped_model()
    )
    model.nodes[0].name = "Root"
    model.nodes[0].base_translation.y += pelvis_offset
    model.nodes[0].translation.y += pelvis_offset
    var sm = RagdollStateMachine()
    sm.ragdoll.bind_skeleton(model)
    var pelvis = sm.ragdoll.add_limb_from_model(model, 0, -1, 20.0, 0.16, 0.3)
    if performer:
        sm.chest_limb_index = sm.ragdoll.add_limb_from_model(
            model, model.find_node("Chest"), pelvis, 10.0, 0.14, 0.25
        )
        for i in range(len(model.animations)):
            if model.animations[i].name == "GetupSupine":
                sm.supine_getup_clip = i
            elif model.animations[i].name == "GetupProne":
                sm.prone_getup_clip = i
        assert_true(sm.supine_getup_clip >= 0 and sm.prone_getup_clip >= 0)
        assert_true(sm.prone_getup_clip != sm.supine_getup_clip)
    var state = StateDef()
    state.name = "Run" if walking else "Idle"
    state.clip = 1 if walking else 0
    if walking:
        state.root_motion = ROOT_EXTRACT
    var graph = AnimGraphDef()
    graph.states.append(state^)
    var clips = List[ClipInfo]()
    for anim in model.animations:
        var clip = ClipInfo(anim.name, anim.duration, True, Float32(0.0))
        clip.events.append(ClipEventInfo("foot", "left", Float32(0.5)))
        clips.append(clip^)
    ref cache = app.resources.get_ptr_mut[AnimAssetCache]().value()[]
    cache.register_model(11, model.copy())
    cache.register_graph(22, graph^, clips^)
    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3()))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[SkeletonComponent](e, SkeletonComponent(11))
    var anim = AnimatorComponent(22, apply_root_motion=walking, enable_foot_plant=not walking)
    if performer:
        anim.foot_plant.add_leg(model.find_node("Hip.L"), model.find_node("Knee.L"), model.find_node("Foot.L"), 0.08)
    else:
        anim.foot_plant.add_leg(1, 2, 3, Float32(0.08))
    anim.foot_in_stance.append(True)
    _ = app.world.insert[AnimatorComponent](e, anim^)
    if attached:
        _ = app.world.insert[RagdollComponent](e, RagdollComponent(sm^))
    return e


def _deck(attached: Bool, render_hz: Int) raises:
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin())
    app.add_plugin(AnimPlugin())
    app.startup()
    ref world = app.resources.get_ptr_mut[PhysicsWorld]().value()[]
    var deck = world.add_kinematic_box(Vec3(0, 0.125, 0), Vec3(2, 0.125, 2))
    var e = _actor(app, attached, False, pelvis_offset=-0.98)
    var spawn = Vec3(0, 1.15, 0)
    app.world.get_ptr_mut[LocalTransform](e).value()[].translation = spawn
    _ = app.world.insert[CharacterControllerComponent](e, CharacterControllerComponent(spawn))
    # Acquire the support before the first deck motion at every display rate.
    for _ in range(12):
        app.step_one_frame(Float32(1.0 / 60.0))
    var anchor = Vec3()
    for i in range(render_hz + 1):
        var t = Float32(i) / Float32(render_hz)
        var q = angle_axis(t * Float32(0.5235987756), Vec3(0, 1, 0))
        world.bodies[deck].pos = Vec3(t, 0.125, 0)
        world.bodies[deck].rot = q
        world.rebuild_broadphase()
        app.step_one_frame(Float32(1.0) / Float32(render_hz))
        ref lt = app.world.get_ptr[LocalTransform](e).value()[]
        ref sk = app.world.get_ptr[SkeletonComponent](e).value()[].skeleton
        var foot = lt.translation + lt.rotation.rotate(sk.global_position(3))
        assert_true(abs(foot.y - Float32(0.33)) < Float32(0.002), "final ankle on raised deck")
        var local = q.conjugate().rotate(foot - world.bodies[deck].pos)
        if i == 0:
            anchor = local
        assert_true((local - anchor).length() < Float32(0.02), "foot slipped in deck space")
    var rider = app.world.get[LocalTransform](e)
    assert_true(abs(rider.translation.x - Float32(1.0)) < Float32(0.02), "deck carries rider")
    var front = rider.rotation.rotate(Vec3(0, 0, 1))
    assert_true(abs(front.x - Float32(0.5)) < Float32(0.02), "rider inherits deck heading")
    app.shutdown()


def _wall(render_hz: Int) raises -> Float32:
    var app = EngineApp(AppConfig(headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    # Plugin order must not move sampling past the controller sweep.
    app.add_plugin(AnimPlugin())
    app.add_plugin(PhysicsPlugin())
    app.startup()
    ref world = app.resources.get_ptr_mut[PhysicsWorld]().value()[]
    _ = world.add_box(Vec3(0, -0.5, 0), Vec3(8, 0.5, 8), mass=0)
    _ = world.add_box(Vec3(0, 1, 1.5), Vec3(8, 1, 0.1), mass=0)
    world.rebuild_broadphase()
    var e = _actor(app, False, True)
    var spawn = Vec3(0, 0.9, 0)
    app.world.get_ptr_mut[LocalTransform](e).value()[].translation = spawn
    _ = app.world.insert[CharacterControllerComponent](e, CharacterControllerComponent(spawn))
    for frame in range(render_hz * 2):
        app.step_one_frame(Float32(1.0) / Float32(render_hz))
        ref kcc = app.world.get_ptr[CharacterControllerComponent](e).value()[]
        var z = app.world.get[LocalTransform](e).translation.z
        assert_true(abs(z - kcc.controller.pos.z) < Float32(0.00001))
        assert_true(z < Float32(1.07), "capsule must remain ahead of wall")
    var end = app.world.get[LocalTransform](e).translation.z
    ref anim = app.world.get_ptr[AnimatorComponent](e).value()[]
    assert_true(end > Float32(0.95), "root motion must actually move the controller")
    assert_true(anim.animator.root_delta.z > Float32(0.01))
    assert_true(abs(anim.actual_displacement.z) < Float32(0.001), "blocked root travel feedback")
    app.shutdown()
    return end


def _recovery_actor(mut app: EngineApp) raises -> Entity:
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin())
    app.add_plugin(AnimPlugin())
    app.startup()
    ref world = app.resources.get_ptr_mut[PhysicsWorld]().value()[]
    _ = world.add_box(Vec3(0, -0.5, 0), Vec3(8, 0.5, 8), mass=0)
    world.rebuild_broadphase()
    _ = world.add_kinematic_box(Vec3(0, 0.45, 0), Vec3(3, 0.15, 3))
    world.rebuild_broadphase()
    var e = _actor(app, True, False, performer=True)
    app.world.get_ptr_mut[LocalTransform](e).value()[].translation.y = 1.50
    return e


def _recovery(manual: Bool, supine: Bool) raises -> Vec3:
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    var e = _recovery_actor(app)
    var dt = Float32(1.0 / 60.0)
    for _ in range(5):
        app.step_one_frame(dt)
    ref sm = app.world.get_ptr_mut[RagdollComponent](e).value()[].state_machine
    ref sk = app.world.get_ptr_mut[SkeletonComponent](e).value()[].skeleton
    ref model = app.resources.get_ptr[AnimAssetCache]().value()[].models[0]
    var fallback = sk.pose.copy()
    var place = Placement()
    place.position = app.world.get[LocalTransform](e).translation
    sm.authored_floor_y = -0.90
    # Rotate the whole posed skeleton around its pelvis before transferring to physics.
    var pitch = Float32(-1.57079632679) if supine else Float32(1.57079632679)
    sk.pose.r[0] = angle_axis(pitch, Vec3(1, 0, 0))
    sk.update_matrices(model)
    sm.enter_limp(sk, place, model, dt)
    assert_true(sm.is_supine() == supine, "chest orientation must select the expected recovery")
    var expected_clip = sm.supine_getup_clip if supine else sm.prone_getup_clip
    var expected_state = RAGDOLL_SM_GETUP_SUPINE if supine else RAGDOLL_SM_GETUP_PRONE
    var wrong_clip = sm.prone_getup_clip if supine else sm.supine_getup_clip
    ref world = app.resources.get_ptr_mut[PhysicsWorld]().value()[]
    sm.ragdoll.still_seconds = 1.0
    sm.min_limp_time = 0.0
    _ = sm.attach_to_world(world)
    world.rebuild_broadphase()
    if manual:
        sm.begin_getup_world(world, sk, place, model, fallback)
    else:
        sm.update_world(world, sk, place, model, fallback, dt, auto_recover=True, step_world=False)
    assert_true(sm.recovery_count == 1 and sm.state == expected_state)
    assert_true(abs(place.position.y - Float32(1.5)) < Float32(0.0001),
                "capsule origin must settle 0.9m above the elevated deck")
    assert_true(sm.last_pelvis_snap_m < Float32(0.001), "rebase must retain the world pelvis")
    app.world.get_ptr_mut[LocalTransform](e).value()[].translation = place.position
    for _ in range(6):
        app.step_one_frame(dt)
    assert_true(not sm.inertial.is_settled)
    var expected = Pose()
    var opposite = Pose()
    sample_pose(model, expected_clip, sm.getup_time + dt, expected)
    sample_pose(model, wrong_clip, sm.getup_time + dt, opposite)
    var different_channel = False
    for b in range(sk.node_count):
        var actual = sm.scratch_ahead.r[b]
        var want = expected.r[b]
        var other_rot = opposite.r[b]
        assert_true(actual.x == want.x and actual.y == want.y and actual.z == want.z and actual.w == want.w,
                    "recovery must sample the classified clip")
        different_channel = different_channel or abs(want.x - other_rot.x) > Float32(0.02)
    assert_true(different_channel, "the two performances must contain distinguishable rotations")
    var hand_before_restore = sk.global_position(model.find_node("Hand.L"))
    var bytes = app.world.snapshot_blob()

    var other = EngineApp(AppConfig(headless=True, worker_threads=0))
    _ = _recovery_actor(other)
    other.world.restore_blob(bytes)
    var events = 0
    var stance_after_getup = False
    # Include the longer prone clip and at least half a second of returned locomotion.
    for _ in range(150):
        app.step_one_frame(dt)
        other.step_one_frame(dt)
        ref restored = other.world.get_ptr[RagdollComponent](e).value()[].state_machine
        assert_true(sm.state == restored.state)
        assert_true(sm.authored_floor_y == restored.authored_floor_y)
        ref restored_sk = other.world.get_ptr[SkeletonComponent](e).value()[].skeleton
        for b in range(sk.node_count):
            for k in range(16):
                assert_true(sk.world_matrices[b].values[k] == restored_sk.world_matrices[b].values[k],
                            "world-space bone matrix diverged after restore")
        ref leg = app.world.get_ptr[AnimatorComponent](e).value()[].foot_plant.legs[0]
        ref restored_leg = other.world.get_ptr[AnimatorComponent](e).value()[].foot_plant.legs[0]
        assert_true(leg.locked == restored_leg.locked)
        assert_true((leg.locked_world_pos - restored_leg.locked_world_pos).length_squared() == Float32(0.0))
        stance_after_getup = stance_after_getup or leg.locked
        ref emitted = app.resources.get_ptr[AnimEventsResource]().value()[].events.older
        ref replayed = other.resources.get_ptr[AnimEventsResource]().value()[].events.older
        assert_true(len(emitted) == len(replayed), "event tick diverged after restore")
        for i in range(len(emitted)):
            assert_true(emitted[i].event.kind == replayed[i].event.kind)
            assert_true(emitted[i].event.arg == replayed[i].event.arg)
            assert_true(emitted[i].event.clip == replayed[i].event.clip)
            events += 1
    assert_true(events > 0 and stance_after_getup)
    app.shutdown()
    other.shutdown()
    return hand_before_restore


def run_character_motion_integration() raises:
    for hz in [30, 60, 120]:
        _deck(False, hz)
        _deck(True, hz)
    var endpoint = _wall(60)
    assert_true(abs(_wall(30) - endpoint) < Float32(0.01))
    assert_true(abs(_wall(120) - endpoint) < Float32(0.01))
    for manual in [True, False]:
        var face_up_hand = _recovery(manual, True)
        var face_down_hand = _recovery(manual, False)
        assert_true((face_up_hand - face_down_hand).length() > Float32(0.05),
                    "the displayed pre-restore get-up poses must be visibly distinct")


def main() raises:
    run_character_motion_integration()
    print("PASS: final planted poses, moving supports, fixed root motion and recovery snapshots")
