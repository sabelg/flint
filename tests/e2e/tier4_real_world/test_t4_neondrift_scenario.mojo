"""Tier 4 Real-World Application Scenarios: S1 NeonDrift 2099 Night Cyberpunk Racing.
Genuine multi-frame `EngineApp` slice uniting `WindowPlugin`, `TransformPlugin`, `InputPlugin`,
`PhysicsPlugin` (`VehicleChassis` + `PacejkaTire` + `TriggerVolume`), `AudioPlugin` (`AudioServer`),
`RenderPlugin` (`ScalableRenderer` + `MaterialAsset` + `PointLightComponent` + `CameraComponent`),
and `FlintReplayStream` + `World.snapshot_blob`.
"""
from std.collections import List
from std.ffi import external_call
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.replay_stream import FlintReplayStream
from flint.audio import AudioServer
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.core.math import Vec3
from flint.core.serial import BlobWriter
from flint.core.transform import Quat
from flint.input import ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin, InputStateResource
from flint.physics.plugin import (
    ColliderComponent,
    PhysicsPlugin,
    RigidBodyComponent,
    TriggerEventsResource,
    VehicleComponent,
)
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld, TRIGGER_ENTER
from flint.render.image_diff import compare_pixels
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import create_mock_pcm16_sine


def _pixel_variance(px: List[UInt32]) -> Int:
    if len(px) == 0:
        return 0
    var p0 = px[0]
    var diff = 0
    for i in range(1, len(px)):
        if px[i] != p0:
            diff += 1
    return diff


def test_t4_neondrift_full_race_slice(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="NeonDrift2099", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(WindowPlugin(headless=True, width=160, height=120, title="NeonDrift2099"))
    app.add_plugin(TransformPlugin())
    app.add_plugin(InputPlugin(enable_evdev=False))
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    # 1. InputMap actions for driving
    var key_w = 87
    var key_s = 83
    var key_d = 68
    ref imap = app.resources.get_ptr_mut[InputMap]().value()[unsafe_offset=0]
    _ = imap.register_action("throttle", ACTION_BOOL)
    _ = imap.register_action("brake", ACTION_BOOL)
    _ = imap.register_action("steer_right", ACTION_BOOL)
    var ictx = InputContext("neondrift", priority=10)
    ictx.bind_key("throttle", key_w)
    ictx.bind_key("brake", key_s)
    ictx.bind_key("steer_right", key_d)
    imap.add_context(ictx^)
    imap.push_context("neondrift")

    # 2. Register wet asphalt road & neon chassis materials/meshes
    ref ren = app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
    var asphalt = MaterialAsset(base_color=Vec3(0.08, 0.09, 0.12), roughness=0.18, metallic=0.05)
    asphalt.clearcoat = Float32(0.95)
    asphalt.clearcoat_roughness = Float32(0.08)
    var mat_asphalt = ren.register_material(asphalt)
    var mb_road = MeshBuilder()
    mb_road.add_beveled_box(Vec3(0.0, -0.25, 50.0), Vec3(12.0, 0.25, 70.0), 0.05)
    var mesh_road = ren.register_mesh(mb_road^.finish(), mat_asphalt)

    var neon_mat = MaterialAsset(base_color=Vec3(0.95, 0.15, 0.55), roughness=0.25, metallic=0.8)
    neon_mat.emissive = Vec3(1.5, 0.2, 0.8)
    var mat_car = ren.register_material(neon_mat)
    var mb_car = MeshBuilder()
    mb_car.add_beveled_box(Vec3(0.0, 0.05, 0.0), Vec3(0.9, 0.30, 2.0), 0.08)
    var mesh_car = ren.register_mesh(mb_car^.finish(), mat_car)

    # 3. Road slab, guardrail & checkpoint trigger volume
    var road_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](road_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](road_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](road_e, PrevGlobalTransform())
    _ = app.world.insert[RigidBodyComponent](road_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](road_e, ColliderComponent.box(Vec3(40.0, 0.5, 200.0)))
    _ = app.world.insert[MeshRendererComponent](road_e, MeshRendererComponent(mesh_road, mat_asphalt))

    var rail_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](rail_e, LocalTransform(Vec3(12.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](rail_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](rail_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](rail_e, ColliderComponent.box(Vec3(1.0, 1.5, 200.0)))

    var chk_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](chk_e, LocalTransform(Vec3(0.0, 1.0, 10.0)))
    _ = app.world.insert[GlobalTransform](chk_e, GlobalTransform())
    _ = app.world.insert[ColliderComponent](chk_e, ColliderComponent.box(Vec3(10.0, 2.0, 2.0), is_trigger=True))

    # 4. 4 neon PointLightComponents along corridor
    var light_zs = [Float32(10.0), Float32(20.0), Float32(30.0), Float32(40.0)]
    var light_xs = [Float32(-8.0), Float32(8.0), Float32(-8.0), Float32(8.0)]
    var light_cols = [
        Vec3(16.0, 2.0, 9.0),
        Vec3(2.0, 16.0, 16.0),
        Vec3(16.0, 13.0, 2.0),
        Vec3(9.0, 3.0, 16.0),
    ]
    for li in range(4):
        var le = app.world.spawn()
        _ = app.world.insert[LocalTransform](le, LocalTransform(Vec3(light_xs[li], 4.5, light_zs[li])))
        _ = app.world.insert[GlobalTransform](le, GlobalTransform())
        _ = app.world.insert[PointLightComponent](le, PointLightComponent(light_cols[li], 35.0, True))

    # 5. Player VehicleChassis + AudioEmitterComponent
    var engine_aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(
        create_mock_pcm16_sine(226.0, 4096, 48000), 1, 48000
    )
    var car = VehicleChassis(1420.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.85, 0.0)
    car.linear_vel = Vec3(0.0, 0.0, 0.0)
    var car_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](car_e, LocalTransform(Vec3(0.0, 0.85, 0.0)))
    _ = app.world.insert[GlobalTransform](car_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](car_e, PrevGlobalTransform())
    _ = app.world.insert[VehicleComponent](car_e, VehicleComponent(car^))
    _ = app.world.insert[MeshRendererComponent](car_e, MeshRendererComponent(mesh_car, mat_car))
    _ = app.world.insert[AudioEmitterComponent](
        car_e, AudioEmitterComponent(asset_id=engine_aid, asset_path="turbo.wav", looping=True, spatial=True, play_on_spawn=True)
    )

    # 6. Chase Camera + AudioListenerComponent
    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 3.2, -7.5)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=60.0, active=True, look_target_offset=Vec3(0.0, -1.5, 7.5)))
    _ = app.world.insert[AudioListenerComponent](cam_e, AudioListenerComponent(active=True, priority=1))

    # 7. Replay recording
    var w_init = BlobWriter()
    app.world.snapshot_blob(w_init)
    var replay = FlintReplayStream()
    replay.begin_recording("NeonDrift2099", w_init)

    var dt = Float32(1.0 / 60.0)
    var tev_cursor = 0
    var saw_checkpoint_trigger = False
    var speed_f119 = Float32(0.0)
    var speed_f179 = Float32(0.0)
    var px_f1 = List[UInt32]()
    var px_f179 = List[UInt32]()

    for frame in range(180):
        ref inp_state = app.resources.get_ptr_mut[InputStateResource]().value()[unsafe_offset=0].input
        if frame < 60:
            inp_state.keys[key_w] = True
            inp_state.keys[key_s] = False
            inp_state.keys[key_d] = False
        elif frame < 120:
            inp_state.keys[key_w] = True
            inp_state.keys[key_s] = False
            inp_state.keys[key_d] = True
        else:
            inp_state.keys[key_w] = False
            inp_state.keys[key_s] = True
            inp_state.keys[key_d] = True

        ref im_cur = app.resources.get_ptr[InputMap]().value()[unsafe_offset=0]
        ref vc = app.world.get_ptr_mut[VehicleComponent](car_e).value()[unsafe_offset=0]
        vc.throttle = Float32(1.0) if (frame == 0 or im_cur.is_down("throttle")) and frame < 120 else Float32(0.0)
        vc.brake = Float32(0.85) if frame >= 120 and im_cur.is_down("brake") else Float32(0.0)
        vc.steer = Float32(0.14) if (frame >= 60 and im_cur.is_down("steer_right")) else Float32(0.0)

        var car_pos = app.world.get[LocalTransform](car_e).translation
        app.world.get_ptr_mut[LocalTransform](cam_e).value()[unsafe_offset=0].translation = car_pos + Vec3(0.0, 2.8, -7.5)
        app.world.get_ptr_mut[CameraComponent](cam_e).value()[unsafe_offset=0].look_target_offset = Vec3(0.0, -2.0, 7.5)

        app.step_one_frame(dt)
        if frame == 5:
            for _ in range(15):
                if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() >= 1:
                    break
                _ = external_call["usleep", Int32](UInt32(2000))

        var ib = List[UInt8]()
        ib.append(UInt8(1 if frame < 120 else 2))
        replay.record_tick(dt, ib)
        if frame == 0 or frame == 90 or frame == 179:
            replay.add_camera_knot(frame, car_pos + Vec3(0.0, 2.8, -7.5), car_pos, Float32(60.0))

        var tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
        for ti in range(len(tevs)):
            if tevs[ti].kind == TRIGGER_ENTER:
                saw_checkpoint_trigger = True

        var slot = app.world.get_ptr[VehicleComponent](car_e).value()[unsafe_offset=0].vehicle_slot
        var vst = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].vehicle_state(slot)
        if frame == 119:
            speed_f119 = vst.linear_vel.length()
        elif frame == 179:
            speed_f179 = vst.linear_vel.length()

        if frame == 1:
            px_f1 = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
        elif frame == 179:
            px_f179 = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()

    var w_end = BlobWriter()
    app.world.snapshot_blob(w_end)
    _ = replay.finish_recording(w_end)

    var v_slot_final = app.world.get_ptr[VehicleComponent](car_e).value()[unsafe_offset=0].vehicle_slot
    var vst_final = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].vehicle_state(v_slot_final)
    ref rstate = app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
    var rframes = rstate.rendered_frames
    var rok = rstate.last_render_ok
    var rw_inst = app.resources.get_ptr[RenderWorld]().value()[unsafe_offset=0].instance_count()
    var rw_lights = len(app.resources.get_ptr[RenderWorld]().value()[unsafe_offset=0].cluster_lights)
    var audio_srv_ptr = app.resources.get_ptr[AudioServer]()
    var backend_before = audio_srv_ptr.value()[unsafe_offset=0].audio_backend_name()
    var active_voices = audio_srv_ptr.value()[unsafe_offset=0].active_logical_count()
    var emitter_valid = app.world.get_ptr[AudioEmitterComponent](car_e).value()[unsafe_offset=0].handle.is_valid()
    var spline_eval = replay.evaluate_camera_spline(Float32(90.0))

    app.shutdown()
    var backend_after = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].audio_backend_name()

    ctx.assert_true(vst_final.wheel_states[0].grounded and vst_final.wheel_states[2].grounded, "vehicle wheels grounded in PhysicsWorld")
    ctx.assert_true(vst_final.wheel_states[0].compression > Float32(0.0), "vehicle suspension compressed")
    ctx.assert_true(speed_f119 > Float32(5.0), "throttle accelerates vehicle from rest past 5 m/s")
    ctx.assert_true(vst_final.pos.z > Float32(10.0), "vehicle accelerates forward past checkpoint")
    ctx.assert_true(abs(vst_final.pos.x) > Float32(0.2), "vehicle corners laterally under steer input")
    ctx.assert_true(saw_checkpoint_trigger, "TriggerVolume detects vehicle crossing checkpoint")
    ctx.assert_true(speed_f179 < speed_f119 - Float32(4.0), "braking phase decelerates vehicle speed beyond coasting drag")
    ctx.assert_true(rframes >= 180 and rok, "RenderPlugin rendered all 180 frames")
    ctx.assert_true(rw_inst >= 2 and rw_lights == 4, "RenderWorld extracted road, car, and 4 neon lights")
    ctx.assert_true(_pixel_variance(px_f179) > 20, "rendered frame has non-uniform pixel content")
    ctx.assert_true(compare_pixels(px_f1, px_f179, 160, 120, 4).differing > 10, "framebuffer evolves between frame 1 and frame 180")
    ctx.assert_true(emitter_valid and active_voices >= 1, "AudioPlugin activated spatial engine voice")
    ctx.assert_equal(backend_before, "headless", "AudioServer started in headless backend")
    ctx.assert_equal(backend_after, "none", "AudioServer shut down cleanly")
    ctx.assert_equal(replay.tick_count(), 180, "FlintReplayStream recorded 180 ticks")
    ctx.assert_true(spline_eval[0], "camera director spline evaluates midpoint")


# =========================================================================
# Aggregator for Tier 4 NeonDrift Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_NeonDrift")
    try: test_t4_neondrift_full_race_slice(ctx); ctx.record_pass("test_t4_neondrift_full_race_slice") except e: ctx.record_fail("test_t4_neondrift_full_race_slice", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
