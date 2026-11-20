"""Tier 4 Real-World Application Scenarios: S2 Rollercoaster Physics & Track Simulation.
Genuine multi-frame `EngineApp` slice coupling a 3-cart train via `JointComponent` (`JOINT_REVOLUTE`)
descending a `Bvh8MeshCollider` coaster ramp under `PhysicsPlugin` gravity, avoiding an offset
pylon, tracked by `AudioPlugin` (`AudioServer`) spatial audio and `RenderPlugin` (`ScalableRenderer`).
"""
from std.collections import List
from std.ffi import external_call
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin
from flint.assets.mesh_builder import MeshBuilder
from flint.audio import AudioServer
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.ecs import Entity
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.events import EVENT_BEGIN, EVENT_HIT
from flint.physics.joints import JOINT_REVOLUTE
from flint.physics.plugin import (
    ColliderComponent,
    JointComponent,
    PhysicsEventsResource,
    PhysicsPlugin,
    RigidBodyComponent,
)
from flint.physics.world import PhysicsWorld
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
)
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RenderWorld, ScalableRenderer
from flint.scene import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import create_mock_pcm16_sine


def _build_coaster_track_bvh() -> Bvh8MeshCollider:
    # Smooth multi-segment descending coaster deck with gentle valley trough:
    # z=0..24 drops from y=18 -> y=4, z=24..50 levels through y=2 trough -> y=5
    var z_pts = [Float32(-4.0), Float32(0.0), Float32(12.0), Float32(24.0), Float32(36.0), Float32(48.0), Float32(64.0)]
    var y_pts = [Float32(20.0), Float32(18.0), Float32(11.0), Float32(4.0), Float32(2.0), Float32(2.5), Float32(5.5)]
    var verts = List[Vec3]()
    var idxs = List[Int32]()
    var half_w = Float32(3.5)
    for i in range(len(z_pts)):
        verts.append(Vec3(-half_w, y_pts[i], z_pts[i]))
        verts.append(Vec3(half_w, y_pts[i], z_pts[i]))
    for i in range(len(z_pts) - 1):
        var a = Int32(i * 2)
        var b = Int32(i * 2 + 1)
        var c = Int32((i + 1) * 2)
        var d = Int32((i + 1) * 2 + 1)
        # CCW upward normals (+Y)
        idxs.append(a); idxs.append(c); idxs.append(b)
        idxs.append(b); idxs.append(c); idxs.append(d)
    return Bvh8MeshCollider.build(verts, idxs)


def _pixel_variance(px: List[UInt32]) -> Int:
    if len(px) == 0:
        return 0
    var p0 = px[0]
    var diff = 0
    for i in range(1, len(px)):
        if px[i] != p0:
            diff += 1
    return diff


def test_t4_rollercoaster_full_run(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="Rollercoaster", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(WindowPlugin(headless=True, width=160, height=120, title="Rollercoaster"))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    # 1. Register coaster track BVH8 static mesh into PhysicsWorld
    ref pw = app.resources.get_ptr_mut[PhysicsWorld]().value()[unsafe_offset=0]
    _ = pw.add_static_mesh(_build_coaster_track_bvh(), friction=0.01, restitution=0.0)
    pw.rebuild_broadphase()

    # 2. Render assets
    ref ren = app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
    var mat_cart = ren.register_material(MaterialAsset(base_color=Vec3(0.9, 0.25, 0.15), roughness=0.3, metallic=0.6))
    var mb_cart = MeshBuilder()
    mb_cart.add_beveled_box(Vec3(-0.7, -0.4, -0.9), Vec3(0.7, 0.4, 0.9), 0.08)
    var mesh_cart = ren.register_mesh(mb_cart^.finish(), mat_cart)

    var mat_pylon = ren.register_material(MaterialAsset(base_color=Vec3(0.35, 0.45, 0.6), roughness=0.6, metallic=0.2))
    var mb_pylon = MeshBuilder()
    mb_pylon.add_beveled_box(Vec3(-0.8, -6.0, -0.8), Vec3(0.8, 6.0, 0.8), 0.05)
    var mesh_pylon = ren.register_mesh(mb_pylon^.finish(), mat_pylon)

    # 3. Structural support pylon offset at x = 6.0
    var pylon_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](pylon_e, LocalTransform(Vec3(6.0, 6.0, 24.0)))
    _ = app.world.insert[GlobalTransform](pylon_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](pylon_e, PrevGlobalTransform())
    _ = app.world.insert[RigidBodyComponent](pylon_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](pylon_e, ColliderComponent.box(Vec3(0.8, 6.0, 0.8)))
    _ = app.world.insert[MeshRendererComponent](pylon_e, MeshRendererComponent(mesh_pylon, mat_pylon))

    # 4. Audio asset
    var rattle_aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(
        create_mock_pcm16_sine(180.0, 4096, 48000), 1, 48000
    )

    # 5. Spawn 3 dynamic coaster carts on the slope, coupled end-to-end by JointComponents
    var start_positions = [Vec3(0.0, 15.8, 4.6), Vec3(0.0, 17.0, 2.4), Vec3(0.0, 18.2, 0.2)]
    var cart_es = List[Entity]()
    for ci in range(3):
        var ce = app.world.spawn()
        _ = app.world.insert[LocalTransform](ce, LocalTransform(start_positions[ci]))
        _ = app.world.insert[GlobalTransform](ce, GlobalTransform())
        _ = app.world.insert[PrevGlobalTransform](ce, PrevGlobalTransform())
        var rb = RigidBodyComponent.dynamic(450.0)
        rb.friction = Float32(0.01)
        rb.restitution = Float32(0.0)
        rb.linear_damping = Float32(0.0)
        rb.angular_damping = Float32(0.05)
        rb.linear_vel = Vec3(0.0, 0.0, 0.0) if ci == 2 else Vec3(0.0, -2.0, 4.5)
        _ = app.world.insert[RigidBodyComponent](ce, rb)
        _ = app.world.insert[ColliderComponent](ce, ColliderComponent.box(Vec3(0.7, 0.4, 0.9)))
        _ = app.world.insert[MeshRendererComponent](ce, MeshRendererComponent(mesh_cart, mat_cart))
        if ci == 0:
            _ = app.world.insert[AudioEmitterComponent](
                ce, AudioEmitterComponent(asset_id=rattle_aid, asset_path="rattle.wav", looping=True, spatial=True, play_on_spawn=True)
            )
        cart_es.append(ce)

    var hitch_01 = (start_positions[0] + start_positions[1]) * Float32(0.5)
    var hitch_12 = (start_positions[1] + start_positions[2]) * Float32(0.5)
    _ = app.world.insert[JointComponent](
        cart_es[0],
        JointComponent(
            joint_kind=JOINT_REVOLUTE,
            entity_b=cart_es[1],
            anchor_a=hitch_01,
            anchor_b=hitch_01,
            axis=Vec3(1.0, 0.0, 0.0),
        ),
    )
    _ = app.world.insert[JointComponent](
        cart_es[1],
        JointComponent(
            joint_kind=JOINT_REVOLUTE,
            entity_b=cart_es[2],
            anchor_a=hitch_12,
            anchor_b=hitch_12,
            axis=Vec3(1.0, 0.0, 0.0),
        ),
    )

    # 6. Spectator AudioListener + Light + Follow Camera
    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(10.0, 8.0, 16.0)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=60.0, active=True))
    _ = app.world.insert[AudioListenerComponent](cam_e, AudioListenerComponent(active=True, priority=1))

    var sun_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](sun_e, LocalTransform(Vec3(-6.0, 22.0, 15.0)))
    _ = app.world.insert[GlobalTransform](sun_e, GlobalTransform())
    _ = app.world.insert[PointLightComponent](sun_e, PointLightComponent(Vec3(40.0, 38.0, 34.0), 80.0, True))

    var dt = Float32(1.0 / 60.0)
    var peak_speed = Float32(0.0)
    var pylon_hit = False
    var pev_cursor = 0

    for frame in range(120):
        var lead_pos = app.world.get[LocalTransform](cart_es[0]).translation
        var cam_pos = lead_pos + Vec3(8.0, 3.5, -6.0)
        app.world.get_ptr_mut[LocalTransform](cam_e).value()[unsafe_offset=0].translation = cam_pos
        app.world.get_ptr_mut[CameraComponent](cam_e).value()[unsafe_offset=0].look_target_offset = lead_pos - cam_pos

        app.step_one_frame(dt)
        if frame == 5:
            for _ in range(15):
                if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() >= 1:
                    break
                _ = external_call["usleep", Int32](UInt32(2000))

        var slot0 = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].find_slot_by_id(
            app.world.get_ptr[RigidBodyComponent](cart_es[0]).value()[unsafe_offset=0].body_id
        )
        if slot0 >= 0:
            var spd = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies[slot0].linear_vel.length()
            if spd > peak_speed:
                peak_speed = spd

        var pylon_bid = app.world.get_ptr[RigidBodyComponent](pylon_e).value()[unsafe_offset=0].body_id
        var pevs = app.resources.get_ptr[PhysicsEventsResource]().value()[unsafe_offset=0].events.read_since(pev_cursor)
        for ei in range(len(pevs)):
            if (pevs[ei].kind == EVENT_HIT or pevs[ei].kind == EVENT_BEGIN) and (pevs[ei].a == pylon_bid or pevs[ei].b == pylon_bid):
                pylon_hit = True

    var p0_end = app.world.get[GlobalTransform](cart_es[0]).translation
    var p2_end = app.world.get[GlobalTransform](cart_es[2]).translation
    var train_span = (p0_end - p2_end).length()
    var slot2 = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].find_slot_by_id(
        app.world.get_ptr[RigidBodyComponent](cart_es[2]).value()[unsafe_offset=0].body_id
    )
    var cart2_speed = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies[slot2].linear_vel.length()

    ref rstate = app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
    var rframes = rstate.rendered_frames
    var rok = rstate.last_render_ok
    var rw_inst = app.resources.get_ptr[RenderWorld]().value()[unsafe_offset=0].instance_count()
    var px = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var emitter_valid = app.world.get_ptr[AudioEmitterComponent](cart_es[0]).value()[unsafe_offset=0].handle.is_valid()
    var active_voices = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count()
    app.shutdown()

    ctx.assert_true(p0_end.y < start_positions[0].y - Float32(3.0), "lead cart descends coaster drop")
    ctx.assert_true(p0_end.z > start_positions[0].z + Float32(10.0), "lead cart advances along coaster track")
    ctx.assert_true(peak_speed > Float32(10.0), "coaster accelerates past 10 m/s down drop")
    ctx.assert_true(train_span > Float32(3.0) and train_span < Float32(8.0), "JointComponents keep 3-cart train coupled")
    ctx.assert_true(cart2_speed > Float32(5.0), "trailing cart acquires forward velocity through joints")
    ctx.assert_false(pylon_hit, "train clears offset structural pylon with zero collisions")
    ctx.assert_true(rframes >= 120 and rok, "RenderPlugin rendered all 120 frames")
    ctx.assert_true(rw_inst >= 4, "RenderWorld extracted 3 carts + pylon")
    ctx.assert_true(_pixel_variance(px) > 20, "rendered coaster frame has non-uniform pixels")
    ctx.assert_true(emitter_valid and active_voices >= 1, "AudioPlugin activated spatial coaster rattle voice")


# =========================================================================
# Aggregator for Tier 4 Rollercoaster Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_Rollercoaster")
    try: test_t4_rollercoaster_full_run(ctx); ctx.record_pass("test_t4_rollercoaster_full_run") except e: ctx.record_fail("test_t4_rollercoaster_full_run", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
