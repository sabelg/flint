"""Tier 4 Real-World Application Scenarios: S5 Character Combat & Dynamic Ragdoll.
Genuine 8-plugin `EngineApp` slice uniting `WindowPlugin`, `TransformPlugin`, `InputPlugin`,
`PhysicsPlugin`, `AnimPlugin` (`SkeletonComponent`, `AnimatorComponent` + `FootPlantSolver`,
`RagdollComponent` + `RagdollStateMachine`, `BoneSocketComponent`), `AudioPlugin` (`AudioServer`),
`RenderPlugin` (`ScalableRenderer`), and `NetPlugin` (`LagCompHistory` temporal rewind raycast).
"""
from std.collections import List
from std.ffi import external_call
from flint.anim.flanim_ecs import (
    BoneSocketAttachment,
    RAGDOLL_SM_ANIMATED,
    RAGDOLL_SM_LIMP,
    RagdollStateMachine,
)
from flint.anim.graph import AnimGraphDef, StateDef
from flint.anim.plugin import (
    AnimAssetCache,
    AnimPlugin,
    AnimatorComponent,
    BoneSocketComponent,
    RagdollComponent,
    SkeletonComponent,
)
from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.anim.sync import ClipInfo
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeAnimation, NativeChannel, NativeModel, NativeNode
from flint.audio import AudioServer
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.input.plugin import InputPlugin
from flint.net.plugin import (
    NetPlugin,
    NetReplicationResource,
    NetworkIdentityComponent,
    ReplicationComponent,
)
from flint.physics.character import CharacterControllerConfig
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
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
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, ScalableRenderer
from flint.scene import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import create_mock_pcm16_sine


def _make_combat_biped_model() -> NativeModel:
    var m = NativeModel()
    # 0: Pelvis, 1: Chest, 2: Hand.R, 3: Thigh.L, 4: Shin.L, 5: Foot.L
    var names = [
        String("Pelvis"),
        String("Chest"),
        String("Hand.R"),
        String("Thigh.L"),
        String("Shin.L"),
        String("Foot.L"),
    ]
    var parents = [-1, 0, 1, 0, 3, 4]
    var offsets = [
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.42, 0.0),
        Vec3(0.35, 0.05, 0.25),
        Vec3(-0.15, -0.05, 0.0),
        Vec3(0.0, -0.45, 0.0),
        Vec3(0.0, -0.42, 0.0),
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

    var anim = NativeAnimation()
    anim.name = "Idle"
    anim.duration = Float32(1.0)
    var ch = NativeChannel()
    ch.node = 1
    ch.path = 0
    ch.times.append(Float32(0.0)); ch.times.append(Float32(1.0))
    ch.values.append(Float32(0.0)); ch.values.append(Float32(0.42)); ch.values.append(Float32(0.0))
    ch.values.append(Float32(0.0)); ch.values.append(Float32(0.45)); ch.values.append(Float32(0.0))
    anim.channels.append(ch^)
    m.animations.append(anim^)
    return m^


def _pixel_variance(px: List[UInt32]) -> Int:
    if len(px) == 0:
        return 0
    var p0 = px[0]
    var diff = 0
    for i in range(1, len(px)):
        if px[i] != p0:
            diff += 1
    return diff


def test_t4_character_combat_full_sequence(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="CharacterCombat", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(WindowPlugin(headless=True, width=160, height=120, title="CharacterCombat"))
    app.add_plugin(TransformPlugin())
    app.add_plugin(InputPlugin(enable_evdev=False))
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.add_plugin(NetPlugin(max_entities=64, max_clients=4))
    app.startup()

    # 1. Floor + raised stone slab for foot-plant & ragdoll impact
    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(30.0, 0.5, 30.0)))

    var slab_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](slab_e, LocalTransform(Vec3(0.8, 0.05, 0.0)))
    _ = app.world.insert[GlobalTransform](slab_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](slab_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](slab_e, ColliderComponent.box(Vec3(1.2, 0.05, 1.5)))

    # 2. Register biped model + Idle graph in AnimAssetCache
    var biped = _make_combat_biped_model()
    var graph = AnimGraphDef()
    graph.name = "CombatIdle"
    var st = StateDef()
    st.name = "Idle"
    st.clip = 0
    graph.states.append(st^)
    var clips = List[ClipInfo]()
    clips.append(ClipInfo("Idle", Float32(1.0), True, Float32(0.0)))

    var model_id = UInt64(301)
    var graph_id = UInt64(401)
    ref acache = app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0]
    acache.register_model(model_id, biped^)
    acache.register_graph(graph_id, graph^, clips^)

    # 3. Render assets
    ref ren = app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
    var mat_char = ren.register_material(MaterialAsset(base_color=Vec3(0.25, 0.65, 0.9), roughness=0.4))
    var mat_wpn = ren.register_material(MaterialAsset(base_color=Vec3(0.9, 0.8, 0.25), roughness=0.25, metallic=0.85))
    var mb_char = MeshBuilder()
    mb_char.add_beveled_box(Vec3(-0.35, -0.7, -0.25), Vec3(0.35, 0.8, 0.25), 0.05)
    var mesh_char = ren.register_mesh(mb_char^.finish(), mat_char)
    var mb_wpn = MeshBuilder()
    mb_wpn.add_beveled_box(Vec3(-0.06, -0.06, -0.45), Vec3(0.06, 0.06, 0.45), 0.02)
    var mesh_wpn = ren.register_mesh(mb_wpn^.finish(), mat_wpn)

    # 4. Combatant entity (Skeleton + Animator + FootPlant + KCC + Ragdoll + Net LagComp)
    var char_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](char_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](char_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](char_e, PrevGlobalTransform())
    _ = app.world.insert[SkeletonComponent](char_e, SkeletonComponent(model_id))
    var anim_c = AnimatorComponent(graph_id, enable_foot_plant=True)
    anim_c.foot_plant.add_leg(3, 4, 5, ankle_height=Float32(0.08))
    anim_c.foot_in_stance.append(True)
    _ = app.world.insert[AnimatorComponent](char_e, anim_c^)
    _ = app.world.insert[CharacterControllerComponent](
        char_e, CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3(1.5, 0.0, 0.0))
    )

    ref model_ref = app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0]
    var rag = ArticulatedRagdoll()
    rag.bind_skeleton(model_ref)
    var l_pelvis = rag.add_limb_from_model(model_ref, 0, -1, 20.0, 0.16, 0.30)
    _ = rag.add_limb_from_model(model_ref, 1, l_pelvis, 18.0, 0.18, 0.40)
    var rsm = RagdollStateMachine()
    rsm.ragdoll = rag^
    rsm.state = RAGDOLL_SM_ANIMATED
    _ = app.world.insert[RagdollComponent](char_e, RagdollComponent(rsm^, auto_recover=True, pelvis_syncs_transform=True))
    _ = app.world.insert[MeshRendererComponent](char_e, MeshRendererComponent(mesh_char, mat_char))
    _ = app.world.insert[NetworkIdentityComponent](char_e, NetworkIdentityComponent(net_id=1))
    _ = app.world.insert[ReplicationComponent](char_e, ReplicationComponent(hitbox_radius=0.8, record_lag_comp=True))

    # 5. Weapon BoneSocket attached to Hand.R (node 2) with kinematic body
    ref pw = app.resources.get_ptr_mut[PhysicsWorld]().value()[unsafe_offset=0]
    var kin_slot = pw.add_kinematic_box(Vec3(0.35, 1.45, 0.25), Vec3(0.08, 0.08, 0.45))
    var kin_bid = pw.body_id(kin_slot)
    var wpn_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](wpn_e, LocalTransform(Vec3(0.35, 1.45, 0.25)))
    _ = app.world.insert[GlobalTransform](wpn_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](wpn_e, PrevGlobalTransform())
    _ = app.world.insert[BoneSocketComponent](
        wpn_e,
        BoneSocketComponent(char_e, BoneSocketAttachment("WeaponSocket", 2, -1, Vec3(0.0, 0.0, 0.2), Quat(), Vec3(1, 1, 1), kin_bid)),
    )
    _ = app.world.insert[MeshRendererComponent](wpn_e, MeshRendererComponent(mesh_wpn, mat_wpn))

    # 6. Camera + AudioListener + Impact AudioEmitter + Light
    var impact_aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(
        create_mock_pcm16_sine(520.0, 4096, 48000), 1, 48000
    )
    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.5, 1.8, 4.5)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=55.0, active=True))
    _ = app.world.insert[AudioListenerComponent](cam_e, AudioListenerComponent(active=True, priority=1))

    var sfx_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](sfx_e, LocalTransform(Vec3(0.4, 1.4, 0.0)))
    _ = app.world.insert[GlobalTransform](sfx_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        sfx_e, AudioEmitterComponent(asset_id=impact_aid, asset_path="clang.wav", looping=True, spatial=True, play_on_spawn=False)
    )

    var lgt_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(2.0, 4.0, 3.0)))
    _ = app.world.insert[GlobalTransform](lgt_e, GlobalTransform())
    _ = app.world.insert[PointLightComponent](lgt_e, PointLightComponent(Vec3(18.0, 17.0, 16.0), 25.0, True))

    var dt = Float32(1.0 / 60.0)
    var start_phys_tick = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].tick

    # Phase 1 (frames 0..14): locomotion + foot-plant + socket sync + lag comp recording
    for _ in range(15):
        app.step_one_frame(dt)

    var char_x_p1 = app.world.get[LocalTransform](char_e).translation.x
    var wpn_gt_p1 = app.world.get[GlobalTransform](wpn_e).translation
    var lag_ticks_p1 = app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].lag_comp.tick_count()

    # Phase 2 (frame 15): rewind raycast hit -> knockback impulse -> RAGDOLL_SM_LIMP + impact audio
    var rewind_hits = app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].lag_comp.raycast_at_tick(
        10, Vec3(0.3, 1.0, 5.0), Vec3(0.0, 0.0, -1.0), Float32(20.0)
    )
    var char_pos_now = app.world.get[LocalTransform](char_e).translation
    app.world.get_ptr_mut[CharacterControllerComponent](char_e).value()[unsafe_offset=0].desired_vel = Vec3()
    ref rc_mut = app.world.get_ptr_mut[RagdollComponent](char_e).value()[unsafe_offset=0]
    rc_mut.state_machine.apply_hit_impulse(
        app.world.get_ptr[SkeletonComponent](char_e).value()[unsafe_offset=0].skeleton,
        Placement(char_pos_now, Vec3(1, 1, 1), Float32(0.0)),
        app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0],
        1,
        Vec3(-260.0, 40.0, -90.0),
        char_pos_now + Vec3(0.0, 0.4, 0.0),
        go_limp_threshold=Float32(100.0),
        dt=dt,
    )
    var rag_state_after_hit = rc_mut.state_machine.state
    app.world.get_ptr_mut[AudioEmitterComponent](sfx_e).value()[unsafe_offset=0].play()

    # Phase 3 (frames 15..44): ragdoll physical collapse + render + audio (no double-step)
    for frame in range(15, 45):
        app.step_one_frame(dt)
        if frame == 17:
            for _ in range(15):
                var em_ok = app.world.get_ptr[AudioEmitterComponent](sfx_e).value()[unsafe_offset=0].handle.is_valid()
                var srv_ok = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() >= 1
                if em_ok and srv_ok:
                    break
                _ = external_call["usleep", Int32](UInt32(2000))

    var end_phys_tick = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].tick
    var pelvis_end = app.world.get_ptr[RagdollComponent](char_e).value()[unsafe_offset=0].state_machine.ragdoll.pelvis_world_position()
    var emitter_valid = app.world.get_ptr[AudioEmitterComponent](sfx_e).value()[unsafe_offset=0].handle.is_valid()
    var active_voices = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count()
    ref rstate = app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
    var rframes = rstate.rendered_frames
    var rok = rstate.last_render_ok
    var px = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    app.shutdown()

    ctx.assert_true(char_x_p1 > Float32(0.25), "CharacterControllerComponent walks character forward")
    ctx.assert_true(wpn_gt_p1.x > Float32(0.20), "BoneSocketComponent follows animated hand in world space")
    ctx.assert_true(lag_ticks_p1 >= 15, "NetPlugin records >= 15 LagCompHistory ticks")
    ctx.assert_true(rewind_hits.hit and rewind_hits.entity_id == 1, "LagCompHistory raycast_at_tick hits rewound combatant")
    ctx.assert_equal(rag_state_after_hit, RAGDOLL_SM_LIMP, "apply_hit_impulse transitions RagdollStateMachine to LIMP")
    ctx.assert_equal(end_phys_tick - start_phys_tick, 45, "PhysicsWorld advances 1:1 per frame with zero double-stepping")
    ctx.assert_true(pelvis_end.x < char_pos_now.x, "ragdoll pelvis collapses along negative-X hit impulse")
    ctx.assert_true(emitter_valid and active_voices >= 1, "impact AudioEmitterComponent acquires valid AudioServer voice")
    ctx.assert_true(rframes >= 45 and rok, "RenderPlugin rendered all 45 combat frames")
    ctx.assert_true(_pixel_variance(px) > 20, "combat scene rendered non-uniform pixels")


# =========================================================================
# Aggregator for Tier 4 Character Combat Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_CharacterCombat")
    try: test_t4_character_combat_full_sequence(ctx); ctx.record_pass("test_t4_character_combat_full_sequence") except e: ctx.record_fail("test_t4_character_combat_full_sequence", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
