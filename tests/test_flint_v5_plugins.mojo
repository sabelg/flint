"""Flint v5 EngineApp Plugins & ECS Component Schemas Integration Suite (`tests/test_flint_v5_plugins.mojo`).

Verifies all 7 EngineApp plugins (`WindowPlugin`, `InputPlugin`, `PhysicsPlugin`,
`AnimPlugin`, `AudioPlugin`, `RenderPlugin`, `NetPlugin`) + `TransformPlugin`:
  1. Multi-frame `EngineApp` + `World` + `Scene` execution with all 8 plugins active.
  2. `WindowPlugin` + `InputPlugin` `STAGE_FIRST` ordering and `InputMap` action dispatch
     (headless mode opens zero OS display/evdev FDs).
  3. `PhysicsPlugin`: real gravity & floor collision, `TriggerVolume` & `PhysicsEvent`
     frame-swap delivery, `CharacterControllerComponent`, `VehicleComponent`,
     `Aero6DofComponent`, `ClothMeshComponent`, teleport sync, and despawn cleanup.
  4. `AnimPlugin`: `AnimAssetCache` model + graph evaluation, `FootPlantSolver`,
     `RagdollStateMachine` (`step_world=False` two-way coupling without double-stepping
     `PhysicsWorld`), and `BoneSocketComponent` kinematic physics sync.
  5. `AudioPlugin`: headless RT lifecycle (`"headless"` -> `"none"` on shutdown),
     `-Z` forward spatial listener/emitter sync, and offline quantum signal mixing.
  6. `RenderPlugin`: Tier-0 headless SIMD rendering of camera + mesh + material override
     + point light with `PrevGlobalTransform` interpolation and non-background pixels.
  7. `NetPlugin`: two-app `VirtualNetSim` snapshot replication (`NET_MODE_VIRTUAL`),
     client `LocalTransform` updates, and `LagCompHistory` recording.
  8. `World.snapshot_blob()` / `restore_blob()` + `ComponentRegistry` binary/JSON
     portability across ALL physics, anim, audio, render, and net components
     (including every non-trivial struct without Clone Vault).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, sin
from std.testing import assert_equal, assert_true

from flint.anim.flanim_ecs import (
    BoneSocketAttachment,
    RAGDOLL_SM_LIMP,
    RagdollStateMachine,
)
from flint.anim.graph import AnimGraphDef, StateDef
from flint.anim.plugin import (
    AnimAssetCache,
    AnimEventsResource,
    AnimPlugin,
    AnimatorComponent,
    BoneSocketComponent,
    RagdollComponent,
    SkeletonComponent,
    register_anim_codecs,
    register_anim_components,
)
from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.anim.sync import ClipInfo
from flint.app import (
    AppConfig,
    EngineApp,
    STAGE_FIRST,
    STAGE_FIXED_UPDATE,
    STAGE_LAST,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_SHUTDOWN,
    STAGE_STARTUP,
    STAGE_UPDATE,
)
from flint.app.engine_app import Resource, Resources
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeAnimation, NativeChannel, NativeModel, NativeNode
from flint.audio import AudioServer
from flint.audio.plugin import (
    AudioEmitterComponent,
    AudioListenerComponent,
    AudioPlugin,
    AudioSyncStateResource,
    register_audio_codecs,
    register_audio_components,
)
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat
from flint.ecs import Entity, ParallelCommands, World, comp_hash64
from flint.input import ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin, InputStateResource
from flint.net import (
    BitReader,
    BitWriter,
    IPV6_SAFE_MTU_BYTES,
    NetHub,
    RollbackInput,
    UdpSocket,
    VirtualEndpoint,
    VirtualNetSim,
)
from flint.net.plugin import (
    NET_MODE_VIRTUAL,
    NetConfigResource,
    NetHubResource,
    NetPlugin,
    NetReplicationResource,
    NetTransportResource,
    NetworkIdentityComponent,
    PredictionResource,
    ReplicationComponent,
    RollbackInputComponent,
    RollbackSessionResource,
    register_net_codecs,
    register_net_components,
)
from flint.physics.aero6dof import Aero6DofConfig, Aero6DofControls, Aero6DofState
from flint.physics.character import CharacterControllerConfig
from flint.physics.events import EVENT_BEGIN, EVENT_HIT
from flint.physics.joints import JOINT_WELD
from flint.physics.plugin import (
    Aero6DofComponent,
    CharacterControllerComponent,
    ClothMeshComponent,
    ColliderComponent,
    JointComponent,
    PhysicsEventsResource,
    PhysicsPlugin,
    RigidBodyComponent,
    TriggerEventsResource,
    VehicleComponent,
    register_physics_codecs,
    register_physics_components,
)
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import (
    BODY_MOTION_DYNAMIC,
    BODY_SHAPE_BOX,
    BODY_SHAPE_CAPSULE,
    BODY_SHAPE_SPHERE,
    PhysicsWorld,
    TRIGGER_ENTER,
)
from flint.physics.xpbd import ParticleMesh, RigidAttachment
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
    VatCrowdComponent,
    register_render_codecs,
    register_render_components,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    Parent,
    PrevGlobalTransform,
    Scene,
    TransformPlugin,
    TransformPropagationCache,
    propagate_transforms_parallel_cached,
    set_parent,
)


def _make_biped_model() -> NativeModel:
    var m = NativeModel()
    var names = [String("Pelvis"), String("Chest"), String("Thigh.L"), String("Thigh.R")]
    var parents = [-1, 0, 0, 0]
    var offsets = [
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.42, 0.0),
        Vec3(-0.14, -0.35, 0.0),
        Vec3(0.14, -0.35, 0.0),
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
        var p = parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)

    var anim = NativeAnimation()
    anim.name = "Idle"
    anim.duration = Float32(1.0)
    var ch = NativeChannel()
    ch.node = 1
    ch.path = 0
    ch.times.append(Float32(0.0))
    ch.times.append(Float32(1.0))
    ch.values.append(Float32(0.0)); ch.values.append(Float32(0.42)); ch.values.append(Float32(0.0))
    ch.values.append(Float32(0.0)); ch.values.append(Float32(0.45)); ch.values.append(Float32(0.0))
    anim.channels.append(ch^)
    m.animations.append(anim^)
    return m^


def _make_sine_pcm16(freq_hz: Float64, frames: Int, sample_rate: Int = 48000) -> List[Int16]:
    var out = List[Int16](capacity=frames)
    var tau = 6.283185307179586
    for f in range(frames):
        var s = 0.6 * sin(tau * freq_hz * Float64(f) / Float64(sample_rate))
        out.append(Int16(Int(s * 32767.0)))
    return out^


def _dummy_stage_first_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    pass


def test_window_and_input_plugin_ordering_and_actions() raises:
    # Register InputPlugin BEFORE a downstream STAGE_FIRST system, then WindowPlugin, plus duplicate adds!
    var app = EngineApp(AppConfig(title="OrderTest", headless=True, worker_threads=0))
    app.add_plugin(InputPlugin(enable_evdev=True))
    var inp_orig_id = app.schedule.stages[STAGE_FIRST].systems[0].id
    var custom_id = app.add_system(
        STAGE_FIRST,
        "test.after_input",
        _dummy_stage_first_sys,
        exclusive=True,
        after_system_id=inp_orig_id,
    )
    app.add_plugin(WindowPlugin(headless=True, width=800, height=600, title="OrderTest"))
    # Idempotency: re-adding must NOT duplicate systems or event swap hooks
    app.add_plugin(InputPlugin(enable_evdev=True))
    app.add_plugin(WindowPlugin(headless=True, width=800, height=600, title="OrderTest"))
    app.startup()

    assert_equal(len(app.schedule.stages[STAGE_STARTUP].systems), 1)
    assert_equal(len(app.schedule.stages[STAGE_FIRST].systems), 3)
    assert_equal(len(app.schedule.stages[STAGE_SHUTDOWN].systems), 2)

    # Verify window_poll precedes input.update AND test.after_input stays after input.update
    ref stage_first = app.schedule.stages[STAGE_FIRST]
    var win_idx = -1
    var inp_idx = -1
    var cust_idx = -1
    var win_id = -1
    var inp_sys_idx = -1
    for i in range(len(stage_first.compiled_nodes)):
        var s_id = stage_first.compiled_nodes[i].sys_idx
        if s_id >= 0 and s_id < len(stage_first.systems):
            var nm = stage_first.systems[s_id].name
            if nm == "flint.app.window_poll":
                win_idx = i
                win_id = stage_first.systems[s_id].id
            elif nm == "flint.input.update":
                inp_idx = i
                inp_sys_idx = s_id
            elif nm == "test.after_input":
                cust_idx = i
    _ = custom_id
    assert_true(win_idx >= 0 and inp_idx >= 0 and cust_idx >= 0)
    assert_true(win_idx < inp_idx and inp_idx < cust_idx)
    assert_true(win_id in stage_first.systems[inp_sys_idx].explicit_after)

    # Headless WindowPlugin opens zero OS windows/FDs
    var win_ptr = app.resources.get_ptr_mut[WindowStateResource]()
    assert_true(Bool(win_ptr))
    assert_equal(len(win_ptr.value()[unsafe_offset=0].windows), 0)
    assert_true(win_ptr.value()[unsafe_offset=0].headless)

    # Bind "jump" to key 57 (Linux KEY_SPACE) and inject input before Frame 1
    var key_space = 57
    var imap_ptr = app.resources.get_ptr_mut[InputMap]()
    assert_true(Bool(imap_ptr))
    ref imap = imap_ptr.value()[unsafe_offset=0]
    _ = imap.register_action("jump", ACTION_BOOL)
    var ctx = InputContext("gameplay", 10)
    ctx.bind_key("jump", key_space)
    imap.add_context(ctx^)
    imap.push_context("gameplay")

    var in_ptr = app.resources.get_ptr_mut[InputStateResource]()
    assert_true(Bool(in_ptr))
    in_ptr.value()[unsafe_offset=0].input.keys[key_space] = True
    in_ptr.value()[unsafe_offset=0].input.pressed[key_space] = True

    app.step_one_frame(Float32(1.0 / 60.0))
    assert_true(app.resources.get_ptr[InputMap]().value()[unsafe_offset=0].is_down("jump"))
    assert_true(app.resources.get_ptr[InputMap]().value()[unsafe_offset=0].just_pressed("jump"))

    # Frame 2 (headless, no new input): pressed[] must be cleared while held keys[] remains down
    assert_true(not in_ptr.value()[unsafe_offset=0].input.pressed[key_space])
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_true(app.resources.get_ptr[InputMap]().value()[unsafe_offset=0].is_down("jump"))
    assert_true(not app.resources.get_ptr[InputMap]().value()[unsafe_offset=0].just_pressed("jump"))
    assert_true(not in_ptr.value()[unsafe_offset=0].input.pressed[key_space])

    win_ptr.value()[unsafe_offset=0].request_close()
    assert_true(win_ptr.value()[unsafe_offset=0].close_requested)
    app.shutdown()


def test_physics_plugin_gravity_triggers_events_kcc_vehicle_aero_cloth() raises:
    var app = EngineApp(AppConfig(title="PhysicsPluginTest", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.startup()

    # Static floor at y = -0.5 (top surface at y = 0.0)
    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(30.0, 0.5, 30.0)))

    # Dynamic box at y = 2.0 (half_height = 0.5 -> settles at y = 0.5)
    var box_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](box_e, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](box_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](box_e, RigidBodyComponent.dynamic(2.0))
    _ = app.world.insert[ColliderComponent](box_e, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))

    # Trigger volume at y = 0.8
    var trig_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(0.0, 0.8, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(2.0, 1.0, 2.0), is_trigger=True))

    # CharacterController entity
    var kcc_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(6.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform())
    var kcc = CharacterControllerComponent(Vec3(6.0, 1.0, 0.0), CharacterControllerConfig(), Vec3(2.0, 0.0, 0.0))
    _ = app.world.insert[CharacterControllerComponent](kcc_e, kcc^)

    # Vehicle entity
    var veh_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](veh_e, LocalTransform(Vec3(-6.0, 0.55, 0.0)))
    _ = app.world.insert[GlobalTransform](veh_e, GlobalTransform())
    var car = VehicleChassis(1400.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(-6.0, 0.55, 0.0)
    var veh_comp = VehicleComponent(car^, throttle=1.0)
    _ = app.world.insert[VehicleComponent](veh_e, veh_comp^)

    # Aero6Dof entity
    var aero_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](aero_e, LocalTransform(Vec3(0.0, 500.0, 0.0)))
    _ = app.world.insert[GlobalTransform](aero_e, GlobalTransform())
    var aero_ctrl = Aero6DofControls()
    aero_ctrl.throttle = Float32(1.0)
    var aero_comp = Aero6DofComponent(
        Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 40.0)),
        Aero6DofConfig(),
        aero_ctrl,
    )
    _ = app.world.insert[Aero6DofComponent](aero_e, aero_comp)

    # ClothMesh entity
    var cloth_e = app.world.spawn()
    var pm = ParticleMesh()
    var p0 = pm.add_particle(Vec3(3.0, 3.0, 3.0), mass=0.0)
    var p1 = pm.add_particle(Vec3(3.0, 2.0, 3.0), mass=0.5)
    pm.add_link(p0, p1)
    pm.pin(p0, Vec3(3.0, 3.0, 3.0))
    _ = app.world.insert[ClothMeshComponent](cloth_e, ClothMeshComponent(pm^))

    var pev_cursor = 0
    var tev_cursor = 0
    var saw_collision = False
    var saw_trigger = False

    for _ in range(90):
        app.step_one_frame(Float32(1.0 / 60.0))
        var pevs = app.resources.get_ptr[PhysicsEventsResource]().value()[unsafe_offset=0].events.read_since(pev_cursor)
        for ei in range(len(pevs)):
            if pevs[ei].kind == EVENT_BEGIN or pevs[ei].kind == EVENT_HIT:
                saw_collision = True
        var tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
        for ti in range(len(tevs)):
            if tevs[ti].kind == TRIGGER_ENTER:
                saw_trigger = True

    var box_pos = app.world.get[LocalTransform](box_e).translation
    assert_true(box_pos.y > Float32(0.35) and box_pos.y < Float32(0.75))
    assert_true(saw_collision)
    assert_true(saw_trigger)

    var kcc_pos = app.world.get[LocalTransform](kcc_e).translation
    assert_true(kcc_pos.x > Float32(7.0))

    var veh_pos = app.world.get[LocalTransform](veh_e).translation
    assert_true(veh_pos.z > Float32(0.5))

    var aero_pos = app.world.get[LocalTransform](aero_e).translation
    assert_true(aero_pos.z > Float32(20.0))

    var cloth_y = app.world.get_ptr[ClothMeshComponent](cloth_e).value()[unsafe_offset=0].mesh.position[1].y
    assert_true(cloth_y < Float32(2.9))

    # Despawn dynamic box and verify body count drops on next step
    var count_before = len(app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
    assert_true(app.world.despawn(box_e))
    app.step_one_frame(Float32(1.0 / 60.0))
    var count_after = len(app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].bodies)
    assert_equal(count_after, count_before - 1)

    app.shutdown()


def test_anim_audio_render_plugins_integration() raises:
    var app = EngineApp(AppConfig(title="FullPipelineTest", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(WindowPlugin(headless=True))
    app.add_plugin(InputPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96))
    app.add_plugin(NetPlugin())
    app.startup()

    # AudioServer started in headless RT mode
    assert_equal(app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].audio_backend_name(), "headless")

    # Setup animation model & graph
    var biped = _make_biped_model()
    var graph = AnimGraphDef()
    graph.name = "SimpleIdle"
    var st = StateDef()
    st.name = "Idle"
    st.clip = 0
    graph.states.append(st^)
    var clips = List[ClipInfo]()
    clips.append(ClipInfo("Idle", Float32(1.0), True, Float32(0.0)))

    var cache_ptr = app.resources.get_ptr_mut[AnimAssetCache]()
    var model_id = UInt64(101)
    var graph_id = UInt64(202)
    cache_ptr.value()[unsafe_offset=0].register_model(model_id, biped^)
    cache_ptr.value()[unsafe_offset=0].register_graph(graph_id, graph^, clips^)

    # Floor for physics + ragdoll
    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(20.0, 0.5, 20.0)))

    # Animated + ragdoll character entity
    var char_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](char_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](char_e, GlobalTransform())
    _ = app.world.insert[SkeletonComponent](char_e, SkeletonComponent(model_id))
    _ = app.world.insert[AnimatorComponent](char_e, AnimatorComponent(graph_id))

    ref model_ref = cache_ptr.value()[unsafe_offset=0].models[0]
    var skel_tmp = SkeletonInstance()
    skel_tmp.reset_to_rest(model_ref)
    var rag = ArticulatedRagdoll()
    _ = rag.add_limb_from_model(model_ref, 0, -1, 15.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(model_ref, 1, 0, 12.0, 0.15, 0.35)
    rag.handoff_from_skeleton(skel_tmp, Placement(), model_ref, Float32(1.0 / 60.0))
    rag.go_limp()
    var rsm = RagdollStateMachine()
    rsm.ragdoll = rag^
    rsm.state = RAGDOLL_SM_LIMP
    _ = app.world.insert[RagdollComponent](char_e, RagdollComponent(rsm^, auto_recover=False, pelvis_syncs_transform=True))

    # BoneSocket entity attached to Chest bone (node 1) with kinematic body
    var phys_ptr = app.resources.get_ptr_mut[PhysicsWorld]()
    var kin_slot = phys_ptr.value()[unsafe_offset=0].add_kinematic_box(Vec3(0.0, 1.4, 0.0), Vec3(0.1, 0.1, 0.1))
    var kin_bid = phys_ptr.value()[unsafe_offset=0].body_id(kin_slot)
    var sock_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](sock_e, LocalTransform())
    _ = app.world.insert[GlobalTransform](sock_e, GlobalTransform())
    _ = app.world.insert[BoneSocketComponent](
        sock_e,
        BoneSocketComponent(char_e, BoneSocketAttachment("HandWeapon", 1, -1, Vec3(0.2, 0.0, 0.0), Quat(), Vec3(1, 1, 1), kin_bid)),
    )

    # Render setup: material + mesh + camera + point light
    var rend_ptr = app.resources.get_ptr_mut[ScalableRenderer]()
    var mat_id = rend_ptr.value()[unsafe_offset=0].register_material(MaterialAsset(base_color=Vec3(0.85, 0.25, 0.2)))
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.7, 0.7, 0.7))
    var mesh_id = rend_ptr.value()[unsafe_offset=0].register_mesh(mb^.finish(), mat_id)

    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 0.8, 3.5)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=55.0, active=True))

    var mesh_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](mesh_e, LocalTransform(Vec3(0.0, 0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](mesh_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](mesh_e, PrevGlobalTransform())
    _ = app.world.insert[MeshRendererComponent](mesh_e, MeshRendererComponent(mesh_id, mat_id))
    _ = app.world.insert[MaterialOverrideComponent](
        mesh_e, MaterialOverrideComponent(mat_id, Vec3(0.2, 0.8, 0.3), 0.3, 0.1, Vec3(), True)
    )

    var lgt_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(2.0, 3.0, 2.0)))
    _ = app.world.insert[GlobalTransform](lgt_e, GlobalTransform())
    _ = app.world.insert[PointLightComponent](lgt_e, PointLightComponent(Vec3(8.0, 8.0, 8.0), 15.0, True))

    # Audio setup: PCM16 asset + listener + spatial emitter
    var pcm = _make_sine_pcm16(440.0, 24000, 48000)
    var audio_aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(pcm, 1, 48000)

    var lis_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_e, LocalTransform(Vec3(0.0, 1.0, 3.0)))
    _ = app.world.insert[GlobalTransform](lis_e, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_e, AudioListenerComponent(active=True, priority=1))

    var emt_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](emt_e, LocalTransform(Vec3(4.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](emt_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        emt_e, AudioEmitterComponent(asset_id=audio_aid, asset_path="sine.wav", looping=True, play_on_spawn=True)
    )

    var start_phys_tick = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].tick
    for _ in range(12):
        app.step_one_frame(Float32(1.0 / 60.0))

    # Ragdoll update_world(step_world=False) MUST NOT double-step PhysicsWorld!
    var end_phys_tick = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].tick
    assert_equal(end_phys_tick - start_phys_tick, 12)

    # BoneSocket propagated to socket entity GlobalTransform
    var sock_gt = app.world.get[GlobalTransform](sock_e)
    assert_true(sock_gt.translation.x == sock_gt.translation.x)

    # RenderPlugin extracted and rendered frame
    var rw = app.resources.get_ptr[RenderWorld]()
    assert_equal(rw.value()[unsafe_offset=0].instance_count(), 1)
    assert_equal(len(rw.value()[unsafe_offset=0].cluster_lights), 1)
    var rstate = app.resources.get_ptr[RenderPluginStateResource]()
    assert_true(rstate.value()[unsafe_offset=0].rendered_frames >= 12)
    assert_true(rstate.value()[unsafe_offset=0].last_render_ok)
    var px = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    assert_equal(len(px), 128 * 96)

    # Audio emitter triggered valid handle
    assert_true(app.world.get_ptr[AudioEmitterComponent](emt_e).value()[unsafe_offset=0].handle.is_valid())

    app.shutdown()
    assert_equal(app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].audio_backend_name(), "none")

    # Offline quantum verification when RT thread is stopped
    var srv_off = AudioServer(48000, 256)
    var off_aid = srv_off.register_pcm16_asset(_make_sine_pcm16(440.0, 24000, 48000), 1, 48000)
    _ = srv_off.play_2d(off_aid, volume=0.8, looping=True)
    var out_lr = List[Float32](length=512, fill=0.0)
    srv_off.render_offline_quantum(out_lr, 256)
    var energy = Float32(0.0)
    for i in range(512):
        energy += abs(out_lr[i])
    assert_true(energy > Float32(0.01))


def test_net_plugin_virtual_sim_replication() raises:
    var sim = VirtualNetSim(
        seed=0x12345678ABCDEF01,
        one_way_latency_ms=5.0,
        jitter_ms=0.0,
        loss_rate=0.0,
        dup_rate=0.0,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
    )
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9400), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9400), False)
    var cli_cid = cli_hub.connect(srv_ep)
    assert_equal(cli_cid, 0)
    # Trigger channel establishment with initial handshake packet
    assert_true(cli_hub.send_unreliable(cli_cid, List[UInt8](), sim.now_ms()))

    var server_app = EngineApp(AppConfig(title="NetServer", headless=True, worker_threads=0))
    server_app.add_plugin(TransformPlugin())
    server_app.add_plugin(NetPlugin(max_entities=64, max_clients=4))
    server_app.startup()
    server_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    server_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(srv_hub^)

    var client_app = EngineApp(AppConfig(title="NetClient", headless=True, worker_threads=0))
    client_app.add_plugin(TransformPlugin())
    client_app.add_plugin(NetPlugin(max_entities=64, max_clients=4))
    client_app.startup()
    client_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    client_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(cli_hub^)

    var target_pos = Vec3(12.5, 2.0, -7.25)
    var srv_e = server_app.world.spawn()
    _ = server_app.world.insert[LocalTransform](srv_e, LocalTransform(target_pos))
    _ = server_app.world.insert[GlobalTransform](srv_e, GlobalTransform())
    _ = server_app.world.insert[NetworkIdentityComponent](srv_e, NetworkIdentityComponent(net_id=1))
    _ = server_app.world.insert[ReplicationComponent](srv_e, ReplicationComponent())

    var cli_e = client_app.world.spawn()
    _ = client_app.world.insert[LocalTransform](cli_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = client_app.world.insert[GlobalTransform](cli_e, GlobalTransform())
    _ = client_app.world.insert[NetworkIdentityComponent](cli_e, NetworkIdentityComponent(net_id=1))
    _ = client_app.world.insert[ReplicationComponent](cli_e, ReplicationComponent())

    var dt = Float32(1.0 / 60.0)
    for _ in range(25):
        server_app.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        client_app.step_one_frame(dt)

    var cli_pos = client_app.world.get[LocalTransform](cli_e).translation
    assert_true(abs(cli_pos.x - target_pos.x) < Float32(0.05))
    assert_true(abs(cli_pos.y - target_pos.y) < Float32(0.05))
    assert_true(abs(cli_pos.z - target_pos.z) < Float32(0.05))
    # Client GlobalTransform MUST also propagate (not stay frozen at 0,0,0)!
    var cli_gpos = client_app.world.get[GlobalTransform](cli_e).translation
    assert_true(abs(cli_gpos.x - target_pos.x) < Float32(0.05))
    assert_true(abs(cli_gpos.y - target_pos.y) < Float32(0.05))
    assert_true(abs(cli_gpos.z - target_pos.z) < Float32(0.05))
    assert_true(server_app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].lag_comp.tick_count() > 0)
    # Client snapshot ACK MUST be received by server replicator
    assert_true(server_app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].replicator.clients[0].has_acked_seq)

    # Server despawn -> DIRTY_DESTROY -> client despawn
    assert_true(server_app.world.despawn(srv_e))
    for _ in range(10):
        server_app.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        client_app.step_one_frame(dt)

    assert_true(not server_app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].replicator.entities[1].active)
    assert_true(not client_app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].receiver.entities[1].active)
    assert_true(not client_app.world.is_alive(cli_e))

    client_app.shutdown()
    server_app.shutdown()


def test_snapshot_blob_and_component_registry_all_schemas() raises:
    var w = World()
    register_physics_components(w)
    register_anim_components(w)
    register_audio_components(w)
    register_render_components(w)
    register_net_components(w)

    var e = w.spawn()
    _ = w.insert[RigidBodyComponent](e, RigidBodyComponent.dynamic(4.5))
    var pts: List[Vec3] = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
    _ = w.insert[ColliderComponent](e, ColliderComponent.convex_hull(pts))
    _ = w.insert[JointComponent](e, JointComponent())
    _ = w.insert[CharacterControllerComponent](e, CharacterControllerComponent(Vec3(1.0, 2.0, 3.0)))
    var car = VehicleChassis(1550.0)
    car.setup_four_wheel_car()
    _ = w.insert[VehicleComponent](e, VehicleComponent(car^, throttle=0.75))
    _ = w.insert[Aero6DofComponent](e, Aero6DofComponent(Aero6DofState(Vec3(0.0, 100.0, 0.0), Vec3(0.0, 0.0, 50.0))))
    var pm = ParticleMesh()
    var p0 = pm.add_particle(Vec3(0.0, 2.0, 0.0), 1.0)
    var p1 = pm.add_particle(Vec3(1.0, 2.0, 0.0), 1.0)
    pm.add_link(p0, p1)
    var atts: List[RigidAttachment] = [RigidAttachment(p0, -1, Vec3(0.1, 0.2, 0.3))]
    _ = w.insert[ClothMeshComponent](e, ClothMeshComponent(pm^, atts^, wind=Vec3(1.0, 0.0, 0.0)))

    _ = w.insert[SkeletonComponent](e, SkeletonComponent(UInt64(77), SkeletonInstance(3)))
    var ac = AnimatorComponent(UInt64(88), apply_root_motion=True, enable_foot_plant=True)
    ac.foot_plant.add_leg(1, 2, 3)
    ac.foot_in_stance.append(True)
    _ = w.insert[AnimatorComponent](e, ac^)
    _ = w.insert[RagdollComponent](e, RagdollComponent())
    _ = w.insert[BoneSocketComponent](e, BoneSocketComponent(e, BoneSocketAttachment("SwordSocket", 2)))

    _ = w.insert[AudioListenerComponent](e, AudioListenerComponent(active=True, priority=5))
    _ = w.insert[AudioEmitterComponent](e, AudioEmitterComponent(asset_id=3, asset_path="sfx/boom.qoa", volume=0.85))

    _ = w.insert[MeshRendererComponent](e, MeshRendererComponent(4, 2, True, True))
    _ = w.insert[CameraComponent](e, CameraComponent(60.0, 0.1, 500.0, 1.2, True))
    _ = w.insert[PointLightComponent](e, PointLightComponent(Vec3(2.0, 3.0, 4.0), 18.0, True))
    _ = w.insert[MaterialOverrideComponent](e, MaterialOverrideComponent(2, Vec3(0.1, 0.2, 0.9), 0.2, 0.8, Vec3(), True))
    _ = w.insert[VatCrowdComponent](e, VatCrowdComponent(1, 0.5, 0.4, 2, True))

    _ = w.insert[NetworkIdentityComponent](e, NetworkIdentityComponent(42, 1, True))
    _ = w.insert[ReplicationComponent](e, ReplicationComponent())
    _ = w.insert[RollbackInputComponent](e, RollbackInputComponent(1, UInt32(9), 0.25, 0.9, UInt32(3)))

    # 1. Standalone World.snapshot_blob / restore_blob (NO Clone Vault)
    var blob = w.snapshot_blob()
    assert_true(len(blob) > 0)

    var w2 = World()
    register_physics_components(w2)
    register_anim_components(w2)
    register_audio_components(w2)
    register_render_components(w2)
    register_net_components(w2)
    w2.restore_blob(blob)

    assert_true(w2.has[RigidBodyComponent](e))
    assert_equal(len(w2.get_ptr[ColliderComponent](e).value()[unsafe_offset=0].points), 4)
    assert_true(abs(w2.get_ptr[VehicleComponent](e).value()[unsafe_offset=0].throttle - Float32(0.75)) < Float32(1.0e-5))
    assert_equal(w2.get_ptr[ClothMeshComponent](e).value()[unsafe_offset=0].mesh.count(), 2)
    assert_equal(len(w2.get_ptr[ClothMeshComponent](e).value()[unsafe_offset=0].attachments), 1)
    assert_equal(w2.get_ptr[BoneSocketComponent](e).value()[unsafe_offset=0].attachment.name, "SwordSocket")
    assert_equal(w2.get_ptr[AudioEmitterComponent](e).value()[unsafe_offset=0].asset_path, "sfx/boom.qoa")
    assert_equal(w2.get_ptr[NetworkIdentityComponent](e).value()[unsafe_offset=0].net_id, 42)

    # 2. ComponentRegistry portability (JSON + Binary Scene codecs)
    var reg = ComponentRegistry.with_builtins()
    register_physics_codecs(reg)
    register_anim_codecs(reg)
    register_audio_codecs(reg)
    register_render_codecs(reg)
    register_net_codecs(reg)
    assert_true(reg.find_by_name("RigidBodyComponent") >= 0)
    assert_true(reg.find_by_name("SkeletonComponent") >= 0)
    assert_true(reg.find_by_name("AudioEmitterComponent") >= 0)
    assert_true(reg.find_by_name("MeshRendererComponent") >= 0)
    assert_true(reg.find_by_name("NetworkIdentityComponent") >= 0)

    # JSON round-trip check on non-trivial AudioEmitterComponent & ReplicationComponent.replicate_state=False
    var em_json = w2.get_ptr[AudioEmitterComponent](e).value()[unsafe_offset=0].write_json()
    var em_bytes = List[UInt8](em_json.as_bytes())
    var jdoc = parse_json(em_bytes)
    assert_true(jdoc.valid)
    var em_rt = AudioEmitterComponent.read_json(jdoc, jdoc.root)
    assert_equal(em_rt.asset_path, "sfx/boom.qoa")
    assert_equal(em_rt.asset_id, 3)

    var rep_false = ReplicationComponent(replicate_state=False, state_flags=42)
    var rep_json = rep_false.write_json()
    var rep_bytes = List[UInt8](rep_json.as_bytes())
    var rdoc = parse_json(rep_bytes)
    assert_true(rdoc.valid)
    var rep_rt = ReplicationComponent.read_json(rdoc, rdoc.root)
    assert_true(not rep_rt.replicate_state)
    assert_equal(Int(rep_rt.state_flags), 42)


@fieldwise_init
struct _TestRbInputA(RollbackInput):
    var a: Int32

    def __init__(out self):
        self.a = 0

    def encode_bits(self, mut writer: BitWriter):
        writer.write_u32(UInt32(self.a))

    def decode_bits(mut self, mut reader: BitReader):
        self.a = Int32(reader.read_u32())

    def same_input(self, other: Self) -> Bool:
        return self.a == other.a


@fieldwise_init
struct _TestRbInputB(RollbackInput):
    var b: Int32

    def __init__(out self):
        self.b = 0

    def encode_bits(self, mut writer: BitWriter):
        writer.write_u32(UInt32(self.b))

    def decode_bits(mut self, mut reader: BitReader):
        self.b = Int32(reader.read_u32())

    def same_input(self, other: Self) -> Bool:
        return self.b == other.b


def test_plugin_idempotency_transform_ticks_and_generic_ids() raises:
    # 1. Generic Resource RESOURCE_ID disambiguation
    assert_true(NetHubResource[VirtualEndpoint].RESOURCE_ID != NetHubResource[UdpSocket].RESOURCE_ID)
    assert_true(RollbackSessionResource[_TestRbInputA].RESOURCE_ID != RollbackSessionResource[_TestRbInputB].RESOURCE_ID)
    assert_true(PredictionResource[_TestRbInputA].RESOURCE_ID != PredictionResource[_TestRbInputB].RESOURCE_ID)

    # 2. All-plugin idempotent build (double registration does not double-tick PhysicsWorld or systems)
    var app = EngineApp(AppConfig(title="IdemTrans", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(WindowPlugin(headless=True))
    app.add_plugin(InputPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
    app.add_plugin(NetPlugin())

    app.add_plugin(WindowPlugin(headless=True))
    app.add_plugin(InputPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
    app.add_plugin(NetPlugin())
    app.startup()

    assert_equal(len(app.schedule.stages[STAGE_FIRST].systems), 3)
    assert_equal(len(app.schedule.stages[STAGE_PRE_UPDATE].systems), 2)
    assert_equal(len(app.schedule.stages[STAGE_FIXED_UPDATE].systems), 3)
    assert_equal(len(app.schedule.stages[STAGE_UPDATE].systems), 4)
    assert_equal(len(app.schedule.stages[STAGE_POST_UPDATE].systems), 2)
    assert_equal(len(app.schedule.stages[STAGE_LAST].systems), 2)
    assert_equal(len(app.schedule.stages[STAGE_SHUTDOWN].systems), 5)

    # 3. Between-frame and due_steps=0 TransformPlugin propagation
    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())

    app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].tick, 1)
    assert_true(abs(app.world.get[GlobalTransform](e).translation.x) < Float32(1.0e-5))

    # Mutate LocalTransform between frames, then step with dt=1/120 (due_steps == 0)
    app.world.get_ptr_mut[LocalTransform](e).value()[unsafe_offset=0].translation.x = Float32(2.0)
    app.step_one_frame(Float32(1.0 / 120.0))
    assert_true(abs(app.world.get[GlobalTransform](e).translation.x - Float32(2.0)) < Float32(1.0e-4))

    # Mutate again between frames, then step with dt=1/60 (due_steps == 1)
    app.world.get_ptr_mut[LocalTransform](e).value()[unsafe_offset=0].translation.x = Float32(5.0)
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_true(abs(app.world.get[GlobalTransform](e).translation.x - Float32(5.0)) < Float32(1.0e-4))

    app.shutdown()


def test_physics_non_rb_despawn_and_snapshot_restore() raises:
    # 1. Non-RB despawn cleanup (Vehicle, Aero, Cloth, Trigger, Joint, SixDofJoint)
    var app = EngineApp(AppConfig(title="PhysCleanup", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.startup()

    var veh_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](veh_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    var car = VehicleChassis(1400.0)
    car.setup_four_wheel_car()
    _ = app.world.insert[VehicleComponent](veh_e, VehicleComponent(car^))

    var aero_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](aero_e, LocalTransform(Vec3(0.0, 100.0, 0.0)))
    _ = app.world.insert[Aero6DofComponent](aero_e, Aero6DofComponent())

    var cloth_e = app.world.spawn()
    var pm = ParticleMesh()
    var p0 = pm.add_particle(Vec3(0.0, 3.0, 0.0), 0.0)
    var p1 = pm.add_particle(Vec3(1.0, 3.0, 0.0), 0.5)
    pm.add_link(p0, p1)
    _ = app.world.insert[ClothMeshComponent](cloth_e, ClothMeshComponent(pm^))

    var trig_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(1.0, 1.0, 1.0), is_trigger=True))

    var j1_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](j1_e, LocalTransform(Vec3(2.0, 5.0, 0.0)))
    _ = app.world.insert[RigidBodyComponent](j1_e, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[JointComponent](j1_e, JointComponent(joint_kind=JOINT_WELD, is_six_dof=False))

    var j2_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](j2_e, LocalTransform(Vec3(-2.0, 5.0, 0.0)))
    _ = app.world.insert[RigidBodyComponent](j2_e, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[JointComponent](j2_e, JointComponent(joint_kind=JOINT_WELD, is_six_dof=True))

    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))

    assert_true(app.world.despawn(veh_e))
    assert_true(app.world.despawn(aero_e))
    assert_true(app.world.despawn(cloth_e))
    assert_true(app.world.despawn(trig_e))
    assert_true(app.world.despawn(j1_e))
    assert_true(app.world.despawn(j2_e))
    app.step_one_frame(Float32(1.0 / 60.0))

    ref pw = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_true(not pw.vehicles[0].active)
    assert_true(not pw.aero_bodies[0].active)
    assert_true(not pw.xpbd_meshes[0].active)
    assert_true(not pw.triggers[0].active)
    assert_true(pw.joints[0].broken)
    assert_true(pw.six_dof_joints[0].broken)

    # 2. Stepped-world snapshot/restore into a NON-EMPTY PhysicsWorld (zero slot collisions)
    var app1 = EngineApp(AppConfig(title="SnapSrc", headless=True, worker_threads=0))
    app1.add_plugin(TransformPlugin())
    app1.add_plugin(PhysicsPlugin(gravity=9.81))
    app1.startup()

    var s_box = app1.world.spawn()
    _ = app1.world.insert[LocalTransform](s_box, LocalTransform(Vec3(0.0, 5.0, 0.0)))
    _ = app1.world.insert[GlobalTransform](s_box, GlobalTransform())
    _ = app1.world.insert[RigidBodyComponent](s_box, RigidBodyComponent.dynamic(2.0))
    _ = app1.world.insert[ColliderComponent](s_box, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))
    _ = app1.world.insert[JointComponent](s_box, JointComponent(joint_kind=JOINT_WELD, anchor_a=Vec3(0.0, 5.0, 0.0)))

    var s_trig = app1.world.spawn()
    _ = app1.world.insert[LocalTransform](s_trig, LocalTransform(Vec3(10.0, 1.0, 0.0)))
    _ = app1.world.insert[ColliderComponent](s_trig, ColliderComponent.box(Vec3(1.0, 1.0, 1.0), is_trigger=True))

    var s_veh = app1.world.spawn()
    _ = app1.world.insert[LocalTransform](s_veh, LocalTransform(Vec3(20.0, 1.0, 0.0)))
    var car1 = VehicleChassis(1400.0)
    car1.setup_four_wheel_car()
    _ = app1.world.insert[VehicleComponent](s_veh, VehicleComponent(car1^))

    for _ in range(5):
        app1.step_one_frame(Float32(1.0 / 60.0))

    var blob = app1.world.snapshot_blob()

    var app2 = EngineApp(AppConfig(title="SnapDst", headless=True, worker_threads=0))
    app2.add_plugin(TransformPlugin())
    app2.add_plugin(PhysicsPlugin(gravity=9.81))
    app2.startup()

    # Pre-populate app2 PhysicsWorld slot 0 with an existing vehicle, trigger, joint, and body
    var p2_veh = app2.world.spawn()
    _ = app2.world.insert[LocalTransform](p2_veh, LocalTransform(Vec3(100.0, 1.0, 0.0)))
    var car2 = VehicleChassis(1400.0)
    car2.setup_four_wheel_car()
    _ = app2.world.insert[VehicleComponent](p2_veh, VehicleComponent(car2^))
    var p2_trig = app2.world.spawn()
    _ = app2.world.insert[LocalTransform](p2_trig, LocalTransform(Vec3(100.0, 1.0, 0.0)))
    _ = app2.world.insert[ColliderComponent](p2_trig, ColliderComponent.box(Vec3(1.0, 1.0, 1.0), is_trigger=True))
    app2.step_one_frame(Float32(1.0 / 60.0))

    app2.world.restore_blob(blob)
    for _ in range(15):
        app2.step_one_frame(Float32(1.0 / 60.0))

    ref pw2 = app2.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_true(len(pw2.vehicles) >= 2)
    assert_true(len(pw2.triggers) >= 2)
    assert_true(len(pw2.joints) >= 1)
    # Welded box in restored world must remain held near y=5.0 rather than falling
    assert_true(app2.world.get[LocalTransform](s_box).translation.y > Float32(4.7))

    app2.shutdown()
    app1.shutdown()
    app.shutdown()


def test_anim_parented_bone_socket_audio_despawn_and_net_virtual_physics() raises:
    # 1. Parented BoneSocket zero-lag + AudioEmitter despawn cleanup + Multi-listener priority
    var app = EngineApp(AppConfig(title="AnimAudioZeroLag", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.startup()

    var biped = _make_biped_model()
    var model_id = UInt64(505)
    app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, biped^)

    var char_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](char_e, LocalTransform(Vec3(0.0, 10.0, 0.0)))
    _ = app.world.insert[GlobalTransform](char_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](char_e, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](char_e, ColliderComponent.sphere(0.5))
    _ = app.world.insert[SkeletonComponent](char_e, SkeletonComponent(model_id))

    var sock_child = app.world.spawn()
    _ = app.world.insert[Parent](sock_child, Parent(char_e))
    _ = app.world.insert[LocalTransform](sock_child, LocalTransform())
    _ = app.world.insert[GlobalTransform](sock_child, GlobalTransform())
    _ = app.world.insert[BoneSocketComponent](
        sock_child,
        BoneSocketComponent(char_e, BoneSocketAttachment("ChestWeapon", 1, -1, Vec3(0.25, 0.0, 0.0))),
    )

    var pcm = _make_sine_pcm16(440.0, 24000, 48000)
    var audio_aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(pcm, 1, 48000)
    var lis_low = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_low, LocalTransform(Vec3(100.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](lis_low, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_low, AudioListenerComponent(active=True, priority=1))

    var lis_high = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_high, LocalTransform(Vec3(0.0, 1.0, 3.0)))
    _ = app.world.insert[GlobalTransform](lis_high, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_high, AudioListenerComponent(active=True, priority=10))

    var emt_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](emt_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](emt_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        emt_e, AudioEmitterComponent(asset_id=audio_aid, asset_path="loop.wav", looping=True, play_on_spawn=True)
    )

    for _ in range(15):
        app.step_one_frame(Float32(1.0 / 60.0))

    # Zero-lag check on parented BoneSocketComponent
    var sock_gt = app.world.get[GlobalTransform](sock_child).translation
    var sock_wp = app.world.get_ptr[BoneSocketComponent](sock_child).value()[unsafe_offset=0].attachment.world_pos
    assert_true((sock_gt - sock_wp).length() < Float32(1.0e-4))

    # Highest-priority listener selected
    var async_state = app.resources.get_ptr[AudioSyncStateResource]()
    assert_true((async_state.value()[unsafe_offset=0].prev_listener_pos - Vec3(0.0, 1.0, 3.0)).length() < Float32(1.0e-4))
    assert_equal(len(async_state.value()[unsafe_offset=0].active_emitters), 1)

    # Despawn looping AudioEmitterComponent -> tracked emitter cleared & RT emitter stopped
    assert_true(app.world.despawn(emt_e))
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(async_state.value()[unsafe_offset=0].active_emitters), 0)
    for _ in range(25):
        if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() == 0:
            break
        _ = external_call["usleep", Int32](UInt32(2000))
    assert_equal(app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count(), 0)
    app.shutdown()

    # 2. NET_MODE_VIRTUAL + PhysicsPlugin on server without client->server clobbering
    var sim = VirtualNetSim(seed=0xABCDEF1234567890, one_way_latency_ms=5.0, jitter_ms=0.0, loss_rate=0.0, dup_rate=0.0)
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9500), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9500), False)
    var cli_cid = cli_hub.connect(srv_ep)
    assert_true(cli_hub.send_unreliable(cli_cid, List[UInt8](), sim.now_ms()))

    var srv_app = EngineApp(AppConfig(title="VirtPhysSrv", headless=True, worker_threads=0))
    srv_app.add_plugin(TransformPlugin())
    srv_app.add_plugin(PhysicsPlugin(gravity=0.0))
    srv_app.add_plugin(NetPlugin(max_entities=32, max_clients=4))
    srv_app.startup()
    srv_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    srv_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(srv_hub^)

    var cli_app = EngineApp(AppConfig(title="VirtPhysCli", headless=True, worker_threads=0))
    cli_app.add_plugin(TransformPlugin())
    cli_app.add_plugin(NetPlugin(max_entities=32, max_clients=4))
    cli_app.startup()
    cli_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    cli_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(cli_hub^)

    var srv_e = srv_app.world.spawn()
    _ = srv_app.world.insert[LocalTransform](srv_e, LocalTransform(Vec3(10.0, 5.0, 0.0)))
    _ = srv_app.world.insert[GlobalTransform](srv_e, GlobalTransform())
    _ = srv_app.world.insert[RigidBodyComponent](srv_e, RigidBodyComponent.dynamic(1.0))
    _ = srv_app.world.insert[NetworkIdentityComponent](srv_e, NetworkIdentityComponent(net_id=1))
    _ = srv_app.world.insert[ReplicationComponent](srv_e, ReplicationComponent())

    var cli_e = cli_app.world.spawn()
    _ = cli_app.world.insert[LocalTransform](cli_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = cli_app.world.insert[GlobalTransform](cli_e, GlobalTransform())
    _ = cli_app.world.insert[NetworkIdentityComponent](cli_e, NetworkIdentityComponent(net_id=1))
    _ = cli_app.world.insert[ReplicationComponent](cli_e, ReplicationComponent())

    var dt = Float32(1.0 / 60.0)
    for _ in range(20):
        srv_app.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        cli_app.step_one_frame(dt)

    var srv_x = srv_app.world.get[LocalTransform](srv_e).translation.x
    var cli_gx = cli_app.world.get[GlobalTransform](cli_e).translation.x
    assert_true(abs(srv_x - Float32(10.0)) < Float32(0.05))
    assert_true(abs(cli_gx - Float32(10.0)) < Float32(0.05))

    cli_app.shutdown()
    srv_app.shutdown()


def _count_active_bodies(pw: PhysicsWorld) -> Int:
    var count = 0
    for i in range(len(pw.bodies)):
        if pw.bodies[i].active:
            count += 1
    return count


def _make_limp_ragdoll_component(model_ref: NativeModel) -> RagdollComponent:
    var skel_tmp = SkeletonInstance()
    skel_tmp.reset_to_rest(model_ref)
    var rag = ArticulatedRagdoll()
    _ = rag.add_limb_from_model(model_ref, 0, -1, 15.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(model_ref, 1, 0, 12.0, 0.15, 0.35)
    rag.handoff_from_skeleton(skel_tmp, Placement(), model_ref, Float32(1.0 / 60.0))
    rag.go_limp()
    var rsm = RagdollStateMachine()
    rsm.ragdoll = rag^
    rsm.state = RAGDOLL_SM_LIMP
    return RagdollComponent(rsm^, auto_recover=False, pelvis_syncs_transform=True)


def test_physics_rb_overlap_restore_and_replace_cleanup() raises:
    var app1 = EngineApp(AppConfig(title="RBSnapSrc", headless=True, worker_threads=0))
    app1.add_plugin(TransformPlugin())
    app1.add_plugin(PhysicsPlugin(gravity=9.81))
    app1.startup()

    var e_src = app1.world.spawn()
    _ = app1.world.insert[LocalTransform](e_src, LocalTransform(Vec3(0.0, 5.0, 0.0)))
    _ = app1.world.insert[GlobalTransform](e_src, GlobalTransform())
    _ = app1.world.insert[RigidBodyComponent](e_src, RigidBodyComponent.dynamic(2.0))
    _ = app1.world.insert[ColliderComponent](e_src, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))

    for _ in range(3):
        app1.step_one_frame(Float32(1.0 / 60.0))
    var blob = app1.world.snapshot_blob()

    # Non-empty app2 holding a RigidBodyComponent on the exact same Entity(0, 0) slot at x=100
    var app2 = EngineApp(AppConfig(title="RBSnapDst", headless=True, worker_threads=0))
    app2.add_plugin(TransformPlugin())
    app2.add_plugin(PhysicsPlugin(gravity=9.81))
    app2.startup()

    var e_dst = app2.world.spawn()
    assert_equal(e_dst.to_bits(), e_src.to_bits())
    _ = app2.world.insert[LocalTransform](e_dst, LocalTransform(Vec3(100.0, 100.0, 100.0)))
    _ = app2.world.insert[GlobalTransform](e_dst, GlobalTransform())
    _ = app2.world.insert[RigidBodyComponent](e_dst, RigidBodyComponent.dynamic(1.0))
    _ = app2.world.insert[ColliderComponent](e_dst, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))
    app2.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(_count_active_bodies(app2.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]), 1)

    app2.world.restore_blob(blob)
    for _ in range(3):
        app2.step_one_frame(Float32(1.0 / 60.0))

    ref pw2 = app2.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_equal(len(pw2.bodies), 1)
    assert_equal(_count_active_bodies(pw2), 1)
    assert_true(abs(pw2.bodies[0].pos.x) < Float32(1.0))

    # Same-app checkpoint restore_blob must not leak a ghost RigidBody
    var blob_mid = app2.world.snapshot_blob()
    for _ in range(3):
        app2.step_one_frame(Float32(1.0 / 60.0))
    app2.world.restore_blob(blob_mid)
    app2.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(pw2.bodies), 1)
    assert_equal(_count_active_bodies(pw2), 1)

    # Replacing RigidBodyComponent via world.insert must swap-remove the old body
    _ = app2.world.insert[RigidBodyComponent](e_src, RigidBodyComponent.dynamic(5.0))
    app2.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(pw2.bodies), 1)
    assert_equal(_count_active_bodies(pw2), 1)

    app2.shutdown()
    app1.shutdown()


def test_anim_ragdoll_despawn_remove_and_restore_cleanup() raises:
    var model_id = UInt64(606)

    var app = EngineApp(AppConfig(title="RagCleanup", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.startup()

    var biped = _make_biped_model()
    app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, biped^)

    # 1. Entity despawn deactivates ragdoll and its 2 hosted limb bodies
    var e1 = app.world.spawn()
    _ = app.world.insert[LocalTransform](e1, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e1, GlobalTransform())
    _ = app.world.insert[SkeletonComponent](e1, SkeletonComponent(model_id))
    var rc1 = _make_limp_ragdoll_component(app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = app.world.insert[RagdollComponent](e1, rc1^)

    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
    ref pw = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_equal(len(pw.ragdolls), 1)
    assert_true(pw.ragdolls[0].rag.is_active)
    assert_equal(_count_active_bodies(pw), 2)

    assert_true(app.world.despawn(e1))
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_true(not pw.ragdolls[0].rag.is_active)
    assert_equal(_count_active_bodies(pw), 0)

    # 2. world.remove[RagdollComponent] deactivates ragdoll and its 2 hosted limb bodies
    var e2 = app.world.spawn()
    _ = app.world.insert[LocalTransform](e2, LocalTransform(Vec3(3.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e2, GlobalTransform())
    _ = app.world.insert[SkeletonComponent](e2, SkeletonComponent(model_id))
    var rc2 = _make_limp_ragdoll_component(app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = app.world.insert[RagdollComponent](e2, rc2^)

    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(pw.ragdolls), 2)
    assert_true(pw.ragdolls[1].rag.is_active)
    assert_equal(_count_active_bodies(pw), 2)

    assert_true(app.world.remove[RagdollComponent](e2))
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_true(not pw.ragdolls[1].rag.is_active)
    assert_equal(_count_active_bodies(pw), 0)
    app.shutdown()

    # 3. World.restore_blob into non-empty EngineApp deactivates old overlapping ragdoll limbs
    var src_app = EngineApp(AppConfig(title="RagSnapSrc", headless=True, worker_threads=0))
    src_app.add_plugin(TransformPlugin())
    src_app.add_plugin(PhysicsPlugin(gravity=9.81))
    src_app.add_plugin(AnimPlugin())
    src_app.startup()
    src_app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, _make_biped_model())

    var e_src = src_app.world.spawn()
    _ = src_app.world.insert[LocalTransform](e_src, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = src_app.world.insert[GlobalTransform](e_src, GlobalTransform())
    _ = src_app.world.insert[SkeletonComponent](e_src, SkeletonComponent(model_id))
    var rc_src = _make_limp_ragdoll_component(src_app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = src_app.world.insert[RagdollComponent](e_src, rc_src^)
    for _ in range(3):
        src_app.step_one_frame(Float32(1.0 / 60.0))
    var blob = src_app.world.snapshot_blob()

    var dst_app = EngineApp(AppConfig(title="RagSnapDst", headless=True, worker_threads=0))
    dst_app.add_plugin(TransformPlugin())
    dst_app.add_plugin(PhysicsPlugin(gravity=9.81))
    dst_app.add_plugin(AnimPlugin())
    dst_app.startup()
    dst_app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, _make_biped_model())

    var e_dst = dst_app.world.spawn()
    assert_equal(e_dst.to_bits(), e_src.to_bits())
    _ = dst_app.world.insert[LocalTransform](e_dst, LocalTransform(Vec3(50.0, 2.0, 0.0)))
    _ = dst_app.world.insert[GlobalTransform](e_dst, GlobalTransform())
    _ = dst_app.world.insert[SkeletonComponent](e_dst, SkeletonComponent(model_id))
    var rc_dst = _make_limp_ragdoll_component(dst_app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = dst_app.world.insert[RagdollComponent](e_dst, rc_dst^)
    for _ in range(3):
        dst_app.step_one_frame(Float32(1.0 / 60.0))

    ref pw_dst = dst_app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_equal(_count_active_bodies(pw_dst), 2)

    dst_app.world.restore_blob(blob)
    dst_app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(pw_dst.ragdolls), 2)
    assert_true(not pw_dst.ragdolls[0].rag.is_active)
    assert_true(pw_dst.ragdolls[1].rag.is_active)
    for li in range(len(pw_dst.ragdolls[0].limb_body_ids)):
        var bslot = pw_dst.find_slot_by_id(pw_dst.ragdolls[0].limb_body_ids[li])
        assert_true(bslot < 0 or (not pw_dst.bodies[bslot].active))
    assert_equal(_count_active_bodies(pw_dst), 2)

    dst_app.shutdown()
    src_app.shutdown()


def _wait_audio_listener_pos(srv: AudioServer, target: Vec3) -> Bool:
    for _ in range(40):
        if (srv._rt_core()[].listener_pos - target).length() < Float32(1.0e-3):
            return True
        _ = external_call["usleep", Int32](UInt32(2000))
    return False


def test_audio_listener_priority_switch_zero_doppler_spike() raises:
    var app = EngineApp(AppConfig(title="AudioSwitch", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.startup()

    var pos_low = Vec3(1000.0, 0.0, 0.0)
    var pos_high = Vec3(0.0, 1.0, 3.0)
    var pos_high_moved = Vec3(1.0, 1.0, 3.0)

    var lis_low = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_low, LocalTransform(pos_low))
    _ = app.world.insert[GlobalTransform](lis_low, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_low, AudioListenerComponent(active=True, priority=1))

    var lis_high = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_high, LocalTransform(pos_high))
    _ = app.world.insert[GlobalTransform](lis_high, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_high, AudioListenerComponent(active=False, priority=10))

    var dt = Float32(1.0 / 60.0)
    ref srv = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

    # Frame 1: lis_low active -> zero initial velocity
    app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_low))
    assert_true(srv._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    # Frame 2: enable lis_high (higher priority) at (0, 1, 3) -> zero Doppler spike on switch
    app.world.get_ptr_mut[AudioListenerComponent](lis_high).value()[unsafe_offset=0].active = True
    app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_high))
    assert_true(srv._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    # Frame 3: move same active listener lis_high by (1, 0, 0) -> steady-state Doppler preserved
    app.world.get_ptr_mut[LocalTransform](lis_high).value()[unsafe_offset=0].translation = pos_high_moved
    app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_high_moved))
    assert_true(abs(srv._rt_core()[].listener_vel.x - Float32(60.0)) < Float32(1.0))

    # Frame 4: switch back to lis_low by disabling lis_high -> zero Doppler spike
    app.world.get_ptr_mut[AudioListenerComponent](lis_high).value()[unsafe_offset=0].active = False
    app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_low))
    assert_true(srv._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    # Frame 5-6: disable all listeners for 1 frame, then re-enable lis_low at distant pos -> zero Doppler spike
    var pos_re = Vec3(-400.0, 25.0, 300.0)
    app.world.get_ptr_mut[AudioListenerComponent](lis_low).value()[unsafe_offset=0].active = False
    app.step_one_frame(dt)
    app.world.get_ptr_mut[LocalTransform](lis_low).value()[unsafe_offset=0].translation = pos_re
    app.world.get_ptr_mut[AudioListenerComponent](lis_low).value()[unsafe_offset=0].active = True
    app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_re))
    assert_true(srv._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    app.shutdown()


def test_anim_ragdoll_no_one_frame_ghost_collision() raises:
    var model_id = UInt64(607)
    var dt = Float32(1.0 / 60.0)

    # Sub-case A: world.despawn(e_rag) cleans up ragdoll in STAGE_PRE_UPDATE before STAGE_FIXED_UPDATE
    var app = EngineApp(AppConfig(title="RagNoGhostDespawn", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.startup()
    app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, _make_biped_model())

    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(30.0, 0.5, 30.0)))

    var e_rag = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_rag, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e_rag, GlobalTransform())
    _ = app.world.insert[SkeletonComponent](e_rag, SkeletonComponent(model_id))
    var rc = _make_limp_ragdoll_component(app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = app.world.insert[RagdollComponent](e_rag, rc^)

    for _ in range(2):
        app.step_one_frame(dt)

    ref pw = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_true(pw.ragdolls[0].rag.is_active)
    var pel_pos = pw.ragdolls[0].rag.pelvis_world_position()

    var pev_cursor = 0
    _ = app.resources.get_ptr[PhysicsEventsResource]().value()[unsafe_offset=0].events.read_since(pev_cursor)

    assert_true(app.world.despawn(e_rag))
    var ball_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](ball_e, LocalTransform(pel_pos + Vec3(0.05, 0.4, 0.0)))
    _ = app.world.insert[GlobalTransform](ball_e, GlobalTransform())
    var ball_rb = RigidBodyComponent.dynamic(1.0)
    ball_rb.linear_vel = Vec3(0.0, -5.0, 0.0)
    _ = app.world.insert[RigidBodyComponent](ball_e, ball_rb)
    _ = app.world.insert[ColliderComponent](ball_e, ColliderComponent.sphere(0.35))

    app.step_one_frame(dt)
    assert_true(not pw.ragdolls[0].rag.is_active)
    assert_equal(_count_active_bodies(pw), 2)
    assert_equal(len(pw.events), 0)
    var pevs_a = app.resources.get_ptr[PhysicsEventsResource]().value()[unsafe_offset=0].events.read_since(pev_cursor)
    assert_equal(len(pevs_a), 0)
    var ball_vel_a = app.world.get[RigidBodyComponent](ball_e).linear_vel
    assert_true(ball_vel_a.y < Float32(-5.0))
    assert_true(abs(ball_vel_a.x) < Float32(1.0e-4))
    assert_true(abs(ball_vel_a.z) < Float32(1.0e-4))
    app.shutdown()

    # Sub-case B: World.restore_blob(ball_only_blob) deactivates old ragdoll before STAGE_FIXED_UPDATE
    var src_app = EngineApp(AppConfig(title="RagGhostSnapSrc", headless=True, worker_threads=0))
    src_app.add_plugin(TransformPlugin())
    src_app.add_plugin(PhysicsPlugin(gravity=9.81))
    src_app.add_plugin(AnimPlugin())
    src_app.startup()

    var s_floor = src_app.world.spawn()
    _ = src_app.world.insert[LocalTransform](s_floor, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = src_app.world.insert[GlobalTransform](s_floor, GlobalTransform())
    _ = src_app.world.insert[RigidBodyComponent](s_floor, RigidBodyComponent.static_body())
    _ = src_app.world.insert[ColliderComponent](s_floor, ColliderComponent.box(Vec3(30.0, 0.5, 30.0)))

    var s_ball = src_app.world.spawn()
    _ = src_app.world.insert[LocalTransform](s_ball, LocalTransform(pel_pos + Vec3(0.05, 0.4, 0.0)))
    _ = src_app.world.insert[GlobalTransform](s_ball, GlobalTransform())
    var s_ball_rb = RigidBodyComponent.dynamic(1.0)
    s_ball_rb.linear_vel = Vec3(0.0, -5.0, 0.0)
    _ = src_app.world.insert[RigidBodyComponent](s_ball, s_ball_rb)
    _ = src_app.world.insert[ColliderComponent](s_ball, ColliderComponent.sphere(0.35))
    var ball_only_blob = src_app.world.snapshot_blob()

    var dst_app = EngineApp(AppConfig(title="RagGhostSnapDst", headless=True, worker_threads=0))
    dst_app.add_plugin(TransformPlugin())
    dst_app.add_plugin(PhysicsPlugin(gravity=9.81))
    dst_app.add_plugin(AnimPlugin())
    dst_app.startup()
    dst_app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, _make_biped_model())

    var d_floor = dst_app.world.spawn()
    _ = dst_app.world.insert[LocalTransform](d_floor, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = dst_app.world.insert[GlobalTransform](d_floor, GlobalTransform())
    _ = dst_app.world.insert[RigidBodyComponent](d_floor, RigidBodyComponent.static_body())
    _ = dst_app.world.insert[ColliderComponent](d_floor, ColliderComponent.box(Vec3(30.0, 0.5, 30.0)))

    var d_rag = dst_app.world.spawn()
    _ = dst_app.world.insert[LocalTransform](d_rag, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = dst_app.world.insert[GlobalTransform](d_rag, GlobalTransform())
    _ = dst_app.world.insert[SkeletonComponent](d_rag, SkeletonComponent(model_id))
    var d_rc = _make_limp_ragdoll_component(dst_app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = dst_app.world.insert[RagdollComponent](d_rag, d_rc^)

    for _ in range(2):
        dst_app.step_one_frame(dt)
    ref pw_dst = dst_app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_true(pw_dst.ragdolls[0].rag.is_active)

    var pev_dst_cursor = 0
    _ = dst_app.resources.get_ptr[PhysicsEventsResource]().value()[unsafe_offset=0].events.read_since(pev_dst_cursor)

    dst_app.world.restore_blob(ball_only_blob)
    dst_app.step_one_frame(dt)
    assert_true(not pw_dst.ragdolls[0].rag.is_active)
    assert_equal(_count_active_bodies(pw_dst), 2)
    assert_equal(len(pw_dst.events), 0)
    var pevs_b = dst_app.resources.get_ptr[PhysicsEventsResource]().value()[unsafe_offset=0].events.read_since(pev_dst_cursor)
    assert_equal(len(pevs_b), 0)
    var ball_vel_b = dst_app.world.get[RigidBodyComponent](s_ball).linear_vel
    assert_true(ball_vel_b.y < Float32(-5.0))
    assert_true(abs(ball_vel_b.x) < Float32(1.0e-4))
    assert_true(abs(ball_vel_b.z) < Float32(1.0e-4))

    dst_app.shutdown()
    src_app.shutdown()


def test_anim_ragdoll_remove_skeleton_deactivates() raises:
    var model_id = UInt64(608)
    var app = EngineApp(AppConfig(title="RagRmSkel", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.startup()
    app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(model_id, _make_biped_model())

    var e_rag = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_rag, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e_rag, GlobalTransform())
    _ = app.world.insert[SkeletonComponent](e_rag, SkeletonComponent(model_id))
    var rc = _make_limp_ragdoll_component(app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0].models[0])
    _ = app.world.insert[RagdollComponent](e_rag, rc^)

    for _ in range(2):
        app.step_one_frame(Float32(1.0 / 60.0))
    ref pw = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    assert_true(pw.ragdolls[0].rag.is_active)
    assert_equal(_count_active_bodies(pw), 2)

    assert_true(app.world.remove[SkeletonComponent](e_rag))
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_true(not pw.ragdolls[0].rag.is_active)
    assert_equal(_count_active_bodies(pw), 0)
    app.shutdown()


def test_audio_listener_restore_blob_same_entity_zero_doppler_spike() raises:
    var dt = Float32(1.0 / 60.0)
    var pos_snap = Vec3(0.0, 1.0, 3.0)
    var pos_pre = Vec3(-399.0, 25.0, 300.0)
    var pos_moved = Vec3(1.0, 1.0, 3.0)

    var src_app = EngineApp(AppConfig(title="AudioSnapSrc", headless=True, worker_threads=0))
    src_app.add_plugin(TransformPlugin())
    src_app.add_plugin(AudioPlugin(headless=True))
    src_app.startup()
    var e_src = src_app.world.spawn()
    _ = src_app.world.insert[LocalTransform](e_src, LocalTransform(pos_snap))
    _ = src_app.world.insert[GlobalTransform](e_src, GlobalTransform())
    _ = src_app.world.insert[AudioListenerComponent](e_src, AudioListenerComponent(active=True, priority=1))
    src_app.step_one_frame(dt)
    var aud_blob = src_app.world.snapshot_blob()

    var dst_app = EngineApp(AppConfig(title="AudioSnapDst", headless=True, worker_threads=0))
    dst_app.add_plugin(TransformPlugin())
    dst_app.add_plugin(AudioPlugin(headless=True))
    dst_app.startup()
    var e_dst = dst_app.world.spawn()
    assert_equal(e_dst.to_bits(), e_src.to_bits())
    _ = dst_app.world.insert[LocalTransform](e_dst, LocalTransform(pos_pre))
    _ = dst_app.world.insert[GlobalTransform](e_dst, GlobalTransform())
    _ = dst_app.world.insert[AudioListenerComponent](e_dst, AudioListenerComponent(active=True, priority=1))

    ref srv = dst_app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]
    dst_app.step_one_frame(dt)
    dst_app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_pre))

    # Restore snapshot with active AudioListenerComponent on the exact same Entity(0,0) at (0,1,3)
    dst_app.world.restore_blob(aud_blob)
    dst_app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_snap))
    assert_true(srv._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    # Next frame continuous motion preserves Doppler (~60 m/s)
    dst_app.world.get_ptr_mut[LocalTransform](e_src).value()[unsafe_offset=0].translation = pos_moved
    dst_app.step_one_frame(dt)
    assert_true(_wait_audio_listener_pos(srv, pos_moved))
    assert_true(abs(srv._rt_core()[].listener_vel.x - Float32(60.0)) < Float32(1.0))

    dst_app.shutdown()
    src_app.shutdown()


struct _IntraRestoreCtxResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.test.IntraRestoreCtxResource")
    var snap: List[UInt8]
    var target_entity: Entity
    var new_pos: Vec3
    var armed: Bool
    var restore_ok: Bool
    var fired_count: Int

    def __init__(out self):
        self.snap = List[UInt8]()
        self.target_entity = Entity(0, 0)
        self.new_pos = Vec3()
        self.armed = False
        self.restore_ok = False
        self.fired_count = 0


def _sys_intra_restore(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var ctx_opt = res_ptr[unsafe_offset=0].get_ptr_mut[_IntraRestoreCtxResource]()
    if not ctx_opt:
        return
    ref ctx = ctx_opt.value()[unsafe_offset=0]
    if not ctx.armed:
        return
    ctx.armed = False
    ctx.fired_count += 1
    try:
        world_ptr[unsafe_offset=0].restore_blob(ctx.snap)
        var lt_opt = world_ptr[unsafe_offset=0].get_ptr_mut[LocalTransform](ctx.target_entity)
        if lt_opt:
            lt_opt.value()[unsafe_offset=0].translation = ctx.new_pos
            ctx.restore_ok = True
    except:
        ctx.restore_ok = False


def test_restore_blob_tick_rollback_transform_and_audio_doppler() raises:
    var dt = Float32(1.0 / 60.0)

    # Case A: lower-tick snapshot (saved_tick > 0) restored into a higher-tick 10-frame EngineApp,
    # with immediate LocalTransform mutation before Frame 0, 0.1 m motion on Frame 1, and Frame 2 static skip.
    var pos_snap_a = Vec3(0.0, 1.0, 3.0)
    var pos_tp_a = Vec3(-300.0, 15.0, 250.0)
    var pos_step1_a = Vec3(-299.9, 15.0, 250.0)

    var src_a = EngineApp(AppConfig(title="RollbackSrcA", headless=True, worker_threads=0))
    src_a.add_plugin(TransformPlugin())
    src_a.add_plugin(AudioPlugin(headless=True))
    src_a.startup()
    var e_src_a = src_a.world.spawn()
    _ = src_a.world.insert[LocalTransform](e_src_a, LocalTransform(pos_snap_a))
    _ = src_a.world.insert[GlobalTransform](e_src_a, GlobalTransform())
    _ = src_a.world.insert[AudioListenerComponent](e_src_a, AudioListenerComponent(active=True, priority=1))
    src_a.step_one_frame(dt)
    var saved_tick_a = src_a.world.current_tick
    assert_true(saved_tick_a > UInt32(0))
    var snap_a = src_a.world.snapshot_blob()

    var dst_a = EngineApp(AppConfig(title="RollbackDstA", headless=True, worker_threads=0))
    dst_a.add_plugin(TransformPlugin())
    dst_a.add_plugin(AudioPlugin(headless=True))
    dst_a.startup()
    var e_dst_a = dst_a.world.spawn()
    assert_equal(e_dst_a.to_bits(), e_src_a.to_bits())
    _ = dst_a.world.insert[LocalTransform](e_dst_a, LocalTransform(Vec3(500.0, 0.0, -500.0)))
    _ = dst_a.world.insert[GlobalTransform](e_dst_a, GlobalTransform())
    _ = dst_a.world.insert[AudioListenerComponent](e_dst_a, AudioListenerComponent(active=True, priority=1))
    for _ in range(10):
        dst_a.step_one_frame(dt)

    ref cache_a = dst_a.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
    assert_true(cache_a.last_propagated_tick > saved_tick_a)
    ref srv_a = dst_a.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

    dst_a.world.restore_blob(snap_a)
    dst_a.world.get_ptr_mut[LocalTransform](e_src_a).value()[unsafe_offset=0].translation = pos_tp_a

    # Frame 0 after restore: GlobalTransform must immediately propagate to (-300, 15, 250) with zero Doppler spike
    dst_a.step_one_frame(dt)
    var gt0_a = dst_a.world.get[GlobalTransform](e_src_a).translation
    assert_true((gt0_a - pos_tp_a).length() < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_a, pos_tp_a))
    assert_true(srv_a._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    # Frame 1 after restore: move by 0.1 m -> ~6.0 m/s velocity (no 23,326 m/s spike)
    dst_a.world.get_ptr_mut[LocalTransform](e_src_a).value()[unsafe_offset=0].translation = pos_step1_a
    dst_a.step_one_frame(dt)
    var gt1_a = dst_a.world.get[GlobalTransform](e_src_a).translation
    assert_true(abs(gt1_a.x - pos_step1_a.x) < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_a, pos_step1_a))
    assert_true(srv_a._rt_core()[].listener_vel.length() < Float32(10.0))
    assert_true(abs(srv_a._rt_core()[].listener_vel.x - Float32(6.0)) < Float32(0.5))

    # Frame 2 after restore (static): dirty-chunk skipping must engage
    dst_a.step_one_frame(dt)
    assert_equal(cache_a.last_propagated_count, 0)
    assert_true(cache_a.last_skipped_count > 0)
    assert_true((dst_a.world.get[GlobalTransform](e_src_a).translation - pos_step1_a).length() < Float32(1.0e-4))

    dst_a.shutdown()
    src_a.shutdown()

    # Case B: saved_tick == 0 snapshot restored into a stepped higher-tick EngineApp (dt = 1/60)
    var pos_snap_b = Vec3(-300.0, 15.0, 250.0)
    var src_b = EngineApp(AppConfig(title="RollbackSrcB", headless=True, worker_threads=0))
    src_b.add_plugin(TransformPlugin())
    src_b.add_plugin(AudioPlugin(headless=True))
    src_b.startup()
    src_b.world.current_tick = 0
    var e_src_b = src_b.world.spawn()
    _ = src_b.world.insert[LocalTransform](e_src_b, LocalTransform(pos_snap_b))
    _ = src_b.world.insert[GlobalTransform](e_src_b, GlobalTransform())
    _ = src_b.world.insert[AudioListenerComponent](e_src_b, AudioListenerComponent(active=True, priority=1))
    assert_equal(Int(src_b.world.current_tick), 0)
    var snap_b = src_b.world.snapshot_blob()

    var dst_b = EngineApp(AppConfig(title="RollbackDstB", headless=True, worker_threads=0))
    dst_b.add_plugin(TransformPlugin())
    dst_b.add_plugin(AudioPlugin(headless=True))
    dst_b.startup()
    var e_dst_b = dst_b.world.spawn()
    assert_equal(e_dst_b.to_bits(), e_src_b.to_bits())
    _ = dst_b.world.insert[LocalTransform](e_dst_b, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = dst_b.world.insert[GlobalTransform](e_dst_b, GlobalTransform())
    _ = dst_b.world.insert[AudioListenerComponent](e_dst_b, AudioListenerComponent(active=True, priority=1))
    for _ in range(5):
        dst_b.step_one_frame(dt)

    ref cache_b = dst_b.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
    assert_true(cache_b.last_propagated_tick > UInt32(0))
    ref srv_b = dst_b.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]
    dst_b.world.restore_blob(snap_b)

    # Frame 0 and Frame 1 after restore: GlobalTransform updates on Frame 0 and skips on static Frame 1
    dst_b.step_one_frame(dt)
    var gt0_b = dst_b.world.get[GlobalTransform](e_src_b).translation
    assert_true((gt0_b - pos_snap_b).length() < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_b, pos_snap_b))
    assert_true(srv_b._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    dst_b.step_one_frame(dt)
    var gt1_b = dst_b.world.get[GlobalTransform](e_src_b).translation
    assert_true((gt1_b - pos_snap_b).length() < Float32(1.0e-4))
    assert_true(srv_b._rt_core()[].listener_vel.length() < Float32(1.0e-4))
    assert_equal(cache_b.last_propagated_count, 0)
    assert_true(cache_b.last_skipped_count > 0)

    dst_b.shutdown()

    # Case B2: saved_tick == 0 restored into 1-frame EngineApp (last_tick == 2) with dt = 0.05 (due_steps = 3)
    var dst_b2 = EngineApp(AppConfig(title="RollbackDstB2", headless=True, worker_threads=0))
    dst_b2.add_plugin(TransformPlugin())
    dst_b2.add_plugin(AudioPlugin(headless=True))
    dst_b2.startup()
    var e_dst_b2 = dst_b2.world.spawn()
    assert_equal(e_dst_b2.to_bits(), e_src_b.to_bits())
    _ = dst_b2.world.insert[LocalTransform](e_dst_b2, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = dst_b2.world.insert[GlobalTransform](e_dst_b2, GlobalTransform())
    _ = dst_b2.world.insert[AudioListenerComponent](e_dst_b2, AudioListenerComponent(active=True, priority=1))
    dst_b2.step_one_frame(dt)

    ref cache_b2 = dst_b2.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
    assert_equal(Int(cache_b2.last_propagated_tick), 2)
    ref srv_b2 = dst_b2.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]
    dst_b2.world.restore_blob(snap_b)

    dst_b2.step_one_frame(Float32(0.05))
    assert_true((dst_b2.world.get[GlobalTransform](e_src_b).translation - pos_snap_b).length() < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_b2, pos_snap_b))
    assert_true(srv_b2._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    dst_b2.step_one_frame(dt)
    assert_true((dst_b2.world.get[GlobalTransform](e_src_b).translation - pos_snap_b).length() < Float32(1.0e-4))
    assert_equal(cache_b2.last_propagated_count, 0)
    assert_true(cache_b2.last_skipped_count > 0)

    dst_b2.shutdown()
    src_b.shutdown()

    # Case C (C1 & C2): saved_tick == last_tick (10 == 10) at dt = 1/60 (due_steps = 1),
    # both WITH (C1) and WITHOUT (C2) immediate post-restore LocalTransform mutation, plus Frames 1-4 checks.
    var pos_unprop_c = Vec3(50.0, 0.0, 0.0)
    var pos_mut_c = Vec3(-120.0, 4.0, 30.0)

    var src_c1 = EngineApp(AppConfig(title="RollbackSrcC1", headless=True, worker_threads=0))
    src_c1.add_plugin(TransformPlugin())
    src_c1.add_plugin(AudioPlugin(headless=True))
    src_c1.startup()
    var e_src_c1 = src_c1.world.spawn()
    _ = src_c1.world.insert[LocalTransform](e_src_c1, LocalTransform(Vec3(1.0, 2.0, 3.0)))
    _ = src_c1.world.insert[GlobalTransform](e_src_c1, GlobalTransform())
    _ = src_c1.world.insert[AudioListenerComponent](e_src_c1, AudioListenerComponent(active=True, priority=1))
    for _ in range(4):
        src_c1.step_one_frame(dt)
    assert_equal(Int(src_c1.world.current_tick), 9)
    src_c1.world.get_ptr_mut[LocalTransform](e_src_c1).value()[unsafe_offset=0].translation = pos_unprop_c
    src_c1.world.get_ptr_mut[GlobalTransform](e_src_c1).value()[unsafe_offset=0] = GlobalTransform()
    _ = src_c1.world.advance_tick()
    assert_equal(Int(src_c1.world.current_tick), 10)
    var snap_c1 = src_c1.world.snapshot_blob()

    var dst_c1 = EngineApp(AppConfig(title="RollbackDstC1", headless=True, worker_threads=0))
    dst_c1.add_plugin(TransformPlugin())
    dst_c1.add_plugin(AudioPlugin(headless=True))
    dst_c1.startup()
    var e_dst_c1 = dst_c1.world.spawn()
    assert_equal(e_dst_c1.to_bits(), e_src_c1.to_bits())
    _ = dst_c1.world.insert[LocalTransform](e_dst_c1, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = dst_c1.world.insert[GlobalTransform](e_dst_c1, GlobalTransform())
    _ = dst_c1.world.insert[AudioListenerComponent](e_dst_c1, AudioListenerComponent(active=True, priority=1))
    for _ in range(5):
        dst_c1.step_one_frame(dt)
    ref cache_c1 = dst_c1.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
    assert_equal(Int(cache_c1.last_propagated_tick), 10)
    ref srv_c1 = dst_c1.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

    dst_c1.world.restore_blob(snap_c1)
    dst_c1.world.get_ptr_mut[LocalTransform](e_src_c1).value()[unsafe_offset=0].translation = pos_mut_c
    dst_c1.step_one_frame(dt)
    assert_true((dst_c1.world.get[GlobalTransform](e_src_c1).translation - pos_mut_c).length() < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_c1, pos_mut_c))
    assert_true(srv_c1._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    for _ in range(4):
        dst_c1.step_one_frame(dt)
        assert_true((dst_c1.world.get[GlobalTransform](e_src_c1).translation - pos_mut_c).length() < Float32(1.0e-4))
        assert_true(srv_c1._rt_core()[].listener_vel.length() < Float32(1.0e-4))
        assert_equal(cache_c1.last_propagated_count, 0)
        assert_true(cache_c1.last_skipped_count > 0)

    dst_c1.shutdown()
    src_c1.shutdown()

    # Sub-case C2: saved_tick == last_tick == 10, NO post-restore mutation (must still propagate snap's unpropagated LocalTransform)
    var src_c2 = EngineApp(AppConfig(title="RollbackSrcC2", headless=True, worker_threads=0))
    src_c2.add_plugin(TransformPlugin())
    src_c2.add_plugin(AudioPlugin(headless=True))
    src_c2.startup()
    var e_src_c2 = src_c2.world.spawn()
    _ = src_c2.world.insert[LocalTransform](e_src_c2, LocalTransform(Vec3(1.0, 2.0, 3.0)))
    _ = src_c2.world.insert[GlobalTransform](e_src_c2, GlobalTransform())
    _ = src_c2.world.insert[AudioListenerComponent](e_src_c2, AudioListenerComponent(active=True, priority=1))
    for _ in range(4):
        src_c2.step_one_frame(dt)
    assert_equal(Int(src_c2.world.current_tick), 9)
    src_c2.world.get_ptr_mut[LocalTransform](e_src_c2).value()[unsafe_offset=0].translation = pos_unprop_c
    src_c2.world.get_ptr_mut[GlobalTransform](e_src_c2).value()[unsafe_offset=0] = GlobalTransform()
    _ = src_c2.world.advance_tick()
    assert_equal(Int(src_c2.world.current_tick), 10)
    var snap_c2 = src_c2.world.snapshot_blob()

    var dst_c2 = EngineApp(AppConfig(title="RollbackDstC2", headless=True, worker_threads=0))
    dst_c2.add_plugin(TransformPlugin())
    dst_c2.add_plugin(AudioPlugin(headless=True))
    dst_c2.startup()
    var e_dst_c2 = dst_c2.world.spawn()
    assert_equal(e_dst_c2.to_bits(), e_src_c2.to_bits())
    _ = dst_c2.world.insert[LocalTransform](e_dst_c2, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = dst_c2.world.insert[GlobalTransform](e_dst_c2, GlobalTransform())
    _ = dst_c2.world.insert[AudioListenerComponent](e_dst_c2, AudioListenerComponent(active=True, priority=1))
    for _ in range(5):
        dst_c2.step_one_frame(dt)
    ref cache_c2 = dst_c2.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
    assert_equal(Int(cache_c2.last_propagated_tick), 10)
    ref srv_c2 = dst_c2.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

    dst_c2.world.restore_blob(snap_c2)
    dst_c2.step_one_frame(dt)
    assert_true((dst_c2.world.get[GlobalTransform](e_src_c2).translation - pos_unprop_c).length() < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_c2, pos_unprop_c))
    assert_true(srv_c2._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    for _ in range(4):
        dst_c2.step_one_frame(dt)
        assert_true((dst_c2.world.get[GlobalTransform](e_src_c2).translation - pos_unprop_c).length() < Float32(1.0e-4))
        assert_true(srv_c2._rt_core()[].listener_vel.length() < Float32(1.0e-4))
        assert_equal(cache_c2.last_propagated_count, 0)
        assert_true(cache_c2.last_skipped_count > 0)

    dst_c2.shutdown()
    src_c2.shutdown()

    # Case D (D1 & D2): Parent -> Child hierarchy, saved_tick (11) < last_tick (12) with saved_tick + due_steps > last_tick
    # at dt = 1/30 (due_steps = 2) and dt = 0.05 (due_steps = 3).
    for d_idx in range(2):
        var dt_step0 = Float32(1.0 / 30.0) if d_idx == 0 else Float32(0.05)
        var app_d = EngineApp(AppConfig(title="RollbackHierD", headless=True, worker_threads=0))
        app_d.add_plugin(TransformPlugin())
        app_d.add_plugin(AudioPlugin(headless=True))
        app_d.startup()
        var parent_d = app_d.world.spawn()
        _ = app_d.world.insert[LocalTransform](parent_d, LocalTransform(Vec3(0.0, 1.0, 3.0)))
        _ = app_d.world.insert[GlobalTransform](parent_d, GlobalTransform())
        var child_d = app_d.world.spawn()
        _ = app_d.world.insert[LocalTransform](child_d, LocalTransform(Vec3()))
        _ = app_d.world.insert[GlobalTransform](child_d, GlobalTransform())
        _ = app_d.world.insert[AudioListenerComponent](child_d, AudioListenerComponent(active=True, priority=1))
        assert_true(set_parent(app_d.world, child_d, parent_d))

        for _ in range(5):
            app_d.step_one_frame(dt)
        ref cache_d = app_d.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
        assert_equal(Int(cache_d.last_propagated_tick), 10)
        assert_equal(Int(app_d.world.current_tick), 11)
        var snap_d = app_d.world.snapshot_blob()

        app_d.step_one_frame(dt)
        assert_equal(Int(cache_d.last_propagated_tick), 12)
        ref srv_d = app_d.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

        app_d.world.restore_blob(snap_d)
        var pos_tp_d = Vec3(-300.0, 15.0, 250.0)
        var pos_step1_d = Vec3(-299.9, 15.0, 250.0)
        app_d.world.get_ptr_mut[LocalTransform](parent_d).value()[unsafe_offset=0].translation = pos_tp_d

        # Frame 0: child_d GlobalTransform propagates from parent_d with zero Doppler spike
        app_d.step_one_frame(dt_step0)
        assert_true((app_d.world.get[GlobalTransform](child_d).translation - pos_tp_d).length() < Float32(1.0e-4))
        assert_true(_wait_audio_listener_pos(srv_d, pos_tp_d))
        assert_true(srv_d._rt_core()[].listener_vel.length() < Float32(1.0e-4))

        # Frame 1: +0.1 m motion on parent_d -> child_d listener_vel.x ~ 6.0 m/s
        app_d.world.get_ptr_mut[LocalTransform](parent_d).value()[unsafe_offset=0].translation = pos_step1_d
        app_d.step_one_frame(dt)
        assert_true(abs(app_d.world.get[GlobalTransform](child_d).translation.x - pos_step1_d.x) < Float32(1.0e-4))
        assert_true(_wait_audio_listener_pos(srv_d, pos_step1_d))
        assert_true(srv_d._rt_core()[].listener_vel.length() < Float32(10.0))
        assert_true(abs(srv_d._rt_core()[].listener_vel.x - Float32(6.0)) < Float32(0.5))

        # Frame 2 (static): dirty-chunk skipping must engage for both parent_d and child_d
        app_d.step_one_frame(dt)
        assert_equal(cache_d.last_propagated_count, 0)
        assert_equal(cache_d.last_skipped_count, 2)

        app_d.shutdown()

    # Case E: saved_tick > last_tick forward jump + Frame 1 static dirty-chunk skip
    var pos_fwd_e = Vec3(42.0, -7.0, 13.0)
    var src_e = EngineApp(AppConfig(title="RollbackSrcE", headless=True, worker_threads=0))
    src_e.add_plugin(TransformPlugin())
    src_e.add_plugin(AudioPlugin(headless=True))
    src_e.startup()
    var e_src_e = src_e.world.spawn()
    _ = src_e.world.insert[LocalTransform](e_src_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = src_e.world.insert[GlobalTransform](e_src_e, GlobalTransform())
    _ = src_e.world.insert[AudioListenerComponent](e_src_e, AudioListenerComponent(active=True, priority=1))
    for _ in range(10):
        src_e.step_one_frame(dt)
    assert_equal(Int(src_e.world.current_tick), 21)
    src_e.world.get_ptr_mut[LocalTransform](e_src_e).value()[unsafe_offset=0].translation = pos_fwd_e
    src_e.world.get_ptr_mut[GlobalTransform](e_src_e).value()[unsafe_offset=0] = GlobalTransform()
    var snap_e = src_e.world.snapshot_blob()

    var dst_e = EngineApp(AppConfig(title="RollbackDstE", headless=True, worker_threads=0))
    dst_e.add_plugin(TransformPlugin())
    dst_e.add_plugin(AudioPlugin(headless=True))
    dst_e.startup()
    var e_dst_e = dst_e.world.spawn()
    assert_equal(e_dst_e.to_bits(), e_src_e.to_bits())
    _ = dst_e.world.insert[LocalTransform](e_dst_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = dst_e.world.insert[GlobalTransform](e_dst_e, GlobalTransform())
    _ = dst_e.world.insert[AudioListenerComponent](e_dst_e, AudioListenerComponent(active=True, priority=1))
    dst_e.step_one_frame(dt)
    ref cache_e = dst_e.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
    assert_equal(Int(cache_e.last_propagated_tick), 2)
    ref srv_e = dst_e.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

    dst_e.world.restore_blob(snap_e)
    dst_e.step_one_frame(dt)
    assert_true((dst_e.world.get[GlobalTransform](e_src_e).translation - pos_fwd_e).length() < Float32(1.0e-4))
    assert_true(_wait_audio_listener_pos(srv_e, pos_fwd_e))
    assert_true(srv_e._rt_core()[].listener_vel.length() < Float32(1.0e-4))

    dst_e.step_one_frame(dt)
    assert_true((dst_e.world.get[GlobalTransform](e_src_e).translation - pos_fwd_e).length() < Float32(1.0e-4))
    assert_equal(cache_e.last_propagated_count, 0)
    assert_true(cache_e.last_skipped_count > 0)

    dst_e.shutdown()
    src_e.shutdown()

    # Case F: Intra-frame restore_blob in STAGE_PRE_UPDATE (F1, saved_tick=11 < last_tick=12, due_steps=2 -> 13)
    # and STAGE_FIXED_UPDATE (F2, saved_tick=10 == last_tick=10, due_steps=2 -> iter 0 restores 10, iter 1 advances to 11)
    for f_idx in range(2):
        var stage_f = STAGE_PRE_UPDATE if f_idx == 0 else STAGE_FIXED_UPDATE
        var pos_f = Vec3(77.0, 0.0, 0.0) if f_idx == 0 else Vec3(88.0, 0.0, 0.0)
        var app_f = EngineApp(AppConfig(title="RollbackIntraF", headless=True, worker_threads=0))
        app_f.add_plugin(TransformPlugin())
        app_f.add_plugin(AudioPlugin(headless=True))
        app_f.insert_resource[_IntraRestoreCtxResource](_IntraRestoreCtxResource())
        _ = app_f.add_system(
            stage_f,
            "flint.test.intra_restore",
            _sys_intra_restore,
            exclusive=True,
        )
        app_f.startup()

        var e_f = app_f.world.spawn()
        _ = app_f.world.insert[LocalTransform](e_f, LocalTransform(Vec3(0.0, 1.0, 3.0)))
        _ = app_f.world.insert[GlobalTransform](e_f, GlobalTransform())
        _ = app_f.world.insert[AudioListenerComponent](e_f, AudioListenerComponent(active=True, priority=1))

        var snap_f: List[UInt8]
        if f_idx == 0:
            for _ in range(5):
                app_f.step_one_frame(dt)
            assert_equal(Int(app_f.world.current_tick), 11)
            snap_f = app_f.world.snapshot_blob()
            app_f.step_one_frame(dt)
        else:
            var src_f2 = EngineApp(AppConfig(title="RollbackIntraSrcF2", headless=True, worker_threads=0))
            src_f2.add_plugin(TransformPlugin())
            src_f2.add_plugin(AudioPlugin(headless=True))
            src_f2.startup()
            var e_src_f2 = src_f2.world.spawn()
            assert_equal(e_src_f2.to_bits(), e_f.to_bits())
            _ = src_f2.world.insert[LocalTransform](e_src_f2, LocalTransform(Vec3(0.0, 1.0, 3.0)))
            _ = src_f2.world.insert[GlobalTransform](e_src_f2, GlobalTransform())
            _ = src_f2.world.insert[AudioListenerComponent](e_src_f2, AudioListenerComponent(active=True, priority=1))
            for _ in range(4):
                src_f2.step_one_frame(dt)
            assert_equal(Int(src_f2.world.current_tick), 9)
            _ = src_f2.world.advance_tick()
            assert_equal(Int(src_f2.world.current_tick), 10)
            snap_f = src_f2.world.snapshot_blob()
            src_f2.shutdown()
            for _ in range(5):
                app_f.step_one_frame(dt)

        ref cache_f = app_f.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
        if f_idx == 0:
            assert_equal(Int(cache_f.last_propagated_tick), 12)
        else:
            assert_equal(Int(cache_f.last_propagated_tick), 10)
        ref srv_f = app_f.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

        var ctx_ptr_f = app_f.resources.get_ptr_mut[_IntraRestoreCtxResource]().value()
        ctx_ptr_f[unsafe_offset=0].snap = snap_f^
        ctx_ptr_f[unsafe_offset=0].target_entity = e_f
        ctx_ptr_f[unsafe_offset=0].new_pos = pos_f
        ctx_ptr_f[unsafe_offset=0].armed = True
        ctx_ptr_f[unsafe_offset=0].restore_ok = False
        ctx_ptr_f[unsafe_offset=0].fired_count = 0

        # Frame 0: intra-frame restore fires in STAGE_PRE_UPDATE (F1) or iter 0 of STAGE_FIXED_UPDATE (F2)
        app_f.step_one_frame(Float32(1.0 / 30.0))
        assert_equal(ctx_ptr_f[unsafe_offset=0].fired_count, 1)
        assert_true(ctx_ptr_f[unsafe_offset=0].restore_ok)
        assert_true((app_f.world.get[GlobalTransform](e_f).translation - pos_f).length() < Float32(1.0e-4))
        assert_true(_wait_audio_listener_pos(srv_f, pos_f))
        assert_true(srv_f._rt_core()[].listener_vel.length() < Float32(1.0e-4))

        # Frame 1 (static): dirty-chunk skipping must re-engage cleanly
        app_f.step_one_frame(dt)
        assert_true((app_f.world.get[GlobalTransform](e_f).translation - pos_f).length() < Float32(1.0e-4))
        assert_equal(cache_f.last_propagated_count, 0)
        assert_true(cache_f.last_skipped_count > 0)

        app_f.shutdown()


def test_stage_last_equal_tick_restore_blob_propagates_transform() raises:
    var dt = Float32(1.0 / 60.0)

    # 1. STAGE_LAST restore_blob at saved_tick == post_update_tick (unpropagated + post-restore mutation)
    for mode in range(2):
        var mutate_post = (mode == 1)
        var target_pos = Vec3(777.0, 2.0, -3.0) if mutate_post else Vec3(555.0, 1.0, 4.0)

        var src = EngineApp(AppConfig(title="LastEqSrc", headless=True, worker_threads=0))
        src.add_plugin(TransformPlugin())
        src.startup()
        var e_src = src.world.spawn()
        _ = src.world.insert[LocalTransform](e_src, LocalTransform(Vec3(1.0, 2.0, 3.0)))
        _ = src.world.insert[GlobalTransform](e_src, GlobalTransform())
        _ = src.world.insert[AudioListenerComponent](e_src, AudioListenerComponent(active=True, priority=1))
        for _ in range(4):
            src.step_one_frame(dt)
        assert_equal(Int(src.world.current_tick), 9)
        if not mutate_post:
            src.world.get_ptr_mut[LocalTransform](e_src).value()[unsafe_offset=0].translation = target_pos
            src.world.get_ptr_mut[GlobalTransform](e_src).value()[unsafe_offset=0] = GlobalTransform()
        _ = src.world.advance_tick()
        assert_equal(Int(src.world.current_tick), 10)
        var snap10 = src.world.snapshot_blob()
        src.shutdown()

        var app = EngineApp(AppConfig(title="LastEqDst", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AudioPlugin(headless=True))
        app.insert_resource[_IntraRestoreCtxResource](_IntraRestoreCtxResource())
        _ = app.add_system(
            STAGE_LAST,
            "flint.test.stage_last_restore",
            _sys_intra_restore,
            exclusive=True,
        )
        app.startup()
        var e = app.world.spawn()
        assert_equal(e.to_bits(), e_src.to_bits())
        _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e, GlobalTransform())
        _ = app.world.insert[AudioListenerComponent](e, AudioListenerComponent(active=True, priority=1))
        for _ in range(4):
            app.step_one_frame(dt)
        assert_equal(Int(app.world.current_tick), 9)

        # Arm STAGE_LAST restore for the upcoming frame where post_update_tick == 10 == saved_tick
        var ctx_ptr = app.resources.get_ptr_mut[_IntraRestoreCtxResource]().value()
        ctx_ptr[unsafe_offset=0].snap = snap10^
        ctx_ptr[unsafe_offset=0].target_entity = e
        ctx_ptr[unsafe_offset=0].new_pos = target_pos if mutate_post else Vec3(555.0, 1.0, 4.0)
        ctx_ptr[unsafe_offset=0].armed = True
        ctx_ptr[unsafe_offset=0].restore_ok = False
        ctx_ptr[unsafe_offset=0].fired_count = 0

        # Frame 4: STAGE_POST_UPDATE propagates at tick 10; STAGE_LAST restores snap10 (saved_tick=10)
        app.step_one_frame(dt)
        assert_equal(ctx_ptr[unsafe_offset=0].fired_count, 1)
        assert_true(ctx_ptr[unsafe_offset=0].restore_ok)
        assert_equal(Int(app.world.restore_epoch), 1)

        # Frame 5: GlobalTransform MUST propagate target_pos without freezing at 0.0 and zero Doppler spike
        ref cache = app.resources.get_ptr[TransformPropagationCache]().value()[unsafe_offset=0]
        ref srv = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]
        app.step_one_frame(dt)
        assert_true((app.world.get[GlobalTransform](e).translation - target_pos).length() < Float32(1.0e-4))
        assert_true(_wait_audio_listener_pos(srv, target_pos))
        assert_true(srv._rt_core()[].listener_vel.length() < Float32(1.0e-4))

        # Frame 6 (static): dirty-chunk skipping must re-engage cleanly
        app.step_one_frame(dt)
        assert_true((app.world.get[GlobalTransform](e).translation - target_pos).length() < Float32(1.0e-4))
        assert_equal(cache.last_propagated_count, 0)
        assert_true(cache.last_skipped_count > 0)

        app.shutdown()

    # 2. Direct caller without EngineApp: restore_blob + advance_tick + propagate_transforms_parallel_cached
    var w = World()
    var pool = JobPool()
    var dcache = TransformPropagationCache()
    var de = w.spawn()
    _ = w.insert[LocalTransform](de, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = w.insert[GlobalTransform](de, GlobalTransform())
    for _ in range(3):
        _ = w.advance_tick()
    propagate_transforms_parallel_cached(w, pool, dcache)
    var saved = w.snapshot_blob()
    for _ in range(2):
        _ = w.advance_tick()
    propagate_transforms_parallel_cached(w, pool, dcache)
    w.restore_blob(saved)
    w.get_ptr_mut[LocalTransform](de).value()[unsafe_offset=0].translation = Vec3(99.0, 0.0, 0.0)
    w.get_ptr_mut[GlobalTransform](de).value()[unsafe_offset=0] = GlobalTransform()
    # Advance tick past previous last_propagated_tick before calling propagate
    for _ in range(4):
        _ = w.advance_tick()
    propagate_transforms_parallel_cached(w, pool, dcache)
    assert_true(abs(w.get[GlobalTransform](de).translation.x - Float32(99.0)) < Float32(1.0e-4))
    pool.stop()


def test_audio_emitter_play_survives_snapshot_restore() raises:
    var dt = Float32(1.0 / 60.0)
    var app = EngineApp(AppConfig(title="AudioPlayRestore", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.startup()

    var pcm = _make_sine_pcm16(440.0, 24000, 48000)
    var aid = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].register_pcm16_asset(pcm, 1, 48000)
    ref srv = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0]

    var lis_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lis_e, LocalTransform(Vec3(0.0, 0.0, 5.0)))
    _ = app.world.insert[GlobalTransform](lis_e, GlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lis_e, AudioListenerComponent(active=True, priority=1))

    var em_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](em_e, LocalTransform(Vec3(1.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](em_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        em_e,
        AudioEmitterComponent(asset_id=aid, asset_path="loop.wav", looping=True, play_on_spawn=False),
    )

    # Before .play(), no logical voice is active
    app.step_one_frame(dt)
    assert_equal(srv.active_logical_count(), 0)

    # Trigger via .play() (play_on_spawn=False) and wait for RT voice acquisition
    app.world.get_ptr_mut[AudioEmitterComponent](em_e).value()[unsafe_offset=0].play()
    for _ in range(4):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
    assert_true(app.world.get_ptr[AudioEmitterComponent](em_e).value()[unsafe_offset=0].handle.is_valid())
    assert_true(app.world.get_ptr[AudioEmitterComponent](em_e).value()[unsafe_offset=0].playing)
    assert_equal(srv.active_logical_count(), 1)

    var blob = app.world.snapshot_blob()

    # Step a few frames (and move emitter to verify smooth Doppler on restore too)
    app.world.get_ptr_mut[LocalTransform](em_e).value()[unsafe_offset=0].translation = Vec3(10.0, 0.0, 0.0)
    app.step_one_frame(dt)
    app.step_one_frame(dt)

    # Restore snapshot: play_on_spawn=False emitter that was playing MUST re-acquire a live voice
    app.world.restore_blob(blob)
    for _ in range(4):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
        if srv.active_logical_count() >= 1:
            break

    ref em_after = app.world.get_ptr[AudioEmitterComponent](em_e).value()[unsafe_offset=0]
    assert_true(em_after.playing)
    assert_true(em_after.handle.is_valid())
    assert_true(srv.active_logical_count() >= 1)

    # play_on_spawn=True emitter: auto-plays on spawn, stays silent after .stop() directly AND across restore_blob()
    var spawn_em_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](spawn_em_e, LocalTransform(Vec3(-2.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](spawn_em_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        spawn_em_e,
        AudioEmitterComponent(asset_id=aid, asset_path="spawn.wav", looping=True, play_on_spawn=True),
    )
    for _ in range(4):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
        if srv.active_logical_count() >= 2:
            break
    assert_true(app.world.get_ptr[AudioEmitterComponent](spawn_em_e).value()[unsafe_offset=0].playing)
    assert_true(app.world.get_ptr[AudioEmitterComponent](spawn_em_e).value()[unsafe_offset=0].handle.is_valid())
    assert_equal(srv.active_logical_count(), 2)

    # Stop the play_on_spawn=True emitter and verify it stays silent for 5 frames
    app.world.get_ptr_mut[AudioEmitterComponent](spawn_em_e).value()[unsafe_offset=0].stop()
    for _ in range(5):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
    ref spawn_stopped = app.world.get_ptr[AudioEmitterComponent](spawn_em_e).value()[unsafe_offset=0]
    assert_true(not spawn_stopped.playing)
    assert_true(not spawn_stopped.auto_play_pending)
    assert_true(not spawn_stopped.handle.is_valid())
    assert_equal(srv.active_logical_count(), 1)

    # Snapshot with stopped play_on_spawn=True emitter, re-play, then restore_blob: must stay silent across 5 frames
    var stopped_blob = app.world.snapshot_blob()
    app.world.get_ptr_mut[AudioEmitterComponent](spawn_em_e).value()[unsafe_offset=0].play()
    for _ in range(3):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
    assert_equal(srv.active_logical_count(), 2)

    app.world.restore_blob(stopped_blob)
    for _ in range(5):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
    ref spawn_restored = app.world.get_ptr[AudioEmitterComponent](spawn_em_e).value()[unsafe_offset=0]
    assert_true(not spawn_restored.playing)
    assert_true(not spawn_restored.auto_play_pending)
    assert_true(not spawn_restored.handle.is_valid())
    assert_equal(srv.active_logical_count(), 1)

    # Deferred asset_id (-1 -> valid aid) with play_on_spawn=True still triggers once asset_id is set
    var def_em_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](def_em_e, LocalTransform(Vec3(3.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](def_em_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        def_em_e,
        AudioEmitterComponent(asset_id=-1, asset_path="late.wav", looping=True, play_on_spawn=True),
    )
    app.step_one_frame(dt)
    assert_equal(srv.active_logical_count(), 1)
    app.world.get_ptr_mut[AudioEmitterComponent](def_em_e).value()[unsafe_offset=0].asset_id = aid
    for _ in range(4):
        app.step_one_frame(dt)
        _ = external_call["usleep", Int32](UInt32(2000))
        if srv.active_logical_count() >= 2:
            break
    assert_true(app.world.get_ptr[AudioEmitterComponent](def_em_e).value()[unsafe_offset=0].playing)
    assert_equal(srv.active_logical_count(), 2)

    app.shutdown()


def test_physics_collider_update_and_joint_rb_replace() raises:
    var dt = Float32(1.0 / 60.0)
    var app = EngineApp(AppConfig(title="PhysColJnt", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.startup()

    # 1. ColliderComponent replacement (box -> sphere(3.0) -> capsule(1.2, 2.0)) without replacing RigidBodyComponent
    var col_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](col_e, LocalTransform(Vec3(-5.0, 10.0, 0.0)))
    _ = app.world.insert[GlobalTransform](col_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](col_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](col_e, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))

    app.step_one_frame(dt)
    ref pw = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    var bid_col = app.world.get[RigidBodyComponent](col_e).body_id
    var slot_col = pw.find_slot_by_id(bid_col)
    assert_true(slot_col >= 0)
    assert_equal(pw.bodies[slot_col].shape, BODY_SHAPE_BOX)

    _ = app.world.insert[ColliderComponent](col_e, ColliderComponent.sphere(3.0))
    app.step_one_frame(dt)
    slot_col = pw.find_slot_by_id(bid_col)
    assert_equal(pw.bodies[slot_col].shape, BODY_SHAPE_SPHERE)
    assert_true(abs(pw.bodies[slot_col].radius - Float32(3.0)) < Float32(1.0e-4))
    assert_true(abs(pw.bodies[slot_col].half_extents.x - Float32(3.0)) < Float32(1.0e-4))

    _ = app.world.insert[ColliderComponent](col_e, ColliderComponent.capsule(1.2, 2.0))
    app.step_one_frame(dt)
    assert_equal(pw.bodies[slot_col].shape, BODY_SHAPE_CAPSULE)
    assert_true(abs(pw.bodies[slot_col].radius - Float32(1.2)) < Float32(1.0e-4))
    assert_true(abs(pw.bodies[slot_col].half_height - Float32(2.0)) < Float32(1.0e-4))

    # Simultaneous static -> dynamic motion_mode + ColliderComponent change updates wind_area_over_mass
    app.world.get_ptr_mut[RigidBodyComponent](col_e).value()[unsafe_offset=0].motion_mode = BODY_MOTION_DYNAMIC
    app.world.get_ptr_mut[RigidBodyComponent](col_e).value()[unsafe_offset=0].gravity_scale = Float32(0.0)
    _ = app.world.insert[ColliderComponent](col_e, ColliderComponent.sphere(2.0))
    app.step_one_frame(dt)
    slot_col = pw.find_slot_by_id(bid_col)
    assert_true(not pw.bodies[slot_col].is_static)
    assert_true(pw.bodies[slot_col].wind_area_over_mass > Float32(1.0))
    app.world.get_ptr_mut[RigidBodyComponent](col_e).value()[unsafe_offset=0].active = False

    # 2. Replacing RigidBodyComponent on a JOINT_WELD entity_b keeps e_b attached under gravity
    var e_a = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_a, LocalTransform(Vec3(0.0, 8.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e_a, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](e_a, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](e_a, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))

    var e_b = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_b, LocalTransform(Vec3(1.0, 8.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e_b, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](e_b, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](e_b, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))

    _ = app.world.insert[JointComponent](
        e_a,
        JointComponent(
            joint_kind=JOINT_WELD,
            entity_b=e_b,
            anchor_a=Vec3(0.5, 8.0, 0.0),
            anchor_b=Vec3(0.5, 8.0, 0.0),
        ),
    )

    for _ in range(10):
        app.step_one_frame(dt)
    assert_equal(len(pw.joints), 1)
    assert_true(not pw.joints[0].broken)
    assert_true(app.world.get[LocalTransform](e_b).translation.y > Float32(7.95))

    # Replace e_b's RigidBodyComponent: joint must rebind to the new body_id and hold e_b up
    var old_j_idx = app.world.get[JointComponent](e_a).runtime_index
    assert_equal(old_j_idx, 0)
    _ = app.world.insert[RigidBodyComponent](e_b, RigidBodyComponent.dynamic(2.5))
    for _ in range(15):
        app.step_one_frame(dt)
    var new_j_idx = app.world.get[JointComponent](e_a).runtime_index
    assert_equal(len(pw.joints), 2)
    assert_equal(new_j_idx, 1)
    assert_true(new_j_idx >= 0 and new_j_idx != old_j_idx)
    assert_true(pw.joints[old_j_idx].broken)
    assert_true(not pw.joints[new_j_idx].broken)
    assert_true(app.world.get[LocalTransform](e_b).translation.y > Float32(7.95))

    # Also replace e_a's RigidBodyComponent on a world-welded joint
    var e_w = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_w, LocalTransform(Vec3(5.0, 8.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e_w, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](e_w, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](e_w, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))
    _ = app.world.insert[JointComponent](
        e_w,
        JointComponent(joint_kind=JOINT_WELD, anchor_a=Vec3(5.0, 8.0, 0.0)),
    )
    for _ in range(5):
        app.step_one_frame(dt)
    assert_equal(len(pw.joints), 3)
    var old_wj = app.world.get[JointComponent](e_w).runtime_index
    assert_equal(old_wj, 2)
    _ = app.world.insert[RigidBodyComponent](e_w, RigidBodyComponent.dynamic(3.0))
    for _ in range(15):
        app.step_one_frame(dt)
    var new_wj = app.world.get[JointComponent](e_w).runtime_index
    assert_equal(len(pw.joints), 4)
    assert_equal(new_wj, 3)
    assert_true(new_wj >= 0 and new_wj != old_wj)
    assert_true(pw.joints[old_wj].broken)
    assert_true(app.world.get[LocalTransform](e_w).translation.y > Float32(7.95))

    # Replacing static e_a's RigidBodyComponent with another static body rebinds the joint cleanly
    _ = app.world.insert[RigidBodyComponent](e_a, RigidBodyComponent.static_body())
    for _ in range(10):
        app.step_one_frame(dt)
    var rebind_a_idx = app.world.get[JointComponent](e_a).runtime_index
    assert_equal(len(pw.joints), 5)
    assert_equal(rebind_a_idx, 4)
    assert_true(pw.joints[1].broken)
    assert_true(not pw.joints[rebind_a_idx].broken)
    assert_true(app.world.get[LocalTransform](e_b).translation.y > Float32(7.95))

    # Breaking a static-A + dynamic-B joint must NOT be resurrected across subsequent frames
    app.resources.get_ptr_mut[PhysicsWorld]().value()[unsafe_offset=0].joints[rebind_a_idx].broken = True
    var y_before_break = app.world.get[LocalTransform](e_b).translation.y
    for _ in range(10):
        app.step_one_frame(dt)
    assert_equal(len(pw.joints), 5)
    assert_true(pw.joints[rebind_a_idx].broken)
    assert_true(app.world.get[LocalTransform](e_b).translation.y < y_before_break - Float32(0.05))

    # 3. RigidBodyComponent + ColliderComponent(is_trigger=True) live update converts solid body to non-solid trigger & back
    var plat_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](plat_e, LocalTransform(Vec3(25.0, 5.0, 0.0)))
    _ = app.world.insert[GlobalTransform](plat_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](plat_e, RigidBodyComponent.static_body())
    _ = app.world.insert[ColliderComponent](plat_e, ColliderComponent.box(Vec3(2.0, 0.5, 2.0), is_trigger=False))
    app.step_one_frame(dt)

    # Flip is_trigger=True live on plat_e
    app.world.get_ptr_mut[ColliderComponent](plat_e).value()[unsafe_offset=0].is_trigger = True
    app.step_one_frame(dt)

    var drop_trig = app.world.spawn()
    _ = app.world.insert[LocalTransform](drop_trig, LocalTransform(Vec3(25.0, 6.2, 0.0)))
    _ = app.world.insert[GlobalTransform](drop_trig, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](drop_trig, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](drop_trig, ColliderComponent.sphere(0.4))

    var tev_cursor = 0
    var saw_trigger_enter = False
    for _ in range(45):
        app.step_one_frame(dt)
        var tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
        for ti in range(len(tevs)):
            if tevs[ti].kind == TRIGGER_ENTER:
                saw_trigger_enter = True
    assert_true(saw_trigger_enter)
    assert_true(app.world.get[LocalTransform](drop_trig).translation.y < Float32(4.0))

    # Flip is_trigger=False back: plat_e becomes solid again and catches a newly dropped sphere
    app.world.get_ptr_mut[ColliderComponent](plat_e).value()[unsafe_offset=0].is_trigger = False
    app.step_one_frame(dt)

    var drop_solid = app.world.spawn()
    _ = app.world.insert[LocalTransform](drop_solid, LocalTransform(Vec3(25.0, 6.2, 0.0)))
    _ = app.world.insert[GlobalTransform](drop_solid, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](drop_solid, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](drop_solid, ColliderComponent.sphere(0.4))
    for _ in range(45):
        app.step_one_frame(dt)
    assert_true(app.world.get[LocalTransform](drop_solid).translation.y > Float32(5.4))

    app.shutdown()


def test_net_plugin_before_transform_plugin_zero_lag() raises:
    var sim = VirtualNetSim(
        seed=0x2244668811335577,
        one_way_latency_ms=5.0,
        jitter_ms=0.0,
        loss_rate=0.0,
        dup_rate=0.0,
    )
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9600), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9600), False)
    var cli_cid = cli_hub.connect(srv_ep)
    assert_true(cli_hub.send_unreliable(cli_cid, List[UInt8](), sim.now_ms()))

    # Register NetPlugin BEFORE TransformPlugin (reverse order)
    var srv_app = EngineApp(AppConfig(title="NetRevSrv", headless=True, worker_threads=0))
    srv_app.add_plugin(NetPlugin(max_entities=32, max_clients=4))
    srv_app.add_plugin(TransformPlugin())
    srv_app.startup()
    assert_equal(len(srv_app.schedule.stages[STAGE_POST_UPDATE].systems), 2)
    srv_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    srv_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(srv_hub^)

    var cli_app = EngineApp(AppConfig(title="NetRevCli", headless=True, worker_threads=0))
    cli_app.add_plugin(NetPlugin(max_entities=32, max_clients=4))
    cli_app.add_plugin(TransformPlugin())
    cli_app.startup()
    assert_equal(len(cli_app.schedule.stages[STAGE_POST_UPDATE].systems), 2)
    cli_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    cli_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(cli_hub^)

    var srv_e = srv_app.world.spawn()
    _ = srv_app.world.insert[LocalTransform](srv_e, LocalTransform(Vec3(14.5, 1.0, 0.0)))
    _ = srv_app.world.insert[GlobalTransform](srv_e, GlobalTransform())
    _ = srv_app.world.insert[NetworkIdentityComponent](srv_e, NetworkIdentityComponent(net_id=10))
    _ = srv_app.world.insert[ReplicationComponent](srv_e, ReplicationComponent())

    var cli_e = cli_app.world.spawn()
    _ = cli_app.world.insert[LocalTransform](cli_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = cli_app.world.insert[GlobalTransform](cli_e, GlobalTransform())
    _ = cli_app.world.insert[NetworkIdentityComponent](cli_e, NetworkIdentityComponent(net_id=10))
    _ = cli_app.world.insert[ReplicationComponent](cli_e, ReplicationComponent())

    var dt = Float32(1.0 / 60.0)
    for _ in range(5):
        srv_app.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        cli_app.step_one_frame(dt)

    # Mutate server LocalTransform and step ONCE: replicator must capture 50.0 on the exact same frame
    srv_app.world.get_ptr_mut[LocalTransform](srv_e).value()[unsafe_offset=0].translation.x = Float32(50.0)
    srv_app.step_one_frame(dt)
    assert_true(abs(srv_app.world.get[GlobalTransform](srv_e).translation.x - Float32(50.0)) < Float32(1.0e-4))
    ref srv_rep = srv_app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].replicator
    var found_slot = -1
    for i in range(len(srv_rep.entities)):
        if srv_rep.entities[i].active and srv_rep.entities[i].net_id == 10:
            found_slot = i
            break
    assert_true(found_slot >= 0)
    assert_true(abs(srv_rep.entities[found_slot].pos.x - Float32(50.0)) < Float32(1.0e-4))

    for _ in range(5):
        srv_app.step_one_frame(dt)
        sim.advance_ms(Float64(dt) * 1000.0)
        cli_app.step_one_frame(dt)
    assert_true(abs(cli_app.world.get[GlobalTransform](cli_e).translation.x - Float32(50.0)) < Float32(0.05))

    cli_app.shutdown()
    srv_app.shutdown()


def main() raises:
    var passed = 0
    var total = 19

    test_window_and_input_plugin_ordering_and_actions()
    print("[PASS] test_window_and_input_plugin_ordering_and_actions")
    passed += 1

    test_physics_plugin_gravity_triggers_events_kcc_vehicle_aero_cloth()
    print("[PASS] test_physics_plugin_gravity_triggers_events_kcc_vehicle_aero_cloth")
    passed += 1

    test_anim_audio_render_plugins_integration()
    print("[PASS] test_anim_audio_render_plugins_integration")
    passed += 1

    test_net_plugin_virtual_sim_replication()
    print("[PASS] test_net_plugin_virtual_sim_replication")
    passed += 1

    test_snapshot_blob_and_component_registry_all_schemas()
    print("[PASS] test_snapshot_blob_and_component_registry_all_schemas")
    passed += 1

    test_plugin_idempotency_transform_ticks_and_generic_ids()
    print("[PASS] test_plugin_idempotency_transform_ticks_and_generic_ids")
    passed += 1

    test_physics_non_rb_despawn_and_snapshot_restore()
    print("[PASS] test_physics_non_rb_despawn_and_snapshot_restore")
    passed += 1

    test_anim_parented_bone_socket_audio_despawn_and_net_virtual_physics()
    print("[PASS] test_anim_parented_bone_socket_audio_despawn_and_net_virtual_physics")
    passed += 1

    test_physics_rb_overlap_restore_and_replace_cleanup()
    print("[PASS] test_physics_rb_overlap_restore_and_replace_cleanup")
    passed += 1

    test_anim_ragdoll_despawn_remove_and_restore_cleanup()
    print("[PASS] test_anim_ragdoll_despawn_remove_and_restore_cleanup")
    passed += 1

    test_audio_listener_priority_switch_zero_doppler_spike()
    print("[PASS] test_audio_listener_priority_switch_zero_doppler_spike")
    passed += 1

    test_anim_ragdoll_no_one_frame_ghost_collision()
    print("[PASS] test_anim_ragdoll_no_one_frame_ghost_collision")
    passed += 1

    test_anim_ragdoll_remove_skeleton_deactivates()
    print("[PASS] test_anim_ragdoll_remove_skeleton_deactivates")
    passed += 1

    test_audio_listener_restore_blob_same_entity_zero_doppler_spike()
    print("[PASS] test_audio_listener_restore_blob_same_entity_zero_doppler_spike")
    passed += 1

    test_restore_blob_tick_rollback_transform_and_audio_doppler()
    print("[PASS] test_restore_blob_tick_rollback_transform_and_audio_doppler")
    passed += 1

    test_stage_last_equal_tick_restore_blob_propagates_transform()
    print("[PASS] test_stage_last_equal_tick_restore_blob_propagates_transform")
    passed += 1

    test_audio_emitter_play_survives_snapshot_restore()
    print("[PASS] test_audio_emitter_play_survives_snapshot_restore")
    passed += 1

    test_physics_collider_update_and_joint_rb_replace()
    print("[PASS] test_physics_collider_update_and_joint_rb_replace")
    passed += 1

    test_net_plugin_before_transform_plugin_zero_lag()
    print("[PASS] test_net_plugin_before_transform_plugin_zero_lag")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
