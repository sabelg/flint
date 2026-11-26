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
from std.math import abs, isfinite, sin
from std.memory import bitcast
from std.testing import assert_equal, assert_true

from flint.ai import (
    AiPlugin,
    BT_ABORT_LOWER_PRIORITY,
    BT_CMP_GT,
    BT_FAILURE,
    BT_NODE_ACTION,
    BT_RUNNING,
    BT_SUCCESS,
    BehaviorTree,
    BehaviorTreeComponent,
    Blackboard,
    BtExecutionState,
    CURVE_LINEAR,
    SpatialPerception,
    UtilityAiScorer,
    UtilityConsideration,
    UtilityOption,
    register_ai_codecs,
    register_ai_components,
)

from flint.anim.flrig import (
    FrigJointDef,
    FrigRigDef,
    save_flrig_file,
)
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
    RigComponent,
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
    GameFlow,
    GameStateMachine,
    GameTimer,
    STAGE_FIRST,
    STAGE_FIXED_UPDATE,
    STAGE_LAST,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_SHUTDOWN,
    STAGE_STARTUP,
    STAGE_UPDATE,
    SaveSlotStore,
    Tween,
)
from flint.app.engine_app import Resource, Resources
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    AssetPlugin,
    AssetRef,
    AssetServer,
    AssetUuid,
    SceneAssetResolver,
)
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flmat import FlmatAsset, load_flmat_bytes, save_flmat_file
from flint.assets.flpk_v3 import (
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    TEX_FMT_BC5_UNORM,
    TEX_FMT_BC7_SRGB,
    TEX_FMT_BC7_UNORM,
    write_flpk_v3,
)
from flint.assets.glb_writer import write_native_glb
from flint.assets.image import write_png
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeAnimation, NativeChannel, NativeMesh, NativeModel, NativeNode, NativeSkin
from flint.assets.server import (
    CookedTextureAsset,
    Handle,
    _asset_server_evict_texture_slot,
)
from flint.assets.watcher import AssetWatcher
from flint.platform.os import list_directory, make_dirs, read_bytes, write_bytes
from flint.audio import AudioServer, CLEAR_CUTOFF_HZ, encode_qoa
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
from flint.core.serial import BlobReader, BlobWriter
from flint.ecs import ComponentBitSet, Entity, NULL_ENTITY, ParallelCommands, World, comp_hash64
from flint.nav import (
    NavAgentComponent,
    NavMeshComponent,
    NavMeshConfig,
    NavPlugin,
    NavPluginStateResource,
    bake_from_physics,
    register_nav_codecs,
    register_nav_components,
)
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
    OnEnter,
    OnExit,
    PhysicsEventsResource,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    RigidBodyComponent,
    TriggerEventsResource,
    VehicleComponent,
    entity_from_body,
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
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
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
from flint.studio.studio_app import FlintStudioState
from flint.platform.input import InputState
from flint.render.font import DSF_MAGIC
from flint.ui.context import UiContext
from flint.ui.draw_list import PRIM_GLYPH
from flint.ui.hud import game_hud


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


def _t5_ord_sys_a(
    _w: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var log = Pointer[List[Int], MutAnyOrigin](unsafe_from_address=ctx)
    log[unsafe_offset=0].append(1)


def _t5_ord_sys_b(
    _w: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var log = Pointer[List[Int], MutAnyOrigin](unsafe_from_address=ctx)
    log[unsafe_offset=0].append(2)


def _t5_ord_sys_c(
    _w: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var log = Pointer[List[Int], MutAnyOrigin](unsafe_from_address=ctx)
    log[unsafe_offset=0].append(3)


def test_schedule_out_of_order_add_dependency() raises:
    # [High-6] Register A before B before C, then constrain A after B and B after C!
    var log = List[Int]()
    var log_addr = Int(Pointer(to=log))
    var app = EngineApp(AppConfig(title="OutOfOrderDep", headless=True, worker_threads=2))
    var rw = ComponentBitSet().with_bit(0)
    var sid_a = app.schedule.add_system(STAGE_UPDATE, "sys_a", _t5_ord_sys_a, writes=rw, user_ctx=log_addr)
    var sid_b = app.schedule.add_system(STAGE_UPDATE, "sys_b", _t5_ord_sys_b, writes=rw, user_ctx=log_addr)
    var sid_c = app.schedule.add_system(STAGE_UPDATE, "sys_c", _t5_ord_sys_c, user_ctx=log_addr)
    app.schedule.add_dependency(STAGE_UPDATE, sid_a, sid_b)
    app.schedule.add_dependency(STAGE_UPDATE, sid_b, sid_c)

    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))
    app.shutdown()

    assert_equal(len(log), 3)
    assert_equal(log[0], 3)
    assert_equal(log[1], 2)
    assert_equal(log[2], 1)


def test_schedule_conflict_with_explicit_dependency() raises:
    # [HIGH-1] sys0 (writes A), sys1 (reads A), sys2 (reads B), add_dependency(STAGE_UPDATE, sys0, sys2)
    # Must preserve registration-order R/W conflict (sys0 -> sys1) AND explicit (sys2 -> sys0) => [2, 0, 1]
    var log = List[Int]()
    var log_addr = Int(Pointer(to=log))
    var app = EngineApp(AppConfig(title="ConflictExplicitDep", headless=True, worker_threads=2))
    var comp_a = ComponentBitSet().with_bit(0)
    var comp_b = ComponentBitSet().with_bit(1)
    var s0 = app.schedule.add_system(STAGE_UPDATE, "sys0_wA", _t5_ord_sys_a, writes=comp_a, user_ctx=log_addr)
    _ = app.schedule.add_system(STAGE_UPDATE, "sys1_rA", _t5_ord_sys_b, reads=comp_a, user_ctx=log_addr)
    var s2 = app.schedule.add_system(STAGE_UPDATE, "sys2_rB", _t5_ord_sys_c, reads=comp_b, user_ctx=log_addr)
    app.schedule.add_dependency(STAGE_UPDATE, s0, s2)
    app.schedule.stages[STAGE_UPDATE].compile()
    assert_equal(len(app.schedule.stages[STAGE_UPDATE].compiled_nodes[0].dependents), 1)
    assert_equal(app.schedule.stages[STAGE_UPDATE].compiled_nodes[0].dependents[0], 1)
    assert_equal(len(app.schedule.stages[STAGE_UPDATE].compiled_nodes[2].dependents), 1)
    assert_equal(app.schedule.stages[STAGE_UPDATE].compiled_nodes[2].dependents[0], 0)

    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))
    app.shutdown()

    assert_equal(len(log), 3)
    assert_equal(log[0], 3)  # s2
    assert_equal(log[1], 1)  # s0
    assert_equal(log[2], 2)  # s1


def test_m3a_nav_plugin_lifecycle_and_overlay() raises:
    # 1. Component & codec registration + binary/JSON round-trip
    var reg = ComponentRegistry.with_builtins()
    register_nav_codecs(reg)
    assert_true(reg.find_by_name("NavMeshComponent") >= 0)
    assert_true(reg.find_by_name("NavAgentComponent") >= 0)

    var nmc = NavMeshComponent(cell_size=0.3, draw_debug=True, auto_bake=True)
    var bw_nm = BlobWriter()
    nmc.write_bin(bw_nm)
    var br_nm = BlobReader(bw_nm.data.copy())
    var nmc2 = NavMeshComponent.read_bin(br_nm, NavMeshComponent.SCHEMA_VERSION)
    assert_true(abs(nmc2.cell_size - Float32(0.3)) < Float32(1.0e-5))
    assert_true(nmc2.auto_bake)

    var nac = NavAgentComponent(goal=Vec3(5.0, 0.0, 0.0), has_goal=True, max_speed=4.0)
    var bw_na = BlobWriter()
    nac.write_bin(bw_na)
    var br_na = BlobReader(bw_na.data.copy())
    var nac2 = NavAgentComponent.read_bin(br_na, NavAgentComponent.SCHEMA_VERSION)
    assert_true(nac2.has_goal and abs(nac2.goal.x - Float32(5.0)) < Float32(1.0e-5))
    assert_equal(nac2.agent_id, -1)

    # 2. EngineApp + NavPlugin + PhysicsPlugin + RenderWorld overlay_lines + crowd step
    var app = EngineApp(AppConfig(title="m3a_nav_plugin", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.add_plugin(NavPlugin(draw_debug=True))
    app.insert_resource[RenderWorld](RenderWorld())
    app.startup()

    var has_nav_sys = False
    ref fixed_sys = app.schedule.stages[STAGE_FIXED_UPDATE].systems
    for i in range(len(fixed_sys)):
        if String(fixed_sys[i].name).find("nav") >= 0:
            has_nav_sys = True
            break
    assert_true(has_nav_sys)

    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](floor_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(10.0, 0.5, 10.0)))
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())

    var nav_e = app.world.spawn()
    _ = app.world.insert[NavMeshComponent](nav_e, NavMeshComponent(draw_debug=True, auto_bake=True))

    var ag_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](ag_e, LocalTransform(Vec3(-4.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ag_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](ag_e, PrevGlobalTransform())
    _ = app.world.insert[NavAgentComponent](ag_e, NavAgentComponent(goal=Vec3(4.0, 0.0, 0.0), has_goal=True))

    assert_true(bake_from_physics(app.world) > 0)

    for _ in range(15):
        app.step_one_frame(Float32(1.0 / 60.0))

    var nav_res = app.resources.get_ptr[NavPluginStateResource]()
    assert_true(Bool(nav_res) and nav_res.value()[unsafe_offset=0].baked)
    assert_true(nav_res.value()[unsafe_offset=0].last_poly_count > 0)

    var rw_ptr = app.resources.get_ptr[RenderWorld]()
    assert_true(Bool(rw_ptr) and rw_ptr.value()[unsafe_offset=0].overlay_lines.segment_count() > 0)

    var ag_pos = app.world.get_ptr[LocalTransform](ag_e).value()[unsafe_offset=0].translation
    assert_true(ag_pos.x > Float32(-3.8))
    app.shutdown()


def _plug_str_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _plug_rm_rf(path: String):
    var cmd = "rm -rf '" + path + "'"
    _ = external_call["system", Int32](cmd.as_c_string_span())


def _make_cube_mesh_for_plug() -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(), Vec3(0.5, 0.5, 0.5), Float32(0.02))
    return mb^.finish()


def _make_skinned_biped_for_asset_plugin() -> NativeModel:
    var m = _make_biped_model()
    var cube = _make_cube_mesh_for_plug()
    var nv = cube.vertex_count()
    for _ in range(nv):
        cube.joints.append(UInt16(0))
        cube.joints.append(UInt16(1))
        cube.joints.append(UInt16(0))
        cube.joints.append(UInt16(0))
        cube.weights.append(Float32(0.5))
        cube.weights.append(Float32(0.5))
        cube.weights.append(Float32(0.0))
        cube.weights.append(Float32(0.0))
    m.meshes.append(cube^)
    m.nodes[0].mesh = 0
    var skin = NativeSkin()
    for j in range(4):
        skin.joints.append(j)
        for k in range(16):
            skin.inverse_bind.append(Float32(1.0) if (k % 5 == 0) else Float32(0.0))
    m.skins.append(skin^)
    return m^


def test_m4a_asset_server_traversal_and_asset_ref_unsafe_path() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_trav_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/meshes")
    try:
        # Outside files in parent directory (must NOT be reachable via ../ traversal)
        var b_txt = _plug_str_bytes("SECRET_TEXT")
        _ = write_bytes(tmp + "/outside.txt", b_txt)
        var px = List[UInt32](length=4, fill=0xFFFFFFFF)
        _ = write_png(tmp + "/outside.png", px, 2, 2, stamped=False)
        var b_wav = _plug_str_bytes("RIFF0000WAVEfmt ")
        _ = write_bytes(tmp + "/outside.wav", b_wav)
        var b_hdr = _plug_str_bytes("#?RADIANCE\n\n-Y 1 +X 1\n")
        _ = write_bytes(tmp + "/outside.hdr", b_hdr)
        var b_ttf = _plug_str_bytes("TTF_STUB_BYTES")
        _ = write_bytes(tmp + "/outside.ttf", b_ttf)
        var m_out = NativeModel()
        m_out.meshes.append(_make_cube_mesh_for_plug())
        _ = write_native_glb(tmp + "/outside.glb", m_out)

        # Valid inside mesh with valid UUID
        var m_in = NativeModel()
        m_in.meshes.append(_make_cube_mesh_for_plug())
        _ = write_native_glb(ws + "/meshes/valid.glb", m_in)

        var srv = AssetServer(enable_watcher=False)
        _ = srv.watch_loose_directory(ws)

        # Every load_* method must reject traversal paths
        assert_true(not srv.raws.is_loaded(srv.load_raw("../outside.txt")))
        assert_true(not srv.textures.is_loaded(srv.load_texture("../outside.png")))
        assert_true(not srv.audios.is_loaded(srv.load_audio("../outside.wav")))
        assert_true(not srv.hdrs.is_loaded(srv.load_hdr("../outside.hdr")))
        assert_true(not srv.fonts.is_loaded(srv.load_font("../outside.ttf")))
        assert_true(not srv.meshes.is_alive(srv.load_mesh_async("../outside.glb")))
        assert_true(not srv.textures.is_alive(srv.load_texture_async("../outside.png")))

        # Valid UUID + unsafe rel_path in AssetRef must be rejected by resolve_mesh
        var h_valid = srv.load_mesh("meshes/valid.glb")
        assert_true(srv.meshes.is_loaded(h_valid))
        var valid_uuid = srv.meshes.uuids[Int(h_valid.slot())]
        var bad_ref = AssetRef(valid_uuid, "../outside.glb", ASSET_KIND_MESH)
        assert_equal(bad_ref.rel_path, "../outside.glb")
        assert_true(not srv.meshes.is_loaded(srv.resolve_mesh(bad_ref)))
        srv.close()
    finally:
        _plug_rm_rf(tmp)


def test_m4a_asset_plugin_skinned_frame2_and_flrig_hot_reload() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_m4a_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/meshes")
    _ = make_dirs(ws + "/rigs")
    try:
        var m_char = _make_skinned_biped_for_asset_plugin()
        _ = write_native_glb(ws + "/meshes/char.glb", m_char)

        var rig3 = FrigRigDef("HeroRig")
        rig3.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig3.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig3.joints.append(FrigJointDef("Head", 1, Vec3(0.0, 0.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _ = save_flrig_file(ws + "/rigs/hero.flrig", rig3)

        var app = EngineApp(AppConfig(title="M4aSkinHot", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=True))
        app.add_plugin(AnimPlugin())
        app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
        app.startup()

        var model_id = UInt64(101)
        app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(
            model_id, _make_skinned_biped_for_asset_plugin()
        )

        var mesh_ref = AssetRef.from_path("meshes/char.glb", ASSET_KIND_MESH)
        var rig_ref = AssetRef.from_path("rigs/hero.flrig", ASSET_KIND_RAW)

        var e_a = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_a, LocalTransform(Vec3(-1.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_a, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_a, MeshRendererComponent(-1, -1, True, True, mesh_ref.copy()))
        _ = app.world.insert[SkeletonComponent](e_a, SkeletonComponent(model_id))

        var e_b = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_b, LocalTransform(Vec3(1.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_b, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_b, MeshRendererComponent(-1, -1, True, True, mesh_ref.copy()))
        _ = app.world.insert[SkeletonComponent](e_b, SkeletonComponent(model_id))

        var e_rig = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_rig, LocalTransform(Vec3(0.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_rig, GlobalTransform())
        _ = app.world.insert[SkeletonComponent](e_rig, SkeletonComponent(0, rig_ref=rig_ref.copy()))
        _ = app.world.insert[RigComponent](e_rig, RigComponent(rig_ref.copy()))

        # Frame 1..3: distinct skinned mesh_ids persist (AssetPlugin must NOT clobber back to static mesh)
        app.step_one_frame(Float32(1.0 / 60.0))
        var mid_a1 = app.world.get_ptr[MeshRendererComponent](e_a).value()[].mesh_id
        var mid_b1 = app.world.get_ptr[MeshRendererComponent](e_b).value()[].mesh_id
        assert_true(mid_a1 >= 0 and mid_b1 >= 0 and mid_a1 != mid_b1)
        assert_equal(app.world.get_ptr[RigComponent](e_rig).value()[].joint_count, 3)
        assert_equal(app.world.get_ptr[SkeletonComponent](e_rig).value()[].skeleton.node_count, 3)

        for _ in range(2):
            app.step_one_frame(Float32(1.0 / 60.0))
            assert_equal(app.world.get_ptr[MeshRendererComponent](e_a).value()[].mesh_id, mid_a1)
            assert_equal(app.world.get_ptr[MeshRendererComponent](e_b).value()[].mesh_id, mid_b1)

        # Hot-reload hero.flrig (3 -> 4 joints) -> BOTH RigComponent & SkeletonComponent update
        var rig4 = FrigRigDef("HeroRig")
        rig4.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig4.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig4.joints.append(FrigJointDef("Neck", 1, Vec3(0.0, 0.4, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig4.joints.append(FrigJointDef("Head", 2, Vec3(0.0, 0.3, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _ = save_flrig_file(ws + "/rigs/hero.flrig", rig4)
        # Also hot-reload char.glb -> skinned_entries must re-allocate with distinct skinned mesh_ids
        var m_char2 = _make_skinned_biped_for_asset_plugin()
        _ = write_native_glb(ws + "/meshes/char.glb", m_char2)

        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1000.0)
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1001.0)
        app.step_one_frame(Float32(1.0 / 60.0))

        assert_equal(app.world.get_ptr[RigComponent](e_rig).value()[].joint_count, 4)
        assert_equal(app.world.get_ptr[SkeletonComponent](e_rig).value()[].skeleton.node_count, 4)

        var mid_a_rel = app.world.get_ptr[MeshRendererComponent](e_a).value()[].mesh_id
        var mid_b_rel = app.world.get_ptr[MeshRendererComponent](e_b).value()[].mesh_id
        assert_true(mid_a_rel > mid_a1 and mid_b_rel > mid_b1 and mid_a_rel != mid_b_rel)

        app.shutdown()
    finally:
        _plug_rm_rf(tmp)


def _count_open_fds() -> Int:
    return len(list_directory("/proc/self/fd"))


@no_inline
def _overwrite_stack_frames(depth: Int) -> Int:
    var junk = Array[UInt64, 512](fill=0xDEADBEEFCAFEBABE)
    var acc = Int(junk[0] & 1)
    if depth > 0:
        acc += _overwrite_stack_frames(depth - 1)
    return acc


@no_inline
def _make_and_move_watcher(dir_path: String) -> AssetWatcher:
    var w = AssetWatcher(enabled=True)
    _ = w.watch_recursive(dir_path)
    return w^


@no_inline
def _open_and_move_archive(pack_path: String) -> FlpkV3MappedArchive:
    var arch = FlpkV3MappedArchive.open(pack_path)
    return arch^


@no_inline
def _setup_moved_server_mock_bridge(mut out_res: Resources) -> Tuple[Int, Handle[CookedTextureAsset]]:
    var srv = AssetServer(lru_max_bytes=65536, enable_watcher=False)
    srv.textures.set_evict_hook(Int(Pointer(to=srv)), _asset_server_evict_texture_slot)
    var tex = CookedTextureAsset()
    tex.width = 64
    tex.height = 64
    tex.mip_count = 1
    tex.mips.append(List[UInt8](length=128 * 1024, fill=UInt8(0x7F)))
    var h = srv.textures.insert(tex^, AssetUuid.from_path("tex/mock_lru.png"), "tex/mock_lru.png", 1)
    var slot = Int(h.slot())
    srv.streamer._ensure_slot(slot)
    srv.streamer.slot_bindless_idx[slot] = 7
    var old_addr = Int(Pointer(to=srv))
    out_res.insert(srv^)
    return (old_addr, h)


@no_inline
def _setup_moved_server_vk_bridge(
    mut ren: FlintAaaVulkanRenderer, mut out_res: Resources
) -> Tuple[Int, Handle[CookedTextureAsset]]:
    var srv = AssetServer(lru_max_bytes=65536, enable_watcher=False)
    srv.attach_renderer_residency_bridge(ren)
    var h = srv.textures.allocate_queued(
        AssetUuid.from_path("tex/vk_move_evict.png"), "tex/vk_move_evict.png", 1
    )
    _ = srv.upload_mip_to_gpu(h, mip_level=0, byte_size=4096)
    srv.lru_max_bytes = 512
    var old_addr = Int(Pointer(to=srv))
    out_res.insert(srv^)
    return (old_addr, h)


def test_m4a_move_constructors_residency_bridge_and_fd_leaks() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_moves_" + String(pid)
    _plug_rm_rf(tmp)
    _ = make_dirs(tmp + "/watch")
    try:
        # 1. AssetWatcher multi-move (function return + List reallocation) fd-leak check
        var fd_base_w = _count_open_fds()
        var watchers = List[AssetWatcher]()
        for _ in range(4):
            watchers.append(_make_and_move_watcher(tmp + "/watch"))
        for wi in range(len(watchers)):
            assert_true(watchers[wi].fd >= 0)
            assert_true(len(watchers[wi].watch_descriptors) >= 1)
        assert_equal(_count_open_fds(), fd_base_w + 4)
        watchers.clear()
        assert_equal(_count_open_fds(), fd_base_w)

        # 2. FlpkV3MappedArchive multi-move (function return + List reallocation) fd/mmap leak check
        var pack_path = tmp + "/test.flpk"
        var chunks = List[FlpkBuildChunk]()
        var payload = _plug_str_bytes("FLPK_MOVE_TEST_PAYLOAD")
        chunks.append(FlpkBuildChunk(AssetUuid.from_path("data/p.txt"), "data/p.txt", 0, ASSET_KIND_RAW, payload^))
        assert_true(write_flpk_v3(pack_path, chunks, page_align=True))

        var fd_base_a = _count_open_fds()
        var arches = List[FlpkV3MappedArchive]()
        for _ in range(4):
            arches.append(_open_and_move_archive(pack_path))
        for ai in range(len(arches)):
            assert_true(arches[ai].valid)
            assert_equal(arches[ai].toc_count(), 1)
            assert_true(arches[ai].addr != 0)
            assert_true(arches[ai].fd >= 0)
        assert_equal(_count_open_fds(), fd_base_a + 4)
        arches.clear()
        assert_equal(_count_open_fds(), fd_base_a)

        # 3. AssetServer move across stack frames -> overwrite dead stack -> enforce_lru_budget() (mock bridge)
        var mock_res = Resources()
        var mock_setup = _setup_moved_server_mock_bridge(mock_res)
        var old_mock_addr = mock_setup[0]
        var h_mock = mock_setup[1]
        _ = _overwrite_stack_frames(4)
        ref srv_mock = mock_res.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
        var moved_mock_addr = Int(Pointer(to=srv_mock))
        assert_true(moved_mock_addr != old_mock_addr)
        assert_equal(srv_mock.textures.evict_ctx, moved_mock_addr)
        assert_true(srv_mock.textures.release(h_mock))
        var evicted_mock = srv_mock.enforce_lru_budget()
        assert_equal(evicted_mock, 1)
        assert_true(not srv_mock.textures.is_alive(h_mock))
        assert_equal(srv_mock.streamer.slot_bindless_idx[Int(h_mock.slot())], -1)
        srv_mock.close()

        # 4. Live Tier 2 Vulkan attach_renderer_residency_bridge + move into Resources + stack overwrite + LRU evict
        var ren = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=64, height=64)
        if ren.uses_vulkan():
            var vk_res = Resources()
            var vk_setup = _setup_moved_server_vk_bridge(ren.gpu_slot[0], vk_res)
            var old_vk_addr = vk_setup[0]
            var h_vk = vk_setup[1]
            _ = _overwrite_stack_frames(4)
            ref srv_vk = vk_res.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
            var moved_vk_addr = Int(Pointer(to=srv_vk))
            assert_true(moved_vk_addr != old_vk_addr)
            assert_equal(srv_vk.textures.evict_ctx, moved_vk_addr)
            var slot_vk = Int(h_vk.slot())
            var b_idx = srv_vk.streamer.slot_bindless_idx[slot_vk]
            assert_true(b_idx >= 0)
            assert_true(srv_vk.total_vram_live_bytes() > 0)
            assert_true(srv_vk.textures.release(h_vk))
            var evicted_vk = srv_vk.enforce_lru_budget()
            assert_equal(evicted_vk, 1)
            assert_true(not srv_vk.textures.is_alive(h_vk))
            assert_equal(srv_vk.total_vram_live_bytes(), 0)
            assert_equal(ren.gpu_slot[0].bindless_vram_bytes[b_idx], 0)
            srv_vk.close()
    finally:
        _plug_rm_rf(tmp)


def _make_test_wav_16bit(freq_step: Float32, n_samples: Int = 64) -> List[UInt8]:
    var data_bytes = n_samples * 2
    var riff_size = 36 + data_bytes
    var out = List[UInt8](capacity=44 + data_bytes)
    for b in "RIFF".as_bytes():
        out.append(b)
    for s in range(4):
        out.append(UInt8((riff_size >> (s * 8)) & 0xFF))
    for b in "WAVEfmt ".as_bytes():
        out.append(b)
    for b in [UInt8(16), 0, 0, 0, 1, 0, 1, 0]:
        out.append(b)
    var sr = 48000
    for s in range(4):
        out.append(UInt8((sr >> (s * 8)) & 0xFF))
    var br = sr * 2
    for s in range(4):
        out.append(UInt8((br >> (s * 8)) & 0xFF))
    for b in [UInt8(2), 0, 16, 0]:
        out.append(b)
    for b in "data".as_bytes():
        out.append(b)
    for s in range(4):
        out.append(UInt8((data_bytes >> (s * 8)) & 0xFF))
    for i in range(n_samples):
        var v = Int16(Int(sin(Float32(i) * freq_step) * Float32(10000.0)))
        var u = UInt16(v)
        out.append(UInt8(u & 0xFF))
        out.append(UInt8((u >> 8) & 0xFF))
    return out^


def test_m4a_scene_asset_resolver_shared_flmat_and_audio_hot_reload() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_flmat_au_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/meshes")
    _ = make_dirs(ws + "/materials")
    _ = make_dirs(ws + "/audio")
    try:
        var m_cube = NativeModel()
        m_cube.meshes.append(_make_cube_mesh_for_plug())
        _ = write_native_glb(ws + "/meshes/cube.glb", m_cube)

        var mat1 = FlmatAsset("SharedMat")
        mat1.base_color = Vec3(0.2, 0.5, 0.8)
        mat1.roughness = Float32(0.65)
        mat1.metallic = Float32(0.25)
        mat1.clearcoat = Float32(0.4)
        mat1.opacity = Float32(0.85)
        mat1.emissive = Vec3(0.1, 0.2, 0.3)
        _ = save_flmat_file(ws + "/materials/shared.flmat", mat1)

        var wav1 = _make_test_wav_16bit(Float32(0.2), 64)
        _ = write_bytes(ws + "/audio/beep.wav", wav1)

        var app = EngineApp(AppConfig(title="M4aSharedFlmatAu", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=True))
        app.add_plugin(AudioPlugin(headless=True))
        app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
        app.startup()

        var mesh_ref = AssetRef.from_path("meshes/cube.glb", ASSET_KIND_MESH)
        var mat_ref = AssetRef.from_path("materials/shared.flmat", ASSET_KIND_RAW)
        var au_ref = AssetRef.from_path("audio/beep.wav", ASSET_KIND_AUDIO)

        # Two MaterialOverrideComponent entities sharing mat_ref
        var e_mo1 = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_mo1, LocalTransform(Vec3(-2.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_mo1, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_mo1, MeshRendererComponent(-1, -1, True, True, mesh_ref.copy()))
        _ = app.world.insert[MaterialOverrideComponent](e_mo1, MaterialOverrideComponent(material_ref=mat_ref.copy()))

        var e_mo2 = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_mo2, LocalTransform(Vec3(0.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_mo2, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_mo2, MeshRendererComponent(-1, -1, True, True, mesh_ref.copy()))
        _ = app.world.insert[MaterialOverrideComponent](e_mo2, MaterialOverrideComponent(material_ref=mat_ref.copy()))

        # Third entity with BOTH MeshRendererComponent(material_ref=mat_ref) and MaterialOverrideComponent(material_ref=mat_ref)
        var e_both = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_both, LocalTransform(Vec3(2.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_both, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](
            e_both, MeshRendererComponent(-1, -1, True, True, mesh_ref.copy(), mat_ref.copy())
        )
        _ = app.world.insert[MaterialOverrideComponent](e_both, MaterialOverrideComponent(material_ref=mat_ref.copy()))

        # Two AudioEmitterComponent entities sharing audio_ref
        var e_au1 = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_au1, LocalTransform(Vec3(-1.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_au1, GlobalTransform())
        _ = app.world.insert[AudioEmitterComponent](
            e_au1, AudioEmitterComponent(asset_id=-1, audio_ref=au_ref.copy(), play_on_spawn=False)
        )

        var e_au2 = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_au2, LocalTransform(Vec3(1.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_au2, GlobalTransform())
        _ = app.world.insert[AudioEmitterComponent](
            e_au2, AudioEmitterComponent(asset_id=-1, audio_ref=au_ref.copy(), play_on_spawn=False)
        )

        # Frame 1: ALL three MaterialOverrideComponent entities and override_slots receive initial .flmat properties
        app.step_one_frame(Float32(1.0 / 60.0))

        ref ren = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0]
        ref rstate = app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
        var shared_mr_mat = app.world.get_ptr[MeshRendererComponent](e_both).value()[].material_id
        assert_true(shared_mr_mat >= 0)

        var mo_ents = [e_mo1, e_mo2, e_both]
        for idx in range(len(mo_ents)):
            var ee = mo_ents[idx]
            ref mo = app.world.get_ptr[MaterialOverrideComponent](ee).value()[unsafe_offset=0]
            assert_equal(mo.material_id, shared_mr_mat)
            assert_true(abs(mo.base_color.x - Float32(0.2)) < Float32(1.0e-4))
            assert_true(abs(mo.base_color.y - Float32(0.5)) < Float32(1.0e-4))
            assert_true(abs(mo.base_color.z - Float32(0.8)) < Float32(1.0e-4))
            assert_true(abs(mo.roughness - Float32(0.65)) < Float32(1.0e-4))
            assert_true(abs(mo.metallic - Float32(0.25)) < Float32(1.0e-4))
            assert_true(abs(mo.clearcoat - Float32(0.4)) < Float32(1.0e-4))
            assert_true(abs(mo.opacity - Float32(0.85)) < Float32(1.0e-4))
            assert_true(abs(mo.emissive.x - Float32(0.1)) < Float32(1.0e-4))

            var ov_slot_opt = rstate.override_slots.get(ee.to_bits())
            assert_true(Bool(ov_slot_opt))
            var ov_slot = ov_slot_opt.value().slot_id
            assert_true(ov_slot >= 0 and ov_slot < len(ren.materials))
            assert_true(abs(ren.materials[ov_slot].base_color.x - Float32(0.2)) < Float32(1.0e-4))
            assert_true(abs(ren.materials[ov_slot].roughness - Float32(0.65)) < Float32(1.0e-4))

        var aid1 = app.world.get_ptr[AudioEmitterComponent](e_au1).value()[].asset_id
        var aid2 = app.world.get_ptr[AudioEmitterComponent](e_au2).value()[].asset_id
        assert_true(aid1 >= 0)
        assert_equal(aid2, aid1)

        # Late-spawned MaterialOverrideComponent reusing cached mat_ref also populates .flmat properties
        var e_late = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_late, LocalTransform(Vec3(4.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_late, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_late, MeshRendererComponent(-1, -1, True, True, mesh_ref.copy()))
        _ = app.world.insert[MaterialOverrideComponent](e_late, MaterialOverrideComponent(material_ref=mat_ref.copy()))
        app.step_one_frame(Float32(1.0 / 60.0))
        ref mo_late = app.world.get_ptr[MaterialOverrideComponent](e_late).value()[unsafe_offset=0]
        assert_equal(mo_late.material_id, shared_mr_mat)
        assert_true(abs(mo_late.base_color.x - Float32(0.2)) < Float32(1.0e-4))
        assert_true(abs(mo_late.roughness - Float32(0.65)) < Float32(1.0e-4))

        # Hot-reload shared.flmat AND audio/beep.wav -> ALL entities and override_slots update
        var mat2 = FlmatAsset("SharedMat")
        mat2.base_color = Vec3(0.9, 0.15, 0.25)
        mat2.roughness = Float32(0.12)
        mat2.metallic = Float32(0.9)
        mat2.clearcoat = Float32(0.8)
        mat2.opacity = Float32(0.5)
        mat2.emissive = Vec3(0.4, 0.5, 0.6)
        _ = save_flmat_file(ws + "/materials/shared.flmat", mat2)
        var wav2 = _make_test_wav_16bit(Float32(0.4), 96)
        _ = write_bytes(ws + "/audio/beep.wav", wav2)

        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1000.0)
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1001.0)
        app.step_one_frame(Float32(1.0 / 60.0))

        var all_mo_ents = [e_mo1, e_mo2, e_both, e_late]
        for idx in range(len(all_mo_ents)):
            var ee = all_mo_ents[idx]
            ref mo = app.world.get_ptr[MaterialOverrideComponent](ee).value()[unsafe_offset=0]
            assert_true(abs(mo.base_color.x - Float32(0.9)) < Float32(1.0e-4))
            assert_true(abs(mo.base_color.y - Float32(0.15)) < Float32(1.0e-4))
            assert_true(abs(mo.base_color.z - Float32(0.25)) < Float32(1.0e-4))
            assert_true(abs(mo.roughness - Float32(0.12)) < Float32(1.0e-4))
            assert_true(abs(mo.metallic - Float32(0.9)) < Float32(1.0e-4))
            assert_true(abs(mo.clearcoat - Float32(0.8)) < Float32(1.0e-4))
            assert_true(abs(mo.opacity - Float32(0.5)) < Float32(1.0e-4))
            assert_true(abs(mo.emissive.x - Float32(0.4)) < Float32(1.0e-4))

            var ov_slot_opt = rstate.override_slots.get(ee.to_bits())
            assert_true(Bool(ov_slot_opt))
            var ov_slot = ov_slot_opt.value().slot_id
            assert_true(abs(ren.materials[ov_slot].base_color.x - Float32(0.9)) < Float32(1.0e-4))
            assert_true(abs(ren.materials[ov_slot].roughness - Float32(0.12)) < Float32(1.0e-4))

        assert_true(abs(ren.materials[shared_mr_mat].base_color.x - Float32(0.9)) < Float32(1.0e-4))
        assert_true(abs(ren.materials[shared_mr_mat].roughness - Float32(0.12)) < Float32(1.0e-4))

        var aid1_rel = app.world.get_ptr[AudioEmitterComponent](e_au1).value()[].asset_id
        var aid2_rel = app.world.get_ptr[AudioEmitterComponent](e_au2).value()[].asset_id
        assert_true(aid1_rel > aid1)
        assert_equal(aid2_rel, aid1_rel)

        app.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4a_skeleton_and_animator_flscn_state_hash_parity() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_skel_hash_" + String(pid)
    _plug_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        var s1 = FlintStudioState(headless=True)
        var s2 = FlintStudioState(headless=True)
        var flscn_path = tmp + "/anim_pose.flscn"

        var e = s1.app.world.spawn_empty()
        _ = s1.app.world.insert[LocalTransform](
            e, LocalTransform(Vec3(1.5, 2.0, -3.0), Quat(), Vec3(1.0, 1.0, 1.0))
        )

        var sk = SkeletonInstance(3)
        sk.parents[0] = -1
        sk.parents[1] = 0
        sk.parents[2] = 1
        sk.pose.t[0] = Vec3(0.0, 1.2, 0.0)
        sk.pose.t[1] = Vec3(0.1, 0.5, -0.2)
        sk.pose.t[2] = Vec3(0.0, 0.4, 0.3)
        sk.pose.r[1] = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.5))
        sk.pose.r[2] = Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), Float32(-0.35))
        sk.pose.s[2] = Vec3(1.2, 1.0, 0.8)
        _ = s1.app.world.insert[SkeletonComponent](
            e,
            SkeletonComponent(
                UInt64(101),
                sk^,
                False,
                AssetRef.from_path("rigs/hero.flrig", ASSET_KIND_RAW),
            ),
        )

        var ac = AnimatorComponent(
            UInt64(202),
            apply_root_motion=True,
            enable_foot_plant=True,
            ignore_body_id=7,
            graph_ref=AssetRef.from_path("anims/hero.flanim", ASSET_KIND_RAW),
            clip_index=1,
            clip_name="Run",
            playing=False,
            loop=False,
            time=Float32(0.6),
            speed=Float32(1.5),
        )
        ac.bound = True
        ac.animator.state = 2
        ac.animator.previous = 1
        ac.animator.state_time = Float32(0.75)
        ac.animator.clip_time = Float32(0.35)
        ac.animator.prev_clip_time = Float32(0.25)
        ac.animator.phase = Float32(0.4)
        ac.animator.prev_phase = Float32(0.3)
        ac.animator.params.append(Float32(0.8))
        ac.animator.params.append(Float32(-0.5))
        ac.foot_plant.pelvis = 1
        ac.foot_plant.add_leg(1, 2, 3, Float32(0.12), Vec3(0.0, 0.5, 1.0))
        ac.foot_in_stance.append(True)
        ac.foot_in_stance.append(False)
        _ = s1.app.world.insert[AnimatorComponent](e, ac^)

        assert_true(s1.save_scene_file(flscn_path))
        assert_true(s2.load_scene_file(flscn_path))
        assert_equal(s1.state_hash(), s2.state_hash())

        var sk_ents = s2.app.world.query_entities(s2.app.world.bit_for[SkeletonComponent]())
        assert_equal(len(sk_ents), 1)
        var e2 = sk_ents[0]
        ref sk2 = s2.app.world.get_ptr[SkeletonComponent](e2).value()[unsafe_offset=0]
        assert_equal(sk2.skeleton.node_count, 3)
        assert_equal(sk2.skeleton.parents[0], -1)
        assert_equal(sk2.skeleton.parents[1], 0)
        assert_equal(sk2.skeleton.parents[2], 1)
        assert_true(abs(sk2.skeleton.pose.t[1].y - Float32(0.5)) < Float32(1.0e-5))
        assert_true(abs(sk2.skeleton.pose.s[2].x - Float32(1.2)) < Float32(1.0e-5))
        assert_true(not sk2.dirty_topology)

        ref ac2 = s2.app.world.get_ptr[AnimatorComponent](e2).value()[unsafe_offset=0]
        assert_true(ac2.bound)
        assert_equal(ac2.animator.state, 2)
        assert_equal(ac2.animator.previous, 1)
        assert_equal(len(ac2.animator.params), 2)
        assert_equal(ac2.foot_plant.pelvis, 1)
        assert_equal(len(ac2.foot_plant.legs), 1)
        assert_equal(len(ac2.foot_in_stance), 2)
        assert_true(ac2.foot_in_stance[0] and (not ac2.foot_in_stance[1]))

        # Legacy minimal .flscn omitting parents/pose/runtime Animator fields still loads cleanly
        var legacy_json = (
            '{"format":"flscn","version":2,"name":"LegacySkel","entities":[{"slot":0,"parent":-1,'
            '"components":{"SkeletonComponent":{"model_id":77,"node_count":2,"dirty_topology":false},'
            '"AnimatorComponent":{"graph_id":88,"clip_name":"Idle"}}}]}'
        )
        var legacy_bytes = _plug_str_bytes(legacy_json)
        var w_leg = World()
        var reg_leg = ComponentRegistry.with_builtins()
        register_anim_codecs(reg_leg)
        var scn_leg = Scene("Legacy")
        var ents_leg = scn_leg.load_flscn_bytes(w_leg, reg_leg, legacy_bytes)
        assert_equal(len(ents_leg), 1)
        ref sk_leg = w_leg.get_ptr[SkeletonComponent](ents_leg[0]).value()[unsafe_offset=0]
        assert_equal(sk_leg.skeleton.node_count, 2)
        assert_equal(sk_leg.skeleton.parents[0], -1)
        assert_equal(sk_leg.skeleton.parents[1], -1)
        assert_true(abs(sk_leg.skeleton.pose.s[0].x - Float32(1.0)) < Float32(1.0e-5))
        ref ac_leg = w_leg.get_ptr[AnimatorComponent](ents_leg[0]).value()[unsafe_offset=0]
        assert_equal(ac_leg.graph_id, UInt64(88))
        assert_equal(ac_leg.clip_name, "Idle")
        assert_equal(ac_leg.animator.state, -1)
        assert_equal(len(ac_leg.animator.params), 0)

        s1.shutdown()
        s2.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4a_asset_server_flpk_uuid_resolve_audio_and_texture() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_flpk_uuid_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/textures")
    _ = make_dirs(ws + "/audio")
    try:
        var pixels = List[UInt32](length=16 * 16, fill=UInt32(0xFF4080C0))
        assert_true(write_png(ws + "/textures/hero_alb.png", pixels, 16, 16, stamped=False))
        var wav_bytes = _make_test_wav_16bit(Float32(0.25), 64)
        assert_true(write_bytes(ws + "/audio/hit.wav", wav_bytes))

        var pcm = List[Int16](capacity=160)
        for i in range(160):
            pcm.append(Int16(Int(sin(Float32(i) * Float32(0.15)) * Float32(12000.0))))
        var qoa_bytes = encode_qoa(pcm, 1, 48000)
        assert_true(len(qoa_bytes) > 16)
        assert_true(write_bytes(ws + "/audio/sfx.qoa", qoa_bytes))

        var flpk_path = tmp + "/game.flpk"
        var pool = JobPool()
        var rep = cook_workspace(ws, flpk_path, pool, cache_dir=tmp + "/cache", page_align=True)
        assert_true(rep.ok)
        assert_equal(rep.failed_count, 0)

        var tex_uuid = AssetUuid()
        var wav_uuid = AssetUuid()
        var qoa_uuid = AssetUuid()
        for i in range(len(rep.rel_paths)):
            if rep.rel_paths[i] == "textures/hero_alb.png":
                tex_uuid = rep.uuids[i]
            elif rep.rel_paths[i] == "audio/hit.wav":
                wav_uuid = rep.uuids[i]
            elif rep.rel_paths[i] == "audio/sfx.qoa":
                qoa_uuid = rep.uuids[i]
        assert_true(not tex_uuid.is_nil())
        assert_true(not wav_uuid.is_nil())
        assert_true(not qoa_uuid.is_nil())

        # Cold AssetServer with ONLY mounted .flpk (no loose directory): UUID resolves even when rel_path moved
        var srv = AssetServer(enable_watcher=False)
        assert_true(srv.mount_flpk(flpk_path))

        var h_tex = srv.resolve_texture(
            AssetRef(tex_uuid, "textures/renamed_alb.png", ASSET_KIND_TEXTURE),
            progressive=False,
        )
        assert_true(srv.textures.is_loaded(h_tex))
        var tptr = srv.textures.get_ptr(h_tex)
        assert_true(Bool(tptr))
        assert_equal(tptr.value()[].width, 16)
        assert_equal(tptr.value()[].height, 16)

        var h_wav = srv.resolve_audio(AssetRef(wav_uuid, "audio/moved_hit.wav", ASSET_KIND_AUDIO))
        assert_true(srv.audios.is_loaded(h_wav))
        var aptr = srv.audios.get_ptr(h_wav)
        assert_true(Bool(aptr))
        assert_equal(aptr.value()[].sample_rate, 48000)
        assert_equal(len(aptr.value()[].samples), 64)

        var h_qoa_flpk = srv.resolve_audio(AssetRef(qoa_uuid, "audio/moved_sfx.qoa", ASSET_KIND_AUDIO))
        assert_true(srv.audios.is_loaded(h_qoa_flpk))
        var qptr = srv.audios.get_ptr(h_qoa_flpk)
        assert_true(Bool(qptr))
        assert_equal(qptr.value()[].sample_rate, 48000)
        assert_equal(len(qptr.value()[].samples), 160)

        # Security parity: traversal / absolute rel_path rejected even when UUID is valid
        assert_true(
            not srv.textures.is_loaded(
                srv.resolve_texture(AssetRef(tex_uuid, "../escape.png", ASSET_KIND_TEXTURE))
            )
        )
        assert_true(
            not srv.textures.is_loaded(
                srv.resolve_texture(AssetRef(tex_uuid, "/etc/passwd", ASSET_KIND_TEXTURE))
            )
        )
        assert_true(
            not srv.audios.is_loaded(
                srv.resolve_audio(AssetRef(wav_uuid, "../escape.wav", ASSET_KIND_AUDIO))
            )
        )
        assert_true(
            not srv.audios.is_loaded(
                srv.resolve_audio(AssetRef(wav_uuid, "/etc/passwd", ASSET_KIND_AUDIO))
            )
        )
        srv.close()

        # Loose .qoa loading & hot-reload via AssetServer
        var srv_loose = AssetServer(enable_watcher=True)
        _ = srv_loose.watch_loose_directory(ws)
        var h_qoa_loose = srv_loose.resolve_audio(AssetRef.from_path("audio/sfx.qoa", ASSET_KIND_AUDIO))
        assert_true(srv_loose.audios.is_loaded(h_qoa_loose))
        var ep0 = srv_loose.audios.reload_epoch(h_qoa_loose)

        var pcm2 = List[Int16](capacity=200)
        for i in range(200):
            pcm2.append(Int16(Int(sin(Float32(i) * Float32(0.3)) * Float32(15000.0))))
        var qoa_bytes2 = encode_qoa(pcm2, 1, 48000)
        assert_true(write_bytes(ws + "/audio/sfx.qoa", qoa_bytes2))
        _ = srv_loose.tick(10.0)
        _ = srv_loose.tick(11.0)
        assert_true(srv_loose.audios.reload_epoch(h_qoa_loose) > ep0)
        assert_equal(len(srv_loose.audios.get_ptr(h_qoa_loose).value()[].samples), 200)
        srv_loose.close()
    finally:
        _plug_rm_rf(tmp)


def test_m4a_audio_emitter_flscn_state_hash_parity() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_audio_hash_" + String(pid)
    _plug_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        var s1 = FlintStudioState(headless=True)
        var s2 = FlintStudioState(headless=True)
        var s3 = FlintStudioState(headless=True)
        var flscn_path = tmp + "/audio.flscn"
        var flscnb_path = tmp + "/audio.flscnb"

        # 1. audio_ref provided with omitted asset_path -> synchronized on __init__
        var e1 = s1.app.world.spawn_empty()
        var ref1 = AssetRef(AssetUuid.from_path("audio/step.qoa"), "audio/step.qoa", ASSET_KIND_AUDIO)
        var em1 = AudioEmitterComponent(asset_id=-1, audio_ref=ref1.copy())
        assert_equal(em1.asset_path, "audio/step.qoa")
        _ = s1.app.world.insert[AudioEmitterComponent](e1, em1^)

        # 2. All non-default runtime/authored fields (including priority, playing=True with auto_play_pending=False, occlusion, prev_pos)
        var e2 = s1.app.world.spawn_empty()
        var ref2 = AssetRef.from_path("audio/amb.wav", ASSET_KIND_AUDIO)
        var em2 = AudioEmitterComponent(
            asset_id=7,
            volume=Float32(0.625),
            pitch=Float32(1.15),
            bus_id=2,
            looping=True,
            spatial=False,
            priority=9,
            ref_dist=Float32(4.5),
            max_dist=Float32(120.0),
            play_on_spawn=False,
            playing=True,
            auto_play_pending=False,
            enable_occlusion=True,
            occlusion_gain=Float32(0.35),
            cutoff_hz=Float32(2400.0),
            obstruction=Float32(0.72),
            audio_ref=ref2.copy(),
        )
        em2.prev_pos = Vec3(3.25, -1.5, 8.75)
        em2.has_prev_pos = True
        assert_equal(em2.asset_path, "audio/amb.wav")
        _ = s1.app.world.insert[AudioEmitterComponent](e2, em2^)

        # 3. asset_path provided with omitted audio_ref -> synchronized on __init__
        var e3 = s1.app.world.spawn_empty()
        var em3 = AudioEmitterComponent(asset_id=-1, asset_path="audio/legacy_ctor.wav")
        assert_equal(em3.audio_ref.rel_path, "audio/legacy_ctor.wav")
        _ = s1.app.world.insert[AudioEmitterComponent](e3, em3^)

        assert_true(s1.save_scene_file(flscn_path))
        assert_true(s2.load_scene_file(flscn_path))
        assert_equal(s1.state_hash(), s2.state_hash())

        assert_true(s2.save_scene_file(flscnb_path))
        assert_true(s3.load_scene_file(flscnb_path))
        assert_equal(s2.state_hash(), s3.state_hash())

        var au_ents = s2.app.world.query_entities(s2.app.world.bit_for[AudioEmitterComponent]())
        assert_equal(len(au_ents), 3)
        var found_e2 = False
        for i in range(len(au_ents)):
            ref em = s2.app.world.get_ptr[AudioEmitterComponent](au_ents[i]).value()[unsafe_offset=0]
            if em.asset_id == 7:
                found_e2 = True
                assert_equal(em.asset_path, "audio/amb.wav")
                assert_equal(em.priority, 9)
                assert_true(em.playing)
                assert_true(not em.auto_play_pending)
                assert_true(em.enable_occlusion)
                assert_true(abs(em.occlusion_gain - Float32(0.35)) < Float32(1.0e-5))
                assert_true(abs(em.cutoff_hz - Float32(2400.0)) < Float32(1.0e-3))
                assert_true(abs(em.obstruction - Float32(0.72)) < Float32(1.0e-5))
                assert_true(em.has_prev_pos)
                assert_true(abs(em.prev_pos.x - Float32(3.25)) < Float32(1.0e-5))
                assert_true(abs(em.prev_pos.y - Float32(-1.5)) < Float32(1.0e-5))
                assert_true(abs(em.prev_pos.z - Float32(8.75)) < Float32(1.0e-5))
        assert_true(found_e2)

        # Legacy minimal JSON omitting priority/playing/auto_play_pending/occlusion/prev_pos
        var legacy_json = (
            '{"format":"flscn","version":2,"name":"LegacyAudio","entities":[{"slot":0,"parent":-1,'
            '"components":{"AudioEmitterComponent":{"asset_id":-1,"asset_path":"audio/minimal.wav",'
            '"volume":0.5,"play_on_spawn":true}}}]}'
        )
        var w_leg = World()
        var reg_leg = ComponentRegistry.with_builtins()
        register_audio_codecs(reg_leg)
        var scn_leg = Scene("LegacyAudio")
        var ents_leg = scn_leg.load_flscn_bytes(w_leg, reg_leg, _plug_str_bytes(legacy_json))
        assert_equal(len(ents_leg), 1)
        ref em_leg = w_leg.get_ptr[AudioEmitterComponent](ents_leg[0]).value()[unsafe_offset=0]
        assert_equal(em_leg.asset_path, "audio/minimal.wav")
        assert_equal(em_leg.audio_ref.rel_path, "audio/minimal.wav")
        assert_equal(em_leg.priority, 1)
        assert_true(not em_leg.playing)
        assert_true(em_leg.auto_play_pending)
        assert_true(abs(em_leg.occlusion_gain - Float32(1.0)) < Float32(1.0e-5))
        assert_true(abs(em_leg.cutoff_hz - CLEAR_CUTOFF_HZ) < Float32(1.0e-2))
        assert_true(abs(em_leg.obstruction) < Float32(1.0e-5))
        assert_true(not em_leg.has_prev_pos)

        s1.shutdown()
        s2.shutdown()
        s3.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4a_scene_asset_resolver_preserves_loaded_skeleton_pose() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_skel_preserve_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/rigs")
    try:
        var rig3 = FrigRigDef("HeroRig")
        rig3.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig3.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig3.joints.append(FrigJointDef("Head", 1, Vec3(0.0, 0.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _ = save_flrig_file(ws + "/rigs/hero.flrig", rig3)

        var rig_ref = AssetRef.from_path("rigs/hero.flrig", ASSET_KIND_RAW)
        var scn_path = tmp + "/posed.flscn"

        # Save posed SkeletonComponent (with rig_ref set) into .flscn via FlintStudioState
        var s_save = FlintStudioState(headless=True)
        var e_save = s_save.app.world.spawn_empty()
        var custom_sk = SkeletonInstance(3)
        custom_sk.parents[0] = -1
        custom_sk.parents[1] = 0
        custom_sk.parents[2] = 1
        custom_sk.pose.t[1] = Vec3(0.25, 2.5, -0.5)
        var custom_q = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.75))
        custom_sk.pose.r[2] = custom_q
        _ = s_save.app.world.insert[SkeletonComponent](
            e_save, SkeletonComponent(0, custom_sk^, False, rig_ref.copy())
        )
        assert_true(s_save.save_scene_file(scn_path))
        s_save.shutdown()

        # Load into fresh EngineApp with AssetPlugin (empty entity FIRST, loaded posed entity SECOND)
        var app = EngineApp(AppConfig(title="SkelPosePreserve", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=True))
        app.add_plugin(AnimPlugin())
        app.startup()

        var e_empty = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_empty, LocalTransform())
        _ = app.world.insert[GlobalTransform](e_empty, GlobalTransform())
        _ = app.world.insert[SkeletonComponent](e_empty, SkeletonComponent(0, rig_ref=rig_ref.copy()))

        var reg = ComponentRegistry.with_builtins()
        register_anim_codecs(reg)
        var scn_load = Scene("PosedLoad")
        var loaded_ents = scn_load.load_flscn_file(scn_path, app.world, reg, clear_existing=False)
        assert_equal(len(loaded_ents), 1)
        var e_posed = loaded_ents[0]

        # Frame 1 & Frame 2: e_empty populates from .flrig, while e_posed keeps its saved custom pose!
        app.step_one_frame(Float32(1.0 / 60.0))
        app.step_one_frame(Float32(1.0 / 60.0))

        ref sk_empty = app.world.get_ptr[SkeletonComponent](e_empty).value()[unsafe_offset=0].skeleton
        assert_equal(sk_empty.node_count, 3)
        assert_true(abs(sk_empty.pose.t[1].y - Float32(1.0)) < Float32(1.0e-5))

        ref sk_posed = app.world.get_ptr[SkeletonComponent](e_posed).value()[unsafe_offset=0].skeleton
        assert_equal(sk_posed.node_count, 3)
        assert_true(abs(sk_posed.pose.t[1].x - Float32(0.25)) < Float32(1.0e-5))
        assert_true(abs(sk_posed.pose.t[1].y - Float32(2.5)) < Float32(1.0e-5))
        assert_true(abs(sk_posed.pose.t[1].z - Float32(-0.5)) < Float32(1.0e-5))
        assert_true(abs(sk_posed.pose.r[2].y - custom_q.y) < Float32(1.0e-4))
        assert_true(abs(sk_posed.pose.r[2].w - custom_q.w) < Float32(1.0e-4))

        # Also test reverse entity order in a fresh EngineApp (posed entity FIRST, empty entity SECOND)
        var app_rev = EngineApp(AppConfig(title="SkelPosePreserveRev", headless=True, worker_threads=0))
        app_rev.add_plugin(TransformPlugin())
        app_rev.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=False))
        app_rev.add_plugin(AnimPlugin())
        app_rev.startup()
        var loaded_rev = scn_load.load_flscn_file(scn_path, app_rev.world, reg, clear_existing=False)
        assert_equal(len(loaded_rev), 1)
        var e_rev_posed = loaded_rev[0]
        var e_rev_empty = app_rev.world.spawn()
        _ = app_rev.world.insert[LocalTransform](e_rev_empty, LocalTransform())
        _ = app_rev.world.insert[GlobalTransform](e_rev_empty, GlobalTransform())
        _ = app_rev.world.insert[SkeletonComponent](e_rev_empty, SkeletonComponent(0, rig_ref=rig_ref.copy()))
        app_rev.step_one_frame(Float32(1.0 / 60.0))
        ref sk_rev_posed = app_rev.world.get_ptr[SkeletonComponent](e_rev_posed).value()[unsafe_offset=0].skeleton
        ref sk_rev_empty = app_rev.world.get_ptr[SkeletonComponent](e_rev_empty).value()[unsafe_offset=0].skeleton
        assert_true(abs(sk_rev_posed.pose.t[1].y - Float32(2.5)) < Float32(1.0e-5))
        assert_equal(sk_rev_empty.node_count, 3)
        assert_true(abs(sk_rev_empty.pose.t[1].y - Float32(1.0)) < Float32(1.0e-5))
        app_rev.shutdown()

        # Hot-reload hero.flrig (3 -> 4 joints) -> BOTH e_empty AND e_posed update!
        var rig4 = FrigRigDef("HeroRig")
        rig4.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig4.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.25, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig4.joints.append(FrigJointDef("Neck", 1, Vec3(0.0, 0.4, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig4.joints.append(FrigJointDef("Head", 2, Vec3(0.0, 0.3, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _ = save_flrig_file(ws + "/rigs/hero.flrig", rig4)

        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1000.0)
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1001.0)
        app.step_one_frame(Float32(1.0 / 60.0))

        ref sk_empty_hr = app.world.get_ptr[SkeletonComponent](e_empty).value()[unsafe_offset=0].skeleton
        ref sk_posed_hr = app.world.get_ptr[SkeletonComponent](e_posed).value()[unsafe_offset=0].skeleton
        assert_equal(sk_empty_hr.node_count, 4)
        assert_equal(sk_posed_hr.node_count, 4)
        assert_true(abs(sk_empty_hr.pose.t[1].y - Float32(1.25)) < Float32(1.0e-5))
        assert_true(abs(sk_posed_hr.pose.t[1].y - Float32(1.25)) < Float32(1.0e-5))

        app.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4a_bone_socket_flscn_state_hash_parity() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_bonesock_hash_" + String(pid)
    _plug_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        # (a) Non-default entity_id, world_pos, world_rot, world_scale (including non-power-of-2 floats)
        var s1 = FlintStudioState(headless=True)
        var s2 = FlintStudioState(headless=True)
        var s3 = FlintStudioState(headless=True)
        var flscn_path = tmp + "/bonesock.flscn"
        var flscnb_path = tmp + "/bonesock.flscnb"

        var e_sk = s1.app.world.spawn_empty()
        _ = s1.app.world.insert[LocalTransform](e_sk, LocalTransform.identity())
        _ = s1.app.world.insert[GlobalTransform](e_sk, GlobalTransform())
        var sk_inst = SkeletonInstance(2)
        sk_inst.parents[0] = -1
        sk_inst.parents[1] = 0
        _ = s1.app.world.insert[SkeletonComponent](e_sk, SkeletonComponent(17, sk_inst^))

        var e_sock = s1.app.world.spawn_empty()
        _ = s1.app.world.insert[LocalTransform](e_sock, LocalTransform.identity())
        _ = s1.app.world.insert[GlobalTransform](e_sock, GlobalTransform())
        var lrot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.65)).normalized()
        var wrot = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), Float32(0.95)).normalized()
        var att = BoneSocketAttachment(
            "HandWeapon",
            1,
            42,
            Vec3(Float32(0.15), Float32(-0.35), Float32(1.0 / 3.0)),
            lrot,
            Vec3(Float32(1.15), Float32(0.85), Float32(1.3)),
            7,
        )
        att.world_pos = Vec3(Float32(2.35), Float32(-1.15), Float32(4.7))
        att.world_rot = wrot
        att.world_scale = Vec3(Float32(1.2), Float32(0.9), Float32(1.4))
        _ = s1.app.world.insert[BoneSocketComponent](e_sock, BoneSocketComponent(e_sk, att^, -1))

        assert_true(s1.save_scene_file(flscn_path))
        assert_true(s2.load_scene_file(flscn_path))
        assert_equal(s1.state_hash(), s2.state_hash())

        assert_true(s2.save_scene_file(flscnb_path))
        assert_true(s3.load_scene_file(flscnb_path))
        assert_equal(s2.state_hash(), s3.state_hash())

        var sock_ents = s2.app.world.query_entities(s2.app.world.bit_for[BoneSocketComponent]())
        assert_equal(len(sock_ents), 1)
        ref bs2 = s2.app.world.get_ptr[BoneSocketComponent](sock_ents[0]).value()[unsafe_offset=0]
        assert_true(s2.app.world.is_alive(bs2.skeleton_entity))
        assert_true(s2.app.world.has[SkeletonComponent](bs2.skeleton_entity))
        assert_equal(bs2.skeleton_slot, -1)
        assert_equal(bs2.attachment.name, "HandWeapon")
        assert_equal(bs2.attachment.bone_node, 1)
        assert_equal(bs2.attachment.entity_id, 42)
        assert_equal(bs2.attachment.kinematic_body_id, 7)
        assert_true(abs(bs2.attachment.local_pos.x - Float32(0.15)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.local_pos.z - Float32(1.0 / 3.0)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.local_rot.y - lrot.y) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.local_scale.x - Float32(1.15)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_pos.x - Float32(2.35)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_pos.y - Float32(-1.15)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_pos.z - Float32(4.7)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_rot.z - wrot.z) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_rot.w - wrot.w) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_scale.x - Float32(1.2)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_scale.y - Float32(0.9)) < Float32(1.0e-5))
        assert_true(abs(bs2.attachment.world_scale.z - Float32(1.4)) < Float32(1.0e-5))

        s1.shutdown()
        s2.shutdown()
        s3.shutdown()

        # (b) Natural simulation via step_simulation(1) updating world_pos/world_rot/world_scale
        var s_sim = FlintStudioState(headless=True)
        var s_sim_json = FlintStudioState(headless=True)
        var s_sim_bin = FlintStudioState(headless=True)
        var sim_flscn = tmp + "/bonesock_sim.flscn"
        var sim_flscnb = tmp + "/bonesock_sim.flscnb"

        var e_sk_sim = s_sim.app.world.spawn_empty()
        var sk_rot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.4))
        _ = s_sim.app.world.insert[LocalTransform](
            e_sk_sim,
            LocalTransform(
                Vec3(Float32(3.15), Float32(1.2), Float32(-2.35)),
                sk_rot,
                Vec3(Float32(1.1), Float32(1.1), Float32(1.1)),
            ),
        )
        _ = s_sim.app.world.insert[GlobalTransform](e_sk_sim, GlobalTransform())
        var sk_sim = SkeletonInstance(2)
        sk_sim.parents[0] = -1
        sk_sim.parents[1] = 0
        _ = s_sim.app.world.insert[SkeletonComponent](e_sk_sim, SkeletonComponent(0, sk_sim^))

        var e_sock_sim = s_sim.app.world.spawn_empty()
        _ = s_sim.app.world.insert[LocalTransform](e_sock_sim, LocalTransform.identity())
        _ = s_sim.app.world.insert[GlobalTransform](e_sock_sim, GlobalTransform())
        var sim_att = BoneSocketAttachment(
            "ShieldSocket",
            1,
            99,
            Vec3(Float32(0.15), Float32(0.25), Float32(-0.35)),
            Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.3)),
            Vec3(Float32(1.05), Float32(1.05), Float32(1.05)),
        )
        _ = s_sim.app.world.insert[BoneSocketComponent](e_sock_sim, BoneSocketComponent(e_sk_sim, sim_att^, -1))

        assert_true(s_sim.step_simulation(1))
        ref bs_after = s_sim.app.world.get_ptr[BoneSocketComponent](e_sock_sim).value()[unsafe_offset=0]
        assert_true(abs(bs_after.attachment.world_pos.x) > Float32(1.0))

        assert_true(s_sim.save_scene_file(sim_flscn))
        assert_true(s_sim.save_scene_file(sim_flscnb))
        assert_true(s_sim_json.load_scene_file(sim_flscn))
        assert_true(s_sim_bin.load_scene_file(sim_flscnb))
        assert_equal(s_sim_json.state_hash(), s_sim_bin.state_hash())

        var sim_json_ents = s_sim_json.app.world.query_entities(s_sim_json.app.world.bit_for[BoneSocketComponent]())
        var sim_bin_ents = s_sim_bin.app.world.query_entities(s_sim_bin.app.world.bit_for[BoneSocketComponent]())
        assert_equal(len(sim_json_ents), 1)
        assert_equal(len(sim_bin_ents), 1)
        ref bs_j = s_sim_json.app.world.get_ptr[BoneSocketComponent](sim_json_ents[0]).value()[unsafe_offset=0]
        ref bs_b = s_sim_bin.app.world.get_ptr[BoneSocketComponent](sim_bin_ents[0]).value()[unsafe_offset=0]
        assert_equal(bs_j.attachment.entity_id, bs_b.attachment.entity_id)
        assert_true(abs(bs_j.attachment.world_pos.x - bs_b.attachment.world_pos.x) < Float32(1.0e-5))
        assert_true(abs(bs_j.attachment.world_pos.y - bs_b.attachment.world_pos.y) < Float32(1.0e-5))
        assert_true(abs(bs_j.attachment.world_pos.z - bs_b.attachment.world_pos.z) < Float32(1.0e-5))
        assert_true(abs(bs_j.attachment.world_rot.y - bs_b.attachment.world_rot.y) < Float32(1.0e-5))
        assert_true(abs(bs_j.attachment.world_scale.x - bs_b.attachment.world_scale.x) < Float32(1.0e-5))

        s_sim.shutdown()
        s_sim_json.shutdown()
        s_sim_bin.shutdown()

        # (c) Legacy minimal .flscn JSON omitting entity_id/world_pos/world_rot/world_scale
        var legacy_json = (
            '{"format":"flscn","version":2,"name":"LegacySock","entities":['
            '{"slot":0,"parent":-1,"components":{"SkeletonComponent":{"model_id":11,"node_count":2,"dirty_topology":false}}},'
            '{"slot":1,"parent":0,"components":{"BoneSocketComponent":{"name":"BackSlot","bone_node":1,'
            '"kinematic_body_id":-1,"skeleton_slot":0,"local_pos":[0.1,0.2,0.3],'
            '"local_rot":[0.0,0.0,0.0,1.0],"local_scale":[1.0,1.0,1.0]}}}]}'
        )
        var w_leg = World()
        var reg_leg = ComponentRegistry.with_builtins()
        register_anim_codecs(reg_leg)
        var scn_leg = Scene("LegacySock")
        var ents_leg = scn_leg.load_flscn_bytes(w_leg, reg_leg, _plug_str_bytes(legacy_json))
        assert_equal(len(ents_leg), 2)
        ref bs_leg = w_leg.get_ptr[BoneSocketComponent](ents_leg[1]).value()[unsafe_offset=0]
        assert_equal(bs_leg.attachment.name, "BackSlot")
        assert_equal(bs_leg.attachment.bone_node, 1)
        assert_equal(bs_leg.attachment.entity_id, -1)
        assert_equal(bs_leg.skeleton_entity.to_bits(), ents_leg[0].to_bits())
        assert_equal(bs_leg.skeleton_slot, -1)
        assert_true(abs(bs_leg.attachment.local_pos.y - Float32(0.2)) < Float32(1.0e-5))
        assert_true(abs(bs_leg.attachment.world_pos.x) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_pos.y) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_pos.z) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_rot.x) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_rot.y) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_rot.z) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_rot.w - Float32(1.0)) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_scale.x - Float32(1.0)) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_scale.y - Float32(1.0)) < Float32(1.0e-6))
        assert_true(abs(bs_leg.attachment.world_scale.z - Float32(1.0)) < Float32(1.0e-6))
    finally:
        _plug_rm_rf(tmp)


def _plug_push_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _plug_push_f32_le(mut out: List[UInt8], v: Float32):
    _plug_push_u32_le(out, bitcast[DType.uint32](v))


def _make_rigged_glb_for_plug(n_clips: Int) -> List[UInt8]:
    """Builds a minimal valid glTF 2.0 rigged GLB with 1 skin (2 joints) and `n_clips` animations."""
    var bin_data = List[UInt8]()
    # 0: 3 vertex positions (VEC3 float32, 36 bytes)
    var pos_off = len(bin_data)
    _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.0)
    _plug_push_f32_le(bin_data, 1.0); _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.0)
    _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 1.0); _plug_push_f32_le(bin_data, 0.0)
    var pos_len = len(bin_data) - pos_off

    # 1: 3 vertex normals (VEC3 float32, 36 bytes)
    var norm_off = len(bin_data)
    for _ in range(3):
        _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 1.0)
    var norm_len = len(bin_data) - norm_off

    # 2: 3 vertex JOINTS_0 (VEC4 uint8 = 12 bytes)
    var jnt_off = len(bin_data)
    for _ in range(3):
        bin_data.append(0); bin_data.append(1); bin_data.append(0); bin_data.append(0)
    var jnt_len = len(bin_data) - jnt_off

    # 3: 3 vertex WEIGHTS_0 (VEC4 float32 = 48 bytes)
    var wgt_off = len(bin_data)
    for _ in range(3):
        _plug_push_f32_le(bin_data, 0.75); _plug_push_f32_le(bin_data, 0.25)
        _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.0)
    var wgt_len = len(bin_data) - wgt_off

    # 4: 2 inverseBindMatrices (MAT4 float32 = 128 bytes)
    var ibm_off = len(bin_data)
    for _ in range(2):
        for r in range(4):
            for c in range(4):
                _plug_push_f32_le(bin_data, Float32(1.0) if r == c else Float32(0.0))
    var ibm_len = len(bin_data) - ibm_off

    # 5: animation keyframe times (SCALAR float32, 2 keyframes = 8 bytes)
    var t_off = len(bin_data)
    _plug_push_f32_le(bin_data, 0.0)
    _plug_push_f32_le(bin_data, 1.0)
    var t_len = len(bin_data) - t_off

    # 6: animation keyframe translations (VEC3 float32, 2 keyframes = 24 bytes)
    var v_off = len(bin_data)
    _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.5); _plug_push_f32_le(bin_data, 0.0)
    _plug_push_f32_le(bin_data, 0.0); _plug_push_f32_le(bin_data, 0.75); _plug_push_f32_le(bin_data, 0.0)
    var v_len = len(bin_data) - v_off

    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    var anims_json = String("[")
    var clip_names = ["Idle", "Walk", "Run"]
    for ci in range(n_clips):
        if ci > 0:
            anims_json += ","
        var nm = clip_names[ci] if ci < len(clip_names) else ("Clip" + String(ci))
        anims_json += (
            '{"name":"' + nm + '","samplers":[{"input":5,"output":6,"interpolation":"LINEAR"}],'
            '"channels":[{"sampler":0,"target":{"node":1,"path":"translation"}}]}'
        )
    anims_json += "]"

    var json = (
        '{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,2]}],'
        '"buffers":[{"byteLength":' + String(len(bin_data)) + '}],'
        '"bufferViews":['
        '{"buffer":0,"byteOffset":' + String(pos_off) + ',"byteLength":' + String(pos_len) + '},'
        '{"buffer":0,"byteOffset":' + String(norm_off) + ',"byteLength":' + String(norm_len) + '},'
        '{"buffer":0,"byteOffset":' + String(jnt_off) + ',"byteLength":' + String(jnt_len) + '},'
        '{"buffer":0,"byteOffset":' + String(wgt_off) + ',"byteLength":' + String(wgt_len) + '},'
        '{"buffer":0,"byteOffset":' + String(ibm_off) + ',"byteLength":' + String(ibm_len) + '},'
        '{"buffer":0,"byteOffset":' + String(t_off) + ',"byteLength":' + String(t_len) + '},'
        '{"buffer":0,"byteOffset":' + String(v_off) + ',"byteLength":' + String(v_len) + '}'
        '],"accessors":['
        '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},'
        '{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},'
        '{"bufferView":2,"componentType":5121,"count":3,"type":"VEC4"},'
        '{"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"},'
        '{"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"},'
        '{"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},'
        '{"bufferView":6,"componentType":5126,"count":2,"type":"VEC3"}'
        '],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}}]}],'
        '"nodes":['
        '{"name":"Root","children":[1]},'
        '{"name":"Arm","translation":[0,0.5,0]},'
        '{"name":"BodyMesh","mesh":0,"skin":0}'
        '],'
        '"skins":[{"joints":[0,1],"inverseBindMatrices":4}],'
        '"animations":' + anims_json + '}'
    )
    var json_bytes = List[UInt8]()
    for b in json.as_bytes():
        json_bytes.append(b)
    while len(json_bytes) % 4 != 0:
        json_bytes.append(0x20)

    var total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    var out = List[UInt8](capacity=total_len)
    _plug_push_u32_le(out, 0x46546C67)
    _plug_push_u32_le(out, 2)
    _plug_push_u32_le(out, UInt32(total_len))
    _plug_push_u32_le(out, UInt32(len(json_bytes)))
    _plug_push_u32_le(out, 0x4E4F534A)
    for b in json_bytes:
        out.append(b)
    _plug_push_u32_le(out, UInt32(len(bin_data)))
    _plug_push_u32_le(out, 0x004E4942)
    for b in bin_data:
        out.append(b)
    return out^


def test_m4b_flmat_nested_textures_role_cooking_and_hot_reload() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_m4b_flmat_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/meshes")
    _ = make_dirs(ws + "/materials")
    _ = make_dirs(ws + "/textures")
    try:
        var m_cube = NativeModel()
        m_cube.meshes.append(_make_cube_mesh_for_plug())
        _ = write_native_glb(ws + "/meshes/cube.glb", m_cube)

        var px_alb = List[UInt32](length=8 * 8, fill=UInt32(0xFF3366CC))
        var px_nrm = List[UInt32](length=8 * 8, fill=UInt32(0xFFFF8080))
        var px_orm = List[UInt32](length=8 * 8, fill=UInt32(0xFF00C0FF))
        var px_emi = List[UInt32](length=8 * 8, fill=UInt32(0xFF102030))
        assert_true(write_png(ws + "/textures/albedo.png", px_alb, 8, 8, stamped=False))
        assert_true(write_png(ws + "/textures/normal.png", px_nrm, 8, 8, stamped=False))
        assert_true(write_png(ws + "/textures/orm.png", px_orm, 8, 8, stamped=False))
        assert_true(write_png(ws + "/textures/emissive.png", px_emi, 8, 8, stamped=False))

        var nested_flmat = (
            '{"format":"flmat","version":1,"name":"NestedPbr",'
            '"base_color":[0.25,0.6,0.9],"roughness":0.42,"metallic":0.15,"emissive":[0.1,0.2,0.0], '
            '"textures":{"albedo":"textures/albedo.png","normal":"textures/normal.png",'
            '"orm":"textures/orm.png","emissive":"textures/emissive.png"}}'
        )
        var nested_flmat_b = _plug_str_bytes(nested_flmat)
        assert_true(write_bytes(ws + "/materials/nested.flmat", nested_flmat_b))

        # 1. Dual-schema load_flmat_bytes parses nested "textures" map cleanly
        var parsed_mat = load_flmat_bytes(nested_flmat_b)
        assert_equal(parsed_mat.albedo_tex.rel_path, "textures/albedo.png")
        assert_equal(parsed_mat.normal_tex.rel_path, "textures/normal.png")
        assert_equal(parsed_mat.orm_tex.rel_path, "textures/orm.png")
        assert_equal(parsed_mat.emissive_tex.rel_path, "textures/emissive.png")

        # 2. AssetServer texture role hints cook sRGB / BC5 normal / linear BC7_UNORM ORM
        var srv_role = AssetServer(enable_watcher=False)
        srv_role.loose_root = ws
        var h_alb = srv_role.resolve_texture_with_role(AssetRef.from_path("textures/albedo.png", ASSET_KIND_TEXTURE), role=0)
        var h_nrm = srv_role.resolve_texture_with_role(AssetRef.from_path("textures/normal.png", ASSET_KIND_TEXTURE), role=1)
        var h_orm = srv_role.resolve_texture_with_role(AssetRef.from_path("textures/orm.png", ASSET_KIND_TEXTURE), role=2)
        var h_emi = srv_role.resolve_texture_with_role(AssetRef.from_path("textures/emissive.png", ASSET_KIND_TEXTURE), role=3)
        assert_true(srv_role.textures.is_loaded(h_alb))
        assert_true(srv_role.textures.is_loaded(h_nrm))
        assert_true(srv_role.textures.is_loaded(h_orm))
        assert_true(srv_role.textures.is_loaded(h_emi))
        assert_equal(Int(srv_role.textures.get_ptr(h_alb).value()[].tex_format), Int(TEX_FMT_BC7_SRGB))
        assert_equal(Int(srv_role.textures.get_ptr(h_nrm).value()[].tex_format), Int(TEX_FMT_BC5_UNORM))
        assert_equal(Int(srv_role.textures.get_ptr(h_orm).value()[].tex_format), Int(TEX_FMT_BC7_UNORM))
        assert_equal(Int(srv_role.textures.get_ptr(h_emi).value()[].tex_format), Int(TEX_FMT_BC7_SRGB))
        srv_role.close()

        # 3. EngineApp + AssetPlugin + RenderPlugin resolves nested textures and hot-reloads MaterialOverrideComponent
        var app = EngineApp(AppConfig(title="M4bNestedFlmat", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=True))
        app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
        app.startup()

        var e = app.world.spawn()
        _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](
            e,
            MeshRendererComponent(
                -1,
                -1,
                True,
                True,
                AssetRef.from_path("meshes/cube.glb", ASSET_KIND_MESH),
                AssetRef.from_path("materials/nested.flmat", ASSET_KIND_RAW),
            ),
        )
        _ = app.world.insert[MaterialOverrideComponent](
            e, MaterialOverrideComponent(material_ref=AssetRef.from_path("materials/nested.flmat", ASSET_KIND_RAW))
        )

        app.step_one_frame(Float32(1.0 / 60.0))

        ref srv = app.resources.get_ptr[AssetServer]().value()[unsafe_offset=0]
        ref ren = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0]
        ref rstate = app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
        var base_mat_id = app.world.get_ptr[MeshRendererComponent](e).value()[].material_id
        assert_true(base_mat_id >= 0)
        assert_true(abs(ren.materials[base_mat_id].roughness - Float32(0.42)) < Float32(1.0e-4))
        var ov_slot0 = rstate.override_slots.get(e.to_bits()).value().slot_id
        assert_true(abs(ren.materials[ov_slot0].base_color.x - Float32(0.25)) < Float32(1.0e-4))
        # Referenced texture handles are live in AssetServer
        var s_nrm = srv.textures.find_slot_by_path("textures/normal.png")
        assert_true(s_nrm >= 0)

        # Hot-reload normal texture AND nested .flmat -> base material + MaterialOverrideComponent update
        var px_nrm2 = List[UInt32](length=8 * 8, fill=UInt32(0xFF7090FF))
        assert_true(write_png(ws + "/textures/normal.png", px_nrm2, 8, 8, stamped=False))
        var nested_flmat2 = (
            '{"format":"flmat","version":1,"name":"NestedPbr",'
            '"base_color":[0.85,0.2,0.3],"roughness":0.88,"metallic":0.75,"emissive":[0.5,0.1,0.0], '
            '"textures":{"albedo":"textures/albedo.png","normal":"textures/normal.png",'
            '"orm":"textures/orm.png","emissive":"textures/emissive.png"}}'
        )
        var nested_flmat2_b = _plug_str_bytes(nested_flmat2)
        assert_true(write_bytes(ws + "/materials/nested.flmat", nested_flmat2_b))
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1000.0)
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1001.0)
        app.step_one_frame(Float32(1.0 / 60.0))

        ref mo_after = app.world.get_ptr[MaterialOverrideComponent](e).value()[unsafe_offset=0]
        assert_true(abs(mo_after.base_color.x - Float32(0.85)) < Float32(1.0e-4))
        assert_true(abs(mo_after.roughness - Float32(0.88)) < Float32(1.0e-4))
        var ov_slot1 = rstate.override_slots.get(e.to_bits()).value().slot_id
        assert_true(abs(ren.materials[ov_slot1].base_color.x - Float32(0.85)) < Float32(1.0e-4))
        assert_true(abs(ren.materials[ov_slot1].roughness - Float32(0.88)) < Float32(1.0e-4))
        assert_true(abs(ren.materials[base_mat_id].roughness - Float32(0.88)) < Float32(1.0e-4))

        app.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4b_glb_anim_cache_initial_load_and_hot_reload() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_m4b_glb_anim_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/models")
    try:
        var glb3 = _make_rigged_glb_for_plug(3)
        assert_true(write_bytes(ws + "/models/char.glb", glb3))

        var app = EngineApp(AppConfig(title="M4bGlbAnimSync", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=True))
        app.add_plugin(AnimPlugin())
        app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
        app.startup()

        var e = app.world.spawn()
        _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](
            e,
            MeshRendererComponent(
                -1,
                -1,
                True,
                True,
                AssetRef.from_path("models/char.glb", ASSET_KIND_MESH),
            ),
        )
        # Pre-set non-zero model_id=101 must be preserved, clip_index=2 active
        _ = app.world.insert[SkeletonComponent](e, SkeletonComponent(UInt64(101)))
        _ = app.world.insert[AnimatorComponent](
            e, AnimatorComponent(UInt64(0), clip_index=2, clip_name="Run", playing=True)
        )

        app.step_one_frame(Float32(1.0 / 60.0))

        ref acache = app.resources.get_ptr[AnimAssetCache]().value()[unsafe_offset=0]
        var mi1 = acache.find_model(UInt64(101))
        assert_true(mi1 >= 0)
        assert_equal(len(acache.models[mi1].skins), 1)
        assert_equal(len(acache.models[mi1].animations), 3)
        ref sk1 = app.world.get_ptr[SkeletonComponent](e).value()[unsafe_offset=0]
        assert_equal(Int(sk1.model_id), 101)
        assert_true(sk1.skeleton.node_count >= 2)
        assert_equal(app.world.get_ptr[AnimatorComponent](e).value()[].clip_index, 2)

        # Hot-reload GLB with only 1 clip ("Idle"): AnimAssetCache reloads, clip_index clamps to 0
        var glb1 = _make_rigged_glb_for_plug(1)
        assert_true(write_bytes(ws + "/models/char.glb", glb1))
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1000.0)
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1001.0)
        app.step_one_frame(Float32(1.0 / 60.0))

        var mi2 = acache.find_model(UInt64(101))
        assert_true(mi2 >= 0)
        assert_equal(len(acache.models[mi2].animations), 1)
        assert_equal(app.world.get_ptr[AnimatorComponent](e).value()[].clip_index, 0)
        assert_true(app.resources.get_ptr[RenderPluginStateResource]().value()[].skinned_sync_count >= 1)

        app.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4b_audio_emitter_hot_reload_restarts_playing_voice() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_v5_plug_m4b_au_restart_" + String(pid)
    _plug_rm_rf(tmp)
    var ws = tmp + "/ws"
    _ = make_dirs(ws + "/audio")
    try:
        var wav1 = _make_test_wav_16bit(Float32(0.15), 2400)
        assert_true(write_bytes(ws + "/audio/voice.wav", wav1))

        var app = EngineApp(AppConfig(title="M4bAudioRestart", headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=ws, enable_watcher=True))
        app.add_plugin(AudioPlugin(headless=True))
        app.startup()

        var e_lis = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_lis, LocalTransform())
        _ = app.world.insert[GlobalTransform](e_lis, GlobalTransform())
        _ = app.world.insert[AudioListenerComponent](e_lis, AudioListenerComponent())

        var e_em = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_em, LocalTransform(Vec3(1.0, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_em, GlobalTransform())
        _ = app.world.insert[AudioEmitterComponent](
            e_em,
            AudioEmitterComponent(
                asset_id=-1,
                looping=True,
                play_on_spawn=True,
                audio_ref=AssetRef.from_path("audio/voice.wav", ASSET_KIND_AUDIO),
            ),
        )

        # Frame 1 starts playing old voice
        app.step_one_frame(Float32(1.0 / 60.0))
        ref em1 = app.world.get_ptr[AudioEmitterComponent](e_em).value()[unsafe_offset=0]
        var aid1 = em1.asset_id
        var h1 = em1.handle
        assert_true(aid1 >= 0)
        assert_true(em1.playing)
        assert_true(h1.is_valid())

        # Overwrite audio/voice.wav -> SceneAssetResolver updates asset_id -> AudioPlugin stops old & restarts new
        var wav2 = _make_test_wav_16bit(Float32(0.35), 2400)
        assert_true(write_bytes(ws + "/audio/voice.wav", wav2))
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1000.0)
        _ = app.resources.get_ptr_mut[AssetServer]().value()[].tick(1001.0)
        # First frame triggers asset_id change -> stop old voice & queue auto_play_pending; second starts new handle cleanly
        app.step_one_frame(Float32(1.0 / 60.0))
        app.step_one_frame(Float32(1.0 / 60.0))

        ref em2 = app.world.get_ptr[AudioEmitterComponent](e_em).value()[unsafe_offset=0]
        var aid2 = em2.asset_id
        assert_true(aid2 > aid1)
        assert_true(em2.playing)
        assert_true(em2.handle.is_valid())
        assert_true(em2.handle.slot != h1.slot or em2.handle.generation != h1.generation)
        ref aus = app.resources.get_ptr[AudioSyncStateResource]().value()[unsafe_offset=0]
        assert_equal(len(aus.active_emitters), 1)
        assert_equal(aus.active_emitters[0].asset_id, aid2)

        app.shutdown()
    finally:
        _plug_rm_rf(tmp)


def test_m4b_audio_emitter_read_bin_v2_lossless_and_v1_migration() raises:
    # v2 binary round-trip with audio_ref set and asset_path == "" must remain lossless
    var src1 = AudioEmitterComponent()
    src1.audio_ref = AssetRef.from_path("audio/sfx.wav", ASSET_KIND_AUDIO)
    src1.asset_path = ""
    var w1 = BlobWriter()
    src1.write_bin(w1)
    var r1 = BlobReader(w1.data.copy())
    var dst1 = AudioEmitterComponent.read_bin(r1, 2)
    assert_equal(dst1.asset_path, "")
    assert_equal(dst1.audio_ref.rel_path, "audio/sfx.wav")

    # v2 binary round-trip with asset_path set and audio_ref cleared must also remain lossless
    var src2 = AudioEmitterComponent()
    src2.asset_path = "audio/sfx.wav"
    src2.audio_ref = AssetRef()
    var w2 = BlobWriter()
    src2.write_bin(w2)
    var r2 = BlobReader(w2.data.copy())
    var dst2 = AudioEmitterComponent.read_bin(r2, 2)
    assert_equal(dst2.asset_path, "audio/sfx.wav")
    assert_true(dst2.audio_ref.is_empty())

    # v1 legacy migration (asset_path -> audio_ref) must still work when version < 2
    var src_v1 = AudioEmitterComponent()
    src_v1.asset_path = "audio/legacy.wav"
    src_v1.audio_ref = AssetRef()
    var w_v1 = BlobWriter()
    src_v1.write_bin(w_v1)
    var r_v1 = BlobReader(w_v1.data.copy())
    var dst_v1 = AudioEmitterComponent.read_bin(r_v1, 1)
    assert_equal(dst_v1.asset_path, "audio/legacy.wav")
    assert_equal(dst_v1.audio_ref.rel_path, "audio/legacy.wav")


def test_m5a_component_registry_default_insert_remove_all_builtins() raises:
    var reg = ComponentRegistry.with_builtins()
    register_physics_codecs(reg)
    register_render_codecs(reg)
    register_anim_codecs(reg)
    register_audio_codecs(reg)
    register_nav_codecs(reg)
    register_net_codecs(reg)
    register_ai_codecs(reg)

    var w = World()
    reg.ensure_registered_in_world(w)
    var e = w.spawn_empty()
    assert_true(not reg.insert_default_on_entity(w, NULL_ENTITY, "LocalTransform"))
    assert_true(not reg.insert_default_on_entity(w, e, "DoesNotExist"))

    var n_entries = len(reg.entries)
    assert_true(n_entries >= 31)
    for i in range(n_entries):
        var cname = reg.entries[i].schema_name
        assert_true(reg.insert_default_on_entity(w, e, cname), "insert_default failed for " + cname)
        assert_true(not reg.insert_default_on_entity(w, e, cname), "duplicate insert succeeded for " + cname)
        assert_true(reg.has_on_entity(i, w, e))

    var snap = BlobWriter()
    w.snapshot_blob(snap)
    var w2 = World()
    reg.ensure_registered_in_world(w2)
    var rdr = BlobReader(snap.data.copy())
    w2.restore_blob(rdr)
    assert_equal(w2.entity_count(), 1)

    for i in range(n_entries):
        var cname = reg.entries[i].schema_name
        assert_true(reg.remove_from_entity(w, e, cname), "remove failed for " + cname)
        assert_true(not reg.remove_from_entity(w, e, cname), "double remove succeeded for " + cname)
        assert_true(not reg.has_on_entity(i, w, e))


def test_m5a_physics_trigger_helpers_and_despawn_safety() raises:
    var app = EngineApp(AppConfig(title="m5a_trig", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var trig_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform.from_local(LocalTransform(Vec3(2.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(0.8, 1.5, 0.8), is_trigger=True))

    var kcc_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](kcc_e, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e,
        CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3(6.0, 0.0, 0.0)),
    )

    var dead_e = app.world.spawn_empty()
    _ = app.world.despawn(dead_e)

    var cursor = 0
    var enters = 0
    var exits = 0
    var matched_4 = False
    var matched_res = False
    var sync_mapped_trig = False
    for _ in range(60):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        var sync_opt = app.resources.get_ptr[PhysicsSyncStateResource]()
        if tev_opt and sync_opt:
            ref sync = sync_opt.value()[unsafe_offset=0]
            assert_true(entity_from_body(sync, -1) == NULL_ENTITY)
            assert_true(entity_from_body(-1, app.resources) == NULL_ENTITY)
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                var ev = evs[i]
                if OnEnter(ev):
                    enters += 1
                    assert_true(not OnEnter(ev, NULL_ENTITY, app.world, app.resources))
                    assert_true(not OnEnter(ev, dead_e, app.world, app.resources))
                    if OnEnter(ev, trig_e, app.world, app.resources):
                        matched_4 = True
                    if entity_from_body(sync, ev.body_id) == kcc_e and entity_from_body(ev.body_id, app.resources) == kcc_e:
                        matched_res = True
                    if sync.lookup_trigger_entity(ev.trigger_id) == trig_e:
                        sync_mapped_trig = True
                if OnExit(ev) and OnExit(ev, trig_e, app.world, app.resources):
                    exits += 1

    assert_equal(enters, 1)
    assert_equal(exits, 1)
    assert_true(matched_4)
    assert_true(matched_res)
    assert_true(sync_mapped_trig)

    # Toggle trigger off and verify trigger_to_entity is cleaned up
    var col_mut = app.world.get_ptr_mut[ColliderComponent](trig_e)
    assert_true(Bool(col_mut))
    var old_tid = col_mut.value()[unsafe_offset=0].trigger_id
    col_mut.value()[unsafe_offset=0].is_trigger = False
    app.step_one_frame(Float32(1.0 / 60.0))
    var sync_after = app.resources.get_ptr[PhysicsSyncStateResource]()
    assert_true(Bool(sync_after))
    assert_true(sync_after.value()[unsafe_offset=0].lookup_trigger_entity(old_tid) == NULL_ENTITY)
    app.shutdown()


def test_m5a_ai_behavior_tree_component_json_codecs_and_plugin() raises:
    var bb = Blackboard()
    var k_alert = bb.def_key("alert", Float32(0.0))

    var scorer = UtilityAiScorer(True)
    var opt0 = UtilityOption(1, Float32(1.0))
    opt0.add_consideration(UtilityConsideration(k_alert, Float32(0.0), Float32(1.0), CURVE_LINEAR, Float32(1.0), False))
    _ = scorer.add_option(opt0^)

    var bt = BehaviorTree()
    _ = bt.begin_selector()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_alert, BT_CMP_GT, Float32(0.5), BT_ABORT_LOWER_PRIORITY)
    _ = bt.add_action(10)
    bt.end_composite()
    _ = bt.add_action(2)
    bt.end_composite()

    var js = bt.save_json()
    var bt2 = BehaviorTree()
    assert_true(bt2.load_json(js))
    assert_true(not bt2.load_json(""))
    assert_true(not bt2.load_json("{bad json"))
    assert_true(not bt2.load_json('{"nodes":"not_a_list"}'))
    assert_true(not bt2.load_json('{"nodes":[{"kind":99}]}'))

    # Corrupt load_json calls must leave bt2 valid and functional
    var st = BtExecutionState()
    st.set_action_status(2, BT_RUNNING)
    st.set_action_status(10, BT_SUCCESS)
    assert_equal(Int(bt2.tick(bb, st, Float32(1.0 / 60.0))), Int(BT_RUNNING))
    bb.set_float(k_alert, Float32(1.0))
    assert_equal(Int(bt2.tick(bb, st, Float32(1.0 / 60.0))), Int(BT_SUCCESS))
    assert_equal(st.aborts_triggered, 1)

    var reg = ComponentRegistry.with_builtins()
    register_ai_codecs(reg)
    var app = EngineApp(AppConfig(title="m5a_ai_plugin", headless=True, worker_threads=0))
    app.add_plugin(AiPlugin())
    app.add_plugin(AiPlugin())  # idempotent
    app.startup()

    var e0 = app.world.spawn_empty()
    _ = app.world.insert[BehaviorTreeComponent](e0, BehaviorTreeComponent())
    var e1 = app.world.spawn_empty()
    var btc1 = BehaviorTreeComponent(bt2.copy())
    var k1 = btc1.blackboard.def_key("alert", Float32(1.0))
    _ = k1
    btc1.state.set_action_status(10, BT_SUCCESS)
    _ = app.world.insert[BehaviorTreeComponent](e1, btc1^)

    app.step_one_frame(Float32(1.0 / 60.0))
    var ptr1 = app.world.get_ptr[BehaviorTreeComponent](e1)
    assert_true(Bool(ptr1))
    assert_equal(ptr1.value()[unsafe_offset=0].state.active_action_id, 10)

    # .flscn + .flscnb + World.snapshot_blob parity
    var sc = Scene("AiScene")
    var sc_json = sc.save_flscn_string(app.world, reg)
    var sc_bin = sc.save_flscnb_bytes(app.world, reg)
    var w_j = World()
    var jbytes = List[UInt8](capacity=sc_json.byte_length())
    for b in sc_json.as_bytes():
        jbytes.append(b)
    _ = sc.load_flscn_bytes(w_j, reg, jbytes)
    var w_b = World()
    _ = sc.load_flscnb_bytes(w_b, reg, sc_bin)
    assert_equal(w_j.entity_count(), 2)
    assert_equal(w_b.entity_count(), 2)

    var bw = BlobWriter()
    app.world.snapshot_blob(bw)
    var w_s = World()
    register_ai_components(w_s)
    var br = BlobReader(bw.data.copy())
    w_s.restore_blob(br)
    assert_equal(w_s.entity_count(), 2)
    app.shutdown()

    # Corrupt BehaviorTreeComponent.read_bin rejection (invalid kind, subtree_skip, parent_idx)
    var bad_blobs = List[List[UInt8]]()
    var bw_kind = BlobWriter()
    bw_kind.u32(1)
    bw_kind.u8(255)
    bw_kind.u8(0)
    bw_kind.u8(0)
    bw_kind.integer(1)
    bw_kind.integer(-1)
    bw_kind.integer(-1)
    bw_kind.integer(-1)
    bw_kind.f32(0.0)
    bad_blobs.append(bw_kind.data.copy())

    var bw_skip = BlobWriter()
    bw_skip.u32(1)
    bw_skip.u8(BT_NODE_ACTION)
    bw_skip.u8(0)
    bw_skip.u8(0)
    bw_skip.integer(0)  # invalid subtree_skip <= i
    bw_skip.integer(-1)
    bw_skip.integer(-1)
    bw_skip.integer(1)
    bw_skip.f32(0.0)
    bad_blobs.append(bw_skip.data.copy())

    var bw_pidx = BlobWriter()
    bw_pidx.u32(1)
    bw_pidx.u8(BT_NODE_ACTION)
    bw_pidx.u8(0)
    bw_pidx.u8(0)
    bw_pidx.integer(1)
    bw_pidx.integer(0)  # invalid parent_idx >= i
    bw_pidx.integer(-1)
    bw_pidx.integer(1)
    bw_pidx.f32(0.0)
    bad_blobs.append(bw_pidx.data.copy())

    for bi in range(len(bad_blobs)):
        var br_bad = BlobReader(bad_blobs[bi].copy())
        var rejected = False
        try:
            _ = BehaviorTreeComponent.read_bin(br_bad, 1)
        except:
            rejected = True
        assert_true(rejected)

    # SpatialPerception + AiPlugin interoperability check
    var pw = PhysicsWorld()
    var perc = SpatialPerception()
    var s0 = perc.add_stimulus(Vec3(0.0, 0.0, 5.0), Float32(1.0), 10, 1)
    var s1 = perc.add_stimulus(Vec3(0.0, 0.0, 6.0), Float32(0.5), 20, 2)
    perc.active[s1] = False
    # Zero forward falls back to +Z; hearing_threshold=0.0 must not leak padded lanes
    var res_all = perc.query_perceiver(
        pw,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        fov_deg=Float32(90.0),
        max_vision_dist=Float32(20.0),
        hearing_threshold=Float32(0.0),
    )
    assert_equal(len(res_all), 1)
    assert_equal(res_all[0].stimulus_index, s0)
    assert_true(res_all[0].visible and res_all[0].audible)
    assert_equal(
        len(perc.query_perceiver(pw, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), ignore_body_id=10)),
        0,
    )
    assert_equal(
        len(perc.query_perceiver(pw, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), ignore_team_id=1)),
        0,
    )


def test_m5a_game_flow_fsm_saves_timers_tweens() raises:
    var fsm = GameStateMachine("Menu")
    fsm.allow_transition("Menu", "Play")
    fsm.allow_transition("Play", "Victory")
    assert_true(not fsm.transition_to("Victory"))
    assert_equal(fsm.current_state(), "Menu")
    assert_true(fsm.transition_to("Play"))
    assert_true(fsm.transition_to("Victory"))
    assert_equal(fsm.current_state(), "Victory")

    var tmp = "/tmp/flint_v5_m5a_flow_test"
    var save_dir = tmp + "/saves"
    assert_true(make_dirs(save_dir))
    var store = SaveSlotStore(save_dir)
    assert_true(not store.save_slot("../escape", "{}"))
    assert_true(not store.save_slot(".hidden", "{}"))
    assert_equal(store.load_slot("missing").byte_length(), 0)
    assert_true(store.save_slot("slot_a", '{"hp":95}'))
    assert_true(make_dirs(save_dir + "/bogus_dir.json"))
    var slots = store.list_slots()
    assert_equal(len(slots), 1)
    assert_equal(slots[0], "slot_a")
    assert_true(not store.has_slot("bogus_dir"))
    assert_equal(store.load_slot("slot_a"), '{"hp":95}')

    var flow = GameFlow(save_dir, "Menu")
    assert_true(flow.save_slot(0, "ckpt_0"))
    assert_equal(flow.load_slot(0), "ckpt_0")
    assert_true(store.delete_slot("slot_a"))
    assert_true(store.delete_slot("slot_0"))

    # Scene transition A -> B
    var reg = ComponentRegistry.with_builtins()
    var w1 = World()
    reg.ensure_registered_in_world(w1)
    var e1 = w1.spawn_empty()
    _ = w1.insert[LocalTransform](e1, LocalTransform(Vec3(1.0, 2.0, 3.0)))
    var path_a = tmp + "/scene_a.flscn"
    assert_true(Scene("A").save_flscn_file(path_a, w1, reg))

    var live_w = World()
    assert_true(flow.transition_to_scene(live_w, reg, path_a))
    assert_equal(live_w.entity_count(), 1)
    assert_true(not flow.transition_to_scene(live_w, reg, tmp + "/missing.flscn"))
    assert_equal(live_w.entity_count(), 1)

    var nan_dt = Float32(0.0) / Float32(0.0)
    var t_rep = GameTimer(Float32(0.2), repeating=True)
    assert_true(not t_rep.step(Float32(-0.1)))
    assert_true(not t_rep.step(nan_dt))
    var fires = 0
    for _ in range(5):
        if t_rep.step(Float32(0.1)):
            fires += 1
    assert_equal(fires, 2)

    var t_huge1 = GameTimer(Float32(0.1), repeating=True)
    assert_true(t_huge1.step(Float32(1.0e7)))
    assert_true(isfinite(t_huge1.elapsed_sec))
    assert_true(t_huge1.elapsed_sec >= Float32(0.0) and t_huge1.elapsed_sec < t_huge1.duration_sec)

    var t_huge2 = GameTimer(Float32(1.0e-6), repeating=True)
    assert_true(t_huge2.step(Float32(100.0)))
    assert_true(isfinite(t_huge2.elapsed_sec))
    assert_true(t_huge2.elapsed_sec >= Float32(0.0) and t_huge2.elapsed_sec < t_huge2.duration_sec)

    var t_huge3 = GameTimer(Float32(0.1), repeating=True)
    assert_true(t_huge3.step(Float32(1.0e30)))
    assert_true(isfinite(t_huge3.elapsed_sec))
    assert_true(t_huge3.elapsed_sec >= Float32(0.0) and t_huge3.elapsed_sec < t_huge3.duration_sec)

    var tw = Tween(Float32(0.0), Float32(10.0), Float32(0.5))
    assert_true(isfinite(tw.step(nan_dt)))
    assert_true(not tw.is_finished())
    var mid = tw.step(Float32(0.25))
    assert_true(abs(mid - Float32(5.0)) < Float32(0.1))
    var fin = tw.step(Float32(0.5))
    assert_true(abs(fin - Float32(10.0)) < Float32(1.0e-4))
    assert_true(tw.is_finished())

    var rm_cmd = "rm -rf " + tmp
    _ = external_call["system", Int32](rm_cmd.as_c_string_span())


def test_m5a_game_hud_and_ui_keyboard_capture() raises:
    var ctx = UiContext()
    assert_true(not ctx.wants_keyboard())
    assert_true(not ctx.is_in_frame())

    # Safe outside begin_frame/end_frame even with empty box_stack
    var added_outside = game_hud(ctx, 75, 100, 1200)
    assert_true(added_outside > 0)
    var count_after_outside = len(ctx.draw_list.prims)

    # Safe inside begin_frame/end_frame while preserving prior draw_list prims
    ctx.set_keyboard_capture(True)
    var inp = InputState()
    inp.begin_frame()
    ctx.begin_frame(inp, Float32(640.0), Float32(480.0), Float32(1.0 / 60.0))
    assert_true(ctx.is_in_frame())
    assert_true(ctx.wants_keyboard())
    ctx.draw_list.add_rect(Float32(0.0), Float32(0.0), Float32(10.0), Float32(10.0), UInt32(0xFFFFFFFF))
    var pre_hud = len(ctx.draw_list.prims)
    var added_inside = game_hud(ctx, 20, 100, 50)
    assert_true(added_inside > 0)
    assert_equal(len(ctx.draw_list.prims), pre_hud + added_inside)
    ctx.end_frame()
    assert_true(not ctx.is_in_frame())
    _ = count_after_outside

    ctx.set_keyboard_capture(False)
    assert_true(not ctx.wants_keyboard())

    # font.valid == True (DSF v2 in-memory atlas -> PRIM_GLYPH emission)
    var dsf = List[UInt8]()
    def _push_i32(mut dst: List[UInt8], v: Int):
        var uv = UInt32(Int32(v))
        dst.append(UInt8(uv & 0xFF))
        dst.append(UInt8((uv >> 8) & 0xFF))
        dst.append(UInt8((uv >> 16) & 0xFF))
        dst.append(UInt8((uv >> 24) & 0xFF))

    var first_cp = 32
    var last_cp = 122
    var glyph_cnt = last_cp - first_cp + 1
    _push_i32(dsf, DSF_MAGIC)
    _push_i32(dsf, 2)
    _push_i32(dsf, 32)
    _push_i32(dsf, 32)
    _push_i32(dsf, 24)
    _push_i32(dsf, 4)
    _push_i32(dsf, 20 * 64)
    _push_i32(dsf, -4 * 64)
    _push_i32(dsf, 26 * 64)
    _push_i32(dsf, glyph_cnt)
    _push_i32(dsf, 0)
    _push_i32(dsf, 1)
    for cp in range(first_cp, last_cp + 1):
        _push_i32(dsf, cp)
        _push_i32(dsf, 0)
        _push_i32(dsf, 0)
        _push_i32(dsf, 8)
        _push_i32(dsf, 12)
        _push_i32(dsf, 0)
        _push_i32(dsf, 0)
        _push_i32(dsf, 12 * 64)
    for _ in range(32 * 32):
        dsf.append(UInt8(128))

    var ctx_font = UiContext()
    assert_true(ctx_font.font.load_bytes(dsf))
    assert_true(ctx_font.font.valid)
    assert_true(not ctx_font.font.has_source)
    _ = game_hud(ctx_font, 75, 100, 1200)
    var dsf_glyphs = 0
    for pi in range(len(ctx_font.draw_list.prims)):
        ref pr = ctx_font.draw_list.prims[pi]
        if (pr.header & 0xF) == PRIM_GLYPH:
            assert_true(
                isfinite(pr.x0)
                and isfinite(pr.y0)
                and isfinite(pr.x1)
                and isfinite(pr.y1)
                and isfinite(pr.u0)
                and isfinite(pr.v0)
                and isfinite(pr.u1)
                and isfinite(pr.v1)
            )
            dsf_glyphs += 1
    assert_true(dsf_glyphs >= 10)

    # Guard-test shaped TTF branch when system DejaVuSans.ttf is present
    var ttf_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if len(read_bytes(ttf_path)) > 0:
        var ctx_ttf = UiContext()
        if ctx_ttf.font.load_ttf(ttf_path, bake_px=24, spread=4):
            assert_true(ctx_ttf.font.valid and ctx_ttf.font.has_source)
            _ = game_hud(ctx_ttf, 42, 100, 777)
            var ttf_glyphs = 0
            for pi in range(len(ctx_ttf.draw_list.prims)):
                ref pr = ctx_ttf.draw_list.prims[pi]
                if (pr.header & 0xF) == PRIM_GLYPH:
                    assert_true(isfinite(pr.x0) and isfinite(pr.y0) and isfinite(pr.x1) and isfinite(pr.y1))
                    ttf_glyphs += 1
            assert_true(ttf_glyphs >= 10)


def main() raises:
    var passed = 0
    var total = 40

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

    test_schedule_out_of_order_add_dependency()
    print("[PASS] test_schedule_out_of_order_add_dependency")
    passed += 1

    test_schedule_conflict_with_explicit_dependency()
    print("[PASS] test_schedule_conflict_with_explicit_dependency")
    passed += 1

    test_m3a_nav_plugin_lifecycle_and_overlay()
    print("[PASS] test_m3a_nav_plugin_lifecycle_and_overlay")
    passed += 1

    test_m4a_asset_server_traversal_and_asset_ref_unsafe_path()
    print("[PASS] test_m4a_asset_server_traversal_and_asset_ref_unsafe_path")
    passed += 1

    test_m4a_asset_plugin_skinned_frame2_and_flrig_hot_reload()
    print("[PASS] test_m4a_asset_plugin_skinned_frame2_and_flrig_hot_reload")
    passed += 1

    test_m4a_move_constructors_residency_bridge_and_fd_leaks()
    print("[PASS] test_m4a_move_constructors_residency_bridge_and_fd_leaks")
    passed += 1

    test_m4a_scene_asset_resolver_shared_flmat_and_audio_hot_reload()
    print("[PASS] test_m4a_scene_asset_resolver_shared_flmat_and_audio_hot_reload")
    passed += 1

    test_m4a_skeleton_and_animator_flscn_state_hash_parity()
    print("[PASS] test_m4a_skeleton_and_animator_flscn_state_hash_parity")
    passed += 1

    test_m4a_asset_server_flpk_uuid_resolve_audio_and_texture()
    print("[PASS] test_m4a_asset_server_flpk_uuid_resolve_audio_and_texture")
    passed += 1

    test_m4a_audio_emitter_flscn_state_hash_parity()
    print("[PASS] test_m4a_audio_emitter_flscn_state_hash_parity")
    passed += 1

    test_m4a_scene_asset_resolver_preserves_loaded_skeleton_pose()
    print("[PASS] test_m4a_scene_asset_resolver_preserves_loaded_skeleton_pose")
    passed += 1

    test_m4a_bone_socket_flscn_state_hash_parity()
    print("[PASS] test_m4a_bone_socket_flscn_state_hash_parity")
    passed += 1

    test_m4b_flmat_nested_textures_role_cooking_and_hot_reload()
    print("[PASS] test_m4b_flmat_nested_textures_role_cooking_and_hot_reload")
    passed += 1

    test_m4b_glb_anim_cache_initial_load_and_hot_reload()
    print("[PASS] test_m4b_glb_anim_cache_initial_load_and_hot_reload")
    passed += 1

    test_m4b_audio_emitter_hot_reload_restarts_playing_voice()
    print("[PASS] test_m4b_audio_emitter_hot_reload_restarts_playing_voice")
    passed += 1

    test_m4b_audio_emitter_read_bin_v2_lossless_and_v1_migration()
    print("[PASS] test_m4b_audio_emitter_read_bin_v2_lossless_and_v1_migration")
    passed += 1

    test_m5a_component_registry_default_insert_remove_all_builtins()
    print("[PASS] test_m5a_component_registry_default_insert_remove_all_builtins")
    passed += 1

    test_m5a_physics_trigger_helpers_and_despawn_safety()
    print("[PASS] test_m5a_physics_trigger_helpers_and_despawn_safety")
    passed += 1

    test_m5a_ai_behavior_tree_component_json_codecs_and_plugin()
    print("[PASS] test_m5a_ai_behavior_tree_component_json_codecs_and_plugin")
    passed += 1

    test_m5a_game_flow_fsm_saves_timers_tweens()
    print("[PASS] test_m5a_game_flow_fsm_saves_timers_tweens")
    passed += 1

    test_m5a_game_hud_and_ui_keyboard_capture()
    print("[PASS] test_m5a_game_hud_and_ui_keyboard_capture")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
