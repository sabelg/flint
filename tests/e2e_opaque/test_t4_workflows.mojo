"""Tier 4 Opaque-Box E2E Suite: Real-World Multi-Subsystem Workflows (5 Mojo Scenarios).

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`,
zero `gated/` imports, and zero M6-legacy module imports).
All disk outputs live under an isolated `/tmp/flint-e2e-opaque-t4-<pid>` directory
cleaned on both normal and exception exit paths.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isinf, isnan, sin, sqrt
from std.testing import assert_equal, assert_false, assert_true

from flint.anim.flanim_ecs import (
    RAGDOLL_SM_LIMP,
    RagdollStateMachine,
)
from flint.anim.graph import AnimGraphDef, StateDef
from flint.anim.sync import ClipInfo
from flint.anim.plugin import (
    AnimAssetCache,
    AnimPlugin,
    AnimatorComponent,
    RagdollComponent,
    SkeletonComponent,
    register_anim_codecs,
)
from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    SUB_ID_BVH8_COLLIDER,
    AssetMeta,
    AssetUuid,
    FlpkV3MappedArchive,
    write_flmeta,
)
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.assets.hdr_image import decode_hdr_or_exr, encode_radiance_hdr
from flint.assets.image import decode_png
from flint.assets.jpeg import decode_jpeg, encode_jpeg_baseline
from flint.assets.json import JsonDocument, parse_json
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeMesh,
    NativeModel,
    NativeNode,
    NativeTexture,
)
from flint.assets.server import (
    ASSET_STATE_LOADING,
    ASSET_STATE_READY,
    AssetServer,
    RawBytesAsset,
)
from flint.audio import (
    AudioServer,
    BUS_UI,
)
from flint.audio.plugin import (
    AudioEmitterComponent,
    AudioListenerComponent,
    AudioPlugin,
    register_audio_codecs,
)
from flint.ecs import World
from flint.core.detmath import f32_bits, set_sim_float_mode
from flint.core.jobs import JobPool
from flint.core.lz4 import hash_list
from flint.core.math import Mat4, Quat, Vec3
from flint.core.transform import trs
from flint.devtools.mcp_v2 import McpServerV2
from flint.input import ACTION_AXIS2D, ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin
from flint.lab import EngineBenchSuite
from flint.net import (
    AeadKey,
    ClientPredictionBuffer,
    HitboxPose,
    IPV6_SAFE_MTU_BYTES,
    JITTER_TRIANGULAR,
    LagCompHistory,
    NetEndpoint,
    NetHub,
    PredictedState,
    PredictionStepper,
    ReplicatedEntity,
    RollbackSession,
    SnapshotClientReceiver,
    SnapshotInterpolationBuffer,
    SnapshotReplicator,
    VirtualEndpoint,
    VirtualNetSim,
    VisualSmoother,
)
from flint.net.rollback import PhysicsRollbackAdapter, PhysicsRollbackInput
from flint.physics.plugin import (
    ColliderComponent,
    PhysicsPlugin,
    PhysicsWorld,
    RigidBodyComponent,
    register_physics_codecs,
)
from flint.platform.input import InputState
from flint.platform.os import file_size, make_dirs, read_bytes, write_bytes
from flint.render.aaa_ibl import bake_hdr_image_ibl_cubemap_and_sh9
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import FLIP_PPD, flip_error_map, flip_mean
from flint.render.font import SdfFont
from flint.render.image_diff import compare_pixels
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderExtractor,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    Scene,
    TransformPlugin,
    set_parent,
)
from flint.studio import (
    COLLIDER_BOX,
    COLLIDER_SPHERE,
    FIELD_F32,
    FlintStudioState,
    InspectedField,
    VisBufferGpuPicker,
    decode_visbuffer_pick,
    encode_visbuffer_words,
)
from flint.text.line_break import wrap
from flint.text.rich_text import layout_rich, parse_rich, plain_text
from flint.text.sdf_bake import DSF_MAGIC, bake_font, bake_glyph_msdf, dsf_bytes
from flint.text.ttf import TrueTypeFont
from flint.text.utf8 import REPLACEMENT, decode_utf8
from flint.ui.context import (
    UI_KEY_A,
    UI_KEY_BACKSPACE,
    UI_KEY_C,
    UI_KEY_LEFT,
    UI_KEY_LEFT_CTRL,
    UiContext,
)
from flint.ui.draw_list import UiDrawList
from flint.ui.flexbox_dock import DockSpace, RectF


comptime TAU64: Float64 = 6.283185307179586
comptime SYSTEM_TTF_PATH: String = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
comptime SYSTEM_OTF_PATH: String = "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf"


def t4_rm_rf_tmp(path: String):
    if path.startswith("/tmp/flint-e2e-opaque-t4-"):
        var cmd = "rm -rf " + path
        _ = external_call["system", Int32](cmd.as_c_string_span())


def t4_str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def t4_parse_json_str(s: String) -> JsonDocument:
    var raw = t4_str_to_bytes(s)
    return parse_json(raw)


def t4_make_sine_pcm16(
    freq_hz: Float64,
    frames: Int,
    channels: Int = 1,
    amp: Float64 = 0.75,
    sample_rate: Int = 48000,
) -> List[Int16]:
    var out = List[Int16](capacity=frames * channels)
    for f in range(frames):
        var s = amp * sin(TAU64 * freq_hz * Float64(f) / Float64(sample_rate))
        var q = Int16(max(-32768, min(32767, Int(s * 32767.0))))
        for _ in range(channels):
            out.append(q)
    return out^


def t4_make_procedural_biped() -> NativeModel:
    var m = NativeModel()
    var names = [
        String("Pelvis"), String("Chest"), String("Head"),
        String("UpperArm.R"), String("Hand.R"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
    ]
    var parents = [-1, 0, 1, 1, 3, 0, 5, 6, 0, 8, 9]
    var offsets = [
        Vec3(0.0, 2.5, 0.0), Vec3(0.0, 0.42, 0.0), Vec3(0.0, 0.32, 0.0),
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
    anim.duration = 0.50
    var ch = NativeChannel()
    ch.node = 0
    ch.path = 0
    ch.times = [Float32(0.0), Float32(0.50)]
    ch.values = [
        Float32(0.0), Float32(2.5), Float32(0.0),
        Float32(0.0), Float32(2.5), Float32(0.0),
    ]
    anim.channels.append(ch^)
    m.animations.append(anim^)
    return m^


def t4_make_prop_glb_model() -> NativeModel:
    var model = NativeModel()
    var tex = NativeTexture()
    tex.width = 16
    tex.height = 16
    tex.pixels = List[UInt32](length=256, fill=0xFFE0C890)
    for y in range(16):
        for x in range(16):
            if ((x + y) & 3) == 0:
                tex.pixels[y * 16 + x] = 0xFF3A7BD5
    model.textures.append(tex^)

    var mb = MeshBuilder(
        base_color=0xFFE87038,
        roughness=0.28,
        metallic=0.65,
        texture=0,
    )
    mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.75, 0.75, 0.75))
    var prims = List[NativeMesh]()
    prims.append(mb^.finish())
    _ = append_node_meshes(model, "prop_box", prims)
    return model^


def t4_make_payload(tag: Int, size: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=size)
    for i in range(size):
        out.append(UInt8((tag * 31 + i * 17 + 7) & 0xFF))
    return out^


struct T4PredictionStepper(PredictionStepper, Movable):
    def __init__(out self):
        pass

    def step_prediction(
        mut self,
        mut state: PredictedState,
        input_ptr: Int,
        input_seq: Int,
        tick: Int,
        dt: Float32,
    ):
        _ = input_seq
        _ = tick
        ref cmd = Pointer[PhysicsRollbackInput, ImmutAnyOrigin](
            unsafe_from_address=input_ptr
        )[unsafe_offset=0]
        state.vel = Vec3(cmd.impulse_x, cmd.impulse_y, cmd.impulse_z)
        state.pos = state.pos + state.vel * dt


# ===========================================================================
# Scenario 1: Headless Game Loop
# ===========================================================================

def t4_s1_headless_game_loop() raises:
    comptime W: Int = 128
    comptime H: Int = 96
    var app = EngineApp(
        AppConfig(title="T4_GameLoop", headless=True, worker_threads=0, fixed_hz=60.0)
    )
    app.add_plugin(TransformPlugin())
    app.add_plugin(WindowPlugin(headless=True, width=W, height=H, title="T4_GameLoop"))
    app.add_plugin(InputPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True, sample_rate=48000, quantum_frames=256))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H))

    app.startup()

    var audio_ptr = app.resources.get_ptr_mut[AudioServer]()
    try:
        assert_true(Bool(audio_ptr))
        assert_equal(audio_ptr.value()[unsafe_offset=0].audio_backend_name(), "headless")
        assert_true(app.resources.has[PhysicsWorld]())
        assert_true(app.resources.has[ScalableRenderer]())
        assert_true(app.resources.has[AnimAssetCache]())

        # Register biped model & graph in AnimAssetCache
        var biped = t4_make_procedural_biped()
        var graph = AnimGraphDef()
        graph.name = "T4BipedGraph"
        var st = StateDef()
        st.name = "Idle"
        st.clip = 0
        graph.states.append(st^)
        var clips = List[ClipInfo]()
        clips.append(ClipInfo("Idle", 0.50, True, 1.0))
        var anim_cache = app.resources.get_ptr_mut[AnimAssetCache]()
        assert_true(Bool(anim_cache))
        anim_cache.value()[unsafe_offset=0].register_model(UInt64(1), biped.copy())
        anim_cache.value()[unsafe_offset=0].register_graph(UInt64(10), graph^, clips^)

        # Register material & beveled box mesh on ScalableRenderer
        var ren_ptr = app.resources.get_ptr_mut[ScalableRenderer]()
        assert_true(Bool(ren_ptr))
        ref renderer = ren_ptr.value()[unsafe_offset=0]
        var mat_id = renderer.register_material(
            MaterialAsset(base_color=Vec3(0.92, 0.35, 0.14), roughness=0.35, metallic=0.1)
        )
        var mb = MeshBuilder()
        mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.65, 0.65, 0.65))
        var mesh_id = renderer.register_mesh(mb^.finish(), mat_id, build_lods=False)

        # Register 440 Hz tone in AudioServer
        var pcm = t4_make_sine_pcm16(440.0, 24000, channels=1, amp=0.8, sample_rate=48000)
        var aid = audio_ptr.value()[unsafe_offset=0].register_pcm16_asset(pcm, 1, 48000)
        assert_true(aid >= 0)

        # Spawn static floor
        var floor_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0))))
        assert_true(app.world.insert[GlobalTransform](floor_e, GlobalTransform()))
        assert_true(app.world.insert[PrevGlobalTransform](floor_e, PrevGlobalTransform()))
        assert_true(app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body()))
        assert_true(app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(20.0, 0.5, 20.0))))

        # Spawn dynamic falling crate with MeshRendererComponent
        var crate_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](crate_e, LocalTransform(Vec3(0.0, 4.0, 0.0))))
        assert_true(app.world.insert[GlobalTransform](crate_e, GlobalTransform()))
        assert_true(app.world.insert[PrevGlobalTransform](crate_e, PrevGlobalTransform()))
        assert_true(app.world.insert[RigidBodyComponent](crate_e, RigidBodyComponent.dynamic(mass=2.0)))
        assert_true(app.world.insert[ColliderComponent](crate_e, ColliderComponent.box(Vec3(0.5, 0.5, 0.5))))
        assert_true(app.world.insert[MeshRendererComponent](crate_e, MeshRendererComponent(mesh_id, mat_id)))

        # Spawn character with SkeletonComponent + AnimatorComponent + RagdollComponent
        var char_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](char_e, LocalTransform(Vec3(-2.5, 0.0, 0.0))))
        assert_true(app.world.insert[GlobalTransform](char_e, GlobalTransform()))
        assert_true(app.world.insert[PrevGlobalTransform](char_e, PrevGlobalTransform()))
        assert_true(app.world.insert[SkeletonComponent](char_e, SkeletonComponent(UInt64(1))))
        assert_true(app.world.insert[AnimatorComponent](char_e, AnimatorComponent(UInt64(10))))

        var rsm = RagdollStateMachine()
        _ = rsm.ragdoll.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
        _ = rsm.ragdoll.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
        _ = rsm.ragdoll.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
        _ = rsm.ragdoll.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)
        var init_skel = SkeletonInstance()
        init_skel.reset_to_rest(biped)
        var dt = Float32(1.0 / 60.0)
        rsm.ragdoll.handoff_from_skeleton(init_skel, Placement(Vec3(-2.5, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.0), biped, dt)
        rsm.ragdoll.go_limp()
        rsm.ragdoll.apply_impulse_at_bone(1, Vec3(8.0, 0.0, 4.0), Vec3(-2.5, 2.9, 0.0))
        rsm.state = RAGDOLL_SM_LIMP
        var initial_pelvis_y = rsm.ragdoll.pelvis_world_position().y
        assert_true(app.world.insert[RagdollComponent](char_e, RagdollComponent(rsm^, auto_recover=False, pelvis_syncs_transform=True)))

        # Camera + Light + AudioListener + AudioEmitter
        var cam_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 2.5, 6.0))))
        assert_true(app.world.insert[GlobalTransform](cam_e, GlobalTransform()))
        assert_true(app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform()))
        assert_true(app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=55.0, active=True)))

        var lgt_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(3.0, 6.0, 4.0))))
        assert_true(app.world.insert[GlobalTransform](lgt_e, GlobalTransform()))
        assert_true(app.world.insert[PointLightComponent](lgt_e, PointLightComponent(Vec3(24.0, 22.0, 20.0), Float32(18.0), True)))

        var lis_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](lis_e, LocalTransform(Vec3(0.0, 1.5, 0.0))))
        assert_true(app.world.insert[GlobalTransform](lis_e, GlobalTransform()))
        assert_true(app.world.insert[AudioListenerComponent](lis_e, AudioListenerComponent(active=True, priority=1)))

        var emt_e = app.world.spawn()
        assert_true(app.world.insert[LocalTransform](emt_e, LocalTransform(Vec3(0.0, 1.5, -1.0))))
        assert_true(app.world.insert[GlobalTransform](emt_e, GlobalTransform()))
        assert_true(
            app.world.insert[AudioEmitterComponent](
                emt_e,
                AudioEmitterComponent(asset_id=aid, volume=0.9, looping=True, spatial=True, play_on_spawn=True),
            )
        )

        # Scene codecs & .flscnb round-trip + PIE capture
        var reg = ComponentRegistry.with_builtins()
        register_physics_codecs(reg)
        register_anim_codecs(reg)
        register_render_codecs(reg)
        register_audio_codecs(reg)
        var scene = Scene("T4GameLoop")
        var flscnb = scene.save_flscnb_bytes(app.world, reg)
        assert_true(len(flscnb) > 32)
        var roundtrip_world = World()
        reg.ensure_registered_in_world(roundtrip_world)
        var rt_ents = scene.load_flscnb_bytes(roundtrip_world, reg, flscnb)
        assert_equal(len(rt_ents), 7)

        scene.enter_pie(app.world)

        var phys_ptr = app.resources.get_ptr[PhysicsWorld]()
        assert_true(Bool(phys_ptr))
        var start_tick = phys_ptr.value()[unsafe_offset=0].tick

        # Step 30 frames of full EngineApp schedule
        for _ in range(30):
            app.step_one_frame(dt)

        # Physics verification
        var end_tick = phys_ptr.value()[unsafe_offset=0].tick
        assert_equal(end_tick, start_tick + 30)
        var crate_lt = app.world.get[LocalTransform](crate_e)
        var crate_rb = app.world.get[RigidBodyComponent](crate_e)
        assert_true(crate_rb.body_id >= 0)
        assert_true(crate_lt.translation.y < Float32(3.5))

        # Ragdoll verification inside PhysicsWorld
        var rag_ptr = app.world.get_ptr[RagdollComponent](char_e)
        assert_true(Bool(rag_ptr))
        assert_true(rag_ptr.value()[unsafe_offset=0].state_machine.world_ragdoll_slot >= 0)
        var end_pelvis_y = rag_ptr.value()[unsafe_offset=0].state_machine.ragdoll.pelvis_world_position().y
        assert_true(end_pelvis_y < initial_pelvis_y - Float32(0.1))

        # Render + FLIP/ImageDiff verification against a same-camera empty RenderWorld
        var rstate_ptr = app.resources.get_ptr[RenderPluginStateResource]()
        assert_true(Bool(rstate_ptr))
        assert_true(rstate_ptr.value()[unsafe_offset=0].rendered_frames >= 30)
        assert_true(rstate_ptr.value()[unsafe_offset=0].last_render_ok)
        var rw_ptr = app.resources.get_ptr[RenderWorld]()
        assert_true(Bool(rw_ptr))
        assert_true(len(rw_ptr.value()[unsafe_offset=0].instances) >= 1)

        var empty_ui = UiDrawList()
        var blank_rw = RenderWorld()
        blank_rw.camera = rw_ptr.value()[unsafe_offset=0].camera
        assert_true(renderer.render_world(blank_rw, empty_ui, capture_readback=True, force_readback=True))
        var blank_px = renderer.readback_pixels()
        assert_equal(len(blank_px), W * H)

        assert_true(
            renderer.render_world(
                rw_ptr.value()[unsafe_offset=0],
                empty_ui,
                capture_readback=True,
                force_readback=True,
            )
        )
        var px = renderer.readback_pixels()
        assert_equal(len(px), W * H)
        var diff = compare_pixels(px, blank_px, W, H, tolerance=4)
        assert_true(diff.over_tolerance >= 400)
        assert_true(diff.worst >= 24)
        assert_true(diff.flip_mean > 0.01)

        # Audio active emitter check prior to shutdown
        assert_true(audio_ptr.value()[unsafe_offset=0].active_logical_count() >= 1)

        # Restore PIE snapshot & verify bit-exact crate reset
        scene.exit_pie(app.world)
        var restored_crate_lt = app.world.get[LocalTransform](crate_e)
        assert_equal(f32_bits(restored_crate_lt.translation.y), f32_bits(Float32(4.0)))
    except e:
        app.shutdown()
        raise e^

    # Shutdown EngineApp -> RT audio thread stops, then offline quantum renders active voice deterministically
    app.shutdown()
    assert_equal(audio_ptr.value()[unsafe_offset=0].audio_backend_name(), "none")
    var out_lr = List[Float32](length=512, fill=0.0)
    var peak = Float32(0.0)
    for _ in range(12):
        audio_ptr.value()[unsafe_offset=0].render_offline_quantum(out_lr, 256)
        for s in out_lr:
            assert_false(isnan(s) or isinf(s))
            var a = abs(s)
            if a > peak:
                peak = a
    assert_true(peak > Float32(0.01))
    assert_true(audio_ptr.value()[unsafe_offset=0].momentary_lufs() > Float32(-40.0))
    assert_true(audio_ptr.value()[unsafe_offset=0].integrated_lufs() > Float32(-40.0))


# ===========================================================================
# Scenario 2: Asset Cook-to-Render Pipeline
# ===========================================================================

def t4_s2_asset_cook_to_render_pipeline(tmp_root: String) raises:
    comptime W: Int = 128
    comptime H: Int = 96
    var s2_dir = tmp_root + "/s2"
    var ws = s2_dir + "/ws"
    var cache_dir = s2_dir + "/cache"
    var pack_path = s2_dir + "/bundle.flpk"
    assert_true(make_dirs(ws + "/meshes"))
    assert_true(make_dirs(ws + "/textures"))
    assert_true(make_dirs(ws + "/env"))

    # 1. Populate workspace with GLB + .flmeta, baseline JPEG, and Radiance HDR
    var glb_mdl = t4_make_prop_glb_model()
    var glb_path = ws + "/meshes/prop.glb"
    assert_true(write_native_glb(glb_path, glb_mdl))
    var meta = AssetMeta()
    meta.uuid = AssetUuid.from_path("meshes/prop.glb")
    meta.asset_kind = ASSET_KIND_MESH
    meta.build_lods = True
    meta.build_collider = True
    meta.source_path = "meshes/prop.glb"
    assert_true(write_flmeta(glb_path + ".flmeta", meta))

    var jpg_px = List[UInt32](length=32 * 32, fill=0)
    for y in range(32):
        for x in range(32):
            jpg_px[y * 32 + x] = (
                UInt32(0xFF000000)
                | (UInt32((x * 8) & 0xFF) << 16)
                | (UInt32((y * 8) & 0xFF) << 8)
                | UInt32(160)
            )
    var jpg_bytes = encode_jpeg_baseline(jpg_px, 32, 32, quality=92)
    assert_true(len(jpg_bytes) > 32)
    assert_true(write_bytes(ws + "/textures/albedo.jpg", jpg_bytes))

    var hdr_px = List[Vec3](length=16 * 8, fill=Vec3(2.2, 1.4, 0.8))
    var hdr_bytes = encode_radiance_hdr(hdr_px, 16, 8)
    assert_true(write_bytes(ws + "/env/sky.hdr", hdr_bytes))

    var pool = JobPool()
    _ = pool.start(2)
    var srv = AssetServer(
        stream_budget_bytes_per_tick=512,
        lru_max_bytes=65536,
        enable_watcher=False,
    )
    var ren = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H)

    try:
        # 2. First cook + incremental cache-hit second cook
        var rep1 = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep1.ok)
        assert_equal(rep1.total_assets, 3)
        assert_equal(rep1.cooked_count, 3)
        assert_equal(rep1.failed_count, 0)
        assert_true(rep1.archive_bytes > 64)

        var rep2 = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep2.ok)
        assert_equal(rep2.total_assets, 3)
        assert_equal(rep2.cache_hits, 3)
        assert_equal(rep2.cooked_count, 0)

        # 3. FlpkV3MappedArchive verification (TOC, CRC64, BVH8 sub-ID, zero-copy mmap slice)
        var arch = FlpkV3MappedArchive.open(pack_path)
        assert_true(arch.valid)
        assert_true(arch.toc_count() >= 4)
        var u_mesh = AssetUuid.from_path("meshes/prop.glb")
        var idx_mesh = arch.find_by_path("meshes/prop.glb", 0)
        assert_true(idx_mesh >= 0)
        assert_equal(idx_mesh, arch.find_by_uuid(u_mesh, 0))
        assert_true(arch.verify_entry_crc64(idx_mesh))
        assert_true(arch.find_by_uuid(u_mesh, SUB_ID_BVH8_COLLIDER) >= 0)
        var zc_slice = arch.mmap_slice_by_uuid(u_mesh, 0, verify_crc=True)
        assert_true(zc_slice.valid and zc_slice.addr != 0 and zc_slice.size > 0)
        arch.close()

        # 4. AssetServer mount, GPU residency bridge, async loads, progressive mips, LRU eviction & reload
        assert_true(srv.mount_flpk(pack_path))
        if ren.uses_vulkan():
            srv.attach_renderer_residency_bridge(ren.gpu_slot[0])

        var h_mesh = srv.load_mesh_async("meshes/prop.glb")
        var h_tex = srv.load_texture_async("textures/albedo.jpg", progressive=True)
        assert_equal(srv.mesh_state(h_mesh), ASSET_STATE_LOADING)
        assert_equal(srv.texture_state(h_tex), ASSET_STATE_LOADING)

        var done_async = srv.poll_async_loads(pool)
        assert_true(done_async >= 2)
        assert_equal(srv.mesh_state(h_mesh), ASSET_STATE_READY)
        assert_equal(srv.texture_state(h_tex), ASSET_STATE_READY)

        var mp = srv.meshes.borrow(h_mesh)
        assert_true(mp[unsafe_offset=0].has_collider)
        assert_true(mp[unsafe_offset=0].geom.total_vertices() > 0)
        assert_true(mp[unsafe_offset=0].geom.total_meshlets() > 0)

        for _ in range(12):
            if srv.textures.borrow(h_tex)[unsafe_offset=0].resident_min_mip == 0:
                break
            _ = srv.tick(0.0)
        assert_equal(srv.textures.borrow(h_tex)[unsafe_offset=0].resident_min_mip, 0)
        if ren.uses_vulkan():
            assert_true(srv.total_vram_live_bytes() > 0)

        var h_hdr = srv.load_hdr("env/sky.hdr")
        assert_true(srv.hdrs.is_loaded(h_hdr))

        # Pin >64 KiB raw asset, release mesh, enforce LRU eviction, and reload
        var pin_bytes = List[UInt8](length=70000, fill=0x42)
        var h_pin = srv.raws.insert(
            RawBytesAsset(pin_bytes^),
            AssetUuid.from_path("pinned.bin"),
            "pinned.bin",
            1,
        )
        assert_true(srv.total_resident_bytes() > 65536)
        assert_true(srv.meshes.release(h_mesh))
        var evicted = srv.enforce_lru_budget()
        assert_true(evicted >= 1)
        assert_false(srv.meshes.is_alive(h_mesh))
        assert_true(srv.textures.is_alive(h_tex))
        assert_true(srv.raws.is_alive(h_pin))

        var h_mesh2 = srv.load_mesh_async("meshes/prop.glb")
        _ = srv.poll_async_loads(pool)
        assert_equal(srv.mesh_state(h_mesh2), ASSET_STATE_READY)
        assert_true(h_mesh2 != h_mesh)

        # 5. Codec decode -> ScalableRenderer multi-frame render + FLIP / ImageDiff
        var dec_glb = load_native_glb(glb_path)
        var dec_jpg = decode_jpeg(read_bytes(ws + "/textures/albedo.jpg"), 1024)
        var dec_hdr = decode_hdr_or_exr(read_bytes(ws + "/env/sky.hdr"))
        assert_true(dec_glb.loaded() and len(dec_glb.meshes) >= 1)
        assert_true(dec_jpg.valid and dec_jpg.width == 32 and dec_jpg.height == 32)
        assert_true(dec_hdr.valid and dec_hdr.width == 16 and dec_hdr.height == 8)

        var sample_px = dec_jpg.pixels[16 * 32 + 16]
        var mat_col = Vec3(
            Float32((sample_px >> 16) & 0xFF) / Float32(255.0),
            Float32((sample_px >> 8) & 0xFF) / Float32(255.0),
            Float32(sample_px & 0xFF) / Float32(255.0),
        )
        if ren.uses_vulkan():
            var bslot = ren.gpu_slot[0].upload_bc7_texture(dec_jpg.pixels, dec_jpg.width, dec_jpg.height)
            assert_true(bslot >= 0)
            var ibl = bake_hdr_image_ibl_cubemap_and_sh9(
                dec_hdr,
                base_dim=ren.gpu_slot[0].env_cube_dim,
                mip_levels=ren.gpu_slot[0].env_mip_levels,
                ggx_samples=8,
            )
            ren.gpu_slot[0].upload_ibl_environment(ibl)

        var mat_id = ren.register_material(MaterialAsset(base_color=mat_col, roughness=0.35, metallic=0.2))
        var ren_mesh_id = ren.register_mesh(dec_glb.meshes[0], mat_id, build_lods=True)

        var empty_ui = UiDrawList()
        var rw_blank = RenderWorld()
        rw_blank.camera.eye = Vec3(0.0, 1.0, 4.0)
        rw_blank.camera.target = Vec3(0.0, 0.0, 0.0)
        assert_true(ren.render_world(rw_blank, empty_ui, capture_readback=True, force_readback=True))
        var blank_px = ren.readback_pixels()

        var ext = RenderExtractor()
        var rw1 = RenderWorld()
        rw1.camera.eye = Vec3(0.0, 1.0, 4.0)
        rw1.camera.target = Vec3(0.0, 0.0, 0.0)
        ext.begin_extraction(rw1, 1)
        ext.extract_entity(rw1, UInt64(1), ren_mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), mat_id)
        ext.end_extraction()
        assert_true(ren.render_world(rw1, empty_ui, capture_readback=True, force_readback=True))
        var px1 = ren.readback_pixels()

        var rw2 = RenderWorld()
        rw2.camera.eye = Vec3(0.0, 1.0, 4.0)
        rw2.camera.target = Vec3(0.0, 0.0, 0.0)
        ext.begin_extraction(rw2, 2)
        ext.extract_entity(
            rw2,
            UInt64(1),
            ren_mesh_id,
            trs(Vec3(1.1, 0.2, 0.0), Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.5)), Vec3(1.0, 1.0, 1.0)),
            mat_id,
        )
        ext.end_extraction()
        assert_true(ren.render_world(rw2, empty_ui, capture_readback=True, force_readback=True))
        var px2 = ren.readback_pixels()

        var d_geo = compare_pixels(px1, blank_px, W, H, tolerance=4)
        assert_true(d_geo.over_tolerance >= 400)
        assert_true(d_geo.worst >= 24)
        assert_true(d_geo.flip_mean > 0.01)

        var d_self = compare_pixels(px1, px1, W, H, tolerance=2)
        assert_true(d_self.passes(0, 0.001))

        var d_move = compare_pixels(px2, px1, W, H, tolerance=4)
        assert_true(d_move.over_tolerance >= 100)
        var move_emap = flip_error_map(px1, px2, W, H, ppd=FLIP_PPD)
        assert_true(flip_mean(move_emap) > 0.005)
    finally:
        ren.shutdown()
        srv.close()
        pool.stop()


# ===========================================================================
# Scenario 3: Multiplayer Match Simulation
# ===========================================================================

def t4_make_phys_input(peer: Int, tick: Int) -> PhysicsRollbackInput:
    var ix = Float32(((tick * 3 + peer * 5) % 7) - 3) * Float32(2.0)
    var iz = Float32(((tick + peer * 2) % 5) - 2) * Float32(2.0)
    var yaw = Float32(0.4) if (peer == 0) else Float32(-0.25)
    return PhysicsRollbackInput(ix, 0.0, iz, yaw, 1 if (tick % 6 == 0) else 0)


def t4_s3_multiplayer_match_simulation() raises:
    set_sim_float_mode()

    # 1. Impaired VirtualNetSim + NetHub cookie handshake + encrypted fragmented reliable & sequenced exchange
    var sim = VirtualNetSim.from_rtt(
        rtt_ms=40.0,
        jitter_ms=10.0,
        loss_rate=0.05,
        dup_rate=0.01,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
        seed=0x5C3A2026,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var cookie_key = AeadKey.from_u64x4(0x1111, 0x2222, 0x3333, 0x4444)
    var enc_key = AeadKey.from_u64x4(0x5555, 0x6666, 0x7777, 0x8888)

    var host = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9500), True)
    host.enable_cookie_handshake(cookie_key, bucket_ms=10000.0)
    host.enable_channel_encryption(enc_key)
    var host_ep = host.local_endpoint()

    var cli0 = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9500), False)
    var cli1 = NetHub[VirtualEndpoint](sim.create_ipv4_peer(2, 9500), False)
    cli0.enable_channel_encryption(enc_key)
    cli1.enable_channel_encryption(enc_key)
    assert_true(cli0.send_handshake_init(host_ep))
    assert_true(cli1.send_handshake_init(host_ep))

    for _ in range(120):
        var now = sim.now_ms()
        host.poll(now)
        cli0.poll(now)
        cli1.poll(now)
        if host.open_count() == 2:
            break
        sim.advance_ms(10.0)
    assert_equal(host.open_count(), 2)
    assert_true(host.cookies_sent >= 2)

    var cid0 = cli0.connect(host_ep)
    var cid1 = cli1.connect(host_ep)
    assert_true(cid0 >= 0 and cid1 >= 0)
    var roster_frag = t4_make_payload(7, 2048)  # >1.5 KiB fragmented reliable payload
    var ready_msg = t4_make_payload(9, 48)
    assert_true(cli0.send_reliable(cid0, roster_frag))
    assert_true(cli1.send_reliable(cid1, ready_msg))

    var got_frag = False
    var got_ready = False
    var got_seq_ping = False
    for step in range(220):
        var now = sim.now_ms()
        if step < 10:
            var ping = List[UInt8]()
            ping.append(UInt8(step))
            _ = cli0.send_unreliable(cid0, ping, now)
        host.poll(now)
        cli0.poll(now)
        cli1.poll(now)
        for cid in range(host.count):
            while True:
                var rm = host.pop_reliable(cid)
                if not rm[0]:
                    break
                if len(rm[1]) == 2048:
                    assert_equal(hash_list(rm[1]), hash_list(roster_frag))
                    got_frag = True
                elif len(rm[1]) == 48:
                    assert_equal(hash_list(rm[1]), hash_list(ready_msg))
                    got_ready = True
            while True:
                var um = host.pop_unreliable(cid)
                if not um[0]:
                    break
                got_seq_ping = True
        if got_frag and got_ready and got_seq_ping:
            break
        sim.advance_ms(10.0)
    assert_true(got_frag and got_ready and got_seq_ping)

    # 2. RollbackSession + PhysicsRollbackAdapter + blackout survival + VisualSmoother
    var rb_ep0 = sim.create_ipv4_peer(3, 9600)
    var rb_ep1 = sim.create_ipv4_peer(4, 9600)
    var rb_addrs = List[NetEndpoint]()
    rb_addrs.append(rb_ep0.local_endpoint())
    rb_addrs.append(rb_ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s0 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=0,
        peer_endpoints=rb_addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
    )
    var s1 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=1,
        peer_endpoints=rb_addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
    )
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    comptime TARGET_TICKS: Int = 120
    for frame in range(280):
        if frame == 50:
            sim.drop_next_outbound_packets(3, 6)
        var now = sim.now_ms()
        var i0 = t4_make_phys_input(0, s0.current_tick)
        var i1 = t4_make_phys_input(1, s1.current_tick)
        _ = s0.step_frame(i0, pw0, rb_ep0, now)
        _ = s1.step_frame(i1, pw1, rb_ep1, now)
        if len(s0.confirmed_hash_history) >= TARGET_TICKS and len(s1.confirmed_hash_history) >= TARGET_TICKS:
            break
        sim.advance_ms(1000.0 / 60.0)

    assert_false(s0.desync_detected)
    assert_false(s1.desync_detected)
    assert_true(s0.rollback_count + s1.rollback_count > 0)
    assert_true(len(s0.confirmed_hash_history) >= TARGET_TICKS)
    assert_true(len(s1.confirmed_hash_history) >= TARGET_TICKS)
    for t in range(TARGET_TICKS):
        assert_equal(s0.confirmed_hash_history[t], s1.confirmed_hash_history[t])

    var slot_p0_a = pw0.world.find_slot_by_id(pw0.player_body_ids[0])
    var slot_p0_b = pw1.world.find_slot_by_id(pw1.player_body_ids[0])
    assert_true(slot_p0_a >= 0 and slot_p0_b >= 0)
    assert_true((pw0.world.bodies[slot_p0_a].pos - pw1.world.bodies[slot_p0_b].pos).length() < Float32(0.05))

    var smoother = VisualSmoother(decay_rate=18.0)
    smoother.record_correction(Vec3(0.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0))
    smoother.step(Float32(0.04))
    var sp1 = smoother.smooth_pos(Vec3(3.0, 0.0, 0.0))
    assert_true(sp1.x > Float32(0.1) and sp1.x < Float32(2.9))
    smoother.step(Float32(0.04))
    var sp2 = smoother.smooth_pos(Vec3(3.0, 0.0, 0.0))
    assert_true(sp2.x > sp1.x)

    # 3. Distant open-world AOI SnapshotReplicator + Interpolation + Despawn + Prediction + LagComp
    var rep = SnapshotReplicator(
        max_entities=16,
        max_clients=2,
        max_packet_bytes=900,
        interest_radius=80.0,
        cell_size=16.0,
        use_3d_grid=True,
        chunk_size=256.0,
    )
    var recv0 = SnapshotClientReceiver(max_entities=16, chunk_size=256.0)
    var world_offset = Vec3(Float32(100_000.0), Float32(0.0), Float32(-100_000.0))
    var p0_world = pw0.world.bodies[slot_p0_a].pos + world_offset
    var proj_world = p0_world + Vec3(5.0, 1.0, 0.0)

    rep.set_client_view(0, p0_world, 80.0)
    rep.set_entity(0, p0_world, Quat.identity(), pw0.world.bodies[slot_p0_a].linear_vel, UInt32(1))
    rep.set_entity(1, proj_world, Quat.identity(), Vec3(2.0, 0.0, 0.0), UInt32(2))
    rep.set_entity(2, p0_world + Vec3(500.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(3))  # outside AOI
    rep.set_client_acked_input_seq(0, 42)
    rep.prepare_tick()
    var snap_a = rep.build_client_packet(0)
    var app_a = recv0.apply_packet(snap_a[1])
    assert_true(app_a[0])
    rep.notify_client_ack(0, app_a[1])
    assert_true(recv0.entities[0].active and recv0.entities[1].active)
    assert_false(recv0.entities[2].active)
    assert_true(recv0.has_acked_input_seq and recv0.last_acked_input_seq == 42)
    assert_true((recv0.entities[0].pos - p0_world).length() < Float32(0.1))

    var interp = SnapshotInterpolationBuffer(max_entities=16, interp_delay_ms=100.0)
    interp.push_snapshot(100.0, recv0.entities)

    # Move player + destroy projectile on tick 2
    var p0_next = p0_world + Vec3(4.0, 0.0, 0.0)
    rep.set_entity(0, p0_next, Quat.identity(), Vec3(4.0, 0.0, 0.0), UInt32(1))
    rep.destroy_entity(1)
    rep.prepare_tick()
    var snap_b = rep.build_client_packet(0)
    var app_b = recv0.apply_packet(snap_b[1])
    assert_true(app_b[0])
    assert_false(recv0.entities[1].active)
    interp.push_snapshot(200.0, recv0.entities)

    var sampled = List[ReplicatedEntity]()
    assert_true(interp.sample(250.0, sampled))
    assert_true(sampled[0].active)
    assert_true(abs(sampled[0].pos.x - (p0_world.x + Float32(2.0))) < Float32(0.25))

    # ClientPredictionBuffer reconciliation
    var pred = ClientPredictionBuffer[PhysicsRollbackInput](capacity=16)
    var stepper = T4PredictionStepper()
    for seq in range(1, 5):
        var cmd = PhysicsRollbackInput(2.0, 0.0, 0.0, 0.0, 0)
        pred.push_predicted(
            seq,
            cmd,
            PredictedState(Vec3(Float32(seq) * 2.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)),
            tick=seq,
            dt=Float32(1.0),
        )
    var rec = pred.reconcile_with(
        acked_seq=1,
        auth_state=PredictedState(Vec3(3.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)),
        stepper=stepper,
        default_dt=Float32(1.0),
    )
    assert_equal(pred.mispredictions, 1)
    assert_true(pred.replayed_inputs >= 2)
    assert_true(abs(rec.pos.x - Float32(9.0)) < Float32(1e-4))

    # LagCompHistory temporal raycast hit-vs-miss
    var lag = LagCompHistory(max_entities=16, max_history_ticks=16)
    for tick in range(10):
        var poses = List[HitboxPose]()
        poses.append(HitboxPose.sphere(1, Vec3(Float32(tick) * 2.0, 1.0, 0.0), Float32(0.75)))
        lag.record_tick(tick, poses)
    var hit_past = lag.raycast_at_tick(3, Vec3(6.0, 1.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_true(hit_past.hit and hit_past.entity_id == 1)
    var miss_now = lag.raycast_at_tick(9, Vec3(6.0, 1.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_false(miss_now.hit)


# ===========================================================================
# Scenario 4: Interactive UI, Text & Input Pipeline
# ===========================================================================

def t4_s4_interactive_ui_text_input_pipeline() raises:
    # 1. InputMap contexts (Gameplay vs Menu priority blocking, 2D axis, buffered jump)
    var imap = InputMap()
    _ = imap.register_action("Move", ACTION_AXIS2D)
    _ = imap.register_action("Jump", ACTION_BOOL)
    _ = imap.register_action("MenuConfirm", ACTION_BOOL)

    var gameplay = InputContext("Gameplay", priority=0, blocks_lower_all=False)
    gameplay.bind_key("Move", 68, scale_x=1.0, scale_y=0.0)   # D
    gameplay.bind_key("Move", 65, scale_x=-1.0, scale_y=0.0)  # A
    gameplay.bind_key("Jump", 32)                             # Space
    imap.add_context(gameplay^)

    var menu = InputContext("Menu", priority=10, blocks_lower_all=True)
    menu.bind_key("MenuConfirm", 32, consume=True)
    imap.add_context(menu^)
    imap.push_context("Gameplay")

    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(68, True)
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    assert_true(imap.just_pressed("Jump"))
    assert_true(imap.axis_2d("Move")[0] > Float32(0.5))

    # Buffered jump across 4 frames while airborne
    inp.set_logical_key(68, False)
    inp.set_logical_key(32, False)
    imap.set_grounded(False)
    for _ in range(4):
        inp.begin_frame()
        imap.update(inp, 0.016)
    assert_true(imap.consume_buffered("Jump", 0.120))
    assert_false(imap.consume_buffered("Jump", 0.120))

    # Menu context priority blocks Gameplay Jump
    imap.push_context("Menu")
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    assert_true(imap.just_pressed("MenuConfirm"))
    assert_false(imap.is_down("Jump"))
    inp.set_logical_key(32, False)
    assert_true(imap.pop_context("Menu"))

    # 2. TTF + OTF CFF font loading, SDF atlas + MSDF glyph baking, wrapping & rich text
    var ttf = TrueTypeFont()
    assert_true(ttf.load(SYSTEM_TTF_PATH))
    assert_true(ttf.valid and not ttf.is_cff)

    var otf = TrueTypeFont()
    assert_true(otf.load(SYSTEM_OTF_PATH))
    assert_true(otf.valid and otf.is_cff)

    var msdf_a = bake_glyph_msdf(otf, 65, 32, 4)
    assert_true(msdf_a.w > 0 and msdf_a.h > 0 and msdf_a.channels == 3)

    var codes = List[Int]()
    for c in range(32, 127):
        codes.append(c)
    var baked = bake_font(ttf, codes, 32, 4, atlas_width=512, channels=1)
    var dsf = dsf_bytes(baked)
    var magic = Int(dsf[0]) | (Int(dsf[1]) << 8) | (Int(dsf[2]) << 16) | (Int(dsf[3]) << 24)
    assert_equal(magic, DSF_MAGIC)

    var sdf = SdfFont()
    assert_true(sdf.load_bytes(dsf))
    assert_true(sdf.valid)
    _ = sdf.attach_source(SYSTEM_TTF_PATH)

    var wrapped = wrap(sdf, decode_utf8("Flint UI & Rich Text Pipeline"), 18.0, 120.0)
    assert_true(len(wrapped) >= 2)
    var spans = parse_rich("[c=78E2C8]HP 100[/c] [s=120]Ready[/s] [btn=jump]")
    assert_true(plain_text(spans).find("HP 100") >= 0)
    var rlayout = layout_rich(spans, sdf, sdf, 18.0, 0, 240.0, 0, False, False)
    assert_true(rlayout.lines >= 1 and len(rlayout.runs) >= 3)

    # 3. 5-panel DockSpace + UiContext multi-frame interaction (button, UTF-8 text_input, clipboard)
    var dock = DockSpace.create_studio_5panel()
    var screen = RectF(0.0, 0.0, 1280.0, 720.0)
    dock.solve_layout(screen)
    assert_equal(len(dock.leaf_rects()), 5)

    var ui = UiContext(preset_hud=False)
    ui.font = sdf.copy()

    var srv = AudioServer(sample_rate=48000, quantum_frames=256)
    var click_pcm = t4_make_sine_pcm16(880.0, 4800, channels=1, amp=0.85, sample_rate=48000)
    var click_aid = srv.register_pcm16_asset(click_pcm, 1, 48000)
    assert_true(click_aid >= 0)

    try:
        var chat_text = String("")
        var btn_rect = RectF()
        var input_rect = RectF()

        # Frame 1: draw widgets and capture rects
        inp.begin_frame()
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, screen)
        if ui.begin_dock_panel(dock, "Outliner"):
            _ = ui.button("btn_play", "Play")
            btn_rect = ui.last_item_rect
            ui.end_dock_panel()
        if ui.begin_dock_panel(dock, "Console"):
            ui.rich_label("[c=3EE88A]Match Ready[/c] [btn=jump]", 240.0)
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            input_rect = ui.last_item_rect
            ui.end_dock_panel()
        ui.end_frame()
        assert_true(btn_rect.w > Float32(0.0) and input_rect.w > Float32(0.0))

        # Frame 1b: hover without press/release -> must NOT click
        inp.begin_frame()
        inp.move_mouse(btn_rect.center_x(), btn_rect.center_y())
        inp.set_mouse(0, False)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, screen)
        var clicked_hover_only = False
        if ui.begin_dock_panel(dock, "Outliner"):
            clicked_hover_only = ui.button("btn_play", "Play")
            ui.end_dock_panel()
        if ui.begin_dock_panel(dock, "Console"):
            ui.rich_label("[c=3EE88A]Match Ready[/c] [btn=jump]", 240.0)
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        assert_false(clicked_hover_only)

        # Frame 1c: press inside Play button, then release outside -> must NOT click
        inp.begin_frame()
        inp.move_mouse(btn_rect.center_x(), btn_rect.center_y())
        inp.set_mouse(0, True)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, screen)
        var clicked_press_before_off = False
        if ui.begin_dock_panel(dock, "Outliner"):
            clicked_press_before_off = ui.button("btn_play", "Play")
            ui.end_dock_panel()
        if ui.begin_dock_panel(dock, "Console"):
            ui.rich_label("[c=3EE88A]Match Ready[/c] [btn=jump]", 240.0)
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        assert_false(clicked_press_before_off)

        inp.begin_frame()
        inp.move_mouse(btn_rect.x + btn_rect.w + Float32(40.0), btn_rect.y + btn_rect.h + Float32(40.0))
        inp.set_mouse(0, False)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, screen)
        var clicked_release_off = False
        if ui.begin_dock_panel(dock, "Outliner"):
            clicked_release_off = ui.button("btn_play", "Play")
            ui.end_dock_panel()
        if ui.begin_dock_panel(dock, "Console"):
            ui.rich_label("[c=3EE88A]Match Ready[/c] [btn=jump]", 240.0)
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        assert_false(clicked_release_off)

        # Frame 2: mouse press on Play button
        inp.begin_frame()
        inp.move_mouse(btn_rect.center_x(), btn_rect.center_y())
        inp.set_mouse(0, True)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, screen)
        var clicked_press = False
        if ui.begin_dock_panel(dock, "Outliner"):
            clicked_press = ui.button("btn_play", "Play")
            ui.end_dock_panel()
        if ui.begin_dock_panel(dock, "Console"):
            ui.rich_label("[c=3EE88A]Match Ready[/c] [btn=jump]", 240.0)
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        assert_false(clicked_press)

        # Frame 3: mouse release on Play button -> triggers click + UI audio chime
        inp.begin_frame()
        inp.set_mouse(0, False)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, screen)
        var clicked_rel = False
        if ui.begin_dock_panel(dock, "Outliner"):
            clicked_rel = ui.button("btn_play", "Play")
            ui.end_dock_panel()
        if ui.begin_dock_panel(dock, "Console"):
            ui.rich_label("[c=3EE88A]Match Ready[/c] [btn=jump]", 240.0)
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        assert_true(clicked_rel)

        var h_ui = srv.play_2d(click_aid, volume=0.9, bus_id=BUS_UI, direct_stereo=True)
        assert_true(h_ui.is_valid())

        # Frame 4: click inside text_input and push UTF-8 text
        inp.begin_frame()
        inp.move_mouse(input_rect.x + Float32(12.0), input_rect.center_y())
        inp.set_mouse(0, True)
        inp.push_text_utf8("Go! ★")
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        if ui.begin_dock_panel(dock, "Console"):
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        assert_equal(chat_text, "Go! ★")

        # Frame 5: release mouse, press Left, verify UTF-8 codepoint boundary
        inp.set_mouse(0, False)
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT, True)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        if ui.begin_dock_panel(dock, "Console"):
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        inp.set_logical_key(UI_KEY_LEFT, False)
        var boff = ui.text_cursor_byte_offset("chat_box", chat_text)
        if boff < chat_text.byte_length():
            assert_true((Int(chat_text.as_bytes()[boff]) & 0xC0) != 0x80)

        # Frame 6: Backspace removes space before ★
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_BACKSPACE, True)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        if ui.begin_dock_panel(dock, "Console"):
            assert_true(ui.text_input("chat_box", chat_text, 220.0, placeholder="Type..."))
            ui.end_dock_panel()
        ui.end_frame()
        inp.set_logical_key(UI_KEY_BACKSPACE, False)
        assert_equal(chat_text, "Go!★")
        for cp in decode_utf8(chat_text):
            assert_true(cp != REPLACEMENT)

        # Frame 7: Ctrl+A (select all)
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
        inp.set_logical_key(UI_KEY_A, True)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        if ui.begin_dock_panel(dock, "Console"):
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        inp.set_logical_key(UI_KEY_A, False)

        # Frame 8: Ctrl+C (copy selection to clipboard)
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_C, True)
        ui.begin_frame(inp, 1280.0, 720.0, Float32(1.0 / 60.0))
        if ui.begin_dock_panel(dock, "Console"):
            _ = ui.text_input("chat_box", chat_text, 220.0, placeholder="Type...")
            ui.end_dock_panel()
        ui.end_frame()
        inp.set_logical_key(UI_KEY_C, False)
        inp.set_logical_key(UI_KEY_LEFT_CTRL, False)

        assert_equal(ui.get_clipboard(), "Go!★")
        assert_true(ui.draw_list.count() > 0)

        # 4. AudioServer offline UI feedback verification
        var out_lr = List[Float32](length=512, fill=0.0)
        var peak = Float32(0.0)
        for _ in range(8):
            srv.render_offline_quantum(out_lr, 256)
            for s in out_lr:
                assert_false(isnan(s) or isinf(s))
                var a = abs(s)
                if a > peak:
                    peak = a
        assert_true(peak > Float32(0.05) and peak <= Float32(0.99))
        assert_equal(srv.quantums_rendered(), UInt64(8))
        assert_true(srv.momentary_lufs() > Float32(-40.0))
    finally:
        srv.stop()
        inp.close()


# ===========================================================================
# Scenario 5: Studio & DevTools Authoring Workflow
# ===========================================================================

def t4_s5_studio_devtools_authoring_workflow(tmp_root: String) raises:
    var ws = tmp_root + "/s5_ws"
    assert_true(make_dirs(ws))

    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=320,
        height=180,
        worker_threads=2,
    )
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    _ = pool.start(2)

    try:
        # 1. Entity creation, multi-select, cycle-safe reparenting, coalesced transforms & bulk patch, undo/redo
        var root_id = studio.spawn_entity("RootRig", position=Vec3(2.0, 0.0, 0.0), collider_kind=COLLIDER_BOX, mass=0.0)
        var turret_id = studio.spawn_entity("Turret", position=Vec3(0.0, 1.5, 0.0), collider_kind=COLLIDER_BOX, mass=0.0, parent_id=root_id)
        var orb_id = studio.spawn_entity("DynamicOrb", position=Vec3(0.0, 5.0, 0.0), collider_kind=COLLIDER_SPHERE, mass=4.0)
        assert_true(root_id >= 0 and turret_id >= 0 and orb_id >= 0)

        studio.select_entity(turret_id, multi_select=False)
        studio.select_entity(orb_id, multi_select=True)
        assert_equal(studio.selected_count(), 2)
        var pivot = studio.selection_gizmo_pivot()
        assert_true(pivot[0])

        assert_true(studio.reparent_entity(orb_id, turret_id, preserve_world_transform=True, record_undo=True))
        assert_equal(studio.parent_of(orb_id), turret_id)
        var orb_wmat = studio.world_matrix(orb_id)
        assert_true(abs(orb_wmat.at(0, 3)) < Float32(1e-3))
        assert_true(abs(orb_wmat.at(1, 3) - Float32(5.0)) < Float32(1e-3))
        # Cycle rejection: root cannot be parented under its descendant orb
        assert_false(studio.reparent_entity(root_id, orb_id, preserve_world_transform=False, record_undo=True))

        var h_pre_edit = studio.state_hash()
        assert_true(
            studio.set_entity_transform(
                turret_id,
                Vec3(0.5, 1.5, 0.0),
                Quat(),
                Vec3(1.0, 1.0, 1.0),
                coalesce_key=101,
            )
        )
        assert_true(
            studio.set_entity_transform(
                turret_id,
                Vec3(1.0, 1.5, 0.0),
                Quat(),
                Vec3(1.0, 1.0, 1.0),
                coalesce_key=101,
            )
        )
        var f_rough = InspectedField("roughness", FIELD_F32)
        f_rough.f_val = Float32(0.72)
        assert_equal(studio.bulk_patch_field("roughness", f_rough, coalesce_key=202), 2)
        var h_post_edit = studio.state_hash()
        assert_true(h_post_edit != h_pre_edit)

        assert_true(studio.undo())  # undo bulk_patch_field
        assert_true(studio.undo())  # undo coalesced transform
        assert_equal(studio.state_hash(), h_pre_edit)
        assert_true(studio.redo())
        assert_true(studio.redo())
        assert_equal(studio.state_hash(), h_post_edit)

        # 2. PIE play / pause / step / stop exact hash restoration
        var pre_pie_hash = studio.state_hash()
        var pre_orb_y = studio.entities[orb_id].translation.y
        assert_true(studio.pie_play())
        assert_true(studio.pie_pause())
        assert_true(studio.pie_pause())
        assert_equal(studio.pie_step(45), 45)
        assert_true(studio.entities[orb_id].translation.y < pre_orb_y)
        assert_true(studio.state_hash() != pre_pie_hash)
        assert_true(studio.pie_stop())
        assert_true(studio.pie.last_restore_verified)
        assert_equal(studio.state_hash(), pre_pie_hash)

        # 3. Save & reload .flscn AND .flscnb
        var flscn_path = ws + "/level.flscn"
        var flscnb_path = ws + "/level.flscnb"
        assert_true(studio.save_scene_file(flscn_path))
        assert_true(studio.save_scene_file(flscnb_path))

        var studio_json = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=180)
        var studio_bin = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=180)
        try:
            assert_true(studio_json.load_scene_file(flscn_path))
            assert_equal(studio_json.state_hash(), pre_pie_hash)
            assert_equal(studio_json.parent_of(orb_id), turret_id)

            assert_true(studio_bin.load_scene_file(flscnb_path))
            assert_equal(studio_bin.state_hash(), pre_pie_hash)
            assert_equal(studio_bin.parent_of(orb_id), turret_id)
        finally:
            studio_json.shutdown()
            studio_bin.shutdown()

        # 4. Register default mesh on studio's ScalableRenderer, render editor frame, raycast & VisBufferGpuPicker
        var ren_ptr = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        assert_true(Bool(ren_ptr))
        var s_mat_id = ren_ptr.value()[unsafe_offset=0].register_material(
            MaterialAsset(base_color=Vec3(0.82, 0.84, 0.88), roughness=0.38, metallic=0.1)
        )
        var s_mb = MeshBuilder()
        s_mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5))
        _ = ren_ptr.value()[unsafe_offset=0].register_mesh(s_mb^.finish(), s_mat_id, build_lods=False)

        studio.render_editor_frame(320, 180)
        assert_true(len(studio.ui_ctx.draw_list.prims) > 0)
        var orb_world = studio.world_matrix(orb_id)
        var orb_pos = Vec3(orb_world.at(0, 3), orb_world.at(1, 3), orb_world.at(2, 3))
        var rhit = studio.raycast_scene(orb_pos + Vec3(0.0, 5.0, 0.0), Vec3(0.0, -1.0, 0.0), 20.0)
        assert_true(rhit.hit)
        assert_equal(rhit.entity_id, orb_id)

        var inst_to_ent = List[Int]()
        var inst_to_bits = List[UInt64]()
        studio.fill_render_instance_entity_maps(inst_to_ent, inst_to_bits)
        assert_true(len(inst_to_ent) >= 3)

        var words = encode_visbuffer_words(instance_id=1, meshlet_id=2, material_id=4, local_prim_id=7)
        var dec_pick = decode_visbuffer_pick(
            160,
            90,
            words[0],
            words[1],
            Float32(0.05),
            320,
            180,
            studio.cam_pos,
            studio.cam_fwd,
            studio.cam_right,
            studio.cam_up,
            studio.fov_y_deg,
            studio.near_z,
            inst_to_ent,
            Float32(0.0),
            Float32(0.0),
            inst_to_bits,
        )
        assert_true(dec_pick.hit)
        assert_equal(dec_pick.entity_id, inst_to_ent[1])
        assert_equal(dec_pick.meshlet_id, 2)
        assert_equal(dec_pick.material_id, 4)
        assert_equal(dec_pick.local_prim_id, 7)

        var picker = VisBufferGpuPicker()
        var req_id = picker.request_pick(160, 90, inst_to_ent, inst_to_bits)
        picker.push_software_pick_result(req_id, dec_pick)
        assert_true(picker.has_completed_pick())
        if ren_ptr.value()[unsafe_offset=0].uses_vulkan():
            var polled = picker.poll_completed_pick_entry(ren_ptr.value()[unsafe_offset=0].gpu_slot[0])
            assert_true(Bool(polled))
            assert_equal(polled.value()[0], req_id)
            assert_equal(polled.value()[1].entity_id, inst_to_ent[1])
            picker.shutdown(ren_ptr.value()[unsafe_offset=0].gpu_slot[0])
        else:
            var probe_gpu = FlintAaaVulkanRenderer(
                width=16,
                height=16,
                max_draws=256,
                max_instances=256,
                env_cube_dim=16,
            )
            var polled = picker.poll_completed_pick_entry(probe_gpu)
            assert_true(Bool(polled))
            assert_equal(polled.value()[0], req_id)
            assert_equal(polled.value()[1].entity_id, inst_to_ent[1])
            picker.shutdown(probe_gpu)
            probe_gpu.shutdown()

        # 5. McpServerV2 JSON-RPC 2.0 session + capture_viewport FLIP diff
        var mcp = McpServerV2(workspace_root=ws, auth_token="t4-studio-token")
        var init_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","auth_token":"t4-studio-token"}}',
                studio,
                assets,
                pool,
            )
        )
        assert_true(init_doc.valid and mcp.session_authenticated)
        assert_equal(
            mcp.handle_jsonrpc('{"jsonrpc":"2.0","method":"notifications/initialized"}', studio, assets, pool),
            "",
        )

        var list_doc = t4_parse_json_str(
            mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"tools/list"}', studio, assets, pool)
        )
        var tools_arr = list_doc.object_get(list_doc.object_get(list_doc.root, "result"), "tools")
        assert_equal(list_doc.array_size(tools_arr), 14)

        var gen_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"procedural.generate_asset","arguments":{"preset":"material_orb","output_glb_path":"assets/orb.glb","spawn_in_scene":true,"position":[1.5,1.2,1.0]}}}',
                studio,
                assets,
                pool,
            )
        )
        var gen_sc = gen_doc.object_get(gen_doc.object_get(gen_doc.root, "result"), "structuredContent")
        assert_true(gen_doc.boolean(gen_doc.object_get(gen_sc, "ok"), False))
        var gen_eid = gen_doc.integer(gen_doc.object_get(gen_sc, "spawned_entity_id"), -1)
        assert_true(gen_eid >= 0)
        assert_true(file_size(ws + "/assets/orb.glb") > 0)

        var cap1_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"baseline.png","width":128,"height":96}}}',
                studio,
                assets,
                pool,
            )
        )
        var cap1_sc = cap1_doc.object_get(cap1_doc.object_get(cap1_doc.root, "result"), "structuredContent")
        assert_true(cap1_doc.boolean(cap1_doc.object_get(cap1_sc, "ok"), False))
        assert_false(cap1_doc.boolean(cap1_doc.object_get(cap1_sc, "has_flip_diff"), True))
        assert_true(file_size(ws + "/baseline.png") > 0)

        var cap_same_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"baseline_copy.png","reference_path":"baseline.png","width":128,"height":96}}}',
                studio,
                assets,
                pool,
            )
        )
        var cap_same_sc = cap_same_doc.object_get(cap_same_doc.object_get(cap_same_doc.root, "result"), "structuredContent")
        assert_true(cap_same_doc.boolean(cap_same_doc.object_get(cap_same_sc, "ok"), False))
        assert_true(cap_same_doc.boolean(cap_same_doc.object_get(cap_same_sc, "has_flip_diff"), False))
        assert_equal(cap_same_doc.number(cap_same_doc.object_get(cap_same_sc, "flip_mean"), -1.0), 0.0)

        var patch_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ecs.patch_components","arguments":{"entity_id":'
                + String(gen_eid)
                + ',"position":[0.2,2.0,-0.5],"base_color":[0.95,0.12,0.10]}}}',
                studio,
                assets,
                pool,
            )
        )
        var patch_sc = patch_doc.object_get(patch_doc.object_get(patch_doc.root, "result"), "structuredContent")
        assert_true(patch_doc.boolean(patch_doc.object_get(patch_sc, "ok"), False))

        var cap2_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"edited.png","reference_path":"baseline.png","width":128,"height":96}}}',
                studio,
                assets,
                pool,
            )
        )
        var cap2_sc = cap2_doc.object_get(cap2_doc.object_get(cap2_doc.root, "result"), "structuredContent")
        assert_true(cap2_doc.boolean(cap2_doc.object_get(cap2_sc, "ok"), False))
        assert_true(cap2_doc.boolean(cap2_doc.object_get(cap2_sc, "has_flip_diff"), False))
        assert_true(cap2_doc.number(cap2_doc.object_get(cap2_sc, "flip_mean"), 0.0) > 0.0)

        var insp_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"render.inspect_gpu_frame","arguments":{}}}',
                studio,
                assets,
                pool,
            )
        )
        var insp_sc = insp_doc.object_get(insp_doc.object_get(insp_doc.root, "result"), "structuredContent")
        assert_true(insp_doc.boolean(insp_doc.object_get(insp_sc, "ok"), False))

        var undo_doc = t4_parse_json_str(
            mcp.handle_jsonrpc(
                '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}',
                studio,
                assets,
                pool,
            )
        )
        var undo_sc = undo_doc.object_get(undo_doc.object_get(undo_doc.root, "result"), "structuredContent")
        assert_equal(undo_doc.integer(undo_doc.object_get(undo_sc, "applied_steps"), 0), 1)

        # 6. EngineBenchSuite determinism, latency samples, verify_golden_frame & CSV export
        var det_a = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
        var det_b = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
        var hash_a: UInt64
        var hash_b: UInt64
        try:
            _ = det_a.spawn_entity("DetBall", position=Vec3(0.0, 4.0, 0.0), collider_kind=COLLIDER_SPHERE, mass=2.0)
            _ = det_b.spawn_entity("DetBall", position=Vec3(0.0, 4.0, 0.0), collider_kind=COLLIDER_SPHERE, mass=2.0)
            assert_true(det_a.pie_play() and det_b.pie_play())
            _ = det_a.pie_step(30)
            _ = det_b.pie_step(30)
            hash_a = det_a.state_hash()
            hash_b = det_b.state_hash()
            assert_true(det_a.pie_stop() and det_b.pie_stop())
        finally:
            det_a.shutdown()
            det_b.shutdown()

        var suite = EngineBenchSuite("t4_s5_studio")
        assert_true(
            suite.record_determinism_gate(
                "studio_pie_determinism",
                hash_a,
                hash_b,
                elapsed_ms=2.0,
                budget_ms=500.0,
            )
        )
        var samples = List[Float64]()
        samples.append(0.9)
        samples.append(1.1)
        samples.append(1.3)
        samples.append(1.5)
        assert_true(suite.record_samples("editor_frame", samples, budget_p99_ms=20.0))

        var base_img = decode_png(read_bytes(ws + "/baseline.png"), 4096)
        assert_true(base_img.valid)
        assert_true(
            suite.verify_golden_frame(
                "mcp_baseline_self",
                base_img.pixels,
                base_img.width,
                base_img.height,
                ws + "/baseline.png",
                heatmap_out_path=ws + "/unused_heat.png",
                bootstrap_if_missing=False,
            )
        )
        assert_true(suite.all_passed())
        var csv_path = ws + "/studio_bench.csv"
        assert_true(suite.append_perf_history_csv(csv_path, package_name="e2e-t4", date_str="2026-11-20"))
        assert_true(file_size(csv_path) > 0)
    finally:
        assets.close()
        pool.stop()
        studio.shutdown()


def main() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_root = "/tmp/flint-e2e-opaque-t4-" + String(pid)
    t4_rm_rf_tmp(tmp_root)
    assert_true(make_dirs(tmp_root))

    var passed = 0
    var total = 5

    try:
        t4_s1_headless_game_loop()
        print("[PASS] t4_s1_headless_game_loop")
        passed += 1

        t4_s2_asset_cook_to_render_pipeline(tmp_root)
        print("[PASS] t4_s2_asset_cook_to_render_pipeline")
        passed += 1

        t4_s3_multiplayer_match_simulation()
        print("[PASS] t4_s3_multiplayer_match_simulation")
        passed += 1

        t4_s4_interactive_ui_text_input_pipeline()
        print("[PASS] t4_s4_interactive_ui_text_input_pipeline")
        passed += 1

        t4_s5_studio_devtools_authoring_workflow(tmp_root)
        print("[PASS] t4_s5_studio_devtools_authoring_workflow")
        passed += 1
    except e:
        t4_rm_rf_tmp(tmp_root)
        raise e^

    t4_rm_rf_tmp(tmp_root)
    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
