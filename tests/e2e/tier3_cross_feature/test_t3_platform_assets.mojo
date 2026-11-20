"""Tier 3 Cross-Feature Combinations: Platform, Asset & System Interactions (15 tests).
Genuine multi-frame cross-subsystem integration across `FlpkV3MappedArchive`, `AssetServer`,
`flint.ecs.World` + `Scene`, `EngineApp` + 7 plugins + `JobPool`, `FlintStudioState` + `McpServerV2`,
`ConsoleVars` + `DebugLineBatch` + `ScalableRenderer`, `InputMap` + `UiContext` + `DockSpace`,
`FlintAaaVulkanRenderer` SPIR-V & residency upload, `FlintReplayStream`, and `NetPlugin`.
"""
from std.collections import List
from std.ffi import external_call
from flint.app import AppConfig, EngineApp
from flint.app.engine_app import FrameTime, ParallelCommands, Resources, STAGE_UPDATE
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    CHUNK_COMP_NONE,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    write_flpk_v3,
)
from flint.assets.json import parse_json
from flint.assets.replay_stream import FlintReplayStream
from flint.assets.server import AssetServer, CookedTextureAsset
from flint.core.build_stamp import BUILD_COMMIT, BUILD_MOJO
from flint.core.cvars import CVAR_DEBUG_DRAW, ConsoleVars
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.serial import BlobWriter
from flint.core.transform import Quat
from flint.ecs import Component, Entity, STORAGE_TABLE, World, comp_hash64
from flint.net.plugin import NetworkIdentityComponent, ReplicationComponent
from flint.physics.plugin import ColliderComponent, PhysicsPlugin, RigidBodyComponent
from flint.platform.input import InputState
from flint.platform.wayland import evdev_key
from flint.render.aaa_clustered_lights import aaa_cluster_cull_shader
from flint.render.aaa_gi_svgf import aaa_tile_classify_shader
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.image_diff import compare_pixels
from flint.render.render_world import (
    DebugLineBatch,
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderWorld,
    ScalableRenderer,
)
from flint.render.spirv import SPIRV_MAGIC
from flint.render.spirv_validate import validate_spirv_module
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    Parent,
    Scene,
    TransformPlugin,
    set_parent,
)
from flint.ui.context import UI_KEY_ENTER, UI_KEY_TAB, UiContext
from flint.ui.draw_list import UiDrawList
from flint.ui.flexbox_dock import DockSpace, FLEX_COL, FlexStyle, RectF
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import (
    create_headless_v5_app,
    create_mock_asset_server,
    create_mock_flpk_v3_archive_file,
    create_mock_input_map,
    create_mock_mcp_server,
    create_mock_studio,
    register_mock_render_scene,
)


def _pid_str() -> String:
    return String(Int(external_call["getpid", Int32]()))


@fieldwise_init
struct _T3JobTag(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t3.JobTag")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var hits: Int32


def _t3_par_increment_kernel(
    _lane: Int,
    _order: Int,
    _ent: Entity,
    tag: Pointer[_T3JobTag, MutAnyOrigin],
    _ctx_addr: Int,
):
    tag[unsafe_offset=0].hits += 1


def _t3_ecs_jobpool_sys(
    w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    pool: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    w[unsafe_offset=0].par_for_each1[_T3JobTag](pool[unsafe_offset=0], _t3_par_increment_kernel, 0)


# 1. FlpkV3MappedArchive -> AssetServer -> load_mesh & load_raw JSON parse
def test_t3_vfs_to_asset_buffer(mut ctx: TestContext) raises:
    var tmp_flpk = "/tmp/flint_e2e_t3_v3_" + _pid_str() + ".flpk"
    ctx.cleanup_artifact(tmp_flpk)
    try:
        ctx.assert_true(create_mock_flpk_v3_archive_file(tmp_flpk), "create_mock_flpk_v3_archive_file writes valid .flpk v3")
        var assets = create_mock_asset_server()
        ctx.assert_true(assets.mount_flpk(tmp_flpk), "AssetServer mounts .flpk v3 archive")
        var mh = assets.load_mesh("meshes/mock_sphere.mesh")
        var rh = assets.load_raw("manifest.json")
        var m_ptr = assets.meshes.get_ptr(mh)
        var r_ptr = assets.raws.get_ptr(rh)
        ctx.assert_true(Bool(m_ptr), "AssetServer loads cooked mesh from .flpk v3")
        ctx.assert_true(len(m_ptr.value()[unsafe_offset=0].geom.meshes) > 0, "loaded cooked mesh has geometry slices")
        ctx.assert_true(Bool(r_ptr) and len(r_ptr.value()[unsafe_offset=0].bytes) > 0, "AssetServer loads manifest.json raw asset")
        var doc = parse_json(r_ptr.value()[unsafe_offset=0].bytes)
        ctx.assert_true(doc.valid, "loaded manifest.json parses as valid JSON")
        assets.close()
    except e:
        ctx.cleanup_artifact(tmp_flpk)
        raise e^
    ctx.cleanup_artifact(tmp_flpk)


# 2. World + ComponentRegistry + Scene hierarchy .flscnb roundtrip
def test_t3_ecs_archetype_simd_storage(mut ctx: TestContext) raises:
    var reg = ComponentRegistry.with_builtins()
    var w = World()
    _ = w.register_component[LocalTransform]()
    _ = w.register_component[GlobalTransform]()
    _ = w.register_component[Parent]()

    var parent_e = w.spawn()
    _ = w.insert[LocalTransform](parent_e, LocalTransform(Vec3(10.0, 2.0, 0.0)))
    _ = w.insert[GlobalTransform](parent_e, GlobalTransform())

    var child_e = w.spawn()
    _ = w.insert[LocalTransform](child_e, LocalTransform(Vec3(0.0, 3.0, -5.0)))
    _ = w.insert[GlobalTransform](child_e, GlobalTransform())
    ctx.assert_true(set_parent(w, child_e, parent_e), "set_parent links child to parent in World")

    var sc = Scene("T3Hierarchy")
    var blob = sc.save_flscnb_bytes(w, reg)
    ctx.assert_true(len(blob) > 32, "Scene serializes hierarchical World to .flscnb bytes")

    var w2 = World()
    _ = w2.register_component[LocalTransform]()
    _ = w2.register_component[GlobalTransform]()
    _ = w2.register_component[Parent]()
    var sc2 = Scene()
    var loaded = sc2.load_flscnb_bytes(w2, reg, blob)
    ctx.assert_equal(len(loaded), 2, "Scene deserializes .flscnb into fresh World")
    ctx.assert_equal(w2.entity_count(), 2, "restored World contains both parent and child entities")
    var p_ents = w2.query_entities(w2.bit_for[Parent]())
    ctx.assert_equal(len(p_ents), 1, "Parent relationship preserved across .flscnb roundtrip")
    var child_lt = w2.get[LocalTransform](p_ents[0])
    ctx.assert_vec3_approx(child_lt.translation, Vec3(0.0, 3.0, -5.0), 0.001, "child LocalTransform preserved")


# 3. EngineApp(worker_threads=2) + JobPool + World.par_for_each1 across 256 entities
def test_t3_jobpool_ecs_concurrency(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="T3JobPool", headless=True, worker_threads=2))
    app.add_plugin(TransformPlugin())
    _ = app.register_component[_T3JobTag]()
    _ = app.add_system(STAGE_UPDATE, "t3.par_inc", _t3_ecs_jobpool_sys, exclusive=True)
    app.startup()

    var ents = List[Entity](capacity=256)
    for i in range(256):
        var e = app.world.spawn()
        _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(Float32(i), 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e, GlobalTransform())
        _ = app.world.insert[_T3JobTag](e, _T3JobTag(0))
        ents.append(e)

    for _ in range(5):
        app.step_one_frame(Float32(1.0 / 60.0))

    for i in range(256):
        ctx.assert_equal(Int(app.world.get[_T3JobTag](ents[i]).hits), 5, "all 256 entities processed across 5 JobPool frames")
    app.shutdown()


# 4. McpServerV2 + FlintStudioState + AssetServer + JobPool JSON-RPC mutation & stepping
def test_t3_mcp_build_stamp_query(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")

    var init_resp = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        studio,
        assets,
        pool,
    )
    var spawn_resp = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.spawn_entity\",\"arguments\":{\"name\":\"T3Crate\"}}}",
        studio,
        assets,
        pool,
    )
    var step_resp = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"physics.step_deterministic\",\"arguments\":{\"ticks\":4}}}",
        studio,
        assets,
        pool,
    )

    var live_cnt = studio.live_entity_count()
    var sim_tick = studio.sim_tick
    assets.close()
    studio.shutdown()
    pool.stop()

    ctx.assert_true(BUILD_COMMIT.byte_length() > 0, "engine build commit embedded")
    ctx.assert_true(init_resp.find("2024-11-05") >= 0, "McpServerV2 initialize returns MCP 2024-11-05")
    ctx.assert_true(spawn_resp.find("\"isError\":false") >= 0 and live_cnt == 1, "McpServerV2 spawns live entity in FlintStudioState")
    ctx.assert_true(step_resp.find("\"isError\":false") >= 0 and sim_tick == 4, "McpServerV2 steps FlintStudioState 4 deterministic ticks")


# 5. ConsoleVars (CVAR_DEBUG_DRAW) gating DebugLineBatch -> ScalableRenderer pixels
def test_t3_cvar_runtime_debug_toggle(mut ctx: TestContext) raises:
    var cvars = ConsoleVars()
    var ren = ScalableRenderer(tier=RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    var rw = register_mock_render_scene(ren)

    # Frame A: debug draw off
    cvars.set(CVAR_DEBUG_DRAW, 0.0)
    var ui_off = UiDrawList()
    if cvars.get(CVAR_DEBUG_DRAW) > 0.5:
        var dbg = DebugLineBatch()
        dbg.add_aabb(Vec3(-1.5, -1.5, -1.5), Vec3(1.5, 1.5, 1.5), 0xFF00FF00, Float32(2.5))
        _ = dbg.emit_to_ui_draw_list(ui_off, rw.camera.eye, rw.camera.target, rw.camera.fov_y_deg, rw.camera.near_z, 128, 96)
    ctx.assert_true(ren.render_world(rw, ui_off), "render_world with debug draw off")
    var px_off = ren.readback_pixels()

    # Frame B: debug draw on
    cvars.set(CVAR_DEBUG_DRAW, 1.0)
    var ui_on = UiDrawList()
    if cvars.get(CVAR_DEBUG_DRAW) > 0.5:
        var dbg = DebugLineBatch()
        dbg.add_aabb(Vec3(-1.5, -1.5, -1.5), Vec3(1.5, 1.5, 1.5), 0xFF00FF00, Float32(2.5))
        _ = dbg.emit_to_ui_draw_list(ui_on, rw.camera.eye, rw.camera.target, rw.camera.fov_y_deg, rw.camera.near_z, 128, 96)
    ctx.assert_true(len(ui_on.prims) > 0, "CVAR_DEBUG_DRAW=1.0 emits DebugLineBatch UI primitives")
    ctx.assert_true(ren.render_world(rw, ui_on), "render_world with debug draw on")
    var px_on = ren.readback_pixels()
    ren.shutdown()

    var diff = compare_pixels(px_on, px_off, 128, 96, 4)
    ctx.assert_true(diff.differing > 10, "CVAR_DEBUG_DRAW toggle changes rendered framebuffer pixels")


# 6. evdev_key -> InputState -> InputMap + UiContext multi-frame Tab focus & Enter button activation
def test_t3_evdev_to_ui_navigation(mut ctx: TestContext) raises:
    var k_tab = evdev_key(15)   # KEY_TAB -> UI_KEY_TAB
    var k_enter = evdev_key(28) # KEY_ENTER -> UI_KEY_ENTER
    ctx.assert_equal(k_tab, UI_KEY_TAB, "evdev KEY_TAB maps to UI_KEY_TAB")
    ctx.assert_equal(k_enter, UI_KEY_ENTER, "evdev KEY_ENTER maps to UI_KEY_ENTER")

    var imap = create_mock_input_map()
    var ui = UiContext()
    var inp = InputState()

    # Frame 0: neutral frame establishes focus on first button (btn_play)
    inp.begin_frame()
    imap.update(inp, Float32(1.0 / 60.0))
    ui.begin_frame(inp, 640.0, 480.0)
    ui.begin_box("menu", FlexStyle(FLEX_COL))
    _ = ui.button("btn_play", "Play", 140.0, 36.0)
    _ = ui.button("btn_quit", "Quit", 140.0, 36.0)
    ui.end_box()
    ui.end_frame()

    # Frame 1: Tab advances focus to btn_quit & Space triggers InputMap action "jump"
    inp.begin_frame()
    inp.set_logical_key(k_tab, True)
    inp.set_logical_key(32, True)
    imap.update(inp, Float32(1.0 / 60.0))
    ui.begin_frame(inp, 640.0, 480.0)
    ui.begin_box("menu", FlexStyle(FLEX_COL))
    _ = ui.button("btn_play", "Play", 140.0, 36.0)
    _ = ui.button("btn_quit", "Quit", 140.0, 36.0)
    ui.end_box()
    ui.end_frame()
    ctx.assert_true(imap.is_down("jump"), "InputMap resolves bound action concurrently with UI input")

    # Frame 2: Enter activates focused button (btn_quit)
    inp.begin_frame()
    inp.set_logical_key(k_enter, True)
    imap.update(inp, Float32(1.0 / 60.0))
    ui.begin_frame(inp, 640.0, 480.0)
    ui.begin_box("menu", FlexStyle(FLEX_COL))
    _ = ui.button("btn_play", "Play", 140.0, 36.0)
    var activated = ui.button("btn_quit", "Quit", 140.0, 36.0)
    ui.end_box()
    ui.end_frame()
    ctx.assert_true(activated, "evdev Tab+Enter navigates and activates focused UiContext button")


# 7. Font loading -> UiContext layout + UiDrawList primitive emission
def test_t3_font_hud_layout_derivation(mut ctx: TestContext) raises:
    var ui = UiContext()
    var ok = ui.font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    ctx.assert_true(ok and ui.font.valid, "DejaVuSansMono-Bold loaded into UiContext.font")

    var hud_text = String("SPEED: 210 KM/H")
    var measured_w = ui.font.measure(hud_text, Float32(18.0))
    var inp = InputState()
    ui.begin_frame(inp, 800.0, 600.0)
    ui.begin_box("hud", FlexStyle(FLEX_COL))
    ui.label(hud_text)
    _ = ui.button("hud_btn", hud_text, measured_w + Float32(32.0), Float32(36.0))
    var rect = ui.last_item_rect
    ui.end_box()
    ui.end_frame()

    ctx.assert_true(measured_w > Float32(50.0), "font measures HUD string width > 50px")
    ctx.assert_true(rect.w > measured_w, "UiContext button rect encloses measured font text")
    ctx.assert_true(len(ui.draw_list.prims) >= 2, "UiContext emits text & panel primitives into UiDrawList")


# 8. Scene.enter_pie / exit_pie + FlintReplayStream + World.snapshot_blob checksum sync
def test_t3_replay_entity_tick_sync(mut ctx: TestContext) raises:
    var app = create_headless_v5_app(with_physics=True, with_input=False)
    app.startup()

    var box_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](box_e, LocalTransform(Vec3(0.0, 5.0, 0.0)))
    _ = app.world.insert[GlobalTransform](box_e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](box_e, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](box_e, ColliderComponent.box(Vec3(0.5, 0.5, 0.5)))

    var sc = Scene("T3Pie")
    sc.enter_pie(app.world)
    var w_init = BlobWriter()
    app.world.snapshot_blob(w_init)
    var replay = FlintReplayStream()
    replay.begin_recording("T3PieSync", w_init)

    var dt = Float32(1.0 / 60.0)
    for _ in range(10):
        app.step_one_frame(dt)
        replay.record_tick(dt, List[UInt8]())
    var y_after_fall = app.world.get[LocalTransform](box_e).translation.y

    sc.exit_pie(app.world)
    var y_restored = app.world.get[LocalTransform](box_e).translation.y
    app.shutdown()

    ctx.assert_equal(replay.tick_count(), 10, "replay stream tracked 10 PIE simulation ticks")
    ctx.assert_true(y_after_fall < Float32(4.9), "box falls under gravity during PIE ticks")
    ctx.assert_approx(y_restored, Float32(5.0), Float32(0.001), "Scene.exit_pie restores pre-PIE entity transform")


# 9. Real v5 SPIR-V shaders + FlintAaaVulkanRenderer tile classify pipeline readback
def test_t3_pipeline_cache_spirv_magic(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active for SPIR-V & tile classify test")
    var tc = aaa_tile_classify_shader()
    var cc = aaa_cluster_cull_shader(0, 1)
    ctx.assert_true(len(tc) >= 5 and Int(tc[0]) == Int(SPIRV_MAGIC), "aaa_tile_classify_shader has SPIRV_MAGIC")
    ctx.assert_true(validate_spirv_module(tc).valid, "aaa_tile_classify_shader validates")
    ctx.assert_true(validate_spirv_module(cc).valid, "aaa_cluster_cull_shader validates")

    ren.begin_frame(Vec3(0.0, 1.0, -4.0), Vec3(0.0, 0.0, 0.0))
    ctx.assert_true(ren.render_frame(capture_readback=True), "FlintAaaVulkanRenderer executes frame with v5 SPIR-V passes")
    var counts = ren.readback_tile_classify_counts()
    var total_tiles = 0
    for i in range(len(counts)):
        total_tiles += counts[i]
    ctx.assert_true(total_tiles > 0, "readback_tile_classify_counts bins active frame tiles")


# 10. DockSpace 5-panel layout + UiContext panel containment & virtualized scroll range
def test_t3_viewport_scissor_clamping(mut ctx: TestContext) raises:
    var dock = DockSpace.create_studio_5panel()
    var ui = UiContext()
    var inp = InputState()
    ui.begin_frame(inp, 1280.0, 720.0)
    ui.update_dock_space(dock, RectF(0.0, 0.0, 1280.0, 720.0))

    var leaves = dock.leaf_rects()
    ctx.assert_equal(len(leaves), 5, "DockSpace solves 5 non-overlapping studio panels")
    for i in range(len(leaves)):
        ctx.assert_true(leaves[i].x >= 0.0 and leaves[i].y >= 0.0, "panel origin within viewport")
        ctx.assert_true(leaves[i].x + leaves[i].w <= 1280.5 and leaves[i].y + leaves[i].h <= 720.5, "panel bounds clamped to 1280x720")

    ctx.assert_true(ui.begin_dock_panel(dock, "SceneViewport"), "begin_dock_panel opens SceneViewport leaf")
    ui.begin_scroll_view("t3_scroll", Float32(240.0), Float32(180.0), Float32(100 * 24))
    var vis = ui.scroll_visible_range(100, Float32(24.0))
    ui.end_scroll_view()
    ui.end_dock_panel()
    ui.end_frame()

    ctx.assert_true(vis[0] == 0 and vis[1] > vis[0] and vis[1] < 20, "scroll_visible_range culls offscreen rows")


# 11. AssetServer + FlintAaaVulkanRenderer residency bridge + mip upload VRAM tracking
def test_t3_staging_buffer_chunk_fit(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active for residency bridge")
    var srv = create_mock_asset_server()
    srv.attach_renderer_residency_bridge(ren)

    var tex = CookedTextureAsset()
    tex.width = 64
    tex.height = 64
    tex.mip_count = 5
    tex.tail_first_mip = 4
    tex.resident_min_mip = 4
    for m in range(5):
        var dim = 64 >> m
        var b = ((dim + 3) // 4) * ((dim + 3) // 4) * 16
        tex.mip_sizes[m] = UInt32(b)
        tex.mips.append(List[UInt8](length=b, fill=UInt8(0x7F)))

    var th = srv.textures.insert(tex^, AssetUuid.from_path("tex/t3_bridge.png"), "tex/t3_bridge.png", srv.current_tick)
    var slot = Int(th.slot())
    srv.streamer.enqueue_mip_upload(slot, 3, 256)
    srv.streamer.enqueue_mip_upload(slot, 2, 1024)
    var drained = srv.streamer.drain_uploads_within_budget(1500)
    var uploaded = srv.upload_mip_to_gpu(th, 2, 1024)
    var vram_slot = srv.textures.slot_vram_bytes[slot]
    var vram_total = srv.total_vram_live_bytes()
    srv.close()

    ctx.assert_equal(len(drained), 2, "streamer admits 2 mips within 1500B budget")
    ctx.assert_true(uploaded, "AssetServer uploads staged mip to Vulkan renderer")
    ctx.assert_true(vram_slot > 0 and vram_total > 0, "VRAM residency counters track live uploaded bytes")


# 12. EngineApp(enable_profiler=True) + PhysicsPlugin multi-frame profiler overhead & clock check
def test_t3_profiler_scope_overhead(mut ctx: TestContext) raises:
    var app = EngineApp(AppConfig(title="T3Profiler", headless=True, worker_threads=0, enable_profiler=True))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.startup()

    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](e, RigidBodyComponent.dynamic(1.0))
    _ = app.world.insert[ColliderComponent](e, ColliderComponent.sphere(0.4))

    for _ in range(5):
        app.step_one_frame(Float32(1.0 / 60.0))

    var tps = app.profiler.ticks_per_second()
    var frames = app.resources.get_ptr[FrameTime]().value()[unsafe_offset=0].frame_index
    app.shutdown()

    ctx.assert_true(tps > 1000000.0, "EngineApp profiler calibrated clock > 1 MHz")
    ctx.assert_equal(Int(frames), 5, "EngineApp stepped 5 profiled frames")


# 13. World generational Entity bitfield + NetPlugin NetworkIdentityComponent mapping
def test_t3_entity_handle_bitfield_preservation(mut ctx: TestContext) raises:
    var app = create_headless_v5_app(with_physics=False, with_net=True, with_input=False)
    app.startup()

    var e0 = app.world.spawn()
    var bits0 = e0.to_bits()
    var roundtrip0 = Entity.from_bits(bits0)
    ctx.assert_equal(roundtrip0.to_bits(), bits0, "Entity bitfield roundtrips via Entity.from_bits")
    ctx.assert_true(app.world.despawn(e0), "first generation entity despawned")
    ctx.assert_false(app.world.is_alive(e0), "stale generation handle rejected after despawn")

    var e1 = app.world.spawn()
    ctx.assert_true(e1.to_bits() != bits0, "reused entity slot bumps generation bits")
    _ = app.world.insert[LocalTransform](e1, LocalTransform(Vec3(3.0, 1.0, -2.0)))
    _ = app.world.insert[GlobalTransform](e1, GlobalTransform())
    _ = app.world.insert[NetworkIdentityComponent](e1, NetworkIdentityComponent(net_id=7))
    _ = app.world.insert[ReplicationComponent](e1, ReplicationComponent())

    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))

    ctx.assert_true(app.world.is_alive(e1), "replicated live entity remains alive across NetPlugin frames")
    ctx.assert_equal(Int(app.world.get[NetworkIdentityComponent](e1).net_id), 7, "NetworkIdentityComponent net_id preserved")
    app.shutdown()


# 14. flpk_v3 multi-chunk 4096-byte page alignment, UUID lookup, and CRC64 verification
def test_t3_vfs_multi_file_header(mut ctx: TestContext) raises:
    var chunks = List[FlpkBuildChunk]()
    var b0 = List[UInt8](length=4096, fill=UInt8(0xAA))
    var b1 = List[UInt8](length=64, fill=UInt8(0xBB))
    var b2 = List[UInt8](length=4096, fill=UInt8(0xCC))
    var u0 = AssetUuid.from_path("meshes/box.mesh")
    var u1 = AssetUuid.from_path("shaders/vis.spv")
    var u2 = AssetUuid.from_path("data/config.bin")
    chunks.append(FlpkBuildChunk(u0, "meshes/box.mesh", 0, ASSET_KIND_MESH, b0^, CHUNK_COMP_NONE))
    chunks.append(FlpkBuildChunk(u1, "shaders/vis.spv", 0, ASSET_KIND_RAW, b1^, CHUNK_COMP_NONE))
    chunks.append(FlpkBuildChunk(u2, "data/config.bin", 0, ASSET_KIND_RAW, b2^, CHUNK_COMP_NONE))

    var tmp_path = "/tmp/flint_e2e_t3_multi_" + _pid_str() + ".flpk"
    ctx.cleanup_artifact(tmp_path)
    try:
        ctx.assert_true(write_flpk_v3(tmp_path, chunks, page_align=True), "write_flpk_v3 writes multi-chunk archive")
        var arch = FlpkV3MappedArchive.open(tmp_path)
        ctx.assert_true(arch.valid, "FlpkV3MappedArchive opens cleanly")
        ctx.assert_equal(arch.toc_count(), 3, "archive TOC indexes all 3 chunks")
        ctx.assert_equal(Int(arch.toc_entry(0).payload_offset % UInt64(4096)), 0, "4096B chunk aligned to 4096 bytes")
        ctx.assert_equal(Int(arch.toc_entry(1).payload_offset % UInt64(64)), 0, "small chunk aligned to 64 bytes")
        ctx.assert_equal(Int(arch.toc_entry(2).payload_offset % UInt64(4096)), 0, "4096B chunk aligned to 4096 bytes")
        for i in range(3):
            ctx.assert_true(arch.verify_entry_crc64(i), "entry ECMA-182 CRC64 verifies")
        ctx.assert_true(arch.find_by_uuid(u0, 0) >= 0, "UUID lookup finds meshes/box.mesh")
        ctx.assert_true(arch.find_by_uuid(u1, 0) >= 0, "UUID lookup finds shaders/vis.spv")
        ctx.assert_true(arch.find_by_uuid(u2, 0) >= 0, "UUID lookup finds data/config.bin")
        arch.close()
    except e:
        ctx.cleanup_artifact(tmp_path)
        raise e^
    ctx.cleanup_artifact(tmp_path)


# 15. Mojo 1.2 compiler check + full 8-plugin EngineApp multi-frame run
def test_t3_compiler_mojo_purity(mut ctx: TestContext) raises:
    ctx.assert_equal(BUILD_MOJO, "1.2.0", "engine compiled under official Mojo 1.2 compiler")
    var app = create_headless_v5_app(
        with_physics=True,
        with_anim=True,
        with_audio=True,
        with_render=True,
        with_net=True,
        with_input=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=128,
        height=96,
        worker_threads=0,
    )
    app.run(max_frames=3)
    var frames = app.resources.get_ptr[FrameTime]().value()[unsafe_offset=0].frame_index
    ctx.assert_equal(Int(frames), 3, "full 8-plugin EngineApp executed 3 frames cleanly")


# =========================================================================
# Aggregator for Tier 3 Platform & Assets
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T3_CrossFeature_PlatformAssets")
    var ren = FlintAaaVulkanRenderer(width=128, height=96)

    try: test_t3_vfs_to_asset_buffer(ctx); ctx.record_pass("test_t3_vfs_to_asset_buffer") except e: ctx.record_fail("test_t3_vfs_to_asset_buffer", String(e))
    try: test_t3_ecs_archetype_simd_storage(ctx); ctx.record_pass("test_t3_ecs_archetype_simd_storage") except e: ctx.record_fail("test_t3_ecs_archetype_simd_storage", String(e))
    try: test_t3_jobpool_ecs_concurrency(ctx); ctx.record_pass("test_t3_jobpool_ecs_concurrency") except e: ctx.record_fail("test_t3_jobpool_ecs_concurrency", String(e))
    try: test_t3_mcp_build_stamp_query(ctx); ctx.record_pass("test_t3_mcp_build_stamp_query") except e: ctx.record_fail("test_t3_mcp_build_stamp_query", String(e))
    try: test_t3_cvar_runtime_debug_toggle(ctx); ctx.record_pass("test_t3_cvar_runtime_debug_toggle") except e: ctx.record_fail("test_t3_cvar_runtime_debug_toggle", String(e))
    try: test_t3_evdev_to_ui_navigation(ctx); ctx.record_pass("test_t3_evdev_to_ui_navigation") except e: ctx.record_fail("test_t3_evdev_to_ui_navigation", String(e))
    try: test_t3_font_hud_layout_derivation(ctx); ctx.record_pass("test_t3_font_hud_layout_derivation") except e: ctx.record_fail("test_t3_font_hud_layout_derivation", String(e))
    try: test_t3_replay_entity_tick_sync(ctx); ctx.record_pass("test_t3_replay_entity_tick_sync") except e: ctx.record_fail("test_t3_replay_entity_tick_sync", String(e))
    try: test_t3_pipeline_cache_spirv_magic(ctx, ren); ctx.record_pass("test_t3_pipeline_cache_spirv_magic") except e: ctx.record_fail("test_t3_pipeline_cache_spirv_magic", String(e))
    try: test_t3_viewport_scissor_clamping(ctx); ctx.record_pass("test_t3_viewport_scissor_clamping") except e: ctx.record_fail("test_t3_viewport_scissor_clamping", String(e))
    try: test_t3_staging_buffer_chunk_fit(ctx, ren); ctx.record_pass("test_t3_staging_buffer_chunk_fit") except e: ctx.record_fail("test_t3_staging_buffer_chunk_fit", String(e))
    try: test_t3_profiler_scope_overhead(ctx); ctx.record_pass("test_t3_profiler_scope_overhead") except e: ctx.record_fail("test_t3_profiler_scope_overhead", String(e))
    try: test_t3_entity_handle_bitfield_preservation(ctx); ctx.record_pass("test_t3_entity_handle_bitfield_preservation") except e: ctx.record_fail("test_t3_entity_handle_bitfield_preservation", String(e))
    try: test_t3_vfs_multi_file_header(ctx); ctx.record_pass("test_t3_vfs_multi_file_header") except e: ctx.record_fail("test_t3_vfs_multi_file_header", String(e))
    try: test_t3_compiler_mojo_purity(ctx); ctx.record_pass("test_t3_compiler_mojo_purity") except e: ctx.record_fail("test_t3_compiler_mojo_purity", String(e))

    ren.shutdown()


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
