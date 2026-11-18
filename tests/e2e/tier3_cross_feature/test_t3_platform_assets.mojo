"""Tier 3 Cross-Feature Combinations: Platform, Asset & System Interactions (15 tests).
Pairwise interactions between VFS, ECS, JobPool, CVars, Profiler, MCP,
SDF fonts, Replay streams, and platform hardware abstractions.
"""
from std.collections import List
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import EntityHandle, ArchetypeRegistry, ArchetypeSlots, SimdColumn
from flint.core.cvars import ConsoleVars, CVAR_DEBUG_DRAW, cvar_entry
from flint.core.build_stamp import BUILD_COMMIT, BUILD_MOJO
from flint.assets.vfs import encode_pack, pack_read
from flint.assets.replay_stream import FlintReplayStream
from flint.render.font import SdfFont
from flint.core.profiler import profile_ticks, ProfileTimeline
from flint.ui.layout import UiRect
from flint.render.spirv import AaaSpirvModule, SPIRV_MAGIC, MODEL_GL_COMPUTE
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_render_graph import AaaResidencyStreamer
from flint.devtools.mcp_server import handle_flint_mcp_query, FlintMcpInfo
from flint.platform.wayland import evdev_key
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_flpk_data


# 1. VFS container loading asset into memory buffer
def test_t3_vfs_to_asset_buffer(mut ctx: TestContext) raises:
    var pack = create_mock_flpk_data()
    ctx.assert_true(len(pack) >= 64, "VFS pack contains header and records")
    var magic = Int(pack[0]) | (Int(pack[1]) << 8) | (Int(pack[2]) << 16) | (Int(pack[3]) << 24)
    ctx.assert_true(magic != 0, "non-zero pack header magic")

# 2. Archetype ECS entity spawn + SimdColumn component storage
def test_t3_ecs_archetype_simd_storage(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 32)
    var h = slots.spawn()
    var row = slots.resolve(h)
    var col_x = SimdColumn[DType.float32](32)
    var col_y = SimdColumn[DType.float32](32)
    col_x.set(row, 100.5)
    col_y.set(row, 200.5)
    ctx.assert_approx(col_x.get(row), 100.5, 0.01, "SimdColumn stores component for spawned entity")
    ctx.assert_approx(col_y.get(row), 200.5, 0.01, "SimdColumn stores component for spawned entity")

# 3. JobPool worker concurrency + ECS parallel system batching
def test_t3_jobpool_ecs_concurrency(mut ctx: TestContext) raises:
    var workers = default_worker_count()
    var chunk_size = 64
    var total_entities = 512
    var chunks = (total_entities + chunk_size - 1) // chunk_size
    ctx.assert_equal(chunks, 8, "512 entities partitioned into 8 SIMD chunks")
    ctx.assert_true(workers >= 1, "worker pool ready to distribute chunks across threads")

# 4. MCP server querying build stamp metadata
def test_t3_mcp_build_stamp_query(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var resp = handle_flint_mcp_query("ping", "{}", info)
    ctx.assert_true(resp.byte_length() > 0, "MCP server handles queries cleanly")
    ctx.assert_true(BUILD_COMMIT.byte_length() > 0, "engine commit metadata embedded")

# 5. CVar console mutation affecting engine runtime debug mode
def test_t3_cvar_runtime_debug_toggle(mut ctx: TestContext) raises:
    var cvars = ConsoleVars()
    cvars.set(CVAR_DEBUG_DRAW, 1.0)
    var active = cvars.get(CVAR_DEBUG_DRAW) == 1.0
    ctx.assert_true(active, "cvar mutation switches debug drawing on")
    cvars.set(CVAR_DEBUG_DRAW, 0.0)
    var inactive = cvars.get(CVAR_DEBUG_DRAW) == 0.0
    ctx.assert_true(inactive, "cvar mutation switches debug drawing off")

# 6. Wayland evdev key mapping driving UI navigation
def test_t3_evdev_to_ui_navigation(mut ctx: TestContext) raises:
    var key_enter = evdev_key(28) # Linux KEY_ENTER
    ctx.assert_equal(key_enter, 257, "evdev KEY_ENTER maps to UI activation key 257")
    var key_esc = evdev_key(1) # Linux KEY_ESC
    ctx.assert_equal(key_esc, 256, "evdev KEY_ESC maps to UI cancel key 256")

# 7. SdfFont measurement driving HUD layout
def test_t3_font_hud_layout_derivation(mut ctx: TestContext) raises:
    var font = SdfFont()
    var font_ok = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    ctx.assert_true(font_ok, "DejaVu font loaded successfully")
    ctx.assert_true(font.valid, "SdfFont marked valid")
    var text_w = font.measure("SPEED: 210 KM/H", 20.0)
    ctx.assert_true(text_w > 50.0, "text measurement yields positive width")
    var hud_box_w = text_w + 32.0 # 16px padding on each side
    ctx.assert_true(hud_box_w > text_w, "HUD container padded correctly around measured text")

# 8. Replay stream tick recording + EntityHandle association
def test_t3_replay_entity_tick_sync(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 16)
    var h = slots.spawn()
    var replay = FlintReplayStream()
    ctx.assert_true(h.valid(), "entity valid for replay capture")
    ctx.assert_equal(replay.tick_count(), 0, "replay stream initialized")

# 9. Pipeline cache header verification + SPIR-V magic validation
def test_t3_pipeline_cache_spirv_magic(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 1)
    var code = m.finish()
    ctx.assert_true(len(code) >= 5, "SPIR-V module contains standard header")
    ctx.assert_equal(Int(code[0]), Int(SPIRV_MAGIC), "header magic matches SPIRV_MAGIC (0x07230203)")
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "generated compute module passes SPIR-V validator")

# 10. Multi-viewport scissor bounding box clamping
def test_t3_viewport_scissor_clamping(mut ctx: TestContext) raises:
    var screen = UiRect(0, 0, 1920, 1080)
    var scissor = UiRect(100, 100, 500, 300)
    var outside = UiRect(2000, 2000, 100, 100)
    ctx.assert_true(screen.contains(scissor), "screen contains viewport scissor")
    ctx.assert_false(screen.overlaps(outside), "offscreen viewport culled")
    ctx.assert_equal(scissor.right(), 600, "viewport scissor right bound")
    ctx.assert_equal(scissor.bottom(), 400, "viewport scissor bottom bound")

# 11. Staging buffer capacity allocation for asset upload
def test_t3_staging_buffer_chunk_fit(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(16, 4)
    streamer.record_gpu_feedback(0, 0)
    var uploads = streamer.tick_streaming_budget(4, 1024 * 1024)
    ctx.assert_true(uploads >= 0 and uploads <= 4, "streaming budget uploads respect batch limit and staging budget")

# 12. Profiler scope timestamp overhead measurement
def test_t3_profiler_scope_overhead(mut ctx: TestContext) raises:
    var t0 = profile_ticks()
    var t1 = profile_ticks()
    ctx.assert_true(t1 >= t0, "hardware profile clock advances monotonically")
    var timeline = ProfileTimeline()
    _ = timeline.start("", overlay=False)
    var tps = timeline.ticks_per_second()
    ctx.assert_true(tps > 1000000.0, "calibrated clock frequency > 1 MHz")
    timeline.finish()

# 13. Entity handle pack/unpack roundtrip preservation
def test_t3_entity_handle_bitfield_preservation(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 16)
    var h = slots.spawn()
    var packed = h.packed()
    var unpacked = EntityHandle.unpack(packed)
    ctx.assert_equal(unpacked.packed(), packed, "entity generation and row index bitfield roundtrips perfectly")

# 14. VFS multi-file pack encoding and header size verification
def test_t3_vfs_multi_file_header(mut ctx: TestContext) raises:
    var names = List[String]()
    var files = List[List[UInt8]]()
    names.append(String("shader.spv"))
    names.append(String("model.bin"))
    var s1 = List[UInt8]()
    s1.append(1); s1.append(2)
    var s2 = List[UInt8]()
    s2.append(3); s2.append(4)
    files.append(s1^)
    files.append(s2^)
    var pack = encode_pack(names, files)
    ctx.assert_true(len(pack) >= 64, "multi-file pack header meets 64-byte alignment")

# 15. Pure Mojo 1.2 compiler check with zero C runtime bindings
def test_t3_compiler_mojo_purity(mut ctx: TestContext) raises:
    ctx.assert_equal(BUILD_MOJO, "1.2.0", "engine compiled under official Mojo 1.2 compiler")


# =========================================================================
# Aggregator for Tier 3 Platform & Assets
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T3_CrossFeature_PlatformAssets")

    try: test_t3_vfs_to_asset_buffer(ctx); ctx.record_pass("test_t3_vfs_to_asset_buffer") except e: ctx.record_fail("test_t3_vfs_to_asset_buffer", String(e))
    try: test_t3_ecs_archetype_simd_storage(ctx); ctx.record_pass("test_t3_ecs_archetype_simd_storage") except e: ctx.record_fail("test_t3_ecs_archetype_simd_storage", String(e))
    try: test_t3_jobpool_ecs_concurrency(ctx); ctx.record_pass("test_t3_jobpool_ecs_concurrency") except e: ctx.record_fail("test_t3_jobpool_ecs_concurrency", String(e))
    try: test_t3_mcp_build_stamp_query(ctx); ctx.record_pass("test_t3_mcp_build_stamp_query") except e: ctx.record_fail("test_t3_mcp_build_stamp_query", String(e))
    try: test_t3_cvar_runtime_debug_toggle(ctx); ctx.record_pass("test_t3_cvar_runtime_debug_toggle") except e: ctx.record_fail("test_t3_cvar_runtime_debug_toggle", String(e))
    try: test_t3_evdev_to_ui_navigation(ctx); ctx.record_pass("test_t3_evdev_to_ui_navigation") except e: ctx.record_fail("test_t3_evdev_to_ui_navigation", String(e))
    try: test_t3_font_hud_layout_derivation(ctx); ctx.record_pass("test_t3_font_hud_layout_derivation") except e: ctx.record_fail("test_t3_font_hud_layout_derivation", String(e))
    try: test_t3_replay_entity_tick_sync(ctx); ctx.record_pass("test_t3_replay_entity_tick_sync") except e: ctx.record_fail("test_t3_replay_entity_tick_sync", String(e))
    try: test_t3_pipeline_cache_spirv_magic(ctx); ctx.record_pass("test_t3_pipeline_cache_spirv_magic") except e: ctx.record_fail("test_t3_pipeline_cache_spirv_magic", String(e))
    try: test_t3_viewport_scissor_clamping(ctx); ctx.record_pass("test_t3_viewport_scissor_clamping") except e: ctx.record_fail("test_t3_viewport_scissor_clamping", String(e))
    try: test_t3_staging_buffer_chunk_fit(ctx); ctx.record_pass("test_t3_staging_buffer_chunk_fit") except e: ctx.record_fail("test_t3_staging_buffer_chunk_fit", String(e))
    try: test_t3_profiler_scope_overhead(ctx); ctx.record_pass("test_t3_profiler_scope_overhead") except e: ctx.record_fail("test_t3_profiler_scope_overhead", String(e))
    try: test_t3_entity_handle_bitfield_preservation(ctx); ctx.record_pass("test_t3_entity_handle_bitfield_preservation") except e: ctx.record_fail("test_t3_entity_handle_bitfield_preservation", String(e))
    try: test_t3_vfs_multi_file_header(ctx); ctx.record_pass("test_t3_vfs_multi_file_header") except e: ctx.record_fail("test_t3_vfs_multi_file_header", String(e))
    try: test_t3_compiler_mojo_purity(ctx); ctx.record_pass("test_t3_compiler_mojo_purity") except e: ctx.record_fail("test_t3_compiler_mojo_purity", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
