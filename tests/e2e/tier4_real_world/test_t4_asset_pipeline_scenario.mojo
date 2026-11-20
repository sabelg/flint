"""Tier 4 Real-World Application Scenarios: S4 Asset Pipeline & Studio Ingestion.
Genuine end-to-end ingestion slice: `GLB + PNG + WAV -> cooker_v2 -> .flpk v3 (FlpkV3MappedArchive)
-> AssetServer (JobPool async load + AaaResidencyStreamer VRAM bridge) -> FlintAaaVulkanRenderer
+ FlintStudioState + McpServerV2`.
"""
from std.collections import List
from std.ffi import external_call
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import FlpkV3MappedArchive
from flint.assets.image import write_png
from flint.assets.server import ASSET_STATE_READY, AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.devtools.mcp_v2 import McpServerV2
from flint.platform.os import make_dirs, write_bytes
from flint.render.material_graph import MaterialAsset
from flint.render.render_world import RENDER_TIER_2_AAA_VISBUFFER_RT, ScalableRenderer
from flint.studio import FlintStudioState
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_glb_bytes, create_mock_wav_bytes


def _pid_str() -> String:
    return String(Int(external_call["getpid", Int32]()))


def _cleanup_dir(path: String):
    var cmd = String("rm -rf ") + path
    _ = external_call["system", Int32](cmd.as_c_string_span())


def test_t4_asset_pipeline_full_ingestion(mut ctx: TestContext) raises:
    var tmp_root = "/tmp/flint_e2e_t4_assets_" + _pid_str()
    var src_dir = tmp_root + "/src"
    var cache_dir = tmp_root + "/cache"
    var flpk_path = tmp_root + "/bundle.flpk"
    _cleanup_dir(tmp_root)
    _ = make_dirs(src_dir)
    _ = make_dirs(cache_dir)

    var pool = JobPool()
    _ = pool.allocate()
    _ = pool.start(2)
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT,
        width=160,
        height=120,
        worker_threads=0,
    )
    var assets = AssetServer(stream_budget_bytes_per_tick=4096, lru_max_bytes=16 * 1024 * 1024, enable_watcher=False)

    try:
        # 1. Author real GLB, PNG, and WAV source files in temp workspace
        var glb_b = create_mock_glb_bytes()
        ctx.assert_true(write_bytes(src_dir + "/model.glb", glb_b), "write source model.glb")
        var rgba = List[UInt32](length=32 * 32, fill=0xFF3399EE)
        for y in range(32):
            for x in range(32):
                if ((x // 4) + (y // 4)) % 2 == 0:
                    rgba[y * 32 + x] = 0xFFEEAA22
        ctx.assert_true(write_png(src_dir + "/albedo.png", rgba, 32, 32), "write source albedo.png")
        var wav_b = create_mock_wav_bytes()
        ctx.assert_true(write_bytes(src_dir + "/beep.wav", wav_b), "write source beep.wav")

        # 2. Cook workspace across JobPool into .flpk v3
        var rep = cook_workspace(src_dir, flpk_path, pool, cache_dir=cache_dir, force_recook=True, page_align=True)
        ctx.assert_true(rep.ok, "cook_workspace succeeds")
        ctx.assert_true(rep.total_assets >= 2, "cook_workspace cooked >= 2 assets")
        ctx.assert_true(rep.archive_bytes >= 64, "cooked .flpk archive non-trivial")

        # 3. Verify FlpkV3MappedArchive alignment & ECMA-182 CRC64
        var arch = FlpkV3MappedArchive.open(flpk_path)
        ctx.assert_true(arch.valid, "FlpkV3MappedArchive opens cooked bundle")
        var toc_glb = arch.find_by_path("model.glb", 0)
        ctx.assert_true(toc_glb >= 0, "TOC locates cooked model.glb")
        ctx.assert_true(arch.verify_entry_crc64(toc_glb), "model.glb CRC64 verifies")
        ctx.assert_equal(Int(arch.toc_entry(toc_glb).payload_offset & 63), 0, "cooked payload 64-byte aligned")
        arch.close()

        # 4. Mount .flpk v3 in AssetServer & bridge residency to Vulkan Tier 2 renderer
        ctx.assert_true(assets.mount_flpk(flpk_path), "AssetServer mounts cooked .flpk v3")
        ref ren = studio.app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
        ctx.assert_true(ren.uses_vulkan(), "FlintStudioState runs Tier 2 Vulkan renderer")
        assets.attach_renderer_residency_bridge(ren.gpu_slot[0])

        var h_mesh = assets.load_mesh_async("model.glb")
        var h_tex = assets.load_texture_async("albedo.png", progressive=True)
        var done = assets.poll_async_loads(pool)
        ctx.assert_true(done >= 2, "JobPool completes async mesh + texture loads")
        ctx.assert_equal(assets.meshes.asset_state(h_mesh), ASSET_STATE_READY, "cooked mesh reaches ASSET_STATE_READY")
        ctx.assert_equal(assets.textures.asset_state(h_tex), ASSET_STATE_READY, "cooked texture reaches ASSET_STATE_READY")

        var mesh_ptr = assets.meshes.get_ptr(h_mesh)
        ctx.assert_true(Bool(mesh_ptr) and mesh_ptr.value()[unsafe_offset=0].has_collider, "cooked mesh includes BVH8 collider")
        ctx.assert_true(len(mesh_ptr.value()[unsafe_offset=0].geom.meshes) > 0, "cooked mesh includes geometry LOD slices")

        # Register loaded asset in renderer + spawn studio entity via McpServerV2
        var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.8, 0.6, 0.4), roughness=0.3))
        _ = mat_id
        var mcp = McpServerV2(workspace_root=tmp_root, auth_token="")
        var spawn_resp = mcp.handle_jsonrpc(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.spawn_entity\",\"arguments\":{\"name\":\"CookedProp\"}}}",
            studio,
            assets,
            pool,
        )
        ctx.assert_true(spawn_resp.find("\"isError\":false") >= 0, "McpServerV2 spawns studio entity")

        # 5. Multi-frame progressive mip streaming + PIE + studio render loop
        for f in range(6):
            _ = assets.tick(Float64(f) * 0.0167)
            _ = studio.pie_step(1)
            studio.render_editor_frame(160, 120)

        var tex_slot = Int(h_tex.slot())
        var min_mip = assets.textures.get_ptr(h_tex).value()[unsafe_offset=0].resident_min_mip
        var slot_vram = assets.textures.slot_vram_bytes[tex_slot]
        var total_vram = assets.total_vram_live_bytes()
        var gpu_frame = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].gpu_slot[0].frame_index

        ctx.assert_true(slot_vram > 0 and total_vram > 0, "AaaResidencyStreamer tracks non-zero GPU VRAM residency")
        ctx.assert_equal(min_mip, 0, "progressive texture drains to mip 0")
        ctx.assert_true(studio.live_entity_count() >= 1, "studio owns live spawned entity")
        ctx.assert_true(studio.sim_tick >= 6, "studio PIE stepped 6 frames")
        ctx.assert_true(gpu_frame >= 6, "Vulkan renderer executed >= 6 editor frames")
    except e:
        assets.close()
        studio.shutdown()
        pool.stop()
        _cleanup_dir(tmp_root)
        raise e^

    assets.close()
    studio.shutdown()
    pool.stop()
    _cleanup_dir(tmp_root)


# =========================================================================
# Aggregator for Tier 4 Asset Pipeline Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_AssetPipeline")
    try: test_t4_asset_pipeline_full_ingestion(ctx); ctx.record_pass("test_t4_asset_pipeline_full_ingestion") except e: ctx.record_fail("test_t4_asset_pipeline_full_ingestion", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
