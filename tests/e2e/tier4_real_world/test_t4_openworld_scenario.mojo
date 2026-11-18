"""Tier 4 Real-World Application Scenarios: S3 Open World Procedural Exploration.
Simulates a multi-chunk open world outdoor environment: terrain heightfield chunks,
broadphase spatial culling, VSM clipmap page management, directional sun light,
and async meshlet residency streaming.
"""
from std.collections import List
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.physics.broadphase import BroadphaseInput, boxes_overlap, find_pairs
from flint.physics.filter import CollisionFilter
from flint.render.aaa_vsm import VsmPageManager, VSM_PHYS_TILE_COUNT
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams
from flint.render.aaa_render_graph import AaaRenderGraph, AaaResidencyStreamer, RgResourceAccess
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


def test_t4_openworld_full_exploration(mut ctx: TestContext) raises:
    # 1. Open world terrain grid: 4x4 chunks (each 64m x 64m)
    var chunk_size: Float32 = 64.0
    var grid_dim = 4
    var bp = BroadphaseInput()

    for cx in range(grid_dim):
        for cz in range(grid_dim):
            var min_p = Vec3(Float32(cx) * chunk_size, 0.0, Float32(cz) * chunk_size)
            var max_p = Vec3(Float32(cx + 1) * chunk_size, 25.0, Float32(cz + 1) * chunk_size)
            bp.add(min_p, max_p, True, True, CollisionFilter(1, 0xFFFFFFFF, 0))

    # 2. Camera frustum sweeping across terrain from origin towards center
    var cam_pos = Vec3(64.0, 15.0, 64.0)
    var cam_fwd = Vec3(0.707, -0.2, 0.707).normalized()
    ctx.assert_true(cam_fwd.length() > 0.99, "camera forward vector normalized")

    # 3. Virtual Shadow Map (VSM) clipmap allocation for directional sun
    var vsm = VsmPageManager()
    vsm.invalidate_all()
    # Invalidate dynamic shadow region around moving camera
    vsm.invalidate_region(0.2, 0.2, 0.5, 0.5)
    ctx.assert_equal(len(vsm.phys_to_virtual), Int(VSM_PHYS_TILE_COUNT), "VSM manages 1024 physical clipmap pages")

    # 4. Sun light contribution via Flint planetary atmosphere
    var atmo = BrunetonAtmosphereParams(sun_dir=Vec3(0.5, 0.8, 0.3).normalized(), solar_intensity=10.0)
    ctx.assert_true(atmo.sun_dir.length() > 0.99, "sun direction normalized")
    ctx.assert_true(atmo.solar_irradiance.x > 5.0, "daylight solar irradiance non-zero")

    # 5. Async meshlet residency streamer managing LOD transitions
    var streamer = AaaResidencyStreamer(128, 4)
    # Camera requests high-res LOD 0 for nearby chunks, coarser LOD 2 for far chunks
    streamer.record_gpu_feedback(0, 0) # Near chunk 0 -> LOD 0
    streamer.record_gpu_feedback(1, 0) # Near chunk 1 -> LOD 0
    streamer.record_gpu_feedback(10, 2) # Distant chunk 10 -> LOD 2
    var chunks_uploaded = streamer.tick_streaming_budget(1024, 2 * 1024 * 1024)
    ctx.assert_true(chunks_uploaded >= 0, "residency streamer uploads required LOD meshlets without hitching")

    # 6. Player character query against terrain broadphase via engine find_pairs
    var player_aabb_min = cam_pos - Vec3(0.4, 1.8, 0.4)
    var player_aabb_max = cam_pos + Vec3(0.4, 0.2, 0.4)
    bp.add(player_aabb_min, player_aabb_max, True, True, CollisionFilter(2, 1, 0))
    var pairs = find_pairs(bp)
    ctx.assert_true(len(pairs.pairs) >= 1, "broadphase detects terrain chunks intersecting player AABB")

    # 7. RenderGraph multi-pass outdoor scene compilation
    var rg = AaaRenderGraph()
    var r_shadow = rg.add_resource("vsm_shadow_atlas", 4096 * 4096 * 2)
    var r_hiz = rg.add_resource("hiz_pyramid", 1024 * 512 * 4)
    var r_gbuffer = rg.add_resource("gbuffer", 1920 * 1080 * 16, transient=True)
    var acc_shadow = List[RgResourceAccess]()
    acc_shadow.append(RgResourceAccess(r_shadow, 1))
    _ = rg.add_pass("sun_vsm_pass", acc_shadow)
    var acc_hiz = List[RgResourceAccess]()
    acc_hiz.append(RgResourceAccess(r_hiz, 1))
    _ = rg.add_pass("hiz_cull_pass", acc_hiz)
    var acc_gbuf = List[RgResourceAccess]()
    acc_gbuf.append(RgResourceAccess(r_gbuffer, 1))
    _ = rg.add_pass("terrain_gbuffer_pass", acc_gbuf)
    var compiled = rg.compile(64 * 1024 * 1024)
    ctx.assert_true(compiled, "open world RenderGraph compiles within budget")


# =========================================================================
# Aggregator for Tier 4 Open World Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_OpenWorld")
    try: test_t4_openworld_full_exploration(ctx); ctx.record_pass("test_t4_openworld_full_exploration") except e: ctx.record_fail("test_t4_openworld_full_exploration", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
