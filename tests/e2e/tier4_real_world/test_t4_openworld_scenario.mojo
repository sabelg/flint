"""Tier 4 Real-World Application Scenarios: S3 Open World Procedural Exploration.
Genuine multi-frame server + client `EngineApp` slice uniting `PhysicsPlugin` (`HeightfieldCollider`
+ `CharacterControllerComponent` at >2000m world coords), `RenderPlugin` (`RENDER_TIER_2_AAA_VISBUFFER_RT`
+ `TerrainClipmapDesc` + `ScalableRenderer`), `AssetServer` + `FlpkV3MappedArchive` mip residency,
and `NetPlugin` chunk-relative `VirtualNetSim` replication + `LagCompHistory`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, cos, sin
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin
from flint.assets.flpk_v3 import AssetUuid
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.server import CookedTextureAsset
from flint.core.math import Vec3
from flint.net import IPV6_SAFE_MTU_BYTES, NetHub, VirtualEndpoint, VirtualNetSim
from flint.net.plugin import (
    NET_MODE_VIRTUAL,
    NetConfigResource,
    NetPlugin,
    NetReplicationResource,
    NetTransportResource,
    NetworkIdentityComponent,
    ReplicationComponent,
)
from flint.physics.character import CharacterControllerConfig
from flint.physics.plugin import CharacterControllerComponent, PhysicsPlugin
from flint.physics.shapes_v5 import HeightfieldCollider
from flint.physics.world import PhysicsWorld
from flint.render.material_graph import MaterialAsset, TerrainClipmapDesc
from flint.render.plugin import (
    CameraComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
)
from flint.render.render_world import (
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    ScalableRenderer,
    TerrainClipmapPacket,
)
from flint.scene import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import create_mock_asset_server, create_mock_flpk_v3_archive_file


def _pid_str() -> String:
    return String(Int(external_call["getpid", Int32]()))


def _pixel_variance(px: List[UInt32]) -> Int:
    if len(px) == 0:
        return 0
    var p0 = px[0]
    var diff = 0
    for i in range(1, len(px)):
        if px[i] != p0:
            diff += 1
    return diff


def test_t4_openworld_full_exploration(mut ctx: TestContext) raises:
    var flpk_path = "/tmp/flint_e2e_t4_openworld_" + _pid_str() + ".flpk"
    ctx.cleanup_artifact(flpk_path)

    var sim = VirtualNetSim(
        seed=0x4F50454E574F524C,
        one_way_latency_ms=4.0,
        jitter_ms=0.0,
        loss_rate=0.0,
        dup_rate=0.0,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
    )
    var srv_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9500), True)
    var srv_ep = srv_hub.local_endpoint()
    var cli_hub = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9500), False)
    var cli_cid = cli_hub.connect(srv_ep)
    ctx.assert_equal(cli_cid, 0, "client hub connected to openworld server")
    ctx.assert_true(cli_hub.send_unreliable(cli_cid, List[UInt8](), sim.now_ms()), "initial handshake packet queued")

    var server_app = EngineApp(AppConfig(title="OpenWorldSrv", headless=True, worker_threads=0, fixed_hz=60.0))
    server_app.add_plugin(WindowPlugin(headless=True, width=160, height=120, title="OpenWorldSrv"))
    server_app.add_plugin(TransformPlugin())
    server_app.add_plugin(PhysicsPlugin(gravity=9.81))
    server_app.add_plugin(RenderPlugin(tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=160, height=120))
    server_app.add_plugin(NetPlugin(max_entities=128, max_clients=4, world_bounds=4096.0, chunk_size=64.0))
    server_app.startup()
    server_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    server_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(srv_hub^)

    var client_app = EngineApp(AppConfig(title="OpenWorldCli", headless=True, worker_threads=0, fixed_hz=60.0))
    client_app.add_plugin(WindowPlugin(headless=True))
    client_app.add_plugin(TransformPlugin())
    client_app.add_plugin(NetPlugin(max_entities=128, max_clients=4, world_bounds=4096.0, chunk_size=64.0))
    client_app.startup()
    client_app.resources.get_ptr_mut[NetConfigResource]().value()[unsafe_offset=0].mode = NET_MODE_VIRTUAL
    client_app.resources.get_ptr_mut[NetTransportResource]().value()[unsafe_offset=0].attach_virtual_hub(cli_hub^)

    var assets = create_mock_asset_server()
    try:
        # 1. Build 33x33 alpine heightfield centered past 2000m open-world coordinates
        var nx = 33
        var nz = 33
        var heights = List[Float32](capacity=nx * nz)
        for z in range(nz):
            for x in range(nx):
                var h = Float32(4.0) + Float32(1.5) * sin(Float32(x) * Float32(0.22)) * cos(Float32(z) * Float32(0.22))
                heights.append(h)

        var hf = HeightfieldCollider.build(heights, nx, nz, origin=Vec3(2048.0, 0.0, 2048.0), cell_size=4.0)
        ref pw = server_app.resources.get_ptr_mut[PhysicsWorld]().value()[unsafe_offset=0]
        _ = pw.add_heightfield(hf)
        pw.rebuild_broadphase()

        # 2. Configure Vulkan Tier 2 renderer + TerrainClipmapDesc + AssetServer VRAM bridge
        ref ren = server_app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
        ctx.assert_true(ren.uses_vulkan(), "OpenWorld uses Tier 2 headless Vulkan renderer")
        ren.gpu_slot[0].force_readback = True

        var mat_terrain = ren.register_material(MaterialAsset(base_color=Vec3(0.28, 0.45, 0.24), roughness=0.85))
        var mat_explorer = ren.register_material(MaterialAsset(base_color=Vec3(0.95, 0.45, 0.15), roughness=0.35, metallic=0.2))
        var mb = MeshBuilder()
        mb.add_beveled_box(Vec3(-0.6, -0.9, -0.6), Vec3(0.6, 0.9, 0.6), 0.1)
        var mesh_explorer = ren.register_mesh(mb^.finish(), mat_explorer)

        var clip_desc = TerrainClipmapDesc(ring_count=3, grid_dim_per_ring=32, base_cell_size_m=0.5, height_scale_m=12.0)
        var synced_desc = ren.set_terrain_clipmap(clip_desc, mat_terrain, heights, nx, nz)
        server_app.resources.get_ptr_mut[RenderPluginStateResource]().value()[unsafe_offset=0].terrain_clipmap = TerrainClipmapPacket(
            True, synced_desc^, mat_terrain
        )

        ctx.assert_true(create_mock_flpk_v3_archive_file(flpk_path), "write openworld .flpk v3 pack")
        ctx.assert_true(assets.mount_flpk(flpk_path), "AssetServer mounts openworld .flpk v3")
        assets.attach_renderer_residency_bridge(ren.gpu_slot[0])
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
            tex.mips.append(List[UInt8](length=b, fill=UInt8(0x66)))
        var th = assets.textures.insert(tex^, AssetUuid.from_path("terrain/grass.png"), "terrain/grass.png", assets.current_tick)
        assets.streamer.enqueue_mip_upload(Int(th.slot()), 3, 256)
        _ = assets.streamer.drain_uploads_within_budget(1024)
        ctx.assert_true(assets.upload_mip_to_gpu(th, 3, 256), "AssetServer uploads openworld mip to Vulkan")

        # 3. Spawn explorer entity at (2064, 12, 2064) on server + mirrored net_id=1 on client
        var start_pos = Vec3(2064.0, 12.0, 2064.0)
        var srv_e = server_app.world.spawn()
        _ = server_app.world.insert[LocalTransform](srv_e, LocalTransform(start_pos))
        _ = server_app.world.insert[GlobalTransform](srv_e, GlobalTransform())
        _ = server_app.world.insert[PrevGlobalTransform](srv_e, PrevGlobalTransform())
        _ = server_app.world.insert[CharacterControllerComponent](
            srv_e, CharacterControllerComponent(start_pos, CharacterControllerConfig(), Vec3(6.0, 0.0, 4.0))
        )
        _ = server_app.world.insert[MeshRendererComponent](srv_e, MeshRendererComponent(mesh_explorer, mat_explorer))
        _ = server_app.world.insert[PointLightComponent](srv_e, PointLightComponent(Vec3(35.0, 32.0, 28.0), 40.0, True))
        _ = server_app.world.insert[NetworkIdentityComponent](srv_e, NetworkIdentityComponent(net_id=1))
        _ = server_app.world.insert[ReplicationComponent](srv_e, ReplicationComponent(interest_radius=512.0, record_lag_comp=True))

        server_app.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0].replicator.set_client_view(
            0, start_pos, Float32(512.0)
        )

        var cam_e = server_app.world.spawn()
        _ = server_app.world.insert[LocalTransform](cam_e, LocalTransform(start_pos + Vec3(-6.0, 6.0, -8.0)))
        _ = server_app.world.insert[GlobalTransform](cam_e, GlobalTransform())
        _ = server_app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
        _ = server_app.world.insert[CameraComponent](
            cam_e, CameraComponent(fov_y_deg=60.0, active=True, look_target_offset=Vec3(6.0, -5.0, 8.0))
        )

        var cli_e = client_app.world.spawn()
        _ = client_app.world.insert[LocalTransform](cli_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
        _ = client_app.world.insert[GlobalTransform](cli_e, GlobalTransform())
        _ = client_app.world.insert[NetworkIdentityComponent](cli_e, NetworkIdentityComponent(net_id=1))
        _ = client_app.world.insert[ReplicationComponent](cli_e, ReplicationComponent())

        # 4. Multi-frame exploration + replication + rendering loop
        var dt = Float32(1.0 / 60.0)
        for frame in range(30):
            var cur_p = server_app.world.get[LocalTransform](srv_e).translation
            server_app.world.get_ptr_mut[LocalTransform](cam_e).value()[unsafe_offset=0].translation = cur_p + Vec3(-6.0, 6.0, -8.0)
            server_app.resources.get_ptr_mut[NetReplicationResource]().value()[unsafe_offset=0].replicator.set_client_view(
                0, cur_p, Float32(512.0)
            )
            server_app.step_one_frame(dt)
            sim.advance_ms(Float64(dt) * 1000.0)
            client_app.step_one_frame(dt)
            _ = assets.tick(Float64(frame) * 0.0167)

        var srv_pos = server_app.world.get[GlobalTransform](srv_e).translation
        var cli_pos = client_app.world.get[GlobalTransform](cli_e).translation
        var terrain_hit = server_app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].raycast(
            srv_pos + Vec3(0.0, 5.0, 0.0), Vec3(0.0, -1.0, 0.0), Float32(20.0)
        )
        ref rstate = server_app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
        var rframes = rstate.rendered_frames
        var rok = rstate.last_render_ok
        ref ren_end = server_app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
        var terrain_synced = ren_end.gpu_terrain_synced
        var px = ren_end.readback_pixels()
        var hdr = ren_end.gpu_slot[0].readback_resolved_hdr()
        var vram_live = assets.total_vram_live_bytes()
        var lag_ticks = server_app.resources.get_ptr[NetReplicationResource]().value()[unsafe_offset=0].lag_comp.tick_count()

        ctx.assert_true(srv_pos.x > Float32(2066.0) and srv_pos.z > Float32(2065.0), "explorer traverses HeightfieldCollider past 2064m")
        ctx.assert_true(terrain_hit.hit, "HeightfieldCollider responds to downward raycast under explorer")
        ctx.assert_true(terrain_synced, "TerrainClipmapDesc synced on Vulkan Tier 2 renderer")
        ctx.assert_true(rframes >= 30 and rok, "RenderPlugin rendered 30 open-world frames")
        ctx.assert_equal(len(px), 160 * 120, "Vulkan framebuffer readback populated")
        ctx.assert_true(_pixel_variance(px) > 20, "Vulkan open-world render produces non-uniform pixels")
        ctx.assert_equal(len(hdr), 160 * 120, "resolved HDR buffer readback populated")
        ctx.assert_true(vram_live > 0, "AssetServer streams terrain mips with live VRAM tracking")
        ctx.assert_true(abs(cli_pos.x - srv_pos.x) < Float32(0.15) and abs(cli_pos.z - srv_pos.z) < Float32(0.15), "chunk-relative NetPlugin replicates position beyond 2000m")
        ctx.assert_true(lag_ticks > 0, "NetPlugin records LagCompHistory across frames")
    except e:
        assets.close()
        client_app.shutdown()
        server_app.shutdown()
        ctx.cleanup_artifact(flpk_path)
        raise e^

    assets.close()
    client_app.shutdown()
    server_app.shutdown()
    ctx.cleanup_artifact(flpk_path)


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
