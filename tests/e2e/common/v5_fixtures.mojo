"""Reusable Flint Engine v5 E2E test fixtures respecting Mojo 1.2 non-Copyable RAII."""
from std.collections import List
from std.math import sin
from flint.anim.plugin import AnimPlugin
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    CHUNK_COMP_NONE,
    AssetUuid,
    FlpkBuildChunk,
    encode_cooked_mesh_slab,
    write_flpk_v3,
)
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.server import AssetServer
from flint.audio.plugin import AudioPlugin
from flint.core.math import Mat4, Vec3
from flint.devtools.mcp_v2 import McpServerV2
from flint.input import ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin
from flint.net.plugin import NetPlugin
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.plugin import PhysicsPlugin
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import RenderPlugin
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderCameraPacket,
    RenderInstancePacket,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import ComponentRegistry, TransformPlugin
from flint.studio import FlintStudioState
from e2e.common.mock_data import create_mock_glb_bytes, create_mock_meshlet_mesh, create_mock_wav_bytes


def create_headless_v5_app(
    with_physics: Bool = True,
    with_anim: Bool = False,
    with_audio: Bool = False,
    with_render: Bool = False,
    with_net: Bool = False,
    with_input: Bool = True,
    render_tier: UInt32 = RENDER_TIER_0_HEADLESS_SIMD,
    width: Int = 160,
    height: Int = 120,
    worker_threads: Int = 0,
) -> EngineApp:
    var app = EngineApp(
        AppConfig(
            title="FlintE2EV5App",
            headless=True,
            worker_threads=worker_threads,
            fixed_hz=60.0,
        )
    )
    app.add_plugin(WindowPlugin(headless=True, width=width, height=height, title="FlintE2EV5App"))
    app.add_plugin(TransformPlugin())
    if with_input:
        app.add_plugin(InputPlugin(enable_evdev=False))
    if with_physics:
        app.add_plugin(PhysicsPlugin(gravity=9.81))
    if with_anim:
        app.add_plugin(AnimPlugin())
    if with_audio:
        app.add_plugin(AudioPlugin(headless=True))
    if with_render:
        app.add_plugin(RenderPlugin(tier=render_tier, width=width, height=height))
    if with_net:
        app.add_plugin(NetPlugin(max_entities=128, max_clients=4))
    return app^


def create_v5_component_registry() -> ComponentRegistry:
    return ComponentRegistry.with_builtins()


def create_mock_flpk_v3_archive_file(path: String) -> Bool:
    var pool = AaaGeometryPool()
    _ = pool.append_mesh(create_mock_meshlet_mesh(8), 0, build_lods=True)
    var mesh_slab = encode_cooked_mesh_slab(pool)

    var chunks = List[FlpkBuildChunk]()
    chunks.append(
        FlpkBuildChunk(
            AssetUuid.from_path("meshes/mock_sphere.mesh"),
            "meshes/mock_sphere.mesh",
            0,
            ASSET_KIND_MESH,
            mesh_slab^,
            CHUNK_COMP_NONE,
        )
    )
    chunks.append(
        FlpkBuildChunk(
            AssetUuid.from_path("model.glb"),
            "model.glb",
            0,
            ASSET_KIND_RAW,
            create_mock_glb_bytes(),
            CHUNK_COMP_NONE,
        )
    )
    chunks.append(
        FlpkBuildChunk(
            AssetUuid.from_path("audio/beep.wav"),
            "audio/beep.wav",
            0,
            ASSET_KIND_RAW,
            create_mock_wav_bytes(),
            CHUNK_COMP_NONE,
        )
    )
    var json_bytes = List[UInt8]()
    for b in String("{\"fixture\":\"v5_flpk_v3\"}").as_bytes():
        json_bytes.append(b)
    chunks.append(
        FlpkBuildChunk(
            AssetUuid.from_path("manifest.json"),
            "manifest.json",
            0,
            ASSET_KIND_RAW,
            json_bytes^,
            CHUNK_COMP_NONE,
        )
    )
    return write_flpk_v3(path, chunks, page_align=True)


def create_mock_pcm16_sine(freq_hz: Float64 = 440.0, frames: Int = 1024, sample_rate: Int = 48000) -> List[Int16]:
    var out = List[Int16](capacity=frames)
    var tau = 6.283185307179586
    for f in range(frames):
        var s = 0.5 * sin(tau * freq_hz * Float64(f) / Float64(sample_rate))
        out.append(Int16(Int(s * 32767.0)))
    return out^


def create_mock_bvh8_floor(half_extent: Float32 = 25.0, y: Float32 = 0.0) -> Bvh8MeshCollider:
    var verts = List[Vec3]()
    verts.append(Vec3(-half_extent, y, -half_extent))
    verts.append(Vec3(half_extent, y, -half_extent))
    verts.append(Vec3(half_extent, y, half_extent))
    verts.append(Vec3(-half_extent, y, half_extent))
    var idxs: List[Int32] = [Int32(0), Int32(2), Int32(1), Int32(0), Int32(3), Int32(2)]
    return Bvh8MeshCollider.build(verts, idxs)


def create_mock_input_map() -> InputMap:
    var imap = InputMap()
    _ = imap.register_action("jump", ACTION_BOOL)
    _ = imap.register_action("fire", ACTION_BOOL)
    var ctx = InputContext("gameplay", priority=0)
    ctx.bind_key("jump", 32)
    ctx.bind_key("fire", 70)
    imap.add_context(ctx^)
    imap.push_context("gameplay")
    return imap^


def create_mock_asset_server() -> AssetServer:
    return AssetServer(stream_budget_bytes_per_tick=1048576, lru_max_bytes=16777216, enable_watcher=False)


def create_mock_studio(headless: Bool = True, width: Int = 320, height: Int = 240) -> FlintStudioState:
    return FlintStudioState(headless=headless, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=width, height=height)


def create_mock_mcp_server(workspace_root: String = "/tmp") -> McpServerV2:
    return McpServerV2(workspace_root=workspace_root, auth_token="")


def register_mock_render_scene(mut ren: ScalableRenderer) -> RenderWorld:
    var mat = MaterialAsset(base_color=Vec3(0.85, 0.35, 0.2), roughness=0.35, metallic=0.25)
    var mat_id = ren.register_material(mat)
    var mesh_id = ren.register_mesh(create_mock_meshlet_mesh(10), mat_id)
    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(
        eye=Vec3(0.0, 2.0, 6.0),
        target=Vec3(0.0, 0.0, 0.0),
        fov_y_deg=Float32(60.0),
        near_z=Float32(0.1),
    )
    var id_m = Mat4()
    rw.instances.append(RenderInstancePacket(1, mesh_id, id_m, id_m, material_id=mat_id))
    rw.sun_dir = Vec3(0.4, 0.8, 0.4).normalized()
    rw.sun_radiance = Vec3(3.0, 2.88, 2.7)
    return rw^
