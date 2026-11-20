"""Flint Engine v5 AAA Open-World Feature Sandbox (`examples/open_world/main.mojo`).

A `512m x 512m` art-directed Alpine Sanctuary ("Sunspire Valley") running on the unified
Engine v5 plugin spine (`EngineApp`, `WindowPlugin`, `TransformPlugin`, `InputPlugin`,
`PhysicsPlugin`, `AnimPlugin`, `AudioPlugin`, `RenderPlugin`, `flint.ecs.World`,
`flint.scene.Prefab`/`Scene`, and `flint.assets.flpk_v3` / `AssetServer`) at `1280 x 720`
Vulkan 1.3/1.4 hardware ray-traced quality:
  - **Multi-Biome Ridged-Multifractal Terrain (`16` Sectors, `QEM` `LOD 0..3` + `BLAS`)**:
    Stratified dark basalt cliffs, lush emerald alpine meadows, snow-crowned glacier peaks,
    carved viaduct stone bridge, `300` instanced subsurface-lit pine trees & `140` granite boulders.
  - **Bindless `BC7 / BC5 / BC6H` Materials + `.flpk` v3 + `AssetServer` + `Prefab`**:
    `1024x1024` biome splat, `256x256` PBR textures, and a classical 12-column domed
    `Sunspire Sanctum` packaged into `build/open_world_assets.flpk` via `flpk_v3` and
    streamed through `AssetServer.mount_flpk` + `load_raw` + `Prefab.instantiate`.
  - **4-Bone Dual-Quaternion Skinned `Aegis Sentinel` Knight + `AnimPlugin` + Live `BLAS` Refit**
  - **Parallel `flint.ecs.World` + `PhysicsPlugin` (`256` Swept-Wing Gliders & Lantern Wisps)**
    simulated with `World.par_for_each2`, `RigidBodyComponent`/`ColliderComponent`,
    `174` `PointLightComponent`s, and `18` Heitz `LTC` `AreaLightComponent`s (`192` lights).
  - **Bruneton Atmosphere + `256x256` `BC6H` Sky + 5-subframe TSR `--shots`**.

Usage:
  - Interactive (Wayland / X11): `bash tools/flint.sh world`
    Controls: `WASD` + `Q/E` fly, `Arrows` look, `Shift` boost, `1/2/3` Noon/Sunset/Night,
              `B` Bloom, `F` Bokeh DoF, `G` RT GI, `~` Console, `F2` GPU Profiler, `F3` Memory, `Esc` quit.
  - Headless 3-Vista Cinema Showcase & Benchmark: `bash tools/flint.sh world --shots`
"""
from std.math import cos, exp, floor, max, min, sin, sqrt
from std.sys import argv
from flint.anim.plugin import AnimAssetCache, AnimPlugin, SkeletonComponent
from flint.app import (
    AppConfig,
    EngineApp,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_LAST,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.flpk_v3 import (
    ASSET_KIND_RAW,
    AssetUuid,
    CHUNK_COMP_NONE,
    FlpkBuildChunk,
    SUB_ID_PRIMARY_OR_TAIL,
    write_flpk_v3,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.hdr_image import HdrImage
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeModel, NativeNode, NativeTexture
from flint.assets.server import AssetServer
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.audio.server import AudioServer, BUS_AMBIENCE
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.devtools.overlays import FlintDevOverlays
from flint.ecs import (
    Component,
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.input.input_map import ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin, InputStateResource
from flint.physics.plugin import ColliderComponent, PhysicsPlugin, RigidBodyComponent
from flint.physics.world import PhysicsWorld
from flint.platform.kms import on_virtual_console
from flint.platform.os import env_flag, env_value, get_time, make_dirs
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_ibl import bake_hdr_image_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709, AaaPhysicalCamera
from flint.render.aaa_renderer import AaaSkinnedMeshEntry, FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import DualQuat
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams, BrunetonSkyLuts
from flint.render.material_graph import (
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_TWO_SIDED_FOLIAGE,
    MAT_FLAG_DOUBLE_SIDED,
    MAT_FLAG_SUBSURFACE,
    MaterialAsset,
)
from flint.render.plugin import (
    CameraComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
    VatCrowdComponent,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderCameraPacket,
    RenderWorld,
    ScalableRenderer,
)
from flint.render.simd_canvas import SimdCanvas
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    Prefab,
    PrevGlobalTransform,
    Scene,
    TransformPlugin,
    interpolate_global_transform,
)

comptime WORLD_HALF_EXTENT: Float32 = 256.0
comptime CHUNK_GRID: Int = 4
comptime CHUNK_VERTS_SIDE: Int = 45
comptime ENTITY_COUNT: Int = 256
comptime LAKE_LEVEL: Float32 = -1.10


# ============================================================================
# 1. Deterministic Noise & Multi-Biome Ridged-Multifractal Alpine Heightfield
# ============================================================================

@always_inline
def _hash2d_01(ix: Int, iy: Int) -> Float32:
    var n = UInt32(((ix * 374761393) + (iy * 668265263)) & 0x7FFFFFFF)
    n = (n ^ (n >> 13)) * UInt32(1274126177)
    n = n ^ (n >> 16)
    return Float32(n & 0x00FFFFFF) / Float32(16777216.0)


@always_inline
def _vnoise2(x: Float32, y: Float32) -> Float32:
    var ix = Int(floor(x))
    var iy = Int(floor(y))
    var fx = x - Float32(ix)
    var fy = y - Float32(iy)
    var ux = fx * fx * (Float32(3.0) - Float32(2.0) * fx)
    var uy = fy * fy * (Float32(3.0) - Float32(2.0) * fy)
    var c00 = _hash2d_01(ix, iy)
    var c10 = _hash2d_01(ix + 1, iy)
    var c01 = _hash2d_01(ix, iy + 1)
    var c11 = _hash2d_01(ix + 1, iy + 1)
    var top = c00 + (c10 - c00) * ux
    var bot = c01 + (c11 - c01) * ux
    return top + (bot - top) * uy


@always_inline
def _value_noise_tileable(x: Float32, y: Float32, period: Int) -> Float32:
    var ix = Int(floor(x))
    var iy = Int(floor(y))
    var fx = x - Float32(ix)
    var fy = y - Float32(iy)
    var ux = fx * fx * (Float32(3.0) - Float32(2.0) * fx)
    var uy = fy * fy * (Float32(3.0) - Float32(2.0) * fy)

    var x0 = ((ix % period) + period) % period
    var y0 = ((iy % period) + period) % period
    var x1 = (x0 + 1) % period
    var y1 = (y0 + 1) % period

    var c00 = _hash2d_01(x0, y0)
    var c10 = _hash2d_01(x1, y0)
    var c01 = _hash2d_01(x0, y1)
    var c11 = _hash2d_01(x1, y1)
    var top = c00 + (c10 - c00) * ux
    var bot = c01 + (c11 - c01) * ux
    return top + (bot - top) * uy


@always_inline
def _ridged_fbm(wx: Float32, wz: Float32) -> Float32:
    """4-octave eroded ridged multifractal mountain synthesis."""
    var sum = Float32(0.0)
    var amp = Float32(0.55)
    var freq = Float32(0.0105)
    var weight = Float32(1.0)
    for _ in range(4):
        var n = Float32(1.0) - abs(
            _vnoise2(wx * freq + Float32(17.3), wz * freq - Float32(9.1)) * Float32(2.0)
            - Float32(1.0)
        )
        var signal = n * n * weight
        weight = max(Float32(0.25), min(Float32(1.0), signal * Float32(1.85)))
        sum += signal * amp
        freq *= Float32(2.07)
        amp *= Float32(0.48)
    return sum


@always_inline
def sample_valley_height(wx: Float32, wz: Float32) -> Float32:
    """Sculpted alpine glacial valley heightfield."""
    var dx_l = wx / Float32(46.0)
    var dz_l = (wz - Float32(12.0)) / Float32(38.0)
    var lake_r2 = dx_l * dx_l + dz_l * dz_l

    var dx_i = wx / Float32(10.5)
    var dz_i = (wz - Float32(24.0)) / Float32(10.5)
    var isl_r2 = dx_i * dx_i + dz_i * dz_i
    var island_mound = Float32(3.6) * exp(-isl_r2 * Float32(1.35))

    var dx_s = wx / Float32(16.0)
    var dz_s = (wz + Float32(28.0)) / Float32(14.0)
    var shore_mound = Float32(3.4) * exp(-(dx_s * dx_s + dz_s * dz_s) * Float32(0.9))

    var ridge = _ridged_fbm(wx, wz)
    var foothills = (_vnoise2(wx * Float32(0.025), wz * Float32(0.025)) - Float32(0.35)) * Float32(6.5)

    var valley_wall = max(Float32(0.0), lake_r2 - Float32(0.85))
    var mtn_mask = min(Float32(1.0), valley_wall * Float32(0.42))
    var far_dist = sqrt(wx * wx + (wz - Float32(10.0)) * (wz - Float32(10.0))) / WORLD_HALF_EXTENT
    var rim = max(Float32(0.0), far_dist - Float32(0.26))

    var pass_dx = (wx - Float32(22.0)) / Float32(48.0)
    var pass_cut = Float32(1.0) - Float32(0.46) * exp(-(pass_dx * pass_dx)) * (
        Float32(1.0) if wz > Float32(30.0) else Float32(0.0)
    )

    var mtn_h = (
        (ridge * Float32(52.0) + max(Float32(0.0), foothills)) * mtn_mask * pass_cut
        + rim * rim * Float32(44.0) * pass_cut
    )
    var basin_floor = Float32(-3.6) + Float32(2.4) * min(Float32(1.0), lake_r2 * Float32(0.65))
    return basin_floor + mtn_h + island_mound + shore_mound


@always_inline
def sample_valley_normal(wx: Float32, wz: Float32) -> Vec3:
    var eps = Float32(0.55)
    var hl = sample_valley_height(wx - eps, wz)
    var hr = sample_valley_height(wx + eps, wz)
    var hd = sample_valley_height(wx, wz - eps)
    var hu = sample_valley_height(wx, wz + eps)
    return Vec3(hl - hr, Float32(2.0) * eps, hd - hu).normalized()


@always_inline
def _smoothstep(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = max(Float32(0.0), min(Float32(1.0), (x - e0) / max(Float32(1e-5), e1 - e0)))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


def build_terrain_chunk_mesh(cx: Int, cz: Int) -> NativeMesh:
    """Builds one watertight `(CHUNK_VERTS_SIDE x CHUNK_VERTS_SIDE)` open-world terrain sector."""
    var chunk_span = (WORLD_HALF_EXTENT * Float32(2.0)) / Float32(CHUNK_GRID)
    var x0 = -WORLD_HALF_EXTENT + Float32(cx) * chunk_span
    var z0 = -WORLD_HALF_EXTENT + Float32(cz) * chunk_span
    var step = chunk_span / Float32(CHUNK_VERTS_SIDE - 1)
    var inv_world = Float32(1.0) / (WORLD_HALF_EXTENT * Float32(2.0))

    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.82, metallic=0.0)
    for iz in range(CHUNK_VERTS_SIDE):
        var wz = z0 + Float32(iz) * step
        for ix in range(CHUNK_VERTS_SIDE):
            var wx = x0 + Float32(ix) * step
            var wy = sample_valley_height(wx, wz)
            var nrm = sample_valley_normal(wx, wz)
            var u = max(Float32(0.004), min(Float32(0.996), (wx + WORLD_HALF_EXTENT) * inv_world))
            var v = max(Float32(0.004), min(Float32(0.996), (wz + WORLD_HALF_EXTENT) * inv_world))
            _ = mb.add_vertex(Vec3(wx, wy, wz), nrm, u, v)

    for iz in range(CHUNK_VERTS_SIDE - 1):
        for ix in range(CHUNK_VERTS_SIDE - 1):
            var i00 = iz * CHUNK_VERTS_SIDE + ix
            var i10 = i00 + 1
            var i01 = (iz + 1) * CHUNK_VERTS_SIDE + ix
            var i11 = i01 + 1
            mb.add_tri(i00, i01, i10)
            mb.add_tri(i10, i01, i11)

    return mb^.finish()


def make_world_biome_splat_1024() -> Tuple[List[UInt32], List[UInt32]]:
    """Bakes a seamless `1024 x 1024` continuous multi-biome world-space `BC7` Albedo + ORM map."""
    comptime DIM: Int = 1024
    var albedo = List[UInt32](length=DIM * DIM, fill=0xFFFFFFFF)
    var orm = List[UInt32](length=DIM * DIM, fill=0xFFFFFFFF)
    var span = WORLD_HALF_EXTENT * Float32(2.0)

    for y in range(DIM):
        var wz = -WORLD_HALF_EXTENT + (Float32(y) + Float32(0.5)) / Float32(DIM) * span
        for x in range(DIM):
            var wx = -WORLD_HALF_EXTENT + (Float32(x) + Float32(0.5)) / Float32(DIM) * span
            var wy = sample_valley_height(wx, wz)
            var wn = sample_valley_normal(wx, wz)

            var n_fine = _vnoise2(wx * Float32(0.38), wz * Float32(0.38))
            var n_med = _vnoise2(wx * Float32(0.11) + Float32(9.2), wz * Float32(0.11) - Float32(4.7))
            var n_strata = _vnoise2(wx * Float32(0.06), wy * Float32(0.45))

            var meadow_r = Float32(46.0) + n_med * Float32(34.0) + n_fine * Float32(16.0)
            var meadow_g = Float32(84.0) + n_med * Float32(44.0) + n_fine * Float32(18.0)
            var meadow_b = Float32(34.0) + n_med * Float32(18.0) + n_fine * Float32(10.0)
            var rough_f = Float32(212.0) - n_fine * Float32(24.0)
            var ao_f = Float32(190.0) + n_fine * Float32(60.0)

            var shore_w = Float32(1.0) - _smoothstep(
                LAKE_LEVEL - Float32(0.1),
                LAKE_LEVEL + Float32(1.05),
                wy + (n_med - Float32(0.5)) * Float32(0.4),
            )
            if shore_w > Float32(0.001):
                var wet = Float32(1.0) - _smoothstep(LAKE_LEVEL - Float32(0.3), LAKE_LEVEL + Float32(0.35), wy)
                var sr = (Float32(118.0) + n_fine * Float32(28.0)) * (Float32(1.0) - wet * Float32(0.52))
                var sg = (Float32(108.0) + n_fine * Float32(26.0)) * (Float32(1.0) - wet * Float32(0.48))
                var sb = (Float32(92.0) + n_fine * Float32(24.0)) * (Float32(1.0) - wet * Float32(0.42))
                meadow_r = meadow_r * (Float32(1.0) - shore_w) + sr * shore_w
                meadow_g = meadow_g * (Float32(1.0) - shore_w) + sg * shore_w
                meadow_b = meadow_b * (Float32(1.0) - shore_w) + sb * shore_w
                rough_f = rough_f * (Float32(1.0) - shore_w) + (Float32(165.0) - wet * Float32(95.0)) * shore_w

            var cliff_w = Float32(1.0) - _smoothstep(
                Float32(0.68), Float32(0.84), wn.y + (n_med - Float32(0.5)) * Float32(0.08)
            )
            if cliff_w > Float32(0.001):
                var rock_sh = Float32(58.0) + n_strata * Float32(48.0) + n_fine * Float32(24.0)
                var rr = rock_sh * Float32(1.03)
                var rg = rock_sh * Float32(1.00)
                var rb = rock_sh * Float32(0.98) + Float32(5.0)
                meadow_r = meadow_r * (Float32(1.0) - cliff_w) + rr * cliff_w
                meadow_g = meadow_g * (Float32(1.0) - cliff_w) + rg * cliff_w
                meadow_b = meadow_b * (Float32(1.0) - cliff_w) + rb * cliff_w
                rough_f = rough_f * (Float32(1.0) - cliff_w) + Float32(178.0) * cliff_w
                ao_f = ao_f * (Float32(1.0) - cliff_w) + (Float32(135.0) + n_strata * Float32(110.0)) * cliff_w

            var snow_alt = _smoothstep(Float32(13.5), Float32(19.5), wy + (n_med - Float32(0.5)) * Float32(4.5))
            var snow_slope = _smoothstep(Float32(0.56), Float32(0.76), wn.y)
            var snow_w = snow_alt * snow_slope
            if snow_w > Float32(0.001):
                var sn = Float32(228.0) + n_fine * Float32(22.0)
                meadow_r = meadow_r * (Float32(1.0) - snow_w) + sn * snow_w
                meadow_g = meadow_g * (Float32(1.0) - snow_w) + min(Float32(255.0), sn + Float32(3.0)) * snow_w
                meadow_b = meadow_b * (Float32(1.0) - snow_w) + min(Float32(255.0), sn + Float32(8.0)) * snow_w
                rough_f = rough_f * (Float32(1.0) - snow_w) + Float32(118.0) * snow_w
                ao_f = ao_f * (Float32(1.0) - snow_w) + Float32(246.0) * snow_w

            var r_u = UInt32(max(0, min(255, Int(meadow_r))))
            var g_u = UInt32(max(0, min(255, Int(meadow_g))))
            var b_u = UInt32(max(0, min(255, Int(meadow_b))))
            var ao_u = UInt32(max(80, min(255, Int(ao_f))))
            var rg_u = UInt32(max(60, min(252, Int(rough_f))))
            albedo[y * DIM + x] = UInt32(0xFF000000) | (r_u << 16) | (g_u << 8) | b_u
            orm[y * DIM + x] = UInt32(0xFF000000) | (ao_u << 16) | (rg_u << 8) | UInt32(4)

    return (albedo^, orm^)


# ============================================================================
# 2. High-Resolution Procedural `BC7 / BC5` PBR Material Sets & `256x256` HDR Sky
# ============================================================================

def make_biome_textures_256(mode: Int) -> Tuple[List[UInt32], List[Vec3], List[UInt32]]:
    """Generates `256x256` seamless PBR `(albedo_argb, tangent_normals, orm_argb)` textures."""
    comptime DIM: Int = 256
    var albedo = List[UInt32](length=DIM * DIM, fill=0xFFFFFFFF)
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))
    var orm = List[UInt32](length=DIM * DIM, fill=0xFFFFFFFF)
    var h_map = List[Float32](length=DIM * DIM, fill=0.0)
    var bump_scale = Float32(2.2)

    for y in range(DIM):
        var u = Float32(y) / Float32(DIM)
        for x in range(DIM):
            var v = Float32(x) / Float32(DIM)
            var f1 = _value_noise_tileable(v * Float32(8.0), u * Float32(8.0), 8)
            var f2 = _value_noise_tileable(v * Float32(16.0), u * Float32(16.0), 16)
            var f3 = _value_noise_tileable(v * Float32(32.0), u * Float32(32.0), 32)
            var f4 = _value_noise_tileable(v * Float32(64.0), u * Float32(64.0), 64)
            var fbm = f1 * Float32(0.48) + f2 * Float32(0.28) + f3 * Float32(0.16) + f4 * Float32(0.08)

            if mode == 0:
                var lush = max(Float32(0.0), min(Float32(1.0), (fbm - Float32(0.32)) * Float32(2.4)))
                var dry = max(Float32(0.0), min(Float32(1.0), (f2 - Float32(0.62)) * Float32(3.0)))
                var r = Int(Float32(54.0) * (Float32(1.0) - lush) + Float32(68.0) * lush + Float32(48.0) * dry)
                var g = Int(Float32(88.0) * (Float32(1.0) - lush) + Float32(118.0) * lush + Float32(28.0) * dry)
                var b = Int(Float32(42.0) * (Float32(1.0) - lush) + Float32(48.0) * lush + Float32(12.0) * dry)
                albedo[y * DIM + x] = (
                    UInt32(0xFF000000) | (UInt32(min(255, r)) << 16) | (UInt32(min(255, g)) << 8) | UInt32(min(255, b))
                )
                var ao = max(145, min(255, Int(Float32(175.0) + fbm * Float32(80.0))))
                var rough = max(170, min(250, Int(Float32(205.0) + f4 * Float32(40.0))))
                orm[y * DIM + x] = UInt32(0xFF000000) | (UInt32(ao) << 16) | (UInt32(rough) << 8) | UInt32(0)
                h_map[y * DIM + x] = fbm * Float32(0.65) + f4 * Float32(0.35)
                bump_scale = Float32(1.8)
            elif mode == 1:
                var strata = _value_noise_tileable(v * Float32(6.0), u * Float32(24.0), 24)
                var crack = abs(
                    _value_noise_tileable(v * Float32(16.0), u * Float32(8.0), 16) * Float32(2.0) - Float32(1.0)
                )
                var h_rock = strata * Float32(0.55) + min(Float32(1.0), crack * Float32(1.6)) * Float32(0.45)
                var shade = Float32(52.0) + h_rock * Float32(58.0)
                var r = Int(shade * Float32(1.04))
                var g = Int(shade * Float32(1.00))
                var b = Int(shade * Float32(0.98) + Float32(6.0))
                albedo[y * DIM + x] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                var ao = max(90, min(255, Int(Float32(120.0) + h_rock * Float32(135.0))))
                var rough = max(135, min(235, Int(Float32(215.0) - h_rock * Float32(55.0))))
                orm[y * DIM + x] = UInt32(0xFF000000) | (UInt32(ao) << 16) | (UInt32(rough) << 8) | UInt32(8)
                h_map[y * DIM + x] = h_rock
                bump_scale = Float32(3.4)
            elif mode == 2:
                var drift = _value_noise_tileable(v * Float32(12.0), u * Float32(6.0), 12)
                var sparkle = f4
                var base_s = Float32(226.0) + drift * Float32(24.0)
                var r = min(255, Int(base_s))
                var g = min(255, Int(base_s + Float32(3.0)))
                var b = min(255, Int(base_s + Float32(8.0)))
                albedo[y * DIM + x] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                var rough = max(85, min(185, Int(Float32(150.0) - sparkle * Float32(50.0))))
                orm[y * DIM + x] = UInt32(0xFF000000) | (UInt32(245) << 16) | (UInt32(rough) << 8) | UInt32(0)
                h_map[y * DIM + x] = drift * Float32(0.75) + f3 * Float32(0.25)
                bump_scale = Float32(1.2)
            else:
                var row = (y // 32) % 8
                var stagger = 16 if (row & 1) == 1 else 0
                var bx = (x + stagger) % 64
                var by = y % 32
                var mortar = 1 if (bx < 3 or by < 3) else 0
                var block_hash = _hash2d_01((x + stagger) // 64, y // 32)
                var stone_v = Float32(148.0) + block_hash * Float32(34.0) + (fbm - Float32(0.5)) * Float32(22.0)
                if mortar == 1:
                    stone_v = Float32(68.0)
                var r = Int(stone_v * Float32(1.05))
                var g = Int(stone_v * Float32(0.99))
                var b = Int(stone_v * Float32(0.91))
                albedo[y * DIM + x] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                var ao = 110 if mortar == 1 else 242
                var rough = 225 if mortar == 1 else Int(Float32(115.0) + f3 * Float32(45.0))
                orm[y * DIM + x] = UInt32(0xFF000000) | (UInt32(ao) << 16) | (UInt32(rough) << 8) | UInt32(10)
                h_map[y * DIM + x] = Float32(0.15) if mortar == 1 else (Float32(0.85) + f3 * Float32(0.15))
                bump_scale = Float32(2.8)

    for y in range(DIM):
        for x in range(DIM):
            var hl = h_map[y * DIM + ((x + DIM - 1) % DIM)]
            var hr = h_map[y * DIM + ((x + 1) % DIM)]
            var hd = h_map[((y + DIM - 1) % DIM) * DIM + x]
            var hu = h_map[((y + 1) % DIM) * DIM + x]
            normals[y * DIM + x] = Vec3((hl - hr) * bump_scale, (hd - hu) * bump_scale, Float32(1.0)).normalized()

    return (albedo^, normals^, orm^)


def make_water_ripple_normals_256() -> List[Vec3]:
    """Generates a seamless `256x256` isotropic 5-octave FBM wind-chop `BC5` water normal map."""
    comptime DIM: Int = 256
    var h_map = List[Float32](length=DIM * DIM, fill=0.0)
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))

    for y in range(DIM):
        var fy = Float32(y) / Float32(DIM)
        for x in range(DIM):
            var fx = Float32(x) / Float32(DIM)
            var w1 = _value_noise_tileable(fx * Float32(8.0), fy * Float32(12.0), 8) * Float32(0.36)
            var w2 = _value_noise_tileable(fx * Float32(16.0) + Float32(3.7), fy * Float32(24.0), 16) * Float32(0.28)
            var w3 = _value_noise_tileable(fx * Float32(32.0), fy * Float32(32.0) + Float32(5.1), 32) * Float32(0.20)
            var w4 = _value_noise_tileable(fx * Float32(64.0), fy * Float32(64.0), 64) * Float32(0.16)
            h_map[y * DIM + x] = w1 + w2 + w3 + w4

    for y in range(DIM):
        for x in range(DIM):
            var hl = h_map[y * DIM + ((x + DIM - 1) % DIM)]
            var hr = h_map[y * DIM + ((x + 1) % DIM)]
            var hd = h_map[((y + DIM - 1) % DIM) * DIM + x]
            var hu = h_map[((y + DIM - 1 + 2) % DIM) * DIM + x]
            normals[y * DIM + x] = Vec3((hl - hr) * Float32(0.42), (hd - hu) * Float32(0.42), Float32(1.0)).normalized()

    return normals^


@always_inline
def pow_fast4(x: Float32) -> Float32:
    var x2 = x * x
    return x2 * x2


def bake_cloud_and_star_sky_ibl(mut ren: FlintAaaVulkanRenderer, mut rw: RenderWorld, preset: Int):
    """Bakes the `256 x 256` `BC6H` Bruneton cloud/star sky IBL and syncs `RenderWorld` sun/IBL state."""
    var atmo: BrunetonAtmosphereParams
    if preset == 0:
        atmo = BrunetonAtmosphereParams(
            sun_dir=Vec3(0.42, 0.72, 0.55),
            turbidity=1.0,
            solar_intensity=6.8,
            aerial_km_per_unit=0.0038,
        )
        atmo.ground_albedo = Vec3(0.12, 0.16, 0.12)
    elif preset == 1:
        atmo = BrunetonAtmosphereParams(
            sun_dir=Vec3(0.18, 0.135, 0.974),
            turbidity=1.18,
            solar_intensity=8.5,
            aerial_km_per_unit=0.0055,
        )
        atmo.ground_albedo = Vec3(0.18, 0.14, 0.10)
    else:
        atmo = BrunetonAtmosphereParams(
            sun_dir=Vec3(-0.30, -0.06, 0.952),
            turbidity=1.05,
            solar_intensity=0.85,
            aerial_km_per_unit=0.0032,
        )
        atmo.ground_albedo = Vec3(0.04, 0.05, 0.08)

    var luts = BrunetonSkyLuts(atmo)
    var hdr = HdrImage()
    hdr.valid = True
    hdr.width = 640
    hdr.height = 320
    hdr.pixels = List[Vec3](length=hdr.width * hdr.height, fill=Vec3(0.0, 0.0, 0.0))

    var sun_d = atmo.sun_dir.normalized()
    var t_sun_obs = luts.sun_transmittance_at_observer()
    var pi = Float32(3.14159265359)
    var tau = Float32(6.28318530718)

    for y in range(hdr.height):
        var theta = (Float32(y) + Float32(0.5)) / Float32(hdr.height) * pi
        var sin_t = sin(theta)
        var cos_t = cos(theta)
        for x in range(hdr.width):
            var phi = ((Float32(x) + Float32(0.5)) / Float32(hdr.width) - Float32(0.5)) * tau
            var d = Vec3(sin_t * cos(phi), cos_t, sin_t * sin(phi)).normalized()

            var base_sky = luts.sample_sky(d, include_sun_disk=False)
            if preset == 0:
                base_sky = Vec3(base_sky.x * Float32(0.72), base_sky.y * Float32(0.88), base_sky.z * Float32(1.32))
            elif preset == 1:
                var horizon_glow = max(Float32(0.0), Float32(1.0) - abs(d.y) * Float32(3.2))
                var sun_az = max(Float32(0.0), d.dot(sun_d))
                var warm_add = Vec3(2.4, 0.95, 0.22) * (horizon_glow * horizon_glow * sun_az * sun_az)
                base_sky = (
                    Vec3(base_sky.x * Float32(1.25), base_sky.y * Float32(0.95), base_sky.z * Float32(0.88))
                    + warm_add
                )
            else:
                var zen = max(Float32(0.0), d.y)
                var aur = exp(-abs(d.y - Float32(0.16)) * Float32(14.0)) * _vnoise2(
                    phi * Float32(3.5), d.y * Float32(8.0)
                )
                base_sky = Vec3(
                    Float32(0.012) + Float32(0.032) * (Float32(1.0) - zen),
                    Float32(0.022) + Float32(0.052) * (Float32(1.0) - zen) + aur * Float32(0.045),
                    Float32(0.062) + Float32(0.105) * (Float32(1.0) - zen) + aur * Float32(0.065),
                )
                if d.y > Float32(0.06):
                    var star_cell_x = Int(floor((phi + pi) * Float32(160.0)))
                    var star_cell_y = Int(floor(theta * Float32(160.0)))
                    var sh = _hash2d_01(star_cell_x, star_cell_y)
                    if sh > Float32(0.9955):
                        var mag = (sh - Float32(0.9955)) * Float32(95.0)
                        var tint = _hash2d_01(star_cell_x + 71, star_cell_y - 37)
                        var scol = Vec3(0.45, 0.40, 0.32) if tint < Float32(0.4) else Vec3(0.32, 0.38, 0.48)
                        base_sky = base_sky + scol * (mag * zen)

            if d.y > Float32(0.025) and preset < 2:
                var cx_p = d.x / (d.y + Float32(0.14)) * Float32(3.2)
                var cz_p = d.z / (d.y + Float32(0.14)) * Float32(3.2)
                var c_fbm = (
                    _vnoise2(cx_p + Float32(4.2), cz_p - Float32(1.7)) * Float32(0.52)
                    + _vnoise2(cx_p * Float32(2.15) - Float32(3.1), cz_p * Float32(2.15)) * Float32(0.28)
                    + _vnoise2(cx_p * Float32(4.5), cz_p * Float32(4.5) + Float32(5.4)) * Float32(0.13)
                    + _vnoise2(cx_p * Float32(9.2), cz_p * Float32(9.2)) * Float32(0.07)
                )
                var c_sun_s = _vnoise2(
                    cx_p + sun_d.x * Float32(0.35) + Float32(4.2),
                    cz_p + sun_d.z * Float32(0.35) - Float32(1.7),
                )
                var density = max(Float32(0.0), min(Float32(1.0), (c_fbm - Float32(0.50)) * Float32(3.4)))
                var horizon_fade = min(Float32(1.0), (d.y - Float32(0.025)) * Float32(5.5))
                var cld_alpha = density * horizon_fade * Float32(0.88)

                if cld_alpha > Float32(0.002):
                    var lit = max(Float32(0.15), min(Float32(1.0), Float32(0.65) + (c_fbm - c_sun_s) * Float32(2.6)))
                    var sdot = max(Float32(0.0), d.dot(sun_d))
                    var hg_lining = pow_fast4(sdot) * (Float32(1.0) - density * Float32(0.6))
                    var cld_lit_col: Vec3
                    var cld_shd_col: Vec3
                    if preset == 0:
                        cld_lit_col = Vec3(3.4, 3.35, 3.25)
                        cld_shd_col = Vec3(0.42, 0.56, 0.82)
                    else:
                        cld_lit_col = Vec3(4.8, 2.35, 0.72) + Vec3(3.6, 1.8, 0.45) * hg_lining
                        cld_shd_col = Vec3(0.28, 0.24, 0.38)
                    var cld_col = cld_shd_col * (Float32(1.0) - lit) + cld_lit_col * lit
                    base_sky = base_sky * (Float32(1.0) - cld_alpha) + cld_col * cld_alpha

            if d.y > Float32(0.0) and preset < 2:
                var sdot = d.dot(sun_d)
                if sdot > Float32(0.9985):
                    var disc = min(Float32(1.0), (sdot - Float32(0.9985)) / Float32(0.0012))
                    var sun_col = Vec3(28.0, 25.0, 20.0) if preset == 0 else Vec3(32.0, 14.5, 3.8)
                    base_sky = base_sky + sun_col * disc

            hdr.pixels[y * hdr.width + x] = base_sky

    var ibl = bake_hdr_image_ibl_cubemap_and_sh9(
        hdr,
        base_dim=ren.env_cube_dim,
        mip_levels=ren.env_mip_levels,
        ggx_samples=40,
    )
    ren.upload_ibl_environment(ibl)
    ren.sun_dir = sun_d
    ren.sun_angular_radius = Float32(0.0045)
    if preset == 0:
        ren.sun_radiance = Vec3(4.6, 4.35, 3.95)
    elif preset == 1:
        ren.sun_radiance = Vec3(5.2, 2.55, 0.82)
    else:
        ren.sun_dir = Vec3(-0.28, 0.42, -0.86).normalized()
        ren.sun_radiance = Vec3(0.08, 0.13, 0.26)
    _ = t_sun_obs

    rw.sun_dir = ren.sun_dir
    rw.sun_radiance = ren.sun_radiance

    var ap = luts.compute_aerial_perspective_constants()
    ren.aerial_sigma_ext = ap[0]
    ren.aerial_r_term = ap[1] * (Float32(0.55) if preset < 2 else Float32(0.15))
    ren.aerial_m_term = ap[2] * (Float32(0.45) if preset < 2 else Float32(0.10))
    ren.aerial_mie_g = ap[3]
    ren.enable_aerial_perspective = True


# ============================================================================
# 3. Sculpted `.flpk` v3 glTF Sanctum, Viaduct Bridge, Pine Forest & Knight
# ============================================================================

def build_and_pack_sunspire_sanctum_flpk(
    pack_path: String,
    mut asset_server: AssetServer,
) -> NativeModel:
    """Builds the 12-column domed `Sunspire Sanctum`, packs it via `flpk_v3.write_flpk_v3`,
    and streams it back through `AssetServer.mount_flpk` + `load_raw` + `parse_native_glb`."""
    var src = NativeModel()

    var tw = 64
    var th = 64
    var t_alb = NativeTexture()
    t_alb.width = tw
    t_alb.height = th
    t_alb.pixels = List[UInt32](length=tw * th, fill=0xFFFFFFFF)
    for y in range(th):
        for x in range(tw):
            var vein = _value_noise_tileable(Float32(x) * Float32(0.25), Float32(y) * Float32(0.5), 16)
            var gold_trim = 1 if (y % 16 == 0 or y % 16 == 1) else 0
            var r = 216 if gold_trim == 1 else Int(Float32(205.0) + vein * Float32(28.0))
            var g = 174 if gold_trim == 1 else Int(Float32(196.0) + vein * Float32(26.0))
            var b = 92 if gold_trim == 1 else Int(Float32(178.0) + vein * Float32(24.0))
            t_alb.pixels[y * tw + x] = UInt32(0xFF000000) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
    src.textures.append(t_alb^)

    var t_nrm = NativeTexture()
    t_nrm.width = 32
    t_nrm.height = 32
    t_nrm.pixels = List[UInt32](length=32 * 32, fill=0xFF8080FF)
    for y in range(32):
        for x in range(32):
            var flute = Int(Float32(128.0) + sin(Float32(x) / Float32(32.0) * Float32(6.2831853 * 8.0)) * Float32(28.0))
            var ny = 152 if (y % 8 == 0) else 128
            t_nrm.pixels[y * 32 + x] = UInt32(0xFF0000FF) | (UInt32(flute) << 16) | (UInt32(ny) << 8)
    src.textures.append(t_nrm^)

    var t_orm = NativeTexture()
    t_orm.width = 16
    t_orm.height = 16
    t_orm.pixels = List[UInt32](length=16 * 16, fill=0xFFF65818)
    src.textures.append(t_orm^)

    var m0_b = MeshBuilder(base_color=0xFFF4EFE6, roughness=0.38, metallic=0.06, texture=0)
    m0_b.add_cylinder(Vec3(0.0, -5.5, 0.0), Vec3(0.0, 0.0, 0.0), 5.8, 5.6, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.55, 0.0), 5.6, 5.4, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 0.55, 0.0), Vec3(0.0, 1.10, 0.0), 5.0, 4.85, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 1.10, 0.0), Vec3(0.0, 1.45, 0.0), 4.45, 4.40, segments=28)
    for c_i in range(12):
        var ang = Float32(c_i) * Float32(6.2831853 / 12.0)
        var cx = cos(ang) * Float32(3.85)
        var cz = sin(ang) * Float32(3.85)
        m0_b.add_cylinder(Vec3(cx, 1.45, cz), Vec3(cx, 1.80, cz), 0.42, 0.36, segments=14)
        m0_b.add_cylinder(Vec3(cx, 1.80, cz), Vec3(cx, 6.35, cz), 0.33, 0.27, segments=16)
        m0_b.add_cylinder(Vec3(cx, 6.35, cz), Vec3(cx, 6.72, cz), 0.29, 0.44, segments=14)
    m0_b.add_cylinder(Vec3(0.0, 6.72, 0.0), Vec3(0.0, 7.35, 0.0), 4.35, 4.48, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 7.35, 0.0), Vec3(0.0, 7.85, 0.0), 4.10, 3.95, segments=28)
    var m0 = m0_b^.finish()
    m0.normal_orm_texture = 1
    m0.orm_texture = 2
    src.meshes.append(m0^)

    var m1_b = MeshBuilder(base_color=0xFFE5B869, roughness=0.22, metallic=0.88, texture=0)
    m1_b.add_ellipsoid(
        Vec3(0.0, 0.0, 0.0),
        Vec3(3.85, 2.55, 3.85),
        rings=12,
        segments=28,
        theta_start=0.0,
        theta_end=1.52,
    )
    m1_b.add_cylinder(Vec3(0.0, 2.45, 0.0), Vec3(0.0, 3.55, 0.0), 0.85, 0.68, segments=16)
    m1_b.add_cylinder(Vec3(0.0, 3.55, 0.0), Vec3(0.0, 6.20, 0.0), 0.42, 0.03, segments=14)
    m1_b.add_cylinder(Vec3(0.0, -6.40, 0.0), Vec3(0.0, -5.25, 0.0), 1.15, 0.78, segments=18)
    var m1 = m1_b^.finish()
    m1.normal_orm_texture = 1
    src.meshes.append(m1^)

    var m2_b = MeshBuilder(
        base_color=0xFFFFFFFF,
        roughness=0.10,
        metallic=0.0,
        emissive_color=Vec3(1.0, 0.72, 0.28),
        emissive_strength=18.0,
    )
    m2_b.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.35, 0.0), 0.04, 0.68, segments=8)
    m2_b.add_cylinder(Vec3(0.0, 1.35, 0.0), Vec3(0.0, 2.70, 0.0), 0.68, 0.04, segments=8)
    src.meshes.append(m2_b^.finish())

    var r_node = NativeNode()
    r_node.name = "sanctum_root"
    r_node.children.append(1)
    r_node.children.append(2)
    r_node.children.append(3)
    var n_col = NativeNode()
    n_col.name = "rotunda_colonnade"
    n_col.mesh = 0
    n_col.mesh_indices.append(0)
    var n_dome = NativeNode()
    n_dome.name = "gilded_cupola"
    n_dome.mesh = 1
    n_dome.mesh_indices.append(1)
    n_dome.translation = Vec3(0.0, 7.80, 0.0)
    var n_core = NativeNode()
    n_core.name = "celestial_crystal"
    n_core.mesh = 2
    n_core.mesh_indices.append(2)
    n_core.translation = Vec3(0.0, 2.85, 0.0)
    src.nodes.append(r_node^)
    src.nodes.append(n_col^)
    src.nodes.append(n_dome^)
    src.nodes.append(n_core^)
    src.roots.append(0)

    var rel_glb = "models/sunspire_sanctum.glb"
    var glb_bytes = encode_native_glb(src, jpeg_texture_mask=1)
    var chunks = List[FlpkBuildChunk]()
    chunks.append(
        FlpkBuildChunk(
            AssetUuid.from_path(rel_glb),
            rel_glb,
            SUB_ID_PRIMARY_OR_TAIL,
            ASSET_KIND_RAW,
            glb_bytes^,
            CHUNK_COMP_NONE,
        )
    )
    _ = make_dirs("build")
    _ = write_flpk_v3(pack_path, chunks, page_align=True)
    _ = asset_server.mount_flpk(pack_path)
    var h_raw = asset_server.load_raw(rel_glb)
    var raw_opt = asset_server.raws.get_ptr(h_raw)
    if raw_opt:
        return parse_native_glb(raw_opt.value()[unsafe_offset=0].bytes)
    return src^


def build_viaduct_bridge_meshes() -> Tuple[NativeMesh, NativeMesh, NativeMesh]:
    """Builds the multi-arch stone viaduct bridge (`z = -26 .. +15`) spanning the glacial lake."""
    var mb_stone = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.65, metallic=0.04)
    var mb_bronze = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.26, metallic=0.86)
    var mb_glow = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.12, metallic=0.0)

    var deck_y = Float32(1.65)
    for bay in range(11):
        var bz = Float32(-25.0) + Float32(bay) * Float32(3.9)
        mb_stone.add_beveled_box(Vec3(0.0, deck_y - Float32(0.20), bz), Vec3(2.85, 0.22, 1.96), bevel=0.03)
        mb_stone.add_beveled_box(Vec3(-2.62, deck_y + Float32(0.32), bz), Vec3(0.22, 0.32, 1.96), bevel=0.03)
        mb_stone.add_beveled_box(Vec3(2.62, deck_y + Float32(0.32), bz), Vec3(0.22, 0.32, 1.96), bevel=0.03)
        mb_stone.add_beveled_box(Vec3(-2.62, deck_y + Float32(0.68), bz), Vec3(0.28, 0.06, 1.96), bevel=0.02)
        mb_stone.add_beveled_box(Vec3(2.62, deck_y + Float32(0.68), bz), Vec3(0.28, 0.06, 1.96), bevel=0.02)

    for st in range(5):
        var sy = deck_y - Float32(st) * Float32(0.28)
        mb_stone.add_beveled_box(
            Vec3(0.0, sy - Float32(0.2), Float32(-27.2) - Float32(st) * Float32(0.75)),
            Vec3(3.2, 0.22, 0.45),
            bevel=0.03,
        )

    for a_i in range(5):
        var z_pier = Float32(-21.5) + Float32(a_i) * Float32(8.0)
        mb_stone.add_beveled_box(Vec3(0.0, -0.85, z_pier), Vec3(2.95, 2.35, 0.85), bevel=0.06)
        mb_stone.add_cylinder(Vec3(-2.95, -2.8, z_pier), Vec3(-2.95, 1.35, z_pier), 0.65, 0.55, segments=10)
        mb_stone.add_cylinder(Vec3(2.95, -2.8, z_pier), Vec3(2.95, 1.35, z_pier), 0.65, 0.55, segments=10)

        for side in range(2):
            var sx = Float32(-2.62) if side == 0 else Float32(2.62)
            mb_stone.add_beveled_box(Vec3(sx, deck_y + Float32(0.65), z_pier), Vec3(0.38, 0.42, 0.38), bevel=0.04)
            mb_bronze.add_cylinder(
                Vec3(sx, deck_y + Float32(1.05), z_pier),
                Vec3(sx, deck_y + Float32(2.45), z_pier),
                0.12,
                0.08,
                segments=10,
            )
            mb_bronze.add_cylinder(
                Vec3(sx, deck_y + Float32(2.85), z_pier),
                Vec3(sx, deck_y + Float32(3.22), z_pier),
                0.28,
                0.03,
                segments=8,
            )
            mb_glow.add_ellipsoid(
                Vec3(sx, deck_y + Float32(2.65), z_pier), Vec3(0.16, 0.22, 0.16), rings=8, segments=10
            )

        if a_i < 4:
            var z_mid = z_pier + Float32(4.0)
            for seg in range(7):
                var th0 = Float32(0.25) + Float32(seg) / Float32(7.0) * Float32(2.64)
                var th1 = Float32(0.25) + Float32(seg + 1) / Float32(7.0) * Float32(2.64)
                var thm = (th0 + th1) * Float32(0.5)
                var az = z_mid - cos(thm) * Float32(3.25)
                var ay = -Float32(0.82) + sin(thm) * Float32(1.85)
                mb_stone.add_beveled_box(Vec3(0.0, ay, az), Vec3(2.65, 0.28, 0.52), bevel=0.03)

    return (mb_stone^.finish(), mb_bronze^.finish(), mb_glow^.finish())


def build_alpine_pine_tree_meshes() -> Tuple[NativeMesh, NativeMesh]:
    """Builds a natural asymmetric multi-whorl Alpine Spruce/Fir."""
    var mb_trunk = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.88, metallic=0.0)
    mb_trunk.add_cylinder(Vec3(0.0, -0.4, 0.0), Vec3(0.0, 6.8, 0.0), 0.30, 0.05, segments=9)

    var mb_foliage = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.78, metallic=0.0, double_sided=True)
    var tier_y = [Float32(1.2), Float32(2.1), Float32(3.0), Float32(3.85), Float32(4.65), Float32(5.4), Float32(6.05)]
    var tier_h = [Float32(1.55), Float32(1.45), Float32(1.35), Float32(1.25), Float32(1.15), Float32(1.05), Float32(1.15)]
    var tier_r = [Float32(1.95), Float32(1.72), Float32(1.46), Float32(1.20), Float32(0.94), Float32(0.68), Float32(0.38)]
    for t_i in range(7):
        var ox = sin(Float32(t_i) * Float32(2.1)) * Float32(0.09)
        var oz = cos(Float32(t_i) * Float32(1.7)) * Float32(0.09)
        mb_foliage.add_cylinder(
            Vec3(ox, tier_y[t_i], oz),
            Vec3(0.0, tier_y[t_i] + tier_h[t_i], 0.0),
            tier_r[t_i],
            0.04,
            segments=7,
        )
    return (mb_trunk^.finish(), mb_foliage^.finish())


def build_granite_boulder_mesh() -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.76, metallic=0.03)
    mb.add_ellipsoid(Vec3(0.0, 0.35, 0.0), Vec3(1.15, 0.72, 0.92), rings=7, segments=9)
    mb.add_ellipsoid(Vec3(0.55, 0.22, 0.35), Vec3(0.68, 0.48, 0.62), rings=6, segments=8)
    return mb^.finish()


def build_swept_glider_mesh() -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.78, double_sided=True)
    mb.add_cylinder(Vec3(0.0, 0.0, -0.42), Vec3(0.0, 0.0, 0.48), 0.03, 0.07, segments=8)
    mb.add_cylinder(Vec3(0.0, 0.0, 0.48), Vec3(0.0, 0.0, 0.72), 0.07, 0.01, segments=8)
    mb.add_airfoil(Vec3(0.0, 0.0, 0.25), 0.28, 0.09, Vec3(-0.95, 0.12, -0.22), 0.10, 0.06)
    mb.add_airfoil(Vec3(0.0, 0.0, 0.25), 0.28, 0.09, Vec3(0.95, 0.12, -0.22), 0.10, 0.06)
    return mb^.finish()


def upload_aegis_sentinel_knight(mut ren: FlintAaaVulkanRenderer, mat_id: Int) -> AaaSkinnedMeshEntry:
    """Builds the 4-bone articulated `Aegis Sentinel` knight for GPU DQ skinning + BLAS refit."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.24, metallic=0.86, double_sided=True)

    for side in range(2):
        var sx = Float32(-0.46) if side == 0 else Float32(0.46)
        mb.add_beveled_box(Vec3(sx, 0.14, 0.12), Vec3(0.20, 0.14, 0.34), bevel=0.05)
        mb.add_cylinder(Vec3(sx, 0.24, 0.0), Vec3(sx, 1.32, 0.0), 0.17, 0.23, segments=12)
        mb.add_ellipsoid(Vec3(sx, 1.36, 0.14), Vec3(0.24, 0.22, 0.18), rings=8, segments=10)
        mb.add_cylinder(Vec3(sx, 1.40, 0.0), Vec3(sx, 2.42, 0.0), 0.23, 0.28, segments=12)
    mb.add_cylinder(Vec3(0.0, 2.32, 0.0), Vec3(0.0, 2.82, 0.0), 0.68, 0.56, segments=14)

    mb.add_cylinder(Vec3(0.0, 2.82, 0.0), Vec3(0.0, 4.18, 0.0), 0.56, 0.82, segments=16)
    mb.add_ellipsoid(Vec3(0.0, 3.65, 0.14), Vec3(0.72, 0.62, 0.48), rings=10, segments=14)
    mb.add_cylinder(Vec3(0.0, 4.18, 0.0), Vec3(0.0, 4.46, 0.0), 0.52, 0.34, segments=14)
    mb.add_ellipsoid(Vec3(0.0, 4.82, 0.02), Vec3(0.36, 0.44, 0.40), rings=10, segments=14)
    mb.add_beveled_box(Vec3(0.0, 5.26, -0.06), Vec3(0.07, 0.26, 0.44), bevel=0.03)

    mb.add_ellipsoid(Vec3(-1.02, 4.02, 0.0), Vec3(0.38, 0.32, 0.36), rings=8, segments=12)
    mb.add_cylinder(Vec3(-1.10, 4.00, 0.0), Vec3(-1.46, 2.88, 0.28), 0.18, 0.14, segments=10)
    mb.add_beveled_box(Vec3(-1.58, 2.95, 0.46), Vec3(0.08, 0.92, 0.56), bevel=0.04)

    mb.add_ellipsoid(Vec3(1.02, 4.02, 0.0), Vec3(0.38, 0.32, 0.36), rings=8, segments=12)
    mb.add_cylinder(Vec3(1.10, 4.00, 0.0), Vec3(1.42, 3.05, 0.24), 0.18, 0.14, segments=10)
    mb.add_cylinder(Vec3(1.44, 0.10, 0.28), Vec3(1.44, 6.25, 0.28), 0.055, 0.050, segments=8)
    mb.add_cylinder(Vec3(1.44, 6.25, 0.28), Vec3(1.44, 7.20, 0.28), 0.14, 0.01, segments=8)
    mb.add_beveled_box(Vec3(1.44, 5.95, 0.52), Vec3(0.03, 0.42, 0.26), bevel=0.02)

    var raw = mb^.finish()
    var vcnt = raw.vertex_count()
    var j_pack = List[UInt32](capacity=vcnt)
    var j_w012 = List[Vec3](capacity=vcnt)

    for v in range(vcnt):
        var px = raw.positions[v * 3 + 0]
        var py = raw.positions[v * 3 + 1]
        if px < Float32(-0.86):
            j_pack.append(UInt32(2) | (UInt32(1) << 8))
            j_w012.append(Vec3(0.88, 0.12, 0.0))
        elif px > Float32(0.86):
            j_pack.append(UInt32(3) | (UInt32(1) << 8))
            j_w012.append(Vec3(0.90, 0.10, 0.0))
        elif py > Float32(2.65):
            var t_up = min(Float32(1.0), (py - Float32(2.65)) * Float32(1.2))
            j_pack.append(UInt32(1) | (UInt32(0) << 8))
            j_w012.append(Vec3(t_up, Float32(1.0) - t_up, 0.0))
        else:
            j_pack.append(UInt32(0))
            j_w012.append(Vec3(1.0, 0.0, 0.0))

    return ren.add_skinned_mesh(raw, j_pack, j_w012, mat_id, bone_count=4)


def evaluate_sentinel_bones(t_sec: Float32) -> List[DualQuat]:
    var bones = List[DualQuat](length=4, fill=DualQuat())
    var torso_yaw = sin(t_sec * Float32(0.85)) * Float32(0.14)
    var q_torso = Quat(0.0, sin(torso_yaw * Float32(0.5)), 0.0, cos(torso_yaw * Float32(0.5)))
    bones[1] = DualQuat.from_quat_trans(q_torso, Vec3(0.0, sin(t_sec * Float32(1.7)) * Float32(0.05), 0.0))

    var l_pitch = Float32(-0.10) + sin(t_sec * Float32(1.2)) * Float32(0.12)
    var q_l = Quat(sin(l_pitch * Float32(0.5)), 0.0, 0.0, cos(l_pitch * Float32(0.5))).multiply(q_torso)
    bones[2] = DualQuat.from_quat_trans(q_l, Vec3(0.0, 0.0, 0.0))

    var r_pitch = Float32(0.06) + cos(t_sec * Float32(1.2)) * Float32(0.10)
    var q_r = Quat(sin(r_pitch * Float32(0.5)), 0.0, 0.0, cos(r_pitch * Float32(0.5))).multiply(q_torso)
    bones[3] = DualQuat.from_quat_trans(q_r, Vec3(0.0, 0.0, 0.0))
    return bones^


# ============================================================================
# 4. ECS Components, State Resource & Parallel Simulation Kernel
# ============================================================================

@fieldwise_init
struct GliderOrWispComponent(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.examples.open_world.GliderOrWispComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var index: Int
    var is_glider: Bool
    var flock_id: Int
    var member: Int
    var angle: Float32
    var radius: Float32
    var speed: Float32


@fieldwise_init
struct AreaLightComponent(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.examples.open_world.AreaLightComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var light: AaaClusterLight


@fieldwise_init
struct ScaledLightTag(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.examples.open_world.ScaledLightTag")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var base_radiance: Vec3
    var min_scale: Float32


@fieldwise_init
struct FlockSimCtx(Copyable, ImplicitlyCopyable, Movable):
    var time_sec: Float32
    var dt: Float32


struct OpenWorldStateResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.examples.open_world.OpenWorldStateResource")
    var headless_shots: Bool
    var freeze_sim: Bool
    var sim_time: Float32
    var tod_preset: Int
    var tod_label: String
    var light_scale: Float32
    var cam_entity: Entity
    var cam_pos: Vec3
    var cam_yaw: Float32
    var cam_pitch: Float32
    var sentinel_skin: AaaSkinnedMeshEntry
    var prev_bones: List[DualQuat]
    var asset_server: AssetServer
    var scene: Scene
    var overlays: FlintDevOverlays
    var last_sim_us: Int
    var last_blas_refit_us: Int
    var workers: Int

    def __init__(
        out self,
        headless_shots: Bool,
        var asset_server: AssetServer,
        var sentinel_skin: AaaSkinnedMeshEntry,
        workers: Int,
    ):
        self.headless_shots = headless_shots
        self.freeze_sim = False
        self.sim_time = Float32(0.0)
        self.tod_preset = 0
        self.tod_label = "HIGH NOON ALPINE SANCTUARY"
        self.light_scale = Float32(0.08)
        self.cam_entity = Entity()
        self.cam_pos = Vec3(
            -22.0,
            max(Float32(15.5), sample_valley_height(-22.0, -34.0) + Float32(8.5)),
            -34.0,
        )
        self.cam_yaw = Float32(0.38)
        self.cam_pitch = Float32(-0.17)
        self.sentinel_skin = sentinel_skin^
        self.prev_bones = evaluate_sentinel_bones(Float32(0.0))
        self.asset_server = asset_server^
        self.scene = Scene("SunspireValley")
        self.overlays = FlintDevOverlays()
        self.overlays.show_profiler = True
        self.last_sim_us = 1
        self.last_blas_refit_us = 1
        self.workers = workers


@always_inline
def _glider_wisp_pose(comp: GliderOrWispComponent, angle: Float32, time_sec: Float32) -> Tuple[Vec3, Quat]:
    if comp.is_glider:
        var wing_side = Float32(-1.0) if (comp.member & 1) == 0 else Float32(1.0)
        var rank = Float32((comp.member + 1) // 2)
        var lead_ang = angle + Float32(comp.flock_id) * Float32(1.5707963)
        var lead_r = Float32(58.0) + Float32(comp.flock_id) * Float32(18.0)
        var lx = cos(lead_ang) * lead_r + wing_side * rank * Float32(1.35)
        var lz = sin(lead_ang) * lead_r + Float32(28.0) - rank * Float32(1.65)
        var ly = (
            Float32(34.0)
            + Float32(comp.flock_id) * Float32(7.5)
            + sin(time_sec * Float32(1.6) + Float32(comp.member) * Float32(0.3)) * Float32(0.8)
        )
        var heading = -angle
        var qg = Quat(0.0, sin(heading * Float32(0.5)), 0.0, cos(heading * Float32(0.5)))
        return (Vec3(lx, ly, lz), qg)

    var r = comp.radius
    var wx = cos(angle) * r
    var wz = sin(angle) * r + Float32(18.0)
    var gy = max(LAKE_LEVEL, sample_valley_height(wx, wz))
    var wy = gy + Float32(1.35) + Float32(0.55) * sin(time_sec * Float32(1.8) + Float32(comp.index))
    return (Vec3(wx, wy, wz), Quat())


def _step_glider_wisp_kernel(
    _lane: Int,
    _ord: Int,
    _entity: Entity,
    gw_ptr: Pointer[GliderOrWispComponent, MutAnyOrigin],
    lt_ptr: Pointer[LocalTransform, MutAnyOrigin],
    ctx_addr: Int,
):
    var ctx = Pointer[FlockSimCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref gw = gw_ptr[unsafe_offset=0]
    ref lt = lt_ptr[unsafe_offset=0]
    var a = gw.angle + gw.speed * ctx.dt
    gw.angle = a
    var pose = _glider_wisp_pose(gw, a, ctx.time_sec)
    lt.translation = pose[0]
    lt.rotation = pose[1]


def apply_time_of_day_preset(
    mut world: World,
    mut rw: RenderWorld,
    mut renderer: ScalableRenderer,
    mut state: OpenWorldStateResource,
    preset: Int,
):
    if not renderer.uses_vulkan():
        return
    ref ren = renderer.gpu_slot[0]
    bake_cloud_and_star_sky_ibl(ren, rw, preset)
    state.tod_preset = preset

    if preset == 0:
        ren.ibl_intensity = Float32(0.95)
        ren.auto_exposure_key = Float32(0.22)
        ren.bloom_intensity = Float32(0.05)
        ren.enable_bokeh_dof = False
        ren.camera_lens.vignette_strength = Float32(0.16)
        state.tod_label = "HIGH NOON ALPINE SANCTUARY"
        state.light_scale = Float32(0.08)
    elif preset == 1:
        ren.ibl_intensity = Float32(1.10)
        ren.auto_exposure_key = Float32(0.24)
        ren.bloom_intensity = Float32(0.12)
        ren.enable_bokeh_dof = False
        ren.camera_lens.vignette_strength = Float32(0.22)
        state.tod_label = "GOLDEN HOUR SUNSET"
        state.light_scale = Float32(0.42)
    else:
        ren.ibl_intensity = Float32(0.55)
        ren.auto_exposure_key = Float32(0.11)
        ren.bloom_intensity = Float32(0.18)
        ren.enable_bokeh_dof = True
        ren.camera_lens = AaaPhysicalCamera(
            aperture_fstop=2.2,
            shutter_seconds=1.0 / 60.0,
            iso=800.0,
            focal_length_mm=42.0,
            focus_distance_m=22.0,
            diaphragm_blades=6,
            vignette_strength=0.28,
        )
        ren.camera_lens.max_coc_px = Float32(5.5)
        state.tod_label = "STARRY TWILIGHT LTC + BOKEH"
        state.light_scale = Float32(1.0)

    rw.ibl_intensity = ren.ibl_intensity

    var pt_mask = world.bit_for[PointLightComponent]().union(world.bit_for[ScaledLightTag]())
    var pt_ents = world.query_entities(pt_mask)
    for i in range(len(pt_ents)):
        var e = pt_ents[i]
        var lc_opt = world.get_ptr_mut[PointLightComponent](e)
        var tag_opt = world.get_ptr[ScaledLightTag](e)
        if lc_opt and tag_opt:
            ref tag = tag_opt.value()[unsafe_offset=0]
            var sc = max(tag.min_scale, state.light_scale)
            lc_opt.value()[unsafe_offset=0].radiance = tag.base_radiance * sc

    var al_mask = world.bit_for[AreaLightComponent]().union(world.bit_for[ScaledLightTag]())
    var al_ents = world.query_entities(al_mask)
    for i in range(len(al_ents)):
        var e = al_ents[i]
        var ac_opt = world.get_ptr_mut[AreaLightComponent](e)
        var tag_opt = world.get_ptr[ScaledLightTag](e)
        if ac_opt and tag_opt:
            ref tag = tag_opt.value()[unsafe_offset=0]
            var sc = max(tag.min_scale, state.light_scale)
            ac_opt.value()[unsafe_offset=0].light.radiance = tag.base_radiance * sc


# ============================================================================
# 5. Scheduled EngineApp Systems (`STAGE_PRE_UPDATE`, `STAGE_UPDATE`, `STAGE_LAST`)
# ============================================================================

def _sys_open_world_pre_resize(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var win_opt = resources.get_ptr[WindowStateResource]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    if not win_opt or not rend_opt:
        return
    ref win_res = win_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]
    if win_res.resized and win_res.width > 0 and win_res.height > 0:
        _ = renderer.resize(win_res.width, win_res.height)
    if (
        renderer.cpu_canvas.width != renderer.width
        or renderer.cpu_canvas.height != renderer.height
    ):
        renderer.cpu_canvas = SimdCanvas(renderer.width, renderer.height)


def _sys_open_world_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    ref pool = pool_ptr[unsafe_offset=0]

    var st_opt = resources.get_ptr_mut[OpenWorldStateResource]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    var rw_opt = resources.get_ptr_mut[RenderWorld]()
    if not st_opt or not rend_opt or not rw_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]
    ref rw = rw_opt.value()[unsafe_offset=0]
    if not renderer.uses_vulkan():
        return
    ref ren = renderer.gpu_slot[0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        dt = ft_opt.value()[unsafe_offset=0].delta_seconds

    if not state.headless_shots:
        var win_opt = resources.get_ptr_mut[WindowStateResource]()
        var in_opt = resources.get_ptr_mut[InputStateResource]()
        var imap_opt = resources.get_ptr[InputMap]()
        if in_opt and imap_opt:
            ref input_state = in_opt.value()[unsafe_offset=0].input
            ref imap = imap_opt.value()[unsafe_offset=0]
            if imap.just_pressed("quit"):
                if win_opt:
                    win_opt.value()[unsafe_offset=0].request_close()
                return
            if not state.overlays.handle_input(input_state):
                if imap.just_pressed("tod_1"):
                    apply_time_of_day_preset(world, rw, renderer, state, 0)
                if imap.just_pressed("tod_2"):
                    apply_time_of_day_preset(world, rw, renderer, state, 1)
                if imap.just_pressed("tod_3"):
                    apply_time_of_day_preset(world, rw, renderer, state, 2)
                if imap.just_pressed("toggle_bloom"):
                    ren.bloom_intensity = Float32(0.0) if ren.bloom_intensity > Float32(0.01) else Float32(0.14)
                if imap.just_pressed("toggle_bokeh"):
                    ren.enable_bokeh_dof = not ren.enable_bokeh_dof
                if imap.just_pressed("toggle_rtgi"):
                    ren.enable_rt_gi = (not ren.enable_rt_gi) and ren.vk.has_ray_query

                if imap.is_down("look_left"):
                    state.cam_yaw += Float32(1.5) * dt
                if imap.is_down("look_right"):
                    state.cam_yaw -= Float32(1.5) * dt
                if imap.is_down("look_up"):
                    state.cam_pitch = min(Float32(1.25), state.cam_pitch + Float32(1.1) * dt)
                if imap.is_down("look_down"):
                    state.cam_pitch = max(Float32(-1.25), state.cam_pitch - Float32(1.1) * dt)

                var fwd = Vec3(
                    sin(state.cam_yaw) * cos(state.cam_pitch),
                    sin(state.cam_pitch),
                    cos(state.cam_yaw) * cos(state.cam_pitch),
                ).normalized()
                var rgt = fwd.cross(Vec3(0.0, 1.0, 0.0)).normalized()
                var move_spd = Float32(36.0) if imap.is_down("boost") else Float32(14.0)
                if imap.is_down("move_forward"):
                    state.cam_pos = state.cam_pos + fwd * (move_spd * dt)
                if imap.is_down("move_back"):
                    state.cam_pos = state.cam_pos - fwd * (move_spd * dt)
                if imap.is_down("move_right"):
                    state.cam_pos = state.cam_pos + rgt * (move_spd * dt)
                if imap.is_down("move_left"):
                    state.cam_pos = state.cam_pos - rgt * (move_spd * dt)
                if imap.is_down("move_up"):
                    state.cam_pos.y += move_spd * dt
                if imap.is_down("move_down"):
                    state.cam_pos.y -= move_spd * dt
                var min_ground = (
                    max(LAKE_LEVEL + Float32(0.5), sample_valley_height(state.cam_pos.x, state.cam_pos.z))
                    + Float32(1.2)
                )
                if state.cam_pos.y < min_ground:
                    state.cam_pos.y = min_ground

                var cam_lt = world.get_ptr_mut[LocalTransform](state.cam_entity)
                var cam_c = world.get_ptr_mut[CameraComponent](state.cam_entity)
                if cam_lt and cam_c:
                    cam_lt.value()[unsafe_offset=0].translation = state.cam_pos
                    cam_lt.value()[unsafe_offset=0].rotation = Quat()
                    cam_c.value()[unsafe_offset=0].look_target_offset = fwd * Float32(10.0)

    if not state.freeze_sim:
        state.sim_time += dt
        var sim_t0 = get_time()
        var ctx = FlockSimCtx(state.sim_time, dt)
        world.par_for_each2[GliderOrWispComponent, LocalTransform](
            pool, _step_glider_wisp_kernel, Int(Pointer(to=ctx))
        )
        state.last_sim_us = max(1, Int((get_time() - sim_t0) * 1_000_000.0))

        var cur_bones = evaluate_sentinel_bones(state.sim_time)
        var refit_ms = ren.update_skinned_mesh_bones(state.sentinel_skin, cur_bones, state.prev_bones)
        state.prev_bones = cur_bones^
        state.last_blas_refit_us = max(1, Int(refit_ms * 1000.0))
        _ = state.asset_server.tick()


def _interpolated_gt_local(world: World, entity: Entity, alpha: Float32) -> GlobalTransform:
    var gt_opt = world.get_ptr[GlobalTransform](entity)
    if not gt_opt:
        return GlobalTransform()
    ref curr = gt_opt.value()[unsafe_offset=0]
    var prev_opt = world.get_ptr[PrevGlobalTransform](entity)
    if prev_opt:
        return interpolate_global_transform(prev_opt.value()[unsafe_offset=0].value, curr, alpha)
    return curr.copy()


def _sys_open_world_extract_and_draw(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var rw_opt = resources.get_ptr_mut[RenderWorld]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    var rstate_opt = resources.get_ptr_mut[RenderPluginStateResource]()
    if not rw_opt or not rend_opt or not rstate_opt:
        return

    ref render_world = rw_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]
    ref rstate = rstate_opt.value()[unsafe_offset=0]

    var frame_idx = UInt64(0)
    var frame_dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        frame_idx = ft_opt.value()[unsafe_offset=0].frame_index
        frame_dt = ft_opt.value()[unsafe_offset=0].delta_seconds

    var alpha = Float32(1.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt:
        alpha = fix_opt.value()[unsafe_offset=0].overstep_alpha

    rstate.extractor.begin_extraction(render_world, frame_idx, frame_dt)
    if rstate.terrain_clipmap.active:
        render_world.terrain_clipmap = rstate.terrain_clipmap.copy()

    # 1. Active CameraComponent
    var cam_mask = world.bit_for[CameraComponent]().union(world.bit_for[GlobalTransform]())
    var cam_entities = world.query_entities(cam_mask)
    for i in range(len(cam_entities)):
        var e = cam_entities[i]
        var cam_opt = world.get_ptr[CameraComponent](e)
        if not cam_opt or not cam_opt.value()[unsafe_offset=0].active:
            continue
        ref cam = cam_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt_local(world, e, alpha)
        var eye = gt.translation
        var off = cam.look_target_offset
        if off.length_squared() <= Float32(1.0e-8):
            off = Vec3(0.0, 0.0, -1.0)
        var target = eye + gt.rotation.rotate(off)
        var up = gt.rotation.rotate(Vec3(0.0, 1.0, 0.0)).normalized()
        render_world.camera = RenderCameraPacket(
            eye, target, up, cam.fov_y_deg, cam.near_z, cam.far_z, cam.exposure
        )
        break

    # 2. MeshRendererComponents
    var mesh_mask = world.bit_for[MeshRendererComponent]().union(world.bit_for[GlobalTransform]())
    var mesh_entities = world.query_entities(mesh_mask)
    for i in range(len(mesh_entities)):
        var e = mesh_entities[i]
        var mc_opt = world.get_ptr[MeshRendererComponent](e)
        if not mc_opt or not mc_opt.value()[unsafe_offset=0].visible:
            continue
        ref mc = mc_opt.value()[unsafe_offset=0]
        var eff_mat = mc.material_id
        var ov_opt = world.get_ptr[MaterialOverrideComponent](e)
        if ov_opt and ov_opt.value()[unsafe_offset=0].enabled:
            ref ov = ov_opt.value()[unsafe_offset=0]
            if ov.material_id >= 0:
                eff_mat = ov.material_id
            if eff_mat >= 0 and eff_mat < len(renderer.materials):
                renderer.materials[eff_mat].base_color = ov.base_color
                renderer.materials[eff_mat].roughness = ov.roughness
                renderer.materials[eff_mat].metallic = ov.metallic
                renderer.materials[eff_mat].emissive = ov.emissive
                if renderer.uses_vulkan():
                    renderer.gpu_slot[0].update_material_asset(eff_mat, renderer.materials[eff_mat])

        var gt = _interpolated_gt_local(world, e, alpha)
        rstate.extractor.extract_entity(
            render_world,
            e.to_bits(),
            mc.mesh_id,
            gt.matrix.copy(),
            eff_mat,
            mc.cast_shadows,
        )

    # 3. VatCrowdComponents
    var vat_mask = world.bit_for[VatCrowdComponent]().union(world.bit_for[GlobalTransform]())
    var vat_entities = world.query_entities(vat_mask)
    for i in range(len(vat_entities)):
        var e = vat_entities[i]
        var vc_opt = world.get_ptr[VatCrowdComponent](e)
        if not vc_opt or not vc_opt.value()[unsafe_offset=0].visible:
            continue
        ref vc = vc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt_local(world, e, alpha)
        rstate.extractor.extract_vat_crowd_entity(
            render_world,
            e.to_bits(),
            vc.clip_id,
            vc.phase_sec,
            vc.prev_phase_sec,
            gt.matrix.copy(),
            vc.material_id,
        )

    # 4. PointLightComponents
    var lgt_mask = world.bit_for[PointLightComponent]().union(world.bit_for[GlobalTransform]())
    var lgt_entities = world.query_entities(lgt_mask)
    for i in range(len(lgt_entities)):
        var e = lgt_entities[i]
        var lc_opt = world.get_ptr[PointLightComponent](e)
        if not lc_opt or not lc_opt.value()[unsafe_offset=0].enabled:
            continue
        ref lc = lc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt_local(world, e, alpha)
        render_world.cluster_lights.append(
            AaaClusterLight.point(gt.translation, lc.radiance, lc.radius)
        )

    # 5. AreaLightComponents (18 Heitz LTC rect/tube lights)
    var area_mask = world.bit_for[AreaLightComponent]()
    var area_entities = world.query_entities(area_mask)
    for i in range(len(area_entities)):
        var ae_opt = world.get_ptr[AreaLightComponent](area_entities[i])
        if ae_opt:
            render_world.cluster_lights.append(ae_opt.value()[unsafe_offset=0].light.copy())

    var st_opt = resources.get_ptr_mut[OpenWorldStateResource]()
    if st_opt:
        ref state = st_opt.value()[unsafe_offset=0]
        if (not state.headless_shots) and renderer.surface.is_zero_copy_gpu():
            rstate.ui_draw_list.add_rect(
                14.0,
                8.0,
                Float32(max(32, renderer.width - 28)),
                32.0,
                fill_color=0xCC0B131E,
                corner_radius=6.0,
                border_width=1.5,
                border_color=0xFFF49E38,
            )

    rstate.extractor.end_extraction()
    rstate.last_render_ok = renderer.render_world(render_world, rstate.ui_draw_list)
    rstate.rendered_frames += 1
    rstate.ui_draw_list.clear()

    if st_opt and renderer.uses_vulkan():
        ref state = st_opt.value()[unsafe_offset=0]
        if not state.headless_shots:
            ref ren = renderer.gpu_slot[0]
            state.overlays.sync_aaa_renderer(ren)
            if not renderer.surface.is_zero_copy_gpu():
                var win_opt = resources.get_ptr_mut[WindowStateResource]()
                if win_opt and len(win_opt.value()[unsafe_offset=0].windows) > 0:
                    if (
                        renderer.cpu_canvas.width != renderer.width
                        or renderer.cpu_canvas.height != renderer.height
                    ):
                        renderer.cpu_canvas = SimdCanvas(renderer.width, renderer.height)
                    var gpu_px = renderer.readback_pixels()
                    var n_px = min(len(gpu_px), len(renderer.cpu_canvas.pixels))
                    for px_i in range(n_px):
                        renderer.cpu_canvas.pixels[px_i] = gpu_px[px_i]

                    var ae = ren.get_adapted_exposure()
                    var exp_x100 = Int(ae[0] * Float32(100.0))
                    var gpu_tot_us = max(1, Int(ren.render_graph.total_gpu_ms * 1000.0))
                    var pass_cnt = len(ren.render_graph.pass_timings)
                    var bp_nodes = 0
                    var phys_opt = resources.get_ptr[PhysicsWorld]()
                    if phys_opt:
                        bp_nodes = len(phys_opt.value()[unsafe_offset=0].broadphase_nodes)

                    renderer.cpu_canvas.triangles_drawn = ren.last_culled_meshlet_draws * 64
                    renderer.cpu_canvas.sprites_drawn = ENTITY_COUNT
                    state.overlays.tick_us = state.last_sim_us
                    state.overlays.render_us = gpu_tot_us
                    state.overlays.fps = max(1, 1_000_000 // gpu_tot_us)
                    renderer.cpu_canvas.draw_panel(14, 8, max(32, renderer.width - 28), 34, 0x0B131E, 0xF49E38)
                    renderer.cpu_canvas.draw_text(
                        24,
                        16,
                        "FLINT AAA OPEN WORLD  |  "
                        + state.tod_label
                        + "  |  DRAWS: "
                        + String(ren.last_culled_meshlet_draws)
                        + "  |  GPU: "
                        + String(gpu_tot_us)
                        + " US ("
                        + String(pass_cnt)
                        + " PASSES)  |  AE: "
                        + String(exp_x100 // 100)
                        + "."
                        + String((exp_x100 % 100) // 10)
                        + "X  |  BVH8: "
                        + String(bp_nodes)
                        + "  |  BLAS REFIT: "
                        + String(state.last_blas_refit_us)
                        + " US",
                        0xFFE082,
                        1,
                    )
                    state.overlays.render(renderer.cpu_canvas)
                    win_opt.value()[unsafe_offset=0].windows[0].present(
                        renderer.cpu_canvas.pixels, renderer.width, renderer.height
                    )


# ============================================================================
# 6. Scene, Input, Audio & Prefab Population
# ============================================================================

def _setup_open_world_input(mut app: EngineApp):
    var imap_opt = app.resources.get_ptr_mut[InputMap]()
    if not imap_opt:
        return
    ref imap = imap_opt.value()[unsafe_offset=0]
    var actions = [
        "move_forward",
        "move_back",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "boost",
        "look_left",
        "look_right",
        "look_up",
        "look_down",
        "tod_1",
        "tod_2",
        "tod_3",
        "toggle_bloom",
        "toggle_bokeh",
        "toggle_rtgi",
        "quit",
    ]
    for i in range(len(actions)):
        _ = imap.register_action(actions[i], ACTION_BOOL)

    var ctx = InputContext("open_world_fly", priority=10)
    ctx.bind_key("move_forward", 87)
    ctx.bind_key("move_back", 83)
    ctx.bind_key("move_right", 68)
    ctx.bind_key("move_left", 65)
    ctx.bind_key("move_up", 69)
    ctx.bind_key("move_down", 81)
    ctx.bind_key("boost", 340)
    ctx.bind_key("look_left", 263)
    ctx.bind_key("look_right", 262)
    ctx.bind_key("look_up", 265)
    ctx.bind_key("look_down", 264)
    ctx.bind_key("tod_1", 49)
    ctx.bind_key("tod_2", 50)
    ctx.bind_key("tod_3", 51)
    ctx.bind_key("toggle_bloom", 66)
    ctx.bind_key("toggle_bokeh", 70)
    ctx.bind_key("toggle_rtgi", 71)
    ctx.bind_key("quit", 256)
    imap.add_context(ctx^)
    imap.push_context("open_world_fly")


def _setup_open_world_audio(mut app: EngineApp):
    var aud_opt = app.resources.get_ptr_mut[AudioServer]()
    if not aud_opt:
        return
    ref audio_server = aud_opt.value()[unsafe_offset=0]
    var pcm = List[Int16](capacity=48000)
    for s in range(48000):
        var ph = Float32(s) / Float32(48000.0) * Float32(6.2831853)
        var wv = (
            sin(ph * Float32(82.5)) * Float32(0.35)
            + sin(ph * Float32(123.75)) * Float32(0.22)
            + sin(ph * Float32(247.5)) * Float32(0.12)
        )
        pcm.append(Int16(Int(wv * Float32(10000.0))))
    var aid = audio_server.register_pcm16_asset(pcm, channels=1, sample_rate=48000)

    var em_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](em_e, LocalTransform())
    _ = app.world.insert[GlobalTransform](em_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        em_e,
        AudioEmitterComponent(
            asset_id=aid,
            volume=Float32(0.45),
            bus_id=BUS_AMBIENCE,
            looping=True,
            spatial=False,
            play_on_spawn=True,
        ),
    )


def _spawn_static_mesh_entity(mut world: World, mesh_id: Int, mat_id: Int, pos: Vec3, rot: Quat, scl: Vec3):
    var e = world.spawn()
    _ = world.insert[LocalTransform](e, LocalTransform(pos, rot, scl))
    _ = world.insert[GlobalTransform](e, GlobalTransform())
    _ = world.insert[MeshRendererComponent](e, MeshRendererComponent(mesh_id, mat_id))


def _spawn_point_light(
    mut world: World,
    pos: Vec3,
    base_radiance: Vec3,
    radius: Float32,
    light_scale: Float32,
    min_scale: Float32 = 0.0,
):
    var e = world.spawn()
    _ = world.insert[LocalTransform](e, LocalTransform(pos))
    _ = world.insert[GlobalTransform](e, GlobalTransform())
    _ = world.insert[PointLightComponent](
        e, PointLightComponent(base_radiance * max(min_scale, light_scale), radius=radius, enabled=True)
    )
    _ = world.insert[ScaledLightTag](e, ScaledLightTag(base_radiance, min_scale))


def _spawn_area_light(
    mut world: World,
    var light: AaaClusterLight,
    base_radiance: Vec3,
    light_scale: Float32,
    min_scale: Float32 = 0.0,
):
    var e = world.spawn()
    light.radiance = base_radiance * max(min_scale, light_scale)
    _ = world.insert[AreaLightComponent](e, AreaLightComponent(light^))
    _ = world.insert[ScaledLightTag](e, ScaledLightTag(base_radiance, min_scale))


def _populate_open_world_scene(
    mut app: EngineApp,
    headless_shots: Bool,
) -> Bool:
    var rend_opt = app.resources.get_ptr_mut[ScalableRenderer]()
    if not rend_opt:
        return False
    ref renderer = rend_opt.value()[unsafe_offset=0]
    if not renderer.uses_vulkan():
        return False

    print(
        "[Flint Open World] Cooking 512m Sunspire Valley (3-biome ridged peaks, 300 pines, 140 boulders,"
        " viaduct bridge, .flpk v3 glTF Sanctum & Aegis Sentinel)..."
    )
    var t_cook0 = get_time()

    var splat_1k = make_world_biome_splat_1024()
    var tex_meadow = make_biome_textures_256(0)
    var tex_cliff = make_biome_textures_256(1)
    var tex_paver = make_biome_textures_256(3)
    var water_nrm_vecs = make_water_ripple_normals_256()

    var splat_alb = renderer.gpu_slot[0].upload_bc7_texture(splat_1k[0], 1024, 1024, srgb=True, generate_mips=True)
    var splat_orm = renderer.gpu_slot[0].upload_bc7_texture(splat_1k[1], 1024, 1024, srgb=False, generate_mips=True)
    var meadow_alb_pine = renderer.gpu_slot[0].upload_bc7_texture(tex_meadow[0], 256, 256, srgb=True, generate_mips=True)
    var meadow_nrm = renderer.gpu_slot[0].upload_bc5_normal_texture(tex_meadow[1], 256, 256, generate_mips=True)
    var cliff_alb = renderer.gpu_slot[0].upload_bc7_texture(tex_cliff[0], 256, 256, srgb=True, generate_mips=True)
    var cliff_nrm = renderer.gpu_slot[0].upload_bc5_normal_texture(tex_cliff[1], 256, 256, generate_mips=True)
    var cliff_orm = renderer.gpu_slot[0].upload_bc7_texture(tex_cliff[2], 256, 256, srgb=False, generate_mips=True)
    var paver_alb = renderer.gpu_slot[0].upload_bc7_texture(tex_paver[0], 256, 256, srgb=True, generate_mips=True)
    var paver_nrm = renderer.gpu_slot[0].upload_bc5_normal_texture(tex_paver[1], 256, 256, generate_mips=True)
    var paver_orm = renderer.gpu_slot[0].upload_bc7_texture(tex_paver[2], 256, 256, srgb=False, generate_mips=True)
    var water_nrm = renderer.gpu_slot[0].upload_bc5_normal_texture(water_nrm_vecs, 256, 256, generate_mips=True)
    var has_rt = renderer.gpu_slot[0].vk.has_ray_query

    # Register all 10 PBR MaterialAssets through ScalableRenderer
    var mat_terrain = MaterialAsset(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.84, metallic=0.0)
    mat_terrain.set_albedo_slot(splat_alb)
    mat_terrain.set_orm_slot(splat_orm)
    mat_terrain.detail_normal_slot = UInt16(cliff_nrm)
    mat_terrain.detail_uv_scale = Float32(56.0)
    mat_terrain.detail_normal_strength = Float32(0.52)
    var mat_terrain_id = renderer.register_material(mat_terrain)

    var mat_cliff = MaterialAsset(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.76, metallic=0.04)
    mat_cliff.set_albedo_slot(cliff_alb)
    mat_cliff.set_normal_slot(cliff_nrm)
    mat_cliff.set_orm_slot(cliff_orm)
    mat_cliff.detail_normal_slot = UInt16(cliff_nrm)
    mat_cliff.detail_uv_scale = Float32(4.0)
    mat_cliff.detail_normal_strength = Float32(0.45)
    var mat_cliff_id = renderer.register_material(mat_cliff)

    var mat_bridge = MaterialAsset(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.62, metallic=0.04)
    mat_bridge.set_albedo_slot(paver_alb)
    mat_bridge.set_normal_slot(paver_nrm)
    mat_bridge.set_orm_slot(paver_orm)
    var mat_bridge_id = renderer.register_material(mat_bridge)

    var mat_bronze_id = renderer.register_material(
        MaterialAsset(base_color=Vec3(0.78, 0.56, 0.26), roughness=0.24, metallic=0.90)
    )

    var mat_lantern = MaterialAsset(base_color=Vec3(1.0, 0.85, 0.55), roughness=0.15, metallic=0.0)
    mat_lantern.emissive = Vec3(18.0, 11.0, 3.8)
    var mat_lantern_id = renderer.register_material(mat_lantern)

    var mat_lake = MaterialAsset(
        base_color=Vec3(0.012, 0.058, 0.072),
        roughness=0.045,
        metallic=0.0,
        domain=MAT_DOMAIN_CLEARCOAT,
    )
    mat_lake.set_normal_slot(water_nrm)
    mat_lake.detail_normal_slot = UInt16(water_nrm)
    mat_lake.detail_uv_scale = Float32(5.5)
    mat_lake.detail_normal_strength = Float32(0.55)
    mat_lake.clearcoat = Float32(0.92)
    mat_lake.clearcoat_roughness = Float32(0.035)
    mat_lake.iridescence = Float32(0.08)
    mat_lake.iridescence_thickness_nm = Float32(390.0)
    var mat_lake_id = renderer.register_material(mat_lake)

    var mat_bark = MaterialAsset(base_color=Vec3(0.16, 0.10, 0.06), roughness=0.88, metallic=0.0)
    mat_bark.set_normal_slot(cliff_nrm)
    var mat_bark_id = renderer.register_material(mat_bark)

    var mat_pine = MaterialAsset(
        base_color=Vec3(0.32, 0.42, 0.28),
        roughness=0.78,
        metallic=0.0,
        domain=MAT_DOMAIN_TWO_SIDED_FOLIAGE,
        flags=MAT_FLAG_DOUBLE_SIDED | MAT_FLAG_SUBSURFACE,
    )
    mat_pine.set_albedo_slot(meadow_alb_pine)
    mat_pine.set_normal_slot(meadow_nrm)
    mat_pine.detail_normal_slot = UInt16(meadow_nrm)
    mat_pine.detail_uv_scale = Float32(6.0)
    mat_pine.detail_normal_strength = Float32(0.55)
    mat_pine.subsurface_color = Vec3(0.14, 0.32, 0.07)
    mat_pine.subsurface_amount = Float32(0.28)
    var mat_pine_id = renderer.register_material(mat_pine)

    var mat_sentinel = MaterialAsset(
        base_color=Vec3(0.76, 0.66, 0.46),
        roughness=0.22,
        metallic=0.88,
        domain=MAT_DOMAIN_CLEARCOAT,
    )
    mat_sentinel.clearcoat = Float32(0.45)
    mat_sentinel.clearcoat_roughness = Float32(0.08)
    var mat_sentinel_id = renderer.register_material(mat_sentinel)

    var mat_glider_id = renderer.register_material(
        MaterialAsset(base_color=Vec3(0.18, 0.22, 0.28), roughness=0.28, metallic=0.78)
    )

    # Register Terrain, Lake, Bridge, Props & Glider meshes via ScalableRenderer
    for cz in range(CHUNK_GRID):
        for cx in range(CHUNK_GRID):
            var tm_id = renderer.register_mesh(build_terrain_chunk_mesh(cx, cz), mat_terrain_id, build_lods=True)
            _spawn_static_mesh_entity(app.world, tm_id, mat_terrain_id, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0))

    var lake_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.05, metallic=0.0)
    var lv0 = lake_b.add_vertex(Vec3(-160.0, LAKE_LEVEL, -140.0), Vec3(0.0, 1.0, 0.0), 0.0, 0.0)
    var lv1 = lake_b.add_vertex(Vec3(160.0, LAKE_LEVEL, -140.0), Vec3(0.0, 1.0, 0.0), 22.0, 0.0)
    var lv2 = lake_b.add_vertex(Vec3(160.0, LAKE_LEVEL, 180.0), Vec3(0.0, 1.0, 0.0), 22.0, 22.0)
    var lv3 = lake_b.add_vertex(Vec3(-160.0, LAKE_LEVEL, 180.0), Vec3(0.0, 1.0, 0.0), 0.0, 22.0)
    lake_b.add_quad(lv0, lv3, lv2, lv1)
    var lake_mesh_id = renderer.register_mesh(lake_b^.finish(), mat_lake_id, build_lods=False)
    _spawn_static_mesh_entity(app.world, lake_mesh_id, mat_lake_id, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0))

    var bridge_meshes = build_viaduct_bridge_meshes()
    var bridge_stone_id = renderer.register_mesh(bridge_meshes[0], mat_bridge_id, build_lods=True)
    var bridge_bronze_id = renderer.register_mesh(bridge_meshes[1], mat_bronze_id, build_lods=True)
    var bridge_glow_id = renderer.register_mesh(bridge_meshes[2], mat_lantern_id, build_lods=False)
    _spawn_static_mesh_entity(app.world, bridge_stone_id, mat_bridge_id, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0))
    _spawn_static_mesh_entity(app.world, bridge_bronze_id, mat_bronze_id, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0))
    _spawn_static_mesh_entity(app.world, bridge_glow_id, mat_lantern_id, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0))

    var pine_meshes = build_alpine_pine_tree_meshes()
    var pine_trunk_id = renderer.register_mesh(pine_meshes[0], mat_bark_id, build_lods=True)
    var pine_bough_id = renderer.register_mesh(pine_meshes[1], mat_pine_id, build_lods=True)
    var boulder_id = renderer.register_mesh(build_granite_boulder_mesh(), mat_cliff_id, build_lods=True)
    var glider_mesh_id = renderer.register_mesh(build_swept_glider_mesh(), mat_glider_id, build_lods=False)

    var prop_count = 0
    var placed_trees = 0
    var seed_t = 0
    while placed_trees < 300 and seed_t < 4000:
        seed_t += 1
        var rx = (_hash2d_01(seed_t, 17) * Float32(2.0) - Float32(1.0)) * Float32(145.0)
        var rz = (_hash2d_01(seed_t, 53) * Float32(2.0) - Float32(1.0)) * Float32(145.0) + Float32(16.0)
        if abs(rx) < Float32(9.0) and rz > Float32(-38.0) and rz < Float32(36.0):
            continue
        var near_shrine = (
            ((rx + Float32(36.0)) * (rx + Float32(36.0)) + (rz - Float32(42.0)) * (rz - Float32(42.0)) < Float32(64.0))
            or ((rx - Float32(38.0)) * (rx - Float32(38.0)) + (rz - Float32(38.0)) * (rz - Float32(38.0)) < Float32(64.0))
            or ((rx + Float32(44.0)) * (rx + Float32(44.0)) + (rz + Float32(2.0)) * (rz + Float32(2.0)) < Float32(64.0))
            or ((rx - Float32(42.0)) * (rx - Float32(42.0)) + (rz - Float32(6.0)) * (rz - Float32(6.0)) < Float32(64.0))
        )
        if near_shrine:
            continue
        var wy = sample_valley_height(rx, rz)
        var wn = sample_valley_normal(rx, rz)
        if wy > LAKE_LEVEL + Float32(0.55) and wy < Float32(17.5) and wn.y > Float32(0.79):
            var sc = Float32(0.75) + _hash2d_01(seed_t, 91) * Float32(0.72)
            var yaw = _hash2d_01(seed_t, 131) * Float32(6.2831853)
            var qt = Quat(0.0, sin(yaw * Float32(0.5)), 0.0, cos(yaw * Float32(0.5)))
            var pos = Vec3(rx, wy - Float32(0.15), rz)
            var scl = Vec3(sc, sc * Float32(1.08), sc)
            _spawn_static_mesh_entity(app.world, pine_trunk_id, mat_bark_id, pos, qt, scl)
            _spawn_static_mesh_entity(app.world, pine_bough_id, mat_pine_id, pos, qt, scl)
            prop_count += 2
            placed_trees += 1

    var placed_rocks = 0
    var seed_r = 0
    while placed_rocks < 140 and seed_r < 3000:
        seed_r += 1
        var rx = (_hash2d_01(seed_r + 500, 29) * Float32(2.0) - Float32(1.0)) * Float32(115.0)
        var rz = (_hash2d_01(seed_r + 500, 71) * Float32(2.0) - Float32(1.0)) * Float32(115.0) + Float32(12.0)
        if abs(rx) < Float32(5.5) and rz > Float32(-34.0) and rz < Float32(30.0):
            continue
        var wy = sample_valley_height(rx, rz)
        if wy > LAKE_LEVEL - Float32(0.45) and wy < Float32(14.0):
            var sc = Float32(0.65) + _hash2d_01(seed_r + 500, 113) * Float32(1.45)
            var yaw = _hash2d_01(seed_r + 500, 173) * Float32(6.2831853)
            var qr = Quat(0.0, sin(yaw * Float32(0.5)), 0.0, cos(yaw * Float32(0.5)))
            _spawn_static_mesh_entity(
                app.world,
                boulder_id,
                mat_cliff_id,
                Vec3(rx, wy - Float32(0.25), rz),
                qr,
                Vec3(sc, sc * Float32(0.82), sc),
            )
            prop_count += 1
            placed_rocks += 1

    # Package & stream Sunspire Sanctum via flpk_v3 + AssetServer, instantiate via Prefab + Scene
    var asset_server = AssetServer(enable_watcher=False)
    var sanctum_model = build_and_pack_sunspire_sanctum_flpk("build/open_world_assets.flpk", asset_server)
    var sanctum_gpu = renderer.gpu_slot[0].upload_native_model(sanctum_model, build_lods=True)

    var comp_reg = ComponentRegistry.with_builtins()
    register_render_codecs(comp_reg)
    var sanctum_prefab = Prefab("SunspireSanctum")
    var s_root = sanctum_prefab.add_node("sanctum_root")
    sanctum_prefab.set_component(s_root, LocalTransform())
    sanctum_prefab.set_component(s_root, GlobalTransform())
    if len(sanctum_gpu.mesh_entries) >= 3:
        var s_col = sanctum_prefab.add_node("rotunda_colonnade", parent_slot=s_root)
        sanctum_prefab.set_component(s_col, LocalTransform())
        sanctum_prefab.set_component(s_col, GlobalTransform())
        sanctum_prefab.set_component(
            s_col, MeshRendererComponent(sanctum_gpu.mesh_entries[0].mesh_id, sanctum_gpu.material_ids[0])
        )
        var s_dome = sanctum_prefab.add_node("gilded_cupola", parent_slot=s_root)
        sanctum_prefab.set_component(s_dome, LocalTransform(Vec3(0.0, 7.80, 0.0)))
        sanctum_prefab.set_component(s_dome, GlobalTransform())
        sanctum_prefab.set_component(
            s_dome, MeshRendererComponent(sanctum_gpu.mesh_entries[1].mesh_id, sanctum_gpu.material_ids[1])
        )
        var s_core = sanctum_prefab.add_node("celestial_crystal", parent_slot=s_root)
        sanctum_prefab.set_component(s_core, LocalTransform(Vec3(0.0, 2.85, 0.0)))
        sanctum_prefab.set_component(s_core, GlobalTransform())
        sanctum_prefab.set_component(
            s_core, MeshRendererComponent(sanctum_gpu.mesh_entries[2].mesh_id, sanctum_gpu.material_ids[2])
        )

    var isl_y = max(LAKE_LEVEL + Float32(0.35), sample_valley_height(0.0, 24.0) - Float32(0.15))
    var sanctum_places = [
        Vec3(0.0, isl_y, 24.0),
        Vec3(-36.0, max(LAKE_LEVEL + Float32(0.4), sample_valley_height(-36.0, 42.0) - Float32(0.2)), 42.0),
        Vec3(38.0, max(LAKE_LEVEL + Float32(0.4), sample_valley_height(38.0, 38.0) - Float32(0.2)), 38.0),
        Vec3(-44.0, max(LAKE_LEVEL + Float32(0.4), sample_valley_height(-44.0, -2.0) - Float32(0.2)), -2.0),
        Vec3(42.0, max(LAKE_LEVEL + Float32(0.4), sample_valley_height(42.0, 6.0) - Float32(0.2)), 6.0),
    ]
    var sanctum_scales = [Float32(1.12), Float32(0.58), Float32(0.58), Float32(0.58), Float32(0.58)]
    for p_i in range(len(sanctum_places)):
        try:
            var ents = sanctum_prefab.instantiate(app.world, comp_reg)
            if len(ents) > 0:
                var lt_opt = app.world.get_ptr_mut[LocalTransform](ents[0])
                if lt_opt:
                    var sc = sanctum_scales[p_i]
                    lt_opt.value()[unsafe_offset=0] = LocalTransform(
                        sanctum_places[p_i], Quat(), Vec3(sc, sc, sc)
                    )
        except err:
            print("[Flint Open World] WARNING: SunspireSanctum prefab instantiate failed:", err)

    # Upload 4-bone skinned Aegis Sentinel knight + register AnimAssetCache model
    var sentinel_skin = upload_aegis_sentinel_knight(renderer.gpu_slot[0], mat_sentinel_id)
    var anim_cache_opt = app.resources.get_ptr_mut[AnimAssetCache]()
    if anim_cache_opt:
        var sent_anim_model = NativeModel()
        for b_i in range(4):
            var bn = NativeNode()
            bn.name = "sentinel_bone_" + String(b_i)
            if b_i > 0:
                sent_anim_model.nodes[0].children.append(b_i)
            sent_anim_model.nodes.append(bn^)
        sent_anim_model.roots.append(0)
        anim_cache_opt.value()[unsafe_offset=0].register_model(UInt64(1), sent_anim_model^)

    var sent_e = app.world.spawn()
    var yaw_s = Float32(3.14159265)
    var q_sent = Quat(0.0, sin(yaw_s * Float32(0.5)), 0.0, cos(yaw_s * Float32(0.5)))
    _ = app.world.insert[LocalTransform](sent_e, LocalTransform(Vec3(-1.15, 1.65, 10.5), q_sent, Vec3(0.62, 0.62, 0.62)))
    _ = app.world.insert[GlobalTransform](sent_e, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](
        sent_e, MeshRendererComponent(sentinel_skin.mesh_entry.mesh_id, mat_sentinel_id)
    )
    _ = app.world.insert[SkeletonComponent](sent_e, SkeletonComponent(model_id=UInt64(1)))

    # 256 Parallel ECS + PhysicsPlugin Gliders & Lantern Wisps
    var init_light_scale = Float32(0.08)
    for i in range(ENTITY_COUNT):
        var e = app.world.spawn()
        var is_glider = i < 192
        var ang = Float32(i) * Float32(0.1963495)
        var rad = Float32(7.5) + Float32(i % 32) * Float32(1.15)
        var spd = Float32(0.18) if is_glider else (Float32(0.25) + Float32(i % 11) * Float32(0.03))
        var gw = GliderOrWispComponent(i, is_glider, i // 48, i % 48, ang, rad, spd)
        var pose = _glider_wisp_pose(gw, ang, Float32(0.0))
        _ = app.world.insert[GliderOrWispComponent](e, gw)
        _ = app.world.insert[LocalTransform](
            e, LocalTransform(pose[0], pose[1], Vec3(1.1, 1.1, 1.1) if is_glider else Vec3(1.0, 1.0, 1.0))
        )
        _ = app.world.insert[GlobalTransform](e, GlobalTransform())
        _ = app.world.insert[RigidBodyComponent](e, RigidBodyComponent.kinematic())
        _ = app.world.insert[ColliderComponent](e, ColliderComponent.sphere(Float32(1.8)))
        if is_glider:
            _ = app.world.insert[MeshRendererComponent](e, MeshRendererComponent(glider_mesh_id, mat_glider_id))
        else:
            var w_i = i - 192
            var warm = (w_i % 3) != 0
            var lcol = Vec3(4.5, 2.6, 0.75) if warm else Vec3(0.8, 3.4, 5.2)
            _ = app.world.insert[PointLightComponent](
                e, PointLightComponent(lcol * init_light_scale, radius=Float32(10.5), enabled=True)
            )
            _ = app.world.insert[ScaledLightTag](e, ScaledLightTag(lcol, Float32(0.0)))

    # 110 Perimeter Torch Point Lights
    for p_i in range(110):
        var ang = Float32(p_i) * Float32(6.2831853 / 110.0)
        var rad = Float32(36.0) + Float32(p_i % 3) * Float32(14.0)
        var lx = cos(ang) * rad
        var lz = sin(ang) * rad + Float32(16.0)
        var ly = max(LAKE_LEVEL + Float32(1.2), sample_valley_height(lx, lz) + Float32(1.8))
        _spawn_point_light(app.world, Vec3(lx, ly, lz), Vec3(3.2, 1.8, 0.6), Float32(11.0), init_light_scale)

    # 18 Heitz LTC Rectangular & Tube Area Lights
    for a_i in range(5):
        var z_pier = Float32(-21.5) + Float32(a_i) * Float32(8.0)
        _spawn_area_light(
            app.world,
            AaaClusterLight.rect_ltc(
                center=Vec3(0.0, 4.30, z_pier),
                right_dir=Vec3(1.0, 0.0, 0.0),
                up_dir=Vec3(0.0, 1.0, 0.0),
                half_width=2.4,
                half_height=0.22,
                radiance=Vec3(12.0, 7.2, 2.4),
                range_radius=18.0,
                two_sided=True,
            ),
            Vec3(12.0, 7.2, 2.4),
            init_light_scale,
        )
        _spawn_area_light(
            app.world,
            AaaClusterLight.tube_ltc(
                start_pos=Vec3(-2.62, 2.6, z_pier),
                end_pos=Vec3(-2.62, 4.5, z_pier),
                tube_radius=0.14,
                radiance=Vec3(15.0, 9.2, 3.2),
                range_radius=16.0,
            ),
            Vec3(15.0, 9.2, 3.2),
            init_light_scale,
        )
        _spawn_area_light(
            app.world,
            AaaClusterLight.tube_ltc(
                start_pos=Vec3(2.62, 2.6, z_pier),
                end_pos=Vec3(2.62, 4.5, z_pier),
                tube_radius=0.14,
                radiance=Vec3(15.0, 9.2, 3.2),
                range_radius=16.0,
            ),
            Vec3(15.0, 9.2, 3.2),
            init_light_scale,
        )

    _spawn_area_light(
        app.world,
        AaaClusterLight.rect_ltc(
            center=Vec3(0.0, isl_y + Float32(4.5), 24.0),
            right_dir=Vec3(1.0, 0.0, 0.0),
            up_dir=Vec3(0.0, 1.0, 0.0),
            half_width=1.8,
            half_height=1.8,
            radiance=Vec3(16.0, 10.5, 3.8),
            range_radius=34.0,
            two_sided=True,
        ),
        Vec3(16.0, 10.5, 3.8),
        init_light_scale,
        min_scale=Float32(0.35),
    )
    _spawn_area_light(
        app.world,
        AaaClusterLight.tube_ltc(
            start_pos=Vec3(-6.5, LAKE_LEVEL + Float32(0.8), 21.0),
            end_pos=Vec3(-6.5, LAKE_LEVEL + Float32(3.6), 21.0),
            tube_radius=0.16,
            radiance=Vec3(2.2, 10.5, 16.5),
            range_radius=20.0,
        ),
        Vec3(2.2, 10.5, 16.5),
        init_light_scale,
    )
    _spawn_area_light(
        app.world,
        AaaClusterLight.tube_ltc(
            start_pos=Vec3(6.5, LAKE_LEVEL + Float32(0.8), 21.0),
            end_pos=Vec3(6.5, LAKE_LEVEL + Float32(3.6), 21.0),
            tube_radius=0.16,
            radiance=Vec3(2.2, 10.5, 16.5),
            range_radius=20.0,
        ),
        Vec3(2.2, 10.5, 16.5),
        init_light_scale,
    )

    var cook_ms = Int((get_time() - t_cook0) * 1000.0)
    var mesh_cnt = len(renderer.gpu_slot[0].geom.meshes)
    var meshlet_cnt = len(renderer.gpu_slot[0].geom.meshlets)
    var tex_cnt = len(renderer.gpu_slot[0].bindless_images)
    var workers = max(1, app.config.worker_threads)

    var state = OpenWorldStateResource(headless_shots, asset_server^, sentinel_skin^, workers)
    state.overlays.active_backend_name = "VULKAN-AAA-RT" if has_rt else "VULKAN-AAA"

    # Camera + Audio Listener entity (no PrevGlobalTransform so shot teleports never interpolate)
    var cam_e = app.world.spawn()
    var look_dir = Vec3(
        sin(state.cam_yaw) * cos(state.cam_pitch),
        sin(state.cam_pitch),
        cos(state.cam_yaw) * cos(state.cam_pitch),
    ).normalized()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(state.cam_pos))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[CameraComponent](
        cam_e,
        CameraComponent(
            fov_y_deg=Float32(54.0),
            near_z=Float32(0.20),
            far_z=Float32(1000.0),
            exposure=Float32(1.0),
            active=True,
            look_target_offset=look_dir * Float32(10.0),
        ),
    )
    _ = app.world.insert[AudioListenerComponent](cam_e, AudioListenerComponent(active=True))
    state.cam_entity = cam_e

    app.insert_resource[OpenWorldStateResource](state^)

    print(
        "[Flint Open World] World ready in "
        + String(cook_ms)
        + " ms (Meshes="
        + String(mesh_cnt)
        + ", Meshlets="
        + String(meshlet_cnt)
        + ", Props="
        + String(prop_count)
        + ", BindlessTex="
        + String(tex_cnt)
        + ", Workers="
        + String(workers)
        + ")"
    )
    return True


def _run_headless_shots(mut app: EngineApp):
    var shot_paths = [
        String("build/shots/open_world_noon_vista.png"),
        String("build/shots/open_world_golden_citadel.png"),
        String("build/shots/open_world_night_ltc_bokeh.png"),
    ]
    var e0_y = max(Float32(15.5), sample_valley_height(-22.0, -34.0) + Float32(8.5))
    var e1_y = max(Float32(4.8), sample_valley_height(13.5, -12.0) + Float32(3.6))
    var cam_eyes = [
        Vec3(-22.0, e0_y, -34.0),
        Vec3(13.5, e1_y, -12.0),
        Vec3(0.85, 3.35, -11.5),
    ]
    var cam_targets = [
        Vec3(2.0, 5.2, 24.0),
        Vec3(-1.0, 5.6, 24.0),
        Vec3(-0.35, 4.2, 18.0),
    ]

    _ = make_dirs("build/shots")
    app.startup()

    var gw_mask = app.world.bit_for[GliderOrWispComponent]().union(app.world.bit_for[LocalTransform]())
    for s_idx in range(3):
        var t_sim = Float32(1.8) + Float32(s_idx) * Float32(2.4)
        var st_opt = app.resources.get_ptr_mut[OpenWorldStateResource]()
        var rend_opt = app.resources.get_ptr_mut[ScalableRenderer]()
        var rw_opt = app.resources.get_ptr_mut[RenderWorld]()
        if not st_opt or not rend_opt or not rw_opt:
            break
        ref state = st_opt.value()[unsafe_offset=0]
        ref renderer = rend_opt.value()[unsafe_offset=0]
        ref rw = rw_opt.value()[unsafe_offset=0]
        ref ren = renderer.gpu_slot[0]

        apply_time_of_day_preset(app.world, rw, renderer, state, s_idx)
        ren.auto_exposure_speed = Float32(1.0)
        state.sim_time = t_sim

        # Position gliders, wisps, sentinel bones, and camera cleanly at t_sim
        var sim_t0 = get_time()
        var ctx = FlockSimCtx(t_sim, Float32(0.5))
        app.world.par_for_each2[GliderOrWispComponent, LocalTransform](
            app.pool, _step_glider_wisp_kernel, Int(Pointer(to=ctx))
        )
        _ = gw_mask
        state.last_sim_us = max(1, Int((get_time() - sim_t0) * 1_000_000.0))

        var cur_bones = evaluate_sentinel_bones(t_sim)
        var refit_ms = ren.update_skinned_mesh_bones(state.sentinel_skin, cur_bones, state.prev_bones)
        state.prev_bones = cur_bones^
        state.last_blas_refit_us = max(1, Int(refit_ms * 1000.0))

        var cam_lt = app.world.get_ptr_mut[LocalTransform](state.cam_entity)
        var cam_c = app.world.get_ptr_mut[CameraComponent](state.cam_entity)
        if cam_lt and cam_c:
            cam_lt.value()[unsafe_offset=0].translation = cam_eyes[s_idx]
            cam_lt.value()[unsafe_offset=0].rotation = Quat()
            cam_c.value()[unsafe_offset=0].look_target_offset = cam_targets[s_idx] - cam_eyes[s_idx]

        # Accumulate 5 temporal TSR subframes (`5 * 8 = 40` ray-traced GI rays/pixel)
        state.freeze_sim = True
        for sub in range(5):
            ren.tsr_history_weight = Float32(0.0) if sub == 0 else Float32(0.80)
            app.step_one_frame(Float32(1.0 / 60.0) if sub == 0 else Float32(0.0))

        var px = renderer.readback_pixels()
        _ = write_png(shot_paths[s_idx], px, renderer.width, renderer.height)
        var bp_nodes = 0
        var phys_opt = app.resources.get_ptr[PhysicsWorld]()
        if phys_opt:
            bp_nodes = len(phys_opt.value()[unsafe_offset=0].broadphase_nodes)
        print(
            "  [SHOT "
            + String(s_idx + 1)
            + "/3] "
            + shot_paths[s_idx]
            + " | "
            + state.tod_label
            + " | CulledDraws="
            + String(ren.last_culled_meshlet_draws)
            + " | BroadphaseNodes="
            + String(bp_nodes)
            + " | "
            + ren.render_graph.format_gpu_profile_summary()
        )

    app.shutdown()


def main():
    var headless_shots = False
    for i in range(1, len(argv())):
        var arg = String(argv()[i])
        if arg == "--shots" or arg == "--headless":
            headless_shots = True

    var no_display = (
        env_flag("FLINT_HEADLESS")
        or (
            env_value("FLINT_BACKEND").byte_length() == 0
            and env_value("WAYLAND_DISPLAY").byte_length() == 0
            and env_value("DISPLAY").byte_length() == 0
            and not on_virtual_console()
        )
    )
    if (not headless_shots) and no_display:
        print(
            "[Flint Open World] No Wayland/X11 display detected. Run with '--shots' to render open-world PNGs."
        )
        return

    var out_w = 1280
    var out_h = 720

    var cfg = AppConfig(
        title="FLINT AAA OPEN WORLD - SUNSPIRE VALLEY",
        headless=headless_shots,
        fixed_hz=Float32(60.0),
    )
    var app = EngineApp(cfg^)

    # Pre-insert 256-cubemap ScalableRenderer BEFORE RenderPlugin
    var sr = ScalableRenderer(tier=RENDER_TIER_0_HEADLESS_SIMD, width=out_w, height=out_h, tsr_scale=1.0)
    var gpu = FlintAaaVulkanRenderer(
        width=out_w,
        height=out_h,
        tsr_scale=1.0,
        max_draws=65536,
        max_instances=16384,
        env_cube_dim=256,
    )
    if not gpu.active:
        gpu.shutdown()
        print("[Flint Open World] ERROR: Vulkan 1.3 hardware driver unavailable.")
        return

    gpu.display_mode = AAA_DISPLAY_AGX_REC709
    gpu.enable_vsm = True
    gpu.vsm_scene_center = Vec3(0.0, 2.0, 12.0)
    gpu.vsm_ortho_half_extent = Float32(128.0)
    gpu.enable_rt_shadows = gpu.vk.has_ray_query
    gpu.enable_rt_reflections = gpu.vk.has_ray_query
    gpu.enable_rt_gi = gpu.vk.has_ray_query
    gpu.gi_rays_per_pixel = 8
    gpu.svgf_passes = 0
    gpu.enable_auto_exposure = True
    gpu.auto_exposure_speed = Float32(0.65)
    gpu.max_lod_error_px = Float32(0.85)
    sr.gpu_slot.append(gpu^)
    sr.tier = RENDER_TIER_2_AAA_VISBUFFER_RT
    sr.requested_tier = RENDER_TIER_2_AAA_VISBUFFER_RT
    app.insert_resource[ScalableRenderer](sr^)

    app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=True))
    app.add_plugin(
        WindowPlugin(
            headless=headless_shots,
            width=out_w,
            height=out_h,
            title="FLINT AAA OPEN WORLD - SUNSPIRE VALLEY",
        )
    )
    app.add_plugin(InputPlugin(enable_evdev=not headless_shots))
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=headless_shots))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=out_w, height=out_h, tsr_scale=1.0))

    # Override RenderPlugin's STAGE_LAST extract_and_draw fn_ptr to include 18 Heitz LTC AreaLightComponents + HUD
    for s_i in range(len(app.schedule.stages[STAGE_LAST].systems)):
        if app.schedule.stages[STAGE_LAST].systems[s_i].name == "flint.render.extract_and_draw":
            app.schedule.stages[STAGE_LAST].systems[s_i].fn_ptr = _sys_open_world_extract_and_draw
            break

    _ = app.register_component[GliderOrWispComponent]()
    _ = app.register_component[AreaLightComponent]()
    _ = app.register_component[ScaledLightTag]()

    _ = app.add_system(STAGE_PRE_UPDATE, "open_world.pre_resize", _sys_open_world_pre_resize, exclusive=True)
    _ = app.add_system(STAGE_UPDATE, "open_world.update", _sys_open_world_update, exclusive=True)

    _setup_open_world_input(app)
    _setup_open_world_audio(app)
    if not _populate_open_world_scene(app, headless_shots):
        print("[Flint Open World] ERROR: Failed to populate Sunspire Valley scene.")
        app.shutdown()
        return

    if headless_shots:
        _run_headless_shots(app)
        return

    app.startup()
    var win_open = False
    var win_opt = app.resources.get_ptr_mut[WindowStateResource]()
    var rend_opt = app.resources.get_ptr_mut[ScalableRenderer]()
    if win_opt:
        ref win_res = win_opt.value()[unsafe_offset=0]
        win_open = win_res.is_open()
        if win_open and rend_opt and len(win_res.windows) > 0:
            _ = rend_opt.value()[unsafe_offset=0].attach_window_surface(win_res.windows[0])

    if not win_open:
        print(
            "[Flint Open World] No Wayland/X11 display detected. Run with '--shots' to render open-world PNGs."
        )
        app.shutdown()
        return

    var st_opt = app.resources.get_ptr_mut[OpenWorldStateResource]()
    var rw_opt = app.resources.get_ptr_mut[RenderWorld]()
    if st_opt and rend_opt and rw_opt:
        apply_time_of_day_preset(
            app.world,
            rw_opt.value()[unsafe_offset=0],
            rend_opt.value()[unsafe_offset=0],
            st_opt.value()[unsafe_offset=0],
            0,
        )

    var t_last = get_time()
    while True:
        var open_now = False
        var wo = app.resources.get_ptr[WindowStateResource]()
        if wo:
            open_now = wo.value()[unsafe_offset=0].is_open()
        if not open_now:
            break
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - t_last)))
        t_last = now
        app.step_one_frame(dt)

    app.shutdown()
