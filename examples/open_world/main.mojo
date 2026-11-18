"""Flint Engine AAA Open-World Feature Sandbox (`examples/open_world/main.mojo`).

A `512m x 512m` art-directed Alpine Sanctuary ("Sunspire Valley") exercising every Flint Engine
subsystem end-to-end at native `1280 x 720` Vulkan 1.3/1.4 hardware ray-traced quality:
  - **Multi-Biome Ridged-Multifractal Terrain (`48` Sub-Meshes, `QEM` `LOD 0..3` DAG + `BLAS`)**:
    Stratified dark basalt cliffs, lush emerald alpine meadows, snow-crowned glacier peaks,
    carved viaduct stone bridge, `300` instanced subsurface-lit pine trees & `140` granite boulders.
  - **Bindless `BC7 / BC5 / BC6H` Materials + `.flpk` VFS + glTF 2.0 `.glb` Bridge (`V17, V19`)**:
    `256x256` procedural `BC7_SRGB` albedo, `BC5_UNORM` erosion/ripple normals & `BC7_UNORM` ORM,
    plus a classical 12-column domed `Sunspire Sanctum` packed into `build/open_world_assets.flpk`
    with Baseline JPEG & PNG textures and streamed via `pack_read` + `ren.upload_native_model(...)`.
  - **4-Bone Dual-Quaternion Skinned `Aegis Sentinel` Knight + Live `BLAS` Refit (`V18`)**:
    Sculpted armored guardian with shield & halberd deformed on GPU (`aaa_dq_skinning_shader`)
    with zero-allocation `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR` `BLAS` refit.
  - **Parallel `SimdColumn` ECS + `JobPool` + `Broadphase` (`256` Swept-Wing Gliders & Lantern Wisps)**:
    `256` autonomous entities simulated across `JobPool` workers with spatial-hash `Broadphase`
    (`find_pairs`) alongside `192` Clustered Punctual + Heitz `LTC` Rectangular & Tube Area Lights.
  - **Bruneton Atmosphere + Volumetric Cloud/Star `256x256` `BC6H` Sky + Cinema Post (`V18, V19`)**:
    Dynamic Time-of-Day (`Alpine Noon`, `Golden Sunset`, `Starry Twilight`) with `16k` `VSM + PCSS`,
    `2`-bounce `SPV_KHR_ray_query` Lumen Surface-Cache GI, `64`-bin Auto-Exposure, `5`-mip Dual-Kawase
    HDR Bloom, `6`-blade Polygonal Bokeh DoF, `AgX Punchy` Cinema Grade, and `VkQueryPool` GPU Profiler!

Usage:
  - Interactive (Wayland / X11): `bash tools/flint.sh world`
    Controls: `WASD` + `Q/E` fly, `Arrows` look, `Shift` boost, `1/2/3` Noon/Sunset/Night,
              `B` Bloom, `F` Bokeh DoF, `G` RT GI, `~` Console, `F2` GPU Profiler, `F3` Memory, `Esc` quit.
  - Headless 3-Vista Cinema Showcase & Benchmark: `bash tools/flint.sh world --shots`
"""
from std.ffi import external_call
from std.math import cos, exp, floor, max, min, sin, sqrt
from std.sys import argv
from flint.core.ecs import ArchetypeRegistry, ArchetypeSlots, SimdColumn, run_parallel_ctx
from flint.core.jobs import JobPool, default_worker_count
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.physics.broadphase import BroadphaseInput, find_pairs
from flint.physics.filter import CollisionFilter
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.hdr_image import HdrImage
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeModel, NativeNode, NativeTexture
from flint.assets.vfs import encode_pack, pack_read
from flint.platform.audio import NativeAudio, WavClip
from flint.platform.input import InputState
from flint.platform.os import get_time, write_bytes
from flint.platform.wayland import WaylandWindow
from flint.platform.x11 import X11Window
from flint.devtools.overlays import FlintDevOverlays
from flint.render.simd_canvas import SimdCanvas
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_ibl import bake_hdr_image_ibl_cubemap_and_sh9
from flint.render.aaa_oit_volumetrics import AaaVolumetricPlume
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709, AaaPhysicalCamera
from flint.render.aaa_renderer import AaaSkinnedMeshEntry, AaaUploadedModel, FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import DualQuat
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams, BrunetonSkyLuts

comptime WORLD_HALF_EXTENT: Float32 = 256.0
comptime CHUNK_GRID: Int = 4
comptime CHUNK_VERTS_SIDE: Int = 45
comptime ENTITY_COUNT: Int = 256
comptime ENTITY_CHUNK: Int = 64
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
        var n = Float32(1.0) - abs(_vnoise2(wx * freq + Float32(17.3), wz * freq - Float32(9.1)) * Float32(2.0) - Float32(1.0))
        var signal = n * n * weight
        weight = max(Float32(0.25), min(Float32(1.0), signal * Float32(1.85)))
        sum += signal * amp
        freq *= Float32(2.07)
        amp *= Float32(0.48)
    return sum


@always_inline
def sample_valley_height(wx: Float32, wz: Float32) -> Float32:
    """Sculpted alpine glacial valley:
    - Deep turquoise glacial lake around `(0, 14)` (`r ~ 42m`), with a central rocky sanctum island at `(0, 24)`
    - Gentle forested meadow terraces (`r in 42..95m`)
    - Towering ridged-multifractal alpine peaks framing the horizon (`+22m .. +68m`) with a scenic sunset pass!
    """
    var dx_l = wx / Float32(46.0)
    var dz_l = (wz - Float32(12.0)) / Float32(38.0)
    var lake_r2 = dx_l * dx_l + dz_l * dz_l

    # Central Sanctum Island rising smoothly out of the lake at `(0, 24)`
    var dx_i = wx / Float32(10.5)
    var dz_i = (wz - Float32(24.0)) / Float32(10.5)
    var isl_r2 = dx_i * dx_i + dz_i * dz_i
    var island_mound = Float32(3.6) * exp(-isl_r2 * Float32(1.35))

    # Approach plaza shore at `(0, -28)`
    var dx_s = wx / Float32(16.0)
    var dz_s = (wz + Float32(28.0)) / Float32(14.0)
    var shore_mound = Float32(3.4) * exp(-(dx_s * dx_s + dz_s * dz_s) * Float32(0.9))

    var ridge = _ridged_fbm(wx, wz)
    var foothills = (_vnoise2(wx * Float32(0.025), wz * Float32(0.025)) - Float32(0.35)) * Float32(6.5)

    # Carve central valley corridor while framing left/right & far mountain amphitheater
    var valley_wall = max(Float32(0.0), lake_r2 - Float32(0.85))
    var mtn_mask = min(Float32(1.0), valley_wall * Float32(0.42))
    var far_dist = sqrt(wx * wx + (wz - Float32(10.0)) * (wz - Float32(10.0))) / WORLD_HALF_EXTENT
    var rim = max(Float32(0.0), far_dist - Float32(0.26))

    # Carve a dramatic V-shaped mountain pass at `(wx ~ +18, wz > 60)` so the Golden-Hour sun beams through!
    var pass_dx = (wx - Float32(22.0)) / Float32(48.0)
    var pass_cut = Float32(1.0) - Float32(0.46) * exp(-(pass_dx * pass_dx)) * (Float32(1.0) if wz > Float32(30.0) else Float32(0.0))

    var mtn_h = (ridge * Float32(52.0) + max(Float32(0.0), foothills)) * mtn_mask * pass_cut + rim * rim * Float32(44.0) * pass_cut
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
    """Builds one watertight `(CHUNK_VERTS_SIDE x CHUNK_VERTS_SIDE)` open-world terrain sector
    with `[0..1]` world-space UVs (`u = (wx + 256) / 512, v = (wz + 256) / 512`) so `QEM` `LOD 0..3`
    simplification has zero biome-split cracks and samples the `1024x1024` continuous biome splat!"""
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
    """Bakes a seamless `1024 x 1024` (`0.5m`/texel) continuous multi-biome world-space `BC7` Albedo + ORM map:
    smoothly blends wet dark lakebed pebbles, sandy gravel shoreline, lush emerald/sunlit alpine meadow,
    fractured dark basalt cliffs (`slope`), and wind-drifted sparkling glacier snowcaps (`altitude + aspect`)!"""
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

            # 1. Lush Alpine Meadow (deep emerald clover + sunlit warm olive grass blades)
            var meadow_r = Float32(46.0) + n_med * Float32(34.0) + n_fine * Float32(16.0)
            var meadow_g = Float32(84.0) + n_med * Float32(44.0) + n_fine * Float32(18.0)
            var meadow_b = Float32(34.0) + n_med * Float32(18.0) + n_fine * Float32(10.0)
            var rough_f = Float32(212.0) - n_fine * Float32(24.0)
            var ao_f = Float32(190.0) + n_fine * Float32(60.0)

            # 2. Wet Gravel & Sandy Shoreline Transition (`wy in [LAKE_LEVEL - 1.5 .. LAKE_LEVEL + 0.9]`)
            var shore_w = Float32(1.0) - _smoothstep(LAKE_LEVEL - Float32(0.1), LAKE_LEVEL + Float32(1.05), wy + (n_med - Float32(0.5)) * Float32(0.4))
            if shore_w > Float32(0.001):
                var wet = Float32(1.0) - _smoothstep(LAKE_LEVEL - Float32(0.3), LAKE_LEVEL + Float32(0.35), wy)
                var sr = (Float32(118.0) + n_fine * Float32(28.0)) * (Float32(1.0) - wet * Float32(0.52))
                var sg = (Float32(108.0) + n_fine * Float32(26.0)) * (Float32(1.0) - wet * Float32(0.48))
                var sb = (Float32(92.0) + n_fine * Float32(24.0)) * (Float32(1.0) - wet * Float32(0.42))
                meadow_r = meadow_r * (Float32(1.0) - shore_w) + sr * shore_w
                meadow_g = meadow_g * (Float32(1.0) - shore_w) + sg * shore_w
                meadow_b = meadow_b * (Float32(1.0) - shore_w) + sb * shore_w
                rough_f = rough_f * (Float32(1.0) - shore_w) + (Float32(165.0) - wet * Float32(95.0)) * shore_w

            # 3. Stratified Dark Slate/Basalt Cliff Rock (`wn.y < 0.82`)
            var cliff_w = Float32(1.0) - _smoothstep(Float32(0.68), Float32(0.84), wn.y + (n_med - Float32(0.5)) * Float32(0.08))
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

            # 4. Sparkling Wind-Swept Alpine Glacier Snowpack (`wy > 14.5m` on non-vertical slopes!)
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
    """Generates `256x256` seamless PBR `(albedo_argb, tangent_normals, orm_argb)` textures:
    - `mode == 0`: Lush Alpine Meadow Grass, Clover & Wildflower Moss.
    - `mode == 1`: Stratified Dark Basalt/Granite Cliff Rock with Sharp Crevices.
    - `mode == 2`: Sparkling Wind-Swept Glacier Snowpack.
    - `mode == 3`: Carved Flagstone Bridge Masonry & Mortar.
    """
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
                # Rich alpine meadow: emerald-olive grass blades + warm sunlit dry grass + limestone pebbles
                var lush = max(Float32(0.0), min(Float32(1.0), (fbm - Float32(0.32)) * Float32(2.4)))
                var dry = max(Float32(0.0), min(Float32(1.0), (f2 - Float32(0.62)) * Float32(3.0)))
                var r = Int(Float32(54.0) * (Float32(1.0) - lush) + Float32(68.0) * lush + Float32(48.0) * dry)
                var g = Int(Float32(88.0) * (Float32(1.0) - lush) + Float32(118.0) * lush + Float32(28.0) * dry)
                var b = Int(Float32(42.0) * (Float32(1.0) - lush) + Float32(48.0) * lush + Float32(12.0) * dry)
                albedo[y * DIM + x] = UInt32(0xFF000000) | (UInt32(min(255, r)) << 16) | (UInt32(min(255, g)) << 8) | UInt32(min(255, b))
                var ao = max(145, min(255, Int(Float32(175.0) + fbm * Float32(80.0))))
                var rough = max(170, min(250, Int(Float32(205.0) + f4 * Float32(40.0))))
                orm[y * DIM + x] = UInt32(0xFF000000) | (UInt32(ao) << 16) | (UInt32(rough) << 8) | UInt32(0)
                h_map[y * DIM + x] = fbm * Float32(0.65) + f4 * Float32(0.35)
                bump_scale = Float32(1.8)
            elif mode == 1:
                # Stratified dark slate/basalt cliff rock with horizontal bedding planes & vertical fractures
                var strata = _value_noise_tileable(v * Float32(6.0), u * Float32(24.0), 24)
                var crack = abs(_value_noise_tileable(v * Float32(16.0), u * Float32(8.0), 16) * Float32(2.0) - Float32(1.0))
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
                # Crisp wind-rippled alpine glacier snow
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
                # Carved ashlar stone pavers with recessed mortar joints
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
    """Generates a seamless `256x256` isotropic 5-octave FBM wind-chop `BC5` water normal map (zero periodic stripes!)."""
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


def bake_cloud_and_star_sky_ibl(mut ren: FlintAaaVulkanRenderer, preset: Int):
    """Combines physical `BrunetonSkyLuts` Rayleigh/Mie multiple scattering (`V18`) with a `640 x 320`
    `HdrImage` equirectangular sky synthesis (`V17`) featuring sunlit 3D cumulus/cirrus clouds,
    crisp solar corona, and 1,500+ pinpoint twilight stars, baked to `256 x 256` `BC6H_UFLOAT`!"""
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

            # Base Bruneton multiple-scattering radiance, boosted in deep cobalt Rayleigh richness
            var base_sky = luts.sample_sky(d, include_sun_disk=False)
            if preset == 0:
                base_sky = Vec3(base_sky.x * Float32(0.72), base_sky.y * Float32(0.88), base_sky.z * Float32(1.32))
            elif preset == 1:
                var horizon_glow = max(Float32(0.0), Float32(1.0) - abs(d.y) * Float32(3.2))
                var sun_az = max(Float32(0.0), d.dot(sun_d))
                var warm_add = Vec3(2.4, 0.95, 0.22) * (horizon_glow * horizon_glow * sun_az * sun_az)
                base_sky = Vec3(base_sky.x * Float32(1.25), base_sky.y * Float32(0.95), base_sky.z * Float32(0.88)) + warm_add
            else:
                # Deep midnight indigo twilight gradient + aurora horizon glow!
                var zen = max(Float32(0.0), d.y)
                var aur = exp(-abs(d.y - Float32(0.16)) * Float32(14.0)) * _vnoise2(phi * Float32(3.5), d.y * Float32(8.0))
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

            # Ray-plane projected volumetric cumulus & high-altitude cirrus clouds above horizon!
            if d.y > Float32(0.025) and preset < 2:
                var cx_p = d.x / (d.y + Float32(0.14)) * Float32(3.2)
                var cz_p = d.z / (d.y + Float32(0.14)) * Float32(3.2)
                var c_fbm = (
                    _vnoise2(cx_p + Float32(4.2), cz_p - Float32(1.7)) * Float32(0.52)
                    + _vnoise2(cx_p * Float32(2.15) - Float32(3.1), cz_p * Float32(2.15)) * Float32(0.28)
                    + _vnoise2(cx_p * Float32(4.5), cz_p * Float32(4.5) + Float32(5.4)) * Float32(0.13)
                    + _vnoise2(cx_p * Float32(9.2), cz_p * Float32(9.2)) * Float32(0.07)
                )
                # Offset sample toward sun for self-shadowing / golden silver-lining!
                var c_sun_s = _vnoise2(cx_p + sun_d.x * Float32(0.35) + Float32(4.2), cz_p + sun_d.z * Float32(0.35) - Float32(1.7))
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

            # Crisp solar disc & atmospheric aureole
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
        ren.sun_dir = Vec3(-0.28, 0.42, -0.86).normalized()  # Soft silver-blue moonlight!
        ren.sun_radiance = Vec3(0.08, 0.13, 0.26)
    _ = t_sun_obs

    var ap = luts.compute_aerial_perspective_constants()
    ren.aerial_sigma_ext = ap[0]
    ren.aerial_r_term = ap[1] * (Float32(0.55) if preset < 2 else Float32(0.15))
    ren.aerial_m_term = ap[2] * (Float32(0.45) if preset < 2 else Float32(0.10))
    ren.aerial_mie_g = ap[3]
    ren.enable_aerial_perspective = True


@always_inline
def pow_fast4(x: Float32) -> Float32:
    var x2 = x * x
    return x2 * x2


# ============================================================================
# 3. Sculpted Classical `.flpk` glTF Sanctum, Viaduct Bridge, Pine Forest & Knight
# ============================================================================

def build_and_pack_sunspire_sanctum_flpk(pack_path: String) -> NativeModel:
    """Builds a sculpted classical 12-column domed rotunda (`Sunspire Sanctum`) with embedded
    Baseline JPEG fluted travertine marble albedo + PNG ashlar normal + PNG ORM + gilded bronze dome
    + radiant celestial astrolabe core, packed into `.flpk` and streamed via `parse_native_glb`!"""
    var src = NativeModel()

    # Texture 0: Warm Travertine Marble & Gilded Frieze (encoded as Baseline JPEG in `.glb`!)
    var tw = 64
    var th = 64
    var t_alb = NativeTexture()
    t_alb.width = tw; t_alb.height = th
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

    # Texture 1: Fluted Column & Ashlar Normal Map (PNG)
    var t_nrm = NativeTexture()
    t_nrm.width = 32; t_nrm.height = 32
    t_nrm.pixels = List[UInt32](length=32 * 32, fill=0xFF8080FF)
    for y in range(32):
        for x in range(32):
            var flute = Int(Float32(128.0) + sin(Float32(x) / Float32(32.0) * Float32(6.2831853 * 8.0)) * Float32(28.0))
            var ny = 152 if (y % 8 == 0) else 128
            t_nrm.pixels[y * 32 + x] = UInt32(0xFF0000FF) | (UInt32(flute) << 16) | (UInt32(ny) << 8)
    src.textures.append(t_nrm^)

    # Texture 2: Polished Marble / Bronze ORM (PNG)
    var t_orm = NativeTexture()
    t_orm.width = 16; t_orm.height = 16
    t_orm.pixels = List[UInt32](length=16 * 16, fill=0xFFF65818)
    src.textures.append(t_orm^)

    # Mesh 0: Deep Foundation Drum + Stepped Circular Rotunda Stylobate + 12 Classical Entasis Columns
    var m0_b = MeshBuilder(base_color=0xFFF4EFE6, roughness=0.38, metallic=0.06, texture=0)
    m0_b.add_cylinder(Vec3(0.0, -5.5, 0.0), Vec3(0.0, 0.0, 0.0), 5.8, 5.6, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.55, 0.0), 5.6, 5.4, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 0.55, 0.0), Vec3(0.0, 1.10, 0.0), 5.0, 4.85, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 1.10, 0.0), Vec3(0.0, 1.45, 0.0), 4.45, 4.40, segments=28)

    for c_i in range(12):
        var ang = Float32(c_i) * Float32(6.2831853 / 12.0)
        var cx = cos(ang) * Float32(3.85)
        var cz = sin(ang) * Float32(3.85)
        # Plinth + tapered Doric/Corinthian shaft + echinus capital
        m0_b.add_cylinder(Vec3(cx, 1.45, cz), Vec3(cx, 1.80, cz), 0.42, 0.36, segments=14)
        m0_b.add_cylinder(Vec3(cx, 1.80, cz), Vec3(cx, 6.35, cz), 0.33, 0.27, segments=16)
        m0_b.add_cylinder(Vec3(cx, 6.35, cz), Vec3(cx, 6.72, cz), 0.29, 0.44, segments=14)

    # Circular architrave & cornice drum
    m0_b.add_cylinder(Vec3(0.0, 6.72, 0.0), Vec3(0.0, 7.35, 0.0), 4.35, 4.48, segments=28)
    m0_b.add_cylinder(Vec3(0.0, 7.35, 0.0), Vec3(0.0, 7.85, 0.0), 4.10, 3.95, segments=28)
    var m0 = m0_b^.finish()
    m0.normal_orm_texture = 1
    m0.orm_texture = 2
    src.meshes.append(m0^)

    # Mesh 1: Gilded Hammered-Bronze Hemispherical Cupola Dome, Finial Spire & Astrolabe Rings
    var m1_b = MeshBuilder(base_color=0xFFE5B869, roughness=0.22, metallic=0.88, texture=0)
    m1_b.add_ellipsoid(
        Vec3(0.0, 0.0, 0.0),
        Vec3(3.85, 2.55, 3.85),
        rings=12,
        segments=28,
        theta_start=0.0,
        theta_end=1.52,
    )
    # Lantern cupola & tapered golden finial spire atop dome
    m1_b.add_cylinder(Vec3(0.0, 2.45, 0.0), Vec3(0.0, 3.55, 0.0), 0.85, 0.68, segments=16)
    m1_b.add_cylinder(Vec3(0.0, 3.55, 0.0), Vec3(0.0, 6.20, 0.0), 0.42, 0.03, segments=14)
    # Inner sanctuary altar pedestal
    m1_b.add_cylinder(Vec3(0.0, -6.40, 0.0), Vec3(0.0, -5.25, 0.0), 1.15, 0.78, segments=18)
    var m1 = m1_b^.finish()
    m1.normal_orm_texture = 1
    src.meshes.append(m1^)

    # Mesh 2: Levitating Celestial Solar Crystal (`KHR_emissive_strength = 18.0`)
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

    var r_node = NativeNode(); r_node.name = "sanctum_root"
    r_node.children.append(1); r_node.children.append(2); r_node.children.append(3)
    var n_col = NativeNode(); n_col.name = "rotunda_colonnade"; n_col.mesh = 0; n_col.mesh_indices.append(0)
    var n_dome = NativeNode(); n_dome.name = "gilded_cupola"; n_dome.mesh = 1; n_dome.mesh_indices.append(1)
    n_dome.translation = Vec3(0.0, 7.80, 0.0)
    var n_core = NativeNode(); n_core.name = "celestial_crystal"; n_core.mesh = 2; n_core.mesh_indices.append(2)
    n_core.translation = Vec3(0.0, 2.85, 0.0)
    src.nodes.append(r_node^); src.nodes.append(n_col^); src.nodes.append(n_dome^); src.nodes.append(n_core^)
    src.roots.append(0)

    var glb_bytes = encode_native_glb(src, jpeg_texture_mask=1)
    var names = List[String]()
    names.append("models/sunspire_sanctum.glb")
    var files = List[List[UInt8]]()
    files.append(glb_bytes^)
    var flpk_bytes = encode_pack(names, files)
    _ = write_bytes(pack_path, flpk_bytes)
    return parse_native_glb(pack_read(pack_path, "models/sunspire_sanctum.glb"))


def build_viaduct_bridge_meshes() -> Tuple[NativeMesh, NativeMesh, NativeMesh]:
    """Builds the multi-arch stone viaduct bridge (`z = -26 .. +15`) spanning the glacial lake:
    returns `(stone_bridge_mesh, bronze_braziers_mesh, lantern_crystals_mesh)`."""
    var mb_stone = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.65, metallic=0.04)
    var mb_bronze = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.26, metallic=0.86)
    var mb_glow = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.12, metallic=0.0)

    var deck_y = Float32(1.65)
    # Modular 11-bay stone causeway deck & carved parapet walls for crisp proportional ashlar masonry UVs!
    for bay in range(11):
        var bz = Float32(-25.0) + Float32(bay) * Float32(3.9)
        mb_stone.add_beveled_box(Vec3(0.0, deck_y - Float32(0.20), bz), Vec3(2.85, 0.22, 1.96), bevel=0.03)
        mb_stone.add_beveled_box(Vec3(-2.62, deck_y + Float32(0.32), bz), Vec3(0.22, 0.32, 1.96), bevel=0.03)
        mb_stone.add_beveled_box(Vec3( 2.62, deck_y + Float32(0.32), bz), Vec3(0.22, 0.32, 1.96), bevel=0.03)
        mb_stone.add_beveled_box(Vec3(-2.62, deck_y + Float32(0.68), bz), Vec3(0.28, 0.06, 1.96), bevel=0.02)
        mb_stone.add_beveled_box(Vec3( 2.62, deck_y + Float32(0.68), bz), Vec3(0.28, 0.06, 1.96), bevel=0.02)

    # Stepped approach stairs on south shore (`z = -27 .. -31`) & north sanctum plaza (`z = +16 .. +18.5`)
    for st in range(5):
        var sy = deck_y - Float32(st) * Float32(0.28)
        mb_stone.add_beveled_box(Vec3(0.0, sy - Float32(0.2), Float32(-27.2) - Float32(st) * Float32(0.75)), Vec3(3.2, 0.22, 0.45), bevel=0.03)

    # 5 Classical Semicircular Viaduct Arches + Cutwater Stone Piers + 12 Bronze Lantern Pylons
    for a_i in range(5):
        var z_pier = Float32(-21.5) + Float32(a_i) * Float32(8.0)
        mb_stone.add_beveled_box(Vec3(0.0, -0.85, z_pier), Vec3(2.95, 2.35, 0.85), bevel=0.06)
        mb_stone.add_cylinder(Vec3(-2.95, -2.8, z_pier), Vec3(-2.95, 1.35, z_pier), 0.65, 0.55, segments=10)
        mb_stone.add_cylinder(Vec3( 2.95, -2.8, z_pier), Vec3( 2.95, 1.35, z_pier), 0.65, 0.55, segments=10)

        # Twin ornate bronze lantern standards atop each pier
        for side in range(2):
            var sx = Float32(-2.62) if side == 0 else Float32(2.62)
            mb_stone.add_beveled_box(Vec3(sx, deck_y + Float32(0.65), z_pier), Vec3(0.38, 0.42, 0.38), bevel=0.04)
            mb_bronze.add_cylinder(Vec3(sx, deck_y + Float32(1.05), z_pier), Vec3(sx, deck_y + Float32(2.45), z_pier), 0.12, 0.08, segments=10)
            mb_bronze.add_cylinder(Vec3(sx, deck_y + Float32(2.85), z_pier), Vec3(sx, deck_y + Float32(3.22), z_pier), 0.28, 0.03, segments=8)
            mb_glow.add_ellipsoid(Vec3(sx, deck_y + Float32(2.65), z_pier), Vec3(0.16, 0.22, 0.16), rings=8, segments=10)

        # Semicircular arch span to next pier
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
    """Builds a natural asymmetric multi-whorl Alpine Spruce/Fir (`(bark_trunk_mesh, needle_boughs_mesh)`)."""
    var mb_trunk = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.88, metallic=0.0)
    mb_trunk.add_cylinder(Vec3(0.0, -0.4, 0.0), Vec3(0.0, 6.8, 0.0), 0.30, 0.05, segments=9)

    var mb_foliage = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.78, metallic=0.0, double_sided=True)
    # 7 natural staggered branch whorls with slight organic offsets & slender spire crown
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
    """Builds a faceted weathered alpine boulder cluster."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.76, metallic=0.03)
    mb.add_ellipsoid(Vec3(0.0, 0.35, 0.0), Vec3(1.15, 0.72, 0.92), rings=7, segments=9)
    mb.add_ellipsoid(Vec3(0.55, 0.22, 0.35), Vec3(0.68, 0.48, 0.62), rings=6, segments=8)
    return mb^.finish()


def build_swept_glider_mesh() -> NativeMesh:
    """Builds a sleek NACA-airfoil swept-wing reconnaissance glider / alpine swift (`MeshBuilder.add_airfoil`)!"""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.78, double_sided=True)
    mb.add_cylinder(Vec3(0.0, 0.0, -0.42), Vec3(0.0, 0.0, 0.48), 0.03, 0.07, segments=8)
    mb.add_cylinder(Vec3(0.0, 0.0, 0.48), Vec3(0.0, 0.0, 0.72), 0.07, 0.01, segments=8)
    mb.add_airfoil(Vec3(0.0, 0.0, 0.25), 0.28, 0.09, Vec3(-0.95, 0.12, -0.22), 0.10, 0.06)
    mb.add_airfoil(Vec3(0.0, 0.0, 0.25), 0.28, 0.09, Vec3( 0.95, 0.12, -0.22), 0.10, 0.06)
    return mb^.finish()


def upload_aegis_sentinel_knight(mut ren: FlintAaaVulkanRenderer, mat_id: Int) -> AaaSkinnedMeshEntry:
    """Builds a sculpted 4-bone articulated `Aegis Sentinel` armored knight (`Bone 0`=greaves/pelvis,
    `Bone 1`=cuirass & crested visor helm, `Bone 2`=left shield arm, `Bone 3`=right halberd arm)
    for live GPU Dual-Quaternion compute skinning + zero-allocation `BLAS` refit!"""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.24, metallic=0.86, double_sided=True)

    # Bone 0: Sculpted Sabatons, Tapered Greaves, Knee Poleyn Rondels, Cuisses & Tasset Belt
    for side in range(2):
        var sx = Float32(-0.46) if side == 0 else Float32(0.46)
        mb.add_beveled_box(Vec3(sx, 0.14, 0.12), Vec3(0.20, 0.14, 0.34), bevel=0.05)
        mb.add_cylinder(Vec3(sx, 0.24, 0.0), Vec3(sx, 1.32, 0.0), 0.17, 0.23, segments=12)
        mb.add_ellipsoid(Vec3(sx, 1.36, 0.14), Vec3(0.24, 0.22, 0.18), rings=8, segments=10)
        mb.add_cylinder(Vec3(sx, 1.40, 0.0), Vec3(sx, 2.42, 0.0), 0.23, 0.28, segments=12)
    mb.add_cylinder(Vec3(0.0, 2.32, 0.0), Vec3(0.0, 2.82, 0.0), 0.68, 0.56, segments=14)

    # Bone 1: Fluted Cuirass, Gorget, Crested Corinthian/Knight Helm & Plume Crest
    mb.add_cylinder(Vec3(0.0, 2.82, 0.0), Vec3(0.0, 4.18, 0.0), 0.56, 0.82, segments=16)
    mb.add_ellipsoid(Vec3(0.0, 3.65, 0.14), Vec3(0.72, 0.62, 0.48), rings=10, segments=14)
    mb.add_cylinder(Vec3(0.0, 4.18, 0.0), Vec3(0.0, 4.46, 0.0), 0.52, 0.34, segments=14)
    mb.add_ellipsoid(Vec3(0.0, 4.82, 0.02), Vec3(0.36, 0.44, 0.40), rings=10, segments=14)
    mb.add_beveled_box(Vec3(0.0, 5.26, -0.06), Vec3(0.07, 0.26, 0.44), bevel=0.03)

    # Bone 2 (Left `x < -0.88`): Layered Pauldron, Vambrace & Curved Heater Shield
    mb.add_ellipsoid(Vec3(-1.02, 4.02, 0.0), Vec3(0.38, 0.32, 0.36), rings=8, segments=12)
    mb.add_cylinder(Vec3(-1.10, 4.00, 0.0), Vec3(-1.46, 2.88, 0.28), 0.18, 0.14, segments=10)
    mb.add_beveled_box(Vec3(-1.58, 2.95, 0.46), Vec3(0.08, 0.92, 0.56), bevel=0.04)

    # Bone 3 (Right `x > +0.88`): Layered Pauldron, Vambrace & Tall Ceremonial Polearm Halberd
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
    """Animates the 4-bone `Aegis Sentinel` with poised breathing yaw, shield guard & halberd salute."""
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
# 4. Parallel `SimdColumn` ECS + `JobPool` Soaring Glider Flock & Lantern Wisps
# ============================================================================

@fieldwise_init
struct FlockSimCtx(Copyable, ImplicitlyCopyable, Movable):
    var ang_addr: Int
    var rad_addr: Int
    var spd_addr: Int
    var px_addr: Int
    var py_addr: Int
    var pz_addr: Int
    var time_sec: Float32
    var dt: Float32


def step_flock_chunk(ctx_addr: Int, chunk_idx: Int):
    var ctx = Pointer[FlockSimCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var start = chunk_idx * ENTITY_CHUNK
    var end = min(ENTITY_COUNT, start + ENTITY_CHUNK)
    var ang_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.ang_addr)
    var rad_p = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.rad_addr)
    var spd_p = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.spd_addr)
    var px_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.px_addr)
    var py_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.py_addr)
    var pz_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.pz_addr)

    for i in range(start, end):
        var a = ang_p[unsafe_offset=i] + spd_p[unsafe_offset=i] * ctx.dt
        ang_p[unsafe_offset=i] = a
        if i < 192:
            # 192 high-altitude soaring alpine swifts/gliders organized into 4 tight natural V-flocks!
            var flock_id = i // 48
            var member = i % 48
            var wing_side = Float32(-1.0) if (member & 1) == 0 else Float32(1.0)
            var rank = Float32((member + 1) // 2)
            var lead_ang = a + Float32(flock_id) * Float32(1.5707963)
            var lead_r = Float32(58.0) + Float32(flock_id) * Float32(18.0)
            var lx = cos(lead_ang) * lead_r + wing_side * rank * Float32(1.35)
            var lz = sin(lead_ang) * lead_r + Float32(28.0) - rank * Float32(1.65)
            var ly = Float32(34.0) + Float32(flock_id) * Float32(7.5) + sin(ctx.time_sec * Float32(1.6) + Float32(member) * Float32(0.3)) * Float32(0.8)
            px_p[unsafe_offset=i] = lx
            py_p[unsafe_offset=i] = ly
            pz_p[unsafe_offset=i] = lz
        else:
            # 64 gentle golden sanctuary firefly/lantern wisps drifting around the rotunda & shoreline!
            var r = rad_p[unsafe_offset=i]
            var wx = cos(a) * r
            var wz = sin(a) * r + Float32(18.0)
            var gy = max(LAKE_LEVEL, sample_valley_height(wx, wz))
            px_p[unsafe_offset=i] = wx
            py_p[unsafe_offset=i] = gy + Float32(1.35) + Float32(0.55) * sin(ctx.time_sec * Float32(1.8) + Float32(i))
            pz_p[unsafe_offset=i] = wz


def apply_time_of_day_preset(mut ren: FlintAaaVulkanRenderer, preset: Int) -> Tuple[String, Float32]:
    """Bakes the `256 x 256` `BC6H` cloud/star sky IBL & configures camera exposure/bloom/bokeh."""
    bake_cloud_and_star_sky_ibl(ren, preset)
    if preset == 0:
        ren.ibl_intensity = Float32(0.95)
        ren.auto_exposure_key = Float32(0.22)
        ren.bloom_intensity = Float32(0.05)
        ren.enable_bokeh_dof = False
        ren.camera_lens.vignette_strength = Float32(0.16)
        return ("HIGH NOON ALPINE SANCTUARY", Float32(0.08))
    elif preset == 1:
        ren.ibl_intensity = Float32(1.10)
        ren.auto_exposure_key = Float32(0.24)
        ren.bloom_intensity = Float32(0.12)
        ren.enable_bokeh_dof = False
        ren.camera_lens.vignette_strength = Float32(0.22)
        return ("GOLDEN HOUR SUNSET", Float32(0.42))
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
        return ("STARRY TWILIGHT LTC + BOKEH", Float32(1.0))


@fieldwise_init
struct StaticPropInstance(Copyable, Movable):
    var mesh_id: Int
    var mat: Mat4
    var scale: Float32


def populate_open_world_frame(
    mut ren: FlintAaaVulkanRenderer,
    terrain_mesh_ids: List[Int],
    lake_mesh_id: Int,
    bridge_stone_id: Int,
    bridge_bronze_id: Int,
    bridge_glow_id: Int,
    glider_mesh_id: Int,
    props: List[StaticPropInstance],
    mut sanctum_model: NativeModel,
    sanctum_gpu: AaaUploadedModel,
    sentinel_skin: AaaSkinnedMeshEntry,
    col_ang: SimdColumn[DType.float32],
    col_px: SimdColumn[DType.float32],
    col_py: SimdColumn[DType.float32],
    col_pz: SimdColumn[DType.float32],
    light_scale: Float32,
):
    # 1. Multi-Biome Ridged Terrain (`Meadow`, `Cliff`, `Snow`) + Rippling Emerald Lake + Viaduct Bridge
    for m_id in terrain_mesh_ids:
        ren.draw_instance(m_id, Mat4())
    ren.draw_instance(lake_mesh_id, Mat4())
    ren.draw_instance(bridge_stone_id, Mat4())
    ren.draw_instance(bridge_bronze_id, Mat4())
    ren.draw_instance(bridge_glow_id, Mat4())

    # 2. Central Island `Sunspire Sanctum` Rotunda + 4 Distant Valley Watch-Shrines
    var isl_y = max(LAKE_LEVEL + Float32(0.35), sample_valley_height(0.0, 24.0) - Float32(0.15))
    ren.draw_native_model(sanctum_model, sanctum_gpu, trs(Vec3(0.0, isl_y, 24.0), Quat(), Vec3(1.12, 1.12, 1.12)))

    var shrine_coords = [
        Vec3(-36.0, 0.0, 42.0),
        Vec3( 38.0, 0.0, 38.0),
        Vec3(-44.0, 0.0, -2.0),
        Vec3( 42.0, 0.0,  6.0),
    ]
    for s_i in range(4):
        var sc = shrine_coords[s_i]
        var sy = max(LAKE_LEVEL + Float32(0.4), sample_valley_height(sc.x, sc.z) - Float32(0.2))
        ren.draw_native_model(sanctum_model, sanctum_gpu, trs(Vec3(sc.x, sy, sc.z), Quat(), Vec3(0.58, 0.58, 0.58)))

    # 3. `440+` GPU-Culled Instanced Alpine Pine Trees & Weathered Granite Shore Boulders
    for p_idx in range(len(props)):
        ref pr = props[p_idx]
        ren.draw_instance(pr.mesh_id, pr.mat, pr.mat, -1, pr.scale, True)

    # 4. Sculpted 4-Bone Dual-Quaternion Skinned `Aegis Sentinel` Knight Guarding the Bridge Approach!
    var yaw_s = Float32(3.14159265)  # Facing south toward the approaching bridge traveler
    var q_sent = Quat(0.0, sin(yaw_s * Float32(0.5)), 0.0, cos(yaw_s * Float32(0.5)))
    var sent_world = trs(Vec3(-1.15, 1.65, 10.5), q_sent, Vec3(0.62, 0.62, 0.62))
    ren.draw_instance(sentinel_skin.mesh_entry.mesh_id, sent_world, sent_world, -1, 0.62, True)

    # 5. Soaring NACA Swept-Wing Glider Flocks (`192` ECS aerial craft oriented along flight tangent)
    for g_i in range(192):
        var gp = Vec3(col_px.get(g_i), col_py.get(g_i), col_pz.get(g_i))
        var heading = -col_ang.get(g_i)
        var qg = Quat(0.0, sin(heading * Float32(0.5)), 0.0, cos(heading * Float32(0.5)))
        ren.draw_instance(glider_mesh_id, trs(gp, qg, Vec3(1.1, 1.1, 1.1)))

    # 6. `192` Clustered Punctual + Heitz `LTC` Rectangular & Tube Area Lights!
    # 10 Viaduct Bridge Lanterns (`LTC` warm amber area panels + vertical `LTC` runes)
    for a_i in range(5):
        var z_pier = Float32(-21.5) + Float32(a_i) * Float32(8.0)
        ren.add_cluster_light(
            AaaClusterLight.rect_ltc(
                center=Vec3(0.0, 4.30, z_pier),
                right_dir=Vec3(1.0, 0.0, 0.0),
                up_dir=Vec3(0.0, 1.0, 0.0),
                half_width=2.4,
                half_height=0.22,
                radiance=Vec3(12.0, 7.2, 2.4) * light_scale,
                range_radius=18.0,
                two_sided=True,
            )
        )
        ren.add_cluster_light(
            AaaClusterLight.tube_ltc(
                start_pos=Vec3(-2.62, 2.6, z_pier),
                end_pos=Vec3(-2.62, 4.5, z_pier),
                tube_radius=0.14,
                radiance=Vec3(15.0, 9.2, 3.2) * light_scale,
                range_radius=16.0,
            )
        )
        ren.add_cluster_light(
            AaaClusterLight.tube_ltc(
                start_pos=Vec3(2.62, 2.6, z_pier),
                end_pos=Vec3(2.62, 4.5, z_pier),
                tube_radius=0.14,
                radiance=Vec3(15.0, 9.2, 3.2) * light_scale,
                range_radius=16.0,
            )
        )

    # Central Sanctum Celestial Core & Cyan Runic Water-Edge Tubes (`3` lights -> `18` LTC total)
    ren.add_cluster_light(
        AaaClusterLight.rect_ltc(
            center=Vec3(0.0, isl_y + Float32(4.5), 24.0),
            right_dir=Vec3(1.0, 0.0, 0.0),
            up_dir=Vec3(0.0, 1.0, 0.0),
            half_width=1.8,
            half_height=1.8,
            radiance=Vec3(16.0, 10.5, 3.8) * max(Float32(0.35), light_scale),
            range_radius=34.0,
            two_sided=True,
        )
    )
    ren.add_cluster_light(
        AaaClusterLight.tube_ltc(
            start_pos=Vec3(-6.5, LAKE_LEVEL + Float32(0.8), 21.0),
            end_pos=Vec3(-6.5, LAKE_LEVEL + Float32(3.6), 21.0),
            tube_radius=0.16,
            radiance=Vec3(2.2, 10.5, 16.5) * light_scale,
            range_radius=20.0,
        )
    )
    ren.add_cluster_light(
        AaaClusterLight.tube_ltc(
            start_pos=Vec3(6.5, LAKE_LEVEL + Float32(0.8), 21.0),
            end_pos=Vec3(6.5, LAKE_LEVEL + Float32(3.6), 21.0),
            tube_radius=0.16,
            radiance=Vec3(2.2, 10.5, 16.5) * light_scale,
            range_radius=20.0,
        )
    )

    # 64 Drifting Sanctuary Firefly/Lantern Wisps + 110 Perimeter Torch Lights = `192` Active Clustered Lights!
    for w_i in range(64):
        var idx = 192 + w_i
        var wp = Vec3(col_px.get(idx), col_py.get(idx), col_pz.get(idx))
        var warm = (w_i % 3) != 0
        var lcol = (Vec3(4.5, 2.6, 0.75) if warm else Vec3(0.8, 3.4, 5.2)) * light_scale
        ren.add_cluster_light(AaaClusterLight.point(wp, lcol, radius=10.5))

    for p_i in range(110):
        var ang = Float32(p_i) * Float32(6.2831853 / 110.0)
        var rad = Float32(36.0) + Float32(p_i % 3) * Float32(14.0)
        var lx = cos(ang) * rad
        var lz = sin(ang) * rad + Float32(16.0)
        var ly = max(LAKE_LEVEL + Float32(1.2), sample_valley_height(lx, lz) + Float32(1.8))
        ren.add_cluster_light(AaaClusterLight.point(Vec3(lx, ly, lz), Vec3(3.2, 1.8, 0.6) * light_scale, radius=11.0))

    # Subtle low-lying sanctuary water mist wisps (`OIT` 6-way volumetric scattering at gentle realistic density!)
    if light_scale < Float32(0.6):
        ren.draw_volumetric_plume(
            AaaVolumetricPlume(
                center=Vec3(-14.0, LAKE_LEVEL + Float32(0.9), 28.0),
                radius=Float32(6.5),
                scatter_albedo=Vec3(0.92, 0.95, 1.0),
                density=Float32(0.012),
            )
        )
        ren.draw_volumetric_plume(
            AaaVolumetricPlume(
                center=Vec3(15.0, LAKE_LEVEL + Float32(0.9), 30.0),
                radius=Float32(6.5),
                scatter_albedo=Vec3(0.92, 0.95, 1.0),
                density=Float32(0.012),
            )
        )


def composite_open_world_hud(
    mut canvas: SimdCanvas,
    ren: FlintAaaVulkanRenderer,
    mut overlays: FlintDevOverlays,
    tod_label: String,
    broadphase_pairs: Int,
    blas_refit_us: Int,
    draw_debug_hud: Bool,
):
    var gpu_px = ren.readback_pixels()
    _ = external_call["memcpy", Int](Int(canvas.pixels.unsafe_ptr()), Int(gpu_px.unsafe_ptr()), canvas.width * canvas.height * 4)
    _ = len(gpu_px)

    if not draw_debug_hud:
        return

    var ae = ren.get_adapted_exposure()
    var exp_x100 = Int(ae[0] * Float32(100.0))
    var gpu_tot_us = max(1, Int(ren.render_graph.total_gpu_ms * 1000.0))
    var pass_cnt = len(ren.render_graph.pass_timings)
    canvas.triangles_drawn = ren.last_culled_meshlet_draws * 64
    canvas.sprites_drawn = ENTITY_COUNT
    overlays.render_us = gpu_tot_us
    overlays.fps = max(1, 1_000_000 // gpu_tot_us)

    canvas.draw_panel(14, 8, canvas.width - 28, 34, 0x0B131E, 0xF49E38)
    canvas.draw_text(
        24,
        16,
        "FLINT AAA OPEN WORLD  |  "
        + tod_label
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
        + "X  |  PAIRS: "
        + String(broadphase_pairs)
        + "  |  BLAS REFIT: "
        + String(blas_refit_us)
        + " US",
        0xFFE082,
        1,
    )
    overlays.render(canvas)


def main():
    var headless_shots = False
    for i in range(1, len(argv())):
        var arg = String(argv()[i])
        if arg == "--shots" or arg == "--headless":
            headless_shots = True

    var out_w = 1280
    var out_h = 720
    # Native 1280x720 VisBuffer + `256x256` 7-mip `BC6H` Sky/IBL Cubemap for razor-sharp AAA detail!
    var ren = FlintAaaVulkanRenderer(width=out_w, height=out_h, tsr_scale=1.0, env_cube_dim=256)
    if not ren.active:
        print("[Flint Open World] ERROR: Vulkan 1.3 hardware driver unavailable.")
        return

    print("[Flint Open World] Cooking 512m Sunspire Valley (3-biome ridged peaks, 300 pines, 140 boulders, viaduct bridge, .flpk glTF Sanctum & Aegis Sentinel)...")
    var t_cook0 = get_time()

    # 1. Upload `1024x1024` Continuous World-Biome Splat + `256x256` Micro-Detail & Architectural Materials!
    var splat_1k = make_world_biome_splat_1024()
    var splat_alb = ren.upload_bc7_texture(splat_1k[0], 1024, 1024, srgb=True, generate_mips=True)
    var splat_orm = ren.upload_bc7_texture(splat_1k[1], 1024, 1024, srgb=False, generate_mips=True)

    var tex_meadow = make_biome_textures_256(0)
    var meadow_alb_pine = ren.upload_bc7_texture(tex_meadow[0], 256, 256, srgb=True, generate_mips=True)
    var meadow_nrm = ren.upload_bc5_normal_texture(tex_meadow[1], 256, 256, generate_mips=True)

    var tex_cliff = make_biome_textures_256(1)
    var cliff_alb = ren.upload_bc7_texture(tex_cliff[0], 256, 256, srgb=True, generate_mips=True)
    var cliff_nrm = ren.upload_bc5_normal_texture(tex_cliff[1], 256, 256, generate_mips=True)
    var cliff_orm = ren.upload_bc7_texture(tex_cliff[2], 256, 256, srgb=False, generate_mips=True)

    var tex_paver = make_biome_textures_256(3)
    var paver_alb = ren.upload_bc7_texture(tex_paver[0], 256, 256, srgb=True, generate_mips=True)
    var paver_nrm = ren.upload_bc5_normal_texture(tex_paver[1], 256, 256, generate_mips=True)
    var paver_orm = ren.upload_bc7_texture(tex_paver[2], 256, 256, srgb=False, generate_mips=True)

    var water_nrm = ren.upload_bc5_normal_texture(make_water_ripple_normals_256(), 256, 256, generate_mips=True)

    var mat_terrain = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.84, metallic=0.0)
    mat_terrain.albedo_tex = splat_alb
    mat_terrain.orm_tex = splat_orm
    mat_terrain.detail_normal_tex = cliff_nrm
    mat_terrain.detail_uv_scale = Float32(56.0)
    mat_terrain.detail_normal_strength = Float32(0.52)
    var mat_terrain_id = ren.add_material(mat_terrain)

    var mat_cliff = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.76, metallic=0.04)
    mat_cliff.albedo_tex = cliff_alb
    mat_cliff.normal_tex = cliff_nrm
    mat_cliff.orm_tex = cliff_orm
    mat_cliff.detail_normal_tex = cliff_nrm
    mat_cliff.detail_uv_scale = Float32(4.0)
    mat_cliff.detail_normal_strength = Float32(0.45)
    var mat_cliff_id = ren.add_material(mat_cliff)

    var mat_bridge = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.62, metallic=0.04)
    mat_bridge.albedo_tex = paver_alb
    mat_bridge.normal_tex = paver_nrm
    mat_bridge.orm_tex = paver_orm
    var mat_bridge_id = ren.add_material(mat_bridge)

    var mat_bronze = AaaMaterial(base_color=Vec3(0.78, 0.56, 0.26), roughness=0.24, metallic=0.90)
    var mat_bronze_id = ren.add_material(mat_bronze)

    var mat_lantern = AaaMaterial(base_color=Vec3(1.0, 0.85, 0.55), roughness=0.15, metallic=0.0)
    mat_lantern.emissive = Vec3(18.0, 11.0, 3.8)
    var mat_lantern_id = ren.add_material(mat_lantern)

    # Rippling Emerald-Turquoise Glacial Mountain Lake
    var mat_lake = AaaMaterial(base_color=Vec3(0.012, 0.058, 0.072), roughness=0.045, metallic=0.0)
    mat_lake.normal_tex = water_nrm
    mat_lake.detail_normal_tex = water_nrm
    mat_lake.detail_uv_scale = Float32(5.5)
    mat_lake.detail_normal_strength = Float32(0.55)
    mat_lake.clearcoat = Float32(0.92)
    mat_lake.clearcoat_roughness = Float32(0.035)
    mat_lake.iridescence = Float32(0.08)
    mat_lake.iridescence_thickness_nm = Float32(390.0)
    var mat_lake_id = ren.add_material(mat_lake)

    # Alpine Pine Bark & Subsurface-Lit Evergreen Foliage
    var mat_bark = AaaMaterial(base_color=Vec3(0.16, 0.10, 0.06), roughness=0.88, metallic=0.0)
    mat_bark.normal_tex = cliff_nrm
    var mat_bark_id = ren.add_material(mat_bark)

    var mat_pine = AaaMaterial(base_color=Vec3(0.32, 0.42, 0.28), roughness=0.78, metallic=0.0)
    mat_pine.albedo_tex = meadow_alb_pine
    mat_pine.normal_tex = meadow_nrm
    mat_pine.detail_normal_tex = meadow_nrm
    mat_pine.detail_uv_scale = Float32(6.0)
    mat_pine.detail_normal_strength = Float32(0.55)
    mat_pine.subsurface_color = Vec3(0.14, 0.32, 0.07)
    mat_pine.subsurface_amount = Float32(0.28)
    var mat_pine_id = ren.add_material(mat_pine)

    var mat_sentinel = AaaMaterial(base_color=Vec3(0.76, 0.66, 0.46), roughness=0.22, metallic=0.88)
    mat_sentinel.clearcoat = Float32(0.45)
    mat_sentinel.clearcoat_roughness = Float32(0.08)
    var mat_sentinel_id = ren.add_material(mat_sentinel)

    var mat_glider = AaaMaterial(base_color=Vec3(0.18, 0.22, 0.28), roughness=0.28, metallic=0.78)
    var mat_glider_id = ren.add_material(mat_glider)

    # 2. Cook `16` Watertight Terrain Sectors (`1024x1024` Continuous Biome Splat + `56x` RNM Detail + `QEM` `LOD 0..3` + `BLAS`)!
    var terrain_mesh_ids = List[Int]()
    for cz in range(CHUNK_GRID):
        for cx in range(CHUNK_GRID):
            terrain_mesh_ids.append(ren.add_mesh(build_terrain_chunk_mesh(cx, cz), mat_terrain_id, build_lods=True).mesh_id)

    # Fine-UV Rippling Lake Surface
    var lake_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.05, metallic=0.0)
    var lv0 = lake_b.add_vertex(Vec3(-160.0, LAKE_LEVEL, -140.0), Vec3(0.0, 1.0, 0.0), 0.0, 0.0)
    var lv1 = lake_b.add_vertex(Vec3( 160.0, LAKE_LEVEL, -140.0), Vec3(0.0, 1.0, 0.0), 22.0, 0.0)
    var lv2 = lake_b.add_vertex(Vec3( 160.0, LAKE_LEVEL,  180.0), Vec3(0.0, 1.0, 0.0), 22.0, 22.0)
    var lv3 = lake_b.add_vertex(Vec3(-160.0, LAKE_LEVEL,  180.0), Vec3(0.0, 1.0, 0.0), 0.0, 22.0)
    lake_b.add_quad(lv0, lv3, lv2, lv1)
    var lake_entry = ren.add_mesh(lake_b^.finish(), mat_lake_id, build_lods=False)

    var bridge_meshes = build_viaduct_bridge_meshes()
    var bridge_stone_e = ren.add_mesh(bridge_meshes[0], mat_bridge_id, build_lods=True)
    var bridge_bronze_e = ren.add_mesh(bridge_meshes[1], mat_bronze_id, build_lods=True)
    var bridge_glow_e = ren.add_mesh(bridge_meshes[2], mat_lantern_id, build_lods=False)

    var pine_meshes = build_alpine_pine_tree_meshes()
    var pine_trunk_e = ren.add_mesh(pine_meshes[0], mat_bark_id, build_lods=True)
    var pine_bough_e = ren.add_mesh(pine_meshes[1], mat_pine_id, build_lods=True)
    var boulder_e = ren.add_mesh(build_granite_boulder_mesh(), mat_cliff_id, build_lods=True)
    var glider_e = ren.add_mesh(build_swept_glider_mesh(), mat_glider_id, build_lods=False)

    # Scatter `300` Alpine Pine Trees (`600` trunk+bough instances) & `140` Shore/Cliff Boulders deterministically!
    var props = List[StaticPropInstance]()
    var placed_trees = 0
    var seed_t = 0
    while placed_trees < 300 and seed_t < 4000:
        seed_t += 1
        var rx = (_hash2d_01(seed_t, 17) * Float32(2.0) - Float32(1.0)) * Float32(145.0)
        var rz = (_hash2d_01(seed_t, 53) * Float32(2.0) - Float32(1.0)) * Float32(145.0) + Float32(16.0)
        # Keep bridge corridor, central sanctum island & 4 lakeside shrines clear
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
            var tm = trs(Vec3(rx, wy - Float32(0.15), rz), qt, Vec3(sc, sc * Float32(1.08), sc))
            props.append(StaticPropInstance(pine_trunk_e.mesh_id, tm.copy(), sc * Float32(1.08)))
            props.append(StaticPropInstance(pine_bough_e.mesh_id, tm^, sc * Float32(1.08)))
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
            var rm = trs(Vec3(rx, wy - Float32(0.25), rz), qr, Vec3(sc, sc * Float32(0.82), sc))
            props.append(StaticPropInstance(boulder_e.mesh_id, rm^, sc))
            placed_rocks += 1

    # 3. Package & Stream Classical 12-Column `.glb` Sanctum from `.flpk` VFS + Upload Skinned `Aegis Sentinel`!
    var sanctum_model = build_and_pack_sunspire_sanctum_flpk("build/open_world_assets.flpk")
    var sanctum_gpu = ren.upload_native_model(sanctum_model, build_lods=True)
    var sentinel_skin = upload_aegis_sentinel_knight(ren, mat_sentinel_id)

    # 4. Initialize `SimdColumn` ECS + `JobPool` for `256` Autonomous Gliders & Sanctuary Wisps
    var reg = ArchetypeRegistry()
    var arch = reg.register("open_world_autonomous_entity")
    var slots = ArchetypeSlots(arch, ENTITY_COUNT)
    var col_ang = SimdColumn[DType.float32](ENTITY_COUNT)
    var col_rad = SimdColumn[DType.float32](ENTITY_COUNT)
    var col_spd = SimdColumn[DType.float32](ENTITY_COUNT)
    var col_px = SimdColumn[DType.float32](ENTITY_COUNT)
    var col_py = SimdColumn[DType.float32](ENTITY_COUNT)
    var col_pz = SimdColumn[DType.float32](ENTITY_COUNT)

    for i in range(ENTITY_COUNT):
        _ = slots.spawn()
        col_ang.set(i, Float32(i) * Float32(0.1963495))
        col_rad.set(i, Float32(7.5) + Float32(i % 32) * Float32(1.15))
        col_spd.set(i, Float32(0.18) if i < 192 else (Float32(0.25) + Float32(i % 11) * Float32(0.03)))

    var pool = JobPool()
    var workers = max(1, default_worker_count())
    _ = pool.start(workers)

    # Configure Crisp Native-Res AAA Pipeline (`svgf_passes = 0` + Temporal TSR GI Accumulation = Zero Texture Blur!)
    ren.display_mode = AAA_DISPLAY_AGX_REC709
    ren.enable_vsm = True
    ren.vsm_scene_center = Vec3(0.0, 2.0, 12.0)
    ren.vsm_ortho_half_extent = Float32(128.0)
    ren.enable_rt_shadows = ren.vk.has_ray_query
    ren.enable_rt_reflections = ren.vk.has_ray_query
    ren.enable_rt_gi = ren.vk.has_ray_query
    ren.gi_rays_per_pixel = 8
    ren.svgf_passes = 0
    ren.enable_auto_exposure = True
    ren.auto_exposure_speed = Float32(0.65)
    ren.max_lod_error_px = Float32(0.85)

    var cook_ms = Int((get_time() - t_cook0) * 1000.0)
    print(
        "[Flint Open World] World ready in "
        + String(cook_ms)
        + " ms (Meshes="
        + String(len(ren.geom.meshes))
        + ", Meshlets="
        + String(len(ren.geom.meshlets))
        + ", Props="
        + String(len(props))
        + ", BindlessTex="
        + String(len(ren.bindless_images))
        + ", Workers="
        + String(workers)
        + ")"
    )

    var canvas = SimdCanvas(out_w, out_h)
    var overlays = FlintDevOverlays()
    overlays.active_backend_name = "VULKAN-AAA-RT" if ren.vk.has_ray_query else "VULKAN-AAA"
    overlays.show_profiler = True

    if headless_shots:
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

        var prev_bones = evaluate_sentinel_bones(Float32(0.0))
        for s_idx in range(3):
            var tod_cfg = apply_time_of_day_preset(ren, s_idx)
            var tod_label = tod_cfg[0]
            var l_scale = tod_cfg[1]
            ren.auto_exposure_speed = Float32(1.0)
            var t_sim = Float32(1.8) + Float32(s_idx) * Float32(2.4)

            var sim_t0 = get_time()
            var f_ctx = FlockSimCtx(
                col_ang.base_addr(), col_rad.base_addr(), col_spd.base_addr(),
                col_px.base_addr(), col_py.base_addr(), col_pz.base_addr(), t_sim, Float32(0.5),
            )
            run_parallel_ctx(pool, ENTITY_COUNT // ENTITY_CHUNK, f_ctx, step_flock_chunk)
            col_ang.keep_alive(); col_rad.keep_alive(); col_spd.keep_alive()

            var bp_in = BroadphaseInput()
            var filt = CollisionFilter()
            for e_i in range(ENTITY_COUNT):
                var ep = Vec3(col_px.get(e_i), col_py.get(e_i), col_pz.get(e_i))
                bp_in.add(ep - Vec3(1.8, 1.8, 1.8), ep + Vec3(1.8, 1.8, 1.8), True, True, filt)
            var bp_res = find_pairs(bp_in, Int(Pointer(to=pool)))
            var sim_us = max(1, Int((get_time() - sim_t0) * 1_000_000.0))

            var cur_bones = evaluate_sentinel_bones(t_sim)
            var refit_ms = ren.update_skinned_mesh_bones(sentinel_skin, cur_bones, prev_bones)
            prev_bones = cur_bones^

            # Accumulate 5 temporal TSR frames (`5 * 8 = 40` ray-traced GI rays/pixel!) for clean, crisp cinema shots!
            for sub in range(5):
                ren.tsr_history_weight = Float32(0.0) if sub == 0 else Float32(0.80)
                ren.begin_frame(cam_eyes[s_idx], cam_targets[s_idx], fov_y_deg=54.0, near_z=0.20)
                populate_open_world_frame(
                    ren, terrain_mesh_ids, lake_entry.mesh_id,
                    bridge_stone_e.mesh_id, bridge_bronze_e.mesh_id, bridge_glow_e.mesh_id,
                    glider_e.mesh_id, props, sanctum_model, sanctum_gpu, sentinel_skin,
                    col_ang, col_px, col_py, col_pz, l_scale,
                )
                _ = ren.render_frame()

            overlays.tick_us = sim_us
            composite_open_world_hud(canvas, ren, overlays, tod_label, len(bp_res.pairs), Int(refit_ms * 1000.0), draw_debug_hud=False)
            _ = write_png(shot_paths[s_idx], canvas.pixels, out_w, out_h)
            print(
                "  [SHOT " + String(s_idx + 1) + "/3] " + shot_paths[s_idx]
                + " | " + tod_label
                + " | CulledDraws=" + String(ren.last_culled_meshlet_draws)
                + " | BroadphasePairs=" + String(len(bp_res.pairs))
                + " | " + ren.render_graph.format_gpu_profile_summary()
            )

        pool.stop()
        ren.shutdown()
        return

    # Interactive Window Mode (Wayland primary, direct X11 wire-protocol fallback!)
    var wl_win = WaylandWindow()
    var x11_win = X11Window()
    var use_wayland = wl_win.initialize(out_w, out_h)
    var use_x11 = False
    if not use_wayland:
        use_x11 = x11_win.initialize(out_w, out_h, "FLINT AAA OPEN WORLD - SUNSPIRE VALLEY")
    if not use_wayland and not use_x11:
        pool.stop()
        ren.shutdown()
        print("[Flint Open World] No Wayland/X11 display detected. Run with '--shots' to render open-world PNGs.")
        return

    var audio = NativeAudio()
    if audio.initialize():
        var wind = WavClip()
        wind.valid = True
        wind.channels = 1
        wind.sample_rate = 48000
        for s in range(48000):
            var ph = Float32(s) / Float32(48000.0) * Float32(6.2831853)
            var wv = sin(ph * Float32(82.5)) * Float32(0.35) + sin(ph * Float32(123.75)) * Float32(0.22) + sin(ph * Float32(247.5)) * Float32(0.12)
            wind.samples.append(Int16(Int(wv * Float32(10000.0))))
        audio.clips.append(wind^)
        audio.drone_clip = len(audio.clips) - 1
        audio.set_drone(0.45)

    var input = InputState()
    var tod_cfg0 = apply_time_of_day_preset(ren, 0)
    var tod_label = tod_cfg0[0]
    var cur_light_scale = tod_cfg0[1]
    var cam_pos = Vec3(-22.0, max(Float32(15.5), sample_valley_height(-22.0, -34.0) + Float32(8.5)), -34.0)
    var cam_yaw = Float32(0.38)
    var cam_pitch = Float32(-0.17)
    var prev_bones = evaluate_sentinel_bones(Float32(0.0))

    var t_start = get_time()
    var t_last = t_start

    while True:
        if use_wayland and wl_win.close_requested:
            break
        if use_x11 and not x11_win.active():
            break

        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - t_last)))
        t_last = now
        var elapsed = Float32(now - t_start)

        input.begin_frame()
        if use_wayland:
            wl_win.poll(input)
        else:
            x11_win.poll(input)

        if input.key_pressed(256):  # Esc
            break

        if not overlays.handle_input(input):
            if input.key_pressed(49):  # '1'
                var cfg = apply_time_of_day_preset(ren, 0); tod_label = cfg[0]; cur_light_scale = cfg[1]
            if input.key_pressed(50):  # '2'
                var cfg = apply_time_of_day_preset(ren, 1); tod_label = cfg[0]; cur_light_scale = cfg[1]
            if input.key_pressed(51):  # '3'
                var cfg = apply_time_of_day_preset(ren, 2); tod_label = cfg[0]; cur_light_scale = cfg[1]
            if input.key_pressed(66):  # 'B' Bloom toggle
                ren.bloom_intensity = Float32(0.0) if ren.bloom_intensity > Float32(0.01) else Float32(0.14)
            if input.key_pressed(70):  # 'F' Bokeh DoF toggle
                ren.enable_bokeh_dof = not ren.enable_bokeh_dof
            if input.key_pressed(71):  # 'G' RT GI toggle
                ren.enable_rt_gi = (not ren.enable_rt_gi) and ren.vk.has_ray_query

            if input.key_down(263): cam_yaw -= Float32(1.5) * dt
            if input.key_down(262): cam_yaw += Float32(1.5) * dt
            if input.key_down(265): cam_pitch = min(Float32(1.25), cam_pitch + Float32(1.1) * dt)
            if input.key_down(264): cam_pitch = max(Float32(-1.25), cam_pitch - Float32(1.1) * dt)

            var fwd = Vec3(sin(cam_yaw) * cos(cam_pitch), sin(cam_pitch), cos(cam_yaw) * cos(cam_pitch)).normalized()
            var rgt = Vec3(cos(cam_yaw), Float32(0.0), -sin(cam_yaw)).normalized()
            var move_spd = Float32(36.0) if input.key_down(340) else Float32(14.0)
            if input.key_down(87): cam_pos = cam_pos + fwd * (move_spd * dt)
            if input.key_down(83): cam_pos = cam_pos - fwd * (move_spd * dt)
            if input.key_down(68): cam_pos = cam_pos + rgt * (move_spd * dt)
            if input.key_down(65): cam_pos = cam_pos - rgt * (move_spd * dt)
            if input.key_down(69): cam_pos.y += move_spd * dt
            if input.key_down(81): cam_pos.y -= move_spd * dt
            var min_ground = max(LAKE_LEVEL + Float32(0.5), sample_valley_height(cam_pos.x, cam_pos.z)) + Float32(1.2)
            if cam_pos.y < min_ground:
                cam_pos.y = min_ground

        var sim_t0 = get_time()
        var f_ctx = FlockSimCtx(
            col_ang.base_addr(), col_rad.base_addr(), col_spd.base_addr(),
            col_px.base_addr(), col_py.base_addr(), col_pz.base_addr(), elapsed, dt,
        )
        run_parallel_ctx(pool, ENTITY_COUNT // ENTITY_CHUNK, f_ctx, step_flock_chunk)
        col_ang.keep_alive(); col_rad.keep_alive(); col_spd.keep_alive()

        var bp_in = BroadphaseInput()
        var filt = CollisionFilter()
        for e_i in range(ENTITY_COUNT):
            var ep = Vec3(col_px.get(e_i), col_py.get(e_i), col_pz.get(e_i))
            bp_in.add(ep - Vec3(1.8, 1.8, 1.8), ep + Vec3(1.8, 1.8, 1.8), True, True, filt)
        var bp_res = find_pairs(bp_in, Int(Pointer(to=pool)))
        var sim_us = Int((get_time() - sim_t0) * 1_000_000.0)

        var cur_bones = evaluate_sentinel_bones(elapsed)
        var refit_ms = ren.update_skinned_mesh_bones(sentinel_skin, cur_bones, prev_bones)
        prev_bones = cur_bones^

        var look_dir = Vec3(sin(cam_yaw) * cos(cam_pitch), sin(cam_pitch), cos(cam_yaw) * cos(cam_pitch)).normalized()
        ren.begin_frame(cam_pos, cam_pos + look_dir * Float32(10.0), fov_y_deg=54.0, near_z=0.20)
        populate_open_world_frame(
            ren, terrain_mesh_ids, lake_entry.mesh_id,
            bridge_stone_e.mesh_id, bridge_bronze_e.mesh_id, bridge_glow_e.mesh_id,
            glider_e.mesh_id, props, sanctum_model, sanctum_gpu, sentinel_skin,
            col_ang, col_px, col_py, col_pz, cur_light_scale,
        )
        _ = ren.render_frame()

        overlays.tick_us = sim_us
        composite_open_world_hud(canvas, ren, overlays, tod_label, len(bp_res.pairs), Int(refit_ms * 1000.0), draw_debug_hud=True)

        if use_wayland:
            wl_win.present(canvas.pixels, out_w, out_h)
        else:
            x11_win.present(canvas.pixels, out_w, out_h)
        audio.update(0.40)

    audio.close()
    if use_wayland:
        wl_win.close()
    if use_x11:
        x11_win.close()
    pool.stop()
    ren.shutdown()
