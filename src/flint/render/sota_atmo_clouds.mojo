"""Area H (sota:atmo, round 3): raymarched volumetric clouds over the physical sky.

Design (Schneider & Vos, "The Real-time Volumetric Cloudscapes of Horizon Zero Dawn", SIGGRAPH 2015 / "Nubis",
2017; Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016; Wrenninge et
al., "Oz: The Great and Volumetric", SIGGRAPH 2013 talk, for the multiple-scattering octaves):

* **Layer.** A spherical shell `bottom_km .. top_km` above sea level around the same planet as the sky LUTs (the
  planet centre is under the camera, world xz in km anchors the noise, so clouds stay put while the camera moves).
* **Noise (GPU-generated once, tileable).** 64^3 base volume: Perlin fBm remapped by Worley fBm (the
  "Perlin-Worley" low-frequency shape) blended with a Worley fBm; 32^3 detail volume: Worley fBm (erosion).
* **Weather map.** 256^2 texels over `weather_tile_km` (default 40 km): coverage (Perlin fBm + Worley cells) and
  cloud type (0 = stratus .. 1 = cumulus, which sets the height profile). Regenerated when its seed or tile changes;
  `coverage` and `type_bias` reshape it per frame without a rebuild.
* **Density.** `profile(h, type)`, then `remap(base * profile, 1 - coverage)` and a detail erosion that is billowy at
  the top and wispy at the base. Extinction `density_per_km` (no absorption: albedo 1).
* **March.** Shell entry to exit (capped at `max_dist_km`), `steps` samples with a per-pixel, per-frame jitter
  (interleaved gradient noise + golden-ratio frame offset), early out below 1 % transmittance. Energy-conserving
  integration (Hillaire 2015).
* **Lighting.** Sun and moon: TOA irradiance x the transmittance LUT at the sample (so sunset clouds glow orange and
  the earth shadow climbs them) x the light-march optical depth (`light_steps` growing steps); dual-lobe HG phase
  and 3 multiple-scattering octaves (a = b = c = 0.5). Ambient: the SH9 of the physical sky (written by the GPU IBL),
  up / down by height in the cloud. Aerial perspective at the transmittance-weighted cloud depth from the AP volume:
  the stored result is `S' = T_ap * S_cloud + (1 - T_cloud) * S_ap`, composited over sky pixels as
  `L = L_sky * T_cloud + S'` (exact split of the sky ray at the cloud).
* **Resolution and history.** The march runs at render resolution / `res_div` (default 2) and only where the
  footprint (+1 px ring) holds a sky pixel; others write a sentinel. Each pixel reprojects its cloud point (moved by
  the wind since the last frame) into the previous camera and blends `alpha` (default 0.15) of the new sample into
  the bilinear history. Off-screen, sentinel or invalid history restarts the pixel. The apply pass upsamples
  bilinearly (sentinel taps dropped).

CPU twins (test reference): `perlin3p_cpu`, `worley3p_cpu`, `cloud_base_texel_cpu`, `cloud_detail_texel_cpu`,
`cloud_weather_texel_cpu`, `cloud_march_cpu`.
"""
from std.collections import List
from std.math import abs, cos, exp, floor, max, min, sqrt
from flint.core.math import Vec3
from flint.platform.vulkan import VulkanGpu
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_GLOBALS, BIND_AAA_HDR_OUT, BIND_AAA_HISTORY
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BufferRef,
    GpuBool,
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    exp2_f,
    floor_f,
    fract_f,
    max_f,
    min_f,
    min_u,
    saturate,
    select_f,
    select_u,
    select_vec3,
    sqrt_f,
)
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.sota_atmo_core import RGB, SOTA_ATMO_AP_DIM, SotaAtmoLuts, sample_transmittance, sun_visibility
from flint.render.sota_atmo_dsl import (
    SOTA_ATMO_G2,
    _Atmo,
    _ap_fetch,
    _gexp,
    _ground_dist,
    _sample_trans,
    _sun_vis,
    _top_dist,
)
from flint.render.sota_atmo_night import hash_u32, hash_unit

comptime SOTA_CLOUD_BASE_N: Int = 64
comptime SOTA_CLOUD_DETAIL_N: Int = 32
comptime SOTA_CLOUD_WEATHER_N: Int = 256
comptime SOTA_CLOUD_SEED_BASE: UInt32 = 11
comptime SOTA_CLOUD_SEED_DETAIL: UInt32 = 23

# b_globals[SOTA_ATMO_G2 + ...]
comptime CW_BOTTOM: Int = 20
comptime CW_TOP: Int = 21
comptime CW_COVERAGE: Int = 22
comptime CW_DENSITY: Int = 23
comptime CW_WIND_X: Int = 24
comptime CW_WIND_Z: Int = 25
comptime CW_TILE: Int = 26
comptime CW_BASE_SCALE: Int = 27
comptime CW_DETAIL_SCALE: Int = 28
comptime CW_DETAIL_STRENGTH: Int = 29
comptime CW_STEPS: Int = 30
comptime CW_LIGHT_STEPS: Int = 31
comptime CW_LIGHT_STEP: Int = 32
comptime CW_MAX_DIST: Int = 33
comptime CW_G_FWD: Int = 34
comptime CW_G_BACK: Int = 35
comptime CW_G_MIX: Int = 36
comptime CW_MS_A: Int = 37
comptime CW_MS_B: Int = 38
comptime CW_MS_C: Int = 39
comptime CW_AMBIENT: Int = 40
comptime CW_TYPE_BIAS: Int = 41
comptime CW_ALPHA: Int = 42
comptime CW_FRAME: Int = 43
comptime CW_DWIND_X: Int = 44      # wind displacement since the previous frame (world units)
comptime CW_DWIND_Z: Int = 45
comptime CW_JITTER: Int = 46       # 1 = per-frame jitter, 0 = fixed mid-step samples
comptime CW_NO_AP: Int = 47        # 1 = skip the aerial perspective in front of the clouds (tests)
comptime CW_PREV_CAM: Int = 48     # previous camera: pos 48, fwd 51, right 54, up 57, tan_x 60, tan_y 61
comptime CW_HIST_VALID: Int = 62
comptime CW_DIV: Int = 63

comptime _MS_OCTAVES: Int = 3


# ============================================================================
# Noise (CPU twins; the shaders below mirror them operation for operation)
# ============================================================================

@always_inline
def _hash3(ix: Int, iy: Int, iz: Int, seed: UInt32) -> UInt32:
    var k = UInt32((ix & 1023) | ((iy & 1023) << 10) | ((iz & 1023) << 20))
    return hash_u32(k ^ (seed * UInt32(0x9E3779B9)))


@always_inline
def _wrap(i: Int, n: Int) -> Int:
    return ((i % n) + n) % n


def perlin3p_cpu(p: Vec3, period: Int, seed: UInt32) -> Float32:
    """Periodic gradient noise (period `period` lattice cells on every axis), roughly in [-1, 1]."""
    var fx0 = floor(p.x)
    var fy0 = floor(p.y)
    var fz0 = floor(p.z)
    var ix = Int(fx0)
    var iy = Int(fy0)
    var iz = Int(fz0)
    var fx = p.x - fx0
    var fy = p.y - fy0
    var fz = p.z - fz0
    var ux = fx * fx * fx * (fx * (fx * Float32(6.0) - Float32(15.0)) + Float32(10.0))
    var uy = fy * fy * fy * (fy * (fy * Float32(6.0) - Float32(15.0)) + Float32(10.0))
    var uz = fz * fz * fz * (fz * (fz * Float32(6.0) - Float32(15.0)) + Float32(10.0))
    var acc = Float32(0.0)
    for k in range(8):
        var dx = k & 1
        var dy = (k >> 1) & 1
        var dz = (k >> 2) & 1
        var h = _hash3(_wrap(ix + dx, period), _wrap(iy + dy, period), _wrap(iz + dz, period), seed)
        var gx = Float32(h & UInt32(255)) * Float32(1.0 / 127.5) - Float32(1.0)
        var gy = Float32((h >> UInt32(8)) & UInt32(255)) * Float32(1.0 / 127.5) - Float32(1.0)
        var gz = Float32((h >> UInt32(16)) & UInt32(255)) * Float32(1.0 / 127.5) - Float32(1.0)
        var n = gx * (fx - Float32(dx)) + gy * (fy - Float32(dy)) + gz * (fz - Float32(dz))
        var w = (ux if dx == 1 else Float32(1.0) - ux) * (uy if dy == 1 else Float32(1.0) - uy) * (uz if dz == 1 else Float32(1.0) - uz)
        acc += w * n
    return acc


def worley3p_cpu(p: Vec3, cells: Int, seed: UInt32) -> Float32:
    """Distance to the nearest feature point (one per cell, periodic), clamped to 1."""
    var fx0 = floor(p.x)
    var fy0 = floor(p.y)
    var fz0 = floor(p.z)
    var ix = Int(fx0)
    var iy = Int(fy0)
    var iz = Int(fz0)
    var fx = p.x - fx0
    var fy = p.y - fy0
    var fz = p.z - fz0
    var best = Float32(1.0)
    for k in range(27):
        var ox = k % 3
        var oy = (k // 3) % 3
        var oz = k // 9
        var h = _hash3(_wrap(ix + ox - 1, cells), _wrap(iy + oy - 1, cells), _wrap(iz + oz - 1, cells), seed + UInt32(77))
        var jx = Float32(h & UInt32(1023)) * Float32(1.0 / 1024.0)
        var jy = Float32((h >> UInt32(10)) & UInt32(1023)) * Float32(1.0 / 1024.0)
        var jz = Float32((h >> UInt32(20)) & UInt32(1023)) * Float32(1.0 / 1024.0)
        var dx = Float32(ox - 1) + jx - fx
        var dy = Float32(oy - 1) + jy - fy
        var dz = Float32(oz - 1) + jz - fz
        best = min(best, dx * dx + dy * dy + dz * dz)
    return min(Float32(1.0), sqrt(best))


def _wfbm_cpu(p: Vec3, f0: Int, seed: UInt32) -> Float32:
    return (Float32(1.0) - worley3p_cpu(p * Float32(f0), f0, seed)) * Float32(0.625) + (Float32(1.0) - worley3p_cpu(p * Float32(f0 * 2), f0 * 2, seed)) * Float32(0.25) + (Float32(1.0) - worley3p_cpu(p * Float32(f0 * 4), f0 * 4, seed)) * Float32(0.125)


def cloud_base_texel_cpu(x: Int, y: Int, z: Int) -> Float32:
    var n = Float32(SOTA_CLOUD_BASE_N)
    var p = Vec3((Float32(x) + Float32(0.5)) / n, (Float32(y) + Float32(0.5)) / n, (Float32(z) + Float32(0.5)) / n)
    var s = SOTA_CLOUD_SEED_BASE
    var pn = perlin3p_cpu(p * Float32(4.0), 4, s) + perlin3p_cpu(p * Float32(8.0), 8, s) * Float32(0.5) + perlin3p_cpu(p * Float32(16.0), 16, s) * Float32(0.25)
    var pf = min(Float32(1.0), max(Float32(0.0), Float32(0.5) + pn * Float32(0.4571428)))
    var wf = _wfbm_cpu(p, 4, s)
    var pw = min(Float32(1.0), max(Float32(0.0), (pf - (wf - Float32(1.0))) / (Float32(2.0) - wf)))
    return min(Float32(1.0), max(Float32(0.0), pw * Float32(0.6) + wf * Float32(0.4)))


def cloud_detail_texel_cpu(x: Int, y: Int, z: Int) -> Float32:
    var n = Float32(SOTA_CLOUD_DETAIL_N)
    var p = Vec3((Float32(x) + Float32(0.5)) / n, (Float32(y) + Float32(0.5)) / n, (Float32(z) + Float32(0.5)) / n)
    return _wfbm_cpu(p, 2, SOTA_CLOUD_SEED_DETAIL)


def cloud_weather_texel_cpu(x: Int, y: Int, seed: UInt32) -> Tuple[Float32, Float32]:
    var n = Float32(SOTA_CLOUD_WEATHER_N)
    var p = Vec3((Float32(x) + Float32(0.5)) / n, (Float32(y) + Float32(0.5)) / n, Float32(0.37))
    var c = perlin3p_cpu(p * Float32(3.0), 3, seed) + perlin3p_cpu(p * Float32(6.0), 6, seed) * Float32(0.5) + perlin3p_cpu(p * Float32(12.0), 12, seed) * Float32(0.25) + perlin3p_cpu(p * Float32(24.0), 24, seed) * Float32(0.125)
    var cv = Float32(0.5) + c * Float32(0.48)
    var wc = Float32(1.0) - worley3p_cpu(p * Float32(6.0), 6, seed)
    var cov = min(Float32(1.0), max(Float32(0.0), cv * Float32(0.75) + wc * Float32(0.35) - Float32(0.05)))
    var tn = perlin3p_cpu(Vec3(p.x * Float32(2.0), p.y * Float32(2.0), Float32(3.1)), 2, seed + UInt32(5))
    var typ = min(Float32(1.0), max(Float32(0.0), Float32(0.5) + tn * Float32(0.8)))
    return (cov, typ)


# ============================================================================
# DSL noise
# ============================================================================

def _gh(x: GpuU32) -> GpuU32:
    var s = x * UInt32(747796405) + UInt32(2891336453)
    var w = ((s >> ((s >> UInt32(28)) + UInt32(4))) ^ s) * UInt32(277803737)
    return (w >> UInt32(22)) ^ w


def _gh3(ix: GpuU32, iy: GpuU32, iz: GpuU32, seed: UInt32) -> GpuU32:
    var k = (ix & UInt32(1023)) | ((iy & UInt32(1023)) << UInt32(10)) | ((iz & UInt32(1023)) << UInt32(20))
    return _gh(k ^ (seed * UInt32(0x9E3779B9)))


def _g_perlin3p(mut dsl: ShaderDsl, p: GpuVec3, period: Int, seed: UInt32) -> GpuF32:
    var fx0 = floor_f(p.x())
    var fy0 = floor_f(p.y())
    var fz0 = floor_f(p.z())
    var ix = fx0.saturate_u32()
    var iy = fy0.saturate_u32()
    var iz = fz0.saturate_u32()
    var fx = p.x() - fx0
    var fy = p.y() - fy0
    var fz = p.z() - fz0
    var ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    var uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    var uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0)
    var acc = fx * 0.0
    for k in range(8):
        var dx = k & 1
        var dy = (k >> 1) & 1
        var dz = (k >> 2) & 1
        var h = _gh3((ix + UInt32(dx)) % UInt32(period), (iy + UInt32(dy)) % UInt32(period), (iz + UInt32(dz)) % UInt32(period), seed)
        var gx = (h & UInt32(255)).to_f32() * Float32(1.0 / 127.5) - 1.0
        var gy = ((h >> UInt32(8)) & UInt32(255)).to_f32() * Float32(1.0 / 127.5) - 1.0
        var gz = ((h >> UInt32(16)) & UInt32(255)).to_f32() * Float32(1.0 / 127.5) - 1.0
        var n = gx * (fx - Float32(dx)) + gy * (fy - Float32(dy)) + gz * (fz - Float32(dz))
        var wx = ux if dx == 1 else 1.0 - ux
        var wy = uy if dy == 1 else 1.0 - uy
        var wz = uz if dz == 1 else 1.0 - uz
        acc = acc + wx * wy * wz * n
    return acc


def _g_worley3p(mut dsl: ShaderDsl, p: GpuVec3, cells: Int, seed: UInt32) -> GpuF32:
    var fx0 = floor_f(p.x())
    var fy0 = floor_f(p.y())
    var fz0 = floor_f(p.z())
    var ix = fx0.saturate_u32() + UInt32(cells - 1)
    var iy = fy0.saturate_u32() + UInt32(cells - 1)
    var iz = fz0.saturate_u32() + UInt32(cells - 1)
    var fx = p.x() - fx0
    var fy = p.y() - fy0
    var fz = p.z() - fz0
    var best = dsl.var_f32_const(1.0)
    var k_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var k = k_v.get()
    dsl.break_if(k.ge(UInt32(27)))
    var ox = k % UInt32(3)
    var oy = (k // UInt32(3)) % UInt32(3)
    var oz = k // UInt32(9)
    var h = _gh3((ix + ox) % UInt32(cells), (iy + oy) % UInt32(cells), (iz + oz) % UInt32(cells), seed + UInt32(77))
    var jx = (h & UInt32(1023)).to_f32() * Float32(1.0 / 1024.0)
    var jy = ((h >> UInt32(10)) & UInt32(1023)).to_f32() * Float32(1.0 / 1024.0)
    var jz = ((h >> UInt32(20)) & UInt32(1023)).to_f32() * Float32(1.0 / 1024.0)
    var dx = ox.to_f32() - 1.0 + jx - fx
    var dy = oy.to_f32() - 1.0 + jy - fy
    var dz = oz.to_f32() - 1.0 + jz - fz
    best.set(min_f(best.get(), dx * dx + dy * dy + dz * dz))
    k_v.set(k + UInt32(1))
    dsl.end_loop()
    return min_f(sqrt_f(best.get()), 1.0)


def _g_wfbm(mut dsl: ShaderDsl, p: GpuVec3, f0: Int, seed: UInt32) -> GpuF32:
    var a = 1.0 - _g_worley3p(dsl, p * Float32(f0), f0, seed)
    var b = 1.0 - _g_worley3p(dsl, p * Float32(f0 * 2), f0 * 2, seed)
    var c = 1.0 - _g_worley3p(dsl, p * Float32(f0 * 4), f0 * 4, seed)
    return a * 0.625 + b * 0.25 + c * 0.125


def sota_cloud_noise_shader() -> List[UInt32]:
    """Push: 0 kind (0 base 64^3, 1 detail 32^3, 2 weather 256^2), 1 out_off, 2 weather seed. Thread (x, y) loops z."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var x = dsl.global_id_x()
    var y = dsl.global_id_y()
    var kind = dsl.push_u32(0)
    var o_off = dsl.push_u32(1)
    var wseed = dsl.push_u32(2)
    _ = wseed
    dsl.if_then(kind.eq(UInt32(0)))
    comptime NB = SOTA_CLOUD_BASE_N
    dsl.if_then(x.lt(UInt32(NB)) & y.lt(UInt32(NB)))
    var z_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var z = z_v.get()
    dsl.break_if(z.ge(UInt32(NB)))
    var inv = Float32(1.0 / Float32(NB))
    var p = GpuVec3((x.to_f32() + 0.5) * inv, (y.to_f32() + 0.5) * inv, (z.to_f32() + 0.5) * inv)
    var s = SOTA_CLOUD_SEED_BASE
    var pn = _g_perlin3p(dsl, p * 4.0, 4, s) + _g_perlin3p(dsl, p * 8.0, 8, s) * 0.5 + _g_perlin3p(dsl, p * 16.0, 16, s) * 0.25
    var pf = saturate(pn * 0.4571428 + 0.5)
    var wf = _g_wfbm(dsl, p, 4, s)
    var pw = saturate((pf - (wf - 1.0)) / (2.0 - wf))
    b_hist.store_f32(o_off + (z * UInt32(NB) + y) * UInt32(NB) + x, saturate(pw * 0.6 + wf * 0.4))
    z_v.set(z + UInt32(1))
    dsl.end_loop()
    dsl.end_if()
    dsl.end_if()
    dsl.if_then(kind.eq(UInt32(1)))
    comptime ND = SOTA_CLOUD_DETAIL_N
    dsl.if_then(x.lt(UInt32(ND)) & y.lt(UInt32(ND)))
    var z2_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var z2 = z2_v.get()
    dsl.break_if(z2.ge(UInt32(ND)))
    var inv2 = Float32(1.0 / Float32(ND))
    var p2 = GpuVec3((x.to_f32() + 0.5) * inv2, (y.to_f32() + 0.5) * inv2, (z2.to_f32() + 0.5) * inv2)
    b_hist.store_f32(o_off + (z2 * UInt32(ND) + y) * UInt32(ND) + x, _g_wfbm(dsl, p2, 2, SOTA_CLOUD_SEED_DETAIL))
    z2_v.set(z2 + UInt32(1))
    dsl.end_loop()
    dsl.end_if()
    dsl.end_if()
    return dsl.finish()


def sota_cloud_weather_shader(seed: UInt32) -> List[UInt32]:
    """Weather map (coverage, type) for one seed (the seed is baked in: the shader is rebuilt when it changes).
    Push: 0 out_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var x = dsl.global_id_x()
    var y = dsl.global_id_y()
    var o_off = dsl.push_u32(0)
    comptime NW = SOTA_CLOUD_WEATHER_N
    dsl.return_if(x.ge(UInt32(NW)) | y.ge(UInt32(NW)))
    var inv = Float32(1.0 / Float32(NW))
    var p = GpuVec3((x.to_f32() + 0.5) * inv, (y.to_f32() + 0.5) * inv, dsl.f32(0.37))
    var c = _g_perlin3p(dsl, p * 3.0, 3, seed) + _g_perlin3p(dsl, p * 6.0, 6, seed) * 0.5 + _g_perlin3p(dsl, p * 12.0, 12, seed) * 0.25 + _g_perlin3p(dsl, p * 24.0, 24, seed) * 0.125
    var cv = c * 0.48 + 0.5
    var wc = 1.0 - _g_worley3p(dsl, p * 6.0, 6, seed)
    var cov = saturate(cv * 0.75 + wc * 0.35 - 0.05)
    var tn = _g_perlin3p(dsl, GpuVec3(p.x() * 2.0, p.y() * 2.0, dsl.f32(3.1)), 2, seed + UInt32(5))
    var typ = saturate(tn * 0.8 + 0.5)
    var o = o_off + (y * UInt32(NW) + x) * UInt32(2)
    b_hist.store_f32(o, cov)
    b_hist.store_f32(o + UInt32(1), typ)
    return dsl.finish()


# ============================================================================
# Density model (CPU + DSL)
# ============================================================================

@always_inline
def _sat(x: Float32) -> Float32:
    return min(Float32(1.0), max(Float32(0.0), x))


def _smooth_cpu(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = _sat((x - e0) / (e1 - e0))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


def cloud_profile_cpu(h: Float32, typ: Float32) -> Float32:
    """Height profile in the layer: rounded base, top at 0.3 (stratus) .. 1.0 (cumulus) of the layer."""
    var top = Float32(0.3) + Float32(0.7) * typ
    return _smooth_cpu(0.0, 0.08, h) * (Float32(1.0) - _smooth_cpu(top * Float32(0.55), top, h))


def cloud_coverage_cpu(cov_w: Float32, coverage: Float32) -> Float32:
    return _sat((cov_w - (Float32(1.0) - coverage)) * Float32(2.5) + Float32(0.5)) * _sat(coverage * Float32(4.0))


def _wrapf(x: Float32, n: Int) -> Float32:
    var fnn = Float32(n)
    return x - floor(x / fnn) * fnn


def tex3_cpu(data: List[Float32], n: Int, q: Vec3) -> Float32:
    """Trilinear periodic fetch, `q` in texel units (texel centres at integer + 0.5)."""
    var qx = _wrapf(q.x - Float32(0.5), n)
    var qy = _wrapf(q.y - Float32(0.5), n)
    var qz = _wrapf(q.z - Float32(0.5), n)
    var x0 = min(n - 1, Int(floor(qx)))
    var y0 = min(n - 1, Int(floor(qy)))
    var z0 = min(n - 1, Int(floor(qz)))
    var tx = qx - Float32(x0)
    var ty = qy - Float32(y0)
    var tz = qz - Float32(z0)
    var acc = Float32(0.0)
    for k in range(8):
        var dx = k & 1
        var dy = (k >> 1) & 1
        var dz = (k >> 2) & 1
        var w = (tx if dx == 1 else Float32(1.0) - tx) * (ty if dy == 1 else Float32(1.0) - ty) * (tz if dz == 1 else Float32(1.0) - tz)
        acc += w * data[(((z0 + dz) % n) * n + (y0 + dy) % n) * n + (x0 + dx) % n]
    return acc


def tex2w_cpu(data: List[Float32], n: Int, qx_in: Float32, qy_in: Float32) -> Tuple[Float32, Float32]:
    var qx = _wrapf(qx_in - Float32(0.5), n)
    var qy = _wrapf(qy_in - Float32(0.5), n)
    var x0 = min(n - 1, Int(floor(qx)))
    var y0 = min(n - 1, Int(floor(qy)))
    var tx = qx - Float32(x0)
    var ty = qy - Float32(y0)
    var a = Float32(0.0)
    var b = Float32(0.0)
    for k in range(4):
        var dx = k & 1
        var dy = k >> 1
        var w = (tx if dx == 1 else Float32(1.0) - tx) * (ty if dy == 1 else Float32(1.0) - ty)
        var i = (((y0 + dy) % n) * n + (x0 + dx) % n) * 2
        a += w * data[i]
        b += w * data[i + 1]
    return (a, b)


def _gtex3(mut dsl: ShaderDsl, b: BufferRef[1], off: GpuU32, n: Int, q: GpuVec3) -> GpuF32:
    var fnn = Float32(n)
    var qx0 = q.x() - 0.5
    var qy0 = q.y() - 0.5
    var qz0 = q.z() - 0.5
    var qx = qx0 - floor_f(qx0 / fnn) * fnn
    var qy = qy0 - floor_f(qy0 / fnn) * fnn
    var qz = qz0 - floor_f(qz0 / fnn) * fnn
    var nm1 = dsl.u32(UInt32(n - 1))
    var x0 = min_u(floor_f(qx).saturate_u32(), nm1)
    var y0 = min_u(floor_f(qy).saturate_u32(), nm1)
    var z0 = min_u(floor_f(qz).saturate_u32(), nm1)
    var tx = qx - x0.to_f32()
    var ty = qy - y0.to_f32()
    var tz = qz - z0.to_f32()
    var x1 = (x0 + UInt32(1)) % UInt32(n)
    var y1 = (y0 + UInt32(1)) % UInt32(n)
    var z1 = (z0 + UInt32(1)) % UInt32(n)
    var un = UInt32(n)
    var c000 = b.load_f32(off + (z0 * un + y0) * un + x0)
    var c100 = b.load_f32(off + (z0 * un + y0) * un + x1)
    var c010 = b.load_f32(off + (z0 * un + y1) * un + x0)
    var c110 = b.load_f32(off + (z0 * un + y1) * un + x1)
    var c001 = b.load_f32(off + (z1 * un + y0) * un + x0)
    var c101 = b.load_f32(off + (z1 * un + y0) * un + x1)
    var c011 = b.load_f32(off + (z1 * un + y1) * un + x0)
    var c111 = b.load_f32(off + (z1 * un + y1) * un + x1)
    var a0 = c000 + (c100 - c000) * tx
    var a1 = c010 + (c110 - c010) * tx
    var a2 = c001 + (c101 - c001) * tx
    var a3 = c011 + (c111 - c011) * tx
    var b0 = a0 + (a1 - a0) * ty
    var b1 = a2 + (a3 - a2) * ty
    return b0 + (b1 - b0) * tz


def _gtex2w(mut dsl: ShaderDsl, b: BufferRef[1], off: GpuU32, n: Int, qx_in: GpuF32, qy_in: GpuF32) -> Tuple[GpuF32, GpuF32]:
    var fnn = Float32(n)
    var qx0 = qx_in - 0.5
    var qy0 = qy_in - 0.5
    var qx = qx0 - floor_f(qx0 / fnn) * fnn
    var qy = qy0 - floor_f(qy0 / fnn) * fnn
    var nm1 = dsl.u32(UInt32(n - 1))
    var x0 = min_u(floor_f(qx).saturate_u32(), nm1)
    var y0 = min_u(floor_f(qy).saturate_u32(), nm1)
    var tx = qx - x0.to_f32()
    var ty = qy - y0.to_f32()
    var x1 = (x0 + UInt32(1)) % UInt32(n)
    var y1 = (y0 + UInt32(1)) % UInt32(n)
    var un = UInt32(n)
    var i00 = off + (y0 * un + x0) * UInt32(2)
    var i10 = off + (y0 * un + x1) * UInt32(2)
    var i01 = off + (y1 * un + x0) * UInt32(2)
    var i11 = off + (y1 * un + x1) * UInt32(2)
    var w00 = (1.0 - tx) * (1.0 - ty)
    var w10 = tx * (1.0 - ty)
    var w01 = (1.0 - tx) * ty
    var w11 = tx * ty
    var a = b.load_f32(i00) * w00 + b.load_f32(i10) * w10 + b.load_f32(i01) * w01 + b.load_f32(i11) * w11
    var c = b.load_f32(i00 + UInt32(1)) * w00 + b.load_f32(i10 + UInt32(1)) * w10 + b.load_f32(i01 + UInt32(1)) * w01 + b.load_f32(i11 + UInt32(1)) * w11
    return (a, c)


def _gsmooth(e0: Float32, e1: Float32, x: GpuF32) -> GpuF32:
    var t = saturate((x - e0) * Float32(1.0 / (e1 - e0)))
    return t * t * (3.0 - t * 2.0)


def _gsmooth_dyn(e0: GpuF32, e1: GpuF32, x: GpuF32) -> GpuF32:
    var t = saturate((x - e0) / max_f(e1 - e0, 1e-4))
    return t * t * (3.0 - t * 2.0)


struct CloudParams(Copyable, ImplicitlyCopyable, Movable):
    """The per-frame cloud constants (`SotaClouds` resolved for this frame), shared by the CPU march and the
    globals writer."""
    var bottom_km: Float32
    var top_km: Float32
    var coverage: Float32
    var density_per_km: Float32
    var wind_x_km: Float32
    var wind_z_km: Float32
    var tile_km: Float32
    var base_scale_km: Float32
    var detail_scale_km: Float32
    var detail_strength: Float32
    var steps: Int
    var light_steps: Int
    var light_step_km: Float32
    var max_dist_km: Float32
    var g_fwd: Float32
    var g_back: Float32
    var g_mix: Float32
    var ms_a: Float32
    var ms_b: Float32
    var ms_c: Float32
    var ambient: Float32
    var type_bias: Float32

    def __init__(out self):
        self.bottom_km = 1.5
        self.top_km = 4.0
        self.coverage = 0.45
        self.density_per_km = 30.0
        self.wind_x_km = 0.0
        self.wind_z_km = 0.0
        self.tile_km = 40.0
        self.base_scale_km = 9.0
        self.detail_scale_km = 1.2
        self.detail_strength = 0.35
        self.steps = 64
        self.light_steps = 5
        self.light_step_km = 0.08
        self.max_dist_km = 60.0
        self.g_fwd = 0.8
        self.g_back = -0.3
        self.g_mix = 0.2
        self.ms_a = 0.5
        self.ms_b = 0.5
        self.ms_c = 0.5
        self.ambient = 1.0
        self.type_bias = 0.0


struct _CloudG(Copyable, Movable):
    var bottom: GpuF32
    var top: GpuF32
    var coverage: GpuF32
    var density: GpuF32
    var wind_x: GpuF32
    var wind_z: GpuF32
    var tile: GpuF32
    var base_scale: GpuF32
    var detail_scale: GpuF32
    var detail_strength: GpuF32
    var steps: GpuU32
    var light_steps: GpuU32
    var light_step: GpuF32
    var max_dist: GpuF32
    var g_fwd: GpuF32
    var g_back: GpuF32
    var g_mix: GpuF32
    var ms_a: GpuF32
    var ms_b: GpuF32
    var ms_c: GpuF32
    var ambient: GpuF32
    var type_bias: GpuF32

    def __init__(out self, mut dsl: ShaderDsl, b: BufferRef[1]):
        var g = SOTA_ATMO_G2
        self.bottom = b.load_f32(dsl.u32(UInt32(g + CW_BOTTOM)))
        self.top = b.load_f32(dsl.u32(UInt32(g + CW_TOP)))
        self.coverage = b.load_f32(dsl.u32(UInt32(g + CW_COVERAGE)))
        self.density = b.load_f32(dsl.u32(UInt32(g + CW_DENSITY)))
        self.wind_x = b.load_f32(dsl.u32(UInt32(g + CW_WIND_X)))
        self.wind_z = b.load_f32(dsl.u32(UInt32(g + CW_WIND_Z)))
        self.tile = b.load_f32(dsl.u32(UInt32(g + CW_TILE)))
        self.base_scale = b.load_f32(dsl.u32(UInt32(g + CW_BASE_SCALE)))
        self.detail_scale = b.load_f32(dsl.u32(UInt32(g + CW_DETAIL_SCALE)))
        self.detail_strength = b.load_f32(dsl.u32(UInt32(g + CW_DETAIL_STRENGTH)))
        self.steps = b.load_u32(dsl.u32(UInt32(g + CW_STEPS)))
        self.light_steps = b.load_u32(dsl.u32(UInt32(g + CW_LIGHT_STEPS)))
        self.light_step = b.load_f32(dsl.u32(UInt32(g + CW_LIGHT_STEP)))
        self.max_dist = b.load_f32(dsl.u32(UInt32(g + CW_MAX_DIST)))
        self.g_fwd = b.load_f32(dsl.u32(UInt32(g + CW_G_FWD)))
        self.g_back = b.load_f32(dsl.u32(UInt32(g + CW_G_BACK)))
        self.g_mix = b.load_f32(dsl.u32(UInt32(g + CW_G_MIX)))
        self.ms_a = b.load_f32(dsl.u32(UInt32(g + CW_MS_A)))
        self.ms_b = b.load_f32(dsl.u32(UInt32(g + CW_MS_B)))
        self.ms_c = b.load_f32(dsl.u32(UInt32(g + CW_MS_C)))
        self.ambient = b.load_f32(dsl.u32(UInt32(g + CW_AMBIENT)))
        self.type_bias = b.load_f32(dsl.u32(UInt32(g + CW_TYPE_BIAS)))


struct CloudNoiseCpu(Movable):
    """Read-back copies of the GPU noise volumes and weather map (for the CPU march reference)."""
    var base: List[Float32]
    var detail: List[Float32]
    var weather: List[Float32]

    def __init__(out self, var base: List[Float32], var detail: List[Float32], var weather: List[Float32]):
        self.base = base^
        self.detail = detail^
        self.weather = weather^


def cloud_density_cpu(nz: CloudNoiseCpu, cp: CloudParams, wx_km: Float32, wz_km: Float32, y_km: Float32, h: Float32, full: Bool) -> Float32:
    """Extinction (per km) at world (x, z) km, height fraction `h` in the layer. `full = False` skips the erosion."""
    if h <= Float32(0.0) or h >= Float32(1.0):
        return Float32(0.0)
    var px = wx_km + cp.wind_x_km
    var pz = wz_km + cp.wind_z_km
    var nw = Float32(SOTA_CLOUD_WEATHER_N)
    var wm = tex2w_cpu(nz.weather, SOTA_CLOUD_WEATHER_N, px / cp.tile_km * nw, pz / cp.tile_km * nw)
    var cov = cloud_coverage_cpu(wm[0], cp.coverage)
    var prof = cloud_profile_cpu(h, _sat(wm[1] + cp.type_bias))
    if cov * prof <= Float32(0.0):
        return Float32(0.0)
    var nb = Float32(SOTA_CLOUD_BASE_N)
    var base = tex3_cpu(nz.base, SOTA_CLOUD_BASE_N, Vec3(px, y_km, pz) * (nb / cp.base_scale_km))
    var d0 = _sat((base * prof - (Float32(1.0) - cov)) / max(cov, Float32(0.01)))
    if d0 <= Float32(0.0):
        return Float32(0.0)
    if full:
        var nd = Float32(SOTA_CLOUD_DETAIL_N)
        var det = tex3_cpu(nz.detail, SOTA_CLOUD_DETAIL_N, Vec3(px, y_km, pz) * (nd / cp.detail_scale_km))
        var hm = _sat(h * Float32(4.0))
        var dm = (det * (Float32(1.0) - hm) + (Float32(1.0) - det) * hm) * cp.detail_strength
        d0 = _sat((d0 - dm) / max(Float32(1.0) - dm, Float32(0.05)))
    return d0 * cp.density_per_km


def _gdensity(
    mut dsl: ShaderDsl, b: BufferRef[1], base_off: GpuU32, det_off: GpuU32, w_off: GpuU32, c: _CloudG,
    wx_km: GpuF32, wz_km: GpuF32, y_km: GpuF32, h: GpuF32, full: Bool,
) -> GpuF32:
    var res = dsl.var_f32_const(0.0)
    var px = wx_km + c.wind_x
    var pz = wz_km + c.wind_z
    var nw = Float32(SOTA_CLOUD_WEATHER_N)
    dsl.if_then(h.gt(0.0) & h.lt(1.0))
    var wm = _gtex2w(dsl, b, w_off, SOTA_CLOUD_WEATHER_N, px / c.tile * nw, pz / c.tile * nw)
    var cov = saturate((wm[0] - (1.0 - c.coverage)) * 2.5 + 0.5) * saturate(c.coverage * 4.0)
    var typ = saturate(wm[1] + c.type_bias)
    var top = typ * 0.7 + 0.3
    var prof = _gsmooth(0.0, 0.08, h) * (1.0 - _gsmooth_dyn(top * 0.55, top, h))
    dsl.if_then((cov * prof).gt(0.0))
    var nb = Float32(SOTA_CLOUD_BASE_N)
    var sb = nb / c.base_scale
    var base = _gtex3(dsl, b, base_off, SOTA_CLOUD_BASE_N, GpuVec3(px * sb, y_km * sb, pz * sb))
    var d0 = saturate((base * prof - (1.0 - cov)) / max_f(cov, 0.01))
    if full:
        dsl.if_then(d0.gt(0.0))
        var nd = Float32(SOTA_CLOUD_DETAIL_N)
        var sd = nd / c.detail_scale
        var det = _gtex3(dsl, b, det_off, SOTA_CLOUD_DETAIL_N, GpuVec3(px * sd, y_km * sd, pz * sd))
        var hm = saturate(h * 4.0)
        var dm = (det * (1.0 - hm) + (1.0 - det) * hm) * c.detail_strength
        res.set(saturate((d0 - dm) / max_f(1.0 - dm, 0.05)) * c.density)
        dsl.end_if()
    else:
        res.set(d0 * c.density)
    dsl.end_if()
    dsl.end_if()
    return res.get()


# ============================================================================
# Lighting helpers
# ============================================================================

def hg_phase_cpu(c: Float32, g: Float32) -> Float32:
    var g2 = g * g
    var den = max(Float32(1e-4), Float32(1.0) + g2 - Float32(2.0) * g * c)
    return (Float32(1.0) - g2) / (Float32(12.566370614) * den * sqrt(den))


def _ghg(c: GpuF32, g: GpuF32) -> GpuF32:
    var g2 = g * g
    var den = max_f(g2 + 1.0 - g * c * 2.0, 1e-4)
    return (1.0 - g2) / (den * sqrt_f(den) * Float32(12.566370614))


def cloud_phase_cpu(c: Float32, cp: CloudParams, gscale: Float32) -> Float32:
    return hg_phase_cpu(c, cp.g_fwd * gscale) * (Float32(1.0) - cp.g_mix) + hg_phase_cpu(c, cp.g_back * gscale) * cp.g_mix


@always_inline
def cloud_ign(x: Int, y: Int, frame: Int) -> Float32:
    """Interleaved gradient noise (Jimenez 2014) + golden-ratio frame offset."""
    var a = Float32(0.06711056) * Float32(x) + Float32(0.00583715) * Float32(y)
    var f = Float32(52.9829189) * (a - floor(a))
    var j = f - floor(f) + Float32(frame % 64) * Float32(0.61803398875)
    return j - floor(j)


struct CloudShell(Copyable, ImplicitlyCopyable, Movable):
    var t0: Float32
    var t1: Float32
    var hit: Bool

    def __init__(out self, t0: Float32, t1: Float32, hit: Bool):
        self.t0 = t0
        self.t1 = t1
        self.hit = hit


def _sph_exit(r: Float32, mu: Float32, rs: Float32) -> Float32:
    var disc = r * r * (mu * mu - Float32(1.0)) + rs * rs
    return max(Float32(0.0), sqrt(max(Float32(0.0), disc)) - r * mu)


def _sph_entry(r: Float32, mu: Float32, rs: Float32) -> Float32:
    """Entry distance into the sphere `rs` from outside (or -1)."""
    var disc = r * r * (mu * mu - Float32(1.0)) + rs * rs
    var d = -r * mu - sqrt(max(Float32(0.0), disc))
    if mu < Float32(0.0) and disc >= Float32(0.0) and d > Float32(0.0):
        return d
    return Float32(-1.0)


def cloud_shell_cpu(r0: Float32, mu: Float32, rb: Float32, rc0: Float32, rc1: Float32, max_dist: Float32) -> CloudShell:
    var t0 = Float32(0.0)
    var t1: Float32
    if r0 < rc0:
        if _sph_entry(r0, mu, rb) > Float32(0.0):
            return CloudShell(0.0, 0.0, False)
        t0 = _sph_exit(r0, mu, rc0)
        t1 = _sph_exit(r0, mu, rc1)
    elif r0 <= rc1:
        var inner = _sph_entry(r0, mu, rc0)
        t1 = inner if inner > Float32(0.0) else _sph_exit(r0, mu, rc1)
    else:
        var e = _sph_entry(r0, mu, rc1)
        if e < Float32(0.0):
            return CloudShell(0.0, 0.0, False)
        t0 = e
        var inner = _sph_entry(r0, mu, rc0)
        var disc = r0 * r0 * (mu * mu - Float32(1.0)) + rc1 * rc1
        t1 = inner if inner > Float32(0.0) else (-r0 * mu + sqrt(max(Float32(0.0), disc)))
    t1 = min(t1, max_dist)
    return CloudShell(t0, t1, t1 > t0)


def _gsph_exit(r: GpuF32, mu: GpuF32, rs: GpuF32) -> GpuF32:
    return _top_dist(r, mu, rs)


def _gsph_entry(r: GpuF32, mu: GpuF32, rs: GpuF32) -> GpuF32:
    return _ground_dist(r, mu, rs)


# ============================================================================
# March (CPU reference)
# ============================================================================

struct CloudSample(Copyable, ImplicitlyCopyable, Movable):
    var rgb: RGB
    var trans: Float32
    var t_mean: Float32

    def __init__(out self, rgb: RGB, trans: Float32, t_mean: Float32):
        self.rgb = rgb
        self.trans = trans
        self.t_mean = t_mean


def cloud_march_cpu(
    nz: CloudNoiseCpu,
    cp: CloudParams,
    luts: SotaAtmoLuts,
    cam_x_km: Float32,
    cam_z_km: Float32,
    r0: Float32,
    d: Vec3,
    jitter: Float32,
    sun_dir: Vec3,
    sun_e: RGB,
    moon_dir: Vec3,
    moon_e: RGB,
    amb_up: RGB,
    amb_dn: RGB,
) -> CloudSample:
    """CPU twin of the GPU march (no aerial perspective, no history): in-scattered radiance, transmittance and the
    transmittance-weighted depth. `amb_up` / `amb_dn`: the SH9 sky evaluated up / down (radiance)."""
    var p = luts.params
    var rb = p.r_bottom_km
    var rc0 = rb + cp.bottom_km
    var rc1 = rb + cp.top_km
    var sh = cloud_shell_cpu(r0, d.y, rb, rc0, rc1, cp.max_dist_km)
    if not sh.hit:
        return CloudSample(RGB(0.0), 1.0, 0.5 * (cp.bottom_km + cp.top_km))
    var dt = (sh.t1 - sh.t0) / Float32(cp.steps)
    var tr = Float32(1.0)
    var l = RGB(0.0)
    var tw = Float32(0.0)
    var ww = Float32(0.0)
    for i in range(cp.steps):
        if tr < Float32(0.01):
            break
        var t = sh.t0 + (Float32(i) + jitter) * dt
        var px = d.x * t
        var py = r0 + d.y * t
        var pz = d.z * t
        var r = sqrt(px * px + py * py + pz * pz)
        var h = (r - rc0) / (rc1 - rc0)
        var sig = cloud_density_cpu(nz, cp, cam_x_km + px, cam_z_km + pz, r - rb, h, True)
        if sig <= Float32(0.0):
            continue
        var pos = Vec3(px, py, pz)
        var src = RGB(0.0)
        for li in range(2):
            var ld = sun_dir if li == 0 else moon_dir
            var le = sun_e if li == 0 else moon_e
            if li == 1 and le[0] + le[1] + le[2] <= Float32(0.0):
                continue
            var mu_l = pos.dot(ld) / r
            var t_atm = sample_transmittance(luts.trans, r, mu_l, p) * sun_visibility(r, mu_l, p)
            var od = Float32(0.0)
            var s_acc = Float32(0.0)
            var ds = cp.light_step_km
            for k in range(cp.light_steps):
                var sm = s_acc + ds * Float32(0.5)
                var q = pos + ld * sm
                var rq = sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
                var hq = (rq - rc0) / (rc1 - rc0)
                od += cloud_density_cpu(nz, cp, cam_x_km + q.x, cam_z_km + q.z, rq - rb, hq, k < 2) * ds
                s_acc += ds
                ds *= Float32(2.0)
            var c = d.dot(ld)
            var scat = Float32(0.0)
            var ak = Float32(1.0)
            var bk = Float32(1.0)
            var ck = Float32(1.0)
            for _ in range(_MS_OCTAVES):
                scat += bk * cloud_phase_cpu(c, cp, ck) * Float32(exp(Float64(-od * ak)))
                ak *= cp.ms_a
                bk *= cp.ms_b
                ck *= cp.ms_c
            src = src + RGB(le[0] * t_atm[0] * scat, le[1] * t_atm[1] * scat, le[2] * t_atm[2] * scat, 0.0)
        var hf = Float32(0.35) + Float32(0.65) * _sat(h)
        for ch in range(3):
            src[ch] = src[ch] + (amb_dn[ch] + (amb_up[ch] - amb_dn[ch]) * hf) * cp.ambient
        var st = Float32(exp(Float64(-sig * dt)))
        var wgt = tr * (Float32(1.0) - st)
        # Energy-conserving step: S * sigma_s * (1 - e^{-sigma dt}) / sigma_t with sigma_s = sigma_t.
        for ch in range(3):
            l[ch] = l[ch] + src[ch] * wgt
        tw += t * wgt
        ww += wgt
        tr *= st
    var tm = tw / ww if ww > Float32(1e-6) else Float32(0.5) * (sh.t0 + sh.t1)
    return CloudSample(l, tr, tm)


# ============================================================================
# March + temporal reprojection (GPU)
# ============================================================================

def sota_cloud_march_shader() -> List[UInt32]:
    """Push: 0 cw, 1 ch, 2 div, 3 cur_off, 4 prev_off, 5 trans_off, 6 ap_off, 7 base_off, 8 detail_off,
    9 weather_off, 10 render_w, 11 render_h, 12 ap_max (f32 km). Output per cloud pixel: S' rgb, T (2 = no sky)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hdr = dsl.buffer[1](BIND_AAA_HDR_OUT)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var cx = dsl.global_id_x()
    var cy = dsl.global_id_y()
    var cw = dsl.push_u32(0)
    var chh = dsl.push_u32(1)
    var div = dsl.push_u32(2)
    var cur_off = dsl.push_u32(3)
    var prev_off = dsl.push_u32(4)
    var t_off = dsl.push_u32(5)
    var ap_off = dsl.push_u32(6)
    var base_off = dsl.push_u32(7)
    var det_off = dsl.push_u32(8)
    var w_off = dsl.push_u32(9)
    var rw = dsl.push_u32(10)
    var rh = dsl.push_u32(11)
    dsl.return_if(cx.ge(cw) | cy.ge(chh))
    var a = _Atmo(dsl, b_glob)
    var c = _CloudG(dsl, b_glob)
    var g2 = SOTA_ATMO_G2
    var out_w = cur_off + (cy * cw + cx) * UInt32(4)

    # --- any sky pixel in the footprint (+1 px ring)? ---
    var x0i = cx * div
    var y0i = cy * div
    var any_sky = dsl.var_u32(dsl.u32(0))
    var step = max_u_g(dsl, div >> UInt32(1), dsl.u32(1))
    var span = div + UInt32(2)
    var jy_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var jy = jy_v.get()
    dsl.break_if(jy.ge(span))
    var jx_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var jx = jx_v.get()
    dsl.break_if(jx.ge(span))
    var sx = min_u(select_u((x0i + jx).ge(UInt32(1)), x0i + jx - UInt32(1), dsl.u32(0)), rw - UInt32(1))
    var sy = min_u(select_u((y0i + jy).ge(UInt32(1)), y0i + jy - UInt32(1), dsl.u32(0)), rh - UInt32(1))
    var zz = b_hdr.load_f32((sy * rw + sx) * UInt32(8) + UInt32(3))
    any_sky.set(select_u(zz.ge(1.0e5), dsl.u32(1), any_sky.get()))
    jx_v.set(jx + step)
    dsl.end_loop()
    jy_v.set(jy + step)
    dsl.end_loop()
    dsl.if_then(any_sky.get().eq(UInt32(0)))
    b_hist.store_vec3(out_w, dsl.vec3_const(0.0, 0.0, 0.0))
    b_hist.store_f32(out_w + UInt32(3), dsl.f32(2.0))
    dsl.end_if()
    dsl.return_if(any_sky.get().eq(UInt32(0)))

    # --- ray (no TAA jitter: the cloud image is a stable field) ---
    var cam = b_glob.load_vec3(dsl.u32(32))
    var fwd = b_glob.load_vec3(dsl.u32(36))
    var tan_x = b_glob.load_f32(dsl.u32(39))
    var rgt = b_glob.load_vec3(dsl.u32(40))
    var tan_y = b_glob.load_f32(dsl.u32(43))
    var up = b_glob.load_vec3(dsl.u32(44))
    var fw = rw.to_f32()
    var fh = rh.to_f32()
    var fdiv = div.to_f32()
    var ndc_x = (cx.to_f32() + 0.5) * fdiv / fw * 2.0 - 1.0
    var ndc_y = 1.0 - (cy.to_f32() + 0.5) * fdiv / fh * 2.0
    var d = (fwd + rgt * (ndc_x * tan_x) + up * (ndc_y * tan_y)).normalized()
    var r0 = a.r_cam
    var rc0 = a.rb + c.bottom
    var rc1 = a.rb + c.top
    var mu = d.y()

    # --- shell interval ---
    var below = r0.lt(rc0)
    var above = r0.gt(rc1)
    var ground = _gsph_entry(r0, mu, a.rb)
    var e_top = _gsph_entry(r0, mu, rc1)
    var e_bot = _gsph_entry(r0, mu, rc0)
    var disc1 = r0 * r0 * (mu * mu - 1.0) + rc1 * rc1
    var exit_top_out = sqrt_f(max_f(disc1, 0.0)) - r0 * mu
    var t0 = select_f(below, _gsph_exit(r0, mu, rc0), select_f(above, max_f(e_top, 0.0), dsl.f32(0.0)))
    var t1_in = select_f(e_bot.gt(0.0), e_bot, _gsph_exit(r0, mu, rc1))
    var t1_out = select_f(e_bot.gt(0.0), e_bot, exit_top_out)
    var t1 = min_f(select_f(below, _gsph_exit(r0, mu, rc1), select_f(above, t1_out, t1_in)), c.max_dist)
    var hit = t1.gt(t0) & (~(below & ground.gt(0.0))) & (~(above & e_top.lt(0.0)))

    # --- march ---
    var frame = b_glob.load_u32(dsl.u32(UInt32(g2 + CW_FRAME)))
    var jit_on = b_glob.load_u32(dsl.u32(UInt32(g2 + CW_JITTER)))
    var ia = cx.to_f32() * 0.06711056 + cy.to_f32() * 0.00583715
    var ign = fract_f(fract_f(ia) * 52.9829189)
    var jit = select_f(jit_on.eq(UInt32(0)), dsl.f32(0.5), fract_f(ign + (frame % UInt32(64)).to_f32() * 0.61803398875))
    var l_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var tr_v = dsl.var_f32_const(1.0)
    var tw_v = dsl.var_f32_const(0.0)
    var ww_v = dsl.var_f32_const(0.0)
    var cam_xk = cam.x() * a.kpu
    var cam_zk = cam.z() * a.kpu
    var amb_up_sh = emit_eval_sh9_irradiance(dsl.raw_module()[], b_glob.buf_id, 72, dsl.f32(0.0).id, dsl.f32(1.0).id, dsl.f32(0.0).id)
    var amb_dn_sh = emit_eval_sh9_irradiance(dsl.raw_module()[], b_glob.buf_id, 72, dsl.f32(0.0).id, dsl.f32(-1.0).id, dsl.f32(0.0).id)
    var amb_up = GpuVec3(GpuF32(dsl.mod_addr, amb_up_sh[0]), GpuF32(dsl.mod_addr, amb_up_sh[1]), GpuF32(dsl.mod_addr, amb_up_sh[2]))
    var amb_dn = GpuVec3(GpuF32(dsl.mod_addr, amb_dn_sh[0]), GpuF32(dsl.mod_addr, amb_dn_sh[1]), GpuF32(dsl.mod_addr, amb_dn_sh[2]))
    var has_moon = (a.moon_e.x() + a.moon_e.y() + a.moon_e.z()).gt(0.0)
    var cs = d.dot(a.sun_d)
    var cm = d.dot(a.moon_d)
    var nsteps = max_u_g(dsl, c.steps, dsl.u32(1))
    var dt = (t1 - t0) / nsteps.to_f32()
    dsl.if_then(hit)
    var i_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var i = i_v.get()
    dsl.break_if(i.ge(nsteps) | tr_v.get().lt(0.01))
    i_v.set(i + UInt32(1))
    var t = t0 + (i.to_f32() + jit) * dt
    var px = d.x() * t
    var py = r0 + d.y() * t
    var pz = d.z() * t
    var r = sqrt_f(px * px + py * py + pz * pz)
    var h = (r - rc0) / (rc1 - rc0)
    var sig = _gdensity(dsl, b_hist, base_off, det_off, w_off, c, cam_xk + px, cam_zk + pz, r - a.rb, h, True)
    dsl.if_then(sig.gt(0.0))
    var pos = GpuVec3(px, py, pz)
    var src_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    for li in range(2):
        var ld = a.sun_d if li == 0 else a.moon_d
        var le = a.sun_e if li == 0 else a.moon_e
        var cl = cs if li == 0 else cm
        if li == 1:
            dsl.if_then(has_moon)
        var mu_l = pos.dot(ld) / r
        var t_atm = _sample_trans(dsl, b_hist, t_off, a, r, mu_l) * _sun_vis(r, mu_l, a)
        # A light below this point's horizon contributes nothing: skip its light march (the sun at night).
        dsl.if_then((t_atm.x() + t_atm.y() + t_atm.z()).gt(0.0))
        var od_v = dsl.var_f32_const(0.0)
        var sacc_v = dsl.var_f32_const(0.0)
        var ds_v = dsl.var_f32(c.light_step)
        var k_v = dsl.var_u32(dsl.u32(0))
        dsl.begin_loop()
        var k = k_v.get()
        dsl.break_if(k.ge(c.light_steps))
        var ds = ds_v.get()
        var sm = sacc_v.get() + ds * 0.5
        var q = pos + ld * sm
        var rq = sqrt_f(q.x() * q.x() + q.y() * q.y() + q.z() * q.z())
        var hq = (rq - rc0) / (rc1 - rc0)
        var dq = dsl.var_f32_const(0.0)
        dsl.if_then(k.lt(UInt32(2)))
        dq.set(_gdensity(dsl, b_hist, base_off, det_off, w_off, c, cam_xk + q.x(), cam_zk + q.z(), rq - a.rb, hq, True))
        dsl.else_branch()
        dq.set(_gdensity(dsl, b_hist, base_off, det_off, w_off, c, cam_xk + q.x(), cam_zk + q.z(), rq - a.rb, hq, False))
        dsl.end_if()
        od_v.set(od_v.get() + dq.get() * ds)
        sacc_v.set(sacc_v.get() + ds)
        ds_v.set(ds * 2.0)
        k_v.set(k + UInt32(1))
        dsl.end_loop()
        var od = od_v.get()
        var scat = od * 0.0
        var ak = dsl.f32(1.0)
        var bk = dsl.f32(1.0)
        var ck = dsl.f32(1.0)
        for _ in range(_MS_OCTAVES):
            var ph = _ghg(cl, c.g_fwd * ck) * (1.0 - c.g_mix) + _ghg(cl, c.g_back * ck) * c.g_mix
            scat = scat + bk * ph * _gexp(-(od * ak))
            ak = ak * c.ms_a
            bk = bk * c.ms_b
            ck = ck * c.ms_c
        src_v.set(src_v.get() + le * t_atm * scat)
        dsl.end_if()
        if li == 1:
            dsl.end_if()
    var hf = saturate(h) * 0.65 + 0.35
    var src = src_v.get() + (amb_dn + (amb_up - amb_dn) * hf) * c.ambient
    var st = _gexp(-(sig * dt))
    var tr = tr_v.get()
    var wgt = tr * (1.0 - st)
    l_v.set(l_v.get() + src * wgt)
    tw_v.set(tw_v.get() + t * wgt)
    ww_v.set(ww_v.get() + wgt)
    tr_v.set(tr * st)
    dsl.end_if()
    dsl.end_loop()
    dsl.end_if()
    var tc = tr_v.get()
    var t_rep = select_f(ww_v.get().gt(1e-6), tw_v.get() / max_f(ww_v.get(), 1e-6), (t0 + t1) * 0.5)
    t_rep = select_f(hit, t_rep, (c.bottom + c.top) * 0.5 / max_f(abs_f(mu), 0.05))

    # --- aerial perspective between the eye and the cloud ---
    comptime N = SOTA_ATMO_AP_DIM
    var ap_max = dsl.push_f32(12)
    var fxc = (ndc_x + 1.0) * (0.5 * Float32(N)) - 0.5
    var fyc = (1.0 - ndc_y) * (0.5 * Float32(N)) - 0.5
    var sc = min_f(sqrt_f(max_f(t_rep / ap_max, 0.0)) * Float32(N) - 1.0, Float32(N - 1))
    var s0f = clamp_f(floor_f(sc), 0.0, Float32(N - 1))
    var s0 = s0f.saturate_u32()
    var s1 = min_u(s0 + UInt32(1), dsl.u32(UInt32(N - 1)))
    var fs = saturate(sc - s0f)
    var f0 = _ap_fetch(dsl, b_hist, ap_off, fxc, fyc, s0)
    var f1 = _ap_fetch(dsl, b_hist, ap_off, fxc, fyc, s1)
    var s_ap = f0[0] * (1.0 - fs) + f1[0] * fs
    var t_ap = f0[1] * (1.0 - fs) + f1[1] * fs
    var no_ap = b_glob.load_u32(dsl.u32(UInt32(g2 + CW_NO_AP))).ne(UInt32(0))
    var s_out = select_vec3(no_ap, l_v.get(), l_v.get() * t_ap + s_ap * (1.0 - tc))

    # --- temporal reprojection (cloud point moved by the wind since the last frame) ---
    var valid = b_glob.load_u32(dsl.u32(UInt32(g2 + CW_HIST_VALID))).ne(UInt32(0))
    var alpha = b_glob.load_f32(dsl.u32(UInt32(g2 + CW_ALPHA)))
    var p_cam = b_glob.load_vec3(dsl.u32(UInt32(g2 + CW_PREV_CAM)))
    var p_fwd = b_glob.load_vec3(dsl.u32(UInt32(g2 + CW_PREV_CAM + 3)))
    var p_rgt = b_glob.load_vec3(dsl.u32(UInt32(g2 + CW_PREV_CAM + 6)))
    var p_up = b_glob.load_vec3(dsl.u32(UInt32(g2 + CW_PREV_CAM + 9)))
    var p_tx = b_glob.load_f32(dsl.u32(UInt32(g2 + CW_PREV_CAM + 12)))
    var p_ty = b_glob.load_f32(dsl.u32(UInt32(g2 + CW_PREV_CAM + 13)))
    var dwx = b_glob.load_f32(dsl.u32(UInt32(g2 + CW_DWIND_X)))
    var dwz = b_glob.load_f32(dsl.u32(UInt32(g2 + CW_DWIND_Z)))
    var pw = cam + d * (t_rep / a.kpu)
    var v = GpuVec3(pw.x() + dwx, pw.y(), pw.z() + dwz) - p_cam
    var vz = v.dot(p_fwd)
    var pnx = v.dot(p_rgt) / max_f(vz, 1e-6) / p_tx
    var pny = v.dot(p_up) / max_f(vz, 1e-6) / p_ty
    var hx = (pnx + 1.0) * 0.5 * fw / fdiv - 0.5
    var hy = (1.0 - pny) * 0.5 * fh / fdiv - 0.5
    var fcw = cw.to_f32()
    var fch = chh.to_f32()
    var on = valid & vz.gt(0.0) & hx.ge(-0.5) & hy.ge(-0.5) & hx.lt(fcw - 0.5) & hy.lt(fch - 0.5)
    var res_v = dsl.var_vec3(s_out)
    var res_t = dsl.var_f32(tc)
    dsl.if_then(on & alpha.lt(0.999))
    var hxc = min_f(max_f(hx, 0.0), fcw - 1.0)
    var hyc = min_f(max_f(hy, 0.0), fch - 1.0)
    var hx0 = floor_f(hxc).saturate_u32()
    var hy0 = floor_f(hyc).saturate_u32()
    var hx1 = min_u(hx0 + UInt32(1), cw - UInt32(1))
    var hy1 = min_u(hy0 + UInt32(1), chh - UInt32(1))
    var tx = hxc - hx0.to_f32()
    var ty = hyc - hy0.to_f32()
    var i00 = prev_off + (hy0 * cw + hx0) * UInt32(4)
    var i10 = prev_off + (hy0 * cw + hx1) * UInt32(4)
    var i01 = prev_off + (hy1 * cw + hx0) * UInt32(4)
    var i11 = prev_off + (hy1 * cw + hx1) * UInt32(4)
    var q00 = b_hist.load_f32(i00 + UInt32(3))
    var q10 = b_hist.load_f32(i10 + UInt32(3))
    var q01 = b_hist.load_f32(i01 + UInt32(3))
    var q11 = b_hist.load_f32(i11 + UInt32(3))
    var ok = q00.lt(1.5) & q10.lt(1.5) & q01.lt(1.5) & q11.lt(1.5)
    dsl.if_then(ok)
    var w00 = (1.0 - tx) * (1.0 - ty)
    var w10 = tx * (1.0 - ty)
    var w01 = (1.0 - tx) * ty
    var w11 = tx * ty
    var hs = b_hist.load_vec3(i00) * w00 + b_hist.load_vec3(i10) * w10 + b_hist.load_vec3(i01) * w01 + b_hist.load_vec3(i11) * w11
    var ht = q00 * w00 + q10 * w10 + q01 * w01 + q11 * w11
    res_v.set(hs + (s_out - hs) * alpha)
    res_t.set(ht + (tc - ht) * alpha)
    dsl.end_if()
    dsl.end_if()
    b_hist.store_vec3(out_w, res_v.get())
    b_hist.store_f32(out_w + UInt32(3), res_t.get())
    return dsl.finish()


@always_inline
def max_u_g(mut dsl: ShaderDsl, a: GpuU32, b: GpuU32) -> GpuU32:
    return select_u(a.ge(b), a, b)


def emit_cloud_composite(
    mut dsl: ShaderDsl, b_hist: BufferRef[1], cur_off: GpuU32, cw: GpuU32, chh: GpuU32, div: GpuU32,
    fx0: GpuF32, fy0: GpuF32, sky: GpuVec3,
) -> GpuVec3:
    """`L_sky * T + S'` from a bilinear (sentinel-aware) fetch of the cloud image at render pixel (fx0, fy0)."""
    var fdiv = div.to_f32()
    var fcw = cw.to_f32()
    var fch = chh.to_f32()
    var hx = min_f(max_f(fx0 / fdiv - 0.5, 0.0), fcw - 1.0)
    var hy = min_f(max_f(fy0 / fdiv - 0.5, 0.0), fch - 1.0)
    var x0 = floor_f(hx).saturate_u32()
    var y0 = floor_f(hy).saturate_u32()
    var x1 = min_u(x0 + UInt32(1), cw - UInt32(1))
    var y1 = min_u(y0 + UInt32(1), chh - UInt32(1))
    var tx = hx - x0.to_f32()
    var ty = hy - y0.to_f32()
    var i00 = cur_off + (y0 * cw + x0) * UInt32(4)
    var i10 = cur_off + (y0 * cw + x1) * UInt32(4)
    var i01 = cur_off + (y1 * cw + x0) * UInt32(4)
    var i11 = cur_off + (y1 * cw + x1) * UInt32(4)
    var q00 = b_hist.load_f32(i00 + UInt32(3))
    var q10 = b_hist.load_f32(i10 + UInt32(3))
    var q01 = b_hist.load_f32(i01 + UInt32(3))
    var q11 = b_hist.load_f32(i11 + UInt32(3))
    var w00 = select_f(q00.lt(1.5), (1.0 - tx) * (1.0 - ty), dsl.f32(0.0))
    var w10 = select_f(q10.lt(1.5), tx * (1.0 - ty), dsl.f32(0.0))
    var w01 = select_f(q01.lt(1.5), (1.0 - tx) * ty, dsl.f32(0.0))
    var w11 = select_f(q11.lt(1.5), tx * ty, dsl.f32(0.0))
    var ws = w00 + w10 + w01 + w11
    var inv = 1.0 / max_f(ws, 1e-6)
    var s = (b_hist.load_vec3(i00) * w00 + b_hist.load_vec3(i10) * w10 + b_hist.load_vec3(i01) * w01 + b_hist.load_vec3(i11) * w11) * inv
    var t = (q00 * w00 + q10 * w10 + q01 * w01 + q11 * w11) * inv
    return select_vec3(ws.gt(1e-6), sky * t + s, sky)


# ============================================================================
# Settings + state
# ============================================================================

struct SotaClouds(Copyable, Movable):
    """Volumetric-cloud cvars (`ren.sota_atmo.clouds`). Off by default."""
    var enabled: Bool
    var params: CloudParams
    var wind_mps: Vec3          # x / z used (m/s); the field moves with `scene_time_sec`
    var weather_seed: UInt32
    var res_div: Int            # 1, 2 (default) or 4
    var temporal_alpha: Float32 # weight of the new sample (1 = no history)
    var jitter: Bool
    var aerial_perspective: Bool  # haze between the eye and the clouds (off only for the CPU parity test)
    # state
    var pipe_noise: Int
    var pipe_weather: Int
    var pipe_march: Int
    var noise_ready: Bool
    var weather_seed_built: UInt32
    var weather_ready: Bool
    var frame: Int
    var parity: Int
    var hist_valid: Bool
    var last_cw: Int
    var last_ch: Int
    var last_wind_x_km: Float32
    var last_wind_z_km: Float32
    var prev_cam: List[Float32]   # 14 floats (pos, fwd, right, up, tan_x, tan_y)

    def __init__(out self):
        self.enabled = False
        self.params = CloudParams()
        self.wind_mps = Vec3(12.0, 0.0, 4.0)
        self.weather_seed = 7
        self.res_div = 2
        self.temporal_alpha = 0.15
        self.jitter = True
        self.aerial_perspective = True
        self.pipe_noise = 0
        self.pipe_weather = 0
        self.pipe_march = 0
        self.noise_ready = False
        self.weather_seed_built = 0xFFFFFFFF
        self.weather_ready = False
        self.frame = 0
        self.parity = 0
        self.hist_valid = False
        self.last_cw = 0
        self.last_ch = 0
        self.last_wind_x_km = 0.0
        self.last_wind_z_km = 0.0
        self.prev_cam = List[Float32](length=14, fill=0.0)

    def create(mut self, mut vk: VulkanGpu, pipe_layout: Int):
        self.pipe_noise = vk.create_aaa_compute_pipeline(pipe_layout, sota_cloud_noise_shader())
        self.pipe_march = vk.create_aaa_compute_pipeline(pipe_layout, sota_cloud_march_shader())

    def destroy(mut self, mut vk: VulkanGpu):
        for p in [self.pipe_noise, self.pipe_weather, self.pipe_march]:
            if p != 0:
                vk.destroy_pipeline(p)
        self.pipe_noise = 0
        self.pipe_weather = 0
        self.pipe_march = 0

    def ready(self) -> Bool:
        return self.pipe_noise != 0 and self.pipe_march != 0

    def div(self) -> Int:
        return 4 if self.res_div >= 4 else (1 if self.res_div <= 1 else 2)

    def dims(self, render_w: Int, render_h: Int) -> Tuple[Int, Int]:
        var dv = self.div()
        return ((render_w + dv - 1) // dv, (render_h + dv - 1) // dv)


def cloud_buffer_words(max_pix: Int) -> Int:
    """One cloud image (4 words per pixel) at half resolution or finer-than-quarter fallback; the recorder raises
    `res_div` when the render resolution does not fit."""
    return (max_pix // 4 + 8192) * 4


def cloud_region_words(max_pix: Int) -> Int:
    return SOTA_CLOUD_BASE_N * SOTA_CLOUD_BASE_N * SOTA_CLOUD_BASE_N + SOTA_CLOUD_DETAIL_N * SOTA_CLOUD_DETAIL_N * SOTA_CLOUD_DETAIL_N + SOTA_CLOUD_WEATHER_N * SOTA_CLOUD_WEATHER_N * 2 + 2 * cloud_buffer_words(max_pix)
