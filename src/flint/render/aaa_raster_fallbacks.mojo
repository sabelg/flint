"""Raster fallbacks for GPUs without ray tracing (`aaa_raster_fallbacks.mojo`).

Without `VK_KHR_ray_query` the AAA renderer used to lose its AO, reflections, local-light shadows, and
there was no blended pass for real meshes. This module adds four passes that need no ray tracing:

  1. GTAO (Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate Indirect Occlusion"): per
     pixel, `AAA_GTAO_SLICES` screen-space slices marched `AAA_GTAO_STEPS` taps each way over the linear
     depth the deferred pass wrote to `BIND_AAA_HDR_OUT[pix * 8 + 3]`; horizon angles are clamped to the
     hemisphere around the slice-projected depth normal and integrated analytically (cosine weighted).
     A 5x5 depth-aware spatial filter denoises it and applies a gray multi-bounce fit to the HDR color.
  2. Screen-space reflections: a closest-depth (max `1/z`) pyramid of the frame's linear depth, then a
     hierarchical-Z ray march in `(x, y, 1/z)` screen space (linear along the ray), with a thickness test,
     screen-edge and roughness fades. Each hit replaces the split-sum environment-probe specular the deferred
     pass already added (`spec * (ssr - env)`), so a miss falls back to the probe. A roughness-sized
     bilateral blur widens glossy reflections.
  3. Local-light shadow maps: the `N` most important point/spot lights get depth maps (6 cube faces or one
     spot frustum, `AAA_LOCAL_SHADOW_TILE^2` each) rendered into a D32 atlas, copied into the cluster buffer
     (`aaa_clustered_lights.mojo`), and sampled with 2x2 PCF by the clustered-light loop of the deferred pass.
  4. Alpha-blended meshes: materials flagged alpha-blend (`MAT_FLAG_ALPHA_BLEND` / `opacity`) leave the
     VisBuffer and render after opaque lighting in a forward pass, sorted back to front per object, depth
     tested against the opaque depth (no depth write), into an RGBA16F target with `SRC_ALPHA` blending; a
     compute pass composites it over the HDR frame (`hdr * (1 - A) + C`).

Toggles default to the current look: GTAO, SSR and local shadow maps are `AAA_FALLBACK_AUTO` (only on GPUs
without ray query), and the blended pass only touches materials flagged for blending.

`BIND_AAA_HISTORY` layout past the RT shadow regions (`raster_fallback_base_offset(max_pix)`, 64-word aligned):
  `[+0]`             GTAO raw visibility, 1 float / pixel
  `[+max_pix]`       SSR result, 5 floats / pixel `(dr, dg, db, weight, roughness)`
  `[+max_pix * 6]`   blended-mesh accumulation copied from the RGBA16F target, 2 words / pixel (f16x4)
  `[+max_pix * 8]`   SSR depth pyramid: a 64-word table `(offset, w | h << 16)` per level, then the levels
"""
from std.math import cos, max, min, sin, sqrt, tan
from std.memory import bitcast
from flint.core.detmath import det_acos
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    ACCESS_COLOR_ATTACHMENT_WRITE,
    ACCESS_DEPTH_ATTACHMENT_READ,
    ACCESS_DEPTH_ATTACHMENT_WRITE,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    COMPARE_GREATER_OR_EQUAL,
    COMPARE_LESS_OR_EQUAL,
    CULL_MODE_NONE,
    FORMAT_D32_SFLOAT,
    FORMAT_R16G16B16A16_SFLOAT,
    FRONT_FACE_CW,
    IMAGE_USAGE_COLOR_ATTACHMENT,
    IMAGE_USAGE_DEPTH_ATTACHMENT,
    IMAGE_USAGE_TRANSFER_SRC,
    LAYOUT_COLOR_ATTACHMENT,
    LAYOUT_DEPTH_ATTACHMENT,
    LAYOUT_SHADER_READ_ONLY,
    LAYOUT_TRANSFER_SRC,
    PIPELINE_COLOR_ATTACHMENT_OUTPUT,
    PIPELINE_COMPUTE,
    PIPELINE_EARLY_FRAGMENT_TESTS,
    PIPELINE_LATE_FRAGMENT_TESTS,
    PIPELINE_TOP,
    PIPELINE_TRANSFER,
    VulkanGpu,
    VulkanImage,
)
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_FRAGMENT,
    MODEL_GL_COMPUTE,
    MODEL_VERTEX,
    OP_LOAD,
    OP_STORE,
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    emit,
)
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_BRDF_LUT,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.aaa_vsm import emit_vsm_pcss_shadow
from flint.render.aaa_gi_svgf import AAA_FLAG_VSM
from flint.render.aaa_raytracing import emit_decode_vertex_color_rgb_spv
from flint.render.aaa_rt_shadow import rt_shadow_raw_offset, rt_shadow_total_words
from flint.render.aaa_clustered_lights import (
    AAA_LIGHT_POINT,
    AAA_LIGHT_SPOT,
    AAA_LOCAL_SHADOW_ATLAS_BASE,
    AAA_LOCAL_SHADOW_MAX_REGIONS,
    AAA_LOCAL_SHADOW_MAX_TILES,
    AAA_LOCAL_SHADOW_REGION_WORDS,
    AAA_LOCAL_SHADOW_TABLE_BASE,
    AAA_LOCAL_SHADOW_TILE,
    AaaClusterLight,
    local_shadow_point_face_basis,
    local_shadow_spot_basis,
    select_local_shadow_lights,
)

# Toggle modes for the fallbacks that change the look of existing scenes.
comptime AAA_FALLBACK_AUTO: Int = 0  # on only when the GPU has no ray query (the gate's golden scenes keep their look)
comptime AAA_FALLBACK_ON: Int = 1  # always on (also complements the RT path)
comptime AAA_FALLBACK_OFF: Int = 2

comptime AAA_GTAO_SLICES: Int = 2
comptime AAA_GTAO_STEPS: Int = 6
comptime AAA_SSR_MAX_ITERS: Int = 56
comptime AAA_SSR_MAX_LEVELS: Int = 12
comptime AAA_SSR_RESULT_WORDS: Int = 5
comptime AAA_SKY_Z: Float32 = 1.0e5  # linear depth at or past this is sky (the sky pass writes 1e6)


@always_inline
def fallback_active(mode: Int, has_ray_query: Bool) -> Bool:
    """`AAA_FALLBACK_ON` always runs; `AAA_FALLBACK_AUTO` runs only on GPUs without ray query."""
    return mode == AAA_FALLBACK_ON or (mode == AAA_FALLBACK_AUTO and not has_ray_query)


@always_inline
def raster_fallback_base_offset(max_pix: Int) -> Int:
    return (rt_shadow_raw_offset(max_pix) + rt_shadow_total_words(max_pix) + 63) & ~63


@always_inline
def gtao_raw_offset(max_pix: Int) -> Int:
    return raster_fallback_base_offset(max_pix)


@always_inline
def ssr_result_offset(max_pix: Int) -> Int:
    return raster_fallback_base_offset(max_pix) + max_pix


@always_inline
def blend_accum_offset(max_pix: Int) -> Int:
    return raster_fallback_base_offset(max_pix) + max_pix * 6


@always_inline
def ssr_pyramid_offset(max_pix: Int) -> Int:
    return raster_fallback_base_offset(max_pix) + max_pix * 8


def raster_fallback_history_words(max_pix: Int) -> Int:
    """Words `BIND_AAA_HISTORY` needs past the RT shadow regions (alignment slack included)."""
    return 64 + max_pix * 10 + 4096


def ssr_pyramid_levels(w: Int, h: Int, base: Int) -> List[Tuple[Int, Int, Int]]:
    """`(word offset, width, height)` of each closest-depth pyramid level; level `l` is `max(1, w >> l)` wide
    and its last column/row also covers the leftover texels of odd parents."""
    var out = List[Tuple[Int, Int, Int]]()
    var off = base + 64
    var l = 0
    while l < AAA_SSR_MAX_LEVELS:
        var wl = max(1, w >> l)
        var hl = max(1, h >> l)
        out.append((off, wl, hl))
        off += wl * hl
        if wl == 1 and hl == 1:
            break
        l += 1
    return out^


def gtao_cosine_visibility_cpu(h1: Float32, h2: Float32, n_ang: Float32) -> Float32:
    """CPU reference of the per-slice GTAO integral (`1.0` for an open hemisphere around the normal)."""
    var half_pi = Float32(1.5707963)
    var a = n_ang + max(h1 - n_ang, -half_pi)
    var b = n_ang + min(h2 - n_ang, half_pi)
    var cn = cos(n_ang)
    var sn = sin(n_ang)
    return Float32(0.25) * ((cn + Float32(2.0) * a * sn - cos(Float32(2.0) * a - n_ang)) + (cn + Float32(2.0) * b * sn - cos(Float32(2.0) * b - n_ang)))


# ============================================================================
# Shared SPIR-V helpers
# ============================================================================

@fieldwise_init
struct _CamIds(Copyable, Movable):
    var cx: UInt32
    var cy: UInt32
    var cz: UInt32
    var fx: UInt32
    var fy: UInt32
    var fz: UInt32
    var rx: UInt32
    var ry: UInt32
    var rz: UInt32
    var ux: UInt32
    var uy: UInt32
    var uz: UInt32
    var tan_x: UInt32
    var tan_y: UInt32
    var jit_x: UInt32
    var jit_y: UInt32


def _emit_cam(mut s: AaaSpirvModule, b_globals: UInt32) -> _CamIds:
    return _CamIds(
        s.load_float(b_globals, s.uint(32)),
        s.load_float(b_globals, s.uint(33)),
        s.load_float(b_globals, s.uint(34)),
        s.load_float(b_globals, s.uint(36)),
        s.load_float(b_globals, s.uint(37)),
        s.load_float(b_globals, s.uint(38)),
        s.load_float(b_globals, s.uint(40)),
        s.load_float(b_globals, s.uint(41)),
        s.load_float(b_globals, s.uint(42)),
        s.load_float(b_globals, s.uint(44)),
        s.load_float(b_globals, s.uint(45)),
        s.load_float(b_globals, s.uint(46)),
        s.load_float(b_globals, s.uint(39)),
        s.load_float(b_globals, s.uint(43)),
        s.load_float(b_globals, s.uint(101)),
        s.load_float(b_globals, s.uint(102)),
    )


def _emit_recon(
    mut s: AaaSpirvModule, c: _CamIds, xc: UInt32, yc: UInt32, fw: UInt32, fh: UInt32, z: UInt32
) -> Tuple[UInt32, UInt32, UInt32]:
    """World position at continuous pixel coords `(xc, yc)` (pixel centers at `+0.5`) and linear view depth `z`,
    inverting the TSR-jittered camera ray of the deferred pass (`dot(ray, fwd) == 1`)."""
    var fx0 = s.fsub(xc, s.fmul(c.jit_x, s.fmul(fw, s.float(0.5))))
    var fy0 = s.fadd(yc, s.fmul(c.jit_y, s.fmul(fh, s.float(0.5))))
    var ndc_x = s.fsub(s.fmul(s.fdiv(fx0, fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(fy0, fh), s.float(2.0)))
    var ax = s.fmul(ndc_x, c.tan_x)
    var ay = s.fmul(ndc_y, c.tan_y)
    var dx = s.fma(c.rx, ax, s.fma(c.ux, ay, c.fx))
    var dy = s.fma(c.ry, ax, s.fma(c.uy, ay, c.fy))
    var dz = s.fma(c.rz, ax, s.fma(c.uz, ay, c.fz))
    return (s.fma(dx, z, c.cx), s.fma(dy, z, c.cy), s.fma(dz, z, c.cz))


def _emit_project(
    mut s: AaaSpirvModule, c: _CamIds, fw: UInt32, fh: UInt32, qx: UInt32, qy: UInt32, qz: UInt32
) -> Tuple[UInt32, UInt32, UInt32]:
    """Continuous pixel coords and inverse linear depth `(X, Y, 1/z)` of world point `q` (inverse of `_emit_recon`)."""
    var relx = s.fsub(qx, c.cx)
    var rely = s.fsub(qy, c.cy)
    var relz = s.fsub(qz, c.cz)
    var zq = s.nmax(s.float(1e-4), emit_vec3_dot(s, relx, rely, relz, c.fx, c.fy, c.fz))
    var iz = s.fdiv(s.float(1.0), zq)
    var ndc_x = s.fdiv(s.fmul(emit_vec3_dot(s, relx, rely, relz, c.rx, c.ry, c.rz), iz), c.tan_x)
    var ndc_y = s.fdiv(s.fmul(emit_vec3_dot(s, relx, rely, relz, c.ux, c.uy, c.uz), iz), c.tan_y)
    var hw = s.fmul(fw, s.float(0.5))
    var hh = s.fmul(fh, s.float(0.5))
    var X = s.fma(s.fadd(ndc_x, s.float(1.0)), hw, s.fmul(c.jit_x, hw))
    var Y = s.fsub(s.fmul(s.fsub(s.float(1.0), ndc_y), hh), s.fmul(c.jit_y, hh))
    return (X, Y, iz)


@always_inline
def _emit_load_z(mut s: AaaSpirvModule, b_hdr: UInt32, ix: UInt32, iy: UInt32, w_u: UInt32) -> UInt32:
    return s.load_float(b_hdr, s.iadd(s.imul(s.iadd(s.imul(iy, w_u), ix), s.uint(8)), s.uint(3)))


def _emit_clamp_coord(mut s: AaaSpirvModule, v: UInt32, lim_f: UInt32) -> UInt32:
    """Float pixel coordinate -> clamped `uint` texel index in `[0, lim - 1]`."""
    return s.saturate_uint(s.nclamp(v, s.float(0.0), s.fsub(lim_f, s.float(1.0))))


def _emit_depth_normal(
    mut s: AaaSpirvModule,
    c: _CamIds,
    b_hdr: UInt32,
    px: UInt32,
    py: UInt32,
    w_u: UInt32,
    h_u: UInt32,
    fw: UInt32,
    fh: UInt32,
    z: UInt32,
    p: Tuple[UInt32, UInt32, UInt32],
) -> Tuple[UInt32, UInt32, UInt32]:
    """Geometric normal from the reconstructed positions of the 4 neighbors, picking per axis the side with the
    smaller depth step (so silhouettes do not bend it); oriented toward the camera."""
    var xf = s.float_from_uint(px)
    var yf = s.float_from_uint(py)
    var lx = _emit_clamp_coord(s, s.fsub(xf, s.float(1.0)), fw)
    var rx = _emit_clamp_coord(s, s.fadd(xf, s.float(1.0)), fw)
    var uy = _emit_clamp_coord(s, s.fsub(yf, s.float(1.0)), fh)
    var dy = _emit_clamp_coord(s, s.fadd(yf, s.float(1.0)), fh)
    var zl = _emit_load_z(s, b_hdr, lx, py, w_u)
    var zr = _emit_load_z(s, b_hdr, rx, py, w_u)
    var zu = _emit_load_z(s, b_hdr, px, uy, w_u)
    var zd = _emit_load_z(s, b_hdr, px, dy, w_u)
    var yc = s.fadd(yf, s.float(0.5))
    var xc = s.fadd(xf, s.float(0.5))
    var pl = _emit_recon(s, c, s.fadd(s.float_from_uint(lx), s.float(0.5)), yc, fw, fh, zl)
    var pr = _emit_recon(s, c, s.fadd(s.float_from_uint(rx), s.float(0.5)), yc, fw, fh, zr)
    var pu = _emit_recon(s, c, xc, s.fadd(s.float_from_uint(uy), s.float(0.5)), fw, fh, zu)
    var pd = _emit_recon(s, c, xc, s.fadd(s.float_from_uint(dy), s.float(0.5)), fw, fh, zd)
    var big = s.float(1e9)
    var dzl = s.select_f(s.ieq(lx, px), big, s.fabs(s.fsub(z, zl)))
    var dzr = s.select_f(s.ieq(rx, px), big, s.fabs(s.fsub(zr, z)))
    var dzu = s.select_f(s.ieq(uy, py), big, s.fabs(s.fsub(z, zu)))
    var dzd = s.select_f(s.ieq(dy, py), big, s.fabs(s.fsub(zd, z)))
    var use_r = s.flt(dzr, dzl)
    var use_d = s.flt(dzd, dzu)
    var ex = (
        s.select_f(use_r, s.fsub(pr[0], p[0]), s.fsub(p[0], pl[0])),
        s.select_f(use_r, s.fsub(pr[1], p[1]), s.fsub(p[1], pl[1])),
        s.select_f(use_r, s.fsub(pr[2], p[2]), s.fsub(p[2], pl[2])),
    )
    var ey = (
        s.select_f(use_d, s.fsub(pd[0], p[0]), s.fsub(p[0], pu[0])),
        s.select_f(use_d, s.fsub(pd[1], p[1]), s.fsub(p[1], pu[1])),
        s.select_f(use_d, s.fsub(pd[2], p[2]), s.fsub(p[2], pu[2])),
    )
    var cr = emit_vec3_cross(s, ey[0], ey[1], ey[2], ex[0], ex[1], ex[2])
    var n = emit_vec3_normalize(s, cr[0], cr[1], cr[2])
    var to_cam = emit_vec3_dot(s, n[0], n[1], n[2], s.fsub(c.cx, p[0]), s.fsub(c.cy, p[1]), s.fsub(c.cz, p[2]))
    var sg = s.select_f(s.flt(to_cam, s.float(0.0)), s.float(-1.0), s.float(1.0))
    return (s.fmul(n[0], sg), s.fmul(n[1], sg), s.fmul(n[2], sg))


def _emit_ign(mut s: AaaSpirvModule, px: UInt32, py: UInt32, frame_idx: UInt32) -> UInt32:
    """Interleaved gradient noise in `[0, 1)`, shifted along the golden ratio each frame."""
    var f = s.fmul(s.float_from_uint(s.bit_and(frame_idx, s.uint(63))), s.float(5.588238))
    var fx = s.fadd(s.float_from_uint(px), f)
    var fy = s.fadd(s.float_from_uint(py), f)
    return s.fract(s.fmul(s.float(52.9829189), s.fract(s.fma(fx, s.float(0.06711056), s.fmul(fy, s.float(0.00583715))))))


@always_inline
def _emit_acos(mut s: AaaSpirvModule, x: UInt32) -> UInt32:
    var xc = s.nclamp(x, s.float(-1.0), s.float(1.0))
    return s.fatan2(s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(xc, xc)))), xc)


def _emit_gray_multibounce(mut s: AaaSpirvModule, ao: UInt32) -> UInt32:
    """Jimenez 2016 multi-bounce fit for a mid-gray (0.5) albedo: brightens the darkest occlusion slightly."""
    var poly = s.fmul(s.fma(s.fma(ao, s.float(0.6878), s.float(-1.75585)), ao, s.float(2.0679)), ao)
    return s.nclamp(s.nmax(ao, poly), s.float(0.0), s.float(1.0))


# ============================================================================
# 1. GTAO
# ============================================================================

def gtao_trace_shader() -> List[UInt32]:
    """Ground-truth AO trace (`group = 8 x 8`). Push[0] = render_w, [1] = render_h, [2] = AO offset in
    `BIND_AAA_HISTORY`, [3] = world radius (f32), [4] = max screen radius in pixels (f32)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var ao_off = s.push_word(2)
    var radius = s.nmax(s.push_float(3), s.float(0.01))
    var max_px = s.push_float(4)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var z = _emit_load_z(s, b_hdr, px, py, w_u)
    var out_ao = s.fvar(s.float(1.0))

    s.begin_if(s.flt(z, s.float(AAA_SKY_Z)))
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var c = _emit_cam(s, b_globals)
    var xc = s.fadd(s.float_from_uint(px), s.float(0.5))
    var yc = s.fadd(s.float_from_uint(py), s.float(0.5))
    var p = _emit_recon(s, c, xc, yc, fw, fh, z)
    var n = _emit_depth_normal(s, c, b_hdr, px, py, w_u, h_u, fw, fh, z, p)
    var v = emit_vec3_normalize(s, s.fsub(c.cx, p[0]), s.fsub(c.cy, p[1]), s.fsub(c.cz, p[2]))
    var r_px = s.nmin(max_px, s.fdiv(s.fmul(radius, s.fmul(fh, s.float(0.5))), s.fmul(z, c.tan_y)))

    s.begin_if(s.fgt(r_px, s.float(1.0)))
    var frame_idx = s.load_word(b_globals, s.uint(100))
    var noise = _emit_ign(s, px, py, frame_idx)
    var jit = s.fract(s.fma(noise, s.float(7.31), s.float(0.37)))
    var inv_r2 = s.fdiv(s.float(1.0), s.fmul(radius, radius))
    var half_pi = s.float(1.5707963)
    var acc: UInt32 = s.float(0.0)
    var wacc: UInt32 = s.float(0.0)
    for sl in range(AAA_GTAO_SLICES):
        var phi = s.fmul(s.fadd(s.float(Float32(sl)), noise), s.float(3.14159265 / Float32(AAA_GTAO_SLICES)))
        var cs = s.fcos(phi)
        var sn = s.fsin(phi)
        # World direction of the screen line: +x = right, +y (down) = -up
        var dwx = s.fsub(s.fmul(c.rx, cs), s.fmul(c.ux, sn))
        var dwy = s.fsub(s.fmul(c.ry, cs), s.fmul(c.uy, sn))
        var dwz = s.fsub(s.fmul(c.rz, cs), s.fmul(c.uz, sn))
        var dv = emit_vec3_dot(s, dwx, dwy, dwz, v[0], v[1], v[2])
        var ortho = emit_vec3_normalize(s, s.fsub(dwx, s.fmul(v[0], dv)), s.fsub(dwy, s.fmul(v[1], dv)), s.fsub(dwz, s.fmul(v[2], dv)))
        var axis = emit_vec3_cross(s, ortho[0], ortho[1], ortho[2], v[0], v[1], v[2])
        var n_ax = emit_vec3_dot(s, n[0], n[1], n[2], axis[0], axis[1], axis[2])
        var npx = s.fsub(n[0], s.fmul(axis[0], n_ax))
        var npy = s.fsub(n[1], s.fmul(axis[1], n_ax))
        var npz = s.fsub(n[2], s.fmul(axis[2], n_ax))
        var np_len = s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, npx, npy, npz, npx, npy, npz)))
        var inv_np = s.fdiv(s.float(1.0), np_len)
        var cos_n = s.nclamp(s.fmul(emit_vec3_dot(s, npx, npy, npz, v[0], v[1], v[2]), inv_np), s.float(-1.0), s.float(1.0))
        var sgn = s.select_f(s.flt(emit_vec3_dot(s, npx, npy, npz, ortho[0], ortho[1], ortho[2]), s.float(0.0)), s.float(-1.0), s.float(1.0))
        var n_ang = s.fmul(sgn, _emit_acos(s, cos_n))
        var c_pos: UInt32 = s.float(-1.0)
        var c_neg: UInt32 = s.float(-1.0)
        for k in range(AAA_GTAO_STEPS):
            var tk = s.fmul(s.fadd(s.float(Float32(k)), jit), s.float(1.0 / Float32(AAA_GTAO_STEPS)))
            var off = s.fma(s.fmul(tk, tk), s.fsub(r_px, s.float(1.0)), s.float(1.0))
            for side in range(2):
                var sgn_side = Float32(1.0) if side == 0 else Float32(-1.0)
                var sxf = s.fma(s.fmul(cs, off), s.float(sgn_side), xc)
                var syf = s.fma(s.fmul(sn, off), s.float(sgn_side), yc)
                var inb = s.logical_and(
                    s.logical_and(s.fge(sxf, s.float(0.0)), s.flt(sxf, fw)),
                    s.logical_and(s.fge(syf, s.float(0.0)), s.flt(syf, fh)),
                )
                var ix = _emit_clamp_coord(s, sxf, fw)
                var iy = _emit_clamp_coord(s, syf, fh)
                var zs = _emit_load_z(s, b_hdr, ix, iy, w_u)
                var sp = _emit_recon(s, c, s.fadd(s.float_from_uint(ix), s.float(0.5)), s.fadd(s.float_from_uint(iy), s.float(0.5)), fw, fh, zs)
                var ddx = s.fsub(sp[0], p[0])
                var ddy = s.fsub(sp[1], p[1])
                var ddz = s.fsub(sp[2], p[2])
                var dl2 = s.nmax(s.float(1e-8), emit_vec3_dot(s, ddx, ddy, ddz, ddx, ddy, ddz))
                var cth = s.fmul(emit_vec3_dot(s, ddx, ddy, ddz, v[0], v[1], v[2]), s.inverse_sqrt(dl2))
                var fall = s.nclamp(s.fsub(s.float(1.0), s.fmul(dl2, inv_r2)), s.float(0.0), s.float(1.0))
                var ok = s.logical_and(inb, s.flt(zs, s.float(AAA_SKY_Z)))
                var c_eff = s.select_f(ok, s.fma(s.fadd(cth, s.float(1.0)), fall, s.float(-1.0)), s.float(-1.0))
                if side == 0:
                    c_pos = s.nmax(c_pos, c_eff)
                else:
                    c_neg = s.nmax(c_neg, c_eff)
        var h2 = _emit_acos(s, c_pos)
        var h1 = s.fneg(_emit_acos(s, c_neg))
        h1 = s.fadd(n_ang, s.nmax(s.fsub(h1, n_ang), s.fneg(half_pi)))
        h2 = s.fadd(n_ang, s.nmin(s.fsub(h2, n_ang), half_pi))
        var sin_n = s.fsin(n_ang)
        var a1 = s.fsub(s.fma(s.fmul(s.float(2.0), h1), sin_n, cos_n), s.fcos(s.fsub(s.fmul(s.float(2.0), h1), n_ang)))
        var a2 = s.fsub(s.fma(s.fmul(s.float(2.0), h2), sin_n, cos_n), s.fcos(s.fsub(s.fmul(s.float(2.0), h2), n_ang)))
        var vis = s.fmul(s.float(0.25), s.fadd(a1, a2))
        acc = s.fma(np_len, vis, acc)
        wacc = s.fadd(wacc, np_len)
    s.store(out_ao, s.nclamp(s.fdiv(acc, s.nmax(wacc, s.float(1e-4))), s.float(0.0), s.float(1.0)))
    s.end_if()
    s.end_if()
    s.store_float(b_hist, s.iadd(ao_off, pix), s.load_f(out_ao))
    return s.finish()


def gtao_denoise_apply_shader() -> List[UInt32]:
    """5x5 depth-aware spatial GTAO filter + application to the HDR color (`group = 8 x 8`).
    Push[0] = render_w, [1] = render_h, [2] = AO offset, [3] = strength (f32)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var ao_off = s.push_word(2)
    var strength = s.nclamp(s.push_float(3), s.float(0.0), s.float(1.0))
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var z = _emit_load_z(s, b_hdr, px, py, w_u)
    s.return_if(s.fge(z, s.float(AAA_SKY_Z)))
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var xf = s.float_from_uint(px)
    var yf = s.float_from_uint(py)
    var tol = s.nmax(s.float(1e-3), s.fmul(z, s.float(0.06)))
    var sum: UInt32 = s.float(0.0)
    var wsum: UInt32 = s.float(0.0)
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            var ix = _emit_clamp_coord(s, s.fadd(xf, s.float(Float32(ox))), fw)
            var iy = _emit_clamp_coord(s, s.fadd(yf, s.float(Float32(oy))), fh)
            var zs = _emit_load_z(s, b_hdr, ix, iy, w_u)
            var a = s.load_float(b_hist, s.iadd(ao_off, s.iadd(s.imul(iy, w_u), ix)))
            var wk = s.nclamp(s.fsub(s.float(1.0), s.fdiv(s.fabs(s.fsub(zs, z)), tol)), s.float(0.0), s.float(1.0))
            sum = s.fma(a, wk, sum)
            wsum = s.fadd(wsum, wk)
    var ao = _emit_gray_multibounce(s, s.fdiv(sum, s.nmax(wsum, s.float(1e-4))))
    var k = s.fma(s.fsub(ao, s.float(1.0)), strength, s.float(1.0))
    var base = s.imul(s.iadd(s.imul(py, w_u), px), s.uint(8))
    s.store_float(b_hdr, base, s.fmul(s.load_float(b_hdr, base), k))
    s.store_float(b_hdr, s.iadd(base, s.uint(1)), s.fmul(s.load_float(b_hdr, s.iadd(base, s.uint(1))), k))
    s.store_float(b_hdr, s.iadd(base, s.uint(2)), s.fmul(s.load_float(b_hdr, s.iadd(base, s.uint(2))), k))
    return s.finish()


# ============================================================================
# 2. Screen-space reflections
# ============================================================================

def ssr_pyramid_shader() -> List[UInt32]:
    """One level of the closest-depth pyramid (`group = 8 x 8`), storing `1/z` (`0` for sky).
    Push[0] = dst_w, [1] = dst_h, [2] = src_w, [3] = src_h, [4] = src offset, [5] = dst offset,
    [6] = 1 to build level 0 from `BIND_AAA_HDR_OUT` (src = the render size)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var dw = s.push_word(0)
    var dh = s.push_word(1)
    var sw = s.push_word(2)
    var sh = s.push_word(3)
    var src_off = s.push_word(4)
    var dst_off = s.push_word(5)
    var lvl0 = s.ine(s.push_word(6), s.uint(0))
    s.return_if(s.logical_or(s.uge(px, dw), s.uge(py, dh)))
    var dst = s.iadd(dst_off, s.iadd(s.imul(py, dw), px))
    s.begin_if(lvl0)
    var z = _emit_load_z(s, b_hdr, px, py, sw)
    s.store_float(b_hist, dst, s.select_f(s.flt(z, s.float(AAA_SKY_Z)), s.fdiv(s.float(1.0), s.nmax(z, s.float(1e-4))), s.float(0.0)))
    s.begin_else()
    var x2 = s.imul(px, s.uint(2))
    var y2 = s.imul(py, s.uint(2))
    var swm = s.isub(sw, s.uint(1))
    var shm = s.isub(sh, s.uint(1))
    # The last column/row also covers the leftover texel of an odd parent
    var last_x = s.logical_and(s.ieq(px, s.isub(dw, s.uint(1))), s.ine(s.bit_and(sw, s.uint(1)), s.uint(0)))
    var last_y = s.logical_and(s.ieq(py, s.isub(dh, s.uint(1))), s.ine(s.bit_and(sh, s.uint(1)), s.uint(0)))
    var xs = List[UInt32]()
    xs.append(s.umin(x2, swm))
    xs.append(s.umin(s.iadd(x2, s.uint(1)), swm))
    xs.append(s.umin(s.select_u(last_x, s.iadd(x2, s.uint(2)), s.iadd(x2, s.uint(1))), swm))
    var ys = List[UInt32]()
    ys.append(s.umin(y2, shm))
    ys.append(s.umin(s.iadd(y2, s.uint(1)), shm))
    ys.append(s.umin(s.select_u(last_y, s.iadd(y2, s.uint(2)), s.iadd(y2, s.uint(1))), shm))
    var m: UInt32 = s.float(0.0)
    for j in range(3):
        for i in range(3):
            m = s.nmax(m, s.load_float(b_hist, s.iadd(src_off, s.iadd(s.imul(ys[j], sw), xs[i]))))
    s.store_float(b_hist, dst, m)
    s.end_if()
    return s.finish()


def _emit_view_normal_from_hdr(
    mut s: AaaSpirvModule,
    b_hdr: UInt32,
    base: UInt32,
    dn: Tuple[UInt32, UInt32, UInt32],
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Shading normal from the deferred pass's `(nx, ny)` with `nz` signed like the depth normal; falls back to
    the depth normal when the result faces away from the camera."""
    var hnx = s.load_float(b_hdr, s.iadd(base, s.uint(6)))
    var hny = s.load_float(b_hdr, s.iadd(base, s.uint(7)))
    var nz_mag = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(hnx, hnx, s.fmul(hny, hny)))))
    var hnz = s.select_f(s.flt(dn[2], s.float(0.0)), s.fneg(nz_mag), nz_mag)
    var faces = s.fgt(emit_vec3_dot(s, hnx, hny, hnz, vx, vy, vz), s.float(0.02))
    return (s.select_f(faces, hnx, dn[0]), s.select_f(faces, hny, dn[1]), s.select_f(faces, hnz, dn[2]))


def ssr_trace_shader() -> List[UInt32]:
    """Hierarchical-Z screen-space reflection trace (`group = 8 x 8`).
    Push[0] = render_w, [1] = render_h, [2] = pyramid base (table), [3] = result offset, [4] = level count,
    [5] = max roughness (f32), [6] = thickness in meters (f32), [7] = max distance in meters (f32), [8] = intensity (f32)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var brdf_lut = s.sampled_image_2d(BIND_AAA_BRDF_LUT)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var pyr = s.push_word(2)
    var res_off = s.push_word(3)
    var n_lvls = s.push_word(4)
    var max_rough = s.push_float(5)
    var thick = s.push_float(6)
    var max_dist = s.push_float(7)
    var intensity = s.push_float(8)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var ro = s.iadd(res_off, s.imul(pix, s.uint(UInt32(AAA_SSR_RESULT_WORDS))))
    var base = s.imul(pix, s.uint(8))
    var z = s.load_float(b_hdr, s.iadd(base, s.uint(3)))
    var out_r = s.fvar(s.float(0.0))
    var out_g = s.fvar(s.float(0.0))
    var out_b = s.fvar(s.float(0.0))
    var out_w = s.fvar(s.float(0.0))
    var out_rough = s.fvar(s.float(1.0))

    var vis_uv4 = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var vis0 = s.extract_u(vis_uv4, 0)
    var vis1 = s.extract_u(vis_uv4, 1)
    s.begin_if(s.logical_and(s.flt(z, s.float(AAA_SKY_Z)), s.ine(vis0, s.uint(0))))
    var inst_id = s.bit_or(s.shr(vis0, s.uint(22)), s.shl(s.bit_and(s.shr(vis1, s.uint(8)), s.uint(0xFF)), s.uint(10)))
    var mat_id = s.bit_and(s.load_word(b_instances, s.iadd(s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS))), s.uint(39))), s.uint(0xFFFF))
    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var rough = s.nclamp(s.fabs(s.load_float(b_materials, s.iadd(m_base, s.uint(4)))), s.float(0.02), s.float(1.0))
    var metal = s.nclamp(s.load_float(b_materials, s.iadd(m_base, s.uint(5))), s.float(0.0), s.float(1.0))
    var alb_r = s.load_float(b_materials, m_base)
    var alb_g = s.load_float(b_materials, s.iadd(m_base, s.uint(1)))
    var alb_b = s.load_float(b_materials, s.iadd(m_base, s.uint(2)))
    s.store(out_rough, rough)

    s.begin_if(s.flt(rough, max_rough))
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var c = _emit_cam(s, b_globals)
    var xc = s.fadd(s.float_from_uint(px), s.float(0.5))
    var yc = s.fadd(s.float_from_uint(py), s.float(0.5))
    var p = _emit_recon(s, c, xc, yc, fw, fh, z)
    var v = emit_vec3_normalize(s, s.fsub(c.cx, p[0]), s.fsub(c.cy, p[1]), s.fsub(c.cz, p[2]))
    var dn = _emit_depth_normal(s, c, b_hdr, px, py, w_u, h_u, fw, fh, z, p)
    var n = _emit_view_normal_from_hdr(s, b_hdr, base, dn, v[0], v[1], v[2])
    var ndv = s.nclamp(emit_vec3_dot(s, n[0], n[1], n[2], v[0], v[1], v[2]), s.float(0.0), s.float(1.0))
    var two_nv = s.fmul(s.float(2.0), ndv)
    var rx = s.fsub(s.fmul(n[0], two_nv), v[0])
    var ry = s.fsub(s.fmul(n[1], two_nv), v[1])
    var rz = s.fsub(s.fmul(n[2], two_nv), v[2])

    # Ray end: stay in front of the near plane, then clip to the screen in (X, Y, 1/z) space
    var r_fwd = emit_vec3_dot(s, rx, ry, rz, c.fx, c.fy, c.fz)
    var near_z = s.load_float(b_globals, s.uint(35))
    var len_lim = s.select_f(
        s.flt(r_fwd, s.float(-1e-4)),
        s.fdiv(s.fsub(z, s.fmul(near_z, s.float(1.5))), s.fneg(r_fwd)),
        max_dist,
    )
    var ray_len = s.nclamp(len_lim, s.float(0.0), max_dist)
    var s0 = _emit_project(s, c, fw, fh, p[0], p[1], p[2])
    var s1 = _emit_project(s, c, fw, fh, s.fma(rx, ray_len, p[0]), s.fma(ry, ray_len, p[1]), s.fma(rz, ray_len, p[2]))
    var ddx = s.fsub(s1[0], s0[0])
    var ddy = s.fsub(s1[1], s0[1])
    var ddz = s.fsub(s1[2], s0[2])
    var big = s.float(1e9)
    var fx_hi = s.select_f(s.fgt(ddx, s.float(1e-6)), s.fdiv(s.fsub(fw, s.float(0.001)), ddx), big)
    var fx_lo = s.select_f(s.flt(ddx, s.float(-1e-6)), s.fdiv(s.fneg(s0[0]), ddx), big)
    var fy_hi = s.select_f(s.fgt(ddy, s.float(1e-6)), s.fdiv(s.fsub(fh, s.float(0.001)), ddy), big)
    var fy_lo = s.select_f(s.flt(ddy, s.float(-1e-6)), s.fdiv(s.fneg(s0[1]), ddy), big)
    var clip_f = s.nclamp(s.nmin(s.nmin(fx_hi, fx_lo), s.nmin(fy_hi, fy_lo)), s.float(0.0), s.float(1.0))
    ddx = s.fmul(ddx, clip_f)
    ddy = s.fmul(ddy, clip_f)
    ddz = s.fmul(ddz, clip_f)
    var seg = s.nmax(s.fabs(ddx), s.fabs(ddy))

    s.begin_if(s.fgt(seg, s.float(2.0)))
    var inv_seg = s.fdiv(s.float(1.0), seg)
    var dx = s.fmul(ddx, inv_seg)
    var dy = s.fmul(ddy, inv_seg)
    var diz = s.fmul(ddz, inv_seg)
    var max_lvl = s.isub(s.umax(n_lvls, s.uint(1)), s.uint(1))
    var t_v = s.fvar(s.float(1.5))
    var lvl = s.uvar(s.uint(0))
    var it = s.uvar(s.uint(0))
    var hit = s.uvar(s.uint(0))
    var hit_t = s.fvar(s.float(0.0))
    s.begin_loop()
    var it_c = s.load_u(it)
    var tc = s.load_f(t_v)
    s.break_if(s.logical_or(s.logical_or(s.uge(it_c, s.uint(UInt32(AAA_SSR_MAX_ITERS))), s.fgt(tc, seg)), s.ine(s.load_u(hit), s.uint(0))))
    s.store(it, s.iadd(it_c, s.uint(1)))
    var lv = s.load_u(lvl)
    var tab = s.iadd(pyr, s.imul(lv, s.uint(2)))
    var l_off = s.load_word(b_hist, tab)
    var l_dims = s.load_word(b_hist, s.iadd(tab, s.uint(1)))
    var lw = s.bit_and(l_dims, s.uint(0xFFFF))
    var lh = s.shr(l_dims, s.uint(16))
    var csz = s.float_from_uint(s.shl(s.uint(1), lv))
    var X = s.fma(dx, tc, s0[0])
    var Y = s.fma(dy, tc, s0[1])
    var IZ = s.fma(diz, tc, s0[2])
    var cx = s.umin(s.saturate_uint(s.ffloor(s.fdiv(X, csz))), s.isub(lw, s.uint(1)))
    var cy = s.umin(s.saturate_uint(s.ffloor(s.fdiv(Y, csz))), s.isub(lh, s.uint(1)))
    var cxf = s.float_from_uint(cx)
    var cyf = s.float_from_uint(cy)
    # Cell exit along the ray (the last cell of a level reaches the screen edge)
    var bx = s.select_f(
        s.fge(dx, s.float(0.0)),
        s.select_f(s.ieq(cx, s.isub(lw, s.uint(1))), fw, s.fmul(s.fadd(cxf, s.float(1.0)), csz)),
        s.fmul(cxf, csz),
    )
    var by = s.select_f(
        s.fge(dy, s.float(0.0)),
        s.select_f(s.ieq(cy, s.isub(lh, s.uint(1))), fh, s.fmul(s.fadd(cyf, s.float(1.0)), csz)),
        s.fmul(cyf, csz),
    )
    var tx = s.select_f(s.fgt(s.fabs(dx), s.float(1e-6)), s.fdiv(s.fsub(bx, s0[0]), dx), big)
    var ty = s.select_f(s.fgt(s.fabs(dy), s.float(1e-6)), s.fdiv(s.fsub(by, s0[1]), dy), big)
    var t_exit = s.nmax(s.nmin(tx, ty), s.fadd(tc, s.float(0.01)))
    var iz_exit = s.fma(diz, t_exit, s0[2])
    var surf_iz = s.load_float(b_hist, s.iadd(l_off, s.iadd(s.imul(cy, lw), cx)))
    var t_next = s.fadd(t_exit, s.float(0.02))
    var in_front = s.fgt(s.nmin(IZ, iz_exit), surf_iz)
    s.begin_if(in_front)
    s.store(t_v, t_next)
    s.store(lvl, s.umin(s.iadd(lv, s.uint(1)), max_lvl))
    s.begin_else()
    var t_cross = s.select_f(s.flt(diz, s.float(-1e-12)), s.fdiv(s.fsub(surf_iz, s0[2]), diz), tc)
    s.begin_if(s.ieq(lv, s.uint(0)))
    s.begin_if(s.fgt(IZ, surf_iz))
    s.store(hit_t, s.nclamp(t_cross, tc, t_exit))
    s.store(hit, s.uint(1))
    s.begin_else()
    var dz_behind = s.fsub(s.fdiv(s.float(1.0), s.nmax(IZ, s.float(1e-6))), s.fdiv(s.float(1.0), s.nmax(surf_iz, s.float(1e-6))))
    s.begin_if(s.flt(dz_behind, thick))
    s.store(hit_t, tc)
    s.store(hit, s.uint(1))
    s.begin_else()
    s.store(t_v, t_next)
    s.end_if()
    s.end_if()
    s.begin_else()
    s.begin_if(s.fgt(IZ, surf_iz))
    s.store(t_v, s.nmax(tc, s.fsub(t_cross, s.float(0.05))))
    s.end_if()
    s.store(lvl, s.isub(lv, s.uint(1)))
    s.end_if()
    s.end_if()
    s.end_loop()

    s.begin_if(s.ine(s.load_u(hit), s.uint(0)))
    var th = s.load_f(hit_t)
    var hx = s.fma(dx, th, s0[0])
    var hy = s.fma(dy, th, s0[1])
    var hix = _emit_clamp_coord(s, hx, fw)
    var hiy = _emit_clamp_coord(s, hy, fh)
    var hbase = s.imul(s.iadd(s.imul(hiy, w_u), hix), s.uint(8))
    var hz = s.load_float(b_hdr, s.iadd(hbase, s.uint(3)))
    s.begin_if(s.flt(hz, s.float(AAA_SKY_Z)))
    var sr = s.load_float(b_hdr, hbase)
    var sg = s.load_float(b_hdr, s.iadd(hbase, s.uint(1)))
    var sb = s.load_float(b_hdr, s.iadd(hbase, s.uint(2)))
    # Fades: screen edge, end of the ray, roughness, and reflections turning back to the camera
    var edge = s.nmin(s.nmin(hx, s.fsub(fw, hx)), s.nmin(hy, s.fsub(fh, hy)))
    var edge_f = s.nclamp(s.fdiv(edge, s.nmax(s.float(1.0), s.fmul(fh, s.float(0.08)))), s.float(0.0), s.float(1.0))
    var end_f = s.nclamp(s.fmul(s.fsub(s.float(1.0), s.fdiv(th, seg)), s.float(6.0)), s.float(0.0), s.float(1.0))
    var rough_f = s.nclamp(s.fsub(s.float(1.0), s.fdiv(rough, s.nmax(max_rough, s.float(0.01)))), s.float(0.0), s.float(1.0))
    var back_f = s.nclamp(s.fma(r_fwd, s.float(2.0), s.float(1.4)), s.float(0.0), s.float(1.0))
    var wgt = s.fmul(s.fmul(edge_f, end_f), s.fmul(s.fmul(rough_f, s.fsqrt(rough_f)), back_f))
    # Replace the split-sum environment probe the deferred pass added: spec * (ssr - env)
    var lut = s.sample_2d_lod(brdf_lut, s.vec2(s.nclamp(ndv, s.float(0.01), s.float(0.99)), s.nclamp(rough, s.float(0.01), s.float(0.99))), s.float(0.0))
    var lut_a = s.extract_f(lut, 0)
    var lut_b = s.extract_f(lut, 1)
    var f0r = s.fma(s.fsub(alb_r, s.float(0.04)), metal, s.float(0.04))
    var f0g = s.fma(s.fsub(alb_g, s.float(0.04)), metal, s.float(0.04))
    var f0b = s.fma(s.fsub(alb_b, s.float(0.04)), metal, s.float(0.04))
    var max_mip = s.load_float(b_globals, s.uint(103))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var env = s.sample_cube_lod(env_cube, s.vec3(rx, ry, rz), s.fmul(rough, max_mip))
    var kw = s.fmul(wgt, intensity)
    s.store(out_r, s.fmul(s.fma(f0r, lut_a, lut_b), s.fmul(s.fsub(sr, s.fmul(s.extract_f(env, 0), ibl_int)), kw)))
    s.store(out_g, s.fmul(s.fma(f0g, lut_a, lut_b), s.fmul(s.fsub(sg, s.fmul(s.extract_f(env, 1), ibl_int)), kw)))
    s.store(out_b, s.fmul(s.fma(f0b, lut_a, lut_b), s.fmul(s.fsub(sb, s.fmul(s.extract_f(env, 2), ibl_int)), kw)))
    s.store(out_w, wgt)
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()
    s.store_float(b_hist, ro, s.load_f(out_r))
    s.store_float(b_hist, s.iadd(ro, s.uint(1)), s.load_f(out_g))
    s.store_float(b_hist, s.iadd(ro, s.uint(2)), s.load_f(out_b))
    s.store_float(b_hist, s.iadd(ro, s.uint(3)), s.load_f(out_w))
    s.store_float(b_hist, s.iadd(ro, s.uint(4)), s.load_f(out_rough))
    return s.finish()


def ssr_composite_shader() -> List[UInt32]:
    """Roughness-sized bilateral gather of the SSR deltas added onto the HDR color (`group = 8 x 8`).
    Push[0] = render_w, [1] = render_h, [2] = result offset."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var res_off = s.push_word(2)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var base = s.imul(pix, s.uint(8))
    var z = s.load_float(b_hdr, s.iadd(base, s.uint(3)))
    s.return_if(s.fge(z, s.float(AAA_SKY_Z)))
    var ro = s.iadd(res_off, s.imul(pix, s.uint(UInt32(AAA_SSR_RESULT_WORDS))))
    var rough = s.load_float(b_hist, s.iadd(ro, s.uint(4)))
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var xf = s.float_from_uint(px)
    var yf = s.float_from_uint(py)
    var step = s.fma(rough, s.float(10.0), s.float(0.0))
    var tol = s.nmax(s.float(1e-3), s.fmul(z, s.float(0.05)))
    var acc_r: UInt32 = s.float(0.0)
    var acc_g: UInt32 = s.float(0.0)
    var acc_b: UInt32 = s.float(0.0)
    var wsum: UInt32 = s.float(0.0)
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            var ix = _emit_clamp_coord(s, s.fma(step, s.float(Float32(ox)), xf), fw)
            var iy = _emit_clamp_coord(s, s.fma(step, s.float(Float32(oy)), yf), fh)
            var sp = s.iadd(s.imul(iy, w_u), ix)
            var zs = s.load_float(b_hdr, s.iadd(s.imul(sp, s.uint(8)), s.uint(3)))
            var so = s.iadd(res_off, s.imul(sp, s.uint(UInt32(AAA_SSR_RESULT_WORDS))))
            var wk = s.nclamp(s.fsub(s.float(1.0), s.fdiv(s.fabs(s.fsub(zs, z)), tol)), s.float(0.0), s.float(1.0))
            if ox == 0 and oy == 0:
                wk = s.float(1.0)
            acc_r = s.fma(s.load_float(b_hist, so), wk, acc_r)
            acc_g = s.fma(s.load_float(b_hist, s.iadd(so, s.uint(1))), wk, acc_g)
            acc_b = s.fma(s.load_float(b_hist, s.iadd(so, s.uint(2))), wk, acc_b)
            wsum = s.fadd(wsum, wk)
    var inv_w = s.fdiv(s.float(1.0), wsum)
    var r0 = s.load_float(b_hdr, base)
    var g0 = s.load_float(b_hdr, s.iadd(base, s.uint(1)))
    var b0 = s.load_float(b_hdr, s.iadd(base, s.uint(2)))
    s.store_float(b_hdr, base, s.nmax(s.float(0.0), s.fma(acc_r, inv_w, r0)))
    s.store_float(b_hdr, s.iadd(base, s.uint(1)), s.nmax(s.float(0.0), s.fma(acc_g, inv_w, g0)))
    s.store_float(b_hdr, s.iadd(base, s.uint(2)), s.nmax(s.float(0.0), s.fma(acc_b, inv_w, b0)))
    return s.finish()


# ============================================================================
# 3. Local-light shadow maps
# ============================================================================

def local_shadow_depth_vertex_shader() -> List[UInt32]:
    """Depth-only vertex shader for one shadow tile (non-indexed draw of `3 * triangle_count` vertices).
    Push[0] = instance, [1] = mesh first index word, [2..4] = light position, [5..7] = forward axis `A`,
    [8..10] = right axis `T` (`B = A x T`), [11] = 1 / tan(half fov), [12] = near, [13] = far.
    Writes `z_ndc = f / (f - n) * (1 - n / w)` with `w = dot(p - light, A)`."""
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var vid = s.vertex_index()
    var inst = s.push_word(0)
    var first_idx = s.push_word(1)
    var real_vtx = s.load_word(b_indices, s.iadd(first_idx, vid))
    var v_base = s.imul(real_vtx, s.uint(12))
    var inst_base = s.imul(inst, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var wp = emit_mat4_mul_point(
        s, b_instances, inst_base,
        s.load_float(b_verts, v_base), s.load_float(b_verts, s.iadd(v_base, s.uint(1))), s.load_float(b_verts, s.iadd(v_base, s.uint(2))),
    )
    var relx = s.fsub(wp[0], s.push_float(2))
    var rely = s.fsub(wp[1], s.push_float(3))
    var relz = s.fsub(wp[2], s.push_float(4))
    var ax = s.push_float(5)
    var ay = s.push_float(6)
    var az = s.push_float(7)
    var tx = s.push_float(8)
    var ty = s.push_float(9)
    var tz = s.push_float(10)
    var inv_tan = s.push_float(11)
    var near = s.push_float(12)
    var far = s.push_float(13)
    var bv = emit_vec3_cross(s, ax, ay, az, tx, ty, tz)
    var cw = emit_vec3_dot(s, relx, rely, relz, ax, ay, az)
    var cx = s.fmul(emit_vec3_dot(s, relx, rely, relz, tx, ty, tz), inv_tan)
    var cy = s.fmul(emit_vec3_dot(s, relx, rely, relz, bv[0], bv[1], bv[2]), inv_tan)
    var k = s.fdiv(far, s.nmax(s.float(1e-4), s.fsub(far, near)))
    var cz = s.fmul(k, s.fsub(cw, near))
    s.write_clip_position(cx, cy, cz, cw)
    return s.finish()


@fieldwise_init
struct AaaLocalShadowRegion(Copyable, Movable):
    var light_index: Int
    var is_point: Bool
    var first_tile: Int
    var position: Vec3
    var axis_a: Vec3
    var axis_t: Vec3
    var inv_tan: Float32
    var near: Float32
    var far: Float32

    def table_words(self) -> List[UInt32]:
        var out = List[UInt32](length=AAA_LOCAL_SHADOW_REGION_WORDS, fill=0)
        out[0] = UInt32(self.first_tile)
        out[1] = UInt32(1) if self.is_point else UInt32(0)
        out[2] = bitcast[DType.uint32](self.near)
        out[3] = bitcast[DType.uint32](self.far)
        out[4] = bitcast[DType.uint32](self.inv_tan)
        out[5] = bitcast[DType.uint32](self.axis_a.x)
        out[6] = bitcast[DType.uint32](self.axis_a.y)
        out[7] = bitcast[DType.uint32](self.axis_a.z)
        out[8] = bitcast[DType.uint32](self.axis_t.x)
        out[9] = bitcast[DType.uint32](self.axis_t.y)
        out[10] = bitcast[DType.uint32](self.axis_t.z)
        out[11] = bitcast[DType.uint32](Float32(1.0))
        return out^


# ============================================================================
# 4. Alpha-blended meshes
# ============================================================================

def blend_mesh_vertex_shader() -> List[UInt32]:
    """Forward blended-mesh vertex shader (non-indexed draw of `3 * triangle_count` vertices).
    Push[0] = instance, [1] = mesh first index word, [2] = material, [3] = opacity (f32).
    Outputs world position (loc 0), world normal (loc 1), uv (loc 2) and linear vertex color (loc 3)."""
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var out_wp = s.add_stage_io(STORAGE_OUTPUT, s.type_vec3, location=0)
    var out_n = s.add_stage_io(STORAGE_OUTPUT, s.type_vec3, location=1)
    var out_uv = s.add_stage_io(STORAGE_OUTPUT, s.type_vec2, location=2)
    var out_vc = s.add_stage_io(STORAGE_OUTPUT, s.type_vec3, location=3)
    var vid = s.vertex_index()
    var inst = s.push_word(0)
    var first_idx = s.push_word(1)
    var real_vtx = s.load_word(b_indices, s.iadd(first_idx, vid))
    var v_base = s.imul(real_vtx, s.uint(12))
    var inst_base = s.imul(inst, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var wp = emit_mat4_mul_point(
        s, b_instances, inst_base,
        s.load_float(b_verts, v_base), s.load_float(b_verts, s.iadd(v_base, s.uint(1))), s.load_float(b_verts, s.iadd(v_base, s.uint(2))),
    )
    var wn = emit_mat4_mul_dir(
        s, b_instances, inst_base,
        s.load_float(b_verts, s.iadd(v_base, s.uint(3))), s.load_float(b_verts, s.iadd(v_base, s.uint(4))), s.load_float(b_verts, s.iadd(v_base, s.uint(5))),
    )
    var clip = emit_mat4_mul_point(s, b_globals, s.uint(0), wp[0], wp[1], wp[2])
    var jx = s.load_float(b_globals, s.uint(101))
    var jy = s.load_float(b_globals, s.uint(102))
    s.write_clip_position(s.fma(jx, clip[3], clip[0]), s.fma(jy, clip[3], clip[1]), clip[2], clip[3])
    var meta = s.load_word(b_verts, s.iadd(v_base, s.uint(11)))
    var vc = emit_decode_vertex_color_rgb_spv(s, meta, s.ieq(s.uint(1), s.uint(0)))
    emit(s.code, OP_STORE, out_wp[1], s.vec3(wp[0], wp[1], wp[2]))
    emit(s.code, OP_STORE, out_n[1], s.vec3(wn[0], wn[1], wn[2]))
    emit(s.code, OP_STORE, out_uv[1], s.vec2(s.load_float(b_verts, s.iadd(v_base, s.uint(9))), s.load_float(b_verts, s.iadd(v_base, s.uint(10)))))
    emit(s.code, OP_STORE, out_vc[1], s.vec3(vc[0], vc[1], vc[2]))
    return s.finish()


def blend_mesh_fragment_shader() -> List[UInt32]:
    """Forward blended-mesh shading from the same inputs as the deferred pass: material base color / albedo
    texture / vertex color, roughness, metalness, emissive; VSM-shadowed sun GGX, SH9 ambient and the
    split-sum environment probe. Writes straight `(rgb, alpha)` for `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` blending;
    the specular is divided by alpha so thin glass keeps its reflections."""
    var s = AaaSpirvModule(MODEL_FRAGMENT, push_words=AAA_PUSH_WORDS, enable_bindless=True, early_depth=True)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var brdf_lut = s.sampled_image_2d(BIND_AAA_BRDF_LUT)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var in_wp = s.add_stage_io(STORAGE_INPUT, s.type_vec3, location=0)
    var in_n = s.add_stage_io(STORAGE_INPUT, s.type_vec3, location=1)
    var in_uv = s.add_stage_io(STORAGE_INPUT, s.type_vec2, location=2)
    var in_vc = s.add_stage_io(STORAGE_INPUT, s.type_vec3, location=3)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_vec4, location=0)

    var wp_v = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec3, wp_v, in_wp[1])
    var n_v = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec3, n_v, in_n[1])
    var uv_v = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec2, uv_v, in_uv[1])
    var vc_v = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec3, vc_v, in_vc[1])
    var wx = s.extract_f(wp_v, 0)
    var wy = s.extract_f(wp_v, 1)
    var wz = s.extract_f(wp_v, 2)
    var uv_x = s.extract_f(uv_v, 0)
    var uv_y = s.extract_f(uv_v, 1)
    var ddx_u = s.dpdx(uv_x)
    var ddx_v = s.dpdx(uv_y)
    var ddy_u = s.dpdy(uv_x)
    var ddy_v = s.dpdy(uv_y)

    var mat = s.push_word(2)
    var opacity = s.nclamp(s.push_float(3), s.float(0.0), s.float(1.0))
    var m_base = s.imul(mat, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var tex_pk = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))
    var t_alb = s.bit_and(tex_pk, s.uint(0xFF))
    var alb_r = s.fvar(s.fmul(s.load_float(b_materials, m_base), s.extract_f(vc_v, 0)))
    var alb_g = s.fvar(s.fmul(s.load_float(b_materials, s.iadd(m_base, s.uint(1))), s.extract_f(vc_v, 1)))
    var alb_b = s.fvar(s.fmul(s.load_float(b_materials, s.iadd(m_base, s.uint(2))), s.extract_f(vc_v, 2)))
    var alpha_v = s.fvar(opacity)
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var tx = s.sample_bindless_grad(bindless_arr, t_alb, uv_v, s.vec2(ddx_u, ddx_v), s.vec2(ddy_u, ddy_v))
    s.store(alb_r, s.fmul(s.load_f(alb_r), s.extract_f(tx, 0)))
    s.store(alb_g, s.fmul(s.load_f(alb_g), s.extract_f(tx, 1)))
    s.store(alb_b, s.fmul(s.load_f(alb_b), s.extract_f(tx, 2)))
    s.store(alpha_v, s.fmul(opacity, s.extract_f(tx, 3)))
    s.end_if()
    var ar = s.load_f(alb_r)
    var ag = s.load_f(alb_g)
    var ab = s.load_f(alb_b)
    var alpha = s.load_f(alpha_v)
    var rough = s.nclamp(s.fabs(s.load_float(b_materials, s.iadd(m_base, s.uint(4)))), s.float(0.04), s.float(1.0))
    var metal = s.nclamp(s.load_float(b_materials, s.iadd(m_base, s.uint(5))), s.float(0.0), s.float(1.0))
    var em_r = s.load_float(b_materials, s.iadd(m_base, s.uint(8)))
    var em_g = s.load_float(b_materials, s.iadd(m_base, s.uint(9)))
    var em_b = s.load_float(b_materials, s.iadd(m_base, s.uint(10)))

    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var v = emit_vec3_normalize(s, s.fsub(cam_x, wx), s.fsub(cam_y, wy), s.fsub(cam_z, wz))
    var n0 = emit_vec3_normalize(s, s.extract_f(n_v, 0), s.extract_f(n_v, 1), s.extract_f(n_v, 2))
    var flip = s.select_f(s.flt(emit_vec3_dot(s, n0[0], n0[1], n0[2], v[0], v[1], v[2]), s.float(0.0)), s.float(-1.0), s.float(1.0))
    var nx = s.fmul(n0[0], flip)
    var ny = s.fmul(n0[1], flip)
    var nz = s.fmul(n0[2], flip)
    var ndv = s.nclamp(emit_vec3_dot(s, nx, ny, nz, v[0], v[1], v[2]), s.float(1e-3), s.float(1.0))

    var f0r = s.fma(s.fsub(ar, s.float(0.04)), metal, s.float(0.04))
    var f0g = s.fma(s.fsub(ag, s.float(0.04)), metal, s.float(0.04))
    var f0b = s.fma(s.fsub(ab, s.float(0.04)), metal, s.float(0.04))
    var one_m_met = s.fsub(s.float(1.0), metal)
    var lut = s.sample_2d_lod(brdf_lut, s.vec2(s.nclamp(ndv, s.float(0.01), s.float(0.99)), s.nclamp(rough, s.float(0.01), s.float(0.99))), s.float(0.0))
    var lut_a = s.extract_f(lut, 0)
    var lut_b = s.extract_f(lut, 1)
    var spr = s.fma(f0r, lut_a, lut_b)
    var spg = s.fma(f0g, lut_a, lut_b)
    var spb = s.fma(f0b, lut_a, lut_b)
    var kd_r = s.fmul(s.fmul(ar, one_m_met), s.fsub(s.float(1.0), spr))
    var kd_g = s.fmul(s.fmul(ag, one_m_met), s.fsub(s.float(1.0), spg))
    var kd_b = s.fmul(s.fmul(ab, one_m_met), s.fsub(s.float(1.0), spb))

    # Sun: GGX + Lambert, shadowed by the VSM when it is on
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var r_flags = s.load_word(b_globals, s.uint(99))
    var sun_vis = s.fvar(s.float(1.0))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_VSM)), s.uint(0)))
    s.store(sun_vis, emit_vsm_pcss_shadow(s, b_globals, b_vsm, vsm_samp, wx, wy, wz, nx, ny, nz, s.load_float(b_globals, s.uint(51))))
    s.end_if()
    var ndl = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, sun_x, sun_y, sun_z))
    var hv = emit_vec3_normalize(s, s.fadd(v[0], sun_x), s.fadd(v[1], sun_y), s.fadd(v[2], sun_z))
    var ndh = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, hv[0], hv[1], hv[2]))
    var vdh = s.nmax(s.float(0.0), emit_vec3_dot(s, v[0], v[1], v[2], hv[0], hv[1], hv[2]))
    var a_r = s.fmul(rough, rough)
    var a2 = s.fmul(a_r, a_r)
    var d_den = s.fma(s.fmul(ndh, ndh), s.fsub(a2, s.float(1.0)), s.float(1.0))
    var d_ggx = s.fdiv(a2, s.nmax(s.float(1e-6), s.fmul(s.float(3.14159265), s.fmul(d_den, d_den))))
    var gv = s.fmul(ndl, s.fsqrt(s.fma(s.fmul(ndv, ndv), s.fsub(s.float(1.0), a2), a2)))
    var gl = s.fmul(ndv, s.fsqrt(s.fma(s.fmul(ndl, ndl), s.fsub(s.float(1.0), a2), a2)))
    var vis_g = s.fdiv(s.float(0.5), s.nmax(s.float(1e-5), s.fadd(gv, gl)))
    var om = s.fsub(s.float(1.0), vdh)
    var om2 = s.fmul(om, om)
    var fc = s.fmul(s.fmul(om2, om2), om)
    var dv_s = s.fmul(d_ggx, vis_g)
    var e_sun = s.fmul(ndl, s.load_f(sun_vis))
    var inv_pi = s.float(0.318309886)
    var sun_r = s.fmul(s.load_float(b_globals, s.uint(52)), e_sun)
    var sun_g = s.fmul(s.load_float(b_globals, s.uint(53)), e_sun)
    var sun_b = s.fmul(s.load_float(b_globals, s.uint(54)), e_sun)
    var spec_sun_r = s.fmul(sun_r, s.fmul(s.fma(s.fsub(s.float(1.0), f0r), fc, f0r), dv_s))
    var spec_sun_g = s.fmul(sun_g, s.fmul(s.fma(s.fsub(s.float(1.0), f0g), fc, f0g), dv_s))
    var spec_sun_b = s.fmul(sun_b, s.fmul(s.fma(s.fsub(s.float(1.0), f0b), fc, f0b), dv_s))

    # Ambient SH9 + split-sum environment probe
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var sh = emit_eval_sh9_irradiance(s, b_globals, 72, nx, ny, nz)
    var two_nv = s.fmul(s.float(2.0), ndv)
    var rx = s.fsub(s.fmul(nx, two_nv), v[0])
    var ry = s.fsub(s.fmul(ny, two_nv), v[1])
    var rz = s.fsub(s.fmul(nz, two_nv), v[2])
    var env = s.sample_cube_lod(env_cube, s.vec3(rx, ry, rz), s.fmul(rough, s.load_float(b_globals, s.uint(103))))
    var diff_r = s.fmul(kd_r, s.fma(sun_r, inv_pi, s.fmul(sh[0], ibl_int)))
    var diff_g = s.fmul(kd_g, s.fma(sun_g, inv_pi, s.fmul(sh[1], ibl_int)))
    var diff_b = s.fmul(kd_b, s.fma(sun_b, inv_pi, s.fmul(sh[2], ibl_int)))
    var inv_a = s.fdiv(s.float(1.0), s.nmax(alpha, s.float(0.2)))
    var spec_r = s.fma(spr, s.fmul(s.extract_f(env, 0), ibl_int), spec_sun_r)
    var spec_g = s.fma(spg, s.fmul(s.extract_f(env, 1), ibl_int), spec_sun_g)
    var spec_b = s.fma(spb, s.fmul(s.extract_f(env, 2), ibl_int), spec_sun_b)
    var out_r = s.fma(spec_r, inv_a, s.fadd(diff_r, em_r))
    var out_g = s.fma(spec_g, inv_a, s.fadd(diff_g, em_g))
    var out_b = s.fma(spec_b, inv_a, s.fadd(diff_b, em_b))
    emit(s.code, OP_STORE, out_col[1], s.vec4(out_r, out_g, out_b, alpha))
    return s.finish()


def blend_composite_shader() -> List[UInt32]:
    """`hdr = hdr * (1 - A) + C` from the blended-mesh target copied into `BIND_AAA_HISTORY` as f16x4
    (`group = 8 x 8`). Push[0] = render_w, [1] = render_h, [2] = accumulation offset."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var acc_off = s.push_word(2)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var ao = s.iadd(acc_off, s.imul(pix, s.uint(2)))
    var rg = s.unpack_f16x2(s.load_word(b_hist, ao))
    var ba = s.unpack_f16x2(s.load_word(b_hist, s.iadd(ao, s.uint(1))))
    var a = s.nclamp(ba[1], s.float(0.0), s.float(1.0))
    s.return_if(s.fle(a, s.float(0.0)))
    var keep = s.fsub(s.float(1.0), a)
    var base = s.imul(pix, s.uint(8))
    s.store_float(b_hdr, base, s.fma(s.load_float(b_hdr, base), keep, s.nmax(rg[0], s.float(0.0))))
    s.store_float(b_hdr, s.iadd(base, s.uint(1)), s.fma(s.load_float(b_hdr, s.iadd(base, s.uint(1))), keep, s.nmax(rg[1], s.float(0.0))))
    s.store_float(b_hdr, s.iadd(base, s.uint(2)), s.fma(s.load_float(b_hdr, s.iadd(base, s.uint(2))), keep, s.nmax(ba[0], s.float(0.0))))
    return s.finish()


@fieldwise_init
struct AaaBlendDraw(Copyable, Movable):
    var instance: Int
    var first_index: Int
    var triangle_count: Int
    var material: Int
    var opacity: Float32
    var dist2: Float32


@fieldwise_init
struct AaaShadowCaster(Copyable, Movable):
    var instance: Int
    var first_index: Int
    var triangle_count: Int


# ============================================================================
# Pass state & recording
# ============================================================================

struct AaaRasterFallbacks(Movable):
    """Toggles, pipelines and per-frame lists of the raster fallbacks (owned by `FlintAaaVulkanRenderer`)."""

    # GTAO (`gtao_mode`: AAA_FALLBACK_AUTO / ON / OFF)
    var gtao_mode: Int
    var gtao_radius_m: Float32
    var gtao_strength: Float32
    var gtao_max_radius_px: Float32
    # Screen-space reflections
    var ssr_mode: Int
    var ssr_intensity: Float32
    var ssr_max_roughness: Float32
    var ssr_thickness_m: Float32
    var ssr_max_distance_m: Float32
    # Local-light shadow maps
    var local_shadow_mode: Int
    var local_shadow_max_lights: Int
    var local_shadow_near_m: Float32
    # Alpha-blended meshes
    var enable_alpha_blend: Bool
    var material_blend_opacity: List[Float32]  # per material id: < 0 = opaque, else the blend opacity

    var pipe_gtao_trace: Int
    var pipe_gtao_apply: Int
    var pipe_ssr_pyramid: Int
    var pipe_ssr_trace: Int
    var pipe_ssr_composite: Int
    var pipe_shadow_gfx: Int
    var pipe_blend_gfx: Int
    var pipe_blend_composite: Int
    var blend_img: VulkanImage
    var shadow_atlas_img: VulkanImage

    var frame_blend_draws: List[AaaBlendDraw]
    var frame_shadow_casters: List[AaaShadowCaster]
    var shadow_regions: List[AaaLocalShadowRegion]
    var shadow_tiles_used: Int

    # Read-only stats of the last frame
    var last_gtao_ran: Bool
    var last_ssr_ran: Bool
    var last_shadow_tiles: Int
    var last_blend_draws: Int

    def __init__(out self):
        self.gtao_mode = AAA_FALLBACK_AUTO
        self.gtao_radius_m = Float32(0.9)
        self.gtao_strength = Float32(1.0)
        self.gtao_max_radius_px = Float32(48.0)
        self.ssr_mode = AAA_FALLBACK_AUTO
        self.ssr_intensity = Float32(1.0)
        self.ssr_max_roughness = Float32(0.6)
        self.ssr_thickness_m = Float32(0.35)
        self.ssr_max_distance_m = Float32(40.0)
        self.local_shadow_mode = AAA_FALLBACK_AUTO
        self.local_shadow_max_lights = 4
        self.local_shadow_near_m = Float32(0.05)
        self.enable_alpha_blend = True
        self.material_blend_opacity = List[Float32]()
        self.pipe_gtao_trace = 0
        self.pipe_gtao_apply = 0
        self.pipe_ssr_pyramid = 0
        self.pipe_ssr_trace = 0
        self.pipe_ssr_composite = 0
        self.pipe_shadow_gfx = 0
        self.pipe_blend_gfx = 0
        self.pipe_blend_composite = 0
        self.blend_img = VulkanImage()
        self.shadow_atlas_img = VulkanImage()
        self.frame_blend_draws = List[AaaBlendDraw]()
        self.frame_shadow_casters = List[AaaShadowCaster]()
        self.shadow_regions = List[AaaLocalShadowRegion]()
        self.shadow_tiles_used = 0
        self.last_gtao_ran = False
        self.last_ssr_ran = False
        self.last_shadow_tiles = 0
        self.last_blend_draws = 0

    # ------------------------------------------------------------------
    # Material alpha-blend flags & per-frame lists
    # ------------------------------------------------------------------

    def set_material_alpha_blend(mut self, mat_id: Int, opacity: Float32):
        """Marks `mat_id` alpha-blended with `opacity` (`< 0` makes it opaque again)."""
        if mat_id < 0:
            return
        while len(self.material_blend_opacity) <= mat_id:
            self.material_blend_opacity.append(Float32(-1.0))
        self.material_blend_opacity[mat_id] = max(Float32(-1.0), min(Float32(1.0), opacity))

    def is_alpha_blend_material(self, mat_id: Int) -> Bool:
        return (
            self.enable_alpha_blend
            and mat_id >= 0
            and mat_id < len(self.material_blend_opacity)
            and self.material_blend_opacity[mat_id] >= Float32(0.0)
        )

    def clear_frame(mut self):
        self.frame_blend_draws.clear()
        self.frame_shadow_casters.clear()

    def add_blend_draw(mut self, instance: Int, first_index: Int, triangle_count: Int, mat_id: Int, world: Mat4, cam_pos: Vec3):
        if triangle_count <= 0:
            return
        var d = Vec3(world.values[12], world.values[13], world.values[14]) - cam_pos
        self.frame_blend_draws.append(
            AaaBlendDraw(instance, first_index, triangle_count, mat_id, self.material_blend_opacity[mat_id], d.dot(d))
        )

    def add_shadow_caster(mut self, instance: Int, first_index: Int, triangle_count: Int):
        if triangle_count > 0:
            self.frame_shadow_casters.append(AaaShadowCaster(instance, first_index, triangle_count))

    # ------------------------------------------------------------------
    # Local-light shadow maps
    # ------------------------------------------------------------------

    def plan_local_shadows(mut self, mut lights: List[AaaClusterLight], cam_pos: Vec3, has_ray_query: Bool, allowed: Bool):
        """Picks the most important point/spot lights and writes their region ids into `lights` (packed into the
        cluster buffer, read by the deferred pass). Every other light gets `vsm_region_id = -1`."""
        self.shadow_regions.clear()
        self.shadow_tiles_used = 0
        for i in range(len(lights)):
            lights[i].vsm_region_id = -1
        if not allowed or not fallback_active(self.local_shadow_mode, has_ray_query) or len(lights) == 0:
            return
        var picks = select_local_shadow_lights(
            lights, cam_pos, min(self.local_shadow_max_lights, AAA_LOCAL_SHADOW_MAX_REGIONS), AAA_LOCAL_SHADOW_MAX_TILES
        )
        for k in range(len(picks)):
            var li = picks[k]
            ref lt = lights[li]
            var is_point = lt.kind == AAA_LIGHT_POINT
            var a = Vec3(1.0, 0.0, 0.0)
            var t = Vec3(0.0, 0.0, -1.0)
            var inv_tan = Float32(1.0)
            if not is_point:
                var basis = local_shadow_spot_basis(lt.axis_u)
                a = basis[0]
                t = basis[1]
                var c_out = max(Float32(0.17), min(Float32(0.9999), lt.param0))
                var half = min(Float32(1.4), det_acos(c_out) + Float32(0.05))
                inv_tan = Float32(1.0) / tan(half)
            var region = AaaLocalShadowRegion(
                li, is_point, self.shadow_tiles_used, lt.position, a, t, inv_tan, self.local_shadow_near_m, max(lt.radius, self.local_shadow_near_m * Float32(4.0))
            )
            lt.vsm_region_id = len(self.shadow_regions)
            self.shadow_tiles_used += 6 if is_point else 1
            self.shadow_regions.append(region^)

    def record_local_shadows(mut self, mut vk: VulkanGpu, slot: Int, pipe_layout: Int, desc_set: Int, cluster_buf: GpuBuffer):
        """Renders the planned shadow tiles, copies them and the region table into `cluster_buf`."""
        self.last_shadow_tiles = 0
        if len(self.shadow_regions) == 0 or self.shadow_tiles_used <= 0:
            return
        var tile = AAA_LOCAL_SHADOW_TILE
        if self.pipe_shadow_gfx == 0:
            self.pipe_shadow_gfx = vk.create_graphics_pipeline(
                pipe_layout,
                local_shadow_depth_vertex_shader(),
                List[UInt32](),
                has_color=False,
                color_format=0,
                depth_format=FORMAT_D32_SFLOAT,
                depth_test=True,
                depth_write=True,
                depth_compare=COMPARE_LESS_OR_EQUAL,
                cull_mode=CULL_MODE_NONE,
                front_face=FRONT_FACE_CW,
            )
        if self.shadow_atlas_img.image == 0:
            self.shadow_atlas_img = vk.create_image_2d(
                tile, tile * AAA_LOCAL_SHADOW_MAX_TILES, FORMAT_D32_SFLOAT, IMAGE_USAGE_DEPTH_ATTACHMENT | IMAGE_USAGE_TRANSFER_SRC
            )
        if self.pipe_shadow_gfx == 0 or self.shadow_atlas_img.image == 0:
            return
        var n_tiles = min(self.shadow_tiles_used, AAA_LOCAL_SHADOW_MAX_TILES)
        vk.transition_image(
            slot, self.shadow_atlas_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_TRANSFER, ACCESS_TRANSFER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
        )
        vk.begin_rendering(slot, 0, 0, tile, tile * n_tiles, False, 0, False, 0, 0, True, self.shadow_atlas_img.view, True, 1.0)
        for r in range(len(self.shadow_regions)):
            ref reg = self.shadow_regions[r]
            var faces = 6 if reg.is_point else 1
            for f in range(faces):
                var t_idx = reg.first_tile + f
                if t_idx >= n_tiles:
                    continue
                var a = reg.axis_a
                var t = reg.axis_t
                if reg.is_point:
                    var fb = local_shadow_point_face_basis(f)
                    a = fb[0]
                    t = fb[1]
                vk.set_viewport_and_scissor(slot, 0, t_idx * tile, tile, tile, flip_y=False)
                var push = List[UInt32](length=16, fill=0)
                push[2] = bitcast[DType.uint32](reg.position.x)
                push[3] = bitcast[DType.uint32](reg.position.y)
                push[4] = bitcast[DType.uint32](reg.position.z)
                push[5] = bitcast[DType.uint32](a.x)
                push[6] = bitcast[DType.uint32](a.y)
                push[7] = bitcast[DType.uint32](a.z)
                push[8] = bitcast[DType.uint32](t.x)
                push[9] = bitcast[DType.uint32](t.y)
                push[10] = bitcast[DType.uint32](t.z)
                push[11] = bitcast[DType.uint32](reg.inv_tan)
                push[12] = bitcast[DType.uint32](reg.near)
                push[13] = bitcast[DType.uint32](reg.far)
                for ci in range(len(self.frame_shadow_casters)):
                    ref cst = self.frame_shadow_casters[ci]
                    push[0] = UInt32(cst.instance)
                    push[1] = UInt32(cst.first_index)
                    vk.bind_graphics_pipeline(slot, self.pipe_shadow_gfx, pipe_layout, desc_set, push)
                    vk.draw(slot, cst.triangle_count * 3)
        vk.end_rendering(slot)
        vk.transition_image(
            slot, self.shadow_atlas_img, LAYOUT_TRANSFER_SRC, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_TRANSFER, ACCESS_TRANSFER_READ
        )
        vk.copy_image_to_buffer(slot, self.shadow_atlas_img, cluster_buf, tile, tile * n_tiles, dst_offset=AAA_LOCAL_SHADOW_ATLAS_BASE * 4)
        for r in range(len(self.shadow_regions)):
            var words = self.shadow_regions[r].table_words()
            var t_base = (AAA_LOCAL_SHADOW_TABLE_BASE + r * AAA_LOCAL_SHADOW_REGION_WORDS) * 4
            for wi in range(len(words)):
                vk.fill_buffer(slot, cluster_buf, t_base + wi * 4, 4, words[wi])
        vk.memory_barrier2(slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ))
        self.last_shadow_tiles = n_tiles

    # ------------------------------------------------------------------
    # GTAO + SSR (after opaque lighting and its denoisers)
    # ------------------------------------------------------------------

    def record_gtao_ssr(
        mut self,
        mut vk: VulkanGpu,
        slot: Int,
        pipe_layout: Int,
        desc_set: Int,
        hist: GpuBuffer,
        max_pix: Int,
        w: Int,
        h: Int,
        has_ray_query: Bool,
    ):
        self.last_gtao_ran = False
        self.last_ssr_ran = False
        var gx = (w + 7) // 8
        var gy = (h + 7) // 8
        var run_gtao = fallback_active(self.gtao_mode, has_ray_query) and self.gtao_strength > Float32(0.0)
        var run_ssr = fallback_active(self.ssr_mode, has_ray_query) and self.ssr_intensity > Float32(0.0)
        if run_gtao:
            if self.pipe_gtao_trace == 0:
                self.pipe_gtao_trace = vk.create_aaa_compute_pipeline(pipe_layout, gtao_trace_shader())
            if self.pipe_gtao_apply == 0:
                self.pipe_gtao_apply = vk.create_aaa_compute_pipeline(pipe_layout, gtao_denoise_apply_shader())
            if self.pipe_gtao_trace != 0 and self.pipe_gtao_apply != 0:
                var push = List[UInt32](length=16, fill=0)
                push[0] = UInt32(w)
                push[1] = UInt32(h)
                push[2] = UInt32(gtao_raw_offset(max_pix))
                push[3] = bitcast[DType.uint32](self.gtao_radius_m)
                push[4] = bitcast[DType.uint32](self.gtao_max_radius_px)
                vk.dispatch_aaa(slot, self.pipe_gtao_trace, pipe_layout, desc_set, push, gx, gy)
                vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
                push[3] = bitcast[DType.uint32](self.gtao_strength)
                vk.dispatch_aaa(slot, self.pipe_gtao_apply, pipe_layout, desc_set, push, gx, gy)
                vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
                self.last_gtao_ran = True
        if run_ssr:
            if self.pipe_ssr_pyramid == 0:
                self.pipe_ssr_pyramid = vk.create_aaa_compute_pipeline(pipe_layout, ssr_pyramid_shader())
            if self.pipe_ssr_trace == 0:
                self.pipe_ssr_trace = vk.create_aaa_compute_pipeline(pipe_layout, ssr_trace_shader())
            if self.pipe_ssr_composite == 0:
                self.pipe_ssr_composite = vk.create_aaa_compute_pipeline(pipe_layout, ssr_composite_shader())
            if self.pipe_ssr_pyramid != 0 and self.pipe_ssr_trace != 0 and self.pipe_ssr_composite != 0:
                var pyr_base = ssr_pyramid_offset(max_pix)
                var levels = ssr_pyramid_levels(w, h, pyr_base)
                vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE))
                for l in range(len(levels)):
                    vk.fill_buffer(slot, hist, (pyr_base + l * 2) * 4, 4, UInt32(levels[l][0]))
                    vk.fill_buffer(slot, hist, (pyr_base + l * 2 + 1) * 4, 4, UInt32(levels[l][1]) | (UInt32(levels[l][2]) << 16))
                vk.memory_barrier2(slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
                for l in range(len(levels)):
                    var push_p = List[UInt32](length=16, fill=0)
                    push_p[0] = UInt32(levels[l][1])
                    push_p[1] = UInt32(levels[l][2])
                    push_p[2] = UInt32(w if l == 0 else levels[l - 1][1])
                    push_p[3] = UInt32(h if l == 0 else levels[l - 1][2])
                    push_p[4] = UInt32(0 if l == 0 else levels[l - 1][0])
                    push_p[5] = UInt32(levels[l][0])
                    push_p[6] = UInt32(1) if l == 0 else UInt32(0)
                    vk.dispatch_aaa(slot, self.pipe_ssr_pyramid, pipe_layout, desc_set, push_p, (levels[l][1] + 7) // 8, (levels[l][2] + 7) // 8)
                    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
                var push_t = List[UInt32](length=16, fill=0)
                push_t[0] = UInt32(w)
                push_t[1] = UInt32(h)
                push_t[2] = UInt32(pyr_base)
                push_t[3] = UInt32(ssr_result_offset(max_pix))
                push_t[4] = UInt32(len(levels))
                push_t[5] = bitcast[DType.uint32](self.ssr_max_roughness)
                push_t[6] = bitcast[DType.uint32](self.ssr_thickness_m)
                push_t[7] = bitcast[DType.uint32](self.ssr_max_distance_m)
                push_t[8] = bitcast[DType.uint32](self.ssr_intensity)
                vk.dispatch_aaa(slot, self.pipe_ssr_trace, pipe_layout, desc_set, push_t, gx, gy)
                vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
                var push_c = List[UInt32](length=16, fill=0)
                push_c[0] = UInt32(w)
                push_c[1] = UInt32(h)
                push_c[2] = UInt32(ssr_result_offset(max_pix))
                vk.dispatch_aaa(slot, self.pipe_ssr_composite, pipe_layout, desc_set, push_c, gx, gy)
                vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
                self.last_ssr_ran = True

    # ------------------------------------------------------------------
    # Alpha-blended meshes
    # ------------------------------------------------------------------

    def record_blend(
        mut self,
        mut vk: VulkanGpu,
        slot: Int,
        pipe_layout: Int,
        desc_set: Int,
        mut depth_img: VulkanImage,
        hist: GpuBuffer,
        max_pix: Int,
        w: Int,
        h: Int,
    ):
        self.last_blend_draws = 0
        var n = len(self.frame_blend_draws)
        if n == 0 or depth_img.image == 0:
            return
        if self.pipe_blend_gfx == 0:
            self.pipe_blend_gfx = vk.create_graphics_pipeline(
                pipe_layout,
                blend_mesh_vertex_shader(),
                blend_mesh_fragment_shader(),
                has_color=True,
                color_format=FORMAT_R16G16B16A16_SFLOAT,
                depth_format=FORMAT_D32_SFLOAT,
                depth_test=True,
                depth_write=False,
                depth_compare=COMPARE_GREATER_OR_EQUAL,
                cull_mode=CULL_MODE_NONE,
                front_face=FRONT_FACE_CW,
                alpha_blend=True,
            )
        if self.pipe_blend_composite == 0:
            self.pipe_blend_composite = vk.create_aaa_compute_pipeline(pipe_layout, blend_composite_shader())
        if self.blend_img.image != 0 and (self.blend_img.width != w or self.blend_img.height != h):
            vk.destroy_image(self.blend_img)
        if self.blend_img.image == 0:
            self.blend_img = vk.create_image_2d(w, h, FORMAT_R16G16B16A16_SFLOAT, IMAGE_USAGE_COLOR_ATTACHMENT | IMAGE_USAGE_TRANSFER_SRC)
        if self.pipe_blend_gfx == 0 or self.pipe_blend_composite == 0 or self.blend_img.image == 0:
            return

        # Back-to-front per object (stable insertion sort on the camera distance)
        var order = List[Int](capacity=n)
        for i in range(n):
            order.append(i)
        for i in range(1, n):
            var cur = order[i]
            var j = i - 1
            while j >= 0 and self.frame_blend_draws[order[j]].dist2 < self.frame_blend_draws[cur].dist2:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = cur

        vk.transition_image(slot, self.blend_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE)
        vk.transition_image(
            slot, depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_READ
        )
        vk.begin_rendering(slot, 0, 0, w, h, True, self.blend_img.view, True, 0, 0, True, depth_img.view, False, 0.0)
        vk.set_viewport_and_scissor(slot, 0, 0, w, h, flip_y=True)
        for k in range(n):
            ref d = self.frame_blend_draws[order[k]]
            var push = List[UInt32](length=16, fill=0)
            push[0] = UInt32(d.instance)
            push[1] = UInt32(d.first_index)
            push[2] = UInt32(d.material)
            push[3] = bitcast[DType.uint32](d.opacity)
            vk.bind_graphics_pipeline(slot, self.pipe_blend_gfx, pipe_layout, desc_set, push)
            vk.draw(slot, d.triangle_count * 3)
        vk.end_rendering(slot)
        vk.transition_image(
            slot, depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_READ, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        vk.transition_image(
            slot, self.blend_img, LAYOUT_TRANSFER_SRC, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_TRANSFER, ACCESS_TRANSFER_READ
        )
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE))
        vk.copy_image_to_buffer(slot, self.blend_img, hist, w, h, dst_offset=blend_accum_offset(max_pix) * 4)
        vk.memory_barrier2(slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        var push_c = List[UInt32](length=16, fill=0)
        push_c[0] = UInt32(w)
        push_c[1] = UInt32(h)
        push_c[2] = UInt32(blend_accum_offset(max_pix))
        vk.dispatch_aaa(slot, self.pipe_blend_composite, pipe_layout, desc_set, push_c, (w + 7) // 8, (h + 7) // 8)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))
        self.last_blend_draws = n

    def shutdown(mut self, mut vk: VulkanGpu):
        vk.destroy_pipeline(self.pipe_gtao_trace)
        vk.destroy_pipeline(self.pipe_gtao_apply)
        vk.destroy_pipeline(self.pipe_ssr_pyramid)
        vk.destroy_pipeline(self.pipe_ssr_trace)
        vk.destroy_pipeline(self.pipe_ssr_composite)
        vk.destroy_pipeline(self.pipe_shadow_gfx)
        vk.destroy_pipeline(self.pipe_blend_gfx)
        vk.destroy_pipeline(self.pipe_blend_composite)
        self.pipe_gtao_trace = 0
        self.pipe_gtao_apply = 0
        self.pipe_ssr_pyramid = 0
        self.pipe_ssr_trace = 0
        self.pipe_ssr_composite = 0
        self.pipe_shadow_gfx = 0
        self.pipe_blend_gfx = 0
        self.pipe_blend_composite = 0
        if self.blend_img.image != 0:
            vk.destroy_image(self.blend_img)
        if self.shadow_atlas_img.image != 0:
            vk.destroy_image(self.shadow_atlas_img)

