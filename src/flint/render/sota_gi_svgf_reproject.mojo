"""Area B round 4: jitter-aware depth test for SVGF-style temporal reprojection (`ShaderDsl` helper).

The v5 SVGF reprojection (`aaa_gi_svgf.svgf_temporal_reproject_shader`) and area C's learned-denoiser
temporal pass (`sota_ml_denoise._temporal_eval`) accept the previous frame's pixel when
`|z - z_prev| < max(0.012 z, 0.0096)`. Under the TSR Halton jitter the sample point moves by up to one
pixel per axis between frames, and the history fetch is nearest-pixel (another half pixel), so on any
slanted surface the depth legitimately changes by about a pixel's depth slope: on the Cornell box half of
the pixels lost their history every few frames (mean length 13 of 24, flicker 1.45).

The fix widens the tolerance by the local depth slope, measured as the smaller one-sided neighbour
difference per axis (robust at silhouettes, where one side jumps; the same idea as the `fwidth(z)`
term of Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering", HPG), capped at a quarter of
the depth so it can never accept a different surface layer.
"""
from flint.render.shader_dsl import BufferRef, GpuBool, GpuF32, GpuU32, ShaderDsl, abs_f, max_f, min_f, min_u, select_u


def emit_svgf_jitter_depth_ok(
    mut dsl: ShaderDsl,
    b_hdr: BufferRef[8],
    px: GpuU32,
    py: GpuU32,
    w_u: GpuU32,
    h_u: GpuU32,
    cur_z: GpuF32,
    prev_z: GpuF32,
) -> GpuBool:
    """True when `prev_z` is the same surface as `cur_z` within the relative tolerance plus 1.5 pixels of
    depth slope (`b_hdr` holds the linear depth in word 3 of every 8-word pixel record)."""
    var xl = select_u(px.ge(1), px - 1, px)
    var xr = min_u(w_u - 1, px + 1)
    var yu = select_u(py.ge(1), py - 1, py)
    var yd = min_u(h_u - 1, py + 1)
    var z_l = b_hdr.load_f32(b_hdr.base_word(py * w_u + xl) + 3)
    var z_r = b_hdr.load_f32(b_hdr.base_word(py * w_u + xr) + 3)
    var z_u = b_hdr.load_f32(b_hdr.base_word(yu * w_u + px) + 3)
    var z_d = b_hdr.load_f32(b_hdr.base_word(yd * w_u + px) + 3)
    var gz_x = min_f(abs_f(z_r - cur_z), abs_f(z_l - cur_z))
    var gz_y = min_f(abs_f(z_d - cur_z), abs_f(z_u - cur_z))
    var z_slope = min_f(gz_x + gz_y, cur_z * 0.25)
    var tol = max_f(cur_z * 0.012, 0.0096) + z_slope * 1.5
    return abs_f(cur_z - prev_z).lt(tol)
