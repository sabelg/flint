"""Area D (sota:camera): compute variable-rate shading for the compute `VisBuffer` shading pass.

Flint shades in compute (`DeferredMultiLobeRtShade`), so `VK_KHR_fragment_shading_rate` cannot reduce its
cost. This is the compute equivalent of tier-2 VRS (per-tile rates driven by image content and motion),
with work compaction so the skipped pixels really free whole warps:

1. `sota_vrs_rate_shader` (end of frame N, after DoF / CA / bloom / motion blur / metering, before lens
   noise): for every 8x8 tile of the *render* grid it replays the coarse reconstruction on the finished
   (exposed, perceptually encoded) image: the error of bilinearly interpolating each pixel from the
   2x2-grid and 4x4-grid anchors. Detail that DoF or motion blur already removed costs nothing, so those
   tiles drop to coarse rates by themselves. The tolerance grows with on-screen speed (motion blur and the
   TSR history hide detail on moving surfaces). One word per tile: 0 = 1x1, 1 = 2x2, 2 = 4x4.
2. `sota_vrs_classify_shader` (frame N+1, after the VisBuffer, before shading): a pixel is skipped when it
   is not an anchor of its tile's grid, every anchor it would be interpolated from exists, and all of them
   show the same surface (same instance + meshlet and same material + primitive word of the VisBuffer as
   the pixel itself, so no interpolation ever crosses a silhouette or a material edge). The pixels to shade
   are compacted per 8x8 workgroup (workgroup atomic for the slot, one global atomic per tile for the base)
   into a list of packed `x | y << 16` coordinates; the skipped ones go into a 64-bit mask per tile.
3. The shading pass reads its pixel from that list (`emit_sota_vrs_remap`, a one-line hook in
   `aaa_deferred_shade_shader`): invocation `g` shades list entry `g`, and invocations past the count exit,
   so whole warps retire instead of idling beside the shaded anchors.
4. `sota_vrs_fill_shader` (one workgroup per coarse tile, anchors staged in shared memory): every skipped
   pixel is rebuilt from its anchors: bilinear colour, velocity and
   normal, perspective-correct depth (`1/z` is affine in screen space on a plane), the demodulated SVGF
   inputs, and the nearest anchor's specular-denoiser record.

The rate map of frame N steers frame N+1 (the image it is computed from is one frame old, as in
"Variable Rate Shading: Tier 2" image-based rate selection, e.g. Gears 5 / Doom Eternal practice). The
first frame after enabling or a resize shades at full rate.
"""
from std.collections import List
from std.math import abs, max, min, sqrt
from std.memory import bitcast
from flint.platform.vulkan import (
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    PIPELINE_BOTTOM,
    PIPELINE_COMPUTE,
    VulkanGpu,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BufferRef,
    GpuBool,
    GpuF32,
    GpuSharedArrayF32,
    GpuU32,
    ShaderDsl,
    abs_f,
    clamp_f,
    max_f,
    min_f,
    min_u,
    select_f,
    select_u,
    sqrt_f,
)
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_VIS_TEX,
)

comptime SOTA_VRS_TILE: Int = 8
comptime SOTA_VRS_RATE_1X1: Int = 0
comptime SOTA_VRS_RATE_2X2: Int = 1
comptime SOTA_VRS_RATE_4X4: Int = 2
comptime _POST_BASE: Int = 1024
comptime _ADAPTED_EXP_WORD: Int = 64


struct SotaVrsLayout(Copyable, ImplicitlyCopyable, Movable):
    """Post-scratch words: the persistent rate map at the top of the per-pixel post region (it must live from
    the end of frame N to the shading of frame N+1, so it sits above every post buffer) followed by the
    per-tile skip masks, and the per-frame shade list (count + one packed word per pixel) in the lens /
    quarter-res region, which is only used later in the frame."""
    var tiles_x: Int
    var tiles_y: Int
    var rate_off: Int
    var mask_off: Int
    var list_off: Int
    var list_end: Int

    def __init__(out self, render_w: Int, render_h: Int, max_pix: Int):
        self.tiles_x = (render_w + SOTA_VRS_TILE - 1) // SOTA_VRS_TILE
        self.tiles_y = (render_h + SOTA_VRS_TILE - 1) // SOTA_VRS_TILE
        self.rate_off = _POST_BASE + max_pix * 10 - max_pix // 16 - 1024
        self.mask_off = self.rate_off + self.tiles_x * self.tiles_y + 16  # 2 words / tile: skip bits
        self.list_off = _POST_BASE + max_pix * 8 + max_pix // 4 + 64
        self.list_end = self.list_off + 2 + max_pix


# ============================================================================
# CPU references
# ============================================================================

def vrs_perceptual_cpu(luma: Float32, exposure: Float32) -> Float32:
    """Display-referred encoding the rate test runs in: exposed luminance through `l / (1 + l)`."""
    var l = max(Float32(0.0), luma * exposure)
    return l / (Float32(1.0) + l)


def vrs_block_error_cpu(p: List[Float32], w: Int, h: Int, tx: Int, ty: Int, b: Int) -> Tuple[Float32, Float32]:
    """(mean, max) |p - bilinear(anchors)| over the 8x8 tile `(tx, ty)` when only the `b`-grid anchors are shaded."""
    var sum = Float32(0.0)
    var mx = Float32(0.0)
    for j in range(SOTA_VRS_TILE):
        for i in range(SOTA_VRS_TILE):
            var x = tx * SOTA_VRS_TILE + i
            var y = ty * SOTA_VRS_TILE + j
            var i0 = (i // b) * b
            var j0 = (j // b) * b
            var fx = Float32(i - i0) / Float32(b)
            var fy = Float32(j - j0) / Float32(b)
            var x0 = min(w - 1, tx * SOTA_VRS_TILE + i0)
            var y0 = min(h - 1, ty * SOTA_VRS_TILE + j0)
            var x1 = min(w - 1, tx * SOTA_VRS_TILE + i0 + b)
            var y1 = min(h - 1, ty * SOTA_VRS_TILE + j0 + b)
            var xc = min(w - 1, x)
            var yc = min(h - 1, y)
            var top = p[y0 * w + x0] * (Float32(1.0) - fx) + p[y0 * w + x1] * fx
            var bot = p[y1 * w + x0] * (Float32(1.0) - fx) + p[y1 * w + x1] * fx
            var e = abs(p[yc * w + xc] - (top * (Float32(1.0) - fy) + bot * fy))
            sum += e
            mx = max(mx, e)
    return (sum / Float32(SOTA_VRS_TILE * SOTA_VRS_TILE), mx)


def vrs_tile_rate_cpu(p: List[Float32], w: Int, h: Int, tx: Int, ty: Int, threshold: Float32, speed_px: Float32, motion_px: Float32, allow_4x4: Bool) -> Int:
    """Rate of tile `(tx, ty)`: the coarsest grid whose mean reconstruction error stays under the
    (motion-relaxed) threshold and whose worst pixel stays under 4x that."""
    var t = threshold * (Float32(1.0) + speed_px / max(motion_px, Float32(1.0e-3)))
    if allow_4x4:
        var e4 = vrs_block_error_cpu(p, w, h, tx, ty, 4)
        if e4[0] < t and e4[1] < t * Float32(4.0):
            return SOTA_VRS_RATE_4X4
    var e2 = vrs_block_error_cpu(p, w, h, tx, ty, 2)
    if e2[0] < t and e2[1] < t * Float32(4.0):
        return SOTA_VRS_RATE_2X2
    return SOTA_VRS_RATE_1X1


def vrs_shaded_fraction(rates: List[Int]) -> Float32:
    """Fraction of pixels the shading pass runs for a rate map (ignoring the surface-continuity vetoes)."""
    var acc = Float32(0.0)
    for r in rates:
        acc += Float32(1.0) / Float32((1 << r) * (1 << r))
    return acc / Float32(max(1, len(rates)))


# ============================================================================
# 1. Rate map (ShaderDsl, one thread per render tile)
# ============================================================================

def sota_vrs_rate_shader() -> List[UInt32]:
    """One 8x8 workgroup per render tile (dispatch exactly tiles_x x tiles_y groups). Push: 0 out_w, 1 out_h,
    2 render_w, 3 render_h, 4 tiles_x, 5 tiles_y, 6 rate_off, 7 threshold, 8 motion_px (speed in output px /
    frame at which the tolerance doubles), 9 allow 4x4 (0/1)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var sp = dsl.shared_array_f32[81]()
    var se2 = dsl.shared_array_f32[64]()
    var se4 = dsl.shared_array_f32[64]()
    var sm2 = dsl.shared_array_f32[64]()
    var sm4 = dsl.shared_array_f32[64]()
    var gx = dsl.global_id_x()
    var gy = dsl.global_id_y()
    var lx = gx & UInt32(7)
    var ly = gy & UInt32(7)
    var tx = gx >> UInt32(3)
    var ty = gy >> UInt32(3)
    var ow = dsl.push_u32(0)
    var oh = dsl.push_u32(1)
    var rw = dsl.push_u32(2)
    var rh = dsl.push_u32(3)
    var tiles_x = dsl.push_u32(4)
    var rate_off = dsl.push_u32(6)
    var thr = dsl.push_f32(7)
    var motion_px = max_f(dsl.push_f32(8), 1.0e-3)
    var allow4 = dsl.push_u32(9).ne(UInt32(0))

    var base_exp = b_glob.load_f32(dsl.u32(47))
    var auto_on = b_glob.load_u32(dsl.u32(119)).ne(UInt32(0))
    var adapt = b_post.load_f32(dsl.u32(UInt32(_ADAPTED_EXP_WORD)))
    var expo = select_f(auto_on & adapt.gt(1.0e-5), base_exp * adapt, base_exp)
    var sx = ow.to_f32() / rw.to_f32()
    var sy = oh.to_f32() / rh.to_f32()

    # 9x9 perceptual samples in shared memory: the tile plus the next tile's first row / column (anchors).
    for k in range(4):
        var ox_add = 1 if (k == 1 or k == 3) else 0
        var oy_add = 1 if (k == 2 or k == 3) else 0
        var cond = lx.lt(UInt32(8))
        if ox_add == 1:
            cond = cond & lx.eq(UInt32(7))
        if oy_add == 1:
            cond = cond & ly.eq(UInt32(7))
        dsl.if_then(cond)
        var i = lx + UInt32(ox_add)
        var j = ly + UInt32(oy_add)
        var rx = min_u(rw - UInt32(1), tx * UInt32(SOTA_VRS_TILE) + i)
        var ry = min_u(rh - UInt32(1), ty * UInt32(SOTA_VRS_TILE) + j)
        var ox = min_u(ow - UInt32(1), ((rx.to_f32() + Float32(0.5)) * sx).saturate_u32())
        var oy = min_u(oh - UInt32(1), ((ry.to_f32() + Float32(0.5)) * sy).saturate_u32())
        var c = b_hdr.load_vec3(b_hdr.base_word(oy * ow + ox))
        var l = max_f((c.x() * Float32(0.2126) + c.y() * Float32(0.7152) + c.z() * Float32(0.0722)) * expo, 0.0)
        sp.store(j * UInt32(9) + i, l / (l + Float32(1.0)))
        dsl.end_if()
    dsl.workgroup_barrier()

    # This pixel's error under 2x2 and 4x4 anchor grids (bilinear reconstruction, as the fill pass does).
    var me = sp.load(ly * UInt32(9) + lx)
    var local_i = ly * UInt32(8) + lx
    for bi in range(2):
        var b = 2 if bi == 0 else 4
        var i0 = lx & UInt32(8 - b)
        var j0 = ly & UInt32(8 - b)
        var fx = (lx - i0).to_f32() * Float32(1.0 / Float32(b))
        var fy = (ly - j0).to_f32() * Float32(1.0 / Float32(b))
        var one = dsl.f32(1.0)
        var p00 = sp.load(j0 * UInt32(9) + i0)
        var p10 = sp.load(j0 * UInt32(9) + i0 + UInt32(b))
        var p01 = sp.load((j0 + UInt32(b)) * UInt32(9) + i0)
        var p11 = sp.load((j0 + UInt32(b)) * UInt32(9) + i0 + UInt32(b))
        var pred = (p00 * (one - fx) + p10 * fx) * (one - fy) + (p01 * (one - fx) + p11 * fx) * fy
        var e = abs_f(me - pred)
        if bi == 0:
            se2.store(local_i, e)
            sm2.store(local_i, e)
        else:
            se4.store(local_i, e)
            sm4.store(local_i, e)
    dsl.workgroup_barrier()
    # Tree reduction (sum and max of both grids) over the 64 pixels.
    for step in [32, 16, 8, 4, 2, 1]:
        dsl.if_then(local_i.lt(UInt32(step)))
        var o = local_i + UInt32(step)
        se2.store(local_i, se2.load(local_i) + se2.load(o))
        se4.store(local_i, se4.load(local_i) + se4.load(o))
        sm2.store(local_i, max_f(sm2.load(local_i), sm2.load(o)))
        sm4.store(local_i, max_f(sm4.load(local_i), sm4.load(o)))
        dsl.end_if()
        dsl.workgroup_barrier()

    dsl.if_then(local_i.eq(UInt32(0)))
    var z0 = dsl.u32(0)
    var sum2 = se2.load(z0)
    var max2 = sm2.load(z0)
    var sum4 = se4.load(z0)
    var max4 = sm4.load(z0)
    # On-screen speed at the tile centre (velocity is a screen-uv delta per frame).
    var cx = min_u(ow - UInt32(1), ((tx * UInt32(SOTA_VRS_TILE) + UInt32(4)).to_f32() * sx).saturate_u32())
    var cy = min_u(oh - UInt32(1), ((ty * UInt32(SOTA_VRS_TILE) + UInt32(4)).to_f32() * sy).saturate_u32())
    var cb = b_hdr.base_word(cy * ow + cx)
    var vx = b_hdr.load_f32(cb + UInt32(4)) * ow.to_f32()
    var vy = b_hdr.load_f32(cb + UInt32(5)) * oh.to_f32()
    var speed = sqrt_f(vx * vx + vy * vy)
    var t = thr * (dsl.f32(1.0) + speed / motion_px)
    var inv = Float32(1.0 / Float32(SOTA_VRS_TILE * SOTA_VRS_TILE))
    var ok2 = (sum2 * inv).lt(t) & max2.lt(t * Float32(4.0))
    var ok4 = (sum4 * inv).lt(t) & max4.lt(t * Float32(4.0)) & allow4
    var rate = select_u(ok4, dsl.u32(UInt32(SOTA_VRS_RATE_4X4)), select_u(ok2, dsl.u32(UInt32(SOTA_VRS_RATE_2X2)), dsl.u32(UInt32(SOTA_VRS_RATE_1X1))))
    b_post.store_u32(rate_off + ty * tiles_x + tx, rate)
    dsl.end_if()
    return dsl.finish()


# ============================================================================
# 2. Classify + compact (AaaSpirvModule: needs the uint VisBuffer texture)
# ============================================================================

def sota_vrs_classify_shader() -> List[UInt32]:
    """One 8x8 workgroup per render tile. Push: 0 render_w, 1 render_h, 2 rate_off, 3 tiles_x, 4 list_off,
    5 mode (0: reset the list count; 1: classify + compact), 6 mask_off.
    List: word 0 = shade count, word 1 reserved, then packed `x | y << 16` entries of the pixels to shade.
    Skip mask: 2 words per tile (bit = local pixel index) of the pixels the fill pass rebuilds."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var wg = s.workgroup_array_u32(4)  # 0 shade count, 1 shade base, 2/3 skip bits of pixels 0-31 / 32-63
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var lx = s.local_axis(0)
    var ly = s.local_axis(1)
    var gx = s.workgroup_axis(0)
    var gy = s.workgroup_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var rate_off = s.push_word(2)
    var tiles_x = s.push_word(3)
    var list_off = s.push_word(4)
    var mode = s.push_word(5)
    var mask_off = s.push_word(6)
    var local_i = s.iadd(s.imul(ly, s.uint(8)), lx)
    var is_first = s.ieq(local_i, s.uint(0))

    s.begin_if(s.ieq(mode, s.uint(0)))
    s.begin_if(s.logical_and(is_first, s.logical_and(s.ieq(gx, s.uint(0)), s.ieq(gy, s.uint(0)))))
    s.store_word(b_post, list_off, s.uint(0))
    s.store_word(b_post, s.iadd(list_off, s.uint(1)), s.uint(0))
    s.end_if()
    s.begin_else()

    s.begin_if(is_first)
    for k in range(4):
        s.store_workgroup_u32(wg[0], wg[1], s.uint(UInt32(k)), s.uint(0))
    s.end_if()
    s.workgroup_barrier()

    var inb = s.logical_and(s.ult(px, w_u), s.ult(py, h_u))
    var rate = s.umin(s.load_word(b_post, s.iadd(rate_off, s.iadd(s.imul(gy, tiles_x), gx))), s.uint(2))
    var bsz = s.shl(s.uint(1), rate)
    var m = s.isub(bsz, s.uint(1))
    var fx = s.bit_and(px, m)
    var fy = s.bit_and(py, m)
    var non_anchor = s.logical_and(inb, s.ine(s.bit_or(fx, fy), s.uint(0)))
    var skip_v = s.uvar(s.uint(0))
    s.begin_if(non_anchor)
    var ax0 = s.isub(px, fx)
    var ay0 = s.isub(py, fy)
    var ax1 = s.select_u(s.ine(fx, s.uint(0)), s.iadd(ax0, bsz), ax0)
    var ay1 = s.select_u(s.ine(fy, s.uint(0)), s.iadd(ay0, bsz), ay0)
    var in_grid = s.logical_and(s.ult(ax1, w_u), s.ult(ay1, h_u))
    var cx1 = s.umin(ax1, s.isub(w_u, s.uint(1)))
    var cy1 = s.umin(ay1, s.isub(h_u, s.uint(1)))
    var v_self = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var s0 = s.extract_u(v_self, 0)
    var s1 = s.extract_u(v_self, 1)
    var same = s.ine(s0, s.uint(0))
    var corners = List[Tuple[UInt32, UInt32]]()
    corners.append((ax0, ay0))
    corners.append((cx1, ay0))
    corners.append((ax0, cy1))
    corners.append((cx1, cy1))
    for k in range(4):
        var v = s.texel_fetch_u2d(vis_samp, s.uvec2(corners[k][0], corners[k][1]), s.uint(0))
        same = s.logical_and(same, s.logical_and(s.ieq(s.extract_u(v, 0), s0), s.ieq(s.extract_u(v, 1), s1)))
    s.store(skip_v, s.select_u(s.logical_and(in_grid, same), s.uint(1), s.uint(0)))
    s.end_if()
    var skip = s.ine(s.load_u(skip_v), s.uint(0))
    var shade = s.logical_and(inb, s.logical_not(skip))
    var slot_v = s.uvar(s.uint(0))
    s.begin_if(shade)
    s.store(slot_v, s.atomic_add_workgroup_u32(wg[0], wg[1], s.uint(0), s.uint(1)))
    s.end_if()
    s.begin_if(skip)
    # Distinct bits, so the atomic add is an atomic or.
    _ = s.atomic_add_workgroup_u32(
        wg[0], wg[1], s.iadd(s.uint(2), s.shr(local_i, s.uint(5))), s.shl(s.uint(1), s.bit_and(local_i, s.uint(31)))
    )
    s.end_if()
    s.workgroup_barrier()
    s.begin_if(is_first)
    var n_sh = s.load_workgroup_u32(wg[0], wg[1], s.uint(0))
    s.store_workgroup_u32(wg[0], wg[1], s.uint(1), s.atomic_add_word(b_post, list_off, n_sh))
    var m_word = s.iadd(mask_off, s.imul(s.iadd(s.imul(gy, tiles_x), gx), s.uint(2)))
    s.store_word(b_post, m_word, s.load_workgroup_u32(wg[0], wg[1], s.uint(2)))
    s.store_word(b_post, s.iadd(m_word, s.uint(1)), s.load_workgroup_u32(wg[0], wg[1], s.uint(3)))
    s.end_if()
    s.workgroup_barrier()
    s.begin_if(shade)
    var i_sh = s.iadd(s.load_workgroup_u32(wg[0], wg[1], s.uint(1)), s.load_u(slot_v))
    s.store_word(b_post, s.iadd(s.iadd(list_off, s.uint(2)), i_sh), s.bit_or(px, s.shl(py, s.uint(16))))
    s.end_if()

    s.end_if()
    return s.finish()


def emit_sota_vrs_remap(mut s: AaaSpirvModule, b_post: UInt32, px: UInt32, py: UInt32, list_off: UInt32, groups_x: UInt32) -> Tuple[UInt32, UInt32]:
    """Shading-pass hook: with `list_off != 0`, invocation `g` (group-major, 64 per 8x8 group) shades list
    entry `g` and invocations past the list count return, so idle warps retire as a whole. With
    `list_off == 0` the pass is unchanged."""
    var on = s.ine(list_off, s.uint(0))
    var x_v = s.uvar(px)
    var y_v = s.uvar(py)
    s.begin_if(on)
    var gx = s.workgroup_axis(0)
    var gy = s.workgroup_axis(1)
    var lx = s.local_axis(0)
    var ly = s.local_axis(1)
    var g = s.iadd(s.imul(s.iadd(s.imul(gy, groups_x), gx), s.uint(64)), s.iadd(s.imul(ly, s.uint(8)), lx))
    s.return_if(s.uge(g, s.load_word(b_post, list_off)))
    var e = s.load_word(b_post, s.iadd(s.iadd(list_off, s.uint(2)), g))
    s.store(x_v, s.bit_and(e, s.uint(0xFFFF)))
    s.store(y_v, s.shr(e, s.uint(16)))
    s.end_if()
    return (s.load_u(x_v), s.load_u(y_v))


# ============================================================================
# 3. Fill (ShaderDsl, full render res)
# ============================================================================

def _fill_tile(
    mut dsl: ShaderDsl,
    b_hdr: BufferRef[8],
    b_post: BufferRef[1],
    b_hist: BufferRef[1],
    a_hdr: GpuSharedArrayF32[200],
    a_post: GpuSharedArrayF32[200],
    tx: GpuU32,
    ty: GpuU32,
    lx: GpuU32,
    ly: GpuU32,
    local_i: GpuU32,
    skipped: GpuBool,
    w_u: GpuU32,
    h_u: GpuU32,
    flags: GpuU32,
    spec_off: GpuU32,
    b: Int,
):
    """Coarse tile with anchor spacing `b`: stage the (8/b + 1)^2 anchors (8 HDR words + 8 demodulated-GI
    words each) in shared memory, then rebuild each skipped pixel from its 4 anchors."""
    var n = SOTA_VRS_TILE // b + 1
    var demod = (flags & UInt32(1)).ne(UInt32(0))
    dsl.if_then(local_i.lt(UInt32(n * n)))
    var ai = local_i % UInt32(n)
    var aj = local_i // UInt32(n)
    var ax = min_u(w_u - UInt32(1), tx * UInt32(SOTA_VRS_TILE) + ai * UInt32(b))
    var ay = min_u(h_u - UInt32(1), ty * UInt32(SOTA_VRS_TILE) + aj * UInt32(b))
    var ai_word = (ay * w_u + ax) * UInt32(8)
    for c in range(8):
        a_hdr.store(local_i * UInt32(8) + UInt32(c), b_hdr.load_f32(ai_word + UInt32(c)))
    dsl.if_then(demod)
    for c in range(8):
        a_post.store(local_i * UInt32(8) + UInt32(c), b_post.load_f32(ai_word + UInt32(_POST_BASE + c)))
    dsl.end_if()
    dsl.end_if()
    dsl.workgroup_barrier()

    dsl.if_then(skipped)
    var px = tx * UInt32(SOTA_VRS_TILE) + lx
    var py = ty * UInt32(SOTA_VRS_TILE) + ly
    var i0 = lx // UInt32(b)
    var j0 = ly // UInt32(b)
    var fx = (lx - i0 * UInt32(b)).to_f32() * Float32(1.0 / Float32(b))
    var fy = (ly - j0 * UInt32(b)).to_f32() * Float32(1.0 / Float32(b))
    var one = dsl.f32(1.0)
    var w00 = (one - fx) * (one - fy)
    var w10 = fx * (one - fy)
    var w01 = (one - fx) * fy
    var w11 = fx * fy
    var k00 = (j0 * UInt32(n) + i0) * UInt32(8)
    var k10 = k00 + UInt32(8)
    var k01 = k00 + UInt32(n * 8)
    var k11 = k01 + UInt32(8)
    var me = (py * w_u + px) * UInt32(8)
    for c in range(8):
        var c00 = a_hdr.load(k00 + UInt32(c))
        var c10 = a_hdr.load(k10 + UInt32(c))
        var c01 = a_hdr.load(k01 + UInt32(c))
        var c11 = a_hdr.load(k11 + UInt32(c))
        var v: GpuF32
        if c == 3:
            # Perspective-correct depth: 1 / z is affine in screen space across a plane.
            var iz = w00 / max_f(c00, 1.0e-6) + w10 / max_f(c10, 1.0e-6) + w01 / max_f(c01, 1.0e-6) + w11 / max_f(c11, 1.0e-6)
            v = one / max_f(iz, 1.0e-12)
        else:
            v = c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11
        b_hdr.store_f32(me + UInt32(c), v)
    # Demodulated SVGF inputs (crisp rgb, diffuse weight, velocity), 8 words / px from post word 1024.
    dsl.if_then(demod)
    for c in range(8):
        var v = a_post.load(k00 + UInt32(c)) * w00 + a_post.load(k10 + UInt32(c)) * w10 + a_post.load(k01 + UInt32(c)) * w01 + a_post.load(k11 + UInt32(c)) * w11
        b_post.store_f32(me + UInt32(_POST_BASE + c), v)
    dsl.end_if()
    # Specular-denoiser record (packed fp16): copy the nearest anchor's.
    dsl.if_then((flags & UInt32(2)).ne(UInt32(0)))
    var nx = tx * UInt32(SOTA_VRS_TILE) + (i0 + select_u(fx.ge(0.5), dsl.u32(1), dsl.u32(0))) * UInt32(b)
    var ny = ty * UInt32(SOTA_VRS_TILE) + (j0 + select_u(fy.ge(0.5), dsl.u32(1), dsl.u32(0))) * UInt32(b)
    var src = spec_off + (ny * w_u + nx) * UInt32(4)
    var dst = spec_off + (py * w_u + px) * UInt32(4)
    for c in range(4):
        b_hist.store_u32(dst + UInt32(c), b_hist.load_u32(src + UInt32(c)))
    dsl.end_if()
    dsl.end_if()


def sota_vrs_fill_shader() -> List[UInt32]:
    """One 8x8 workgroup per render tile; full-rate tiles exit at once. Push: 0 render_w, 1 render_h,
    2 rate_off, 3 tiles_x, 4 flags (1 demodulated GI, 2 specular denoiser record), 5 spec_off, 6 mask_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var a_hdr = dsl.shared_array_f32[200]()
    var a_post = dsl.shared_array_f32[200]()
    var gx = dsl.global_id_x()
    var gy = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var rate_off = dsl.push_u32(2)
    var tiles_x = dsl.push_u32(3)
    var flags = dsl.push_u32(4)
    var spec_off = dsl.push_u32(5)
    var mask_off = dsl.push_u32(6)
    var lx = gx & UInt32(7)
    var ly = gy & UInt32(7)
    var tx = gx >> UInt32(3)
    var ty = gy >> UInt32(3)
    var tile = ty * tiles_x + tx
    var rate = b_post.load_u32(rate_off + tile)
    dsl.return_if(rate.eq(UInt32(SOTA_VRS_RATE_1X1)))  # uniform over the workgroup
    var local_i = ly * UInt32(8) + lx
    var m = b_post.load_u32(mask_off + tile * UInt32(2) + (local_i >> UInt32(5)))
    var skipped = ((m >> (local_i & UInt32(31))) & UInt32(1)).ne(UInt32(0))
    dsl.if_then(rate.eq(UInt32(SOTA_VRS_RATE_4X4)))
    _fill_tile(dsl, b_hdr, b_post, b_hist, a_hdr, a_post, tx, ty, lx, ly, local_i, skipped, w_u, h_u, flags, spec_off, 4)
    dsl.else_branch()
    _fill_tile(dsl, b_hdr, b_post, b_hist, a_hdr, a_post, tx, ty, lx, ly, local_i, skipped, w_u, h_u, flags, spec_off, 2)
    dsl.end_if()
    return dsl.finish()


# ============================================================================
# Recording (called from the sota:camera hooks in aaa_renderer.mojo)
# ============================================================================

def _barrier(mut vk: VulkanGpu, slot: Int):
    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))


def _prof_begin(mut vk: VulkanGpu, slot: Int, name: String, use_prof: Bool, mut next_q: Int, q_limit: Int, mut prof_pairs: List[Tuple[String, Int]]) -> Int:
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((name, q))
        return q
    return -1


def _prof_end(mut vk: VulkanGpu, slot: Int, q: Int):
    if q >= 0:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_vrs_record_rate(
    mut vk: VulkanGpu,
    slot: Int,
    pipe: Int,
    pipe_layout: Int,
    desc_set: Int,
    out_w: Int,
    out_h: Int,
    render_w: Int,
    render_h: Int,
    max_pix: Int,
    threshold: Float32,
    motion_px: Float32,
    allow_4x4: Bool,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    var q = _prof_begin(vk, slot, String("SotaVrsRateMap"), use_prof, next_q, q_limit, prof_pairs)
    var lay = SotaVrsLayout(render_w, render_h, max_pix)
    var p = List[UInt32](length=16, fill=0)
    p[0] = UInt32(out_w); p[1] = UInt32(out_h); p[2] = UInt32(render_w); p[3] = UInt32(render_h)
    p[4] = UInt32(lay.tiles_x); p[5] = UInt32(lay.tiles_y); p[6] = UInt32(lay.rate_off)
    p[7] = bitcast[DType.uint32](threshold); p[8] = bitcast[DType.uint32](motion_px); p[9] = UInt32(1) if allow_4x4 else UInt32(0)
    _barrier(vk, slot)
    vk.dispatch_aaa(slot, pipe, pipe_layout, desc_set, p, lay.tiles_x, lay.tiles_y)
    _barrier(vk, slot)
    _prof_end(vk, slot, q)


def sota_vrs_record_classify(
    mut vk: VulkanGpu,
    slot: Int,
    pipe: Int,
    pipe_layout: Int,
    desc_set: Int,
    render_w: Int,
    render_h: Int,
    max_pix: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
) -> Int:
    """Records reset + classify; returns the list offset for the shading pass (push word 7)."""
    var q = _prof_begin(vk, slot, String("SotaVrsClassify"), use_prof, next_q, q_limit, prof_pairs)
    var lay = SotaVrsLayout(render_w, render_h, max_pix)
    var p = List[UInt32](length=16, fill=0)
    p[0] = UInt32(render_w); p[1] = UInt32(render_h); p[2] = UInt32(lay.rate_off); p[3] = UInt32(lay.tiles_x)
    p[4] = UInt32(lay.list_off); p[5] = UInt32(0); p[6] = UInt32(lay.mask_off)
    _barrier(vk, slot)
    vk.dispatch_aaa(slot, pipe, pipe_layout, desc_set, p, 1, 1)
    _barrier(vk, slot)
    p[5] = UInt32(1)
    vk.dispatch_aaa(slot, pipe, pipe_layout, desc_set, p, lay.tiles_x, lay.tiles_y)
    _barrier(vk, slot)
    _prof_end(vk, slot, q)
    return lay.list_off


def sota_vrs_record_fill(
    mut vk: VulkanGpu,
    slot: Int,
    pipe: Int,
    pipe_layout: Int,
    desc_set: Int,
    render_w: Int,
    render_h: Int,
    max_pix: Int,
    demod_gi: Bool,
    demod_spec: Bool,
    spec_off: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    var q = _prof_begin(vk, slot, String("SotaVrsFill"), use_prof, next_q, q_limit, prof_pairs)
    var lay = SotaVrsLayout(render_w, render_h, max_pix)
    var p = List[UInt32](length=16, fill=0)
    p[0] = UInt32(render_w); p[1] = UInt32(render_h); p[2] = UInt32(lay.rate_off); p[3] = UInt32(lay.tiles_x)
    p[4] = (UInt32(1) if demod_gi else UInt32(0)) | (UInt32(2) if demod_spec else UInt32(0)); p[5] = UInt32(spec_off)
    p[6] = UInt32(lay.mask_off)
    _barrier(vk, slot)
    vk.dispatch_aaa(slot, pipe, pipe_layout, desc_set, p, lay.tiles_x, lay.tiles_y)
    _barrier(vk, slot)
    _prof_end(vk, slot, q)
