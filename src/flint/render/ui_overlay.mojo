"""Feature 15: Tile-Binned GPU UI SDF Overlay Compute Pass.

Compiles a 16x16 tile-binned compute shader evaluating analytic rounded rects,
circles, and bindless R8_UNORM font glyphs with workgroup LDS coarse culling,
composited per tile in draw (painter's) order.
"""
from std.collections import List
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_GL_COMPUTE,
)
from flint.render.aaa_visbuffer import (
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_HISTORY,
    BIND_AAA_POST_SCRATCH,
)

comptime AAA_POST_UI_PRIMS_BASE_OFFSET: Int = 1024
comptime TILE_PRIM_CAP: Int = 256  # prims one 16x16 tile can composite (one per invocation)


def aaa_ui_overlay_shader() -> List[UInt32]:
    """Tile-Binned GPU UI SDF Overlay Compute Shader (16x16 workgroups).
    Push[0] = out_w, Push[1] = out_h, Push[2] = prim_count,
    Push[3] = prims_word_offset (in BIND_AAA_POST_SCRATCH),
    Push[4] = out_ldr_off (in BIND_AAA_HISTORY),
    Push[5] = display_mode (0=AgX, 1=HDR10_PQ, 2=Linear_sRGB).
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=16,
        group_height=16,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
    )
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)

    var gx = s.invocation_axis(0)
    var gy = s.invocation_axis(1)
    var out_w = s.push_word(0)
    var out_h = s.push_word(1)
    var prim_cnt = s.push_word(2)
    var prims_off = s.push_word(3)
    var out_ldr_off = s.push_word(4)
    _ = s.push_word(5) # disp_mode

    var wx = s.workgroup_axis(0)
    var wy = s.workgroup_axis(1)
    var lx = s.local_axis(0)
    var ly = s.local_axis(1)
    var local_idx = s.iadd(s.imul(ly, s.uint(16)), lx)

    # 16x16 Tile bounds in screen pixels
    var tile_x0 = s.float_from_uint(s.imul(wx, s.uint(16)))
    var tile_y0 = s.float_from_uint(s.imul(wy, s.uint(16)))
    var tile_x1 = s.fadd(tile_x0, s.float(16.0))
    var tile_y1 = s.fadd(tile_y0, s.float(16.0))

    # Workgroup Shared Memory: 1 counter + up to 256 primitive indices (one per invocation),
    # binned in arbitrary atomic order, then put back in draw order for painter's compositing.
    var shared_cnt = s.workgroup_array_u32(1)
    var shared_prims = s.workgroup_array_u32(TILE_PRIM_CAP)
    var ordered_prims = s.workgroup_array_u32(TILE_PRIM_CAP)

    s.begin_if(s.ieq(local_idx, s.uint(0)))
    s.store_workgroup_u32(shared_cnt[0], shared_cnt[1], s.uint(0), s.uint(0))
    s.end_if()
    s.workgroup_barrier()

    # Coarse Culling: Compact primitives overlapping tile AABB into shared LDS
    var p_var = s.uvar(local_idx)
    s.begin_loop()
    var p = s.load_u(p_var)
    s.break_if(s.uge(p, prim_cnt))

    var base_w = s.iadd(prims_off, s.imul(p, s.uint(16)))
    var px0 = s.load_float(b_post, base_w)
    var py0 = s.load_float(b_post, s.iadd(base_w, s.uint(1)))
    var px1 = s.load_float(b_post, s.iadd(base_w, s.uint(2)))
    var py1 = s.load_float(b_post, s.iadd(base_w, s.uint(3)))

    var outside = s.logical_or(
        s.logical_or(s.fge(tile_x0, px1), s.fge(px0, tile_x1)),
        s.logical_or(s.fge(tile_y0, py1), s.fge(py0, tile_y1)),
    )
    s.begin_if(s.logical_not(outside))
    var slot = s.atomic_add_workgroup_u32(shared_cnt[0], shared_cnt[1], s.uint(0), s.uint(1))
    s.begin_if(s.ult(slot, s.uint(UInt32(TILE_PRIM_CAP))))
    s.store_workgroup_u32(shared_prims[0], shared_prims[1], slot, p)
    s.end_if()
    s.end_if()

    s.store(p_var, s.iadd(p, s.uint(256)))
    s.end_loop()
    s.workgroup_barrier()

    # Early exit if tile is empty
    var total_prims = s.umin(s.load_workgroup_u32(shared_cnt[0], shared_cnt[1], s.uint(0)), s.uint(UInt32(TILE_PRIM_CAP)))
    s.return_if(s.ieq(total_prims, s.uint(0)))  # uniform across the workgroup

    # The atomic binning above lists a tile's prims in whatever order the invocations got
    # there, and compositing is order-dependent (a button's text over its background). Rank
    # sort back into draw order: invocation k places entry k at the count of smaller indices.
    s.begin_if(s.ult(local_idx, total_prims))
    var mine = s.load_workgroup_u32(shared_prims[0], shared_prims[1], local_idx)
    var rank_var = s.uvar(s.uint(0))
    var j_var = s.uvar(s.uint(0))
    s.begin_loop()
    var j = s.load_u(j_var)
    s.break_if(s.uge(j, total_prims))
    var other = s.load_workgroup_u32(shared_prims[0], shared_prims[1], j)
    s.begin_if(s.ult(other, mine))
    s.store(rank_var, s.iadd(s.load_u(rank_var), s.uint(1)))
    s.end_if()
    s.store(j_var, s.iadd(j, s.uint(1)))
    s.end_loop()
    s.store_workgroup_u32(ordered_prims[0], ordered_prims[1], s.load_u(rank_var), mine)
    s.end_if()
    s.workgroup_barrier()

    s.return_if(s.logical_or(s.uge(gx, out_w), s.uge(gy, out_h)))

    # Load 3D scene pixel from history buffer
    var pix_idx = s.iadd(out_ldr_off, s.iadd(s.imul(gy, out_w), gx))
    var bg_raw = s.load_word(b_hist, pix_idx)
    var bg_r = s.fdiv(s.float_from_uint(s.bit_and(bg_raw, s.uint(0xFF))), s.float(255.0))
    var bg_g = s.fdiv(s.float_from_uint(s.bit_and(s.shr(bg_raw, s.uint(8)), s.uint(0xFF))), s.float(255.0))
    var bg_b = s.fdiv(s.float_from_uint(s.bit_and(s.shr(bg_raw, s.uint(16)), s.uint(0xFF))), s.float(255.0))

    var accum_r = s.fvar(bg_r)
    var accum_g = s.fvar(bg_g)
    var accum_b = s.fvar(bg_b)

    var fgx = s.fadd(s.float_from_uint(gx), s.float(0.5))
    var fgy = s.fadd(s.float_from_uint(gy), s.float(0.5))

    var i_var = s.uvar(s.uint(0))
    s.begin_loop()
    var i = s.load_u(i_var)
    s.break_if(s.uge(i, total_prims))

    var prim_idx = s.load_workgroup_u32(ordered_prims[0], ordered_prims[1], i)
    var pb = s.iadd(prims_off, s.imul(prim_idx, s.uint(16)))

    # Clip rect check
    var cxy0 = s.load_word(b_post, s.iadd(pb, s.uint(8)))
    var cxy1 = s.load_word(b_post, s.iadd(pb, s.uint(9)))
    var cx0 = s.bit_and(cxy0, s.uint(0xFFFF))
    var cy0 = s.shr(cxy0, s.uint(16))
    var cx1 = s.bit_and(cxy1, s.uint(0xFFFF))
    var cy1 = s.shr(cxy1, s.uint(16))

    var in_clip = s.logical_and(
        s.logical_and(s.uge(gx, cx0), s.ult(gx, cx1)),
        s.logical_and(s.uge(gy, cy0), s.ult(gy, cy1)),
    )
    s.begin_if(in_clip)

    var x0 = s.load_float(b_post, pb)
    var y0 = s.load_float(b_post, s.iadd(pb, s.uint(1)))
    var x1 = s.load_float(b_post, s.iadd(pb, s.uint(2)))
    var y1 = s.load_float(b_post, s.iadd(pb, s.uint(3)))

    var in_box = s.logical_and(
        s.logical_and(s.fge(fgx, x0), s.fge(x1, fgx)),
        s.logical_and(s.fge(fgy, y0), s.fge(y1, fgy)),
    )
    s.begin_if(in_box)

    var hdr = s.load_word(b_post, s.iadd(pb, s.uint(15)))
    var kind = s.bit_and(hdr, s.uint(0xF))
    var tex_slot = s.bit_and(s.shr(hdr, s.uint(4)), s.uint(0xFFF))
    var flags = s.shr(hdr, s.uint(16))

    var fill_c = s.load_word(b_post, s.iadd(pb, s.uint(10)))
    var cr = s.fdiv(s.float_from_uint(s.bit_and(fill_c, s.uint(0xFF))), s.float(255.0))
    var cg = s.fdiv(s.float_from_uint(s.bit_and(s.shr(fill_c, s.uint(8)), s.uint(0xFF))), s.float(255.0))
    var cb = s.fdiv(s.float_from_uint(s.bit_and(s.shr(fill_c, s.uint(16)), s.uint(0xFF))), s.float(255.0))
    var ca = s.fdiv(s.float_from_uint(s.bit_and(s.shr(fill_c, s.uint(24)), s.uint(0xFF))), s.float(255.0))
    var alpha_var = s.fvar(ca)

    # PRIM_GLYPH: Sample bindless R8_UNORM font atlas
    s.begin_if(s.ieq(kind, s.uint(1)))
    var u0 = s.load_float(b_post, s.iadd(pb, s.uint(4)))
    var v0 = s.load_float(b_post, s.iadd(pb, s.uint(5)))
    var u1 = s.load_float(b_post, s.iadd(pb, s.uint(6)))
    var v1 = s.load_float(b_post, s.iadd(pb, s.uint(7)))
    var tu = s.fadd(u0, s.fmul(s.fdiv(s.fsub(fgx, x0), s.fsub(x1, x0)), s.fsub(u1, u0)))
    var tv = s.fadd(v0, s.fmul(s.fdiv(s.fsub(fgy, y0), s.fsub(y1, y0)), s.fsub(v1, v0)))
    var uv = s.vec2(tu, tv)
    var samp = s.sample_bindless_lod(bindless_arr, tex_slot, uv, s.float(0.0))
    var d_val = s.extract_f(samp, 0)
    var param0 = s.load_float(b_post, s.iadd(pb, s.uint(12)))
    var sd = s.fmul(s.fsub(d_val, s.float(0.5)), param0)
    var cov = s.nclamp(s.fadd(sd, s.float(0.5)), s.float(0.0), s.float(1.0))
    s.store(alpha_var, s.fmul(ca, cov))
    s.end_if()

    # PRIM_SHAPE: Rounded box / Circle SDF
    s.begin_if(s.ieq(kind, s.uint(0)))
    var radius = s.load_float(b_post, s.iadd(pb, s.uint(12)))
    var w_half = s.fmul(s.fsub(x1, x0), s.float(0.5))
    var h_half = s.fmul(s.fsub(y1, y0), s.float(0.5))
    var cx = s.fadd(x0, w_half)
    var cy = s.fadd(y0, h_half)
    var is_circle = s.ine(s.bit_and(flags, s.uint(4)), s.uint(0))
    s.begin_if(is_circle)
    var dx = s.fsub(fgx, cx)
    var dy = s.fsub(fgy, cy)
    var dist_c = s.fsub(s.fsqrt(s.fadd(s.fmul(dx, dx), s.fmul(dy, dy))), radius)
    var cov_c = s.nclamp(s.fsub(s.float(0.5), dist_c), s.float(0.0), s.float(1.0))
    s.store(alpha_var, s.fmul(ca, cov_c))
    s.begin_else()
    var qx = s.fsub(s.fabs(s.fsub(fgx, cx)), s.fsub(w_half, radius))
    var qy = s.fsub(s.fabs(s.fsub(fgy, cy)), s.fsub(h_half, radius))
    var ox = s.nmax(qx, s.float(0.0))
    var oy = s.nmax(qy, s.float(0.0))
    var dist_r = s.fsub(s.fadd(s.fsqrt(s.fadd(s.fmul(ox, ox), s.fmul(oy, oy))), s.nmin(s.nmax(qx, qy), s.float(0.0))), radius)
    var cov_r = s.nclamp(s.fsub(s.float(0.5), dist_r), s.float(0.0), s.float(1.0))
    s.store(alpha_var, s.fmul(ca, cov_r))
    s.end_if()
    s.end_if()

    # Linear premultiplied alpha composite
    var eff_a = s.load_f(alpha_var)
    var one_minus_a = s.fsub(s.float(1.0), eff_a)
    var cur_r = s.load_f(accum_r)
    var cur_g = s.load_f(accum_g)
    var cur_b = s.load_f(accum_b)
    s.store(accum_r, s.fadd(s.fmul(cr, eff_a), s.fmul(cur_r, one_minus_a)))
    s.store(accum_g, s.fadd(s.fmul(cg, eff_a), s.fmul(cur_g, one_minus_a)))
    s.store(accum_b, s.fadd(s.fmul(cb, eff_a), s.fmul(cur_b, one_minus_a)))

    s.end_if() # in_box
    s.end_if() # in_clip

    s.store(i_var, s.iadd(i, s.uint(1)))
    s.end_loop()

    # Re-pack RGBA8 and store back to history buffer
    var final_r = s.saturate_uint(s.fmul(s.nclamp(s.load_f(accum_r), s.float(0.0), s.float(1.0)), s.float(255.0)))
    var final_g = s.saturate_uint(s.fmul(s.nclamp(s.load_f(accum_g), s.float(0.0), s.float(1.0)), s.float(255.0)))
    var final_b = s.saturate_uint(s.fmul(s.nclamp(s.load_f(accum_b), s.float(0.0), s.float(1.0)), s.float(255.0)))
    var packed_out = s.bit_or(
        s.bit_or(final_r, s.shl(final_g, s.uint(8))),
        s.bit_or(s.shl(final_b, s.uint(16)), s.shl(s.uint(255), s.uint(24))),
    )
    s.store_word(b_hist, pix_idx, packed_out)

    return s.finish()
