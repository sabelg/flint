"""Sota:geometry — virtual texturing (VT) with GPU feedback and budgeted page streaming.

A virtual texture is a mip chain of `128 x 128` pages (up to `16k x 16k`, `lp = log2(pages per side) <= 7`).
Only the pages the camera needs are resident in one shared physical atlas (`16 x 16` slots of
`136 x 136` = page + 4-texel border, `B8G8R8A8_SRGB`, 18.9 MB), whatever the virtual size.

  Feedback   a compute pass over 1 in 16 visbuffer pixels (4x4 stride, per-frame jitter) rebuilds the
             triangle, the perspective-correct UV and its screen-space derivatives (same 3-ray
             differentials as the deferred shade), picks the mip, and stamps the frame number into
             that (VT, mip, page) word of a feedback buffer. No clear pass: a word equals this frame's
             stamp iff it was requested.
  Streaming  the CPU reads the previous frame's stamps, queues every missing page in an
             `AaaResidencyStreamer` (aaa_render_graph.mojo; coarse mips first), drains it within a
             per-frame page budget, evicts least-recently-requested slots (the 1-page root mip is pinned),
             and records the page copies (staging -> atlas) at the top of the frame's command buffer.
  Page table one indirection texture per VT holding every mip side by side (`2P x P` texels). Each texel
             is `(slot x, slot y, resident mip, 255)` of the finest resident ancestor of that page, so a
             missing page always falls back to the next coarser resident one (never garbage).
  Shading    the deferred shade (hook in aaa_gi_svgf.mojo) recognises a VT material by a negative word
             15 (`-(1 + indirection slot + 256 * atlas slot + 65536 * lp + 2^20 * vt)`), looks up the
             page table at mips floor(m) and floor(m) + 1 and blends them (trilinear across pages).
             RT/GI hits keep the material's base colour, which is set to the texture's mean.

Sources: Barrett "Sparse Virtual Textures" (GDC 2008); van Waveren "id Tech 5 Challenges" (SIGGRAPH 2009);
Mittring "Advanced Virtual Texture Topics" (SIGGRAPH 2008). No code copied.
"""
from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.math import max, min
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    VulkanGpu,
    VulkanImage,
    get32,
    put32,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_HOST_READ,
    ACCESS_TRANSFER_WRITE,
    IMAGE_USAGE_SAMPLED,
    IMAGE_USAGE_TRANSFER_TARGET,
    LAYOUT_SHADER_READ_ONLY,
    LAYOUT_TRANSFER_TARGET,
    PIPELINE_COMPUTE,
    PIPELINE_HOST,
    PIPELINE_TOP,
    PIPELINE_TRANSFER,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    BIND_AAA_BINDLESS_TEX,
    emit_instance_point,
    emit_ray_barycentrics,
    emit_vec3_normalize,
)
from flint.render.aaa_render_graph import AaaResidencyStreamer
from flint.platform.os import get_time

comptime SOTA_VT_PAGE: Int = 128
comptime SOTA_VT_BORDER: Int = 4
comptime SOTA_VT_SLOT: Int = 136                        # page + 2 * border
comptime SOTA_VT_ATLAS_SLOTS: Int = 16                  # per side -> 256 resident pages
comptime SOTA_VT_ATLAS_DIM: Int = 2176
comptime SOTA_VT_MAX_LP: Int = 7                        # 128 pages per side -> 16k x 16k
comptime SOTA_VT_MAX_TEXTURES: Int = 16
comptime SOTA_VT_FB_STRIDE: Int = 21845                 # (4^8 - 1) / 3 words: every page of every mip at lp = 7
comptime SOTA_VT_FB_STEP: Int = 4                       # 1 feedback sample per 4x4 pixels
comptime SOTA_VT_PAGE_BYTES: Int = 136 * 136 * 4
comptime SOTA_VT_FORMAT_BGRA8_SRGB: UInt32 = 50
comptime SOTA_VT_FORMAT_BGRA8_UNORM: UInt32 = 44
comptime SOTA_VT_STAGE_ALL: Int = 0x10000

# Feedback pass layout (own descriptor sets; the shared AAA layout is untouched)
comptime SOTA_VT_BIND_VERTS: Int = 0
comptime SOTA_VT_BIND_INDICES: Int = 1
comptime SOTA_VT_BIND_MESHLETS: Int = 2
comptime SOTA_VT_BIND_INSTANCES: Int = 3
comptime SOTA_VT_BIND_MATERIALS: Int = 4
comptime SOTA_VT_BIND_GLOBALS: Int = 5
comptime SOTA_VT_BIND_FEEDBACK: Int = 6
comptime SOTA_VT_BIND_SPARE: Int = 7
comptime SOTA_VT_STORAGE_COUNT: Int = 8
comptime SOTA_VT_BIND_VIS: Int = 8                      # usampler2D visbuffer
comptime SOTA_VT_PUSH_WORDS: Int = 16
comptime SOTA_VT_MATERIAL_WORDS: Int = 24               # AAA_MATERIAL_WORDS
comptime SOTA_VT_MATERIAL_CODE_WORD: Int = 15           # iridescence IOR word; negative = VT code


def vt_mip_offset(lp: Int, m: Int) -> Int:
    """Word offset of mip `m` in a VT's feedback / page arrays (mip 0 first): sum_{k<m} 4^(lp-k)."""
    return ((1 << (2 * lp + 2)) - (1 << (2 * (lp - m) + 2))) // 3


def vt_total_pages(lp: Int) -> Int:
    return vt_mip_offset(lp, lp + 1)


def vt_encode_material_code(ind_slot: Int, atlas_slot: Int, lp: Int, vt_index: Int) -> Float32:
    """Material word 15 for a VT material (exact in f32: the code is < 2^24)."""
    var code = ind_slot + 256 * atlas_slot + 65536 * lp + (1 << 20) * vt_index
    return -Float32(1 + code)


# ---------------------------------------------------------------------------------------------
# Shading: page-table lookup + atlas fetch (used by the deferred-shade hook in aaa_gi_svgf.mojo)
# ---------------------------------------------------------------------------------------------
def _emit_vt_fetch_mip(
    mut s: AaaSpirvModule, bindless_arr: Tuple[UInt32, UInt32], ind: UInt32, atl: UInt32, lp: UInt32,
    pages0: UInt32, fu: UInt32, fv: UInt32, mf: UInt32,
) -> UInt32:
    """Sample the atlas for virtual mip `mf` (integral float), falling back through the page table."""
    var ind_w = s.fmul(pages0, s.float(2.0))
    var pm = s.fmul(pages0, s.exp2(s.fneg(mf)))
    var tx = s.nmin(s.ffloor(s.fmul(fu, pm)), s.fsub(pm, s.float(1.0)))
    var ty = s.nmin(s.ffloor(s.fmul(fv, pm)), s.fsub(pm, s.float(1.0)))
    var xoff = s.fsub(ind_w, s.fmul(ind_w, s.exp2(s.fneg(mf))))
    var iu = s.fdiv(s.fadd(s.fadd(xoff, tx), s.float(0.5)), ind_w)
    var iv = s.fdiv(s.fadd(ty, s.float(0.5)), pages0)
    var e = s.sample_bindless_lod(bindless_arr, ind, s.vec2(iu, iv), s.float(0.0))
    var sx = s.ffloor(s.fma(s.extract_f(e, 0), s.float(255.0), s.float(0.5)))
    var sy = s.ffloor(s.fma(s.extract_f(e, 1), s.float(255.0), s.float(0.5)))
    var rm = s.ffloor(s.fma(s.extract_f(e, 2), s.float(255.0), s.float(0.5)))
    var prm = s.fmul(pages0, s.exp2(s.fneg(rm)))
    var lu = s.fmul(s.fract(s.fmul(fu, prm)), s.float(Float32(SOTA_VT_PAGE)))
    var lv = s.fmul(s.fract(s.fmul(fv, prm)), s.float(Float32(SOTA_VT_PAGE)))
    var inv_atlas = Float32(1.0) / Float32(SOTA_VT_ATLAS_DIM)
    var au = s.fmul(s.fadd(s.fma(sx, s.float(Float32(SOTA_VT_SLOT)), s.float(Float32(SOTA_VT_BORDER))), lu), s.float(inv_atlas))
    var av = s.fmul(s.fadd(s.fma(sy, s.float(Float32(SOTA_VT_SLOT)), s.float(Float32(SOTA_VT_BORDER))), lv), s.float(inv_atlas))
    _ = lp
    return s.sample_bindless_lod(bindless_arr, atl, s.vec2(au, av), s.float(0.0))


def emit_vt_albedo(
    mut s: AaaSpirvModule, bindless_arr: Tuple[UInt32, UInt32], code_word_f: UInt32,
    uv_u: UInt32, uv_v: UInt32, ddx_u: UInt32, ddx_v: UInt32, ddy_u: UInt32, ddy_v: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Linear albedo of a VT material. `code_word_f` = material word 15 (< 0). Trilinear across VT mips."""
    var code = s.uint_from_float(s.fadd(s.fsub(s.fneg(code_word_f), s.float(1.0)), s.float(0.5)))
    var ind = s.bit_and(code, s.uint(0xFF))
    var atl = s.bit_and(s.shr(code, s.uint(8)), s.uint(0xFF))
    var lp = s.bit_and(s.shr(code, s.uint(16)), s.uint(0xF))
    var lpf = s.float_from_uint(lp)
    var pages0 = s.exp2(lpf)
    var dim = s.fmul(pages0, s.float(Float32(SOTA_VT_PAGE)))
    var lx = s.fsqrt(s.fma(ddx_u, ddx_u, s.fmul(ddx_v, ddx_v)))
    var ly = s.fsqrt(s.fma(ddy_u, ddy_u, s.fmul(ddy_v, ddy_v)))
    var rho = s.fmul(s.nmax(lx, ly), dim)
    var m = s.nclamp(s.log2(s.nmax(rho, s.float(1e-6))), s.float(0.0), lpf)
    var m0 = s.ffloor(m)
    var t = s.fsub(m, m0)
    var m1 = s.nmin(s.fadd(m0, s.float(1.0)), lpf)
    var fu = s.fract(uv_u)
    var fv = s.fract(uv_v)
    var c0 = _emit_vt_fetch_mip(s, bindless_arr, ind, atl, lp, pages0, fu, fv, m0)
    var c1 = _emit_vt_fetch_mip(s, bindless_arr, ind, atl, lp, pages0, fu, fv, m1)
    var r = s.fma(s.fsub(s.extract_f(c1, 0), s.extract_f(c0, 0)), t, s.extract_f(c0, 0))
    var g = s.fma(s.fsub(s.extract_f(c1, 1), s.extract_f(c0, 1)), t, s.extract_f(c0, 1))
    var b = s.fma(s.fsub(s.extract_f(c1, 2), s.extract_f(c0, 2)), t, s.extract_f(c0, 2))
    return (r, g, b)


# ---------------------------------------------------------------------------------------------
# Feedback pass
# ---------------------------------------------------------------------------------------------
def sota_vt_feedback_shader() -> List[UInt32]:
    """Push: [0] render w [1] render h [2] stamp [3] jitter x [4] jitter y (sample offset inside the 4x4 cell)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=SOTA_VT_PUSH_WORDS)
    var b_verts = s.buffer(SOTA_VT_BIND_VERTS)
    var b_indices = s.buffer(SOTA_VT_BIND_INDICES)
    var b_meshlets = s.buffer(SOTA_VT_BIND_MESHLETS)
    var b_instances = s.buffer(SOTA_VT_BIND_INSTANCES)
    var b_materials = s.buffer(SOTA_VT_BIND_MATERIALS)
    var b_globals = s.buffer(SOTA_VT_BIND_GLOBALS)
    var b_fb = s.buffer(SOTA_VT_BIND_FEEDBACK)
    var vis_samp = s.sampled_image_2d(SOTA_VT_BIND_VIS, is_uint=True)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var stamp = s.push_word(2)
    var jx = s.push_word(3)
    var jy = s.push_word(4)
    var px = s.iadd(s.imul(s.invocation_axis(0), s.uint(UInt32(SOTA_VT_FB_STEP))), jx)
    var py = s.iadd(s.imul(s.invocation_axis(1), s.uint(UInt32(SOTA_VT_FB_STEP))), jy)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var vis = s.texel_fetch_u2d(vis_samp, s.uvec2(px, py), s.uint(0))
    var vis0 = s.extract_u(vis, 0)
    var vis1 = s.extract_u(vis, 1)
    s.return_if(s.ieq(vis0, s.uint(0)))
    var inst_id = s.shr(vis0, s.uint(17))
    var ml_id = s.isub(s.bit_and(vis0, s.uint(0x1FFFF)), s.uint(1))
    var mat_id = s.shr(vis1, s.uint(16))
    var local_prim = s.bit_and(vis1, s.uint(0xFFFF))
    var code_f = s.load_float(b_materials, s.iadd(s.imul(mat_id, s.uint(UInt32(SOTA_VT_MATERIAL_WORDS))), s.uint(UInt32(SOTA_VT_MATERIAL_CODE_WORD))))
    s.return_if(s.fge(code_f, s.float(0.0)))
    var code = s.uint_from_float(s.fadd(s.fsub(s.fneg(code_f), s.float(1.0)), s.float(0.5)))
    var lp = s.bit_and(s.shr(code, s.uint(16)), s.uint(0xF))
    var vt_i = s.shr(code, s.uint(20))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))
    var fx0 = s.fsub(s.fadd(s.float_from_uint(px), s.float(0.5)), s.fmul(jit_x, s.fmul(fw, s.float(0.5))))
    var fy0 = s.fadd(s.fadd(s.float_from_uint(py), s.float(0.5)), s.fmul(jit_y, s.fmul(fh, s.float(0.5))))
    var ndc_x = s.fsub(s.fmul(s.fdiv(fx0, fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(fy0, fh), s.float(2.0)))
    var step_nx = s.fdiv(s.float(2.0), fw)
    var step_ny = s.fneg(s.fdiv(s.float(2.0), fh))
    var ray_c = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var nx1 = s.fadd(ndc_x, step_nx)
    var ray_dx = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(nx1, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(nx1, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(nx1, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var ny1 = s.fadd(ndc_y, step_ny)
    var ray_dy = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ny1, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ny1, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ny1, tan_y), fwd_z)),
    )

    var ml_base = s.imul(ml_id, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var idx_off = s.imul(s.iadd(first_tri, local_prim), s.uint(3))
    var v0_off = s.imul(s.load_word(b_indices, idx_off), s.uint(12))
    var v1_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(1))), s.uint(12))
    var v2_off = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(2))), s.uint(12))
    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var p0 = emit_instance_point(
        s, b_instances, b_globals, b_verts, inst_base, v0_off,
        s.load_float(b_verts, v0_off), s.load_float(b_verts, s.iadd(v0_off, s.uint(1))), s.load_float(b_verts, s.iadd(v0_off, s.uint(2))),
    )
    var p1 = emit_instance_point(
        s, b_instances, b_globals, b_verts, inst_base, v1_off,
        s.load_float(b_verts, v1_off), s.load_float(b_verts, s.iadd(v1_off, s.uint(1))), s.load_float(b_verts, s.iadd(v1_off, s.uint(2))),
    )
    var p2 = emit_instance_point(
        s, b_instances, b_globals, b_verts, inst_base, v2_off,
        s.load_float(b_verts, v2_off), s.load_float(b_verts, s.iadd(v2_off, s.uint(1))), s.load_float(b_verts, s.iadd(v2_off, s.uint(2))),
    )
    var bary = emit_ray_barycentrics(
        s, cam_x, cam_y, cam_z,
        ray_c[0], ray_c[1], ray_c[2], ray_dx[0], ray_dx[1], ray_dx[2], ray_dy[0], ray_dy[1], ray_dy[2],
        p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
    )
    var u0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(9)))
    var t0 = s.load_float(b_verts, s.iadd(v0_off, s.uint(10)))
    var u1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(9)))
    var t1 = s.load_float(b_verts, s.iadd(v1_off, s.uint(10)))
    var u2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(9)))
    var t2 = s.load_float(b_verts, s.iadd(v2_off, s.uint(10)))
    var b1 = bary[0]
    var b2 = bary[1]
    var b0 = s.fsub(s.fsub(s.float(1.0), b1), b2)
    var uu = s.fma(u0, b0, s.fma(u1, b1, s.fmul(u2, b2)))
    var vv = s.fma(t0, b0, s.fma(t1, b1, s.fmul(t2, b2)))
    var b0_dx = s.fsub(s.fsub(s.float(1.0), bary[2]), bary[3])
    var b0_dy = s.fsub(s.fsub(s.float(1.0), bary[4]), bary[5])
    var bias = s.exp2(s.load_float(b_globals, s.uint(127)))   # same TSR mip bias as the shade pass
    var ddx_u = s.fmul(s.fsub(s.fma(u0, b0_dx, s.fma(u1, bary[2], s.fmul(u2, bary[3]))), uu), bias)
    var ddx_v = s.fmul(s.fsub(s.fma(t0, b0_dx, s.fma(t1, bary[2], s.fmul(t2, bary[3]))), vv), bias)
    var ddy_u = s.fmul(s.fsub(s.fma(u0, b0_dy, s.fma(u1, bary[4], s.fmul(u2, bary[5]))), uu), bias)
    var ddy_v = s.fmul(s.fsub(s.fma(t0, b0_dy, s.fma(t1, bary[4], s.fmul(t2, bary[5]))), vv), bias)

    var lpf = s.float_from_uint(lp)
    var dim = s.fmul(s.exp2(lpf), s.float(Float32(SOTA_VT_PAGE)))
    var lx = s.fsqrt(s.fma(ddx_u, ddx_u, s.fmul(ddx_v, ddx_v)))
    var ly = s.fsqrt(s.fma(ddy_u, ddy_u, s.fmul(ddy_v, ddy_v)))
    var mf = s.nclamp(s.ffloor(s.log2(s.nmax(s.fmul(s.nmax(lx, ly), dim), s.float(1e-6)))), s.float(0.0), lpf)
    var mi = s.uint_from_float(mf)
    var shift = s.isub(lp, mi)
    var pm = s.shl(s.uint(1), shift)
    var pmf = s.float_from_uint(pm)
    var pgx = s.umin(s.uint_from_float(s.fmul(s.fract(uu), pmf)), s.isub(pm, s.uint(1)))
    var pgy = s.umin(s.uint_from_float(s.fmul(s.fract(vv), pmf)), s.isub(pm, s.uint(1)))
    # mip offset = (4^(lp+1) - 4^(lp-m+1)) / 3
    var full = s.shl(s.uint(1), s.iadd(s.imul(lp, s.uint(2)), s.uint(2)))
    var part = s.shl(s.uint(1), s.iadd(s.imul(shift, s.uint(2)), s.uint(2)))
    var moff = s.udiv(s.isub(full, part), s.uint(3))
    var off = s.iadd(s.imul(vt_i, s.uint(UInt32(SOTA_VT_FB_STRIDE))), s.iadd(moff, s.iadd(s.imul(pgy, pm), pgx)))
    s.store_word(b_fb, off, stamp)
    return s.finish()


# ---------------------------------------------------------------------------------------------
# CPU residency: page table, LRU slots, fallback indirection (pure CPU, unit-tested)
# ---------------------------------------------------------------------------------------------
struct VtPageUpload(Copyable, ImplicitlyCopyable, Movable):
    var vt: Int
    var mip: Int
    var px: Int
    var py: Int
    var slot: Int

    def __init__(out self, vt: Int, mip: Int, px: Int, py: Int, slot: Int):
        self.vt = vt
        self.mip = mip
        self.px = px
        self.py = py
        self.slot = slot


struct VtResidency(Movable):
    """Physical-slot allocator + per-VT page tables. Keys: `vt * SOTA_VT_FB_STRIDE + vt_mip_offset(lp, m) + y * Pm + x`."""
    var n_slots: Int
    var slot_key: List[Int]          # -1 = free
    var slot_last: List[Int]         # frame stamp of the last request
    var slot_pinned: List[Bool]
    var vt_lp: List[Int]
    var page_slot: List[List[Int]]   # per VT, per page key (local): slot or -1
    var streamer: AaaResidencyStreamer
    var evictions_total: Int
    var uploads_total: Int
    var last_requests: Int
    var last_missing: Int

    def __init__(out self, n_slots: Int = SOTA_VT_ATLAS_SLOTS * SOTA_VT_ATLAS_SLOTS):
        self.n_slots = n_slots
        self.slot_key = List[Int](length=n_slots, fill=-1)
        self.slot_last = List[Int](length=n_slots, fill=0)
        self.slot_pinned = List[Bool](length=n_slots, fill=False)
        self.vt_lp = List[Int]()
        self.page_slot = List[List[Int]]()
        self.streamer = AaaResidencyStreamer(max_textures=0)
        self.evictions_total = 0
        self.uploads_total = 0
        self.last_requests = 0
        self.last_missing = 0

    def add_vt(mut self, lp: Int) -> Int:
        self.vt_lp.append(lp)
        self.page_slot.append(List[Int](length=vt_total_pages(lp), fill=-1))
        return len(self.vt_lp) - 1

    def resident_pages(self) -> Int:
        var n = 0
        for k in self.slot_key:
            if k >= 0:
                n += 1
        return n

    def _alloc_slot(mut self, stamp: Int) -> Int:
        """A free slot, else the least-recently-requested unpinned slot not requested at `stamp`."""
        var best = -1
        var best_last = stamp
        for i in range(self.n_slots):
            if self.slot_key[i] < 0:
                return i
            if not self.slot_pinned[i] and self.slot_last[i] < best_last:
                best_last = self.slot_last[i]
                best = i
        if best >= 0:
            var key = self.slot_key[best]
            var vt = key // SOTA_VT_FB_STRIDE
            self.page_slot[vt][key - vt * SOTA_VT_FB_STRIDE] = -1
            self.slot_key[best] = -1
            self.evictions_total += 1
        return best

    def place(mut self, vt: Int, mip: Int, px: Int, py: Int, stamp: Int, pinned: Bool = False) -> Int:
        var lp = self.vt_lp[vt]
        var local = vt_mip_offset(lp, mip) + py * (1 << (lp - mip)) + px
        if self.page_slot[vt][local] >= 0:
            return self.page_slot[vt][local]
        var sl = self._alloc_slot(stamp)
        if sl < 0:
            return -1
        self.page_slot[vt][local] = sl
        self.slot_key[sl] = vt * SOTA_VT_FB_STRIDE + local
        self.slot_last[sl] = stamp
        self.slot_pinned[sl] = pinned
        self.uploads_total += 1
        return sl

    def process_feedback(mut self, words: List[UInt32], stamp: Int, budget_pages: Int) -> List[VtPageUpload]:
        """`words` = the feedback buffer (`SOTA_VT_FB_STRIDE` words per VT). Marks requested resident pages as
        used, queues the missing ones in the residency streamer (coarse mips first) and drains up to
        `budget_pages` of them into slots. Returns the uploads to perform."""
        self.streamer.upload_queue.clear()
        var st = UInt32(stamp)
        var reqs = 0
        var missing = 0
        for vt in range(len(self.vt_lp)):
            var lp = self.vt_lp[vt]
            var base = vt * SOTA_VT_FB_STRIDE
            var n = vt_total_pages(lp)
            if base + n > len(words):
                continue
            for local in range(n):
                if words[base + local] != st:
                    continue
                reqs += 1
                var sl = self.page_slot[vt][local]
                if sl >= 0:
                    self.slot_last[sl] = stamp
                    continue
                var mip = 0
                while mip < lp and local >= vt_mip_offset(lp, mip + 1):
                    mip += 1
                missing += 1
                if len(self.streamer.upload_queue) < 4096:
                    self.streamer.enqueue_mip_upload(base + local, mip, SOTA_VT_PAGE_BYTES, 0, False)
        self.last_requests = reqs
        self.last_missing = missing
        var out = List[VtPageUpload]()
        var drained = self.streamer.drain_uploads_within_budget(max(1, budget_pages) * SOTA_VT_PAGE_BYTES)
        for it in drained:
            var key = it.target_id
            var vt = key // SOTA_VT_FB_STRIDE
            var local = key - vt * SOTA_VT_FB_STRIDE
            var lp = self.vt_lp[vt]
            var mip = it.sub_level
            var rel = local - vt_mip_offset(lp, mip)
            var pm = 1 << (lp - mip)
            var sl = self.place(vt, mip, rel % pm, rel // pm, stamp)
            if sl >= 0:
                out.append(VtPageUpload(vt, mip, rel % pm, rel // pm, sl))
        return out^

    def indirection_texels(self, vt: Int) -> List[UInt32]:
        """`2P x P` ARGB texels: every mip side by side (mip m at x = 2P - 2P/2^m). Each texel names the finest
        resident ancestor page: R = slot x, G = slot y, B = its mip. Unresolvable (no root) -> slot 0, mip lp."""
        var lp = self.vt_lp[vt]
        var p0 = 1 << lp
        var w = 2 * p0
        var tex = List[UInt32](length=w * p0, fill=UInt32(0))
        for m in range(lp + 1):
            var pm = 1 << (lp - m)
            var xoff = w - (w >> m)
            for y in range(pm):
                for x in range(pm):
                    var am = m
                    var ax = x
                    var ay = y
                    var sl: Int
                    while True:
                        sl = self.page_slot[vt][vt_mip_offset(lp, am) + ay * (1 << (lp - am)) + ax]
                        if sl >= 0 or am >= lp:
                            break
                        am += 1
                        ax >>= 1
                        ay >>= 1
                    if sl < 0:
                        sl = 0
                        am = lp
                    var sx = sl % SOTA_VT_ATLAS_SLOTS
                    var sy = sl // SOTA_VT_ATLAS_SLOTS
                    tex[y * w + xoff + x] = (UInt32(255) << 24) | (UInt32(sx) << 16) | (UInt32(sy) << 8) | UInt32(am)
        return tex^


# ---------------------------------------------------------------------------------------------
# Source data: mip chain of the virtual image (ARGB, in RAM; a disk/.flpk tile source plugs in here)
# ---------------------------------------------------------------------------------------------
def vt_build_mips(pixels: List[UInt32], dim: Int) -> List[List[UInt32]]:
    """2x2 box-filtered mip chain (in linear light) down to 128 x 128 (the root page)."""
    var out = List[List[UInt32]]()
    out.append(pixels.copy())
    var cur = dim
    var lut = List[Float32](length=256, fill=0.0)
    for i in range(256):
        var c = Float32(i) / Float32(255.0)
        lut[i] = c / Float32(12.92) if c <= Float32(0.04045) else ((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4)
    while cur > SOTA_VT_PAGE:
        var half = cur // 2
        var nxt = List[UInt32](length=half * half, fill=UInt32(0))
        ref src = out[len(out) - 1]
        for y in range(half):
            for x in range(half):
                var acc_r = Float32(0.0)
                var acc_g = Float32(0.0)
                var acc_b = Float32(0.0)
                var acc_a = 0
                for dy in range(2):
                    for dx in range(2):
                        var p = src[(2 * y + dy) * cur + 2 * x + dx]
                        acc_r += lut[Int((p >> 16) & 255)]
                        acc_g += lut[Int((p >> 8) & 255)]
                        acc_b += lut[Int(p & 255)]
                        acc_a += Int(p >> 24)
                nxt[y * half + x] = (UInt32(acc_a // 4) << 24) | (UInt32(_lin_to_srgb8(acc_r * 0.25)) << 16) | (UInt32(_lin_to_srgb8(acc_g * 0.25)) << 8) | UInt32(_lin_to_srgb8(acc_b * 0.25))
        out.append(nxt^)
        cur = half
    return out^


def _lin_to_srgb8(v: Float32) -> Int:
    var c = max(Float32(0.0), min(Float32(1.0), v))
    var s = c * Float32(12.92) if c <= Float32(0.0031308) else Float32(1.055) * c ** (Float32(1.0) / Float32(2.4)) - Float32(0.055)
    return Int(s * Float32(255.0) + Float32(0.5))


def vt_fill_page(mips: List[List[UInt32]], lp: Int, mip: Int, px: Int, py: Int, dst: Int):
    """Write the 136 x 136 slot image (page + 4-texel wrapped border) of page (px, py) at `mip` to `dst`."""
    var dim = (1 << (lp - mip)) * SOTA_VT_PAGE
    ref src = mips[mip]
    var x0 = px * SOTA_VT_PAGE - SOTA_VT_BORDER
    var y0 = py * SOTA_VT_PAGE - SOTA_VT_BORDER
    for y in range(SOTA_VT_SLOT):
        var sy = (y0 + y + dim) % dim
        var row = dst + y * SOTA_VT_SLOT * 4
        var sx0 = x0 + dim
        for x in range(SOTA_VT_SLOT):
            var sx = (sx0 + x) % dim
            put32(row + x * 4, src[sy * dim + sx])


# ---------------------------------------------------------------------------------------------
# Renderer-side state
# ---------------------------------------------------------------------------------------------
struct SotaVirtualTextures(Movable):
    """Area A virtual texturing owned by `FlintAaaVulkanRenderer` (field `sota_vt`). Inactive (zero cost) until
    the first `add_virtual_texture`."""
    var enabled: Bool                  # cvar r.sota.vt: feedback + streaming (off = keep what is resident)
    var budget_pages: Int              # cvar r.sota.vt.budget: page uploads per frame (16 = 1.2 MB)
    var model_textures: Bool           # cvar r.sota.vt.models: upload_native_model streams big square albedo maps via VT
    var model_min_dim: Int             # cvar r.sota.vt.models_min: smallest albedo side that goes through VT
    var active: Bool
    var res: VtResidency
    var mips: List[List[List[UInt32]]]
    var ind_slot: List[Int]
    var ind_img: List[VulkanImage]
    var ind_dirty: List[Bool]
    var mean_rgb: List[Tuple[Float32, Float32, Float32]]
    var atlas: VulkanImage
    var atlas_slot: Int
    var fb: List[GpuBuffer]
    var fb_stamp: List[Int]
    var staging: List[GpuBuffer]
    var pending: List[List[VtPageUpload]]
    var pending_ind: List[List[Int]]
    var dsl: Int
    var layout: Int
    var pool: Int
    var sets: List[Int]
    var pipe_fb: Int
    var last_cpu_ms: Float64
    var last_uploads: Int

    def __init__(out self):
        self.enabled = True
        self.budget_pages = 16
        self.model_textures = False
        self.model_min_dim = 1024
        self.active = False
        self.res = VtResidency()
        self.mips = List[List[List[UInt32]]]()
        self.ind_slot = List[Int]()
        self.ind_img = List[VulkanImage]()
        self.ind_dirty = List[Bool]()
        self.mean_rgb = List[Tuple[Float32, Float32, Float32]]()
        self.atlas = VulkanImage()
        self.atlas_slot = -1
        self.fb = List[GpuBuffer]()
        self.fb_stamp = List[Int]()
        self.staging = List[GpuBuffer]()
        self.pending = List[List[VtPageUpload]]()
        self.pending_ind = List[List[Int]]()
        self.dsl = 0
        self.layout = 0
        self.pool = 0
        self.sets = List[Int]()
        self.pipe_fb = 0
        self.last_cpu_ms = 0.0
        self.last_uploads = 0

    def wants_model_texture(self, width: Int, height: Int) -> Bool:
        """Square power-of-two albedo maps of at least `model_min_dim`, while VT slots remain."""
        if not (self.model_textures and self.active and self.enabled) or width != height:
            return False
        if width < max(SOTA_VT_PAGE, self.model_min_dim) or width > 16384 or (width & (width - 1)) != 0:
            return False
        return self.count() < SOTA_VT_MAX_TEXTURES

    def count(self) -> Int:
        return len(self.mips)

    def _staging_bytes(self) -> Int:
        return max(1, self.budget_pages) * SOTA_VT_PAGE_BYTES + SOTA_VT_MAX_TEXTURES * 2 * 128 * 128 * 4

    def init(
        mut self, mut vk: VulkanGpu, frames: Int,
        buf_vertices: GpuBuffer, buf_indices: GpuBuffer, buf_meshlets: GpuBuffer, ring_instances: List[GpuBuffer],
        buf_materials: GpuBuffer, ring_globals: List[GpuBuffer], vis_view: Int, sampler: Int,
    ) -> Bool:
        """Pipeline, per-frame feedback + staging buffers and descriptor sets. The atlas is created lazily."""
        var lay = vk.create_aaa_layouts(SOTA_VT_STORAGE_COUNT, 1, 0, False, SOTA_VT_PUSH_WORDS * 4)
        self.dsl = lay[0]
        self.layout = lay[1]
        if self.dsl == 0 or self.layout == 0:
            return False
        var ps = vk.allocate_aaa_descriptor_sets(self.dsl, frames, SOTA_VT_STORAGE_COUNT, 1, 0, False)
        self.pool = ps[0]
        var sets = ps[1].copy()
        if len(sets) < frames:
            return False
        self.pipe_fb = vk.create_aaa_compute_pipeline(self.layout, sota_vt_feedback_shader())
        if self.pipe_fb == 0:
            return False
        for f in range(frames):
            var b = vk.create_buffer(SOTA_VT_MAX_TEXTURES * SOTA_VT_FB_STRIDE * 4, local=False, cpu_reads=True)
            var st = vk.create_buffer(self._staging_bytes(), local=False)
            if b.size == 0 or st.size == 0 or b.cpu_address == 0 or st.cpu_address == 0:
                return False
            _ = external_call["memset", Int](b.cpu_address, Int32(0), b.size)
            var ds = sets[f]
            vk.write_descriptor(ds, SOTA_VT_BIND_VERTS, buf_vertices)
            vk.write_descriptor(ds, SOTA_VT_BIND_INDICES, buf_indices)
            vk.write_descriptor(ds, SOTA_VT_BIND_MESHLETS, buf_meshlets)
            vk.write_descriptor(ds, SOTA_VT_BIND_INSTANCES, ring_instances[f])
            vk.write_descriptor(ds, SOTA_VT_BIND_MATERIALS, buf_materials)
            vk.write_descriptor(ds, SOTA_VT_BIND_GLOBALS, ring_globals[f])
            vk.write_descriptor(ds, SOTA_VT_BIND_FEEDBACK, b)
            vk.write_descriptor(ds, SOTA_VT_BIND_SPARE, b)
            vk.write_aaa_texture(ds, SOTA_VT_BIND_VIS, 0, sampler, vis_view)
            self.sets.append(ds)
            self.fb.append(b^)
            self.staging.append(st^)
            self.fb_stamp.append(-1)
            self.pending.append(List[VtPageUpload]())
            self.pending_ind.append(List[Int]())
        self.active = True
        return True

    def rebind_geometry(mut self, mut vk: VulkanGpu, buf_vertices: GpuBuffer, buf_indices: GpuBuffer, buf_meshlets: GpuBuffer):
        """The renderer grew its geometry buffers: point the feedback pass at the new ones."""
        for ds in self.sets:
            vk.write_descriptor(ds, SOTA_VT_BIND_VERTS, buf_vertices)
            vk.write_descriptor(ds, SOTA_VT_BIND_INDICES, buf_indices)
            vk.write_descriptor(ds, SOTA_VT_BIND_MESHLETS, buf_meshlets)

    def _register_bindless(
        mut self, mut vk: VulkanGpu, img: VulkanImage, sampler: Int, desc_sets: List[Int], mut bindless_images: List[VulkanImage],
        max_bindless: Int,
    ) -> Int:
        var slot = len(bindless_images)
        if slot >= max_bindless or slot > 255:
            return -1
        for ds in desc_sets:
            vk.write_aaa_texture(ds, BIND_AAA_BINDLESS_TEX, slot, sampler, img.view)
        bindless_images.append(img.copy())
        return slot

    def add_virtual_texture(
        mut self, mut vk: VulkanGpu, pixels: List[UInt32], dim: Int, desc_sets: List[Int],
        samp_nearest: Int, samp_linear: Int, mut bindless_images: List[VulkanImage], max_bindless: Int,
    ) -> Int:
        """Registers a square power-of-two `dim` (128..16384) ARGB sRGB image as a VT. Uploads its root page and
        page table synchronously (load time). Returns the VT index or -1."""
        if not self.active or len(self.mips) >= SOTA_VT_MAX_TEXTURES or len(pixels) < dim * dim:
            return -1
        var lp = 0
        while (SOTA_VT_PAGE << lp) < dim:
            lp += 1
        if (SOTA_VT_PAGE << lp) != dim or lp > SOTA_VT_MAX_LP:
            return -1
        if self.atlas_slot < 0:
            var at = vk.create_image_2d(
                SOTA_VT_ATLAS_DIM, SOTA_VT_ATLAS_DIM, SOTA_VT_FORMAT_BGRA8_SRGB, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED
            )
            if at.image == 0:
                return -1
            self.atlas_slot = self._register_bindless(vk, at, samp_linear, desc_sets, bindless_images, max_bindless)
            if self.atlas_slot < 0:
                vk.destroy_image(at)
                return -1
            self.atlas = at^
        var ind = vk.create_image_2d(2 << lp, 1 << lp, SOTA_VT_FORMAT_BGRA8_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED)
        if ind.image == 0:
            return -1
        var islot = self._register_bindless(vk, ind, samp_nearest, desc_sets, bindless_images, max_bindless)
        if islot < 0:
            vk.destroy_image(ind)
            return -1
        var vt = self.res.add_vt(lp)
        self.mips.append(vt_build_mips(pixels, dim))
        self.ind_slot.append(islot)
        self.ind_img.append(ind^)
        self.ind_dirty.append(False)
        # mean colour (linear) of the root page, for RT / GI hits that do not run the VT lookup
        var acc_r = Float64(0.0)
        var acc_g = Float64(0.0)
        var acc_b = Float64(0.0)
        for p in self.mips[vt][lp]:
            acc_r += Float64(_srgb8_to_lin(Int((p >> 16) & 255)))
            acc_g += Float64(_srgb8_to_lin(Int((p >> 8) & 255)))
            acc_b += Float64(_srgb8_to_lin(Int(p & 255)))
        var nr = Float64(len(self.mips[vt][lp]))
        self.mean_rgb.append((Float32(acc_r / nr), Float32(acc_g / nr), Float32(acc_b / nr)))
        # synchronous root page + page table
        var sl = self.res.place(vt, lp, 0, 0, 0, pinned=True)
        if sl < 0:
            return -1
        var ups = List[VtPageUpload]()
        ups.append(VtPageUpload(vt, lp, 0, 0, sl))
        var dirty = List[Int]()
        dirty.append(vt)
        self.pending[0].clear()
        self.pending_ind[0].clear()
        self._stage(0, ups, dirty)
        if not vk.begin_commands(0):
            return -1
        vk.transition_image(0, self.atlas, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        vk.transition_image(0, self.ind_img[vt], LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self._record_copies(vk, 0)
        vk.transition_image(0, self.atlas, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        vk.transition_image(0, self.ind_img[vt], LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = vk.submit(0)
        if f == 0 or not vk.wait(f):
            return -1
        self.pending[0].clear()
        self.pending_ind[0].clear()
        return vt

    def material_code(self, vt: Int) -> Float32:
        return vt_encode_material_code(self.ind_slot[vt], self.atlas_slot, self.res.vt_lp[vt], vt)

    def _stage(mut self, slot: Int, ups: List[VtPageUpload], dirty: List[Int]):
        """Writes page images then page tables into `staging[slot]` and records them as pending for that slot."""
        var base = self.staging[slot].cpu_address
        var n = min(len(ups), max(1, self.budget_pages))
        for i in range(n):
            var u = ups[i]
            vt_fill_page(self.mips[u.vt], self.res.vt_lp[u.vt], u.mip, u.px, u.py, base + i * SOTA_VT_PAGE_BYTES)
            self.pending[slot].append(u)
        var off = max(1, self.budget_pages) * SOTA_VT_PAGE_BYTES
        for vt in dirty:
            var tex = self.res.indirection_texels(vt)
            _ = external_call["memcpy", Int](base + off, Int(tex.unsafe_ptr()), len(tex) * 4)
            _ = len(tex)
            self.pending_ind[slot].append(vt)
            off += 2 * 128 * 128 * 4

    def _record_copies(mut self, mut vk: VulkanGpu, slot: Int):
        for i in range(len(self.pending[slot])):
            var u = self.pending[slot][i]
            var sx = (u.slot % SOTA_VT_ATLAS_SLOTS) * SOTA_VT_SLOT
            var sy = (u.slot // SOTA_VT_ATLAS_SLOTS) * SOTA_VT_SLOT
            vk.upload_image_subregion(slot, self.atlas, self.staging[slot], i * SOTA_VT_PAGE_BYTES, sx, sy, SOTA_VT_SLOT, SOTA_VT_SLOT)
        var off = max(1, self.budget_pages) * SOTA_VT_PAGE_BYTES
        for vt in self.pending_ind[slot]:
            var lp = self.res.vt_lp[vt]
            vk.upload_image_subregion(slot, self.ind_img[vt], self.staging[slot], off, 0, 0, 2 << lp, 1 << lp)
            off += 2 * 128 * 128 * 4

    def begin_frame(mut self, slot: Int, prev_slot: Int):
        """CPU: consume the previous frame's feedback (complete: render_frame retires it first), stream pages."""
        if not self.active or len(self.mips) == 0 or not self.enabled:
            return
        var t0 = get_time()
        self.pending[slot].clear()
        self.pending_ind[slot].clear()
        self.last_uploads = 0
        var stamp = self.fb_stamp[prev_slot] if prev_slot < len(self.fb_stamp) else -1
        if stamp < 0:
            return
        var nw = len(self.mips) * SOTA_VT_FB_STRIDE
        var words = List[UInt32](length=nw, fill=UInt32(0))
        _ = external_call["memcpy", Int](Int(words.unsafe_ptr()), self.fb[prev_slot].cpu_address, nw * 4)
        var ev0 = self.res.evictions_total
        var ups = self.res.process_feedback(words, stamp, self.budget_pages)
        _ = len(words)
        var dirty = List[Int]()
        if self.res.evictions_total != ev0:
            # an eviction may have unmapped a page of any VT: refresh every table (they are tiny)
            for v in range(len(self.mips)):
                dirty.append(v)
        else:
            for u in ups:
                var seen = False
                for d in dirty:
                    if d == u.vt:
                        seen = True
                if not seen:
                    dirty.append(u.vt)
        self._stage(slot, ups, dirty)
        self.last_uploads = len(self.pending[slot])
        self.last_cpu_ms = (get_time() - t0) * 1000.0

    def record_uploads(mut self, mut vk: VulkanGpu, slot: Int):
        """Top of the frame's command buffer: wait for earlier queue work (pages may be evicted), copy, release."""
        if len(self.pending[slot]) == 0 and len(self.pending_ind[slot]) == 0:
            return
        vk.memory_barrier2(slot, SOTA_VT_STAGE_ALL, Int(ACCESS_SHADER_READ), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE))
        vk.transition_image(slot, self.atlas, LAYOUT_TRANSFER_TARGET, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        for vt in self.pending_ind[slot]:
            vk.transition_image(slot, self.ind_img[vt], LAYOUT_TRANSFER_TARGET, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self._record_copies(vk, slot)
        vk.transition_image(slot, self.atlas, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        for vt in self.pending_ind[slot]:
            vk.transition_image(slot, self.ind_img[vt], LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

    def record_feedback(mut self, mut vk: VulkanGpu, slot: Int, render_w: Int, render_h: Int, stamp: Int):
        """After the visbuffer raster (visbuffer in SHADER_READ_ONLY)."""
        if not self.active or len(self.mips) == 0 or not self.enabled:
            return
        var k = stamp % 16
        var push = List[UInt32](length=SOTA_VT_PUSH_WORDS, fill=0)
        push[0] = UInt32(render_w)
        push[1] = UInt32(render_h)
        push[2] = UInt32(stamp)
        push[3] = UInt32(k % 4)
        push[4] = UInt32((k // 4 + k) % 4)
        var gw = (render_w + SOTA_VT_FB_STEP * 8 - 1) // (SOTA_VT_FB_STEP * 8)
        var gh = (render_h + SOTA_VT_FB_STEP * 8 - 1) // (SOTA_VT_FB_STEP * 8)
        vk.dispatch_aaa(slot, self.pipe_fb, self.layout, self.sets[slot], push, gw, gh)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
        self.fb_stamp[slot] = stamp

    def shutdown(mut self, mut vk: VulkanGpu):
        """Images belong to the renderer's bindless list (destroyed there)."""
        if self.pipe_fb != 0:
            vk.destroy_pipeline(self.pipe_fb)
        self.pipe_fb = 0
        for b in self.fb:
            vk.destroy_buffer(b)
        for b in self.staging:
            vk.destroy_buffer(b)
        self.fb = List[GpuBuffer]()
        self.staging = List[GpuBuffer]()
        if self.dsl != 0:
            vk.destroy_aaa_descriptors(self.pool, self.dsl, self.layout)
        self.pool = 0
        self.dsl = 0
        self.layout = 0
        self.active = False


def _srgb8_to_lin(v: Int) -> Float32:
    var c = Float32(v) / Float32(255.0)
    return c / Float32(12.92) if c <= Float32(0.04045) else ((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4)
