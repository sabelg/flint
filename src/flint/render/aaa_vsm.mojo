"""Milestone 2 (`V13`): `16,384 x 16,384` Demand-Paged Virtual Shadow Maps (`VSM`) + `PCSS`.

Architecture:
  - Virtual Resolution: `16,384 x 16,384` (`128 x 128` virtual pages of `128 x 128` texels).
  - Physical Page Pool: `2,048 x 2,048` `VK_FORMAT_D32_SFLOAT` (`16 x 16 = 256` physical pages).
  - Static Page Caching: `VsmPageManager` caches clean virtual pages across frames so static
    architecture/terrain shadows cost `0` draw calls after frame 0, rasterizing only newly
    uncovered or dirty pages via `VK_KHR_dynamic_rendering` per-tile `renderArea`.
  - Contact-Hardening `PCSS`: 8-tap Vogel-disk blocker search + 12-tap variable-penumbra PCF.
"""
from std.math import cos, sin, sqrt
from flint.core.math import Vec3
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_GL_COMPUTE,
    MODEL_VERTEX,
)
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_PUSH_WORDS,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_GLOBALS,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MESHLETS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    emit_mat4_mul_point,
)

comptime VSM_VIRTUAL_PAGES_DIM: Int = 128         # 128 x 128 = 16,384 virtual pages
comptime VSM_VIRTUAL_PAGE_COUNT: Int = 16384
comptime VSM_PAGE_TEXELS: Int = 128               # 128 x 128 texels/page -> 16,384 x 16,384 virtual shadow map!
comptime VSM_VIRTUAL_RES: Int = 16384
comptime VSM_PHYS_TILES_DIM: Int = 32             # 32 x 32 = 1,024 physical pages
comptime VSM_PHYS_TILE_COUNT: Int = 1024
comptime VSM_PHYS_ATLAS_RES: Int = 4096           # 32 * 128 = 4096 x 4096 physical D32_SFLOAT image
comptime VSM_TABLE_WORDS: Int = 32768             # [0..16383] = page entries, [16384..32767] = request flags


struct VsmPageAllocation(Copyable, Movable):
    var vx: Int
    var vy: Int
    var px: Int
    var py: Int

    def __init__(out self, vx: Int, vy: Int, px: Int, py: Int):
        self.vx = vx
        self.vy = vy
        self.px = px
        self.py = py


struct VsmPageManager(Movable):
    """Demand-paged `16k x 16k` Virtual Shadow Map allocator with cross-frame static tile caching."""
    var virtual_to_phys: List[Int]    # -1 if unmapped, else 0..255
    var virtual_valid: List[Bool]     # True if physical tile holds up-to-date depth
    var phys_to_virtual: List[Int]    # -1 if free, else 0..16383
    var phys_last_used: List[Int]
    var current_epoch: Int
    var cached_hits_last_frame: Int
    var newly_rendered_pages_last_frame: Int

    def __init__(out self):
        self.virtual_to_phys = List[Int](length=VSM_VIRTUAL_PAGE_COUNT, fill=-1)
        self.virtual_valid = List[Bool](length=VSM_VIRTUAL_PAGE_COUNT, fill=False)
        self.phys_to_virtual = List[Int](length=VSM_PHYS_TILE_COUNT, fill=-1)
        self.phys_last_used = List[Int](length=VSM_PHYS_TILE_COUNT, fill=0)
        self.current_epoch = 1
        self.cached_hits_last_frame = 0
        self.newly_rendered_pages_last_frame = 0

    def invalidate_all(mut self):
        for i in range(VSM_VIRTUAL_PAGE_COUNT):
            self.virtual_valid[i] = False

    def invalidate_region(mut self, min_uv_x: Float32, min_uv_y: Float32, max_uv_x: Float32, max_uv_y: Float32):
        var x0 = max(0, min(VSM_VIRTUAL_PAGES_DIM - 1, Int(min_uv_x * Float32(VSM_VIRTUAL_PAGES_DIM))))
        var y0 = max(0, min(VSM_VIRTUAL_PAGES_DIM - 1, Int(min_uv_y * Float32(VSM_VIRTUAL_PAGES_DIM))))
        var x1 = max(0, min(VSM_VIRTUAL_PAGES_DIM - 1, Int(max_uv_x * Float32(VSM_VIRTUAL_PAGES_DIM))))
        var y1 = max(0, min(VSM_VIRTUAL_PAGES_DIM - 1, Int(max_uv_y * Float32(VSM_VIRTUAL_PAGES_DIM))))
        for vy in range(y0, y1 + 1):
            for vx in range(x0, x1 + 1):
                self.virtual_valid[vy * VSM_VIRTUAL_PAGES_DIM + vx] = False

    def process_requests(
        mut self,
        mut table_words: List[UInt32],
    ) -> List[VsmPageAllocation]:
        """Consumes GPU request flags in `table_words[16384..32767]`, preserves cached valid pages,
        and returns only the newly allocated / invalidated pages that need hardware depth rasterization!"""
        self.current_epoch += 1
        self.cached_hits_last_frame = 0
        var to_render = List[VsmPageAllocation]()

        for v_idx in range(VSM_VIRTUAL_PAGE_COUNT):
            var req = table_words[VSM_VIRTUAL_PAGE_COUNT + v_idx]
            table_words[VSM_VIRTUAL_PAGE_COUNT + v_idx] = 0
            if req == 0:
                continue

            var vx = v_idx & 127
            var vy = v_idx >> 7
            var existing_p = self.virtual_to_phys[v_idx]
            if existing_p >= 0 and self.virtual_valid[v_idx]:
                self.phys_last_used[existing_p] = self.current_epoch
                self.cached_hits_last_frame += 1
                var px = existing_p & 31
                var py = existing_p >> 5
                table_words[v_idx] = UInt32(0x10000 | (py << 8) | px)
                continue

            var chosen_p = existing_p
            if chosen_p < 0:
                # Find free physical tile or LRU tile not touched this frame
                var oldest_ep = self.current_epoch
                var best_slot = -1
                for pi in range(VSM_PHYS_TILE_COUNT):
                    if self.phys_to_virtual[pi] < 0:
                        best_slot = pi
                        break
                    if self.phys_last_used[pi] < oldest_ep:
                        oldest_ep = self.phys_last_used[pi]
                        best_slot = pi
                if best_slot < 0:
                    table_words[v_idx] = 0
                    continue
                chosen_p = best_slot
                var evicted_v = self.phys_to_virtual[chosen_p]
                if evicted_v >= 0:
                    self.virtual_to_phys[evicted_v] = -1
                    self.virtual_valid[evicted_v] = False
                    table_words[evicted_v] = 0
                self.phys_to_virtual[chosen_p] = v_idx
                self.virtual_to_phys[v_idx] = chosen_p

            self.virtual_valid[v_idx] = True
            self.phys_last_used[chosen_p] = self.current_epoch
            var px = chosen_p & 31
            var py = chosen_p >> 5
            table_words[v_idx] = UInt32(0x10000 | (py << 8) | px)
            to_render.append(VsmPageAllocation(vx, vy, px, py))

        self.newly_rendered_pages_last_frame = len(to_render)
        return to_render^


def make_ortho_directional_light_vp(
    light_dir_toward_sun: Vec3,
    scene_center: Vec3,
    ortho_half_extent: Float32,
    depth_half_range: Float32 = 120.0,
) -> List[Float32]:
    """Builds a column-major `[0..1]` Z orthographic `light_view_proj` matrix for `16k` VSM."""
    var fwd = (Vec3(0.0, 0.0, 0.0) - light_dir_toward_sun).normalized()  # light propagation direction
    var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
    var right = up_ref.cross(fwd).normalized()
    var up = fwd.cross(right).normalized()
    var eye = scene_center - fwd * depth_half_range

    var sx = Float32(1.0) / max(Float32(0.01), ortho_half_extent)
    var sy = Float32(1.0) / max(Float32(0.01), ortho_half_extent)
    var sz = Float32(1.0) / max(Float32(0.01), depth_half_range * Float32(2.0))

    var tx = -right.dot(eye) * sx
    var ty = -up.dot(eye) * sy
    var tz = -fwd.dot(eye) * sz

    var m = List[Float32](length=16, fill=0.0)
    # Column 0
    m[0] = right.x * sx; m[1] = up.x * sy; m[2] = fwd.x * sz; m[3] = Float32(0.0)
    # Column 1
    m[4] = right.y * sx; m[5] = up.y * sy; m[6] = fwd.y * sz; m[7] = Float32(0.0)
    # Column 2
    m[8] = right.z * sx; m[9] = up.z * sy; m[10] = fwd.z * sz; m[11] = Float32(0.0)
    # Column 3
    m[12] = tx;          m[13] = ty;       m[14] = tz;         m[15] = Float32(1.0)
    return m^


def vsm_mark_pages_shader() -> List[UInt32]:
    """Compute Pass: Reads Reverse-Z screen depth, reconstructs world position, projects into
    `16,384 x 16,384` Virtual Shadow Map space, and marks visible `(vx, vy)` pages (+1 border)!
    Push[0] = width, Push[1] = height.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var vis_samp = s.sampled_image_2d(BIND_AAA_VIS_TEX, is_uint=True)
    var depth_samp = s.sampled_image_2d(BIND_AAA_DEPTH_TEX)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    # Sample every 2x2 pixel quad center for fast page marking
    var coord = s.uvec2(px, py)
    var vis_id = s.extract_u(s.texel_fetch_u2d(vis_samp, coord, s.uint(0)), 0)
    s.return_if(s.ieq(vis_id, s.uint(0)))

    var rev_z = s.extract_f(s.texel_fetch_f2d(depth_samp, coord, s.uint(0)), 0)
    s.return_if(s.flt(rev_z, s.float(1e-6)))

    var near_z = s.load_float(b_globals, s.uint(35))
    var z_view = s.fdiv(near_z, rev_z)
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var ndc_x = s.fsub(s.fmul(s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh), s.float(2.0)))

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

    var vx_s = s.fmul(s.fmul(ndc_x, tan_x), z_view)
    var vy_s = s.fmul(s.fmul(ndc_y, tan_y), z_view)
    var wx = s.fma(rgt_x, vx_s, s.fma(up_x, vy_s, s.fma(fwd_x, z_view, cam_x)))
    var wy = s.fma(rgt_y, vx_s, s.fma(up_y, vy_s, s.fma(fwd_y, z_view, cam_y)))
    var wz = s.fma(rgt_z, vx_s, s.fma(up_z, vy_s, s.fma(fwd_z, z_view, cam_z)))

    var lclip = emit_mat4_mul_point(s, b_globals, s.uint(56), wx, wy, wz)
    var lu = s.fma(lclip[0], s.float(0.5), s.float(0.5))
    var lv = s.fma(s.fneg(lclip[1]), s.float(0.5), s.float(0.5))
    var in_bounds = s.logical_and(
        s.logical_and(s.fge(lu, s.float(0.0)), s.flt(lu, s.float(1.0))),
        s.logical_and(s.fge(lv, s.float(0.0)), s.flt(lv, s.float(1.0))),
    )
    s.begin_if(in_bounds)
    var pvx = s.umin(s.uint(127), s.saturate_uint(s.fmul(lu, s.float(128.0))))
    var pvy = s.umin(s.uint(127), s.saturate_uint(s.fmul(lv, s.float(128.0))))
    var req_idx = s.iadd(s.uint(16384), s.iadd(s.imul(pvy, s.uint(128)), pvx))
    s.store_word(b_vsm, req_idx, s.uint(1))
    # Also mark immediate neighbor pages for wide PCSS penumbra kernel continuity
    var pvx_l = s.select_u(s.ieq(pvx, s.uint(0)), s.uint(0), s.isub(pvx, s.uint(1)))
    var pvx_r = s.umin(s.uint(127), s.iadd(pvx, s.uint(1)))
    var pvy_u = s.select_u(s.ieq(pvy, s.uint(0)), s.uint(0), s.isub(pvy, s.uint(1)))
    var pvy_d = s.umin(s.uint(127), s.iadd(pvy, s.uint(1)))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy, s.uint(128)), pvx_l)), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy, s.uint(128)), pvx_r)), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy_u, s.uint(128)), pvx)), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy_d, s.uint(128)), pvx)), s.uint(1))
    s.end_if()

    return s.finish()


def vsm_page_vertex_shader() -> List[UInt32]:
    """Depth-only Hardware Vertex Shader rasterizing shadow geometry into a `128x128` VSM physical tile!
    Push[0] = scale_x (128.0), Push[1] = scale_y (128.0), Push[2] = bias_x, Push[3] = bias_y.
    """
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)

    var v_idx = s.vertex_index()
    var packed_inst = s.instance_index()
    var inst_id = s.shr(packed_inst, s.uint(16))
    var ml_id = s.bit_and(packed_inst, s.uint(0xFFFF))

    var local_prim = s.udiv(v_idx, s.uint(3))
    var corner = s.umod(v_idx, s.uint(3))
    var ml_base = s.imul(ml_id, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var global_tri = s.iadd(first_tri, local_prim)
    var real_vtx = s.load_word(b_indices, s.iadd(s.imul(global_tri, s.uint(3)), corner))
    var v_base = s.imul(real_vtx, s.uint(12))

    var lpx = s.load_float(b_verts, v_base)
    var lpy = s.load_float(b_verts, s.iadd(v_base, s.uint(1)))
    var lpz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var wp = emit_mat4_mul_point(s, b_instances, inst_base, lpx, lpy, lpz)
    var lclip = emit_mat4_mul_point(s, b_globals, s.uint(56), wp[0], wp[1], wp[2])

    # Zoom virtual page `(vx, vy)` into clip `[-1 .. +1]^2`
    var sc_x = s.push_float(0)
    var sc_y = s.push_float(1)
    var bi_x = s.push_float(2)
    var bi_y = s.push_float(3)
    var page_cx = s.fma(lclip[0], sc_x, bi_x)
    var page_cy = s.fma(lclip[1], sc_y, bi_y)
    var page_cz = s.nclamp(lclip[2], s.float(0.0), s.float(1.0))
    s.write_clip_position(page_cx, page_cy, page_cz, s.float(1.0))

    return s.finish()


def emit_sample_vsm_depth(
    mut s: AaaSpirvModule,
    b_vsm: UInt32,
    vsm_samp: Tuple[UInt32, UInt32],
    lu: UInt32,
    lv: UInt32,
) -> UInt32:
    """Translates virtual `(lu, lv) in [0..1]^2` through the `128x128` VSM Page Table into the `2048x2048` Physical Atlas."""
    var u_cl = s.nclamp(lu, s.float(0.00005), s.float(0.99995))
    var v_cl = s.nclamp(lv, s.float(0.00005), s.float(0.99995))
    var sx = s.fmul(u_cl, s.float(128.0))
    var sy = s.fmul(v_cl, s.float(128.0))
    var vx = s.umin(s.uint(127), s.saturate_uint(sx))
    var vy = s.umin(s.uint(127), s.saturate_uint(sy))
    var entry = s.load_word(b_vsm, s.iadd(s.imul(vy, s.uint(128)), vx))

    var out_z = s.fvar(s.float(1.0))
    s.begin_if(s.ine(s.bit_and(entry, s.uint(0x10000)), s.uint(0)))
    var px = s.float_from_uint(s.bit_and(entry, s.uint(0xFF)))
    var py = s.float_from_uint(s.bit_and(s.shr(entry, s.uint(8)), s.uint(0xFF)))
    # Clamp intra-page fractional UV by half a texel so bilinear/point never bleeds across neighbor physical tiles!
    var fu = s.nclamp(s.fract(sx), s.float(0.004), s.float(0.996))
    var fv = s.nclamp(s.fract(sy), s.float(0.004), s.float(0.996))
    var phys_u = s.fmul(s.fadd(px, fu), s.float(0.03125))  # / 32.0
    var phys_v = s.fmul(s.fadd(py, fv), s.float(0.03125))
    var d_val = s.extract_f(s.sample_2d_lod(vsm_samp, s.vec2(phys_u, phys_v), s.float(0.0)), 0)
    s.store(out_z, d_val)
    s.end_if()
    return s.load_f(out_z)


def emit_vsm_pcss_shadow(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    b_vsm: UInt32,
    vsm_samp: Tuple[UInt32, UInt32],
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    sun_radius_uv: UInt32,
) -> UInt32:
    """Blocker-Search `PCSS` over `16,384 x 16,384` Virtual Shadow Map!
     Sharp `1/16384` contact shadows at `d_receiver -> d_blocker`, physically softening with occluder distance.
    """
    var bx = s.fma(nx, s.float(0.012), wx)
    var by = s.fma(ny, s.float(0.012), wy)
    var bz = s.fma(nz, s.float(0.012), wz)
    var lclip = emit_mat4_mul_point(s, b_globals, s.uint(56), bx, by, bz)
    var lu = s.fma(lclip[0], s.float(0.5), s.float(0.5))
    var lv = s.fma(s.fneg(lclip[1]), s.float(0.5), s.float(0.5))
    var z_rec = lclip[2]

    var shadow_vis = s.fvar(s.float(1.0))
    var in_map = s.logical_and(
        s.logical_and(s.fgt(lu, s.float(0.002)), s.flt(lu, s.float(0.998))),
        s.logical_and(s.fgt(lv, s.float(0.002)), s.flt(lv, s.float(0.998))),
    )
    s.begin_if(s.logical_and(in_map, s.flt(z_rec, s.float(0.999))))
    var z_bias = s.float(0.0015)

    # 1. 8-Tap Vogel-Disk Blocker Search
    var blk_sum = s.fvar(s.float(0.0))
    var blk_cnt = s.fvar(s.float(0.0))
    var search_r = s.nclamp(sun_radius_uv, s.float(0.00025), s.float(0.012))

    for tap in range(8):
        var r_norm = sqrt((Float32(tap) + Float32(0.5)) / Float32(8.0))
        var theta = Float32(tap) * Float32(2.39996323)
        var ox = s.fmul(search_r, s.float(r_norm * cos(theta)))
        var oy = s.fmul(search_r, s.float(r_norm * sin(theta)))
        var dz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, s.fadd(lu, ox), s.fadd(lv, oy))
        s.begin_if(s.flt(dz, s.fsub(z_rec, z_bias)))
        s.store(blk_sum, s.fadd(s.load_f(blk_sum), dz))
        s.store(blk_cnt, s.fadd(s.load_f(blk_cnt), s.float(1.0)))
        s.end_if()

    var bcnt = s.load_f(blk_cnt)
    s.begin_if(s.fgt(bcnt, s.float(0.5)))
    var avg_blk = s.fdiv(s.load_f(blk_sum), bcnt)
    var penumbra_ratio = s.fdiv(s.nmax(s.float(0.0), s.fsub(z_rec, avg_blk)), s.nmax(s.float(0.04), avg_blk))
    # Minimum 1.25 virtual texels (1.25 / 16384 = 0.000076) for crisp contact, scaling up with blocker distance!
    var filter_r = s.nclamp(s.fmul(penumbra_ratio, search_r), s.float(0.000076), s.float(0.010))

    # 2. 12-Tap Golden-Angle Vogel-Disk PCF
    var lit_sum = s.fvar(s.float(0.0))
    for p_tap in range(12):
        var pr = sqrt((Float32(p_tap) + Float32(0.5)) / Float32(12.0))
        var pth = Float32(p_tap) * Float32(2.39996323) + Float32(0.62)
        var pox = s.fmul(filter_r, s.float(pr * cos(pth)))
        var poy = s.fmul(filter_r, s.float(pr * sin(pth)))
        var pz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, s.fadd(lu, pox), s.fadd(lv, poy))
        var is_lit = s.select_f(s.flt(pz, s.fsub(z_rec, z_bias)), s.float(0.0), s.float(1.0))
        s.store(lit_sum, s.fadd(s.load_f(lit_sum), is_lit))

    s.store(shadow_vis, s.fmul(s.load_f(lit_sum), s.float(1.0 / 12.0)))
    s.end_if()
    s.end_if()

    return s.load_f(shadow_vis)
