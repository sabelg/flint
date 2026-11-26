"""Milestone 2 (`V13`): `16,384 x 16,384` Demand-Paged Virtual Shadow Maps (`VSM`) + `PCSS`.

Architecture:
  - Virtual Resolution: `16,384 x 16,384` (`128 x 128` virtual pages of `128 x 128` texels).
  - Physical Page Pool: `2,048 x 2,048` `VK_FORMAT_D32_SFLOAT` (`16 x 16 = 256` physical pages).
  - Static Page Caching: `VsmPageManager` caches clean virtual pages across frames so static
    architecture/terrain shadows cost `0` draw calls after frame 0, rasterizing only newly
    uncovered or dirty pages via `VK_KHR_dynamic_rendering` per-tile `renderArea`.
  - Contact-Hardening `PCSS`: 8-tap Vogel-disk blocker search + 12-tap variable-penumbra PCF.
"""
from std.math import abs, cos, floor, max, min, sin, sqrt
from flint.core.math import Vec3
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
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_GLOBALS,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    emit_mat4_mul_point,
    emit_instance_point,
)
from flint.render.sota_geometry_visid import emit_sota_draw_decode, emit_sota_vis_payload

comptime VSM_VIRTUAL_PAGES_DIM: Int = 128         # 128 x 128 = 16,384 virtual pages
comptime VSM_VIRTUAL_PAGE_COUNT: Int = 16384
comptime VSM_PAGE_TEXELS: Int = 128               # 128 x 128 texels/page -> 16,384 x 16,384 virtual shadow map!
comptime VSM_VIRTUAL_RES: Int = 16384
comptime VSM_PHYS_TILES_DIM: Int = 32             # 32 x 32 = 1,024 physical pages
comptime VSM_PHYS_TILE_COUNT: Int = 1024
comptime VSM_PHYS_ATLAS_RES: Int = 4096           # 32 * 128 = 4096 x 4096 physical D32_SFLOAT image
comptime VSM_TABLE_WORDS: Int = 32768             # [0..16383] = page entries, [16384..32767] = request flags

comptime VSM_CASCADE_COUNT: Int = 4
comptime VSM_CASCADE_PAGES_DIM: Int = 64
comptime VSM_CASCADE_PAGE_COUNT: Int = 4096       # 64 x 64
comptime VSM_TOTAL_CLIPMAP_PAGES: Int = 16384     # 4 * 4096 = exactly VSM_VIRTUAL_PAGE_COUNT!

# Half-extents for the 4 concentric cascades (±16m, ±64m, ±256m, ±1024m):
comptime VSM_CASCADE_0_EXTENT: Float32 = 16.0     # Near fine detail (±16m)
comptime VSM_CASCADE_1_EXTENT: Float32 = 64.0     # Mid-range (±64m)
comptime VSM_CASCADE_2_EXTENT: Float32 = 256.0    # Far environment (±256m)
comptime VSM_CASCADE_3_EXTENT: Float32 = 1024.0   # Horizon / mountains (±1024m)

# `b_globals` word offsets for 4-Cascade VSM (`0..166` untouched; Cascade 0 stays at `56..71`):
comptime VSM_GLOBALS_CASCADE_FLAG_WORD: Int = 167
comptime VSM_GLOBALS_CASCADE_1_WORD: Int = 168    # 16 floats: [168..183]
comptime VSM_GLOBALS_CASCADE_2_WORD: Int = 184    # 16 floats: [184..199]
comptime VSM_GLOBALS_CASCADE_3_WORD: Int = 200    # 16 floats: [200..215]


@always_inline
def vsm_cascade_quadrant_origin(cascade: Int) -> Tuple[Int, Int]:
    """Returns `(quad_x, quad_y)` (`0` or `64`) for cascade `0..3` inside the `128x128` VSM Page Table."""
    var c = max(0, min(VSM_CASCADE_COUNT - 1, cascade))
    return ((c & 1) * VSM_CASCADE_PAGES_DIM, (c >> 1) * VSM_CASCADE_PAGES_DIM)


@always_inline
def vsm_cascade_from_virtual_page(vx: Int, vy: Int) -> Int:
    """Returns cascade `0..3` owning virtual page `(vx, vy)` in `0..127`."""
    var qx = 1 if vx >= VSM_CASCADE_PAGES_DIM else 0
    var qy = 1 if vy >= VSM_CASCADE_PAGES_DIM else 0
    return qy * 2 + qx


@always_inline
def vsm_cascade_globals_word_offset(cascade: Int) -> Int:
    """Returns the `b_globals` float/word offset of cascade `0..3`'s `light_view_proj` matrix."""
    if cascade <= 0:
        return 56
    if cascade == 1:
        return VSM_GLOBALS_CASCADE_1_WORD
    if cascade == 2:
        return VSM_GLOBALS_CASCADE_2_WORD
    return VSM_GLOBALS_CASCADE_3_WORD


def get_vsm_cascade_index(dist: Float32) -> Int:
    """Selects clipmap cascade 0..3 based on receiver distance from camera."""
    if dist <= VSM_CASCADE_0_EXTENT:
        return 0
    elif dist <= VSM_CASCADE_1_EXTENT:
        return 1
    elif dist <= VSM_CASCADE_2_EXTENT:
        return 2
    return 3


def compute_toroidal_page_index(cascade: Int, vx: Int, vy: Int) -> Int:
    """Maps toroidal `(vx mod 64, vy mod 64)` into linear cascade slot `cascade * 4096 + ty * 64 + tx`."""
    var c = max(0, min(VSM_CASCADE_COUNT - 1, cascade))
    var tx = (vx % VSM_CASCADE_PAGES_DIM + VSM_CASCADE_PAGES_DIM) % VSM_CASCADE_PAGES_DIM
    var ty = (vy % VSM_CASCADE_PAGES_DIM + VSM_CASCADE_PAGES_DIM) % VSM_CASCADE_PAGES_DIM
    return c * VSM_CASCADE_PAGE_COUNT + ty * VSM_CASCADE_PAGES_DIM + tx


def compute_quadrant_page_index(cascade: Int, vx: Int, vy: Int) -> Int:
    """Maps toroidal `(vx mod 64, vy mod 64)` into cascade `cascade`'s `64x64` quadrant of the `128x128` VSM Page Table!"""
    var tx = (vx % VSM_CASCADE_PAGES_DIM + VSM_CASCADE_PAGES_DIM) % VSM_CASCADE_PAGES_DIM
    var ty = (vy % VSM_CASCADE_PAGES_DIM + VSM_CASCADE_PAGES_DIM) % VSM_CASCADE_PAGES_DIM
    var q = vsm_cascade_quadrant_origin(cascade)
    return (q[1] + ty) * VSM_VIRTUAL_PAGES_DIM + (q[0] + tx)



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

    def invalidate_cascade(mut self, cascade: Int):
        """Invalidates only cascade `cascade`'s `64x64` quadrant inside the `128x128` VSM Page Table."""
        var q = vsm_cascade_quadrant_origin(cascade)
        for ly in range(VSM_CASCADE_PAGES_DIM):
            var row = (q[1] + ly) * VSM_VIRTUAL_PAGES_DIM + q[0]
            for lx in range(VSM_CASCADE_PAGES_DIM):
                self.virtual_valid[row + lx] = False

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
        # Once a scan finds no free / evictable tile, every tile has been touched this epoch and stays so for the
        # rest of this call, so later scans would fail too: skip them (identical output, O(1) per extra request
        # instead of O(VSM_PHYS_TILE_COUNT) when more pages are requested than the physical atlas holds).
        var phys_exhausted = False

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
                if phys_exhausted:
                    table_words[v_idx] = 0
                    continue
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
                    phys_exhausted = True
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


def compute_vsm_cascade_page_origins(
    light_dir_toward_sun: Vec3,
    camera_pos: Vec3,
    base_half_extent: Float32 = 16.0,
) -> List[Tuple[Int, Int]]:
    """Computes integer light-space page origins `(page_x, page_y)` for each of the 4 toroidal VSM cascades
    (`±16m, ±64m, ±256m, ±1024m`) so camera translation snaps cleanly to `128x128`-texel page boundaries!"""
    var fwd = (Vec3(0.0, 0.0, 0.0) - light_dir_toward_sun).normalized()
    var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
    var right = up_ref.cross(fwd).normalized()
    var up = fwd.cross(right).normalized()
    var cam_lx = right.dot(camera_pos)
    var cam_ly = up.dot(camera_pos)

    var origins = List[Tuple[Int, Int]](capacity=VSM_CASCADE_COUNT)
    for c in range(VSM_CASCADE_COUNT):
        var half_ext = max(Float32(1.0), base_half_extent) * Float32(1 << (2 * c))
        var page_world = (Float32(2.0) * half_ext) / Float32(VSM_CASCADE_PAGES_DIM)
        var px = Int(floor(cam_lx / page_world + Float32(0.5)))
        var py = Int(floor(cam_ly / page_world + Float32(0.5)))
        origins.append((px, py))
    return origins^


def compute_vsm_cascade_matrices(
    light_dir_toward_sun: Vec3,
    camera_pos: Vec3,
    base_half_extent: Float32 = 16.0,
    min_depth_half_range: Float32 = 160.0,
) -> List[List[Float32]]:
    """Builds 4 concentric page-snapped column-major `[0..1]` Z orthographic `light_view_proj` matrices
    covering `±16m, ±64m, ±256m, ±1024m` (`base_half_extent * 4^c`) around `camera_pos`."""
    var fwd = (Vec3(0.0, 0.0, 0.0) - light_dir_toward_sun).normalized()
    var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
    var right = up_ref.cross(fwd).normalized()
    var up = fwd.cross(right).normalized()
    var cam_lx = right.dot(camera_pos)
    var cam_ly = up.dot(camera_pos)
    var cam_lz = fwd.dot(camera_pos)
    var snap_lz = floor(cam_lz / Float32(32.0) + Float32(0.5)) * Float32(32.0)

    var mats = List[List[Float32]](capacity=VSM_CASCADE_COUNT)
    for c in range(VSM_CASCADE_COUNT):
        var half_ext = max(Float32(1.0), base_half_extent) * Float32(1 << (2 * c))
        var page_world = (Float32(2.0) * half_ext) / Float32(VSM_CASCADE_PAGES_DIM)
        var snap_lx = floor(cam_lx / page_world + Float32(0.5)) * page_world
        var snap_ly = floor(cam_ly / page_world + Float32(0.5)) * page_world
        var snapped_center = right * snap_lx + up * snap_ly + fwd * snap_lz
        var depth_half = max(min_depth_half_range, half_ext * Float32(1.5))
        mats.append(make_ortho_directional_light_vp(light_dir_toward_sun, snapped_center, half_ext, depth_half))
    return mats^


def mark_toroidal_cascade_pages(
    mut vsm: VsmPageManager,
    mut table_words: List[UInt32],
    prev_origins: List[Tuple[Int, Int]],
    new_origins: List[Tuple[Int, Int]],
    active_radius_pages: Int = 2,
) -> Int:
    """Invalidates only newly scrolled toroidal ring strips on camera translation and marks active
    toroidal clipmap pages in `table_words[16384..32767]`, returning the number of marked pages."""
    if len(table_words) < VSM_TABLE_WORDS or len(new_origins) < VSM_CASCADE_COUNT:
        return 0
    var half_dim = VSM_CASCADE_PAGES_DIM // 2
    var has_prev = len(prev_origins) >= VSM_CASCADE_COUNT
    var marked = 0
    var rad = max(0, min(half_dim - 1, active_radius_pages))

    for c in range(VSM_CASCADE_COUNT):
        var nx = new_origins[c][0]
        var ny = new_origins[c][1]
        if has_prev:
            var ox = prev_origins[c][0]
            var oy = prev_origins[c][1]
            var dx = nx - ox
            var dy = ny - oy
            if abs(dx) >= VSM_CASCADE_PAGES_DIM or abs(dy) >= VSM_CASCADE_PAGES_DIM:
                vsm.invalidate_cascade(c)
            else:
                if dx > 0:
                    for gx in range(ox + half_dim, nx + half_dim):
                        for gy in range(ny - half_dim, ny + half_dim):
                            vsm.virtual_valid[compute_quadrant_page_index(c, gx, gy)] = False
                elif dx < 0:
                    for gx in range(nx - half_dim, ox - half_dim):
                        for gy in range(ny - half_dim, ny + half_dim):
                            vsm.virtual_valid[compute_quadrant_page_index(c, gx, gy)] = False
                if dy > 0:
                    for gy in range(oy + half_dim, ny + half_dim):
                        for gx in range(nx - half_dim, nx + half_dim):
                            vsm.virtual_valid[compute_quadrant_page_index(c, gx, gy)] = False
                elif dy < 0:
                    for gy in range(ny - half_dim, oy - half_dim):
                        for gx in range(nx - half_dim, nx + half_dim):
                            vsm.virtual_valid[compute_quadrant_page_index(c, gx, gy)] = False

        for oy_loc in range(-rad, rad + 1):
            for ox_loc in range(-rad, rad + 1):
                var t_idx = compute_quadrant_page_index(c, nx + ox_loc, ny + oy_loc)
                table_words[VSM_VIRTUAL_PAGE_COUNT + t_idx] = 1
                marked += 1

    return marked


def _emit_cascade_fits(
    mut s: AaaSpirvModule,
    lc: Tuple[UInt32, UInt32, UInt32, UInt32],
    xy_limit: Float32 = 0.95,
) -> UInt32:
    """Returns `Bool` SSA true iff orthographic light-clip point `lc` lies inside `[-xy_limit..+xy_limit]^2 x [0.001..0.999]`."""
    var in_xy = s.logical_and(
        s.flt(s.fabs(lc[0]), s.float(xy_limit)),
        s.flt(s.fabs(lc[1]), s.float(xy_limit)),
    )
    var in_z = s.logical_and(
        s.fgt(lc[2], s.float(0.001)),
        s.flt(lc[2], s.float(0.999)),
    )
    return s.logical_and(in_xy, in_z)


def _emit_select_vsm_cascade(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Projects `(wx, wy, wz)` across Cascades `0..3` (`±16m, ±64m, ±256m, ±1024m`) and selects the finest fitting cascade!
    Returns `(valid_bool, casc_u32, mat_off_u32, lc_x, lc_y, lc_z, n_bias_f32)`.
    """
    var c0 = emit_mat4_mul_point(s, b_globals, s.uint(56), wx, wy, wz)
    var c1 = emit_mat4_mul_point(s, b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_1_WORD)), wx, wy, wz)
    var c2 = emit_mat4_mul_point(s, b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_2_WORD)), wx, wy, wz)
    var c3 = emit_mat4_mul_point(s, b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_3_WORD)), wx, wy, wz)

    var fit0 = _emit_cascade_fits(s, c0, 0.95)
    var fit1 = _emit_cascade_fits(s, c1, 0.95)
    var fit2 = _emit_cascade_fits(s, c2, 0.95)
    var fit3 = _emit_cascade_fits(s, c3, 0.99)

    var casc = s.select_u(
        fit0,
        s.uint(0),
        s.select_u(
            fit1,
            s.uint(1),
            s.select_u(fit2, s.uint(2), s.uint(3)),
        ),
    )
    var mat_off = s.select_u(
        fit0,
        s.uint(56),
        s.select_u(
            fit1,
            s.uint(UInt32(VSM_GLOBALS_CASCADE_1_WORD)),
            s.select_u(
                fit2,
                s.uint(UInt32(VSM_GLOBALS_CASCADE_2_WORD)),
                s.uint(UInt32(VSM_GLOBALS_CASCADE_3_WORD)),
            ),
        ),
    )
    var lcx = s.select_f(fit0, c0[0], s.select_f(fit1, c1[0], s.select_f(fit2, c2[0], c3[0])))
    var lcy = s.select_f(fit0, c0[1], s.select_f(fit1, c1[1], s.select_f(fit2, c2[1], c3[1])))
    var lcz = s.select_f(fit0, c0[2], s.select_f(fit1, c1[2], s.select_f(fit2, c2[2], c3[2])))
    var n_bias = s.select_f(
        fit0,
        s.float(0.012),
        s.select_f(
            fit1,
            s.float(0.036),
            s.select_f(fit2, s.float(0.11), s.float(0.35)),
        ),
    )
    var any_fit = s.logical_or(s.logical_or(fit0, fit1), s.logical_or(fit2, fit3))
    return (any_fit, casc, mat_off, lcx, lcy, lcz, n_bias)


def vsm_mark_pages_shader() -> List[UInt32]:
    """Compute Pass: Reads Reverse-Z screen depth, reconstructs world position, projects into
    `16,384 x 16,384` Virtual Shadow Map space (4-Cascade `64x64` Quadrants `±16m..±1024m` when `b_globals[167] != 0`,
    or single-cascade `128x128`), and marks visible `(vx, vy)` pages (+1 quadrant-clamped border)!
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

    var casc_flag = s.load_word(b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_FLAG_WORD)))
    var is_casc = s.ine(casc_flag, s.uint(0))

    s.begin_if(is_casc)
    var sel = _emit_select_vsm_cascade(s, b_globals, wx, wy, wz)
    s.begin_if(sel[0])
    var casc = sel[1]
    var lu_c = s.fma(sel[3], s.float(0.5), s.float(0.5))
    var lv_c = s.fma(s.fneg(sel[4]), s.float(0.5), s.float(0.5))
    var qx = s.imul(s.bit_and(casc, s.uint(1)), s.uint(64))
    var qy = s.imul(s.shr(casc, s.uint(1)), s.uint(64))
    var lx_u = s.umin(s.uint(63), s.saturate_uint(s.fmul(lu_c, s.float(64.0))))
    var ly_u = s.umin(s.uint(63), s.saturate_uint(s.fmul(lv_c, s.float(64.0))))
    var pvx_c = s.iadd(qx, lx_u)
    var pvy_c = s.iadd(qy, ly_u)
    var lx_l = s.select_u(s.ieq(lx_u, s.uint(0)), s.uint(0), s.isub(lx_u, s.uint(1)))
    var lx_r = s.umin(s.uint(63), s.iadd(lx_u, s.uint(1)))
    var ly_u0 = s.select_u(s.ieq(ly_u, s.uint(0)), s.uint(0), s.isub(ly_u, s.uint(1)))
    var ly_d0 = s.umin(s.uint(63), s.iadd(ly_u, s.uint(1)))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy_c, s.uint(128)), pvx_c)), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy_c, s.uint(128)), s.iadd(qx, lx_l))), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(pvy_c, s.uint(128)), s.iadd(qx, lx_r))), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(s.iadd(qy, ly_u0), s.uint(128)), pvx_c)), s.uint(1))
    s.store_word(b_vsm, s.iadd(s.uint(16384), s.iadd(s.imul(s.iadd(qy, ly_d0), s.uint(128)), pvx_c)), s.uint(1))
    s.end_if()
    s.end_if()

    s.begin_if(s.ieq(casc_flag, s.uint(0)))
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

    # --- sota:geometry hook --- 24-bit meshlet id (sota_geometry_visid): hi bits ride in vertexOffset / firstVertex
    var draw_ids = emit_sota_draw_decode(s, s.vertex_index(), s.instance_index())
    var v_idx = draw_ids[0]
    var inst_id = draw_ids[1]
    var ml_id = draw_ids[2]

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
    var wp = emit_instance_point(s, b_instances, b_globals, b_verts, inst_base, v_base, lpx, lpy, lpz)
    var raw_mat_off = s.push_word(4)
    var mat_off = s.select_u(s.ieq(raw_mat_off, s.uint(0)), s.uint(56), raw_mat_off)
    var lclip = emit_mat4_mul_point(s, b_globals, mat_off, wp[0], wp[1], wp[2])

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


def vsm_alphamask_vertex_shader() -> List[UInt32]:
    """Two-Bucket Alpha-Masked VSM Page Vertex Shader (`128x128` tile zoom + `uv` & `t_alb` forwarding)."""
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)

    var out_uv = s.add_stage_io(STORAGE_OUTPUT, s.type_vec2, location=0, flat=False)
    var out_talb = s.add_stage_io(STORAGE_OUTPUT, s.type_uint, location=1, flat=True)

    # --- sota:geometry hook --- 24-bit meshlet id (sota_geometry_visid): hi bits ride in vertexOffset / firstVertex
    var draw_ids = emit_sota_draw_decode(s, s.vertex_index(), s.instance_index())
    var v_idx = draw_ids[0]
    var inst_id = draw_ids[1]
    var ml_id = draw_ids[2]

    var local_prim = s.udiv(v_idx, s.uint(3))
    var corner = s.umod(v_idx, s.uint(3))
    var ml_base = s.imul(ml_id, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var ml_mat = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(11)))

    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var mat_ov = s.bit_and(s.load_word(b_instances, s.iadd(inst_base, s.uint(34))), s.uint(0xFFFF))
    var final_mat = s.select_u(s.ieq(mat_ov, s.uint(0xFFFF)), ml_mat, mat_ov)
    var tex_pk = s.load_word(b_materials, s.iadd(s.imul(final_mat, s.uint(24)), s.uint(23)))
    var t_alb = s.bit_and(tex_pk, s.uint(0xFF))

    var global_tri = s.iadd(first_tri, local_prim)
    var real_vtx = s.load_word(b_indices, s.iadd(s.imul(global_tri, s.uint(3)), corner))
    var v_base = s.imul(real_vtx, s.uint(12))

    var lpx = s.load_float(b_verts, v_base)
    var lpy = s.load_float(b_verts, s.iadd(v_base, s.uint(1)))
    var lpz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))
    var uv_u = s.load_float(b_verts, s.iadd(v_base, s.uint(9)))
    var uv_v = s.load_float(b_verts, s.iadd(v_base, s.uint(10)))

    var wp = emit_instance_point(s, b_instances, b_globals, b_verts, inst_base, v_base, lpx, lpy, lpz)
    var raw_mat_off = s.push_word(4)
    var mat_off = s.select_u(s.ieq(raw_mat_off, s.uint(0)), s.uint(56), raw_mat_off)
    var lclip = emit_mat4_mul_point(s, b_globals, mat_off, wp[0], wp[1], wp[2])

    var sc_x = s.push_float(0)
    var sc_y = s.push_float(1)
    var bi_x = s.push_float(2)
    var bi_y = s.push_float(3)
    var page_cx = s.fma(lclip[0], sc_x, bi_x)
    var page_cy = s.fma(lclip[1], sc_y, bi_y)
    var page_cz = s.nclamp(lclip[2], s.float(0.0), s.float(1.0))
    s.write_clip_position(page_cx, page_cy, page_cz, s.float(1.0))
    emit(s.code, OP_STORE, out_uv[1], s.vec2(uv_u, uv_v))
    emit(s.code, OP_STORE, out_talb[1], t_alb)
    return s.finish()


def vsm_alphamask_fragment_shader() -> List[UInt32]:
    """Depth-only Alpha-Cutout VSM Fragment Shader (`early_depth = False`, `enable_bindless = True`):
    Casts crisp serrated oak-twig, conifer-needle & wrought-iron fence shadows into `16k` VSM tiles!"""
    var s = AaaSpirvModule(
        MODEL_FRAGMENT,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        early_depth=False,
    )
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var in_uv = s.add_stage_io(STORAGE_INPUT, s.type_vec2, location=0, flat=False)
    var in_talb = s.add_stage_io(STORAGE_INPUT, s.type_uint, location=1, flat=True)

    var uv_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec2, uv_val, in_uv[1])
    var uv_x = s.extract_f(uv_val, 0)
    var uv_y = s.extract_f(uv_val, 1)

    var ddx_u = s.dpdx(uv_x)
    var ddx_v = s.dpdx(uv_y)
    var ddy_u = s.dpdy(uv_x)
    var ddy_v = s.dpdy(uv_y)
    var gx2 = s.fma(ddx_u, ddx_u, s.fmul(ddx_v, ddx_v))
    var gy2 = s.fma(ddy_u, ddy_u, s.fmul(ddy_v, ddy_v))
    var px_foot = s.fmul(s.fsqrt(s.nmax(s.float(1e-8), s.nmax(gx2, gy2))), s.float(512.0))
    var mip_est = s.nclamp(s.log2(s.nmax(s.float(1.0), px_foot)), s.float(0.0), s.float(6.0))

    var t_alb = s.fresh()
    emit(s.code, OP_LOAD, s.type_uint, t_alb, in_talb[1])
    var alpha_v = s.fvar(s.float(1.0))
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var rgba = s.sample_bindless_grad(
        bindless_arr,
        t_alb,
        uv_val,
        s.vec2(ddx_u, ddx_v),
        s.vec2(ddy_u, ddy_v),
    )
    var raw_a = s.extract_f(rgba, 3)
    s.store(alpha_v, s.fmul(raw_a, s.fma(mip_est, s.float(0.18), s.float(1.0))))
    s.end_if()

    s.discard_if(s.flt(s.load_f(alpha_v), s.float(0.48)))
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
    """Blocker-Search `PCSS` over `16,384 x 16,384` Virtual Shadow Map across all 4 Clipmap Cascades (`±16m..±1024m`)!
    Automatically selects Cascade `0 -> 1 -> 2 -> 3` when `b_globals[167] != 0` and samples its `64x64` quadrant.
    """
    var casc_flag = s.load_word(b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_FLAG_WORD)))
    var is_casc = s.ine(casc_flag, s.uint(0))

    var sel = _emit_select_vsm_cascade(s, b_globals, wx, wy, wz)
    var casc_u = s.select_u(is_casc, sel[1], s.uint(0))
    var mat_off = s.select_u(is_casc, sel[2], s.uint(56))
    var n_bias = s.select_f(is_casc, sel[6], s.float(0.012))

    var bx = s.fma(nx, n_bias, wx)
    var by = s.fma(ny, n_bias, wy)
    var bz = s.fma(nz, n_bias, wz)
    var lclip = emit_mat4_mul_point(s, b_globals, mat_off, bx, by, bz)
    var lu = s.fma(lclip[0], s.float(0.5), s.float(0.5))
    var lv = s.fma(s.fneg(lclip[1]), s.float(0.5), s.float(0.5))
    var z_rec = lclip[2]

    # Map cascade-local `lu, lv in [0..1]` into quadrant `(qx + lu_tap) * 0.5` (or `lu_tap * 1.0` in single-cascade mode)
    var q_xf = s.select_f(is_casc, s.float_from_uint(s.bit_and(casc_u, s.uint(1))), s.float(0.0))
    var q_yf = s.select_f(is_casc, s.float_from_uint(s.shr(casc_u, s.uint(1))), s.float(0.0))
    var atlas_sc = s.select_f(is_casc, s.float(0.5), s.float(1.0))

    # Scale penumbra UV radius by `1 / 4^cascade` so world-space penumbra width stays physically consistent across cascades!
    var casc_uv_sc = s.select_f(
        s.ieq(casc_u, s.uint(0)),
        s.float(1.0),
        s.select_f(
            s.ieq(casc_u, s.uint(1)),
            s.float(0.25),
            s.select_f(s.ieq(casc_u, s.uint(2)), s.float(0.0625), s.float(0.015625)),
        ),
    )

    var shadow_vis = s.fvar(s.float(1.0))
    var in_map = s.logical_and(
        s.logical_and(s.fgt(lu, s.float(0.002)), s.flt(lu, s.float(0.998))),
        s.logical_and(s.fgt(lv, s.float(0.002)), s.flt(lv, s.float(0.998))),
    )
    var casc_ok = s.logical_or(s.ieq(casc_flag, s.uint(0)), sel[0])
    var valid_rec = s.logical_and(
        casc_ok,
        s.logical_and(in_map, s.logical_and(s.fgt(z_rec, s.float(0.001)), s.flt(z_rec, s.float(0.999)))),
    )
    s.begin_if(valid_rec)
    var z_bias = s.float(0.0015)

    # 1. 8-Tap Vogel-Disk Blocker Search
    var blk_sum = s.fvar(s.float(0.0))
    var blk_cnt = s.fvar(s.float(0.0))
    var base_search_r = s.nclamp(sun_radius_uv, s.float(0.00025), s.float(0.012))
    var search_r = s.nmax(s.float(0.00015), s.fmul(base_search_r, casc_uv_sc))

    for tap in range(8):
        var r_norm = sqrt((Float32(tap) + Float32(0.5)) / Float32(8.0))
        var theta = Float32(tap) * Float32(2.39996323)
        var ox = s.fmul(search_r, s.float(r_norm * cos(theta)))
        var oy = s.fmul(search_r, s.float(r_norm * sin(theta)))
        var tu = s.fmul(s.fadd(q_xf, s.nclamp(s.fadd(lu, ox), s.float(0.0002), s.float(0.9998))), atlas_sc)
        var tv = s.fmul(s.fadd(q_yf, s.nclamp(s.fadd(lv, oy), s.float(0.0002), s.float(0.9998))), atlas_sc)
        var dz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, tu, tv)
        s.begin_if(s.flt(dz, s.fsub(z_rec, z_bias)))
        s.store(blk_sum, s.fadd(s.load_f(blk_sum), dz))
        s.store(blk_cnt, s.fadd(s.load_f(blk_cnt), s.float(1.0)))
        s.end_if()

    var bcnt = s.load_f(blk_cnt)
    s.begin_if(s.fgt(bcnt, s.float(0.5)))
    var avg_blk = s.fdiv(s.load_f(blk_sum), bcnt)
    var penumbra_ratio = s.fdiv(s.nmax(s.float(0.0), s.fsub(z_rec, avg_blk)), s.nmax(s.float(0.04), avg_blk))
    var min_flt = s.select_f(is_casc, s.float(0.00015), s.float(0.000076))
    var filter_r = s.nclamp(s.fmul(penumbra_ratio, search_r), min_flt, s.float(0.010))

    # 2. 12-Tap Golden-Angle Vogel-Disk PCF
    var lit_sum = s.fvar(s.float(0.0))
    for p_tap in range(12):
        var pr = sqrt((Float32(p_tap) + Float32(0.5)) / Float32(12.0))
        var pth = Float32(p_tap) * Float32(2.39996323) + Float32(0.62)
        var pox = s.fmul(filter_r, s.float(pr * cos(pth)))
        var poy = s.fmul(filter_r, s.float(pr * sin(pth)))
        var ptu = s.fmul(s.fadd(q_xf, s.nclamp(s.fadd(lu, pox), s.float(0.0002), s.float(0.9998))), atlas_sc)
        var ptv = s.fmul(s.fadd(q_yf, s.nclamp(s.fadd(lv, poy), s.float(0.0002), s.float(0.9998))), atlas_sc)
        var pz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, ptu, ptv)
        var is_lit = s.select_f(s.flt(pz, s.fsub(z_rec, z_bias)), s.float(0.0), s.float(1.0))
        s.store(lit_sum, s.fadd(s.load_f(lit_sum), is_lit))

    s.store(shadow_vis, s.fmul(s.load_f(lit_sum), s.float(1.0 / 12.0)))
    s.end_if()
    s.end_if()

    return s.load_f(shadow_vis)
