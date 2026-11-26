"""Area G (shadows) render SOTA, round 2: VSM correctness and caching (`sota_shadow_vsm`).

1. **Truncated pages were cached as lit (the Forest "HiZ drops shadow casters" bug).**
   `_pack_vsm_indirect_commands` wrote at most `max_draws` commands into `buf_vsm_indirect`, a buffer sized for the
   *visible* meshlet count. Every meshlet is duplicated into every dirty page it touches, so a fresh Forest frame
   needs ~2-3x more VSM commands than visible draws. The packer stopped early, the remaining pages were cleared to
   depth 1.0 (fully lit) and `VsmPageManager` cached them as valid: the shadows never came back. HiZ only changed
   how many visible draws there were (and so whether `max_draws` grew before or after the first VSM frame), which
   is why it looked like an occlusion-culling bug. Fix: the VSM command buffer is sized by what the packer needs
   (`SotaShadowState.vsm_cmd_capacity`, grown with headroom); a page whose commands still do not fit (growth failed)
   is left invalid so it is retried, and counted.
2. **Toroidal clipmap addressing** (`toroidal`, default on): a cascade page slot is the world-anchored light-space
   page coordinate mod 64 instead of its position in the camera-relative window. When the camera moves, only the
   page rows/columns that scroll into the window are re-rendered (was: the whole cascade). See
   `sota_shadow_toroidal_offsets` and the `emit_sample_vsm_depth` hook in `aaa_vsm.mojo`.
"""
from std.collections import List
from std.math import abs, floor, max, min
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_mat4_mul_point
from flint.render.aaa_vsm import (
    VSM_CASCADE_COUNT,
    VSM_CASCADE_PAGES_DIM,
    VSM_GLOBALS_CASCADE_FLAG_WORD,
    VSM_VIRTUAL_PAGES_DIM,
    SOTA_SHADOW_GLOBALS_CONTACT_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_OFFSETS_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_WORD,
    VsmPageManager,
    _emit_select_vsm_cascade,
    emit_sota_shadow_toroidal_remap,
    vsm_cascade_quadrant_origin,
)



struct SotaShadowState(Copyable, Movable):
    """Per-renderer state and settings of the Area G shadow work (`ren.sota_shadow`)."""

    var vsm_cmd_capacity: Int  # commands `buf_vsm_indirect` holds (independent of `max_draws`)
    var vsm_cmds_last: Int  # commands the packer wanted last time it ran
    var vsm_cmds_dropped_total: Int  # commands that did not fit (growth failed); their pages stay invalid
    var vsm_pages_retried_total: Int
    var vsm_cmd_grows: Int
    var toroidal: Bool  # cvar r.sota_shadow.toroidal (default off): toroidal clipmap addressing, scroll keeps pages
    var toroidal_active: Bool  # toroidal addressing was used for the current frame's cascades
    var toroidal_scrolled_pages_last: Int  # pages invalidated by window scroll this frame
    var cascade_full_invalidations: Int  # whole-cascade invalidations (lz snap / big jump / non-toroidal)
    var depth_snap: Int  # light-space depth snap index the cached cascade pages were rendered with
    var has_depth_snap: Bool
    var offsets: List[Tuple[Int, Int]]  # toroidal slot offset per cascade for the current frame ((0,0) when off)
    var forced_sync_frame: Bool  # this frame invalidated whole cascades (old requests are meaningless)
    var contact: Bool  # cvar r.sota_shadow.contact: screen-space contact shadows on the VSM sun path
    var contact_length_m: Float32
    var contact_thickness_m: Float32
    var gpu_pages: Bool  # cvar r.sota_shadow.gpu_pages (default off): GPU page allocation + cull + raster
    var gpu_active: Bool  # the GPU page path ran for the current frame
    var gpu_was_active: Bool

    def __init__(out self):
        self.vsm_cmd_capacity = 0
        self.vsm_cmds_last = 0
        self.vsm_cmds_dropped_total = 0
        self.vsm_pages_retried_total = 0
        self.vsm_cmd_grows = 0
        self.toroidal = False
        self.toroidal_active = False
        self.toroidal_scrolled_pages_last = 0
        self.cascade_full_invalidations = 0
        self.depth_snap = 0
        self.has_depth_snap = False
        self.offsets = List[Tuple[Int, Int]]()
        for _ in range(VSM_CASCADE_COUNT):
            self.offsets.append((0, 0))
        self.forced_sync_frame = False
        self.contact = False
        self.contact_length_m = Float32(0.35)
        self.contact_thickness_m = Float32(0.15)
        self.gpu_pages = False
        self.gpu_active = False
        self.gpu_was_active = False

    def update_cascades(
        mut self,
        mut vsm: VsmPageManager,
        prev_origins: List[Tuple[Int, Int]],
        new_origins: List[Tuple[Int, Int]],
        depth_snap: Int,
        use_cascades: Bool,
    ):
        """Once per frame, before the VSM requests are processed: invalidates what the cascade windows' move
        made stale and sets `offsets` / `toroidal_active` for this frame.
        - toroidal: only the page rows / columns entering each window;
        - v5 addressing: the whole cascade whenever its origin shifts (and the frame must re-mark synchronously);
        - both: every cascade when the 32 m light-space depth snap changes (the cached depth is relative to it)."""
        self.toroidal_scrolled_pages_last = 0
        self.forced_sync_frame = False
        self.toroidal_active = self.toroidal and use_cascades
        if not use_cascades or len(new_origins) < VSM_CASCADE_COUNT:
            for c in range(VSM_CASCADE_COUNT):
                self.offsets[c] = (0, 0)
            return
        if self.has_depth_snap and depth_snap != self.depth_snap:
            for c in range(VSM_CASCADE_COUNT):
                vsm.invalidate_cascade(c)
            self.cascade_full_invalidations += VSM_CASCADE_COUNT
            self.forced_sync_frame = True
        self.depth_snap = depth_snap
        self.has_depth_snap = True
        var has_prev = len(prev_origins) >= VSM_CASCADE_COUNT
        for c in range(VSM_CASCADE_COUNT):
            var moved = has_prev and (new_origins[c][0] != prev_origins[c][0] or new_origins[c][1] != prev_origins[c][1])
            if self.toroidal_active:
                self.offsets[c] = sota_shadow_toroidal_offset(new_origins[c])
                if moved:
                    var n = sota_shadow_scroll_invalidate(vsm, c, prev_origins[c], new_origins[c])
                    self.toroidal_scrolled_pages_last += n
                    if n >= VSM_CASCADE_PAGES_DIM * VSM_CASCADE_PAGES_DIM:
                        self.cascade_full_invalidations += 1
                        self.forced_sync_frame = True
            else:
                self.offsets[c] = (0, 0)
                if moved:
                    vsm.invalidate_cascade(c)
                    self.cascade_full_invalidations += 1
                    self.forced_sync_frame = True


def sota_shadow_vsm_cmd_target(need: Int) -> Int:
    """Capacity to grow the VSM command buffer to when `need` commands do not fit (25 % headroom, 4k granules)."""
    var t = need + need // 4
    return ((t + 4095) // 4096) * 4096


# ---------------------------------------------------------------------------------------------
# Toroidal clipmap addressing
# ---------------------------------------------------------------------------------------------
# Cascade c's window is 64 x 64 pages centred on the camera's page-snapped light-space position `origin`
# (`compute_vsm_cascade_page_origins`). Window-local page lx = 0 is global light-space page gx = ox - 32 (u grows
# with light-space x); ly = 0 is the *top* row (v grows with -y), global gyd = -oy - 32 with gyd = floor(-Y / page).
# The toroidal page-table slot is the global page mod 64, so slot = (local + offset) mod 64 with
#   offset_x = (ox - 32) mod 64,   offset_y = (-oy - 32) mod 64.
# A page keeps its slot (and its rendered depth) while the window scrolls; only pages entering the window change.


def sota_shadow_toroidal_offset(origin: Tuple[Int, Int]) -> Tuple[Int, Int]:
    var d = VSM_CASCADE_PAGES_DIM
    var ox = ((origin[0] - d // 2) % d + d) % d
    var oy = ((-origin[1] - d // 2) % d + d) % d
    return (ox, oy)


def sota_shadow_toroidal_offsets(origins: List[Tuple[Int, Int]]) -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]](capacity=VSM_CASCADE_COUNT)
    for c in range(VSM_CASCADE_COUNT):
        if c < len(origins):
            out.append(sota_shadow_toroidal_offset(origins[c]))
        else:
            out.append((0, 0))
    return out^


def sota_shadow_slot_from_local(cascade: Int, lx: Int, ly: Int, offsets: List[Tuple[Int, Int]]) -> Int:
    """Virtual page index (0..16383 in the 128x128 table) of cascade-window page (lx, ly)."""
    var d = VSM_CASCADE_PAGES_DIM
    var q = vsm_cascade_quadrant_origin(cascade)
    var ox = offsets[cascade][0] if cascade < len(offsets) else 0
    var oy = offsets[cascade][1] if cascade < len(offsets) else 0
    var sx = ((lx + ox) % d + d) % d
    var sy = ((ly + oy) % d + d) % d
    return (q[1] + sy) * VSM_VIRTUAL_PAGES_DIM + q[0] + sx


def sota_shadow_local_from_slot(vx: Int, vy: Int, offsets: List[Tuple[Int, Int]]) -> Tuple[Int, Int]:
    """Cascade-window page (lx, ly) of table page (vx, vy) (the inverse of `sota_shadow_slot_from_local`)."""
    var d = VSM_CASCADE_PAGES_DIM
    var c = (1 if vy >= d else 0) * 2 + (1 if vx >= d else 0)
    var ox = offsets[c][0] if c < len(offsets) else 0
    var oy = offsets[c][1] if c < len(offsets) else 0
    return ((((vx % d) - ox) % d + d) % d, (((vy % d) - oy) % d + d) % d)


def sota_shadow_scroll_invalidate(
    mut vsm: VsmPageManager, cascade: Int, prev_origin: Tuple[Int, Int], new_origin: Tuple[Int, Int]
) -> Int:
    """Toroidal scroll of one cascade window: invalidates only the slots whose global page changed (the rows and
    columns that entered the window). Returns the number of slots invalidated; a jump of a full window or more
    invalidates the whole cascade."""
    var d = VSM_CASCADE_PAGES_DIM
    var dx = new_origin[0] - prev_origin[0]
    var dy = -(new_origin[1] - prev_origin[1])  # window rows grow with -y
    if dx == 0 and dy == 0:
        return 0
    var q = vsm_cascade_quadrant_origin(cascade)
    if abs(dx) >= d or abs(dy) >= d:
        vsm.invalidate_cascade(cascade)
        return d * d
    var offs = List[Tuple[Int, Int]]()
    for c in range(VSM_CASCADE_COUNT):
        offs.append(sota_shadow_toroidal_offset(new_origin) if c == cascade else (0, 0))
    var n = 0
    for ly in range(d):
        for lx in range(d):
            # a window page is new iff it lies outside the previous window: its old local coord (lx + dx, ly + dy)
            var plx = lx + dx
            var ply = ly + dy
            if plx < 0 or plx >= d or ply < 0 or ply >= d:
                var v = sota_shadow_slot_from_local(cascade, lx, ly, offs)
                vsm.virtual_valid[v] = False
                n += 1
    _ = q
    return n


def sota_shadow_depth_snap_index(light_dir_toward_sun: Vec3, camera_pos: Vec3) -> Int:
    """Index of the 32 m light-space depth snap `compute_vsm_cascade_matrices` uses; cached cascade depth is only
    valid while it is unchanged (v5 kept stale pages when the camera moved > 32 m along the light direction)."""
    var fwd = (Vec3(0.0, 0.0, 0.0) - light_dir_toward_sun).normalized()
    return Int(floor(fwd.dot(camera_pos) / Float32(32.0) + Float32(0.5)))


# --- GPU side --------------------------------------------------------------------------------


def emit_sota_shadow_cascade_uv(
    mut s: AaaSpirvModule, b_globals: UInt32, wx: UInt32, wy: UInt32, wz: UInt32
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Single-tap VSM lookup coordinates for world point (wx, wy, wz), cascade-aware.
    Returns `(lu_table, lv_table, lu_local, lv_local, z_light)`: table coords go to `emit_sample_vsm_depth`, local
    coords are the cascade's [0..1] window (for in-map tests). v5's froxel / water / RT-hit lookups used cascade 0's
    matrix with full-table coordinates, which in cascade mode read the other cascades' quadrants."""
    var casc_flag = s.load_word(b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_FLAG_WORD)))
    var is_casc = s.ine(casc_flag, s.uint(0))
    var sel = _emit_select_vsm_cascade(s, b_globals, wx, wy, wz)
    var casc_u = s.select_u(is_casc, sel[1], s.uint(0))
    var lclip = emit_mat4_mul_point(s, b_globals, s.select_u(is_casc, sel[2], s.uint(56)), wx, wy, wz)
    var lu = s.fma(lclip[0], s.float(0.5), s.float(0.5))
    var lv = s.fma(s.fneg(lclip[1]), s.float(0.5), s.float(0.5))
    var q_xf = s.select_f(is_casc, s.float_from_uint(s.bit_and(casc_u, s.uint(1))), s.float(0.0))
    var q_yf = s.select_f(is_casc, s.float_from_uint(s.shr(casc_u, s.uint(1))), s.float(0.0))
    var atlas_sc = s.select_f(is_casc, s.float(0.5), s.float(1.0))
    var tu = s.fmul(s.fadd(q_xf, s.nclamp(lu, s.float(0.0002), s.float(0.9998))), atlas_sc)
    var tv = s.fmul(s.fadd(q_yf, s.nclamp(lv, s.float(0.0002), s.float(0.9998))), atlas_sc)
    return (tu, tv, lu, lv, lclip[2])
