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
from std.math import abs, cos, floor, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_mat4_mul_point
from flint.render.aaa_vsm import (
    SOTA_SHADOW_FLAG_COARSE_FALLBACK,
    SOTA_SHADOW_FLAG_FOG_REQUESTS,
    SOTA_SHADOW_FLAG_FOG_SOFT,
    SOTA_SHADOW_FLAG_PARENT_MARKS,
    SOTA_SHADOW_FLAG_TIGHT_MARKS,
    SOTA_SHADOW_GLOBALS_FLAGS_WORD,
    SOTA_SHADOW_GLOBALS_FOG_WORD,
    VSM_CASCADE_COUNT,
    VSM_CASCADE_PAGES_DIM,
    VSM_GLOBALS_CASCADE_1_WORD,
    VSM_GLOBALS_CASCADE_2_WORD,
    VSM_GLOBALS_CASCADE_3_WORD,
    VSM_GLOBALS_CASCADE_FLAG_WORD,
    VSM_VIRTUAL_PAGES_DIM,
    SOTA_SHADOW_GLOBALS_CONTACT_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_OFFSETS_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_WORD,
    VsmPageManager,
    _emit_select_vsm_cascade,
    emit_sample_vsm_depth,
    emit_sota_shadow_flag,
    emit_sota_shadow_page_slot,
    emit_sota_shadow_select_resident_cascade,
    emit_sota_shadow_select_resident_cascade_ex,
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
    # r3
    var coarse_fallback: Bool  # cvar r.sota_shadow.coarse_fallback (bug fix, on): parent pages + resident lookups
    var fog_requests: Bool  # cvar r.sota_shadow.fog_requests (bug fix, on): froxel fog requests the pages it samples
    var fog_soft: Bool  # cvar r.sota_shadow.fog_soft (on): footprint-matched cascade + 4 jittered fog taps
    var fog_footprint_scale: Float32  # fog cascade = finest whose texel >= scale x froxel width
    var tight_marks: Bool  # cvar r.sota_shadow.tight_marks: request neighbour pages only where the PCSS kernel reaches
    var cascade_lod_from: Int  # cvar r.sota_shadow.cascade_lod (GPU page path): cascades >= this cast a coarse LOD (4 = off)
    var parent_marks_active: Bool  # the pool was oversubscribed recently: parent pages requested, coarse-first alloc
    var pressure_hold_frames: Int  # frames parent marking stays on after the last dropped page
    var pressure_left: Int
    var pool_drops_last: Int  # requested pages the pool could not map (newest frame the CPU knows about)
    var pressure_frames_total: Int

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
        self.coarse_fallback = True
        self.fog_requests = True
        self.fog_soft = True
        self.fog_footprint_scale = Float32(1.0)
        self.tight_marks = True
        self.cascade_lod_from = 4  # off: no measurable gain on Forest (14 of 47 meshes have LODs), LOD self-shadow risk
        self.parent_marks_active = False
        self.pressure_hold_frames = 30
        self.pressure_left = 0
        self.pool_drops_last = 0
        self.pressure_frames_total = 0

    def update_pressure(mut self, dropped: Int, gpu_pages: Bool = True):
        """Once per frame with the newest known count of requested-but-unmapped pages: parent marking (and coarse-first
        allocation) turns on when the pool overflows and stays on `pressure_hold_frames` after the last overflow.
        GPU page path only: the CPU allocator drops in table-scan order and keeps its cached pages, so there parent
        pages only add drops (measured: FLIP vs RT 0.057 -> 0.061 on the oversubscribed view); the CPU path keeps
        just the samplers' resident fallback, which only touches pixels whose page is unmapped."""
        self.pool_drops_last = dropped
        if not self.coarse_fallback or not gpu_pages:
            self.pressure_left = 0
        elif dropped > 0:
            self.pressure_left = self.pressure_hold_frames
        elif self.pressure_left > 0:
            self.pressure_left -= 1
        self.parent_marks_active = self.pressure_left > 0
        if self.parent_marks_active:
            self.pressure_frames_total += 1

    def globals_flags(self) -> UInt32:
        """`b_globals[234]` (see `SOTA_SHADOW_GLOBALS_FLAGS_WORD`)."""
        var f = UInt32(0)
        if self.coarse_fallback:
            f |= UInt32(SOTA_SHADOW_FLAG_COARSE_FALLBACK)
        if self.fog_requests:
            f |= UInt32(SOTA_SHADOW_FLAG_FOG_REQUESTS)
        if self.fog_soft:
            f |= UInt32(SOTA_SHADOW_FLAG_FOG_SOFT)
        if self.parent_marks_active:
            f |= UInt32(SOTA_SHADOW_FLAG_PARENT_MARKS)
        if self.tight_marks:
            f |= UInt32(SOTA_SHADOW_FLAG_TIGHT_MARKS)
        return f

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
    mut s: AaaSpirvModule, b_globals: UInt32, wx: UInt32, wy: UInt32, wz: UInt32, b_vsm: UInt32 = 0
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """Single-tap VSM lookup coordinates for world point (wx, wy, wz), cascade-aware.
    Returns `(lu_table, lv_table, lu_local, lv_local, z_light)`: table coords go to `emit_sample_vsm_depth`, local
    coords are the cascade's [0..1] window (for in-map tests). v5's froxel / water / RT-hit lookups used cascade 0's
    matrix with full-table coordinates, which in cascade mode read the other cascades' quadrants."""
    var casc_flag = s.load_word(b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_FLAG_WORD)))
    var is_casc = s.ine(casc_flag, s.uint(0))
    # r3: with `b_vsm` the lookup falls back to the finest resident cascade (coarse fallback flag)
    var sel = (
        emit_sota_shadow_select_resident_cascade(s, b_globals, b_vsm, wx, wy, wz) if b_vsm != 0
        else _emit_select_vsm_cascade(s, b_globals, wx, wy, wz)
    )
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


# --- r3: froxel fog sun visibility --------------------------------------------------------------
def emit_sota_shadow_fog_visibility(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    b_vsm: UInt32,
    vsm_samp: Tuple[UInt32, UInt32],
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    froxel_width_m: UInt32,
    ign: UInt32,
    has_fog: UInt32,
) -> UInt32:
    """Sun visibility (float 0..1) of a froxel centre for the fog in-scatter.

    v5 / r2 took one tap in the finest cascade that contains the point. Fog points in mid-air project to pages that
    no visible surface requested (the ground under a roof seen from below, the air in front of a huge wall seen
    from its shadowed side): those pages were unmapped and read as lit, which drew page-sized lit blocks (the
    "checker" in the fog shadow volumes). The finest cascade is also far sharper than a froxel, which aliases.

    - `fog_soft` (flag bit 2): the cascade is the finest whose texel is >= `fog_footprint_scale` x the froxel width
      (cascade c texel = 2 / (|row0(M_c)| x 8192)), and 4 Vogel taps jittered per froxel and frame inside half a froxel
      are averaged (the froxel history's temporal reprojection integrates them).
    - coarse fallback (bit 0): an unmapped page falls back to the finest coarser resident cascade.
    - `fog_requests` (bit 1): the froxel re-requests the resident page it sampled (keeps it alive and re-rendered after
      an invalidation) or, when no cascade is resident there, the footprint-cascade page, so next frame the fog has
      a page even where no surface asked for one. Requesting the footprint page for every froxel instead cost Forest
      +85 ms VSM raster per full refresh (coarse pages hold every tree at full LOD).
    All flags off reproduces the r2 lookup (finest fitting cascade, one tap)."""
    var is_casc = s.ine(s.load_word(b_globals, s.uint(UInt32(VSM_GLOBALS_CASCADE_FLAG_WORD))), s.uint(0))
    var soft = s.logical_and(is_casc, emit_sota_shadow_flag(s, b_globals, SOTA_SHADOW_FLAG_FOG_SOFT))
    var req_on = s.logical_and(is_casc, emit_sota_shadow_flag(s, b_globals, SOTA_SHADOW_FLAG_FOG_REQUESTS))
    var scale = s.load_float(b_globals, s.uint(UInt32(SOTA_SHADOW_GLOBALS_FOG_WORD)))
    var target = s.fmul(s.nmax(s.float(0.0), scale), froxel_width_m)
    # footprint cascade: number of cascades (of 0..2) whose texel is finer than the target
    var mats = List[UInt32]()
    mats.append(s.uint(56))
    mats.append(s.uint(UInt32(VSM_GLOBALS_CASCADE_1_WORD)))
    mats.append(s.uint(UInt32(VSM_GLOBALS_CASCADE_2_WORD)))
    mats.append(s.uint(UInt32(VSM_GLOBALS_CASCADE_3_WORD)))
    var sxs = List[UInt32]()
    for k in range(4):
        var m0 = s.load_float(b_globals, mats[k])
        var m4 = s.load_float(b_globals, s.iadd(mats[k], s.uint(4)))
        var m8 = s.load_float(b_globals, s.iadd(mats[k], s.uint(8)))
        sxs.append(s.fsqrt(s.fma(m0, m0, s.fma(m4, m4, s.fmul(m8, m8)))))
    var min_c = s.uint(0)
    for k in range(3):
        # texel_k = 2 / (sx_k * 8192) < target  <=>  sx_k * 4096 * target > 1
        var finer = s.fgt(s.fmul(s.fmul(sxs[k], s.float(4096.0)), target), s.float(1.0))
        min_c = s.iadd(min_c, s.select_u(finer, s.uint(1), s.uint(0)))
    min_c = s.select_u(soft, min_c, s.uint(0))
    var sel = emit_sota_shadow_select_resident_cascade_ex(s, b_globals, b_vsm, wx, wy, wz, min_c)
    var casc_u = s.select_u(is_casc, sel[1], s.uint(0))
    var mat = s.select_u(is_casc, sel[2], s.uint(56))
    var lclip = emit_mat4_mul_point(s, b_globals, mat, wx, wy, wz)
    var lu = s.fma(lclip[0], s.float(0.5), s.float(0.5))
    var lv = s.fma(s.fneg(lclip[1]), s.float(0.5), s.float(0.5))
    var z_rec = lclip[2]
    var q_xf = s.select_f(is_casc, s.float_from_uint(s.bit_and(casc_u, s.uint(1))), s.float(0.0))
    var q_yf = s.select_f(is_casc, s.float_from_uint(s.shr(casc_u, s.uint(1))), s.float(0.0))
    var atlas_sc = s.select_f(is_casc, s.float(0.5), s.float(1.0))
    var sx_sel = s.select_f(
        s.ieq(casc_u, s.uint(0)),
        sxs[0],
        s.select_f(s.ieq(casc_u, s.uint(1)), sxs[1], s.select_f(s.ieq(casc_u, s.uint(2)), sxs[2], sxs[3])),
    )
    # tap disc: half a froxel, in window uv (clip * 0.5)
    var r_uv = s.select_f(soft, s.fmul(s.fmul(froxel_width_m, s.float(0.25)), sx_sel), s.float(0.0))
    var vis = s.fvar(s.float(1.0))
    var in_map = s.logical_and(
        s.logical_and(s.fgt(lu, s.float(0.002)), s.flt(lu, s.float(0.998))),
        s.logical_and(s.fgt(lv, s.float(0.002)), s.flt(lv, s.float(0.998))),
    )
    var casc_ok = s.logical_or(s.logical_not(is_casc), sel[0])
    s.begin_if(s.logical_and(casc_ok, s.logical_and(in_map, s.flt(z_rec, s.float(0.999)))))
    var acc = s.fvar(s.float(0.0))
    var rot = s.fmul(ign, s.float(6.2831853))
    var cr = s.fcos(rot)
    var sr = s.fsin(rot)
    for t in range(4):
        var rn = sqrt((Float32(t) + Float32(0.5)) / Float32(4.0))
        var th = Float32(t) * Float32(2.39996323)
        var ox0 = rn * cos(th)
        var oy0 = rn * sin(th)
        # the Vogel tap rotated by the per-froxel angle
        var ox = s.fmul(r_uv, s.fsub(s.fmul(cr, s.float(ox0)), s.fmul(sr, s.float(oy0))))
        var oy = s.fmul(r_uv, s.fadd(s.fmul(sr, s.float(ox0)), s.fmul(cr, s.float(oy0))))
        var tu = s.fmul(s.fadd(q_xf, s.nclamp(s.fadd(lu, ox), s.float(0.0002), s.float(0.9998))), atlas_sc)
        var tv = s.fmul(s.fadd(q_yf, s.nclamp(s.fadd(lv, oy), s.float(0.0002), s.float(0.9998))), atlas_sc)
        var dz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, tu, tv, b_globals)
        var lit = s.select_f(s.flt(dz, s.fsub(z_rec, s.float(0.0015))), s.float(0.0), s.float(1.0))
        s.store(acc, s.fadd(s.load_f(acc), lit))
    s.store(vis, s.fmul(s.load_f(acc), s.float(0.25)))
    s.end_if()
    # page requests for the next frame (only where there is fog to light)
    s.begin_if(s.logical_and(req_on, has_fog))
    var m_min = s.select_u(
        s.ieq(min_c, s.uint(0)),
        mats[0],
        s.select_u(s.ieq(min_c, s.uint(1)), mats[1], s.select_u(s.ieq(min_c, s.uint(2)), mats[2], mats[3])),
    )
    var lcm = emit_mat4_mul_point(s, b_globals, m_min, wx, wy, wz)
    var fit_m = s.logical_and(
        s.logical_and(s.flt(s.fabs(lcm[0]), s.float(0.95)), s.flt(s.fabs(lcm[1]), s.float(0.95))),
        s.logical_and(s.fgt(lcm[2], s.float(0.001)), s.flt(lcm[2], s.float(0.999))),
    )
    # a resident page is re-requested (keeps it alive and re-rendered after invalidations; surfaces usually request
    # it already); with nothing resident the footprint-cascade page is requested, so only fog-only points add pages
    var new_pg = s.logical_and(s.logical_not(sel[7]), fit_m)
    var rq_c = s.select_u(new_pg, min_c, casc_u)
    var rq_u = s.select_f(new_pg, s.fma(lcm[0], s.float(0.5), s.float(0.5)), lu)
    var rq_v = s.select_f(new_pg, s.fma(s.fneg(lcm[1]), s.float(0.5), s.float(0.5)), lv)
    s.begin_if(sel[0])
    var ri = s.iadd(s.uint(16384), emit_sota_shadow_page_slot(s, b_globals, rq_c, rq_u, rq_v))
    s.begin_if(s.ieq(s.load_word(b_vsm, ri), s.uint(0)))
    s.store_word(b_vsm, ri, s.uint(1))
    s.end_if()
    s.end_if()
    s.end_if()
    return s.load_f(vis)
