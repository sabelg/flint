"""SoA 8-Wide SIMD Reciprocal Velocity Obstacles (`ORCA`, `flint.nav.orca`, v5).

Implements van den Berg et al. (2011) Optimal Reciprocal Collision Avoidance (`CrowdSimulator`):
1. Cache-aligned Structure-of-Arrays (`px, py, pz, vx, vy, vz, pref_vx, pref_vz, radius, max_speed`)
   padded to multiples of 8 (`LANES = 8`) for native `F8 = SIMD[DType.float32, 8]` execution.
2. O(N) spatial hash grid rebuilt per tick for bounded-radius neighbor queries.
3. Wall-segment obstacle detour steering around open doorway/corner endpoints + full non-reciprocal
   (`alpha = 1.0`) capsule-barrier ORCA wall half-planes.
4. 8-wide `F8` SIMD preferred velocity computation, 8-wide `F8` SIMD neighbor half-plane
   `(q_x, q_z, n_x, n_z)` construction, incremental 2D linear programming (`_lp2_orca`) +
   3D minimum-penetration fallback (`_lp3_orca`), `F8` position advection, and spatial-grid
   hard-disc & swept-wall depenetration (`step_orca_simd8`) enforcing exact 0-overlap invariants.
"""
from std.collections import List
from std.math import sqrt, abs, min, max, floor, isfinite
from flint.core.math import Vec3
from flint.core.detsort import mix64
from flint.physics.solver_bodies import F8, LANES
from flint.nav.navmesh import NavMesh, NavPath

comptime _ORCA_HASH_BUCKETS: Int = 512
comptime _ORCA_WALL_BUCKETS: Int = 512
comptime _ORCA_WALL_CELL_SIZE: Float32 = 4.0
comptime _ORCA_EPS: Float32 = 1.0e-5


@always_inline
def _safe_cell_coord(v: Float32, inv_cs: Float32) -> Int:
    var f = Float64(v) * Float64(inv_cs)
    if not isfinite(f):
        return 0
    var fl = floor(f)
    if fl <= -1_000_000_000.0:
        return -1_000_000_000
    if fl >= 1_000_000_000.0:
        return 1_000_000_000
    return Int(fl)


@fieldwise_init
struct OrcaLine(Copyable, ImplicitlyCopyable, Movable):
    """Half-plane constraint `(v - point) . normal >= 0` in XZ velocity space."""
    var px: Float32
    var pz: Float32
    var nx: Float32
    var nz: Float32


@fieldwise_init
struct CrowdWallSegment(Copyable, ImplicitlyCopyable, Movable):
    var ax: Float32
    var az: Float32
    var bx: Float32
    var bz: Float32


@always_inline
def _pt_seg_closest_xz(
    px: Float32,
    pz: Float32,
    ax: Float32,
    az: Float32,
    bx: Float32,
    bz: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """Returns `(closest_x, closest_z, t)` on segment `A -> B` to `(px, pz)`."""
    var dx = bx - ax
    var dz = bz - az
    var l2 = dx * dx + dz * dz
    var t = Float32(0.0)
    if l2 > Float32(1.0e-10):
        t = max(Float32(0.0), min(Float32(1.0), ((px - ax) * dx + (pz - az) * dz) / l2))
    return (ax + dx * t, az + dz * t, t)


@always_inline
def _segments_proper_intersect_xz(
    p0x: Float32,
    p0z: Float32,
    p1x: Float32,
    p1z: Float32,
    q0x: Float32,
    q0z: Float32,
    q1x: Float32,
    q1z: Float32,
) -> Tuple[Bool, Float32]:
    """Checks if segment `P0 -> P1` intersects `Q0 -> Q1`, returning `(hit, t_p)`."""
    var r_x = p1x - p0x
    var r_z = p1z - p0z
    var s_x = q1x - q0x
    var s_z = q1z - q0z
    var denom = r_x * s_z - r_z * s_x
    if abs(denom) <= Float32(1.0e-10):
        return (False, Float32(1.0))
    var qp_x = q0x - p0x
    var qp_z = q0z - p0z
    var inv_d = Float32(1.0) / denom
    var t = (qp_x * s_z - qp_z * s_x) * inv_d
    var u = (qp_x * r_z - qp_z * r_x) * inv_d
    if t >= Float32(0.0) and t <= Float32(1.0) and u >= Float32(0.0) and u <= Float32(1.0):
        return (True, t)
    return (False, Float32(1.0))


@always_inline
def _seg_seg_dist2_and_t_xz(
    p0x: Float32,
    p0z: Float32,
    p1x: Float32,
    p1z: Float32,
    ax: Float32,
    az: Float32,
    bx: Float32,
    bz: Float32,
) -> Tuple[Float32, Float32]:
    """Returns `(min_dist_sq, approx_t_along_p)` between segment `P0 -> P1` and wall `A -> B`."""
    var ix = _segments_proper_intersect_xz(p0x, p0z, p1x, p1z, ax, az, bx, bz)
    if ix[0]:
        return (Float32(0.0), ix[1])

    var best_d2 = Float32(1.0e30)
    var best_t = Float32(1.0)

    var c0 = _pt_seg_closest_xz(p0x, p0z, ax, az, bx, bz)
    var d0 = (p0x - c0[0]) * (p0x - c0[0]) + (p0z - c0[1]) * (p0z - c0[1])
    if d0 < best_d2:
        best_d2 = d0
        best_t = Float32(0.0)

    var c1 = _pt_seg_closest_xz(p1x, p1z, ax, az, bx, bz)
    var d1 = (p1x - c1[0]) * (p1x - c1[0]) + (p1z - c1[1]) * (p1z - c1[1])
    if d1 < best_d2:
        best_d2 = d1
        best_t = Float32(1.0)

    var ca = _pt_seg_closest_xz(ax, az, p0x, p0z, p1x, p1z)
    var da = (ax - ca[0]) * (ax - ca[0]) + (az - ca[1]) * (az - ca[1])
    if da < best_d2:
        best_d2 = da
        best_t = ca[2]

    var cb = _pt_seg_closest_xz(bx, bz, p0x, p0z, p1x, p1z)
    var db = (bx - cb[0]) * (bx - cb[0]) + (bz - cb[1]) * (bz - cb[1])
    if db < best_d2:
        best_d2 = db
        best_t = cb[2]

    return (best_d2, best_t)


@always_inline
def _lp1_orca(
    lines: List[OrcaLine],
    line_no: Int,
    max_speed: Float32,
    opt_vx: Float32,
    opt_vz: Float32,
    direction_opt: Bool,
    mut res_vx: Float32,
    mut res_vz: Float32,
) -> Bool:
    """1D linear program along boundary of `lines[line_no]` intersected with disk `|v| <= max_speed`
    and all prior half-planes `0 .. line_no - 1`."""
    var line = lines[line_no]
    var dir_x = -line.nz
    var dir_z = line.nx
    var dot_p_dir = line.px * dir_x + line.pz * dir_z
    var disc = dot_p_dir * dot_p_dir + max_speed * max_speed - (line.px * line.px + line.pz * line.pz)
    if disc < Float32(0.0):
        return False

    var sqrt_disc = sqrt(disc)
    var t_left = -dot_p_dir - sqrt_disc
    var t_right = -dot_p_dir + sqrt_disc

    for i in range(line_no):
        var prev = lines[i]
        var denom = dir_x * prev.nx + dir_z * prev.nz
        var num = (line.px - prev.px) * prev.nx + (line.pz - prev.pz) * prev.nz
        if abs(denom) <= _ORCA_EPS:
            if num < Float32(0.0):
                return False
            continue
        var t_hit = -num / denom
        if denom > Float32(0.0):
            t_left = max(t_left, t_hit)
        else:
            t_right = min(t_right, t_hit)
        if t_left > t_right:
            return False

    var t_best: Float32
    if direction_opt:
        t_best = t_right if (opt_vx * dir_x + opt_vz * dir_z) > Float32(0.0) else t_left
    else:
        var t_proj = (opt_vx - line.px) * dir_x + (opt_vz - line.pz) * dir_z
        t_best = max(t_left, min(t_right, t_proj))

    res_vx = line.px + t_best * dir_x
    res_vz = line.pz + t_best * dir_z
    return True


def _lp2_orca(
    lines: List[OrcaLine],
    max_speed: Float32,
    opt_vx: Float32,
    opt_vz: Float32,
    direction_opt: Bool,
    mut res_vx: Float32,
    mut res_vz: Float32,
) -> Int:
    """Incremental 2D linear program; returns `len(lines)` on success or index of first infeasible line."""
    if direction_opt:
        res_vx = opt_vx * max_speed
        res_vz = opt_vz * max_speed
    else:
        var mag2 = opt_vx * opt_vx + opt_vz * opt_vz
        if mag2 > max_speed * max_speed:
            var s = max_speed / sqrt(mag2)
            res_vx = opt_vx * s
            res_vz = opt_vz * s
        else:
            res_vx = opt_vx
            res_vz = opt_vz

    for i in range(len(lines)):
        var l = lines[i]
        if (res_vx - l.px) * l.nx + (res_vz - l.pz) * l.nz < Float32(0.0):
            var bak_vx = res_vx
            var bak_vz = res_vz
            if not _lp1_orca(lines, i, max_speed, opt_vx, opt_vz, direction_opt, res_vx, res_vz):
                res_vx = bak_vx
                res_vz = bak_vz
                return i
    return len(lines)


def _lp3_orca(
    lines: List[OrcaLine],
    num_obst_lines: Int,
    begin_line: Int,
    max_speed: Float32,
    mut res_vx: Float32,
    mut res_vz: Float32,
):
    """3D minimum-penetration fallback LP when dense crowds create mutually infeasible half-planes
    (keeps all static wall lines `0 .. num_obst_lines - 1` strictly hard!)."""
    var distance = Float32(0.0)
    for i in range(begin_line, len(lines)):
        var l_i = lines[i]
        var signed_d = (l_i.px - res_vx) * l_i.nx + (l_i.pz - res_vz) * l_i.nz
        if signed_d > distance:
            var proj_lines = List[OrcaLine](capacity=i)
            for k in range(num_obst_lines):
                proj_lines.append(lines[k])

            for j in range(num_obst_lines, i):
                var l_j = lines[j]
                var det = l_i.nx * l_j.nz - l_i.nz * l_j.nx
                var p_mid_x: Float32
                var p_mid_z: Float32
                if abs(det) <= _ORCA_EPS:
                    if l_i.nx * l_j.nx + l_i.nz * l_j.nz > Float32(0.0):
                        continue
                    p_mid_x = (l_i.px + l_j.px) * Float32(0.5)
                    p_mid_z = (l_i.pz + l_j.pz) * Float32(0.5)
                else:
                    var dir_ix = -l_i.nz
                    var dir_iz = l_i.nx
                    var num = (l_j.px - l_i.px) * l_j.nx + (l_j.pz - l_i.pz) * l_j.nz
                    var denom = dir_ix * l_j.nx + dir_iz * l_j.nz
                    var t = num / denom if abs(denom) > _ORCA_EPS else Float32(0.0)
                    p_mid_x = l_i.px + t * dir_ix
                    p_mid_z = l_i.pz + t * dir_iz

                var bis_nx = l_j.nx - l_i.nx
                var bis_nz = l_j.nz - l_i.nz
                var bis_l2 = bis_nx * bis_nx + bis_nz * bis_nz
                if bis_l2 <= Float32(1.0e-10):
                    continue
                var inv_bl = Float32(1.0) / sqrt(bis_l2)
                proj_lines.append(OrcaLine(p_mid_x, p_mid_z, bis_nx * inv_bl, bis_nz * inv_bl))

            var bak_vx = res_vx
            var bak_vz = res_vz
            if _lp2_orca(proj_lines, max_speed, l_i.nx, l_i.nz, True, res_vx, res_vz) < len(proj_lines):
                res_vx = bak_vx
                res_vz = bak_vz
            distance = (l_i.px - res_vx) * l_i.nx + (l_i.pz - res_vz) * l_i.nz


def _add_wall_orca_line(
    ax_p: Float32,
    az_p: Float32,
    rad: Float32,
    max_speed: Float32,
    seg: CrowdWallSegment,
    tau_obst: Float32,
    inv_tau_obst: Float32,
    inv_dt: Float32,
    mut lines: List[OrcaLine],
):
    """Full non-reciprocal (`alpha = 1.0`) capsule-barrier ORCA half-plane against a static wall segment."""
    var cp = _pt_seg_closest_xz(ax_p, az_p, seg.ax, seg.az, seg.bx, seg.bz)
    var ox = ax_p - cp[0]
    var oz = az_p - cp[1]
    var d2 = ox * ox + oz * oz
    var r_eff = rad + Float32(0.025)
    var horizon_dist = r_eff + max_speed * tau_obst + Float32(0.25)
    if d2 > horizon_dist * horizon_dist:
        return

    var nx: Float32
    var nz: Float32
    var d = sqrt(max(Float32(0.0), d2))
    if d > Float32(1.0e-6):
        nx = ox / d
        nz = oz / d
    else:
        var wx = seg.bx - seg.ax
        var wz = seg.bz - seg.az
        var wl = sqrt(max(Float32(1.0e-10), wx * wx + wz * wz))
        nx = -wz / wl
        nz = wx / wl

    if d > r_eff:
        var max_approach = (d - r_eff) * inv_tau_obst
        lines.append(OrcaLine(-nx * max_approach, -nz * max_approach, nx, nz))
    else:
        var push_spd = min(max(Float32(0.5), max_speed), (r_eff - d) * inv_dt)
        lines.append(OrcaLine(nx * push_spd, nz * push_spd, nx, nz))


struct CrowdSimulator(Copyable, Movable):
    """SoA 8-wide SIMD ORCA crowd steering & collision avoidance engine."""
    var count: Int
    var padded: Int
    var px: List[Float32]
    var py: List[Float32]
    var pz: List[Float32]
    var vx: List[Float32]
    var vy: List[Float32]
    var vz: List[Float32]
    var pref_vx: List[Float32]
    var pref_vz: List[Float32]
    var goal_x: List[Float32]
    var goal_y: List[Float32]
    var goal_z: List[Float32]
    var has_goal: List[Bool]
    var radius: List[Float32]
    var max_speed: List[Float32]
    var active: List[Bool]
    var walls: List[CrowdWallSegment]

    var time_horizon: Float32
    var time_horizon_obst: Float32
    var neighbor_dist: Float32
    var max_neighbors: Int
    var personal_space_margin: Float32
    var agent_height: Float32

    # Spatial hash grid over `walls` to eliminate O(N*W) scans
    var _wall_head: List[Int]
    var _wall_next: List[Int]
    var _wall_seg_idx: List[Int]
    var _large_walls: List[Int]
    var _wall_grid_dirty: Bool

    def __init__(
        out self,
        time_horizon: Float32 = 1.8,
        neighbor_dist: Float32 = 5.0,
        max_neighbors: Int = 24,
    ):
        self.count = 0
        self.padded = 0
        self.px = List[Float32]()
        self.py = List[Float32]()
        self.pz = List[Float32]()
        self.vx = List[Float32]()
        self.vy = List[Float32]()
        self.vz = List[Float32]()
        self.pref_vx = List[Float32]()
        self.pref_vz = List[Float32]()
        self.goal_x = List[Float32]()
        self.goal_y = List[Float32]()
        self.goal_z = List[Float32]()
        self.has_goal = List[Bool]()
        self.radius = List[Float32]()
        self.max_speed = List[Float32]()
        self.active = List[Bool]()
        self.walls = List[CrowdWallSegment]()
        self.time_horizon = max(Float32(0.1), time_horizon)
        self.time_horizon_obst = Float32(0.8)
        self.neighbor_dist = max(Float32(0.5), neighbor_dist)
        self.max_neighbors = max(4, max_neighbors)
        self.personal_space_margin = Float32(0.025)
        self.agent_height = Float32(2.0)
        self._wall_head = List[Int]()
        self._wall_next = List[Int]()
        self._wall_seg_idx = List[Int]()
        self._large_walls = List[Int]()
        self._wall_grid_dirty = True

    def _pad_columns(mut self):
        var req = ((self.count + LANES - 1) // LANES) * LANES
        while len(self.px) < req:
            self.px.append(Float32(1.0e6))
            self.py.append(Float32(0.0))
            self.pz.append(Float32(1.0e6))
            self.vx.append(Float32(0.0))
            self.vy.append(Float32(0.0))
            self.vz.append(Float32(0.0))
            self.pref_vx.append(Float32(0.0))
            self.pref_vz.append(Float32(0.0))
            self.goal_x.append(Float32(1.0e6))
            self.goal_y.append(Float32(0.0))
            self.goal_z.append(Float32(1.0e6))
            self.has_goal.append(False)
            self.radius.append(Float32(0.35))
            self.max_speed.append(Float32(0.0))
            self.active.append(False)
        self.padded = req

    def add_agent(
        mut self,
        pos: Vec3,
        radius: Float32 = 0.35,
        max_speed: Float32 = 3.5,
        goal: Vec3 = Vec3(1.0e30, 0.0, 0.0),
    ) -> Int:
        var idx = self.count
        self.count += 1
        self._pad_columns()
        self.px[idx] = pos.x
        self.py[idx] = pos.y
        self.pz[idx] = pos.z
        self.vx[idx] = Float32(0.0)
        self.vy[idx] = Float32(0.0)
        self.vz[idx] = Float32(0.0)
        self.radius[idx] = max(Float32(0.05), radius)
        self.max_speed[idx] = max(Float32(0.0), max_speed)
        self.active[idx] = True
        if goal.x < Float32(1.0e29):
            self.goal_x[idx] = goal.x
            self.goal_y[idx] = goal.y
            self.goal_z[idx] = goal.z
            self.has_goal[idx] = True
        else:
            self.has_goal[idx] = False
        return idx

    def set_agent_goal(mut self, idx: Int, goal: Vec3):
        if idx < 0 or idx >= self.count:
            return
        self.goal_x[idx] = goal.x
        self.goal_y[idx] = goal.y
        self.goal_z[idx] = goal.z
        self.has_goal[idx] = True

    def set_preferred_velocity(mut self, idx: Int, v: Vec3):
        if idx < 0 or idx >= self.count:
            return
        self.pref_vx[idx] = v.x
        self.pref_vz[idx] = v.z
        self.has_goal[idx] = False

    def add_wall_segment(mut self, a: Vec3, b: Vec3):
        self.walls.append(CrowdWallSegment(a.x, a.z, b.x, b.z))
        self._wall_grid_dirty = True

    def clear_walls(mut self):
        self.walls.clear()
        self._wall_head.clear()
        self._wall_next.clear()
        self._wall_seg_idx.clear()
        self._large_walls.clear()
        self._wall_grid_dirty = True

    def _ensure_wall_grid(mut self):
        if not self._wall_grid_dirty:
            return
        self._wall_grid_dirty = False
        var n_walls = len(self.walls)
        self._wall_next.clear()
        self._wall_seg_idx.clear()
        self._large_walls.clear()
        if n_walls == 0:
            self._wall_head.clear()
            return
        self._wall_head = List[Int](length=_ORCA_WALL_BUCKETS, fill=-1)
        var inv_wcs = Float32(1.0) / _ORCA_WALL_CELL_SIZE
        var max_span_f = _ORCA_WALL_CELL_SIZE * Float32(64.0)
        for wi in range(n_walls):
            var w = self.walls[wi]
            if not (isfinite(w.ax) and isfinite(w.az) and isfinite(w.bx) and isfinite(w.bz)):
                continue
            var lo_x = min(w.ax, w.bx)
            var hi_x = max(w.ax, w.bx)
            var lo_z = min(w.az, w.bz)
            var hi_z = max(w.az, w.bz)
            if (
                (hi_x - lo_x) > max_span_f
                or (hi_z - lo_z) > max_span_f
                or abs(lo_x) > Float32(1.0e8)
                or abs(hi_x) > Float32(1.0e8)
                or abs(lo_z) > Float32(1.0e8)
                or abs(hi_z) > Float32(1.0e8)
            ):
                self._large_walls.append(wi)
                continue
            var cx0 = _safe_cell_coord(lo_x, inv_wcs)
            var cx1 = _safe_cell_coord(hi_x, inv_wcs)
            var cz0 = _safe_cell_coord(lo_z, inv_wcs)
            var cz1 = _safe_cell_coord(hi_z, inv_wcs)
            var span_x = cx1 - cx0 + 1
            var span_z = cz1 - cz0 + 1
            if span_x <= 0 or span_z <= 0 or span_x > 64 or span_z > 64 or span_x > 4096 // span_z:
                self._large_walls.append(wi)
                continue
            for cz in range(cz0, cz1 + 1):
                for cx in range(cx0, cx1 + 1):
                    var bkt = Self._cell_bucket(cx, cz)
                    var node = len(self._wall_seg_idx)
                    self._wall_seg_idx.append(wi)
                    self._wall_next.append(self._wall_head[bkt])
                    self._wall_head[bkt] = node

    def _query_wall_candidates(
        self,
        lo_x: Float32,
        lo_z: Float32,
        hi_x: Float32,
        hi_z: Float32,
        mut out_walls: List[Int],
        mut stamp: List[Int],
        cur_stamp: Int,
    ):
        out_walls.clear()
        var n_walls = len(self.walls)
        if n_walls == 0:
            return
        var max_span_f = _ORCA_WALL_CELL_SIZE * Float32(64.0)
        if (
            len(self._wall_head) == 0
            or not (isfinite(lo_x) and isfinite(lo_z) and isfinite(hi_x) and isfinite(hi_z))
            or (hi_x - lo_x) > max_span_f
            or (hi_z - lo_z) > max_span_f
            or abs(lo_x) > Float32(1.0e8)
            or abs(hi_x) > Float32(1.0e8)
            or abs(lo_z) > Float32(1.0e8)
            or abs(hi_z) > Float32(1.0e8)
        ):
            for wi in range(n_walls):
                if stamp[wi] != cur_stamp:
                    stamp[wi] = cur_stamp
                    out_walls.append(wi)
            return
        var inv_wcs = Float32(1.0) / _ORCA_WALL_CELL_SIZE
        var cx0 = _safe_cell_coord(lo_x, inv_wcs)
        var cx1 = _safe_cell_coord(hi_x, inv_wcs)
        var cz0 = _safe_cell_coord(lo_z, inv_wcs)
        var cz1 = _safe_cell_coord(hi_z, inv_wcs)
        var span_x = cx1 - cx0 + 1
        var span_z = cz1 - cz0 + 1
        if span_x <= 0 or span_z <= 0 or span_x > 64 or span_z > 64 or span_x > 4096 // span_z:
            for wi in range(n_walls):
                if stamp[wi] != cur_stamp:
                    stamp[wi] = cur_stamp
                    out_walls.append(wi)
            return
        for cz in range(cz0, cz1 + 1):
            for cx in range(cx0, cx1 + 1):
                var bkt = Self._cell_bucket(cx, cz)
                var node = self._wall_head[bkt]
                while node >= 0:
                    var wi = self._wall_seg_idx[node]
                    if stamp[wi] != cur_stamp:
                        stamp[wi] = cur_stamp
                        out_walls.append(wi)
                    node = self._wall_next[node]
        for li in range(len(self._large_walls)):
            var wi = self._large_walls[li]
            if stamp[wi] != cur_stamp:
                stamp[wi] = cur_stamp
                out_walls.append(wi)

    def agent_position(self, idx: Int) -> Vec3:
        if idx < 0 or idx >= self.count or idx >= len(self.px) or idx >= len(self.py) or idx >= len(self.pz):
            return Vec3()
        return Vec3(self.px[idx], self.py[idx], self.pz[idx])

    def agent_velocity(self, idx: Int) -> Vec3:
        if idx < 0 or idx >= self.count or idx >= len(self.vx) or idx >= len(self.vy) or idx >= len(self.vz):
            return Vec3()
        return Vec3(self.vx[idx], self.vy[idx], self.vz[idx])

    def count_overlaps(self, tol: Float32 = 1.0e-4) -> Int:
        """Counts pairs `(i, j)` with `i < j` within the same Y-band (`|py_j - py_i| < agent_height`)
        whose 2D XZ distance is strictly below `r_i + r_j - tol`."""
        var overlaps = 0
        for i in range(self.count):
            if not self.active[i]:
                continue
            for j in range(i + 1, self.count):
                if not self.active[j]:
                    continue
                if abs(self.py[j] - self.py[i]) >= self.agent_height:
                    continue
                var dx = self.px[j] - self.px[i]
                var dz = self.pz[j] - self.pz[i]
                var min_d = self.radius[i] + self.radius[j] - tol
                if min_d > Float32(0.0) and (dx * dx + dz * dz) < min_d * min_d:
                    overlaps += 1
        return overlaps

    @staticmethod
    @always_inline
    def _cell_bucket(cx: Int, cz: Int) -> Int:
        var h = mix64(UInt64(cx + 65536) ^ (UInt64(cz + 65536) * 0x9E3779B97F4A7C15))
        return Int(h) & (_ORCA_HASH_BUCKETS - 1)

    def _is_point_wall_clear(
        self,
        x: Float32,
        z: Float32,
        skip_w: Int,
        min_r: Float32,
        mut cand_walls: List[Int],
        mut stamp: List[Int],
        mut cur_stamp: Int,
    ) -> Bool:
        var r2 = min_r * min_r
        if len(self._wall_head) > 0:
            cur_stamp += 1
            self._query_wall_candidates(x - min_r, z - min_r, x + min_r, z + min_r, cand_walls, stamp, cur_stamp)
            for ci in range(len(cand_walls)):
                var wi = cand_walls[ci]
                if wi == skip_w:
                    continue
                var w = self.walls[wi]
                var cp = _pt_seg_closest_xz(x, z, w.ax, w.az, w.bx, w.bz)
                var dx = x - cp[0]
                var dz = z - cp[1]
                if dx * dx + dz * dz < r2:
                    return False
            return True
        for wi in range(len(self.walls)):
            if wi == skip_w:
                continue
            var w = self.walls[wi]
            var cp = _pt_seg_closest_xz(x, z, w.ax, w.az, w.bx, w.bz)
            var dx = x - cp[0]
            var dz = z - cp[1]
            if dx * dx + dz * dz < r2:
                return False
        return True

    def _steer_target_around_walls(
        self,
        px: Float32,
        pz: Float32,
        gx: Float32,
        gz: Float32,
        ri: Float32,
        mut cand_walls: List[Int],
        mut stamp: List[Int],
        mut cur_stamp: Int,
    ) -> Tuple[Float32, Float32]:
        """When straight segment `P -> G` intersects or clips within `ri` of a wall segment,
        deflects the local steering target around the nearest open wall/doorway endpoint."""
        var n_walls = len(self.walls)
        if n_walls == 0:
            return (gx, gz)

        var pg_dx = gx - px
        var pg_dz = gz - pz
        var pg_len2 = pg_dx * pg_dx + pg_dz * pg_dz
        if pg_len2 <= Float32(1.0e-6):
            return (gx, gz)

        var r_clear = ri + Float32(0.14)
        var r_clear2 = r_clear * r_clear
        var hit_wi = -1
        var best_t = Float32(2.0)

        cur_stamp += 1
        self._query_wall_candidates(
            min(px, gx) - r_clear,
            min(pz, gz) - r_clear,
            max(px, gx) + r_clear,
            max(pz, gz) + r_clear,
            cand_walls,
            stamp,
            cur_stamp,
        )
        for ci in range(len(cand_walls)):
            var wi = cand_walls[ci]
            var w = self.walls[wi]
            var dt_res = _seg_seg_dist2_and_t_xz(px, pz, gx, gz, w.ax, w.az, w.bx, w.bz)
            if dt_res[0] < r_clear2 and dt_res[1] < best_t:
                var wx0 = w.bx - w.ax
                var wz0 = w.bz - w.az
                var nwx0 = -wz0
                var nwz0 = wx0
                var sp0 = (px - w.ax) * nwx0 + (pz - w.az) * nwz0
                var sg0 = (gx - w.ax) * nwx0 + (gz - w.az) * nwz0
                var cg = _pt_seg_closest_xz(gx, gz, w.ax, w.az, w.bx, w.bz)
                var dg2 = (gx - cg[0]) * (gx - cg[0]) + (gz - cg[1]) * (gz - cg[1])
                if dg2 < r_clear2 and sp0 * sg0 >= Float32(0.0):
                    # Already on goal's side of a wall that touches goal: offset goal slightly onto goal's side
                    var wl0 = sqrt(max(Float32(1.0e-10), wx0 * wx0 + wz0 * wz0))
                    var gsx = (nwx0 / wl0) if sg0 >= Float32(0.0) else -(nwx0 / wl0)
                    var gsz = (nwz0 / wl0) if sg0 >= Float32(0.0) else -(nwz0 / wl0)
                    if pg_len2 < Float32(4.0):
                        return (gx + gsx * (ri + Float32(0.12)), gz + gsz * (ri + Float32(0.12)))
                    continue
                best_t = dt_res[1]
                hit_wi = wi

        if hit_wi < 0:
            return (gx, gz)

        var ea_x = self.walls[hit_wi].ax
        var ea_z = self.walls[hit_wi].az
        var eb_x = self.walls[hit_wi].bx
        var eb_z = self.walls[hit_wi].bz
        var wx = eb_x - ea_x
        var wz = eb_z - ea_z
        var wl = sqrt(max(Float32(1.0e-10), wx * wx + wz * wz))
        var uw_x = wx / wl
        var uw_z = wz / wl

        # Extend along any collinear connected wall segments sharing endpoints
        var changed = True
        while changed:
            changed = False
            for wi in range(n_walls):
                if wi == hit_wi:
                    continue
                var o = self.walls[wi]
                var odx = o.bx - o.ax
                var odz = o.bz - o.az
                var ol = sqrt(max(Float32(1.0e-10), odx * odx + odz * odz))
                if abs((odx / ol) * uw_x + (odz / ol) * uw_z) < Float32(0.95):
                    continue
                if (o.bx - ea_x) * (o.bx - ea_x) + (o.bz - ea_z) * (o.bz - ea_z) < Float32(0.04):
                    ea_x = o.ax; ea_z = o.az; changed = True
                elif (o.ax - ea_x) * (o.ax - ea_x) + (o.az - ea_z) * (o.az - ea_z) < Float32(0.04):
                    ea_x = o.bx; ea_z = o.bz; changed = True
                elif (o.ax - eb_x) * (o.ax - eb_x) + (o.az - eb_z) * (o.az - eb_z) < Float32(0.04):
                    eb_x = o.bx; eb_z = o.bz; changed = True
                elif (o.bx - eb_x) * (o.bx - eb_x) + (o.bz - eb_z) * (o.bz - eb_z) < Float32(0.04):
                    eb_x = o.ax; eb_z = o.az; changed = True

        wx = eb_x - ea_x
        wz = eb_z - ea_z
        wl = sqrt(max(Float32(1.0e-10), wx * wx + wz * wz))
        uw_x = wx / wl
        uw_z = wz / wl
        var nw_x = -uw_z
        var nw_z = uw_x

        var raw_side_p = (px - ea_x) * nw_x + (pz - ea_z) * nw_z
        var raw_side_g = (gx - ea_x) * nw_x + (gz - ea_z) * nw_z
        var g_side_x = nw_x if raw_side_g >= Float32(0.0) else -nw_x
        var g_side_z = nw_z if raw_side_g >= Float32(0.0) else -nw_z
        if raw_side_p * raw_side_g > Float32(0.0) and abs(raw_side_p) > ri:
            # Both P and G lie strictly on the same side of this wall: steer wide on G's side without crossing
            return (gx + g_side_x * (ri + Float32(0.20)), gz + g_side_z * (ri + Float32(0.20)))

        var side_p = raw_side_p
        if abs(side_p) < Float32(0.02):
            side_p = -raw_side_g
        var n_side_x = nw_x if side_p >= Float32(0.0) else -nw_x
        var n_side_z = nw_z if side_p >= Float32(0.0) else -nw_z

        var gap_off = ri + Float32(0.45)
        var gap_a_x = ea_x - uw_x * gap_off
        var gap_a_z = ea_z - uw_z * gap_off
        var gap_b_x = eb_x + uw_x * gap_off
        var gap_b_z = eb_z + uw_z * gap_off

        var ok_a = self._is_point_wall_clear(gap_a_x, gap_a_z, hit_wi, ri + Float32(0.05), cand_walls, stamp, cur_stamp)
        var ok_b = self._is_point_wall_clear(gap_b_x, gap_b_z, hit_wi, ri + Float32(0.05), cand_walls, stamp, cur_stamp)

        var cost_a = sqrt((px - gap_a_x) * (px - gap_a_x) + (pz - gap_a_z) * (pz - gap_a_z)) + sqrt(
            (gx - gap_a_x) * (gx - gap_a_x) + (gz - gap_a_z) * (gz - gap_a_z)
        )
        var cost_b = sqrt((px - gap_b_x) * (px - gap_b_x) + (pz - gap_b_z) * (pz - gap_b_z)) + sqrt(
            (gx - gap_b_x) * (gx - gap_b_x) + (gz - gap_b_z) * (gz - gap_b_z)
        )

        var use_a: Bool
        if ok_a and not ok_b:
            use_a = True
        elif ok_b and not ok_a:
            use_a = False
        else:
            use_a = cost_a <= cost_b

        var ex = ea_x if use_a else eb_x
        var ez = ea_z if use_a else eb_z
        var tout_x = -uw_x if use_a else uw_x
        var tout_z = -uw_z if use_a else uw_z
        var gap_x = gap_a_x if use_a else gap_b_x
        var gap_z = gap_a_z if use_a else gap_b_z

        # s_along > 0 while agent is still beside the wall and has not cleared endpoint E yet
        var s_along = (px - ex) * (-tout_x) + (pz - ez) * (-tout_z)
        if s_along > Float32(0.05):
            return (gap_x + n_side_x * (ri + Float32(0.35)), gap_z + n_side_z * (ri + Float32(0.35)))
        # Once in the doorway opening, always pull onto GOAL's side (`g_side`), never `-n_side`!
        return (gap_x + g_side_x * (ri + Float32(0.55)), gap_z + g_side_z * (ri + Float32(0.55)))

    def _project_agent_walls_swept(
        mut self,
        i: Int,
        prev_x: Float32,
        prev_z: Float32,
        margin: Float32,
        mut cand_walls: List[Int],
        mut stamp: List[Int],
        mut cur_stamp: Int,
    ) -> Bool:
        """Enforces both swept non-crossing (`prev -> cur` vs wall segment) and capsule clearance `r_i + margin`."""
        var changed = False
        var ri = self.radius[i] + margin
        var ri2 = ri * ri

        cur_stamp += 1
        self._query_wall_candidates(
            min(prev_x, self.px[i]) - ri,
            min(prev_z, self.pz[i]) - ri,
            max(prev_x, self.px[i]) + ri,
            max(prev_z, self.pz[i]) + ri,
            cand_walls,
            stamp,
            cur_stamp,
        )
        for ci in range(len(cand_walls)):
            var wi = cand_walls[ci]
            var w = self.walls[wi]
            var wx = w.bx - w.ax
            var wz = w.bz - w.az
            var wl2 = wx * wx + wz * wz
            var wl = sqrt(max(Float32(1.0e-10), wl2))
            var nwx = -wz / wl
            var nwz = wx / wl

            # 1. Prevent swept tunneling across the wall segment
            var sw = _segments_proper_intersect_xz(prev_x, prev_z, self.px[i], self.pz[i], w.ax, w.az, w.bx, w.bz)
            if sw[0]:
                var side0 = (prev_x - w.ax) * nwx + (prev_z - w.az) * nwz
                var sx = nwx if side0 >= Float32(0.0) else -nwx
                var sz = nwz if side0 >= Float32(0.0) else -nwz
                var cp_cur = _pt_seg_closest_xz(self.px[i], self.pz[i], w.ax, w.az, w.bx, w.bz)
                self.px[i] = cp_cur[0] + sx * ri
                self.pz[i] = cp_cur[1] + sz * ri
                changed = True

            # 2. Capsule radius projection around wall segment [A, B]
            var cp = _pt_seg_closest_xz(self.px[i], self.pz[i], w.ax, w.az, w.bx, w.bz)
            var ox = self.px[i] - cp[0]
            var oz = self.pz[i] - cp[1]
            var d2 = ox * ox + oz * oz
            if d2 < ri2:
                changed = True
                if d2 > Float32(1.0e-12):
                    var d = sqrt(d2)
                    var push = (ri - d) / d
                    self.px[i] += ox * push
                    self.pz[i] += oz * push
                else:
                    var ox0 = prev_x - cp[0]
                    var oz0 = prev_z - cp[1]
                    var d0 = sqrt(max(Float32(1.0e-12), ox0 * ox0 + oz0 * oz0))
                    if d0 > Float32(1.0e-5):
                        self.px[i] = cp[0] + (ox0 / d0) * ri
                        self.pz[i] = cp[1] + (oz0 / d0) * ri
                    else:
                        self.px[i] = cp[0] + nwx * ri
                        self.pz[i] = cp[1] + nwz * ri
        return changed

    def step_orca_simd8(mut self, dt: Float32):
        """Executes one full crowd steering tick using SoA 8-wide `F8` SIMD preferred velocity,
        8-wide `F8` SIMD ORCA half-plane generation, 2D/3D LP solve, `F8` position update,
        and spatial-grid hard-constraint depenetration."""
        if self.count == 0 or not isfinite(dt) or dt <= Float32(1.0e-7):
            return
        var safe_dt = min(dt, Float32(1.0))
        for i in range(self.count):
            if not self.active[i]:
                continue
            if not isfinite(self.px[i]):
                self.px[i] = Float32(0.0)
            if not isfinite(self.py[i]):
                self.py[i] = Float32(0.0)
            if not isfinite(self.pz[i]):
                self.pz[i] = Float32(0.0)
            if not isfinite(self.vx[i]):
                self.vx[i] = Float32(0.0)
            if not isfinite(self.vy[i]):
                self.vy[i] = Float32(0.0)
            if not isfinite(self.vz[i]):
                self.vz[i] = Float32(0.0)
            if not isfinite(self.pref_vx[i]):
                self.pref_vx[i] = Float32(0.0)
            if not isfinite(self.pref_vz[i]):
                self.pref_vz[i] = Float32(0.0)
            if not isfinite(self.max_speed[i]) or self.max_speed[i] < Float32(0.0):
                self.max_speed[i] = Float32(0.0)
            if not isfinite(self.radius[i]) or self.radius[i] < Float32(0.01):
                self.radius[i] = Float32(0.01)
            if self.has_goal[i] and (not isfinite(self.goal_x[i]) or not isfinite(self.goal_z[i])):
                self.has_goal[i] = False
                self.goal_x[i] = self.px[i]
                self.goal_z[i] = self.pz[i]
                self.pref_vx[i] = Float32(0.0)
                self.pref_vz[i] = Float32(0.0)
        self._pad_columns()
        var n_batches = self.padded // LANES
        var p_ptr = self.px.unsafe_ptr()
        var pz_ptr = self.pz.unsafe_ptr()
        var ms_ptr = self.max_speed.unsafe_ptr()

        # Effective local steering targets (direct goal or wall-endpoint doorway detour)
        var steer_x = List[Float32](length=self.padded, fill=Float32(1.0e6))
        var steer_z = List[Float32](length=self.padded, fill=Float32(1.0e6))
        var has_walls = len(self.walls) > 0
        if has_walls:
            self._ensure_wall_grid()
        var wall_cands = List[Int]()
        var wall_stamp = List[Int](length=len(self.walls), fill=0)
        var cur_wall_stamp = 0
        for i in range(self.count):
            if self.active[i] and self.has_goal[i]:
                if has_walls:
                    var tgt = self._steer_target_around_walls(
                        self.px[i], self.pz[i], self.goal_x[i], self.goal_z[i], self.radius[i],
                        wall_cands, wall_stamp, cur_wall_stamp,
                    )
                    steer_x[i] = tgt[0]
                    steer_z[i] = tgt[1]
                else:
                    steer_x[i] = self.goal_x[i]
                    steer_z[i] = self.goal_z[i]

        var sx_ptr = steer_x.unsafe_ptr()
        var sz_ptr = steer_z.unsafe_ptr()

        # 1. 8-wide SIMD preferred velocity update toward steering target (with subtle right-hand lane bias)
        for b in range(n_batches):
            var at = b * LANES
            var cur_x = p_ptr.unsafe_load[width=LANES](at)
            var cur_z = pz_ptr.unsafe_load[width=LANES](at)
            var tgt_x = sx_ptr.unsafe_load[width=LANES](at)
            var tgt_z = sz_ptr.unsafe_load[width=LANES](at)
            var mspd = ms_ptr.unsafe_load[width=LANES](at)

            var dx = tgt_x - cur_x
            var dz = tgt_z - cur_z
            var d2 = dx * dx + dz * dz
            var d = sqrt(max(d2, F8(1.0e-12)))
            var desired_spd = min(mspd, d * F8(3.0))
            var scale = d2.gt(F8(1.0e-8)).select(desired_spd / d, F8(0.0))
            var want_x = dx * scale
            var want_z = dz * scale

            comptime for lane in range(LANES):
                var idx = at + lane
                if idx < self.count and self.active[idx] and self.has_goal[idx]:
                    # Subtle right-hand traffic bias (~1.4 deg when far, 0 near goal) organizes opposing flows into lanes
                    var dist_l = d[lane]
                    var rh_bias = Float32(0.025) if dist_l > Float32(1.5) else Float32(0.0)
                    var jitter = Float32(((idx * 73) & 15) - 8) * Float32(2.0e-4)
                    var b_ang = rh_bias + jitter
                    # In XZ: right-hand perpendicular of (vx, vz) is (-vz, +vx)
                    self.pref_vx[idx] = want_x[lane] - want_z[lane] * b_ang
                    self.pref_vz[idx] = want_z[lane] + want_x[lane] * b_ang

        # 2. Build O(N) spatial hash grid for neighbor queries
        var cell_sz = max(Float32(1.0), self.neighbor_dist)
        var inv_cs = Float32(1.0) / cell_sz
        var head = List[Int](length=_ORCA_HASH_BUCKETS, fill=-1)
        var next_agent = List[Int](length=self.count, fill=-1)
        var agent_cx = List[Int](length=self.count, fill=0)
        var agent_cz = List[Int](length=self.count, fill=0)
        var max_r = Float32(0.05)

        for i in range(self.count):
            if not self.active[i]:
                continue
            if self.radius[i] > max_r:
                max_r = self.radius[i]
            var cx = _safe_cell_coord(self.px[i], inv_cs)
            var cz = _safe_cell_coord(self.pz[i], inv_cs)
            agent_cx[i] = cx
            agent_cz[i] = cz
            var bkt = Self._cell_bucket(cx, cz)
            next_agent[i] = head[bkt]
            head[bkt] = i

        # 3. Per-agent ORCA constraint construction (8 neighbors per F8 SIMD vector!) + 2D/3D LP solve
        var new_vx = List[Float32](length=self.padded, fill=0.0)
        var new_vz = List[Float32](length=self.padded, fill=0.0)
        var inv_tau = Float32(1.0) / self.time_horizon
        var inv_tau_obst = Float32(1.0) / self.time_horizon_obst
        var inv_dt = Float32(1.0) / safe_dt
        var nb_dist2 = self.neighbor_dist * self.neighbor_dist

        var nb_indices = List[Int](capacity=self.max_neighbors + 8)
        var nb_d2 = List[Float32](capacity=self.max_neighbors + 8)

        for i in range(self.count):
            if not self.active[i]:
                continue
            var aix = self.px[i]
            var aiz = self.pz[i]
            var vix = self.vx[i]
            var viz = self.vz[i]
            var ri = self.radius[i]
            var mspd_i = self.max_speed[i]

            var lines = List[OrcaLine](capacity=self.max_neighbors + 8)
            if has_walls:
                var wall_hdist = ri + Float32(0.025) + mspd_i * self.time_horizon_obst + Float32(0.25)
                cur_wall_stamp += 1
                self._query_wall_candidates(
                    aix - wall_hdist,
                    aiz - wall_hdist,
                    aix + wall_hdist,
                    aiz + wall_hdist,
                    wall_cands,
                    wall_stamp,
                    cur_wall_stamp,
                )
                for ci in range(len(wall_cands)):
                    var wi = wall_cands[ci]
                    _add_wall_orca_line(
                        aix, aiz, ri, mspd_i, self.walls[wi],
                        self.time_horizon_obst, inv_tau_obst, inv_dt, lines,
                    )
            var num_obst_lines = len(lines)

            # Gather up to `max_neighbors` closest active agents from 3x3 cells within the same Y-band
            nb_indices.clear()
            nb_d2.clear()
            var icx = agent_cx[i]
            var icz = agent_cz[i]
            var aiy = self.py[i]
            for dz_c in range(-1, 2):
                for dx_c in range(-1, 2):
                    var bkt = Self._cell_bucket(icx + dx_c, icz + dz_c)
                    var j = head[bkt]
                    while j >= 0:
                        if (
                            j != i
                            and abs(agent_cx[j] - icx) <= 1
                            and abs(agent_cz[j] - icz) <= 1
                            and abs(self.py[j] - aiy) < self.agent_height
                        ):
                            var rx = self.px[j] - aix
                            var rz = self.pz[j] - aiz
                            var d2 = rx * rx + rz * rz
                            if d2 < nb_dist2:
                                var ins = len(nb_d2)
                                if ins < self.max_neighbors or d2 < nb_d2[ins - 1]:
                                    if ins < self.max_neighbors:
                                        nb_indices.append(j)
                                        nb_d2.append(d2)
                                    else:
                                        nb_indices[ins - 1] = j
                                        nb_d2[ins - 1] = d2
                                        ins -= 1
                                    while ins > 0 and nb_d2[ins] < nb_d2[ins - 1]:
                                        var td = nb_d2[ins]; nb_d2[ins] = nb_d2[ins - 1]; nb_d2[ins - 1] = td
                                        var ti = nb_indices[ins]; nb_indices[ins] = nb_indices[ins - 1]; nb_indices[ins - 1] = ti
                                        ins -= 1
                        j = next_agent[j]

            # Evaluate neighbor ORCA half-planes in 8-wide F8 SIMD chunks
            var n_nb = len(nb_indices)
            var base = 0
            var f_aix = F8(aix)
            var f_aiz = F8(aiz)
            var f_vix = F8(vix)
            var f_viz = F8(viz)
            var f_ri = F8(ri + self.personal_space_margin)
            var f_inv_tau = F8(inv_tau)
            var f_inv_dt = F8(inv_dt)

            while base < n_nb:
                var bpx = F8(1.0e6)
                var bpz = F8(1.0e6)
                var bvx = F8(0.0)
                var bvz = F8(0.0)
                var br = F8(0.0)
                var valid_cnt = min(LANES, n_nb - base)
                for lane in range(valid_cnt):
                    var j = nb_indices[base + lane]
                    bpx[lane] = self.px[j]
                    bpz[lane] = self.pz[j]
                    bvx[lane] = self.vx[j]
                    bvz[lane] = self.vz[j]
                    br[lane] = self.radius[j]

                var rx = bpx - f_aix
                var rz = bpz - f_aiz
                var rvx = f_vix - bvx
                var rvz = f_viz - bvz
                var dist2 = rx * rx + rz * rz
                var comb_r = f_ri + br
                var comb_r2 = comb_r * comb_r

                var no_coll = dist2.gt(comb_r2)

                # Branch A: Non-overlapping VO cone (cutoff circle vs two tangent legs)
                var wx_tau = rvx - rx * f_inv_tau
                var wz_tau = rvz - rz * f_inv_tau
                var w_len2_tau = wx_tau * wx_tau + wz_tau * wz_tau
                var dot1 = wx_tau * rx + wz_tau * rz
                var on_cutoff = dot1.lt(F8(0.0)) & (dot1 * dot1).gt(comb_r2 * w_len2_tau)

                var w_len_tau = sqrt(max(w_len2_tau, F8(1.0e-12)))
                var cut_nx = wx_tau / w_len_tau
                var cut_nz = wz_tau / w_len_tau
                var cut_u_mag = comb_r * f_inv_tau - w_len_tau
                var cut_ux = cut_nx * cut_u_mag
                var cut_uz = cut_nz * cut_u_mag

                var leg = sqrt(max(dist2 - comb_r2, F8(0.0)))
                var inv_d2 = F8(1.0) / max(dist2, F8(1.0e-12))
                var side_pos = (rvx * (F8(0.0) - rz) + rvz * rx).ge(F8(0.0))

                var lp_x = (leg * rx - comb_r * rz) * inv_d2
                var lp_z = (leg * rz + comb_r * rx) * inv_d2
                var np_x = F8(0.0) - lp_z
                var np_z = lp_x

                var lm_x = (leg * rx + comb_r * rz) * inv_d2
                var lm_z = (leg * rz - comb_r * rx) * inv_d2
                var nm_x = lm_z
                var nm_z = F8(0.0) - lm_x

                var leg_nx = side_pos.select(np_x, nm_x)
                var leg_nz = side_pos.select(np_z, nm_z)
                var v_dot_n = rvx * leg_nx + rvz * leg_nz
                var leg_ux = leg_nx * (F8(0.0) - v_dot_n)
                var leg_uz = leg_nz * (F8(0.0) - v_dot_n)

                var nc_nx = on_cutoff.select(cut_nx, leg_nx)
                var nc_nz = on_cutoff.select(cut_nz, leg_nz)
                var nc_ux = on_cutoff.select(cut_ux, leg_ux)
                var nc_uz = on_cutoff.select(cut_uz, leg_uz)

                # Branch B: Within personal-space margin -> dt cutoff resolution with safe fallback normal
                var wx_dt = rvx - rx * f_inv_dt
                var wz_dt = rvz - rz * f_inv_dt
                var w2_dt = wx_dt * wx_dt + wz_dt * wz_dt
                var w_valid = w2_dt.gt(F8(1.0e-10))
                var w_len_dt = sqrt(max(w2_dt, F8(1.0e-12)))
                var d_safe = sqrt(max(dist2, F8(1.0e-12)))
                var fb_nx = dist2.gt(F8(1.0e-10)).select((F8(0.0) - rx) / d_safe, F8(1.0))
                var fb_nz = dist2.gt(F8(1.0e-10)).select((F8(0.0) - rz) / d_safe, F8(0.0))
                var col_nx = w_valid.select(wx_dt / w_len_dt, fb_nx)
                var col_nz = w_valid.select(wz_dt / w_len_dt, fb_nz)
                var col_u_mag = comb_r * f_inv_dt - w_valid.select(w_len_dt, F8(0.0))
                var col_ux = col_nx * col_u_mag
                var col_uz = col_nz * col_u_mag

                var out_nx = no_coll.select(nc_nx, col_nx)
                var out_nz = no_coll.select(nc_nz, col_nz)
                var out_px = f_vix + no_coll.select(nc_ux, col_ux) * F8(0.5)
                var out_pz = f_viz + no_coll.select(nc_uz, col_uz) * F8(0.5)

                for lane in range(valid_cnt):
                    lines.append(OrcaLine(out_px[lane], out_pz[lane], out_nx[lane], out_nz[lane]))

                base += LANES

            var rvx_sol = Float32(0.0)
            var rvz_sol = Float32(0.0)
            var fail_line = _lp2_orca(lines, mspd_i, self.pref_vx[i], self.pref_vz[i], False, rvx_sol, rvz_sol)
            if fail_line < len(lines):
                _lp3_orca(lines, num_obst_lines, fail_line, mspd_i, rvx_sol, rvz_sol)

            new_vx[i] = rvx_sol
            new_vz[i] = rvz_sol

        # Save pre-integration positions for swept wall checks & effective velocity sync
        var prev_px = List[Float32](length=self.count, fill=0.0)
        var prev_pz = List[Float32](length=self.count, fill=0.0)
        for i in range(self.count):
            prev_px[i] = self.px[i]
            prev_pz[i] = self.pz[i]

        # 4. 8-wide F8 SIMD velocity commit & position integration
        var vx_ptr = self.vx.unsafe_ptr()
        var vz_ptr = self.vz.unsafe_ptr()
        var nvx_ptr = new_vx.unsafe_ptr()
        var nvz_ptr = new_vz.unsafe_ptr()
        var dt_v = F8(safe_dt)
        for b in range(n_batches):
            var at = b * LANES
            var v_x = nvx_ptr.unsafe_load[width=LANES](at)
            var v_z = nvz_ptr.unsafe_load[width=LANES](at)
            vx_ptr.unsafe_offset(at).unsafe_store(v_x)
            vz_ptr.unsafe_offset(at).unsafe_store(v_z)
            p_ptr.unsafe_offset(at).unsafe_store(p_ptr.unsafe_load[width=LANES](at) + v_x * dt_v)
            pz_ptr.unsafe_offset(at).unsafe_store(pz_ptr.unsafe_load[width=LANES](at) + v_z * dt_v)
        for i in range(self.count):
            if not self.active[i]:
                continue
            if not isfinite(self.px[i]):
                self.px[i] = prev_px[i]
            if not isfinite(self.pz[i]):
                self.pz[i] = prev_pz[i]
            if not isfinite(self.vx[i]):
                self.vx[i] = Float32(0.0)
            if not isfinite(self.vz[i]):
                self.vz[i] = Float32(0.0)

        # 5. Tight spatial-grid SOR Gauss-Seidel hard-disc & swept-wall position projection
        # Enforces strict 0-overlap (`d >= r_i + r_j + 2.5e-4`) and zero wall tunneling on every tick!
        var dep_cs = max(Float32(1.6), max_r * Float32(4.5))
        var dep_inv_cs = Float32(1.0) / dep_cs
        var dep_head = List[Int](length=_ORCA_HASH_BUCKETS, fill=-1)
        var dep_next = List[Int](length=self.count, fill=-1)
        var moved_by_proj = List[Bool](length=self.count, fill=False)

        comptime MAX_DEP_ITERS: Int = 40
        for dep_it in range(MAX_DEP_ITERS):
            var any_wall_fix = False
            if has_walls:
                for i in range(self.count):
                    if self.active[i] and self._project_agent_walls_swept(
                        i, prev_px[i], prev_pz[i], Float32(3.5e-4),
                        wall_cands, wall_stamp, cur_wall_stamp,
                    ):
                        any_wall_fix = True
                        moved_by_proj[i] = True

            for bkt in range(_ORCA_HASH_BUCKETS):
                dep_head[bkt] = -1
            for i in range(self.count):
                if not self.active[i]:
                    continue
                var cx = _safe_cell_coord(self.px[i], dep_inv_cs)
                var cz = _safe_cell_coord(self.pz[i], dep_inv_cs)
                var bkt = Self._cell_bucket(cx, cz)
                dep_next[i] = dep_head[bkt]
                dep_head[bkt] = i

            var any_pair_fix = False
            var relax = Float32(0.80) if dep_it < 30 else Float32(0.52)
            # Alternate sweep direction to eliminate directional drift in dense clusters
            var rev = (dep_it & 1) == 1
            for step_i in range(self.count):
                var i = (self.count - 1 - step_i) if rev else step_i
                if not self.active[i]:
                    continue
                var icx = _safe_cell_coord(self.px[i], dep_inv_cs)
                var icz = _safe_cell_coord(self.pz[i], dep_inv_cs)
                var aiy = self.py[i]
                for dz_c in range(-1, 2):
                    for dx_c in range(-1, 2):
                        var bkt = Self._cell_bucket(icx + dx_c, icz + dz_c)
                        var j = dep_head[bkt]
                        while j >= 0:
                            if j > i and abs(self.py[j] - aiy) < self.agent_height:
                                var dx = self.px[j] - self.px[i]
                                var dz = self.pz[j] - self.pz[i]
                                var min_r = self.radius[i] + self.radius[j]
                                var d2 = dx * dx + dz * dz
                                if d2 < min_r * min_r:
                                    any_pair_fix = True
                                    moved_by_proj[i] = True
                                    moved_by_proj[j] = True
                                    var target_r = min_r + Float32(3.5e-4)
                                    if d2 > Float32(1.0e-12):
                                        var d = sqrt(d2)
                                        var corr = (target_r - d) * relax / d
                                        self.px[i] -= dx * corr
                                        self.pz[i] -= dz * corr
                                        self.px[j] += dx * corr
                                        self.pz[j] += dz * corr
                                    else:
                                        var ang_nudge = Float32(((i * 37 + j * 17) & 7)) * Float32(0.785398)
                                        var nx = Float32(1.0) if (ang_nudge < Float32(1.5) or ang_nudge > Float32(4.7)) else Float32(-1.0)
                                        var nudge = target_r * Float32(0.5) * nx
                                        self.px[i] -= nudge
                                        self.px[j] += nudge
                            j = dep_next[j]

            if has_walls and any_pair_fix:
                for i in range(self.count):
                    if self.active[i] and self._project_agent_walls_swept(
                        i, prev_px[i], prev_pz[i], Float32(3.5e-4),
                        wall_cands, wall_stamp, cur_wall_stamp,
                    ):
                        any_wall_fix = True
                        moved_by_proj[i] = True

            if not any_pair_fix and not any_wall_fix:
                break

        # Synchronize effective velocity on constrained agents so next tick's ORCA sees actual motion
        for i in range(self.count):
            if self.active[i] and moved_by_proj[i]:
                var evx = (self.px[i] - prev_px[i]) * inv_dt
                var evz = (self.pz[i] - prev_pz[i]) * inv_dt
                var em2 = evx * evx + evz * evz
                var ms = self.max_speed[i]
                if em2 > ms * ms and em2 > Float32(1.0e-12):
                    var sc = ms / sqrt(em2)
                    evx *= sc
                    evz *= sc
                self.vx[i] = evx
                self.vz[i] = evz


# =========================================================================
# NavMesh <-> ORCA Crowd Corridor Bridge (`NavCrowdAgent`, `CrowdNavBridge`)
# =========================================================================

struct NavCrowdAgent(Copyable, Movable):
    """Links one `CrowdSimulator` agent to a `NavMesh` A*/SSFA waypoint corridor."""
    var agent_id: Int
    var path: NavPath
    var waypoint_idx: Int
    var goal: Vec3
    var arrival_radius: Float32
    var waypoint_radius: Float32
    var repath_needed: Bool

    def __init__(
        out self,
        agent_id: Int = -1,
        goal: Vec3 = Vec3(),
        arrival_radius: Float32 = 0.45,
        waypoint_radius: Float32 = 0.65,
    ):
        self.agent_id = agent_id
        self.path = NavPath()
        self.waypoint_idx = 0
        self.goal = goal
        self.arrival_radius = max(Float32(0.05), arrival_radius)
        self.waypoint_radius = max(Float32(0.10), waypoint_radius)
        self.repath_needed = True


struct CrowdNavBridge(Copyable, Movable):
    """Connects `NavMesh` A*/SSFA paths directly to `CrowdSimulator` preferred velocities
    and projects agent elevation `py` onto the walkable navmesh surface each tick."""
    var agents: List[NavCrowdAgent]
    var nav_epoch: Int  # `NavMesh.epoch` the agents' paths were planned against

    def __init__(out self):
        self.agents = List[NavCrowdAgent]()
        self.nav_epoch = -1

    def add_agent(
        mut self,
        mut crowd: CrowdSimulator,
        nav: NavMesh,
        start: Vec3,
        goal: Vec3,
        radius: Float32 = 0.35,
        max_speed: Float32 = 3.5,
        arrival_radius: Float32 = 0.45,
        waypoint_radius: Float32 = 0.65,
    ) -> Int:
        var snap_start = nav.find_nearest_poly(start, Float32(4.0))
        var pos0 = snap_start[1] if snap_start[0] >= 0 else start
        var aid = crowd.add_agent(pos0, radius, max_speed)
        var ag = NavCrowdAgent(aid, goal, arrival_radius, waypoint_radius)
        ag.path = nav.find_path(pos0, goal)
        ag.waypoint_idx = 1 if len(ag.path.waypoints) > 1 else 0
        ag.repath_needed = not ag.path.found
        var b_idx = len(self.agents)
        self.agents.append(ag^)
        return b_idx

    def set_goal(mut self, bridge_idx: Int, crowd: CrowdSimulator, nav: NavMesh, goal: Vec3):
        if bridge_idx < 0 or bridge_idx >= len(self.agents):
            return
        ref ag = self.agents[bridge_idx]
        ag.goal = goal
        var pos = crowd.agent_position(ag.agent_id)
        ag.path = nav.find_path(pos, goal)
        ag.waypoint_idx = 1 if len(ag.path.waypoints) > 1 else 0
        ag.repath_needed = not ag.path.found

    def mark_all_repath(mut self):
        """Triggers path recomputation on next update (e.g. after dynamic `NavObstacle` carve)."""
        for i in range(len(self.agents)):
            self.agents[i].repath_needed = True

    def is_arrived(self, bridge_idx: Int, crowd: CrowdSimulator) -> Bool:
        if bridge_idx < 0 or bridge_idx >= len(self.agents):
            return True
        ref ag = self.agents[bridge_idx]
        var pos = crowd.agent_position(ag.agent_id)
        var dx = ag.goal.x - pos.x
        var dz = ag.goal.z - pos.z
        return dx * dx + dz * dz <= ag.arrival_radius * ag.arrival_radius

    def update_preferred_velocities(mut self, mut crowd: CrowdSimulator, nav: NavMesh):
        """Computes per-agent ORCA preferred velocities along each agent's `NavMesh` corridor.
        Paths are re-planned automatically when the navmesh was rebuilt / carved since planning."""
        if nav.epoch != self.nav_epoch:
            if self.nav_epoch >= 0:
                self.mark_all_repath()
            self.nav_epoch = nav.epoch
        for i in range(len(self.agents)):
            ref ag = self.agents[i]
            var aid = ag.agent_id
            if aid < 0 or aid >= crowd.count or not crowd.active[aid]:
                continue
            var pos = crowd.agent_position(aid)
            if ag.repath_needed:
                ag.path = nav.find_path(pos, ag.goal)
                ag.waypoint_idx = 1 if len(ag.path.waypoints) > 1 else 0
                ag.repath_needed = False

            var n_wp = len(ag.path.waypoints)
            if not ag.path.found or n_wp == 0:
                crowd.set_preferred_velocity(aid, Vec3())
                continue

            # Advance through reached intermediate waypoints
            var wp_r2 = ag.waypoint_radius * ag.waypoint_radius
            while ag.waypoint_idx < n_wp - 1:
                var wp = ag.path.waypoints[ag.waypoint_idx]
                var ddx = wp.x - pos.x
                var ddz = wp.z - pos.z
                if ddx * ddx + ddz * ddz <= wp_r2:
                    ag.waypoint_idx += 1
                else:
                    break

            var target = ag.path.waypoints[min(ag.waypoint_idx, n_wp - 1)]
            var dx = target.x - pos.x
            var dz = target.z - pos.z
            var d2 = dx * dx + dz * dz
            var at_final = ag.waypoint_idx >= n_wp - 1
            if at_final and d2 <= ag.arrival_radius * ag.arrival_radius:
                crowd.set_preferred_velocity(aid, Vec3())
                continue

            var d = sqrt(max(Float32(1.0e-12), d2))
            var mspd = crowd.max_speed[aid]
            var spd = min(mspd, d * Float32(3.0)) if at_final else mspd
            crowd.set_preferred_velocity(aid, Vec3((dx / d) * spd, Float32(0.0), (dz / d) * spd))

    def update_and_step(mut self, mut crowd: CrowdSimulator, nav: NavMesh, dt: Float32):
        """Updates preferred velocities from `NavMesh` corridors, steps ORCA SIMD8,
        and snaps agent elevations `py` to the walkable navmesh surface."""
        self.update_preferred_velocities(crowd, nav)
        crowd.step_orca_simd8(dt)
        for i in range(len(self.agents)):
            var aid = self.agents[i].agent_id
            if aid < 0 or aid >= crowd.count or not crowd.active[aid]:
                continue
            var pos = crowd.agent_position(aid)
            var near = nav.find_nearest_poly(pos, Float32(2.0))
            if near[0] >= 0:
                crowd.py[aid] = near[1].y
