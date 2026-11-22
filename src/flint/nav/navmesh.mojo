"""Tiled Recast/Detour Voxel Navigation Mesh, Sub-0.25ms Dynamic Obstacle Carving,
Binary-Heap A* & Taut SSFA + Visibility Pathfinding (`flint.nav.navmesh`, v5).

Implements:
1. Tiled Recast voxel pipeline `NavMesh.build_from_physics(world, cfg)`:
   - Symmetric grid bounds & Sutherland-Hodgman cell-clipped triangle span rasterization.
   - Walkable slope, standing `walkable_height` headroom, and `walkable_climb` ledge filtering.
   - Exact standing-wall `walkable_radius` cell-AABB clearance erosion + 8-neighbor open-ledge
     Chamfer distance erosion (`base_area` vs live `area`).
   - Spatial `(tx, tz)` `NavTile` partitioning with cached per-tile span lists, unlimited `Int32`
     region IDs, greedy maximal coplanar convex rectangle decomposition, and incremental swap-pop
     polygon & border-portal splicing for `< 0.25 ms` dynamic `NavObstacle` carving on large maps.
2. `OffMeshLink`s (`OFFMESH_JUMP`, `OFFMESH_DROP`, `OFFMESH_CLIMB`) integrated into A* search.
3. Deterministic `MinHeap` binary-heap A* (`O(E log V)`) with portal-projected entry anchors +
   Mikko Mononen's Simple Stupid Funnel Algorithm (SSFA) + exact `NavPoly` portal-graph line-of-sight
   string-pulling producing taut Euclidean shortest paths.
"""
from std.collections import List
from std.math import sqrt, abs, min, max, floor
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_cos
from flint.core.detsort import MinHeap
from flint.core.serial import float32_bits, BlobWriter, BlobReader, fourcc
from flint.physics.narrowphase import get_box_axes
from flint.physics.shapes_v5 import (
    COMPOUND_CHILD_BOX,
    COMPOUND_CHILD_CONVEX,
)
from flint.physics.world import (
    PhysicsWorld,
    BODY_MOTION_DYNAMIC,
    BODY_SHAPE_BOX,
    BODY_SHAPE_CONVEX_HULL,
    BODY_SHAPE_MESH_BVH8,
    BODY_SHAPE_HEIGHTFIELD,
    BODY_SHAPE_COMPOUND,
)

comptime NAV_DEG_TO_RAD: Float32 = 0.01745329252
comptime NAVMESH_BLOB_TAG: UInt32 = fourcc("NAVM")
comptime NAVMESH_BLOB_VERSION: Int = 1

comptime OFFMESH_JUMP: UInt8 = 0
comptime OFFMESH_DROP: UInt8 = 1
comptime OFFMESH_CLIMB: UInt8 = 2

comptime OBSTACLE_BOX: UInt8 = 0
comptime OBSTACLE_CYLINDER: UInt8 = 1


@always_inline
def _triarea2_xz(a: Vec3, b: Vec3, c: Vec3) -> Float32:
    var ax = b.x - a.x
    var az = b.z - a.z
    var bx = c.x - a.x
    var bz = c.z - a.z
    return bx * az - ax * bz


@always_inline
def _pt_seg_dist2_xz(px: Float32, pz: Float32, ax: Float32, az: Float32, bx: Float32, bz: Float32) -> Float32:
    var abx = bx - ax
    var abz = bz - az
    var l2 = abx * abx + abz * abz
    var t = Float32(0.0)
    if l2 > Float32(1.0e-10):
        t = max(Float32(0.0), min(Float32(1.0), ((px - ax) * abx + (pz - az) * abz) / l2))
    var dx = px - (ax + abx * t)
    var dz = pz - (az + abz * t)
    return dx * dx + dz * dz


@always_inline
def _pt_aabb_dist2_xz(px: Float32, pz: Float32, x0: Float32, z0: Float32, x1: Float32, z1: Float32) -> Float32:
    var cx = max(x0, min(x1, px))
    var cz = max(z0, min(z1, pz))
    var dx = px - cx
    var dz = pz - cz
    return dx * dx + dz * dz


def _seg_intersects_aabb_xz(
    ax: Float32, az: Float32, bx: Float32, bz: Float32,
    x0: Float32, z0: Float32, x1: Float32, z1: Float32,
) -> Bool:
    var dx = bx - ax
    var dz = bz - az
    var tmin = Float32(0.0)
    var tmax = Float32(1.0)
    if abs(dx) <= Float32(1.0e-9):
        if ax < x0 or ax > x1:
            return False
    else:
        var inv_dx = Float32(1.0) / dx
        var tx1 = (x0 - ax) * inv_dx
        var tx2 = (x1 - ax) * inv_dx
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))
        if tmin > tmax:
            return False

    if abs(dz) <= Float32(1.0e-9):
        if az < z0 or az > z1:
            return False
    else:
        var inv_dz = Float32(1.0) / dz
        var tz1 = (z0 - az) * inv_dz
        var tz2 = (z1 - az) * inv_dz
        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))
        if tmin > tmax:
            return False
    return True


def _aabb_poly_dist2_xz(
    x0: Float32, z0: Float32, x1: Float32, z1: Float32, poly: List[Vec3]
) -> Float32:
    """Exact squared 2D XZ distance between cell AABB `[x0, x1] x [z0, z1]` and convex `poly`."""
    var n = len(poly)
    if n == 0:
        return Float32(1.0e30)
    if n == 1:
        return _pt_aabb_dist2_xz(poly[0].x, poly[0].z, x0, z0, x1, z1)

    for i in range(n):
        var v = poly[i]
        if v.x >= x0 and v.x <= x1 and v.z >= z0 and v.z <= z1:
            return Float32(0.0)

    if n >= 3:
        var cx = (x0 + x1) * Float32(0.5)
        var cz = (z0 + z1) * Float32(0.5)
        var p_c = Vec3(cx, 0.0, cz)
        var pos_c = 0
        var neg_c = 0
        var area2 = Float32(0.0)
        for i in range(n):
            var va = poly[i]
            var vb = poly[(i + 1) % n]
            area2 += va.x * vb.z - vb.x * va.z
            var tr = _triarea2_xz(va, vb, p_c)
            if tr > Float32(1.0e-6):
                pos_c += 1
            elif tr < Float32(-1.0e-6):
                neg_c += 1
        if abs(area2) > Float32(1.0e-6) and (pos_c == 0 or neg_c == 0):
            return Float32(0.0)

    var best = Float32(1.0e30)
    var edge_cnt = 1 if n == 2 else n
    for i in range(edge_cnt):
        var a = poly[i]
        var b = poly[(i + 1) % n]
        if _seg_intersects_aabb_xz(a.x, a.z, b.x, b.z, x0, z0, x1, z1):
            return Float32(0.0)
        best = min(best, _pt_aabb_dist2_xz(a.x, a.z, x0, z0, x1, z1))
        best = min(best, _pt_aabb_dist2_xz(b.x, b.z, x0, z0, x1, z1))
        best = min(best, _pt_seg_dist2_xz(x0, z0, a.x, a.z, b.x, b.z))
        best = min(best, _pt_seg_dist2_xz(x1, z0, a.x, a.z, b.x, b.z))
        best = min(best, _pt_seg_dist2_xz(x1, z1, a.x, a.z, b.x, b.z))
        best = min(best, _pt_seg_dist2_xz(x0, z1, a.x, a.z, b.x, b.z))
    return best


@fieldwise_init
struct NavMeshConfig(Copyable, ImplicitlyCopyable, Movable):
    var cell_size: Float32
    var cell_height: Float32
    var walkable_slope_deg: Float32
    var walkable_height: Float32
    var walkable_climb: Float32
    var walkable_radius: Float32
    var max_edge_len: Float32
    var max_simplification_error: Float32
    var min_region_cells: Int
    var tile_size: Int
    var max_verts_per_poly: Int

    def __init__(out self):
        self.cell_size = 0.25
        self.cell_height = 0.10
        self.walkable_slope_deg = 46.0
        self.walkable_height = 1.80
        self.walkable_climb = 0.40
        self.walkable_radius = 0.35
        self.max_edge_len = 6.0
        self.max_simplification_error = 0.35
        self.min_region_cells = 3
        self.tile_size = 16
        self.max_verts_per_poly = 6


@fieldwise_init
struct _SolidSpan(Copyable, ImplicitlyCopyable, Movable):
    var smin: UInt16
    var smax: UInt16
    var walkable: Bool


@fieldwise_init
struct CompactSpan(Copyable, ImplicitlyCopyable, Movable):
    var x: Int32
    var z: Int32
    var y: UInt16
    var ceiling: UInt16
    var base_area: UInt8
    var area: UInt8
    var dist: UInt16
    var reg: Int32
    var n0: Int32
    var n1: Int32
    var n2: Int32
    var n3: Int32

    def __init__(out self, x: Int, z: Int, y: UInt16, ceiling: UInt16, walkable: Bool):
        self.x = Int32(x)
        self.z = Int32(z)
        self.y = y
        self.ceiling = ceiling
        var a = UInt8(1) if walkable else UInt8(0)
        self.base_area = a
        self.area = a
        self.dist = 0
        self.reg = 0
        self.n0 = -1
        self.n1 = -1
        self.n2 = -1
        self.n3 = -1

    @always_inline
    def get_neighbor(self, dir: Int) -> Int:
        if dir == 0: return Int(self.n0)
        if dir == 1: return Int(self.n1)
        if dir == 2: return Int(self.n2)
        return Int(self.n3)

    @always_inline
    def set_neighbor(mut self, dir: Int, val: Int):
        if dir == 0: self.n0 = Int32(val)
        elif dir == 1: self.n1 = Int32(val)
        elif dir == 2: self.n2 = Int32(val)
        else: self.n3 = Int32(val)


struct CompactHeightfield(Copyable, Movable):
    """Cached compressed open-span heightfield supporting `< 0.25 ms` tiled dynamic obstacle carving."""
    var width: Int
    var depth: Int
    var origin: Vec3
    var cell_size: Float32
    var cell_height: Float32
    var col_first: List[Int32]
    var col_count: List[UInt16]
    var spans: List[CompactSpan]

    def __init__(out self):
        self.width = 0
        self.depth = 0
        self.origin = Vec3()
        self.cell_size = 0.25
        self.cell_height = 0.10
        self.col_first = List[Int32]()
        self.col_count = List[UInt16]()
        self.spans = List[CompactSpan]()


@fieldwise_init
struct NavObstacle(Copyable, ImplicitlyCopyable, Movable):
    var id: Int
    var shape: UInt8
    var pos: Vec3
    var rot: Quat
    var half_extents: Vec3
    var radius: Float32
    var half_height: Float32
    var active: Bool

    @staticmethod
    def box(id: Int, pos: Vec3, half_extents: Vec3, rot: Quat = Quat()) -> Self:
        return Self(id, OBSTACLE_BOX, pos, rot.normalized(), half_extents, half_extents.length(), half_extents.y, True)

    @staticmethod
    def cylinder(id: Int, pos: Vec3, radius: Float32, half_height: Float32) -> Self:
        return Self(id, OBSTACLE_CYLINDER, pos, Quat(), Vec3(radius, half_height, radius), radius, half_height, True)


@fieldwise_init
struct OffMeshLink(Copyable, ImplicitlyCopyable, Movable):
    var id: Int
    var start_pos: Vec3
    var end_pos: Vec3
    var radius: Float32
    var kind: UInt8
    var bidirectional: Bool
    var cost_scale: Float32
    var enabled: Bool
    var start_poly: Int
    var end_poly: Int

    def __init__(
        out self,
        id: Int,
        start_pos: Vec3,
        end_pos: Vec3,
        radius: Float32 = 0.5,
        kind: UInt8 = OFFMESH_JUMP,
        bidirectional: Bool = True,
        cost_scale: Float32 = 1.2,
    ):
        self.id = id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.radius = radius
        self.kind = kind
        self.bidirectional = bidirectional
        self.cost_scale = cost_scale
        self.enabled = True
        self.start_poly = -1
        self.end_poly = -1


struct NavPoly(Copyable, Movable):
    var verts: List[Vec3]
    var center: Vec3
    var plane_n: Vec3
    var plane_d: Float32
    var neighbors: List[Int]
    var portal_left: List[Vec3]
    var portal_right: List[Vec3]
    var area_cost: Float32
    var region_id: Int32

    def __init__(out self):
        self.verts = List[Vec3]()
        self.center = Vec3()
        self.plane_n = Vec3(0, 1, 0)
        self.plane_d = 0.0
        self.neighbors = List[Int]()
        self.portal_left = List[Vec3]()
        self.portal_right = List[Vec3]()
        self.area_cost = 1.0
        self.region_id = 0

    def sample_y(self, x: Float32, z: Float32) -> Float32:
        if abs(self.plane_n.y) > Float32(1.0e-4):
            return (self.plane_d - self.plane_n.x * x - self.plane_n.z * z) / self.plane_n.y
        return self.center.y

    def contains_xz(self, p: Vec3, eps: Float32 = 1.0e-3) -> Bool:
        """Checks if `(p.x, p.z)` lies inside this convex polygon."""
        var n = len(self.verts)
        if n < 3:
            return False
        var pos_cnt = 0
        var neg_cnt = 0
        for i in range(n):
            var a = self.verts[i]
            var b = self.verts[(i + 1) % n]
            var area = _triarea2_xz(a, b, p)
            if area > eps:
                pos_cnt += 1
            elif area < -eps:
                neg_cnt += 1
        return pos_cnt == 0 or neg_cnt == 0

    def closest_point_xz(self, p: Vec3) -> Vec3:
        if self.contains_xz(p, Float32(1.0e-4)):
            return Vec3(p.x, self.sample_y(p.x, p.z), p.z)
        var best_pt = self.center
        var best_d2 = Float32(1.0e30)
        var n = len(self.verts)
        for i in range(n):
            var a = self.verts[i]
            var b = self.verts[(i + 1) % n]
            var abx = b.x - a.x
            var abz = b.z - a.z
            var l2 = abx * abx + abz * abz
            var t = max(Float32(0.0), min(Float32(1.0), ((p.x - a.x) * abx + (p.z - a.z) * abz) / l2)) if l2 > Float32(1.0e-10) else Float32(0.0)
            var cand = a + (b - a) * t
            var dx = p.x - cand.x
            var dz = p.z - cand.z
            var d2 = dx * dx + dz * dz + (p.y - cand.y) * (p.y - cand.y) * Float32(0.25)
            if d2 < best_d2:
                best_d2 = d2
                best_pt = cand
        return best_pt


struct NavTile(Copyable, Movable):
    var tx: Int
    var tz: Int
    var x0: Int
    var x1: Int
    var z0: Int
    var z1: Int
    var span_indices: List[Int32]
    var poly_ids: List[Int]

    def __init__(out self, tx: Int = 0, tz: Int = 0, x0: Int = 0, x1: Int = 0, z0: Int = 0, z1: Int = 0):
        self.tx = tx
        self.tz = tz
        self.x0 = x0
        self.x1 = x1
        self.z0 = z0
        self.z1 = z1
        self.span_indices = List[Int32]()
        self.poly_ids = List[Int]()


struct NavPath(Copyable, Movable):
    var found: Bool
    var waypoints: List[Vec3]
    var poly_corridor: List[Int]
    var used_off_mesh_links: List[Int]
    var total_length: Float32

    def __init__(out self):
        self.found = False
        self.waypoints = List[Vec3]()
        self.poly_corridor = List[Int]()
        self.used_off_mesh_links = List[Int]()
        self.total_length = 0.0


def _append_box_tris(
    pos: Vec3, rot: Quat, half: Vec3,
    mut out_a: List[Vec3], mut out_b: List[Vec3], mut out_c: List[Vec3],
):
    var hx = half.x; var hy = half.y; var hz = half.z
    var lv = [
        Vec3(-hx, -hy, -hz), Vec3(hx, -hy, -hz), Vec3(hx, hy, -hz), Vec3(-hx, hy, -hz),
        Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz),
    ]
    var wv = List[Vec3](capacity=8)
    for i in range(8):
        wv.append(pos + rot.rotate(lv[i]))
    var quads = [
        (3, 7, 6, 2),  # +Y top
        (0, 1, 5, 4),  # -Y bottom
        (1, 2, 6, 5),  # +X
        (4, 7, 3, 0),  # -X
        (4, 5, 6, 7),  # +Z
        (1, 0, 3, 2),  # -Z
    ]
    for q in range(6):
        var i0 = quads[q][0]; var i1 = quads[q][1]; var i2 = quads[q][2]; var i3 = quads[q][3]
        out_a.append(wv[i0]); out_b.append(wv[i1]); out_c.append(wv[i2])
        out_a.append(wv[i0]); out_b.append(wv[i2]); out_c.append(wv[i3])


def _clip_poly_axis_xz(poly: List[Vec3], axis_is_z: Bool, bound: Float32, keep_ge: Bool) -> List[Vec3]:
    var out = List[Vec3](capacity=len(poly) + 2)
    var n = len(poly)
    if n == 0:
        return out^
    for i in range(n):
        var a = poly[i]
        var b = poly[(i + 1) % n]
        var ca = a.z if axis_is_z else a.x
        var cb = b.z if axis_is_z else b.x
        var da = (bound - ca) if keep_ge else (ca - bound)
        var db = (bound - cb) if keep_ge else (cb - bound)
        if da <= Float32(0.0):
            out.append(a)
            if db > Float32(0.0):
                var t = da / (da - db)
                out.append(a + (b - a) * t)
        elif db <= Float32(0.0):
            var t = da / (da - db)
            out.append(a + (b - a) * t)
    return out^


def _clip_poly_y_band(v0: Vec3, v1: Vec3, v2: Vec3, y_lo: Float32, y_hi: Float32) -> List[Vec3]:
    var cur = List[Vec3](capacity=5)
    cur.append(v0); cur.append(v1); cur.append(v2)

    var ge_out = List[Vec3](capacity=5)
    var n0 = len(cur)
    for i in range(n0):
        var a = cur[i]
        var b = cur[(i + 1) % n0]
        var da = y_lo - a.y
        var db = y_lo - b.y
        if da <= Float32(0.0):
            ge_out.append(a)
            if db > Float32(0.0):
                ge_out.append(a + (b - a) * (da / (da - db)))
        elif db <= Float32(0.0):
            ge_out.append(a + (b - a) * (da / (da - db)))

    var out = List[Vec3](capacity=6)
    var n1 = len(ge_out)
    if n1 == 0:
        return out^
    for i in range(n1):
        var a = ge_out[i]
        var b = ge_out[(i + 1) % n1]
        var da = a.y - y_hi
        var db = b.y - y_hi
        if da <= Float32(0.0):
            out.append(a)
            if db > Float32(0.0):
                out.append(a + (b - a) * (da / (da - db)))
        elif db <= Float32(0.0):
            out.append(a + (b - a) * (da / (da - db)))
    return out^


def _signed_poly_area2_xz(verts: List[Vec3]) -> Float32:
    var sum_a = Float32(0.0)
    var n = len(verts)
    for i in range(n):
        var a = verts[i]
        var b = verts[(i + 1) % n]
        sum_a += a.x * b.z - b.x * a.z
    return sum_a


@always_inline
def _portal_best_crossing_xz(p_from: Vec3, p_goal: Vec3, l_pt: Vec3, r_pt: Vec3) -> Vec3:
    """Finds where segment `p_from -> p_goal` crosses portal `[l_pt, r_pt]` (or closest clamped endpoint)."""
    var rx = p_goal.x - p_from.x
    var rz = p_goal.z - p_from.z
    var sx = r_pt.x - l_pt.x
    var sz = r_pt.z - l_pt.z
    var denom = rx * sz - rz * sx
    if abs(denom) > Float32(1.0e-8):
        var qpx = l_pt.x - p_from.x
        var qpz = l_pt.z - p_from.z
        var u = (qpx * rz - qpz * rx) / denom
        var u_clamped = max(Float32(0.02), min(Float32(0.98), u))
        return l_pt + (r_pt - l_pt) * u_clamped

    var l2 = sx * sx + sz * sz
    if l2 <= Float32(1.0e-10):
        return (l_pt + r_pt) * Float32(0.5)
    var mid_ref = (p_from + p_goal) * Float32(0.5)
    var u_proj = ((mid_ref.x - l_pt.x) * sx + (mid_ref.z - l_pt.z) * sz) / l2
    return l_pt + (r_pt - l_pt) * max(Float32(0.02), min(Float32(0.98), u_proj))


struct NavMesh(Copyable, Movable):
    var cfg: NavMeshConfig
    var chf: CompactHeightfield
    var tiles_x: Int
    var tiles_z: Int
    var tile_size: Int
    var tiles: List[NavTile]
    var polys: List[NavPoly]
    var poly_owner_tile: List[Int]
    var next_region_id: Int32
    var obstacles: List[NavObstacle]
    var off_mesh_links: List[OffMeshLink]

    def __init__(out self, cfg: NavMeshConfig = NavMeshConfig()):
        self.cfg = cfg
        self.chf = CompactHeightfield()
        self.tiles_x = 0
        self.tiles_z = 0
        self.tile_size = max(8, cfg.tile_size)
        self.tiles = List[NavTile]()
        self.polys = List[NavPoly]()
        self.poly_owner_tile = List[Int]()
        self.next_region_id = 1
        self.obstacles = List[NavObstacle]()
        self.off_mesh_links = List[OffMeshLink]()

    def poly_count(self) -> Int:
        return len(self.polys)

    @staticmethod
    def build_from_physics(
        world: PhysicsWorld,
        cfg: NavMeshConfig = NavMeshConfig(),
    ) -> NavMesh:
        """Extracts static/kinematic geometry from `PhysicsWorld` and bakes full tiled Recast `NavMesh`."""
        var tri_a = List[Vec3]()
        var tri_b = List[Vec3]()
        var tri_c = List[Vec3]()

        for i in range(len(world.bodies)):
            ref b = world.bodies[i]
            if not b.active or b.motion_mode == BODY_MOTION_DYNAMIC:
                continue
            if b.shape == BODY_SHAPE_BOX:
                _append_box_tris(b.pos, b.rot, b.half_extents, tri_a, tri_b, tri_c)
            elif b.shape == BODY_SHAPE_CONVEX_HULL and b.shape_index >= 0 and b.shape_index < len(world.hulls):
                ref h = world.hulls[b.shape_index]
                if len(h.indices) >= 12:
                    var nt = len(h.indices) // 3
                    for t in range(nt):
                        tri_a.append(b.pos + b.rot.rotate(h.vertices[Int(h.indices[t * 3 + 0])]))
                        tri_b.append(b.pos + b.rot.rotate(h.vertices[Int(h.indices[t * 3 + 1])]))
                        tri_c.append(b.pos + b.rot.rotate(h.vertices[Int(h.indices[t * 3 + 2])]))
                else:
                    _append_box_tris(b.pos, b.rot, b.half_extents, tri_a, tri_b, tri_c)
            elif b.shape == BODY_SHAPE_MESH_BVH8 and b.shape_index >= 0 and b.shape_index < len(world.meshes):
                ref mesh = world.meshes[b.shape_index]
                for t in range(len(mesh.triangles)):
                    ref mt = mesh.triangles[t]
                    tri_a.append(b.pos + b.rot.rotate(mt.v0))
                    tri_b.append(b.pos + b.rot.rotate(mt.v1))
                    tri_c.append(b.pos + b.rot.rotate(mt.v2))
            elif b.shape == BODY_SHAPE_HEIGHTFIELD and b.shape_index >= 0 and b.shape_index < len(world.heightfields):
                ref hf = world.heightfields[b.shape_index]
                for cz in range(hf.nz - 1):
                    for cx in range(hf.nx - 1):
                        var ct = hf.cell_triangles(cx, cz)
                        tri_a.append(b.pos + b.rot.rotate(ct[0].v0))
                        tri_b.append(b.pos + b.rot.rotate(ct[0].v1))
                        tri_c.append(b.pos + b.rot.rotate(ct[0].v2))
                        tri_a.append(b.pos + b.rot.rotate(ct[1].v0))
                        tri_b.append(b.pos + b.rot.rotate(ct[1].v1))
                        tri_c.append(b.pos + b.rot.rotate(ct[1].v2))
            elif b.shape == BODY_SHAPE_COMPOUND and b.shape_index >= 0 and b.shape_index < len(world.compounds):
                ref cmp = world.compounds[b.shape_index]
                for ci in range(len(cmp.children)):
                    ref ch = cmp.children[ci]
                    var wp_wr = ch.world_pose(b.pos, b.rot)
                    if ch.shape == COMPOUND_CHILD_BOX:
                        _append_box_tris(wp_wr[0], wp_wr[1], ch.half_extents, tri_a, tri_b, tri_c)
                    elif ch.shape == COMPOUND_CHILD_CONVEX and len(ch.hull.indices) >= 12:
                        var nt = len(ch.hull.indices) // 3
                        for t in range(nt):
                            tri_a.append(wp_wr[0] + wp_wr[1].rotate(ch.hull.vertices[Int(ch.hull.indices[t * 3 + 0])]))
                            tri_b.append(wp_wr[0] + wp_wr[1].rotate(ch.hull.vertices[Int(ch.hull.indices[t * 3 + 1])]))
                            tri_c.append(wp_wr[0] + wp_wr[1].rotate(ch.hull.vertices[Int(ch.hull.indices[t * 3 + 2])]))

        return NavMesh.build_from_triangle_lists(tri_a, tri_b, tri_c, cfg)

    @staticmethod
    def build_from_triangle_lists(
        tri_a: List[Vec3],
        tri_b: List[Vec3],
        tri_c: List[Vec3],
        cfg: NavMeshConfig = NavMeshConfig(),
    ) -> NavMesh:
        var nav = NavMesh(cfg)
        var n_tris = len(tri_a)
        if n_tris == 0:
            return nav^

        var raw_min = tri_a[0]
        var raw_max = tri_a[0]
        for i in range(n_tris):
            var tv = [tri_a[i], tri_b[i], tri_c[i]]
            for k in range(3):
                var v = tv[k]
                raw_min.x = min(raw_min.x, v.x); raw_min.y = min(raw_min.y, v.y); raw_min.z = min(raw_min.z, v.z)
                raw_max.x = max(raw_max.x, v.x); raw_max.y = max(raw_max.y, v.y); raw_max.z = max(raw_max.z, v.z)

        var cs = max(Float32(0.05), cfg.cell_size)
        var ch = max(Float32(0.02), cfg.cell_height)
        var span_x = (raw_max.x - raw_min.x) + cs * Float32(2.0)
        var span_z = (raw_max.z - raw_min.z) + cs * Float32(2.0)
        var w = max(2, min(1024, Int(floor(span_x / cs)) + 1))
        var d = max(2, min(1024, Int(floor(span_z / cs)) + 1))
        var mid_x = (raw_min.x + raw_max.x) * Float32(0.5)
        var mid_z = (raw_min.z + raw_max.z) * Float32(0.5)
        var bmin = Vec3(
            mid_x - Float32(w) * cs * Float32(0.5),
            raw_min.y - ch * Float32(2.0),
            mid_z - Float32(d) * cs * Float32(0.5),
        )

        var inv_cs = Float32(1.0) / cs
        var inv_ch = Float32(1.0) / ch
        var cos_walk = det_cos(cfg.walkable_slope_deg * NAV_DEG_TO_RAD)

        # 1. Solid Voxel Span Rasterization + collect non-walkable wall triangles for exact radius erosion
        var cols = List[List[_SolidSpan]](capacity=w * d)
        for _ in range(w * d):
            cols.append(List[_SolidSpan]())

        var wall_tris = List[Int]()
        var min_walk_cell_area2 = cs * cs * Float32(0.05)

        for t in range(n_tris):
            var v0 = tri_a[t]
            var v1 = tri_b[t]
            var v2 = tri_c[t]
            var raw_n = (v1 - v0).cross(v2 - v0)
            var nl = raw_n.length()
            if nl <= Float32(1.0e-8):
                continue
            var n = raw_n * (Float32(1.0) / nl)
            var walkable = n.y >= cos_walk
            if not walkable and n.y > Float32(-0.85):
                wall_tris.append(t)

            var tlo_x = min(v0.x, min(v1.x, v2.x))
            var thi_x = max(v0.x, max(v1.x, v2.x))
            var tlo_z = min(v0.z, min(v1.z, v2.z))
            var thi_z = max(v0.z, max(v1.z, v2.z))

            var iz0 = max(0, Int(floor((tlo_z - bmin.z) * inv_cs)))
            var iz1 = min(d - 1, Int(floor((thi_z - bmin.z) * inv_cs)))
            var ix0 = max(0, Int(floor((tlo_x - bmin.x) * inv_cs)))
            var ix1 = min(w - 1, Int(floor((thi_x - bmin.x) * inv_cs)))

            var base_poly = List[Vec3](capacity=3)
            base_poly.append(v0); base_poly.append(v1); base_poly.append(v2)

            for iz in range(iz0, iz1 + 1):
                var z_lo = bmin.z + Float32(iz) * cs
                var z_hi = z_lo + cs
                var z_poly = _clip_poly_axis_xz(base_poly, True, z_lo, True)
                if len(z_poly) < 2:
                    continue
                z_poly = _clip_poly_axis_xz(z_poly, True, z_hi, False)
                if len(z_poly) < 2:
                    continue

                for ix in range(ix0, ix1 + 1):
                    var x_lo = bmin.x + Float32(ix) * cs
                    var x_hi = x_lo + cs
                    var cell_poly = _clip_poly_axis_xz(z_poly, False, x_lo, True)
                    if len(cell_poly) < 2:
                        continue
                    cell_poly = _clip_poly_axis_xz(cell_poly, False, x_hi, False)
                    if walkable:
                        if len(cell_poly) < 3 or abs(_signed_poly_area2_xz(cell_poly)) < min_walk_cell_area2:
                            continue
                    elif len(cell_poly) < 2:
                        continue

                    var cy_min = cell_poly[0].y
                    var cy_max = cell_poly[0].y
                    for cp_i in range(1, len(cell_poly)):
                        cy_min = min(cy_min, cell_poly[cp_i].y)
                        cy_max = max(cy_max, cell_poly[cp_i].y)

                    var smin = UInt16(max(0, min(65534, Int(floor((cy_min - bmin.y) * inv_ch)))))
                    var smax = UInt16(max(Int(smin) + 1, min(65535, Int(floor((cy_max - bmin.y) * inv_ch)) + 1)))

                    ref col = cols[iz * w + ix]
                    var inserted = False
                    var k = 0
                    var cur_min = smin
                    var cur_max = smax
                    var cur_walk = walkable
                    while k < len(col):
                        if col[k].smax < cur_min:
                            k += 1
                        elif col[k].smin > cur_max:
                            col.insert(k, _SolidSpan(cur_min, cur_max, cur_walk))
                            inserted = True
                            break
                        else:
                            if abs(Int(cur_max) - Int(col[k].smax)) <= 1:
                                cur_walk = cur_walk or col[k].walkable
                            elif col[k].smax > cur_max:
                                cur_walk = col[k].walkable
                            cur_min = min(cur_min, col[k].smin)
                            cur_max = max(cur_max, col[k].smax)
                            var rem_col = List[_SolidSpan](capacity=len(col))
                            for rk in range(len(col)):
                                if rk != k:
                                    rem_col.append(col[rk])
                            col = rem_col^
                    if not inserted:
                        col.append(_SolidSpan(cur_min, cur_max, cur_walk))

        # 2. Build CompactHeightfield + Standing Headroom & Ledge Neighbor Links
        var chf = CompactHeightfield()
        chf.width = w
        chf.depth = d
        chf.origin = bmin
        chf.cell_size = cs
        chf.cell_height = ch
        chf.col_first = List[Int32](length=w * d, fill=0)
        chf.col_count = List[UInt16](length=w * d, fill=0)

        var walk_h_vox = max(1, Int(floor(cfg.walkable_height * inv_ch + Float32(0.999))))
        var climb_vox = max(1, Int(floor(cfg.walkable_climb * inv_ch)))

        for iz in range(d):
            for ix in range(w):
                var c_idx = iz * w + ix
                ref col = cols[c_idx]
                chf.col_first[c_idx] = Int32(len(chf.spans))
                var added = 0
                for si in range(len(col)):
                    if not col[si].walkable:
                        continue
                    var open_y = col[si].smax
                    var ceil_y = col[si + 1].smin if (si + 1 < len(col)) else UInt16(65535)
                    if Int(ceil_y) - Int(open_y) >= walk_h_vox:
                        chf.spans.append(CompactSpan(ix, iz, open_y, ceil_y, True))
                        added += 1
                chf.col_count[c_idx] = UInt16(added)

        var dx_arr = [-1, 0, 1, 0]
        var dz_arr = [0, 1, 0, -1]
        for s_idx in range(len(chf.spans)):
            var sx = Int(chf.spans[s_idx].x)
            var sz = Int(chf.spans[s_idx].z)
            var sy = Int(chf.spans[s_idx].y)
            var sceil = Int(chf.spans[s_idx].ceiling)
            for dir in range(4):
                var nx = sx + dx_arr[dir]
                var nz = sz + dz_arr[dir]
                if nx < 0 or nx >= w or nz < 0 or nz >= d:
                    continue
                var nc = nz * w + nx
                var nf = Int(chf.col_first[nc])
                var ncnt = Int(chf.col_count[nc])
                var best_nb = -1
                var best_dy = climb_vox + 1
                for ni in range(nf, nf + ncnt):
                    var ny = Int(chf.spans[ni].y)
                    var nceil = Int(chf.spans[ni].ceiling)
                    var dy = abs(ny - sy)
                    if dy <= climb_vox and (min(sceil, nceil) - max(sy, ny)) >= walk_h_vox:
                        if dy < best_dy:
                            best_dy = dy
                            best_nb = ni
                if best_nb >= 0:
                    chf.spans[s_idx].set_neighbor(dir, best_nb)

        # 3a. Open-ledge 8-neighbor Chamfer distance erosion (cliffs & floor drop-offs)
        nav.chf = chf^
        var rad = max(Float32(0.0), cfg.walkable_radius)
        if rad > Float32(1.0e-4):
            Self._compute_open_ledge_chamfer(nav.chf)
            var ledge_thresh = UInt16(max(2, Int(floor((rad * inv_cs) * Float32(2.0) + Float32(0.5)))))
            for i in range(len(nav.chf.spans)):
                if nav.chf.spans[i].dist < ledge_thresh:
                    nav.chf.spans[i].base_area = 0
                    nav.chf.spans[i].area = 0

            # 3b. Exact standing-wall cell-AABB clearance erosion (`dist(cell_aabb, wall_xz) < walkable_radius`)
            var rad_eff = max(Float32(0.0), rad - Float32(1.0e-4))
            var rad2 = rad_eff * rad_eff
            for wt in range(len(wall_tris)):
                var tidx = wall_tris[wt]
                var v0 = tri_a[tidx]
                var v1 = tri_b[tidx]
                var v2 = tri_c[tidx]
                var ty_min = min(v0.y, min(v1.y, v2.y))
                var ty_max = max(v0.y, max(v1.y, v2.y))
                var tlo_x = min(v0.x, min(v1.x, v2.x)) - rad
                var thi_x = max(v0.x, max(v1.x, v2.x)) + rad
                var tlo_z = min(v0.z, min(v1.z, v2.z)) - rad
                var thi_z = max(v0.z, max(v1.z, v2.z)) + rad

                var ix0 = max(0, Int(floor((tlo_x - bmin.x) * inv_cs)))
                var ix1 = min(w - 1, Int(floor((thi_x - bmin.x) * inv_cs)))
                var iz0 = max(0, Int(floor((tlo_z - bmin.z) * inv_cs)))
                var iz1 = min(d - 1, Int(floor((thi_z - bmin.z) * inv_cs)))

                for iz in range(iz0, iz1 + 1):
                    var z0 = bmin.z + Float32(iz) * cs
                    var z1 = z0 + cs
                    for ix in range(ix0, ix1 + 1):
                        var c = iz * w + ix
                        var f = Int(nav.chf.col_first[c])
                        var cnt = Int(nav.chf.col_count[c])
                        if cnt == 0:
                            continue
                        var x0 = bmin.x + Float32(ix) * cs
                        var x1 = x0 + cs
                        for s in range(f, f + cnt):
                            if nav.chf.spans[s].base_area == 0:
                                continue
                            var surf_y = bmin.y + Float32(max(0, Int(nav.chf.spans[s].y) - 1)) * ch
                            var band_lo = surf_y + cfg.walkable_climb + ch * Float32(0.25)
                            var band_hi = surf_y + cfg.walkable_height
                            if ty_max <= band_lo or ty_min >= band_hi:
                                continue
                            var clipped = _clip_poly_y_band(v0, v1, v2, band_lo, band_hi)
                            if len(clipped) >= 2 and _aabb_poly_dist2_xz(x0, z0, x1, z1, clipped) < rad2:
                                nav.chf.spans[s].base_area = 0
                                nav.chf.spans[s].area = 0

        # 4. Initialize spatial `(tx, tz)` tiles and bake initial tiled polygon mesh
        nav._init_tiles_and_build_all()
        return nav^

    @staticmethod
    def _compute_open_ledge_chamfer(mut chf: CompactHeightfield):
        """2-pass 8-neighbor Rosenfeld-Pfaltz Chamfer distance from open drop-off ledges (`nb < 0`)."""
        var n_spans = len(chf.spans)
        for i in range(n_spans):
            if chf.spans[i].base_area == 0:
                chf.spans[i].dist = 0
                continue
            var n0 = chf.spans[i].n0
            var n1 = chf.spans[i].n1
            var n2 = chf.spans[i].n2
            var n3 = chf.spans[i].n3
            if n0 < 0 or n1 < 0 or n2 < 0 or n3 < 0:
                chf.spans[i].dist = 0
                continue
            # Also check 4 diagonal neighbors so ledge corners erode cleanly
            var diag_open = (
                chf.spans[Int(n0)].n3 < 0
                or chf.spans[Int(n0)].n1 < 0
                or chf.spans[Int(n2)].n3 < 0
                or chf.spans[Int(n2)].n1 < 0
            )
            chf.spans[i].dist = UInt16(1) if diag_open else UInt16(65530)

        for iz in range(chf.depth):
            for ix in range(chf.width):
                var c = iz * chf.width + ix
                var f = Int(chf.col_first[c])
                var cnt = Int(chf.col_count[c])
                for s in range(f, f + cnt):
                    var d_cur = chf.spans[s].dist
                    if d_cur == 0:
                        continue
                    var n0 = chf.spans[s].n0
                    if n0 >= 0:
                        d_cur = min(d_cur, chf.spans[Int(n0)].dist + UInt16(2))
                        var n03 = chf.spans[Int(n0)].n3
                        if n03 >= 0:
                            d_cur = min(d_cur, chf.spans[Int(n03)].dist + UInt16(3))
                    var n3 = chf.spans[s].n3
                    if n3 >= 0:
                        d_cur = min(d_cur, chf.spans[Int(n3)].dist + UInt16(2))
                        var n32 = chf.spans[Int(n3)].n2
                        if n32 >= 0:
                            d_cur = min(d_cur, chf.spans[Int(n32)].dist + UInt16(3))
                    chf.spans[s].dist = d_cur

        var iz = chf.depth - 1
        while iz >= 0:
            var ix = chf.width - 1
            while ix >= 0:
                var c = iz * chf.width + ix
                var f = Int(chf.col_first[c])
                var cnt = Int(chf.col_count[c])
                for s in range(f, f + cnt):
                    var d_cur = chf.spans[s].dist
                    if d_cur == 0:
                        continue
                    var n2 = chf.spans[s].n2
                    if n2 >= 0:
                        d_cur = min(d_cur, chf.spans[Int(n2)].dist + UInt16(2))
                        var n21 = chf.spans[Int(n2)].n1
                        if n21 >= 0:
                            d_cur = min(d_cur, chf.spans[Int(n21)].dist + UInt16(3))
                    var n1 = chf.spans[s].n1
                    if n1 >= 0:
                        d_cur = min(d_cur, chf.spans[Int(n1)].dist + UInt16(2))
                        var n10 = chf.spans[Int(n1)].n0
                        if n10 >= 0:
                            d_cur = min(d_cur, chf.spans[Int(n10)].dist + UInt16(3))
                    chf.spans[s].dist = d_cur
                ix -= 1
            iz -= 1

    # =========================================================================
    # Tiled Mesh Initialization & Sub-0.25ms Incremental Dynamic Obstacle Carve
    # =========================================================================

    def _init_tiles_and_build_all(mut self):
        self.tile_size = max(8, self.cfg.tile_size)
        self.tiles_x = (self.chf.width + self.tile_size - 1) // self.tile_size
        self.tiles_z = (self.chf.depth + self.tile_size - 1) // self.tile_size
        self.tiles.clear()
        self.polys.clear()
        self.poly_owner_tile.clear()
        self.next_region_id = 1

        var total_tiles = self.tiles_x * self.tiles_z
        self.tiles.reserve(total_tiles)
        for tz in range(self.tiles_z):
            var z0 = tz * self.tile_size
            var z1 = min(self.chf.depth, z0 + self.tile_size)
            for tx in range(self.tiles_x):
                var x0 = tx * self.tile_size
                var x1 = min(self.chf.width, x0 + self.tile_size)
                var tile = NavTile(tx, tz, x0, x1, z0, z1)
                for iz in range(z0, z1):
                    for ix in range(x0, x1):
                        var c = iz * self.chf.width + ix
                        var f = Int(self.chf.col_first[c])
                        var cnt = Int(self.chf.col_count[c])
                        for s in range(f, f + cnt):
                            tile.span_indices.append(Int32(s))
                self.tiles.append(tile^)

        for tid in range(total_tiles):
            var local_polys = self._mesh_tile_polys(tid)
            for pi in range(len(local_polys)):
                var pid = len(self.polys)
                self.polys.append(local_polys[pi].copy())
                self.poly_owner_tile.append(tid)
                self.tiles[tid].poly_ids.append(pid)

        # Link portals within every tile and across +X / +Z tile borders in O(tiles)
        for tz in range(self.tiles_z):
            for tx in range(self.tiles_x):
                var tid = tz * self.tiles_x + tx
                self._link_within_tile(tid)
                if tx + 1 < self.tiles_x:
                    self._link_between_tiles(tid, tid + 1)
                if tz + 1 < self.tiles_z:
                    self._link_between_tiles(tid, tid + self.tiles_x)

    def _obstacle_tile_bounds(self, obs: NavObstacle) -> Tuple[Int, Int, Int, Int]:
        """Returns `(tx0, tx1, tz0, tz1)` inclusive tile indices touched by `obs` inflated by `walkable_radius`."""
        if self.tiles_x == 0 or self.tiles_z == 0:
            return (0, -1, 0, -1)
        var cs = self.chf.cell_size
        var inv_cs = Float32(1.0) / max(Float32(1.0e-5), cs)
        var orig = self.chf.origin
        var pad_r = self.cfg.walkable_radius + cs

        var wx_e: Float32
        var wz_e: Float32
        if obs.shape == OBSTACLE_CYLINDER:
            wx_e = obs.radius + pad_r
            wz_e = obs.radius + pad_r
        else:
            var he = obs.half_extents + Vec3(pad_r, 0.0, pad_r)
            var axes = get_box_axes(obs.rot)
            wx_e = abs(axes[0].x) * he.x + abs(axes[1].x) * he.y + abs(axes[2].x) * he.z
            wz_e = abs(axes[0].z) * he.x + abs(axes[1].z) * he.y + abs(axes[2].z) * he.z

        var ix0 = max(0, Int(floor((obs.pos.x - wx_e - orig.x) * inv_cs)))
        var ix1 = min(self.chf.width - 1, Int(floor((obs.pos.x + wx_e - orig.x) * inv_cs)))
        var iz0 = max(0, Int(floor((obs.pos.z - wz_e - orig.z) * inv_cs)))
        var iz1 = min(self.chf.depth - 1, Int(floor((obs.pos.z + wz_e - orig.z) * inv_cs)))
        if ix0 > ix1 or iz0 > iz1:
            return (0, -1, 0, -1)
        return (
            ix0 // self.tile_size,
            min(self.tiles_x - 1, ix1 // self.tile_size),
            iz0 // self.tile_size,
            min(self.tiles_z - 1, iz1 // self.tile_size),
        )

    def _carve_tile_spans(mut self, tid: Int):
        """Restores `area = base_area` ONLY in `tiles[tid]` and carves overlapping active `NavObstacle`s."""
        ref tile = self.tiles[tid]
        for k in range(len(tile.span_indices)):
            var s = Int(tile.span_indices[k])
            self.chf.spans[s].area = self.chf.spans[s].base_area

        var cs = self.chf.cell_size
        var ch = self.chf.cell_height
        var inv_cs = Float32(1.0) / max(Float32(1.0e-5), cs)
        var orig = self.chf.origin
        var pad_r = max(Float32(0.0), self.cfg.walkable_radius - Float32(1.0e-4))
        var half_cs = cs * Float32(0.5)

        for oi in range(len(self.obstacles)):
            ref obs = self.obstacles[oi]
            if not obs.active:
                continue
            var y_lo = obs.pos.y - obs.half_height - ch * Float32(2.0)
            var y_hi = obs.pos.y + obs.half_height + self.cfg.walkable_height

            if obs.shape == OBSTACLE_CYLINDER:
                var r_eff = obs.radius + pad_r
                var r2 = r_eff * r_eff
                var ix0 = max(tile.x0, Int(floor((obs.pos.x - r_eff - orig.x) * inv_cs)))
                var ix1 = min(tile.x1 - 1, Int(floor((obs.pos.x + r_eff - orig.x) * inv_cs)))
                var iz0 = max(tile.z0, Int(floor((obs.pos.z - r_eff - orig.z) * inv_cs)))
                var iz1 = min(tile.z1 - 1, Int(floor((obs.pos.z + r_eff - orig.z) * inv_cs)))
                for iz in range(iz0, iz1 + 1):
                    var z0 = orig.z + Float32(iz) * cs
                    var z1 = z0 + cs
                    for ix in range(ix0, ix1 + 1):
                        var x0 = orig.x + Float32(ix) * cs
                        var x1 = x0 + cs
                        if _pt_aabb_dist2_xz(obs.pos.x, obs.pos.z, x0, z0, x1, z1) < r2:
                            var c = iz * self.chf.width + ix
                            var f = Int(self.chf.col_first[c])
                            var cnt = Int(self.chf.col_count[c])
                            for s in range(f, f + cnt):
                                var sy = orig.y + Float32(max(0, Int(self.chf.spans[s].y) - 1)) * ch
                                if sy >= y_lo and sy <= y_hi:
                                    self.chf.spans[s].area = 0
            else:
                var he_cell = obs.half_extents + Vec3(pad_r + half_cs, 0.0, pad_r + half_cs)
                var axes = get_box_axes(obs.rot)
                var wx_e = abs(axes[0].x) * he_cell.x + abs(axes[1].x) * he_cell.y + abs(axes[2].x) * he_cell.z
                var wz_e = abs(axes[0].z) * he_cell.x + abs(axes[1].z) * he_cell.y + abs(axes[2].z) * he_cell.z
                var ix0 = max(tile.x0, Int(floor((obs.pos.x - wx_e - orig.x) * inv_cs)))
                var ix1 = min(tile.x1 - 1, Int(floor((obs.pos.x + wx_e - orig.x) * inv_cs)))
                var iz0 = max(tile.z0, Int(floor((obs.pos.z - wz_e - orig.z) * inv_cs)))
                var iz1 = min(tile.z1 - 1, Int(floor((obs.pos.z + wz_e - orig.z) * inv_cs)))
                if ix0 > ix1 or iz0 > iz1:
                    continue
                var inv_q = obs.rot.conjugate()
                for iz in range(iz0, iz1 + 1):
                    var wz = orig.z + (Float32(iz) + Float32(0.5)) * cs
                    for ix in range(ix0, ix1 + 1):
                        var c = iz * self.chf.width + ix
                        var f = Int(self.chf.col_first[c])
                        var cnt = Int(self.chf.col_count[c])
                        if cnt == 0:
                            continue
                        var wx = orig.x + (Float32(ix) + Float32(0.5)) * cs
                        var lp = inv_q.rotate(Vec3(wx, obs.pos.y, wz) - obs.pos)
                        if abs(lp.x) < he_cell.x and abs(lp.z) < he_cell.z:
                            for s in range(f, f + cnt):
                                var sy = orig.y + Float32(max(0, Int(self.chf.spans[s].y) - 1)) * ch
                                if sy >= y_lo and sy <= y_hi:
                                    self.chf.spans[s].area = 0

    def _mesh_tile_polys(mut self, tid: Int) -> List[NavPoly]:
        """Fast O(tile_cells) greedy maximal coplanar convex rectangle decomposition with unlimited `Int32` regions."""
        var out = List[NavPoly]()
        var x0 = self.tiles[tid].x0
        var x1 = self.tiles[tid].x1
        var z0 = self.tiles[tid].z0
        var z1 = self.tiles[tid].z1
        var n_ts = len(self.tiles[tid].span_indices)
        if n_ts == 0:
            return out^

        for k in range(n_ts):
            var s = Int(self.tiles[tid].span_indices[k])
            self.chf.spans[s].reg = Int32(-1) if self.chf.spans[s].area > 0 else Int32(0)

        # Prune tiny completely-isolated components (< min_region_cells with zero cross-tile links)
        if self.cfg.min_region_cells > 1:
            var comp_q = List[Int]()
            for k in range(n_ts):
                var seed = Int(self.tiles[tid].span_indices[k])
                if self.chf.spans[seed].reg != Int32(-1):
                    continue
                comp_q.clear()
                comp_q.append(seed)
                self.chf.spans[seed].reg = Int32(-2)
                var touches_outside = False
                var qh = 0
                while qh < len(comp_q):
                    var cur = comp_q[qh]
                    qh += 1
                    for dir in range(4):
                        var nb = self.chf.spans[cur].get_neighbor(dir)
                        if nb >= 0 and self.chf.spans[nb].area > 0:
                            var nx = Int(self.chf.spans[nb].x)
                            var nz = Int(self.chf.spans[nb].z)
                            if nx < x0 or nx >= x1 or nz < z0 or nz >= z1:
                                touches_outside = True
                            elif self.chf.spans[nb].reg == Int32(-1):
                                self.chf.spans[nb].reg = Int32(-2)
                                comp_q.append(nb)
                if not touches_outside and len(comp_q) < self.cfg.min_region_cells:
                    for ci in range(len(comp_q)):
                        self.chf.spans[comp_q[ci]].reg = Int32(0)

            for k in range(n_ts):
                var s = Int(self.tiles[tid].span_indices[k])
                if self.chf.spans[s].reg == Int32(-2):
                    self.chf.spans[s].reg = Int32(-1)

        var cs = self.chf.cell_size
        var ch = self.chf.cell_height
        var orig = self.chf.origin
        var cur_row = List[Int](capacity=self.tile_size)
        var next_row = List[Int](capacity=self.tile_size)

        for k in range(n_ts):
            var s00 = Int(self.tiles[tid].span_indices[k])
            if self.chf.spans[s00].reg != Int32(-1):
                continue

            var reg_id = self.next_region_id
            self.next_region_id += 1

            var sx0 = Int(self.chf.spans[s00].x)
            var sz0 = Int(self.chf.spans[s00].z)
            var sy00 = Int(self.chf.spans[s00].y)

            cur_row.clear()
            cur_row.append(s00)
            while True:
                var last_s = cur_row[len(cur_row) - 1]
                var nb_x = self.chf.spans[last_s].get_neighbor(2)
                if nb_x < 0 or Int(self.chf.spans[nb_x].x) >= x1 or self.chf.spans[nb_x].reg != Int32(-1):
                    break
                if self.chf.spans[nb_x].get_neighbor(0) != last_s:
                    break
                var rc = len(cur_row)
                if rc >= 2:
                    var pred_y = sy00 + ((Int(self.chf.spans[last_s].y) - sy00) * rc) // (rc - 1)
                    if abs(Int(self.chf.spans[nb_x].y) - pred_y) > 2:
                        break
                cur_row.append(nb_x)

            var rw = len(cur_row)
            for ri in range(rw):
                self.chf.spans[cur_row[ri]].reg = reg_id

            var s_xhi_z0 = cur_row[rw - 1]
            var s_x0_zhi = s00
            var s_xhi_zhi = s_xhi_z0
            var rd = 1

            while True:
                next_row.clear()
                var row_ok = True
                for ri in range(rw):
                    var cur_s = cur_row[ri]
                    var nb_z = self.chf.spans[cur_s].get_neighbor(1)
                    if nb_z < 0 or Int(self.chf.spans[nb_z].z) >= z1 or self.chf.spans[nb_z].reg != Int32(-1):
                        row_ok = False
                        break
                    if self.chf.spans[nb_z].get_neighbor(3) != cur_s:
                        row_ok = False
                        break
                    if ri > 0:
                        var prev_nz = next_row[ri - 1]
                        if self.chf.spans[nb_z].get_neighbor(0) != prev_nz or self.chf.spans[prev_nz].get_neighbor(2) != nb_z:
                            row_ok = False
                            break
                    if rd >= 2 and ri == 0:
                        var pred_zy = sy00 + ((Int(self.chf.spans[s_x0_zhi].y) - sy00) * rd) // (rd - 1)
                        if abs(Int(self.chf.spans[nb_z].y) - pred_zy) > 2:
                            row_ok = False
                            break
                    if ri > 0:
                        var exp_y = Int(self.chf.spans[next_row[0]].y) + (
                            Int(self.chf.spans[cur_s].y) - Int(self.chf.spans[cur_row[0]].y)
                        )
                        if abs(Int(self.chf.spans[nb_z].y) - exp_y) > 2:
                            row_ok = False
                            break
                    next_row.append(nb_z)

                if not row_ok:
                    break
                for ri in range(rw):
                    self.chf.spans[next_row[ri]].reg = reg_id
                    cur_row[ri] = next_row[ri]
                s_x0_zhi = cur_row[0]
                s_xhi_zhi = cur_row[rw - 1]
                rd += 1

            var wx0 = orig.x + Float32(sx0) * cs
            var wx1 = orig.x + Float32(sx0 + rw) * cs
            var wz0 = orig.z + Float32(sz0) * cs
            var wz1 = orig.z + Float32(sz0 + rd) * cs

            var y00 = orig.y + Float32(max(0, sy00 - 1)) * ch
            var y10 = orig.y + Float32(max(0, Int(self.chf.spans[s_xhi_z0].y) - 1)) * ch
            var y11 = orig.y + Float32(max(0, Int(self.chf.spans[s_xhi_zhi].y) - 1)) * ch
            var y01 = orig.y + Float32(max(0, Int(self.chf.spans[s_x0_zhi].y) - 1)) * ch

            var np = NavPoly()
            np.region_id = reg_id
            np.verts.append(Vec3(wx0, y00, wz0))
            np.verts.append(Vec3(wx1, y10, wz0))
            np.verts.append(Vec3(wx1, y11, wz1))
            np.verts.append(Vec3(wx0, y01, wz1))
            np.center = (np.verts[0] + np.verts[1] + np.verts[2] + np.verts[3]) * Float32(0.25)

            var pn = (np.verts[1] - np.verts[0]).cross(np.verts[3] - np.verts[0])
            if pn.y < Float32(0.0):
                pn = -pn
            var pnl = pn.length()
            np.plane_n = pn * (Float32(1.0) / pnl) if pnl > Float32(1.0e-7) else Vec3(0, 1, 0)
            np.plane_d = np.plane_n.dot(np.center)
            out.append(np^)

        return out^

    def _remove_neighbor_ref(mut self, pid: Int, target_nb: Int):
        ref p = self.polys[pid]
        var k = 0
        while k < len(p.neighbors):
            if p.neighbors[k] == target_nb:
                var last = len(p.neighbors) - 1
                p.neighbors[k] = p.neighbors[last]
                p.portal_left[k] = p.portal_left[last]
                p.portal_right[k] = p.portal_right[last]
                _ = p.neighbors.pop()
                _ = p.portal_left.pop()
                _ = p.portal_right.pop()
            else:
                k += 1

    def _remap_neighbor_ref(mut self, pid: Int, old_nb: Int, new_nb: Int):
        ref p = self.polys[pid]
        for k in range(len(p.neighbors)):
            if p.neighbors[k] == old_nb:
                p.neighbors[k] = new_nb

    def _try_link_poly_pair(mut self, pa: Int, pb: Int, cs: Float32, max_climb: Float32, eps_xz: Float32):
        var na = len(self.polys[pa].verts)
        var nb = len(self.polys[pb].verts)
        for ea in range(na):
            var a0 = self.polys[pa].verts[ea]
            var a1 = self.polys[pa].verts[(ea + 1) % na]
            var dax = a1.x - a0.x
            var daz = a1.z - a0.z
            var la2 = dax * dax + daz * daz
            if la2 <= Float32(1.0e-6):
                continue
            var la = sqrt(la2)
            var uax = dax / la
            var uaz = daz / la

            for eb in range(nb):
                var b0 = self.polys[pb].verts[eb]
                var b1 = self.polys[pb].verts[(eb + 1) % nb]
                var dbx = b1.x - b0.x
                var dbz = b1.z - b0.z
                var lb = sqrt(dbx * dbx + dbz * dbz)
                if lb <= Float32(1.0e-3):
                    continue
                var dot_u = uax * (dbx / lb) + uaz * (dbz / lb)
                if dot_u > Float32(-0.998):
                    continue
                var perp0 = abs((b0.x - a0.x) * (-uaz) + (b0.z - a0.z) * uax)
                if perp0 > eps_xz:
                    continue

                var s0 = (b0.x - a0.x) * uax + (b0.z - a0.z) * uaz
                var s1 = (b1.x - a0.x) * uax + (b1.z - a0.z) * uaz
                var s_lo = max(Float32(0.0), min(s0, s1))
                var s_hi = min(la, max(s0, s1))
                if s_hi - s_lo <= cs * Float32(0.25):
                    continue

                var p_lo = a0 + (a1 - a0) * (s_lo / la)
                var p_hi = a0 + (a1 - a0) * (s_hi / la)
                var mid_x = (p_lo.x + p_hi.x) * Float32(0.5)
                var mid_z = (p_lo.z + p_hi.z) * Float32(0.5)
                var ya = self.polys[pa].sample_y(mid_x, mid_z)
                var yb = self.polys[pb].sample_y(mid_x, mid_z)
                if abs(ya - yb) > max_climb:
                    continue

                p_lo.y = max(self.polys[pa].sample_y(p_lo.x, p_lo.z), self.polys[pb].sample_y(p_lo.x, p_lo.z))
                p_hi.y = max(self.polys[pa].sample_y(p_hi.x, p_hi.z), self.polys[pb].sample_y(p_hi.x, p_hi.z))

                var l_ab = p_lo
                var r_ab = p_hi
                if _triarea2_xz(self.polys[pa].center, l_ab, r_ab) < Float32(0.0):
                    l_ab = p_hi
                    r_ab = p_lo
                self.polys[pa].neighbors.append(pb)
                self.polys[pa].portal_left.append(l_ab)
                self.polys[pa].portal_right.append(r_ab)

                var l_ba = r_ab
                var r_ba = l_ab
                if _triarea2_xz(self.polys[pb].center, l_ba, r_ba) < Float32(0.0):
                    var tmp = l_ba; l_ba = r_ba; r_ba = tmp
                self.polys[pb].neighbors.append(pa)
                self.polys[pb].portal_left.append(l_ba)
                self.polys[pb].portal_right.append(r_ba)
                return

    def _link_within_tile(mut self, tid: Int):
        var n_p = len(self.tiles[tid].poly_ids)
        if n_p < 2:
            return
        var cs = self.chf.cell_size
        var max_climb = self.cfg.walkable_climb + self.chf.cell_height * Float32(2.5)
        var eps_xz = cs * Float32(0.25)
        for i in range(n_p):
            var pa = self.tiles[tid].poly_ids[i]
            for j in range(i + 1, n_p):
                var pb = self.tiles[tid].poly_ids[j]
                self._try_link_poly_pair(pa, pb, cs, max_climb, eps_xz)

    def _link_between_tiles(mut self, tid_a: Int, tid_b: Int):
        var na = len(self.tiles[tid_a].poly_ids)
        var nb = len(self.tiles[tid_b].poly_ids)
        if na == 0 or nb == 0:
            return
        var cs = self.chf.cell_size
        var max_climb = self.cfg.walkable_climb + self.chf.cell_height * Float32(2.5)
        var eps_xz = cs * Float32(0.25)
        for i in range(na):
            var pa = self.tiles[tid_a].poly_ids[i]
            for j in range(nb):
                var pb = self.tiles[tid_b].poly_ids[j]
                self._try_link_poly_pair(pa, pb, cs, max_climb, eps_xz)

    def _rebuild_dirty_tiles(mut self, dirty_mask: List[Bool]):
        """Incrementally re-carves, re-meshes, and re-stitches ONLY dirty tiles (`< 0.25 ms` on `60m x 60m`)."""
        var dirty_tids = List[Int]()
        for tid in range(len(self.tiles)):
            if dirty_mask[tid]:
                dirty_tids.append(tid)
        if len(dirty_tids) == 0:
            return

        # 1. Sever all outgoing/incoming neighbor portal links on dirty tiles' existing polygons
        for di in range(len(dirty_tids)):
            var tid = dirty_tids[di]
            for k in range(len(self.tiles[tid].poly_ids)):
                var pid = self.tiles[tid].poly_ids[k]
                for ni in range(len(self.polys[pid].neighbors)):
                    var nb = self.polys[pid].neighbors[ni]
                    self._remove_neighbor_ref(nb, pid)
                self.polys[pid].neighbors.clear()
                self.polys[pid].portal_left.clear()
                self.polys[pid].portal_right.clear()

        # 2. Re-carve spans & re-mesh polygons per dirty tile, splicing slots in O(delta)
        var excess_slots = List[Int]()
        for di in range(len(dirty_tids)):
            var tid = dirty_tids[di]
            self._carve_tile_spans(tid)
            var fresh_polys = self._mesh_tile_polys(tid)
            var old_cnt = len(self.tiles[tid].poly_ids)
            var new_cnt = len(fresh_polys)
            var reuse_cnt = min(old_cnt, new_cnt)

            for k in range(reuse_cnt):
                var slot = self.tiles[tid].poly_ids[k]
                self.polys[slot] = fresh_polys[k].copy()
                self.poly_owner_tile[slot] = tid

            if new_cnt > old_cnt:
                for k in range(old_cnt, new_cnt):
                    var slot = len(self.polys)
                    self.polys.append(fresh_polys[k].copy())
                    self.poly_owner_tile.append(tid)
                    self.tiles[tid].poly_ids.append(slot)
            elif new_cnt < old_cnt:
                for k in range(new_cnt, old_cnt):
                    excess_slots.append(self.tiles[tid].poly_ids[k])
                self.tiles[tid].poly_ids.resize(new_cnt, 0)

        # Sort excess slots descending so swap-pop from the tail never invalidates another excess slot
        if len(excess_slots) > 0:
            for i in range(1, len(excess_slots)):
                var key = excess_slots[i]
                var j = i - 1
                while j >= 0 and excess_slots[j] < key:
                    excess_slots[j + 1] = excess_slots[j]
                    j -= 1
                excess_slots[j + 1] = key

            for ei in range(len(excess_slots)):
                var hole = excess_slots[ei]
                var last = len(self.polys) - 1
                if hole == last:
                    _ = self.polys.pop()
                    _ = self.poly_owner_tile.pop()
                else:
                    var moved_poly = self.polys.pop()
                    var moved_owner = self.poly_owner_tile.pop()
                    for ni in range(len(moved_poly.neighbors)):
                        var nb = moved_poly.neighbors[ni]
                        self._remap_neighbor_ref(nb, last, hole)
                    for pk in range(len(self.tiles[moved_owner].poly_ids)):
                        if self.tiles[moved_owner].poly_ids[pk] == last:
                            self.tiles[moved_owner].poly_ids[pk] = hole
                            break
                    self.polys[hole] = moved_poly^
                    self.poly_owner_tile[hole] = moved_owner

        # 3. Re-link portals inside dirty tiles and across their 4 tile borders
        for di in range(len(dirty_tids)):
            var tid = dirty_tids[di]
            self._link_within_tile(tid)
            var tx = self.tiles[tid].tx
            var tz = self.tiles[tid].tz
            var nb_coords = [(tx - 1, tz), (tx + 1, tz), (tx, tz - 1), (tx, tz + 1)]
            for d4 in range(4):
                var ntx = nb_coords[d4][0]
                var ntz = nb_coords[d4][1]
                if ntx < 0 or ntx >= self.tiles_x or ntz < 0 or ntz >= self.tiles_z:
                    continue
                var ntid = ntz * self.tiles_x + ntx
                if dirty_mask[ntid] and ntid < tid:
                    continue
                self._link_between_tiles(tid, ntid)

        if len(self.off_mesh_links) > 0:
            self._rebind_off_mesh_links()

    def _mark_obs_tiles(self, obs: NavObstacle, mut dirty_mask: List[Bool]):
        var tb = self._obstacle_tile_bounds(obs)
        if tb[0] <= tb[1] and tb[2] <= tb[3]:
            for tz in range(tb[2], tb[3] + 1):
                for tx in range(tb[0], tb[1] + 1):
                    dirty_mask[tz * self.tiles_x + tx] = True

    def add_obstacle(mut self, obs: NavObstacle) -> Int:
        var idx = len(self.obstacles)
        self.obstacles.append(obs)
        if len(self.tiles) > 0 and obs.active:
            var mask = List[Bool](length=len(self.tiles), fill=False)
            self._mark_obs_tiles(obs, mask)
            self._rebuild_dirty_tiles(mask)
        return idx

    def update_obstacle(mut self, obs: NavObstacle):
        var mask = List[Bool](length=len(self.tiles), fill=False)
        var found = False
        for i in range(len(self.obstacles)):
            if self.obstacles[i].id == obs.id:
                if len(mask) > 0:
                    self._mark_obs_tiles(self.obstacles[i], mask)
                self.obstacles[i] = obs
                if len(mask) > 0:
                    self._mark_obs_tiles(obs, mask)
                found = True
        if not found:
            self.obstacles.append(obs)
            if len(mask) > 0 and obs.active:
                self._mark_obs_tiles(obs, mask)
        if len(mask) > 0:
            self._rebuild_dirty_tiles(mask)

    def set_obstacle_active(mut self, obs_id: Int, active: Bool):
        var mask = List[Bool](length=len(self.tiles), fill=False)
        for i in range(len(self.obstacles)):
            if self.obstacles[i].id == obs_id:
                if self.obstacles[i].active != active:
                    self.obstacles[i].active = active
                    if len(mask) > 0:
                        self._mark_obs_tiles(self.obstacles[i], mask)
        if len(mask) > 0:
            self._rebuild_dirty_tiles(mask)

    def remove_obstacle(mut self, obs_id: Int):
        var mask = List[Bool](length=len(self.tiles), fill=False)
        var kept = List[NavObstacle](capacity=len(self.obstacles))
        for i in range(len(self.obstacles)):
            if self.obstacles[i].id == obs_id:
                if len(mask) > 0:
                    self._mark_obs_tiles(self.obstacles[i], mask)
            else:
                kept.append(self.obstacles[i])
        self.obstacles = kept^
        if len(mask) > 0:
            self._rebuild_dirty_tiles(mask)

    def carve_obstacles(mut self):
        """Re-carves all tiles touched by obstacles (or all tiles if resetting)."""
        if len(self.tiles) == 0:
            return
        var mask = List[Bool](length=len(self.tiles), fill=True)
        self._rebuild_dirty_tiles(mask)

    # =========================================================================
    # Off-Mesh Links, Binary-Heap A*, SSFA & NavPoly Visibility String-Pulling
    # =========================================================================

    def find_nearest_poly(self, p: Vec3, max_search_dist: Float32 = 10.0) -> Tuple[Int, Vec3]:
        var best_idx = -1
        var best_pt = p
        var best_d2 = max_search_dist * max_search_dist

        if len(self.tiles) > 0 and self.tiles_x > 0 and self.tiles_z > 0 and self.tile_size > 0:
            var tile_w = Float32(self.tile_size) * self.chf.cell_size
            var inv_tw = Float32(1.0) / max(Float32(1.0e-5), tile_w)
            var orig = self.chf.origin
            var tx0 = max(0, Int(floor((p.x - max_search_dist - orig.x) * inv_tw)))
            var tx1 = min(self.tiles_x - 1, Int(floor((p.x + max_search_dist - orig.x) * inv_tw)))
            var tz0 = max(0, Int(floor((p.z - max_search_dist - orig.z) * inv_tw)))
            var tz1 = min(self.tiles_z - 1, Int(floor((p.z + max_search_dist - orig.z) * inv_tw)))
            if tx0 > tx1 or tz0 > tz1:
                return (-1, p)

            # Check the tile containing `p` first to tighten `best_d2` before scanning neighbors
            var ctx = max(tx0, min(tx1, Int(floor((p.x - orig.x) * inv_tw))))
            var ctz = max(tz0, min(tz1, Int(floor((p.z - orig.z) * inv_tw))))
            ref c_pids = self.tiles[ctz * self.tiles_x + ctx].poly_ids
            for k in range(len(c_pids)):
                var pid = c_pids[k]
                var cp = self.polys[pid].closest_point_xz(p)
                var d2 = (cp - p).length_squared()
                if d2 < best_d2 or (d2 == best_d2 and (best_idx < 0 or pid < best_idx)):
                    best_d2 = d2
                    best_idx = pid
                    best_pt = cp

            for tz in range(tz0, tz1 + 1):
                var tile_z0 = orig.z + Float32(tz) * tile_w
                var tile_z1 = tile_z0 + tile_w
                var dz = Float32(0.0)
                if p.z < tile_z0:
                    dz = tile_z0 - p.z
                elif p.z > tile_z1:
                    dz = p.z - tile_z1
                var dz2 = dz * dz
                if dz2 > best_d2:
                    continue
                for tx in range(tx0, tx1 + 1):
                    if tx == ctx and tz == ctz:
                        continue
                    var tile_x0 = orig.x + Float32(tx) * tile_w
                    var tile_x1 = tile_x0 + tile_w
                    var dx = Float32(0.0)
                    if p.x < tile_x0:
                        dx = tile_x0 - p.x
                    elif p.x > tile_x1:
                        dx = p.x - tile_x1
                    if dx * dx + dz2 > best_d2:
                        continue
                    ref pids = self.tiles[tz * self.tiles_x + tx].poly_ids
                    for k in range(len(pids)):
                        var pid = pids[k]
                        var cp = self.polys[pid].closest_point_xz(p)
                        var d2 = (cp - p).length_squared()
                        if d2 < best_d2 or (d2 == best_d2 and (best_idx < 0 or pid < best_idx)):
                            best_d2 = d2
                            best_idx = pid
                            best_pt = cp
            return (best_idx, best_pt)

        for i in range(len(self.polys)):
            var cp = self.polys[i].closest_point_xz(p)
            var d2 = (cp - p).length_squared()
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
                best_pt = cp
        return (best_idx, best_pt)

    def _rebind_off_mesh_links(mut self):
        for i in range(len(self.off_mesh_links)):
            var s = self.find_nearest_poly(self.off_mesh_links[i].start_pos, max(Float32(2.0), self.off_mesh_links[i].radius * Float32(3.0)))
            var e = self.find_nearest_poly(self.off_mesh_links[i].end_pos, max(Float32(2.0), self.off_mesh_links[i].radius * Float32(3.0)))
            self.off_mesh_links[i].start_poly = s[0]
            self.off_mesh_links[i].end_poly = e[0]

    def add_off_mesh_link(mut self, var link: OffMeshLink) -> Int:
        var s = self.find_nearest_poly(link.start_pos, max(Float32(2.0), link.radius * Float32(3.0)))
        var e = self.find_nearest_poly(link.end_pos, max(Float32(2.0), link.radius * Float32(3.0)))
        link.start_poly = s[0]
        link.end_poly = e[0]
        var idx = len(self.off_mesh_links)
        self.off_mesh_links.append(link^)
        return idx

    def _find_poly_containing_probe(self, p: Vec3, ref_y: Float32, max_dy: Float32) -> Int:
        if len(self.tiles) > 0 and self.tiles_x > 0 and self.tiles_z > 0 and self.tile_size > 0:
            var tile_w = Float32(self.tile_size) * self.chf.cell_size
            var inv_tw = Float32(1.0) / max(Float32(1.0e-5), tile_w)
            var orig = self.chf.origin
            var pad = Float32(2.0e-3)
            var tx0 = max(0, Int(floor((p.x - pad - orig.x) * inv_tw)))
            var tx1 = min(self.tiles_x - 1, Int(floor((p.x + pad - orig.x) * inv_tw)))
            var tz0 = max(0, Int(floor((p.z - pad - orig.z) * inv_tw)))
            var tz1 = min(self.tiles_z - 1, Int(floor((p.z + pad - orig.z) * inv_tw)))
            for tz in range(tz0, tz1 + 1):
                for tx in range(tx0, tx1 + 1):
                    ref pids = self.tiles[tz * self.tiles_x + tx].poly_ids
                    for k in range(len(pids)):
                        var pid = pids[k]
                        if self.polys[pid].contains_xz(p, Float32(1.5e-3)):
                            if abs(self.polys[pid].sample_y(p.x, p.z) - ref_y) <= max_dy:
                                return pid
            return -1

        for i in range(len(self.polys)):
            if self.polys[i].contains_xz(p, Float32(1.5e-3)):
                if abs(self.polys[i].sample_y(p.x, p.z) - ref_y) <= max_dy:
                    return i
        return -1

    def raycast_walkable(self, a: Vec3, b: Vec3) -> Bool:
        """Exact Detour-style portal-graph line-of-sight check across convex `NavPoly`s.
        Guarantees 100% of segment `a -> b` stays inside `walkable_radius`-eroded `NavPoly`s
        without crossing cliffs, holes, or obstacle corners."""
        var dx = b.x - a.x
        var dz = b.z - a.z
        var len_xz = sqrt(dx * dx + dz * dz)
        if len_xz <= Float32(1.0e-4):
            return True

        var max_climb = self.cfg.walkable_climb + self.chf.cell_height * Float32(3.0) + Float32(0.5)
        var eps_step = min(Float32(0.15), Float32(2.5e-3) / len_xz)
        var p0 = a + (b - a) * eps_step
        var cur_poly = self._find_poly_containing_probe(p0, p0.y, max_climb + abs(b.y - a.y))
        if cur_poly < 0:
            return False

        var t_cur = eps_step
        var guard = 0
        while guard < 512:
            guard += 1
            if self.polys[cur_poly].contains_xz(b, Float32(1.5e-3)):
                var yb = self.polys[cur_poly].sample_y(b.x, b.z)
                return abs(yb - b.y) <= max_climb

            # Find earliest exiting edge of convex `cur_poly`
            ref vs = self.polys[cur_poly].verts
            var nv = len(vs)
            var t_exit = Float32(2.0)
            for e in range(nv):
                var v0 = vs[e]
                var v1 = vs[(e + 1) % nv]
                # Outward normal for (x_lo,z_lo)->(x_hi,z_lo)->(x_hi,z_hi)->(x_lo,z_hi) is (ez, -ex)
                var n_out_x = v1.z - v0.z
                var n_out_z = -(v1.x - v0.x)
                var denom = dx * n_out_x + dz * n_out_z
                if denom > Float32(1.0e-8):
                    var num = (v0.x - a.x) * n_out_x + (v0.z - a.z) * n_out_z
                    var te = num / denom
                    if te < t_exit:
                        t_exit = te

            if t_exit >= Float32(1.0) - Float32(1.0e-4):
                var yb = self.polys[cur_poly].sample_y(b.x, b.z)
                return abs(yb - b.y) <= max_climb
            if t_exit < t_cur - Float32(2.0e-3):
                return False

            var t_next = min(Float32(1.0), max(t_cur + eps_step * Float32(0.25), t_exit + eps_step))
            var p_next = a + (b - a) * t_next
            var p_exit = a + (b - a) * t_exit
            var cur_y = self.polys[cur_poly].sample_y(p_exit.x, p_exit.z)

            var next_poly = -1
            for ni in range(len(self.polys[cur_poly].neighbors)):
                var nb = self.polys[cur_poly].neighbors[ni]
                if self.polys[nb].contains_xz(p_next, Float32(1.5e-3)):
                    if abs(self.polys[nb].sample_y(p_exit.x, p_exit.z) - cur_y) <= max_climb:
                        next_poly = nb
                        break

            # Handle exact 4-tile grid-vertex crossings ONLY when all 4 meeting quadrants are walkable
            if next_poly < 0:
                var step_m = Float32(4.0e-3)
                var sx = step_m if dx >= Float32(0.0) else -step_m
                var sz = step_m if dz >= Float32(0.0) else -step_m
                var qx_ok = self._find_poly_containing_probe(Vec3(p_exit.x + sx, cur_y, p_exit.z - sz), cur_y, max_climb) >= 0
                var qz_ok = self._find_poly_containing_probe(Vec3(p_exit.x - sx, cur_y, p_exit.z + sz), cur_y, max_climb) >= 0
                if qx_ok and qz_ok:
                    for ni in range(len(self.polys[cur_poly].neighbors)):
                        var nb1 = self.polys[cur_poly].neighbors[ni]
                        if not self.polys[nb1].contains_xz(p_exit, Float32(3.0e-3)):
                            continue
                        for nj in range(len(self.polys[nb1].neighbors)):
                            var nb2 = self.polys[nb1].neighbors[nj]
                            if self.polys[nb2].contains_xz(p_next, Float32(2.0e-3)):
                                if abs(self.polys[nb2].sample_y(p_exit.x, p_exit.z) - cur_y) <= max_climb:
                                    next_poly = nb2
                                    break
                        if next_poly >= 0:
                            break

            if next_poly < 0:
                return False
            cur_poly = next_poly
            t_cur = t_next

        return False

    def _smooth_chain_los(self, chain: List[Vec3]) -> List[Vec3]:
        """Collapses mutually line-of-sight walkable waypoints into taut straight Euclidean segments."""
        var n = len(chain)
        if n <= 2:
            return chain.copy()
        var out = List[Vec3](capacity=n)
        var cur = 0
        out.append(chain[0])
        while cur < n - 1:
            var chosen = cur + 1
            var cand = n - 1
            while cand > cur + 1:
                if self.raycast_walkable(chain[cur], chain[cand]):
                    chosen = cand
                    break
                cand -= 1
            out.append(chain[chosen])
            cur = chosen
        return out^

    @staticmethod
    def _run_ssfa(
        start_pt: Vec3,
        end_pt: Vec3,
        p_left: List[Vec3],
        p_right: List[Vec3],
        mut out_pts: List[Vec3],
    ):
        """Mikko Mononen's Simple Stupid Funnel Algorithm (SSFA) producing exact taut corner waypoints."""
        var lefts = List[Vec3](capacity=len(p_left) + 2)
        var rights = List[Vec3](capacity=len(p_right) + 2)
        lefts.append(start_pt)
        rights.append(start_pt)
        for i in range(len(p_left)):
            lefts.append(p_left[i])
            rights.append(p_right[i])
        lefts.append(end_pt)
        rights.append(end_pt)

        var apex = start_pt
        var portal_left = start_pt
        var portal_right = start_pt
        var apex_idx: Int
        var left_idx = 0
        var right_idx = 0

        if len(out_pts) == 0 or (out_pts[len(out_pts) - 1] - start_pt).length_squared() > Float32(1.0e-6):
            out_pts.append(start_pt)

        var n_portals = len(lefts)
        var i = 1
        var guard = 0
        var max_iters = n_portals * n_portals + 32

        while i < n_portals and guard < max_iters:
            guard += 1
            var nl = lefts[i]
            var nr = rights[i]

            if _triarea2_xz(apex, portal_right, nr) <= Float32(0.0):
                var apex_eq_r = (apex.x - portal_right.x) * (apex.x - portal_right.x) + (
                    apex.z - portal_right.z
                ) * (apex.z - portal_right.z) <= Float32(1.0e-10)
                if apex_eq_r or _triarea2_xz(apex, portal_left, nr) > Float32(0.0):
                    portal_right = nr
                    right_idx = i
                else:
                    if (out_pts[len(out_pts) - 1] - portal_left).length_squared() > Float32(1.0e-6):
                        out_pts.append(portal_left)
                    apex = portal_left
                    apex_idx = left_idx
                    portal_left = apex
                    portal_right = apex
                    left_idx = apex_idx
                    right_idx = apex_idx
                    i = apex_idx + 1
                    continue

            if _triarea2_xz(apex, portal_left, nl) >= Float32(0.0):
                var apex_eq_l = (apex.x - portal_left.x) * (apex.x - portal_left.x) + (
                    apex.z - portal_left.z
                ) * (apex.z - portal_left.z) <= Float32(1.0e-10)
                if apex_eq_l or _triarea2_xz(apex, portal_right, nl) < Float32(0.0):
                    portal_left = nl
                    left_idx = i
                else:
                    if (out_pts[len(out_pts) - 1] - portal_right).length_squared() > Float32(1.0e-6):
                        out_pts.append(portal_right)
                    apex = portal_right
                    apex_idx = right_idx
                    portal_left = apex
                    portal_right = apex
                    left_idx = apex_idx
                    right_idx = apex_idx
                    i = apex_idx + 1
                    continue

            i += 1

        if (out_pts[len(out_pts) - 1] - end_pt).length_squared() > Float32(1.0e-6):
            out_pts.append(end_pt)

    @staticmethod
    @always_inline
    def _heap_key_f32_node(f_val: Float32, node: Int) -> Int:
        var bits = Int(float32_bits(max(Float32(0.0), f_val)))
        return (bits << 20) | (node & 0xFFFFF)

    def find_path(self, start: Vec3, goal: Vec3, max_search_dist: Float32 = 10.0) -> NavPath:
        """Deterministic Binary-Heap A* (`O(E log V)`) over convex `NavPoly`s + `OffMeshLink`s,
        followed by SSFA funnel string-pulling and exact `NavPoly` visibility smoothing."""
        var res = NavPath()
        var n_polys = len(self.polys)
        if n_polys == 0:
            return res^

        var sp = self.find_nearest_poly(start, max_search_dist)
        var gp = self.find_nearest_poly(goal, max_search_dist)
        var start_poly = sp[0]
        var goal_poly = gp[0]
        if start_poly < 0 or goal_poly < 0:
            return res^

        var start_snapped = sp[1]
        var goal_snapped = gp[1]

        if start_poly == goal_poly:
            res.found = True
            res.poly_corridor.append(start_poly)
            res.waypoints.append(start_snapped)
            if (goal_snapped - start_snapped).length_squared() > Float32(1.0e-6):
                res.waypoints.append(goal_snapped)
            res.total_length = (goal_snapped - start_snapped).length()
            return res^

        var g_score = List[Float32](length=n_polys, fill=Float32(1.0e30))
        var f_score = List[Float32](length=n_polys, fill=Float32(1.0e30))
        var entry_pt = List[Vec3](length=n_polys, fill=Vec3())
        var came_from = List[Int](length=n_polys, fill=-1)
        var came_link = List[Int](length=n_polys, fill=-1)
        var closed = List[Bool](length=n_polys, fill=False)

        var open_heap = MinHeap()
        g_score[start_poly] = Float32(0.0)
        entry_pt[start_poly] = start_snapped
        f_score[start_poly] = (goal_snapped - start_snapped).length()
        open_heap.push(Self._heap_key_f32_node(f_score[start_poly], start_poly), start_poly)

        var reached = False
        while len(open_heap) > 0:
            var best_u = open_heap.pop().value
            if closed[best_u]:
                continue
            closed[best_u] = True

            if best_u == goal_poly:
                reached = True
                break

            var u_anchor = entry_pt[best_u]

            # 1. Standard portal neighbors with portal-projected Euclidean entry point
            ref u_poly = self.polys[best_u]
            for ni in range(len(u_poly.neighbors)):
                var v = u_poly.neighbors[ni]
                if closed[v]:
                    continue
                var p_cross = _portal_best_crossing_xz(
                    u_anchor, goal_snapped, u_poly.portal_left[ni], u_poly.portal_right[ni]
                )
                var step_cost = (p_cross - u_anchor).length() * self.polys[v].area_cost
                if v == goal_poly:
                    step_cost += (goal_snapped - p_cross).length() * self.polys[v].area_cost
                var cand_g = g_score[best_u] + step_cost
                if cand_g < g_score[v]:
                    g_score[v] = cand_g
                    entry_pt[v] = p_cross
                    var h_v = Float32(0.0) if v == goal_poly else (goal_snapped - p_cross).length()
                    f_score[v] = cand_g + h_v
                    came_from[v] = best_u
                    came_link[v] = -1
                    open_heap.push(Self._heap_key_f32_node(f_score[v], v), v)

            # 2. Off-Mesh Links
            for li in range(len(self.off_mesh_links)):
                ref lnk = self.off_mesh_links[li]
                if not lnk.enabled or lnk.start_poly < 0 or lnk.end_poly < 0:
                    continue
                var v_lnk = -1
                var p_enter = Vec3()
                var p_exit = Vec3()
                if lnk.start_poly == best_u:
                    v_lnk = lnk.end_poly
                    p_enter = lnk.start_pos
                    p_exit = lnk.end_pos
                elif lnk.bidirectional and lnk.end_poly == best_u:
                    v_lnk = lnk.start_poly
                    p_enter = lnk.end_pos
                    p_exit = lnk.start_pos
                if v_lnk >= 0 and not closed[v_lnk]:
                    var lnk_cost = (p_enter - u_anchor).length() + (p_exit - p_enter).length() * lnk.cost_scale
                    if v_lnk == goal_poly:
                        lnk_cost += (goal_snapped - p_exit).length() * self.polys[v_lnk].area_cost
                    var cand_g = g_score[best_u] + lnk_cost
                    if cand_g < g_score[v_lnk]:
                        g_score[v_lnk] = cand_g
                        entry_pt[v_lnk] = p_exit
                        var h_lnk = Float32(0.0) if v_lnk == goal_poly else (goal_snapped - p_exit).length()
                        f_score[v_lnk] = cand_g + h_lnk
                        came_from[v_lnk] = best_u
                        came_link[v_lnk] = li
                        open_heap.push(Self._heap_key_f32_node(f_score[v_lnk], v_lnk), v_lnk)

        if not reached:
            return res^

        var rev_corridor = List[Int]()
        var rev_links = List[Int]()
        var cur = goal_poly
        while cur >= 0:
            rev_corridor.append(cur)
            rev_links.append(came_link[cur])
            if cur == start_poly:
                break
            cur = came_from[cur]

        var clen = len(rev_corridor)
        var step_links = List[Int](capacity=clen)
        for k in range(clen):
            var idx_rev = clen - 1 - k
            res.poly_corridor.append(rev_corridor[idx_rev])
            step_links.append(rev_links[idx_rev])

        # Run SSFA + NavPoly LOS smoothing across each contiguous portal segment between OffMeshLinks
        var seg_start = start_snapped
        var seg_lefts = List[Vec3]()
        var seg_rights = List[Vec3]()

        for k in range(len(res.poly_corridor) - 1):
            var u = res.poly_corridor[k]
            var v = res.poly_corridor[k + 1]
            var lnk_idx = step_links[k + 1]
            if lnk_idx >= 0:
                ref lnk = self.off_mesh_links[lnk_idx]
                var p_enter = lnk.start_pos if lnk.start_poly == u else lnk.end_pos
                var p_exit = lnk.end_pos if lnk.start_poly == u else lnk.start_pos
                var seg_wps = List[Vec3]()
                Self._run_ssfa(seg_start, p_enter, seg_lefts, seg_rights, seg_wps)
                var smooth_seg = self._smooth_chain_los(seg_wps)
                for si in range(len(smooth_seg)):
                    if len(res.waypoints) == 0 or (res.waypoints[len(res.waypoints) - 1] - smooth_seg[si]).length_squared() > Float32(1.0e-6):
                        res.waypoints.append(smooth_seg[si])
                res.waypoints.append(p_exit)
                res.used_off_mesh_links.append(lnk.id)
                seg_start = p_exit
                seg_lefts.clear()
                seg_rights.clear()
            else:
                for ni in range(len(self.polys[u].neighbors)):
                    if self.polys[u].neighbors[ni] == v:
                        seg_lefts.append(self.polys[u].portal_left[ni])
                        seg_rights.append(self.polys[u].portal_right[ni])
                        break

        var final_wps = List[Vec3]()
        Self._run_ssfa(seg_start, goal_snapped, seg_lefts, seg_rights, final_wps)
        var smooth_final = self._smooth_chain_los(final_wps)
        for si in range(len(smooth_final)):
            if len(res.waypoints) == 0 or (res.waypoints[len(res.waypoints) - 1] - smooth_final[si]).length_squared() > Float32(1.0e-6):
                res.waypoints.append(smooth_final[si])

        var dist_sum = Float32(0.0)
        for i in range(len(res.waypoints) - 1):
            dist_sum += (res.waypoints[i + 1] - res.waypoints[i]).length()
        res.total_length = dist_sum
        res.found = True
        return res^

    # =========================================================================
    # Binary Serialization (`NAVM` versioned blob)
    # =========================================================================

    def serialize(self, mut w: BlobWriter):
        """Writes complete tiled `NavMesh` (config, compact heightfield, tiles, polygons,
        portals, obstacles, and off-mesh links) as a bit-exact versioned `'NAVM'` blob."""
        w.begin(NAVMESH_BLOB_TAG, NAVMESH_BLOB_VERSION)

        # 1. Config
        w.f32(self.cfg.cell_size)
        w.f32(self.cfg.cell_height)
        w.f32(self.cfg.walkable_slope_deg)
        w.f32(self.cfg.walkable_height)
        w.f32(self.cfg.walkable_climb)
        w.f32(self.cfg.walkable_radius)
        w.f32(self.cfg.max_edge_len)
        w.f32(self.cfg.max_simplification_error)
        w.integer(self.cfg.min_region_cells)
        w.integer(self.cfg.tile_size)
        w.integer(self.cfg.max_verts_per_poly)

        # 2. CompactHeightfield
        w.integer(self.chf.width)
        w.integer(self.chf.depth)
        w.vec3(self.chf.origin)
        w.f32(self.chf.cell_size)
        w.f32(self.chf.cell_height)
        w.integer(len(self.chf.col_first))
        for i in range(len(self.chf.col_first)):
            w.integer(Int(self.chf.col_first[i]))
            w.integer(Int(self.chf.col_count[i]))
        w.integer(len(self.chf.spans))
        for i in range(len(self.chf.spans)):
            ref sp = self.chf.spans[i]
            w.integer(Int(sp.x))
            w.integer(Int(sp.z))
            w.integer(Int(sp.y))
            w.integer(Int(sp.ceiling))
            w.u8(sp.base_area)
            w.u8(sp.area)
            w.integer(Int(sp.dist))
            w.integer(Int(sp.reg))
            w.integer(Int(sp.n0))
            w.integer(Int(sp.n1))
            w.integer(Int(sp.n2))
            w.integer(Int(sp.n3))

        # 3. Tile grid
        w.integer(self.tiles_x)
        w.integer(self.tiles_z)
        w.integer(self.tile_size)
        w.integer(len(self.tiles))
        for ti in range(len(self.tiles)):
            ref t = self.tiles[ti]
            w.integer(t.tx)
            w.integer(t.tz)
            w.integer(t.x0)
            w.integer(t.x1)
            w.integer(t.z0)
            w.integer(t.z1)
            w.integer(len(t.span_indices))
            for k in range(len(t.span_indices)):
                w.integer(Int(t.span_indices[k]))
            w.integer(len(t.poly_ids))
            for k in range(len(t.poly_ids)):
                w.integer(t.poly_ids[k])

        # 4. Polygons & portals
        w.integer(len(self.polys))
        for pi in range(len(self.polys)):
            ref p = self.polys[pi]
            w.integer(len(p.verts))
            for k in range(len(p.verts)):
                w.vec3(p.verts[k])
            w.vec3(p.center)
            w.vec3(p.plane_n)
            w.f32(p.plane_d)
            w.integer(len(p.neighbors))
            for k in range(len(p.neighbors)):
                w.integer(p.neighbors[k])
                w.vec3(p.portal_left[k])
                w.vec3(p.portal_right[k])
            w.f32(p.area_cost)
            w.integer(Int(p.region_id))

        w.integer(len(self.poly_owner_tile))
        for i in range(len(self.poly_owner_tile)):
            w.integer(self.poly_owner_tile[i])
        w.integer(Int(self.next_region_id))

        # 5. Obstacles
        w.integer(len(self.obstacles))
        for i in range(len(self.obstacles)):
            ref obs = self.obstacles[i]
            w.integer(obs.id)
            w.u8(obs.shape)
            w.vec3(obs.pos)
            w.f32(obs.rot.x); w.f32(obs.rot.y); w.f32(obs.rot.z); w.f32(obs.rot.w)
            w.vec3(obs.half_extents)
            w.f32(obs.radius)
            w.f32(obs.half_height)
            w.boolean(obs.active)

        # 6. Off-mesh links
        w.integer(len(self.off_mesh_links))
        for i in range(len(self.off_mesh_links)):
            ref lnk = self.off_mesh_links[i]
            w.integer(lnk.id)
            w.vec3(lnk.start_pos)
            w.vec3(lnk.end_pos)
            w.f32(lnk.radius)
            w.u8(lnk.kind)
            w.boolean(lnk.bidirectional)
            w.f32(lnk.cost_scale)
            w.boolean(lnk.enabled)
            w.integer(lnk.start_poly)
            w.integer(lnk.end_poly)

        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        """Restores a `NavMesh` from a `'NAVM'` binary blob."""
        var ver = r.enter(NAVMESH_BLOB_TAG, NAVMESH_BLOB_VERSION)
        _ = ver

        # 1. Config
        self.cfg = NavMeshConfig()
        self.cfg.cell_size = r.f32()
        self.cfg.cell_height = r.f32()
        self.cfg.walkable_slope_deg = r.f32()
        self.cfg.walkable_height = r.f32()
        self.cfg.walkable_climb = r.f32()
        self.cfg.walkable_radius = r.f32()
        self.cfg.max_edge_len = r.f32()
        self.cfg.max_simplification_error = r.f32()
        self.cfg.min_region_cells = r.integer()
        self.cfg.tile_size = r.integer()
        self.cfg.max_verts_per_poly = r.integer()

        # 2. CompactHeightfield
        self.chf = CompactHeightfield()
        self.chf.width = r.integer()
        self.chf.depth = r.integer()
        self.chf.origin = r.vec3()
        self.chf.cell_size = r.f32()
        self.chf.cell_height = r.f32()
        var n_cols = r.integer()
        if r.failed or n_cols < 0 or n_cols > (1 << 22) or n_cols > max(0, r.limit() - r.offset) // 16:
            raise Error("navmesh blob: invalid column count")
        self.chf.col_first = List[Int32](capacity=n_cols)
        self.chf.col_count = List[UInt16](capacity=n_cols)
        for _ in range(n_cols):
            var cf = r.integer()
            var cc = r.integer()
            if r.failed or cf < 0 or cc < 0 or cc > 65535:
                raise Error("navmesh blob: invalid column range")
            self.chf.col_first.append(Int32(cf))
            self.chf.col_count.append(UInt16(cc))
        var n_spans = r.integer()
        if r.failed or n_spans < 0 or n_spans > (1 << 22) or n_spans > max(0, r.limit() - r.offset) // 82:
            raise Error("navmesh blob: invalid span count")
        for ci in range(n_cols):
            var cf = Int(self.chf.col_first[ci])
            var cc = Int(self.chf.col_count[ci])
            if cf > n_spans or cc > n_spans - cf:
                raise Error("navmesh blob: column span range out of bounds")
        self.chf.spans = List[CompactSpan](capacity=n_spans)
        for _ in range(n_spans):
            var sx = Int32(r.integer())
            var sz = Int32(r.integer())
            var sy = UInt16(r.integer())
            var scei = UInt16(r.integer())
            var ba = r.u8()
            var ar = r.u8()
            var dist = UInt16(r.integer())
            var reg = Int32(r.integer())
            var n0 = r.integer()
            var n1 = r.integer()
            var n2 = r.integer()
            var n3 = r.integer()
            if (
                r.failed
                or n0 < -1 or n0 >= n_spans
                or n1 < -1 or n1 >= n_spans
                or n2 < -1 or n2 >= n_spans
                or n3 < -1 or n3 >= n_spans
            ):
                raise Error("navmesh blob: span neighbor out of bounds")
            self.chf.spans.append(CompactSpan(sx, sz, sy, scei, ba, ar, dist, reg, Int32(n0), Int32(n1), Int32(n2), Int32(n3)))

        # 3. Tile grid
        self.tiles_x = r.integer()
        self.tiles_z = r.integer()
        self.tile_size = r.integer()
        var n_tiles = r.integer()
        if r.failed or n_tiles < 0 or n_tiles > (1 << 16) or n_tiles > max(0, r.limit() - r.offset) // 64:
            raise Error("navmesh blob: invalid tile count")
        self.tiles = List[NavTile](capacity=n_tiles)
        for _ in range(n_tiles):
            var tx = r.integer()
            var tz = r.integer()
            var x0 = r.integer()
            var x1 = r.integer()
            var z0 = r.integer()
            var z1 = r.integer()
            var t = NavTile(tx, tz, x0, x1, z0, z1)
            var n_sp = r.integer()
            if r.failed or n_sp < 0 or n_sp > (1 << 22) or n_sp > max(0, r.limit() - r.offset) // 8:
                raise Error("navmesh blob: invalid tile span count")
            t.span_indices = List[Int32](capacity=n_sp)
            for _ in range(n_sp):
                var sp_idx = r.integer()
                if r.failed or sp_idx < 0 or sp_idx >= n_spans:
                    raise Error("navmesh blob: tile span index out of bounds")
                t.span_indices.append(Int32(sp_idx))
            var n_p = r.integer()
            if r.failed or n_p < 0 or n_p > (1 << 22) or n_p > max(0, r.limit() - r.offset) // 8:
                raise Error("navmesh blob: invalid tile poly count")
            t.poly_ids = List[Int](capacity=n_p)
            for _ in range(n_p):
                t.poly_ids.append(r.integer())
            self.tiles.append(t^)

        # 4. Polygons & portals
        var n_polys = r.integer()
        if r.failed or n_polys < 0 or n_polys > (1 << 20) or n_polys > max(0, r.limit() - r.offset) // 56:
            raise Error("navmesh blob: invalid poly count")
        for ti in range(n_tiles):
            for pk in range(len(self.tiles[ti].poly_ids)):
                var pid = self.tiles[ti].poly_ids[pk]
                if pid < 0 or pid >= n_polys:
                    raise Error("navmesh blob: tile poly_id out of bounds")
        self.polys = List[NavPoly](capacity=n_polys)
        for _ in range(n_polys):
            var p = NavPoly()
            var nv = r.integer()
            if r.failed or nv < 0 or nv > (1 << 16) or nv > max(0, r.limit() - r.offset) // 12:
                raise Error("navmesh blob: invalid poly vertex count")
            p.verts = List[Vec3](capacity=nv)
            for _ in range(nv):
                p.verts.append(r.vec3())
            p.center = r.vec3()
            p.plane_n = r.vec3()
            p.plane_d = r.f32()
            var nn = r.integer()
            if r.failed or nn < 0 or nn > (1 << 16) or nn > max(0, r.limit() - r.offset) // 32:
                raise Error("navmesh blob: invalid poly neighbor count")
            p.neighbors = List[Int](capacity=nn)
            p.portal_left = List[Vec3](capacity=nn)
            p.portal_right = List[Vec3](capacity=nn)
            for _ in range(nn):
                var nb = r.integer()
                if r.failed or nb < 0 or nb >= n_polys:
                    raise Error("navmesh blob: poly neighbor out of bounds")
                p.neighbors.append(nb)
                p.portal_left.append(r.vec3())
                p.portal_right.append(r.vec3())
            if len(p.portal_left) != len(p.neighbors) or len(p.portal_right) != len(p.neighbors):
                raise Error("navmesh blob: mismatched portal arrays")
            p.area_cost = r.f32()
            p.region_id = Int32(r.integer())
            self.polys.append(p^)

        var n_own = r.integer()
        if r.failed or n_own != n_polys or n_own > max(0, r.limit() - r.offset) // 8:
            raise Error("navmesh blob: invalid poly owner count")
        self.poly_owner_tile = List[Int](capacity=n_own)
        for _ in range(n_own):
            var owner_t = r.integer()
            if r.failed or owner_t < 0 or owner_t >= n_tiles:
                raise Error("navmesh blob: poly_owner_tile out of bounds")
            self.poly_owner_tile.append(owner_t)
        self.next_region_id = Int32(r.integer())

        # 5. Obstacles
        var n_obs = r.integer()
        if r.failed or n_obs < 0 or n_obs > (1 << 16) or n_obs > max(0, r.limit() - r.offset) // 58:
            raise Error("navmesh blob: invalid obstacle count")
        self.obstacles = List[NavObstacle](capacity=n_obs)
        for _ in range(n_obs):
            var oid = r.integer()
            var osh = r.u8()
            var opos = r.vec3()
            var qx = r.f32(); var qy = r.f32(); var qz = r.f32(); var qw = r.f32()
            var ohe = r.vec3()
            var orad = r.f32()
            var ohh = r.f32()
            var oact = r.boolean()
            self.obstacles.append(NavObstacle(oid, osh, opos, Quat(qx, qy, qz, qw), ohe, orad, ohh, oact))

        # 6. Off-mesh links
        var n_lnk = r.integer()
        if r.failed or n_lnk < 0 or n_lnk > (1 << 16) or n_lnk > max(0, r.limit() - r.offset) // 59:
            raise Error("navmesh blob: invalid link count")
        self.off_mesh_links = List[OffMeshLink](capacity=n_lnk)
        for _ in range(n_lnk):
            var lid = r.integer()
            var sp = r.vec3()
            var ep = r.vec3()
            var lrad = r.f32()
            var lkind = r.u8()
            var lbi = r.boolean()
            var lcs = r.f32()
            var lenb = r.boolean()
            var lstart = r.integer()
            var lend = r.integer()
            if (
                r.failed
                or lstart < -1 or lstart >= n_polys
                or lend < -1 or lend >= n_polys
            ):
                raise Error("navmesh blob: off_mesh_link poly index out of bounds")
            self.off_mesh_links.append(OffMeshLink(lid, sp, ep, lrad, lkind, lbi, lcs, lenb, lstart, lend))

        r.leave()
