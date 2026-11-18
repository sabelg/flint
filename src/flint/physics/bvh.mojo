"""BVH8 static triangle-mesh collider with O(N) signed dihedral active-edge smoothing (v4).

Implements:
1. Multi-level recursive 16-bin Surface Area Heuristic (SAH) BVH collapsed top-down
   into 8-wide SIMD `Bvh8Node` packets.
2. O(N) quantized undirected edge hash table (`detsort.mix64`) for signed dihedral
   active-edge smoothing: marks shared edges inactive when concave/coplanar
   `((v_opp - v0) . n1 >= -1e-4)` or smooth convex `(n1 . n2 > 0.996)`, with
   Voronoi boundary-vertex normal clamping in `closest_point`.
3. True 8-wide `F8` SIMD slab traversal ordered by near `t` in `raycast` and
   `sweep_sphere`, plus `swept_sphere_moving_bvh8` and discrete `collide_sphere`,
   `collide_capsule`, and `collide_box` returning 4-point `ContactManifold`s.
"""
from std.collections import List
from std.math import sqrt, min, max, floor, abs
from flint.core.math import Vec3, Quat
from flint.core.detsort import mix64
from flint.physics.solver_bodies import F8, I8, LANES
from flint.physics.narrowphase import (
    ContactPoint,
    ContactManifold,
    reduce_manifold_candidates,
    get_box_axes,
    closest_points_segments,
)


comptime BVH8_CHILDREN: Int = 8
comptime _SAH_BINS: Int = 16
comptime _MAX_LEAF_TRIS: Int = 4
comptime EDGE_ACTIVE_01: UInt8 = 1
comptime EDGE_ACTIVE_12: UInt8 = 2
comptime EDGE_ACTIVE_20: UInt8 = 4
comptime EDGE_ACTIVE_ALL: UInt8 = 7


@always_inline
def vec3_eq(a: Vec3, b: Vec3) -> Bool:
    return abs(a.x - b.x) < 1.0e-5 and abs(a.y - b.y) < 1.0e-5 and abs(a.z - b.z) < 1.0e-5


@always_inline
def _clamp_vertex_normal(
    d: Vec3,
    face_n: Vec3,
    act_a: Bool,
    m_a: Vec3,
    act_b: Bool,
    m_b: Vec3,
) -> Vec3:
    var d2 = d.length_squared()
    if d2 <= Float32(1.0e-8):
        return face_n
    if act_a and act_b:
        return d * (Float32(1.0) / sqrt(d2))
    if (not act_a) and (not act_b):
        return face_n
    var m_act = m_a if act_a else m_b
    var m_inact = m_b if act_a else m_a
    if d.dot(m_act) <= Float32(1.0e-5):
        return face_n
    var dc = d
    var p_inact = dc.dot(m_inact)
    if p_inact > Float32(0.0):
        dc = dc - m_inact * p_inact
    var dc2 = dc.length_squared()
    return dc * (Float32(1.0) / sqrt(dc2)) if dc2 > Float32(1.0e-8) else face_n


@fieldwise_init
struct MeshTriangle(Copyable, ImplicitlyCopyable, Movable):
    """Static mesh triangle with precomputed plane normal and active-edge mask."""
    var v0: Vec3
    var v1: Vec3
    var v2: Vec3
    var normal: Vec3
    var active_edges: UInt8  # bit 0: (v0, v1), bit 1: (v1, v2), bit 2: (v2, v0)
    var material_id: UInt16

    def __init__(out self, v0: Vec3, v1: Vec3, v2: Vec3, material_id: UInt16 = 0):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        var e1 = v1 - v0
        var e2 = v2 - v0
        var n = e1.cross(e2)
        var l = n.length()
        self.normal = n * (Float32(1.0) / l) if l > Float32(1.0e-7) else Vec3(0, 1, 0)
        self.active_edges = EDGE_ACTIVE_ALL
        self.material_id = material_id

    @always_inline
    def bounds(self) -> Tuple[Vec3, Vec3]:
        var lo = Vec3(
            min(self.v0.x, min(self.v1.x, self.v2.x)),
            min(self.v0.y, min(self.v1.y, self.v2.y)),
            min(self.v0.z, min(self.v1.z, self.v2.z)),
        )
        var hi = Vec3(
            max(self.v0.x, max(self.v1.x, self.v2.x)),
            max(self.v0.y, max(self.v1.y, self.v2.y)),
            max(self.v0.z, max(self.v1.z, self.v2.z)),
        )
        return (lo, hi)

    @always_inline
    def center(self) -> Vec3:
        return (self.v0 + self.v1 + self.v2) * Float32(1.0 / 3.0)

    def closest_point(self, p: Vec3) -> Tuple[Vec3, Vec3]:
        """Calculates closest point on triangle to p with active-edge and boundary-vertex smoothing."""
        var ab = self.v1 - self.v0
        var ac = self.v2 - self.v0
        var ap = p - self.v0

        var d1 = ab.dot(ap)
        var d2 = ac.dot(ap)
        if d1 <= 0.0 and d2 <= 0.0:
            var act01 = (self.active_edges & EDGE_ACTIVE_01) != 0
            var act20 = (self.active_edges & EDGE_ACTIVE_20) != 0
            var m01 = ab.cross(self.normal).normalized()
            var m20 = (self.v0 - self.v2).cross(self.normal).normalized()
            var n = _clamp_vertex_normal(ap, self.normal, act01, m01, act20, m20)
            return (self.v0, n)

        var bp = p - self.v1
        var d3 = ab.dot(bp)
        var d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            var act01 = (self.active_edges & EDGE_ACTIVE_01) != 0
            var act12 = (self.active_edges & EDGE_ACTIVE_12) != 0
            var m01 = ab.cross(self.normal).normalized()
            var m12 = (self.v2 - self.v1).cross(self.normal).normalized()
            var n = _clamp_vertex_normal(bp, self.normal, act01, m01, act12, m12)
            return (self.v1, n)

        var vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            var v = d1 / (d1 - d3)
            var pt = self.v0 + ab * v
            var n = (p - pt).normalized() if (p - pt).length_squared() > 1.0e-8 else self.normal
            if (self.active_edges & EDGE_ACTIVE_01) == 0:
                n = self.normal
            return (pt, n)

        var cp = p - self.v2
        var d5 = ab.dot(cp)
        var d6 = ac.dot(cp)
        if d6 >= 0.0 and d5 <= d6:
            var act12 = (self.active_edges & EDGE_ACTIVE_12) != 0
            var act20 = (self.active_edges & EDGE_ACTIVE_20) != 0
            var m12 = (self.v2 - self.v1).cross(self.normal).normalized()
            var m20 = (self.v0 - self.v2).cross(self.normal).normalized()
            var n = _clamp_vertex_normal(cp, self.normal, act12, m12, act20, m20)
            return (self.v2, n)

        var vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            var w = d2 / (d2 - d6)
            var pt = self.v0 + ac * w
            var n = (p - pt).normalized() if (p - pt).length_squared() > 1.0e-8 else self.normal
            if (self.active_edges & EDGE_ACTIVE_20) == 0:
                n = self.normal
            return (pt, n)

        var va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            var w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            var pt = self.v1 + (self.v2 - self.v1) * w
            var n = (p - pt).normalized() if (p - pt).length_squared() > 1.0e-8 else self.normal
            if (self.active_edges & EDGE_ACTIVE_12) == 0:
                n = self.normal
            return (pt, n)

        var denom = Float32(1.0) / (va + vb + vc)
        var v_coord = vb * denom
        var w_coord = vc * denom
        var pt = self.v0 + ab * v_coord + ac * w_coord
        return (pt, self.normal)

    def ray_intersect(self, orig: Vec3, dir: Vec3, max_t: Float32) -> Tuple[Bool, Float32, Vec3]:
        """Möller-Trumbore ray-triangle intersection."""
        var e1 = self.v1 - self.v0
        var e2 = self.v2 - self.v0
        var pvec = dir.cross(e2)
        var det = e1.dot(pvec)
        if abs(det) < 1.0e-7:
            return (False, max_t, self.normal)
        var inv_det = Float32(1.0) / det
        var tvec = orig - self.v0
        var u = tvec.dot(pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return (False, max_t, self.normal)
        var qvec = tvec.cross(e1)
        var v = dir.dot(qvec) * inv_det
        if v < 0.0 or (u + v) > 1.0:
            return (False, max_t, self.normal)
        var t = e2.dot(qvec) * inv_det
        if t >= 0.0 and t <= max_t:
            return (True, t, self.normal)
        return (False, max_t, self.normal)


@fieldwise_init
struct Bvh8Node(Copyable, ImplicitlyCopyable, Movable):
    """Packed 8-wide BVH node with SIMD child bounding boxes.
    `child_indices[c] > 0` and `child_counts[c] == 0`: internal `Bvh8Node` index.
    `child_indices[c] < 0` and `child_counts[c] > 0`: leaf triangles `[-(child_indices[c] + 1), + child_counts[c])`.
    `child_indices[c] == 0` and `child_counts[c] == 0`: empty slot.
    """
    var min_x: F8
    var min_y: F8
    var min_z: F8
    var max_x: F8
    var max_y: F8
    var max_z: F8
    var child_indices: I8
    var child_counts: I8

    def __init__(out self):
        self.min_x = F8(1.0e30)
        self.min_y = F8(1.0e30)
        self.min_z = F8(1.0e30)
        self.max_x = F8(-1.0e30)
        self.max_y = F8(-1.0e30)
        self.max_z = F8(-1.0e30)
        self.child_indices = I8(0)
        self.child_counts = I8(0)


@fieldwise_init
struct RayHit(Copyable, ImplicitlyCopyable, Movable):
    var hit: Bool
    var t: Float32
    var point: Vec3
    var normal: Vec3
    var triangle_index: Int
    var material_id: UInt16

    def __init__(out self):
        self.hit = False
        self.t = 1.0e30
        self.point = Vec3()
        self.normal = Vec3(0, 1, 0)
        self.triangle_index = -1
        self.material_id = 0


@fieldwise_init
struct SweepHit(Copyable, ImplicitlyCopyable, Movable):
    var hit: Bool
    var toi: Float32
    var point: Vec3
    var normal: Vec3
    var triangle_index: Int

    def __init__(out self):
        self.hit = False
        self.toi = 1.0e30
        self.point = Vec3()
        self.normal = Vec3(0, 1, 0)
        self.triangle_index = -1


@always_inline
def _quant_vert(v: Vec3) -> Tuple[Int64, Int64, Int64]:
    var qx = Int64(floor(Float64(v.x) * 10000.0 + 0.5))
    var qy = Int64(floor(Float64(v.y) * 10000.0 + 0.5))
    var qz = Int64(floor(Float64(v.z) * 10000.0 + 0.5))
    return (qx, qy, qz)


@always_inline
def _undirected_edge_hash(a: Vec3, b: Vec3) -> UInt64:
    var qa = _quant_vert(a)
    var qb = _quant_vert(b)
    var swap: Bool
    if qa[0] != qb[0]:
        swap = qa[0] > qb[0]
    elif qa[1] != qb[1]:
        swap = qa[1] > qb[1]
    else:
        swap = qa[2] > qb[2]
    var p0 = qb if swap else qa
    var p1 = qa if swap else qb
    var h0 = mix64(UInt64(p0[0]) ^ mix64(UInt64(p0[1]) ^ mix64(UInt64(p0[2]))))
    var h1 = mix64(UInt64(p1[0]) ^ mix64(UInt64(p1[1]) ^ mix64(UInt64(p1[2]))))
    var key = mix64(h0 ^ (h1 * 0x9E3779B97F4A7C15))
    return key if key != 0 else UInt64(1)


@always_inline
def _edge_endpoints(tri: MeshTriangle, edge_idx: Int) -> Tuple[Vec3, Vec3, Vec3]:
    """Returns `(edge_v0, edge_v1, opp_v)` for edge 0, 1, or 2."""
    if edge_idx == 0:
        return (tri.v0, tri.v1, tri.v2)
    if edge_idx == 1:
        return (tri.v1, tri.v2, tri.v0)
    return (tri.v2, tri.v0, tri.v1)


@always_inline
def _edge_bit(edge_idx: Int) -> UInt8:
    if edge_idx == 0:
        return EDGE_ACTIVE_01
    if edge_idx == 1:
        return EDGE_ACTIVE_12
    return EDGE_ACTIVE_20


@always_inline
def _aabb_half_area(lo: Vec3, hi: Vec3) -> Float32:
    var dx = max(Float32(0.0), hi.x - lo.x)
    var dy = max(Float32(0.0), hi.y - lo.y)
    var dz = max(Float32(0.0), hi.z - lo.z)
    return dx * dy + dy * dz + dz * dx


@fieldwise_init
struct _MeshBinNode(Copyable, ImplicitlyCopyable, Movable):
    var lo: Vec3
    var hi: Vec3
    var left: Int32   # >= 0: left child; < 0: -(first_tri + 1)
    var right: Int32  # >= 0: right child; < 0: -tri_count (when leaf)


def _build_mesh_bvh2(
    tris: List[MeshTriangle],
    t_lo: List[Vec3],
    t_hi: List[Vec3],
    t_center: List[Vec3],
    mut perm: List[Int],
    start: Int,
    end: Int,
    mut out_nodes: List[_MeshBinNode],
) -> Int:
    var node_idx = len(out_nodes)
    var count = end - start

    var n_lo = Vec3(1.0e30, 1.0e30, 1.0e30)
    var n_hi = Vec3(-1.0e30, -1.0e30, -1.0e30)
    var c_lo = Vec3(1.0e30, 1.0e30, 1.0e30)
    var c_hi = Vec3(-1.0e30, -1.0e30, -1.0e30)

    for i in range(start, end):
        var ti = perm[i]
        var blo = t_lo[ti]
        var bhi = t_hi[ti]
        var bc = t_center[ti]
        n_lo.x = min(n_lo.x, blo.x); n_lo.y = min(n_lo.y, blo.y); n_lo.z = min(n_lo.z, blo.z)
        n_hi.x = max(n_hi.x, bhi.x); n_hi.y = max(n_hi.y, bhi.y); n_hi.z = max(n_hi.z, bhi.z)
        c_lo.x = min(c_lo.x, bc.x); c_lo.y = min(c_lo.y, bc.y); c_lo.z = min(c_lo.z, bc.z)
        c_hi.x = max(c_hi.x, bc.x); c_hi.y = max(c_hi.y, bc.y); c_hi.z = max(c_hi.z, bc.z)

    if count <= _MAX_LEAF_TRIS:
        out_nodes.append(_MeshBinNode(n_lo, n_hi, Int32(-(start + 1)), Int32(-count)))
        return node_idx

    out_nodes.append(_MeshBinNode(n_lo, n_hi, Int32(0), Int32(0)))

    # Evaluate 16-bin SAH across all 3 axes
    var best_cost = Float32(1.0e30)
    var best_axis = -1
    var best_split = -1

    var spans = [c_hi.x - c_lo.x, c_hi.y - c_lo.y, c_hi.z - c_lo.z]
    var c_mins = [c_lo.x, c_lo.y, c_lo.z]

    for ax in range(3):
        if spans[ax] <= Float32(1.0e-6):
            continue
        var inv_span = Float32(_SAH_BINS) / spans[ax]
        var c_min = c_mins[ax]
        var bin_cnt = Array[Int, _SAH_BINS](fill=0)
        var b_lox = Array[Float32, _SAH_BINS](fill=1.0e30)
        var b_loy = Array[Float32, _SAH_BINS](fill=1.0e30)
        var b_loz = Array[Float32, _SAH_BINS](fill=1.0e30)
        var b_hix = Array[Float32, _SAH_BINS](fill=-1.0e30)
        var b_hiy = Array[Float32, _SAH_BINS](fill=-1.0e30)
        var b_hiz = Array[Float32, _SAH_BINS](fill=-1.0e30)

        for i in range(start, end):
            var ti = perm[i]
            var bc = t_center[ti]
            var coord = bc.x if ax == 0 else (bc.y if ax == 1 else bc.z)
            var b = Int((coord - c_min) * inv_span)
            if b < 0:
                b = 0
            elif b >= _SAH_BINS:
                b = _SAH_BINS - 1
            bin_cnt[b] += 1
            var blo = t_lo[ti]
            var bhi = t_hi[ti]
            b_lox[b] = min(b_lox[b], blo.x); b_loy[b] = min(b_loy[b], blo.y); b_loz[b] = min(b_loz[b], blo.z)
            b_hix[b] = max(b_hix[b], bhi.x); b_hiy[b] = max(b_hiy[b], bhi.y); b_hiz[b] = max(b_hiz[b], bhi.z)

        var right_area = Array[Float32, _SAH_BINS](fill=0.0)
        var right_cnt = Array[Int, _SAH_BINS](fill=0)
        var r_lo = Vec3(1.0e30, 1.0e30, 1.0e30)
        var r_hi = Vec3(-1.0e30, -1.0e30, -1.0e30)
        var r_c = 0
        var br = _SAH_BINS - 1
        while br >= 1:
            if bin_cnt[br] > 0:
                r_c += bin_cnt[br]
                r_lo.x = min(r_lo.x, b_lox[br]); r_lo.y = min(r_lo.y, b_loy[br]); r_lo.z = min(r_lo.z, b_loz[br])
                r_hi.x = max(r_hi.x, b_hix[br]); r_hi.y = max(r_hi.y, b_hiy[br]); r_hi.z = max(r_hi.z, b_hiz[br])
            right_cnt[br] = r_c
            right_area[br] = _aabb_half_area(r_lo, r_hi) if r_c > 0 else Float32(0.0)
            br -= 1

        var l_lo = Vec3(1.0e30, 1.0e30, 1.0e30)
        var l_hi = Vec3(-1.0e30, -1.0e30, -1.0e30)
        var l_c = 0
        for b in range(_SAH_BINS - 1):
            if bin_cnt[b] > 0:
                l_c += bin_cnt[b]
                l_lo.x = min(l_lo.x, b_lox[b]); l_lo.y = min(l_lo.y, b_loy[b]); l_lo.z = min(l_lo.z, b_loz[b])
                l_hi.x = max(l_hi.x, b_hix[b]); l_hi.y = max(l_hi.y, b_hiy[b]); l_hi.z = max(l_hi.z, b_hiz[b])
            var rc = right_cnt[b + 1]
            if l_c > 0 and rc > 0:
                var cost = _aabb_half_area(l_lo, l_hi) * Float32(l_c) + right_area[b + 1] * Float32(rc)
                if cost < best_cost:
                    best_cost = cost
                    best_axis = ax
                    best_split = b

    var mid = (start + end) // 2
    if best_axis >= 0 and best_split >= 0:
        var inv_span = Float32(_SAH_BINS) / spans[best_axis]
        var c_min = c_mins[best_axis]
        var il = start
        var ir = end - 1
        while il <= ir:
            var ti = perm[il]
            var bc = t_center[ti]
            var coord = bc.x if best_axis == 0 else (bc.y if best_axis == 1 else bc.z)
            var b = Int((coord - c_min) * inv_span)
            if b < 0:
                b = 0
            elif b >= _SAH_BINS:
                b = _SAH_BINS - 1
            if b <= best_split:
                il += 1
            else:
                perm[il] = perm[ir]
                perm[ir] = ti
                ir -= 1
        if il > start and il < end:
            mid = il

    var left_idx = _build_mesh_bvh2(tris, t_lo, t_hi, t_center, perm, start, mid, out_nodes)
    var right_idx = _build_mesh_bvh2(tris, t_lo, t_hi, t_center, perm, mid, end, out_nodes)
    out_nodes[node_idx].left = Int32(left_idx)
    out_nodes[node_idx].right = Int32(right_idx)
    return node_idx


def _collapse_mesh_bvh8(bin_nodes: List[_MeshBinNode], bin_idx: Int, mut out8: List[Bvh8Node]) -> Int:
    var out_idx = len(out8)
    out8.append(Bvh8Node())

    var frontier = Array[Int, BVH8_CHILDREN](fill=-1)
    var f_count: Int
    var root_bin = bin_nodes[bin_idx]
    if root_bin.right < 0:
        frontier[0] = bin_idx
        f_count = 1
    else:
        frontier[0] = Int(root_bin.left)
        frontier[1] = Int(root_bin.right)
        f_count = 2
        while f_count < BVH8_CHILDREN:
            var best_k = -1
            var best_area = Float32(-1.0)
            for k in range(f_count):
                var bn = bin_nodes[frontier[k]]
                if bn.right >= 0:
                    var area = _aabb_half_area(bn.lo, bn.hi)
                    if area > best_area:
                        best_area = area
                        best_k = k
            if best_k < 0:
                break
            var expand = bin_nodes[frontier[best_k]]
            frontier[best_k] = Int(expand.left)
            frontier[f_count] = Int(expand.right)
            f_count += 1

    var node8 = Bvh8Node()
    for c in range(f_count):
        var b_i = frontier[c]
        var bn = bin_nodes[b_i]
        node8.min_x[c] = bn.lo.x
        node8.min_y[c] = bn.lo.y
        node8.min_z[c] = bn.lo.z
        node8.max_x[c] = bn.hi.x
        node8.max_y[c] = bn.hi.y
        node8.max_z[c] = bn.hi.z
        if bn.right < 0:
            node8.child_indices[c] = bn.left        # -(first_tri + 1)
            node8.child_counts[c] = -bn.right       # +tri_count
        else:
            var child8 = _collapse_mesh_bvh8(bin_nodes, b_i, out8)
            node8.child_indices[c] = Int32(child8)  # > 0 internal node index
            node8.child_counts[c] = Int32(0)        # 0 marks internal node
    out8[out_idx] = node8
    return out_idx


@always_inline
def _safe_inv_dir(d: Float32) -> Float32:
    if abs(d) > Float32(1.0e-8):
        return Float32(1.0) / d
    return Float32(1.0e30) if d >= Float32(0.0) else Float32(-1.0e30)


struct Bvh8MeshCollider(Copyable, Movable):
    """8-wide static triangle-mesh SAH BVH collider with O(N) signed dihedral active-edge smoothing."""
    var triangles: List[MeshTriangle]
    var nodes: List[Bvh8Node]
    var root_node: Int

    def __init__(out self):
        self.triangles = List[MeshTriangle]()
        self.nodes = List[Bvh8Node]()
        self.root_node = -1

    @staticmethod
    def build(vertices: List[Vec3], indices: List[Int32], material_ids: List[UInt16] = List[UInt16]()) -> Self:
        var col = Self()
        var tri_count = len(indices) // 3
        if tri_count <= 0:
            return col^

        col.triangles.reserve(tri_count)
        for i in range(tri_count):
            var i0 = Int(indices[i * 3])
            var i1 = Int(indices[i * 3 + 1])
            var i2 = Int(indices[i * 3 + 2])
            var v0 = vertices[i0]
            var v1 = vertices[i1]
            var v2 = vertices[i2]
            var mat_id: UInt16 = material_ids[i] if i < len(material_ids) else 0
            col.triangles.append(MeshTriangle(v0, v1, v2, mat_id))

        col.smooth_active_edges()
        col.build_tree()
        return col^

    def smooth_active_edges(mut self):
        """O(N) quantized undirected edge hash table (`detsort.mix64`) marking shared
        edges inactive when concave/coplanar `((v_opp - v0) . n1 >= -1e-4)` or smooth convex `(n1 . n2 > 0.996)`."""
        var n = len(self.triangles)
        if n <= 1:
            return

        var cap = 32
        var needed = n * 8
        while cap < needed:
            cap *= 2
        var mask = cap - 1

        var keys = List[UInt64](length=cap, fill=0)
        var tri_ids = List[Int32](length=cap, fill=-1)
        var edge_ids = List[UInt8](length=cap, fill=0)

        for i in range(n):
            for e1 in range(3):
                var ep1 = _edge_endpoints(self.triangles[i], e1)
                var a1 = ep1[0]
                var b1 = ep1[1]
                var opp1 = ep1[2]
                var key = _undirected_edge_hash(a1, b1)
                var slot = Int(key) & mask
                var matched = False

                for _ in range(cap):
                    var k_at = keys[slot]
                    if k_at == 0:
                        keys[slot] = key
                        tri_ids[slot] = Int32(i)
                        edge_ids[slot] = UInt8(e1)
                        break
                    if k_at == key:
                        var j = Int(tri_ids[slot])
                        var e2 = Int(edge_ids[slot])
                        if j != i:
                            var ep2 = _edge_endpoints(self.triangles[j], e2)
                            var a2 = ep2[0]
                            var b2 = ep2[1]
                            var opp2 = ep2[2]
                            if (vec3_eq(a1, a2) and vec3_eq(b1, b2)) or (vec3_eq(a1, b2) and vec3_eq(b1, a2)):
                                var n1 = self.triangles[i].normal
                                var n2 = self.triangles[j].normal
                                var cos_phi = n1.dot(n2)
                                var d_opp2 = (opp2 - a1).dot(n1)
                                var d_opp1 = (opp1 - a2).dot(n2)
                                # Inactive when concave/coplanar OR smooth convex (cos_phi > 0.996)
                                if (cos_phi > Float32(-0.5) and d_opp2 >= Float32(-1.0e-4) and d_opp1 >= Float32(-1.0e-4)) or cos_phi > Float32(0.996):
                                    self.triangles[i].active_edges &= ~_edge_bit(e1)
                                    self.triangles[j].active_edges &= ~_edge_bit(e2)
                                matched = True
                    slot = (slot + 1) & mask
                    if matched and keys[slot] == 0:
                        keys[slot] = key
                        tri_ids[slot] = Int32(i)
                        edge_ids[slot] = UInt8(e1)
                        break

    def build_tree(mut self):
        self.nodes.clear()
        self.root_node = -1
        var count = len(self.triangles)
        if count == 0:
            return

        # Small meshes (<= 8 triangles): pack 1 triangle per lane in original index order
        if count <= BVH8_CHILDREN:
            var root = Bvh8Node()
            for c in range(count):
                var b = self.triangles[c].bounds()
                root.min_x[c] = b[0].x; root.min_y[c] = b[0].y; root.min_z[c] = b[0].z
                root.max_x[c] = b[1].x; root.max_y[c] = b[1].y; root.max_z[c] = b[1].z
                root.child_indices[c] = Int32(-(c + 1))
                root.child_counts[c] = Int32(1)
            self.nodes.append(root)
            self.root_node = 0
            return

        # Multi-level recursive 16-bin SAH BVH collapsed into 8-wide SIMD Bvh8Nodes
        var t_lo = List[Vec3](capacity=count)
        var t_hi = List[Vec3](capacity=count)
        var t_center = List[Vec3](capacity=count)
        var perm = List[Int](capacity=count)
        for i in range(count):
            var b = self.triangles[i].bounds()
            t_lo.append(b[0])
            t_hi.append(b[1])
            t_center.append(self.triangles[i].center())
            perm.append(i)

        var bin_nodes = List[_MeshBinNode](capacity=2 * count)
        var bin_root = _build_mesh_bvh2(self.triangles, t_lo, t_hi, t_center, perm, 0, count, bin_nodes)

        var reordered = List[MeshTriangle](capacity=count)
        for i in range(count):
            reordered.append(self.triangles[perm[i]])
        self.triangles = reordered^

        self.nodes.reserve((count + BVH8_CHILDREN - 1) // BVH8_CHILDREN + 4)
        self.root_node = _collapse_mesh_bvh8(bin_nodes, bin_root, self.nodes)

    def raycast(self, origin: Vec3, dir: Vec3, max_t: Float32 = 1.0e6) -> RayHit:
        var hit = RayHit()
        hit.t = max_t
        if len(self.nodes) == 0 or self.root_node < 0:
            return hit^

        var ox = F8(origin.x)
        var oy = F8(origin.y)
        var oz = F8(origin.z)
        var idx = F8(_safe_inv_dir(dir.x))
        var idy = F8(_safe_inv_dir(dir.y))
        var idz = F8(_safe_inv_dir(dir.z))

        var stack_node = Array[Int32, 96](fill=0)
        var stack_t = Array[Float32, 96](fill=0.0)
        var sp = 1
        stack_node[0] = Int32(self.root_node)
        stack_t[0] = Float32(0.0)

        var child_buf_idx = Array[Int32, BVH8_CHILDREN](fill=0)
        var child_buf_t = Array[Float32, BVH8_CHILDREN](fill=0.0)

        while sp > 0:
            sp -= 1
            if stack_t[sp] >= hit.t:
                continue
            ref node = self.nodes[Int(stack_node[sp])]

            var tx1 = (node.min_x - ox) * idx
            var tx2 = (node.max_x - ox) * idx
            var tmin = min(tx1, tx2)
            var tmax = max(tx1, tx2)

            var ty1 = (node.min_y - oy) * idy
            var ty2 = (node.max_y - oy) * idy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

            var tz1 = (node.min_z - oz) * idz
            var tz2 = (node.max_z - oz) * idz
            tmin = max(tmin, min(tz1, tz2))
            tmax = min(tmax, max(tz1, tz2))

            var t_enter = max(F8(0.0), tmin)
            var mask = tmax.ge(t_enter) & t_enter.lt(F8(hit.t))
            if not mask.reduce_or():
                continue

            var internal_count = 0
            comptime for c in range(BVH8_CHILDREN):
                if mask[c]:
                    var c_idx = Int(node.child_indices[c])
                    var c_cnt = Int(node.child_counts[c])
                    if c_cnt > 0:
                        var leaf_first = -c_idx - 1
                        for ti in range(leaf_first, leaf_first + c_cnt):
                            var res = self.triangles[ti].ray_intersect(origin, dir, hit.t)
                            if res[0] and res[1] < hit.t:
                                hit.hit = True
                                hit.t = res[1]
                                hit.point = origin + dir * res[1]
                                hit.normal = res[2]
                                hit.triangle_index = ti
                                hit.material_id = self.triangles[ti].material_id
                    elif c_idx > 0:
                        var te = t_enter[c]
                        if te < hit.t:
                            child_buf_idx[internal_count] = Int32(c_idx)
                            child_buf_t[internal_count] = te
                            internal_count += 1

            # Sort internal children descending by t_enter so nearest is popped first
            for i in range(1, internal_count):
                var ki = child_buf_idx[i]
                var kt = child_buf_t[i]
                var j = i - 1
                while j >= 0 and child_buf_t[j] < kt:
                    child_buf_idx[j + 1] = child_buf_idx[j]
                    child_buf_t[j + 1] = child_buf_t[j]
                    j -= 1
                child_buf_idx[j + 1] = ki
                child_buf_t[j + 1] = kt

            for i in range(internal_count):
                if child_buf_t[i] < hit.t and sp < 96:
                    stack_node[sp] = child_buf_idx[i]
                    stack_t[sp] = child_buf_t[i]
                    sp += 1

        return hit^

    @staticmethod
    def _sweep_sphere_segment(
        start: Vec3, ndir: Vec3, max_dist: Float32, radius: Float32, a: Vec3, b: Vec3
    ) -> Tuple[Bool, Float32, Vec3, Vec3]:
        """Earliest intersection of swept sphere (radius) with segment a-b (capsule)."""
        var r2 = radius * radius
        var ab = b - a
        var ab_len2 = ab.length_squared()
        var best_t = max_dist + Float32(1.0)
        var best_pt = Vec3()
        var best_n = Vec3(0, 1, 0)
        var found = False

        var ends = [a, b]
        for ei in range(2):
            var oc = start - ends[ei]
            var c = oc.length_squared() - r2
            if c <= Float32(1.0e-5):
                return (True, Float32(0.0), ends[ei], (oc.normalized() if oc.length_squared() > 1.0e-8 else Vec3(0, 1, 0)))
            var b_coef = oc.dot(ndir)
            var disc = b_coef * b_coef - c
            if disc >= Float32(0.0):
                var t_s = -b_coef - sqrt(disc)
                if t_s >= Float32(0.0) and t_s <= max_dist and t_s < best_t:
                    best_t = t_s
                    best_pt = ends[ei]
                    var hit_c = start + ndir * t_s
                    var nrm = hit_c - ends[ei]
                    best_n = nrm.normalized() if nrm.length_squared() > 1.0e-8 else Vec3(0, 1, 0)
                    found = True

        if ab_len2 > Float32(1.0e-8):
            var inv_ab = Float32(1.0) / ab_len2
            var ao = start - a
            var d_dot_ab = ndir.dot(ab)
            var ao_dot_ab = ao.dot(ab)
            var d_perp = ndir - ab * (d_dot_ab * inv_ab)
            var ao_perp = ao - ab * (ao_dot_ab * inv_ab)
            var A = d_perp.length_squared()
            var B = ao_perp.dot(d_perp)
            var C = ao_perp.length_squared() - r2
            if A > Float32(1.0e-8):
                var disc = B * B - A * C
                if disc >= Float32(0.0):
                    var t_c = (-B - sqrt(disc)) / A
                    if t_c >= Float32(0.0) and t_c <= max_dist and t_c < best_t:
                        var s_proj = (ao_dot_ab + t_c * d_dot_ab) * inv_ab
                        if s_proj >= Float32(0.0) and s_proj <= Float32(1.0):
                            var seg_pt = a + ab * s_proj
                            var hit_c = start + ndir * t_c
                            var nrm = hit_c - seg_pt
                            best_t = t_c
                            best_pt = seg_pt
                            best_n = nrm.normalized() if nrm.length_squared() > 1.0e-8 else Vec3(0, 1, 0)
                            found = True

        return (found, best_t, best_pt, best_n)

    def _sweep_sphere_tri(
        self,
        tri_idx: Int,
        start: Vec3,
        ndir: Vec3,
        dir_len: Float32,
        max_dist: Float32,
        radius: Float32,
        r2: Float32,
        mut hit: SweepHit,
        mut hit_smoothed: Bool,
    ):
        ref tri = self.triangles[tri_idx]
        # 1. Overlapping at t = 0
        var cp0 = tri.closest_point(start)
        var diff0 = start - cp0[0]
        if diff0.length_squared() <= r2 + Float32(1.0e-5):
            var sm0 = vec3_eq(cp0[1], tri.normal)
            if not hit.hit or hit.toi > Float32(1.0e-5) or (not sm0 and hit_smoothed):
                hit.hit = True
                hit.toi = Float32(0.0)
                hit.point = cp0[0]
                hit.normal = cp0[1]
                hit.triangle_index = tri_idx
                hit_smoothed = sm0
            return

        # 2. Triangle plane slab hit
        var d_start = (start - tri.v0).dot(tri.normal)
        var d_vel = ndir.dot(tri.normal)
        if abs(d_vel) >= Float32(1.0e-6):
            var t_plane = (radius - d_start) / d_vel if d_start >= Float32(0.0) else (-radius - d_start) / d_vel
            if t_plane >= Float32(0.0) and t_plane <= max_dist:
                var p_at = start + ndir * t_plane
                var cp = tri.closest_point(p_at)
                var dist_sq = (p_at - cp[0]).length_squared()
                if dist_sq <= r2 + Float32(1.0e-4):
                    var toi = t_plane / dir_len
                    if not hit.hit or toi < hit.toi - Float32(1.0e-4):
                        hit.hit = True
                        hit.toi = toi
                        hit.point = cp[0]
                        hit.normal = cp[1]
                        hit.triangle_index = tri_idx
                        hit_smoothed = vec3_eq(cp[1], tri.normal)

        # 3. Triangle edges & vertices
        var e0 = Self._sweep_sphere_segment(start, ndir, max_dist, radius, tri.v0, tri.v1)
        if e0[0]:
            var toi0 = e0[1] / dir_len
            var active0 = (tri.active_edges & EDGE_ACTIVE_01) != 0
            if not hit.hit or toi0 < hit.toi - Float32(1.0e-4):
                hit.hit = True
                hit.toi = toi0
                hit.point = e0[2]
                hit.normal = e0[3] if active0 else tri.normal
                hit.triangle_index = tri_idx
                hit_smoothed = not active0
            elif abs(toi0 - hit.toi) <= Float32(1.0e-4):
                if active0 and (hit_smoothed or vec3_eq(hit.normal, tri.normal)):
                    hit.toi = min(hit.toi, toi0)
                    hit.point = e0[2]
                    hit.normal = e0[3]
                    hit.triangle_index = tri_idx
                    hit_smoothed = False
                elif active0 and not hit_smoothed and toi0 < hit.toi:
                    hit.toi = toi0
                    hit.point = e0[2]
                    hit.normal = e0[3]
                    hit.triangle_index = tri_idx

        var e1 = Self._sweep_sphere_segment(start, ndir, max_dist, radius, tri.v1, tri.v2)
        if e1[0]:
            var toi1 = e1[1] / dir_len
            var active1 = (tri.active_edges & EDGE_ACTIVE_12) != 0
            if not hit.hit or toi1 < hit.toi - Float32(1.0e-4):
                hit.hit = True
                hit.toi = toi1
                hit.point = e1[2]
                hit.normal = e1[3] if active1 else tri.normal
                hit.triangle_index = tri_idx
                hit_smoothed = not active1
            elif abs(toi1 - hit.toi) <= Float32(1.0e-4):
                if active1 and (hit_smoothed or vec3_eq(hit.normal, tri.normal)):
                    hit.toi = min(hit.toi, toi1)
                    hit.point = e1[2]
                    hit.normal = e1[3]
                    hit.triangle_index = tri_idx
                    hit_smoothed = False
                elif active1 and not hit_smoothed and toi1 < hit.toi:
                    hit.toi = toi1
                    hit.point = e1[2]
                    hit.normal = e1[3]
                    hit.triangle_index = tri_idx

        var e2 = Self._sweep_sphere_segment(start, ndir, max_dist, radius, tri.v2, tri.v0)
        if e2[0]:
            var toi2 = e2[1] / dir_len
            var active2 = (tri.active_edges & EDGE_ACTIVE_20) != 0
            if not hit.hit or toi2 < hit.toi - Float32(1.0e-4):
                hit.hit = True
                hit.toi = toi2
                hit.point = e2[2]
                hit.normal = e2[3] if active2 else tri.normal
                hit.triangle_index = tri_idx
                hit_smoothed = not active2
            elif abs(toi2 - hit.toi) <= Float32(1.0e-4):
                if active2 and (hit_smoothed or vec3_eq(hit.normal, tri.normal)):
                    hit.toi = min(hit.toi, toi2)
                    hit.point = e2[2]
                    hit.normal = e2[3]
                    hit.triangle_index = tri_idx
                    hit_smoothed = False
                elif active2 and not hit_smoothed and toi2 < hit.toi:
                    hit.toi = toi2
                    hit.point = e2[2]
                    hit.normal = e2[3]
                    hit.triangle_index = tri_idx

    def sweep_sphere(self, start: Vec3, dir: Vec3, radius: Float32, max_t: Float32) -> SweepHit:
        """Traverses `self.nodes` using 8-wide `F8` radius-inflated AABB slab tests ordered by near `t`."""
        var hit = SweepHit()
        hit.toi = max_t
        if len(self.nodes) == 0 or self.root_node < 0:
            return hit^

        var dir_len = dir.length()
        if dir_len < Float32(1.0e-6):
            return hit^

        var ndir = dir * (Float32(1.0) / dir_len)
        var max_dist = max_t * dir_len
        var r2 = radius * radius
        var r_f8 = F8(radius)
        var ox = F8(start.x)
        var oy = F8(start.y)
        var oz = F8(start.z)
        var idx = F8(_safe_inv_dir(ndir.x))
        var idy = F8(_safe_inv_dir(ndir.y))
        var idz = F8(_safe_inv_dir(ndir.z))
        var tie_slack = Float32(1.0e-3) * max(Float32(1.0), dir_len)

        var stack_node = Array[Int32, 96](fill=0)
        var stack_dist = Array[Float32, 96](fill=0.0)
        var sp = 1
        stack_node[0] = Int32(self.root_node)
        stack_dist[0] = Float32(0.0)

        var child_buf_idx = Array[Int32, BVH8_CHILDREN](fill=0)
        var child_buf_d = Array[Float32, BVH8_CHILDREN](fill=0.0)
        var hit_smoothed = False

        while sp > 0:
            sp -= 1
            var limit_dist = min(max_dist, hit.toi * dir_len) + tie_slack
            if stack_dist[sp] > limit_dist:
                continue

            ref node = self.nodes[Int(stack_node[sp])]
            var tx1 = ((node.min_x - r_f8) - ox) * idx
            var tx2 = ((node.max_x + r_f8) - ox) * idx
            var tmin = min(tx1, tx2)
            var tmax = max(tx1, tx2)

            var ty1 = ((node.min_y - r_f8) - oy) * idy
            var ty2 = ((node.max_y + r_f8) - oy) * idy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

            var tz1 = ((node.min_z - r_f8) - oz) * idz
            var tz2 = ((node.max_z + r_f8) - oz) * idz
            tmin = max(tmin, min(tz1, tz2))
            tmax = min(tmax, max(tz1, tz2))

            var t_enter = max(F8(0.0), tmin)
            var mask = tmax.ge(t_enter) & t_enter.le(F8(limit_dist))
            if not mask.reduce_or():
                continue

            var internal_count = 0
            comptime for c in range(BVH8_CHILDREN):
                if mask[c]:
                    var c_idx = Int(node.child_indices[c])
                    var c_cnt = Int(node.child_counts[c])
                    if c_cnt > 0:
                        var leaf_first = -c_idx - 1
                        var cur_max_d = min(max_dist, hit.toi * dir_len + tie_slack)
                        for ti in range(leaf_first, leaf_first + c_cnt):
                            self._sweep_sphere_tri(ti, start, ndir, dir_len, cur_max_d, radius, r2, hit, hit_smoothed)
                    elif c_idx > 0:
                        var te = t_enter[c]
                        if te <= min(max_dist, hit.toi * dir_len) + tie_slack:
                            child_buf_idx[internal_count] = Int32(c_idx)
                            child_buf_d[internal_count] = te
                            internal_count += 1

            for i in range(1, internal_count):
                var ki = child_buf_idx[i]
                var kd = child_buf_d[i]
                var j = i - 1
                while j >= 0 and child_buf_d[j] < kd:
                    child_buf_idx[j + 1] = child_buf_idx[j]
                    child_buf_d[j + 1] = child_buf_d[j]
                    j -= 1
                child_buf_idx[j + 1] = ki
                child_buf_d[j + 1] = kd

            var push_limit = min(max_dist, hit.toi * dir_len) + tie_slack
            for i in range(internal_count):
                if child_buf_d[i] <= push_limit and sp < 96:
                    stack_node[sp] = child_buf_idx[i]
                    stack_dist[sp] = child_buf_d[i]
                    sp += 1

        return hit^

    def swept_sphere_moving_bvh8(
        self,
        p0: Vec3,
        p1: Vec3,
        radius: Float32,
        mesh_pos0: Vec3,
        mesh_rot0: Quat,
        mesh_pos1: Vec3,
        mesh_rot1: Quat,
    ) -> SweepHit:
        """Sweeps a sphere (`p0 -> p1`) against a moving & rotating `Bvh8MeshCollider`
        (`(mesh_pos0, mesh_rot0) -> (mesh_pos1, mesh_rot1)`), returning a world-space `SweepHit`."""
        var lp0 = mesh_rot0.conjugate().rotate(p0 - mesh_pos0)
        var lp1 = mesh_rot1.conjugate().rotate(p1 - mesh_pos1)
        var ldir = lp1 - lp0
        if ldir.length_squared() < Float32(1.0e-10):
            var m0 = self.collide_sphere(lp0, radius)
            if m0.has_collision:
                var wn0 = mesh_rot0.rotate(m0.p0.normal).normalized()
                var wp0 = mesh_pos0 + mesh_rot0.rotate(m0.p0.point)
                return SweepHit(True, Float32(0.0), wp0, wn0, Int(m0.p0.id))
            return SweepHit()

        var sh = self.sweep_sphere(lp0, ldir, radius, Float32(1.0))
        if not sh.hit:
            return sh^

        var t = max(Float32(0.0), min(Float32(1.0), sh.toi))
        var m_pos = mesh_pos0 + (mesh_pos1 - mesh_pos0) * t
        var m_rot = mesh_rot0.slerp(mesh_rot1, t)
        sh.point = m_pos + m_rot.rotate(sh.point)
        sh.normal = m_rot.rotate(sh.normal).normalized()
        return sh^

    def _query_aabb_triangles(self, lo: Vec3, hi: Vec3, mut out_tris: List[Int]):
        """Collects triangle indices whose AABBs overlap `[lo, hi]` using 8-wide SIMD traversal."""
        if len(self.nodes) == 0 or self.root_node < 0:
            return
        var qlo_x = F8(lo.x); var qlo_y = F8(lo.y); var qlo_z = F8(lo.z)
        var qhi_x = F8(hi.x); var qhi_y = F8(hi.y); var qhi_z = F8(hi.z)
        var stack = Array[Int32, 96](fill=0)
        var sp = 1
        stack[0] = Int32(self.root_node)

        while sp > 0:
            sp -= 1
            ref node = self.nodes[Int(stack[sp])]
            var ov = (
                qlo_x.le(node.max_x)
                & qhi_x.ge(node.min_x)
                & qlo_y.le(node.max_y)
                & qhi_y.ge(node.min_y)
                & qlo_z.le(node.max_z)
                & qhi_z.ge(node.min_z)
            )
            if not ov.reduce_or():
                continue
            comptime for c in range(BVH8_CHILDREN):
                if ov[c]:
                    var c_idx = Int(node.child_indices[c])
                    var c_cnt = Int(node.child_counts[c])
                    if c_cnt > 0:
                        var first = -c_idx - 1
                        for ti in range(first, first + c_cnt):
                            out_tris.append(ti)
                    elif c_idx > 0 and sp < 96:
                        stack[sp] = Int32(c_idx)
                        sp += 1

    def collide_sphere(self, center: Vec3, radius: Float32) -> ContactManifold:
        """Discrete sphere vs BVH8 mesh collision manifold (normal points outward from mesh toward sphere)."""
        var m = ContactManifold(0, -1)
        var r_vec = Vec3(radius, radius, radius)
        var cand_tris = List[Int]()
        self._query_aabb_triangles(center - r_vec, center + r_vec, cand_tris)
        if len(cand_tris) == 0:
            return m^

        var cands = List[ContactPoint]()
        var r2 = radius * radius
        for k in range(len(cand_tris)):
            var ti = cand_tris[k]
            ref tri = self.triangles[ti]
            var cp = tri.closest_point(center)
            var diff = center - cp[0]
            var d2 = diff.length_squared()
            if d2 <= r2:
                var dist = sqrt(d2)
                var n = cp[1]
                if dist > Float32(1.0e-5) and diff.dot(n) < Float32(0.0):
                    n = -n
                var depth = radius - dist
                cands.append(ContactPoint(cp[0], n, depth, UInt32(ti)))

        reduce_manifold_candidates(m, cands)
        return m^

    @staticmethod
    def _segment_triangle_closest(p0: Vec3, p1: Vec3, tri: MeshTriangle) -> Tuple[Vec3, Vec3, Vec3]:
        """Returns `(seg_pt, tri_pt, smoothed_normal)` minimizing distance between segment `[p0, p1]` and `tri`."""
        var best_d2 = Float32(1.0e38)
        var best_s = p0
        var best_t = p0
        var best_n = tri.normal

        # 1. Segment intersection with triangle plane
        var seg = p1 - p0
        var seg_len = seg.length()
        if seg_len > Float32(1.0e-6):
            var rh = tri.ray_intersect(p0, seg * (Float32(1.0) / seg_len), seg_len)
            if rh[0]:
                var hit_p = p0 + seg * (rh[1] / seg_len)
                return (hit_p, hit_p, tri.normal)

        # 2. Segment endpoints against triangle
        var ends = [p0, p1]
        for ei in range(2):
            var cp = tri.closest_point(ends[ei])
            var d2 = (ends[ei] - cp[0]).length_squared()
            if d2 < best_d2:
                best_d2 = d2
                best_s = ends[ei]
                best_t = cp[0]
                best_n = cp[1]

        # 3. Triangle vertices projected onto segment
        var seg_l2 = seg.length_squared()
        if seg_l2 > Float32(1.0e-8):
            var tverts = [tri.v0, tri.v1, tri.v2]
            for vi in range(3):
                var t_proj = max(Float32(0.0), min(Float32(1.0), (tverts[vi] - p0).dot(seg) / seg_l2))
                var sp = p0 + seg * t_proj
                var cp = tri.closest_point(sp)
                var d2 = (sp - cp[0]).length_squared()
                if d2 < best_d2:
                    best_d2 = d2
                    best_s = sp
                    best_t = cp[0]
                    best_n = cp[1]

        # 4. Segment against 3 triangle edges
        var edges = [(tri.v0, tri.v1, EDGE_ACTIVE_01), (tri.v1, tri.v2, EDGE_ACTIVE_12), (tri.v2, tri.v0, EDGE_ACTIVE_20)]
        for ei in range(3):
            var cps = closest_points_segments(p0, p1, edges[ei][0], edges[ei][1])
            var d2 = (cps[0] - cps[1]).length_squared()
            if d2 < best_d2:
                best_d2 = d2
                best_s = cps[0]
                best_t = cps[1]
                var active = (tri.active_edges & edges[ei][2]) != 0
                if active and d2 > Float32(1.0e-8):
                    best_n = (cps[0] - cps[1]) * (Float32(1.0) / sqrt(d2))
                else:
                    best_n = tri.normal

        return (best_s, best_t, best_n)

    def collide_capsule(self, p0: Vec3, p1: Vec3, radius: Float32) -> ContactManifold:
        """Discrete capsule vs BVH8 mesh collision manifold (normal points outward from mesh toward capsule)."""
        var m = ContactManifold(0, -1)
        var lo = Vec3(min(p0.x, p1.x) - radius, min(p0.y, p1.y) - radius, min(p0.z, p1.z) - radius)
        var hi = Vec3(max(p0.x, p1.x) + radius, max(p0.y, p1.y) + radius, max(p0.z, p1.z) + radius)
        var cand_tris = List[Int]()
        self._query_aabb_triangles(lo, hi, cand_tris)
        if len(cand_tris) == 0:
            return m^

        var cands = List[ContactPoint]()
        var r2 = radius * radius
        for k in range(len(cand_tris)):
            var ti = cand_tris[k]
            ref tri = self.triangles[ti]
            # Check both endpoints for parallel lying-flat support + minimum segment-triangle point
            var cp_a = tri.closest_point(p0)
            var da2 = (p0 - cp_a[0]).length_squared()
            if da2 <= r2:
                var dist = sqrt(da2)
                var n = cp_a[1]
                if dist > Float32(1.0e-5) and (p0 - cp_a[0]).dot(n) < Float32(0.0):
                    n = -n
                cands.append(ContactPoint(cp_a[0], n, radius - dist, UInt32(ti * 4)))

            var cp_b = tri.closest_point(p1)
            var db2 = (p1 - cp_b[0]).length_squared()
            if db2 <= r2 and (p1 - p0).length_squared() > Float32(1.0e-6):
                var dist = sqrt(db2)
                var n = cp_b[1]
                if dist > Float32(1.0e-5) and (p1 - cp_b[0]).dot(n) < Float32(0.0):
                    n = -n
                cands.append(ContactPoint(cp_b[0], n, radius - dist, UInt32(ti * 4 + 1)))

            var st_cp = Self._segment_triangle_closest(p0, p1, tri)
            var ds2 = (st_cp[0] - st_cp[1]).length_squared()
            if ds2 <= r2:
                var dist = sqrt(ds2)
                var n = st_cp[2]
                if dist > Float32(1.0e-5) and (st_cp[0] - st_cp[1]).dot(n) < Float32(0.0):
                    n = -n
                cands.append(ContactPoint(st_cp[1], n, radius - dist, UInt32(ti * 4 + 2)))

        reduce_manifold_candidates(m, cands)
        return m^

    @staticmethod
    def _collide_local_box_tri(
        half: Vec3,
        lv0: Vec3,
        lv1: Vec3,
        lv2: Vec3,
        ln_tri: Vec3,
        active_edges: UInt8,
        box_center: Vec3,
        box_rot: Quat,
        tri_id: Int,
        mut cands: List[ContactPoint],
    ):
        """13-axis SAT + active-edge suppression between local AABB `[-half, +half]` and local triangle."""
        # 1. Test 3 box face axes
        var min_x = min(lv0.x, min(lv1.x, lv2.x)); var max_x = max(lv0.x, max(lv1.x, lv2.x))
        if min_x > half.x or max_x < -half.x:
            return
        var min_y = min(lv0.y, min(lv1.y, lv2.y)); var max_y = max(lv0.y, max(lv1.y, lv2.y))
        if min_y > half.y or max_y < -half.y:
            return
        var min_z = min(lv0.z, min(lv1.z, lv2.z)); var max_z = max(lv0.z, max(lv1.z, lv2.z))
        if min_z > half.z or max_z < -half.z:
            return

        # 2. Test triangle plane normal
        var r_plane = half.x * abs(ln_tri.x) + half.y * abs(ln_tri.y) + half.z * abs(ln_tri.z)
        var d_plane = ln_tri.dot(lv0)
        if abs(d_plane) > r_plane:
            return

        # Oriented outward normal from triangle toward box center (0, 0, 0): (-lv0) . n >= 0 => -d_plane >= 0
        var n_out = ln_tri if (-d_plane) >= Float32(0.0) else -ln_tri
        var best_pen = r_plane - abs(d_plane)
        var best_n = n_out

        # Check 3 box face axes overlap
        var ox = min(half.x - min_x, max_x + half.x)
        if ox < best_pen:
            var sx = Float32(-1.0) if (lv0.x + lv1.x + lv2.x) > Float32(0.0) else Float32(1.0)
            if (active_edges == EDGE_ACTIVE_ALL) or abs(n_out.x) > Float32(0.95):
                best_pen = ox
                best_n = Vec3(sx, 0.0, 0.0)
        var oy = min(half.y - min_y, max_y + half.y)
        if oy < best_pen:
            var sy = Float32(-1.0) if (lv0.y + lv1.y + lv2.y) > Float32(0.0) else Float32(1.0)
            if (active_edges == EDGE_ACTIVE_ALL) or abs(n_out.y) > Float32(0.95):
                best_pen = oy
                best_n = Vec3(0.0, sy, 0.0)
        var oz = min(half.z - min_z, max_z + half.z)
        if oz < best_pen:
            var sz = Float32(-1.0) if (lv0.z + lv1.z + lv2.z) > Float32(0.0) else Float32(1.0)
            if (active_edges == EDGE_ACTIVE_ALL) or abs(n_out.z) > Float32(0.95):
                best_pen = oz
                best_n = Vec3(0.0, 0.0, sz)

        # 3. Test 9 cross-product edge axes (box_axis x tri_edge)
        var edges = [lv1 - lv0, lv2 - lv1, lv0 - lv2]
        var ebits = [EDGE_ACTIVE_01, EDGE_ACTIVE_12, EDGE_ACTIVE_20]
        var b_axes = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]

        for ei in range(3):
            var e_active = (active_edges & ebits[ei]) != 0
            for bi in range(3):
                var raw_ax = b_axes[bi].cross(edges[ei])
                var al2 = raw_ax.length_squared()
                if al2 <= Float32(1.0e-8):
                    continue
                var ax = raw_ax * (Float32(1.0) / sqrt(al2))
                var p0 = lv0.dot(ax)
                var p1 = lv1.dot(ax)
                var p2 = lv2.dot(ax)
                var t_min = min(p0, min(p1, p2))
                var t_max = max(p0, max(p1, p2))
                var r_b = half.x * abs(ax.x) + half.y * abs(ax.y) + half.z * abs(ax.z)
                if t_min > r_b or t_max < -r_b:
                    return
                if e_active:
                    var ov_e = min(r_b - t_min, t_max + r_b)
                    if ov_e < best_pen * Float32(0.99):
                        best_pen = ov_e
                        var tri_mid = (p0 + p1 + p2) * Float32(1.0 / 3.0)
                        best_n = -ax if tri_mid > Float32(0.0) else ax

        if best_pen <= Float32(0.0):
            return

        var wn = box_rot.rotate(best_n).normalized()

        # Clip triangle vertices & box corners penetrating along best_n to generate up to 4 manifold points
        # Supporting plane on box in direction -best_n is at -r_sup; triangle plane along n_out
        for sx_i in range(2):
            var cx = -half.x if sx_i == 0 else half.x
            for sy_i in range(2):
                var cy = -half.y if sy_i == 0 else half.y
                for sz_i in range(2):
                    var cz = -half.z if sz_i == 0 else half.z
                    var corner = Vec3(cx, cy, cz)
                    # Check if box corner penetrates behind triangle plane and projects inside triangle
                    var pen_c = -(corner - lv0).dot(n_out)
                    if pen_c >= Float32(-1.0e-4) and pen_c <= best_pen + Float32(0.05):
                        var proj = corner + n_out * pen_c
                        # Barycentric inside check
                        var e0 = lv1 - lv0; var e1 = lv2 - lv0; var vp = proj - lv0
                        var d00 = e0.dot(e0); var d01 = e0.dot(e1); var d11 = e1.dot(e1)
                        var d20 = vp.dot(e0); var d21 = vp.dot(e1)
                        var denom = d00 * d11 - d01 * d01
                        if abs(denom) > Float32(1.0e-10):
                            var bv = (d11 * d20 - d01 * d21) / denom
                            var bw = (d00 * d21 - d01 * d20) / denom
                            if bv >= Float32(-1.0e-3) and bw >= Float32(-1.0e-3) and (bv + bw) <= Float32(1.001):
                                var wc = box_center + box_rot.rotate(corner)
                                var corner_idx = sx_i * 4 + sy_i * 2 + sz_i
                                cands.append(ContactPoint(wc, wn, max(Float32(0.0), min(best_pen, pen_c)), UInt32(tri_id * 16 + corner_idx)))

        # Also check triangle vertices inside local box [-half, +half]
        var tvs = [lv0, lv1, lv2]
        for vi in range(3):
            var tv = tvs[vi]
            if abs(tv.x) <= half.x + Float32(1.0e-3) and abs(tv.y) <= half.y + Float32(1.0e-3) and abs(tv.z) <= half.z + Float32(1.0e-3):
                var r_box = half.x * abs(best_n.x) + half.y * abs(best_n.y) + half.z * abs(best_n.z)
                var pen_v = r_box + tv.dot(best_n)
                if pen_v > Float32(0.0):
                    var wv = box_center + box_rot.rotate(tv)
                    cands.append(ContactPoint(wv, wn, min(best_pen, pen_v), UInt32(tri_id * 16 + 8 + vi)))

        # Fallback deepest SAT witness if edge-edge contact had neither corner-in-tri nor vert-in-box
        if len(cands) == 0:
            var sup_c = Vec3(
                -half.x if best_n.x >= Float32(0.0) else half.x,
                -half.y if best_n.y >= Float32(0.0) else half.y,
                -half.z if best_n.z >= Float32(0.0) else half.z,
            )
            var cl_tri = Vec3(
                max(-half.x, min(half.x, (lv0.x + lv1.x + lv2.x) * Float32(1.0 / 3.0))),
                max(-half.y, min(half.y, (lv0.y + lv1.y + lv2.y) * Float32(1.0 / 3.0))),
                max(-half.z, min(half.z, (lv0.z + lv1.z + lv2.z) * Float32(1.0 / 3.0))),
            )
            var mid_l = Vec3(
                sup_c.x if abs(best_n.x) > Float32(0.5) else cl_tri.x,
                sup_c.y if abs(best_n.y) > Float32(0.5) else cl_tri.y,
                sup_c.z if abs(best_n.z) > Float32(0.5) else cl_tri.z,
            )
            cands.append(ContactPoint(box_center + box_rot.rotate(mid_l), wn, best_pen, UInt32(tri_id * 16 + 15)))

    def collide_box(self, center: Vec3, rot: Quat, half_extents: Vec3) -> ContactManifold:
        """Discrete oriented box vs BVH8 mesh collision manifold (normal points outward from mesh toward box)."""
        var m = ContactManifold(0, -1)
        var axes = get_box_axes(rot)
        var u0 = axes[0]; var u1 = axes[1]; var u2 = axes[2]
        var wx = abs(u0.x) * half_extents.x + abs(u1.x) * half_extents.y + abs(u2.x) * half_extents.z
        var wy = abs(u0.y) * half_extents.x + abs(u1.y) * half_extents.y + abs(u2.y) * half_extents.z
        var wz = abs(u0.z) * half_extents.x + abs(u1.z) * half_extents.y + abs(u2.z) * half_extents.z
        var w_ext = Vec3(wx, wy, wz)

        var cand_tris = List[Int]()
        self._query_aabb_triangles(center - w_ext, center + w_ext, cand_tris)
        if len(cand_tris) == 0:
            return m^

        var inv_q = rot.conjugate()
        var cands = List[ContactPoint]()
        for k in range(len(cand_tris)):
            var ti = cand_tris[k]
            ref tri = self.triangles[ti]
            var lv0 = inv_q.rotate(tri.v0 - center)
            var lv1 = inv_q.rotate(tri.v1 - center)
            var lv2 = inv_q.rotate(tri.v2 - center)
            var ln = inv_q.rotate(tri.normal)
            Self._collide_local_box_tri(half_extents, lv0, lv1, lv2, ln, tri.active_edges, center, rot, ti, cands)

        reduce_manifold_candidates(m, cands)
        return m^
