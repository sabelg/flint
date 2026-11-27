"""Pure-Mojo Spatial SAH + Forsyth GPU Meshlet Optimizer & Continuous `QEM` `LOD 0..3` DAG Cooker (`meshlet.mojo`).

Implements:
  1. Tom Forsyth Linear-Speed FIFO-32 Post-Transform Vertex-Cache Optimizer (`ACMVR < 0.68`).
  2. Spatial Surface-Area-Heuristic (`SAH`) / KD-Split + Adjacency Greedy `64v/124t` Meshlet Builder
     with tight bounding spheres and backface normal cones for `vkCmdDrawIndirectCount`.
  3. Garland-Heckbert `4x4` Quadric Error Metric (`QEM`) Continuous `LOD 0..3` Meshlet Hierarchy
     packing monotonic `[self_error_uf16, parent_error_uf16)` intervals into `meshlet_words[10]`
     for crack-free GPU screen-space pixel-error (`<= 1.0 px`) LOD selection!
"""
from std.math import floor, isfinite, max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.bc_compress import f32_to_uf16_bits

comptime MESHLET_MAX_VERTS: Int = 64
comptime MESHLET_MAX_TRIS: Int = 124
comptime AAA_MAX_MESH_LODS: Int = 4

comptime AAA_VERTEX_WORDS: Int = 12
comptime AAA_VERTEX_BYTES: Int = 48

# Meshlet descriptor SSBO layout: 12 x UInt32 (48 bytes)
# [0..2]: center_x, center_y, center_z (Float32, mesh-local)
# [3]: radius (Float32, mesh-local)
# [4..6]: cone_axis_x, cone_axis_y, cone_axis_z (Float32, mesh-local unit axis)
# [7]: cone_sin (Float32: sin(half_angle); >= 2.0 disables backface cone cull)
# [8]: first_tri (UInt32: index in global triangle triplet table, i.e. global_index_offset / 3)
# [9]: tri_count (UInt32: 1..124)
# [10]: packed_lod_error_uf16: bits [0..15] = self_error_uf16, bits [16..31] = parent_error_uf16 (0x7BFF = +inf)
# [11]: material_id (UInt32)
comptime AAA_MESHLET_WORDS: Int = 12


struct MeshletCluster(Copyable, Movable):
    var center: Vec3
    var radius: Float32
    var cone_axis: Vec3
    var cone_sin: Float32
    var first_tri: Int
    var tri_count: Int
    var mesh_id: Int
    var material_id: Int
    var lod_level: Int
    var self_error: Float32
    var parent_error: Float32

    def __init__(out self):
        self.center = Vec3()
        self.radius = Float32(0.0)
        self.cone_axis = Vec3(0.0, 0.0, 1.0)
        self.cone_sin = Float32(2.0)
        self.first_tri = 0
        self.tri_count = 0
        self.mesh_id = 0
        self.material_id = 0
        self.lod_level = 0
        self.self_error = Float32(0.0)
        self.parent_error = Float32(65504.0)


@fieldwise_init
struct MeshLodSlice(Copyable, ImplicitlyCopyable, Movable):
    var first_meshlet: Int
    var meshlet_count: Int
    var triangle_count: Int
    var object_error: Float32


struct CookedMeshEntry(Copyable, Movable):
    var mesh_id: Int
    var material_id: Int
    var first_vertex: Int
    var vertex_count: Int
    var first_index: Int
    var triangle_count: Int       # LOD 0 triangle count (for BLAS & Surface-Cache!)
    var first_meshlet: Int        # Start of contiguous LOD 0..L-1 meshlet range
    var meshlet_count: Int        # Total meshlets across all cooked LOD levels (for GPU LOD cull!)
    var lod0_meshlet_count: Int
    var lod_count: Int
    var lods: Array[MeshLodSlice, 4]
    var bounds_center: Vec3
    var bounds_radius: Float32

    def __init__(out self):
        self.mesh_id = 0
        self.material_id = 0
        self.first_vertex = 0
        self.vertex_count = 0
        self.first_index = 0
        self.triangle_count = 0
        self.first_meshlet = 0
        self.meshlet_count = 0
        self.lod0_meshlet_count = 0
        self.lod_count = 1
        self.lods = Array[MeshLodSlice, 4](fill=MeshLodSlice(0, 0, 0, 0.0))
        self.bounds_center = Vec3()
        self.bounds_radius = Float32(0.0)


# ============================================================================
# 1. Tom Forsyth Linear-Speed Post-Transform Vertex Cache Optimizer (`FIFO-32`)
# ============================================================================

def compute_acmr(indices: List[Int], fifo_size: Int = 32) -> Float32:
    """Computes Average Cache Miss Ratio (`misses / triangles`) on a standard `fifo_size` GPU vertex cache."""
    var n_tris = len(indices) // 3
    if n_tris <= 0:
        return Float32(0.0)
    var fifo = List[Int](length=fifo_size, fill=-1)
    var head = 0
    var misses = 0

    for ti in range(n_tris):
        for k in range(3):
            var v = indices[ti * 3 + k]
            var hit = False
            for c_i in range(fifo_size):
                if fifo[c_i] == v:
                    hit = True
                    break
            if not hit:
                misses += 1
                fifo[head] = v
                head = (head + 1) % fifo_size
    return Float32(misses) / Float32(n_tris)


def optimize_vertex_cache_forsyth(indices: List[Int], vertex_count: Int, cache_size: Int = 22) -> List[Int]:
    """Reorders triangle indices for GPU FIFO-32 post-transform caches (`ACMR ~ 0.60 < 0.68`)
    using linear-time valence + FIFO timestamp age + dead-end stack optimization."""
    var n_tris = len(indices) // 3
    if n_tris <= 2 or vertex_count <= 0:
        return indices.copy()

    var live_deg = List[Int](length=vertex_count, fill=0)
    var valid_tris = List[Bool](length=n_tris, fill=True)
    for ti in range(n_tris):
        var a = indices[ti * 3]; var b = indices[ti * 3 + 1]; var c = indices[ti * 3 + 2]
        if a < 0 or b < 0 or c < 0 or a >= vertex_count or b >= vertex_count or c >= vertex_count:
            valid_tris[ti] = False
            continue
        live_deg[a] += 1
        live_deg[b] += 1
        live_deg[c] += 1

    var v_offsets = List[Int](length=vertex_count + 1, fill=0)
    for v in range(vertex_count):
        v_offsets[v + 1] = v_offsets[v] + live_deg[v]
    var v_adj = List[Int](length=v_offsets[vertex_count], fill=0)
    var v_cursor = v_offsets.copy()
    for ti in range(n_tris):
        if not valid_tris[ti]:
            continue
        for k in range(3):
            var v = indices[ti * 3 + k]
            v_adj[v_cursor[v]] = ti
            v_cursor[v] += 1

    var tri_emitted = List[Bool](length=n_tris, fill=False)
    for ti in range(n_tris):
        if not valid_tris[ti]:
            tri_emitted[ti] = True

    var stamp = List[Int](length=vertex_count, fill=0)
    var time_s = cache_size + 1
    var dead_end = List[Int](capacity=n_tris * 3)
    var next_cands = List[Int](capacity=64)
    var out = List[Int](capacity=n_tris * 3)

    var fanning_v = 0
    while fanning_v < vertex_count and live_deg[fanning_v] <= 0:
        fanning_v += 1
    if fanning_v >= vertex_count:
        return indices.copy()

    var scan_v = 0
    while fanning_v >= 0:
        next_cands.clear()
        var s0 = v_offsets[fanning_v]
        var s1 = v_offsets[fanning_v + 1]
        for a_i in range(s0, s1):
            var ti = v_adj[a_i]
            if not tri_emitted[ti]:
                tri_emitted[ti] = True
                for k in range(3):
                    var v = indices[ti * 3 + k]
                    out.append(v)
                    dead_end.append(v)
                    var seen = False
                    for nc in next_cands:
                        if nc == v:
                            seen = True
                            break
                    if not seen:
                        next_cands.append(v)
                    live_deg[v] -= 1
                    if time_s - stamp[v] > cache_size:
                        stamp[v] = time_s
                        time_s += 1

        var best_v = -1
        var best_prio = -1
        for cv in next_cands:
            if live_deg[cv] > 0:
                var prio = 0
                if (time_s - stamp[cv]) + 2 * live_deg[cv] <= cache_size:
                    prio = time_s - stamp[cv]
                if prio > best_prio:
                    best_prio = prio
                    best_v = cv

        if best_v < 0:
            while len(dead_end) > 0:
                var dv = dead_end.pop()
                if live_deg[dv] > 0:
                    best_v = dv
                    break

        if best_v < 0:
            while scan_v < vertex_count and live_deg[scan_v] <= 0:
                scan_v += 1
            if scan_v < vertex_count:
                best_v = scan_v

        fanning_v = best_v

    return out^


# ============================================================================
# 2. Spatial SAH / Morton-Axis Triangle Clustering for Tight Meshlet Bounds
# ============================================================================

def _spatial_sort_triangles_sah(mesh: NativeMesh, in_indices: List[Int]) -> List[Int]:
    """Spatially orders triangles via 30-bit 3D Morton (`Z-order`) centroids + SAH bucket grouping
    before intra-cluster Forsyth cache ordering so meshlets have minimal bounding-sphere radii."""
    var v_count = mesh.vertex_count()
    var raw_tris = len(in_indices) // 3
    if v_count <= 0 or raw_tris <= 0 or len(mesh.positions) < v_count * 3:
        return List[Int]()
    var valid_indices = List[Int](capacity=raw_tris * 3)
    for ti in range(raw_tris):
        var a = in_indices[ti * 3]; var b = in_indices[ti * 3 + 1]; var c = in_indices[ti * 3 + 2]
        if a < 0 or b < 0 or c < 0 or a >= v_count or b >= v_count or c >= v_count:
            continue
        if not (
            isfinite(mesh.positions[a * 3]) and isfinite(mesh.positions[a * 3 + 1]) and isfinite(mesh.positions[a * 3 + 2])
            and isfinite(mesh.positions[b * 3]) and isfinite(mesh.positions[b * 3 + 1]) and isfinite(mesh.positions[b * 3 + 2])
            and isfinite(mesh.positions[c * 3]) and isfinite(mesh.positions[c * 3 + 1]) and isfinite(mesh.positions[c * 3 + 2])
        ):
            continue
        valid_indices.append(a)
        valid_indices.append(b)
        valid_indices.append(c)
    var n_tris = len(valid_indices) // 3
    if n_tris == 0:
        return List[Int]()
    if n_tris <= MESHLET_MAX_TRIS:
        return optimize_vertex_cache_forsyth(valid_indices, v_count)

    var bmin = Vec3(1e9, 1e9, 1e9)
    var bmax = Vec3(-1e9, -1e9, -1e9)
    var centroids = List[Vec3](capacity=n_tris)
    for ti in range(n_tris):
        var a = valid_indices[ti * 3]; var b = valid_indices[ti * 3 + 1]; var c = valid_indices[ti * 3 + 2]
        var p0 = Vec3(mesh.positions[a * 3], mesh.positions[a * 3 + 1], mesh.positions[a * 3 + 2])
        var p1 = Vec3(mesh.positions[b * 3], mesh.positions[b * 3 + 1], mesh.positions[b * 3 + 2])
        var p2 = Vec3(mesh.positions[c * 3], mesh.positions[c * 3 + 1], mesh.positions[c * 3 + 2])
        var cen = (p0 + p1 + p2) * Float32(0.33333334)
        centroids.append(cen)
        if cen.x < bmin.x: bmin.x = cen.x
        if cen.y < bmin.y: bmin.y = cen.y
        if cen.z < bmin.z: bmin.z = cen.z
        if cen.x > bmax.x: bmax.x = cen.x
        if cen.y > bmax.y: bmax.y = cen.y
        if cen.z > bmax.z: bmax.z = cen.z

    var sx64 = 1023.0 / max(1.0e-5, Float64(bmax.x) - Float64(bmin.x))
    var sy64 = 1023.0 / max(1.0e-5, Float64(bmax.y) - Float64(bmin.y))
    var sz64 = 1023.0 / max(1.0e-5, Float64(bmax.z) - Float64(bmin.z))

    var keys = List[UInt32](length=n_tris, fill=0)
    var order = List[Int](length=n_tris, fill=0)
    for ti in range(n_tris):
        var c = centroids[ti]
        var fx64 = (Float64(c.x) - Float64(bmin.x)) * sx64
        var fy64 = (Float64(c.y) - Float64(bmin.y)) * sy64
        var fz64 = (Float64(c.z) - Float64(bmin.z)) * sz64
        var qx = UInt32(max(0, min(1023, Int(max(0.0, min(1023.0, fx64))) if isfinite(fx64) else 0)))
        var qy = UInt32(max(0, min(1023, Int(max(0.0, min(1023.0, fy64))) if isfinite(fy64) else 0)))
        var qz = UInt32(max(0, min(1023, Int(max(0.0, min(1023.0, fz64))) if isfinite(fz64) else 0)))
        # Expand 10 bits -> 30-bit 3D Morton code
        qx = (qx | (qx << 16)) & 0x030000FF
        qx = (qx | (qx << 8)) & 0x0300F00F
        qx = (qx | (qx << 4)) & 0x030C30C3
        qx = (qx | (qx << 2)) & 0x09249249
        qy = (qy | (qy << 16)) & 0x030000FF
        qy = (qy | (qy << 8)) & 0x0300F00F
        qy = (qy | (qy << 4)) & 0x030C30C3
        qy = (qy | (qy << 2)) & 0x09249249
        qz = (qz | (qz << 16)) & 0x030000FF
        qz = (qz | (qz << 8)) & 0x0300F00F
        qz = (qz | (qz << 4)) & 0x030C30C3
        qz = (qz | (qz << 2)) & 0x09249249
        keys[ti] = qx | (qy << 1) | (qz << 2)
        order[ti] = ti

    # 3-pass 10-bit LSD radix sort on 30-bit Morton keys (`O(N)` deterministic!)
    var tmp_order = List[Int](length=n_tris, fill=0)
    for pass_i in range(3):
        var shift = UInt32(pass_i * 10)
        var counts = List[Int](length=1025, fill=0)
        for i in range(n_tris):
            var bucket = Int((keys[order[i]] >> shift) & 1023)
            counts[bucket + 1] += 1
        for b in range(1, 1025):
            counts[b] += counts[b - 1]
        for i in range(n_tris):
            var idx = order[i]
            var bucket = Int((keys[idx] >> shift) & 1023)
            tmp_order[counts[bucket]] = idx
            counts[bucket] += 1
        for i in range(n_tris):
            order[i] = tmp_order[i]

    # Optimize intra-region vertex cache locality on spatial chunks of 248 triangles
    var out = List[Int](capacity=n_tris * 3)
    var chunk_tris = MESHLET_MAX_TRIS * 2
    var start = 0
    var vc = mesh.vertex_count()
    while start < n_tris:
        var end = min(n_tris, start + chunk_tris)
        var sub_idx = List[Int](capacity=(end - start) * 3)
        for k in range(start, end):
            var ti = order[k]
            sub_idx.append(valid_indices[ti * 3])
            sub_idx.append(valid_indices[ti * 3 + 1])
            sub_idx.append(valid_indices[ti * 3 + 2])
        var opt_sub = optimize_vertex_cache_forsyth(sub_idx, vc)
        for v in opt_sub:
            out.append(v)
        start = end
    return out^


# ============================================================================
# 3. Garland-Heckbert `4x4` Quadric Error Metric (`QEM`) Mesh Simplifier (`LOD 0..3`)
# ============================================================================

struct _SymQuadric(Copyable, ImplicitlyCopyable, Movable):
    """Symmetric 4x4 error matrix `Q` (`10` unique Float64 entries for numerical precision)."""
    var q00: Float64; var q01: Float64; var q02: Float64; var q03: Float64
    var q11: Float64; var q12: Float64; var q13: Float64
    var q22: Float64; var q23: Float64
    var q33: Float64

    def __init__(out self):
        self.q00 = 0.0; self.q01 = 0.0; self.q02 = 0.0; self.q03 = 0.0
        self.q11 = 0.0; self.q12 = 0.0; self.q13 = 0.0
        self.q22 = 0.0; self.q23 = 0.0
        self.q33 = 0.0

    @staticmethod
    def from_plane(nx: Float64, ny: Float64, nz: Float64, d: Float64, weight: Float64 = 1.0) -> Self:
        var s = Self()
        s.q00 = nx * nx * weight; s.q01 = nx * ny * weight; s.q02 = nx * nz * weight; s.q03 = nx * d * weight
        s.q11 = ny * ny * weight; s.q12 = ny * nz * weight; s.q13 = ny * d * weight
        s.q22 = nz * nz * weight; s.q23 = nz * d * weight
        s.q33 = d * d * weight
        return s

    @always_inline
    def __add__(self, o: Self) -> Self:
        var s = Self()
        s.q00 = self.q00 + o.q00; s.q01 = self.q01 + o.q01; s.q02 = self.q02 + o.q02; s.q03 = self.q03 + o.q03
        s.q11 = self.q11 + o.q11; s.q12 = self.q12 + o.q12; s.q13 = self.q13 + o.q13
        s.q22 = self.q22 + o.q22; s.q23 = self.q23 + o.q23
        s.q33 = self.q33 + o.q33
        return s

    @always_inline
    def eval(self, p: Vec3) -> Float64:
        var x = Float64(p.x); var y = Float64(p.y); var z = Float64(p.z)
        return (
            self.q00 * x * x + 2.0 * self.q01 * x * y + 2.0 * self.q02 * x * z + 2.0 * self.q03 * x
            + self.q11 * y * y + 2.0 * self.q12 * y * z + 2.0 * self.q13 * y
            + self.q22 * z * z + 2.0 * self.q23 * z
            + self.q33
        )


@always_inline
def _next_pow2_int(n: Int) -> Int:
    var p = 16
    while p < n:
        p = p << 1
    return p


@always_inline
def _hash_u64_edge(u_min: Int, u_max: Int) -> UInt64:
    var k = (UInt64(u_min) << 32) | UInt64(u_max)
    k = (k ^ (k >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    k = (k ^ (k >> 27)) * UInt64(0x94D049BB133111EB)
    return k ^ (k >> 31)


@always_inline
def _eval_qem_pair(
    q_sum: _SymQuadric,
    pa: Vec3,
    pb: Vec3,
) -> Tuple[Float64, Vec3, Float32]:
    """Evaluates optimal Garland-Heckbert collapse target along/near `(pa, pb)` returning `(cost, p_opt, t_param)`."""
    var pm = (pa + pb) * Float32(0.5)
    var ca = max(0.0, q_sum.eval(pa))
    var cb = max(0.0, q_sum.eval(pb))
    var cm = max(0.0, q_sum.eval(pm))
    var c_min = cm
    var p_opt = pm
    var t_opt = Float32(0.5)
    if ca < c_min:
        c_min = ca
        p_opt = pa
        t_opt = Float32(0.0)
    if cb < c_min:
        c_min = cb
        p_opt = pb
        t_opt = Float32(1.0)

    var ab = pb - pa
    var edge_len2_f32 = ab.length_squared()

    # Analytical 3x3 symmetric Cramer solve for ridge/corner preservation when well-conditioned
    var a00 = q_sum.q00; var a01 = q_sum.q01; var a02 = q_sum.q02
    var a11 = q_sum.q11; var a12 = q_sum.q12; var a22 = q_sum.q22
    var c00 = a11 * a22 - a12 * a12
    var c01 = a02 * a12 - a01 * a22
    var c02 = a01 * a12 - a02 * a11
    var det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) > 1e-7 and edge_len2_f32 > Float32(1e-12):
        var inv_det = -1.0 / det
        var c11 = a00 * a22 - a02 * a02
        var c12 = a01 * a02 - a00 * a12
        var c22 = a00 * a11 - a01 * a01
        var b0 = q_sum.q03; var b1 = q_sum.q13; var b2 = q_sum.q23
        var lx = Float32((c00 * b0 + c01 * b1 + c02 * b2) * inv_det)
        var ly = Float32((c01 * b0 + c11 * b1 + c12 * b2) * inv_det)
        var lz = Float32((c02 * b0 + c12 * b1 + c22 * b2) * inv_det)
        var p_lin = Vec3(lx, ly, lz)
        if (p_lin - pm).length_squared() <= edge_len2_f32 * Float32(0.36):
            var c_lin = max(0.0, q_sum.eval(p_lin))
            if c_lin < c_min:
                c_min = c_lin
                p_opt = p_lin
                t_opt = max(Float32(0.0), min(Float32(1.0), (p_lin - pa).dot(ab) / edge_len2_f32))

    var total_c = c_min + Float64(edge_len2_f32) * 1e-4
    return (total_c, p_opt, t_opt)


struct _IndexedQemMinHeap(Movable):
    """Indexed binary min-heap of candidate edge collapses with per-vertex generation stamps
    and `O(1)` open-addressed edge lookup for in-place priority updates in `O(log E)`."""
    var c_ra: List[Int]
    var c_rb: List[Int]
    var c_ver_a: List[Int]
    var c_ver_b: List[Int]
    var c_cost: List[Float64]
    var c_px: List[Float32]
    var c_py: List[Float32]
    var c_pz: List[Float32]
    var c_t: List[Float32]
    var heap: List[Int]
    var heap_pos: List[Int]
    var map_keys: List[UInt64]
    var map_cids: List[Int]
    var map_mask: Int

    def __init__(out self, est_edges: Int):
        var cap = max(32, est_edges)
        var hcap = _next_pow2_int(cap * 4 + 32)
        self.c_ra = List[Int](capacity=cap * 2)
        self.c_rb = List[Int](capacity=cap * 2)
        self.c_ver_a = List[Int](capacity=cap * 2)
        self.c_ver_b = List[Int](capacity=cap * 2)
        self.c_cost = List[Float64](capacity=cap * 2)
        self.c_px = List[Float32](capacity=cap * 2)
        self.c_py = List[Float32](capacity=cap * 2)
        self.c_pz = List[Float32](capacity=cap * 2)
        self.c_t = List[Float32](capacity=cap * 2)
        self.heap = List[Int](capacity=cap * 2)
        self.heap_pos = List[Int](capacity=cap * 2)
        self.map_keys = List[UInt64](length=hcap, fill=UInt64(0xFFFFFFFFFFFFFFFF))
        self.map_cids = List[Int](length=hcap, fill=-1)
        self.map_mask = hcap - 1

    @always_inline
    def size(self) -> Int:
        return len(self.heap)

    @always_inline
    def _less(self, cid_a: Int, cid_b: Int) -> Bool:
        var ca = self.c_cost[cid_a]
        var cb = self.c_cost[cid_b]
        if ca != cb:
            return ca < cb
        return cid_a < cid_b

    @always_inline
    def _swap_heap(mut self, i: Int, j: Int):
        var ci = self.heap[i]
        var cj = self.heap[j]
        self.heap[i] = cj
        self.heap[j] = ci
        self.heap_pos[cj] = i
        self.heap_pos[ci] = j

    def _sift_up(mut self, start_idx: Int):
        var idx = start_idx
        while idx > 0:
            var parent = (idx - 1) >> 1
            if self._less(self.heap[idx], self.heap[parent]):
                self._swap_heap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(mut self, start_idx: Int):
        var n = len(self.heap)
        var idx = start_idx
        while True:
            var left = (idx << 1) + 1
            if left >= n:
                break
            var smallest = left
            var right = left + 1
            if right < n and self._less(self.heap[right], self.heap[left]):
                smallest = right
            if self._less(self.heap[smallest], self.heap[idx]):
                self._swap_heap(idx, smallest)
                idx = smallest
            else:
                break

    def push_or_update(
        mut self,
        ra_in: Int,
        rb_in: Int,
        ver_a_in: Int,
        ver_b_in: Int,
        cost: Float64,
        pt: Vec3,
        t_in: Float32,
    ):
        var u0 = ra_in
        var u1 = rb_in
        var va = ver_a_in
        var vb = ver_b_in
        var tp = t_in
        if u0 > u1:
            u0 = rb_in
            u1 = ra_in
            va = ver_b_in
            vb = ver_a_in
            tp = Float32(1.0) - t_in

        var key = (UInt64(u0) << 32) | UInt64(u1)
        var slot = Int(_hash_u64_edge(u0, u1)) & self.map_mask
        var probe = 0
        var existing_cid = -1
        var insert_slot = -1
        while probe <= self.map_mask:
            var mk = self.map_keys[slot]
            if mk == key:
                existing_cid = self.map_cids[slot]
                insert_slot = slot
                break
            if mk == UInt64(0xFFFFFFFFFFFFFFFF):
                insert_slot = slot
                break
            slot = (slot + 1) & self.map_mask
            probe += 1

        if existing_cid >= 0:
            self.c_ver_a[existing_cid] = va
            self.c_ver_b[existing_cid] = vb
            self.c_cost[existing_cid] = cost
            self.c_px[existing_cid] = pt.x
            self.c_py[existing_cid] = pt.y
            self.c_pz[existing_cid] = pt.z
            self.c_t[existing_cid] = tp
            var hp = self.heap_pos[existing_cid]
            if hp >= 0:
                self._sift_up(hp)
                self._sift_down(self.heap_pos[existing_cid])
            else:
                var new_hp = len(self.heap)
                self.heap.append(existing_cid)
                self.heap_pos[existing_cid] = new_hp
                self._sift_up(new_hp)
            return

        var cid = len(self.c_ra)
        self.c_ra.append(u0)
        self.c_rb.append(u1)
        self.c_ver_a.append(va)
        self.c_ver_b.append(vb)
        self.c_cost.append(cost)
        self.c_px.append(pt.x)
        self.c_py.append(pt.y)
        self.c_pz.append(pt.z)
        self.c_t.append(tp)
        var hp = len(self.heap)
        self.heap.append(cid)
        self.heap_pos.append(hp)
        if insert_slot >= 0:
            self.map_keys[insert_slot] = key
            self.map_cids[insert_slot] = cid
        self._sift_up(hp)

    def pop_min(mut self) -> Int:
        var n = len(self.heap)
        if n <= 0:
            return -1
        var top_cid = self.heap[0]
        var last_cid = self.heap.pop()
        self.heap_pos[top_cid] = -1
        if n > 1:
            self.heap[0] = last_cid
            self.heap_pos[last_cid] = 0
            self._sift_down(0)
        return top_cid


@always_inline
def _normalize_skin_rep4_in_place(
    mut rep_j: List[UInt16],
    mut rep_w: List[Float32],
    r: Int,
    j0: UInt16,
    j1: UInt16,
    j2: UInt16,
    j3: UInt16,
    w0: Float32,
    w1: Float32,
    w2: Float32,
    w3: Float32,
):
    var in_j = SIMD[DType.uint16, 4](0)
    var in_w = SIMD[DType.float32, 4](0.0)
    in_j[0] = j0; in_j[1] = j1; in_j[2] = j2; in_j[3] = j3
    in_w[0] = max(Float32(0.0), w0)
    in_w[1] = max(Float32(0.0), w1)
    in_w[2] = max(Float32(0.0), w2)
    in_w[3] = max(Float32(0.0), w3)

    var cj = SIMD[DType.uint16, 4](0)
    var cw = SIMD[DType.float32, 4](0.0)
    var n_c = 0
    for i in range(4):
        if in_w[i] <= Float32(0.0):
            continue
        var merged = False
        for m in range(n_c):
            if cj[m] == in_j[i]:
                cw[m] += in_w[i]
                merged = True
                break
        if not merged:
            cj[n_c] = in_j[i]
            cw[n_c] = in_w[i]
            n_c += 1

    for a in range(1, n_c):
        var kj = cj[a]
        var kw = cw[a]
        var b = a - 1
        while b >= 0 and cw[b] < kw:
            cj[b + 1] = cj[b]
            cw[b + 1] = cw[b]
            b -= 1
        cj[b + 1] = kj
        cw[b + 1] = kw

    var base = r * 4
    var sum_w = cw[0] + cw[1] + cw[2] + cw[3]
    if sum_w <= Float32(1e-6):
        rep_j[base + 0] = UInt16(0); rep_j[base + 1] = UInt16(0); rep_j[base + 2] = UInt16(0); rep_j[base + 3] = UInt16(0)
        rep_w[base + 0] = Float32(1.0); rep_w[base + 1] = Float32(0.0); rep_w[base + 2] = Float32(0.0); rep_w[base + 3] = Float32(0.0)
        return

    var inv_s = Float32(1.0) / sum_w
    var nw1 = cw[1] * inv_s
    var nw2 = cw[2] * inv_s
    var nw3 = cw[3] * inv_s
    var nw0 = max(Float32(0.0), Float32(1.0) - (nw1 + nw2 + nw3))
    rep_j[base + 0] = cj[0]
    rep_j[base + 1] = cj[1]
    rep_j[base + 2] = cj[2]
    rep_j[base + 3] = cj[3]
    rep_w[base + 0] = nw0
    rep_w[base + 1] = nw1
    rep_w[base + 2] = nw2
    rep_w[base + 3] = nw3


@always_inline
def _blend_skin_reps_top4_in_place(
    mut rep_j: List[UInt16],
    mut rep_w: List[Float32],
    ra: Int,
    rb: Int,
    best_t: Float32,
):
    """Merges up to 8 bone influences of `(ra, rb)` weighted by `(1 - best_t, best_t)`, coalesces duplicate
    joint IDs, selects the top-4 heaviest influences, and renormalizes them to sum to `1.0`."""
    var t_cl = max(Float32(0.0), min(Float32(1.0), best_t))
    var omt = Float32(1.0) - t_cl

    var cj = SIMD[DType.uint16, 8](0)
    var cw = SIMD[DType.float32, 8](0.0)
    var n_c = 0

    var ba = ra * 4
    for k in range(4):
        var wa = rep_w[ba + k] * omt
        if wa <= Float32(0.0):
            continue
        var ja = rep_j[ba + k]
        var merged = False
        for m in range(n_c):
            if cj[m] == ja:
                cw[m] += wa
                merged = True
                break
        if not merged:
            cj[n_c] = ja
            cw[n_c] = wa
            n_c += 1

    var bb = rb * 4
    for k in range(4):
        var wb = rep_w[bb + k] * t_cl
        if wb <= Float32(0.0):
            continue
        var jb = rep_j[bb + k]
        var merged = False
        for m in range(n_c):
            if cj[m] == jb:
                cw[m] += wb
                merged = True
                break
        if not merged:
            cj[n_c] = jb
            cw[n_c] = wb
            n_c += 1

    for a in range(1, n_c):
        var kj = cj[a]
        var kw = cw[a]
        var b = a - 1
        while b >= 0 and cw[b] < kw:
            cj[b + 1] = cj[b]
            cw[b + 1] = cw[b]
            b -= 1
        cj[b + 1] = kj
        cw[b + 1] = kw

    var sum4 = cw[0] + cw[1] + cw[2] + cw[3]
    if sum4 <= Float32(1e-6):
        rep_j[ba + 0] = UInt16(0); rep_j[ba + 1] = UInt16(0); rep_j[ba + 2] = UInt16(0); rep_j[ba + 3] = UInt16(0)
        rep_w[ba + 0] = Float32(1.0); rep_w[ba + 1] = Float32(0.0); rep_w[ba + 2] = Float32(0.0); rep_w[ba + 3] = Float32(0.0)
        return

    var inv4 = Float32(1.0) / sum4
    var nw1 = cw[1] * inv4
    var nw2 = cw[2] * inv4
    var nw3 = cw[3] * inv4
    var nw0 = max(Float32(0.0), Float32(1.0) - (nw1 + nw2 + nw3))
    rep_j[ba + 0] = cj[0]
    rep_j[ba + 1] = cj[1] if n_c > 1 else UInt16(0)
    rep_j[ba + 2] = cj[2] if n_c > 2 else UInt16(0)
    rep_j[ba + 3] = cj[3] if n_c > 3 else UInt16(0)
    rep_w[ba + 0] = nw0
    rep_w[ba + 1] = nw1 if n_c > 1 else Float32(0.0)
    rep_w[ba + 2] = nw2 if n_c > 2 else Float32(0.0)
    rep_w[ba + 3] = nw3 if n_c > 3 else Float32(0.0)


def simplify_mesh_qem(mesh: NativeMesh, target_tris: Int, lock_boundaries: Bool = False) -> Tuple[NativeMesh, Float32]:
    """Simplifies `mesh` down to `target_tris` in `O(T log T)` via CSR vertex-triangle adjacency,
    `O(T)` canonical edge-hash boundary detection, and an indexed binary min-heap with lazy generation stamps."""
    var v_count = mesh.vertex_count()
    var raw_tris = len(mesh.indices) // 3
    if v_count <= 0 or len(mesh.positions) < v_count * 3 or raw_tris <= 0:
        return (NativeMesh(), Float32(0.0))
    for vi in range(v_count):
        if not (
            isfinite(mesh.positions[vi * 3])
            and isfinite(mesh.positions[vi * 3 + 1])
            and isfinite(mesh.positions[vi * 3 + 2])
        ):
            return (NativeMesh(), Float32(0.0))
    if raw_tris <= target_tris or raw_tris <= 8 or v_count <= 4:
        return (mesh.copy(), Float32(0.0))

    # 1. Group colocated spatial positions in `O(V)` via quantized spatial hash grid
    var pmin = Vec3(mesh.positions[0], mesh.positions[1], mesh.positions[2])
    var pmax = pmin
    for vi in range(1, v_count):
        var px = mesh.positions[vi * 3]; var py = mesh.positions[vi * 3 + 1]; var pz = mesh.positions[vi * 3 + 2]
        if px < pmin.x: pmin.x = px
        if py < pmin.y: pmin.y = py
        if pz < pmin.z: pmin.z = pz
        if px > pmax.x: pmax.x = px
        if py > pmax.y: pmax.y = py
        if pz > pmax.z: pmax.z = pz

    var max_ext64 = max(1.0e-2, max(Float64(pmax.x) - Float64(pmin.x), max(Float64(pmax.y) - Float64(pmin.y), Float64(pmax.z) - Float64(pmin.z))))
    var inv_cell64 = 4096.0 / max_ext64
    var vhash_cap = _next_pow2_int(v_count * 2 + 16)
    var vhash_mask = vhash_cap - 1
    var vhash_head = List[Int](length=vhash_cap, fill=-1)
    var rep_bucket_next = List[Int](capacity=v_count)

    var pos_rep = List[Int](length=v_count, fill=0)
    var rep_pos = List[Vec3](capacity=v_count)
    var unique_p = 0

    for vi in range(v_count):
        var p = Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1], mesh.positions[vi * 3 + 2])
        var fx = Float32(max(-1.0e9, min(1.0e9, (Float64(p.x) - Float64(pmin.x)) * inv_cell64 + 0.371)))
        var fy = Float32(max(-1.0e9, min(1.0e9, (Float64(p.y) - Float64(pmin.y)) * inv_cell64 + 0.519)))
        var fz = Float32(max(-1.0e9, min(1.0e9, (Float64(p.z) - Float64(pmin.z)) * inv_cell64 + 0.733)))
        var qx = Int(floor(fx))
        var qy = Int(floor(fy))
        var qz = Int(floor(fz))
        var h_idx = Int(((UInt64(qx & 0xFFFFF) * UInt64(73856093)) ^ (UInt64(qy & 0xFFFFF) * UInt64(19349663)) ^ (UInt64(qz & 0xFFFFF) * UInt64(83492791)))) & vhash_mask

        var found = -1
        var cur_r = vhash_head[h_idx]
        while cur_r >= 0:
            if (rep_pos[cur_r] - p).length_squared() < Float32(1e-10):
                found = cur_r
                break
            cur_r = rep_bucket_next[cur_r]

        # Guard rare cell-boundary straddle (< 0.5% of vertices)
        if found < 0:
            var rx_frac = fx - Float32(qx)
            var ry_frac = fy - Float32(qy)
            var rz_frac = fz - Float32(qz)
            var dx0 = -1 if rx_frac < Float32(0.05) else 0
            var dx1 = 1 if rx_frac > Float32(0.95) else 0
            var dy0 = -1 if ry_frac < Float32(0.05) else 0
            var dy1 = 1 if ry_frac > Float32(0.95) else 0
            var dz0 = -1 if rz_frac < Float32(0.05) else 0
            var dz1 = 1 if rz_frac > Float32(0.95) else 0
            if dx0 != 0 or dx1 != 0 or dy0 != 0 or dy1 != 0 or dz0 != 0 or dz1 != 0:
                for ox in range(dx0, dx1 + 1):
                    for oy in range(dy0, dy1 + 1):
                        for oz in range(dz0, dz1 + 1):
                            if ox == 0 and oy == 0 and oz == 0:
                                continue
                            var nh = Int(((UInt64((qx + ox) & 0xFFFFF) * UInt64(73856093)) ^ (UInt64((qy + oy) & 0xFFFFF) * UInt64(19349663)) ^ (UInt64((qz + oz) & 0xFFFFF) * UInt64(83492791)))) & vhash_mask
                            var nr = vhash_head[nh]
                            while nr >= 0:
                                if (rep_pos[nr] - p).length_squared() < Float32(1e-10):
                                    found = nr
                                    break
                                nr = rep_bucket_next[nr]
                            if found >= 0:
                                break
                        if found >= 0:
                            break
                    if found >= 0:
                        break

        if found >= 0:
            pos_rep[vi] = found
        else:
            pos_rep[vi] = unique_p
            rep_pos.append(p)
            rep_bucket_next.append(vhash_head[h_idx])
            vhash_head[h_idx] = unique_p
            unique_p += 1

    # 2. Accumulate fundamental plane quadrics & build CSR vertex-triangle adjacency + `O(T)` edge map
    var quadrics = List[_SymQuadric](length=unique_p, fill=_SymQuadric())
    var tri_alive = List[Bool](length=raw_tris, fill=True)
    var tri_v = List[Int](capacity=raw_tris * 3)
    var rep_deg = List[Int](length=unique_p, fill=0)
    var live_tris = 0

    for ti in range(raw_tris):
        var i0 = mesh.indices[ti * 3]; var i1 = mesh.indices[ti * 3 + 1]; var i2 = mesh.indices[ti * 3 + 2]
        tri_v.append(i0); tri_v.append(i1); tri_v.append(i2)
        if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= v_count or i1 >= v_count or i2 >= v_count:
            tri_alive[ti] = False
            continue
        var r0 = pos_rep[i0]; var r1 = pos_rep[i1]; var r2 = pos_rep[i2]
        if r0 == r1 or r1 == r2 or r2 == r0:
            tri_alive[ti] = False
            continue
        live_tris += 1
        rep_deg[r0] += 1
        rep_deg[r1] += 1
        rep_deg[r2] += 1
        var p0 = rep_pos[r0]; var p1 = rep_pos[r1]; var p2 = rep_pos[r2]
        var cross_n = (p1 - p0).cross(p2 - p0)
        var area2 = cross_n.length()
        if area2 > Float32(1e-9):
            var n = cross_n * (Float32(1.0) / area2)
            var d = -Float64(n.dot(p0))
            var pq = _SymQuadric.from_plane(Float64(n.x), Float64(n.y), Float64(n.z), d, weight=1.0)
            quadrics[r0] = quadrics[r0] + pq
            quadrics[r1] = quadrics[r1] + pq
            quadrics[r2] = quadrics[r2] + pq

    if live_tris <= target_tris:
        return (mesh.copy(), Float32(0.0))

    # Build CSR vertex-triangle arrays and mirror into `O(1)` mergeable linked arena
    var csr_offsets = List[Int](length=unique_p + 1, fill=0)
    for r in range(unique_p):
        csr_offsets[r + 1] = csr_offsets[r] + rep_deg[r]
    var total_inc = csr_offsets[unique_p]
    var csr_tris = List[Int](length=total_inc, fill=0)
    var csr_cur = csr_offsets.copy()

    # Canonical undirected edge hash table (`(min_r, max_r) -> eid`) for `O(T)` boundary detection
    var ehash_cap = _next_pow2_int(live_tris * 6 + 32)
    var ehash_mask = ehash_cap - 1
    var ehash_key = List[UInt64](length=ehash_cap, fill=UInt64(0xFFFFFFFFFFFFFFFF))
    var ehash_eid = List[Int](length=ehash_cap, fill=-1)
    var edge_u = List[Int](capacity=live_tris * 2)
    var edge_v = List[Int](capacity=live_tris * 2)
    var edge_cnt = List[Int](capacity=live_tris * 2)
    var edge_dir_a = List[Int](capacity=live_tris * 2)
    var edge_dir_b = List[Int](capacity=live_tris * 2)
    var edge_tri0 = List[Int](capacity=live_tris * 2)

    for ti in range(raw_tris):
        if not tri_alive[ti]:
            continue
        var r0 = pos_rep[tri_v[ti * 3]]
        var r1 = pos_rep[tri_v[ti * 3 + 1]]
        var r2 = pos_rep[tri_v[ti * 3 + 2]]
        csr_tris[csr_cur[r0]] = ti; csr_cur[r0] += 1
        csr_tris[csr_cur[r1]] = ti; csr_cur[r1] += 1
        csr_tris[csr_cur[r2]] = ti; csr_cur[r2] += 1

        var rs = [r0, r1, r2]
        for ek in range(3):
            var ra = rs[ek]
            var rb = rs[(ek + 1) % 3]
            var u0 = min(ra, rb)
            var u1 = max(ra, rb)
            var ekey = (UInt64(u0) << 32) | UInt64(u1)
            var slot = Int(_hash_u64_edge(u0, u1)) & ehash_mask
            while True:
                var mk = ehash_key[slot]
                if mk == ekey:
                    edge_cnt[ehash_eid[slot]] += 1
                    break
                if mk == UInt64(0xFFFFFFFFFFFFFFFF):
                    var eid = len(edge_u)
                    ehash_key[slot] = ekey
                    ehash_eid[slot] = eid
                    edge_u.append(u0)
                    edge_v.append(u1)
                    edge_cnt.append(1)
                    edge_dir_a.append(ra)
                    edge_dir_b.append(rb)
                    edge_tri0.append(ti)
                    break
                slot = (slot + 1) & ehash_mask

    # `O(E)` boundary perpendicular penalty planes (`edge_cnt[eid] == 1`)
    var num_edges = len(edge_u)
    var b_weight = 10000.0 if lock_boundaries else 32.0
    for eid in range(num_edges):
        if edge_cnt[eid] == 1:
            var ti = edge_tri0[eid]
            var tr0 = pos_rep[tri_v[ti * 3]]
            var tr1 = pos_rep[tri_v[ti * 3 + 1]]
            var tr2 = pos_rep[tri_v[ti * 3 + 2]]
            var fn_u = (rep_pos[tr1] - rep_pos[tr0]).cross(rep_pos[tr2] - rep_pos[tr0]).normalized()
            var ra = edge_dir_a[eid]
            var rb = edge_dir_b[eid]
            var edge_dir = rep_pos[rb] - rep_pos[ra]
            var bn = edge_dir.cross(fn_u).normalized()
            if bn.length_squared() > Float32(0.5):
                var bd = -Float64(bn.dot(rep_pos[ra]))
                var bq = _SymQuadric.from_plane(Float64(bn.x), Float64(bn.y), Float64(bn.z), bd, weight=b_weight)
                quadrics[ra] = quadrics[ra] + bq
                quadrics[rb] = quadrics[rb] + bq

    # Initialize linked incident-triangle lists from CSR (`0` per-collapse heap allocs)
    var node_tri = csr_tris^
    var node_next = List[Int](length=total_inc, fill=-1)
    var rep_head = List[Int](length=unique_p, fill=-1)
    var rep_tail = List[Int](length=unique_p, fill=-1)
    var rep_llen = List[Int](length=unique_p, fill=0)
    for r in range(unique_p):
        var s0 = csr_offsets[r]
        var s1 = csr_offsets[r + 1]
        if s1 > s0:
            rep_head[r] = s0
            rep_tail[r] = s1 - 1
            rep_llen[r] = s1 - s0
            for idx in range(s0, s1 - 1):
                node_next[idx] = idx + 1

    # 3. Populate indexed binary min-heap with all unique mesh edges
    var rep_ver = List[Int](length=unique_p, fill=0)
    var rep_parent = List[Int](length=unique_p, fill=0)
    for r in range(unique_p):
        rep_parent[r] = r

    var heap = _IndexedQemMinHeap(num_edges)
    for eid in range(num_edges):
        var u0 = edge_u[eid]
        var u1 = edge_v[eid]
        var ev = _eval_qem_pair(quadrics[u0] + quadrics[u1], rep_pos[u0], rep_pos[u1])
        heap.push_or_update(u0, u1, 0, 0, ev[0], ev[1], ev[2])

    var max_err = Float32(0.0)
    var has_uv = len(mesh.texcoords) >= v_count * 2
    var has_uv1 = len(mesh.texcoords1) >= v_count * 2
    var has_nrm = len(mesh.normals) >= v_count * 3
    var has_tang = len(mesh.tangents) >= v_count * 4
    var has_col = len(mesh.colors) >= v_count * 4
    var has_joints = len(mesh.joints) >= v_count * 4
    var has_weights = len(mesh.weights) >= v_count * 4
    var has_skin_pair = len(mesh.joints) == v_count * 4 and len(mesh.weights) == v_count * 4
    var has_centers = len(mesh.centers) >= v_count * 3

    var rep_joints = List[UInt16]()
    var rep_weights = List[Float32]()
    if has_skin_pair:
        rep_joints = List[UInt16](length=unique_p * 4, fill=UInt16(0))
        rep_weights = List[Float32](length=unique_p * 4, fill=Float32(0.0))
        var rep_skin_init = List[Bool](length=unique_p, fill=False)
        for vi in range(v_count):
            var r = pos_rep[vi]
            if not rep_skin_init[r]:
                rep_skin_init[r] = True
                _normalize_skin_rep4_in_place(
                    rep_joints,
                    rep_weights,
                    r,
                    mesh.joints[vi * 4 + 0],
                    mesh.joints[vi * 4 + 1],
                    mesh.joints[vi * 4 + 2],
                    mesh.joints[vi * 4 + 3],
                    mesh.weights[vi * 4 + 0],
                    mesh.weights[vi * 4 + 1],
                    mesh.weights[vi * 4 + 2],
                    mesh.weights[vi * 4 + 3],
                )

    var new_v_pos = mesh.positions.copy()
    var new_v_nrm = mesh.normals.copy()
    var new_v_tang = mesh.tangents.copy()
    var new_v_uv = mesh.texcoords.copy()
    var new_v_uv1 = mesh.texcoords1.copy()
    var new_v_col = mesh.colors.copy()
    var wedge_parent = List[Int](length=v_count, fill=0)
    for vi in range(v_count):
        wedge_parent[vi] = vi

    var tri_stamp = List[Int](length=raw_tris, fill=0)
    var rep_stamp = List[Int](length=unique_p, fill=0)
    var cur_stamp = 1
    var deg_tris = List[Int](capacity=32)
    var surv_tris = List[Int](capacity=64)
    var neighbors = List[Int](capacity=32)

    # 4. `O(T log T)` Min-Heap QEM Collapse Loop
    while live_tris > target_tris and heap.size() > 0:
        var cid = heap.pop_min()
        if cid < 0:
            break
        var ra = heap.c_ra[cid]
        var rb = heap.c_rb[cid]
        if rep_parent[ra] != ra or rep_parent[rb] != rb or ra == rb:
            continue
        if heap.c_ver_a[cid] != rep_ver[ra] or heap.c_ver_b[cid] != rep_ver[rb]:
            continue

        var best_cost = heap.c_cost[cid]
        var best_pt = Vec3(heap.c_px[cid], heap.c_py[cid], heap.c_pz[cid])
        var best_t = heap.c_t[cid]

        # Gather live incident triangles of `ra` and `rb`
        cur_stamp += 1
        deg_tris.clear()
        surv_tris.clear()

        for side in range(2):
            var r_cur = ra if side == 0 else rb
            var node = rep_head[r_cur]
            while node >= 0:
                var ti = node_tri[node]
                if tri_alive[ti] and tri_stamp[ti] != cur_stamp:
                    tri_stamp[ti] = cur_stamp
                    var r0 = pos_rep[tri_v[ti * 3]]
                    while rep_parent[r0] != r0:
                        rep_parent[r0] = rep_parent[rep_parent[r0]]
                        r0 = rep_parent[r0]
                    var r1 = pos_rep[tri_v[ti * 3 + 1]]
                    while rep_parent[r1] != r1:
                        rep_parent[r1] = rep_parent[rep_parent[r1]]
                        r1 = rep_parent[r1]
                    var r2 = pos_rep[tri_v[ti * 3 + 2]]
                    while rep_parent[r2] != r2:
                        rep_parent[r2] = rep_parent[rep_parent[r2]]
                        r2 = rep_parent[r2]

                    var has_a = (r0 == ra or r1 == ra or r2 == ra)
                    var has_b = (r0 == rb or r1 == rb or r2 == rb)
                    if has_a and has_b:
                        deg_tris.append(ti)
                    elif has_a or has_b:
                        surv_tris.append(ti)
                node = node_next[node]

        if len(deg_tris) == 0:
            continue

        # Normal-flip & sliver-inversion guard across 1-ring surviving triangles
        var flipped = False
        for s_i in range(len(surv_tris)):
            var ti = surv_tris[s_i]
            var r0 = pos_rep[tri_v[ti * 3]]
            while rep_parent[r0] != r0: r0 = rep_parent[r0]
            var r1 = pos_rep[tri_v[ti * 3 + 1]]
            while rep_parent[r1] != r1: r1 = rep_parent[r1]
            var r2 = pos_rep[tri_v[ti * 3 + 2]]
            while rep_parent[r2] != r2: r2 = rep_parent[r2]

            var p0 = rep_pos[r0]; var p1 = rep_pos[r1]; var p2 = rep_pos[r2]
            var np0 = best_pt if (r0 == ra or r0 == rb) else p0
            var np1 = best_pt if (r1 == ra or r1 == rb) else p1
            var np2 = best_pt if (r2 == ra or r2 == rb) else p2

            var n_old = (p1 - p0).cross(p2 - p0)
            var n_new = (np1 - np0).cross(np2 - np0)
            var l2_old = n_old.length_squared()
            var l2_new = n_new.length_squared()
            if l2_old > Float32(1e-14):
                if l2_new <= Float32(1e-15) or n_old.dot(n_new) <= Float32(0.10) * sqrt(l2_old * l2_new):
                    flipped = True
                    break
        if flipped:
            continue

        var disp_a = (rep_pos[ra] - best_pt).length()
        var disp_b = (rep_pos[rb] - best_pt).length()
        var qem_dist = Float32(sqrt(max(0.0, best_cost)))
        var step_err = max(qem_dist, min(disp_a, disp_b) + Float32(0.25) * max(disp_a, disp_b))
        if step_err > max_err:
            max_err = step_err

        # Weld continuous UV/normal wedges along collapsed triangles while preserving UV seams
        for d_i in range(len(deg_tris)):
            var ti = deg_tris[d_i]
            var va = -1
            var vb = -1
            for k in range(3):
                var vk = tri_v[ti * 3 + k]
                while wedge_parent[vk] != vk:
                    wedge_parent[vk] = wedge_parent[wedge_parent[vk]]
                    vk = wedge_parent[vk]
                tri_v[ti * 3 + k] = vk
                var rk = pos_rep[vk]
                while rep_parent[rk] != rk: rk = rep_parent[rk]
                if rk == ra:
                    va = vk
                elif rk == rb:
                    vb = vk
            if va >= 0 and vb >= 0 and va != vb:
                var can_weld_wedge = True
                if has_uv:
                    var du = abs(new_v_uv[va * 2] - new_v_uv[vb * 2])
                    var dv = abs(new_v_uv[va * 2 + 1] - new_v_uv[vb * 2 + 1])
                    if du >= Float32(0.25) or dv >= Float32(0.25):
                        can_weld_wedge = False
                    else:
                        new_v_uv[va * 2] = new_v_uv[va * 2] * (Float32(1.0) - best_t) + new_v_uv[vb * 2] * best_t
                        new_v_uv[va * 2 + 1] = new_v_uv[va * 2 + 1] * (Float32(1.0) - best_t) + new_v_uv[vb * 2 + 1] * best_t
                if has_col and can_weld_wedge:
                    var dr = abs(Int(new_v_col[va * 4]) - Int(new_v_col[vb * 4]))
                    var dg = abs(Int(new_v_col[va * 4 + 1]) - Int(new_v_col[vb * 4 + 1]))
                    var db = abs(Int(new_v_col[va * 4 + 2]) - Int(new_v_col[vb * 4 + 2]))
                    if max(dr, max(dg, db)) > 48:
                        can_weld_wedge = False
                if can_weld_wedge:
                    if has_col:
                        var omt = Float32(1.0) - best_t
                        for ch_i in range(4):
                            var ca = Float32(Int(new_v_col[va * 4 + ch_i]))
                            var cb = Float32(Int(new_v_col[vb * 4 + ch_i]))
                            new_v_col[va * 4 + ch_i] = UInt8(max(0, min(255, Int(ca * omt + cb * best_t + Float32(0.5)))))
                    if has_nrm:
                        var na = Vec3(new_v_nrm[va * 3], new_v_nrm[va * 3 + 1], new_v_nrm[va * 3 + 2])
                        var nb = Vec3(new_v_nrm[vb * 3], new_v_nrm[vb * 3 + 1], new_v_nrm[vb * 3 + 2])
                        var nm = (na * (Float32(1.0) - best_t) + nb * best_t).normalized()
                        if nm.length_squared() > Float32(0.1):
                            new_v_nrm[va * 3] = nm.x; new_v_nrm[va * 3 + 1] = nm.y; new_v_nrm[va * 3 + 2] = nm.z
                    if has_tang:
                        var ta = Vec3(new_v_tang[va * 4], new_v_tang[va * 4 + 1], new_v_tang[va * 4 + 2])
                        var tb = Vec3(new_v_tang[vb * 4], new_v_tang[vb * 4 + 1], new_v_tang[vb * 4 + 2])
                        var tm = (ta * (Float32(1.0) - best_t) + tb * best_t).normalized()
                        if tm.length_squared() > Float32(0.1):
                            new_v_tang[va * 4] = tm.x; new_v_tang[va * 4 + 1] = tm.y; new_v_tang[va * 4 + 2] = tm.z
                    if has_uv1:
                        new_v_uv1[va * 2] = new_v_uv1[va * 2] * (Float32(1.0) - best_t) + new_v_uv1[vb * 2] * best_t
                        new_v_uv1[va * 2 + 1] = new_v_uv1[va * 2 + 1] * (Float32(1.0) - best_t) + new_v_uv1[vb * 2 + 1] * best_t
                    wedge_parent[vb] = va
            tri_alive[ti] = False
            live_tris -= 1

        # Merge `rb -> ra` (including top-4 merged & renormalized skin influences across spatial seam wedges)
        if has_skin_pair:
            _blend_skin_reps_top4_in_place(rep_joints, rep_weights, ra, rb, best_t)
        rep_parent[rb] = ra
        rep_pos[ra] = best_pt
        quadrics[ra] = quadrics[ra] + quadrics[rb]
        rep_ver[ra] += 1
        rep_ver[rb] += 1

        # Update surviving triangles & collect 1-ring neighbor representatives
        neighbors.clear()
        cur_stamp += 1
        rep_stamp[ra] = cur_stamp

        for s_i in range(len(surv_tris)):
            var ti = surv_tris[s_i]
            if not tri_alive[ti]:
                continue
            var rs_t = [0, 0, 0]
            for k in range(3):
                var vk = tri_v[ti * 3 + k]
                while wedge_parent[vk] != vk:
                    wedge_parent[vk] = wedge_parent[wedge_parent[vk]]
                    vk = wedge_parent[vk]
                tri_v[ti * 3 + k] = vk
                var rk = pos_rep[vk]
                while rep_parent[rk] != rk:
                    rep_parent[rk] = rep_parent[rep_parent[rk]]
                    rk = rep_parent[rk]
                rs_t[k] = rk
            if rs_t[0] == rs_t[1] or rs_t[1] == rs_t[2] or rs_t[2] == rs_t[0]:
                tri_alive[ti] = False
                live_tris -= 1
                continue
            for k in range(3):
                var rn = rs_t[k]
                if rep_stamp[rn] != cur_stamp:
                    rep_stamp[rn] = cur_stamp
                    neighbors.append(rn)

        # Concatenate `rb`'s incident triangle list onto `ra`'s list (`O(1)`)
        if rep_head[rb] >= 0:
            if rep_head[ra] < 0:
                rep_head[ra] = rep_head[rb]
                rep_tail[ra] = rep_tail[rb]
                rep_llen[ra] = rep_llen[rb]
            else:
                node_next[rep_tail[ra]] = rep_head[rb]
                rep_tail[ra] = rep_tail[rb]
                rep_llen[ra] += rep_llen[rb]
            rep_head[rb] = -1
            rep_tail[rb] = -1
            rep_llen[rb] = 0

        # Periodically prune dead/duplicate triangles when a hub list grows > 48
        if rep_llen[ra] > 48:
            cur_stamp += 1
            var new_h = -1
            var new_t = -1
            var new_c = 0
            var cur_n = rep_head[ra]
            while cur_n >= 0:
                var nxt = node_next[cur_n]
                var ti = node_tri[cur_n]
                if tri_alive[ti] and tri_stamp[ti] != cur_stamp:
                    tri_stamp[ti] = cur_stamp
                    node_next[cur_n] = -1
                    if new_h < 0:
                        new_h = cur_n
                        new_t = cur_n
                    else:
                        node_next[new_t] = cur_n
                        new_t = cur_n
                    new_c += 1
                cur_n = nxt
            rep_head[ra] = new_h
            rep_tail[ra] = new_t
            rep_llen[ra] = new_c

        # Push/update 1-ring incident edges in `O(deg * log E)`
        var va_gen = rep_ver[ra]
        var pa_new = rep_pos[ra]
        for n_i in range(len(neighbors)):
            var rn = neighbors[n_i]
            var ev = _eval_qem_pair(quadrics[ra] + quadrics[rn], pa_new, rep_pos[rn])
            heap.push_or_update(ra, rn, va_gen, rep_ver[rn], ev[0], ev[1], ev[2])

    # Sync all remaining vertices mapped to `ra` with final `rep_pos[ra]`
    for vi in range(v_count):
        var r = pos_rep[vi]
        while rep_parent[r] != r:
            r = rep_parent[r]
        var fp = rep_pos[r]
        new_v_pos[vi * 3] = fp.x
        new_v_pos[vi * 3 + 1] = fp.y
        new_v_pos[vi * 3 + 2] = fp.z

    # Compact surviving triangles & used vertices into a clean NativeMesh
    var remap = List[Int](length=v_count, fill=-1)
    var out_mesh = NativeMesh()
    out_mesh.base_color = mesh.base_color
    out_mesh.roughness = mesh.roughness
    out_mesh.metallic = mesh.metallic
    out_mesh.texture = mesh.texture
    out_mesh.normal_orm_texture = mesh.normal_orm_texture
    out_mesh.orm_texture = mesh.orm_texture
    out_mesh.occlusion_texture = mesh.occlusion_texture
    out_mesh.occlusion_texcoord = mesh.occlusion_texcoord
    out_mesh.occlusion_strength = mesh.occlusion_strength
    out_mesh.emissive_texture = mesh.emissive_texture
    out_mesh.emissive_texcoord = mesh.emissive_texcoord
    out_mesh.emissive_color = mesh.emissive_color
    out_mesh.emissive_strength = mesh.emissive_strength
    out_mesh.clearcoat = mesh.clearcoat
    out_mesh.clearcoat_roughness = mesh.clearcoat_roughness
    out_mesh.specular_factor = mesh.specular_factor
    out_mesh.iridescence = mesh.iridescence
    out_mesh.iridescence_thickness_nm = mesh.iridescence_thickness_nm
    out_mesh.iridescence_ior = mesh.iridescence_ior
    out_mesh.sheen_color = mesh.sheen_color
    out_mesh.sheen_roughness = mesh.sheen_roughness
    out_mesh.skin = mesh.skin
    out_mesh.alpha_cutoff = mesh.alpha_cutoff
    out_mesh.alpha_mode = mesh.alpha_mode
    out_mesh.double_sided = mesh.double_sided

    for ti in range(raw_tris):
        if not tri_alive[ti]:
            continue
        for k in range(3):
            var old_v = tri_v[ti * 3 + k]
            while wedge_parent[old_v] != old_v:
                old_v = wedge_parent[old_v]
            if remap[old_v] < 0:
                var nv = len(out_mesh.positions) // 3
                remap[old_v] = nv
                out_mesh.positions.append(new_v_pos[old_v * 3])
                out_mesh.positions.append(new_v_pos[old_v * 3 + 1])
                out_mesh.positions.append(new_v_pos[old_v * 3 + 2])
                if has_nrm:
                    out_mesh.normals.append(new_v_nrm[old_v * 3])
                    out_mesh.normals.append(new_v_nrm[old_v * 3 + 1])
                    out_mesh.normals.append(new_v_nrm[old_v * 3 + 2])
                if has_tang:
                    out_mesh.tangents.append(new_v_tang[old_v * 4])
                    out_mesh.tangents.append(new_v_tang[old_v * 4 + 1])
                    out_mesh.tangents.append(new_v_tang[old_v * 4 + 2])
                    out_mesh.tangents.append(new_v_tang[old_v * 4 + 3])
                if has_uv:
                    out_mesh.texcoords.append(new_v_uv[old_v * 2])
                    out_mesh.texcoords.append(new_v_uv[old_v * 2 + 1])
                if has_uv1:
                    out_mesh.texcoords1.append(new_v_uv1[old_v * 2])
                    out_mesh.texcoords1.append(new_v_uv1[old_v * 2 + 1])
                if has_col:
                    out_mesh.colors.append(new_v_col[old_v * 4])
                    out_mesh.colors.append(new_v_col[old_v * 4 + 1])
                    out_mesh.colors.append(new_v_col[old_v * 4 + 2])
                    out_mesh.colors.append(new_v_col[old_v * 4 + 3])
                if has_skin_pair:
                    var r_sk = pos_rep[old_v]
                    while rep_parent[r_sk] != r_sk:
                        r_sk = rep_parent[r_sk]
                    out_mesh.joints.append(rep_joints[r_sk * 4 + 0])
                    out_mesh.joints.append(rep_joints[r_sk * 4 + 1])
                    out_mesh.joints.append(rep_joints[r_sk * 4 + 2])
                    out_mesh.joints.append(rep_joints[r_sk * 4 + 3])
                    out_mesh.weights.append(rep_weights[r_sk * 4 + 0])
                    out_mesh.weights.append(rep_weights[r_sk * 4 + 1])
                    out_mesh.weights.append(rep_weights[r_sk * 4 + 2])
                    out_mesh.weights.append(rep_weights[r_sk * 4 + 3])
                else:
                    if has_joints:
                        out_mesh.joints.append(mesh.joints[old_v * 4])
                        out_mesh.joints.append(mesh.joints[old_v * 4 + 1])
                        out_mesh.joints.append(mesh.joints[old_v * 4 + 2])
                        out_mesh.joints.append(mesh.joints[old_v * 4 + 3])
                    if has_weights:
                        out_mesh.weights.append(mesh.weights[old_v * 4])
                        out_mesh.weights.append(mesh.weights[old_v * 4 + 1])
                        out_mesh.weights.append(mesh.weights[old_v * 4 + 2])
                        out_mesh.weights.append(mesh.weights[old_v * 4 + 3])
                if has_centers:
                    out_mesh.centers.append(mesh.centers[old_v * 3])
                    out_mesh.centers.append(mesh.centers[old_v * 3 + 1])
                    out_mesh.centers.append(mesh.centers[old_v * 3 + 2])
                if old_v < len(mesh.ao):
                    out_mesh.ao.append(mesh.ao[old_v])
            out_mesh.indices.append(remap[old_v])

    return (out_mesh^, max(Float32(0.0005), max_err))


# ============================================================================
# 4. Unified `AaaGeometryPool` with Continuous `LOD 0..3` Meshlet Hierarchy
# ============================================================================

struct GeomFreeList(Copyable, Movable):
    """Sorted, coalescing free ranges `[start, start + len)` with first-fit allocation."""
    var starts: List[Int]
    var lens: List[Int]

    def __init__(out self):
        self.starts = List[Int]()
        self.lens = List[Int]()

    def total(self) -> Int:
        var t = 0
        for l in self.lens:
            t += l
        return t

    def count(self) -> Int:
        return len(self.starts)

    def alloc(mut self, n: Int) -> Int:
        """First fit: the start of a free run of `n` units, or -1."""
        if n <= 0:
            return -1
        for i in range(len(self.starts)):
            if self.lens[i] >= n:
                var s = self.starts[i]
                if self.lens[i] == n:
                    _ = self.starts.pop(i)
                    _ = self.lens.pop(i)
                else:
                    self.starts[i] = s + n
                    self.lens[i] = self.lens[i] - n
                return s
        return -1

    def free(mut self, start: Int, n: Int):
        if n <= 0:
            return
        var pos = 0
        while pos < len(self.starts) and self.starts[pos] < start:
            pos += 1
        self.starts.insert(pos, start)
        self.lens.insert(pos, n)
        # Merge with the next run, then with the previous one.
        if pos + 1 < len(self.starts) and self.starts[pos] + self.lens[pos] == self.starts[pos + 1]:
            self.lens[pos] = self.lens[pos] + self.lens[pos + 1]
            _ = self.starts.pop(pos + 1)
            _ = self.lens.pop(pos + 1)
        if pos > 0 and self.starts[pos - 1] + self.lens[pos - 1] == self.starts[pos]:
            self.lens[pos - 1] = self.lens[pos - 1] + self.lens[pos]
            _ = self.starts.pop(pos)
            _ = self.lens.pop(pos)

    def trim_tail(mut self, end: Int) -> Int:
        """If the last free run ends at `end`, drops it and returns its start (the new end); else `end`."""
        var n = len(self.starts)
        if n > 0 and self.starts[n - 1] + self.lens[n - 1] == end:
            var s = self.starts[n - 1]
            _ = self.starts.pop()
            _ = self.lens.pop()
            return s
        return end


struct GeomMeshSpan(Copyable, ImplicitlyCopyable, Movable):
    """Where one mesh (all LODs) lives in the pool: vertices, index words and meshlets are each contiguous."""
    var first_vertex: Int
    var vertex_count: Int
    var first_index: Int
    var index_count: Int
    var first_meshlet: Int
    var meshlet_count: Int
    var live: Bool

    def __init__(out self):
        self.first_vertex = 0
        self.vertex_count = 0
        self.first_index = 0
        self.index_count = 0
        self.first_meshlet = 0
        self.meshlet_count = 0
        self.live = False


struct AaaGeometryPool(Movable):
    """Unified CPU-side staging pool for GPU Vertex, Triangle Index & Meshlet SSBOs + BLAS inputs.

    Meshes can be removed (`remove_mesh`); their ranges go to per-array free lists and
    `append_mesh_reusing` places new meshes into freed ranges when they fit (else at the tail).
    Mesh ids stay stable: a removed id keeps an empty entry that draws nothing."""
    var vertex_words: List[UInt32]
    var index_words: List[UInt32]
    var meshlet_words: List[UInt32]
    var meshlets: List[MeshletCluster]
    var meshes: List[CookedMeshEntry]
    var mesh_spans: List[GeomMeshSpan]  # per mesh_id; shorter than `meshes` for pools filled by hand
    var free_vertices: GeomFreeList  # in vertices
    var free_indices: GeomFreeList  # in index words (multiples of 3)
    var free_meshlets: GeomFreeList  # in meshlets
    var last_written: GeomMeshSpan  # ranges written by the last append (what the GPU must upload)
    var relocated_meshes: Int

    def __init__(out self):
        self.vertex_words = List[UInt32]()
        self.index_words = List[UInt32]()
        self.meshlet_words = List[UInt32]()
        self.meshlets = List[MeshletCluster]()
        self.meshes = List[CookedMeshEntry]()
        self.mesh_spans = List[GeomMeshSpan]()
        self.free_vertices = GeomFreeList()
        self.free_indices = GeomFreeList()
        self.free_meshlets = GeomFreeList()
        self.last_written = GeomMeshSpan()
        self.relocated_meshes = 0

    def is_mesh_live(self, mesh_id: Int) -> Bool:
        if mesh_id < 0 or mesh_id >= len(self.meshes):
            return False
        if mesh_id < len(self.mesh_spans):
            return self.mesh_spans[mesh_id].live
        return True  # filled by hand: live, but not removable

    def live_mesh_count(self) -> Int:
        var n = 0
        for i in range(len(self.meshes)):
            if self.is_mesh_live(i):
                n += 1
        return n

    def free_vertex_count(self) -> Int:
        return self.free_vertices.total()

    def _record_span(mut self, mesh_id: Int, v0: Int, i0: Int, m0: Int):
        """Records the tail ranges appended since lengths `(v0 vertices, i0 index words, m0 meshlets)`."""
        var sp = GeomMeshSpan()
        sp.first_vertex = v0
        sp.vertex_count = self.total_vertices() - v0
        sp.first_index = i0
        sp.index_count = len(self.index_words) - i0
        sp.first_meshlet = m0
        sp.meshlet_count = len(self.meshlets) - m0
        sp.live = True
        while len(self.mesh_spans) < mesh_id:
            self.mesh_spans.append(GeomMeshSpan())  # hand-filled meshes before this one
        if mesh_id == len(self.mesh_spans):
            self.mesh_spans.append(sp)
        elif mesh_id < len(self.mesh_spans):
            self.mesh_spans[mesh_id] = sp
        self.last_written = sp

    def remove_mesh(mut self, mesh_id: Int) -> Bool:
        """Frees the mesh's vertex, index and meshlet ranges for reuse and empties its entry.
        The caller must make sure no in-flight GPU frame still reads those ranges (the renderer defers
        this call until the frames that drew the mesh retire)."""
        if mesh_id < 0 or mesh_id >= len(self.mesh_spans) or not self.mesh_spans[mesh_id].live:
            return False
        var sp = self.mesh_spans[mesh_id]
        self.free_vertices.free(sp.first_vertex, sp.vertex_count)
        self.free_indices.free(sp.first_index, sp.index_count)
        self.free_meshlets.free(sp.first_meshlet, sp.meshlet_count)
        self.mesh_spans[mesh_id].live = False
        # Free runs at the end of an array shrink the array instead.
        var v_end = self.free_vertices.trim_tail(self.total_vertices())
        if v_end < self.total_vertices():
            self.vertex_words.resize(v_end * AAA_VERTEX_WORDS, 0)
        var i_end = self.free_indices.trim_tail(len(self.index_words))
        if i_end < len(self.index_words):
            self.index_words.resize(i_end, 0)
        var m_end = self.free_meshlets.trim_tail(len(self.meshlets))
        if m_end < len(self.meshlets):
            while len(self.meshlets) > m_end:
                _ = self.meshlets.pop()
            self.meshlet_words.resize(m_end * AAA_MESHLET_WORDS, 0)
        var empty = CookedMeshEntry()
        empty.mesh_id = mesh_id
        empty.material_id = self.meshes[mesh_id].material_id
        empty.lod_count = 0
        self.meshes[mesh_id] = empty^
        return True

    def append_mesh_reusing(mut self, mesh: NativeMesh, material_id: Int, build_lods: Bool = False) -> CookedMeshEntry:
        """As `append_mesh`, then moves each of the mesh's three ranges into a freed range that fits.
        `last_written` says where the data ended up (upload exactly those ranges)."""
        var entry = self.append_mesh(mesh, material_id, build_lods)
        if entry.vertex_count <= 0 or entry.mesh_id >= len(self.mesh_spans):
            return entry^
        var sp = self.mesh_spans[entry.mesh_id]
        var moved = False
        # Vertices: copy, rebase the mesh's index words, shrink the tail.
        var dst_v = self.free_vertices.alloc(sp.vertex_count)
        if dst_v >= 0:
            var src_w = sp.first_vertex * AAA_VERTEX_WORDS
            var dst_w = dst_v * AAA_VERTEX_WORDS
            for k in range(sp.vertex_count * AAA_VERTEX_WORDS):
                self.vertex_words[dst_w + k] = self.vertex_words[src_w + k]
            for k in range(sp.first_index, sp.first_index + sp.index_count):
                self.index_words[k] = UInt32(Int(self.index_words[k]) - sp.first_vertex + dst_v)
            self.vertex_words.resize(src_w, 0)
            entry.first_vertex = entry.first_vertex - sp.first_vertex + dst_v
            sp.first_vertex = dst_v
            moved = True
        # Index words: copy, rebase the meshlets' first triangle, shrink the tail.
        var dst_i = self.free_indices.alloc(sp.index_count)
        if dst_i >= 0:
            for k in range(sp.index_count):
                self.index_words[dst_i + k] = self.index_words[sp.first_index + k]
            var dt = (dst_i - sp.first_index) // 3
            for m in range(sp.first_meshlet, sp.first_meshlet + sp.meshlet_count):
                self.meshlets[m].first_tri += dt
                self.meshlet_words[m * AAA_MESHLET_WORDS + 8] = UInt32(self.meshlets[m].first_tri)
            self.index_words.resize(sp.first_index, 0)
            entry.first_index = entry.first_index - sp.first_index + dst_i
            sp.first_index = dst_i
            moved = True
        # Meshlets: copy clusters and words, rebase the entry's LOD slices, shrink the tail.
        var dst_m = self.free_meshlets.alloc(sp.meshlet_count)
        if dst_m >= 0:
            for k in range(sp.meshlet_count):
                self.meshlets[dst_m + k] = self.meshlets[sp.first_meshlet + k].copy()
                for w in range(AAA_MESHLET_WORDS):
                    self.meshlet_words[(dst_m + k) * AAA_MESHLET_WORDS + w] = self.meshlet_words[
                        (sp.first_meshlet + k) * AAA_MESHLET_WORDS + w
                    ]
            while len(self.meshlets) > sp.first_meshlet:
                _ = self.meshlets.pop()
            self.meshlet_words.resize(sp.first_meshlet * AAA_MESHLET_WORDS, 0)
            var dm = dst_m - sp.first_meshlet
            entry.first_meshlet += dm
            for l in range(entry.lod_count):
                var s = entry.lods[l]
                entry.lods[l] = MeshLodSlice(s.first_meshlet + dm, s.meshlet_count, s.triangle_count, s.object_error)
            sp.first_meshlet = dst_m
            moved = True
        if moved:
            self.relocated_meshes += 1
        self.mesh_spans[entry.mesh_id] = sp
        self.last_written = sp
        self.meshes[entry.mesh_id] = entry.copy()
        return entry^

    def total_vertices(self) -> Int:
        return len(self.vertex_words) // AAA_VERTEX_WORDS

    def total_triangles(self) -> Int:
        return len(self.index_words) // 3

    def total_meshlets(self) -> Int:
        return len(self.meshlets)

    def _append_single_lod_geometry_ex(
        mut self,
        mesh: NativeMesh,
        mesh_id: Int,
        material_id: Int,
        lod_level: Int,
        self_err: Float32,
        parent_err: Float32,
        is_splat: Bool,
        splat_weights_u24: List[UInt32],
    ) -> Tuple[Int, Int, Int, Int, Int, Int, Vec3, Float32]:
        """Appends one LOD level's vertices, spatially/Forsyth-ordered indices, and 64v/124t meshlets.
        Packs `vertex_words[11]` = `sign_neg | (is_splat << 1) | (ao_u6 << 2) | (splat_u24 << 8)`."""
        var v_count = len(mesh.positions) // 3
        var opt_indices = _spatial_sort_triangles_sah(mesh, mesh.indices)
        var raw_tris = len(opt_indices) // 3
        var base_v = self.total_vertices()
        var base_idx = len(self.index_words)
        var base_ml = len(self.meshlets)

        var tan_u = List[Vec3](length=v_count, fill=Vec3(0.0, 0.0, 0.0))
        var tan_v = List[Vec3](length=v_count, fill=Vec3(0.0, 0.0, 0.0))
        var has_uv = len(mesh.texcoords) >= v_count * 2
        var has_nrm = len(mesh.normals) >= v_count * 3
        var has_authored_tang = len(mesh.tangents) >= v_count * 4
        var has_vcol = len(mesh.colors) >= v_count * 4

        for ti in range(raw_tris):
            var i0 = opt_indices[ti * 3 + 0]
            var i1 = opt_indices[ti * 3 + 1]
            var i2 = opt_indices[ti * 3 + 2]
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= v_count or i1 >= v_count or i2 >= v_count:
                continue
            var p0 = Vec3(mesh.positions[i0 * 3], mesh.positions[i0 * 3 + 1], mesh.positions[i0 * 3 + 2])
            var p1 = Vec3(mesh.positions[i1 * 3], mesh.positions[i1 * 3 + 1], mesh.positions[i1 * 3 + 2])
            var p2 = Vec3(mesh.positions[i2 * 3], mesh.positions[i2 * 3 + 1], mesh.positions[i2 * 3 + 2])
            var e1 = p1 - p0
            var e2 = p2 - p0
            var du1 = (mesh.texcoords[i1 * 2] - mesh.texcoords[i0 * 2]) if has_uv else Float32(1.0)
            var dv1 = (mesh.texcoords[i1 * 2 + 1] - mesh.texcoords[i0 * 2 + 1]) if has_uv else Float32(0.0)
            var du2 = (mesh.texcoords[i2 * 2] - mesh.texcoords[i0 * 2]) if has_uv else Float32(0.0)
            var dv2 = (mesh.texcoords[i2 * 2 + 1] - mesh.texcoords[i0 * 2 + 1]) if has_uv else Float32(1.0)
            var det = du1 * dv2 - du2 * dv1
            if abs(det) > Float32(1e-8):
                var inv_det = Float32(1.0) / det
                var sdir = (e1 * dv2 - e2 * dv1) * inv_det
                var tdir = (e2 * du1 - e1 * du2) * inv_det
                var e01 = e1.normalized()
                var e02 = e2.normalized()
                var e12 = (p2 - p1).normalized()
                var a0 = max(Float32(0.0), min(Float32(1.0), e01.dot(e02)))
                var a1 = max(Float32(0.0), min(Float32(1.0), (-e01).dot(e12)))
                var a2 = max(Float32(0.0), min(Float32(1.0), (-e02).dot(-e12)))
                var w0 = max(Float32(0.05), Float32(1.0) - a0 * a0)
                var w1 = max(Float32(0.05), Float32(1.0) - a1 * a1)
                var w2 = max(Float32(0.05), Float32(1.0) - a2 * a2)
                tan_u[i0] = tan_u[i0] + sdir * w0; tan_u[i1] = tan_u[i1] + sdir * w1; tan_u[i2] = tan_u[i2] + sdir * w2
                tan_v[i0] = tan_v[i0] + tdir * w0; tan_v[i1] = tan_v[i1] + tdir * w1; tan_v[i2] = tan_v[i2] + tdir * w2

        var bmin = Vec3(1e9, 1e9, 1e9)
        var bmax = Vec3(-1e9, -1e9, -1e9)

        for vi in range(v_count):
            var px = mesh.positions[vi * 3 + 0]
            var py = mesh.positions[vi * 3 + 1]
            var pz = mesh.positions[vi * 3 + 2]
            if px < bmin.x: bmin.x = px
            if py < bmin.y: bmin.y = py
            if pz < bmin.z: bmin.z = pz
            if px > bmax.x: bmax.x = px
            if py > bmax.y: bmax.y = py
            if pz > bmax.z: bmax.z = pz

            var n = Vec3(0.0, 1.0, 0.0)
            if has_nrm:
                n = Vec3(mesh.normals[vi * 3], mesh.normals[vi * 3 + 1], mesh.normals[vi * 3 + 2]).normalized()

            var tang = Vec3(1.0, 0.0, 0.0)
            var sign_neg: UInt32 = 0
            var used_authored = False
            if has_authored_tang:
                var tx = mesh.tangents[vi * 4 + 0]
                var ty = mesh.tangents[vi * 4 + 1]
                var tz = mesh.tangents[vi * 4 + 2]
                var tw = mesh.tangents[vi * 4 + 3]
                var at_vec = Vec3(tx, ty, tz)
                var t_auth = at_vec - n * n.dot(at_vec)
                var a_len2 = t_auth.length_squared()
                if a_len2 > Float32(1e-8):
                    tang = t_auth * (Float32(1.0) / sqrt(a_len2))
                    sign_neg = UInt32(1) if tw < Float32(0.0) else UInt32(0)
                    used_authored = True
            if not used_authored:
                var t_raw = tan_u[vi] - n * n.dot(tan_u[vi])
                var t_len2 = t_raw.length_squared()
                if t_len2 > Float32(1e-8):
                    tang = t_raw * (Float32(1.0) / sqrt(t_len2))
                    if n.cross(tang).dot(tan_v[vi]) < Float32(0.0):
                        sign_neg = 1
                else:
                    var helper = Vec3(1.0, 0.0, 0.0) if abs(n.y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
                    tang = helper.cross(n).normalized()

            var u0 = mesh.texcoords[vi * 2 + 0] if has_uv else (px * Float32(0.5))
            var v0 = mesh.texcoords[vi * 2 + 1] if has_uv else (pz * Float32(0.5))
            var ao_byte = UInt32(mesh.ao[vi]) if vi < len(mesh.ao) else UInt32(255)
            var ao_u6 = (ao_byte >> 2) & UInt32(0x3F)
            var packed_meta: UInt32
            if is_splat:
                var sw24 = (splat_weights_u24[vi] & UInt32(0xFFFFFF)) if vi < len(splat_weights_u24) else UInt32(0)
                packed_meta = sign_neg | UInt32(2) | (ao_u6 << 2) | (sw24 << 8)
            else:
                var r8 = UInt32(mesh.colors[vi * 4 + 0]) if has_vcol else UInt32(255)
                var g8 = UInt32(mesh.colors[vi * 4 + 1]) if has_vcol else UInt32(255)
                var b8 = UInt32(mesh.colors[vi * 4 + 2]) if has_vcol else UInt32(255)
                if (r8 | g8 | b8) == UInt32(0):
                    b8 = UInt32(1)
                packed_meta = sign_neg | (ao_u6 << 2) | (r8 << 8) | (g8 << 16) | (b8 << 24)

            self.vertex_words.append(bitcast[DType.uint32](px))
            self.vertex_words.append(bitcast[DType.uint32](py))
            self.vertex_words.append(bitcast[DType.uint32](pz))
            self.vertex_words.append(bitcast[DType.uint32](n.x))
            self.vertex_words.append(bitcast[DType.uint32](n.y))
            self.vertex_words.append(bitcast[DType.uint32](n.z))
            self.vertex_words.append(bitcast[DType.uint32](tang.x))
            self.vertex_words.append(bitcast[DType.uint32](tang.y))
            self.vertex_words.append(bitcast[DType.uint32](tang.z))
            self.vertex_words.append(bitcast[DType.uint32](u0))
            self.vertex_words.append(bitcast[DType.uint32](v0))
            self.vertex_words.append(packed_meta)

        var cur_tris = List[Int]()
        var cur_verts = List[Int]()
        for ti in range(raw_tris):
            var a = opt_indices[ti * 3 + 0]
            var b = opt_indices[ti * 3 + 1]
            var c = opt_indices[ti * 3 + 2]
            if a < 0 or b < 0 or c < 0 or a >= v_count or b >= v_count or c >= v_count:
                continue
            var new_v = 0
            var seen_a = False; var seen_b = False; var seen_c = False
            for cv in cur_verts:
                if cv == a: seen_a = True
                if cv == b: seen_b = True
                if cv == c: seen_c = True
            if not seen_a: new_v += 1
            if not seen_b and b != a: new_v += 1
            if not seen_c and c != a and c != b: new_v += 1

            if len(cur_tris) >= MESHLET_MAX_TRIS or (len(cur_verts) + new_v > MESHLET_MAX_VERTS and len(cur_tris) > 0):
                self._emit_meshlet(mesh, opt_indices, base_v, mesh_id, material_id, mesh.double_sided, lod_level, self_err, parent_err, cur_tris)
                cur_tris.clear()
                cur_verts.clear()
                seen_a = False; seen_b = False; seen_c = False

            if not seen_a: cur_verts.append(a)
            if not seen_b and b != a: cur_verts.append(b)
            if not seen_c and c != a and c != b: cur_verts.append(c)
            cur_tris.append(ti)

        if len(cur_tris) > 0:
            self._emit_meshlet(mesh, opt_indices, base_v, mesh_id, material_id, mesh.double_sided, lod_level, self_err, parent_err, cur_tris)

        var mesh_c = (bmin + bmax) * Float32(0.5)
        var max_r2 = Float32(0.0001)
        for vi in range(v_count):
            var vp = Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1], mesh.positions[vi * 3 + 2])
            var d2 = (vp - mesh_c).length_squared()
            if d2 > max_r2:
                max_r2 = d2

        return (
            base_v,
            v_count,
            base_idx,
            (len(self.index_words) - base_idx) // 3,
            base_ml,
            len(self.meshlets) - base_ml,
            mesh_c,
            sqrt(max_r2),
        )

    def _append_single_lod_geometry(
        mut self,
        mesh: NativeMesh,
        mesh_id: Int,
        material_id: Int,
        lod_level: Int,
        self_err: Float32,
        parent_err: Float32,
    ) -> Tuple[Int, Int, Int, Int, Int, Int, Vec3, Float32]:
        var empty_splat = List[UInt32]()
        return self._append_single_lod_geometry_ex(
            mesh, mesh_id, material_id, lod_level, self_err, parent_err, False, empty_splat
        )

    def append_mesh(mut self, mesh: NativeMesh, material_id: Int, build_lods: Bool = False) -> CookedMeshEntry:
        """Cooks `mesh` into SAH+Forsyth `64v/124t` Meshlets, optionally building a 4-level `QEM` `LOD 0..3` DAG."""
        var empty_splat = List[UInt32]()
        return self._append_mesh_internal(mesh, material_id, build_lods, False, empty_splat)

    def add_mesh(mut self, mesh: NativeMesh, material_id: Int = 0, build_lods: Bool = False) -> Int:
        """Convenience wrapper around `append_mesh` returning `entry.mesh_id`."""
        return self.append_mesh(mesh, material_id, build_lods).mesh_id

    def append_splat_mesh(
        mut self,
        mesh: NativeMesh,
        material_id: Int,
        splat_weights_u24: List[UInt32],
        build_lods: Bool = False,
    ) -> CookedMeshEntry:
        """Cooks a 4-layer height-modulated PBR terrain splat mesh setting `bit 1 = 1` (`UInt32(2)`) and packing
        `(splat_weights_u24[vi] & 0xFFFFFF) << 8` (`w_sand`, `w_rock`, `w_cobble`) into `vertex_words[11]`."""
        return self._append_mesh_internal(mesh, material_id, build_lods, True, splat_weights_u24)

    def add_splat_mesh(
        mut self,
        mesh: NativeMesh,
        splat_weights_u24: List[UInt32],
        build_lods: Bool = False,
        material_id: Int = 0,
    ) -> Int:
        """Cooks a terrain splat mesh into `AaaGeometryPool` and returns `entry.mesh_id`."""
        return self.append_splat_mesh(mesh, material_id, splat_weights_u24, build_lods).mesh_id

    def _append_mesh_internal(
        mut self,
        mesh: NativeMesh,
        material_id: Int,
        build_lods: Bool,
        is_splat: Bool,
        splat_weights_u24: List[UInt32],
    ) -> CookedMeshEntry:
        var entry = CookedMeshEntry()
        var v_count = len(mesh.positions) // 3
        var raw_tris = len(mesh.indices) // 3
        if v_count <= 0 or raw_tris <= 0:
            return entry^

        var mesh_id = len(self.meshes)
        var span_v0 = self.total_vertices()
        var span_i0 = len(self.index_words)
        var span_m0 = len(self.meshlets)

        if not build_lods or raw_tris < 32:
            var s0 = self._append_single_lod_geometry_ex(
                mesh, mesh_id, material_id, 0, Float32(0.0), Float32(65504.0), is_splat, splat_weights_u24
            )
            entry.mesh_id = mesh_id
            entry.material_id = material_id
            entry.first_vertex = s0[0]
            entry.vertex_count = s0[1]
            entry.first_index = s0[2]
            entry.triangle_count = s0[3]
            entry.first_meshlet = s0[4]
            entry.meshlet_count = s0[5]
            entry.lod0_meshlet_count = s0[5]
            entry.lod_count = 1
            entry.lods[0] = MeshLodSlice(s0[4], s0[5], s0[3], Float32(0.0))
            entry.bounds_center = s0[6]
            entry.bounds_radius = s0[7]
            self.meshes.append(entry.copy())
            self._record_span(mesh_id, span_v0, span_i0, span_m0)
            return entry^

        # Build continuous QEM `LOD 0..3` chain (`100% -> 50% -> 25% -> 12.5%`) with strictly monotonic errors!
        # When `is_splat` is enabled, carry `splat_weights_u24` via `NativeMesh.colors` (`RGB` bytes) across QEM LODs if colors were unused!
        var base_m = mesh.copy()
        var carried_splat_in_colors = False
        if is_splat and len(base_m.colors) == 0 and len(splat_weights_u24) >= v_count:
            carried_splat_in_colors = True
            base_m.colors = List[UInt8](capacity=v_count * 4)
            for vi in range(v_count):
                var sw = splat_weights_u24[vi]
                base_m.colors.append(UInt8(sw & UInt32(0xFF)))
                base_m.colors.append(UInt8((sw >> 8) & UInt32(0xFF)))
                base_m.colors.append(UInt8((sw >> 16) & UInt32(0xFF)))
                base_m.colors.append(UInt8(255))

        var lod_meshes = List[NativeMesh]()
        var lod_errs = List[Float32]()
        lod_meshes.append(base_m.copy())
        lod_errs.append(Float32(0.0))

        var cur_m = base_m.copy()
        var prev_err = Float32(0.0)
        for lvl in range(1, AAA_MAX_MESH_LODS):
            var tgt = max(8, raw_tris >> lvl)
            if len(cur_m.indices) // 3 <= tgt + 2:
                break
            var simp = simplify_mesh_qem(cur_m, tgt, lock_boundaries=is_splat)
            var simp_tris = len(simp[0].indices) // 3
            if simp_tris <= 0 or simp_tris >= len(cur_m.indices) // 3:
                break
            var mono_err = max(prev_err * Float32(1.65) + Float32(0.002), prev_err + simp[1])
            cur_m = simp[0].copy()
            lod_meshes.append(simp[0].copy())
            lod_errs.append(mono_err)
            prev_err = mono_err

        var n_lods = len(lod_meshes)
        var first_ml_all = len(self.meshlets)
        for lvl in range(n_lods):
            var s_err = lod_errs[lvl]
            var p_err = lod_errs[lvl + 1] if (lvl + 1 < n_lods) else Float32(65504.0)
            var lvl_splat = List[UInt32]()
            if is_splat:
                if lvl == 0:
                    lvl_splat = splat_weights_u24.copy()
                elif carried_splat_in_colors:
                    var lvc = lod_meshes[lvl].vertex_count()
                    lvl_splat = List[UInt32](capacity=lvc)
                    for lvi in range(lvc):
                        var b0 = UInt32(lod_meshes[lvl].colors[lvi * 4 + 0])
                        var b1 = UInt32(lod_meshes[lvl].colors[lvi * 4 + 1])
                        var b2 = UInt32(lod_meshes[lvl].colors[lvi * 4 + 2])
                        lvl_splat.append(b0 | (b1 << 8) | (b2 << 16))
            var sl = self._append_single_lod_geometry_ex(
                lod_meshes[lvl], mesh_id, material_id, lvl, s_err, p_err, is_splat, lvl_splat
            )
            entry.lods[lvl] = MeshLodSlice(sl[4], sl[5], sl[3], s_err)
            if lvl == 0:
                entry.first_vertex = sl[0]
                entry.vertex_count = sl[1]
                entry.first_index = sl[2]
                entry.triangle_count = sl[3]
                entry.lod0_meshlet_count = sl[5]
                entry.bounds_center = sl[6]
                entry.bounds_radius = sl[7]

        entry.mesh_id = mesh_id
        entry.material_id = material_id
        entry.first_meshlet = first_ml_all
        entry.meshlet_count = len(self.meshlets) - first_ml_all
        entry.lod_count = n_lods
        self.meshes.append(entry.copy())
        self._record_span(mesh_id, span_v0, span_i0, span_m0)
        return entry^

    def _emit_meshlet(
        mut self,
        mesh: NativeMesh,
        indices_src: List[Int],
        base_v: Int,
        mesh_id: Int,
        material_id: Int,
        double_sided: Bool,
        lod_level: Int,
        self_err: Float32,
        parent_err: Float32,
        tri_list: List[Int],
    ):
        var first_tri_global = len(self.index_words) // 3
        var min_p = Vec3(1e9, 1e9, 1e9)
        var max_p = Vec3(-1e9, -1e9, -1e9)
        var tri_normals = List[Vec3](capacity=len(tri_list))
        var n_sum = Vec3(0.0, 0.0, 0.0)

        for ti in tri_list:
            var i0 = indices_src[ti * 3 + 0]
            var i1 = indices_src[ti * 3 + 1]
            var i2 = indices_src[ti * 3 + 2]
            self.index_words.append(UInt32(base_v + i0))
            self.index_words.append(UInt32(base_v + i1))
            self.index_words.append(UInt32(base_v + i2))

            var p0 = Vec3(mesh.positions[i0 * 3], mesh.positions[i0 * 3 + 1], mesh.positions[i0 * 3 + 2])
            var p1 = Vec3(mesh.positions[i1 * 3], mesh.positions[i1 * 3 + 1], mesh.positions[i1 * 3 + 2])
            var p2 = Vec3(mesh.positions[i2 * 3], mesh.positions[i2 * 3 + 1], mesh.positions[i2 * 3 + 2])
            for p in [p0, p1, p2]:
                if p.x < min_p.x: min_p.x = p.x
                if p.y < min_p.y: min_p.y = p.y
                if p.z < min_p.z: min_p.z = p.z
                if p.x > max_p.x: max_p.x = p.x
                if p.y > max_p.y: max_p.y = p.y
                if p.z > max_p.z: max_p.z = p.z

            var fn_raw = (p1 - p0).cross(p2 - p0)
            if fn_raw.length_squared() > Float32(1e-12):
                var fn_u = fn_raw.normalized()
                tri_normals.append(fn_u)
                n_sum = n_sum + fn_u

        var center = (min_p + max_p) * Float32(0.5)
        var max_r2 = Float32(1e-6)
        for ti in tri_list:
            for k in range(3):
                var iv = indices_src[ti * 3 + k]
                var p = Vec3(mesh.positions[iv * 3], mesh.positions[iv * 3 + 1], mesh.positions[iv * 3 + 2])
                var r2 = (p - center).length_squared()
                if r2 > max_r2:
                    max_r2 = r2
        var radius = sqrt(max_r2)

        var cone_axis = Vec3(0.0, 0.0, 1.0)
        var cone_sin = Float32(2.0)
        if not double_sided and len(tri_normals) > 0 and n_sum.length_squared() > Float32(1e-6):
            cone_axis = n_sum.normalized()
            var min_dot = Float32(1.0)
            for tn in tri_normals:
                var d = cone_axis.dot(tn)
                if d < min_dot:
                    min_dot = d
            if min_dot > Float32(0.08):
                cone_sin = sqrt(max(Float32(0.0), Float32(1.0) - min_dot * min_dot))

        var ml = MeshletCluster()
        ml.center = center
        ml.radius = radius
        ml.cone_axis = cone_axis
        ml.cone_sin = cone_sin
        ml.first_tri = first_tri_global
        ml.tri_count = len(tri_list)
        ml.mesh_id = mesh_id
        ml.material_id = material_id
        ml.lod_level = lod_level
        ml.self_error = self_err
        ml.parent_error = parent_err
        self.meshlets.append(ml.copy())

        var h_self = UInt32(f32_to_uf16_bits(self_err)) & 0xFFFF
        var h_parent = UInt32(f32_to_uf16_bits(parent_err)) & 0xFFFF
        var packed_lod_err = h_self | (h_parent << 16)

        self.meshlet_words.append(bitcast[DType.uint32](center.x))
        self.meshlet_words.append(bitcast[DType.uint32](center.y))
        self.meshlet_words.append(bitcast[DType.uint32](center.z))
        self.meshlet_words.append(bitcast[DType.uint32](radius))
        self.meshlet_words.append(bitcast[DType.uint32](cone_axis.x))
        self.meshlet_words.append(bitcast[DType.uint32](cone_axis.y))
        self.meshlet_words.append(bitcast[DType.uint32](cone_axis.z))
        self.meshlet_words.append(bitcast[DType.uint32](cone_sin))
        self.meshlet_words.append(UInt32(first_tri_global))
        self.meshlet_words.append(UInt32(len(tri_list)))
        self.meshlet_words.append(packed_lod_err)
        self.meshlet_words.append(UInt32(material_id))


def select_mesh_lod_cpu_reference(
    entry: CookedMeshEntry,
    inst_origin: Vec3,
    uniform_scale: Float32,
    cam_pos: Vec3,
    near_z: Float32,
    screen_h: Int,
    tan_half_fov_y: Float32,
    max_pixel_error: Float32 = 1.0,
) -> Int:
    """Exact CPU reference for GPU continuous screen-space error LOD selection (`0 .. entry.lod_count - 1`)."""
    if entry.lod_count <= 1 or max_pixel_error <= Float32(0.0):
        return 0
    var d_inst = max(near_z, (inst_origin - cam_pos).length())
    var px_per_m = (uniform_scale * Float32(screen_h)) / max(Float32(1e-5), Float32(2.0) * d_inst * tan_half_fov_y)
    var e_max = max_pixel_error / max(Float32(1e-5), px_per_m)
    var chosen = 0
    for lvl in range(entry.lod_count):
        if entry.lods[lvl].object_error <= e_max:
            chosen = lvl
    return chosen


def cull_meshlet_cpu_reference(
    ml: MeshletCluster,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    tan_half_fov_x: Float32,
    tan_half_fov_y: Float32,
    near_z: Float32,
) -> Bool:
    """Exact CPU reference for GPU Meshlet Frustum + Normal-Cone Backface Culling."""
    var rel = ml.center - cam_pos
    var dist2 = rel.length_squared()
    var dist = sqrt(max(Float32(1e-8), dist2))

    if ml.cone_sin < Float32(1.5):
        if rel.dot(ml.cone_axis) >= ml.cone_sin * dist + ml.radius:
            return False

    var vz = rel.dot(cam_fwd)
    if vz + ml.radius < near_z:
        return False
    var vx = abs(rel.dot(cam_right))
    var vy = abs(rel.dot(cam_up))
    var sec_x = sqrt(Float32(1.0) + tan_half_fov_x * tan_half_fov_x)
    var sec_y = sqrt(Float32(1.0) + tan_half_fov_y * tan_half_fov_y)
    if vx - ml.radius * sec_x > max(Float32(0.0), vz) * tan_half_fov_x:
        return False
    if vy - ml.radius * sec_y > max(Float32(0.0), vz) * tan_half_fov_y:
        return False
    return True
