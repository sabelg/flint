"""LOD chains and meshlet clusters for Flint's geometry kit (`flint.assets.mesh_lod`).

  - `build_lod_chain`: a chain of QEM edge-collapse LODs (Garland & Heckbert 1997 quadrics with
    boundary-plane penalties, seam-preserving attribute interpolation of normals / UVs / colours and
    top-4 renormalised skin weights; see `flint.assets.meshlet.simplify_mesh_qem`). Each level is
    simplified from the previous one, measured against LOD 0 with a sampled symmetric Hausdorff
    distance, and the chain stops early once a level would exceed the error budget.
  - `build_meshlets`: greedy adjacency-driven meshlet builder after Kapoulkine's meshoptimizer
    `buildMeshlets` (published algorithm, re-implemented): grow each cluster through triangles that
    share its vertices, preferring candidates that add the fewest new vertices, then the ones whose
    vertices have the fewest remaining triangles (so no orphans are left behind), then the closest
    to the cluster's centroid and normal cone. When a cluster has no adjacent candidate it restarts
    from the nearest unassigned triangle in Morton order. Each meshlet carries a bounding sphere and
    a backface-culling normal cone (apex / axis / cutoff).
"""
from std.math import abs, max, min, sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.meshlet import MESHLET_MAX_TRIS, MESHLET_MAX_VERTS, simplify_mesh_qem
from flint.assets.mesh_kit import unit, EdgeMap, HausdorffResult, hausdorff_distance, vpos, mesh_bounds


# ------------------------------------------------------------------------------- LOD chain

struct LodLevel(Copyable, Movable):
    var mesh: NativeMesh
    var target_ratio: Float32
    var qem_error: Float32
    var hausdorff: HausdorffResult

    def __init__(out self, var mesh: NativeMesh, target_ratio: Float32, qem_error: Float32, h: HausdorffResult):
        self.mesh = mesh^
        self.target_ratio = target_ratio
        self.qem_error = qem_error
        self.hausdorff = h


struct LodConfig(Copyable, Movable):
    """`ratios` are triangle-count fractions of LOD 0 (first entry 1.0 = the source).
    `max_relative_error` stops the chain when a level's Hausdorff error over the bounding-box
    diagonal exceeds it. `measure_samples` = 0 skips Hausdorff measurement (fast cooking)."""
    var ratios: List[Float32]
    var lock_boundaries: Bool
    var max_relative_error: Float32
    var measure_samples: Int
    var min_triangles: Int

    def __init__(out self):
        self.ratios = [Float32(1.0), 0.5, 0.25, 0.125, 0.0625]
        self.lock_boundaries = False
        self.max_relative_error = 0.05
        self.measure_samples = 20000
        self.min_triangles = 32


def build_lod_chain(mesh: NativeMesh, cfg: LodConfig) -> List[LodLevel]:
    var out = List[LodLevel]()
    var base_tris = mesh.triangle_count()
    var zero = HausdorffResult(0, 0, 0, 0, 0)
    out.append(LodLevel(mesh.copy(), 1.0, 0.0, zero))
    var prev = mesh.copy()
    for i in range(len(cfg.ratios)):
        var r = cfg.ratios[i]
        if r >= 0.999:
            continue
        var target = max(cfg.min_triangles, Int(Float32(base_tris) * r))
        if target >= prev.triangle_count():
            continue
        var res = simplify_mesh_qem(prev, target, cfg.lock_boundaries)
        var lod = res[0].copy()
        if lod.triangle_count() >= prev.triangle_count():
            break
        var h = zero
        if cfg.measure_samples > 0:
            h = hausdorff_distance(mesh, lod, cfg.measure_samples)
            if h.relative() > cfg.max_relative_error:
                break
        prev = lod.copy()
        out.append(LodLevel(lod^, r, res[1], h))
    return out^


# ------------------------------------------------------------------------------- meshlets

@fieldwise_init
struct MeshletInfo(Copyable, ImplicitlyCopyable, Movable):
    var vertex_offset: Int
    var vertex_count: Int
    var triangle_offset: Int   # into `MeshletSet.local_indices`, in units of indices (3 per tri)
    var triangle_count: Int
    var center: Vec3
    var radius: Float32
    var cone_apex: Vec3
    var cone_axis: Vec3
    var cone_cutoff: Float32   # cull when dot(normalize(apex - eye), axis) >= cutoff


struct MeshletSet(Movable):
    var meshlets: List[MeshletInfo]
    var vertices: List[Int]         # global vertex ids per meshlet (concatenated)
    var local_indices: List[UInt8]  # 3 local ids per triangle (concatenated)
    var max_vertices: Int
    var max_triangles: Int

    def __init__(out self, max_vertices: Int, max_triangles: Int):
        self.meshlets = List[MeshletInfo]()
        self.vertices = List[Int]()
        self.local_indices = List[UInt8]()
        self.max_vertices = max_vertices
        self.max_triangles = max_triangles

    def count(self) -> Int:
        return len(self.meshlets)

    def triangle_total(self) -> Int:
        return len(self.local_indices) // 3

    def vertex_refs_per_triangle(self) -> Float32:
        """Meshlet vertex references per triangle (lower = better reuse; ideal ~0.5 on grids)."""
        return Float32(len(self.vertices)) / max(Float32(1), Float32(self.triangle_total()))

    def fill_ratio(self) -> Float32:
        return Float32(self.triangle_total()) / max(Float32(1), Float32(self.count() * self.max_triangles))


@always_inline
def _part1by2(v: Int) -> UInt32:
    var x = UInt32(v & 0x3FF)
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def _morton_order(mesh: NativeMesh) -> List[Int]:
    var nt = mesh.triangle_count()
    var b = mesh_bounds(mesh)
    var ext = b[1] - b[0]
    var s = 1023.0 / max(Float32(1e-9), max(ext.x, max(ext.y, ext.z)))
    var keys = List[UInt32](length=nt, fill=0)
    for t in range(nt):
        var c = (vpos(mesh, mesh.indices[t * 3]) + vpos(mesh, mesh.indices[t * 3 + 1]) + vpos(mesh, mesh.indices[t * 3 + 2])) * (1.0 / 3.0)
        var q = (c - b[0]) * s
        keys[t] = _part1by2(Int(q.x)) | (_part1by2(Int(q.y)) << 1) | (_part1by2(Int(q.z)) << 2)
    # LSD radix sort, 3 passes of 10 bits
    var order = List[Int](length=nt, fill=0)
    for t in range(nt):
        order[t] = t
    var tmp = List[Int](length=nt, fill=0)
    for p in range(3):
        var shift = UInt32(p * 10)
        var hist = List[Int](length=1025, fill=0)
        for i in range(nt):
            hist[Int((keys[order[i]] >> shift) & 1023) + 1] += 1
        for i in range(1024):
            hist[i + 1] += hist[i]
        for i in range(nt):
            var k = Int((keys[order[i]] >> shift) & 1023)
            tmp[hist[k]] = order[i]
            hist[k] += 1
        for i in range(nt):
            order[i] = tmp[i]
    return order^


def _tri_normal(mesh: NativeMesh, t: Int) -> Vec3:
    var a = vpos(mesh, mesh.indices[t * 3])
    return unit((vpos(mesh, mesh.indices[t * 3 + 1]) - a).cross(vpos(mesh, mesh.indices[t * 3 + 2]) - a))


def _tri_center(mesh: NativeMesh, t: Int) -> Vec3:
    return (vpos(mesh, mesh.indices[t * 3]) + vpos(mesh, mesh.indices[t * 3 + 1]) + vpos(mesh, mesh.indices[t * 3 + 2])) * (1.0 / 3.0)


def build_meshlets(mesh: NativeMesh, max_vertices: Int = MESHLET_MAX_VERTS, max_triangles: Int = MESHLET_MAX_TRIS,
                   cone_weight: Float32 = 0.25) -> MeshletSet:
    var nv = mesh.vertex_count()
    var nt = mesh.triangle_count()
    var out = MeshletSet(max_vertices, max_triangles)
    if nt == 0:
        return out^
    # vertex -> triangle CSR adjacency and live counts
    var live = List[Int](length=nv, fill=0)
    for i in range(nt * 3):
        live[mesh.indices[i]] += 1
    var off = List[Int](length=nv + 1, fill=0)
    for v in range(nv):
        off[v + 1] = off[v] + live[v]
    var adj = List[Int](length=off[nv], fill=0)
    var cur = off.copy()
    for t in range(nt):
        for k in range(3):
            var v = mesh.indices[t * 3 + k]
            adj[cur[v]] = t
            cur[v] += 1
    var emitted = List[Bool](length=nt, fill=False)
    var local_id = List[Int](length=nv, fill=-1)   # vertex -> slot in the current meshlet
    var morton = _morton_order(mesh)
    var morton_pos = 0
    var tri_n = List[Vec3](capacity=nt)
    for t in range(nt):
        tri_n.append(_tri_normal(mesh, t))

    var m_verts = List[Int](capacity=max_vertices)
    var m_tris = List[Int](capacity=max_triangles)
    var center_sum = Vec3(0, 0, 0)
    var normal_sum = Vec3(0, 0, 0)
    var remaining = nt
    var b = mesh_bounds(mesh)
    var diag = max(Float32(1e-9), (b[1] - b[0]).length())

    while remaining > 0:
        # choose the next triangle: best adjacent candidate, else nearest unassigned in Morton order
        var best = -1
        if len(m_tris) > 0:
            var best_extra = 4
            var best_live = 1 << 30
            var best_score = Float32(3.0e38)
            var c = center_sum * (1.0 / Float32(len(m_tris)))
            var axis = unit(normal_sum)
            for mv in range(len(m_verts)):
                var v = m_verts[mv]
                for s in range(off[v], off[v + 1]):
                    var t = adj[s]
                    if emitted[t]:
                        continue
                    var extra = 0
                    var lv = 0
                    for k in range(3):
                        var w = mesh.indices[t * 3 + k]
                        if local_id[w] < 0:
                            extra += 1
                        lv += live[w]
                    if len(m_verts) + extra > max_vertices:
                        continue
                    # topological priority: fewer new vertices, then triangles that close off vertices
                    var prio = extra
                    if extra > 0 and lv <= 3 + extra:
                        prio = 0
                    var d = (_tri_center(mesh, t) - c).length() / diag
                    var spread = 1.0 - axis.dot(tri_n[t]) * cone_weight
                    var score = d * spread
                    if prio < best_extra or (prio == best_extra and (lv < best_live or (lv == best_live and score < best_score))):
                        best_extra = prio
                        best_live = lv
                        best_score = score
                        best = t
        if best < 0 and len(m_tris) > 0:
            # cluster is stuck: close it and seed a new one
            _flush_meshlet(mesh, out, m_verts, m_tris, local_id, tri_n)
            center_sum = Vec3(0, 0, 0)
            normal_sum = Vec3(0, 0, 0)
            continue
        if best < 0:
            while emitted[morton[morton_pos]]:
                morton_pos += 1
            best = morton[morton_pos]
        # append best (flush first when it does not fit)
        var extra2 = 0
        for k in range(3):
            if local_id[mesh.indices[best * 3 + k]] < 0:
                extra2 += 1
        if len(m_verts) + extra2 > max_vertices or len(m_tris) + 1 > max_triangles:
            _flush_meshlet(mesh, out, m_verts, m_tris, local_id, tri_n)
            center_sum = Vec3(0, 0, 0)
            normal_sum = Vec3(0, 0, 0)
        for k in range(3):
            var w = mesh.indices[best * 3 + k]
            if local_id[w] < 0:
                local_id[w] = len(m_verts)
                m_verts.append(w)
            live[w] -= 1
        m_tris.append(best)
        emitted[best] = True
        remaining -= 1
        center_sum = center_sum + _tri_center(mesh, best)
        normal_sum = normal_sum + tri_n[best]
        if len(m_tris) == max_triangles:
            _flush_meshlet(mesh, out, m_verts, m_tris, local_id, tri_n)
            center_sum = Vec3(0, 0, 0)
            normal_sum = Vec3(0, 0, 0)
    if len(m_tris) > 0:
        _flush_meshlet(mesh, out, m_verts, m_tris, local_id, tri_n)
    return out^


def _flush_meshlet(mesh: NativeMesh, mut out: MeshletSet, mut m_verts: List[Int], mut m_tris: List[Int],
                   mut local_id: List[Int], tri_n: List[Vec3]):
    var vo = len(out.vertices)
    var to = len(out.local_indices)
    for i in range(len(m_verts)):
        out.vertices.append(m_verts[i])
    for i in range(len(m_tris)):
        var t = m_tris[i]
        for k in range(3):
            out.local_indices.append(UInt8(local_id[mesh.indices[t * 3 + k]]))
    # bounding sphere: Ritter-style from the AABB centre, then exact radius
    var lo = vpos(mesh, m_verts[0])
    var hi = lo
    for i in range(len(m_verts)):
        var p = vpos(mesh, m_verts[i])
        lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
        hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
    var center = (lo + hi) * 0.5
    var radius: Float32 = 0
    for i in range(len(m_verts)):
        radius = max(radius, (vpos(mesh, m_verts[i]) - center).length())
    # normal cone (meshoptimizer `computeMeshletBounds` formulation)
    var axis = Vec3(0, 0, 0)
    for i in range(len(m_tris)):
        axis = axis + tri_n[m_tris[i]]
    axis = unit(axis)
    var mindp: Float32 = 1
    for i in range(len(m_tris)):
        mindp = min(mindp, axis.dot(tri_n[m_tris[i]]))
    var apex = center
    var cutoff: Float32 = 1
    if mindp > 0.0 and axis.length_squared() > 0.5:
        # apex: move back along -axis so every triangle plane is in front of it
        var maxt: Float32 = 0
        for i in range(len(m_tris)):
            var t = m_tris[i]
            var n = tri_n[t]
            var p0 = vpos(mesh, mesh.indices[t * 3])
            var dc = (center - p0).dot(n)
            var dn = axis.dot(n)
            if dn > 1e-6:
                maxt = max(maxt, dc / dn)
        apex = center - axis * maxt
        cutoff = sqrt(max(Float32(0), 1.0 - mindp * mindp))
    out.meshlets.append(MeshletInfo(vo, len(m_verts), to, len(m_tris), center, radius, apex, axis, cutoff))
    for i in range(len(m_verts)):
        local_id[m_verts[i]] = -1
    m_verts.clear()
    m_tris.clear()


def meshlet_cone_culls(m: MeshletInfo, eye: Vec3) -> Bool:
    """True when every triangle of the meshlet faces away from `eye`."""
    if m.cone_cutoff >= 1.0:
        return False
    var d = (m.cone_apex - eye)
    var l = d.length()
    if l < 1e-9:
        return False
    return d.dot(m.cone_axis) / l >= m.cone_cutoff


def validate_meshlets(mesh: NativeMesh, ms: MeshletSet) -> Bool:
    """Every source triangle appears exactly once and every meshlet respects its limits."""
    var nt = mesh.triangle_count()
    if ms.triangle_total() != nt:
        return False
    # multiset of sorted-vertex triangle keys must match the source
    var src_keys = List[UInt64](capacity=nt)
    for t in range(nt):
        src_keys.append(_tri_key(mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]))
    var counts = List[Int]()
    var key_map = EdgeMap(nt)
    for t in range(nt):
        var slot = key_map.insert(src_keys[t], len(counts))
        if slot == len(counts):
            counts.append(0)
        counts[slot] += 1
    for mi in range(ms.count()):
        var m = ms.meshlets[mi]
        if m.vertex_count > ms.max_vertices or m.triangle_count > ms.max_triangles:
            return False
        for t in range(m.triangle_count):
            var a = ms.vertices[m.vertex_offset + Int(ms.local_indices[m.triangle_offset + t * 3])]
            var b = ms.vertices[m.vertex_offset + Int(ms.local_indices[m.triangle_offset + t * 3 + 1])]
            var c = ms.vertices[m.vertex_offset + Int(ms.local_indices[m.triangle_offset + t * 3 + 2])]
            var slot = key_map.find(_tri_key(a, b, c))
            if slot < 0:
                return False
            counts[slot] -= 1
            # sphere must contain the triangle
            for v in [a, b, c]:
                if (vpos(mesh, v) - m.center).length() > m.radius * 1.0001 + 1e-6:
                    return False
    for i in range(len(counts)):
        if counts[i] != 0:
            return False
    return True


@always_inline
def _tri_key(a: Int, b: Int, c: Int) -> UInt64:
    # rotation-invariant (keeps winding): start at the smallest index
    var x = a
    var y = b
    var z = c
    if b < x and b < c:
        x = b; y = c; z = a
    elif c < x and c < b:
        x = c; y = a; z = b
    return (UInt64(x & 0x1FFFFF) << 42) | (UInt64(y & 0x1FFFFF) << 21) | UInt64(z & 0x1FFFFF)
