"""Sota:geometry — offline continuous cluster-LOD DAG (Nanite-style) + CPU reference LOD cut.

Build (per mesh, offline / at load):
  1. Weld wedges by exact position (topology ids, `pid`), keep the original wedge vertex buffer:
     every DAG level indexes the SAME vertices (half-edge collapses never invent positions).
  2. Clusterize LOD0 triangles into <= 64-vertex / <= 124-triangle clusters by adjacency-driven region growing
     seeded in Morton order (meshoptimizer `buildMeshlets` idea, simplified).
  3. Repeat per level: partition clusters into groups of ~8 by shared-border-edge count (greedy, METIS-lite);
     merge each group's triangles; QEM half-edge-collapse simplify to 50 % with the group border LOCKED
     (any vertex also used by another group, any edge open inside the group, any UV/normal seam);
     re-split the simplified triangles into clusters. Group error = max(child error) + simplify error,
     group sphere = union of child spheres, so errors and spheres are monotonic up the DAG.
  4. Groups that do not reach 85 % reduction are carried to the next round; when a round makes no progress the
     remaining clusters become roots (`parent_error = +inf`).

Runtime cut (identical on CPU here and on the GPU in `sota_geometry_cull.mojo`):
  draw cluster  <=>  proj(self_err, self_sphere) <= t  AND  proj(parent_err, parent_sphere) > t
  with proj(e, c, r) = e * (screen_h / (2 tan(fov_y/2))) / max(|c - eye| - r, near).
  Siblings share the same parent sphere/error bit-for-bit, so a group switches atomically and the locked
  borders guarantee a watertight cut at any mix of levels.

Sources (read for learning, not copied): Karis et al., "Nanite: A Deep Dive" (SIGGRAPH 2021 Advances);
meshoptimizer `demo/clusterlod.h` (MIT, Arseny Kapoulkine); Garland & Heckbert 1997 (QEM).
"""
from std.collections import List
from std.math import floor, max, min, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.detsort import stable_order
from flint.assets.model import NativeMesh
from flint.assets.meshlet import AaaGeometryPool, CookedMeshEntry, MeshLodSlice

comptime SOTA_DAG_MAX_VERTS: Int = 64
comptime SOTA_DAG_MAX_TRIS: Int = 124
comptime SOTA_DAG_GROUP_SIZE: Int = 8
comptime SOTA_DAG_RECORD_WORDS: Int = 12
comptime SOTA_DAG_INF_ERROR: Float32 = 1.0e30
comptime SOTA_DAG_MAX_LEVELS: Int = 24

# GPU DAG record (12 words per global meshlet id, `sota_geometry_cull.mojo` binding SOTA_BIND_DAG):
# [0..3]  self LOD sphere (mesh-local centre xyz, radius)    [4] self error (mesh-local metres)
# [5..8]  parent LOD sphere (centre xyz, radius)             [9] parent error (1e30 = root)
# [10]    flags: bit0 = DAG cluster (0 = legacy discrete-LOD meshlet)   [11] DAG level


struct DagCluster(Copyable, Movable):
    var tris: List[Int]          # wedge-index triplets
    var level: Int
    var self_center: Vec3
    var self_radius: Float32
    var self_error: Float32
    var parent_center: Vec3
    var parent_radius: Float32
    var parent_error: Float32

    def __init__(out self, var tris: List[Int], level: Int):
        self.tris = tris^
        self.level = level
        self.self_center = Vec3()
        self.self_radius = Float32(0.0)
        self.self_error = Float32(0.0)
        self.parent_center = Vec3()
        self.parent_radius = Float32(0.0)
        self.parent_error = SOTA_DAG_INF_ERROR

    def tri_count(self) -> Int:
        return len(self.tris) // 3


struct ClusterDag(Movable):
    var clusters: List[DagCluster]
    var levels: Int
    var lod0_clusters: Int
    var lod0_tris: Int
    var groups_built: Int
    var groups_failed: Int

    def __init__(out self):
        self.clusters = List[DagCluster]()
        self.levels = 0
        self.lod0_clusters = 0
        self.lod0_tris = 0
        self.groups_built = 0
        self.groups_failed = 0

    def total_tris(self) -> Int:
        var n = 0
        for c in self.clusters:
            n += c.tri_count()
        return n


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def _pos(mesh: NativeMesh, v: Int) -> Vec3:
    return Vec3(mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2])


def _nrm(mesh: NativeMesh, v: Int) -> Vec3:
    if len(mesh.normals) >= (v + 1) * 3:
        return Vec3(mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2])
    return Vec3(0.0, 1.0, 0.0)


def _part1by2(v: Int) -> Int:
    var x = v & 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def _morton_keys(points: List[Vec3]) -> List[Int]:
    var keys = List[Int](capacity=len(points))
    if len(points) == 0:
        return keys^
    var lo = points[0]
    var hi = points[0]
    for p in points:
        lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
        hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
    var ext = max(Float32(1e-6), max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z)))
    var s = Float32(1023.0) / ext
    for p in points:
        var qx = Int((p.x - lo.x) * s)
        var qy = Int((p.y - lo.y) * s)
        var qz = Int((p.z - lo.z) * s)
        keys.append(_part1by2(qx) | (_part1by2(qy) << 1) | (_part1by2(qz) << 2))
    return keys^


def sphere_merge(c0: Vec3, r0: Float32, c1: Vec3, r1: Float32) -> Tuple[Vec3, Float32]:
    """Smallest sphere containing both spheres (exact for two spheres)."""
    var d = (c1 - c0).length()
    if d + r1 <= r0:
        return (c0, r0)
    if d + r0 <= r1:
        return (c1, r1)
    var r = (d + r0 + r1) * Float32(0.5)
    var t = (r - r0) / max(Float32(1e-20), d)
    var c = c0 + (c1 - c0) * t
    # Float rounding: grow a hair so containment holds bit-exactly for the monotonic LOD test.
    return (c, r * Float32(1.000001) + Float32(1e-7))


def _tri_bounds(mesh: NativeMesh, tris: List[Int]) -> Tuple[Vec3, Float32]:
    var lo = Vec3(1e30, 1e30, 1e30)
    var hi = Vec3(-1e30, -1e30, -1e30)
    for v in tris:
        var p = _pos(mesh, v)
        lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
        hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
    var c = (lo + hi) * Float32(0.5)
    var r2 = Float32(0.0)
    for v in tris:
        r2 = max(r2, (_pos(mesh, v) - c).length_squared())
    return (c, sqrt(r2) * Float32(1.000001) + Float32(1e-7))


def weld_positions(mesh: NativeMesh) -> Tuple[List[Int], Int]:
    """Topology id per wedge: wedges with bit-identical positions share an id."""
    var n = mesh.vertex_count()
    var keys = List[Int](capacity=n)
    for v in range(n):
        var bx = Int(bitcast[DType.uint32](mesh.positions[v * 3]))
        var by = Int(bitcast[DType.uint32](mesh.positions[v * 3 + 1]))
        var bz = Int(bitcast[DType.uint32](mesh.positions[v * 3 + 2]))
        var h = (bx * 73856093) ^ (by * 19349663) ^ (bz * 83492791)
        keys.append(h & 0x3FFFFFFFFFFFFFFF)
    var order = stable_order(keys)
    var pid = List[Int](length=n, fill=-1)
    var next_id = 0
    var i = 0
    while i < n:
        var j = i
        while j < n and keys[order[j]] == keys[order[i]]:
            j += 1
        # exact compare inside the hash run
        for a in range(i, j):
            var va = order[a]
            if pid[va] >= 0:
                continue
            pid[va] = next_id
            for b in range(a + 1, j):
                var vb = order[b]
                if pid[vb] < 0 and mesh.positions[va * 3] == mesh.positions[vb * 3] and mesh.positions[va * 3 + 1] == mesh.positions[vb * 3 + 1] and mesh.positions[va * 3 + 2] == mesh.positions[vb * 3 + 2]:
                    pid[vb] = next_id
            next_id += 1
        i = j
    return (pid^, next_id)


def _edge_key(a: Int, b: Int, n: Int) -> Int:
    return (min(a, b) * n + max(a, b))


# ---------------------------------------------------------------------------------------------
# Clusterization: adjacency region growing, <= 64 verts / <= 124 tris
# ---------------------------------------------------------------------------------------------
def clusterize_triangles(
    mesh: NativeMesh, pid: List[Int], n_pid: Int, tris: List[Int], mut vstamp: List[Int], mut stamp_base: Int
) -> List[List[Int]]:
    """Splits a wedge-triplet list into clusters (each a wedge-triplet list). `vstamp` is a scratch array
    sized to the wedge count; `stamp_base` is advanced so stamps never collide between calls."""
    var out = List[List[Int]]()
    var nt = len(tris) // 3
    if nt == 0:
        return out^
    # triangle adjacency through shared pid edges
    var ekeys = List[Int](capacity=nt * 3)
    for t in range(nt):
        for k in range(3):
            var a = pid[tris[t * 3 + k]]
            var b = pid[tris[t * 3 + (k + 1) % 3]]
            ekeys.append(_edge_key(a, b, n_pid))
    var eorder = stable_order(ekeys)
    var adj_cnt = List[Int](length=nt + 1, fill=0)
    var i = 0
    var ne = len(ekeys)
    while i < ne:
        var j = i
        while j < ne and ekeys[eorder[j]] == ekeys[eorder[i]]:
            j += 1
        if j - i >= 2:
            for a in range(i, j):
                adj_cnt[eorder[a] // 3] += j - i - 1
        i = j
    var adj_off = List[Int](length=nt + 1, fill=0)
    for t in range(nt):
        adj_off[t + 1] = adj_off[t] + adj_cnt[t]
    var adj = List[Int](length=adj_off[nt], fill=0)
    var fillp = adj_off.copy()
    i = 0
    while i < ne:
        var j = i
        while j < ne and ekeys[eorder[j]] == ekeys[eorder[i]]:
            j += 1
        if j - i >= 2:
            for a in range(i, j):
                var ta = eorder[a] // 3
                for b in range(i, j):
                    if a != b:
                        adj[fillp[ta]] = eorder[b] // 3
                        fillp[ta] += 1
        i = j

    var cents = List[Vec3](capacity=nt)
    for t in range(nt):
        cents.append((_pos(mesh, tris[t * 3]) + _pos(mesh, tris[t * 3 + 1]) + _pos(mesh, tris[t * 3 + 2])) * Float32(1.0 / 3.0))
    var morder = stable_order(_morton_keys(cents))
    var assigned = List[Bool](length=nt, fill=False)
    var cursor = 0
    var frontier = List[Int]()
    while True:
        while cursor < nt and assigned[morder[cursor]]:
            cursor += 1
        if cursor >= nt:
            break
        stamp_base += 1
        var cid = stamp_base
        var ctris = List[Int]()
        var nverts = 0
        var csum = Vec3()
        frontier.clear()
        var seed = morder[cursor]
        frontier.append(seed)
        while len(ctris) // 3 < SOTA_DAG_MAX_TRIS:
            # pick best frontier triangle: fewest new vertices, then nearest to the cluster centroid
            var best = -1
            var best_new = 4
            var best_d = Float32(1e30)
            var cc = csum * (Float32(1.0) / Float32(max(1, len(ctris) // 3)))
            var w = 0
            for fi in range(len(frontier)):
                var t = frontier[fi]
                if assigned[t]:
                    continue
                frontier[w] = t
                w += 1
                var nn = 0
                for k in range(3):
                    var v = tris[t * 3 + k]
                    if vstamp[v] != cid:
                        nn += 1
                # duplicates inside the tri (degenerate wedges) are rare; over-count is conservative
                var d = (cents[t] - cc).length_squared() if len(ctris) > 0 else Float32(0.0)
                if nn < best_new or (nn == best_new and d < best_d):
                    best = t
                    best_new = nn
                    best_d = d
            while len(frontier) > w:
                _ = frontier.pop()
            if best < 0:
                # disconnected: continue with the next unassigned triangle in Morton order (spatially close)
                if len(ctris) // 3 >= SOTA_DAG_MAX_TRIS // 2:
                    break
                var c2 = cursor
                while c2 < nt and assigned[morder[c2]]:
                    c2 += 1
                if c2 >= nt:
                    break
                best = morder[c2]
                best_new = 0
                for k in range(3):
                    if vstamp[tris[best * 3 + k]] != cid:
                        best_new += 1
            if nverts + best_new > SOTA_DAG_MAX_VERTS:
                break
            assigned[best] = True
            for k in range(3):
                var v = tris[best * 3 + k]
                if vstamp[v] != cid:
                    vstamp[v] = cid
                    nverts += 1
                ctris.append(v)
            csum = csum + cents[best]
            for a in range(adj_off[best], adj_off[best + 1]):
                if not assigned[adj[a]]:
                    frontier.append(adj[a])
        out.append(ctris^)
    return out^


# ---------------------------------------------------------------------------------------------
# Group partition: clusters sharing the most border edges go together
# ---------------------------------------------------------------------------------------------
def partition_clusters(
    mesh: NativeMesh, pid: List[Int], n_pid: Int, clusters: List[DagCluster], ids: List[Int], group_size: Int
) -> List[List[Int]]:
    var nc = len(ids)
    var groups = List[List[Int]]()
    if nc == 0:
        return groups^
    # border edges per cluster -> (edge, cluster) pairs
    var pkeys = List[Int]()
    var powner = List[Int]()
    for li in range(nc):
        ref c = clusters[ids[li]]
        var ek = List[Int](capacity=len(c.tris))
        for t in range(c.tri_count()):
            for k in range(3):
                ek.append(_edge_key(pid[c.tris[t * 3 + k]], pid[c.tris[t * 3 + (k + 1) % 3]], n_pid))
        var eo = stable_order(ek)
        var i = 0
        while i < len(ek):
            var j = i
            while j < len(ek) and ek[eo[j]] == ek[eo[i]]:
                j += 1
            if j - i == 1:
                pkeys.append(ek[eo[i]])
                powner.append(li)
            i = j
    var po = stable_order(pkeys)
    # neighbour pairs (li * nc + lj) with weights
    var pair_keys = List[Int]()
    var i = 0
    while i < len(pkeys):
        var j = i
        while j < len(pkeys) and pkeys[po[j]] == pkeys[po[i]]:
            j += 1
        for a in range(i, j):
            for b in range(i, j):
                var ca = powner[po[a]]
                var cb = powner[po[b]]
                if ca != cb:
                    pair_keys.append(ca * nc + cb)
        i = j
    var pairo = stable_order(pair_keys)
    var nb_cnt = List[Int](length=nc + 1, fill=0)
    var nb_list = List[Int]()
    var nb_w = List[Int]()
    var nb_owner = List[Int]()
    i = 0
    while i < len(pair_keys):
        var j = i
        while j < len(pair_keys) and pair_keys[pairo[j]] == pair_keys[pairo[i]]:
            j += 1
        var k = pair_keys[pairo[i]]
        nb_owner.append(k // nc)
        nb_list.append(k % nc)
        nb_w.append(j - i)
        nb_cnt[k // nc] += 1
        i = j
    var nb_off = List[Int](length=nc + 1, fill=0)
    for c in range(nc):
        nb_off[c + 1] = nb_off[c] + nb_cnt[c]
    # pairs were sorted by owner, so nb_* are already grouped per owner in order

    var cents = List[Vec3](capacity=nc)
    for li in range(nc):
        cents.append(clusters[ids[li]].self_center)
    var morder = stable_order(_morton_keys(cents))
    var gid = List[Int](length=nc, fill=-1)
    var score = List[Int](length=nc, fill=0)
    for mi in range(nc):
        var seed = morder[mi]
        if gid[seed] >= 0:
            continue
        var g = len(groups)
        var members = List[Int]()
        members.append(seed)
        gid[seed] = g
        var touched = List[Int]()
        while len(members) < group_size:
            # accumulate shared-edge weight from the newest member
            var last = members[len(members) - 1]
            for a in range(nb_off[last], nb_off[last + 1]):
                var nbr = nb_list[a]
                if gid[nbr] < 0:
                    if score[nbr] == 0:
                        touched.append(nbr)
                    score[nbr] += nb_w[a]
            var best = -1
            var best_s = 0
            var best_d = Float32(1e30)
            for t in touched:
                if gid[t] >= 0 or score[t] <= 0:
                    continue
                var d = (cents[t] - cents[seed]).length_squared()
                if score[t] > best_s or (score[t] == best_s and d < best_d):
                    best = t
                    best_s = score[t]
                    best_d = d
            if best < 0:
                break
            gid[best] = g
            members.append(best)
        for t in touched:
            score[t] = 0
        var gm = List[Int](capacity=len(members))
        for m in members:
            gm.append(ids[m])
        groups.append(gm^)
    _ = nb_owner^
    return groups^


# ---------------------------------------------------------------------------------------------
# Locked-border QEM half-edge-collapse simplifier (positions never move: all levels share vertices)
# ---------------------------------------------------------------------------------------------
struct _Quadric(Copyable, ImplicitlyCopyable, Movable):
    var a: Float64
    var b: Float64
    var c: Float64
    var d: Float64
    var e: Float64
    var f: Float64
    var g: Float64
    var h: Float64
    var i: Float64
    var j: Float64
    var w: Float64

    def __init__(out self):
        self.a = 0.0; self.b = 0.0; self.c = 0.0; self.d = 0.0; self.e = 0.0
        self.f = 0.0; self.g = 0.0; self.h = 0.0; self.i = 0.0; self.j = 0.0; self.w = 0.0

    def add_plane(mut self, n: Vec3, d0: Float64, wt: Float64):
        var x = Float64(n.x); var y = Float64(n.y); var z = Float64(n.z)
        self.a += wt * x * x; self.b += wt * x * y; self.c += wt * x * z; self.d += wt * x * d0
        self.e += wt * y * y; self.f += wt * y * z; self.g += wt * y * d0
        self.h += wt * z * z; self.i += wt * z * d0
        self.j += wt * d0 * d0
        self.w += wt

    def add(mut self, o: Self):
        self.a += o.a; self.b += o.b; self.c += o.c; self.d += o.d; self.e += o.e
        self.f += o.f; self.g += o.g; self.h += o.h; self.i += o.i; self.j += o.j; self.w += o.w

    def error(self, p: Vec3) -> Float32:
        """RMS plane distance of `p` (object units)."""
        var x = Float64(p.x); var y = Float64(p.y); var z = Float64(p.z)
        var q = (self.a * x * x + 2.0 * self.b * x * y + 2.0 * self.c * x * z + 2.0 * self.d * x
                 + self.e * y * y + 2.0 * self.f * y * z + 2.0 * self.g * y
                 + self.h * z * z + 2.0 * self.i * z + self.j)
        if self.w <= 0.0:
            return Float32(0.0)
        return Float32(sqrt(max(0.0, q / self.w)))


def simplify_group_locked(
    mesh: NativeMesh,
    pid: List[Int],
    n_pid: Int,
    tris_in: List[Int],
    locked_pid: List[Bool],
    target_tris: Int,
) -> Tuple[List[Int], Float32]:
    """Half-edge-collapse QEM simplification of a small triangle soup with locked vertices. Collapses
    u -> v only between existing positions, so the result indexes the original wedges. Returns (tris, error)."""
    var tris = tris_in.copy()
    var nt0 = len(tris) // 3
    var max_err = Float32(0.0)
    if nt0 <= target_tris:
        return (tris^, max_err)
    # local pid table
    var lmap = List[Int]()        # local -> pid
    var lpos = List[Vec3]()
    var lwedge = List[Int]()      # representative wedge per local vertex
    var key = List[Int](capacity=len(tris))
    for v in tris:
        key.append(pid[v])
    var ko = stable_order(key)
    var to_local = List[Int](length=len(tris), fill=0)   # per corner
    var wedge_multi = List[Bool]()
    var i = 0
    while i < len(key):
        var j = i
        while j < len(key) and key[ko[j]] == key[ko[i]]:
            j += 1
        var l = len(lmap)
        lmap.append(key[ko[i]])
        var w0 = tris[ko[i]]
        lwedge.append(w0)
        lpos.append(_pos(mesh, w0))
        var multi = False
        for a in range(i, j):
            to_local[ko[a]] = l
            if tris[ko[a]] != w0:
                multi = True
        wedge_multi.append(multi)
        i = j
    var nl = len(lmap)
    var lock = List[Bool](length=nl, fill=False)
    for l in range(nl):
        lock[l] = locked_pid[lmap[l]] or wedge_multi[l]
    # local triangles: corners as local ids; wedges kept per corner
    var lt = List[Int](capacity=len(tris))
    for c in range(len(tris)):
        lt.append(to_local[c])
    # open edges inside the group -> lock endpoints
    var ek = List[Int](capacity=len(lt))
    for t in range(nt0):
        for k in range(3):
            ek.append(_edge_key(lt[t * 3 + k], lt[t * 3 + (k + 1) % 3], nl))
    var eo = stable_order(ek)
    i = 0
    while i < len(ek):
        var j = i
        while j < len(ek) and ek[eo[j]] == ek[eo[i]]:
            j += 1
        if j - i == 1:
            var kk = ek[eo[i]]
            lock[kk // nl] = True
            lock[kk % nl] = True
        i = j
    # quadrics
    var quad = List[_Quadric](length=nl, fill=_Quadric())
    for t in range(nt0):
        var p0 = lpos[lt[t * 3]]; var p1 = lpos[lt[t * 3 + 1]]; var p2 = lpos[lt[t * 3 + 2]]
        var cr = (p1 - p0).cross(p2 - p0)
        var area2 = cr.length()
        if area2 <= Float32(1e-20):
            continue
        var n = cr * (Float32(1.0) / area2)
        var d0 = -Float64(n.dot(p0))
        var wt = Float64(area2) * 0.5
        for k in range(3):
            quad[lt[t * 3 + k]].add_plane(n, d0, wt)
    var alive = List[Bool](length=nt0, fill=True)
    var nalive = nt0
    var remap = List[Int](length=nl, fill=-1)   # collapsed target (local)
    for l in range(nl):
        remap[l] = l
    var nstamp = List[Int](length=nl, fill=0)
    var stamp_id = 0
    for _pass in range(12):
        if nalive <= target_tris:
            break
        # vertex -> incident alive triangles (CSR)
        var vcnt = List[Int](length=nl + 1, fill=0)
        for t in range(nt0):
            if alive[t]:
                for k in range(3):
                    vcnt[lt[t * 3 + k]] += 1
        var voff = List[Int](length=nl + 1, fill=0)
        for l in range(nl):
            voff[l + 1] = voff[l] + vcnt[l]
        var vtri = List[Int](length=voff[nl], fill=0)
        var vfill = voff.copy()
        for t in range(nt0):
            if alive[t]:
                for k in range(3):
                    var l = lt[t * 3 + k]
                    vtri[vfill[l]] = t
                    vfill[l] += 1
        # best collapse per unlocked vertex
        var cand_u = List[Int]()
        var cand_v = List[Int]()
        var cand_key = List[Int]()
        var cand_err = List[Float32]()
        for u in range(nl):
            if lock[u] or voff[u + 1] == voff[u]:
                continue
            var best_v = -1
            var best_e = Float32(1e30)
            for a in range(voff[u], voff[u + 1]):
                var t = vtri[a]
                for k in range(3):
                    var v = lt[t * 3 + k]
                    if v == u:
                        continue
                    var q = quad[u]
                    q.add(quad[v])
                    var e = q.error(lpos[v])
                    if e < best_e:
                        best_e = e
                        best_v = v
            if best_v >= 0:
                cand_u.append(u)
                cand_v.append(best_v)
                cand_err.append(best_e)
                cand_key.append(Int(bitcast[DType.uint32](best_e)))
        if len(cand_u) == 0:
            break
        var co = stable_order(cand_key)
        var touched = List[Bool](length=nl, fill=False)
        var progressed = False
        for ci in range(len(co)):
            if nalive <= target_tris:
                break
            var idx = co[ci]
            var u = cand_u[idx]
            var v = cand_v[idx]
            if touched[u] or touched[v]:
                continue
            # link condition: |N(u) & N(v)| must equal the number of triangles on edge (u, v)
            var shared_tris = 0
            stamp_id += 2
            for b2 in range(voff[v], voff[v + 1]):
                var t2 = vtri[b2]
                if not alive[t2]:
                    continue
                for k in range(3):
                    nstamp[lt[t2 * 3 + k]] = stamp_id
            var common = 0
            for a in range(voff[u], voff[u + 1]):
                var t = vtri[a]
                if not alive[t]:
                    continue
                if lt[t * 3] == v or lt[t * 3 + 1] == v or lt[t * 3 + 2] == v:
                    shared_tris += 1
                for k in range(3):
                    var x = lt[t * 3 + k]
                    if x == u or x == v:
                        continue
                    if nstamp[x] == stamp_id:
                        nstamp[x] = stamp_id + 1   # count each common neighbour once
                        common += 1
            if common > shared_tris:
                continue
            # flip check
            var ok = True
            var pv = lpos[v]
            for a in range(voff[u], voff[u + 1]):
                var t = vtri[a]
                if not alive[t]:
                    continue
                var has_v = lt[t * 3] == v or lt[t * 3 + 1] == v or lt[t * 3 + 2] == v
                if has_v:
                    continue
                var q0 = lpos[lt[t * 3]]; var q1 = lpos[lt[t * 3 + 1]]; var q2 = lpos[lt[t * 3 + 2]]
                var n_old = (q1 - q0).cross(q2 - q0)
                var r0 = pv if lt[t * 3] == u else q0
                var r1 = pv if lt[t * 3 + 1] == u else q1
                var r2 = pv if lt[t * 3 + 2] == u else q2
                var n_new = (r1 - r0).cross(r2 - r0)
                var l_new = n_new.length_squared()
                if l_new <= Float32(1e-24) or n_old.dot(n_new) <= Float32(0.2) * sqrt(n_old.length_squared() * l_new):
                    ok = False
                    break
            if not ok:
                continue
            # perform u -> v
            touched[u] = True
            touched[v] = True
            for a in range(voff[u], voff[u + 1]):
                touched[lt[vtri[a] * 3]] = True
                touched[lt[vtri[a] * 3 + 1]] = True
                touched[lt[vtri[a] * 3 + 2]] = True
            for a in range(voff[u], voff[u + 1]):
                var t = vtri[a]
                if not alive[t]:
                    continue
                var has_v = lt[t * 3] == v or lt[t * 3 + 1] == v or lt[t * 3 + 2] == v
                if has_v:
                    alive[t] = False
                    nalive -= 1
                    continue
                for k in range(3):
                    if lt[t * 3 + k] == u:
                        lt[t * 3 + k] = v
                        tris[t * 3 + k] = lwedge[v]
            var qn = quad[v]
            qn.add(quad[u])
            quad[v] = qn
            remap[u] = v
            max_err = max(max_err, cand_err[idx])
            progressed = True
        if not progressed:
            break
    var out = List[Int](capacity=nalive * 3)
    for t in range(nt0):
        if alive[t]:
            out.append(tris[t * 3])
            out.append(tris[t * 3 + 1])
            out.append(tris[t * 3 + 2])
    _ = remap^
    return (out^, max_err)


# ---------------------------------------------------------------------------------------------
# DAG build
# ---------------------------------------------------------------------------------------------
def build_cluster_dag(mesh: NativeMesh, group_size: Int = SOTA_DAG_GROUP_SIZE) -> ClusterDag:
    var dag = ClusterDag()
    var nv = mesh.vertex_count()
    var nt = len(mesh.indices) // 3
    if nv == 0 or nt == 0:
        return dag^
    var welded = weld_positions(mesh)
    var pid = welded[0].copy()
    var n_pid = welded[1]
    var vstamp = List[Int](length=nv, fill=0)
    var stamp_base = 0
    var lod0 = List[Int](capacity=nt * 3)
    for t in range(nt):
        var a = mesh.indices[t * 3]; var b = mesh.indices[t * 3 + 1]; var c = mesh.indices[t * 3 + 2]
        if a == b or b == c or a == c or a < 0 or b < 0 or c < 0 or a >= nv or b >= nv or c >= nv:
            continue
        lod0.append(a); lod0.append(b); lod0.append(c)
    var cl0 = clusterize_triangles(mesh, pid, n_pid, lod0, vstamp, stamp_base)
    var pending = List[Int]()
    for ct in cl0:
        var c = DagCluster(ct.copy(), 0)
        var bs = _tri_bounds(mesh, c.tris)
        c.self_center = bs[0]
        c.self_radius = bs[1]
        c.self_error = Float32(0.0)
        pending.append(len(dag.clusters))
        dag.clusters.append(c^)
    dag.lod0_clusters = len(dag.clusters)
    dag.lod0_tris = len(lod0) // 3
    var level = 0
    var owner = List[Int](length=n_pid, fill=-1)
    var locked = List[Bool](length=n_pid, fill=False)
    while len(pending) > 1 and level < SOTA_DAG_MAX_LEVELS:
        var groups = partition_clusters(mesh, pid, n_pid, dag.clusters, pending, group_size)
        # vertices used by more than one group are locked
        for p in range(n_pid):
            owner[p] = -1
            locked[p] = False
        for g in range(len(groups)):
            for ci in groups[g]:
                for v in dag.clusters[ci].tris:
                    var p = pid[v]
                    if owner[p] < 0:
                        owner[p] = g
                    elif owner[p] != g:
                        locked[p] = True
        var next_pending = List[Int]()
        var any_built = False
        for g in range(len(groups)):
            ref members = groups[g]
            if len(members) <= 1:
                for ci in members:
                    next_pending.append(ci)
                continue
            var merged = List[Int]()
            var gc = dag.clusters[members[0]].self_center
            var gr = dag.clusters[members[0]].self_radius
            var child_err = Float32(0.0)
            for ci in members:
                ref c = dag.clusters[ci]
                for v in c.tris:
                    merged.append(v)
                var sm = sphere_merge(gc, gr, c.self_center, c.self_radius)
                gc = sm[0]
                gr = sm[1]
                child_err = max(child_err, c.self_error)
            var n_in = len(merged) // 3
            var target = n_in // 2
            var simp = simplify_group_locked(mesh, pid, n_pid, merged, locked, target)
            var n_out = len(simp[0]) // 3
            if n_out == 0 or Float32(n_out) > Float32(n_in) * Float32(0.85):
                dag.groups_failed += 1
                for ci in members:
                    next_pending.append(ci)
                continue
            any_built = True
            dag.groups_built += 1
            var gerr = child_err + simp[1]
            # strictly above every child so the cut is unambiguous even for a zero-error simplification
            gerr = max(gerr, child_err * Float32(1.0001) + Float32(1e-7))
            for ci in members:
                ref c = dag.clusters[ci]
                c.parent_center = gc
                c.parent_radius = gr
                c.parent_error = gerr
            var parts = clusterize_triangles(mesh, pid, n_pid, simp[0], vstamp, stamp_base)
            for pt in parts:
                var nc = DagCluster(pt.copy(), level + 1)
                nc.self_center = gc
                nc.self_radius = gr
                nc.self_error = gerr
                next_pending.append(len(dag.clusters))
                dag.clusters.append(nc^)
        level += 1
        pending = next_pending^
        if not any_built:
            break
    # whatever is left are roots: parent = +inf, parent sphere = own
    for ci in pending:
        ref c = dag.clusters[ci]
        c.parent_center = c.self_center
        c.parent_radius = c.self_radius
        c.parent_error = SOTA_DAG_INF_ERROR
    var maxl = 0
    for c in dag.clusters:
        maxl = max(maxl, c.level)
    dag.levels = maxl + 1
    return dag^


# ---------------------------------------------------------------------------------------------
# CPU reference cut (bit-for-bit the same formula as the GPU culler)
# ---------------------------------------------------------------------------------------------
def dag_projected_error(
    err: Float32, center: Vec3, radius: Float32, eye: Vec3, near_z: Float32, px_per_rad: Float32,
    threshold_px: Float32 = 0.0, rt_bias_base: Float32 = 0.0, rt_bias_slope: Float32 = 0.0,
) -> Float32:
    """Projected error in pixels. With `rt_bias_base > 0` the RT-consistency cap also applies: a cluster whose
    world error exceeds the ray-origin bias at that distance (`base + slope * d`) scores above the threshold,
    so coarse clusters never sit deeper below the LOD0 BLAS than the RT passes' self-hit bias. Both terms
    grow with error and shrink with distance, so the cut stays monotonic (crack-free)."""
    if err >= SOTA_DAG_INF_ERROR:
        return Float32(3.0e38)
    var d = max((center - eye).length() - radius, near_z)
    var pe = err * px_per_rad / d
    if rt_bias_base > Float32(0.0):
        pe = max(pe, err * threshold_px / (rt_bias_base + rt_bias_slope * d))
    return pe


def dag_cluster_selected(
    c: DagCluster, eye_local: Vec3, near_z: Float32, px_per_rad: Float32, threshold_px: Float32,
    rt_bias_base: Float32 = 0.0, rt_bias_slope: Float32 = 0.0,
) -> Bool:
    if threshold_px <= Float32(0.0):
        return c.level == 0
    var ps = dag_projected_error(c.self_error, c.self_center, c.self_radius, eye_local, near_z, px_per_rad, threshold_px, rt_bias_base, rt_bias_slope)
    var pp = dag_projected_error(c.parent_error, c.parent_center, c.parent_radius, eye_local, near_z, px_per_rad, threshold_px, rt_bias_base, rt_bias_slope)
    return ps <= threshold_px and pp > threshold_px


def dag_select_cut(
    dag: ClusterDag, eye_local: Vec3, near_z: Float32, screen_h: Int, tan_half_fov_y: Float32, threshold_px: Float32,
    rt_bias_base: Float32 = 0.0, rt_bias_slope: Float32 = 0.0,
) -> List[Int]:
    var k = Float32(screen_h) / (Float32(2.0) * tan_half_fov_y)
    var out = List[Int]()
    for i in range(len(dag.clusters)):
        if dag_cluster_selected(dag.clusters[i], eye_local, near_z, k, threshold_px, rt_bias_base, rt_bias_slope):
            out.append(i)
    return out^


def dag_cut_open_edges(mesh: NativeMesh, dag: ClusterDag, cut: List[Int]) -> Int:
    """Number of edges used by exactly one triangle of the cut (position-welded). For a closed input mesh a
    crack-free cut has 0; for an open mesh it equals the open-edge count of LOD0."""
    var welded = weld_positions(mesh)
    var n = welded[1]
    var wp = welded[0].copy()
    var ek = List[Int]()
    for ci in cut:
        ref c = dag.clusters[ci]
        for t in range(c.tri_count()):
            for k in range(3):
                ek.append(_edge_key(wp[c.tris[t * 3 + k]], wp[c.tris[t * 3 + (k + 1) % 3]], n))
    var eo = stable_order(ek)
    var open = 0
    var i = 0
    while i < len(ek):
        var j = i
        while j < len(ek) and ek[eo[j]] == ek[eo[i]]:
            j += 1
        if j - i == 1:
            open += 1
        i = j
    return open


def dag_check_monotonic(dag: ClusterDag) -> Bool:
    """Every cluster: parent error >= self error and parent sphere contains self sphere."""
    for c in dag.clusters:
        if c.parent_error < c.self_error:
            return False
        if c.parent_error < SOTA_DAG_INF_ERROR:
            var d = (c.parent_center - c.self_center).length()
            if d + c.self_radius > c.parent_radius * Float32(1.0001) + Float32(1e-5):
                return False
    return True


# ---------------------------------------------------------------------------------------------
# Cook into the shared AaaGeometryPool (+ GPU DAG records)
# ---------------------------------------------------------------------------------------------
def dag_pack_records(dag: ClusterDag, order: List[Int]) -> List[UInt32]:
    var w = List[UInt32](capacity=len(order) * SOTA_DAG_RECORD_WORDS)
    for ci in order:
        ref c = dag.clusters[ci]
        w.append(bitcast[DType.uint32](c.self_center.x))
        w.append(bitcast[DType.uint32](c.self_center.y))
        w.append(bitcast[DType.uint32](c.self_center.z))
        w.append(bitcast[DType.uint32](c.self_radius))
        w.append(bitcast[DType.uint32](c.self_error))
        w.append(bitcast[DType.uint32](c.parent_center.x))
        w.append(bitcast[DType.uint32](c.parent_center.y))
        w.append(bitcast[DType.uint32](c.parent_center.z))
        w.append(bitcast[DType.uint32](c.parent_radius))
        w.append(bitcast[DType.uint32](c.parent_error))
        w.append(UInt32(1))
        w.append(UInt32(c.level))
    return w^


def dag_emit_order(dag: ClusterDag) -> List[Int]:
    """Clusters sorted by level (LOD0 first, so LOD0 indices are contiguous for the BLAS)."""
    var keys = List[Int](capacity=len(dag.clusters))
    for c in dag.clusters:
        keys.append(c.level)
    return stable_order(keys)


def cook_cluster_dag_into_pool(
    mut pool: AaaGeometryPool, mesh: NativeMesh, material_id: Int, dag: ClusterDag
) -> Tuple[CookedMeshEntry, List[UInt32]]:
    """Appends `mesh`'s vertices + every DAG cluster as a meshlet (LOD0 first). Legacy LOD words: LOD0 clusters
    `[0, inf)`, coarser clusters `[inf, inf)` so the legacy culler draws exactly LOD0. Returns the entry and the
    DAG records for meshlets `[entry.first_meshlet, entry.first_meshlet + entry.meshlet_count)`."""
    var entry = CookedMeshEntry()
    var records = List[UInt32]()
    if len(dag.clusters) == 0:
        return (entry^, records^)
    var mesh_id = len(pool.meshes)
    var slot_v = pool.total_vertices()
    var slot_i = len(pool.index_words)
    var slot_m = len(pool.meshlets)
    var empty_splat = List[UInt32]()
    var s0 = pool._append_single_lod_geometry_ex(mesh, mesh_id, material_id, 0, Float32(0.0), Float32(65504.0), False, empty_splat)
    # keep the packed vertices, drop the legacy meshlets/indices
    while len(pool.index_words) > slot_i:
        _ = pool.index_words.pop()
    while len(pool.meshlets) > slot_m:
        _ = pool.meshlets.pop()
    while len(pool.meshlet_words) > slot_m * 12:
        _ = pool.meshlet_words.pop()
    var order = dag_emit_order(dag)
    var all_tris = List[Int]()
    var ranges = List[Int]()
    for ci in order:
        ranges.append(len(all_tris) // 3)
        for v in dag.clusters[ci].tris:
            all_tris.append(v)
    ranges.append(len(all_tris) // 3)
    var lod0_tris = 0
    for oi in range(len(order)):
        ref c = dag.clusters[order[oi]]
        var tl = List[Int]()
        for t in range(ranges[oi], ranges[oi + 1]):
            tl.append(t)
        var se = Float32(0.0) if c.level == 0 else Float32(65504.0)
        pool._emit_meshlet(mesh, all_tris, s0[0], mesh_id, material_id, mesh.double_sided, min(c.level, 3), se, Float32(65504.0), tl)
        if c.level == 0:
            lod0_tris += c.tri_count()
    entry.mesh_id = mesh_id
    entry.material_id = material_id
    entry.first_vertex = s0[0]
    entry.vertex_count = s0[1]
    entry.first_index = slot_i
    entry.triangle_count = lod0_tris
    entry.first_meshlet = slot_m
    entry.meshlet_count = len(pool.meshlets) - slot_m
    entry.lod0_meshlet_count = dag.lod0_clusters
    entry.lod_count = 1
    entry.lods[0] = MeshLodSlice(slot_m, dag.lod0_clusters, lod0_tris, Float32(0.0))
    entry.bounds_center = s0[6]
    entry.bounds_radius = s0[7]
    pool.meshes.append(entry.copy())
    pool._record_slot(slot_v, slot_i, slot_m)
    records = dag_pack_records(dag, order)
    return (entry^, records^)
