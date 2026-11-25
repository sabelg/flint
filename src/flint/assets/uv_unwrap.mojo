"""Automatic UV unwrapping for Flint's geometry kit (`flint.assets.uv_unwrap`), xatlas-style.

Pipeline (all from the published algorithms, re-implemented in pure Mojo):
  1. Chart segmentation: greedy priority-queue region growing over the position-welded face graph
     (cost = normal deviation from the chart's area-weighted normal + roundness + a penalty for
     crossing hard edges / existing UV seams, as in xatlas's chart metrics), followed by Lloyd
     relaxation rounds that re-seed each chart at its most central face and regrow all charts at
     once (Sander et al. 2003 "Multi-chart geometry images" / Cohen-Steiner VSA style).
  2. Parameterisation: Least Squares Conformal Maps (Lévy, Petitjean, Ray & Maillot 2002), solved
     matrix-free with Jacobi-preconditioned conjugate gradients from a planar-projection warm
     start, two pinned vertices. Charts that are not topological discs, that fold (flipped
     triangles), self-overlap or exceed the stretch bound are bisected and re-parameterised.
  3. Packing: every chart is rotated to its minimum-area bounding rectangle (rotating calipers on
     the convex hull), scaled to one uniform texel density and placed with a skyline bottom-left
     packer (both orientations tried) with `padding` texels on every side, inside a square atlas.
  4. Metrics: Sander et al. 2001 L2 / L-inf texture stretch, flipped triangles, exact triangle-pair
     UV overlap and texel-level (conservative raster) chart overlap.

Output vertices are split along chart borders; every other attribute (normals, colours, joints,
weights, AO) is carried over. TEXCOORD_0 is replaced by the atlas UVs (re-bake textures).
"""
from std.math import abs, atan2, ceil, cos, floor, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.mesh_kit import (
    unit, EdgeMap, copy_mesh_material, pair_key, pos_at, vpos, weld_positions, mesh_bounds,
)


# ------------------------------------------------------------------------------- config / result

struct UnwrapConfig(Copyable, Movable):
    var max_normal_deviation_deg: Float32  # vs. the chart's average normal
    var max_seed_deviation_deg: Float32    # vs. the seed face normal
    var roundness_weight: Float32
    var seam_weight: Float32
    var lloyd_iterations: Int
    var min_chart_faces: Int               # smaller charts are merged into a neighbour
    var max_l2_stretch: Float32            # per chart, after LSCM; worse charts are split
    var resolution: Int                    # atlas edge in texels
    var padding: Int                       # texels of gutter around every chart
    var cg_iterations: Int
    var cg_tolerance: Float64

    def __init__(out self):
        self.max_normal_deviation_deg = 55.0
        self.max_seed_deviation_deg = 70.0
        self.roundness_weight = 0.05
        self.seam_weight = 0.5
        self.lloyd_iterations = 2
        self.min_chart_faces = 12
        self.max_l2_stretch = 1.25
        self.resolution = 1024
        self.padding = 2
        self.cg_iterations = 2000
        self.cg_tolerance = 1e-10


struct UnwrapResult(Movable):
    var mesh: NativeMesh
    var chart_of_triangle: List[Int]
    var chart_count: Int
    var resolution: Int
    var utilization: Float32       # chart UV area / atlas area
    var splits: Int                # charts bisected because they failed validation

    def __init__(out self):
        self.mesh = NativeMesh()
        self.chart_of_triangle = List[Int]()
        self.chart_count = 0
        self.resolution = 0
        self.utilization = 0
        self.splits = 0


# ------------------------------------------------------------------------------- min-heap

struct _CostHeap(Movable):
    var cost: List[Float32]
    var face: List[Int]
    var chart: List[Int]

    def __init__(out self):
        self.cost = List[Float32]()
        self.face = List[Int]()
        self.chart = List[Int]()

    def size(self) -> Int:
        return len(self.cost)

    def clear(mut self):
        self.cost.clear()
        self.face.clear()
        self.chart.clear()

    def _swap(mut self, i: Int, j: Int):
        var c = self.cost[i]; self.cost[i] = self.cost[j]; self.cost[j] = c
        var f = self.face[i]; self.face[i] = self.face[j]; self.face[j] = f
        var h = self.chart[i]; self.chart[i] = self.chart[j]; self.chart[j] = h

    def push(mut self, c: Float32, f: Int, ch: Int):
        self.cost.append(c)
        self.face.append(f)
        self.chart.append(ch)
        var i = len(self.cost) - 1
        while i > 0:
            var p = (i - 1) // 2
            if self.cost[p] <= self.cost[i]:
                break
            self._swap(i, p)
            i = p

    def pop(mut self) -> Tuple[Float32, Int, Int]:
        var top = (self.cost[0], self.face[0], self.chart[0])
        var last = len(self.cost) - 1
        self._swap(0, last)
        _ = self.cost.pop()
        _ = self.face.pop()
        _ = self.chart.pop()
        var n = len(self.cost)
        var i = 0
        while True:
            var l = i * 2 + 1
            var r = l + 1
            var m = i
            if l < n and self.cost[l] < self.cost[m]:
                m = l
            if r < n and self.cost[r] < self.cost[m]:
                m = r
            if m == i:
                break
            self._swap(i, m)
            i = m
        return top


# ------------------------------------------------------------------------------- face graph

struct _FaceGraph(Movable):
    var nf: Int
    var fr: List[Int]           # welded rep per corner
    var rp: List[Float32]       # rep positions
    var nrep: Int
    var fn_: List[Vec3]
    var fa: List[Float32]
    var fc: List[Vec3]
    var adj: List[Int]          # 3 per face: neighbour across edge (k, k+1) or -1
    var seam: List[Bool]        # 3 per face: edge is an existing UV seam / hard edge

    def __init__(out self, mesh: NativeMesh):
        var w = weld_positions(mesh, 0.0)
        self.nrep = w[1]
        self.rp = List[Float32](length=self.nrep * 3, fill=0.0)
        for v in range(mesh.vertex_count()):
            var r = w[0][v]
            self.rp[r * 3] = mesh.positions[v * 3]
            self.rp[r * 3 + 1] = mesh.positions[v * 3 + 1]
            self.rp[r * 3 + 2] = mesh.positions[v * 3 + 2]
        self.nf = mesh.triangle_count()
        self.fr = List[Int](length=self.nf * 3, fill=0)
        for i in range(self.nf * 3):
            self.fr[i] = w[0][mesh.indices[i]]
        self.fn_ = List[Vec3](capacity=self.nf)
        self.fa = List[Float32](capacity=self.nf)
        self.fc = List[Vec3](capacity=self.nf)
        for f in range(self.nf):
            var a = pos_at(self.rp, self.fr[f * 3])
            var b = pos_at(self.rp, self.fr[f * 3 + 1])
            var c = pos_at(self.rp, self.fr[f * 3 + 2])
            var cr = (b - a).cross(c - a)
            var l = cr.length()
            self.fa.append(l * 0.5)
            self.fn_.append(cr * (1.0 / l) if l > 1e-30 else Vec3(0, 0, 1))
            self.fc.append((a + b + c) * (1.0 / 3.0))
        self.adj = List[Int](length=self.nf * 3, fill=-1)
        self.seam = List[Bool](length=self.nf * 3, fill=False)
        var em = EdgeMap(self.nf * 3)
        var uses = List[Int]()
        for f in range(self.nf):
            for k in range(3):
                var a = self.fr[f * 3 + k]
                var b = self.fr[f * 3 + (k + 1) % 3]
                if a == b:
                    continue
                var key = pair_key(min(a, b), max(a, b))
                var slot = em.insert(key, len(uses))
                if slot == len(uses):
                    uses.append(f * 3 + k)
                    continue
                var prev = uses[slot]
                if prev < 0:
                    continue  # already non-manifold
                var g = prev // 3
                if self.adj[prev] >= 0 or g == f:
                    # third use: non-manifold, cut all
                    var other = self.adj[prev]
                    if other >= 0:
                        for kk in range(3):
                            if self.adj[other * 3 + kk] == g:
                                self.adj[other * 3 + kk] = -1
                    self.adj[prev] = -1
                    uses[slot] = -1
                    continue
                self.adj[prev] = f
                self.adj[f * 3 + k] = g
                # original corners differ -> UV seam / normal split
                var kg = prev % 3
                var ga = mesh.indices[g * 3 + kg]
                var gb = mesh.indices[g * 3 + (kg + 1) % 3]
                var fa_ = mesh.indices[f * 3 + k]
                var fb = mesh.indices[f * 3 + (k + 1) % 3]
                var s = not ((ga == fb and gb == fa_) or (ga == fa_ and gb == fb))
                self.seam[prev] = s
                self.seam[f * 3 + k] = s


# ------------------------------------------------------------------------------- segmentation

def _grow_cost(g: _FaceGraph, f: Int, from_face: Int, edge: Int, n_avg: Vec3, centroid: Vec3, area: Float32,
               cfg: UnwrapConfig) -> Float32:
    var dev = 1.0 - g.fn_[f].dot(n_avg)
    var r = (g.fc[f] - centroid).length() / max(Float32(1e-12), sqrt(area))
    var c = dev + cfg.roundness_weight * r
    if from_face >= 0 and g.seam[from_face * 3 + edge]:
        c += cfg.seam_weight
    return c


def _segment(g: _FaceGraph, cfg: UnwrapConfig) -> List[Int]:
    var chart_of = List[Int](length=g.nf, fill=-1)
    var cos_avg = cos(cfg.max_normal_deviation_deg * 0.017453292)
    var cos_seed = cos(cfg.max_seed_deviation_deg * 0.017453292)
    var seeds = List[Int]()
    # per-chart accumulators
    var n_sum = List[Vec3]()
    var c_sum = List[Vec3]()
    var a_sum = List[Float32]()
    var heap = _CostHeap()

    # pass 0: sequential growth gives initial seeds; later passes regrow from relaxed seeds at once
    for it in range(cfg.lloyd_iterations + 1):
        if it > 0:
            # relaxed seeds: the face nearest the chart centroid among faces aligned with its normal
            var nc = len(seeds)
            var best_score = List[Float32](length=nc, fill=Float32(3.0e38))
            var new_seed = seeds.copy()
            for f in range(g.nf):
                var c = chart_of[f]
                if c < 0:
                    continue
                var n = unit(n_sum[c])
                var cen = c_sum[c] * (1.0 / max(Float32(1e-30), a_sum[c]))
                var s = (g.fc[f] - cen).length() / max(Float32(1e-12), sqrt(a_sum[c])) + (1.0 - g.fn_[f].dot(n)) * 2.0
                if s < best_score[c]:
                    best_score[c] = s
                    new_seed[c] = f
            seeds = new_seed^
            for f in range(g.nf):
                chart_of[f] = -1
        n_sum.clear()
        c_sum.clear()
        a_sum.clear()
        heap.clear()
        var seed_n = List[Vec3]()
        for c in range(len(seeds)):
            var s = seeds[c]
            chart_of[s] = c
            n_sum.append(g.fn_[s] * g.fa[s])
            c_sum.append(g.fc[s] * g.fa[s])
            a_sum.append(g.fa[s])
            seed_n.append(g.fn_[s])
            for k in range(3):
                var nb = g.adj[s * 3 + k]
                if nb >= 0 and chart_of[nb] < 0:
                    heap.push(_grow_cost(g, nb, s, k, g.fn_[s], g.fc[s], g.fa[s], cfg), nb, c)
        var next_unassigned = 0
        while True:
            # multi-source growth of all live charts
            while heap.size() > 0:
                var top = heap.pop()
                var f = top[1]
                var c = top[2]
                if chart_of[f] >= 0:
                    continue
                var n_avg = unit(n_sum[c])
                if g.fn_[f].dot(n_avg) < cos_avg or g.fn_[f].dot(seed_n[c]) < cos_seed:
                    continue
                chart_of[f] = c
                n_sum[c] = n_sum[c] + g.fn_[f] * g.fa[f]
                c_sum[c] = c_sum[c] + g.fc[f] * g.fa[f]
                a_sum[c] += g.fa[f]
                var cen = c_sum[c] * (1.0 / max(Float32(1e-30), a_sum[c]))
                var nn = unit(n_sum[c])
                for k in range(3):
                    var nb = g.adj[f * 3 + k]
                    if nb >= 0 and chart_of[nb] < 0:
                        heap.push(_grow_cost(g, nb, f, k, nn, cen, a_sum[c], cfg), nb, c)
            # seed a new chart on the next unassigned face
            while next_unassigned < g.nf and chart_of[next_unassigned] >= 0:
                next_unassigned += 1
            if next_unassigned >= g.nf:
                break
            var s = next_unassigned
            var c = len(n_sum)
            if it == 0 or c >= len(seeds):
                seeds.append(s)
            else:
                seeds[c] = s
            chart_of[s] = c
            n_sum.append(g.fn_[s] * g.fa[s])
            c_sum.append(g.fc[s] * g.fa[s])
            a_sum.append(g.fa[s])
            seed_n.append(g.fn_[s])
            for k in range(3):
                var nb = g.adj[s * 3 + k]
                if nb >= 0 and chart_of[nb] < 0:
                    heap.push(_grow_cost(g, nb, s, k, g.fn_[s], g.fc[s], g.fa[s], cfg), nb, c)
        # drop seeds of charts that never formed in this round
        while len(seeds) > len(n_sum):
            _ = seeds.pop()
    _merge_small_charts(g, chart_of, cfg)
    return chart_of^


def _merge_small_charts(g: _FaceGraph, mut chart_of: List[Int], cfg: UnwrapConfig):
    """Folds charts smaller than `min_chart_faces` into the neighbour they share the most edges
    with (when their normals are compatible), then renumbers charts densely."""
    var nc = 0
    for f in range(g.nf):
        nc = max(nc, chart_of[f] + 1)
    var cos_lim = cos(cfg.max_seed_deviation_deg * 0.017453292)
    for _ in range(3):
        var count = List[Int](length=nc, fill=0)
        var nsum = List[Vec3](length=nc, fill=Vec3(0, 0, 0))
        var members = List[List[Int]]()
        for _c in range(nc):
            members.append(List[Int]())
        for f in range(g.nf):
            var c = chart_of[f]
            count[c] += 1
            nsum[c] = nsum[c] + g.fn_[f] * g.fa[f]
            members[c].append(f)
        var changed = False
        var shared = List[Int](length=nc, fill=0)
        for c in range(nc):
            if count[c] == 0 or count[c] >= cfg.min_chart_faces:
                continue
            var touched = List[Int]()
            for i in range(len(members[c])):
                var f = members[c][i]
                for k in range(3):
                    var nb = g.adj[f * 3 + k]
                    if nb < 0:
                        continue
                    var o = chart_of[nb]
                    if o == c:
                        continue
                    if shared[o] == 0:
                        touched.append(o)
                    shared[o] += 1
            var best = -1
            var best_score: Float32 = -1
            var nc_c = unit(nsum[c], g.fn_[members[c][0]])
            for i in range(len(touched)):
                var o = touched[i]
                var d = nc_c.dot(unit(nsum[o], nc_c))
                if d >= cos_lim:
                    var sc = Float32(shared[o]) * (1.0 + d)
                    if sc > best_score:
                        best_score = sc
                        best = o
                shared[o] = 0
            if best >= 0:
                for i in range(len(members[c])):
                    chart_of[members[c][i]] = best
                count[best] += count[c]
                nsum[best] = nsum[best] + nsum[c]
                count[c] = 0
                changed = True
        if not changed:
            break
    # dense renumbering
    var remap = List[Int](length=nc, fill=-1)
    var next = 0
    for f in range(g.nf):
        var c = chart_of[f]
        if remap[c] < 0:
            remap[c] = next
            next += 1
        chart_of[f] = remap[c]


# ------------------------------------------------------------------------------- LSCM

struct _Chart(Movable):
    var faces: List[Int]
    var verts: List[Int]        # welded reps
    var local: List[Int]        # 3 per face: index into verts
    var u: List[Float64]
    var v: List[Float64]

    def __init__(out self):
        self.faces = List[Int]()
        self.verts = List[Int]()
        self.local = List[Int]()
        self.u = List[Float64]()
        self.v = List[Float64]()


def _build_chart(g: _FaceGraph, faces: List[Int], mut slot: List[Int]) -> _Chart:
    var ch = _Chart()
    ch.faces = faces.copy()
    for i in range(len(faces)):
        var f = faces[i]
        for k in range(3):
            var r = g.fr[f * 3 + k]
            if slot[r] < 0:
                slot[r] = len(ch.verts)
                ch.verts.append(r)
            ch.local.append(slot[r])
    for i in range(len(ch.verts)):
        slot[ch.verts[i]] = -1
    return ch^


def _chart_is_disc(g: _FaceGraph, ch: _Chart) -> Bool:
    """Euler characteristic V - E + F == 1 and a single boundary loop."""
    var em = EdgeMap(len(ch.faces) * 3)
    var cnt = List[Int]()
    for i in range(len(ch.faces)):
        for k in range(3):
            var a = ch.local[i * 3 + k]
            var b = ch.local[i * 3 + (k + 1) % 3]
            if a == b:
                return False
            var s = em.insert(pair_key(min(a, b), max(a, b)), len(cnt))
            if s == len(cnt):
                cnt.append(0)
            cnt[s] += 1
    for e in range(len(cnt)):
        if cnt[e] > 2:
            return False
    return len(ch.verts) - len(cnt) + len(ch.faces) == 1


def _lscm(g: _FaceGraph, mut ch: _Chart, cfg: UnwrapConfig):
    var nv = len(ch.verts)
    var nf = len(ch.faces)
    # chart frame for the planar warm start
    var n = Vec3(0, 0, 0)
    for i in range(nf):
        n = n + g.fn_[ch.faces[i]] * g.fa[ch.faces[i]]
    n = unit(n)
    if n.length_squared() < 0.5:
        n = Vec3(0, 0, 1)
    var ax = Vec3(1, 0, 0) if abs(n.x) < 0.9 else Vec3(0, 1, 0)
    var tx = unit(ax.cross(n))
    var ty = n.cross(tx)
    ch.u = List[Float64](length=nv, fill=0.0)
    ch.v = List[Float64](length=nv, fill=0.0)
    for i in range(nv):
        var p = pos_at(g.rp, ch.verts[i])
        ch.u[i] = Float64(p.dot(tx))
        ch.v[i] = Float64(p.dot(ty))
    if nf == 1 or nv <= 3:
        _single_triangle_layout(g, ch)
        return
    # per-face complex coefficients W_j = (x_{j+2} - x_{j+1}) + i (y_{j+2} - y_{j+1}), / sqrt(2A)
    var wr = List[Float64](length=nf * 3, fill=0.0)
    var wi = List[Float64](length=nf * 3, fill=0.0)
    for i in range(nf):
        var p0 = pos_at(g.rp, ch.verts[ch.local[i * 3]])
        var p1 = pos_at(g.rp, ch.verts[ch.local[i * 3 + 1]])
        var p2 = pos_at(g.rp, ch.verts[ch.local[i * 3 + 2]])
        var e1 = p1 - p0
        var l1 = e1.length()
        var fnn = e1.cross(p2 - p0)
        var area2 = fnn.length()
        if l1 < 1e-20 or area2 < 1e-30:
            continue
        var xx = e1 * (1.0 / l1)
        var yy = (fnn * (1.0 / area2)).cross(xx)
        var xs = [Float64(0), Float64(l1), Float64((p2 - p0).dot(xx))]
        var ys = [Float64(0), Float64(0), Float64((p2 - p0).dot(yy))]
        var s = 1.0 / sqrt(Float64(area2))
        for j in range(3):
            var j1 = (j + 1) % 3
            var j2 = (j + 2) % 3
            wr[i * 3 + j] = (xs[j2] - xs[j1]) * s
            wi[i * 3 + j] = (ys[j2] - ys[j1]) * s
    # pins: the two vertices farthest apart along the dominant projected axis
    var du_lo = 0
    var du_hi = 0
    var dv_lo = 0
    var dv_hi = 0
    for i in range(nv):
        if ch.u[i] < ch.u[du_lo]: du_lo = i
        if ch.u[i] > ch.u[du_hi]: du_hi = i
        if ch.v[i] < ch.v[dv_lo]: dv_lo = i
        if ch.v[i] > ch.v[dv_hi]: dv_hi = i
    var use_u = ch.u[du_hi] - ch.u[du_lo] >= ch.v[dv_hi] - ch.v[dv_lo]
    var pin_a = du_lo if use_u else dv_lo
    var pin_b = du_hi if use_u else dv_hi
    var pinned = List[Bool](length=nv, fill=False)
    pinned[pin_a] = True
    pinned[pin_b] = True
    # Jacobi diagonal
    var diag = List[Float64](length=nv, fill=0.0)
    for i in range(nf):
        for j in range(3):
            diag[ch.local[i * 3 + j]] += wr[i * 3 + j] * wr[i * 3 + j] + wi[i * 3 + j] * wi[i * 3 + j]
    # CG on the normal equations; x = (u, v) interleaved, pinned entries fixed
    var x = List[Float64](length=nv * 2, fill=0.0)
    for i in range(nv):
        x[i * 2] = ch.u[i]
        x[i * 2 + 1] = ch.v[i]
    var r = List[Float64](length=nv * 2, fill=0.0)
    _lscm_apply(ch, wr, wi, x, r)  # r = A^T A x (with pins included)
    for i in range(nv * 2):
        r[i] = -r[i]
    for i in range(nv):
        if pinned[i]:
            r[i * 2] = 0
            r[i * 2 + 1] = 0
    var z = List[Float64](length=nv * 2, fill=0.0)
    for i in range(nv):
        var d = 1.0 / max(diag[i], 1e-30)
        z[i * 2] = r[i * 2] * d
        z[i * 2 + 1] = r[i * 2 + 1] * d
    var p = z.copy()
    var ap = List[Float64](length=nv * 2, fill=0.0)
    var rz: Float64 = 0
    var r0: Float64 = 0
    for i in range(nv * 2):
        rz += r[i] * z[i]
        r0 += r[i] * r[i]
    var it = 0
    while it < cfg.cg_iterations and r0 > 0:
        _lscm_apply(ch, wr, wi, p, ap)
        for i in range(nv):
            if pinned[i]:
                ap[i * 2] = 0
                ap[i * 2 + 1] = 0
        var pap: Float64 = 0
        for i in range(nv * 2):
            pap += p[i] * ap[i]
        if pap <= 1e-300:
            break
        var alpha = rz / pap
        var rr: Float64 = 0
        for i in range(nv * 2):
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
            rr += r[i] * r[i]
        if rr <= cfg.cg_tolerance * r0:
            break
        var rz_new: Float64 = 0
        for i in range(nv):
            var d = 1.0 / max(diag[i], 1e-30)
            z[i * 2] = r[i * 2] * d
            z[i * 2 + 1] = r[i * 2 + 1] * d
            rz_new += r[i * 2] * z[i * 2] + r[i * 2 + 1] * z[i * 2 + 1]
        var beta = rz_new / rz
        rz = rz_new
        for i in range(nv * 2):
            p[i] = z[i] + beta * p[i]
        it += 1
    for i in range(nv):
        ch.u[i] = x[i * 2]
        ch.v[i] = x[i * 2 + 1]


def _lscm_apply(ch: _Chart, wr: List[Float64], wi: List[Float64], x: List[Float64], mut out: List[Float64]):
    for i in range(len(out)):
        out[i] = 0
    for f in range(len(ch.faces)):
        var re: Float64 = 0
        var im: Float64 = 0
        for j in range(3):
            var l = ch.local[f * 3 + j]
            var a = wr[f * 3 + j]
            var b = wi[f * 3 + j]
            var uu = x[l * 2]
            var vv = x[l * 2 + 1]
            re += a * uu - b * vv
            im += b * uu + a * vv
        for j in range(3):
            var l = ch.local[f * 3 + j]
            var a = wr[f * 3 + j]
            var b = wi[f * 3 + j]
            out[l * 2] += a * re + b * im
            out[l * 2 + 1] += -b * re + a * im


def _single_triangle_layout(g: _FaceGraph, mut ch: _Chart):
    """Exact isometric layout of each triangle (used for 1-face charts)."""
    var f = 0
    var p0 = pos_at(g.rp, ch.verts[ch.local[f * 3]])
    var p1 = pos_at(g.rp, ch.verts[ch.local[f * 3 + 1]])
    var p2 = pos_at(g.rp, ch.verts[ch.local[f * 3 + 2]])
    var e1 = p1 - p0
    var l1 = max(Float32(1e-20), e1.length())
    var xx = e1 * (1.0 / l1)
    var nn = unit(e1.cross(p2 - p0))
    var yy = nn.cross(xx)
    ch.u[ch.local[0]] = 0
    ch.v[ch.local[0]] = 0
    ch.u[ch.local[1]] = Float64(l1)
    ch.v[ch.local[1]] = 0
    ch.u[ch.local[2]] = Float64((p2 - p0).dot(xx))
    ch.v[ch.local[2]] = Float64((p2 - p0).dot(yy))


# ------------------------------------------------------------------------------- chart metrics

@fieldwise_init
struct StretchStats(Copyable, ImplicitlyCopyable, Movable, Writable):
    var l2: Float32          # area-weighted Sander L2 stretch, normalised (1 = isometric)
    var linf: Float32        # largest per-triangle L-inf stretch (normalised)
    var flipped: Int
    var degenerate: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write("L2 stretch=", self.l2, " Linf=", self.linf, " flipped=", self.flipped, " degenerate=", self.degenerate)


def _stretch_tri(q1: Vec3, q2: Vec3, q3: Vec3, s1: Float64, t1: Float64, s2: Float64, t2: Float64,
                 s3: Float64, t3: Float64) -> Tuple[Float64, Float64, Float64]:
    """(signed uv area, L2^2, Linf) of one triangle (Sander, Snyder, Gortler & Hoppe 2001)."""
    var a2 = (s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)
    if abs(a2) < 1e-30:
        return (a2 * 0.5, 0.0, 0.0)
    var inv = 1.0 / a2
    var ss = (q1 * Float32(t2 - t3) + q2 * Float32(t3 - t1) + q3 * Float32(t1 - t2)) * Float32(inv)
    var st = (q1 * Float32(s3 - s2) + q2 * Float32(s1 - s3) + q3 * Float32(s2 - s1)) * Float32(inv)
    var a = Float64(ss.dot(ss))
    var b = Float64(ss.dot(st))
    var c = Float64(st.dot(st))
    var l2sq = (a + c) * 0.5
    var big = sqrt(max(0.0, ((a + c) + sqrt((a - c) * (a - c) + 4.0 * b * b)) * 0.5))
    return (a2 * 0.5, l2sq, big)


def _chart_stretch(g: _FaceGraph, ch: _Chart) -> StretchStats:
    var sum_a3: Float64 = 0
    var sum_auv: Float64 = 0
    var sum_l2: Float64 = 0
    var linf: Float64 = 0
    var flipped = 0
    var degenerate = 0
    for i in range(len(ch.faces)):
        var f = ch.faces[i]
        var l0 = ch.local[i * 3]
        var l1 = ch.local[i * 3 + 1]
        var l2 = ch.local[i * 3 + 2]
        var r = _stretch_tri(pos_at(g.rp, ch.verts[l0]), pos_at(g.rp, ch.verts[l1]), pos_at(g.rp, ch.verts[l2]),
                             ch.u[l0], ch.v[l0], ch.u[l1], ch.v[l1], ch.u[l2], ch.v[l2])
        var a3 = Float64(g.fa[f])
        if a3 <= 1e-30:
            continue
        if r[0] <= 0:
            if abs(r[0]) < 1e-12 * a3:
                degenerate += 1
            else:
                flipped += 1
            continue
        sum_a3 += a3
        sum_auv += r[0]
        sum_l2 += r[1] * a3
        linf = max(linf, r[2])
    if sum_a3 <= 0:
        return StretchStats(1.0, 1.0, flipped, degenerate)
    var norm = sqrt(sum_auv / sum_a3)
    return StretchStats(Float32(sqrt(sum_l2 / sum_a3) * norm), Float32(linf * norm), flipped, degenerate)


@always_inline
def _orient(ax: Float64, ay: Float64, bx: Float64, by: Float64, cx: Float64, cy: Float64) -> Float64:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def tri_tri_overlap_2d(a: List[Float64], b: List[Float64], eps: Float64) -> Bool:
    """Interior overlap of two 2D triangles (6 coords each) by the separating axis test; each
    triangle is shrunk by `eps` (relative) toward its centroid so shared edges/vertices do not count."""
    var pa = List[Float64](length=6, fill=0.0)
    var pb = List[Float64](length=6, fill=0.0)
    var cax = (a[0] + a[2] + a[4]) / 3.0
    var cay = (a[1] + a[3] + a[5]) / 3.0
    var cbx = (b[0] + b[2] + b[4]) / 3.0
    var cby = (b[1] + b[3] + b[5]) / 3.0
    for k in range(3):
        pa[k * 2] = cax + (a[k * 2] - cax) * (1.0 - eps)
        pa[k * 2 + 1] = cay + (a[k * 2 + 1] - cay) * (1.0 - eps)
        pb[k * 2] = cbx + (b[k * 2] - cbx) * (1.0 - eps)
        pb[k * 2 + 1] = cby + (b[k * 2 + 1] - cby) * (1.0 - eps)
    for side in range(2):
        for k in range(3):
            var x0 = pa[k * 2] if side == 0 else pb[k * 2]
            var y0 = pa[k * 2 + 1] if side == 0 else pb[k * 2 + 1]
            var x1 = pa[((k + 1) % 3) * 2] if side == 0 else pb[((k + 1) % 3) * 2]
            var y1 = pa[((k + 1) % 3) * 2 + 1] if side == 0 else pb[((k + 1) % 3) * 2 + 1]
            var nx = -(y1 - y0)
            var ny = x1 - x0
            var amin: Float64 = 1e300
            var amax: Float64 = -1e300
            var bmin: Float64 = 1e300
            var bmax: Float64 = -1e300
            for j in range(3):
                var da = pa[j * 2] * nx + pa[j * 2 + 1] * ny
                var db = pb[j * 2] * nx + pb[j * 2 + 1] * ny
                amin = min(amin, da); amax = max(amax, da)
                bmin = min(bmin, db); bmax = max(bmax, db)
            if amax <= bmin or bmax <= amin:
                return False
    return True


def count_uv_overlaps(uv: List[Float64], tri_count: Int, eps: Float64 = 1e-3, limit: Int = 1 << 30) -> Int:
    """Number of overlapping triangle pairs among `tri_count` triangles (6 coords each)."""
    if tri_count < 2:
        return 0
    var lo_x: Float64 = 1e300
    var lo_y: Float64 = 1e300
    var hi_x: Float64 = -1e300
    var hi_y: Float64 = -1e300
    for i in range(tri_count * 3):
        lo_x = min(lo_x, uv[i * 2]); hi_x = max(hi_x, uv[i * 2])
        lo_y = min(lo_y, uv[i * 2 + 1]); hi_y = max(hi_y, uv[i * 2 + 1])
    var side = max(1, Int(sqrt(Float64(tri_count))))
    var w = max(hi_x - lo_x, 1e-30)
    var h = max(hi_y - lo_y, 1e-30)
    var gx = side
    var gy = side
    var counts = List[Int](length=gx * gy + 1, fill=0)
    var ranges = List[Int](length=tri_count * 4, fill=0)
    for t in range(tri_count):
        var x0 = min(uv[t * 6], min(uv[t * 6 + 2], uv[t * 6 + 4]))
        var x1 = max(uv[t * 6], max(uv[t * 6 + 2], uv[t * 6 + 4]))
        var y0 = min(uv[t * 6 + 1], min(uv[t * 6 + 3], uv[t * 6 + 5]))
        var y1 = max(uv[t * 6 + 1], max(uv[t * 6 + 3], uv[t * 6 + 5]))
        var cx0 = max(0, min(gx - 1, Int((x0 - lo_x) / w * Float64(gx))))
        var cx1 = max(0, min(gx - 1, Int((x1 - lo_x) / w * Float64(gx))))
        var cy0 = max(0, min(gy - 1, Int((y0 - lo_y) / h * Float64(gy))))
        var cy1 = max(0, min(gy - 1, Int((y1 - lo_y) / h * Float64(gy))))
        ranges[t * 4] = cx0; ranges[t * 4 + 1] = cx1; ranges[t * 4 + 2] = cy0; ranges[t * 4 + 3] = cy1
        for y in range(cy0, cy1 + 1):
            for x in range(cx0, cx1 + 1):
                counts[y * gx + x + 1] += 1
    for i in range(gx * gy):
        counts[i + 1] += counts[i]
    var cells = List[Int](length=counts[gx * gy], fill=0)
    var fill = counts.copy()
    for t in range(tri_count):
        for y in range(ranges[t * 4 + 2], ranges[t * 4 + 3] + 1):
            for x in range(ranges[t * 4], ranges[t * 4 + 1] + 1):
                cells[fill[y * gx + x]] = t
                fill[y * gx + x] += 1
    var ta = List[Float64](length=6, fill=0.0)
    var tb = List[Float64](length=6, fill=0.0)
    var found = 0
    for t in range(tri_count):
        for k in range(6):
            ta[k] = uv[t * 6 + k]
        for y in range(ranges[t * 4 + 2], ranges[t * 4 + 3] + 1):
            for x in range(ranges[t * 4], ranges[t * 4 + 1] + 1):
                var ci = y * gx + x
                for s in range(counts[ci], counts[ci + 1]):
                    var o = cells[s]
                    if o <= t:
                        continue
                    # count each pair once: only in the first cell both share
                    var fx = max(ranges[t * 4], ranges[o * 4])
                    var fy = max(ranges[t * 4 + 2], ranges[o * 4 + 2])
                    if fx != x or fy != y:
                        continue
                    for k in range(6):
                        tb[k] = uv[o * 6 + k]
                    if tri_tri_overlap_2d(ta, tb, eps):
                        found += 1
                        if found >= limit:
                            return found
    return found


def _chart_self_overlaps(ch: _Chart) -> Bool:
    var uv = List[Float64](capacity=len(ch.faces) * 6)
    for i in range(len(ch.faces) * 3):
        uv.append(ch.u[ch.local[i]])
        uv.append(ch.v[ch.local[i]])
    return count_uv_overlaps(uv, len(ch.faces), 1e-3, 1) > 0


def _bisect(g: _FaceGraph, faces: List[Int]) -> List[List[Int]]:
    """Split a face set in two at the median of its widest centroid axis; returns connected pieces."""
    var lo = g.fc[faces[0]]
    var hi = lo
    for i in range(len(faces)):
        var c = g.fc[faces[i]]
        lo = Vec3(min(lo.x, c.x), min(lo.y, c.y), min(lo.z, c.z))
        hi = Vec3(max(hi.x, c.x), max(hi.y, c.y), max(hi.z, c.z))
    var e = hi - lo
    var axis = 0
    if e.y > e.x and e.y >= e.z:
        axis = 1
    elif e.z > e.x and e.z > e.y:
        axis = 2
    var keys = List[Float32](capacity=len(faces))
    for i in range(len(faces)):
        var c = g.fc[faces[i]]
        keys.append(c.x if axis == 0 else (c.y if axis == 1 else c.z))
    var sorted_k = keys.copy()
    sort(sorted_k)
    var med = sorted_k[len(sorted_k) // 2]
    var half = List[Int](length=g.nf, fill=-1)
    var n_lo = 0
    for i in range(len(faces)):
        var side = 0 if keys[i] < med else 1
        n_lo += 1 - side
        half[faces[i]] = side
    if n_lo == 0 or n_lo == len(faces):
        # all keys equal: split by list order
        for i in range(len(faces)):
            half[faces[i]] = 0 if i < len(faces) // 2 else 1
    return _components(g, faces, half)


def _components(g: _FaceGraph, faces: List[Int], label: List[Int]) -> List[List[Int]]:
    """Connected components of `faces`, never crossing a label change."""
    var out = List[List[Int]]()
    var comp = List[Int](length=g.nf, fill=-2)
    for i in range(len(faces)):
        comp[faces[i]] = -1
    var stack = List[Int]()
    for i in range(len(faces)):
        var s = faces[i]
        if comp[s] != -1:
            continue
        var piece = List[Int]()
        comp[s] = len(out)
        stack.append(s)
        while len(stack) > 0:
            var f = stack.pop()
            piece.append(f)
            for k in range(3):
                var nb = g.adj[f * 3 + k]
                if nb >= 0 and comp[nb] == -1 and label[nb] == label[f]:
                    comp[nb] = len(out)
                    stack.append(nb)
        out.append(piece^)
    return out^


# ------------------------------------------------------------------------------- packing

def _convex_hull(xs: List[Float64], ys: List[Float64]) -> List[Int]:
    var n = len(xs)
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    _sort_by_key(order, xs, ys)
    if n < 3:
        return order^
    var hull = List[Int]()
    for pass_ in range(2):
        var start = len(hull)
        for ii in range(n):
            var i = order[ii] if pass_ == 0 else order[n - 1 - ii]
            while len(hull) >= start + 2 and _orient(xs[hull[len(hull) - 2]], ys[hull[len(hull) - 2]],
                                                    xs[hull[len(hull) - 1]], ys[hull[len(hull) - 1]], xs[i], ys[i]) <= 0:
                _ = hull.pop()
            hull.append(i)
        _ = hull.pop()
    return hull^


def _sort_by_key(mut order: List[Int], xs: List[Float64], ys: List[Float64]):
    # bottom-up merge sort on (x, y)
    var n = len(order)
    var tmp = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var i = 0
        while i < n:
            var mid = min(i + width, n)
            var hi = min(i + 2 * width, n)
            var a = i
            var b = mid
            var k = i
            while a < mid and b < hi:
                var oa = order[a]
                var ob = order[b]
                if xs[oa] < xs[ob] or (xs[oa] == xs[ob] and ys[oa] <= ys[ob]):
                    tmp[k] = oa; a += 1
                else:
                    tmp[k] = ob; b += 1
                k += 1
            while a < mid:
                tmp[k] = order[a]; a += 1; k += 1
            while b < hi:
                tmp[k] = order[b]; b += 1; k += 1
            i += 2 * width
        for j in range(n):
            order[j] = tmp[j]
        width *= 2


def _min_area_rotation(mut ch: _Chart):
    """Rotates the chart so its minimum-area bounding rectangle is axis aligned, and moves it to
    the origin. Rotating calipers over the convex-hull edge directions."""
    var hull = _convex_hull(ch.u, ch.v)
    var best_area: Float64 = 1e300
    var best_c: Float64 = 1
    var best_s: Float64 = 0
    var nh = len(hull)
    for i in range(nh):
        var a = hull[i]
        var b = hull[(i + 1) % nh]
        var dx = ch.u[b] - ch.u[a]
        var dy = ch.v[b] - ch.v[a]
        var l = sqrt(dx * dx + dy * dy)
        if l < 1e-30:
            continue
        var c = dx / l
        var s = dy / l
        var x0: Float64 = 1e300
        var x1: Float64 = -1e300
        var y0: Float64 = 1e300
        var y1: Float64 = -1e300
        for j in range(nh):
            var h = hull[j]
            var x = ch.u[h] * c + ch.v[h] * s
            var y = -ch.u[h] * s + ch.v[h] * c
            x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
        var area = (x1 - x0) * (y1 - y0)
        if area < best_area:
            best_area = area
            best_c = c
            best_s = s
    var lo_x: Float64 = 1e300
    var lo_y: Float64 = 1e300
    for i in range(len(ch.u)):
        var x = ch.u[i] * best_c + ch.v[i] * best_s
        var y = -ch.u[i] * best_s + ch.v[i] * best_c
        ch.u[i] = x
        ch.v[i] = y
        lo_x = min(lo_x, x)
        lo_y = min(lo_y, y)
    for i in range(len(ch.u)):
        ch.u[i] -= lo_x
        ch.v[i] -= lo_y


@fieldwise_init
struct _Placement(Copyable, ImplicitlyCopyable, Movable):
    var x: Int
    var y: Int
    var rotated: Bool


def _skyline_pack(ws: List[Int], hs: List[Int], width: Int, mut out: List[_Placement]) -> Int:
    """Skyline bottom-left packing of rectangles (largest first, both orientations); returns the
    packed height."""
    var n = len(ws)
    out = List[_Placement](length=n, fill=_Placement(0, 0, False))
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    # sort by max side, descending (simple merge sort on negative keys)
    var keys_x = List[Float64](capacity=n)
    var keys_y = List[Float64](capacity=n)
    for i in range(n):
        keys_x.append(-Float64(max(ws[i], hs[i])))
        keys_y.append(-Float64(min(ws[i], hs[i])))
    _sort_by_key(order, keys_x, keys_y)
    var sx = List[Int]()
    var sy = List[Int]()
    var sw = List[Int]()
    sx.append(0); sy.append(0); sw.append(width)
    var top = 0
    for oi in range(n):
        var r = order[oi]
        var best_y = 1 << 60
        var best_x = 0
        var best_rot = False
        var best_seg = -1
        for rot in range(2):
            var w = ws[r] if rot == 0 else hs[r]
            var h = hs[r] if rot == 0 else ws[r]
            if w > width:
                continue
            for s in range(len(sx)):
                var x = sx[s]
                if x + w > width:
                    break
                # fit: max skyline height across [x, x + w)
                var y = 0
                var j = s
                var remaining = w
                while remaining > 0 and j < len(sx):
                    y = max(y, sy[j])
                    remaining -= sw[j] - (x - sx[j] if j == s else 0)
                    j += 1
                if y + h < best_y or (y + h == best_y and x < best_x):
                    best_y = y + h
                    best_x = x
                    best_rot = rot == 1
                    best_seg = s
        if best_seg < 0:
            return 1 << 60
        var w = hs[r] if best_rot else ws[r]
        var h = ws[r] if best_rot else hs[r]
        var y_base = best_y - h
        out[r] = _Placement(best_x, y_base, best_rot)
        top = max(top, best_y)
        # update skyline: insert segment [best_x, best_x + w) at height best_y
        var nx = List[Int]()
        var ny = List[Int]()
        var nw = List[Int]()
        var inserted = False
        for s in range(len(sx)):
            var s0 = sx[s]
            var s1 = sx[s] + sw[s]
            var r0 = best_x
            var r1 = best_x + w
            if s1 <= r0 or s0 >= r1:
                if s0 >= r1 and not inserted:
                    nx.append(r0); ny.append(best_y); nw.append(w); inserted = True
                nx.append(s0); ny.append(sy[s]); nw.append(sw[s])
                continue
            if s0 < r0:
                nx.append(s0); ny.append(sy[s]); nw.append(r0 - s0)
            if not inserted:
                nx.append(r0); ny.append(best_y); nw.append(w); inserted = True
            if s1 > r1:
                nx.append(r1); ny.append(sy[s]); nw.append(s1 - r1)
        if not inserted:
            nx.append(best_x); ny.append(best_y); nw.append(w)
        # merge equal-height neighbours
        sx.clear(); sy.clear(); sw.clear()
        for s in range(len(nx)):
            if len(sx) > 0 and sy[len(sy) - 1] == ny[s] and sx[len(sx) - 1] + sw[len(sw) - 1] == nx[s]:
                sw[len(sw) - 1] += nw[s]
            else:
                sx.append(nx[s]); sy.append(ny[s]); sw.append(nw[s])
    return top


# ------------------------------------------------------------------------------- driver

def unwrap_mesh(mesh: NativeMesh, cfg: UnwrapConfig) -> UnwrapResult:
    var res = UnwrapResult()
    var g = _FaceGraph(mesh)
    if g.nf == 0:
        res.mesh = mesh.copy()
        return res^
    var seg = _segment(g, cfg)
    var nseg = 0
    for f in range(g.nf):
        nseg = max(nseg, seg[f] + 1)
    var groups = List[List[Int]]()
    for _ in range(nseg):
        groups.append(List[Int]())
    for f in range(g.nf):
        groups[seg[f]].append(f)
    # validate / parameterise, splitting failures (work stack of face sets)
    var work = List[List[Int]]()
    for c in range(nseg - 1, -1, -1):
        if len(groups[c]) > 0:
            # growth can leave a chart split by non-manifold cuts: take components
            var lab = List[Int](length=g.nf, fill=0)
            var pieces = _components(g, groups[c], lab)
            for p in range(len(pieces)):
                work.append(pieces[p].copy())
    var charts = List[_Chart]()
    var slot = List[Int](length=g.nrep, fill=-1)
    while len(work) > 0:
        var faces = work.pop()
        var ch = _build_chart(g, faces, slot)
        var ok = len(faces) == 1 or _chart_is_disc(g, ch)
        if ok:
            _lscm(g, ch, cfg)
            if len(faces) > 1:
                var st = _chart_stretch(g, ch)
                ok = st.flipped == 0 and st.degenerate == 0 and st.l2 <= cfg.max_l2_stretch
                if ok:
                    ok = not _chart_self_overlaps(ch)
        if ok:
            charts.append(ch^)
        else:
            res.splits += 1
            var pieces = _bisect(g, faces)
            for p in range(len(pieces)):
                work.append(pieces[p].copy())
    # uniform texel density: scale every chart so uv area == 3D area, then rotate to min bbox
    var total_area: Float64 = 0
    for c in range(len(charts)):
        var a3: Float64 = 0
        var auv: Float64 = 0
        for i in range(len(charts[c].faces)):
            var f = charts[c].faces[i]
            a3 += Float64(g.fa[f])
            var l0 = charts[c].local[i * 3]
            var l1 = charts[c].local[i * 3 + 1]
            var l2 = charts[c].local[i * 3 + 2]
            auv += _orient(charts[c].u[l0], charts[c].v[l0], charts[c].u[l1], charts[c].v[l1], charts[c].u[l2], charts[c].v[l2]) * 0.5
        var s = sqrt(a3 / auv) if auv > 1e-30 and a3 > 0 else 1.0
        for i in range(len(charts[c].u)):
            charts[c].u[i] *= s
            charts[c].v[i] *= s
        _min_area_rotation(charts[c])
        total_area += a3
    # find the texel density that fills the atlas: bisection on the scale
    var R = cfg.resolution
    var pad = cfg.padding
    var nc = len(charts)
    var cw = List[Float64](length=nc, fill=0.0)
    var chh = List[Float64](length=nc, fill=0.0)
    for c in range(nc):
        for i in range(len(charts[c].u)):
            cw[c] = max(cw[c], charts[c].u[i])
            chh[c] = max(chh[c], charts[c].v[i])
    var lo_s: Float64 = 0
    var hi_s = sqrt(Float64(R) * Float64(R) / max(total_area, 1e-30))
    var best_s: Float64 = 0
    var best_place = List[_Placement]()
    var place = List[_Placement]()
    var ws = List[Int](length=nc, fill=0)
    var hs = List[Int](length=nc, fill=0)
    for _ in range(18):
        var s = (lo_s + hi_s) * 0.5 if best_s > 0 or lo_s > 0 else hi_s * 0.5
        for c in range(nc):
            ws[c] = Int(ceil(cw[c] * s)) + 1 + 2 * pad
            hs[c] = Int(ceil(chh[c] * s)) + 1 + 2 * pad
        var h = _skyline_pack(ws, hs, R, place)
        if h <= R:
            best_s = s
            best_place = place.copy()
            lo_s = s
        else:
            hi_s = s
    if best_s <= 0:
        # cannot fit even at tiny scale (too many charts for the padding): fall back to 1 texel/unit
        best_s = 1e-6
        for c in range(nc):
            ws[c] = 1 + 2 * pad
            hs[c] = 1 + 2 * pad
        _ = _skyline_pack(ws, hs, R * 64, best_place)
    # emit the output mesh with vertices split per chart
    var out = NativeMesh()
    copy_mesh_material(mesh, out)
    var nv = mesh.vertex_count()
    var has_n = len(mesh.normals) >= nv * 3
    var has_c = len(mesh.colors) >= nv * 4
    var has_j = len(mesh.joints) >= nv * 4
    var has_w = len(mesh.weights) >= nv * 4
    var has_ao = len(mesh.ao) >= nv
    var vmap = EdgeMap(nv * 2)
    res.chart_of_triangle = List[Int](length=g.nf, fill=-1)
    var tri_out = List[Int](length=g.nf * 3, fill=0)
    var uv_area: Float64 = 0
    var invR = 1.0 / Float64(R)
    for c in range(nc):
        var pl = best_place[c]
        for i in range(len(charts[c].faces)):
            var f = charts[c].faces[i]
            res.chart_of_triangle[f] = c
            for k in range(3):
                var ov = mesh.indices[f * 3 + k]
                var l = charts[c].local[i * 3 + k]
                var key = pair_key(ov, c)
                var id = vmap.insert(key, out.vertex_count())
                if id == out.vertex_count():
                    for q in range(3):
                        out.positions.append(mesh.positions[ov * 3 + q])
                    if has_n:
                        for q in range(3):
                            out.normals.append(mesh.normals[ov * 3 + q])
                    if has_c:
                        for q in range(4):
                            out.colors.append(mesh.colors[ov * 4 + q])
                    if has_j:
                        for q in range(4):
                            out.joints.append(mesh.joints[ov * 4 + q])
                    if has_w:
                        for q in range(4):
                            out.weights.append(mesh.weights[ov * 4 + q])
                    if has_ao:
                        out.ao.append(mesh.ao[ov])
                    var x = charts[c].u[l] * best_s
                    var y = charts[c].v[l] * best_s
                    if pl.rotated:
                        var t = x
                        x = chh[c] * best_s - y
                        y = t
                    out.texcoords.append(Float32((Float64(pl.x + pad) + 0.5 + x) * invR))
                    out.texcoords.append(Float32((Float64(pl.y + pad) + 0.5 + y) * invR))
                tri_out[f * 3 + k] = id
    for f in range(g.nf):
        var t0 = tri_out[f * 3]
        var t1 = tri_out[f * 3 + 1]
        var t2 = tri_out[f * 3 + 2]
        out.indices.append(t0)
        out.indices.append(t1)
        out.indices.append(t2)
        uv_area += abs(_orient(Float64(out.texcoords[t0 * 2]), Float64(out.texcoords[t0 * 2 + 1]),
                               Float64(out.texcoords[t1 * 2]), Float64(out.texcoords[t1 * 2 + 1]),
                               Float64(out.texcoords[t2 * 2]), Float64(out.texcoords[t2 * 2 + 1]))) * 0.5
    res.mesh = out^
    res.chart_count = nc
    res.resolution = R
    res.utilization = Float32(uv_area)
    return res^


# ------------------------------------------------------------------------------- validation

@fieldwise_init
struct UvReport(Copyable, ImplicitlyCopyable, Movable, Writable):
    var charts: Int
    var overlapping_pairs: Int
    var texel_overlaps: Int       # texels covered by two different charts (conservative raster)
    var out_of_range: Int
    var stretch: StretchStats
    var max_chart_l2: Float32

    def write_to[W: Writer](self, mut writer: W):
        writer.write("charts=", self.charts, " overlap_pairs=", self.overlapping_pairs, " texel_overlaps=", self.texel_overlaps,
                     " out_of_range=", self.out_of_range, " ", self.stretch, " max_chart_L2=", self.max_chart_l2)


def uv_report(mesh: NativeMesh, chart_of_triangle: List[Int], resolution: Int) -> UvReport:
    """Validates an atlas: triangle-pair UV overlaps, texel-level chart overlaps (each chart's
    triangles conservatively rasterised at `resolution`), and Sander stretch (global and per chart)."""
    var nt = mesh.triangle_count()
    var uv = List[Float64](capacity=nt * 6)
    var oor = 0
    for i in range(nt * 3):
        var v = mesh.indices[i]
        var u = Float64(mesh.texcoords[v * 2])
        var w = Float64(mesh.texcoords[v * 2 + 1])
        if u < 0 or u > 1 or w < 0 or w > 1:
            oor += 1
        uv.append(u)
        uv.append(w)
    var pairs = count_uv_overlaps(uv, nt, 1e-3)
    # conservative raster per chart
    var nchart = 0
    for t in range(nt):
        nchart = max(nchart, chart_of_triangle[t] + 1)
    var owner = List[Int](length=resolution * resolution, fill=-1)
    var texel_over = 0
    var R = Float64(resolution)
    for t in range(nt):
        var c = chart_of_triangle[t]
        var x0 = Int(floor(min(uv[t * 6], min(uv[t * 6 + 2], uv[t * 6 + 4])) * R))
        var x1 = Int(floor(max(uv[t * 6], max(uv[t * 6 + 2], uv[t * 6 + 4])) * R))
        var y0 = Int(floor(min(uv[t * 6 + 1], min(uv[t * 6 + 3], uv[t * 6 + 5])) * R))
        var y1 = Int(floor(max(uv[t * 6 + 1], max(uv[t * 6 + 3], uv[t * 6 + 5])) * R))
        for y in range(max(0, y0), min(resolution - 1, y1) + 1):
            for x in range(max(0, x0), min(resolution - 1, x1) + 1):
                if not _tri_touches_texel(uv, t, Float64(x) / R, Float64(y) / R, 1.0 / R):
                    continue
                var o = owner[y * resolution + x]
                if o < 0:
                    owner[y * resolution + x] = c
                elif o != c:
                    texel_over += 1
                    owner[y * resolution + x] = c
    # stretch, global and per chart
    var sum_a3: Float64 = 0
    var sum_auv: Float64 = 0
    var sum_l2: Float64 = 0
    var linf: Float64 = 0
    var flipped = 0
    var degenerate = 0
    var c_a3 = List[Float64](length=nchart, fill=0.0)
    var c_auv = List[Float64](length=nchart, fill=0.0)
    var c_l2 = List[Float64](length=nchart, fill=0.0)
    var linfs = List[Float64]()
    for t in range(nt):
        var i0 = mesh.indices[t * 3]
        var i1 = mesh.indices[t * 3 + 1]
        var i2 = mesh.indices[t * 3 + 2]
        var q1 = vpos(mesh, i0)
        var q2 = vpos(mesh, i1)
        var q3 = vpos(mesh, i2)
        var a3 = Float64((q2 - q1).cross(q3 - q1).length() * 0.5)
        if a3 <= 1e-30:
            continue
        var r = _stretch_tri(q1, q2, q3, uv[t * 6], uv[t * 6 + 1], uv[t * 6 + 2], uv[t * 6 + 3], uv[t * 6 + 4], uv[t * 6 + 5])
        if r[0] <= 0:
            if r[0] == 0:
                degenerate += 1
            else:
                flipped += 1
            continue
        sum_a3 += a3
        sum_auv += r[0]
        sum_l2 += r[1] * a3
        linfs.append(r[2])
        var c = chart_of_triangle[t]
        c_a3[c] += a3
        c_auv[c] += r[0]
        c_l2[c] += r[1] * a3
    var norm = sqrt(sum_auv / max(sum_a3, 1e-30))
    for i in range(len(linfs)):
        linf = max(linf, linfs[i])
    var max_chart: Float64 = 0
    for c in range(nchart):
        if c_a3[c] > 0 and c_auv[c] > 0:
            # each chart normalised by its own area ratio (charts share one texel density, so
            # the global norm and the chart norm agree up to LSCM's local scale drift)
            max_chart = max(max_chart, sqrt(c_l2[c] / c_a3[c]) * sqrt(c_auv[c] / c_a3[c]))
    var st = StretchStats(Float32(sqrt(sum_l2 / max(sum_a3, 1e-30)) * norm), Float32(linf * norm), flipped, degenerate)
    return UvReport(nchart, pairs, texel_over, oor, st, Float32(max_chart))


def _tri_touches_texel(uv: List[Float64], t: Int, x0: Float64, y0: Float64, s: Float64) -> Bool:
    """Conservative triangle / axis-aligned square overlap (SAT on the square axes + triangle edges)."""
    var x1 = x0 + s
    var y1 = y0 + s
    for k in range(3):
        var ax = uv[t * 6 + k * 2]
        var ay = uv[t * 6 + k * 2 + 1]
        var bx = uv[t * 6 + ((k + 1) % 3) * 2]
        var by = uv[t * 6 + ((k + 1) % 3) * 2 + 1]
        var cx = uv[t * 6 + ((k + 2) % 3) * 2]
        var cy = uv[t * 6 + ((k + 2) % 3) * 2 + 1]
        var nx = -(by - ay)
        var ny = bx - ax
        var side_c = (cx - ax) * nx + (cy - ay) * ny
        # all four corners strictly on the other side of edge ab than c -> separated
        var sep = True
        for corner in range(4):
            var px = x0 if (corner & 1) == 0 else x1
            var py = y0 if (corner & 2) == 0 else y1
            var d = (px - ax) * nx + (py - ay) * ny
            if d * side_c >= 0:
                sep = False
        if sep and side_c != 0:
            return False
    return True
