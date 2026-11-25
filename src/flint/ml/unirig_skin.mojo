"""UniRig skinning stage (`flint.ml.unirig_skin`): per-vertex skin weights for a mesh and a skeleton.

Pure Mojo except the three network graphs, which `flint.ml.onnx` runs (matrix products on a MAX
GPU when present):

  1. Normalise mesh + skeleton into [-1, 1]^3 (bbox of vertices and joints).
  2. `voxel_skin`: a geodesic prior. The mesh is voxelised from six orthographic depth images
     (a small software rasteriser reproducing UniRig's pyrender set-up), mesh vertices and voxel
     centres form one graph, and Dijkstra from every joint gives distances d; the prior is
     (1 / ((1 - a) d + a d^2))^2, normalised over joints.
  3. 32768 surface samples (8192 vertices, the rest area^2-weighted on faces), their normals and
     the interpolated prior.
  4. Point Transformer V3 input: 5 mm voxel grid coordinates, the four space-filling-curve orders
     (z, z-trans, Hilbert, Hilbert-trans, shuffled) and sub-manifold convolution neighbour tables.
  5. Networks: Michelangelo global latents, the PTv3 + bone body, and the per-vertex skin head; the
     prediction is multiplied by (prior summed into parents)^3 and renormalised.
  6. `reskin`: back onto the original vertices (median of the 7 nearest samples, one diffusion
     step along mesh edges, weights under 0.03 dropped).
"""

from std.math import sqrt, exp, abs
from std.collections import Dict
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import Tensor, DT_I64
from flint.ml.onnx import OnnxModel
from flint.ml.unirig import UrRng

comptime UR_SKIN_SAMPLES: Int = 32768
comptime UR_SKIN_VERTEX_SAMPLES: Int = 8192
comptime UR_VOXEL_GRID: Int = 196
comptime UR_PTV3_GRID: Float32 = 0.005
comptime _BIG: Float64 = 1.0e300


# --------------------------------------------------------------------------- spatial grid (kNN)

struct UrPointGrid(Movable):
    """Uniform bucket grid over a point set for k-nearest and radius queries."""
    var pts: List[Float32]
    var n: Int
    var lo: List[Float32]
    var cell: Float32
    var dim: List[Int]
    var start: List[Int]
    var items: List[Int]

    def __init__(out self, pts: List[Float32], per_cell: Int = 4):
        self.pts = pts.copy()
        self.n = len(pts) // 3
        self.lo = List[Float32](length=3, fill=3.4e38)
        var hi = List[Float32](length=3, fill=-3.4e38)
        for i in range(self.n):
            for c in range(3):
                self.lo[c] = min(self.lo[c], pts[i * 3 + c])
                hi[c] = max(hi[c], pts[i * 3 + c])
        var ext = max(max(hi[0] - self.lo[0], hi[1] - self.lo[1]), hi[2] - self.lo[2])
        if ext <= 0:
            ext = 1.0
        var cells_per_axis = max(1, min(256, Int((Float64(self.n) / Float64(per_cell)) ** (1.0 / 3.0)) + 1))
        self.cell = ext / Float32(cells_per_axis) * 1.0001
        self.dim = List[Int]()
        for c in range(3):
            self.dim.append(max(1, Int((hi[c] - self.lo[c]) / self.cell) + 1))
        self.start = List[Int]()
        self.items = List[Int]()
        var total = self.dim[0] * self.dim[1] * self.dim[2]
        var count = List[Int](length=total + 1, fill=0)
        var cid = List[Int](length=self.n, fill=0)
        for i in range(self.n):
            var k = self._cell_of(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2])
            cid[i] = k
            count[k + 1] += 1
        for k in range(total):
            count[k + 1] += count[k]
        self.start = count.copy()
        self.items = List[Int](length=self.n, fill=0)
        var fillp = count^
        for i in range(self.n):
            self.items[fillp[cid[i]]] = i
            fillp[cid[i]] += 1

    @always_inline
    def _axis(self, v: Float32, c: Int) -> Int:
        var a = Int((v - self.lo[c]) / self.cell)
        return max(0, min(self.dim[c] - 1, a))

    @always_inline
    def _cell_of(self, x: Float32, y: Float32, z: Float32) -> Int:
        return (self._axis(x, 0) * self.dim[1] + self._axis(y, 1)) * self.dim[2] + self._axis(z, 2)

    def knn(self, x: Float32, y: Float32, z: Float32, k: Int, mut idx: List[Int], mut dist: List[Float64]):
        """The k nearest points (ascending distance, lower index first on ties)."""
        idx.clear()
        dist.clear()
        var cx = self._axis(x, 0)
        var cy = self._axis(y, 1)
        var cz = self._axis(z, 2)
        var maxr = max(max(self.dim[0], self.dim[1]), self.dim[2])
        var r = 0
        while r <= maxr:
            for ix in range(max(0, cx - r), min(self.dim[0], cx + r + 1)):
                for iy in range(max(0, cy - r), min(self.dim[1], cy + r + 1)):
                    for iz in range(max(0, cz - r), min(self.dim[2], cz + r + 1)):
                        if max(max(abs(ix - cx), abs(iy - cy)), abs(iz - cz)) != r:
                            continue
                        var c = (ix * self.dim[1] + iy) * self.dim[2] + iz
                        for s in range(self.start[c], self.start[c + 1]):
                            var p = self.items[s]
                            var dx = Float64(self.pts[p * 3]) - Float64(x)
                            var dy = Float64(self.pts[p * 3 + 1]) - Float64(y)
                            var dz = Float64(self.pts[p * 3 + 2]) - Float64(z)
                            var d = sqrt(dx * dx + dy * dy + dz * dz)
                            if len(idx) == k and (d > dist[k - 1] or (d == dist[k - 1] and p > idx[k - 1])):
                                continue
                            # insert sorted
                            var pos = len(idx)
                            if len(idx) < k:
                                idx.append(p)
                                dist.append(d)
                            else:
                                pos = k - 1
                                idx[pos] = p
                                dist[pos] = d
                            while pos > 0 and (dist[pos - 1] > dist[pos] or (dist[pos - 1] == dist[pos] and idx[pos - 1] > idx[pos])):
                                var ti = idx[pos - 1]
                                idx[pos - 1] = idx[pos]
                                idx[pos] = ti
                                var td = dist[pos - 1]
                                dist[pos - 1] = dist[pos]
                                dist[pos] = td
                                pos -= 1
            # every point outside the searched cube is at least r * cell away
            if len(idx) == k and dist[k - 1] <= Float64(r) * Float64(self.cell):
                break
            if len(idx) == self.n:
                break
            r += 1


# --------------------------------------------------------------------------- mesh helpers

def normals_of(v: List[Float32], f: List[Int], mut out: List[Float32], mut fnrm: List[Float32]):
    """Area-weighted vertex normals `out` [V, 3] and unit face normals `fnrm` [F, 3]."""
    var nv = len(v) // 3
    var nf = len(f) // 3
    var vn = List[Float64](length=nv * 3, fill=0.0)
    fnrm = List[Float32](length=nf * 3, fill=0.0)
    for t in range(nf):
        var a = f[t * 3]
        var b = f[t * 3 + 1]
        var c = f[t * 3 + 2]
        var e1x = Float64(v[b * 3]) - Float64(v[a * 3])
        var e1y = Float64(v[b * 3 + 1]) - Float64(v[a * 3 + 1])
        var e1z = Float64(v[b * 3 + 2]) - Float64(v[a * 3 + 2])
        var e2x = Float64(v[c * 3]) - Float64(v[a * 3])
        var e2y = Float64(v[c * 3 + 1]) - Float64(v[a * 3 + 1])
        var e2z = Float64(v[c * 3 + 2]) - Float64(v[a * 3 + 2])
        var nx = e1y * e2z - e1z * e2y
        var ny = e1z * e2x - e1x * e2z
        var nz = e1x * e2y - e1y * e2x
        for k in range(3):
            var vi = f[t * 3 + k]
            vn[vi * 3] += nx
            vn[vi * 3 + 1] += ny
            vn[vi * 3 + 2] += nz
        var l = max(sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
        fnrm[t * 3] = Float32(nx / l)
        fnrm[t * 3 + 1] = Float32(ny / l)
        fnrm[t * 3 + 2] = Float32(nz / l)
    out = List[Float32](length=nv * 3, fill=0.0)
    for i in range(nv):
        var l = max(sqrt(vn[i * 3] ** 2 + vn[i * 3 + 1] ** 2 + vn[i * 3 + 2] ** 2), 1e-12)
        for c in range(3):
            out[i * 3 + c] = Float32(vn[i * 3 + c] / l)


struct UrSamples(Movable):
    """Surface samples: position, normal and where they came from (vertex id, or face + barycentric
    lengths r0, r1 with p = v0 + r0 (v1 - v0) + r1 (v2 - v0))."""
    var xyz: List[Float32]
    var nrm: List[Float32]
    var vert: List[Int]       # >= 0 for vertex samples, -1 for face samples
    var face: List[Int]
    var r0: List[Float64]
    var r1: List[Float64]

    def __init__(out self):
        self.xyz = List[Float32]()
        self.nrm = List[Float32]()
        self.vert = List[Int]()
        self.face = List[Int]()
        self.r0 = List[Float64]()
        self.r1 = List[Float64]()

    def count(self) -> Int:
        return len(self.vert)

    def interpolate(self, attr: List[Float32], width: Int, f: List[Int]) -> List[Float32]:
        """A per-vertex attribute [V, width] at every sample."""
        var n = self.count()
        var out = List[Float32](length=n * width, fill=0.0)
        for s in range(n):
            if self.vert[s] >= 0:
                for c in range(width):
                    out[s * width + c] = attr[self.vert[s] * width + c]
            else:
                var t = self.face[s]
                var a = f[t * 3]
                var b = f[t * 3 + 1]
                var cc = f[t * 3 + 2]
                for c in range(width):
                    var o = Float64(attr[a * width + c])
                    out[s * width + c] = Float32(o + (Float64(attr[b * width + c]) - o) * self.r0[s]
                                                 + (Float64(attr[cc * width + c]) - o) * self.r1[s])
        return out^


def sample_mesh(v: List[Float32], f: List[Int], vn: List[Float32], fnrm: List[Float32], num: Int, vertex_num: Int,
                mut rng: UrRng) -> UrSamples:
    """UniRig's SamplerMix: `vertex_num` random distinct vertices, then faces picked with weight
    |cross(e1, e2)|^2 (UniRig's choice) and a uniform point on each."""
    var s = UrSamples()
    var nv = len(v) // 3
    var nf = len(f) // 3
    var perm = List[Int](length=nv, fill=0)
    for i in range(nv):
        perm[i] = i
    var take = min(vertex_num, nv)
    for i in range(take):
        var j = i + rng.below(nv - i)
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
        var p = perm[i]
        s.vert.append(p)
        s.face.append(-1)
        s.r0.append(0.0)
        s.r1.append(0.0)
        for c in range(3):
            s.xyz.append(v[p * 3 + c])
            s.nrm.append(vn[p * 3 + c])
    var cum = List[Float64](length=nf, fill=0.0)
    var acc: Float64 = 0
    for t in range(nf):
        var a = f[t * 3]
        var b = f[t * 3 + 1]
        var c = f[t * 3 + 2]
        var e1x = v[b * 3] - v[a * 3]
        var e1y = v[b * 3 + 1] - v[a * 3 + 1]
        var e1z = v[b * 3 + 2] - v[a * 3 + 2]
        var e2x = v[c * 3] - v[a * 3]
        var e2y = v[c * 3 + 1] - v[a * 3 + 1]
        var e2z = v[c * 3 + 2] - v[a * 3 + 2]
        var nx = e1y * e2z - e1z * e2y
        var ny = e1z * e2x - e1x * e2z
        var nz = e1x * e2y - e1y * e2x
        acc += Float64(nx * nx + ny * ny + nz * nz)
        cum[t] = acc
    for _ in range(num - take):
        var pick = rng.uniform() * acc
        var lo = 0
        var hi = nf - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cum[mid] < pick:
                lo = mid + 1
            else:
                hi = mid
        var t = lo
        var r0 = rng.uniform()
        var r1 = rng.uniform()
        if r0 + r1 > 1.0:
            r0 = 1.0 - r0
            r1 = 1.0 - r1
        var a = f[t * 3]
        var b = f[t * 3 + 1]
        var c = f[t * 3 + 2]
        s.vert.append(-1)
        s.face.append(t)
        s.r0.append(r0)
        s.r1.append(r1)
        for k in range(3):
            var o = Float64(v[a * 3 + k])
            s.xyz.append(Float32(o + (Float64(v[b * 3 + k]) - o) * r0 + (Float64(v[c * 3 + k]) - o) * r1))
            s.nrm.append(fnrm[t * 3 + k])
    return s^


def farthest_point_sample(pts: List[Float32], stride: Int, m: Int) -> List[Int]:
    """torch_cluster.fps without a random start: first point, then the farthest from those chosen."""
    var n = len(pts) // stride
    var out = List[Int]()
    var best = List[Float32](length=n, fill=3.4e38)
    var cur = 0
    for _ in range(min(m, n)):
        out.append(cur)
        var nxt = 0
        var far: Float32 = -1
        for i in range(n):
            var dx = pts[i * stride] - pts[cur * stride]
            var dy = pts[i * stride + 1] - pts[cur * stride + 1]
            var dz = pts[i * stride + 2] - pts[cur * stride + 2]
            var d = dx * dx + dy * dy + dz * dz
            if d < best[i]:
                best[i] = d
            if best[i] > far:
                far = best[i]
                nxt = i
        cur = nxt
    return out^


def michelangelo_queries(xyz: List[Float32], nrm: List[Float32], mut rng: UrRng, tokens: Int = 1024) -> List[Float32]:
    """The encoder's query points: 4 x `tokens` random samples, then FPS down to `tokens` ([tokens, 6])."""
    var n = len(xyz) // 3
    var pre = tokens * 4
    var pick = List[Int]()
    if pre <= n:
        var perm = List[Int](length=n, fill=0)
        for i in range(n):
            perm[i] = i
        for i in range(pre):
            var j = i + rng.below(n - i)
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
            pick.append(perm[i])
    else:
        for _ in range(pre):
            pick.append(rng.below(n))
    var sub = List[Float32](length=pre * 6, fill=0.0)
    for i in range(pre):
        for c in range(3):
            sub[i * 6 + c] = xyz[pick[i] * 3 + c]
            sub[i * 6 + 3 + c] = nrm[pick[i] * 3 + c]
    var idx = farthest_point_sample(sub, 6, tokens)
    var out = List[Float32](length=len(idx) * 6, fill=0.0)
    for i in range(len(idx)):
        for c in range(6):
            out[i * 6 + c] = sub[idx[i] * 6 + c]
    return out^


# --------------------------------------------------------------------------- voxelisation

def _depth_image(v: List[Float32], f: List[Int], R: List[Float32], t: List[Float32], g: Int,
                 ox: Float32 = 0.375, oy: Float32 = 0.125) -> List[Float32]:
    """Orthographic (xmag = ymag = 1) depth image of the double-sided mesh seen by a camera with
    camera-to-world rotation R (row major) and position t; row 0 is the top row, like pyrender's
    read-back. Uncovered pixels (or depth outside [0.05, 4]) read 4.0 (zfar). Each pixel is sampled
    at window offset (ox, oy) = (0.375, 0.125): pyrender renders 4x MSAA and blits depth with
    GL_NEAREST, which keeps sample 0 of the standard pattern (measured against pyrender: median
    depth difference 8e-6 and identical coverage)."""
    var znear: Float32 = 0.05
    var zfar: Float32 = 4.0
    var img = List[Float32](length=g * g, fill=zfar)
    var nv = len(v) // 3
    var sx = List[Float32](length=nv, fill=0.0)
    var sy = List[Float32](length=nv, fill=0.0)
    var sz = List[Float32](length=nv, fill=0.0)
    for i in range(nv):
        var px = v[i * 3] - t[0]
        var py = v[i * 3 + 1] - t[1]
        var pz = v[i * 3 + 2] - t[2]
        # eye = R^T (p - t)
        var ex = R[0] * px + R[3] * py + R[6] * pz
        var ey = R[1] * px + R[4] * py + R[7] * pz
        var ez = R[2] * px + R[5] * py + R[8] * pz
        sx[i] = (ex + 1.0) * 0.5 * Float32(g)
        sy[i] = (ey + 1.0) * 0.5 * Float32(g)
        sz[i] = -ez
    var nf = len(f) // 3
    for tr in range(nf):
        var a = f[tr * 3]
        var b = f[tr * 3 + 1]
        var c = f[tr * 3 + 2]
        var x0 = sx[a]
        var y0 = sy[a]
        var x1 = sx[b]
        var y1 = sy[b]
        var x2 = sx[c]
        var y2 = sy[c]
        var area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0:
            continue
        var minx = max(0, Int(min(min(x0, x1), x2) - 0.5))
        var maxx = min(g - 1, Int(max(max(x0, x1), x2) + 0.5))
        var miny = max(0, Int(min(min(y0, y1), y2) - 0.5))
        var maxy = min(g - 1, Int(max(max(y0, y1), y2) + 0.5))
        var inv = 1.0 / area
        for py in range(miny, maxy + 1):
            var cy = Float32(py) + oy
            for px in range(minx, maxx + 1):
                var cx = Float32(px) + ox
                var w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
                var w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
                var w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                var d = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
                if d < znear or d > zfar:
                    continue
                var row = g - 1 - py
                if d < img[row * g + px]:
                    img[row * g + px] = d
    return img^


def voxelize(v: List[Float32], f: List[Int], g: Int) -> List[Float32]:
    """Voxel centres [M, 3] inside the mesh (UniRig `voxelization`, pyrender backend): a voxel is
    kept when it lies behind the surface along two of the three axes from both sides."""
    var eye: Float32 = 2.0
    # camera poses: rotation (row major) + position
    var Rs = List[List[Float32]]()
    var ts = List[List[Float32]]()
    Rs.append([1, 0, 0, 0, 1, 0, 0, 0, 1]); ts.append([0, 0, eye])            # +z
    Rs.append([-1, 0, 0, 0, 1, 0, 0, 0, -1]); ts.append([0, 0, -eye])         # -z
    Rs.append([1, 0, 0, 0, 0, -1, 0, 1, 0]); ts.append([0, -eye, 0])          # +y
    Rs.append([1, 0, 0, 0, 0, 1, 0, -1, 0]); ts.append([0, eye, 0])           # -y
    Rs.append([0, 0, -1, 0, 1, 0, 1, 0, 0]); ts.append([-eye, 0, 0])          # +x
    Rs.append([0, 0, 1, 0, 1, 0, -1, 0, 0]); ts.append([eye, 0, 0])           # -x
    var D = List[List[Float32]]()
    for k in range(6):
        D.append(_depth_image(v, f, Rs[k], ts[k], g))
    var out = List[Float32]()
    var inv_g = Float32(1.0 / Float64(g))
    for i in range(g):
        var gx = Float32(i * 2) / Float32(g) - 1.0 + inv_g
        for j in range(g):
            var gy = Float32(j * 2) / Float32(g) - 1.0 + inv_g
            for k in range(g):
                var gz = Float32((g - 1 - k) * 2) / Float32(g) - 1.0 + inv_g
                var mz = (-gz + D[0][(g - 1 - j) * g + i] <= eye) and (gz + D[1][(g - 1 - j) * g + (g - 1 - i)] <= eye)
                var mx = (-gx + D[4][(g - 1 - j) * g + (g - 1 - k)] <= eye) and (gx + D[5][(g - 1 - j) * g + k] <= eye)
                var my = (-gy + D[2][k * g + i] <= eye) and (gy + D[3][(g - 1 - k) * g + i] <= eye)
                if (mx and my) or (mx and mz) or (my and mz):
                    out.append(gx)
                    out.append(gy)
                    out.append(gz)
    return out^


# --------------------------------------------------------------------------- voxel skin (geodesic prior)

def _add_edge(mut key_of: Dict[Int, Int], mut es: List[Int], mut ed: List[Int], mut ew: List[Float64], nodes: Int,
              s: Int, d: Int, w: Float64) raises:
    """Adds (s -> d, w); a repeated (s, d) adds its weight, like scipy's csr_matrix."""
    var key = s * nodes + d
    var hit = key_of.get(key)
    if hit:
        ew[hit.value()] += w
    else:
        key_of[key] = len(es)
        es.append(s)
        ed.append(d)
        ew.append(w)


struct _DijkstraCtx(Movable):
    var adj_start: Int     # Int pointers into the CSR adjacency
    var adj_to: Int
    var adj_w: Int
    var nodes: Int
    var sources: Int       # Int list of source node per job
    var out: Int           # Float64 [jobs, keep] distances of the first `keep` nodes
    var keep: Int

    def __init__(out self):
        self.adj_start = 0
        self.adj_to = 0
        self.adj_w = 0
        self.nodes = 0
        self.sources = 0
        self.out = 0
        self.keep = 0


def _dijkstra_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_DijkstraCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var n = c[].nodes
    var st = Pointer[Int, MutAnyOrigin](unsafe_from_address=c[].adj_start)
    var to = Pointer[Int, MutAnyOrigin](unsafe_from_address=c[].adj_to)
    var wt = Pointer[Float64, MutAnyOrigin](unsafe_from_address=c[].adj_w)
    var src = Pointer[Int, MutAnyOrigin](unsafe_from_address=c[].sources).unsafe_offset(job).unsafe_load()
    var dist = List[Float64](length=n, fill=_BIG)
    var done = List[Bool](length=n, fill=False)
    # binary heap of (dist, node)
    var hd = List[Float64]()
    var hn = List[Int]()
    dist[src] = 0
    hd.append(0.0)
    hn.append(src)
    while len(hd) > 0:
        var d = hd[0]
        var u = hn[0]
        var last = len(hd) - 1
        hd[0] = hd[last]
        hn[0] = hn[last]
        _ = hd.pop()
        _ = hn.pop()
        var i = 0
        var m = len(hd)
        while True:
            var l = 2 * i + 1
            if l >= m:
                break
            var r = l + 1
            var s = l
            if r < m and hd[r] < hd[l]:
                s = r
            if hd[s] < hd[i]:
                var td = hd[s]
                hd[s] = hd[i]
                hd[i] = td
                var tn = hn[s]
                hn[s] = hn[i]
                hn[i] = tn
                i = s
            else:
                break
        if done[u] or d > dist[u]:
            continue
        done[u] = True
        for e in range(st.unsafe_offset(u).unsafe_load(), st.unsafe_offset(u + 1).unsafe_load()):
            var w = to.unsafe_offset(e).unsafe_load()
            var nd = d + wt.unsafe_offset(e).unsafe_load()
            if nd < dist[w]:
                dist[w] = nd
                hd.append(nd)
                hn.append(w)
                var k = len(hd) - 1
                while k > 0:
                    var p = (k - 1) // 2
                    if hd[k] < hd[p]:
                        var td = hd[k]
                        hd[k] = hd[p]
                        hd[p] = td
                        var tn = hn[k]
                        hn[k] = hn[p]
                        hn[p] = tn
                        k = p
                    else:
                        break
    var o = Pointer[Float64, MutAnyOrigin](unsafe_from_address=c[].out + job * c[].keep * 8)
    for k in range(c[].keep):
        o.unsafe_offset(k).unsafe_store(dist[k])


def voxel_skin(mut pool: JobPool, v_in: List[Float32], f: List[Int], joints_in: List[Float32], g: Int = UR_VOXEL_GRID,
               alpha: Float64 = 0.5, link_dis: Float64 = 0.00001, grid_weight: Float64 = 3.0) raises -> List[Float32]:
    """UniRig `VertexGroupVoxelSkin` (grid 196, alpha 0.5, grid_query 7, vertex_query 1): [V, J]."""
    var nv = len(v_in) // 3
    var J = len(joints_in) // 3
    # normalise by the vertex bbox (float32, like numpy on float32 arrays)
    var lo = List[Float32](length=3, fill=3.4e38)
    var hi = List[Float32](length=3, fill=-3.4e38)
    for i in range(nv):
        for c in range(3):
            lo[c] = min(lo[c], v_in[i * 3 + c])
            hi[c] = max(hi[c], v_in[i * 3 + c])
    var scale = max(max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]) / 2
    var v = List[Float32](length=nv * 3, fill=0.0)
    for i in range(nv):
        for c in range(3):
            v[i * 3 + c] = (v_in[i * 3 + c] - (lo[c] + hi[c]) / 2) / scale
    var joints = List[Float32](length=J * 3, fill=0.0)
    for j in range(J):
        for c in range(3):
            joints[j * 3 + c] = (joints_in[j * 3 + c] - (lo[c] + hi[c]) / 2) / scale
    var vox = voxelize(v, f, g)
    var M = len(vox) // 3
    var nodes = nv + M
    var step = 2.0 / Float64(g) * 1.001
    # ---- edges (summing duplicate (src, dst) like scipy's csr_matrix)
    var key_of = Dict[Int, Int]()
    var es = List[Int]()
    var ed = List[Int]()
    var ew = List[Float64]()

    # close vertices (3 nearest others, 0 < d < link_dis)
    var vgrid = UrPointGrid(v)
    var qi = List[Int]()
    var qd = List[Float64]()
    for i in range(nv):
        vgrid.knn(v[i * 3], v[i * 3 + 1], v[i * 3 + 2], 4, qi, qd)
        for s in range(1, len(qi)):
            if qd[s] > 0 and qd[s] < link_dis * 1.001:
                _add_edge(key_of, es, ed, ew, nodes, i, qi[s], qd[s])
    # mesh edges (float32 lengths)
    var nf = len(f) // 3
    for k in range(3):
        for t in range(nf):
            var a = f[t * 3 + k]
            var b = f[t * 3 + (k + 1) % 3]
            var dx = v[a * 3] - v[b * 3]
            var dy = v[a * 3 + 1] - v[b * 3 + 1]
            var dz = v[a * 3 + 2] - v[b * 3 + 2]
            _add_edge(key_of, es, ed, ew, nodes, a, b, Float64(sqrt(dx * dx + dy * dy + dz * dz)))
    # voxel <-> face-adjacent voxels (weight x grid_weight), via a dense index of the voxel grid
    var cell_index = Dict[Int, Int]()
    for m in range(M):
        var ix = Int((vox[m * 3] + 1.0) * Float32(g) / 2.0)
        var iy = Int((vox[m * 3 + 1] + 1.0) * Float32(g) / 2.0)
        var iz = Int((vox[m * 3 + 2] + 1.0) * Float32(g) / 2.0)
        cell_index[(ix * g + iy) * g + iz] = m
    for m in range(M):
        var ix = Int((vox[m * 3] + 1.0) * Float32(g) / 2.0)
        var iy = Int((vox[m * 3 + 1] + 1.0) * Float32(g) / 2.0)
        var iz = Int((vox[m * 3 + 2] + 1.0) * Float32(g) / 2.0)
        for a in range(6):
            var jx = ix + (1 if a == 0 else (-1 if a == 1 else 0))
            var jy = iy + (1 if a == 2 else (-1 if a == 3 else 0))
            var jz = iz + (1 if a == 4 else (-1 if a == 5 else 0))
            if jx < 0 or jy < 0 or jz < 0 or jx >= g or jy >= g or jz >= g:
                continue
            var other = cell_index.get((jx * g + jy) * g + jz)
            if other:
                var o = other.value()
                var dx = Float64(vox[m * 3]) - Float64(vox[o * 3])
                var dy = Float64(vox[m * 3 + 1]) - Float64(vox[o * 3 + 1])
                var dz = Float64(vox[m * 3 + 2]) - Float64(vox[o * 3 + 2])
                var d = sqrt(dx * dx + dy * dy + dz * dz)
                if d > 0 and d < step:
                    _add_edge(key_of, es, ed, ew, nodes, nv + m, nv + o, d * grid_weight)
    # voxel -> nearest mesh vertex
    for m in range(M):
        vgrid.knn(vox[m * 3], vox[m * 3 + 1], vox[m * 3 + 2], 1, qi, qd)
        if len(qi) == 1 and qd[0] > 0 and qd[0] < step:
            _add_edge(key_of, es, ed, ew, nodes, nv + m, qi[0], qd[0])
    # ---- undirected CSR
    var deg = List[Int](length=nodes + 1, fill=0)
    for e in range(len(es)):
        deg[es[e] + 1] += 1
        deg[ed[e] + 1] += 1
    for k in range(nodes):
        deg[k + 1] += deg[k]
    var adj_to = List[Int](length=deg[nodes], fill=0)
    var adj_w = List[Float64](length=deg[nodes], fill=0.0)
    var fillp = deg.copy()
    for e in range(len(es)):
        adj_to[fillp[es[e]]] = ed[e]
        adj_w[fillp[es[e]]] = ew[e]
        fillp[es[e]] += 1
        adj_to[fillp[ed[e]]] = es[e]
        adj_w[fillp[ed[e]]] = ew[e]
        fillp[ed[e]] += 1
    # ---- joint -> nearest node
    var sources = List[Int](length=J, fill=0)
    for j in range(J):
        var best = _BIG
        for k in range(nodes):
            var dx: Float64
            var dy: Float64
            var dz: Float64
            if k < nv:
                dx = Float64(v[k * 3]) - Float64(joints[j * 3])
                dy = Float64(v[k * 3 + 1]) - Float64(joints[j * 3 + 1])
                dz = Float64(v[k * 3 + 2]) - Float64(joints[j * 3 + 2])
            else:
                dx = Float64(vox[(k - nv) * 3]) - Float64(joints[j * 3])
                dy = Float64(vox[(k - nv) * 3 + 1]) - Float64(joints[j * 3 + 1])
                dz = Float64(vox[(k - nv) * 3 + 2]) - Float64(joints[j * 3 + 2])
            var d = dx * dx + dy * dy + dz * dz
            if d < best:
                best = d
                sources[j] = k
    var dist = List[Float64](length=J * nv, fill=0.0)
    var ctx = _DijkstraCtx()
    ctx.adj_start = Int(deg.unsafe_ptr())
    ctx.adj_to = Int(adj_to.unsafe_ptr())
    ctx.adj_w = Int(adj_w.unsafe_ptr())
    ctx.nodes = nodes
    ctx.sources = Int(sources.unsafe_ptr())
    ctx.out = Int(dist.unsafe_ptr())
    ctx.keep = nv
    run_parallel_ctx(pool, J, ctx, _dijkstra_worker)
    _ = deg^
    _ = adj_to^
    _ = adj_w^
    _ = sources^
    # ---- unreachable vertices: Euclidean distance to the min(J, 3) nearest joints
    var kj = min(J, 3)
    for i in range(nv):
        var reach = False
        for j in range(J):
            if dist[j * nv + i] < _BIG:
                reach = True
                break
        if reach:
            continue
        var dj = List[Float64](length=J, fill=0.0)
        var used = List[Bool](length=J, fill=False)
        for j in range(J):
            var dx = Float64(v[i * 3]) - Float64(joints[j * 3])
            var dy = Float64(v[i * 3 + 1]) - Float64(joints[j * 3 + 1])
            var dz = Float64(v[i * 3 + 2]) - Float64(joints[j * 3 + 2])
            dj[j] = sqrt(dx * dx + dy * dy + dz * dz)
        for _ in range(kj):
            var b = -1
            for j in range(J):
                if not used[j] and (b < 0 or dj[j] < dj[b]):
                    b = j
            used[b] = True
            dist[b * nv + i] = dj[b]
    var max_dis: Float64 = 0
    for k in range(len(dist)):
        if dist[k] < _BIG:
            max_dis = max(max_dis, dist[k])
    var out = List[Float32](length=nv * J, fill=0.0)
    for i in range(nv):
        var tot: Float64 = 0
        var w = List[Float64](length=J, fill=0.0)
        for j in range(J):
            var d = dist[j * nv + i]
            if d >= _BIG:
                d = max_dis
            d = max(d, 1e-6)
            var q = 1.0 / ((1.0 - alpha) * d + alpha * d * d)
            w[j] = q * q
            tot += w[j]
        for j in range(J):
            out[i * J + j] = Float32(w[j] / tot)
    return out^


# --------------------------------------------------------------------------- PTv3 serialisation

@always_inline
def _zorder(x: Int, y: Int, z: Int, depth: Int) -> Int:
    var key = 0
    for i in range(depth):
        var m = 1 << i
        key |= ((x & m) << (2 * i + 2)) | ((y & m) << (2 * i + 1)) | ((z & m) << (2 * i))
    return key


def _hilbert(x: Int, y: Int, z: Int, nb: Int) -> Int:
    """Skilling's transpose-based Hilbert index, as Pointcept's vectorised `hilbert.encode`."""
    var g = List[Int](length=3 * nb, fill=0)   # g[dim * nb + bit], bit 0 = most significant
    var c = List[Int]()
    c.append(x)
    c.append(y)
    c.append(z)
    for d in range(3):
        for b in range(nb):
            g[d * nb + b] = (c[d] >> (nb - 1 - b)) & 1
    for bit in range(nb):
        for dim in range(3):
            var mask = g[dim * nb + bit]
            for b in range(bit + 1, nb):
                g[b] ^= mask
                var flip = (1 - mask) & (g[b] ^ g[dim * nb + b])
                g[dim * nb + b] ^= flip
                g[b] ^= flip
    var code = 0
    var acc = 0
    for bit in range(nb):
        for dim in range(3):
            acc ^= g[dim * nb + bit]
            code = (code << 1) | acc
    return code


struct UrSerialized(Movable):
    var grid: List[Int]       # [N, 3] grid coordinates
    var depth: Int
    var order: List[Int]      # [4, N]
    var inverse: List[Int]    # [4, N]

    def __init__(out self):
        self.grid = List[Int]()
        self.depth = 0
        self.order = List[Int]()
        self.inverse = List[Int]()


def _argsort_stable(codes: List[Int]) -> List[Int]:
    var n = len(codes)
    var idx = List[Int](length=n, fill=0)
    for i in range(n):
        idx[i] = i
    # bottom-up merge sort
    var tmp = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:
                if codes[idx[b]] < codes[idx[a]]:
                    tmp[k] = idx[b]
                    b += 1
                else:
                    tmp[k] = idx[a]
                    a += 1
                k += 1
            while a < mid:
                tmp[k] = idx[a]
                a += 1
                k += 1
            while b < hi:
                tmp[k] = idx[b]
                b += 1
                k += 1
            lo += 2 * width
        for i in range(n):
            idx[i] = tmp[i]
        width *= 2
    return idx^


def serialize(xyz: List[Float32], mut rng: UrRng, grid_size: Float32 = UR_PTV3_GRID) -> UrSerialized:
    """Grid coordinates, depth and the four curve orders ("z", "z-trans", "hilbert",
    "hilbert-trans"), shuffled like PTv3's `shuffle_orders`."""
    var n = len(xyz) // 3
    var s = UrSerialized()
    var mn = List[Float32](length=3, fill=3.4e38)
    for i in range(n):
        for c in range(3):
            mn[c] = min(mn[c], xyz[i * 3 + c])
    s.grid = List[Int](length=n * 3, fill=0)
    var mx = 0
    for i in range(n):
        for c in range(3):
            var q = (xyz[i * 3 + c] - mn[c]) / grid_size
            var gi = Int(q)  # trunc (q >= 0)
            s.grid[i * 3 + c] = gi
            mx = max(mx, gi)
    var depth = 0
    while (1 << depth) <= mx:
        depth += 1
    s.depth = depth
    var orders = List[List[Int]]()
    for o in range(4):
        var codes = List[Int](length=n, fill=0)
        for i in range(n):
            var x = s.grid[i * 3]
            var y = s.grid[i * 3 + 1]
            var z = s.grid[i * 3 + 2]
            if o == 1 or o == 3:
                var t = x
                x = y
                y = t
            codes[i] = _zorder(x, y, z, depth) if o < 2 else _hilbert(x, y, z, depth)
        orders.append(_argsort_stable(codes))
    var perm = List[Int]()
    for o in range(4):
        perm.append(o)
    for i in range(3, 0, -1):
        var j = rng.below(i + 1)
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    s.order = List[Int](length=4 * n, fill=0)
    s.inverse = List[Int](length=4 * n, fill=0)
    for o in range(4):
        var src = perm[o]
        for i in range(n):
            var p = orders[src][i]
            s.order[o * n + i] = p
            s.inverse[o * n + p] = i
    return s^


def neighbor_table(grid: List[Int], k: Int) -> List[Int]:
    """Sub-manifold convolution neighbours [N, k^3] (index n = empty). Slot (a, b, c) holds the
    point at grid + (a - k/2, b - k/2, c - k/2); the first point in a voxel answers for it."""
    var n = len(grid) // 3
    var r = k // 2
    var S = 1 << 20
    var first = Dict[Int, Int]()
    for i in range(n):
        var key = ((grid[i * 3] + r) * S + (grid[i * 3 + 1] + r)) * S + (grid[i * 3 + 2] + r)
        if key not in first:
            first[key] = i
    var out = List[Int](length=n * k * k * k, fill=n)
    for i in range(n):
        var slot = 0
        for a in range(k):
            for b in range(k):
                for c in range(k):
                    var key = ((grid[i * 3] + a) * S + (grid[i * 3 + 1] + b)) * S + (grid[i * 3 + 2] + c)
                    var hit = first.get(key)
                    if hit:
                        out[i * k * k * k + slot] = hit.value()
                    slot += 1
    return out^


# --------------------------------------------------------------------------- reskin

def reskin(samples: List[Float32], v: List[Float32], f: List[Int], parents: List[Int], skin_s: List[Float32],
           nearest: Int = 7, alpha: Float64 = 2.0, threshold: Float32 = 0.03) -> List[Float32]:
    """UniRig `reskin(sample_method='median')`: sample weights [N, J] -> vertex weights [V, J]."""
    var nv = len(v) // 3
    var J = len(parents)
    var grid = UrPointGrid(samples)
    var skin = List[Float64](length=nv * J, fill=0.0)
    var qi = List[Int]()
    var qd = List[Float64]()
    var col = List[Float32](length=nearest, fill=0.0)
    for i in range(nv):
        grid.knn(v[i * 3], v[i * 3 + 1], v[i * 3 + 2], nearest, qi, qd)
        var m = len(qi)
        for j in range(J):
            for s in range(m):
                col[s] = skin_s[qi[s] * J + j]
            # median (sort the m values)
            for a in range(1, m):
                var b = a
                while b > 0 and col[b - 1] > col[b]:
                    var t = col[b - 1]
                    col[b - 1] = col[b]
                    col[b] = t
                    b -= 1
            var med = Float64(col[m // 2]) if m % 2 == 1 else (Float64(col[m // 2 - 1]) + Float64(col[m // 2])) / 2
            skin[i * J + j] = med
    # one diffusion step over edges (both directions), hotter -> cooler, on subtree sums
    var sum_skin = skin.copy()
    var i = J - 1
    while i >= 0:
        if parents[i] >= 0:
            for vv in range(nv):
                sum_skin[vv * J + parents[i]] += sum_skin[vv * J + i]
        i -= 1
    var nf = len(f) // 3
    var E = nf * 3
    # numpy fancy-index `+=` keeps only the last edge that targets a vertex
    var last = List[Int](length=nv, fill=-1)
    for e in range(2 * E):
        var half = e % E
        var t = half % nf
        var k = half // nf
        var a = f[t * 3 + k]
        var b = f[t * 3 + (k + 1) % 3]
        var dst = b if e < E else a
        last[dst] = e
    var nb = List[Float64](length=nv * J, fill=0.0)
    var nco = List[Float64](length=nv * J, fill=0.0)
    for vv in range(nv):
        var e = last[vv]
        if e < 0:
            continue
        var half = e % E
        var t = half % nf
        var k = half // nf
        var a = f[t * 3 + k]
        var b = f[t * 3 + (k + 1) % 3]
        var src = a if e < E else b
        var dx = v[a * 3] - v[b * 3]
        var dy = v[a * 3 + 1] - v[b * 3 + 1]
        var dz = v[a * 3 + 2] - v[b * 3 + 2]
        var co = exp(-Float64(sqrt(dx * dx + dy * dy + dz * dz)) * alpha)
        for j in range(J):
            if sum_skin[vv * J + j] < sum_skin[src * J + j]:
                nb[vv * J + j] = sum_skin[src * J + j] * co
                nco[vv * J + j] = Float64(Float32(co))
    for k in range(nv * J):
        sum_skin[k] = (sum_skin[k] + nb[k]) / (1.0 + nco[k])
    var any_child = False
    for j in range(J):
        if parents[j] >= 0:
            any_child = True
            for vv in range(nv):
                sum_skin[vv * J + parents[j]] -= sum_skin[vv * J + j]
    if any_child:
        for vv in range(nv):
            var tot: Float64 = 0
            for j in range(J):
                tot += sum_skin[vv * J + j]
            for j in range(J):
                skin[vv * J + j] = sum_skin[vv * J + j] / tot
    var out = List[Float32](length=nv * J, fill=0.0)
    for vv in range(nv):
        var keep = False
        for j in range(J):
            if Float32(skin[vv * J + j]) >= threshold:
                keep = True
        var tot: Float64 = 0
        for j in range(J):
            var w = skin[vv * J + j]
            if keep and Float32(w) < threshold:
                w = 0
            skin[vv * J + j] = w
            tot += w
        for j in range(J):
            out[vv * J + j] = Float32(skin[vv * J + j] / tot)
    return out^


def weld(v: List[Float32], f: List[Int], mut wv: List[Float32], mut wf: List[Int]) -> List[Int]:
    """Merges vertices at exactly the same position (UV / normal seams). Returns, for every input
    vertex, its welded index; `wv` / `wf` receive the welded mesh."""
    var nv = len(v) // 3
    var seen = Dict[String, Int]()
    var remap = List[Int](length=nv, fill=0)
    wv = List[Float32]()
    for i in range(nv):
        var key = String(v[i * 3]) + "," + String(v[i * 3 + 1]) + "," + String(v[i * 3 + 2])
        var hit = seen.get(key)
        if hit:
            remap[i] = hit.value()
        else:
            var k = len(wv) // 3
            seen[key] = k
            remap[i] = k
            wv.append(v[i * 3])
            wv.append(v[i * 3 + 1])
            wv.append(v[i * 3 + 2])
    wf = List[Int](length=len(f), fill=0)
    for k in range(len(f)):
        wf[k] = remap[f[k]]
    return remap^


# --------------------------------------------------------------------------- the model

struct UniRigSkin(Movable):
    """The three skin graphs (`unirig_skin_global.onnx`, `unirig_skin_body.onnx`,
    `unirig_skin_head.onnx`) and a job pool for the geometry."""
    var glob: OnnxModel
    var body: OnnxModel
    var head: OnnxModel
    var pool: JobPool
    var head_chunk: Int

    def __init__(out self, var glob: OnnxModel, var body: OnnxModel, var head: OnnxModel):
        self.glob = glob^
        self.body = body^
        self.head = head^
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())
        self.head_chunk = 1024

    @staticmethod
    def load(dir: String, gpu: Bool = True) raises -> UniRigSkin:
        var g = OnnxModel.load(dir + "/unirig_skin_global.onnx")
        var b = OnnxModel.load(dir + "/unirig_skin_body.onnx")
        var h = OnnxModel.load(dir + "/unirig_skin_head.onnx")
        if gpu:
            _ = g.use_gpu()
            _ = b.use_gpu()
            _ = h.use_gpu()
        return UniRigSkin(g^, b^, h^)

    def __deinit__(deinit self):
        self.pool.stop()

    def device(self) raises -> String:
        return self.body.gpu_name()

    def predict_samples(mut self, xyz: List[Float32], nrm: List[Float32], vskin: List[Float32], joints: List[Float32],
                        parents: List[Int], queries: List[Float32], ser: UrSerialized,
                        nbr3: List[Int], nbr5: List[Int]) raises -> List[Float32]:
        """Network part: sample weights [N, J] (voxel-masked and renormalised)."""
        var N = len(xyz) // 3
        var J = len(parents)
        var M = len(queries) // 6
        var qp = List[Float32](length=M * 3, fill=0.0)
        var qn = List[Float32](length=M * 3, fill=0.0)
        for i in range(M):
            for c in range(3):
                qp[i * 3 + c] = queries[i * 6 + c]
                qn[i * 3 + c] = queries[i * 6 + 3 + c]
        self.glob.set_input("pc", Tensor.floats([1, N, 3], xyz.copy()))
        self.glob.set_input("feats", Tensor.floats([1, N, 3], nrm.copy()))
        self.glob.set_input("qpc", Tensor.floats([1, M, 3], qp^))
        self.glob.set_input("qfeats", Tensor.floats([1, M, 3], qn^))
        self.glob.run()
        var glat = self.glob.output("glat")
        var feat = List[Float32](length=N * 9, fill=0.0)
        var mn = List[Float32](length=3, fill=3.4e38)
        for i in range(N):
            for c in range(3):
                feat[i * 9 + c] = xyz[i * 3 + c]
                feat[i * 9 + 3 + c] = nrm[i * 3 + c]
                mn[c] = min(mn[c], xyz[i * 3 + c])
        var bones = List[Float32](length=J * 3, fill=0.0)
        for j in range(J):
            for c in range(3):
                bones[j * 3 + c] = joints[j * 3 + c] - mn[c]
        var gl = Tensor.floats([M, glat.numel() // M], List[Float32](length=glat.numel(), fill=0.0))
        for k in range(glat.numel()):
            gl.f[k] = glat.fget(k)
        self.body.set_input("feat", Tensor.floats([N, 9], feat^))
        self.body.set_input("nbr3", Tensor.ints(DT_I64, [N, 27], nbr3.copy()))
        self.body.set_input("nbr5", Tensor.ints(DT_I64, [N, 125], nbr5.copy()))
        self.body.set_input("order", Tensor.ints(DT_I64, [4, N], ser.order.copy()))
        self.body.set_input("inverse", Tensor.ints(DT_I64, [4, N], ser.inverse.copy()))
        self.body.set_input("bones", Tensor.floats([J, 3], bones^))
        self.body.set_input("glat", gl^)
        self.body.run()
        var mfeat = self.body.output("mfeat")
        var bonek = self.body.output("bonek")
        var C = mfeat.numel() // N
        var pred = List[Float32](length=N * J, fill=0.0)
        var s = 0
        while s < N:
            var n = min(self.head_chunk, N - s)
            var mf = List[Float32](length=n * C, fill=0.0)
            for k in range(n * C):
                mf[k] = mfeat.fget(s * C + k)
            var vs = List[Float32](length=n * J, fill=0.0)
            for k in range(n * J):
                vs[k] = vskin[s * J + k]
            self.head.set_input("mfeat", Tensor.floats([n, C], mf^))
            self.head.set_input("bonek", bonek.copy())
            self.head.set_input("vskin", Tensor.floats([n, J], vs^))
            self.head.run()
            var p = self.head.output("pred")
            for k in range(n * J):
                pred[s * J + k] = p.fget(k)
            s += n
        # voxel mask: prior summed into parents (forward order, as UniRig), cubed
        var mask = List[Float32](length=J, fill=0.0)
        for i in range(N):
            for j in range(J):
                mask[j] = vskin[i * J + j]
            for j in range(J):
                if parents[j] >= 0:
                    mask[parents[j]] += mask[j]
            var tot: Float32 = 0
            for j in range(J):
                var w = pred[i * J + j] * mask[j] * mask[j] * mask[j]
                tot += w
            if tot > 0:
                for j in range(J):
                    pred[i * J + j] = pred[i * J + j] * mask[j] * mask[j] * mask[j] / tot
        return pred^

    def skin(mut self, v_in: List[Float32], f_in: List[Int], joints_orig: List[Float32], parents: List[Int],
             seed: UInt64 = 0) raises -> List[Float32]:
        """Mesh (any space) + skeleton joints (same space) -> per-vertex weights [V, J].

        Coincident vertices are welded first (and get identical weights): seams then neither cut
        the geodesic graph nor make its nearest-vertex choices arbitrary."""
        var v_orig = List[Float32]()
        var f = List[Int]()
        var remap = weld(v_in, f_in, v_orig, f)
        var w = self._skin_welded(v_orig, f, joints_orig, parents, seed)
        var J = len(parents)
        var out = List[Float32](length=len(remap) * J, fill=0.0)
        for i in range(len(remap)):
            for j in range(J):
                out[i * J + j] = w[remap[i] * J + j]
        return out^

    def _skin_welded(mut self, v_orig: List[Float32], f: List[Int], joints_orig: List[Float32], parents: List[Int],
                     seed: UInt64) raises -> List[Float32]:
        var nv = len(v_orig) // 3
        var J = len(parents)
        if J == 0:
            raise Error("UniRig skin: empty skeleton")
        var lo = List[Float64](length=3, fill=1e300)
        var hi = List[Float64](length=3, fill=-1e300)
        for i in range(nv):
            for c in range(3):
                lo[c] = min(lo[c], Float64(v_orig[i * 3 + c]))
                hi[c] = max(hi[c], Float64(v_orig[i * 3 + c]))
        for j in range(J):
            for c in range(3):
                lo[c] = min(lo[c], Float64(joints_orig[j * 3 + c]))
                hi[c] = max(hi[c], Float64(joints_orig[j * 3 + c]))
        var sc = max(max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]) / 2
        var v = List[Float32](length=nv * 3, fill=0.0)
        for i in range(nv):
            for c in range(3):
                v[i * 3 + c] = Float32((Float64(v_orig[i * 3 + c]) - (lo[c] + hi[c]) / 2) / sc)
        var joints = List[Float32](length=J * 3, fill=0.0)
        for j in range(J):
            for c in range(3):
                joints[j * 3 + c] = Float32((Float64(joints_orig[j * 3 + c]) - (lo[c] + hi[c]) / 2) / sc)
        var vn = List[Float32]()
        var fnrm = List[Float32]()
        normals_of(v, f, vn, fnrm)
        var vskin_v = voxel_skin(self.pool, v, f, joints)
        var rng = UrRng(seed * 0x9E3779B97F4A7C15 + 7)
        var smp = sample_mesh(v, f, vn, fnrm, UR_SKIN_SAMPLES, UR_SKIN_VERTEX_SAMPLES, rng)
        var vs = smp.interpolate(vskin_v, J, f)
        var q = michelangelo_queries(smp.xyz, smp.nrm, rng)
        var ser = serialize(smp.xyz, rng)
        var n3 = neighbor_table(ser.grid, 3)
        var n5 = neighbor_table(ser.grid, 5)
        var pred = self.predict_samples(smp.xyz, smp.nrm, vs, joints, parents, q, ser, n3, n5)
        return reskin(smp.xyz, v, f, parents, pred)
