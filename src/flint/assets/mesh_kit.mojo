"""Shared mesh-processing utilities for Flint's geometry kit (`flint.assets.mesh_kit`).

Everything an AI-generated or scanned mesh passes through before it ships needs the same few
building blocks, so they live here once:
  - `EdgeMap`: open-addressing `(u, v) -> Int` hash keyed on undirected (or directed) vertex pairs.
  - `weld_positions`: colocated-vertex welding (quantised spatial hash, exact epsilon test), giving
    the position topology that attribute seams (UV/normal splits) hide.
  - `mesh_topology`: boundary / non-manifold edge counts, connected components, Euler characteristic
    and a `watertight` verdict on the welded surface.
  - `TriangleGrid`: uniform-grid closest-point queries on a triangle soup (exact point-triangle
    distance, Ericson "Real-Time Collision Detection" 5.1.5), used by Hausdorff measurement and by
    remeshing's back-projection.
  - `hausdorff_distance`: symmetric sampled Hausdorff / mean / RMS surface deviation.
  - `recompute_normals`, `mesh_bounds`, `copy_mesh_material`, `flatten_model_meshes`.
"""
from std.math import abs, floor, max, min, sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeMesh, NativeModel


# ------------------------------------------------------------------------------- hashing

@always_inline
def next_pow2(n: Int) -> Int:
    var p = 16
    while p < n:
        p <<= 1
    return p


@always_inline
def mix_u64(key: UInt64) -> UInt64:
    var k = key
    k = (k ^ (k >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    k = (k ^ (k >> 27)) * UInt64(0x94D049BB133111EB)
    return k ^ (k >> 31)


@always_inline
def pair_key(a: Int, b: Int) -> UInt64:
    """Directed pair key; use `pair_key(min, max)` for undirected edges."""
    return (UInt64(a) << 32) | UInt64(b & 0xFFFFFFFF)


comptime EMPTY_KEY: UInt64 = 0xFFFFFFFFFFFFFFFF


struct EdgeMap(Movable):
    """Open-addressing hash map from a 64-bit pair key to an Int value (linear probing)."""
    var keys: List[UInt64]
    var vals: List[Int]
    var mask: Int
    var count: Int

    def __init__(out self, expected: Int):
        var cap = next_pow2(expected * 2 + 16)
        self.keys = List[UInt64](length=cap, fill=EMPTY_KEY)
        self.vals = List[Int](length=cap, fill=-1)
        self.mask = cap - 1
        self.count = 0

    def _grow(mut self):
        var old_k = self.keys.copy()
        var old_v = self.vals.copy()
        var cap = (self.mask + 1) * 2
        self.keys = List[UInt64](length=cap, fill=EMPTY_KEY)
        self.vals = List[Int](length=cap, fill=-1)
        self.mask = cap - 1
        self.count = 0
        for i in range(len(old_k)):
            if old_k[i] != EMPTY_KEY:
                _ = self.insert(old_k[i], old_v[i])

    def find(self, key: UInt64) -> Int:
        """The value stored under `key`, or -1."""
        var slot = Int(mix_u64(key)) & self.mask
        while True:
            var k = self.keys[slot]
            if k == key:
                return self.vals[slot]
            if k == EMPTY_KEY:
                return -1
            slot = (slot + 1) & self.mask

    def insert(mut self, key: UInt64, val: Int) -> Int:
        """Stores `val` under `key` if absent; returns the value that is now stored."""
        if (self.count + 1) * 2 > self.mask + 1:
            self._grow()
        var slot = Int(mix_u64(key)) & self.mask
        while True:
            var k = self.keys[slot]
            if k == key:
                return self.vals[slot]
            if k == EMPTY_KEY:
                self.keys[slot] = key
                self.vals[slot] = val
                self.count += 1
                return val
            slot = (slot + 1) & self.mask

    def set(mut self, key: UInt64, val: Int):
        if (self.count + 1) * 2 > self.mask + 1:
            self._grow()
        var slot = Int(mix_u64(key)) & self.mask
        while True:
            var k = self.keys[slot]
            if k == key or k == EMPTY_KEY:
                if k == EMPTY_KEY:
                    self.count += 1
                self.keys[slot] = key
                self.vals[slot] = val
                return
            slot = (slot + 1) & self.mask


# ------------------------------------------------------------------------------- vectors

@always_inline
def unit(v: Vec3, fallback: Vec3 = Vec3(0, 0, 0)) -> Vec3:
    """Scale-free normalisation (`Vec3.normalized` zeroes vectors shorter than 1e-5, which
    breaks area-weighted sums on small meshes)."""
    var l2 = Float64(v.x) * Float64(v.x) + Float64(v.y) * Float64(v.y) + Float64(v.z) * Float64(v.z)
    if l2 <= 1e-60:
        return fallback
    var inv = Float32(1.0 / sqrt(l2))
    return Vec3(v.x * inv, v.y * inv, v.z * inv)

@always_inline
def vpos(mesh: NativeMesh, v: Int) -> Vec3:
    return Vec3(mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2])


@always_inline
def pos_at(p: List[Float32], v: Int) -> Vec3:
    return Vec3(p[v * 3], p[v * 3 + 1], p[v * 3 + 2])


@always_inline
def set_pos(mut p: List[Float32], v: Int, x: Vec3):
    p[v * 3] = x.x
    p[v * 3 + 1] = x.y
    p[v * 3 + 2] = x.z


def mesh_bounds(mesh: NativeMesh) -> Tuple[Vec3, Vec3]:
    var n = mesh.vertex_count()
    if n == 0:
        return (Vec3(0, 0, 0), Vec3(0, 0, 0))
    var lo = vpos(mesh, 0)
    var hi = lo
    for v in range(1, n):
        var p = vpos(mesh, v)
        lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
        hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
    return (lo, hi)


def bounds_diagonal(mesh: NativeMesh) -> Float32:
    var b = mesh_bounds(mesh)
    return (b[1] - b[0]).length()


def triangle_area(a: Vec3, b: Vec3, c: Vec3) -> Float32:
    return (b - a).cross(c - a).length() * 0.5


def surface_area(mesh: NativeMesh) -> Float64:
    var s: Float64 = 0
    for t in range(mesh.triangle_count()):
        s += Float64(triangle_area(vpos(mesh, mesh.indices[t * 3]), vpos(mesh, mesh.indices[t * 3 + 1]), vpos(mesh, mesh.indices[t * 3 + 2])))
    return s


def copy_mesh_material(src: NativeMesh, mut dst: NativeMesh):
    """Copies every non-geometry field (material, skin binding, alpha) from `src` to `dst`."""
    dst.texture = src.texture
    dst.normal_orm_texture = src.normal_orm_texture
    dst.orm_texture = src.orm_texture
    dst.base_color = src.base_color
    dst.roughness = src.roughness
    dst.metallic = src.metallic
    dst.emissive_color = src.emissive_color
    dst.emissive_strength = src.emissive_strength
    dst.clearcoat = src.clearcoat
    dst.clearcoat_roughness = src.clearcoat_roughness
    dst.specular_factor = src.specular_factor
    dst.iridescence = src.iridescence
    dst.iridescence_thickness_nm = src.iridescence_thickness_nm
    dst.iridescence_ior = src.iridescence_ior
    dst.sheen_color = src.sheen_color
    dst.sheen_roughness = src.sheen_roughness
    dst.skin = src.skin
    dst.alpha_mode = src.alpha_mode
    dst.double_sided = src.double_sided


def recompute_normals(mut mesh: NativeMesh, weld_seams: Bool = True):
    """Area-weighted smooth vertex normals. With `weld_seams`, colocated vertices (UV seams) share
    one normal so seams do not shade as creases."""
    var n = mesh.vertex_count()
    var rep = List[Int]()
    var nrep = n
    if weld_seams:
        var w = weld_positions(mesh, 0.0)
        rep = w[0].copy()
        nrep = w[1]
    else:
        for v in range(n):
            rep.append(v)
    var acc = List[Float32](length=nrep * 3, fill=0.0)
    for t in range(mesh.triangle_count()):
        var i0 = mesh.indices[t * 3]
        var i1 = mesh.indices[t * 3 + 1]
        var i2 = mesh.indices[t * 3 + 2]
        var fn_ = (vpos(mesh, i1) - vpos(mesh, i0)).cross(vpos(mesh, i2) - vpos(mesh, i0))
        for k in range(3):
            var r = rep[mesh.indices[t * 3 + k]]
            acc[r * 3] += fn_.x
            acc[r * 3 + 1] += fn_.y
            acc[r * 3 + 2] += fn_.z
    mesh.normals = List[Float32](length=n * 3, fill=0.0)
    for v in range(n):
        var r = rep[v]
        var nn = Vec3(acc[r * 3], acc[r * 3 + 1], acc[r * 3 + 2])
        var l = nn.length()
        if l > 1e-20:
            nn = nn * (1.0 / l)
        else:
            nn = Vec3(0, 1, 0)
        set_pos(mesh.normals, v, nn)


# ------------------------------------------------------------------------------- welding

def weld_positions(mesh: NativeMesh, eps: Float32 = 0.0) -> Tuple[List[Int], Int]:
    """Maps every vertex to a representative id shared by all vertices within `eps` (exact match
    for eps = 0). Returns (vertex -> rep id, rep count); rep ids are dense in first-seen order."""
    var n = mesh.vertex_count()
    var rep = List[Int](length=n, fill=0)
    if n == 0:
        return (rep^, 0)
    var b = mesh_bounds(mesh)
    var ext = max(Float32(1e-6), max(b[1].x - b[0].x, max(b[1].y - b[0].y, b[1].z - b[0].z)))
    var cell = max(eps * 2.0, ext / 4096.0)
    var inv = 1.0 / cell
    var grid = EdgeMap(n)
    var nxt = List[Int](length=n, fill=-1)
    var head_v = List[Int]()  # representative vertex of each rep id
    var count = 0
    var eps2 = eps * eps
    for v in range(n):
        var p = vpos(mesh, v)
        var qx = Int(floor((p.x - b[0].x) * inv))
        var qy = Int(floor((p.y - b[0].y) * inv))
        var qz = Int(floor((p.z - b[0].z) * inv))
        var found = -1
        var r = 1 if eps > 0 else 0
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if found >= 0:
                        continue
                    var key = _cell_key(qx + dx, qy + dy, qz + dz)
                    var cur = grid.find(key)
                    while cur >= 0:
                        var d2 = (vpos(mesh, head_v[cur]) - p).length_squared()
                        if d2 <= eps2:
                            found = cur
                            break
                        cur = nxt[cur]
        if found < 0:
            found = count
            head_v.append(v)
            var key = _cell_key(qx, qy, qz)
            var prev = grid.find(key)
            if count >= len(nxt):
                nxt.append(-1)
            nxt[count] = prev
            grid.set(key, count)
            count += 1
        rep[v] = found
    return (rep^, count)


@always_inline
def _cell_key(x: Int, y: Int, z: Int) -> UInt64:
    return (UInt64(x & 0x1FFFFF) << 42) | (UInt64(y & 0x1FFFFF) << 21) | UInt64(z & 0x1FFFFF)


# ------------------------------------------------------------------------------- topology

@fieldwise_init
struct TopologyReport(Copyable, ImplicitlyCopyable, Movable, Writable):
    var vertices: Int
    var triangles: Int
    var welded_vertices: Int
    var edges: Int
    var boundary_edges: Int
    var nonmanifold_edges: Int
    var inconsistent_edges: Int   # an edge used twice in the same direction (flipped neighbour)
    var degenerate_triangles: Int
    var components: Int
    var euler: Int

    def watertight(self) -> Bool:
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0 and self.triangles > 0

    def oriented_manifold(self) -> Bool:
        return self.nonmanifold_edges == 0 and self.inconsistent_edges == 0

    def write_to[W: Writer](self, mut writer: W):
        writer.write("V=", self.vertices, " T=", self.triangles, " Vw=", self.welded_vertices,
                     " E=", self.edges, " boundary=", self.boundary_edges, " nonmanifold=", self.nonmanifold_edges,
                     " flipped=", self.inconsistent_edges, " degenerate=", self.degenerate_triangles,
                     " components=", self.components, " chi=", self.euler,
                     " watertight=", self.watertight())


def _uf_find(mut parent: List[Int], x: Int) -> Int:
    var r = x
    while parent[r] != r:
        parent[r] = parent[parent[r]]
        r = parent[r]
    return r


def mesh_topology(mesh: NativeMesh) -> TopologyReport:
    """Edge-manifold report on the position-welded surface (UV seams do not count as boundaries)."""
    var w = weld_positions(mesh, 0.0)
    ref rep = w[0]
    var nr = w[1]
    var nt = mesh.triangle_count()
    var und = EdgeMap(nt * 3)
    var cnt = List[Int]()
    var dir_cnt = EdgeMap(nt * 3)
    var dup = 0
    var degenerate = 0
    var parent = List[Int](length=nr, fill=0)
    for i in range(nr):
        parent[i] = i
    var used = List[Bool](length=nr, fill=False)
    for t in range(nt):
        var r0 = rep[mesh.indices[t * 3]]
        var r1 = rep[mesh.indices[t * 3 + 1]]
        var r2 = rep[mesh.indices[t * 3 + 2]]
        if r0 == r1 or r1 == r2 or r0 == r2:
            degenerate += 1
            continue
        used[r0] = True
        used[r1] = True
        used[r2] = True
        var a = _uf_find(parent, r0)
        var b = _uf_find(parent, r1)
        if a != b:
            parent[a] = b
        a = _uf_find(parent, r1)
        b = _uf_find(parent, r2)
        if a != b:
            parent[a] = b
        var rs = [r0, r1, r2]
        for k in range(3):
            var u = rs[k]
            var v = rs[(k + 1) % 3]
            var e = und.insert(pair_key(min(u, v), max(u, v)), len(cnt))
            if e == len(cnt):
                cnt.append(0)
            cnt[e] += 1
            var dk = pair_key(u, v)
            var prev = dir_cnt.find(dk)
            if prev >= 0:
                dup += 1
            else:
                dir_cnt.set(dk, 1)
    var boundary = 0
    var nonman = 0
    for e in range(len(cnt)):
        if cnt[e] == 1:
            boundary += 1
        elif cnt[e] > 2:
            nonman += 1
    var comps = 0
    var nv_used = 0
    for i in range(nr):
        if used[i]:
            nv_used += 1
            if _uf_find(parent, i) == i:
                comps += 1
    var live = nt - degenerate
    return TopologyReport(mesh.vertex_count(), nt, nv_used, len(cnt), boundary, nonman, dup, degenerate, comps,
                          nv_used - len(cnt) + live)


# ------------------------------------------------------------------------------- closest point

def closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """Exact closest point on triangle abc to p (Voronoi-region walk, Ericson 2005 §5.1.5)."""
    var ab = b - a
    var ac = c - a
    var ap = p - a
    var d1 = ab.dot(ap)
    var d2 = ac.dot(ap)
    if d1 <= 0 and d2 <= 0:
        return a
    var bp = p - b
    var d3 = ab.dot(bp)
    var d4 = ac.dot(bp)
    if d3 >= 0 and d4 <= d3:
        return b
    var vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        var v = d1 / (d1 - d3)
        return a + ab * v
    var cp = p - c
    var d5 = ab.dot(cp)
    var d6 = ac.dot(cp)
    if d6 >= 0 and d5 <= d6:
        return c
    var vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        var w = d2 / (d2 - d6)
        return a + ac * w
    var va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        var w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w
    var denom = 1.0 / (va + vb + vc)
    var v2 = vb * denom
    var w2 = vc * denom
    return a + ab * v2 + ac * w2


@fieldwise_init
struct ClosestHit(Copyable, ImplicitlyCopyable, Movable):
    var dist2: Float32
    var tri: Int
    var point: Vec3


struct TriangleGrid(Movable):
    """Uniform grid over a triangle set for exact nearest-surface queries."""
    var pos: List[Float32]
    var idx: List[Int]
    var lo: Vec3
    var cell: Float32
    var nx: Int
    var ny: Int
    var nz: Int
    var cell_start: List[Int]
    var cell_tris: List[Int]
    var stamp: List[Int]
    var cur_stamp: Int

    def __init__(out self, mesh: NativeMesh, cells_per_tri: Float32 = 1.0):
        self.pos = mesh.positions.copy()
        self.idx = mesh.indices.copy()
        var b = mesh_bounds(mesh)
        var nt = max(1, mesh.triangle_count())
        var ext = b[1] - b[0]
        var pad = max(Float32(1e-5), max(ext.x, max(ext.y, ext.z)) * 0.001)
        self.lo = b[0] - Vec3(pad, pad, pad)
        var size = ext + Vec3(pad * 2, pad * 2, pad * 2)
        var vol = Float64(size.x) * Float64(size.y) * Float64(size.z)
        var target = Float64(nt) * Float64(cells_per_tri)
        var c = Float32((vol / target) ** (1.0 / 3.0))
        # flat meshes have zero volume: fall back to an area-based cell size
        var area_cell = Float32(sqrt(Float64(max(size.x * size.y, max(size.y * size.z, size.x * size.z))) / Float64(nt) * 2.0))
        c = max(c, area_cell)
        c = max(c, max(size.x, max(size.y, size.z)) / 256.0)
        self.cell = max(c, Float32(1e-6))
        self.nx = max(1, Int(size.x / self.cell) + 1)
        self.ny = max(1, Int(size.y / self.cell) + 1)
        self.nz = max(1, Int(size.z / self.cell) + 1)
        var ncell = self.nx * self.ny * self.nz
        self.cell_start = List[Int]()
        self.cell_tris = List[Int]()
        self.stamp = List[Int](length=len(self.idx) // 3, fill=0)
        self.cur_stamp = 0
        var counts = List[Int](length=ncell + 1, fill=0)
        for pass_ in range(2):
            var fill_pos = List[Int]()
            if pass_ == 1:
                self.cell_start = List[Int](length=ncell + 1, fill=0)
                for i in range(ncell):
                    self.cell_start[i + 1] = self.cell_start[i] + counts[i]
                self.cell_tris = List[Int](length=self.cell_start[ncell], fill=0)
                fill_pos = self.cell_start.copy()
            for t in range(len(self.idx) // 3):
                var a = pos_at(self.pos, self.idx[t * 3])
                var bb = pos_at(self.pos, self.idx[t * 3 + 1])
                var cc = pos_at(self.pos, self.idx[t * 3 + 2])
                var tlo = Vec3(min(a.x, min(bb.x, cc.x)), min(a.y, min(bb.y, cc.y)), min(a.z, min(bb.z, cc.z)))
                var thi = Vec3(max(a.x, max(bb.x, cc.x)), max(a.y, max(bb.y, cc.y)), max(a.z, max(bb.z, cc.z)))
                var x0 = self._cx(tlo.x); var x1 = self._cx(thi.x)
                var y0 = self._cy(tlo.y); var y1 = self._cy(thi.y)
                var z0 = self._cz(tlo.z); var z1 = self._cz(thi.z)
                for z in range(z0, z1 + 1):
                    for y in range(y0, y1 + 1):
                        for x in range(x0, x1 + 1):
                            var ci = (z * self.ny + y) * self.nx + x
                            if pass_ == 0:
                                counts[ci] += 1
                            else:
                                self.cell_tris[fill_pos[ci]] = t
                                fill_pos[ci] += 1

    @always_inline
    def _cx(self, x: Float32) -> Int:
        return max(0, min(self.nx - 1, Int((x - self.lo.x) / self.cell)))

    @always_inline
    def _cy(self, y: Float32) -> Int:
        return max(0, min(self.ny - 1, Int((y - self.lo.y) / self.cell)))

    @always_inline
    def _cz(self, z: Float32) -> Int:
        return max(0, min(self.nz - 1, Int((z - self.lo.z) / self.cell)))

    def tri_count(self) -> Int:
        return len(self.idx) // 3

    def closest(mut self, p: Vec3) -> ClosestHit:
        """Nearest surface point. Expands Chebyshev rings until no farther ring can beat the best."""
        self.cur_stamp += 1
        var cx = self._cx(p.x)
        var cy = self._cy(p.y)
        var cz = self._cz(p.z)
        var cc = Vec3(min(max(p.x, self.lo.x), self.lo.x + Float32(self.nx) * self.cell),
                      min(max(p.y, self.lo.y), self.lo.y + Float32(self.ny) * self.cell),
                      min(max(p.z, self.lo.z), self.lo.z + Float32(self.nz) * self.cell))
        var outside = (p - cc).length()
        var best = ClosestHit(Float32(3.0e38), -1, p)
        var max_ring = max(self.nx, max(self.ny, self.nz))
        var ring = 0
        while ring <= max_ring:
            for z in range(cz - ring, cz + ring + 1):
                if z < 0 or z >= self.nz:
                    continue
                for y in range(cy - ring, cy + ring + 1):
                    if y < 0 or y >= self.ny:
                        continue
                    var on_shell_zy = (abs(z - cz) == ring) or (abs(y - cy) == ring)
                    var x = cx - ring
                    while x <= cx + ring:
                        if x >= 0 and x < self.nx:
                            var ci = (z * self.ny + y) * self.nx + x
                            for s in range(self.cell_start[ci], self.cell_start[ci + 1]):
                                var t = self.cell_tris[s]
                                if self.stamp[t] == self.cur_stamp:
                                    continue
                                self.stamp[t] = self.cur_stamp
                                var q = closest_point_on_triangle(
                                    p, pos_at(self.pos, self.idx[t * 3]), pos_at(self.pos, self.idx[t * 3 + 1]),
                                    pos_at(self.pos, self.idx[t * 3 + 2]))
                                var d2 = (q - p).length_squared()
                                if d2 < best.dist2:
                                    best = ClosestHit(d2, t, q)
                        if on_shell_zy or ring == 0:
                            x += 1
                        else:
                            x = cx + ring if x == cx - ring else cx + ring + 1
            if best.tri >= 0:
                var bound = Float32(ring) * self.cell - outside
                if bound > 0 and bound * bound >= best.dist2:
                    break
            ring += 1
        return best


# ------------------------------------------------------------------------------- Hausdorff

@fieldwise_init
struct HausdorffResult(Copyable, ImplicitlyCopyable, Movable, Writable):
    var a_to_b_max: Float32
    var b_to_a_max: Float32
    var mean: Float32
    var rms: Float32
    var diagonal: Float32

    def max_error(self) -> Float32:
        return max(self.a_to_b_max, self.b_to_a_max)

    def relative(self) -> Float32:
        """Symmetric Hausdorff distance over the bounding-box diagonal of the reference."""
        return self.max_error() / max(Float32(1e-12), self.diagonal)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("hausdorff=", self.max_error(), " (", self.relative() * 100.0, "% of diag)",
                     " mean=", self.mean, " rms=", self.rms)


@always_inline
def _hash01(seed: UInt64) -> Float32:
    return Float32(mix_u64(seed) >> 40) * (1.0 / 16777216.0)


def surface_samples(mesh: NativeMesh, count: Int, seed: UInt64 = 1) -> List[Vec3]:
    """All vertices plus `count` area-weighted stratified random points on the surface."""
    var out = List[Vec3](capacity=mesh.vertex_count() + count)
    for v in range(mesh.vertex_count()):
        out.append(vpos(mesh, v))
    var nt = mesh.triangle_count()
    if nt == 0 or count <= 0:
        return out^
    var cdf = List[Float64](capacity=nt)
    var acc: Float64 = 0
    for t in range(nt):
        acc += Float64(triangle_area(vpos(mesh, mesh.indices[t * 3]), vpos(mesh, mesh.indices[t * 3 + 1]), vpos(mesh, mesh.indices[t * 3 + 2])))
        cdf.append(acc)
    if acc <= 0:
        return out^
    var t = 0
    for s in range(count):
        var u = (Float64(s) + Float64(_hash01(seed * 7919 + UInt64(s)))) / Float64(count) * acc
        while t < nt - 1 and cdf[t] < u:
            t += 1
        var r1 = sqrt(_hash01(seed * 104729 + UInt64(s) * 3 + 1))
        var r2 = _hash01(seed * 1299709 + UInt64(s) * 3 + 2)
        var a = vpos(mesh, mesh.indices[t * 3])
        var b = vpos(mesh, mesh.indices[t * 3 + 1])
        var c = vpos(mesh, mesh.indices[t * 3 + 2])
        out.append(a * (1.0 - r1) + b * (r1 * (1.0 - r2)) + c * (r1 * r2))
    return out^


def one_sided_distance(samples: List[Vec3], mut grid: TriangleGrid) -> Tuple[Float32, Float64, Float64]:
    """(max, sum, sum of squares) of distances from samples to the grid's surface."""
    var mx: Float32 = 0
    var s: Float64 = 0
    var s2: Float64 = 0
    for i in range(len(samples)):
        var h = grid.closest(samples[i])
        var d = sqrt(h.dist2)
        mx = max(mx, d)
        s += Float64(d)
        s2 += Float64(d) * Float64(d)
    return (mx, s, s2)


def hausdorff_distance(reference: NativeMesh, test: NativeMesh, samples_per_mesh: Int = 20000) -> HausdorffResult:
    """Symmetric sampled Hausdorff distance between two surfaces (vertices + area samples each way)."""
    var ga = TriangleGrid(reference)
    var gb = TriangleGrid(test)
    var sa = surface_samples(reference, samples_per_mesh, 11)
    var sb = surface_samples(test, samples_per_mesh, 23)
    var ab = one_sided_distance(sa, gb)
    var ba = one_sided_distance(sb, ga)
    var n = Float64(len(sa) + len(sb))
    var mean = (ab[1] + ba[1]) / n
    var rms = sqrt((ab[2] + ba[2]) / n)
    return HausdorffResult(ab[0], ba[0], Float32(mean), Float32(rms), bounds_diagonal(reference))


# ------------------------------------------------------------------------------- models

def flatten_model_meshes(model: NativeModel) -> List[Int]:
    """Indices of meshes that carry triangles (skips empty primitives)."""
    var out = List[Int]()
    for m in range(len(model.meshes)):
        if model.meshes[m].triangle_count() > 0:
            out.append(m)
    return out^


def compact_mesh(mesh: NativeMesh, keep_tri: List[Bool]) -> NativeMesh:
    """A copy holding only the kept triangles and the vertices they use (all attributes carried)."""
    var nv = mesh.vertex_count()
    var remap = List[Int](length=nv, fill=-1)
    var out = NativeMesh()
    copy_mesh_material(mesh, out)
    var has_n = len(mesh.normals) >= nv * 3
    var has_uv = len(mesh.texcoords) >= nv * 2
    var has_c = len(mesh.colors) >= nv * 4
    var has_j = len(mesh.joints) >= nv * 4
    var has_w = len(mesh.weights) >= nv * 4
    var has_ao = len(mesh.ao) >= nv
    for t in range(mesh.triangle_count()):
        if not keep_tri[t]:
            continue
        for k in range(3):
            var v = mesh.indices[t * 3 + k]
            if remap[v] < 0:
                remap[v] = out.vertex_count()
                for c in range(3):
                    out.positions.append(mesh.positions[v * 3 + c])
                if has_n:
                    for c in range(3):
                        out.normals.append(mesh.normals[v * 3 + c])
                if has_uv:
                    out.texcoords.append(mesh.texcoords[v * 2])
                    out.texcoords.append(mesh.texcoords[v * 2 + 1])
                if has_c:
                    for c in range(4):
                        out.colors.append(mesh.colors[v * 4 + c])
                if has_j:
                    for c in range(4):
                        out.joints.append(mesh.joints[v * 4 + c])
                if has_w:
                    for c in range(4):
                        out.weights.append(mesh.weights[v * 4 + c])
                if has_ao:
                    out.ao.append(mesh.ao[v])
            out.indices.append(remap[v])
    return out^
