"""Mesh preprocessing for MagicArticulate / Puppeteer (`flint.ml.magicarticulate_mesh`).

The skeleton models were trained on point clouds sampled from a "watertight" version of each mesh
(MeshAnything's `MeshProcessor.convert_to_watertight`): the mesh is scaled into [-0.95, 0.95]^3,
the unsigned distance to its triangles is evaluated on a 128^3 grid (mesh2sdf, i.e. Christopher
Batty's SDFGen: exact distances in a one-cell band around every triangle, fast sweeping elsewhere)
and the level set |d| = 2 / 128 is extracted. That shell is closed even for triangle soups with
holes and self-intersections. 8192 points are then sampled uniformly by area, each with its
triangle's normal, and normalised into [-1, 1]^3.

Here the level set is extracted with marching tetrahedra (six tetrahedra per cube around the main
diagonal), which approximates the same isosurface as scikit-image's marching cubes; triangles are
oriented along the field's gradient (towards larger |d|), the orientation scikit-image produces.
"""

from std.math import sqrt, floor, ceil
from flint.ml.unirig import UrRng


@fieldwise_init
struct _V3(Copyable, Movable, ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    def __add__(self, o: _V3) -> _V3:
        return _V3(self.x + o.x, self.y + o.y, self.z + o.z)

    @always_inline
    def __sub__(self, o: _V3) -> _V3:
        return _V3(self.x - o.x, self.y - o.y, self.z - o.z)

    @always_inline
    def scale(self, s: Float32) -> _V3:
        return _V3(self.x * s, self.y * s, self.z * s)

    @always_inline
    def dot(self, o: _V3) -> Float32:
        return self.x * o.x + self.y * o.y + self.z * o.z


@always_inline
def _dist(a: _V3, b: _V3) -> Float32:
    var d = a - b
    return sqrt(d.dot(d))


@always_inline
def _point_segment_distance(x0: _V3, x1: _V3, x2: _V3) -> Float32:
    var dx = x2 - x1
    var m2 = Float64(dx.dot(dx))
    var s12 = Float32(Float64((x2 - x0).dot(dx)) / m2)
    if s12 < 0:
        s12 = 0
    elif s12 > 1:
        s12 = 1
    return _dist(x0, x1.scale(s12) + x2.scale(1 - s12))


@always_inline
def point_triangle_distance(x0: _V3, x1: _V3, x2: _V3, x3: _V3) -> Float32:
    """SDFGen's `point_triangle_distance` (float precision, same branch order)."""
    var x13 = x1 - x3
    var x23 = x2 - x3
    var x03 = x0 - x3
    var m13 = x13.dot(x13)
    var m23 = x23.dot(x23)
    var d = x13.dot(x23)
    var invdet = 1.0 / max(m13 * m23 - d * d, Float32(1e-30))
    var a = x13.dot(x03)
    var b = x23.dot(x03)
    var w23 = invdet * (m23 * a - d * b)
    var w31 = invdet * (m13 * b - d * a)
    var w12 = 1 - w23 - w31
    if w23 >= 0 and w31 >= 0 and w12 >= 0:
        return _dist(x0, x1.scale(w23) + x2.scale(w31) + x3.scale(w12))
    if w23 > 0:
        return min(_point_segment_distance(x0, x1, x2), _point_segment_distance(x0, x1, x3))
    elif w31 > 0:
        return min(_point_segment_distance(x0, x1, x2), _point_segment_distance(x0, x2, x3))
    return min(_point_segment_distance(x0, x1, x3), _point_segment_distance(x0, x2, x3))


struct _Grid(Movable):
    var n: Int
    var dx: Float32
    var phi: List[Float32]
    var tri: List[Int32]

    def __init__(out self, n: Int):
        self.n = n
        self.dx = Float32(2.0 / Float64(n))
        self.phi = List[Float32](length=n * n * n, fill=Float32(3 * n) * self.dx)
        self.tri = List[Int32](length=n * n * n, fill=-1)

    @always_inline
    def at(self, i: Int, j: Int, k: Int) -> Int:
        return i + self.n * (j + self.n * k)


@always_inline
def _vert(v: List[Float32], i: Int) -> _V3:
    return _V3(v[i * 3], v[i * 3 + 1], v[i * 3 + 2])


@always_inline
def _check(mut g: _Grid, v: List[Float32], f: List[Int], gx: _V3, c0: Int, c1: Int):
    var t = Int(g.tri[c1])
    if t < 0 or t == Int(g.tri[c0]):
        return      # same closest triangle: its distance is already phi[c0]
    var d = point_triangle_distance(gx, _vert(v, f[t * 3]), _vert(v, f[t * 3 + 1]), _vert(v, f[t * 3 + 2]))
    if d < g.phi[c0]:
        g.phi[c0] = d
        g.tri[c0] = Int32(t)


def _sweep(mut g: _Grid, v: List[Float32], f: List[Int], di: Int, dj: Int, dk: Int):
    var n = g.n
    var i0 = 1 if di > 0 else n - 2
    var i1 = n if di > 0 else -1
    var j0 = 1 if dj > 0 else n - 2
    var j1 = n if dj > 0 else -1
    var k0 = 1 if dk > 0 else n - 2
    var k1 = n if dk > 0 else -1
    var k = k0
    while k != k1:
        var j = j0
        while j != j1:
            var i = i0
            while i != i1:
                var gx = _V3(Float32(i) * g.dx - 1.0, Float32(j) * g.dx - 1.0, Float32(k) * g.dx - 1.0)
                var c0 = g.at(i, j, k)
                _check(g, v, f, gx, c0, g.at(i - di, j, k))
                _check(g, v, f, gx, c0, g.at(i, j - dj, k))
                _check(g, v, f, gx, c0, g.at(i - di, j - dj, k))
                _check(g, v, f, gx, c0, g.at(i, j, k - dk))
                _check(g, v, f, gx, c0, g.at(i - di, j, k - dk))
                _check(g, v, f, gx, c0, g.at(i, j - dj, k - dk))
                _check(g, v, f, gx, c0, g.at(i - di, j - dj, k - dk))
                i += di
            j += dj
        k += dk


def unsigned_distance_grid(v: List[Float32], f: List[Int], n: Int) -> List[Float32]:
    """|mesh2sdf.core.compute(v, f, n)|: grid point (i, j, k) at -1 + (i, j, k) * 2 / n, stored at
    i + n * (j + n * k). `v` must lie in [-1, 1]^3."""
    var g = _Grid(n)
    var nt = len(f) // 3
    for t in range(nt):
        var p = _vert(v, f[t * 3])
        var q = _vert(v, f[t * 3 + 1])
        var r = _vert(v, f[t * 3 + 2])
        var fi = List[Float64]()
        for c in range(3):
            var lo: Float64 = 1e300
            var hi: Float64 = -1e300
            for pt in [p, q, r]:
                var x = Float64(pt.x if c == 0 else (pt.y if c == 1 else pt.z))
                var u = (x - (-1.0)) / Float64(g.dx)
                lo = min(lo, u)
                hi = max(hi, u)
            fi.append(lo)
            fi.append(hi)
        var i0 = max(0, min(n - 1, Int(fi[0]) - 1))
        var i1 = max(0, min(n - 1, Int(fi[1]) + 2))
        var j0 = max(0, min(n - 1, Int(fi[2]) - 1))
        var j1 = max(0, min(n - 1, Int(fi[3]) + 2))
        var k0 = max(0, min(n - 1, Int(fi[4]) - 1))
        var k1 = max(0, min(n - 1, Int(fi[5]) + 2))
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    var gx = _V3(Float32(i) * g.dx - 1.0, Float32(j) * g.dx - 1.0, Float32(k) * g.dx - 1.0)
                    var d = point_triangle_distance(gx, p, q, r)
                    var c0 = g.at(i, j, k)
                    if d < g.phi[c0]:
                        g.phi[c0] = d
                        g.tri[c0] = Int32(t)
    for _ in range(2):
        _sweep(g, v, f, 1, 1, 1)
        _sweep(g, v, f, -1, -1, -1)
        _sweep(g, v, f, 1, 1, -1)
        _sweep(g, v, f, -1, -1, 1)
        _sweep(g, v, f, 1, -1, 1)
        _sweep(g, v, f, -1, 1, -1)
        _sweep(g, v, f, 1, -1, -1)
        _sweep(g, v, f, -1, 1, 1)
    var phi = g.phi^
    g.phi = List[Float32]()
    return phi^


# cube corners 0..7: (0,0,0) (1,0,0) (1,1,0) (0,1,0) (0,0,1) (1,0,1) (1,1,1) (0,1,1)
@always_inline
def _cx(c: Int) -> Int:
    return (c & 1) ^ ((c >> 1) & 1)


@always_inline
def _cy(c: Int) -> Int:
    return (c >> 1) & 1


@always_inline
def _cz(c: Int) -> Int:
    return (c >> 2) & 1


@always_inline
def _edge(pa: _V3, pb: _V3, va: Float32, vb: Float32, iso: Float32) -> _V3:
    var t = (iso - va) / (vb - va)
    return pa + (pb - pa).scale(t)


def _emit(mut out: List[Float32], a: _V3, b: _V3, c: _V3, up: _V3):
    """Appends triangle a b c, flipped if its normal points against `up` (towards larger values);
    degenerate triangles are dropped."""
    var e1 = b - a
    var e2 = c - a
    var nx = e1.y * e2.z - e1.z * e2.y
    var ny = e1.z * e2.x - e1.x * e2.z
    var nz = e1.x * e2.y - e1.y * e2.x
    if nx * nx + ny * ny + nz * nz <= 0:
        return      # zero area (the level passes through grid points): never sampled
    var flip = nx * up.x + ny * up.y + nz * up.z < 0
    var s = List[_V3]()
    s.append(a)
    if flip:
        s.append(c)
        s.append(b)
    else:
        s.append(b)
        s.append(c)
    for p in s:
        out.append(p.x)
        out.append(p.y)
        out.append(p.z)


def extract_level_set(field: List[Float32], n: Int, iso: Float32) -> List[Float32]:
    """Triangle soup [T * 9] of the level set `field = iso` in grid-index coordinates (marching
    tetrahedra); the inside is `field < iso`."""
    var out = List[Float32]()
    # the six tetrahedra around the 0-6 diagonal (shared faces match between neighbouring cubes)
    var tets: List[Int] = [0, 6, 1, 2, 0, 6, 2, 3, 0, 6, 3, 7, 0, 6, 7, 4, 0, 6, 4, 5, 0, 6, 5, 1]
    var cp = List[_V3](length=8, fill=_V3(0, 0, 0))
    var cv = List[Float32](length=8, fill=0.0)
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                var nin = 0
                for c in range(8):
                    var x = i + _cx(c)
                    var y = j + _cy(c)
                    var z = k + _cz(c)
                    cv[c] = field[x + n * (y + n * z)]
                    if cv[c] < iso:
                        nin += 1
                if nin == 0 or nin == 8:
                    continue
                for c in range(8):
                    cp[c] = _V3(Float32(i + _cx(c)), Float32(j + _cy(c)), Float32(k + _cz(c)))
                for t in range(6):
                    var ins = List[Int]()
                    var outs = List[Int]()
                    for q in range(4):
                        var c = tets[t * 4 + q]
                        if cv[c] < iso:
                            ins.append(c)
                        else:
                            outs.append(c)
                    if len(ins) == 0 or len(ins) == 4:
                        continue
                    # towards larger values: mean outside corner - mean inside corner
                    var mo = _V3(0, 0, 0)
                    var mi = _V3(0, 0, 0)
                    for c in outs:
                        mo = mo + cp[c]
                    for c in ins:
                        mi = mi + cp[c]
                    var up = mo.scale(1.0 / Float32(len(outs))) - mi.scale(1.0 / Float32(len(ins)))
                    if len(ins) == 1 or len(ins) == 3:
                        var lone = ins[0] if len(ins) == 1 else outs[0]
                        var others = outs.copy() if len(ins) == 1 else ins.copy()
                        var a = _edge(cp[lone], cp[others[0]], cv[lone], cv[others[0]], iso)
                        var b = _edge(cp[lone], cp[others[1]], cv[lone], cv[others[1]], iso)
                        var c3 = _edge(cp[lone], cp[others[2]], cv[lone], cv[others[2]], iso)
                        _emit(out, a, b, c3, up)
                    else:
                        var p00 = _edge(cp[ins[0]], cp[outs[0]], cv[ins[0]], cv[outs[0]], iso)
                        var p01 = _edge(cp[ins[0]], cp[outs[1]], cv[ins[0]], cv[outs[1]], iso)
                        var p11 = _edge(cp[ins[1]], cp[outs[1]], cv[ins[1]], cv[outs[1]], iso)
                        var p10 = _edge(cp[ins[1]], cp[outs[0]], cv[ins[1]], cv[outs[0]], iso)
                        _emit(out, p00, p01, p11, up)
                        _emit(out, p00, p11, p10, up)
    return out^


def watertight_soup(v: List[Float32], f: List[Int], octree_depth: Int = 7) -> List[Float32]:
    """`MeshProcessor.convert_to_watertight` as a triangle soup [T * 9] in the mesh's own space."""
    var nv = len(v) // 3
    var lo = List[Float64](length=3, fill=1e300)
    var hi = List[Float64](length=3, fill=-1e300)
    for i in range(nv):
        for c in range(3):
            lo[c] = min(lo[c], Float64(v[i * 3 + c]))
            hi[c] = max(hi[c], Float64(v[i * 3 + c]))
    var center = List[Float64]()
    var ext: Float64 = 0
    for c in range(3):
        center.append((lo[c] + hi[c]) * 0.5)
        ext = max(ext, hi[c] - lo[c])
    var scale = 2.0 * 0.95 / ext
    var vn = List[Float32](length=nv * 3, fill=0.0)
    for i in range(nv):
        for c in range(3):
            vn[i * 3 + c] = Float32((Float64(v[i * 3 + c]) - center[c]) * scale)
    var n = 1 << octree_depth
    var udf = unsigned_distance_grid(vn, f, n)
    var soup = extract_level_set(udf, n, Float32(2.0 / Float64(n)))
    for e in range(len(soup)):
        var c = e % 3
        soup[e] = Float32((Float64(soup[e]) / Float64(n) * 2.0 - 1.0) / scale + center[c])
    return soup^


def sample_soup(soup: List[Float32], count: Int, mut rng: UrRng, mut xyz: List[Float32], mut nrm: List[Float32]):
    """trimesh `sample_surface`: triangles chosen by area, uniform barycentric points, each with its
    triangle's unit normal."""
    var nt = len(soup) // 9
    var cum = List[Float64](length=nt, fill=0.0)
    var fnr = List[Float32](length=nt * 3, fill=0.0)
    var tot: Float64 = 0
    for t in range(nt):
        var a = _V3(soup[t * 9], soup[t * 9 + 1], soup[t * 9 + 2])
        var b = _V3(soup[t * 9 + 3], soup[t * 9 + 4], soup[t * 9 + 5])
        var c = _V3(soup[t * 9 + 6], soup[t * 9 + 7], soup[t * 9 + 8])
        var e1 = b - a
        var e2 = c - a
        var nx = Float64(e1.y) * Float64(e2.z) - Float64(e1.z) * Float64(e2.y)
        var ny = Float64(e1.z) * Float64(e2.x) - Float64(e1.x) * Float64(e2.z)
        var nz = Float64(e1.x) * Float64(e2.y) - Float64(e1.y) * Float64(e2.x)
        var l = sqrt(nx * nx + ny * ny + nz * nz)
        tot += l * 0.5
        cum[t] = tot
        if l > 0:
            fnr[t * 3] = Float32(nx / l)
            fnr[t * 3 + 1] = Float32(ny / l)
            fnr[t * 3 + 2] = Float32(nz / l)
    xyz = List[Float32](capacity=count * 3)
    nrm = List[Float32](capacity=count * 3)
    if nt == 0 or tot <= 0:
        return
    for _ in range(count):
        var r = rng.uniform() * tot
        var lo = 0
        var hi = nt - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        var t = lo
        var u = Float32(rng.uniform())
        var w = Float32(rng.uniform())
        if u + w > 1:
            u = 1 - u
            w = 1 - w
        for c in range(3):
            var a = soup[t * 9 + c]
            xyz.append(a + u * (soup[t * 9 + 3 + c] - a) + w * (soup[t * 9 + 6 + c] - a))
        for c in range(3):
            nrm.append(fnr[t * 3 + c])


@always_inline
def round_f16(x: Float32) -> Float32:
    """Rounds to the nearest float16 (the demo stores point clouds as np.float16)."""
    return SIMD[DType.float32, 1](x).cast[DType.float16]().cast[DType.float32]()[0]


struct MaPointCloud(Movable):
    """Encoder input (normalised, fp16-rounded) and the transforms back to the mesh's space:
    mesh = ((p * pc_scale + pc_center) / scale + center) for a point p in the [-0.5, 0.5] joint frame."""
    var xyz: List[Float32]
    var nrm: List[Float32]
    var center: List[Float64]
    var scale: Float64
    var pc_center: List[Float64]
    var pc_scale: Float64

    def __init__(out self):
        self.xyz = List[Float32]()
        self.nrm = List[Float32]()
        self.center = List[Float64](length=3, fill=0.0)
        self.scale = 1.0
        self.pc_center = List[Float64](length=3, fill=0.0)
        self.pc_scale = 1.0

    def count(self) -> Int:
        return len(self.xyz) // 3

    def to_mesh(self, x: Float64, c: Int) -> Float64:
        return (x * self.pc_scale + self.pc_center[c]) / self.scale + self.center[c]


def normalize_point_cloud(var xyz: List[Float32], var nrm: List[Float32]) -> MaPointCloud:
    """`normalize_to_unit_cube(scale_factor=0.9995)` plus the demo's bounds (`transform_params`)."""
    var pc = MaPointCloud()
    var n = len(xyz) // 3
    var lo = List[Float64](length=3, fill=1e300)
    var hi = List[Float64](length=3, fill=-1e300)
    for i in range(n):
        for c in range(3):
            lo[c] = min(lo[c], Float64(xyz[i * 3 + c]))
            hi[c] = max(hi[c], Float64(xyz[i * 3 + c]))
    var m: Float64 = 0
    for c in range(3):
        pc.center[c] = (lo[c] + hi[c]) / 2
    for i in range(n):
        for c in range(3):
            m = max(m, abs(Float64(xyz[i * 3 + c]) - pc.center[c]))
    pc.scale = 0.9995 / m if m > 0 else 1.0
    for i in range(n):
        for c in range(3):
            xyz[i * 3 + c] = round_f16(Float32((Float64(xyz[i * 3 + c]) - pc.center[c]) * pc.scale))
            nrm[i * 3 + c] = round_f16(nrm[i * 3 + c])
    for c in range(3):
        lo[c] = 1e300
        hi[c] = -1e300
    for i in range(n):
        for c in range(3):
            lo[c] = min(lo[c], Float64(xyz[i * 3 + c]))
            hi[c] = max(hi[c], Float64(xyz[i * 3 + c]))
    var ext: Float64 = 0
    for c in range(3):
        pc.pc_center[c] = (lo[c] + hi[c]) / 2
        ext = max(ext, hi[c] - lo[c])
    pc.pc_scale = ext + 1e-5
    pc.xyz = xyz^
    pc.nrm = nrm^
    return pc^


def mesh_point_cloud(v: List[Float32], f: List[Int], count: Int, seed: UInt64, watertight: Bool = True,
                     octree_depth: Int = 7) -> MaPointCloud:
    """The demo's input pipeline: watertight shell (optional), `count` area-uniform samples with
    face normals, normalisation."""
    var soup: List[Float32]
    if watertight:
        soup = watertight_soup(v, f, octree_depth)
    else:
        soup = List[Float32](capacity=len(f) * 3)
        for i in f:
            soup.append(v[i * 3])
            soup.append(v[i * 3 + 1])
            soup.append(v[i * 3 + 2])
    var rng = UrRng(seed * 0x9E3779B97F4A7C15 + 11)
    var xyz = List[Float32]()
    var nrm = List[Float32]()
    sample_soup(soup, count, rng, xyz, nrm)
    return normalize_point_cloud(xyz^, nrm^)
