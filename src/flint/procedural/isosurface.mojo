"""Iso-surface extraction from SDF grids for Flint's geometry kit (`flint.procedural.isosurface`).

Grids:
  - `SdfGrid`: dense `nx * ny * nz` samples (index `(k * ny + j) * nx + i`), world point
    `origin + (i, j, k) * spacing`.
  - `SparseSdfGrid`: 8^3 bricks allocated on demand in a hash map; unallocated space reads as
    `background` (a large positive distance = outside). Image-to-3D decoders (TripoSG-style
    occupancy / SDF queries) only fill bricks near the surface.

Extractors (both produce watertight, outward-facing triangle meshes for closed level sets):
  - `marching_cubes`: Lorensen & Cline 1987 cell polygonisation, but table-free: each cube's
    polygons are built by walking its six faces (an inside arc on a face runs from an entry to an
    exit crossing; segments join each exit to the entry that closes the outside region, and chain
    across faces into loops). Ambiguous faces (four crossings) are resolved by the face-centre
    bilinear value, computed from the face's four samples in a canonical order so both cubes
    that share the face agree - this is what makes the result crack-free (the asymptotic-decider
    idea of Nielson & Hamann 1991, in its simplest consistent form). Vertices are shared per grid
    edge; normals come from the central-difference gradient.
  - `dual_contouring`: Ju, Losasso, Schaefer & Warren 2002. One vertex per sign-changing cell
    placed at the minimiser of the QEF sum (n_i . (x - p_i))^2 over its edge crossings (Hermite
    data from the gradient), regularised toward the crossings' mass point and clamped to the
    cell; one quad per sign-changing grid edge between its four cells. Keeps sharp edges and
    corners that marching cubes bevels.

Conventions: `inside_negative = True` treats `value < level` as solid (signed distance fields);
pass False for occupancy / density fields where larger means inside.
"""
from std.math import abs, max, min, sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.mesh_kit import EdgeMap, recompute_normals, unit


comptime BRICK: Int = 8
comptime BRICK3: Int = BRICK * BRICK * BRICK


trait ScalarField:
    def value(self, i: Int, j: Int, k: Int) -> Float32:
        ...

    def point(self, i: Int, j: Int, k: Int) -> Vec3:
        ...


struct SdfGrid(Copyable, Movable, ScalarField):
    var nx: Int
    var ny: Int
    var nz: Int
    var origin: Vec3
    var spacing: Float32
    var values: List[Float32]

    def __init__(out self, nx: Int, ny: Int, nz: Int, origin: Vec3, spacing: Float32, fill: Float32 = 1.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.origin = origin
        self.spacing = spacing
        self.values = List[Float32](length=nx * ny * nz, fill=fill)

    @always_inline
    def index(self, i: Int, j: Int, k: Int) -> Int:
        return (k * self.ny + j) * self.nx + i

    def value(self, i: Int, j: Int, k: Int) -> Float32:
        var ci = max(0, min(self.nx - 1, i))
        var cj = max(0, min(self.ny - 1, j))
        var ck = max(0, min(self.nz - 1, k))
        return self.values[self.index(ci, cj, ck)]

    def point(self, i: Int, j: Int, k: Int) -> Vec3:
        return self.origin + Vec3(Float32(i), Float32(j), Float32(k)) * self.spacing

    def set(mut self, i: Int, j: Int, k: Int, v: Float32):
        self.values[self.index(i, j, k)] = v


@always_inline
def _brick_key(bx: Int, by: Int, bz: Int) -> UInt64:
    return (UInt64(bx & 0x1FFFFF) << 42) | (UInt64(by & 0x1FFFFF) << 21) | UInt64(bz & 0x1FFFFF)


@always_inline
def _floor_div(a: Int, b: Int) -> Int:
    return a // b if a >= 0 else -((-a + b - 1) // b)


struct SparseSdfGrid(Movable, ScalarField):
    var origin: Vec3
    var spacing: Float32
    var background: Float32
    var map: EdgeMap
    var brick_coords: List[Int]   # 3 per brick
    var data: List[Float32]       # BRICK3 per brick
    var tiles: List[Float32]      # uniform (inactive) bricks, e.g. the solid interior

    def __init__(out self, origin: Vec3, spacing: Float32, background: Float32 = 1.0e9):
        self.origin = origin
        self.spacing = spacing
        self.background = background
        self.map = EdgeMap(64)
        self.brick_coords = List[Int]()
        self.data = List[Float32]()
        self.tiles = List[Float32]()

    def set_tile(mut self, bx: Int, by: Int, bz: Int, v: Float32):
        """Marks a whole brick as one constant value without storing voxels (an OpenVDB tile)."""
        var key = _brick_key(bx, by, bz)
        if self.map.find(key) >= 0:
            return
        self.map.set(key, -2 - len(self.tiles))
        self.tiles.append(v)

    def brick_count(self) -> Int:
        return len(self.brick_coords) // 3

    def _brick(mut self, bx: Int, by: Int, bz: Int) -> Int:
        var key = _brick_key(bx, by, bz)
        var b = self.map.find(key)
        if b >= 0:
            return b
        var tile_v = self.background
        if b <= -2:
            tile_v = self.tiles[-2 - b]
        b = self.brick_count()
        self.map.set(key, b)
        self.brick_coords.append(bx)
        self.brick_coords.append(by)
        self.brick_coords.append(bz)
        for _ in range(BRICK3):
            self.data.append(tile_v)
        return b

    def set(mut self, i: Int, j: Int, k: Int, v: Float32):
        var bx = _floor_div(i, BRICK)
        var by = _floor_div(j, BRICK)
        var bz = _floor_div(k, BRICK)
        var b = self._brick(bx, by, bz)
        var li = i - bx * BRICK
        var lj = j - by * BRICK
        var lk = k - bz * BRICK
        self.data[b * BRICK3 + (lk * BRICK + lj) * BRICK + li] = v

    def value(self, i: Int, j: Int, k: Int) -> Float32:
        var bx = _floor_div(i, BRICK)
        var by = _floor_div(j, BRICK)
        var bz = _floor_div(k, BRICK)
        var b = self.map.find(_brick_key(bx, by, bz))
        if b < 0:
            return self.background if b == -1 else self.tiles[-2 - b]
        var li = i - bx * BRICK
        var lj = j - by * BRICK
        var lk = k - bz * BRICK
        return self.data[b * BRICK3 + (lk * BRICK + lj) * BRICK + li]

    def point(self, i: Int, j: Int, k: Int) -> Vec3:
        return self.origin + Vec3(Float32(i), Float32(j), Float32(k)) * self.spacing


# ------------------------------------------------------------------------------- analytic fills

def sdf_sphere(p: Vec3, c: Vec3, r: Float32) -> Float32:
    return (p - c).length() - r


def sdf_box(p: Vec3, c: Vec3, half: Vec3) -> Float32:
    var d = p - c
    var q = Vec3(abs(d.x) - half.x, abs(d.y) - half.y, abs(d.z) - half.z)
    var outside = Vec3(max(q.x, 0), max(q.y, 0), max(q.z, 0)).length()
    return outside + min(max(q.x, max(q.y, q.z)), Float32(0))


def sdf_torus(p: Vec3, c: Vec3, major: Float32, minor: Float32) -> Float32:
    var d = p - c
    var qx = sqrt(d.x * d.x + d.z * d.z) - major
    return sqrt(qx * qx + d.y * d.y) - minor


def grid_from_sphere(n: Int, radius: Float32, extent: Float32) -> SdfGrid:
    var sp = 2.0 * extent / Float32(n - 1)
    var g = SdfGrid(n, n, n, Vec3(-extent, -extent, -extent), sp)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                g.set(i, j, k, sdf_sphere(g.point(i, j, k), Vec3(0, 0, 0), radius))
    return g^


def grid_from_box(n: Int, half: Vec3, extent: Float32) -> SdfGrid:
    var sp = 2.0 * extent / Float32(n - 1)
    var g = SdfGrid(n, n, n, Vec3(-extent, -extent, -extent), sp)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                g.set(i, j, k, sdf_box(g.point(i, j, k), Vec3(0, 0, 0), half))
    return g^


def grid_from_torus(n: Int, major: Float32, minor: Float32, extent: Float32) -> SdfGrid:
    var sp = 2.0 * extent / Float32(n - 1)
    var g = SdfGrid(n, n, n, Vec3(-extent, -extent, -extent), sp)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                g.set(i, j, k, sdf_torus(g.point(i, j, k), Vec3(0, 0, 0), major, minor))
    return g^


def sparse_from_dense(g: SdfGrid, band: Float32) -> SparseSdfGrid:
    """Narrow-band copy: bricks holding any |value| <= band store their voxels; bricks entirely
    inside become negative tiles; everything else is the (outside) background."""
    var s = SparseSdfGrid(g.origin, g.spacing, 1.0e9)
    var nbx = (g.nx + BRICK - 1) // BRICK
    var nby = (g.ny + BRICK - 1) // BRICK
    var nbz = (g.nz + BRICK - 1) // BRICK
    for bz in range(nbz):
        for by in range(nby):
            for bx in range(nbx):
                var near = False
                var all_in = True
                for lk in range(BRICK):
                    for lj in range(BRICK):
                        for li in range(BRICK):
                            var v = g.value(bx * BRICK + li, by * BRICK + lj, bz * BRICK + lk)
                            if abs(v) <= band:
                                near = True
                            if v >= 0:
                                all_in = False
                if near:
                    for lk in range(BRICK):
                        for lj in range(BRICK):
                            for li in range(BRICK):
                                var i = bx * BRICK + li
                                var j = by * BRICK + lj
                                var k = bz * BRICK + lk
                                s.set(i, j, k, g.value(i, j, k))
                elif all_in:
                    s.set_tile(bx, by, bz, -s.background)
    return s^


# ------------------------------------------------------------------------------- shared cell data



@always_inline
def _corner(c: Int) -> Tuple[Int, Int, Int]:
    return (c & 1, (c >> 1) & 1, (c >> 2) & 1)


def _edge_id(a: Int, b: Int) -> Int:
    """Cube edge 0..11 for corners a, b (c = dx + 2 dy + 4 dz): axis * 4 + the other two bits."""
    var lo = min(a, b)
    var d = a ^ b
    var axis = 0 if d == 1 else (1 if d == 2 else 2)
    var rest: Int
    if axis == 0:
        rest = (lo >> 1) & 3
    elif axis == 1:
        rest = (lo & 1) | (((lo >> 2) & 1) << 1)
    else:
        rest = lo & 3
    return axis * 4 + rest


@always_inline
def _gkey(i: Int, j: Int, k: Int, axis: Int) -> UInt64:
    return (UInt64((i + 1) & 0xFFFFF) << 42) | (UInt64((j + 1) & 0xFFFFF) << 22) | (UInt64((k + 1) & 0xFFFFF) << 2) | UInt64(axis)


def _gradient[F: ScalarField](f: F, i: Int, j: Int, k: Int) -> Vec3:
    return Vec3(f.value(i + 1, j, k) - f.value(i - 1, j, k),
                f.value(i, j + 1, k) - f.value(i, j - 1, k),
                f.value(i, j, k + 1) - f.value(i, j, k - 1))


struct _CellTables(Movable):
    var e_a: List[Int]
    var e_b: List[Int]
    var faces: List[Int]

    def __init__(out self):
        self.e_a = List[Int](length=12, fill=0)
        self.e_b = List[Int](length=12, fill=0)
        # cube faces as corners counter-clockwise seen from outside: -x, +x, -y, +y, -z, +z
        self.faces = [0, 4, 6, 2, 1, 3, 7, 5, 0, 1, 5, 4, 2, 6, 7, 3, 0, 2, 3, 1, 4, 5, 7, 6]
        for a in range(8):
            for bit in [1, 2, 4]:
                var b = a | bit
                if b != a:
                    var e = _edge_id(a, b)
                    self.e_a[e] = a
                    self.e_b[e] = b


# ------------------------------------------------------------------------------- marching cubes

struct _McState(Movable):
    var mesh: NativeMesh
    var verts: EdgeMap
    var tab: _CellTables
    var nxt: List[Int]
    var vid: List[Int]
    var cv: List[Float32]
    var ins: List[Bool]
    var loop: List[Int]

    def __init__(out self, expected: Int):
        self.mesh = NativeMesh()
        self.verts = EdgeMap(expected)
        self.tab = _CellTables()
        self.nxt = List[Int](length=12, fill=-1)
        self.vid = List[Int](length=12, fill=-1)
        self.cv = List[Float32](length=8, fill=0.0)
        self.ins = List[Bool](length=8, fill=False)
        self.loop = List[Int]()


def _mc_cell[F: ScalarField](f: F, i: Int, j: Int, k: Int, level: Float32, inside_negative: Bool, mut st: _McState):
    var mask = 0
    for c in range(8):
        var o = _corner(c)
        var v = f.value(i + o[0], j + o[1], k + o[2])
        st.cv[c] = v
        var inside = (v < level) if inside_negative else (v > level)
        st.ins[c] = inside
        if inside:
            mask |= 1 << c
    if mask == 0 or mask == 255:
        return
    for e in range(12):
        st.nxt[e] = -1
        st.vid[e] = -1
    var cross = List[Int](length=4, fill=0)
    var is_exit = List[Bool](length=4, fill=False)
    for fc in range(6):
        var nc = 0
        for s in range(4):
            var a = st.tab.faces[fc * 4 + s]
            var b = st.tab.faces[fc * 4 + (s + 1) % 4]
            if st.ins[a] != st.ins[b]:
                cross[nc] = _edge_id(a, b)
                is_exit[nc] = st.ins[a]
                nc += 1
        if nc == 2:
            if is_exit[0]:
                st.nxt[cross[0]] = cross[1]
            else:
                st.nxt[cross[1]] = cross[0]
        elif nc == 4:
            var s0 = 0 if not is_exit[0] else 1
            var e1 = cross[s0]
            var l1 = cross[(s0 + 1) % 4]
            var e2 = cross[(s0 + 2) % 4]
            var l2 = cross[(s0 + 3) % 4]
            # face centre value, summed in a canonical (sorted) order so neighbours agree exactly
            var vals = List[Float32](capacity=4)
            for s in range(4):
                vals.append(st.cv[st.tab.faces[fc * 4 + s]])
            sort(vals)
            var centre = ((vals[0] + vals[1]) + (vals[2] + vals[3])) * 0.25
            var centre_in = (centre < level) if inside_negative else (centre > level)
            if centre_in:
                st.nxt[l1] = e2
                st.nxt[l2] = e1
            else:
                st.nxt[l1] = e1
                st.nxt[l2] = e2
    for e in range(12):
        if st.nxt[e] < 0 or st.vid[e] == -2:
            continue
        st.loop.clear()
        var cur = e
        while cur >= 0 and st.vid[cur] != -2:
            st.vid[cur] = -2
            var a = st.tab.e_a[cur]
            var axis = cur // 4
            var oa = _corner(a)
            var gi = i + oa[0]
            var gj = j + oa[1]
            var gk = k + oa[2]
            var key = _gkey(gi, gj, gk, axis)
            var vi = st.verts.insert(key, st.mesh.vertex_count())
            if vi == st.mesh.vertex_count():
                var b = st.tab.e_b[cur]
                var va = st.cv[a]
                var vb = st.cv[b]
                var t = (level - va) / (vb - va) if vb != va else Float32(0.5)
                t = max(Float32(0), min(Float32(1), t))
                var ob = _corner(b)
                var pa = f.point(gi, gj, gk)
                var pb = f.point(i + ob[0], j + ob[1], k + ob[2])
                var p = pa + (pb - pa) * t
                st.mesh.positions.append(p.x); st.mesh.positions.append(p.y); st.mesh.positions.append(p.z)
                var ga = _gradient(f, gi, gj, gk)
                var gb = _gradient(f, i + ob[0], j + ob[1], k + ob[2])
                var n = unit(ga + (gb - ga) * t, Vec3(0, 1, 0))
                if not inside_negative:
                    n = -n
                st.mesh.normals.append(n.x); st.mesh.normals.append(n.y); st.mesh.normals.append(n.z)
            st.loop.append(vi)
            cur = st.nxt[cur]
        for s in range(1, len(st.loop) - 1):
            st.mesh.indices.append(st.loop[0])
            st.mesh.indices.append(st.loop[s + 1])
            st.mesh.indices.append(st.loop[s])


def marching_cubes(g: SdfGrid, level: Float32 = 0.0, inside_negative: Bool = True) -> NativeMesh:
    var st = _McState(g.nx * g.ny * 4)
    for k in range(g.nz - 1):
        for j in range(g.ny - 1):
            for i in range(g.nx - 1):
                _mc_cell(g, i, j, k, level, inside_negative, st)
    return st.mesh.copy()


def _sparse_cells(s: SparseSdfGrid) -> List[Int]:
    """Cells (min corners, 3 ints each) whose 8 corners touch an allocated brick, deduplicated."""
    var out = List[Int]()
    var seen = EdgeMap(s.brick_count() * BRICK3 * 2)
    for b in range(s.brick_count()):
        var bx = s.brick_coords[b * 3]
        var by = s.brick_coords[b * 3 + 1]
        var bz = s.brick_coords[b * 3 + 2]
        for lk in range(-1, BRICK):
            for lj in range(-1, BRICK):
                for li in range(-1, BRICK):
                    var i = bx * BRICK + li
                    var j = by * BRICK + lj
                    var k = bz * BRICK + lk
                    var border = li < 0 or lj < 0 or lk < 0
                    if border:
                        # owned by its own brick when that brick stores voxels
                        if s.map.find(_brick_key(_floor_div(i, BRICK), _floor_div(j, BRICK), _floor_div(k, BRICK))) >= 0:
                            continue
                        var key = _gkey(i, j, k, 0)
                        if seen.find(key) >= 0:
                            continue
                        seen.set(key, 1)
                    out.append(i); out.append(j); out.append(k)
    return out^


def marching_cubes_sparse(s: SparseSdfGrid, level: Float32 = 0.0, inside_negative: Bool = True) -> NativeMesh:
    var st = _McState(s.brick_count() * BRICK * BRICK * 4)
    var cells = _sparse_cells(s)
    for c in range(len(cells) // 3):
        _mc_cell(s, cells[c * 3], cells[c * 3 + 1], cells[c * 3 + 2], level, inside_negative, st)
    return st.mesh.copy()


def marching_cubes_values(values: List[Float32], n: Int, origin: Float32, spacing: Float32, level: Float32 = 0.0,
                          flip: Bool = False) -> NativeMesh:
    """Drop-in for `flint.ml.triposg_mesh.marching_cubes`: an n^3 grid indexed `(i * n + j) * n + k`
    at `origin + (i, j, k) * spacing`, inside where v > level (`flip` swaps it)."""
    var g = SdfGrid(n, n, n, Vec3(origin, origin, origin), spacing)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # transpose (i, j, k) -> dense x = i, y = j, z = k
                g.values[g.index(i, j, k)] = values[(i * n + j) * n + k]
    return marching_cubes(g, level, flip)


# ------------------------------------------------------------------------------- dual contouring

def _solve_qef(ata: List[Float64], atb: Vec3, mass: Vec3, count: Int, reg: Float64) -> Vec3:
    """Minimise |A x - b|^2 + reg * count * |x - mass|^2 (symmetric 3x3, Cramer's rule)."""
    var w = reg * Float64(count)
    var a00 = ata[0] + w
    var a01 = ata[1]
    var a02 = ata[2]
    var a11 = ata[3] + w
    var a12 = ata[4]
    var a22 = ata[5] + w
    var bx = Float64(atb.x) + w * Float64(mass.x)
    var by = Float64(atb.y) + w * Float64(mass.y)
    var bz = Float64(atb.z) + w * Float64(mass.z)
    var c00 = a11 * a22 - a12 * a12
    var c01 = a02 * a12 - a01 * a22
    var c02 = a01 * a12 - a02 * a11
    var det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) < 1e-30:
        return mass
    var c11 = a00 * a22 - a02 * a02
    var c12 = a01 * a02 - a00 * a12
    var c22 = a00 * a11 - a01 * a01
    var x = (c00 * bx + c01 * by + c02 * bz) / det
    var y = (c01 * bx + c11 * by + c12 * bz) / det
    var z = (c02 * bx + c12 * by + c22 * bz) / det
    return Vec3(Float32(x), Float32(y), Float32(z))


def _dc_vertex[F: ScalarField](f: F, i: Int, j: Int, k: Int, level: Float32, tab: _CellTables, reg: Float64) -> Tuple[Bool, Vec3]:
    var ata = List[Float64](length=6, fill=0.0)
    var atb = Vec3(0, 0, 0)
    var mass = Vec3(0, 0, 0)
    var count = 0
    var cv = List[Float32](length=8, fill=0.0)
    for c in range(8):
        var o = _corner(c)
        cv[c] = f.value(i + o[0], j + o[1], k + o[2])
    for e in range(12):
        var a = tab.e_a[e]
        var b = tab.e_b[e]
        if (cv[a] < level) == (cv[b] < level):
            continue
        var t = (level - cv[a]) / (cv[b] - cv[a])
        var oa = _corner(a)
        var ob = _corner(b)
        var pa = f.point(i + oa[0], j + oa[1], k + oa[2])
        var pb = f.point(i + ob[0], j + ob[1], k + ob[2])
        var p = pa + (pb - pa) * t
        var ga = _gradient(f, i + oa[0], j + oa[1], k + oa[2])
        var gb = _gradient(f, i + ob[0], j + ob[1], k + ob[2])
        var n = unit(ga + (gb - ga) * t, Vec3(0, 0, 0))
        var nx = Float64(n.x)
        var ny = Float64(n.y)
        var nz = Float64(n.z)
        ata[0] += nx * nx; ata[1] += nx * ny; ata[2] += nx * nz
        ata[3] += ny * ny; ata[4] += ny * nz; ata[5] += nz * nz
        var d = n.dot(p)
        atb = atb + n * d
        mass = mass + p
        count += 1
    if count == 0:
        return (False, Vec3(0, 0, 0))
    mass = mass * (1.0 / Float32(count))
    var x = _solve_qef(ata, atb, mass, count, reg)
    # clamp into the cell (slightly padded); fall back to the mass point if the QEF ran away
    var lo = f.point(i, j, k)
    var hi = f.point(i + 1, j + 1, k + 1)
    var pad = (hi.x - lo.x) * 0.05
    if x.x < lo.x - pad or x.y < lo.y - pad or x.z < lo.z - pad or x.x > hi.x + pad or x.y > hi.y + pad or x.z > hi.z + pad:
        x = Vec3(max(lo.x, min(hi.x, x.x)), max(lo.y, min(hi.y, x.y)), max(lo.z, min(hi.z, x.z)))
    return (True, x)


def _dc_run[F: ScalarField](f: F, cells: List[Int], level: Float32, inside_negative: Bool, reg: Float64) -> NativeMesh:
    var tab = _CellTables()
    var out = NativeMesh()
    var cell_v = EdgeMap(len(cells))
    for c in range(len(cells) // 3):
        var i = cells[c * 3]
        var j = cells[c * 3 + 1]
        var k = cells[c * 3 + 2]
        var r = _dc_vertex(f, i, j, k, level, tab, reg)
        if r[0]:
            cell_v.set(_gkey(i, j, k, 0), out.vertex_count())
            out.positions.append(r[1].x); out.positions.append(r[1].y); out.positions.append(r[1].z)
    # one quad per sign-changing grid edge, between the four cells around it
    for c in range(len(cells) // 3):
        var i = cells[c * 3]
        var j = cells[c * 3 + 1]
        var k = cells[c * 3 + 2]
        var v0 = f.value(i, j, k)
        for axis in range(3):
            var di = 1 if axis == 0 else 0
            var dj = 1 if axis == 1 else 0
            var dk = 1 if axis == 2 else 0
            var v1 = f.value(i + di, j + dj, k + dk)
            var in0 = (v0 < level)
            var in1 = (v1 < level)
            if in0 == in1:
                continue
            # the 4 cells sharing edge (i,j,k)->(+axis): offsets in the two other axes
            var q = List[Int](length=4, fill=-1)
            var ok = True
            for s in range(4):
                var u = 1 if (s == 1 or s == 2) else 0
                var w = 1 if s >= 2 else 0
                var ci = i
                var cj = j
                var ck = k
                if axis == 0:
                    cj -= u; ck -= w
                elif axis == 1:
                    ck -= u; ci -= w
                else:
                    ci -= u; cj -= w
                var vid = cell_v.find(_gkey(ci, cj, ck, 0))
                if vid < 0:
                    ok = False
                    break
                q[s] = vid
            if not ok:
                continue
            # orientation: normal along +axis when the solid is on the low side of the edge
            var solid_low = in0 if inside_negative else in1
            var a = q[0]
            var b = q[1]
            var cc = q[2]
            var d = q[3]
            var pa = Vec3(out.positions[a * 3], out.positions[a * 3 + 1], out.positions[a * 3 + 2])
            var pb = Vec3(out.positions[b * 3], out.positions[b * 3 + 1], out.positions[b * 3 + 2])
            var pc = Vec3(out.positions[cc * 3], out.positions[cc * 3 + 1], out.positions[cc * 3 + 2])
            var pd = Vec3(out.positions[d * 3], out.positions[d * 3 + 1], out.positions[d * 3 + 2])
            var split_ac = (pa - pc).length_squared() <= (pb - pd).length_squared()
            if solid_low:
                if split_ac:
                    _tri(out, a, b, cc); _tri(out, a, cc, d)
                else:
                    _tri(out, a, b, d); _tri(out, b, cc, d)
            else:
                if split_ac:
                    _tri(out, a, cc, b); _tri(out, a, d, cc)
                else:
                    _tri(out, a, d, b); _tri(out, b, d, cc)
    recompute_normals(out, False)
    return out^


@always_inline
def _tri(mut m: NativeMesh, a: Int, b: Int, c: Int):
    m.indices.append(a)
    m.indices.append(b)
    m.indices.append(c)


def dual_contouring(g: SdfGrid, level: Float32 = 0.0, inside_negative: Bool = True, regularization: Float64 = 0.02) -> NativeMesh:
    var cells = List[Int](capacity=(g.nx - 1) * (g.ny - 1) * (g.nz - 1) * 3)
    for k in range(g.nz - 1):
        for j in range(g.ny - 1):
            for i in range(g.nx - 1):
                cells.append(i); cells.append(j); cells.append(k)
    if inside_negative:
        return _dc_run(g, cells, level, True, regularization)
    var neg = g.copy()
    for i in range(len(neg.values)):
        neg.values[i] = -neg.values[i]
    return _dc_run(neg, cells, -level, True, regularization)


def dual_contouring_sparse(s: SparseSdfGrid, level: Float32 = 0.0, regularization: Float64 = 0.02) -> NativeMesh:
    var cells = _sparse_cells(s)
    return _dc_run(s, cells, level, True, regularization)
