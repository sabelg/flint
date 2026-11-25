"""Marching cubes and mesh packaging for `flint.ml.triposg`.

`marching_cubes` meshes the level set of a scalar grid without the classic 256-entry tables:
each cube's polygons are built by walking its faces. On a face (corners counter-clockwise seen
from outside) every sign-changing edge is an entry or an exit of the inside region; a segment
runs from each exit to the entry that closes that inside arc, and segments chain across
faces into loops (each crossing is an exit on one face and an entry on the other). A face with
four crossings is resolved by the bilinear centre value (the face's four corners summed in
grid-index order, so both cubes that share it decide the same way). Shared edge vertices plus
consistent face decisions make the mesh watertight; loops are fan-triangulated with the inside
behind the counter-clockwise front face (normals point out of the solid).

NOTE: compact local copy until the geometry kit's marching cubes (branch
claude/sad-bardeen-7abefd) lands on v5; dedupe then.
"""

from std.collections import Dict
from std.math import sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeModel, NativeMesh, NativeNode


struct TriMesh(Copyable, Movable):
    var positions: List[Float32]   # xyz per vertex
    var indices: List[Int]         # three per triangle

    def __init__(out self):
        self.positions = List[Float32]()
        self.indices = List[Int]()

    def vertex_count(self) -> Int:
        return len(self.positions) // 3

    def triangle_count(self) -> Int:
        return len(self.indices) // 3


def _faces() -> List[Int]:
    """Cube faces as corners (c = dx + 2 dy + 4 dz) counter-clockwise seen from outside:
    -x, +x, -y, +y, -z, +z."""
    var f: List[Int] = [0, 4, 6, 2, 1, 3, 7, 5, 0, 1, 5, 4, 2, 6, 7, 3, 0, 2, 3, 1, 4, 5, 7, 6]
    return f^


def _edge_local(a: Int, b: Int) -> Int:
    """Cube edge id 0..11 of corners a, b (differing in one bit): axis * 4 + the other two bits."""
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


def marching_cubes(values: List[Float32], n: Int, origin: Float32, spacing: Float32, level: Float32 = 0.0,
                   flip: Bool = False) -> TriMesh:
    """Triangle mesh of {v = level} for an n^3 grid (index (i * n + j) * n + k, point
    origin + (i, j, k) * spacing); "inside" is v > level (`flip` swaps it)."""
    var mesh = TriMesh()
    var verts = Dict[Int, Int]()
    var faces = _faces()
    var e_a = List[Int](length=12, fill=0)
    var e_b = List[Int](length=12, fill=0)
    for a in range(8):
        for bit in [1, 2, 4]:
            var b = a | bit
            if b != a:
                var e = _edge_local(a, b)
                e_a[e] = a
                e_b[e] = b
    var nxt = List[Int](length=12, fill=-1)
    var vid = List[Int](length=12, fill=-1)
    var cv = List[Float32](length=8, fill=0.0)
    var ci = List[Int](length=8, fill=0)
    var inside = List[Bool](length=8, fill=False)
    var cross = List[Int](length=4, fill=0)
    var ctype = List[Bool](length=4, fill=False)   # True = exit (inside -> outside)
    var loop = List[Int]()
    var nn = n * n
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                var mask = 0
                for c in range(8):
                    var idx = (i + (c & 1)) * nn + (j + ((c >> 1) & 1)) * n + k + ((c >> 2) & 1)
                    ci[c] = idx
                    var v = values[idx]
                    cv[c] = v
                    var ins = (v > level) != flip
                    inside[c] = ins
                    if ins:
                        mask |= 1 << c
                if mask == 0 or mask == 255:
                    continue
                for e in range(12):
                    nxt[e] = -1
                    vid[e] = -1
                for f in range(6):
                    var nc = 0
                    for s in range(4):
                        var a = faces[f * 4 + s]
                        var b = faces[f * 4 + (s + 1) % 4]
                        if inside[a] != inside[b]:
                            cross[nc] = _edge_local(a, b)
                            ctype[nc] = inside[a]
                            nc += 1
                    if nc == 2:
                        if ctype[0]:
                            nxt[cross[0]] = cross[1]
                        else:
                            nxt[cross[1]] = cross[0]
                    elif nc == 4:
                        # rotate so the order is E1, L1, E2, L2
                        var s0 = 0 if not ctype[0] else 1
                        var e1 = cross[s0]
                        var l1 = cross[(s0 + 1) % 4]
                        var e2 = cross[(s0 + 2) % 4]
                        var l2 = cross[(s0 + 3) % 4]
                        # centre value: corners summed in grid-index order (shared-face consistency)
                        var ids = List[Int]()
                        for s in range(4):
                            ids.append(ci[faces[f * 4 + s]])
                        sort(ids)
                        var tot: Float32 = 0
                        for s in range(4):
                            tot += values[ids[s]]
                        var centre_in = ((tot * 0.25) > level) != flip
                        if centre_in:
                            nxt[l1] = e2
                            nxt[l2] = e1
                        else:
                            nxt[l1] = e1
                            nxt[l2] = e2
                for e in range(12):
                    if nxt[e] < 0 or vid[e] == -2:
                        continue
                    loop.clear()
                    var cur = e
                    while cur >= 0 and vid[cur] != -2:
                        vid[cur] = -2
                        var a = e_a[cur]
                        var b = e_b[cur]
                        var axis = cur // 4
                        var key = ci[a] * 3 + axis
                        var got = verts.get(key)
                        var vi: Int
                        if got:
                            vi = got.value()
                        else:
                            vi = len(mesh.positions) // 3
                            verts[key] = vi
                            var va = cv[a]
                            var vb = cv[b]
                            var t = (level - va) / (vb - va) if vb != va else Float32(0.5)
                            var pa = List[Float32](length=3, fill=0.0)
                            pa[0] = Float32(i + (a & 1))
                            pa[1] = Float32(j + ((a >> 1) & 1))
                            pa[2] = Float32(k + ((a >> 2) & 1))
                            pa[axis] += t
                            for d in range(3):
                                mesh.positions.append(origin + pa[d] * spacing)
                        loop.append(vi)
                        cur = nxt[cur]
                    for s in range(1, len(loop) - 1):
                        mesh.indices.append(loop[0])
                        mesh.indices.append(loop[s + 1])
                        mesh.indices.append(loop[s])
    return mesh^


def signed_volume(m: TriMesh) -> Float64:
    """Divergence-theorem volume; positive when the normals point out of the solid."""
    var v: Float64 = 0
    var p = m.positions.copy()
    for t in range(m.triangle_count()):
        var a = m.indices[3 * t] * 3
        var b = m.indices[3 * t + 1] * 3
        var c = m.indices[3 * t + 2] * 3
        var ax = Float64(p[a])
        var ay = Float64(p[a + 1])
        var az = Float64(p[a + 2])
        var bx = Float64(p[b])
        var by = Float64(p[b + 1])
        var bz = Float64(p[b + 2])
        var cx = Float64(p[c])
        var cy = Float64(p[c + 1])
        var cz = Float64(p[c + 2])
        v += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return v / 6.0


def boundary_edges(m: TriMesh) -> Int:
    """Directed edges without their opposite twin: 0 for a closed, consistently wound surface."""
    var count = Dict[Int, Int]()
    var nv = m.vertex_count()
    for t in range(m.triangle_count()):
        for s in range(3):
            var a = m.indices[3 * t + s]
            var b = m.indices[3 * t + (s + 1) % 3]
            var key = a * nv + b
            count[key] = count.get(key).or_else(0) + 1
    var bad = 0
    for t in range(m.triangle_count()):
        for s in range(3):
            var a = m.indices[3 * t + s]
            var b = m.indices[3 * t + (s + 1) % 3]
            if count.get(b * nv + a).or_else(0) != count.get(a * nv + b).or_else(0):
                bad += 1
    return bad


def vertex_normals(m: TriMesh) -> List[Float32]:
    var nv = m.vertex_count()
    var nrm = List[Float32](length=nv * 3, fill=0.0)
    for t in range(m.triangle_count()):
        var a = m.indices[3 * t] * 3
        var b = m.indices[3 * t + 1] * 3
        var c = m.indices[3 * t + 2] * 3
        var ux = m.positions[b] - m.positions[a]
        var uy = m.positions[b + 1] - m.positions[a + 1]
        var uz = m.positions[b + 2] - m.positions[a + 2]
        var vx = m.positions[c] - m.positions[a]
        var vy = m.positions[c + 1] - m.positions[a + 1]
        var vz = m.positions[c + 2] - m.positions[a + 2]
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        for q in [a, b, c]:
            nrm[q] += nx
            nrm[q + 1] += ny
            nrm[q + 2] += nz
    for v in range(nv):
        var l = sqrt(nrm[3 * v] ** 2 + nrm[3 * v + 1] ** 2 + nrm[3 * v + 2] ** 2)
        if l > 0:
            for d in range(3):
                nrm[3 * v + d] /= l
        else:
            nrm[3 * v + 1] = 1.0
    return nrm^


def chamfer_distance(a: List[Float32], b: List[Float32], cell: Float32 = 0.02) -> Float64:
    """Symmetric mean nearest-neighbour distance between two point sets (xyz lists), via a hash grid."""
    return (_one_way(a, b, cell) + _one_way(b, a, cell)) * 0.5


def _cell_key(x: Float32, y: Float32, z: Float32, cell: Float32) -> Int:
    var i = Int((x + 2.0) / cell)
    var j = Int((y + 2.0) / cell)
    var k = Int((z + 2.0) / cell)
    return (i * 4096 + j) * 4096 + k


def _one_way(src: List[Float32], dst: List[Float32], cell: Float32) -> Float64:
    var buckets = Dict[Int, List[Int]]()
    var nd = len(dst) // 3
    for p in range(nd):
        var key = _cell_key(dst[3 * p], dst[3 * p + 1], dst[3 * p + 2], cell)
        if key in buckets:
            try:
                buckets[key].append(p)
            except:
                pass
        else:
            var l = List[Int]()
            l.append(p)
            buckets[key] = l^
    var total: Float64 = 0
    var ns = len(src) // 3
    for p in range(ns):
        var x = src[3 * p]
        var y = src[3 * p + 1]
        var z = src[3 * p + 2]
        var best: Float32 = 1.0e30
        var ring = 1
        while True:
            var ci = Int((x + 2.0) / cell)
            var cj = Int((y + 2.0) / cell)
            var ck = Int((z + 2.0) / cell)
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    for dk in range(-ring, ring + 1):
                        var key = ((ci + di) * 4096 + cj + dj) * 4096 + ck + dk
                        var got = buckets.get(key)
                        if got:
                            for q in got.value():
                                var dx = dst[3 * q] - x
                                var dy = dst[3 * q + 1] - y
                                var dz = dst[3 * q + 2] - z
                                best = min(best, dx * dx + dy * dy + dz * dz)
            if best <= (Float32(ring) * cell) ** 2 or ring > 64:
                break
            ring += 1
        total += Float64(sqrt(best))
    return total / Float64(max(ns, 1))


def mesh_to_model(m: TriMesh, name: String = "TripoSG") -> NativeModel:
    """A one-node static model (vertex normals, light grey material) for `write_native_glb`."""
    var model = NativeModel()
    var mesh = NativeMesh()
    mesh.positions = m.positions.copy()
    mesh.normals = vertex_normals(m)
    mesh.indices = m.indices.copy()
    mesh.base_color = 0xFFD0D0D0
    mesh.roughness = 0.6
    model.meshes.append(mesh^)
    var node = NativeNode()
    node.name = name
    node.mesh = 0
    node.mesh_indices.append(0)
    model.nodes.append(node^)
    model.roots.append(0)
    return model^


def render_preview(m: TriMesh, width: Int, height: Int, yaw_deg: Float32 = 0.0, pitch_deg: Float32 = 10.0) -> List[UInt32]:
    """CPU z-buffered Lambert render (orthographic, y up, camera on +z after yaw / pitch) as
    0xAARRGGBB pixels: a quick look at a generated mesh without a GPU window."""
    from std.math import sin, cos
    var px = List[UInt32](length=width * height, fill=0xFF2B2F36)
    var zb = List[Float32](length=width * height, fill=-1.0e30)
    var ya = yaw_deg * 0.017453292
    var pa = pitch_deg * 0.017453292
    var cy = cos(ya)
    var sy = sin(ya)
    var cp = cos(pa)
    var sp = sin(pa)
    var nv = m.vertex_count()
    var sx = List[Float32](length=nv, fill=0.0)
    var syv = List[Float32](length=nv, fill=0.0)
    var sz = List[Float32](length=nv, fill=0.0)
    var scale = Float32(min(width, height)) * 0.46
    for v in range(nv):
        var x = m.positions[3 * v]
        var y = m.positions[3 * v + 1]
        var z = m.positions[3 * v + 2]
        var x1 = cy * x + sy * z
        var z1 = -sy * x + cy * z
        var y2 = cp * y - sp * z1
        var z2 = sp * y + cp * z1
        sx[v] = Float32(width) * 0.5 + x1 * scale
        syv[v] = Float32(height) * 0.5 - y2 * scale
        sz[v] = z2
    var lx: Float32 = -0.35
    var ly: Float32 = 0.55
    var lz: Float32 = 0.76
    for t in range(m.triangle_count()):
        var a = m.indices[3 * t]
        var b = m.indices[3 * t + 1]
        var c = m.indices[3 * t + 2]
        var ux = sx[b] - sx[a]
        var uy = syv[b] - syv[a]
        var vx = sx[c] - sx[a]
        var vy = syv[c] - syv[a]
        var area = ux * vy - uy * vx
        if area >= 0:      # back face (screen y points down)
            continue
        # view-space normal for shading
        var ex = sx[b] - sx[a]
        var ey = -(syv[b] - syv[a])
        var ez = (sz[b] - sz[a]) * scale
        var fx = sx[c] - sx[a]
        var fy = -(syv[c] - syv[a])
        var fz = (sz[c] - sz[a]) * scale
        var nx = ey * fz - ez * fy
        var ny = ez * fx - ex * fz
        var nz = ex * fy - ey * fx
        var nl = sqrt(nx * nx + ny * ny + nz * nz)
        if nl <= 0:
            continue
        var lam = max(Float32(0), (nx * lx + ny * ly + nz * lz) / nl)
        var shade = Float32(0.18) + 0.82 * lam
        var col = UInt32(Int(min(Float32(255), 225.0 * shade)))
        var colg = UInt32(Int(min(Float32(255), 214.0 * shade)))
        var colb = UInt32(Int(min(Float32(255), 198.0 * shade)))
        var x0 = max(0, Int(min(sx[a], min(sx[b], sx[c]))))
        var x1 = min(width - 1, Int(max(sx[a], max(sx[b], sx[c]))) + 1)
        var y0 = max(0, Int(min(syv[a], min(syv[b], syv[c]))))
        var y1 = min(height - 1, Int(max(syv[a], max(syv[b], syv[c]))) + 1)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                var qx = Float32(xx) + 0.5
                var qy = Float32(yy) + 0.5
                var w0 = ((sx[b] - qx) * (syv[c] - qy) - (syv[b] - qy) * (sx[c] - qx)) / area
                var w1 = ((sx[c] - qx) * (syv[a] - qy) - (syv[c] - qy) * (sx[a] - qx)) / area
                var w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                var z = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
                var k = yy * width + xx
                if z > zb[k]:
                    zb[k] = z
                    px[k] = 0xFF000000 | (col << 16) | (colg << 8) | colb
    return px^
