"""Hi3D sparse FlexiCubes extraction. Mojo adaptation, see licenses/hi3d."""
from std.math import exp, tanh, atan2
from flint.assets.model import NativeMesh, NativeModel, NativeNode
from flint.assets.meshlet import simplify_mesh_qem
from flint.core.detsort import stable_order
from flint.core.math import Vec3
from flint.ml.hi3d_sparse import SparseGrid
from flint.ml.hi3d_cube_tables import edge_groups, ambiguous_cases
from flint.ml.tensor import DT_F32, Tensor


@fieldwise_init
struct _Corner(Copyable, Movable):
    var prob: Float32
    var sdf: Float32
    var delta: Vec3
    var count: Int


def _xyz(g: SparseGrid, i: Int, corner: Int) -> Int:
    return (
        ((g.xyz.i[i * 3] + (corner & 1)) * 513 + g.xyz.i[i * 3 + 1] + ((corner >> 1) & 1)) * 513
        + g.xyz.i[i * 3 + 2]
        + ((corner >> 2) & 1)
    )


def _point(key: Int, c: _Corner) -> Vec3:
    return Vec3(Float32(key // (513 * 513)), Float32((key // 513) % 513), Float32(key % 513)) + c.delta


def _roots(mut parents: List[Int], i: Int) -> Int:
    var k = i
    while parents[k] != k:
        parents[k] = parents[parents[k]]
        k = parents[k]
    return k


def _close_and_filter(verts: List[Vec3], mut faces: List[Int]) raises:
    # Collect directed boundary edges and face adjacency on exactly two incident faces.
    var n = len(verts)
    var counts = Dict[Int, Int]()
    var owners = Dict[Int, Int]()
    var parents = List[Int](capacity=len(faces) // 3)
    for f in range(len(faces) // 3):
        parents.append(f)
        for j in range(3):
            var a = faces[3 * f + j]
            var b = faces[3 * f + (j + 1) % 3]
            var k = min(a, b) * n + max(a, b)
            counts[k] = counts.get(k, 0) + 1
            owners[k] = f
    var outgoing = Dict[Int, List[Int]]()
    for f in range(len(parents)):
        for j in range(3):
            var a = faces[3 * f + j]
            var b = faces[3 * f + (j + 1) % 3]
            var k = min(a, b) * n + max(a, b)
            if counts[k] == 2:
                var left = _roots(parents, f)
                var right = _roots(parents, owners[k])
                parents[left] = right
            elif counts[k] == 1:
                if a not in outgoing:
                    outgoing[a] = List[Int]()
                outgoing[a].append(b)
    var used = Dict[Int, Bool]()
    var original = len(parents)
    for f in range(original):
        for j in range(3):
            var start = faces[f * 3 + j]
            var end = faces[f * 3 + (j + 1) % 3]
            if counts[min(start, end) * n + max(start, end)] != 1 or start * n + end in used:
                continue
            var loop = List[Int]()
            loop.append(start)
            var a = start
            var b = end
            while True:
                used[a * n + b] = True
                if b == start:
                    break
                loop.append(b)
                if b not in outgoing:
                    break
                var next = -1
                for v in outgoing[b]:
                    if b * n + v not in used:
                        next = v
                        break
                if next < 0:
                    break
                a = b
                b = next
            if b != start or len(loop) < 3 or len(loop) > 10000:
                continue
            while len(loop) > 3:
                var mid = Vec3()
                for v in loop:
                    mid = mid + verts[v]
                mid = mid * (1 / Float32(len(loop)))
                var best = -1
                var top = Float32(-1e30)
                for k in range(len(loop)):
                    var prev = verts[loop[(k + len(loop) - 1) % len(loop)]]
                    var curr = verts[loop[k]]
                    var after = verts[loop[(k + 1) % len(loop)]]
                    var u = curr - prev
                    var v = after - curr
                    u = u * (1 / (u.length() + 1e-6))
                    v = v * (1 / (v.length() + 1e-6))
                    var cross = (-u).cross(v)
                    var s = (after - mid).cross(curr - mid).dot(cross)
                    var angle = atan2(
                        cross.length() * Float32(1 if s > 0 else (-1 if s < 0 else 0)), -u.dot(v)
                    )
                    var score = Float32(1e30) if angle == 0 else 1 / angle
                    if score > top:
                        best = k
                        top = score
                faces.append(loop[(best + 1) % len(loop)])
                faces.append(loop[best])
                faces.append(loop[(best + len(loop) - 1) % len(loop)])
                parents.append(f)
                _ = loop.pop(best)
            faces.append(loop[2])
            faces.append(loop[1])
            faces.append(loop[0])
            parents.append(f)
    var sizes = List[Int](length=len(parents), fill=0)
    var biggest = 0
    for f in range(len(parents)):
        var root = _roots(parents, f)
        parents[f] = root
        sizes[root] += 1
        if sizes[root] > sizes[biggest]:
            biggest = root
    var kept = List[Int](capacity=len(faces))
    for f in range(len(parents)):
        var root = parents[f]
        if root == biggest or sizes[root] * 200 >= len(parents):
            for j in range(3):
                kept.append(faces[3 * f + j])
    faces = kept^


def hi3d_surface(g: SparseGrid, attr: Tensor, postprocess: Bool = True) raises -> NativeMesh:
    """Extract the 512 geometry field and optionally clean, simplify and normalize the mesh."""
    var n = g.xyz.shape[0]
    if (
        g.edge != 512
        or attr.dtype != DT_F32
        or len(attr.shape) != 2
        or attr.shape[0] != n
        or attr.shape[1] != 40
        or len(attr.f) != n * 40
    ):
        raise Error("Expected [N,40] decoded attributes on the 512 grid")
    var keys = List[Int](capacity=n)
    var vertex_map = Dict[Int, Int]()
    var corners = List[_Corner]()
    for i in range(n):
        keys.append(_xyz(g, i, 0))
        for k in range(8):
            var key = _xyz(g, i, k)
            var id = vertex_map.get(key, -1)
            if id < 0:
                id = len(corners)
                vertex_map[key] = id
                corners.append(_Corner(0, 0, Vec3(), 0))
            ref c = corners[id]
            c.count += 1
            c.prob += 1 / (1 + exp(-attr.f[i * 40 + k]))
            c.sdf += tanh(attr.f[i * 40 + 8 + k])
            var j = i * 40 + 16 + 3 * k
            c.delta = c.delta + Vec3(tanh(attr.f[j]), tanh(attr.f[j + 1]), tanh(attr.f[j + 2])) * 0.5
    for i in range(len(corners)):
        ref c = corners[i]
        var inv = 1 / Float32(c.count)
        c.sdf = abs(c.sdf) * inv * Float32(1 if c.prob >= Float32(c.count) * 0.5 else -1)
        c.delta = c.delta * inv
    var cases = List[Int](length=n, fill=0)
    for i in range(n):
        for k in range(8):
            if corners[vertex_map[_xyz(g, i, k)]].sdf < 0:
                cases[i] |= 1 << k
    var check = ambiguous_cases()
    var groups = edge_groups()
    var order = stable_order(keys)
    var verts = List[Vec3]()
    var ends: List[Int] = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4]
    var quads = Dict[Int, List[Int]]()
    var positive = Dict[Int, Bool]()
    for i in order:
        var case_id = cases[i]
        if case_id == 0 or case_id == 255:
            continue
        var ci = 5 * case_id
        if check[ci] != 0:
            var adjacent = g.get(
                g.xyz.i[3 * i] + check[ci + 1],
                g.xyz.i[3 * i + 1] + check[ci + 2],
                g.xyz.i[3 * i + 2] + check[ci + 3],
            )
            if adjacent >= 0 and check[5 * cases[adjacent]] == 1:
                case_id = check[ci + 4]
        for group in range(4):
            var packed = groups[4 * case_id + group]
            if packed == 0:
                break
            var sum = Vec3()
            var count = 0
            while packed > 0:
                var edge = (packed & 15) - 1
                packed >>= 4
                var a = _xyz(g, i, ends[edge * 2])
                var b = _xyz(g, i, ends[edge * 2 + 1])
                ref c0 = corners[vertex_map[a]]
                ref c1 = corners[vertex_map[b]]
                var t = c0.sdf / (c0.sdf - c1.sdf)
                sum = sum + (_point(a, c0) * (1 - t) + _point(b, c1) * t)
                count += 1
                var axis = 0 if abs(a - b) == 1 else (1 if abs(a - b) == 513 else 2)
                var id = min(a, b) * 3 + axis
                if id not in quads:
                    quads[id] = List[Int]()
                    positive[id] = c0.sdf > 0
                quads[id].append(len(verts))
            verts.append(sum * (1 / Float32(count)))
    var faces = List[Int]()
    # Insertion into each edge group follows lexicographic cube order.
    # Sort the keys before output so exported files are deterministic.
    var edges = List[Int](capacity=len(quads))
    for key in quads.keys():
        edges.append(key)
    var edge_order = stable_order(edges)
    for i in edge_order:
        var id = edges[i]
        ref q = quads[id]
        if len(q) != 4:
            continue
        var a = q[0] if positive[id] else q[2]
        var b = q[1] if positive[id] else q[3]
        var c = q[3] if positive[id] else q[1]
        var d = q[2] if positive[id] else q[0]
        faces.append(a)
        faces.append(b)
        faces.append(d)
        faces.append(d)
        faces.append(b)
        faces.append(c)
    if len(faces) == 0:
        raise Error("Hi3D produced an empty isosurface")
    if postprocess:
        _close_and_filter(verts, faces)
    var mesh = NativeMesh()
    var mapping = List[Int](length=len(verts), fill=-1)
    for v in faces:
        if mapping[v] < 0:
            mapping[v] = len(mesh.positions) // 3
            mesh.positions.append(verts[v].x)
            mesh.positions.append(verts[v].y)
            mesh.positions.append(verts[v].z)
        mesh.indices.append(mapping[v])
    if postprocess and len(mesh.indices) > 3000000:
        var reduced = simplify_mesh_qem(mesh, 1000000, True)
        mesh = reduced[0].copy()
    var sum_x = Float64(0)
    var sum_y = Float64(0)
    var sum_z = Float64(0)
    for i in range(0, len(mesh.positions), 3):
        sum_x += Float64(mesh.positions[i])
        sum_y += Float64(mesh.positions[i + 1])
        sum_z += Float64(mesh.positions[i + 2])
    var count = Float64(len(mesh.positions) // 3)
    var mid = Vec3(Float32(sum_x / count), Float32(sum_y / count), Float32(sum_z / count))
    var radius = Float32(0)
    for i in range(0, len(mesh.positions), 3):
        var v = Vec3(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]) - mid
        mesh.positions[i] = v.x
        mesh.positions[i + 1] = v.y
        mesh.positions[i + 2] = v.z
        radius = max(radius, v.length())
    if radius <= 0:
        raise Error("Hi3D surface has zero radius")
    for i in range(len(mesh.positions)):
        mesh.positions[i] /= radius
    # The GLB writer supplies a flat default when a mesh has no normals.
    var normals = List[Vec3](length=len(mesh.positions) // 3, fill=Vec3())
    for f in range(0, len(mesh.indices), 3):
        var a = mesh.indices[f]
        var b = mesh.indices[f + 1]
        var c = mesh.indices[f + 2]
        var p = Vec3(mesh.positions[3 * a], mesh.positions[3 * a + 1], mesh.positions[3 * a + 2])
        var q = Vec3(mesh.positions[3 * b], mesh.positions[3 * b + 1], mesh.positions[3 * b + 2])
        var r = Vec3(mesh.positions[3 * c], mesh.positions[3 * c + 1], mesh.positions[3 * c + 2])
        var normal = (q - p).cross(r - p)
        normals[a] = normals[a] + normal
        normals[b] = normals[b] + normal
        normals[c] = normals[c] + normal
    mesh.normals = List[Float32](capacity=len(mesh.positions))
    for v in normals:
        var vn = v * (1 / max(Float32(1e-30), v.length()))
        mesh.normals.append(vn.x)
        mesh.normals.append(vn.y)
        mesh.normals.append(vn.z)
    return mesh^


def hi3d_model(var mesh: NativeMesh) -> NativeModel:
    var model = NativeModel()
    var node = NativeNode()
    node.name = "Hi3D"
    node.mesh = 0
    node.mesh_indices.append(0)
    model.meshes.append(mesh^)
    model.nodes.append(node^)
    model.roots.append(0)
    return model^
