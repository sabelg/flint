"""Surface support for Hi3D's 512 pass. Mojo port, see licenses/hi3d."""
from std.math import isfinite
from flint.assets.model import NativeModel
from flint.core.math import Vec3
from flint.ml.tensor import Tensor, DT_I32


def _overlap(a: Vec3, b: Vec3, c: Vec3, axis: Vec3) -> Bool:
    var p = a.dot(axis)
    var q = b.dot(axis)
    var r = c.dot(axis)
    var half = Float32(0.5) * (abs(axis.x) + abs(axis.y) + abs(axis.z))
    return min(p, min(q, r)) <= half and max(p, max(q, r)) >= -half


def _edge_axes(a: Vec3, b: Vec3, c: Vec3, e: Vec3) -> Bool:
    return (
        _overlap(a, b, c, Vec3(0, e.z, -e.y))
        and _overlap(a, b, c, Vec3(-e.z, 0, e.x))
        and _overlap(a, b, c, Vec3(e.y, -e.x, 0))
    )


def _mark(mut occupied: List[UInt8], a: Vec3, b: Vec3, c: Vec3, n: Int):
    for z in range(max(0, Int(min(a.z, min(b.z, c.z)))), min(n, 1 + Int(max(a.z, max(b.z, c.z))))):
        for y in range(max(0, Int(min(a.y, min(b.y, c.y)))), min(n, 1 + Int(max(a.y, max(b.y, c.y))))):
            for x in range(max(0, Int(min(a.x, min(b.x, c.x)))), min(n, 1 + Int(max(a.x, max(b.x, c.x))))):
                var idx = (z * n + y) * n + x
                if occupied[idx]:
                    continue
                var center = Vec3(Float32(x) + 0.5, Float32(y) + 0.5, Float32(z) + 0.5)
                var p = a - center
                var q = b - center
                var r = c - center
                var ab = q - p
                var bc = r - q
                var normal = ab.cross(bc)
                if (
                    abs(normal.dot(p)) <= Float32(0.5) * (abs(normal.x) + abs(normal.y) + abs(normal.z))
                    and _edge_axes(p, q, r, ab)
                    and _edge_axes(p, q, r, bc)
                    and _edge_axes(p, q, r, p - r)
                ):
                    occupied[idx] = 1


def hi3d_voxels(vertices: List[Vec3], faces: List[Int], budget: Int = 80000) raises -> Tensor:
    """Return ordered [N,3] voxel coordinates from indexed world-space triangles."""
    if len(vertices) == 0 or len(faces) == 0 or len(faces) % 3 != 0 or budget < 1:
        raise Error("Hi3D needs an indexed triangle mesh and a positive voxel budget")
    var lo = vertices[0]
    var hi = lo
    for v in vertices:
        if not (isfinite(v.x) and isfinite(v.y) and isfinite(v.z)):
            raise Error("Nonfinite mesh position")
        lo = Vec3(min(lo.x, v.x), min(lo.y, v.y), min(lo.z, v.z))
        hi = Vec3(max(hi.x, v.x), max(hi.y, v.y), max(hi.z, v.z))
    for i in faces:
        if i < 0 or i >= len(vertices):
            raise Error("Triangle index out of bounds")
    if hi.x == lo.x and hi.y == lo.y and hi.z == lo.z:
        raise Error("Mesh has zero extent")
    var eps = Vec3(1e-6, 1e-6, 1e-6)
    lo = lo - eps
    hi = hi + eps
    var center = (lo + hi) * 0.5
    var size = hi - lo
    var extent = max(size.x, max(size.y, size.z))
    if not (isfinite(extent) and isfinite(center.x) and isfinite(center.y) and isfinite(center.z)):
        raise Error("Mesh bounds exceed float32 range")
    var factor = Float64(1)
    while Int(64 * factor) >= 7:
        var n = Int(64 * factor)
        var scale = Float32(Float64(n - 6) - 2e-6) / extent
        var mid = Float32(n) * 0.5
        var grid = List[Vec3](capacity=len(vertices))
        for v in vertices:
            grid.append((v - center) * scale + Vec3(mid, mid, mid))
        var occupied = List[UInt8](length=n * n * n, fill=0)
        for i in range(0, len(faces), 3):
            _mark(occupied, grid[faces[i]], grid[faces[i + 1]], grid[faces[i + 2]], n)
        var count = 0
        for flag in occupied:
            count += Int(flag)
        if count == 0:
            raise Error("Empty surface support")
        if count <= budget:
            var xyz = List[Int](capacity=count * 3)
            for i in range(len(occupied)):
                if occupied[i]:
                    xyz.append(i % n)
                    xyz.append((i // n) % n)
                    xyz.append(i // (n * n))
            return Tensor.ints(DT_I32, [count, 3], xyz^)
        factor *= 0.95
    raise Error("Voxel budget too small for this mesh")


def hi3d_model_voxels(mut model: NativeModel) raises -> Tensor:
    """Flatten the static mesh instances in Flint's imported scene at their node transforms."""
    model.update_cached_transforms()
    var vertices = List[Vec3]()
    var faces = List[Int]()
    var seen = List[Bool](length=len(model.nodes), fill=False)
    var stack = model.roots.copy()
    while len(stack) > 0:
        var i = stack.pop()
        if i < 0 or i >= len(seen) or seen[i]:
            continue
        seen[i] = True
        ref node = model.nodes[i]
        for child in node.children:
            stack.append(child)
        for j in range(len(node.mesh_indices) if len(node.mesh_indices) > 0 else Int(node.mesh >= 0)):
            var id = node.mesh_indices[j] if len(node.mesh_indices) > 0 else node.mesh
            if id < 0 or id >= len(model.meshes):
                raise Error("Invalid mesh instance")
            ref mesh = model.meshes[id]
            var base = len(vertices)
            var n = len(mesh.positions) // 3
            if len(mesh.positions) % 3 != 0 or len(mesh.indices) % 3 != 0:
                raise Error("Invalid mesh primitive")
            for k in range(0, len(mesh.positions), 3):
                vertices.append(
                    model.cached_transforms[i].transform_point(
                        Vec3(mesh.positions[k], mesh.positions[k + 1], mesh.positions[k + 2])
                    )
                )
            for k in mesh.indices:
                if k < 0 or k >= n:
                    raise Error("Invalid primitive index")
                faces.append(base + k)
    return hi3d_voxels(vertices, faces)
