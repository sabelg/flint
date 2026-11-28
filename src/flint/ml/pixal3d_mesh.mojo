"""Extract Pixal3D's flexible dual grid and its per-surface PBR predictions."""
from std.math import exp, log, pow
from std.utils import StaticTuple
from flint.assets.model import NativeMesh, NativeModel, NativeTexture
from flint.core.math import Vec3
from flint.ml.hi3d_mesh import hi3d_model
from flint.ml.pixal3d_decoder import _grid
from flint.ml.tensor import Tensor


def _unit(x: Float32) -> Float32:
    return min(Float32(1), max(Float32(0), x * 0.5 + 0.5))


def pixal_surface(coords: Tensor, field: Tensor, materials: Tensor, resolution: Int) raises -> NativeModel:
    var grid = _grid(coords, resolution)
    var n = grid.xyz.shape[0]
    var painted = materials.rank() != 0
    if (
        field.shape != [n, 7]
        or not field.is_float()
        or not field.has_valid_storage()
        or (
            painted
            and (materials.shape != [n, 6] or not materials.is_float() or not materials.has_valid_storage())
        )
    ):
        raise Error("Incompatible Pixal3D dual-grid attributes")
    var mesh = NativeMesh()
    mesh.double_sided = True
    var points = List[Vec3](capacity=n)
    var splits = List[Float32](capacity=n)
    var normals = List[Vec3](length=n, fill=Vec3())
    for r in range(n):
        var p = StaticTuple[Float32, 3](fill=0)
        for j in range(3):
            var offset = 2 / (1 + exp(-field.fget(r * 7 + j))) - 0.5
            p[j] = (Float32(grid.xyz.i[r * 3 + j]) + offset) / Float32(resolution) - 0.5
        # Compose o-voxel’s glTF axis conversion with Pixal3D’s export rotation.
        points.append(Vec3(-p[0], p[1], -p[2]))
        var split = field.fget(r * 7 + 6)
        splits.append(split if split > 20 else log(1 + exp(split)))
    var offsets: List[Int] = [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
    ]
    var split_a: List[Int] = [0, 1, 2, 0, 2, 3]
    var split_b: List[Int] = [0, 1, 3, 3, 1, 2]
    for r in range(n):
        for axis in range(3):
            if field.fget(7 * r + 3 + axis) <= 0:
                continue
            var quad = StaticTuple[Int, 4](fill=-1)
            var complete = True
            for c in range(4):
                var o = 12 * axis + 3 * c
                quad[c] = grid.get(
                    grid.xyz.i[r * 3] + offsets[o],
                    grid.xyz.i[r * 3 + 1] + offsets[o + 1],
                    grid.xyz.i[r * 3 + 2] + offsets[o + 2],
                )
                complete = complete and quad[c] >= 0
            if complete:
                var first = splits[quad[0]] * splits[quad[2]] > splits[quad[1]] * splits[quad[3]]
                for j in range(6):
                    mesh.indices.append(quad[split_a[j] if first else split_b[j]])
    if not len(mesh.indices):
        raise Error("Pixal3D produced an empty surface")
    for f in range(0, len(mesh.indices), 3):
        var a = mesh.indices[f]
        var b = mesh.indices[f + 1]
        var c = mesh.indices[f + 2]
        var normal = (points[b] - points[a]).cross(points[c] - points[a])
        for j in range(3):
            var idx = mesh.indices[f + j]
            normals[idx] = normals[idx] + normal
    for r in range(n):
        var p = points[r]
        mesh.positions.extend([p.x, p.y, p.z])
        # The learned field can contain isolated sites with no incident face.
        var length = normals[r].length()
        var norm = normals[r] * (1 / length) if length > 1e-20 else Vec3(0, 1, 0)
        mesh.normals.extend([norm.x, norm.y, norm.z])
        if painted:
            for ch in range(3):
                var s = _unit(materials.fget(r * 6 + ch))
                var linear = s / 12.92 if s <= 0.04045 else pow((s + 0.055) / 1.055, Float32(2.4))
                mesh.colors.append(UInt8(linear * 255 + 0.5))
            mesh.colors.append(UInt8(_unit(materials.fget(r * 6 + 5)) * 255 + 0.5))
            mesh.texcoords.extend(
                [
                    (_unit(materials.fget(r * 6 + 3)) * 255 + 0.5) / 256,
                    (_unit(materials.fget(r * 6 + 4)) * 255 + 0.5) / 256,
                ]
            )
    var model = hi3d_model(mesh^)
    model.nodes[0].name = "Pixal3D"
    if painted:
        # A linear two-channel lookup preserves independent per-vertex metal
        # and roughness gradients using glTF's standard material texture.
        var mr = NativeTexture()
        mr.width = 256
        mr.height = 256
        for roughness in range(256):
            for metallic in range(256):
                mr.pixels.append(0xFFFF0000 | (UInt32(roughness) << 8) | UInt32(metallic))
        model.textures.append(mr^)
        model.meshes[0].orm_texture = 0
        model.meshes[0].metallic = 1
        model.meshes[0].roughness = 1
    return model^
