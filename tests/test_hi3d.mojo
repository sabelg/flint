"""Small deterministic regressions for the Hi3D 512 path, without model downloads."""
from std.math import sqrt
from flint.ml.hi3d import hi3d_latents
from flint.ml.hi3d_transformer import Hi3DTransformer
from flint.ml.hi3d_io import hi3d_read_rows
from flint.platform.os import env_now
from std.testing import assert_equal, assert_true
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.ml.hi3d_flow import Hi3DVelocityModel, hi3d_sample, hi3d_times
from flint.ml.hi3d_mesh import hi3d_surface, _close_and_filter
from flint.ml.hi3d_noise import hi3d_noise
from flint.ml.hi3d_sparse import SparseGrid, SparseParents, sparse_conv3
from flint.ml.hi3d_voxels import hi3d_voxels
from flint.ml.tensor import DT_I32, Tensor


struct _Velocity(Hi3DVelocityModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def velocity(mut self, x: Tensor, time: Float32, conditional: Bool) -> Tensor:
        self.calls += 1
        var out = x.copy()
        for i in range(len(out.f)):
            out.f[i] = 1 if conditional else 0
        return out^


def _test_connected_faces() raises:
    # Two fans sharing a rim form a single closed component. Joining faces in
    # interleaved order produces multi-level union-find paths.
    comptime segments = 256
    var vertices = List[Vec3](length=segments + 5, fill=Vec3())
    var faces = List[Int]()
    for i in range(segments):
        var a = 2 + i
        var b = 2 + (i + 1) % segments
        faces.extend([0, a, b, 1, b, a])
    var expected = faces.copy()
    faces.extend([segments + 2, segments + 3, segments + 4])
    _close_and_filter(vertices, faces)
    assert_equal(len(faces), len(expected))
    for i in range(len(faces)):
        assert_equal(faces[i], expected[i])


def _test_checkpoint(root: String) raises:
    # Exactly representable inputs keep this checkpoint check independent
    # of random-number generator and platform math-library differences.
    var x = Tensor.zeros([8, 32])
    var ctx = Tensor.zeros([1, 16, 2048])
    for i in range(len(x.f)):
        x.f[i] = Float32((i * 15 + 7) % 127 - 63) / 64
    for i in range(len(ctx.f)):
        ctx.f[i] = Float32((i * 17 + 3) % 101 - 50) / 2048
    var coords = Tensor.ints(
        DT_I32, [8, 3], [7, 8, 9, 7, 8, 10, 8, 8, 10, 8, 9, 10, 9, 8, 10, 16, 17, 18, 17, 16, 18, 10, 10, 10]
    )
    var net = Hi3DTransformer(root + "/model.safetensors", coords^, ctx^)
    var actual = hi3d_latents(net, x^, root, 1)
    var expected = hi3d_read_rows("tests/fixtures/hi3d/step.f32", 32)
    assert_equal(actual.shape, expected.shape)
    var square_error = Float32(0)
    var a2 = Float32(0)
    var b2 = Float32(0)
    var dot = Float32(0)
    for i in range(len(actual.f)):
        var a = actual.f[i]
        var b = expected.f[i]
        square_error += (a - b) * (a - b)
        a2 += a * a
        b2 += b * b
        dot += a * b
    var rmse = sqrt(square_error / Float32(len(actual.f)))
    var cosine = dot / sqrt(a2 * b2)
    assert_true(rmse < 0.03)
    assert_true(cosine > 0.999)
    print("  Hi3D 512 checkpoint Heun step vs. reference, RMSE", rmse, "cosine", cosine)


def test_hi3d() raises:
    _test_connected_faces()
    var noise = hi3d_noise(1, 0)
    assert_true(abs(noise.f[0] + 1.12583983) < 1e-6)
    assert_true(abs(noise.f[8] - 0.32227492) < 1e-6)
    var net = _Velocity()
    var x = hi3d_sample(net, Tensor.zeros([1, 32]), 1)
    assert_equal(net.calls, 3)
    assert_true(abs(x.f[0] + 2.595) < 1e-6)
    var times = hi3d_times(24, 25, 9180)
    assert_true(abs(times[1] - 0.002) < 1e-7)
    var pts: List[Vec3] = [Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(-1, 1, 0)]
    var surface = hi3d_voxels(pts, [0, 1, 2])
    assert_true(surface.shape[0] > 1000)
    for i in range(surface.shape[0]):
        assert_equal(surface.i[3 * i + 2], 32)
    var g = SparseGrid(Tensor.ints(DT_I32, [2, 3], [0, 0, 0, 1, 0, 0]), 64)
    var parents = SparseParents(g)
    var feat = Tensor.floats([2, 1], [3, 6])
    var parent = parents.down(feat)
    assert_equal(parent.f[0], Float32(3))
    var up = parents.up(parent)
    assert_equal(up.f[1], Float32(3))
    var kernel = Tensor.zeros([1, 3, 3, 3, 1])
    kernel.f[22] = 2
    var pool = JobPool()
    var conv = sparse_conv3(pool, g, feat, kernel, Tensor.floats([1], [1]))
    assert_equal(conv.f[0], Float32(13))
    assert_equal(conv.f[1], Float32(1))

    # A single enclosed negative grid vertex yields a cube of eight dual vertices.
    var coords = List[Int]()
    var attr = Tensor.zeros([8, 40])
    for i in range(8):
        coords.append(i & 1)
        coords.append((i >> 1) & 1)
        coords.append((i >> 2) & 1)
        for k in range(8):
            attr.f[40 * i + k] = -5 if (i ^ k) == 7 else 5
            attr.f[40 * i + 8 + k] = 1
    var grid = SparseGrid(Tensor.ints(DT_I32, [8, 3], coords^), 512)
    var mesh = hi3d_surface(grid, attr)
    assert_equal(len(mesh.positions), 24)
    assert_equal(len(mesh.indices), 36)
    assert_equal(len(mesh.normals), 24)
    for i in range(0, 24, 3):
        var p = Vec3(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2])
        var normal = Vec3(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2])
        assert_true(p.dot(normal) > 0.8)
    print("  Hi3D seeded noise, Heun flow, sparse support, convolution and mesh export: ok")
    var root = env_now("FLINT_HI3D_TRANSFORMER")
    if root.byte_length() > 0:
        _test_checkpoint(root)
