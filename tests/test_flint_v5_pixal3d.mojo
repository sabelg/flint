"""Regression cases for the Pixal3D camera, cascade, shape head and material packing."""
from std.testing import assert_equal, assert_true
from flint.ml.pixal3d import PixalCamera, project_patch_features
from flint.ml.pixal3d_decoder import cascade_coords, denormalize_slat
from flint.ml.pixal3d_mesh import pixal_surface
from flint.ml.hi3d_noise import TorchNoiseStream, hi3d_noise
from flint.ml.tensor import DT_I32, Tensor
from flint.ml.trellis2_flow import sample_pixal_slat
from test_trellis2 import FixtureVelocity


def main() raises:
    # Two channels on a 2x2 patch grid; large prefix values expose accidental CLS sampling.
    var tokens = Tensor.floats(
        [1, 9, 2],
        [
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
            0.0,
            10.0,
            1.0,
            11.0,
            2.0,
            12.0,
            3.0,
            13.0,
        ],
    )
    var camera = PixalCamera.from_fov(0.85)
    var sites = Tensor.ints(DT_I32, [3, 4], [0, 1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 2])
    var dense = project_patch_features(tokens, 32, 3, camera)
    var sparse = project_patch_features(tokens, 32, 3, camera, sites)
    var expected: List[Float32] = [1.59375, 11.59375, 0, 10, 3, 13]
    var indices: List[Int] = [13, 8, 20]
    for i in range(6):
        assert_true(abs(sparse.f[i] - expected[i]) < 1e-5)
        assert_equal(sparse.f[i], dense.f[indices[i // 2] * 2 + i % 2])
    var xyz = Tensor.ints(DT_I32, [3, 4], [0, 511, 511, 511, 0, 0, 0, 0, 0, 1, 1, 1])
    var cascade = cascade_coords(xyz, 1536, 2)
    assert_equal(cascade[1], 64)
    var expected_coords: List[Int] = [0, 0, 0, 0, 0, 63, 63, 63]
    assert_true(cascade[0].i == expected_coords)
    var cube = Tensor.ints(DT_I32, [5, 4], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 7, 7, 7])
    var field = Tensor.zeros([5, 7])
    field.f[3] = 1
    var material = Tensor.zeros([5, 6])
    var model = pixal_surface(cube, field, material, 16)
    assert_equal(len(model.meshes[0].indices), 6)
    assert_equal(model.meshes[0].normals[13], Float32(1))
    assert_equal(model.meshes[0].orm_texture, 0)
    assert_equal(model.textures[0].pixels[123 * 256 + 87], UInt32(0xFFFF7B57))
    var shape = denormalize_slat(Tensor.zeros([1, 32]), False)
    assert_true(abs(shape.f[0] - 0.781296) < 1e-6)
    var rng = TorchNoiseStream(42)
    var a = rng.rows(1)
    var b = rng.rows(1)
    var c = hi3d_noise(2, 42)
    for i in range(32):
        assert_equal(a.f[i], c.f[i])
        assert_equal(b.f[i], c.f[i + 32])
    var probe = Tensor.zeros([1, 32])
    for i in range(32):
        probe.f[i] = Float32(i) * 2 / 31 - 1
    var shape_model = FixtureVelocity()
    var material_model = FixtureVelocity()
    var sampled = sample_pixal_slat(shape_model, probe.copy(), 12, False)
    var painted = sample_pixal_slat(material_model, probe^, 12, True)
    assert_equal(shape_model.negatives, 9)
    assert_equal(material_model.negatives, 0)
    assert_true(abs(sampled.f[19] + 1.53169243) < 2e-5)
    assert_true(abs(painted.f[31] - 0.22764712) < 2e-5)
    print("Pixal3D stage regressions passed")
