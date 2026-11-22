from std.testing import assert_equal, assert_true
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.image import NativeImage
from flint.assets.model import NativeModel
from flint.procedural.reference_fern import FernReference, build_reference_fern


def _sample(wide: Bool, cutout: Bool, leaf_color: UInt32) -> NativeImage:
    var im = NativeImage()
    im.valid = True
    im.width = 32
    im.height = 40
    var bg = UInt32(0) if cutout else UInt32(0xFFF7F5EE)
    im.pixels = List[UInt32](length=1280, fill=bg)
    for y in range(4, 36):
        var width = 2 + min(y - 4, 35 - y) // (2 if wide else 4)
        for x in range(16 - width, 17 + width):
            im.pixels[y * 32 + x] = leaf_color
    return im^


def test_reference_fern() raises:
    var profile = FernReference(_sample(True, True, 0xFF41832F))
    var opaque = FernReference(_sample(True, False, 0xFF41832F))
    var narrow = FernReference(_sample(False, True, 0xFF996633))
    assert_equal(profile.widths, opaque.widths)
    assert_equal(profile.atlas.pixels, opaque.atlas.pixels)
    assert_true(profile.widths[8] > narrow.widths[8])
    assert_true(profile.atlas.pixels[0] != narrow.atlas.pixels[0])
    var expected = [560, 168, 84]
    for lod in range(3):
        var mesh = build_reference_fern(profile, lod, 42)
        var same = build_reference_fern(profile, lod, 42)
        var other = build_reference_fern(narrow, lod, 42)
        var reseed = build_reference_fern(profile, lod, 43)
        assert_equal(mesh.positions, same.positions)
        assert_true(mesh.positions != other.positions)
        assert_true(mesh.positions != reseed.positions)
        assert_equal(mesh.triangle_count(), expected[lod])
        assert_true(mesh.double_sided and mesh.alpha_mode == 0)
        var roots = 0
        var tips = 0
        for i in mesh.indices:
            assert_true(0 <= i and i < mesh.vertex_count())
        for i in range(mesh.vertex_count()):
            var nx = mesh.normals[3 * i]
            var ny = mesh.normals[3 * i + 1]
            var nz = mesh.normals[3 * i + 2]
            assert_true(abs(nx * nx + ny * ny + nz * nz - 1) < 0.001)
            var w = mesh.texcoords1[2 * i]
            assert_true(w >= 0 and w <= 1)
            if mesh.positions[3 * i + 1] == 0:
                roots += 1
                assert_equal(w, Float32(0))
            if w == 1:
                tips += 1
        assert_equal(roots, 14)
        assert_true(tips >= 14)
        var model = NativeModel()
        model.textures.append(profile.atlas.copy())
        model.meshes.append(mesh^)
        var imported = parse_native_glb(encode_native_glb(model))
        assert_true(imported.loaded())
        assert_equal(imported.meshes[0].texcoords1, same.texcoords1)
        assert_equal(imported.meshes[0].triangle_count(), expected[lod])
        assert_equal(imported.textures[0].pixels, profile.atlas.pixels)
    var bad_lod = False
    try:
        _ = build_reference_fern(profile, 3)
    except:
        bad_lod = True
    assert_true(bad_lod)
    var blank = _sample(True, True, 0)
    var rejected = False
    try:
        _ = FernReference(blank)
    except:
        rejected = True
    assert_true(rejected)


def main() raises:
    test_reference_fern()
    print("Reference fern authoring and GLB round trip passed")
