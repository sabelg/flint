"""Native stage tests, including a full-resolution reference when FLINT_TRELLIS2_DECODER is set."""

from flint.assets.image import NativeImage, decode_png
from flint.ml.trellis2_image import prepare_trellis_image
from flint.ml.trellis2_dino import TrellisImageEncoder
from std.memory import bitcast
from flint.ml.safetensors import SafeTensorWeights
from std.testing import assert_equal, assert_true
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.tensor import Tensor
from flint.ml.trellis2 import TrellisStructureDecoder, read_structure_latent, structure_coordinates
from flint.ml.trellis2_flow import StructureVelocityModel, sample_structure_flow, structure_time
from flint.platform.os import env_now, read_bytes


struct FixtureVelocity(StructureVelocityModel):
    var positives: Int
    var negatives: Int

    def __init__(out self):
        self.positives = 0
        self.negatives = 0

    def predict(mut self, latent: Tensor, model_time: Float32, conditional: Bool) raises -> Tensor:
        self.positives += Int(conditional)
        self.negatives += Int(not conditional)
        var t = model_time / 1000
        var out = Tensor.zeros(latent.shape.copy())
        for i in range(out.numel()):
            if conditional:
                out.f[i] = latent.fget(i) * 0.15 + Float32(i) * 0.02 + t * 0.1
            else:
                out.f[i] = latent.fget(i) * 0.07 - Float32(i) * 0.03 - t * 0.04
        return out^


def test_structure_flow() raises:
    var model = FixtureVelocity()
    var noise = Tensor.zeros([8, 1, 1, 1])
    for i in range(8):
        noise.f[i] = Float32(i - 3) / 7
    var sample = sample_structure_flow(model, noise^)
    var expected: List[Float32] = [-0.38439396, -0.37475699, -0.36512002, -0.35548303,
                                    -0.34584609, -0.33620912, -0.32657218, -0.31693515]
    for i in range(8):
        assert_true(abs(sample.f[i] - expected[i]) < Float32(2e-6))
    assert_equal(model.positives, 12)
    assert_equal(model.negatives, 10)
    assert_equal(structure_time(0), Float64(1))
    assert_equal(structure_time(12), Float64(0))


def test_occupancy_pooling() raises:
    var x = Tensor.zeros([1, 4, 4, 4])
    x.f[0] = -1
    x.f[3] = 1
    x.f[7] = 2
    x.f[48] = 1
    var coords = structure_coordinates(x, 2)
    assert_equal(coords.shape, [2, 4])
    assert_equal(coords.i, [0, 0, 0, 1, 0, 1, 0, 0])
    var invalid = False
    try:
        _ = structure_coordinates(x, 3)
    except:
        invalid = True
    assert_true(invalid)


def test_checkpoint(path: String) raises:
    var dir = "tests/fixtures/trellis2/"
    var decoder = TrellisStructureDecoder(path)
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var latent = read_structure_latent(dir + "open_cube_latent.f32", 16)
    var out = decoder.decode(pool, latent)
    var expected = read_bytes(dir + "open_cube_occupancy.bits")
    assert_equal(out.shape, [1, 64, 64, 64])
    assert_equal(len(expected) * 8, out.numel())
    for i in range(out.numel()):
        var occupied = (Int(expected[i // 8]) >> (i % 8)) & 1
        assert_equal(Int(out.f[i] > 0), occupied)
    var indices = [0, 2345, 66640, 87381, 132177, 262143]
    var values: List[Float32] = [-83.16177, -138.68279, 81.71911, -165.85542, -79.52173, -84.27283]
    for i in range(len(indices)):
        assert_true(abs(out.f[indices[i]] - values[i]) < Float32(0.2))


def test_image_preparation(w: Int, h: Int, target: Int, expected: UInt64) raises:
    var image = NativeImage()
    image.valid = True
    image.width = w
    image.height = h
    for y in range(h):
        for x in range(w):
            var alpha = 180 if x % 5 == 0 and y % 7 == 0 else 0
            if x >= w // 3 and x < w - 1 and y < h // 2:
                alpha = 255
            image.pixels.append((UInt32(alpha) << 24) | (UInt32((x * 17 + y) % 256) << 16)
                                | (UInt32((y * 23 + x) % 256) << 8) | UInt32((x + 3 * y) % 256))
    var pixels = prepare_trellis_image(image, target)
    assert_equal(pixels.shape, [1, 3, target, target])
    var digest = UInt64(14695981039346656037)
    var mean: List[Float32] = [0.485, 0.456, 0.406]
    var std: List[Float32] = [0.229, 0.224, 0.225]
    for i in range(target * target):
        for ch in range(3):
            var rgb = (pixels.f[ch * target * target + i] * std[ch] + mean[ch]) * 255
            digest = (digest ^ UInt64(Int(rgb + 0.5))) * UInt64(1099511628211)
    assert_equal(digest, expected)


def test_image_errors() raises:
    var image = NativeImage()
    image.valid = True
    image.width = 4
    image.height = 4
    for alpha in [UInt32(255), UInt32(0)]:
        image.pixels = List[UInt32](length=16, fill=alpha << 24)
        var rejected = False
        try:
            _ = prepare_trellis_image(image)
        except:
            rejected = True
        assert_true(rejected)


def test_dino_checkpoint(path: String) raises:
    var image = decode_png(read_bytes("tests/fixtures/trellis2/encoder_cutout.png"))
    var encoder = TrellisImageEncoder(path)
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var features = encoder.encode(pool, image, 512)
    assert_equal(features.shape, [1, 1029, 1024])
    # CLS, two registers and four spatial positions from the independent Transformers run.
    var positions = [0, 2, 4, 5, 37, 514, 1028]
    var channels = [0, 37, 511, 1023]
    var expected: List[Float32] = [
        0.1121047, -0.5070174, -0.1958065, -0.4351617,
        0.2725074, 0.0272698, 0.0501591, 0.0363135,
        0.7986072, -0.0990647, 0.6732807, 0.9687344,
        0.0110417, -0.0533386, 0.8113737, 0.9487719,
        0.0285825, 0.0330601, 0.3923555, 0.8969824,
        0.0836478, -0.0715002, 0.6590893, 0.6575747,
        -0.0611563, -0.1846425, 0.8906967, 0.5530831,
    ]
    for r in range(len(positions)):
        for c in range(len(channels)):
            assert_true(abs(features.f[positions[r] * 1024 + channels[c]] - expected[r * 4 + c]) < Float32(0.001))


def test_bfloat16_weights() raises:
    var archive = SafeTensorWeights("tests/fixtures/trellis2/bfloat16.safetensors", 1024)
    var values = archive.tensor("sample", [8])
    var bits: List[UInt32] = [0, 0x80000000, 0x3f800000, 0xc0200000,
                               0x00010000, 0x7f800000, 0xff800000, 0x7fc10000]
    for i in range(len(bits)):
        assert_equal(bitcast[DType.uint32](values.f[i]), bits[i])


def test_trellis2() raises:
    test_bfloat16_weights()
    test_image_errors()
    test_image_preparation(11, 9, 512, 2859442258794205788)
    test_image_preparation(1100, 832, 1024, 5067091118613662997)
    print("PASS reference RGB pixels at both DINOv3 resolutions")
    test_structure_flow()
    test_occupancy_pooling()
    print("PASS structure sampler and sparse coordinates")
    var encoder_path = env_now("FLINT_TRELLIS2_DINO")
    if encoder_path:
        test_dino_checkpoint(encoder_path)
        print("PASS real DINOv3 encoder checkpoint and reference features")
    else:
        print("SKIP DINOv3 checkpoint (set FLINT_TRELLIS2_DINO)")
    var path = env_now("FLINT_TRELLIS2_DECODER")
    if path:
        test_checkpoint(path)
        print("PASS real checkpoint, 262144 reference occupancy decisions and signed logits")
    else:
        print("SKIP real checkpoint (set FLINT_TRELLIS2_DECODER)")


def main() raises:
    test_trellis2()
