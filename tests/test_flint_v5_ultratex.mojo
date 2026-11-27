"""Native UltraTex regressions with an independent FLUX.2 velocity fixture."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from flint.assets.model import NativeMesh
from flint.assets.image import decode_png
from flint.core.jobs import JobPool
from flint.ml.hi3d_io import hi3d_read_rows
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.ultratex import ultratex_noise, ultratex_time, ultratex_views
from flint.ml.ultratex_attention import block_attention
from flint.ml.ultratex_gpu import UltraTexGpu, _project_tiled
from flint.ml.ultratex_gpu_blocks import DeviceRows
from flint.ml.ultratex_gpu_attention import gpu_sparse_attention
from max.gpu.host import DeviceContext
from flint.ml.ultratex_dit import UltraTexDiT
from flint.ml.ultratex_mesh import UltraTexProjector
from flint.ml.ultratex_vae import UltraTexVAE, ultratex_encoder_path
from flint.ml.vision_ops import weighted_linear
from flint.platform.os import env_now, file_size, read_bytes


def _test_attention(mut pool: JobPool) raises:
    var q = Tensor.zeros([11, 2])
    var k = Tensor.zeros([11, 2])
    var v = Tensor.zeros([11, 2])
    for i in range(11):
        q.f[i * 2] = Float32(i % 7 - 3)
        q.f[i * 2 + 1] = 1 + Float32(i) * 0.125
        k.f[i * 2] = Float32((5 * i + 1) % 13 - 6)
        k.f[i * 2 + 1] = Float32(i % 3) * 0.75
        v.f[i * 2] = Float32(i) * 0.5
        v.f[i * 2 + 1] = Float32((i * i) % 13)
    var expected: List[Float32] = [
        2.520653,
        11.9982201,
        2.579369,
        11.9871692,
        2.7616842,
        11.9003311,
        2.9737431,
        10.1141012,
        2.9049185,
        6.3314575,
        3.5386146,
        7.1717335,
        4.0692772,
        7.8365507,
        0.0000468,
        0.0000758,
        2.5819892,
        11.994901,
        2.7803821,
        11.9566618,
        3.0754211,
        10.6398393,
    ]
    var actual = block_attention(pool, q, k, v, 0.5, 4, 3)
    for i in range(len(expected)):
        assert_true(abs(expected[i] - actual.f[i]) < 0.00002)


def _test_velocity(bf16: Bool) raises:
    var p = "tests/fixtures/ultratex/small-"
    var gpu = env_now("FLINT_ULTRATEX_GPU") == "1"
    var base = (
        p
        + "bf16.safetensors" if bf16 else "tests/fixtures/ultratex/small.safetensors"
    )
    var model = UltraTexDiT(
        base, p + "adapter.safetensors", 32, 2, 1, 1, 48, 4, 1, gpu
    )
    var x = hi3d_read_rows(p + "image.f32", 128)
    var ctx = hi3d_read_rows(p + "ctx.f32", 48)
    if bf16:
        for i in range(len(x.f)):
            x.f[i] = Float32(BFloat16(x.f[i]))
        for i in range(len(ctx.f)):
            ctx.f[i] = Float32(BFloat16(ctx.f[i]))
    var actual = model.velocity(x, hi3d_read_rows(p + "ids.f32", 4), ctx, 0.625)
    if bf16:
        model.load_adapter(p + "adapter.safetensors")
        var again = model.velocity(
            x, hi3d_read_rows(p + "ids.f32", 4), ctx, 0.625
        )
        assert_equal(again.f, actual.f)
    var expected = hi3d_read_rows(
        p + ("bf16-velocity.f32" if bf16 else "velocity.f32"), 128
    )
    assert_equal(actual.shape, expected.shape)
    var largest = Float32(0)
    var rms = Float32(0)
    for i in range(len(expected.f)):
        var delta = actual.f[i] - expected.f[i]
        largest = max(largest, abs(delta))
        rms += delta * delta
    assert_true(largest < Float32(0.025 if (bf16 or gpu) else 0.00002))
    assert_true(rms / Float32(len(expected.f)) < 0.00004)


def _test_projection() raises:
    var mesh = NativeMesh()
    mesh.positions = [-1, -1, 0, 1, -1, 0, 0, 1, 0]
    mesh.normals = [0, 0, 1, 0, 0, 1, 0, 0, 1]
    mesh.indices = [0, 1, 2]
    var projector = UltraTexProjector(mesh, 32)
    assert_equal(projector.normals.width, 96)
    assert_equal(projector.normals.height, 64)
    var pixel = projector.normals.pixels[16 * 96 + 16]
    assert_equal(pixel, UInt32(0xFF800080))
    var mapped = projector.mesh_with_uvs()
    assert_equal(mapped.positions, mesh.positions)
    assert_equal(mapped.indices, mesh.indices)
    for uv in mapped.texcoords:
        assert_true(uv >= 0 and uv <= 1)
    var tex = projector.blend(projector.normals)
    assert_equal(tex.pixels[16 * 96 + 16], pixel)


def _test_views(mut vae: UltraTexVAE, original: Bool) raises:
    var p = "tests/fixtures/ultratex/"
    var model = UltraTexDiT(
        p + "small.safetensors",
        p + "small-adapter.safetensors",
        32,
        2,
        1,
        1,
        48,
        4,
        1,
    )
    var normals = decode_png(read_bytes(p + "test-normals.png"))
    var ref_image = decode_png(read_bytes(p + "test-reference.png"))
    var noise = Tensor.zeros([24, 128])
    for i in range(len(noise.f)):
        noise.f[i] = Float32((i * 19) % 251 - 125) / 64
    var actual = ultratex_views(
        vae,
        model,
        normals,
        ref_image,
        hi3d_read_rows(p + "small-ctx.f32", 48),
        Tensor(),
        noise,
        2,
    )
    var expected = decode_png(
        read_bytes(
            p
            + (
                "test-original-vae-result.png" if original else "test-reference-result.png"
            )
        )
    )
    assert_true(expected.valid)
    assert_equal(actual.width, expected.width)
    assert_equal(actual.height, expected.height)
    var total = 0
    for i in range(len(expected.pixels)):
        comptime for channel in range(3):
            var a = Int((actual.pixels[i] >> UInt32(8 * channel)) & 255)
            var b = Int((expected.pixels[i] >> UInt32(8 * channel)) & 255)
            assert_true(abs(a - b) <= 1)
            total += abs(a - b)
    assert_true(total < 24)


def _test_released(root: String, mut pool: JobPool) raises:
    var vae = UltraTexVAE(
        ultratex_encoder_path(root), root + "/decoder.safetensors"
    )
    var _, precision = vae.base.storage(
        "encoder.conv_in.weight", [128, 3, 3, 3]
    )
    var original = precision == "F32"
    _test_views(vae, original)
    var img = Tensor.zeros([32, 48, 3])
    for i in range(len(img.f)):
        img.f[i] = Float32((7 * i) % 101 - 50) / 64
    var z = vae.encode(img)
    var rgb = vae.decode(z)
    assert_equal(z.shape, [2, 3, 128])
    assert_equal(rgb.shape, [32, 48, 3])
    var ez: List[Float32] = [-1.7710735, -1.2243279, -1.2306994, -0.17921697]
    var er: List[Float32] = [-0.15476736, -0.34110415, 0.20182061, -0.4828118]
    if original:
        ez = [-1.7698828, -1.2213943, -1.2251633, -0.17827575]
        er = [-0.1507616, -0.3362404, 0.20364106, -0.48136008]
    var vae_tol = Float32(0.0001)
    for j in range(4):
        var zi = [0, len(z.f) // 3, len(z.f) // 2, len(z.f) - 1]
        var ri = [0, len(rgb.f) // 3, len(rgb.f) // 2, len(rgb.f) - 1]
        assert_true(abs(z.f[zi[j]] - ez[j]) < vae_tol)
        assert_true(abs(rgb.f[ri[j]] - er[j]) < vae_tol)
    var x = Tensor.zeros([3, 4096])
    for i in range(len(x.f)):
        x.f[i] = Float32((i * 13) % 119 - 59) / 64
    var ea: List[Float32] = [
        -0.006142786,
        0.10061257,
        0.02972950,
        -0.04243604,
        -0.00907568,
        -0.05360615,
        0.04263556,
        0.03706593,
        0.03597111,
        0.00661627,
    ]
    for task in range(2):
        var sub = "flux2" if task == 0 else "flux2_mr"
        var weights = SafeTensorWeights(
            root + "/" + sub + "/lora/dit_lora.safetensors", 1 << 30
        )
        var prefix = "base_model.model.double_blocks.0.img_attn.qkv.lora_"
        var a = weighted_linear(
            weights, pool, x, prefix + "A.default", 64, False
        )
        var delta = weighted_linear(
            weights, pool, a, prefix + "B.default", 12288, False
        )
        var positions = [0, 31, 12001, 12288, 36863]
        for j in range(5):
            assert_true(abs(delta.f[positions[j]] - ea[task * 5 + j]) < 0.00002)


def _test_released_gpu_merge(root: String) raises:
    var path = root + "/../flux-2-klein-base-9b.safetensors"
    if file_size(path) <= 0:
        return
    var gpu = UltraTexGpu()
    if gpu.ctx.target.is_nvidia_gpu():
        var base = SafeTensorWeights(path, 32 << 30, private_mapping=True)
        var adapter = SafeTensorWeights(
            root + "/flux2/lora/dit_lora.safetensors", 2 << 30
        )
        var key = "double_blocks.0.img_attn.qkv"
        var id = gpu.adapted_weight_id(base, adapter, key, 12288, 4096, 64)
        assert_equal(
            gpu.adapted_weight_id(base, adapter, key, 12288, 4096, 64), id
        )
        var sample = gpu.weights[id].create_sub_buffer[DType.bfloat16](0, 1024)
        # BF16 bit patterns from the official CUDA high-precision PEFT merge.
        # These coordinates distinguish TF32 rank products from IEEE FP32.
        var pos = [161, 186, 289, 387, 519, 545, 565, 592, 696, 917]
        var expected = [
            48221,
            47813,
            47168,
            14448,
            14688,
            14950,
            47990,
            15443,
            15318,
            48035,
        ]
        with sample.map_to_host() as host:
            for i in range(len(pos)):
                assert_equal(
                    Int(bitcast[DType.uint16](host[pos[i]])), expected[i]
                )


def _test_seed() raises:
    # Expected BF16 values from PyTorch CUDA seed 42 on a 48-SM GB10.
    var noise = ultratex_noise(512, 42, 48 * 1536)
    var at = [0, 1, 6, 3901, 24576, 49151, 393216, 786431]
    var expected: List[Float32] = [
        0.1943359375,
        2.15625,
        -0.6484375,
        0.28515625,
        -0.375,
        -0.69921875,
        -1.171875,
        0.37109375,
    ]
    for i in range(len(at)):
        assert_equal(noise.f[at[i]], expected[i])


def _test_device_blocks(ctx: DeviceContext) raises:
    var x = Tensor.zeros([33, 128])
    var m = Tensor.zeros([1, 384])
    for i in range(len(x.f)):
        x.f[i] = Float32(i * 37 % 121 - 60) / 32
    for i in range(len(m.f)):
        m.f[i] = Float32(i * 11 % 101 - 50) / 64
    var a = DeviceRows.upload(ctx, x)
    var mods = DeviceRows.upload(ctx, m)
    var h = a.norm(ctx, mods, 0)
    var glu = h.glu(ctx, 0).host()
    var pos = [0, 15, 31, 64, 511, 819, 1711, 2111]
    var expected: List[Float32] = [
        -0.10595703125,
        -0.400390625,
        0.1279296875,
        -0.302734375,
        -0.49609375,
        -0.2734375,
        0.15234375,
        -0.470703125,
    ]
    for i in range(len(pos)):
        assert_equal(glu.f[pos[i]], expected[i])
    a.residual(ctx, h, mods, 2)
    var result = a.host()
    pos = [0, 7, 33, 128, 379, 911, 2200, 4223]
    expected = [
        -4.125,
        -1.625,
        -1.4921875,
        -3.09375,
        1.1953125,
        0.296875,
        0.875,
        -0.71875,
    ]
    for i in range(len(pos)):
        assert_equal(result.f[pos[i]], expected[i])


def _test_gpu_projection(ctx: DeviceContext) raises:
    var m = 37
    var n = 96
    var k = 64
    var a = ctx.enqueue_create_buffer[DType.bfloat16](m * k)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    with a.map_to_host() as host:
        for i in range(m * k):
            host[i] = BFloat16(i % 19 - 9)
    with b.map_to_host() as host:
        for i in range(n * k):
            host[i] = BFloat16(i % 23 - 11)
    ctx.enqueue_function[_project_tiled[DType.float32]](
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        c.unsafe_ptr(),
        Int32(m),
        Int32(k),
        Int32(n),
        grid_dim=(2, 1),
        block_dim=128,
    )
    with c.map_to_host() as host:
        for r in range(m):
            for col in range(n):
                var expected = 0
                for j in range(k):
                    expected += ((r * k + j) % 19 - 9) * (
                        (col * k + j) % 23 - 11
                    )
                assert_equal(host[r * n + col], Float32(expected))


def _test_gpu_sparse() raises:
    var x = Tensor.zeros([937, 3 * 2 * 128])
    for i in range(len(x.f)):
        var v = UInt32(i)
        v = (v ^ (v >> 16)) * 0x45D9F3B
        v = (v ^ (v >> 16)) * 0x45D9F3B
        x.f[i] = Float32(Int((v ^ (v >> 16)) % 127) - 63) / 32
    var ctx = DeviceContext()
    if ctx.target.is_nvidia_gpu():
        _test_device_blocks(ctx)
        _test_gpu_projection(ctx)
        var result = gpu_sparse_attention(ctx, x, 2, 3)
        # Official Triton SLA, including an incomplete final query and key block.
        var at = [
            0,
            6,
            127,
            128,
            4095,
            32768,
            32769,
            64000,
            65535,
            131072,
            193001,
            239871,
        ]
        var expected: List[Float32] = [
            0.1455078125,
            -0.18359375,
            0.23828125,
            -0.283203125,
            0.0177001953125,
            0.27734375,
            0.2099609375,
            0.1279296875,
            0.16796875,
            0.134765625,
            0.08740234375,
            -0.059814453125,
        ]
        for i in range(len(at)):
            assert_true(abs(result.f[at[i]] - expected[i]) < 0.0001)


def main() raises:
    var pool = JobPool()
    _ = pool.start(2)
    _test_seed()
    if env_now("FLINT_ULTRATEX_GPU") == "1":
        _test_gpu_sparse()
    _test_attention(pool)
    _test_velocity(False)
    _test_velocity(True)
    _test_projection()
    assert_equal(ultratex_time(0, 25, 128), Float32(1))
    assert_equal(ultratex_time(25, 25, 128), Float32(0))
    assert_true(ultratex_time(12, 25, 128) > ultratex_time(13, 25, 128))
    var root = env_now("FLINT_ULTRATEX_WEIGHTS")
    if root.byte_length() > 0:
        _test_released(root, pool)
        if env_now("FLINT_ULTRATEX_GPU") == "1":
            _test_released_gpu_merge(root)
    print("UltraTex regression checks passed")
