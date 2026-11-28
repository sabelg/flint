"""FLUX.2 VAE for UltraTex. Native CPU inference, HWC feature maps."""
from std.math import exp, sqrt
from flint.ml.ultratex_gpu_vae import UltraTexVaeGpu
from flint.platform.os import env_now, file_size
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.ops import matmul_2d
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor, DT_F32


def ultratex_encoder_path(weights_dir: String) -> String:
    """Locate the official autoencoder or a previously packed encoder."""
    for path in [
        weights_dir + "/../ae.safetensors",
        weights_dir + "/ae.safetensors",
    ]:
        if file_size(path) > 0:
            return path.copy()
    return weights_dir + "/encoder.safetensors"


def _conv(
    w: SafeTensorWeights,
    mut pool: JobPool,
    mut gpu: List[UltraTexVaeGpu],
    x: Tensor,
    key: String,
    co: Int,
    k: Int,
    stride: Int = 1,
) raises -> Tensor:
    if len(gpu):
        return gpu[0].conv(w, x, key, co, k, stride)
    var h = x.shape[0]
    var width = x.shape[1]
    var ci = x.shape[2]
    var oh = h // stride
    var ow = width // stride
    var inner = ci * k * k
    var kernel = w.tensor(key + ".weight", [co, ci, k, k])
    var bias = w.tensor(key + ".bias", [co])
    var output = Tensor.zeros([oh, ow, co])
    var batch = min(128, oh * ow)
    var patches = Tensor.zeros([batch, inner])
    var pad = k // 2 if stride == 1 else 0
    for first in range(0, oh * ow, batch):
        var n = min(batch, oh * ow - first)
        for row in range(n):
            var y = (row + first) // ow * stride - pad
            var xx = (row + first) % ow * stride - pad
            for dy in range(k):
                for dx in range(k):
                    var yy = y + dy
                    var col = xx + dx
                    for c in range(ci):
                        patches.f[row * inner + (c * k + dy) * k + dx] = (
                            x.f[(yy * width + col) * ci + c] if yy >= 0
                            and yy < h
                            and col >= 0
                            and col < width else 0
                        )
        matmul_2d(
            pool,
            patches.fptr(),
            kernel.fptr(),
            output.fptr() + first * co * 4,
            bias.fptr(),
            n,
            inner,
            co,
            True,
        )
    _ = patches.fptr()
    _ = kernel.fptr()
    _ = bias.fptr()
    return output^


def _group(
    w: SafeTensorWeights,
    gpu: List[UltraTexVaeGpu],
    var x: Tensor,
    key: String,
    activate: Bool = True,
) raises -> Tensor:
    if len(gpu):
        return gpu[0].group(w, x^, key, activate)
    var c = x.shape[2]
    var span = c // 32
    var gain = w.tensor(key + ".weight", [c])
    var bias = w.tensor(key + ".bias", [c])
    var means = List[Float64](length=32, fill=0)
    var variances = List[Float64](length=32, fill=0)
    var count = Float64(len(x.f) // 32)
    for i in range(len(x.f)):
        means[(i % c) // span] += Float64(x.f[i]) / count
    for i in range(len(x.f)):
        var g = (i % c) // span
        var v = Float64(x.f[i]) - means[g]
        variances[g] += v * v / count
    for g in range(32):
        variances[g] = 1 / sqrt(variances[g] + 1e-6)
    for i in range(len(x.f)):
        var j = i % c
        var g = j // span
        var v = (x.f[i] - Float32(means[g])) * Float32(variances[g]) * gain.f[
            j
        ] + bias.f[j]
        x.f[i] = v / (1 + exp(-v)) if activate else v
    return x^


def _res(
    w: SafeTensorWeights,
    mut pool: JobPool,
    mut gpu: List[UltraTexVaeGpu],
    var x: Tensor,
    key: String,
    co: Int,
) raises -> Tensor:
    var h = _conv(
        w,
        pool,
        gpu,
        _group(w, gpu, x.copy(), key + ".norm1"),
        key + ".conv1",
        co,
        3,
    )
    h = _conv(
        w, pool, gpu, _group(w, gpu, h^, key + ".norm2"), key + ".conv2", co, 3
    )
    if x.shape[2] != co:
        x = _conv(w, pool, gpu, x, key + ".nin_shortcut", co, 1)
    for i in range(len(x.f)):
        x.f[i] += h.f[i]
    return x^


def _mid(
    w: SafeTensorWeights,
    mut pool: JobPool,
    mut gpu: List[UltraTexVaeGpu],
    var x: Tensor,
    root: String,
) raises -> Tensor:
    x = _res(w, pool, gpu, x^, root + ".block_1", 512)
    var key = root + ".attn_1."
    var h = _group(w, gpu, x.copy(), key + "norm", False)
    var q = _conv(w, pool, gpu, h, key + "q", 512, 1)
    var k = _conv(w, pool, gpu, h, key + "k", 512, 1)
    var v = _conv(w, pool, gpu, h, key + "v", 512, 1)
    if len(gpu):
        h = gpu[0].attention(q, k, v)
    else:
        var n = len(q.f) // 512
        var tile = min(32, n)
        var scores = Tensor.zeros([tile, n])
        for first in range(0, n, tile):
            var rows = min(tile, n - first)
            matmul_2d(
                pool,
                q.fptr() + first * 512 * 4,
                k.fptr(),
                scores.fptr(),
                0,
                rows,
                512,
                n,
                True,
            )
            for r in range(rows):
                var off = r * n
                var peak = scores.f[off]
                for j in range(1, n):
                    peak = max(peak, scores.f[off + j])
                var total = Float32(0)
                for j in range(n):
                    scores.f[off + j] = exp(
                        (scores.f[off + j] - peak) / sqrt(Float32(512))
                    )
                    total += scores.f[off + j]
                for j in range(n):
                    scores.f[off + j] /= total
            matmul_2d(
                pool,
                scores.fptr(),
                v.fptr(),
                h.fptr() + first * 512 * 4,
                0,
                rows,
                n,
                512,
                False,
            )
        _ = q.fptr()
        _ = k.fptr()
        _ = v.fptr()
    h = _conv(w, pool, gpu, h, key + "proj_out", 512, 1)
    for i in range(len(h.f)):
        x.f[i] += h.f[i]
    return _res(w, pool, gpu, x^, root + ".block_2", 512)


def _up(x: Tensor) -> Tensor:
    var width = x.shape[1]
    var c = x.shape[2]
    var out = Tensor.zeros([2 * x.shape[0], 2 * width, c])
    for y in range(out.shape[0]):
        for col in range(2 * width):
            var src = ((y // 2) * width + col // 2) * c
            for j in range(c):
                out.f[(y * 2 * width + col) * c + j] = x.f[src + j]
    return out^


struct UltraTexVAE(Movable):
    var base: SafeTensorWeights
    var decoder: SafeTensorWeights
    var pool: JobPool
    var gpu: List[UltraTexVaeGpu]
    var mean: Tensor
    var scale: Tensor

    def __init__(out self, base_path: String, decoder_path: String) raises:
        self.base = SafeTensorWeights(base_path, 1 << 30)
        self.decoder = SafeTensorWeights(decoder_path, 1 << 30)
        self.mean = self.base.tensor("bn.running_mean", [128])
        self.scale = self.base.tensor("bn.running_var", [128])
        for i in range(128):
            self.scale.f[i] = sqrt(self.scale.f[i] + 1e-4)
        self.gpu = List[UltraTexVaeGpu]()
        if env_now("FLINT_ULTRATEX_GPU") == "1":
            var device = UltraTexVaeGpu()
            if device.ctx.target.is_nvidia_gpu():
                self.gpu.append(device^)
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())

    def encode(mut self, image: Tensor) raises -> Tensor:
        """Encode [H,W,3] RGB in [-1,1] to normalized [H/16,W/16,128]."""
        if (
            image.dtype != DT_F32
            or len(image.shape) != 3
            or image.shape[2] != 3
            or min(image.shape[0], image.shape[1]) < 16
            or image.shape[0] % 16
            or image.shape[1] % 16
            or len(image.f) != image.numel()
        ):
            raise Error(
                "Expected an HWC RGB tensor with height and width divisible"
                " by 16"
            )
        var x = _conv(
            self.base, self.pool, self.gpu, image, "encoder.conv_in", 128, 3
        )
        for i in range(4):
            var key = "encoder.down." + String(i) + "."
            var co = min(512, 128 << i)
            for b in range(2):
                x = _res(
                    self.base,
                    self.pool,
                    self.gpu,
                    x^,
                    key + "block." + String(b),
                    co,
                )
            if i < 3:
                x = _conv(
                    self.base,
                    self.pool,
                    self.gpu,
                    x,
                    key + "downsample.conv",
                    co,
                    3,
                    2,
                )
        x = _mid(self.base, self.pool, self.gpu, x^, "encoder.mid")
        x = _conv(
            self.base,
            self.pool,
            self.gpu,
            _group(self.base, self.gpu, x^, "encoder.norm_out"),
            "encoder.conv_out",
            64,
            3,
        )
        x = _conv(
            self.base, self.pool, self.gpu, x, "encoder.quant_conv", 64, 1
        )
        var h = x.shape[0] // 2
        var width = x.shape[1] // 2
        var packed = Tensor.zeros([h, width, 128])
        for p in range(h * width):
            for j in range(128):
                var orig = (
                    (2 * (p // width) + (j % 4) // 2) * 2 * width
                    + 2 * (p % width)
                    + j % 2
                )
                packed.f[p * 128 + j] = (
                    x.f[orig * 64 + j // 4] - self.mean.f[j]
                ) / self.scale.f[j]
        return packed^

    def decode(mut self, z: Tensor) raises -> Tensor:
        """Decode one normalized HWC latent tile with the trained UltraTex decoder.
        """
        if (
            z.dtype != DT_F32
            or len(z.shape) != 3
            or z.shape[2] != 128
            or min(z.shape[0], z.shape[1]) < 1
            or len(z.f) != z.numel()
        ):
            raise Error("Expected [H,W,128] packed FLUX.2 latents")
        var width = z.shape[1]
        var x = Tensor.zeros([z.shape[0] * 2, width * 2, 32])
        for p in range(z.shape[0] * width):
            for j in range(128):
                var dst = (
                    (2 * (p // width) + (j % 4) // 2) * 2 * width
                    + 2 * (p % width)
                    + j % 2
                )
                x.f[dst * 32 + j // 4] = (
                    z.f[p * 128 + j] * self.scale.f[j] + self.mean.f[j]
                )
        x = _conv(
            self.decoder, self.pool, self.gpu, x, "post_quant_conv", 32, 1
        )
        x = _conv(self.decoder, self.pool, self.gpu, x, "conv_in", 512, 3)
        x = _mid(self.decoder, self.pool, self.gpu, x^, "mid")
        for stage in range(3, -1, -1):
            var key = "up." + String(stage) + "."
            var co = min(512, 128 << stage)
            for b in range(3):
                x = _res(
                    self.decoder,
                    self.pool,
                    self.gpu,
                    x^,
                    key + "block." + String(b),
                    co,
                )
            if stage > 0:
                x = _conv(
                    self.decoder,
                    self.pool,
                    self.gpu,
                    _up(x),
                    key + "upsample.conv",
                    co,
                    3,
                )
        return _conv(
            self.decoder,
            self.pool,
            self.gpu,
            _group(self.decoder, self.gpu, x^, "norm_out"),
            "conv_out",
            3,
            3,
        )
