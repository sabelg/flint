"""PyTorch-compatible CPU random numbers (`flint.ml.diffusion.rng`).

`TorchGenerator(seed)` reproduces `torch.Generator().manual_seed(seed)` on the CPU:

  * the engine is PyTorch's `at::mt19937` (standard MT19937, seeded with the low 32 bits);
  * `randn(n)` follows ATen's `normal_fill` for float32 tensors of 16+ elements (what
    `diffusers.utils.torch_utils.randn_tensor` draws for latents and scheduler noise on a CPU
    generator): n uniforms `(u32 & 0xFFFFFF) * 2^-24`, then Box-Muller over blocks of 16
    (values j and j + 8 of each block are the cos / sin pair; a ragged tail recomputes the last 16
    from fresh uniforms). `logf` / `cosf` / `sinf` come from the C library, as in ATen's scalar
    (non-AVX2, e.g. aarch64) kernel, so the streams are bit-identical there;
  * `randn(n)` with n < 16 uses ATen's per-element `normal_distribution<double>`.

So a seed passed to Flint gives the same initial latents (and LCM step noise) as the same seed in
a diffusers pipeline with `generator=torch.Generator().manual_seed(seed)`.
"""

from std.ffi import external_call
from std.math import sqrt, log, cos, sin

comptime _N: Int = 624
comptime _M: Int = 397
comptime _PI: Float64 = 3.141592653589793


struct TorchGenerator(Copyable, Movable):
    var state: List[UInt32]
    var index: Int
    var next_normal: Float64
    var has_next_normal: Bool

    def __init__(out self, seed: Int):
        self.state = List[UInt32](length=_N, fill=0)
        self.index = _N
        self.next_normal = 0.0
        self.has_next_normal = False
        self.seed(seed)

    def seed(mut self, seed: Int):
        self.state[0] = UInt32(UInt64(seed) & 0xFFFFFFFF)
        for j in range(1, _N):
            var p = self.state[j - 1]
            self.state[j] = UInt32(1812433253) * (p ^ (p >> 30)) + UInt32(j)
        self.index = _N
        self.has_next_normal = False

    def _twist(mut self):
        for i in range(_N):
            var y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % _N] & 0x7FFFFFFF)
            var v = self.state[(i + _M) % _N] ^ (y >> 1)
            if (y & 1) != 0:
                v ^= UInt32(0x9908B0DF)
            self.state[i] = v
        self.index = 0

    def random(mut self) -> UInt32:
        """One 32-bit MT19937 output (`CPUGeneratorImpl::random`)."""
        if self.index >= _N:
            self._twist()
        var y = self.state[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def random64(mut self) -> UInt64:
        var hi = UInt64(self.random())
        var lo = UInt64(self.random())
        return (hi << 32) | lo

    def uniform_f32(mut self) -> Float32:
        """`uniform_real_distribution<float>(0, 1)`: 24 random bits."""
        return Float32(Float64(self.random() & 0xFFFFFF) * (1.0 / 16777216.0))

    def uniform_f64(mut self) -> Float64:
        return Float64(self.random64() & ((UInt64(1) << 53) - 1)) * (1.0 / 9007199254740992.0)

    def normal_f64(mut self) -> Float64:
        """`at::normal_distribution<double>(0, 1)` with its cached second sample."""
        if self.has_next_normal:
            self.has_next_normal = False
            return self.next_normal
        var u1 = self.uniform_f64()
        var u2 = self.uniform_f64()
        var r = sqrt(-2.0 * external_call["log1p", Float64](-u2))
        var theta = 2.0 * _PI * u1
        self.next_normal = r * sin(theta)
        self.has_next_normal = True
        return r * cos(theta)

    def randn(mut self, n: Int) -> List[Float32]:
        """`torch.randn(n, generator=g)` for float32 on the CPU."""
        var out = List[Float32](length=n, fill=0.0)
        if n < 16:
            for i in range(n):
                out[i] = Float32(self.normal_f64())
            return out^
        for i in range(n):
            out[i] = self.uniform_f32()
        var i = 0
        while i + 16 <= n:
            _normal_fill_16(out, i)
            i += 16
        if n % 16 != 0:
            var base = n - 16
            for j in range(16):
                out[base + j] = self.uniform_f32()
            _normal_fill_16(out, base)
        return out^


def _normal_fill_16(mut d: List[Float32], base: Int):
    for j in range(8):
        var u1 = Float32(1.0) - d[base + j]
        var u2 = d[base + j + 8]
        var lg = external_call["logf", Float32](u1)
        var radius = external_call["sqrtf", Float32](Float32(-2.0) * lg)
        var theta = Float32(2.0 * _PI * Float64(u2))
        d[base + j] = radius * external_call["cosf", Float32](theta)
        d[base + j + 8] = radius * external_call["sinf", Float32](theta)
