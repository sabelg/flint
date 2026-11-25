"""Bit-exact `jax.random` (threefry2x32, partitionable mode) for JAX ports (`flint.ml.jax_random`).

JAX's default PRNG since 0.5 is threefry2x32 with `jax_threefry_partitionable=True`. With a key
`(k0, k1)` and a 64-bit counter split into `(hi, lo)` uint32 halves:

  - `split(key, n)[i]`   = threefry2x32(key, (0, i))
  - `fold_in(key, d)`    = threefry2x32(key, (0, d))        (identical to `split(key, d + 1)[d]`)
  - `bits(key, (n,))[i]` = x0 ^ x1 of threefry2x32(key, (0, i))
  - `uniform(key, lo, hi)` (legacy mantissa mode, the default): `bitcast(bits >> 9 | 0x3F800000) - 1`,
    scaled to [lo, hi) and clamped below at lo.
  - `gumbel(key)` (mode "low", the default): `-log(-log(uniform(key, tiny, 1)))`.

`PRNGKey(seed)` for a 32-bit seed is `(0, seed)`. Only flat (1-D) shapes are needed here; the
counter of element i of an n-element draw is i, the row-major index, for any shape.
"""

from std.math import log
from std.memory import bitcast


@fieldwise_init
struct JaxKey(Copyable, Movable, ImplicitlyCopyable):
    var k0: UInt32
    var k1: UInt32


@always_inline
def _rotl(x: UInt32, r: UInt32) -> UInt32:
    return (x << r) | (x >> (32 - r))


@always_inline
def threefry2x32(key: JaxKey, c0: UInt32, c1: UInt32) -> Tuple[UInt32, UInt32]:
    """The 20-round Threefry-2x32 block function (Salmon et al.), as in `jax.random`."""
    var ks0 = key.k0
    var ks1 = key.k1
    var ks2 = ks0 ^ ks1 ^ 0x1BD11BDA
    var x0 = c0 + ks0
    var x1 = c1 + ks1
    var ks = SIMD[DType.uint32, 4](ks0, ks1, ks2, 0)
    for i in range(5):
        if i % 2 == 0:
            x0 += x1; x1 = _rotl(x1, 13); x1 ^= x0
            x0 += x1; x1 = _rotl(x1, 15); x1 ^= x0
            x0 += x1; x1 = _rotl(x1, 26); x1 ^= x0
            x0 += x1; x1 = _rotl(x1, 6); x1 ^= x0
        else:
            x0 += x1; x1 = _rotl(x1, 17); x1 ^= x0
            x0 += x1; x1 = _rotl(x1, 29); x1 ^= x0
            x0 += x1; x1 = _rotl(x1, 16); x1 ^= x0
            x0 += x1; x1 = _rotl(x1, 24); x1 ^= x0
        x0 += ks[(i + 1) % 3]
        x1 += ks[(i + 2) % 3] + UInt32(i + 1)
    return (x0, x1)


def prng_key(seed: Int) -> JaxKey:
    """`jax.random.PRNGKey(seed)` for a non-negative seed below 2^32."""
    return JaxKey(UInt32((seed >> 32) & 0xFFFFFFFF), UInt32(seed & 0xFFFFFFFF))


def split_one(key: JaxKey, n: Int, i: Int) -> JaxKey:
    """Element `i` of `jax.random.split(key, n)` (independent of n in partitionable mode)."""
    _ = n
    var r = threefry2x32(key, 0, UInt32(i))
    return JaxKey(r[0], r[1])


def fold_in(key: JaxKey, data: Int) -> JaxKey:
    """`jax.random.fold_in(key, data)` for 0 <= data < 2^32."""
    var r = threefry2x32(key, 0, UInt32(data & 0xFFFFFFFF))
    return JaxKey(r[0], r[1])


@always_inline
def random_bits32(key: JaxKey, i: Int) -> UInt32:
    """Element `i` (row-major) of `jax.random.bits(key, shape, uint32)`."""
    var r = threefry2x32(key, UInt32((i >> 32) & 0xFFFFFFFF), UInt32(i & 0xFFFFFFFF))
    return r[0] ^ r[1]


@always_inline
def uniform01(key: JaxKey, i: Int, minval: Float32) -> Float32:
    """Element `i` of `jax.random.uniform(key, shape, float32, minval, 1.0)` (legacy mode)."""
    var b = (random_bits32(key, i) >> 9) | 0x3F800000
    var f = bitcast[DType.float32](b) - 1.0
    var v = f * (Float32(1.0) - minval) + minval
    return max(minval, v)


@always_inline
def gumbel(key: JaxKey, i: Int) -> Float32:
    """Element `i` of `jax.random.gumbel(key, shape, float32)` (mode "low")."""
    var u = uniform01(key, i, Float32(1.17549435e-38))
    return -log(-log(u))
