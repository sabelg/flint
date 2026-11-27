"""Deterministic seeded random numbers (`flint.core.rng`).

`DetRng` is a PCG32 (XSH-RR 64/32, O'Neill 2014) generator: 64-bit state plus a
64-bit odd stream increment, so independent streams can be forked from one seed.
Every operation is pure integer arithmetic (floats are built from integer bits
scaled by an exact power of two), so the same seed gives bit-identical output on
every machine, build mode and `--fp-mode`. The whole state is two words, which a
rollback snapshot captures via `serialize`/`deserialize` or `state_words`.
"""
from flint.core.serial import BlobReader, BlobWriter, Serializable, fourcc
from flint.core.detsort import mix64

comptime DET_RNG_BLOB_TAG: UInt32 = fourcc("DRNG")
comptime DET_RNG_BLOB_VERSION: Int = 1
comptime _PCG_MULT: UInt64 = 6364136223846793005
comptime _DEFAULT_STREAM: UInt64 = 0xDA3E39CB94B95BDB


struct DetRng(Copyable, ImplicitlyCopyable, Movable, Equatable, Serializable):
    """PCG32 deterministic RNG with forkable streams and serializable two-word state."""
    var state: UInt64
    var inc: UInt64

    def __init__(out self, seed: UInt64 = 0x853C49E6748FEA9B, stream: UInt64 = _DEFAULT_STREAM):
        self.state = 0
        self.inc = (stream << 1) | 1
        _ = self.next_u32()
        self.state += seed
        _ = self.next_u32()

    @staticmethod
    def from_state(state: UInt64, inc: UInt64) -> Self:
        """Rebuilds a generator from `state_words()` exactly (no re-seeding)."""
        var r = Self(0, 0)
        r.state = state
        r.inc = inc | 1
        return r

    def __eq__(self, other: Self) -> Bool:
        return self.state == other.state and self.inc == other.inc

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    def next_u32(mut self) -> UInt32:
        var old = self.state
        self.state = old * _PCG_MULT + self.inc
        var xorshifted = UInt32(((old >> 18) ^ old) >> 27)
        var rot = UInt32(old >> 59)
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1) & 31))

    @always_inline
    def next_u64(mut self) -> UInt64:
        var hi = UInt64(self.next_u32())
        var lo = UInt64(self.next_u32())
        return (hi << 32) | lo

    @always_inline
    def next_f32(mut self) -> Float32:
        """Uniform in [0, 1): 24 random mantissa bits times 2^-24."""
        return Float32(self.next_u32() >> 8) * Float32(5.9604644775390625e-08)

    @always_inline
    def next_f64(mut self) -> Float64:
        """Uniform in [0, 1): 53 random bits times 2^-53."""
        return Float64(self.next_u64() >> 11) * Float64(1.1102230246251565e-16)

    def next_bool(mut self) -> Bool:
        return (self.next_u32() >> 31) != 0

    def below(mut self, bound: UInt32) -> UInt32:
        """Unbiased uniform integer in [0, bound) (rejection sampling); 0 when `bound == 0`."""
        if bound <= 1:
            return 0
        var threshold = (~bound + 1) % bound
        while True:
            var r = self.next_u32()
            if r >= threshold:
                return r % bound

    def range_int(mut self, lo: Int, hi: Int) -> Int:
        """Unbiased uniform integer in [lo, hi); returns `lo` when the range is empty."""
        if hi <= lo:
            return lo
        var span = UInt64(hi - lo)
        if span <= UInt64(0xFFFFFFFF):
            return lo + Int(self.below(UInt32(span)))
        var threshold = (~span + 1) % span
        while True:
            var r = self.next_u64()
            if r >= threshold:
                return lo + Int(r % span)

    def range_f32(mut self, lo: Float32, hi: Float32) -> Float32:
        """Uniform in [lo, hi)."""
        return lo + (hi - lo) * self.next_f32()

    def chance(mut self, probability: Float32) -> Bool:
        return self.next_f32() < probability

    def fork(mut self, stream: UInt64) -> Self:
        """Derives an independent child generator on stream `stream`.

        Advances this generator by one draw, so forking is itself deterministic and
        two forks with the same `stream` id at different times still differ."""
        var child_seed = mix64(self.next_u64() ^ mix64(stream))
        return Self(child_seed, stream ^ _DEFAULT_STREAM)

    def advance(mut self, delta: UInt64):
        """Jumps ahead `delta` draws in O(log delta) (Brown's LCG skip-ahead)."""
        var cur_mult = _PCG_MULT
        var cur_plus = self.inc
        var acc_mult = UInt64(1)
        var acc_plus = UInt64(0)
        var d = delta
        while d > 0:
            if (d & 1) != 0:
                acc_mult *= cur_mult
                acc_plus = acc_plus * cur_mult + cur_plus
            cur_plus = (cur_mult + 1) * cur_plus
            cur_mult *= cur_mult
            d >>= 1
        self.state = acc_mult * self.state + acc_plus

    def state_words(self) -> Tuple[UInt64, UInt64]:
        return (self.state, self.inc)

    def serialize(self, mut out: BlobWriter):
        out.begin(DET_RNG_BLOB_TAG, DET_RNG_BLOB_VERSION)
        out.u64(self.state)
        out.u64(self.inc)
        out.end()

    def deserialize(mut self, mut input: BlobReader) raises:
        _ = input.enter(DET_RNG_BLOB_TAG, DET_RNG_BLOB_VERSION)
        var s = input.u64()
        var i = input.u64()
        input.leave()
        if input.failed:
            raise Error("DetRng: truncated state blob")
        self.state = s
        self.inc = i | 1
