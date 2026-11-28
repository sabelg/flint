"""MT19937 and the CPU float32 Box–Muller layout used to seed Hi3D latents."""
from std.math import cos, log, sin, sqrt
from flint.ml.tensor import Tensor


struct TorchNoiseStream(Movable):
    var state: List[UInt32]
    var index: Int

    def __init__(out self, seed: UInt32):
        self.state = List[UInt32](length=624, fill=0)
        self.state[0] = seed
        for i in range(1, 624):
            self.state[i] = UInt32(1812433253) * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + UInt32(i)
        self.index = 624

    def rows(mut self, rows: Int) raises -> Tensor:
        if rows < 1 or rows > 64**3:
            raise Error("Noise site count exceeds the 1024-resolution grid")
        var noise = Tensor.zeros([rows, 32])
        for first in range(0, len(noise.f), 16):
            for j in range(16):
                if self.index == 624:
                    for i in range(624):
                        var y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7FFFFFFF)
                        self.state[i] = (
                            self.state[(i + 397) % 624] ^ (y >> 1) ^ ((UInt32(0) - (y & 1)) & 0x9908B0DF)
                        )
                    self.index = 0
                var y = self.state[self.index]
                self.index += 1
                y ^= y >> 11
                y ^= (y << 7) & 0x9D2C5680
                y ^= (y << 15) & 0xEFC60000
                y ^= y >> 18
                noise.f[first + j] = Float32(y & 0xFFFFFF) / 16777216
            for j in range(8):
                var rad = sqrt(-2 * log(1 - noise.f[first + j]))
                var phase = Float32(6.283185307179586 * Float64(noise.f[first + 8 + j]))
                noise.f[first + j] = rad * cos(phase)
                noise.f[first + 8 + j] = rad * sin(phase)
        return noise^


def hi3d_noise(rows: Int, seed: UInt32) raises -> Tensor:
    if rows < 1 or rows > 80000:
        raise Error("Hi3D noise requires 1 to 80000 sparse sites")
    var stream = TorchNoiseStream(seed)
    return stream.rows(rows)
