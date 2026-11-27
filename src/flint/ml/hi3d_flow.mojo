"""Token-warped Hi3D flow schedule, Mojo port. See licenses/hi3d."""
from flint.ml.tensor import Tensor, DT_F32


trait Hi3DVelocityModel:
    def velocity(mut self, x: Tensor, time: Float32, conditional: Bool) raises -> Tensor:
        ...


def _warp(step: Int, steps: Int, tokens: Int) -> Float32:
    var u = Float32(steps - step) / Float32(steps)
    u = 2 * u / (1 + u)
    var n = Float64(tokens)
    var k = Float32(max(Float64(1), min(Float64(5), (19 * n - 20000) / (3 * n + 60000))))
    return k * u / (1 + (k - 1) * u)


def hi3d_times(step: Int, steps: Int, tokens: Int) raises -> Tuple[Float32, Float32]:
    if steps < 1 or step < 0 or step >= steps or tokens < 1:
        raise Error("Invalid Hi3D integration step")
    var w = _warp(step, steps, tokens)
    var sigma = Float32(0.002) + Float32(0.998) * w
    var inflated = sigma * (Float32(1.04) if w > 0.985 else Float32(1.002))
    return (inflated, Float32(0.002) + Float32(0.998) * _warp(step + 1, steps, tokens))


def _slope[M: Hi3DVelocityModel](mut model: M, x: Tensor, t: Float32) raises -> Tensor:
    var pos = model.velocity(x, 1000 * t, True)
    if pos.shape != x.shape or pos.dtype != DT_F32 or len(pos.f) != len(x.f):
        raise Error("Invalid Hi3D conditional prediction")
    if t >= 0.7:
        var neg = model.velocity(x, 1000 * t, False)
        if neg.shape != x.shape or neg.dtype != DT_F32 or len(neg.f) != len(x.f):
            raise Error("Invalid Hi3D unconditional prediction")
        var scale = Float32(4) if t >= 1 else 10 * t * t
        for i in range(len(x.f)):
            pos.f[i] = neg.f[i] + scale * (pos.f[i] - neg.f[i])
    return pos^


def hi3d_sample[M: Hi3DVelocityModel](mut model: M, var noise: Tensor, steps: Int = 25) raises -> Tensor:
    """Integrate caller-supplied [N,32] noise, returning latents before channel denormalization."""
    if (
        steps < 1
        or len(noise.shape) != 2
        or noise.shape[1] != 32
        or noise.shape[0] < 1
        or noise.dtype != DT_F32
        or len(noise.f) != noise.shape[0] * 32
    ):
        raise Error("Invalid Hi3D noise tensor or step count")
    for step in range(steps):
        var times = hi3d_times(step, steps, noise.shape[0])
        var dt = times[0] - times[1]
        var a = _slope(model, noise, times[0])
        var predicted = noise.copy()
        for i in range(len(noise.f)):
            predicted.f[i] -= dt * a.f[i]
        var b = _slope(model, predicted, times[1])
        for i in range(len(noise.f)):
            noise.f[i] -= dt * (Float32(0.5) * (a.f[i] + b.f[i]))
    return noise^
