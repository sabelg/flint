"""TRELLIS.2-4B dense structure flow schedule and single-sample interval guidance."""

from std.math import isfinite, sqrt
from flint.ml.tensor import Tensor


trait StructureVelocityModel:
    def predict(mut self, latent: Tensor, model_time: Float32, conditional: Bool) raises -> Tensor:
        """Evaluate one dense structure transformer pass with image or zero conditioning."""
        ...


def structure_time(step: Int, steps: Int = 12) raises -> Float64:
    if steps < 1 or step < 0 or step > steps:
        raise Error("Invalid structure flow schedule index")
    var linear = Float64(steps - step) / Float64(steps)
    return 5 * linear / (1 + 4 * linear)


def _check_velocity(x: Tensor, velocity: Tensor) raises:
    if not velocity.is_float() or velocity.shape != x.shape or not velocity.has_valid_storage():
        raise Error("Structure flow model returned incompatible velocity")


def sample_structure_flow[M: StructureVelocityModel](mut model: M, var latent: Tensor, steps: Int = 12) raises -> Tensor:
    """Integrate the released structure sampler from supplied noise, using a caller-owned model."""
    if (steps < 1 or latent.rank() != 4 or latent.shape[0] != 8 or latent.shape[1] < 1
            or latent.shape[2] != latent.shape[1] or latent.shape[3] != latent.shape[1]
            or not latent.is_float() or not latent.has_valid_storage()):
        raise Error("Invalid structure flow latent or step count")
    if latent.ext != 0:
        latent = Tensor.floats(latent.shape.copy(), latent.owned_floats())
    var count = latent.numel()
    for i in range(steps):
        var time = structure_time(i, steps)
        var dt = Float32(time - structure_time(i + 1, steps))
        var model_time = Float32(1000 * time)
        var pos = model.predict(latent, model_time, True)
        _check_velocity(latent, pos)
        if time < 0.6:
            for p in range(count):
                latent.f[p] -= dt * pos.fget(p)
            continue

        var neg = model.predict(latent, model_time, False)
        _check_velocity(latent, neg)
        var guided = List[Float32](length=count, fill=0)
        var sigma = Float32(1e-5)
        var s = sigma + (1 - sigma) * Float32(time)
        var avg_pos = Float64(0)
        var avg_guided = Float64(0)
        var var_pos = Float64(0)
        var var_guided = Float64(0)
        for p in range(count):
            var x = (1 - sigma) * latent.f[p]
            var pos_x0 = x - s * pos.fget(p)
            var cfg_x0 = x - s * (Float32(7.5) * pos.fget(p) - Float32(6.5) * neg.fget(p))
            guided[p] = cfg_x0
            # Welford sums; the sample-variance denominators cancel in the ratio.
            var delta_pos = Float64(pos_x0) - avg_pos
            var delta_cfg = Float64(cfg_x0) - avg_guided
            avg_pos += delta_pos / Float64(p + 1)
            avg_guided += delta_cfg / Float64(p + 1)
            var_pos += delta_pos * (Float64(pos_x0) - avg_pos)
            var_guided += delta_cfg * (Float64(cfg_x0) - avg_guided)
        if var_guided <= 0 or not isfinite(var_guided) or not isfinite(var_pos):
            raise Error("Degenerate structure flow prediction during guidance rescale")
        var ratio = Float32(sqrt(var_pos / var_guided))
        for p in range(count):
            var x0 = Float32(0.7) * (guided[p] * ratio) + Float32(0.3) * guided[p]
            var velocity = ((1 - sigma) * latent.f[p] - x0) / s
            latent.f[p] -= dt * velocity
    return latent^
