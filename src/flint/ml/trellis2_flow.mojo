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
    return _flow_time(step, steps, 5)


def _flow_time(step: Int, steps: Int, shift: Float64) -> Float64:
    var linear = Float64(steps - step) / Float64(steps)
    return shift * linear / (1 + (shift - 1) * linear)


def _check_velocity(x: Tensor, velocity: Tensor) raises:
    if not velocity.is_float() or velocity.shape != x.shape or not velocity.has_valid_storage():
        raise Error("Structure flow model returned incompatible velocity")


def sample_structure_flow[M: StructureVelocityModel](mut model: M, var latent: Tensor, steps: Int = 12) raises -> Tensor:
    """Integrate the released structure sampler from supplied noise, using a caller-owned model."""
    if (steps < 1 or latent.rank() != 4 or latent.shape[0] != 8 or latent.shape[1] < 1
            or latent.shape[2] != latent.shape[1] or latent.shape[3] != latent.shape[1]
            or not latent.is_float() or not latent.has_valid_storage()):
        raise Error("Invalid structure flow latent or step count")
    return _sample_flow(model, latent^, steps, 5, 7.5, 0.7)


def sample_pixal_slat[M: StructureVelocityModel](mut model: M, var latent: Tensor,
                                                steps: Int, texture: Bool) raises -> Tensor:
    """Released single-view Pixal3D shape or material schedule, in normalized SLat space."""
    if (steps < 1 or latent.rank() != 2 or latent.shape[0] < 1 or latent.shape[1] != 32
            or not latent.is_float() or not latent.has_valid_storage()):
        raise Error("Invalid Pixal3D sparse latent or step count")
    return _sample_flow(model, latent^, steps, 3, Float32(1) if texture else Float32(7.5), Float32(0) if texture else Float32(0.5))


def _sample_flow[M: StructureVelocityModel](mut model: M, var latent: Tensor, steps: Int,
        shift: Float64, guidance: Float32, rescale: Float32) raises -> Tensor:
    if latent.ext != 0:
        latent = Tensor.floats(latent.shape.copy(), latent.owned_floats())
    var count = latent.numel()
    for i in range(steps):
        var time = _flow_time(i, steps, shift)
        var dt = Float32(time - _flow_time(i + 1, steps, shift))
        var model_time = Float32(1000 * time)
        var pos = model.predict(latent, model_time, True)
        _check_velocity(latent, pos)
        if guidance == 1 or time < 0.6:
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
            var cfg_x0 = x - s * (guidance * pos.fget(p) + (1-guidance) * neg.fget(p))
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
            var x0 = rescale * (guided[p] * ratio) + (1-rescale) * guided[p]
            var velocity = ((1 - sigma) * latent.f[p] - x0) / s
            latent.f[p] -= dt * velocity
    return latent^
