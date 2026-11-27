"""Hi3D 512 image- and mesh-conditioned refinement on the CPU, in Mojo 1.2."""
from std.math import isfinite
from flint.assets.image import NativeImage
from flint.assets.json import parse_json
from flint.assets.model import NativeModel
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_decoder import Hi3DField
from flint.ml.hi3d_flow import hi3d_sample
from flint.ml.hi3d_image import hi3d_condition
from flint.ml.hi3d_mesh import hi3d_surface, hi3d_model
from flint.ml.hi3d_transformer import Hi3DTransformer
from flint.ml.hi3d_voxels import hi3d_model_voxels
from flint.ml.hi3d_noise import hi3d_noise
from flint.ml.tensor import Tensor
from flint.platform.os import read_bytes


def hi3d_latents(mut net: Hi3DTransformer, var x: Tensor, root: String, steps: Int) raises -> Tensor:
    var cfg = parse_json(read_bytes(root + "/config.json"))
    var mean = cfg.object_get(cfg.root, "latent_mean")
    var std = cfg.object_get(cfg.root, "latent_std")
    if not cfg.valid or cfg.array_size(mean) != 32 or cfg.array_size(std) != 32:
        raise Error("Expected 32 latent channel statistics")
    var means = List[Float32]()
    var scales = List[Float32]()
    for j in range(32):
        if cfg.nodes[cfg.at(mean, j)].kind != 2 or cfg.nodes[cfg.at(std, j)].kind != 2:
            raise Error("Expected numeric latent channel statistics")
        var m = Float32(cfg.number(cfg.at(mean, j)))
        var s = Float32(cfg.number(cfg.at(std, j)))
        if not isfinite(m) or not isfinite(s) or s <= 0:
            raise Error("Invalid latent channel statistics")
        means.append(m)
        scales.append(s)
    var latent = hi3d_sample(net, x^, steps)
    for i in range(latent.shape[0]):
        for j in range(32):
            latent.f[i * 32 + j] = latent.f[i * 32 + j] * scales[j] + means[j]
    return latent^


def _predict(model_dir: String, coord: Tensor, var cond: Tensor, seed: UInt32, steps: Int) raises -> Tensor:
    var root = model_dir + "/transformer_512"
    var net = Hi3DTransformer(root + "/model.safetensors", coord.copy(), cond^)
    var noise = hi3d_noise(coord.shape[0], seed)
    return hi3d_latents(net, noise^, root, steps)


def _condition(image: NativeImage, dino_path: String, clip_path: String) raises -> Tensor:
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    return hi3d_condition(pool, image, dino_path, clip_path)


def refine_hi3d(
    model_dir: String,
    dino_path: String,
    clip_path: String,
    image: NativeImage,
    mut coarse: NativeModel,
    seed: UInt32 = 0,
    steps: Int = 25,
) raises -> NativeModel:
    if steps < 1:
        raise Error("Hi3D requires a positive step count")
    var coords = hi3d_model_voxels(coarse)
    var cond = _condition(image, dino_path, clip_path)
    var latents = _predict(model_dir, coords, cond^, seed, steps)
    var field = Hi3DField(coords^, latents^, model_dir + "/vae/model.safetensors")
    return hi3d_model(hi3d_surface(field.grid, field.attributes))
