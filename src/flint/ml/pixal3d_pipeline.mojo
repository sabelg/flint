"""Native single-view Pixal3D image-to-GLB cascade using original SafeTensors."""
from flint.assets.model import NativeModel
from flint.assets.image import NativeImage
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_noise import TorchNoiseStream
from flint.ml.pixal3d import PixalCamera, structure_condition
from flint.ml.pixal3d_decoder import cascade_coords, decode_slat, denormalize_slat
from flint.ml.pixal3d_mesh import pixal_surface
from flint.ml.pixal3d_naf import sparse_condition
from flint.ml.pixal3d_flow import PixalFlow
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.trellis2 import TrellisStructureDecoder, structure_coordinates
from flint.ml.trellis2_dino import TrellisImageEncoder
from flint.ml.trellis2_flow import sample_structure_flow, sample_pixal_slat
from flint.ml.trellis2_image import prepare_trellis_image


def _structure(
    base: String,
    tokens: Tensor,
    camera: PixalCamera,
    mut rng: TorchNoiseStream,
    mut pool: JobPool,
    steps: Int,
) raises -> Tensor:
    var conditions = structure_condition(tokens, camera)
    var globals = conditions[0].copy()
    var projected = conditions[1].copy()
    globals.shape = [5, 1024]
    projected.shape = [4096, 1024]
    var flow = PixalFlow(base + "ss_flow_img_dit_1_3B_64_bf16.safetensors", globals^, projected^)
    var noise = rng.rows(1024)
    noise.shape = [8, 16, 16, 16]
    var latent = sample_structure_flow(flow, noise^, steps)
    var decoder = TrellisStructureDecoder(base + "ss_dec_conv3d_16l8_fp16.safetensors")
    return structure_coordinates(decoder.decode(pool, latent), 32)


def _sample(
    path: String,
    image: NativeImage,
    tokens: Tensor,
    coords: Tensor,
    camera: PixalCamera,
    naf: SafeTensorWeights,
    size: Int,
    target: Int,
    edge: Int,
    mut rng: TorchNoiseStream,
    steps: Int,
    var shape: Tensor = Tensor(),
) raises -> Tensor:
    var projected = sparse_condition(
        tokens, prepare_trellis_image(image, size, 1.1), coords, edge, camera, naf, target
    )
    var globals = Tensor.zeros([5, 1024])
    for i in range(globals.numel()):
        globals.f[i] = tokens.f[i]
    var texture = shape.rank() != 0
    var flow = PixalFlow(path, globals^, projected^, coords, shape^)
    return sample_pixal_slat(flow, rng.rows(coords.shape[0]), steps, texture)


def generate_pixal3d(
    image: NativeImage,
    checkpoint_dir: String,
    dino_path: String,
    naf_path: String,
    fov: Float32,
    resolution: Int = 1536,
    seed: UInt32 = 42,
    steps: Int = 12,
    verbose: Bool = False,
) raises -> NativeModel:
    """Generate one static PBR asset from an alpha cutout using the manual-FOV path."""
    if (resolution != 1024 and resolution != 1536) or steps < 1:
        raise Error("Expected a 1024 or 1536 Pixal3D cascade and positive sampling steps")
    var base = checkpoint_dir + "/"
    var camera = PixalCamera.from_fov(fov)
    var rng = TorchNoiseStream(seed)
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var encoder = TrellisImageEncoder(dino_path)
    var naf = SafeTensorWeights(naf_path, 8 << 20)
    if verbose:
        print("Encoding the reference and sampling the sparse structure")
    var tokens = encoder.encode(pool, image, 512, 1.1)
    var coords = _structure(base, tokens, camera, rng, pool, steps)
    return _cascade(
        base, image, camera, encoder, naf, tokens^, coords^, rng, pool, resolution, steps, verbose
    )


def _cascade(
    base: String,
    image: NativeImage,
    camera: PixalCamera,
    encoder: TrellisImageEncoder,
    naf: SafeTensorWeights,
    var tokens: Tensor,
    var coords: Tensor,
    mut rng: TorchNoiseStream,
    mut pool: JobPool,
    resolution: Int,
    steps: Int,
    verbose: Bool = False,
) raises -> NativeModel:
    if verbose:
        print("Sampling the 512 shape at", coords.shape[0], "sites")
    var lr = _sample(
        base + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        image,
        tokens,
        coords,
        camera,
        naf,
        512,
        512,
        32,
        rng,
        steps,
    )
    var shape_decoder = SafeTensorWeights(base + "shape_dec_next_dc_f16c32_fp16.safetensors", 2 << 30, True)
    var low_masks = List[Tensor]()
    var low = decode_slat(shape_decoder, coords^, denormalize_slat(lr^, False), 32, low_masks, False, True)
    var cascade = cascade_coords(low[0], resolution)
    var hr = cascade[0].copy()
    var edge = cascade[1]
    if verbose:
        print("Sampling the", edge * 16, "shape at", hr.shape[0], "sites")
    tokens = encoder.encode(pool, image, 1024, 1.1)
    var shape = _sample(
        base + "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        image,
        tokens,
        hr,
        camera,
        naf,
        1024,
        512,
        edge,
        rng,
        steps,
    )
    if verbose:
        print("Sampling surface materials")
    var tex = _sample(
        base + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        image,
        tokens,
        hr,
        camera,
        naf,
        1024,
        1024,
        edge,
        rng,
        steps,
        shape.copy(),
    )
    var masks = List[Tensor]()
    if verbose:
        print("Decoding the surface and materials")
    var surface = decode_slat(shape_decoder, hr.copy(), denormalize_slat(shape^, False), edge, masks)
    var material = decode_slat(
        SafeTensorWeights(base + "tex_dec_next_dc_f16c32_fp16.safetensors", 2 << 30, True),
        hr^,
        denormalize_slat(tex^, True),
        edge,
        masks,
        True,
    )
    return pixal_surface(surface[0], surface[1], material[1], edge * 16)
