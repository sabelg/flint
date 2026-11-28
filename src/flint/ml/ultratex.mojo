"""Image-guided UltraTex FLUX.2 inference. Six views in TexVerse's 2-by-3 layout."""
from flint.platform.os import env_now
from std.math import cos, exp, fma, isfinite, log, sin, sqrt
from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu import global_idx
from std.sys._assembly import inlined_assembly
from flint.assets.image import NativeImage
from flint.ml.tensor import DT_F32, Tensor
from flint.ml.ultratex_attention import _precision
from flint.ml.ultratex_dit import UltraTexDiT, _rows
from flint.ml.ultratex_vae import UltraTexVAE


def _filtered(
    pixels: List[UInt32], width: Int, height: Int, size: Int, vertical: Bool
) -> List[UInt32]:
    var old = height if vertical else width
    var scale = Float64(old) / Float64(size)
    var support = max(Float64(1), scale)
    var ow = width if vertical else size
    var oh = size if vertical else height
    var out = List[UInt32](length=ow * oh, fill=0)
    for i in range(size):
        var center = (Float64(i) + 0.5) * scale
        var start = max(0, Int(center - support + 0.5))
        var end = min(old, Int(center + support + 0.5))
        var weights = List[Float64]()
        var total = Float64(0)
        for j in range(start, end):
            var w = max(
                Float64(0), 1 - abs((Float64(j) + 0.5 - center) / support)
            )
            total += w
            weights.append(w)
        for j in range(len(weights)):
            weights[j] = Float64(Int(weights[j] * (1 << 22) / total + 0.5))
        for row in range(width if vertical else height):
            var sums = SIMD[DType.int64, 4](1 << 21)
            for j in range(start, end):
                var pix = pixels[
                    j * width + row if vertical else row * width + j
                ]
                comptime for c in range(4):
                    sums[c] += Int64((pix >> UInt32(8 * c)) & 255) * Int64(
                        weights[j - start]
                    )
            var p = UInt32(0)
            comptime for c in range(4):
                p |= UInt32(min(Int64(255), sums[c] >> 22)) << UInt32(8 * c)
            out[i * width + row if vertical else row * size + i] = p
    return out^


def _resize(
    var pixels: List[UInt32], width: Int, height: Int, size: Int
) -> List[UInt32]:
    if width != size:
        pixels = _filtered(pixels, width, height, size, False)
    if height != size:
        pixels = _filtered(pixels, size, height, size, True)
    return pixels^


def _resize_rgba(
    var pixels: List[UInt32], width: Int, height: Int, side: Int
) -> List[UInt32]:
    if width != side or height != side:
        # PIL resizes RGBA in premultiplied-alpha space.
        for i in range(len(pixels)):
            var p = pixels[i]
            var a = p >> 24
            var v = a << 24
            comptime for c in range(3):
                v |= (
                    (((p >> UInt32(8 * c)) & 255) * a + 127) // 255
                ) << UInt32(8 * c)
            pixels[i] = v
        pixels = _resize(pixels^, width, height, side)
        for i in range(len(pixels)):
            var p = pixels[i]
            var a = p >> 24
            var v = a << 24
            if a > 0:
                comptime for c in range(3):
                    v |= min(
                        UInt32(255), ((p >> UInt32(8 * c)) & 255) * 255 // a
                    ) << UInt32(8 * c)
            pixels[i] = v
    return pixels^


def _mask(pixels: List[UInt32], side: Int, source_side: Int = 0) -> List[Bool]:
    var n = side // 16
    var width = source_side if source_side else side
    var small = _resize(pixels.copy(), width, width, n)
    var keep = List[Bool](length=n * n, fill=False)
    # OpenCV's 2x2 dilation anchors at (1,1), independently in each tile.
    for y in range(n):
        for x in range(n):
            for yy in range(max(0, y - 1), y + 1):
                for xx in range(max(0, x - 1), x + 1):
                    if small[yy * n + xx] >> 24:
                        keep[y * n + x] = True
    return keep^


def _rgb(pix: List[UInt32], side: Int) -> Tensor:
    var out = Tensor.zeros([side, side, 3])
    for i in range(len(pix)):
        var a = Int(pix[i] >> 24)
        comptime for c in range(3):
            var val = Int((pix[i] >> UInt32(8 * (2 - c))) & 255)
            var white = (val * a + 255 * (255 - a) + 127) // 255
            out.f[3 * i + c] = Float32(white) / 127.5 - 1
    return out^


def _tile(atlas: NativeImage, index: Int, side: Int) -> List[UInt32]:
    var out = List[UInt32](capacity=side * side)
    for y in range(side):
        var base = (y + (index // 3) * side) * atlas.width + (index % 3) * side
        for x in range(side):
            out.append(atlas.pixels[base + x])
    return out^


def _index(tile: Int, local: Int, n: Int) -> Int:
    return ((tile // 3) * n + local // n) * 3 * n + (tile % 3) * n + local % n


struct _Foreground(Movable):
    var positions: List[Int]
    var ids: Tensor
    var values: Tensor

    def __init__(
        out self, keep: List[Bool], grid: Tensor, width: Int, t: Int
    ) raises:
        self.positions = List[Int]()
        for i in range(len(keep)):
            if keep[i]:
                self.positions.append(i)
        if len(self.positions) == 0:
            raise Error("UltraTex input has no foreground tokens")
        var n = len(self.positions)
        self.ids = Tensor.zeros([n, 4])
        self.values = Tensor.zeros([n, 128])
        for i in range(n):
            var p = self.positions[i]
            self.ids.f[i * 4] = Float32(t)
            self.ids.f[i * 4 + 1] = Float32(p // width)
            self.ids.f[i * 4 + 2] = Float32(p % width)
            for j in range(128):
                self.values.f[i * 128 + j] = grid.f[p * 128 + j]


def ultratex_time(step: Int, steps: Int, side: Int) raises -> Float32:
    if (
        steps < 1
        or step < 0
        or step > steps
        or side < 16
        or side > 2048
        or side % 16
    ):
        raise Error("Invalid UltraTex schedule")
    var seq = ((3 * side) // 8) * ((2 * side) // 8) // 256
    var mu = 0.5 + (Float64(seq) - 256) * (1.15 - 0.5) / (4096 - 256)
    var u = Float64(steps - step) / Float64(steps)
    return Float32(exp(mu) * u / (1 - u + exp(mu) * u))


def _denoise(
    mut net: UltraTexDiT,
    mut target: _Foreground,
    refs: Tensor,
    ids: Tensor,
    ctx: Tensor,
    positive: Tensor,
    side: Int,
    steps: Int,
    guidance: Float32,
) raises:
    var progress = env_now("FLINT_ULTRATEX_PROGRESS") == "1"
    for step in range(steps):
        if progress:
            print("UltraTex denoising step", step + 1, "of", steps)
        var t = ultratex_time(step, steps, side)
        var dt = ultratex_time(step + 1, steps, side) - t
        var input = _rows(target.values, refs)
        var time = Float32(BFloat16(t))
        var velocity = net.velocity(input, ids, ctx, time)
        if len(positive.f) > 0:
            var pos = net.velocity(input, ids, positive, time)
            for i in range(len(target.values.f)):
                var delta = _precision(pos.f[i] - velocity.f[i], net.bf16)
                velocity.f[i] = _precision(
                    velocity.f[i] + _precision(guidance * delta, net.bf16),
                    net.bf16,
                )
        for i in range(len(target.values.f)):
            target.values.f[i] = Float32(
                BFloat16(
                    target.values.f[i]
                    + _precision(dt * velocity.f[i], net.bf16)
                )
            )


def ultratex_views(
    mut vae: UltraTexVAE,
    mut net: UltraTexDiT,
    normals: NativeImage,
    reference: NativeImage,
    empty_ctx: Tensor,
    positive_ctx: Tensor,
    noise: Tensor,
    steps: Int = 25,
    guidance: Float32 = 4,
    view_size: Int = 0,
) raises -> NativeImage:
    """Generate six views from an RGBA normal atlas, RGBA reference and explicit noise.
    An empty positive tensor selects upstream's image-only, empty-prompt path.
    """
    var source_side = normals.width // 3
    var side = view_size if view_size else source_side
    if (
        not normals.valid
        or normals.width != source_side * 3
        or normals.height != source_side * 2
        or source_side < 1
        or source_side > 2048
        or side < 16
        or side > 2048
        or side % 16
        or len(normals.pixels) != normals.width * normals.height
    ):
        raise Error(
            "Expected a 3x2 atlas of square RGBA normal views (16 to 2048"
            " pixels per view)"
        )
    if (
        not reference.valid
        or min(reference.width, reference.height) < 1
        or reference.width > 40000000 // reference.height
        or len(reference.pixels) != reference.width * reference.height
    ):
        raise Error("Invalid UltraTex reference image")
    var n = side // 16
    var rows = 6 * n * n
    if (
        steps < 1
        or not isfinite(guidance)
        or noise.shape != [rows, 128]
        or noise.dtype != DT_F32
        or len(noise.f) != rows * 128
    ):
        raise Error(
            "Expected finite guidance, positive step count and"
            " [6*(side/16)^2,128] noise"
        )
    var grid = Tensor.zeros([rows, 128])
    var keep = List[Bool](length=rows, fill=False)
    var alpha = List[Bool](length=6 * side * side, fill=False)
    for tile in range(6):
        var pix = _tile(normals, tile, source_side)
        var mask = _mask(pix, side, source_side)
        pix = _resize_rgba(pix^, source_side, source_side, side)
        for i in range(side * side):
            alpha[_index(tile, i, side)] = Bool(pix[i] >> 24)
        var z = vae.encode(_rgb(pix, side))
        for j in range(n * n):
            var row = _index(tile, j, n)
            keep[row] = mask[j]
            for c in range(128):
                grid.f[row * 128 + c] = Float32(BFloat16(z.f[j * 128 + c]))
    var normal = _Foreground(keep, grid, 3 * n, 10)
    var ref_pixels = _resize_rgba(
        reference.pixels.copy(), reference.width, reference.height, side
    )
    var ref_grid = vae.encode(_rgb(ref_pixels, side))
    var photo = _Foreground(_mask(ref_pixels, side), ref_grid, n, 20)
    for i in range(len(photo.values.f)):
        photo.values.f[i] = Float32(BFloat16(photo.values.f[i]))
    var target = _Foreground(keep, noise, 3 * n, 0)
    for i in range(len(target.values.f)):
        target.values.f[i] = Float32(BFloat16(target.values.f[i]))
    var ids = _rows(target.ids, _rows(normal.ids, photo.ids))
    _denoise(
        net,
        target,
        _rows(normal.values, photo.values),
        ids,
        empty_ctx,
        positive_ctx,
        side,
        steps,
        guidance,
    )
    var black = Tensor.zeros([side, side, 3])
    for i in range(len(black.f)):
        black.f[i] = -1
    var background = vae.encode(black)
    for tile in range(6):
        for j in range(n * n):
            var row = _index(tile, j, n)
            for c in range(128):
                grid.f[row * 128 + c] = Float32(
                    BFloat16(background.f[j * 128 + c])
                )
    for i in range(len(target.positions)):
        for c in range(128):
            grid.f[target.positions[i] * 128 + c] = target.values.f[i * 128 + c]
    var out = NativeImage()
    out.valid = True
    out.width = side * 3
    out.height = side * 2
    out.pixels = List[UInt32](length=out.width * out.height, fill=0xFFFFFFFF)
    for tile in range(6):
        var z = Tensor.zeros([n, n, 128])
        for j in range(n * n):
            var row = _index(tile, j, n)
            for c in range(128):
                z.f[j * 128 + c] = grid.f[row * 128 + c]
        var image = vae.decode(z)
        for i in range(side * side):
            var row = _index(tile, i, side)
            if alpha[row]:
                var p = UInt32(0xFF000000)
                comptime for c in range(3):
                    p |= UInt32(
                        max(
                            Float32(0),
                            min(Float32(255), (image.f[3 * i + c] + 1) * 127.5),
                        )
                    ) << UInt32(8 * (2 - c))
                out.pixels[row] = p
    return out^


def _philox(thread: Int, counter: Int, seed: UInt32) -> SIMD[DType.uint32, 4]:
    var x = UInt32(counter)
    var y = UInt32(0)
    var z = UInt32(thread)
    var w = UInt32(0)
    var key0 = seed
    var key1 = UInt32(0)
    for _ in range(10):
        var a = UInt64(x) * 0xD2511F53
        var b = UInt64(z) * 0xCD9E8D57
        x, y = UInt32(b >> 32) ^ y ^ key0, UInt32(b)
        z, w = UInt32(a >> 32) ^ w ^ key1, UInt32(a)
        key0 += 0x9E3779B9
        key1 += 0xBB67AE85
    return SIMD[DType.uint32, 4](x, y, z, w)


def _cuda_noise(
    dest: Pointer[Float32, MutAnyOrigin],
    length: Int32,
    stride: Int32,
    seed: UInt32,
):
    var thread = Int(global_idx.x)
    if thread >= min(Int(stride), Int(length)):
        return
    for start in range(thread, Int(length), 4 * Int(stride)):
        var words = _philox(thread, start // (4 * Int(stride)), seed)
        for pair in range(2):
            var pos = start + 2 * pair * Int(stride)
            if pos >= Int(length):
                break
            comptime unit = Float32(2.3283064365386963e-10)
            comptime circle = Float32(1.4629180792671596e-9)
            var u = fma(Float32(words[pair * 2]), unit, unit * 0.5)
            var phi = fma(Float32(words[pair * 2 + 1]), circle, circle * 0.5)
            var radius = inlined_assembly[
                "sqrt.rn.f32 $0, $1;",
                Float32,
                constraints="=f,f",
                has_side_effect=False,
            ](-2 * Float32(log(Float64(u))))
            var sn = inlined_assembly[
                "sin.approx.f32 $0, $1;",
                Float32,
                constraints="=f,f",
                has_side_effect=False,
            ](phi)
            dest.unsafe_store(pos, Float32(BFloat16(sn * radius)))
            if pos + Int(stride) < Int(length):
                var cs = inlined_assembly[
                    "cos.approx.f32 $0, $1;",
                    Float32,
                    constraints="=f,f",
                    has_side_effect=False,
                ](phi)
                dest.unsafe_store(
                    pos + Int(stride), Float32(BFloat16(cs * radius))
                )


def ultratex_noise(
    side: Int, seed: UInt32, launch_threads: Int = 0
) raises -> Tensor:
    """Reproduce the CUDA generator's Philox subsequences and BF16 normal layout.
    """
    if (
        side < 16
        or side > 2048
        or side % 16
        or launch_threads < 0
        or launch_threads % 256
    ):
        raise Error(
            "Expected a supported tile resolution and a multiple of 256 launch"
            " threads"
        )
    var n = 6 * (side // 16) * (side // 16)
    var count = n * 128
    var threads = launch_threads
    var result = Tensor.zeros([n, 128])
    if env_now("FLINT_ULTRATEX_GPU") == "1":
        var ctx = DeviceContext()
        if ctx.target.is_nvidia_gpu():
            if threads == 0:
                var sms = ctx.get_attribute(
                    DeviceAttribute.MULTIPROCESSOR_COUNT
                )
                var resident = ctx.get_attribute(
                    DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR
                )
                threads = sms * (resident // 256) * 256
            threads = min(threads, ((count + 255) // 256) * 256)
            var buf = ctx.enqueue_create_buffer[DType.float32](count)
            ctx.enqueue_function[_cuda_noise](
                buf.unsafe_ptr(),
                Int32(count),
                Int32(threads),
                seed,
                grid_dim=threads // 256,
                block_dim=256,
            )
            with buf.map_to_host() as host:
                for i in range(count):
                    result.f[i] = host[i]
            return result^
    if threads == 0:
        # CPU-only runs use the launch geometry of the upstream H100.
        threads = 132 * 2048
    threads = min(threads, ((count + 255) // 256) * 256)
    comptime unit = Float32(2.3283064365386963e-10)
    comptime circle = Float32(1.4629180792671596e-9)
    for start in range(0, count, 4 * threads):
        for thread in range(min(threads, count - start)):
            var words = _philox(thread, start // (4 * threads), seed)
            for pair in range(2):
                var i = start + thread + 2 * pair * threads
                if i >= count:
                    break
                var u = fma(Float32(words[pair * 2]), unit, unit * 0.5)
                var angle = fma(
                    Float32(words[pair * 2 + 1]), circle, circle * 0.5
                )
                var radius = sqrt(-2 * log(u))
                result.f[i] = Float32(BFloat16(sin(angle) * radius))
                if i + threads < count:
                    result.f[i + threads] = Float32(
                        BFloat16(cos(angle) * radius)
                    )
    return result^
