"""FLUX.2 Klein two-stream and single-stream denoiser with UltraTex adapters."""
from std.collections import Set
from std.math import cos, exp, pow, sin, sqrt
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor, DT_F32
from flint.ml.ultratex_attention import _selected_attention, _precision
from flint.ml.ops import matmul_2d
from flint.ml.vision_ops import weighted_linear
from flint.ml.ultratex_gpu import UltraTexGpu
from flint.ml.ultratex_gpu_attention import (
    gpu_sparse_attention,
    gpu_sparse_buffer,
)
from flint.ml.ultratex_gpu_blocks import DeviceRows, device_floats
from max.gpu.host import DeviceBuffer


def _rows(a: Tensor, b: Tensor) -> Tensor:
    var out = Tensor.zeros([a.shape[0] + b.shape[0], a.shape[1]])
    for i in range(len(a.f)):
        out.f[i] = a.f[i]
    for i in range(len(b.f)):
        out.f[len(a.f) + i] = b.f[i]
    return out^


def _slice(x: Tensor, start: Int, n: Int) -> Tensor:
    var c = x.shape[1]
    var out = Tensor.zeros([n, c])
    for i in range(n * c):
        out.f[i] = x.f[start * c + i]
    return out^


def _cols(x: Tensor, start: Int, n: Int) -> Tensor:
    var out = Tensor.zeros([x.shape[0], n])
    for i in range(x.shape[0]):
        for j in range(n):
            out.f[i * n + j] = x.f[i * x.shape[1] + start + j]
    return out^


def _silu(mut x: Tensor, bf16: Bool):
    for i in range(len(x.f)):
        var a = x.f[i]
        x.f[i] = _precision(a / (1 + exp(-a)), bf16)


def _modulate(x: Tensor, m: Tensor, chunk: Int, bf16: Bool) -> Tensor:
    var c = x.shape[1]
    var out = Tensor.zeros(x.shape.copy())
    for i in range(x.shape[0]):
        var base = i * c
        var mean = Float32(0)
        for j in range(c):
            mean += x.f[base + j]
        mean /= Float32(c)
        var varsum = Float32(0)
        for j in range(c):
            var v = x.f[base + j] - mean
            varsum += v * v
        var inv = 1 / sqrt(varsum / Float32(c) + 1e-6)
        for j in range(c):
            var norm = _precision((x.f[base + j] - mean) * inv, bf16)
            var scale = _precision(1 + m.f[(chunk + 1) * c + j], bf16)
            out.f[base + j] = _precision(
                _precision(norm * scale, bf16) + m.f[chunk * c + j], bf16
            )
    return out^


def _residual(mut x: Tensor, y: Tensor, m: Tensor, gate: Int, bf16: Bool):
    var c = x.shape[1]
    for i in range(len(x.f)):
        x.f[i] = _precision(
            x.f[i] + _precision(m.f[gate * c + i % c] * y.f[i], bf16), bf16
        )


def _swiglu(x: Tensor, offset: Int, bf16: Bool) -> Tensor:
    var c = (x.shape[1] - offset) // 2
    var out = Tensor.zeros([x.shape[0], c])
    for i in range(x.shape[0]):
        for j in range(c):
            var p = i * x.shape[1] + offset + j
            var a = x.f[p]
            out.f[i * c + j] = _precision(
                _precision(a / (1 + exp(-a)), bf16) * x.f[p + c], bf16
            )
    return out^


def _angles(ids: Tensor, head_dim: Int) -> Tensor:
    var axis_dim = head_dim // 4
    var out = Tensor.zeros([ids.shape[0], head_dim])
    var frequencies = List[Float32](capacity=axis_dim // 2)
    for j in range(0, axis_dim, 2):
        frequencies.append(
            1 / Float32(pow(Float64(2000), Float64(j) / Float64(axis_dim)))
        )
    for i in range(ids.shape[0]):
        for axis in range(4):
            for j in range(0, axis_dim, 2):
                var theta = ids.f[4 * i + axis] * frequencies[j // 2]
                var off = i * head_dim + axis * axis_dim + j
                out.f[off] = cos(theta)
                out.f[off + 1] = sin(theta)
    return out^


def _prepare(
    mut packed: Tensor,
    angles: Tensor,
    w: SafeTensorWeights,
    key: String,
    heads: Int,
    bf16: Bool,
) raises:
    var d = angles.shape[1]
    var width = heads * d
    var q = w.tensor(key + ".query_norm.scale", [d])
    var k = w.tensor(key + ".key_norm.scale", [d])
    for r in range(packed.shape[0]):
        for part in range(2):
            for head in range(heads):
                var off = r * packed.shape[1] + part * width + head * d
                var ss = Float32(0)
                for j in range(d):
                    ss += packed.f[off + j] * packed.f[off + j]
                var s = 1 / sqrt(ss / Float32(d) + 1e-6)
                for j in range(0, d, 2):
                    var a = _precision(
                        _precision(packed.f[off + j] * s, bf16)
                        * (k.f[j] if part else q.f[j]),
                        bf16,
                    )
                    var b = _precision(
                        _precision(packed.f[off + j + 1] * s, bf16)
                        * (k.f[j + 1] if part else q.f[j + 1]),
                        bf16,
                    )
                    var cs = angles.f[r * d + j]
                    var sn = angles.f[r * d + j + 1]
                    packed.f[off + j] = _precision(cs * a - sn * b, bf16)
                    packed.f[off + j + 1] = _precision(sn * a + cs * b, bf16)


@fieldwise_init
struct _HeadWork(Movable):
    var packed: Int
    var output: Int
    var rows: Int
    var width: Int
    var heads: Int
    var lanes: Int
    var keep: Int
    var bf16: Bool


def _head_worker(address: Int, lane: Int):
    ref spec = Pointer[_HeadWork, ImmutAnyOrigin](unsafe_from_address=address)[]
    var src = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=spec.packed)
    var dst = Pointer[Float32, MutAnyOrigin](unsafe_from_address=spec.output)
    var d = spec.width // spec.heads
    var n = spec.rows
    var local = JobPool()
    var q = Tensor.zeros([n, d])
    var k = Tensor.zeros([n, d])
    var v = Tensor.zeros([n, d])
    for head in range(lane, spec.heads, spec.lanes):
        for r in range(n):
            var off = r * 3 * spec.width + head * d
            for j in range(d):
                q.f[r * d + j] = src.unsafe_load(off + j)
                k.f[r * d + j] = src.unsafe_load(off + spec.width + j)
                v.f[r * d + j] = src.unsafe_load(off + 2 * spec.width + j)
        var att = _selected_attention(
            local, q, k, v, min(128, n), min(64, n), spec.keep, spec.bf16
        )
        for r in range(n):
            for j in range(d):
                dst.unsafe_store(
                    r * spec.width + head * d + j, att.f[r * d + j]
                )


struct UltraTexDiT(Movable):
    var base: SafeTensorWeights
    var adapter: SafeTensorWeights
    var pool: JobPool
    var width: Int
    var heads: Int
    var double_depth: Int
    var single_depth: Int
    var context_width: Int
    var rank: Int
    var gpu: List[UltraTexGpu]
    var qk_gains: List[DeviceRows]
    var qk_ids: Dict[String, Int]
    var bf16: Bool
    var merged: Set[String]
    var ratio: Float32

    def __init__(
        out self,
        base_path: String,
        adapter_path: String,
        width: Int = 4096,
        heads: Int = 32,
        double_depth: Int = 8,
        single_depth: Int = 24,
        context_width: Int = 12288,
        rank: Int = 64,
        ratio: Float32 = 0.2,
        use_gpu: Bool = False,
    ) raises:
        if (
            min(width, heads) < 1
            or width % (8 * heads) != 0
            or min(double_depth, single_depth) < 1
            or min(context_width, rank) < 1
            or not (ratio > 0 and ratio <= 1)
        ):
            raise Error("Invalid UltraTex denoiser dimensions")
        self.base = SafeTensorWeights(base_path, 32 << 30, private_mapping=True)
        self.adapter = SafeTensorWeights(adapter_path, 2 << 30)
        self.merged = Set[String]()
        var _, dtype = self.base.storage("img_in.weight", [width, 128])
        self.bf16 = dtype == "BF16"
        self.width = width
        self.heads = heads
        self.double_depth = double_depth
        self.single_depth = single_depth
        self.context_width = context_width
        self.gpu = List[UltraTexGpu]()
        self.qk_gains = List[DeviceRows]()
        self.qk_ids = Dict[String, Int]()
        if use_gpu:
            self.gpu.append(UltraTexGpu())
            var shared_ctx = self.gpu[0].ctx
            self.gpu.append(UltraTexGpu(shared_ctx))
        self.rank = rank
        self.ratio = ratio
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())

    def load_adapter(mut self, path: String) raises:
        """Switch material tasks while retaining the resident FLUX.2 backbone.
        """
        var next = SafeTensorWeights(path, 2 << 30)
        if len(self.gpu):
            var shared_ctx = self.gpu[0].ctx
            self.gpu[1] = UltraTexGpu(shared_ctx)
        if len(self.merged):
            self.base.discard_private_changes()
            self.merged.clear()
        self.adapter = next^

    def _merge(mut self, key: String, co: Int, k: Int) raises:
        # PEFT safe_merge adds the rank update to the original BF16 matrix.
        # Only the private in-memory archive changes. Tile B @ A to limit scratch.
        if key in self.merged:
            return
        var offset, dtype = self.base.storage(key + ".weight", [co, k])
        if dtype != "BF16":
            raise Error("Expected BF16 base matrix for " + key)
        var prefix = "base_model.model." + key + ".lora_"
        var a = self.adapter.tensor(prefix + "A.default.weight", [self.rank, k])
        var b = self.adapter.tensor(
            prefix + "B.default.weight", [co, self.rank]
        )
        var tile = Tensor.zeros([min(256, co), k])
        var ptr = Pointer[BFloat16, MutAnyOrigin](
            unsafe_from_address=self.base.address() + offset
        )
        for first in range(0, co, 256):
            var nr = min(256, co - first)
            matmul_2d(
                self.pool,
                b.fptr() + first * self.rank * 4,
                a.fptr(),
                tile.fptr(),
                0,
                nr,
                self.rank,
                k,
                False,
            )
            for i in range(nr * k):
                var orig = Float32(ptr.unsafe_load[alignment=1](first * k + i))
                ptr.unsafe_store[alignment=1](
                    first * k + i, BFloat16(orig + Float32(BFloat16(tile.f[i])))
                )
        _ = a.fptr()
        _ = b.fptr()
        self.merged.add(key)

    def _linear(
        mut self, x: Tensor, key: String, co: Int, adapt: Bool = False
    ) raises -> Tensor:
        if self.bf16 and adapt:
            self._merge(key, co, x.shape[1])
        var out = self.gpu[Int(adapt)].linear(self.base, x, key, co) if len(
            self.gpu
        ) else weighted_linear(self.base, self.pool, x, key, co, False)
        if self.bf16:
            for i in range(len(out.f)):
                out.f[i] = Float32(BFloat16(out.f[i]))
        elif adapt:
            var prefix = "base_model.model." + key + ".lora_"
            var down = weighted_linear(
                self.adapter,
                self.pool,
                x,
                prefix + "A.default",
                self.rank,
                False,
            )
            var delta = weighted_linear(
                self.adapter, self.pool, down, prefix + "B.default", co, False
            )
            for i in range(len(out.f)):
                out.f[i] += delta.f[i]
        return out^

    def _attention(mut self, qkv: Tensor) raises -> Tensor:
        var n = qkv.shape[0]
        var d = self.width // self.heads
        var out = Tensor.zeros([n, self.width])
        var keep = Int(self.ratio * Float32((n + 63) // 64))
        if keep < 1:
            raise Error("Sparse attention selected zero key blocks")
        if (
            self.bf16
            and d == 128
            and len(self.gpu)
            and self.gpu[0].ctx.target.is_nvidia_gpu()
        ):
            return gpu_sparse_attention(self.gpu[0].ctx, qkv, self.heads, keep)
        # Parallelize independent heads. Bound simultaneous Q/K/V scratch at
        # high resolution and avoid nested worker dispatch within each head.
        var lanes = min(self.heads, max(1, (128 << 20) // (4 * 4 * n * d)))
        var spec = _HeadWork(
            qkv.fptr(),
            out.fptr(),
            n,
            self.width,
            self.heads,
            lanes,
            keep,
            self.bf16,
        )
        if self.pool.ready():
            run_parallel_ctx(self.pool, lanes, spec, _head_worker)
        else:
            for i in range(lanes):
                _head_worker(Int(Pointer(to=spec)), i)
        _ = qkv.fptr()
        return out^

    def _double_res(
        mut self, mut x: Tensor, y: Tensor, m: Tensor, key: String
    ) raises:
        _residual(
            x,
            self._linear(y, key + "_attn.proj", self.width, True),
            m,
            2,
            self.bf16,
        )
        var h = self._linear(
            _modulate(x, m, 3, self.bf16), key + "_mlp.0", 6 * self.width
        )
        _residual(
            x,
            self._linear(_swiglu(h, 0, self.bf16), key + "_mlp.2", self.width),
            m,
            5,
            self.bf16,
        )

    def _device_linear(
        mut self, x: DeviceRows, key: String, co: Int, adapt: Bool = False
    ) raises -> DeviceRows:
        var bank = Int(adapt)
        var weight = (
            self.gpu[bank]
            .adapted_weight_id(
                self.base, self.adapter, key, co, x.cols, self.rank
            ) if adapt else self.gpu[bank]
            .weight_id(self.base, key, co, x.cols)
        )
        return x.project(self.gpu[bank], weight, co)

    def _device_rotate(
        mut self,
        mut x: DeviceRows,
        rot: DeviceBuffer[DType.float32],
        key: String,
    ) raises:
        var id = self.qk_ids.get(key, -1)
        if id < 0:
            var q = self.base.tensor(key + ".query_norm.scale", [128])
            var k = self.base.tensor(key + ".key_norm.scale", [128])
            var gains = Tensor.zeros([2, 128])
            for c in range(128):
                gains.f[c] = q.f[c]
                gains.f[c + 128] = k.f[c]
            id = len(self.qk_gains)
            self.qk_ids[key] = id
            self.qk_gains.append(DeviceRows.upload(self.gpu[0].ctx, gains))
        x.rotate(self.gpu[0].ctx, rot, self.qk_gains[id], self.heads)

    def _device_attention(self, x: DeviceRows) raises -> DeviceRows:
        var keep = Int(self.ratio * Float32((x.rows + 63) // 64))
        if keep < 1:
            raise Error("Sparse attention selected zero key blocks")
        return DeviceRows(
            gpu_sparse_buffer(self.gpu[0].ctx, x.buf, x.rows, self.heads, keep),
            x.rows,
            self.width,
        )

    def _device_double_res(
        mut self, mut x: DeviceRows, y: DeviceRows, m: DeviceRows, key: String
    ) raises:
        var device = self.gpu[0].ctx
        x.residual(
            device,
            self._device_linear(y, key + "_attn.proj", self.width, True),
            m,
            2,
        )
        var h = self._device_linear(
            x.norm(device, m, 3), key + "_mlp.0", 6 * self.width
        )
        x.residual(
            device,
            self._device_linear(h.glu(device, 0), key + "_mlp.2", self.width),
            m,
            5,
        )

    def _device_velocity(
        mut self,
        image: Tensor,
        ctx: Tensor,
        imrot: Tensor,
        txrot: Tensor,
        allrot: Tensor,
        mi: Tensor,
        mt: Tensor,
        ms: Tensor,
        final_mod: Tensor,
    ) raises -> Tensor:
        var device = self.gpu[0].ctx
        var image_rot = device_floats(device, imrot)
        var text_rot = device_floats(device, txrot)
        var joint_rot = device_floats(device, allrot)
        var im = DeviceRows.upload(device, mi)
        var tm = DeviceRows.upload(device, mt)
        var sm = DeviceRows.upload(device, ms)
        var final = DeviceRows.upload(device, final_mod)
        var img = self._device_linear(
            DeviceRows.upload(device, image), "img_in", self.width
        )
        var txt = self._device_linear(
            DeviceRows.upload(device, ctx), "txt_in", self.width
        )
        for i in range(self.double_depth):
            var key = "double_blocks." + String(i) + "."
            var qi = self._device_linear(
                img.norm(device, im, 0),
                key + "img_attn.qkv",
                3 * self.width,
                True,
            )
            var qt = self._device_linear(
                txt.norm(device, tm, 0),
                key + "txt_attn.qkv",
                3 * self.width,
                True,
            )
            self._device_rotate(qi, image_rot, key + "img_attn.norm")
            self._device_rotate(qt, text_rot, key + "txt_attn.norm")
            var att = self._device_attention(DeviceRows.join(device, qt, qi))
            self._device_double_res(
                img, att.slice(txt.rows, img.rows), im, key + "img"
            )
            self._device_double_res(
                txt, att.slice(0, txt.rows), tm, key + "txt"
            )
        var both = DeviceRows.join(device, txt, img)
        for i in range(self.single_depth):
            var key = "single_blocks." + String(i) + "."
            var h = self._device_linear(
                both.norm(device, sm, 0), key + "linear1", 9 * self.width, True
            )
            var qkv = DeviceRows.empty(device, both.rows, 3 * self.width)
            h.copy_columns(device, qkv, qkv.cols)
            var mlp = h.glu(device, 3 * self.width)
            self._device_rotate(qkv, joint_rot, key + "norm")
            var att = self._device_attention(qkv)
            var joint = DeviceRows.empty(device, both.rows, 4 * self.width)
            att.copy_columns(device, joint, self.width)
            mlp.copy_columns(device, joint, 3 * self.width, self.width)
            both.residual(
                device,
                self._device_linear(joint, key + "linear2", self.width, True),
                sm,
                2,
            )
        var last = both.slice(ctx.shape[0], image.shape[0]).norm(
            device, final, 0
        )
        return self._device_linear(last, "final_layer.linear", 128).host()

    def velocity(
        mut self, image: Tensor, image_ids: Tensor, ctx: Tensor, time: Float32
    ) raises -> Tensor:
        """Predict velocity for target and reference image tokens at one model time.
        """
        if (
            len(image.shape) != 2
            or image.shape[1] != 128
            or len(ctx.shape) != 2
            or ctx.shape[1] != self.context_width
            or image.shape[0] < 1
            or ctx.shape[0] < 1
            or image_ids.shape != [image.shape[0], 4]
        ):
            raise Error(
                "Expected [N,128] image tokens, [N,4] positions and compatible"
                " text tokens"
            )
        if (
            image.dtype != DT_F32
            or ctx.dtype != DT_F32
            or image_ids.dtype != DT_F32
            or len(image.f) != image.numel()
            or len(ctx.f) != ctx.numel()
            or len(image_ids.f) != image_ids.numel()
        ):
            raise Error("Invalid denoiser tensor storage")
        var d = self.width // self.heads
        var txt_ids = Tensor.zeros([ctx.shape[0], 4])
        for i in range(ctx.shape[0]):
            txt_ids.f[4 * i + 3] = Float32(i)
        var txrot = _angles(txt_ids, d)
        var imrot = _angles(image_ids, d)
        var allrot = _rows(txrot, imrot)
        var time_emb = Tensor.zeros([1, 256])
        for j in range(128):
            var phase = _precision(time * 1000, self.bf16) * exp(
                -Float32(9.210340372) * Float32(j) / 128
            )
            time_emb.f[j] = _precision(cos(phase), self.bf16)
            time_emb.f[j + 128] = _precision(sin(phase), self.bf16)
        var vec = self._linear(time_emb, "time_in.in_layer", self.width)
        _silu(vec, self.bf16)
        vec = self._linear(vec, "time_in.out_layer", self.width)
        _silu(vec, self.bf16)
        var mi = self._linear(
            vec, "double_stream_modulation_img.lin", 6 * self.width
        )
        var mt = self._linear(
            vec, "double_stream_modulation_txt.lin", 6 * self.width
        )
        var ms = self._linear(
            vec, "single_stream_modulation.lin", 3 * self.width
        )
        var final_mod = self._linear(
            vec, "final_layer.adaLN_modulation.1", 2 * self.width
        )
        if (
            self.bf16
            and d == 128
            and len(self.gpu)
            and self.gpu[0].ctx.target.is_nvidia_gpu()
            and self.context_width % 32 == 0
        ):
            return self._device_velocity(
                image, ctx, imrot, txrot, allrot, mi, mt, ms, final_mod
            )
        var img = self._linear(image, "img_in", self.width)
        var txt = self._linear(ctx, "txt_in", self.width)
        for i in range(self.double_depth):
            var key = "double_blocks." + String(i) + "."
            var qi = self._linear(
                _modulate(img, mi, 0, self.bf16),
                key + "img_attn.qkv",
                3 * self.width,
                True,
            )
            var qt = self._linear(
                _modulate(txt, mt, 0, self.bf16),
                key + "txt_attn.qkv",
                3 * self.width,
                True,
            )
            _prepare(
                qi,
                imrot,
                self.base,
                key + "img_attn.norm",
                self.heads,
                self.bf16,
            )
            _prepare(
                qt,
                txrot,
                self.base,
                key + "txt_attn.norm",
                self.heads,
                self.bf16,
            )
            var att = self._attention(_rows(qt, qi))
            self._double_res(
                img, _slice(att, txt.shape[0], img.shape[0]), mi, key + "img"
            )
            self._double_res(txt, _slice(att, 0, txt.shape[0]), mt, key + "txt")
        var both = _rows(txt, img)
        for i in range(self.single_depth):
            var key = "single_blocks." + String(i) + "."
            var h = self._linear(
                _modulate(both, ms, 0, self.bf16),
                key + "linear1",
                9 * self.width,
                True,
            )
            var qkv = _cols(h, 0, 3 * self.width)
            _prepare(
                qkv, allrot, self.base, key + "norm", self.heads, self.bf16
            )
            var a = self._attention(qkv)
            var mlp = _swiglu(h, 3 * self.width, self.bf16)
            var joint = Tensor.zeros([both.shape[0], 4 * self.width])
            for r in range(joint.shape[0]):
                for j in range(joint.shape[1]):
                    joint.f[r * joint.shape[1] + j] = (
                        a.f[r * self.width + j] if j
                        < self.width else mlp.f[
                            r * 3 * self.width + j - self.width
                        ]
                    )
            _residual(
                both,
                self._linear(joint, key + "linear2", self.width, True),
                ms,
                2,
                self.bf16,
            )
        return self._linear(
            _modulate(
                _slice(both, ctx.shape[0], image.shape[0]),
                final_mod,
                0,
                self.bf16,
            ),
            "final_layer.linear",
            128,
        )
