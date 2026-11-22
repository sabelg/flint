"""Native Qwen3-8B and CLIP-L conditioning for the HY-Motion model."""
from std.ffi import external_call
from std.math import exp, sqrt
from flint.ml.kimodo_text import KimodoTokenizer, _norm, _rope
from flint.ml.tensor import Tensor, DT_I64, fptr_at
from flint.ml.onnx import OnnxModel
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool, default_worker_count
from flint.platform.os import open_read, close_fd, env_now

comptime PREFIX = 101
comptime HIDDEN = 4096
comptime EMBED_BYTES = 151936 * HIDDEN * 2
comptime LAYER_BYTES = (4096 * 2 + 128 * 2) * 4 + 2 * PREFIX * 1024 * 4 + 2 * (
    4096 * 4096 * 2 + 1024 * 4096 * 2 + 4096 * 12288 * 3
)
comptime FILE_SIZE = 4 + EMBED_BYTES + 36 * LAYER_BYTES + 4096 * 4


@fieldwise_init
struct DenseRows(Copyable, Movable):
    var weight: Int
    var inputs: Int
    var output: Int
    var tokens: Int
    var width: Int
    var rows: Int


def project_rows(addr: Int, job: Int):
    var ctx = Pointer[DenseRows, MutAnyOrigin](unsafe_from_address=addr)
    var weights = Pointer[BFloat16, MutAnyOrigin](
        unsafe_from_address=ctx[].weight
    )
    for row in range(job * 16, (job + 1) * 16):
        for token in range(ctx[].tokens):
            var total = SIMD[DType.float32, 8](0)
            var w = weights.unsafe_offset(row * ctx[].width)
            var a = fptr_at(ctx[].inputs).unsafe_offset(token * ctx[].width)
            for i in range(0, ctx[].width, 8):
                total += w.unsafe_load[width=8, alignment=1](i).cast[
                    DType.float32
                ]() * a.unsafe_load[width=8, alignment=1](i)
            fptr_at(ctx[].output)[
                unsafe_offset=token * ctx[].rows + row
            ] = total.reduce_add()


def dense(
    mut pool: JobPool,
    mut address: Int,
    x: List[Float32],
    tokens: Int,
    width: Int,
    rows: Int,
) -> List[Float32]:
    var out = List[Float32](length=tokens * rows, fill=0)
    var ctx = DenseRows(
        address, Int(x.unsafe_ptr()), Int(out.unsafe_ptr()), tokens, width, rows
    )
    run_parallel_ctx(pool, rows // 16, ctx, project_rows)
    _ = x[0]
    address += width * rows * 2
    return out^


def causal(
    q: List[Float32],
    k: List[Float32],
    v: List[Float32],
    pk: Int,
    pv: Int,
    n: Int,
) -> List[Float32]:
    var out = List[Float32](length=n * 4096, fill=0)
    var scores = List[Float32](length=PREFIX + n, fill=0)
    for t in range(n):
        var end = PREFIX + t + 1
        for h in range(32):
            var dst = (t * 32 + h) * 128
            var at = (h // 4) * 128
            var peak: Float32 = -1e30
            for j in range(end):
                var p = fptr_at(pk).unsafe_offset(
                    j * 1024 + at
                ) if j < PREFIX else k.unsafe_ptr().unsafe_offset(
                    (j - PREFIX) * 1024 + at
                )
                var dot = SIMD[DType.float32, 8](0)
                for d in range(0, 128, 8):
                    dot += q.unsafe_ptr().unsafe_load[width=8, alignment=1](
                        dst + d
                    ) * p.unsafe_load[width=8, alignment=1](d)
                scores[j] = dot.reduce_add() * 0.08838834764831845
                peak = max(peak, scores[j])
            var denom: Float32 = 0
            for j in range(end):
                scores[j] = exp(scores[j] - peak)
                denom += scores[j]
            for j in range(end):
                var p = fptr_at(pv).unsafe_offset(
                    j * 1024 + at
                ) if j < PREFIX else v.unsafe_ptr().unsafe_offset(
                    (j - PREFIX) * 1024 + at
                )
                var w = scores[j] / denom
                for d in range(128):
                    out[dst + d] += p[unsafe_offset=d] * w
    return out^


struct Qwen(Movable):
    var addr: Int
    var pool: JobPool

    def __init__(out self, path: String) raises:
        self.addr = 0
        self.pool = JobPool()
        var fd = open_read(path)
        if fd < 0:
            raise Error("Missing packed Qwen3 weights at " + path)
        var size = external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2))
        if size != FILE_SIZE:
            close_fd(fd)
            raise Error("Unexpected packed Qwen3 weights size")
        var addr = external_call["mmap", Int](
            Int(0), size, Int32(1), Int32(2), Int32(fd), Int64(0)
        )
        close_fd(fd)
        if addr <= 0:
            raise Error("Unable to map Qwen3 weights")
        self.addr = addr
        if (
            Pointer[UInt32, MutAnyOrigin](unsafe_from_address=addr)[]
            != 0x31515948
        ):
            raise Error("Invalid HYQ1 header")
        _ = self.pool.start(default_worker_count())

    def __deinit__(deinit self):
        if self.addr:
            _ = external_call["munmap", Int32](self.addr, Int64(FILE_SIZE))

    def encode(mut self, ids: List[Int]) raises -> List[Float32]:
        var n = len(ids)
        var x = List[Float32](capacity=n * 4096)
        for id in ids:
            if id < 0 or id >= 151936:
                raise Error("Invalid Qwen token")
            var row = Pointer[BFloat16, MutAnyOrigin](
                unsafe_from_address=self.addr + 4 + id * 8192
            )
            for i in range(4096):
                x.append(Float32(row[unsafe_offset=i]))
        var off = self.addr + 4 + EMBED_BYTES
        for _ in range(36):
            var norm1 = off
            var norm2 = norm1 + 4096 * 4
            var qnorm = norm2 + 4096 * 4
            var knorm = qnorm + 128 * 4
            var pk = knorm + 128 * 4
            var pv = pk + PREFIX * 1024 * 4
            off = pv + PREFIX * 1024 * 4
            var z = _norm(x, norm1, 4096, 1e-6)
            var q = _norm(
                dense(self.pool, off, z, n, 4096, 4096), qnorm, 128, 1e-6
            )
            var k = _norm(
                dense(self.pool, off, z, n, 4096, 1024), knorm, 128, 1e-6
            )
            var v = dense(self.pool, off, z, n, 4096, 1024)
            _rope(q, 32, n, 1000000, PREFIX)
            _rope(k, 8, n, 1000000, PREFIX)
            var att = causal(q, k, v, pk, pv, n)
            var proj = dense(self.pool, off, att, n, 4096, 4096)
            for i in range(len(x)):
                x[i] += proj[i]
            z = _norm(x, norm2, 4096, 1e-6)
            var gate = dense(self.pool, off, z, n, 4096, 12288)
            var up = dense(self.pool, off, z, n, 4096, 12288)
            for j in range(len(up)):
                up[j] *= gate[j] / (1 + exp(-gate[j]))
            var down = dense(self.pool, off, up, n, 12288, 4096)
            for j in range(len(x)):
                x[j] += down[j]
        return _norm(x, off, 4096, 1e-6)


def encode_hy(
    prompt: String, dir: String
) raises -> Tuple[List[Float32], List[Float32]]:
    var tokenizer = KimodoTokenizer(dir + "/qwen-tokens.json")
    var ids = tokenizer.encode(prompt, qwen=True)
    ids.extend([151645, 198])
    var model = Qwen(dir + "/qwen.bf16")
    var context = model.encode(ids)
    var ct = KimodoTokenizer(dir + "/clip-tokens.json")
    var ci = ct.encode(prompt, clip=True)
    var end = len(ci) - 1
    while len(ci) < 77:
        ci.append(49407)
    var clip = OnnxModel.load(dir + "/clip-text.onnx", 4)
    if env_now("FLINT_HYMOTION_CPU") != "1":
        _ = clip.use_gpu()
    clip.set_input("ids", Tensor.ints(DT_I64, [1, 77], ci^))
    clip.run()
    var states = clip.output("states")
    if states.shape != [1, 77, 768]:
        raise Error("Invalid CLIP text output")
    var pooled = List[Float32]()
    for i in range(768):
        pooled.append(states.fget(end * 768 + i))
    return (pooled^, context^)
