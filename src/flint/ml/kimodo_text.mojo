"""Native inference for the pinned LLM2Vec Llama 3 Q4/128 text encoder used by Kimodo."""
from std.ffi import external_call
from std.math import cos, exp, sin, sqrt
from flint.assets.json import parse_json
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.tensor import fptr_at
from flint.platform.os import close_fd, open_read, read_bytes

# Layout of KitsuMate's model_int4_b128_fp32act_s64.onnx_data, rev 88361596.
comptime _EMBED = 1064960
comptime _LAYERS = 2102411264
comptime _STRIDE = 115867648
comptime _SIZE = _LAYERS + 32 * _STRIDE


def _alpha(b: UInt8) -> Bool:
    return (b >= 65 and b <= 90) or (b >= 97 and b <= 122)


def _digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57


struct KimodoTokenizer(Movable):
    var vocab: Dict[String, Int]
    var merges: Dict[String, Int]

    def __init__(out self, path: String) raises:
        var doc = parse_json(read_bytes(path))
        var model = doc.object_get(doc.root, "model")
        var vocab = doc.object_get(model, "vocab")
        var merges = doc.object_get(model, "merges")
        if not doc.valid or vocab < 0 or merges < 0:
            raise Error("Invalid LLM2Vec tokenizer.json")
        self.vocab = Dict[String, Int]()
        self.merges = Dict[String, Int]()
        for i in range(len(doc.nodes[vocab].keys)):
            self.vocab[doc.nodes[vocab].keys[i]] = doc.integer(
                doc.nodes[vocab].children[i]
            )
        for i in range(doc.array_size(merges)):
            self.merges[doc.text(doc.at(merges, i))] = i

    def encode(
        self, text: String, *, qwen: Bool = False, clip: Bool = False
    ) raises -> List[Int]:
        # English movement cues. Normalise spaces before Llama 3's word splitting.
        var b = List[UInt8]()
        for byte in text.as_bytes():
            if byte >= 127 or (
                byte < 32 and byte != 9 and byte != 10 and byte != 13
            ):
                raise Error("Use a printable English movement prompt")
            if byte <= 32:
                if len(b) and b[len(b) - 1] != 32:
                    b.append(32)
            else:
                b.append(byte)
        if len(b) and b[len(b) - 1] == 32:
            _ = b.pop()
        if not len(b) or len(b) > 1024:
            raise Error(
                "Expected a nonempty movement prompt of at most 1024 bytes"
            )
        var ids = List[Int]()
        if clip:
            ids.append(49406)
            for i in range(len(b)):
                if b[i] >= 65 and b[i] <= 90:
                    b[i] += 32
        elif not qwen:
            ids.append(128000)
        var i = 0
        while i < len(b):
            if clip and b[i] == 32:
                i += 1
                continue
            var start = i
            var contraction = False
            if b[i] == 39:
                for ending in ["re", "ve", "ll", "s", "t", "m", "d"]:
                    var end = i + 1 + ending.byte_length()
                    if (
                        end <= len(b)
                        and String(unsafe_from_utf8=b[i + 1 : end]).lower()
                        == ending
                    ):
                        i = end
                        contraction = True
                        break
            if not contraction:
                if (
                    not clip
                    and not _digit(b[i])
                    and not _alpha(b[i])
                    and i + 1 < len(b)
                    and _alpha(b[i + 1])
                ):
                    i += 1
                if _alpha(b[i]):
                    while i < len(b) and _alpha(b[i]):
                        i += 1
                elif _digit(b[i]):
                    while (
                        i < len(b)
                        and _digit(b[i])
                        and i - start < (1 if qwen or clip else 3)
                    ):
                        i += 1
                else:
                    i += 1
                    while (
                        i < len(b)
                        and b[i] != 32
                        and not _digit(b[i])
                        and not _alpha(b[i])
                    ):
                        i += 1
            var pieces = List[String]()
            for j in range(start, i):
                pieces.append(
                    "Ġ" if b[j] == 32 else String(unsafe_from_utf8=[b[j]])
                )
            if clip:
                pieces[len(pieces) - 1] += "</w>"
            while len(pieces) > 1:
                var best = -1
                var rank = 1 << 30
                for j in range(len(pieces) - 1):
                    var value = self.merges.get(
                        pieces[j] + " " + pieces[j + 1]
                    ).or_else(1 << 30)
                    if value < rank:
                        rank = value
                        best = j
                if best < 0:
                    break
                var merged = pieces[best] + pieces[best + 1]
                pieces[best] = merged^
                _ = pieces.pop(best + 1)
            for piece in pieces:
                var id = self.vocab.get(piece).or_else(-1)
                if id < 0 or id >= (
                    151936 if qwen else 49408 if clip else 128256
                ):
                    raise Error("Invalid text encoder vocabulary")
                ids.append(id)
        if clip:
            ids.append(49407)
        if len(ids) > (30 if qwen else 77 if clip else 64):
            raise Error("Prompt exceeds the text encoder window")
        return ids^


@fieldwise_init
struct _Projection(Copyable, Movable):
    var packed: Int
    var scales: Int
    var x: Int
    var y: Int
    var tokens: Int
    var k: Int
    var n: Int


def _project_rows(addr: Int, job: Int):
    var c = Pointer[_Projection, MutAnyOrigin](unsafe_from_address=addr)
    var weights = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=c[].packed)
    var blocks = c[].k // 128
    for row in range(job * 16, min(c[].n, (job + 1) * 16)):
        for t in range(c[].tokens):
            var total = SIMD[DType.float32, 8](0)
            for h in range(blocks):
                var w = weights.unsafe_offset((row * blocks + h) * 64)
                var x = fptr_at(c[].x).unsafe_offset(t * c[].k + h * 128)
                var dot = SIMD[DType.float32, 8](0)
                for u in range(0, 64, 8):
                    var packed = w.unsafe_load[width=8, alignment=1](u)
                    var lo = (packed & 15).cast[DType.float32]() - 8
                    var hi = (packed >> 4).cast[DType.float32]() - 8
                    dot += lo * x.unsafe_load[width=8, alignment=1](
                        u
                    ) + hi * x.unsafe_load[width=8, alignment=1](64 + u)
                total += (
                    dot * fptr_at(c[].scales)[unsafe_offset=row * blocks + h]
                )
            fptr_at(c[].y)[unsafe_offset=t * c[].n + row] = total.reduce_add()


def _linear(
    mut pool: JobPool,
    base: Int,
    mut offset: Int,
    x: List[Float32],
    tokens: Int,
    k: Int,
    n: Int,
) -> List[Float32]:
    # Deinterleave even and odd activations to match each low/high nibble pair.
    var ordered = List[Float32](length=len(x), fill=0)
    for block in range(len(x) // 128):
        for j in range(64):
            ordered[block * 128 + j] = x[block * 128 + 2 * j]
            ordered[block * 128 + 64 + j] = x[block * 128 + 2 * j + 1]
    var out = List[Float32](length=tokens * n, fill=0)
    var scale = offset + n * k // 2
    var ctx = _Projection(
        base + offset,
        base + scale,
        Int(ordered.unsafe_ptr()),
        Int(out.unsafe_ptr()),
        tokens,
        k,
        n,
    )
    run_parallel_ctx(pool, n // 16, ctx, _project_rows)
    _ = ordered[0]
    offset = scale + (n * k // 128) * 4
    return out^


def _norm(
    x: List[Float32], weight: Int, width: Int = 4096, epsilon: Float32 = 1e-5
) -> List[Float32]:
    var out = x.copy()
    for t in range(len(x) // width):
        var sum = Float32(0)
        for i in range(width):
            var v = x[t * width + i]
            sum += v * v
        var scale = 1 / sqrt(sum / Float32(width) + epsilon)
        for i in range(width):
            out[t * width + i] *= scale * fptr_at(weight)[unsafe_offset=i]
    return out^


def _rope(
    mut x: List[Float32],
    heads: Int,
    tokens: Int,
    theta: Float32 = 500000,
    first: Int = 0,
):
    for t in range(tokens):
        for i in range(64):
            var angle = Float32(t + first) * theta ** (-Float32(i) / 64)
            var cs = cos(angle)
            var sn = sin(angle)
            for h in range(heads):
                var j = (t * heads + h) * 128 + i
                var a = x[j]
                var b = x[j + 64]
                x[j] = a * cs - b * sn
                x[j + 64] = b * cs + a * sn


def _attention(
    q: List[Float32], k: List[Float32], v: List[Float32], tokens: Int
) -> List[Float32]:
    var out = List[Float32](length=tokens * 4096, fill=0)
    var scores = List[Float32](length=tokens, fill=0)
    for t in range(tokens):
        for h in range(32):
            var dst = (t * 32 + h) * 128
            var peak = Float32(-1e30)
            for j in range(tokens):
                var src = (j * 8 + h // 4) * 128
                var dot = Float32(0)
                for d in range(128):
                    dot += q[dst + d] * k[src + d]
                scores[j] = dot / sqrt(Float32(128))
                peak = max(peak, scores[j])
            var denom = Float32(0)
            for j in range(tokens):
                scores[j] = exp(scores[j] - peak)
                denom += scores[j]
            for j in range(tokens):
                var src = (j * 8 + h // 4) * 128
                var w = scores[j] / denom
                for d in range(128):
                    out[dst + d] += w * v[src + d]
    return out^


struct KimodoTextEncoder(Movable):
    var address: Int
    var pool: JobPool

    def __init__(out self, path: String) raises:
        self.address = 0
        self.pool = JobPool()
        var fd = open_read(path)
        if fd < 0:
            raise Error("Missing Kimodo Q4 text weights: " + path)
        var size = Int(
            external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2))
        )
        if size != _SIZE:
            close_fd(fd)
            raise Error("Unexpected size for pinned Kimodo Q4 text weights")
        var addr = external_call["mmap", Int](
            Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0)
        )
        close_fd(fd)
        if addr == -1 or addr == 0:
            raise Error("Could not map Kimodo Q4 weights")
        self.address = addr
        _ = self.pool.start(default_worker_count())

    def __deinit__(deinit self):
        if self.address:
            _ = external_call["munmap", Int32](self.address, Int64(_SIZE))

    def encode(mut self, ids: List[Int]) raises -> List[Float32]:
        var n = len(ids)
        if n < 2 or n > 64 or ids[0] != 128000:
            raise Error("Invalid LLM2Vec token sequence")
        var x = List[Float32](capacity=n * 4096)
        for id in ids:
            if id < 0 or id >= 128256:
                raise Error("Token outside LLM2Vec vocabulary")
            var row = fptr_at(self.address + _EMBED + id * 4096 * 4)
            for i in range(4096):
                x.append(row[unsafe_offset=i])
        for layer in range(32):
            var offset = _LAYERS + layer * _STRIDE
            var norm = _norm(x, self.address + layer * 32768)
            var q = _linear(
                self.pool, self.address, offset, norm, n, 4096, 4096
            )
            var k = _linear(
                self.pool, self.address, offset, norm, n, 4096, 1024
            )
            var v = _linear(
                self.pool, self.address, offset, norm, n, 4096, 1024
            )
            _rope(q, 32, n)
            _rope(k, 8, n)
            var att = _attention(q, k, v, n)
            var proj = _linear(
                self.pool, self.address, offset, att, n, 4096, 4096
            )
            for i in range(len(x)):
                x[i] += proj[i]
            norm = _norm(x, self.address + layer * 32768 + 16384)
            var gate = _linear(
                self.pool, self.address, offset, norm, n, 4096, 14336
            )
            var up = _linear(
                self.pool, self.address, offset, norm, n, 4096, 14336
            )
            for i in range(len(up)):
                up[i] *= gate[i] / (1 + exp(-gate[i]))
            var down = _linear(
                self.pool, self.address, offset, up, n, 14336, 4096
            )
            for i in range(len(x)):
                x[i] += down[i]
        x = _norm(x, self.address + 1048576)
        var pooled = List[Float32](length=4096, fill=0)
        for t in range(1, n):
            for i in range(4096):
                pooled[i] += x[t * 4096 + i] / Float32(n - 1)
        return pooled^
