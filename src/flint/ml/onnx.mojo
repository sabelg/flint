"""Load and run ONNX models in pure Mojo.

`OnnxModel.load(path)` memory-maps the file, parses the protobuf graph and keeps every float
initializer as a zero-copy view into the mapping. `run()` interprets the nodes in their
(topological) file order on the CPU, freeing each intermediate after its last use.

Supported: opset 13-17 graphs without control flow, using float32/int32/int64/bool tensors and
the operators listed in `op_code`. Unsupported operators fail at load time, not mid-run.
"""

from std.collections import Dict
from std.math import isfinite
from std.time import perf_counter_ns
from std.ffi import external_call
from flint.core.jobs import JobPool, default_worker_count
from flint.platform.os import open_read, close_fd, write_bytes
from std.collections import Optional
from flint.ml.protobuf import PbReader, WIRE_LEN
from flint.ml.gpu import GpuMatmul, FP, gpu_binary_supported, gpu_unary_supported, GU_BF16
from flint.ml.tensor import Tensor, DT_F32, DT_I32, DT_I64, DT_BOOL, DT_BF16, numel_of, shape_str, norm_axis
from flint.ml.ops import (
    MAX_RANGE_ELEMS,
    unary, binary, where, clip, cast, transpose, concat, gather, gather_elements, scatter_elements,
    scatter_nd, slice, expand, tile, range_op, onehot, resize_nearest, reduce, argmax, topk, cumsum,
    softmax, softmax_rows, layer_norm, batch_norm, matmul, gemm, conv1d, einsum, fused_attention,
    U_SQRT, U_RELU, U_ATAN, U_COS, U_SIN, U_ABS, U_NEG, U_NOT, U_LEAKY,
    U_ERF, U_EXP, U_TANH, U_SIGMOID, U_RECIP, U_FLOOR,
    B_ADD, B_SUB, B_MUL, B_DIV, B_MOD, B_MAX, B_EQ, B_LT, B_GT, B_AND, B_POW, B_MIN, B_OR,
)

comptime OP_CONSTANT: Int = 0
comptime OP_SHAPE: Int = 1
comptime OP_UNSQUEEZE: Int = 2
comptime OP_SQUEEZE: Int = 3
comptime OP_RESHAPE: Int = 4
comptime OP_FLATTEN: Int = 5
comptime OP_CONCAT: Int = 6
comptime OP_GATHER: Int = 7
comptime OP_GATHER_ELEMENTS: Int = 8
comptime OP_SCATTER_ELEMENTS: Int = 9
comptime OP_SCATTER_ND: Int = 10
comptime OP_TRANSPOSE: Int = 11
comptime OP_SLICE: Int = 12
comptime OP_CAST: Int = 13
comptime OP_CONSTANT_OF_SHAPE: Int = 14
comptime OP_WHERE: Int = 15
comptime OP_EXPAND: Int = 16
comptime OP_TILE: Int = 17
comptime OP_RANGE: Int = 18
comptime OP_MATMUL: Int = 19
comptime OP_GEMM: Int = 20
comptime OP_CONV: Int = 21
comptime OP_LAYER_NORM: Int = 22
comptime OP_SOFTMAX: Int = 23
comptime OP_REDUCE_L2: Int = 24
comptime OP_REDUCE_SUM: Int = 25
comptime OP_ARGMAX: Int = 26
comptime OP_TOPK: Int = 27
comptime OP_CUMSUM: Int = 28
comptime OP_ONEHOT: Int = 29
comptime OP_RESIZE: Int = 30
comptime OP_CLIP: Int = 31
comptime OP_SPLIT: Int = 32
comptime OP_IDENTITY: Int = 33
comptime OP_EINSUM: Int = 34
comptime OP_BATCH_NORM: Int = 35
comptime OP_UNARY: Int = 100   # + U_* code
comptime OP_BINARY: Int = 200  # + B_* code


def op_code(name: String) -> Int:
    """The interpreter's code for an ONNX op type, or -1 when it is not supported."""
    if name == "Split": return OP_SPLIT
    if name == "Identity": return OP_IDENTITY
    if name == "Einsum": return OP_EINSUM
    if name == "BatchNormalization": return OP_BATCH_NORM
    if name == "Erf": return OP_UNARY + U_ERF
    if name == "Exp": return OP_UNARY + U_EXP
    if name == "Tanh": return OP_UNARY + U_TANH
    if name == "Sigmoid": return OP_UNARY + U_SIGMOID
    if name == "Reciprocal": return OP_UNARY + U_RECIP
    if name == "Floor": return OP_UNARY + U_FLOOR
    if name == "Pow": return OP_BINARY + B_POW
    if name == "Min": return OP_BINARY + B_MIN
    if name == "Or": return OP_BINARY + B_OR
    if name == "Constant": return OP_CONSTANT
    if name == "Shape": return OP_SHAPE
    if name == "Unsqueeze": return OP_UNSQUEEZE
    if name == "Squeeze": return OP_SQUEEZE
    if name == "Reshape": return OP_RESHAPE
    if name == "Flatten": return OP_FLATTEN
    if name == "Concat": return OP_CONCAT
    if name == "Gather": return OP_GATHER
    if name == "GatherElements": return OP_GATHER_ELEMENTS
    if name == "ScatterElements": return OP_SCATTER_ELEMENTS
    if name == "ScatterND": return OP_SCATTER_ND
    if name == "Transpose": return OP_TRANSPOSE
    if name == "Slice": return OP_SLICE
    if name == "Cast": return OP_CAST
    if name == "ConstantOfShape": return OP_CONSTANT_OF_SHAPE
    if name == "Where": return OP_WHERE
    if name == "Expand": return OP_EXPAND
    if name == "Tile": return OP_TILE
    if name == "Range": return OP_RANGE
    if name == "MatMul": return OP_MATMUL
    if name == "Gemm": return OP_GEMM
    if name == "Conv": return OP_CONV
    if name == "LayerNormalization": return OP_LAYER_NORM
    if name == "Softmax": return OP_SOFTMAX
    if name == "ReduceL2": return OP_REDUCE_L2
    if name == "ReduceSum": return OP_REDUCE_SUM
    if name == "ArgMax": return OP_ARGMAX
    if name == "TopK": return OP_TOPK
    if name == "CumSum": return OP_CUMSUM
    if name == "OneHot": return OP_ONEHOT
    if name == "Resize": return OP_RESIZE
    if name == "Clip": return OP_CLIP
    if name == "Sqrt": return OP_UNARY + U_SQRT
    if name == "Relu": return OP_UNARY + U_RELU
    if name == "Atan": return OP_UNARY + U_ATAN
    if name == "Cos": return OP_UNARY + U_COS
    if name == "Sin": return OP_UNARY + U_SIN
    if name == "Abs": return OP_UNARY + U_ABS
    if name == "Neg": return OP_UNARY + U_NEG
    if name == "Not": return OP_UNARY + U_NOT
    if name == "LeakyRelu": return OP_UNARY + U_LEAKY
    if name == "Add": return OP_BINARY + B_ADD
    if name == "Sub": return OP_BINARY + B_SUB
    if name == "Mul": return OP_BINARY + B_MUL
    if name == "Div": return OP_BINARY + B_DIV
    if name == "Mod": return OP_BINARY + B_MOD
    if name == "Max": return OP_BINARY + B_MAX
    if name == "Equal": return OP_BINARY + B_EQ
    if name == "Less": return OP_BINARY + B_LT
    if name == "Greater": return OP_BINARY + B_GT
    if name == "And": return OP_BINARY + B_AND
    return -1


struct Attr(Copyable, Movable):
    var name: String
    var i: Int
    var f: Float32
    var s: String
    var ints: List[Int]
    var floats: List[Float32]
    var t: Tensor
    var has_t: Bool

    def __init__(out self):
        self.name = String()
        self.i = 0
        self.f = 0
        self.s = String()
        self.ints = List[Int]()
        self.floats = List[Float32]()
        self.t = Tensor()
        self.has_t = False


struct Node(Copyable, Movable):
    var op: Int
    var op_type: String
    var name: String
    var inputs: List[Int]
    var outputs: List[Int]
    var attrs: List[Attr]

    def __init__(out self):
        self.op = -1
        self.op_type = String()
        self.name = String()
        self.inputs = List[Int]()
        self.outputs = List[Int]()
        self.attrs = List[Attr]()

    def find(self, name: String) -> Int:
        for k in range(len(self.attrs)):
            if self.attrs[k].name == name:
                return k
        return -1

    def attr_i(self, name: String, default: Int) -> Int:
        var k = self.find(name)
        return self.attrs[k].i if k >= 0 else default

    def attr_f(self, name: String, default: Float32) -> Float32:
        var k = self.find(name)
        return self.attrs[k].f if k >= 0 else default

    def attr_ints(self, name: String) -> List[Int]:
        var k = self.find(name)
        return self.attrs[k].ints.copy() if k >= 0 else List[Int]()

    def attr_s(self, name: String) -> String:
        var k = self.find(name)
        return self.attrs[k].s if k >= 0 else String()


struct ExternalData(Movable):
    """The side files (`<model>.onnx.data`) of a graph saved with external tensor data, each
    memory-mapped once; tensors borrow their bytes from the mapping like raw_data."""
    var dir: String
    var names: List[String]
    var addrs: List[Int]
    var sizes: List[Int]

    def __init__(out self, dir: String):
        self.dir = dir
        self.names = List[String]()
        self.addrs = List[Int]()
        self.sizes = List[Int]()

    def __deinit__(deinit self):
        for k in range(len(self.addrs)):
            _ = external_call["munmap", Int32](self.addrs[k], Int64(self.sizes[k]))

    def span(mut self, location: String, offset: Int, length: Int) raises -> Int:
        """The address of `length` bytes at `offset` in the side file `location`."""
        if location.find("..") >= 0 or location.startswith("/"):
            raise Error("ONNX: external data must sit beside the model (" + location + ")")
        var f = -1
        for k in range(len(self.names)):
            if self.names[k] == location:
                f = k
        if f < 0:
            var path = self.dir + location
            var fd = open_read(path)
            if fd < 0:
                raise Error("ONNX: cannot open external data " + path)
            var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
            var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0))
            close_fd(fd)
            if addr == 0 or addr == -1:
                raise Error("ONNX: cannot map " + path)
            self.names.append(location)
            self.addrs.append(addr)
            self.sizes.append(size)
            f = len(self.names) - 1
        if offset < 0 or length < 0 or offset + length > self.sizes[f]:
            raise Error("ONNX: external data range outside " + location)
        return self.addrs[f] + offset


def _parse_tensor(mut r: PbReader) raises -> Tuple[String, Tensor]:
    """A TensorProto with inline data (attributes); see `_parse_tensor_ext`."""
    var none = ExternalData(String())
    return _parse_tensor_ext(r, none)


def _parse_tensor_ext(mut r: PbReader, mut ext: ExternalData) raises -> Tuple[String, Tensor]:
    """A TensorProto. Float raw data stays in the mapping (borrowed), as does external data (a side
    file named by `external_data`, mapped through `ext`); the rest is copied."""
    var name = String()
    var dims = List[Int]()
    var dtype = DT_F32
    var raw_start = -1
    var raw_end = -1
    var floats = List[Float32]()
    var ints = List[Int]()
    var external = False
    var location = String()
    var ext_offset = 0
    var ext_length = -1
    while r.more():
        r.key()
        if r.field == 1:
            r.append_ints(dims)
        elif r.field == 2:
            dtype = r.varint()
        elif r.field == 4:
            r.append_floats(floats)
        elif r.field == 5 or r.field == 7:
            r.append_ints(ints)
        elif r.field == 8:
            name = r.string()
        elif r.field == 9:
            var span = r.length_delimited()
            raw_start = span[0]
            raw_end = span[1]
        elif r.field == 13:
            # StringStringEntryProto {key = 1, value = 2}
            var er = r.sub()
            var key = String()
            var value = String()
            while er.more():
                er.key()
                if er.field == 1:
                    key = er.string()
                elif er.field == 2:
                    value = er.string()
                else:
                    er.skip()
            if key == "location":
                location = value
            elif key == "offset":
                ext_offset = atol(value)
            elif key == "length":
                ext_length = atol(value)
        elif r.field == 14:
            external = r.varint() == 1
        else:
            r.skip()
    var n = numel_of(dims)
    if n < 0:
        raise Error("ONNX: tensor shape overflow for " + name)
    if external:
        if ext.dir.byte_length() == 0 or location.byte_length() == 0:
            raise Error("ONNX: external tensor data is not supported here (" + name + ")")
        var width = 4 if dtype == DT_F32 or dtype == DT_I32 else (8 if dtype == DT_I64 else 1)
        var nbytes = n * width
        if ext_length >= 0 and ext_length != nbytes:
            raise Error("ONNX: external size mismatch for " + name)
        raw_start = ext.span(location, ext_offset, nbytes)
        raw_end = raw_start + nbytes
    if dtype == DT_F32:
        if raw_start >= 0:
            var raw_bytes = raw_end - raw_start
            if raw_bytes // 4 != n or raw_bytes % 4 != 0:
                raise Error("ONNX: raw size mismatch for " + name)
            return (name, Tensor.borrowed(dims^, raw_start))
        if len(floats) < n:
            raise Error("ONNX: float data size mismatch for " + name)
        return (name, Tensor.floats(dims^, floats^))
    if dtype != DT_I32 and dtype != DT_I64 and dtype != DT_BOOL:
        raise Error("ONNX: tensor " + name + " has unsupported type " + String(dtype))
    if raw_start >= 0:
        var width = 8 if dtype == DT_I64 else (4 if dtype == DT_I32 else 1)
        var raw_bytes = raw_end - raw_start
        if raw_bytes // width != n or raw_bytes % width != 0:
            raise Error("ONNX: raw size mismatch for " + name)
        ints = List[Int](length=n, fill=0)
        for k in range(n):
            var at = raw_start + k * width
            if width == 8:
                ints[k] = Int(Pointer[Int64, MutAnyOrigin](unsafe_from_address=at).unsafe_load[alignment=1]())
            elif width == 4:
                ints[k] = Int(Pointer[Int32, MutAnyOrigin](unsafe_from_address=at).unsafe_load[alignment=1]())
            else:
                ints[k] = Int(Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at).unsafe_load())
    elif len(ints) < n:
        raise Error("ONNX: int data size mismatch for " + name)
    return (name, Tensor.ints(dtype, dims^, ints^))


def _value_name(mut r: PbReader) raises -> String:
    """The name of a ValueInfoProto."""
    var name = String()
    while r.more():
        r.key()
        if r.field == 1:
            name = r.string()
        else:
            r.skip()
    return name


struct OnnxModel(Movable):
    var path: String
    var map_addr: Int
    var map_size: Int
    var nodes: List[Node]
    var names: List[String]
    var ids: Dict[String, Int]
    var values: List[Tensor]
    var static_value: List[Bool]
    var last_use: List[Int]
    var input_ids: List[Int]
    var output_ids: List[Int]
    var pool: JobPool
    var trace_path: String
    var gpu: Optional[GpuMatmul]
    var profile: Dict[String, Int]
    var profiling: Bool
    var fuse: List[Int]          # 0 normal, 1 skipped (part of a fused chain), 2 fused attention,
                                 # 3 gather-matmul (GPU), 4 skipped on the GPU only (gather / reshape feeding 3)
    var fuse_q: List[Int]
    var fuse_k: List[Int]
    var fuse_scale: List[Float32]
    var fuse_w: List[Int]        # gather-matmul GPU weight-cache key
    # GPU residency: with a GPU, products / attention / element-wise / LayerNorm results stay in
    # pooled device buffers (`dslot`); the host copy is filled only when a CPU op reads it.
    var keep_on_gpu: Bool
    var resident: Bool
    var dslot: List[Int]
    var host_ok: List[Bool]
    var pend_slot: List[Int]
    var pend_host: List[Bool]
    var split_outs: List[Tensor]
    var ext: ExternalData        # mapped `.onnx.data` side files (graphs saved with external data)

    def __init__(out self):
        self.path = String()
        self.map_addr = 0
        self.map_size = 0
        self.nodes = List[Node]()
        self.names = List[String]()
        self.ids = Dict[String, Int]()
        self.values = List[Tensor]()
        self.static_value = List[Bool]()
        self.last_use = List[Int]()
        self.input_ids = List[Int]()
        self.output_ids = List[Int]()
        self.pool = JobPool()
        self.trace_path = String()
        self.gpu = None
        self.profile = Dict[String, Int]()
        self.profiling = False
        self.fuse = List[Int]()
        self.fuse_q = List[Int]()
        self.fuse_k = List[Int]()
        self.fuse_scale = List[Float32]()
        self.fuse_w = List[Int]()
        self.keep_on_gpu = True
        self.resident = False
        self.dslot = List[Int]()
        self.host_ok = List[Bool]()
        self.pend_slot = List[Int]()
        self.pend_host = List[Bool]()
        self.split_outs = List[Tensor]()
        self.ext = ExternalData(String())

    def __deinit__(deinit self):
        if self.map_addr != 0:
            _ = external_call["munmap", Int32](self.map_addr, Int64(self.map_size))
        self.pool.stop()

    def _id(mut self, name: String) -> Int:
        if name.byte_length() == 0:
            return -1
        var found = self.ids.get(name)
        if found:
            return found.value()
        var k = len(self.names)
        self.ids[name] = k
        self.names.append(name)
        self.values.append(Tensor())
        self.static_value.append(False)
        self.last_use.append(-1)
        return k

    @staticmethod
    def load(path: String, workers: Int = -1) raises -> OnnxModel:
        var m = OnnxModel()
        m.path = path
        var slash = path.rfind("/")
        m.ext = ExternalData(String(path[byte=0:slash + 1]) if slash >= 0 else String("./"))
        var fd = open_read(path)
        if fd < 0:
            raise Error("ONNX: cannot open " + path)
        var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
        # PROT_READ = 1, MAP_PRIVATE = 2
        var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0))
        close_fd(fd)
        if addr == 0 or addr == -1:
            raise Error("ONNX: cannot map " + path)
        m.map_addr = addr
        m.map_size = size
        var top = PbReader(addr, addr + size)
        var graph_found = False
        while top.more():
            top.key()
            if top.field == 7 and top.wire == WIRE_LEN:
                var g = top.sub()
                m._parse_graph(g)
                graph_found = True
            else:
                top.skip()
        if not graph_found:
            raise Error("ONNX: no graph in " + path)
        for k in range(len(m.nodes)):
            for v in m.nodes[k].inputs:
                if v >= 0:
                    m.last_use[v] = k
        for v in m.output_ids:
            if v < 0 or v >= len(m.last_use):
                raise Error("ONNX: invalid graph output id")
            m.last_use[v] = len(m.nodes)
        m._plan_fusions()
        _ = m.pool.start(default_worker_count() if workers < 0 else workers)
        return m^

    def _gpu_attention(mut self, k: Int) raises -> Tensor:
        """Runs a fused attention chain on the GPU (head dim <= 64); empty tensor on fallback."""
        var qi = self.fuse_q[k]
        var ki = self.fuse_k[k]
        var vi = self.nodes[k].inputs[1]
        if qi < 0 or ki < 0 or vi < 0:
            return Tensor()
        ref q = self.values[qi]
        ref kk = self.values[ki]
        ref v = self.values[vi]
        var r = q.rank()
        if r < 3 or kk.rank() != r or v.rank() != r or not q.is_float() or not kk.is_float() or not v.is_float():
            return Tensor()
        var D = q.shape[r - 1]
        var H = q.shape[r - 2]
        var L = q.shape[r - 3]
        var S = kk.shape[r - 3]
        if kk.shape[r - 1] != D or kk.shape[r - 2] != H or v.shape[r - 1] != D or v.shape[r - 2] != H or v.shape[r - 3] != S:
            return Tensor()
        var X = 1
        for d in range(r - 3):
            if kk.shape[d] != q.shape[d] or v.shape[d] != q.shape[d]:
                return Tensor()
            X *= q.shape[d]
        if D > 64 or D <= 0 or H <= 0 or L <= 0 or S <= 0 or X <= 0:
            return Tensor()
        var qshape = q.shape.copy()
        if self.resident:
            var pq = self._dev(qi)
            var pk = self._dev(ki)
            var pv = self._dev(vi)
            var outd = self._dev_out(qshape^)
            _ = self.gpu.value().attention_dev(pq, pk, pv, self._pend_ptr(), X, L, S, H, D, self.fuse_scale[k])
            return outd^
        var out = Tensor.zeros(qshape^)
        if not self.gpu.value().attention(self.values[qi].fptr(), self.values[ki].fptr(),
                                          self.values[vi].fptr(), out.fptr(), X, L, S, H, D,
                                          self.fuse_scale[k]):
            return Tensor()
        return out^

    def _gpu_gather_matmul(mut self, k: Int) raises -> Tensor:
        var qi = self.fuse_q[k]
        var ki = self.fuse_k[k]
        var wi = self.nodes[k].inputs[1]
        self._ensure_host(ki)
        self._ensure_host(wi)
        ref x = self.values[qi]
        ref ix = self.values[ki]
        ref w = self.values[wi]
        var rows = x.shape[0]
        var cin = x.shape[1]
        var m = ix.shape[0]
        var slots = ix.shape[1]
        if w.rank() != 2 or x.rank() != 2 or ix.rank() != 2 or w.shape[0] != slots * cin or not w.is_float():
            raise Error("gather-matmul: shape mismatch")
        var n = w.shape[1]
        var idx = List[Float32](length=m * slots, fill=0.0)
        for e in range(m * slots):
            var r = ix.int_at(e)
            if r < 0:
                r += rows
            if r < 0 or r >= rows:
                raise Error("gather-matmul: index out of range")
            idx[e] = Float32(r)
        var widx = self.gpu.value().weight(self.fuse_w[k], w.fptr(), w.shape[0], n, False)
        if self.resident:
            var px = self._dev(qi)
            var outd = self._dev_out([m, n])
            self.gpu.value().gather_matmul_dev(px, cin, idx, m, slots, widx, self._pend_ptr(), n)
            return outd^
        self._ensure_host(qi)
        var out = Tensor.zeros([m, n])
        self.gpu.value().gather_matmul(self.values[qi].fptr(), rows, cin, idx, m, slots, widx, out.fptr(), n)
        return out^

    def _plan_gather_matmul(mut self, uses: List[Int], producer: List[Int]):
        """Gather(X [R, C], idx [N, K], axis 0) -> Reshape [N, K*C] -> MatMul(., static W [K*C, n]):
        the GPU runs it as one gather-GEMM (the gathered [N, K*C] matrix is never built)."""
        for c in range(len(self.nodes)):
            if self.nodes[c].op != OP_MATMUL or len(self.nodes[c].inputs) != 2 or self.fuse[c] != 0:
                continue
            var wid = self.nodes[c].inputs[1]
            if wid < 0 or wid >= len(self.values):
                continue
            var key = wid * 4
            if not self.static_value[wid]:
                var t = producer[wid]
                if t < 0 or self.nodes[t].op != OP_TRANSPOSE or len(self.nodes[t].inputs) < 1:
                    continue
                var src = self.nodes[t].inputs[0]
                if src < 0 or src >= len(self.values):
                    continue
                if not self.static_value[src]:
                    var rs = producer[src]
                    if rs < 0 or self.nodes[rs].op != OP_RESHAPE or len(self.nodes[rs].inputs) < 1:
                        continue
                    var rsrc = self.nodes[rs].inputs[0]
                    if rsrc < 0 or rsrc >= len(self.values) or not self.static_value[rsrc]:
                        continue
                    src = rsrc
                key = (1 << 40) + src * 4
            elif self.values[wid].rank() != 2 or not self.values[wid].is_float():
                continue
            var rv = self.nodes[c].inputs[0]
            if rv < 0 or rv >= len(uses):
                continue
            var r = producer[rv]
            if r < 0 or self.nodes[r].op != OP_RESHAPE or uses[rv] != 1 or len(self.nodes[r].inputs) < 1:
                continue
            var gv = self.nodes[r].inputs[0]
            if gv < 0 or gv >= len(uses):
                continue
            var gn = producer[gv]
            if gn < 0 or self.nodes[gn].op != OP_GATHER or uses[gv] != 1 or self.nodes[gn].attr_i("axis", 0) != 0 \
                    or len(self.nodes[gn].inputs) < 2:
                continue
            self.fuse[r] = 4
            self.fuse[gn] = 4
            self.fuse_w[c] = key
            self.fuse[c] = 3
            self.fuse_q[c] = self.nodes[gn].inputs[0]
            self.fuse_k[c] = self.nodes[gn].inputs[1]
            for v in self.nodes[gn].inputs:
                if v >= 0 and self.last_use[v] < c:
                    self.last_use[v] = c

    def _plan_fusions(mut self) raises:
        """Finds Einsum(q, k) -> [Mul|Div by a constant scalar] -> Softmax(last axis) -> Einsum(., v)
        chains in multi-head-attention layout and runs each as one fused kernel at the last Einsum."""
        var nn = len(self.nodes)
        self.fuse = List[Int](length=nn, fill=0)
        self.fuse_q = List[Int](length=nn, fill=-1)
        self.fuse_k = List[Int](length=nn, fill=-1)
        self.fuse_scale = List[Float32](length=nn, fill=1.0)
        self.fuse_w = List[Int](length=nn, fill=-1)
        var uses = List[Int](length=len(self.values), fill=0)
        var producer = List[Int](length=len(self.values), fill=-1)
        for k in range(nn):
            for v in self.nodes[k].inputs:
                if v >= 0 and v < len(uses):
                    uses[v] += 1
            for v in self.nodes[k].outputs:
                if v >= 0 and v < len(producer):
                    producer[v] = k
        for v in self.output_ids:
            if v >= 0 and v < len(uses):
                uses[v] += 1
        for c in range(nn):
            if self.nodes[c].op != OP_EINSUM or len(self.nodes[c].inputs) != 2:
                continue
            var pv = self.nodes[c].inputs[0]
            if pv < 0 or pv >= len(uses):
                continue
            var sm = producer[pv]
            if sm < 0 or self.nodes[sm].op != OP_SOFTMAX or uses[pv] != 1 or len(self.nodes[sm].inputs) < 1:
                continue
            var sv = self.nodes[sm].inputs[0]
            if sv < 0 or sv >= len(uses):
                continue
            var mid = producer[sv]
            if mid < 0 or uses[sv] != 1:
                continue
            var scale: Float32 = 1.0
            var a = mid
            var scale_node = -1
            var mop = self.nodes[mid].op
            if mop == OP_BINARY + B_MUL or mop == OP_BINARY + B_DIV:
                if len(self.nodes[mid].inputs) != 2:
                    continue
                var i0 = self.nodes[mid].inputs[0]
                var i1 = self.nodes[mid].inputs[1]
                if i0 < 0 or i1 < 0 or not self.static_value[i1] or self.values[i1].numel() != 1 \
                        or not self.values[i1].is_float():
                    continue
                var f = self.values[i1].fget(0)
                scale = f if mop == OP_BINARY + B_MUL else 1.0 / f
                a = producer[i0]
                if a < 0 or uses[i0] != 1:
                    continue
                scale_node = mid
            if self.nodes[a].op != OP_EINSUM or len(self.nodes[a].inputs) != 2:
                continue
            if not _attention_equations(self.nodes[a].attr_s("equation"), self.nodes[c].attr_s("equation")):
                continue
            var rank = self.nodes[sm].attr_i("axis", -1)
            if rank != -1 and rank != 3 and rank != 2:
                continue
            self.fuse[a] = 1
            if scale_node >= 0:
                self.fuse[scale_node] = 1
            self.fuse[sm] = 1
            self.fuse[c] = 2
            self.fuse_q[c] = self.nodes[a].inputs[0]
            self.fuse_k[c] = self.nodes[a].inputs[1]
            self.fuse_scale[c] = scale
            for v in self.nodes[a].inputs:
                if v >= 0 and self.last_use[v] < c:
                    self.last_use[v] = c
        self._plan_gather_matmul(uses, producer)

    def _parse_graph(mut self, mut g: PbReader) raises:
        var graph_inputs = List[String]()
        while g.more():
            g.key()
            if g.field == 1:
                var nr = g.sub()
                self._parse_node(nr)
            elif g.field == 5:
                var tr = g.sub()
                var nt = _parse_tensor_ext(tr, self.ext)
                var k = self._id(nt[0])
                if k < 0:
                    raise Error("ONNX: initializer has empty name")
                self.values[k] = nt[1].copy()
                self.static_value[k] = True
            elif g.field == 11:
                var vr = g.sub()
                graph_inputs.append(_value_name(vr))
            elif g.field == 12:
                var vr = g.sub()
                var nm = _value_name(vr)
                var vid = self._id(nm)
                if vid < 0:
                    raise Error("ONNX: graph output has empty name")
                self.output_ids.append(vid)
            else:
                g.skip()
        for nm in graph_inputs:
            var k = self._id(nm)
            if k < 0:
                raise Error("ONNX: graph input has empty name")
            if not self.static_value[k]:
                self.input_ids.append(k)

    def _parse_node(mut self, mut r: PbReader) raises:
        var node = Node()
        var ins = List[String]()
        var outs = List[String]()
        while r.more():
            r.key()
            if r.field == 1:
                ins.append(r.string())
            elif r.field == 2:
                outs.append(r.string())
            elif r.field == 3:
                node.name = r.string()
            elif r.field == 4:
                node.op_type = r.string()
            elif r.field == 5:
                var ar = r.sub()
                node.attrs.append(self._parse_attr(ar))
            elif r.field == 7:
                var domain = r.string()
                if domain.byte_length() > 0 and domain != "ai.onnx":
                    raise Error("ONNX: custom domain " + domain + " is not supported")
            else:
                r.skip()
        node.op = op_code(node.op_type)
        if node.op < 0:
            raise Error("ONNX: operator " + node.op_type + " is not supported (" + node.name + ")")
        for nm in ins:
            node.inputs.append(self._id(nm))
        for nm in outs:
            node.outputs.append(self._id(nm))
        if node.op == OP_CONSTANT:
            # fold constants into static values now
            var k = node.find("value")
            if k < 0 or not node.attrs[k].has_t:
                raise Error("ONNX: Constant " + node.name + " needs a tensor 'value'")
            if len(node.outputs) == 0 or node.outputs[0] < 0 or node.outputs[0] >= len(self.values):
                raise Error("ONNX: Constant " + node.name + " needs a named output")
            self.values[node.outputs[0]] = node.attrs[k].t.copy()
            self.static_value[node.outputs[0]] = True
            return
        self.nodes.append(node^)

    def _parse_attr(mut self, mut r: PbReader) raises -> Attr:
        var a = Attr()
        while r.more():
            r.key()
            if r.field == 1:
                a.name = r.string()
            elif r.field == 2:
                var fl = List[Float32]()
                r.append_floats(fl)
                if len(fl) == 0:
                    raise Error("ONNX: truncated float attribute")
                a.f = fl[0]
            elif r.field == 3:
                a.i = r.varint()
            elif r.field == 4:
                a.s = r.string()
            elif r.field == 5:
                var tr = r.sub()
                var nt = _parse_tensor(tr)
                a.t = nt[1].copy()
                a.has_t = True
            elif r.field == 7:
                r.append_floats(a.floats)
            elif r.field == 8:
                r.append_ints(a.ints)
            else:
                r.skip()
        return a^

    # ------------------------------------------------------------------------------ running

    def use_gpu(mut self) -> Bool:
        """Run the matrix products on the first MAX GPU device; False (CPU only) if there is none."""
        try:
            self.gpu = GpuMatmul()
            return True
        except:
            self.gpu = None
            return False

    def gpu_name(self) raises -> String:
        if self.gpu:
            return self.gpu.value().name()
        return String("cpu")

    # ------------------------------------------------------------------ device residency

    def _grow_slots(mut self):
        while len(self.dslot) < len(self.values):
            self.dslot.append(-1)
            self.host_ok.append(True)

    def _on_gpu(self, v: Int) -> Bool:
        return v >= 0 and v < len(self.dslot) and self.dslot[v] >= 0

    def _drop_slot(mut self, v: Int):
        if self._on_gpu(v):
            self.gpu.value().release(self.dslot[v])
            self.dslot[v] = -1
            self.host_ok[v] = True

    def _ensure_host(mut self, v: Int) raises:
        """Fills the host copy of a device-resident value (once)."""
        if not self._on_gpu(v) or self.host_ok[v]:
            return
        var n = self.values[v].numel()
        self.values[v].f = List[Float32](length=n, fill=0.0)
        self.values[v].ext = 0
        self.gpu.value().download(self.dslot[v], Int(self.values[v].f.unsafe_ptr()), n)
        self.host_ok[v] = True

    def _dev(mut self, v: Int) raises -> FP:
        """Device pointer of a float value: its resident buffer, the weight cache for initializers,
        else an upload (which then stays resident until the value's last use)."""
        if self._on_gpu(v):
            return self.gpu.value().ptr(self.dslot[v])
        var n = self.values[v].numel()
        var src = self.values[v].fptr()
        if self.static_value[v]:
            return self.gpu.value().wptr(self.gpu.value().weight((3 << 40) + v * 4, src, 1, n, False))
        var slot = self.gpu.value().alloc(n)
        self.gpu.value().upload(slot, src, n)
        self._grow_slots()
        self.dslot[v] = slot
        self.host_ok[v] = True
        return self.gpu.value().ptr(slot)

    def _dev_out(mut self, var shape: List[Int]) raises -> Tensor:
        """A float result that lives only on the device (pending for the node's next output)."""
        var slot = self.gpu.value().alloc(numel_of(shape))
        self.pend_slot.append(slot)
        self.pend_host.append(False)
        var t = Tensor()
        t.shape = shape^
        return t^

    def _pend_ptr(self) -> FP:
        return self.gpu.value().ptr(self.pend_slot[len(self.pend_slot) - 1])

    def _gpu_elementwise(mut self, k: Int) raises -> Tensor:
        """Binary / unary / Cast->BF16 / Gather / Concat / Split / LayerNorm on the GPU when an
        input already lives there; an empty tensor when the node does not qualify (CPU runs)."""
        var op = self.nodes[k].op
        var nin = len(self.nodes[k].inputs)
        if nin == 0:
            return Tensor()
        var a = self.nodes[k].inputs[0]
        if a < 0 or a >= len(self.values):
            return Tensor()
        if op >= OP_BINARY and nin == 2:
            var code = op - OP_BINARY
            var b = self.nodes[k].inputs[1]
            if b < 0 or b >= len(self.values):
                return Tensor()
            if not gpu_binary_supported(code) or (not self._on_gpu(a) and not self._on_gpu(b)):
                return Tensor()
            if not self.values[a].is_float() or not self.values[b].is_float():
                return Tensor()
            var na = self.values[a].numel()
            var nb = self.values[b].numel()
            if na <= 0 or nb <= 0:
                return Tensor()
            var big = a
            var small = b
            var swap = False
            if nb > na:
                big = b
                small = a
                swap = True
            var nbig = max(na, nb)
            var nsmall = min(na, nb)
            var mode: Int
            if na == nb:
                if self.values[a].shape != self.values[b].shape:
                    return Tensor()
                mode = 0
                big = a
                small = b
                swap = False
            elif nsmall == 1:
                mode = 1
            else:
                # the small operand must be the trailing dims of the big one
                var ok = self.values[small].rank() <= self.values[big].rank()
                var i = self.values[small].rank() - 1
                var j = self.values[big].rank() - 1
                while ok and i >= 0:
                    if self.values[small].shape[i] != self.values[big].shape[j]:
                        ok = False
                    i -= 1
                    j -= 1
                if not ok or nbig % nsmall != 0:
                    return Tensor()
                mode = 2
            var scalar: Float32 = 0
            var pb: FP
            if mode == 1:
                self._ensure_host(small)
                scalar = self.values[small].fget(0)
                pb = self._dev(big)
            else:
                pb = self._dev(small)
            var pa = self._dev(big)
            var out = self._dev_out(self.values[big].shape.copy())
            self.gpu.value().binary_dev(code, pa, pb, self._pend_ptr(), nbig, nsmall, mode, scalar, swap)
            return out^
        if op == OP_CAST and self.nodes[k].attr_i("to", DT_F32) == DT_BF16 and self._on_gpu(a) \
                and self.values[a].is_float() and self.values[a].numel() > 0:
            var pc = self._dev(a)
            var outc = self._dev_out(self.values[a].shape.copy())
            self.gpu.value().unary_dev(GU_BF16, pc, self._pend_ptr(), self.values[a].numel())
            return outc^
        if op >= OP_UNARY and op < OP_BINARY:
            var code = op - OP_UNARY
            if not gpu_unary_supported(code) or not self._on_gpu(a) or not self.values[a].is_float() \
                    or self.values[a].numel() <= 0:
                return Tensor()
            var pa = self._dev(a)
            var out = self._dev_out(self.values[a].shape.copy())
            self.gpu.value().unary_dev(code, pa, self._pend_ptr(), self.values[a].numel())
            return out^
        if op == OP_GATHER and self._on_gpu(a) and self.values[a].is_float() and nin == 2:
            var r = self.values[a].rank()
            if r < 1 or norm_axis(self.nodes[k].attr_i("axis", 0), r) != 0:
                return Tensor()
            var ii = self.nodes[k].inputs[1]
            if ii < 0 or ii >= len(self.values):
                return Tensor()
            self._ensure_host(ii)
            var rows = self.values[a].shape[0]
            var row = self.values[a].numel() // rows if rows > 0 else 0
            var ni = self.values[ii].numel()
            if row <= 0 or ni <= 0:
                return Tensor()
            var idx = List[Float32](length=ni, fill=0.0)
            for e in range(ni):
                var q = self.values[ii].int_at(e)
                if q < 0:
                    q += rows
                if q < 0 or q >= rows:
                    raise Error("Gather: index out of range")
                idx[e] = Float32(q)
            var shape = self.values[ii].shape.copy()
            for d in range(1, r):
                shape.append(self.values[a].shape[d])
            var px = self._dev(a)
            var out = self._dev_out(shape^)
            self.gpu.value().gather_rows_dev(px, idx, self._pend_ptr(), row)
            return out^
        if op == OP_CONCAT:
            var any_dev = False
            var r = -1
            for j in range(nin):
                var v = self.nodes[k].inputs[j]
                if v < 0 or v >= len(self.values):
                    return Tensor()
                if self._on_gpu(v):
                    any_dev = True
                if not self.values[v].is_float():
                    return Tensor()
                if self.values[v].numel() > 0:
                    r = self.values[v].rank()
            if not any_dev or r < 1:
                return Tensor()
            var axis = norm_axis(self.nodes[k].attr_i("axis", 0), r)
            if axis < 0 or axis >= r:
                return Tensor()
            var shape = List[Int]()
            var total_a = 0
            for j in range(nin):
                var v = self.nodes[k].inputs[j]
                if self.values[v].numel() == 0:
                    continue
                if self.values[v].rank() != r:
                    return Tensor()
                if len(shape) == 0:
                    shape = self.values[v].shape.copy()
                for d in range(r):
                    if d != axis and self.values[v].shape[d] != shape[d]:
                        return Tensor()
                total_a += self.values[v].shape[axis]
            shape[axis] = total_a
            var outer = 1
            for d in range(axis):
                outer *= shape[d]
            var inner = 1
            for d in range(axis + 1, r):
                inner *= shape[d]
            var out = self._dev_out(shape^)
            var po = self._pend_ptr()
            var off = 0
            for j in range(nin):
                var v = self.nodes[k].inputs[j]
                if self.values[v].numel() == 0:
                    continue
                var ca = self.values[v].shape[axis]
                var pv = self._dev(v)
                self.gpu.value().copy_dev(pv, po, outer, ca * inner, ca * inner, 0, total_a * inner, off * inner)
                off += ca
            return out^
        if op == OP_SPLIT and self._on_gpu(a) and self.values[a].is_float():
            var r = self.values[a].rank()
            if r < 1:
                return Tensor()
            var axis = norm_axis(self.nodes[k].attr_i("axis", 0), r)
            var parts = len(self.nodes[k].outputs)
            if axis < 0 or axis >= r or parts < 1:
                return Tensor()
            var sizes = self.nodes[k].attr_ints("split")
            var si = self._in(k, 1)
            if si >= 0:
                self._ensure_host(si)
                sizes = self.values[si].int_list()
            var dim = self.values[a].shape[axis]
            if len(sizes) == 0:
                if self.nodes[k].attr_i("num_outputs", 0) != 0:
                    if self.nodes[k].attr_i("num_outputs", 0) != parts:
                        return Tensor()
                    var base = dim // parts
                    var rem = dim % parts
                    sizes = List[Int](capacity=parts)
                    for sq in range(parts):
                        sizes.append(base + 1 if sq < rem else base)
                elif dim % parts != 0:
                    return Tensor()
                else:
                    sizes = List[Int](length=parts, fill=dim // parts)
            if len(sizes) != parts:
                return Tensor()
            var sum_sz = 0
            for sz in sizes:
                if sz < 0 or sz > dim - sum_sz:
                    return Tensor()
                sum_sz += sz
            if sum_sz != dim:
                return Tensor()
            var outer = 1
            for d in range(axis):
                outer *= self.values[a].shape[d]
            var inner = 1
            for d in range(axis + 1, r):
                inner *= self.values[a].shape[d]
            var px = self._dev(a)
            var start = 0
            # several outputs: the caller takes them from `split_outs`
            self.split_outs = List[Tensor]()
            for sz in sizes:
                var shape = self.values[a].shape.copy()
                shape[axis] = sz
                var t = self._dev_out(shape^)
                self.gpu.value().copy_dev(px, self._pend_ptr(), outer, sz * inner, dim * inner, start * inner,
                                          sz * inner, 0)
                self.split_outs.append(t^)
                start += sz
            var marker = Tensor()
            marker.shape = [len(sizes)]
            return marker^
        if op == OP_LAYER_NORM and self._on_gpu(a) and self.values[a].is_float():
            var r = self.values[a].rank()
            if r < 1 or norm_axis(self.nodes[k].attr_i("axis", -1), r) != r - 1:
                return Tensor()
            var eps = self.nodes[k].attr_f("epsilon", 1e-5)
            if not isfinite(eps) or eps < Float32(0.0):
                return Tensor()
            var n = self.values[a].shape[r - 1]
            if n <= 0 or self.values[a].numel() <= 0:
                return Tensor()
            if nin < 2:
                return Tensor()
            var gi = self.nodes[k].inputs[1]
            var bi = self._in(k, 2)
            if gi < 0 or gi >= len(self.values) or not self.static_value[gi] or self.values[gi].numel() != n \
                    or (bi >= 0 and (bi >= len(self.values) or not self.static_value[bi] or self.values[bi].numel() != n)):
                return Tensor()
            var pg = self._dev(gi)
            var pbb = pg
            if bi >= 0:
                pbb = self._dev(bi)
            var px = self._dev(a)
            var out = self._dev_out(self.values[a].shape.copy())
            self.gpu.value().layer_norm_dev(px, pg, pbb, self._pend_ptr(), self.values[a].numel() // n, n,
                                            eps, bi >= 0)
            return out^
        return Tensor()

    def _gpu_product(mut self, k: Int) raises -> Tensor:
        """MatMul / Gemm / Conv of node k on the GPU (static weights only); an empty tensor
        (rank 0, no data) when the node does not qualify."""
        ref node = self.nodes[k]
        if len(node.inputs) < 2 or node.inputs[0] < 0 or node.inputs[1] < 0:
            return Tensor()
        if node.inputs[0] >= len(self.values) or node.inputs[1] >= len(self.values):
            return Tensor()
        var xi = node.inputs[0]
        var wid = node.inputs[1]
        if not self.static_value[wid] or not self.values[wid].is_float() or not self.values[wid].has_valid_storage():
            return Tensor()
        ref x = self.values[xi]
        ref w = self.values[wid]
        if not x.is_float() or (not self._on_gpu(xi) and not x.has_valid_storage()):
            return Tensor()
        if node.op == OP_MATMUL:
            if w.rank() != 2 or x.rank() < 2:
                return Tensor()
            var kk = w.shape[0]
            var n = w.shape[1]
            if x.shape[x.rank() - 1] != kk or kk <= 0 or n <= 0:
                return Tensor()
            var m = 1
            for d in range(x.rank() - 1):
                var dim = x.shape[d]
                if dim <= 0 or m > 0x7FFFFFFF // dim:
                    return Tensor()
                m *= dim
            if m > 0x7FFFFFFF // max(kk, n) or m > MAX_RANGE_ELEMS // max(1, n):
                return Tensor()
            var s = x.shape.copy()
            s[len(s) - 1] = n
            if numel_of(s) < 0:
                return Tensor()
            var widx = self.gpu.value().weight(wid * 4, w.fptr(), kk, n, False)
            if self.resident:
                var pa = self._dev(xi)
                var outd = self._dev_out(s^)
                self.gpu.value().matmul_dev(pa, widx, self._pend_ptr(), 0, 0, m, kk, n)
                return outd^
            var out = Tensor.zeros(s^)
            self.gpu.value().matmul(x.fptr(), widx, out.fptr(), 0, 0, m, kk, n)
            return out^
        if node.op == OP_GEMM:
            if x.rank() != 2 or w.rank() != 2:
                return Tensor()
            if node.attr_i("transA", 0) != 0 or node.attr_f("alpha", 1.0) != 1 or node.attr_f("beta", 1.0) != 1:
                return Tensor()
            var tb = node.attr_i("transB", 0) != 0
            var m = x.shape[0]
            var kk = x.shape[1]
            var bk = w.shape[1] if tb else w.shape[0]
            var n = w.shape[0] if tb else w.shape[1]
            if bk != kk or m <= 0 or kk <= 0 or n <= 0:
                return Tensor()
            if m > 0x7FFFFFFF // max(kk, n) or m > MAX_RANGE_ELEMS // max(1, n):
                return Tensor()
            var ci = self._in(k, 2)
            var bias = 0
            var bias_key = 0
            if ci >= 0:
                if ci >= len(self.values) or not self.static_value[ci]:
                    return Tensor()
                ref c = self.values[ci]
                if not c.is_float() or not c.has_valid_storage():
                    return Tensor()
                var is_row_bias = (c.rank() == 1 and c.shape[0] == n) or (c.rank() == 2 and c.shape[0] == 1 and c.shape[1] == n)
                if not is_row_bias:
                    return Tensor()
                bias = c.fptr()
                bias_key = ci * 4 + 2
            var widx = self.gpu.value().weight(wid * 4 + (1 if tb else 0), w.fptr(), w.shape[0], w.shape[1], tb)
            if self.resident:
                var pa = self._dev(xi)
                var outd = self._dev_out([m, n])
                self.gpu.value().matmul_dev(pa, widx, self._pend_ptr(), bias, bias_key, m, kk, n)
                return outd^
            var out = Tensor.zeros([m, n])
            self.gpu.value().matmul(x.fptr(), widx, out.fptr(), bias, bias_key, m, kk, n)
            return out^
        # Conv1d: im2col^T [t, cin*kw] @ W^T [cin*kw, cout]
        if x.rank() != 3 or w.rank() != 3 or node.attr_i("group", 1) != 1:
            return Tensor()
        var strides = node.attr_ints("strides")
        if len(strides) > 0 and strides[0] != 1:
            return Tensor()
        var pads = node.attr_ints("pads")
        var dils = node.attr_ints("dilations")
        var pad0 = pads[0] if len(pads) > 0 else 0
        var pad1 = pads[1] if len(pads) > 1 else 0
        var dil = dils[0] if len(dils) > 0 else 1
        var batch = x.shape[0]
        var cin = x.shape[1]
        var t_in = x.shape[2]
        var cout = w.shape[0]
        var kw = w.shape[2]
        if w.shape[1] != cin or pad0 < 0 or pad1 < 0 or dil < 1 or kw < 1:
            return Tensor()
        if batch <= 0 or cin <= 0 or t_in < 0 or cout <= 0:
            return Tensor()
        if kw - 1 > 0 and dil > 0x7FFFFFFF // (kw - 1):
            return Tensor()
        if t_in > 0x3FFFFFFF or pad0 > 0x3FFFFFFF or pad1 > 0x3FFFFFFF:
            return Tensor()
        var t_out = t_in + pad0 + pad1 - dil * (kw - 1)
        if t_out <= 0:
            return Tensor()
        if cin > 0x7FFFFFFF // kw:
            return Tensor()
        var j = cin * kw
        if numel_of([batch, cout, t_out]) < 0 or numel_of([t_out, j]) < 0 or numel_of([t_out, cout]) < 0:
            return Tensor()
        if t_out > MAX_RANGE_ELEMS // max(j, cout) or batch > MAX_RANGE_ELEMS // max(1, cout * t_out):
            return Tensor()
        var bi = self._in(k, 2)
        var bias = 0
        var bias_key = 0
        if bi >= 0:
            if bi >= len(self.values) or not self.static_value[bi]:
                return Tensor()
            ref bv = self.values[bi]
            if not bv.is_float() or not bv.has_valid_storage() or bv.rank() != 1 or bv.shape[0] != cout:
                return Tensor()
            bias = bv.fptr()
            bias_key = bi * 4 + 2
        var widx = self.gpu.value().weight(wid * 4 + 1, w.fptr(), cout, j, True)
        var out = Tensor.zeros([batch, cout, t_out])
        var col = List[Float32](length=t_out * j, fill=0.0)
        var tmp = List[Float32](length=t_out * cout, fill=0.0)
        for b in range(batch):
            for t in range(t_out):
                for ci in range(cin):
                    for q in range(kw):
                        var src = t + q * dil - pad0
                        col[t * j + ci * kw + q] = x.fget((b * cin + ci) * t_in + src) if src >= 0 and src < t_in else 0
            self.gpu.value().matmul(Int(col.unsafe_ptr()), widx, Int(tmp.unsafe_ptr()), bias, bias_key, t_out, j, cout)
            for co in range(cout):
                for t in range(t_out):
                    out.f[(b * cout + co) * t_out + t] = tmp[t * cout + co]
        return out^

    def input_names(self) -> List[String]:
        var out = List[String]()
        for k in self.input_ids:
            out.append(self.names[k])
        return out^

    def set_input(mut self, name: String, var t: Tensor) raises:
        var found = self.ids.get(name)
        if not found:
            raise Error("ONNX: no input named " + name)
        if self.gpu:
            self._drop_slot(found.value())
        self.values[found.value()] = t^

    def output(self, name: String) raises -> Tensor:
        var found = self.ids.get(name)
        if not found:
            raise Error("ONNX: no output named " + name)
        return self.values[found.value()].copy()

    def has(self, name: String) -> Bool:
        return name in self.ids

    def weight(self, name: String) raises -> Tensor:
        """A named initializer or value without copying a mapped one (a weights-only model's
        tensors are borrowed from the file mapping)."""
        var found = self.ids.get(name)
        if not found:
            raise Error("ONNX: no tensor named " + name + " in " + self.path)
        ref t = self.values[found.value()]
        if t.ext != 0:
            return t.reshaped(t.shape.copy())
        return t.copy()

    def run(mut self) raises:
        var trace = self.trace_path.byte_length() > 0
        var lines = String()
        self.resident = Bool(self.gpu) and self.keep_on_gpu and not trace
        if self.gpu:
            self._grow_slots()
        for k in range(len(self.nodes)):
            if len(self.fuse) == len(self.nodes) and (self.fuse[k] == 1 or (self.fuse[k] == 4 and self.gpu)):
                continue
            var outs: List[Tensor]
            self.pend_slot.clear()
            self.pend_host.clear()
            var t0 = perf_counter_ns() if self.profiling else 0
            try:
                outs = self._exec(k)
            except e:
                raise Error("ONNX: node " + self.nodes[k].name + " (" + self.nodes[k].op_type + "): " + String(e))
            if self.profiling:
                if self.gpu:
                    self.gpu.value().sync()
                var key = self.nodes[k].op_type
                if len(self.fuse) == len(self.nodes):
                    if self.fuse[k] == 2:
                        key = "Attention(fused)"
                    elif self.fuse[k] == 3 and self.gpu:
                        key = "GatherMatMul(fused)"
                if (key == "Add" or key == "Mul" or key == "MatMul" or key == "Gather") \
                        and len(self.nodes[k].inputs) >= 2 and self.nodes[k].inputs[0] >= 0 and self.nodes[k].inputs[1] >= 0:
                    key += " " + shape_str(self.values[self.nodes[k].inputs[0]].shape) + shape_str(self.values[self.nodes[k].inputs[1]].shape)
                var dt = Int(perf_counter_ns() - t0)
                self.profile[key] = self.profile.get(key).or_else(0) + dt
            var j = len(outs) - 1
            while j >= 0:
                var t = outs.pop()
                var ps = self.pend_slot[j] if j < len(self.pend_slot) else -1
                if j < len(self.nodes[k].outputs) and self.nodes[k].outputs[j] >= 0:
                    var v = self.nodes[k].outputs[j]
                    if trace:
                        lines += _trace_line(self.names[v], t)
                    if self.gpu:
                        self._drop_slot(v)
                    self.values[v] = t^
                    if ps >= 0:
                        self.dslot[v] = ps
                        self.host_ok[v] = self.pend_host[j]
                elif ps >= 0:
                    self.gpu.value().release(ps)
                j -= 1
            for ii in range(len(self.nodes[k].inputs)):
                var v = self.nodes[k].inputs[ii]
                if v >= 0 and self.last_use[v] == k and not self.static_value[v]:
                    self.values[v].clear()
                    if self.gpu:
                        self._drop_slot(v)
        if self.gpu:
            for v in self.output_ids:
                self._ensure_host(v)
        if trace:
            var bytes = List[UInt8](lines.as_bytes())
            _ = write_bytes(self.trace_path, bytes)

    def _shape_only(self, k: Int) raises -> List[Int]:
        """Output shape of a Reshape / Unsqueeze / Squeeze / Flatten / Identity / Cast node."""
        ref node = self.nodes[k]
        ref x = self.values[node.inputs[0]]
        var op = node.op
        var s = List[Int]()
        if op == OP_RESHAPE:
            var want = self.values[self._req_in(k, 1)].int_list()
            var allowzero = node.attr_i("allowzero", 0) != 0
            var infer = -1
            var known = 1
            for d in range(len(want)):
                var v = want[d]
                if v == 0 and not allowzero:
                    if d >= x.rank():
                        raise Error("Reshape: 0 dim at index " + String(d) + " exceeds input rank " + String(x.rank()))
                    v = x.shape[d]
                if v == -1:
                    if infer >= 0:
                        raise Error("Reshape: multiple -1 dimensions")
                    infer = d
                    s.append(1)
                elif v < -1:
                    raise Error("Reshape: invalid negative dimension " + String(v))
                else:
                    s.append(v)
                    if v > 0 and known > 0x7FFFFFFFFFFFFFFF // v:
                        raise Error("Reshape: shape overflow")
                    known *= v
            if infer >= 0:
                if known == 0 or x.numel() % known != 0:
                    raise Error("Reshape " + shape_str(x.shape) + " -> " + shape_str(want))
                s[infer] = x.numel() // known
            if numel_of(s) != x.numel():
                raise Error("Reshape " + shape_str(x.shape) + " -> " + shape_str(s))
        elif op == OP_UNSQUEEZE:
            var axes = node.attr_ints("axes")
            if self._in(k, 1) >= 0:
                axes = self.values[self._in(k, 1)].int_list()
            elif not axes:
                raise Error("Unsqueeze needs axes")
            var r = x.rank() + len(axes)
            var ins = List[Bool](length=r, fill=False)
            for a in axes:
                var na = norm_axis(a, r)
                if na < 0 or na >= r:
                    raise Error("Unsqueeze: axis " + String(a) + " out of range for rank " + String(r))
                if ins[na]:
                    raise Error("Unsqueeze: duplicate axis " + String(a))
                ins[na] = True
            var src = 0
            for d in range(r):
                if ins[d]:
                    s.append(1)
                else:
                    s.append(x.shape[src])
                    src += 1
        elif op == OP_SQUEEZE:
            var axes = node.attr_ints("axes")
            var ai = self._in(k, 1)
            if ai >= 0:
                axes = self.values[ai].int_list()
            var seen = List[Bool](length=x.rank(), fill=False)
            for a in axes:
                var na = norm_axis(a, x.rank())
                if na < 0 or na >= x.rank():
                    raise Error("Squeeze: axis " + String(a) + " out of range for rank " + String(x.rank()))
                if seen[na]:
                    raise Error("Squeeze: duplicate axis " + String(a))
                if x.shape[na] != 1:
                    raise Error("Squeeze: axis " + String(a) + " has size " + String(x.shape[na]) + " != 1")
                seen[na] = True
            for d in range(x.rank()):
                var drop = (x.shape[d] == 1) if len(axes) == 0 else seen[d]
                if not drop:
                    s.append(x.shape[d])
        elif op == OP_FLATTEN:
            var axis = norm_axis(node.attr_i("axis", 1), x.rank())
            if axis < 0 or axis > x.rank():
                raise Error("Flatten: axis out of range")
            var a = 1
            for d in range(axis):
                a *= x.shape[d]
            s.append(a)
            s.append(x.numel() // a if a > 0 else 0)
        else:
            s = x.shape.copy()
        return s^

    def _in(self, k: Int, j: Int) -> Int:
        """Value id of input j of node k, or -1 if it is absent."""
        if j < 0 or j >= len(self.nodes[k].inputs):
            return -1
        return self.nodes[k].inputs[j]

    def _req_in(self, k: Int, j: Int) raises -> Int:
        """Value id of required input j of node k, raising if absent or out of range."""
        var idx = self._in(k, j)
        if idx < 0 or idx >= len(self.values):
            raise Error("ONNX: missing required input " + String(j))
        return idx

    def _exec(mut self, k: Int) raises -> List[Tensor]:
        var out = List[Tensor]()
        var op0 = self.nodes[k].op
        var has_fuse = len(self.fuse) == len(self.nodes)
        if self.gpu and op0 == OP_MATMUL and has_fuse and self.fuse[k] == 3:
            out.append(self._gpu_gather_matmul(k))
            return out^
        if self.gpu and (op0 == OP_MATMUL or op0 == OP_GEMM or op0 == OP_CONV):
            if op0 == OP_CONV and len(self.nodes[k].inputs) > 0 and self.nodes[k].inputs[0] >= 0:
                self._ensure_host(self.nodes[k].inputs[0])
            var g = self._gpu_product(k)
            if len(g.shape) > 0:
                out.append(g^)
                return out^
        var in0_opt = self.nodes[k].inputs[0] if len(self.nodes[k].inputs) > 0 else -1
        if in0_opt >= 0 and in0_opt < len(self.values) and self.last_use[in0_opt] == k \
                and not self.static_value[in0_opt] and self.values[in0_opt].ext == 0 \
                and (op0 == OP_RESHAPE or op0 == OP_UNSQUEEZE or op0 == OP_SQUEEZE or op0 == OP_FLATTEN
                     or op0 == OP_IDENTITY or (op0 == OP_CAST and self.values[in0_opt].is_float()
                                               and self.nodes[k].attr_i("to", DT_F32) == DT_F32)):
            if self.resident and len(self.nodes[k].inputs) > 1 and self.nodes[k].inputs[1] >= 0:
                self._ensure_host(self.nodes[k].inputs[1])
            var shape = self._shape_only(k)
            if self._on_gpu(in0_opt):
                # the device buffer moves with the value
                self.pend_slot.append(self.dslot[in0_opt])
                self.pend_host.append(self.host_ok[in0_opt])
                self.dslot[in0_opt] = -1
                self.host_ok[in0_opt] = True
            self.values.append(Tensor())
            self.values.swap_elements(in0_opt, len(self.values) - 1)
            var t = self.values.pop()
            t.shape = shape^
            out.append(t^)
            return out^
        if op0 == OP_EINSUM and has_fuse and self.fuse[k] == 2 and self.gpu:
            var g = self._gpu_attention(k)
            if g.rank() > 0:
                out.append(g^)
                return out^
        if self.resident:
            var ge = self._gpu_elementwise(k)
            if ge.rank() > 0:
                if op0 == OP_SPLIT:
                    for j in range(len(self.split_outs)):
                        out.append(self.split_outs[j].copy())
                    self.split_outs = List[Tensor]()
                    return out^
                out.append(ge^)
                return out^
            # CPU op: bring its device-resident inputs back first (Shape reads only the shape)
            if op0 != OP_SHAPE:
                for j in range(len(self.nodes[k].inputs)):
                    self._ensure_host(self.nodes[k].inputs[j])
            if has_fuse and self.fuse[k] == 2:
                self._ensure_host(self.fuse_q[k])
                self._ensure_host(self.fuse_k[k])
        ref node = self.nodes[k]
        var op = node.op
        if op == OP_EINSUM and has_fuse and self.fuse[k] == 2:
            var att = fused_attention(self.pool, self.values[self.fuse_q[k]], self.values[self.fuse_k[k]],
                                      self.values[self._req_in(k, 1)], self.fuse_scale[k])
            if att.rank() == 0:
                raise Error("fused attention: unsupported operand shapes")
            out.append(att^)
            return out^
        if op == OP_CONCAT:
            var xs = List[Tensor]()
            for v in node.inputs:
                if v >= 0 and v < len(self.values) and (self.values[v].numel() > 0 or len(node.inputs) == 1):
                    xs.append(self.values[v].copy())
            if len(xs) == 0:
                raise Error("ONNX: Concat missing valid inputs")
            out.append(concat(xs, node.attr_i("axis", 0)))
            return out^
        var in0 = self._req_in(k, 0)
        ref x = self.values[in0]
        if op >= OP_BINARY:
            var code = op - OP_BINARY
            var acc = binary(code, x, self.values[self._req_in(k, 1)])
            for j in range(2, len(node.inputs)):
                acc = binary(code, acc, self.values[self._req_in(k, j)])
            out.append(acc^)
        elif op >= OP_UNARY:
            out.append(unary(op - OP_UNARY, x, node.attr_f("alpha", 0.01)))
        elif op == OP_SPLIT:
            var axis = norm_axis(node.attr_i("axis", 0), x.rank())
            var count = len(node.outputs)
            if axis < 0 or axis >= x.rank() or count < 1:
                raise Error("Split: invalid axis or output count")
            var sizes = node.attr_ints("split")
            if len(node.inputs) > 1 and node.inputs[1] >= 0:
                sizes = self.values[node.inputs[1]].int_list()
            if len(sizes) == 0:
                if node.attr_i("num_outputs", 0) != 0:
                    if node.attr_i("num_outputs", 0) != count:
                        raise Error("Split: num_outputs mismatches graph outputs")
                    var dim = x.shape[axis]
                    var base = dim // count
                    var rem = dim % count
                    sizes = List[Int](capacity=count)
                    for si in range(count):
                        sizes.append(base + 1 if si < rem else base)
                elif x.shape[axis] % count != 0:
                    raise Error("Split: uneven chunks require explicit sizes")
                else:
                    sizes = List[Int](length=count, fill=x.shape[axis] // count)
            if len(sizes) != count:
                raise Error("Split: incorrect number of sizes")
            var start = 0
            for size in sizes:
                if size < 0 or size > x.shape[axis] - start:
                    raise Error("Split: size out of bounds")
                out.append(slice(x, [start], [start + size], [axis], [1]))
                start += size
            if start != x.shape[axis]:
                raise Error("Split: sizes do not cover the input")
        elif op == OP_SHAPE:
            var s = List[Int]()
            for d in x.shape:
                s.append(d)
            out.append(Tensor.ints(DT_I64, [len(s)], s^))
        elif op == OP_UNSQUEEZE:
            var axes = node.attr_ints("axes")
            if self._in(k, 1) >= 0:
                axes = self.values[self._in(k, 1)].int_list()
            elif not axes:
                raise Error("Unsqueeze needs axes")
            var r = x.rank() + len(axes)
            var ins = List[Bool](length=r, fill=False)
            for a in axes:
                var na = norm_axis(a, r)
                if na < 0 or na >= r:
                    raise Error("Unsqueeze: axis " + String(a) + " out of range for rank " + String(r))
                if ins[na]:
                    raise Error("Unsqueeze: duplicate axis " + String(a))
                ins[na] = True
            var s = List[Int]()
            var src = 0
            for d in range(r):
                if ins[d]:
                    s.append(1)
                else:
                    s.append(x.shape[src])
                    src += 1
            out.append(x.reshaped(s^))
        elif op == OP_SQUEEZE:
            var axes = node.attr_ints("axes")
            var ai = self._in(k, 1)
            if ai >= 0:
                axes = self.values[ai].int_list()
            var seen = List[Bool](length=x.rank(), fill=False)
            for a in axes:
                var na = norm_axis(a, x.rank())
                if na < 0 or na >= x.rank():
                    raise Error("Squeeze: axis " + String(a) + " out of range for rank " + String(x.rank()))
                if seen[na]:
                    raise Error("Squeeze: duplicate axis " + String(a))
                if x.shape[na] != 1:
                    raise Error("Squeeze: axis " + String(a) + " has size " + String(x.shape[na]) + " != 1")
                seen[na] = True
            var s = List[Int]()
            for d in range(x.rank()):
                var drop = (x.shape[d] == 1) if len(axes) == 0 else seen[d]
                if not drop:
                    s.append(x.shape[d])
            out.append(x.reshaped(s^))
        elif op == OP_RESHAPE:
            var want = self.values[self._req_in(k, 1)].int_list()
            var allowzero = node.attr_i("allowzero", 0) != 0
            var s = List[Int]()
            var infer = -1
            var known = 1
            for d in range(len(want)):
                var v = want[d]
                if v == 0 and not allowzero:
                    if d >= x.rank():
                        raise Error("Reshape: 0 dim at index " + String(d) + " exceeds input rank " + String(x.rank()))
                    v = x.shape[d]
                if v == -1:
                    if infer >= 0:
                        raise Error("Reshape: multiple -1 dimensions")
                    infer = d
                    s.append(1)
                elif v < -1:
                    raise Error("Reshape: invalid negative dimension " + String(v))
                else:
                    s.append(v)
                    if v > 0 and known > 0x7FFFFFFFFFFFFFFF // v:
                        raise Error("Reshape: shape overflow")
                    known *= v
            if infer >= 0:
                if known == 0 or x.numel() % known != 0:
                    raise Error("Reshape " + shape_str(x.shape) + " -> " + shape_str(want))
                s[infer] = x.numel() // known
            if numel_of(s) != x.numel():
                raise Error("Reshape " + shape_str(x.shape) + " -> " + shape_str(s))
            out.append(x.reshaped(s^))
        elif op == OP_FLATTEN:
            var axis = norm_axis(node.attr_i("axis", 1), x.rank())
            if axis < 0 or axis > x.rank():
                raise Error("Flatten: axis out of range")
            var a = 1
            for d in range(axis):
                a *= x.shape[d]
            out.append(x.reshaped([a, x.numel() // a if a > 0 else 0]))
        elif op == OP_GATHER:
            out.append(gather(x, self.values[self._req_in(k, 1)], node.attr_i("axis", 0)))
        elif op == OP_GATHER_ELEMENTS:
            out.append(gather_elements(x, self.values[self._req_in(k, 1)], node.attr_i("axis", 0)))
        elif op == OP_SCATTER_ELEMENTS:
            if node.attr_s("reduction").byte_length() > 0 and node.attr_s("reduction") != "none":
                raise Error("ScatterElements reduction is not supported")
            out.append(scatter_elements(x, self.values[self._req_in(k, 1)], self.values[self._req_in(k, 2)],
                                        node.attr_i("axis", 0)))
        elif op == OP_SCATTER_ND:
            out.append(scatter_nd(x, self.values[self._req_in(k, 1)], self.values[self._req_in(k, 2)]))
        elif op == OP_TRANSPOSE:
            var perm = node.attr_ints("perm")
            if len(perm) == 0:
                for d in range(x.rank()):
                    perm.append(x.rank() - 1 - d)
            out.append(transpose(x, perm))
        elif op == OP_SLICE:
            var axes = List[Int]()
            var steps = List[Int]()
            var ai = self._in(k, 3)
            if ai >= 0:
                axes = self.values[ai].int_list()
            var si = self._in(k, 4)
            if si >= 0:
                steps = self.values[si].int_list()
            out.append(slice(x, self.values[self._req_in(k, 1)].int_list(), self.values[self._req_in(k, 2)].int_list(),
                             axes, steps))
        elif op == OP_CAST:
            out.append(cast(x, node.attr_i("to", DT_F32)))
        elif op == OP_CONSTANT_OF_SHAPE:
            if not x.has_valid_storage():
                raise Error("ConstantOfShape: invalid shape tensor storage")
            var s = x.int_list()
            for dim in s:
                if dim < 0:
                    raise Error("ConstantOfShape: negative dimension " + String(dim))
            var vk = node.find("value")
            var n = numel_of(s)
            if n < 0 or n > MAX_RANGE_ELEMS:
                raise Error("ConstantOfShape: shape overflow or exceeds maximum element count")
            if vk >= 0 and node.attrs[vk].has_t:
                ref vt = node.attrs[vk].t
                if vt.numel() != 1:
                    raise Error("ConstantOfShape: value tensor must have 1 element")
                if not vt.is_float():
                    if len(vt.i) == 0:
                        raise Error("ConstantOfShape: empty integer value tensor")
                    out.append(Tensor.ints(vt.dtype, s^, List[Int](length=n, fill=vt.i[0])))
                else:
                    if vt.ext == 0 and len(vt.f) == 0:
                        raise Error("ConstantOfShape: empty float value tensor")
                    out.append(Tensor.floats(s^, List[Float32](length=n, fill=vt.fget(0))))
            else:
                out.append(Tensor.floats(s^, List[Float32](length=n, fill=0.0)))
        elif op == OP_WHERE:
            out.append(where(x, self.values[self._req_in(k, 1)], self.values[self._req_in(k, 2)]))
        elif op == OP_EXPAND:
            out.append(expand(x, self.values[self._req_in(k, 1)].int_list()))
        elif op == OP_TILE:
            out.append(tile(x, self.values[self._req_in(k, 1)].int_list()))
        elif op == OP_RANGE:
            out.append(range_op(x, self.values[self._req_in(k, 1)], self.values[self._req_in(k, 2)]))
        elif op == OP_MATMUL:
            out.append(matmul(self.pool, x, self.values[self._req_in(k, 1)]))
        elif op == OP_GEMM:
            var ci = self._in(k, 2)
            out.append(gemm(self.pool, x, self.values[self._req_in(k, 1)], self.values[ci if ci >= 0 else in0],
                            ci >= 0, node.attr_f("alpha", 1.0), node.attr_f("beta", 1.0),
                            node.attr_i("transA", 0) != 0, node.attr_i("transB", 0) != 0))
        elif op == OP_CONV:
            var pads = node.attr_ints("pads")
            var dil = node.attr_ints("dilations")
            var strides = node.attr_ints("strides")
            var bi = self._in(k, 2)
            out.append(conv1d(self.pool, x, self.values[self._req_in(k, 1)], self.values[bi if bi >= 0 else in0],
                              bi >= 0, pads[0] if len(pads) > 0 else 0, pads[1] if len(pads) > 1 else 0,
                              dil[0] if len(dil) > 0 else 1, strides[0] if len(strides) > 0 else 1,
                              node.attr_i("group", 1)))
        elif op == OP_LAYER_NORM:
            if x.rank() == 0 or norm_axis(node.attr_i("axis", -1), x.rank()) != x.rank() - 1:
                raise Error("LayerNormalization: only the last axis is supported")
            var bi = self._in(k, 2)
            out.append(layer_norm(x, self.values[self._req_in(k, 1)], self.values[bi if bi >= 0 else in0],
                                  bi >= 0, node.attr_f("epsilon", 1e-5)))
        elif op == OP_SOFTMAX:
            if x.rank() > 0 and norm_axis(node.attr_i("axis", -1), x.rank()) == x.rank() - 1 \
                    and x.is_float() and x.numel() >= 1024:
                out.append(softmax_rows(self.pool, x))
            else:
                out.append(softmax(x, node.attr_i("axis", -1)))
        elif op == OP_REDUCE_L2:
            out.append(reduce(x, node.attr_ints("axes"), node.attr_i("keepdims", 1) != 0, True))
        elif op == OP_REDUCE_SUM:
            var axes = node.attr_ints("axes")  # opsets before 13 use the attribute
            var ai = self._in(k, 1)
            if ai >= 0:
                axes = self.values[ai].int_list()
            if len(axes) == 0 and node.attr_i("noop_with_empty_axes", 0) != 0:
                out.append(x.reshaped(x.shape.copy()))
            else:
                out.append(reduce(x, axes, node.attr_i("keepdims", 1) != 0, False))
        elif op == OP_ARGMAX:
            if node.attr_i("select_last_index", 0) != 0:
                raise Error("ArgMax select_last_index is not supported")
            out.append(argmax(x, node.attr_i("axis", 0), node.attr_i("keepdims", 1) != 0))
        elif op == OP_TOPK:
            ref kt = self.values[self._req_in(k, 1)]
            if not kt.has_valid_storage() or kt.numel() == 0:
                raise Error("TopK: empty or invalid k tensor")
            var kk = kt.int_at(0)
            var res = topk(x, kk, node.attr_i("axis", -1), node.attr_i("largest", 1) != 0)
            for t in res:
                out.append(t.copy())
        elif op == OP_CUMSUM:
            ref at = self.values[self._req_in(k, 1)]
            if not at.has_valid_storage() or at.numel() == 0:
                raise Error("CumSum: empty or invalid axis tensor")
            out.append(cumsum(x, at.int_at(0), node.attr_i("exclusive", 0) != 0,
                              node.attr_i("reverse", 0) != 0))
        elif op == OP_ONEHOT:
            if norm_axis(node.attr_i("axis", -1), x.rank() + 1) != x.rank():
                raise Error("OneHot: only axis=-1 is supported")
            ref dt = self.values[self._req_in(k, 1)]
            if not dt.has_valid_storage() or dt.numel() == 0:
                raise Error("OneHot: empty or invalid depth tensor")
            out.append(onehot(x, dt.int_at(0), self.values[self._req_in(k, 2)]))
        elif op == OP_RESIZE:
            if node.attr_s("mode") != "nearest" or node.attr_s("coordinate_transformation_mode") != "asymmetric" \
                    or node.attr_s("nearest_mode") != "floor":
                raise Error("Resize: only nearest/floor/asymmetric is supported")
            var s = List[Int]()
            var zi = self._in(k, 3)
            var si = self._in(k, 2)
            if zi >= 0 and self.values[zi].numel() > 0:
                if not self.values[zi].has_valid_storage():
                    raise Error("Resize: sizes tensor has invalid storage")
                if self.values[zi].numel() != x.rank():
                    raise Error("Resize: sizes length must match input rank")
                if self.values[zi].is_float():
                    for d in range(x.rank()):
                        var sz_f = self.values[zi].get(d)
                        if not isfinite(sz_f) or sz_f < 0.0 or sz_f > Float64(MAX_RANGE_ELEMS):
                            raise Error("Resize: invalid output size")
                        s.append(Int(sz_f))
                else:
                    s = self.values[zi].int_list()
                    for d in range(len(s)):
                        if s[d] < 0 or s[d] > MAX_RANGE_ELEMS:
                            raise Error("Resize: invalid output size")
            elif si >= 0 and self.values[si].numel() >= x.rank():
                if not self.values[si].has_valid_storage():
                    raise Error("Resize: scales tensor has invalid storage")
                if not self.values[si].is_float():
                    raise Error("Resize: scales must be float")
                for d in range(x.rank()):
                    var sc = self.values[si].get(d)
                    if not isfinite(sc) or sc <= 0.0:
                        raise Error("Resize: scales must be finite and positive")
                    var dim_f = Float64(x.shape[d]) * sc
                    if not isfinite(dim_f) or dim_f < 0.0 or dim_f > Float64(MAX_RANGE_ELEMS):
                        raise Error("Resize: invalid scaled output dimension")
                    s.append(Int(dim_f))
            else:
                raise Error("Resize needs scales or sizes")
            var out_n = numel_of(s)
            if out_n < 0 or out_n > MAX_RANGE_ELEMS:
                raise Error("Resize: output element count overflow or exceeds maximum limit")
            out.append(resize_nearest(x, s))
        elif op == OP_CLIP:
            var lo = -3.4e38
            var hi = 3.4e38
            var a = self._in(k, 1)
            if a >= 0 and self.values[a].numel() > 0:
                if not self.values[a].has_valid_storage():
                    raise Error("Clip: min tensor has invalid storage")
                lo = self.values[a].get(0)
            var b = self._in(k, 2)
            if b >= 0 and self.values[b].numel() > 0:
                if not self.values[b].has_valid_storage():
                    raise Error("Clip: max tensor has invalid storage")
                hi = self.values[b].get(0)
            out.append(clip(x, lo, hi))
        elif op == OP_IDENTITY:
            out.append(x.reshaped(x.shape.copy()) if x.ext != 0 else x.copy())
        elif op == OP_EINSUM:
            out.append(einsum(self.pool, node.attr_s("equation"), x, self.values[self._req_in(k, 1)]))
        elif op == OP_BATCH_NORM:
            out.append(batch_norm(x, self.values[self._req_in(k, 1)], self.values[self._req_in(k, 2)],
                                  self.values[self._req_in(k, 3)], self.values[self._req_in(k, 4)],
                                  node.attr_f("epsilon", 1e-5)))
        else:
            raise Error("operator not implemented")
        return out^


def _attention_equations(e1: String, e2: String) -> Bool:
    """e1 = "X L H D, X S H D -> X H L S" and e2 = "X H L S, X S H D -> X L H D" (X: optional batch
    labels), each with its own letters."""
    var p1 = e1.replace(" ", "").split("->")
    var p2 = e2.replace(" ", "").split("->")
    if len(p1) != 2 or len(p2) != 2:
        return False
    var i1 = p1[0].split(",")
    var i2 = p2[0].split(",")
    if len(i1) != 2 or len(i2) != 2:
        return False
    var q = String(i1[0])
    var kk = String(i1[1])
    var o1 = String(p1[1])
    var a = String(i2[0])
    var v = String(i2[1])
    var o2 = String(p2[1])
    var r = q.byte_length()
    if r < 3 or kk.byte_length() != r or o1.byte_length() != r or a.byte_length() != r or v.byte_length() != r \
            or o2.byte_length() != r:
        return False
    var nb = r - 3
    var X = String(q[byte=0:nb])
    if String(kk[byte=0:nb]) != X or String(o1[byte=0:nb]) != X:
        return False
    var L = String(q[byte=nb])
    var H = String(q[byte=nb + 1])
    var D = String(q[byte=nb + 2])
    var S = String(kk[byte=nb])
    if String(kk[byte=nb + 1]) != H or String(kk[byte=nb + 2]) != D or L == S or L == H or L == D or S == H or S == D or H == D:
        return False
    if o1 != X + H + L + S:
        return False
    var X2 = String(a[byte=0:nb])
    if String(v[byte=0:nb]) != X2 or String(o2[byte=0:nb]) != X2:
        return False
    var H2 = String(a[byte=nb])
    var L2 = String(a[byte=nb + 1])
    var S2 = String(a[byte=nb + 2])
    if String(v[byte=nb]) != S2 or String(v[byte=nb + 1]) != H2:
        return False
    var D2 = String(v[byte=nb + 2])
    return o2 == X2 + L2 + H2 + D2 and L2 != S2 and H2 != D2 and L2 != H2 and S2 != D2


def _trace_line(name: String, t: Tensor) -> String:
    var s = 0.0
    var a = 0.0
    for e in range(t.numel()):
        var v = t.get(e)
        s += v
        a += v if v >= 0 else -v
    return name + "\t" + String(t.dtype) + "\t" + shape_str(t.shape) + "\t" + String(s) + "\t" + String(a) + "\n"
