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
from flint.ml.gpu import GpuMatmul
from flint.ml.tensor import Tensor, DT_F32, DT_I32, DT_I64, DT_BOOL, numel_of, shape_str, norm_axis
from flint.ml.ops import (
    MAX_RANGE_ELEMS,
    unary, binary, where, clip, cast, transpose, concat, gather, gather_elements, scatter_elements,
    scatter_nd, slice, expand, tile, range_op, onehot, resize_nearest, reduce, argmax, topk, cumsum,
    softmax, layer_norm, matmul, gemm, conv1d,
    U_SQRT, U_RELU, U_ATAN, U_COS, U_SIN, U_ABS, U_NEG, U_NOT, U_LEAKY, U_ERF, U_SIGMOID,
    B_ADD, B_SUB, B_MUL, B_DIV, B_MOD, B_MAX, B_EQ, B_LT, B_GT, B_AND,
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
comptime OP_UNARY: Int = 100   # + U_* code
comptime OP_BINARY: Int = 200  # + B_* code


def op_code(name: String) -> Int:
    """The interpreter's code for an ONNX op type, or -1 when it is not supported."""
    if name == "Split": return OP_SPLIT
    if name == "Erf": return OP_UNARY + U_ERF
    if name == "Sigmoid": return OP_UNARY + U_SIGMOID
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


def _parse_tensor(mut r: PbReader) raises -> Tuple[String, Tensor]:
    """A TensorProto. Float raw data stays in the mapping (borrowed); the rest is copied."""
    var name = String()
    var dims = List[Int]()
    var dtype = DT_F32
    var raw_start = -1
    var raw_end = -1
    var floats = List[Float32]()
    var ints = List[Int]()
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
        elif r.field == 14:
            if r.varint() == 1:
                raise Error("ONNX: external tensor data is not supported (" + name + ")")
        else:
            r.skip()
    var n = numel_of(dims)
    if n < 0:
        raise Error("ONNX: tensor shape overflow for " + name)
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
        _ = m.pool.start(default_worker_count() if workers < 0 else workers)
        return m^

    def _parse_graph(mut self, mut g: PbReader) raises:
        var graph_inputs = List[String]()
        while g.more():
            g.key()
            if g.field == 1:
                var nr = g.sub()
                self._parse_node(nr)
            elif g.field == 5:
                var tr = g.sub()
                var nt = _parse_tensor(tr)
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

    def _gpu_product(mut self, k: Int) raises -> Tensor:
        """MatMul / Gemm / Conv of node k on the GPU (static weights only); an empty tensor
        (rank 0, no data) when the node does not qualify."""
        ref node = self.nodes[k]
        if len(node.inputs) < 2 or node.inputs[0] < 0 or node.inputs[1] < 0:
            return Tensor()
        if node.inputs[0] >= len(self.values) or node.inputs[1] >= len(self.values):
            return Tensor()
        var wid = node.inputs[1]
        if not self.static_value[wid] or not self.values[wid].is_float() or not self.values[wid].has_valid_storage():
            return Tensor()
        ref x = self.values[node.inputs[0]]
        ref w = self.values[wid]
        if not x.is_float() or not x.has_valid_storage():
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
        self.values[found.value()] = t^

    def output(self, name: String) raises -> Tensor:
        var found = self.ids.get(name)
        if not found:
            raise Error("ONNX: no output named " + name)
        return self.values[found.value()].copy()

    def run(mut self) raises:
        var trace = self.trace_path.byte_length() > 0
        var lines = String()
        for k in range(len(self.nodes)):
            var outs: List[Tensor]
            var t0 = perf_counter_ns() if self.profiling else 0
            try:
                outs = self._exec(k)
            except e:
                raise Error("ONNX: node " + self.nodes[k].name + " (" + self.nodes[k].op_type + "): " + String(e))
            if self.profiling:
                var key = self.nodes[k].op_type
                if (key == "Add" or key == "Mul") and len(self.nodes[k].inputs) >= 2 and self.nodes[k].inputs[0] >= 0 and self.nodes[k].inputs[1] >= 0:
                    key += " " + shape_str(self.values[self.nodes[k].inputs[0]].shape) + shape_str(self.values[self.nodes[k].inputs[1]].shape)
                var dt = Int(perf_counter_ns() - t0)
                self.profile[key] = self.profile.get(key).or_else(0) + dt
            var j = len(outs) - 1
            while j >= 0:
                var t = outs.pop()
                if j < len(self.nodes[k].outputs) and self.nodes[k].outputs[j] >= 0:
                    var v = self.nodes[k].outputs[j]
                    if trace:
                        lines += _trace_line(self.names[v], t)
                    self.values[v] = t^
                j -= 1
            for v in self.nodes[k].inputs:
                if v >= 0 and self.last_use[v] == k and not self.static_value[v]:
                    self.values[v].clear()
        if trace:
            var bytes = List[UInt8](lines.as_bytes())
            _ = write_bytes(self.trace_path, bytes)

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
        if self.gpu and (op0 == OP_MATMUL or op0 == OP_GEMM or op0 == OP_CONV):
            var g = self._gpu_product(k)
            if len(g.shape) > 0:
                out.append(g^)
                return out^
        ref node = self.nodes[k]
        var op = node.op
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
        else:
            raise Error("operator not implemented")
        return out^


def _trace_line(name: String, t: Tensor) -> String:
    var s = 0.0
    var a = 0.0
    for e in range(t.numel()):
        var v = t.get(e)
        s += v
        a += v if v >= 0 else -v
    return name + "\t" + String(t.dtype) + "\t" + shape_str(t.shape) + "\t" + String(s) + "\t" + String(a) + "\n"
