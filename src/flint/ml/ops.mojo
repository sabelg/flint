"""CPU kernels for the ONNX operators the interpreter supports (opset 17 semantics).

Heavy kernels (MatMul, Gemm, Conv) run on a `JobPool`, split by output columns, with 8-wide
SIMD inner loops; everything else is a plain loop, which is cheap for the small activations
of a motion model.
"""

from std.math import sqrt, exp, erf, atan, cos, sin, ceil, floor, isnan, isinf, isfinite
from std.memory import bitcast
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import (
    Tensor, DT_F32, DT_I32, DT_I64, DT_BOOL, numel_of, strides_of, shape_str,
    broadcast_shape, broadcast_offsets, norm_axis, fptr_at,
)

comptime LANES: Int = 8
comptime F8 = SIMD[DType.float32, LANES]
comptime MAX_RANGE_ELEMS: Int = 16_777_216


@always_inline
def _checked_mul(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0:
        raise Error("ops: negative dimension")
    if a == 0 or b == 0:
        return 0
    if a > 0x7FFFFFFFFFFFFFFF // b:
        raise Error("ops: integer multiplication overflow")
    return a * b


@always_inline
def _checked_add(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0:
        raise Error("ops: negative dimension")
    if a > 0x7FFFFFFFFFFFFFFF - b:
        raise Error("ops: integer addition overflow")
    return a + b


def _require_valid_tensor(t: Tensor, op_name: String) raises:
    if not t.has_valid_storage():
        raise Error(op_name + ": invalid tensor shape or undersized storage")


# ----------------------------------------------------------------------------------- unary

comptime U_SQRT: Int = 0
comptime U_RELU: Int = 1
comptime U_ATAN: Int = 2
comptime U_COS: Int = 3
comptime U_SIN: Int = 4
comptime U_ABS: Int = 5
comptime U_NEG: Int = 6
comptime U_NOT: Int = 7
comptime U_LEAKY: Int = 8
comptime U_ERF: Int = 9
comptime U_SIGMOID: Int = 10


def unary(op: Int, x: Tensor, alpha: Float32 = 0.01) raises -> Tensor:
    _require_valid_tensor(x, "unary")
    var n = x.numel()
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("unary: tensor shape overflow or exceeds maximum limit")
    if op == U_NOT:
        if x.is_float():
            raise Error("unary Not requires integer/bool tensor")
        var o = List[Int](length=n, fill=0)
        for k in range(n):
            o[k] = 1 if x.i[k] == 0 else 0
        return Tensor.ints(DT_BOOL, x.shape.copy(), o^)
    if not x.is_float():
        var o = List[Int](length=n, fill=0)
        for k in range(n):
            var v = x.i[k]
            if op == U_ABS:
                o[k] = -v if v < 0 else v
            elif op == U_NEG:
                o[k] = -v
            elif op == U_RELU:
                o[k] = max(v, 0)
            else:
                raise Error("unary op " + String(op) + " on an integer tensor")
        return Tensor.ints(x.dtype, x.shape.copy(), o^)
    var o = List[Float32](length=n, fill=0.0)
    if op == U_RELU and x.ext == 0:
        var px = x.f.unsafe_ptr()
        var po = o.unsafe_ptr()
        var e = 0
        while e + LANES <= n:
            po.unsafe_offset(e).unsafe_store(max(px.unsafe_load[width=LANES](e), F8(0)))
            e += LANES
        while e < n:
            o[e] = max(x.f[e], 0)
            e += 1
        return Tensor.floats(x.shape.copy(), o^)
    for k in range(n):
        var v = x.fget(k)
        var r: Float32
        if op == U_SQRT:
            r = sqrt(v)
        elif op == U_RELU:
            r = v if v > 0 else 0
        elif op == U_ATAN:
            r = atan(v)
        elif op == U_ERF:
            r = erf(v)
        elif op == U_SIGMOID:
            var z = exp(-abs(v))
            r = 1 / (1 + z) if v >= 0 else z / (1 + z)
        elif op == U_COS:
            r = cos(v)
        elif op == U_SIN:
            r = sin(v)
        elif op == U_ABS:
            r = -v if v < 0 else v
        elif op == U_NEG:
            r = -v
        else:
            r = v if v >= 0 else v * alpha
        o[k] = r
    return Tensor.floats(x.shape.copy(), o^)


# ---------------------------------------------------------------------------------- binary

comptime B_ADD: Int = 0
comptime B_SUB: Int = 1
comptime B_MUL: Int = 2
comptime B_DIV: Int = 3
comptime B_MOD: Int = 4
comptime B_MAX: Int = 5
comptime B_EQ: Int = 6
comptime B_LT: Int = 7
comptime B_GT: Int = 8
comptime B_AND: Int = 9


@always_inline
def _fmod_f32(a: Float32, b: Float32) -> Float32:
    """Exact pure-Mojo IEEE-754 float32 remainder (`fmodf`) without float-to-int overflow UB."""
    var hx = bitcast[DType.uint32](a)
    var hy = bitcast[DType.uint32](b)
    var sx = hx & UInt32(0x80000000)
    hx &= UInt32(0x7FFFFFFF)
    hy &= UInt32(0x7FFFFFFF)
    if hy == 0 or hx >= UInt32(0x7F800000) or hy > UInt32(0x7F800000):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if hx < hy:
        return a
    if hx == hy:
        return bitcast[DType.float32](sx)
    var ix: Int
    var mx: UInt32
    if hx < UInt32(0x00800000):
        ix = -126
        mx = hx
        while mx < UInt32(0x00800000):
            mx <<= UInt32(1)
            ix -= 1
    else:
        ix = Int(hx >> UInt32(23)) - 127
        mx = (hx & UInt32(0x007FFFFF)) | UInt32(0x00800000)
    var iy: Int
    var my: UInt32
    if hy < UInt32(0x00800000):
        iy = -126
        my = hy
        while my < UInt32(0x00800000):
            my <<= UInt32(1)
            iy -= 1
    else:
        iy = Int(hy >> UInt32(23)) - 127
        my = (hy & UInt32(0x007FFFFF)) | UInt32(0x00800000)

    var n = ix - iy
    while n > 0:
        var hz = Int64(mx) - Int64(my)
        if hz < 0:
            mx = mx + mx
        else:
            if hz == 0:
                return bitcast[DType.float32](sx)
            mx = UInt32(hz) + UInt32(hz)
        n -= 1
    var hz = Int64(mx) - Int64(my)
    if hz >= 0:
        mx = UInt32(hz)
    if mx == 0:
        return bitcast[DType.float32](sx)
    while mx < UInt32(0x00800000):
        mx = mx + mx
        iy -= 1
    if iy >= -126:
        mx = (mx - UInt32(0x00800000)) | (UInt32(iy + 127) << UInt32(23))
    else:
        var shift = -126 - iy
        if shift >= 32:
            mx = 0
        else:
            mx >>= UInt32(shift)
    return bitcast[DType.float32](sx | mx)


@always_inline
def _fop(op: Int, a: Float32, b: Float32) -> Float32:
    if op == B_ADD:
        return a + b
    if op == B_SUB:
        return a - b
    if op == B_MUL:
        return a * b
    if op == B_DIV:
        return a / b
    if op == B_MAX:
        return a if a >= b else b
    return _fmod_f32(a, b)


@always_inline
def _iop(op: Int, a: Int, b: Int) -> Int:
    if op == B_ADD:
        return a + b
    if op == B_SUB:
        return a - b
    if op == B_MUL:
        return a * b
    if op == B_DIV:
        if b == -1 and a == Int(-0x7FFFFFFFFFFFFFFF - 1):
            return a
        # C++ integer division truncates toward zero
        var q = a // b
        if (a % b != 0) and ((a < 0) != (b < 0)):
            q += 1
        return q
    if op == B_MAX:
        return a if a >= b else b
    if op == B_AND:
        return 1 if (a != 0 and b != 0) else 0
    # Mod with fmod=0: the result takes the divisor's sign (Python's %)
    if b == -1 and a == Int(-0x7FFFFFFFFFFFFFFF - 1):
        return 0
    return a % b


def _trailing_period(small: List[Int], out_shape: List[Int]) -> Int:
    """numel(small) when `small` (leading 1s ignored) equals the trailing dims of `out_shape`,
    so that broadcasting just repeats it; 0 otherwise."""
    var lead = 0
    while lead < len(small) and small[lead] == 1 and len(small) - lead > 0:
        lead += 1
    var t = len(small) - lead
    if t > len(out_shape):
        return 0
    for d in range(t):
        if small[lead + d] != out_shape[len(out_shape) - t + d]:
            return 0
    return numel_of(small)


def _float_repeat(op: Int, big: Tensor, small: Tensor, period: Int, var out_shape: List[Int], swapped: Bool) -> Tensor:
    """big (op) small, with small repeating every `period` elements; operands swapped if `swapped`."""
    var n = big.numel()
    var o = List[Float32](length=n, fill=0.0)
    var sm = small.owned_floats()
    var e = 0
    while e < n:
        for q in range(period):
            var x = big.fget(e + q)
            var y = sm[q]
            o[e + q] = _fop(op, y, x) if swapped else _fop(op, x, y)
        e += period
    return Tensor.floats(out_shape^, o^)


def binary(op: Int, a: Tensor, b: Tensor) raises -> Tensor:
    _require_valid_tensor(a, "binary")
    _require_valid_tensor(b, "binary")
    var compare = op == B_EQ or op == B_LT or op == B_GT
    if not compare and a.is_float() != b.is_float():
        raise Error("binary: mismatched tensor dtypes")
    var out_shape = broadcast_shape(a.shape, b.shape)
    var n = numel_of(out_shape)
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("binary: tensor shape overflow or exceeds maximum limit")
    var same_a = a.numel() == n
    var same_b = b.numel() == n
    var plain_a = same_a and len(a.shape) == len(out_shape)
    var plain_b = same_b and len(b.shape) == len(out_shape)
    if a.is_float() and plain_a and not plain_b and op <= B_DIV:
        var period = _trailing_period(b.shape, out_shape)
        if period > 0:
            return _float_repeat(op, a, b, period, out_shape^, False)
    if a.is_float() and plain_b and not plain_a and op <= B_DIV:
        var period = _trailing_period(a.shape, out_shape)
        if period > 0:
            return _float_repeat(op, b, a, period, out_shape^, True)
    var oa = List[Int]()
    var ob = List[Int]()
    if not plain_a:
        oa = broadcast_offsets(a.shape, out_shape)
    if not plain_b:
        ob = broadcast_offsets(b.shape, out_shape)
    if compare:
        var o = List[Int](length=n, fill=0)
        for e in range(n):
            var ia = e if plain_a else oa[e]
            var ib = e if plain_b else ob[e]
            var x = a.get(ia)
            var y = b.get(ib)
            var r: Bool
            if op == B_EQ:
                r = x == y
            elif op == B_LT:
                r = x < y
            else:
                r = x > y
            o[e] = 1 if r else 0
        return Tensor.ints(DT_BOOL, out_shape^, o^)
    if a.is_float():
        var o = List[Float32](length=n, fill=0.0)
        if plain_a and plain_b and a.ext == 0 and b.ext == 0:
            var pa = a.f.unsafe_ptr()
            var pb = b.f.unsafe_ptr()
            var po = o.unsafe_ptr()
            var e = 0
            if op <= B_MUL:
                while e + LANES <= n:
                    var x = pa.unsafe_load[width=LANES](e)
                    var y = pb.unsafe_load[width=LANES](e)
                    var r: F8
                    if op == B_ADD:
                        r = x + y
                    elif op == B_SUB:
                        r = x - y
                    else:
                        r = x * y
                    po.unsafe_offset(e).unsafe_store(r)
                    e += LANES
            while e < n:
                o[e] = _fop(op, a.f[e], b.f[e])
                e += 1
            return Tensor.floats(out_shape^, o^)
        for e in range(n):
            var ia = e if plain_a else oa[e]
            var ib = e if plain_b else ob[e]
            o[e] = _fop(op, a.fget(ia), b.fget(ib))
        return Tensor.floats(out_shape^, o^)
    var o = List[Int](length=n, fill=0)
    for e in range(n):
        var ia = e if plain_a else oa[e]
        var ib = e if plain_b else ob[e]
        var bv = b.i[ib]
        if (op == B_DIV or op == B_MOD) and bv == 0:
            raise Error("binary: integer division by zero")
        o[e] = _iop(op, a.i[ia], bv)
    return Tensor.ints(a.dtype, out_shape^, o^)


def where(c: Tensor, x: Tensor, y: Tensor) raises -> Tensor:
    _require_valid_tensor(c, "where")
    _require_valid_tensor(x, "where")
    _require_valid_tensor(y, "where")
    if c.is_float():
        raise Error("where: condition must be integer/bool")
    if x.is_float() != y.is_float():
        raise Error("where: mismatched branch tensor dtypes")
    var s = broadcast_shape(broadcast_shape(c.shape, x.shape), y.shape)
    var n = numel_of(s)
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("where: tensor shape overflow or exceeds maximum limit")
    var oc = broadcast_offsets(c.shape, s)
    var ox = broadcast_offsets(x.shape, s)
    var oy = broadcast_offsets(y.shape, s)
    var out = Tensor.like(x, s.copy())
    for e in range(n):
        var take_x = c.i[oc[e]] != 0
        if x.is_float():
            out.f[e] = x.fget(ox[e]) if take_x else y.fget(oy[e])
        else:
            out.i[e] = x.i[ox[e]] if take_x else y.i[oy[e]]
    return out^


@always_inline
def _f64_to_i64_sat(v: Float64) -> Int:
    if isnan(v):
        return 0
    if v >= 9223372036854775808.0:
        return 0x7FFFFFFFFFFFFFFF
    if v <= -9223372036854775808.0:
        return Int(-0x7FFFFFFFFFFFFFFF - 1)
    return Int(v)


@always_inline
def _f64_to_i32_sat(v: Float64) -> Int:
    if isnan(v):
        return 0
    if v >= 2147483647.0:
        return 2147483647
    if v <= -2147483648.0:
        return -2147483648
    return Int(Int32(Int(v)))


def clip(x: Tensor, lo: Float64, hi: Float64) raises -> Tensor:
    _require_valid_tensor(x, "clip")
    var n = x.numel()
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("clip: tensor shape overflow or exceeds maximum limit")
    var out = Tensor.like(x, x.shape.copy())
    for e in range(n):
        if x.is_float():
            out.f[e] = min(max(x.fget(e), Float32(lo)), Float32(hi))
        else:
            var v = Float64(x.i[e])
            if not isnan(lo) and v < lo:
                v = lo
            if not isnan(hi) and v > hi:
                v = hi
            out.i[e] = _f64_to_i64_sat(v)
    return out^


def cast(x: Tensor, to: Int) raises -> Tensor:
    _require_valid_tensor(x, "cast")
    var n = x.numel()
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("cast: tensor shape overflow or exceeds maximum limit")
    if to == DT_F32:
        if x.is_float():
            return x.reshaped(x.shape.copy())
        var o = List[Float32](length=n, fill=0.0)
        for e in range(n):
            o[e] = Float32(x.i[e])
        return Tensor.floats(x.shape.copy(), o^)
    if to != DT_I32 and to != DT_I64 and to != DT_BOOL:
        raise Error("Cast to unsupported type " + String(to))
    var o = List[Int](length=n, fill=0)
    for e in range(n):
        var v: Int
        if x.is_float():
            var f = x.fget(e)
            if to == DT_BOOL:
                v = 1 if f != Float32(0.0) else 0
            elif to == DT_I32:
                v = _f64_to_i32_sat(Float64(f))
            else:
                v = _f64_to_i64_sat(Float64(f))
        else:
            if to == DT_BOOL:
                v = 1 if x.i[e] != 0 else 0
            elif to == DT_I32:
                v = Int(Int32(x.i[e]))
            else:
                v = x.i[e]
        o[e] = v
    return Tensor.ints(to, x.shape.copy(), o^)


# -------------------------------------------------------------------------- data movement

def _copy_elem(mut dst: Tensor, di: Int, src: Tensor, si: Int):
    if src.is_float():
        dst.f[di] = src.fget(si)
    else:
        dst.i[di] = src.i[si]


def transpose(x: Tensor, perm: List[Int]) raises -> Tensor:
    _require_valid_tensor(x, "transpose")
    var r = x.rank()
    if len(perm) != r:
        raise Error("transpose: perm length must match tensor rank")
    var seen = List[Bool](length=r, fill=False)
    for k in range(r):
        var p = perm[k]
        if p < 0 or p >= r or seen[p]:
            raise Error("transpose: invalid permutation axis")
        seen[p] = True
    var out_shape = List[Int](length=r, fill=1)
    for k in range(r):
        out_shape[k] = x.shape[perm[k]]
    var n = numel_of(out_shape)
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("transpose: tensor shape overflow or exceeds maximum limit")
    var st = strides_of(x.shape)
    var pst = List[Int](length=r, fill=0)
    for k in range(r):
        pst[k] = st[perm[k]]
    var out = Tensor.like(x, out_shape.copy())
    if r == 0:
        if n > 0:
            if (x.is_float() and (x.ext != 0 or len(x.f) > 0)) or (not x.is_float() and len(x.i) > 0):
                _copy_elem(out, 0, x, 0)
        return out^
    if n == 0:
        return out^
    var ctr = List[Int](length=r, fill=0)
    var cur = 0
    if x.is_float():
        var src = fptr_at(x.fptr())
        var dst = out.f.unsafe_ptr()
        var inner = out_shape[r - 1]
        var istride = pst[r - 1]
        var e = 0
        while e < n:
            for q in range(inner):
                dst.unsafe_offset(e + q).unsafe_store(src.unsafe_offset(cur + q * istride).unsafe_load[alignment=1]())
            e += inner
            var d = r - 2
            while d >= 0:
                ctr[d] += 1
                cur += pst[d]
                if ctr[d] < out_shape[d]:
                    break
                cur -= pst[d] * ctr[d]
                ctr[d] = 0
                d -= 1
        return out^
    for e in range(n):
        _copy_elem(out, e, x, cur)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            cur += pst[d]
            if ctr[d] < out_shape[d]:
                break
            cur -= pst[d] * ctr[d]
            ctr[d] = 0
            d -= 1
    return out^


def concat(xs: List[Tensor], axis_in: Int) raises -> Tensor:
    if len(xs) == 0:
        raise Error("concat: empty input list")
    var r = xs[0].rank()
    if r == 0:
        raise Error("concat: cannot concatenate 0-D tensors")
    var axis = norm_axis(axis_in, r)
    if axis < 0 or axis >= r:
        raise Error("concat: axis out of range")
    var out_shape = xs[0].shape.copy()
    var total = 0
    for t in xs:
        _require_valid_tensor(t, "concat")
        if t.dtype != xs[0].dtype:
            raise Error("concat: mismatched tensor dtypes")
        if t.rank() != r:
            raise Error("concat: mismatched tensor ranks")
        for d in range(r):
            if d != axis and t.shape[d] != out_shape[d]:
                raise Error("concat: mismatched non-axis dimension")
        total = _checked_add(total, t.shape[axis])
    out_shape[axis] = total
    var out_n = numel_of(out_shape)
    if out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("concat: tensor shape overflow or exceeds maximum limit")
    var outer = 1
    for k in range(axis):
        outer = _checked_mul(outer, out_shape[k])
    var inner = 1
    for k in range(axis + 1, r):
        inner = _checked_mul(inner, out_shape[k])
    var row = _checked_mul(total, inner)
    var out = Tensor.like(xs[0], out_shape^)
    var off = 0
    for t in xs:
        var chunk = _checked_mul(t.shape[axis], inner)
        for o in range(outer):
            for c in range(chunk):
                _copy_elem(out, o * row + off + c, t, o * chunk + c)
        off += chunk
    return out^


def gather(data: Tensor, idx: Tensor, axis_in: Int) raises -> Tensor:
    _require_valid_tensor(data, "gather")
    _require_valid_tensor(idx, "gather")
    var r = data.rank()
    if r == 0:
        raise Error("gather: data must have rank >= 1")
    var axis = norm_axis(axis_in, r)
    if axis < 0 or axis >= r:
        raise Error("gather: axis out of range")
    var out_shape = List[Int]()
    for k in range(axis):
        out_shape.append(data.shape[k])
    for d in idx.shape:
        out_shape.append(d)
    for k in range(axis + 1, r):
        out_shape.append(data.shape[k])
    var out_n = numel_of(out_shape)
    if out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("gather: tensor shape overflow or exceeds maximum limit")
    var outer = 1
    for k in range(axis):
        outer = _checked_mul(outer, data.shape[k])
    var inner = 1
    for k in range(axis + 1, r):
        inner = _checked_mul(inner, data.shape[k])
    var dim = data.shape[axis]
    var ni = idx.numel()
    for j in range(ni):
        var g = idx.int_at(j)
        if g < 0:
            g += dim
        if g < 0 or g >= dim:
            raise Error("Gather: index out of range")
    var out = Tensor.like(data, out_shape^)
    for o in range(outer):
        for j in range(ni):
            var g = idx.int_at(j)
            if g < 0:
                g += dim
            for c in range(inner):
                _copy_elem(out, (o * ni + j) * inner + c, data, (o * dim + g) * inner + c)
    return out^


def gather_elements(data: Tensor, idx: Tensor, axis_in: Int) raises -> Tensor:
    _require_valid_tensor(data, "gather_elements")
    _require_valid_tensor(idx, "gather_elements")
    var r = data.rank()
    if r == 0 or idx.rank() != r:
        raise Error("gather_elements: data and indices must have matching rank >= 1")
    var axis = norm_axis(axis_in, r)
    if axis < 0 or axis >= r:
        raise Error("gather_elements: axis out of range")
    for d in range(r):
        if d != axis and idx.shape[d] > data.shape[d]:
            raise Error("gather_elements: indices shape exceeds data shape on non-axis dimension")
    var n = idx.numel()
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("gather_elements: tensor shape overflow or exceeds maximum limit")
    var dst = strides_of(data.shape)
    var out = Tensor.like(data, idx.shape.copy())
    var ctr = List[Int](length=r, fill=0)
    for e in range(n):
        var g = idx.int_at(e)
        if g < 0:
            g += data.shape[axis]
        if g < 0 or g >= data.shape[axis]:
            raise Error("gather_elements: index out of range")
        var off = 0
        for d in range(r):
            off += (g if d == axis else ctr[d]) * dst[d]
        _copy_elem(out, e, data, off)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < idx.shape[d]:
                break
            ctr[d] = 0
            d -= 1
    return out^


def scatter_elements(data: Tensor, idx: Tensor, upd: Tensor, axis_in: Int) raises -> Tensor:
    _require_valid_tensor(data, "scatter_elements")
    _require_valid_tensor(idx, "scatter_elements")
    _require_valid_tensor(upd, "scatter_elements")
    var r = data.rank()
    if r == 0 or idx.rank() != r:
        raise Error("scatter_elements: data and indices must have matching rank >= 1")
    if data.is_float() != upd.is_float():
        raise Error("scatter_elements: mismatched data and updates dtypes")
    var axis = norm_axis(axis_in, r)
    if axis < 0 or axis >= r:
        raise Error("scatter_elements: axis out of range")
    if upd.rank() != r:
        raise Error("scatter_elements: updates shape must match indices")
    for d in range(r):
        if upd.shape[d] != idx.shape[d]:
            raise Error("scatter_elements: updates shape must match indices")
        if d != axis and idx.shape[d] > data.shape[d]:
            raise Error("scatter_elements: indices shape exceeds data shape on non-axis dimension")
    var n = idx.numel()
    if data.numel() > MAX_RANGE_ELEMS or n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("scatter_elements: tensor shape overflow or exceeds maximum limit")
    var dst = strides_of(data.shape)
    var out = data.reshaped(data.shape.copy())
    if out.is_float() and out.ext != 0:
        out = Tensor.floats(data.shape.copy(), data.owned_floats())
    var ctr = List[Int](length=r, fill=0)
    for e in range(n):
        var g = idx.int_at(e)
        if g < 0:
            g += data.shape[axis]
        if g < 0 or g >= data.shape[axis]:
            raise Error("scatter_elements: index out of range")
        var off = 0
        for d in range(r):
            off += (g if d == axis else ctr[d]) * dst[d]
        _copy_elem(out, off, upd, e)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < idx.shape[d]:
                break
            ctr[d] = 0
            d -= 1
    return out^


def scatter_nd(data: Tensor, idx: Tensor, upd: Tensor) raises -> Tensor:
    _require_valid_tensor(data, "scatter_nd")
    _require_valid_tensor(idx, "scatter_nd")
    _require_valid_tensor(upd, "scatter_nd")
    if idx.rank() < 1:
        raise Error("scatter_nd: indices must have rank >= 1")
    if data.is_float() != upd.is_float():
        raise Error("scatter_nd: mismatched data and updates dtypes")
    var k = idx.shape[idx.rank() - 1]
    if k <= 0 or k > data.rank():
        raise Error("scatter_nd: invalid index tuple size")
    if data.numel() > MAX_RANGE_ELEMS:
        raise Error("scatter_nd: tensor exceeds maximum limit")
    var out = data.reshaped(data.shape.copy())
    if out.is_float() and out.ext != 0:
        out = Tensor.floats(data.shape.copy(), data.owned_floats())
    var dst = strides_of(data.shape)
    var slice_n = 1
    for d in range(k, data.rank()):
        slice_n = _checked_mul(slice_n, data.shape[d])
    var rows = idx.numel() // k
    if upd.numel() < _checked_mul(rows, slice_n):
        raise Error("scatter_nd: updates tensor too small")
    for j in range(rows):
        var off = 0
        for d in range(k):
            var g = idx.int_at(j * k + d)
            if g < 0:
                g += data.shape[d]
            if g < 0 or g >= data.shape[d]:
                raise Error("scatter_nd: index out of range")
            off += g * dst[d]
        for c in range(slice_n):
            _copy_elem(out, off + c, upd, j * slice_n + c)
    return out^


def slice(data: Tensor, starts: List[Int], ends: List[Int], axes_in: List[Int], steps_in: List[Int]) raises -> Tensor:
    _require_valid_tensor(data, "slice")
    if len(ends) != len(starts):
        raise Error("slice: starts and ends length mismatch")
    if len(axes_in) != 0 and len(axes_in) != len(starts):
        raise Error("slice: axes length mismatch")
    if len(steps_in) != 0 and len(steps_in) != len(starts):
        raise Error("slice: steps length mismatch")
    var r = data.rank()
    var st = List[Int](length=r, fill=0)
    var en = data.shape.copy()
    var sp = List[Int](length=r, fill=1)
    var seen = List[Bool](length=r, fill=False)
    for j in range(len(starts)):
        var ax = norm_axis(axes_in[j] if len(axes_in) > 0 else j, r)
        if ax < 0 or ax >= r:
            raise Error("slice: axis out of range")
        if seen[ax]:
            raise Error("slice: duplicate axis")
        seen[ax] = True
        var dim = data.shape[ax]
        var step = steps_in[j] if len(steps_in) > 0 else 1
        if step == 0:
            raise Error("Slice: zero step")
        var s = starts[j]
        var e = ends[j]
        if s < 0:
            s += dim
        if e < 0:
            e += dim
        if step > 0:
            s = min(max(s, 0), dim)
            e = min(max(e, 0), dim)
        else:
            s = min(max(s, 0), dim - 1)
            e = min(max(e, -1), dim - 1)
        st[ax] = s
        en[ax] = e
        sp[ax] = step
    var out_shape = List[Int](length=r, fill=0)
    for d in range(r):
        var cnt: Int = 0
        if sp[d] > 0:
            if en[d] > st[d]:
                cnt = (en[d] - st[d] - 1) // sp[d] + 1
        else:
            if st[d] > en[d]:
                if sp[d] < -0x7FFFFFFFFFFFFFFF:
                    cnt = 1
                else:
                    cnt = (st[d] - en[d] - 1) // (-sp[d]) + 1
        out_shape[d] = cnt
    var n = numel_of(out_shape)
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("slice: tensor shape overflow or exceeds maximum limit")
    var out = Tensor.like(data, out_shape.copy())
    if n == 0:
        return out^
    var dst = strides_of(data.shape)
    var ctr = List[Int](length=r, fill=0)
    for e in range(n):
        var off = 0
        for d in range(r):
            off += (st[d] + ctr[d] * sp[d]) * dst[d]
        _copy_elem(out, e, data, off)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < out_shape[d]:
                break
            ctr[d] = 0
            d -= 1
    return out^


def expand(x: Tensor, shape: List[Int]) raises -> Tensor:
    _require_valid_tensor(x, "expand")
    for d in shape:
        if d < 0:
            raise Error("expand: negative dimension")
    var s = broadcast_shape(x.shape, shape)
    var n = numel_of(s)
    if n < 0 or n > MAX_RANGE_ELEMS:
        raise Error("expand: tensor shape overflow or exceeds maximum limit")
    var offs = broadcast_offsets(x.shape, s)
    var out = Tensor.like(x, s.copy())
    for e in range(len(offs)):
        _copy_elem(out, e, x, offs[e])
    return out^


def tile(x: Tensor, reps: List[Int]) raises -> Tensor:
    _require_valid_tensor(x, "tile")
    var r = x.rank()
    if len(reps) != r:
        raise Error("tile: reps length must match tensor rank")
    var out_shape = List[Int](length=r, fill=0)
    for d in range(r):
        out_shape[d] = _checked_mul(x.shape[d], reps[d])
    var out_n = numel_of(out_shape)
    if out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("tile: tensor shape overflow or exceeds maximum limit")
    var out = Tensor.like(x, out_shape.copy())
    var dst = strides_of(x.shape)
    var ctr = List[Int](length=r, fill=0)
    for e in range(out_n):
        var off = 0
        for d in range(r):
            off += (ctr[d] % x.shape[d]) * dst[d]
        _copy_elem(out, e, x, off)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < out_shape[d]:
                break
            ctr[d] = 0
            d -= 1
    return out^


def range_op(start: Tensor, limit: Tensor, delta: Tensor) raises -> Tensor:
    _require_valid_tensor(start, "range_op")
    _require_valid_tensor(limit, "range_op")
    _require_valid_tensor(delta, "range_op")
    if start.numel() <= 0 or limit.numel() <= 0 or delta.numel() <= 0:
        raise Error("range_op: operands must be non-empty")
    if start.is_float() != limit.is_float() or start.is_float() != delta.is_float():
        raise Error("range_op: mismatched operand dtypes")
    var s = start.get(0)
    var l = limit.get(0)
    var d = delta.get(0)
    if isnan(s) or isinf(s) or isnan(l) or isinf(l) or isnan(d) or isinf(d):
        raise Error("range_op: operands must be finite")
    if d == 0.0:
        raise Error("range_op: delta must be non-zero")
    var span = (l - s) / d
    if isnan(span) or isinf(span):
        raise Error("range_op: range span overflow")
    if span > Float64(MAX_RANGE_ELEMS):
        raise Error("range_op: element count exceeds maximum limit")
    var n = 0 if span <= 0.0 else Int(ceil(span))
    if start.is_float():
        var o = List[Float32](length=n, fill=0.0)
        for k in range(n):
            o[k] = Float32(s + Float64(k) * d)
        return Tensor.floats([n], o^)
    var o = List[Int](length=n, fill=0)
    for k in range(n):
        o[k] = start.i[0] + k * delta.i[0]
    return Tensor.ints(start.dtype, [n], o^)


def onehot(idx: Tensor, depth: Int, values: Tensor) raises -> Tensor:
    _require_valid_tensor(idx, "onehot")
    _require_valid_tensor(values, "onehot")
    if depth < 0 or values.numel() < 2:
        raise Error("onehot: invalid depth or values tensor")
    var out_shape = idx.shape.copy()
    out_shape.append(depth)
    var out_n = numel_of(out_shape)
    if out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("onehot: tensor shape overflow or exceeds maximum limit")
    var out = Tensor.like(values, out_shape^)
    var n = idx.numel()
    for j in range(n):
        var g = idx.int_at(j)
        if g < 0:
            g += depth
        for c in range(depth):
            _copy_elem(out, j * depth + c, values, 1 if c == g else 0)
    return out^


def resize_nearest(x: Tensor, out_shape: List[Int]) raises -> Tensor:
    """Resize with mode=nearest, nearest_mode=floor, coordinate_transformation_mode=asymmetric."""
    _require_valid_tensor(x, "resize_nearest")
    var r = x.rank()
    var out_n = numel_of(out_shape)
    if len(out_shape) != r or out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("resize_nearest: invalid or oversized output shape")
    if x.numel() == 0 and out_n > 0:
        raise Error("resize_nearest: cannot resize empty input to non-empty output")
    var out = Tensor.like(x, out_shape.copy())
    var dst = strides_of(x.shape)
    var ctr = List[Int](length=r, fill=0)
    for e in range(out.numel()):
        var off = 0
        for d in range(r):
            var scale = Float64(out_shape[d]) / Float64(x.shape[d])
            var src = Int(floor(Float64(ctr[d]) / scale))
            src = min(max(src, 0), x.shape[d] - 1)
            off += src * dst[d]
        _copy_elem(out, e, x, off)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < out_shape[d]:
                break
            ctr[d] = 0
            d -= 1
    return out^


# ------------------------------------------------------------------------------- reductions

def _outer_inner(shape: List[Int], axis: Int) raises -> Tuple[Int, Int, Int]:
    var outer = 1
    for k in range(axis):
        outer = _checked_mul(outer, shape[k])
    var inner = 1
    for k in range(axis + 1, len(shape)):
        inner = _checked_mul(inner, shape[k])
    return (outer, shape[axis], inner)


def reduce(x: Tensor, axes_in: List[Int], keepdims: Bool, l2: Bool) raises -> Tensor:
    _require_valid_tensor(x, "reduce")
    var r = x.rank()
    var red = List[Bool](length=r, fill=False)
    if len(axes_in) == 0:
        for d in range(r):
            red[d] = True
    for a in axes_in:
        var na = norm_axis(a, r)
        if na < 0 or na >= r:
            raise Error("reduce: axis out of range")
        red[na] = True
    var out_shape = List[Int]()
    var kept = List[Int]()
    for d in range(r):
        kept.append(1 if red[d] else x.shape[d])
        if not red[d]:
            out_shape.append(x.shape[d])
        elif keepdims:
            out_shape.append(1)
    var kst = strides_of(kept)
    var nk = numel_of(kept)
    if nk < 0 or nk > MAX_RANGE_ELEMS:
        raise Error("reduce: tensor shape overflow or exceeds maximum limit")
    var acc = List[Float64](length=nk, fill=0.0)
    var ctr = List[Int](length=r, fill=0)
    for e in range(x.numel()):
        var o = 0
        for d in range(r):
            if not red[d]:
                o += ctr[d] * kst[d]
        var v = x.get(e)
        acc[o] += v * v if l2 else v
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < x.shape[d]:
                break
            ctr[d] = 0
            d -= 1
    if x.is_float():
        var o = List[Float32](length=nk, fill=0.0)
        for k in range(nk):
            o[k] = Float32(sqrt(acc[k]) if l2 else acc[k])
        return Tensor.floats(out_shape^, o^)
    var o = List[Int](length=nk, fill=0)
    for k in range(nk):
        o[k] = _f64_to_i64_sat(acc[k])
    return Tensor.ints(x.dtype, out_shape^, o^)


def argmax(x: Tensor, axis_in: Int, keepdims: Bool) raises -> Tensor:
    _require_valid_tensor(x, "argmax")
    if x.rank() == 0:
        raise Error("argmax: input must have rank >= 1")
    var axis = norm_axis(axis_in, x.rank())
    if axis < 0 or axis >= x.rank():
        raise Error("argmax: axis out of range")
    var oi = _outer_inner(x.shape, axis)
    if oi[1] <= 0:
        raise Error("argmax: empty axis dimension")
    var out_shape = List[Int]()
    for d in range(x.rank()):
        if d != axis:
            out_shape.append(x.shape[d])
        elif keepdims:
            out_shape.append(1)
    var out_n = _checked_mul(oi[0], oi[2])
    if out_n > MAX_RANGE_ELEMS:
        raise Error("argmax: tensor exceeds maximum limit")
    var o = List[Int](length=out_n, fill=0)
    for a in range(oi[0]):
        for c in range(oi[2]):
            var best = 0
            var bv = x.get(a * oi[1] * oi[2] + c)
            for j in range(1, oi[1]):
                var v = x.get((a * oi[1] + j) * oi[2] + c)
                if v > bv:
                    bv = v
                    best = j
            o[a * oi[2] + c] = best
    return Tensor.ints(DT_I64, out_shape^, o^)


def topk(x: Tensor, k: Int, axis_in: Int, largest: Bool) raises -> List[Tensor]:
    _require_valid_tensor(x, "topk")
    if x.rank() == 0:
        raise Error("topk: input must have rank >= 1")
    var axis = norm_axis(axis_in, x.rank())
    if axis < 0 or axis >= x.rank():
        raise Error("topk: axis out of range")
    if k < 0 or k > x.shape[axis]:
        raise Error("topk: k out of range")
    var oi = _outer_inner(x.shape, axis)
    var out_shape = x.shape.copy()
    out_shape[axis] = k
    var out_n = numel_of(out_shape)
    if out_n < 0 or out_n > MAX_RANGE_ELEMS or oi[1] > MAX_RANGE_ELEMS:
        raise Error("topk: tensor shape overflow or exceeds maximum limit")
    var vals = Tensor.like(x, out_shape.copy())
    var inds = Tensor.ints(DT_I64, out_shape.copy(), List[Int](length=out_n, fill=0))
    var n = oi[1]
    for a in range(oi[0]):
        for c in range(oi[2]):
            # stable selection: on ties the lower index comes first
            var order = List[Int](capacity=n)
            for j in range(n):
                order.append(j)
            for s in range(k):
                var best = s
                for j in range(s + 1, n):
                    var vj = x.get((a * n + order[j]) * oi[2] + c)
                    var vb = x.get((a * n + order[best]) * oi[2] + c)
                    var better = vj > vb if largest else vj < vb
                    if better or (vj == vb and order[j] < order[best]):
                        best = j
                var tmp = order[s]
                order[s] = order[best]
                order[best] = tmp
                var dst = (a * k + s) * oi[2] + c
                _copy_elem(vals, dst, x, (a * n + order[s]) * oi[2] + c)
                inds.i[dst] = order[s]
    var out = List[Tensor]()
    out.append(vals^)
    out.append(inds^)
    return out^


def cumsum(x: Tensor, axis_in: Int, exclusive: Bool, reverse: Bool) raises -> Tensor:
    _require_valid_tensor(x, "cumsum")
    if x.rank() == 0:
        raise Error("cumsum: input must have rank >= 1")
    var axis = norm_axis(axis_in, x.rank())
    if axis < 0 or axis >= x.rank():
        raise Error("cumsum: axis out of range")
    var oi = _outer_inner(x.shape, axis)
    if x.numel() > MAX_RANGE_ELEMS:
        raise Error("cumsum: tensor exceeds maximum limit")
    var out = Tensor.like(x, x.shape.copy())
    for a in range(oi[0]):
        for c in range(oi[2]):
            var acc = 0.0
            var iacc = 0
            for jj in range(oi[1]):
                var j = oi[1] - 1 - jj if reverse else jj
                var at = (a * oi[1] + j) * oi[2] + c
                if x.is_float():
                    if exclusive:
                        out.f[at] = Float32(acc)
                        acc += Float64(x.fget(at))
                    else:
                        acc += Float64(x.fget(at))
                        out.f[at] = Float32(acc)
                else:
                    if exclusive:
                        out.i[at] = iacc
                        iacc += x.i[at]
                    else:
                        iacc += x.i[at]
                        out.i[at] = iacc
    return out^


def softmax(x: Tensor, axis_in: Int) raises -> Tensor:
    _require_valid_tensor(x, "softmax")
    if x.rank() == 0:
        raise Error("softmax: input must have rank >= 1")
    if not x.is_float():
        raise Error("softmax: input must be float")
    var axis = norm_axis(axis_in, x.rank())
    if axis < 0 or axis >= x.rank():
        raise Error("softmax: axis out of range")
    var oi = _outer_inner(x.shape, axis)
    if oi[1] <= 0:
        raise Error("softmax: empty axis dimension")
    if x.numel() > MAX_RANGE_ELEMS:
        raise Error("softmax: tensor exceeds maximum limit")
    var out = Tensor.like(x, x.shape.copy())
    for a in range(oi[0]):
        for c in range(oi[2]):
            var m = x.fget(a * oi[1] * oi[2] + c)
            if isnan(m):
                raise Error("softmax: NaN in input")
            for j in range(1, oi[1]):
                var v = x.fget((a * oi[1] + j) * oi[2] + c)
                if isnan(v):
                    raise Error("softmax: NaN in input")
                if v > m:
                    m = v
            if isinf(m):
                if m > Float32(0.0):
                    var s_inf: Float32 = 0.0
                    for j in range(oi[1]):
                        var at = (a * oi[1] + j) * oi[2] + c
                        var v = x.fget(at)
                        var e = Float32(1.0) if (isinf(v) and v > Float32(0.0)) else Float32(0.0)
                        out.f[at] = e
                        s_inf += e
                    var inv_s = Float32(1.0) / max(Float32(1.0), s_inf)
                    for j in range(oi[1]):
                        out.f[(a * oi[1] + j) * oi[2] + c] *= inv_s
                else:
                    var uni = Float32(1.0) / Float32(oi[1])
                    for j in range(oi[1]):
                        out.f[(a * oi[1] + j) * oi[2] + c] = uni
                continue
            var s: Float32 = 0
            for j in range(oi[1]):
                var at = (a * oi[1] + j) * oi[2] + c
                var e = exp(x.fget(at) - m)
                out.f[at] = e
                s += e
            for j in range(oi[1]):
                out.f[(a * oi[1] + j) * oi[2] + c] /= s
    return out^


def layer_norm(x: Tensor, scale: Tensor, bias: Tensor, has_bias: Bool, eps: Float32) raises -> Tensor:
    _require_valid_tensor(x, "layer_norm")
    _require_valid_tensor(scale, "layer_norm")
    if has_bias:
        _require_valid_tensor(bias, "layer_norm")
    if not isfinite(eps) or eps < Float32(0.0):
        raise Error("layer_norm: epsilon must be finite and non-negative")
    if not x.is_float() or not scale.is_float() or (has_bias and not bias.is_float()):
        raise Error("layer_norm: inputs must be float tensors")
    if x.rank() == 0:
        raise Error("layer_norm: input must have rank >= 1")
    var n = x.shape[x.rank() - 1]
    if n <= 0 or scale.numel() < n or (has_bias and bias.numel() < n):
        raise Error("layer_norm: invalid normalization dimension")
    if x.numel() > MAX_RANGE_ELEMS:
        raise Error("layer_norm: tensor exceeds maximum limit")
    for j in range(scale.numel()):
        if not isfinite(scale.fget(j)):
            raise Error("layer_norm: non-finite element in scale")
    if has_bias:
        for j in range(bias.numel()):
            if not isfinite(bias.fget(j)):
                raise Error("layer_norm: non-finite element in bias")
    var rows = x.numel() // n
    var out = Tensor.like(x, x.shape.copy())
    for r in range(rows):
        var base = r * n
        var mean: Float64 = 0.0
        for j in range(n):
            var xj = x.fget(base + j)
            if not isfinite(xj):
                raise Error("layer_norm: non-finite element in input")
            mean += Float64(xj)
        mean /= Float64(n)
        var var_: Float64 = 0.0
        for j in range(n):
            var d = Float64(x.fget(base + j)) - mean
            var_ += d * d
        var_ /= Float64(n)
        var denom = var_ + Float64(eps)
        var inv = (1.0 / sqrt(denom)) if denom > 0.0 else 0.0
        for j in range(n):
            var v = (Float64(x.fget(base + j)) - mean) * inv * Float64(scale.fget(j))
            if has_bias:
                v += Float64(bias.fget(j))
            out.f[base + j] = Float32(v)
    return out^


# --------------------------------------------------------------------------- matrix products

struct _MatCtx(Movable):
    var a: Int
    var b: Int
    var c: Int
    var bias: Int
    var m: Int
    var k: Int
    var n: Int
    var cols_per_job: Int
    var trans_b: Bool

    def __init__(out self, a: Int, b: Int, c: Int, bias: Int, m: Int, k: Int, n: Int, cols_per_job: Int, trans_b: Bool):
        self.a = a
        self.b = b
        self.c = c
        self.bias = bias
        self.m = m
        self.k = k
        self.n = n
        self.cols_per_job = cols_per_job
        self.trans_b = trans_b


def _mat_cols(a: Int, b: Int, c: Int, bias: Int, m: Int, k: Int, n: Int, n0: Int, n1: Int):
    """C[:, n0:n1] = A[m,k] @ B[k,n] (+ bias[n]). B row major, read once per 4 rows of A."""
    var pa = fptr_at(a)
    var pb = fptr_at(b)
    var pc = fptr_at(c)
    var j = n0
    while j < n1:
        var w = min(16, n1 - j)
        var i = 0
        while i < m:
            var rows = min(4, m - i)
            if w == 16:
                var acc0 = F8(0)
                var acc1 = F8(0)
                var acc2 = F8(0)
                var acc3 = F8(0)
                var acc4 = F8(0)
                var acc5 = F8(0)
                var acc6 = F8(0)
                var acc7 = F8(0)
                for kk in range(k):
                    var brow = pb.unsafe_offset(kk * n + j)
                    var b0 = brow.unsafe_load[width=LANES, alignment=1](0)
                    var b1 = brow.unsafe_load[width=LANES, alignment=1](LANES)
                    var x0 = pa.unsafe_offset(i * k + kk).unsafe_load()
                    acc0 += x0 * b0
                    acc1 += x0 * b1
                    if rows > 1:
                        var x1 = pa.unsafe_offset((i + 1) * k + kk).unsafe_load()
                        acc2 += x1 * b0
                        acc3 += x1 * b1
                    if rows > 2:
                        var x2 = pa.unsafe_offset((i + 2) * k + kk).unsafe_load()
                        acc4 += x2 * b0
                        acc5 += x2 * b1
                    if rows > 3:
                        var x3 = pa.unsafe_offset((i + 3) * k + kk).unsafe_load()
                        acc6 += x3 * b0
                        acc7 += x3 * b1
                var bz0 = F8(0)
                var bz1 = F8(0)
                if bias != 0:
                    bz0 = fptr_at(bias).unsafe_offset(j).unsafe_load[width=LANES, alignment=1](0)
                    bz1 = fptr_at(bias).unsafe_offset(j).unsafe_load[width=LANES, alignment=1](LANES)
                pc.unsafe_offset(i * n + j).unsafe_store[alignment=1](acc0 + bz0)
                pc.unsafe_offset(i * n + j + LANES).unsafe_store[alignment=1](acc1 + bz1)
                if rows > 1:
                    pc.unsafe_offset((i + 1) * n + j).unsafe_store[alignment=1](acc2 + bz0)
                    pc.unsafe_offset((i + 1) * n + j + LANES).unsafe_store[alignment=1](acc3 + bz1)
                if rows > 2:
                    pc.unsafe_offset((i + 2) * n + j).unsafe_store[alignment=1](acc4 + bz0)
                    pc.unsafe_offset((i + 2) * n + j + LANES).unsafe_store[alignment=1](acc5 + bz1)
                if rows > 3:
                    pc.unsafe_offset((i + 3) * n + j).unsafe_store[alignment=1](acc6 + bz0)
                    pc.unsafe_offset((i + 3) * n + j + LANES).unsafe_store[alignment=1](acc7 + bz1)
            else:
                for r in range(rows):
                    for jj in range(j, j + w):
                        var s: Float32 = 0
                        for kk in range(k):
                            s += pa.unsafe_offset((i + r) * k + kk).unsafe_load() * pb.unsafe_offset(kk * n + jj).unsafe_load[alignment=1]()
                        if bias != 0:
                            s += fptr_at(bias).unsafe_offset(jj).unsafe_load[alignment=1]()
                        pc.unsafe_offset((i + r) * n + jj).unsafe_store(s)
            i += rows
        j += w


def _dot_cols(a: Int, b: Int, c: Int, bias: Int, m: Int, k: Int, n: Int, n0: Int, n1: Int):
    """C[:, n0:n1] = A[m,k] @ B[n,k]^T (+ bias[n]): contiguous dot products."""
    var pa = fptr_at(a)
    var pb = fptr_at(b)
    var pc = fptr_at(c)
    for j in range(n0, n1):
        var brow = pb.unsafe_offset(j * k)
        for i in range(m):
            var arow = pa.unsafe_offset(i * k)
            var acc = F8(0)
            var kk = 0
            while kk + LANES <= k:
                acc += arow.unsafe_load[width=LANES, alignment=1](kk) * brow.unsafe_load[width=LANES, alignment=1](kk)
                kk += LANES
            var s = acc.reduce_add()
            while kk < k:
                s += arow.unsafe_offset(kk).unsafe_load() * brow.unsafe_offset(kk).unsafe_load[alignment=1]()
                kk += 1
            if bias != 0:
                s += fptr_at(bias).unsafe_offset(j).unsafe_load[alignment=1]()
            pc.unsafe_offset(i * n + j).unsafe_store(s)


def _mat_worker(ctx_addr: Int, job: Int):
    var ctx = Pointer[_MatCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var n0 = job * ctx[].cols_per_job
    var n1 = min(ctx[].n, n0 + ctx[].cols_per_job)
    if n0 >= n1:
        return
    if ctx[].trans_b:
        _dot_cols(ctx[].a, ctx[].b, ctx[].c, ctx[].bias, ctx[].m, ctx[].k, ctx[].n, n0, n1)
    else:
        _mat_cols(ctx[].a, ctx[].b, ctx[].c, ctx[].bias, ctx[].m, ctx[].k, ctx[].n, n0, n1)


def matmul_2d(mut pool: JobPool, a: Int, b: Int, c: Int, bias: Int, m: Int, k: Int, n: Int, trans_b: Bool):
    """C = A @ B (+bias), or A @ B^T when trans_b; all float32 row major at raw addresses."""
    var work = m * k * n
    if work < 1 << 17 or not pool.ready():
        if trans_b:
            _dot_cols(a, b, c, bias, m, k, n, 0, n)
        else:
            _mat_cols(a, b, c, bias, m, k, n, 0, n)
        return
    var jobs = min(64, max(1, n // 16))
    var cols = ((n + jobs - 1) // jobs + 15) // 16 * 16
    jobs = (n + cols - 1) // cols
    var ctx = _MatCtx(a, b, c, bias, m, k, n, cols, trans_b)
    run_parallel_ctx(pool, jobs, ctx, _mat_worker)


def _contiguous(t: Tensor) -> Tensor:
    if t.ext != 0 or len(t.f) == t.numel():
        return t.reshaped(t.shape.copy())
    return Tensor.floats(t.shape.copy(), t.owned_floats())


def matmul(mut pool: JobPool, a: Tensor, b: Tensor) raises -> Tensor:
    _require_valid_tensor(a, "matmul")
    _require_valid_tensor(b, "matmul")
    if not a.is_float() or not b.is_float():
        raise Error("MatMul: operands must be float tensors")
    if a.rank() < 2 or b.rank() < 2:
        raise Error("MatMul: rank-1 operands are not supported")
    var ra = a.rank()
    var rb = b.rank()
    var m = a.shape[ra - 2]
    var k = a.shape[ra - 1]
    var n = b.shape[rb - 1]
    if b.shape[rb - 2] != k:
        raise Error("MatMul: inner dims " + shape_str(a.shape) + " x " + shape_str(b.shape))
    var ba = List[Int]()
    for d in range(ra - 2):
        ba.append(a.shape[d])
    var bb = List[Int]()
    for d in range(rb - 2):
        bb.append(b.shape[d])
    var bs = broadcast_shape(ba, bb)
    var out_shape = bs.copy()
    out_shape.append(m)
    out_shape.append(n)
    var batches = numel_of(bs)
    var out_n = numel_of(out_shape)
    if batches < 0 or out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("matmul: tensor shape overflow or exceeds maximum limit")
    var out = Tensor.zeros(out_shape^)
    if batches == 0 or m == 0 or n == 0:
        return out^
    var oa = broadcast_offsets(ba, bs)
    var ob = broadcast_offsets(bb, bs)
    var pa = a.fptr()
    var pb = b.fptr()
    var pc = out.fptr()
    for t in range(batches):
        matmul_2d(pool, pa + oa[t] * m * k * 4, pb + ob[t] * k * n * 4, pc + t * m * n * 4, 0, m, k, n, False)
    return out^


def gemm(mut pool: JobPool, a: Tensor, b: Tensor, c: Tensor, has_c: Bool, alpha: Float32, beta: Float32,
         trans_a: Bool, trans_b: Bool) raises -> Tensor:
    _require_valid_tensor(a, "gemm")
    _require_valid_tensor(b, "gemm")
    if has_c:
        _require_valid_tensor(c, "gemm")
    if not a.is_float() or not b.is_float() or (has_c and not c.is_float()):
        raise Error("Gemm: operands must be float tensors")
    if a.rank() != 2 or b.rank() != 2:
        raise Error("Gemm: inputs must be 2-D")
    if trans_a or alpha != 1 or (has_c and beta != 1):
        raise Error("Gemm: only alpha=beta=1 without transA is supported")
    var m = a.shape[0]
    var k = a.shape[1]
    var bk = b.shape[1] if trans_b else b.shape[0]
    var n = b.shape[0] if trans_b else b.shape[1]
    if bk != k:
        raise Error("Gemm: inner dims " + shape_str(a.shape) + " x " + shape_str(b.shape))
    var out_n = numel_of([m, n])
    if out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("Gemm: tensor shape overflow or exceeds maximum limit")
    if has_c:
        if c.rank() > 2:
            raise Error("Gemm: bias C must have rank <= 2")
        if c.rank() == 1 and c.shape[0] != n and c.shape[0] != 1:
            raise Error("Gemm: incompatible bias C shape " + shape_str(c.shape))
        if c.rank() == 2 and ((c.shape[0] != m and c.shape[0] != 1) or (c.shape[1] != n and c.shape[1] != 1)):
            raise Error("Gemm: incompatible bias C shape " + shape_str(c.shape))
    var out = Tensor.zeros([m, n])
    if out_n == 0:
        return out^
    var bias = 0
    var cfull = Tensor()
    if has_c:
        if (c.rank() == 1 and c.shape[0] == n) or (c.rank() == 2 and c.shape[0] == 1 and c.shape[1] == n):
            bias = c.fptr()
        else:
            cfull = expand(c, [m, n])
    matmul_2d(pool, a.fptr(), b.fptr(), out.fptr(), bias, m, k, n, trans_b)
    if has_c and bias == 0:
        for e in range(out_n):
            out.f[e] += cfull.f[e]
    return out^


def conv1d(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pad0: Int, pad1: Int, dil: Int,
           stride: Int, group: Int) raises -> Tensor:
    _require_valid_tensor(x, "conv1d")
    _require_valid_tensor(w, "conv1d")
    if has_b:
        _require_valid_tensor(b, "conv1d")
    if not x.is_float() or not w.is_float() or (has_b and not b.is_float()):
        raise Error("Conv: operands must be float tensors")
    if x.rank() != 3 or w.rank() != 3 or group != 1 or stride != 1:
        raise Error("Conv: only 1-D, group 1, stride 1 is supported")
    var batch = x.shape[0]
    var cin = x.shape[1]
    var t_in = x.shape[2]
    var cout = w.shape[0]
    var kw = w.shape[2]
    if w.shape[1] != cin:
        raise Error("Conv: weight input channels mismatch")
    if pad0 < 0 or pad1 < 0 or dil < 1 or kw < 1:
        raise Error("Conv: invalid padding, dilation, or kernel size")
    if has_b and (b.rank() != 1 or b.shape[0] != cout):
        raise Error("Conv: bias must be 1-D with length cout")
    var padded_t = _checked_add(_checked_add(t_in, pad0), pad1)
    var span_k = _checked_mul(dil, kw - 1)
    var t_out = padded_t - span_k
    var out_n = numel_of([batch, cout, t_out])
    if t_out < 0 or out_n < 0 or out_n > MAX_RANGE_ELEMS:
        raise Error("Conv: invalid or oversized output shape")
    var j = _checked_mul(cin, kw)
    var col_n = _checked_mul(t_out, j)
    var tmp_n = _checked_mul(t_out, cout)
    if col_n > MAX_RANGE_ELEMS or tmp_n > MAX_RANGE_ELEMS:
        raise Error("Conv: workspace exceeds maximum limit")
    var out = Tensor.zeros([batch, cout, t_out])
    if out_n == 0:
        return out^
    var bias_ptr = b.fptr() if has_b else 0
    # im2col^T: col[t, ci*kw + kk] so each output row is a dot product with W[co, :]
    var col = List[Float32](length=col_n, fill=0.0)
    for bi in range(batch):
        for t in range(t_out):
            for ci in range(cin):
                for kk in range(kw):
                    var src = t + kk * dil - pad0
                    if src >= 0 and src < t_in:
                        col[t * j + ci * kw + kk] = x.fget((bi * cin + ci) * t_in + src)
                    else:
                        col[t * j + ci * kw + kk] = 0
        # out^T[t, co] = col[t, :] . W[co, :]
        var tmp = List[Float32](length=tmp_n, fill=0.0)
        matmul_2d(pool, Int(col.unsafe_ptr()), w.fptr(), Int(tmp.unsafe_ptr()), bias_ptr,
                  t_out, j, cout, True)
        for co in range(cout):
            for t in range(t_out):
                out.f[(bi * cout + co) * t_out + t] = tmp[t * cout + co]
    return out^
