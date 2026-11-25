"""CPU kernels for the ONNX operators the interpreter supports (opset 17 semantics).

Heavy kernels (MatMul, Gemm, Conv) run on a `JobPool`, split by output columns, with 8-wide
SIMD inner loops; everything else is a plain loop, which is cheap for the small activations
of a motion model.
"""

from std.math import sqrt, exp, atan, cos, sin, ceil, floor, erf, tanh, log
from std.collections import Dict
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import (
    Tensor, DT_F32, DT_I32, DT_I64, DT_BOOL, numel_of, strides_of, shape_str,
    broadcast_shape, broadcast_offsets, norm_axis, fptr_at,
)

comptime LANES: Int = 8
comptime F8 = SIMD[DType.float32, LANES]


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
comptime U_EXP: Int = 10
comptime U_TANH: Int = 11
comptime U_SIGMOID: Int = 12
comptime U_RECIP: Int = 13
comptime U_FLOOR: Int = 14


def unary(op: Int, x: Tensor, alpha: Float32 = 0.01) raises -> Tensor:
    var n = x.numel()
    if op == U_NOT:
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
        elif op == U_COS:
            r = cos(v)
        elif op == U_SIN:
            r = sin(v)
        elif op == U_ABS:
            r = -v if v < 0 else v
        elif op == U_NEG:
            r = -v
        elif op == U_ERF:
            r = erf(v)
        elif op == U_EXP:
            r = exp(v)
        elif op == U_TANH:
            r = tanh(v)
        elif op == U_SIGMOID:
            r = 1.0 / (1.0 + exp(-v))
        elif op == U_RECIP:
            r = 1.0 / v
        elif op == U_FLOOR:
            r = floor(v)
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
comptime B_POW: Int = 10
comptime B_MIN: Int = 11
comptime B_OR: Int = 12


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
    if op == B_MIN:
        return a if a <= b else b
    if op == B_POW:
        if b == 2.0:
            return a * a
        if b == 0.5:
            return sqrt(a)
        if a > 0:
            return exp(b * log(a))
        if b == 0:
            return 1.0
        if a == 0:
            return 0.0
        # negative base: defined for integer exponents (odd ones keep the sign)
        if floor(b) == b:
            var m = exp(b * log(-a))
            return -m if Int(b) % 2 != 0 else m
        return Float32(0.0) / Float32(0.0)
    # fmod
    return a - b * Float32(Int(a / b))


@always_inline
def _iop(op: Int, a: Int, b: Int) -> Int:
    if op == B_ADD:
        return a + b
    if op == B_SUB:
        return a - b
    if op == B_MUL:
        return a * b
    if op == B_DIV:
        # C++ integer division truncates toward zero
        var q = a // b
        if (a % b != 0) and ((a < 0) != (b < 0)):
            q += 1
        return q
    if op == B_MAX:
        return a if a >= b else b
    if op == B_AND:
        return 1 if (a != 0 and b != 0) else 0
    if op == B_OR:
        return 1 if (a != 0 or b != 0) else 0
    if op == B_MIN:
        return a if a <= b else b
    if op == B_POW:
        var r = 1
        for _ in range(b):
            r *= a
        return r
    # Mod with fmod=0: the result takes the divisor's sign (Python's %)
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
    var out_shape = broadcast_shape(a.shape, b.shape)
    var n = numel_of(out_shape)
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
    var compare = op == B_EQ or op == B_LT or op == B_GT
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
        o[e] = _iop(op, a.i[ia], b.i[ib])
    return Tensor.ints(a.dtype, out_shape^, o^)


def where(c: Tensor, x: Tensor, y: Tensor) raises -> Tensor:
    var s = broadcast_shape(broadcast_shape(c.shape, x.shape), y.shape)
    var oc = broadcast_offsets(c.shape, s)
    var ox = broadcast_offsets(x.shape, s)
    var oy = broadcast_offsets(y.shape, s)
    var n = numel_of(s)
    var out = Tensor.like(x, s.copy())
    for e in range(n):
        var take_x = c.i[oc[e]] != 0
        if x.is_float():
            out.f[e] = x.fget(ox[e]) if take_x else y.fget(oy[e])
        else:
            out.i[e] = x.i[ox[e]] if take_x else y.i[oy[e]]
    return out^


def clip(x: Tensor, lo: Float64, hi: Float64) -> Tensor:
    var out = Tensor.like(x, x.shape.copy())
    for e in range(x.numel()):
        if x.is_float():
            out.f[e] = min(max(x.fget(e), Float32(lo)), Float32(hi))
        else:
            out.i[e] = Int(min(max(Float64(x.i[e]), lo), hi))
    return out^


def cast(x: Tensor, to: Int) raises -> Tensor:
    var n = x.numel()
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
            v = (1 if f != 0 else 0) if to == DT_BOOL else Int(f)
        else:
            v = (1 if x.i[e] != 0 else 0) if to == DT_BOOL else x.i[e]
        if to == DT_I32:
            v = Int(Int32(v))
        o[e] = v
    return Tensor.ints(to, x.shape.copy(), o^)


# -------------------------------------------------------------------------- data movement

def _copy_elem(mut dst: Tensor, di: Int, src: Tensor, si: Int):
    if src.is_float():
        dst.f[di] = src.fget(si)
    else:
        dst.i[di] = src.i[si]


def transpose(x: Tensor, perm: List[Int]) -> Tensor:
    var r = x.rank()
    var out_shape = List[Int](length=r, fill=1)
    for k in range(r):
        out_shape[k] = x.shape[perm[k]]
    var st = strides_of(x.shape)
    var pst = List[Int](length=r, fill=0)
    for k in range(r):
        pst[k] = st[perm[k]]
    var out = Tensor.like(x, out_shape.copy())
    var n = out.numel()
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
    var r = xs[0].rank()
    var axis = norm_axis(axis_in, r)
    var out_shape = xs[0].shape.copy()
    var total = 0
    for t in xs:
        total += t.shape[axis]
    out_shape[axis] = total
    var outer = 1
    for k in range(axis):
        outer *= out_shape[k]
    var inner = 1
    for k in range(axis + 1, r):
        inner *= out_shape[k]
    var out = Tensor.like(xs[0], out_shape^)
    var row = total * inner
    var off = 0
    for t in xs:
        var chunk = t.shape[axis] * inner
        for o in range(outer):
            for c in range(chunk):
                _copy_elem(out, o * row + off + c, t, o * chunk + c)
        off += chunk
    return out^


def gather(data: Tensor, idx: Tensor, axis_in: Int) raises -> Tensor:
    var r = data.rank()
    var axis = norm_axis(axis_in, r)
    var out_shape = List[Int]()
    for k in range(axis):
        out_shape.append(data.shape[k])
    for d in idx.shape:
        out_shape.append(d)
    for k in range(axis + 1, r):
        out_shape.append(data.shape[k])
    var outer = 1
    for k in range(axis):
        outer *= data.shape[k]
    var inner = 1
    for k in range(axis + 1, r):
        inner *= data.shape[k]
    var dim = data.shape[axis]
    var ni = idx.numel()
    var out = Tensor.like(data, out_shape^)
    for o in range(outer):
        for j in range(ni):
            var g = idx.int_at(j)
            if g < 0:
                g += dim
            if g < 0 or g >= dim:
                raise Error("Gather: index out of range")
            for c in range(inner):
                _copy_elem(out, (o * ni + j) * inner + c, data, (o * dim + g) * inner + c)
    return out^


def gather_elements(data: Tensor, idx: Tensor, axis_in: Int) raises -> Tensor:
    var r = data.rank()
    var axis = norm_axis(axis_in, r)
    var dst = strides_of(data.shape)
    var out = Tensor.like(data, idx.shape.copy())
    var n = idx.numel()
    var ctr = List[Int](length=r, fill=0)
    for e in range(n):
        var g = idx.int_at(e)
        if g < 0:
            g += data.shape[axis]
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
    var r = data.rank()
    var axis = norm_axis(axis_in, r)
    var dst = strides_of(data.shape)
    var out = data.reshaped(data.shape.copy())
    if out.is_float() and out.ext != 0:
        out = Tensor.floats(data.shape.copy(), data.owned_floats())
    var n = idx.numel()
    var ctr = List[Int](length=r, fill=0)
    for e in range(n):
        var g = idx.int_at(e)
        if g < 0:
            g += data.shape[axis]
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
    var out = data.reshaped(data.shape.copy())
    if out.is_float() and out.ext != 0:
        out = Tensor.floats(data.shape.copy(), data.owned_floats())
    var k = idx.shape[idx.rank() - 1]
    var dst = strides_of(data.shape)
    var slice_n = 1
    for d in range(k, data.rank()):
        slice_n *= data.shape[d]
    var rows = idx.numel() // k
    for j in range(rows):
        var off = 0
        for d in range(k):
            var g = idx.int_at(j * k + d)
            if g < 0:
                g += data.shape[d]
            off += g * dst[d]
        for c in range(slice_n):
            _copy_elem(out, off + c, upd, j * slice_n + c)
    return out^


def slice(data: Tensor, starts: List[Int], ends: List[Int], axes_in: List[Int], steps_in: List[Int]) raises -> Tensor:
    var r = data.rank()
    var st = List[Int](length=r, fill=0)
    var en = data.shape.copy()
    var sp = List[Int](length=r, fill=1)
    for j in range(len(starts)):
        var ax = norm_axis(axes_in[j] if len(axes_in) > 0 else j, r)
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
        var cnt: Int
        if sp[d] > 0:
            cnt = (en[d] - st[d] + sp[d] - 1) // sp[d]
        else:
            cnt = (st[d] - en[d] + (-sp[d]) - 1) // (-sp[d])
        out_shape[d] = max(cnt, 0)
    var out = Tensor.like(data, out_shape.copy())
    var n = out.numel()
    var dst = strides_of(data.shape)
    if data.is_float() and n > 0:
        # copy contiguous runs: everything after the last partial / strided axis is whole
        var last = -1
        for d in range(r):
            if st[d] != 0 or out_shape[d] != data.shape[d] or sp[d] != 1:
                last = d
        if last < 0 or sp[last] == 1:
            var run = 1
            var d0 = last if last >= 0 else 0
            for d in range(d0, r):
                run *= out_shape[d]
            var outer = n // run
            var ps = fptr_at(data.fptr())
            var pd = out.f.unsafe_ptr()
            var oc = List[Int](length=r, fill=0)
            for o in range(outer):
                var off = 0
                for d in range(d0):
                    off += (st[d] + oc[d] * sp[d]) * dst[d]
                if last >= 0:
                    off += st[last] * dst[last]
                for q in range(run):
                    pd.unsafe_offset(o * run + q).unsafe_store(ps.unsafe_offset(off + q).unsafe_load[alignment=1]())
                var d = d0 - 1
                while d >= 0:
                    oc[d] += 1
                    if oc[d] < out_shape[d]:
                        break
                    oc[d] = 0
                    d -= 1
            return out^
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
    var s = broadcast_shape(x.shape, shape)
    var offs = broadcast_offsets(x.shape, s)
    var out = Tensor.like(x, s.copy())
    for e in range(len(offs)):
        _copy_elem(out, e, x, offs[e])
    return out^


def tile(x: Tensor, reps: List[Int]) raises -> Tensor:
    var r = x.rank()
    var out_shape = List[Int](length=r, fill=0)
    for d in range(r):
        out_shape[d] = x.shape[d] * reps[d]
    var out = Tensor.like(x, out_shape.copy())
    var dst = strides_of(x.shape)
    var ctr = List[Int](length=r, fill=0)
    for e in range(out.numel()):
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
    var s = start.get(0)
    var l = limit.get(0)
    var d = delta.get(0)
    var n = max(Int(ceil((l - s) / d)), 0)
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
    var out_shape = idx.shape.copy()
    out_shape.append(depth)
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
    var r = x.rank()
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

def _outer_inner(shape: List[Int], axis: Int) -> Tuple[Int, Int, Int]:
    var outer = 1
    for k in range(axis):
        outer *= shape[k]
    var inner = 1
    for k in range(axis + 1, len(shape)):
        inner *= shape[k]
    return (outer, shape[axis], inner)


def reduce(x: Tensor, axes_in: List[Int], keepdims: Bool, l2: Bool) raises -> Tensor:
    var r = x.rank()
    var red = List[Bool](length=r, fill=False)
    if len(axes_in) == 0:
        for d in range(r):
            red[d] = True
    for a in axes_in:
        red[norm_axis(a, r)] = True
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
        o[k] = Int(acc[k])
    return Tensor.ints(x.dtype, out_shape^, o^)


def argmax(x: Tensor, axis_in: Int, keepdims: Bool) raises -> Tensor:
    var axis = norm_axis(axis_in, x.rank())
    var oi = _outer_inner(x.shape, axis)
    var out_shape = List[Int]()
    for d in range(x.rank()):
        if d != axis:
            out_shape.append(x.shape[d])
        elif keepdims:
            out_shape.append(1)
    var o = List[Int](length=oi[0] * oi[2], fill=0)
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
    var axis = norm_axis(axis_in, x.rank())
    var oi = _outer_inner(x.shape, axis)
    var out_shape = x.shape.copy()
    out_shape[axis] = k
    var vals = Tensor.like(x, out_shape.copy())
    var inds = Tensor.ints(DT_I64, out_shape.copy(), List[Int](length=numel_of(out_shape), fill=0))
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
    var axis = norm_axis(axis_in, x.rank())
    var oi = _outer_inner(x.shape, axis)
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


struct _SoftCtx(Movable):
    var src: Int
    var dst: Int
    var rows: Int
    var cols: Int
    var rows_per_job: Int

    def __init__(out self, src: Int, dst: Int, rows: Int, cols: Int, rows_per_job: Int):
        self.src = src
        self.dst = dst
        self.rows = rows
        self.cols = cols
        self.rows_per_job = rows_per_job


def _softmax_rows(src: Int, dst: Int, cols: Int, r0: Int, r1: Int):
    var ps = fptr_at(src)
    var pd = fptr_at(dst)
    for r in range(r0, r1):
        var base = r * cols
        var mv = F8(-3.4e38)
        var j = 0
        while j + LANES <= cols:
            mv = max(mv, ps.unsafe_offset(base + j).unsafe_load[width=LANES, alignment=1]())
            j += LANES
        var m = mv.reduce_max()
        while j < cols:
            m = max(m, ps.unsafe_offset(base + j).unsafe_load[alignment=1]())
            j += 1
        var sv = F8(0)
        j = 0
        while j + LANES <= cols:
            var e = exp(ps.unsafe_offset(base + j).unsafe_load[width=LANES, alignment=1]() - F8(m))
            pd.unsafe_offset(base + j).unsafe_store[alignment=1](e)
            sv += e
            j += LANES
        var tot = sv.reduce_add()
        while j < cols:
            var e = exp(ps.unsafe_offset(base + j).unsafe_load[alignment=1]() - m)
            pd.unsafe_offset(base + j).unsafe_store[alignment=1](e)
            tot += e
            j += 1
        var inv = F8(1.0 / tot)
        j = 0
        while j + LANES <= cols:
            pd.unsafe_offset(base + j).unsafe_store[alignment=1](pd.unsafe_offset(base + j).unsafe_load[width=LANES, alignment=1]() * inv)
            j += LANES
        while j < cols:
            pd.unsafe_offset(base + j).unsafe_store[alignment=1](pd.unsafe_offset(base + j).unsafe_load[alignment=1]() * (1.0 / tot))
            j += 1


def _softmax_worker(ctx_addr: Int, job: Int):
    var ctx = Pointer[_SoftCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var r0 = job * ctx[].rows_per_job
    var r1 = min(ctx[].rows, r0 + ctx[].rows_per_job)
    if r0 < r1:
        _softmax_rows(ctx[].src, ctx[].dst, ctx[].cols, r0, r1)


def softmax_rows(mut pool: JobPool, x: Tensor) raises -> Tensor:
    """Softmax over the last axis, rows spread over the pool with 8-wide SIMD."""
    var cols = x.shape[x.rank() - 1]
    var rows = x.numel() // cols if cols > 0 else 0
    var out = Tensor.zeros(x.shape.copy())
    var src = x.fptr()
    var dst = out.fptr()
    if rows * cols < 1 << 16 or not pool.ready():
        _softmax_rows(src, dst, cols, 0, rows)
        return out^
    var jobs = min(256, rows)
    var per = (rows + jobs - 1) // jobs
    jobs = (rows + per - 1) // per
    var ctx = _SoftCtx(src, dst, rows, cols, per)
    run_parallel_ctx(pool, jobs, ctx, _softmax_worker)
    return out^


def softmax(x: Tensor, axis_in: Int) raises -> Tensor:
    var axis = norm_axis(axis_in, x.rank())
    var oi = _outer_inner(x.shape, axis)
    var out = Tensor.like(x, x.shape.copy())
    for a in range(oi[0]):
        for c in range(oi[2]):
            var m = x.fget(a * oi[1] * oi[2] + c)
            for j in range(1, oi[1]):
                m = max(m, x.fget((a * oi[1] + j) * oi[2] + c))
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
    var n = x.shape[x.rank() - 1]
    var rows = x.numel() // n
    var out = Tensor.like(x, x.shape.copy())
    for r in range(rows):
        var base = r * n
        var mean: Float32 = 0
        for j in range(n):
            mean += x.fget(base + j)
        mean /= Float32(n)
        var var_: Float32 = 0
        for j in range(n):
            var d = x.fget(base + j) - mean
            var_ += d * d
        var_ /= Float32(n)
        var inv = 1.0 / sqrt(var_ + eps)
        for j in range(n):
            var v = (x.fget(base + j) - mean) * inv * scale.fget(j)
            if has_bias:
                v += bias.fget(j)
            out.f[base + j] = v
    return out^


def batch_norm(x: Tensor, scale: Tensor, bias: Tensor, mean: Tensor, var_: Tensor, eps: Float32) raises -> Tensor:
    """Inference BatchNormalization over channel axis 1 (running statistics)."""
    if x.rank() < 2:
        raise Error("BatchNormalization: rank >= 2 expected")
    var C = x.shape[1]
    var inner = 1
    for d in range(2, x.rank()):
        inner *= x.shape[d]
    var outer = x.numel() // (C * inner)
    var mul = List[Float32](length=C, fill=0.0)
    var add = List[Float32](length=C, fill=0.0)
    for c in range(C):
        mul[c] = scale.fget(c) / sqrt(var_.fget(c) + eps)
        add[c] = bias.fget(c) - mean.fget(c) * mul[c]
    var out = Tensor.like(x, x.shape.copy())
    for o in range(outer):
        for c in range(C):
            var base = (o * C + c) * inner
            for i in range(inner):
                out.f[base + i] = x.fget(base + i) * mul[c] + add[c]
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
    var rows_per_job: Int
    var col_jobs: Int

    def __init__(out self, a: Int, b: Int, c: Int, bias: Int, m: Int, k: Int, n: Int, cols_per_job: Int, trans_b: Bool):
        self.rows_per_job = m
        self.col_jobs = 1
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
    var cj = job % ctx[].col_jobs
    var rj = job // ctx[].col_jobs
    var n0 = cj * ctx[].cols_per_job
    var n1 = min(ctx[].n, n0 + ctx[].cols_per_job)
    var m0 = rj * ctx[].rows_per_job
    var m1 = min(ctx[].m, m0 + ctx[].rows_per_job)
    if n0 >= n1 or m0 >= m1:
        return
    var a = ctx[].a + m0 * ctx[].k * 4
    var c = ctx[].c + m0 * ctx[].n * 4
    if ctx[].trans_b:
        _dot_cols(a, ctx[].b, c, ctx[].bias, m1 - m0, ctx[].k, ctx[].n, n0, n1)
    else:
        _mat_cols(a, ctx[].b, c, ctx[].bias, m1 - m0, ctx[].k, ctx[].n, n0, n1)


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
    ctx.col_jobs = jobs
    # few columns: also split the rows so every worker has a share
    var row_jobs = 1
    if jobs < 32 and m >= 64:
        row_jobs = min((64 + jobs - 1) // jobs, m // 16)
    var rows = (m + row_jobs - 1) // row_jobs
    rows = (rows + 3) // 4 * 4
    row_jobs = (m + rows - 1) // rows
    ctx.rows_per_job = rows
    run_parallel_ctx(pool, jobs * row_jobs, ctx, _mat_worker)


def _contiguous(t: Tensor) -> Tensor:
    if t.ext != 0 or len(t.f) == t.numel():
        return t.reshaped(t.shape.copy())
    return Tensor.floats(t.shape.copy(), t.owned_floats())


def matmul(mut pool: JobPool, a: Tensor, b: Tensor) raises -> Tensor:
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
    var oa = broadcast_offsets(ba, bs)
    var ob = broadcast_offsets(bb, bs)
    var out_shape = bs.copy()
    out_shape.append(m)
    out_shape.append(n)
    var out = Tensor.zeros(out_shape^)
    var batches = numel_of(bs)
    var pa = a.fptr()
    var pb = b.fptr()
    var pc = out.fptr()
    for t in range(batches):
        matmul_2d(pool, pa + oa[t] * m * k * 4, pb + ob[t] * k * n * 4, pc + t * m * n * 4, 0, m, k, n, False)
    return out^


def gemm(mut pool: JobPool, a: Tensor, b: Tensor, c: Tensor, has_c: Bool, alpha: Float32, beta: Float32,
         trans_a: Bool, trans_b: Bool) raises -> Tensor:
    if trans_a or alpha != 1 or (has_c and beta != 1):
        raise Error("Gemm: only alpha=beta=1 without transA is supported")
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[0] if trans_b else b.shape[1]
    var out = Tensor.zeros([m, n])
    var bias = 0
    var cfull = Tensor()
    if has_c:
        if c.numel() == n:
            bias = c.fptr()
        else:
            cfull = expand(c, [m, n])
    matmul_2d(pool, a.fptr(), b.fptr(), out.fptr(), bias, m, k, n, trans_b)
    if has_c and bias == 0:
        for e in range(m * n):
            out.f[e] += cfull.f[e]
    return out^


def conv1d(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pad0: Int, pad1: Int, dil: Int,
           stride: Int, group: Int) raises -> Tensor:
    if x.rank() != 3 or w.rank() != 3 or group != 1 or stride != 1:
        raise Error("Conv: only 1-D, group 1, stride 1 is supported")
    var batch = x.shape[0]
    var cin = x.shape[1]
    var t_in = x.shape[2]
    var cout = w.shape[0]
    var kw = w.shape[2]
    var t_out = t_in + pad0 + pad1 - dil * (kw - 1)
    var j = cin * kw
    var out = Tensor.zeros([batch, cout, t_out])
    # im2col^T: col[t, ci*kw + kk] so each output row is a dot product with W[co, :]
    var col = List[Float32](length=t_out * j, fill=0.0)
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
        var tmp = List[Float32](length=t_out * cout, fill=0.0)
        matmul_2d(pool, Int(col.unsafe_ptr()), w.fptr(), Int(tmp.unsafe_ptr()), b.fptr() if has_b else 0,
                  t_out, j, cout, True)
        for co in range(cout):
            for t in range(t_out):
                out.f[(bi * cout + co) * t_out + t] = tmp[t * cout + co]
    return out^


def _permuted(x: Tensor, perm: List[Int]) -> Tensor:
    """`x` transposed by `perm` and contiguous; an identity permutation shares a mapping
    instead of copying."""
    var ident = True
    for i in range(len(perm)):
        if perm[i] != i:
            ident = False
    if ident and x.is_float() and (x.ext != 0 or len(x.f) == x.numel()):
        var v = Tensor()
        v.dtype = x.dtype
        v.shape = x.shape.copy()
        v.ext = x.fptr()      # a view: only used while `x` is alive
        return v^
    return transpose(x, perm)


def einsum(mut pool: JobPool, equation: String, a: Tensor, b: Tensor) raises -> Tensor:
    """Two-operand Einsum as one batched MatMul: labels in both inputs and the output are batch
    labels, labels in both inputs only are contracted, the rest are the free rows / columns."""
    var parts = equation.replace(" ", "").split("->")
    if len(parts) != 2:
        raise Error("Einsum: needs an explicit output (" + equation + ")")
    var ins = parts[0].split(",")
    if len(ins) != 2:
        raise Error("Einsum: only two operands are supported (" + equation + ")")
    var la = String(ins[0])
    var lb = String(ins[1])
    var lo = String(parts[1])
    if la.byte_length() != a.rank() or lb.byte_length() != b.rank():
        raise Error("Einsum: label count does not match operand rank (" + equation + ")")
    var batch = List[Int]()      # positions in the output order
    var contract = List[String]()
    var free_a = List[String]()
    var free_b = List[String]()
    var size = Dict[String, Int]()
    for i in range(la.byte_length()):
        size[String(la[byte=i])] = a.shape[i]
    for i in range(lb.byte_length()):
        var c = String(lb[byte=i])
        if c in size and size[c] != b.shape[i]:
            raise Error("Einsum: size mismatch for label " + c)
        size[c] = b.shape[i]
    var batch_l = List[String]()
    for i in range(lo.byte_length()):
        var c = String(lo[byte=i])
        var in_a = la.find(c) >= 0
        var in_b = lb.find(c) >= 0
        if in_a and in_b:
            batch_l.append(c)
        elif in_a:
            free_a.append(c)
        elif in_b:
            free_b.append(c)
        else:
            raise Error("Einsum: output label " + c + " is in no input")
    for i in range(la.byte_length()):
        var c = String(la[byte=i])
        if lo.find(c) < 0:
            if lb.find(c) < 0:
                raise Error("Einsum: summing a label of one operand is not supported")
            contract.append(c)
    _ = batch
    var perm_a = List[Int]()
    var perm_b = List[Int]()
    var bsz = 1
    var m = 1
    var n = 1
    var kk = 1
    for c in batch_l:
        perm_a.append(la.find(c))
        perm_b.append(lb.find(c))
        bsz *= size[c]
    for c in free_a:
        perm_a.append(la.find(c))
        m *= size[c]
    for c in contract:
        perm_a.append(la.find(c))
        perm_b.append(lb.find(c))
        kk *= size[c]
    for c in free_b:
        perm_b.append(lb.find(c))
        n *= size[c]
    var ta = _permuted(a, perm_a).reshaped([bsz, m, kk])
    var tb = _permuted(b, perm_b).reshaped([bsz, kk, n])
    var prod = matmul(pool, ta, tb)
    # prod labels: batch_l + free_a + free_b -> permute to the output order
    var cur = List[String]()
    var cur_shape = List[Int]()
    for c in batch_l:
        cur.append(c)
        cur_shape.append(size[c])
    for c in free_a:
        cur.append(c)
        cur_shape.append(size[c])
    for c in free_b:
        cur.append(c)
        cur_shape.append(size[c])
    var r = prod.reshaped(cur_shape^)
    var perm_o = List[Int]()
    var identity = True
    for i in range(lo.byte_length()):
        var c = String(lo[byte=i])
        for j in range(len(cur)):
            if cur[j] == c:
                perm_o.append(j)
                if j != i:
                    identity = False
    if identity:
        return r^
    return transpose(r, perm_o)
