"""Sota:perf r3: a radix `stable_order` for the cluster-DAG build.

`flint.core.detsort.stable_order` is a bottom-up merge sort that reads every key through the index array, so on
the Forest's 2.4M-triangle mesh (7.2M edge keys) each of its ~23 passes is a pass of random reads. This is an LSD
radix sort of (key, index) pairs, 11 bits per pass, with passes over constant digits skipped. A stable sort's
permutation is unique, so the result is identical to `stable_order` for every input; negative keys and short
lists go to `stable_order` itself.
"""
from std.collections import List
from std.math import max
from flint.core.detsort import stable_order

comptime _RADIX_BITS: Int = 11
comptime _RADIX: Int = 1 << _RADIX_BITS
comptime _RADIX_MIN_N: Int = 1024


def sota_stable_order(keys: List[Int]) -> List[Int]:
    """Indices that visit `keys` in ascending order, ties in index order (== `stable_order(keys)`)."""
    var n = len(keys)
    if n < _RADIX_MIN_N:
        return stable_order(keys)
    var mx = 0
    for i in range(n):
        var k = keys[i]
        if k < 0:
            return stable_order(keys)
        mx = max(mx, k)
    var bits = 0
    while bits < 63 and (mx >> bits) > 0:
        bits += 1
    var ka = keys.copy()
    var kb = List[Int](length=n, fill=0)
    var oa = List[Int](capacity=n)
    for i in range(n):
        oa.append(i)
    var ob = List[Int](length=n, fill=0)
    var counts = List[Int](length=_RADIX, fill=0)
    var shift = 0
    while shift < bits:
        for d in range(_RADIX):
            counts[d] = 0
        var pk = ka.unsafe_ptr()
        for i in range(n):
            counts[(pk[unsafe_offset=i] >> shift) & (_RADIX - 1)] += 1
        # one bucket holds every key: this digit is constant, the pass would be the identity
        if counts[(pk[unsafe_offset=0] >> shift) & (_RADIX - 1)] == n:
            shift += _RADIX_BITS
            continue
        var run = 0
        for d in range(_RADIX):
            var c = counts[d]
            counts[d] = run
            run += c
        var po = oa.unsafe_ptr()
        var qk = kb.unsafe_ptr()
        var qo = ob.unsafe_ptr()
        for i in range(n):
            var k = pk[unsafe_offset=i]
            var d = (k >> shift) & (_RADIX - 1)
            var at = counts[d]
            counts[d] = at + 1
            qk[unsafe_offset=at] = k
            qo[unsafe_offset=at] = po[unsafe_offset=i]
        var tk = ka^
        ka = kb^
        kb = tk^
        var to = oa^
        oa = ob^
        ob = to^
        shift += _RADIX_BITS
    _ = len(kb)
    _ = len(ob)
    return oa^
