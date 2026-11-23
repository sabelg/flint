"""Reduce synchronized baked keys and verify world-space error on the skinned rig."""
from std.math import isfinite, sqrt
from flint.assets.model import NativeAnimation, NativeModel
from flint.core.math import Vec3


def _markers(mut model: NativeModel, index: Int, t: Float32) -> List[Vec3]:
    model.sample_clip(index, t)
    var transforms = model.global_transforms()
    var result = List[Vec3]()
    for m in transforms:
        result.append(m.transform_point(Vec3()))
        # Eccentric markers detect twist on leaf joints as well as position changes.
        result.append(m.transform_point(Vec3(0.08, 0, 0)))
        result.append(m.transform_point(Vec3(0, 0.08, 0)))
    return result^


def _keys(a: NativeAnimation, keep: List[Bool]) -> NativeAnimation:
    var result = a.copy()
    for idx in range(len(result.channels)):
        ref ch = result.channels[idx]
        var width = 4 if ch.path == 1 else 3
        var values = List[Float32]()
        var times = List[Float32]()
        for i in range(len(keep)):
            if keep[i]:
                times.append(ch.times[i])
                for j in range(width):
                    values.append(ch.values[i * width + j])
        ch.times = times^
        ch.values = values^
    return result^


def optimize_bake(
    mut model: NativeModel, index: Int, tolerance: Float32
) raises -> Float32:
    if index < 0 or index >= len(model.animations) or not isfinite(tolerance) or tolerance <= 0:
        raise Error("A positive optimization tolerance is required")
    var source = model.animations[index].copy()
    if len(source.channels) == 0:
        raise Error("Animation has no channels to optimize")
    var n = len(source.channels[0].times)
    if n < 2:
        raise Error("Animation requires at least two keyframes to optimize")
    for idx in range(len(source.channels)):
        ref ch = source.channels[idx]
        var width = 4 if ch.path == 1 else 3
        if len(ch.times) != n or len(ch.values) < n * width:
            raise Error("Animation channels must have synchronized keyframe counts")
        for f in range(n):
            if not isfinite(ch.times[f]):
                raise Error("Animation channel keyframe times must be finite")
            if f > 0 and ch.times[f] < ch.times[f - 1]:
                raise Error("Animation channel keyframe times must be monotonically non-decreasing")
        for v in ch.values:
            if not isfinite(v):
                raise Error("Animation channel keyframe values must be finite")
    ref times = source.channels[0].times
    var originals = List[List[Vec3]]()
    var samples = List[Float32]()
    for f in range(n):
        samples.append(times[f])
        if f + 1 < n:
            var mid = (times[f] + times[f + 1]) * 0.5
            if not isfinite(mid):
                raise Error("Animation sample midpoint must be finite")
            samples.append(mid)
    for t in samples:
        var m = _markers(model, index, t)
        for p in m:
            if not isfinite(p.x) or not isfinite(p.y) or not isfinite(p.z):
                raise Error("Animation marker transform produced non-finite coordinates")
        originals.append(m^)
    var keep = List[Bool](length=n, fill=False)
    keep[0] = True
    keep[n - 1] = True
    var pending: List[Tuple[Int, Int]] = [(0, n - 1)]
    var peak: Float32 = 0
    while pending:
        model.animations[index] = _keys(source, keep)
        var next = List[Tuple[Int, Int]]()
        for interval in pending:
            var a = interval[0]
            var b = interval[1]
            var worst: Float32 = 0
            var at = a
            for s in range(2 * a, 2 * b + 1):
                var cand = _markers(model, index, samples[s])
                var err: Float32 = 0
                for j in range(len(cand)):
                    var sq = (cand[j] - originals[s][j]).length_squared()
                    if not isfinite(sq):
                        model.animations[index] = source^
                        raise Error("Animation optimization produced non-finite marker error")
                    err = max(err, sq)
                if err > worst:
                    worst = err
                    at = (s + 1) // 2
            if worst > tolerance * tolerance and b - a > 1:
                at = max(a + 1, min(b - 1, at))
                keep[at] = True
                next.append((a, at))
                next.append((at, b))
            else:
                peak = max(peak, worst)
        pending = next^

    if not isfinite(peak):
        model.animations[index] = source^
        raise Error("Animation optimization produced non-finite peak error")

    print(
        "Optimized",
        n,
        "to",
        len(model.animations[index].channels[0].times),
        "keys per track; peak marker error",
        sqrt(peak),
    )
    return sqrt(peak)
