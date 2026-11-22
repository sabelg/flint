"""Reduce synchronized baked keys and verify world-space error on the skinned rig."""
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
    if index < 0 or index >= len(model.animations) or tolerance <= 0:
        raise Error("A positive optimization tolerance is required")
    var source = model.animations[index].copy()
    ref times = source.channels[0].times
    var n = len(times)
    var originals = List[List[Vec3]]()
    var samples = List[Float32]()
    for f in range(n):
        samples.append(times[f])
        if f + 1 < n:
            samples.append((times[f] + times[f + 1]) * 0.5)
    for t in samples:
        originals.append(_markers(model, index, t))
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
                    err = max(err, (cand[j] - originals[s][j]).length_squared())
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
    from std.math import sqrt

    print(
        "Optimized",
        n,
        "to",
        len(model.animations[index].channels[0].times),
        "keys per track; peak marker error",
        sqrt(peak),
    )
    return sqrt(peak)
