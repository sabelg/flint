"""The decision recorder must agree with the raw and SIMD inertial search."""
from std.testing import assert_equal, assert_true
from flint.anim.motion_debug import MotionMatchCandidate, MotionMatchHistory
from flint.anim.motion_matching import MM_DIM, MM_LANES, MotionDatabase, MotionFeature


def feature(t: Float32) -> MotionFeature:
    var f = MotionFeature()
    f.traj_pos_800.x = t
    f.traj_facing_400 = t * Float32(0.4)
    f.left_foot_pos.y = t * Float32(0.3)
    f.hip_vel.z = t * Float32(0.8)
    f.ang_vel.y = t * Float32(0.2)
    return f


def check_score(db: MotionDatabase, candidate: MotionMatchCandidate, query: MotionFeature) raises:
    ref f = db.frames[candidate.index].feature
    var expected = Float32(0)
    if db.is_finalized:
        var q = Array[Float32, MM_DIM](fill=0)
        db._normalize_query(query, q)
        var lane = candidate.index % MM_LANES
        var group_offset = candidate.index // MM_LANES * MM_DIM * MM_LANES
        for d in range(MM_DIM):
            var diff = q[d] - db.soa_features[group_offset + d * MM_LANES + lane]
            expected += diff * diff
        if query.stance_tag >= 0 and f.stance_tag != query.stance_tag:
            expected += db.stance_mismatch_cost
    else:
        expected = db._raw_frame_cost(f, query)
    var grouped = candidate.bias
    for i in range(6):
        grouped += candidate.terms[i]
    var tolerance = Float32(1e-5) * max(Float32(1), abs(candidate.total))
    assert_true(abs(grouped - candidate.total) < tolerance)
    assert_true(abs(expected + candidate.bias - candidate.total) < tolerance)


def exercise(finalized: Bool) raises:
    var db = MotionDatabase()
    var q = feature(0.137)
    db.add_frame(0, 30, 1.0, feature(2.0))
    db.add_frame(0, 31, 1.03, q)  # Neighbour is a perfect match but excluded.
    for i in range(72):
        db.add_frame(1, 100 + i, Float32(i) / 30, feature(Float32(72 - i) / 100))
    if finalized:
        db.finalize_and_normalize()
    var history = MotionMatchHistory(2)
    var disabled = MotionMatchHistory()
    var expected = db.query_inertial(q, 0, 1.0)
    assert_equal(disabled.search(db, 5, q, 0, 1.0), expected)
    assert_equal(len(disabled.decisions), 0)
    assert_equal(history.search(db, 10, q, 0, 1.0), expected)
    var first = history.rewind(10).value().copy()
    assert_equal(first.winner.index, expected)
    assert_equal(first.continuation.index, 0)
    assert_equal(first.winner.source_frame, db.frames[expected].frame_index)
    assert_true(first.winner.index > 1)
    assert_true(first.alternative.index > 1 and first.alternative.index != expected)
    assert_true(first.winner.total <= first.alternative.total)
    assert_equal(first.continuation.bias, Float32(-0.08))
    for g in range(5):
        assert_true(first.continuation.terms[g] > 0)
    check_score(db, first.winner, q)
    check_score(db, first.alternative, q)
    check_score(db, first.continuation, q)
    q.stance_tag = 1
    assert_equal(history.search(db, 20, q, -1, 0), db.query_inertial(q, -1, 0))
    var second = history.rewind(25).value().copy()
    assert_equal(second.continuation.index, -1)
    assert_equal(second.winner.terms[5], db.stance_mismatch_cost)
    check_score(db, second.winner, q)
    assert_equal(history.search(db, 30, q, 0, 1.0, 10000), 0)
    var third = history.rewind(45).value().copy()
    assert_equal(third.winner.index, third.continuation.index)
    assert_equal(third.winner.bias, Float32(-10000))
    assert_equal(third.tick, 30)
    assert_equal(history.rewind(29).value().tick, 20)
    assert_true(not history.rewind(19))
    assert_equal(len(history.decisions), 2)


def main() raises:
    exercise(False)
    exercise(True)
    var history = MotionMatchHistory(3)
    var db = MotionDatabase()
    var q = MotionFeature()
    assert_equal(history.search(db, 1, q, -1, 0), -1)
    assert_equal(history.rewind(1).value().winner.index, -1)
    db.add_frame(3, 4, 0, q)
    assert_equal(history.search(db, 2, q, 3, 0), 0)
    assert_equal(history.rewind(2).value().alternative.index, -1)
    print("[PASS] Motion search winner, feature scores, continuation discount, exclusion and tick rewind (raw + SIMD8)")
