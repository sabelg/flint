"""Optional tick history and feature costs for inertial motion matching."""
from flint.anim.motion_matching import MM_DIM, MM_LANES, MotionDatabase, MotionFeature


struct MotionMatchCandidate(Copyable, Movable):
    var index: Int
    var clip: Int
    var source_frame: Int
    var time_seconds: Float32
    # Trajectory position, facing, feet, linear velocity, angular velocity, stance.
    var terms: Array[Float32, 6]
    var bias: Float32
    var total: Float32

    def __init__(out self):
        self.index = -1
        self.clip = -1
        self.source_frame = -1
        self.time_seconds = 0
        self.terms = Array[Float32, 6](fill=0)
        self.bias = 0
        self.total = 0


@fieldwise_init
struct MotionMatchDecision(Copyable, Movable):
    var tick: Int
    var query: MotionFeature
    var winner: MotionMatchCandidate
    var continuation: MotionMatchCandidate
    var alternative: MotionMatchCandidate


def _candidate(
    db: MotionDatabase, index: Int, query: MotionFeature, normalized: Array[Float32, MM_DIM],
    continuation: Int, continuing_bias: Float32,
) -> MotionMatchCandidate:
    var out = MotionMatchCandidate()
    ref frame = db.frames[index]
    out.index = index
    out.clip = frame.clip_index
    out.source_frame = frame.frame_index
    out.time_seconds = frame.time_seconds
    if db.is_finalized:
        # The SIMD search accumulates the weighted z-score channels in dimension order.
        # Read the same lane from the SoA that the search used, including baked weights.
        var offset = (index // MM_LANES) * MM_DIM * MM_LANES + index % MM_LANES
        var bounds = [0, 9, 15, 27, 30, MM_DIM]
        for group in range(5):
            for d in range(bounds[group], bounds[group + 1]):
                var diff = db.soa_features[offset + d * MM_LANES] - normalized[d]
                var cost = diff * diff
                out.terms[group] += cost
                out.total += cost
        if query.stance_tag >= 0 and frame.feature.stance_tag != query.stance_tag:
            out.terms[5] = db.stance_mismatch_cost
            out.total += out.terms[5]
    else:
        out.terms = db.raw_cost_terms(frame.feature, query)
        for i in range(6):
            out.total += out.terms[i]
    if index == continuation:
        out.bias = -continuing_bias
        out.total += out.bias
    return out^


struct MotionMatchHistory(Movable):
    """A bounded ring. Capacity zero uses the normal search without scoring alternative frames."""
    var capacity: Int
    var decisions: List[MotionMatchDecision]
    var next_slot: Int

    def __init__(out self, capacity: Int = 0):
        self.capacity = max(0, capacity)
        self.decisions = List[MotionMatchDecision](capacity=self.capacity)
        self.next_slot = 0

    def search(
        mut self, db: MotionDatabase, tick: Int, query: MotionFeature, cur_clip: Int, cur_time: Float32,
        continuing_bias: Float32 = 0.08, exclude_window: Float32 = 0.12,
    ) -> Int:
        var selected = db.query_inertial(query, cur_clip, cur_time, continuing_bias, exclude_window)
        if self.capacity == 0:
            return selected
        var cont = db.continuation_frame(cur_clip, cur_time, exclude_window)
        var normalized = Array[Float32, MM_DIM](fill=0)
        if db.is_finalized:
            db._normalize_query(query, normalized)
        var winning = MotionMatchCandidate()
        var continuing = MotionMatchCandidate()
        var alternative = MotionMatchCandidate()
        for i in range(db.count()):
            if db.excludes_frame(i, cur_clip, cur_time, cont, exclude_window):
                continue
            var candidate = _candidate(db, i, query, normalized, cont, continuing_bias)
            if i == cont:
                continuing = candidate.copy()
            if i == selected:
                winning = candidate^
            elif alternative.index < 0 or candidate.total < alternative.total:
                alternative = candidate^
        var decision = MotionMatchDecision(tick, query, winning^, continuing^, alternative^)
        if len(self.decisions) < self.capacity:
            self.decisions.append(decision^)
        else:
            self.decisions[self.next_slot] = decision^
        self.next_slot = (self.next_slot + 1) % self.capacity
        return selected

    def rewind(self, tick: Int) -> Optional[MotionMatchDecision]:
        """Returns the most recently recorded decision at or before this simulation tick."""
        var n = len(self.decisions)
        for back in range(n):
            var index = (self.next_slot + n - 1 - back) % n
            if self.decisions[index].tick <= tick:
                return self.decisions[index].copy()
        return None
