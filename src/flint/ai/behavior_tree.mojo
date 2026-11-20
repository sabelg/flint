"""Flat DFS Behavior Trees, Compensated Utility AI & SIMD8 Spatial Perception (`flint.ai.behavior_tree`, v5).

Implements:
1. `Blackboard`: zero-allocation typed simulation blackboard (`Float32`, `Int`, `Vec3`).
2. `UtilityAiScorer`: Dave Mark Infinite-Axis Utility System with response curves
   (`CURVE_LINEAR`, `CURVE_POLYNOMIAL`, `CURVE_LOGISTIC`, `CURVE_INVERSE`) and both
   geometric-mean (`(prod s_i)^(1/k)`) and `(1 - 1/k)` make-up score compensation.
3. Flat DFS pre-order `BehaviorTree` (`BtNode` with O(1) `subtree_skip` jumps,
   `Sequence`, `Selector`, `Parallel`, `Inverter`, `Cooldown`, `BlackboardCmp`,
   `UtilitySelect`, `Action`, and reactive `BT_ABORT_LOWER_PRIORITY` / `BT_ABORT_SELF`).
4. SoA 8-wide SIMD `SpatialPerception` combining `F8` vision-cone frustum culling,
   `PhysicsWorld.raycast` line-of-sight wall occlusion, and inverse-square acoustic hearing.
"""
from std.collections import List
from std.math import sqrt, abs, min, max
from flint.core.math import Vec3
from flint.core.detmath import det_cos, det_exp, det_pow
from flint.physics.filter import CollisionFilter, CAT_ALL
from flint.physics.solver_bodies import F8, LANES
from flint.physics.world import PhysicsWorld

comptime BT_SUCCESS: UInt8 = 0
comptime BT_FAILURE: UInt8 = 1
comptime BT_RUNNING: UInt8 = 2

comptime BT_NODE_SEQUENCE: UInt8 = 0
comptime BT_NODE_SELECTOR: UInt8 = 1
comptime BT_NODE_PARALLEL: UInt8 = 2
comptime BT_NODE_INVERTER: UInt8 = 3
comptime BT_NODE_COOLDOWN: UInt8 = 4
comptime BT_NODE_BLACKBOARD_CMP: UInt8 = 5
comptime BT_NODE_UTILITY_SELECT: UInt8 = 6
comptime BT_NODE_ACTION: UInt8 = 7

comptime BT_ABORT_NONE: UInt8 = 0
comptime BT_ABORT_SELF: UInt8 = 1
comptime BT_ABORT_LOWER_PRIORITY: UInt8 = 2
comptime BT_ABORT_BOTH: UInt8 = 3

comptime BT_CMP_LT: UInt8 = 0
comptime BT_CMP_LE: UInt8 = 1
comptime BT_CMP_GT: UInt8 = 2
comptime BT_CMP_GE: UInt8 = 3
comptime BT_CMP_EQ: UInt8 = 4
comptime BT_CMP_NE: UInt8 = 5

comptime CURVE_LINEAR: UInt8 = 0
comptime CURVE_POLYNOMIAL: UInt8 = 1
comptime CURVE_LOGISTIC: UInt8 = 2
comptime CURVE_INVERSE: UInt8 = 3

comptime _AI_DEG_TO_RAD: Float32 = 0.01745329252


# =========================================================================
# 1. Blackboard
# =========================================================================

struct Blackboard(Copyable, Movable):
    var names: List[String]
    var floats: List[Float32]
    var ints: List[Int]
    var vecs: List[Vec3]

    def __init__(out self):
        self.names = List[String]()
        self.floats = List[Float32]()
        self.ints = List[Int]()
        self.vecs = List[Vec3]()

    def def_key(mut self, name: String, initial_f32: Float32 = 0.0, initial_int: Int = 0, initial_vec: Vec3 = Vec3()) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        var idx = len(self.names)
        self.names.append(name)
        self.floats.append(initial_f32)
        self.ints.append(initial_int)
        self.vecs.append(initial_vec)
        return idx

    def find_key(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    @always_inline
    def set_float(mut self, key: Int, val: Float32):
        if key >= 0 and key < len(self.floats):
            self.floats[key] = val

    @always_inline
    def get_float(self, key: Int) -> Float32:
        if key >= 0 and key < len(self.floats):
            return self.floats[key]
        return Float32(0.0)

    @always_inline
    def set_int(mut self, key: Int, val: Int):
        if key >= 0 and key < len(self.ints):
            self.ints[key] = val

    @always_inline
    def get_int(self, key: Int) -> Int:
        if key >= 0 and key < len(self.ints):
            return self.ints[key]
        return 0

    @always_inline
    def set_vec3(mut self, key: Int, val: Vec3):
        if key >= 0 and key < len(self.vecs):
            self.vecs[key] = val

    @always_inline
    def get_vec3(self, key: Int) -> Vec3:
        if key >= 0 and key < len(self.vecs):
            return self.vecs[key]
        return Vec3()


# =========================================================================
# 2. Dave Mark Geometric-Mean / Make-Up Compensated Utility AI Scorer
# =========================================================================

struct UtilityConsideration(Copyable, ImplicitlyCopyable, Movable):
    var bb_key: Int
    var input_min: Float32
    var input_max: Float32
    var curve_kind: UInt8
    var exponent: Float32
    var invert: Bool

    def __init__(
        out self,
        bb_key: Int,
        input_min: Float32 = 0.0,
        input_max: Float32 = 1.0,
        curve_kind: UInt8 = CURVE_LINEAR,
        exponent: Float32 = 1.0,
        invert: Bool = False,
    ):
        self.bb_key = bb_key
        self.input_min = input_min
        self.input_max = input_max
        self.curve_kind = curve_kind
        self.exponent = exponent
        self.invert = invert

    def evaluate(self, bb: Blackboard) -> Float32:
        var raw = bb.get_float(self.bb_key)
        var denom = self.input_max - self.input_min
        var x = max(Float32(0.0), min(Float32(1.0), (raw - self.input_min) / denom)) if abs(denom) > Float32(1.0e-7) else Float32(0.0)
        var y: Float32
        if self.curve_kind == CURVE_POLYNOMIAL:
            y = det_pow(x, max(Float32(0.05), self.exponent))
        elif self.curve_kind == CURVE_LOGISTIC:
            var k = max(Float32(0.5), self.exponent * Float32(10.0))
            y = Float32(1.0) / (Float32(1.0) + det_exp(-k * (x - Float32(0.5))))
        elif self.curve_kind == CURVE_INVERSE:
            y = Float32(1.0) / (Float32(1.0) + max(Float32(0.1), self.exponent) * x)
        else:
            y = x
        y = max(Float32(0.0), min(Float32(1.0), y))
        return (Float32(1.0) - y) if self.invert else y


struct UtilityOption(Copyable, Movable):
    var option_id: Int
    var weight: Float32
    var considerations: List[UtilityConsideration]

    def __init__(out self, option_id: Int, weight: Float32 = 1.0):
        self.option_id = option_id
        self.weight = weight
        self.considerations = List[UtilityConsideration]()

    def add_consideration(mut self, c: UtilityConsideration):
        self.considerations.append(c)


struct UtilityAiScorer(Copyable, Movable):
    """Dave Mark Utility AI scorer supporting both exact geometric-mean compensation
    `(prod s_i)^(1/k)` and GDC `(1 - 1/k)` modification-factor compensation."""
    var options: List[UtilityOption]
    var use_geometric_mean: Bool

    def __init__(out self, use_geometric_mean: Bool = True):
        self.options = List[UtilityOption]()
        self.use_geometric_mean = use_geometric_mean

    def add_option(mut self, var opt: UtilityOption) -> Int:
        var idx = len(self.options)
        self.options.append(opt^)
        return idx

    @staticmethod
    def compensate_dave_mark(raw_product: Float32, num_considerations: Int) -> Float32:
        if num_considerations <= 1 or raw_product <= Float32(0.0):
            return max(Float32(0.0), raw_product)
        var mod_factor = Float32(1.0) - Float32(1.0) / Float32(num_considerations)
        var make_up = (Float32(1.0) - raw_product) * mod_factor
        return raw_product + make_up * raw_product

    @staticmethod
    def compensate_geometric_mean(raw_product: Float32, num_considerations: Int) -> Float32:
        if num_considerations <= 1 or raw_product <= Float32(0.0):
            return max(Float32(0.0), raw_product)
        return det_pow(raw_product, Float32(1.0) / Float32(num_considerations))

    def score_option(self, opt_index: Int, bb: Blackboard) -> Float32:
        if opt_index < 0 or opt_index >= len(self.options):
            return Float32(0.0)
        ref opt = self.options[opt_index]
        var k = len(opt.considerations)
        if k == 0:
            return opt.weight
        var prod = Float32(1.0)
        for i in range(k):
            var s = opt.considerations[i].evaluate(bb)
            if s <= Float32(1.0e-7):
                return Float32(0.0)
            prod *= s
        var comp = Self.compensate_geometric_mean(prod, k) if self.use_geometric_mean else Self.compensate_dave_mark(prod, k)
        return comp * opt.weight

    def select_best(self, bb: Blackboard) -> Tuple[Int, Float32]:
        """Returns `(best_option_id, best_score)`."""
        var best_id = -1
        var best_score = Float32(-1.0)
        for i in range(len(self.options)):
            var s = self.score_option(i, bb)
            if s > best_score:
                best_score = s
                best_id = self.options[i].option_id
        return (best_id, max(Float32(0.0), best_score))


# =========================================================================
# 3. Flat DFS Pre-Order BehaviorTree with O(1) subtree_skip & Reactive Aborts
# =========================================================================

@fieldwise_init
struct BtNode(Copyable, ImplicitlyCopyable, Movable):
    var kind: UInt8
    var abort_mode: UInt8
    var cmp_op: UInt8
    var subtree_skip: Int32
    var parent_idx: Int32
    var bb_key: Int32
    var action_id: Int32
    var param_f32: Float32

    def __init__(out self, kind: UInt8 = BT_NODE_ACTION):
        self.kind = kind
        self.abort_mode = BT_ABORT_NONE
        self.cmp_op = BT_CMP_EQ
        self.subtree_skip = 1
        self.parent_idx = -1
        self.bb_key = -1
        self.action_id = -1
        self.param_f32 = 0.0

    @always_inline
    def eval_condition(self, bb: Blackboard) -> Bool:
        var v = bb.get_float(Int(self.bb_key))
        var t = self.param_f32
        if self.cmp_op == BT_CMP_LT: return v < t
        if self.cmp_op == BT_CMP_LE: return v <= t
        if self.cmp_op == BT_CMP_GT: return v > t
        if self.cmp_op == BT_CMP_GE: return v >= t
        if self.cmp_op == BT_CMP_EQ: return abs(v - t) <= Float32(1.0e-4)
        return abs(v - t) > Float32(1.0e-4)


struct BtExecutionState(Copyable, Movable):
    var time_s: Float32
    var running_node: Int
    var node_running: List[Bool]
    var active_action_id: Int
    var last_aborted_node: Int
    var last_aborted_action_id: Int
    var aborts_triggered: Int
    var cooldown_ready_at: List[Float32]
    var action_statuses: List[UInt8]
    var action_exec_counts: List[Int]

    def __init__(out self, max_actions: Int = 32):
        self.time_s = 0.0
        self.running_node = -1
        self.node_running = List[Bool]()
        self.active_action_id = -1
        self.last_aborted_node = -1
        self.last_aborted_action_id = -1
        self.aborts_triggered = 0
        self.cooldown_ready_at = List[Float32]()
        self.action_statuses = List[UInt8](length=max(8, max_actions), fill=BT_SUCCESS)
        self.action_exec_counts = List[Int](length=max(8, max_actions), fill=0)

    def set_action_status(mut self, action_id: Int, status: UInt8):
        if action_id < 0:
            return
        if action_id >= len(self.action_statuses):
            var new_len = action_id + 8
            self.action_statuses.resize(new_len, BT_SUCCESS)
            self.action_exec_counts.resize(new_len, 0)
        self.action_statuses[action_id] = status

    def is_node_running(self, idx: Int) -> Bool:
        if idx >= 0 and idx < len(self.node_running):
            return self.node_running[idx]
        return self.running_node == idx

    def first_running_in_range(self, lo: Int, hi: Int) -> Int:
        var hi_clamped = min(hi, len(self.node_running))
        var lo_clamped = max(0, lo)
        for k in range(lo_clamped, hi_clamped):
            if self.node_running[k]:
                return k
        if self.running_node >= lo and self.running_node < hi:
            return self.running_node
        return -1

    def has_running_in_range(self, lo: Int, hi: Int) -> Bool:
        return self.first_running_in_range(lo, hi) >= 0

    def _sync_running_node(mut self):
        for k in range(len(self.node_running)):
            if self.node_running[k]:
                self.running_node = k
                return
        self.running_node = -1

    def set_node_running(mut self, idx: Int, running: Bool):
        if idx < 0:
            return
        if idx >= len(self.node_running):
            self.node_running.resize(idx + 1, False)
        self.node_running[idx] = running
        if running:
            self.running_node = idx
        elif self.running_node == idx:
            self._sync_running_node()

    def clear_running_in_range(mut self, lo: Int, hi: Int):
        var hi_clamped = min(hi, len(self.node_running))
        var lo_clamped = max(0, lo)
        for k in range(lo_clamped, hi_clamped):
            self.node_running[k] = False
        if self.running_node >= lo and self.running_node < hi:
            self._sync_running_node()


struct BehaviorTree(Copyable, Movable):
    """Flat contiguous DFS pre-order Behavior Tree with O(1) `subtree_skip` jumps,
    UtilitySelect composites, and reactive `BT_ABORT_LOWER_PRIORITY` / `BT_ABORT_SELF`."""
    var nodes: List[BtNode]
    var open_stack: List[Int]
    var utility_scorers: List[UtilityAiScorer]

    def __init__(out self):
        self.nodes = List[BtNode]()
        self.open_stack = List[Int]()
        self.utility_scorers = List[UtilityAiScorer]()

    def _current_parent(self) -> Int32:
        return Int32(self.open_stack[len(self.open_stack) - 1]) if len(self.open_stack) > 0 else Int32(-1)

    def _begin_composite(mut self, kind: UInt8, param: Float32 = 0.0, scorer_idx: Int = -1) -> Int:
        var idx = len(self.nodes)
        var n = BtNode(kind)
        n.parent_idx = self._current_parent()
        n.subtree_skip = Int32(idx + 1)
        n.param_f32 = param
        n.action_id = Int32(scorer_idx)
        self.nodes.append(n)
        self.open_stack.append(idx)
        return idx

    def begin_sequence(mut self) -> Int:
        return self._begin_composite(BT_NODE_SEQUENCE)

    def begin_selector(mut self) -> Int:
        return self._begin_composite(BT_NODE_SELECTOR)

    def begin_parallel(mut self, required_successes: Int = 1) -> Int:
        return self._begin_composite(BT_NODE_PARALLEL, Float32(max(1, required_successes)))

    def begin_inverter(mut self) -> Int:
        return self._begin_composite(BT_NODE_INVERTER)

    def begin_cooldown(mut self, cooldown_seconds: Float32) -> Int:
        return self._begin_composite(BT_NODE_COOLDOWN, max(Float32(0.0), cooldown_seconds))

    def begin_utility_select(mut self, var scorer: UtilityAiScorer) -> Int:
        var s_idx = len(self.utility_scorers)
        self.utility_scorers.append(scorer^)
        return self._begin_composite(BT_NODE_UTILITY_SELECT, Float32(0.0), s_idx)

    def end_composite(mut self):
        """Stamps exact O(1) `subtree_skip` index on the closing composite/decorator."""
        if len(self.open_stack) == 0:
            return
        var idx = self.open_stack.pop()
        self.nodes[idx].subtree_skip = Int32(len(self.nodes))

    def add_blackboard_cmp(
        mut self,
        bb_key: Int,
        cmp_op: UInt8,
        cmp_val: Float32,
        abort_mode: UInt8 = BT_ABORT_NONE,
    ) -> Int:
        var idx = len(self.nodes)
        var n = BtNode(BT_NODE_BLACKBOARD_CMP)
        n.parent_idx = self._current_parent()
        n.subtree_skip = Int32(idx + 1)
        n.bb_key = Int32(bb_key)
        n.cmp_op = cmp_op
        n.param_f32 = cmp_val
        n.abort_mode = abort_mode
        self.nodes.append(n)
        return idx

    def add_action(mut self, action_id: Int) -> Int:
        var idx = len(self.nodes)
        var n = BtNode(BT_NODE_ACTION)
        n.parent_idx = self._current_parent()
        n.subtree_skip = Int32(idx + 1)
        n.action_id = Int32(action_id)
        self.nodes.append(n)
        return idx

    def _find_enclosing_selector_branch(self, guard_idx: Int) -> Tuple[Int, Int]:
        """Finds `(branch_end, selector_end)` for a `BT_ABORT_LOWER_PRIORITY` guard."""
        var cur = guard_idx
        while cur >= 0:
            var p = Int(self.nodes[cur].parent_idx)
            if p < 0:
                break
            if self.nodes[p].kind == BT_NODE_SELECTOR:
                return (Int(self.nodes[cur].subtree_skip), Int(self.nodes[p].subtree_skip))
            cur = p
        return (-1, -1)

    def _check_reactive_aborts(self, bb: Blackboard, mut state: BtExecutionState):
        if state.running_node < 0 or state.running_node >= len(self.nodes):
            return

        for i in range(len(self.nodes)):
            ref nd = self.nodes[i]
            if nd.kind != BT_NODE_BLACKBOARD_CMP or nd.abort_mode == BT_ABORT_NONE:
                continue
            var cond_ok = nd.eval_condition(bb)

            # 1. BT_ABORT_LOWER_PRIORITY / BT_ABORT_BOTH: condition became TRUE while a lower-priority sibling branch is running!
            if (nd.abort_mode == BT_ABORT_LOWER_PRIORITY or nd.abort_mode == BT_ABORT_BOTH) and cond_ok:
                var scope = self._find_enclosing_selector_branch(i)
                if scope[0] >= 0:
                    var r_node = state.first_running_in_range(scope[0], scope[1])
                    if r_node >= 0:
                        state.aborts_triggered += 1
                        state.last_aborted_node = r_node
                        state.last_aborted_action_id = Int(self.nodes[r_node].action_id)
                        state.clear_running_in_range(scope[0], scope[1])
                        if state.running_node < 0:
                            state.active_action_id = -1
                        return

            # 2. BT_ABORT_SELF / BT_ABORT_BOTH: condition became FALSE while its own guarded Sequence is running!
            if (nd.abort_mode == BT_ABORT_SELF or nd.abort_mode == BT_ABORT_BOTH) and (not cond_ok):
                var p = Int(nd.parent_idx)
                if p >= 0:
                    var r_node = state.first_running_in_range(i + 1, Int(self.nodes[p].subtree_skip))
                    if r_node >= 0:
                        state.aborts_triggered += 1
                        state.last_aborted_node = r_node
                        state.last_aborted_action_id = Int(self.nodes[r_node].action_id)
                        state.clear_running_in_range(i + 1, Int(self.nodes[p].subtree_skip))
                        if state.running_node < 0:
                            state.active_action_id = -1
                        return

    def _eval_node(self, idx: Int, bb: Blackboard, mut state: BtExecutionState) -> UInt8:
        ref nd = self.nodes[idx]
        var end_sub = Int(nd.subtree_skip)

        if nd.kind == BT_NODE_BLACKBOARD_CMP:
            return BT_SUCCESS if nd.eval_condition(bb) else BT_FAILURE

        if nd.kind == BT_NODE_ACTION:
            var aid = Int(nd.action_id)
            var st = state.action_statuses[aid] if (aid >= 0 and aid < len(state.action_statuses)) else BT_SUCCESS
            if aid >= 0 and aid < len(state.action_exec_counts):
                state.action_exec_counts[aid] += 1
            if st == BT_RUNNING:
                state.set_node_running(idx, True)
                state.active_action_id = aid
            else:
                state.set_node_running(idx, False)
                state.active_action_id = aid
            return st

        if nd.kind == BT_NODE_INVERTER:
            if idx + 1 >= end_sub:
                return BT_FAILURE
            var cs = self._eval_node(idx + 1, bb, state)
            if cs == BT_SUCCESS: return BT_FAILURE
            if cs == BT_FAILURE: return BT_SUCCESS
            return BT_RUNNING

        if nd.kind == BT_NODE_COOLDOWN:
            if idx + 1 >= end_sub:
                return BT_FAILURE
            if state.time_s < state.cooldown_ready_at[idx]:
                return BT_FAILURE
            var cs = self._eval_node(idx + 1, bb, state)
            if cs == BT_SUCCESS:
                state.cooldown_ready_at[idx] = state.time_s + nd.param_f32
            return cs

        if nd.kind == BT_NODE_SEQUENCE:
            var child = idx + 1
            while child < end_sub:
                var next_sib = Int(self.nodes[child].subtree_skip)
                # Memory resume: if a later child in this Sequence is already running, skip completed earlier children
                if state.has_running_in_range(next_sib, end_sub):
                    child = next_sib
                    continue
                var cs = self._eval_node(child, bb, state)
                if cs != BT_SUCCESS:
                    return cs
                child = next_sib
            return BT_SUCCESS

        if nd.kind == BT_NODE_SELECTOR:
            var child = idx + 1
            while child < end_sub:
                var next_sib = Int(self.nodes[child].subtree_skip)
                # Memory selector: if a lower-priority child is currently BT_RUNNING (and was NOT aborted by _check_reactive_aborts!),
                # jump directly to the running child in O(1) using subtree_skip!
                if state.has_running_in_range(next_sib, end_sub):
                    child = next_sib
                    continue
                var cs = self._eval_node(child, bb, state)
                if cs != BT_FAILURE:
                    return cs
                child = next_sib
            return BT_FAILURE

        if nd.kind == BT_NODE_PARALLEL:
            var req_succ = max(1, Int(nd.param_f32))
            var succ_cnt = 0
            var run_cnt = 0
            var child = idx + 1
            while child < end_sub:
                var cs = self._eval_node(child, bb, state)
                if cs == BT_SUCCESS:
                    succ_cnt += 1
                elif cs == BT_RUNNING:
                    run_cnt += 1
                child = Int(self.nodes[child].subtree_skip)
            if succ_cnt >= req_succ:
                state.clear_running_in_range(idx + 1, end_sub)
                return BT_SUCCESS
            if run_cnt > 0:
                return BT_RUNNING
            state.clear_running_in_range(idx + 1, end_sub)
            return BT_FAILURE

        if nd.kind == BT_NODE_UTILITY_SELECT:
            var s_idx = Int(nd.action_id)
            var child_starts = List[Int]()
            var child_scores = List[Float32]()
            var child = idx + 1
            var k_opt = 0
            while child < end_sub:
                child_starts.append(child)
                var sc = Float32(0.0)
                if s_idx >= 0 and s_idx < len(self.utility_scorers):
                    sc = self.utility_scorers[s_idx].score_option(k_opt, bb)
                child_scores.append(sc)
                child = Int(self.nodes[child].subtree_skip)
                k_opt += 1

            # Sort children descending by compensated utility score
            var order = List[Int](capacity=len(child_starts))
            for i in range(len(child_starts)):
                order.append(i)
            for i in range(1, len(order)):
                var key = order[i]
                var j = i - 1
                while j >= 0 and child_scores[order[j]] < child_scores[key]:
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = key

            for i in range(len(order)):
                var ci = order[i]
                if child_scores[ci] <= Float32(0.0):
                    break
                var cs = self._eval_node(child_starts[ci], bb, state)
                if cs != BT_FAILURE:
                    return cs
            return BT_FAILURE

        return BT_FAILURE

    def tick(self, bb: Blackboard, mut state: BtExecutionState, dt: Float32) -> UInt8:
        if len(self.nodes) == 0:
            return BT_FAILURE
        state.time_s += max(Float32(0.0), dt)
        if len(state.cooldown_ready_at) < len(self.nodes):
            state.cooldown_ready_at.resize(len(self.nodes), Float32(0.0))
        if len(state.node_running) < len(self.nodes):
            state.node_running.resize(len(self.nodes), False)

        # 1. Reactive observer pass (`BT_ABORT_LOWER_PRIORITY` / `BT_ABORT_SELF`)
        self._check_reactive_aborts(bb, state)

        # 2. Flat DFS pre-order evaluation with O(1) `subtree_skip` jumps
        return self._eval_node(0, bb, state)


# =========================================================================
# 4. SIMD8 SpatialPerception (Vision Cones + PhysicsWorld LOS + Hearing)
# =========================================================================

@fieldwise_init
struct PerceivedStimulus(Copyable, ImplicitlyCopyable, Movable):
    var stimulus_index: Int
    var body_id: Int
    var team_id: Int
    var visible: Bool
    var audible: Bool
    var distance: Float32
    var direction: Vec3
    var vision_score: Float32
    var sound_intensity: Float32


struct SpatialPerception(Copyable, Movable):
    """SoA 8-wide SIMD sensory perception system evaluating 8 stimuli per `F8` vector
    for vision-cone frustum & inverse-square acoustic attenuation, gated by `PhysicsWorld.raycast` LOS."""
    var count: Int
    var padded: Int
    var sx: List[Float32]
    var sy: List[Float32]
    var sz: List[Float32]
    var loudness: List[Float32]
    var body_id: List[Int]
    var team_id: List[Int]
    var active: List[Bool]

    def __init__(out self):
        self.count = 0
        self.padded = 0
        self.sx = List[Float32]()
        self.sy = List[Float32]()
        self.sz = List[Float32]()
        self.loudness = List[Float32]()
        self.body_id = List[Int]()
        self.team_id = List[Int]()
        self.active = List[Bool]()

    def clear(mut self):
        self.count = 0
        self.padded = 0
        self.sx.clear()
        self.sy.clear()
        self.sz.clear()
        self.loudness.clear()
        self.body_id.clear()
        self.team_id.clear()
        self.active.clear()

    def _pad_columns(mut self):
        var req = ((self.count + LANES - 1) // LANES) * LANES
        while len(self.sx) < req:
            self.sx.append(Float32(1.0e6))
            self.sy.append(Float32(1.0e6))
            self.sz.append(Float32(1.0e6))
            self.loudness.append(Float32(0.0))
            self.body_id.append(-1)
            self.team_id.append(-1)
            self.active.append(False)
        self.padded = req

    def add_stimulus(
        mut self,
        pos: Vec3,
        loudness: Float32 = 0.0,
        body_id: Int = -1,
        team_id: Int = 0,
    ) -> Int:
        var idx = self.count
        self.count += 1
        self._pad_columns()
        self.sx[idx] = pos.x
        self.sy[idx] = pos.y
        self.sz[idx] = pos.z
        self.loudness[idx] = max(Float32(0.0), loudness)
        self.body_id[idx] = body_id
        self.team_id[idx] = team_id
        self.active[idx] = True
        return idx

    def update_stimulus_pos(mut self, idx: Int, pos: Vec3, loudness: Float32 = -1.0):
        if idx < 0 or idx >= self.count:
            return
        self.sx[idx] = pos.x
        self.sy[idx] = pos.y
        self.sz[idx] = pos.z
        if loudness >= Float32(0.0):
            self.loudness[idx] = loudness

    def query_perceiver(
        self,
        world: PhysicsWorld,
        eye_pos: Vec3,
        forward: Vec3,
        fov_deg: Float32 = 110.0,
        max_vision_dist: Float32 = 35.0,
        hearing_threshold: Float32 = 0.05,
        occlude_filter: CollisionFilter = CollisionFilter(CAT_ALL, CAT_ALL, 0),
        ignore_body_id: Int = -1,
        ignore_team_id: Int = -999999,
    ) -> List[PerceivedStimulus]:
        var out = List[PerceivedStimulus]()
        if self.count == 0:
            return out^

        var fwd = forward.normalized()
        if fwd.length_squared() < Float32(0.5):
            fwd = Vec3(0.0, 0.0, 1.0)
        var cos_half_fov = det_cos(fov_deg * Float32(0.5) * _AI_DEG_TO_RAD)

        var ex = F8(eye_pos.x)
        var ey = F8(eye_pos.y)
        var ez = F8(eye_pos.z)
        var fx = F8(fwd.x)
        var fy = F8(fwd.y)
        var fz = F8(fwd.z)
        var max_v_f8 = F8(max_vision_dist)
        var cos_f8 = F8(cos_half_fov)
        var hear_f8 = F8(hearing_threshold)

        var sx_ptr = self.sx.unsafe_ptr()
        var sy_ptr = self.sy.unsafe_ptr()
        var sz_ptr = self.sz.unsafe_ptr()
        var loud_ptr = self.loudness.unsafe_ptr()

        var n_batches = self.padded // LANES
        for b in range(n_batches):
            var at = b * LANES
            var dx = sx_ptr.unsafe_load[width=LANES](at) - ex
            var dy = sy_ptr.unsafe_load[width=LANES](at) - ey
            var dz = sz_ptr.unsafe_load[width=LANES](at) - ez
            var d2 = dx * dx + dy * dy + dz * dz
            var dist = sqrt(max(d2, F8(1.0e-12)))
            var inv_d = F8(1.0) / dist
            var cos_ang = (dx * fx + dy * fy + dz * fz) * inv_d

            var in_cone = dist.le(max_v_f8) & cos_ang.ge(cos_f8)
            var sound_int = loud_ptr.unsafe_load[width=LANES](at) / max(F8(1.0), d2)
            var is_audible = sound_int.ge(hear_f8)
            var candidate = in_cone | is_audible
            if not candidate.reduce_or():
                continue

            comptime for lane in range(LANES):
                var idx = at + lane
                if idx < self.count and self.active[idx] and candidate[lane]:
                    var t_id = self.team_id[idx]
                    var b_id = self.body_id[idx]
                    if b_id != ignore_body_id and t_id != ignore_team_id:
                        var d_s = dist[lane]
                        var dir_s = Vec3(dx[lane], dy[lane], dz[lane]) * inv_d[lane] if d_s > Float32(1.0e-5) else fwd
                        var vis = False
                        var vis_score = Float32(0.0)

                        if in_cone[lane]:
                            if d_s <= Float32(1.0e-4):
                                vis = True
                                vis_score = Float32(1.0)
                            else:
                                var hit = world.raycast(eye_pos, dir_s, d_s, occlude_filter, ignore_body_id)
                                if (not hit.hit) or hit.body_id == b_id or hit.distance >= d_s - Float32(0.05):
                                    vis = True
                                    var dist_falloff = Float32(1.0) - d_s / max(Float32(0.01), max_vision_dist)
                                    vis_score = max(Float32(0.0), dist_falloff * max(Float32(0.0), cos_ang[lane]))

                        var aud = is_audible[lane]
                        if vis or aud:
                            out.append(PerceivedStimulus(
                                idx, b_id, t_id, vis, aud, d_s, dir_s, vis_score, sound_int[lane],
                            ))

        return out^
