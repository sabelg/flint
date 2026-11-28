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
from std.math import sqrt, abs, min, max, isfinite
from flint.app.engine_app import EngineApp, EnginePlugin, FrameTime, Resources, STAGE_UPDATE
from flint.assets.asset_ref import AssetRef
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.detmath import det_cos, det_exp, det_pow
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import ParallelCommands, STORAGE_TABLE, World, comp_hash64
from flint.physics.filter import CollisionFilter, CAT_ALL
from flint.physics.solver_bodies import F8, LANES
from flint.physics.world import PhysicsWorld
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import SerializableComponent

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
comptime BT_NODE_REPEAT: UInt8 = 8  # decorator: re-runs child `param_f32` times (0 = forever), one iteration per tick
comptime BT_NODE_TIMEOUT: UInt8 = 9  # decorator: fails a child that stays RUNNING longer than `param_f32` s
comptime BT_NODE_SUCCEEDER: UInt8 = 10  # decorator: SUCCESS unless child is RUNNING
comptime BT_NODE_FORCE_FAILURE: UInt8 = 11  # decorator: FAILURE unless child is RUNNING
comptime BT_NODE_WAIT: UInt8 = 12  # leaf: RUNNING for `param_f32` seconds, then SUCCESS
comptime BT_NODE_SUBTREE: UInt8 = 13  # pass-through root of an inlined subtree (`action_id` = subtree slot)
comptime BT_NODE_SERVICE: UInt8 = 14  # decorator: schedules service `action_id` every `param_f32` s while active
comptime BT_NODE_LAST: UInt8 = 14

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
comptime BT_CMP_IS_TRUE: UInt8 = 6  # Bool key
comptime BT_CMP_IS_FALSE: UInt8 = 7  # Bool key
comptime BT_CMP_ENTITY_SET: UInt8 = 8  # entity key != 0
comptime BT_CMP_ENTITY_UNSET: UInt8 = 9  # entity key == 0
comptime BT_CMP_LAST: UInt8 = 9

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
    var entities: List[UInt64]  # entity bits (`Entity.to_bits()`), 0 = none
    var bools: List[Bool]

    def __init__(out self):
        self.names = List[String]()
        self.floats = List[Float32]()
        self.ints = List[Int]()
        self.vecs = List[Vec3]()
        self.entities = List[UInt64]()
        self.bools = List[Bool]()

    def def_key(mut self, name: String, initial_f32: Float32 = 0.0, initial_int: Int = 0, initial_vec: Vec3 = Vec3()) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        var idx = len(self.names)
        self.names.append(name)
        self.floats.append(initial_f32)
        self.ints.append(initial_int)
        self.vecs.append(initial_vec)
        self._sync_ext()
        return idx

    def def_bool_key(mut self, name: String, initial: Bool = False) -> Int:
        var existed = self.find_key(name) >= 0
        var k = self.def_key(name)
        if not existed:
            self.set_bool(k, initial)
        return k

    def def_entity_key(mut self, name: String, initial: UInt64 = 0) -> Int:
        var existed = self.find_key(name) >= 0
        var k = self.def_key(name)
        if not existed:
            self.set_entity(k, initial)
        return k

    def _sync_ext(mut self):
        """Keeps `entities` / `bools` as long as `names` (codecs restore only the original columns)."""
        var n = len(self.names)
        if len(self.entities) < n:
            self.entities.resize(n, UInt64(0))
        if len(self.bools) < n:
            self.bools.resize(n, False)

    def set_entity(mut self, key: Int, val: UInt64):
        if key >= 0 and key < len(self.names):
            self._sync_ext()
            self.entities[key] = val

    def get_entity(self, key: Int) -> UInt64:
        if key >= 0 and key < len(self.entities):
            return self.entities[key]
        return UInt64(0)

    def set_bool(mut self, key: Int, val: Bool):
        if key >= 0 and key < len(self.names):
            self._sync_ext()
            self.bools[key] = val

    def get_bool(self, key: Int) -> Bool:
        if key >= 0 and key < len(self.bools):
            return self.bools[key]
        return False

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
        if not isfinite(raw):
            return Float32(0.0)
        if not (isfinite(self.input_min) and isfinite(self.input_max) and isfinite(self.exponent)):
            return Float32(0.0)
        var denom = self.input_max - self.input_min
        if not isfinite(denom) or abs(denom) <= Float32(1.0e-7):
            return Float32(0.0)
        var t_norm = (raw - self.input_min) / denom
        if not isfinite(t_norm):
            return Float32(0.0)
        var x = max(Float32(0.0), min(Float32(1.0), t_norm))
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
        if not isfinite(y):
            return Float32(0.0)
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
        if not isfinite(raw_product) or raw_product <= Float32(0.0):
            return Float32(0.0)
        if num_considerations <= 1:
            return raw_product
        var mod_factor = Float32(1.0) - Float32(1.0) / Float32(num_considerations)
        var make_up = (Float32(1.0) - raw_product) * mod_factor
        var res = raw_product + make_up * raw_product
        return res if isfinite(res) else Float32(0.0)

    @staticmethod
    def compensate_geometric_mean(raw_product: Float32, num_considerations: Int) -> Float32:
        if not isfinite(raw_product) or raw_product <= Float32(0.0):
            return Float32(0.0)
        if num_considerations <= 1:
            return raw_product
        var res = det_pow(raw_product, Float32(1.0) / Float32(num_considerations))
        return res if isfinite(res) else Float32(0.0)

    def score_option(self, opt_index: Int, bb: Blackboard) -> Float32:
        if opt_index < 0 or opt_index >= len(self.options):
            return Float32(0.0)
        ref opt = self.options[opt_index]
        if not isfinite(opt.weight) or opt.weight <= Float32(0.0):
            return Float32(0.0)
        var k = len(opt.considerations)
        if k == 0:
            return opt.weight
        var prod = Float32(1.0)
        for i in range(k):
            var s = opt.considerations[i].evaluate(bb)
            if not isfinite(s) or s <= Float32(1.0e-7):
                return Float32(0.0)
            prod *= s
        var comp = Self.compensate_geometric_mean(prod, k) if self.use_geometric_mean else Self.compensate_dave_mark(prod, k)
        var total = comp * opt.weight
        return total if isfinite(total) else Float32(0.0)

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
        if self.cmp_op == BT_CMP_IS_TRUE: return bb.get_bool(Int(self.bb_key))
        if self.cmp_op == BT_CMP_IS_FALSE: return not bb.get_bool(Int(self.bb_key))
        if self.cmp_op == BT_CMP_ENTITY_SET: return bb.get_entity(Int(self.bb_key)) != 0
        if self.cmp_op == BT_CMP_ENTITY_UNSET: return bb.get_entity(Int(self.bb_key)) == 0
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
    var node_timer: List[Float32]  # per-node start time (Wait / Timeout), < 0 = inactive
    var node_counter: List[Int]  # per-node iteration counter (Repeat)
    var service_next: List[Float32]  # per-node next service time
    var pending_services: List[Int]  # service ids due this tick (drained by game / task library)

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
        self.node_timer = List[Float32]()
        self.node_counter = List[Int]()
        self.service_next = List[Float32]()
        self.pending_services = List[Int]()

    def _ensure_nodes(mut self, n: Int):
        if len(self.node_timer) < n:
            self.node_timer.resize(n, Float32(-1.0))
        if len(self.node_counter) < n:
            self.node_counter.resize(n, 0)
        if len(self.service_next) < n:
            self.service_next.resize(n, Float32(0.0))

    def take_services(mut self) -> List[Int]:
        var out = self.pending_services^
        self.pending_services = List[Int]()
        return out^

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
    var subtree_names: List[String]

    def __init__(out self):
        self.nodes = List[BtNode]()
        self.open_stack = List[Int]()
        self.utility_scorers = List[UtilityAiScorer]()
        self.subtree_names = List[String]()

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

    def begin_repeat(mut self, count: Int = 0) -> Int:
        """Repeats its single child `count` times (0 = forever); fails as soon as the child fails."""
        return self._begin_composite(BT_NODE_REPEAT, Float32(max(0, count)))

    def begin_timeout(mut self, seconds: Float32) -> Int:
        return self._begin_composite(BT_NODE_TIMEOUT, max(Float32(0.0), seconds))

    def begin_succeeder(mut self) -> Int:
        return self._begin_composite(BT_NODE_SUCCEEDER)

    def begin_force_failure(mut self) -> Int:
        return self._begin_composite(BT_NODE_FORCE_FAILURE)

    def begin_service(mut self, service_id: Int, interval_seconds: Float32) -> Int:
        """Decorator: while its child is being evaluated, pushes `service_id` onto
        `BtExecutionState.pending_services` every `interval_seconds` (first time on entry)."""
        var idx = self._begin_composite(BT_NODE_SERVICE, max(Float32(0.0), interval_seconds))
        self.nodes[idx].action_id = Int32(service_id)
        return idx

    def add_wait(mut self, seconds: Float32) -> Int:
        var idx = len(self.nodes)
        var n = BtNode(BT_NODE_WAIT)
        n.parent_idx = self._current_parent()
        n.subtree_skip = Int32(idx + 1)
        n.param_f32 = max(Float32(0.0), seconds)
        self.nodes.append(n)
        return idx

    def add_blackboard_bool(mut self, bb_key: Int, expected: Bool = True, abort_mode: UInt8 = BT_ABORT_NONE) -> Int:
        return self.add_blackboard_cmp(bb_key, BT_CMP_IS_TRUE if expected else BT_CMP_IS_FALSE, 0.0, abort_mode)

    def add_blackboard_entity_set(mut self, bb_key: Int, is_set: Bool = True, abort_mode: UInt8 = BT_ABORT_NONE) -> Int:
        return self.add_blackboard_cmp(bb_key, BT_CMP_ENTITY_SET if is_set else BT_CMP_ENTITY_UNSET, 0.0, abort_mode)

    def add_subtree(mut self, sub: BehaviorTree, name: String = "") -> Int:
        """Inlines `sub` under a `BT_NODE_SUBTREE` node (reusable subtree reference resolved at build
        time: indices, parents, skips and utility scorers are rebased). Returns the subtree node index."""
        var slot = len(self.subtree_names)
        self.subtree_names.append(name)
        var root = self._begin_composite(BT_NODE_SUBTREE)
        self.nodes[root].action_id = Int32(slot)
        var base = len(self.nodes)
        var scorer_base = len(self.utility_scorers)
        for i in range(len(sub.utility_scorers)):
            self.utility_scorers.append(sub.utility_scorers[i].copy())
        for i in range(len(sub.nodes)):
            var n = sub.nodes[i]
            n.subtree_skip = Int32(Int(n.subtree_skip) + base)
            n.parent_idx = Int32(root) if n.parent_idx < 0 else Int32(Int(n.parent_idx) + base)
            if n.kind == BT_NODE_UTILITY_SELECT and n.action_id >= 0:
                n.action_id = Int32(Int(n.action_id) + scorer_base)
            self.nodes.append(n)
        self.end_composite()
        return root

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

        if nd.kind == BT_NODE_WAIT:
            if nd.param_f32 <= Float32(0.0):
                return BT_SUCCESS
            if not state.is_node_running(idx) or state.node_timer[idx] < Float32(0.0):
                state.node_timer[idx] = state.time_s
                state.set_node_running(idx, True)
                return BT_RUNNING
            if state.time_s - state.node_timer[idx] >= nd.param_f32 - Float32(1.0e-5):
                state.node_timer[idx] = Float32(-1.0)
                state.set_node_running(idx, False)
                return BT_SUCCESS
            return BT_RUNNING

        if nd.kind == BT_NODE_SUBTREE:
            if idx + 1 >= end_sub:
                return BT_SUCCESS
            return self._eval_node(idx + 1, bb, state)

        if nd.kind == BT_NODE_SUCCEEDER or nd.kind == BT_NODE_FORCE_FAILURE:
            if idx + 1 >= end_sub:
                return BT_SUCCESS if nd.kind == BT_NODE_SUCCEEDER else BT_FAILURE
            var cs = self._eval_node(idx + 1, bb, state)
            if cs == BT_RUNNING:
                return BT_RUNNING
            return BT_SUCCESS if nd.kind == BT_NODE_SUCCEEDER else BT_FAILURE

        if nd.kind == BT_NODE_SERVICE:
            if not state.has_running_in_range(idx + 1, end_sub) or state.time_s >= state.service_next[idx]:
                if Int(nd.action_id) >= 0:
                    state.pending_services.append(Int(nd.action_id))
                state.service_next[idx] = state.time_s + nd.param_f32
            if idx + 1 >= end_sub:
                return BT_SUCCESS
            return self._eval_node(idx + 1, bb, state)

        if nd.kind == BT_NODE_TIMEOUT:
            if idx + 1 >= end_sub:
                return BT_FAILURE
            if not state.has_running_in_range(idx + 1, end_sub) or state.node_timer[idx] < Float32(0.0):
                state.node_timer[idx] = state.time_s
            var cs = self._eval_node(idx + 1, bb, state)
            if cs == BT_RUNNING:
                if state.time_s - state.node_timer[idx] >= nd.param_f32:
                    state.clear_running_in_range(idx + 1, end_sub)
                    state.node_timer[idx] = Float32(-1.0)
                    return BT_FAILURE
                return BT_RUNNING
            state.node_timer[idx] = Float32(-1.0)
            return cs

        if nd.kind == BT_NODE_REPEAT:
            if idx + 1 >= end_sub:
                return BT_FAILURE
            var cs = self._eval_node(idx + 1, bb, state)
            if cs == BT_RUNNING:
                state.set_node_running(idx, True)
                return BT_RUNNING
            if cs == BT_FAILURE:
                state.node_counter[idx] = 0
                state.set_node_running(idx, False)
                return BT_FAILURE
            state.node_counter[idx] += 1
            var limit = Int(nd.param_f32)
            if limit > 0 and state.node_counter[idx] >= limit:
                state.node_counter[idx] = 0
                state.set_node_running(idx, False)
                return BT_SUCCESS
            state.set_node_running(idx, True)
            return BT_RUNNING

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
        var safe_dt = dt if (isfinite(dt) and dt > Float32(0.0)) else Float32(0.0)
        if not isfinite(state.time_s):
            state.time_s = Float32(0.0)
        state.time_s += safe_dt
        if len(state.cooldown_ready_at) < len(self.nodes):
            state.cooldown_ready_at.resize(len(self.nodes), Float32(0.0))
        if len(state.node_running) < len(self.nodes):
            state.node_running.resize(len(self.nodes), False)
        state._ensure_nodes(len(self.nodes))

        # 1. Reactive observer pass (`BT_ABORT_LOWER_PRIORITY` / `BT_ABORT_SELF`)
        self._check_reactive_aborts(bb, state)

        # 2. Flat DFS pre-order evaluation with O(1) `subtree_skip` jumps
        return self._eval_node(0, bb, state)

    def save_json(self) -> String:
        return _bt_to_json(self)

    def load_json(mut self, json_str: String) -> Bool:
        if json_str.byte_length() == 0:
            return False
        var raw = List[UInt8](capacity=json_str.byte_length())
        for b in json_str.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid:
            return False
        var parsed = _bt_parse_json_node(doc, doc.root)
        if not parsed[0]:
            return False
        self.nodes = parsed[1].copy()
        self.open_stack.clear()
        self.utility_scorers = parsed[2].copy()
        return True


@always_inline
def _bt_f32_s(v: Float32) -> String:
    return String(Float64(v)) if isfinite(v) else String("0.0")


def _bt_vec3_json(v: Vec3) -> String:
    var x = _bt_f32_s(v.x)
    var y = _bt_f32_s(v.y)
    var z = _bt_f32_s(v.z)
    return "[" + x + ", " + y + ", " + z + "]"


def _bt_to_json(tree: BehaviorTree) -> String:
    var out = String('{"version": 1, "nodes": [')
    for i in range(len(tree.nodes)):
        if i > 0:
            out += ", "
        ref nd = tree.nodes[i]
        out += (
            '{"kind": ' + String(Int(nd.kind))
            + ', "abort_mode": ' + String(Int(nd.abort_mode))
            + ', "cmp_op": ' + String(Int(nd.cmp_op))
            + ', "subtree_skip": ' + String(Int(nd.subtree_skip))
            + ', "parent_idx": ' + String(Int(nd.parent_idx))
            + ', "bb_key": ' + String(Int(nd.bb_key))
            + ', "action_id": ' + String(Int(nd.action_id))
            + ', "param_f32": ' + _bt_f32_s(nd.param_f32)
            + "}"
        )
    out += '], "utility_scorers": ['
    for si in range(len(tree.utility_scorers)):
        if si > 0:
            out += ", "
        ref sc = tree.utility_scorers[si]
        out += '{"use_geometric_mean": ' + ("true" if sc.use_geometric_mean else "false") + ', "options": ['
        for oi in range(len(sc.options)):
            if oi > 0:
                out += ", "
            ref opt = sc.options[oi]
            out += (
                '{"option_id": ' + String(opt.option_id)
                + ', "weight": ' + _bt_f32_s(opt.weight)
                + ', "considerations": ['
            )
            for ci in range(len(opt.considerations)):
                if ci > 0:
                    out += ", "
                ref c = opt.considerations[ci]
                out += (
                    '{"bb_key": ' + String(c.bb_key)
                    + ', "input_min": ' + _bt_f32_s(c.input_min)
                    + ', "input_max": ' + _bt_f32_s(c.input_max)
                    + ', "curve_kind": ' + String(Int(c.curve_kind))
                    + ', "exponent": ' + _bt_f32_s(c.exponent)
                    + ', "invert": ' + ("true" if c.invert else "false")
                    + "}"
                )
            out += "]}"
        out += "]}"
    out += "]}"
    return out^


def _bt_parse_json_node(
    doc: JsonDocument, node: Int
) -> Tuple[Bool, List[BtNode], List[UtilityAiScorer]]:
    var empty_nodes = List[BtNode]()
    var empty_scorers = List[UtilityAiScorer]()
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 5:
        return (False, empty_nodes^, empty_scorers^)

    var nodes_arr = doc.object_get(node, "nodes")
    if nodes_arr < 0 or nodes_arr >= len(doc.nodes) or doc.nodes[nodes_arr].kind != 4:
        return (False, empty_nodes^, empty_scorers^)

    var new_scorers = List[UtilityAiScorer]()
    var scorers_arr = doc.object_get(node, "utility_scorers")
    if scorers_arr >= 0:
        if scorers_arr >= len(doc.nodes) or doc.nodes[scorers_arr].kind != 4:
            return (False, empty_nodes^, empty_scorers^)
        var n_sc = doc.array_size(scorers_arr)
        if n_sc > 4096:
            return (False, empty_nodes^, empty_scorers^)
        for si in range(n_sc):
            var sc_n = doc.at(scorers_arr, si)
            if sc_n < 0 or doc.nodes[sc_n].kind != 5:
                return (False, empty_nodes^, empty_scorers^)
            var use_geo = doc.boolean(doc.object_get(sc_n, "use_geometric_mean"), True)
            var scorer = UtilityAiScorer(use_geo)
            var opts_n = doc.object_get(sc_n, "options")
            if opts_n >= 0:
                if doc.nodes[opts_n].kind != 4:
                    return (False, empty_nodes^, empty_scorers^)
                var n_op = doc.array_size(opts_n)
                if n_op > 4096:
                    return (False, empty_nodes^, empty_scorers^)
                for oi in range(n_op):
                    var op_n = doc.at(opts_n, oi)
                    if op_n < 0 or doc.nodes[op_n].kind != 5:
                        return (False, empty_nodes^, empty_scorers^)
                    var oid = doc.integer(doc.object_get(op_n, "option_id"), 0)
                    var w_d = doc.number(doc.object_get(op_n, "weight"), 1.0)
                    if not isfinite(w_d) or not isfinite(Float32(w_d)):
                        return (False, empty_nodes^, empty_scorers^)
                    var opt = UtilityOption(oid, Float32(w_d))
                    var cons_n = doc.object_get(op_n, "considerations")
                    if cons_n >= 0:
                        if doc.nodes[cons_n].kind != 4:
                            return (False, empty_nodes^, empty_scorers^)
                        var n_con = doc.array_size(cons_n)
                        if n_con > 4096:
                            return (False, empty_nodes^, empty_scorers^)
                        for ci in range(n_con):
                            var cn = doc.at(cons_n, ci)
                            if cn < 0 or doc.nodes[cn].kind != 5:
                                return (False, empty_nodes^, empty_scorers^)
                            var c_bb = doc.integer(doc.object_get(cn, "bb_key"), -1)
                            var c_min = doc.number(doc.object_get(cn, "input_min"), 0.0)
                            var c_max = doc.number(doc.object_get(cn, "input_max"), 1.0)
                            var c_kind = doc.integer(doc.object_get(cn, "curve_kind"), Int(CURVE_LINEAR))
                            var c_exp = doc.number(doc.object_get(cn, "exponent"), 1.0)
                            var c_inv = doc.boolean(doc.object_get(cn, "invert"), False)
                            if (
                                c_kind < 0
                                or c_kind > Int(CURVE_INVERSE)
                                or not isfinite(c_min)
                                or not isfinite(Float32(c_min))
                                or not isfinite(c_max)
                                or not isfinite(Float32(c_max))
                                or not isfinite(c_exp)
                                or not isfinite(Float32(c_exp))
                            ):
                                return (False, empty_nodes^, empty_scorers^)
                            opt.add_consideration(
                                UtilityConsideration(
                                    c_bb,
                                    Float32(c_min),
                                    Float32(c_max),
                                    UInt8(c_kind),
                                    Float32(c_exp),
                                    c_inv,
                                )
                            )
                    _ = scorer.add_option(opt^)
            new_scorers.append(scorer^)

    var n_nodes = doc.array_size(nodes_arr)
    if n_nodes > 65536:
        return (False, empty_nodes^, empty_scorers^)
    var new_nodes = List[BtNode](capacity=n_nodes)
    for i in range(n_nodes):
        var nd_n = doc.at(nodes_arr, i)
        if nd_n < 0 or doc.nodes[nd_n].kind != 5:
            return (False, empty_nodes^, empty_scorers^)
        var k_n = doc.object_get(nd_n, "kind")
        if k_n < 0 or doc.nodes[k_n].kind != 2:
            return (False, empty_nodes^, empty_scorers^)
        var kind = doc.integer(k_n, -1)
        if kind < 0 or kind > Int(BT_NODE_LAST):
            return (False, empty_nodes^, empty_scorers^)
        var abort_mode = doc.integer(doc.object_get(nd_n, "abort_mode"), Int(BT_ABORT_NONE))
        if abort_mode < 0 or abort_mode > Int(BT_ABORT_BOTH):
            return (False, empty_nodes^, empty_scorers^)
        var cmp_op = doc.integer(doc.object_get(nd_n, "cmp_op"), Int(BT_CMP_EQ))
        if cmp_op < 0 or cmp_op > Int(BT_CMP_LAST):
            return (False, empty_nodes^, empty_scorers^)
        var subtree_skip = doc.integer(doc.object_get(nd_n, "subtree_skip"), i + 1)
        if subtree_skip <= i or subtree_skip > n_nodes:
            return (False, empty_nodes^, empty_scorers^)
        var parent_idx = doc.integer(doc.object_get(nd_n, "parent_idx"), -1)
        if parent_idx < -1 or parent_idx >= i:
            return (False, empty_nodes^, empty_scorers^)
        var bb_key = doc.integer(doc.object_get(nd_n, "bb_key"), -1)
        var action_id = doc.integer(doc.object_get(nd_n, "action_id"), -1)
        var pf_d = doc.number(doc.object_get(nd_n, "param_f32"), 0.0)
        if not isfinite(pf_d) or not isfinite(Float32(pf_d)):
            return (False, empty_nodes^, empty_scorers^)
        if kind == Int(BT_NODE_UTILITY_SELECT) and action_id >= len(new_scorers):
            return (False, empty_nodes^, empty_scorers^)
        new_nodes.append(
            BtNode(
                UInt8(kind),
                UInt8(abort_mode),
                UInt8(cmp_op),
                Int32(subtree_skip),
                Int32(parent_idx),
                Int32(bb_key),
                Int32(action_id),
                Float32(pf_d),
            )
        )
    return (True, new_nodes^, new_scorers^)


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


# =========================================================================
# 5. BehaviorTreeComponent, Codecs & AiPlugin
# =========================================================================

struct BehaviorTreeComponent(SerializableComponent):
    """ECS component holding a `BehaviorTree`, `Blackboard`, `BtExecutionState`, and tick cadence."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.ai.BehaviorTreeComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "BehaviorTreeComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("AIBT")
    comptime SCHEMA_VERSION: Int = 1

    var tree: BehaviorTree
    var blackboard: Blackboard
    var state: BtExecutionState
    var tree_ref: AssetRef
    var enabled: Bool
    var tick_interval_sec: Float32
    var accum_sec: Float32

    def __init__(out self):
        self.tree = BehaviorTree()
        self.blackboard = Blackboard()
        self.state = BtExecutionState()
        self.tree_ref = AssetRef()
        self.enabled = True
        self.tick_interval_sec = 0.0
        self.accum_sec = 0.0

    def __init__(
        out self,
        var tree: BehaviorTree,
        enabled: Bool = True,
        tick_interval_sec: Float32 = 0.0,
    ):
        self.tree = tree^
        self.blackboard = Blackboard()
        self.state = BtExecutionState()
        self.tree_ref = AssetRef()
        self.enabled = enabled
        self.tick_interval_sec = tick_interval_sec if isfinite(tick_interval_sec) else Float32(0.0)
        self.accum_sec = 0.0

    def __init__(out self, *, copy: Self):
        self.tree = copy.tree.copy()
        self.blackboard = copy.blackboard.copy()
        self.state = copy.state.copy()
        self.tree_ref = copy.tree_ref.copy()
        self.enabled = copy.enabled
        self.tick_interval_sec = copy.tick_interval_sec
        self.accum_sec = copy.accum_sec

    def write_bin(self, mut out: BlobWriter):
        # 1. Tree nodes & utility scorers
        out.u32(UInt32(len(self.tree.nodes)))
        for i in range(len(self.tree.nodes)):
            ref nd = self.tree.nodes[i]
            out.u8(nd.kind)
            out.u8(nd.abort_mode)
            out.u8(nd.cmp_op)
            out.integer(Int(nd.subtree_skip))
            out.integer(Int(nd.parent_idx))
            out.integer(Int(nd.bb_key))
            out.integer(Int(nd.action_id))
            out.f32(nd.param_f32)
        out.u32(UInt32(len(self.tree.utility_scorers)))
        for si in range(len(self.tree.utility_scorers)):
            ref sc = self.tree.utility_scorers[si]
            out.boolean(sc.use_geometric_mean)
            out.u32(UInt32(len(sc.options)))
            for oi in range(len(sc.options)):
                ref opt = sc.options[oi]
                out.integer(opt.option_id)
                out.f32(opt.weight)
                out.u32(UInt32(len(opt.considerations)))
                for ci in range(len(opt.considerations)):
                    ref c = opt.considerations[ci]
                    out.integer(c.bb_key)
                    out.f32(c.input_min)
                    out.f32(c.input_max)
                    out.u8(c.curve_kind)
                    out.f32(c.exponent)
                    out.boolean(c.invert)

        # 2. Blackboard
        out.u32(UInt32(len(self.blackboard.names)))
        for i in range(len(self.blackboard.names)):
            out.text(self.blackboard.names[i])
            out.f32(self.blackboard.floats[i])
            out.integer(self.blackboard.ints[i])
            out.vec3(self.blackboard.vecs[i])

        # 3. BtExecutionState
        out.f32(self.state.time_s)
        out.integer(self.state.running_node)
        out.u32(UInt32(len(self.state.node_running)))
        for i in range(len(self.state.node_running)):
            out.boolean(self.state.node_running[i])
        out.integer(self.state.active_action_id)
        out.integer(self.state.last_aborted_node)
        out.integer(self.state.last_aborted_action_id)
        out.integer(self.state.aborts_triggered)
        out.u32(UInt32(len(self.state.cooldown_ready_at)))
        for i in range(len(self.state.cooldown_ready_at)):
            out.f32(self.state.cooldown_ready_at[i])
        out.u32(UInt32(len(self.state.action_statuses)))
        for i in range(len(self.state.action_statuses)):
            out.u8(self.state.action_statuses[i])
        out.u32(UInt32(len(self.state.action_exec_counts)))
        for i in range(len(self.state.action_exec_counts)):
            out.integer(self.state.action_exec_counts[i])

        # 4. AssetRef & component settings
        self.tree_ref.write_bin(out)
        out.boolean(self.enabled)
        out.f32(self.tick_interval_sec)
        out.f32(self.accum_sec)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var out = Self()
        var n_nodes = Int(reader.u32())
        if reader.failed or n_nodes < 0 or n_nodes > 65536 or n_nodes > max(0, reader.limit() - reader.offset) // 16:
            raise Error("BehaviorTreeComponent: invalid node count")
        for i in range(n_nodes):
            var kind = reader.u8()
            var abort_mode = reader.u8()
            var cmp_op = reader.u8()
            var skip = reader.integer()
            var pidx = reader.integer()
            var bbk = reader.integer()
            var aid = reader.integer()
            var pf = reader.f32()
            if reader.failed:
                raise Error("BehaviorTreeComponent: truncated node")
            if kind > BT_NODE_LAST or abort_mode > BT_ABORT_BOTH or cmp_op > BT_CMP_LAST:
                raise Error("BehaviorTreeComponent: invalid node enum")
            if skip <= i or skip > n_nodes or pidx < -1 or pidx >= i:
                raise Error("BehaviorTreeComponent: invalid node hierarchy bounds")
            out.tree.nodes.append(
                BtNode(kind, abort_mode, cmp_op, Int32(skip), Int32(pidx), Int32(bbk), Int32(aid), pf)
            )

        var n_sc = Int(reader.u32())
        if reader.failed or n_sc < 0 or n_sc > 4096:
            raise Error("BehaviorTreeComponent: invalid scorer count")
        for _ in range(n_sc):
            var use_geo = reader.boolean()
            var sc = UtilityAiScorer(use_geo)
            var n_op = Int(reader.u32())
            if reader.failed or n_op < 0 or n_op > 4096:
                raise Error("BehaviorTreeComponent: invalid option count")
            for _ in range(n_op):
                var oid = reader.integer()
                var wt = reader.f32()
                var opt = UtilityOption(oid, wt)
                var n_con = Int(reader.u32())
                if reader.failed or n_con < 0 or n_con > 4096:
                    raise Error("BehaviorTreeComponent: invalid consideration count")
                for _ in range(n_con):
                    var cbb = reader.integer()
                    var cmin = reader.f32()
                    var cmax = reader.f32()
                    var ckind = reader.u8()
                    var cexp = reader.f32()
                    var cinv = reader.boolean()
                    if reader.failed:
                        raise Error("BehaviorTreeComponent: truncated consideration")
                    if ckind > CURVE_INVERSE:
                        raise Error("BehaviorTreeComponent: invalid curve_kind")
                    opt.add_consideration(UtilityConsideration(cbb, cmin, cmax, ckind, cexp, cinv))
                _ = sc.add_option(opt^)
            out.tree.utility_scorers.append(sc^)
        for i in range(n_nodes):
            if (
                out.tree.nodes[i].kind == BT_NODE_UTILITY_SELECT
                and Int(out.tree.nodes[i].action_id) >= len(out.tree.utility_scorers)
            ):
                raise Error("BehaviorTreeComponent: invalid utility_select action_id")

        var n_bb = Int(reader.u32())
        if reader.failed or n_bb < 0 or n_bb > 65536 or n_bb > max(0, reader.limit() - reader.offset) // 16:
            raise Error("BehaviorTreeComponent: invalid blackboard count")
        for _ in range(n_bb):
            var bn = reader.text()
            var bf = reader.f32()
            var bi = reader.integer()
            var bv = reader.vec3()
            if reader.failed:
                raise Error("BehaviorTreeComponent: truncated blackboard entry")
            out.blackboard.names.append(bn)
            out.blackboard.floats.append(bf)
            out.blackboard.ints.append(bi)
            out.blackboard.vecs.append(bv)

        out.state.time_s = reader.f32()
        out.state.running_node = reader.integer()
        var n_nr = Int(reader.u32())
        if reader.failed or n_nr < 0 or n_nr > 65536 or n_nr > max(0, reader.limit() - reader.offset):
            raise Error("BehaviorTreeComponent: invalid node_running count")
        out.state.node_running = List[Bool](capacity=n_nr)
        for _ in range(n_nr):
            out.state.node_running.append(reader.boolean())
        out.state.active_action_id = reader.integer()
        out.state.last_aborted_node = reader.integer()
        out.state.last_aborted_action_id = reader.integer()
        out.state.aborts_triggered = reader.integer()

        var n_cd = Int(reader.u32())
        if reader.failed or n_cd < 0 or n_cd > 65536 or n_cd > max(0, reader.limit() - reader.offset) // 4:
            raise Error("BehaviorTreeComponent: invalid cooldown count")
        out.state.cooldown_ready_at = List[Float32](capacity=n_cd)
        for _ in range(n_cd):
            out.state.cooldown_ready_at.append(reader.f32())

        var n_as = Int(reader.u32())
        if reader.failed or n_as < 0 or n_as > 65536 or n_as > max(0, reader.limit() - reader.offset):
            raise Error("BehaviorTreeComponent: invalid action_statuses count")
        out.state.action_statuses = List[UInt8](capacity=n_as)
        for _ in range(n_as):
            out.state.action_statuses.append(reader.u8())

        var n_ae = Int(reader.u32())
        if reader.failed or n_ae < 0 or n_ae > 65536 or n_ae > max(0, reader.limit() - reader.offset) // 8:
            raise Error("BehaviorTreeComponent: invalid action_exec_counts count")
        out.state.action_exec_counts = List[Int](capacity=n_ae)
        for _ in range(n_ae):
            out.state.action_exec_counts.append(reader.integer())

        out.tree_ref = AssetRef.read_bin(reader, 1)
        out.enabled = reader.boolean()
        out.tick_interval_sec = reader.f32()
        out.accum_sec = reader.f32()
        if reader.failed:
            raise Error("BehaviorTreeComponent: truncated tail")
        return out^

    def write_json(self) -> String:
        var out = String('{"tree": ') + self.tree.save_json()
        out += ', "blackboard": ['
        for i in range(len(self.blackboard.names)):
            if i > 0:
                out += ", "
            out += (
                '{"name": ' + json_quote(self.blackboard.names[i])
                + ', "f32": ' + _bt_f32_s(self.blackboard.floats[i])
                + ', "int": ' + String(self.blackboard.ints[i])
                + ', "vec3": ' + _bt_vec3_json(self.blackboard.vecs[i])
                + "}"
            )
        out += '], "state": {'
        out += '"time_s": ' + _bt_f32_s(self.state.time_s)
        out += ', "running_node": ' + String(self.state.running_node)
        out += ', "active_action_id": ' + String(self.state.active_action_id)
        out += ', "last_aborted_node": ' + String(self.state.last_aborted_node)
        out += ', "last_aborted_action_id": ' + String(self.state.last_aborted_action_id)
        out += ', "aborts_triggered": ' + String(self.state.aborts_triggered)
        out += ', "node_running": ['
        for i in range(len(self.state.node_running)):
            if i > 0:
                out += ", "
            out += "true" if self.state.node_running[i] else "false"
        out += '], "cooldown_ready_at": ['
        for i in range(len(self.state.cooldown_ready_at)):
            if i > 0:
                out += ", "
            out += _bt_f32_s(self.state.cooldown_ready_at[i])
        out += '], "action_statuses": ['
        for i in range(len(self.state.action_statuses)):
            if i > 0:
                out += ", "
            out += String(Int(self.state.action_statuses[i]))
        out += '], "action_exec_counts": ['
        for i in range(len(self.state.action_exec_counts)):
            if i > 0:
                out += ", "
            out += String(self.state.action_exec_counts[i])
        out += "]}"
        out += ', "tree_ref": ' + self.tree_ref.write_json()
        out += ', "enabled": ' + ("true" if self.enabled else "false")
        out += ', "tick_interval_sec": ' + _bt_f32_s(self.tick_interval_sec)
        out += ', "accum_sec": ' + _bt_f32_s(self.accum_sec)
        out += "}"
        return out^

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var out = Self()
        if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 5:
            return out^

        var tree_n = doc.object_get(node, "tree")
        if tree_n >= 0:
            var parsed = _bt_parse_json_node(doc, tree_n)
            if not parsed[0]:
                raise Error("BehaviorTreeComponent: invalid tree JSON")
            out.tree.nodes = parsed[1].copy()
            out.tree.utility_scorers = parsed[2].copy()

        var bb_n = doc.object_get(node, "blackboard")
        if bb_n >= 0 and doc.nodes[bb_n].kind == 4:
            for i in range(doc.array_size(bb_n)):
                var entry = doc.at(bb_n, i)
                if entry < 0 or doc.nodes[entry].kind != 5:
                    continue
                var bname = doc.text(doc.object_get(entry, "name"), "")
                var bf = Float32(doc.number(doc.object_get(entry, "f32"), 0.0))
                var bi = doc.integer(doc.object_get(entry, "int"), 0)
                var vn = doc.object_get(entry, "vec3")
                var bv = Vec3()
                if vn >= 0 and doc.nodes[vn].kind == 4 and doc.array_size(vn) >= 3:
                    bv = Vec3(
                        Float32(doc.number(doc.at(vn, 0), 0.0)),
                        Float32(doc.number(doc.at(vn, 1), 0.0)),
                        Float32(doc.number(doc.at(vn, 2), 0.0)),
                    )
                out.blackboard.names.append(bname)
                out.blackboard.floats.append(bf if isfinite(bf) else Float32(0.0))
                out.blackboard.ints.append(bi)
                out.blackboard.vecs.append(bv)

        var st_n = doc.object_get(node, "state")
        if st_n >= 0 and doc.nodes[st_n].kind == 5:
            var ts = Float32(doc.number(doc.object_get(st_n, "time_s"), 0.0))
            out.state.time_s = ts if isfinite(ts) else Float32(0.0)
            out.state.running_node = doc.integer(doc.object_get(st_n, "running_node"), -1)
            out.state.active_action_id = doc.integer(doc.object_get(st_n, "active_action_id"), -1)
            out.state.last_aborted_node = doc.integer(doc.object_get(st_n, "last_aborted_node"), -1)
            out.state.last_aborted_action_id = doc.integer(doc.object_get(st_n, "last_aborted_action_id"), -1)
            out.state.aborts_triggered = doc.integer(doc.object_get(st_n, "aborts_triggered"), 0)

            var nr_n = doc.object_get(st_n, "node_running")
            if nr_n >= 0 and doc.nodes[nr_n].kind == 4:
                out.state.node_running = List[Bool](capacity=doc.array_size(nr_n))
                for i in range(doc.array_size(nr_n)):
                    out.state.node_running.append(doc.boolean(doc.at(nr_n, i), False))

            var cd_n = doc.object_get(st_n, "cooldown_ready_at")
            if cd_n >= 0 and doc.nodes[cd_n].kind == 4:
                out.state.cooldown_ready_at = List[Float32](capacity=doc.array_size(cd_n))
                for i in range(doc.array_size(cd_n)):
                    var cv = Float32(doc.number(doc.at(cd_n, i), 0.0))
                    out.state.cooldown_ready_at.append(cv if isfinite(cv) else Float32(0.0))

            var as_n = doc.object_get(st_n, "action_statuses")
            if as_n >= 0 and doc.nodes[as_n].kind == 4:
                out.state.action_statuses = List[UInt8](capacity=doc.array_size(as_n))
                for i in range(doc.array_size(as_n)):
                    out.state.action_statuses.append(UInt8(max(0, min(255, doc.integer(doc.at(as_n, i), 0)))))

            var ae_n = doc.object_get(st_n, "action_exec_counts")
            if ae_n >= 0 and doc.nodes[ae_n].kind == 4:
                out.state.action_exec_counts = List[Int](capacity=doc.array_size(ae_n))
                for i in range(doc.array_size(ae_n)):
                    out.state.action_exec_counts.append(doc.integer(doc.at(ae_n, i), 0))

        var ref_n = doc.object_get(node, "tree_ref")
        if ref_n >= 0:
            out.tree_ref = AssetRef.read_json(doc, ref_n)
        out.enabled = doc.boolean(doc.object_get(node, "enabled"), True)
        var ti = Float32(doc.number(doc.object_get(node, "tick_interval_sec"), 0.0))
        out.tick_interval_sec = ti if isfinite(ti) else Float32(0.0)
        var acc = Float32(doc.number(doc.object_get(node, "accum_sec"), 0.0))
        out.accum_sec = acc if isfinite(acc) else Float32(0.0)
        return out^


def _ai_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _ai_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
        var val = C.read_bin(reader, ver)
        reader.leave()
        var dst = Pointer[C, MutAnyOrigin](unsafe_from_address=comp_addr)
        __get_address_as_uninit_lvalue(dst._mlir_value) = val^
        return True
    except:
        return False


def _register_non_trivial_ai[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    if tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(tid, _ai_bin_write[C], _ai_bin_read[C])


def register_ai_components(mut world: World):
    """Registers `BehaviorTreeComponent` and its binary snapshot hooks in `world`."""
    _register_non_trivial_ai[BehaviorTreeComponent](world)


def register_ai_codecs(mut reg: ComponentRegistry):
    """Registers `BehaviorTreeComponent` `.flscn`/`.flscnb` codecs on `ComponentRegistry`."""
    reg.register[BehaviorTreeComponent]()


def _sys_ai_behavior_tree_tick(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        var raw_dt = ft_opt.value()[unsafe_offset=0].delta_seconds
        dt = max(Float32(0.0), raw_dt) if isfinite(raw_dt) else Float32(0.0)

    var bt_mask = world.bit_for[BehaviorTreeComponent]()
    var ents = world.query_entities(bt_mask)
    for i in range(len(ents)):
        var e = ents[i]
        var btc_opt = world.get_ptr_mut[BehaviorTreeComponent](e)
        if not btc_opt:
            continue
        ref btc = btc_opt.value()[unsafe_offset=0]
        if not btc.enabled or len(btc.tree.nodes) == 0:
            continue
        if btc.tick_interval_sec > Float32(0.0):
            if not isfinite(btc.accum_sec):
                btc.accum_sec = Float32(0.0)
            btc.accum_sec += dt
            if btc.accum_sec < btc.tick_interval_sec:
                continue
            var step_dt = btc.accum_sec
            btc.accum_sec = Float32(0.0)
            _ = btc.tree.tick(btc.blackboard, btc.state, step_dt)
        else:
            _ = btc.tree.tick(btc.blackboard, btc.state, dt)


def _has_ai_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct AiPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `BehaviorTreeComponent` and schedules `STAGE_UPDATE` BehaviorTree ticking."""

    def __init__(out self):
        pass

    def build(self, mut app: EngineApp):
        register_ai_components(app.world)
        if _has_ai_system(app, STAGE_UPDATE, "flint.ai.behavior_tree_tick"):
            return
        var b_bt = app.world.bit_for[BehaviorTreeComponent]()
        _ = app.add_system(
            STAGE_UPDATE,
            "flint.ai.behavior_tree_tick",
            _sys_ai_behavior_tree_tick,
            reads=b_bt,
            writes=b_bt,
            exclusive=True,
        )
