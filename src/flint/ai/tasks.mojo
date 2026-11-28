"""Reusable Behaviour-Tree Leaf Task Library (`flint.ai.tasks`, v5).

`BehaviorTree` action leaves are integer `action_id`s whose status the game writes into
`BtExecutionState.action_statuses`. `BtTaskLibrary` binds action ids to ready-made leaf
implementations so games do not have to hand-write common behaviours:

- `MoveTo`: navmesh path following (`NavPathFollower`, sliced A*, repath, off-mesh links) to a
  `Vec3` blackboard key; SUCCESS within `acceptance_radius`, FAILURE if no path exists.
- `Wait`: RUNNING for N seconds (see also the native `BT_NODE_WAIT` leaf).
- `RotateTo`: turns `BtAgentContext.yaw_deg` toward a `Vec3` blackboard key at `turn_rate_deg`.
- `SetBlackboard`: writes a Float32 / Int / Vec3 / Bool / entity value, then SUCCESS.

Protocol: `BtTaskLibrary.tick(tree, bb, state, ctx, nav, dt)` ticks the tree, then advances every
task leaf the tree evaluated this tick. A finished task reports its terminal status on the next
evaluation (one-tick latency) and re-arms to RUNNING afterwards; aborted tasks reset.
"""
from std.collections import List
from std.math import sqrt, abs, min, max, isfinite
from flint.core.math import Vec3
from flint.core.detmath import det_atan2
from flint.nav.navmesh import NavMesh
from flint.nav.pathfollow import NavPathFollower, NAV_FOLLOW_FAILED, NAV_FOLLOW_ARRIVED, NAV_FOLLOW_OFFMESH_TRAVERSE
from flint.ai.behavior_tree import (
    Blackboard,
    BehaviorTree,
    BtExecutionState,
    BT_SUCCESS,
    BT_FAILURE,
    BT_RUNNING,
)

comptime BT_TASK_MOVE_TO: Int = 0
comptime BT_TASK_WAIT: Int = 1
comptime BT_TASK_ROTATE_TO: Int = 2
comptime BT_TASK_SET_BLACKBOARD: Int = 3

comptime BB_VALUE_FLOAT: Int = 0
comptime BB_VALUE_INT: Int = 1
comptime BB_VALUE_VEC3: Int = 2
comptime BB_VALUE_BOOL: Int = 3
comptime BB_VALUE_ENTITY: Int = 4

comptime _TASK_IDLE: Int = 0
comptime _TASK_ACTIVE: Int = 1
comptime _TASK_DONE: Int = 2

comptime _RAD_TO_DEG: Float32 = 57.29577951


struct BtTaskSpec(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var action_id: Int
    var bb_key: Int
    var value_type: Int
    var value_f32: Float32
    var value_int: Int
    var value_vec: Vec3
    var value_bool: Bool
    var value_entity: UInt64
    var radius: Float32  # MoveTo acceptance radius / RotateTo tolerance (deg) / Wait seconds

    def __init__(out self, kind: Int, action_id: Int, bb_key: Int = -1, radius: Float32 = 0.0):
        self.kind = kind
        self.action_id = action_id
        self.bb_key = bb_key
        self.value_type = BB_VALUE_FLOAT
        self.value_f32 = 0.0
        self.value_int = 0
        self.value_vec = Vec3()
        self.value_bool = False
        self.value_entity = 0
        self.radius = radius


struct BtAgentContext(Copyable, Movable):
    """Agent body state the task library reads and writes."""
    var position: Vec3
    var yaw_deg: Float32  # forward = (sin(yaw), 0, cos(yaw))
    var turn_rate_deg: Float32
    var velocity: Vec3
    var follower: NavPathFollower
    var auto_integrate: Bool  # True: the library moves `position` by `velocity * dt` (snapped to the navmesh)

    def __init__(out self, position: Vec3 = Vec3(), max_speed: Float32 = 3.5):
        self.position = position
        self.yaw_deg = 0.0
        self.turn_rate_deg = 360.0
        self.velocity = Vec3()
        self.follower = NavPathFollower(max_speed)
        self.auto_integrate = True


struct BtTaskLibrary(Copyable, Movable):
    var specs: List[BtTaskSpec]
    var phase: List[Int]
    var last_exec: List[Int]
    var timer: List[Float32]
    var goal_cache: List[Vec3]

    def __init__(out self):
        self.specs = List[BtTaskSpec]()
        self.phase = List[Int]()
        self.last_exec = List[Int]()
        self.timer = List[Float32]()
        self.goal_cache = List[Vec3]()

    def _add(mut self, spec: BtTaskSpec) -> Int:
        self.specs.append(spec)
        self.phase.append(_TASK_IDLE)
        self.last_exec.append(0)
        self.timer.append(0.0)
        self.goal_cache.append(Vec3())
        return len(self.specs) - 1

    def add_move_to(mut self, action_id: Int, target_key: Int, acceptance_radius: Float32 = 0.4) -> Int:
        return self._add(BtTaskSpec(BT_TASK_MOVE_TO, action_id, target_key, max(Float32(0.05), acceptance_radius)))

    def add_wait(mut self, action_id: Int, seconds: Float32) -> Int:
        return self._add(BtTaskSpec(BT_TASK_WAIT, action_id, -1, max(Float32(0.0), seconds)))

    def add_rotate_to(mut self, action_id: Int, target_key: Int, tolerance_deg: Float32 = 3.0) -> Int:
        return self._add(BtTaskSpec(BT_TASK_ROTATE_TO, action_id, target_key, max(Float32(0.1), tolerance_deg)))

    def add_set_float(mut self, action_id: Int, key: Int, value: Float32) -> Int:
        var s = BtTaskSpec(BT_TASK_SET_BLACKBOARD, action_id, key)
        s.value_type = BB_VALUE_FLOAT
        s.value_f32 = value
        return self._add(s)

    def add_set_int(mut self, action_id: Int, key: Int, value: Int) -> Int:
        var s = BtTaskSpec(BT_TASK_SET_BLACKBOARD, action_id, key)
        s.value_type = BB_VALUE_INT
        s.value_int = value
        return self._add(s)

    def add_set_vec3(mut self, action_id: Int, key: Int, value: Vec3) -> Int:
        var s = BtTaskSpec(BT_TASK_SET_BLACKBOARD, action_id, key)
        s.value_type = BB_VALUE_VEC3
        s.value_vec = value
        return self._add(s)

    def add_set_bool(mut self, action_id: Int, key: Int, value: Bool) -> Int:
        var s = BtTaskSpec(BT_TASK_SET_BLACKBOARD, action_id, key)
        s.value_type = BB_VALUE_BOOL
        s.value_bool = value
        return self._add(s)

    def add_set_entity(mut self, action_id: Int, key: Int, value: UInt64) -> Int:
        var s = BtTaskSpec(BT_TASK_SET_BLACKBOARD, action_id, key)
        s.value_type = BB_VALUE_ENTITY
        s.value_entity = value
        return self._add(s)

    def prime(self, mut state: BtExecutionState):
        """Arms every task action as RUNNING so the first evaluation starts the task."""
        for i in range(len(self.specs)):
            state.set_action_status(self.specs[i].action_id, BT_RUNNING)

    def _exec_count(self, state: BtExecutionState, aid: Int) -> Int:
        if aid >= 0 and aid < len(state.action_exec_counts):
            return state.action_exec_counts[aid]
        return 0

    def _start(mut self, i: Int, bb: Blackboard, mut ctx: BtAgentContext, nav: NavMesh):
        var sp = self.specs[i]
        self.timer[i] = 0.0
        if sp.kind == BT_TASK_MOVE_TO:
            var tgt = bb.get_vec3(sp.bb_key)
            self.goal_cache[i] = tgt
            ctx.follower.arrival_radius = sp.radius
            ctx.follower.set_goal(nav, ctx.position, tgt)

    def _abort(mut self, i: Int, mut ctx: BtAgentContext):
        if self.specs[i].kind == BT_TASK_MOVE_TO:
            ctx.follower.stop()
            ctx.velocity = Vec3()

    def _step(mut self, i: Int, mut bb: Blackboard, mut ctx: BtAgentContext, nav: NavMesh, dt: Float32) -> UInt8:
        var sp = self.specs[i]
        if sp.kind == BT_TASK_SET_BLACKBOARD:
            if sp.value_type == BB_VALUE_FLOAT:
                bb.set_float(sp.bb_key, sp.value_f32)
            elif sp.value_type == BB_VALUE_INT:
                bb.set_int(sp.bb_key, sp.value_int)
            elif sp.value_type == BB_VALUE_VEC3:
                bb.set_vec3(sp.bb_key, sp.value_vec)
            elif sp.value_type == BB_VALUE_BOOL:
                bb.set_bool(sp.bb_key, sp.value_bool)
            else:
                bb.set_entity(sp.bb_key, sp.value_entity)
            return BT_SUCCESS
        if sp.kind == BT_TASK_WAIT:
            self.timer[i] += dt
            return BT_SUCCESS if self.timer[i] >= sp.radius - Float32(1.0e-5) else BT_RUNNING
        if sp.kind == BT_TASK_ROTATE_TO:
            var tgt = bb.get_vec3(sp.bb_key)
            var dx = tgt.x - ctx.position.x
            var dz = tgt.z - ctx.position.z
            if dx * dx + dz * dz < Float32(1.0e-8):
                return BT_SUCCESS
            var want = det_atan2(dx, dz) * _RAD_TO_DEG
            var diff = want - ctx.yaw_deg
            while diff > Float32(180.0):
                diff -= Float32(360.0)
            while diff < Float32(-180.0):
                diff += Float32(360.0)
            if abs(diff) <= sp.radius:
                return BT_SUCCESS
            var step = ctx.turn_rate_deg * dt
            if abs(diff) <= step:
                ctx.yaw_deg = want
                return BT_SUCCESS
            ctx.yaw_deg += step if diff > Float32(0.0) else -step
            return BT_RUNNING
        # MoveTo
        var tgt = bb.get_vec3(sp.bb_key)
        if (tgt - self.goal_cache[i]).length_squared() > sp.radius * sp.radius:
            self.goal_cache[i] = tgt
            ctx.follower.set_goal(nav, ctx.position, tgt)
        var dx = tgt.x - ctx.position.x
        var dz = tgt.z - ctx.position.z
        if dx * dx + dz * dz <= sp.radius * sp.radius:
            ctx.velocity = Vec3()
            ctx.follower.stop()
            return BT_SUCCESS
        ctx.velocity = ctx.follower.update(nav, ctx.position, dt)
        if ctx.follower.state == NAV_FOLLOW_FAILED:
            ctx.velocity = Vec3()
            return BT_FAILURE
        if ctx.follower.state == NAV_FOLLOW_ARRIVED:
            ctx.velocity = Vec3()
            return BT_SUCCESS
        if ctx.velocity.length_squared() > Float32(1.0e-8):
            ctx.yaw_deg = det_atan2(ctx.velocity.x, ctx.velocity.z) * _RAD_TO_DEG
        return BT_RUNNING

    def tick(
        mut self,
        tree: BehaviorTree,
        mut bb: Blackboard,
        mut state: BtExecutionState,
        mut ctx: BtAgentContext,
        nav: NavMesh,
        dt: Float32,
    ) -> UInt8:
        var safe_dt = dt if (isfinite(dt) and dt > Float32(0.0)) else Float32(0.0)
        var st = tree.tick(bb, state, safe_dt)
        ctx.velocity = Vec3()
        for i in range(len(self.specs)):
            var aid = self.specs[i].action_id
            var cnt = self._exec_count(state, aid)
            var evaluated = cnt != self.last_exec[i]
            self.last_exec[i] = cnt
            if self.phase[i] == _TASK_DONE:
                # Terminal status was consumed by this evaluation (or the branch moved on): re-arm
                self.phase[i] = _TASK_IDLE
                state.set_action_status(aid, BT_RUNNING)
                continue
            if evaluated:
                if self.phase[i] == _TASK_IDLE:
                    self._start(i, bb, ctx, nav)
                    self.phase[i] = _TASK_ACTIVE
                var r = self._step(i, bb, ctx, nav, safe_dt)
                if r != BT_RUNNING:
                    self.phase[i] = _TASK_DONE
                    state.set_action_status(aid, r)
            elif self.phase[i] == _TASK_ACTIVE:
                self.phase[i] = _TASK_IDLE
                self._abort(i, ctx)
        if ctx.auto_integrate:
            if ctx.follower.offmesh_landed:
                ctx.follower.offmesh_landed = False
                ctx.position = ctx.follower.offmesh_to
            if ctx.follower.state == NAV_FOLLOW_OFFMESH_TRAVERSE:
                ctx.position = ctx.follower.offmesh_position()
            elif ctx.velocity.length_squared() > Float32(0.0):
                var np = ctx.position + ctx.velocity * safe_dt
                var near = nav.find_nearest_poly(np, Float32(2.0))
                if near[0] >= 0:
                    np.y = near[1].y
                ctx.position = np
        return st
