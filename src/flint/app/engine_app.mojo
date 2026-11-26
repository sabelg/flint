"""Flint v5 Application Kernel, Type-Erased Resources, FixedTime & Parallel Conflict-DAG Schedule (`flint.app.engine_app`).

Provides:
- `trait Resource(Movable, Deinitable)` and `Resources` type-erased singleton registry
- `FixedTime` (fixed-dt accumulator + spiral-of-death clamp + `overstep_alpha` with harmonic snap)
- `FrameTime` (per-frame variable delta, elapsed wall time, frame counter)
- 8-stage `Schedule` (`STAGE_STARTUP .. STAGE_SHUTDOWN`) with `ComponentBitSet` read/write conflict DAG
  compiler and automatic `JobPool` `MAX_DEPENDENTS = 16` zero-chunk relay node splitting
- `trait EnginePlugin`, `AppConfig`, and `EngineApp` (`add_plugin`, `insert_resource`,
  `register_component`, `add_system`, `step_one_frame`, `run` supporting headless `max_frames`).
- `TimeControl` (pause / time scale / single-step honoured by the fixed and update stages) and
  `RngResource` (deterministic seeded `DetRng`, snapshot-friendly two-word state).

Scheduler safety rules:
- A system that declares no component reads/writes and no resource reads/writes is treated as
  `exclusive` (it may touch anything). Pass `parallel=True` to opt such a system into running
  concurrently with others (only when it really touches no shared state).
- Resource access is declared with `res_reads` / `res_writes` (lists of `Resource.RESOURCE_ID`) and
  participates in the conflict check exactly like component bits.
- Every system runs inside its own profiler zone (named after the system) when the app profiler is on.
"""
from std.collections import List
from std.ffi import external_call
from std.math import isfinite, min, max
from std.sys import size_of
from std.time import monotonic
from flint.core.crash import install_crash_handler
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool, MAX_DEPENDENTS, NO_JOB, default_worker_count
from flint.core.profiler import ProfileTimeline, Profiler, ZONE_NONE, profile_ticks
from flint.core.rng import DetRng
from flint.ecs.world import (
    World,
    Component,
    ComponentBitSet,
    ParallelCommands,
    comp_hash64,
)

# 8-stage Engine Schedule
comptime STAGE_STARTUP: Int = 0
comptime STAGE_FIRST: Int = 1
comptime STAGE_PRE_UPDATE: Int = 2
comptime STAGE_FIXED_UPDATE: Int = 3
comptime STAGE_UPDATE: Int = 4
comptime STAGE_POST_UPDATE: Int = 5
comptime STAGE_LAST: Int = 6
comptime STAGE_SHUTDOWN: Int = 7
comptime NUM_SCHEDULE_STAGES: Int = 8


# ---------------------------------------------------------------------------
# Resources: Type-Erased Resource Registry
# ---------------------------------------------------------------------------

trait Resource(Movable, Deinitable):
    """Trait implemented by engine and gameplay singleton resources."""
    comptime RESOURCE_ID: UInt64


comptime _ResDropFn = def(Int) thin -> None


def _noop_res_drop(_addr: Int):
    pass


def _res_emplace[R: Movable](addr: Int, var val: R):
    var p = Pointer[R, MutAnyOrigin](unsafe_from_address=addr)
    __get_address_as_uninit_lvalue(p._mlir_value) = val^


def _res_drop[R: Movable & Deinitable](addr: Int):
    var p = Pointer[R, MutAnyOrigin](unsafe_from_address=addr)
    var owned = __get_address_as_owned_value(p._mlir_value)
    _ = owned^


@fieldwise_init
struct _ResourceSlot(Copyable, ImplicitlyCopyable, Movable):
    var resource_id: UInt64
    var ptr: Int
    var byte_size: Int
    var has_drop: Bool
    var drop_fn: _ResDropFn

    @always_inline
    def drop_value(self):
        if self.ptr != 0 and self.byte_size > 0 and self.has_drop:
            var dfn = self.drop_fn
            dfn(self.ptr)


struct Resources(Movable):
    """Type-erased singleton resource container with stable 64-byte aligned pointers."""
    var slots: List[_ResourceSlot]
    var index_keys: List[UInt64]
    var index_vals: List[Int]

    def __init__(out self):
        self.slots = List[_ResourceSlot]()
        self.index_keys = List[UInt64]()
        self.index_vals = List[Int]()

    def __deinit__(deinit self):
        for i in range(len(self.slots)):
            self.slots[i].drop_value()
            if self.slots[i].ptr != 0:
                external_call["free", NoneType](self.slots[i].ptr)

    @always_inline
    def find_index(self, res_id: UInt64) -> Int:
        var cap = len(self.index_vals)
        if cap == 0:
            return -1
        var mask = cap - 1
        var slot = Int(res_id ^ (res_id >> 33)) & mask
        for _ in range(cap):
            var idx = self.index_vals[slot]
            if idx < 0:
                return -1
            if self.index_keys[slot] == res_id:
                return idx
            slot = (slot + 1) & mask
        return -1

    def _rebuild_index(mut self, min_cap: Int = 0):
        var cap = max(32, min_cap)
        var needed = len(self.slots) * 2
        while cap < needed:
            cap *= 2
        self.index_keys = List[UInt64](length=cap, fill=0)
        self.index_vals = List[Int](length=cap, fill=-1)
        var mask = cap - 1
        for i in range(len(self.slots)):
            var rid = self.slots[i].resource_id
            var slot = Int(rid ^ (rid >> 33)) & mask
            while self.index_vals[slot] >= 0:
                slot = (slot + 1) & mask
            self.index_keys[slot] = rid
            self.index_vals[slot] = i

    def _index_insert(mut self, res_id: UInt64, slot_idx: Int):
        var cap = len(self.index_vals)
        if cap == 0 or (len(self.slots) * 2) > cap:
            self._rebuild_index(cap * 2 if cap > 0 else 32)
            return
        var mask = cap - 1
        var slot = Int(res_id ^ (res_id >> 33)) & mask
        while self.index_vals[slot] >= 0 and self.index_keys[slot] != res_id:
            slot = (slot + 1) & mask
        self.index_keys[slot] = res_id
        self.index_vals[slot] = slot_idx

    def insert[R: Resource](mut self, var resource: R):
        comptime has_nontriv_drop = not R.__del__is_trivial
        var dfn: _ResDropFn = _noop_res_drop
        if has_nontriv_drop:
            dfn = _res_drop[R]
        var bsz = size_of[R]()
        var alloc_bytes = max(64, (max(1, bsz) + 63) & ~63)
        var mem = external_call["aligned_alloc", Int](64, alloc_bytes)
        if mem != 0:
            _ = external_call["memset", Int](mem, Int32(0), alloc_bytes)
            if bsz > 0:
                _res_emplace[R](mem, resource^)
            else:
                _ = resource^
        else:
            _ = resource^
        var idx = self.find_index(R.RESOURCE_ID)
        if idx >= 0:
            self.slots[idx].drop_value()
            if self.slots[idx].ptr != 0:
                external_call["free", NoneType](self.slots[idx].ptr)
            self.slots[idx].ptr = mem
            self.slots[idx].byte_size = bsz
            self.slots[idx].has_drop = has_nontriv_drop
            self.slots[idx].drop_fn = dfn
            return
        var new_idx = len(self.slots)
        self.slots.append(_ResourceSlot(R.RESOURCE_ID, mem, bsz, has_nontriv_drop, dfn))
        self._index_insert(R.RESOURCE_ID, new_idx)

    def insert_pod_raw(mut self, res_id: UInt64, byte_size: Int, src_ptr: Int) -> Int:
        """Inserts or overwrites a host-resident POD resource (reallocating cleanly across `.so` reloads)."""
        var safe_size = max(0, byte_size)
        var alloc_bytes = max(64, (max(1, safe_size) + 63) & ~63)
        var mem = external_call["aligned_alloc", Int](64, alloc_bytes)
        if mem != 0:
            _ = external_call["memset", Int](mem, Int32(0), alloc_bytes)
            if safe_size > 0 and src_ptr != 0:
                _ = external_call["memcpy", Int](mem, src_ptr, safe_size)
        var idx = self.find_index(res_id)
        if idx >= 0:
            self.slots[idx].drop_value()
            if self.slots[idx].ptr != 0:
                external_call["free", NoneType](self.slots[idx].ptr)
            self.slots[idx].ptr = mem
            self.slots[idx].byte_size = safe_size
            self.slots[idx].has_drop = False
            self.slots[idx].drop_fn = _noop_res_drop
            return mem
        var new_idx = len(self.slots)
        self.slots.append(_ResourceSlot(res_id, mem, safe_size, False, _noop_res_drop))
        self._index_insert(res_id, new_idx)
        return mem

    def byte_size_of(self, res_id: UInt64) -> Int:
        var idx = self.find_index(res_id)
        return self.slots[idx].byte_size if idx >= 0 else 0

    def get_raw_ptr(self, res_id: UInt64) -> Int:
        var idx = self.find_index(res_id)
        return self.slots[idx].ptr if idx >= 0 else 0

    def has[R: Resource](self) -> Bool:
        return self.find_index(R.RESOURCE_ID) >= 0

    def get_ptr[R: Resource](self) -> Optional[Pointer[R, ImmutAnyOrigin]]:
        var addr = self.get_raw_ptr(R.RESOURCE_ID)
        if addr == 0:
            return None
        return Pointer[R, ImmutAnyOrigin](unsafe_from_address=addr)

    def get_ptr_mut[R: Resource](mut self) -> Optional[Pointer[R, MutAnyOrigin]]:
        var addr = self.get_raw_ptr(R.RESOURCE_ID)
        if addr == 0:
            return None
        return Pointer[R, MutAnyOrigin](unsafe_from_address=addr)

    def remove[R: Resource](mut self) -> Bool:
        var idx = self.find_index(R.RESOURCE_ID)
        if idx < 0:
            return False
        self.slots[idx].drop_value()
        if self.slots[idx].ptr != 0:
            external_call["free", NoneType](self.slots[idx].ptr)
        var last = len(self.slots) - 1
        if idx < last:
            self.slots[idx] = self.slots[last]
        _ = self.slots.pop()
        self._rebuild_index()
        return True

    def count(self) -> Int:
        return len(self.slots)


# ---------------------------------------------------------------------------
# FixedTime & FrameTime Resources
# ---------------------------------------------------------------------------

@fieldwise_init
struct FixedTime(Resource, Copyable, ImplicitlyCopyable):
    """Fixed-timestep simulation clock with spiral-of-death clamp and `overstep_alpha`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.FixedTime")
    var dt: Float32
    var dt_f64: Float64
    var accumulator: Float64
    var max_accumulator: Float64
    var max_steps_per_frame: Int
    var overstep_alpha: Float32
    var tick_count: UInt64
    var clamped_last_frame: Bool

    def __init__(out self, hz: Float32 = 60.0, max_steps_per_frame: Int = 8):
        var safe_hz = max(Float32(1.0), hz) if isfinite(hz) else Float32(60.0)
        var step64 = Float64(1.0) / Float64(safe_hz)
        var max_steps = max(1, max_steps_per_frame)
        self.dt = Float32(step64)
        self.dt_f64 = step64
        self.accumulator = 0.0
        self.max_accumulator = step64 * Float64(max_steps)
        self.max_steps_per_frame = max_steps
        self.overstep_alpha = 0.0
        self.tick_count = 0
        self.clamped_last_frame = False

    def advance_frame(mut self, frame_dt: Float32) -> Int:
        """Accumulates `frame_dt`, clamps spiral-of-death, returns due fixed steps, and updates `overstep_alpha`."""
        var safe_dt = frame_dt if isfinite(frame_dt) else Float32(0.0)
        var delta = max(Float64(0.0), Float64(safe_dt))
        self.accumulator += delta
        self.clamped_last_frame = False
        if self.accumulator > self.max_accumulator:
            self.accumulator = self.max_accumulator
            self.clamped_last_frame = True

        # Sub-microsecond snap tolerance (1e-5 * dt) ensures harmonic display rates
        # (30 Hz, 60 Hz, 120 Hz, 144 Hz, 240 Hz) produce identical 60 Hz tick counts.
        var eps = self.dt_f64 * 1.0e-5
        var steps = 0
        while steps < self.max_steps_per_frame and (self.accumulator + eps) >= self.dt_f64:
            self.accumulator -= self.dt_f64
            if self.accumulator < 0.0 or self.accumulator <= eps:
                self.accumulator = 0.0
            steps += 1

        var alpha = self.accumulator / self.dt_f64 if self.dt_f64 > 0.0 else 0.0
        if alpha < 0.0:
            alpha = 0.0
        elif alpha >= 1.0:
            alpha = 0.999999
        self.overstep_alpha = Float32(alpha)
        return steps


@fieldwise_init
struct FrameTime(Resource, Copyable, ImplicitlyCopyable):
    """Variable render/app frame timing resource."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.FrameTime")
    var delta_seconds: Float32
    var elapsed_seconds: Float64
    var frame_index: UInt64

    def __init__(out self):
        self.delta_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.frame_index = 0


@fieldwise_init
struct WorldTickSyncResource(Resource, Copyable, ImplicitlyCopyable):
    """Tracks minimum `World.current_tick` observed prior to `STAGE_POST_UPDATE` so
    `TransformPlugin` detects rollbacks even when `STAGE_FIXED_UPDATE` advances ticks."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.WorldTickSyncResource")
    var min_tick_since_post_update: UInt32
    var has_observed: Bool

    def __init__(out self):
        self.min_tick_since_post_update = UInt32(0xFFFFFFFF)
        self.has_observed = False

    @always_inline
    def observe(mut self, tick: UInt32):
        if (not self.has_observed) or tick < self.min_tick_since_post_update:
            self.min_tick_since_post_update = tick
            self.has_observed = True

    @always_inline
    def reset(mut self):
        self.min_tick_since_post_update = UInt32(0xFFFFFFFF)
        self.has_observed = False


struct TimeControl(Resource, Copyable, ImplicitlyCopyable, Movable):
    """Game-time control honoured by `EngineApp.step_one_frame`.

    - `paused`: `STAGE_FIXED_UPDATE` gets no ticks, `STAGE_UPDATE` systems are skipped unless they
      were added with `run_when_paused=True` (set `pause_update_stage=False` to keep the whole update
      stage running), and `FrameTime.delta_seconds` is `0`. `STAGE_FIRST` (window/input),
      `STAGE_PRE_UPDATE`, `STAGE_POST_UPDATE` and `STAGE_LAST` (render/UI) keep running.
    - `time_scale`: multiplies the game delta fed to `FixedTime` and `FrameTime` (`0..64`).
    - `request_step(n)`: while paused, advances exactly one fixed tick per frame for `n` frames.
    - `real_delta_seconds` / `real_elapsed_seconds`: unscaled wall delta, for UI that must animate
      while paused."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.TimeControl")
    var paused: Bool
    var time_scale: Float32
    var pending_steps: Int
    var pause_update_stage: Bool
    var real_delta_seconds: Float32
    var real_elapsed_seconds: Float64
    var game_elapsed_seconds: Float64
    var stepping_this_frame: Bool

    def __init__(out self, paused: Bool = False, time_scale: Float32 = 1.0, pause_update_stage: Bool = True):
        self.paused = paused
        self.time_scale = Float32(1.0)
        self.pending_steps = 0
        self.pause_update_stage = pause_update_stage
        self.real_delta_seconds = 0.0
        self.real_elapsed_seconds = 0.0
        self.game_elapsed_seconds = 0.0
        self.stepping_this_frame = False
        self.set_time_scale(time_scale)

    def set_time_scale(mut self, scale: Float32):
        if not isfinite(scale):
            return
        self.time_scale = min(Float32(64.0), max(Float32(0.0), scale))

    def request_step(mut self, frames: Int = 1):
        self.pending_steps += max(0, frames)

    @always_inline
    def update_stage_paused(self) -> Bool:
        """True when this frame's `STAGE_UPDATE` should skip systems not marked `run_when_paused`."""
        return self.paused and self.pause_update_stage and not self.stepping_this_frame


struct RngResource(Resource, Copyable, ImplicitlyCopyable, Movable):
    """App-wide deterministic RNG. Snapshot `rng.state_words()` (or `rng.serialize`) with rollback state."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.RngResource")
    var rng: DetRng
    var seed: UInt64

    def __init__(out self, seed: UInt64 = 0x5EED5EED5EED5EED):
        self.seed = seed
        self.rng = DetRng(seed)

    def reseed(mut self, seed: UInt64):
        self.seed = seed
        self.rng = DetRng(seed)

    def fork(mut self, stream: UInt64) -> DetRng:
        return self.rng.fork(stream)


def _res_ids_intersect(a: List[UInt64], b: List[UInt64]) -> Bool:
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                return True
    return False


# ---------------------------------------------------------------------------
# 8-Stage Conflict-DAG Schedule with Automatic MAX_DEPENDENTS=16 Relay Splitting
# ---------------------------------------------------------------------------

comptime SystemFn = def(
    Pointer[World, MutAnyOrigin],
    Pointer[Resources, MutAnyOrigin],
    Pointer[JobPool, MutAnyOrigin],
    Pointer[ParallelCommands, MutAnyOrigin],
    Int,
) thin -> None


@fieldwise_init
struct SystemDescriptor(Copyable, Movable):
    var id: Int
    var name: String
    var fn_ptr: SystemFn
    var reads: ComponentBitSet
    var writes: ComponentBitSet
    var exclusive: Bool
    var user_ctx: Int
    var explicit_after: List[Int]
    var res_reads: List[UInt64]
    var res_writes: List[UInt64]
    var run_when_paused: Bool
    var profile_zone: Int


struct _CompiledNode(Copyable, Movable):
    """Either a real 1-chunk system node (`sys_idx >= 0`) or a 0-chunk relay node (`sys_idx == -1`)."""
    var sys_idx: Int
    var dependents: List[Int]

    def __init__(out self, sys_idx: Int):
        self.sys_idx = sys_idx
        self.dependents = List[Int]()

    def __init__(out self, *, copy: Self):
        self.sys_idx = copy.sys_idx
        self.dependents = copy.dependents.copy()


struct _StageGraph(Movable):
    var systems: List[SystemDescriptor]
    var compiled_nodes: List[_CompiledNode]
    var dirty: Bool
    var relay_count: Int

    def __init__(out self):
        self.systems = List[SystemDescriptor]()
        self.compiled_nodes = List[_CompiledNode]()
        self.dirty = True
        self.relay_count = 0

    @staticmethod
    def _conflicts(a: SystemDescriptor, b: SystemDescriptor) -> Bool:
        if a.exclusive or b.exclusive:
            return True
        if a.writes.intersects(b.writes):
            return True
        if a.writes.intersects(b.reads):
            return True
        if a.reads.intersects(b.writes):
            return True
        if _res_ids_intersect(a.res_writes, b.res_writes):
            return True
        if _res_ids_intersect(a.res_writes, b.res_reads):
            return True
        if _res_ids_intersect(a.res_reads, b.res_writes):
            return True
        return False

    def compile(mut self):
        if not self.dirty:
            return
        self.compiled_nodes.clear()
        self.relay_count = 0
        var n = len(self.systems)
        if n == 0:
            self.dirty = False
            return

        # Step 1: Build combined dependency graph over 0..n-1 with explicit dependencies taking
        # precedence and registration-order R/W & W/W conflict edges included before topological sort.
        var explicit_adj = List[List[Int]](capacity=n)
        var indegree = List[Int](length=n, fill=0)
        var step1_reach = List[Bool](length=n * n, fill=False)
        for i in range(n):
            explicit_adj.append(List[Int]())
            self.compiled_nodes.append(_CompiledNode(i))

        for j in range(n):
            for k in range(len(self.systems[j].explicit_after)):
                var target_id = self.systems[j].explicit_after[k]
                for i in range(n):
                    if i != j and self.systems[i].id == target_id:
                        var dup = False
                        for e in range(len(explicit_adj[i])):
                            if explicit_adj[i][e] == j:
                                dup = True
                                break
                        if not dup:
                            explicit_adj[i].append(j)
                            indegree[j] += 1
                            step1_reach[i * n + j] = True
                        break

        # Transitive closure of explicit edges so conflict edges never reverse or cycle against them.
        for mid in range(n):
            for u in range(n):
                if step1_reach[u * n + mid]:
                    for v in range(n):
                        if step1_reach[mid * n + v]:
                            step1_reach[u * n + v] = True

        # Add registration-order conflict edges (i < j) unless j is already ordered before i.
        for i in range(n):
            for j in range(i + 1, n):
                if Self._conflicts(self.systems[i], self.systems[j]):
                    if not step1_reach[j * n + i] and not step1_reach[i * n + j]:
                        explicit_adj[i].append(j)
                        indegree[j] += 1
                        for u in range(n):
                            if u == i or step1_reach[u * n + i]:
                                for v in range(n):
                                    if v == j or step1_reach[j * n + v]:
                                        step1_reach[u * n + v] = True

        var order = List[Int](capacity=n)
        var placed = List[Bool](length=n, fill=False)
        for _ in range(n):
            var picked = -1
            for cand in range(n):
                if not placed[cand] and indegree[cand] == 0:
                    picked = cand
                    break
            if picked < 0:
                break
            placed[picked] = True
            order.append(picked)
            for e in range(len(explicit_adj[picked])):
                var nxt = explicit_adj[picked][e]
                indegree[nxt] -= 1
        if len(order) < n:
            for cand in range(n):
                if not placed[cand]:
                    order.append(cand)

        # Step 2: Reachability bitmatrix (n x n booleans) in topological order to prune redundant
        # transitive edges while respecting explicit dependencies and R/W & W/W conflicts.
        var reach = List[Bool](length=n * n, fill=False)
        for oj in range(n):
            var j = order[oj]
            var oi = oj - 1
            while oi >= 0:
                var i = order[oi]
                var must_order = Self._conflicts(self.systems[i], self.systems[j])
                if not must_order:
                    for k in range(len(self.systems[j].explicit_after)):
                        if self.systems[j].explicit_after[k] == self.systems[i].id:
                            must_order = True
                            break
                if must_order and not reach[oi * n + oj]:
                    self.compiled_nodes[i].dependents.append(j)
                    reach[oi * n + oj] = True
                    for o_anc in range(oi):
                        if reach[o_anc * n + oi]:
                            reach[o_anc * n + oj] = True
                oi -= 1

        # Automatic JobPool MAX_DEPENDENTS = 16 zero-chunk relay node splitting:
        # Whenever any node u has > MAX_DEPENDENTS (16) outgoing dependents, keep the first
        # (MAX_DEPENDENTS - 1) = 15 dependents on u, create a zero-chunk relay node r,
        # attach u -> r (total 16), and move the remaining dependents onto r (iteratively!).
        var cursor = 0
        while cursor < len(self.compiled_nodes):
            var deg = len(self.compiled_nodes[cursor].dependents)
            if deg > MAX_DEPENDENTS:
                var keep = MAX_DEPENDENTS - 1
                var overflow = List[Int](capacity=deg - keep)
                for idx in range(keep, deg):
                    overflow.append(self.compiled_nodes[cursor].dependents[idx])
                self.compiled_nodes[cursor].dependents.resize(keep, 0)
                var relay_idx = len(self.compiled_nodes)
                var relay = _CompiledNode(-1)
                relay.dependents = overflow^
                self.compiled_nodes.append(relay^)
                self.compiled_nodes[cursor].dependents.append(relay_idx)
                self.relay_count += 1
            cursor += 1

        self.dirty = False


@fieldwise_init
struct _SysJobCtx(Copyable, ImplicitlyCopyable, Movable):
    var world_addr: Int
    var res_addr: Int
    var pool_addr: Int
    var cmds_addr: Int
    var fn_ptr: SystemFn
    var user_ctx: Int
    var skip: Bool
    var profiler_addr: Int
    var profile_zone: Int


def _exec_system_job(ctx_addr: Int, _chunk: Int):
    var ctx = Pointer[_SysJobCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    if ctx.skip:
        return
    var sfn = ctx.fn_ptr
    var timed = ctx.profiler_addr != 0 and ctx.profile_zone >= 0
    var started = profile_ticks() if timed else UInt64(0)
    sfn(
        Pointer[World, MutAnyOrigin](unsafe_from_address=ctx.world_addr),
        Pointer[Resources, MutAnyOrigin](unsafe_from_address=ctx.res_addr),
        Pointer[JobPool, MutAnyOrigin](unsafe_from_address=ctx.pool_addr),
        Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=ctx.cmds_addr),
        ctx.user_ctx,
    )
    if timed:
        Profiler(ctx.profiler_addr).record(ctx.profile_zone, started, profile_ticks())


@fieldwise_init
struct _StageParallelCtx(Copyable, ImplicitlyCopyable, Movable):
    var ctx_base: Int


def _exec_independent_stage_job(ctx_addr: Int, chunk: Int):
    var base = Pointer[_StageParallelCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[
        unsafe_offset=0
    ].ctx_base
    _exec_system_job(base + chunk * size_of[_SysJobCtx](), 0)


def _noop_relay_job(_ctx: Int, _chunk: Int):
    pass


struct Schedule(Movable):
    """8-stage parallel ECS schedule with conflict-DAG compilation and relay splitting."""
    var stages: List[_StageGraph]
    var next_system_id: Int

    def __init__(out self):
        self.stages = List[_StageGraph](capacity=NUM_SCHEDULE_STAGES)
        for _ in range(NUM_SCHEDULE_STAGES):
            self.stages.append(_StageGraph())
        self.next_system_id = 0

    def add_system(
        mut self,
        stage: Int,
        name: String,
        fn_ptr: SystemFn,
        reads: ComponentBitSet = ComponentBitSet(),
        writes: ComponentBitSet = ComponentBitSet(),
        exclusive: Bool = False,
        user_ctx: Int = 0,
        after_system_id: Int = -1,
        res_reads: List[UInt64] = List[UInt64](),
        res_writes: List[UInt64] = List[UInt64](),
        parallel: Bool = False,
        run_when_paused: Bool = False,
        profile_zone: Int = ZONE_NONE,
    ) -> Int:
        """Adds a system. A system declaring no component and no resource access is exclusive
        unless `parallel=True` explicitly opts it into concurrent execution."""
        if stage < 0 or stage >= NUM_SCHEDULE_STAGES:
            return -1
        var sid = self.next_system_id
        self.next_system_id += 1
        var after = List[Int]()
        if after_system_id >= 0:
            after.append(after_system_id)
        var undeclared = (
            reads.is_empty() and writes.is_empty() and len(res_reads) == 0 and len(res_writes) == 0
        )
        var is_exclusive = exclusive or (undeclared and not parallel)
        self.stages[stage].systems.append(
            SystemDescriptor(
                sid,
                name,
                fn_ptr,
                reads.copy(),
                writes.copy(),
                is_exclusive,
                user_ctx,
                after^,
                res_reads.copy(),
                res_writes.copy(),
                run_when_paused,
                profile_zone,
            )
        )
        self.stages[stage].dirty = True
        return sid

    def add_dependency(mut self, stage: Int, system_id: Int, after_system_id: Int):
        if stage < 0 or stage >= NUM_SCHEDULE_STAGES:
            return
        for i in range(len(self.stages[stage].systems)):
            if self.stages[stage].systems[i].id == system_id:
                self.stages[stage].systems[i].explicit_after.append(after_system_id)
                self.stages[stage].dirty = True
                return

    def compile_all(mut self):
        for s in range(NUM_SCHEDULE_STAGES):
            self.stages[s].compile()

    def compiled_relay_count(mut self, stage: Int) -> Int:
        if stage < 0 or stage >= NUM_SCHEDULE_STAGES:
            return 0
        self.stages[stage].compile()
        return self.stages[stage].relay_count

    def compiled_node_count(mut self, stage: Int) -> Int:
        if stage < 0 or stage >= NUM_SCHEDULE_STAGES:
            return 0
        self.stages[stage].compile()
        return len(self.stages[stage].compiled_nodes)

    def run_stage(
        mut self,
        stage: Int,
        mut world: World,
        mut resources: Resources,
        mut pool: JobPool,
        mut commands: ParallelCommands,
        profiler_addr: Int = 0,
        paused: Bool = False,
    ):
        """Runs one stage. `profiler_addr` (a `Profiler.address`, 0 = off) times each system in its
        zone; `paused=True` skips systems not added with `run_when_paused=True`."""
        if stage < 0 or stage >= NUM_SCHEDULE_STAGES:
            return
        self.stages[stage].compile()
        var total_nodes = len(self.stages[stage].compiled_nodes)
        if total_nodes == 0:
            return
        _ = pool.allocate()
        pool.rouse()

        var world_addr = Int(Pointer(to=world))
        var res_addr = Int(Pointer(to=resources))
        var pool_addr = Int(Pointer(to=pool))
        var cmds_addr = Int(Pointer(to=commands))

        var num_sys = len(self.stages[stage].systems)
        var sys_ctxs = List[_SysJobCtx](capacity=max(1, num_sys))
        for i in range(num_sys):
            ref sd = self.stages[stage].systems[i]
            sys_ctxs.append(
                _SysJobCtx(
                    world_addr,
                    res_addr,
                    pool_addr,
                    cmds_addr,
                    sd.fn_ptr,
                    sd.user_ctx,
                    paused and not sd.run_when_paused,
                    profiler_addr,
                    sd.profile_zone,
                )
            )
        var ctx_base = Int(sys_ctxs.unsafe_ptr())

        # Fast path: when all compiled nodes are independent systems (zero DAG edges),
        # dispatch directly via `flint.core.ecs.run_parallel_ctx`.
        var has_edges = False
        for u in range(total_nodes):
            if (
                self.stages[stage].compiled_nodes[u].sys_idx < 0
                or len(self.stages[stage].compiled_nodes[u].dependents) > 0
            ):
                has_edges = True
                break

        if not has_edges:
            var stage_ctx = _StageParallelCtx(ctx_base)
            run_parallel_ctx(pool, num_sys, stage_ctx, _exec_independent_stage_job)
            world.keep_alive()
            var anchor_ind_ctxs = len(sys_ctxs)
            var anchor_ind_r = resources.count()
            var anchor_ind_c = commands.pending_count()
            _ = anchor_ind_ctxs
            _ = anchor_ind_r
            _ = anchor_ind_c
            return

        var handles = List[Int](length=total_nodes, fill=NO_JOB)
        for u in range(total_nodes):
            var sidx = self.stages[stage].compiled_nodes[u].sys_idx
            if sidx >= 0:
                handles[u] = pool.create(
                    _exec_system_job,
                    ctx_base + sidx * size_of[_SysJobCtx](),
                    chunks=1,
                )
            else:
                # Zero-chunk relay node: completes automatically in JobPool.release()
                handles[u] = pool.create(_noop_relay_job, 0, chunks=0)

        var sink = pool.create(_noop_relay_job, 0, chunks=0)

        for u in range(total_nodes):
            var deg = len(self.stages[stage].compiled_nodes[u].dependents)
            if deg == 0:
                pool.depend(sink, handles[u])
            else:
                for d in range(deg):
                    var v = self.stages[stage].compiled_nodes[u].dependents[d]
                    pool.depend(handles[v], handles[u])

        for u in range(total_nodes):
            pool.submit(handles[u])
        pool.submit(sink)
        pool.wait(sink)

        # Lifetime anchors past the JobPool barrier
        world.keep_alive()
        var anchor_ctxs = len(sys_ctxs)
        var anchor_handles = len(handles)
        var anchor_r = resources.count()
        var anchor_c = commands.pending_count()
        _ = anchor_ctxs
        _ = anchor_handles
        _ = anchor_r
        _ = anchor_c


# ---------------------------------------------------------------------------
# EnginePlugin, AppConfig, and EngineApp
# ---------------------------------------------------------------------------

comptime EventSwapFn = def(Pointer[Resources, MutAnyOrigin]) thin -> None


struct AppConfig(Copyable, Movable):
    var title: String
    var headless: Bool
    var worker_threads: Int
    var fixed_hz: Float32
    var max_fixed_steps_per_frame: Int
    var default_frame_dt: Float32
    var max_frame_dt: Float32
    var enable_profiler: Bool
    var profile_path: String
    var enable_crash_handler: Bool
    var rng_seed: UInt64

    def __init__(
        out self,
        title: String = "Flint v5 EngineApp",
        headless: Bool = True,
        worker_threads: Int = -1,
        fixed_hz: Float32 = 60.0,
        max_fixed_steps_per_frame: Int = 8,
        default_frame_dt: Float32 = Float32(1.0 / 60.0),
        max_frame_dt: Float32 = Float32(0.1),
        enable_profiler: Bool = False,
        profile_path: String = "",
        enable_crash_handler: Bool = False,
        rng_seed: UInt64 = 0x5EED5EED5EED5EED,
    ):
        self.title = title
        self.headless = headless
        self.worker_threads = default_worker_count() if worker_threads < 0 else worker_threads
        self.fixed_hz = max(Float32(1.0), fixed_hz) if isfinite(fixed_hz) else Float32(60.0)
        self.max_fixed_steps_per_frame = max_fixed_steps_per_frame
        self.default_frame_dt = (
            max(Float32(0.0), default_frame_dt) if isfinite(default_frame_dt) else Float32(1.0 / 60.0)
        )
        self.max_frame_dt = (
            max(Float32(0.001), max_frame_dt) if isfinite(max_frame_dt) else Float32(0.1)
        )
        self.enable_profiler = enable_profiler
        self.profile_path = profile_path
        self.enable_crash_handler = enable_crash_handler
        self.rng_seed = rng_seed


trait EnginePlugin:
    """Modular plugin installing components, resources, and scheduled systems into an `EngineApp`."""
    def build(self, mut app: EngineApp):
        ...


struct EngineApp(Movable):
    """Unified Flint v5 Engine Application hosting `World`, `Resources`, `Schedule`, and `JobPool`."""
    var config: AppConfig
    var world: World
    var resources: Resources
    var schedule: Schedule
    var pool: JobPool
    var commands: ParallelCommands
    var event_swap_hooks: List[EventSwapFn]
    var profiler: ProfileTimeline
    var crash_handler_installed: Bool
    var started: Bool
    var shut_down: Bool

    def __init__(out self, var config: AppConfig = AppConfig()):
        var fixed_hz = config.fixed_hz
        var max_steps = config.max_fixed_steps_per_frame
        var rng_seed = config.rng_seed
        self.config = config^
        self.world = World()
        self.resources = Resources()
        self.schedule = Schedule()
        self.pool = JobPool()
        self.commands = ParallelCommands()
        self.event_swap_hooks = List[EventSwapFn]()
        self.profiler = ProfileTimeline()
        self.crash_handler_installed = False
        self.started = False
        self.shut_down = False
        self.resources.insert[FixedTime](FixedTime(fixed_hz, max_steps))
        self.resources.insert[FrameTime](FrameTime())
        self.resources.insert[WorldTickSyncResource](WorldTickSyncResource())
        self.resources.insert[TimeControl](TimeControl())
        self.resources.insert[RngResource](RngResource(rng_seed))

    def __deinit__(deinit self):
        if self.profiler.active:
            self.profiler.finish()
        self.pool.stop()

    def add_plugin[P: EnginePlugin](mut self, plugin: P):
        plugin.build(self)

    def insert_resource[R: Resource](mut self, var resource: R):
        self.resources.insert[R](resource^)

    def register_component[C: Component](mut self) -> Int:
        return self.world.register_component[C]()

    def add_event_swap_hook(mut self, hook: EventSwapFn):
        """Registers an automatic per-frame `Events` buffer swap callback invoked after `STAGE_LAST`."""
        self.event_swap_hooks.append(hook)

    def register_profiler_zone(mut self, name: String) -> Int:
        """Dynamically registers a custom profiler zone on this app's `ProfileTimeline`."""
        return self.profiler.register_zone(name)

    def add_system(
        mut self,
        stage: Int,
        name: String,
        fn_ptr: SystemFn,
        reads: ComponentBitSet = ComponentBitSet(),
        writes: ComponentBitSet = ComponentBitSet(),
        exclusive: Bool = False,
        user_ctx: Int = 0,
        after_system_id: Int = -1,
        res_reads: List[UInt64] = List[UInt64](),
        res_writes: List[UInt64] = List[UInt64](),
        parallel: Bool = False,
        run_when_paused: Bool = False,
    ) -> Int:
        """Adds a system (see `Schedule.add_system`); it gets a profiler zone named after it."""
        var zone = self.profiler.register_zone(name)
        return self.schedule.add_system(
            stage,
            name,
            fn_ptr,
            reads,
            writes,
            exclusive,
            user_ctx,
            after_system_id,
            res_reads,
            res_writes,
            parallel,
            run_when_paused,
            zone,
        )

    # ---- Pause / time scale ----

    def pause(mut self):
        var tc = self.resources.get_ptr_mut[TimeControl]()
        if tc:
            tc.value()[unsafe_offset=0].paused = True

    def resume(mut self):
        var tc = self.resources.get_ptr_mut[TimeControl]()
        if tc:
            tc.value()[unsafe_offset=0].paused = False
            tc.value()[unsafe_offset=0].pending_steps = 0

    def is_paused(self) -> Bool:
        var tc = self.resources.get_ptr[TimeControl]()
        return tc.value()[unsafe_offset=0].paused if tc else False

    def set_time_scale(mut self, scale: Float32):
        var tc = self.resources.get_ptr_mut[TimeControl]()
        if tc:
            tc.value()[unsafe_offset=0].set_time_scale(scale)

    def request_step(mut self, frames: Int = 1):
        """While paused, advance exactly one fixed tick (and run `STAGE_UPDATE`) per frame for `frames` frames."""
        var tc = self.resources.get_ptr_mut[TimeControl]()
        if tc:
            tc.value()[unsafe_offset=0].request_step(frames)

    @always_inline
    def _profiler_addr(self) -> Int:
        return self.profiler.handle.address if self.profiler.active else 0

    def startup(mut self):
        if self.started:
            return
        if self.config.enable_crash_handler and not self.crash_handler_installed:
            self.crash_handler_installed = install_crash_handler(self.config.title)
        if self.config.enable_profiler and not self.profiler.active:
            _ = self.profiler.start(self.config.profile_path, overlay=not self.config.headless)
        _ = self.pool.allocate()
        if self.config.worker_threads > 0:
            _ = self.pool.start(self.config.worker_threads)
        self.schedule.compile_all()
        var paddr = self._profiler_addr()
        self.schedule.run_stage(
            STAGE_STARTUP, self.world, self.resources, self.pool, self.commands, paddr
        )
        self.world.flush_commands(self.commands)
        self.started = True

    @always_inline
    def _observe_tick_sync(mut self):
        var sync_opt = self.resources.get_ptr_mut[WorldTickSyncResource]()
        if sync_opt:
            sync_opt.value()[unsafe_offset=0].observe(self.world.current_tick)

    def step_one_frame(mut self, frame_dt: Float32):
        if not self.started:
            self.startup()

        self._observe_tick_sync()

        if self.profiler.active:
            self.profiler.begin_frame()
        var paddr = self._profiler_addr()

        var safe_frame_dt = frame_dt if isfinite(frame_dt) else Float32(0.0)
        var safe_max_dt = self.config.max_frame_dt if isfinite(self.config.max_frame_dt) else Float32(0.1)
        var clamped_dt = max(Float32(0.0), min(safe_max_dt, safe_frame_dt))

        # Game time: scaled by `TimeControl.time_scale`; zero while paused, except a requested
        # single step, which advances exactly one fixed tick.
        var game_dt = clamped_dt
        var update_paused = False
        var force_one_tick = False
        var tc_opt = self.resources.get_ptr_mut[TimeControl]()
        if tc_opt:
            ref tc = tc_opt.value()[unsafe_offset=0]
            tc.real_delta_seconds = clamped_dt
            tc.real_elapsed_seconds += Float64(clamped_dt)
            tc.stepping_this_frame = False
            if tc.paused:
                if tc.pending_steps > 0:
                    tc.pending_steps -= 1
                    tc.stepping_this_frame = True
                    force_one_tick = True
                game_dt = Float32(0.0)
            else:
                game_dt = clamped_dt * tc.time_scale
            update_paused = tc.update_stage_paused()

        if force_one_tick:
            var fix_ro = self.resources.get_ptr[FixedTime]()
            if fix_ro:
                game_dt = fix_ro.value()[unsafe_offset=0].dt
        if tc_opt:
            tc_opt.value()[unsafe_offset=0].game_elapsed_seconds += Float64(game_dt)

        var ft_opt = self.resources.get_ptr_mut[FrameTime]()
        if ft_opt:
            var ft_ptr = ft_opt.value()
            ft_ptr[unsafe_offset=0].delta_seconds = game_dt
            ft_ptr[unsafe_offset=0].elapsed_seconds += Float64(game_dt)
            ft_ptr[unsafe_offset=0].frame_index += 1

        self.schedule.run_stage(STAGE_FIRST, self.world, self.resources, self.pool, self.commands, paddr)
        self.world.flush_commands(self.commands)
        self._observe_tick_sync()

        self.schedule.run_stage(STAGE_PRE_UPDATE, self.world, self.resources, self.pool, self.commands, paddr)
        self.world.flush_commands(self.commands)
        self._observe_tick_sync()

        var due_steps = 0
        var fix_opt = self.resources.get_ptr_mut[FixedTime]()
        if fix_opt:
            if force_one_tick:
                due_steps = 1
            else:
                due_steps = fix_opt.value()[unsafe_offset=0].advance_frame(game_dt)

        for _ in range(due_steps):
            _ = self.world.advance_tick()
            self.schedule.run_stage(STAGE_FIXED_UPDATE, self.world, self.resources, self.pool, self.commands, paddr)
            self.world.flush_commands(self.commands)
            self._observe_tick_sync()
            var fp2 = self.resources.get_ptr_mut[FixedTime]()
            if fp2:
                fp2.value()[unsafe_offset=0].tick_count += 1

        self.schedule.run_stage(
            STAGE_UPDATE, self.world, self.resources, self.pool, self.commands, paddr, update_paused
        )
        self.world.flush_commands(self.commands)
        self._observe_tick_sync()

        self.schedule.run_stage(STAGE_POST_UPDATE, self.world, self.resources, self.pool, self.commands, paddr)
        self.world.flush_commands(self.commands)

        var post_update_tick = self.world.current_tick
        var sync_rst = self.resources.get_ptr_mut[WorldTickSyncResource]()
        if sync_rst:
            sync_rst.value()[unsafe_offset=0].reset()

        self.schedule.run_stage(STAGE_LAST, self.world, self.resources, self.pool, self.commands, paddr)
        self.world.flush_commands(self.commands)
        if self.world.current_tick < post_update_tick:
            self._observe_tick_sync()

        var res_ptr = Pointer[Resources, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=self.resources)))
        for i in range(len(self.event_swap_hooks)):
            var hook = self.event_swap_hooks[i]
            hook(res_ptr)

        # Removed-component / despawn events stay readable for one frame after they happen.
        self.world.update_removal_events()
        _ = self.world.advance_tick()

        if self.profiler.active:
            self.profiler.end_frame()

    def shutdown(mut self):
        if self.shut_down:
            return
        if self.started:
            var paddr = self._profiler_addr()
            self.schedule.run_stage(
                STAGE_SHUTDOWN, self.world, self.resources, self.pool, self.commands, paddr
            )
            self.world.flush_commands(self.commands)
        if self.profiler.active:
            self.profiler.finish()
        self.pool.stop()
        self.shut_down = True

    def run(mut self, max_frames: Int = 0):
        """Runs startup, executes `max_frames` frames cleanly in headless/bounded mode, and shuts down."""
        self.startup()
        for _ in range(max(0, max_frames)):
            self.step_one_frame(self.config.default_frame_dt)
        self.shutdown()

    def run_realtime(mut self, max_frames: Int = 0, max_seconds: Float64 = 0.0):
        """Wall-clock `monotonic()` frame loop with automatic `max_frame_dt` clamping."""
        self.startup()
        var start_ns = monotonic()
        var last_ns = start_ns
        var frame = 0
        while True:
            if max_frames > 0 and frame >= max_frames:
                break
            var now_ns = monotonic()
            if max_seconds > 0.0 and Float64(now_ns - start_ns) * 1e-9 >= max_seconds:
                break
            var raw_dt = Float32(Float64(now_ns - last_ns) * 1e-9)
            last_ns = now_ns
            var safe_raw_dt = raw_dt if isfinite(raw_dt) else Float32(0.0)
            var safe_max_dt = self.config.max_frame_dt if isfinite(self.config.max_frame_dt) else Float32(0.1)
            var dt = min(safe_max_dt, max(Float32(0.0), safe_raw_dt))
            self.step_one_frame(dt)
            frame += 1
        self.shutdown()
