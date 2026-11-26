"""Flint v5 core gap fixes: deterministic RNG, scheduler safety, pause/time scale, per-system
profiler zones, ECS removal events + hierarchy-safe despawn, and hot-reload rollback.

Run from the repository root (the hot-reload section compiles small guest `.so` plugins with the
Mojo toolchain found via `MOJO_PREFIX` or `CONDA_PREFIX`; it prints `[SKIP]` when neither is set).
"""
from std.collections import List
from std.ffi import external_call
from std.testing import assert_true, assert_equal

from flint.core.rng import DetRng
from flint.core.serial import BlobWriter, BlobReader
from flint.core.jobs import JobPool
from flint.core.profiler import ZONE_COUNT, HISTORY_FRAMES, profile_ticks
from flint.platform.os import env_now, file_size, remove_file, write_bytes
from flint.ecs.world import (
    World,
    Entity,
    Component,
    ParallelCommands,
    STORAGE_TABLE,
    STORAGE_SPARSE,
    comp_hash64,
)
from flint.app.engine_app import (
    EngineApp,
    AppConfig,
    Resource,
    Resources,
    Schedule,
    FrameTime,
    FixedTime,
    TimeControl,
    RngResource,
    STAGE_FIXED_UPDATE,
    STAGE_UPDATE,
    STAGE_LAST,
)
from flint.app.hot_reload import HotReloadHost, PodSchemaRegistryResource, PodComponentSchema
from flint.scene.transform_hierarchy import (
    Parent,
    Children,
    HierarchyDepth,
    register_transform_components,
    set_parent,
    despawn_recursive,
    despawn_recursive_deferred,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@fieldwise_init
struct GapTag(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.gaps.GapTag")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var v: Int32


@fieldwise_init
struct GapSparse(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.gaps.GapSparse")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var v: Int32


@fieldwise_init
struct GapCounters(Resource, Copyable, ImplicitlyCopyable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("test.gaps.GapCounters")
    var fixed_runs: Int
    var update_runs: Int
    var update_when_paused_runs: Int
    var last_runs: Int
    var last_update_delta: Float32


@fieldwise_init
struct GapHotRes(Resource, Copyable, ImplicitlyCopyable):
    comptime RESOURCE_ID: UInt64 = 0x6A95107000000001
    var loads: Int64
    var unloads: Int64


def _counters(res_ptr: Pointer[Resources, MutAnyOrigin]) -> Pointer[GapCounters, MutAnyOrigin]:
    return res_ptr[unsafe_offset=0].get_ptr_mut[GapCounters]().value()


def _sys_noop(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    pass


def _sys_fixed_count(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    _counters(r)[unsafe_offset=0].fixed_runs += 1


def _sys_update_count(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var dt = r[unsafe_offset=0].get_ptr[FrameTime]().value()[unsafe_offset=0].delta_seconds
    var c = _counters(r)
    c[unsafe_offset=0].update_runs += 1
    c[unsafe_offset=0].last_update_delta = dt


def _sys_update_paused_ok(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    _counters(r)[unsafe_offset=0].update_when_paused_runs += 1


def _sys_last_count(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    _counters(r)[unsafe_offset=0].last_runs += 1


def _sys_busy(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var t0 = profile_ticks()
    while profile_ticks() - t0 < 200000:
        pass


# ---------------------------------------------------------------------------
# 1. Deterministic RNG
# ---------------------------------------------------------------------------

def test_det_rng() raises:
    # PCG32 reference vector (pcg32-demo: seed 42, stream 54).
    var ref_rng = DetRng(42, 54)
    assert_equal(ref_rng.next_u32(), UInt32(0xA15C02B7))
    assert_equal(ref_rng.next_u32(), UInt32(0x7B47F409))
    assert_equal(ref_rng.next_u32(), UInt32(0xBA1D3330))

    var a = DetRng(1234)
    var b = DetRng(1234)
    for _ in range(1000):
        assert_equal(a.next_u64(), b.next_u64())
    var c = DetRng(1235)
    var a2 = DetRng(1234)
    assert_true(a2.next_u64() != c.next_u64(), "different seeds must diverge")

    var r = DetRng(7)
    var lo_hit = False
    var hi_hit = False
    for _ in range(20000):
        var f = r.next_f32()
        assert_true(f >= Float32(0.0) and f < Float32(1.0))
        var d = r.next_f64()
        assert_true(d >= 0.0 and d < 1.0)
        var k = r.range_int(-3, 4)
        assert_true(k >= -3 and k < 4)
        if k == -3:
            lo_hit = True
        if k == 3:
            hi_hit = True
        var g = r.range_f32(Float32(2.0), Float32(5.0))
        assert_true(g >= Float32(2.0) and g < Float32(5.0))
    assert_true(lo_hit and hi_hit, "range_int must reach both ends")
    assert_equal(r.below(0), UInt32(0))
    assert_equal(r.range_int(5, 5), 5)

    # Forks: deterministic, independent of each other and of the parent.
    var p1 = DetRng(99)
    var p2 = DetRng(99)
    var f1 = p1.fork(1)
    var f1b = p2.fork(1)
    assert_equal(f1.next_u64(), f1b.next_u64())
    var f2 = p1.fork(2)
    var f2b = p2.fork(1)
    assert_true(f2.next_u64() != f2b.next_u64(), "different fork streams must diverge")

    # Skip-ahead equals stepping.
    var s1 = DetRng(555)
    var s2 = DetRng(555)
    for _ in range(777):
        _ = s1.next_u32()
    s2.advance(777)
    assert_true(s1 == s2, "advance(n) must equal n draws")

    # Serialization round-trip (rollback snapshots).
    var live = DetRng(2024)
    for _ in range(13):
        _ = live.next_u32()
    var w = BlobWriter()
    live.serialize(w)
    var expected = live.next_u64()
    var restored = DetRng(0)
    var reader = BlobReader(w.data.copy())
    restored.deserialize(reader)
    assert_equal(restored.next_u64(), expected)
    var words = live.state_words()
    var rebuilt = DetRng.from_state(words[0], words[1])
    assert_equal(rebuilt.next_u64(), live.next_u64())

    # RngResource is installed by EngineApp with the configured seed.
    var app = EngineApp(AppConfig(title="RngApp", headless=True, worker_threads=0, rng_seed=77))
    var rr = app.resources.get_ptr_mut[RngResource]()
    assert_true(Bool(rr))
    assert_equal(rr.value()[unsafe_offset=0].seed, UInt64(77))
    var want = DetRng(77)
    assert_equal(rr.value()[unsafe_offset=0].rng.next_u32(), want.next_u32())


# ---------------------------------------------------------------------------
# 2. Scheduler safety
# ---------------------------------------------------------------------------

def test_scheduler_safety() raises:
    var sched = Schedule()
    _ = sched.add_system(STAGE_UPDATE, "undeclared", _sys_noop)
    _ = sched.add_system(STAGE_UPDATE, "opt_in_a", _sys_noop, parallel=True)
    _ = sched.add_system(STAGE_UPDATE, "opt_in_b", _sys_noop, parallel=True)
    assert_true(sched.stages[STAGE_UPDATE].systems[0].exclusive, "undeclared systems must be exclusive")
    assert_true(not sched.stages[STAGE_UPDATE].systems[1].exclusive, "parallel=True must opt out")
    sched.stages[STAGE_UPDATE].compile()
    assert_equal(len(sched.stages[STAGE_UPDATE].compiled_nodes[0].dependents), 2)
    assert_equal(len(sched.stages[STAGE_UPDATE].compiled_nodes[1].dependents), 0)
    assert_equal(len(sched.stages[STAGE_UPDATE].compiled_nodes[2].dependents), 0)

    # Resource read/write sets participate in the conflict check.
    var rx = comp_hash64("test.gaps.ResX")
    var ry = comp_hash64("test.gaps.ResY")
    _ = sched.add_system(STAGE_FIXED_UPDATE, "w_x", _sys_noop, res_writes=[rx])
    _ = sched.add_system(STAGE_FIXED_UPDATE, "r_x", _sys_noop, res_reads=[rx])
    _ = sched.add_system(STAGE_FIXED_UPDATE, "r_y", _sys_noop, res_reads=[ry])
    _ = sched.add_system(STAGE_FIXED_UPDATE, "r_x2", _sys_noop, res_reads=[rx])
    ref st = sched.stages[STAGE_FIXED_UPDATE]
    for i in range(4):
        assert_true(not st.systems[i].exclusive, "declared resource access must not force exclusive")
    st.compile()
    assert_equal(len(st.compiled_nodes[0].dependents), 2)  # w_x -> r_x, r_x2
    assert_equal(len(st.compiled_nodes[1].dependents), 0)  # readers never conflict
    assert_equal(len(st.compiled_nodes[2].dependents), 0)  # disjoint resource
    var hits_rx2 = False
    for d in range(len(st.compiled_nodes[0].dependents)):
        if st.compiled_nodes[0].dependents[d] == 3:
            hits_rx2 = True
    assert_true(hits_rx2, "writer of X must precede every reader of X")

    # Exclusive-by-default systems run correctly through EngineApp.
    var app = EngineApp(AppConfig(title="SchedApp", headless=True, worker_threads=2))
    app.insert_resource[GapCounters](GapCounters(0, 0, 0, 0, 0.0))
    for _ in range(8):
        _ = app.add_system(STAGE_UPDATE, "count", _sys_update_count)
    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))
    app.shutdown()
    assert_equal(app.resources.get_ptr[GapCounters]().value()[unsafe_offset=0].update_runs, 8)


# ---------------------------------------------------------------------------
# 3. Pause, single-step & time scale
# ---------------------------------------------------------------------------

def _counts(app: EngineApp) -> GapCounters:
    return app.resources.get_ptr[GapCounters]().value()[unsafe_offset=0]


def test_pause_and_time_scale() raises:
    var cid: List[UInt64] = [GapCounters.RESOURCE_ID]
    var app = EngineApp(AppConfig(title="PauseApp", headless=True, worker_threads=2, fixed_hz=60.0))
    app.insert_resource[GapCounters](GapCounters(0, 0, 0, 0, 0.0))
    _ = app.add_system(STAGE_FIXED_UPDATE, "fixed", _sys_fixed_count, res_writes=cid)
    _ = app.add_system(STAGE_UPDATE, "update", _sys_update_count, res_writes=cid)
    _ = app.add_system(STAGE_UPDATE, "update_ui", _sys_update_paused_ok, res_writes=cid, run_when_paused=True)
    _ = app.add_system(STAGE_LAST, "render", _sys_last_count, res_writes=cid)
    app.startup()
    var dt = Float32(1.0 / 60.0)
    for _ in range(10):
        app.step_one_frame(dt)
    var c0 = _counts(app)
    assert_equal(c0.fixed_runs, 10)
    assert_equal(c0.update_runs, 10)
    assert_equal(c0.last_runs, 10)

    app.pause()
    assert_true(app.is_paused())
    for _ in range(5):
        app.step_one_frame(dt)
    var c1 = _counts(app)
    assert_equal(c1.fixed_runs, 10, "paused: no fixed ticks")
    assert_equal(c1.update_runs, 10, "paused: update systems skipped")
    assert_equal(c1.update_when_paused_runs, 15, "run_when_paused systems keep running")
    assert_equal(c1.last_runs, 15, "render stage keeps running while paused")
    assert_equal(app.resources.get_ptr[FrameTime]().value()[unsafe_offset=0].delta_seconds, Float32(0.0))
    var tc = app.resources.get_ptr[TimeControl]().value()[unsafe_offset=0]
    assert_true(tc.real_delta_seconds > Float32(0.0), "real delta keeps flowing while paused")

    # Single step: exactly one fixed tick and one update per requested frame.
    app.request_step(2)
    for _ in range(4):
        app.step_one_frame(dt)
    var c2 = _counts(app)
    assert_equal(c2.fixed_runs, 12)
    assert_equal(c2.update_runs, 12)

    # Time scale 0.5: 60 frames at 60 Hz -> 30 fixed ticks.
    app.resume()
    app.set_time_scale(Float32(0.5))
    for _ in range(60):
        app.step_one_frame(dt)
    var c3 = _counts(app)
    assert_equal(c3.fixed_runs - c2.fixed_runs, 30)
    assert_true(abs(c3.last_update_delta - dt * Float32(0.5)) < Float32(1.0e-6))
    app.shutdown()


# ---------------------------------------------------------------------------
# 4. Per-system profiler zones
# ---------------------------------------------------------------------------

def test_profiler_zones() raises:
    var app = EngineApp(
        AppConfig(title="ProfApp", headless=True, worker_threads=2, enable_profiler=True)
    )
    _ = app.add_system(STAGE_UPDATE, "gaps.busy_system", _sys_busy, parallel=True)
    var zone = app.schedule.stages[STAGE_UPDATE].systems[0].profile_zone
    assert_true(zone >= ZONE_COUNT, "each system gets its own custom zone")
    assert_equal(app.profiler.zone_label(zone), "gaps.busy_system")
    app.startup()
    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
    var last_frame = app.profiler.frame - 1
    var ms = app.profiler.history[zone * HISTORY_FRAMES + last_frame % HISTORY_FRAMES]
    assert_true(ms > Float32(0.0), "system zone must record time")
    app.shutdown()

    # Profiler off: zones registered but nothing recorded and systems still run.
    var app2 = EngineApp(AppConfig(title="NoProf", headless=True, worker_threads=2))
    app2.insert_resource[GapCounters](GapCounters(0, 0, 0, 0, 0.0))
    _ = app2.add_system(STAGE_UPDATE, "update", _sys_update_count)
    app2.run(max_frames=3)
    assert_equal(_counts(app2).update_runs, 3)
    assert_true(not app2.profiler.active)


# ---------------------------------------------------------------------------
# 5. ECS removal events & hierarchy-safe despawn
# ---------------------------------------------------------------------------

def _depth(w: World, e: Entity) -> Int:
    var d = w.get_ptr[HierarchyDepth](e)
    return Int(d.value()[unsafe_offset=0].depth) if d else -1


def test_ecs_removal_and_hierarchy() raises:
    var w = World()
    _ = w.register_component[GapTag]()
    _ = w.register_component[GapSparse]()
    var e1 = w.spawn_empty()
    var e2 = w.spawn_empty()
    _ = w.insert[GapTag](e1, GapTag(1))
    _ = w.insert[GapTag](e2, GapTag(2))
    _ = w.insert[GapSparse](e2, GapSparse(3))
    assert_equal(len(w.removed[GapTag]()), 0)

    assert_true(w.remove[GapTag](e1))
    var rem = w.removed[GapTag]()
    assert_equal(len(rem), 1)
    assert_true(rem[0] == e1)
    assert_true(not w.remove[GapTag](e1), "second remove is a no-op")
    assert_equal(len(w.removed[GapTag]()), 1)

    assert_true(w.despawn(e2))
    assert_equal(len(w.removed[GapTag]()), 2)
    assert_equal(len(w.removed[GapSparse]()), 1)
    var desp = w.despawned_entities()
    assert_equal(len(desp), 1)
    assert_true(desp[0] == e2 and w.was_despawned(e2))

    w.update_removal_events()  # one frame later: still readable
    assert_equal(len(w.removed[GapTag]()), 2)
    assert_equal(len(w.despawned_entities()), 1)
    w.update_removal_events()  # two frames later: gone
    assert_equal(len(w.removed[GapTag]()), 0)
    assert_equal(len(w.despawned_entities()), 0)

    # Hierarchy: despawning a parent orphans its children (no dangling links).
    register_transform_components(w)
    var p = w.spawn_empty()
    var a = w.spawn_empty()
    var b = w.spawn_empty()
    var a1 = w.spawn_empty()
    assert_true(set_parent(w, a, p))
    assert_true(set_parent(w, b, p))
    assert_true(set_parent(w, a1, a))
    assert_equal(_depth(w, a1), 2)
    assert_true(w.despawn(p))
    assert_true(w.is_alive(a) and w.is_alive(b) and w.is_alive(a1))
    assert_true(not w.has[Parent](a) and not w.has[Parent](b), "orphans lose Parent")
    assert_equal(_depth(w, a), 0)
    assert_equal(_depth(w, a1), 1)
    assert_true(w.get_ptr[Parent](a1).value()[unsafe_offset=0].entity == a)
    assert_equal(len(w.removed[Parent]()), 2)

    # Despawning a child unlinks it from its parent's Children.
    assert_true(set_parent(w, b, a))
    assert_equal(w.get_ptr[Children](a).value()[unsafe_offset=0].len(), 2)
    assert_true(w.despawn(b))
    assert_equal(w.get_ptr[Children](a).value()[unsafe_offset=0].len(), 1)
    assert_true(w.get_ptr[Children](a).value()[unsafe_offset=0].at(0) == a1)

    # Recursive despawn removes the subtree and unlinks the root from its parent.
    var root = w.spawn_empty()
    assert_true(set_parent(w, a, root))
    var n_before = w.entity_count()
    assert_equal(despawn_recursive(w, a), 2)
    assert_true(not w.is_alive(a) and not w.is_alive(a1))
    assert_equal(w.entity_count(), n_before - 2)
    assert_equal(w.get_ptr[Children](root).value()[unsafe_offset=0].len(), 0)

    # Deferred recursive despawn through ParallelCommands.
    var r2 = w.spawn_empty()
    var k1 = w.spawn_empty()
    var k2 = w.spawn_empty()
    assert_true(set_parent(w, k1, r2))
    assert_true(set_parent(w, k2, k1))
    var cmds = ParallelCommands()
    despawn_recursive_deferred(w, cmds, 0, r2)
    w.flush_commands(cmds)
    assert_true(not w.is_alive(r2) and not w.is_alive(k1) and not w.is_alive(k2))

    # EngineApp rotates removal events once per frame.
    var app = EngineApp(AppConfig(title="RmApp", headless=True, worker_threads=0))
    _ = app.register_component[GapTag]()
    var x = app.world.spawn_empty()
    _ = app.world.insert[GapTag](x, GapTag(9))
    app.startup()
    _ = app.world.despawn(x)
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(app.world.despawned_entities()), 1)
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(app.world.despawned_entities()), 0)
    app.shutdown()


# ---------------------------------------------------------------------------
# 6. Hot-reload rollback
# ---------------------------------------------------------------------------

def _write_text(path: String, text: String) -> Bool:
    var bytes = List[UInt8](capacity=text.byte_length())
    for byte in text.as_bytes():
        bytes.append(byte)
    return write_bytes(path, bytes)


def _guest_source(pod_size: Int, load_code: Int, step_tag: Int) -> String:
    var s = String("from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, ")
    s += "PodComponentSchema, PodFieldDesc, POD_FIELD_F32\n\n"
    s += "@export(\"flint_plugin_api_version\")\n"
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return FLINT_PLUGIN_API_VERSION\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(api_ptr: Int, world_ptr: Int, res_ptr: Int, _is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var rptr = api.get_resource_ptr(res_ptr, UInt64(0x6A95107000000001))\n"
    s += "    if rptr != 0:\n"
    s += "        Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=0] += 1\n"
    s += "    var schema = PodComponentSchema(\"GapsHotPod\", " + String(pod_size) + ", 4, 0)\n"
    s += "    _ = schema.add_field(PodFieldDesc(\"a\", POD_FIELD_F32, 0, 4))\n"
    s += "    _ = api.register_schema(world_ptr, res_ptr, schema)\n"
    s += "    return Int32(" + String(load_code) + ")\n\n"
    s += "@export(\"flint_plugin_on_step\")\n"
    s += "def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi(\"C\") -> Int32:\n"
    s += "    return Int32(" + String(step_tag) + ")\n\n"
    s += "@export(\"flint_plugin_on_unload\")\n"
    s += "def flint_plugin_on_unload(api_ptr: Int, _world_ptr: Int, res_ptr: Int, _is_reload: Int32) abi(\"C\") -> None:\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    var rptr = api.get_resource_ptr(res_ptr, UInt64(0x6A95107000000001))\n"
    s += "    if rptr != 0:\n"
    s += "        Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=1] += 1\n"
    return s^


def _hot(res: Resources) -> GapHotRes:
    return res.get_ptr[GapHotRes]().value()[unsafe_offset=0]


def test_hot_reload_rollback() raises -> Bool:
    var prefix = env_now("MOJO_PREFIX")
    if prefix.byte_length() == 0 or file_size(prefix + "/bin/mojo") <= 0:
        prefix = env_now("CONDA_PREFIX")
    if prefix.byte_length() == 0 or file_size(prefix + "/bin/mojo") <= 0:
        return False
    var pid = String(Int(external_call["getpid", Int32]()))
    var base = "/tmp/flint-gaps-hr-" + pid
    var names = List[String]()
    names.append("good")
    names.append("fail")
    names.append("refuse")
    names.append("good2")
    assert_true(_write_text(base + "-good.mojo", _guest_source(4, 0, 11)))
    assert_true(_write_text(base + "-fail.mojo", _guest_source(4, 5, 22)))
    assert_true(_write_text(base + "-refuse.mojo", _guest_source(8, 0, 33)))
    assert_true(_write_text(base + "-good2.mojo", _guest_source(4, 0, 44)))
    var cmd = (
        "export MODULAR_HOME=" + prefix + "/share/max; export LD_LIBRARY_PATH=" + prefix
        + "/lib:${LD_LIBRARY_PATH:-}; "
    )
    for i in range(len(names)):
        cmd += (
            prefix + "/bin/mojo build -I src " + base + "-" + names[i] + ".mojo --emit shared-lib -o "
            + base + "-" + names[i] + ".so -O0 >/dev/null 2>&1 & "
        )
    cmd += "wait"
    _ = external_call["system", Int32](cmd.as_c_string_span())
    for i in range(len(names)):
        assert_true(file_size(base + "-" + names[i] + ".so") > 0, "guest build failed: " + names[i])

    var world = World()
    var res = Resources()
    res.insert[GapHotRes](GapHotRes(0, 0))
    var host = HotReloadHost(base + "-stage")
    assert_true(host.load_or_reload(base + "-good.so", world, res), host.last_error)
    assert_equal(Int(host.step(world, res, Float32(0.0))), 11)
    var good_temp = host.active_temp_path
    assert_equal(Int(_hot(res).loads), 1)

    # on_load error -> previous plugin restored (unloaded for reload, then reloaded).
    assert_true(not host.load_or_reload(base + "-fail.so", world, res))
    assert_true(host.is_loaded(), "previous plugin must stay loaded")
    assert_true(host.last_error.find("rolled back") >= 0, host.last_error)
    assert_true(host.last_error.find("error code 5") >= 0, host.last_error)
    assert_equal(Int(host.step(world, res, Float32(0.0))), 11)
    assert_equal(host.active_temp_path, good_temp)
    assert_equal(host.rollback_count, 1)
    assert_equal(Int(_hot(res).loads), 3)  # good, fail, good (restore)
    assert_equal(Int(_hot(res).unloads), 2)  # good (for reload), fail (cleanup)

    # Schema refusal (POD layout changed) -> rollback even though on_load returned 0.
    assert_true(not host.load_or_reload(base + "-refuse.so", world, res))
    assert_true(host.is_loaded())
    assert_true(host.last_error.find("schema refused") >= 0, host.last_error)
    assert_equal(Int(host.step(world, res, Float32(0.0))), 11)
    assert_equal(host.rollback_count, 2)

    # A good reload still succeeds afterwards and clears the error.
    assert_true(host.load_or_reload(base + "-good2.so", world, res), host.last_error)
    assert_equal(Int(host.step(world, res, Float32(0.0))), 44)
    assert_equal(host.last_error, "")
    assert_true(file_size(good_temp) <= 0, "previous staged copy removed after a successful swap")
    assert_true(host.unload(world, res))

    # A failing first load leaves nothing loaded and explains why (no rollback target).
    var fresh = HotReloadHost(base + "-stage2")
    assert_true(not fresh.load_or_reload(base + "-fail.so", world, res))
    assert_true(not fresh.is_loaded())
    assert_true(fresh.last_error.find("error code 5") >= 0, fresh.last_error)

    for i in range(len(names)):
        _ = remove_file(base + "-" + names[i] + ".mojo")
        _ = remove_file(base + "-" + names[i] + ".so")
    return True


def main() raises:
    print("[Flint v5] Core gap fixes suite")
    test_det_rng()
    print("  [PASS] (1) DetRng PCG32: reference vector, ranges, forks, skip-ahead, serialize, RngResource")
    test_scheduler_safety()
    print("  [PASS] (2) Scheduler: undeclared systems exclusive, parallel opt-in, resource R/W conflicts")
    test_pause_and_time_scale()
    print("  [PASS] (3) TimeControl: pause keeps render/UI, single-step, time scale 0.5")
    test_profiler_zones()
    print("  [PASS] (4) Per-system profiler zones (recorded when on, inert when off)")
    test_ecs_removal_and_hierarchy()
    print("  [PASS] (5) ECS Removed<C>/despawn events + hierarchy-safe and recursive despawn")
    if test_hot_reload_rollback():
        print("  [PASS] (6) Hot-reload rollback on on_load error and schema refusal")
    else:
        print("  [SKIP] (6) Hot-reload rollback: set MOJO_PREFIX to the Mojo toolchain prefix")
    print("[Flint v5] Core gap fixes suite: ALL PASS")
