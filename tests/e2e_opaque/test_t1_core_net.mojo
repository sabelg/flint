"""Tier 1 (Happy Path) Opaque-Box E2E Suite for FA2 (Core/App/ECS/Scene/Jobs/Math/Memory) & FA7 (Net).

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`).
"""
from std.atomic import Atomic
from std.collections import List
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.app import (
    STAGE_STARTUP,
    STAGE_FIXED_UPDATE,
    STAGE_UPDATE,
    STAGE_POST_UPDATE,
    STAGE_SHUTDOWN,
    Resource,
    Resources,
    FixedTime,
    FrameTime,
    AppConfig,
    EngineApp,
)
from flint.ecs import (
    STORAGE_TABLE,
    STORAGE_SPARSE,
    comp_hash64,
    Entity,
    Component,
    ParallelCommands,
    Events,
    World,
)
from flint.scene import (
    INLINE_CHILD_CAP,
    SerializableComponent,
    LocalTransform,
    GlobalTransform,
    PrevGlobalTransform,
    Children,
    HierarchyDepth,
    TransformPlugin,
    set_parent,
    interpolate_global_transform,
    ComponentRegistry,
    Prefab,
    PrefabOverride,
    NestedPrefabRef,
    Scene,
)
from flint.core.detmath import (
    PI,
    f32_bits,
    det_sin,
    det_cos,
    det_atan,
    det_atan2,
    det_asin,
    det_acos,
    det_exp,
    det_log,
    det_pow,
    scale_by_pow2,
    set_sim_float_mode,
)
from flint.core.jobs import JobPool, run_jobs_affine
from flint.core.lz4 import hash_list
from flint.core.math import Vec3, Quat
from flint.core.memory import Arena, FrameArenas, MEM_FRAME
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.assets.json import JsonDocument

from flint.net import (
    AEAD_TAG_BYTES,
    AeadKey,
    BitReader,
    BitWriter,
    DatagramView,
    IPV6_SAFE_MTU_BYTES,
    JITTER_TRIANGULAR,
    NetChannel,
    NetEndpoint,
    NetHub,
    ReplayWindow64,
    ReplicatedEntity,
    RollbackSession,
    SnapshotClientReceiver,
    SnapshotInterpolationBuffer,
    SnapshotReplicator,
    StatelessCookieAuth,
    UdpSocket,
    VirtualEndpoint,
    VirtualNetSim,
    VisualSmoother,
    aead_open,
    aead_seal,
    quat_angular_error_deg,
)
from flint.net.rollback import PhysicsRollbackAdapter, PhysicsRollbackInput


# ===========================================================================
# Shared Test Types & System Callbacks (FA2)
# ===========================================================================

@fieldwise_init
struct T1Pos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t1.T1Pos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct T1Vel(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t1.T1Vel")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var vx: Float32
    var vy: Float32
    var vz: Float32


@fieldwise_init
struct T1Health(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t1.T1Health")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var hp: Float32


@fieldwise_init
struct T1SparseTag(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t1.T1SparseTag")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var code: Int32


@fieldwise_init
struct T1Event(Copyable, ImplicitlyCopyable, Movable):
    var entity: Entity
    var value: Int32


@fieldwise_init
struct T1Loot(SerializableComponent, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t1.T1Loot")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "T1Loot"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("T1LT")
    comptime SCHEMA_VERSION: Int = 1

    var gold: Int32
    var rarity: Float32

    def write_bin(self, mut out: BlobWriter):
        out.u32(UInt32(self.gold))
        out.f32(self.rarity)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var g = Int32(reader.u32())
        var r = reader.f32()
        return Self(g, r)

    def write_json(self) -> String:
        return "{\"gold\": " + String(self.gold) + ", \"rarity\": " + String(Float64(self.rarity)) + "}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var g = Int32(doc.integer(doc.object_get(node, "gold"), 0))
        var r = Float32(doc.number(doc.object_get(node, "rarity"), 0.0))
        return Self(g, r)


@fieldwise_init
struct T1AppCounters(Resource, Copyable, ImplicitlyCopyable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("e2e_opaque.t1.T1AppCounters")
    var startup_calls: Int
    var fixed_calls: Int
    var update_calls: Int
    var post_calls: Int
    var shutdown_calls: Int
    var swap_calls: Int


def t1_sys_startup(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var opt = r[unsafe_offset=0].get_ptr_mut[T1AppCounters]()
    if opt:
        opt.value()[unsafe_offset=0].startup_calls += 1


def t1_sys_fixed(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var opt = r[unsafe_offset=0].get_ptr_mut[T1AppCounters]()
    if opt:
        opt.value()[unsafe_offset=0].fixed_calls += 1


def t1_sys_update(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var opt = r[unsafe_offset=0].get_ptr_mut[T1AppCounters]()
    if opt:
        opt.value()[unsafe_offset=0].update_calls += 1


def t1_sys_post(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var opt = r[unsafe_offset=0].get_ptr_mut[T1AppCounters]()
    if opt:
        opt.value()[unsafe_offset=0].post_calls += 1


def t1_sys_shutdown(
    _w: Pointer[World, MutAnyOrigin],
    r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var opt = r[unsafe_offset=0].get_ptr_mut[T1AppCounters]()
    if opt:
        opt.value()[unsafe_offset=0].shutdown_calls += 1


def t1_event_swap_hook(r: Pointer[Resources, MutAnyOrigin]):
    var opt = r[unsafe_offset=0].get_ptr_mut[T1AppCounters]()
    if opt:
        opt.value()[unsafe_offset=0].swap_calls += 1


# ===========================================================================
# FA2 Tier 1 Tests (8 Happy-Path Tests)
# ===========================================================================

def test_fa2_t1_engine_app_lifecycle_and_fixed_timestep() raises:
    var cfg = AppConfig(
        title="FA2_T1_Lifecycle",
        headless=True,
        worker_threads=2,
        fixed_hz=60.0,
        max_fixed_steps_per_frame=8,
        default_frame_dt=Float32(1.0 / 60.0),
    )
    var app = EngineApp(cfg^)
    app.insert_resource[T1AppCounters](T1AppCounters(0, 0, 0, 0, 0, 0))
    assert_true(app.register_component[T1Pos]() >= 0)
    assert_true(app.add_system(STAGE_STARTUP, "t1.startup", t1_sys_startup) >= 0)
    assert_true(app.add_system(STAGE_FIXED_UPDATE, "t1.fixed", t1_sys_fixed) >= 0)
    assert_true(app.add_system(STAGE_UPDATE, "t1.update", t1_sys_update) >= 0)
    assert_true(app.add_system(STAGE_POST_UPDATE, "t1.post", t1_sys_post) >= 0)
    assert_true(app.add_system(STAGE_SHUTDOWN, "t1.shutdown", t1_sys_shutdown) >= 0)
    app.add_event_swap_hook(t1_event_swap_hook)

    app.run(max_frames=10)

    var ft = app.resources.get_ptr[FrameTime]().value()[unsafe_offset=0]
    var fx = app.resources.get_ptr[FixedTime]().value()[unsafe_offset=0]
    var cnt = app.resources.get_ptr[T1AppCounters]().value()[unsafe_offset=0]
    assert_equal(Int(ft.frame_index), 10)
    assert_equal(Int(fx.tick_count), 10)
    assert_equal(cnt.startup_calls, 1)
    assert_equal(cnt.fixed_calls, 10)
    assert_equal(cnt.update_calls, 10)
    assert_equal(cnt.post_calls, 10)
    assert_equal(cnt.swap_calls, 10)
    assert_equal(cnt.shutdown_calls, 1)


def test_fa2_t1_ecs_table_sparse_changed_and_events() raises:
    var w = World()
    var e1 = w.spawn_empty()
    var e2 = w.spawn_empty()
    var e1_orig = e1

    assert_true(w.insert[T1Pos](e1, T1Pos(1.0, 2.0, 3.0)))
    assert_true(w.insert[T1Vel](e1, T1Vel(4.0, 5.0, 6.0)))
    assert_true(w.insert[T1Health](e1, T1Health(100.0)))
    assert_true(w.insert[T1SparseTag](e1, T1SparseTag(77)))
    assert_true(e1 == e1_orig and w.is_alive(e1))

    assert_true(w.insert[T1Pos](e2, T1Pos(9.0, 8.0, 7.0)))
    assert_true(w.remove[T1Vel](e1))
    assert_true(e1 == e1_orig and w.has[T1Pos](e1) and not w.has[T1Vel](e1))
    assert_true(w.has[T1SparseTag](e1))
    assert_equal(Int(w.get[T1SparseTag](e1).code), 77)

    var with_mask = w.bit_for[T1Pos]().union(w.bit_for[T1SparseTag]())
    var without_mask = w.bit_for[T1Vel]()
    var q = w.query_entities(with_mask, without_mask)
    assert_equal(len(q), 1)
    assert_true(q[0] == e1)

    # Changed[C] query after tick advance + mutable borrow
    var prev_tick = w.current_tick
    _ = w.advance_tick()
    var mp = w.get_ptr_mut[T1Pos](e2)
    assert_true(Bool(mp))
    mp.value()[unsafe_offset=0].x = Float32(42.0)
    var changed = w.query_changed_entities[T1Pos](prev_tick)
    var found_e2 = False
    for i in range(len(changed)):
        if changed[i] == e2:
            found_e2 = True
    assert_true(found_e2)

    # Events[T] 2-frame retention across swap_buffers
    var evs = Events[T1Event]()
    var c_fast = 0
    var c_late = 0
    _ = evs.send(T1Event(e1, 10))
    _ = evs.send(T1Event(e2, 20))
    assert_equal(len(evs.read_since(c_fast)), 2)
    evs.swap_buffers()
    _ = evs.send(T1Event(e1, 30))
    assert_equal(len(evs.read_since(c_late)), 3)
    assert_equal(len(evs.read_since(c_fast)), 1)
    evs.swap_buffers()
    var c_very_late = 0
    var rem = evs.read_since(c_very_late)
    assert_equal(len(rem), 1)
    assert_equal(Int(rem[0].value), 30)


@fieldwise_init
struct T1ParCtx(Copyable, ImplicitlyCopyable, Movable):
    var cmds_addr: Int
    var step: Int


def t1_par_step_kernel(
    lane: Int,
    order: Int,
    entity: Entity,
    pos: Pointer[T1Pos, MutAnyOrigin],
    vel: Pointer[T1Vel, MutAnyOrigin],
    ctx_addr: Int,
):
    ref ctx = Pointer[T1ParCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref cmds = Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=ctx.cmds_addr)[unsafe_offset=0]

    var phase = Float32(order) * Float32(0.03) + Float32(ctx.step) * Float32(0.1)
    vel[unsafe_offset=0].vx += det_sin(phase) * Float32(0.1)
    pos[unsafe_offset=0].x += vel[unsafe_offset=0].vx * Float32(1.0 / 60.0)
    pos[unsafe_offset=0].y += vel[unsafe_offset=0].vy * Float32(1.0 / 60.0)

    if (order + ctx.step) % 31 == 0:
        var spawned = cmds.spawn_empty(lane, order_key=order)
        cmds.insert[T1Pos](lane, spawned, T1Pos(pos[unsafe_offset=0].x, Float32(ctx.step), 0.0), order_key=order)
        cmds.insert[T1Vel](lane, spawned, T1Vel(1.0, -0.5, 0.25), order_key=order)
    elif (order + ctx.step) % 53 == 0 and ctx.step >= 2:
        cmds.despawn(lane, entity, order_key=order)


def t1_run_par_sim(workers: Int) -> Tuple[Int, UInt64]:
    var w = World()
    _ = w.register_component[T1Pos]()
    _ = w.register_component[T1Vel]()
    for i in range(256):
        var e = w.spawn_empty()
        _ = w.insert[T1Pos](e, T1Pos(Float32(i) * 0.5, Float32(i % 13), -1.0))
        _ = w.insert[T1Vel](e, T1Vel(1.25, -0.75, 0.5))

    var pool = JobPool()
    if workers > 0:
        _ = pool.start(workers)
    else:
        _ = pool.allocate()

    var cmds = ParallelCommands()
    for step in range(6):
        _ = w.advance_tick()
        var ctx_box = List[T1ParCtx](capacity=1)
        ctx_box.append(T1ParCtx(Int(Pointer(to=cmds)), step))
        w.par_for_each2[T1Pos, T1Vel](pool, t1_par_step_kernel, Int(ctx_box.unsafe_ptr()))
        var keep_ctx = len(ctx_box)
        _ = keep_ctx
        w.flush_commands(cmds)

    var cnt = w.entity_count()
    var cs = w.state_checksum()
    pool.stop()
    return (cnt, cs)


def test_fa2_t1_parallel_commands_determinism() raises:
    var res_0 = t1_run_par_sim(0)
    var res_4 = t1_run_par_sim(4)
    assert_true(res_0[0] > 256)
    assert_equal(res_0[0], res_4[0])
    assert_equal(res_0[1], res_4[1])


def test_fa2_t1_transform_plugin_hierarchy_and_interpolation() raises:
    var cfg = AppConfig(
        title="FA2_T1_Transform",
        headless=True,
        worker_threads=2,
        fixed_hz=60.0,
        default_frame_dt=Float32(1.0 / 60.0),
    )
    var app = EngineApp(cfg^)
    app.add_plugin(TransformPlugin())

    var root = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](root, LocalTransform(Vec3(10.0, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
    _ = app.world.insert[GlobalTransform](root, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](root, PrevGlobalTransform())

    var child0 = Entity()
    var num_children = INLINE_CHILD_CAP + 2  # 6 children > inline 4
    for i in range(num_children):
        var c = app.world.spawn_empty()
        _ = app.world.insert[LocalTransform](c, LocalTransform(Vec3(Float32(i + 1), 2.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
        _ = app.world.insert[GlobalTransform](c, GlobalTransform())
        assert_true(set_parent(app.world, c, root))
        if i == 0:
            child0 = c

    var grand = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](grand, LocalTransform(Vec3(0.0, 0.0, 5.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
    _ = app.world.insert[GlobalTransform](grand, GlobalTransform())
    assert_true(set_parent(app.world, grand, child0))

    app.step_one_frame(Float32(1.0 / 60.0))

    assert_equal(app.world.get[Children](root).len(), num_children)
    assert_equal(Int(app.world.get[HierarchyDepth](grand).depth), 2)
    var g_pos = app.world.get[GlobalTransform](grand).translation
    assert_true(abs(g_pos.x - Float32(11.0)) < Float32(1e-4))
    assert_true(abs(g_pos.y - Float32(2.0)) < Float32(1e-4))
    assert_true(abs(g_pos.z - Float32(5.0)) < Float32(1e-4))
    app.shutdown()

    # Interpolation check
    var p_prev = GlobalTransform.from_local(LocalTransform(Vec3(0.0, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
    var p_curr = GlobalTransform.from_local(
        LocalTransform(Vec3(10.0, 4.0, -2.0), Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), PI * Float32(0.5)), Vec3(3.0, 3.0, 3.0))
    )
    var mid = interpolate_global_transform(p_prev, p_curr, Float32(0.5))
    assert_true(abs(mid.translation.x - Float32(5.0)) < Float32(1e-4))
    assert_true(abs(mid.translation.y - Float32(2.0)) < Float32(1e-4))
    assert_true(abs(mid.scale.x - Float32(2.0)) < Float32(1e-4))


def test_fa2_t1_scene_prefab_roundtrip_and_pie() raises:
    var reg = ComponentRegistry.with_builtins()
    reg.register[T1Loot]()

    var child_prefab = Prefab("T1Crate")
    var c_root = child_prefab.add_node("CrateRoot", -1)
    child_prefab.set_component[LocalTransform](c_root, LocalTransform(Vec3(1.0, 2.0, 3.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
    child_prefab.set_component[T1Loot](c_root, T1Loot(10, 0.25))

    var lib = List[Prefab]()
    lib.append(child_prefab^)

    var parent_prefab = Prefab("T1Depot")
    var p_base = parent_prefab.add_node("DepotBase", -1)
    parent_prefab.set_component[LocalTransform](p_base, LocalTransform(Vec3(10.0, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
    parent_prefab.set_component[T1Loot](p_base, T1Loot(50, 0.5))

    var nested = NestedPrefabRef("T1Crate", attach_parent_slot=p_base, local_offset=LocalTransform(Vec3(2.0, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
    nested.add_override(PrefabOverride.for_name[T1Loot]("CrateRoot", T1Loot(999, 0.95)))
    parent_prefab.add_nested(nested^)

    var w = World()
    var ents = parent_prefab.instantiate(w, reg, lib)
    assert_equal(len(ents), 2)
    assert_equal(Int(w.get[T1Loot](ents[1]).gold), 999)

    var scene = Scene("T1VaultScene")
    var json_text = scene.save_flscn_string(w, reg)
    var json_bytes = List[UInt8](capacity=json_text.byte_length())
    for b in json_text.as_bytes():
        json_bytes.append(b)
    var bin_bytes = scene.save_flscnb_bytes(w, reg)

    var w_json = World()
    var s_json = Scene()
    var ej = s_json.load_flscn_bytes(w_json, reg, json_bytes)
    assert_equal(len(ej), 2)
    assert_equal(s_json.name, "T1VaultScene")
    assert_equal(Int(w_json.get[T1Loot](ej[1]).gold), 999)

    var w_bin = World()
    var s_bin = Scene()
    var eb = s_bin.load_flscnb_bytes(w_bin, reg, bin_bytes)
    assert_equal(len(eb), 2)
    assert_equal(s_bin.name, "T1VaultScene")
    assert_equal(Int(w_bin.get[T1Loot](eb[1]).gold), 999)

    # PIE snapshot / mutate / restore
    var pre_cs = w.state_checksum()
    scene.enter_pie(w)
    _ = w.insert[T1Loot](ents[1], T1Loot(1, 0.0))
    _ = w.despawn(ents[0])
    assert_true(w.state_checksum() != pre_cs)
    scene.exit_pie(w)
    assert_equal(w.state_checksum(), pre_cs)


def test_fa2_t1_detmath_scalar_identities_and_determinism() raises:
    var angles = List[Float32]()
    angles.append(Float32(0.0))
    angles.append(Float32(0.5))
    angles.append(Float32(-1.25))
    angles.append(Float32(2.75))
    for i in range(len(angles)):
        var a = angles[i]
        var s = det_sin(a)
        var c = det_cos(a)
        assert_true(abs((s * s + c * c) - Float32(1.0)) < Float32(1e-5))
        assert_equal(f32_bits(s), f32_bits(det_sin(a)))
        assert_equal(f32_bits(c), f32_bits(det_cos(a)))

    assert_true(abs(det_atan(Float32(1.0)) - PI * Float32(0.25)) < Float32(1e-5))
    assert_true(abs(det_atan2(Float32(1.0), Float32(1.0)) - PI * Float32(0.25)) < Float32(1e-5))
    assert_true(abs(det_asin(Float32(0.5)) - PI / Float32(6.0)) < Float32(1e-5))
    assert_true(abs(det_acos(Float32(0.5)) - PI / Float32(3.0)) < Float32(1e-5))

    var x = Float32(2.75)
    assert_true(abs(det_exp(det_log(x)) - x) < Float32(1e-5))
    assert_true(abs(det_pow(Float32(2.5), Float32(3.0)) - Float32(15.625)) < Float32(1e-5))
    assert_equal(scale_by_pow2(Float32(3.0), 4), Float32(48.0))


def t1_dag_step_a(ctx: Int, _chunk: Int):
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].store(1)


def t1_dag_step_b(ctx: Int, _chunk: Int):
    var a_val = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].load()
    if a_val == 1:
        _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].fetch_add(10)


def t1_dag_step_c(ctx: Int, _chunk: Int):
    var a_val = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].load()
    if a_val == 1:
        _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].fetch_add(20)


def t1_dag_step_d(ctx: Int, _chunk: Int):
    var bc = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].load()
    if bc == 30:
        Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 16)[unsafe_offset=0].store(99)


def t1_chain_or_affine_body(ctx: Int, chunk: Int):
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].fetch_add(Int64(chunk + 1))


def test_fa2_t1_jobpool_dag_chain_and_affine() raises:
    var pool = JobPool()
    assert_true(pool.start(4))
    assert_true(pool.ready())

    var state = List[Int64](length=3, fill=0)
    var s_addr = Int(state.unsafe_ptr())

    # Diamond DAG: A -> B, C -> D
    var j_a = pool.create(t1_dag_step_a, s_addr, 1)
    var j_b = pool.create(t1_dag_step_b, s_addr, 1)
    var j_c = pool.create(t1_dag_step_c, s_addr, 1)
    var j_d = pool.create(t1_dag_step_d, s_addr, 1)
    pool.depend(j_b, j_a)
    pool.depend(j_c, j_a)
    pool.depend(j_d, j_b)
    pool.depend(j_d, j_c)
    pool.submit(j_b)
    pool.submit(j_c)
    pool.submit(j_d)
    pool.submit(j_a)
    pool.wait(j_d)
    assert_equal(Int(state[0]), 1)
    assert_equal(Int(state[1]), 30)
    assert_equal(Int(state[2]), 99)
    _ = len(state)

    # Chain of 6 single-chunk (chunk=0) jobs -> sum of 6 * (0 + 1) = 6
    var sum_box = List[Int64](length=1, fill=0)
    var last_chain = pool.chain(t1_chain_or_affine_body, Int(sum_box.unsafe_ptr()), 6)
    pool.wait(last_chain)
    assert_equal(Int(sum_box[0]), 6)

    # run_affine + run_jobs_affine
    sum_box[0] = 0
    pool.run_affine(t1_chain_or_affine_body, Int(sum_box.unsafe_ptr()), 8)
    assert_equal(Int(sum_box[0]), 36)

    sum_box[0] = 0
    run_jobs_affine(Int(Pointer(to=pool)), t1_chain_or_affine_body, Int(sum_box.unsafe_ptr()), 5)
    assert_equal(Int(sum_box[0]), 15)
    _ = len(sum_box)

    pool.stop()


def test_fa2_t1_memory_arena_and_frame_arenas() raises:
    var arena = Arena(MEM_FRAME, 4096)
    var p16 = arena.alloc(24, 16)
    var p64 = arena.alloc(32, 64)
    assert_true(p16 != 0 and (p16 & 15) == 0)
    assert_true(p64 != 0 and (p64 & 63) == 0)
    assert_true(arena.owns(p16) and arena.owns(p64))
    assert_equal(arena.overflows, 0)
    arena.reset()
    assert_equal(arena.used, 0)
    var p_after = arena.alloc(16, 16)
    assert_true(p_after != 0 and arena.owns(p_after))

    var frames = FrameArenas(3, 1024)
    assert_equal(frames.overflows(), 0)
    assert_true(frames.capacity() >= 3072)
    frames.begin(0)
    var f0 = frames.alloc(64, 16)
    assert_true(f0 != 0 and frames.used() >= 64)
    var next_s = frames.next_slot()
    assert_equal(next_s, 1)
    frames.begin(next_s)
    assert_equal(frames.used(), 0)
    assert_true(frames.peak() >= 64)


# ===========================================================================
# FA7 Tier 1 Tests (8 Happy-Path Tests)
# ===========================================================================

def t1_make_payload(tag: Int, size: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=size)
    for i in range(size):
        out.append(UInt8((tag * 37 + i * 13 + 5) & 0xFF))
    return out^


def test_fa7_t1_udp_dual_stack_loopback_burst() raises:
    var srv = UdpSocket.open_dual_stack(0)
    assert_true(srv.is_open() and srv.local_port() > 0)

    var cli6 = UdpSocket()
    var cli4 = UdpSocket()
    assert_true(cli6.bind(NetEndpoint.ipv6_loopback(0)))
    assert_true(cli4.bind(NetEndpoint.ipv4_loopback(0)))

    var srv_v6 = NetEndpoint.ipv6_loopback(srv.local_port())
    var srv_v4 = NetEndpoint.ipv4_loopback(srv.local_port())

    for i in range(16):
        var p6 = List[UInt8]()
        p6.append(UInt8(0x66))
        p6.append(UInt8(i))
        assert_true(cli6.queue_send_bytes(srv_v6, p6))

        var p4 = List[UInt8]()
        p4.append(UInt8(0x44))
        p4.append(UInt8(i))
        assert_true(cli4.queue_send_bytes(srv_v4, p4))

    assert_equal(cli6.flush_sends(), 16)
    assert_equal(cli4.flush_sends(), 16)

    var views = List[DatagramView](capacity=64)
    var got6 = 0
    var got4 = 0
    for _ in range(60):
        var n = srv.recv_batch(views)
        for i in range(n):
            var v = views[i]
            assert_equal(v.length, 2)
            if v.byte_at(0) == UInt8(0x66):
                assert_true(v.sender.is_ipv6_loopback())
                assert_true(v.sender == cli6.local_endpoint())
                got6 += 1
            elif v.byte_at(0) == UInt8(0x44):
                assert_true(v.sender.is_ipv4_mapped())
                assert_true(v.sender == cli4.local_endpoint())
                got4 += 1
            assert_true(srv.queue_send_raw(v.sender, v.data_ptr, v.length))
        if got6 == 16 and got4 == 16:
            break

    assert_equal(got6, 16)
    assert_equal(got4, 16)
    assert_equal(srv.flush_sends(), 32)

    var echo6 = 0
    var echo4 = 0
    for _ in range(60):
        echo6 += cli6.recv_batch(views)
        echo4 += cli4.recv_batch(views)
        if echo6 == 16 and echo4 == 16:
            break
    assert_equal(echo6, 16)
    assert_equal(echo4, 16)


def test_fa7_t1_bitpack_roundtrip_and_quat_accuracy() raises:
    var w = BitWriter(8192)
    var ints = List[Int](capacity=200)
    var vecs = List[Vec3](capacity=200)
    var quats = List[Quat](capacity=200)

    for i in range(200):
        var iv = -500 + (i * 7) % 1001
        ints.append(iv)
        w.write_int_bounded(iv, -500, 500)
        w.write_bool((i & 1) == 0)

        var v = Vec3(Float32(i % 50) - 25.0, Float32((i * 3) % 80) - 40.0, Float32((i * 7) % 60) - 30.0)
        vecs.append(v)
        w.write_vec3_quantized(v, -128.0, 128.0, 16)

        var q = Quat.from_euler(Float32(i) * 0.03, Float32(i) * 0.07, Float32(i) * -0.02).normalized()
        quats.append(q)
        w.write_quat_smallest_three(q)
        w.write_quat_smallest_three_48(q)

    w.flush_align()
    var raw_tail = t1_make_payload(7, 19)
    w.write_bytes_aligned(raw_tail)
    var packed = w.finish()

    var r = BitReader(packed)
    var max_err32 = Float32(0.0)
    var max_err48 = Float32(0.0)
    for i in range(200):
        assert_equal(r.read_int_bounded(-500, 500), ints[i])
        assert_equal(r.read_bool(), (i & 1) == 0)
        var dv = r.read_vec3_quantized(-128.0, 128.0, 16)
        assert_true((dv - vecs[i]).length() < Float32(0.01))
        var q32 = r.read_quat_smallest_three()
        var q48 = r.read_quat_smallest_three_48()
        var e32 = quat_angular_error_deg(quats[i], q32)
        var e48 = quat_angular_error_deg(quats[i], q48)
        if e32 > max_err32:
            max_err32 = e32
        if e48 > max_err48:
            max_err48 = e48

    r.align_to_byte()
    var dec_tail = r.read_bytes_aligned(len(raw_tail))
    assert_false(r.overflowed)
    assert_equal(r.remaining_bits(), 0)
    assert_equal(hash_list(dec_tail), hash_list(raw_tail))
    assert_true(max_err32 < Float32(0.10))
    assert_true(max_err48 < Float32(0.005))


def test_fa7_t1_aead_and_replay_window() raises:
    var key = AeadKey.from_u64x4(0x1020304050607080, 0x1122334455667788, 0x99AABBCCDDEEFF00, 0x0102030405060708)
    var aad = t1_make_payload(3, 18)
    var pt = t1_make_payload(9, 137)
    var ct_tag = List[UInt8]()
    aead_seal(key, UInt64(1), aad, pt, ct_tag)
    assert_equal(len(ct_tag), len(pt) + AEAD_TAG_BYTES)

    var out_pt = List[UInt8]()
    assert_true(aead_open(key, UInt64(1), aad, ct_tag, out_pt))
    assert_equal(hash_list(out_pt), hash_list(pt))

    var rw = ReplayWindow64()
    var nonces = List[UInt64]()
    nonces.append(1)
    nonces.append(3)
    nonces.append(2)
    nonces.append(65)
    for i in range(len(nonces)):
        var n = nonces[i]
        assert_true(rw.would_accept(n))
        rw.record(n)

    var cookie_auth = StatelessCookieAuth(key, bucket_ms=10000.0)
    var ep = NetEndpoint.ipv4_loopback(9450)
    var cookie = cookie_auth.issue_cookie(ep, 1000.0)
    var c_ptr = Int(Pointer(to=cookie))
    assert_true(cookie_auth.verify_cookie(ep, c_ptr, 1500.0))
    _ = cookie[0]


def test_fa7_t1_net_channel_reliable_fragment_and_sequenced() raises:
    var sim = VirtualNetSim(
        seed=0xFA71C4A1,
        one_way_latency_ms=25.0,
        jitter_ms=10.0,
        loss_rate=0.05,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var ep0 = sim.create_ipv4_peer(0, 9100)
    var ep1 = sim.create_ipv4_peer(1, 9100)

    var key = AeadKey.from_u64x4(0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD)
    var ch0 = NetChannel(ep1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var ch1 = NetChannel(ep0.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    ch0.enable_encryption(key)
    ch1.enable_encryption(key)

    var msg_4k = t1_make_payload(11, 4096)
    var msg_unord = t1_make_payload(22, 256)
    assert_true(ch0.send_reliable(msg_4k))
    assert_true(ch0.send_reliable_unordered(msg_unord))

    var got_4k = False
    var got_unord = False
    var last_seq = -1
    var seq_monotonic = True

    for step in range(400):
        var now = sim.now_ms()
        if step < 20:
            var u = List[UInt8]()
            u.append(UInt8(step))
            _ = ch0.send_unreliable_sequenced(u, ep0, now)
        ch0.poll(ep0, now)
        ch1.poll(ep1, now)

        while True:
            var up = ch1.pop_unreliable_message()
            if not up[0]:
                break
            var val = Int(up[1][0])
            if val <= last_seq:
                seq_monotonic = False
            last_seq = val

        while True:
            var rp = ch1.pop_reliable_message()
            if not rp[0]:
                break
            assert_equal(hash_list(rp[1]), hash_list(msg_4k))
            got_4k = True

        while True:
            var rup = ch1.pop_reliable_unordered_message()
            if not rup[0]:
                break
            assert_equal(hash_list(rup[1]), hash_list(msg_unord))
            got_unord = True

        if got_4k and got_unord and ch0.is_all_reliable_acked():
            break
        sim.advance_ms(10.0)

    assert_true(sim.now_ms() >= 0.0)
    assert_equal(sim.packets_dropped_mtu(), 0)
    assert_true(got_4k and got_unord)
    assert_true(seq_monotonic)
    assert_true(ch0.is_all_reliable_acked())
    assert_true(ch0.rtt_initialized)


def test_fa7_t1_net_hub_cookie_handshake_and_multi_client() raises:
    var sim = VirtualNetSim(seed=0xFA714B, one_way_latency_ms=15.0, jitter_ms=3.0, loss_rate=0.02)
    var host = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9200), True)
    var cookie_key = AeadKey.from_u64x4(0x11, 0x22, 0x33, 0x44)
    var enc_key = AeadKey.from_u64x4(0x55, 0x66, 0x77, 0x88)
    host.enable_cookie_handshake(cookie_key, bucket_ms=10000.0)
    host.enable_channel_encryption(enc_key)

    var host_ep = host.local_endpoint()
    var clients = List[NetHub[VirtualEndpoint]]()
    for c in range(3):
        var cli = NetHub[VirtualEndpoint](sim.create_ipv4_peer(c + 1, 9200), False)
        cli.enable_channel_encryption(enc_key)
        assert_true(cli.send_handshake_init(host_ep))
        clients.append(cli^)

    for _ in range(120):
        var now = sim.now_ms()
        host.poll(now)
        for c in range(3):
            clients[c].poll(now)
        if host.open_count() == 3:
            break
        sim.advance_ms(10.0)

    assert_equal(host.open_count(), 3)
    assert_true(host.cookies_sent >= 3)

    for c in range(3):
        var cid = clients[c].connect(host_ep)
        assert_equal(cid, 0)
        var sz = 128 if c < 2 else 2400  # 3rd client sends > 2 KiB fragmented payload
        assert_true(clients[c].send_reliable(cid, t1_make_payload(c + 10, sz)))

    var replied = 0
    for _ in range(200):
        var now = sim.now_ms()
        host.poll(now)
        for cid in range(host.count):
            while True:
                var m = host.pop_reliable(cid)
                if not m[0]:
                    break
                var client_tag = 2 if len(m[1]) == 2400 else (0 if m[1][0] == t1_make_payload(10, 128)[0] else 1)
                assert_equal(hash_list(m[1]), hash_list(t1_make_payload(client_tag + 10, len(m[1]))))
                assert_true(host.send_reliable(cid, t1_make_payload(100 + client_tag, 48)))
                replied += 1
        for c in range(3):
            clients[c].poll(now)
        if replied == 3:
            break
        sim.advance_ms(10.0)

    assert_equal(replied, 3)
    for _ in range(80):
        var now = sim.now_ms()
        host.poll(now)
        for c in range(3):
            clients[c].poll(now)
        sim.advance_ms(10.0)

    for c in range(3):
        var r = clients[c].pop_reliable(0)
        assert_true(r[0])
        assert_equal(hash_list(r[1]), hash_list(t1_make_payload(100 + c, 48)))
    assert_true(sim.now_ms() >= 0.0)


def test_fa7_t1_replicator_aoi_despawn_and_interpolation() raises:
    var rep = SnapshotReplicator(
        max_entities=32,
        max_clients=2,
        max_packet_bytes=900,
        interest_radius=60.0,
        cell_size=16.0,
        use_3d_grid=True,
        world_bounds=256.0,
    )
    var cli0 = SnapshotClientReceiver(32, 256.0)
    var cli1 = SnapshotClientReceiver(32, 256.0)

    rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 60.0)
    rep.set_client_view(1, Vec3(200.0, 0.0, 0.0), 60.0)

    rep.set_entity(0, Vec3(10.0, 2.0, -5.0), Quat.identity(), Vec3(1.0, 0.0, 0.0), UInt32(1))
    rep.set_entity(1, Vec3(20.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(2))
    rep.set_entity(2, Vec3(205.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(3))  # only in cli1 AOI

    rep.prepare_tick()
    var p0 = rep.build_client_packet(0)
    var p1 = rep.build_client_packet(1)
    var a0 = cli0.apply_packet(p0[1])
    var a1 = cli1.apply_packet(p1[1])
    assert_true(a0[0] and a1[0])
    rep.notify_client_ack(0, a0[1])
    rep.notify_client_ack(1, a1[1])

    assert_true(cli0.entities[0].active and cli0.entities[1].active)
    assert_false(cli0.entities[2].active)
    assert_true(cli1.entities[2].active and not cli1.entities[0].active)
    assert_true((cli0.entities[0].pos - Vec3(10.0, 2.0, -5.0)).length() < Float32(0.05))

    # Despawn entity 0 and move entity 1 outside cli0 AOI
    rep.destroy_entity(0)
    rep.set_entity(1, Vec3(150.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(2))
    rep.prepare_tick()
    var p0_next = rep.build_client_packet(0)
    var a0_next = cli0.apply_packet(p0_next[1])
    assert_true(a0_next[0])
    assert_false(cli0.entities[0].active)
    assert_false(cli0.entities[1].active)

    # SnapshotInterpolationBuffer
    var interp = SnapshotInterpolationBuffer(max_entities=4, interp_delay_ms=100.0)
    var sa = List[ReplicatedEntity]()
    var sb = List[ReplicatedEntity]()
    for i in range(4):
        sa.append(ReplicatedEntity(i))
        sb.append(ReplicatedEntity(i))
    sa[0].active = True
    sa[0].pos = Vec3(0.0, 0.0, 0.0)
    sa[0].rot = Quat.identity()
    sb[0].active = True
    sb[0].pos = Vec3(10.0, 6.0, -4.0)
    sb[0].rot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), PI * Float32(0.5))
    interp.push_snapshot(100.0, sa)
    interp.push_snapshot(200.0, sb)

    var sampled = List[ReplicatedEntity]()
    assert_true(interp.sample(250.0, sampled))
    assert_true(sampled[0].active)
    assert_true(abs(sampled[0].pos.x - Float32(5.0)) < Float32(0.01))
    assert_true(abs(sampled[0].pos.y - Float32(3.0)) < Float32(0.01))
    assert_true(quat_angular_error_deg(sampled[0].rot, Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), PI * Float32(0.25))) < Float32(0.2))


def test_fa7_t1_rollback_physics_arena_and_visual_smoother() raises:
    var sim = VirtualNetSim.from_rtt(rtt_ms=40.0, jitter_ms=10.0, loss_rate=0.05, seed=0xFA718B)
    var ep0 = sim.create_ipv4_peer(0, 9300)
    var ep1 = sim.create_ipv4_peer(1, 9300)
    var addrs = List[NetEndpoint]()
    addrs.append(ep0.local_endpoint())
    addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)

    var s0 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=0,
        peer_endpoints=addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
    )
    var s1 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=1,
        peer_endpoints=addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
    )
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    for frame in range(220):
        var now = sim.now_ms()
        var in0 = PhysicsRollbackInput(Float32((frame % 7) - 3) * 2.0, 0.0, 4.0, 0.5, 1 if frame % 5 == 0 else 0)
        var in1 = PhysicsRollbackInput(-2.0, 0.0, Float32((frame % 5) - 2) * 2.0, -0.25, 0)
        _ = s0.step_frame(in0, pw0, ep0, now)
        _ = s1.step_frame(in1, pw1, ep1, now)
        if len(s0.confirmed_hash_history) >= 120 and len(s1.confirmed_hash_history) >= 120:
            break
        sim.advance_ms(1000.0 / 60.0)

    assert_true(sim.now_ms() >= 0.0)
    assert_false(s0.desync_detected)
    assert_false(s1.desync_detected)
    assert_true(len(s0.confirmed_hash_history) >= 120)
    assert_true(len(s1.confirmed_hash_history) >= 120)
    for t in range(120):
        assert_equal(s0.confirmed_hash_history[t], s1.confirmed_hash_history[t])
    assert_true(s0.rollback_count + s1.rollback_count > 0)

    var slot0 = pw0.world.find_slot_by_id(pw0.player_body_ids[0])
    assert_true(slot0 >= 0)
    assert_true((pw0.world.bodies[slot0].pos - Vec3(5.0, 1.0, 0.0)).length() > Float32(0.1))

    # VisualSmoother decay check
    var sm = VisualSmoother(decay_rate=18.0)
    sm.record_correction(Vec3(0.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    sm.step(Float32(0.05))
    var p1 = sm.smooth_pos(Vec3(4.0, 0.0, 0.0))
    assert_true(p1.x > Float32(0.1) and p1.x < Float32(3.9))
    sm.step(Float32(0.05))
    var p2 = sm.smooth_pos(Vec3(4.0, 0.0, 0.0))
    assert_true(p2.x > p1.x and p2.x <= Float32(4.0))


def test_fa7_t1_virtual_sim_blackout_and_k8_survival() raises:
    var sim = VirtualNetSim.from_rtt(rtt_ms=20.0, jitter_ms=0.0, loss_rate=0.0, seed=0x484848)
    var ep0 = sim.create_ipv4_peer(0, 9400)
    var ep1 = sim.create_ipv4_peer(1, 9400)
    var addrs = List[NetEndpoint]()
    addrs.append(ep0.local_endpoint())
    addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s0 = RollbackSession[PhysicsRollbackInput](0, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    var s1 = RollbackSession[PhysicsRollbackInput](1, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    for f in range(100):
        if f == 15:
            sim.drop_next_outbound_packets(0, 7)
        var now = sim.now_ms()
        var c0 = PhysicsRollbackInput(Float32((f % 4) + 1), 0.0, 2.0, 0.1, 1)
        var c1 = PhysicsRollbackInput(-1.5, 0.0, Float32(f % 3), -0.1, 0)
        _ = s0.step_frame(c0, pw0, ep0, now)
        _ = s1.step_frame(c1, pw1, ep1, now)
        sim.advance_ms(1000.0 / 60.0)

    assert_true(sim.now_ms() >= 0.0)
    assert_false(s0.desync_detected)
    assert_false(s1.desync_detected)
    assert_true(s0.confirmed_tick >= 50 and s1.confirmed_tick >= 50)
    for t in range(50):
        assert_equal(s0.confirmed_hash_history[t], s1.confirmed_hash_history[t])


def main() raises:
    set_sim_float_mode()
    var passed = 0
    var total = 16

    test_fa2_t1_engine_app_lifecycle_and_fixed_timestep()
    print("[PASS] test_fa2_t1_engine_app_lifecycle_and_fixed_timestep")
    passed += 1

    test_fa2_t1_ecs_table_sparse_changed_and_events()
    print("[PASS] test_fa2_t1_ecs_table_sparse_changed_and_events")
    passed += 1

    test_fa2_t1_parallel_commands_determinism()
    print("[PASS] test_fa2_t1_parallel_commands_determinism")
    passed += 1

    test_fa2_t1_transform_plugin_hierarchy_and_interpolation()
    print("[PASS] test_fa2_t1_transform_plugin_hierarchy_and_interpolation")
    passed += 1

    test_fa2_t1_scene_prefab_roundtrip_and_pie()
    print("[PASS] test_fa2_t1_scene_prefab_roundtrip_and_pie")
    passed += 1

    test_fa2_t1_detmath_scalar_identities_and_determinism()
    print("[PASS] test_fa2_t1_detmath_scalar_identities_and_determinism")
    passed += 1

    test_fa2_t1_jobpool_dag_chain_and_affine()
    print("[PASS] test_fa2_t1_jobpool_dag_chain_and_affine")
    passed += 1

    test_fa2_t1_memory_arena_and_frame_arenas()
    print("[PASS] test_fa2_t1_memory_arena_and_frame_arenas")
    passed += 1

    test_fa7_t1_udp_dual_stack_loopback_burst()
    print("[PASS] test_fa7_t1_udp_dual_stack_loopback_burst")
    passed += 1

    test_fa7_t1_bitpack_roundtrip_and_quat_accuracy()
    print("[PASS] test_fa7_t1_bitpack_roundtrip_and_quat_accuracy")
    passed += 1

    test_fa7_t1_aead_and_replay_window()
    print("[PASS] test_fa7_t1_aead_and_replay_window")
    passed += 1

    test_fa7_t1_net_channel_reliable_fragment_and_sequenced()
    print("[PASS] test_fa7_t1_net_channel_reliable_fragment_and_sequenced")
    passed += 1

    test_fa7_t1_net_hub_cookie_handshake_and_multi_client()
    print("[PASS] test_fa7_t1_net_hub_cookie_handshake_and_multi_client")
    passed += 1

    test_fa7_t1_replicator_aoi_despawn_and_interpolation()
    print("[PASS] test_fa7_t1_replicator_aoi_despawn_and_interpolation")
    passed += 1

    test_fa7_t1_rollback_physics_arena_and_visual_smoother()
    print("[PASS] test_fa7_t1_rollback_physics_arena_and_visual_smoother")
    passed += 1

    test_fa7_t1_virtual_sim_blackout_and_k8_survival()
    print("[PASS] test_fa7_t1_virtual_sim_blackout_and_k8_survival")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
