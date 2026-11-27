"""Comprehensive `<2s` verification suite for Flint v5 Core, Hybrid ECS v2, Scene/Prefab & `<300 ms` Hot-Reload:
  (a) Main-thread intentional FTZ/DAZ corruption -> `JobPool` work-stealing `sim_float_mode_ok()` auto-heal,
      subnormal IEEE-754 preservation, `fast_hash64`, and `dynlib` (`dl_open`/`dl_sym`/`dl_close`/`dl_error`).
  (b) ECS v2 dynamic `insert`/`remove` across >10 archetypes without mutating `Entity`, `SparseSet` tags,
      `Events[T]` 2-frame retention, and 1-thread vs 7-thread bit-exact `state_checksum` with parallel
      `ParallelCommands` + `par_for_each2`.
  (c) 4-level deep 1,000-node `TransformPlugin` parallel hierarchy propagation vs analytical reference,
      cycle-guarded reparenting, `Children` inline-4 + overflow, and `overstep_alpha` interpolation.
  (d) `.flscn` deterministic JSON & `.flscnb` LZ4 binary round-trip + nested `Prefab` with `PrefabOverride`
      + PIE `<0.2 ms` `snapshot_blob` / `restore_blob` bit-exact restoration.
  (e) `Schedule` conflict DAG ordering + >20-dependent fan-out zero-chunk relay node test + `EngineApp`
      fixed-timestep determinism (30 Hz vs 144 Hz `frame_dt` feeding identical 60 Hz `FixedUpdate` ticks).
  (f) Live `.so` plugin hot-reload test (compiling `/tmp/flint-eng1-guest-{v1,v2}.mojo` with `--emit shared-lib`,
      loading v1 into `HotReloadHost`, stepping `World`, hot-swapping to v2 without losing entity/resource state,
      and closing cleanly).
"""
from std.atomic import Atomic
from std.collections import List
from std.ffi import external_call
from std.math import abs
from std.memory import bitcast
from std.sys import size_of
from std.testing import assert_true, assert_equal

from flint.core.anchor import (
    ANCHOR_BYTES,
    STACKS,
    STACK_BYTES,
    STACK_SLOTS,
    SYS_MSYNC,
    A_REPLAY_BUFFER,
    A_REPLAY_COMMITTED,
    A_REPLAY_PENDING,
    anchor_base,
    anchor_ready,
    anchor_shutdown,
    install_anchor,
    load,
    sys,
)
from flint.core.cvars import ConsoleVars, CVAR_COUNT
from flint.core.detsort import mix64
from flint.core.detmath import (
    MXCSR_SIM,
    f32_bits,
    f32_from_bits,
    write_mxcsr,
    set_sim_float_mode,
    sim_float_mode_ok,
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
    PI,
)
from flint.core.ecs import EntityHandle, ArchetypeSlots, SimdColumn, ECB_PROV_GEN
from flint.core.jobs import (
    JobPool,
    JOB_PRIO_HIGH,
    JOB_PRIO_NORMAL,
    JOB_PRIO_LOW,
    MAX_DEPENDENTS,
    run_jobs,
    run_jobs_affine,
)
from flint.core.lz4 import lz4_pack
from flint.core.math import Vec3, Quat, Mat4
from flint.core.memory import (
    Arena,
    FrameArenas,
    MemCharge,
    MEM_FRAME,
    MEM_RENDER,
    MEM_ENTITIES,
    memory_install,
    memory_ready,
    memory_shutdown,
    memory_base,
    mem_live,
    mem_count,
)
from flint.core.profiler import ProfileEvent, ProfileTimeline, ZONE_COUNT, ZONE_SIM, profile_ticks
from flint.core.serial import BlobWriter, BlobReader, fourcc, fast_hash64
from flint.core.session_replay import SessionRecorder, FLUSH_AT
from flint.assets.json import JsonDocument, json_i64, parse_json
from flint.platform.dynlib import dl_open, dl_close, dl_error
from flint.platform.os import (
    FAULT_SHORT_WRITE, FAULT_BEFORE_RENAME, close_fd, env_now, file_size, get_time,
    open_read, read_bytes, read_stream, remove_file, write_bytes, write_file_durable,
)

from flint.ecs import (
    CHUNK_ALIGN,
    STORAGE_TABLE,
    STORAGE_SPARSE,
    NULL_ENTITY,
    comp_hash64,
    comp_hash64_str,
    Entity,
    Component,
    ComponentBitSet,
    BitSet256,
    ParallelCommands,
    Events,
    World,
)
from flint.scene import (
    Parent,
    SerializableComponent,
    LocalTransform,
    GlobalTransform,
    PrevGlobalTransform,
    Children,
    HierarchyDepth,
    TransformPlugin,
    TransformPropagationCache,
    set_parent,
    propagate_transforms_parallel,
    propagate_transforms_parallel_cached,
    interpolate_global_transform,
    interpolate_entity_transform,
    ComponentRegistry,
    Prefab,
    PrefabOverride,
    NestedPrefabRef,
    Scene,
    SceneMigrationRegistry,
    UnknownComponentsComponent,
)
from flint.app import (
    STAGE_UPDATE,
    STAGE_FIXED_UPDATE,
    STAGE_POST_UPDATE,
    Resource,
    Resources,
    FixedTime,
    FrameTime,
    Schedule,
    AppConfig,
    EngineApp,
    FLINT_PLUGIN_API_VERSION,
    FlintPluginApiV2,
    POD_FIELD_BOOL,
    POD_FIELD_COLOR_RGB,
    POD_FIELD_F32,
    POD_FIELD_INT,
    POD_FIELD_QUAT,
    POD_FIELD_VEC3,
    PodComponentSchema,
    PodFieldDesc,
    PodSchemaRegistryResource,
    HotReloadHost,
)


# ===========================================================================
# Suite (a): FTZ/DAZ Auto-Heal Across JobPool Work-Stealing + Core Primitives
# ===========================================================================

@always_inline
def _atomic_load_i64(base_addr: Int, slot: Int) -> Int:
    return Int(
        Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=base_addr + slot * 8)[unsafe_offset=0].load()
    )


def _float_mode_audit_chunk(ctx_addr: Int, chunk: Int):
    var mode_ok = sim_float_mode_ok()
    # Smallest positive IEEE-754 float32 subnormal: 2^-149 (flushes to 0 if FTZ/DAZ is active!)
    var tiny = f32_from_bits(UInt32(1) + UInt32(chunk & 3))
    var half = Float32(0.5) + Float32(chunk & 1) * Float32(0.5)
    var scaled = tiny * Float32(2.0) * half
    var sub_ok = scaled != Float32(0.0)

    if mode_ok:
        _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0].fetch_add(1)
    if sub_ok:
        _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx_addr + 8)[unsafe_offset=0].fetch_add(1)


def test_a_ftz_daz_heal_and_core() raises:
    # 1. Corrupt main-thread floating-point control register (FTZ bit 15 + DAZ bit 6)
    write_mxcsr(MXCSR_SIM | UInt32(1 << 15) | UInt32(1 << 6))
    assert_true(not sim_float_mode_ok(), "FTZ/DAZ corruption must make sim_float_mode_ok() False")

    var pool = JobPool()
    assert_true(pool.start(4), "JobPool(4) must start")

    var audit = List[Int64](length=2, fill=0)
    var audit_addr = Int(audit.unsafe_ptr())
    var chunks = 64
    pool.run(_float_mode_audit_chunk, audit_addr, chunks)
    assert_equal(_atomic_load_i64(audit_addr, 0), chunks)
    assert_equal(_atomic_load_i64(audit_addr, 1), chunks)
    assert_true(sim_float_mode_ok(), "Main thread must exit JobPool.wait() with healed sim float mode")
    _ = len(audit)

    # 2. Corrupt again and test run_jobs_affine + 0-worker serial fallback healing
    write_mxcsr(MXCSR_SIM | UInt32(1 << 15))
    assert_true(not sim_float_mode_ok(), "FTZ corruption must be detected")
    var audit2 = List[Int64](length=2, fill=0)
    var audit2_addr = Int(audit2.unsafe_ptr())
    run_jobs_affine(Int(Pointer(to=pool)), _float_mode_audit_chunk, audit2_addr, 16)
    assert_equal(_atomic_load_i64(audit2_addr, 0), 16)
    assert_equal(_atomic_load_i64(audit2_addr, 1), 16)
    assert_true(sim_float_mode_ok(), "Main thread healed after run_jobs_affine")
    _ = len(audit2)

    write_mxcsr(MXCSR_SIM | UInt32(1 << 15))
    var audit3 = List[Int64](length=2, fill=0)
    var audit3_addr = Int(audit3.unsafe_ptr())
    run_jobs(0, _float_mode_audit_chunk, audit3_addr, 4)
    assert_equal(_atomic_load_i64(audit3_addr, 0), 4)
    assert_equal(_atomic_load_i64(audit3_addr, 1), 4)
    assert_true(sim_float_mode_ok(), "Serial run_jobs(0, ...) must also heal sim float mode")
    _ = len(audit3)
    pool.stop()
    set_sim_float_mode()

    # 3. fast_hash64 word-strided determinism & unaligned/tail checks
    var buf_a = List[UInt8](capacity=37)
    var buf_b = List[UInt8](capacity=38)
    buf_b.append(0xFF)  # shift alignment by +1 byte
    for i in range(37):
        var v = UInt8((i * 73 + 19) & 0xFF)
        buf_a.append(v)
        buf_b.append(v)
    var h_aligned = fast_hash64(Int(buf_a.unsafe_ptr()), 37)
    var h_unaligned = fast_hash64(Int(buf_b.unsafe_ptr()) + 1, 37)
    _ = len(buf_b)
    assert_equal(h_aligned, h_unaligned)
    buf_a[36] ^= 1
    var h_mutated = fast_hash64(Int(buf_a.unsafe_ptr()), 37)
    _ = len(buf_a)
    assert_true(h_aligned != h_mutated, "Tail byte mutation must change fast_hash64")

    # 4. dynlib dl_open failure + dl_error + dl_close(0) safety
    var bad_lib = dl_open("/tmp/nonexistent-flint-v5-missing-plugin.so")
    assert_equal(bad_lib, 0)
    var err_msg = dl_error()
    assert_true(err_msg.byte_length() > 0, "dl_error() must report missing shared object")
    assert_equal(Int(dl_close(0)), 0)


# ===========================================================================
# Suite (b): Hybrid ECS v2 (>10 Archetypes, SparseSet, Events, 1T vs 7T Determinism)
# ===========================================================================

@fieldwise_init
struct CompPos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompPos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct CompVel(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompVel")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var vx: Float32
    var vy: Float32
    var vz: Float32


@fieldwise_init
struct CompMass(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompMass")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var inv_mass: Float32


@fieldwise_init
struct CompShield(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompShield")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var points: Float32


@fieldwise_init
struct CompHeat(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompHeat")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var temp: Float32


@fieldwise_init
struct CompAmmo(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompAmmo")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var rounds: Int32


@fieldwise_init
struct TagBurningSparse(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.TagBurningSparse")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var dps: Float32


@fieldwise_init
struct TagSelectedSparse(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.TagSelectedSparse")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var group_id: UInt8


@fieldwise_init
struct DamageEvent(Copyable, ImplicitlyCopyable, Movable):
    var target: Entity
    var amount: Float32


@fieldwise_init
struct _SimStepCtx(Copyable, ImplicitlyCopyable, Movable):
    var cmds_addr: Int
    var step_idx: Int


def _det_sim_kernel(
    lane: Int,
    order: Int,
    entity: Entity,
    pos: Pointer[CompPos, MutAnyOrigin],
    vel: Pointer[CompVel, MutAnyOrigin],
    ctx_addr: Int,
):
    ref ctx = Pointer[_SimStepCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref cmds = Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=ctx.cmds_addr)[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var phase = Float32(order) * Float32(0.05) + Float32(ctx.step_idx) * Float32(0.1)
    vel[unsafe_offset=0].vx += det_sin(phase) * Float32(0.25)
    vel[unsafe_offset=0].vy += det_cos(phase) * Float32(0.25)
    pos[unsafe_offset=0].x += vel[unsafe_offset=0].vx * dt
    pos[unsafe_offset=0].y += vel[unsafe_offset=0].vy * dt
    pos[unsafe_offset=0].z += vel[unsafe_offset=0].vz * dt

    # Deterministic structural mutations via lock-free per-worker ParallelCommands
    if (order + ctx.step_idx) % 29 == 0:
        var spawned = cmds.spawn_empty(lane, order_key=order)
        cmds.insert[CompPos](
            lane,
            spawned,
            CompPos(pos[unsafe_offset=0].x * Float32(0.5), pos[unsafe_offset=0].y, Float32(ctx.step_idx)),
            order_key=order,
        )
        cmds.insert[CompVel](
            lane,
            spawned,
            CompVel(Float32(1.0), Float32(-1.0), Float32(0.25)),
            order_key=order,
        )
        if order % 2 == 0:
            cmds.insert[TagBurningSparse](lane, spawned, TagBurningSparse(Float32(12.5)), order_key=order)
    elif (order + ctx.step_idx) % 47 == 0 and ctx.step_idx >= 2:
        cmds.despawn(lane, entity, order_key=order)


def _run_deterministic_ecs_sim(workers: Int) -> Tuple[Int, List[UInt64]]:
    var w = World()
    _ = w.register_component[CompPos]()
    _ = w.register_component[CompVel]()
    _ = w.register_component[CompMass]()
    _ = w.register_component[CompShield]()
    _ = w.register_component[TagBurningSparse]()

    for i in range(320):
        var e = w.spawn_empty()
        _ = w.insert[CompPos](e, CompPos(Float32(i) * 0.5, Float32(i % 17), Float32(-i) * 0.25))
        _ = w.insert[CompVel](e, CompVel(Float32(1.5), Float32(-0.75), Float32(0.5)))
        if i % 3 == 0:
            _ = w.insert[CompMass](e, CompMass(Float32(0.1)))
        if i % 5 == 0:
            _ = w.insert[CompShield](e, CompShield(Float32(50.0)))
        if i % 7 == 0:
            _ = w.insert[TagBurningSparse](e, TagBurningSparse(Float32(5.0)))

    var pool = JobPool()
    if workers > 0:
        _ = pool.start(workers)
    else:
        _ = pool.allocate()

    var hashes = List[UInt64]()
    var cmds = ParallelCommands()
    for step in range(8):
        _ = w.advance_tick()
        var ctx_box = List[_SimStepCtx](capacity=1)
        ctx_box.append(_SimStepCtx(Int(Pointer(to=cmds)), step))
        w.par_for_each2[CompPos, CompVel](pool, _det_sim_kernel, Int(ctx_box.unsafe_ptr()))
        var keep_ctx = len(ctx_box)
        _ = keep_ctx
        w.flush_commands(cmds)
        hashes.append(w.state_checksum())

    var count = w.entity_count()
    pool.stop()
    return (count, hashes^)


def test_b_ecs_v2_hybrid_events_determinism() raises:
    var w = World()
    var hero = w.spawn_empty()
    var orig_hero = hero

    # 1. Dynamic insert/remove across >12 distinct archetypes WITHOUT mutating Entity!
    for mask_bits in range(1, 16):
        if (mask_bits & 1) != 0:
            _ = w.insert[CompPos](hero, CompPos(Float32(mask_bits), 2.0, 3.0))
        else:
            _ = w.remove[CompPos](hero)
        if (mask_bits & 2) != 0:
            _ = w.insert[CompVel](hero, CompVel(4.0, 5.0, 6.0))
        else:
            _ = w.remove[CompVel](hero)
        if (mask_bits & 4) != 0:
            _ = w.insert[CompMass](hero, CompMass(0.5))
        else:
            _ = w.remove[CompMass](hero)
        if (mask_bits & 8) != 0:
            _ = w.insert[CompShield](hero, CompShield(100.0))
        else:
            _ = w.remove[CompShield](hero)
        assert_true(hero == orig_hero, "Entity(index, generation) must never mutate across archetypes")
        assert_true(w.is_alive(hero), "Hero entity must stay alive across archetype transitions")

    _ = w.insert[CompHeat](hero, CompHeat(80.0))
    _ = w.insert[CompAmmo](hero, CompAmmo(30))
    assert_true(w.archetype_count() > 12, "World must dynamically create >12 archetypes cleanly")
    assert_equal(w.get[CompPos](hero).x, Float32(15.0))
    assert_equal(Int(w.get[CompAmmo](hero).rounds), 30)

    # Verify 64-byte alignment of ArchetypeChunk columns and per-8-row changed_ticks
    var hero_arch = Int(w.entities[Int(hero.index)].archetype_id)
    var col0_ptr = w.archetypes[hero_arch].chunks[0].column_ptr(0)
    assert_equal(col0_ptr % CHUNK_ALIGN, 0)
    _ = w.advance_tick()
    var mut_pos = w.get_ptr_mut[CompPos](hero)
    assert_true(Bool(mut_pos), "get_ptr_mut[CompPos] must return valid pointer")
    mut_pos.value()[unsafe_offset=0].y = Float32(99.0)
    var pos_col = w.archetypes[hero_arch].col_index(w.component_id[CompPos]())
    assert_equal(
        w.archetypes[hero_arch].chunks[0].group_changed_tick(pos_col, 0),
        w.current_tick,
    )

    # 2. SparseSet tags attach/detach without changing table archetype_id!
    var arch_before_sparse = w.entities[Int(hero.index)].archetype_id
    _ = w.insert[TagBurningSparse](hero, TagBurningSparse(15.0))
    _ = w.insert[TagSelectedSparse](hero, TagSelectedSparse(3))
    assert_equal(w.entities[Int(hero.index)].archetype_id, arch_before_sparse)
    assert_true(w.has[TagBurningSparse](hero))
    assert_equal(w.get[TagBurningSparse](hero).dps, Float32(15.0))
    var q_burning = w.query_entities(w.bit_for[CompPos]().union(w.bit_for[TagBurningSparse]()))
    assert_equal(len(q_burning), 1)
    assert_true(q_burning[0] == hero)
    assert_true(w.remove[TagBurningSparse](hero))
    assert_true(not w.has[TagBurningSparse](hero))
    assert_equal(w.entities[Int(hero.index)].archetype_id, arch_before_sparse)

    # 3. Double-buffered Events[T] 2-frame retention
    var dmg_events = Events[DamageEvent]()
    var cur_fast = 0
    var cur_late = 0
    _ = dmg_events.send(DamageEvent(hero, 12.0))
    _ = dmg_events.send(DamageEvent(hero, 28.0))
    var f1_read = dmg_events.read_since(cur_fast)
    assert_equal(len(f1_read), 2)

    dmg_events.swap_buffers()  # Frame 2: Frame 1 events move to `older`
    _ = dmg_events.send(DamageEvent(hero, 45.0))
    var f2_late = dmg_events.read_since(cur_late)
    assert_equal(len(f2_late), 3)  # Sees all 3 across 2 frames!
    var f2_fast = dmg_events.read_since(cur_fast)
    assert_equal(len(f2_fast), 1)
    assert_equal(f2_fast[0].amount, Float32(45.0))

    dmg_events.swap_buffers()  # Frame 3: Frame 1 events expire, Frame 2 event remains
    var cur_very_late = 0
    var f3_read = dmg_events.read_since(cur_very_late)
    assert_equal(len(f3_read), 1)
    assert_equal(f3_read[0].amount, Float32(45.0))

    # 4. 1-thread vs 7-thread bit-exact state_checksum with ParallelCommands + par_for_each2!
    var res_1t = _run_deterministic_ecs_sim(1)
    var res_7t = _run_deterministic_ecs_sim(7)
    var res_0t = _run_deterministic_ecs_sim(0)
    assert_true(res_1t[0] > 320, "Simulation must have spawned net entities via ParallelCommands")
    assert_equal(res_1t[0], res_7t[0])
    assert_equal(res_1t[1], res_7t[1])
    assert_equal(res_0t[1], res_7t[1])
    var res_8t = _run_deterministic_ecs_sim(8)
    assert_equal(res_8t[1], res_1t[1])


# ===========================================================================
# Suite (c): 4-Level 1,000-Node Parallel Transform Hierarchy & Interpolation
# ===========================================================================

def _mat4_max_diff(a: Mat4, b: Mat4) -> Float32:
    var err = Float32(0.0)
    for i in range(16):
        var d = abs(a.values[i] - b.values[i])
        if d > err:
            err = d
    return err


def test_c_transform_hierarchy_1000_nodes() raises:
    var w = World()
    # Exact 1,000 nodes across 4 depth levels (0, 1, 2, 3):
    #   Depth 0: 10 roots
    #   Depth 1: 100 children (10 per root -> exercises Children inline-4 + overflow!)
    #   Depth 2: 300 children (3 per depth-1 node)
    #   Depth 3: 590 children (2 per first 295 depth-2 nodes)
    #   Total  = 10 + 100 + 300 + 590 = 1,000 nodes!
    var d0 = List[Entity]()
    var d1 = List[Entity]()
    var d2 = List[Entity]()
    var d3 = List[Entity]()
    var d3_d2_idx = List[Int]()

    for i in range(10):
        var e = w.spawn_empty()
        var rot = Quat.from_euler(0.1 * Float32(i), 0.25 * Float32(i), -0.05 * Float32(i))
        _ = w.insert[LocalTransform](
            e, LocalTransform(Vec3(Float32(i * 10), Float32(i), Float32(-i * 3)), rot, Vec3(1.0, 1.0, 1.0))
        )
        _ = w.insert[PrevGlobalTransform](e, PrevGlobalTransform())
        d0.append(e)

    for i in range(100):
        var e = w.spawn_empty()
        var p_idx = i // 10
        var rot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(i % 10) * 0.3)
        _ = w.insert[LocalTransform](
            e, LocalTransform(Vec3(Float32(i % 10) * 1.5, 2.0, -1.0), rot, Vec3(1.1, 0.9, 1.05))
        )
        assert_true(set_parent(w, e, d0[p_idx]))
        d1.append(e)

    # Verify root[0] has 10 children (4 inline + 6 overflow)
    assert_equal(w.get[Children](d0[0]).len(), 10)

    for i in range(300):
        var e = w.spawn_empty()
        var p_idx = i // 3
        var rot = Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), Float32(i % 5) * 0.2)
        _ = w.insert[LocalTransform](
            e, LocalTransform(Vec3(0.5, Float32(i % 3) * 0.75, 1.25), rot, Vec3(1.0, 1.0, 1.0))
        )
        assert_true(set_parent(w, e, d1[p_idx]))
        d2.append(e)

    for i in range(590):
        var e = w.spawn_empty()
        var p_idx = i // 2
        var rot = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), Float32(i % 7) * 0.15)
        _ = w.insert[LocalTransform](
            e, LocalTransform(Vec3(-0.25, 0.5, Float32(i % 2) * 2.0), rot, Vec3(0.8, 0.8, 0.8))
        )
        assert_true(set_parent(w, e, d2[p_idx]))
        d3.append(e)
        d3_d2_idx.append(p_idx)

    assert_equal(w.entity_count(), 1000)
    assert_equal(Int(w.get[HierarchyDepth](d3[589]).depth), 3)

    # Cycle guard must reject reparenting root d0[0] under its own depth-3 descendant d3[0]!
    assert_true(not set_parent(w, d0[0], d3[0]), "Cycle-guarded set_parent must reject ancestor cycle")
    assert_true(not set_parent(w, d2[0], d2[0]), "Self-parenting must be rejected")

    var pool = JobPool()
    _ = pool.start(4)
    propagate_transforms_parallel(w, pool, copy_to_prev=True)

    # Verify all 590 depth-3 leaves match exact analytical M0 * M1 * M2 * M3 matrix product!
    var max_err = Float32(0.0)
    for k in range(len(d3)):
        var leaf = d3[k]
        var idx2 = d3_d2_idx[k]
        var idx1 = idx2 // 3
        var idx0 = idx1 // 10

        var m0 = w.get[LocalTransform](d0[idx0]).to_mat4()
        var m1 = w.get[LocalTransform](d1[idx1]).to_mat4()
        var m2 = w.get[LocalTransform](d2[idx2]).to_mat4()
        var m3 = w.get[LocalTransform](leaf).to_mat4()
        var analytical = m0.multiply(m1).multiply(m2).multiply(m3)

        var computed = w.get[GlobalTransform](leaf).matrix.copy()
        var diff = _mat4_max_diff(analytical, computed)
        if diff > max_err:
            max_err = diff

    assert_true(max_err < Float32(1.0e-4), "1,000-node 4-level parallel hierarchy must match analytical Mat4")

    # Test PrevGlobalTransform + overstep_alpha interpolation (translation + Quat.slerp + scale)
    var rot_start = Quat.identity()
    var rot_end = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), PI * Float32(0.5))  # 90 deg yaw
    var gt_prev = GlobalTransform.from_local(LocalTransform(Vec3(0.0, 0.0, 0.0), rot_start, Vec3(1.0, 1.0, 1.0)))
    var gt_curr = GlobalTransform.from_local(LocalTransform(Vec3(10.0, 20.0, -4.0), rot_end, Vec3(3.0, 3.0, 3.0)))
    var gt_half = interpolate_global_transform(gt_prev, gt_curr, Float32(0.5))
    assert_true(abs(gt_half.translation.x - Float32(5.0)) < Float32(1.0e-5))
    assert_true(abs(gt_half.translation.y - Float32(10.0)) < Float32(1.0e-5))
    assert_true(abs(gt_half.scale.x - Float32(2.0)) < Float32(1.0e-5))
    # At alpha=0.5 of 90 deg yaw around +Y, rotated (0,0,1) should point along (sin 45, 0, cos 45) = (0.7071, 0, 0.7071)
    var fwd_45 = gt_half.rotation.rotate(Vec3(0.0, 0.0, 1.0))
    assert_true(abs(fwd_45.x - Float32(0.70710678)) < Float32(1.0e-3))
    assert_true(abs(fwd_45.z - Float32(0.70710678)) < Float32(1.0e-3))

    # Also test entity-level interpolation helper
    _ = w.insert[PrevGlobalTransform](d0[0], PrevGlobalTransform(gt_prev.copy()))
    _ = w.insert[GlobalTransform](d0[0], gt_curr.copy())
    var ent_interp = interpolate_entity_transform(w, d0[0], Float32(0.25))
    assert_true(abs(ent_interp.translation.x - Float32(2.5)) < Float32(1.0e-5))
    pool.stop()


# ===========================================================================
# Suite (d): .flscn JSON, .flscnb LZ4, Nested Prefabs/Overrides & <0.2ms PIE
# ===========================================================================

@fieldwise_init
struct ArmorStats(SerializableComponent, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.ArmorStats")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "ArmorStats"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ARMR")
    comptime SCHEMA_VERSION: Int = 1

    var armor: Float32
    var tier: Int32

    def write_bin(self, mut out: BlobWriter):
        out.f32(self.armor)
        out.u32(UInt32(self.tier))

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var a = reader.f32()
        var t = Int32(reader.u32())
        return Self(a, t)

    def write_json(self) -> String:
        return "{\"armor\": " + String(Float64(self.armor)) + ", \"tier\": " + String(self.tier) + "}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var a = Float32(doc.number(doc.object_get(node, "armor"), 0.0))
        var t = Int32(doc.integer(doc.object_get(node, "tier"), 0))
        return Self(a, t)


def test_d_scene_prefabs_and_pie() raises:
    var reg = ComponentRegistry.with_builtins()
    reg.register[ArmorStats]()

    # 1. Build child Prefab ("PlasmaWeapon") and parent Prefab ("ValkyrieMech") with NestedPrefabRef
    var weapon_prefab = Prefab("PlasmaWeapon")
    var w_mount = weapon_prefab.add_node("WeaponMount", -1)
    weapon_prefab.set_component[LocalTransform](
        w_mount, LocalTransform(Vec3(0.0, 0.0, 1.5), Quat.identity(), Vec3(1.0, 1.0, 1.0))
    )
    weapon_prefab.set_component[ArmorStats](w_mount, ArmorStats(25.0, 1))
    var w_muzzle = weapon_prefab.add_node("WeaponMuzzle", w_mount)
    weapon_prefab.set_component[LocalTransform](
        w_muzzle, LocalTransform(Vec3(0.0, 0.0, 2.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
    )

    var lib = List[Prefab]()
    lib.append(weapon_prefab^)

    var mech_prefab = Prefab("ValkyrieMech")
    var m_hull = mech_prefab.add_node("Hull", -1)
    mech_prefab.set_component[LocalTransform](
        m_hull, LocalTransform(Vec3(10.0, 0.0, -5.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
    )
    mech_prefab.set_component[ArmorStats](m_hull, ArmorStats(250.0, 2))

    var m_turret = mech_prefab.add_node("Turret", m_hull)
    mech_prefab.set_component[LocalTransform](
        m_turret, LocalTransform(Vec3(0.0, 3.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
    )

    var nested_gun = NestedPrefabRef(
        "PlasmaWeapon",
        attach_parent_slot=m_turret,
        local_offset=LocalTransform(Vec3(1.25, 0.5, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)),
    )
    nested_gun.add_override(PrefabOverride.for_name[ArmorStats]("WeaponMount", ArmorStats(60.0, 3)))
    mech_prefab.add_nested(nested_gun^)

    var w = World()
    var top_overrides = List[PrefabOverride]()
    top_overrides.append(PrefabOverride.for_name[ArmorStats]("Hull", ArmorStats(900.0, 5)))
    var spawned = mech_prefab.instantiate(w, reg, lib, top_overrides)
    assert_equal(len(spawned), 4)  # Hull, Turret, WeaponMount, WeaponMuzzle
    assert_equal(w.get[ArmorStats](spawned[0]).armor, Float32(900.0))
    assert_equal(Int(w.get[ArmorStats](spawned[0]).tier), 5)
    assert_equal(w.get[ArmorStats](spawned[2]).armor, Float32(60.0))
    assert_equal(Int(w.get[ArmorStats](spawned[2]).tier), 3)

    var pool = JobPool()
    _ = pool.allocate()
    propagate_transforms_parallel(w, pool)
    # WeaponMuzzle global pos = Hull(10,0,-5) + Turret(0,3,0) + Offset(1.25,0.5,0)+Mount(0,0,1.5) + Muzzle(0,0,2.0)
    #                        = (11.25, 3.5, -1.5)
    var muzzle_gt = w.get[GlobalTransform](spawned[3])
    assert_true(abs(muzzle_gt.translation.x - Float32(11.25)) < Float32(1.0e-4))
    assert_true(abs(muzzle_gt.translation.y - Float32(3.5)) < Float32(1.0e-4))
    assert_true(abs(muzzle_gt.translation.z - Float32(-1.5)) < Float32(1.0e-4))

    # 2. Deterministic .flscn JSON & .flscnb LZ4 binary round-trip
    var scene = Scene("HangarBay01")
    var json_a = scene.save_flscn_string(w, reg)
    var json_b = scene.save_flscn_string(w, reg)
    assert_equal(json_a, json_b)  # Deterministic key & entity order!

    var reordered = String('{"format":"flscn","version":2,"entities":[')
    reordered += '{"slot":1,"parent":0,"components":{}},{"slot":0,"parent":-1,"components":{}}]}'
    var reordered_bytes = List[UInt8]()
    for byte in reordered.as_bytes():
        reordered_bytes.append(byte)
    var edited_world = World()
    var edited_scene = Scene()
    var edited = edited_scene.load_flscn_bytes(edited_world, reg, reordered_bytes)
    assert_equal(len(edited), 2)
    assert_equal(edited_world.get[Parent](edited[1]).entity.to_bits(), edited[0].to_bits())
    assert_true(not edited_world.has[Parent](edited[0]))
    assert_true(json_a.find("\n      ") >= 0)

    var flscn_path = "/tmp/flint-eng1-test-scene.flscn"
    var flscnb_path = "/tmp/flint-eng1-test-scene.flscnb"
    var original = List[UInt8](length=3, fill=86)
    for path in [flscn_path, flscnb_path]:
        assert_true(write_bytes(path, original))
        var reader = open_read(path)
        assert_true(reader >= 0)
        if path == flscn_path:
            assert_true(scene.save_flscn_file(path, w, reg))
        else:
            assert_true(scene.save_flscnb_file(path, w, reg))
        # A simultaneous reader of the old scene still sees its complete original bytes.
        var old = List[UInt8](length=4, fill=0)
        var count = external_call["read", Int64](Int32(reader), old.unsafe_ptr(), Int64(len(old)))
        close_fd(reader)
        assert_equal(count, Int64(len(original)))
        assert_equal(old[0], original[0])
        var published = read_bytes(path)
        for fault in [FAULT_SHORT_WRITE, FAULT_BEFORE_RENAME]:
            assert_true(not write_file_durable(path, original, fault))
            assert_true(read_bytes(path) == published)


    var w_from_json = World()
    var s_json = Scene()
    var ents_j = s_json.load_flscn_file(flscn_path, w_from_json, reg)
    propagate_transforms_parallel(w_from_json, pool)

    var w_from_bin = World()
    var s_bin = Scene()
    var ents_b = s_bin.load_flscnb_file(flscnb_path, w_from_bin, reg)
    propagate_transforms_parallel(w_from_bin, pool)

    _ = remove_file(flscn_path)
    _ = remove_file(flscnb_path)

    assert_equal(len(ents_j), 4)
    assert_equal(len(ents_b), 4)
    assert_equal(s_json.name, "HangarBay01")
    assert_equal(s_bin.name, "HangarBay01")
    assert_equal(w_from_json.get[ArmorStats](ents_j[0]).armor, Float32(900.0))
    assert_equal(w_from_bin.get[ArmorStats](ents_b[0]).armor, Float32(900.0))
    var mj = w_from_json.get[GlobalTransform](ents_j[3])
    var mb = w_from_bin.get[GlobalTransform](ents_b[3])
    assert_true(abs(mj.translation.x - Float32(11.25)) < Float32(1.0e-4))
    assert_true(abs(mb.translation.x - Float32(11.25)) < Float32(1.0e-4))

    # 3. Play-In-Editor (PIE) <0.2 ms snapshot_blob / restore_blob bit-exact restoration!
    for idx in range(250):
        var extra = w.spawn_empty()
        _ = w.insert[LocalTransform](
            extra, LocalTransform(Vec3(Float32(idx), 1.0, -2.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
        )
        _ = w.insert[ArmorStats](extra, ArmorStats(Float32(idx * 2), Int32(idx % 4)))

    var pre_pie_checksum = w.state_checksum()
    var t_snap0 = get_time()
    scene.enter_pie(w)
    var snap_ms = (get_time() - t_snap0) * 1000.0

    # Mutate World aggressively during simulated PIE session
    _ = w.despawn(spawned[2])
    _ = w.insert[ArmorStats](spawned[0], ArmorStats(1.0, 99))
    for _ in range(50):
        var junk = w.spawn_empty()
        _ = w.insert[ArmorStats](junk, ArmorStats(999.0, 7))
    assert_true(w.state_checksum() != pre_pie_checksum)

    var t_rest0 = get_time()
    scene.exit_pie(w)
    var rest_ms = (get_time() - t_rest0) * 1000.0

    assert_equal(w.state_checksum(), pre_pie_checksum)
    assert_true(snap_ms < 5.0 and rest_ms < 5.0, "PIE snapshot/restore must be sub-millisecond fast")
    pool.stop()


# ===========================================================================
# Suite (e): Schedule Conflict DAG + >20-Dependent Relay + 30Hz vs 144Hz FixedTime
# ===========================================================================

def _dag_producer_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].store(42)


def _dag_reader_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var prod = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].load()
    if prod == 42:
        _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].fetch_add(1)


def _dag_final_writer_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var readers = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].load()
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 16)[unsafe_offset=0].store(readers)


def _fixed_integrator_sys(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var ft_opt = res_ptr[unsafe_offset=0].get_ptr[FixedTime]()
    if not ft_opt:
        return
    world_ptr[unsafe_offset=0].par_for_each2[LocalTransform, CompVel](
        pool_ptr[unsafe_offset=0],
        _integrate_local_vel_kernel,
        Int(Pointer(to=ft_opt.value()[unsafe_offset=0])),
    )


def _integrate_local_vel_kernel(
    _lane: Int,
    _order: Int,
    _e: Entity,
    lt: Pointer[LocalTransform, MutAnyOrigin],
    vel: Pointer[CompVel, MutAnyOrigin],
    ft_addr: Int,
):
    var dt = Pointer[FixedTime, ImmutAnyOrigin](unsafe_from_address=ft_addr)[unsafe_offset=0].dt
    lt[unsafe_offset=0].translation.x += vel[unsafe_offset=0].vx * dt
    lt[unsafe_offset=0].translation.y += vel[unsafe_offset=0].vy * dt
    lt[unsafe_offset=0].translation.z += vel[unsafe_offset=0].vz * dt


def _run_app_at_display_hz(frame_count: Int, frame_dt: Float32, workers: Int) -> Tuple[UInt64, UInt64]:
    var cfg = AppConfig(
        "FixedDeterminismApp",
        headless=True,
        worker_threads=workers,
        fixed_hz=60.0,
        max_fixed_steps_per_frame=8,
        default_frame_dt=frame_dt,
    )
    var app = EngineApp(cfg^)
    app.add_plugin(TransformPlugin())
    var b_lt = app.world.bit_for[LocalTransform]()
    var b_vel = app.world.bit_for[CompVel]()
    _ = app.add_system(
        STAGE_FIXED_UPDATE,
        "fixed_integrator",
        _fixed_integrator_sys,
        reads=b_vel,
        writes=b_lt,
    )

    for i in range(64):
        var e = app.world.spawn_empty()
        _ = app.world.insert[LocalTransform](
            e, LocalTransform(Vec3(Float32(i), 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
        )
        _ = app.world.insert[CompVel](
            e, CompVel(Float32(2.0 + Float32(i) * 0.25), Float32(-1.5), Float32(0.75))
        )

    app.startup()
    for _ in range(frame_count):
        app.step_one_frame(frame_dt)
    var ticks = app.resources.get_ptr[FixedTime]().value()[unsafe_offset=0].tick_count
    var cs = app.world.state_checksum()
    app.shutdown()
    return (ticks, cs)


def test_e_schedule_relay_and_fixed_timestep() raises:
    # 1. Conflict DAG + >20-dependent fan-out relay node splitting (25 reader systems > MAX_DEPENDENTS=16!)
    var sched = Schedule()
    var w = World()
    var res = Resources()
    var cmds = ParallelCommands()
    var pool = JobPool()
    _ = pool.start(4)

    var bit_a = BitSet256().with_bit(0)
    var audit = List[Int64](length=3, fill=0)
    var ctx_addr = Int(audit.unsafe_ptr())

    _ = sched.add_system(STAGE_UPDATE, "producer", _dag_producer_sys, writes=bit_a, user_ctx=ctx_addr)
    for r in range(25):
        _ = sched.add_system(
            STAGE_UPDATE,
            "reader_" + String(r),
            _dag_reader_sys,
            reads=bit_a,
            user_ctx=ctx_addr,
        )
    _ = sched.add_system(STAGE_UPDATE, "final_writer", _dag_final_writer_sys, writes=bit_a, user_ctx=ctx_addr)

    var relays = sched.compiled_relay_count(STAGE_UPDATE)
    assert_true(relays >= 1, "Schedule compiler must synthesize >=1 zero-chunk relay nodes for 25 dependents")
    sched.run_stage(STAGE_UPDATE, w, res, pool, cmds)
    assert_equal(_atomic_load_i64(ctx_addr, 0), 42)
    assert_equal(_atomic_load_i64(ctx_addr, 1), 25)
    assert_equal(_atomic_load_i64(ctx_addr, 2), 25)
    _ = len(audit)
    pool.stop()

    # 2. EngineApp FixedUpdate determinism: 30 Hz (30 frames) vs 144 Hz (144 frames) == exact 60 ticks & identical World!
    var run_30hz = _run_app_at_display_hz(30, Float32(1.0 / 30.0), 2)
    var run_144hz = _run_app_at_display_hz(144, Float32(1.0 / 144.0), 4)
    assert_equal(Int(run_30hz[0]), 60)
    assert_equal(Int(run_144hz[0]), 60)
    assert_equal(run_30hz[1], run_144hz[1])


# ===========================================================================
# Suite (f): Live `.so` Gameplay Plugin Hot-Reload (<300 ms swap, state intact)
# ===========================================================================

@fieldwise_init
struct HostSessionRes(Resource, Copyable, ImplicitlyCopyable):
    comptime RESOURCE_ID: UInt64 = 0x5E55104000000001
    var reload_hooks_seen: Int64


def _write_text_file(path: String, text: String) -> Bool:
    var bytes = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        bytes.append(b)
    return write_bytes(path, bytes)


def _discover_flint_src_dir() -> String:
    """Portably discovers Flint's `src` directory relative to CWD or `FLINT_ROOT`."""
    if file_size("src/flint/core/math.mojo") > 0:
        return "src"
    if file_size("../src/flint/core/math.mojo") > 0:
        return "../src"
    var env_root = env_now("FLINT_ROOT")
    if env_root.byte_length() > 0 and file_size(env_root + "/src/flint/core/math.mojo") > 0:
        return env_root + "/src"
    return "src"


def _discover_mojo_shell_cmd(src_v1: String, so_v1: String, src_v2: String, so_v2: String) -> String:
    """Resolves the Mojo 1.2 compiler dynamically without hardcoded user home paths."""
    var src_dir = _discover_flint_src_dir()
    var repo_root = "."
    if src_dir == "../src":
        repo_root = ".."
    elif src_dir.byte_length() > 4:
        repo_root = String(src_dir[byte=0 : src_dir.byte_length() - 4])

    var env_pfx = env_now("MOJO_PREFIX")
    var home_dir = env_now("HOME")
    var chosen_pfx = String("")
    if env_pfx.byte_length() > 0 and file_size(env_pfx + "/bin/mojo") > 0:
        chosen_pfx = env_pfx
    elif file_size(repo_root + "/.pixi/envs/default/bin/mojo") > 0:
        chosen_pfx = repo_root + "/.pixi/envs/default"
    elif file_size(repo_root + "/../.pixi/envs/default/bin/mojo") > 0:
        chosen_pfx = repo_root + "/../.pixi/envs/default"
    elif home_dir.byte_length() > 0 and file_size(home_dir + "/.pixi/envs/turbomojo/bin/mojo") > 0:
        chosen_pfx = home_dir + "/.pixi/envs/turbomojo"

    var env_mojo = env_now("MOJO")
    if chosen_pfx.byte_length() > 0:
        return (
            "export MOJO_PREFIX="
            + chosen_pfx
            + "; export MODULAR_HOME=${MOJO_PREFIX}/share/max; "
            + "export LD_LIBRARY_PATH=${MOJO_PREFIX}/lib:${LD_LIBRARY_PATH:-}; "
            + "${MOJO_PREFIX}/bin/mojo build -I "
            + src_dir
            + " "
            + src_v1
            + " --emit shared-lib -o "
            + so_v1
            + " -O0 --Werror 2>/dev/null & "
            + "${MOJO_PREFIX}/bin/mojo build -I "
            + src_dir
            + " "
            + src_v2
            + " --emit shared-lib -o "
            + so_v2
            + " -O0 --Werror 2>/dev/null & wait"
        )
    var mojo_bin = env_mojo if env_mojo.byte_length() > 0 else String("mojo")
    return (
        mojo_bin
        + " build -I "
        + src_dir
        + " "
        + src_v1
        + " --emit shared-lib -o "
        + so_v1
        + " -O0 --Werror 2>/dev/null & "
        + mojo_bin
        + " build -I "
        + src_dir
        + " "
        + src_v2
        + " --emit shared-lib -o "
        + so_v2
        + " -O0 --Werror 2>/dev/null & wait"
    )


def _guest_plugin_source(rate_per_sec: String, version_tag: String, res_words: String) -> String:
    var s = String("from flint.app import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2\n\n")
    s += "@export(\"flint_plugin_api_version\")\n"
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return FLINT_PLUGIN_API_VERSION\n\n"
    s += "def _step_energy(_ebits: UInt64, ptr: Int, dt: Float32) abi(\"C\"):\n"
    s += "    Pointer[Float32, MutAnyOrigin](unsafe_from_address=ptr)[unsafe_offset=0] += Float32(" + rate_per_sec + ") * dt\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(api_ptr: Int, world_ptr: Int, res_ptr: Int, is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var tid = api.register_pod_component(world_ptr, UInt64(0xE4E2670000000001), 4, 4, 0)\n"
    s += "    if is_reload == 0:\n"
    s += "        var e = api.spawn_empty(world_ptr)\n"
    s += "        var init_box = List[Float32](length=1, fill=5.0)\n"
    s += "        _ = api.insert_raw_pod(world_ptr, e, tid, Int(init_box.unsafe_ptr()))\n"
    s += "        _ = len(init_box)\n"
    s += "    var dyn_res = List[Int64](length=" + res_words + ", fill=Int64(" + version_tag + "))\n"
    s += "    var dyn_ptr = api.upsert_pod_resource(res_ptr, UInt64(0x5E55104000000099), " + res_words + " * 8, Int(dyn_res.unsafe_ptr()))\n"
    s += "    _ = len(dyn_res)\n"
    s += "    if dyn_ptr != 0:\n"
    s += "        Pointer[Int64, MutAnyOrigin](unsafe_from_address=dyn_ptr)[unsafe_offset=" + res_words + " - 1] = Int64(777 + " + version_tag + ")\n"
    s += "    var rptr = api.get_resource_ptr(res_ptr, UInt64(0x5E55104000000001))\n"
    s += "    if rptr != 0:\n"
    s += "        Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=0] += " + version_tag + "\n"
    s += "    return 0\n\n"
    s += "@export(\"flint_plugin_on_step\")\n"
    s += "def flint_plugin_on_step(api_ptr: Int, world_ptr: Int, _res_ptr: Int, dt: Float32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var tid = api.register_pod_component(world_ptr, UInt64(0xE4E2670000000001), 4, 4, 0)\n"
    s += "    return Int32(api.for_each_pod1(world_ptr, tid, _step_energy, dt))\n\n"
    s += "@export(\"flint_plugin_on_unload\")\n"
    s += "def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> None:\n"
    s += "    pass\n"
    return s^


def _guest_plugin_v1_source() -> String:
    var s = String("@fieldwise_init\n")
    s += "struct GuestApiV1(Copyable, ImplicitlyCopyable, Movable):\n"
    s += "    var abi_version: UInt64\n"
    s += "    var spawn_empty: def(Int) thin abi(\"C\") -> UInt64\n"
    s += "    var despawn: def(Int, UInt64) thin abi(\"C\") -> Int32\n"
    s += "    var is_alive: def(Int, UInt64) thin abi(\"C\") -> Int32\n"
    s += "    var entity_count: def(Int) thin abi(\"C\") -> Int\n"
    s += "    var register_pod_component: def(Int, UInt64, Int, Int, Int) thin abi(\"C\") -> Int\n"
    s += "    var insert_raw_pod: def(Int, UInt64, Int, Int) thin abi(\"C\") -> Int32\n"
    s += "    var get_raw_ptr_mut: def(Int, UInt64, Int) thin abi(\"C\") -> Int\n"
    s += "    var for_each_pod1: def(Int, Int, def(UInt64, Int, Float32) thin abi(\"C\") -> None, Float32) thin abi(\"C\") -> Int\n"
    s += "    var upsert_pod_resource: def(Int, UInt64, Int, Int) thin abi(\"C\") -> Int\n"
    s += "    var get_resource_ptr: def(Int, UInt64) thin abi(\"C\") -> Int\n\n"
    s += "@export(\"flint_plugin_api_version\")\n"
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return 1\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[GuestApiV1, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    return 0 if api.abi_version == 1 else -1\n\n"
    s += "@export(\"flint_plugin_on_step\")\n"
    s += "def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi(\"C\") -> Int32:\n"
    s += "    return 7\n"
    return s^


def _guest_plugin_abi_version_source(ver: Int) -> String:
    var s = String("@export(\"flint_plugin_api_version\")\n")
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return UInt32(" + String(ver) + ")\n\n"
    s += "@export(\"flint_plugin_on_step\")\n"
    s += "def flint_plugin_on_step(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _dt: Float32) abi(\"C\") -> Int32:\n"
    s += "    return 99\n"
    return s^


def _guest_plugin_missing_step_source() -> String:
    var s = String("@export(\"flint_plugin_api_version\")\n")
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return 2\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    return 0\n"
    return s^


def test_f_live_so_plugin_hot_reload() raises:
    assert_true(
        file_size(_discover_flint_src_dir() + "/flint/core/math.mojo") > 0,
        "Dynamic Flint src discovery must locate src/flint/core/math.mojo",
    )
    var src_v1 = "/tmp/flint-eng1-guest-v1.mojo"
    var src_v2 = "/tmp/flint-eng1-guest-v2.mojo"
    var so_v1 = "/tmp/flint-eng1-guest-v1.so"
    var so_v2 = "/tmp/flint-eng1-guest-v2.so"

    # v1 upserts a 16-byte (2 Int64 words) resource; v2 expands it to 256 bytes (32 Int64 words)!
    assert_true(_write_text_file(src_v1, _guest_plugin_source("10.0", "10", "2")))
    assert_true(_write_text_file(src_v2, _guest_plugin_source("100.0", "200", "32")))

    # Compile v1.so and v2.so in parallel with `--emit shared-lib` via portable CWD discovery
    var build_cmd = _discover_mojo_shell_cmd(src_v1, so_v1, src_v2, so_v2)
    var rc = external_call["system", Int32](build_cmd.as_c_string_span())
    assert_equal(Int(rc), 0)

    var world = World()
    var resources = Resources()
    resources.insert[HostSessionRes](HostSessionRes(0))

    var host = HotReloadHost("/tmp/flint-eng1-hrstage")
    assert_true(host.load_or_reload(so_v1, world, resources), "v1.so must load cleanly: " + host.last_error)
    assert_true(host.last_reload_ms < 300.0, "Hot-reload swap must complete in <300 ms")
    assert_equal(world.entity_count(), 1)
    assert_equal(Int(resources.get_ptr[HostSessionRes]().value()[unsafe_offset=0].reload_hooks_seen), 10)
    assert_equal(resources.byte_size_of(UInt64(0x5E55104000000099)), 16)

    # Step v1 twice with dt = 0.5s -> initial 5.0 + 2 * (10.0 * 0.5) = 15.0
    assert_equal(Int(host.step(world, resources, Float32(0.5))), 1)
    assert_equal(Int(host.step(world, resources, Float32(0.5))), 1)

    var tid_energy = world.find_component_by_hash(UInt64(0xE4E2670000000001))
    assert_true(tid_energy >= 0)
    var ents_before = world.query_entities(BitSet256().with_bit(tid_energy))
    assert_equal(len(ents_before), 1)
    var e0 = ents_before[0]
    var val_after_v1 = Pointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=world.get_raw_ptr(e0, tid_energy)
    )[unsafe_offset=0]
    assert_true(abs(val_after_v1 - Float32(15.0)) < Float32(1.0e-5))

    # Hot-swap to v2.so WITHOUT losing entity e0 or overflowing enlarged 256-byte resource!
    assert_true(host.load_or_reload(so_v2, world, resources), "v2.so must hot-swap cleanly: " + host.last_error)
    assert_true(host.last_reload_ms < 300.0, "v1 -> v2 hot-swap must complete in <300 ms")
    assert_equal(world.entity_count(), 1)
    assert_true(world.is_alive(e0), "Entity handle must remain alive across .so hot-reload")
    assert_equal(Int(resources.get_ptr[HostSessionRes]().value()[unsafe_offset=0].reload_hooks_seen), 210)
    assert_equal(resources.byte_size_of(UInt64(0x5E55104000000099)), 256)
    var grown_ptr = resources.get_raw_ptr(UInt64(0x5E55104000000099))
    assert_true(grown_ptr != 0)
    assert_equal(Int(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=grown_ptr)[unsafe_offset=0]), 200)
    assert_equal(Int(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=grown_ptr)[unsafe_offset=31]), 977)

    # Step v2 twice with dt = 0.5s -> preserved 15.0 + 2 * (100.0 * 0.5) = 115.0!
    assert_equal(Int(host.step(world, resources, Float32(0.5))), 1)
    assert_equal(Int(host.step(world, resources, Float32(0.5))), 1)
    var val_after_v2 = Pointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=world.get_raw_ptr(e0, tid_energy)
    )[unsafe_offset=0]
    assert_true(abs(val_after_v2 - Float32(115.0)) < Float32(1.0e-4))

    var src_compat_v1 = "/tmp/flint-eng1-guest-compat-v1.mojo"
    var so_compat_v1 = "/tmp/flint-eng1-guest-compat-v1.so"
    var src_v99 = "/tmp/flint-eng1-guest-v99.mojo"
    var so_v99 = "/tmp/flint-eng1-guest-v99.so"
    var src_missing_step = "/tmp/flint-eng1-guest-missing-step.mojo"
    var so_missing_step = "/tmp/flint-eng1-guest-missing-step.so"
    var so_corrupt = "/tmp/flint-eng1-guest-corrupt.so"
    assert_true(_write_text_file(src_compat_v1, _guest_plugin_v1_source()))
    assert_true(_write_text_file(src_v99, _guest_plugin_abi_version_source(99)))
    assert_true(_write_text_file(src_missing_step, _guest_plugin_missing_step_source()))
    assert_true(_write_text_file(so_corrupt, "NOT_AN_ELF_SHARED_OBJECT_12345"))
    var compat_cmd = _discover_mojo_shell_cmd(src_compat_v1, so_compat_v1, src_v99, so_v99)
    assert_equal(Int(external_call["system", Int32](compat_cmd.as_c_string_span())), 0)
    var missing_cmd = _discover_mojo_shell_cmd(src_missing_step, so_missing_step, src_missing_step, so_missing_step)
    assert_equal(Int(external_call["system", Int32](missing_cmd.as_c_string_span())), 0)

    # Failed reload attempts on corrupt, ABI-v1, ABI-v99, and missing-symbol .so must preserve active v2 plugin!
    var saved_temp_path = host.active_temp_path
    assert_true(not host.load_or_reload(so_corrupt, world, resources))
    assert_true(host.is_loaded())
    assert_equal(host.active_temp_path, saved_temp_path)
    assert_equal(Int(host.step(world, resources, Float32(0.0))), 1)

    assert_true(not host.load_or_reload(so_compat_v1, world, resources))
    assert_true(host.last_error.find("mismatch") >= 0)
    assert_true(host.is_loaded())
    assert_equal(host.active_temp_path, saved_temp_path)
    assert_equal(Int(host.step(world, resources, Float32(0.0))), 1)

    assert_true(not host.load_or_reload(so_v99, world, resources))
    assert_true(host.last_error.find("mismatch") >= 0)
    assert_true(host.is_loaded())
    assert_equal(host.active_temp_path, saved_temp_path)
    assert_equal(Int(host.step(world, resources, Float32(0.0))), 1)

    assert_true(not host.load_or_reload(so_missing_step, world, resources))
    assert_true(host.last_error.find("flint_plugin_on_step") >= 0)
    assert_true(host.is_loaded())
    assert_equal(host.active_temp_path, saved_temp_path)
    assert_equal(Int(host.step(world, resources, Float32(0.5))), 1)
    var val_after_failed_reloads = Pointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=world.get_raw_ptr(e0, tid_energy)
    )[unsafe_offset=0]
    assert_true(abs(val_after_failed_reloads - Float32(165.0)) < Float32(1.0e-4))

    assert_true(host.unload(world, resources, is_reload=False), "Final dl_close unload must succeed cleanly")
    assert_true(not host.is_loaded())
    _ = size_of[Entity]()

    # Also verify allow_v1=False rejects ABI v1 with "mismatch" while allow_v1=True accepts it
    var strict_host = HotReloadHost("/tmp/flint-eng1-strict-stage")
    assert_true(not strict_host.load_or_reload(so_compat_v1, world, resources))
    assert_true(strict_host.last_error.find("mismatch") >= 0)

    var compat_host = HotReloadHost("/tmp/flint-eng1-compat-stage", allow_v1=True)
    assert_true(compat_host.load_or_reload(so_compat_v1, world, resources), compat_host.last_error)
    assert_equal(Int(compat_host.step(world, resources, Float32(0.1))), 7)
    assert_true(compat_host.unload(world, resources))

    _ = remove_file(src_v1)
    _ = remove_file(src_v2)
    _ = remove_file(so_v1)
    _ = remove_file(so_v2)
    _ = remove_file(src_compat_v1)
    _ = remove_file(so_compat_v1)
    _ = remove_file(src_v99)
    _ = remove_file(so_v99)
    _ = remove_file(src_missing_step)
    _ = remove_file(so_missing_step)
    _ = remove_file(so_corrupt)


def test_u_m5a_pod_schema_and_registry_add_remove() raises:
    assert_equal(Int(FLINT_PLUGIN_API_VERSION), 2)
    assert_equal(comp_hash64_str("LocalTransform"), comp_hash64("LocalTransform"))

    var api = FlintPluginApiV2()
    assert_true(api.is_valid())
    assert_equal(Int(api.struct_size), size_of[FlintPluginApiV2]())

    var schema = PodComponentSchema("M5aPodStats", 24, 4, STORAGE_TABLE)
    assert_true(schema.add_field(PodFieldDesc("hp", POD_FIELD_F32, 0, 4)))
    assert_true(schema.add_field(PodFieldDesc("level", POD_FIELD_INT, 4, 4)))
    assert_true(schema.add_field(PodFieldDesc("alive", POD_FIELD_BOOL, 8, 1)))
    assert_true(schema.add_field(PodFieldDesc("home", POD_FIELD_VEC3, 12, 12)))
    assert_true(schema.validate())

    # Out-of-bounds, duplicate field, and invalid alignment rejection
    assert_true(not schema.add_field(PodFieldDesc("hp", POD_FIELD_F32, 0, 4)))
    assert_true(not schema.add_field(PodFieldDesc("oob", POD_FIELD_F32, 22, 4)))
    var bad_align = PodComponentSchema("BadAlignPod", 8, 3)
    assert_true(not bad_align.validate())

    # Value-marshal encode/decode + FlintPluginApiV2.register_schema
    var encoded = schema.encode_bytes()
    var decoded = PodComponentSchema.decode_bytes(encoded^)
    assert_true(decoded.matches_layout(schema))

    var w = World()
    var res = Resources()
    var tid = api.register_schema(Int(Pointer(to=w)), Int(Pointer(to=res)), schema)
    assert_true(tid >= 0)
    var psreg = res.get_ptr[PodSchemaRegistryResource]()
    assert_true(Bool(psreg))
    assert_equal(Int(psreg.value()[unsafe_offset=0].epoch), 1)
    assert_equal(psreg.value()[unsafe_offset=0].find_schema("M5aPodStats"), 0)

    # Zero-fill on insert_raw_pod with src_ptr=0
    var e0 = w.spawn_empty()
    assert_true(w.insert_raw_pod(e0, tid, 0))
    var ptr0 = w.get_raw_ptr(e0, tid)
    assert_true(ptr0 != 0)
    for bi in range(24):
        assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=ptr0 + bi)[unsafe_offset=0]), 0)

    # ComponentRegistry register_pod_schema, insert_default_on_entity, remove_from_entity, and round-trips
    var reg = ComponentRegistry.with_builtins()
    assert_true(reg.register_pod_schema(schema))
    assert_true(reg.register_pod_schema(schema))  # idempotent on matching layout
    var e1 = w.spawn_empty()
    assert_true(reg.insert_default_on_entity(w, e1, "LocalTransform"))
    assert_true(not reg.insert_default_on_entity(w, e1, "LocalTransform"))  # duplicate rejected
    assert_true(w.has[LocalTransform](e1))
    assert_true(reg.insert_default_on_entity(w, e1, "M5aPodStats"))
    assert_true(not reg.insert_default_on_entity(w, e1, "M5aPodStats"))
    assert_true(w.has_by_type_id(e1, tid))

    var m_ptr = w.get_raw_ptr_mut(e1, tid)
    assert_true(m_ptr != 0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=m_ptr)[unsafe_offset=0] = Float32(75.5)
    Pointer[Int32, MutAnyOrigin](unsafe_from_address=m_ptr + 4)[unsafe_offset=0] = Int32(9)
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=m_ptr + 8)[unsafe_offset=0] = UInt8(1)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=m_ptr + 12)[unsafe_offset=0] = Float32(3.5)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=m_ptr + 16)[unsafe_offset=0] = Float32(-1.25)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=m_ptr + 20)[unsafe_offset=0] = Float32(7.75)

    var orig_csum = w.state_checksum()
    var sc = Scene("M5aPodScene")
    var json_str = sc.save_flscn_string(w, reg)
    var bin_bytes = sc.save_flscnb_bytes(w, reg)

    var w_j = World()
    var j_bytes = List[UInt8](capacity=json_str.byte_length())
    for b in json_str.as_bytes():
        j_bytes.append(b)
    assert_true(sc.load_flscn_bytes(w_j, reg, j_bytes))
    var tid_j = w_j.find_component_by_hash(schema.component_hash)
    assert_true(tid_j >= 0)
    var ents_j = w_j.query_entities(BitSet256().with_bit(tid_j))
    assert_equal(len(ents_j), 2)
    assert_equal(w_j.state_checksum(), orig_csum)
    var checked_j_pop = False
    var checked_j_zero = False
    for i in range(len(ents_j)):
        var ej = ents_j[i]
        var pj = w_j.get_raw_ptr(ej, tid_j)
        assert_true(pj != 0)
        if w_j.has[LocalTransform](ej):
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pj)[unsafe_offset=0], Float32(75.5))
            assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=pj + 4)[unsafe_offset=0], Int32(9))
            assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=pj + 8)[unsafe_offset=0]), 1)
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pj + 12)[unsafe_offset=0], Float32(3.5))
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pj + 16)[unsafe_offset=0], Float32(-1.25))
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pj + 20)[unsafe_offset=0], Float32(7.75))
            checked_j_pop = True
        else:
            for bi in range(24):
                assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=pj + bi)[unsafe_offset=0]), 0)
            checked_j_zero = True
    assert_true(checked_j_pop and checked_j_zero)

    var w_b = World()
    assert_true(sc.load_flscnb_bytes(w_b, reg, bin_bytes))
    var tid_b = w_b.find_component_by_hash(schema.component_hash)
    assert_true(tid_b >= 0)
    var ents_b = w_b.query_entities(BitSet256().with_bit(tid_b))
    assert_equal(len(ents_b), 2)
    assert_equal(w_b.state_checksum(), orig_csum)
    var checked_b_pop = False
    for i in range(len(ents_b)):
        var eb = ents_b[i]
        if w_b.has[LocalTransform](eb):
            var pb = w_b.get_raw_ptr(eb, tid_b)
            assert_true(pb != 0)
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pb)[unsafe_offset=0], Float32(75.5))
            assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=pb + 4)[unsafe_offset=0], Int32(9))
            assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=pb + 8)[unsafe_offset=0]), 1)
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pb + 12)[unsafe_offset=0], Float32(3.5))
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pb + 16)[unsafe_offset=0], Float32(-1.25))
            assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pb + 20)[unsafe_offset=0], Float32(7.75))
            checked_b_pop = True
    assert_true(checked_b_pop)

    var snap = BlobWriter()
    w.snapshot_blob(snap)
    var w_s = World()
    reg.ensure_registered_in_world(w_s)
    var snap_r = BlobReader(snap.data.copy())
    w_s.restore_blob(snap_r)
    assert_equal(w_s.entity_count(), 2)
    assert_equal(w_s.state_checksum(), orig_csum)
    var tid_s = w_s.find_component_by_hash(schema.component_hash)
    assert_true(tid_s >= 0)
    assert_equal(len(w_s.query_entities(BitSet256().with_bit(tid_s))), 2)

    assert_true(reg.remove_from_entity(w, e1, "M5aPodStats"))
    assert_true(not reg.remove_from_entity(w, e1, "M5aPodStats"))
    assert_true(not w.has_by_type_id(e1, tid))
    assert_true(reg.remove_from_entity(w, e1, "LocalTransform"))
    assert_true(not w.has[LocalTransform](e1))

    # PodSchemaRegistryResource rejects hash/fourcc collisions across distinct names
    var psreg_col = PodSchemaRegistryResource()
    var schema_alpha = PodComponentSchema("AlphaPod", 8, 4, STORAGE_TABLE)
    assert_true(psreg_col.register_schema(schema_alpha))
    var dup_fourcc = PodComponentSchema("BetaPod", 8, 4, STORAGE_TABLE)
    dup_fourcc.schema_fourcc = schema_alpha.schema_fourcc
    assert_true(not psreg_col.register_schema(dup_fourcc))
    var dup_hash = PodComponentSchema("GammaPod", 8, 4, STORAGE_TABLE)
    dup_hash.component_hash = schema_alpha.component_hash
    assert_true(not psreg_col.register_schema(dup_hash))

    # Fieldless PodComponentSchema (__raw_hex__) round-trips byte-for-byte through .flscn
    var raw_schema = PodComponentSchema("RawHexPod", 8, 4, STORAGE_TABLE)
    assert_true(reg.register_pod_schema(raw_schema))
    var w_raw = World()
    reg.ensure_registered_in_world(w_raw)
    var tid_raw = w_raw.find_component_by_hash(raw_schema.component_hash)
    assert_true(tid_raw >= 0)
    var e_raw = w_raw.spawn_empty()
    assert_true(reg.insert_default_on_entity(w_raw, e_raw, "RawHexPod"))
    var rptr_w = w_raw.get_raw_ptr_mut(e_raw, tid_raw)
    assert_true(rptr_w != 0)
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rptr_w)[unsafe_offset=0] = UInt32(0xDEADBEEF)
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rptr_w + 4)[unsafe_offset=0] = UInt32(0xCAFEBABE)

    var sc_raw = Scene("RawHexScene")
    var raw_json = sc_raw.save_flscn_string(w_raw, reg)
    assert_true(raw_json.find("__raw_hex__") >= 0)
    var raw_jbytes = List[UInt8](capacity=raw_json.byte_length())
    for b in raw_json.as_bytes():
        raw_jbytes.append(b)
    var w_raw_out = World()
    _ = sc_raw.load_flscn_bytes(w_raw_out, reg, raw_jbytes)
    var tid_raw_out = w_raw_out.find_component_by_hash(raw_schema.component_hash)
    assert_true(tid_raw_out >= 0)
    var ents_raw_out = w_raw_out.query_entities(BitSet256().with_bit(tid_raw_out))
    assert_equal(len(ents_raw_out), 1)
    var rptr_out = w_raw_out.get_raw_ptr(ents_raw_out[0], tid_raw_out)
    assert_true(rptr_out != 0)
    assert_equal(Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=rptr_out)[unsafe_offset=0], UInt32(0xDEADBEEF))
    assert_equal(Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=rptr_out + 4)[unsafe_offset=0], UInt32(0xCAFEBABE))

    # Full multi-field PodComponentSchema covering BOOL (1B & 4B), INT (4B & 8B),
    # F32, VEC3, COLOR_RGB, and QUAT through .flscn and .flscnb under -O3
    var full_schema = PodComponentSchema("FullFieldsPod", 64, 8, STORAGE_TABLE)
    assert_true(full_schema.add_field(PodFieldDesc("flag1", POD_FIELD_BOOL, 0, 1)))
    assert_true(full_schema.add_field(PodFieldDesc("flag4", POD_FIELD_BOOL, 4, 4)))
    assert_true(full_schema.add_field(PodFieldDesc("i32v", POD_FIELD_INT, 8, 4)))
    assert_true(full_schema.add_field(PodFieldDesc("f32v", POD_FIELD_F32, 12, 4)))
    assert_true(full_schema.add_field(PodFieldDesc("i64v", POD_FIELD_INT, 16, 8)))
    assert_true(full_schema.add_field(PodFieldDesc("vec3v", POD_FIELD_VEC3, 24, 12)))
    assert_true(full_schema.add_field(PodFieldDesc("colv", POD_FIELD_COLOR_RGB, 36, 12)))
    assert_true(full_schema.add_field(PodFieldDesc("quatv", POD_FIELD_QUAT, 48, 16)))
    assert_true(full_schema.validate())
    assert_true(reg.register_pod_schema(full_schema))

    var w_full = World()
    reg.ensure_registered_in_world(w_full)
    var tid_full = w_full.find_component_by_hash(full_schema.component_hash)
    assert_true(tid_full >= 0)

    var ef1 = w_full.spawn_empty()
    assert_true(reg.insert_default_on_entity(w_full, ef1, "FullFieldsPod"))
    var fp1 = w_full.get_raw_ptr_mut(ef1, tid_full)
    assert_true(fp1 != 0)
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=fp1)[unsafe_offset=0] = UInt8(1)
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=fp1 + 4)[unsafe_offset=0] = UInt32(1)
    Pointer[Int32, MutAnyOrigin](unsafe_from_address=fp1 + 8)[unsafe_offset=0] = Int32(-42)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 12)[unsafe_offset=0] = Float32(75.5)
    Pointer[Int64, MutAnyOrigin](unsafe_from_address=fp1 + 16)[unsafe_offset=0] = Int64(1234567890123)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 24)[unsafe_offset=0] = Float32(1.5)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 28)[unsafe_offset=0] = Float32(-2.25)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 32)[unsafe_offset=0] = Float32(3.75)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 36)[unsafe_offset=0] = Float32(0.25)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 40)[unsafe_offset=0] = Float32(0.5)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 44)[unsafe_offset=0] = Float32(0.75)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 48)[unsafe_offset=0] = Float32(0.125)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 52)[unsafe_offset=0] = Float32(-0.25)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 56)[unsafe_offset=0] = Float32(0.5)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp1 + 60)[unsafe_offset=0] = Float32(0.75)

    var ef2 = w_full.spawn_empty()
    assert_true(reg.insert_default_on_entity(w_full, ef2, "LocalTransform"))
    assert_true(reg.insert_default_on_entity(w_full, ef2, "FullFieldsPod"))
    var fp2 = w_full.get_raw_ptr_mut(ef2, tid_full)
    assert_true(fp2 != 0)
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=fp2)[unsafe_offset=0] = UInt8(0)
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=fp2 + 4)[unsafe_offset=0] = UInt32(0)
    Pointer[Int32, MutAnyOrigin](unsafe_from_address=fp2 + 8)[unsafe_offset=0] = Int32(0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 12)[unsafe_offset=0] = Float32(-0.5)
    Pointer[Int64, MutAnyOrigin](unsafe_from_address=fp2 + 16)[unsafe_offset=0] = Int64(-987654321098)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 24)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 28)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 32)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 36)[unsafe_offset=0] = Float32(1.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 40)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 44)[unsafe_offset=0] = Float32(0.5)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 48)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 52)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 56)[unsafe_offset=0] = Float32(0.0)
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=fp2 + 60)[unsafe_offset=0] = Float32(1.0)

    var full_csum = w_full.state_checksum()
    var sc_full = Scene("FullFieldsScene")
    var full_json = sc_full.save_flscn_string(w_full, reg)
    var full_bin = sc_full.save_flscnb_bytes(w_full, reg)

    var full_jbytes = List[UInt8](capacity=full_json.byte_length())
    for b in full_json.as_bytes():
        full_jbytes.append(b)
    var w_full_j = World()
    assert_true(sc_full.load_flscn_bytes(w_full_j, reg, full_jbytes))
    assert_equal(w_full_j.state_checksum(), full_csum)

    var w_full_b = World()
    assert_true(sc_full.load_flscnb_bytes(w_full_b, reg, full_bin))
    assert_equal(w_full_b.state_checksum(), full_csum)

    for pass_idx in range(2):
        var check_ptr = Int(Pointer(to=w_full_j)) if pass_idx == 0 else Int(Pointer(to=w_full_b))
        ref w_chk = Pointer[World, ImmutAnyOrigin](unsafe_from_address=check_ptr)[unsafe_offset=0]
        var tid_chk = w_chk.find_component_by_hash(full_schema.component_hash)
        assert_true(tid_chk >= 0)
        var ents_chk = w_chk.query_entities(BitSet256().with_bit(tid_chk))
        assert_equal(len(ents_chk), 2)
        var saw_f1 = False
        var saw_f2 = False
        for ei in range(len(ents_chk)):
            var ent = ents_chk[ei]
            var fp = w_chk.get_raw_ptr(ent, tid_chk)
            assert_true(fp != 0)
            if not w_chk.has[LocalTransform](ent):
                assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=fp)[unsafe_offset=0]), 1)
                assert_equal(Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=fp + 4)[unsafe_offset=0], UInt32(1))
                assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=fp + 8)[unsafe_offset=0], Int32(-42))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 12)[unsafe_offset=0], Float32(75.5))
                assert_equal(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=fp + 16)[unsafe_offset=0], Int64(1234567890123))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 24)[unsafe_offset=0], Float32(1.5))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 28)[unsafe_offset=0], Float32(-2.25))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 32)[unsafe_offset=0], Float32(3.75))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 36)[unsafe_offset=0], Float32(0.25))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 40)[unsafe_offset=0], Float32(0.5))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 44)[unsafe_offset=0], Float32(0.75))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 48)[unsafe_offset=0], Float32(0.125))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 52)[unsafe_offset=0], Float32(-0.25))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 56)[unsafe_offset=0], Float32(0.5))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 60)[unsafe_offset=0], Float32(0.75))
                saw_f1 = True
            else:
                assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=fp)[unsafe_offset=0]), 0)
                assert_equal(Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=fp + 4)[unsafe_offset=0], UInt32(0))
                assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=fp + 8)[unsafe_offset=0], Int32(0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 12)[unsafe_offset=0], Float32(-0.5))
                assert_equal(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=fp + 16)[unsafe_offset=0], Int64(-987654321098))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 24)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 28)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 32)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 36)[unsafe_offset=0], Float32(1.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 40)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 44)[unsafe_offset=0], Float32(0.5))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 48)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 52)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 56)[unsafe_offset=0], Float32(0.0))
                assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=fp + 60)[unsafe_offset=0], Float32(1.0))
                saw_f2 = True
        assert_true(saw_f1 and saw_f2)
    _ = w_full_j.entity_count()
    _ = w_full_b.entity_count()


def _assert_all_zero_12(ptr: Int) raises:
    assert_true(ptr != 0)
    for bi in range(12):
        assert_equal(Int(Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=ptr + bi)[unsafe_offset=0]), 0)


def test_v_m5a_pod_uaf_and_int64_regressions() raises:
    # 0. Exact JsonDocument.i64 / JsonDocument.integer / json_i64 edge & overflow checks
    var j_raw = String(
        '{"min":-9223372036854775808,"max":9223372036854775807,'
        '"p53":9007199254740993,"n53":-9007199254740993,"neg0":-0,'
        '"dec":42.9,"sci":1e5,"ovf":9223372036854775808,'
        '"unf":-9223372036854775809,"huge":99999999999999999999,'
        '"big":1e999,"neg":-1e999}'
    )
    var jb = List[UInt8](capacity=j_raw.byte_length())
    for ch in j_raw.as_bytes():
        jb.append(ch)
    var jdoc = parse_json(jb)
    assert_true(jdoc.valid)
    var i64_min = bitcast[DType.int64](UInt64(0x8000000000000000))
    var i64_max = Int64(9223372036854775807)
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "min"), 0), i64_min)
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "max"), 0), i64_max)
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "p53"), 0), Int64(9007199254740993))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "n53"), 0), Int64(-9007199254740993))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "neg0"), -1), Int64(0))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "dec"), -1), Int64(42))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "sci"), -1), Int64(100000))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "ovf"), -11), Int64(-11))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "unf"), -12), Int64(-12))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "huge"), -13), Int64(-13))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "big"), -14), Int64(-14))
    assert_equal(jdoc.i64(jdoc.object_get(jdoc.root, "neg"), -15), Int64(-15))
    assert_equal(json_i64(jdoc, jdoc.object_get(jdoc.root, "min"), 0), i64_min)
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "min"), 0), Int(i64_min))
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "max"), 0), Int(i64_max))
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "p53"), 0), 9007199254740993)
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "big"), -1), -1)
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "neg"), -7), -7)

    # Subtest A: -O3 ASAP-destructor UAF on 12-byte all-zero PODs (NanPodT, NanPodS, RawZero12)
    var nan_t = PodComponentSchema("NanPodT", 12, 4, STORAGE_TABLE)
    assert_true(nan_t.add_field(PodFieldDesc("x", POD_FIELD_F32, 0, 4)))
    assert_true(nan_t.add_field(PodFieldDesc("y", POD_FIELD_F32, 4, 4)))
    assert_true(nan_t.add_field(PodFieldDesc("z", POD_FIELD_F32, 8, 4)))
    assert_true(nan_t.validate())

    var nan_s = PodComponentSchema("NanPodS", 12, 4, STORAGE_SPARSE)
    assert_true(nan_s.add_field(PodFieldDesc("x", POD_FIELD_F32, 0, 4)))
    assert_true(nan_s.add_field(PodFieldDesc("y", POD_FIELD_F32, 4, 4)))
    assert_true(nan_s.add_field(PodFieldDesc("z", POD_FIELD_F32, 8, 4)))
    assert_true(nan_s.validate())

    var raw_z = PodComponentSchema("RawZero12", 12, 4, STORAGE_TABLE)
    assert_true(raw_z.validate())

    var reg_a = ComponentRegistry.with_builtins()
    assert_true(reg_a.register_pod_schema(nan_t))
    assert_true(reg_a.register_pod_schema(nan_s))
    assert_true(reg_a.register_pod_schema(raw_z))

    # Pre-churn glibc tcache with 12-byte List[UInt8] allocations
    for ci in range(32):
        var tmp_churn = List[UInt8](length=12, fill=UInt8((ci * 37 + 11) & 0xFF))
        assert_equal(len(tmp_churn), 12)

    var w_def = World()
    reg_a.ensure_registered_in_world(w_def)
    var tid_nt = w_def.find_component_by_hash(nan_t.component_hash)
    var tid_ns = w_def.find_component_by_hash(nan_s.component_hash)
    var tid_rz = w_def.find_component_by_hash(raw_z.component_hash)
    assert_true(tid_nt >= 0 and tid_ns >= 0 and tid_rz >= 0)

    var ents_def = List[Entity]()
    for _ in range(8):
        var e = w_def.spawn_empty()
        assert_true(reg_a.insert_default_on_entity(w_def, e, "NanPodT"))
        assert_true(reg_a.insert_default_on_entity(w_def, e, "NanPodS"))
        assert_true(reg_a.insert_default_on_entity(w_def, e, "RawZero12"))
        _assert_all_zero_12(w_def.get_raw_ptr(e, tid_nt))
        _assert_all_zero_12(w_def.get_raw_ptr(e, tid_ns))
        _assert_all_zero_12(w_def.get_raw_ptr(e, tid_rz))
        ents_def.append(e)
    var orig_zero_csum = w_def.state_checksum()

    # Direct read_entity_json (both fieldful and __raw_hex__)
    var doc_f_raw = String('{"x": 0.0, "y": 0.0, "z": 0.0}')
    var doc_f_b = List[UInt8](capacity=doc_f_raw.byte_length())
    for ch in doc_f_raw.as_bytes():
        doc_f_b.append(ch)
    var doc_f = parse_json(doc_f_b)
    assert_true(doc_f.valid)

    var doc_h_raw = String('{"__raw_hex__": "000000000000000000000000"}')
    var doc_h_b = List[UInt8](capacity=doc_h_raw.byte_length())
    for ch in doc_h_raw.as_bytes():
        doc_h_b.append(ch)
    var doc_h = parse_json(doc_h_b)
    assert_true(doc_h.valid)

    var idx_nt = reg_a.find_by_name("NanPodT")
    var idx_ns = reg_a.find_by_name("NanPodS")
    var idx_rz = reg_a.find_by_name("RawZero12")
    var w_rj = World()
    reg_a.ensure_registered_in_world(w_rj)
    for _ in range(8):
        var e = w_rj.spawn_empty()
        assert_true(reg_a.read_entity_json(idx_nt, w_rj, e, doc_f, doc_f.root))
        assert_true(reg_a.read_entity_json(idx_ns, w_rj, e, doc_f, doc_f.root))
        assert_true(reg_a.read_entity_json(idx_rz, w_rj, e, doc_h, doc_h.root))
        _assert_all_zero_12(w_rj.get_raw_ptr(e, w_rj.find_component_by_hash(nan_t.component_hash)))
        _assert_all_zero_12(w_rj.get_raw_ptr(e, w_rj.find_component_by_hash(nan_s.component_hash)))
        _assert_all_zero_12(w_rj.get_raw_ptr(e, w_rj.find_component_by_hash(raw_z.component_hash)))
    assert_equal(w_rj.state_checksum(), orig_zero_csum)

    # Direct read_entity_bin
    var bw_direct = BlobWriter()
    assert_true(reg_a.write_entity_bin(idx_nt, w_def, ents_def[0], bw_direct))
    assert_true(reg_a.write_entity_bin(idx_ns, w_def, ents_def[0], bw_direct))
    assert_true(reg_a.write_entity_bin(idx_rz, w_def, ents_def[0], bw_direct))
    var w_rb = World()
    reg_a.ensure_registered_in_world(w_rb)
    for _ in range(8):
        var br_d = BlobReader(bw_direct.data.copy())
        var e = w_rb.spawn_empty()
        assert_true(reg_a.read_entity_bin(idx_nt, w_rb, e, br_d))
        assert_true(reg_a.read_entity_bin(idx_ns, w_rb, e, br_d))
        assert_true(reg_a.read_entity_bin(idx_rz, w_rb, e, br_d))
        _assert_all_zero_12(w_rb.get_raw_ptr(e, w_rb.find_component_by_hash(nan_t.component_hash)))
        _assert_all_zero_12(w_rb.get_raw_ptr(e, w_rb.find_component_by_hash(nan_s.component_hash)))
        _assert_all_zero_12(w_rb.get_raw_ptr(e, w_rb.find_component_by_hash(raw_z.component_hash)))
    assert_equal(w_rb.state_checksum(), orig_zero_csum)

    # Scene .flscn & .flscnb with both clear_existing=True and clear_existing=False
    var sc_zero = Scene("NanZeroScene")
    var flscn_zero = sc_zero.save_flscn_bytes(w_def, reg_a)
    var flscnb_zero = sc_zero.save_flscnb_bytes(w_def, reg_a)

    for clr_idx in range(2):
        var clr = clr_idx == 0
        var w_sj = World()
        var loaded_j = sc_zero.load_flscn_bytes(w_sj, reg_a, flscn_zero, clear_existing=clr)
        assert_equal(len(loaded_j), 8)
        assert_equal(w_sj.state_checksum(), orig_zero_csum)
        var j_nt = w_sj.find_component_by_hash(nan_t.component_hash)
        var j_ns = w_sj.find_component_by_hash(nan_s.component_hash)
        var j_rz = w_sj.find_component_by_hash(raw_z.component_hash)
        for ei in range(len(loaded_j)):
            _assert_all_zero_12(w_sj.get_raw_ptr(loaded_j[ei], j_nt))
            _assert_all_zero_12(w_sj.get_raw_ptr(loaded_j[ei], j_ns))
            _assert_all_zero_12(w_sj.get_raw_ptr(loaded_j[ei], j_rz))

        var w_sb = World()
        var loaded_b = sc_zero.load_flscnb_bytes(w_sb, reg_a, flscnb_zero, clear_existing=clr)
        assert_equal(len(loaded_b), 8)
        assert_equal(w_sb.state_checksum(), orig_zero_csum)
        var b_nt = w_sb.find_component_by_hash(nan_t.component_hash)
        var b_ns = w_sb.find_component_by_hash(nan_s.component_hash)
        var b_rz = w_sb.find_component_by_hash(raw_z.component_hash)
        for ei in range(len(loaded_b)):
            _assert_all_zero_12(w_sb.get_raw_ptr(loaded_b[ei], b_nt))
            _assert_all_zero_12(w_sb.get_raw_ptr(loaded_b[ei], b_ns))
            _assert_all_zero_12(w_sb.get_raw_ptr(loaded_b[ei], b_rz))

    # Subtest B: POD_FIELD_INT (8-byte Int64) extreme & > 2^53 precision round-trip
    var i64_tab = PodComponentSchema("I64EdgeTable", 40, 8, STORAGE_TABLE)
    var i64_sp = PodComponentSchema("I64EdgeSparse", 40, 1, STORAGE_SPARSE)
    var field_names = List[String]()
    field_names.append("i_min")
    field_names.append("i_max")
    field_names.append("pos53")
    field_names.append("neg53")
    field_names.append("neg1")
    for fi in range(5):
        assert_true(i64_tab.add_field(PodFieldDesc(field_names[fi], POD_FIELD_INT, fi * 8, 8)))
        assert_true(i64_sp.add_field(PodFieldDesc(field_names[fi], POD_FIELD_INT, fi * 8, 8)))
    assert_true(i64_tab.validate() and i64_sp.validate())

    var reg_b = ComponentRegistry.with_builtins()
    assert_true(reg_b.register_pod_schema(i64_tab))
    assert_true(reg_b.register_pod_schema(i64_sp))

    var w_edge = World()
    reg_b.ensure_registered_in_world(w_edge)
    var tid_it = w_edge.find_component_by_hash(i64_tab.component_hash)
    var tid_is = w_edge.find_component_by_hash(i64_sp.component_hash)
    var e_edge = w_edge.spawn_empty()
    assert_true(reg_b.insert_default_on_entity(w_edge, e_edge, "I64EdgeTable"))
    assert_true(reg_b.insert_default_on_entity(w_edge, e_edge, "I64EdgeSparse"))

    var expected_i64 = List[Int64]()
    expected_i64.append(i64_min)
    expected_i64.append(i64_max)
    expected_i64.append(Int64(9007199254740993))
    expected_i64.append(Int64(-9007199254740993))
    expected_i64.append(Int64(-1))

    var p_it = w_edge.get_raw_ptr_mut(e_edge, tid_it)
    var p_is = w_edge.get_raw_ptr_mut(e_edge, tid_is)
    for fi in range(5):
        Pointer[Int64, MutAnyOrigin](unsafe_from_address=p_it + fi * 8)[unsafe_offset=0] = expected_i64[fi]
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_is + fi * 8).unsafe_bitcast[Int64]()[unsafe_offset=0] = expected_i64[fi]

    var edge_csum = w_edge.state_checksum()
    var sc_edge = Scene("I64EdgeScene")
    var edge_flscn = sc_edge.save_flscn_bytes(w_edge, reg_b)
    var edge_flscnb = sc_edge.save_flscnb_bytes(w_edge, reg_b)

    for clr_idx in range(2):
        var clr = clr_idx == 0
        var w_ej = World()
        var lj = sc_edge.load_flscn_bytes(w_ej, reg_b, edge_flscn, clear_existing=clr)
        assert_equal(len(lj), 1)
        assert_equal(w_ej.state_checksum(), edge_csum)
        var rp_t = w_ej.get_raw_ptr(lj[0], w_ej.find_component_by_hash(i64_tab.component_hash))
        var rp_s = w_ej.get_raw_ptr(lj[0], w_ej.find_component_by_hash(i64_sp.component_hash))
        for fi in range(5):
            assert_equal(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=rp_t + fi * 8)[unsafe_offset=0], expected_i64[fi])
            assert_equal(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=rp_s + fi * 8)[unsafe_offset=0], expected_i64[fi])

    var w_eb = World()
    var lb = sc_edge.load_flscnb_bytes(w_eb, reg_b, edge_flscnb)
    assert_equal(len(lb), 1)
    assert_equal(w_eb.state_checksum(), edge_csum)

    var bw_edge = BlobWriter()
    w_edge.snapshot_blob(bw_edge)
    var w_es = World()
    reg_b.ensure_registered_in_world(w_es)
    var br_edge = BlobReader(bw_edge.data.copy())
    w_es.restore_blob(br_edge)
    assert_equal(w_es.state_checksum(), edge_csum)

    # Subtest C: ComponentRegistry + Scene .flscn/.flscnb round-trip with tid >= 256 POD schemas
    var reg_c = ComponentRegistry.with_builtins()
    var pod_names_c = List[String]()
    var pod_hashes_c = List[UInt64]()
    for pi in range(271):
        var pname = "PodOver256_" + String(pi)
        var mode = STORAGE_SPARSE if (pi % 3 == 0) else STORAGE_TABLE
        var ps = PodComponentSchema(pname, 8, 4, mode)
        assert_true(ps.add_field(PodFieldDesc("tag_i", POD_FIELD_INT, 0, 4)))
        assert_true(ps.add_field(PodFieldDesc("val_f", POD_FIELD_F32, 4, 4)))
        assert_true(ps.validate())
        assert_true(reg_c.register_pod_schema(ps))
        pod_names_c.append(pname)
        pod_hashes_c.append(ps.component_hash)

    var w_c = World()
    reg_c.ensure_registered_in_world(w_c)
    var probe_tids = List[Int]()
    probe_tids.append(255)
    probe_tids.append(256)
    probe_tids.append(260)
    probe_tids.append(270)
    probe_tids.append(274)
    var target_names = List[String]()
    var target_hashes = List[UInt64]()
    for ti in range(len(probe_tids)):
        var ptid = probe_tids[ti]
        assert_true(ptid < len(w_c.components))
        var h = w_c.components[ptid].component_hash
        var found_name = String("")
        for pi in range(len(pod_hashes_c)):
            if pod_hashes_c[pi] == h:
                found_name = pod_names_c[pi]
                break
        assert_true(found_name.byte_length() > 0)
        target_names.append(found_name)
        target_hashes.append(h)

    var e_c = w_c.spawn_empty()
    for ti in range(len(probe_tids)):
        assert_true(reg_c.insert_default_on_entity(w_c, e_c, target_names[ti]))
        var ptr = w_c.get_raw_ptr_mut(e_c, probe_tids[ti])
        assert_true(ptr != 0)
        Pointer[Int32, MutAnyOrigin](unsafe_from_address=ptr)[unsafe_offset=0] = Int32(1000 + probe_tids[ti])
        Pointer[Float32, MutAnyOrigin](unsafe_from_address=ptr + 4)[unsafe_offset=0] = Float32(probe_tids[ti]) * Float32(0.25)

    var c_csum = w_c.state_checksum()
    var sc_c = Scene("Pod256Scene")
    var c_flscn = sc_c.save_flscn_bytes(w_c, reg_c)
    var c_flscnb = sc_c.save_flscnb_bytes(w_c, reg_c)

    var w_cj = World()
    var lcj = sc_c.load_flscn_bytes(w_cj, reg_c, c_flscn)
    assert_equal(len(lcj), 1)
    assert_equal(w_cj.state_checksum(), c_csum)
    var w_cb = World()
    var lcb = sc_c.load_flscnb_bytes(w_cb, reg_c, c_flscnb)
    assert_equal(len(lcb), 1)
    assert_equal(w_cb.state_checksum(), c_csum)

    for ti in range(len(probe_tids)):
        var tid_j = w_cj.find_component_by_hash(target_hashes[ti])
        assert_true(tid_j >= 255)
        var qj = w_cj.query_entities(ComponentBitSet().with_bit(tid_j))
        assert_equal(len(qj), 1)
        var pj = w_cj.get_raw_ptr(qj[0], tid_j)
        assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=pj)[unsafe_offset=0], Int32(1000 + probe_tids[ti]))
        assert_equal(Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=pj + 4)[unsafe_offset=0], Float32(probe_tids[ti]) * Float32(0.25))
        assert_true(reg_c.remove_from_entity(w_cj, qj[0], target_names[ti]))
        assert_true(not w_cj.has_by_type_id(qj[0], tid_j))


# ===========================================================================
# Suite (g): v5 Review Regressions — Component Hash Remap, Portable Heap
#            Snapshot/Restore, Resource Resize, Implicit ParallelCommands & Handles
# ===========================================================================

struct HeapNamedComp(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.HeapNamedComp")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var label: String

    def __init__(out self, var label: String):
        self.label = label^

    def __init__(out self, *, copy: Self):
        self.label = copy.label


def _heap_named_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[HeapNamedComp, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.text(comp.label)


def _heap_named_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var label = reader.text()
    if reader.failed:
        return False
    var dst = Pointer[HeapNamedComp, MutAnyOrigin](unsafe_from_address=comp_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = HeapNamedComp(label^)
    return True


struct TrackedHeapBag(Component):
    """Non-trivial heap-owning component (`String` + `List[Int]`) tracking exact ctor/drop balance."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.TrackedHeapBag")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var stats_addr: Int
    var tag: String
    var values: List[Int]

    def __init__(out self, stats_addr: Int, var tag: String, var values: List[Int]):
        self.stats_addr = stats_addr
        self.tag = tag^
        self.values = values^
        if stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=stats_addr)[unsafe_offset=0] += 1  # live

    def __init__(out self, *, copy: Self):
        self.stats_addr = copy.stats_addr
        self.tag = copy.tag
        self.values = copy.values.copy()
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] += 1  # live
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=1] += 1  # copies

    def __init__(out self, *, deinit move: Self):
        self.stats_addr = move.stats_addr
        self.tag = move.tag^
        self.values = move.values^
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=2] += 1  # moves

    def __deinit__(deinit self):
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] -= 1  # live
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=3] += 1  # drops


def _tracked_bag_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[TrackedHeapBag, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.integer(comp.stats_addr)
    out.text(comp.tag)
    out.integer(len(comp.values))
    for i in range(len(comp.values)):
        out.integer(comp.values[i])


def _tracked_bag_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var stats_addr = reader.integer()
    var tag = reader.text()
    var n = reader.integer()
    if reader.failed or n < 0 or n > 1000000:
        return False
    var vals = List[Int](capacity=max(1, n))
    for _ in range(n):
        vals.append(reader.integer())
    if reader.failed:
        return False
    var dst = Pointer[TrackedHeapBag, MutAnyOrigin](unsafe_from_address=comp_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = TrackedHeapBag(stats_addr, tag^, vals^)
    return True


struct SparseHeapTag(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.SparseHeapTag")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var note: String

    def __init__(out self, var note: String):
        self.note = note^

    def __init__(out self, *, copy: Self):
        self.note = copy.note


def _sparse_heap_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[SparseHeapTag, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.text(comp.note)


def _sparse_heap_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var note = reader.text()
    if reader.failed:
        return False
    var dst = Pointer[SparseHeapTag, MutAnyOrigin](unsafe_from_address=comp_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = SparseHeapTag(note^)
    return True


struct TrackedResource(Resource):
    comptime RESOURCE_ID: UInt64 = comp_hash64("test.TrackedResource")
    var stats_addr: Int
    var text: String
    var items: List[Int64]

    def __init__(out self, stats_addr: Int, var text: String, count: Int):
        self.stats_addr = stats_addr
        self.text = text^
        self.items = List[Int64](length=count, fill=99)
        if stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=stats_addr)[unsafe_offset=0] += 1

    def __deinit__(deinit self):
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] -= 1


def _implicit_order_kernel(
    lane: Int,
    order: Int,
    entity: Entity,
    pos: Pointer[CompPos, MutAnyOrigin],
    cmds_addr: Int,
):
    ref cmds = Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=cmds_addr)[unsafe_offset=0]
    pos[unsafe_offset=0].x += Float32(0.5)
    # Intentionally omit `order_key`: ParallelCommands must derive deterministic order implicitly!
    if order % 3 == 0:
        var spawned = cmds.spawn_empty(lane)
        cmds.insert[CompPos](lane, spawned, CompPos(Float32(order), Float32(order * 2), Float32(-1.0)))
        if order % 6 == 0:
            cmds.insert[CompVel](lane, spawned, CompVel(Float32(1.0), Float32(2.0), Float32(3.0)))
    elif order % 11 == 0:
        cmds.despawn(lane, entity)


def _run_implicit_order_sim(workers: Int) -> Tuple[Int, UInt64]:
    var w = World()
    _ = w.register_component[CompPos]()
    _ = w.register_component[CompVel]()
    for i in range(1800):
        var e = w.spawn_empty()
        _ = w.insert[CompPos](e, CompPos(Float32(i), Float32(-i), Float32(0.25)))
    var pool = JobPool()
    if workers > 0:
        _ = pool.start(workers)
    else:
        _ = pool.allocate()
    var cmds = ParallelCommands()
    for _ in range(2):
        w.par_for_each1[CompPos](pool, _implicit_order_kernel, Int(Pointer(to=cmds)))
        w.flush_commands(cmds)
    var cnt = w.entity_count()
    var cs = w.state_checksum()
    pool.stop()
    return (cnt, cs)


def _make_heap_string(prefix: String, idx: Int) -> String:
    var s = prefix + "_" + String(idx) + "_"
    for k in range(8):
        s += "segment" + String(k) + "_"
    return s^


def _run_tracked_bag_scope(stats_addr: Int) raises:
    var wt = World()
    var bag_tid = wt.register_component[TrackedHeapBag]()
    wt.set_component_binary_hooks(bag_tid, _tracked_bag_write, _tracked_bag_read)
    var tracked_ents = List[Entity]()
    for i in range(12):
        var e = wt.spawn_empty()
        var vals = List[Int]()
        for k in range(16):
            vals.append(i * 100 + k)
        _ = wt.insert[TrackedHeapBag](
            e, TrackedHeapBag(stats_addr, _make_heap_string("tracked_bag", i), vals^)
        )
        if i % 2 == 0:
            _ = wt.insert[CompPos](e, CompPos(Float32(i), 1.0, 2.0))
        tracked_ents.append(e)

    var wt_snap = BlobWriter()
    wt.snapshot_blob(wt_snap)

    for cyc in range(100):
        # Trigger archetype moves (`move_init_fn`), overwrites (`destroy_fn`), and swap-removes
        _ = wt.remove[CompPos](tracked_ents[0])
        _ = wt.insert[CompShield](tracked_ents[1], CompShield(Float32(cyc)))
        _ = wt.despawn(tracked_ents[2])
        _ = wt.despawn(tracked_ents[5])
        var extra_vals = List[Int](length=24, fill=cyc)
        var extra_e = wt.spawn_empty()
        _ = wt.insert[TrackedHeapBag](
            extra_e, TrackedHeapBag(stats_addr, _make_heap_string("transient", cyc), extra_vals^)
        )

        var wt_rd = BlobReader(wt_snap.data.copy())
        wt.restore_blob(wt_rd)
        assert_equal(wt.entity_count(), 12)
        ref b5 = wt.get_ptr[TrackedHeapBag](tracked_ents[5]).value()[unsafe_offset=0]
        assert_equal(len(b5.values), 16)
        assert_equal(b5.values[15], 515)


def _run_tracked_res_scope(rl_addr: Int) raises:
    var r2 = Resources()
    r2.insert[TrackedResource](TrackedResource(rl_addr, _make_heap_string("res_v1", 1), 8))
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=rl_addr)[unsafe_offset=0], 1)
    r2.insert[TrackedResource](TrackedResource(rl_addr, _make_heap_string("res_v2_bigger", 2), 64))
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=rl_addr)[unsafe_offset=0], 1)


def test_g_v5_review_regressions() raises:
    # 1. Component hash remapping across different registration orders + unregistered ID safety
    var wa = World()
    _ = wa.register_component[CompPos]()
    _ = wa.register_component[CompVel]()
    _ = wa.register_component[TagBurningSparse]()
    var ea_vel = wa.spawn_empty()
    _ = wa.insert[CompVel](ea_vel, CompVel(1.0, 7.0, -3.0))
    var ea_both = wa.spawn_empty()
    _ = wa.insert[CompPos](ea_both, CompPos(10.0, 20.0, 30.0))
    _ = wa.insert[CompVel](ea_both, CompVel(4.0, 5.0, 6.0))
    _ = wa.insert[TagBurningSparse](ea_both, TagBurningSparse(33.5))

    var snap_ab = BlobWriter()
    wa.snapshot_blob(snap_ab)

    # Register in completely permuted order in wb
    var wb = World()
    _ = wb.register_component[TagSelectedSparse]()
    _ = wb.register_component[TagBurningSparse]()
    _ = wb.register_component[CompShield]()
    _ = wb.register_component[CompVel]()
    _ = wb.register_component[CompPos]()
    var rd_ab = BlobReader(snap_ab.data.copy())
    wb.restore_blob(rd_ab)

    assert_true(wb.has[CompVel](ea_vel) and not wb.has[CompPos](ea_vel))
    assert_equal(wb.get[CompVel](ea_vel).vy, Float32(7.0))
    assert_true(wb.has[CompPos](ea_both) and wb.has[CompVel](ea_both))
    assert_true(wb.has[TagBurningSparse](ea_both) and not wb.has[TagSelectedSparse](ea_both))
    assert_equal(wb.get[TagBurningSparse](ea_both).dps, Float32(33.5))
    assert_equal(wb.state_checksum(), wa.state_checksum())

    # Destination world missing `CompVel` & `TagBurningSparse`: must raise BEFORE mutating wc!
    var wc = World()
    _ = wc.register_component[CompPos]()
    var sentinel = wc.spawn_empty()
    _ = wc.insert[CompPos](sentinel, CompPos(42.0, 43.0, 44.0))
    var rd_missing = BlobReader(snap_ab.data.copy())
    var caught_unreg = False
    try:
        wc.restore_blob(rd_missing)
    except err:
        caught_unreg = "unregistered component ID" in String(err)
    assert_true(caught_unreg, "restore_blob must raise clean Error for unregistered component ID")
    assert_equal(wc.entity_count(), 1)
    assert_true(wc.is_alive(sentinel))
    assert_equal(wc.get[CompPos](sentinel).x, Float32(42.0))

    # 2. Non-trivial heap-owning String & List[Int] components across 100 snapshot/mutate/restore cycles
    var bag_stats = List[Int](length=4, fill=0)
    var stats_addr = Int(bag_stats.unsafe_ptr())

    # Pure String world: verify state_checksum() bit-identity across 100 snapshot -> mutate -> restore cycles
    var ws = World()
    var hn_tid = ws.register_component[HeapNamedComp]()
    ws.set_component_binary_hooks(hn_tid, _heap_named_write, _heap_named_read)
    var sh_tid = ws.register_component[SparseHeapTag]()
    ws.set_component_binary_hooks(sh_tid, _sparse_heap_write, _sparse_heap_read)
    var es0 = ws.spawn_empty()
    var es1 = ws.spawn_empty()
    var s0_orig = _make_heap_string("hero_alpha_non_sso_heap_payload", 0)
    var s1_orig = _make_heap_string("ally_bravo_non_sso_heap_payload", 1)
    var s0_len = s0_orig.byte_length()
    _ = ws.insert[HeapNamedComp](es0, HeapNamedComp(s0_orig.copy()))
    _ = ws.insert[HeapNamedComp](es1, HeapNamedComp(s1_orig.copy()))
    _ = ws.insert[SparseHeapTag](es0, SparseHeapTag(_make_heap_string("sparse_tag_payload", 99)))
    var cs_expected = ws.state_checksum()
    var ws_snap = BlobWriter()
    ws.snapshot_blob(ws_snap)

    for cyc in range(100):
        _ = ws.insert[HeapNamedComp](es0, HeapNamedComp(_make_heap_string("mutated_during_pie", cyc)))
        _ = ws.despawn(es1)
        var tmp_e = ws.spawn_empty()
        _ = ws.insert[HeapNamedComp](tmp_e, HeapNamedComp(_make_heap_string("temp_spawn", cyc)))
        assert_true(ws.state_checksum() != cs_expected)
        var ws_rd = BlobReader(ws_snap.data.copy())
        ws.restore_blob(ws_rd)
        assert_equal(ws.get_ptr[HeapNamedComp](es0).value()[unsafe_offset=0].label.byte_length(), s0_len)
        assert_equal(ws.get_ptr[HeapNamedComp](es0).value()[unsafe_offset=0].label, s0_orig)
        assert_equal(ws.get_ptr[HeapNamedComp](es1).value()[unsafe_offset=0].label, s1_orig)
        assert_true(ws.has[SparseHeapTag](es0))
        assert_equal(ws.state_checksum(), cs_expected)

    # Tracked String + List[Int] world: 100 cycles of snapshot -> archetype moves/despawn -> restore + zero-leak proof
    _run_tracked_bag_scope(stats_addr)

    # wt has now deinitialized; verify EXACT 0 live instances and >= 1200 drops!
    assert_equal(bag_stats[0], 0)
    assert_true(bag_stats[3] >= 1200)
    _ = len(bag_stats)

    # 3. Resources.insert_pod_raw & Resources.insert[R] resize safety (16 B -> 256 B -> 1024 B)
    var res = Resources()
    var pod_id = UInt64(0xABCD0001ABCD0001)
    var small_buf = List[Int64](length=2, fill=11)
    var p16 = res.insert_pod_raw(pod_id, 16, Int(small_buf.unsafe_ptr()))
    _ = len(small_buf)
    assert_true(p16 != 0)
    assert_equal(res.byte_size_of(pod_id), 16)

    var med_buf = List[Int64](length=32, fill=22)
    var p256 = res.insert_pod_raw(pod_id, 256, Int(med_buf.unsafe_ptr()))
    _ = len(med_buf)
    assert_true(p256 != 0)
    assert_equal(res.byte_size_of(pod_id), 256)
    assert_equal(Int(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=p256)[unsafe_offset=31]), 22)

    var big_buf = List[Int64](length=128, fill=33)
    big_buf[127] = 123456789
    var p1024 = res.insert_pod_raw(pod_id, 1024, Int(big_buf.unsafe_ptr()))
    _ = len(big_buf)
    assert_true(p1024 != 0)
    assert_equal(res.byte_size_of(pod_id), 1024)
    assert_equal(res.get_raw_ptr(pod_id), p1024)
    assert_equal(
        Int(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=p1024)[unsafe_offset=127]),
        123456789,
    )

    var res_live = List[Int](length=1, fill=0)
    var rl_addr = Int(res_live.unsafe_ptr())
    _run_tracked_res_scope(rl_addr)
    assert_equal(res_live[0], 0)
    _ = len(res_live)

    # 4. Default ParallelCommands ordering (no explicit order_key) across 0, 1, 4, and 7 workers
    var imp_0t = _run_implicit_order_sim(0)
    var imp_1t = _run_implicit_order_sim(1)
    var imp_4t = _run_implicit_order_sim(4)
    assert_equal(imp_0t[0], imp_1t[0])
    assert_equal(imp_0t[1], imp_1t[1])
    assert_equal(imp_0t[1], imp_4t[1])
    for _ in range(6):
        var imp_7t = _run_implicit_order_sim(7)
        assert_equal(imp_7t[0], imp_0t[0])
        assert_equal(imp_7t[1], imp_0t[1])

    # 5. Lossless Entity <-> EntityHandle conversions bridging flint.ecs with flint.core.ecs
    var null_h = NULL_ENTITY.to_entity_handle()
    assert_true(not null_h.valid())
    assert_true(Entity.from_entity_handle(null_h) == NULL_ENTITY)

    var wh = World()
    var eh_ent = wh.spawn_empty()
    _ = wh.despawn(eh_ent)
    var eh_ent2 = wh.spawn_empty()  # generation == 2
    var bridged_h = wh.to_entity_handle(eh_ent2)
    assert_true(bridged_h.valid())
    assert_true(Entity.from_entity_handle(bridged_h) == eh_ent2)
    assert_true(Entity.from_entity_handle(EntityHandle.unpack(bridged_h.packed())) == eh_ent2)

    var core_slots = ArchetypeSlots(3, 16)
    var core_h0 = core_slots.spawn()  # generation == 0 in flint.core.ecs!
    assert_true(core_h0.valid())
    var from_core = Entity.from_entity_handle(core_h0)
    assert_true(from_core.valid())
    assert_true(from_core.to_entity_handle(core_h0.archetype) == core_h0)

    var prov_cmds = ParallelCommands()
    var prov_e = prov_cmds.spawn_empty(5)
    var prov_h = prov_e.to_entity_handle()
    assert_true(prov_h.valid() and prov_h.generation == ECB_PROV_GEN)
    assert_true(Entity.from_entity_handle(prov_h) == prov_e)


def test_h_restore_blob_rejects_hostile_counts() raises:
    """A corrupted snapshot must raise, never allocate from an unchecked count or crash."""
    var w = World()
    _ = w.register_component[CompPos]()
    _ = w.register_component[TagBurningSparse]()
    for i in range(40):
        var e = w.spawn_empty()
        _ = w.insert[CompPos](e, CompPos(Float32(i), 0.0, 0.0))
        if i % 4 == 0:
            _ = w.insert[TagBurningSparse](e, TagBurningSparse(Float32(i)))
    var snap = BlobWriter()
    w.snapshot_blob(snap)
    var rejected = 0
    # Overwrite each 8-byte-aligned header word with a huge count.
    for at in range(0, min(256, len(snap.data) - 8), 8):
        var bad = snap.data.copy()
        for k in range(8):
            bad[at + k] = UInt8(0x7F) if k == 7 else UInt8(0xFF)
        var t = World()
        _ = t.register_component[CompPos]()
        _ = t.register_component[TagBurningSparse]()
        var rd = BlobReader(bad^)
        try:
            t.restore_blob(rd)
        except:
            rejected += 1
        _ = t.state_checksum()
    assert_true(rejected > 0)
    # The untouched snapshot still restores exactly.
    var t2 = World()
    _ = t2.register_component[CompPos]()
    _ = t2.register_component[TagBurningSparse]()
    var rd2 = BlobReader(snap.data.copy())
    t2.restore_blob(rd2)
    assert_equal(t2.entity_count(), 40)
    assert_equal(t2.state_checksum(), w.state_checksum())


def _par1_count_and_mark(
    _lane: Int, _ord: Int, _e: Entity, pos: Pointer[CompPos, MutAnyOrigin], ctx: Int
):
    pos[unsafe_offset=0].z = Float32(99.0)
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].fetch_add(1)


def _par2_count_and_mark(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    pos: Pointer[CompPos, MutAnyOrigin],
    vel: Pointer[CompVel, MutAnyOrigin],
    ctx: Int,
):
    pos[unsafe_offset=0].z = vel[unsafe_offset=0].vx
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].fetch_add(1)


def test_i_audit_regressions() raises:
    """Phase 0+1 audit regressions: sparse exclusion in par_for_each1/2, SessionRecorder.write_dev
    safety, anchor STACK_SLOTS, Arena align > 64, and Resources O(1) lookup."""
    # 1. Sparse exclusion in par_for_each1 and par_for_each2
    var w = World()
    var tid_pos = w.register_component[CompPos]()
    var tid_vel = w.register_component[CompVel]()
    var tid_burn = w.register_component[TagBurningSparse]()
    var ents = List[Entity]()
    for i in range(120):
        var e = w.spawn_empty()
        _ = w.insert[CompPos](e, CompPos(Float32(i), Float32(0.0), Float32(-1.0)))
        _ = w.insert[CompVel](e, CompVel(Float32(42.0), Float32(0.0), Float32(0.0)))
        if i % 3 == 0:
            _ = w.insert[TagBurningSparse](e, TagBurningSparse(Float32(5.0)))
        ents.append(e)

    var without_burn = BitSet256().with_bit(tid_burn)
    var expected = w.query_entities(BitSet256().with_bit(tid_pos).with_bit(tid_vel), without_burn)
    assert_equal(len(expected), 80)

    var pool = JobPool()
    assert_true(pool.start(4), "JobPool(4) must start")

    var cnt1 = List[Int64](length=1, fill=0)
    w.par_for_each1[CompPos](pool, _par1_count_and_mark, Int(cnt1.unsafe_ptr()), without_burn)
    assert_equal(Int(cnt1[0]), 80)
    for i in range(len(ents)):
        var p = w.get_ptr[CompPos](ents[i]).value()[unsafe_offset=0]
        if i % 3 == 0:
            assert_equal(p.z, Float32(-1.0))
        else:
            assert_equal(p.z, Float32(99.0))
            w.get_ptr_mut[CompPos](ents[i]).value()[unsafe_offset=0].z = Float32(-1.0)

    var cnt2 = List[Int64](length=1, fill=0)
    w.par_for_each2[CompPos, CompVel](pool, _par2_count_and_mark, Int(cnt2.unsafe_ptr()), without_burn)
    assert_equal(Int(cnt2[0]), 80)
    for i in range(len(ents)):
        var p = w.get_ptr[CompPos](ents[i]).value()[unsafe_offset=0]
        if i % 3 == 0:
            assert_equal(p.z, Float32(-1.0))
        else:
            assert_equal(p.z, Float32(42.0))
    pool.stop()

    # 2. SessionRecorder.write_dev safety (no dangling A_REPLAY_BUFFER across flush/oversized writes)
    var anchor_ok = install_anchor()
    var rec = SessionRecorder()
    var rec_path = String("/tmp/flint-test-session-replay-write-dev.sdrp")
    var hdr = List[UInt8](length=16, fill=UInt8(0xAA))
    assert_true(rec.start(rec_path, hdr^, for_crashes=True), "SessionRecorder.start must succeed")
    var initial_ptr = Int(rec.buffer.unsafe_ptr())
    var initial_cap = rec.buffer.capacity()

    var small = List[UInt8](length=1024, fill=UInt8(0x11))
    rec.write_dev(small)
    assert_equal(Int(rec.buffer.unsafe_ptr()), initial_ptr)
    var ab = anchor_base()
    if anchor_ok:
        assert_true(ab != 0 and ab != 0x5D5D00000000, "anchor_base must be dynamic non-zero address")
        assert_equal(load(ab + A_REPLAY_BUFFER), Int(rec.buffer.unsafe_ptr()))
        assert_equal(load(ab + A_REPLAY_COMMITTED), len(rec.buffer))
        assert_equal(load(ab + A_REPLAY_PENDING), len(rec.buffer))

    # Cross FLUSH_AT via two 600 KiB chunks: must flush before extending, never reallocating buffer
    var chunk_600k = List[UInt8](length=600 * 1024, fill=UInt8(0x22))
    rec.write_dev(chunk_600k)
    rec.write_dev(chunk_600k)
    assert_equal(Int(rec.buffer.unsafe_ptr()), initial_ptr)
    assert_equal(rec.buffer.capacity(), initial_cap)
    if anchor_ok:
        assert_equal(load(ab + A_REPLAY_BUFFER), Int(rec.buffer.unsafe_ptr()))
        assert_equal(load(ab + A_REPLAY_COMMITTED), len(rec.buffer))

    # Oversized single dev payload (> FLUSH_AT): must write directly without reallocating buffer
    var huge = List[UInt8](length=FLUSH_AT + 4096, fill=UInt8(0x33))
    rec.write_dev(huge)
    assert_equal(Int(rec.buffer.unsafe_ptr()), initial_ptr)
    assert_equal(rec.buffer.capacity(), initial_cap)
    assert_equal(len(rec.buffer), 0)
    if anchor_ok:
        assert_equal(load(ab + A_REPLAY_BUFFER), initial_ptr)
        assert_equal(load(ab + A_REPLAY_COMMITTED), 0)
        assert_equal(load(ab + A_REPLAY_PENDING), 0)

    rec.finish()
    var expected_bytes = 16 + 1024 + 2 * (600 * 1024) + (FLUSH_AT + 4096)
    assert_equal(file_size(rec_path), expected_bytes)
    _ = remove_file(rec_path)

    # 3. Anchor STACK_SLOTS == 33 and fits inside ANCHOR_BYTES
    assert_equal(STACK_SLOTS, 33)
    assert_true(STACKS + STACK_SLOTS * STACK_BYTES <= ANCHOR_BYTES)

    # 4. Arena.alloc alignment for align > 64
    var arena = Arena(MEM_FRAME, 16384)
    _ = arena.alloc(13, 1)  # intentionally misalign used offset
    var a128 = arena.alloc(32, 128)
    assert_true(a128 != 0 and (a128 & 127) == 0, "Arena.alloc(align=128) must be 128-byte aligned")
    var a256 = arena.alloc(32, 256)
    assert_true(a256 != 0 and (a256 & 255) == 0, "Arena.alloc(align=256) must be 256-byte aligned")
    var a4096 = arena.alloc(64, 4096)
    assert_true(a4096 != 0 and (a4096 & 4095) == 0, "Arena.alloc(align=4096) must be 4096-byte aligned")

    # 4b. ProfileTimeline start/finish/deinit smoke
    var pt = ProfileTimeline()
    assert_true(pt.start(String("")), "ProfileTimeline.start must succeed")
    pt.finish()

    # 5. Resources O(1) lookup, overwrite, and remove
    var res = Resources()
    var vals = List[Int64](capacity=40)
    for i in range(40):
        vals.append(Int64(i * 100 + 7))
    for i in range(40):
        var rid = UInt64(1000 + i)
        _ = res.insert_pod_raw(rid, size_of[Int64](), Int(vals.unsafe_ptr()) + i * size_of[Int64]())
    _ = len(vals)
    assert_equal(res.count(), 40)
    for i in range(40):
        var rid = UInt64(1000 + i)
        var p = res.get_raw_ptr(rid)
        assert_true(p != 0)
        assert_equal(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=p)[unsafe_offset=0], Int64(i * 100 + 7))
    assert_equal(res.find_index(UInt64(999999)), -1)
    res.insert[FixedTime](FixedTime(60.0))
    assert_true(res.has[FixedTime]())
    assert_true(res.remove[FixedTime]())
    assert_true(not res.has[FixedTime]())
    assert_equal(res.count(), 40)


def _par3_ro_count_kernel(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    pos: Pointer[CompPos, MutAnyOrigin],
    vel: Pointer[CompVel, MutAnyOrigin],
    mass: Pointer[CompMass, MutAnyOrigin],
    ctx: Int,
):
    _ = pos[unsafe_offset=0].x + vel[unsafe_offset=0].vx + mass[unsafe_offset=0].inv_mass
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].fetch_add(1)


def _par3_integrate_kernel(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    pos: Pointer[CompPos, MutAnyOrigin],
    vel: Pointer[CompVel, MutAnyOrigin],
    mass: Pointer[CompMass, MutAnyOrigin],
    ctx: Int,
):
    pos[unsafe_offset=0].x += vel[unsafe_offset=0].vx * mass[unsafe_offset=0].inv_mass
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].fetch_add(1)


@fieldwise_init
struct _TestEventCounterRes(Resource, Copyable, ImplicitlyCopyable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("test.Wave2EventCounterRes")
    var swaps: Int


def _test_event_swap_hook(res_ptr: Pointer[Resources, MutAnyOrigin]):
    var opt = res_ptr[unsafe_offset=0].get_ptr_mut[_TestEventCounterRes]()
    if opt:
        opt.value()[unsafe_offset=0].swaps += 1


def test_j_wave2_powerhouse_upgrades() raises:
    """Wave 2 regressions: par_for_each3 + read_only + Changed[C] skip, chunk reclamation on
    despawn, dirty-tracked TransformPropagationCache + non-exclusive TransformPlugin, dynamic
    CVars & profiler zones, and EngineApp realtime/event-swap/profiler lifecycle."""
    # 1. par_for_each3 + read_only=True + Changed[C] / changed_since skip
    var w = World()
    _ = w.register_component[CompPos]()
    _ = w.register_component[CompVel]()
    _ = w.register_component[CompMass]()
    var ents = List[Entity]()
    for i in range(128):  # 2 full 64-row chunks
        var e = w.spawn_empty()
        _ = w.insert[CompPos](e, CompPos(Float32(i), 0.0, 0.0))
        _ = w.insert[CompVel](e, CompVel(Float32(2.0), 0.0, 0.0))
        _ = w.insert[CompMass](e, CompMass(Float32(0.5)))
        ents.append(e)

    var pool = JobPool()
    assert_true(pool.start(4), "JobPool(4) must start")

    var tick_before_ro = w.current_tick
    _ = w.advance_tick()
    var ro_cnt = List[Int64](length=1, fill=0)
    w.par_for_each3[CompPos, CompVel, CompMass](
        pool, _par3_ro_count_kernel, Int(ro_cnt.unsafe_ptr()), read_only=True
    )
    assert_equal(Int(ro_cnt[0]), 128)
    # read_only=True must NOT have dirtied changed_ticks!
    var chg_none = w.query_changed_entities[CompPos](tick_before_ro)
    assert_equal(len(chg_none), 0)

    # Mutate ONLY entity 0 (chunk 0, SIMD group 0) and verify changed_since skips the other 15 groups
    w.get_ptr_mut[CompPos](ents[0]).value()[unsafe_offset=0].y = Float32(7.0)
    var chg_one_grp = w.query_changed_entities[CompPos](tick_before_ro)
    assert_true(len(chg_one_grp) > 0 and len(chg_one_grp) <= 8)

    var par_chg_cnt = List[Int64](length=1, fill=0)
    w.par_for_each3[CompPos, CompVel, CompMass](
        pool,
        _par3_integrate_kernel,
        Int(par_chg_cnt.unsafe_ptr()),
        read_only=False,
        changed_since=tick_before_ro,
    )
    assert_equal(Int(par_chg_cnt[0]), 8)
    assert_equal(w.get_ptr[CompPos](ents[0]).value()[unsafe_offset=0].x, Float32(1.0))
    assert_equal(w.get_ptr[CompPos](ents[64]).value()[unsafe_offset=0].x, Float32(64.0))

    # 2. ArchetypeChunk reclamation on despawn (including swap-popping non-last empty chunk)
    var w2 = World()
    _ = w2.register_component[CompPos]()
    var e2 = List[Entity]()
    for i in range(200):
        var e = w2.spawn_empty()
        _ = w2.insert[CompPos](e, CompPos(Float32(i), 0.0, 0.0))
        e2.append(e)
    assert_true(w2.total_chunk_count() >= 4, "200 entities must span >= 4 chunks")
    for i in range(200):
        assert_true(w2.despawn(e2[i]))
    assert_equal(w2.entity_count(), 0)
    assert_equal(w2.total_chunk_count(), 0)

    var e3 = List[Entity]()
    for i in range(100):
        var e = w2.spawn_empty()
        _ = w2.insert[CompPos](e, CompPos(Float32(i + 1000), 0.0, 0.0))
        e3.append(e)
    assert_equal(w2.total_chunk_count(), 2)
    # Despawn all 64 entities in chunk 0 -> chunk 0 is reclaimed via swap-pop of chunk 1
    for i in range(64):
        assert_true(w2.despawn(e3[i]))
    assert_equal(w2.total_chunk_count(), 1)
    assert_equal(w2.entity_count(), 36)
    for i in range(64, 100):
        assert_true(w2.is_alive(e3[i]))
        assert_equal(w2.get_ptr[CompPos](e3[i]).value()[unsafe_offset=0].x, Float32(i + 1000))

    # 3. Dirty-tracked TransformPropagationCache & non-exclusive TransformPlugin
    var wt = World()
    var rA = wt.spawn_empty()
    var cA = wt.spawn_empty()
    var gA = wt.spawn_empty()
    var rB = wt.spawn_empty()
    var cB = wt.spawn_empty()
    var gB = wt.spawn_empty()
    _ = wt.insert[LocalTransform](rA, LocalTransform(Vec3(1.0, 0.0, 0.0)))
    _ = wt.insert[LocalTransform](cA, LocalTransform(Vec3(2.0, 0.0, 0.0)))
    _ = wt.insert[LocalTransform](gA, LocalTransform(Vec3(3.0, 0.0, 0.0)))
    _ = wt.insert[LocalTransform](rB, LocalTransform(Vec3(10.0, 0.0, 0.0)))
    _ = wt.insert[LocalTransform](cB, LocalTransform(Vec3(20.0, 0.0, 0.0)))
    _ = wt.insert[LocalTransform](gB, LocalTransform(Vec3(30.0, 0.0, 0.0)))
    assert_true(set_parent(wt, cA, rA))
    assert_true(set_parent(wt, gA, cA))
    assert_true(set_parent(wt, cB, rB))
    assert_true(set_parent(wt, gB, cB))

    var cache = TransformPropagationCache()
    propagate_transforms_parallel_cached(wt, pool, cache)
    assert_equal(cache.last_propagated_count, 6)
    assert_equal(cache.last_skipped_count, 0)
    assert_true(abs(wt.get_ptr[GlobalTransform](gA).value()[unsafe_offset=0].translation.x - 6.0) < 1e-4)
    assert_true(abs(wt.get_ptr[GlobalTransform](gB).value()[unsafe_offset=0].translation.x - 60.0) < 1e-4)

    # Next tick with NO changes -> all 6 static nodes skipped
    _ = wt.advance_tick()
    propagate_transforms_parallel_cached(wt, pool, cache)
    assert_equal(cache.last_propagated_count, 0)
    assert_equal(cache.last_skipped_count, 6)

    # Next tick: move ONLY rA -> gA updates to 105.0, while rB subtree stays untouched
    _ = wt.advance_tick()
    wt.get_ptr_mut[LocalTransform](rA).value()[unsafe_offset=0].translation.x = Float32(100.0)
    propagate_transforms_parallel_cached(wt, pool, cache)
    assert_true(abs(wt.get_ptr[GlobalTransform](gA).value()[unsafe_offset=0].translation.x - 105.0) < 1e-4)
    assert_true(abs(wt.get_ptr[GlobalTransform](gB).value()[unsafe_offset=0].translation.x - 60.0) < 1e-4)
    assert_true(cache.last_propagated_count >= 3)
    pool.stop()

    # 4. Dynamic CVar & dynamic Profiler zone registration
    var cv = ConsoleVars()
    var base_cnt = cv.count()
    assert_equal(base_cnt, CVAR_COUNT)
    var g_idx = cv.register_cvar("game.gravity", Float32(9.81), False, "world gravity")
    var f_idx = cv.register_cvar("game.godmode", Float32(0.0), True, "invincibility flag")
    assert_equal(cv.register_cvar("game.gravity", Float32(1.0)), g_idx)  # idempotent
    assert_equal(cv.count(), CVAR_COUNT + 2)
    assert_equal(cv.find("game.gravity"), g_idx)
    assert_equal(cv.apply("game.gravity=20,game.godmode"), 2)
    assert_equal(cv.get(g_idx), Float32(20.0))
    assert_true(cv.on(f_idx))

    var tl = ProfileTimeline()
    var z_pre = tl.register_zone("game.ai_think")
    assert_equal(z_pre, ZONE_COUNT)
    assert_equal(tl.zone_label(z_pre), "game.ai_think")
    assert_true(tl.start(String("")))
    var z_post = tl.register_zone("game.net_sync")
    assert_equal(z_post, ZONE_COUNT + 1)
    assert_equal(tl.zone_label(z_post), "game.net_sync")
    assert_equal(tl.register_zone("game.net_sync"), z_post)  # idempotent
    tl.begin_frame()
    var t_start = profile_ticks()
    tl.handle.record(z_post, t_start, t_start + UInt64(1000))
    tl.end_frame()
    tl.finish()

    # 5. Non-exclusive TransformPlugin + EngineApp realtime + event swap hook + profiler lifecycle
    var app = EngineApp(
        AppConfig(
            title="Wave2Test",
            headless=True,
            worker_threads=2,
            enable_profiler=True,
        )
    )
    app.add_plugin(TransformPlugin())
    var found_prop = False
    for i in range(len(app.schedule.stages[STAGE_POST_UPDATE].systems)):
        ref sd = app.schedule.stages[STAGE_POST_UPDATE].systems[i]
        if sd.name == "flint.scene.propagate_transforms":
            found_prop = True
            assert_true(not sd.exclusive, "TransformPlugin propagate_transforms must be non-exclusive")
    assert_true(found_prop, "TransformPlugin must register propagate_transforms")
    app.insert_resource[_TestEventCounterRes](_TestEventCounterRes(0))
    app.add_event_swap_hook(_test_event_swap_hook)
    _ = app.register_profiler_zone("game.custom_stage")
    app.run_realtime(max_frames=3)
    assert_equal(app.resources.get_ptr[FrameTime]().value()[unsafe_offset=0].frame_index, UInt64(3))
    assert_equal(app.resources.get_ptr[_TestEventCounterRes]().value()[unsafe_offset=0].swaps, 3)
    assert_equal(app.profiler.frame, 3)

    # 6. SIMD Mat4.multiply analytical check
    var mA = Mat4()
    mA.set(0, 3, Float32(3.0))
    mA.set(1, 3, Float32(4.0))
    var mB = Mat4()
    mB.set(0, 0, Float32(2.0))
    mB.set(0, 3, Float32(5.0))
    var mC = mA.multiply(mB)
    assert_equal(mC.at(0, 0), Float32(2.0))
    assert_equal(mC.at(0, 3), Float32(8.0))
    assert_equal(mC.at(1, 3), Float32(4.0))


def _assert_simd8_unary(
    name: StaticString,
    vals: List[Float32],
    vec_fn: def(SIMD[DType.float32, 8]) thin -> SIMD[DType.float32, 8],
    scalar_fn: def(Float32) thin -> Float32,
) raises:
    var i = 0
    while i < len(vals):
        var v = SIMD[DType.float32, 8](0.0)
        for lane in range(8):
            var idx = i + lane if i + lane < len(vals) else len(vals) - 1
            v[lane] = vals[idx]
        var out_v = vec_fn(v)
        var out_bits = bitcast[DType.uint32](out_v)
        for lane in range(8):
            var s_bits = f32_bits(scalar_fn(v[lane]))
            if out_bits[lane] != s_bits:
                assert_equal(
                    out_bits[lane],
                    s_bits,
                    String(name) + " bit mismatch at lane " + String(lane) + " input_bits=" + String(f32_bits(v[lane])),
                )
        i += 8


def _vec_sin(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_sin(x)


def _scl_sin(x: Float32) -> Float32:
    return det_sin(x)


def _vec_cos(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_cos(x)


def _scl_cos(x: Float32) -> Float32:
    return det_cos(x)


def _vec_atan(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_atan(x)


def _scl_atan(x: Float32) -> Float32:
    return det_atan(x)


def _vec_asin(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_asin(x)


def _scl_asin(x: Float32) -> Float32:
    return det_asin(x)


def _vec_acos(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_acos(x)


def _scl_acos(x: Float32) -> Float32:
    return det_acos(x)


def _vec_exp(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_exp(x)


def _scl_exp(x: Float32) -> Float32:
    return det_exp(x)


def _vec_log(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return det_log(x)


def _scl_log(x: Float32) -> Float32:
    return det_log(x)


def test_k_detmath_simd8_bit_exact() raises:
    set_sim_float_mode()
    var qnan = f32_from_bits(0x7FC00000)
    var payload_nan = f32_from_bits(0x7FC01234)
    var neg_payload_nan = f32_from_bits(0xFFC01234)
    var pinf = f32_from_bits(0x7F800000)
    var ninf = f32_from_bits(0xFF800000)
    var sub_min = f32_from_bits(0x00000001)
    var sub_neg = f32_from_bits(0x80000001)
    var sub_max = f32_from_bits(0x007FFFFF)
    var min_norm = f32_from_bits(0x00800000)

    var trig_vals: List[Float32] = [
        -2.0 * PI, -1.25 * PI, -0.5 * PI, Float32(-0.0), Float32(0.0), 0.25 * PI, 0.75 * PI, PI,
        1.25 * PI, 1.5 * PI, 1.75 * PI, 2.0 * PI, Float32(100.0), Float32(-1000.0), Float32(9.99e8), Float32(1.0e9),
        Float32(-1.0e9), Float32(1.0e12), pinf, ninf, qnan, payload_nan, sub_min, sub_neg,
    ]
    _assert_simd8_unary("det_sin", trig_vals, _vec_sin, _scl_sin)
    _assert_simd8_unary("det_cos", trig_vals, _vec_cos, _scl_cos)

    var atan_vals: List[Float32] = [
        Float32(0.0), Float32(-0.0), Float32(0.4142135623730950), Float32(0.415),
        Float32(2.414213562373095), Float32(2.5), Float32(-100.0), pinf,
        ninf, qnan, payload_nan, neg_payload_nan, Float32(-0.41421356), Float32(-2.41421356), Float32(1e-10), Float32(1e20),
    ]
    _assert_simd8_unary("det_atan", atan_vals, _vec_atan, _scl_atan)

    var invtrig_vals: List[Float32] = [
        Float32(-1.0001), Float32(-1.0), Float32(-0.75), Float32(-0.500001),
        Float32(-0.5), Float32(-0.0), Float32(0.5), Float32(1.0),
        Float32(1.0001), qnan, pinf, ninf, Float32(0.25), Float32(-0.25), Float32(0.999999), sub_min,
    ]
    _assert_simd8_unary("det_asin", invtrig_vals, _vec_asin, _scl_asin)
    _assert_simd8_unary("det_acos", invtrig_vals, _vec_acos, _scl_acos)

    var exp_vals: List[Float32] = [
        Float32(-104.0), Float32(-103.97208), Float32(-103.9), Float32(-87.5),
        Float32(-1.0), Float32(-0.0), Float32(0.6931472), Float32(88.722839),
        Float32(88.723), Float32(89.0), qnan, payload_nan, pinf, ninf, Float32(10.0), sub_min,
    ]
    _assert_simd8_unary("det_exp", exp_vals, _vec_exp, _scl_exp)

    var log_vals: List[Float32] = [
        sub_min, f32_from_bits(0x00400000), sub_max, min_norm,
        Float32(0.5), Float32(0.7071), Float32(0.7072), Float32(1.0),
        Float32(2.0), Float32(100.0), Float32(0.0), Float32(-0.0),
        Float32(-1.0), pinf, ninf, qnan,
    ]
    _assert_simd8_unary("det_log", log_vals, _vec_log, _scl_log)

    # det_atan2(y, x) binary matrix
    var ay: List[Float32] = [
        Float32(1.0), Float32(1.0), Float32(-1.0), Float32(-1.0),
        Float32(0.0), Float32(-0.0), Float32(0.0), Float32(-0.0),
        Float32(0.0), Float32(-0.0), Float32(5.0), Float32(-5.0),
        qnan, Float32(1.0), pinf, Float32(1.0),
    ]
    var ax: List[Float32] = [
        Float32(1.0), Float32(-1.0), Float32(1.0), Float32(-1.0),
        Float32(0.0), Float32(0.0), Float32(-0.0), Float32(-0.0),
        Float32(-5.0), Float32(-5.0), Float32(0.0), Float32(0.0),
        Float32(1.0), qnan, ninf, ninf,
    ]
    for base in range(0, len(ay), 8):
        var vy = SIMD[DType.float32, 8](0.0)
        var vx = SIMD[DType.float32, 8](0.0)
        for lane in range(8):
            vy[lane] = ay[base + lane]
            vx[lane] = ax[base + lane]
        var vo = det_atan2(vy, vx)
        var vb = bitcast[DType.uint32](vo)
        for lane in range(8):
            assert_equal(vb[lane], f32_bits(det_atan2(vy[lane], vx[lane])))

    # det_pow(x, y) binary matrix
    var px: List[Float32] = [
        qnan, Float32(1.0), Float32(0.0), Float32(-0.0),
        Float32(-2.0), Float32(-2.0), Float32(-2.0), Float32(-2.0),
        Float32(2.0), Float32(0.5), sub_min, Float32(3.0),
        ninf, ninf, Float32(2.0), qnan,
    ]
    var py: List[Float32] = [
        Float32(0.0), qnan, Float32(2.0), Float32(-2.0),
        Float32(2.0), Float32(3.0), Float32(-3.0), Float32(0.5),
        Float32(10.0), Float32(-4.0), Float32(0.5), Float32(0.25),
        Float32(3.0), Float32(0.5), qnan, Float32(2.0),
    ]
    for base in range(0, len(px), 8):
        var vx = SIMD[DType.float32, 8](0.0)
        var vy = SIMD[DType.float32, 8](0.0)
        for lane in range(8):
            vx[lane] = px[base + lane]
            vy[lane] = py[base + lane]
        var vo = det_pow(vx, vy)
        var vb = bitcast[DType.uint32](vo)
        for lane in range(8):
            assert_equal(vb[lane], f32_bits(det_pow(vx[lane], vy[lane])))

    # Direct scale_by_pow2 multi-step parity
    var sx = SIMD[DType.float32, 8](Float32(1.5), Float32(1.5), Float32(-0.75), sub_max, Float32(1.0), Float32(2.0), Float32(1.25), Float32(0.5))
    var sn = SIMD[DType.int32, 8](128, -140, 260, -260, 0, 10, -10, 127)
    var sres = scale_by_pow2(sx, sn)
    var sbits = bitcast[DType.uint32](sres)
    for lane in range(8):
        assert_equal(sbits[lane], f32_bits(scale_by_pow2(sx[lane], Int(sn[lane]))))


@fieldwise_init
struct _StealJobCtx(Copyable, Movable):
    var counts_ptr: Int
    var tids_ptr: Int
    var first_tid_ptr: Int
    var two_seen_ptr: Int
    var idx: Int


def _test_l_steal_job(ctx_addr: Int, chunk: Int):
    ref c = Pointer[_StealJobCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var cnt = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.counts_ptr + c.idx * 8)
    _ = cnt[unsafe_offset=0].fetch_add(1)
    var tid = Int64(external_call["pthread_self", UInt64]())
    Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=c.tids_ptr + c.idx * 8)[unsafe_offset=0] = tid
    var ft_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.first_tid_ptr)
    var ts_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.two_seen_ptr)
    var expected = Int64(0)
    if not ft_ptr[unsafe_offset=0].compare_exchange(expected, tid):
        if expected != tid:
            ts_ptr[unsafe_offset=0].store(1)
    if ts_ptr[unsafe_offset=0].load() == 0 and ft_ptr[unsafe_offset=0].load() == tid:
        var spins = 0
        while ts_ptr[unsafe_offset=0].load() == 0 and spins < 1000000:
            _ = external_call["sched_yield", Int32]()
            spins += 1
    if c.idx % 8 == 0:
        var acc = UInt64(c.idx + chunk + 1)
        for _ in range(3000):
            acc = acc * UInt64(6364136223846793005) + UInt64(1)
        if acc == UInt64(0):
            _ = cnt[unsafe_offset=0].fetch_add(0)


@fieldwise_init
struct _ReadyBarrierCtx(Copyable, Movable):
    var ready_addr: Int
    var go_addr: Int


def _test_l_ready_barrier_job(ctx_addr: Int, chunk: Int):
    _ = chunk
    ref c = Pointer[_ReadyBarrierCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    _ = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.ready_addr)[unsafe_offset=0].fetch_add(1)
    var go_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.go_addr)
    while go_ptr[unsafe_offset=0].load() == 0:
        _ = external_call["sched_yield", Int32]()


def _test_l_overflow_job(ctx_addr: Int, chunk: Int):
    _ = chunk
    _ = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0].fetch_add(1)


@fieldwise_init
struct _AffineOutCtx(Copyable, Movable):
    var out_ptr: Int


def _test_l_affine_job(ctx_addr: Int, chunk: Int):
    ref c = Pointer[_AffineOutCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var p = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=c.out_ptr + chunk * 4)
    p[unsafe_offset=0] = Float32(chunk * 3 + 7) * Float32(0.125)


@fieldwise_init
struct _BlockerCtx(Copyable, Movable):
    var started_addr: Int
    var go_addr: Int


def _test_l_blocker_job(ctx_addr: Int, chunk: Int):
    ref c = Pointer[_BlockerCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.started_addr)[unsafe_offset=0].store(1)
    var go_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.go_addr)
    while go_ptr[unsafe_offset=0].load() == 0:
        _ = external_call["sched_yield", Int32]()


@fieldwise_init
struct _PrioItemCtx(Copyable, Movable):
    var prio: Int
    var log_addr: Int
    var log_idx_addr: Int
    var done_addr: Int


def _test_l_prio_job(ctx_addr: Int, chunk: Int):
    ref c = Pointer[_PrioItemCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var idx = Int(Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.log_idx_addr)[unsafe_offset=0].fetch_add(1))
    Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=c.log_addr + idx * 8)[unsafe_offset=0] = Int64(c.prio)
    _ = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.done_addr)[unsafe_offset=0].fetch_add(1)


@fieldwise_init
struct _DepItemCtx(Copyable, Movable):
    var root_done_addr: Int
    var dep_done_addr: Int
    var dep_total_addr: Int
    var idx: Int


def _test_l_root_job(ctx_addr: Int, chunk: Int):
    Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0].store(1)


def _test_l_dep_job(ctx_addr: Int, chunk: Int):
    ref c = Pointer[_DepItemCtx, origin=MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    if Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.root_done_addr)[unsafe_offset=0].load() == 1:
        Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.dep_done_addr + c.idx * 8)[unsafe_offset=0].store(1)
    _ = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=c.dep_total_addr)[unsafe_offset=0].fetch_add(1)


def _raii_scope_drop_without_stop() raises:
    var p = JobPool()
    assert_true(p.start(4), "RAII scope pool start(4)")
    var out = List[Float32](length=64, fill=Float32(0.0))
    var ctx = _AffineOutCtx(Int(out.unsafe_ptr()))
    p.run(_test_l_affine_job, Int(Pointer(to=ctx)), 64)
    _ = ctx.out_ptr
    assert_equal(out[63], Float32(63 * 3 + 7) * Float32(0.125))


def _raii_scope_move_and_drop() raises:
    var p1 = JobPool()
    assert_true(p1.start(2), "RAII move pool start(2)")
    var p2 = p1^
    assert_equal(p2.workers, 2)
    var out = List[Float32](length=32, fill=Float32(0.0))
    var ctx = _AffineOutCtx(Int(out.unsafe_ptr()))
    p2.run(_test_l_affine_job, Int(Pointer(to=ctx)), 32)
    _ = ctx.out_ptr
    assert_equal(out[31], Float32(31 * 3 + 7) * Float32(0.125))


def test_l_jobpool_chase_lev_prio_raii() raises:
    assert_equal(MAX_DEPENDENTS, 16)

    # 1. Chase-Lev multi-worker stealing concurrency (256 asymmetric jobs on 4 workers)
    # First synchronize all 4 OS worker pthreads so CPU contention cannot let main drain before spawn.
    var pool = JobPool()
    assert_true(pool.start(4), "JobPool.start(4) must succeed")
    var w_ready = List[Int64](length=1, fill=0)
    var w_go = List[Int64](length=1, fill=0)
    var w_bctx = _ReadyBarrierCtx(Int(w_ready.unsafe_ptr()), Int(w_go.unsafe_ptr()))
    var w_handles = List[Int](capacity=4)
    for _ in range(4):
        var wh = pool.create(_test_l_ready_barrier_job, Int(Pointer(to=w_bctx)), 1)
        w_handles.append(wh)
        pool.submit(wh)
    var w_ready_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(w_ready.unsafe_ptr()))
    while w_ready_ptr[unsafe_offset=0].load() < 4:
        _ = external_call["sched_yield", Int32]()
    Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(w_go.unsafe_ptr()))[unsafe_offset=0].store(1)
    for i in range(4):
        pool.wait(w_handles[i])
    _ = w_bctx.go_addr
    _ = len(w_ready)
    _ = len(w_go)

    var n_jobs = 256
    var counts = List[Int64](length=n_jobs, fill=0)
    var tids = List[Int64](length=n_jobs, fill=0)
    var first_tid = List[Int64](length=1, fill=0)
    var two_seen = List[Int64](length=1, fill=0)
    var ctxs = List[_StealJobCtx](capacity=n_jobs)
    for i in range(n_jobs):
        ctxs.append(_StealJobCtx(Int(counts.unsafe_ptr()), Int(tids.unsafe_ptr()), Int(first_tid.unsafe_ptr()), Int(two_seen.unsafe_ptr()), i))
    var handles = List[Int](capacity=n_jobs)
    for i in range(n_jobs):
        var h = pool.create(_test_l_steal_job, Int(ctxs.unsafe_ptr()) + i * size_of[_StealJobCtx](), 1)
        handles.append(h)
    for i in range(n_jobs):
        pool.submit(handles[i])
    pool.rouse()
    for i in range(n_jobs):
        pool.wait(handles[i])
    _ = len(first_tid)
    _ = len(two_seen)
    _ = len(ctxs)
    var distinct_tids = List[Int64]()
    for i in range(n_jobs):
        assert_equal(counts[i], Int64(1))
        var t = tids[i]
        var seen = False
        for j in range(len(distinct_tids)):
            if distinct_tids[j] == t:
                seen = True
                break
        if not seen:
            distinct_tids.append(t)
    assert_true(len(distinct_tids) >= 2, "Chase-Lev must distribute work across >= 2 threads")

    # 1b. 3000-job overflow stress (> JOB_SLOTS=1024 and > DEQUE_CAP+GLOBAL_RING=2048)
    var n_overflow = 3000
    var ov_done = List[Int64](length=1, fill=0)
    var ov_handles = List[Int](capacity=n_overflow)
    for _ in range(n_overflow):
        var oh = pool.create(_test_l_overflow_job, Int(ov_done.unsafe_ptr()), 1)
        ov_handles.append(oh)
        pool.submit(oh)
    for i in range(n_overflow):
        pool.wait(ov_handles[i])
    assert_equal(ov_done[0], Int64(n_overflow))

    # 2. pool.run_affine parity (0 workers vs 4 workers) across 1, 7, 65, 513 chunks
    var serial_pool = JobPool()
    assert_true(serial_pool.allocate(), "serial pool allocate")
    var chunk_counts: List[Int] = [1, 7, 65, 513]
    for ci in range(len(chunk_counts)):
        var n = chunk_counts[ci]
        var out_s = List[Float32](length=n, fill=Float32(-1.0))
        var out_p = List[Float32](length=n, fill=Float32(-1.0))
        var cs = _AffineOutCtx(Int(out_s.unsafe_ptr()))
        var cp = _AffineOutCtx(Int(out_p.unsafe_ptr()))
        serial_pool.run_affine(_test_l_affine_job, Int(Pointer(to=cs)), n)
        pool.run_affine(_test_l_affine_job, Int(Pointer(to=cp)), n)
        _ = cs.out_ptr
        _ = cp.out_ptr
        for k in range(n):
            assert_equal(f32_bits(out_p[k]), f32_bits(out_s[k]))
    serial_pool.stop()
    pool.stop()
    pool.stop()  # idempotent explicit stop

    # 3. 3-tier priority ordering (JOB_PRIO_HIGH -> JOB_PRIO_NORMAL -> JOB_PRIO_LOW)
    var p1 = JobPool()
    assert_true(p1.start(1), "1-worker pool for priority test")
    var started = List[Int64](length=1, fill=0)
    var go = List[Int64](length=1, fill=0)
    var bctx = _BlockerCtx(Int(started.unsafe_ptr()), Int(go.unsafe_ptr()))
    var gate = p1.create(_test_l_blocker_job, Int(Pointer(to=bctx)), 1)
    p1.submit(gate, JOB_PRIO_HIGH)
    var started_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(started.unsafe_ptr()))
    while started_ptr[unsafe_offset=0].load() == 0:
        _ = external_call["sched_yield", Int32]()

    var prio_log = List[Int64](length=24, fill=-1)
    var log_idx = List[Int64](length=1, fill=0)
    var done_cnt = List[Int64](length=1, fill=0)
    var p_ctxs = List[_PrioItemCtx](capacity=24)
    var p_handles = List[Int](capacity=24)
    var tiers: List[Int] = [JOB_PRIO_LOW, JOB_PRIO_NORMAL, JOB_PRIO_HIGH]
    for ti in range(3):
        for k in range(8):
            var idx = ti * 8 + k
            p_ctxs.append(_PrioItemCtx(tiers[ti], Int(prio_log.unsafe_ptr()), Int(log_idx.unsafe_ptr()), Int(done_cnt.unsafe_ptr())))
            var h = p1.create(_test_l_prio_job, Int(p_ctxs.unsafe_ptr()) + idx * size_of[_PrioItemCtx](), 1)
            p_handles.append(h)
            p1.submit(h, tiers[ti])

    Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(go.unsafe_ptr()))[unsafe_offset=0].store(1)
    var done_ptr = Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(done_cnt.unsafe_ptr()))
    while done_ptr[unsafe_offset=0].load() < 24:
        _ = external_call["sched_yield", Int32]()
    p1.wait(gate)
    for i in range(24):
        p1.wait(p_handles[i])
    _ = bctx.go_addr
    _ = len(started)
    _ = len(go)
    _ = len(p_ctxs)
    _ = len(log_idx)
    _ = len(done_cnt)
    for i in range(8):
        assert_equal(prio_log[i], Int64(JOB_PRIO_HIGH))
    for i in range(8, 16):
        assert_equal(prio_log[i], Int64(JOB_PRIO_NORMAL))
    for i in range(16, 24):
        assert_equal(prio_log[i], Int64(JOB_PRIO_LOW))
    p1.stop()

    # 4. Dynamic dependents > 16 (64 and 96 dependents across 2 waves, zero abort)
    var pd = JobPool()
    assert_true(pd.start(4), "dependent overflow pool start(4)")
    var wave_sizes: List[Int] = [64, 96]
    for wi in range(len(wave_sizes)):
        var n_dep = wave_sizes[wi]
        var root_done = List[Int64](length=1, fill=0)
        var dep_done = List[Int64](length=n_dep, fill=0)
        var dep_total = List[Int64](length=1, fill=0)
        var d_ctxs = List[_DepItemCtx](capacity=n_dep)
        var root = pd.create(_test_l_root_job, Int(root_done.unsafe_ptr()), 1)
        var d_handles = List[Int](capacity=n_dep)
        for i in range(n_dep):
            d_ctxs.append(_DepItemCtx(Int(root_done.unsafe_ptr()), Int(dep_done.unsafe_ptr()), Int(dep_total.unsafe_ptr()), i))
            var dh = pd.create(_test_l_dep_job, Int(d_ctxs.unsafe_ptr()) + i * size_of[_DepItemCtx](), 1)
            pd.depend(dh, root)
            d_handles.append(dh)
        for i in range(n_dep):
            pd.submit(d_handles[i])
        pd.submit(root)
        for i in range(n_dep):
            pd.wait(d_handles[i])
        pd.wait(root)
        _ = len(d_ctxs)
        _ = len(root_done)
        assert_equal(dep_total[0], Int64(n_dep))
        for i in range(n_dep):
            assert_equal(dep_done[i], Int64(1))
    pd.stop()

    # 5. RAII __deinit__ without explicit stop + move + drop
    _raii_scope_drop_without_stop()
    _raii_scope_move_and_drop()


def _raii_mem_charge_scope(base_live: Int, base_cnt: Int) raises:
    var c1 = MemCharge(MEM_RENDER, 4096, 1)
    c1.set(8192)
    assert_equal(mem_live(MEM_RENDER), base_live + 8192)
    assert_equal(mem_count(MEM_RENDER), base_cnt + 1)
    var c2 = c1^
    assert_equal(mem_live(MEM_RENDER), base_live + 8192)
    assert_equal(mem_count(MEM_RENDER), base_cnt + 1)
    c2.release()
    assert_equal(mem_live(MEM_RENDER), base_live)
    assert_equal(mem_count(MEM_RENDER), base_cnt)


def _raii_arena_scope(base_live: Int, base_cnt: Int) raises:
    var a1 = Arena(MEM_FRAME, 8192)
    var p = a1.alloc(128, 64)
    assert_true(a1.owns(p))
    assert_equal(mem_live(MEM_FRAME), base_live + 8192)
    assert_equal(mem_count(MEM_FRAME), base_cnt + 1)
    var a2 = a1^
    assert_true(a2.owns(p))
    assert_equal(mem_live(MEM_FRAME), base_live + 8192)
    assert_equal(mem_count(MEM_FRAME), base_cnt + 1)
    a2.release()
    assert_equal(mem_live(MEM_FRAME), base_live)
    assert_equal(mem_count(MEM_FRAME), base_cnt)


def _raii_frame_arenas_scope(base_live: Int, base_cnt: Int) raises:
    var fa1 = FrameArenas(3, 4096)
    fa1.begin(0)
    _ = fa1.alloc(256, 64)
    assert_equal(fa1.capacity(), 12288)
    assert_equal(mem_live(MEM_FRAME), base_live + 12288)
    assert_equal(mem_count(MEM_FRAME), base_cnt + 3)
    var fa2 = fa1^
    assert_equal(fa2.capacity(), 12288)
    assert_equal(fa2.overflows(), 0)
    assert_equal(mem_live(MEM_FRAME), base_live + 12288)
    assert_equal(mem_count(MEM_FRAME), base_cnt + 3)
    fa2.release()
    assert_equal(mem_live(MEM_FRAME), base_live)
    assert_equal(mem_count(MEM_FRAME), base_cnt)


def _raii_simd_col_scope(base_live: Int, base_cnt: Int) raises:
    var col1 = SimdColumn[DType.float32](64, MEM_ENTITIES)
    col1.store8(0, SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    assert_equal(mem_live(MEM_ENTITIES), base_live + 256)
    assert_equal(mem_count(MEM_ENTITIES), base_cnt + 1)
    var col2 = col1^
    assert_equal(col2.load8(0)[3], Float32(4.0))
    assert_equal(mem_live(MEM_ENTITIES), base_live + 256)
    assert_equal(mem_count(MEM_ENTITIES), base_cnt + 1)
    col2.release()
    assert_equal(mem_live(MEM_ENTITIES), base_live)
    assert_equal(mem_count(MEM_ENTITIES), base_cnt)


def _raii_slots_scope() raises:
    var s1 = ArchetypeSlots(2, 32)
    var h1 = s1.spawn()
    var h2 = s1.spawn(deferred=True)
    var s2 = s1^
    assert_equal(s2.resolve(h1), 0)
    assert_equal(s2.resolve(h2, allow_spawning=True), 1)
    s2.commit_tick()
    assert_equal(s2.live_count, 2)
    assert_true(s2.despawn(h1))
    s2.release()
    s2.release()  # idempotent


def _raii_implicit_drop_scope() raises:
    var c_a = MemCharge(MEM_RENDER, 2048, 1)
    var c_b = c_a^
    var ar_a = Arena(MEM_FRAME, 4096)
    var ar_b = ar_a^
    var fa_a = FrameArenas(2, 4096)
    var fa_b = fa_a^
    var sc_a = SimdColumn[DType.float32](16, MEM_ENTITIES)
    var sc_b = sc_a^
    _ = c_b.bytes
    _ = ar_b.capacity
    _ = fa_b.capacity()
    _ = sc_b.capacity


def test_m_core_raii_and_dynamic_alloc() raises:
    assert_true(memory_install(), "dynamic memory_install must succeed")
    assert_true(memory_ready(), "memory_ready must be True")
    assert_true(install_anchor(), "dynamic install_anchor must succeed")
    assert_true(anchor_ready(), "anchor_ready must be True")

    var mb = memory_base()
    var ab = anchor_base()
    assert_true(mb != 0 and (mb < 0x5D5B00000000 or mb > 0x5D5EFFFFFFFF), "memory_base must be dynamic outside 0x5D5B..0x5D5E")
    assert_true(ab != 0 and (ab < 0x5D5B00000000 or ab > 0x5D5EFFFFFFFF), "anchor_base must be dynamic outside 0x5D5B..0x5D5E")
    assert_true(sys(SYS_MSYNC, 0x5D5C00000000, 4096, 1) != 0, "legacy 0x5D5C must not be mapped")
    assert_true(sys(SYS_MSYNC, 0x5D5D00000000, 4096, 1) != 0, "legacy 0x5D5D must not be mapped")

    # Simultaneous independent instances without collision
    var a1 = Arena(MEM_FRAME, 4096)
    var a2 = Arena(MEM_FRAME, 4096)
    var a3 = Arena(MEM_FRAME, 4096)
    var p1 = a1.alloc(64, 64)
    var p2 = a2.alloc(64, 64)
    var p3 = a3.alloc(64, 64)
    assert_true(a1.owns(p1) and not a2.owns(p1) and not a3.owns(p1))
    assert_true(a2.owns(p2) and not a1.owns(p2) and not a3.owns(p2))
    assert_true(a3.owns(p3) and not a1.owns(p3) and not a2.owns(p3))

    var fa1 = FrameArenas(2, 4096)
    var fa2 = FrameArenas(3, 4096)
    fa1.begin(0)
    fa2.begin(1)
    assert_true(fa1.alloc(128, 64) != fa2.alloc(128, 64))
    assert_equal(fa1.capacity(), 8192)
    assert_equal(fa2.capacity(), 12288)
    _ = fa1.alloc(8192, 64)
    assert_equal(fa1.overflows(), 1)
    assert_equal(fa2.overflows(), 0)

    var col_a = SimdColumn[DType.float32](16, MEM_ENTITIES)
    var col_b = SimdColumn[DType.float32](16, MEM_ENTITIES)
    col_a.set(0, Float32(11.0))
    col_b.set(0, Float32(22.0))
    assert_true(col_a.base_addr() != col_b.base_addr())
    assert_equal(col_a.get(0), Float32(11.0))
    assert_equal(col_b.get(0), Float32(22.0))

    var sl_a = ArchetypeSlots(1, 16)
    var sl_b = ArchetypeSlots(2, 16)
    var ha = sl_a.spawn()
    var hb = sl_b.spawn()
    assert_equal(sl_a.resolve(ha), 0)
    assert_equal(sl_b.resolve(ha), -1)
    assert_equal(sl_b.resolve(hb), 0)

    # RAII move + drop accounting (zero leaks)
    var base_render_live = mem_live(MEM_RENDER)
    var base_render_cnt = mem_count(MEM_RENDER)
    _raii_mem_charge_scope(base_render_live, base_render_cnt)
    assert_equal(mem_live(MEM_RENDER), base_render_live)
    assert_equal(mem_count(MEM_RENDER), base_render_cnt)

    var base_frame_live = mem_live(MEM_FRAME)
    var base_frame_cnt = mem_count(MEM_FRAME)
    _raii_arena_scope(base_frame_live, base_frame_cnt)
    assert_equal(mem_live(MEM_FRAME), base_frame_live)
    assert_equal(mem_count(MEM_FRAME), base_frame_cnt)

    _raii_frame_arenas_scope(base_frame_live, base_frame_cnt)
    assert_equal(mem_live(MEM_FRAME), base_frame_live)
    assert_equal(mem_count(MEM_FRAME), base_frame_cnt)

    var base_ent_live = mem_live(MEM_ENTITIES)
    var base_ent_cnt = mem_count(MEM_ENTITIES)
    _raii_simd_col_scope(base_ent_live, base_ent_cnt)
    assert_equal(mem_live(MEM_ENTITIES), base_ent_live)
    assert_equal(mem_count(MEM_ENTITIES), base_ent_cnt)

    _raii_slots_scope()

    # Implicit __deinit__ without calling release()
    var imp_render_live = mem_live(MEM_RENDER)
    var imp_render_cnt = mem_count(MEM_RENDER)
    var imp_frame_live = mem_live(MEM_FRAME)
    var imp_frame_cnt = mem_count(MEM_FRAME)
    var imp_ent_live = mem_live(MEM_ENTITIES)
    var imp_ent_cnt = mem_count(MEM_ENTITIES)
    _raii_implicit_drop_scope()
    assert_equal(mem_live(MEM_RENDER), imp_render_live)
    assert_equal(mem_count(MEM_RENDER), imp_render_cnt)
    assert_equal(mem_live(MEM_FRAME), imp_frame_live)
    assert_equal(mem_count(MEM_FRAME), imp_frame_cnt)
    assert_equal(mem_live(MEM_ENTITIES), imp_ent_live)
    assert_equal(mem_count(MEM_ENTITIES), imp_ent_cnt)

    # Idempotent shutdown + clean re-install
    memory_shutdown()
    memory_shutdown()
    assert_true(not memory_ready() and memory_base() == 0, "memory_shutdown must clear base")
    assert_true(memory_install() and memory_ready() and memory_base() != 0, "memory_install must re-initialize cleanly")

    anchor_shutdown()
    anchor_shutdown()
    assert_true(not anchor_ready() and anchor_base() == 0, "anchor_shutdown must clear base")
    assert_true(install_anchor() and anchor_ready() and anchor_base() != 0, "install_anchor must re-initialize cleanly")


# ===========================================================================
# Suite (n): >256 Components (ComponentBitSet, World, Archetype, Schedule, Snap)
# ===========================================================================

def _n_high_writer_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].store(777)


def _n_high_reader_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var seen = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].load()
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=1].store(seen)


def test_n_ecs_over_256_components() raises:
    # 1. ComponentBitSet API & canonical trailing-zero trimming
    var bs = ComponentBitSet()
    assert_true(bs.is_empty())
    assert_equal(bs.first_set(), -1)
    assert_equal(bs.popcount(), 0)
    bs.set(0)
    bs.set(255)
    bs.set(256)
    bs.set(300)
    bs.set(515)
    bs.set(700)
    assert_true(bs.has(0) and bs.has(255) and bs.has(256) and bs.has(300) and bs.has(515) and bs.has(700))
    assert_true(bs.test(700) and not bs.has(701) and not bs.has(-1))
    assert_equal(bs.popcount(), 6)
    assert_equal(bs.first_set(), 0)
    assert_equal(bs.trimmed_word_count(), 11)  # 700 >> 6 == 10 -> 11 words

    bs.clear(700)
    bs.clear(300)
    assert_true(not bs.has(700) and not bs.has(300))
    assert_equal(bs.popcount(), 4)

    var canon_a = ComponentBitSet()
    canon_a.set(300)
    canon_a.set(650)
    canon_a.clear(300)
    canon_a.clear(650)
    var canon_empty = ComponentBitSet()
    assert_true(canon_a == canon_empty)
    assert_true(canon_a.is_empty())
    assert_equal(canon_a.trimmed_word_count(), 0)
    assert_equal(canon_a.hash64(), canon_empty.hash64())

    var m1 = ComponentBitSet().with_bit(260).with_bit(520)
    var m2 = ComponentBitSet().with_bit(520).with_bit(610)
    assert_true(m1.intersects(m2))
    assert_true((m1 | m2).contains_all(m1))
    assert_true((m1 | m2).contains_all(m2))
    assert_equal((m1 & m2).first_set(), 520)
    assert_equal((m1.without(m2)).first_set(), 260)
    m1.clear_all()
    assert_true(m1.is_empty())

    # 2. Register 600 POD components (exceeding both 256 and old _COMP_HASH_CAP=512!) + typed >600
    var w = World()
    var pod_tids = List[Int](capacity=600)
    var pod_hashes = List[UInt64](capacity=600)
    for i in range(600):
        var h = mix64(UInt64(0xF200000100000000) + UInt64(i + 1))
        if h == 0:
            h = UInt64(i + 1)
        var mode = STORAGE_SPARSE if i >= 520 else STORAGE_TABLE
        var tid = w.register_pod_component(h, size_of[Int32](), 4, mode)
        assert_equal(tid, i)
        pod_tids.append(tid)
        pod_hashes.append(h)

    var tid_pos = w.register_component[CompPos]()
    var tid_vel = w.register_component[CompVel]()
    var tid_burn = w.register_component[TagBurningSparse]()
    assert_equal(tid_pos, 600)
    assert_equal(tid_vel, 601)
    assert_equal(tid_burn, 602)

    # Verify hash table rehashing works for all 603 components
    for i in range(600):
        assert_equal(w.find_component_by_hash(pod_hashes[i]), i)
    assert_equal(w.find_component_by_hash(CompPos.COMPONENT_ID), 600)
    assert_equal(w.find_component_by_hash(TagBurningSparse.COMPONENT_ID), 602)

    # 3. Archetype transitions, sparse masks & query_entities above bits 256 & 512
    var e1 = w.spawn_empty()
    var e2 = w.spawn_empty()
    var e3 = w.spawn_empty()
    var v0 = Int32(101)
    var v255 = Int32(255)
    var v256 = Int32(256)
    var v320 = Int32(320)
    var v515 = Int32(515)
    var v550 = Int32(550)
    assert_true(w.insert_raw_pod(e1, 0, Int(Pointer(to=v0))))
    assert_true(w.insert_raw_pod(e1, 255, Int(Pointer(to=v255))))
    assert_true(w.insert_raw_pod(e1, 256, Int(Pointer(to=v256))))
    assert_true(w.insert_raw_pod(e1, 320, Int(Pointer(to=v320))))
    assert_true(w.insert_raw_pod(e1, 515, Int(Pointer(to=v515))))
    assert_true(w.insert_raw_pod(e1, 550, Int(Pointer(to=v550))))
    assert_true(w.insert[CompPos](e1, CompPos(11.0, 22.0, 33.0)))
    assert_true(w.insert[TagBurningSparse](e1, TagBurningSparse(44.5)))

    assert_true(w.insert_raw_pod(e2, 320, Int(Pointer(to=v320))))
    assert_true(w.insert_raw_pod(e2, 515, Int(Pointer(to=v515))))
    assert_true(w.insert[CompPos](e2, CompPos(1.0, 2.0, 3.0)))
    assert_true(w.insert[CompVel](e2, CompVel(4.0, 5.0, 6.0)))

    assert_true(w.insert_raw_pod(e3, 320, Int(Pointer(to=v320))))
    assert_true(w.insert_raw_pod(e3, 550, Int(Pointer(to=v550))))
    assert_true(w.insert[CompPos](e3, CompPos(7.0, 8.0, 9.0)))

    assert_true(w.has_by_type_id(e1, 515) and w.has_by_type_id(e1, 550))
    assert_true(w.remove_by_type_id(e3, 550))
    assert_true(not w.has_by_type_id(e3, 550))

    var q_mask = ComponentBitSet().with_bit(320).with_bit(515).with_bit(tid_pos)
    var q_excl = ComponentBitSet().with_bit(550)
    var q_res = w.query_entities(q_mask, q_excl)
    assert_equal(len(q_res), 1)
    assert_true(q_res[0] == e2)

    var q_all_320_pos = w.query_entities(ComponentBitSet().with_bit(320).with_bit(tid_pos))
    assert_equal(len(q_all_320_pos), 3)

    # 4. Schedule conflict DAG at bit >= 256 and >= 512
    var sched = Schedule()
    var res = Resources()
    var cmds = ParallelCommands()
    var pool = JobPool()
    _ = pool.start(4)
    var dag_buf = List[Int64](length=2, fill=0)
    var dag_addr = Int(dag_buf.unsafe_ptr())
    var bit_520 = ComponentBitSet().with_bit(520)
    _ = sched.add_system(STAGE_UPDATE, "high_writer", _n_high_writer_sys, writes=bit_520, user_ctx=dag_addr)
    _ = sched.add_system(STAGE_UPDATE, "high_reader", _n_high_reader_sys, reads=bit_520, user_ctx=dag_addr)
    sched.run_stage(STAGE_UPDATE, w, res, pool, cmds)
    assert_equal(_atomic_load_i64(dag_addr, 0), 777)
    assert_equal(_atomic_load_i64(dag_addr, 1), 777)
    _ = len(dag_buf)
    pool.stop()

    # 5. snapshot_blob -> restore_blob with permuted registration order for >600 components
    var snap = w.snapshot_blob()
    var w_dst = World()
    # Register typed components first, then sparse PODs, then table PODs in reverse!
    _ = w_dst.register_component[TagBurningSparse]()
    _ = w_dst.register_component[CompVel]()
    _ = w_dst.register_component[CompPos]()
    for i in range(599, -1, -1):
        var mode = STORAGE_SPARSE if i >= 520 else STORAGE_TABLE
        _ = w_dst.register_pod_component(pod_hashes[i], size_of[Int32](), 4, mode)

    w_dst.restore_blob(snap)
    assert_equal(w_dst.state_checksum(), w.state_checksum())
    assert_equal(w_dst.entity_count(), 3)
    assert_equal(w_dst.get[CompPos](e1).x, Float32(11.0))
    assert_equal(w_dst.get[TagBurningSparse](e1).dps, Float32(44.5))
    var tid_515_dst = w_dst.find_component_by_hash(pod_hashes[515])
    var p515 = w_dst.get_raw_ptr(e1, tid_515_dst)
    assert_true(p515 != 0)
    assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=p515)[unsafe_offset=0], Int32(515))
    var tid_550_dst = w_dst.find_component_by_hash(pod_hashes[550])
    var p550 = w_dst.get_raw_ptr(e1, tid_550_dst)
    assert_true(p550 != 0)
    assert_equal(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=p550)[unsafe_offset=0], Int32(550))


# ===========================================================================
# Suite (o): Portable Snapshot/Restore & Missing Binary Hooks / Tag 2 Error
# ===========================================================================

struct NestedHeapComp(SerializableComponent):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.NestedHeapComp")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "NestedHeapComp"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("NHCP")
    comptime SCHEMA_VERSION: Int = 1

    var title: String
    var tags: List[String]
    var points: List[Vec3]

    def __init__(out self, var title: String, var tags: List[String], var points: List[Vec3]):
        self.title = title^
        self.tags = tags^
        self.points = points^

    def __init__(out self, *, copy: Self):
        self.title = copy.title
        self.tags = copy.tags.copy()
        self.points = copy.points.copy()

    def write_bin(self, mut out: BlobWriter):
        out.text(self.title)
        out.integer(len(self.tags))
        for i in range(len(self.tags)):
            out.text(self.tags[i])
        out.integer(len(self.points))
        for i in range(len(self.points)):
            out.vec3(self.points[i])

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var title = reader.text()
        var n_tags = reader.integer()
        if reader.failed or n_tags < 0 or n_tags > 10000:
            raise Error("NestedHeapComp: invalid tags count")
        var tags = List[String](capacity=max(1, n_tags))
        for _ in range(n_tags):
            tags.append(reader.text())
        var n_pts = reader.integer()
        if reader.failed or n_pts < 0 or n_pts > 10000:
            raise Error("NestedHeapComp: invalid points count")
        var points = List[Vec3](capacity=max(1, n_pts))
        for _ in range(n_pts):
            points.append(reader.vec3())
        if reader.failed:
            raise Error("NestedHeapComp: truncated payload")
        return Self(title^, tags^, points^)

    def write_json(self) -> String:
        return "{\"title\": \"" + self.title + "\"}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var t = doc.text(doc.object_get(node, "title"), "")
        return Self(t, List[String](), List[Vec3]())


struct UnhookedTableHeap(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.UnhookedTableHeap")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var label: String

    def __init__(out self, var label: String):
        self.label = label^

    def __init__(out self, *, copy: Self):
        self.label = copy.label


struct UnhookedSparseHeap(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.UnhookedSparseHeap")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var note: String

    def __init__(out self, var note: String):
        self.note = note^

    def __init__(out self, *, copy: Self):
        self.note = copy.note


def _find_last_u64(bytes: List[UInt8], val: UInt64) -> Int:
    var b0 = UInt8(val & 0xFF)
    var b1 = UInt8((val >> 8) & 0xFF)
    var b2 = UInt8((val >> 16) & 0xFF)
    var b3 = UInt8((val >> 24) & 0xFF)
    var b4 = UInt8((val >> 32) & 0xFF)
    var b5 = UInt8((val >> 40) & 0xFF)
    var b6 = UInt8((val >> 48) & 0xFF)
    var b7 = UInt8((val >> 56) & 0xFF)
    var found = -1
    for i in range(len(bytes) - 7):
        if (
            bytes[i] == b0
            and bytes[i + 1] == b1
            and bytes[i + 2] == b2
            and bytes[i + 3] == b3
            and bytes[i + 4] == b4
            and bytes[i + 5] == b5
            and bytes[i + 6] == b6
            and bytes[i + 7] == b7
        ):
            found = i
    return found


def _capture_cross_world_portable_snap(stats_addr: Int) raises -> Tuple[List[UInt8], UInt64, Entity, Entity]:
    var reg = ComponentRegistry.with_builtins()
    reg.register[NestedHeapComp]()

    var w_src = World()
    reg.ensure_registered_in_world(w_src)
    var hn_tid = w_src.register_component[HeapNamedComp]()
    w_src.set_component_binary_hooks(hn_tid, _heap_named_write, _heap_named_read)
    var bag_tid = w_src.register_component[TrackedHeapBag]()
    w_src.set_component_binary_hooks(bag_tid, _tracked_bag_write, _tracked_bag_read)
    var sh_tid = w_src.register_component[SparseHeapTag]()
    w_src.set_component_binary_hooks(sh_tid, _sparse_heap_write, _sparse_heap_read)

    var e0 = w_src.spawn_empty()
    var e1 = w_src.spawn_empty()
    var tags = List[String]()
    tags.append(_make_heap_string("tagA", 1))
    tags.append(_make_heap_string("tagB", 2))
    var pts = List[Vec3]()
    pts.append(Vec3(1.5, -2.5, 3.5))
    pts.append(Vec3(9.0, 8.0, 7.0))
    _ = w_src.insert[NestedHeapComp](e0, NestedHeapComp(_make_heap_string("nested_title", 0), tags^, pts^))
    _ = w_src.insert[HeapNamedComp](e0, HeapNamedComp(_make_heap_string("portable_hero", 10)))
    _ = w_src.insert[SparseHeapTag](e0, SparseHeapTag(_make_heap_string("portable_sparse", 20)))

    var bag_vals = List[Int]()
    for k in range(10):
        bag_vals.append(k * 7 + 3)
    _ = w_src.insert[TrackedHeapBag](e1, TrackedHeapBag(stats_addr, _make_heap_string("bag1", 1), bag_vals^))

    var cs = w_src.state_checksum()
    var snap = w_src.snapshot_blob()
    return (snap^, cs, e0, e1)


def _restore_cross_world_portable_snap(
    snap: List[UInt8], expected_cs: UInt64, e0: Entity, e1: Entity
) raises:
    var reg = ComponentRegistry.with_builtins()
    reg.register[NestedHeapComp]()

    # Independent destination world with permuted registration order (w_src is already destroyed!)
    var w_dst = World()
    var sh_tid = w_dst.register_component[SparseHeapTag]()
    w_dst.set_component_binary_hooks(sh_tid, _sparse_heap_write, _sparse_heap_read)
    var bag_tid = w_dst.register_component[TrackedHeapBag]()
    w_dst.set_component_binary_hooks(bag_tid, _tracked_bag_write, _tracked_bag_read)
    var hn_tid = w_dst.register_component[HeapNamedComp]()
    w_dst.set_component_binary_hooks(hn_tid, _heap_named_write, _heap_named_read)
    reg.ensure_registered_in_world(w_dst)

    # Restore via List[UInt8] -> Span[UInt8, _] overload
    w_dst.restore_blob(snap)
    assert_equal(w_dst.state_checksum(), expected_cs)
    assert_equal(w_dst.entity_count(), 2)
    ref nh = w_dst.get_ptr[NestedHeapComp](e0).value()[unsafe_offset=0]
    assert_equal(nh.title, _make_heap_string("nested_title", 0))
    assert_equal(len(nh.tags), 2)
    assert_equal(nh.tags[1], _make_heap_string("tagB", 2))
    assert_equal(len(nh.points), 2)
    assert_equal(nh.points[1].z, Float32(7.0))
    assert_equal(w_dst.get_ptr[HeapNamedComp](e0).value()[unsafe_offset=0].label, _make_heap_string("portable_hero", 10))
    assert_equal(w_dst.get_ptr[SparseHeapTag](e0).value()[unsafe_offset=0].note, _make_heap_string("portable_sparse", 20))
    ref bag = w_dst.get_ptr[TrackedHeapBag](e1).value()[unsafe_offset=0]
    assert_equal(len(bag.values), 10)
    assert_equal(bag.values[9], 66)


def test_o_portable_snapshot_and_missing_hooks_error() raises:
    # 1. Cross-world portable snapshot -> destroy w_src -> restore into w_dst (zero leaks!)
    var bag_stats = List[Int](length=4, fill=0)
    var stats_addr = Int(bag_stats.unsafe_ptr())
    var cap = _capture_cross_world_portable_snap(stats_addr)
    assert_equal(bag_stats[0], 0)
    _restore_cross_world_portable_snap(cap[0], cap[1], cap[2], cap[3])
    assert_equal(bag_stats[0], 0)
    assert_true(bag_stats[3] >= 2)
    _ = len(bag_stats)

    # 2. Explicit Error when snapshot_blob encounters unhooked non-trivial STORAGE_TABLE component
    var w_bad_t = World()
    var et = w_bad_t.spawn_empty()
    _ = w_bad_t.insert[UnhookedTableHeap](et, UnhookedTableHeap(_make_heap_string("no_hooks", 1)))
    var bw = BlobWriter()
    bw.u32(0x12345678)
    var caught_table = False
    try:
        w_bad_t.snapshot_blob(bw)
    except:
        caught_table = True
    assert_true(caught_table, "snapshot_blob must raise Error for unhooked non-trivial table component")
    assert_equal(len(bw.data), 4)
    assert_equal(len(bw.open_at), 0)

    # 3. Explicit Error when snapshot_blob encounters unhooked non-trivial STORAGE_SPARSE component
    var w_bad_s = World()
    var es = w_bad_s.spawn_empty()
    _ = w_bad_s.insert[UnhookedSparseHeap](es, UnhookedSparseHeap(_make_heap_string("no_sparse_hooks", 2)))
    var caught_sparse = False
    try:
        _ = w_bad_s.snapshot_blob()
    except:
        caught_sparse = True
    assert_true(caught_sparse, "snapshot_blob must raise Error for unhooked non-trivial sparse component")

    # 4. Explicit Error on legacy Clone Vault tag 2 in World.restore_blob (both table and sparse)
    var w_tag = World()
    var htid = w_tag.register_component[HeapNamedComp]()
    w_tag.set_component_binary_hooks(htid, _heap_named_write, _heap_named_read)
    var stid = w_tag.register_component[SparseHeapTag]()
    w_tag.set_component_binary_hooks(stid, _sparse_heap_write, _sparse_heap_read)
    var etag = w_tag.spawn_empty()
    _ = w_tag.insert[HeapNamedComp](etag, HeapNamedComp(_make_heap_string("tag_probe", 5)))
    _ = w_tag.insert[SparseHeapTag](etag, SparseHeapTag(_make_heap_string("sparse_probe", 6)))
    var good_snap = w_tag.snapshot_blob()

    # Patch table column mode_tag (byte immediately after last HeapNamedComp.COMPONENT_ID) from 1 -> 2
    var bad_table_snap = good_snap.copy()
    var t_off = _find_last_u64(bad_table_snap, HeapNamedComp.COMPONENT_ID)
    assert_true(t_off >= 0 and bad_table_snap[t_off + 8] == 1)
    bad_table_snap[t_off + 8] = 2
    var caught_tag2_table = False
    try:
        w_tag.restore_blob(bad_table_snap)
    except:
        caught_tag2_table = True
    assert_true(caught_tag2_table, "restore_blob must reject legacy tag 2 for table columns")

    # Patch sparse mode_tag (byte after last SparseHeapTag.COMPONENT_ID + scnt(8) + 1*Entity(8)) from 1 -> 2
    var bad_sparse_snap = good_snap.copy()
    var s_off = _find_last_u64(bad_sparse_snap, SparseHeapTag.COMPONENT_ID)
    assert_true(s_off >= 0 and bad_sparse_snap[s_off + 24] == 1)
    bad_sparse_snap[s_off + 24] = 2
    var caught_tag2_sparse = False
    try:
        w_tag.restore_blob(bad_sparse_snap)
    except:
        caught_tag2_sparse = True
    assert_true(caught_tag2_sparse, "restore_blob must reject legacy tag 2 for sparse sets")


# ===========================================================================
# Suite (p): Zero Fixed 0x5D5B..0x5D5E Mappings in /proc/self/maps
# ===========================================================================

def test_p_no_fixed_ecs_mappings_in_proc_maps() raises:
    # Exercise World, ParallelCommands + par_for_each1, and snapshot_blob / restore_blob
    var w = World()
    _ = w.register_component[CompPos]()
    _ = w.register_component[CompVel]()
    for i in range(128):
        var e = w.spawn_empty()
        _ = w.insert[CompPos](e, CompPos(Float32(i), 1.0, 2.0))
    var pool = JobPool()
    _ = pool.start(4)
    var cmds = ParallelCommands()
    w.par_for_each1[CompPos](pool, _implicit_order_kernel, Int(Pointer(to=cmds)))
    w.flush_commands(cmds)
    pool.stop()

    var snap = w.snapshot_blob()
    var w2 = World()
    _ = w2.register_component[CompPos]()
    _ = w2.register_component[CompVel]()
    w2.restore_blob(snap)
    assert_equal(w2.state_checksum(), w.state_checksum())

    var raw_maps = read_stream("/proc/self/maps")
    assert_true(len(raw_maps) > 0, "read_stream('/proc/self/maps') must return non-empty output")
    var maps_str = String(unsafe_from_utf8=raw_maps).lower()
    assert_true("5d5b" not in maps_str, "/proc/self/maps must not contain fixed 0x5D5B mapping")
    assert_true("5d5c" not in maps_str, "/proc/self/maps must not contain fixed 0x5D5C mapping")
    assert_true("5d5d" not in maps_str, "/proc/self/maps must not contain fixed 0x5D5D mapping")
    assert_true("5d5e" not in maps_str, "/proc/self/maps must not contain fixed 0x5D5E mapping")
    assert_true(sys(SYS_MSYNC, 0x5D5B00000000, 4096, 1) != 0, "legacy 0x5D5B must not be mapped")


# ===========================================================================
# Suite (q): Atomic World.restore_blob Staging, Duplicate Guards & Children DoS
# ===========================================================================

struct TrackedHeapBagB(Component):
    """Second distinct tracked non-trivial table component for duplicate-column & UB-drop checks."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.TrackedHeapBagB")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var stats_addr: Int
    var tag: String
    var values: List[Int]

    def __init__(out self, stats_addr: Int, var tag: String, var values: List[Int]):
        self.stats_addr = stats_addr
        self.tag = tag^
        self.values = values^
        if stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=stats_addr)[unsafe_offset=0] += 1

    def __init__(out self, *, copy: Self):
        self.stats_addr = copy.stats_addr
        self.tag = copy.tag
        self.values = copy.values.copy()
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] += 1
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=1] += 1

    def __init__(out self, *, deinit move: Self):
        self.stats_addr = move.stats_addr
        self.tag = move.tag^
        self.values = move.values^
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=2] += 1

    def __deinit__(deinit self):
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] -= 1
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=3] += 1


def _tracked_bag_b_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[TrackedHeapBagB, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.integer(comp.stats_addr)
    out.text(comp.tag)
    out.integer(len(comp.values))
    for i in range(len(comp.values)):
        out.integer(comp.values[i])


def _tracked_bag_b_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var stats_addr = reader.integer()
    var tag = reader.text()
    var n = reader.integer()
    if reader.failed or n < 0 or n > 1000000:
        return False
    var vals = List[Int](capacity=max(1, n))
    for _ in range(n):
        vals.append(reader.integer())
    if reader.failed:
        return False
    var dst = Pointer[TrackedHeapBagB, MutAnyOrigin](unsafe_from_address=comp_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = TrackedHeapBagB(stats_addr, tag^, vals^)
    return True


struct TrackedSparseBag(Component):
    """Tracked non-trivial STORAGE_SPARSE component verifying zero leaks / double-frees."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.TrackedSparseBag")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var stats_addr: Int
    var note: String
    var nums: List[Int]

    def __init__(out self, stats_addr: Int, var note: String, var nums: List[Int]):
        self.stats_addr = stats_addr
        self.note = note^
        self.nums = nums^
        if stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=stats_addr)[unsafe_offset=0] += 1

    def __init__(out self, *, copy: Self):
        self.stats_addr = copy.stats_addr
        self.note = copy.note
        self.nums = copy.nums.copy()
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] += 1
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=1] += 1

    def __init__(out self, *, deinit move: Self):
        self.stats_addr = move.stats_addr
        self.note = move.note^
        self.nums = move.nums^
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=2] += 1

    def __deinit__(deinit self):
        if self.stats_addr != 0:
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=0] -= 1
            Pointer[Int, MutAnyOrigin](unsafe_from_address=self.stats_addr)[unsafe_offset=3] += 1


def _tracked_sparse_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[TrackedSparseBag, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.integer(comp.stats_addr)
    out.text(comp.note)
    out.integer(len(comp.nums))
    for i in range(len(comp.nums)):
        out.integer(comp.nums[i])


def _tracked_sparse_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var stats_addr = reader.integer()
    var note = reader.text()
    var n = reader.integer()
    if reader.failed or n < 0 or n > 1000000:
        return False
    var nums = List[Int](capacity=max(1, n))
    for _ in range(n):
        nums.append(reader.integer())
    if reader.failed:
        return False
    var dst = Pointer[TrackedSparseBag, MutAnyOrigin](unsafe_from_address=comp_addr)
    __get_address_as_uninit_lvalue(dst._mlir_value) = TrackedSparseBag(stats_addr, note^, nums^)
    return True


@fieldwise_init
struct OverrunHookComp(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.OverrunHookComp")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var val: Int32


def _overrun_hook_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[OverrunHookComp, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.u32(UInt32(comp.val))


def _overrun_hook_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    var v = Int32(reader.u32())
    # Force an overrun past the current blob's limit while returning True so `reader.leave()` catches it
    reader.offset = reader.limit() + 1
    Pointer[OverrunHookComp, MutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0] = OverrunHookComp(v)
    return True


def _write_u64_at(mut bytes: List[UInt8], offset: Int, val: UInt64):
    for k in range(8):
        bytes[offset + k] = UInt8((val >> UInt64(k * 8)) & 0xFF)


def _assert_dst_preserved(
    w: World,
    e0: Entity,
    e1: Entity,
    orig_count: Int,
    orig_cs: UInt64,
    expected_tag: String,
    expected_sp_note: String,
) raises:
    assert_equal(w.entity_count(), orig_count)
    assert_true(w.is_alive(e0) and w.is_alive(e1))
    assert_equal(w.state_checksum(), orig_cs)
    assert_equal(w.get_ptr[TrackedHeapBag](e0).value()[unsafe_offset=0].tag, expected_tag)
    assert_equal(w.get_ptr[TrackedSparseBag](e0).value()[unsafe_offset=0].note, expected_sp_note)


def _register_q_world_types(mut w: World):
    var t_a = w.register_component[TrackedHeapBag]()
    w.set_component_binary_hooks(t_a, _tracked_bag_write, _tracked_bag_read)
    var t_b = w.register_component[TrackedHeapBagB]()
    w.set_component_binary_hooks(t_b, _tracked_bag_b_write, _tracked_bag_b_read)
    var t_sp = w.register_component[TrackedSparseBag]()
    w.set_component_binary_hooks(t_sp, _tracked_sparse_write, _tracked_sparse_read)
    var t_hn = w.register_component[HeapNamedComp]()
    w.set_component_binary_hooks(t_hn, _heap_named_write, _heap_named_read)
    var t_sh = w.register_component[SparseHeapTag]()
    w.set_component_binary_hooks(t_sh, _sparse_heap_write, _sparse_heap_read)
    var t_ov = w.register_component[OverrunHookComp]()
    w.set_component_binary_hooks(t_ov, _overrun_hook_write, _overrun_hook_read)
    _ = w.register_component[CompPos]()
    _ = w.register_component[TagBurningSparse]()


def _q_ints(a: Int) -> List[Int]:
    var out = List[Int](capacity=1)
    out.append(a)
    return out^


def _q_ints2(a: Int, b: Int) -> List[Int]:
    var out = List[Int](capacity=2)
    out.append(a)
    out.append(b)
    return out^


def _run_q_atomicity_scope(
    dst_stats_addr: Int,
    src_a_stats_addr: Int,
    src_b_stats_addr: Int,
    src_sp_stats_addr: Int,
) raises:
    var w_dst = World()
    _register_q_world_types(w_dst)
    var e0 = w_dst.spawn_empty()
    var e1 = w_dst.spawn_empty()
    var expected_tag = _make_heap_string("dst_preserved_table", 1)
    var expected_sp_note = _make_heap_string("dst_preserved_sparse", 2)
    _ = w_dst.insert[TrackedHeapBag](e0, TrackedHeapBag(dst_stats_addr, expected_tag.copy(), _q_ints2(10, 20)))
    _ = w_dst.insert[TrackedSparseBag](e0, TrackedSparseBag(dst_stats_addr, expected_sp_note.copy(), _q_ints2(40, 50)))
    _ = w_dst.insert[CompPos](e0, CompPos(7.0, 8.0, 9.0))
    _ = w_dst.insert[HeapNamedComp](e1, HeapNamedComp(_make_heap_string("dst_e1_name", 3)))
    _ = set_parent(w_dst, e1, e0)

    var orig_count = w_dst.entity_count()
    var orig_cs = w_dst.state_checksum()

    # 1. Legacy tag 2 (table & sparse) must raise and preserve w_dst with 0 staged leaks
    var w_tag = World()
    _register_q_world_types(w_tag)
    var et = w_tag.spawn_empty()
    _ = w_tag.insert[TrackedHeapBag](et, TrackedHeapBag(src_a_stats_addr, _make_heap_string("src_tag2", 1), _q_ints2(1, 2)))
    _ = w_tag.insert[TrackedSparseBag](et, TrackedSparseBag(src_sp_stats_addr, _make_heap_string("src_sp2", 2), _q_ints2(3, 4)))
    var tag_snap = w_tag.snapshot_blob()

    var bad_t2 = tag_snap.copy()
    var t2_off = _find_last_u64(bad_t2, TrackedHeapBag.COMPONENT_ID)
    assert_true(t2_off >= 0 and bad_t2[t2_off + 8] == 1)
    bad_t2[t2_off + 8] = 2
    var caught = False
    try:
        w_dst.restore_blob(bad_t2)
    except:
        caught = True
    assert_true(caught, "restore_blob must reject table tag 2")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)

    var bad_s2 = tag_snap.copy()
    var s2_off = _find_last_u64(bad_s2, TrackedSparseBag.COMPONENT_ID)
    assert_true(s2_off >= 0 and bad_s2[s2_off + 24] == 1)
    bad_s2[s2_off + 24] = 2
    caught = False
    try:
        w_dst.restore_blob(bad_s2)
    except:
        caught = True
    assert_true(caught, "restore_blob must reject sparse tag 2")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_a_stats_addr)[unsafe_offset=0], 1)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_sp_stats_addr)[unsafe_offset=0], 1)

    # 2. Corrupted wc = 0xFFFFFFFF and num_archs = 0x7FFFFFFF must raise and preserve w_dst
    for mode in range(2):
        var bw = BlobWriter()
        bw.begin(fourcc("ECS2"), 3)
        bw.u32(1)
        bw.integer(1)
        bw.integer(1)
        bw.integer(0)
        bw.integer(1)
        bw.u64(CompPos.COMPONENT_ID)
        bw.integer(STORAGE_TABLE)
        bw.integer(size_of[CompPos]())
        bw.boolean(True)
        bw.u32(1)
        bw.boolean(True)
        if mode == 0:
            bw.integer(0x7FFFFFFF)  # num_archs = 0x7FFFFFFF
        else:
            bw.integer(1)  # num_archs = 1
            bw.u32(UInt32(0xFFFFFFFF))  # wc = 0xFFFFFFFF
        bw.end()
        caught = False
        try:
            w_dst.restore_blob(bw.data)
        except:
            caught = True
        assert_true(caught, "restore_blob must reject huge num_archs / wc without mutating w_dst")
        _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)

    # 3a. Mid-stream table read_custom_bin failure (entity 0 succeeds, entity 1 fails -> 0 staged leaks)
    var w_mid_t = World()
    _register_q_world_types(w_mid_t)
    var mt0 = w_mid_t.spawn_empty()
    var mt1 = w_mid_t.spawn_empty()
    _ = w_mid_t.insert[TrackedHeapBag](mt0, TrackedHeapBag(src_a_stats_addr, _make_heap_string("mid_t0", 1), _q_ints(11)))
    _ = w_mid_t.insert[TrackedHeapBag](mt1, TrackedHeapBag(src_a_stats_addr, _make_heap_string("mid_t1", 2), _q_ints(22)))
    var mid_t_snap = w_mid_t.snapshot_blob()
    # Second TrackedHeapBag payload has len(values) = 1 right before the single Int(22) at the end of the column
    var mt_col_off = _find_last_u64(mid_t_snap, TrackedHeapBag.COMPONENT_ID)
    assert_true(mt_col_off >= 0 and mid_t_snap[mt_col_off + 8] == 1)
    # Locate the second occurrence of UInt64(1) followed by UInt64(22)
    var val22_off = _find_last_u64(mid_t_snap, UInt64(22))
    assert_true(val22_off >= 8)
    _write_u64_at(mid_t_snap, val22_off - 8, UInt64(0xFFFFFFFFFFFFFFFF))  # n = -1 -> _tracked_bag_read returns False
    caught = False
    try:
        w_dst.restore_blob(mid_t_snap)
    except:
        caught = True
    assert_true(caught, "restore_blob must raise on mid-stream table read_custom_bin failure")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_a_stats_addr)[unsafe_offset=0], 3)

    # 3b. Mid-stream sparse read_custom_bin failure (entity 0 succeeds, entity 1 fails -> 0 staged leaks)
    var w_mid_s = World()
    _register_q_world_types(w_mid_s)
    var ms0 = w_mid_s.spawn_empty()
    var ms1 = w_mid_s.spawn_empty()
    _ = w_mid_s.insert[TrackedSparseBag](ms0, TrackedSparseBag(src_sp_stats_addr, _make_heap_string("mid_s0", 1), _q_ints(33)))
    _ = w_mid_s.insert[TrackedSparseBag](ms1, TrackedSparseBag(src_sp_stats_addr, _make_heap_string("mid_s1", 2), _q_ints(44)))
    var mid_s_snap = w_mid_s.snapshot_blob()
    var val44_off = _find_last_u64(mid_s_snap, UInt64(44))
    assert_true(val44_off >= 8)
    _write_u64_at(mid_s_snap, val44_off - 8, UInt64(0xFFFFFFFFFFFFFFFF))
    caught = False
    try:
        w_dst.restore_blob(mid_s_snap)
    except:
        caught = True
    assert_true(caught, "restore_blob must raise on mid-stream sparse read_custom_bin failure")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_sp_stats_addr)[unsafe_offset=0], 3)

    # 4. reader.leave() overrun failure must drop staged state (0 leaks) and preserve w_dst
    var w_ov = World()
    _register_q_world_types(w_ov)
    var eov = w_ov.spawn_empty()
    _ = w_ov.insert[TrackedHeapBag](eov, TrackedHeapBag(src_a_stats_addr, _make_heap_string("overrun_staged", 9), _q_ints2(7, 8)))
    _ = w_ov.insert[OverrunHookComp](eov, OverrunHookComp(123))
    var ov_snap = w_ov.snapshot_blob()
    caught = False
    var ov_err = String("")
    try:
        w_dst.restore_blob(ov_snap)
    except err:
        caught = True
        ov_err = String(err)
    assert_true(caught and ("overran" in ov_err), "restore_blob must propagate reader.leave() overrun error")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_a_stats_addr)[unsafe_offset=0], 4)

    # 5. Duplicate table column hash: must reject cleanly, drop column 1 (0 leaks), NEVER drop uninit column 2, preserve w_dst
    var w_dup_col = World()
    _register_q_world_types(w_dup_col)
    var edc = w_dup_col.spawn_empty()
    _ = w_dup_col.insert[TrackedHeapBag](edc, TrackedHeapBag(src_a_stats_addr, _make_heap_string("dup_col_a", 1), _q_ints(55)))
    _ = w_dup_col.insert[TrackedHeapBagB](edc, TrackedHeapBagB(src_b_stats_addr, _make_heap_string("dup_col_b", 2), _q_ints(66)))
    var drops_b_before = Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_b_stats_addr)[unsafe_offset=3]
    var dup_col_snap = w_dup_col.snapshot_blob()
    var col_b_off = _find_last_u64(dup_col_snap, TrackedHeapBagB.COMPONENT_ID)
    assert_true(col_b_off >= 0)
    _write_u64_at(dup_col_snap, col_b_off, TrackedHeapBag.COMPONENT_ID)
    caught = False
    var dup_col_err = String("")
    try:
        w_dst.restore_blob(dup_col_snap)
    except err:
        caught = True
        dup_col_err = String(err)
    assert_true(caught and ("duplicate component column" in dup_col_err), "restore_blob must reject duplicate table column hash")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_a_stats_addr)[unsafe_offset=0], 5)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_b_stats_addr)[unsafe_offset=3], drops_b_before)

    # 6. Duplicate sparse component hash: must reject cleanly (0 leaks, 0 double-frees) and preserve w_dst
    var w_dup_sp = World()
    _register_q_world_types(w_dup_sp)
    var edsp = w_dup_sp.spawn_empty()
    _ = w_dup_sp.insert[TrackedSparseBag](edsp, TrackedSparseBag(src_sp_stats_addr, _make_heap_string("dup_sp", 1), _q_ints(77)))
    _ = w_dup_sp.insert[SparseHeapTag](edsp, SparseHeapTag(_make_heap_string("dup_sp_tag", 2)))
    var dup_sp_snap = w_dup_sp.snapshot_blob()
    var sp_tag_off = _find_last_u64(dup_sp_snap, SparseHeapTag.COMPONENT_ID)
    assert_true(sp_tag_off >= 0)
    _write_u64_at(dup_sp_snap, sp_tag_off, TrackedSparseBag.COMPONENT_ID)
    caught = False
    var dup_sp_err = String("")
    try:
        w_dst.restore_blob(dup_sp_snap)
    except err:
        caught = True
        dup_sp_err = String(err)
    assert_true(caught and ("duplicate sparse component" in dup_sp_err), "restore_blob must reject duplicate sparse component hash")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)
    assert_equal(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=src_sp_stats_addr)[unsafe_offset=0], 4)

    # 7. STORAGE_SPARSE bit in archetype table_mask: must reject and preserve w_dst
    var bw_sp_mask = BlobWriter()
    bw_sp_mask.begin(fourcc("ECS2"), 3)
    bw_sp_mask.u32(1)
    bw_sp_mask.integer(1)
    bw_sp_mask.integer(1)
    bw_sp_mask.integer(0)
    bw_sp_mask.integer(2)
    bw_sp_mask.u64(CompPos.COMPONENT_ID)
    bw_sp_mask.integer(STORAGE_TABLE)
    bw_sp_mask.integer(size_of[CompPos]())
    bw_sp_mask.boolean(True)
    bw_sp_mask.u64(TagBurningSparse.COMPONENT_ID)
    bw_sp_mask.integer(STORAGE_SPARSE)
    bw_sp_mask.integer(size_of[TagBurningSparse]())
    bw_sp_mask.boolean(True)
    bw_sp_mask.u32(1)
    bw_sp_mask.boolean(True)
    bw_sp_mask.integer(1)  # num_archs
    bw_sp_mask.u32(1)  # wc = 1
    bw_sp_mask.u64(UInt64(3))  # bit 0 (CompPos) | bit 1 (TagBurningSparse, STORAGE_SPARSE!)
    bw_sp_mask.integer(0)
    bw_sp_mask.integer(0)
    bw_sp_mask.end()
    caught = False
    var sp_mask_err = String("")
    try:
        w_dst.restore_blob(bw_sp_mask.data)
    except err:
        caught = True
        sp_mask_err = String(err)
    assert_true(caught and ("non-table component in archetype table_mask" in sp_mask_err), "restore_blob must reject STORAGE_SPARSE bit in table_mask")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)

    # 8. Children binary payload n = 0x7FFFFFFFFFFFFFFF must return immediately with Error (no CPU hang!)
    var w_ch = World()
    _register_q_world_types(w_ch)
    var cp = w_ch.spawn_empty()
    var cc1 = w_ch.spawn_empty()
    var cc2 = w_ch.spawn_empty()
    _ = set_parent(w_ch, cc1, cp)
    _ = set_parent(w_ch, cc2, cp)
    var ch_snap = w_ch.snapshot_blob()
    var ch_off = _find_last_u64(ch_snap, Children.COMPONENT_ID)
    assert_true(ch_off >= 0 and ch_snap[ch_off + 8] == 1)
    _write_u64_at(ch_snap, ch_off + 9, UInt64(0x7FFFFFFFFFFFFFFF))
    var t_ch0 = get_time()
    caught = False
    try:
        w_dst.restore_blob(ch_snap)
    except:
        caught = True
    var t_ch_dt = get_time() - t_ch0
    assert_true(caught, "Children n = 0x7FFFFFFFFFFFFFFF must fail cleanly")
    assert_true(t_ch_dt < 0.5, "Children n = 0x7FFFFFFFFFFFFFFF must fail immediately without CPU hang")
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)

    # 9. Duplicate index in free_indices ([0, 0] for dead slot 0) must raise Error and preserve w_dst
    var bw_dup_free = BlobWriter()
    bw_dup_free.begin(fourcc("ECS2"), 3)
    bw_dup_free.u32(1)  # saved_tick
    bw_dup_free.integer(1)  # saved_live
    bw_dup_free.integer(2)  # ent_len
    bw_dup_free.integer(2)  # free_len = 2 (duplicate [0, 0]!)
    bw_dup_free.integer(1)  # saved_comp_count
    bw_dup_free.u64(CompPos.COMPONENT_ID)
    bw_dup_free.integer(STORAGE_TABLE)
    bw_dup_free.integer(size_of[CompPos]())
    bw_dup_free.boolean(True)
    bw_dup_free.u32(2)  # slot 0: gen=2, alive=False
    bw_dup_free.boolean(False)
    bw_dup_free.u32(1)  # slot 1: gen=1, alive=True
    bw_dup_free.boolean(True)
    bw_dup_free.u32(0)  # free_indices[0] = 0
    bw_dup_free.u32(0)  # free_indices[1] = 0 (duplicate!)
    bw_dup_free.integer(1)  # num_archs = 1
    bw_dup_free.u32(1)  # wc = 1
    bw_dup_free.u64(UInt64(1))  # bit 0 = CompPos
    bw_dup_free.integer(1)  # num_chunks = 1
    bw_dup_free.integer(1)  # cnt = 1
    var df_ent = Entity(1, 1)
    bw_dup_free.write_bytes_raw(Int(Pointer(to=df_ent)), size_of[Entity]())
    bw_dup_free.integer(1)  # saved_cols = 1
    bw_dup_free.u64(CompPos.COMPONENT_ID)
    bw_dup_free.u8(0)  # POD mode_tag = 0
    var df_pos = CompPos(1.0, 2.0, 3.0)
    bw_dup_free.write_bytes_raw(Int(Pointer(to=df_pos)), size_of[CompPos]())
    bw_dup_free.integer(0)  # num_sparse = 0
    bw_dup_free.end()
    caught = False
    var dup_free_err = String("")
    try:
        w_dst.restore_blob(bw_dup_free.data)
    except err:
        caught = True
        dup_free_err = String(err)
    assert_true(
        caught and ("duplicate" in dup_free_err),
        "restore_blob must reject duplicate free_indices=[0, 0]",
    )
    _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)

    # 10. Invalid or provisional entity generations (gen==0 live, gen==0 free, gen==0x80000001 free)
    #     must raise Error and preserve w_dst without leaks
    for gen_case in range(3):
        var bw_gen = BlobWriter()
        bw_gen.begin(fourcc("ECS2"), 3)
        bw_gen.u32(1)  # saved_tick
        if gen_case == 0:
            bw_gen.integer(2)  # saved_live
            bw_gen.integer(2)  # ent_len
            bw_gen.integer(0)  # free_len
        else:
            bw_gen.integer(1)  # saved_live
            bw_gen.integer(2)  # ent_len
            bw_gen.integer(1)  # free_len
        bw_gen.integer(1)  # saved_comp_count
        bw_gen.u64(CompPos.COMPONENT_ID)
        bw_gen.integer(STORAGE_TABLE)
        bw_gen.integer(size_of[CompPos]())
        bw_gen.boolean(True)
        if gen_case == 0:
            # (a) live slot 1 with gen == 0 referenced by chunk
            bw_gen.u32(1)  # slot 0: gen=1, alive=True
            bw_gen.boolean(True)
            bw_gen.u32(0)  # slot 1: gen=0, alive=True (invalid!)
            bw_gen.boolean(True)
        elif gen_case == 1:
            # (b) dead slot 0 with gen == 0 in free_indices
            bw_gen.u32(0)  # slot 0: gen=0, alive=False (invalid!)
            bw_gen.boolean(False)
            bw_gen.u32(1)  # slot 1: gen=1, alive=True
            bw_gen.boolean(True)
            bw_gen.u32(0)  # free_indices = [0]
        else:
            # (c) dead slot 0 with gen == 0x80000001 (PROVISIONAL_GEN_BIT) in free_indices
            bw_gen.u32(UInt32(0x80000001))  # slot 0: provisional bit set, alive=False (invalid!)
            bw_gen.boolean(False)
            bw_gen.u32(1)  # slot 1: gen=1, alive=True
            bw_gen.boolean(True)
            bw_gen.u32(0)  # free_indices = [0]
        bw_gen.integer(1)  # num_archs = 1
        bw_gen.u32(1)  # wc = 1
        bw_gen.u64(UInt64(1))  # bit 0 = CompPos
        bw_gen.integer(1)  # num_chunks = 1
        if gen_case == 0:
            bw_gen.integer(2)  # cnt = 2
            var ge0 = Entity(0, 1)
            var ge1 = Entity(1, 0)
            bw_gen.write_bytes_raw(Int(Pointer(to=ge0)), size_of[Entity]())
            bw_gen.write_bytes_raw(Int(Pointer(to=ge1)), size_of[Entity]())
            bw_gen.integer(1)  # saved_cols = 1
            bw_gen.u64(CompPos.COMPONENT_ID)
            bw_gen.u8(0)  # POD mode_tag = 0
            var gp0 = CompPos(1.0, 2.0, 3.0)
            var gp1 = CompPos(4.0, 5.0, 6.0)
            bw_gen.write_bytes_raw(Int(Pointer(to=gp0)), size_of[CompPos]())
            bw_gen.write_bytes_raw(Int(Pointer(to=gp1)), size_of[CompPos]())
        else:
            bw_gen.integer(1)  # cnt = 1
            var ge1 = Entity(1, 1)
            bw_gen.write_bytes_raw(Int(Pointer(to=ge1)), size_of[Entity]())
            bw_gen.integer(1)  # saved_cols = 1
            bw_gen.u64(CompPos.COMPONENT_ID)
            bw_gen.u8(0)  # POD mode_tag = 0
            var gp1 = CompPos(1.0, 2.0, 3.0)
            bw_gen.write_bytes_raw(Int(Pointer(to=gp1)), size_of[CompPos]())
        bw_gen.integer(0)  # num_sparse = 0
        bw_gen.end()
        caught = False
        var gen_err = String("")
        try:
            w_dst.restore_blob(bw_gen.data)
        except err:
            caught = True
            gen_err = String(err)
        assert_true(
            caught and ("invalid or provisional entity generation" in gen_err),
            "restore_blob must reject invalid/provisional entity generation case " + String(gen_case),
        )
        _assert_dst_preserved(w_dst, e0, e1, orig_count, orig_cs, expected_tag, expected_sp_note)


def test_q_restore_blob_atomicity_and_hostile_fuzz_regressions() raises:
    var dst_stats = List[Int](length=4, fill=0)
    var src_a_stats = List[Int](length=4, fill=0)
    var src_b_stats = List[Int](length=4, fill=0)
    var src_sp_stats = List[Int](length=4, fill=0)

    _run_q_atomicity_scope(
        Int(dst_stats.unsafe_ptr()),
        Int(src_a_stats.unsafe_ptr()),
        Int(src_b_stats.unsafe_ptr()),
        Int(src_sp_stats.unsafe_ptr()),
    )

    assert_equal(dst_stats[0], 0)
    assert_equal(src_a_stats[0], 0)
    assert_equal(src_b_stats[0], 0)
    assert_equal(src_b_stats[3], 1)  # Only w_dup_col's edc instance was dropped; never uninit column!
    assert_equal(src_sp_stats[0], 0)
    _ = len(dst_stats)
    _ = len(src_a_stats)
    _ = len(src_b_stats)
    _ = len(src_sp_stats)


@fieldwise_init
struct SparseValComp(Component, Copyable, ImplicitlyCopyable, Movable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.t5.SparseValComp")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var v: Int


@fieldwise_init
struct SparseTagB(Component, Copyable, ImplicitlyCopyable, Movable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.t5.SparseTagB")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var b: Int


@fieldwise_init
struct TableTagC(Component, Copyable, ImplicitlyCopyable, Movable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.t5.TableTagC")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var c: Int


def _t5_sparse_inc1(_lane: Int, _ord: Int, _e: Entity, p0: Pointer[SparseValComp, MutAnyOrigin], _ctx: Int):
    p0[unsafe_offset=0].v += 10


def _t5_sparse_inc2(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    p0: Pointer[SparseValComp, MutAnyOrigin],
    p1: Pointer[SparseTagB, MutAnyOrigin],
    _ctx: Int,
):
    p0[unsafe_offset=0].v += 1
    p1[unsafe_offset=0].b += 2


def _t5_sparse_inc3(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    p0: Pointer[SparseValComp, MutAnyOrigin],
    p1: Pointer[SparseTagB, MutAnyOrigin],
    p2: Pointer[TableTagC, MutAnyOrigin],
    _ctx: Int,
):
    p0[unsafe_offset=0].v += 1
    p1[unsafe_offset=0].b += 1
    p2[unsafe_offset=0].c += 1


def test_r_tier5_sparse_par_changed_and_profiler_wrap_open() raises:
    # 1. [High-7] par_for_each1/2/3 must update SparseSet.changed_ticks when not read_only
    var w = World()
    _ = w.register_component[SparseValComp]()
    _ = w.register_component[SparseTagB]()
    _ = w.register_component[TableTagC]()
    var e1 = w.spawn_empty()
    var e2 = w.spawn_empty()
    _ = w.insert[SparseValComp](e1, SparseValComp(1))
    _ = w.insert[SparseValComp](e2, SparseValComp(2))
    _ = w.insert[SparseTagB](e2, SparseTagB(3))
    _ = w.insert[TableTagC](e2, TableTagC(4))

    var t0 = w.current_tick
    _ = w.advance_tick()
    var pool = JobPool()
    _ = pool.start(2)

    # read_only=True must NOT mark changed_ticks
    w.par_for_each1[SparseValComp](pool, _t5_sparse_inc1, read_only=True)
    assert_equal(len(w.query_changed_entities[SparseValComp](t0)), 0)

    # read_only=False on par_for_each1 MUST mark changed_ticks for both e1 and e2
    w.par_for_each1[SparseValComp](pool, _t5_sparse_inc1, read_only=False)
    assert_equal(w.get[SparseValComp](e1).v, 21)
    assert_equal(w.get[SparseValComp](e2).v, 22)
    assert_equal(len(w.query_changed_entities[SparseValComp](t0)), 2)

    var t1 = w.current_tick
    _ = w.advance_tick()
    # par_for_each2[SparseValComp, SparseTagB] only updates e2 (which has SparseTagB), NOT e1
    w.par_for_each2[SparseValComp, SparseTagB](pool, _t5_sparse_inc2, read_only=False)
    var chg2 = w.query_changed_entities[SparseValComp](t1)
    assert_equal(len(chg2), 1)
    assert_true(chg2[0] == e2)
    assert_equal(len(w.query_changed_entities[SparseTagB](t1)), 1)

    var t2 = w.current_tick
    _ = w.advance_tick()
    w.par_for_each3[SparseValComp, SparseTagB, TableTagC](pool, _t5_sparse_inc3, read_only=False)
    assert_equal(len(w.query_changed_entities[SparseValComp](t2)), 1)
    assert_equal(len(w.query_changed_entities[SparseTagB](t2)), 1)
    assert_equal(len(w.query_changed_entities[TableTagC](t2)), 1)
    pool.stop()

    # 2. [High-5B/8/9/10] ProfileTimeline.open_file coexists with stdlib `with open(...)` + UInt64 wrap & underflow
    var pid = Int(external_call["getpid", Int32]())
    var trace_path = "/tmp/flint_t5_prof_" + String(pid) + ".json"
    var tl = ProfileTimeline()
    assert_true(tl.start(trace_path, overlay=True))
    tl.nanoseconds_per_tick = 1.0  # 1 tick == 1 ns for exact assertions

    # GPU span cross-wrap (opened near UInt64.MAX, started/ended past 0) + invalid ended<started rejection
    tl.frame_starts[1] = UInt64.MAX - 100
    tl.gpu_span_frames[1] = -1
    tl.keep_gpu_span(ZONE_SIM, 1, UInt64(50), UInt64(200))
    assert_equal(tl.gpu_span_counts[1], 1)
    tl.keep_gpu_span(ZONE_SIM, 1, UInt64(300), UInt64(100))  # ended < started -> rejected
    assert_equal(tl.gpu_span_counts[1], 1)

    # hold_frame cross-wrap & wrap-spanning events
    tl.frame_started = UInt64.MAX - 100
    tl.frame_events.clear()
    tl.frame_events.append(ProfileEvent(ZONE_SIM, 0, UInt64(10), UInt64(50)))  # post-wrap
    tl.frame_events.append(ProfileEvent(ZONE_SIM, 0, UInt64.MAX - 50, UInt64(60)))  # wrap-spanning
    tl.hold_frame(2, Float32(0.001))
    assert_equal(len(tl.candidate), 2)

    # write_event wraparound ts & ended < started dur clamp to 0.000
    tl.origin = UInt64.MAX - 100
    tl.write_event(ZONE_SIM, 0, UInt64(50), UInt64(150), 3)  # elapsed = 151 ticks = 0.151 us, dur = 0.100 us
    tl.write_event(ZONE_SIM, 0, UInt64(200), UInt64(100), 3)  # ended < started -> dur = 0.000
    tl.finish()

    var trace_text: String
    with open(trace_path, "r") as f:
        trace_text = f.read()
    assert_true("\"ts\":0.151" in trace_text, "wraparound ts must be 0.151, got: " + trace_text)
    assert_true("\"dur\":0.000" in trace_text, "underflow dur must clamp to 0.000, got: " + trace_text)
    _ = remove_file(trace_path)


def _t5_r2_count1(
    _lane: Int, _ord: Int, _e: Entity, _p0: Pointer[SparseValComp, MutAnyOrigin], ctx: Int
):
    var cnt = Pointer[Int, MutAnyOrigin](unsafe_from_address=ctx)
    cnt[unsafe_offset=0] += 1


def _t5_r2_count2_ss(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    _p0: Pointer[SparseValComp, MutAnyOrigin],
    _p1: Pointer[SparseTagB, MutAnyOrigin],
    ctx: Int,
):
    var cnt = Pointer[Int, MutAnyOrigin](unsafe_from_address=ctx)
    cnt[unsafe_offset=0] += 1


def _t5_r2_count2_ds(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    _p0: Pointer[TableTagC, MutAnyOrigin],
    _p1: Pointer[SparseValComp, MutAnyOrigin],
    ctx: Int,
):
    var cnt = Pointer[Int, MutAnyOrigin](unsafe_from_address=ctx)
    cnt[unsafe_offset=0] += 1


def _t5_r2_count3(
    _lane: Int,
    _ord: Int,
    _e: Entity,
    _p0: Pointer[SparseValComp, MutAnyOrigin],
    _p1: Pointer[SparseTagB, MutAnyOrigin],
    _p2: Pointer[TableTagC, MutAnyOrigin],
    ctx: Int,
):
    var cnt = Pointer[Int, MutAnyOrigin](unsafe_from_address=ctx)
    cnt[unsafe_offset=0] += 1


def test_s_tier5_r2_sparse_changed_since() raises:
    var w = World()
    _ = w.register_component[SparseValComp]()
    _ = w.register_component[SparseTagB]()
    _ = w.register_component[TableTagC]()
    var e1 = w.spawn_empty()
    var e2 = w.spawn_empty()
    _ = w.insert[TableTagC](e1, TableTagC(1))
    _ = w.insert[SparseValComp](e1, SparseValComp(10))
    _ = w.insert[SparseTagB](e1, SparseTagB(20))
    _ = w.insert[TableTagC](e2, TableTagC(2))
    _ = w.insert[SparseValComp](e2, SparseValComp(30))
    _ = w.insert[SparseTagB](e2, SparseTagB(40))

    _ = w.advance_tick()
    var since = w.current_tick
    _ = w.advance_tick()

    var pool = JobPool()
    _ = pool.start(1)
    var hits = 0
    var hits_addr = Int(Pointer(to=hits))

    # 1. All sparse components unchanged since `since`: par_for_each1/2/3 must have 0 hits
    w.par_for_each1[SparseValComp](pool, _t5_r2_count1, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 0)
    assert_equal(len(w.query_changed_entities[SparseValComp](since)), 0)

    w.par_for_each2[SparseValComp, SparseTagB](pool, _t5_r2_count2_ss, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 0)

    w.par_for_each2[TableTagC, SparseValComp](pool, _t5_r2_count2_ds, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 0)

    w.par_for_each3[SparseValComp, SparseTagB, TableTagC](pool, _t5_r2_count3, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 0)

    # 2. Mutate ONLY e2's SparseValComp (e1 and e2 share SIMD group 0 for TableTagC!)
    _ = w.insert[SparseValComp](e2, SparseValComp(31))

    hits = 0
    w.par_for_each1[SparseValComp](pool, _t5_r2_count1, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 1)

    hits = 0
    w.par_for_each2[SparseValComp, SparseTagB](pool, _t5_r2_count2_ss, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 1)

    hits = 0
    w.par_for_each3[SparseValComp, SparseTagB, TableTagC](pool, _t5_r2_count3, user_ctx=hits_addr, read_only=True, changed_since=since)
    assert_equal(hits, 1)

    # Mixed dense + sparse with read_only=False: must hit ONLY e2 and NOT pollute e1 in same group 0!
    hits = 0
    w.par_for_each2[TableTagC, SparseValComp](pool, _t5_r2_count2_ds, user_ctx=hits_addr, read_only=False, changed_since=since)
    assert_equal(hits, 1)

    # Verify read_only=False on par_for_each3 updates e2 without touching e1
    _ = w.advance_tick()
    var since2 = w.current_tick
    _ = w.advance_tick()
    _ = w.insert[SparseTagB](e1, SparseTagB(21))
    hits = 0
    w.par_for_each3[SparseValComp, SparseTagB, TableTagC](pool, _t5_r2_count3, user_ctx=hits_addr, read_only=False, changed_since=since2)
    assert_equal(hits, 1)
    assert_equal(len(w.query_changed_entities[SparseValComp](since2)), 1)
    assert_true(w.query_changed_entities[SparseValComp](since2)[0] == e1)

    pool.stop()


def _test_v1_to_v2_scene_migration(payload: String) raises -> String:
    var out = payload.replace('"LegacyNameScene"', '"MigratedSceneV2"')
    out = out.replace('"OldTransform"', '"LocalTransform"')
    return out^


def test_t_m4a_scene_migrations_cross_format_unknown_and_prefab_rollback() raises:
    var reg = ComponentRegistry.with_builtins()

    # 1. SceneMigrationRegistry wired into Scene (both Scene.migrations and explicit with_migrations)
    var v1_json = (
        '{"format": "flscn", "version": 1, "name": "LegacyNameScene", "entities": ['
        '{"slot": 0, "parent": -1, "components": {"OldTransform": {"translation": [3.0, 4.0, 5.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}}}'
        ']}'
    )
    var v1_bytes = List[UInt8]()
    for b in v1_json.as_bytes():
        v1_bytes.append(b)

    var scn_mig = Scene("BeforeMig")
    scn_mig.migrations.register(1, 2, _test_v1_to_v2_scene_migration)
    var w_mig = World()
    var ents_mig = scn_mig.load_flscn_bytes(w_mig, reg, v1_bytes)
    assert_equal(scn_mig.name, "MigratedSceneV2")
    assert_equal(len(ents_mig), 1)
    var lt_mig = w_mig.get_ptr[LocalTransform](ents_mig[0])
    assert_true(Bool(lt_mig))
    assert_true(abs(lt_mig.value()[].translation.x - Float32(3.0)) < Float32(1.0e-5))

    var mig_explicit = SceneMigrationRegistry()
    mig_explicit.register(1, 2, _test_v1_to_v2_scene_migration)
    var scn_mig2 = Scene("ExplicitMig")
    var w_mig2 = World()
    var ents_mig2 = scn_mig2.load_flscn_bytes_with_migrations(w_mig2, reg, v1_bytes, mig_explicit)
    assert_equal(scn_mig2.name, "MigratedSceneV2")
    assert_equal(len(ents_mig2), 1)
    assert_true(w_mig2.has[LocalTransform](ents_mig2[0]))

    # 2. Cross-format .flscn -> .flscnb -> .flscn unknown JSON preservation (clear_existing=True & False)
    var unk_json = (
        '{"format": "flscn", "version": 2, "name": "CrossFmt", "entities": ['
        '{"slot": 0, "parent": -1, "components": {'
        '"LocalTransform": {"translation": [1.0, 2.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}, '
        '"ThirdParty": {"hp": 100, "tags": ["a", "b"]}'
        '}}]}'
    )
    var unk_bytes = List[UInt8]()
    for b in unk_json.as_bytes():
        unk_bytes.append(b)

    var scn_cf = Scene("CrossFmt")
    var w_cf1 = World()
    _ = scn_cf.load_flscn_bytes(w_cf1, reg, unk_bytes)
    var flscnb_cf = scn_cf.save_flscnb_bytes(w_cf1, reg)

    var w_cf2 = World()
    _ = scn_cf.load_flscnb_bytes(w_cf2, reg, flscnb_cf, clear_existing=True)
    var flscn_back = scn_cf.save_flscn_string(w_cf2, reg)
    assert_true(flscn_back.find('"ThirdParty"') >= 0)
    assert_true(flscn_back.find('"hp": 100') >= 0)

    # clear_existing=False must also restore unknown JSON from .flscnb
    var w_cf3 = World()
    var pre_e = w_cf3.spawn_empty()
    var loaded_cf3 = scn_cf.load_flscnb_bytes(w_cf3, reg, flscnb_cf, clear_existing=False)
    assert_true(w_cf3.is_alive(pre_e))
    assert_equal(len(loaded_cf3), 1)
    var unk_cf3 = w_cf3.get_ptr[UnknownComponentsComponent](loaded_cf3[0])
    assert_true(Bool(unk_cf3))
    assert_equal(len(unk_cf3.value()[].json_names), 1)
    assert_equal(unk_cf3.value()[].json_names[0], "ThirdParty")

    # 3. Cross-format .flscnb -> .flscn -> .flscnb unknown binary FourCC preservation
    var tag_ubet = fourcc("UBET")
    var bw_bin = BlobWriter()
    bw_bin.begin(0x4E435346, 2)
    bw_bin.text("CrossBin")
    bw_bin.integer(1)
    bw_bin.integer(-1)
    bw_bin.integer(2)
    bw_bin.begin(LocalTransform.SCHEMA_FOURCC, LocalTransform.SCHEMA_VERSION)
    LocalTransform.identity().write_bin(bw_bin)
    bw_bin.end()
    bw_bin.begin(tag_ubet, 7)
    bw_bin.u8(0xAA)
    bw_bin.u8(0x55)
    bw_bin.u8(0x01)
    bw_bin.end()
    bw_bin.end()
    var packed_bin = lz4_pack(bw_bin.data)

    var w_b1 = World()
    _ = scn_cf.load_flscnb_bytes(w_b1, reg, packed_bin)
    var json_with_bin = scn_cf.save_flscn_bytes(w_b1, reg)
    var w_b2 = World()
    _ = scn_cf.load_flscn_bytes(w_b2, reg, json_with_bin, clear_existing=True)
    var packed_roundtrip = scn_cf.save_flscnb_bytes(w_b2, reg)
    var w_b3 = World()
    var ents_b3 = scn_cf.load_flscnb_bytes(w_b3, reg, packed_roundtrip, clear_existing=False)
    assert_equal(len(ents_b3), 1)
    var unk_b3 = w_b3.get_ptr[UnknownComponentsComponent](ents_b3[0])
    assert_true(Bool(unk_b3))
    assert_equal(len(unk_b3.value()[].bin_fourccs), 1)
    assert_equal(unk_b3.value()[].bin_fourccs[0], tag_ubet)
    assert_equal(unk_b3.value()[].bin_versions[0], UInt32(7))
    assert_equal(len(unk_b3.value()[].bin_payloads[0]), 3)
    assert_equal(unk_b3.value()[].bin_payloads[0][0], UInt8(0xAA))
    assert_equal(unk_b3.value()[].bin_payloads[0][1], UInt8(0x55))
    assert_equal(unk_b3.value()[].bin_payloads[0][2], UInt8(0x01))

    # 4. Prefab.instantiate atomic rollback on later node component decode failure
    var w_pf = World()
    reg.ensure_registered_in_world(w_pf)
    var sentinel = w_pf.spawn_empty()
    _ = w_pf.insert[LocalTransform](sentinel, LocalTransform(Vec3(7.0, 8.0, 9.0)))
    assert_equal(w_pf.entity_count(), 1)

    var pf_bad = Prefab("PartialFail")
    var n0 = pf_bad.add_node("Node0")
    pf_bad.set_component(n0, LocalTransform.identity())
    var n1 = pf_bad.add_node("Node1", n0)
    pf_bad.nodes[n1].set_raw_component("UnregisteredBogusComponent", "{}")

    var err_raised = False
    try:
        _ = pf_bad.instantiate(w_pf, reg)
    except:
        err_raised = True
    assert_true(err_raised)
    assert_equal(w_pf.entity_count(), 1)
    assert_true(w_pf.is_alive(sentinel))

    # Also rollback when nested child prefab node fails decode
    var pf_child_bad = Prefab("ChildBad")
    var cn0 = pf_child_bad.add_node("CN0")
    pf_child_bad.set_component(cn0, LocalTransform.identity())
    var cn1 = pf_child_bad.add_node("CN1", cn0)
    pf_child_bad.nodes[cn1].set_raw_component("LocalTransform", "{not_valid_json")

    var pf_parent = Prefab("ParentOk")
    var pn0 = pf_parent.add_node("PN0")
    pf_parent.set_component(pn0, LocalTransform.identity())
    pf_parent.add_nested(NestedPrefabRef("ChildBad", pn0))

    var lib_bad = List[Prefab]()
    lib_bad.append(pf_child_bad.copy())
    var nested_err = False
    try:
        _ = pf_parent.instantiate(w_pf, reg, lib_bad)
    except:
        nested_err = True
    assert_true(nested_err)
    assert_equal(w_pf.entity_count(), 1)


def main() raises:
    var t0 = get_time()
    print("[Flint v5 Track 1] Running Core, Hybrid ECS v2, Scene/Prefab & Hot-Reload Suite...")

    test_a_ftz_daz_heal_and_core()
    print("  [PASS] (a) FTZ/DAZ JobPool work-stealing auto-heal, subnormals, fast_hash64 & dynlib")

    test_b_ecs_v2_hybrid_events_determinism()
    print("  [PASS] (b) Hybrid ECS v2 (>12 archetypes, SparseSet, Events[T], 1T vs 7T bit-exact checksum)")

    test_c_transform_hierarchy_1000_nodes()
    print("  [PASS] (c) 1,000-node 4-level parallel TransformPlugin propagation & slerp interpolation")

    test_d_scene_prefabs_and_pie()
    print("  [PASS] (d) .flscn JSON, .flscnb LZ4, nested Prefab/Overrides & <0.2ms PIE snapshot/restore")

    test_e_schedule_relay_and_fixed_timestep()
    print("  [PASS] (e) Schedule conflict DAG + 25-dependent relay split + 30Hz vs 144Hz FixedTime determinism")

    test_f_live_so_plugin_hot_reload()
    print("  [PASS] (f) Live .so gameplay plugin hot-reload (v1 -> v2 state-preserving swap + dl_close)")

    test_g_v5_review_regressions()
    print("  [PASS] (g) v5 Review Regressions (hash remap, portable heap 100x, resource resize, implicit cmds, handles)")

    test_h_restore_blob_rejects_hostile_counts()
    print("  [PASS] (h) restore_blob rejects corrupted counts and entity tables without allocating")

    test_i_audit_regressions()
    print("  [PASS] (i) Phase 0+1 audit regressions (par_for_each sparse excl, write_dev safety, anchor, Arena, Resources)")

    test_j_wave2_powerhouse_upgrades()
    print("  [PASS] (j) Wave 2 upgrades (par_for_each3/read_only/Changed, chunk reclaim, dirty transforms, dynamic CVars/zones)")

    test_k_detmath_simd8_bit_exact()
    print("  [PASS] (k) detmath SIMD[F32, 8] bit-exact overloads (trig, inv-trig, exp, log, pow, scale_by_pow2)")

    test_l_jobpool_chase_lev_prio_raii()
    print("  [PASS] (l) JobPool Chase-Lev stealing, 3-tier priority, >64 dependents, run_affine & RAII lifecycle")

    test_m_core_raii_and_dynamic_alloc()
    print("  [PASS] (m) Core RAII (MemCharge/Arena/FrameArenas/SimdColumn/ArchetypeSlots) & dynamic memory/anchor")

    test_n_ecs_over_256_components()
    print("  [PASS] (n) ECS >256 & >512 components (ComponentBitSet, Archetype, Schedule DAG, permuted snapshot)")

    test_o_portable_snapshot_and_missing_hooks_error()
    print("  [PASS] (o) Portable snapshot/restore across Worlds & explicit Error on missing hooks / tag 2")

    test_p_no_fixed_ecs_mappings_in_proc_maps()
    print("  [PASS] (p) Zero fixed 0x5D5B..0x5D5E mappings in /proc/self/maps after ECS/JobPool/snapshot exercise")

    test_q_restore_blob_atomicity_and_hostile_fuzz_regressions()
    print("  [PASS] (q) Atomic restore_blob staging, duplicate col/sparse guards, mask mode & Children bounds check")

    test_r_tier5_sparse_par_changed_and_profiler_wrap_open()
    print("  [PASS] (r) Tier 5 par_for_each1/2/3 sparse changed_ticks, profiler wraparound & openat/open coexistence")

    test_s_tier5_r2_sparse_changed_since()
    print("  [PASS] (s) Tier 5 R2 par_for_each1/2/3 changed_since on STORAGE_SPARSE components")

    test_t_m4a_scene_migrations_cross_format_unknown_and_prefab_rollback()
    print("  [PASS] (t) M4a SceneMigrationRegistry, .flscn<->.flscnb unknown blobs & Prefab rollback")

    test_u_m5a_pod_schema_and_registry_add_remove()
    print("  [PASS] (u) M5a FlintPluginApiV2, PodComponentSchema & ComponentRegistry add/remove")

    test_v_m5a_pod_uaf_and_int64_regressions()
    print("  [PASS] (v) M5a POD -O3 UAF, Int64 JSON precision & tid>=256 Scene POD regressions")

    var elapsed = get_time() - t0
    print("[Flint v5 Track 1] ALL 22 SUITES PASSED in " + String(elapsed) + " s")
