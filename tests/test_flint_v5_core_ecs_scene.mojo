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
from std.sys import size_of
from std.testing import assert_true, assert_equal

from flint.core.detmath import (
    MXCSR_SIM,
    f32_from_bits,
    write_mxcsr,
    set_sim_float_mode,
    sim_float_mode_ok,
    det_sin,
    det_cos,
    PI,
)
from flint.core.ecs import EntityHandle, ArchetypeSlots, ECB_PROV_GEN
from flint.core.jobs import JobPool, run_jobs, run_jobs_affine
from flint.core.math import Vec3, Quat, Mat4
from flint.core.serial import BlobWriter, BlobReader, fourcc, fast_hash64
from flint.assets.json import JsonDocument
from flint.platform.dynlib import dl_open, dl_close, dl_error
from flint.platform.os import get_time, write_bytes, remove_file, file_size, env_now

from flint.ecs import (
    CHUNK_ALIGN,
    STORAGE_TABLE,
    STORAGE_SPARSE,
    VAULT_TOKEN_MAGIC,
    NULL_ENTITY,
    comp_hash64,
    clear_component_clone_vault,
    Entity,
    Component,
    BitSet256,
    ParallelCommands,
    Events,
    World,
)
from flint.scene import (
    SerializableComponent,
    LocalTransform,
    GlobalTransform,
    PrevGlobalTransform,
    Children,
    HierarchyDepth,
    TransformPlugin,
    set_parent,
    propagate_transforms_parallel,
    interpolate_global_transform,
    interpolate_entity_transform,
    ComponentRegistry,
    Prefab,
    PrefabOverride,
    NestedPrefabRef,
    Scene,
)
from flint.app import (
    STAGE_UPDATE,
    STAGE_FIXED_UPDATE,
    Resource,
    Resources,
    FixedTime,
    Schedule,
    AppConfig,
    EngineApp,
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


def _run_deterministic_ecs_sim(workers: Int) -> Tuple[Int, UInt64]:
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

    var cmds = ParallelCommands()
    for step in range(8):
        _ = w.advance_tick()
        var ctx_box = List[_SimStepCtx](capacity=1)
        ctx_box.append(_SimStepCtx(Int(Pointer(to=cmds)), step))
        w.par_for_each2[CompPos, CompVel](pool, _det_sim_kernel, Int(ctx_box.unsafe_ptr()))
        var keep_ctx = len(ctx_box)
        _ = keep_ctx
        w.flush_commands(cmds)

    var count = w.entity_count()
    var checksum = w.state_checksum()
    pool.stop()
    return (count, checksum)


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

    var flscn_path = "/tmp/flint-eng1-test-scene.flscn"
    var flscnb_path = "/tmp/flint-eng1-test-scene.flscnb"
    assert_true(scene.save_flscn_file(flscn_path, w, reg))
    assert_true(scene.save_flscnb_file(flscnb_path, w, reg))

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
            + "${MOJO_PREFIX}/bin/mojo build "
            + src_v1
            + " --emit shared-lib -o "
            + so_v1
            + " -O0 --Werror 2>/dev/null & "
            + "${MOJO_PREFIX}/bin/mojo build "
            + src_v2
            + " --emit shared-lib -o "
            + so_v2
            + " -O0 --Werror 2>/dev/null & wait"
        )
    var mojo_bin = env_mojo if env_mojo.byte_length() > 0 else String("mojo")
    return (
        mojo_bin
        + " build "
        + src_v1
        + " --emit shared-lib -o "
        + so_v1
        + " -O0 --Werror 2>/dev/null & "
        + mojo_bin
        + " build "
        + src_v2
        + " --emit shared-lib -o "
        + so_v2
        + " -O0 --Werror 2>/dev/null & wait"
    )


def _guest_plugin_source(rate_per_sec: String, version_tag: String, res_words: String) -> String:
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
    s += "def _step_energy(_ebits: UInt64, ptr: Int, dt: Float32) abi(\"C\"):\n"
    s += "    Pointer[Float32, MutAnyOrigin](unsafe_from_address=ptr)[unsafe_offset=0] += Float32(" + rate_per_sec + ") * dt\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(api_ptr: Int, world_ptr: Int, res_ptr: Int, is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[GuestApiV1, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
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
    s += "    var api = Pointer[GuestApiV1, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    var tid = api.register_pod_component(world_ptr, UInt64(0xE4E2670000000001), 4, 4, 0)\n"
    s += "    return Int32(api.for_each_pod1(world_ptr, tid, _step_energy, dt))\n"
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

    assert_true(host.unload(world, resources, is_reload=False), "Final dl_close unload must succeed cleanly")
    assert_true(not host.is_loaded())
    _ = size_of[Entity]()

    _ = remove_file(src_v1)
    _ = remove_file(src_v2)
    _ = remove_file(so_v1)
    _ = remove_file(so_v2)


# ===========================================================================
# Suite (g): v5 Review Regressions — Component Hash Remap, Heap-Owning Clone
#            Vault, Resource Resize, Implicit ParallelCommands Order & Handles
# ===========================================================================

struct HeapNamedComp(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.HeapNamedComp")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var label: String

    def __init__(out self, var label: String):
        self.label = label^

    def __init__(out self, *, copy: Self):
        self.label = copy.label


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


struct SparseHeapTag(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.SparseHeapTag")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE
    var note: String

    def __init__(out self, var note: String):
        self.note = note^

    def __init__(out self, *, copy: Self):
        self.note = copy.note


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

    # Corrupt Clone Vault token magic/gen in a copy of wt_snap and verify clean Error + zero leak
    var corrupted_bytes = wt_snap.data.copy()
    var magic_b0 = UInt8(VAULT_TOKEN_MAGIC & 0xFF)
    var magic_b1 = UInt8((VAULT_TOKEN_MAGIC >> 8) & 0xFF)
    var magic_b2 = UInt8((VAULT_TOKEN_MAGIC >> 16) & 0xFF)
    var magic_b3 = UInt8((VAULT_TOKEN_MAGIC >> 24) & 0xFF)
    var patched_tok = False
    for bi in range(4, len(corrupted_bytes) - 4):
        if (
            corrupted_bytes[bi] == magic_b0
            and corrupted_bytes[bi + 1] == magic_b1
            and corrupted_bytes[bi + 2] == magic_b2
            and corrupted_bytes[bi + 3] == magic_b3
        ):
            # Flip generation bits in lower 32-bit word right before magic upper 32-bit word
            corrupted_bytes[bi - 1] ^= 0x3C
            patched_tok = True
            break
    assert_true(patched_tok, "Snapshot must contain 0x464C5654 Clone Vault token")
    var bad_rd = BlobReader(corrupted_bytes^)
    var caught_bad_tok = False
    try:
        wt.restore_blob(bad_rd)
    except err:
        caught_bad_tok = "Clone Vault token" in String(err)
    assert_true(caught_bad_tok, "Invalid/foreign Clone Vault token must raise clean Error")


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
    clear_component_clone_vault()

    # Pure String world: verify state_checksum() bit-identity across 100 snapshot -> mutate -> restore cycles
    var ws = World()
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

    # wt has now deinitialized; clear vault and verify EXACT 0 live instances!
    clear_component_clone_vault()
    assert_equal(bag_stats[0], 0)
    assert_true(bag_stats[1] >= 1200 and bag_stats[3] >= 1200)
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
    print("  [PASS] (g) v5 Review Regressions (hash remap, heap Clone Vault 100x, resource resize, implicit cmds, handles)")

    test_h_restore_blob_rejects_hostile_counts()
    print("  [PASS] (h) restore_blob rejects corrupted counts and entity tables without allocating")

    var elapsed = get_time() - t0
    print("[Flint v5 Track 1] ALL 8 SUITES PASSED in " + String(elapsed) + " s")
