# Verifies M1 Features: F1, F2, F3, F4, F5, F6, F7, F8, F9
"""Gated E2E Contract Verification Suite for Milestone M1 (Core, ECS, Jobs, Math, Memory & Net Depth).

Executed in default mode as [GATED: PENDING] while M1 is in progress, and enforced
under `run_e2e_opaque.sh --strict` once M1 lands.
"""
from std.atomic import Atomic
from std.collections import List
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.app import STAGE_UPDATE, Resources, Schedule
from flint.ecs import (
    STORAGE_TABLE,
    STORAGE_SPARSE,
    comp_hash64,
    Component,
    ComponentBitSet,
    Entity,
    ParallelCommands,
    World,
)
from flint.scene import ComponentRegistry, SerializableComponent
from flint.core.detmath import (
    det_acos,
    det_asin,
    det_atan,
    det_atan2,
    det_cos,
    det_exp,
    det_log,
    det_pow,
    det_sin,
    f32_bits,
    quiet_nan,
    set_sim_float_mode,
)
from flint.core.ecs import ArchetypeSlots, SimdColumn
from flint.core.jobs import (
    JOB_PRIO_HIGH,
    JOB_PRIO_LOW,
    JOB_PRIO_NORMAL,
    JobPool,
)
from flint.core.math import Quat, Vec3
from flint.core.memory import Arena, FrameArenas, MEM_FRAME, MemCharge, mem_live
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.assets.json import JsonDocument

from flint.net import (
    BitReader,
    BitWriter,
    DatagramView,
    NetEndpoint,
    RollbackSession,
    SnapshotClientReceiver,
    SnapshotReplicator,
    VirtualEndpoint,
    VirtualNetSim,
    VisualSmoother,
)
from flint.net.rollback import PhysicsRollbackAdapter, PhysicsRollbackInput
from flint.net.prediction import ClientPredictionBuffer, PredictedState, PredictionStepper
from flint.net.lag_comp import HitboxPose, HitboxSnapshotRing, LagCompHistory
from flint.net.relay import (
    NATPunchProbe,
    PUNCH_STATE_DIRECT,
    PUNCH_STATE_RELAY_FALLBACK,
    RelayForwardHeader,
    RelayServer,
)


# ===========================================================================
# F1: Vault-Free Portable World.snapshot_blob / restore_blob
# ===========================================================================

struct M1PortableBag(SerializableComponent):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.m1.M1PortableBag")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "M1PortableBag"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("M1BG")
    comptime SCHEMA_VERSION: Int = 1

    var label: String
    var items: List[Int32]

    def __init__(out self, var label: String = "", var items: List[Int32] = List[Int32]()):
        self.label = label^
        self.items = items^

    def __init__(out self, *, copy: Self):
        self.label = copy.label
        self.items = copy.items.copy()

    def write_bin(self, mut out: BlobWriter):
        out.text(self.label)
        out.u32(UInt32(len(self.items)))
        for i in range(len(self.items)):
            out.u32(UInt32(self.items[i]))

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var lbl = reader.text()
        var cnt = Int(reader.u32())
        var vals = List[Int32](capacity=cnt)
        for _ in range(cnt):
            vals.append(Int32(reader.u32()))
        return Self(lbl^, vals^)

    def write_json(self) -> String:
        return "{\"label\": \"" + self.label + "\"}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        return Self(doc.text(doc.object_get(node, "label"), ""), List[Int32]())


struct M1UnhookedHeapComp(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.m1.M1UnhookedHeapComp")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var text: String

    def __init__(out self, var text: String = ""):
        self.text = text^

    def __init__(out self, *, copy: Self):
        self.text = copy.text


def test_f1_vault_free_snapshot_blob() raises:
    var reg = ComponentRegistry.with_builtins()
    reg.register[M1PortableBag]()

    var w_src = World()
    reg.ensure_registered_in_world(w_src)
    var e = w_src.spawn_empty()
    var vals = List[Int32]()
    vals.append(101)
    vals.append(202)
    vals.append(303)
    assert_true(w_src.insert[M1PortableBag](e, M1PortableBag("portable_heap_payload", vals^)))

    var writer = BlobWriter()
    w_src.snapshot_blob(writer)

    # Restore into an independent destination World without any Clone Vault
    var w_dst = World()
    reg.ensure_registered_in_world(w_dst)
    var reader = BlobReader(writer.data.copy())
    w_dst.restore_blob(reader)

    assert_equal(w_dst.entity_count(), 1)
    assert_true(w_dst.has[M1PortableBag](e))
    ref bag = w_dst.get_ptr[M1PortableBag](e).value()[unsafe_offset=0]
    assert_equal(bag.label, "portable_heap_payload")
    assert_equal(len(bag.items), 3)
    assert_equal(Int(bag.items[2]), 303)
    assert_equal(w_dst.state_checksum(), w_src.state_checksum())

    # Negative: unregistered/unhooked non-trivial component must fail explicitly (no Clone Vault fallback)
    var w_bad = World()
    var e_bad = w_bad.spawn_empty()
    assert_true(w_bad.insert[M1UnhookedHeapComp](e_bad, M1UnhookedHeapComp("no_hooks")))
    var bad_writer = BlobWriter()
    var raised_on_unhooked = False
    try:
        w_bad.snapshot_blob(bad_writer)
    except:
        raised_on_unhooked = True
    assert_true(raised_on_unhooked)


# ===========================================================================
# F2: Lift MAX_COMPONENTS = 256 Cap + ComponentBitSet (>= 512 bits)
# ===========================================================================

def f2_writer_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].store(11)


def f2_reader_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    var v = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].load()
    if v == 11:
        Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].store(22)


def test_f2_extensible_component_bitset_over_256() raises:
    var bs_a = ComponentBitSet()
    var bs_b = ComponentBitSet()
    bs_a.set(300)
    bs_a.set(450)
    assert_true(bs_a.has(300) and bs_a.has(450))
    bs_b.set(300)
    assert_true(bs_a.contains_all(bs_b))
    assert_true(bs_a.intersects(bs_b))
    bs_a.clear(300)
    assert_false(bs_a.has(300))
    bs_a.clear_all()
    assert_false(bs_a.has(450))

    var w = World()
    var tids = List[Int](capacity=320)
    for i in range(320):
        var mode = STORAGE_SPARSE if i >= 300 else STORAGE_TABLE
        var tid = w.register_pod_component(UInt64(0xE2E20000 + i), 4, 4, mode)
        assert_true(tid >= 0)
        tids.append(tid)

    var e = w.spawn_empty()
    var val_box = List[Int32](length=1, fill=9876)
    assert_true(w.insert_raw_pod(e, tids[290], Int(val_box.unsafe_ptr())))
    assert_true(w.insert_raw_pod(e, tids[310], Int(val_box.unsafe_ptr())))
    _ = len(val_box)
    assert_true(w.has_by_type_id(e, tids[290]))
    assert_true(w.has_by_type_id(e, tids[310]))

    var q_mask = ComponentBitSet()
    q_mask.set(tids[290])
    q_mask.set(tids[310])
    var found = w.query_entities(q_mask)
    assert_equal(len(found), 1)
    assert_true(found[0] == e)

    # Snapshot + restore with > 256 components
    var writer = BlobWriter()
    w.snapshot_blob(writer)
    var w2 = World()
    for i in range(320):
        var mode = STORAGE_SPARSE if i >= 300 else STORAGE_TABLE
        _ = w2.register_pod_component(UInt64(0xE2E20000 + i), 4, 4, mode)
    var reader = BlobReader(writer.data.copy())
    w2.restore_blob(reader)
    assert_equal(w2.state_checksum(), w.state_checksum())

    # Schedule conflict detection on bit >= 256
    var sched = Schedule()
    var res = Resources()
    var cmds = ParallelCommands()
    var pool = JobPool()
    _ = pool.start(2)
    var ctx = List[Int64](length=2, fill=0)
    var c_addr = Int(ctx.unsafe_ptr())
    _ = sched.add_system(STAGE_UPDATE, "w300", f2_writer_sys, writes=q_mask, user_ctx=c_addr)
    _ = sched.add_system(STAGE_UPDATE, "r300", f2_reader_sys, reads=q_mask, user_ctx=c_addr)
    sched.run_stage(STAGE_UPDATE, w, res, pool, cmds)
    assert_equal(Int(ctx[0]), 11)
    assert_equal(Int(ctx[1]), 22)
    _ = len(ctx)


# ===========================================================================
# F3: JobPool Chase-Lev + Priorities + Dynamic Dependents + RAII __deinit__
# ===========================================================================

def f3_fan_root(ctx: Int, _c: Int):
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].store(1)


def f3_fan_leaf(ctx: Int, _c: Int):
    var root_val = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].load()
    if root_val == 1:
        _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx + 8)[unsafe_offset=0].fetch_add(1)


def f3_prio_noop(_ctx: Int, _c: Int):
    pass


def _f3_raii_scope_run(s_addr: Int) raises:
    var pool = JobPool()
    assert_true(pool.start(4))

    var root = pool.create(f3_fan_root, s_addr, 1)
    var leaves = List[Int](capacity=24)
    for _ in range(24):  # > old MAX_DEPENDENTS = 16
        var j = pool.create(f3_fan_leaf, s_addr, 1)
        pool.depend(j, root)
        leaves.append(j)
    for i in range(24):
        pool.submit(leaves[i])
    pool.submit(root, priority=JOB_PRIO_HIGH)
    for i in range(24):
        pool.wait(leaves[i])
    pool.wait(root)

    var p_low = pool.create(f3_prio_noop, 0, 1)
    var p_norm = pool.create(f3_prio_noop, 0, 1)
    var p_high = pool.create(f3_prio_noop, 0, 1)
    pool.submit(p_low, priority=JOB_PRIO_LOW)
    pool.submit(p_norm, priority=JOB_PRIO_NORMAL)
    pool.submit(p_high, priority=JOB_PRIO_HIGH)
    pool.wait(p_low)
    pool.wait(p_norm)
    pool.wait(p_high)


def test_f3_jobpool_raii_dynamic_dependents_and_priorities() raises:
    var state = List[Int64](length=2, fill=0)
    var s_addr = Int(state.unsafe_ptr())

    # Scope exits WITHOUT explicit pool.stop() -> RAII __deinit__ cleans up workers
    _f3_raii_scope_run(s_addr)

    assert_equal(Int(state[0]), 1)
    assert_equal(Int(state[1]), 24)
    _ = len(state)


# ===========================================================================
# F4: detmath SIMD[DType.float32, 8] Bit-Exact Overloads
# ===========================================================================

def test_f4_detmath_simd8_bit_exact() raises:
    var vx = SIMD[DType.float32, 8](-1.5, -0.75, -0.25, 0.0, 0.25, 0.5, 0.85, 1.25)
    var vy = SIMD[DType.float32, 8](0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
    var vu = SIMD[DType.float32, 8](-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.75, 0.99)
    var vp = SIMD[DType.float32, 8](0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5)

    var s8 = det_sin(vx)
    var c8 = det_cos(vx)
    var at8 = det_atan(vx)
    var at2_8 = det_atan2(vy, vx)
    var as8 = det_asin(vu)
    var ac8 = det_acos(vu)
    var ex8 = det_exp(vx)
    var lg8 = det_log(vp)
    var pw8 = det_pow(vp, vy)

    for lane in range(8):
        assert_equal(f32_bits(s8[lane]), f32_bits(det_sin(vx[lane])))
        assert_equal(f32_bits(c8[lane]), f32_bits(det_cos(vx[lane])))
        assert_equal(f32_bits(at8[lane]), f32_bits(det_atan(vx[lane])))
        assert_equal(f32_bits(at2_8[lane]), f32_bits(det_atan2(vy[lane], vx[lane])))
        assert_equal(f32_bits(as8[lane]), f32_bits(det_asin(vu[lane])))
        assert_equal(f32_bits(ac8[lane]), f32_bits(det_acos(vu[lane])))
        assert_equal(f32_bits(ex8[lane]), f32_bits(det_exp(vx[lane])))
        assert_equal(f32_bits(lg8[lane]), f32_bits(det_log(vp[lane])))
        assert_equal(f32_bits(pw8[lane]), f32_bits(det_pow(vp[lane], vy[lane])))


# ===========================================================================
# F5: Core/ECS/Memory RAII __deinit__ & No Fixed 0x5D5B..0x5D5E
# ===========================================================================

def _f5_raii_scope_run(baseline_frame: Int) raises:
    var chg = MemCharge(MEM_FRAME, 2048)
    assert_true(mem_live(MEM_FRAME) >= baseline_frame + 2048)
    var arena = Arena(MEM_FRAME, 4096)
    assert_true(arena.alloc(128, 64) != 0)
    var frames = FrameArenas(2, 1024)
    frames.begin(0)
    assert_true(frames.alloc(64, 16) != 0)
    var slots = ArchetypeSlots(1, 16)
    assert_true(slots.spawn().valid())
    var col = SimdColumn[DType.float32](16)
    _ = col
    _ = chg.bytes


def test_f5_core_memory_raii() raises:
    var baseline_frame = mem_live(MEM_FRAME)
    _f5_raii_scope_run(baseline_frame)
    assert_equal(mem_live(MEM_FRAME), baseline_frame)


# ===========================================================================
# F6: Client-Side Prediction & Reconciliation (`flint.net.prediction`)
# ===========================================================================

struct M1LinearStepper(PredictionStepper):
    def __init__(out self):
        pass

    def step_prediction(
        mut self,
        mut state: PredictedState,
        input_ptr: Int,
        input_seq: Int,
        tick: Int,
        dt: Float32,
    ):
        _ = input_seq
        _ = tick
        ref cmd = Pointer[PhysicsRollbackInput, ImmutAnyOrigin](
            unsafe_from_address=input_ptr
        )[unsafe_offset=0]
        state.vel = Vec3(cmd.impulse_x, cmd.impulse_y, cmd.impulse_z)
        state.pos = state.pos + state.vel * dt


def test_f6_client_prediction_reconciliation() raises:
    var pred = ClientPredictionBuffer[PhysicsRollbackInput](capacity=64)
    for seq in range(1, 6):
        var cmd = PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0)
        pred.push_predicted(seq, cmd, PredictedState(Vec3(Float32(seq), 0.0, 0.0), Vec3(1.0, 0.0, 0.0)))

    # Mismatch at seq=2 -> rewind to authoritative pos=(2.5, 0, 0) and replay seq 3..5
    var reconciled = pred.reconcile(
        acked_seq=2,
        auth_state=PredictedState(Vec3(2.5, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
    )
    assert_equal(pred.last_acked_input_seq, 2)
    assert_equal(pred.mispredictions, 1)
    assert_equal(pred.replayed_inputs, 3)
    assert_true(abs(reconciled.pos.x - Float32(5.5)) < Float32(1e-4))

    # Stale ACK (seq <= last_acked_input_seq) must not regress
    var stale_out = pred.reconcile(
        acked_seq=1,
        auth_state=PredictedState(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)),
    )
    assert_equal(pred.last_acked_input_seq, 2)
    assert_true(abs(stale_out.pos.x - Float32(5.5)) < Float32(1e-4))

    # Negative: NaN authoritative state must not corrupt latest_state
    var nan_state = PredictedState(Vec3(quiet_nan(), 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    var after_nan = pred.reconcile(acked_seq=3, auth_state=nan_state)
    assert_true(abs(after_nan.pos.x - Float32(5.5)) < Float32(1e-4))

    # Stepper replay + VisualSmoother error absorption
    var pred2 = ClientPredictionBuffer[PhysicsRollbackInput](capacity=16)
    var stepper = M1LinearStepper()
    var smoother = VisualSmoother(decay_rate=18.0)
    for seq in range(1, 5):
        var cmd = PhysicsRollbackInput(2.0, 0.0, 0.0, 0.0, 0)
        pred2.push_predicted(
            seq,
            cmd,
            PredictedState(Vec3(Float32(seq) * 2.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)),
            tick=seq,
            dt=Float32(1.0),
        )
    var rec2 = pred2.reconcile_with(
        acked_seq=1,
        auth_state=PredictedState(Vec3(3.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)),
        stepper=stepper,
        default_dt=Float32(1.0),
    )
    assert_equal(pred2.mispredictions, 1)
    assert_true(abs(rec2.pos.x - Float32(9.0)) < Float32(1e-4))

    _ = pred2.reconcile_smoothed(
        acked_seq=2,
        auth_state=PredictedState(Vec3(6.5, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)),
        smoother=smoother,
    )
    assert_true(abs(smoother.pos_offset.x) > Float32(0.5))
    var off_before = abs(smoother.pos_offset.x)
    smoother.step(Float32(0.05))
    assert_true(abs(smoother.pos_offset.x) < off_before)


# ===========================================================================
# F7: Server Hitbox-Rewind Lag Compensation (`flint.net.lag_comp`)
# ===========================================================================

def test_f7_server_lag_compensation_rewind() raises:
    var lag = LagCompHistory(max_entities=16, max_history_ticks=16)
    for tick in range(10):
        var poses = List[HitboxPose]()
        poses.append(HitboxPose(entity_id=1, center=Vec3(Float32(tick) * 2.0, 1.0, 0.0), radius=Float32(0.75)))
        lag.record_tick(tick, poses)

    # At tick 9, target is at x=18.0; high-ping client viewed tick 3 (x=6.0)
    var hit_past = lag.raycast_at_tick(
        view_tick=3,
        origin=Vec3(6.0, 1.0, -5.0),
        direction=Vec3(0.0, 0.0, 1.0),
        max_distance=20.0,
    )
    assert_true(hit_past.hit and hit_past.entity_id == 1)
    assert_false(hit_past.was_clamped)

    # Current tick pose remains at x=18.0 (not stuck in rewound pose)
    var miss_now = lag.raycast_at_tick(
        view_tick=9,
        origin=Vec3(6.0, 1.0, -5.0),
        direction=Vec3(0.0, 0.0, 1.0),
        max_distance=20.0,
    )
    assert_false(miss_now.hit)

    # Out-of-window future tick safely clamped to newest tick (x=18.0)
    var future_hit = lag.raycast_at_tick(9999, Vec3(18.0, 1.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_true(future_hit.was_clamped)
    assert_true(future_hit.hit and future_hit.entity_id == 1)

    # Negative/too-old tick safely clamped to oldest tick (x=0.0)
    var ancient_hit = lag.raycast_at_tick(-100, Vec3(0.0, 1.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_true(ancient_hit.was_clamped)
    assert_true(ancient_hit.hit and ancient_hit.entity_id == 1)

    # Sub-tick linear interpolation & overlap check via HitboxSnapshotRing
    var ring = HitboxSnapshotRing(capacity=8, max_rewind_ms=100.0)
    var p0 = List[HitboxPose]()
    p0.append(HitboxPose.aabb(entity_id=7, center=Vec3(0.0, 1.0, 0.0), half_extents=Vec3(0.5, 0.5, 0.5)))
    ring.record_snapshot(0, 0.0, p0)
    var p1 = List[HitboxPose]()
    p1.append(HitboxPose.aabb(entity_id=7, center=Vec3(10.0, 1.0, 0.0), half_extents=Vec3(0.5, 0.5, 0.5)))
    ring.record_snapshot(1, 10.0, p1)

    var mid_hit = ring.rewind_and_raycast(5.0, Vec3(5.0, 1.0, -4.0), Vec3(0.0, 0.0, 1.0), 15.0)
    assert_false(mid_hit.was_clamped)
    assert_true(mid_hit.hit and mid_hit.entity_id == 7)
    var ov_hits = ring.rewind_and_overlap(5.0, Vec3(5.0, 1.0, 0.0), Float32(0.4))
    assert_equal(len(ov_hits), 1)
    assert_equal(ov_hits[0].entity_id, 7)


# ===========================================================================
# F8: Chunk-Relative Open-World Quantization (`bitpack` / `replicator`)
# ===========================================================================

def test_f8_chunk_relative_open_world_quantization() raises:
    var rep = SnapshotReplicator(
        max_entities=8,
        max_clients=1,
        max_packet_bytes=1000,
        interest_radius=200.0,
        chunk_size=256.0,
    )
    var cli = SnapshotClientReceiver(max_entities=8, chunk_size=256.0)

    var far_pos = Vec3(Float32(100_000.25), Float32(2_500.5), Float32(-100_000.75))
    rep.set_client_view(0, far_pos, 200.0)
    rep.set_entity(0, far_pos, Quat.identity(), Vec3(), UInt32(1))
    rep.prepare_tick()

    var pkt = rep.build_client_packet(0)
    var res = cli.apply_packet(pkt[1])
    assert_true(res[0])
    assert_true((cli.entities[0].pos - far_pos).length() < Float32(0.02))
    rep.notify_client_ack(0, res[1])

    # Chunk-boundary crossing update
    var far_pos2 = Vec3(Float32(100_260.5), Float32(2_500.5), Float32(-99_990.25))
    rep.set_client_view(0, far_pos2, 200.0)
    rep.set_entity(0, far_pos2, Quat.identity(), Vec3(), UInt32(2))
    rep.prepare_tick()
    var pkt2 = rep.build_client_packet(0)
    var res2 = cli.apply_packet(pkt2[1])
    assert_true(res2[0])
    assert_true((cli.entities[0].pos - far_pos2).length() < Float32(0.02))
    rep.notify_client_ack(0, res2[1])

    # Extreme coordinate beyond 16-bit chunk index (> 32767 * 256) -> 32-bit chunk path
    var huge_pos = Vec3(Float32(9_000_000.0), Float32(-500.5), Float32(-9_000_000.0))
    rep.set_client_view(0, huge_pos, 200.0)
    rep.set_entity(0, huge_pos, Quat.identity(), Vec3(), UInt32(3))
    rep.prepare_tick()
    var pkt3 = rep.build_client_packet(0)
    var res3 = cli.apply_packet(pkt3[1])
    assert_true(res3[0])
    assert_true((cli.entities[0].pos - huge_pos).length() < Float32(2.0))


# ===========================================================================
# F9: NAT Traversal & Relay Rollback Topology (`flint.net.relay`)
# ===========================================================================

def test_f9_nat_punch_and_relay_rollback() raises:
    var sim = VirtualNetSim.from_rtt(rtt_ms=20.0, jitter_ms=2.0, loss_rate=0.02, seed=0xF91234)
    var relay_ep = sim.create_ipv4_peer(0, 9900)
    var ep0 = sim.create_ipv4_peer(1, 9901)
    var ep1 = sim.create_ipv4_peer(2, 9902)

    var relay = RelayServer(relay_ep.local_endpoint())
    var hdr = RelayForwardHeader(session_token=UInt64(0xCAFE1234), src_peer=0, dst_peer=1)
    assert_equal(hdr.src_peer, 0)
    # Invalid magic rejected cleanly by RelayForwardHeader.parse_from_ptr
    var bad_hdr_bytes = List[UInt8](length=18, fill=UInt8(0))
    assert_false(RelayForwardHeader.parse_from_ptr(Int(bad_hdr_bytes.unsafe_ptr()), len(bad_hdr_bytes))[0])
    _ = len(bad_hdr_bytes)

    # Direct NATPunchProbe handshake: PROBING -> DIRECT, plus timeout -> RELAY_FALLBACK
    var p_sim = VirtualNetSim.from_rtt(rtt_ms=10.0, jitter_ms=0.0, loss_rate=0.0, seed=0x9090)
    var p_ep0 = p_sim.create_ipv4_peer(0, 9910)
    var p_ep1 = p_sim.create_ipv4_peer(1, 9911)
    var probe0 = NATPunchProbe(
        local_peer=0,
        relay_endpoint=relay_ep.local_endpoint(),
        target_peer=1,
        target_endpoint=p_ep1.local_endpoint(),
        session_token=UInt64(0xCAFE1234),
        probe_timeout_ms=100.0,
    )
    var probe1 = NATPunchProbe(
        local_peer=1,
        relay_endpoint=relay_ep.local_endpoint(),
        target_peer=0,
        target_endpoint=p_ep0.local_endpoint(),
        session_token=UInt64(0xCAFE1234),
        probe_timeout_ms=100.0,
    )
    var rx_views = List[DatagramView](capacity=16)
    for _ in range(10):
        var now = p_sim.now_ms()
        probe0.poll(p_ep0, now)
        probe1.poll(p_ep1, now)
        var n0 = p_ep0.recv_batch(rx_views)
        for i in range(n0):
            _ = probe0.ingest_datagram(rx_views[i], p_ep0, now)
        var n1 = p_ep1.recv_batch(rx_views)
        for i in range(n1):
            _ = probe1.ingest_datagram(rx_views[i], p_ep1, now)
        if probe0.is_direct() and probe1.is_direct():
            break
        p_sim.advance_ms(10.0)
    assert_equal(probe0.state, PUNCH_STATE_DIRECT)
    assert_equal(probe1.state, PUNCH_STATE_DIRECT)

    # Unreachable peer falls back to RELAY_FALLBACK after probe_timeout_ms
    var p_ep_unreach = p_sim.create_ipv4_peer(2, 19999)
    var probe_unreach = NATPunchProbe(
        local_peer=0,
        relay_endpoint=relay_ep.local_endpoint(),
        target_peer=2,
        target_endpoint=p_ep_unreach.local_endpoint(),
        session_token=UInt64(0xCAFE5678),
        probe_timeout_ms=50.0,
    )
    assert_true(probe_unreach.send_probe(p_ep0, p_sim.now_ms()))
    p_sim.advance_ms(60.0)
    probe_unreach.poll(p_ep0, p_sim.now_ms())
    assert_equal(probe_unreach.state, PUNCH_STATE_RELAY_FALLBACK)
    _ = p_ep_unreach.local_endpoint()

    # Forced relay RollbackSession topology: all traffic traverses RelayServer
    var addrs = List[NetEndpoint]()
    addrs.append(ep0.local_endpoint())
    addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s0 = RollbackSession[PhysicsRollbackInput](0, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    var s1 = RollbackSession[PhysicsRollbackInput](1, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    s0.enable_relay(relay_ep.local_endpoint(), attempt_nat_punch=False)
    s1.enable_relay(relay_ep.local_endpoint(), attempt_nat_punch=False)
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    for f in range(80):
        var now = sim.now_ms()
        relay.poll(relay_ep, now)
        var cmd = PhysicsRollbackInput(Float32(f % 3), 0.0, 1.0, 0.0, 0)
        _ = s0.step_frame(cmd, pw0, ep0, now)
        _ = s1.step_frame(cmd, pw1, ep1, now)
        sim.advance_ms(1000.0 / 60.0)

    assert_false(s0.desync_detected)
    assert_false(s1.desync_detected)
    assert_true(s0.is_peer_relayed(1))
    assert_true(s1.is_peer_relayed(0))
    assert_true(relay.packets_forwarded > 0)
    assert_true(s0.relayed_packets_sent > 0 and s0.relayed_packets_received > 0)
    assert_true(s1.relayed_packets_sent > 0 and s1.relayed_packets_received > 0)
    assert_true(len(s0.confirmed_hash_history) >= 40)
    for t in range(40):
        assert_equal(s0.confirmed_hash_history[t], s1.confirmed_hash_history[t])
    assert_true(sim.now_ms() >= 0.0)


def main() raises:
    set_sim_float_mode()
    test_f1_vault_free_snapshot_blob()
    print("[PASS] test_f1_vault_free_snapshot_blob")
    test_f2_extensible_component_bitset_over_256()
    print("[PASS] test_f2_extensible_component_bitset_over_256")
    test_f3_jobpool_raii_dynamic_dependents_and_priorities()
    print("[PASS] test_f3_jobpool_raii_dynamic_dependents_and_priorities")
    test_f4_detmath_simd8_bit_exact()
    print("[PASS] test_f4_detmath_simd8_bit_exact")
    test_f5_core_memory_raii()
    print("[PASS] test_f5_core_memory_raii")
    test_f6_client_prediction_reconciliation()
    print("[PASS] test_f6_client_prediction_reconciliation")
    test_f7_server_lag_compensation_rewind()
    print("[PASS] test_f7_server_lag_compensation_rewind")
    test_f8_chunk_relative_open_world_quantization()
    print("[PASS] test_f8_chunk_relative_open_world_quantization")
    test_f9_nat_punch_and_relay_rollback()
    print("[PASS] test_f9_nat_punch_and_relay_rollback")
    print("[SUMMARY] 9/9 passed")
