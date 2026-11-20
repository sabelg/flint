"""Tier 2 (Boundary, Corner & Negative/Error-Rejection) Opaque-Box E2E Suite for FA2 & FA7.

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`).
"""
from std.atomic import Atomic
from std.collections import List
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.app import (
    NUM_SCHEDULE_STAGES,
    STAGE_UPDATE,
    FixedTime,
    Resources,
    Schedule,
)
from flint.ecs import (
    NULL_ENTITY,
    STORAGE_TABLE,
    comp_hash64,
    Component,
    Entity,
    ParallelCommands,
    World,
)
from flint.scene import (
    MAX_PREFAB_NEST_DEPTH,
    LocalTransform,
    HierarchyDepth,
    set_parent,
    clear_parent,
    ComponentRegistry,
    NestedPrefabRef,
    Prefab,
    Scene,
)
from flint.core.detmath import (
    MXCSR_SIM,
    PI,
    det_acos,
    det_asin,
    det_atan2,
    det_cos,
    det_exp,
    det_log,
    det_pow,
    det_sin,
    f32_bits,
    f32_from_bits,
    infinity,
    is_finite32,
    is_nan,
    quiet_nan,
    scale_by_pow2,
    set_sim_float_mode,
    sim_float_mode_ok,
    write_mxcsr,
)
from flint.core.jobs import JobPool, NO_JOB, run_jobs
from flint.core.lz4 import hash_list
from flint.core.math import Quat, Vec3
from flint.core.memory import Arena, MEM_FRAME
from flint.core.serial import BlobReader, BlobWriter

from flint.net import (
    AeadKey,
    BitReader,
    BitWriter,
    DatagramView,
    IPV6_SAFE_MTU_BYTES,
    NetChannel,
    NetEndpoint,
    NetHub,
    ReplayWindow64,
    RollbackSession,
    SnapshotClientReceiver,
    SnapshotReplicator,
    StatelessCookieAuth,
    UDP_MAX_DATAGRAM_BYTES,
    UdpSocket,
    VirtualEndpoint,
    VirtualNetSim,
    aead_open,
    aead_seal,
)
from flint.net.rollback import (
    DEFAULT_ROLLBACK_SESSION_TOKEN,
    PhysicsRollbackAdapter,
    PhysicsRollbackInput,
)


# ===========================================================================
# Shared Test Types & Helpers (FA2)
# ===========================================================================

@fieldwise_init
struct T2PodA(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t2.T2PodA")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var val: Float32


@fieldwise_init
struct T2PodB(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t2.T2PodB")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var code: Int32


def t2_noop_sys(
    _w: Pointer[World, MutAnyOrigin],
    _r: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    pass


def t2_inc_chunk(ctx: Int, _chunk: Int):
    _ = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=ctx)[unsafe_offset=0].fetch_add(1)


# ===========================================================================
# FA2 Tier 2 Tests (8 Boundary / Corner / Error-Rejection Tests)
# ===========================================================================

def test_fa2_t2_fixed_time_spiral_of_death_and_invalid_schedule() raises:
    var ft = FixedTime(hz=60.0, max_steps_per_frame=4)
    assert_equal(ft.advance_frame(Float32(-0.5)), 0)
    assert_false(ft.clamped_last_frame)
    assert_equal(ft.advance_frame(Float32(0.0)), 0)
    assert_false(ft.clamped_last_frame)

    var steps = ft.advance_frame(Float32(5.0))  # 5 seconds -> spiral-of-death clamped to 4
    assert_equal(steps, 4)
    assert_true(ft.clamped_last_frame)
    assert_true(ft.overstep_alpha >= Float32(0.0) and ft.overstep_alpha < Float32(1.0))

    var ft_zero = FixedTime(hz=0.0, max_steps_per_frame=0)
    assert_equal(ft_zero.max_steps_per_frame, 1)
    assert_true(ft_zero.dt <= Float32(1.0) and ft_zero.dt > Float32(0.0))

    var sched = Schedule()
    assert_equal(sched.add_system(-1, "bad_neg", t2_noop_sys), -1)
    assert_equal(sched.add_system(NUM_SCHEDULE_STAGES, "bad_oob", t2_noop_sys), -1)
    assert_true(sched.add_system(STAGE_UPDATE, "valid", t2_noop_sys) >= 0)


def test_fa2_t2_entity_null_and_stale_generation_guards() raises:
    var w = World()
    assert_false(NULL_ENTITY.valid())
    assert_true(NULL_ENTITY.is_null())
    assert_false(w.is_alive(NULL_ENTITY))
    assert_false(w.despawn(NULL_ENTITY))
    assert_false(w.insert[T2PodA](NULL_ENTITY, T2PodA(1.0)))

    var e1 = w.spawn_empty()
    assert_true(w.insert[T2PodA](e1, T2PodA(12.5)))
    assert_true(w.despawn(e1))

    var e2 = w.spawn_empty()  # Reuses slot 0 with incremented generation
    assert_true(w.is_alive(e2))
    assert_true(e1 != e2)

    assert_false(w.is_alive(e1))
    assert_false(w.despawn(e1))
    assert_false(w.has[T2PodA](e1))
    assert_false(w.insert[T2PodA](e1, T2PodA(99.0)))
    assert_false(w.remove[T2PodA](e1))
    assert_false(Bool(w.get_ptr[T2PodA](e1)))


def test_fa2_t2_scene_cycle_guard_and_clear_parent() raises:
    var w = World()
    var a = w.spawn_empty()
    var b = w.spawn_empty()
    var c = w.spawn_empty()
    var dead = w.spawn_empty()
    assert_true(w.despawn(dead))

    assert_false(set_parent(w, a, a))
    assert_false(set_parent(w, dead, a))
    assert_false(set_parent(w, a, dead))

    assert_true(set_parent(w, b, a))
    assert_true(set_parent(w, c, b))
    assert_false(set_parent(w, a, c))  # 3-node cycle rejected

    assert_true(clear_parent(w, c))
    assert_equal(Int(w.get[HierarchyDepth](c).depth), 0)
    assert_false(clear_parent(w, c))  # idempotent false when already detached


def test_fa2_t2_prefab_max_nest_depth_and_corrupted_scene_rejections() raises:
    var reg = ComponentRegistry.with_builtins()

    # Build chain exceeding MAX_PREFAB_NEST_DEPTH (16): P0 -> P1 -> ... -> P18
    var lib = List[Prefab]()
    for d in range(MAX_PREFAB_NEST_DEPTH + 3):
        var p = Prefab("Chain_" + String(d))
        var root = p.add_node("Root", -1)
        p.set_component[LocalTransform](root, LocalTransform.identity())
        if d > 0:
            p.add_nested(NestedPrefabRef("Chain_" + String(d - 1), attach_parent_slot=root))
        lib.append(p^)

    var w = World()
    var caught_depth = False
    try:
        _ = lib[len(lib) - 1].instantiate(w, reg, lib)
    except:
        caught_depth = True
    assert_true(caught_depth)

    # Corrupted .flscn JSON rejections
    var s = Scene("ErrScene")
    var bad_json_inputs = List[String]()
    bad_json_inputs.append("{not valid json")
    bad_json_inputs.append("{\"format\": \"wrong\", \"version\": 1, \"entities\": []}")
    bad_json_inputs.append("{\"format\": \"flscn\", \"version\": 999, \"entities\": []}")
    for i in range(len(bad_json_inputs)):
        var text = bad_json_inputs[i]
        var bytes = List[UInt8](capacity=text.byte_length())
        for b in text.as_bytes():
            bytes.append(b)
        var caught_json = False
        try:
            _ = s.load_flscn_bytes(w, reg, bytes)
        except:
            caught_json = True
        assert_true(caught_json)

    # Corrupted .flscnb binary rejection
    var bad_bin = List[UInt8](length=12, fill=UInt8(0xAA))
    var caught_bin = False
    try:
        _ = s.load_flscnb_bytes(w, reg, bad_bin)
    except:
        caught_bin = True
    assert_true(caught_bin)

    # exit_pie before enter_pie raises
    var fresh_scene = Scene("NeverEntered")
    var caught_pie = False
    try:
        fresh_scene.exit_pie(w)
    except:
        caught_pie = True
    assert_true(caught_pie)


def test_fa2_t2_restore_blob_error_and_non_mutation() raises:
    var src = World()
    _ = src.register_component[T2PodA]()
    _ = src.register_component[T2PodB]()
    var e = src.spawn_empty()
    _ = src.insert[T2PodA](e, T2PodA(7.5))
    _ = src.insert[T2PodB](e, T2PodB(42))

    var writer = BlobWriter()
    src.snapshot_blob(writer)

    # Destination missing T2PodB must fail AND leave sentinel untouched
    var dst = World()
    _ = dst.register_component[T2PodA]()
    var sentinel = dst.spawn_empty()
    _ = dst.insert[T2PodA](sentinel, T2PodA(123.0))

    var rd_missing = BlobReader(writer.data.copy())
    var caught_missing = False
    try:
        dst.restore_blob(rd_missing)
    except:
        caught_missing = True
    assert_true(caught_missing)
    assert_equal(dst.entity_count(), 1)
    assert_true(dst.is_alive(sentinel))
    assert_equal(dst.get[T2PodA](sentinel).val, Float32(123.0))

    # Truncated blob bytes must raise cleanly
    var trunc = List[UInt8]()
    for i in range(16):
        trunc.append(writer.data[i])
    var rd_trunc = BlobReader(trunc^)
    var caught_trunc = False
    try:
        dst.restore_blob(rd_trunc)
    except:
        caught_trunc = True
    assert_true(caught_trunc)
    assert_equal(dst.entity_count(), 1)


def test_fa2_t2_detmath_extremes_nan_domain_and_mxcsr_heal() raises:
    var qnan = quiet_nan()
    var inf = infinity()

    assert_true(is_nan(det_sin(qnan)))
    assert_true(is_nan(det_cos(inf)))
    assert_equal(det_sin(Float32(1.0e9)), Float32(0.0))
    assert_equal(det_sin(Float32(-1.0e9)), Float32(0.0))
    assert_equal(det_cos(Float32(1.0e9)), Float32(1.0))

    # Signed-zero atan2 quadrants
    assert_true(abs(det_atan2(Float32(0.0), Float32(-1.0)) - PI) < Float32(1e-5))
    assert_true(abs(det_atan2(Float32(-0.0), Float32(-1.0)) + PI) < Float32(1e-5))
    assert_equal(f32_bits(det_atan2(Float32(0.0), Float32(0.0))), UInt32(0))
    assert_equal(f32_bits(det_atan2(Float32(-0.0), Float32(0.0))), UInt32(0x80000000))

    # Out-of-domain asin / acos
    assert_true(is_nan(det_asin(Float32(1.001))))
    assert_true(is_nan(det_asin(Float32(-1.001))))
    assert_true(is_nan(det_acos(Float32(1.001))))
    assert_true(is_nan(det_acos(Float32(-1.001))))

    # log / exp / pow extremes + subnormals
    assert_equal(det_log(Float32(0.0)), -inf)
    assert_true(is_nan(det_log(Float32(-1.0))))
    var sub_log = det_log(f32_from_bits(1))
    assert_true(is_finite32(sub_log) and sub_log < Float32(-100.0))

    assert_equal(det_exp(Float32(89.0)), inf)
    assert_equal(det_exp(Float32(-105.0)), Float32(0.0))

    assert_equal(det_pow(Float32(0.0), Float32(-1.0)), inf)
    assert_true(is_nan(det_pow(Float32(-2.0), Float32(0.5))))
    assert_true(abs(det_pow(Float32(-2.0), Float32(3.0)) - Float32(-8.0)) < Float32(1e-4))
    assert_equal(scale_by_pow2(Float32(1.0), 200), inf)
    assert_equal(scale_by_pow2(Float32(1.0), -200), Float32(0.0))

    # MXCSR / FPCR corruption and heal via JobPool.wait
    write_mxcsr(MXCSR_SIM | UInt32(1 << 15) | UInt32(1 << 6))
    assert_false(sim_float_mode_ok())
    var pool = JobPool()
    _ = pool.allocate()
    pool.wait(NO_JOB)
    assert_true(sim_float_mode_ok())
    pool.stop()
    set_sim_float_mode()


def test_fa2_t2_jobpool_edge_cases_and_double_stop() raises:
    var pool = JobPool()
    assert_true(pool.start(0))
    assert_false(pool.ready())
    assert_true(pool.allocated())

    var cnt = List[Int64](length=1, fill=0)
    var relay = pool.create(t2_inc_chunk, Int(cnt.unsafe_ptr()), chunks=0)
    assert_true(relay != NO_JOB)
    pool.depend(relay, NO_JOB)
    pool.submit(relay)
    pool.wait(relay)
    assert_equal(Int(cnt[0]), 0)

    pool.wait(NO_JOB)
    assert_equal(pool.chain(t2_inc_chunk, Int(cnt.unsafe_ptr()), 0), NO_JOB)

    run_jobs(0, t2_inc_chunk, Int(cnt.unsafe_ptr()), 5)
    assert_equal(Int(cnt[0]), 5)
    _ = len(cnt)

    pool.stop()
    pool.stop()  # double stop must be idempotent


def test_fa2_t2_memory_arena_overflow_negative_and_zero_capacity() raises:
    var a = Arena(MEM_FRAME, 128)
    assert_equal(a.alloc(512), 0)
    assert_equal(a.overflows, 1)
    assert_equal(a.alloc(-1), 0)
    assert_equal(a.overflows, 2)

    var z = Arena(MEM_FRAME, 0)
    assert_equal(z.capacity, 0)
    assert_equal(z.alloc(16), 0)
    assert_equal(z.overflows, 1)


# ===========================================================================
# FA7 Tier 2 Tests (8 Boundary / Corner / Negative Tests)
# ===========================================================================

def t2_make_bytes(tag: Int, n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(UInt8((tag * 29 + i * 17) & 0xFF))
    return out^


def test_fa7_t2_udp_invalid_and_oversized_sends() raises:
    var closed = UdpSocket()
    var dst = NetEndpoint.ipv4_loopback(9999)
    var dummy = t2_make_bytes(1, 8)
    assert_false(closed.is_open())
    assert_false(closed.send_to(dst, dummy))
    assert_equal(closed.flush_sends(), 0)
    var rx = List[DatagramView]()
    assert_equal(closed.recv_batch(rx), 0)

    var sock = UdpSocket.open_dual_stack(0)
    assert_true(sock.is_open())
    var target = NetEndpoint.ipv4_loopback(sock.local_port())

    var empty = List[UInt8]()
    assert_false(sock.send_to(target, empty))
    assert_false(sock.send_raw(target, 0, 10))
    assert_false(sock.send_raw(target, Int(dummy.unsafe_ptr()), 0))
    assert_false(sock.send_raw(target, Int(dummy.unsafe_ptr()), -4))
    _ = len(dummy)

    var oversized = List[UInt8](length=UDP_MAX_DATAGRAM_BYTES + 1, fill=UInt8(0xAA))
    assert_false(sock.send_to(target, oversized))


def test_fa7_t2_bitreader_overflow_null_and_nan_bounds() raises:
    var nr = BitReader.from_raw_ptr(0, 16)
    assert_true(nr.overflowed)
    assert_equal(Int(nr.read_u32()), 0)

    var short3 = List[UInt8](length=3, fill=UInt8(0x55))
    var tr1 = BitReader(short3)
    _ = tr1.read_u32()
    assert_true(tr1.overflowed)

    var tr2 = BitReader(short3)
    var qd = tr2.read_quat_smallest_three()
    assert_true(tr2.overflowed)
    assert_equal(qd.w, Float32(1.0))

    var br_huge = BitReader(short3)
    var huge_bytes = br_huge.read_bytes_aligned(1 << 60)
    assert_true(br_huge.overflowed)
    assert_equal(len(huge_bytes), 0)

    var w = BitWriter(64)
    var zero = Float32(0.0)
    var nan_val = zero / zero
    w.write_f32_quantized(nan_val, -10.0, 10.0, 12)
    w.write_f32_quantized(Float32(1000.0), -10.0, 10.0, 12)
    w.write_f32_quantized(Float32(-1000.0), -10.0, 10.0, 12)
    var mark = w.mark()
    w.write_u32(UInt32(0xDEADBEEF))
    assert_true(w.bit_length() > mark.total_bits)
    w.restore(mark)
    assert_equal(w.bit_length(), mark.total_bits)

    var rr = BitReader(w.finish())
    var f0 = rr.read_f32_quantized(-10.0, 10.0, 12)
    var f1 = rr.read_f32_quantized(-10.0, 10.0, 12)
    var f2 = rr.read_f32_quantized(-10.0, 10.0, 12)
    assert_false(rr.overflowed)
    assert_true(f0 == f0 and f0 >= Float32(-10.0) and f0 <= Float32(10.0))
    assert_true(f1 >= Float32(9.9) and f1 <= Float32(10.0))
    assert_true(f2 <= Float32(-9.9) and f2 >= Float32(-10.0))


def test_fa7_t2_aead_tamper_wrong_key_and_replay_bounds() raises:
    var key = AeadKey.from_u64x4(0x01, 0x02, 0x03, 0x04)
    var wrong_key = AeadKey.from_u64x4(0x01, 0x02, 0x03, 0x05)
    var aad = t2_make_bytes(3, 12)
    var pt = t2_make_bytes(7, 64)
    var ct_tag = List[UInt8]()
    aead_seal(key, UInt64(10), aad, pt, ct_tag)

    var out = List[UInt8]()
    var bad_ct = ct_tag.copy()
    bad_ct[5] ^= UInt8(0x01)
    assert_false(aead_open(key, UInt64(10), aad, bad_ct, out))

    var bad_tag = ct_tag.copy()
    bad_tag[len(bad_tag) - 1] ^= UInt8(0x80)
    assert_false(aead_open(key, UInt64(10), aad, bad_tag, out))

    var bad_aad = aad.copy()
    bad_aad[0] ^= UInt8(0x02)
    assert_false(aead_open(key, UInt64(10), bad_aad, ct_tag, out))

    assert_false(aead_open(key, UInt64(11), aad, ct_tag, out))
    assert_false(aead_open(wrong_key, UInt64(10), aad, ct_tag, out))
    var short_ct = List[UInt8](length=12, fill=UInt8(0))
    assert_false(aead_open(key, UInt64(10), aad, short_ct, out))

    var rw = ReplayWindow64()
    assert_false(rw.would_accept(0))
    rw.record(100)
    assert_false(rw.would_accept(100))
    assert_true(rw.would_accept(36))   # age = 64 -> accepted
    assert_false(rw.would_accept(35))  # age = 65 -> rejected

    var auth = StatelessCookieAuth(key, bucket_ms=10000.0)
    var ep1 = NetEndpoint.ipv4_loopback(8001)
    var ep2 = NetEndpoint.ipv4_loopback(8002)
    var cookie = auth.issue_cookie(ep1, 20000.0)
    assert_false(auth.verify_cookie(ep2, Int(Pointer(to=cookie)), 20000.0))
    assert_true(auth.verify_cookie(ep1, Int(Pointer(to=cookie)), 30000.0))   # bucket - 1
    assert_false(auth.verify_cookie(ep1, Int(Pointer(to=cookie)), 40000.0))  # bucket - 2
    _ = cookie[0]


def test_fa7_t2_net_channel_tamper_replay_corruption_and_rto_backoff() raises:
    var sim = VirtualNetSim(seed=0xF72004, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=0.0)
    var ep0 = sim.create_ipv4_peer(0, 9500)
    var ep1 = sim.create_ipv4_peer(1, 9500)
    var key = AeadKey.from_u64x4(0x123, 0x456, 0x789, 0xABC)

    var ch0 = NetChannel(ep1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var ch1 = NetChannel(ep0.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    ch0.enable_encryption(key)
    ch1.enable_encryption(key)

    var raw_msg = t2_make_bytes(9, 32)
    assert_true(ch0.send_unreliable_raw(raw_msg, ep0, sim.now_ms())[0])
    sim.advance_ms(20.0)

    var views = List[DatagramView](capacity=8)
    assert_equal(ep1.recv_batch(views), 1)
    var captured = views[0].to_bytes()
    assert_true(ep1.local_endpoint().port > 0)

    # 1-byte tamper must fail auth without polluting ReplayWindow64
    var tampered = captured.copy()
    tampered[len(tampered) - 1] ^= UInt8(0x10)
    assert_false(ch1.ingest_datagram(DatagramView(ep0.local_endpoint(), Int(tampered.unsafe_ptr()), len(tampered)), sim.now_ms()))
    assert_equal(len(tampered), len(captured))
    assert_equal(ch1.crypto_auth_failures, 1)

    # Original untampered packet succeeds
    assert_true(ch1.ingest_datagram(DatagramView(ep0.local_endpoint(), Int(captured.unsafe_ptr()), len(captured)), sim.now_ms()))
    var pop = ch1.pop_unreliable_raw_message()
    assert_true(pop[0] and hash_list(pop[1]) == hash_list(raw_msg))

    # Exact replay rejected
    assert_false(ch1.ingest_datagram(DatagramView(ep0.local_endpoint(), Int(captured.unsafe_ptr()), len(captured)), sim.now_ms()))
    assert_equal(len(captured), len(tampered))
    assert_equal(ch1.crypto_replay_drops, 1)

    # Garbage 0x00 magic packet rejected at header parse (no auth counter bump)
    var garbage = List[UInt8](length=20, fill=UInt8(0))
    assert_false(ch1.ingest_datagram(DatagramView(ep0.local_endpoint(), Int(garbage.unsafe_ptr()), len(garbage)), sim.now_ms()))
    assert_equal(len(garbage), 20)
    assert_equal(ch1.crypto_auth_failures, 1)

    # Valid-header unencrypted packet rejected by encrypted NetChannel -> increments crypto_auth_failures
    var ch_plain = NetChannel(ep1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    assert_true(ch_plain.send_unreliable_raw(raw_msg, ep0, sim.now_ms())[0])
    sim.advance_ms(20.0)
    assert_equal(ep1.recv_batch(views), 1)
    var plain_bytes = views[0].to_bytes()
    assert_true(ep1.local_endpoint().port > 0)
    assert_false(ch1.ingest_datagram(DatagramView(ep0.local_endpoint(), Int(plain_bytes.unsafe_ptr()), len(plain_bytes)), sim.now_ms()))
    assert_true(len(plain_bytes) > 0)
    assert_equal(ch1.crypto_auth_failures, 2)

    # 100% loss RTO exponential backoff & AIMD cwnd halving
    var sim_loss = VirtualNetSim(seed=0x1055, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=1.0)
    var l0 = sim_loss.create_ipv4_peer(0, 9510)
    var l1 = sim_loss.create_ipv4_peer(1, 9510)
    var chl = NetChannel(l1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    chl.rto_ms = 100.0
    var init_cwnd = chl.cwnd
    assert_true(chl.send_reliable(t2_make_bytes(1, 64)))
    chl.poll(l0, 0.0)
    assert_equal(chl.retransmissions, 0)
    chl.poll(l0, 100.0)
    assert_equal(chl.retransmissions, 1)
    assert_true(chl.cwnd < init_cwnd)
    chl.poll(l0, 200.0)
    assert_equal(chl.retransmissions, 1)
    chl.poll(l0, 305.0)
    assert_equal(chl.retransmissions, 2)
    assert_true(sim_loss.now_ms() >= 0.0)
    assert_true(sim.now_ms() >= 0.0)


def test_fa7_t2_net_hub_max_channels_strangers_and_idle_eviction() raises:
    var sim = VirtualNetSim(seed=0x7240B, one_way_latency_ms=5.0, jitter_ms=0.0, loss_rate=0.0)
    var host = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9600), True, max_channels=1, idle_timeout_ms=150.0)
    var c0 = NetHub[VirtualEndpoint](sim.create_ipv4_peer(1, 9600), False)
    var c1 = NetHub[VirtualEndpoint](sim.create_ipv4_peer(2, 9600), False)

    assert_equal(c0.connect(host.local_endpoint()), 0)
    assert_equal(c1.connect(host.local_endpoint()), 0)

    assert_true(c0.send_unreliable(0, t2_make_bytes(1, 16), sim.now_ms()))
    for _ in range(5):
        host.poll(sim.now_ms())
        sim.advance_ms(10.0)
    assert_equal(host.open_count(), 1)

    # c1 rejected while host full (max_channels=1)
    assert_true(c1.send_unreliable(0, t2_make_bytes(2, 16), sim.now_ms()))
    for _ in range(5):
        host.poll(sim.now_ms())
        sim.advance_ms(10.0)
    assert_equal(host.open_count(), 1)
    assert_equal(host.connect(c1.local_endpoint()), -1)
    assert_true(host.datagrams_ignored > 0)

    # Idle eviction of c0 after > 150ms -> c1 accepted into freed slot
    sim.advance_ms(200.0)
    host.poll(sim.now_ms())
    assert_equal(len(host.evicted), 1)
    assert_equal(host.open_count(), 0)

    assert_true(c1.send_unreliable(0, t2_make_bytes(3, 16), sim.now_ms()))
    for _ in range(5):
        host.poll(sim.now_ms())
        sim.advance_ms(10.0)
    assert_equal(host.open_count(), 1)

    # Client (accept=False) ignores unsolicited packet
    assert_true(c1.send_unreliable(c1.connect(c0.local_endpoint()), t2_make_bytes(4, 12), sim.now_ms()))
    for _ in range(5):
        c0.poll(sim.now_ms())
        sim.advance_ms(10.0)
    assert_equal(c0.open_count(), 1)
    assert_true(c0.datagrams_ignored > 0)

    # Invalid cid guards
    assert_false(host.send_reliable(-1, t2_make_bytes(1, 8)))
    assert_false(host.send_reliable(999, t2_make_bytes(1, 8)))
    assert_false(host.send_unreliable(-1, t2_make_bytes(1, 8), sim.now_ms()))
    assert_false(host.pop_reliable(-1)[0])
    assert_false(host.pop_reliable(999)[0])
    assert_true(sim.now_ms() >= 0.0)


def test_fa7_t2_replicator_seq_wraparound_truncation_and_budget_overflow() raises:
    var rep = SnapshotReplicator(
        max_entities=32,
        max_clients=1,
        max_packet_bytes=1000,
        interest_radius=200.0,
        world_bounds=256.0,
    )
    var cli = SnapshotClientReceiver(32, 256.0)
    rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 200.0)
    rep.clients[0].next_snap_seq = UInt16(0xFFFE)

    rep.set_entity(0, Vec3(10.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(1))
    rep.set_entity(1, Vec3(20.0, 2.0, 3.0), Quat.identity(), Vec3(), UInt32(7))
    rep.prepare_tick()
    var p_fffe = rep.build_client_packet(0)
    assert_equal(Int(p_fffe[0]), 0xFFFE)
    var r_fffe = cli.apply_packet(p_fffe[1])
    assert_true(r_fffe[0])
    rep.notify_client_ack(0, r_fffe[1])

    rep.set_entity(0, Vec3(11.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(1))
    rep.prepare_tick()
    var p_ffff = rep.build_client_packet(0)
    assert_equal(Int(p_ffff[0]), 0xFFFF)
    var r_ffff = cli.apply_packet(p_ffff[1])
    assert_true(r_ffff[0])
    rep.notify_client_ack(0, r_ffff[1])

    # Wraparound 0xFFFF -> 0x0000 preserves stationary entity 1
    rep.set_entity(0, Vec3(12.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(2))
    rep.prepare_tick()
    var p_0000 = rep.build_client_packet(0)
    assert_equal(Int(p_0000[0]), 0x0000)
    var r_0000 = cli.apply_packet(p_0000[1])
    assert_true(r_0000[0])
    assert_true(cli.entities[1].active)
    assert_equal(Int(cli.entities[1].state_flags), 7)
    rep.notify_client_ack(0, r_0000[1])

    # Stale replay 0xFFFF after 0x0000 rejected
    assert_false(cli.apply_packet(p_ffff[1])[0])

    # Truncated packet 0x0001 fails without corrupting state
    rep.set_entity(0, Vec3(15.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(3))
    rep.prepare_tick()
    var p_0001 = rep.build_client_packet(0)
    var trunc = List[UInt8]()
    for i in range(10):
        if i < len(p_0001[1]):
            trunc.append(p_0001[1][i])
    assert_false(cli.apply_packet(trunc)[0])
    assert_equal(Int(cli.last_decoded_seq), 0x0000)
    assert_true(cli.apply_packet(p_0001[1])[0])
    assert_equal(Int(cli.last_decoded_seq), 0x0001)

    # Tight 64-byte budget enforcement with 20 dirty entities
    var tight_rep = SnapshotReplicator(
        max_entities=32,
        max_clients=1,
        max_packet_bytes=64,
        interest_radius=200.0,
        world_bounds=256.0,
    )
    tight_rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 200.0)
    for i in range(20):
        tight_rep.set_entity(i, Vec3(Float32(i), 0.0, 0.0))
    for _ in range(5):
        tight_rep.prepare_tick()
        var pkt = tight_rep.build_client_packet(0)
        assert_true(len(pkt[1]) <= 64)


def test_fa7_t2_rollback_spoofed_peer_wrong_token_desync_and_bad_snapshot() raises:
    comptime TOKEN_A: UInt64 = DEFAULT_ROLLBACK_SESSION_TOKEN
    comptime TOKEN_B: UInt64 = 0xDEADBEEFCAFE0001

    var sim = VirtualNetSim.from_rtt(rtt_ms=10.0, jitter_ms=0.0, loss_rate=0.0, seed=0xF72007)
    var ep0 = sim.create_ipv4_peer(0, 9700)
    var ep1 = sim.create_ipv4_peer(1, 9700)
    var rogue_ep = sim.create_ipv4_peer(2, 9799)

    var addrs = List[NetEndpoint]()
    addrs.append(ep0.local_endpoint())
    addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw_rogue = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw_badtok = PhysicsRollbackAdapter.create_arena(num_peers=2)

    var s0 = RollbackSession[PhysicsRollbackInput](0, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4, session_token=TOKEN_A)
    var s1 = RollbackSession[PhysicsRollbackInput](1, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4, session_token=TOKEN_A)
    var s_rogue_ep = RollbackSession[PhysicsRollbackInput](1, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4, session_token=TOKEN_A)
    var s_wrong_tok = RollbackSession[PhysicsRollbackInput](1, addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4, session_token=TOKEN_B)

    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s_rogue_ep.enable_rift_pacing = False
    s_wrong_tok.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)
    s_rogue_ep.bind_world(pw_rogue)
    s_wrong_tok.bind_world(pw_badtok)

    # Send rogue-endpoint packet (valid TOKEN_A, wrong source endpoint) & wrong-token packet (from ep1, wrong TOKEN_B)
    var c_in = PhysicsRollbackInput(1.0, 0.0, 1.0, 0.0, 0)
    _ = s_rogue_ep.step_frame(c_in, pw_rogue, rogue_ep, sim.now_ms())
    _ = s_wrong_tok.step_frame(c_in, pw_badtok, ep1, sim.now_ms())
    sim.advance_ms(20.0)
    s0.poll_network(pw0, ep0, sim.now_ms())
    assert_true(s0.spoofed_packets_dropped >= 2)
    assert_false(s0.desync_detected)

    # Run synchronized session then deliberately perturb pw1 physics state -> desync_detected!
    for f in range(30):
        var now = sim.now_ms()
        if f == 12:
            var slot = pw1.world.find_slot_by_id(pw1.player_body_ids[1])
            pw1.world.bodies[slot].pos.x += Float32(5.0)
        _ = s0.step_frame(c_in, pw0, ep0, now)
        _ = s1.step_frame(c_in, pw1, ep1, now)
        sim.advance_ms(1000.0 / 60.0)

    assert_true(s0.desync_detected or s1.desync_detected)

    # Corrupt snapshot load surfaces error flag
    var bad_pw = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var garbage = List[UInt8](length=8, fill=UInt8(0xFF))
    bad_pw.load_snapshot(Int(garbage.unsafe_ptr()), len(garbage))
    _ = len(garbage)
    assert_true(bad_pw.snapshot_load_failed)
    assert_equal(bad_pw.snapshot_load_failures, 1)

    # Rift pacing stall check
    s0.enable_rift_pacing = True
    s0.rift_stall_threshold = Float32(2.0)
    s0.rift_advantage_ticks = Float32(3.5)
    assert_true(s0.should_stall_frame())
    assert_true(sim.now_ms() >= 0.0)


def test_fa7_t2_virtual_sim_mtu_drop_and_global_blackout() raises:
    var sim = VirtualNetSim(seed=0xF72008, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=0.0, mtu_bytes=200)
    var ep0 = sim.create_ipv4_peer(0, 9800)
    var ep1 = sim.create_ipv4_peer(1, 9800)

    # Over-MTU packet (201 B > 200 B) dropped
    assert_true(ep0.send_to(ep1.local_endpoint(), t2_make_bytes(1, 201)))
    sim.advance_ms(20.0)
    var rx = List[DatagramView](capacity=8)
    assert_equal(ep1.recv_batch(rx), 0)
    assert_equal(sim.packets_dropped_mtu(), 1)

    # Global blackout drops traffic during [now, now + 50), resumes afterward
    sim.trigger_global_blackout(50.0)
    assert_true(ep0.send_to(ep1.local_endpoint(), t2_make_bytes(2, 64)))
    sim.advance_ms(30.0)
    assert_equal(ep1.recv_batch(rx), 0)

    sim.advance_ms(25.0)  # t = 55ms past blackout
    assert_true(ep0.send_to(ep1.local_endpoint(), t2_make_bytes(3, 64)))
    sim.advance_ms(20.0)
    assert_equal(ep1.recv_batch(rx), 1)
    assert_true(sim.now_ms() >= 0.0)


def main() raises:
    set_sim_float_mode()
    var passed = 0
    var total = 16

    test_fa2_t2_fixed_time_spiral_of_death_and_invalid_schedule()
    print("[PASS] test_fa2_t2_fixed_time_spiral_of_death_and_invalid_schedule")
    passed += 1

    test_fa2_t2_entity_null_and_stale_generation_guards()
    print("[PASS] test_fa2_t2_entity_null_and_stale_generation_guards")
    passed += 1

    test_fa2_t2_scene_cycle_guard_and_clear_parent()
    print("[PASS] test_fa2_t2_scene_cycle_guard_and_clear_parent")
    passed += 1

    test_fa2_t2_prefab_max_nest_depth_and_corrupted_scene_rejections()
    print("[PASS] test_fa2_t2_prefab_max_nest_depth_and_corrupted_scene_rejections")
    passed += 1

    test_fa2_t2_restore_blob_error_and_non_mutation()
    print("[PASS] test_fa2_t2_restore_blob_error_and_non_mutation")
    passed += 1

    test_fa2_t2_detmath_extremes_nan_domain_and_mxcsr_heal()
    print("[PASS] test_fa2_t2_detmath_extremes_nan_domain_and_mxcsr_heal")
    passed += 1

    test_fa2_t2_jobpool_edge_cases_and_double_stop()
    print("[PASS] test_fa2_t2_jobpool_edge_cases_and_double_stop")
    passed += 1

    test_fa2_t2_memory_arena_overflow_negative_and_zero_capacity()
    print("[PASS] test_fa2_t2_memory_arena_overflow_negative_and_zero_capacity")
    passed += 1

    test_fa7_t2_udp_invalid_and_oversized_sends()
    print("[PASS] test_fa7_t2_udp_invalid_and_oversized_sends")
    passed += 1

    test_fa7_t2_bitreader_overflow_null_and_nan_bounds()
    print("[PASS] test_fa7_t2_bitreader_overflow_null_and_nan_bounds")
    passed += 1

    test_fa7_t2_aead_tamper_wrong_key_and_replay_bounds()
    print("[PASS] test_fa7_t2_aead_tamper_wrong_key_and_replay_bounds")
    passed += 1

    test_fa7_t2_net_channel_tamper_replay_corruption_and_rto_backoff()
    print("[PASS] test_fa7_t2_net_channel_tamper_replay_corruption_and_rto_backoff")
    passed += 1

    test_fa7_t2_net_hub_max_channels_strangers_and_idle_eviction()
    print("[PASS] test_fa7_t2_net_hub_max_channels_strangers_and_idle_eviction")
    passed += 1

    test_fa7_t2_replicator_seq_wraparound_truncation_and_budget_overflow()
    print("[PASS] test_fa7_t2_replicator_seq_wraparound_truncation_and_budget_overflow")
    passed += 1

    test_fa7_t2_rollback_spoofed_peer_wrong_token_desync_and_bad_snapshot()
    print("[PASS] test_fa7_t2_rollback_spoofed_peer_wrong_token_desync_and_bad_snapshot")
    passed += 1

    test_fa7_t2_virtual_sim_mtu_drop_and_global_blackout()
    print("[PASS] test_fa7_t2_virtual_sim_mtu_drop_and_global_blackout")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
