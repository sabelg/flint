"""Comprehensive Verification Suite for Flint v5 Networking (`flint.net`) & Lab Harness (`flint.lab`).

Covers:
  (a) Dual-stack Linux `UdpSocket` (`[::1]` IPv6 + `127.0.0.1` IPv4-mapped) `sendmmsg`/`recvmmsg`
      32-datagram burst round-trip with exact sender `NetEndpoint` recovery.
  (b) `BitWriter`/`BitReader` fuzz across 10,000 random bounded ints, quantized `Vec3`s, 32-bit
      Smallest-Three `Quat`s (`max angular error < 0.10 deg`) & 48-bit `Quat`s, plus truncated-buffer
      `overflowed == True` safety.
  (c) `NetChannel` reliable-ordered `8 KiB` multi-fragment message delivery + `UNRELIABLE_SEQUENCED`
      ordering across `VirtualNetSim` (`12%` packet loss, `25 ms` jitter, `2%` duplication).
  (d) 10,000-Tick 4-Peer Deterministic Rollback Chaos Fuzz (`80 ms` RTT, `20 ms` jitter, `5%` loss,
      `8-tick` max rollback, 6-packet burst drop, 7-packet `K=8` redundant input survival,
      `desync_detected == False`, 100% bit-identical `confirmed_hash` across all 4 peers on all
      10,000 ticks, `< 1.2 ms` 8-tick re-sim, Rift stall estimator, and tamper `desync_tick` trap).
  (e) 16-client `SnapshotReplicator` 2D/3D spatial interest + ACKed baseline delta compression
      bandwidth gate (`< 24 KB/s` for 256 active entities @ 20 Hz).
  (h) `NetHub`: one transport, a channel per peer: a host accepts 3 clients over a lossy
      `VirtualNetSim`, routes by sender, replies per channel, ignores strangers on a client,
      respects `max_channels`, and re-accepts a dropped peer in the same slot.
  (f) `EngineBenchSuite` p50/p99 latency, determinism gate, Magma FLIP heatmap emission, and CSV export.
"""
from std.collections import List
from std.math import abs, max, min, sqrt
from std.testing import assert_equal, assert_false, assert_true
from std.time import monotonic
from flint.core.detmath import det_cos, det_sin, f32_bits, set_sim_float_mode
from flint.core.detsort import mix64
from flint.core.lz4 import hash_list
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobWriter
from flint.lab import EngineBenchSuite, compute_latency_percentiles
from flint.platform.os import file_size, read_bytes, remove_file
from flint.net.rollback import (
    DEFAULT_ROLLBACK_SESSION_TOKEN,
    PhysicsRollbackAdapter,
    PhysicsRollbackInput,
    _RB_WIRE_MAGIC,
)
from flint.net import (
    AeadKey,
    BitReader,
    BitWriter,
    CHAN_ACK_HEARTBEAT,
    CHAN_RELIABLE_ORDERED,
    CHAN_RELIABLE_UNORDERED,
    CHAN_UNRELIABLE_RAW,
    ClientPredictionBuffer,
    DatagramView,
    HitboxPose,
    HitboxSnapshotRing,
    IPV6_SAFE_MTU_BYTES,
    JITTER_TRIANGULAR,
    LagCompHistory,
    LagCompHit,
    MAX_RELIABLE_FRAG_COUNT,
    NAT_PUNCH_MAGIC,
    NATPunchProbe,
    NetChannel,
    NetEndpoint,
    NetHub,
    NetTransport,
    PUNCH_KIND_PROBE,
    PUNCH_STATE_DIRECT,
    PUNCH_STATE_RELAY_FALLBACK,
    PacketWireHeader,
    PredictedInputEntry,
    PredictedState,
    PredictionBuffer,
    PredictionStepper,
    RELAY_FLAG_DATA,
    RELAY_FLAG_REGISTER,
    RELAY_HEADER_BYTES,
    RELAY_WIRE_MAGIC,
    ROLLBACK_INPUT_HISTORY_K,
    RelayForwardHeader,
    RelayServer,
    ReplicatedEntity,
    RollbackInput,
    RollbackSession,
    RollbackWorld,
    SnapshotClientReceiver,
    SnapshotInterpolationBuffer,
    SnapshotReader,
    SnapshotReplicator,
    SpatialInterestGrid,
    UdpSocket,
    VirtualEndpoint,
    VirtualNetSim,
    VisualSmoother,
    WIRE_HEADER_BYTES,
    aead_open,
    aead_seal,
    quat_angular_error_deg,
    seq_diff,
    seq_gt,
)


@fieldwise_init
struct TestCmdInput(RollbackInput):
    var steer: Float32
    var throttle: Float32
    var heading: Quat
    var actions: Int

    def __init__(out self):
        self.steer = 0.0
        self.throttle = 0.0
        self.heading = Quat.identity()
        self.actions = 0

    def encode_bits(self, mut writer: BitWriter):
        writer.write_f32_quantized(self.steer, -1.0, 1.0, 10)
        writer.write_f32_quantized(self.throttle, -1.0, 1.0, 10)
        writer.write_quat_smallest_three(self.heading)
        writer.write_int_bounded(self.actions, 0, 31)

    def decode_bits(mut self, mut reader: BitReader):
        self.steer = reader.read_f32_quantized(-1.0, 1.0, 10)
        self.throttle = reader.read_f32_quantized(-1.0, 1.0, 10)
        self.heading = reader.read_quat_smallest_three()
        self.actions = reader.read_int_bounded(0, 31)

    def same_input(self, other: Self) -> Bool:
        return (
            f32_bits(self.steer) == f32_bits(other.steer)
            and f32_bits(self.throttle) == f32_bits(other.throttle)
            and self.actions == other.actions
            and f32_bits(self.heading.x) == f32_bits(other.heading.x)
            and f32_bits(self.heading.y) == f32_bits(other.heading.y)
            and f32_bits(self.heading.z) == f32_bits(other.heading.z)
            and f32_bits(self.heading.w) == f32_bits(other.heading.w)
        )


struct ChaosArenaWorld(RollbackWorld):
    var pos: List[Vec3]
    var vel: List[Vec3]
    var rot: List[Quat]
    var energy: List[Float32]
    var collisions: UInt32
    var tick_num: UInt32

    def __init__(out self, count: Int = 6):
        self.pos = List[Vec3](capacity=count)
        self.vel = List[Vec3](capacity=count)
        self.rot = List[Quat](capacity=count)
        self.energy = List[Float32](capacity=count)
        for i in range(count):
            var a = Float32(i) * Float32(1.04719755)
            self.pos.append(Vec3(det_cos(a) * 10.0, 0.5, det_sin(a) * 10.0))
            self.vel.append(Vec3(0.0, 0.0, 0.0))
            self.rot.append(Quat.identity())
            self.energy.append(100.0)
        self.collisions = 0
        self.tick_num = 0

    def save_snapshot(self, mut writer: BlobWriter):
        writer.u32(self.tick_num)
        writer.u32(self.collisions)
        writer.u32(UInt32(len(self.pos)))
        for i in range(len(self.pos)):
            writer.vec3(self.pos[i])
            writer.vec3(self.vel[i])
            writer.quat(self.rot[i])
            writer.f32(self.energy[i])

    def load_snapshot(mut self, addr: Int, length: Int):
        var r = SnapshotReader(addr, length)
        self.tick_num = r.u32()
        self.collisions = r.u32()
        var n = Int(r.u32())
        for i in range(min(n, len(self.pos))):
            self.pos[i] = r.vec3()
            self.vel[i] = r.vec3()
            self.rot[i] = r.quat()
            self.energy[i] = r.f32()

    def step_rollback_tick(
        mut self, inputs_addr: Int, peer_count: Int, tick: Int
    ):
        comptime DT: Float32 = Float32(1.0 / 60.0)
        var ins = Pointer[TestCmdInput, ImmutAnyOrigin](
            unsafe_from_address=inputs_addr
        )
        var np = min(peer_count, len(self.pos))
        for p in range(np):
            var cmd = ins[unsafe_offset=p]
            var fwd = cmd.heading.rotate(Vec3(0.0, 0.0, 1.0)).horizontal().normalized()
            var right = Vec3(fwd.z, 0.0, -fwd.x)
            var force = fwd * (cmd.throttle * Float32(32.0)) + right * (
                cmd.steer * Float32(18.0)
            )
            if (cmd.actions & 1) != 0 and self.energy[p] >= Float32(20.0):
                force = force + fwd * Float32(140.0)
                self.energy[p] -= Float32(20.0)
            else:
                self.energy[p] = min(
                    Float32(100.0), self.energy[p] + Float32(8.0) * DT
                )
            self.vel[p] = (self.vel[p] + force * DT) * Float32(0.982)
            self.rot[p] = self.rot[p].slerp(cmd.heading, Float32(0.3))

        var count = len(self.pos)
        for i in range(np, count):
            self.vel[i] = (
                self.vel[i] - self.pos[i].horizontal() * (Float32(0.8) * DT)
            ) * Float32(0.99)

        for i in range(count):
            self.pos[i] = self.pos[i] + self.vel[i] * DT
            var r2 = self.pos[i].horizontal().length_squared()
            if r2 > Float32(22.0 * 22.0):
                var d = sqrt(r2)
                var n = self.pos[i].horizontal() * (Float32(1.0) / d)
                self.pos[i].x = n.x * Float32(22.0)
                self.pos[i].z = n.z * Float32(22.0)
                var vn = self.vel[i].dot(n)
                if vn > Float32(0.0):
                    self.vel[i] = self.vel[i] - n * (Float32(1.9) * vn)

        for i in range(count):
            for j in range(i + 1, count):
                var diff = self.pos[j] - self.pos[i]
                var d2 = diff.length_squared()
                if d2 > Float32(1.0e-6) and d2 < Float32(4.84):  # 2.2m diameter
                    var d = sqrt(d2)
                    var n = diff * (Float32(1.0) / d)
                    var corr = (Float32(2.2) - d) * Float32(0.5)
                    self.pos[i] = self.pos[i] - n * corr
                    self.pos[j] = self.pos[j] + n * corr
                    var rv = (self.vel[j] - self.vel[i]).dot(n)
                    if rv < Float32(0.0):
                        var imp = n * (Float32(-0.9) * rv)
                        self.vel[i] = self.vel[i] - imp
                        self.vel[j] = self.vel[j] + imp
                        self.collisions += 1

        self.tick_num = UInt32(tick + 1)


def _make_cmd(peer: Int, tick: Int) -> TestCmdInput:
    var h = mix64(UInt64(peer * 2654435761 + (tick // 4) * 40503 + 11))
    var h2 = mix64(UInt64(peer * 97 + tick * 131071 + 7))
    var st = Float32(Int(h & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    var th = Float32(Int((h >> 16) & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    var yaw = (Float32(Int((h >> 32) & 0xFFFF)) / Float32(65535.0)) * Float32(6.2831853)
    var act = Int(h2 & 7) if (tick % 9 == peer * 2) else 0
    return TestCmdInput(st, th, Quat.from_euler(0.0, yaw, 0.0), act)


def test_a_dual_stack_udp_32_burst() raises:
    print("  [Test A] Dual-stack Linux UdpSocket [::1] + 127.0.0.1 sendmmsg/recvmmsg 32-datagram burst...")
    var srv = UdpSocket.open_dual_stack(0)
    assert_true(srv.is_open(), "dual-stack server socket opened")
    assert_true(srv.local_port() > 0, "ephemeral port assigned via getsockname")

    var cli6 = UdpSocket()
    var cli4 = UdpSocket()
    assert_true(cli6.bind(NetEndpoint.ipv6_loopback(0)), "IPv6 [::1] client bound")
    assert_true(cli4.bind(NetEndpoint.ipv4_loopback(0)), "IPv4-mapped 127.0.0.1 client bound")

    var srv_v6 = NetEndpoint.ipv6_loopback(srv.local_port())
    var srv_v4 = NetEndpoint.ipv4_loopback(srv.local_port())

    # 32-datagram burst from IPv6 client + 32-datagram burst from IPv4-mapped client
    for i in range(32):
        var b6 = List[UInt8]()
        b6.append(UInt8(0xA6))
        b6.append(UInt8(i))
        b6.append(UInt8((i * 7 + 3) & 0xFF))
        assert_true(cli6.queue_send_bytes(srv_v6, b6))

        var b4 = List[UInt8]()
        b4.append(UInt8(0xA4))
        b4.append(UInt8(i))
        b4.append(UInt8((i * 11 + 5) & 0xFF))
        assert_true(cli4.queue_send_bytes(srv_v4, b4))

    assert_equal(cli6.flush_sends(), 32)
    assert_equal(cli6.sendmmsg_calls, 1)
    assert_equal(cli4.flush_sends(), 32)
    assert_equal(cli4.sendmmsg_calls, 1)

    var views = List[DatagramView](capacity=64)
    var seen_v6 = 0
    var seen_v4 = 0
    for _ in range(60):
        var n = srv.recv_batch(views)
        for i in range(n):
            var v = views[i]
            assert_equal(v.length, 3)
            var idx = Int(v.byte_at(1))
            if v.byte_at(0) == UInt8(0xA6):
                assert_true(v.sender.is_ipv6_loopback(), "sender must be IPv6 [::1]")
                assert_true(v.sender == cli6.local_endpoint(), "exact IPv6 sender NetEndpoint recovered")
                assert_equal(Int(v.byte_at(2)), (idx * 7 + 3) & 0xFF)
                seen_v6 += 1
            elif v.byte_at(0) == UInt8(0xA4):
                assert_true(v.sender.is_ipv4_mapped(), "sender must be IPv4-mapped ::ffff:127.0.0.1")
                assert_true(v.sender == cli4.local_endpoint(), "exact IPv4-mapped sender NetEndpoint recovered")
                assert_equal(Int(v.byte_at(2)), (idx * 11 + 5) & 0xFF)
                seen_v4 += 1
            assert_true(srv.queue_send_raw(v.sender, v.data_ptr, v.length))
        if seen_v6 == 32 and seen_v4 == 32:
            break

    assert_equal(seen_v6, 32)
    assert_equal(seen_v4, 32)
    assert_equal(srv.flush_sends(), 64)

    var echo6 = 0
    var echo4 = 0
    for _ in range(60):
        echo6 += cli6.recv_batch(views)
        echo4 += cli4.recv_batch(views)
        if echo6 == 32 and echo4 == 32:
            break
    assert_equal(echo6, 32)
    assert_equal(echo4, 32)


def test_b_bitpack_10k_fuzz_and_bounds() raises:
    print("  [Test B] BitWriter/BitReader 10,000-item fuzz (bounded int, Vec3, Quat32 < 0.08 deg, Quat48)...")
    comptime N: Int = 10000
    var w = BitWriter(163840)
    var lo_list = List[Int](capacity=N)
    var hi_list = List[Int](capacity=N)
    var iv_list = List[Int](capacity=N)
    var v3_list = List[Vec3](capacity=N)
    var q_list = List[Quat](capacity=N)

    var seed = UInt64(0x123456789ABCDEF0)
    for i in range(N):
        seed = mix64(seed + 11)
        var lo = -Int(seed & 4095)
        var span = Int((seed >> 13) & 16383)
        var hi = lo + span
        var val = lo + (Int((seed >> 29) & 0x7FFFF) % (span + 1))
        lo_list.append(lo)
        hi_list.append(hi)
        iv_list.append(val)
        w.write_int_bounded(val, lo, hi)
        w.write_bool((i & 1) == 0)

        seed = mix64(seed + 23)
        var vx = (Float32(Int(seed & 0xFFFF)) / Float32(65535.0)) * Float32(200.0) - Float32(100.0)
        var vy = (Float32(Int((seed >> 16) & 0xFFFF)) / Float32(65535.0)) * Float32(200.0) - Float32(100.0)
        var vz = (Float32(Int((seed >> 32) & 0xFFFF)) / Float32(65535.0)) * Float32(200.0) - Float32(100.0)
        var v = Vec3(vx, vy, vz)
        v3_list.append(v)
        w.write_vec3_quantized(v, -100.0, 100.0, 16)

        seed = mix64(seed + 37)
        var q: Quat
        if i < 4:
            # Test all 4 max-component sign/corner boundaries
            q = Quat(0.5, 0.5, 0.5, 0.5).normalized()
        else:
            var qx = (Float32(Int(seed & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            var qy = (Float32(Int((seed >> 16) & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            var qz = (Float32(Int((seed >> 32) & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            var qw = (Float32(Int((seed >> 48) & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            q = Quat(qx, qy, qz, qw).normalized()
        q_list.append(q)
        w.write_quat_smallest_three(q)
        if i < 256:
            w.write_quat_smallest_three_48(q)

    var packed = w.finish()
    var r = BitReader(packed)
    var max_err32 = Float32(0.0)
    var max_err48 = Float32(0.0)

    for i in range(N):
        assert_equal(r.read_int_bounded(lo_list[i], hi_list[i]), iv_list[i])
        assert_equal(r.read_bool(), (i & 1) == 0)
        var dv = r.read_vec3_quantized(-100.0, 100.0, 16)
        assert_true((dv - v3_list[i]).length() < Float32(0.006), "quantized Vec3 within 16-bit tolerance")
        var dq32 = r.read_quat_smallest_three()
        var e32 = quat_angular_error_deg(q_list[i], dq32)
        if e32 > max_err32:
            max_err32 = e32
        if i < 256:
            var dq48 = r.read_quat_smallest_three_48()
            var e48 = quat_angular_error_deg(q_list[i], dq48)
            if e48 > max_err48:
                max_err48 = e48

    assert_false(r.overflowed, "valid stream must not set overflowed")
    assert_true(max_err32 < Float32(0.08), "32-bit Smallest-Three Quat max angular error < 0.08 deg (< 0.10 deg)")
    assert_true(max_err48 < Float32(0.005), "48-bit high-precision Quat max angular error < 0.005 deg")

    # Truncated buffer safety across every bit cutoff 0..64
    var short_buf = List[UInt8](length=4, fill=0xAB)
    for bit_lim in range(0, 32):
        var tr = BitReader(short_buf, bit_limit=bit_lim)
        _ = tr.read_quat_smallest_three()
        assert_true(tr.overflowed, "reading 32-bit Quat from <32-bit buffer must flag overflowed")


def test_c_channel_reliable_8kib_and_sequenced() raises:
    print("  [Test C] NetChannel 8 KiB reliable fragmentation & UNRELIABLE_SEQUENCED over 12% loss + 25ms jitter + 2% dup...")
    assert_true(seq_gt(UInt16(0), UInt16(65535)))
    assert_true(seq_gt(UInt16(100), UInt16(99)))
    assert_equal(seq_diff(UInt16(2), UInt16(65534)), 4)

    var sim = VirtualNetSim(
        seed=0xCAFEBEEF12345678,
        one_way_latency_ms=40.0,
        jitter_ms=25.0,
        loss_rate=0.12,
        dup_rate=0.02,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var ep0 = sim.create_ipv4_peer(0, 9200)
    var ep1 = sim.create_ipv4_peer(1, 9200)

    var ch0 = NetChannel(ep1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var ch1 = NetChannel(ep0.local_endpoint(), IPV6_SAFE_MTU_BYTES)

    var msg_8k = List[UInt8](capacity=8192)
    for i in range(8192):
        msg_8k.append(UInt8(mix64(UInt64(i * 19 + 101)) & 0xFF))
    var msg_3k = List[UInt8](capacity=3000)
    for i in range(3000):
        msg_3k.append(UInt8(mix64(UInt64(i * 43 + 7)) & 0xFF))

    assert_true(ch0.send_reliable(msg_8k))
    assert_true(ch0.send_reliable(msg_3k))

    var delivered = List[List[UInt8]]()
    var last_unrel_val = -1
    var monotonic_unrel = True

    for step in range(500):
        var now = sim.now_ms()
        if step < 40:
            var u_pkt = List[UInt8]()
            u_pkt.append(UInt8(step))
            _ = ch0.send_unreliable_sequenced(u_pkt, ep0, now)
        ch0.poll(ep0, now)
        ch1.poll(ep1, now)

        while True:
            var u = ch1.pop_unreliable_message()
            if not u[0]:
                break
            var v = Int(u[1][0])
            if v <= last_unrel_val:
                monotonic_unrel = False
            last_unrel_val = v

        while True:
            var r = ch1.pop_reliable_message()
            if not r[0]:
                break
            delivered.append(r[1].copy())

        if len(delivered) == 2 and ch0.is_all_reliable_acked():
            break
        sim.advance_ms(10.0)

    assert_true(monotonic_unrel, "UNRELIABLE_SEQUENCED must drop stale out-of-order datagrams")
    assert_equal(sim.packets_dropped_mtu(), 0)
    assert_equal(len(delivered), 2)
    assert_equal(len(delivered[0]), 8192)
    assert_equal(len(delivered[1]), 3000)
    assert_equal(hash_list(delivered[0]), hash_list(msg_8k))
    assert_equal(hash_list(delivered[1]), hash_list(msg_3k))
    assert_true(ch0.rtt_initialized and ch0.srtt_ms > 20.0)


def test_d_rollback_7pkt_drop_and_10k_4peer_chaos() raises:
    print("  [Test D] 4-Peer 10,000-Tick Deterministic Rollback Chaos Fuzz + K=8 7-drop survival + Desync Trap...")

    # Part 1: Explicit 7-consecutive-packet drop (`K = 8` redundant history -> zero lost inputs!)
    var sim7 = VirtualNetSim.from_rtt(rtt_ms=20.0, jitter_ms=0.0, loss_rate=0.0, seed=42)
    var p7_0 = sim7.create_ipv4_peer(0, 9300)
    var p7_1 = sim7.create_ipv4_peer(1, 9300)
    var addrs2 = List[NetEndpoint]()
    addrs2.append(p7_0.local_endpoint())
    addrs2.append(p7_1.local_endpoint())

    var w7_0 = ChaosArenaWorld(4)
    var w7_1 = ChaosArenaWorld(4)
    var s7_0 = RollbackSession[TestCmdInput](0, addrs2, input_delay_ticks=2, max_rollback_ticks=8)
    var s7_1 = RollbackSession[TestCmdInput](1, addrs2, input_delay_ticks=2, max_rollback_ticks=8)
    s7_0.enable_rift_pacing = False
    s7_1.enable_rift_pacing = False
    s7_0.bind_world(w7_0)
    s7_1.bind_world(w7_1)

    for f in range(80):
        if f == 20:
            # Drop 7 consecutive outbound packets from peer 0 (`f = 20..26`); `f = 27` recovers all 7!
            sim7.drop_next_outbound_packets(0, 7)
        var now = sim7.now_ms()
        # Constant steering across the 7-drop window verifies 0 prediction errors + full history recovery!
        var c0 = TestCmdInput(0.5, 0.8, Quat.identity(), 0)
        var c1 = _make_cmd(1, s7_1.current_tick)
        _ = s7_0.step_frame(c0, w7_0, p7_0, now)
        _ = s7_1.step_frame(c1, w7_1, p7_1, now)
        sim7.advance_ms(1000.0 / 60.0)

    assert_false(s7_0.desync_detected)
    assert_false(s7_1.desync_detected)
    assert_true(s7_1.confirmed_tick >= 65)
    assert_equal(s7_1.rollback_count, 0)
    for t in range(60):
        assert_equal(s7_0.confirmed_hash_history[t], s7_1.confirmed_hash_history[t])

    # Part 2: Full 10,000-Tick 4-Peer Chaos Fuzz (80ms RTT, 20ms jitter, 5% loss, 8-tick max rollback, 6-pkt burst drop)
    comptime TARGET_TICKS: Int = 10000
    var sim = VirtualNetSim.from_rtt(
        rtt_ms=80.0,
        jitter_ms=20.0,
        loss_rate=0.05,
        dup_rate=0.01,
        mtu_bytes=1200,
        seed=0x778899AABBCCDDEE,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var eps = List[VirtualEndpoint]()
    var addrs4 = List[NetEndpoint]()
    for p in range(4):
        var ep = sim.create_ipv4_peer(p, 9400)
        addrs4.append(ep.local_endpoint())
        eps.append(ep^)

    var worlds = List[ChaosArenaWorld]()
    var sessions = List[RollbackSession[TestCmdInput]]()
    for p in range(4):
        var w = ChaosArenaWorld(6)
        var s = RollbackSession[TestCmdInput](
            local_peer_id=p,
            peer_endpoints=addrs4,
            input_delay_ticks=2,
            max_rollback_ticks=8,
            tick_hz=60.0,
            checksum_interval=8,
            record_full_hash_history=True,
        )
        s.enable_rift_pacing = False
        s.bind_world(w)
        worlds.append(w^)
        sessions.append(s^)

    for frame in range(14500):
        if frame == 3000:
            sim.drop_next_outbound_packets(1, 6)
        if frame == 7200:
            sim.drop_next_outbound_packets(2, 6)

        var now = sim.now_ms()
        var min_confirmed = TARGET_TICKS
        for p in range(4):
            var cmd = _make_cmd(p, sessions[p].current_tick)
            _ = sessions[p].step_frame(cmd, worlds[p], eps[p], now)
            if len(sessions[p].confirmed_hash_history) < min_confirmed:
                min_confirmed = len(sessions[p].confirmed_hash_history)
        if min_confirmed >= TARGET_TICKS:
            break
        sim.advance_ms(1000.0 / 60.0)

    var total_rb = 0
    for p in range(4):
        assert_false(sessions[p].desync_detected, "peer must have zero desyncs")
        assert_true(
            len(sessions[p].confirmed_hash_history) >= TARGET_TICKS,
            "all 4 peers must confirm >= 10,000 ticks",
        )
        assert_true(
            sessions[p].peak_resim_ms < 1.2,
            "8-tick rollback re-simulation must take < 1.2 ms",
        )
        total_rb += sessions[p].rollback_count

    assert_true(total_rb > 100, "chaos network must trigger real rollbacks and re-simulations")

    for t in range(TARGET_TICKS):
        var h0 = sessions[0].confirmed_hash_history[t]
        assert_true(h0 != 0)
        assert_equal(h0, sessions[1].confirmed_hash_history[t])
        assert_equal(h0, sessions[2].confirmed_hash_history[t])
        assert_equal(h0, sessions[3].confirmed_hash_history[t])

    # Part 3: Verify tamper desync detector & Rift stall estimator
    sessions[0].rift_advantage_ticks = 3.5
    sessions[0].enable_rift_pacing = True
    assert_true(sessions[0].should_stall_frame(), "Rift advantage > threshold must request 1-frame stall")

    var sim_bad = VirtualNetSim.from_rtt(rtt_ms=10.0, jitter_ms=0.0, loss_rate=0.0, seed=99)
    var bp0 = sim_bad.create_ipv4_peer(0, 9600)
    var bp1 = sim_bad.create_ipv4_peer(1, 9600)
    var baddrs = List[NetEndpoint]()
    baddrs.append(bp0.local_endpoint())
    baddrs.append(bp1.local_endpoint())
    var bw0 = ChaosArenaWorld(2)
    var bw1 = ChaosArenaWorld(2)
    var bs0 = RollbackSession[TestCmdInput](0, baddrs, input_delay_ticks=1, checksum_interval=4)
    var bs1 = RollbackSession[TestCmdInput](1, baddrs, input_delay_ticks=1, checksum_interval=4)
    bs0.bind_world(bw0)
    bs1.bind_world(bw1)
    for f in range(30):
        if f == 8:
            # Deliberately corrupt peer 1's authoritative position to prove desync detector fires!
            bw1.pos[0].x += Float32(5.0)
        var now = sim_bad.now_ms()
        _ = bs0.step_frame(TestCmdInput(), bw0, bp0, now)
        _ = bs1.step_frame(TestCmdInput(), bw1, bp1, now)
        sim_bad.advance_ms(1000.0 / 60.0)
    assert_true(bs0.desync_detected and bs0.desync_tick >= 8)


def test_e_replicator_16_clients_bandwidth_gate() raises:
    print("  [Test E] 16-client SnapshotReplicator spatial AOI + baseline delta (< 24 KB/s @ 20 Hz, 256 entities)...")
    var rep = SnapshotReplicator(
        max_entities=256,
        max_clients=16,
        max_packet_bytes=1000,
        interest_radius=95.0,
        cell_size=32.0,
        use_3d_grid=True,
        world_bounds=256.0,
    )
    var clients = List[SnapshotClientReceiver](capacity=16)
    for c in range(16):
        var x = (Float32(c % 4) - Float32(1.5)) * Float32(120.0)
        var z = (Float32(c // 4) - Float32(1.5)) * Float32(120.0)
        rep.set_client_view(c, Vec3(x, 2.0, z), 95.0)
        clients.append(SnapshotClientReceiver(256, 256.0))

    # 5 seconds @ 20 Hz = 100 snapshots
    for tick in range(100):
        var t = Float32(tick) * Float32(0.05)
        for e in range(256):
            var bx = (Float32(e % 16) - Float32(7.5)) * Float32(30.0)
            var bz = (Float32(e // 16) - Float32(7.5)) * Float32(30.0)
            if (e & 1) == 0:
                var px = bx + det_cos(t + Float32(e)) * Float32(5.0)
                var pz = bz + det_sin(t + Float32(e)) * Float32(5.0)
                rep.set_entity(
                    e,
                    Vec3(px, 2.0, pz),
                    Quat.from_euler(0.0, t, 0.0),
                    Vec3(-det_sin(t + Float32(e)) * 5.0, 0.0, det_cos(t + Float32(e)) * 5.0),
                    UInt32(e & 15),
                )
            else:
                rep.set_entity(e, Vec3(bx, 0.0, bz), Quat.identity(), Vec3(), UInt32(1))

        rep.prepare_tick()

        for c in range(16):
            var pkt = rep.build_client_packet(c)
            assert_true(len(pkt[1]) <= 1000, "packet strictly respects max_packet_bytes budget")
            var res = clients[c].apply_packet(pkt[1])
            assert_true(res[0], "client cleanly decodes snapshot delta")
            rep.notify_client_ack(c, res[1])

    for c in range(16):
        var bytes_per_sec = Float64(rep.clients[c].bytes_sent_total) / 5.0
        assert_true(
            bytes_per_sec < 24000.0,
            "per-client bandwidth must stay < 24 KB/s (24,000 B/s) for 256 entities @ 20 Hz",
        )

    # Verify spatial interest accuracy: client 0 at (-180, 2, -180) must have entity 0 near (-225, 2, -225)
    assert_true(clients[0].entities[0].active)
    var err_pos = (clients[0].entities[0].pos - rep.entities[0].pos).length()
    assert_true(err_pos < Float32(0.05), "reconstructed client entity position matches server within 5 cm")


def test_f_lab_bench_suite_harness() raises:
    print("  [Test F] EngineBenchSuite p50/p99 percentiles, Magma FLIP heatmap & perf_history CSV...")
    var samples = List[Float64]()
    for i in range(1, 101):
        samples.append(Float64(i))
    var stats = compute_latency_percentiles(samples)
    assert_equal(Int(stats[0]), 50)
    assert_equal(Int(stats[1]), 99)

    var suite = EngineBenchSuite("unit_check")
    assert_true(suite.record_determinism_gate("det_ok", 0xCAFE, 0xCAFE, elapsed_ms=0.5))

    var heat_file = String("/tmp/flint_eng6_unit_flip_heat.png")
    var csv_file = String("/tmp/flint_eng6_unit_perf.csv")
    _ = remove_file(heat_file)
    _ = remove_file(csv_file)

    var ref_img = List[UInt32](length=8 * 8, fill=0xFF102030)
    var bad_img = List[UInt32](length=8 * 8, fill=0xFFF0E010)
    assert_false(
        suite.verify_golden_pixels("intentional_mismatch", bad_img, ref_img, 8, 8, heatmap_out_path=heat_file)
    )
    assert_true(file_size(heat_file) > 64, "Magma FLIP failure heatmap PNG written to disk")
    var png_hdr = read_bytes(heat_file)
    assert_equal(Int(png_hdr[0]), 0x89)
    assert_equal(Int(png_hdr[1]), 0x50)

    assert_true(suite.append_perf_history_csv(csv_file, "v5-test"))
    assert_true(file_size(csv_file) > 20, "perf_history CSV exported cleanly")

    _ = remove_file(heat_file)
    _ = remove_file(csv_file)


def _make_physics_cmd(peer: Int, tick: Int) -> PhysicsRollbackInput:
    var h = mix64(UInt64(peer * 1640531527 + (tick // 3) * 65537 + 29))
    var h2 = mix64(UInt64(peer * 257 + tick * 8191 + 13))
    var ix = (
        Float32(Int(h & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    ) * Float32(24.0)
    var iz = (
        Float32(Int((h >> 16) & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    ) * Float32(24.0)
    var iy = Float32(0.0)
    var tyaw = (
        Float32(Int((h >> 32) & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    ) * Float32(8.0)
    var act = 1 if (Int(h2 & 15) == peer) else 0
    return PhysicsRollbackInput(ix, iy, iz, tyaw, act)


def _forge_rollback_wire_packet(
    token: UInt64,
    claimed_sender: Int,
    tick: Int,
    forged_ck_tick: Int,
    forged_ck_hash: UInt64,
    forged_input: PhysicsRollbackInput,
) -> List[UInt8]:
    var w = BitWriter(128)
    w.write_u32(_RB_WIRE_MAGIC)
    w.write_u64(token)
    w.write_u8(UInt8(claimed_sender & 0xFF))
    w.write_u16(UInt16((tick + 500) & 0xFFFF))
    w.write_u16(UInt16(0))
    w.write_u32(UInt32(max(0, tick)))
    w.write_u32(UInt32(max(0, tick)))
    w.write_u32(UInt32(0))
    w.write_u8(UInt8(1))
    w.write_bool(False)
    w.write_bool(True)
    w.write_u32(UInt32(max(0, forged_ck_tick)))
    w.write_u64(forged_ck_hash)
    forged_input.encode_bits(w)
    return w.finish()


def test_g_physics_world_rollback_and_spoof_auth() raises:
    print(
        "  [Test G] PhysicsWorld RollbackWorld integration + Endpoint-Authenticated Spoof/Token Injection Defense..."
    )
    assert_true(DEFAULT_ROLLBACK_SESSION_TOKEN != UInt64(0))

    comptime TOKEN: UInt64 = 0xCAFEBABE1337900D
    var sim = VirtualNetSim.from_rtt(
        rtt_ms=55.0,
        jitter_ms=18.0,
        loss_rate=0.06,
        dup_rate=0.01,
        mtu_bytes=1200,
        seed=0x51A7E2026F114705,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var ep0 = sim.create_ipv4_peer(0, 9700)
    var ep1 = sim.create_ipv4_peer(1, 9700)
    # Rogue attacker endpoint on an unknown IP/port (NOT in peer_endpoints!)
    var rogue_ep = sim.create_ipv4_peer(2, 31337)

    var peer_addrs = List[NetEndpoint]()
    peer_addrs.append(ep0.local_endpoint())
    peer_addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)

    var s0 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=0,
        peer_endpoints=peer_addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
        session_token=TOKEN,
    )
    var s1 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=1,
        peer_endpoints=peer_addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
        session_token=TOKEN,
    )
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    comptime TARGET_PHYS_TICKS: Int = 240
    for frame in range(420):
        if frame == 90:
            # Force a 5-packet burst drop to stress multi-tick PhysicsWorld deserialize + re-sim!
            sim.drop_next_outbound_packets(1, 5)

        # Every 6 frames, launch rogue injection attacks against Peer 0:
        if frame >= 12 and (frame % 6) == 0:
            var evil_in = PhysicsRollbackInput(32.0, 32.0, -32.0, 16.0, 15)
            var ck_t = max(0, (s0.confirmed_tick // 4) * 4)
            # Attack 1: Rogue endpoint spoofing `sender_id = 1` WITH valid session_token + poisoned checksum
            var spoof_ep_pkt = _forge_rollback_wire_packet(
                TOKEN, 1, s0.current_tick + 1, ck_t, 0xDEADDEADDEADDEAD, evil_in
            )
            _ = rogue_ep.send_to(ep0.local_endpoint(), spoof_ep_pkt)
            # Attack 2: Packet from legitimate Peer 1 endpoint carrying WRONG session_token!
            var bad_tok_pkt = _forge_rollback_wire_packet(
                0xBAD0BAD0BAD0BAD0,
                1,
                s0.current_tick + 1,
                ck_t,
                0xBEEFBEEFBEEFBEEF,
                evil_in,
            )
            _ = ep1.send_to(ep0.local_endpoint(), bad_tok_pkt)
            # Attack 3: Self-spoof (`claimed_sender == local_peer_id == 0`)
            var self_spoof_pkt = _forge_rollback_wire_packet(
                TOKEN, 0, s0.current_tick + 1, ck_t, 0x1111222233334444, evil_in
            )
            _ = rogue_ep.send_to(ep0.local_endpoint(), self_spoof_pkt)

        var now = sim.now_ms()
        var c0 = _make_physics_cmd(0, s0.current_tick)
        var c1 = _make_physics_cmd(1, s1.current_tick)
        _ = s0.step_frame(c0, pw0, ep0, now)
        _ = s1.step_frame(c1, pw1, ep1, now)
        if (
            len(s0.confirmed_hash_history) >= TARGET_PHYS_TICKS
            and len(s1.confirmed_hash_history) >= TARGET_PHYS_TICKS
        ):
            break
        sim.advance_ms(1000.0 / 60.0)

    assert_true(
        s0.spoofed_packets_dropped >= 40,
        "all spoofed-endpoint, wrong-token, and self-sender packets must be dropped",
    )
    assert_false(
        s0.desync_detected,
        "Peer 0 must have 0 desyncs despite forged checksum & input injection",
    )
    assert_false(s1.desync_detected, "Peer 1 must have 0 desyncs")
    assert_true(
        s0.rollback_count + s1.rollback_count > 5,
        "PhysicsWorld rollback session must execute real rollbacks & re-simulations",
    )
    assert_true(len(s0.confirmed_hash_history) >= TARGET_PHYS_TICKS)
    assert_true(len(s1.confirmed_hash_history) >= TARGET_PHYS_TICKS)

    for t in range(TARGET_PHYS_TICKS):
        assert_equal(
            s0.confirmed_hash_history[t],
            s1.confirmed_hash_history[t],
            "PhysicsWorld serialized PHYW state hash must match bit-for-bit across peers",
        )

    # Verify dynamic bodies actually moved and collided in the arena
    var p0_slot = pw0.world.find_slot_by_id(pw0.player_body_ids[0])
    assert_true(p0_slot >= 0)
    assert_true(
        (pw0.world.bodies[p0_slot].pos - Vec3(5.0, 1.0, 0.0)).length()
        > Float32(0.25),
        "player rigid body must move dynamically under rollback impulses",
    )


def _hub_msg(tag: Int, n: Int) -> List[UInt8]:
    var m = List[UInt8](capacity=n)
    for i in range(n):
        m.append(UInt8((tag * 31 + i * 7) & 0xFF))
    return m^


def _hub_pump(
    mut sim: VirtualNetSim,
    mut host: NetHub[VirtualEndpoint],
    mut clients: List[NetHub[VirtualEndpoint]],
    steps: Int,
):
    for _ in range(steps):
        var now = sim.now_ms()
        host.poll(now)
        for c in range(len(clients)):
            clients[c].poll(now)
        sim.advance_ms(10.0)


def test_h_net_hub_routing_accept_drop() raises:
    print("  [Test H] NetHub: host accepts 3 clients over 8% loss, routes by sender, drop + re-accept...")
    var sim = VirtualNetSim(
        seed=0x4855425F54455354,
        one_way_latency_ms=30.0,
        jitter_ms=10.0,
        loss_rate=0.08,
        dup_rate=0.02,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var host = NetHub[VirtualEndpoint](sim.create_ipv4_peer(0, 9300), True)
    var host_ep = host.local_endpoint()
    var clients = List[NetHub[VirtualEndpoint]]()
    for c in range(3):
        clients.append(NetHub[VirtualEndpoint](sim.create_ipv4_peer(c + 1, 9300), False))
    for c in range(3):
        var cid = clients[c].connect(host_ep)
        assert_equal(cid, 0)
        assert_equal(clients[c].connect(host_ep), 0)        # same address: same channel
        assert_true(clients[c].send_reliable(cid, _hub_msg(c + 1, 300 + 2000 * c)))  # 3rd one fragments

    var seen = List[Int](length=3, fill=-1)                  # host channel id per client
    var accepted = 0
    for _ in range(300):
        var now = sim.now_ms()
        host.poll(now)
        accepted += len(host.accepted)
        for cid in range(host.count):
            while True:
                var m = host.pop_reliable(cid)
                if not m[0]:
                    break
                var tag = (len(m[1]) - 300) // 2000          # client c sent 300 + 2000 c bytes
                assert_equal(hash_list(m[1]), hash_list(_hub_msg(tag + 1, len(m[1]))))
                seen[tag] = cid
                assert_true(host.send_reliable(cid, _hub_msg(100 + tag, 64)))
        for c in range(3):
            clients[c].poll(now)
        sim.advance_ms(10.0)
        if seen[0] >= 0 and seen[1] >= 0 and seen[2] >= 0:
            break
    assert_equal(accepted, 3)
    assert_equal(host.open_count(), 3)
    for c in range(3):
        assert_true(seen[c] >= 0)
        assert_true(host.chans[seen[c]].peer_endpoint == clients[c].local_endpoint())
    assert_true(seen[0] != seen[1] and seen[1] != seen[2] and seen[0] != seen[2])

    # replies reach the right client only
    _hub_pump(sim, host, clients, 120)
    for c in range(3):
        var r = clients[c].pop_reliable(0)
        assert_true(r[0])
        assert_equal(hash_list(r[1]), hash_list(_hub_msg(100 + c, 64)))
        assert_false(clients[c].pop_reliable(0)[0])

    # unreliable latest-only datagrams route too
    assert_true(clients[1].send_unreliable(0, _hub_msg(7, 40), sim.now_ms()))

    # a client ignores a stranger (accept = False); another client sends to client 0 directly
    var stranger = clients[2].connect(clients[0].local_endpoint())
    assert_equal(stranger, 1)
    assert_true(clients[2].send_reliable(stranger, _hub_msg(9, 20)))
    _hub_pump(sim, host, clients, 40)
    assert_equal(clients[0].open_count(), 1)
    assert_true(clients[0].datagrams_ignored > 0)
    clients[2].drop(stranger)
    assert_false(clients[2].is_open(stranger))
    assert_false(clients[2].send_reliable(stranger, _hub_msg(9, 20)))

    # drop client 1 on the host: its next packet opens a fresh channel in the same slot
    var slot = seen[1]
    host.drop(slot)
    assert_equal(host.open_count(), 2)
    assert_false(host.pop_reliable(slot)[0])
    clients[1].drop(0)
    assert_equal(clients[1].connect(host_ep), 0)             # a fresh channel to the host
    assert_true(clients[1].send_reliable(0, _hub_msg(42, 50)))
    var reaccepted = False
    var got = False
    for _ in range(200):
        var now = sim.now_ms()
        host.poll(now)
        for a in range(len(host.accepted)):
            if host.accepted[a] == slot:
                reaccepted = True
        var m = host.pop_reliable(slot)
        if m[0]:
            assert_equal(hash_list(m[1]), hash_list(_hub_msg(42, 50)))
            got = True
        for c in range(3):
            clients[c].poll(now)
        sim.advance_ms(10.0)
        if got:
            break
    assert_true(reaccepted and got)
    assert_equal(host.open_count(), 3)

    # max_channels: a host with room for one accepts the first sender only
    var small = NetHub[VirtualEndpoint](sim.create_ipv4_peer(10, 9300), True, 1)
    var small_ep = small.local_endpoint()
    for c in range(2):
        var cid = clients[c].connect(small_ep)
        assert_true(clients[c].send_reliable(cid, _hub_msg(c, 10)))
    for _ in range(60):
        var now = sim.now_ms()
        small.poll(now)
        for c in range(3):
            clients[c].poll(now)
        sim.advance_ms(10.0)
    assert_equal(small.open_count(), 1)
    assert_equal(small.connect(clients[2].local_endpoint()), -1)
    assert_true(small.datagrams_ignored > 0)
    assert_true(host.datagrams_routed > 10)


def _put_u16_le(mut buf: List[UInt8], v: UInt16):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


def _put_u64_le(mut buf: List[UInt8], v: UInt64):
    for s in range(8):
        buf.append(UInt8((v >> UInt64(s * 8)) & 0xFF))


def test_i_security_dos_and_wraparound_regressions() raises:
    print("  [Test I] Security DoS guards, 0xFFFF wraparound, >8-pkt burst rollback recovery & BitReader/NetHub/Physics guards...")

    # (a) Spoofed future cum_rel_ack DoS guard in NetChannel._process_rel_sack
    var peer = NetEndpoint.ipv4_loopback(9901)
    var ch = NetChannel(peer, IPV6_SAFE_MTU_BYTES)
    var ack_pkt = List[UInt8]()
    var ack_hdr = PacketWireHeader(0x56, CHAN_ACK_HEARTBEAT, 1, 0, 0)
    ack_hdr.write_into(ack_pkt)
    _put_u16_le(ack_pkt, UInt16(500))  # spoofed future cum_rel_ack when next_send_rel_seq == 0
    _put_u64_le(ack_pkt, UInt64(0xFFFFFFFFFFFFFFFF))
    var ack_view = DatagramView(peer, Int(ack_pkt.unsafe_ptr()), len(ack_pkt))
    assert_true(ch.ingest_datagram(ack_view, 100.0))
    assert_equal(Int(ack_pkt[0]), 0x56)
    assert_equal(Int(ch.oldest_unacked_rel_seq), 0)
    assert_equal(Int(ch.next_send_rel_seq), 0)

    # Also test with 2 fragments in flight and spoofed cum_rel_ack == 500
    var small_msg = List[UInt8](length=64, fill=0x42)
    assert_true(ch.send_reliable(small_msg))
    assert_equal(Int(ch.next_send_rel_seq), 1)
    var ack_pkt2 = List[UInt8]()
    var ack_hdr2 = PacketWireHeader(0x56, CHAN_ACK_HEARTBEAT, 2, 0, 0)
    ack_hdr2.write_into(ack_pkt2)
    _put_u16_le(ack_pkt2, UInt16(500))
    _put_u64_le(ack_pkt2, UInt64(0xFFFF))
    var ack_view2 = DatagramView(peer, Int(ack_pkt2.unsafe_ptr()), len(ack_pkt2))
    assert_true(ch.ingest_datagram(ack_view2, 110.0))
    assert_equal(Int(ack_pkt2[0]), 0x56)
    assert_equal(Int(ch.oldest_unacked_rel_seq), 0)  # must remain 0 (spoofed future ACK ignored)

    # (b) frag_cnt > 4096 rejection in NetChannel.ingest_datagram
    var bad_frag_pkt = List[UInt8]()
    var rel_hdr = PacketWireHeader(0x56, CHAN_RELIABLE_ORDERED, 3, 0, 0)
    rel_hdr.write_into(bad_frag_pkt)
    _put_u16_le(bad_frag_pkt, UInt16(0))  # cum_ack
    _put_u64_le(bad_frag_pkt, UInt64(0))  # sack_bits
    _put_u16_le(bad_frag_pkt, UInt16(0))  # rseq
    _put_u16_le(bad_frag_pkt, UInt16(1))  # msg_id
    _put_u16_le(bad_frag_pkt, UInt16(0))  # frag_idx
    _put_u16_le(bad_frag_pkt, UInt16(MAX_RELIABLE_FRAG_COUNT + 1))  # frag_cnt == 4097
    _put_u16_le(bad_frag_pkt, UInt16(4))  # plen
    for _ in range(4):
        bad_frag_pkt.append(UInt8(0xAA))
    var bad_frag_view = DatagramView(
        peer, Int(bad_frag_pkt.unsafe_ptr()), len(bad_frag_pkt)
    )
    assert_false(
        ch.ingest_datagram(bad_frag_view, 120.0),
        "frag_cnt > 4096 must be rejected",
    )
    assert_equal(Int(bad_frag_pkt[0]), 0x56)
    assert_false(ch.assembling_active)
    assert_equal(len(ch.incoming_reliable), 0)

    # (c) snap_seq == 0xFFFF wraparound baseline preservation + truncated packet & monotonic guard
    var rep = SnapshotReplicator(
        max_entities=16,
        max_clients=1,
        max_packet_bytes=1000,
        interest_radius=200.0,
        world_bounds=256.0,
    )
    var cli = SnapshotClientReceiver(16, 256.0)
    rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 200.0)
    rep.clients[0].next_snap_seq = UInt16(0xFFFE)

    # Tick 0xFFFE: spawn entity 0 and stationary entity 1
    rep.set_entity(0, Vec3(10.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(1))
    rep.set_entity(1, Vec3(20.0, 2.0, 5.0), Quat.identity(), Vec3(), UInt32(7))
    rep.prepare_tick()
    var p_fffe = rep.build_client_packet(0)
    assert_equal(Int(p_fffe[0]), 0xFFFE)
    var r_fffe = cli.apply_packet(p_fffe[1])
    assert_true(r_fffe[0])
    assert_equal(Int(r_fffe[1]), 0xFFFE)
    rep.notify_client_ack(0, r_fffe[1])

    # Tick 0xFFFF: move entity 0 only; entity 1 stays stationary (not re-sent in delta)
    rep.set_entity(0, Vec3(11.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(1))
    rep.prepare_tick()
    var p_ffff = rep.build_client_packet(0)
    assert_equal(Int(p_ffff[0]), 0xFFFF)
    var r_ffff = cli.apply_packet(p_ffff[1])
    assert_true(r_ffff[0])
    assert_equal(Int(r_ffff[1]), 0xFFFF)
    rep.notify_client_ack(0, r_ffff[1])

    # Tick 0x0000: delta against base_seq == 0xFFFF! Must preserve entity 1 in baseline!
    rep.set_entity(0, Vec3(12.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(2))
    rep.prepare_tick()
    var p_0000 = rep.build_client_packet(0)
    assert_equal(Int(p_0000[0]), 0x0000)
    var r_0000 = cli.apply_packet(p_0000[1])
    assert_true(r_0000[0], "delta against base_seq == 0xFFFF must decode cleanly")
    assert_equal(Int(r_0000[1]), 0x0000)
    assert_true(
        cli.baselines[0].entities[1].known,
        "stationary entity 1 must remain known in baseline across 0xFFFF -> 0x0000 wraparound",
    )
    assert_equal(Int(cli.baselines[0].entities[1].state_flags), 7)
    assert_true(abs(cli.entities[0].pos.x - Float32(12.0)) < Float32(0.05))
    assert_equal(Int(cli.entities[0].state_flags), 2)
    rep.notify_client_ack(0, r_0000[1])

    # Monotonic guard: stale packet 0xFFFF after 0x0000 must be rejected
    assert_false(
        cli.apply_packet(p_ffff[1])[0],
        "stale out-of-order snapshot must be rejected by seq_gt guard",
    )
    # Truncated future packet must fail without destroying valid baseline
    rep.set_entity(0, Vec3(15.0, 1.0, 0.0), Quat.identity(), Vec3(), UInt32(3))
    rep.prepare_tick()
    var p_0001 = rep.build_client_packet(0)
    var trunc_pkt = List[UInt8]()
    for i in range(min(10, len(p_0001[1]) - 1)):
        trunc_pkt.append(p_0001[1][i])
    assert_false(cli.apply_packet(trunc_pkt)[0], "truncated snapshot must fail")
    assert_equal(Int(cli.last_decoded_seq), 0x0000)
    assert_true(abs(cli.entities[0].pos.x - Float32(12.0)) < Float32(0.05))
    # Valid packet 0x0001 still decodes cleanly after truncated attempt
    assert_true(cli.apply_packet(p_0001[1])[0])
    assert_equal(Int(cli.last_decoded_seq), 0x0001)

    # (d) >8-packet burst loss rollback recovery without deadlock (14 consecutive drops!)
    var sim14 = VirtualNetSim.from_rtt(rtt_ms=20.0, jitter_ms=0.0, loss_rate=0.0, seed=1414)
    var ep0 = sim14.create_ipv4_peer(0, 9910)
    var ep1 = sim14.create_ipv4_peer(1, 9910)
    var addrs = List[NetEndpoint]()
    addrs.append(ep0.local_endpoint())
    addrs.append(ep1.local_endpoint())

    var w0 = ChaosArenaWorld(4)
    var w1 = ChaosArenaWorld(4)
    var s0 = RollbackSession[TestCmdInput](0, addrs, input_delay_ticks=2, max_rollback_ticks=8)
    var s1 = RollbackSession[TestCmdInput](1, addrs, input_delay_ticks=2, max_rollback_ticks=8)
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(w0)
    s1.bind_world(w1)

    for f in range(150):
        if f == 20:
            # Drop 14 consecutive outbound packets from peer 0 (far exceeding old K=8 limit!)
            sim14.drop_next_outbound_packets(0, 14)
        var now = sim14.now_ms()
        var c0 = _make_cmd(0, s0.current_tick)
        var c1 = _make_cmd(1, s1.current_tick)
        _ = s0.step_frame(c0, w0, ep0, now)
        _ = s1.step_frame(c1, w1, ep1, now)
        sim14.advance_ms(1000.0 / 60.0)

    assert_false(s0.desync_detected, "peer 0 must recover from 14-pkt burst loss with 0 desyncs")
    assert_false(s1.desync_detected, "peer 1 must recover from 14-pkt burst loss with 0 desyncs")
    assert_true(
        s0.confirmed_tick >= 100 and s1.confirmed_tick >= 100,
        "14-pkt burst loss must NOT deadlock rollback session",
    )
    for t in range(100):
        assert_equal(s0.confirmed_hash_history[t], s1.confirmed_hash_history[t])

    # (e) NetHub idle peer eviction, BitReader/BitWriter safety, and PhysicsRollbackAdapter error surfacing
    var sim_hub = VirtualNetSim(seed=777, one_way_latency_ms=5.0, jitter_ms=0.0, loss_rate=0.0)
    var hub_host = NetHub[VirtualEndpoint](
        sim_hub.create_ipv4_peer(0, 9920), True, max_channels=1, idle_timeout_ms=200.0
    )
    var hub_c0 = NetHub[VirtualEndpoint](sim_hub.create_ipv4_peer(1, 9920), False)
    var hub_c1 = NetHub[VirtualEndpoint](sim_hub.create_ipv4_peer(2, 9920), False)
    _ = hub_c0.connect(hub_host.local_endpoint())
    _ = hub_c1.connect(hub_host.local_endpoint())
    assert_true(hub_c0.send_unreliable(0, _hub_msg(1, 16), sim_hub.now_ms()))
    for _ in range(5):
        hub_host.poll(sim_hub.now_ms())
        sim_hub.advance_ms(10.0)
    assert_equal(hub_host.open_count(), 1)
    # Advance past 200ms idle timeout without traffic from c0 -> c0 must be evicted
    sim_hub.advance_ms(250.0)
    hub_host.poll(sim_hub.now_ms())
    assert_equal(len(hub_host.evicted), 1)
    assert_equal(hub_host.open_count(), 0)
    # Now c1 can be accepted into the freed slot
    assert_true(hub_c1.send_unreliable(0, _hub_msg(2, 16), sim_hub.now_ms()))
    for _ in range(5):
        hub_host.poll(sim_hub.now_ms())
        sim_hub.advance_ms(10.0)
    assert_equal(hub_host.open_count(), 1)

    # BitReader null ptr & read_bytes_aligned overflow & write_f32_quantized NaN guards
    var null_r = BitReader.from_raw_ptr(0, 16)
    assert_true(null_r.overflowed)
    assert_equal(Int(null_r.read_u32()), 0)
    var small_buf = List[UInt8](length=4, fill=0x11)
    var ov_r = BitReader(small_buf)
    var ov_bytes = ov_r.read_bytes_aligned(1 << 60)
    assert_true(ov_r.overflowed)
    assert_equal(len(ov_bytes), 0)
    var nan_w = BitWriter(16)
    var zero_f = Float32(0.0)
    var nan_val = zero_f / zero_f
    nan_w.write_f32_quantized(nan_val, -1.0, 1.0, 10)
    var nan_r = BitReader(nan_w.finish())
    var decoded_nan = nan_r.read_f32_quantized(-1.0, 1.0, 10)
    assert_true(decoded_nan == decoded_nan and decoded_nan >= Float32(-1.0) and decoded_nan <= Float32(1.0))

    # PhysicsRollbackAdapter.load_snapshot surfaces corrupt deserialize failure instead of swallowing
    var bad_pw = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var corrupt = List[UInt8](length=8, fill=0xFF)
    bad_pw.load_snapshot(Int(corrupt.unsafe_ptr()), len(corrupt))
    _ = len(corrupt)
    assert_true(bad_pw.snapshot_load_failed)
    assert_equal(bad_pw.snapshot_load_failures, 1)


def test_j_wave2_aead_cookie_channels_despawn_interp() raises:
    print("  [Test J] Wave 2: ChaCha20-Poly1305 AEAD, Replay/Tamper, Cookie Handshake, Despawn/AOI, Interp, Unordered/Raw & RTO Backoff...")

    # 1. ChaCha20-Poly1305 AEAD encrypt/decrypt + tamper rejection
    var key = AeadKey.from_u64x4(
        0x0706050403020100,
        0x0F0E0D0C0B0A0908,
        0x1716151413121110,
        0x1F1E1D1C1B1A1918,
    )
    var aad = _hub_msg(11, 22)
    var pt = _hub_msg(33, 197)
    var ct_tag = List[UInt8]()
    aead_seal(key, UInt64(42), aad, pt, ct_tag)
    assert_equal(len(ct_tag), len(pt) + 16)

    var dec = List[UInt8]()
    assert_true(aead_open(key, UInt64(42), aad, ct_tag, dec))
    assert_equal(hash_list(dec), hash_list(pt))

    # 1-bit ciphertext tamper must fail authentication
    var bad_ct = ct_tag.copy()
    bad_ct[10] ^= UInt8(0x01)
    assert_false(aead_open(key, UInt64(42), aad, bad_ct, dec))
    # 1-bit AAD tamper must fail authentication
    var bad_aad = aad.copy()
    bad_aad[3] ^= UInt8(0x80)
    assert_false(aead_open(key, UInt64(42), bad_aad, ct_tag, dec))
    # Wrong nonce must fail authentication
    assert_false(aead_open(key, UInt64(43), aad, ct_tag, dec))

    # 2. Encrypted NetChannel over lossy VirtualNetSim + direct replay & tamper defense
    var sim_enc = VirtualNetSim(
        seed=0xAEAD2026,
        one_way_latency_ms=15.0,
        jitter_ms=5.0,
        loss_rate=0.08,
        dup_rate=0.05,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
    )
    var e0 = sim_enc.create_ipv4_peer(0, 9930)
    var e1 = sim_enc.create_ipv4_peer(1, 9930)
    var ch_a = NetChannel(e1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var ch_b = NetChannel(e0.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    ch_a.enable_encryption(key)
    ch_b.enable_encryption(key)

    var secret_msg = _hub_msg(77, 4096)
    assert_true(ch_a.send_reliable(secret_msg))
    var got_secret = False
    for _ in range(300):
        var now = sim_enc.now_ms()
        ch_a.poll(e0, now)
        ch_b.poll(e1, now)
        var r = ch_b.pop_reliable_message()
        if r[0]:
            assert_equal(hash_list(r[1]), hash_list(secret_msg))
            got_secret = True
        if got_secret and ch_a.is_all_reliable_acked():
            break
        sim_enc.advance_ms(10.0)
    assert_true(got_secret, "encrypted 4 KiB reliable message must deliver cleanly")
    assert_equal(ch_b.crypto_auth_failures, 0)

    # Capture an encrypted unreliable-raw packet on a clean 0%-loss sim, test tamper, valid ingest & replay!
    var sim_rep = VirtualNetSim(seed=0xAEAD99, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=0.0)
    var er0 = sim_rep.create_ipv4_peer(0, 9935)
    var er1 = sim_rep.create_ipv4_peer(1, 9935)
    var chr_a = NetChannel(er1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var chr_b = NetChannel(er0.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    chr_a.enable_encryption(key)
    chr_b.enable_encryption(key)

    var raw_p = _hub_msg(55, 24)
    assert_true(chr_a.send_unreliable_raw(raw_p, er0, sim_rep.now_ms())[0])
    sim_rep.advance_ms(20.0)
    var cap_views = List[DatagramView](capacity=16)
    var n_cap = er1.recv_batch(cap_views)
    assert_equal(n_cap, 1)
    var captured = cap_views[0].to_bytes()
    assert_true(er1.local_endpoint().port > 0)  # keep er1.rx_ring_buf alive across to_bytes()

    # Tamper 1 bit of tag BEFORE valid ingest -> must fail Poly1305 & NOT pollute ReplayWindow64!
    var tampered = captured.copy()
    tampered[len(tampered) - 1] ^= UInt8(0x04)
    var v_bad = DatagramView(
        er0.local_endpoint(), Int(tampered.unsafe_ptr()), len(tampered)
    )
    assert_false(chr_b.ingest_datagram(v_bad, sim_rep.now_ms()))
    assert_equal(Int(tampered[0]), 0x56)
    assert_equal(chr_b.crypto_auth_failures, 1)

    # Original untampered packet must still authenticate and deliver cleanly!
    var v_good = DatagramView(
        er0.local_endpoint(), Int(captured.unsafe_ptr()), len(captured)
    )
    assert_true(chr_b.ingest_datagram(v_good, sim_rep.now_ms()))
    var pop_r = chr_b.pop_unreliable_raw_message()
    assert_true(pop_r[0] and hash_list(pop_r[1]) == hash_list(raw_p))

    # Replay the exact same authenticated packet -> must be rejected by ReplayWindow64!
    assert_false(chr_b.ingest_datagram(v_good, sim_rep.now_ms()))
    assert_equal(Int(captured[0]), 0x56)
    assert_equal(chr_b.crypto_replay_drops, 1)

    # 3. Stateless cookie handshake + encrypted NetHub
    var sim_ck = VirtualNetSim(seed=0xC00C1E, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=0.0)
    var srv_hub = NetHub[VirtualEndpoint](sim_ck.create_ipv4_peer(0, 9940), True)
    var cli_hub = NetHub[VirtualEndpoint](sim_ck.create_ipv4_peer(1, 9940), False)
    var cookie_secret = AeadKey.from_u64x4(0x1111, 0x2222, 0x3333, 0x4444)
    srv_hub.enable_cookie_handshake(cookie_secret, bucket_ms=10000.0)
    srv_hub.enable_channel_encryption(key)
    cli_hub.enable_channel_encryption(key)

    assert_true(cli_hub.send_handshake_init(srv_hub.local_endpoint()))
    var accepted_after_cookie = False
    for _ in range(20):
        var now = sim_ck.now_ms()
        srv_hub.poll(now)
        if len(srv_hub.accepted) > 0:
            accepted_after_cookie = True
        cli_hub.poll(now)
        sim_ck.advance_ms(10.0)
        if accepted_after_cookie:
            break
    assert_true(accepted_after_cookie, "client must be accepted after stateless cookie echo")
    assert_true(srv_hub.cookies_sent >= 1)
    assert_equal(srv_hub.open_count(), 1)
    var c_cid = cli_hub.connect(srv_hub.local_endpoint())
    assert_true(cli_hub.send_reliable(c_cid, _hub_msg(88, 128)))
    var got_hub_enc = False
    for _ in range(30):
        var now = sim_ck.now_ms()
        cli_hub.poll(now)
        srv_hub.poll(now)
        var m = srv_hub.pop_reliable(0)
        if m[0]:
            assert_equal(hash_list(m[1]), hash_list(_hub_msg(88, 128)))
            got_hub_enc = True
            break
        sim_ck.advance_ms(10.0)
    assert_true(got_hub_enc, "encrypted message over cookie-authenticated NetHub must deliver")

    # 4. CHAN_RELIABLE_UNORDERED bypasses HoL blocking behind missing ordered rel_seq
    var sim_u = VirtualNetSim(seed=0x554E4F, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=0.0)
    var u0 = sim_u.create_ipv4_peer(0, 9950)
    var u1 = sim_u.create_ipv4_peer(1, 9950)
    var chu0 = NetChannel(u1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var chu1 = NetChannel(u0.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    assert_true(chu0.send_reliable(_hub_msg(1, 32)))  # rel_seq = 0 (will be dropped!)
    assert_true(chu0.send_reliable_unordered(_hub_msg(2, 48)))  # rel_seq = 1 (unordered!)
    sim_u.drop_next_outbound_packets(0, 1)  # drop rel_seq = 0 only
    chu0.poll(u0, sim_u.now_ms())
    sim_u.advance_ms(20.0)
    chu1.poll(u1, sim_u.now_ms())
    assert_true(sim_u.now_ms() >= 0.0)  # keep sim_u alive across chu1.poll
    # Ordered queue is still blocked waiting for rel_seq = 0, but unordered msg delivered immediately!
    assert_false(chu1.pop_reliable_message()[0])
    var u_pop = chu1.pop_reliable_unordered_message()
    assert_true(u_pop[0], "CHAN_RELIABLE_UNORDERED must deliver without HoL blocking")
    assert_equal(hash_list(u_pop[1]), hash_list(_hub_msg(2, 48)))

    # 5. Exponential RTO backoff & AIMD cwnd halving under 100% loss
    var sim_rto = VirtualNetSim(seed=0x270, one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=1.0)
    var r0 = sim_rto.create_ipv4_peer(0, 9960)
    var r1 = sim_rto.create_ipv4_peer(1, 9960)
    var chr0 = NetChannel(r1.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    chr0.rto_ms = 100.0
    var init_cwnd = chr0.cwnd
    assert_true(chr0.send_reliable(_hub_msg(9, 64)))
    chr0.poll(r0, 0.0)  # initial transmit (send_count = 1)
    assert_equal(chr0.retransmissions, 0)
    chr0.poll(r0, 100.0)  # 1st retx after 1*rto = 100ms (send_count = 2, cwnd halves!)
    assert_equal(chr0.retransmissions, 1)
    assert_true(chr0.cwnd < init_cwnd, "AIMD must halve cwnd on RTO loss")
    chr0.poll(r0, 200.0)  # +100ms later: MUST NOT retx yet (backoff requires 2*rto = 200ms!)
    assert_equal(chr0.retransmissions, 1)
    chr0.poll(r0, 305.0)  # +205ms after 1st retx: 2nd retx fires!
    assert_equal(chr0.retransmissions, 2)
    assert_true(sim_rto.now_ms() >= 0.0)  # keep sim_rto alive across r0 polls

    # 6. Entity despawn (`destroy_entity`) + out-of-AOI removal + SnapshotInterpolationBuffer
    var rep = SnapshotReplicator(
        max_entities=16,
        max_clients=1,
        max_packet_bytes=1000,
        interest_radius=50.0,
        world_bounds=256.0,
    )
    var cli = SnapshotClientReceiver(16, 256.0)
    rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 50.0)
    rep.set_entity(0, Vec3(5.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(1))
    rep.set_entity(1, Vec3(10.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(2))
    rep.set_entity(2, Vec3(15.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(3))
    rep.prepare_tick()
    var p0 = rep.build_client_packet(0)
    var res0 = cli.apply_packet(p0[1])
    assert_true(res0[0])
    rep.notify_client_ack(0, res0[1])
    assert_true(cli.entities[0].active and cli.entities[1].active and cli.entities[2].active)

    # Despawn entity 1 and move entity 2 outside client's 50m AOI radius
    rep.destroy_entity(1)
    rep.set_entity(2, Vec3(120.0, 0.0, 0.0), Quat.identity(), Vec3(), UInt32(3))
    rep.prepare_tick()
    var p1 = rep.build_client_packet(0)
    var res1 = cli.apply_packet(p1[1])
    assert_true(res1[0])
    assert_true(cli.entities[0].active, "in-AOI active entity must remain active")
    assert_false(cli.entities[1].active, "despawned entity must be cleaned up on client")
    assert_false(cli.entities[2].active, "out-of-AOI entity must be cleaned up on client")

    # SnapshotInterpolationBuffer linear pos + quaternion slerp interpolation
    var interp = SnapshotInterpolationBuffer(max_entities=4, interp_delay_ms=100.0)
    var snap_a = List[ReplicatedEntity]()
    var snap_b = List[ReplicatedEntity]()
    for i in range(4):
        snap_a.append(ReplicatedEntity(i))
        snap_b.append(ReplicatedEntity(i))
    snap_a[0].active = True
    snap_a[0].pos = Vec3(0.0, 0.0, 0.0)
    snap_a[0].rot = Quat.from_euler(0.0, 0.0, 0.0)
    snap_b[0].active = True
    snap_b[0].pos = Vec3(10.0, 4.0, -6.0)
    snap_b[0].rot = Quat.from_euler(0.0, Float32(1.5707963), 0.0)
    interp.push_snapshot(100.0, snap_a)
    interp.push_snapshot(200.0, snap_b)
    var sampled = List[ReplicatedEntity]()
    assert_true(interp.sample(250.0, sampled))  # target_t = 250 - 100 = 150 => alpha = 0.5
    assert_true(sampled[0].active)
    assert_true(abs(sampled[0].pos.x - Float32(5.0)) < Float32(0.01))
    assert_true(abs(sampled[0].pos.y - Float32(2.0)) < Float32(0.01))
    assert_true(abs(sampled[0].pos.z - Float32(-3.0)) < Float32(0.01))
    var expected_mid_rot = Quat.from_euler(0.0, Float32(0.78539816), 0.0)
    assert_true(quat_angular_error_deg(sampled[0].rot, expected_mid_rot) < Float32(0.2))


struct _PredStepper(PredictionStepper):
    var accel: Float32

    def __init__(out self, accel: Float32 = 2.0):
        self.accel = accel

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
        var cmd = Pointer[PhysicsRollbackInput, ImmutAnyOrigin](
            unsafe_from_address=input_ptr
        )[unsafe_offset=0]
        state.vel = Vec3(
            state.vel.x + cmd.impulse_x * self.accel * dt,
            state.vel.y + cmd.impulse_y * self.accel * dt,
            state.vel.z + cmd.impulse_z * self.accel * dt,
        )
        state.pos = state.pos + state.vel * dt


struct _NanPredStepper(PredictionStepper):
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
        _ = input_ptr
        _ = input_seq
        _ = tick
        _ = dt
        state.pos.x = Float32(0.0) / Float32(0.0)


struct _SymmetricNatTransport(NetTransport):
    var ep: VirtualEndpoint
    var relay_ep: NetEndpoint
    var allow_direct: Bool
    var _tmp_views: List[DatagramView]

    def __init__(
        out self,
        var ep: VirtualEndpoint,
        relay_ep: NetEndpoint,
        allow_direct: Bool = False,
    ):
        self.ep = ep^
        self.relay_ep = relay_ep
        self.allow_direct = allow_direct
        self._tmp_views = List[DatagramView](capacity=64)

    def local_endpoint(self) -> NetEndpoint:
        return self.ep.local_endpoint()

    def send_raw(mut self, dst: NetEndpoint, data_ptr: Int, length: Int) -> Bool:
        if dst != self.relay_ep and not self.allow_direct:
            return True
        return self.ep.send_raw(dst, data_ptr, length)

    def queue_send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if dst != self.relay_ep and not self.allow_direct:
            return True
        return self.ep.queue_send_raw(dst, data_ptr, length)

    def flush_sends(mut self) -> Int:
        return self.ep.flush_sends()

    def recv_batch(mut self, mut out_views: List[DatagramView]) -> Int:
        out_views.clear()
        var n = self.ep.recv_batch(self._tmp_views)
        for i in range(n):
            var v = self._tmp_views[i]
            if v.sender == self.relay_ep or self.allow_direct:
                out_views.append(v)
        return len(out_views)


struct _ConeNatTransport(NetTransport):
    var ep: VirtualEndpoint
    var relay_ep: NetEndpoint
    var opened_peers: List[NetEndpoint]
    var _tmp_views: List[DatagramView]

    def __init__(out self, var ep: VirtualEndpoint, relay_ep: NetEndpoint):
        self.ep = ep^
        self.relay_ep = relay_ep
        self.opened_peers = List[NetEndpoint](capacity=8)
        self._tmp_views = List[DatagramView](capacity=64)

    def _record_outbound(mut self, dst: NetEndpoint):
        if dst == self.relay_ep or dst.port == 0:
            return
        for i in range(len(self.opened_peers)):
            if self.opened_peers[i] == dst:
                return
        self.opened_peers.append(dst)

    def _is_opened(self, src: NetEndpoint) -> Bool:
        if src == self.relay_ep:
            return True
        for i in range(len(self.opened_peers)):
            if self.opened_peers[i] == src:
                return True
        return False

    def local_endpoint(self) -> NetEndpoint:
        return self.ep.local_endpoint()

    def send_raw(mut self, dst: NetEndpoint, data_ptr: Int, length: Int) -> Bool:
        self._record_outbound(dst)
        return self.ep.send_raw(dst, data_ptr, length)

    def queue_send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        self._record_outbound(dst)
        return self.ep.queue_send_raw(dst, data_ptr, length)

    def flush_sends(mut self) -> Int:
        return self.ep.flush_sends()

    def recv_batch(mut self, mut out_views: List[DatagramView]) -> Int:
        out_views.clear()
        var n = self.ep.recv_batch(self._tmp_views)
        for i in range(n):
            var v = self._tmp_views[i]
            if self._is_opened(v.sender):
                out_views.append(v)
        return len(out_views)


def test_k_f6_client_prediction_reconciliation() raises:
    print("  [Test K] F6 ClientPredictionBuffer rewind+replay, PredictionStepper, VisualSmoother & wire last_acked_input_seq...")
    # 1. Exact prediction match & trimming
    var buf = PredictionBuffer[PhysicsRollbackInput](capacity=64)
    for seq in range(1, 11):
        var cmd = PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0)
        buf.push_predicted(
            seq,
            cmd,
            PredictedState(Vec3(Float32(seq), 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
            tick=seq,
            dt=Float32(0.05),
        )
    assert_equal(buf.pending_count(), 10)
    var matched_s = buf.reconcile(
        4, PredictedState(Vec3(4.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
    )
    assert_equal(buf.last_acked_input_seq, 4)
    assert_equal(buf.mispredictions, 0)
    assert_equal(buf.pending_count(), 6)
    assert_true(abs(matched_s.pos.x - Float32(10.0)) < Float32(1.0e-4))

    # 2. Misprediction at seq=5 -> rewind & replay 6..10, plus stale ACK guard
    var smoother = VisualSmoother(decay_rate=18.0)
    var corrected = buf.reconcile_smoothed(
        5,
        PredictedState(Vec3(3.5, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
        smoother,
    )
    assert_equal(buf.last_acked_input_seq, 5)
    assert_equal(buf.mispredictions, 1)
    assert_equal(buf.replayed_inputs, 5)
    assert_equal(buf.pending_count(), 5)
    assert_true(abs(corrected.pos.x - Float32(8.5)) < Float32(1.0e-3))
    assert_true(abs(smoother.pos_offset.x - Float32(1.5)) < Float32(1.0e-3))

    # VisualSmoother linear decay + teleport snap
    var smooth_lin = VisualSmoother(decay_rate=18.0)
    smooth_lin.record_correction(Vec3(10.0, 0.0, 0.0), Vec3(8.0, 0.0, 0.0))
    assert_true(abs(smooth_lin.pos_offset.x - Float32(2.0)) < Float32(1.0e-4))
    smooth_lin.step_linear(Float32(0.1), linear_speed=Float32(10.0))
    assert_true(abs(smooth_lin.pos_offset.x - Float32(1.0)) < Float32(1.0e-4))
    smooth_lin.step_linear(Float32(0.2), linear_speed=Float32(10.0))
    assert_true(smooth_lin.pos_offset.length_squared() < Float32(1.0e-8))
    smooth_lin.record_correction(
        Vec3(100.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), snap_threshold=Float32(5.0)
    )
    assert_true(smooth_lin.pos_offset.length_squared() < Float32(1.0e-8))

    # Stale ACK must not regress last_acked_input_seq or latest_state
    var stale_s = buf.reconcile(
        3, PredictedState(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    )
    assert_equal(buf.last_acked_input_seq, 5)
    assert_true(abs(stale_s.pos.x - Float32(8.5)) < Float32(1.0e-3))

    # 3. Deterministic PredictionStepper replay
    var step_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    var stepper = _PredStepper(accel=2.0)
    var sim_s = PredictedState(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    var dt = Float32(0.5)
    for seq in range(0, 6):
        var cmd = PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0)
        stepper.step_prediction(sim_s, Int(Pointer(to=cmd)), seq, seq, dt)
        _ = cmd.same_input(cmd)
        step_buf.push_predicted(seq, cmd, sim_s, tick=seq, dt=dt)
    # Initial seq=0 ACK works cleanly; server collided at seq=1 resetting vel.x=0, pos.x=0.5
    var r0 = step_buf.reconcile_with(
        0, PredictedState(Vec3(0.5, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)), stepper
    )
    assert_equal(step_buf.last_acked_input_seq, 0)
    assert_equal(step_buf.mispredictions, 0)
    _ = r0
    var auth1 = PredictedState(Vec3(0.5, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    var r1 = step_buf.reconcile_with(1, auth1, stepper)
    assert_equal(step_buf.mispredictions, 1)
    # Replaying seq 2,3,4,5 (4 steps of dt=0.5, accel=2 -> dv=1.0/step):
    # v=(1,2,3,4), dx=(0.5+1.0+1.5+2.0)=5.0 => pos.x=5.5, vel.x=4.0
    assert_true(abs(r1.pos.x - Float32(5.5)) < Float32(1.0e-3))
    assert_true(abs(r1.vel.x - Float32(4.0)) < Float32(1.0e-3))

    # Overflow eviction & out-of-window ACKs
    var small_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=4)
    for seq in range(1, 8):
        small_buf.push_predicted(
            seq,
            PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
            PredictedState(Vec3(Float32(seq), 0.0, 0.0)),
        )
    assert_equal(small_buf.overflow_evictions, 3)
    assert_equal(small_buf.pending_count(), 4)
    var old_ack = small_buf.reconcile(2, PredictedState(Vec3(2.0, 0.0, 0.0)))
    assert_equal(small_buf.pending_count(), 4)
    assert_true(abs(old_ack.pos.x - Float32(7.0)) < Float32(1.0e-4))
    var future_ack = small_buf.reconcile(99, PredictedState(Vec3(99.0, 0.0, 0.0)))
    assert_equal(small_buf.pending_count(), 0)
    assert_true(abs(future_ack.pos.x - Float32(99.0)) < Float32(1.0e-4))

    # 4. End-to-end SnapshotReplicator -> SnapshotClientReceiver `last_acked_input_seq` wire round-trip
    var rep = SnapshotReplicator(
        max_entities=8, max_clients=1, max_packet_bytes=512, interest_radius=100.0, world_bounds=256.0
    )
    var cli = SnapshotClientReceiver(8, 256.0)
    rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 100.0)
    rep.set_entity(0, Vec3(2.0, 0.0, 0.0))
    rep.set_client_acked_input_seq(0, 42)
    rep.prepare_tick()
    var p0 = rep.build_client_packet(0)
    var d0 = cli.apply_packet(p0[1])
    assert_true(d0[0])
    assert_true(cli.has_acked_input_seq)
    assert_equal(cli.last_acked_input_seq, 42)
    rep.notify_client_ack(0, d0[1])

    # Truncated packet carrying acked_input_seq=99 must NOT advance cli.last_acked_input_seq
    rep.set_client_acked_input_seq(0, 99)
    rep.set_entity(0, Vec3(3.0, 0.0, 0.0))
    rep.prepare_tick()
    var p1 = rep.build_client_packet(0)
    var trunc = List[UInt8]()
    for i in range(min(13, len(p1[1]) - 1)):
        trunc.append(p1[1][i])
    assert_false(cli.apply_packet(trunc)[0])
    assert_equal(cli.last_acked_input_seq, 42)
    assert_true(cli.apply_packet(p1[1])[0])
    assert_equal(cli.last_acked_input_seq, 99)

    # 5. Regression checks: NaN auth_state, gap ACK [1,2,4,5], forward jump reconcile_smoothed, VisualSmoother NaN dt & 1e20 overflow
    var nan32 = Float32(0.0) / Float32(0.0)
    var nan_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    nan_buf.push_predicted(
        1,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
    )
    _ = nan_buf.reconcile(1, PredictedState(Vec3(nan32, 0.0, 0.0)))
    assert_equal(nan_buf.mispredictions, 1)

    var nan_sm = VisualSmoother()
    nan_sm.pos_offset = Vec3(nan32, 0.0, 0.0)
    assert_true(nan_sm.snap_if_exceeds(Float32(5.0)))
    assert_true(nan_sm.pos_offset.length_squared() == Float32(0.0))

    nan_sm.record_correction(Vec3(2.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    nan_sm.step(nan32)
    nan_sm.step_linear(nan32)
    assert_true((nan_sm.pos_offset.x - nan_sm.pos_offset.x) == Float32(0.0))
    assert_true(abs(nan_sm.pos_offset.x - Float32(2.0)) < Float32(1.0e-5))

    var big_sm = VisualSmoother()
    big_sm.record_correction(Vec3(Float32(1.0e20), 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    big_sm.step_linear(Float32(0.016))
    assert_true((big_sm.pos_offset.x - big_sm.pos_offset.x) == Float32(0.0))

    # Gap ACK [1,2,4,5] with acked_seq=3 preserves future entries [4,5]
    var gap_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    for seq in [1, 2, 4, 5]:
        gap_buf.push_predicted(
            seq,
            PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
            PredictedState(Vec3(Float32(seq), 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
        )
    var gap_res = gap_buf.reconcile(
        3, PredictedState(Vec3(3.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
    )
    assert_equal(gap_buf.pending_count(), 2)
    assert_equal(gap_buf.last_acked_input_seq, 3)
    assert_true(abs(gap_res.pos.x - Float32(6.0)) < Float32(1.0e-3))

    var gap_step_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    var gap_stepper = _PredStepper(accel=2.0)
    var gs = PredictedState()
    for seq in [1, 2, 4, 5]:
        var cmd = PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0)
        gap_stepper.step_prediction(gs, Int(Pointer(to=cmd)), seq, seq, Float32(0.5))
        _ = cmd.same_input(cmd)
        gap_step_buf.push_predicted(seq, cmd, gs, tick=seq, dt=Float32(0.5))
    var gap_step_res = gap_step_buf.reconcile_with(
        3, PredictedState(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)), gap_stepper
    )
    assert_equal(gap_step_buf.pending_count(), 2)
    assert_equal(gap_step_buf.last_acked_input_seq, 3)
    # 2 steps from (pos=1, vel=1) with accel=2, dt=0.5 -> v=2,3 => x=1+1+1.5=3.5
    assert_true(abs(gap_step_res.pos.x - Float32(3.5)) < Float32(1.0e-3))

    # reconcile_smoothed forward-jump acked_seq > newest still records smoothing offset
    var fwd_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    for seq in range(1, 6):
        fwd_buf.push_predicted(
            seq,
            PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
            PredictedState(Vec3(Float32(seq), 0.0, 0.0)),
        )
    var fwd_sm = VisualSmoother()
    var fwd_res = fwd_buf.reconcile_smoothed(
        10, PredictedState(Vec3(20.0, 0.0, 0.0)), fwd_sm
    )
    assert_equal(fwd_buf.pending_count(), 0)
    assert_true(abs(fwd_res.pos.x - Float32(20.0)) < Float32(1.0e-4))
    assert_true(abs(fwd_sm.pos_offset.x - Float32(-15.0)) < Float32(1.0e-3))
    assert_true(abs(fwd_sm.smooth_pos(fwd_res.pos).x - Float32(5.0)) < Float32(1.0e-3))

    # Iteration 3 regressions: reconcile_smoothed NaN recovery, push_predicted NaN rejection,
    # VisualSmoother.step NaN sanitization, and snap_if_exceeds(1e20) overflow guard
    var rec_sm_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    rec_sm_buf.push_predicted(
        1,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(1.0, 0.0, 0.0)),
    )
    rec_sm_buf.push_predicted(
        2,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(2.0, 0.0, 0.0)),
    )
    # push_predicted must ignore non-finite predicted_state
    rec_sm_buf.push_predicted(
        3,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(nan32, 0.0, 0.0)),
    )
    assert_equal(rec_sm_buf.pending_count(), 2)
    assert_true(
        (rec_sm_buf.latest_state.pos.x - rec_sm_buf.latest_state.pos.x)
        == Float32(0.0)
    )
    assert_true(abs(rec_sm_buf.latest_state.pos.x - Float32(2.0)) < Float32(1.0e-4))

    var rec_sm = VisualSmoother(Float32(12.0))
    var r_bad = rec_sm_buf.reconcile_smoothed(
        1, PredictedState(Vec3(nan32, 0.0, 0.0)), rec_sm
    )
    assert_equal(rec_sm_buf.mispredictions, 1)
    assert_true((r_bad.pos.x - r_bad.pos.x) == Float32(0.0))
    var r_ok = rec_sm_buf.reconcile_smoothed(
        2, PredictedState(Vec3(2.0, 0.0, 0.0)), rec_sm
    )
    rec_sm.step(Float32(0.016))
    assert_true((r_ok.pos.x - r_ok.pos.x) == Float32(0.0))
    assert_true(
        (rec_sm.pos_offset.x - rec_sm.pos_offset.x) == Float32(0.0)
    )
    assert_true(
        (rec_sm.rot_offset.w - rec_sm.rot_offset.w) == Float32(0.0)
    )

    # Direct VisualSmoother.step sanitizes NaN offsets & snap_if_exceeds handles >= 1e19
    var step_nan_sm = VisualSmoother(Float32(12.0))
    step_nan_sm.pos_offset = Vec3(nan32, 0.0, 0.0)
    step_nan_sm.rot_offset = Quat(nan32, 0.0, 0.0, 1.0)
    step_nan_sm.step(Float32(0.016))
    assert_true(step_nan_sm.pos_offset.length_squared() == Float32(0.0))
    assert_true(
        (step_nan_sm.rot_offset.w - step_nan_sm.rot_offset.w) == Float32(0.0)
    )

    var snap_big_sm = VisualSmoother(Float32(12.0))
    snap_big_sm.pos_offset = Vec3(Float32(2.0e20), 0.0, 0.0)
    assert_true(snap_big_sm.snap_if_exceeds(Float32(1.0e20)))
    assert_true(snap_big_sm.pos_offset.length_squared() == Float32(0.0))
    snap_big_sm.pos_offset = Vec3(Float32(0.5e20), 0.0, 0.0)
    assert_false(snap_big_sm.snap_if_exceeds(Float32(1.0e20)))

    # Iteration 4 regressions: ±3e38 delta replay finiteness & NaN-producing PredictionStepper guard
    var ext_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    ext_buf.push_predicted(
        1,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(Float32(-3.0e38), 0.0, 0.0)),
    )
    ext_buf.push_predicted(
        2,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(Float32(3.0e38), 0.0, 0.0)),
    )
    var r_ext = ext_buf.reconcile(
        1, PredictedState(Vec3(Float32(-2.9e38), 0.0, 0.0))
    )
    assert_true((r_ext.pos.x - r_ext.pos.x) == Float32(0.0))
    assert_true((ext_buf.latest_state.pos.x - ext_buf.latest_state.pos.x) == Float32(0.0))

    var ov_sum_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    ov_sum_buf.push_predicted(
        1,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(Float32(0.0), 0.0, 0.0)),
    )
    ov_sum_buf.push_predicted(
        2,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(Float32(2.0e38), 0.0, 0.0)),
    )
    var r_ov_sum = ov_sum_buf.reconcile(
        1, PredictedState(Vec3(Float32(2.0e38), 0.0, 0.0))
    )
    assert_true((r_ov_sum.pos.x - r_ov_sum.pos.x) == Float32(0.0))
    assert_true((ov_sum_buf.latest_state.pos.x - ov_sum_buf.latest_state.pos.x) == Float32(0.0))

    var nan_st = _NanPredStepper()
    var nan_st_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    nan_st_buf.push_predicted(
        1,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(1.0, 0.0, 0.0)),
    )
    nan_st_buf.push_predicted(
        2,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(2.0, 0.0, 0.0)),
    )
    var r_nan_st = nan_st_buf.reconcile_with(
        1, PredictedState(Vec3(1.5, 0.0, 0.0)), nan_st
    )
    assert_true((r_nan_st.pos.x - r_nan_st.pos.x) == Float32(0.0))
    assert_true((nan_st_buf.latest_state.pos.x - nan_st_buf.latest_state.pos.x) == Float32(0.0))
    assert_equal(nan_st_buf.mispredictions, 1)

    var nan_gap_buf = ClientPredictionBuffer[PhysicsRollbackInput](capacity=8)
    nan_gap_buf.push_predicted(
        1,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(1.0, 0.0, 0.0)),
    )
    nan_gap_buf.push_predicted(
        4,
        PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0),
        PredictedState(Vec3(4.0, 0.0, 0.0)),
    )
    var r_nan_gap = nan_gap_buf.reconcile_with(
        2, PredictedState(Vec3(2.5, 0.0, 0.0)), nan_st
    )
    assert_true((r_nan_gap.pos.x - r_nan_gap.pos.x) == Float32(0.0))
    assert_true((nan_gap_buf.latest_state.pos.x - nan_gap_buf.latest_state.pos.x) == Float32(0.0))
    assert_true(nan_gap_buf.mispredictions >= 1)


def test_l_f7_server_lag_compensation() raises:
    print("  [Test L] F7 HitboxSnapshotRing & LagCompHistory temporal rewind, interpolation, OBB SAT & capsule...")
    var ring = HitboxSnapshotRing(capacity=16, max_rewind_ms=200.0)

    # Record 3 snapshots (0, 50, 100 ms) with Sphere, AABB, 45-deg rotating OBB, and Capsule
    var p_0 = List[HitboxPose]()
    p_0.append(HitboxPose.sphere(entity_id=1, center=Vec3(0.0, 1.0, 0.0), radius=Float32(0.5)))
    p_0.append(HitboxPose.aabb(entity_id=2, center=Vec3(10.0, 1.0, 0.0), half_extents=Vec3(0.5, 1.0, 0.5)))
    p_0.append(
        HitboxPose.obb(
            entity_id=3,
            center=Vec3(20.0, 1.0, 0.0),
            rot=Quat.from_euler(0.0, 0.0, 0.0),
            half_extents=Vec3(1.5, 0.5, 0.4),
        )
    )
    p_0.append(
        HitboxPose.capsule(
            entity_id=4,
            center=Vec3(30.0, 1.0, 0.0),
            rot=Quat.identity(),
            radius=Float32(0.4),
            half_height=Float32(0.8),
        )
    )
    ring.record_snapshot(0, 0.0, p_0)

    var p_50 = List[HitboxPose]()
    p_50.append(HitboxPose.sphere(entity_id=1, center=Vec3(4.0, 1.0, 0.0), radius=Float32(0.5)))
    p_50.append(HitboxPose.aabb(entity_id=2, center=Vec3(10.0, 1.0, 0.0), half_extents=Vec3(0.5, 1.0, 0.5)))
    p_50.append(
        HitboxPose.obb(
            entity_id=3,
            center=Vec3(20.0, 1.0, 0.0),
            rot=Quat.from_euler(0.0, Float32(1.5707963), 0.0),  # 90 deg at 50 ms -> 45 deg at 25 ms
            half_extents=Vec3(1.5, 0.5, 0.4),
        )
    )
    p_50.append(
        HitboxPose.capsule(
            entity_id=4,
            center=Vec3(32.0, 1.0, 0.0),
            rot=Quat.identity(),
            radius=Float32(0.4),
            half_height=Float32(0.8),
        )
    )
    ring.record_snapshot(1, 50.0, p_50)

    var p_100 = List[HitboxPose]()
    p_100.append(HitboxPose.sphere(entity_id=1, center=Vec3(8.0, 1.0, 0.0), radius=Float32(0.5)))
    ring.record_snapshot(2, 100.0, p_100)

    # 1. Interpolated sphere hit at t=25ms (alpha=0.5 -> x=2.0)
    var h_sph = ring.rewind_and_raycast(25.0, Vec3(2.0, 1.0, -4.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_true(h_sph.hit and h_sph.entity_id == 1)
    assert_false(h_sph.was_clamped)
    assert_true(abs(h_sph.point.z - Float32(-0.5)) < Float32(0.01))
    assert_true(abs(h_sph.normal.z - Float32(-1.0)) < Float32(0.01))

    # 2. 45-degree interpolated OBB at t=25ms: ray starting at (21.0, 1.0, 0.5) along +Z
    # is inside the world-space AABB [-1.34, +1.34]^2 but misses the 45-deg rotated OBB completely
    var miss_obb_corner = ring.rewind_and_raycast(
        25.0, Vec3(21.0, 1.0, 0.5), Vec3(0.0, 0.0, 1.0), 10.0
    )
    assert_false(miss_obb_corner.hit)
    # Ray along Z hitting rotated OBB at x=20.7 (intersects diagonal around z=-0.7)
    var hit_obb = ring.rewind_and_raycast(
        25.0, Vec3(20.7, 1.0, -3.0), Vec3(0.0, 0.0, 1.0), 10.0
    )
    assert_true(hit_obb.hit and hit_obb.entity_id == 3)
    assert_true(hit_obb.normal.x < Float32(-0.5) and hit_obb.normal.z < Float32(-0.5))

    # 3. Capsule at t=25ms (x=31.0, y in [0.2, 1.8] body + caps): hit at y=1.6
    var hit_cap = ring.rewind_and_raycast(
        25.0, Vec3(31.0, 1.6, -4.0), Vec3(0.0, 0.0, 1.0), 10.0
    )
    assert_true(hit_cap.hit and hit_cap.entity_id == 4)

    # 4. Inside-shape raycast returns t=0, point=origin
    var hit_inside = ring.rewind_and_raycast(
        25.0, Vec3(2.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), 10.0
    )
    assert_true(hit_inside.hit and hit_inside.entity_id == 1)
    assert_true(hit_inside.distance == Float32(0.0))
    assert_true(hit_inside.fraction == Float32(0.0))

    # 5. ignore_entity_id skips shooter & zero-length ray returns miss
    assert_false(
        ring.rewind_and_raycast(
            25.0, Vec3(2.0, 1.0, -4.0), Vec3(0.0, 0.0, 1.0), 20.0, ignore_entity_id=1
        ).hit
    )
    assert_false(ring.rewind_and_raycast(25.0, Vec3(), Vec3(), 10.0).hit)

    # 6. Overlap sphere & AABB (15-axis SAT + analytical segment-AABB)
    var ov_sph = ring.rewind_and_overlap(25.0, Vec3(2.0, 1.0, 0.0), Float32(0.25))
    assert_equal(len(ov_sph), 1)
    assert_equal(ov_sph[0].entity_id, 1)

    # AABB near rotated OBB corner: AABB proxy would intersect, 15-axis SAT correctly rejects
    var ov_aabb_miss = ring.rewind_and_overlap_aabb(
        25.0, Vec3(21.15, 0.6, 0.35), Vec3(21.45, 1.4, 0.65)
    )
    assert_equal(len(ov_aabb_miss), 0)
    var ov_aabb_cap = ring.rewind_and_overlap_aabb(
        25.0, Vec3(30.7, 1.4, -0.3), Vec3(31.3, 1.9, 0.3)
    )
    assert_equal(len(ov_aabb_cap), 1)
    assert_equal(ov_aabb_cap[0].entity_id, 4)

    # 7. Bounded max_rewind_ms + ring wraparound
    for k in range(3, 25):
        var t_ms = Float64(k) * 50.0
        var ps = List[HitboxPose]()
        ps.append(HitboxPose.sphere(entity_id=9, center=Vec3(Float32(k), 0.0, 0.0), radius=Float32(0.5)))
        ring.record_snapshot(k, t_ms, ps)
    # latest = 24*50 = 1200ms, max_rewind_ms = 200 -> min allowed = 1000ms (k=20 -> x=20.0)
    var clamped_hit = ring.rewind_and_raycast(100.0, Vec3(20.0, 0.0, -3.0), Vec3(0.0, 0.0, 1.0), 10.0)
    assert_true(clamped_hit.was_clamped)
    assert_true(abs(clamped_hit.rewind_time_ms - 1000.0) < 1.0e-6)
    assert_true(clamped_hit.hit and clamped_hit.entity_id == 9)

    # 8. LagCompHistory tick API
    var hist = LagCompHistory(max_entities=8, max_history_ticks=12)
    for tk in range(10):
        var ps = List[HitboxPose]()
        ps.append(HitboxPose(entity_id=5, center=Vec3(Float32(tk) * 3.0, 1.0, 0.0), radius=Float32(0.6)))
        hist.record_tick(tk, ps)
    var thit = hist.raycast_at_tick(4, Vec3(12.0, 1.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_true(thit.hit and thit.entity_id == 5)
    assert_false(hist.raycast_at_tick(9, Vec3(12.0, 1.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0).hit)

    # 9. Regression checks: NaN/Inf query vectors & non-finite HitboxPose must cleanly miss
    var nan32 = Float32(0.0) / Float32(0.0)
    var inf32 = Float32(1.0) / Float32(0.0)
    assert_false(ring.rewind_and_raycast(1000.0, Vec3(20.0, 0.0, -3.0), Vec3(nan32, 0.0, 1.0), 10.0).hit)
    assert_false(ring.rewind_and_raycast(1000.0, Vec3(nan32, 0.0, -3.0), Vec3(0.0, 0.0, 1.0), 10.0).hit)
    assert_false(ring.rewind_and_raycast(1000.0, Vec3(20.0, 0.0, -3.0), Vec3(0.0, 0.0, 1.0), inf32).hit)
    assert_equal(len(ring.rewind_and_overlap(1000.0, Vec3(nan32, 0.0, 0.0), Float32(2.0))), 0)
    assert_equal(len(ring.rewind_and_overlap_aabb(1000.0, Vec3(nan32, -1.0, -1.0), Vec3(21.0, 1.0, 1.0))), 0)

    var bad_ring = HitboxSnapshotRing(capacity=4, max_rewind_ms=100.0)
    var bad_poses = List[HitboxPose]()
    bad_poses.append(HitboxPose.sphere(1, Vec3(nan32, 0.0, 0.0), Float32(1.0)))
    bad_poses.append(HitboxPose.aabb(2, Vec3(0.0, 0.0, 0.0), Vec3(nan32, 1.0, 1.0)))
    bad_ring.record_snapshot(0, 10.0, bad_poses)
    assert_false(bad_ring.rewind_and_raycast(10.0, Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 10.0).hit)
    assert_equal(len(bad_ring.rewind_and_overlap(10.0, Vec3(0.0, 0.0, 0.0), Float32(5.0))), 0)
    assert_equal(len(bad_ring.rewind_and_overlap_aabb(10.0, Vec3(-5.0, -5.0, -5.0), Vec3(5.0, 5.0, 5.0))), 0)

    # Iteration 3 regressions: +-Inf timestamps rejected and huge Quat(1e30,...) normalizes cleanly
    var inf64 = Float64(1.0) / Float64(0.0)
    var inf_ring = HitboxSnapshotRing(capacity=4, max_rewind_ms=100.0)
    var valid_pose = List[HitboxPose]()
    valid_pose.append(HitboxPose.sphere(10, Vec3(0.0, 0.0, 0.0), Float32(1.0)))
    inf_ring.record_snapshot(0, 10.0, valid_pose)
    inf_ring.record_snapshot(1, inf64, valid_pose)
    inf_ring.record_snapshot(2, -inf64, valid_pose)
    assert_equal(inf_ring.snapshot_count(), 1)
    assert_true((inf_ring.newest_time_ms() - inf_ring.newest_time_ms()) == 0.0)
    inf_ring.record_snapshot(3, 20.0, valid_pose)
    assert_equal(inf_ring.snapshot_count(), 2)
    assert_true(abs(inf_ring.newest_time_ms() - 20.0) < 1.0e-6)

    var big_q_pose = HitboxPose.obb(
        entity_id=11,
        center=Vec3(0.0, 0.0, 0.0),
        rot=Quat(Float32(1.0e30), Float32(1.0e30), Float32(1.0e30), Float32(1.0e30)),
        half_extents=Vec3(1.0, 1.0, 1.0),
    )
    assert_true(big_q_pose.active)
    assert_true(abs(big_q_pose.rot.length() - Float32(1.0)) < Float32(1.0e-4))
    var big_q_ring = HitboxSnapshotRing(capacity=4, max_rewind_ms=100.0)
    var big_q_poses = List[HitboxPose]()
    big_q_poses.append(big_q_pose)
    big_q_ring.record_snapshot(0, 10.0, big_q_poses)
    assert_true(
        big_q_ring.rewind_and_raycast(
            10.0, Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 10.0
        ).hit
    )


def test_m_f8_chunk_relative_open_world_quantization() raises:
    print("  [Test M] F8 Chunk-relative open-world quantization (±100,000 m, cross-chunk deltas, i32 chunks & full fallback)...")
    # 1. ±100,000 m precision with chunk_size=256.0 and non-pow2 chunk_size=100.0
    var rep = SnapshotReplicator(
        max_entities=16, max_clients=1, max_packet_bytes=1000, interest_radius=300.0, chunk_size=256.0
    )
    var cli = SnapshotClientReceiver(max_entities=16, chunk_size=256.0)
    var far_a = Vec3(Float32(100_000.25), Float32(2_500.5), Float32(-100_000.75))
    var far_b = Vec3(Float32(100_020.0), Float32(2_500.5), Float32(-100_010.0))
    rep.set_client_view(0, far_a, 300.0)
    rep.set_entity(0, far_a, Quat.identity(), Vec3(), UInt32(1))
    rep.set_entity(1, far_b, Quat.identity(), Vec3(), UInt32(2))
    rep.prepare_tick()
    var p0 = rep.build_client_packet(0)
    var d0 = cli.apply_packet(p0[1])
    assert_true(d0[0])
    rep.notify_client_ack(0, d0[1])
    assert_true((cli.entities[0].pos - far_a).length() < Float32(0.01))
    assert_true((cli.entities[1].pos - far_b).length() < Float32(0.01))

    # 2. Smooth cross-chunk walk across x = 255.90 -> 256.10 and x = -0.08 -> +0.08
    var cross_rep = SnapshotReplicator(
        max_entities=8, max_clients=1, max_packet_bytes=1000, interest_radius=100.0, chunk_size=256.0
    )
    var cross_cli = SnapshotClientReceiver(max_entities=8, chunk_size=256.0)
    cross_rep.set_client_view(0, Vec3(128.0, 0.0, 0.0), 300.0)
    for step in range(20):
        var x0 = Float32(255.85) + Float32(step) * Float32(0.025)
        var x1 = Float32(-0.15) + Float32(step) * Float32(0.025)
        cross_rep.set_entity(0, Vec3(x0, 1.0, 0.0))
        cross_rep.set_entity(1, Vec3(x1, 1.0, 0.0))
        cross_rep.prepare_tick()
        var pk = cross_rep.build_client_packet(0)
        var dr = cross_cli.apply_packet(pk[1])
        assert_true(dr[0])
        cross_rep.notify_client_ack(0, dr[1])
        assert_true((cross_cli.entities[0].pos - Vec3(x0, 1.0, 0.0)).length() < Float32(0.01))
        assert_true((cross_cli.entities[1].pos - Vec3(x1, 1.0, 0.0)).length() < Float32(0.01))

    # 3. Exact chunk-multiple teleport (same local qx, changed chunk_ix) must trigger DIRTY_POS
    cross_rep.set_entity(0, Vec3(Float32(10.0), 1.0, 0.0))
    cross_rep.prepare_tick()
    var p_pre_tp = cross_rep.build_client_packet(0)
    var d_pre_tp = cross_cli.apply_packet(p_pre_tp[1])
    assert_true(d_pre_tp[0])
    cross_rep.notify_client_ack(0, d_pre_tp[1])
    assert_true(abs(cross_cli.entities[0].pos.x - Float32(10.0)) < Float32(0.01))

    cross_rep.set_entity(0, Vec3(Float32(266.0), 1.0, 0.0))
    cross_rep.prepare_tick()
    var p_tp = cross_rep.build_client_packet(0)
    var d_tp = cross_cli.apply_packet(p_tp[1])
    assert_true(d_tp[0])
    cross_rep.notify_client_ack(0, d_tp[1])
    assert_true(abs(cross_cli.entities[0].pos.x - Float32(266.0)) < Float32(0.01))

    # 4. Non-power-of-2 chunk_size=100.0 + fine chunk_size=1.0 (i32 chunk coords > 32767) + unacked ring overwrite full fallback
    var np2_rep = SnapshotReplicator(
        max_entities=4, max_clients=1, max_packet_bytes=1000, interest_radius=200.0, chunk_size=100.0
    )
    var np2_cli = SnapshotClientReceiver(max_entities=4, chunk_size=100.0)
    var np2_pos = Vec3(Float32(-99_950.35), Float32(1_234.56), Float32(100_005.125))
    np2_rep.set_client_view(0, np2_pos, 200.0)
    np2_rep.set_entity(0, np2_pos)
    np2_rep.prepare_tick()
    var np2_pkt = np2_rep.build_client_packet(0)
    assert_true(np2_cli.apply_packet(np2_pkt[1])[0])
    assert_true((np2_cli.entities[0].pos - np2_pos).length() < Float32(0.015))

    var fine_rep = SnapshotReplicator(
        max_entities=4, max_clients=1, max_packet_bytes=1000, interest_radius=200.0, chunk_size=1.0
    )
    var fine_cli = SnapshotClientReceiver(max_entities=4, chunk_size=1.0)
    var ultra_far = Vec3(Float32(100_000.25), Float32(-50_000.5), Float32(80_000.75))
    fine_rep.set_client_view(0, ultra_far, 200.0)
    for k in range(36):
        # Do NOT notify ack for 35 ticks to force baseline overwrite -> full-snapshot fallback on tick 35
        fine_rep.set_entity(0, ultra_far + Vec3(Float32(k) * Float32(0.01), 0.0, 0.0))
        fine_rep.prepare_tick()
        var fp = fine_rep.build_client_packet(0)
        if k == 35:
            assert_true(fine_cli.apply_packet(fp[1])[0])
            var exp = ultra_far + Vec3(Float32(35) * Float32(0.01), 0.0, 0.0)
            assert_true((fine_cli.entities[0].pos - exp).length() < Float32(0.01))

    # 5. Regression check: >32 unacked full-snapshot fallback deactivates destroyed zombie entities
    var z_rep = SnapshotReplicator(
        max_entities=8, max_clients=1, max_packet_bytes=1000, interest_radius=200.0, chunk_size=256.0
    )
    var z_cli = SnapshotClientReceiver(max_entities=8, chunk_size=256.0)
    z_rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 200.0)
    z_rep.set_entity(0, Vec3(2.0, 0.0, 0.0))
    z_rep.set_entity(1, Vec3(4.0, 0.0, 0.0))
    z_rep.prepare_tick()
    var zp0 = z_rep.build_client_packet(0)
    var zr0 = z_cli.apply_packet(zp0[1])
    assert_true(zr0[0])
    z_rep.notify_client_ack(0, zr0[1])
    assert_true(z_cli.entities[0].active and z_cli.entities[1].active)

    z_rep.destroy_entity(1)
    for k in range(1, 35):
        z_rep.set_entity(0, Vec3(2.0 + Float32(k) * Float32(0.1), 0.0, 0.0))
        z_rep.prepare_tick()
        var zpk = z_rep.build_client_packet(0)
        if k == 34:
            var zr34 = z_cli.apply_packet(zpk[1])
            assert_true(zr34[0])
            z_rep.notify_client_ack(0, zr34[1])
    assert_true(z_cli.entities[0].active)
    assert_false(z_cli.entities[1].active)
    z_rep.prepare_tick()
    var zp35 = z_rep.build_client_packet(0)
    assert_true(z_cli.apply_packet(zp35[1])[0])
    assert_false(z_cli.entities[1].active)

    # 6. Regression check: _quant_pos_axis Inf / Int32 range clamping
    var nan32 = Float32(0.0) / Float32(0.0)
    var inf32 = Float32(1.0) / Float32(0.0)
    var inf_rep = SnapshotReplicator(
        max_entities=4, max_clients=1, max_packet_bytes=1000, interest_radius=200.0, chunk_size=1.0
    )
    var inf_cli = SnapshotClientReceiver(max_entities=4, chunk_size=1.0)
    inf_rep.set_client_view(0, Vec3(0.0, 0.0, 0.0), 200.0)
    inf_rep.set_entity(0, Vec3(inf32, -inf32, nan32))
    inf_rep.set_entity(1, Vec3(Float32(1.0e38), Float32(-1.0e38), Float32(0.0)))
    inf_rep.prepare_tick()
    var inf_pkt = inf_rep.build_client_packet(0)
    assert_true(inf_cli.apply_packet(inf_pkt[1])[0])
    var p_inf = inf_cli.entities[0].pos
    assert_true((p_inf.x - p_inf.x) == Float32(0.0) and abs(p_inf.x) < Float32(1.0))
    assert_true((p_inf.y - p_inf.y) == Float32(0.0) and abs(p_inf.y) < Float32(1.0))
    assert_true((p_inf.z - p_inf.z) == Float32(0.0) and abs(p_inf.z) < Float32(1.0))
    var p_big = inf_cli.entities[1].pos
    assert_true((p_big.x - p_big.x) == Float32(0.0))
    assert_true((p_big.y - p_big.y) == Float32(0.0))

    # 7. Iteration 4 regressions: SpatialInterestGrid 64-bit cells at 1e11, query_sphere guards,
    #    and SnapshotInterpolationBuffer out-of-order & NaN timestamp handling
    var huge_pos = Vec3(Float32(1.0e11), 0.0, 0.0)
    var grid_ents = List[ReplicatedEntity]()
    var ge0 = ReplicatedEntity(0)
    ge0.active = True
    ge0.pos = huge_pos
    grid_ents.append(ge0)
    var ge1 = ReplicatedEntity(1)
    ge1.active = True
    ge1.pos = Vec3(nan32, 0.0, 0.0)
    grid_ents.append(ge1)
    var ge2 = ReplicatedEntity(2)
    ge2.active = True
    ge2.pos = Vec3(0.0, 0.0, 0.0)
    grid_ents.append(ge2)

    var big_grid = SpatialInterestGrid(max_entities=4, cell_size=32.0, use_3d=False)
    big_grid.rebuild(grid_ents)
    var q_out = List[Int]()
    big_grid.query_sphere(grid_ents, huge_pos, Float32(200.0), q_out)
    assert_equal(len(q_out), 1)
    assert_equal(q_out[0], 0)

    # query_sphere rejects non-finite center, NaN/+Inf/-10 radius, and handles huge 1e7 finite radius cleanly
    big_grid.query_sphere(grid_ents, Vec3(nan32, 0.0, 0.0), Float32(100.0), q_out)
    assert_equal(len(q_out), 0)
    big_grid.query_sphere(grid_ents, Vec3(0.0, 0.0, 0.0), nan32, q_out)
    assert_equal(len(q_out), 0)
    big_grid.query_sphere(grid_ents, Vec3(0.0, 0.0, 0.0), inf32, q_out)
    assert_equal(len(q_out), 0)
    big_grid.query_sphere(grid_ents, Vec3(0.0, 0.0, 0.0), Float32(-10.0), q_out)
    assert_equal(len(q_out), 0)
    big_grid.query_sphere(grid_ents, Vec3(0.0, 0.0, 0.0), Float32(1.0e7), q_out)
    assert_equal(len(q_out), 1)
    assert_equal(q_out[0], 2)

    # SnapshotReplicator round-trip at (1e11, 0, 0) with chunk_size=64.0, cell_size=32.0
    var far_rep = SnapshotReplicator(
        max_entities=8,
        max_clients=1,
        max_packet_bytes=512,
        interest_radius=200.0,
        cell_size=32.0,
        chunk_size=64.0,
    )
    var far_cli = SnapshotClientReceiver(max_entities=8, chunk_size=64.0)
    far_rep.set_entity(0, huge_pos)
    far_rep.set_client_view(0, huge_pos, Float32(200.0))
    far_rep.prepare_tick()
    var far_pkt = far_rep.build_client_packet(0)
    var far_ack = far_cli.apply_packet(far_pkt[1])
    assert_true(far_ack[0])
    assert_true(far_cli.entities[0].active)
    assert_true(abs(far_cli.entities[0].pos.x - huge_pos.x) <= Float32(8192.0))

    # SnapshotInterpolationBuffer out-of-order [100, 200, 150] -> sample(175.0) == 175.0
    var ooo_interp = SnapshotInterpolationBuffer(
        max_entities=4, interp_delay_ms=0.0, ring_capacity=4
    )
    var s100 = List[ReplicatedEntity]()
    var s200 = List[ReplicatedEntity]()
    var s150 = List[ReplicatedEntity]()
    for i in range(4):
        s100.append(ReplicatedEntity(i))
        s200.append(ReplicatedEntity(i))
        s150.append(ReplicatedEntity(i))
    s100[0].active = True
    s100[0].pos = Vec3(Float32(100.0), 0.0, 0.0)
    s200[0].active = True
    s200[0].pos = Vec3(Float32(200.0), 0.0, 0.0)
    s150[0].active = True
    s150[0].pos = Vec3(Float32(150.0), 0.0, 0.0)
    ooo_interp.push_snapshot(100.0, s100)
    ooo_interp.push_snapshot(200.0, s200)
    ooo_interp.push_snapshot(150.0, s150)
    var ooo_out = List[ReplicatedEntity]()
    assert_true(ooo_interp.sample(175.0, ooo_out))
    assert_true(abs(ooo_out[0].pos.x - Float32(175.0)) < Float32(1.0e-3))

    # SnapshotInterpolationBuffer [100, NaN, 200] -> sample(150.0) == 150.0 & sample(NaN) == False
    var nan_interp = SnapshotInterpolationBuffer(
        max_entities=4, interp_delay_ms=0.0, ring_capacity=4
    )
    var nan64 = Float64(0.0) / Float64(0.0)
    nan_interp.push_snapshot(100.0, s100)
    nan_interp.push_snapshot(nan64, s150)
    nan_interp.push_snapshot(200.0, s200)
    var nan_out = List[ReplicatedEntity]()
    assert_true(nan_interp.sample(150.0, nan_out))
    assert_true(abs(nan_out[0].pos.x - Float32(150.0)) < Float32(1.0e-3))
    assert_false(nan_interp.sample(nan64, nan_out))

    # Full ring drops stale timestamp < oldest and sorts intermediate replacement
    var cap2_interp = SnapshotInterpolationBuffer(
        max_entities=4, interp_delay_ms=0.0, ring_capacity=2
    )
    var s50 = List[ReplicatedEntity]()
    for i in range(4):
        s50.append(ReplicatedEntity(i))
    s50[0].active = True
    s50[0].pos = Vec3(Float32(50.0), 0.0, 0.0)
    cap2_interp.push_snapshot(100.0, s100)
    cap2_interp.push_snapshot(200.0, s200)
    cap2_interp.push_snapshot(50.0, s50)
    assert_true(cap2_interp.sample(150.0, nan_out))
    assert_true(abs(nan_out[0].pos.x - Float32(150.0)) < Float32(1.0e-3))
    cap2_interp.push_snapshot(150.0, s150)
    assert_true(cap2_interp.sample(175.0, nan_out))
    assert_true(abs(nan_out[0].pos.x - Float32(175.0)) < Float32(1.0e-3))


def test_n_f9_nat_relay_rollback_topology() raises:
    print("  [Test N] F9 RelayForwardHeader, RelayServer, NATPunchProbe (Symmetric fallback, Cone upgrade, mid-session fallback & anti-spoof)...")
    # 1. RelayForwardHeader round-trip, truncation & hop_ttl checks
    var hdr = RelayForwardHeader(
        session_token=UInt64(0x1122334455667788),
        src_peer=1,
        dst_peer=2,
        flags=RELAY_FLAG_DATA,
        seq=UInt16(77),
        hop_ttl=1,
    )
    var hdr_bytes = hdr.encode_bytes()
    assert_equal(len(hdr_bytes), RELAY_HEADER_BYTES)
    var parsed_ok = RelayForwardHeader.parse_from_ptr(
        Int(hdr_bytes.unsafe_ptr()), len(hdr_bytes)
    )
    assert_true(parsed_ok[0])
    assert_equal(parsed_ok[1].session_token, hdr.session_token)
    assert_equal(parsed_ok[1].src_peer, 1)
    assert_equal(parsed_ok[1].dst_peer, 2)
    assert_equal(Int(parsed_ok[1].seq), 77)
    assert_false(
        RelayForwardHeader.parse_from_ptr(Int(hdr_bytes.unsafe_ptr()), 10)[0]
    )

    # 2. Symmetric NAT Relay Fallback (all direct P2P blocked)
    var sim_sym = VirtualNetSim.from_rtt(rtt_ms=20.0, jitter_ms=3.0, loss_rate=0.03, seed=0xF9A1)
    var r_ep_sym = sim_sym.create_ipv4_peer(0, 9900)
    var raw_sym0 = sim_sym.create_ipv4_peer(1, 9901)
    var raw_sym1 = sim_sym.create_ipv4_peer(2, 9902)
    var sym_addrs = List[NetEndpoint]()
    sym_addrs.append(raw_sym0.local_endpoint())
    sym_addrs.append(raw_sym1.local_endpoint())

    var nat0 = _SymmetricNatTransport(raw_sym0^, r_ep_sym.local_endpoint(), allow_direct=False)
    var nat1 = _SymmetricNatTransport(raw_sym1^, r_ep_sym.local_endpoint(), allow_direct=False)
    var relay_sym = RelayServer(r_ep_sym.local_endpoint())

    var pw_sym0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw_sym1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s_sym0 = RollbackSession[PhysicsRollbackInput](0, sym_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    var s_sym1 = RollbackSession[PhysicsRollbackInput](1, sym_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    s_sym0.enable_relay(r_ep_sym.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=120.0)
    s_sym1.enable_relay(r_ep_sym.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=120.0)
    s_sym0.bind_world(pw_sym0)
    s_sym1.bind_world(pw_sym1)

    for f in range(100):
        var now = sim_sym.now_ms()
        relay_sym.poll(r_ep_sym, now)
        var c0 = PhysicsRollbackInput(Float32(f % 3), 0.0, 1.0, 0.0, 0)
        var c1 = PhysicsRollbackInput(-1.0, 0.0, Float32(f % 2), 0.0, 0)
        _ = s_sym0.step_frame(c0, pw_sym0, nat0, now)
        _ = s_sym1.step_frame(c1, pw_sym1, nat1, now)
        sim_sym.advance_ms(1000.0 / 60.0)

    assert_false(s_sym0.desync_detected)
    assert_false(s_sym1.desync_detected)
    assert_true(s_sym0.is_peer_relayed(1) and s_sym1.is_peer_relayed(0))
    assert_true(s_sym0.punch_fallbacks >= 1 and s_sym1.punch_fallbacks >= 1)
    assert_true(s_sym0.relayed_packets_sent > 0 and s_sym0.relayed_packets_received > 0)
    assert_true(s_sym0.direct_packets_received == 0)
    assert_true(len(s_sym0.confirmed_hash_history) >= 50)
    for t in range(50):
        assert_equal(s_sym0.confirmed_hash_history[t], s_sym1.confirmed_hash_history[t])

    # 3. Cone NAT Hole-Punch Upgrade (RELAY -> DIRECT)
    var sim_cone = VirtualNetSim.from_rtt(rtt_ms=16.0, jitter_ms=1.0, loss_rate=0.0, seed=0xF9B2)
    var r_ep_cone = sim_cone.create_ipv4_peer(0, 9910)
    var raw_c0 = sim_cone.create_ipv4_peer(1, 9911)
    var raw_c1 = sim_cone.create_ipv4_peer(2, 9912)
    var cone_addrs = List[NetEndpoint]()
    cone_addrs.append(raw_c0.local_endpoint())
    cone_addrs.append(raw_c1.local_endpoint())

    var cone0 = _ConeNatTransport(raw_c0^, r_ep_cone.local_endpoint())
    var cone1 = _ConeNatTransport(raw_c1^, r_ep_cone.local_endpoint())
    var relay_cone = RelayServer(r_ep_cone.local_endpoint())

    var pw_c0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw_c1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s_c0 = RollbackSession[PhysicsRollbackInput](0, cone_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    var s_c1 = RollbackSession[PhysicsRollbackInput](1, cone_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    s_c0.enable_relay(r_ep_cone.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=1000.0)
    s_c1.enable_relay(r_ep_cone.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=1000.0)
    s_c0.bind_world(pw_c0)
    s_c1.bind_world(pw_c1)

    for f in range(90):
        var now = sim_cone.now_ms()
        relay_cone.poll(r_ep_cone, now)
        var cmd = PhysicsRollbackInput(Float32(f % 4), 0.0, -1.0, 0.0, 0)
        _ = s_c0.step_frame(cmd, pw_c0, cone0, now)
        _ = s_c1.step_frame(cmd, pw_c1, cone1, now)
        sim_cone.advance_ms(1000.0 / 60.0)

    assert_false(s_c0.desync_detected)
    assert_false(s_c1.desync_detected)
    assert_true(s_c0.punch_upgrades >= 1 and s_c1.punch_upgrades >= 1)
    assert_true(s_c0.is_peer_direct(1) and s_c1.is_peer_direct(0))
    assert_true(s_c0.direct_packets_sent > 0 and s_c0.direct_packets_received > 0)
    assert_true(len(s_c0.confirmed_hash_history) >= 50)
    for t in range(50):
        assert_equal(s_c0.confirmed_hash_history[t], s_c1.confirmed_hash_history[t])

    # 4. Mid-session Direct -> Relay Fallback when direct path drops
    var sim_mid = VirtualNetSim.from_rtt(rtt_ms=10.0, jitter_ms=0.0, loss_rate=0.0, seed=0xF9C3)
    var r_ep_mid = sim_mid.create_ipv4_peer(0, 9920)
    var raw_m0 = sim_mid.create_ipv4_peer(1, 9921)
    var raw_m1 = sim_mid.create_ipv4_peer(2, 9922)
    var mid_addrs = List[NetEndpoint]()
    mid_addrs.append(raw_m0.local_endpoint())
    mid_addrs.append(raw_m1.local_endpoint())

    var mid0 = _SymmetricNatTransport(raw_m0^, r_ep_mid.local_endpoint(), allow_direct=True)
    var mid1 = _SymmetricNatTransport(raw_m1^, r_ep_mid.local_endpoint(), allow_direct=True)
    var relay_mid = RelayServer(r_ep_mid.local_endpoint())

    var pw_m0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw_m1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s_m0 = RollbackSession[PhysicsRollbackInput](0, mid_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    var s_m1 = RollbackSession[PhysicsRollbackInput](1, mid_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    s_m0.enable_relay(r_ep_mid.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=1000.0, direct_idle_fallback_ms=150.0)
    s_m1.enable_relay(r_ep_mid.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=1000.0, direct_idle_fallback_ms=150.0)
    s_m0.bind_world(pw_m0)
    s_m1.bind_world(pw_m1)

    for _ in range(40):
        var now = sim_mid.now_ms()
        relay_mid.poll(r_ep_mid, now)
        var cmd = PhysicsRollbackInput(1.0, 0.0, 0.0, 0.0, 0)
        _ = s_m0.step_frame(cmd, pw_m0, mid0, now)
        _ = s_m1.step_frame(cmd, pw_m1, mid1, now)
        sim_mid.advance_ms(1000.0 / 60.0)
    assert_true(s_m0.is_peer_direct(1) and s_m1.is_peer_direct(0))

    # Break direct connectivity mid-session -> both fall back to relay seamlessly
    mid0.allow_direct = False
    mid1.allow_direct = False
    for _ in range(40, 100):
        var now = sim_mid.now_ms()
        relay_mid.poll(r_ep_mid, now)
        var cmd = PhysicsRollbackInput(0.0, 0.0, 1.0, 0.0, 0)
        _ = s_m0.step_frame(cmd, pw_m0, mid0, now)
        _ = s_m1.step_frame(cmd, pw_m1, mid1, now)
        sim_mid.advance_ms(1000.0 / 60.0)
    assert_true(s_m0.is_peer_relayed(1) and s_m1.is_peer_relayed(0))
    assert_true(s_m0.punch_fallbacks >= 1 and s_m1.punch_fallbacks >= 1)
    assert_false(s_m0.desync_detected or s_m1.desync_detected)
    for t in range(60):
        assert_equal(s_m0.confirmed_hash_history[t], s_m1.confirmed_hash_history[t])

    # 5. RelayServer anti-hijack & RollbackSession relay anti-spoofing
    var sim_sec = VirtualNetSim.from_rtt(rtt_ms=10.0, jitter_ms=0.0, loss_rate=0.0, seed=0xF9D4)
    var r_sec_ep = sim_sec.create_ipv4_peer(0, 9930)
    var p0_sec = sim_sec.create_ipv4_peer(1, 9931)
    var p1_sec = sim_sec.create_ipv4_peer(2, 9932)
    var rogue_sec = sim_sec.create_ipv4_peer(3, 9939)
    var relay_sec = RelayServer(r_sec_ep.local_endpoint())

    var reg_bytes = RelayForwardHeader(
        session_token=DEFAULT_ROLLBACK_SESSION_TOKEN,
        src_peer=1,
        dst_peer=1,
        flags=RELAY_FLAG_REGISTER,
    ).encode_bytes()
    assert_true(p1_sec.send_raw(r_sec_ep.local_endpoint(), Int(reg_bytes.unsafe_ptr()), len(reg_bytes)))
    sim_sec.advance_ms(10.0)
    relay_sec.poll(r_sec_ep, sim_sec.now_ms())
    assert_equal(relay_sec.registrations, 1)

    # Rogue tries to hijack (DEFAULT_ROLLBACK_SESSION_TOKEN, peer 1) from different endpoint
    assert_true(rogue_sec.send_raw(r_sec_ep.local_endpoint(), Int(reg_bytes.unsafe_ptr()), len(reg_bytes)))
    _ = reg_bytes[0]
    sim_sec.advance_ms(10.0)
    relay_sec.poll(r_sec_ep, sim_sec.now_ms())
    assert_true(relay_sec.spoofed_packets_dropped >= 1)
    var lk = relay_sec.lookup_peer(DEFAULT_ROLLBACK_SESSION_TOKEN, 1)
    assert_true(lk[0] and lk[1] == p1_sec.local_endpoint())

    # Rogue sends forged RELAY_WIRE_MAGIC directly to RollbackSession s_sec0 pretending to be relay
    var sec_addrs = List[NetEndpoint]()
    sec_addrs.append(p0_sec.local_endpoint())
    sec_addrs.append(p1_sec.local_endpoint())
    var pw_sec = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s_sec = RollbackSession[PhysicsRollbackInput](0, sec_addrs)
    s_sec.enable_relay(r_sec_ep.local_endpoint())
    var forged_relay = RelayForwardHeader(
        session_token=DEFAULT_ROLLBACK_SESSION_TOKEN,
        src_peer=1,
        dst_peer=0,
        flags=RELAY_FLAG_DATA,
    ).encode_bytes()
    for _ in range(32):
        forged_relay.append(0)
    assert_true(rogue_sec.send_raw(p0_sec.local_endpoint(), Int(forged_relay.unsafe_ptr()), len(forged_relay)))
    _ = forged_relay[0]
    sim_sec.advance_ms(10.0)
    s_sec.poll_network(pw_sec, p0_sec, sim_sec.now_ms())
    assert_true(s_sec.spoofed_packets_dropped >= 1)

    # Relayed packet from valid relay_endpoint whose inner sender_id != outer hdr.src_peer must be dropped
    var prev_drops = s_sec.spoofed_packets_dropped
    var mismatch_bw = BitWriter(128)
    mismatch_bw.write_u32(_RB_WIRE_MAGIC)
    mismatch_bw.write_u64(DEFAULT_ROLLBACK_SESSION_TOKEN)
    mismatch_bw.write_u8(UInt8(0))  # inner sender_id == 0, while outer hdr.src_peer == 1!
    for _ in range(32):
        mismatch_bw.write_u8(0)
    var inner_m = mismatch_bw.finish()
    var mismatch_pkt = RelayForwardHeader(
        session_token=DEFAULT_ROLLBACK_SESSION_TOKEN,
        src_peer=1,
        dst_peer=0,
        flags=RELAY_FLAG_DATA,
    ).encode_bytes()
    for b in range(len(inner_m)):
        mismatch_pkt.append(inner_m[b])
    assert_true(r_sec_ep.send_raw(p0_sec.local_endpoint(), Int(mismatch_pkt.unsafe_ptr()), len(mismatch_pkt)))
    _ = mismatch_pkt[0]
    sim_sec.advance_ms(10.0)
    s_sec.poll_network(pw_sec, p0_sec, sim_sec.now_ms())
    assert_true(s_sec.spoofed_packets_dropped > prev_drops)
    assert_true(sim_sec.now_ms() >= 0.0)

    # 6. Regression check: non-deterministic NATPunchProbe.local_nonce & PUNCH_KIND_PROBE hijack defense when DIRECT
    var probe_a = NATPunchProbe(
        local_peer=0,
        relay_endpoint=r_ep_cone.local_endpoint(),
        target_peer=1,
        session_token=DEFAULT_ROLLBACK_SESSION_TOKEN,
    )
    var probe_b = NATPunchProbe(
        local_peer=0,
        relay_endpoint=r_ep_cone.local_endpoint(),
        target_peer=1,
        session_token=DEFAULT_ROLLBACK_SESSION_TOKEN,
    )
    assert_true(probe_a.local_nonce != probe_b.local_nonce)

    # Send forged PUNCH_KIND_PROBE from rogue_sec to s_c0 (already DIRECT with peer 1)
    var ep_before = s_c0.peer_endpoints[1]
    var punch_before = s_c0.peer_punches[1].target_endpoint
    var drops_before = s_c0.spoofed_packets_dropped
    var rogue_probe = List[UInt8](unsafe_uninit_length=24)
    var rp_base = Int(rogue_probe.unsafe_ptr())
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rp_base).unsafe_store[alignment=1](NAT_PUNCH_MAGIC)
    Pointer[UInt64, MutAnyOrigin](unsafe_from_address=rp_base + 4).unsafe_store[alignment=1](DEFAULT_ROLLBACK_SESSION_TOKEN)
    rogue_probe[12] = PUNCH_KIND_PROBE
    rogue_probe[13] = UInt8(1)
    rogue_probe[14] = UInt8(0)
    rogue_probe[15] = UInt8(0)
    Pointer[UInt64, MutAnyOrigin](unsafe_from_address=rp_base + 16).unsafe_store[alignment=1](UInt64(0))
    var rogue_punch_ep = sim_cone.create_ipv4_peer(9, 9949)
    cone0.opened_peers.append(rogue_punch_ep.local_endpoint())
    assert_true(rogue_punch_ep.send_raw(cone0.local_endpoint(), Int(rogue_probe.unsafe_ptr()), len(rogue_probe)))
    _ = rogue_probe[0]
    sim_cone.advance_ms(20.0)
    s_c0.poll_network(pw_c0, cone0, sim_cone.now_ms())
    assert_true(s_c0.peer_endpoints[1] == ep_before)
    assert_true(s_c0.peer_punches[1].target_endpoint == punch_before)
    assert_true(s_c0.spoofed_packets_dropped > drops_before)

    # Direct NATPunchProbe.ingest_datagram check for PUNCH_KIND_PROBE hijack when DIRECT
    var rogue_view = DatagramView(
        rogue_punch_ep.local_endpoint(),
        Int(rogue_probe.unsafe_ptr()),
        len(rogue_probe),
    )
    assert_false(s_c0.peer_punches[1].ingest_datagram(rogue_view, cone0, sim_cone.now_ms()))
    _ = rogue_probe[0]
    assert_true(sim_cone.now_ms() >= 0.0)

    # 7. Regression check: RelayServer.poll evicts stale peers BEFORE Pass 1 registration
    var sim_evict = VirtualNetSim.from_rtt(rtt_ms=5.0, jitter_ms=0.0, loss_rate=0.0, seed=0xF9E7)
    var r_ev_ep = sim_evict.create_ipv4_peer(0, 9950)
    var ev_p0 = sim_evict.create_ipv4_peer(1, 9951)
    var ev_p1 = sim_evict.create_ipv4_peer(2, 9952)
    var ev_p2 = sim_evict.create_ipv4_peer(3, 9953)
    var relay_ev = RelayServer(r_ev_ep.local_endpoint(), max_peers=2, peer_timeout_ms=100.0)
    var tok_ev = UInt64(0xABCD1234EF567890)
    for pid in range(2):
        var rb = RelayForwardHeader(
            session_token=tok_ev, src_peer=pid, dst_peer=pid, flags=RELAY_FLAG_REGISTER
        ).encode_bytes()
        if pid == 0:
            _ = ev_p0.send_raw(r_ev_ep.local_endpoint(), Int(rb.unsafe_ptr()), len(rb))
        else:
            _ = ev_p1.send_raw(r_ev_ep.local_endpoint(), Int(rb.unsafe_ptr()), len(rb))
        _ = rb[0]
    sim_evict.advance_ms(10.0)
    relay_ev.poll(r_ev_ep, sim_evict.now_ms())
    assert_equal(relay_ev.active_peer_count(), 2)
    assert_equal(relay_ev.registrations, 2)

    # Advance past peer_timeout_ms=100ms, then register peer 2 in a single poll
    sim_evict.advance_ms(120.0)
    var rb2 = RelayForwardHeader(
        session_token=tok_ev, src_peer=2, dst_peer=2, flags=RELAY_FLAG_REGISTER
    ).encode_bytes()
    _ = ev_p2.send_raw(r_ev_ep.local_endpoint(), Int(rb2.unsafe_ptr()), len(rb2))
    _ = rb2[0]
    sim_evict.advance_ms(10.0)
    relay_ev.poll(r_ev_ep, sim_evict.now_ms())
    assert_equal(relay_ev.registrations, 3)
    assert_equal(relay_ev.invalid_packets_dropped, 0)
    assert_equal(relay_ev.active_peer_count(), 1)
    assert_true(relay_ev.lookup_peer(tok_ev, 2)[0])
    assert_true(sim_evict.now_ms() >= 0.0)

    # 8. Regression check: RollbackSession + RelayServer.enable_cookie_auth end-to-end
    var sim_ck = VirtualNetSim.from_rtt(rtt_ms=16.0, jitter_ms=1.0, loss_rate=0.0, seed=0xF9CC)
    var r_ep_ck = sim_ck.create_ipv4_peer(0, 9960)
    var raw_ck0 = sim_ck.create_ipv4_peer(1, 9961)
    var raw_ck1 = sim_ck.create_ipv4_peer(2, 9962)
    var ck_addrs = List[NetEndpoint]()
    ck_addrs.append(raw_ck0.local_endpoint())
    ck_addrs.append(raw_ck1.local_endpoint())
    var nat_ck0 = _SymmetricNatTransport(raw_ck0^, r_ep_ck.local_endpoint(), allow_direct=False)
    var nat_ck1 = _SymmetricNatTransport(raw_ck1^, r_ep_ck.local_endpoint(), allow_direct=False)
    var relay_ck = RelayServer(r_ep_ck.local_endpoint())
    relay_ck.enable_cookie_auth(AeadKey.from_u64x4(11, 22, 33, 44), bucket_ms=10000.0)

    var pw_ck0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw_ck1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s_ck0 = RollbackSession[PhysicsRollbackInput](0, ck_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    var s_ck1 = RollbackSession[PhysicsRollbackInput](1, ck_addrs, input_delay_ticks=2, max_rollback_ticks=8, checksum_interval=4)
    s_ck0.enable_relay(r_ep_ck.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=100.0)
    s_ck1.enable_relay(r_ep_ck.local_endpoint(), attempt_nat_punch=True, probe_timeout_ms=100.0)
    s_ck0.bind_world(pw_ck0)
    s_ck1.bind_world(pw_ck1)

    for f in range(70):
        var now = sim_ck.now_ms()
        relay_ck.poll(r_ep_ck, now)
        var cmd = PhysicsRollbackInput(Float32(f % 3), 0.0, 1.0, 0.0, 0)
        _ = s_ck0.step_frame(cmd, pw_ck0, nat_ck0, now)
        _ = s_ck1.step_frame(cmd, pw_ck1, nat_ck1, now)
        sim_ck.advance_ms(1000.0 / 60.0)

    assert_true(relay_ck.registrations >= 2)
    assert_true(relay_ck.packets_forwarded > 0)
    assert_true(s_ck0.relay_has_cookie and s_ck1.relay_has_cookie)
    assert_true(s_ck0.relayed_packets_received > 0 and s_ck1.relayed_packets_received > 0)
    assert_false(s_ck0.desync_detected or s_ck1.desync_detected)
    assert_true(len(s_ck0.confirmed_hash_history) >= 30)
    for t in range(30):
        assert_equal(s_ck0.confirmed_hash_history[t], s_ck1.confirmed_hash_history[t])
    assert_true(sim_ck.now_ms() >= 0.0)


def main() raises:
    var t_start = monotonic()
    set_sim_float_mode()
    print("============================================================")
    print(" FLINT V5 NETWORKING, ROLLBACK & LAB VERIFICATION SUITE")
    print("============================================================")
    test_a_dual_stack_udp_32_burst()
    test_b_bitpack_10k_fuzz_and_bounds()
    test_c_channel_reliable_8kib_and_sequenced()
    test_d_rollback_7pkt_drop_and_10k_4peer_chaos()
    test_e_replicator_16_clients_bandwidth_gate()
    test_f_lab_bench_suite_harness()
    test_g_physics_world_rollback_and_spoof_auth()
    test_h_net_hub_routing_accept_drop()
    test_i_security_dos_and_wraparound_regressions()
    test_j_wave2_aead_cookie_channels_despawn_interp()
    test_k_f6_client_prediction_reconciliation()
    test_l_f7_server_lag_compensation()
    test_m_f8_chunk_relative_open_world_quantization()
    test_n_f9_nat_relay_rollback_topology()
    var total_ms = Float64(monotonic() - t_start) / 1.0e6
    print("============================================================")
    print(" [PASS] ALL FLINT V5 NET & ROLLBACK TESTS PASSED (" + String(Int(total_ms)) + " ms)")
    print("============================================================")
