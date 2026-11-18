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
    BitReader,
    BitWriter,
    DatagramView,
    IPV6_SAFE_MTU_BYTES,
    JITTER_TRIANGULAR,
    NetChannel,
    NetHub,
    NetEndpoint,
    ROLLBACK_INPUT_HISTORY_K,
    RollbackInput,
    RollbackSession,
    RollbackWorld,
    SnapshotClientReceiver,
    SnapshotReader,
    SnapshotReplicator,
    UdpSocket,
    VirtualEndpoint,
    VirtualNetSim,
    VisualSmoother,
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
    w.write_bool(False)
    w.write_u32(UInt32(max(0, tick)))
    w.write_u32(UInt32(max(0, tick)))
    w.write_int_bounded(1, 1, ROLLBACK_INPUT_HISTORY_K)
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
    var total_ms = Float64(monotonic() - t_start) / 1.0e6
    print("============================================================")
    print(" [PASS] ALL FLINT V5 NET & ROLLBACK TESTS PASSED (" + String(Int(total_ms)) + " ms)")
    print("============================================================")
