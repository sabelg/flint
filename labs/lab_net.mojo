"""Flint v5 Networking, GGPO Rollback & Replication Subsystem Lab (`labs/lab_net.mojo`).

Standalone `<2s` benchmark & determinism gate exercising:
1. Dual-stack Linux `UdpSocket` (`[::1]` IPv6 + `127.0.0.1` IPv4-mapped) `sendmmsg`/`recvmmsg`
   32-datagram burst round-trip with exact sender `NetEndpoint` recovery.
2. `BitWriter`/`BitReader` 10,000-iteration fuzz (`bounded int`, quantized `Vec3`, 32-bit
   Smallest-Three `Quat` `< 0.08 deg` max error, and truncated-buffer `overflowed` safety).
3. `NetChannel` reliable-ordered `8 KiB` multi-fragment delivery across `VirtualNetSim`
   (`12%` packet loss, `25 ms` jitter, `2%` duplication, `<= 1180 B` IPv6-safe MTU).
4. 10,000-Tick 4-Peer Deterministic GGPO Rollback Chaos Fuzz (`80 ms` RTT, `20 ms` jitter,
   `5%` packet loss, `8-tick` max rollback, + 6-packet burst drop + 7-packet `K=8` survival).
5. 16-client `SnapshotReplicator` spatial AOI + baseline delta compression (`< 24 KB/s` for
   256 active entities @ 20 Hz) + `EngineBenchSuite` FLIP golden/CSV verification.
"""
from std.collections import List
from std.math import abs, max, min, sqrt
from std.time import monotonic
from flint.core.detmath import det_cos, det_sin, f32_bits, set_sim_float_mode
from flint.core.detsort import mix64
from flint.core.lz4 import hash_list
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobWriter
from flint.lab.bench_suite import EngineBenchSuite, compute_latency_percentiles
from flint.platform.os import remove_file
from flint.net.rollback import (
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
)


# ============================================================================
# Deterministic 4-Peer Arena Rigid-Body World for Rollback Fuzzing
# ============================================================================

@fieldwise_init
struct ArenaInput(RollbackInput):
    var move_x: Float32
    var move_z: Float32
    var aim_rot: Quat
    var buttons: Int

    def __init__(out self):
        self.move_x = 0.0
        self.move_z = 0.0
        self.aim_rot = Quat.identity()
        self.buttons = 0

    def encode_bits(self, mut writer: BitWriter):
        writer.write_f32_quantized(self.move_x, -1.0, 1.0, 9)
        writer.write_f32_quantized(self.move_z, -1.0, 1.0, 9)
        writer.write_quat_smallest_three(self.aim_rot)
        writer.write_int_bounded(self.buttons, 0, 15)

    def decode_bits(mut self, mut reader: BitReader):
        self.move_x = reader.read_f32_quantized(-1.0, 1.0, 9)
        self.move_z = reader.read_f32_quantized(-1.0, 1.0, 9)
        self.aim_rot = reader.read_quat_smallest_three()
        self.buttons = reader.read_int_bounded(0, 15)

    def same_input(self, other: Self) -> Bool:
        return (
            f32_bits(self.move_x) == f32_bits(other.move_x)
            and f32_bits(self.move_z) == f32_bits(other.move_z)
            and self.buttons == other.buttons
            and f32_bits(self.aim_rot.x) == f32_bits(other.aim_rot.x)
            and f32_bits(self.aim_rot.y) == f32_bits(other.aim_rot.y)
            and f32_bits(self.aim_rot.z) == f32_bits(other.aim_rot.z)
            and f32_bits(self.aim_rot.w) == f32_bits(other.aim_rot.w)
        )


struct ArenaBody(Copyable, ImplicitlyCopyable, Movable):
    var pos: Vec3
    var vel: Vec3
    var rot: Quat
    var ang_vel: Vec3
    var boost_cd: Int
    var hits: UInt32

    def __init__(out self, pos: Vec3 = Vec3()):
        self.pos = pos
        self.vel = Vec3()
        self.rot = Quat.identity()
        self.ang_vel = Vec3()
        self.boost_cd = 0
        self.hits = 0


struct ArenaRollbackWorld(RollbackWorld):
    """Coupled 4-craft + 4-orb nonlinear rigid-body arena world (highly sensitive to input order/bits)."""
    var bodies: List[ArenaBody]
    var tick_counter: UInt32

    def __init__(out self, num_peers: Int = 4):
        self.bodies = List[ArenaBody](capacity=8)
        for i in range(num_peers):
            var ang = Float32(i) * Float32(1.5707963)
            self.bodies.append(
                ArenaBody(Vec3(det_cos(ang) * 12.0, 1.0, det_sin(ang) * 12.0))
            )
        # 4 shared dynamic arena orbs that all 4 peers collide with simultaneously
        for o in range(4):
            var oang = Float32(o) * Float32(1.5707963) + Float32(0.785398)
            self.bodies.append(
                ArenaBody(Vec3(det_cos(oang) * 4.5, 1.0, det_sin(oang) * 4.5))
            )
        self.tick_counter = 0

    def save_snapshot(self, mut writer: BlobWriter):
        writer.u32(self.tick_counter)
        writer.u32(UInt32(len(self.bodies)))
        for i in range(len(self.bodies)):
            writer.vec3(self.bodies[i].pos)
            writer.vec3(self.bodies[i].vel)
            writer.quat(self.bodies[i].rot)
            writer.vec3(self.bodies[i].ang_vel)
            writer.u32(UInt32(self.bodies[i].boost_cd))
            writer.u32(self.bodies[i].hits)

    def load_snapshot(mut self, addr: Int, length: Int):
        var r = SnapshotReader(addr, length)
        self.tick_counter = r.u32()
        var cnt = Int(r.u32())
        if len(self.bodies) != cnt and cnt > 0 and cnt <= 16:
            self.bodies = List[ArenaBody](length=cnt, fill=ArenaBody())
        for i in range(len(self.bodies)):
            self.bodies[i].pos = r.vec3()
            self.bodies[i].vel = r.vec3()
            self.bodies[i].rot = r.quat()
            self.bodies[i].ang_vel = r.vec3()
            self.bodies[i].boost_cd = Int(r.u32())
            self.bodies[i].hits = r.u32()

    def step_rollback_tick(
        mut self, inputs_addr: Int, peer_count: Int, tick: Int
    ):
        comptime DT: Float32 = Float32(1.0 / 60.0)
        comptime ARENA_R: Float32 = Float32(24.0)
        var in_ptr = Pointer[ArenaInput, ImmutAnyOrigin](
            unsafe_from_address=inputs_addr
        )
        var n_peers = min(peer_count, len(self.bodies))

        for p in range(n_peers):
            var inp = in_ptr[unsafe_offset=p]
            var thrust = Vec3(inp.move_x, 0.0, inp.move_z)
            if thrust.length_squared() > Float32(1.0):
                thrust = thrust.normalized()
            var accel = thrust * Float32(28.0)
            if self.bodies[p].boost_cd > 0:
                self.bodies[p].boost_cd -= 1
            if (inp.buttons & 1) != 0 and self.bodies[p].boost_cd == 0:
                var dash_dir = inp.aim_rot.rotate(Vec3(0.0, 0.0, 1.0)).horizontal().normalized()
                self.bodies[p].vel = self.bodies[p].vel + dash_dir * Float32(9.5)
                self.bodies[p].boost_cd = 25
            if (inp.buttons & 2) != 0:
                self.bodies[p].ang_vel.y += Float32(4.5) * DT
            self.bodies[p].vel = (self.bodies[p].vel + accel * DT) * Float32(0.985)
            self.bodies[p].rot = self.bodies[p].rot.slerp(inp.aim_rot, Float32(0.25))

        var total = len(self.bodies)
        for i in range(n_peers, total):
            # Central vortex spring on arena orbs
            var pull = -self.bodies[i].pos.horizontal() * Float32(0.65)
            self.bodies[i].vel = (self.bodies[i].vel + pull * DT) * Float32(0.992)

        for i in range(total):
            self.bodies[i].pos = self.bodies[i].pos + self.bodies[i].vel * DT
            self.bodies[i].rot.integrate(self.bodies[i].ang_vel, DT)
            self.bodies[i].ang_vel = self.bodies[i].ang_vel * Float32(0.96)

            # Circular arena boundary bounce
            var horiz = self.bodies[i].pos.horizontal()
            var dist = horiz.length()
            if dist > ARENA_R:
                var n = horiz * (Float32(1.0) / dist)
                self.bodies[i].pos.x = n.x * ARENA_R
                self.bodies[i].pos.z = n.z * ARENA_R
                var vn = self.bodies[i].vel.dot(n)
                if vn > Float32(0.0):
                    self.bodies[i].vel = self.bodies[i].vel - n * (Float32(1.85) * vn)

        # Pairwise elastic sphere collisions (radius = 1.25m)
        comptime MIN_DIST: Float32 = Float32(2.5)
        for i in range(total):
            for j in range(i + 1, total):
                var delta = self.bodies[j].pos - self.bodies[i].pos
                var d2 = delta.length_squared()
                if d2 > Float32(1.0e-6) and d2 < MIN_DIST * MIN_DIST:
                    var d = sqrt(d2)
                    var n = delta * (Float32(1.0) / d)
                    var pen = (MIN_DIST - d) * Float32(0.5)
                    self.bodies[i].pos = self.bodies[i].pos - n * pen
                    self.bodies[j].pos = self.bodies[j].pos + n * pen
                    var rel_v = (self.bodies[j].vel - self.bodies[i].vel).dot(n)
                    if rel_v < Float32(0.0):
                        var imp = n * (Float32(-0.92) * rel_v)
                        self.bodies[i].vel = self.bodies[i].vel - imp
                        self.bodies[j].vel = self.bodies[j].vel + imp
                        self.bodies[i].hits += 1
                        self.bodies[j].hits += 1

        self.tick_counter = UInt32(tick + 1)


def _synth_player_input(peer_id: Int, frame: Int) -> ArenaInput:
    """Deterministic chaotic input generator with sudden button edges & 3D aim rotations."""
    var h = mix64(UInt64(peer_id * 1000003 + (frame // 5) * 97 + 17))
    var h_fast = mix64(UInt64(peer_id * 7919 + frame * 131 + 53))
    var mx = Float32(Int(h & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    var mz = Float32(Int((h >> 16) & 0xFFFF)) / Float32(32767.5) - Float32(1.0)
    var yaw = (Float32(Int((h >> 32) & 0xFFFF)) / Float32(65535.0)) * Float32(6.2831853)
    var pitch = (Float32(Int((h >> 48) & 0xFF)) / Float32(255.0) - Float32(0.5)) * Float32(0.8)
    var buttons = Int(h_fast & 3) if (frame % 11 == peer_id * 2) else 0
    return ArenaInput(mx, mz, Quat.from_euler(pitch, yaw, 0.0), buttons)


# ============================================================================
# Lab Cases
# ============================================================================

def run_case_dual_stack_udp(mut suite: EngineBenchSuite) -> Bool:
    var t0 = monotonic()
    var server = UdpSocket.open_dual_stack(0)
    if not server.is_open() or server.local_port() <= 0:
        return suite.record_single("udp_dual_stack_mmsg", 0.0, False, detail="bind [::]:0 failed")

    var client_v6 = UdpSocket()
    var client_v4 = UdpSocket()
    var ok_v6 = client_v6.bind(NetEndpoint.ipv6_loopback(0))
    var ok_v4 = client_v4.bind(NetEndpoint.ipv4_loopback(0))
    if not ok_v6 or not ok_v4:
        return suite.record_single("udp_dual_stack_mmsg", 0.0, False, detail="client loopback bind failed")

    var dst_v6 = NetEndpoint.ipv6_loopback(server.local_port())
    var dst_v4 = NetEndpoint.ipv4_loopback(server.local_port())

    # Queue 16 IPv6 + 16 IPv4-mapped datagrams (`32` total) via `sendmmsg`
    for i in range(16):
        var p6 = List[UInt8](capacity=24)
        p6.append(UInt8(6))
        p6.append(UInt8(i))
        for k in range(22):
            p6.append(UInt8((i * 17 + k) & 0xFF))
        _ = client_v6.queue_send_bytes(dst_v6, p6)

        var p4 = List[UInt8](capacity=24)
        p4.append(UInt8(4))
        p4.append(UInt8(i))
        for k in range(22):
            p4.append(UInt8((i * 29 + k + 9) & 0xFF))
        _ = client_v4.queue_send_bytes(dst_v4, p4)

    var sent_v6 = client_v6.flush_sends()
    var sent_v4 = client_v4.flush_sends()

    var views = List[DatagramView](capacity=64)
    var got_v6 = 0
    var got_v4 = 0
    var endpoints_exact = True

    for _ in range(50):
        var n = server.recv_batch(views)
        for i in range(n):
            var v = views[i]
            if v.length != 24:
                endpoints_exact = False
                continue
            var tag = Int(v.byte_at(0))
            if tag == 6:
                got_v6 += 1
                if not v.sender.is_ipv6_loopback() or v.sender != client_v6.local_endpoint():
                    endpoints_exact = False
            elif tag == 4:
                got_v4 += 1
                if not v.sender.is_ipv4_mapped() or v.sender != client_v4.local_endpoint():
                    endpoints_exact = False
            _ = server.queue_send_raw(v.sender, v.data_ptr, v.length)
        if got_v6 + got_v4 >= 32:
            break

    var echoed = server.flush_sends()
    var echo_rx = 0
    for _ in range(50):
        echo_rx += client_v6.recv_batch(views)
        echo_rx += client_v4.recv_batch(views)
        if echo_rx >= 32:
            break

    var elapsed_ms = Float64(monotonic() - t0) / 1.0e6
    var passed = (
        sent_v6 == 16
        and sent_v4 == 16
        and got_v6 == 16
        and got_v4 == 16
        and echoed == 32
        and echo_rx == 32
        and endpoints_exact
        and client_v6.sendmmsg_calls == 1
        and client_v4.sendmmsg_calls == 1
    )
    return suite.record_single(
        "udp_dual_stack_32_mmsg",
        elapsed_ms,
        passed,
        budget_ms=50.0,
        detail="v6=" + String(got_v6) + " v4_mapped=" + String(got_v4) + " echo=" + String(echo_rx),
    )


def run_case_bitpack_fuzz(mut suite: EngineBenchSuite) -> Bool:
    var t0 = monotonic()
    comptime N: Int = 10000
    var writer = BitWriter(131072)
    var mins = List[Int](capacity=N)
    var maxs = List[Int](capacity=N)
    var vals = List[Int](capacity=N)
    var vecs = List[Vec3](capacity=N)
    var quats = List[Quat](capacity=N)

    var rng = UInt64(0xA1B2C3D4E5F60718)
    for i in range(N):
        rng = mix64(rng + 1)
        var lo = -Int(rng & 2047)
        var span = Int((rng >> 12) & 8191)
        var hi = lo + span
        var v = lo + (Int((rng >> 28) & 0xFFFF) % (span + 1))
        mins.append(lo)
        maxs.append(hi)
        vals.append(v)
        writer.write_int_bounded(v, lo, hi)

        rng = mix64(rng + 2)
        var vx = (Float32(Int(rng & 0xFFFF)) / Float32(65535.0)) * Float32(400.0) - Float32(200.0)
        var vy = (Float32(Int((rng >> 16) & 0xFFFF)) / Float32(65535.0)) * Float32(400.0) - Float32(200.0)
        var vz = (Float32(Int((rng >> 32) & 0xFFFF)) / Float32(65535.0)) * Float32(400.0) - Float32(200.0)
        var vec = Vec3(vx, vy, vz)
        vecs.append(vec)
        writer.write_vec3_quantized(vec, -200.0, 200.0, 16)

        rng = mix64(rng + 3)
        var q: Quat
        if i == 0:
            # Adversarial equal-magnitude corner `(0.5, 0.5, 0.5, 0.5)`
            q = Quat(0.5, 0.5, 0.5, 0.5).normalized()
        elif i == 1:
            q = Quat(-0.5, 0.5, -0.5, 0.5).normalized()
        else:
            var qx = (Float32(Int(rng & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            var qy = (Float32(Int((rng >> 16) & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            var qz = (Float32(Int((rng >> 32) & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            var qw = (Float32(Int((rng >> 48) & 0xFFFF)) / Float32(32767.5)) - Float32(1.0)
            q = Quat(qx, qy, qz, qw).normalized()
        quats.append(q)
        writer.write_quat_smallest_three(q)

    var bytes = writer.finish()
    var reader = BitReader(bytes)
    var ints_ok = True
    var vecs_ok = True
    var max_quat_err_deg = Float32(0.0)

    for i in range(N):
        var ri = reader.read_int_bounded(mins[i], maxs[i])
        if ri != vals[i]:
            ints_ok = False
        var rv = reader.read_vec3_quantized(-200.0, 200.0, 16)
        if (rv - vecs[i]).length() > Float32(0.012):
            vecs_ok = False
        var rq = reader.read_quat_smallest_three()
        var err = quat_angular_error_deg(quats[i], rq)
        if err > max_quat_err_deg:
            max_quat_err_deg = err

    # Truncated buffer safety check (`overflowed == True`, zero crash)
    var trunc_bytes = List[UInt8](capacity=5)
    for k in range(5):
        trunc_bytes.append(bytes[k])
    var trunc_reader = BitReader(trunc_bytes, bit_limit=35)
    _ = trunc_reader.read_u32()
    var ok_before = not trunc_reader.overflowed
    _ = trunc_reader.read_quat_smallest_three()
    var overflow_caught = ok_before and trunc_reader.overflowed

    var elapsed_ms = Float64(monotonic() - t0) / 1.0e6
    var passed = (
        not reader.overflowed
        and ints_ok
        and vecs_ok
        and max_quat_err_deg < Float32(0.08)
        and overflow_caught
    )
    return suite.record_single(
        "bitpack_10k_quat32_fuzz",
        elapsed_ms,
        passed,
        budget_ms=50.0,
        detail="max_quat_err_deg=" + String(max_quat_err_deg) + " overflow_safe=" + String(overflow_caught),
    )


def run_case_reliable_channel_8kib(mut suite: EngineBenchSuite) -> Bool:
    var t0 = monotonic()
    var sim = VirtualNetSim(
        seed=0x900DF00DCAFE1234,
        one_way_latency_ms=35.0,
        jitter_ms=25.0,
        loss_rate=0.12,
        dup_rate=0.02,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var ep_a = sim.create_ipv4_peer(0, 9100)
    var ep_b = sim.create_ipv4_peer(1, 9100)

    var chan_a = NetChannel(ep_b.local_endpoint(), IPV6_SAFE_MTU_BYTES)
    var chan_b = NetChannel(ep_a.local_endpoint(), IPV6_SAFE_MTU_BYTES)

    var payload_8k = List[UInt8](capacity=8192)
    for i in range(8192):
        payload_8k.append(UInt8(mix64(UInt64(i * 31 + 7)) & 0xFF))
    var expected_hash = hash_list(payload_8k)

    var queued = chan_a.send_reliable(payload_8k)
    var received_ok = False
    var received_hash = UInt64(0)

    for step in range(400):
        var now = sim.now_ms()
        chan_a.poll(ep_a, now)
        chan_b.poll(ep_b, now)
        var msg = chan_b.pop_reliable_message()
        if msg[0]:
            received_hash = hash_list(msg[1])
            received_ok = (len(msg[1]) == 8192) and (received_hash == expected_hash)
            if received_ok and chan_a.is_all_reliable_acked():
                break
        sim.advance_ms(10.0)
        _ = step

    var elapsed_ms = Float64(monotonic() - t0) / 1.0e6
    var passed = (
        queued
        and received_ok
        and sim.packets_dropped_mtu() == 0
        and chan_a.retransmissions > 0
    )
    return suite.record_determinism_gate(
        "channel_reliable_8kib_loss12",
        expected_hash,
        received_hash if passed else (expected_hash ^ UInt64(1)),
        elapsed_ms=elapsed_ms,
        budget_ms=50.0,
        detail="retx=" + String(chan_a.retransmissions) + " mtu_drops=" + String(sim.packets_dropped_mtu()),
    )


def run_case_rollback_10k_4peer_chaos(mut suite: EngineBenchSuite) -> Bool:
    var t0 = monotonic()
    comptime TARGET_TICKS: Int = 10000
    var sim = VirtualNetSim.from_rtt(
        rtt_ms=80.0,
        jitter_ms=20.0,
        loss_rate=0.05,
        dup_rate=0.01,
        mtu_bytes=1200,
        seed=0x4F4C4C4241434B35,
        jitter_mode=JITTER_TRIANGULAR,
    )

    var eps = List[VirtualEndpoint]()
    var addrs = List[NetEndpoint]()
    for p in range(4):
        var ep = sim.create_ipv4_peer(p, 9500)
        addrs.append(ep.local_endpoint())
        eps.append(ep^)

    var worlds = List[ArenaRollbackWorld]()
    var sessions = List[RollbackSession[ArenaInput]]()
    for p in range(4):
        var w = ArenaRollbackWorld(4)
        var s = RollbackSession[ArenaInput](
            local_peer_id=p,
            peer_endpoints=addrs,
            input_delay_ticks=2,
            max_rollback_ticks=8,
            tick_hz=60.0,
            checksum_interval=8,
            record_full_hash_history=True,
        )
        s.enable_rift_pacing = False  # Stress full 8-tick window under chaos + 6-pkt burst
        s.bind_world(w)
        worlds.append(w^)
        sessions.append(s^)

    var resim_samples_ms = List[Float64](capacity=1024)
    var frame = 0
    var max_frames = 14000
    while frame < max_frames:
        # Inject an explicit 6-packet consecutive burst drop on peer 1 at frame 2500
        # and on peer 3 at frame 6500!
        if frame == 2500:
            sim.drop_next_outbound_packets(1, 6)
        if frame == 6500:
            sim.drop_next_outbound_packets(3, 6)

        var now = sim.now_ms()
        var min_conf = TARGET_TICKS
        for p in range(4):
            var prev_rb = sessions[p].rollback_count
            var inp = _synth_player_input(p, sessions[p].current_tick)
            _ = sessions[p].step_frame(inp, worlds[p], eps[p], now)
            if sessions[p].rollback_count > prev_rb:
                resim_samples_ms.append(sessions[p].last_resim_ms)
            if len(sessions[p].confirmed_hash_history) < min_conf:
                min_conf = len(sessions[p].confirmed_hash_history)

        if min_conf >= TARGET_TICKS:
            break

        sim.advance_ms(1000.0 / 60.0)
        frame += 1

    # Verify zero desyncs and 100% bit-identical confirmed_hash across all 4 peers on ALL 10,000 ticks!
    var any_desync = False
    var all_have_10k = True
    var total_rollbacks = 0
    var max_resim_ms = 0.0
    for p in range(4):
        if sessions[p].desync_detected:
            any_desync = True
        if len(sessions[p].confirmed_hash_history) < TARGET_TICKS:
            all_have_10k = False
        total_rollbacks += sessions[p].rollback_count
        if sessions[p].peak_resim_ms > max_resim_ms:
            max_resim_ms = sessions[p].peak_resim_ms

    var mismatch_ticks = 0
    var rolling_agree_a = UInt64(0xCBF29CE484222325)
    var rolling_agree_b = UInt64(0xCBF29CE484222325)
    if all_have_10k:
        for t in range(TARGET_TICKS):
            var h0 = sessions[0].confirmed_hash_history[t]
            var h1 = sessions[1].confirmed_hash_history[t]
            var h2 = sessions[2].confirmed_hash_history[t]
            var h3 = sessions[3].confirmed_hash_history[t]
            rolling_agree_a = mix64(rolling_agree_a ^ h0)
            if h0 == h1 and h0 == h2 and h0 == h3:
                rolling_agree_b = mix64(rolling_agree_b ^ h3)
            else:
                mismatch_ticks += 1
    else:
        rolling_agree_b = 0

    # Dedicated 8-tick full-depth rollback microbenchmark (< 1.2 ms gate)
    var bench_8tick_samples = List[Float64](capacity=64)
    for _ in range(32):
        var from_t = max(0, sessions[0].current_tick - 8)
        sessions[0].rollback_and_resimulate(from_t, worlds[0])
        bench_8tick_samples.append(sessions[0].last_resim_ms)

    var smoother = VisualSmoother(18.0)
    smoother.record_correction(Vec3(2.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    smoother.step(Float32(1.0 / 60.0))
    var smooth_ok = smoother.pos_offset.x > Float32(0.1) and smoother.pos_offset.x < Float32(1.9)

    var total_ms = Float64(monotonic() - t0) / 1.0e6
    var chaos_resim_p99_ms = (
        compute_latency_percentiles(resim_samples_ms)[1]
        if len(resim_samples_ms) > 0
        else Float64(0.0)
    )
    var chaos_resim_ok = len(resim_samples_ms) == 0 or chaos_resim_p99_ms < 1.2
    var bench_ok = suite.record_samples(
        "rollback_8tick_resim_latency",
        bench_8tick_samples,
        budget_p99_ms=1.2,
        detail="chaos_resim_p99_ms="
        + String(chaos_resim_p99_ms)
        + " peak_chaos_resim_ms="
        + String(max_resim_ms),
    )
    if bench_ok and not chaos_resim_ok:
        var last_idx = len(suite.results) - 1
        suite.results[last_idx].passed = False
        suite.passed_cases -= 1
        suite.failed_cases += 1

    var det_ok = (
        not any_desync
        and all_have_10k
        and mismatch_ticks == 0
        and total_rollbacks > 0
        and ROLLBACK_INPUT_HISTORY_K == 8
        and smooth_ok
    )
    if not det_ok:
        rolling_agree_b = rolling_agree_a ^ UInt64(1)

    return suite.record_determinism_gate(
        "rollback_4peer_10000_ticks",
        rolling_agree_a,
        rolling_agree_b,
        elapsed_ms=total_ms,
        budget_ms=1500.0,
        detail="rollbacks=" + String(total_rollbacks) + " mismatches=" + String(mismatch_ticks),
    )


def run_case_replicator_16_clients(mut suite: EngineBenchSuite) -> Bool:
    var t0 = monotonic()
    comptime NUM_CLIENTS: Int = 16
    comptime NUM_ENTITIES: Int = 256
    comptime SNAP_HZ: Int = 20
    comptime DURATION_SEC: Int = 5
    comptime TOTAL_TICKS: Int = SNAP_HZ * DURATION_SEC  # 100 ticks = 5.0s

    var rep = SnapshotReplicator(
        max_entities=NUM_ENTITIES,
        max_clients=NUM_CLIENTS,
        max_packet_bytes=1100,
        interest_radius=95.0,
        cell_size=32.0,
        use_3d_grid=False,
        world_bounds=256.0,
    )
    var receivers = List[SnapshotClientReceiver](capacity=NUM_CLIENTS)
    for c in range(NUM_CLIENTS):
        var cx = (Float32(c % 4) - Float32(1.5)) * Float32(120.0)
        var cz = (Float32(c // 4) - Float32(1.5)) * Float32(120.0)
        rep.set_client_view(c, Vec3(cx, 0.0, cz), 95.0)
        receivers.append(SnapshotClientReceiver(NUM_ENTITIES, 256.0))

    var step_samples_ms = List[Float64](capacity=TOTAL_TICKS)
    var max_pkt_seen = 0
    var decode_ok = True

    for tick in range(TOTAL_TICKS):
        var st0 = monotonic()
        var phase = Float32(tick) * Float32(0.05)
        for e in range(NUM_ENTITIES):
            var gx = (Float32(e % 16) - Float32(7.5)) * Float32(30.0)
            var gz = (Float32(e // 16) - Float32(7.5)) * Float32(30.0)
            # Half the world consists of dynamic movers; half are stationary/sleeping props
            if (e & 1) == 0:
                var ox = det_cos(phase + Float32(e) * Float32(0.3)) * Float32(6.0)
                var oz = det_sin(phase + Float32(e) * Float32(0.3)) * Float32(6.0)
                var vx = -det_sin(phase + Float32(e) * Float32(0.3)) * Float32(6.0)
                var vz = det_cos(phase + Float32(e) * Float32(0.3)) * Float32(6.0)
                var q = Quat.from_euler(0.0, phase * Float32(0.5), 0.0)
                rep.set_entity(e, Vec3(gx + ox, 1.0, gz + oz), q, Vec3(vx, 0.0, vz), UInt32(e & 7))
            else:
                rep.set_entity(e, Vec3(gx, 0.0, gz), Quat.identity(), Vec3(), UInt32(1))

        rep.prepare_tick()

        for c in range(NUM_CLIENTS):
            var built = rep.build_client_packet(c)
            var plen = len(built[1])
            if plen > max_pkt_seen:
                max_pkt_seen = plen
            var dec = receivers[c].apply_packet(built[1])
            if not dec[0]:
                decode_ok = False
            else:
                rep.notify_client_ack(c, dec[1])

        step_samples_ms.append(Float64(monotonic() - st0) / 1.0e6)

    var max_client_bps = Float64(0.0)
    for c in range(NUM_CLIENTS):
        var bps = Float64(rep.clients[c].bytes_sent_total) / Float64(DURATION_SEC)
        if bps > max_client_bps:
            max_client_bps = bps

    var bw_ok = (
        decode_ok
        and max_pkt_seen <= 1100
        and max_client_bps < 24000.0
    )
    var total_ms = Float64(monotonic() - t0) / 1.0e6
    _ = suite.record_samples(
        "replicator_16c_256e_tick",
        step_samples_ms,
        budget_p99_ms=5.0,
        detail="peak_client_Bps=" + String(Int(max_client_bps)) + " (<24000 B/s) max_pkt=" + String(max_pkt_seen),
    )
    return suite.record_single(
        "replicator_bandwidth_gate_24kbs",
        total_ms,
        bw_ok,
        detail="peak_client_KBps=" + String(max_client_bps / 1024.0),
    )


def run_case_bench_suite_flip_and_csv(mut suite: EngineBenchSuite) -> Bool:
    var t0 = monotonic()
    # Verify golden pass + intentional failure Magma FLIP heatmap emission + perf CSV output
    var sub = EngineBenchSuite("self_check")
    var ref_px = List[UInt32](length=16 * 16, fill=0xFF204080)
    var bad_px = List[UInt32](length=16 * 16, fill=0xFFE06010)
    var heat_path = String("/tmp/flint_eng6_lab_net_flip_heat.png")
    var csv_path = String("/tmp/flint_eng6_lab_net_perf.csv")
    _ = remove_file(heat_path)
    _ = remove_file(csv_path)

    var pass_same = sub.verify_golden_pixels("same", ref_px, ref_px, 16, 16)
    var fail_diff = not sub.verify_golden_pixels(
        "diff", bad_px, ref_px, 16, 16, heatmap_out_path=heat_path
    )
    var samples = List[Float64]()
    samples.append(0.15)
    samples.append(0.22)
    _ = sub.record_samples("sample_op", samples)
    var csv_ok = sub.append_perf_history_csv(csv_path, "v5-net")
    var heat_written = sub.results[1].heatmap_path == heat_path

    _ = remove_file(heat_path)
    _ = remove_file(csv_path)

    var elapsed_ms = Float64(monotonic() - t0) / 1.0e6
    return suite.record_single(
        "lab_harness_flip_and_csv",
        elapsed_ms,
        pass_same and fail_diff and heat_written and csv_ok,
        detail="magma_heatmap_and_perf_csv=ok",
    )


def run_case_physics_rollback_and_spoof_defense(
    mut suite: EngineBenchSuite,
) -> Bool:
    var t0 = monotonic()
    comptime TOKEN: UInt64 = 0x504859535F524235
    var sim = VirtualNetSim.from_rtt(
        rtt_ms=50.0,
        jitter_ms=15.0,
        loss_rate=0.05,
        dup_rate=0.01,
        mtu_bytes=1200,
        seed=0x7711223344556677,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var ep0 = sim.create_ipv4_peer(0, 9750)
    var ep1 = sim.create_ipv4_peer(1, 9750)
    var rogue = sim.create_ipv4_peer(2, 41999)
    var addrs = List[NetEndpoint]()
    addrs.append(ep0.local_endpoint())
    addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var s0 = RollbackSession[PhysicsRollbackInput](
        0,
        addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        checksum_interval=4,
        session_token=TOKEN,
    )
    var s1 = RollbackSession[PhysicsRollbackInput](
        1,
        addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        checksum_interval=4,
        session_token=TOKEN,
    )
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    for frame in range(300):
        if frame == 60:
            sim.drop_next_outbound_packets(1, 4)
        if frame >= 10 and (frame % 8) == 0:
            var w = BitWriter(128)
            w.write_u32(_RB_WIRE_MAGIC)
            w.write_u64(TOKEN)
            w.write_u8(UInt8(1))
            w.write_u16(UInt16(frame + 300))
            w.write_u16(UInt16(0))
            w.write_bool(False)
            w.write_u32(UInt32(s0.current_tick + 1))
            w.write_u32(UInt32(s0.current_tick + 1))
            w.write_int_bounded(1, 1, ROLLBACK_INPUT_HISTORY_K)
            w.write_bool(True)
            w.write_u32(UInt32(max(0, (s0.confirmed_tick // 4) * 4)))
            w.write_u64(UInt64(0xDEADBEEF12345678))
            PhysicsRollbackInput(32.0, 16.0, -32.0, 8.0, 15).encode_bits(w)
            var forged = w.finish()
            _ = rogue.send_to(ep0.local_endpoint(), forged)

        var now = sim.now_ms()
        var h0 = mix64(UInt64((s0.current_tick // 3) * 101 + 17))
        var h1 = mix64(UInt64((s1.current_tick // 3) * 131 + 29))
        var c0 = PhysicsRollbackInput(
            (Float32(Int(h0 & 0xFFFF)) / Float32(32767.5) - Float32(1.0))
            * Float32(22.0),
            0.0,
            (
                Float32(Int((h0 >> 16) & 0xFFFF)) / Float32(32767.5)
                - Float32(1.0)
            )
            * Float32(22.0),
            0.0,
            1 if (s0.current_tick % 19 == 0) else 0,
        )
        var c1 = PhysicsRollbackInput(
            (Float32(Int(h1 & 0xFFFF)) / Float32(32767.5) - Float32(1.0))
            * Float32(22.0),
            0.0,
            (
                Float32(Int((h1 >> 16) & 0xFFFF)) / Float32(32767.5)
                - Float32(1.0)
            )
            * Float32(22.0),
            0.0,
            1 if (s1.current_tick % 23 == 0) else 0,
        )
        _ = s0.step_frame(c0, pw0, ep0, now)
        _ = s1.step_frame(c1, pw1, ep1, now)
        if (
            len(s0.confirmed_hash_history) >= 180
            and len(s1.confirmed_hash_history) >= 180
        ):
            break
        sim.advance_ms(1000.0 / 60.0)

    var ok = (
        not s0.desync_detected
        and not s1.desync_detected
        and s0.spoofed_packets_dropped >= 15
        and (s0.rollback_count + s1.rollback_count) > 0
        and len(s0.confirmed_hash_history) >= 180
        and len(s1.confirmed_hash_history) >= 180
    )
    var agree_a = UInt64(0xCBF29CE484222325)
    var agree_b = UInt64(0xCBF29CE484222325)
    if ok:
        for t in range(180):
            agree_a = mix64(agree_a ^ s0.confirmed_hash_history[t])
            agree_b = mix64(agree_b ^ s1.confirmed_hash_history[t])
    else:
        agree_b = agree_a ^ UInt64(1)

    var elapsed_ms = Float64(monotonic() - t0) / 1.0e6
    return suite.record_determinism_gate(
        "physics_world_rollback_spoof_auth",
        agree_a,
        agree_b,
        elapsed_ms=elapsed_ms,
        budget_ms=1000.0,
        detail="spoof_dropped="
        + String(s0.spoofed_packets_dropped)
        + " rb="
        + String(s0.rollback_count + s1.rollback_count),
    )


def run_net_lab_all() -> Bool:
    set_sim_float_mode()
    var suite = EngineBenchSuite("flint_v5_net")
    _ = run_case_dual_stack_udp(suite)
    _ = run_case_bitpack_fuzz(suite)
    _ = run_case_reliable_channel_8kib(suite)
    _ = run_case_rollback_10k_4peer_chaos(suite)
    _ = run_case_physics_rollback_and_spoof_defense(suite)
    _ = run_case_replicator_16_clients(suite)
    _ = run_case_bench_suite_flip_and_csv(suite)
    return suite.finish_and_export("net")


def main() raises:
    if not run_net_lab_all():
        raise Error("flint lab net failed verification gates")
