"""GGPO-Grade Deterministic Rollback Netcode (`flint.net.rollback`).

Implements:
- `trait RollbackInput` & `trait RollbackWorld` + zero-copy `SnapshotReader`
- `SnapshotRingSlot`: O(1) pre-allocated 64-slot (`tick & 63`) ring reusing `BlobWriter` capacity
  (`0` heap allocations after warmup)
- `RollbackSession[InputT]`:
  - Redundant `K = 8` (`[tick - 7 .. tick]`) quantized input history in every UDP datagram
    (survives 7 consecutive dropped packets without losing a single input!)
  - Canonical local input quantization so local and remote peers execute bit-identical inputs
  - Exact misprediction detection, rollback snapshot restore, and fast multi-tick re-simulation
  - Periodic 4-way confirmed-tick `block_hash` (`core/lz4.mojo`) exchange & exact `desync_tick` detector
  - Rift remote tick-advantage estimator (`should_stall_frame()`)
  - Presentation-space exponential `VisualSmoother` (`det_exp`)
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, max, min
from std.time import monotonic
from flint.core.detmath import (
    det_cos,
    det_exp,
    det_sin,
    f32_bits,
    f32_from_bits,
)
from flint.core.lz4 import address_of, hash_list
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.net.bitpack import BitReader, BitWriter
from flint.net.channel import seq_gt
from flint.net.socket_udp import DatagramView, NetEndpoint, NetTransport
from flint.physics.world import PhysicsWorld

comptime ROLLBACK_SNAPSHOT_RING: Int = 64
comptime ROLLBACK_SNAPSHOT_MASK: Int = 63
comptime ROLLBACK_INPUT_HISTORY_K: Int = 8
comptime _INPUT_RING_CAP: Int = 256
comptime _INPUT_RING_MASK: Int = 255
comptime _CHECKSUM_RING_CAP: Int = 256
comptime _CHECKSUM_RING_MASK: Int = 255
comptime _RB_WIRE_MAGIC: UInt32 = 0x50474746  # "FGGP"
comptime DEFAULT_ROLLBACK_SESSION_TOKEN: UInt64 = 0x464C4E545F524235  # "FLNT_RB5"


trait RollbackInput(Deinitable, Copyable, ImplicitlyCopyable, Movable, Defaultable):
    """Quantizable per-tick player input frame for deterministic rollback simulation."""
    def encode_bits(self, mut writer: BitWriter): ...
    def decode_bits(mut self, mut reader: BitReader): ...
    def same_input(self, other: Self) -> Bool: ...


struct SnapshotReader(Copyable, ImplicitlyCopyable, Movable):
    """Zero-allocation binary reader over `(addr, length)` written by `BlobWriter`."""
    var addr: Int
    var length: Int
    var offset: Int
    var failed: Bool

    def __init__(out self, addr: Int, length: Int):
        self.addr = addr
        self.length = max(0, length)
        self.offset = 0
        self.failed = False

    @always_inline
    def u8(mut self) -> UInt8:
        if self.offset >= self.length:
            self.failed = True
            return 0
        var b = Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=self.addr
        )[unsafe_offset=self.offset]
        self.offset += 1
        return b

    @always_inline
    def boolean(mut self) -> Bool:
        return self.u8() != 0

    @always_inline
    def u32(mut self) -> UInt32:
        if self.offset + 4 > self.length:
            self.failed = True
            self.offset = self.length
            return 0
        var v = Pointer[UInt32, ImmutAnyOrigin](
            unsafe_from_address=self.addr + self.offset
        ).unsafe_load[alignment=1]()
        self.offset += 4
        return v

    @always_inline
    def u64(mut self) -> UInt64:
        if self.offset + 8 > self.length:
            self.failed = True
            self.offset = self.length
            return 0
        var v = Pointer[UInt64, ImmutAnyOrigin](
            unsafe_from_address=self.addr + self.offset
        ).unsafe_load[alignment=1]()
        self.offset += 8
        return v

    @always_inline
    def integer(mut self) -> Int:
        return Int(self.u64())

    @always_inline
    def f32(mut self) -> Float32:
        return f32_from_bits(self.u32())

    @always_inline
    def vec3(mut self) -> Vec3:
        var x = self.f32()
        var y = self.f32()
        var z = self.f32()
        return Vec3(x, y, z)

    @always_inline
    def quat(mut self) -> Quat:
        var x = self.f32()
        var y = self.f32()
        var z = self.f32()
        var w = self.f32()
        return Quat(x, y, z, w)


trait RollbackWorld(Movable):
    """Deterministic simulation world supporting O(1) snapshot save/restore and fixed-tick stepping."""
    def save_snapshot(self, mut writer: BlobWriter): ...
    def load_snapshot(mut self, addr: Int, length: Int): ...
    def step_rollback_tick(
        mut self, inputs_addr: Int, peer_count: Int, tick: Int
    ): ...


struct SnapshotRingSlot(Movable):
    """Single slot in the 64-entry snapshot ring reusing `BlobWriter` heap capacity."""
    var tick: Int
    var state_hash: UInt64
    var writer: BlobWriter

    def __init__(out self):
        self.tick = -1
        self.state_hash = 0
        self.writer = BlobWriter()


struct VisualSmoother(Copyable, ImplicitlyCopyable, Movable):
    """Presentation-space exponential error decay (`det_exp`) hiding rollback snap corrections."""
    var pos_offset: Vec3
    var rot_offset: Quat
    var half_life_rate: Float32

    def __init__(out self, decay_rate: Float32 = 18.0):
        self.pos_offset = Vec3(0.0, 0.0, 0.0)
        self.rot_offset = Quat.identity()
        self.half_life_rate = max(Float32(0.5), decay_rate)

    def record_correction(
        mut self,
        prev_visual_pos: Vec3,
        new_auth_pos: Vec3,
        prev_visual_rot: Quat = Quat.identity(),
        new_auth_rot: Quat = Quat.identity(),
    ):
        self.pos_offset = prev_visual_pos - new_auth_pos
        self.rot_offset = (
            prev_visual_rot * new_auth_rot.conjugate()
        ).normalized()

    def step(mut self, dt: Float32):
        if dt <= Float32(0.0):
            return
        var decay = det_exp(-self.half_life_rate * dt)
        self.pos_offset = self.pos_offset * decay
        self.rot_offset = Quat.identity().slerp(self.rot_offset, decay)

    @always_inline
    def smooth_pos(self, auth_pos: Vec3) -> Vec3:
        return auth_pos + self.pos_offset

    @always_inline
    def smooth_rot(self, auth_rot: Quat) -> Quat:
        return (self.rot_offset * auth_rot).normalized()


struct RollbackSession[InputT: RollbackInput](Movable):
    """Multi-peer (`2..8` peers) deterministic GGPO rollback session."""
    var local_peer_id: Int
    var num_peers: Int
    var peer_endpoints: List[NetEndpoint]
    var session_token: UInt64

    var input_delay_ticks: Int
    var max_rollback_ticks: Int
    var tick_dt_ms: Float64
    var checksum_interval: Int

    var current_tick: Int
    var confirmed_tick: Int
    var confirmed_hash: UInt64
    var local_scheduled_tick: Int

    # 64-slot O(1) snapshot ring (`tick & 63`)
    var snapshots: List[SnapshotRingSlot]

    # Per-peer confirmed & predicted input rings (`num_peers * 256`)
    var auth_inputs: List[Self.InputT]
    var auth_input_tick: List[Int]
    var used_inputs: List[Self.InputT]
    var used_input_tick: List[Int]
    var peer_confirmed_tick: List[Int]
    var step_inputs_scratch: List[Self.InputT]

    # Sequence & RTT / Rift tracking per peer
    var local_pkt_seq: UInt16
    var peer_last_seq: List[UInt16]
    var peer_has_seq: List[Bool]
    var peer_remote_tick: List[Int]
    var peer_sent_seq_time_ms: List[Float64]
    var peer_srtt_ms: List[Float64]
    var rift_advantage_ticks: Float32
    var rift_stall_threshold: Float32
    var enable_rift_pacing: Bool
    var _stall_cooldown: Int

    # Confirmed-tick checksum ring + full optional history for audit/fuzz verification
    var local_checksum_tick: List[Int]
    var local_checksum_hash: List[UInt64]
    var remote_checksum_tick: List[Int]
    var remote_checksum_hash: List[UInt64]
    var record_full_hash_history: Bool
    var confirmed_hash_history: List[UInt64]

    var desync_detected: Bool
    var desync_tick: Int
    var desync_peer: Int

    # Reusable zero-alloc scratch buffers
    var tx_bit_writer: BitWriter
    var canon_writer: BitWriter
    var rx_views: List[DatagramView]

    # Telemetry
    var rollback_count: Int
    var resimulated_ticks: Int
    var max_observed_rollback: Int
    var last_resim_ms: Float64
    var peak_resim_ms: Float64
    var packets_sent: Int
    var packets_received: Int
    var spoofed_packets_dropped: Int
    var stall_frames: Int

    def __init__(
        out self,
        local_peer_id: Int,
        peer_endpoints: List[NetEndpoint],
        input_delay_ticks: Int = 2,
        max_rollback_ticks: Int = 8,
        tick_hz: Float64 = 60.0,
        checksum_interval: Int = 8,
        record_full_hash_history: Bool = True,
        session_token: UInt64 = DEFAULT_ROLLBACK_SESSION_TOKEN,
    ):
        self.local_peer_id = local_peer_id
        self.num_peers = max(1, len(peer_endpoints))
        self.peer_endpoints = peer_endpoints.copy()
        self.session_token = (
            session_token
            if session_token != UInt64(0)
            else DEFAULT_ROLLBACK_SESSION_TOKEN
        )

        self.input_delay_ticks = max(0, min(8, input_delay_ticks))
        self.max_rollback_ticks = max(
            1, min(ROLLBACK_SNAPSHOT_RING - 8, max_rollback_ticks)
        )
        self.tick_dt_ms = 1000.0 / max(1.0, tick_hz)
        self.checksum_interval = max(1, checksum_interval)

        self.current_tick = 0
        self.confirmed_tick = -1
        self.confirmed_hash = 0
        self.local_scheduled_tick = self.input_delay_ticks - 1

        self.snapshots = List[SnapshotRingSlot](
            capacity=ROLLBACK_SNAPSHOT_RING
        )
        for _ in range(ROLLBACK_SNAPSHOT_RING):
            self.snapshots.append(SnapshotRingSlot())

        var total_in_slots = self.num_peers * _INPUT_RING_CAP
        self.auth_inputs = List[Self.InputT](
            length=total_in_slots, fill=Self.InputT()
        )
        self.auth_input_tick = List[Int](length=total_in_slots, fill=-1)
        self.used_inputs = List[Self.InputT](
            length=total_in_slots, fill=Self.InputT()
        )
        self.used_input_tick = List[Int](length=total_in_slots, fill=-1)
        self.peer_confirmed_tick = List[Int](
            length=self.num_peers, fill=self.input_delay_ticks - 1
        )
        self.step_inputs_scratch = List[Self.InputT](
            length=self.num_peers, fill=Self.InputT()
        )

        self.local_pkt_seq = 0
        self.peer_last_seq = List[UInt16](length=self.num_peers, fill=0)
        self.peer_has_seq = List[Bool](length=self.num_peers, fill=False)
        self.peer_remote_tick = List[Int](length=self.num_peers, fill=0)
        self.peer_sent_seq_time_ms = List[Float64](length=256, fill=0.0)
        self.peer_srtt_ms = List[Float64](length=self.num_peers, fill=60.0)
        self.rift_advantage_ticks = 0.0
        self.rift_stall_threshold = 2.25
        self.enable_rift_pacing = True
        self._stall_cooldown = 0

        self.local_checksum_tick = List[Int](
            length=_CHECKSUM_RING_CAP, fill=-1
        )
        self.local_checksum_hash = List[UInt64](
            length=_CHECKSUM_RING_CAP, fill=0
        )
        self.remote_checksum_tick = List[Int](
            length=self.num_peers * _CHECKSUM_RING_CAP, fill=-1
        )
        self.remote_checksum_hash = List[UInt64](
            length=self.num_peers * _CHECKSUM_RING_CAP, fill=0
        )
        self.record_full_hash_history = record_full_hash_history
        self.confirmed_hash_history = List[UInt64](capacity=1024)

        self.desync_detected = False
        self.desync_tick = -1
        self.desync_peer = -1

        self.tx_bit_writer = BitWriter(256)
        self.canon_writer = BitWriter(64)
        self.rx_views = List[DatagramView](capacity=64)

        self.rollback_count = 0
        self.resimulated_ticks = 0
        self.max_observed_rollback = 0
        self.last_resim_ms = 0.0
        self.peak_resim_ms = 0.0
        self.packets_sent = 0
        self.packets_received = 0
        self.spoofed_packets_dropped = 0
        self.stall_frames = 0

        # Pre-seed deterministic neutral inputs for initial delay ticks `[0 .. input_delay_ticks - 1]`
        var neutral_canon = self._canonicalize_input(Self.InputT())
        for p in range(self.num_peers):
            for t in range(self.input_delay_ticks):
                self._store_auth_input(p, t, neutral_canon)

    def _canonicalize_input(mut self, raw_in: Self.InputT) -> Self.InputT:
        """Round-trips `raw_in` through bit quantization so local and remote peers step bit-exact values."""
        self.canon_writer.clear()
        raw_in.encode_bits(self.canon_writer)
        self.canon_writer.flush_align()
        var reader = BitReader.from_raw_ptr(
            Int(self.canon_writer.data.unsafe_ptr()),
            len(self.canon_writer.data),
        )
        var out = Self.InputT()
        out.decode_bits(reader)
        return out

    @always_inline
    def _in_idx(self, peer: Int, tick: Int) -> Int:
        return peer * _INPUT_RING_CAP + (tick & _INPUT_RING_MASK)

    @always_inline
    def _store_auth_input(mut self, peer: Int, tick: Int, val: Self.InputT):
        var idx = self._in_idx(peer, tick)
        self.auth_inputs[idx] = val
        self.auth_input_tick[idx] = tick

    @always_inline
    def _has_auth_input(self, peer: Int, tick: Int) -> Bool:
        if tick < 0:
            return True
        return self.auth_input_tick[self._in_idx(peer, tick)] == tick

    @always_inline
    def _get_auth_input(self, peer: Int, tick: Int) -> Self.InputT:
        if tick < 0:
            return Self.InputT()
        var idx = self._in_idx(peer, tick)
        if self.auth_input_tick[idx] == tick:
            return self.auth_inputs[idx]
        return Self.InputT()

    @always_inline
    def _predict_or_auth_input(self, peer: Int, tick: Int) -> Self.InputT:
        var idx = self._in_idx(peer, tick)
        if self.auth_input_tick[idx] == tick:
            return self.auth_inputs[idx]
        var last_c = self.peer_confirmed_tick[peer]
        if last_c >= 0:
            var c_idx = self._in_idx(peer, last_c)
            if self.auth_input_tick[c_idx] == last_c:
                return self.auth_inputs[c_idx]
        return Self.InputT()

    def bind_world[WorldT: RollbackWorld](mut self, world: WorldT):
        """Captures initial `tick = 0` pre-step world snapshot and pre-warms ring capacities."""
        for s in range(ROLLBACK_SNAPSHOT_RING):
            self.snapshots[s].writer.clear()
            if s == 0:
                world.save_snapshot(self.snapshots[0].writer)
                self.snapshots[0].tick = 0
                self.snapshots[0].state_hash = hash_list(
                    self.snapshots[0].writer.data
                )
            else:
                self.snapshots[s].writer.data.reserve(
                    max(64, len(self.snapshots[0].writer.data))
                )
                self.snapshots[s].tick = -1

    def _save_slot[WorldT: RollbackWorld](
        mut self, tick: Int, world: WorldT
    ):
        var slot = tick & ROLLBACK_SNAPSHOT_MASK
        self.snapshots[slot].writer.clear()
        world.save_snapshot(self.snapshots[slot].writer)
        self.snapshots[slot].tick = tick

    def _verify_checksum_pair(
        mut self, peer: Int, tick: Int, local_h: UInt64, remote_h: UInt64
    ):
        if local_h != 0 and remote_h != 0 and local_h != remote_h:
            if not self.desync_detected or tick < self.desync_tick:
                self.desync_detected = True
                self.desync_tick = tick
                self.desync_peer = peer

    def _record_local_confirmed_tick(mut self, tick: Int):
        var after_slot = (tick + 1) & ROLLBACK_SNAPSHOT_MASK
        if self.snapshots[after_slot].tick != tick + 1:
            return
        var h = hash_list(self.snapshots[after_slot].writer.data)
        if h == 0:
            h = 1
        self.snapshots[after_slot].state_hash = h
        self.confirmed_hash = h

        var c_slot = tick & _CHECKSUM_RING_MASK
        self.local_checksum_tick[c_slot] = tick
        self.local_checksum_hash[c_slot] = h

        if self.record_full_hash_history and tick == len(
            self.confirmed_hash_history
        ):
            self.confirmed_hash_history.append(h)

        for p in range(self.num_peers):
            if p == self.local_peer_id:
                continue
            var r_idx = p * _CHECKSUM_RING_CAP + c_slot
            if self.remote_checksum_tick[r_idx] == tick:
                self._verify_checksum_pair(
                    p, tick, h, self.remote_checksum_hash[r_idx]
                )

    def _advance_confirmed_ticks(mut self):
        var min_peer = self.current_tick - 1
        for p in range(self.num_peers):
            if self.peer_confirmed_tick[p] < min_peer:
                min_peer = self.peer_confirmed_tick[p]
        while self.confirmed_tick < min_peer:
            var next_c = self.confirmed_tick + 1
            self._record_local_confirmed_tick(next_c)
            self.confirmed_tick = next_c

    def rollback_and_resimulate[WorldT: RollbackWorld](
        mut self, from_tick: Int, mut world: WorldT
    ):
        """Restores `snapshots[from_tick & 63]` and re-simulates `[from_tick .. current_tick - 1]`."""
        if from_tick < 0 or from_tick >= self.current_tick:
            return
        var depth = self.current_tick - from_tick
        var slot0 = from_tick & ROLLBACK_SNAPSHOT_MASK
        if self.snapshots[slot0].tick != from_tick:
            self.desync_detected = True
            if self.desync_tick < 0:
                self.desync_tick = from_tick
            return

        var t0_ns = monotonic()
        world.load_snapshot(
            address_of(self.snapshots[slot0].writer.data),
            len(self.snapshots[slot0].writer.data),
        )

        for t in range(from_tick, self.current_tick):
            if t > from_tick:
                self._save_slot(t, world)
            for p in range(self.num_peers):
                var inp = self._predict_or_auth_input(p, t)
                self.step_inputs_scratch[p] = inp
                var u_idx = self._in_idx(p, t)
                self.used_inputs[u_idx] = inp
                self.used_input_tick[u_idx] = t
            world.step_rollback_tick(
                Int(self.step_inputs_scratch.unsafe_ptr()), self.num_peers, t
            )

        self._save_slot(self.current_tick, world)
        var elapsed_ms = Float64(monotonic() - t0_ns) / 1.0e6
        self.last_resim_ms = elapsed_ms
        if elapsed_ms > self.peak_resim_ms:
            self.peak_resim_ms = elapsed_ms
        self.rollback_count += 1
        self.resimulated_ticks += depth
        if depth > self.max_observed_rollback:
            self.max_observed_rollback = depth

    def _broadcast_packet[TransT: NetTransport](
        mut self, mut transport: TransT, now_ms: Float64
    ):
        if self.num_peers <= 1 or self.local_scheduled_tick < 0:
            return

        var head_t = self.local_scheduled_tick
        var first_t = max(0, head_t - (ROLLBACK_INPUT_HISTORY_K - 1))
        var count = head_t - first_t + 1

        var pkt_seq = self.local_pkt_seq
        self.local_pkt_seq = UInt16((Int(self.local_pkt_seq) + 1) & 0xFFFF)
        self.peer_sent_seq_time_ms[Int(pkt_seq) & 255] = now_ms

        var send_ck_tick = -1
        var send_ck_hash = UInt64(0)
        if self.confirmed_tick >= 0:
            var aligned_c = (
                self.confirmed_tick // self.checksum_interval
            ) * self.checksum_interval
            var c_slot = aligned_c & _CHECKSUM_RING_MASK
            if self.local_checksum_tick[c_slot] == aligned_c:
                send_ck_tick = aligned_c
                send_ck_hash = self.local_checksum_hash[c_slot]

        for dst_p in range(self.num_peers):
            if dst_p == self.local_peer_id:
                continue
            self.tx_bit_writer.clear()
            self.tx_bit_writer.write_u32(_RB_WIRE_MAGIC)
            self.tx_bit_writer.write_u64(self.session_token)
            self.tx_bit_writer.write_u8(UInt8(self.local_peer_id & 0xFF))
            self.tx_bit_writer.write_u16(pkt_seq)
            self.tx_bit_writer.write_u16(self.peer_last_seq[dst_p])
            self.tx_bit_writer.write_bool(self.peer_has_seq[dst_p])
            self.tx_bit_writer.write_u32(UInt32(self.current_tick))
            self.tx_bit_writer.write_u32(UInt32(first_t))
            self.tx_bit_writer.write_int_bounded(
                count, 1, ROLLBACK_INPUT_HISTORY_K
            )
            var has_ck = send_ck_tick >= 0 and send_ck_hash != 0
            self.tx_bit_writer.write_bool(has_ck)
            if has_ck:
                self.tx_bit_writer.write_u32(UInt32(send_ck_tick))
                self.tx_bit_writer.write_u64(send_ck_hash)

            for k in range(count):
                var inp = self._get_auth_input(self.local_peer_id, first_t + k)
                inp.encode_bits(self.tx_bit_writer)

            self.tx_bit_writer.flush_align()
            if transport.queue_send_raw(
                self.peer_endpoints[dst_p],
                Int(self.tx_bit_writer.data.unsafe_ptr()),
                len(self.tx_bit_writer.data),
            ):
                self.packets_sent += 1

        _ = transport.flush_sends()

    def poll_network[
        WorldT: RollbackWorld, TransT: NetTransport
    ](mut self, mut world: WorldT, mut transport: TransT, now_ms: Float64):
        """Drains incoming UDP packets, triggers rollback re-simulation on mispredictions,
        and advances confirmed tick checksums."""
        var earliest_mispredict = self.current_tick

        while True:
            var n = transport.recv_batch(self.rx_views)
            if n <= 0:
                break
            for i in range(n):
                var view = self.rx_views[i]
                var reader = BitReader.from_raw_ptr(view.data_ptr, view.length)
                var magic = reader.read_u32()
                if reader.overflowed or magic != _RB_WIRE_MAGIC:
                    continue
                var pkt_token = reader.read_u64()
                var sender_id = Int(reader.read_u8())
                if reader.overflowed:
                    continue
                if (
                    pkt_token != self.session_token
                    or sender_id < 0
                    or sender_id >= self.num_peers
                    or sender_id == self.local_peer_id
                    or view.sender != self.peer_endpoints[sender_id]
                ):
                    self.spoofed_packets_dropped += 1
                    continue

                var pkt_seq = reader.read_u16()
                var echo_seq = reader.read_u16()
                var has_echo = reader.read_bool()
                var remote_sim_tick = Int(reader.read_u32())
                var first_t = Int(reader.read_u32())
                var count = reader.read_int_bounded(
                    1, ROLLBACK_INPUT_HISTORY_K
                )
                var has_ck = reader.read_bool()
                var ck_tick = -1
                var ck_hash = UInt64(0)
                if has_ck:
                    ck_tick = Int(reader.read_u32())
                    ck_hash = reader.read_u64()
                if reader.overflowed:
                    continue

                self.packets_received += 1

                if (
                    not self.peer_has_seq[sender_id]
                    or seq_gt(pkt_seq, self.peer_last_seq[sender_id])
                ):
                    self.peer_has_seq[sender_id] = True
                    self.peer_last_seq[sender_id] = pkt_seq
                    self.peer_remote_tick[sender_id] = remote_sim_tick

                if has_echo:
                    var sent_t = self.peer_sent_seq_time_ms[Int(echo_seq) & 255]
                    var rtt = now_ms - sent_t
                    if rtt >= 0.0 and rtt < 2000.0:
                        self.peer_srtt_ms[sender_id] = (
                            self.peer_srtt_ms[sender_id] * 0.875 + rtt * 0.125
                        )

                if has_ck and ck_tick >= 0 and ck_hash != 0:
                    var c_slot = ck_tick & _CHECKSUM_RING_MASK
                    var r_idx = sender_id * _CHECKSUM_RING_CAP + c_slot
                    self.remote_checksum_tick[r_idx] = ck_tick
                    self.remote_checksum_hash[r_idx] = ck_hash
                    if self.local_checksum_tick[c_slot] == ck_tick:
                        self._verify_checksum_pair(
                            sender_id,
                            ck_tick,
                            self.local_checksum_hash[c_slot],
                            ck_hash,
                        )

                for k in range(count):
                    var t = first_t + k
                    var decoded = Self.InputT()
                    decoded.decode_bits(reader)
                    if reader.overflowed:
                        break
                    if t <= self.peer_confirmed_tick[sender_id]:
                        continue
                    # Prevent overwriting far-future slots beyond ring window
                    if t >= self.confirmed_tick + _INPUT_RING_CAP - 8:
                        continue
                    var already_had = self._has_auth_input(sender_id, t)
                    if not already_had:
                        self._store_auth_input(sender_id, t, decoded)
                        if t < self.current_tick:
                            var u_idx = self._in_idx(sender_id, t)
                            if (
                                self.used_input_tick[u_idx] != t
                                or not decoded.same_input(
                                    self.used_inputs[u_idx]
                                )
                            ):
                                if t < earliest_mispredict:
                                    earliest_mispredict = t

                while self._has_auth_input(
                    sender_id, self.peer_confirmed_tick[sender_id] + 1
                ):
                    self.peer_confirmed_tick[sender_id] += 1

        # Update Rift remote tick-advantage estimate against most-lagging peer
        var worst_adv = Float32(-1000.0)
        var any_remote = False
        for p in range(self.num_peers):
            if p == self.local_peer_id or not self.peer_has_seq[p]:
                continue
            any_remote = True
            var one_way_ticks = Float32(
                (self.peer_srtt_ms[p] * 0.5) / self.tick_dt_ms
            )
            var est_remote_now = (
                Float32(self.peer_remote_tick[p]) + one_way_ticks
            )
            var adv = Float32(self.current_tick) - est_remote_now
            if adv > worst_adv:
                worst_adv = adv
        if any_remote:
            self.rift_advantage_ticks = (
                self.rift_advantage_ticks * Float32(0.85)
                + worst_adv * Float32(0.15)
            )

        if earliest_mispredict < self.current_tick:
            self.rollback_and_resimulate(earliest_mispredict, world)

        self._advance_confirmed_ticks()

    def should_stall_frame(mut self) -> Bool:
        """Returns `True` when local simulation must pause 1 frame for prediction cap or Rift pacing."""
        if self.num_peers <= 1:
            return False
        var unconfirmed_span = self.current_tick - (self.confirmed_tick + 1)
        if unconfirmed_span >= self.max_rollback_ticks:
            return True
        if self._stall_cooldown > 0:
            self._stall_cooldown -= 1
            return False
        if (
            self.enable_rift_pacing
            and self.rift_advantage_ticks > self.rift_stall_threshold
        ):
            self._stall_cooldown = 6
            return True
        return False

    def step_frame[
        WorldT: RollbackWorld, TransT: NetTransport
    ](
        mut self,
        local_input: Self.InputT,
        mut world: WorldT,
        mut transport: TransT,
        now_ms: Float64,
    ) -> Bool:
        """Executes one GGPO frame: polls network, rolls back if needed, checks Rift/window stall,
        transmits `K = 8` redundant inputs, and steps simulation if unstalled."""
        if self.snapshots[0].tick != 0 and self.current_tick == 0:
            self.bind_world(world)

        self.poll_network(world, transport, now_ms)

        if self.should_stall_frame():
            self.stall_frames += 1
            self._broadcast_packet(transport, now_ms)
            return False

        var sched_t = self.current_tick + self.input_delay_ticks
        var canon_in = self._canonicalize_input(local_input)
        self._store_auth_input(self.local_peer_id, sched_t, canon_in)
        if sched_t > self.peer_confirmed_tick[self.local_peer_id]:
            self.peer_confirmed_tick[self.local_peer_id] = sched_t
        self.local_scheduled_tick = sched_t

        self._broadcast_packet(transport, now_ms)

        var t = self.current_tick
        if self.snapshots[t & ROLLBACK_SNAPSHOT_MASK].tick != t:
            self._save_slot(t, world)

        for p in range(self.num_peers):
            var inp = self._predict_or_auth_input(p, t)
            self.step_inputs_scratch[p] = inp
            var u_idx = self._in_idx(p, t)
            self.used_inputs[u_idx] = inp
            self.used_input_tick[u_idx] = t

        world.step_rollback_tick(
            Int(self.step_inputs_scratch.unsafe_ptr()), self.num_peers, t
        )
        self.current_tick = t + 1
        self._save_slot(self.current_tick, world)
        self._advance_confirmed_ticks()
        return True


# ============================================================================
# Concrete `PhysicsWorld` -> `RollbackWorld` Conformance (`PhysicsRollbackAdapter`)
# ============================================================================


@fieldwise_init
struct PhysicsRollbackInput(RollbackInput):
    """Quantized per-tick rigid-body player control input for `PhysicsRollbackAdapter`."""
    var impulse_x: Float32
    var impulse_y: Float32
    var impulse_z: Float32
    var torque_yaw: Float32
    var action_bits: Int

    def __init__(out self):
        self.impulse_x = 0.0
        self.impulse_y = 0.0
        self.impulse_z = 0.0
        self.torque_yaw = 0.0
        self.action_bits = 0

    def encode_bits(self, mut writer: BitWriter):
        writer.write_f32_quantized(self.impulse_x, -32.0, 32.0, 12)
        writer.write_f32_quantized(self.impulse_y, -32.0, 32.0, 12)
        writer.write_f32_quantized(self.impulse_z, -32.0, 32.0, 12)
        writer.write_f32_quantized(self.torque_yaw, -16.0, 16.0, 10)
        writer.write_int_bounded(self.action_bits, 0, 15)

    def decode_bits(mut self, mut reader: BitReader):
        self.impulse_x = reader.read_f32_quantized(-32.0, 32.0, 12)
        self.impulse_y = reader.read_f32_quantized(-32.0, 32.0, 12)
        self.impulse_z = reader.read_f32_quantized(-32.0, 32.0, 12)
        self.torque_yaw = reader.read_f32_quantized(-16.0, 16.0, 10)
        self.action_bits = reader.read_int_bounded(0, 15)

    def same_input(self, other: Self) -> Bool:
        return (
            f32_bits(self.impulse_x) == f32_bits(other.impulse_x)
            and f32_bits(self.impulse_y) == f32_bits(other.impulse_y)
            and f32_bits(self.impulse_z) == f32_bits(other.impulse_z)
            and f32_bits(self.torque_yaw) == f32_bits(other.torque_yaw)
            and self.action_bits == other.action_bits
        )


struct PhysicsRollbackAdapter(RollbackWorld):
    """Drives a real `flint.physics.world.PhysicsWorld` inside `RollbackSession` via
    bit-exact `'PHYW'` `PhysicsWorld.serialize` / `PhysicsWorld.deserialize` snapshots."""
    var world: PhysicsWorld
    var player_body_ids: List[Int]
    var fixed_dt: Float32

    def __init__(
        out self,
        var world: PhysicsWorld,
        player_body_ids: List[Int],
        fixed_dt: Float32 = Float32(1.0 / 60.0),
    ):
        self.world = world^
        self.world.settings.force_serial = True
        self.player_body_ids = player_body_ids.copy()
        self.fixed_dt = fixed_dt

    @staticmethod
    def create_arena(
        num_peers: Int = 2,
        fixed_dt: Float32 = Float32(1.0 / 60.0),
    ) -> PhysicsRollbackAdapter:
        """Builds a walled rigid-body arena populated with player spheres and shared
        colliding dynamic boxes & spheres for deterministic multi-peer rollback."""
        var pw = PhysicsWorld(gravity=Float32(-9.81))
        pw.settings.force_serial = True

        # Static arena floor + 4 containment walls
        _ = pw.add_box(
            Vec3(0.0, -0.5, 0.0),
            Vec3(18.0, 0.5, 18.0),
            mass=0.0,
            friction=0.65,
            restitution=0.35,
        )
        _ = pw.add_box(
            Vec3(14.0, 2.0, 0.0),
            Vec3(0.5, 2.5, 14.5),
            mass=0.0,
            friction=0.4,
            restitution=0.7,
        )
        _ = pw.add_box(
            Vec3(-14.0, 2.0, 0.0),
            Vec3(0.5, 2.5, 14.5),
            mass=0.0,
            friction=0.4,
            restitution=0.7,
        )
        _ = pw.add_box(
            Vec3(0.0, 2.0, 14.0),
            Vec3(14.5, 2.5, 0.5),
            mass=0.0,
            friction=0.4,
            restitution=0.7,
        )
        _ = pw.add_box(
            Vec3(0.0, 2.0, -14.0),
            Vec3(14.5, 2.5, 0.5),
            mass=0.0,
            friction=0.4,
            restitution=0.7,
        )

        # Player-controlled dynamic rigid-body spheres
        var pids = List[Int](capacity=max(1, num_peers))
        for p in range(max(1, num_peers)):
            var ang = Float32(p) * (
                Float32(6.2831853) / Float32(max(1, num_peers))
            )
            var slot = pw.add_sphere(
                Vec3(det_cos(ang) * 5.0, 1.0, det_sin(ang) * 5.0),
                radius=0.75,
                mass=2.0,
                friction=0.45,
                restitution=0.65,
            )
            pw.bodies[slot].linear_damping = 0.25
            pw.bodies[slot].angular_damping = 0.25
            pw.bodies[slot].can_sleep = False
            pids.append(pw.body_id(slot))

        # Shared dynamic colliding boxes & spheres in the center of the arena
        for b_idx in range(4):
            var bang = Float32(b_idx) * Float32(1.5707963) + Float32(0.4)
            var b_slot = pw.add_box(
                Vec3(
                    det_cos(bang) * 1.8,
                    0.65 + Float32(b_idx & 1) * 1.35,
                    det_sin(bang) * 1.8,
                ),
                Vec3(0.55, 0.55, 0.55),
                mass=1.4,
                rot=Quat.from_euler(0.0, bang, 0.0),
                friction=0.55,
                restitution=0.45,
            )
            pw.bodies[b_slot].linear_damping = 0.15
        var center_sphere = pw.add_sphere(
            Vec3(0.0, 2.6, 0.0),
            radius=0.65,
            mass=1.8,
            friction=0.4,
            restitution=0.7,
        )
        pw.bodies[center_sphere].linear_damping = 0.12

        return PhysicsRollbackAdapter(pw^, pids, fixed_dt=fixed_dt)

    def save_snapshot(self, mut writer: BlobWriter):
        self.world.serialize(writer)

    def load_snapshot(mut self, addr: Int, length: Int):
        if addr == 0 or length <= 0:
            return
        var bytes = List[UInt8](unsafe_uninit_length=length)
        _ = external_call["memcpy", Int](
            Int(bytes.unsafe_ptr()), addr, length
        )
        var reader = BlobReader(bytes^)
        try:
            self.world.deserialize(reader)
            self.world.settings.force_serial = True
        except:
            pass

    def step_rollback_tick(
        mut self, inputs_addr: Int, peer_count: Int, tick: Int
    ):
        _ = tick
        var in_ptr = Pointer[PhysicsRollbackInput, ImmutAnyOrigin](
            unsafe_from_address=inputs_addr
        )
        var n_players = min(peer_count, len(self.player_body_ids))
        for p in range(n_players):
            var slot = self.world.find_slot_by_id(self.player_body_ids[p])
            if slot < 0:
                continue
            var cmd = in_ptr[unsafe_offset=p]
            var imp = (
                Vec3(cmd.impulse_x, cmd.impulse_y, cmd.impulse_z)
                * self.fixed_dt
            )
            if (cmd.action_bits & 1) != 0:
                imp = imp + Vec3(
                    cmd.impulse_x * Float32(0.14),
                    Float32(2.2),
                    cmd.impulse_z * Float32(0.14),
                )
            if imp.length_squared() > Float32(0.0):
                self.world.apply_impulse(slot, imp)
            if abs(cmd.torque_yaw) > Float32(1.0e-4):
                var wp = self.world.bodies[slot].pos + Vec3(0.0, 0.0, 0.5)
                self.world.apply_impulse_at(
                    slot,
                    Vec3(cmd.torque_yaw * self.fixed_dt, 0.0, 0.0),
                    wp,
                )

        self.world.step(self.fixed_dt)
