"""UDP Packet Framing, 65-Packet ACK Window, EWMA RTT & Reliable-Ordered Fragmentation (`flint.net.channel`).

Implements:
- Wire header (`magic_ver`, `chan_flags`, 16-bit RFC-1982 `seq`, `ack`, `ack_bits: UInt64` 65-packet
  selective ACK window)
- Karn-free per-wire-packet RTT & jitter EWMA estimator with adaptive RTO
- `UNRELIABLE_SEQUENCED` latest-only channel (drops stale out-of-order datagrams)
- `RELIABLE_ORDERED` sliding-window retransmission channel with automatic `1180 B` IPv6-safe MTU
  fragmentation and ordered multi-fragment reassembly.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isfinite, max, min
from flint.net.crypto import (
    AEAD_NONCE_BYTES,
    AEAD_OVERHEAD_BYTES,
    AEAD_TAG_BYTES,
    AeadKey,
    ReplayWindow64,
    aead_open_raw,
    aead_seal_raw,
)
from flint.net.socket_udp import DatagramView, NetEndpoint, NetTransport

comptime WIRE_MAGIC_VER: UInt8 = 0x56  # 'V' (Flint v5 wire protocol v1)
comptime WIRE_HEADER_BYTES: Int = 14
comptime WIRE_FLAG_HAS_REMOTE_SEQ: UInt8 = 0x80
comptime WIRE_FLAG_ENCRYPTED: UInt8 = 0x40
comptime IPV6_SAFE_MTU_BYTES: Int = 1180
comptime MAX_FRAGMENT_PAYLOAD_BYTES: Int = 1024
comptime MAX_RELIABLE_FRAG_COUNT: Int = 4096
comptime MAX_RELIABLE_MESSAGE_BYTES: Int = 4 * 1024 * 1024
comptime MAX_INCOMING_RELIABLE_MESSAGES: Int = 256
comptime MAX_INCOMING_RELIABLE_BYTES: Int = 8 * 1024 * 1024
comptime MAX_INCOMING_UNRELIABLE_MESSAGES: Int = 256

comptime CHAN_UNRELIABLE_RAW: UInt8 = 0
comptime CHAN_UNRELIABLE_SEQUENCED: UInt8 = 1
comptime CHAN_RELIABLE_ORDERED: UInt8 = 2
comptime CHAN_ACK_HEARTBEAT: UInt8 = 3
comptime CHAN_RELIABLE_UNORDERED: UInt8 = 4
comptime CHAN_HANDSHAKE_INIT: UInt8 = 5
comptime CHAN_HANDSHAKE_COOKIE: UInt8 = 6
comptime CHAN_HANDSHAKE_ECHO: UInt8 = 7

comptime _SENT_TRACK_SIZE: Int = 256
comptime _RELIABLE_WINDOW: Int = 256
comptime _FRAG_HEADER_BYTES: Int = 14


@always_inline
def seq_gt(a: UInt16, b: UInt16) -> Bool:
    """RFC-1982 16-bit serial number comparison `a > b`."""
    return ((a > b) and (a - b <= 32768)) or ((a < b) and (b - a > 32768))


@always_inline
def seq_ge(a: UInt16, b: UInt16) -> Bool:
    return a == b or seq_gt(a, b)


@always_inline
def seq_diff(newer: UInt16, older: UInt16) -> Int:
    """Signed 16-bit wraparound difference `newer - older` in `[-32767, 32768]`."""
    var d = Int(newer) - Int(older)
    if d > 32768:
        d -= 65536
    elif d < -32767:
        d += 65536
    return d


@always_inline
def _put_u16(mut buf: List[UInt8], v: UInt16):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


@always_inline
def _put_u64(mut buf: List[UInt8], v: UInt64):
    for s in range(8):
        buf.append(UInt8((v >> UInt64(s * 8)) & 0xFF))


@always_inline
def _get_u16_ptr(ptr: Int, off: Int) -> UInt16:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=ptr + off)
    return UInt16(p[unsafe_offset=0]) | (UInt16(p[unsafe_offset=1]) << 8)


@always_inline
def _get_u64_ptr(ptr: Int, off: Int) -> UInt64:
    return Pointer[UInt64, ImmutAnyOrigin](
        unsafe_from_address=ptr + off
    ).unsafe_load[alignment=1]()


@fieldwise_init
struct PacketWireHeader(Copyable, ImplicitlyCopyable, Movable):
    """Wire packet header with 16-bit RFC-1982 sequence and 65-packet (`ack` + 64 `ack_bits`) window."""
    var magic_ver: UInt8
    var chan_flags: UInt8
    var seq: UInt16
    var ack: UInt16
    var ack_bits: UInt64

    def __init__(out self):
        self.magic_ver = WIRE_MAGIC_VER
        self.chan_flags = 0
        self.seq = 0
        self.ack = 0
        self.ack_bits = 0

    def write_into(self, mut out: List[UInt8]):
        out.append(self.magic_ver)
        out.append(self.chan_flags)
        _put_u16(out, self.seq)
        _put_u16(out, self.ack)
        _put_u64(out, self.ack_bits)

    @staticmethod
    def parse_from_ptr(data_ptr: Int, length: Int) -> Tuple[Bool, Self]:
        if data_ptr == 0 or length < WIRE_HEADER_BYTES:
            return (False, Self())
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data_ptr)
        var mv = p[unsafe_offset=0]
        if mv != WIRE_MAGIC_VER:
            return (False, Self())
        var cf = p[unsafe_offset=1]
        var seq = _get_u16_ptr(data_ptr, 2)
        var ack = _get_u16_ptr(data_ptr, 4)
        var abits = _get_u64_ptr(data_ptr, 6)
        return (True, Self(mv, cf, seq, ack, abits))


struct _SentPacketRecord(Copyable, ImplicitlyCopyable, Movable):
    var occupied: Bool
    var wire_seq: UInt16
    var acked: Bool
    var sent_time_ms: Float64
    var has_rel_frag: Bool
    var rel_seq: UInt16

    def __init__(out self):
        self.occupied = False
        self.wire_seq = 0
        self.acked = False
        self.sent_time_ms = 0.0
        self.has_rel_frag = False
        self.rel_seq = 0


struct _ReliableOutFrag(Copyable, Movable):
    var active: Bool
    var acked: Bool
    var unordered: Bool
    var rel_seq: UInt16
    var msg_id: UInt16
    var frag_idx: UInt16
    var frag_count: UInt16
    var last_sent_ms: Float64
    var send_count: Int
    var payload: List[UInt8]

    def __init__(out self):
        self.active = False
        self.acked = False
        self.unordered = False
        self.rel_seq = 0
        self.msg_id = 0
        self.frag_idx = 0
        self.frag_count = 0
        self.last_sent_ms = -1.0e9
        self.send_count = 0
        self.payload = List[UInt8]()


struct _ReliableInFrag(Copyable, Movable):
    var received: Bool
    var unordered: Bool
    var rel_seq: UInt16
    var msg_id: UInt16
    var frag_idx: UInt16
    var frag_count: UInt16
    var payload: List[UInt8]

    def __init__(out self):
        self.received = False
        self.unordered = False
        self.rel_seq = 0
        self.msg_id = 0
        self.frag_idx = 0
        self.frag_count = 0
        self.payload = List[UInt8]()


struct _UnorderedAssembly(Copyable, Movable):
    var active: Bool
    var msg_id: UInt16
    var frag_count: UInt16
    var received_count: Int
    var total_bytes: Int
    var frag_got: List[Bool]
    var frag_data: List[List[UInt8]]

    def __init__(out self):
        self.active = False
        self.msg_id = 0
        self.frag_count = 0
        self.received_count = 0
        self.total_bytes = 0
        self.frag_got = List[Bool]()
        self.frag_data = List[List[UInt8]]()


struct NetChannel(Movable):
    """Point-to-point reliable/unreliable multiplexed UDP channel with `1180 B` IPv6-safe MTU."""
    var peer_endpoint: NetEndpoint
    var max_mtu_bytes: Int

    var local_wire_seq: UInt16
    var has_remote_wire_seq: Bool
    var remote_wire_seq: UInt16
    var recv_ack_bits: UInt64

    var srtt_ms: Float64
    var jitter_ms: Float64
    var rto_ms: Float64
    var rtt_initialized: Bool

    # AIMD congestion control & exponential RTO backoff
    var cwnd: Float64
    var ssthresh: Float64
    var _last_loss_ms: Float64

    # Optional ChaCha20-Poly1305 AEAD with 64-bit replay-protected nonces
    var crypto_enabled: Bool
    var aead_key: AeadKey
    var tx_nonce: UInt64
    var rx_replay: ReplayWindow64
    var crypto_auth_failures: Int
    var crypto_replay_drops: Int
    var _crypto_scratch: List[UInt8]
    var _decrypt_scratch: List[UInt8]

    var sent_records: List[_SentPacketRecord]
    var newly_acked_wire_seqs: List[UInt16]

    # Unreliable-sequenced & raw state (head-indexed O(1) queues)
    var next_unrel_seq: UInt16
    var has_remote_unrel_seq: Bool
    var last_remote_unrel_seq: UInt16
    var incoming_unreliable: List[List[UInt8]]
    var incoming_unreliable_head: Int
    var incoming_unreliable_raw: List[List[UInt8]]
    var incoming_unreliable_raw_head: Int

    # Reliable-ordered & unordered sliding window & fragmentation state
    var next_msg_id: UInt16
    var next_send_rel_seq: UInt16
    var oldest_unacked_rel_seq: UInt16
    var out_frags: List[_ReliableOutFrag]
    var pending_out_queue: List[_ReliableOutFrag]
    var pending_out_head: Int

    var next_expected_rel_seq: UInt16
    var in_frags: List[_ReliableInFrag]
    var incoming_reliable: List[List[UInt8]]
    var incoming_reliable_head: Int
    var incoming_reliable_bytes: Int
    var incoming_reliable_unordered: List[List[UInt8]]
    var incoming_reliable_unordered_head: Int
    var incoming_reliable_unordered_bytes: Int
    var unordered_assemblies: List[_UnorderedAssembly]
    var needs_ack_flush: Bool

    # Reassembly staging for current head message
    var assembling_active: Bool
    var assembling_msg_id: UInt16
    var assembling_expected_idx: UInt16
    var assembling_frag_count: UInt16
    var assembling_bytes: List[UInt8]

    var tx_scratch: List[UInt8]
    var rx_views_scratch: List[DatagramView]

    var packets_sent: Int
    var packets_received: Int
    var retransmissions: Int
    var duplicate_packets_ignored: Int

    def __init__(
        out self,
        peer_endpoint: NetEndpoint = NetEndpoint(),
        max_mtu_bytes: Int = IPV6_SAFE_MTU_BYTES,
    ):
        self.peer_endpoint = peer_endpoint
        self.max_mtu_bytes = min(1400, max(256, max_mtu_bytes))

        self.local_wire_seq = 0
        self.has_remote_wire_seq = False
        self.remote_wire_seq = 0
        self.recv_ack_bits = 0

        self.srtt_ms = 60.0
        self.jitter_ms = 15.0
        self.rto_ms = 100.0
        self.rtt_initialized = False

        self.cwnd = 16.0
        self.ssthresh = 64.0
        self._last_loss_ms = -1.0e9

        self.crypto_enabled = False
        self.aead_key = AeadKey()
        self.tx_nonce = 1
        self.rx_replay = ReplayWindow64()
        self.crypto_auth_failures = 0
        self.crypto_replay_drops = 0
        var scratch_cap = max(
            IPV6_SAFE_MTU_BYTES, self.max_mtu_bytes + AEAD_OVERHEAD_BYTES
        )
        self._crypto_scratch = List[UInt8](capacity=scratch_cap)
        self._decrypt_scratch = List[UInt8](capacity=scratch_cap)

        self.sent_records = List[_SentPacketRecord](
            length=_SENT_TRACK_SIZE, fill=_SentPacketRecord()
        )
        self.newly_acked_wire_seqs = List[UInt16](capacity=64)

        self.next_unrel_seq = 0
        self.has_remote_unrel_seq = False
        self.last_remote_unrel_seq = 0
        self.incoming_unreliable = List[List[UInt8]]()
        self.incoming_unreliable_head = 0
        self.incoming_unreliable_raw = List[List[UInt8]]()
        self.incoming_unreliable_raw_head = 0

        self.next_msg_id = 0
        self.next_send_rel_seq = 0
        self.oldest_unacked_rel_seq = 0
        self.out_frags = List[_ReliableOutFrag]()
        for _ in range(_RELIABLE_WINDOW):
            self.out_frags.append(_ReliableOutFrag())
        self.pending_out_queue = List[_ReliableOutFrag]()
        self.pending_out_head = 0

        self.next_expected_rel_seq = 0
        self.in_frags = List[_ReliableInFrag]()
        for _ in range(_RELIABLE_WINDOW):
            self.in_frags.append(_ReliableInFrag())
        self.incoming_reliable = List[List[UInt8]]()
        self.incoming_reliable_head = 0
        self.incoming_reliable_bytes = 0
        self.incoming_reliable_unordered = List[List[UInt8]]()
        self.incoming_reliable_unordered_head = 0
        self.incoming_reliable_unordered_bytes = 0
        self.unordered_assemblies = List[_UnorderedAssembly]()
        self.needs_ack_flush = False

        self.assembling_active = False
        self.assembling_msg_id = 0
        self.assembling_expected_idx = 0
        self.assembling_frag_count = 0
        self.assembling_bytes = List[UInt8]()

        self.tx_scratch = List[UInt8](capacity=IPV6_SAFE_MTU_BYTES)
        self.rx_views_scratch = List[DatagramView](capacity=64)

        self.packets_sent = 0
        self.packets_received = 0
        self.retransmissions = 0
        self.duplicate_packets_ignored = 0

    def enable_encryption(mut self, key: AeadKey):
        """Enables ChaCha20-Poly1305 AEAD encryption with 64-bit replay-protected nonces."""
        self.crypto_enabled = True
        self.aead_key = key
        self.tx_nonce = 1
        self.rx_replay.reset()

    def _make_header(mut self, chan_kind: UInt8) -> PacketWireHeader:
        var seq = self.local_wire_seq
        self.local_wire_seq = UInt16((Int(self.local_wire_seq) + 1) & 0xFFFF)
        var flags = chan_kind & 0x0F
        if self.has_remote_wire_seq:
            flags |= 0x80
        return PacketWireHeader(
            WIRE_MAGIC_VER,
            flags,
            seq,
            self.remote_wire_seq,
            self.recv_ack_bits,
        )

    def _record_sent_packet(
        mut self,
        wire_seq: UInt16,
        now_ms: Float64,
        has_rel: Bool = False,
        rel_seq: UInt16 = 0,
    ):
        var slot = Int(wire_seq) & (_SENT_TRACK_SIZE - 1)
        self.sent_records[slot] = _SentPacketRecord()
        if not isfinite(now_ms):
            return
        self.sent_records[slot].occupied = True
        self.sent_records[slot].wire_seq = wire_seq
        self.sent_records[slot].acked = False
        self.sent_records[slot].sent_time_ms = now_ms
        self.sent_records[slot].has_rel_frag = has_rel
        self.sent_records[slot].rel_seq = rel_seq

    def _update_rtt(mut self, sample_ms: Float64):
        if not isfinite(sample_ms) or sample_ms < 0.0 or sample_ms > 10000.0:
            return
        if not self.rtt_initialized:
            self.srtt_ms = sample_ms
            self.jitter_ms = max(2.0, sample_ms * 0.25)
            self.rtt_initialized = True
        else:
            var delta = sample_ms - self.srtt_ms
            self.srtt_ms += 0.125 * delta
            self.jitter_ms += 0.125 * (abs(delta) - self.jitter_ms)
        self.rto_ms = max(
            25.0, min(600.0, self.srtt_ms + 4.0 * max(4.0, self.jitter_ms))
        )

    def _mark_wire_seq_acked(mut self, acked_seq: UInt16, now_ms: Float64):
        var slot = Int(acked_seq) & (_SENT_TRACK_SIZE - 1)
        if (
            self.sent_records[slot].occupied
            and self.sent_records[slot].wire_seq == acked_seq
            and not self.sent_records[slot].acked
        ):
            self.sent_records[slot].acked = True
            self.newly_acked_wire_seqs.append(acked_seq)
            if isfinite(now_ms) and isfinite(self.sent_records[slot].sent_time_ms):
                self._update_rtt(now_ms - self.sent_records[slot].sent_time_ms)
            if self.sent_records[slot].has_rel_frag:
                self._mark_rel_seq_acked(self.sent_records[slot].rel_seq)

    def _mark_rel_seq_acked(mut self, rseq: UInt16):
        var idx = Int(rseq) & (_RELIABLE_WINDOW - 1)
        if (
            self.out_frags[idx].active
            and self.out_frags[idx].rel_seq == rseq
        ):
            if not self.out_frags[idx].acked:
                self.out_frags[idx].acked = True
                # AIMD congestion window growth (slow start -> congestion avoidance)
                var max_cwnd = Float64(_RELIABLE_WINDOW - 8)
                if self.cwnd < self.ssthresh:
                    self.cwnd = min(max_cwnd, self.cwnd + 1.0)
                else:
                    self.cwnd = min(max_cwnd, self.cwnd + 1.0 / max(1.0, self.cwnd))
        while True:
            var head_idx = Int(self.oldest_unacked_rel_seq) & (
                _RELIABLE_WINDOW - 1
            )
            if (
                self.out_frags[head_idx].active
                and self.out_frags[head_idx].rel_seq
                == self.oldest_unacked_rel_seq
                and self.out_frags[head_idx].acked
            ):
                self.out_frags[head_idx].active = False
                self.out_frags[head_idx].acked = False
                self.out_frags[head_idx].payload.clear()
                self.oldest_unacked_rel_seq = UInt16(
                    (Int(self.oldest_unacked_rel_seq) + 1) & 0xFFFF
                )
            else:
                break
        self._promote_pending_frags()

    def _process_remote_acks(
        mut self, ack: UInt16, ack_bits: UInt64, now_ms: Float64
    ):
        self._mark_wire_seq_acked(ack, now_ms)
        for bit in range(64):
            if ((ack_bits >> UInt64(bit)) & UInt64(1)) != 0:
                var past_seq = UInt16((Int(ack) - 1 - bit) & 0xFFFF)
                self._mark_wire_seq_acked(past_seq, now_ms)

    def _record_remote_wire_seq(mut self, seq: UInt16) -> Bool:
        """Updates the 65-packet `recv_ack_bits` window; returns `False` if duplicate `seq`."""
        if not self.has_remote_wire_seq:
            self.has_remote_wire_seq = True
            self.remote_wire_seq = seq
            self.recv_ack_bits = 0
            return True
        if seq == self.remote_wire_seq:
            self.duplicate_packets_ignored += 1
            return False
        if seq_gt(seq, self.remote_wire_seq):
            var diff = Int((Int(seq) - Int(self.remote_wire_seq)) & 0xFFFF)
            if diff >= 65:
                self.recv_ack_bits = 0
            elif diff == 64:
                self.recv_ack_bits = UInt64(1) << 63
            else:
                self.recv_ack_bits = (
                    self.recv_ack_bits << UInt64(diff)
                ) | (UInt64(1) << UInt64(diff - 1))
            self.remote_wire_seq = seq
            return True
        else:
            var age = Int((Int(self.remote_wire_seq) - Int(seq)) & 0xFFFF)
            if age >= 1 and age <= 64:
                var mask = UInt64(1) << UInt64(age - 1)
                if (self.recv_ack_bits & mask) != 0:
                    self.duplicate_packets_ignored += 1
                    return False
                self.recv_ack_bits |= mask
            return True

    def _rel_recv_sack_bits(self) -> UInt64:
        var bits = UInt64(0)
        for k in range(64):
            var s = UInt16((Int(self.next_expected_rel_seq) + 1 + k) & 0xFFFF)
            var slot = Int(s) & (_RELIABLE_WINDOW - 1)
            if self.in_frags[slot].received and self.in_frags[slot].rel_seq == s:
                bits |= UInt64(1) << UInt64(k)
        return bits

    def _process_rel_sack(mut self, cum_rel_ack: UInt16, rel_sack_bits: UInt64):
        # Reject spoofed future cumulative ACKs beyond what we have actually sent.
        if seq_gt(cum_rel_ack, self.next_send_rel_seq):
            return
        # Cumulative reliable ACK: every rel_seq strictly before `cum_rel_ack` was delivered!
        while (
            seq_gt(cum_rel_ack, self.oldest_unacked_rel_seq)
            and seq_gt(self.next_send_rel_seq, self.oldest_unacked_rel_seq)
        ):
            var prev = self.oldest_unacked_rel_seq
            var head_idx = Int(prev) & (_RELIABLE_WINDOW - 1)
            if (
                not self.out_frags[head_idx].active
                or self.out_frags[head_idx].rel_seq != prev
            ):
                break
            self._mark_rel_seq_acked(prev)
            if self.oldest_unacked_rel_seq == prev:
                break
        # Selective ACK bits for `cum_rel_ack + 1 + k`
        for k in range(64):
            if ((rel_sack_bits >> UInt64(k)) & UInt64(1)) != 0:
                var rseq = UInt16((Int(cum_rel_ack) + 1 + k) & 0xFFFF)
                if seq_gt(self.next_send_rel_seq, rseq):
                    self._mark_rel_seq_acked(rseq)

    def _promote_pending_frags(mut self):
        while self.pending_out_head < len(self.pending_out_queue):
            var in_window = seq_diff(
                self.next_send_rel_seq, self.oldest_unacked_rel_seq
            )
            if in_window >= (_RELIABLE_WINDOW - 8):
                break
            var frag = self.pending_out_queue[self.pending_out_head].copy()
            self.pending_out_queue[self.pending_out_head].payload.clear()
            self.pending_out_head += 1

            var rseq = self.next_send_rel_seq
            self.next_send_rel_seq = UInt16(
                (Int(self.next_send_rel_seq) + 1) & 0xFFFF
            )
            frag.rel_seq = rseq
            frag.active = True
            frag.acked = False
            var slot = Int(rseq) & (_RELIABLE_WINDOW - 1)
            self.out_frags[slot] = frag^

        if self.pending_out_head >= len(self.pending_out_queue):
            self.pending_out_queue.clear()
            self.pending_out_head = 0
        elif (
            self.pending_out_head >= 64
            and self.pending_out_head * 2 >= len(self.pending_out_queue)
        ):
            var rem = len(self.pending_out_queue) - self.pending_out_head
            for i in range(rem):
                self.pending_out_queue[i] = self.pending_out_queue[
                    self.pending_out_head + i
                ].copy()
            while len(self.pending_out_queue) > rem:
                _ = self.pending_out_queue.pop()
            self.pending_out_head = 0

    def _emit_tx_scratch[
        TransT: NetTransport
    ](mut self, mut transport: TransT, immediate: Bool) -> Bool:
        """Sends or queues `self.tx_scratch`, wrapping in ChaCha20-Poly1305 AEAD if enabled."""
        var send_ptr: Int
        var send_len: Int
        if not self.crypto_enabled:
            send_ptr = Int(self.tx_scratch.unsafe_ptr())
            send_len = len(self.tx_scratch)
        else:
            if len(self.tx_scratch) < WIRE_HEADER_BYTES:
                return False
            var pt_len = len(self.tx_scratch) - WIRE_HEADER_BYTES
            self._crypto_scratch.clear()
            self._crypto_scratch.reserve(
                WIRE_HEADER_BYTES + AEAD_NONCE_BYTES + max(0, pt_len) + AEAD_TAG_BYTES
            )
            # Copy 14-byte wire header with WIRE_FLAG_ENCRYPTED set
            for i in range(WIRE_HEADER_BYTES):
                self._crypto_scratch.append(self.tx_scratch[i])
            self._crypto_scratch[1] |= WIRE_FLAG_ENCRYPTED
            # Append 8-byte monotonic nonce
            var nonce = self.tx_nonce
            self.tx_nonce += 1
            _put_u64(self._crypto_scratch, nonce)
            var pt_ptr = (
                Int(self.tx_scratch.unsafe_ptr()) + WIRE_HEADER_BYTES
                if pt_len > 0
                else 0
            )
            var aad_ptr = Int(self._crypto_scratch.unsafe_ptr())
            var n1 = UInt32(nonce & 0xFFFFFFFF)
            var n2 = UInt32(nonce >> 32)
            aead_seal_raw(
                self.aead_key,
                UInt32(0),
                n1,
                n2,
                aad_ptr,
                WIRE_HEADER_BYTES + AEAD_NONCE_BYTES,
                pt_ptr,
                pt_len,
                self._crypto_scratch,
            )
            send_ptr = Int(self._crypto_scratch.unsafe_ptr())
            send_len = len(self._crypto_scratch)

        var ok: Bool
        if immediate:
            ok = transport.send_raw(self.peer_endpoint, send_ptr, send_len)
        else:
            ok = transport.queue_send_raw(self.peer_endpoint, send_ptr, send_len)
        _ = len(self.tx_scratch)
        _ = len(self._crypto_scratch)
        return ok

    def send_unreliable_raw[
        TransT: NetTransport
    ](
        mut self,
        payload: List[UInt8],
        mut transport: TransT,
        now_ms: Float64,
    ) -> Tuple[Bool, UInt16]:
        """Sends an unreliable-raw datagram (`<= MTU`, no sequence drop); returns `(ok, wire_seq)`."""
        var overhead = AEAD_OVERHEAD_BYTES if self.crypto_enabled else 0
        if WIRE_HEADER_BYTES + overhead + len(payload) > self.max_mtu_bytes:
            return (False, UInt16(0))
        self.tx_scratch.clear()
        var hdr = self._make_header(CHAN_UNRELIABLE_RAW)
        hdr.write_into(self.tx_scratch)
        for i in range(len(payload)):
            self.tx_scratch.append(payload[i])
        self._record_sent_packet(hdr.seq, now_ms)
        var ok = self._emit_tx_scratch(transport, True)
        if ok:
            self.packets_sent += 1
        return (ok, hdr.seq)

    def send_unreliable_sequenced[
        TransT: NetTransport
    ](
        mut self,
        payload: List[UInt8],
        mut transport: TransT,
        now_ms: Float64,
    ) -> Tuple[Bool, UInt16]:
        """Sends an unreliable-sequenced datagram (`<= 1180 B`); returns `(ok, wire_seq)`."""
        var overhead = AEAD_OVERHEAD_BYTES if self.crypto_enabled else 0
        if WIRE_HEADER_BYTES + 2 + overhead + len(payload) > self.max_mtu_bytes:
            return (False, UInt16(0))
        self.tx_scratch.clear()
        var hdr = self._make_header(CHAN_UNRELIABLE_SEQUENCED)
        hdr.write_into(self.tx_scratch)
        var useq = self.next_unrel_seq
        self.next_unrel_seq = UInt16((Int(self.next_unrel_seq) + 1) & 0xFFFF)
        _put_u16(self.tx_scratch, useq)
        for i in range(len(payload)):
            self.tx_scratch.append(payload[i])
        self._record_sent_packet(hdr.seq, now_ms)
        var ok = self._emit_tx_scratch(transport, True)
        if ok:
            self.packets_sent += 1
        return (ok, hdr.seq)

    def _queue_reliable_impl(
        mut self, message: List[UInt8], unordered: Bool
    ) -> Bool:
        var crypto_extra = AEAD_OVERHEAD_BYTES if self.crypto_enabled else 0
        var max_chunk = min(
            MAX_FRAGMENT_PAYLOAD_BYTES,
            self.max_mtu_bytes
            - WIRE_HEADER_BYTES
            - _FRAG_HEADER_BYTES
            - 8
            - crypto_extra,
        )
        if max_chunk <= 32:
            return False
        var total = len(message)
        var frag_count = max(1, (total + max_chunk - 1) // max_chunk)
        if frag_count > MAX_RELIABLE_FRAG_COUNT or total > MAX_RELIABLE_MESSAGE_BYTES:
            return False

        var msg_id = self.next_msg_id
        self.next_msg_id = UInt16((Int(self.next_msg_id) + 1) & 0xFFFF)

        for f in range(frag_count):
            var start = f * max_chunk
            var end = min(total, start + max_chunk)
            var frag = _ReliableOutFrag()
            frag.unordered = unordered
            frag.msg_id = msg_id
            frag.frag_idx = UInt16(f)
            frag.frag_count = UInt16(frag_count)
            frag.payload = List[UInt8](capacity=max(1, end - start))
            for k in range(start, end):
                frag.payload.append(message[k])
            self.pending_out_queue.append(frag^)

        self._promote_pending_frags()
        return True

    def send_reliable(mut self, message: List[UInt8]) -> Bool:
        """Queues a reliable-ordered message of arbitrary size, automatically fragmenting into
        `<= 1024 B` chunks so every UDP packet stays strictly `<= 1180 B` IPv6-safe MTU."""
        return self._queue_reliable_impl(message, False)

    def send_reliable_unordered(mut self, message: List[UInt8]) -> Bool:
        """Queues a reliable-unordered message delivered immediately on arrival without HoL blocking."""
        return self._queue_reliable_impl(message, True)

    def _send_one_reliable_frag[
        TransT: NetTransport
    ](mut self, slot: Int, mut transport: TransT, now_ms: Float64) -> Bool:
        self.tx_scratch.clear()
        var chan_kind = (
            CHAN_RELIABLE_UNORDERED
            if self.out_frags[slot].unordered
            else CHAN_RELIABLE_ORDERED
        )
        var hdr = self._make_header(chan_kind)
        hdr.write_into(self.tx_scratch)
        # Piggyback reliable cumulative + selective ACK + fragment metadata (`20` bytes)
        _put_u16(self.tx_scratch, self.next_expected_rel_seq)
        _put_u64(self.tx_scratch, self._rel_recv_sack_bits())
        _put_u16(self.tx_scratch, self.out_frags[slot].rel_seq)
        _put_u16(self.tx_scratch, self.out_frags[slot].msg_id)
        _put_u16(self.tx_scratch, self.out_frags[slot].frag_idx)
        _put_u16(self.tx_scratch, self.out_frags[slot].frag_count)
        var plen = len(self.out_frags[slot].payload)
        _put_u16(self.tx_scratch, UInt16(plen))
        for i in range(plen):
            self.tx_scratch.append(self.out_frags[slot].payload[i])

        if self.out_frags[slot].send_count > 0:
            self.retransmissions += 1
        self.out_frags[slot].send_count += 1
        if isfinite(now_ms):
            self.out_frags[slot].last_sent_ms = now_ms
        self._record_sent_packet(
            hdr.seq, now_ms, True, self.out_frags[slot].rel_seq
        )
        var ok = self._emit_tx_scratch(transport, False)
        if ok:
            self.packets_sent += 1
            self.needs_ack_flush = False
        return ok

    def _send_ack_heartbeat[
        TransT: NetTransport
    ](mut self, mut transport: TransT, now_ms: Float64):
        self.tx_scratch.clear()
        var hdr = self._make_header(CHAN_ACK_HEARTBEAT)
        hdr.write_into(self.tx_scratch)
        _put_u16(self.tx_scratch, self.next_expected_rel_seq)
        _put_u64(self.tx_scratch, self._rel_recv_sack_bits())
        self._record_sent_packet(hdr.seq, now_ms)
        if self._emit_tx_scratch(transport, False):
            self.packets_sent += 1
        self.needs_ack_flush = False

    def _deliver_unordered_frag(
        mut self,
        msg_id: UInt16,
        frag_idx: UInt16,
        frag_cnt: UInt16,
        payload_ptr: Int,
        plen: Int,
    ):
        var live_u = (
            len(self.incoming_reliable_unordered)
            - self.incoming_reliable_unordered_head
        )
        if frag_cnt == 1:
            if (
                live_u < MAX_INCOMING_RELIABLE_MESSAGES
                and self.incoming_reliable_unordered_bytes + plen
                <= MAX_INCOMING_RELIABLE_BYTES
            ):
                var out = List[UInt8](capacity=max(1, plen))
                if plen > 0:
                    out.resize(unsafe_uninit_length=plen)
                    _ = external_call["memcpy", Int](
                        Int(out.unsafe_ptr()), payload_ptr, Int64(plen)
                    )
                self.incoming_reliable_unordered.append(out^)
                self.incoming_reliable_unordered_bytes += plen
            return

        # Multi-fragment unordered reassembly (capped to 16 concurrent unordered messages)
        var slot = -1
        for i in range(len(self.unordered_assemblies)):
            if (
                self.unordered_assemblies[i].active
                and self.unordered_assemblies[i].msg_id == msg_id
                and self.unordered_assemblies[i].frag_count == frag_cnt
            ):
                slot = i
                break
        if slot < 0:
            for i in range(len(self.unordered_assemblies)):
                if not self.unordered_assemblies[i].active:
                    slot = i
                    break
            if slot < 0:
                if len(self.unordered_assemblies) >= 16:
                    return
                self.unordered_assemblies.append(_UnorderedAssembly())
                slot = len(self.unordered_assemblies) - 1
            self.unordered_assemblies[slot].active = True
            self.unordered_assemblies[slot].msg_id = msg_id
            self.unordered_assemblies[slot].frag_count = frag_cnt
            self.unordered_assemblies[slot].received_count = 0
            self.unordered_assemblies[slot].total_bytes = 0
            self.unordered_assemblies[slot].frag_got = List[Bool](
                length=Int(frag_cnt), fill=False
            )
            self.unordered_assemblies[slot].frag_data = List[List[UInt8]]()
            for _ in range(Int(frag_cnt)):
                self.unordered_assemblies[slot].frag_data.append(List[UInt8]())

        var idx = Int(frag_idx)
        if self.unordered_assemblies[slot].frag_got[idx]:
            return
        if (
            self.unordered_assemblies[slot].total_bytes + plen
            > MAX_RELIABLE_MESSAGE_BYTES
        ):
            self.unordered_assemblies[slot].active = False
            self.unordered_assemblies[slot].frag_got.clear()
            self.unordered_assemblies[slot].frag_data.clear()
            return
        var frag_buf = List[UInt8](capacity=max(1, plen))
        if plen > 0:
            frag_buf.resize(unsafe_uninit_length=plen)
            _ = external_call["memcpy", Int](
                Int(frag_buf.unsafe_ptr()), payload_ptr, Int64(plen)
            )
        self.unordered_assemblies[slot].frag_data[idx] = frag_buf^
        self.unordered_assemblies[slot].frag_got[idx] = True
        self.unordered_assemblies[slot].received_count += 1
        self.unordered_assemblies[slot].total_bytes += plen

        if self.unordered_assemblies[slot].received_count >= Int(frag_cnt):
            var total = self.unordered_assemblies[slot].total_bytes
            if (
                live_u < MAX_INCOMING_RELIABLE_MESSAGES
                and self.incoming_reliable_unordered_bytes + total
                <= MAX_INCOMING_RELIABLE_BYTES
            ):
                var full = List[UInt8](capacity=max(1, total))
                for f in range(Int(frag_cnt)):
                    for b in range(len(self.unordered_assemblies[slot].frag_data[f])):
                        full.append(self.unordered_assemblies[slot].frag_data[f][b])
                self.incoming_reliable_unordered.append(full^)
                self.incoming_reliable_unordered_bytes += total
            self.unordered_assemblies[slot].active = False
            self.unordered_assemblies[slot].frag_got.clear()
            self.unordered_assemblies[slot].frag_data.clear()

    def _drain_ordered_reassembly(mut self):
        while True:
            var slot = Int(self.next_expected_rel_seq) & (_RELIABLE_WINDOW - 1)
            if (
                not self.in_frags[slot].received
                or self.in_frags[slot].rel_seq != self.next_expected_rel_seq
            ):
                break

            if not self.in_frags[slot].unordered:
                var f_msg = self.in_frags[slot].msg_id
                var f_idx = self.in_frags[slot].frag_idx
                var f_cnt = self.in_frags[slot].frag_count

                if f_idx == 0:
                    if f_cnt > 0 and Int(f_cnt) <= MAX_RELIABLE_FRAG_COUNT:
                        self.assembling_active = True
                        self.assembling_msg_id = f_msg
                        self.assembling_expected_idx = 0
                        self.assembling_frag_count = f_cnt
                        self.assembling_bytes.clear()
                    else:
                        self.assembling_active = False
                        self.assembling_bytes.clear()

                if (
                    self.assembling_active
                    and self.assembling_msg_id == f_msg
                    and self.assembling_expected_idx == f_idx
                ):
                    var frag_len = len(self.in_frags[slot].payload)
                    if (
                        len(self.assembling_bytes) + frag_len
                        > MAX_RELIABLE_MESSAGE_BYTES
                    ):
                        self.assembling_bytes.clear()
                        self.assembling_active = False
                    else:
                        for b in range(frag_len):
                            self.assembling_bytes.append(
                                self.in_frags[slot].payload[b]
                            )
                        self.assembling_expected_idx += 1
                        if (
                            self.assembling_expected_idx
                            >= self.assembling_frag_count
                        ):
                            var msg_bytes = len(self.assembling_bytes)
                            var live_r = (
                                len(self.incoming_reliable)
                                - self.incoming_reliable_head
                            )
                            if (
                                live_r < MAX_INCOMING_RELIABLE_MESSAGES
                                and self.incoming_reliable_bytes + msg_bytes
                                <= MAX_INCOMING_RELIABLE_BYTES
                            ):
                                self.incoming_reliable.append(
                                    self.assembling_bytes.copy()
                                )
                                self.incoming_reliable_bytes += msg_bytes
                            self.assembling_bytes.clear()
                            self.assembling_active = False

            self.in_frags[slot].received = False
            self.in_frags[slot].unordered = False
            self.in_frags[slot].payload.clear()
            self.next_expected_rel_seq = UInt16(
                (Int(self.next_expected_rel_seq) + 1) & 0xFFFF
            )

    def ingest_datagram(
        mut self, view: DatagramView, now_ms: Float64
    ) -> Bool:
        var parsed = PacketWireHeader.parse_from_ptr(view.data_ptr, view.length)
        if not parsed[0]:
            return False
        var hdr = parsed[1]
        var body_ptr = view.data_ptr + WIRE_HEADER_BYTES
        var body_len = view.length - WIRE_HEADER_BYTES

        var is_encrypted = (hdr.chan_flags & WIRE_FLAG_ENCRYPTED) != 0
        if self.crypto_enabled:
            if not is_encrypted or body_len < AEAD_OVERHEAD_BYTES:
                self.crypto_auth_failures += 1
                return False
            var nonce = _get_u64_ptr(body_ptr, 0)
            if not self.rx_replay.would_accept(nonce):
                self.crypto_replay_drops += 1
                return False
            var ct_len = body_len - AEAD_OVERHEAD_BYTES
            var ct_ptr = body_ptr + AEAD_NONCE_BYTES
            var tag_ptr = ct_ptr + ct_len
            var n1 = UInt32(nonce & 0xFFFFFFFF)
            var n2 = UInt32(nonce >> 32)
            var ok_dec = aead_open_raw(
                self.aead_key,
                UInt32(0),
                n1,
                n2,
                view.data_ptr,
                WIRE_HEADER_BYTES + AEAD_NONCE_BYTES,
                ct_ptr,
                ct_len,
                tag_ptr,
                self._decrypt_scratch,
            )
            if not ok_dec:
                self.crypto_auth_failures += 1
                return False
            self.rx_replay.record(nonce)
            body_ptr = (
                Int(self._decrypt_scratch.unsafe_ptr())
                if len(self._decrypt_scratch) > 0
                else 0
            )
            body_len = len(self._decrypt_scratch)
        elif is_encrypted:
            self.crypto_auth_failures += 1
            return False

        self.packets_received += 1

        if (hdr.chan_flags & WIRE_FLAG_HAS_REMOTE_SEQ) != 0:
            self._process_remote_acks(hdr.ack, hdr.ack_bits, now_ms)

        var is_fresh_wire = self._record_remote_wire_seq(hdr.seq)
        var kind = hdr.chan_flags & 0x0F

        if kind == CHAN_UNRELIABLE_RAW:
            if not is_fresh_wire:
                _ = len(self._decrypt_scratch)
                return True
            var out = List[UInt8](capacity=max(1, body_len))
            if body_len > 0:
                out.resize(unsafe_uninit_length=body_len)
                _ = external_call["memcpy", Int](
                    Int(out.unsafe_ptr()), body_ptr, Int64(body_len)
                )
            if (
                len(self.incoming_unreliable_raw)
                - self.incoming_unreliable_raw_head
                >= MAX_INCOMING_UNRELIABLE_MESSAGES
            ):
                self.incoming_unreliable_raw[
                    self.incoming_unreliable_raw_head
                ].clear()
                self.incoming_unreliable_raw_head += 1
            self.incoming_unreliable_raw.append(out^)
            _ = len(self._decrypt_scratch)
            return True

        if kind == CHAN_UNRELIABLE_SEQUENCED:
            if not is_fresh_wire or body_len < 2:
                _ = len(self._decrypt_scratch)
                return True
            var useq = _get_u16_ptr(body_ptr, 0)
            if (
                not self.has_remote_unrel_seq
                or seq_gt(useq, self.last_remote_unrel_seq)
            ):
                self.has_remote_unrel_seq = True
                self.last_remote_unrel_seq = useq
                var msg_len = body_len - 2
                var out = List[UInt8](capacity=max(1, msg_len))
                if msg_len > 0:
                    out.resize(unsafe_uninit_length=msg_len)
                    _ = external_call["memcpy", Int](
                        Int(out.unsafe_ptr()), body_ptr + 2, Int64(msg_len)
                    )
                if (
                    len(self.incoming_unreliable) - self.incoming_unreliable_head
                    >= MAX_INCOMING_UNRELIABLE_MESSAGES
                ):
                    self.incoming_unreliable[
                        self.incoming_unreliable_head
                    ].clear()
                    self.incoming_unreliable_head += 1
                self.incoming_unreliable.append(out^)
            _ = len(self._decrypt_scratch)
            return True

        if kind == CHAN_ACK_HEARTBEAT:
            if body_len >= 10:
                var cum_ack = _get_u16_ptr(body_ptr, 0)
                var sack_bits = _get_u64_ptr(body_ptr, 2)
                self._process_rel_sack(cum_ack, sack_bits)
            _ = len(self._decrypt_scratch)
            return True

        if kind == CHAN_RELIABLE_ORDERED or kind == CHAN_RELIABLE_UNORDERED:
            if body_len < 20:
                _ = len(self._decrypt_scratch)
                return False
            var rseq = _get_u16_ptr(body_ptr, 10)
            var msg_id = _get_u16_ptr(body_ptr, 12)
            var frag_idx = _get_u16_ptr(body_ptr, 14)
            var frag_cnt = _get_u16_ptr(body_ptr, 16)
            var plen = Int(_get_u16_ptr(body_ptr, 18))
            if (
                20 + plen > body_len
                or frag_cnt == 0
                or Int(frag_cnt) > MAX_RELIABLE_FRAG_COUNT
                or frag_idx >= frag_cnt
            ):
                _ = len(self._decrypt_scratch)
                return False

            var cum_ack = _get_u16_ptr(body_ptr, 0)
            var sack_bits = _get_u64_ptr(body_ptr, 2)
            self._process_rel_sack(cum_ack, sack_bits)

            self.needs_ack_flush = True
            # Ignore already-delivered or far-future fragments outside window
            var ahead = seq_diff(rseq, self.next_expected_rel_seq)
            if ahead < 0 or ahead >= (_RELIABLE_WINDOW - 8):
                _ = len(self._decrypt_scratch)
                return True

            var slot = Int(rseq) & (_RELIABLE_WINDOW - 1)
            if (
                not self.in_frags[slot].received
                or self.in_frags[slot].rel_seq != rseq
            ):
                var is_unordered = kind == CHAN_RELIABLE_UNORDERED
                self.in_frags[slot].received = True
                self.in_frags[slot].unordered = is_unordered
                self.in_frags[slot].rel_seq = rseq
                self.in_frags[slot].msg_id = msg_id
                self.in_frags[slot].frag_idx = frag_idx
                self.in_frags[slot].frag_count = frag_cnt
                self.in_frags[slot].payload.clear()
                if is_unordered:
                    self._deliver_unordered_frag(
                        msg_id, frag_idx, frag_cnt, body_ptr + 20, plen
                    )
                elif plen > 0:
                    self.in_frags[slot].payload.reserve(plen)
                    self.in_frags[slot].payload.resize(
                        unsafe_uninit_length=plen
                    )
                    _ = external_call["memcpy", Int](
                        Int(self.in_frags[slot].payload.unsafe_ptr()),
                        body_ptr + 20,
                        Int64(plen),
                    )
                self._drain_ordered_reassembly()
            _ = len(self._decrypt_scratch)
            return True

        _ = len(self._decrypt_scratch)
        return True

    def poll[
        TransT: NetTransport
    ](
        mut self,
        mut transport: TransT,
        now_ms: Float64,
        drain_transport: Bool = True,
        flush_transport: Bool = True,
    ):
        """Polls incoming datagrams, advances sliding window with AIMD + exponential RTO backoff,
        and transmits due fragments/ACKs."""
        self.newly_acked_wire_seqs.clear()
        if drain_transport:
            while True:
                var n = transport.recv_batch(self.rx_views_scratch)
                if n <= 0:
                    break
                for i in range(n):
                    var view_item = self.rx_views_scratch[i]
                    if view_item.sender == self.peer_endpoint:
                        _ = self.ingest_datagram(view_item, now_ms)

        self._promote_pending_frags()

        # Transmit unsent or exponentially-backed-off RTO-expired reliable fragments within cwnd
        var cwnd_cap = max(1, Int(self.cwnd))
        var budget = min(32, cwnd_cap)
        var span = seq_diff(
            self.next_send_rel_seq, self.oldest_unacked_rel_seq
        )
        var in_flight = 0
        for off in range(max(0, span)):
            if budget <= 0:
                break
            var rseq = UInt16(
                (Int(self.oldest_unacked_rel_seq) + off) & 0xFFFF
            )
            var slot = Int(rseq) & (_RELIABLE_WINDOW - 1)
            if (
                self.out_frags[slot].active
                and self.out_frags[slot].rel_seq == rseq
                and not self.out_frags[slot].acked
            ):
                var sc = self.out_frags[slot].send_count
                var is_retx = sc > 0
                var backoff_shift = min(4, max(0, sc - 1))
                var eff_rto = min(4000.0, self.rto_ms * Float64(1 << backoff_shift))
                var due = (
                    not is_retx
                    or (
                        isfinite(now_ms)
                        and isfinite(self.out_frags[slot].last_sent_ms)
                        and (now_ms - self.out_frags[slot].last_sent_ms) >= eff_rto
                    )
                )
                if not due:
                    in_flight += 1
                    continue
                if not is_retx and in_flight >= cwnd_cap:
                    continue
                if is_retx and isfinite(now_ms) and (now_ms - self._last_loss_ms) >= self.rto_ms:
                    self.ssthresh = max(2.0, self.cwnd * 0.5)
                    self.cwnd = max(2.0, self.ssthresh)
                    self._last_loss_ms = now_ms
                if self._send_one_reliable_frag(slot, transport, now_ms):
                    budget -= 1
                    in_flight += 1

        if self.needs_ack_flush:
            self._send_ack_heartbeat(transport, now_ms)

        if flush_transport:
            _ = transport.flush_sends()

    @staticmethod
    def _pop_head_bytes(
        mut queue: List[List[UInt8]], mut head: Int
    ) -> Tuple[Bool, List[UInt8]]:
        if head >= len(queue):
            if len(queue) > 0:
                queue.clear()
            head = 0
            return (False, List[UInt8]())
        var msg = queue[head].copy()
        queue[head].clear()
        head += 1
        if head >= len(queue):
            queue.clear()
            head = 0
        elif head >= 64 and head * 2 >= len(queue):
            var rem = len(queue) - head
            for i in range(rem):
                queue[i] = queue[head + i].copy()
            while len(queue) > rem:
                _ = queue.pop()
            head = 0
        return (True, msg^)

    def pop_reliable_message(mut self) -> Tuple[Bool, List[UInt8]]:
        var res = Self._pop_head_bytes(
            self.incoming_reliable, self.incoming_reliable_head
        )
        if res[0]:
            self.incoming_reliable_bytes = max(
                0, self.incoming_reliable_bytes - len(res[1])
            )
        return res^

    def pop_reliable_unordered_message(mut self) -> Tuple[Bool, List[UInt8]]:
        var res = Self._pop_head_bytes(
            self.incoming_reliable_unordered,
            self.incoming_reliable_unordered_head,
        )
        if res[0]:
            self.incoming_reliable_unordered_bytes = max(
                0, self.incoming_reliable_unordered_bytes - len(res[1])
            )
        return res^

    def pop_unreliable_message(mut self) -> Tuple[Bool, List[UInt8]]:
        return Self._pop_head_bytes(
            self.incoming_unreliable, self.incoming_unreliable_head
        )

    def pop_unreliable_raw_message(mut self) -> Tuple[Bool, List[UInt8]]:
        return Self._pop_head_bytes(
            self.incoming_unreliable_raw, self.incoming_unreliable_raw_head
        )

    @always_inline
    def is_all_reliable_acked(self) -> Bool:
        return (
            self.pending_out_head >= len(self.pending_out_queue)
            and self.oldest_unacked_rel_seq == self.next_send_rel_seq
        )
