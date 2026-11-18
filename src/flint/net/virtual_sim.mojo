"""Deterministic in-memory network conditioner (`flint.net.virtual_sim`).

Models configurable base one-way latency, uniform or triangular jitter (producing authentic
out-of-order delivery), random packet loss, burst blackouts, packet duplication, and IPv6/IPv4
MTU enforcement using `detsort.MinHeap` and `detsort.mix64` — zero NIC traffic or OS nondeterminism.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max, min
from flint.core.detsort import MinHeap, mix64
from flint.net.socket_udp import (
    DatagramView,
    NetEndpoint,
    NetTransport,
    UDP_MAX_DATAGRAM_BYTES,
    UDP_RING_CAPACITY,
)

comptime JITTER_UNIFORM: Int = 0
comptime JITTER_TRIANGULAR: Int = 1


struct _SimPacket(Copyable, Movable):
    var active: Bool
    var src: NetEndpoint
    var dst_idx: Int
    var payload: List[UInt8]

    def __init__(out self):
        self.active = False
        self.src = NetEndpoint()
        self.dst_idx = -1
        self.payload = List[UInt8]()


struct _SimInbox(Movable):
    var endpoint: NetEndpoint
    var ready_packets: List[Int]  # indices into `_VirtualSimCore.pool`
    var drop_next_outbound: Int
    var drop_next_inbound: Int
    var blackout_until_us: Int

    def __init__(out self, endpoint: NetEndpoint):
        self.endpoint = endpoint
        self.ready_packets = List[Int]()
        self.drop_next_outbound = 0
        self.drop_next_inbound = 0
        self.blackout_until_us = -1


struct _VirtualSimCore(Movable):
    var now_us: Int
    var rng_state: UInt64
    var base_latency_us: Int
    var jitter_us: Int
    var jitter_mode: Int
    var loss_rate: Float64
    var dup_rate: Float64
    var burst_loss_prob: Float64
    var burst_duration_us: Int
    var global_blackout_until_us: Int
    var mtu_bytes: Int

    var inboxes: List[_SimInbox]
    var pool: List[_SimPacket]
    var free_slots: List[Int]
    var flight_heap: MinHeap

    var packets_submitted: Int
    var packets_delivered: Int
    var packets_dropped_loss: Int
    var packets_dropped_blackout: Int
    var packets_dropped_mtu: Int
    var packets_duplicated: Int

    def __init__(
        out self,
        seed: UInt64,
        one_way_latency_ms: Float64,
        jitter_ms: Float64,
        loss_rate: Float64,
        dup_rate: Float64,
        mtu_bytes: Int,
        jitter_mode: Int,
    ):
        self.now_us = 0
        self.rng_state = seed ^ UInt64(0xD1B54A32D192ED03)
        self.base_latency_us = max(0, Int(one_way_latency_ms * 1000.0 + 0.5))
        self.jitter_us = max(0, Int(jitter_ms * 1000.0 + 0.5))
        self.jitter_mode = jitter_mode
        self.loss_rate = max(0.0, min(1.0, loss_rate))
        self.dup_rate = max(0.0, min(1.0, dup_rate))
        self.burst_loss_prob = 0.0
        self.burst_duration_us = 0
        self.global_blackout_until_us = -1
        self.mtu_bytes = max(64, mtu_bytes)

        self.inboxes = List[_SimInbox]()
        self.pool = List[_SimPacket]()
        self.free_slots = List[Int]()
        self.flight_heap = MinHeap()

        self.packets_submitted = 0
        self.packets_delivered = 0
        self.packets_dropped_loss = 0
        self.packets_dropped_blackout = 0
        self.packets_dropped_mtu = 0
        self.packets_duplicated = 0

    @always_inline
    def _next_u64(mut self) -> UInt64:
        self.rng_state += 0x9E3779B97F4A7C15
        return mix64(self.rng_state)

    @always_inline
    def _next_unit_f64(mut self) -> Float64:
        return Float64(self._next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def _sample_delay_us(mut self) -> Int:
        if self.jitter_us <= 0:
            return self.base_latency_us
        var j_offset: Float64
        if self.jitter_mode == JITTER_TRIANGULAR:
            var u1 = self._next_unit_f64()
            var u2 = self._next_unit_f64()
            j_offset = (u1 + u2 - 1.0) * Float64(self.jitter_us)
        else:
            var u = self._next_unit_f64() * 2.0 - 1.0
            j_offset = u * Float64(self.jitter_us)
        return max(0, self.base_latency_us + Int(j_offset))

    def _find_endpoint_idx(self, ep: NetEndpoint) -> Int:
        for i in range(len(self.inboxes)):
            if self.inboxes[i].endpoint == ep:
                return i
        return -1

    def _alloc_packet(
        mut self, src: NetEndpoint, dst_idx: Int, data_ptr: Int, length: Int
    ) -> Int:
        var slot: Int
        if len(self.free_slots) > 0:
            slot = self.free_slots.pop()
            self.pool[slot].active = True
            self.pool[slot].src = src
            self.pool[slot].dst_idx = dst_idx
            self.pool[slot].payload.clear()
            self.pool[slot].payload.reserve(length)
            self.pool[slot].payload.resize(unsafe_uninit_length=length)
            _ = external_call["memcpy", Int](
                Int(self.pool[slot].payload.unsafe_ptr()),
                data_ptr,
                Int64(length),
            )
            return slot
        var pkt = _SimPacket()
        pkt.active = True
        pkt.src = src
        pkt.dst_idx = dst_idx
        pkt.payload = List[UInt8](capacity=max(64, length))
        pkt.payload.resize(unsafe_uninit_length=length)
        _ = external_call["memcpy", Int](
            Int(pkt.payload.unsafe_ptr()), data_ptr, Int64(length)
        )
        slot = len(self.pool)
        self.pool.append(pkt^)
        return slot

    def _release_packet(mut self, slot: Int):
        if slot >= 0 and slot < len(self.pool) and self.pool[slot].active:
            self.pool[slot].active = False
            self.pool[slot].payload.clear()
            self.free_slots.append(slot)

    def submit_datagram(
        mut self, src_idx: Int, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if src_idx < 0 or src_idx >= len(self.inboxes) or data_ptr == 0 or length <= 0:
            return False
        self.packets_submitted += 1

        if length > self.mtu_bytes:
            self.packets_dropped_mtu += 1
            return True

        var dst_idx = self._find_endpoint_idx(dst)
        if dst_idx < 0:
            return False

        # Deterministic burst blackout triggers
        if self.burst_loss_prob > 0.0 and self.burst_duration_us > 0:
            if self._next_unit_f64() < self.burst_loss_prob:
                self.inboxes[src_idx].blackout_until_us = max(
                    self.inboxes[src_idx].blackout_until_us,
                    self.now_us + self.burst_duration_us,
                )

        if self.inboxes[src_idx].drop_next_outbound > 0:
            self.inboxes[src_idx].drop_next_outbound -= 1
            self.packets_dropped_blackout += 1
            return True

        if self.inboxes[dst_idx].drop_next_inbound > 0:
            self.inboxes[dst_idx].drop_next_inbound -= 1
            self.packets_dropped_blackout += 1
            return True

        if (
            self.now_us < self.global_blackout_until_us
            or self.now_us < self.inboxes[src_idx].blackout_until_us
            or self.now_us < self.inboxes[dst_idx].blackout_until_us
        ):
            self.packets_dropped_blackout += 1
            return True

        if self.loss_rate > 0.0 and self._next_unit_f64() < self.loss_rate:
            self.packets_dropped_loss += 1
            return True

        var src_ep = self.inboxes[src_idx].endpoint
        var delay1 = self._sample_delay_us()
        var slot1 = self._alloc_packet(src_ep, dst_idx, data_ptr, length)
        self.flight_heap.push(self.now_us + delay1, slot1)

        if self.dup_rate > 0.0 and self._next_unit_f64() < self.dup_rate:
            var delay2 = self._sample_delay_us() + 250
            var slot2 = self._alloc_packet(src_ep, dst_idx, data_ptr, length)
            self.flight_heap.push(self.now_us + delay2, slot2)
            self.packets_duplicated += 1

        # Deliver zero-latency packets immediately
        self.pump_due()
        return True

    def pump_due(mut self):
        while len(self.flight_heap) > 0:
            var top = self.flight_heap.peek()
            if top.key > self.now_us:
                break
            _ = self.flight_heap.pop()
            var slot = top.value
            if slot >= 0 and slot < len(self.pool) and self.pool[slot].active:
                var dst_idx = self.pool[slot].dst_idx
                if dst_idx >= 0 and dst_idx < len(self.inboxes):
                    self.inboxes[dst_idx].ready_packets.append(slot)
                    self.packets_delivered += 1
                else:
                    self._release_packet(slot)


struct VirtualEndpoint(NetTransport):
    """Virtual NIC endpoint implementing `NetTransport` over `VirtualNetSim`."""
    var core_ptr: Int
    var endpoint_id: Int
    var bound: NetEndpoint
    var rx_ring_buf: List[UInt8]
    var tx_ring_buf: List[UInt8]
    var tx_ring_dst: List[NetEndpoint]
    var tx_ring_len: List[Int]
    var tx_queued: Int

    def __init__(out self, core_ptr: Int, endpoint_id: Int, bound: NetEndpoint):
        self.core_ptr = core_ptr
        self.endpoint_id = endpoint_id
        self.bound = bound
        self.rx_ring_buf = List[UInt8](
            length=UDP_RING_CAPACITY * UDP_MAX_DATAGRAM_BYTES, fill=0
        )
        self.tx_ring_buf = List[UInt8](
            length=UDP_RING_CAPACITY * UDP_MAX_DATAGRAM_BYTES, fill=0
        )
        self.tx_ring_dst = List[NetEndpoint](
            length=UDP_RING_CAPACITY, fill=NetEndpoint()
        )
        self.tx_ring_len = List[Int](length=UDP_RING_CAPACITY, fill=0)
        self.tx_queued = 0

    @always_inline
    def local_endpoint(self) -> NetEndpoint:
        return self.bound

    @always_inline
    def _core(self) -> Pointer[_VirtualSimCore, MutAnyOrigin]:
        return Pointer[_VirtualSimCore, MutAnyOrigin](
            unsafe_from_address=self.core_ptr
        )

    def send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if self.core_ptr == 0 or length <= 0 or length > UDP_MAX_DATAGRAM_BYTES:
            return False
        return self._core()[].submit_datagram(
            self.endpoint_id, dst, data_ptr, length
        )

    def send_to(mut self, dst: NetEndpoint, payload: List[UInt8]) -> Bool:
        if len(payload) == 0:
            return False
        return self.send_raw(dst, Int(payload.unsafe_ptr()), len(payload))

    def queue_send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if (
            self.core_ptr == 0
            or data_ptr == 0
            or length <= 0
            or length > UDP_MAX_DATAGRAM_BYTES
        ):
            return False
        if self.tx_queued >= UDP_RING_CAPACITY:
            _ = self.flush_sends()
        var idx = self.tx_queued
        var dst_ptr = (
            Int(self.tx_ring_buf.unsafe_ptr()) + idx * UDP_MAX_DATAGRAM_BYTES
        )
        _ = external_call["memcpy", Int](dst_ptr, data_ptr, Int64(length))
        self.tx_ring_dst[idx] = dst
        self.tx_ring_len[idx] = length
        self.tx_queued += 1
        return True

    def flush_sends(mut self) -> Int:
        if self.core_ptr == 0 or self.tx_queued <= 0:
            self.tx_queued = 0
            return 0
        var sent = 0
        var base = Int(self.tx_ring_buf.unsafe_ptr())
        for i in range(self.tx_queued):
            if self._core()[].submit_datagram(
                self.endpoint_id,
                self.tx_ring_dst[i],
                base + i * UDP_MAX_DATAGRAM_BYTES,
                self.tx_ring_len[i],
            ):
                sent += 1
        self.tx_queued = 0
        return sent

    def recv_batch(mut self, mut out_views: List[DatagramView]) -> Int:
        out_views.clear()
        if self.core_ptr == 0:
            return 0
        if out_views.capacity() < UDP_RING_CAPACITY:
            out_views.reserve(UDP_RING_CAPACITY)
        var core = self._core()
        core[].pump_due()

        ref ready = core[].inboxes[self.endpoint_id].ready_packets
        if len(ready) == 0:
            return 0

        var take = min(UDP_RING_CAPACITY, len(ready))
        var rx_base = Int(self.rx_ring_buf.unsafe_ptr())
        for i in range(take):
            var pkt_slot = ready[i]
            var sender = core[].pool[pkt_slot].src
            var plen = len(core[].pool[pkt_slot].payload)
            var copy_len = min(UDP_MAX_DATAGRAM_BYTES, plen)
            var dst_slot = rx_base + i * UDP_MAX_DATAGRAM_BYTES
            if copy_len > 0:
                _ = external_call["memcpy", Int](
                    dst_slot,
                    Int(core[].pool[pkt_slot].payload.unsafe_ptr()),
                    Int64(copy_len),
                )
            core[]._release_packet(pkt_slot)
            out_views.append(DatagramView(sender, dst_slot, copy_len))

        if take == len(ready):
            ready.clear()
        else:
            var remaining = len(ready) - take
            for r in range(remaining):
                ready[r] = ready[r + take]
            ready.resize(unsafe_uninit_length=remaining)
        return len(out_views)


struct VirtualNetSim(Movable):
    """Deterministic multi-peer virtual network switch & chaos impairment simulator."""
    var _storage: List[_VirtualSimCore]

    def __init__(
        out self,
        seed: UInt64 = 0x5EED1234CAFEBABE,
        one_way_latency_ms: Float64 = 40.0,
        jitter_ms: Float64 = 10.0,
        loss_rate: Float64 = 0.0,
        dup_rate: Float64 = 0.0,
        mtu_bytes: Int = 1200,
        jitter_mode: Int = JITTER_UNIFORM,
    ):
        self._storage = List[_VirtualSimCore](capacity=1)
        self._storage.append(
            _VirtualSimCore(
                seed,
                one_way_latency_ms,
                jitter_ms,
                loss_rate,
                dup_rate,
                mtu_bytes,
                jitter_mode,
            )
        )

    @staticmethod
    def from_rtt(
        rtt_ms: Float64,
        jitter_ms: Float64 = 0.0,
        loss_rate: Float64 = 0.0,
        dup_rate: Float64 = 0.0,
        mtu_bytes: Int = 1200,
        seed: UInt64 = 0x5EED1234CAFEBABE,
        jitter_mode: Int = JITTER_TRIANGULAR,
    ) -> Self:
        return Self(
            seed=seed,
            one_way_latency_ms=rtt_ms * 0.5,
            jitter_ms=jitter_ms * 0.5,
            loss_rate=loss_rate,
            dup_rate=dup_rate,
            mtu_bytes=mtu_bytes,
            jitter_mode=jitter_mode,
        )

    @always_inline
    def _core_ptr(self) -> Int:
        return Int(self._storage.unsafe_ptr())

    def create_endpoint(mut self, addr: NetEndpoint) -> VirtualEndpoint:
        var ep_id = len(self._storage[0].inboxes)
        self._storage[0].inboxes.append(_SimInbox(addr))
        return VirtualEndpoint(self._core_ptr(), ep_id, addr)

    def create_ipv4_peer(mut self, peer_index: Int, base_port: Int = 9000) -> VirtualEndpoint:
        var ep = NetEndpoint.ipv4(
            10, 0, (peer_index >> 8) & 0xFF, (peer_index & 0xFF) + 1, base_port + peer_index
        )
        return self.create_endpoint(ep)

    def advance_ms(mut self, delta_ms: Float64):
        var delta_us = max(0, Int(delta_ms * 1000.0 + 0.5))
        self._storage[0].now_us += delta_us
        self._storage[0].pump_due()

    def set_time_ms(mut self, time_ms: Float64):
        var target_us = max(self._storage[0].now_us, Int(time_ms * 1000.0 + 0.5))
        self._storage[0].now_us = target_us
        self._storage[0].pump_due()

    @always_inline
    def now_ms(self) -> Float64:
        return Float64(self._storage[0].now_us) / 1000.0

    def configure_burst_loss(
        mut self, prob_per_packet: Float64, blackout_duration_ms: Float64
    ):
        self._storage[0].burst_loss_prob = max(0.0, min(1.0, prob_per_packet))
        self._storage[0].burst_duration_us = max(
            0, Int(blackout_duration_ms * 1000.0 + 0.5)
        )

    def trigger_global_blackout(mut self, duration_ms: Float64):
        var dur_us = max(0, Int(duration_ms * 1000.0 + 0.5))
        self._storage[0].global_blackout_until_us = (
            self._storage[0].now_us + dur_us
        )

    def drop_next_outbound_packets(mut self, endpoint_id: Int, count: Int):
        """Deterministically drops the next `count` outbound packets sent by `endpoint_id`."""
        if endpoint_id >= 0 and endpoint_id < len(self._storage[0].inboxes):
            self._storage[0].inboxes[endpoint_id].drop_next_outbound += max(
                0, count
            )

    def drop_next_inbound_packets(mut self, endpoint_id: Int, count: Int):
        if endpoint_id >= 0 and endpoint_id < len(self._storage[0].inboxes):
            self._storage[0].inboxes[endpoint_id].drop_next_inbound += max(
                0, count
            )

    @always_inline
    def packets_submitted(self) -> Int:
        return self._storage[0].packets_submitted

    @always_inline
    def packets_delivered(self) -> Int:
        return self._storage[0].packets_delivered

    @always_inline
    def packets_dropped_mtu(self) -> Int:
        return self._storage[0].packets_dropped_mtu

    @always_inline
    def in_flight_count(self) -> Int:
        return len(self._storage[0].flight_heap)
