"""NAT Hole-Punching & Relay Fallback Topology (`flint.net.relay`).

Implements:
- `RelayForwardHeader`: 18-byte little-endian relay envelope (`RELAY_WIRE_MAGIC = "RLY5"`)
  with `hop_ttl` loop prevention and `session_token` / `(src_peer, dst_peer)` routing
- `RelayServer`: two-pass zero-allocation batch relay server with peer registration,
  anti-hijack endpoint binding, optional `StatelessCookieAuth`, peer-endpoint discovery,
  and TTL-bounded packet forwarding
- `NATPunchProbe`: direct UDP hole-punching state machine (`IDLE -> PROBING -> DIRECT |
  RELAY_FALLBACK`) with nonce-authenticated `PROBE`/`ACK`/`KEEPALIVE` packets and
  idle-timeout fallback.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max
from std.time import monotonic
from flint.core.detsort import mix64
from flint.net.crypto import AeadKey, StatelessCookieAuth
from flint.net.socket_udp import DatagramView, NetEndpoint, NetTransport

comptime RELAY_WIRE_MAGIC: UInt32 = 0x35594C52  # LE "RLY5"
comptime RELAY_HEADER_BYTES: Int = 18
comptime RELAY_ENDPOINT_PAYLOAD_BYTES: Int = 22
comptime RELAY_FLAG_DATA: UInt8 = 0x00
comptime RELAY_FLAG_REGISTER: UInt8 = 0x01
comptime RELAY_FLAG_KEEPALIVE: UInt8 = 0x02
comptime RELAY_FLAG_REGISTER_ACK: UInt8 = 0x04
comptime RELAY_FLAG_PEER_ENDPOINT: UInt8 = 0x08
comptime RELAY_FLAG_COOKIE_CHALLENGE: UInt8 = 0x10

comptime NAT_PUNCH_MAGIC: UInt32 = 0x3550544E  # LE "NTP5"
comptime NAT_PUNCH_PACKET_BYTES: Int = 24
comptime PUNCH_KIND_PROBE: UInt8 = 1
comptime PUNCH_KIND_ACK: UInt8 = 2
comptime PUNCH_KIND_KEEPALIVE: UInt8 = 3

comptime PUNCH_STATE_IDLE: Int = 0
comptime PUNCH_STATE_PROBING: Int = 1
comptime PUNCH_STATE_DIRECT: Int = 2
comptime PUNCH_STATE_RELAY_FALLBACK: Int = 3

comptime RELAY_DEFAULT_MAX_PEERS: Int = 512
comptime RELAY_DEFAULT_PEER_TIMEOUT_MS: Float64 = 15000.0
comptime RELAY_DEFAULT_SESSION_TOKEN: UInt64 = 0x464C4E545F524235
comptime NAT_PUNCH_DEFAULT_INTERVAL_MS: Float64 = 50.0
comptime NAT_PUNCH_DEFAULT_TIMEOUT_MS: Float64 = 1000.0
comptime NAT_PUNCH_KEEPALIVE_MS: Float64 = 2000.0
comptime NAT_DIRECT_IDLE_FALLBACK_MS: Float64 = 5000.0


@always_inline
def _r_load_u16_le(base: Int, off: Int) -> UInt16:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base + off)
    return UInt16(p[unsafe_offset=0]) | (UInt16(p[unsafe_offset=1]) << 8)


@always_inline
def _r_store_u16_le(base: Int, off: Int, val: UInt16):
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base + off)
    p[unsafe_offset=0] = UInt8(val & 0xFF)
    p[unsafe_offset=1] = UInt8((val >> 8) & 0xFF)


@always_inline
def _r_load_u32_le(base: Int, off: Int) -> UInt32:
    return Pointer[UInt32, ImmutAnyOrigin](
        unsafe_from_address=base + off
    ).unsafe_load[alignment=1]()


@always_inline
def _r_store_u32_le(base: Int, off: Int, val: UInt32):
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=base + off).unsafe_store[
        alignment=1
    ](val)


@always_inline
def _r_load_u64_le(base: Int, off: Int) -> UInt64:
    return Pointer[UInt64, ImmutAnyOrigin](
        unsafe_from_address=base + off
    ).unsafe_load[alignment=1]()


@always_inline
def _r_store_u64_le(base: Int, off: Int, val: UInt64):
    Pointer[UInt64, MutAnyOrigin](unsafe_from_address=base + off).unsafe_store[
        alignment=1
    ](val)


@always_inline
def _encode_endpoint_at(base: Int, off: Int, ep: NetEndpoint):
    _r_store_u64_le(base, off, ep.addr_hi)
    _r_store_u64_le(base, off + 8, ep.addr_lo)
    _r_store_u16_le(base, off + 16, ep.port)
    _r_store_u32_le(base, off + 18, ep.scope_id)


@always_inline
def decode_relay_endpoint_at(base: Int, off: Int) -> NetEndpoint:
    return NetEndpoint(
        _r_load_u64_le(base, off),
        _r_load_u64_le(base, off + 8),
        _r_load_u16_le(base, off + 16),
        _r_load_u32_le(base, off + 18),
    )


@always_inline
def _decode_endpoint_at(base: Int, off: Int) -> NetEndpoint:
    return decode_relay_endpoint_at(base, off)


struct RelayForwardHeader(Copyable, ImplicitlyCopyable, Movable):
    """18-byte little-endian header prepended to relayed datagrams."""
    var magic: UInt32
    var session_token: UInt64
    var src_peer: Int
    var dst_peer: Int
    var flags: UInt8
    var hop_ttl: UInt8
    var seq: UInt16

    def __init__(
        out self,
        session_token: UInt64 = 0,
        src_peer: Int = 0,
        dst_peer: Int = 0,
        flags: UInt8 = RELAY_FLAG_DATA,
        seq: UInt16 = 0,
        hop_ttl: UInt8 = 1,
        magic: UInt32 = RELAY_WIRE_MAGIC,
    ):
        self.magic = magic
        self.session_token = session_token
        self.src_peer = src_peer
        self.dst_peer = dst_peer
        self.flags = flags
        self.hop_ttl = hop_ttl
        self.seq = seq

    def write_into(self, mut out: List[UInt8]):
        if len(out) < RELAY_HEADER_BYTES:
            out.resize(unsafe_uninit_length=RELAY_HEADER_BYTES)
        var base = Int(out.unsafe_ptr())
        _r_store_u32_le(base, 0, self.magic)
        _r_store_u64_le(base, 4, self.session_token)
        var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base)
        p[unsafe_offset=12] = UInt8(self.src_peer & 0xFF)
        p[unsafe_offset=13] = UInt8(self.dst_peer & 0xFF)
        p[unsafe_offset=14] = self.flags
        p[unsafe_offset=15] = self.hop_ttl
        _r_store_u16_le(base, 16, self.seq)

    def encode_bytes(self) -> List[UInt8]:
        var out = List[UInt8](unsafe_uninit_length=RELAY_HEADER_BYTES)
        self.write_into(out)
        return out^

    @staticmethod
    def parse_from_ptr(data_ptr: Int, length: Int) -> Tuple[Bool, Self]:
        if data_ptr == 0 or length < RELAY_HEADER_BYTES:
            return (False, Self())
        var magic = _r_load_u32_le(data_ptr, 0)
        if magic != RELAY_WIRE_MAGIC:
            return (False, Self())
        var tok = _r_load_u64_le(data_ptr, 4)
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data_ptr)
        var src_p = Int(p[unsafe_offset=12])
        var dst_p = Int(p[unsafe_offset=13])
        var flags = p[unsafe_offset=14]
        var hop_ttl = p[unsafe_offset=15]
        var seq = _r_load_u16_le(data_ptr, 16)
        return (
            True,
            Self(
                session_token=tok,
                src_peer=src_p,
                dst_peer=dst_p,
                flags=flags,
                seq=seq,
                hop_ttl=hop_ttl,
                magic=magic,
            ),
        )


@fieldwise_init
struct _RelayPeerEntry(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var session_token: UInt64
    var peer_id: Int
    var endpoint: NetEndpoint
    var last_seen_ms: Float64

    def __init__(out self):
        self.active = False
        self.session_token = 0
        self.peer_id = -1
        self.endpoint = NetEndpoint()
        self.last_seen_ms = 0.0


struct RelayServer(Movable):
    """Stateful-endpoint packet relay server with anti-hijack binding and loop-free forwarding."""
    var listen_endpoint: NetEndpoint
    var max_peers: Int
    var peer_timeout_ms: Float64
    var peers: List[_RelayPeerEntry]
    var require_cookie: Bool
    var cookie_auth: StatelessCookieAuth
    var packets_forwarded: Int
    var bytes_forwarded: Int
    var registrations: Int
    var spoofed_packets_dropped: Int
    var invalid_packets_dropped: Int
    var unroutable_drops: Int
    var _rx_views: List[DatagramView]
    var _tx_scratch: List[UInt8]
    var _ctrl_scratch: List[UInt8]
    var _batch_ok: List[Bool]

    def __init__(
        out self,
        listen_endpoint: NetEndpoint = NetEndpoint(),
        max_peers: Int = RELAY_DEFAULT_MAX_PEERS,
        peer_timeout_ms: Float64 = RELAY_DEFAULT_PEER_TIMEOUT_MS,
    ):
        self.listen_endpoint = listen_endpoint
        self.max_peers = max(1, max_peers)
        self.peer_timeout_ms = max(100.0, peer_timeout_ms)
        self.peers = List[_RelayPeerEntry](capacity=min(64, self.max_peers))
        self.require_cookie = False
        self.cookie_auth = StatelessCookieAuth()
        self.packets_forwarded = 0
        self.bytes_forwarded = 0
        self.registrations = 0
        self.spoofed_packets_dropped = 0
        self.invalid_packets_dropped = 0
        self.unroutable_drops = 0
        self._rx_views = List[DatagramView](capacity=64)
        self._tx_scratch = List[UInt8](capacity=1500)
        self._ctrl_scratch = List[UInt8](
            unsafe_uninit_length=RELAY_HEADER_BYTES + RELAY_ENDPOINT_PAYLOAD_BYTES
        )
        self._batch_ok = List[Bool](capacity=64)

    def enable_cookie_auth(
        mut self, secret: AeadKey, bucket_ms: Float64 = 10000.0
    ):
        self.require_cookie = True
        self.cookie_auth = StatelessCookieAuth(secret, bucket_ms)

    def _find_peer_slot(self, session_token: UInt64, peer_id: Int) -> Int:
        for i in range(len(self.peers)):
            if (
                self.peers[i].active
                and self.peers[i].session_token == session_token
                and self.peers[i].peer_id == peer_id
            ):
                return i
        return -1

    def register_peer(
        mut self,
        session_token: UInt64,
        peer_id: Int,
        endpoint: NetEndpoint,
        now_ms: Float64 = 0.0,
    ) -> Bool:
        if session_token == 0 or peer_id < 0 or peer_id >= 255 or endpoint.port == 0:
            return False
        var existing = self._find_peer_slot(session_token, peer_id)
        if existing >= 0:
            if (
                self.peers[existing].endpoint != endpoint
                and (now_ms - self.peers[existing].last_seen_ms)
                < self.peer_timeout_ms
            ):
                self.spoofed_packets_dropped += 1
                return False
            self.peers[existing].endpoint = endpoint
            self.peers[existing].last_seen_ms = now_ms
            return True

        var free_slot = -1
        for i in range(len(self.peers)):
            if not self.peers[i].active:
                free_slot = i
                break
        if free_slot < 0:
            if len(self.peers) >= self.max_peers:
                return False
            free_slot = len(self.peers)
            self.peers.append(_RelayPeerEntry())

        self.peers[free_slot].active = True
        self.peers[free_slot].session_token = session_token
        self.peers[free_slot].peer_id = peer_id
        self.peers[free_slot].endpoint = endpoint
        self.peers[free_slot].last_seen_ms = now_ms
        self.registrations += 1
        return True

    def lookup_peer(
        self, session_token: UInt64, peer_id: Int
    ) -> Tuple[Bool, NetEndpoint]:
        var slot = self._find_peer_slot(session_token, peer_id)
        if slot < 0:
            return (False, NetEndpoint())
        return (True, self.peers[slot].endpoint)

    def active_peer_count(self) -> Int:
        var c = 0
        for i in range(len(self.peers)):
            if self.peers[i].active:
                c += 1
        return c

    def _send_control[TransT: NetTransport](
        mut self,
        mut transport: TransT,
        dst_ep: NetEndpoint,
        hdr: RelayForwardHeader,
    ):
        hdr.write_into(self._ctrl_scratch)
        _ = transport.queue_send_raw(
            dst_ep, Int(self._ctrl_scratch.unsafe_ptr()), RELAY_HEADER_BYTES
        )

    def _send_peer_endpoint[TransT: NetTransport](
        mut self,
        mut transport: TransT,
        dst_ep: NetEndpoint,
        session_token: UInt64,
        about_peer_id: Int,
        dst_peer_id: Int,
        about_ep: NetEndpoint,
    ):
        var hdr = RelayForwardHeader(
            session_token=session_token,
            src_peer=about_peer_id,
            dst_peer=dst_peer_id,
            flags=RELAY_FLAG_PEER_ENDPOINT,
            seq=0,
            hop_ttl=0,
        )
        hdr.write_into(self._ctrl_scratch)
        _encode_endpoint_at(
            Int(self._ctrl_scratch.unsafe_ptr()), RELAY_HEADER_BYTES, about_ep
        )
        _ = transport.queue_send_raw(
            dst_ep,
            Int(self._ctrl_scratch.unsafe_ptr()),
            RELAY_HEADER_BYTES + RELAY_ENDPOINT_PAYLOAD_BYTES,
        )

    def poll[TransT: NetTransport](
        mut self, mut transport: TransT, now_ms: Float64
    ):
        """Drains incoming relay traffic: binds endpoints, prevents spoofing, and forwards DATA."""
        # Evict stale peer bindings before accepting registrations or advertising peer endpoints
        for i in range(len(self.peers)):
            if (
                self.peers[i].active
                and (now_ms - self.peers[i].last_seen_ms) >= self.peer_timeout_ms
            ):
                self.peers[i].active = False

        while True:
            var n = transport.recv_batch(self._rx_views)
            if n <= 0:
                break
            self._batch_ok.clear()
            for _ in range(n):
                self._batch_ok.append(False)

            # Pass 1: authenticate & register endpoints before forwarding DATA in the same batch
            for i in range(n):
                var view = self._rx_views[i]
                var parsed = RelayForwardHeader.parse_from_ptr(
                    view.data_ptr, view.length
                )
                if not parsed[0]:
                    self.invalid_packets_dropped += 1
                    continue
                var hdr = parsed[1]
                if (
                    hdr.session_token == 0
                    or hdr.src_peer < 0
                    or hdr.src_peer >= 255
                ):
                    self.invalid_packets_dropped += 1
                    continue

                var slot = self._find_peer_slot(hdr.session_token, hdr.src_peer)
                if slot >= 0:
                    if (
                        self.peers[slot].endpoint != view.sender
                        and (now_ms - self.peers[slot].last_seen_ms)
                        < self.peer_timeout_ms
                    ):
                        self.spoofed_packets_dropped += 1
                        continue
                    self.peers[slot].endpoint = view.sender
                    self.peers[slot].last_seen_ms = now_ms
                else:
                    if self.require_cookie:
                        var has_cookie = (
                            (hdr.flags & RELAY_FLAG_REGISTER) != 0
                            and view.length >= RELAY_HEADER_BYTES + 16
                            and self.cookie_auth.verify_cookie(
                                view.sender,
                                view.data_ptr + RELAY_HEADER_BYTES,
                                now_ms,
                            )
                        )
                        if not has_cookie:
                            if (hdr.flags & RELAY_FLAG_REGISTER) != 0:
                                var c = self.cookie_auth.issue_cookie(
                                    view.sender, now_ms
                                )
                                var ch_hdr = RelayForwardHeader(
                                    session_token=hdr.session_token,
                                    src_peer=hdr.src_peer,
                                    dst_peer=hdr.src_peer,
                                    flags=RELAY_FLAG_COOKIE_CHALLENGE,
                                    seq=hdr.seq,
                                    hop_ttl=0,
                                )
                                ch_hdr.write_into(self._ctrl_scratch)
                                for b in range(16):
                                    self._ctrl_scratch[RELAY_HEADER_BYTES + b] = c[b]
                                _ = transport.queue_send_raw(
                                    view.sender,
                                    Int(self._ctrl_scratch.unsafe_ptr()),
                                    RELAY_HEADER_BYTES + 16,
                                )
                            else:
                                self.spoofed_packets_dropped += 1
                            continue
                    if not self.register_peer(
                        hdr.session_token, hdr.src_peer, view.sender, now_ms
                    ):
                        self.invalid_packets_dropped += 1
                        continue

                self._batch_ok[i] = True
                if (hdr.flags & RELAY_FLAG_REGISTER) != 0:
                    var ack = RelayForwardHeader(
                        session_token=hdr.session_token,
                        src_peer=hdr.src_peer,
                        dst_peer=hdr.src_peer,
                        flags=RELAY_FLAG_REGISTER_ACK,
                        seq=hdr.seq,
                        hop_ttl=0,
                    )
                    self._send_control(transport, view.sender, ack)
                    # Advertise known peers in the same session to enable direct NAT hole punching
                    for p_i in range(len(self.peers)):
                        if not self.peers[p_i].active:
                            continue
                        if (
                            self.peers[p_i].session_token == hdr.session_token
                            and self.peers[p_i].peer_id != hdr.src_peer
                        ):
                            var peer_id = self.peers[p_i].peer_id
                            var peer_ep = self.peers[p_i].endpoint
                            self._send_peer_endpoint(
                                transport,
                                view.sender,
                                hdr.session_token,
                                peer_id,
                                hdr.src_peer,
                                peer_ep,
                            )
                            self._send_peer_endpoint(
                                transport,
                                peer_ep,
                                hdr.session_token,
                                hdr.src_peer,
                                peer_id,
                                view.sender,
                            )

            # Pass 2: forward authenticated DATA packets
            for i in range(n):
                if not self._batch_ok[i]:
                    continue
                var view = self._rx_views[i]
                var hdr = RelayForwardHeader.parse_from_ptr(
                    view.data_ptr, view.length
                )[1]
                if hdr.flags != RELAY_FLAG_DATA:
                    continue
                if hdr.hop_ttl == 0 or hdr.dst_peer == hdr.src_peer or hdr.dst_peer >= 255:
                    self.invalid_packets_dropped += 1
                    continue
                var dst = self.lookup_peer(hdr.session_token, hdr.dst_peer)
                if not dst[0]:
                    self.unroutable_drops += 1
                    continue
                if view.length <= RELAY_HEADER_BYTES:
                    self.invalid_packets_dropped += 1
                    continue
                self._tx_scratch.resize(unsafe_uninit_length=view.length)
                _ = external_call["memcpy", Int](
                    Int(self._tx_scratch.unsafe_ptr()),
                    view.data_ptr,
                    Int64(view.length),
                )
                self._tx_scratch[15] = hdr.hop_ttl - 1
                if transport.queue_send_raw(
                    dst[1],
                    Int(self._tx_scratch.unsafe_ptr()),
                    view.length,
                ):
                    self.packets_forwarded += 1
                    self.bytes_forwarded += view.length

        _ = transport.flush_sends()


struct NATPunchProbe(Copyable, Movable):
    """UDP NAT hole-punching state machine with relay fallback and idle downgrade."""
    var local_peer: Int
    var target_peer: Int
    var relay_endpoint: NetEndpoint
    var target_endpoint: NetEndpoint
    var session_token: UInt64
    var state: Int
    var local_nonce: UInt64
    var first_probe_ms: Float64
    var last_probe_sent_ms: Float64
    var last_direct_rx_ms: Float64
    var probe_interval_ms: Float64
    var probe_timeout_ms: Float64
    var keepalive_interval_ms: Float64
    var direct_idle_fallback_ms: Float64
    var probes_sent: Int
    var acks_sent: Int
    var probes_received: Int
    var acks_received: Int
    var direct_upgrades: Int
    var relay_fallbacks: Int
    var has_relay_cookie: Bool
    var relay_cookie: Array[UInt8, 16]
    var tx_scratch: List[UInt8]

    def __init__(
        out self,
        local_peer: Int = 0,
        relay_endpoint: NetEndpoint = NetEndpoint(),
        target_peer: Int = -1,
        target_endpoint: NetEndpoint = NetEndpoint(),
        session_token: UInt64 = RELAY_DEFAULT_SESSION_TOKEN,
        probe_interval_ms: Float64 = NAT_PUNCH_DEFAULT_INTERVAL_MS,
        probe_timeout_ms: Float64 = NAT_PUNCH_DEFAULT_TIMEOUT_MS,
        direct_idle_fallback_ms: Float64 = NAT_DIRECT_IDLE_FALLBACK_MS,
    ):
        self.local_peer = local_peer
        self.target_peer = target_peer
        self.relay_endpoint = relay_endpoint
        self.target_endpoint = target_endpoint
        self.session_token = (
            session_token
            if session_token != UInt64(0)
            else RELAY_DEFAULT_SESSION_TOKEN
        )
        self.state = PUNCH_STATE_IDLE
        self.local_nonce = UInt64(1)
        self.first_probe_ms = -1.0
        self.last_probe_sent_ms = -1.0e9
        self.last_direct_rx_ms = -1.0
        self.probe_interval_ms = max(5.0, probe_interval_ms)
        self.probe_timeout_ms = max(20.0, probe_timeout_ms)
        self.keepalive_interval_ms = NAT_PUNCH_KEEPALIVE_MS
        self.direct_idle_fallback_ms = max(50.0, direct_idle_fallback_ms)
        self.probes_sent = 0
        self.acks_sent = 0
        self.probes_received = 0
        self.acks_received = 0
        self.direct_upgrades = 0
        self.relay_fallbacks = 0
        self.has_relay_cookie = False
        self.relay_cookie = Array[UInt8, 16](fill=UInt8(0))
        self.tx_scratch = List[UInt8](
            unsafe_uninit_length=RELAY_HEADER_BYTES + 16
        )
        var seed = (
            self.session_token
            ^ (UInt64(local_peer + 1) * UInt64(0x9E3779B97F4A7C15))
            ^ (UInt64(target_peer + 2) * UInt64(0xBF58476D1CE4E5B9))
            ^ mix64(UInt64(monotonic()))
            ^ mix64(UInt64(Int(Pointer(to=self))))
        )
        self.local_nonce = mix64(seed) | UInt64(1)

    def set_relay_cookie(mut self, cookie: Array[UInt8, 16]):
        self.relay_cookie = cookie.copy()
        self.has_relay_cookie = True

    def set_target(mut self, target_peer: Int, target_endpoint: NetEndpoint):
        self.target_peer = target_peer
        self.target_endpoint = target_endpoint

    def restart_probe(mut self, now_ms: Float64):
        self.state = PUNCH_STATE_PROBING
        self.first_probe_ms = now_ms
        self.last_probe_sent_ms = -1.0e9

    @always_inline
    def is_direct(self) -> Bool:
        return self.state == PUNCH_STATE_DIRECT

    @always_inline
    def is_relay_fallback(self) -> Bool:
        return self.state == PUNCH_STATE_RELAY_FALLBACK

    def _pack_punch(
        mut self, kind: UInt8, dst_peer: Int, nonce: UInt64
    ):
        self.tx_scratch.resize(unsafe_uninit_length=NAT_PUNCH_PACKET_BYTES)
        var base = Int(self.tx_scratch.unsafe_ptr())
        _r_store_u32_le(base, 0, NAT_PUNCH_MAGIC)
        _r_store_u64_le(base, 4, self.session_token)
        var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base)
        p[unsafe_offset=12] = kind
        p[unsafe_offset=13] = UInt8(self.local_peer & 0xFF)
        p[unsafe_offset=14] = UInt8(dst_peer & 0xFF)
        p[unsafe_offset=15] = 0
        _r_store_u64_le(base, 16, nonce)

    def send_probe[TransT: NetTransport](
        mut self, mut transport: TransT, now_ms: Float64
    ) -> Bool:
        """Sends a direct UDP NAT punch probe (or relay registration if target endpoint is unknown)."""
        if self.target_endpoint.port != 0:
            if self.state == PUNCH_STATE_IDLE:
                self.state = PUNCH_STATE_PROBING
            if self.first_probe_ms < 0.0:
                self.first_probe_ms = now_ms
            var dst_p = self.target_peer if self.target_peer >= 0 else 0
            self._pack_punch(PUNCH_KIND_PROBE, dst_p, self.local_nonce)
            if transport.send_raw(
                self.target_endpoint,
                Int(self.tx_scratch.unsafe_ptr()),
                NAT_PUNCH_PACKET_BYTES,
            ):
                self.probes_sent += 1
                self.last_probe_sent_ms = now_ms
                return True
            return False

        if self.relay_endpoint.port != 0:
            if self.state == PUNCH_STATE_IDLE:
                self.state = PUNCH_STATE_PROBING
            if self.first_probe_ms < 0.0:
                self.first_probe_ms = now_ms
            var reg = RelayForwardHeader(
                session_token=self.session_token,
                src_peer=self.local_peer,
                dst_peer=self.local_peer,
                flags=RELAY_FLAG_REGISTER,
                seq=0,
                hop_ttl=0,
            )
            var reg_len = RELAY_HEADER_BYTES + (
                16 if self.has_relay_cookie else 0
            )
            if len(self.tx_scratch) < reg_len:
                self.tx_scratch.resize(unsafe_uninit_length=reg_len)
            reg.write_into(self.tx_scratch)
            if self.has_relay_cookie:
                for b in range(16):
                    self.tx_scratch[RELAY_HEADER_BYTES + b] = self.relay_cookie[b]
            if transport.send_raw(
                self.relay_endpoint,
                Int(self.tx_scratch.unsafe_ptr()),
                reg_len,
            ):
                self.probes_sent += 1
                self.last_probe_sent_ms = now_ms
                return True
            return False

        return False

    def send_keepalive[TransT: NetTransport](
        mut self, mut transport: TransT, now_ms: Float64
    ) -> Bool:
        if self.target_endpoint.port == 0:
            return False
        var dst_p = self.target_peer if self.target_peer >= 0 else 0
        self._pack_punch(PUNCH_KIND_KEEPALIVE, dst_p, self.local_nonce)
        if transport.send_raw(
            self.target_endpoint,
            Int(self.tx_scratch.unsafe_ptr()),
            NAT_PUNCH_PACKET_BYTES,
        ):
            self.last_probe_sent_ms = now_ms
            return True
        return False

    def ingest_datagram[TransT: NetTransport](
        mut self, view: DatagramView, mut transport: TransT, now_ms: Float64
    ) -> Bool:
        if view.data_ptr == 0 or view.length < NAT_PUNCH_PACKET_BYTES:
            return False
        if _r_load_u32_le(view.data_ptr, 0) != NAT_PUNCH_MAGIC:
            return False
        var tok = _r_load_u64_le(view.data_ptr, 4)
        if tok != self.session_token:
            return False
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=view.data_ptr)
        var kind = p[unsafe_offset=12]
        var src_p = Int(p[unsafe_offset=13])
        var dst_p = Int(p[unsafe_offset=14])
        var nonce = _r_load_u64_le(view.data_ptr, 16)

        if dst_p != self.local_peer:
            return False
        if self.target_peer >= 0 and src_p != self.target_peer:
            return False

        if kind == PUNCH_KIND_PROBE:
            if (
                self.state == PUNCH_STATE_DIRECT
                and view.sender != self.target_endpoint
            ):
                return False
            self.target_endpoint = view.sender
            self.last_direct_rx_ms = now_ms
            self.probes_received += 1
            self._pack_punch(PUNCH_KIND_ACK, src_p, nonce)
            if transport.send_raw(
                view.sender,
                Int(self.tx_scratch.unsafe_ptr()),
                NAT_PUNCH_PACKET_BYTES,
            ):
                self.acks_sent += 1
            return True

        if kind == PUNCH_KIND_ACK:
            if nonce != self.local_nonce:
                return False
            self.target_endpoint = view.sender
            self.last_direct_rx_ms = now_ms
            self.acks_received += 1
            if self.state != PUNCH_STATE_DIRECT:
                self.state = PUNCH_STATE_DIRECT
                self.direct_upgrades += 1
            return True

        if kind == PUNCH_KIND_KEEPALIVE:
            if view.sender == self.target_endpoint:
                self.last_direct_rx_ms = now_ms
                return True
            return False

        return False

    def poll[TransT: NetTransport](
        mut self, mut transport: TransT, now_ms: Float64
    ):
        if self.state == PUNCH_STATE_IDLE or self.state == PUNCH_STATE_PROBING:
            if (
                self.first_probe_ms >= 0.0
                and (now_ms - self.first_probe_ms) >= self.probe_timeout_ms
            ):
                self.state = PUNCH_STATE_RELAY_FALLBACK
                self.relay_fallbacks += 1
                return
            if (now_ms - self.last_probe_sent_ms) >= self.probe_interval_ms:
                _ = self.send_probe(transport, now_ms)
            return

        if self.state == PUNCH_STATE_DIRECT:
            if (
                self.last_direct_rx_ms >= 0.0
                and (now_ms - self.last_direct_rx_ms)
                >= self.direct_idle_fallback_ms
            ):
                self.state = PUNCH_STATE_RELAY_FALLBACK
                self.relay_fallbacks += 1
                return
            if (now_ms - self.last_probe_sent_ms) >= self.keepalive_interval_ms:
                _ = self.send_keepalive(transport, now_ms)
