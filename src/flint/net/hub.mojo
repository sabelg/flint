"""Flint v5 `NetHub`: one datagram transport shared by a reliable-ordered `NetChannel` per peer.

A game usually has one socket and several peers. `NetHub[T: NetTransport]` owns the transport
(`UdpSocket` or a `VirtualEndpoint` on `VirtualNetSim`) and one `NetChannel` per remote address:

- `poll(now_ms)` drains the transport once, routes every valid Flint datagram to its sender's
  channel via an O(1) `Dict[NetEndpoint, Int]` index, evicts idle peers, then lets every channel
  queue its resends/acks and flushes `sendmmsg` once per `poll()`.
- `accept = True` (a host / server) opens a channel the first time an unknown address sends a
  valid packet (or completes the optional stateless cookie handshake when `require_cookie = True`);
  those ids are listed in `accepted` until the next `poll()`. With `accept = False` (a client),
  datagrams from unknown senders are ignored (except `CHAN_HANDSHAKE_COOKIE` replies).
- `connect(endpoint)` opens (or finds) a channel by hand; `drop(cid)` forgets one, and a later
  packet from that address starts a fresh channel in the same slot.
- Channel ids are small stable integers (slots are reused after `drop`), at most `max_channels`.
"""
from std.collections import Dict, List
from flint.net.crypto import AeadKey, StatelessCookieAuth
from flint.net.socket_udp import DatagramView, NetEndpoint, NetTransport, UDP_RING_CAPACITY
from flint.net.channel import (
    CHAN_HANDSHAKE_COOKIE,
    CHAN_HANDSHAKE_ECHO,
    CHAN_HANDSHAKE_INIT,
    IPV6_SAFE_MTU_BYTES,
    NetChannel,
    PacketWireHeader,
    WIRE_HEADER_BYTES,
    WIRE_MAGIC_VER,
)

comptime NET_HUB_DEFAULT_MAX_CHANNELS: Int = 256
comptime NET_HUB_DEFAULT_IDLE_TIMEOUT_MS: Float64 = 30000.0
comptime _COOKIE_BYTES: Int = 16


struct NetHub[T: NetTransport & Deinitable](Movable):
    """One transport, a `NetChannel` per peer, O(1) endpoint routing & single-flush batching."""
    var transport: Self.T
    var chans: List[NetChannel]
    var alive: List[Bool]
    var last_rx_ms: List[Float64]
    var endpoint_to_cid: Dict[NetEndpoint, Int]
    var count: Int
    var accept: Bool
    var max_channels: Int
    var mtu_bytes: Int
    var idle_timeout_ms: Float64
    var accepted: List[Int]
    var evicted: List[Int]
    var datagrams_routed: Int
    var datagrams_ignored: Int
    var cookies_sent: Int
    var cookies_rejected: Int
    var require_cookie: Bool
    var cookie_auth: StatelessCookieAuth
    var default_crypto_enabled: Bool
    var default_aead_key: AeadKey
    var _views: List[DatagramView]
    var _hs_scratch: List[UInt8]

    def __init__(
        out self,
        var transport: Self.T,
        accept: Bool,
        max_channels: Int = NET_HUB_DEFAULT_MAX_CHANNELS,
        mtu_bytes: Int = IPV6_SAFE_MTU_BYTES,
        idle_timeout_ms: Float64 = NET_HUB_DEFAULT_IDLE_TIMEOUT_MS,
    ):
        self.transport = transport^
        self.chans = List[NetChannel]()
        self.alive = List[Bool]()
        self.last_rx_ms = List[Float64]()
        self.endpoint_to_cid = Dict[NetEndpoint, Int]()
        self.count = 0
        self.accept = accept
        self.max_channels = max(1, max_channels)
        self.mtu_bytes = mtu_bytes
        self.idle_timeout_ms = idle_timeout_ms
        self.accepted = List[Int]()
        self.evicted = List[Int]()
        self.datagrams_routed = 0
        self.datagrams_ignored = 0
        self.cookies_sent = 0
        self.cookies_rejected = 0
        self.require_cookie = False
        self.cookie_auth = StatelessCookieAuth()
        self.default_crypto_enabled = False
        self.default_aead_key = AeadKey()
        self._views = List[DatagramView](capacity=UDP_RING_CAPACITY)
        self._hs_scratch = List[UInt8](capacity=64)

    def enable_cookie_handshake(
        mut self, secret: AeadKey, bucket_ms: Float64 = 10000.0
    ):
        """Requires unknown peers to echo a stateless Poly1305 cookie before allocating a slot."""
        self.require_cookie = True
        self.cookie_auth = StatelessCookieAuth(secret, bucket_ms)

    def enable_channel_encryption(mut self, key: AeadKey):
        """Enables ChaCha20-Poly1305 AEAD on all existing and future channels in this hub."""
        self.default_crypto_enabled = True
        self.default_aead_key = key
        for i in range(self.count):
            if self.alive[i]:
                self.chans[i].enable_encryption(key)

    def local_endpoint(self) -> NetEndpoint:
        return self.transport.local_endpoint()

    def is_open(self, cid: Int) -> Bool:
        return cid >= 0 and cid < self.count and self.alive[cid]

    def open_count(self) -> Int:
        var n = 0
        for i in range(self.count):
            if self.alive[i]:
                n += 1
        return n

    def find(self, e: NetEndpoint) -> Int:
        """O(1) lookup of the open channel for that address, or -1."""
        return self.endpoint_to_cid.get(e, -1)

    def _init_channel(self, e: NetEndpoint) -> NetChannel:
        var ch = NetChannel(e, self.mtu_bytes)
        if self.default_crypto_enabled:
            ch.enable_encryption(self.default_aead_key)
        return ch^

    def connect(mut self, e: NetEndpoint) -> Int:
        """The channel id for that address (opened if new, reusing a dropped slot); -1 when
        `max_channels` are already open."""
        var found = self.find(e)
        if found >= 0:
            return found
        for i in range(self.count):
            if not self.alive[i]:
                self.chans[i] = self._init_channel(e)
                self.alive[i] = True
                self.last_rx_ms[i] = -1.0
                self.endpoint_to_cid[e] = i
                return i
        if self.count >= self.max_channels:
            return -1
        var cid = self.count
        self.chans.append(self._init_channel(e))
        self.alive.append(True)
        self.last_rx_ms.append(-1.0)
        self.endpoint_to_cid[e] = cid
        self.count += 1
        return cid

    def drop(mut self, cid: Int):
        """Forget a channel (its queued messages too)."""
        if self.is_open(cid):
            var ep = self.chans[cid].peer_endpoint
            self.alive[cid] = False
            self.last_rx_ms[cid] = -1.0
            _ = self.endpoint_to_cid.pop(ep, -1)

    def _evict_idle_channels(mut self, now_ms: Float64):
        if self.idle_timeout_ms <= 0.0:
            return
        for i in range(self.count):
            if (
                self.alive[i]
                and self.last_rx_ms[i] >= 0.0
                and (now_ms - self.last_rx_ms[i]) >= self.idle_timeout_ms
            ):
                var ep = self.chans[i].peer_endpoint
                self.alive[i] = False
                self.last_rx_ms[i] = -1.0
                _ = self.endpoint_to_cid.pop(ep, -1)
                self.evicted.append(i)

    def send_handshake_init(mut self, dst: NetEndpoint) -> Bool:
        """Client helper: sends a stateless `CHAN_HANDSHAKE_INIT` probe to a cookie-gated server."""
        self._hs_scratch.clear()
        var hdr = PacketWireHeader(WIRE_MAGIC_VER, CHAN_HANDSHAKE_INIT, 0, 0, 0)
        hdr.write_into(self._hs_scratch)
        var ok = self.transport.queue_send_raw(
            dst, Int(self._hs_scratch.unsafe_ptr()), len(self._hs_scratch)
        )
        _ = len(self._hs_scratch)
        _ = self.transport.flush_sends()
        return ok

    def _queue_cookie_reply(mut self, dst: NetEndpoint, now_ms: Float64):
        var cookie = self.cookie_auth.issue_cookie(dst, now_ms)
        self._hs_scratch.clear()
        var hdr = PacketWireHeader(
            WIRE_MAGIC_VER, CHAN_HANDSHAKE_COOKIE, 0, 0, 0
        )
        hdr.write_into(self._hs_scratch)
        for i in range(_COOKIE_BYTES):
            self._hs_scratch.append(cookie[i])
        if self.transport.queue_send_raw(
            dst, Int(self._hs_scratch.unsafe_ptr()), len(self._hs_scratch)
        ):
            self.cookies_sent += 1
        _ = len(self._hs_scratch)

    def _queue_cookie_echo(mut self, dst: NetEndpoint, cookie_ptr: Int):
        self._hs_scratch.clear()
        var hdr = PacketWireHeader(WIRE_MAGIC_VER, CHAN_HANDSHAKE_ECHO, 0, 0, 0)
        hdr.write_into(self._hs_scratch)
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=cookie_ptr)
        for i in range(_COOKIE_BYTES):
            self._hs_scratch.append(p[unsafe_offset=i])
        _ = self.transport.queue_send_raw(
            dst, Int(self._hs_scratch.unsafe_ptr()), len(self._hs_scratch)
        )
        _ = len(self._hs_scratch)

    def poll(mut self, now_ms: Float64):
        """Receive and route everything waiting, evict idle peers, resend/ack on every channel,
        and flush `sendmmsg` once per poll."""
        self.accepted.clear()
        self.evicted.clear()
        while True:
            var n = self.transport.recv_batch(self._views)
            if n <= 0:
                break
            for k in range(n):
                var v = self._views[k]
                if v.length < WIRE_HEADER_BYTES or v.byte_at(0) != WIRE_MAGIC_VER:
                    self.datagrams_ignored += 1
                    continue
                var kind = v.byte_at(1) & 0x0F

                # Client side: automatically echo stateless server handshake cookies
                if kind == CHAN_HANDSHAKE_COOKIE:
                    if v.length >= WIRE_HEADER_BYTES + _COOKIE_BYTES:
                        self._queue_cookie_echo(
                            v.sender, v.data_ptr + WIRE_HEADER_BYTES
                        )
                    continue

                var cid = self.find(v.sender)
                if cid < 0:
                    if not self.accept:
                        self.datagrams_ignored += 1
                        continue
                    if self.require_cookie:
                        if kind == CHAN_HANDSHAKE_ECHO:
                            if (
                                v.length < WIRE_HEADER_BYTES + _COOKIE_BYTES
                                or not self.cookie_auth.verify_cookie(
                                    v.sender,
                                    v.data_ptr + WIRE_HEADER_BYTES,
                                    now_ms,
                                )
                            ):
                                self.cookies_rejected += 1
                                self.datagrams_ignored += 1
                                continue
                        else:
                            # Stateless cookie challenge: zero channel slots allocated!
                            self._queue_cookie_reply(v.sender, now_ms)
                            continue
                    cid = self.connect(v.sender)
                    if cid < 0 and self.idle_timeout_ms > 0.0:
                        self._evict_idle_channels(now_ms)
                        cid = self.connect(v.sender)
                    if cid < 0:
                        self.datagrams_ignored += 1
                        continue
                    self.accepted.append(cid)
                    self.last_rx_ms[cid] = now_ms
                    if kind == CHAN_HANDSHAKE_ECHO:
                        self.datagrams_routed += 1
                        continue

                self.last_rx_ms[cid] = now_ms
                if (
                    kind == CHAN_HANDSHAKE_INIT
                    or kind == CHAN_HANDSHAKE_ECHO
                ):
                    self.datagrams_routed += 1
                    continue
                _ = self.chans[cid].ingest_datagram(v, now_ms)
                self.datagrams_routed += 1

        for i in range(self.count):
            if self.alive[i] and self.last_rx_ms[i] < 0.0:
                self.last_rx_ms[i] = now_ms
        self._evict_idle_channels(now_ms)
        for i in range(self.count):
            if self.alive[i]:
                self.chans[i].poll(self.transport, now_ms, False, False)
        _ = self.transport.flush_sends()

    def send_reliable(mut self, cid: Int, message: List[UInt8]) -> Bool:
        """Queue a reliable-ordered message of any size (sent on the next `poll()`)."""
        if not self.is_open(cid):
            return False
        return self.chans[cid].send_reliable(message)

    def send_reliable_unordered(mut self, cid: Int, message: List[UInt8]) -> Bool:
        """Queue a reliable-unordered message (delivered immediately without HoL blocking)."""
        if not self.is_open(cid):
            return False
        return self.chans[cid].send_reliable_unordered(message)

    def send_unreliable(mut self, cid: Int, payload: List[UInt8], now_ms: Float64) -> Bool:
        """Send a latest-only datagram now (must fit one packet)."""
        if not self.is_open(cid):
            return False
        var ok = self.chans[cid].send_unreliable_sequenced(payload, self.transport, now_ms)[0]
        _ = self.transport.flush_sends()
        return ok

    def send_unreliable_raw(mut self, cid: Int, payload: List[UInt8], now_ms: Float64) -> Bool:
        """Send an unreliable-raw datagram now (must fit one packet)."""
        if not self.is_open(cid):
            return False
        var ok = self.chans[cid].send_unreliable_raw(payload, self.transport, now_ms)[0]
        _ = self.transport.flush_sends()
        return ok

    def pop_reliable(mut self, cid: Int) -> Tuple[Bool, List[UInt8]]:
        if not self.is_open(cid):
            return (False, List[UInt8]())
        return self.chans[cid].pop_reliable_message()

    def pop_reliable_unordered(mut self, cid: Int) -> Tuple[Bool, List[UInt8]]:
        if not self.is_open(cid):
            return (False, List[UInt8]())
        return self.chans[cid].pop_reliable_unordered_message()

    def pop_unreliable(mut self, cid: Int) -> Tuple[Bool, List[UInt8]]:
        if not self.is_open(cid):
            return (False, List[UInt8]())
        return self.chans[cid].pop_unreliable_message()

    def pop_unreliable_raw(mut self, cid: Int) -> Tuple[Bool, List[UInt8]]:
        if not self.is_open(cid):
            return (False, List[UInt8]())
        return self.chans[cid].pop_unreliable_raw_message()
