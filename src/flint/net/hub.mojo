"""Flint v5 `NetHub`: one datagram transport shared by a reliable-ordered `NetChannel` per peer.

A game usually has one socket and several peers. `NetHub[T: NetTransport]` owns the transport
(`UdpSocket` or a `VirtualEndpoint` on `VirtualNetSim`) and one `NetChannel` per remote address:

- `poll(now_ms)` drains the transport once, routes every valid Flint datagram to its sender's
  channel, then lets every channel resend what is due and flush its acks.
- `accept = True` (a host / server) opens a channel the first time an unknown address sends a
  valid packet; those ids are listed in `accepted` until the next `poll()`. With `accept = False`
  (a client) datagrams from unknown senders are ignored.
- `connect(endpoint)` opens (or finds) a channel by hand; `drop(cid)` forgets one, and a later
  packet from that address starts a fresh channel in the same slot.
- Channel ids are small stable integers (slots are reused after `drop`), at most `max_channels`.

Game-agnostic: no message format is imposed; payloads are raw bytes. Upstreamed from flint-py's
`capi/net_api.mojo` (the Python multiplayer and rollback netplay layer runs on it).
"""
from std.collections import List
from flint.net.socket_udp import DatagramView, NetEndpoint, NetTransport, UDP_RING_CAPACITY
from flint.net.channel import IPV6_SAFE_MTU_BYTES, NetChannel, WIRE_HEADER_BYTES, WIRE_MAGIC_VER

comptime NET_HUB_DEFAULT_MAX_CHANNELS: Int = 256
comptime NET_HUB_DEFAULT_IDLE_TIMEOUT_MS: Float64 = 30000.0


struct NetHub[T: NetTransport & Deinitable](Movable):
    """One transport, a `NetChannel` per peer, routing by sender address."""
    var transport: Self.T
    var chans: List[NetChannel]
    var alive: List[Bool]
    var last_rx_ms: List[Float64]
    var count: Int
    var accept: Bool
    var max_channels: Int
    var mtu_bytes: Int
    var idle_timeout_ms: Float64
    var accepted: List[Int]
    var evicted: List[Int]
    var datagrams_routed: Int
    var datagrams_ignored: Int
    var _views: List[DatagramView]

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
        self.count = 0
        self.accept = accept
        self.max_channels = max(1, max_channels)
        self.mtu_bytes = mtu_bytes
        self.idle_timeout_ms = idle_timeout_ms
        self.accepted = List[Int]()
        self.evicted = List[Int]()
        self.datagrams_routed = 0
        self.datagrams_ignored = 0
        self._views = List[DatagramView](capacity=UDP_RING_CAPACITY)

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
        """The open channel for that address, or -1."""
        for i in range(self.count):
            if self.alive[i] and self.chans[i].peer_endpoint == e:
                return i
        return -1

    def connect(mut self, e: NetEndpoint) -> Int:
        """The channel id for that address (opened if new, reusing a dropped slot); -1 when
        `max_channels` are already open."""
        var found = self.find(e)
        if found >= 0:
            return found
        for i in range(self.count):
            if not self.alive[i]:
                self.chans[i] = NetChannel(e, self.mtu_bytes)
                self.alive[i] = True
                self.last_rx_ms[i] = -1.0
                return i
        if self.count >= self.max_channels:
            return -1
        self.chans.append(NetChannel(e, self.mtu_bytes))
        self.alive.append(True)
        self.last_rx_ms.append(-1.0)
        self.count += 1
        return self.count - 1

    def drop(mut self, cid: Int):
        """Forget a channel (its queued messages too)."""
        if self.is_open(cid):
            self.alive[cid] = False
            self.last_rx_ms[cid] = -1.0

    def _evict_idle_channels(mut self, now_ms: Float64):
        if self.idle_timeout_ms <= 0.0:
            return
        for i in range(self.count):
            if (
                self.alive[i]
                and self.last_rx_ms[i] >= 0.0
                and (now_ms - self.last_rx_ms[i]) >= self.idle_timeout_ms
            ):
                self.alive[i] = False
                self.last_rx_ms[i] = -1.0
                self.evicted.append(i)

    def poll(mut self, now_ms: Float64):
        """Receive and route everything waiting, evict idle peers, then resend/ack/flush."""
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
                var cid = self.find(v.sender)
                if cid < 0:
                    if not self.accept:
                        self.datagrams_ignored += 1
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
                _ = self.chans[cid].ingest_datagram(v, now_ms)
                self.datagrams_routed += 1
        for i in range(self.count):
            if self.alive[i] and self.last_rx_ms[i] < 0.0:
                self.last_rx_ms[i] = now_ms
        self._evict_idle_channels(now_ms)
        for i in range(self.count):
            if self.alive[i]:
                self.chans[i].poll(self.transport, now_ms, False)

    def send_reliable(mut self, cid: Int, message: List[UInt8]) -> Bool:
        """Queue a reliable-ordered message of any size (sent on the next `poll()`)."""
        if not self.is_open(cid):
            return False
        return self.chans[cid].send_reliable(message)

    def send_unreliable(mut self, cid: Int, payload: List[UInt8], now_ms: Float64) -> Bool:
        """Send a latest-only datagram now (must fit one packet)."""
        if not self.is_open(cid):
            return False
        var ok = self.chans[cid].send_unreliable_sequenced(payload, self.transport, now_ms)[0]
        _ = self.transport.flush_sends()
        return ok

    def pop_reliable(mut self, cid: Int) -> Tuple[Bool, List[UInt8]]:
        if not self.is_open(cid):
            return (False, List[UInt8]())
        return self.chans[cid].pop_reliable_message()

    def pop_unreliable(mut self, cid: Int) -> Tuple[Bool, List[UInt8]]:
        if not self.is_open(cid):
            return (False, List[UInt8]())
        return self.chans[cid].pop_unreliable_message()
