"""Dual-stack IPv6 + IPv4-mapped non-blocking POSIX UDP transport (`flint.net.socket_udp`).

Implements:
- `NetEndpoint`: 128-bit dual-stack IPv6 + IPv4-mapped (`::ffff:a.b.c.d`) socket address.
- `DatagramView`: zero-copy view into pre-allocated kernel RX rings with full sender `NetEndpoint`.
- `trait NetTransport`: polymorphic datagram interface shared by `UdpSocket` and `VirtualEndpoint`.
- `UdpSocket`: `AF_INET6` (`IPV6_V6ONLY = 0`) non-blocking socket with pre-allocated 64-datagram
  `recvmmsg(2)` / `sendmmsg(2)` kernel rings (`0` per-packet heap allocations) and ephemeral port
  discovery via `getsockname(2)`.
"""
from std.collections import List
from std.ffi import external_call
from flint.platform.os import EINTR, close_fd, last_errno

comptime UDP_RING_CAPACITY: Int = 64
comptime UDP_MAX_DATAGRAM_BYTES: Int = 1500

comptime _AF_INET: Int32 = 2
comptime _AF_INET6: Int32 = 10
comptime _SOCK_DGRAM_NONBLOCK_CLOEXEC: Int32 = 2 | 2048 | 524288
comptime _SOL_SOCKET: Int32 = 1
comptime _SO_REUSEADDR: Int32 = 2
comptime _SO_SNDBUF: Int32 = 7
comptime _SO_RCVBUF: Int32 = 8
comptime _IPPROTO_IPV6: Int32 = 41
comptime _IPV6_V6ONLY: Int32 = 26
comptime _MSG_DONTWAIT: Int32 = 0x40
comptime _MSG_NOSIGNAL: Int32 = 0x4000
comptime _EAGAIN: Int32 = 11

comptime _SOCKADDR_IN6_LEN: Int = 28
comptime _SOCKADDR_STRIDE: Int = 32
comptime _IOVEC_STRIDE: Int = 16
comptime _MMSGHDR_STRIDE: Int = 64


@always_inline
def _store_u16_le(base: Int, off: Int, val: UInt16):
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base + off)
    p[unsafe_offset=0] = UInt8(val & 0xFF)
    p[unsafe_offset=1] = UInt8((val >> 8) & 0xFF)


@always_inline
def _load_u16_le(base: Int, off: Int) -> UInt16:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base + off)
    return UInt16(p[unsafe_offset=0]) | (UInt16(p[unsafe_offset=1]) << 8)


@always_inline
def _store_u32_le(base: Int, off: Int, val: UInt32):
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=base + off).unsafe_store[
        alignment=1
    ](val)


@always_inline
def _load_u32_le(base: Int, off: Int) -> UInt32:
    return Pointer[UInt32, ImmutAnyOrigin](
        unsafe_from_address=base + off
    ).unsafe_load[alignment=1]()


@always_inline
def _store_u64_le(base: Int, off: Int, val: UInt64):
    Pointer[UInt64, MutAnyOrigin](unsafe_from_address=base + off).unsafe_store[
        alignment=1
    ](val)


@fieldwise_init
struct NetEndpoint(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """Dual-stack 128-bit IPv6 / IPv4-mapped (`::ffff:a.b.c.d`) network endpoint."""
    var addr_hi: UInt64
    var addr_lo: UInt64
    var port: UInt16
    var scope_id: UInt32

    def __init__(out self):
        self.addr_hi = 0
        self.addr_lo = 0
        self.port = 0
        self.scope_id = 0

    def __eq__(self, other: Self) -> Bool:
        return (
            self.addr_hi == other.addr_hi
            and self.addr_lo == other.addr_lo
            and self.port == other.port
            and self.scope_id == other.scope_id
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @staticmethod
    @always_inline
    def any_dual_stack(port: Int = 0) -> Self:
        """Wildcard `[::]:port` accepting both IPv6 and IPv4-mapped traffic."""
        return Self(0, 0, UInt16(port & 0xFFFF), 0)

    @staticmethod
    @always_inline
    def ipv6_loopback(port: Int) -> Self:
        """IPv6 loopback `[::1]:port`."""
        return Self(0, 1, UInt16(port & 0xFFFF), 0)

    @staticmethod
    @always_inline
    def ipv4(a: Int, b: Int, c: Int, d: Int, port: Int) -> Self:
        """IPv4-mapped IPv6 endpoint `[::ffff:a.b.c.d]:port`."""
        var v4 = (
            (UInt64(a & 0xFF) << 24)
            | (UInt64(b & 0xFF) << 16)
            | (UInt64(c & 0xFF) << 8)
            | UInt64(d & 0xFF)
        )
        return Self(
            0, (UInt64(0xFFFF) << 32) | v4, UInt16(port & 0xFFFF), 0
        )

    @staticmethod
    @always_inline
    def ipv4_loopback(port: Int) -> Self:
        """IPv4-mapped loopback `127.0.0.1:port` (`[::ffff:127.0.0.1]:port`)."""
        return Self.ipv4(127, 0, 0, 1, port)

    @always_inline
    def is_ipv4_mapped(self) -> Bool:
        return self.addr_hi == 0 and (self.addr_lo >> 32) == 0xFFFF

    @always_inline
    def is_ipv6_loopback(self) -> Bool:
        return self.addr_hi == 0 and self.addr_lo == 1

    @always_inline
    def ipv4_octets(self) -> Tuple[Int, Int, Int, Int]:
        var v = self.addr_lo & 0xFFFFFFFF
        return (
            Int((v >> 24) & 0xFF),
            Int((v >> 16) & 0xFF),
            Int((v >> 8) & 0xFF),
            Int(v & 0xFF),
        )

    @always_inline
    def byte_at(self, idx: Int) -> UInt8:
        if idx < 8:
            var shift = UInt64((7 - idx) * 8)
            return UInt8((self.addr_hi >> shift) & 0xFF)
        var shift = UInt64((15 - idx) * 8)
        return UInt8((self.addr_lo >> shift) & 0xFF)

    def write_sockaddr_in6(self, addr_ptr: Int):
        """Writes a 28-byte Linux `sockaddr_in6` into `addr_ptr` (zero-padded to 32 bytes)."""
        for z in range(4):
            _store_u64_le(addr_ptr, z * 8, 0)
        _store_u16_le(addr_ptr, 0, UInt16(_AF_INET6))
        var p_out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=addr_ptr)
        p_out[unsafe_offset=2] = UInt8((self.port >> 8) & 0xFF)
        p_out[unsafe_offset=3] = UInt8(self.port & 0xFF)
        for i in range(16):
            p_out[unsafe_offset=8 + i] = self.byte_at(i)
        _store_u32_le(addr_ptr, 24, self.scope_id)

    @staticmethod
    def from_sockaddr(addr_ptr: Int) -> Self:
        """Decodes a Linux `sockaddr_in6` (or `sockaddr_in`) from `addr_ptr`."""
        var family = Int(_load_u16_le(addr_ptr, 0))
        var p_in = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr_ptr)
        var port = (UInt16(p_in[unsafe_offset=2]) << 8) | UInt16(
            p_in[unsafe_offset=3]
        )
        if family == Int(_AF_INET):
            return Self.ipv4(
                Int(p_in[unsafe_offset=4]),
                Int(p_in[unsafe_offset=5]),
                Int(p_in[unsafe_offset=6]),
                Int(p_in[unsafe_offset=7]),
                Int(port),
            )
        var hi = UInt64(0)
        var lo = UInt64(0)
        for i in range(8):
            hi = (hi << 8) | UInt64(p_in[unsafe_offset=8 + i])
        for i in range(8):
            lo = (lo << 8) | UInt64(p_in[unsafe_offset=16 + i])
        var scope = _load_u32_le(addr_ptr, 24)
        return Self(hi, lo, port, scope)

    def to_string(self) -> String:
        if self.is_ipv4_mapped():
            var octs = self.ipv4_octets()
            return (
                String(octs[0])
                + "."
                + String(octs[1])
                + "."
                + String(octs[2])
                + "."
                + String(octs[3])
                + ":"
                + String(Int(self.port))
            )
        if self.is_ipv6_loopback():
            return "[::1]:" + String(Int(self.port))
        if self.addr_hi == 0 and self.addr_lo == 0:
            return "[::]:" + String(Int(self.port))
        return (
            "[ipv6:"
            + String(self.addr_hi)
            + ":"
            + String(self.addr_lo)
            + "]:"
            + String(Int(self.port))
        )


@fieldwise_init
struct DatagramView(Copyable, ImplicitlyCopyable, Movable):
    """Zero-copy borrowed datagram slice inside a transport's pre-allocated RX ring."""
    var sender: NetEndpoint
    var data_ptr: Int
    var length: Int

    def __init__(out self):
        self.sender = NetEndpoint()
        self.data_ptr = 0
        self.length = 0

    @always_inline
    def byte_at(self, idx: Int) -> UInt8:
        return Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=self.data_ptr
        )[unsafe_offset=idx]

    def to_bytes(self) -> List[UInt8]:
        if self.data_ptr == 0 or self.length <= 0:
            return List[UInt8]()
        var out = List[UInt8](capacity=self.length)
        out.resize(unsafe_uninit_length=self.length)
        _ = external_call["memcpy", Int](
            Int(out.unsafe_ptr()), self.data_ptr, Int64(self.length)
        )
        return out^


trait NetTransport(Movable):
    """Polymorphic non-blocking datagram transport (`UdpSocket` or `VirtualEndpoint`)."""
    def local_endpoint(self) -> NetEndpoint: ...
    def send_raw(mut self, dst: NetEndpoint, data_ptr: Int, length: Int) -> Bool: ...
    def queue_send_raw(mut self, dst: NetEndpoint, data_ptr: Int, length: Int) -> Bool: ...
    def flush_sends(mut self) -> Int: ...
    def recv_batch(mut self, mut out_views: List[DatagramView]) -> Int: ...


struct UdpSocket(NetTransport):
    """Non-blocking dual-stack Linux `AF_INET6` (`IPV6_V6ONLY = 0`) UDP socket with
    pre-allocated 64-datagram `recvmmsg(2)` / `sendmmsg(2)` rings and zero per-packet heap allocs.
    """
    var fd: Int
    var bound_endpoint: NetEndpoint
    var rx_payloads: List[UInt8]
    var rx_addrs: List[UInt8]
    var rx_iovecs: List[UInt8]
    var rx_mmsgs: List[UInt8]
    var tx_payloads: List[UInt8]
    var tx_addrs: List[UInt8]
    var tx_iovecs: List[UInt8]
    var tx_mmsgs: List[UInt8]
    var tx_queued: Int
    var total_sent_packets: Int
    var total_recv_packets: Int
    var sendmmsg_calls: Int
    var recvmmsg_calls: Int

    def __init__(out self):
        self.fd = -1
        self.bound_endpoint = NetEndpoint()
        self.rx_payloads = List[UInt8](
            length=UDP_RING_CAPACITY * UDP_MAX_DATAGRAM_BYTES, fill=0
        )
        self.rx_addrs = List[UInt8](
            length=UDP_RING_CAPACITY * _SOCKADDR_STRIDE, fill=0
        )
        self.rx_iovecs = List[UInt8](
            length=UDP_RING_CAPACITY * _IOVEC_STRIDE, fill=0
        )
        self.rx_mmsgs = List[UInt8](
            length=UDP_RING_CAPACITY * _MMSGHDR_STRIDE, fill=0
        )
        self.tx_payloads = List[UInt8](
            length=UDP_RING_CAPACITY * UDP_MAX_DATAGRAM_BYTES, fill=0
        )
        self.tx_addrs = List[UInt8](
            length=UDP_RING_CAPACITY * _SOCKADDR_STRIDE, fill=0
        )
        self.tx_iovecs = List[UInt8](
            length=UDP_RING_CAPACITY * _IOVEC_STRIDE, fill=0
        )
        self.tx_mmsgs = List[UInt8](
            length=UDP_RING_CAPACITY * _MMSGHDR_STRIDE, fill=0
        )
        self.tx_queued = 0
        self.total_sent_packets = 0
        self.total_recv_packets = 0
        self.sendmmsg_calls = 0
        self.recvmmsg_calls = 0

    def __deinit__(deinit self):
        if self.fd >= 0:
            close_fd(self.fd)

    def close(mut self):
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
        self.tx_queued = 0

    @always_inline
    def is_open(self) -> Bool:
        return self.fd >= 0

    @always_inline
    def local_endpoint(self) -> NetEndpoint:
        return self.bound_endpoint

    @always_inline
    def local_port(self) -> Int:
        return Int(self.bound_endpoint.port)

    def bind(mut self, endpoint: NetEndpoint = NetEndpoint.any_dual_stack(0)) -> Bool:
        """Opens a non-blocking `AF_INET6` dual-stack socket (`IPV6_V6ONLY = 0`) and binds it."""
        self.close()
        var sfd = Int(
            external_call["socket", Int32](
                _AF_INET6, _SOCK_DGRAM_NONBLOCK_CLOEXEC, Int32(0)
            )
        )
        if sfd < 0:
            return False

        # Enable dual-stack IPv4-mapped support (IPV6_V6ONLY = 0)
        var zero_opt = Int32(0)
        _ = external_call["setsockopt", Int32](
            Int32(sfd),
            _IPPROTO_IPV6,
            _IPV6_V6ONLY,
            Pointer(to=zero_opt),
            UInt32(4),
        )

        # Enable SO_REUSEADDR when binding an explicit port (keep 0 ephemeral ports exclusive)
        if endpoint.port != 0:
            var one_opt = Int32(1)
            _ = external_call["setsockopt", Int32](
                Int32(sfd),
                _SOL_SOCKET,
                _SO_REUSEADDR,
                Pointer(to=one_opt),
                UInt32(4),
            )

        # Request 1 MiB kernel socket buffers for burst absorbtion
        var buf_bytes = Int32(1048576)
        _ = external_call["setsockopt", Int32](
            Int32(sfd), _SOL_SOCKET, _SO_RCVBUF, Pointer(to=buf_bytes), UInt32(4)
        )
        _ = external_call["setsockopt", Int32](
            Int32(sfd), _SOL_SOCKET, _SO_SNDBUF, Pointer(to=buf_bytes), UInt32(4)
        )

        var addr_base = Int(self.tx_addrs.unsafe_ptr())
        endpoint.write_sockaddr_in6(addr_base)
        var brc = external_call["bind", Int32](
            Int32(sfd),
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr_base),
            UInt32(_SOCKADDR_IN6_LEN),
        )
        if brc != 0:
            close_fd(sfd)
            return False

        # Query bound address & ephemeral port via getsockname(2)
        var name_base = Int(self.rx_addrs.unsafe_ptr())
        for z in range(4):
            _store_u64_le(name_base, z * 8, 0)
        var namelen = UInt32(_SOCKADDR_IN6_LEN)
        var grc = external_call["getsockname", Int32](
            Int32(sfd),
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=name_base),
            Pointer(to=namelen),
        )
        self.fd = sfd
        if grc == 0:
            self.bound_endpoint = NetEndpoint.from_sockaddr(name_base)
        else:
            self.bound_endpoint = endpoint
        return True

    @staticmethod
    def open_dual_stack(port: Int = 0) -> Self:
        var sock = Self()
        _ = sock.bind(NetEndpoint.any_dual_stack(port))
        return sock^

    def queue_send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if (
            self.fd < 0
            or data_ptr == 0
            or length <= 0
            or length > UDP_MAX_DATAGRAM_BYTES
        ):
            return False
        if self.tx_queued >= UDP_RING_CAPACITY:
            _ = self.flush_sends()
            if self.tx_queued >= UDP_RING_CAPACITY:
                return False

        var idx = self.tx_queued
        var payload_slot = (
            Int(self.tx_payloads.unsafe_ptr()) + idx * UDP_MAX_DATAGRAM_BYTES
        )
        var addr_slot = Int(self.tx_addrs.unsafe_ptr()) + idx * _SOCKADDR_STRIDE
        var iov_slot = Int(self.tx_iovecs.unsafe_ptr()) + idx * _IOVEC_STRIDE
        var mmsg_slot = Int(self.tx_mmsgs.unsafe_ptr()) + idx * _MMSGHDR_STRIDE

        _ = external_call["memcpy", Int](
            payload_slot, data_ptr, Int64(length)
        )
        dst.write_sockaddr_in6(addr_slot)

        _store_u64_le(iov_slot, 0, UInt64(payload_slot))
        _store_u64_le(iov_slot, 8, UInt64(length))

        for z in range(8):
            _store_u64_le(mmsg_slot, z * 8, 0)
        _store_u64_le(mmsg_slot, 0, UInt64(addr_slot))
        _store_u32_le(mmsg_slot, 8, UInt32(_SOCKADDR_IN6_LEN))
        _store_u64_le(mmsg_slot, 16, UInt64(iov_slot))
        _store_u64_le(mmsg_slot, 24, UInt64(1))

        self.tx_queued += 1
        return True

    def queue_send_bytes(
        mut self, dst: NetEndpoint, payload: List[UInt8]
    ) -> Bool:
        if len(payload) == 0:
            return False
        return self.queue_send_raw(
            dst, Int(payload.unsafe_ptr()), len(payload)
        )

    def flush_sends(mut self) -> Int:
        """Flushes all queued TX datagrams in a single `sendmmsg(2)` kernel call."""
        if self.fd < 0 or self.tx_queued <= 0:
            self.tx_queued = 0
            return 0

        var count = self.tx_queued
        # Ensure ring addresses reflect current buffer pointers
        var base_payload = Int(self.tx_payloads.unsafe_ptr())
        var base_addr = Int(self.tx_addrs.unsafe_ptr())
        var base_iov = Int(self.tx_iovecs.unsafe_ptr())
        var base_mmsg = Int(self.tx_mmsgs.unsafe_ptr())
        for i in range(count):
            var p_slot = base_payload + i * UDP_MAX_DATAGRAM_BYTES
            var a_slot = base_addr + i * _SOCKADDR_STRIDE
            var v_slot = base_iov + i * _IOVEC_STRIDE
            var m_slot = base_mmsg + i * _MMSGHDR_STRIDE
            _store_u64_le(v_slot, 0, UInt64(p_slot))
            _store_u64_le(m_slot, 0, UInt64(a_slot))
            _store_u64_le(m_slot, 16, UInt64(v_slot))

        var sent_total = 0
        while sent_total < count:
            var mmsg_ptr = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=base_mmsg + sent_total * _MMSGHDR_STRIDE
            )
            var rem = count - sent_total
            var rc = Int(
                external_call["sendmmsg", Int32](
                    Int32(self.fd),
                    mmsg_ptr,
                    UInt32(rem),
                    _MSG_NOSIGNAL,
                )
            )
            self.sendmmsg_calls += 1
            if rc < 0:
                if last_errno() == EINTR:
                    continue
                break
            if rc == 0:
                break
            sent_total += rc

        self.tx_queued = 0
        self.total_sent_packets += sent_total
        return sent_total

    def send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if not self.queue_send_raw(dst, data_ptr, length):
            return False
        return self.flush_sends() == 1

    def send_to(mut self, dst: NetEndpoint, payload: List[UInt8]) -> Bool:
        if len(payload) == 0:
            return False
        return self.send_raw(dst, Int(payload.unsafe_ptr()), len(payload))

    def recv_batch(mut self, mut out_views: List[DatagramView]) -> Int:
        """Receives up to 64 UDP datagrams in one `recvmmsg(2)` call with `0` heap allocations."""
        out_views.clear()
        if self.fd < 0:
            return 0
        if out_views.capacity() < UDP_RING_CAPACITY:
            out_views.reserve(UDP_RING_CAPACITY)

        var base_payload = Int(self.rx_payloads.unsafe_ptr())
        var base_addr = Int(self.rx_addrs.unsafe_ptr())
        var base_iov = Int(self.rx_iovecs.unsafe_ptr())
        var base_mmsg = Int(self.rx_mmsgs.unsafe_ptr())

        for i in range(UDP_RING_CAPACITY):
            var p_slot = base_payload + i * UDP_MAX_DATAGRAM_BYTES
            var a_slot = base_addr + i * _SOCKADDR_STRIDE
            var v_slot = base_iov + i * _IOVEC_STRIDE
            var m_slot = base_mmsg + i * _MMSGHDR_STRIDE

            for z in range(4):
                _store_u64_le(a_slot, z * 8, 0)
            _store_u64_le(v_slot, 0, UInt64(p_slot))
            _store_u64_le(v_slot, 8, UInt64(UDP_MAX_DATAGRAM_BYTES))

            for z in range(8):
                _store_u64_le(m_slot, z * 8, 0)
            _store_u64_le(m_slot, 0, UInt64(a_slot))
            _store_u32_le(m_slot, 8, UInt32(_SOCKADDR_IN6_LEN))
            _store_u64_le(m_slot, 16, UInt64(v_slot))
            _store_u64_le(m_slot, 24, UInt64(1))

        var got = 0
        while True:
            var rc = Int(
                external_call["recvmmsg", Int32](
                    Int32(self.fd),
                    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base_mmsg),
                    UInt32(UDP_RING_CAPACITY),
                    _MSG_DONTWAIT,
                    Int(0),
                )
            )
            self.recvmmsg_calls += 1
            if rc < 0:
                var err = last_errno()
                if err == EINTR:
                    continue
                if err == _EAGAIN:
                    return 0
                return 0
            got = rc
            break

        for i in range(got):
            var m_slot = base_mmsg + i * _MMSGHDR_STRIDE
            var a_slot = base_addr + i * _SOCKADDR_STRIDE
            var p_slot = base_payload + i * UDP_MAX_DATAGRAM_BYTES
            var msg_len = Int(_load_u32_le(m_slot, 56))
            if msg_len > 0 and msg_len <= UDP_MAX_DATAGRAM_BYTES:
                var sender = NetEndpoint.from_sockaddr(a_slot)
                out_views.append(DatagramView(sender, p_slot, msg_len))

        self.total_recv_packets += len(out_views)
        return len(out_views)
