"""Dual-stack IPv6 + IPv4-mapped non-blocking multi-OS UDP transport (`flint.net.socket_udp`).

Implements:
- `NetEndpoint`: 128-bit dual-stack IPv6 + IPv4-mapped (`::ffff:a.b.c.d`) socket address
  with Linux, Win32 (`ws2_32.dll`), and macOS (Darwin `sin6_len`) wire encoders/decoders.
- `DatagramView`: zero-copy view into pre-allocated kernel RX rings with full sender `NetEndpoint`.
- `trait NetTransport`: polymorphic datagram interface shared by `UdpSocket` and `VirtualEndpoint`.
- `UdpSocket`: non-`Copyable` RAII `AF_INET6` (`IPV6_V6ONLY = 0`) non-blocking socket with
  pre-allocated 64-datagram TX/RX rings (`recvmmsg`/`sendmmsg` on Linux, `sendto`/`recvfrom`
  loops via `ws2_32.dll` on Windows and Darwin BSD sockets on macOS) and ephemeral port
  discovery via `getsockname`.
"""
from std.collections import List
from std.ffi import external_call
from std.hash import Hasher
from std.sys import inlined_assembly
from std.sys.info import CompilationTarget
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.os import EINTR, close_fd, last_errno

comptime IS_WINDOWS: Bool = (not CompilationTarget.is_linux()) and (not CompilationTarget.is_macos())

comptime SOCK_OS_LINUX: Int = 0
comptime SOCK_OS_WINDOWS: Int = 1
comptime SOCK_OS_MACOS: Int = 2
comptime CURRENT_SOCK_OS: Int = (
    SOCK_OS_WINDOWS if IS_WINDOWS else (SOCK_OS_MACOS if CompilationTarget.is_macos() else SOCK_OS_LINUX)
)

comptime UDP_RING_CAPACITY: Int = 64
comptime UDP_MAX_DATAGRAM_BYTES: Int = 1500

# Explicit per-OS socket constants (exposed for cross-OS protocol tests on Linux)
comptime LINUX_AF_INET6: Int32 = 10
comptime LINUX_SOL_SOCKET: Int32 = 1
comptime LINUX_SO_REUSEADDR: Int32 = 2
comptime LINUX_SO_SNDBUF: Int32 = 7
comptime LINUX_SO_RCVBUF: Int32 = 8
comptime LINUX_IPV6_V6ONLY: Int32 = 26
comptime LINUX_EAGAIN: Int32 = 11
comptime LINUX_ENOBUFS: Int32 = 105

comptime WIN32_AF_INET6: Int32 = 23
comptime WIN32_SOL_SOCKET: Int32 = 0xFFFF
comptime WIN32_SO_REUSEADDR: Int32 = 0x0004
comptime WIN32_SO_SNDBUF: Int32 = 0x1001
comptime WIN32_SO_RCVBUF: Int32 = 0x1002
comptime WIN32_IPV6_V6ONLY: Int32 = 27
comptime WIN32_IPPROTO_UDP: Int32 = 17
comptime WIN32_FIONBIO: UInt32 = 0x8004667E
comptime WIN32_WSADATA_BYTES: Int = 408
comptime WIN32_WSAEWOULDBLOCK: Int32 = 10035
comptime WIN32_WSAENOBUFS: Int32 = 10055
comptime WIN32_WSAEINTR: Int32 = 10004

comptime DARWIN_AF_INET6: Int32 = 30
comptime DARWIN_SOL_SOCKET: Int32 = 0xFFFF
comptime DARWIN_SO_REUSEADDR: Int32 = 0x0004
comptime DARWIN_SO_SNDBUF: Int32 = 0x1001
comptime DARWIN_SO_RCVBUF: Int32 = 0x1002
comptime DARWIN_SO_NOSIGPIPE: Int32 = 0x1022
comptime DARWIN_IPV6_V6ONLY: Int32 = 27
comptime DARWIN_FIONBIO: UInt64 = 0x8004667E
comptime DARWIN_F_SETFD: Int32 = 2
comptime DARWIN_FD_CLOEXEC: Int32 = 1
comptime DARWIN_F_GETFL: Int32 = 3
comptime DARWIN_F_SETFL: Int32 = 4
comptime DARWIN_O_NONBLOCK: Int32 = 0x0004
comptime DARWIN_EAGAIN: Int32 = 35
comptime DARWIN_ENOBUFS: Int32 = 55
comptime DARWIN_SOCKADDR_IN6_LEN: UInt8 = 28

comptime _AF_INET: Int32 = 2
comptime _AF_INET6: Int32 = (
    WIN32_AF_INET6 if IS_WINDOWS else (DARWIN_AF_INET6 if CompilationTarget.is_macos() else LINUX_AF_INET6)
)
comptime _SOCK_DGRAM: Int32 = 2
comptime _SOCK_DGRAM_NONBLOCK_CLOEXEC: Int32 = 2 | 2048 | 524288
comptime _SOL_SOCKET: Int32 = (
    WIN32_SOL_SOCKET if IS_WINDOWS else (DARWIN_SOL_SOCKET if CompilationTarget.is_macos() else LINUX_SOL_SOCKET)
)
comptime _SO_REUSEADDR: Int32 = (
    WIN32_SO_REUSEADDR if IS_WINDOWS else (DARWIN_SO_REUSEADDR if CompilationTarget.is_macos() else LINUX_SO_REUSEADDR)
)
comptime _SO_SNDBUF: Int32 = (
    WIN32_SO_SNDBUF if IS_WINDOWS else (DARWIN_SO_SNDBUF if CompilationTarget.is_macos() else LINUX_SO_SNDBUF)
)
comptime _SO_RCVBUF: Int32 = (
    WIN32_SO_RCVBUF if IS_WINDOWS else (DARWIN_SO_RCVBUF if CompilationTarget.is_macos() else LINUX_SO_RCVBUF)
)
comptime _IPPROTO_IPV6: Int32 = 41
comptime _IPV6_V6ONLY: Int32 = (
    WIN32_IPV6_V6ONLY if IS_WINDOWS else (DARWIN_IPV6_V6ONLY if CompilationTarget.is_macos() else LINUX_IPV6_V6ONLY)
)
comptime _MSG_DONTWAIT: Int32 = 0x80 if CompilationTarget.is_macos() else 0x40
comptime _MSG_NOSIGNAL: Int32 = 0 if (IS_WINDOWS or CompilationTarget.is_macos()) else 0x4000
comptime _EAGAIN: Int32 = (
    WIN32_WSAEWOULDBLOCK if IS_WINDOWS else (DARWIN_EAGAIN if CompilationTarget.is_macos() else LINUX_EAGAIN)
)
comptime _ENOBUFS: Int32 = (
    WIN32_WSAENOBUFS if IS_WINDOWS else (DARWIN_ENOBUFS if CompilationTarget.is_macos() else LINUX_ENOBUFS)
)

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


def _wsa_load_dll_utf16() -> Int:
    comptime if IS_WINDOWS:
        # "ws2_32.dll\0" in UTF-16LE
        var wide = List[UInt16](capacity=11)
        wide.append(UInt16(119))  # w
        wide.append(UInt16(115))  # s
        wide.append(UInt16(50))   # 2
        wide.append(UInt16(95))   # _
        wide.append(UInt16(51))   # 3
        wide.append(UInt16(50))   # 2
        wide.append(UInt16(46))   # .
        wide.append(UInt16(100))  # d
        wide.append(UInt16(108))  # l
        wide.append(UInt16(108))  # l
        wide.append(UInt16(0))
        var h = external_call["LoadLibraryW", Int](wide.unsafe_ptr())
        _ = len(wide)
        return h
    else:
        return 0


def _wsa_proc(lib: Int, name: String) -> Int:
    comptime if IS_WINDOWS:
        if lib == 0:
            return 0
        var owned = name
        var p = external_call["GetProcAddress", Int](lib, owned.as_c_string_span().ptr())
        _ = owned.byte_length()
        return p
    else:
        return 0


def _darwin_errno() -> Int32:
    comptime if CompilationTarget.is_macos():
        return external_call["__error", Pointer[Int32, origin=MutAnyOrigin]]()[unsafe_offset=0]
    else:
        return last_errno()


def _darwin_arm64_fcntl(fd: Int32, cmd: Int32, arg: Int) -> Int32:
    """Calls variadic Darwin `fcntl(fd, cmd, ...)` with 3rd arg on stack per arm64-apple-darwin ABI."""
    comptime if CompilationTarget.is_macos() and not CompilationTarget.is_x86():
        var lib = dl_open("/usr/lib/libSystem.B.dylib")
        if lib == 0:
            return -1
        var fn_addr = dl_sym(lib, "fcntl")
        if fn_addr == 0:
            _ = dl_close(lib)
            return -1
        var res = inlined_assembly[
            "mov x0, $1\nmov x1, $2\nsub sp, sp, #16\nstr $3, [sp]\nblr $4\nadd sp, sp, #16\nmov $0, x0",
            Int64,
            constraints="=&r,r,r,r,r,~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{lr},~{memory},~{cc}",
            has_side_effect=True,
        ](Int64(fd), Int64(cmd), Int64(arg), fn_addr)
        _ = dl_close(lib)
        return Int32(res)
    else:
        _ = fd
        _ = cmd
        _ = arg
        return -1


def _darwin_arm64_ioctl(fd: Int32, request: UInt64, arg_ptr: Int) -> Int32:
    """Calls variadic Darwin `ioctl(fd, request, ...)` with 3rd arg on stack per arm64-apple-darwin ABI."""
    comptime if CompilationTarget.is_macos() and not CompilationTarget.is_x86():
        var lib = dl_open("/usr/lib/libSystem.B.dylib")
        if lib == 0:
            return -1
        var fn_addr = dl_sym(lib, "ioctl")
        if fn_addr == 0:
            _ = dl_close(lib)
            return -1
        var res = inlined_assembly[
            "mov x0, $1\nmov x1, $2\nsub sp, sp, #16\nstr $3, [sp]\nblr $4\nadd sp, sp, #16\nmov $0, x0",
            Int64,
            constraints="=&r,r,r,r,r,~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{lr},~{memory},~{cc}",
            has_side_effect=True,
        ](Int64(fd), Int64(request), Int64(arg_ptr), fn_addr)
        _ = dl_close(lib)
        return Int32(res)
    else:
        _ = fd
        _ = request
        _ = arg_ptr
        return -1


@fieldwise_init
struct NetEndpoint(Copyable, ImplicitlyCopyable, Movable, Equatable, Hashable):
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

    def __hash__[H: Hasher](self, mut hasher: H):
        self.addr_hi.__hash__(hasher)
        self.addr_lo.__hash__(hasher)
        self.port.__hash__(hasher)
        self.scope_id.__hash__(hasher)

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

    def write_sockaddr_in6_for_os(self, addr_ptr: Int, os_id: Int):
        """Writes a 28-byte `sockaddr_in6` for `os_id` (`SOCK_OS_*`) into `addr_ptr` (zero-padded to 32 bytes)."""
        for z in range(4):
            _store_u64_le(addr_ptr, z * 8, 0)
        var p_out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=addr_ptr)
        if os_id == SOCK_OS_MACOS:
            p_out[unsafe_offset=0] = DARWIN_SOCKADDR_IN6_LEN
            p_out[unsafe_offset=1] = UInt8(DARWIN_AF_INET6)
        elif os_id == SOCK_OS_WINDOWS:
            _store_u16_le(addr_ptr, 0, UInt16(WIN32_AF_INET6))
        else:
            _store_u16_le(addr_ptr, 0, UInt16(LINUX_AF_INET6))
        p_out[unsafe_offset=2] = UInt8((self.port >> 8) & 0xFF)
        p_out[unsafe_offset=3] = UInt8(self.port & 0xFF)
        for i in range(16):
            p_out[unsafe_offset=8 + i] = self.byte_at(i)
        _store_u32_le(addr_ptr, 24, self.scope_id)

    def write_sockaddr_in6(self, addr_ptr: Int):
        """Writes a 28-byte host-OS `sockaddr_in6` into `addr_ptr` (zero-padded to 32 bytes)."""
        self.write_sockaddr_in6_for_os(addr_ptr, CURRENT_SOCK_OS)

    @staticmethod
    def from_sockaddr_for_os(addr_ptr: Int, os_id: Int) -> Self:
        """Decodes a `sockaddr_in6` (or `sockaddr_in`) for `os_id` (`SOCK_OS_*`) from `addr_ptr`."""
        var p_in = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr_ptr)
        var family = (
            Int(p_in[unsafe_offset=1]) if os_id == SOCK_OS_MACOS else Int(_load_u16_le(addr_ptr, 0))
        )
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

    @staticmethod
    def from_sockaddr(addr_ptr: Int) -> Self:
        """Decodes a host-OS `sockaddr_in6` (or `sockaddr_in`) from `addr_ptr`."""
        return Self.from_sockaddr_for_os(addr_ptr, CURRENT_SOCK_OS)

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
    """Non-blocking dual-stack multi-OS `AF_INET6` (`IPV6_V6ONLY = 0`) UDP socket with
    pre-allocated 64-datagram TX/RX rings (`0` per-packet heap allocs).
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
    var wsa_lib: Int
    var wsa_active: Bool

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
        self.wsa_lib = 0
        self.wsa_active = False

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.bound_endpoint = existing.bound_endpoint
        self.rx_payloads = existing.rx_payloads^
        self.rx_addrs = existing.rx_addrs^
        self.rx_iovecs = existing.rx_iovecs^
        self.rx_mmsgs = existing.rx_mmsgs^
        self.tx_payloads = existing.tx_payloads^
        self.tx_addrs = existing.tx_addrs^
        self.tx_iovecs = existing.tx_iovecs^
        self.tx_mmsgs = existing.tx_mmsgs^
        self.tx_queued = existing.tx_queued
        self.total_sent_packets = existing.total_sent_packets
        self.total_recv_packets = existing.total_recv_packets
        self.sendmmsg_calls = existing.sendmmsg_calls
        self.recvmmsg_calls = existing.recvmmsg_calls
        self.wsa_lib = existing.wsa_lib
        self.wsa_active = existing.wsa_active
        existing.fd = -1
        existing.tx_queued = 0
        existing.wsa_lib = 0
        existing.wsa_active = False

    def __deinit__(deinit self):
        self.close()

    def close(mut self):
        comptime if IS_WINDOWS:
            if self.fd != -1 and self.wsa_lib != 0:
                var fn_close = _wsa_proc(self.wsa_lib, "closesocket")
                if fn_close != 0:
                    comptime CloseSockFn = def(Int) thin abi("C") -> Int32
                    _ = Pointer(to=fn_close).unsafe_bitcast[CloseSockFn]()[unsafe_offset=0](self.fd)
            if self.wsa_active and self.wsa_lib != 0:
                var fn_clean = _wsa_proc(self.wsa_lib, "WSACleanup")
                if fn_clean != 0:
                    comptime WsaCleanFn = def() thin abi("C") -> Int32
                    _ = Pointer(to=fn_clean).unsafe_bitcast[WsaCleanFn]()[unsafe_offset=0]()
            if self.wsa_lib != 0:
                _ = external_call["FreeLibrary", Int32](self.wsa_lib)
            self.wsa_lib = 0
            self.wsa_active = False
        else:
            if self.fd >= 0:
                close_fd(self.fd)
        self.fd = -1
        self.tx_queued = 0

    @always_inline
    def is_open(self) -> Bool:
        comptime if IS_WINDOWS:
            return self.fd != -1
        else:
            return self.fd >= 0

    @always_inline
    def is_valid(self) -> Bool:
        return self.is_open()

    @always_inline
    def local_endpoint(self) -> NetEndpoint:
        return self.bound_endpoint

    @always_inline
    def local_port(self) -> Int:
        return Int(self.bound_endpoint.port)

    def bind(mut self, endpoint: NetEndpoint = NetEndpoint.any_dual_stack(0)) -> Bool:
        """Opens a non-blocking `AF_INET6` dual-stack socket (`IPV6_V6ONLY = 0`) and binds it."""
        self.close()
        comptime if IS_WINDOWS:
            self.wsa_lib = _wsa_load_dll_utf16()
            if self.wsa_lib == 0:
                return False
            var fn_startup = _wsa_proc(self.wsa_lib, "WSAStartup")
            var fn_socket = _wsa_proc(self.wsa_lib, "socket")
            var fn_setsockopt = _wsa_proc(self.wsa_lib, "setsockopt")
            var fn_ioctlsocket = _wsa_proc(self.wsa_lib, "ioctlsocket")
            var fn_bind = _wsa_proc(self.wsa_lib, "bind")
            var fn_getsockname = _wsa_proc(self.wsa_lib, "getsockname")
            if (
                fn_startup == 0
                or fn_socket == 0
                or fn_setsockopt == 0
                or fn_ioctlsocket == 0
                or fn_bind == 0
                or fn_getsockname == 0
            ):
                self.close()
                return False

            var wsa_data = List[UInt8](length=WIN32_WSADATA_BYTES, fill=0)
            comptime WsaStartFn = def(UInt16, Int) thin abi("C") -> Int32
            if Pointer(to=fn_startup).unsafe_bitcast[WsaStartFn]()[unsafe_offset=0](
                UInt16(0x0202), Int(wsa_data.unsafe_ptr())
            ) != 0:
                self.close()
                return False
            _ = len(wsa_data)
            self.wsa_active = True

            comptime WinSockFn = def(Int32, Int32, Int32) thin abi("C") -> Int
            var sfd = Pointer(to=fn_socket).unsafe_bitcast[WinSockFn]()[unsafe_offset=0](
                WIN32_AF_INET6, _SOCK_DGRAM, WIN32_IPPROTO_UDP
            )
            if sfd == -1:
                self.close()
                return False
            self.fd = sfd

            comptime WinSetSockOptFn = def(Int, Int32, Int32, Int, Int32) thin abi("C") -> Int32
            var zero_opt = Int32(0)
            _ = Pointer(to=fn_setsockopt).unsafe_bitcast[WinSetSockOptFn]()[unsafe_offset=0](
                sfd, _IPPROTO_IPV6, WIN32_IPV6_V6ONLY, Int(Pointer(to=zero_opt)), 4
            )
            if endpoint.port != 0:
                var one_opt = Int32(1)
                _ = Pointer(to=fn_setsockopt).unsafe_bitcast[WinSetSockOptFn]()[unsafe_offset=0](
                    sfd, WIN32_SOL_SOCKET, WIN32_SO_REUSEADDR, Int(Pointer(to=one_opt)), 4
                )
            var buf_bytes = Int32(1048576)
            _ = Pointer(to=fn_setsockopt).unsafe_bitcast[WinSetSockOptFn]()[unsafe_offset=0](
                sfd, WIN32_SOL_SOCKET, WIN32_SO_RCVBUF, Int(Pointer(to=buf_bytes)), 4
            )
            _ = Pointer(to=fn_setsockopt).unsafe_bitcast[WinSetSockOptFn]()[unsafe_offset=0](
                sfd, WIN32_SOL_SOCKET, WIN32_SO_SNDBUF, Int(Pointer(to=buf_bytes)), 4
            )

            var nonblock = UInt32(1)
            comptime WinIoctlFn = def(Int, UInt32, Int) thin abi("C") -> Int32
            if Pointer(to=fn_ioctlsocket).unsafe_bitcast[WinIoctlFn]()[unsafe_offset=0](
                sfd, WIN32_FIONBIO, Int(Pointer(to=nonblock))
            ) != 0:
                self.close()
                return False

            var addr_base = Int(self.tx_addrs.unsafe_ptr())
            endpoint.write_sockaddr_in6_for_os(addr_base, SOCK_OS_WINDOWS)
            comptime WinBindFn = def(Int, Int, Int32) thin abi("C") -> Int32
            if Pointer(to=fn_bind).unsafe_bitcast[WinBindFn]()[unsafe_offset=0](
                sfd, addr_base, Int32(_SOCKADDR_IN6_LEN)
            ) != 0:
                self.close()
                return False

            var name_base = Int(self.rx_addrs.unsafe_ptr())
            for z in range(4):
                _store_u64_le(name_base, z * 8, 0)
            var namelen = Int32(_SOCKADDR_IN6_LEN)
            comptime WinGetNameFn = def(Int, Int, Int) thin abi("C") -> Int32
            var grc = Pointer(to=fn_getsockname).unsafe_bitcast[WinGetNameFn]()[unsafe_offset=0](
                sfd, name_base, Int(Pointer(to=namelen))
            )
            if grc == 0:
                self.bound_endpoint = NetEndpoint.from_sockaddr_for_os(name_base, SOCK_OS_WINDOWS)
            else:
                self.bound_endpoint = endpoint
            return True

        elif CompilationTarget.is_macos():
            var sfd = Int(external_call["socket", Int32](DARWIN_AF_INET6, _SOCK_DGRAM, Int32(0)))
            if sfd < 0:
                return False
            comptime if CompilationTarget.is_x86():
                _ = external_call["fcntl", Int32](Int32(sfd), DARWIN_F_SETFD, DARWIN_FD_CLOEXEC)
                var flags = external_call["fcntl", Int32](Int32(sfd), DARWIN_F_GETFL, Int32(0))
                _ = external_call["fcntl", Int32](Int32(sfd), DARWIN_F_SETFL, flags | DARWIN_O_NONBLOCK)
                var nonblock_one = Int32(1)
                _ = external_call["ioctl", Int32](Int32(sfd), DARWIN_FIONBIO, Pointer(to=nonblock_one))
            else:
                _ = _darwin_arm64_fcntl(Int32(sfd), DARWIN_F_SETFD, Int(DARWIN_FD_CLOEXEC))
                var flags = _darwin_arm64_fcntl(Int32(sfd), DARWIN_F_GETFL, 0)
                _ = _darwin_arm64_fcntl(Int32(sfd), DARWIN_F_SETFL, Int(flags | DARWIN_O_NONBLOCK))
                var nonblock_one = Int32(1)
                _ = _darwin_arm64_ioctl(Int32(sfd), DARWIN_FIONBIO, Int(Pointer(to=nonblock_one)))
                _ = nonblock_one

            var one_opt = Int32(1)
            _ = external_call["setsockopt", Int32](
                Int32(sfd), DARWIN_SOL_SOCKET, DARWIN_SO_NOSIGPIPE, Pointer(to=one_opt), UInt32(4)
            )
            var zero_opt = Int32(0)
            _ = external_call["setsockopt", Int32](
                Int32(sfd), _IPPROTO_IPV6, DARWIN_IPV6_V6ONLY, Pointer(to=zero_opt), UInt32(4)
            )
            if endpoint.port != 0:
                _ = external_call["setsockopt", Int32](
                    Int32(sfd), DARWIN_SOL_SOCKET, DARWIN_SO_REUSEADDR, Pointer(to=one_opt), UInt32(4)
                )
            var buf_bytes = Int32(1048576)
            _ = external_call["setsockopt", Int32](
                Int32(sfd), DARWIN_SOL_SOCKET, DARWIN_SO_RCVBUF, Pointer(to=buf_bytes), UInt32(4)
            )
            _ = external_call["setsockopt", Int32](
                Int32(sfd), DARWIN_SOL_SOCKET, DARWIN_SO_SNDBUF, Pointer(to=buf_bytes), UInt32(4)
            )

            var addr_base = Int(self.tx_addrs.unsafe_ptr())
            endpoint.write_sockaddr_in6_for_os(addr_base, SOCK_OS_MACOS)
            var brc = external_call["bind", Int32](
                Int32(sfd),
                Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr_base),
                UInt32(_SOCKADDR_IN6_LEN),
            )
            if brc != 0:
                close_fd(sfd)
                return False

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
                self.bound_endpoint = NetEndpoint.from_sockaddr_for_os(name_base, SOCK_OS_MACOS)
            else:
                self.bound_endpoint = endpoint
            return True

        else:
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
            not self.is_open()
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

    def _compact_tx_ring(mut self, count: Int, sent_total: Int, should_retain: Bool):
        var base_payload = Int(self.tx_payloads.unsafe_ptr())
        var base_addr = Int(self.tx_addrs.unsafe_ptr())
        var base_iov = Int(self.tx_iovecs.unsafe_ptr())
        var base_mmsg = Int(self.tx_mmsgs.unsafe_ptr())
        var remaining = (count - sent_total) if should_retain else 0
        if remaining > 0 and sent_total > 0:
            _ = external_call["memmove", Int](
                base_payload,
                base_payload + sent_total * UDP_MAX_DATAGRAM_BYTES,
                Int64(remaining * UDP_MAX_DATAGRAM_BYTES),
            )
            _ = external_call["memmove", Int](
                base_addr,
                base_addr + sent_total * _SOCKADDR_STRIDE,
                Int64(remaining * _SOCKADDR_STRIDE),
            )
            _ = external_call["memmove", Int](
                base_iov,
                base_iov + sent_total * _IOVEC_STRIDE,
                Int64(remaining * _IOVEC_STRIDE),
            )
            _ = external_call["memmove", Int](
                base_mmsg,
                base_mmsg + sent_total * _MMSGHDR_STRIDE,
                Int64(remaining * _MMSGHDR_STRIDE),
            )
        self.tx_queued = remaining
        self.total_sent_packets += sent_total

    def flush_sends(mut self) -> Int:
        """Flushes queued TX datagrams (`sendmmsg` on Linux; `sendto` loop on Win32/macOS)."""
        if not self.is_open() or self.tx_queued <= 0:
            self.tx_queued = 0
            return 0

        var count = self.tx_queued
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
        var should_retain = True

        comptime if IS_WINDOWS:
            var fn_sendto = _wsa_proc(self.wsa_lib, "sendto")
            var fn_err = _wsa_proc(self.wsa_lib, "WSAGetLastError")
            if fn_sendto == 0 or fn_err == 0:
                self.tx_queued = 0
                return 0
            comptime WinSendToFn = def(Int, Int, Int32, Int32, Int, Int32) thin abi("C") -> Int32
            comptime WinErrFn = def() thin abi("C") -> Int32
            while sent_total < count:
                var p_slot = base_payload + sent_total * UDP_MAX_DATAGRAM_BYTES
                var a_slot = base_addr + sent_total * _SOCKADDR_STRIDE
                var v_slot = base_iov + sent_total * _IOVEC_STRIDE
                var pkt_len = Int32(Pointer[UInt64, ImmutAnyOrigin](unsafe_from_address=v_slot + 8)[unsafe_offset=0])
                var rc = Pointer(to=fn_sendto).unsafe_bitcast[WinSendToFn]()[unsafe_offset=0](
                    self.fd, p_slot, pkt_len, 0, a_slot, Int32(_SOCKADDR_IN6_LEN)
                )
                self.sendmmsg_calls += 1
                if rc < 0:
                    var err = Pointer(to=fn_err).unsafe_bitcast[WinErrFn]()[unsafe_offset=0]()
                    if err == WIN32_WSAEINTR:
                        continue
                    if err != WIN32_WSAEWOULDBLOCK and err != WIN32_WSAENOBUFS:
                        should_retain = False
                    break
                sent_total += 1
            self._compact_tx_ring(count, sent_total, should_retain)
            return sent_total

        elif CompilationTarget.is_macos():
            while sent_total < count:
                var p_slot = base_payload + sent_total * UDP_MAX_DATAGRAM_BYTES
                var a_slot = base_addr + sent_total * _SOCKADDR_STRIDE
                var v_slot = base_iov + sent_total * _IOVEC_STRIDE
                var pkt_len = Int64(Pointer[UInt64, ImmutAnyOrigin](unsafe_from_address=v_slot + 8)[unsafe_offset=0])
                var rc = Int(
                    external_call["sendto", Int64](
                        Int32(self.fd),
                        Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=p_slot),
                        pkt_len,
                        _MSG_DONTWAIT,
                        Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=a_slot),
                        UInt32(_SOCKADDR_IN6_LEN),
                    )
                )
                self.sendmmsg_calls += 1
                if rc < 0:
                    var err = _darwin_errno()
                    if err == EINTR:
                        continue
                    if err != DARWIN_EAGAIN and err != DARWIN_ENOBUFS:
                        should_retain = False
                    break
                sent_total += 1
            self._compact_tx_ring(count, sent_total, should_retain)
            return sent_total

        else:
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
                    var err = last_errno()
                    if err == EINTR:
                        continue
                    if err != _EAGAIN and err != _ENOBUFS:
                        should_retain = False
                    break
                if rc == 0:
                    break
                sent_total += rc

            self._compact_tx_ring(count, sent_total, should_retain)
            return sent_total

    def send_raw(
        mut self, dst: NetEndpoint, data_ptr: Int, length: Int
    ) -> Bool:
        if not self.queue_send_raw(dst, data_ptr, length):
            return False
        var expected = self.tx_queued
        var sent = self.flush_sends()
        return sent >= expected and self.tx_queued == 0

    def send_to(mut self, dst: NetEndpoint, payload: List[UInt8]) -> Bool:
        if len(payload) == 0:
            return False
        return self.send_raw(dst, Int(payload.unsafe_ptr()), len(payload))

    def recv_batch(mut self, mut out_views: List[DatagramView]) -> Int:
        """Receives up to 64 UDP datagrams (`recvmmsg` on Linux; `recvfrom` loop on Win32/macOS)."""
        out_views.clear()
        if not self.is_open():
            return 0
        if out_views.capacity() < UDP_RING_CAPACITY:
            out_views.reserve(UDP_RING_CAPACITY)

        var base_payload = Int(self.rx_payloads.unsafe_ptr())
        var base_addr = Int(self.rx_addrs.unsafe_ptr())
        var base_iov = Int(self.rx_iovecs.unsafe_ptr())
        var base_mmsg = Int(self.rx_mmsgs.unsafe_ptr())

        comptime if IS_WINDOWS:
            var fn_recvfrom = _wsa_proc(self.wsa_lib, "recvfrom")
            var fn_err = _wsa_proc(self.wsa_lib, "WSAGetLastError")
            if fn_recvfrom == 0 or fn_err == 0:
                return 0
            comptime WinRecvFromFn = def(Int, Int, Int32, Int32, Int, Int) thin abi("C") -> Int32
            comptime WinErrFn = def() thin abi("C") -> Int32
            for i in range(UDP_RING_CAPACITY):
                var p_slot = base_payload + i * UDP_MAX_DATAGRAM_BYTES
                var a_slot = base_addr + i * _SOCKADDR_STRIDE
                for z in range(4):
                    _store_u64_le(a_slot, z * 8, 0)
                var fromlen = Int32(_SOCKADDR_IN6_LEN)
                var rc = Int(
                    Pointer(to=fn_recvfrom).unsafe_bitcast[WinRecvFromFn]()[unsafe_offset=0](
                        self.fd, p_slot, Int32(UDP_MAX_DATAGRAM_BYTES), 0, a_slot, Int(Pointer(to=fromlen))
                    )
                )
                self.recvmmsg_calls += 1
                if rc < 0:
                    var err = Pointer(to=fn_err).unsafe_bitcast[WinErrFn]()[unsafe_offset=0]()
                    if err == WIN32_WSAEINTR:
                        continue
                    break
                if rc > 0 and rc <= UDP_MAX_DATAGRAM_BYTES:
                    var sender = NetEndpoint.from_sockaddr_for_os(a_slot, SOCK_OS_WINDOWS)
                    out_views.append(DatagramView(sender, p_slot, rc))
            self.total_recv_packets += len(out_views)
            return len(out_views)

        elif CompilationTarget.is_macos():
            for i in range(UDP_RING_CAPACITY):
                var p_slot = base_payload + i * UDP_MAX_DATAGRAM_BYTES
                var a_slot = base_addr + i * _SOCKADDR_STRIDE
                for z in range(4):
                    _store_u64_le(a_slot, z * 8, 0)
                var fromlen = UInt32(_SOCKADDR_IN6_LEN)
                var rc = Int(
                    external_call["recvfrom", Int64](
                        Int32(self.fd),
                        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_slot),
                        Int64(UDP_MAX_DATAGRAM_BYTES),
                        _MSG_DONTWAIT,
                        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a_slot),
                        Pointer(to=fromlen),
                    )
                )
                self.recvmmsg_calls += 1
                if rc < 0:
                    var err = _darwin_errno()
                    if err == EINTR:
                        continue
                    break
                if rc > 0 and rc <= UDP_MAX_DATAGRAM_BYTES:
                    var sender = NetEndpoint.from_sockaddr_for_os(a_slot, SOCK_OS_MACOS)
                    out_views.append(DatagramView(sender, p_slot, rc))
            self.total_recv_packets += len(out_views)
            return len(out_views)

        else:
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
