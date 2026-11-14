"""Game-agnostic Flint MCP & LiveLink introspection handler.

Allows AI agents and automated tools to query any running Flint game's build stamp,
tick counter, memory budgets, perceptual FLIP image diff, and screenshots with
zero game-specific coupling.
"""
from std.ffi import external_call
from flint.assets.image import decode_png
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote
from flint.core.build_info import build_stamp
from flint.core.memory import W_CPU_LIVE, W_VRAM_LIVE, mem_total
from flint.platform.os import read_bytes, close_fd
from flint.render.flip import flip_error_map, flip_mean

comptime AF_UNIX: Int32 = 1
comptime SOCK_STREAM: Int32 = 1
comptime SOCK_NONBLOCK: Int32 = 2048
comptime SOCK_CLOEXEC: Int32 = 524288


struct FlintMcpInfo(Copyable, Movable):
    var app_name: String
    var tick: Int
    var backend: String
    var custom_state_json: String

    def __init__(
        out self,
        app_name: String = "flint-app",
        tick: Int = 0,
        backend: String = "SIMD-CPU",
        custom_state_json: String = "{}",
    ):
        self.app_name = app_name
        self.tick = tick
        self.backend = backend
        self.custom_state_json = custom_state_json


def handle_flint_mcp_query(method: String, args_json: String, info: FlintMcpInfo) -> String:
    """Pure JSON responder callable over LiveLink Unix sockets, stdio MCP, or unit tests."""
    if method == "flint_hello":
        return (
            '{"engine":"flint","version":"0.1.0","app":'
            + json_quote(info.app_name)
            + ',"build":'
            + json_quote(build_stamp())
            + ',"backend":'
            + json_quote(info.backend)
            + ',"tick":'
            + String(info.tick)
            + "}"
        )
    if method == "flint_memory":
        return (
            '{"cpu_live_bytes":'
            + String(mem_total(W_CPU_LIVE))
            + ',"vram_live_bytes":'
            + String(mem_total(W_VRAM_LIVE))
            + "}"
        )
    if method == "flint_state":
        return (
            '{"tick":'
            + String(info.tick)
            + ',"state":'
            + info.custom_state_json
            + "}"
        )
    if method == "flint_flip_diff":
        var raw = List[UInt8]()
        for b in args_json.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid:
            return '{"ok":false,"error":"invalid json"}'
        var path_a = doc.text(doc.object_get(doc.root, "a"))
        var path_b = doc.text(doc.object_get(doc.root, "b"))
        var img_a = decode_png(read_bytes(path_a), 4096)
        var img_b = decode_png(read_bytes(path_b), 4096)
        if not img_a.valid or not img_b.valid or img_a.width != img_b.width or img_a.height != img_b.height:
            return '{"ok":false,"error":"mismatched or unreadable pngs"}'
        var err = flip_mean(flip_error_map(img_a.pixels, img_b.pixels, img_a.width, img_a.height))
        return '{"ok":true,"flip_mean":' + String(err) + "}"
    return '{"ok":false,"error":' + json_quote("unknown method: " + method) + "}"


def _make_sockaddr_un(path: String) -> List[UInt8]:
    """Builds a 110-byte Linux `sockaddr_un` (`AF_UNIX = 1` + NUL-terminated path)."""
    var addr = List[UInt8](length=110, fill=0)
    addr[0] = UInt8(1)
    addr[1] = UInt8(0)
    var pbytes = path.as_bytes()
    var limit = min(107, len(pbytes))
    for i in range(limit):
        addr[2 + i] = pbytes[i]
    return addr^


def _unlink_cpath(path: String):
    var cstr = List[UInt8]()
    for b in path.as_bytes():
        cstr.append(b)
    cstr.append(0)
    _ = external_call["unlink", Int32](cstr.unsafe_ptr())


struct FlintMcpLiveEndpoint(Movable):
    """Non-blocking Unix domain socket LiveLink server (`{"method":"flint_state","args":"{}"}`).

    Polled once per frame via `poll_once(info)` with zero blocking or heap churn when idle.
    """
    var socket_path: String
    var listen_fd: Int
    var active: Bool

    def __init__(out self, socket_path: String = ""):
        self.socket_path = socket_path
        self.listen_fd = -1
        self.active = False
        if socket_path.byte_length() > 0:
            _ = self.start(socket_path)

    def start(mut self, socket_path: String) -> Bool:
        self.close()
        self.socket_path = socket_path
        _unlink_cpath(socket_path)
        var fd = Int(external_call["socket", Int32](AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, Int32(0)))
        if fd < 0:
            return False
        var addr = _make_sockaddr_un(socket_path)
        if external_call["bind", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(110)) != 0:
            close_fd(fd)
            return False
        if external_call["listen", Int32](Int32(fd), Int32(8)) != 0:
            close_fd(fd)
            _unlink_cpath(socket_path)
            return False
        self.listen_fd = fd
        self.active = True
        return True

    def poll_once(mut self, info: FlintMcpInfo) -> Int:
        """Accepts and answers any queued non-blocking MCP requests. Returns count answered."""
        if not self.active or self.listen_fd < 0:
            return 0
        var answered = 0
        var buf = List[UInt8](length=4096, fill=0)
        for _ in range(4):
            var cfd = Int(
                external_call["accept4", Int32](
                    Int32(self.listen_fd), Int(0), Int(0), SOCK_CLOEXEC,
                )
            )
            if cfd < 0:
                break
            var n = Int(external_call["read", Int64](Int32(cfd), buf.unsafe_ptr(), Int64(4095)))
            if n > 0:
                var req_bytes = List[UInt8](capacity=n)
                for i in range(n):
                    req_bytes.append(buf[i])
                var method = String("flint_hello")
                var args = String("{}")
                var doc = parse_json(req_bytes)
                if doc.valid:
                    var m_txt = doc.text(doc.object_get(doc.root, "method"))
                    if m_txt.byte_length() > 0:
                        method = m_txt
                    var a_txt = doc.text(doc.object_get(doc.root, "args"))
                    if a_txt.byte_length() > 0:
                        args = a_txt
                var resp = handle_flint_mcp_query(method, args, info) + "\n"
                var rbytes = List[UInt8]()
                for b in resp.as_bytes():
                    rbytes.append(b)
                _ = external_call["write", Int64](Int32(cfd), rbytes.unsafe_ptr(), Int64(len(rbytes)))
                answered += 1
            close_fd(cfd)
        return answered

    def close(mut self):
        if self.listen_fd >= 0:
            close_fd(self.listen_fd)
            self.listen_fd = -1
        if self.active and self.socket_path.byte_length() > 0:
            _unlink_cpath(self.socket_path)
        self.active = False


def query_flint_mcp_unix(socket_path: String, method: String, args_json: String = "{}") -> String:
    """Connects to a running `FlintMcpLiveEndpoint` Unix socket and returns its JSON response."""
    var fd = Int(external_call["socket", Int32](AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, Int32(0)))
    if fd < 0:
        return '{"ok":false,"error":"socket failed"}'
    var addr = _make_sockaddr_un(socket_path)
    if external_call["connect", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(110)) != 0:
        close_fd(fd)
        return '{"ok":false,"error":"connect failed"}'
    var payload = '{"method":' + json_quote(method) + ',"args":' + json_quote(args_json) + "}"
    var pbytes = List[UInt8]()
    for b in payload.as_bytes():
        pbytes.append(b)
    _ = external_call["write", Int64](Int32(fd), pbytes.unsafe_ptr(), Int64(len(pbytes)))
    _ = external_call["shutdown", Int32](Int32(fd), Int32(1))  # SHUT_WR
    var buf = List[UInt8](length=4096, fill=0)
    var n = Int(external_call["read", Int64](Int32(fd), buf.unsafe_ptr(), Int64(4095)))
    close_fd(fd)
    if n <= 0:
        return '{"ok":false,"error":"empty response"}'
    var out = List[UInt8](capacity=n)
    for i in range(n):
        if buf[i] != 10:
            out.append(buf[i])
    return String(unsafe_from_utf8=out)
