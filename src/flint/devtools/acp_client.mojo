"""Pure-Mojo 1.2 Agent Client Protocol (ACP v1) client & JSON-RPC 2.0 session runner (`flint.devtools.acp_client`).

Provides:
- `JsonRpcFramer`: dual-mode JSON-RPC 2.0 framing (`Content-Length` headers or NDJSON lines).
- `PipeProcess`: non-blocking bidirectional POSIX pipe subprocess (`fork`/`execvp`/`close_range`).
- `AcpClient`: full ACP v1 client (`initialize`, `session/new`, `session/load`, `session/prompt`,
  `session/cancel`, `session/set_mode`, `session/set_model`, `session/set_config_option`,
  `session/request_permission`, `fs/*`, `terminal/*`), with `mcp_servers_json` injection for
  connecting ACP agents directly to Flint Studio's live MCP v2 Unix socket.
- Deterministic headless testing via `start_headless()` + `feed_bytes()` / `feed_text()`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max, min
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import (
    JSON_ARRAY,
    JSON_BOOL,
    JSON_NULL,
    JSON_NUMBER,
    JSON_OBJECT,
    JSON_STRING,
    json_quote,
    json_text,
)
from flint.platform.os import close_fd, get_time, last_errno, open_path_fd

comptime ACP_PROTOCOL_VERSION: Int = 1
comptime MAX_RPC_MESSAGE_BYTES: Int = 32 * 1024 * 1024

# JSON-RPC message kinds
comptime RPC_INVALID: Int = 0
comptime RPC_RESPONSE: Int = 1
comptime RPC_ERROR: Int = 2
comptime RPC_REQUEST: Int = 3
comptime RPC_NOTIFICATION: Int = 4

# ACP connection states
comptime ACP_STOPPED: Int = 0
comptime ACP_STARTING: Int = 1
comptime ACP_OPENING: Int = 2
comptime ACP_READY: Int = 3
comptime ACP_FAILED: Int = 4

# Structured ACP event kinds
comptime AE_NONE: Int = 0
comptime AE_THOUGHT: Int = 1
comptime AE_MESSAGE: Int = 2
comptime AE_PLAN: Int = 3
comptime AE_TOOL: Int = 4
comptime AE_TOOL_UPDATE: Int = 5
comptime AE_PERMISSION: Int = 6
comptime AE_FILE_PROPOSAL: Int = 7
comptime AE_CMD_APPROVAL: Int = 8
comptime AE_DONE: Int = 9
comptime AE_ERROR: Int = 10
comptime AE_READY: Int = 11
comptime AE_MODE: Int = 12
comptime AE_MODEL: Int = 13
comptime AE_THINKING: Int = 14
comptime AE_SERVICE: Int = 15
comptime AE_USER: Int = 16
comptime AE_OTHER: Int = 17

comptime ACP_INIT_TIMEOUT_SEC: Float64 = 30.0
comptime ACP_ERR_REFUSED: Int = -32000
comptime ACP_ERR_NOT_FOUND: Int = -32002
comptime ACP_ERR_METHOD_NOT_FOUND: Int = -32601
comptime ACP_ERR_INVALID_PARAMS: Int = -32602

comptime _SIGPIPE: Int32 = 13
comptime _SIGTERM: Int32 = 15
comptime _SIGKILL: Int32 = 9
comptime _EINTR: Int32 = 4
comptime _EAGAIN: Int32 = 11
comptime _ECHILD: Int32 = 10
comptime _O_NONBLOCK: Int32 = 2048


def _str_to_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        out.append(sb[i])
    return out^


def _c_string_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8](capacity=len(sb) + 1)
    for i in range(len(sb)):
        out.append(sb[i])
    out.append(0)
    return out^


def _ignore_sigpipe():
    var act = List[UInt64](length=24, fill=0)
    act[0] = UInt64(1)  # SIG_IGN
    var old = List[UInt64](length=24, fill=0)
    _ = external_call["sigaction", Int32](_SIGPIPE, act.unsafe_ptr(), old.unsafe_ptr())


def _set_nonblocking(fd: Int32):
    if fd < 0:
        return
    var fl = external_call["fcntl", Int32](fd, Int32(3), Int32(0))
    if fl >= 0:
        _ = external_call["fcntl", Int32](fd, Int32(4), fl | _O_NONBLOCK)


def _set_cloexec(fd: Int32):
    if fd < 0:
        return
    _ = external_call["fcntl", Int32](fd, Int32(2), Int32(1))


def clean_terminal_output(data: List[UInt8]) -> List[UInt8]:
    """Strips ANSI CSI/OSC escape sequences and CR bytes from terminal output."""
    var out = List[UInt8](capacity=len(data))
    var n = len(data)
    var i = 0
    while i < n:
        var b = data[i]
        if b == 27 and i + 1 < n:
            var c = data[i + 1]
            if c == 91:  # CSI
                i += 2
                while i < n and not (data[i] >= 64 and data[i] <= 126):
                    i += 1
                i += 1
                continue
            if c == 93:  # OSC
                i += 2
                while i < n and data[i] != 7 and not (data[i] == 27 and i + 1 < n and data[i + 1] == 92):
                    i += 1
                i += 1 if (i < n and data[i] == 7) else 2
                continue
            i += 2
            continue
        if b == 13:
            i += 1
            continue
        out.append(b)
        i += 1
    return out^


def allow_key(tool_kind: String, title: String) -> String:
    """Computes session allow-list key (`execute:<exact cmd>` or `<tool_kind>`)."""
    if tool_kind == "execute" or tool_kind == "run":
        var t = String(title.strip())
        if t.startswith("`") and t.endswith("`") and t.byte_length() >= 2:
            var inner = String(t[byte=1 : t.byte_length() - 1])
            t = inner^
        return "execute:" + String(t.strip())
    return tool_kind if tool_kind.byte_length() > 0 else String("other")


def option_of(option_kinds: List[String], option_ids: List[String], want: String) -> String:
    """Returns the first permission `option_id` matching `want` kind, or empty."""
    var n = min(len(option_kinds), len(option_ids))
    for k in range(n):
        if option_kinds[k] == want:
            return option_ids[k]
    return ""


# ============================================================================
# 1. JSON-RPC 2.0 Stream Framer & Message Classifier
# ============================================================================

struct JsonRpcFramer(Copyable, Movable):
    """Extracts complete JSON-RPC 2.0 bodies from a byte stream (NDJSON or Content-Length)."""
    var buf: List[UInt8]
    var pos: Int
    var dropped: Int
    var line_delimited: Bool

    def __init__(out self, line_delimited: Bool = True):
        self.buf = List[UInt8]()
        self.pos = 0
        self.dropped = 0
        self.line_delimited = line_delimited

    def clear(mut self):
        self.buf.clear()
        self.pos = 0
        self.dropped = 0

    def feed(mut self, data: List[UInt8]):
        if self.pos > 0 and self.pos == len(self.buf):
            self.buf.clear()
            self.pos = 0
        elif self.pos > 65536:
            var rest = List[UInt8](capacity=len(self.buf) - self.pos)
            for k in range(self.pos, len(self.buf)):
                rest.append(self.buf[k])
            self.buf = rest^
            self.pos = 0
        for i in range(len(data)):
            self.buf.append(data[i])

    def _header_end(self) -> Int:
        var n = len(self.buf)
        var i = self.pos
        while i + 3 < n:
            if (
                self.buf[i] == 13
                and self.buf[i + 1] == 10
                and self.buf[i + 2] == 13
                and self.buf[i + 3] == 10
            ):
                return i + 4
            i += 1
        return -1

    def _content_length(self, start: Int, end: Int) -> Int:
        var name = "content-length:".as_bytes()
        var i = start
        while i + len(name) <= end:
            var hit = True
            for k in range(len(name)):
                var c = Int(self.buf[i + k])
                if c >= 65 and c <= 90:
                    c += 32
                if c != Int(name[k]):
                    hit = False
                    break
            if hit:
                var j = i + len(name)
                while j < end and (self.buf[j] == 32 or self.buf[j] == 9):
                    j += 1
                var v = 0
                var digits = 0
                while j < end and self.buf[j] >= 48 and self.buf[j] <= 57 and digits < 12:
                    v = v * 10 + Int(self.buf[j]) - 48
                    digits += 1
                    j += 1
                if digits > 0:
                    return v
            i += 1
        return -1

    def next_body(mut self, mut body: List[UInt8]) -> Bool:
        if self.line_delimited:
            return self._next_line(body)
        while True:
            var he = self._header_end()
            if he < 0:
                return False
            var n = self._content_length(self.pos, he)
            if n < 0 or n > MAX_RPC_MESSAGE_BYTES:
                self.dropped += he - self.pos
                self.pos = he
                continue
            if len(self.buf) - he < n:
                return False
            body.clear()
            for k in range(he, he + n):
                body.append(self.buf[k])
            self.pos = he + n
            return True

    def _next_line(mut self, mut body: List[UInt8]) -> Bool:
        while True:
            var n = len(self.buf)
            var nl = -1
            for i in range(self.pos, n):
                if self.buf[i] == 10:
                    nl = i
                    break
            if nl < 0:
                if n - self.pos > MAX_RPC_MESSAGE_BYTES:
                    self.dropped += n - self.pos
                    self.pos = n
                return False
            var end = nl
            if end > self.pos and self.buf[end - 1] == 13:
                end -= 1
            var start = self.pos
            self.pos = nl + 1
            if end == start:
                continue
            body.clear()
            for k in range(start, end):
                body.append(self.buf[k])
            return True


struct JsonRpcMessage(Copyable, Movable):
    var kind: Int
    var id: Int
    var id_json: String
    var method: String
    var doc: JsonDocument

    def __init__(out self):
        self.kind = RPC_INVALID
        self.id = -1
        self.id_json = ""
        self.method = ""
        self.doc = JsonDocument()

    def result_node(self) -> Int:
        return self.doc.object_get(self.doc.root, "result")

    def params_node(self) -> Int:
        return self.doc.object_get(self.doc.root, "params")

    def error_message(self) -> String:
        var err = self.doc.object_get(self.doc.root, "error")
        if err < 0:
            return ""
        return self.doc.text(self.doc.object_get(err, "message"), "")


# ============================================================================
# 2. Non-Blocking Bidirectional POSIX Pipe Process
# ============================================================================

struct PipeProcess(Movable):
    """Non-blocking child process with stdin/stdout pipes and RAII cleanup."""
    var pid: Int32
    var in_fd: Int32
    var out_fd: Int32
    var running: Bool
    var eof: Bool
    var write_failed: Bool
    var exit_code: Int
    var inbox: List[UInt8]
    var outbox: List[UInt8]

    def __init__(out self):
        self.pid = -1
        self.in_fd = -1
        self.out_fd = -1
        self.running = False
        self.eof = False
        self.write_failed = False
        self.exit_code = -1
        self.inbox = List[UInt8]()
        self.outbox = List[UInt8]()

    def __deinit__(deinit self):
        self.stop()

    def start_shell(mut self, cmd: String, cwd: String, stderr_path: String = "/dev/null") -> Bool:
        var argv = List[String]()
        argv.append("/bin/sh")
        argv.append("-c")
        argv.append(cmd)
        return self.start_argv(argv, cwd, stderr_path)

    def start_argv(mut self, argv: List[String], cwd: String, stderr_path: String = "/dev/null") -> Bool:
        self.stop()
        if len(argv) == 0:
            return False
        _ignore_sigpipe()

        var stdin_pipe = List[Int32](length=2, fill=-1)
        var stdout_pipe = List[Int32](length=2, fill=-1)
        if external_call["pipe", Int32](stdin_pipe.unsafe_ptr()) != 0:
            return False
        if external_call["pipe", Int32](stdout_pipe.unsafe_ptr()) != 0:
            close_fd(Int(stdin_pipe[0]))
            close_fd(Int(stdin_pipe[1]))
            return False

        var in_r = stdin_pipe[0]
        var in_w = stdin_pipe[1]
        var out_r = stdout_pipe[0]
        var out_w = stdout_pipe[1]

        var args = List[List[UInt8]](capacity=len(argv))
        for i in range(len(argv)):
            args.append(_c_string_bytes(argv[i]))
        var ptrs = List[Int](length=len(args) + 1, fill=0)
        for i in range(len(args)):
            ptrs[i] = Int(args[i].unsafe_ptr())
        var dir_c = _c_string_bytes(cwd)
        var err_c = _c_string_bytes(stderr_path if stderr_path.byte_length() > 0 else String("/dev/null"))

        var pid = external_call["fork", Int32]()
        if pid == 0:
            _ = external_call["dup2", Int32](in_r, Int32(0))
            _ = external_call["dup2", Int32](out_w, Int32(1))
            var efd = external_call["openat", Int32](
                Int32(-100), err_c.unsafe_ptr(), Int32(1 | 64 | 1024), Int32(438)
            )
            if efd >= 0:
                _ = external_call["dup2", Int32](efd, Int32(2))
            _ = external_call["close_range", Int32](UInt32(3), UInt32(0xFFFFFFFF), UInt32(0))
            if cwd.byte_length() > 0:
                _ = external_call["chdir", Int32](dir_c.unsafe_ptr())
            _ = external_call["execvp", Int32](args[0].unsafe_ptr(), ptrs.unsafe_ptr())
            external_call["_exit", NoneType](Int32(127))

        # Keep buffers alive across fork
        _ = len(args)
        _ = ptrs[0]
        _ = dir_c[0]
        _ = err_c[0]

        close_fd(Int(in_r))
        close_fd(Int(out_w))
        if pid < 0:
            close_fd(Int(in_w))
            close_fd(Int(out_r))
            return False

        _set_cloexec(in_w)
        _set_cloexec(out_r)
        _set_nonblocking(in_w)
        _set_nonblocking(out_r)

        self.pid = pid
        self.in_fd = in_w
        self.out_fd = out_r
        self.running = True
        self.eof = False
        self.write_failed = False
        self.exit_code = -1
        self.inbox.clear()
        self.outbox.clear()
        return True

    def write(mut self, data: List[UInt8]):
        for i in range(len(data)):
            self.outbox.append(data[i])
        self.flush()

    def flush(mut self):
        if not self.running or self.write_failed or self.in_fd < 0 or len(self.outbox) == 0:
            return
        var off = 0
        var n = len(self.outbox)
        while off < n:
            var w = Int(
                external_call["write", Int64](
                    self.in_fd, self.outbox.unsafe_ptr().unsafe_offset(off), Int64(n - off)
                )
            )
            if w > 0:
                off += w
                continue
            var err = last_errno()
            if err == _EINTR:
                continue
            if err == _EAGAIN:
                break
            self.write_failed = True
            break
        if off > 0:
            var rest = List[UInt8](capacity=n - off)
            for k in range(off, n):
                rest.append(self.outbox[k])
            self.outbox = rest^

    def read_available(mut self) -> Int:
        if not self.running or self.eof or self.out_fd < 0:
            return 0
        var scratch = List[UInt8](length=16384, fill=0)
        var total = 0
        for _ in range(32):
            var n = Int(external_call["read", Int64](self.out_fd, scratch.unsafe_ptr(), Int64(16384)))
            if n > 0:
                for k in range(n):
                    self.inbox.append(scratch[k])
                total += n
                continue
            if n == 0:
                self.eof = True
                break
            var err = last_errno()
            if err == _EINTR:
                continue
            if err == _EAGAIN:
                break
            self.eof = True
            break
        return total

    def take_inbox(mut self) -> List[UInt8]:
        var out = List[UInt8]()
        swap(out, self.inbox)
        return out^

    def poll_exit(mut self) -> Bool:
        if self.pid <= 0:
            return True
        var status = Int32(0)
        var r = external_call["waitpid", Int32](self.pid, Pointer(to=status), Int32(1))
        if r == self.pid:
            var st = Int(status)
            self.exit_code = ((st >> 8) & 255) if (st & 127) == 0 else (128 + (st & 127))
            self.pid = -1
            self.running = False
            self.eof = True
            return True
        if r < 0 and last_errno() == _ECHILD:
            self.pid = -1
            self.running = False
            self.eof = True
            return True
        return False

    def stop(mut self):
        if self.in_fd >= 0:
            close_fd(Int(self.in_fd))
            self.in_fd = -1
        if self.pid > 0:
            if not self.poll_exit():
                _ = external_call["kill", Int32](self.pid, _SIGTERM)
                for _ in range(25):
                    if self.poll_exit():
                        break
                    _ = external_call["usleep", Int32](UInt32(10000))
                if self.pid > 0:
                    _ = external_call["kill", Int32](self.pid, _SIGKILL)
                    var st = Int32(0)
                    _ = external_call["waitpid", Int32](self.pid, Pointer(to=st), Int32(0))
                    self.pid = -1
        if self.out_fd >= 0:
            close_fd(Int(self.out_fd))
            self.out_fd = -1
        self.running = False
        self.eof = True


# ============================================================================
# 3. ACP v1 Events & Client
# ============================================================================

struct AcpEvent(Copyable, Movable):
    """Structured event emitted by `AcpClient` for Agent Mission Control & Code Drawer."""
    var kind: Int
    var text: String                # message/thought chunk, tool text, error reason, or new file content
    var id: String                  # toolCallId
    var title: String               # tool call title or command summary
    var tool_kind: String           # read, edit, execute, ...
    var status: String              # pending, in_progress, completed, failed, mode/model id
    var stop_reason: String         # end_turn, max_tokens, cancelled, error
    var kind_name: String           # raw sessionUpdate or RPC method name
    var request_id: String          # raw JSON id for answering permission/service requests
    var path: String                # target file path (for AE_FILE_PROPOSAL) or cwd (for AE_CMD_APPROVAL)
    var entries: List[String]       # plan entries
    var option_ids: List[String]    # permission option IDs
    var option_names: List[String]  # permission option display names
    var option_kinds: List[String]  # allow_once, allow_always, reject_once, reject_always
    var diffs: List[String]         # file paths changed by tool call
    var locations: List[String]     # "path:line" locations touched by tool call
    var argv: List[String]          # command argv for AE_CMD_APPROVAL

    def __init__(out self, kind: Int = AE_NONE):
        self.kind = kind
        self.text = ""
        self.id = ""
        self.title = ""
        self.tool_kind = ""
        self.status = ""
        self.stop_reason = ""
        self.kind_name = ""
        self.request_id = ""
        self.path = ""
        self.entries = List[String]()
        self.option_ids = List[String]()
        self.option_names = List[String]()
        self.option_kinds = List[String]()
        self.diffs = List[String]()
        self.locations = List[String]()
        self.argv = List[String]()


def _json_bool(doc: JsonDocument, node: Int, fallback: Bool = False) -> Bool:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != JSON_BOOL:
        return fallback
    return doc.nodes[node].boolean


def _content_block_text(doc: JsonDocument, node: Int) -> String:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != JSON_OBJECT:
        return ""
    var t = doc.text(doc.object_get(node, "type"), "")
    if t == "text":
        return doc.text(doc.object_get(node, "text"), "")
    if t == "resource_link":
        return "[" + doc.text(doc.object_get(node, "uri"), "") + "]"
    if t == "resource":
        var r = doc.object_get(node, "resource")
        return "[" + doc.text(doc.object_get(r, "uri"), "") + "]"
    return "[" + t + "]"


def _parse_session_update(doc: JsonDocument, u: Int) -> AcpEvent:
    var su = doc.text(doc.object_get(u, "sessionUpdate"), "")
    var ev = AcpEvent(AE_OTHER)
    ev.kind_name = su
    if su == "agent_message_chunk" or su == "agent_thought_chunk" or su == "user_message_chunk":
        if su == "agent_message_chunk":
            ev.kind = AE_MESSAGE
        elif su == "agent_thought_chunk":
            ev.kind = AE_THOUGHT
        else:
            ev.kind = AE_USER
        ev.text = _content_block_text(doc, doc.object_get(u, "content"))
        return ev^
    if su == "tool_call" or su == "tool_call_update":
        ev.kind = AE_TOOL if su == "tool_call" else AE_TOOL_UPDATE
        ev.id = doc.text(doc.object_get(u, "toolCallId"), "")
        ev.title = doc.text(doc.object_get(u, "title"), "")
        ev.tool_kind = doc.text(doc.object_get(u, "kind"), "")
        ev.status = doc.text(doc.object_get(u, "status"), "")
        if su == "tool_call" and ev.status.byte_length() == 0:
            ev.status = "pending"
        var content_arr = doc.object_get(u, "content")
        var n_content = doc.array_size(content_arr)
        for ci in range(n_content):
            var item = doc.at(content_arr, ci)
            var t = doc.text(doc.object_get(item, "type"), "")
            if t == "content":
                var s = _content_block_text(doc, doc.object_get(item, "content"))
                if s.byte_length() > 0:
                    ev.text += ("\n" if ev.text.byte_length() > 0 else "") + s
            elif t == "diff":
                ev.diffs.append(doc.text(doc.object_get(item, "path"), ""))
            elif t == "terminal":
                var tid = doc.text(doc.object_get(item, "terminalId"), "")
                ev.text += ("\n" if ev.text.byte_length() > 0 else "") + "[terminal " + tid + "]"
        var locs = doc.object_get(u, "locations")
        var n_locs = doc.array_size(locs)
        for li in range(n_locs):
            var loc = doc.at(locs, li)
            var lp = doc.text(doc.object_get(loc, "path"), "")
            var ln_node = doc.object_get(loc, "line")
            if ln_node >= 0 and doc.nodes[ln_node].kind == JSON_NUMBER:
                ev.locations.append(lp + ":" + String(doc.integer(ln_node, 0)))
            else:
                ev.locations.append(lp)
        return ev^
    if su == "plan":
        ev.kind = AE_PLAN
        var ents = doc.object_get(u, "entries")
        var n_ents = doc.array_size(ents)
        for ei in range(n_ents):
            var en = doc.at(ents, ei)
            var st = doc.text(doc.object_get(en, "status"), "pending")
            var body = doc.text(doc.object_get(en, "content"), "")
            ev.entries.append("[" + st + "] " + body)
        return ev^
    return ev^


struct AcpClient(Movable):
    """Non-blocking ACP v1 client supporting both live child processes and headless byte injection."""
    var proc: PipeProcess
    var framer: JsonRpcFramer
    var headless_mode: Bool
    var state: Int
    var problem: String
    var cwd: String
    var agent_name: String
    var version: Int
    var load_session: Bool
    var supports_images: Bool
    var embedded_context: Bool
    var services_enabled: Bool
    var mcp_servers_json: String
    var session_id: String
    var resume_session_id: String
    var resumed: Bool
    var next_req_id: Int
    var pending_ids: List[Int]
    var pending_methods: List[String]
    var init_id: Int
    var new_id: Int
    var load_id: Int
    var prompt_id: Int
    var mode_req_id: Int
    var model_req_id: Int
    var cfg_req_id: Int
    var started_sec: Float64
    var prompt_started_sec: Float64
    var mode: String
    var mode_wanted: String
    var mode_ids: List[String]
    var mode_names: List[String]
    var current_model: String
    var model_wanted: String
    var model_ids: List[String]
    var model_names: List[String]
    var thinking: String
    var thinking_wanted: String
    var thinking_ids: List[String]
    var thinking_names: List[String]
    var open_permissions: List[String]
    var events: List[AcpEvent]
    var outbound_log: List[String]
    var invalid_frames: Int

    def __init__(out self):
        self.proc = PipeProcess()
        self.framer = JsonRpcFramer(line_delimited=True)
        self.headless_mode = False
        self.state = ACP_STOPPED
        self.problem = ""
        self.cwd = ""
        self.agent_name = ""
        self.version = 0
        self.load_session = False
        self.supports_images = False
        self.embedded_context = False
        self.services_enabled = True
        self.mcp_servers_json = "[]"
        self.session_id = ""
        self.resume_session_id = ""
        self.resumed = False
        self.next_req_id = 1
        self.pending_ids = List[Int]()
        self.pending_methods = List[String]()
        self.init_id = -1
        self.new_id = -1
        self.load_id = -1
        self.prompt_id = -1
        self.mode_req_id = -1
        self.model_req_id = -1
        self.cfg_req_id = -1
        self.started_sec = 0.0
        self.prompt_started_sec = 0.0
        self.mode = ""
        self.mode_wanted = ""
        self.mode_ids = List[String]()
        self.mode_names = List[String]()
        self.current_model = ""
        self.model_wanted = ""
        self.model_ids = List[String]()
        self.model_names = List[String]()
        self.thinking = ""
        self.thinking_wanted = ""
        self.thinking_ids = List[String]()
        self.thinking_names = List[String]()
        self.open_permissions = List[String]()
        self.events = List[AcpEvent]()
        self.outbound_log = List[String]()
        self.invalid_frames = 0

    def configure_studio_mcp_socket(
        mut self, studio_exe: String, socket_path: String, server_name: String = "flint-studio"
    ):
        """Configures `mcp_servers_json` so ACP agents connect directly to Flint Studio's live MCP v2 socket."""
        self.mcp_servers_json = (
            '[{"name":'
            + json_quote(server_name)
            + ',"command":'
            + json_quote(studio_exe)
            + ',"args":["--mcp-connect",'
            + json_quote(socket_path)
            + '],"env":[]}]'
        )

    def active(self) -> Bool:
        return self.state != ACP_STOPPED and self.state != ACP_FAILED

    def busy(self) -> Bool:
        return self.prompt_id >= 0

    def _reset_session_state(mut self):
        self.framer.clear()
        self.problem = ""
        self.session_id = ""
        self.resumed = False
        self.pending_ids.clear()
        self.pending_methods.clear()
        self.init_id = -1
        self.new_id = -1
        self.load_id = -1
        self.prompt_id = -1
        self.mode_req_id = -1
        self.model_req_id = -1
        self.cfg_req_id = -1
        self.mode = ""
        self.mode_ids.clear()
        self.mode_names.clear()
        self.current_model = ""
        self.model_ids.clear()
        self.model_names.clear()
        self.thinking = ""
        self.thinking_ids.clear()
        self.thinking_names.clear()
        self.open_permissions.clear()
        self.events.clear()
        self.outbound_log.clear()
        self.invalid_frames = 0

    def _send_json_line(mut self, payload: String):
        if self.headless_mode:
            self.outbound_log.append(payload)
            return
        var b = _str_to_bytes(payload)
        b.append(10)
        self.proc.write(b)

    def _send_request(mut self, method: String, params_json: String) -> Int:
        var id = self.next_req_id
        self.next_req_id += 1
        var msg = (
            '{"jsonrpc":"2.0","id":'
            + String(id)
            + ',"method":'
            + json_quote(method)
            + ',"params":'
            + params_json
            + "}"
        )
        self.pending_ids.append(id)
        self.pending_methods.append(method)
        self._send_json_line(msg)
        return id

    def _send_notify(mut self, method: String, params_json: String):
        var msg = (
            '{"jsonrpc":"2.0","method":'
            + json_quote(method)
            + ',"params":'
            + params_json
            + "}"
        )
        self._send_json_line(msg)

    def _send_response(mut self, id_json: String, result_json: String):
        var body = result_json if result_json.byte_length() > 0 else String("null")
        self._send_json_line('{"jsonrpc":"2.0","id":' + id_json + ',"result":' + body + "}")

    def _send_error(mut self, id_json: String, code: Int, message: String):
        self._send_json_line(
            '{"jsonrpc":"2.0","id":'
            + id_json
            + ',"error":{"code":'
            + String(code)
            + ',"message":'
            + json_quote(message)
            + "}}"
        )

    def _send_initialize(mut self, now_sec: Float64):
        var svc = "true" if self.services_enabled else "false"
        var params = (
            '{"protocolVersion":'
            + String(ACP_PROTOCOL_VERSION)
            + ',"clientCapabilities":{"fs":{"readTextFile":'
            + svc
            + ',"writeTextFile":'
            + svc
            + '},"terminal":'
            + svc
            + '},"clientInfo":{"name":"flint-studio","title":"Flint Studio","version":"5.0"}}'
        )
        self.init_id = self._send_request("initialize", params)
        self.state = ACP_STARTING
        self.started_sec = now_sec

    def start(
        mut self, cmd: String, cwd: String, stderr_path: String = "/dev/null", now_sec: Float64 = -1.0
    ) -> Bool:
        """Starts an external ACP agent CLI and sends `initialize`."""
        self.proc.stop()
        self.headless_mode = False
        self._reset_session_state()
        self.cwd = cwd
        var t = now_sec if now_sec >= 0.0 else get_time()
        if not self.proc.start_shell("exec " + cmd, cwd, stderr_path):
            self._fail("failed to start ACP agent: " + cmd)
            return False
        self._send_initialize(t)
        return True

    def start_headless(mut self, cwd: String = ".", now_sec: Float64 = 0.0):
        """Initializes the ACP client in headless mock mode (no subprocess; driven by `feed_bytes`)."""
        self.proc.stop()
        self.headless_mode = True
        self._reset_session_state()
        self.cwd = cwd
        self._send_initialize(now_sec)

    def stop(mut self):
        if self.busy():
            self.cancel()
            if not self.headless_mode:
                self.proc.flush()
        if not self.headless_mode:
            self.proc.stop()
        self.state = ACP_STOPPED
        self.session_id = ""
        self.prompt_id = -1
        self.open_permissions.clear()

    def _fail(mut self, why: String):
        self.state = ACP_FAILED
        self.problem = why
        self.prompt_id = -1
        var ev = AcpEvent(AE_ERROR)
        ev.text = why
        self.events.append(ev^)

    def send_prompt(mut self, text: String, now_sec: Float64 = -1.0) -> Bool:
        var blocks = '[{"type":"text","text":' + json_quote(text) + "}]"
        return self.send_prompt_blocks(blocks, now_sec)

    def send_prompt_blocks(mut self, blocks_json: String, now_sec: Float64 = -1.0) -> Bool:
        if self.state != ACP_READY or self.busy():
            return False
        var params = (
            '{"sessionId":' + json_quote(self.session_id) + ',"prompt":' + blocks_json + "}"
        )
        self.prompt_id = self._send_request("session/prompt", params)
        self.prompt_started_sec = now_sec if now_sec >= 0.0 else get_time()
        return True

    def cancel(mut self):
        if not self.busy():
            return
        for i in range(len(self.open_permissions)):
            var perm_id = self.open_permissions[i]
            self._send_response(perm_id, '{"outcome":{"outcome":"cancelled"}}')
        self.open_permissions.clear()
        var cancel_params = '{"sessionId":' + json_quote(self.session_id) + "}"
        self._send_notify("session/cancel", cancel_params)

    def answer_permission(mut self, request_id: String, option_id: String) -> Bool:
        var found = -1
        for i in range(len(self.open_permissions)):
            if self.open_permissions[i] == request_id:
                found = i
                break
        if found < 0:
            return False
        _ = self.open_permissions.pop(found)
        if option_id.byte_length() == 0:
            self._send_response(request_id, '{"outcome":{"outcome":"cancelled"}}')
        else:
            self._send_response(
                request_id,
                '{"outcome":{"outcome":"selected","optionId":' + json_quote(option_id) + "}}",
            )
        return True

    def answer_service(mut self, request_id: String, result_json: String = "{}"):
        self._send_response(request_id, result_json)

    def answer_service_error(mut self, request_id: String, code: Int, message: String):
        self._send_error(request_id, code, message)

    def set_mode(mut self, mode_id: String) -> Bool:
        if self.state != ACP_READY or mode_id not in self.mode_ids:
            return False
        self.mode_wanted = mode_id
        self.mode_req_id = self._send_request(
            "session/set_mode",
            '{"sessionId":' + json_quote(self.session_id) + ',"modeId":' + json_quote(mode_id) + "}",
        )
        return True

    def set_model(mut self, model_id: String) -> Bool:
        if self.state != ACP_READY or model_id not in self.model_ids:
            return False
        self.model_wanted = model_id
        self.model_req_id = self._send_request(
            "session/set_model",
            '{"sessionId":'
            + json_quote(self.session_id)
            + ',"modelId":'
            + json_quote(model_id)
            + "}",
        )
        return True

    def set_thinking(mut self, level: String) -> Bool:
        if self.state != ACP_READY or level not in self.thinking_ids:
            return False
        self.thinking_wanted = level
        self.cfg_req_id = self._send_request(
            "session/set_config_option",
            '{"sessionId":'
            + json_quote(self.session_id)
            + ',"configId":"thought_level","value":'
            + json_quote(level)
            + "}",
        )
        return True

    def take_events(mut self) -> List[AcpEvent]:
        var out = List[AcpEvent]()
        swap(out, self.events)
        return out^

    def take_outbound_log(mut self) -> List[String]:
        var out = List[String]()
        swap(out, self.outbound_log)
        return out^

    def _classify_frame(mut self, body: List[UInt8]) -> JsonRpcMessage:
        var m = JsonRpcMessage()
        m.doc = parse_json(body)
        if not m.doc.valid or m.doc.root < 0 or m.doc.nodes[m.doc.root].kind != JSON_OBJECT:
            self.invalid_frames += 1
            return m^
        var idn = m.doc.object_get(m.doc.root, "id")
        var has_id = idn >= 0 and m.doc.nodes[idn].kind != JSON_NULL
        if has_id:
            m.id_json = json_text(m.doc, idn)
            if m.doc.nodes[idn].kind == JSON_NUMBER:
                m.id = m.doc.integer(idn, -1)
        var mn = m.doc.object_get(m.doc.root, "method")
        if mn >= 0 and m.doc.nodes[mn].kind == JSON_STRING:
            m.method = m.doc.text(mn, "")
            if has_id:
                m.kind = RPC_REQUEST
                m.id = -1
            else:
                m.kind = RPC_NOTIFICATION
            return m^
        var has_res = m.doc.object_get(m.doc.root, "result") >= 0
        var has_err = m.doc.object_get(m.doc.root, "error") >= 0
        if has_res or has_err:
            m.kind = RPC_ERROR if has_err else RPC_RESPONSE
            for k in range(len(self.pending_ids)):
                if self.pending_ids[k] == m.id:
                    m.method = self.pending_methods[k]
                    _ = self.pending_ids.pop(k)
                    _ = self.pending_methods.pop(k)
                    break
            return m^
        self.invalid_frames += 1
        return m^

    def feed_bytes(mut self, data: List[UInt8]):
        """Deterministically feeds raw JSON-RPC stream bytes (used by `pump()` and headless tests)."""
        self.framer.feed(data)
        var body = List[UInt8]()
        while self.framer.next_body(body):
            var msg = self._classify_frame(body)
            if msg.kind != RPC_INVALID:
                self._handle_message(msg)

    def feed_text(mut self, text: String):
        """Convenience wrapper around `feed_bytes` for UTF-8 JSON-RPC NDJSON strings."""
        self.feed_bytes(_str_to_bytes(text))

    def pump(mut self, now_sec: Float64 = -1.0):
        """Flushes outbound requests, drains child stdout, parses events, and checks timeouts."""
        if self.headless_mode or not self.proc.running:
            return
        var t = now_sec if now_sec >= 0.0 else get_time()
        self.proc.flush()
        _ = self.proc.read_available()
        var chunk = self.proc.take_inbox()
        if len(chunk) > 0:
            self.feed_bytes(chunk)
        if self.state == ACP_STARTING and (t - self.started_sec) > ACP_INIT_TIMEOUT_SEC:
            self.proc.stop()
            self._fail("ACP agent did not respond to initialize within 30s")
            return
        if self.proc.eof or self.proc.poll_exit():
            self.proc.stop()
            if self.state != ACP_FAILED:
                self._fail(
                    "ACP agent process exited"
                    + (" before session ready" if self.state != ACP_READY else String(""))
                )

    def _parse_config_options(mut self, doc: JsonDocument, opts: Int):
        var n_opts = doc.array_size(opts)
        for oi in range(n_opts):
            var opt = doc.at(opts, oi)
            var cid = doc.text(doc.object_get(opt, "id"), "")
            if cid == "thought_level":
                self.thinking = doc.text(doc.object_get(opt, "currentValue"), "")
                self.thinking_ids.clear()
                self.thinking_names.clear()
                var arr = doc.object_get(opt, "options")
                for k in range(doc.array_size(arr)):
                    var o = doc.at(arr, k)
                    self.thinking_ids.append(doc.text(doc.object_get(o, "value"), ""))
                    self.thinking_names.append(doc.text(doc.object_get(o, "name"), ""))
            elif cid == "model":
                var cv = doc.text(doc.object_get(opt, "currentValue"), "")
                if cv.byte_length() > 0:
                    self.current_model = cv
            elif cid == "mode":
                var mv = doc.text(doc.object_get(opt, "currentValue"), "")
                if mv.byte_length() > 0:
                    self.mode = mv

    def _parse_session_meta(mut self, doc: JsonDocument, r: Int):
        var modes = doc.object_get(r, "modes")
        if modes >= 0:
            self.mode = doc.text(doc.object_get(modes, "currentModeId"), "")
            self.mode_ids.clear()
            self.mode_names.clear()
            var am = doc.object_get(modes, "availableModes")
            for k in range(doc.array_size(am)):
                var md = doc.at(am, k)
                self.mode_ids.append(doc.text(doc.object_get(md, "id"), ""))
                self.mode_names.append(doc.text(doc.object_get(md, "name"), ""))
        var models = doc.object_get(r, "models")
        if models >= 0:
            self.current_model = doc.text(doc.object_get(models, "currentModelId"), "")
            self.model_ids.clear()
            self.model_names.clear()
            var av = doc.object_get(models, "availableModels")
            for k in range(doc.array_size(av)):
                var ml = doc.at(av, k)
                self.model_ids.append(doc.text(doc.object_get(ml, "modelId"), ""))
                self.model_names.append(doc.text(doc.object_get(ml, "name"), ""))
        self._parse_config_options(doc, doc.object_get(r, "configOptions"))

    def _handle_message(mut self, m: JsonRpcMessage):
        if m.kind == RPC_RESPONSE or m.kind == RPC_ERROR:
            self._handle_response(m)
            return
        if m.kind == RPC_NOTIFICATION:
            if m.method == "session/update":
                var p = m.params_node()
                if m.doc.text(m.doc.object_get(p, "sessionId"), "") != self.session_id:
                    return
                var u = m.doc.object_get(p, "update")
                if u >= 0:
                    var ev = _parse_session_update(m.doc, u)
                    if ev.kind_name == "current_mode_update":
                        self.mode = m.doc.text(m.doc.object_get(u, "currentModeId"), "")
                        ev.kind = AE_MODE
                        ev.status = self.mode
                    elif ev.kind_name == "config_option_update":
                        self._parse_config_options(m.doc, m.doc.object_get(u, "configOptions"))
                    self.events.append(ev^)
            return
        if m.kind == RPC_REQUEST:
            var p = m.params_node()
            if m.method == "session/request_permission":
                if not self.busy():
                    self._send_response(m.id_json, '{"outcome":{"outcome":"cancelled"}}')
                    return
                var ev = AcpEvent(AE_PERMISSION)
                ev.request_id = m.id_json
                var tc = m.doc.object_get(p, "toolCall")
                ev.id = m.doc.text(m.doc.object_get(tc, "toolCallId"), "")
                ev.title = m.doc.text(m.doc.object_get(tc, "title"), "")
                ev.tool_kind = m.doc.text(m.doc.object_get(tc, "kind"), "")
                var opts = m.doc.object_get(p, "options")
                for oi in range(m.doc.array_size(opts)):
                    var o = m.doc.at(opts, oi)
                    ev.option_ids.append(m.doc.text(m.doc.object_get(o, "optionId"), ""))
                    ev.option_names.append(m.doc.text(m.doc.object_get(o, "name"), ""))
                    ev.option_kinds.append(m.doc.text(m.doc.object_get(o, "kind"), ""))
                self.open_permissions.append(m.id_json)
                self.events.append(ev^)
            elif self.services_enabled and m.method == "fs/write_text_file":
                var ev = AcpEvent(AE_FILE_PROPOSAL)
                ev.kind_name = m.method
                ev.request_id = m.id_json
                ev.path = m.doc.text(m.doc.object_get(p, "path"), "")
                ev.text = m.doc.text(m.doc.object_get(p, "content"), "")
                self.events.append(ev^)
            elif self.services_enabled and m.method == "terminal/create":
                var ev = AcpEvent(AE_CMD_APPROVAL)
                ev.kind_name = m.method
                ev.request_id = m.id_json
                var cmd = m.doc.text(m.doc.object_get(p, "command"), "")
                ev.path = m.doc.text(m.doc.object_get(p, "cwd"), self.cwd)
                ev.argv.append(cmd)
                var args_node = m.doc.object_get(p, "args")
                var title = cmd
                for ai in range(m.doc.array_size(args_node)):
                    var a = m.doc.text(m.doc.at(args_node, ai), "")
                    ev.argv.append(a)
                    title += " " + a
                ev.title = title
                self.events.append(ev^)
            elif self.services_enabled and (
                m.method.startswith("fs/") or m.method.startswith("terminal/")
            ):
                var ev = AcpEvent(AE_SERVICE)
                ev.kind_name = m.method
                ev.request_id = m.id_json
                ev.text = json_text(m.doc, p) if p >= 0 else String("{}")
                self.events.append(ev^)
            else:
                self._send_error(
                    m.id_json, ACP_ERR_METHOD_NOT_FOUND, "Flint Studio does not offer " + m.method
                )

    def _handle_response(mut self, m: JsonRpcMessage):
        if m.id == self.init_id:
            self.init_id = -1
            if m.kind == RPC_ERROR:
                self._fail("initialize failed: " + m.error_message())
                return
            var r = m.result_node()
            self.version = m.doc.integer(m.doc.object_get(r, "protocolVersion"), 0)
            if self.version != ACP_PROTOCOL_VERSION:
                self.proc.stop()
                self._fail(
                    "unsupported ACP version "
                    + String(self.version)
                    + " (expected "
                    + String(ACP_PROTOCOL_VERSION)
                    + ")"
                )
                return
            var info = m.doc.object_get(r, "agentInfo")
            self.agent_name = m.doc.text(m.doc.object_get(info, "title"), "")
            if self.agent_name.byte_length() == 0:
                self.agent_name = m.doc.text(m.doc.object_get(info, "name"), "Agent")
            var caps = m.doc.object_get(r, "agentCapabilities")
            self.load_session = _json_bool(m.doc, m.doc.object_get(caps, "loadSession"), False)
            var pcaps = m.doc.object_get(caps, "promptCapabilities")
            self.supports_images = _json_bool(m.doc, m.doc.object_get(pcaps, "image"), False)
            self.embedded_context = _json_bool(
                m.doc, m.doc.object_get(pcaps, "embeddedContext"), False
            )
            if self.load_session and self.resume_session_id.byte_length() > 0:
                self.session_id = self.resume_session_id
                self.load_id = self._send_request(
                    "session/load",
                    '{"sessionId":'
                    + json_quote(self.resume_session_id)
                    + ',"cwd":'
                    + json_quote(self.cwd)
                    + ',"mcpServers":'
                    + self.mcp_servers_json
                    + "}",
                )
            else:
                self.new_id = self._send_request(
                    "session/new",
                    '{"cwd":' + json_quote(self.cwd) + ',"mcpServers":' + self.mcp_servers_json + "}",
                )
            self.state = ACP_OPENING
            return

        if m.id == self.load_id:
            self.load_id = -1
            if m.kind == RPC_ERROR:
                self.session_id = ""
                self.resume_session_id = ""
                self.new_id = self._send_request(
                    "session/new",
                    '{"cwd":' + json_quote(self.cwd) + ',"mcpServers":' + self.mcp_servers_json + "}",
                )
                return
            var sid = m.doc.text(m.doc.object_get(m.result_node(), "sessionId"), "")
            if sid.byte_length() > 0:
                self.session_id = sid
            self.resumed = True
            self._parse_session_meta(m.doc, m.result_node())
            self.state = ACP_READY
            self.events.append(AcpEvent(AE_READY))
            return

        if m.id == self.new_id:
            self.new_id = -1
            if m.kind == RPC_ERROR:
                self._fail("session/new failed: " + m.error_message())
                return
            self.session_id = m.doc.text(m.doc.object_get(m.result_node(), "sessionId"), "")
            if self.session_id.byte_length() == 0:
                self._fail("session/new returned empty sessionId")
                return
            self.resumed = False
            self._parse_session_meta(m.doc, m.result_node())
            self.state = ACP_READY
            self.events.append(AcpEvent(AE_READY))
            return

        if m.id == self.mode_req_id:
            self.mode_req_id = -1
            var ev = AcpEvent(AE_MODE)
            if m.kind == RPC_ERROR:
                ev.text = m.error_message()
            else:
                self.mode = self.mode_wanted
            ev.status = self.mode
            self.events.append(ev^)
            return

        if m.id == self.model_req_id:
            self.model_req_id = -1
            var ev = AcpEvent(AE_MODEL)
            if m.kind == RPC_ERROR:
                ev.text = m.error_message()
            else:
                self.current_model = self.model_wanted
            ev.status = self.current_model
            self.events.append(ev^)
            return

        if m.id == self.cfg_req_id:
            self.cfg_req_id = -1
            var ev = AcpEvent(AE_THINKING)
            if m.kind == RPC_ERROR:
                ev.text = m.error_message()
            else:
                self.thinking = self.thinking_wanted
                self._parse_config_options(
                    m.doc, m.doc.object_get(m.result_node(), "configOptions")
                )
            ev.status = self.thinking
            self.events.append(ev^)
            return

        if m.id == self.prompt_id:
            self.prompt_id = -1
            self.open_permissions.clear()
            if m.kind == RPC_ERROR:
                var ev = AcpEvent(AE_ERROR)
                ev.stop_reason = "error"
                ev.text = m.error_message()
                self.events.append(ev^)
            else:
                var ev = AcpEvent(AE_DONE)
                ev.stop_reason = m.doc.text(
                    m.doc.object_get(m.result_node(), "stopReason"), "end_turn"
                )
                self.events.append(ev^)
