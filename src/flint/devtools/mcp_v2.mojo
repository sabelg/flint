"""Flint v5 AI-Native Model Context Protocol v2 (`McpServerV2`, JSON-RPC 2.0, `mcp_v2.mojo`).

Spec-compliant Model Context Protocol (`2024-11-05`) JSON-RPC 2.0 server supporting:
  - **`stdio` Transport**: Both LSP/MCP `Content-Length: <N>\\r\\n\\r\\n` framing and newline-delimited JSON-RPC 2.0.
  - **Persistent Non-Blocking Unix Domain Sockets (`FlintMcpUnixListener`)**: Dynamic per-client stream buffers
    with multi-request session persistence and **zero** fixed `mmap` address collisions!
  - **Core Protocol**: `initialize`, `ping`, `notifications/initialized`, `tools/list`, and `tools/call`.
  - **All 14 Engine / Studio / Asset / Physics / Render AI Tools**:
    1. `ecs.query_entities`
    2. `ecs.spawn_entity`
    3. `ecs.patch_components`
    4. `ecs.despawn_entity`
    5. `render.capture_viewport` (with perceptual `flip.mojo` FLIP mean error diff!)
    6. `render.inspect_gpu_frame`
    7. `assets.list_and_cook`
    8. `procedural.generate_asset`
    9. `physics.step_deterministic`
    10. `physics.raycast_query`
    11. `studio.pie_control`
    12. `studio.undo_redo`
    13. `render.configure_npr_cel`
    14. `render.trigger_post_fx`
"""
from std.ffi import external_call
from std.math import abs, cos, isfinite, max, min, sin
from std.sys.info import CompilationTarget
from flint.assets.checkpoint import pack_checkpoint_lz4
from flint.assets.cooker_v2 import DEFAULT_COOK_CACHE_DIR, cook_workspace
from flint.assets.image import decode_png, write_png
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.assets.server import AssetServer
from flint.core.build_info import build_stamp
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.memory import W_CPU_LIVE, W_VRAM_LIVE, mem_total
from flint.core.serial import BlobWriter
from flint.platform.os import (
    close_fd,
    env_now,
    env_value,
    get_time,
    last_errno,
    make_dirs,
    parent_directory,
    read_bytes,
    write_bytes,
)
from flint.render.aaa_npr_vfx import AaaImpactFrameParams, AaaShockwaveRing
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import flip_error_map, flip_mean
from flint.render.simd_canvas import Camera3D, SimdCanvas, pack_argb
from flint.studio.studio_app import (
    COLLIDER_BOX,
    COLLIDER_CAPSULE,
    COLLIDER_NONE,
    COLLIDER_SPHERE,
    COLLIDER_STATIC_MESH,
    PIE_PAUSED,
    PIE_PLAYING,
    PIE_STOPPED,
    SHADING_NPR_CEL,
    SHADING_PBR,
    SHADING_STYLIZED_VFX,
    FlintStudioState,
    bake_procedural_to_glb,
)

comptime MCP_PROTOCOL_VERSION = "2024-11-05"
comptime MAX_MCP_MESSAGE_BYTES: Int = 4 * 1024 * 1024  # 4 MiB per-client / per-frame cap
comptime MAX_MCP_HEADER_BYTES: Int = 8 * 1024          # 8 KiB max header before \r\n\r\n

comptime JSONRPC_PARSE_ERROR: Int = -32700
comptime JSONRPC_INVALID_REQUEST: Int = -32600
comptime JSONRPC_METHOD_NOT_FOUND: Int = -32601
comptime JSONRPC_INVALID_PARAMS: Int = -32602
comptime JSONRPC_INTERNAL_ERROR: Int = -32603

comptime _AF_UNIX: Int32 = 1
comptime _SOCK_STREAM: Int32 = 1
comptime _SOCK_NONBLOCK: Int32 = 2048
comptime _SOCK_CLOEXEC: Int32 = 524288
comptime _SOL_SOCKET: Int32 = 1
comptime _SO_PEERCRED: Int32 = 17
comptime _MSG_NOSIGNAL: Int32 = 0x4000
comptime _EINTR: Int32 = 4
comptime _EAGAIN: Int32 = 11
comptime _POLLOUT: Int16 = 0x0004
comptime _POLLERR_MASK: Int16 = 0x0008 | 0x0010 | 0x0020

comptime _STAT_MODE_OFF: Int = 24 if CompilationTarget.is_x86() else 16
comptime _STAT_UID_OFF: Int = 28 if CompilationTarget.is_x86() else 24
comptime _S_IFMT: UInt32 = 0o170000
comptime _S_IFDIR: UInt32 = 0o040000
comptime _S_IFLNK: UInt32 = 0o120000


# ============================================================================
# 0. POSIX Socket Permissions, Peer Auth, Bounded Send & Workspace Sandbox
# ============================================================================

def _c_string_from_ptr(addr: Int, max_len: Int = 4096) -> String:
    if addr == 0:
        return String()
    var out = List[UInt8]()
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    for i in range(max_len):
        var b = p[unsafe_offset=i]
        if b == 0:
            break
        out.append(b)
    return String(unsafe_from_utf8=out)


def _lstat_mode_uid(path: String) -> Tuple[Bool, UInt32, UInt32]:
    if path.byte_length() == 0:
        return (False, 0, 0)
    var owned = path
    var buf = List[UInt8](length=256, fill=0)
    if external_call["lstat", Int32](owned.as_c_string_span(), buf.unsafe_ptr()) != 0:
        return (False, 0, 0)
    var base = Int(buf.unsafe_ptr())
    var mode = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=base + _STAT_MODE_OFF)[unsafe_offset=0]
    var uid = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=base + _STAT_UID_OFF)[unsafe_offset=0]
    _ = len(buf)
    return (True, mode, uid)


def _getcwd_str() -> String:
    var buf = List[UInt8](length=4096, fill=0)
    var got = external_call["getcwd", Int](buf.unsafe_ptr(), Int64(4096))
    if got == 0:
        return String(".")
    var s = _c_string_from_ptr(Int(buf.unsafe_ptr()), 4096)
    _ = len(buf)
    return s^


def _realpath_existing(path: String) -> Tuple[Bool, String]:
    if path.byte_length() == 0:
        return (False, String())
    var owned = path
    var resolved = List[UInt8](length=4096, fill=0)
    var got = external_call["realpath", Int](owned.as_c_string_span(), resolved.unsafe_ptr())
    if got == 0:
        return (False, String())
    var s = _c_string_from_ptr(Int(resolved.unsafe_ptr()), 4096)
    _ = len(resolved)
    return (True, s^)


def _strip_trailing_slashes(path: String) -> String:
    var cur = path
    while cur.byte_length() > 1 and cur.as_bytes()[cur.byte_length() - 1] == 47:
        var next_s = String(cur[byte=0 : cur.byte_length() - 1])
        cur = next_s^
    return cur^


def _has_forbidden_path_bytes(s: String) -> Bool:
    for b in s.as_bytes():
        if b < 32 or b == 92 or b == 127:  # NUL, control chars, backslash, DEL
            return True
    return False


def _split_and_reject_dotdot(raw_path: String) raises -> List[String]:
    """Splits `raw_path` on `'/'` and raises immediately if any `'..'` segment is present."""
    var segs = List[String]()
    var b = raw_path.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        while i < n and b[i] == 47:
            i += 1
        if i >= n:
            break
        var j = i
        while j < n and b[j] != 47:
            j += 1
        var seg = String(raw_path[byte=i:j])
        if seg == "..":
            raise Error("path traversal '..' rejected")
        if seg != "." and seg.byte_length() > 0:
            segs.append(seg^)
        i = j
    return segs^


def _resolve_workspace_root(workspace_root: String) raises -> Tuple[String, String]:
    """Returns `(canonical_root, lexical_abs_root)` for `workspace_root`."""
    if workspace_root.byte_length() == 0 or _has_forbidden_path_bytes(workspace_root):
        raise Error("invalid workspace_root")
    var base = workspace_root
    if not base.startswith("/"):
        var cwd = _strip_trailing_slashes(_getcwd_str())
        if base == ".":
            base = cwd
        else:
            base = (cwd + "/" + base) if cwd != "/" else ("/" + base)
    var segs = _split_and_reject_dotdot(base)
    var lex = String("/")
    for i in range(len(segs)):
        if i == 0:
            lex = "/" + segs[i]
        else:
            lex += "/" + segs[i]
    var rp = _realpath_existing(lex)
    if rp[0]:
        return (_strip_trailing_slashes(rp[1]), lex)
    return (lex, lex)


def _sanitize_workspace_path(
    workspace_root: String,
    user_path: String,
    allow_root_dir: Bool = False,
) raises -> String:
    if user_path.byte_length() == 0:
        raise Error("empty path rejected")
    if _has_forbidden_path_bytes(user_path):
        raise Error("forbidden NUL/control/backslash byte in path")

    # Reject any '..' segment in raw input before any normalization!
    var raw_segs = _split_and_reject_dotdot(user_path)
    var roots = _resolve_workspace_root(workspace_root)
    var canon_root = roots[0]
    var lex_root = roots[1]

    var rel_segs = List[String]()
    if user_path.startswith("/"):
        var clean_u = _strip_trailing_slashes(user_path)
        var prefix_c = canon_root + "/"
        var prefix_l = lex_root + "/"
        if allow_root_dir and (clean_u == canon_root or clean_u == lex_root):
            pass
        elif canon_root == "/":
            rel_segs = raw_segs^
        elif clean_u.startswith(prefix_c):
            var sub = String(clean_u[byte=prefix_c.byte_length() : clean_u.byte_length()])
            rel_segs = _split_and_reject_dotdot(sub)
        elif lex_root != canon_root and clean_u.startswith(prefix_l):
            var sub = String(clean_u[byte=prefix_l.byte_length() : clean_u.byte_length()])
            rel_segs = _split_and_reject_dotdot(sub)
        else:
            raise Error("absolute path outside workspace_root rejected: " + user_path)
    else:
        rel_segs = raw_segs^

    if not allow_root_dir and len(rel_segs) == 0:
        raise Error("output path must specify a file inside workspace_root")

    # Walk every prefix under `canon_root` and reject symlinks (`S_ISLNK`) & non-dir parents!
    var root_ls = _lstat_mode_uid(canon_root)
    if root_ls[0] and (root_ls[1] & _S_IFMT) == _S_IFLNK:
        raise Error("workspace_root symlink rejected")

    var cur = canon_root
    var deepest_existing_dir = canon_root
    for i in range(len(rel_segs)):
        cur = (cur + "/" + rel_segs[i]) if cur != "/" else ("/" + rel_segs[i])
        var ls = _lstat_mode_uid(cur)
        if ls[0]:
            var ftype = ls[1] & _S_IFMT
            if ftype == _S_IFLNK:
                raise Error("symlink in sandbox path rejected: " + cur)
            if i + 1 < len(rel_segs):
                if ftype != _S_IFDIR:
                    raise Error("parent component is not a directory: " + cur)
                deepest_existing_dir = cur

    var anc_rp = _realpath_existing(deepest_existing_dir)
    if anc_rp[0]:
        var resolved_anc = _strip_trailing_slashes(anc_rp[1])
        if canon_root != "/" and resolved_anc != canon_root and not resolved_anc.startswith(canon_root + "/"):
            raise Error("resolved parent directory escapes workspace_root")

    return cur^


def sanitize_output_path(workspace_root: String, user_path: String) raises -> String:
    """Validates and normalizes `user_path` strictly inside `workspace_root`:
    rejects empty strings, NUL/control/backslash bytes, leading `'/'` outside `workspace_root`,
    any `'..'` path segment, and any symlink (`lstat` `S_ISLNK`) along the path."""
    return _sanitize_workspace_path(workspace_root, user_path, allow_root_dir=False)


def _ensure_private_dir_0700(dir_path: String, expected_uid: UInt32) -> Bool:
    if dir_path.byte_length() == 0 or _has_forbidden_path_bytes(dir_path) or not dir_path.startswith("/"):
        return False
    if dir_path.find("..") >= 0:
        return False
    var pdir = parent_directory(dir_path)
    if pdir.byte_length() > 0 and pdir != "/" and pdir != ".":
        var pls = _lstat_mode_uid(pdir)
        if not pls[0]:
            return False
    var owned = dir_path
    var old_umask = external_call["umask", UInt32](UInt32(0o077))
    var mrc = external_call["mkdirat", Int32](Int32(-100), owned.as_c_string_span(), UInt32(0o700))
    _ = external_call["umask", UInt32](old_umask)
    if mrc != 0 and last_errno() != Int32(17):  # EEXIST = 17
        return False
    var ls = _lstat_mode_uid(dir_path)
    if not ls[0] or (ls[1] & _S_IFMT) != _S_IFDIR or ls[2] != expected_uid:
        return False
    if external_call["chmod", Int32](owned.as_c_string_span(), UInt32(0o700)) != 0:
        return False
    var ls2 = _lstat_mode_uid(dir_path)
    return ls2[0] and (ls2[1] & _S_IFMT) == _S_IFDIR and ls2[2] == expected_uid and (ls2[1] & UInt32(0o077)) == 0


def default_mcp_socket_path() -> String:
    """Returns a `0700` user-private MCP Unix socket path:
    `$XDG_RUNTIME_DIR/flint-<uid>/mcp.sock`, falling back to `/tmp/flint-mcp-<uid>/mcp.sock`."""
    var uid = external_call["getuid", UInt32]()
    var uid_str = String(Int(uid))
    var xdg = env_now("XDG_RUNTIME_DIR")
    if xdg.byte_length() == 0:
        xdg = env_value("XDG_RUNTIME_DIR")
    if xdg.byte_length() > 0:
        var clean_xdg = _strip_trailing_slashes(xdg)
        var xdg_dir = clean_xdg + "/flint-" + uid_str
        if xdg_dir.byte_length() + 9 < 107 and _ensure_private_dir_0700(xdg_dir, uid):
            return xdg_dir + "/mcp.sock"

    var fallback_dir = "/tmp/flint-mcp-" + uid_str
    _ = _ensure_private_dir_0700(fallback_dir, uid)
    return fallback_dir + "/mcp.sock"


def _const_time_str_eq(a: String, b: String) -> Bool:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    if len(ab) != len(bb):
        return False
    var diff: UInt8 = 0
    for i in range(len(ab)):
        diff |= ab[i] ^ bb[i]
    return diff == 0


def send_all_bounded(
    fd: Int,
    addr: Int,
    length: Int,
    timeout_ms: Int = 2000,
    is_socket: Bool = True,
) -> Bool:
    """Writes all `length` bytes from `addr` to `fd`, retrying `EINTR` and polling `POLLOUT`
    on `EAGAIN`/`EWOULDBLOCK` up to `timeout_ms` so large JSON-RPC payloads never truncate."""
    if fd < 0 or addr == 0 or length < 0:
        return False
    if length == 0:
        return True

    var deadline = get_time() + Float64(max(1, timeout_ms)) / 1000.0
    var sent = 0
    var pfd = List[UInt8](length=8, fill=0)

    while sent < length:
        var cur_ptr = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr + sent)
        var rem = Int64(length - sent)
        var n: Int
        if is_socket:
            n = Int(external_call["send", Int64](Int32(fd), cur_ptr, rem, _MSG_NOSIGNAL))
        else:
            n = Int(external_call["write", Int64](Int32(fd), cur_ptr, rem))

        if n > 0:
            sent += n
            continue
        if n == 0:
            _ = len(pfd)
            return False

        var err = last_errno()
        if err == _EINTR:
            continue
        if err == _EAGAIN:
            var rem_ms = Int((deadline - get_time()) * 1000.0 + 0.999)
            if rem_ms <= 0:
                _ = len(pfd)
                return False
            var pbase = Int(pfd.unsafe_ptr())
            Pointer[Int32, MutAnyOrigin](unsafe_from_address=pbase)[unsafe_offset=0] = Int32(fd)
            Pointer[Int16, MutAnyOrigin](unsafe_from_address=pbase + 4)[unsafe_offset=0] = _POLLOUT
            Pointer[Int16, MutAnyOrigin](unsafe_from_address=pbase + 6)[unsafe_offset=0] = Int16(0)
            var wait_slice = Int32(min(rem_ms, 50))
            var prc = external_call["poll", Int32](pfd.unsafe_ptr(), UInt64(1), wait_slice)
            if prc < 0:
                if last_errno() == _EINTR:
                    continue
                _ = len(pfd)
                return False
            if prc > 0:
                var rev = Pointer[Int16, ImmutAnyOrigin](unsafe_from_address=pbase + 6)[unsafe_offset=0]
                if (rev & _POLLERR_MASK) != 0 and (rev & _POLLOUT) == 0:
                    _ = len(pfd)
                    return False
            continue

        _ = len(pfd)
        return False

    _ = len(pfd)
    return True


# ============================================================================
# 1. JSON-RPC 2.0 Helpers & Stream Frame Codec (`Content-Length` + Newline)
# ============================================================================

def _json_raw_id(doc: JsonDocument, id_node: Int) -> String:
    if id_node < 0 or id_node >= len(doc.nodes):
        return "null"
    var k = doc.nodes[id_node].kind
    if k == 2:
        return doc.nodes[id_node].text
    if k == 3:
        return json_quote(doc.nodes[id_node].text)
    return "null"


def _jsonrpc_ok(id_raw: String, result_json: String) -> String:
    return '{"jsonrpc":"2.0","id":' + id_raw + ',"result":' + result_json + "}"


def _jsonrpc_err(id_raw: String, code: Int, message: String) -> String:
    return (
        '{"jsonrpc":"2.0","id":'
        + id_raw
        + ',"error":{"code":'
        + String(code)
        + ',"message":'
        + json_quote(message)
        + "}}"
    )


def _mcp_tool_result(payload_json: String, is_error: Bool = False) -> String:
    return (
        '{"content":[{"type":"text","text":'
        + json_quote(payload_json)
        + '}],"structuredContent":'
        + payload_json
        + ',"isError":'
        + ("true" if is_error else "false")
        + "}"
    )


@always_inline
def _fmt_json_f64(v: Float64) -> String:
    if not isfinite(v):
        return "0.0"
    return String(v)


def _fmt_vec3_json(v: Vec3) -> String:
    return "[" + _fmt_json_f64(Float64(v.x)) + "," + _fmt_json_f64(Float64(v.y)) + "," + _fmt_json_f64(Float64(v.z)) + "]"


def _fmt_quat_json(q: Quat) -> String:
    return (
        "["
        + _fmt_json_f64(Float64(q.x)) + ","
        + _fmt_json_f64(Float64(q.y)) + ","
        + _fmt_json_f64(Float64(q.z)) + ","
        + _fmt_json_f64(Float64(q.w))
        + "]"
    )


def _parse_finite_f32_arg(doc: JsonDocument, node: Int, fallback: Float32) -> Tuple[Bool, Float32]:
    if node < 0 or node >= len(doc.nodes):
        return (True, fallback)
    if doc.nodes[node].kind != 2:
        return (False, fallback)
    var val = Float32(doc.nodes[node].number)
    if not isfinite(val):
        return (False, fallback)
    return (True, val)


def _parse_vec3_arg(doc: JsonDocument, node: Int, fallback: Vec3) -> Tuple[Bool, Vec3]:
    if node < 0 or node >= len(doc.nodes):
        return (True, fallback)
    if doc.nodes[node].kind != 4 or doc.array_size(node) != 3:
        return (False, fallback)
    var c0 = doc.at(node, 0)
    var c1 = doc.at(node, 1)
    var c2 = doc.at(node, 2)
    if doc.nodes[c0].kind != 2 or doc.nodes[c1].kind != 2 or doc.nodes[c2].kind != 2:
        return (False, fallback)
    var x = Float32(doc.nodes[c0].number)
    var y = Float32(doc.nodes[c1].number)
    var z = Float32(doc.nodes[c2].number)
    if not (isfinite(x) and isfinite(y) and isfinite(z)):
        return (False, fallback)
    return (True, Vec3(x, y, z))


def _parse_quat_arg(doc: JsonDocument, node: Int, fallback: Quat) -> Tuple[Bool, Quat]:
    if node < 0 or node >= len(doc.nodes):
        return (True, fallback)
    if doc.nodes[node].kind != 4:
        return (False, fallback)
    var sz = doc.array_size(node)
    if sz == 4:
        var c0 = doc.at(node, 0); var c1 = doc.at(node, 1); var c2 = doc.at(node, 2); var c3 = doc.at(node, 3)
        if doc.nodes[c0].kind != 2 or doc.nodes[c1].kind != 2 or doc.nodes[c2].kind != 2 or doc.nodes[c3].kind != 2:
            return (False, fallback)
        var qx = Float32(doc.nodes[c0].number)
        var qy = Float32(doc.nodes[c1].number)
        var qz = Float32(doc.nodes[c2].number)
        var qw = Float32(doc.nodes[c3].number)
        if not (isfinite(qx) and isfinite(qy) and isfinite(qz) and isfinite(qw)):
            return (False, fallback)
        var q = Quat(qx, qy, qz, qw).normalized()
        if not (isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)):
            return (False, fallback)
        return (True, q)
    if sz == 3:
        var ev = _parse_vec3_arg(doc, node, Vec3())
        if not ev[0]:
            return (False, fallback)
        var hr = Float32(0.01745329252 * 0.5)
        var qx = Quat(sin(ev[1].x * hr), 0.0, 0.0, cos(ev[1].x * hr))
        var qy = Quat(0.0, sin(ev[1].y * hr), 0.0, cos(ev[1].y * hr))
        var qz = Quat(0.0, 0.0, sin(ev[1].z * hr), cos(ev[1].z * hr))
        var q = qz.multiply(qy).multiply(qx).normalized()
        if not (isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)):
            return (False, fallback)
        return (True, q)
    return (False, fallback)


def _collider_from_string(s: String) -> Int:
    if s == "none":
        return COLLIDER_NONE
    if s == "sphere":
        return COLLIDER_SPHERE
    if s == "capsule":
        return COLLIDER_CAPSULE
    if s == "mesh" or s == "bvh8":
        return COLLIDER_STATIC_MESH
    return COLLIDER_BOX


def _collider_to_string(k: Int) -> String:
    if k == COLLIDER_NONE:
        return "none"
    if k == COLLIDER_SPHERE:
        return "sphere"
    if k == COLLIDER_CAPSULE:
        return "capsule"
    if k == COLLIDER_STATIC_MESH:
        return "bvh8"
    return "box"


def _shading_mode_to_string(mode: Int) -> String:
    if mode == SHADING_NPR_CEL:
        return "npr_cel"
    if mode == SHADING_STYLIZED_VFX:
        return "stylized_vfx"
    return "pbr"


def _shading_mode_from_string(s: String) -> Tuple[Bool, Int]:
    if s == "pbr":
        return (True, SHADING_PBR)
    if s == "npr_cel" or s == "cel":
        return (True, SHADING_NPR_CEL)
    if s == "stylized_vfx" or s == "vfx":
        return (True, SHADING_STYLIZED_VFX)
    return (False, SHADING_PBR)


def encode_mcp_frame(json_payload: String, use_content_length: Bool = False) -> List[UInt8]:
    """Frames `json_payload` as either `Content-Length: <N>\\r\\n\\r\\n<body` or `<body\\n`."""
    var out = List[UInt8]()
    if json_payload.byte_length() == 0:
        return out^
    if use_content_length:
        var hdr = "Content-Length: " + String(json_payload.byte_length()) + "\r\n\r\n"
        for b in hdr.as_bytes():
            out.append(b)
        for b in json_payload.as_bytes():
            out.append(b)
    else:
        for b in json_payload.as_bytes():
            out.append(b)
        out.append(10)
    return out^


@always_inline
def _ascii_lower(b: UInt8) -> UInt8:
    return (b + 32) if (b >= 65 and b <= 90) else b


struct McpFrameParseResult(Movable):
    var has_frame: Bool
    var used_content_length: Bool
    var protocol_error: Bool
    var error_code: Int
    var error_message: String
    var payload: List[UInt8]

    def __init__(out self):
        self.has_frame = False
        self.used_content_length = False
        self.protocol_error = False
        self.error_code = 0
        self.error_message = String()
        self.payload = List[UInt8]()

    @staticmethod
    def fatal(code: Int, message: String, content_len: Bool = False) -> Self:
        var r = Self()
        r.protocol_error = True
        r.used_content_length = content_len
        r.error_code = code
        r.error_message = message
        return r^


def parse_next_mcp_frame(mut buf: List[UInt8]) -> McpFrameParseResult:
    """Bounded, hostile-input-hardened MCP stream parser (`<= 4 MiB` message cap, `<= 8 KiB` header cap).
    Never wedges on negative/non-numeric/oversized `Content-Length` or unterminated garbage headers."""
    if len(buf) > MAX_MCP_MESSAGE_BYTES:
        buf.clear()
        return McpFrameParseResult.fatal(
            JSONRPC_INVALID_REQUEST, "MCP message exceeds 4 MiB maximum size"
        )

    var start = 0
    while start < len(buf) and (buf[start] == 32 or buf[start] == 9 or buf[start] == 10 or buf[start] == 13):
        start += 1
        if start > MAX_MCP_HEADER_BYTES:
            buf.clear()
            return McpFrameParseResult.fatal(
                JSONRPC_PARSE_ERROR, "Excessive leading whitespace in MCP stream"
            )
    if start > 0 and start == len(buf):
        buf.clear()
        return McpFrameParseResult()
    if start >= len(buf):
        return McpFrameParseResult()

    # Case 1: `Content-Length:` HTTP/LSP-style header framing
    var cl_prefix = "content-length:".as_bytes()
    var avail = len(buf) - start
    if buf[start] != 123:  # Not '{'
        var check_n = min(len(cl_prefix), avail)
        var matches_cl = True
        for k in range(check_n):
            if _ascii_lower(buf[start + k]) != cl_prefix[k]:
                matches_cl = False
                break

        if not matches_cl:
            buf.clear()
            return McpFrameParseResult.fatal(
                JSONRPC_PARSE_ERROR, "Malformed MCP frame header (expected Content-Length or JSON object)"
            )

        if avail < len(cl_prefix):
            return McpFrameParseResult()

        var hdr_end = -1
        var body_start = -1
        var scan_limit = min(len(buf) - 1, start + MAX_MCP_HEADER_BYTES + 4)
        for i in range(start, scan_limit):
            if i + 3 < len(buf) and buf[i] == 13 and buf[i + 1] == 10 and buf[i + 2] == 13 and buf[i + 3] == 10:
                hdr_end = i
                body_start = i + 4
                break
            if buf[i] == 10 and buf[i + 1] == 10:
                hdr_end = i
                body_start = i + 2
                break

        if hdr_end < 0:
            if avail > MAX_MCP_HEADER_BYTES:
                buf.clear()
                return McpFrameParseResult.fatal(
                    JSONRPC_INVALID_REQUEST, "MCP header exceeds 8 KiB without CRLFCRLF", content_len=True
                )
            var wait_cl = McpFrameParseResult()
            wait_cl.used_content_length = True
            return wait_cl^

        if hdr_end - start > MAX_MCP_HEADER_BYTES:
            buf.clear()
            return McpFrameParseResult.fatal(
                JSONRPC_INVALID_REQUEST, "MCP header exceeds 8 KiB limit", content_len=True
            )

        # Parse strict positive decimal integer on the Content-Length line
        var line_end = start + len(cl_prefix)
        while line_end < hdr_end and buf[line_end] != 13 and buf[line_end] != 10:
            line_end += 1

        var cur = start + len(cl_prefix)
        while cur < line_end and (buf[cur] == 32 or buf[cur] == 9):
            cur += 1

        var clen = 0
        var digits = 0
        var bad_num = False
        while cur < line_end:
            var ch = buf[cur]
            if ch >= 48 and ch <= 57:
                var d = Int(ch - 48)
                if clen > (MAX_MCP_MESSAGE_BYTES - d) // 10:
                    clen = MAX_MCP_MESSAGE_BYTES + 1
                    bad_num = True
                else:
                    clen = clen * 10 + d
                digits += 1
            elif ch == 32 or ch == 9:
                while cur < line_end and (buf[cur] == 32 or buf[cur] == 9):
                    cur += 1
                if cur < line_end:
                    bad_num = True
                break
            else:
                bad_num = True
                break
            cur += 1

        if (
            digits == 0
            or bad_num
            or clen <= 0
            or clen > MAX_MCP_MESSAGE_BYTES
            or clen > MAX_MCP_MESSAGE_BYTES - body_start
        ):
            buf.clear()
            return McpFrameParseResult.fatal(
                JSONRPC_INVALID_REQUEST, "Invalid or oversized Content-Length value", content_len=True
            )

        if len(buf) - body_start < clen:
            var wait_body = McpFrameParseResult()
            wait_body.used_content_length = True
            return wait_body^

        var msg = List[UInt8](capacity=clen)
        for i in range(clen):
            msg.append(buf[body_start + i])
        var rem_start = body_start + clen
        var rem = List[UInt8](capacity=len(buf) - rem_start)
        for i in range(rem_start, len(buf)):
            rem.append(buf[i])
        buf = rem^

        var ok = McpFrameParseResult()
        ok.has_frame = True
        ok.used_content_length = True
        ok.payload = msg^
        return ok^

    # Case 2: Newline-delimited JSON-RPC (`{...}\n`) or standalone valid JSON object
    var nl_at = -1
    for i in range(start, len(buf)):
        if buf[i] == 10:
            nl_at = i
            break
    if nl_at >= 0:
        var msg = List[UInt8](capacity=nl_at - start)
        for i in range(start, nl_at):
            if buf[i] != 13:
                msg.append(buf[i])
        var rem = List[UInt8](capacity=len(buf) - (nl_at + 1))
        for i in range(nl_at + 1, len(buf)):
            rem.append(buf[i])
        buf = rem^
        var ok_nl = McpFrameParseResult()
        ok_nl.has_frame = True
        ok_nl.used_content_length = False
        ok_nl.payload = msg^
        return ok_nl^

    # Check if buffer holds a complete un-terminated JSON object
    if start < len(buf) and buf[start] == 123 and buf[len(buf) - 1] == 125:
        var cand = List[UInt8](capacity=len(buf) - start)
        for i in range(start, len(buf)):
            cand.append(buf[i])
        var probe = parse_json(cand)
        if probe.valid:
            buf.clear()
            var ok_obj = McpFrameParseResult()
            ok_obj.has_frame = True
            ok_obj.used_content_length = False
            ok_obj.payload = cand^
            return ok_obj^

    return McpFrameParseResult()


def extract_next_mcp_frame(mut buf: List[UInt8]) -> Tuple[Bool, Bool, List[UInt8]]:
    """Extracts the next complete JSON-RPC message from dynamic byte stream `buf`.
    Returns `(has_message, used_content_length_framing, message_bytes)`."""
    var res = parse_next_mcp_frame(buf)
    if res.protocol_error or not res.has_frame:
        return (False, res.used_content_length, List[UInt8]())
    return (True, res.used_content_length, res.payload^)


# ============================================================================
# 2. Tool Schema Catalog (`tools/list` — All 14 Tools)
# ============================================================================

def _mcp_tools_list_json() -> String:
    return (
        '{"tools":['
        + '{"name":"ecs.query_entities","description":"Queries live Studio/ECS entities with optional name or collider filters.","inputSchema":{"type":"object","properties":{"name_contains":{"type":"string"},"collider":{"type":"string"},"visible_only":{"type":"boolean"}}}},'
        + '{"name":"ecs.spawn_entity","description":"Spawns an entity recorded on the reversible Studio Undo/Redo stack.","inputSchema":{"type":"object","properties":{"name":{"type":"string"},"position":{"type":"array"},"rotation":{"type":"array"},"scale":{"type":"array"},"collider":{"type":"string"},"mass":{"type":"number"},"base_color":{"type":"array"},"roughness":{"type":"number"},"metallic":{"type":"number"}}}},'
        + '{"name":"ecs.patch_components","description":"Patches entity TRS, material, visibility, or physics properties via reversible EditorTransaction.","inputSchema":{"type":"object","required":["entity_id"],"properties":{"entity_id":{"type":"integer"},"name":{"type":"string"},"position":{"type":"array"},"rotation":{"type":"array"},"scale":{"type":"array"},"base_color":{"type":"array"},"roughness":{"type":"number"},"metallic":{"type":"number"},"mass":{"type":"number"},"visible":{"type":"boolean"},"coalesce_key":{"type":"integer"}}}},'
        + '{"name":"ecs.despawn_entity","description":"Despawns an entity reversibly (restorable via studio.undo_redo).","inputSchema":{"type":"object","required":["entity_id"],"properties":{"entity_id":{"type":"integer"}}}},'
        + '{"name":"render.capture_viewport","description":"Captures a viewport PNG screenshot inside workspace_root and optionally computes perceptual FLIP error against reference_path.","inputSchema":{"type":"object","required":["output_path"],"properties":{"output_path":{"type":"string"},"reference_path":{"type":"string"},"width":{"type":"integer"},"height":{"type":"integer"}}}},'
        + '{"name":"render.inspect_gpu_frame","description":"Returns GPU render-graph pass timings, VRAM aliasing savings, residency streaming stats, and memory watermarks.","inputSchema":{"type":"object","properties":{}}},'
        + '{"name":"assets.list_and_cook","description":"Lists mounted .flpk v3 archives & live AssetServer arenas, and optionally runs incremental parallel cook_workspace inside workspace_root.","inputSchema":{"type":"object","properties":{"source_dir":{"type":"string"},"output_flpk":{"type":"string"},"cache_dir":{"type":"string"},"force_recook":{"type":"boolean"}}}},'
        + '{"name":"procedural.generate_asset","description":"Generates a Procedural Forge asset (material_orb, catmull_torus, colonnade, reference_grid, foliage_sample, oak, bollard) and bakes .glb inside workspace_root.","inputSchema":{"type":"object","required":["preset"],"properties":{"preset":{"type":"string"},"output_glb_path":{"type":"string"},"seed":{"type":"integer"},"spawn_in_scene":{"type":"boolean"},"position":{"type":"array"}}}},'
        + '{"name":"physics.step_deterministic","description":"Advances deterministic IEEE-754 rigid-body physics for N fixed ticks (optional sandboxed output_path checkpoint).","inputSchema":{"type":"object","properties":{"ticks":{"type":"integer"},"dt":{"type":"number"},"output_path":{"type":"string"}}}},'
        + '{"name":"physics.raycast_query","description":"Casts a 3D analytical ray against scene colliders (Sphere, OBB Box, Capsule, Bvh8MeshCollider).","inputSchema":{"type":"object","required":["origin","direction"],"properties":{"origin":{"type":"array"},"direction":{"type":"array"},"max_distance":{"type":"number"}}}},'
        + '{"name":"studio.pie_control","description":"Controls 1-Click Play-In-Editor (play, pause, step, stop) backed by LZ4 checkpoint restoration.","inputSchema":{"type":"object","required":["action"],"properties":{"action":{"type":"string"},"ticks":{"type":"integer"}}}},'
        + '{"name":"studio.undo_redo","description":"Executes algebraic undo or redo transactions on FlintStudioState.","inputSchema":{"type":"object","required":["action"],"properties":{"action":{"type":"string"},"steps":{"type":"integer"}}}},'
        + '{"name":"render.configure_npr_cel","description":"Configures PBR / NPR Cel / Stylized VFX shading, outlines, rim/SSS/hair-halo, UV-scroll erosion, stepped anime Hz, and volume-preserving smear stretch on an entity with reversible Undo/Redo recording.","inputSchema":{"type":"object","required":["entity_id"],"properties":{"entity_id":{"type":"integer"},"shading_mode":{"type":"string"},"cel_shadow_tint":{"type":"array"},"cel_outline_width_px":{"type":"number"},"cel_outline_color":{"type":"array"},"cel_rim_color":{"type":"array"},"cel_hair_halo":{"type":"number"},"vfx_erosion":{"type":"number"},"vfx_scroll_speed":{"type":"number"},"vfx_core_color":{"type":"array"},"vfx_edge_color":{"type":"array"},"anim_step_hz":{"type":"number"},"smear_stretch":{"type":"number"},"coalesce_key":{"type":"integer"}}}},'
        + '{"name":"render.trigger_post_fx","description":"Sets or clears Manga Impact Frame blend/invert, chromatic-dispersion shockwave ring, and limited-animation SteppedAnimClock hitstop freeze on Studio.","inputSchema":{"type":"object","properties":{"impact_blend":{"type":"number"},"impact_invert":{"type":"boolean"},"shockwave_active":{"type":"boolean"},"shockwave_radius":{"type":"number"},"hitstop_sec":{"type":"number"},"clear":{"type":"boolean"}}}}'
        + "]}"
    )


# ============================================================================
# 3. Deterministic Software Viewport Preview Fallback (for Headless CPU Capture)
# ============================================================================

@always_inline
def _clamp_color_u8(v: Float32) -> Int:
    if not isfinite(v) or v <= Float32(0.0):
        return 0
    if v >= Float32(1.0):
        return 255
    return max(0, min(255, Int(v * Float32(255.0))))


def _render_studio_software_preview(studio: FlintStudioState, width: Int, height: Int) -> SimdCanvas:
    var w = max(64, min(2048, width))
    var h = max(64, min(2048, height))
    var canvas = SimdCanvas(w, h)
    canvas.clear_gradient_simd(0x111B2B, 0x24364F)

    var cam = Camera3D(
        screen_w=w,
        screen_h=h,
        eye=studio.cam_pos,
        target=studio.cam_pos + studio.cam_fwd * Float32(10.0),
        fov_deg=studio.fov_y_deg,
    )

    # Ground reference grid
    for g in range(-8, 9):
        var gf = Float32(g) * Float32(1.5)
        var col = UInt32(0xFF394E6A) if g == 0 else UInt32(0xFF28384E)
        canvas.draw_line_3d(cam, Vec3(gf, studio.ground_plane_y, -12.0), Vec3(gf, studio.ground_plane_y, 12.0), col, 1)
        canvas.draw_line_3d(cam, Vec3(-12.0, studio.ground_plane_y, gf), Vec3(12.0, studio.ground_plane_y, gf), col, 1)

    # Render oriented 3D bounding boxes for all live visible entities
    var corners = [
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
        Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1),
    ]
    var tris = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    for i in range(len(studio.entities)):
        ref ent = studio.entities[i]
        if not ent.alive or not ent.visible:
            continue
        var he = Vec3(
            max(Float32(0.15), ent.collider_half_extents.x * ent.scale.x),
            max(Float32(0.15), ent.collider_half_extents.y * ent.scale.y),
            max(Float32(0.15), ent.collider_half_extents.z * ent.scale.z),
        )
        var wc = List[Vec3](capacity=8)
        for ci in range(8):
            var lc = Vec3(corners[ci].x * he.x, corners[ci].y * he.y, corners[ci].z * he.z)
            wc.append(ent.translation + ent.rotation.rotate(lc))
        var col = pack_argb(
            _clamp_color_u8(ent.base_color.x),
            _clamp_color_u8(ent.base_color.y),
            _clamp_color_u8(ent.base_color.z),
        )
        for ti in range(12):
            var t = tris[ti]
            canvas.draw_tri_3d_lit(cam, wc[t[0]], wc[t[1]], wc[t[2]], col)

    # Draw 3D gizmo axes on selected entity
    if studio.selected_entity_id >= 0 and studio.selected_entity_id < len(studio.entities) and studio.entities[studio.selected_entity_id].alive:
        var p = studio.entities[studio.selected_entity_id].translation
        canvas.draw_line_3d(cam, p, p + Vec3(1.5, 0.0, 0.0), 0xFFE84848, 2)
        canvas.draw_line_3d(cam, p, p + Vec3(0.0, 1.5, 0.0), 0xFF3DDC68, 2)
        canvas.draw_line_3d(cam, p, p + Vec3(0.0, 0.0, 1.5), 0xFF428BF5, 2)

    return canvas^


# ============================================================================
# 4. `McpServerV2`: Stateful JSON-RPC 2.0 Dispatcher
# ============================================================================

struct McpServerV2(Movable):
    var initialized: Bool
    var requests_handled: Int
    var tool_calls_handled: Int
    var client_name: String
    var workspace_root: String
    var auth_token: String
    var session_authenticated: Bool

    def __init__(out self, workspace_root: String = ".", auth_token: String = ""):
        self.initialized = False
        self.requests_handled = 0
        self.tool_calls_handled = 0
        self.client_name = String("unknown-mcp-client")
        self.workspace_root = workspace_root if workspace_root.byte_length() > 0 else String(".")
        self.auth_token = auth_token
        self.session_authenticated = auth_token.byte_length() == 0

    @staticmethod
    def _extract_auth_token(doc: JsonDocument) -> String:
        var p = doc.object_get(doc.root, "params")
        if p >= 0 and doc.nodes[p].kind == 5:
            var t0 = doc.text(doc.object_get(p, "auth_token"), "")
            if t0.byte_length() > 0:
                return t0
            var t1 = doc.text(doc.object_get(p, "authToken"), "")
            if t1.byte_length() > 0:
                return t1
        var r0 = doc.text(doc.object_get(doc.root, "auth_token"), "")
        if r0.byte_length() > 0:
            return r0
        return doc.text(doc.object_get(doc.root, "authToken"), "")

    def _dispatch_tool_call(
        mut self,
        tool_name: String,
        doc: JsonDocument,
        args_node: Int,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
        has_renderer: Bool,
        ren_ptr: Int,
    ) -> Tuple[Bool, String]:
        self.tool_calls_handled += 1

        # 1. ecs.query_entities
        if tool_name == "ecs.query_entities":
            var sub_filter = doc.text(doc.object_get(args_node, "name_contains"), "")
            var col_filter = doc.text(doc.object_get(args_node, "collider"), "")
            var vis_only = doc.boolean(doc.object_get(args_node, "visible_only"), False)

            var items = String("[")
            var matched = 0
            for i in range(len(studio.entities)):
                ref e = studio.entities[i]
                if not e.alive:
                    continue
                if vis_only and not e.visible:
                    continue
                if sub_filter.byte_length() > 0 and e.name.find(sub_filter) < 0:
                    continue
                if col_filter.byte_length() > 0 and _collider_to_string(e.collider_kind) != col_filter:
                    continue
                if matched > 0:
                    items += ","
                items += (
                    '{"entity_id":' + String(e.entity_id)
                    + ',"name":' + json_quote(e.name)
                    + ',"parent_id":' + String(e.parent_id)
                    + ',"position":' + _fmt_vec3_json(e.translation)
                    + ',"rotation":' + _fmt_quat_json(e.rotation)
                    + ',"scale":' + _fmt_vec3_json(e.scale)
                    + ',"collider":' + json_quote(_collider_to_string(e.collider_kind))
                    + ',"shading_mode":' + json_quote(_shading_mode_to_string(e.shading_mode))
                    + ',"mass":' + _fmt_json_f64(Float64(e.mass))
                    + ',"visible":' + ("true" if e.visible else "false")
                    + "}"
                )
                matched += 1
            items += "]"
            return (
                False,
                '{"ok":true,"count":' + String(matched)
                + ',"selected_entity_id":' + String(studio.selected_entity_id)
                + ',"state_hash":' + json_quote(String(studio.state_hash()))
                + ',"entities":' + items + "}",
            )

        # 2. ecs.spawn_entity
        if tool_name == "ecs.spawn_entity":
            var name = doc.text(doc.object_get(args_node, "name"), "SpawnedEntity")
            var p_res = _parse_vec3_arg(doc, doc.object_get(args_node, "position"), Vec3(0.0, 0.0, 0.0))
            var r_res = _parse_quat_arg(doc, doc.object_get(args_node, "rotation"), Quat(0.0, 0.0, 0.0, 1.0))
            var s_res = _parse_vec3_arg(doc, doc.object_get(args_node, "scale"), Vec3(1.0, 1.0, 1.0))
            var bc_res = _parse_vec3_arg(doc, doc.object_get(args_node, "base_color"), Vec3(0.82, 0.84, 0.88))
            if not (p_res[0] and r_res[0] and s_res[0] and bc_res[0]):
                return (True, '{"ok":false,"error":"invalid Vec3/Quat argument"}')
            var col_k = _collider_from_string(doc.text(doc.object_get(args_node, "collider"), "box"))
            var m_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "mass"), Float32(0.0))
            var rg_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "roughness"), Float32(0.38))
            var mt_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "metallic"), Float32(0.10))
            if not (m_res[0] and rg_res[0] and mt_res[0]):
                return (True, '{"ok":false,"error":"invalid non-finite numeric argument"}')
            var mass = max(Float32(0.0), m_res[1])
            var rough = rg_res[1]
            var metal = mt_res[1]

            var eid = studio.spawn_entity(
                name=name,
                position=p_res[1],
                rotation=r_res[1],
                scale=s_res[1],
                collider_kind=col_k,
                mass=mass,
                base_color=bc_res[1],
                roughness=rough,
                metallic=metal,
                record_undo=True,
            )
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"name":' + json_quote(name)
                + ',"undo_depth":' + String(len(studio.undo_stack))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 3. ecs.patch_components
        if tool_name == "ecs.patch_components":
            var eid = doc.integer(doc.object_get(args_node, "entity_id"), -1)
            if eid < 0 or eid >= len(studio.entities) or not studio.entities[eid].alive:
                return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')
            var nd = studio.entities[eid].copy()
            var n_node = doc.object_get(args_node, "name")
            if n_node >= 0 and doc.nodes[n_node].kind == 3:
                nd.name = doc.text(n_node)
            var p_node = doc.object_get(args_node, "position")
            if p_node >= 0:
                var pr = _parse_vec3_arg(doc, p_node, nd.translation)
                if not pr[0]:
                    return (True, '{"ok":false,"error":"invalid position"}')
                nd.translation = pr[1]
            var r_node = doc.object_get(args_node, "rotation")
            if r_node >= 0:
                var rr = _parse_quat_arg(doc, r_node, nd.rotation)
                if not rr[0]:
                    return (True, '{"ok":false,"error":"invalid rotation"}')
                nd.rotation = rr[1]
            var s_node = doc.object_get(args_node, "scale")
            if s_node >= 0:
                var sr = _parse_vec3_arg(doc, s_node, nd.scale)
                if not sr[0]:
                    return (True, '{"ok":false,"error":"invalid scale"}')
                nd.scale = sr[1]
            var bc_node = doc.object_get(args_node, "base_color")
            if bc_node >= 0:
                var br = _parse_vec3_arg(doc, bc_node, nd.base_color)
                if not br[0]:
                    return (True, '{"ok":false,"error":"invalid base_color"}')
                nd.base_color = br[1]
            var rg_node = doc.object_get(args_node, "roughness")
            if rg_node >= 0:
                var rg_r = _parse_finite_f32_arg(doc, rg_node, nd.roughness)
                if not rg_r[0]:
                    return (True, '{"ok":false,"error":"invalid roughness"}')
                nd.roughness = max(Float32(0.04), min(Float32(1.0), rg_r[1]))
            var mt_node = doc.object_get(args_node, "metallic")
            if mt_node >= 0:
                var mt_r = _parse_finite_f32_arg(doc, mt_node, nd.metallic)
                if not mt_r[0]:
                    return (True, '{"ok":false,"error":"invalid metallic"}')
                nd.metallic = max(Float32(0.0), min(Float32(1.0), mt_r[1]))
            var ms_node = doc.object_get(args_node, "mass")
            if ms_node >= 0:
                var ms_r = _parse_finite_f32_arg(doc, ms_node, nd.mass)
                if not ms_r[0]:
                    return (True, '{"ok":false,"error":"invalid mass"}')
                nd.mass = max(Float32(0.0), ms_r[1])
            var vis_node = doc.object_get(args_node, "visible")
            if vis_node >= 0 and doc.nodes[vis_node].kind == 1:
                nd.visible = doc.boolean(vis_node)
            var ckey = UInt64(max(0, doc.integer(doc.object_get(args_node, "coalesce_key"), 0)))

            _ = studio.commit_node_edit(eid, nd^, "MCP Patch", ckey)
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"undo_depth":' + String(len(studio.undo_stack))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 4. ecs.despawn_entity
        if tool_name == "ecs.despawn_entity":
            var eid = doc.integer(doc.object_get(args_node, "entity_id"), -1)
            if not studio.despawn_entity(eid, record_undo=True):
                return (True, '{"ok":false,"error":"entity not found or already despawned"}')
            return (
                False,
                '{"ok":true,"despawned":true,"entity_id":' + String(eid)
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 5. render.capture_viewport (with `flip.mojo` perceptual error diff & workspace sandbox!)
        if tool_name == "render.capture_viewport":
            var raw_out_path = doc.text(doc.object_get(args_node, "output_path"), "build/mcp_capture.png")
            var raw_ref_path = doc.text(doc.object_get(args_node, "reference_path"), "")
            var req_w = max(64, min(8192, doc.integer(doc.object_get(args_node, "width"), 320)))
            var req_h = max(64, min(8192, doc.integer(doc.object_get(args_node, "height"), 180)))

            var out_path: String
            try:
                out_path = sanitize_output_path(self.workspace_root, raw_out_path)
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected output_path: " + String(e)) + "}")

            var ref_path = String("")
            if raw_ref_path.byte_length() > 0:
                try:
                    ref_path = sanitize_output_path(self.workspace_root, raw_ref_path)
                except e:
                    return (True, '{"ok":false,"error":' + json_quote("sandbox rejected reference_path: " + String(e)) + "}")

            var out_parent = parent_directory(out_path)
            if out_parent.byte_length() > 0 and out_parent != ".":
                _ = make_dirs(out_parent)

            var cap_w = req_w
            var cap_h = req_h
            var pixels: List[UInt32]

            if has_renderer and ren_ptr != 0:
                ref ren = Pointer[FlintAaaVulkanRenderer, MutAnyOrigin](unsafe_from_address=ren_ptr)[unsafe_offset=0]
                if ren.active:
                    ren.flush_deferred_frame()
                    cap_w = ren.output_w
                    cap_h = ren.output_h
                    pixels = ren.readback_pixels()
                else:
                    var sw_c = _render_studio_software_preview(studio, req_w, req_h)
                    cap_w = sw_c.width
                    cap_h = sw_c.height
                    pixels = sw_c.pixels.copy()
            else:
                var sw_c = _render_studio_software_preview(studio, req_w, req_h)
                cap_w = sw_c.width
                cap_h = sw_c.height
                pixels = sw_c.pixels.copy()

            var wrote = write_png(out_path, pixels, cap_w, cap_h)
            if not wrote:
                return (True, '{"ok":false,"error":"failed to write viewport PNG"}')

            var flip_val = Float64(-1.0)
            var has_flip = False
            if ref_path.byte_length() > 0:
                var ref_img = decode_png(read_bytes(ref_path), 4096)
                if ref_img.valid and ref_img.width == cap_w and ref_img.height == cap_h:
                    flip_val = flip_mean(flip_error_map(pixels, ref_img.pixels, cap_w, cap_h))
                    has_flip = True

            return (
                False,
                '{"ok":true,"output_path":' + json_quote(out_path)
                + ',"width":' + String(cap_w)
                + ',"height":' + String(cap_h)
                + ',"has_flip_diff":' + ("true" if has_flip else "false")
                + ',"flip_mean":' + _fmt_json_f64(max(Float64(0.0), flip_val)) + "}",
            )

        # 6. render.inspect_gpu_frame
        if tool_name == "render.inspect_gpu_frame":
            var cpu_bytes = mem_total(W_CPU_LIVE)
            var vram_bytes = mem_total(W_VRAM_LIVE)
            if has_renderer and ren_ptr != 0:
                ref ren = Pointer[FlintAaaVulkanRenderer, MutAnyOrigin](unsafe_from_address=ren_ptr)[unsafe_offset=0]
                var passes = String("[")
                for pi in range(len(ren.render_graph.pass_timings)):
                    if pi > 0:
                        passes += ","
                    ref pt = ren.render_graph.pass_timings[pi]
                    passes += '{"name":' + json_quote(pt.pass_name) + ',"gpu_ms":' + _fmt_json_f64(Float64(pt.gpu_ms)) + "}"
                passes += "]"
                return (
                    False,
                    '{"ok":true,"backend":"VULKAN-AAA","frame_index":' + String(ren.frame_index)
                    + ',"total_gpu_ms":' + _fmt_json_f64(Float64(ren.render_graph.total_gpu_ms))
                    + ',"aliasing_savings_ratio":' + _fmt_json_f64(Float64(ren.render_graph.aliasing_savings_ratio()))
                    + ',"culled_meshlet_draws":' + String(ren.last_culled_meshlet_draws)
                    + ',"cpu_live_bytes":' + String(cpu_bytes)
                    + ',"vram_live_bytes":' + String(vram_bytes)
                    + ',"passes":' + passes + "}",
                )
            return (
                False,
                '{"ok":true,"backend":"HEADLESS-STUDIO","build":' + json_quote(build_stamp())
                + ',"sim_tick":' + String(studio.sim_tick)
                + ',"streamed_bytes_total":' + String(assets.streamer.streamed_bytes_total)
                + ',"cpu_live_bytes":' + String(cpu_bytes)
                + ',"vram_live_bytes":' + String(vram_bytes) + "}",
            )

        # 7. assets.list_and_cook (strictly sandboxed inside `workspace_root`)
        if tool_name == "assets.list_and_cook":
            var raw_src_dir = doc.text(doc.object_get(args_node, "source_dir"), "")
            var raw_out_flpk = doc.text(doc.object_get(args_node, "output_flpk"), "")
            var raw_c_dir = doc.text(doc.object_get(args_node, "cache_dir"), DEFAULT_COOK_CACHE_DIR)
            var force_c = doc.boolean(doc.object_get(args_node, "force_recook"), False)

            var cook_json = String("null")
            if raw_out_flpk.byte_length() > 0 or raw_src_dir.byte_length() > 0:
                if raw_src_dir.byte_length() == 0 or raw_out_flpk.byte_length() == 0:
                    return (True, '{"ok":false,"error":"both source_dir and output_flpk are required when cooking"}')
                var src_dir: String
                var out_flpk: String
                var c_dir: String
                try:
                    src_dir = _sanitize_workspace_path(self.workspace_root, raw_src_dir, allow_root_dir=True)
                    out_flpk = sanitize_output_path(self.workspace_root, raw_out_flpk)
                    c_dir = _sanitize_workspace_path(self.workspace_root, raw_c_dir, allow_root_dir=False)
                except e:
                    return (True, '{"ok":false,"error":' + json_quote("sandbox rejected cook path: " + String(e)) + "}")

                var rep = cook_workspace(src_dir, out_flpk, pool, cache_dir=c_dir, force_recook=force_c)
                if rep.ok:
                    _ = assets.mount_flpk(out_flpk)
                cook_json = (
                    '{"ok":' + ("true" if rep.ok else "false")
                    + ',"total_assets":' + String(rep.total_assets)
                    + ',"cooked_count":' + String(rep.cooked_count)
                    + ',"cache_hits":' + String(rep.cache_hits)
                    + ',"archive_bytes":' + String(rep.archive_bytes)
                    + ',"elapsed_ms":' + _fmt_json_f64(Float64(rep.elapsed_ms)) + "}"
                )

            return (
                False,
                '{"ok":true,"mounted_archives":' + String(len(assets.mounts))
                + ',"live_meshes":' + String(assets.meshes.live_count)
                + ',"live_textures":' + String(assets.textures.live_count)
                + ',"resident_bytes":' + String(assets.total_resident_bytes())
                + ',"hot_reloads":' + String(assets.hot_reloads_applied)
                + ',"cook_report":' + cook_json + "}",
            )

        # 8. procedural.generate_asset (strictly sandboxed inside `workspace_root`)
        if tool_name == "procedural.generate_asset":
            var preset = doc.text(doc.object_get(args_node, "preset"), "material_orb")
            var raw_out_glb = doc.text(doc.object_get(args_node, "output_glb_path"), "")
            var seed = max(1, doc.integer(doc.object_get(args_node, "seed"), 1))
            var do_spawn = doc.boolean(doc.object_get(args_node, "spawn_in_scene"), False)
            var pos_r = _parse_vec3_arg(doc, doc.object_get(args_node, "position"), Vec3(0.0, 0.0, 0.0))
            if not pos_r[0]:
                return (True, '{"ok":false,"error":"invalid position"}')

            var out_glb = String("")
            if raw_out_glb.byte_length() > 0:
                try:
                    out_glb = sanitize_output_path(self.workspace_root, raw_out_glb)
                except e:
                    return (True, '{"ok":false,"error":' + json_quote("sandbox rejected output_glb_path: " + String(e)) + "}")
                var pdir = parent_directory(out_glb)
                if pdir.byte_length() > 0 and pdir != ".":
                    _ = make_dirs(pdir)

            var baked = bake_procedural_to_glb(preset, out_glb, seed)
            if not baked[0]:
                return (True, '{"ok":false,"error":"failed to bake procedural GLB"}')

            var spawned_id = -1
            if do_spawn:
                spawned_id = studio.spawn_entity("Proc_" + preset, position=pos_r[1], record_undo=True)

            return (
                False,
                '{"ok":true,"preset":' + json_quote(preset)
                + ',"vertex_count":' + String(baked[1])
                + ',"triangle_count":' + String(baked[2])
                + ',"glb_bytes":' + String(baked[3])
                + ',"output_glb_path":' + json_quote(out_glb)
                + ',"spawned_entity_id":' + String(spawned_id) + "}",
            )

        # 9. physics.step_deterministic (optional checkpoint output strictly sandboxed inside `workspace_root`)
        if tool_name == "physics.step_deterministic":
            var ticks = max(1, min(3600, doc.integer(doc.object_get(args_node, "ticks"), 1)))
            var dt_raw = doc.number(doc.object_get(args_node, "dt"), 1.0 / 60.0)
            if not isfinite(dt_raw):
                return (True, '{"ok":false,"error":"invalid non-finite dt"}')
            var dt = Float32(max(0.0005, min(0.1, dt_raw)))
            var raw_out_cp = doc.text(doc.object_get(args_node, "output_path"), "")
            var safe_out_cp = String("")
            if raw_out_cp.byte_length() > 0:
                try:
                    safe_out_cp = sanitize_output_path(self.workspace_root, raw_out_cp)
                except e:
                    return (True, '{"ok":false,"error":' + json_quote("sandbox rejected output_path: " + String(e)) + "}")

            var done = studio.step_physics_deterministic(ticks, dt)
            if safe_out_cp.byte_length() > 0:
                var pdir = parent_directory(safe_out_cp)
                if pdir.byte_length() > 0 and pdir != ".":
                    _ = make_dirs(pdir)
                var w = BlobWriter()
                studio.serialize(w)
                var cp_bytes = pack_checkpoint_lz4(w)
                if not write_bytes(safe_out_cp, cp_bytes):
                    return (True, '{"ok":false,"error":"failed to write physics checkpoint"}')

            return (
                False,
                '{"ok":true,"ticks_advanced":' + String(done)
                + ',"sim_tick":' + String(studio.sim_tick)
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 10. physics.raycast_query
        if tool_name == "physics.raycast_query":
            var o_res = _parse_vec3_arg(doc, doc.object_get(args_node, "origin"), Vec3(0.0, 5.0, 0.0))
            var d_res = _parse_vec3_arg(doc, doc.object_get(args_node, "direction"), Vec3(0.0, -1.0, 0.0))
            if not (o_res[0] and d_res[0]):
                return (True, '{"ok":false,"error":"origin and direction must be [x,y,z]"}')
            var md_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "max_distance"), Float32(1000.0))
            if not md_res[0]:
                return (True, '{"ok":false,"error":"invalid max_distance"}')
            var max_d = max(Float32(0.01), md_res[1])
            var hit = studio.raycast_scene(o_res[1], d_res[1], max_d)
            var ename = String("")
            if hit.hit and hit.entity_id >= 0 and hit.entity_id < len(studio.entities):
                ename = studio.entities[hit.entity_id].name
            elif hit.hit and hit.entity_id == -2:
                ename = "__ground_plane__"
            return (
                False,
                '{"ok":true,"hit":' + ("true" if hit.hit else "false")
                + ',"entity_id":' + String(hit.entity_id)
                + ',"entity_name":' + json_quote(ename)
                + ',"distance":' + _fmt_json_f64(Float64(hit.distance if hit.hit else Float32(0.0)))
                + ',"point":' + _fmt_vec3_json(hit.point)
                + ',"normal":' + _fmt_vec3_json(hit.normal) + "}",
            )

        # 11. studio.pie_control
        if tool_name == "studio.pie_control":
            var act = doc.text(doc.object_get(args_node, "action"), "play")
            var ticks = max(1, min(3600, doc.integer(doc.object_get(args_node, "ticks"), 1)))
            var restored_exact = False
            if act == "play":
                if not studio.pie_play():
                    return (True, '{"ok":false,"error":"pie_play failed"}')
            elif act == "pause":
                _ = studio.pie_pause()
            elif act == "step":
                _ = studio.pie_step(ticks)
            elif act == "stop":
                try:
                    restored_exact = studio.pie_stop()
                except:
                    return (True, '{"ok":false,"error":"pie_stop LZ4 verification failed"}')
            else:
                return (True, '{"ok":false,"error":"unknown PIE action"}')

            var m_str = "playing" if studio.pie.mode == PIE_PLAYING else ("paused" if studio.pie.mode == PIE_PAUSED else "stopped")
            return (
                False,
                '{"ok":true,"pie_mode":' + json_quote(m_str)
                + ',"ticks_simulated":' + String(studio.pie.ticks_simulated)
                + ',"restored_exact":' + ("true" if restored_exact else "false")
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 12. studio.undo_redo
        if tool_name == "studio.undo_redo":
            var act = doc.text(doc.object_get(args_node, "action"), "undo")
            var steps = max(1, min(1024, doc.integer(doc.object_get(args_node, "steps"), 1)))
            var applied = 0
            if act == "undo":
                for _ in range(steps):
                    if studio.undo():
                        applied += 1
                    else:
                        break
            elif act == "redo":
                for _ in range(steps):
                    if studio.redo():
                        applied += 1
                    else:
                        break
            else:
                return (True, '{"ok":false,"error":"action must be undo or redo"}')
            return (
                False,
                '{"ok":true,"action":' + json_quote(act)
                + ',"applied_steps":' + String(applied)
                + ',"undo_depth":' + String(len(studio.undo_stack))
                + ',"redo_depth":' + String(len(studio.redo_stack))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 13. render.configure_npr_cel
        if tool_name == "render.configure_npr_cel":
            var eid = doc.integer(doc.object_get(args_node, "entity_id"), -1)
            if eid < 0 or eid >= len(studio.entities) or not studio.entities[eid].alive:
                return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')
            var nd = studio.entities[eid].copy()

            var mode_node = doc.object_get(args_node, "shading_mode")
            if mode_node >= 0:
                if doc.nodes[mode_node].kind == 3:
                    var m_parsed = _shading_mode_from_string(doc.text(mode_node))
                    if not m_parsed[0]:
                        return (True, '{"ok":false,"error":"shading_mode must be pbr, npr_cel, or stylized_vfx"}')
                    nd.shading_mode = m_parsed[1]
                elif doc.nodes[mode_node].kind == 2:
                    var mi = doc.integer(mode_node, -1)
                    if mi < SHADING_PBR or mi > SHADING_STYLIZED_VFX:
                        return (True, '{"ok":false,"error":"shading_mode integer must be 0, 1, or 2"}')
                    nd.shading_mode = mi
                else:
                    return (True, '{"ok":false,"error":"invalid shading_mode type"}')
            elif nd.shading_mode == SHADING_PBR:
                var has_vfx_arg = (
                    doc.object_get(args_node, "vfx_erosion") >= 0
                    or doc.object_get(args_node, "vfx_scroll_speed") >= 0
                    or doc.object_get(args_node, "vfx_core_color") >= 0
                    or doc.object_get(args_node, "vfx_edge_color") >= 0
                )
                var has_cel_arg = (
                    doc.object_get(args_node, "cel_shadow_tint") >= 0
                    or doc.object_get(args_node, "cel_outline_width_px") >= 0
                    or doc.object_get(args_node, "cel_outline_color") >= 0
                    or doc.object_get(args_node, "cel_rim_color") >= 0
                    or doc.object_get(args_node, "cel_hair_halo") >= 0
                )
                nd.shading_mode = SHADING_STYLIZED_VFX if (has_vfx_arg and not has_cel_arg) else SHADING_NPR_CEL

            var cst_node = doc.object_get(args_node, "cel_shadow_tint")
            if cst_node >= 0:
                var r_cst = _parse_vec3_arg(doc, cst_node, nd.cel_shadow_tint)
                if not r_cst[0]:
                    return (True, '{"ok":false,"error":"invalid cel_shadow_tint"}')
                nd.cel_shadow_tint = r_cst[1]

            var cow_node = doc.object_get(args_node, "cel_outline_width_px")
            if cow_node >= 0:
                var cow_r = _parse_finite_f32_arg(doc, cow_node, nd.cel_outline_width_px)
                if not cow_r[0]:
                    return (True, '{"ok":false,"error":"invalid cel_outline_width_px"}')
                nd.cel_outline_width_px = max(Float32(0.0), min(Float32(24.0), cow_r[1]))

            var coc_node = doc.object_get(args_node, "cel_outline_color")
            if coc_node >= 0:
                var r_coc = _parse_vec3_arg(doc, coc_node, nd.cel_outline_color)
                if not r_coc[0]:
                    return (True, '{"ok":false,"error":"invalid cel_outline_color"}')
                nd.cel_outline_color = r_coc[1]

            var crc_node = doc.object_get(args_node, "cel_rim_color")
            if crc_node >= 0:
                var r_crc = _parse_vec3_arg(doc, crc_node, nd.cel_rim_color)
                if not r_crc[0]:
                    return (True, '{"ok":false,"error":"invalid cel_rim_color"}')
                nd.cel_rim_color = r_crc[1]

            var chh_node = doc.object_get(args_node, "cel_hair_halo")
            if chh_node >= 0:
                var chh_r = _parse_finite_f32_arg(doc, chh_node, nd.cel_hair_halo)
                if not chh_r[0]:
                    return (True, '{"ok":false,"error":"invalid cel_hair_halo"}')
                nd.cel_hair_halo = max(Float32(0.0), chh_r[1])

            var ver_node = doc.object_get(args_node, "vfx_erosion")
            if ver_node >= 0:
                var ver_r = _parse_finite_f32_arg(doc, ver_node, nd.vfx_erosion)
                if not ver_r[0]:
                    return (True, '{"ok":false,"error":"invalid vfx_erosion"}')
                nd.vfx_erosion = max(Float32(0.0), min(Float32(1.0), ver_r[1]))

            var vss_node = doc.object_get(args_node, "vfx_scroll_speed")
            if vss_node >= 0:
                var vss_r = _parse_finite_f32_arg(doc, vss_node, nd.vfx_scroll_speed)
                if not vss_r[0]:
                    return (True, '{"ok":false,"error":"invalid vfx_scroll_speed"}')
                nd.vfx_scroll_speed = vss_r[1]

            var vcc_node = doc.object_get(args_node, "vfx_core_color")
            if vcc_node >= 0:
                var r_vcc = _parse_vec3_arg(doc, vcc_node, nd.vfx_core_color)
                if not r_vcc[0]:
                    return (True, '{"ok":false,"error":"invalid vfx_core_color"}')
                nd.vfx_core_color = r_vcc[1]

            var vec_node = doc.object_get(args_node, "vfx_edge_color")
            if vec_node >= 0:
                var r_vec = _parse_vec3_arg(doc, vec_node, nd.vfx_edge_color)
                if not r_vec[0]:
                    return (True, '{"ok":false,"error":"invalid vfx_edge_color"}')
                nd.vfx_edge_color = r_vec[1]

            var ash_node = doc.object_get(args_node, "anim_step_hz")
            if ash_node >= 0:
                var ash_r = _parse_finite_f32_arg(doc, ash_node, nd.anim_step_hz)
                if not ash_r[0]:
                    return (True, '{"ok":false,"error":"invalid anim_step_hz"}')
                nd.anim_step_hz = max(Float32(0.0), ash_r[1])

            var sms_node = doc.object_get(args_node, "smear_stretch")
            if sms_node >= 0:
                var sms_r = _parse_finite_f32_arg(doc, sms_node, nd.smear_stretch)
                if not sms_r[0]:
                    return (True, '{"ok":false,"error":"invalid smear_stretch"}')
                nd.smear_stretch = max(Float32(0.55), min(Float32(3.2), sms_r[1]))

            var ckey = UInt64(max(0, doc.integer(doc.object_get(args_node, "coalesce_key"), 0)))
            _ = studio.commit_node_edit(eid, nd^, "MCP Configure NPR/VFX", ckey)
            ref ent_after = studio.entities[eid]
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"shading_mode":' + json_quote(_shading_mode_to_string(ent_after.shading_mode))
                + ',"shading_mode_id":' + String(ent_after.shading_mode)
                + ',"cel_outline_width_px":' + _fmt_json_f64(Float64(ent_after.cel_outline_width_px))
                + ',"vfx_erosion":' + _fmt_json_f64(Float64(ent_after.vfx_erosion))
                + ',"anim_step_hz":' + _fmt_json_f64(Float64(ent_after.anim_step_hz))
                + ',"smear_stretch":' + _fmt_json_f64(Float64(ent_after.smear_stretch))
                + ',"undo_depth":' + String(len(studio.undo_stack))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 14. render.trigger_post_fx
        if tool_name == "render.trigger_post_fx":
            if doc.boolean(doc.object_get(args_node, "clear"), False):
                studio.preview_impact_blend = Float32(0.0)
                studio.preview_impact_invert = False
                studio.preview_shockwave_active = False
                studio.preview_shockwave_radius = Float32(0.22)
                studio.stepped_clock.hitstop_remaining_sec = Float32(0.0)

            var ib_node = doc.object_get(args_node, "impact_blend")
            if ib_node >= 0:
                var ib_r = _parse_finite_f32_arg(doc, ib_node, studio.preview_impact_blend)
                if not ib_r[0]:
                    return (True, '{"ok":false,"error":"invalid impact_blend"}')
                studio.preview_impact_blend = max(Float32(0.0), min(Float32(1.0), ib_r[1]))

            var ii_node = doc.object_get(args_node, "impact_invert")
            if ii_node >= 0 and doc.nodes[ii_node].kind == 1:
                studio.preview_impact_invert = doc.boolean(ii_node)

            var sa_node = doc.object_get(args_node, "shockwave_active")
            if sa_node >= 0 and doc.nodes[sa_node].kind == 1:
                studio.preview_shockwave_active = doc.boolean(sa_node)

            var sr_node = doc.object_get(args_node, "shockwave_radius")
            if sr_node >= 0:
                var sr_r = _parse_finite_f32_arg(doc, sr_node, studio.preview_shockwave_radius)
                if not sr_r[0]:
                    return (True, '{"ok":false,"error":"invalid shockwave_radius"}')
                studio.preview_shockwave_radius = max(Float32(0.002), min(Float32(2.0), sr_r[1]))

            var hs_node = doc.object_get(args_node, "hitstop_sec")
            if hs_node >= 0:
                var hs_r = _parse_finite_f32_arg(doc, hs_node, Float32(0.0))
                if not hs_r[0]:
                    return (True, '{"ok":false,"error":"invalid hitstop_sec"}')
                var hs_val = hs_r[1]
                if hs_val <= Float32(0.0):
                    studio.stepped_clock.hitstop_remaining_sec = Float32(0.0)
                else:
                    studio.stepped_clock.trigger_hitstop(hs_val)

            if has_renderer and ren_ptr != 0:
                ref ren = Pointer[FlintAaaVulkanRenderer, MutAnyOrigin](unsafe_from_address=ren_ptr)[unsafe_offset=0]
                if studio.preview_impact_blend > Float32(0.0):
                    ren.set_impact_frame(
                        AaaImpactFrameParams(
                            blend=studio.preview_impact_blend,
                            invert_polarity=studio.preview_impact_invert,
                        )
                    )
                else:
                    ren.clear_impact_frame()
                if studio.preview_shockwave_active:
                    ren.submit_shockwave(
                        AaaShockwaveRing(
                            center_u=Float32(0.5),
                            center_v=Float32(0.5),
                            radius_uv=studio.preview_shockwave_radius,
                        )
                    )

            return (
                False,
                '{"ok":true,"impact_blend":' + _fmt_json_f64(Float64(studio.preview_impact_blend))
                + ',"impact_invert":' + ("true" if studio.preview_impact_invert else "false")
                + ',"shockwave_active":' + ("true" if studio.preview_shockwave_active else "false")
                + ',"shockwave_radius":' + _fmt_json_f64(Float64(studio.preview_shockwave_radius))
                + ',"hitstop_remaining_sec":' + _fmt_json_f64(Float64(studio.stepped_clock.hitstop_remaining_sec))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        return (True, '{"ok":false,"error":' + json_quote("unknown tool: " + tool_name) + "}")

    def _handle_request_bytes(
        mut self,
        raw: List[UInt8],
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
        has_renderer: Bool,
        ren_ptr: Int,
    ) -> String:
        self.requests_handled += 1
        var doc = parse_json(raw)
        if not doc.valid or doc.root < 0 or doc.nodes[doc.root].kind != 5:
            return _jsonrpc_err("null", JSONRPC_PARSE_ERROR, "Parse error: invalid JSON object")

        var jv = doc.text(doc.object_get(doc.root, "jsonrpc"), "")
        var method_node = doc.object_get(doc.root, "method")
        var id_node = doc.object_get(doc.root, "id")
        var has_id = id_node >= 0
        var id_raw = _json_raw_id(doc, id_node)

        if jv != "2.0" or method_node < 0 or doc.nodes[method_node].kind != 3:
            return _jsonrpc_err(id_raw, JSONRPC_INVALID_REQUEST, "Invalid JSON-RPC 2.0 request")

        var method = doc.text(method_node)

        # Enforce optional `auth_token` on `initialize` and all requests!
        if self.auth_token.byte_length() > 0:
            var supplied_tok = Self._extract_auth_token(doc)
            if method == "initialize":
                if not _const_time_str_eq(supplied_tok, self.auth_token):
                    if not has_id:
                        return String()
                    return _jsonrpc_err(id_raw, JSONRPC_INVALID_REQUEST, "Unauthorized: invalid or missing auth_token")
                self.session_authenticated = True
            else:
                if not self.session_authenticated:
                    if supplied_tok.byte_length() > 0 and _const_time_str_eq(supplied_tok, self.auth_token):
                        self.session_authenticated = True
                    else:
                        if not has_id:
                            return String()
                        return _jsonrpc_err(id_raw, JSONRPC_INVALID_REQUEST, "Unauthorized: session not authenticated")

        # Notifications (`id` omitted) produce no response frame per JSON-RPC 2.0 / MCP spec
        if not has_id:
            if method == "notifications/initialized":
                self.initialized = True
            return String()

        if method == "initialize":
            self.initialized = True
            var params_node = doc.object_get(doc.root, "params")
            if params_node >= 0:
                var ci = doc.object_get(params_node, "clientInfo")
                if ci >= 0:
                    self.client_name = doc.text(doc.object_get(ci, "name"), "mcp-client")
            var init_res = (
                '{"protocolVersion":' + json_quote(MCP_PROTOCOL_VERSION)
                + ',"capabilities":{"tools":{"listChanged":true}}'
                + ',"serverInfo":{"name":"flint-engine-mcp-v2","version":"5.0.0","build":'
                + json_quote(build_stamp()) + "}}"
            )
            return _jsonrpc_ok(id_raw, init_res)

        if method == "ping":
            return _jsonrpc_ok(id_raw, "{}")

        if method == "tools/list":
            return _jsonrpc_ok(id_raw, _mcp_tools_list_json())

        if method == "tools/call":
            var params_node = doc.object_get(doc.root, "params")
            if params_node < 0 or doc.nodes[params_node].kind != 5:
                return _jsonrpc_err(id_raw, JSONRPC_INVALID_PARAMS, "tools/call requires params object")
            var tname = doc.text(doc.object_get(params_node, "name"), "")
            if tname.byte_length() == 0:
                return _jsonrpc_err(id_raw, JSONRPC_INVALID_PARAMS, "tools/call missing tool name")
            var args_node = doc.object_get(params_node, "arguments")
            var outcome = self._dispatch_tool_call(
                tname, doc, args_node, studio, assets, pool, has_renderer, ren_ptr
            )
            return _jsonrpc_ok(id_raw, _mcp_tool_result(outcome[1], is_error=outcome[0]))

        return _jsonrpc_err(id_raw, JSONRPC_METHOD_NOT_FOUND, "Method not found: " + method)

    def handle_jsonrpc(
        mut self,
        req_json: String,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
    ) -> String:
        var raw = List[UInt8](capacity=req_json.byte_length())
        for b in req_json.as_bytes():
            raw.append(b)
        return self._handle_request_bytes(raw, studio, assets, pool, False, 0)

    def handle_jsonrpc_with_renderer(
        mut self,
        req_json: String,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
        mut ren: FlintAaaVulkanRenderer,
    ) -> String:
        var raw = List[UInt8](capacity=req_json.byte_length())
        for b in req_json.as_bytes():
            raw.append(b)
        return self._handle_request_bytes(raw, studio, assets, pool, True, Int(Pointer(to=ren)))

    def serve_stdio_loop(
        mut self,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
    ) -> Int:
        """Runs a blocking MCP `stdio` server reading bounded `Content-Length` or newline JSON-RPC from `stdin` (fd 0)."""
        var stream_buf = List[UInt8]()
        var scratch = List[UInt8](length=8192, fill=0)
        var answered = 0
        while True:
            var space = (MAX_MCP_MESSAGE_BYTES + 1) - len(stream_buf)
            var ask = min(8192, max(1, space))
            var n = Int(external_call["read", Int64](Int32(0), scratch.unsafe_ptr(), Int64(ask)))
            if n < 0:
                if last_errno() == _EINTR:
                    continue
                break
            if n == 0:
                break
            for i in range(n):
                stream_buf.append(scratch[i])

            var fatal_err = False
            while True:
                var frame = parse_next_mcp_frame(stream_buf)
                if frame.protocol_error:
                    var err_resp = _jsonrpc_err("null", frame.error_code, frame.error_message)
                    var err_bytes = encode_mcp_frame(err_resp, use_content_length=frame.used_content_length)
                    _ = send_all_bounded(1, Int(err_bytes.unsafe_ptr()), len(err_bytes), timeout_ms=2000, is_socket=False)
                    _ = len(err_bytes)
                    fatal_err = True
                    break
                if not frame.has_frame:
                    break
                var resp = self._handle_request_bytes(frame.payload, studio, assets, pool, False, 0)
                if resp.byte_length() > 0:
                    var out_bytes = encode_mcp_frame(resp, use_content_length=frame.used_content_length)
                    if not send_all_bounded(1, Int(out_bytes.unsafe_ptr()), len(out_bytes), timeout_ms=2000, is_socket=False):
                        _ = len(out_bytes)
                        fatal_err = True
                        break
                    _ = len(out_bytes)
                    answered += 1
            if fatal_err:
                break
        return answered


# ============================================================================
# 5. Persistent Non-Blocking Unix Domain Socket Server (`FlintMcpUnixListener`)
# ============================================================================

def _build_sockaddr_un(path: String) -> List[UInt8]:
    var addr = List[UInt8](length=110, fill=0)
    addr[0] = UInt8(1)
    addr[1] = UInt8(0)
    var pb = path.as_bytes()
    var lim = min(107, len(pb))
    for i in range(lim):
        addr[2 + i] = pb[i]
    return addr^


def _unlink_socket(path: String):
    if path.byte_length() == 0:
        return
    var owned = path
    _ = external_call["unlink", Int32](owned.as_c_string_span())


def _verify_peer_uid(cfd: Int, expected_uid_override: Int = -1) -> Bool:
    """Verifies `SO_PEERCRED` (`SOL_SOCKET=1, SO_PEERCRED=17`, 12-byte `struct ucred { pid, uid, gid }`)
    on accepted Linux `AF_UNIX` client `cfd`, rejecting any peer where `peer_uid != getuid()`."""
    var ucred = Array[UInt32, 3](fill=0xFFFFFFFF)
    var optlen = Array[UInt32, 1](fill=12)
    var rc = external_call["getsockopt", Int32](
        Int32(cfd),
        _SOL_SOCKET,
        _SO_PEERCRED,
        Pointer(to=ucred[0]),
        Pointer(to=optlen[0]),
    )
    if rc != 0 or optlen[0] != UInt32(12):
        return False
    var req_uid = UInt32(expected_uid_override) if expected_uid_override >= 0 else external_call["getuid", UInt32]()
    return ucred[1] == req_uid


struct FlintMcpUnixListener(Movable):
    """Hardened persistent non-blocking Unix Domain Socket MCP v2 listener:
    `0700` runtime dir + `0600` socket mode, `SO_PEERCRED` UID verification, per-client `4 MiB` rx caps,
    bad-header recovery, and `poll(POLLOUT)` EAGAIN-safe bounded sends."""
    var socket_path: String
    var listen_fd: Int
    var active: Bool
    var expected_peer_uid: Int
    var rejected_peers: Int
    var client_fds: List[Int]
    var client_rx_bufs: List[List[UInt8]]
    var client_authed: List[Bool]
    var _io_scratch: List[UInt8]

    def __init__(out self, socket_path: String = ""):
        self.socket_path = String()
        self.listen_fd = -1
        self.active = False
        self.expected_peer_uid = -1
        self.rejected_peers = 0
        self.client_fds = List[Int]()
        self.client_rx_bufs = List[List[UInt8]]()
        self.client_authed = List[Bool]()
        self._io_scratch = List[UInt8](length=8192, fill=0)
        var effective_path = socket_path if socket_path.byte_length() > 0 else default_mcp_socket_path()
        _ = self.start(effective_path)

    def __deinit__(deinit self):
        for i in range(len(self.client_fds)):
            close_fd(self.client_fds[i])
        if self.listen_fd >= 0:
            close_fd(self.listen_fd)
        if self.active and self.socket_path.byte_length() > 0:
            _unlink_socket(self.socket_path)

    def start(mut self, socket_path: String = "") -> Bool:
        self.close()
        var target_sock = socket_path if socket_path.byte_length() > 0 else default_mcp_socket_path()
        if target_sock.byte_length() == 0 or target_sock.byte_length() > 107 or _has_forbidden_path_bytes(target_sock):
            return False

        var pdir = parent_directory(target_sock)
        if pdir.byte_length() > 0 and pdir != ".":
            _ = make_dirs(pdir)
            if pdir.startswith("/tmp/flint-mcp-"):
                var uid = external_call["getuid", UInt32]()
                if not _ensure_private_dir_0700(pdir, uid):
                    return False

        # Refuse to unlink or bind over an existing symlink
        var existing = _lstat_mode_uid(target_sock)
        if existing[0] and (existing[1] & _S_IFMT) == _S_IFLNK:
            return False

        self.socket_path = target_sock
        _unlink_socket(target_sock)
        var fd = Int(external_call["socket", Int32](_AF_UNIX, _SOCK_STREAM | _SOCK_NONBLOCK | _SOCK_CLOEXEC, Int32(0)))
        if fd < 0:
            return False

        var addr = _build_sockaddr_un(target_sock)
        var old_umask = external_call["umask", UInt32](UInt32(0o077))
        var brc = external_call["bind", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(110))
        _ = external_call["umask", UInt32](old_umask)
        _ = len(addr)
        if brc != 0:
            close_fd(fd)
            return False

        var owned_sock = target_sock
        if external_call["chmod", Int32](owned_sock.as_c_string_span(), UInt32(0o600)) != 0:
            close_fd(fd)
            _unlink_socket(target_sock)
            return False

        if external_call["listen", Int32](Int32(fd), Int32(16)) != 0:
            close_fd(fd)
            _unlink_socket(target_sock)
            return False

        self.listen_fd = fd
        self.active = True
        return True

    def close(mut self):
        for i in range(len(self.client_fds)):
            close_fd(self.client_fds[i])
        self.client_fds.clear()
        self.client_rx_bufs.clear()
        self.client_authed.clear()
        if self.listen_fd >= 0:
            close_fd(self.listen_fd)
            self.listen_fd = -1
        if self.active and self.socket_path.byte_length() > 0:
            _unlink_socket(self.socket_path)
        self.active = False

    def client_count(self) -> Int:
        return len(self.client_fds)

    def _drop_client(mut self, idx: Int):
        close_fd(self.client_fds[idx])
        _ = self.client_fds.pop(idx)
        _ = self.client_rx_bufs.pop(idx)
        _ = self.client_authed.pop(idx)

    def _poll_internal(
        mut self,
        mut mcp: McpServerV2,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
        has_renderer: Bool,
        ren_ptr: Int,
    ) -> Int:
        if not self.active or self.listen_fd < 0:
            return 0

        # Accept up to 8 pending non-blocking connections per tick & verify `SO_PEERCRED`!
        for _ in range(8):
            if len(self.client_fds) >= 32:
                break
            var cfd = Int(
                external_call["accept4", Int32](
                    Int32(self.listen_fd), Int(0), Int(0), _SOCK_CLOEXEC | _SOCK_NONBLOCK
                )
            )
            if cfd < 0:
                break
            if not _verify_peer_uid(cfd, self.expected_peer_uid):
                close_fd(cfd)
                self.rejected_peers += 1
                continue
            self.client_fds.append(cfd)
            self.client_rx_bufs.append(List[UInt8]())
            self.client_authed.append(mcp.auth_token.byte_length() == 0)

        var processed = 0
        var ci = 0
        while ci < len(self.client_fds):
            var cfd = self.client_fds[ci]
            var peer_closed = False
            var peer_error = False

            while len(self.client_rx_bufs[ci]) <= MAX_MCP_MESSAGE_BYTES:
                var ask = min(8192, (MAX_MCP_MESSAGE_BYTES + 1) - len(self.client_rx_bufs[ci]))
                var n = Int(external_call["read", Int64](Int32(cfd), self._io_scratch.unsafe_ptr(), Int64(ask)))
                if n > 0:
                    for k in range(n):
                        self.client_rx_bufs[ci].append(self._io_scratch[k])
                elif n == 0:
                    peer_closed = True
                    break
                else:
                    var err = last_errno()
                    if err == _EINTR:
                        continue
                    if err == _EAGAIN:
                        break
                    peer_error = True
                    break

            if peer_error:
                self._drop_client(ci)
                continue

            var dropped_in_parse = False
            while True:
                var frame = parse_next_mcp_frame(self.client_rx_bufs[ci])
                if frame.protocol_error:
                    var err_resp = _jsonrpc_err("null", frame.error_code, frame.error_message)
                    var err_b = encode_mcp_frame(err_resp, use_content_length=frame.used_content_length)
                    _ = send_all_bounded(cfd, Int(err_b.unsafe_ptr()), len(err_b), timeout_ms=500, is_socket=True)
                    _ = len(err_b)
                    self._drop_client(ci)
                    dropped_in_parse = True
                    break
                if not frame.has_frame:
                    break

                mcp.session_authenticated = self.client_authed[ci]
                var resp = mcp._handle_request_bytes(frame.payload, studio, assets, pool, has_renderer, ren_ptr)
                self.client_authed[ci] = mcp.session_authenticated
                processed += 1

                if resp.byte_length() > 0:
                    var out_b = encode_mcp_frame(resp, use_content_length=frame.used_content_length)
                    var ok_send = send_all_bounded(cfd, Int(out_b.unsafe_ptr()), len(out_b), timeout_ms=2000, is_socket=True)
                    _ = len(out_b)
                    if not ok_send:
                        self._drop_client(ci)
                        dropped_in_parse = True
                        break

            if dropped_in_parse:
                continue

            if peer_closed:
                self._drop_client(ci)
                continue

            ci += 1

        return processed

    def poll_once(
        mut self,
        mut mcp: McpServerV2,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
    ) -> Int:
        return self._poll_internal(mcp, studio, assets, pool, False, 0)

    def poll_once_with_renderer(
        mut self,
        mut mcp: McpServerV2,
        mut studio: FlintStudioState,
        mut assets: AssetServer,
        mut pool: JobPool,
        mut ren: FlintAaaVulkanRenderer,
    ) -> Int:
        return self._poll_internal(mcp, studio, assets, pool, True, Int(Pointer(to=ren)))


struct McpUnixSessionClient(Movable):
    """Persistent Unix Domain Socket client for multi-turn JSON-RPC 2.0 MCP sessions."""
    var fd: Int
    var rx_buf: List[UInt8]
    var use_content_length: Bool

    def __init__(out self, use_content_length: Bool = True):
        self.fd = -1
        self.rx_buf = List[UInt8]()
        self.use_content_length = use_content_length

    def __deinit__(deinit self):
        if self.fd >= 0:
            close_fd(self.fd)

    def connect(mut self, socket_path: String) -> Bool:
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
        self.rx_buf.clear()
        var fd = Int(external_call["socket", Int32](_AF_UNIX, _SOCK_STREAM | _SOCK_CLOEXEC, Int32(0)))
        if fd < 0:
            return False
        var addr = _build_sockaddr_un(socket_path)
        var rc = external_call["connect", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(110))
        _ = len(addr)
        if rc != 0:
            close_fd(fd)
            return False
        self.fd = fd
        return True

    def send_raw_bytes(mut self, raw: List[UInt8], timeout_ms: Int = 2000) -> Bool:
        if self.fd < 0 or len(raw) == 0:
            return False
        var ok = send_all_bounded(self.fd, Int(raw.unsafe_ptr()), len(raw), timeout_ms=timeout_ms, is_socket=True)
        _ = len(raw)
        return ok

    def send_raw_request(mut self, req_json: String, timeout_ms: Int = 2000) -> Bool:
        if self.fd < 0:
            return False
        var frame = encode_mcp_frame(req_json, use_content_length=self.use_content_length)
        var ok = send_all_bounded(self.fd, Int(frame.unsafe_ptr()), len(frame), timeout_ms=timeout_ms, is_socket=True)
        _ = len(frame)
        return ok

    def try_recv_response(mut self) -> Tuple[Bool, String]:
        if self.fd < 0:
            return (False, String())
        var scratch = List[UInt8](length=16384, fill=0)
        while len(self.rx_buf) <= MAX_MCP_MESSAGE_BYTES:
            var n = Int(
                external_call["recv", Int64](Int32(self.fd), scratch.unsafe_ptr(), Int64(16384), Int32(0x40))
            )
            if n < 0:
                if last_errno() == _EINTR:
                    continue
                break
            if n == 0:
                break
            for i in range(n):
                self.rx_buf.append(scratch[i])
        var parsed = parse_next_mcp_frame(self.rx_buf)
        if parsed.protocol_error or not parsed.has_frame:
            return (False, String())
        return (True, String(unsafe_from_utf8=parsed.payload))

    def recv_response_blocking(mut self, timeout_ms: Int = 2000) -> Tuple[Bool, String]:
        """Polls `self.fd` up to `timeout_ms` until a complete response frame arrives."""
        if self.fd < 0:
            return (False, String())
        var deadline = get_time() + Float64(max(1, timeout_ms)) / 1000.0
        var pfd = List[UInt8](length=8, fill=0)
        while True:
            var got = self.try_recv_response()
            if got[0]:
                _ = len(pfd)
                return (True, got[1])
            var rem_ms = Int((deadline - get_time()) * 1000.0 + 0.999)
            if rem_ms <= 0:
                break
            var pbase = Int(pfd.unsafe_ptr())
            Pointer[Int32, MutAnyOrigin](unsafe_from_address=pbase)[unsafe_offset=0] = Int32(self.fd)
            Pointer[Int16, MutAnyOrigin](unsafe_from_address=pbase + 4)[unsafe_offset=0] = Int16(0x0001)  # POLLIN
            Pointer[Int16, MutAnyOrigin](unsafe_from_address=pbase + 6)[unsafe_offset=0] = Int16(0)
            var prc = external_call["poll", Int32](pfd.unsafe_ptr(), UInt64(1), Int32(min(rem_ms, 25)))
            if prc < 0 and last_errno() != _EINTR:
                break
            if prc > 0:
                var rev = Pointer[Int16, ImmutAnyOrigin](unsafe_from_address=pbase + 6)[unsafe_offset=0]
                if (rev & _POLLERR_MASK) != 0 and (rev & Int16(0x0001)) == 0:
                    var tail = self.try_recv_response()
                    _ = len(pfd)
                    return tail
        _ = len(pfd)
        return self.try_recv_response()

    def close(mut self):
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
