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
from flint.anim.flanim_ecs import load_flanim_bytes
from flint.anim.flrig import load_flrig_bytes
from flint.anim.plugin import AnimAssetCache, AnimatorComponent, SkeletonComponent
from flint.assets.asset_ref import AssetRef, normalize_asset_rel_path
from flint.assets.checkpoint import pack_checkpoint_lz4
from flint.assets.cooker_v2 import DEFAULT_COOK_CACHE_DIR, cook_workspace
from flint.assets.flmat import FlmatAsset, load_flmat_bytes
from flint.assets.flpk_v3 import ASSET_KIND_AUDIO, ASSET_KIND_MESH, ASSET_KIND_RAW, ASSET_KIND_TEXTURE
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.image import decode_png, write_png
from flint.assets.jpeg import decode_image_auto
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote, json_text
from flint.assets.server import AssetServer
from flint.audio.qoa import decode_qoa
from flint.audio.server import AudioServer, BUS_SFX
from flint.core.build_info import build_stamp
from flint.devtools.job_runner import JOB_SUCCEEDED, JobRunner
from flint.devtools.problems import (
    DIAG_ERROR,
    DIAG_INFO,
    DIAG_SRC_BUILD,
    DIAG_WARNING,
    Diagnostic,
    ProblemFilter,
    ProblemStore,
    severity_label,
    source_label,
)
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.memory import W_CPU_LIVE, W_VRAM_LIVE, mem_total
from flint.core.serial import BlobWriter
from flint.ecs.world import Entity
from flint.nav.plugin import NavPluginStateResource
from flint.platform.audio import decode_wav
from flint.platform.os import (
    close_fd,
    env_now,
    env_value,
    file_size,
    get_time,
    is_directory,
    is_symlink,
    last_errno,
    list_directory,
    make_dirs,
    parent_directory,
    read_bytes,
    write_bytes,
)
from flint.physics.world import TRIGGER_ENTER, TRIGGER_EXIT, TRIGGER_STAY
from flint.render.aaa_npr_vfx import AaaImpactFrameParams, AaaShockwaveRing
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import flip_error_map, flip_mean
from flint.render.plugin import MaterialOverrideComponent, MeshRendererComponent
from flint.render.render_world import ScalableRenderer
from flint.render.simd_canvas import Camera3D, SimdCanvas, pack_argb
from flint.scene.scene import Prefab, PrefabInstanceComponent, load_flprefab_bytes
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, PrevGlobalTransform
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
    TX_SPAWN_ENTITY,
    FlintStudioState,
    StudioEntityMetaComponent,
    StudioStyleComponent,
    bake_procedural_to_glb,
    generate_procedural_forge_model,
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

def _c_string_from_span(sp: Span[UInt8, _], max_len: Int = 4096) -> String:
    var out = List[UInt8]()
    var lim = min(max_len, len(sp))
    for i in range(lim):
        var b = sp[i]
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
    var mode = (
        UInt32(buf[_STAT_MODE_OFF])
        | (UInt32(buf[_STAT_MODE_OFF + 1]) << 8)
        | (UInt32(buf[_STAT_MODE_OFF + 2]) << 16)
        | (UInt32(buf[_STAT_MODE_OFF + 3]) << 24)
    )
    var uid = (
        UInt32(buf[_STAT_UID_OFF])
        | (UInt32(buf[_STAT_UID_OFF + 1]) << 8)
        | (UInt32(buf[_STAT_UID_OFF + 2]) << 16)
        | (UInt32(buf[_STAT_UID_OFF + 3]) << 24)
    )
    return (True, mode, uid)


def _getcwd_str() -> String:
    var buf = List[UInt8](length=4096, fill=0)
    var got = external_call["getcwd", Int](buf.unsafe_ptr(), Int64(4096))
    if got == 0:
        return String(".")
    return _c_string_from_span(Span(buf), 4096)


def _realpath_existing(path: String) -> Tuple[Bool, String]:
    if path.byte_length() == 0:
        return (False, String())
    var owned = path
    var resolved = List[UInt8](length=4096, fill=0)
    var got = external_call["realpath", Int](owned.as_c_string_span(), resolved.unsafe_ptr())
    if got == 0:
        return (False, String())
    return (True, _c_string_from_span(Span(resolved), 4096))


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


def _trim_ascii_span(bytes: Span[UInt8, _], start: Int, end: Int) -> String:
    var s = start
    var e = end
    while s < e and (bytes[s] == 32 or bytes[s] == 9 or bytes[s] == 13):
        s += 1
    while e > s and (bytes[e - 1] == 32 or bytes[e - 1] == 9 or bytes[e - 1] == 13):
        e -= 1
    var out = List[UInt8](capacity=max(0, e - s))
    for i in range(s, e):
        out.append(bytes[i])
    return String(unsafe_from_utf8=out)


def _resolve_packaged_flpk_name(manifest_path: String) -> String:
    var raw = read_bytes(manifest_path)
    if len(raw) == 0:
        return String("assets.flpk")
    var prefix = "assets_pack="
    var plen = prefix.byte_length()
    var line_start = 0
    var n = len(raw)
    for idx in range(n + 1):
        if idx < n and raw[idx] != 10:
            continue
        var line = _trim_ascii_span(raw, line_start, idx)
        line_start = idx + 1
        if line.startswith(prefix):
            var lb = line.as_bytes()
            var val = _trim_ascii_span(lb, plen, len(lb))
            if val.byte_length() == 0:
                return String("assets.flpk")
            if val == "none":
                return String("")
            if (
                val == "."
                or val == ".."
                or val.find("/") >= 0
                or val.find("..") >= 0
                or _has_forbidden_path_bytes(val)
            ):
                return String("")
            return val^
    return String("assets.flpk")


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


def _normalize_abs_lexical(abs_path: String) -> String:
    """Normalizes `.` and `..` segments in an already-rooted absolute path (for `workspace_root` only)."""
    var segs = List[String]()
    var b = abs_path.as_bytes()
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
        var seg = String(abs_path[byte=i:j])
        if seg == "..":
            if len(segs) > 0:
                _ = segs.pop()
        elif seg != "." and seg.byte_length() > 0:
            segs.append(seg^)
        i = j
    if len(segs) == 0:
        return String("/")
    var out = String()
    for k in range(len(segs)):
        out += "/" + segs[k]
    return out^


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
    var lex = _normalize_abs_lexical(base)
    var rp = _realpath_existing(base)
    if rp[0]:
        return (_strip_trailing_slashes(rp[1]), lex)
    return (lex, lex)


def _sanitize_workspace_rel_and_abs(
    workspace_root: String,
    user_path: String,
    allow_root_dir: Bool = False,
) raises -> Tuple[String, String]:
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
    var rel_out = String()
    var deepest_existing_dir = canon_root
    for i in range(len(rel_segs)):
        if i > 0:
            rel_out += "/"
        rel_out += rel_segs[i]
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

    return (cur^, rel_out^)


def _sanitize_workspace_path(
    workspace_root: String,
    user_path: String,
    allow_root_dir: Bool = False,
) raises -> String:
    return _sanitize_workspace_rel_and_abs(workspace_root, user_path, allow_root_dir=allow_root_dir)[0]


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


def _write_pollfd_bytes(mut pfd: List[UInt8], fd: Int32, events: Int16):
    var ufd = UInt32(fd)
    pfd[0] = UInt8(ufd & 0xFF)
    pfd[1] = UInt8((ufd >> 8) & 0xFF)
    pfd[2] = UInt8((ufd >> 16) & 0xFF)
    pfd[3] = UInt8((ufd >> 24) & 0xFF)
    var uev = UInt16(events)
    pfd[4] = UInt8(uev & 0xFF)
    pfd[5] = UInt8((uev >> 8) & 0xFF)
    pfd[6] = 0
    pfd[7] = 0


def _read_pollfd_revents(pfd: List[UInt8]) -> Int16:
    var ur = UInt16(pfd[6]) | (UInt16(pfd[7]) << 8)
    return Int16(ur)


def send_span_bounded(
    fd: Int,
    data: Span[UInt8, _],
    timeout_ms: Int = 2000,
    is_socket: Bool = True,
) -> Bool:
    """Writes all bytes in `data` to `fd`, retrying `EINTR` and polling `POLLOUT`
    on `EAGAIN`/`EWOULDBLOCK` up to `timeout_ms` while preserving `Span` origin lifetimes."""
    var length = len(data)
    if fd < 0 or length < 0:
        return False
    if length == 0:
        return True

    var deadline = get_time() + Float64(max(1, timeout_ms)) / 1000.0
    var sent = 0
    var pfd = List[UInt8](length=8, fill=0)

    while sent < length:
        var cur_ptr = data.unsafe_ptr().unsafe_offset(sent)
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
            return False

        var err = last_errno()
        if err == _EINTR:
            continue
        if err == _EAGAIN:
            var rem_ms = Int((deadline - get_time()) * 1000.0 + 0.999)
            if rem_ms <= 0:
                return False
            _write_pollfd_bytes(pfd, Int32(fd), _POLLOUT)
            var wait_slice = Int32(min(rem_ms, 50))
            var prc = external_call["poll", Int32](pfd.unsafe_ptr(), UInt64(1), wait_slice)
            if prc < 0:
                if last_errno() == _EINTR:
                    continue
                return False
            if prc > 0:
                var rev = _read_pollfd_revents(pfd)
                if (rev & _POLLERR_MASK) != 0 and (rev & _POLLOUT) == 0:
                    return False
            continue

        return False

    return True


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
    var ptr = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    var sp = Span[UInt8, ImmutAnyOrigin](ptr=ptr, length=length)
    return send_span_bounded(fd, sp, timeout_ms=timeout_ms, is_socket=is_socket)


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


def _parse_strict_json_int(doc: JsonDocument, node: Int) -> Tuple[Bool, Int]:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 2:
        return (False, -1)
    var num = doc.nodes[node].number
    if not isfinite(num) or num < -9223372036854774784.0 or num > 9223372036854774784.0:
        return (False, -1)
    var iv = Int(num)
    if Float64(iv) != num:
        return (False, -1)
    return (True, iv)


def _is_supported_asset_ext(lower_path: String) -> Bool:
    return (
        lower_path.endswith(".glb")
        or lower_path.endswith(".png")
        or lower_path.endswith(".jpg")
        or lower_path.endswith(".jpeg")
        or lower_path.endswith(".hdr")
        or lower_path.endswith(".exr")
        or lower_path.endswith(".wav")
        or lower_path.endswith(".qoa")
        or lower_path.endswith(".flmat")
        or lower_path.endswith(".flprefab")
        or lower_path.endswith(".flrig")
        or lower_path.endswith(".flanim")
        or lower_path.endswith(".flscn")
        or lower_path.endswith(".flscnb")
        or lower_path.endswith(".ttf")
        or lower_path.endswith(".otf")
        or lower_path.endswith(".json")
    )


def _is_forbidden_abs_src(abs_path: String) -> Bool:
    return (
        abs_path.startswith("/etc/")
        or abs_path == "/etc"
        or abs_path.startswith("/proc/")
        or abs_path == "/proc"
        or abs_path.startswith("/sys/")
        or abs_path == "/sys"
        or abs_path.startswith("/dev/")
        or abs_path == "/dev"
        or abs_path.startswith("/boot/")
        or abs_path == "/boot"
        or abs_path.startswith("/root/")
        or abs_path == "/root"
    )


def _first_str_arg(doc: JsonDocument, args_node: Int, k0: String, k1: String = "", k2: String = "") -> String:
    if args_node < 0 or doc.nodes[args_node].kind != 5:
        return String()
    var n0 = doc.object_get(args_node, k0)
    if n0 >= 0 and doc.nodes[n0].kind == 3:
        var s0 = doc.text(n0, "")
        if s0.byte_length() > 0:
            return s0
    if k1.byte_length() > 0:
        var n1 = doc.object_get(args_node, k1)
        if n1 >= 0 and doc.nodes[n1].kind == 3:
            var s1 = doc.text(n1, "")
            if s1.byte_length() > 0:
                return s1
    if k2.byte_length() > 0:
        var n2 = doc.object_get(args_node, k2)
        if n2 >= 0 and doc.nodes[n2].kind == 3:
            var s2 = doc.text(n2, "")
            if s2.byte_length() > 0:
                return s2
    return String()


def _has_valid_scene_ext_and_stem(path: String) -> Bool:
    var b = path.as_bytes()
    var last_slash = -1
    for i in range(len(b)):
        if b[i] == 47:
            last_slash = i
    var base = String(path[byte=last_slash + 1 : len(b)])
    if base.endswith(".flscn"):
        return base.byte_length() > 6
    if base.endswith(".flscnb"):
        return base.byte_length() > 7
    if base.endswith(".flscene"):
        return base.byte_length() > 8
    return False


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


def _parse_quat4_arg(doc: JsonDocument, node: Int, fallback: Quat) -> Tuple[Bool, Quat]:
    if node < 0 or node >= len(doc.nodes):
        return (True, fallback)
    if doc.nodes[node].kind != 4 or doc.array_size(node) != 4:
        return (False, fallback)
    var c0 = doc.at(node, 0); var c1 = doc.at(node, 1); var c2 = doc.at(node, 2); var c3 = doc.at(node, 3)
    if doc.nodes[c0].kind != 2 or doc.nodes[c1].kind != 2 or doc.nodes[c2].kind != 2 or doc.nodes[c3].kind != 2:
        return (False, fallback)
    var qx = Float32(doc.nodes[c0].number)
    var qy = Float32(doc.nodes[c1].number)
    var qz = Float32(doc.nodes[c2].number)
    var qw = Float32(doc.nodes[c3].number)
    if not (isfinite(qx) and isfinite(qy) and isfinite(qz) and isfinite(qw)):
        return (False, fallback)
    var len2 = qx * qx + qy * qy + qz * qz + qw * qw
    if not isfinite(len2) or len2 <= Float32(1.0e-12):
        return (False, fallback)
    var q = Quat(qx, qy, qz, qw).normalized()
    if not (isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)):
        return (False, fallback)
    return (True, q)


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
        + '{"name":"render.trigger_post_fx","description":"Sets or clears Manga Impact Frame blend/invert, chromatic-dispersion shockwave ring, and limited-animation SteppedAnimClock hitstop freeze on Studio.","inputSchema":{"type":"object","properties":{"impact_blend":{"type":"number"},"impact_invert":{"type":"boolean"},"shockwave_active":{"type":"boolean"},"shockwave_radius":{"type":"number"},"hitstop_sec":{"type":"number"},"clear":{"type":"boolean"}}}},'
        + '{"name":"scene.save","description":"Saves the live Studio scene (.flscn or .flscnb) inside workspace_root.","inputSchema":{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}},'
        + '{"name":"scene.load","description":"Loads a Studio scene (.flscn or .flscnb) from inside workspace_root.","inputSchema":{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}},'
        + '{"name":"scene.set_entity_enabled","description":"Enables or disables an entity visibility/enabled state with reversible Undo/Redo recording.","inputSchema":{"type":"object","required":["entity_id","enabled"],"properties":{"entity_id":{"type":"integer"},"enabled":{"type":"boolean"},"coalesce_key":{"type":"integer"}}}},'
        + '{"name":"entity.set_transform","description":"Updates entity position, rotation ([x,y,z,w]), and/or scale with reversible Undo/Redo recording.","inputSchema":{"type":"object","required":["entity_id"],"properties":{"entity_id":{"type":"integer"},"position":{"type":"array"},"rotation":{"type":"array"},"scale":{"type":"array"},"coalesce_key":{"type":"integer"}}}},'
        + '{"name":"asset.import","description":"Validates and imports an asset (.glb, .png, .jpg, .wav, .qoa, .flmat, .flprefab, .flrig, .flanim) into the project workspace.","inputSchema":{"type":"object","properties":{"source_path":{"type":"string"},"src_path":{"type":"string"},"path":{"type":"string"},"dest_path":{"type":"string"},"rel_path":{"type":"string"},"target_path":{"type":"string"}}}},'
        + '{"name":"asset.list","description":"Lists project-relative assets inside workspace_root (default dir: assets).","inputSchema":{"type":"object","properties":{"dir":{"type":"string"}}}},'
        + '{"name":"material.assign","description":"Parses and assigns a .flmat material asset onto an entity with Undo/Redo support.","inputSchema":{"type":"object","required":["entity_id"],"properties":{"entity_id":{"type":"integer"},"material_path":{"type":"string"},"path":{"type":"string"},"asset_path":{"type":"string"}}}},'
        + '{"name":"prefab.spawn","description":"Instantiates a .flprefab asset into the live Studio scene with Undo/Redo support.","inputSchema":{"type":"object","properties":{"prefab_path":{"type":"string"},"path":{"type":"string"},"asset_path":{"type":"string"},"position":{"type":"array"},"rotation":{"type":"array"},"scale":{"type":"array"}}}},'
        + '{"name":"physics.trigger_events","description":"Reads PIE trigger enter/stay/exit events emitted since the last query (requires active PIE).","inputSchema":{"type":"object","properties":{"entity_id":{"type":"integer"}}}},'
        + '{"name":"anim.set_state","description":"Sets animation state/clip on a live entity during active PIE (attaching AnimatorComponent if missing).","inputSchema":{"type":"object","required":["entity_id","state"],"properties":{"entity_id":{"type":"integer"},"state":{"type":"string"},"playing":{"type":"boolean"},"loop":{"type":"boolean"},"speed":{"type":"number"}}}},'
        + '{"name":"audio.play","description":"Plays a workspace-sandboxed .wav or .qoa sound via AudioServer during active PIE.","inputSchema":{"type":"object","required":["asset_path"],"properties":{"asset_path":{"type":"string"},"volume":{"type":"number"},"pitch":{"type":"number"},"looping":{"type":"boolean"},"position":{"type":"array"}}}},'
        + '{"name":"nav.query_path","description":"Queries a deterministic NavMesh A* path between start and goal during active PIE (auto-baking NavMesh if needed).","inputSchema":{"type":"object","required":["start","goal"],"properties":{"start":{"type":"array"},"goal":{"type":"array"},"search_radius":{"type":"number"}}}},'
        + '{"name":"project.package","description":"Packages the open Flint project into a self-contained distribution bundle inside workspace_root via tools/flint.sh package.","inputSchema":{"type":"object","properties":{"output_dir":{"type":"string"},"profile":{"type":"string"},"loose_assets":{"type":"boolean"},"target":{"type":"string"}}}},'
        + '{"name":"code.build_hot_reload","description":"Compiles project gameplay Mojo code into a shared library inside workspace_root, parses compiler diagnostics, and hot-reloads it via HotReloadHost without resetting World state.","inputSchema":{"type":"object","properties":{"entry_path":{"type":"string"},"output_so":{"type":"string"},"reload_only":{"type":"boolean"},"opt_level":{"type":"string"},"compiler":{"type":"string"}}}},'
        + '{"name":"code.diagnostics","description":"Returns merged Mojo LSP, build, test, and agent diagnostics currently tracked by Studio.","inputSchema":{"type":"object","properties":{"file":{"type":"string"},"severity":{"type":"string"},"max_items":{"type":"integer"}}}},'
        + '{"name":"nav.bake","description":"Rebuilds the deterministic scene NavMesh from live static colliders and returns baked polygon count.","inputSchema":{"type":"object","properties":{}}}'
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
# 4. Observable Tool Telemetry Hook, Gate & `McpServerV2` Dispatcher
# ============================================================================

comptime MCP_GATE_AUTO: Int = 0       # All MCP tool calls execute immediately
comptime MCP_GATE_MUTATING: Int = 1   # Mutating tools require explicit approval
comptime MCP_GATE_ALL: Int = 2        # All tool calls require explicit approval
comptime MAX_MCP_TOOL_EVENTS: Int = 512


struct McpToolEvent(Copyable, Movable):
    """Observable telemetry record emitted on every MCP v2 `tools/call` invocation."""
    var seq: UInt64
    var tool_name: String
    var params_json: String
    var result_json: String
    var affected_entities: List[Int]
    var elapsed_ms: Float64
    var ok: Bool
    var is_mutating: Bool
    var blocked_by_gate: Bool

    def __init__(
        out self,
        seq: UInt64 = 0,
        tool_name: String = "",
        params_json: String = "{}",
        result_json: String = "{}",
        elapsed_ms: Float64 = 0.0,
        ok: Bool = True,
        is_mutating: Bool = False,
        blocked_by_gate: Bool = False,
    ):
        self.seq = seq
        self.tool_name = tool_name
        self.params_json = params_json
        self.result_json = result_json
        self.affected_entities = List[Int]()
        self.elapsed_ms = elapsed_ms
        self.ok = ok
        self.is_mutating = is_mutating
        self.blocked_by_gate = blocked_by_gate


def mcp_tool_is_mutating(tool_name: String) -> Bool:
    """Returns False for pure read-only inspection tools and True for scene/code/runtime mutations."""
    if (
        tool_name == "ecs.query_entities"
        or tool_name == "render.capture_viewport"
        or tool_name == "render.inspect_gpu_frame"
        or tool_name == "physics.raycast_query"
        or tool_name == "asset.list"
        or tool_name == "physics.trigger_events"
        or tool_name == "nav.query_path"
        or tool_name == "code.diagnostics"
    ):
        return False
    return True


def _extract_affected_entities(
    args_doc: JsonDocument, args_node: Int, result_json: String
) -> List[Int]:
    var out = List[Int]()
    if args_node >= 0 and args_node < len(args_doc.nodes) and args_doc.nodes[args_node].kind == 5:
        var eid_node = args_doc.object_get(args_node, "entity_id")
        if eid_node >= 0 and args_doc.nodes[eid_node].kind == 2:
            var eid = args_doc.integer(eid_node, -1)
            if eid >= 0:
                out.append(eid)
    if result_json.byte_length() > 0:
        var rb = List[UInt8](capacity=result_json.byte_length())
        for b in result_json.as_bytes():
            rb.append(b)
        var rdoc = parse_json(rb)
        if rdoc.valid and rdoc.root >= 0 and rdoc.nodes[rdoc.root].kind == 5:
            var reid_node = rdoc.object_get(rdoc.root, "entity_id")
            if reid_node >= 0 and rdoc.nodes[reid_node].kind == 2:
                var reid = rdoc.integer(reid_node, -1)
                if reid >= 0 and reid not in out:
                    out.append(reid)
            var arr_node = rdoc.object_get(rdoc.root, "entity_ids")
            if arr_node >= 0 and rdoc.nodes[arr_node].kind == 4:
                for i in range(rdoc.array_size(arr_node)):
                    var it = rdoc.at(arr_node, i)
                    if it >= 0 and rdoc.nodes[it].kind == 2:
                        var idv = rdoc.integer(it, -1)
                        if idv >= 0 and idv not in out:
                            out.append(idv)
    return out^


struct McpServerV2(Movable):
    var initialized: Bool
    var requests_handled: Int
    var tool_calls_handled: Int
    var client_name: String
    var workspace_root: String
    var auth_token: String
    var session_authenticated: Bool
    var trigger_event_cursor: Int
    var diagnostics_store: ProblemStore
    var tool_events: List[McpToolEvent]
    var next_event_seq: UInt64
    var gate_mode: Int
    var once_allow_tools: List[String]
    var session_allow_tools: List[String]

    def __init__(out self, workspace_root: String = ".", auth_token: String = ""):
        self.initialized = False
        self.requests_handled = 0
        self.tool_calls_handled = 0
        self.client_name = String("unknown-mcp-client")
        self.workspace_root = workspace_root if workspace_root.byte_length() > 0 else String(".")
        self.auth_token = auth_token
        self.session_authenticated = auth_token.byte_length() == 0
        self.trigger_event_cursor = 0
        self.diagnostics_store = ProblemStore()
        self.tool_events = List[McpToolEvent]()
        self.next_event_seq = 1
        self.gate_mode = MCP_GATE_AUTO
        self.once_allow_tools = List[String]()
        self.session_allow_tools = List[String]()

    def set_gate_mode(mut self, mode: Int):
        if mode == MCP_GATE_MUTATING or mode == MCP_GATE_ALL:
            self.gate_mode = mode
        else:
            self.gate_mode = MCP_GATE_AUTO

    def allow_tool_once(mut self, tool_name: String):
        if tool_name.byte_length() > 0:
            self.once_allow_tools.append(tool_name)

    def allow_tool_session(mut self, tool_name: String):
        if tool_name.byte_length() > 0 and tool_name not in self.session_allow_tools:
            self.session_allow_tools.append(tool_name)

    def clear_session_allows(mut self):
        self.once_allow_tools.clear()
        self.session_allow_tools.clear()

    def _consume_gate_allowance(mut self, tool_name: String, is_mutating: Bool) -> Bool:
        if self.gate_mode == MCP_GATE_AUTO:
            return True
        if self.gate_mode == MCP_GATE_MUTATING and not is_mutating:
            return True
        if tool_name in self.session_allow_tools:
            return True
        for i in range(len(self.once_allow_tools)):
            if self.once_allow_tools[i] == tool_name:
                _ = self.once_allow_tools.pop(i)
                return True
        return False

    def _record_tool_event(mut self, var ev: McpToolEvent):
        if len(self.tool_events) >= MAX_MCP_TOOL_EVENTS:
            _ = self.tool_events.pop(0)
        self.tool_events.append(ev^)

    def peek_tool_event_count(self) -> Int:
        return len(self.tool_events)

    def take_tool_events(mut self) -> List[McpToolEvent]:
        var out = List[McpToolEvent]()
        swap(out, self.tool_events)
        return out^

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
                var ent = studio.resolve_entity(e.entity_id)
                var mesh_ref = AssetRef()
                var mat_ref = AssetRef()
                if ent.valid():
                    var mr_opt = studio.app.world.get_ptr[MeshRendererComponent](ent)
                    if mr_opt:
                        mesh_ref = mr_opt.value()[].mesh_ref.copy()
                        mat_ref = mr_opt.value()[].material_ref.copy()
                    var mo_opt = studio.app.world.get_ptr[MaterialOverrideComponent](ent)
                    if mo_opt and not mo_opt.value()[].material_ref.is_empty():
                        mat_ref = mo_opt.value()[].material_ref.copy()
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
                    + ',"mesh_id":' + String(e.mesh_id)
                    + ',"material_id":' + String(e.material_id)
                    + ',"mesh_ref":' + mesh_ref.write_json()
                    + ',"material_ref":' + mat_ref.write_json()
                    + ',"material_path":' + json_quote(mat_ref.rel_path)
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
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"ecs.spawn_entity requires PIE to be stopped (PIE is active)"}')
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
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"ecs.patch_components requires PIE to be stopped (PIE is active)"}')
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
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"ecs.despawn_entity requires PIE to be stopped (PIE is active)"}')
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
            if do_spawn and studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"procedural.generate_asset with spawn_in_scene=true requires PIE to be stopped (PIE is active)"}')
            var pos_r = _parse_vec3_arg(doc, doc.object_get(args_node, "position"), Vec3(0.0, 0.0, 0.0))
            if not pos_r[0]:
                return (True, '{"ok":false,"error":"invalid position"}')

            var abs_glb = String("")
            var rel_glb = String("")
            if raw_out_glb.byte_length() > 0:
                try:
                    var san = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_out_glb, allow_root_dir=False)
                    abs_glb = san[0]
                    rel_glb = san[1]
                except e:
                    return (True, '{"ok":false,"error":' + json_quote("sandbox rejected output_glb_path: " + String(e)) + "}")
                var pdir = parent_directory(abs_glb)
                if pdir.byte_length() > 0 and pdir != ".":
                    _ = make_dirs(pdir)

            var mdl = generate_procedural_forge_model(preset, seed)
            var v_sum = 0
            var t_sum = 0
            for mi in range(len(mdl.meshes)):
                v_sum += mdl.meshes[mi].vertex_count()
                t_sum += len(mdl.meshes[mi].indices) // 3
            var glb_bytes = encode_native_glb(mdl)
            if abs_glb.byte_length() > 0:
                if not write_bytes(abs_glb, glb_bytes):
                    return (True, '{"ok":false,"error":"failed to write procedural GLB"}')

            var uploaded_mesh_id = 0
            var uploaded_mat_id = -1
            var rend_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
            if rend_opt:
                var up = rend_opt.value()[unsafe_offset=0].upload_native_model(
                    mdl, build_lods=False, upload_skinned=False
                )
                if len(up.mesh_entries) > 0:
                    uploaded_mesh_id = up.mesh_entries[0].mesh_id
                if len(up.material_ids) > 0:
                    uploaded_mat_id = up.material_ids[0]

            var mesh_ref = (
                AssetRef.from_path(rel_glb, ASSET_KIND_MESH)
                if rel_glb.byte_length() > 0
                else AssetRef()
            )
            var spawned_id = -1
            if do_spawn:
                spawned_id = studio.spawn_entity(
                    "Proc_" + preset,
                    position=pos_r[1],
                    mesh_id=uploaded_mesh_id,
                    record_undo=True,
                )
                var se = studio.resolve_entity(spawned_id)
                if se.valid():
                    var mr_opt = studio.app.world.get_ptr_mut[MeshRendererComponent](se)
                    if mr_opt:
                        mr_opt.value()[].mesh_id = uploaded_mesh_id
                        if uploaded_mat_id >= 0:
                            mr_opt.value()[].material_id = uploaded_mat_id
                        if not mesh_ref.is_empty():
                            mr_opt.value()[].mesh_ref = mesh_ref.copy()
                    studio._sync_ecs_and_physics()
                    if len(studio.undo_stack) > 0:
                        studio.undo_stack[len(studio.undo_stack) - 1].after_world_blob = studio._capture_world_blob()

            if rel_glb.byte_length() > 0:
                studio.asset_browser_scanned = False
                if assets.loose_root.byte_length() == 0:
                    assets.loose_root = self.workspace_root
                assets.index_loose_directory()

            return (
                False,
                '{"ok":true,"preset":' + json_quote(preset)
                + ',"vertex_count":' + String(v_sum)
                + ',"triangle_count":' + String(t_sum)
                + ',"glb_bytes":' + String(len(glb_bytes))
                + ',"mesh_id":' + String(uploaded_mesh_id)
                + ',"output_glb_path":' + json_quote(rel_glb)
                + ',"asset_ref":' + (mesh_ref.write_json() if not mesh_ref.is_empty() else String('""'))
                + ',"spawned_entity_id":' + String(spawned_id) + "}",
            )

        # 9. physics.step_deterministic (optional checkpoint output strictly sandboxed inside `workspace_root`)
        if tool_name == "physics.step_deterministic":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"physics.step_deterministic requires PIE to be stopped (PIE is active)"}')
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
            var was_stopped = studio.pie.mode == PIE_STOPPED
            if act == "play":
                if not studio.pie_play():
                    return (True, '{"ok":false,"error":"pie_play failed"}')
                if was_stopped:
                    self.trigger_event_cursor = studio._pie_trigger_buffer.total_seq
            elif act == "pause":
                _ = studio.pie_pause()
            elif act == "step":
                if was_stopped:
                    self.trigger_event_cursor = studio._pie_trigger_buffer.total_seq
                _ = studio.pie_step(ticks)
            elif act == "stop":
                try:
                    restored_exact = studio.pie_stop()
                except:
                    return (True, '{"ok":false,"error":"pie_stop LZ4 verification failed"}')
                self.trigger_event_cursor = studio._pie_trigger_buffer.total_seq
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
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"studio.undo_redo requires PIE to be stopped (PIE is active)"}')
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
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"render.configure_npr_cel requires PIE to be stopped (PIE is active)"}')
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

        # 15. scene.save
        if tool_name == "scene.save":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"scene.save requires PIE to be stopped (PIE is active)"}')
            var raw_path = _first_str_arg(doc, args_node, "path", "scene_path", "file_path")
            if raw_path.byte_length() == 0:
                return (True, '{"ok":false,"error":"path string is required"}')
            var safe_path: String
            var rel_path: String
            try:
                var san = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_path, allow_root_dir=False)
                safe_path = san[0]
                rel_path = san[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected path: " + String(e)) + "}")
            if not _has_valid_scene_ext_and_stem(safe_path):
                return (True, '{"ok":false,"error":"scene path must have a non-empty filename ending with .flscn or .flscnb"}')
            if is_directory(safe_path, follow_symlinks=True):
                return (True, '{"ok":false,"error":"scene path is a directory"}')
            if not studio.save_scene_file(safe_path):
                return (True, '{"ok":false,"error":"failed to save scene file"}')
            return (
                False,
                '{"ok":true,"path":' + json_quote(rel_path)
                + ',"entity_count":' + String(studio.live_entity_count())
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 16. scene.load
        if tool_name == "scene.load":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"scene.load requires PIE to be stopped (PIE is active)"}')
            var raw_path = _first_str_arg(doc, args_node, "path", "scene_path", "file_path")
            if raw_path.byte_length() == 0:
                return (True, '{"ok":false,"error":"path string is required"}')
            var safe_path: String
            var rel_path: String
            try:
                var san = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_path, allow_root_dir=False)
                safe_path = san[0]
                rel_path = san[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected path: " + String(e)) + "}")
            if not _has_valid_scene_ext_and_stem(safe_path):
                return (True, '{"ok":false,"error":"scene path must have a non-empty filename ending with .flscn or .flscnb"}')
            if is_directory(safe_path, follow_symlinks=True):
                return (True, '{"ok":false,"error":"scene path is a directory"}')
            var raw_bytes = read_bytes(safe_path)
            if len(raw_bytes) == 0:
                return (True, '{"ok":false,"error":"scene file missing or empty"}')
            if not studio.load_scene_file(safe_path):
                return (True, '{"ok":false,"error":"failed to parse or load scene file"}')
            return (
                False,
                '{"ok":true,"path":' + json_quote(rel_path)
                + ',"entity_count":' + String(studio.live_entity_count())
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 17. scene.set_entity_enabled
        if tool_name == "scene.set_entity_enabled":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"scene.set_entity_enabled requires PIE to be stopped (PIE is active)"}')
            var eid_node = doc.object_get(args_node, "entity_id")
            var en_node = doc.object_get(args_node, "enabled")
            var eid_parsed = _parse_strict_json_int(doc, eid_node)
            if not eid_parsed[0]:
                return (True, '{"ok":false,"error":"integer entity_id is required"}')
            if en_node < 0 or doc.nodes[en_node].kind != 1:
                return (True, '{"ok":false,"error":"boolean enabled is required"}')
            var eid = eid_parsed[1]
            if eid < 0 or eid >= len(studio.entities) or not studio.entities[eid].alive:
                return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')
            var enabled = doc.boolean(en_node, True)
            var nd = studio.entities[eid].copy()
            nd.visible = enabled
            var ckey = UInt64(max(0, doc.integer(doc.object_get(args_node, "coalesce_key"), 0)))
            if not studio.commit_node_edit(eid, nd^, "MCP Set Entity Enabled", ckey):
                return (True, '{"ok":false,"error":"failed to update entity enabled state"}')
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"enabled":' + ("true" if enabled else "false")
                + ',"undo_depth":' + String(len(studio.undo_stack))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 18. entity.set_transform
        if tool_name == "entity.set_transform":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"entity.set_transform requires PIE to be stopped (PIE is active)"}')
            var eid_node = doc.object_get(args_node, "entity_id")
            var eid_parsed = _parse_strict_json_int(doc, eid_node)
            if not eid_parsed[0]:
                return (True, '{"ok":false,"error":"integer entity_id is required"}')
            var eid = eid_parsed[1]
            if eid < 0 or eid >= len(studio.entities) or not studio.entities[eid].alive:
                return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')

            var p_node = doc.object_get(args_node, "position")
            var r_node = doc.object_get(args_node, "rotation")
            var s_node = doc.object_get(args_node, "scale")
            if p_node < 0 and r_node < 0 and s_node < 0:
                return (True, '{"ok":false,"error":"at least one of position, rotation, or scale is required"}')

            var pos = studio.entities[eid].translation
            var rot = studio.entities[eid].rotation
            var scl = studio.entities[eid].scale

            if p_node >= 0:
                var pr = _parse_vec3_arg(doc, p_node, pos)
                if not pr[0]:
                    return (True, '{"ok":false,"error":"invalid position (expected 3 finite numbers)"}')
                pos = pr[1]
            if r_node >= 0:
                var rr = _parse_quat4_arg(doc, r_node, rot)
                if not rr[0]:
                    return (True, '{"ok":false,"error":"invalid rotation (expected 4 finite numbers [x,y,z,w])"}')
                rot = rr[1]
            if s_node >= 0:
                var sr = _parse_vec3_arg(doc, s_node, scl)
                if not sr[0]:
                    return (True, '{"ok":false,"error":"invalid scale (expected 3 finite numbers)"}')
                scl = sr[1]

            var ckey = UInt64(max(0, doc.integer(doc.object_get(args_node, "coalesce_key"), 0)))
            if not studio.set_entity_transform(eid, pos, rot, scl, coalesce_key=ckey):
                return (True, '{"ok":false,"error":"failed to apply entity transform"}')
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"undo_depth":' + String(len(studio.undo_stack))
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 19. asset.import
        if tool_name == "asset.import":
            var raw_src = _first_str_arg(doc, args_node, "source_path", "src_path", "path")
            var raw_dst = _first_str_arg(doc, args_node, "dest_path", "rel_path", "target_path")
            if raw_src.byte_length() == 0:
                return (True, '{"ok":false,"error":"source_path or path is required"}')
            if _has_forbidden_path_bytes(raw_src):
                return (True, '{"ok":false,"error":"forbidden byte in source path"}')
            try:
                _ = _split_and_reject_dotdot(raw_src)
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected source_path: " + String(e)) + "}")

            if raw_dst.byte_length() == 0:
                if raw_src.startswith("/"):
                    return (True, '{"ok":false,"error":"dest_path is required when importing from absolute source_path"}')
                raw_dst = raw_src

            var abs_dst: String
            var rel_dst: String
            try:
                var san_dst = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_dst, allow_root_dir=False)
                abs_dst = san_dst[0]
                rel_dst = san_dst[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected dest_path: " + String(e)) + "}")

            var lower_dst = FlintStudioState._ascii_lower(rel_dst)
            if not _is_supported_asset_ext(lower_dst):
                return (True, '{"ok":false,"error":' + json_quote("invalid destination asset extension: " + rel_dst) + "}")

            var abs_src: String
            var check_realpath_forbidden = False
            if raw_src.startswith("/"):
                if _is_forbidden_abs_src(raw_src):
                    return (True, '{"ok":false,"error":"forbidden system source path rejected"}')
                abs_src = _normalize_abs_lexical(raw_src)
                if _is_forbidden_abs_src(abs_src):
                    return (True, '{"ok":false,"error":"forbidden system source path rejected"}')
                var lower_src = FlintStudioState._ascii_lower(abs_src)
                if not _is_supported_asset_ext(lower_src):
                    return (True, '{"ok":false,"error":"invalid source asset extension"}')
                check_realpath_forbidden = True
            else:
                try:
                    var san_src = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_src, allow_root_dir=False)
                    abs_src = san_src[0]
                except e:
                    return (True, '{"ok":false,"error":' + json_quote("sandbox rejected source_path: " + String(e)) + "}")

            var ls = _lstat_mode_uid(abs_src)
            if not ls[0]:
                return (True, '{"ok":false,"error":"source asset file not found"}')
            if (ls[1] & _S_IFMT) != UInt32(0o100000):
                return (True, '{"ok":false,"error":"source path must be a regular non-symlink file"}')
            if check_realpath_forbidden:
                var src_rp = _realpath_existing(abs_src)
                if not src_rp[0]:
                    return (True, '{"ok":false,"error":"source asset file not found"}')
                if _is_forbidden_abs_src(_strip_trailing_slashes(src_rp[1])):
                    return (True, '{"ok":false,"error":"forbidden system source path rejected"}')

            var raw_data = read_bytes(abs_src)
            var vres = FlintStudioState._validate_asset_payload(lower_dst, raw_data)
            if not vres[0]:
                return (True, '{"ok":false,"error":' + json_quote("asset validation failed: " + vres[1]) + "}")

            if abs_src != abs_dst:
                var dst_dir = parent_directory(abs_dst)
                if dst_dir.byte_length() > 0 and dst_dir != ".":
                    _ = make_dirs(dst_dir)
                if not write_bytes(abs_dst, raw_data):
                    return (True, '{"ok":false,"error":"failed to write imported asset"}')

            if assets.loose_root.byte_length() == 0:
                assets.loose_root = self.workspace_root
            assets.index_loose_directory()
            _ = assets._hot_reload_path(rel_dst)
            var app_srv = studio.app.resources.get_ptr_mut[AssetServer]()
            if app_srv:
                if app_srv.value()[].loose_root.byte_length() == 0:
                    app_srv.value()[].loose_root = self.workspace_root
                app_srv.value()[].index_loose_directory()
                _ = app_srv.value()[]._hot_reload_path(rel_dst)
            studio.asset_browser_scanned = False

            var kind = FlintStudioState._asset_kind_for_ext(lower_dst)
            var aref = AssetRef.from_path(rel_dst, kind)
            return (
                False,
                '{"ok":true,"rel_path":' + json_quote(rel_dst)
                + ',"path":' + json_quote(rel_dst)
                + ',"bytes":' + String(len(raw_data))
                + ',"asset_ref":' + aref.write_json() + "}",
            )

        # 20. asset.list
        if tool_name == "asset.list":
            var raw_dir = doc.text(doc.object_get(args_node, "dir"), "assets")
            if raw_dir.byte_length() == 0:
                raw_dir = "assets"
            var abs_dir: String
            var rel_dir: String
            try:
                var san_dir = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_dir, allow_root_dir=True)
                abs_dir = san_dir[0]
                rel_dir = san_dir[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected dir: " + String(e)) + "}")

            if not is_directory(abs_dir, follow_symlinks=False):
                return (False, '{"ok":true,"count":0,"assets":[]}')

            var found_rels = List[String]()
            var dir_stack = List[String]()
            var rel_stack = List[String]()
            var depth_stack = List[Int]()
            dir_stack.append(abs_dir)
            rel_stack.append(rel_dir)
            depth_stack.append(0)
            while len(dir_stack) > 0:
                var cur_d = dir_stack.pop()
                var cur_r = rel_stack.pop()
                var cur_depth = depth_stack.pop()
                var names = list_directory(cur_d)
                for nm in names:
                    if (
                        nm == "."
                        or nm == ".."
                        or nm == ".git"
                        or nm == "build"
                        or nm == ".pixi"
                        or nm == "__pycache__"
                        or nm == ".agents"
                        or nm == "scenes"
                        or nm == "flint.toml"
                        or nm.endswith(".flmeta")
                    ):
                        continue
                    var full = cur_d + "/" + nm
                    if is_symlink(full):
                        continue
                    var rpath = (cur_r + "/" + nm) if cur_r.byte_length() > 0 else nm
                    if is_directory(full, follow_symlinks=False):
                        if cur_depth < 16:
                            dir_stack.append(full)
                            rel_stack.append(rpath)
                            depth_stack.append(cur_depth + 1)
                    else:
                        var low_nm = FlintStudioState._ascii_lower(nm)
                        if (
                            _is_supported_asset_ext(low_nm)
                            and not low_nm.endswith(".flscn")
                            and not low_nm.endswith(".flscnb")
                            and not low_nm.endswith(".flscene")
                        ):
                            found_rels.append(rpath)

            for i in range(1, len(found_rels)):
                var k = found_rels[i]
                var j = i - 1
                while j >= 0 and found_rels[j] > k:
                    found_rels[j + 1] = found_rels[j]
                    j -= 1
                found_rels[j + 1] = k

            var arr_json = String("[")
            for i in range(len(found_rels)):
                if i > 0:
                    arr_json += ","
                var rp = found_rels[i]
                var sz = file_size(self.workspace_root + "/" + rp)
                arr_json += (
                    '{"rel_path":' + json_quote(rp)
                    + ',"path":' + json_quote(rp)
                    + ',"bytes":' + String(max(0, sz)) + "}"
                )
            arr_json += "]"
            return (
                False,
                '{"ok":true,"count":' + String(len(found_rels))
                + ',"assets":' + arr_json + "}",
            )

        # 21. material.assign
        if tool_name == "material.assign":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"material.assign requires PIE to be stopped (PIE is active)"}')
            var eid_parsed = _parse_strict_json_int(doc, doc.object_get(args_node, "entity_id"))
            if not eid_parsed[0]:
                return (True, '{"ok":false,"error":"integer entity_id is required"}')
            var eid = eid_parsed[1]
            if not studio.is_alive_id(eid):
                return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')
            var raw_mat = _first_str_arg(doc, args_node, "material_path", "path", "asset_path")
            if raw_mat.byte_length() == 0:
                return (True, '{"ok":false,"error":"material_path string is required"}')
            var abs_mat: String
            var rel_mat: String
            try:
                var san_mat = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_mat, allow_root_dir=False)
                abs_mat = san_mat[0]
                rel_mat = san_mat[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected material_path: " + String(e)) + "}")
            if not FlintStudioState._ascii_lower(rel_mat).endswith(".flmat"):
                return (True, '{"ok":false,"error":"material_path must end with .flmat"}')

            var ls = _lstat_mode_uid(abs_mat)
            if not ls[0] or (ls[1] & _S_IFMT) != UInt32(0o100000):
                return (True, '{"ok":false,"error":"material file not found"}')
            var raw_bytes = read_bytes(abs_mat)
            if len(raw_bytes) == 0:
                return (True, '{"ok":false,"error":"empty material file"}')
            try:
                _ = load_flmat_bytes(raw_bytes)
            except e:
                return (True, '{"ok":false,"error":' + json_quote("invalid .flmat payload: " + String(e)) + "}")

            if not studio._assign_flmat_to_entity(eid, rel_mat):
                return (True, '{"ok":false,"error":"failed to assign material"}')
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"material_path":' + json_quote(rel_mat)
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 22. prefab.spawn
        if tool_name == "prefab.spawn":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"prefab.spawn requires PIE to be stopped (PIE is active)"}')
            var raw_pf = _first_str_arg(doc, args_node, "prefab_path", "path", "asset_path")
            if raw_pf.byte_length() == 0:
                return (True, '{"ok":false,"error":"prefab_path string is required"}')
            var p_node = doc.object_get(args_node, "position")
            var r_node = doc.object_get(args_node, "rotation")
            var s_node = doc.object_get(args_node, "scale")
            var p_res = _parse_vec3_arg(doc, p_node, Vec3(0.0, 0.0, 0.0))
            var r_res = _parse_quat_arg(doc, r_node, Quat(0.0, 0.0, 0.0, 1.0))
            var s_res = _parse_vec3_arg(doc, s_node, Vec3(1.0, 1.0, 1.0))
            if not (p_res[0] and r_res[0] and s_res[0]):
                return (True, '{"ok":false,"error":"invalid transform argument in prefab.spawn"}')

            var abs_pf: String
            var rel_pf: String
            try:
                var san_pf = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_pf, allow_root_dir=False)
                abs_pf = san_pf[0]
                rel_pf = san_pf[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected prefab_path: " + String(e)) + "}")

            var ls = _lstat_mode_uid(abs_pf)
            if not ls[0] or (ls[1] & _S_IFMT) != UInt32(0o100000):
                return (True, '{"ok":false,"error":"prefab file not found"}')
            var raw_bytes = read_bytes(abs_pf)
            if len(raw_bytes) == 0:
                return (True, '{"ok":false,"error":"empty prefab file"}')
            var pf: Prefab
            try:
                pf = load_flprefab_bytes(raw_bytes)
            except e:
                return (True, '{"ok":false,"error":' + json_quote("invalid .flprefab payload: " + String(e)) + "}")
            if len(pf.nodes) == 0:
                return (True, '{"ok":false,"error":"prefab contains zero nodes"}')

            var before_blob = studio._capture_world_blob()
            var prev_ids = studio.selected_entity_ids.copy()
            var prev_primary = studio.selected_entity_id
            var prev_anchor = studio.selection_anchor_id
            var spawned: List[Entity]
            try:
                spawned = pf.instantiate(studio.app.world, studio.registry)
            except e:
                return (True, '{"ok":false,"error":' + json_quote("prefab instantiate failed: " + String(e)) + "}")
            if len(spawned) == 0:
                return (True, '{"ok":false,"error":"prefab instantiated zero entities"}')

            var pf_ref = AssetRef.from_path(rel_pf, ASSET_KIND_RAW)
            var aff = List[Int]()
            var eids_json = String("[")
            for si in range(len(spawned)):
                var se = spawned[si]
                var seid = Int(se.index)
                aff.append(seid)
                if si > 0:
                    eids_json += ","
                eids_json += String(seid)
                var nname = pf.nodes[si].name if si < len(pf.nodes) else (pf.name + "_" + String(si))
                if not studio.app.world.has[StudioEntityMetaComponent](se):
                    _ = studio.app.world.insert[StudioEntityMetaComponent](se, StudioEntityMetaComponent(name=nname))
                if not studio.app.world.has[StudioStyleComponent](se):
                    _ = studio.app.world.insert[StudioStyleComponent](se, StudioStyleComponent())
                if si < len(pf.nodes) and pf.nodes[si].parent_slot < 0:
                    var lt_opt = studio.app.world.get_ptr_mut[LocalTransform](se)
                    if lt_opt:
                        if p_node >= 0:
                            lt_opt.value()[unsafe_offset=0].translation = p_res[1]
                        if r_node >= 0:
                            lt_opt.value()[unsafe_offset=0].rotation = r_res[1]
                        if s_node >= 0:
                            lt_opt.value()[unsafe_offset=0].scale = s_res[1]
                    else:
                        _ = studio.app.world.insert[LocalTransform](se, LocalTransform(p_res[1], r_res[1], s_res[1]))
                var lt_cur = studio.app.world.get[LocalTransform](se) if studio.app.world.has[LocalTransform](se) else LocalTransform()
                var gt_cur = GlobalTransform.from_local(lt_cur)
                if not studio.app.world.has[GlobalTransform](se):
                    _ = studio.app.world.insert[GlobalTransform](se, gt_cur.copy())
                if not studio.app.world.has[PrevGlobalTransform](se):
                    _ = studio.app.world.insert[PrevGlobalTransform](se, PrevGlobalTransform(gt_cur^))
                if si == 0 and not studio.app.world.has[PrefabInstanceComponent](se):
                    _ = studio.app.world.insert[PrefabInstanceComponent](se, PrefabInstanceComponent(pf.name, pf_ref.copy(), 0))
            eids_json += "]"

            var root_eid = Int(spawned[0].index)
            studio._sync_ecs_and_physics()
            studio._record_world_tx(
                TX_SPAWN_ENTITY, root_eid, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "MCP Spawn Prefab " + pf.name
            )
            return (
                False,
                '{"ok":true,"entity_id":' + String(root_eid)
                + ',"entity_ids":' + eids_json
                + ',"name":' + json_quote(pf.name)
                + ',"prefab_path":' + json_quote(rel_pf)
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 23. physics.trigger_events
        if tool_name == "physics.trigger_events":
            if studio.pie.mode == PIE_STOPPED:
                return (True, '{"ok":false,"error":"physics.trigger_events requires an active PIE session (PIE is stopped)"}')
            var filter_eid = -1
            if args_node >= 0:
                if doc.nodes[args_node].kind != 5:
                    return (True, '{"ok":false,"error":"arguments must be a JSON object"}')
                var eid_node = doc.object_get(args_node, "entity_id")
                if eid_node >= 0:
                    var parsed_eid = _parse_strict_json_int(doc, eid_node)
                    if not parsed_eid[0] or not studio.is_alive_id(parsed_eid[1]):
                        return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')
                    filter_eid = parsed_eid[1]

            studio._capture_pie_trigger_events()
            if self.trigger_event_cursor > studio._pie_trigger_buffer.total_seq:
                self.trigger_event_cursor = 0
            var evs = studio._pie_trigger_buffer.read_since(self.trigger_event_cursor)
            var evs_json = String("[")
            var cnt = 0
            for i in range(len(evs)):
                ref ev = evs[i]
                var trig_eid = ev.trigger_entity_id
                var body_eid = ev.entity_id
                if filter_eid >= 0 and trig_eid != filter_eid and body_eid != filter_eid:
                    continue
                var kind_s = String("unknown")
                if ev.kind == TRIGGER_ENTER:
                    kind_s = "enter"
                elif ev.kind == TRIGGER_STAY:
                    kind_s = "stay"
                elif ev.kind == TRIGGER_EXIT:
                    kind_s = "exit"
                if cnt > 0:
                    evs_json += ","
                evs_json += (
                    '{"kind":' + json_quote(kind_s)
                    + ',"trigger_id":' + String(ev.trigger_id)
                    + ',"body_id":' + String(ev.body_id)
                    + ',"trigger_entity_id":' + String(trig_eid)
                    + ',"entity_id":' + String(body_eid)
                    + ',"point":' + _fmt_vec3_json(ev.point) + "}"
                )
                cnt += 1
            evs_json += "]"
            return (False, '{"ok":true,"count":' + String(cnt) + ',"events":' + evs_json + "}")

        # 24. anim.set_state
        if tool_name == "anim.set_state":
            if studio.pie.mode == PIE_STOPPED:
                return (True, '{"ok":false,"error":"anim.set_state requires an active PIE session (PIE is stopped)"}')
            if args_node < 0 or doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments object is required"}')
            var eid_node = doc.object_get(args_node, "entity_id")
            var parsed_eid = _parse_strict_json_int(doc, eid_node)
            if not parsed_eid[0]:
                return (True, '{"ok":false,"error":"invalid entity_id"}')
            var eid = parsed_eid[1]
            var ent = studio.resolve_entity(eid)
            if not ent.valid():
                return (True, '{"ok":false,"error":"invalid or non-existent entity_id"}')
            var state = _first_str_arg(doc, args_node, "state", "state_name", "clip")
            if state.byte_length() == 0:
                return (True, '{"ok":false,"error":"non-empty state string is required"}')

            var spd_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "speed"), Float32(1.0))
            if not spd_res[0]:
                return (True, '{"ok":false,"error":"speed must be a finite number"}')
            var pl_node = doc.object_get(args_node, "playing")
            if pl_node >= 0 and doc.nodes[pl_node].kind != 1:
                return (True, '{"ok":false,"error":"playing must be a boolean"}')
            var lp_node = doc.object_get(args_node, "loop")
            if lp_node >= 0 and doc.nodes[lp_node].kind != 1:
                return (True, '{"ok":false,"error":"loop must be a boolean"}')
            var playing = doc.boolean(pl_node, True)
            var loop = doc.boolean(lp_node, True)

            if not studio.app.world.has[AnimatorComponent](ent):
                _ = studio.app.world.insert[AnimatorComponent](
                    ent,
                    AnimatorComponent(clip_name=state, playing=playing, loop=loop, speed=spd_res[1]),
                )
            var clip_idx = -1
            var ac_opt = studio.app.world.get_ptr_mut[AnimatorComponent](ent)
            if ac_opt:
                ref ac = ac_opt.value()[unsafe_offset=0]
                ac.clip_name = state
                ac.playing = playing
                ac.loop = loop
                ac.speed = spd_res[1]
                ac.time = Float32(0.0)
                var ac_cache = studio.app.resources.get_ptr[AnimAssetCache]()
                if ac_cache:
                    ref cache = ac_cache.value()[unsafe_offset=0]
                    var st_low = FlintStudioState._ascii_lower(state)
                    var matched_graph_state = False
                    if ac.graph_id != 0:
                        var gi = cache.find_graph(ac.graph_id)
                        if gi >= 0:
                            for si in range(len(cache.graphs[gi].states)):
                                var sname = cache.graphs[gi].states[si].name
                                if sname == state or FlintStudioState._ascii_lower(sname) == st_low:
                                    ac.animator.set_manual_state(si)
                                    matched_graph_state = True
                                    break
                    if not matched_graph_state and ac.animator.configured():
                        for si in range(len(ac.animator.graph.states)):
                            var sname = ac.animator.graph.states[si].name
                            if sname == state or FlintStudioState._ascii_lower(sname) == st_low:
                                ac.animator.set_manual_state(si)
                                matched_graph_state = True
                                break
                    if not matched_graph_state:
                        ac.animator.clear_manual_state()
                    var sk_opt = studio.app.world.get_ptr[SkeletonComponent](ent)
                    if sk_opt and sk_opt.value()[unsafe_offset=0].model_id != 0:
                        var mi = cache.find_model(sk_opt.value()[unsafe_offset=0].model_id)
                        if mi >= 0:
                            ref mdl = cache.models[mi]
                            for ci in range(len(mdl.animations)):
                                var cname = mdl.animations[ci].name
                                if cname == state or FlintStudioState._ascii_lower(cname) == st_low:
                                    ac.clip_index = ci
                                    break
                clip_idx = ac.clip_index
            studio._sync_compat_view()
            return (
                False,
                '{"ok":true,"entity_id":' + String(eid)
                + ',"state":' + json_quote(state)
                + ',"playing":' + ("true" if playing else "false")
                + ',"clip_index":' + String(clip_idx) + "}",
            )

        # 25. audio.play
        if tool_name == "audio.play":
            if studio.pie.mode == PIE_STOPPED:
                return (True, '{"ok":false,"error":"audio.play requires an active PIE session (PIE is stopped)"}')
            if args_node < 0 or doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments object is required"}')
            var raw_path = _first_str_arg(doc, args_node, "asset_path", "path", "sound")
            if raw_path.byte_length() == 0:
                raw_path = _first_str_arg(doc, args_node, "clip")
            if raw_path.byte_length() == 0:
                return (True, '{"ok":false,"error":"asset_path is required"}')
            var abs_path: String
            var rel_path: String
            try:
                var san = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_path, allow_root_dir=False)
                abs_path = san[0]
                rel_path = san[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote(String(e)) + "}")

            var st = _lstat_mode_uid(abs_path)
            if not st[0] or (st[1] & _S_IFMT) != UInt32(0o100000):
                return (True, '{"ok":false,"error":"audio file does not exist or is not a regular file"}')
            var lower_rel = FlintStudioState._ascii_lower(rel_path)
            if not (lower_rel.endswith(".wav") or lower_rel.endswith(".qoa")):
                return (True, '{"ok":false,"error":"audio.play supports only .wav and .qoa files"}')

            var vol_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "volume"), Float32(1.0))
            if not vol_res[0]:
                return (True, '{"ok":false,"error":"volume must be a finite number"}')
            var pch_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "pitch"), Float32(1.0))
            if not pch_res[0]:
                return (True, '{"ok":false,"error":"pitch must be a finite number"}')
            var lp_node = doc.object_get(args_node, "looping")
            if lp_node >= 0 and doc.nodes[lp_node].kind != 1:
                return (True, '{"ok":false,"error":"looping must be a boolean"}')
            var looping = doc.boolean(lp_node, False)
            var pos_node = doc.object_get(args_node, "position")
            var pos_res = _parse_vec3_arg(doc, pos_node, Vec3())
            if not pos_res[0]:
                return (True, '{"ok":false,"error":"position must be a 3-element finite numeric array"}')

            var aud_opt = studio.app.resources.get_ptr_mut[AudioServer]()
            if not aud_opt:
                return (True, '{"ok":false,"error":"AudioServer resource unavailable"}')
            ref aud_srv = aud_opt.value()[unsafe_offset=0]
            var aid = aud_srv.load_audio_cached(abs_path)
            if aid < 0:
                return (True, '{"ok":false,"error":"failed to decode audio file"}')
            var vol = max(Float32(0.0), min(Float32(4.0), vol_res[1]))
            var pch = max(Float32(0.1), min(Float32(4.0), pch_res[1]))
            var h = (
                aud_srv.play_3d(aid, pos_res[1], Vec3(), vol, pch, BUS_SFX, looping)
                if pos_node >= 0
                else aud_srv.play_2d(aid, vol, pch, Float32(0.0), BUS_SFX, looping)
            )
            return (
                False,
                '{"ok":true,"asset_path":' + json_quote(rel_path)
                + ',"asset_id":' + String(aid)
                + ',"slot":' + String(Int(h.slot))
                + ',"generation":' + String(Int(h.generation)) + "}",
            )

        # 26. nav.query_path
        if tool_name == "nav.query_path":
            if studio.pie.mode == PIE_STOPPED:
                return (True, '{"ok":false,"error":"nav.query_path requires an active PIE session (PIE is stopped)"}')
            if args_node < 0 or doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments object is required"}')
            var start_node = doc.object_get(args_node, "start")
            var goal_node = doc.object_get(args_node, "goal")
            if start_node < 0 or goal_node < 0:
                return (True, '{"ok":false,"error":"start and goal 3-element numeric arrays are required"}')
            var start_res = _parse_vec3_arg(doc, start_node, Vec3())
            var goal_res = _parse_vec3_arg(doc, goal_node, Vec3())
            if not start_res[0] or not goal_res[0]:
                return (True, '{"ok":false,"error":"start and goal must be 3-element finite numeric arrays"}')
            var sr_res = _parse_finite_f32_arg(doc, doc.object_get(args_node, "search_radius"), Float32(10.0))
            if not sr_res[0] or sr_res[1] <= Float32(0.0):
                return (True, '{"ok":false,"error":"search_radius must be a positive finite number"}')

            var nav_opt = studio.app.resources.get_ptr[NavPluginStateResource]()
            if not nav_opt or not nav_opt.value()[unsafe_offset=0].baked or nav_opt.value()[unsafe_offset=0].navmesh.poly_count() == 0:
                _ = studio.bake_navmesh()
                nav_opt = studio.app.resources.get_ptr[NavPluginStateResource]()
            if not nav_opt:
                return (True, '{"ok":false,"error":"NavPluginStateResource unavailable"}')

            ref nav_state = nav_opt.value()[unsafe_offset=0]
            var path = nav_state.navmesh.find_path(start_res[1], goal_res[1], max(Float32(0.1), sr_res[1]))
            var wps_json = String("[")
            for wi in range(len(path.waypoints)):
                if wi > 0:
                    wps_json += ","
                wps_json += _fmt_vec3_json(path.waypoints[wi])
            wps_json += "]"
            return (
                False,
                '{"ok":true,"found":' + ("true" if path.found else "false")
                + ',"total_length":' + _fmt_json_f64(Float64(path.total_length))
                + ',"poly_count":' + String(nav_state.navmesh.poly_count())
                + ',"waypoints":' + wps_json + "}",
            )

        # 27. project.package
        if tool_name == "project.package":
            if studio.pie.mode != PIE_STOPPED:
                return (True, '{"ok":false,"error":"project.package requires PIE to be stopped (PIE is active)"}')
            if args_node >= 0 and doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments must be a JSON object"}')

            var raw_out = String("dist/package")
            var out_node = doc.object_get(args_node, "output_dir") if args_node >= 0 else -1
            if out_node < 0 and args_node >= 0:
                out_node = doc.object_get(args_node, "out_dir")
            if out_node >= 0:
                if doc.nodes[out_node].kind != 3:
                    return (True, '{"ok":false,"error":"output_dir must be a string"}')
                raw_out = doc.text(out_node, "")
                if raw_out.byte_length() == 0:
                    return (True, '{"ok":false,"error":"output_dir must be non-empty"}')

            var abs_out: String
            var rel_out: String
            try:
                var san = _sanitize_workspace_rel_and_abs(self.workspace_root, raw_out, allow_root_dir=False)
                abs_out = san[0]
                rel_out = san[1]
            except e:
                return (True, '{"ok":false,"error":' + json_quote("sandbox rejected output_dir: " + String(e)) + "}")

            var ls = _lstat_mode_uid(abs_out)
            if ls[0] and (ls[1] & _S_IFMT) != _S_IFDIR:
                return (True, '{"ok":false,"error":"output_dir exists and is not a directory"}')

            var profile = String("")
            var prof_node = doc.object_get(args_node, "profile") if args_node >= 0 else -1
            if prof_node >= 0:
                if doc.nodes[prof_node].kind != 3:
                    return (True, '{"ok":false,"error":"profile must be a string"}')
                profile = doc.text(prof_node, "")
                if profile.byte_length() > 0 and not FlintStudioState._is_valid_profile_name(profile):
                    return (True, '{"ok":false,"error":"invalid profile name"}')

            var loose_assets = False
            var loose_node = doc.object_get(args_node, "loose_assets") if args_node >= 0 else -1
            if loose_node >= 0:
                if doc.nodes[loose_node].kind != 1:
                    return (True, '{"ok":false,"error":"loose_assets must be a boolean"}')
                loose_assets = doc.boolean(loose_node, False)

            var target = String("")
            var tgt_node = doc.object_get(args_node, "target") if args_node >= 0 else -1
            if tgt_node >= 0:
                if doc.nodes[tgt_node].kind != 3:
                    return (True, '{"ok":false,"error":"target must be a string"}')
                target = doc.text(tgt_node, "")
                if target.byte_length() > 0:
                    if profile.byte_length() > 0:
                        return (True, '{"ok":false,"error":"cannot specify both profile and target"}')
                    if _has_forbidden_path_bytes(target) or target.startswith("-") or target.startswith("/") or target.find("..") >= 0:
                        return (True, '{"ok":false,"error":"invalid target"}')
                    var is_shorthand = (
                        target == "spark"
                        or target == "world"
                        or target == "studio"
                        or target == "cycles"
                        or target == "kms"
                        or target == "kms3d"
                    )
                    if not is_shorthand:
                        try:
                            _ = _sanitize_workspace_rel_and_abs(self.workspace_root, target, allow_root_dir=False)
                        except e:
                            return (True, '{"ok":false,"error":' + json_quote("invalid target: " + String(e)) + "}")

            var res = studio.package_project(
                output_dir=abs_out,
                profile=profile,
                loose_assets=loose_assets,
                target=target,
            )
            if not res[0]:
                return (True, '{"ok":false,"error":' + json_quote(res[1]) + "}")

            var manifest_path = abs_out + "/MANIFEST.txt"
            var run_sh_path = abs_out + "/run.sh"
            var pack_name = _resolve_packaged_flpk_name(manifest_path)
            var flpk_path = (abs_out + "/" + pack_name) if pack_name.byte_length() > 0 else String("")
            var has_flpk = flpk_path.byte_length() > 0 and file_size(flpk_path) > 0
            return (
                False,
                '{"ok":true,"output_dir":' + json_quote(rel_out)
                + ',"abs_output_dir":' + json_quote(abs_out)
                + ',"run_sh":' + json_quote(run_sh_path)
                + ',"has_assets_flpk":' + ("true" if has_flpk else "false")
                + ',"assets_flpk":' + json_quote(flpk_path if has_flpk else String(""))
                + ',"manifest":' + json_quote(manifest_path if file_size(manifest_path) > 0 else String(""))
                + ',"profile":' + json_quote(profile)
                + ',"loose_assets":' + ("true" if loose_assets else "false")
                + ',"state_hash":' + json_quote(String(studio.state_hash())) + "}",
            )

        # 28. code.build_hot_reload
        if tool_name == "code.build_hot_reload":
            if args_node >= 0 and doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments must be a JSON object"}')

            var reload_only = False
            if args_node >= 0:
                var ro_node = doc.object_get(args_node, "reload_only")
                if ro_node >= 0:
                    if doc.nodes[ro_node].kind != 1:
                        return (True, '{"ok":false,"error":"reload_only must be a boolean"}')
                    reload_only = doc.boolean(ro_node, False)

            var raw_entry = _first_str_arg(doc, args_node, "entry_path", "entry", "source_path")
            if raw_entry.byte_length() == 0:
                raw_entry = String("src/gameplay.mojo")
            var raw_so = _first_str_arg(doc, args_node, "output_so", "so_path", "output_path")
            if raw_so.byte_length() == 0:
                raw_so = String("build/gameplay.so")

            var abs_entry: String
            var abs_so: String
            var rel_so: String
            try:
                abs_entry = _sanitize_workspace_rel_and_abs(
                    self.workspace_root, raw_entry, allow_root_dir=False
                )[0]
                var so_san = _sanitize_workspace_rel_and_abs(
                    self.workspace_root, raw_so, allow_root_dir=False
                )
                abs_so = so_san[0]
                rel_so = so_san[1]
            except e:
                return (
                    True,
                    '{"ok":false,"error":' + json_quote("sandbox rejected path: " + String(e)) + "}",
                )

            var opt_level = _first_str_arg(doc, args_node, "opt_level")
            if opt_level.byte_length() == 0:
                opt_level = String("-O1")
            if opt_level != "-O0" and opt_level != "-O1" and opt_level != "-O2" and opt_level != "-O3":
                return (True, '{"ok":false,"error":"opt_level must be one of -O0, -O1, -O2, -O3"}')

            var compiled = False
            var build_ok = True
            var build_output = String("")
            var build_diags = List[Diagnostic]()

            if not reload_only:
                if file_size(abs_entry) <= 0:
                    return (
                        True,
                        '{"ok":false,"compiled":false,"reloaded":false,"error":'
                        + json_quote("entry file not found: " + raw_entry)
                        + "}",
                    )
                var so_parent = parent_directory(abs_so)
                if so_parent.byte_length() > 0 and so_parent != ".":
                    _ = make_dirs(so_parent)
                var compiler = _first_str_arg(doc, args_node, "compiler")
                if compiler.byte_length() == 0:
                    compiler = env_now("FLINT_MOJO_BIN")
                if compiler.byte_length() == 0:
                    compiler = String("mojo")
                if _has_forbidden_path_bytes(compiler):
                    return (True, '{"ok":false,"error":"invalid compiler path"}')

                var inc = List[String]()
                var proj_root = self.workspace_root
                try:
                    proj_root = _resolve_workspace_root(self.workspace_root)[0]
                except:
                    pass
                inc.append(proj_root + "/src")
                if is_directory("src") and proj_root + "/src" != _normalize_abs_lexical(_getcwd_str() + "/src"):
                    inc.append("src")

                var runner = JobRunner()
                var jid = runner.start_gameplay_so_build(
                    proj_root, abs_entry, abs_so, inc, compiler=compiler, opt_flag=opt_level, werror=True
                )
                if jid < 0:
                    return (True, '{"ok":false,"compiled":false,"reloaded":false,"error":"failed to start mojo build"}')
                var deadline = get_time() + 60.0
                var finished = False
                while get_time() < deadline:
                    var done_ids = runner.poll_all()
                    if len(done_ids) > 0:
                        finished = True
                        break
                    _ = external_call["usleep", Int32](UInt32(10000))
                if not finished:
                    _ = runner.cancel(jid)
                    return (True, '{"ok":false,"compiled":false,"reloaded":false,"error":"gameplay.so build timed out after 60s"}')
                var jidx = runner.find_index(jid)
                if jidx >= 0:
                    build_ok = runner.jobs[jidx].state == JOB_SUCCEEDED
                    build_output = runner.jobs[jidx].output_text()
                    build_diags = runner.jobs[jidx].diagnostics.copy()
                compiled = build_ok
                self.diagnostics_store.replace_source(DIAG_SRC_BUILD, build_diags)

            var err_cnt = 0
            var warn_cnt = 0
            var diags_json = String("[")
            for di in range(len(build_diags)):
                ref d = build_diags[di]
                if d.severity == DIAG_ERROR:
                    err_cnt += 1
                elif d.severity == DIAG_WARNING:
                    warn_cnt += 1
                if di > 0:
                    diags_json += ","
                diags_json += (
                    '{"severity":'
                    + json_quote(severity_label(d.severity))
                    + ',"path":'
                    + json_quote(d.path)
                    + ',"line":'
                    + String(d.line)
                    + ',"col":'
                    + String(d.col)
                    + ',"message":'
                    + json_quote(d.message)
                    + "}"
                )
            diags_json += "]"

            if not build_ok:
                return (
                    True,
                    '{"ok":false,"compiled":false,"reloaded":false,"so_path":'
                    + json_quote(rel_so)
                    + ',"error_count":'
                    + String(err_cnt)
                    + ',"warning_count":'
                    + String(warn_cnt)
                    + ',"diagnostics":'
                    + diags_json
                    + ',"output":'
                    + json_quote(build_output)
                    + "}",
                )

            var reloaded = studio.load_gameplay_plugin(abs_so)
            if not reloaded:
                return (
                    True,
                    '{"ok":false,"compiled":'
                    + ("true" if compiled else "false")
                    + ',"reloaded":false,"so_path":'
                    + json_quote(rel_so)
                    + ',"error":'
                    + json_quote(studio.hot_reload_host.last_error)
                    + ',"error_count":'
                    + String(err_cnt)
                    + ',"warning_count":'
                    + String(warn_cnt)
                    + ',"diagnostics":'
                    + diags_json
                    + "}",
                )

            return (
                False,
                '{"ok":true,"compiled":'
                + ("true" if compiled else "false")
                + ',"reloaded":true,"so_path":'
                + json_quote(rel_so)
                + ',"reload_generation":'
                + String(studio.hot_reload_host.reload_generation)
                + ',"reload_ms":'
                + _fmt_json_f64(studio.hot_reload_host.last_reload_ms)
                + ',"error_count":'
                + String(err_cnt)
                + ',"warning_count":'
                + String(warn_cnt)
                + ',"diagnostics":'
                + diags_json
                + ',"state_hash":'
                + json_quote(String(studio.state_hash()))
                + "}",
            )

        # 29. code.diagnostics
        if tool_name == "code.diagnostics":
            if args_node >= 0 and doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments must be a JSON object"}')
            var file_filter = _first_str_arg(doc, args_node, "file", "path")
            var sev_filter = _first_str_arg(doc, args_node, "severity")
            var max_items = 100
            if args_node >= 0:
                var mn = doc.object_get(args_node, "max_items")
                if mn >= 0:
                    var mi = _parse_strict_json_int(doc, mn)
                    if not mi[0] or mi[1] <= 0:
                        return (True, '{"ok":false,"error":"max_items must be a positive integer"}')
                    max_items = min(500, mi[1])

            var pf = ProblemFilter()
            if sev_filter == "error":
                pf.warnings = False
                pf.infos = False
            elif sev_filter == "warning":
                pf.errors = False
                pf.infos = False
            elif sev_filter == "info" or sev_filter == "note":
                pf.errors = False
                pf.warnings = False
            var matched = self.diagnostics_store.query(pf, file_filter)

            var items_json = String("[")
            var emitted = 0
            var errs = 0
            var warns = 0
            for i in range(len(matched)):
                ref d = matched[i]
                if file_filter.byte_length() > 0 and d.path != file_filter and not d.path.endswith(file_filter):
                    continue
                if d.severity == DIAG_ERROR:
                    errs += 1
                elif d.severity == DIAG_WARNING:
                    warns += 1
                if emitted < max_items:
                    if emitted > 0:
                        items_json += ","
                    var notes_json = String("[")
                    for ni in range(len(d.notes)):
                        if ni > 0:
                            notes_json += ","
                        notes_json += json_quote(d.notes[ni])
                    notes_json += "]"
                    items_json += (
                        '{"source":'
                        + json_quote(source_label(d.source))
                        + ',"severity":'
                        + json_quote(severity_label(d.severity))
                        + ',"path":'
                        + json_quote(d.path)
                        + ',"line":'
                        + String(d.line)
                        + ',"col":'
                        + String(d.col)
                        + ',"message":'
                        + json_quote(d.message)
                        + ',"notes":'
                        + notes_json
                        + "}"
                    )
                    emitted += 1
            items_json += "]"
            return (
                False,
                '{"ok":true,"errors":'
                + String(errs)
                + ',"warnings":'
                + String(warns)
                + ',"total":'
                + String(emitted)
                + ',"items":'
                + items_json
                + "}",
            )

        # 30. nav.bake
        if tool_name == "nav.bake":
            if args_node >= 0 and doc.nodes[args_node].kind != 5:
                return (True, '{"ok":false,"error":"arguments must be a JSON object"}')
            var poly_count = studio.bake_navmesh()
            return (
                False,
                '{"ok":true,"polygon_count":'
                + String(poly_count)
                + ',"state_hash":'
                + json_quote(String(studio.state_hash()))
                + "}",
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
            var args_str = json_text(doc, args_node) if args_node >= 0 else String("{}")
            var mutating = mcp_tool_is_mutating(tname)
            var ev_seq = self.next_event_seq
            self.next_event_seq += 1

            if not self._consume_gate_allowance(tname, mutating):
                var blocked_json = (
                    '{"ok":false,"approval_required":true,"gate_seq":'
                    + String(ev_seq)
                    + ',"tool":'
                    + json_quote(tname)
                    + ',"message":"Tool call requires user approval in Studio Agent Mission Control"}'
                )
                var bev = McpToolEvent(
                    seq=ev_seq,
                    tool_name=tname,
                    params_json=args_str,
                    result_json=blocked_json,
                    elapsed_ms=0.0,
                    ok=False,
                    is_mutating=mutating,
                    blocked_by_gate=True,
                )
                bev.affected_entities = _extract_affected_entities(doc, args_node, "")
                self._record_tool_event(bev^)
                return _jsonrpc_ok(id_raw, _mcp_tool_result(blocked_json, is_error=True))

            var t0 = get_time()
            var outcome = self._dispatch_tool_call(
                tname, doc, args_node, studio, assets, pool, has_renderer, ren_ptr
            )
            var elapsed_ms = max(0.0, (get_time() - t0) * 1000.0)
            var ev = McpToolEvent(
                seq=ev_seq,
                tool_name=tname,
                params_json=args_str,
                result_json=outcome[1],
                elapsed_ms=elapsed_ms,
                ok=not outcome[0],
                is_mutating=mutating,
                blocked_by_gate=False,
            )
            ev.affected_entities = _extract_affected_entities(doc, args_node, outcome[1])
            self._record_tool_event(ev^)
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
                    _ = send_span_bounded(1, Span(err_bytes), timeout_ms=2000, is_socket=False)
                    fatal_err = True
                    break
                if not frame.has_frame:
                    break
                var resp = self._handle_request_bytes(frame.payload, studio, assets, pool, False, 0)
                if resp.byte_length() > 0:
                    var out_bytes = encode_mcp_frame(resp, use_content_length=frame.used_content_length)
                    if not send_span_bounded(1, Span(out_bytes), timeout_ms=2000, is_socket=False):
                        fatal_err = True
                        break
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


@no_inline
def _bind_unix(fd: Int32, addr: List[UInt8]) -> Int32:
    return external_call["bind", Int32](fd, addr.unsafe_ptr(), UInt32(110))


@no_inline
def _connect_unix(fd: Int32, addr: List[UInt8]) -> Int32:
    return external_call["connect", Int32](fd, addr.unsafe_ptr(), UInt32(110))


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
        var brc = _bind_unix(Int32(fd), addr)
        _ = external_call["umask", UInt32](old_umask)
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
                    _ = send_span_bounded(cfd, Span(err_b), timeout_ms=500, is_socket=True)
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
                    var ok_send = send_span_bounded(cfd, Span(out_b), timeout_ms=2000, is_socket=True)
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
        var rc = _connect_unix(Int32(fd), addr)
        if rc != 0:
            close_fd(fd)
            return False
        self.fd = fd
        return True

    def send_raw_bytes(mut self, raw: List[UInt8], timeout_ms: Int = 2000) -> Bool:
        if self.fd < 0 or len(raw) == 0:
            return False
        return send_span_bounded(self.fd, Span(raw), timeout_ms=timeout_ms, is_socket=True)

    def send_raw_request(mut self, req_json: String, timeout_ms: Int = 2000) -> Bool:
        if self.fd < 0:
            return False
        var frame = encode_mcp_frame(req_json, use_content_length=self.use_content_length)
        return send_span_bounded(self.fd, Span(frame), timeout_ms=timeout_ms, is_socket=True)

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
                return (True, got[1])
            var rem_ms = Int((deadline - get_time()) * 1000.0 + 0.999)
            if rem_ms <= 0:
                break
            _write_pollfd_bytes(pfd, Int32(self.fd), Int16(0x0001))  # POLLIN
            var prc = external_call["poll", Int32](pfd.unsafe_ptr(), UInt64(1), Int32(min(rem_ms, 25)))
            if prc < 0 and last_errno() != _EINTR:
                break
            if prc > 0:
                var rev = _read_pollfd_revents(pfd)
                if (rev & _POLLERR_MASK) != 0 and (rev & Int16(0x0001)) == 0:
                    return self.try_recv_response()
        return self.try_recv_response()

    def close(mut self):
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1


def _write_two_pollfds(mut pfds: List[UInt8], fd0: Int32, ev0: Int16, fd1: Int32, ev1: Int16):
    var u0 = UInt32(fd0)
    pfds[0] = UInt8(u0 & 0xFF)
    pfds[1] = UInt8((u0 >> 8) & 0xFF)
    pfds[2] = UInt8((u0 >> 16) & 0xFF)
    pfds[3] = UInt8((u0 >> 24) & 0xFF)
    var e0 = UInt16(ev0)
    pfds[4] = UInt8(e0 & 0xFF)
    pfds[5] = UInt8((e0 >> 8) & 0xFF)
    pfds[6] = 0
    pfds[7] = 0

    var u1 = UInt32(fd1)
    pfds[8] = UInt8(u1 & 0xFF)
    pfds[9] = UInt8((u1 >> 8) & 0xFF)
    pfds[10] = UInt8((u1 >> 16) & 0xFF)
    pfds[11] = UInt8((u1 >> 24) & 0xFF)
    var e1 = UInt16(ev1)
    pfds[12] = UInt8(e1 & 0xFF)
    pfds[13] = UInt8((e1 >> 8) & 0xFF)
    pfds[14] = 0
    pfds[15] = 0


def run_mcp_connect_stdio(sock_path: String) -> Int:
    """Pure-Mojo `stdio <-> AF_UNIX` MCP proxy (`flint-studio --mcp-connect <sock>`).

    Connects stdin/stdout directly to a live Studio `FlintMcpUnixListener` socket so
    in-Studio ACP agents can invoke live Studio MCP v2 tools with zero Python wrappers
    and automatic `SO_PEERCRED` UID authentication.
    """
    var target = sock_path if sock_path.byte_length() > 0 else default_mcp_socket_path()
    if target.byte_length() == 0 or target.byte_length() > 107 or _has_forbidden_path_bytes(target):
        return 1

    var sock_fd = Int(external_call["socket", Int32](_AF_UNIX, _SOCK_STREAM | _SOCK_CLOEXEC, Int32(0)))
    if sock_fd < 0:
        return 1
    var addr = _build_sockaddr_un(target)
    if _connect_unix(Int32(sock_fd), addr) != 0:
        close_fd(sock_fd)
        return 1

    comptime _POLLIN: Int16 = 0x0001
    var pfds = List[UInt8](length=16, fill=0)
    var scratch = List[UInt8](length=16384, fill=0)
    var rc_out = 0

    while True:
        _write_two_pollfds(pfds, Int32(0), _POLLIN, Int32(sock_fd), _POLLIN)
        var prc = external_call["poll", Int32](pfds.unsafe_ptr(), UInt64(2), Int32(250))
        if prc < 0:
            if last_errno() == _EINTR:
                continue
            rc_out = 1
            break
        if prc == 0:
            continue

        var rev0 = Int16(UInt16(pfds[6]) | (UInt16(pfds[7]) << 8))
        var rev1 = Int16(UInt16(pfds[14]) | (UInt16(pfds[15]) << 8))

        if (rev0 & _POLLIN) != 0:
            var n0 = Int(external_call["read", Int64](Int32(0), scratch.unsafe_ptr(), Int64(16384)))
            if n0 > 0:
                if not send_span_bounded(sock_fd, Span(scratch)[0:n0], timeout_ms=5000, is_socket=True):
                    rc_out = 1
                    break
            elif n0 == 0:
                break
            elif last_errno() != _EINTR and last_errno() != _EAGAIN:
                rc_out = 1
                break

        if (rev1 & _POLLIN) != 0:
            var n1 = Int(external_call["read", Int64](Int32(sock_fd), scratch.unsafe_ptr(), Int64(16384)))
            if n1 > 0:
                if not send_span_bounded(1, Span(scratch)[0:n1], timeout_ms=5000, is_socket=False):
                    rc_out = 1
                    break
            elif n1 == 0:
                break
            elif last_errno() != _EINTR and last_errno() != _EAGAIN:
                rc_out = 1
                break

        if ((rev0 & _POLLERR_MASK) != 0 and (rev0 & _POLLIN) == 0) or (
            (rev1 & _POLLERR_MASK) != 0 and (rev1 & _POLLIN) == 0
        ):
            break

    close_fd(sock_fd)
    return rc_out
