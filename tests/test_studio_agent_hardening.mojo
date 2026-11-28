"""Regression suite for the Studio AI-agent / process-layer hardening (`tests/test_studio_agent_hardening.mojo`).

Covers, headless and without a live agent:
  1. ACP client: responses with null / string / unknown ids never match a pending request,
     `initialize` advertises `terminal:false`, `terminal/*` gets method-not-found, malformed
     `fs/write_text_file` gets invalid-params, `fs/read_text_file` surfaces as `AE_SERVICE`.
  2. MCP framing: a garbage line costs one `-32700` (id null) and the next frame still parses;
     several frames whose total exceeds 4 MiB parse (the cap is per message); batches are refused.
  3. MCP lifecycle: `require_initialize` refuses tool calls before `initialize`; protocol version
     negotiation; `listChanged:false`; `-32001` for auth failures; per-client turn allowance.
  4. Jobs: a non-zero exit is FAILED, ECHILD is never success, finished jobs are pruned, cancel
     never blocks and kills the whole process group.
  5. AgentBus: identical gated MCP calls coalesce, the per-agent queue is capped, thought text
     is capped, "approve for turn" ends with the turn.
  6. safe_fs + revert guard: symlinks / `..` / hidden paths are refused; a revert never clobbers a
     file that changed after the agent wrote it.
  7. MCP deferred items: `structuredContent` / `outputSchema` only for clients that negotiated
     `2025-06-18`; socket-client cooks (even `force_recook`) run in a child process; `asset.import`
     writes durably and never through a symlink; the Unix listener unlinks only stale sockets and
     only its own socket on close.
"""
from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_false, assert_true

from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.devtools.acp_client import ACP_READY, AE_SERVICE, AcpClient
from flint.devtools.job_runner import (
    JOB_FAILED,
    JOB_KIND_TASK,
    MAX_FINISHED_JOBS,
    JobRunner,
    reap_child,
)
from flint.devtools.mcp_v2 import (
    MCP_ERR_NOT_INITIALIZED,
    MCP_ERR_UNAUTHORIZED,
    MCP_GATE_ALL,
    MCP_PROTOCOL_LATEST,
    MCP_PROTOCOL_VERSION,
    MAX_MCP_MESSAGE_BYTES,
    MCP_GATE_AUTO,
    FlintMcpUnixListener,
    McpServerV2,
    mcp_version_has_structured_content,
    negotiate_mcp_protocol_version,
    parse_next_mcp_frame,
)
from flint.devtools.safe_fs import (
    SAFE_FS_MISSING,
    SAFE_FS_OK,
    read_agent_file,
    resolve_confined_path,
    text_hash64,
    write_confined_durable,
)
from flint.platform.os import file_size, get_time, make_dirs, read_bytes, write_bytes
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD
from flint.studio.agent_bus import (
    APPROVAL_SRC_MCP_TOOL,
    MAX_PENDING_APPROVALS_PER_AGENT,
    MAX_THOUGHT_SUMMARY_BYTES,
    AgentTelemetryBus,
    PendingApproval,
    ROLE_GAMEPLAY_CODER,
)
from flint.studio.studio_app import FlintStudioState


def _bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _joined(log: List[String]) -> String:
    var s = String("")
    for i in range(len(log)):
        s += log[i] + "\n"
    return s^


def _tmp_dir(tag: String) -> String:
    var pid = Int(external_call["getpid", Int32]())
    var d = "/tmp/flint_agent_hardening_" + tag + "_" + String(pid)
    _ = make_dirs(d)
    return d


# ---------------------------------------------------------------------------
# 1. ACP protocol
# ---------------------------------------------------------------------------

def test_acp_ids_terminal_and_fs_requests() raises:
    print("  [1/7] ACP ids, terminal:false, fs/* and terminal/* requests...")
    var acp = AcpClient()
    acp.start_headless(".", 0.0)
    var sent = _joined(acp.take_outbound_log())
    assert_true(sent.find('"method":"initialize"') >= 0, "initialize must be sent: " + sent)
    assert_true(sent.find('"terminal":false') >= 0, "initialize must advertise terminal:false: " + sent)

    # Responses whose id is null, a string or unknown must not complete `initialize`.
    acp.feed_text('{"jsonrpc":"2.0","id":null,"result":{"protocolVersion":1}}\n')
    acp.feed_text('{"jsonrpc":"2.0","id":"1","result":{"protocolVersion":1}}\n')
    acp.feed_text('{"jsonrpc":"2.0","id":77,"result":{"protocolVersion":1}}\n')
    acp.feed_text('{"jsonrpc":"2.0","id":2,"result":{"sessionId":"too-early"}}\n')
    acp.pump(0.1)
    assert_true(acp.state != ACP_READY, "null/string/unknown ids must never match a pending request")
    sent = _joined(acp.take_outbound_log())
    assert_true(sent.find('"session/new"') < 0, "session/new must wait for the real initialize response")

    acp.feed_text('{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1}}\n')
    acp.pump(0.2)
    acp.feed_text('{"jsonrpc":"2.0","id":2,"result":{"sessionId":"hardening-1"}}\n')
    acp.pump(0.3)
    assert_equal(acp.state, ACP_READY)
    _ = acp.take_events()
    _ = acp.take_outbound_log()

    # terminal/* is not offered: the agent gets method-not-found, not silence.
    acp.feed_text('{"jsonrpc":"2.0","id":40,"method":"terminal/create","params":{"sessionId":"hardening-1","command":"ls"}}\n')
    acp.pump(0.4)
    sent = _joined(acp.take_outbound_log())
    assert_true(sent.find('"id":40') >= 0 and sent.find("-32601") >= 0, "terminal/create must get -32601: " + sent)

    # fs/write_text_file with a non-string content is answered with invalid params.
    acp.feed_text('{"jsonrpc":"2.0","id":41,"method":"fs/write_text_file","params":{"sessionId":"hardening-1","path":"a.txt","content":5}}\n')
    acp.pump(0.5)
    sent = _joined(acp.take_outbound_log())
    assert_true(sent.find('"id":41') >= 0 and sent.find('"error"') >= 0, "malformed fs/write_text_file must be answered with an error: " + sent)

    # fs/read_text_file is surfaced to Studio (which reads it sandboxed) and stays open until answered.
    acp.feed_text('{"jsonrpc":"2.0","id":42,"method":"fs/read_text_file","params":{"sessionId":"hardening-1","path":"src/x.mojo","line":3,"limit":2}}\n')
    acp.pump(0.6)
    var evs = acp.take_events()
    var found = False
    for i in range(len(evs)):
        if evs[i].kind == AE_SERVICE and evs[i].path == "src/x.mojo":
            found = True
            assert_true(acp.is_open_service(evs[i].request_id), "read request must stay open until answered")
            assert_true(acp.answer_service(evs[i].request_id, '{"content":""}'))
            assert_false(acp.is_open_service(evs[i].request_id))
            assert_false(acp.answer_service(evs[i].request_id, "{}"), "a request is answered exactly once")
    assert_true(found, "fs/read_text_file must surface as AE_SERVICE")


# ---------------------------------------------------------------------------
# 2. MCP framing
# ---------------------------------------------------------------------------

def test_mcp_framing_recovery_and_per_message_cap() raises:
    print("  [2/7] MCP framing: bad line recovery, per-message cap, batches...")
    var buf = _bytes('this is not json\n{"jsonrpc":"2.0","id":5,"method":"ping"}\n')
    var r1 = parse_next_mcp_frame(buf)
    assert_true(r1.protocol_error and r1.recoverable, "a garbage line is a recoverable -32700")
    assert_equal(r1.error_code, -32700)
    var r2 = parse_next_mcp_frame(buf)
    assert_true(r2.has_frame and String(unsafe_from_utf8=r2.payload).find('"id":5') >= 0, "next frame must still parse")

    # Two 3 MiB newline frames buffered together (6 MiB > 4 MiB) both parse.
    var big = List[UInt8](capacity=6 * 1024 * 1024 + 64)
    for _ in range(2):
        for b in '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"pad":"'.as_bytes():
            big.append(b)
        for _ in range(3 * 1024 * 1024):
            big.append(65)
        for b in '"}}\n'.as_bytes():
            big.append(b)
    assert_true(len(big) > MAX_MCP_MESSAGE_BYTES)
    var f1 = parse_next_mcp_frame(big)
    var f2 = parse_next_mcp_frame(big)
    assert_true(f1.has_frame and f2.has_frame and not f1.protocol_error, "the 4 MiB cap applies per message")
    assert_equal(len(big), 0)

    # A single line over the cap is fatal.
    var huge = List[UInt8](length=MAX_MCP_MESSAGE_BYTES + 16, fill=65)
    huge[0] = 123
    var f3 = parse_next_mcp_frame(huge)
    assert_true(f3.protocol_error and not f3.recoverable, "an over-cap message is fatal")


# ---------------------------------------------------------------------------
# 3. MCP lifecycle, negotiation, auth codes, turn allowance
# ---------------------------------------------------------------------------

def test_mcp_lifecycle_negotiation_auth_and_turn_allowance() raises:
    print("  [3/7] MCP initialize lifecycle, version negotiation, -32001, turn allowance...")
    assert_equal(negotiate_mcp_protocol_version(""), String(MCP_PROTOCOL_VERSION))
    assert_equal(negotiate_mcp_protocol_version("2024-11-05"), String("2024-11-05"))
    assert_equal(negotiate_mcp_protocol_version("2025-06-18"), String("2025-06-18"))
    # Between the two supported revisions: the newest one both speak is the older one.
    assert_equal(negotiate_mcp_protocol_version("2025-03-26"), String("2024-11-05"))
    # Newer than anything we speak: offer our latest.
    assert_equal(negotiate_mcp_protocol_version("2099-01-01"), String(MCP_PROTOCOL_LATEST))
    assert_equal(negotiate_mcp_protocol_version("1999-01-01"), String(MCP_PROTOCOL_LATEST))
    assert_false(mcp_version_has_structured_content("2024-11-05"))
    assert_true(mcp_version_has_structured_content("2025-06-18"))

    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=200, worker_threads=0)
    var assets = AssetServer()
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=_tmp_dir("mcp"))
    mcp.require_initialize = True
    var pre = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":1,"method":"tools/list"}', studio, assets, pool)
    assert_true(pre.find(String(MCP_ERR_NOT_INITIALIZED)) >= 0, "tools/list before initialize must fail: " + pre)
    var ping = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"ping"}', studio, assets, pool)
    assert_true(ping.find('"result":{}') >= 0, "ping is allowed before initialize")
    var init = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}', studio, assets, pool
    )
    assert_true(init.find('"protocolVersion":"2025-06-18"') >= 0, "initialize echoes a supported version: " + init)
    assert_true(init.find('"listChanged":false') >= 0, "the tool list is static: " + init)
    var post = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":4,"method":"tools/list"}', studio, assets, pool)
    assert_true(post.find('"tools"') >= 0, "tools/list works after initialize")
    var batch = mcp.handle_jsonrpc('[{"jsonrpc":"2.0","id":5,"method":"ping"}]', studio, assets, pool)
    assert_true(batch.find("-32700") >= 0 and batch.find('"id":null') >= 0, "batches are refused, not crashed on: " + batch)

    # Turn allowance: with every tool gated, a call is held until the host grants the turn.
    mcp.set_client_gate_mode(0, MCP_GATE_ALL)
    var held = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}', studio, assets, pool
    )
    assert_true(held.find("approval_required") >= 0, "gated call must be held: " + held)
    mcp.set_client_turn_allowance(0, True)
    var allowed = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}', studio, assets, pool
    )
    assert_true(allowed.find("approval_required") < 0, "turn allowance lets the call through: " + allowed)
    mcp.set_client_turn_allowance(0, False)
    var held_again = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}', studio, assets, pool
    )
    assert_true(held_again.find("approval_required") >= 0, "allowance ends when the host clears it")

    var auth = McpServerV2(workspace_root=_tmp_dir("mcp_auth"), auth_token="hardening-token")
    var bad = auth.handle_jsonrpc('{"jsonrpc":"2.0","id":9,"method":"initialize","params":{"auth_token":"nope"}}', studio, assets, pool)
    assert_true(bad.find(String(MCP_ERR_UNAUTHORIZED)) >= 0, "auth failure must be -32001: " + bad)
    assets.close()


# ---------------------------------------------------------------------------
# 4. Jobs: exit codes, ECHILD, pruning, non-blocking group cancel
# ---------------------------------------------------------------------------

def _wait_done(mut runner: JobRunner, jid: Int, timeout_sec: Float64 = 5.0) -> Int:
    var t_end = get_time() + timeout_sec
    while get_time() < t_end:
        _ = runner.poll_all()
        var ji = runner.find_index(jid)
        if ji >= 0 and not runner.jobs[ji].is_running() and not runner.jobs[ji].is_reaping():
            return ji
        _ = external_call["usleep", Int32](UInt32(5000))
    return runner.find_index(jid)


def test_jobs_exit_echild_prune_and_group_cancel() raises:
    print("  [4/7] JobRunner exit codes, ECHILD, pruning, process-group cancel...")
    var reaped = reap_child(Int32(1), block=False)
    assert_true(reaped[0] and reaped[1] == -1, "waitpid ECHILD is a failed (-1) exit, never success")

    var runner = JobRunner()
    var argv = List[String]()
    argv.append("/bin/sh")
    argv.append("-c")
    argv.append("exit 3")
    var jid = runner.start_argv_job(JOB_KIND_TASK, "exit3", argv, ".")
    assert_true(jid >= 0)
    var ji = _wait_done(runner, jid)
    assert_true(ji >= 0 and runner.jobs[ji].state == JOB_FAILED and runner.jobs[ji].exit_code == 3, "exit 3 must be FAILED")

    for _ in range(MAX_FINISHED_JOBS + 6):
        var a2 = List[String]()
        a2.append("/bin/true")
        _ = runner.start_argv_job(JOB_KIND_TASK, "true", a2, ".")
    var t_end = get_time() + 10.0
    while get_time() < t_end and runner.active_count() > 0:
        _ = runner.poll_all()
        _ = external_call["usleep", Int32](UInt32(5000))
    _ = runner.poll_all()
    assert_true(len(runner.jobs) <= MAX_FINISHED_JOBS, "finished jobs must be pruned, have " + String(len(runner.jobs)))

    # Cancel a shell that ignores SIGTERM and has a grandchild: cancel returns at once, and the
    # whole group is gone after the grace period (SIGKILL), polled, never slept on.
    var a3 = List[String]()
    a3.append("/bin/sh")
    a3.append("-c")
    a3.append("trap '' TERM; sleep 30 & wait")
    var cj = runner.start_argv_job(JOB_KIND_TASK, "stubborn", a3, ".")
    assert_true(cj >= 0)
    _ = external_call["usleep", Int32](UInt32(100000))
    var ci = runner.find_index(cj)
    var pgid = Int32(runner.jobs[ci].pid)
    var t0 = get_time()
    _ = runner.cancel(cj)
    assert_true(get_time() - t0 < 0.1, "cancel must not block the caller")
    var ci2 = _wait_done(runner, cj, 5.0)
    assert_true(ci2 < 0 or not runner.jobs[ci2].is_reaping(), "cancelled job must be reaped")
    var alive = external_call["kill", Int32](-pgid, Int32(0))
    assert_true(alive != 0, "the whole process group (grandchild sleep too) must be gone")


# ---------------------------------------------------------------------------
# 5. AgentBus caps, coalescing and turn allowance
# ---------------------------------------------------------------------------

def test_agent_bus_caps_coalescing_and_turn_allowance() raises:
    print("  [5/7] AgentBus approval coalescing/caps, thought cap, turn allowance...")
    var bus = AgentTelemetryBus()
    var aid = bus.create_session(ROLE_GAMEPLAY_CODER, "hardening")
    var a = PendingApproval(agent_id=aid, source=APPROVAL_SRC_MCP_TOOL, title="spawn")
    a.tool_name = "ecs.spawn_entity"
    a.mcp_client_key = 7
    a.raw_payload_json = '{"name":"X"}'
    var b = a.copy()
    var id1 = bus.queue_approval(a^)
    var id2 = bus.queue_approval(b^)
    assert_true(id1 > 0 and id1 == id2, "identical gated calls from one client coalesce")
    assert_equal(bus.pending_count_for_agent(aid), 1)
    for k in range(MAX_PENDING_APPROVALS_PER_AGENT + 4):
        var c = PendingApproval(agent_id=aid, source=APPROVAL_SRC_MCP_TOOL, title="spawn")
        c.tool_name = "ecs.spawn_entity"
        c.mcp_client_key = 7
        c.raw_payload_json = '{"name":"X' + String(k) + '"}'
        _ = bus.queue_approval(c^)
    assert_equal(bus.pending_count_for_agent(aid), MAX_PENDING_APPROVALS_PER_AGENT)

    var chunk = String("")
    for _ in range(1024):
        chunk += "thinking-hard "
    for _ in range(4):
        _ = bus.record_acp_thought(aid, chunk)
    var si = bus.find_session_index(aid)
    var n = len(bus.sessions[si].steps)
    assert_true(bus.sessions[si].steps[n - 1].summary.byte_length() <= MAX_THOUGHT_SUMMARY_BYTES, "thought text is capped")

    _ = bus.begin_turn(aid, "do it")
    bus.sessions[si].auto_approve_turn_id = bus.sessions[si].active_turn_id
    assert_true(bus.turn_allowance_active(aid))
    bus.end_turn(aid)
    assert_false(bus.turn_allowance_active(aid), "approve-for-turn ends with the turn")


# ---------------------------------------------------------------------------
# 6. safe_fs confinement + revert guard
# ---------------------------------------------------------------------------

def test_safe_fs_and_revert_guard() raises:
    print("  [6/7] safe_fs symlink/dotdot/hidden refusal + revert hash guard...")
    var root = _tmp_dir("fs")
    var outside = _tmp_dir("fs_outside")
    var secret = _bytes("secret")
    _ = write_bytes(outside + "/secret.txt", secret)
    var target = outside
    var linkp = root + "/link"
    _ = external_call["unlink", Int32](linkp.as_c_string_span())
    _ = external_call["symlink", Int32](target.as_c_string_span(), linkp.as_c_string_span())
    var via_link = read_agent_file(root, "link/secret.txt")
    assert_true(via_link[0] != SAFE_FS_OK, "reads never follow a symlinked directory")
    var dotdot = read_agent_file(root, "../fs_outside/secret.txt")
    assert_true(dotdot[0] != SAFE_FS_OK, ".. is refused")
    var hidden = read_agent_file(root, ".git/config")
    assert_true(hidden[0] != SAFE_FS_OK, "hidden tool-config paths are refused")
    assert_equal(read_agent_file(root, "nope.txt")[0], SAFE_FS_MISSING)
    var cp_link = resolve_confined_path(root, "link/evil.txt")
    var w_link = write_confined_durable(cp_link, "pwned")
    assert_false(w_link[0], "writes never follow a symlinked directory")
    assert_true(file_size(outside + "/evil.txt") < 0)

    # Revert guard through Studio: only reverts while the disk holds what the agent wrote.
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=200, worker_threads=0)
    studio.agent_runtime.workspace_root = root
    var cp = resolve_confined_path(root, "src/edit.mojo")
    assert_true(write_confined_durable(cp, "agent version\n")[0])
    var post = text_hash64("agent version\n")
    var ok = studio._revert_agent_file("src/edit.mojo", "original\n", post)
    assert_true(ok[0], "revert must succeed while the file is unchanged: " + ok[1])
    assert_equal(String(unsafe_from_utf8=read_bytes(root + "/src/edit.mojo")), String("original\n"))
    _ = write_confined_durable(cp, "user changed it\n")
    var refused = studio._revert_agent_file("src/edit.mojo", "original\n", post)
    assert_false(refused[0], "revert must refuse a file changed after the agent wrote it")
    assert_equal(String(unsafe_from_utf8=read_bytes(root + "/src/edit.mojo")), String("user changed it\n"))


# ---------------------------------------------------------------------------
# 7. Version-gated structuredContent, async cook, durable import, socket ownership
# ---------------------------------------------------------------------------

def _sock_exists(path: String) -> Bool:
    var buf = List[UInt8](length=256, fill=0)
    var owned = path
    return external_call["lstat", Int32](owned.as_c_string_span(), buf.unsafe_ptr()) == 0


def _bind_dead_socket(path: String):
    """Leaves a socket file at `path` with nobody listening (what a crashed Studio leaves behind)."""
    var fd = external_call["socket", Int32](Int32(1), Int32(1), Int32(0))
    var addr = List[UInt8](length=110, fill=0)
    addr[0] = 1
    var pb = path.as_bytes()
    for i in range(len(pb)):
        addr[2 + i] = pb[i]
    _ = external_call["bind", Int32](fd, addr.unsafe_ptr(), UInt32(110))
    _ = external_call["close", Int32](fd)


def _init_req(id: Int, version: String) -> String:
    return (
        '{"jsonrpc":"2.0","id":' + String(id) + ',"method":"initialize","params":{"protocolVersion":"'
        + version + '","clientInfo":{"name":"hardening"}}}'
    )


def test_mcp_deferred_items() raises:
    print("  [7/7] structuredContent gating, async cook, durable import, socket ownership...")
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=200, worker_threads=0)
    var assets = AssetServer()
    var pool = JobPool()
    var call_q = '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}'

    # 2024-11-05 clients: no structuredContent, no outputSchema.
    var old = McpServerV2(workspace_root=_tmp_dir("mcp_v_old"))
    var old_init = old.handle_jsonrpc(_init_req(1, "2024-11-05"), studio, assets, pool)
    assert_true(old_init.find('"protocolVersion":"2024-11-05"') >= 0, old_init)
    var old_call = old.handle_jsonrpc(call_q, studio, assets, pool)
    assert_true(old_call.find('"content":[') >= 0 and old_call.find('"isError":false') >= 0, old_call)
    assert_true(old_call.find("structuredContent") < 0, "2024-11-05 clients get no structuredContent: " + old_call)
    var old_list = old.handle_jsonrpc('{"jsonrpc":"2.0","id":3,"method":"tools/list"}', studio, assets, pool)
    assert_true(old_list.find("outputSchema") < 0, "2024-11-05 clients get no outputSchema")
    # An unknown in-between revision also lands on 2024-11-05.
    var mid = McpServerV2(workspace_root=_tmp_dir("mcp_v_mid"))
    var mid_init = mid.handle_jsonrpc(_init_req(1, "2025-03-26"), studio, assets, pool)
    assert_true(mid_init.find('"protocolVersion":"2024-11-05"') >= 0, mid_init)
    assert_true(mid.handle_jsonrpc(call_q, studio, assets, pool).find("structuredContent") < 0)
    # No initialize at all (lenient in-process key 0): the default revision, so no structuredContent.
    var bare = McpServerV2(workspace_root=_tmp_dir("mcp_v_bare"))
    assert_true(bare.handle_jsonrpc(call_q, studio, assets, pool).find("structuredContent") < 0)

    # 2025-06-18 clients: structuredContent on every result, an object outputSchema on every tool.
    var cur = McpServerV2(workspace_root=_tmp_dir("mcp_v_new"))
    _ = cur.handle_jsonrpc(_init_req(1, "2025-06-18"), studio, assets, pool)
    var new_call = cur.handle_jsonrpc(call_q, studio, assets, pool)
    assert_true(new_call.find('"structuredContent":{') >= 0, "2025-06-18 clients get structuredContent: " + new_call)
    var new_list = cur.handle_jsonrpc('{"jsonrpc":"2.0","id":3,"method":"tools/list"}', studio, assets, pool)
    var n_tools = new_list.count('"inputSchema":')
    assert_true(n_tools > 10 and new_list.count('"outputSchema":{"type":"object"}') == n_tools, "every tool declares outputSchema")
    var unknown = cur.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"no.such_tool","arguments":{}}}', studio, assets, pool
    )
    assert_true(unknown.find('"isError":true') >= 0 and unknown.find('"structuredContent":{"ok":false') >= 0, unknown)

    # The version is per client: a socket client on 2024-11-05 next to the 2025-06-18 key-0 client.
    cur.select_client(41)
    _ = cur.handle_jsonrpc(_init_req(5, "2024-11-05"), studio, assets, pool)
    assert_true(cur.handle_jsonrpc(call_q, studio, assets, pool).find("structuredContent") < 0, "per-client version")
    cur.select_client(0)
    assert_true(cur.handle_jsonrpc(call_q, studio, assets, pool).find('"structuredContent"') >= 0, "key 0 keeps 2025-06-18")

    # scene.set_entity_enabled drives the Inspector's Enabled flag, never Visible.
    var en_srv = McpServerV2(workspace_root=_tmp_dir("mcp_enabled"))
    _ = en_srv.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"EnableProbe"}}}',
        studio, assets, pool,
    )
    var e0 = -1
    for ei in range(len(studio.entities)):
        if studio.entities[ei].alive:
            e0 = ei
            break
    assert_true(e0 >= 0, "the default scene has an entity")
    studio.entities[e0].visible = True
    var dis = en_srv.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"scene.set_entity_enabled","arguments":{"entity_id":'
        + String(e0) + ',"enabled":false}}}',
        studio, assets, pool,
    )
    assert_true(dis.find('"isError":false') >= 0, dis)
    assert_false(studio.entities[e0].enabled, "Enabled cleared")
    assert_true(studio.entities[e0].visible, "Visible is a separate switch and stays on")
    var q_en = en_srv.handle_jsonrpc(call_q, studio, assets, pool)
    assert_true(q_en.find('\\"enabled\\":false') >= 0, "query reports enabled: " + q_en)
    _ = studio.undo()
    assert_true(studio.entities[e0].enabled, "undo restores Enabled")
    _ = en_srv.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"scene.set_entity_enabled","arguments":{"entity_id":'
        + String(e0) + ',"enabled":true}}}',
        studio, assets, pool,
    )
    assert_true(studio.entities[e0].enabled and studio.entities[e0].visible)

    # Async cook: a socket client's cook (even force_recook, with no flint-pack) runs in a child process.
    var ws = _tmp_dir("mcp_cook")
    _ = make_dirs(ws + "/assets")
    var helper = ws + "/fake_cook_worker.sh"
    var script = (
        "#!/bin/sh\n"
        + '[ "$1" = "--mcp-cook-worker" ] && [ "$5" = "1" ] || exit 3\n'
        + "echo 'FLINT_COOK_REPORT {\"ok\":true,\"total_assets\":3,\"cooked_count\":2,\"cache_hits\":1,\"archive_bytes\":99,\"elapsed_ms\":1.0}'\n"
    )
    var script_b = _bytes(script)
    _ = write_bytes(helper, script_b)
    _ = external_call["chmod", Int32](helper.as_c_string_span(), Int32(0o755))
    var cook = McpServerV2(workspace_root=ws)
    cook.cook_helper_exe = helper
    cook.select_client(7)
    cook.set_client_gate_mode(7, MCP_GATE_AUTO)
    _ = cook.handle_jsonrpc(_init_req(1, "2025-06-18"), studio, assets, pool)
    var t0 = get_time()
    var started = cook.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"assets.list_and_cook","arguments":'
        + '{"source_dir":"assets","output_flpk":"build/out.flpk","force_recook":true}}}',
        studio, assets, pool,
    )
    assert_true(get_time() - t0 < 2.0, "starting the cook must not block")
    assert_true(started.find('"pending":true') >= 0 and started.find('"build_id":') >= 0, "cook is deferred: " + started)
    var bid_at = started.find('"build_id":') + 11
    var bid = 0
    var sb = started.as_bytes()
    while bid_at < len(sb) and sb[bid_at] >= 48 and sb[bid_at] <= 57:
        bid = bid * 10 + Int(sb[bid_at] - 48)
        bid_at += 1
    var status = String("")
    var t_end = get_time() + 10.0
    while get_time() < t_end:
        status = cook.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"code.build_status","arguments":{"build_id":'
            + String(bid) + "}}}",
            studio, assets, pool,
        )
        if status.find('"pending":true') < 0:
            break
        _ = external_call["usleep", Int32](UInt32(10000))
    assert_true(status.find('"cook_report":{"ok":true') >= 0, "worker cook succeeds: " + status)
    assert_true(status.find('"total_assets":3') >= 0 and status.find('"cache_hits":1') >= 0, "report lifted: " + status)

    # asset.import: durable copy (no `.tmp` left behind), never through a symlinked directory.
    var imp_ws = _tmp_dir("mcp_import")
    var imp = McpServerV2(workspace_root=imp_ws)
    var flmat = _bytes('{"format":"flmat","version":1,"base_color":[0.2,0.7,0.3],"roughness":0.4}')
    _ = make_dirs(imp_ws + "/src")
    _ = write_bytes(imp_ws + "/src/m.flmat", flmat)
    var ok_imp = imp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"asset.import","arguments":'
        + '{"source_path":"src/m.flmat","dest_path":"assets/mats/m.flmat"}}}',
        studio, assets, pool,
    )
    assert_true(ok_imp.find('"isError":false') >= 0, "import succeeds: " + ok_imp)
    assert_equal(file_size(imp_ws + "/assets/mats/m.flmat"), len(flmat))
    assert_true(file_size(imp_ws + "/assets/mats/m.flmat.tmp") < 0, "durable write leaves no .tmp")
    var outside = _tmp_dir("mcp_import_outside")
    var linkp = imp_ws + "/linked"
    _ = external_call["unlink", Int32](linkp.as_c_string_span())
    _ = external_call["symlink", Int32](outside.as_c_string_span(), linkp.as_c_string_span())
    var bad_imp = imp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"asset.import","arguments":'
        + '{"source_path":"src/m.flmat","dest_path":"linked/m.flmat"}}}',
        studio, assets, pool,
    )
    assert_true(bad_imp.find('"isError":true') >= 0, "import through a symlinked dir is refused: " + bad_imp)
    assert_true(file_size(outside + "/m.flmat") < 0, "nothing was written outside the workspace")

    # Unix listener: a stale socket is replaced, a live one is never unlinked, close removes only ours.
    var sdir = _tmp_dir("mcp_sock")
    var sp = sdir + "/s.sock"
    _ = external_call["unlink", Int32](sp.as_c_string_span())
    _bind_dead_socket(sp)
    assert_true(_sock_exists(sp), "dead socket file left behind")
    var a = FlintMcpUnixListener(sp)
    assert_true(a.active and a.socket_path == sp, "a stale socket (nobody answers) is unlinked and rebound")
    var b = FlintMcpUnixListener(sp)
    assert_true(b.active and b.socket_path != sp, "a live socket is not taken over: " + b.socket_path)
    var b_path = b.socket_path
    assert_true(_sock_exists(sp) and _sock_exists(b_path))
    b.close()
    assert_false(_sock_exists(b_path), "close unlinks our own socket")
    assert_true(_sock_exists(sp), "close never touches another listener's socket")
    # Someone replaced our path after we bound it: close must leave their file alone.
    _ = external_call["unlink", Int32](sp.as_c_string_span())
    _bind_dead_socket(sp)
    a.close()
    assert_true(_sock_exists(sp), "close leaves a socket file that is no longer ours")
    _ = external_call["unlink", Int32](sp.as_c_string_span())
    var c = FlintMcpUnixListener(sp)
    assert_true(c.active, "fresh path binds")
    c.close()
    assert_false(_sock_exists(sp), "close unlinks our own socket")
    # A regular file at the path is never unlinked.
    var not_sock = _bytes("not a socket")
    _ = write_bytes(sp, not_sock)
    var d = FlintMcpUnixListener(sp)
    assert_false(d.active, "a non-socket file at the path is refused")
    assert_equal(file_size(sp), 12)
    _ = external_call["unlink", Int32](sp.as_c_string_span())
    assets.close()


def main() raises:
    print("============================================================")
    print(" FLINT STUDIO AGENT / PROCESS-LAYER HARDENING SUITE")
    print("============================================================")
    test_acp_ids_terminal_and_fs_requests()
    test_mcp_framing_recovery_and_per_message_cap()
    test_mcp_lifecycle_negotiation_auth_and_turn_allowance()
    test_jobs_exit_echild_prune_and_group_cancel()
    test_agent_bus_caps_coalescing_and_turn_allowance()
    test_safe_fs_and_revert_guard()
    test_mcp_deferred_items()
    print("[PASS] All 7 agent hardening suites PASSED!")
