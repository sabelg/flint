"""Phase 3 tests for per-client MCP sessions and command-style scene load (`tests/test_studio_sessions.mojo`).

  - Each MCP connection has its own gate mode and its own approvals: approving agent A's call
    never lets agent B's identical call through.
  - Tool events carry the connection that made them, and each connection maps to its own agent.
  - Loading a scene is an undoable command instead of wiping undo/redo.
  - Typing into one inspector field is one undo step, not one per keystroke (Phase 4).
  - `code.build_hot_reload` with `async` returns at once; `code.build_status` finishes it (Phase 5).
  - Undo entries carry their origin, and an agent revert refuses to wipe later user edits (Phase 6).
  - Autosave writes beside the project without touching the scene path or dirty flag (Phase 7).
"""
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.lz4 import lz4_packed_raw_size
from flint.devtools.mcp_v2 import MCP_GATE_ALL, MCP_GATE_AUTO, McpServerV2
from flint.platform.os import file_size, get_time, make_dirs, write_bytes
from flint.studio import FlintStudioState
from flint.studio.agent_bus import AgentStep


def _run_dir(tag: String) -> String:
    """Per-run scratch folder, so parallel or repeated runs never share files."""
    var d = "build/test_studio_sessions_" + String(Int(external_call["getpid", Int32]())) + "/" + tag
    _ = make_dirs(d)
    return d


def _init(mut mcp: McpServerV2, mut studio: FlintStudioState, mut assets: AssetServer, mut pool: JobPool, name: String):
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"'
        + name + '"}}}',
        studio, assets, pool,
    )


def _spawn_call(mut mcp: McpServerV2, mut studio: FlintStudioState, mut assets: AssetServer, mut pool: JobPool) -> String:
    return mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"Probe"}}}',
        studio, assets, pool,
    )


def test_approvals_are_per_client() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    mcp.set_gate_mode(MCP_GATE_ALL)
    mcp.select_client(1)
    _init(mcp, studio, assets, pool, "agent-a")
    mcp.select_client(2)
    _init(mcp, studio, assets, pool, "agent-b")
    mcp.select_client(0)
    assert_equal(mcp.client_name_of(1), "agent-a")
    assert_equal(mcp.client_name_of(2), "agent-b")

    mcp.allow_tool_once_for_client(1, "ecs.spawn_entity")
    mcp.select_client(2)
    var rb = _spawn_call(mcp, studio, assets, pool)
    assert_true(rb.find("approval_required") >= 0, "B must still be gated after A was approved: " + rb)
    mcp.select_client(1)
    var ra = _spawn_call(mcp, studio, assets, pool)
    assert_true(ra.find("approval_required") < 0, "A's approved retry runs: " + ra)
    var ra2 = _spawn_call(mcp, studio, assets, pool)
    assert_true(ra2.find("approval_required") >= 0, "a one-shot approval is used up")
    mcp.select_client(0)

    var evs = mcp.take_tool_events()
    assert_true(len(evs) >= 3)
    assert_equal(evs[0].client_key, 2)
    assert_equal(evs[0].client_name, "agent-b")
    assert_equal(evs[1].client_key, 1)

    mcp.forget_client(1)
    assert_equal(mcp.client_name_of(1), "", "a dropped connection's session is gone")


def test_each_client_gets_its_own_agent() raises:
    var studio = FlintStudioState()
    var a1 = studio.agent_for_mcp_client(1, "agent-a")
    var a2 = studio.agent_for_mcp_client(2, "agent-b")
    assert_true(a1 != 0 and a2 != 0 and a1 != a2)
    assert_equal(studio.agent_for_mcp_client(1, "agent-a"), a1, "the mapping is stable")


def test_scene_load_is_undoable() raises:
    var studio = FlintStudioState()
    _ = studio.spawn_entity("Saved")
    var path = _run_dir("load_undo") + "/scene.flscn"
    assert_true(studio.save_scene_file(path))
    _ = studio.spawn_entity("Unsaved")
    var before = studio.live_entity_count()
    var undo_before = len(studio.undo_stack)
    assert_true(studio.load_scene_file(path))
    assert_equal(studio.live_entity_count(), before - 1)
    assert_equal(len(studio.undo_stack), undo_before + 1, "load adds one undo step instead of clearing history")
    ref load_tx = studio.undo_stack[len(studio.undo_stack) - 1]
    assert_true(
        len(load_tx.before_world_blob) < 1024 or lz4_packed_raw_size(load_tx.before_world_blob) > 0,
        "undo world blobs are stored LZ4-packed",
    )
    assert_true(studio.undo())
    studio._sync_ecs_and_physics()
    assert_equal(studio.live_entity_count(), before, "undoing the load brings back the edited scene")


def test_typing_is_one_undo_step() raises:
    var studio = FlintStudioState()
    var s0 = studio.ui_edit_seq
    studio.ui_ctx.focused_id = UInt64(77)  # typing into one text field
    studio.begin_edit_interaction()
    studio.begin_edit_interaction()
    studio.begin_edit_interaction()
    assert_equal(studio.ui_edit_seq, s0 + 1, "keystrokes in one focused field share one undo step")
    studio.ui_ctx.focused_id = UInt64(78)
    studio.begin_edit_interaction()
    assert_equal(studio.ui_edit_seq, s0 + 2, "moving to another field starts a new step")
    studio.ui_ctx.focused_id = UInt64(0)
    studio.begin_edit_interaction()
    studio.begin_edit_interaction()
    assert_equal(studio.ui_edit_seq, s0 + 4, "unfocused changes (combo, checkbox) are one step each")


def test_async_build_does_not_block() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    _init(mcp, studio, assets, pool, "builder")
    _ = make_dirs("build")
    var bad_src = List[UInt8]()
    for b in String("def broken(:\n").as_bytes():
        bad_src.append(b)
    _ = write_bytes("build/phase5_bad_gameplay.mojo", bad_src)
    var t0 = get_time()
    var r = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"code.build_hot_reload","arguments":{"entry_path":"build/phase5_bad_gameplay.mojo","output_so":"build/phase5_bad.so","async":true}}}',
        studio, assets, pool,
    )
    assert_true(get_time() - t0 < 2.0, "an async build returns at once")
    assert_true(r.find('\\"pending\\":true') >= 0 or r.find('"pending":true') >= 0, "async build reports pending: " + r)
    var final = String("")
    var waited = 0
    while waited < 600:
        var st = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"code.build_status","arguments":{"build_id":1}}}',
            studio, assets, pool,
        )
        if st.find('"pending":true') < 0:
            final = st
            break
        _ = external_call["usleep", Int32](UInt32(200000))
        waited += 1
    assert_true(final.byte_length() > 0, "the async build finishes")
    assert_true(final.find('"compiled":false') >= 0, "a broken entry reports a failed build: " + final)
    var unknown = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"code.build_status","arguments":{"build_id":999}}}',
        studio, assets, pool,
    )
    assert_true(unknown.find("unknown build_id") >= 0)


def test_revert_never_wipes_later_user_edits() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    mcp.set_gate_mode(MCP_GATE_AUTO)  # socket clients start gated (MCP_GATE_MUTATING)
    mcp.select_client(1)
    _init(mcp, studio, assets, pool, "agent-a")
    _ = _spawn_call(mcp, studio, assets, pool)
    mcp.select_client(0)
    var aid = studio.agent_for_mcp_client(1, "agent-a")
    var top = len(studio.undo_stack) - 1
    assert_equal(studio.undo_stack[top].agent_id, Int(aid), "an MCP edit is tagged with its agent")
    var agent_tx = studio.undo_stack[top].tx_id
    var si = studio.agent_bus.find_session_index(aid)
    var st = AgentStep(step_id=UInt64(4242), agent_id=aid)
    st.tx_id = agent_tx
    studio.agent_bus.sessions[si].steps.append(st^)
    var ids = List[UInt64]()
    ids.append(UInt64(4242))
    assert_equal(studio.user_edits_after_agent_steps(aid, ids), 0)
    _ = studio.spawn_entity("Mine")
    assert_equal(studio.undo_stack[len(studio.undo_stack) - 1].agent_id, 0, "a user edit has origin 0")
    assert_equal(studio.user_edits_after_agent_steps(aid, ids), 1, "the revert guard sees the user's later edit")


def test_autosave_keeps_user_file_and_dirty_flag() raises:
    var studio = FlintStudioState()
    var proj = _run_dir("autosave")
    studio.project_root = proj
    studio.scene_file_path = "scenes/level_one.flscn"
    _ = studio.spawn_entity("Unsaved work")
    studio.mark_dirty(True)
    studio.autosave_interval_sec = Float32(1.0)
    assert_true(not studio.tick_autosave(Float32(0.5)))
    assert_true(studio.tick_autosave(Float32(0.6)), "autosaves after the interval of dirty time")
    assert_true(studio.last_autosave_path.startswith(proj + "/.flint/autosave/level_one-"))
    assert_true(studio.last_autosave_path.endswith(".autosave.flscnb"), "keyed by a hash of the full scene path")
    assert_true(file_size(studio.last_autosave_path) > 0)
    assert_true(studio.dirty, "an autosave is not a save: the scene stays dirty")
    assert_equal(studio.scene_file_path, "scenes/level_one.flscn", "the user's file path is untouched")
    studio.mark_dirty(False)
    assert_true(not studio.tick_autosave(Float32(5.0)), "a clean scene is never autosaved")


def main() raises:
    test_approvals_are_per_client()
    print("[PASS] test_approvals_are_per_client")
    test_each_client_gets_its_own_agent()
    print("[PASS] test_each_client_gets_its_own_agent")
    test_scene_load_is_undoable()
    print("[PASS] test_scene_load_is_undoable")
    test_typing_is_one_undo_step()
    print("[PASS] test_typing_is_one_undo_step")
    test_async_build_does_not_block()
    print("[PASS] test_async_build_does_not_block")
    test_revert_never_wipes_later_user_edits()
    print("[PASS] test_revert_never_wipes_later_user_edits")
    test_autosave_keeps_user_file_and_dirty_flag()
    print("[PASS] test_autosave_keeps_user_file_and_dirty_flag")
    print("[PASS] test_studio_sessions: per-client MCP sessions and undoable load hold")
