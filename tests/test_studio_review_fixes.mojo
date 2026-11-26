"""Regression tests for the second Flint Studio review (`tests/test_studio_review_fixes.mojo`).

  - MCP approvals: a one-shot grant is bound to the exact arguments and expires; an approved call
    replays once and the agent's identical retry gets the cached result instead of running twice.
  - MCP sandbox: hidden / tool-config paths are rejected; builds only write `.so` files under
    `build/` from `.mojo` sources; socket clients always build async and only see their own builds.
  - JSON: `\\uD83D\\uDE00` surrogate pairs decode to one 4-byte UTF-8 scalar; lone halves become U+FFFD.
  - Scene files: undoing a load restores the previous file path and clean state; recovering an
    autosave keeps the user's path and leaves the scene dirty; Save As / Open keep a recent list.
  - Viewport: a cancelled gizmo drag puts the selection back and leaves no undo entry; marquee and
    select-all skip locked entities; the Select tool never picks a gizmo handle.
  - Outliner: search finds children of collapsed parents; arrow keys walk the visible rows.
  - Undo origin: redo keeps the agent tag of the entry it re-applies.
Every file this test writes lives under a per-run directory in `build/`.
"""
from std.ffi import external_call
from std.math import abs
from std.testing import assert_equal, assert_true

from flint.assets.json import parse_json
from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.devtools.mcp_v2 import MCP_GATE_ALL, McpServerV2, sanitize_output_path
from flint.platform.os import file_size, get_time, make_dirs, write_bytes
from flint.studio import FlintStudioState
from flint.studio.gizmo import (
    GIZMO_AXIS_NONE,
    GIZMO_MODE_SELECT,
    GIZMO_MODE_TRANSLATE,
    GIZMO_TARGET_PX,
    gizmo_world_scale,
    project_world_to_viewport,
)
from flint.studio.viewport_transform import ViewportTransform
from flint.ui.flexbox_dock import RectF


def _run_dir(tag: String) -> String:
    var pid = Int(external_call["getpid", Int32]())
    var d = "build/test_review_fixes_" + String(pid) + "/" + tag
    _ = make_dirs(d)
    return d


def _init(mut mcp: McpServerV2, mut studio: FlintStudioState, mut assets: AssetServer, mut pool: JobPool, name: String):
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"'
        + name + '"}}}',
        studio, assets, pool,
    )


def _call(
    mut mcp: McpServerV2, mut studio: FlintStudioState, mut assets: AssetServer, mut pool: JobPool, tool: String, args: String
) -> String:
    return mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"' + tool + '","arguments":' + args + "}}",
        studio, assets, pool,
    )


def test_grants_are_bound_to_args_and_expire() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    mcp.set_gate_mode(MCP_GATE_ALL)
    mcp.select_client(1)
    _init(mcp, studio, assets, pool, "agent-a")
    mcp.select_client(0)
    var probe = String('{"name":"Probe"}')

    mcp.allow_tool_once_for_client(1, "ecs.spawn_entity", '{"name":"Other"}')
    mcp.select_client(1)
    var r1 = _call(mcp, studio, assets, pool, "ecs.spawn_entity", probe)
    assert_true(r1.find("approval_required") >= 0, "a grant for other arguments does not cover this call: " + r1)
    mcp.select_client(0)

    mcp.allow_tool_once_for_client(1, "ecs.spawn_entity", probe)
    mcp.select_client(1)
    var r2 = _call(mcp, studio, assets, pool, "ecs.spawn_entity", probe)
    assert_true(r2.find("approval_required") < 0, "the exact approved call runs: " + r2)

    # A grant older than the TTL is dropped instead of waving a later call through.
    mcp.once_allow_tools.append(McpServerV2._grant_token("ecs.spawn_entity", "", get_time() - 100000.0))
    var r3 = _call(mcp, studio, assets, pool, "ecs.spawn_entity", probe)
    assert_true(r3.find("approval_required") >= 0, "an expired grant is ignored: " + r3)
    mcp.select_client(0)


def test_approved_call_replays_once_and_retry_is_cached() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    mcp.set_gate_mode(MCP_GATE_ALL)
    mcp.select_client(1)
    _init(mcp, studio, assets, pool, "agent-a")
    var args = String('{"name":"Replayed"}')
    var gated = _call(mcp, studio, assets, pool, "ecs.spawn_entity", args)
    assert_true(gated.find("approval_required") >= 0)
    mcp.select_client(0)

    var n0 = studio.live_entity_count()
    _ = mcp.replay_approved_call(1, "ecs.spawn_entity", args, UInt64(99), UInt64(0), studio, assets, pool)
    studio._sync_ecs_and_physics()
    assert_equal(studio.live_entity_count(), n0 + 1, "approving runs the call exactly as the agent sent it")
    assert_equal(mcp.active_client_key, 0, "replay restores the selected client")

    mcp.select_client(1)
    var retry = _call(mcp, studio, assets, pool, "ecs.spawn_entity", args)
    studio._sync_ecs_and_physics()
    assert_true(retry.find("approval_required") < 0, "the agent's retry gets the approved result: " + retry)
    assert_equal(studio.live_entity_count(), n0 + 1, "the retry does not run the call a second time")
    var again = _call(mcp, studio, assets, pool, "ecs.spawn_entity", args)
    assert_true(again.find("approval_required") >= 0, "the cached result is used once")
    mcp.select_client(2)
    _init(mcp, studio, assets, pool, "agent-b")
    var other = _call(mcp, studio, assets, pool, "ecs.spawn_entity", args)
    assert_true(other.find("approval_required") >= 0, "another client never gets agent A's approval")
    mcp.select_client(0)


def test_hidden_paths_are_rejected() raises:
    var rejected = 0
    for p in [".git/config", ".claude/settings.json", "assets/.pixi/x", ".mcp.json"]:
        try:
            _ = sanitize_output_path(".", p)
        except:
            rejected += 1
    assert_equal(rejected, 4, "every hidden or tool-config path is refused")
    var ok_path = sanitize_output_path(".", ".flint/autosave/x.flscnb")
    assert_true(ok_path.find(".flint") >= 0, "Studio's own .flint folder stays writable")


def test_build_paths_ownership_and_async() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    mcp.select_client(1)
    _init(mcp, studio, assets, pool, "builder-a")
    var bad_out = _call(
        mcp, studio, assets, pool, "code.build_hot_reload", '{"entry_path":"build/x.mojo","output_so":"src/evil.so"}'
    )
    assert_true(bad_out.find("output_so must be a .so file under build/") >= 0, bad_out)
    var bad_entry = _call(
        mcp, studio, assets, pool, "code.build_hot_reload", '{"entry_path":"README.md","output_so":"build/x.so"}'
    )
    assert_true(bad_entry.find("entry_path must be a .mojo source file") >= 0, bad_entry)

    var d = _run_dir("build")
    var src = List[UInt8]()
    for b in String("def broken(:\n").as_bytes():
        src.append(b)
    _ = write_bytes(d + "/bad.mojo", src)
    var t0 = get_time()
    var r = _call(
        mcp, studio, assets, pool, "code.build_hot_reload",
        '{"entry_path":"' + d + '/bad.mojo","output_so":"' + d + '/bad.so","async":false}',
    )
    assert_true(get_time() - t0 < 2.0, "a socket client's build never blocks, even with async:false")
    assert_true(r.find('pending\\":true') >= 0 or r.find('"pending":true') >= 0, "socket builds are async: " + r)
    var bid_at = r.find("build_id")
    assert_true(bid_at >= 0)
    var digits = String("")
    var i = bid_at
    var rb = r.as_bytes()
    while i < len(rb) and not (rb[i] >= 48 and rb[i] <= 57):
        i += 1
    while i < len(rb) and rb[i] >= 48 and rb[i] <= 57:
        digits += String(r[byte=i : i + 1])
        i += 1
    var status_args = '{"build_id":' + digits + "}"

    mcp.select_client(2)
    _init(mcp, studio, assets, pool, "builder-b")
    var foreign = _call(mcp, studio, assets, pool, "code.build_status", status_args)
    assert_true(foreign.find("unknown build_id") >= 0, "another client cannot see or finish this build: " + foreign)

    mcp.select_client(1)
    var waited = 0
    var final = String("")
    while waited < 600:
        var st = _call(mcp, studio, assets, pool, "code.build_status", status_args)
        if st.find('"pending":true') < 0 and st.find('pending\\":true') < 0:
            final = st
            break
        _ = external_call["usleep", Int32](UInt32(200000))
        waited += 1
    assert_true(final.find("compiled") >= 0 and final.find("unknown build_id") < 0, "the owner finishes it: " + final)
    mcp.select_client(0)


def _json_string_bytes(src: String) -> List[UInt8]:
    var raw = List[UInt8]()
    for b in src.as_bytes():
        raw.append(b)
    var doc = parse_json(raw)
    var out = List[UInt8]()
    for b in doc.text(doc.root).as_bytes():
        out.append(b)
    return out^


def test_json_surrogate_pairs() raises:
    var pair = _json_string_bytes('"\\ud83d\\ude00"')
    assert_equal(len(pair), 4, "a surrogate pair is one 4-byte scalar, not CESU-8")
    assert_equal(Int(pair[0]), 0xF0)
    assert_equal(Int(pair[1]), 0x9F)
    assert_equal(Int(pair[2]), 0x98)
    assert_equal(Int(pair[3]), 0x80)
    var lone = _json_string_bytes('"\\ud83d"')
    assert_equal(len(lone), 3, "a lone surrogate becomes U+FFFD")
    assert_equal(Int(lone[0]), 0xEF)
    assert_equal(Int(lone[1]), 0xBF)
    assert_equal(Int(lone[2]), 0xBD)


def test_load_undo_restores_path_and_clean_state() raises:
    var d = _run_dir("load_undo")
    var other = FlintStudioState()
    _ = other.spawn_entity("OtherScene")
    var p2 = d + "/other.flscnb"
    assert_true(other.save_scene_file(p2))

    var studio = FlintStudioState()
    studio.project_root = d
    _ = studio.spawn_entity("Mine")
    var p1 = d + "/mine.flscnb"
    assert_true(studio.save_scene_file(p1))
    assert_true(not studio.dirty)
    assert_true(studio.load_scene_file(p2))
    assert_equal(studio.scene_file_path, p2)
    assert_true(studio.undo())
    assert_equal(studio.scene_file_path, p1, "undoing a load goes back to the previous file")
    assert_true(not studio.dirty, "and to its saved (clean) state")
    assert_true(studio.redo())
    assert_equal(studio.scene_file_path, p2, "redo re-applies the load's path")
    assert_true(len(studio.recent_scenes) >= 2 and studio.recent_scenes[0] == p2, "loads and saves feed Recent")
    var again = FlintStudioState()
    again.project_root = d
    again.load_recent_scenes()
    assert_true(len(again.recent_scenes) >= 2 and again.recent_scenes[0] == p2, "the recent list persists")


def test_recover_autosave_keeps_path_and_stays_dirty() raises:
    var d = _run_dir("recover")
    var studio = FlintStudioState()
    studio.project_root = d
    studio.scene_file_path = d + "/scenes/level.flscnb"
    _ = studio.spawn_entity("Saved")
    assert_true(studio.save_scene_file())
    _ = studio.spawn_entity("Unsaved 1")
    _ = studio.spawn_entity("Unsaved 2")
    studio._sync_ecs_and_physics()
    var with_work = studio.live_entity_count()
    studio.autosave_interval_sec = Float32(1.0)
    assert_true(studio.tick_autosave(Float32(1.5)))
    assert_true(studio.autosave_available())

    var fresh = FlintStudioState()
    fresh.project_root = d
    fresh.scene_file_path = d + "/scenes/level.flscnb"
    assert_true(fresh.load_scene_file())
    assert_true(fresh.autosave_available(), "a newer autosave is offered")
    assert_true(fresh.recover_autosave())
    fresh._sync_ecs_and_physics()
    assert_equal(fresh.live_entity_count(), with_work, "the unsaved work comes back")
    assert_equal(fresh.scene_file_path, d + "/scenes/level.flscnb", "the user's file path is kept")
    assert_true(fresh.dirty, "recovered work is unsaved until the user saves it")


def _gizmo_studio() -> FlintStudioState:
    var studio = FlintStudioState()
    studio.viewport_xf = ViewportTransform.from_logical(RectF(0.0, 0.0, 1280.0, 720.0), 1280.0, 720.0, 1.0)
    return studio^


def test_gizmo_drag_cancel_restores_and_leaves_no_undo() raises:
    var studio = _gizmo_studio()
    var eid = studio.spawn_entity("Dragged")
    studio._sync_ecs_and_physics()
    studio.select_entity(eid)
    studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    studio.gizmo_local_space = False
    var piv = studio.selection_gizmo_pivot()
    assert_true(piv[0])
    var start = studio.entities[eid].translation
    var vx = studio.viewport_xf
    var ws = gizmo_world_scale(piv[1], studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, Int(vx.proj_phys_h()), GIZMO_TARGET_PX)
    var grab = project_world_to_viewport(
        piv[1] + Vec3(1.0, 0.0, 0.0) * (ws * Float32(0.8)), Float32(0.0), Float32(0.0), vx.proj_phys_w(), vx.proj_phys_h(),
        studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
    )
    assert_true(grab[3])
    var undo_n = len(studio.undo_stack)
    assert_true(studio.gizmo_try_begin(grab[0], grab[1]), "pressing on the X arm starts a drag")
    _ = studio.gizmo_drag_update(grab[0] + Float32(120.0), grab[1])
    _ = studio.gizmo_drag_update(grab[0] + Float32(200.0), grab[1])
    assert_true(abs(studio.entities[eid].translation.x - start.x) > Float32(1.0e-3), "the drag moved the entity")
    studio.gizmo_drag_cancel()
    assert_true(abs(studio.entities[eid].translation.x - start.x) < Float32(1.0e-4), "cancel puts it back")
    assert_equal(len(studio.undo_stack), undo_n, "a cancelled drag leaves no undo entry")
    assert_true(not studio.gz_dragging)

    studio.gizmo_mode = GIZMO_MODE_SELECT
    assert_equal(studio.gizmo_pick(grab[0], grab[1]), GIZMO_AXIS_NONE, "the Select tool never grabs a handle")


def test_marquee_and_select_all_skip_locked() raises:
    var studio = _gizmo_studio()
    var a = studio.spawn_entity("A")
    var b = studio.spawn_entity("B")
    studio._sync_ecs_and_physics()
    assert_true(studio.set_entity_locked(b, True))
    assert_true(studio.is_entity_locked(b))
    var hits = studio.entities_in_screen_rect(0.0, 0.0, 1280.0, 720.0)
    var has_a = False
    var has_b = False
    for i in range(len(hits)):
        has_a = has_a or hits[i] == a
        has_b = has_b or hits[i] == b
    assert_true(has_a, "an unlocked entity in view is marquee-selectable")
    assert_true(not has_b, "a locked entity is skipped by the marquee")
    studio.select_all_entities()
    for i in range(len(studio.selected_entity_ids)):
        assert_true(studio.selected_entity_ids[i] != b, "select-all skips locked entities")


def test_outliner_search_and_keyboard() raises:
    var studio = FlintStudioState()
    var parent = studio.spawn_entity("Rig")
    var child = studio.spawn_entity("HiddenHandBone")
    var other = studio.spawn_entity("Zeta")
    assert_true(studio.reparent_entity(child, parent, record_undo=False))
    studio._sync_ecs_and_physics()
    studio.hierarchy_expanded[parent] = False
    var found = studio.hierarchy_rows_for_query("handbone")
    assert_equal(len(found), 1, "search reaches children of collapsed parents")
    assert_equal(found[0].entity_id, child)

    var rows = studio.hierarchy_visible_rows()
    var pi = -1
    for i in range(len(rows)):
        if rows[i].entity_id == parent:
            pi = i
    assert_true(pi >= 0)
    studio.select_entity(parent)
    _ = studio.hierarchy_key_nav(False, False, False, True)
    assert_true(studio.is_hierarchy_expanded(parent), "Right expands a collapsed parent")
    _ = studio.hierarchy_key_nav(False, False, False, True)
    assert_equal(studio.selected_entity_id, child, "Right on an expanded parent steps into its first child")
    _ = studio.hierarchy_key_nav(False, False, True, False)
    assert_equal(studio.selected_entity_id, parent, "Left on a leaf jumps to the parent")
    _ = studio.hierarchy_key_nav(False, False, True, False)
    assert_true(not studio.is_hierarchy_expanded(parent), "Left on an expanded parent collapses it")
    _ = other


def test_redo_keeps_agent_origin() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    mcp.select_client(1)
    _init(mcp, studio, assets, pool, "agent-a")
    _ = _call(mcp, studio, assets, pool, "ecs.spawn_entity", '{"name":"AgentMade"}')
    mcp.select_client(0)
    var top = len(studio.undo_stack) - 1
    var aid = studio.undo_stack[top].agent_id
    assert_true(aid != 0, "an MCP edit carries its agent")
    assert_true(studio.undo())
    assert_true(studio.redo())
    assert_equal(studio.undo_stack[len(studio.undo_stack) - 1].agent_id, aid, "redo keeps the agent tag")


def test_console_keeps_toasts() raises:
    var studio = FlintStudioState()
    studio.push_toast("first", 0, 1.0)
    studio.push_toast("bad thing", 2, 1.0)
    studio.tick_toasts(Float32(5.0))
    assert_true(len(studio.toasts) == 0, "toasts expire")
    var n = len(studio.console_log)
    assert_true(n >= 2)
    assert_equal(studio.console_log[n - 1], "[error] bad thing", "the console keeps expired toasts")
    for i in range(400):
        studio.push_toast("spam " + String(i), 0, 0.1)
    assert_true(len(studio.console_log) <= 300, "the console is capped")


def main() raises:
    test_grants_are_bound_to_args_and_expire()
    print("[PASS] test_grants_are_bound_to_args_and_expire")
    test_approved_call_replays_once_and_retry_is_cached()
    print("[PASS] test_approved_call_replays_once_and_retry_is_cached")
    test_hidden_paths_are_rejected()
    print("[PASS] test_hidden_paths_are_rejected")
    test_build_paths_ownership_and_async()
    print("[PASS] test_build_paths_ownership_and_async")
    test_json_surrogate_pairs()
    print("[PASS] test_json_surrogate_pairs")
    test_load_undo_restores_path_and_clean_state()
    print("[PASS] test_load_undo_restores_path_and_clean_state")
    test_recover_autosave_keeps_path_and_stays_dirty()
    print("[PASS] test_recover_autosave_keeps_path_and_stays_dirty")
    test_gizmo_drag_cancel_restores_and_leaves_no_undo()
    print("[PASS] test_gizmo_drag_cancel_restores_and_leaves_no_undo")
    test_marquee_and_select_all_skip_locked()
    print("[PASS] test_marquee_and_select_all_skip_locked")
    test_outliner_search_and_keyboard()
    print("[PASS] test_outliner_search_and_keyboard")
    test_redo_keeps_agent_origin()
    print("[PASS] test_redo_keeps_agent_origin")
    test_console_keeps_toasts()
    print("[PASS] test_console_keeps_toasts")
    print("[PASS] test_studio_review_fixes: review fixes hold")
