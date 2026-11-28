"""Scene data integrity, undo and save (`tests/test_studio_scene_integrity.mojo`).

  - Saves, autosaves, the layout / recent list / InputMap go through `write_file_durable`: an
    interrupted write leaves the old file whole.
  - File > New gives an untitled scene (the first save asks where), and New is one undo step.
  - Time travel is read-only: no save, autosave, undo or edit while it shows a checkpoint.
  - Reverting an agent step is itself undoable and marks the scene dirty.
  - The clipboard holds editor ids, so a reused slot is never pasted; a load clears it.
  - Duplicate copies every registered component and the whole subtree.
  - Autosaves are keyed by the full scene path and record it; a load offers a newer autosave.
  - Undo history is capped by entries and bytes without deep copies; ~600 edits keep undo working.
  - A world snapshot that fails refuses the edit / load / New instead of recording an empty entry.
  - Esc-cancelling a gizmo drag keeps the redo history.
  - `.flscnb` versions other than 1 and 2 are rejected.
Every file this test writes lives under a per-run directory in `build/`.
"""
from std.ffi import external_call
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.math import Vec3
from flint.platform.os import FAULT_SHORT_WRITE, file_size, make_dirs, read_bytes, write_bytes, write_file_durable
from flint.core.serial import BlobWriter
from flint.render.plugin import MeshRendererComponent, PointLightComponent
from flint.nav.plugin import NavAgentComponent
from flint.scene.scene import Scene
from flint.studio import FlintStudioState
from flint.studio.agent_bus import APPROVAL_EDITS, ROLE_SCENE_BUILDER, STEP_REVERTED
from flint.studio.agent_panel import AgentPanelAction, PANEL_ACT_REVERT_STEP
from flint.studio.gizmo import GIZMO_MODE_TRANSLATE, GIZMO_TARGET_PX, gizmo_world_scale, project_world_to_viewport
from flint.studio.studio_app import MAX_UNDO_BLOB_BYTES, MAX_UNDO_STACK, SPAWN_CARD_POINT_LIGHT, StudioEntityMetaComponent
from flint.studio.viewport_transform import ViewportTransform
from flint.ui.flexbox_dock import RectF


def _run_dir(tag: String) -> String:
    var pid = Int(external_call["getpid", Int32]())
    var d = "build/test_scene_integrity_" + String(pid) + "/" + tag
    _ = make_dirs(d)
    return d


def _bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


def _console_has(studio: FlintStudioState, needle: String) -> Bool:
    for i in range(len(studio.console_log)):
        if studio.console_log[i].find(needle) >= 0:
            return True
    return False


def test_interrupted_write_keeps_old_file() raises:
    var d = _run_dir("durable")
    var p = d + "/scene.flscnb"
    var old = _bytes("old scene contents")
    assert_true(write_file_durable(p, old))
    var new = _bytes("new scene contents that are longer")
    assert_false(write_file_durable(p, new, FAULT_SHORT_WRITE), "a crash mid-write reports failure")
    assert_equal(file_size(p), len(old), "the old file is still whole")
    var studio = FlintStudioState()
    studio.project_root = d
    _ = studio.spawn_entity("Saved")
    assert_true(studio.save_scene_file(d + "/level.flscnb"))
    assert_true(file_size(d + "/level.flscnb") > 0)
    assert_true(file_size(d + "/level.flscnb.tmp") < 0, "the temporary file is renamed over the scene")
    var plain = _bytes("plain")
    assert_true(write_bytes(d + "/plain.bin", plain))


def test_new_scene_is_untitled_and_undoable() raises:
    var d = _run_dir("new_scene")
    var studio = FlintStudioState()
    studio.project_root = d
    var p1 = d + "/level.flscnb"
    _ = studio.spawn_entity("Keep")
    assert_true(studio.save_scene_file(p1))
    var saved_size = file_size(p1)
    _ = studio.spawn_entity("Unsaved")
    studio._sync_ecs_and_physics()
    var with_work = studio.live_entity_count()
    assert_true(studio.dirty)
    var undo_n = len(studio.undo_stack)

    assert_true(studio.new_scene(force=True))
    assert_equal(studio.scene_file_path, "", "New is untitled")
    assert_equal(studio.live_entity_count(), 0)
    assert_false(studio.dirty)
    assert_equal(len(studio.undo_stack), undo_n + 1, "New is one undo step, history is kept")

    _ = studio.spawn_entity("Fresh")
    assert_false(studio.save_scene_file(), "the first save of an untitled scene asks where")
    assert_true(studio.file_modal_open and studio.file_modal_save_as, "Save As opens")
    assert_true(studio.file_modal_path != p1, "the suggestion is not the old level")
    assert_equal(file_size(p1), saved_size, "the old level was not overwritten")
    studio.file_modal_open = False

    assert_true(studio.undo())  # the spawn
    assert_true(studio.undo())  # New
    studio._sync_ecs_and_physics()
    assert_equal(studio.scene_file_path, p1, "undoing New brings the old path back")
    assert_equal(studio.live_entity_count(), with_work, "and the unsaved work")
    assert_true(studio.dirty, "which is still unsaved")
    assert_true(studio.redo())
    studio._sync_ecs_and_physics()
    assert_equal(studio.scene_file_path, "")
    assert_equal(studio.live_entity_count(), 0)


def test_time_travel_is_read_only() raises:
    var d = _run_dir("time_travel")
    var studio = FlintStudioState()
    studio.project_root = d
    studio.scene_file_path = d + "/tt.flscnb"
    _ = studio.spawn_entity("A")
    studio._sync_ecs_and_physics()
    var n = studio.live_entity_count()
    var undo_n = len(studio.undo_stack)
    studio.agent_bus.time_travel_active = True
    studio._time_travel_view_blob = studio._capture_world_blob()
    assert_false(studio.save_scene_file(), "no save while scrubbing")
    assert_true(file_size(d + "/tt.flscnb") < 0)
    studio.autosave_interval_sec = Float32(1.0)
    assert_false(studio.tick_autosave(Float32(2.0)), "no autosave while scrubbing")
    _ = studio.spawn_entity("Blocked")
    studio._sync_ecs_and_physics()
    assert_equal(studio.live_entity_count(), n, "an edit is rolled back")
    assert_equal(len(studio.undo_stack), undo_n, "and leaves no undo entry")
    assert_false(studio.undo(), "no undo while scrubbing")
    studio.agent_bus.time_travel_active = False
    assert_true(studio.undo())


def test_agent_revert_is_undoable() raises:
    var studio = FlintStudioState()
    var aid = studio.agent_bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_EDITS)
    var h0 = studio.state_hash()
    var b0 = studio._capture_world_blob()
    var turn_id = studio.agent_bus.begin_turn(aid, "Spawn", b0, h0, now_sec=1.0)
    var s1 = studio.agent_bus.begin_mcp_tool_step(
        aid, "ecs.spawn_entity", "Spawn Statue", "{}", pre_world_lz4=b0, pre_hash=h0, now_sec=1.1,
    )
    var statue = studio.spawn_entity("AgentStatue", position=Vec3(1.0, 1.0, 1.0), record_undo=True)
    var h1 = studio.state_hash()
    var b1 = studio._capture_world_blob()
    var tx1 = studio.undo_stack[len(studio.undo_stack) - 1].tx_id
    _ = studio.record_agent_step_checkpoint(tx1, Int(aid), Int(turn_id), Int(s1), b0, b1, "Spawn Statue")
    assert_true(studio.agent_bus.complete_mcp_tool_step(
        aid, s1, True, '{"ok":true}', 3.0, tx1,
        entity_id=statue, world_pos=Vec3(1.0, 1.0, 1.0), has_world_pos=True, post_world_lz4=b1, post_hash=h1,
    ))
    studio.mark_dirty(False)
    var undo_n = len(studio.undo_stack)
    var rev = AgentPanelAction(PANEL_ACT_REVERT_STEP)
    rev.agent_id = aid
    rev.step_id = s1
    studio.apply_agent_panel_action(rev)
    assert_equal(studio.state_hash(), h0, "the revert restores the pre-step world")
    var si = studio.agent_bus.find_session_index(aid)
    assert_equal(studio.agent_bus.sessions[si].steps[0].status, STEP_REVERTED)
    assert_equal(len(studio.undo_stack), undo_n + 1, "the revert is one undo step")
    assert_true(studio.dirty, "a revert changes the scene")
    assert_true(studio.undo())
    assert_equal(studio.state_hash(), h1, "undoing the revert brings the agent's work back")


def test_clipboard_uses_editor_ids_and_load_clears_it() raises:
    var d = _run_dir("clipboard")
    var studio = FlintStudioState()
    studio.project_root = d
    var a = studio.spawn_entity("A")
    studio._sync_ecs_and_physics()
    studio.select_entity(a)
    assert_equal(studio.copy_selected(), 1)
    assert_true(studio.despawn_entity(a, record_undo=True))
    var c = studio.spawn_entity("C")
    studio._sync_ecs_and_physics()
    assert_equal(c, a, "the freed slot is reused")
    assert_equal(len(studio.paste_clipboard(record_undo=True)), 0, "a reused slot is not the copied entity")

    studio.select_entity(c)
    assert_equal(studio.copy_selected(), 1)
    assert_true(studio.save_scene_file(d + "/clip.flscnb"))
    assert_true(studio.load_scene_file(d + "/clip.flscnb"))
    assert_equal(len(studio.clipboard_editor_ids), 0, "a load clears the clipboard")


def test_duplicate_copies_all_components_and_subtree() raises:
    var studio = FlintStudioState()
    var p = studio.spawn_entity("Parent")
    var ch = studio.spawn_entity("Child", parent_id=p)
    studio._sync_ecs_and_physics()
    var pe = studio.resolve_entity(p)
    var ce = studio.resolve_entity(ch)
    assert_equal(studio.parent_of(ch), p)
    var nav = NavAgentComponent()
    nav.max_speed = Float32(7.25)
    _ = studio.app.world.insert[NavAgentComponent](ce, nav^)
    var pl = PointLightComponent()
    pl.radius = Float32(3.5)
    _ = studio.app.world.insert[PointLightComponent](pe, pl)
    studio._sync_ecs_and_physics()
    var n = studio.live_entity_count()

    studio.select_entity(p)
    var copy = studio.duplicate_entity(p, record_undo=True)
    studio._sync_ecs_and_physics()
    assert_true(copy >= 0)
    assert_equal(studio.live_entity_count(), n + 2, "the child comes along")
    var copy_e = studio.resolve_entity(copy)
    assert_true(studio.app.world.has[PointLightComponent](copy_e))
    assert_true(abs(studio.app.world.get[PointLightComponent](copy_e).radius - Float32(3.5)) < Float32(1.0e-6))
    var kids = studio.children_of(copy)
    assert_equal(len(kids), 1, "the copy has its own child")
    assert_true(kids[0] != ch)
    var kid_e = studio.resolve_entity(kids[0])
    assert_true(studio.app.world.has[NavAgentComponent](kid_e), "a component the old hand list missed is copied")
    assert_true(abs(studio.app.world.get[NavAgentComponent](kid_e).max_speed - Float32(7.25)) < Float32(1.0e-6))
    assert_equal(studio.parent_of(ch), p, "the original child keeps its parent")
    assert_true(studio.editor_id_of(copy) != studio.editor_id_of(p), "the copy gets its own editor id")
    assert_true(studio.editor_id_of(kids[0]) != studio.editor_id_of(ch))
    assert_true(studio.entities[copy].name == "Parent_Copy")

    # A selected child of a selected parent is duplicated once, with the parent.
    studio.selected_entity_ids.clear()
    studio.selected_entity_ids.append(p)
    studio.selected_entity_ids.append(ch)
    studio.selected_entity_id = p
    var made = studio.duplicate_selected(record_undo=True)
    studio._sync_ecs_and_physics()
    assert_equal(len(made), 1)
    assert_equal(studio.live_entity_count(), n + 4)
    assert_true(studio.undo())
    assert_true(studio.undo())
    studio._sync_ecs_and_physics()
    assert_equal(studio.live_entity_count(), n)


def test_autosave_is_keyed_by_full_path() raises:
    var d = _run_dir("autosave_key")
    var a = FlintStudioState()
    a.project_root = d
    a.scene_file_path = d + "/a/level.flscnb"
    var b = FlintStudioState()
    b.project_root = d
    b.scene_file_path = d + "/b/level.flscnb"
    assert_true(a.autosave_path() != b.autosave_path(), "same basename, different scenes, different autosaves")
    assert_true(a.autosave_path().endswith(".autosave.flscnb"))
    assert_true(a.autosave_path().find("/level-") >= 0)

    _ = a.spawn_entity("Saved")
    assert_true(a.save_scene_file())
    _ = a.spawn_entity("Unsaved")
    a.autosave_interval_sec = Float32(1.0)
    assert_true(a.tick_autosave(Float32(1.5)))
    var raw = read_bytes(a.autosave_path())
    var un = FlintStudioState._unwrap_autosave(raw)
    assert_true(un[0])
    assert_equal(un[1], d + "/a/level.flscnb", "the autosave records the scene it belongs to")
    assert_false(b.autosave_available(), "another scene never sees it")

    var fresh = FlintStudioState()
    fresh.project_root = d
    assert_true(fresh.load_scene_file(d + "/a/level.flscnb"))
    assert_true(_console_has(fresh, "newer autosave"), "loading offers the newer autosave")
    assert_true(fresh.recover_autosave())

    # A file at the autosave path that names another scene is refused.
    var foreign = FlintStudioState._wrap_autosave(d + "/b/level.flscnb", un[2])
    assert_true(write_bytes(fresh.autosave_path(), foreign))
    assert_false(fresh.recover_autosave(), "an autosave of another scene is never restored")

    # Discard on New removes it.
    assert_true(a.tick_autosave(Float32(1.5)) or file_size(a.autosave_path()) > 0)
    a.mark_dirty(True)
    var ap = a.autosave_path()
    assert_true(a.new_scene(force=True))
    assert_true(file_size(ap) < 0, "New after Discard drops the old scene's autosave")


def test_undo_history_cap_with_600_edits() raises:
    var d = _run_dir("undo_cap")
    var studio = FlintStudioState()
    studio.project_root = d
    var eid = studio.spawn_entity("Mover")
    studio._sync_ecs_and_physics()
    studio.select_entity(eid)
    var first_id = studio._next_tx_id
    var total = 600
    for i in range(total):
        var upd = studio.entities[eid].copy()
        upd.translation = Vec3(Float32(i + 1), 0.0, 0.0)
        assert_true(studio.commit_node_edit(eid, upd^, "Move"))
    assert_equal(len(studio.undo_stack), MAX_UNDO_STACK, "history is capped")
    assert_equal(
        studio.undo_stack[0].tx_id, first_id + UInt64(total - MAX_UNDO_STACK), "the oldest entries were evicted"
    )
    assert_equal(studio.undo_stack[len(studio.undo_stack) - 1].tx_id, first_id + UInt64(total - 1))
    assert_true(studio.save_scene_file(d + "/cap.flscnb"))
    assert_false(studio.dirty)
    assert_true(studio.undo())
    studio._sync_ecs_and_physics()
    assert_true(abs(studio.entities[eid].translation.x - Float32(total - 1)) < Float32(1.0e-4))
    assert_true(studio.dirty, "undo past the save is dirty")
    assert_true(studio.redo())
    studio._sync_ecs_and_physics()
    assert_true(abs(studio.entities[eid].translation.x - Float32(total)) < Float32(1.0e-4))
    assert_false(studio.dirty, "redo back to the save is clean")
    var undos = 0
    while studio.undo():
        undos += 1
    assert_equal(undos, MAX_UNDO_STACK, "every kept entry undoes")
    studio._sync_ecs_and_physics()
    assert_true(abs(studio.entities[eid].translation.x - Float32(total - MAX_UNDO_STACK)) < Float32(1.0e-4))
    assert_true(studio.dirty, "the bottom of a trimmed history is not the saved state")
    assert_true(studio.undo_history_bytes() > 0)


def test_capture_failure_refuses_edit_and_load() raises:
    var d = _run_dir("capture_fail")
    var studio = FlintStudioState()
    studio.project_root = d
    _ = studio.spawn_entity("A")
    assert_true(studio.save_scene_file(d + "/ok.flscnb"))
    studio._sync_ecs_and_physics()
    var n = studio.live_entity_count()
    var undo_n = len(studio.undo_stack)
    studio.fail_world_capture_for_test = True
    _ = studio.spawn_entity("B")
    assert_equal(len(studio.undo_stack), undo_n, "no entry with an empty blob is recorded")
    studio._sync_ecs_and_physics()
    var n2 = studio.live_entity_count()
    assert_false(studio.load_scene_file(d + "/ok.flscnb"), "a load without a snapshot is refused")
    studio._sync_ecs_and_physics()
    assert_equal(studio.live_entity_count(), n2, "and the world is left as it was")
    assert_false(studio.new_scene(force=True), "New without a snapshot is refused")
    assert_equal(studio.live_entity_count(), n2)
    studio.fail_world_capture_for_test = False
    assert_true(studio.undo(), "undo still works: the top entry is a real one")
    studio._sync_ecs_and_physics()
    assert_true(studio.live_entity_count() < n2 or studio.live_entity_count() == n - 1)


def test_gizmo_cancel_keeps_redo() raises:
    var studio = FlintStudioState()
    studio.viewport_xf = ViewportTransform.from_logical(RectF(0.0, 0.0, 1280.0, 720.0), 1280.0, 720.0, 1.0)
    var eid = studio.spawn_entity("Dragged")
    _ = studio.spawn_entity("Other")
    studio._sync_ecs_and_physics()
    assert_true(studio.undo())  # "Other" goes to redo
    var redo_n = len(studio.redo_stack)
    assert_true(redo_n > 0)
    studio.select_entity(eid)
    studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    studio.gizmo_local_space = False
    var piv = studio.selection_gizmo_pivot()
    assert_true(piv[0])
    var vx = studio.viewport_xf
    var ws = gizmo_world_scale(piv[1], studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, Int(vx.proj_phys_h()), GIZMO_TARGET_PX)
    var grab = project_world_to_viewport(
        piv[1] + Vec3(1.0, 0.0, 0.0) * (ws * Float32(0.8)), Float32(0.0), Float32(0.0), vx.proj_phys_w(), vx.proj_phys_h(),
        studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
    )
    assert_true(grab[3])
    var undo_n = len(studio.undo_stack)
    assert_true(studio.gizmo_try_begin(grab[0], grab[1]))
    _ = studio.gizmo_drag_update(grab[0] + Float32(150.0), grab[1])
    studio.gizmo_drag_cancel()
    assert_equal(len(studio.undo_stack), undo_n, "a cancelled drag leaves no undo entry")
    assert_equal(len(studio.redo_stack), redo_n, "and keeps the redo history")
    assert_true(studio.redo(), "which still redoes")

    # A press without motion keeps redo too.
    assert_true(studio.undo())
    redo_n = len(studio.redo_stack)
    assert_true(studio.gizmo_try_begin(grab[0], grab[1]))
    studio.gizmo_drag_end()
    assert_equal(len(studio.redo_stack), redo_n)


def _patched_version(packed: List[UInt8], version: UInt32) -> List[UInt8]:
    var raw = lz4_unpack(packed)
    raw[4] = UInt8(version & 255)
    raw[5] = UInt8((version >> 8) & 255)
    raw[6] = UInt8((version >> 16) & 255)
    raw[7] = UInt8((version >> 24) & 255)
    return lz4_pack(raw)


def test_flscnb_rejects_unknown_versions() raises:
    var studio = FlintStudioState()
    _ = studio.spawn_entity("V")
    studio._sync_ecs_and_physics()
    var packed = studio.scene.save_flscnb_bytes(studio.app.world, studio.registry)
    var scene = Scene()
    var v1 = _patched_version(packed, 1)
    var got = scene.load_flscnb_bytes(studio.app.world, studio.registry, v1, clear_existing=True)
    assert_true(len(got) > 0, "v1 files still load")
    for bad in [UInt32(0), UInt32(3), UInt32(99)]:
        var refused = False
        try:
            _ = scene.load_flscnb_bytes(studio.app.world, studio.registry, _patched_version(packed, bad), clear_existing=True)
        except:
            refused = True
        assert_true(refused, "an unknown .flscnb version is rejected")


def _id_by_name(studio: FlintStudioState, name: String) -> Int:
    for i in range(len(studio.entities)):
        if studio.entities[i].alive and studio.entities[i].name == name:
            return i
    return -1


def _set_flags(mut studio: FlintStudioState, eid: Int, enabled: Bool, visible: Bool) raises:
    var upd = studio.entities[eid].copy()
    upd.enabled = enabled
    upd.visible = visible
    assert_true(studio.commit_node_edit(eid, upd^, "Flags"))
    studio._sync_ecs_and_physics()


def _assert_flags(studio: FlintStudioState, name: String, enabled: Bool, visible: Bool, label: String) raises:
    var eid = _id_by_name(studio, name)
    assert_true(eid >= 0, label + ": missing " + name)
    assert_equal(studio.entities[eid].enabled, enabled, label + ": " + name + " enabled")
    assert_equal(studio.entities[eid].visible, visible, label + ": " + name + " visible")
    var e = studio.resolve_entity(eid)
    var mr = studio.app.world.get_ptr[MeshRendererComponent](e)
    if mr:
        assert_equal(
            mr.value()[unsafe_offset=0].visible, enabled and visible, label + ": " + name + " draws only when enabled and visible"
        )


def _flag_scene(mut studio: FlintStudioState) raises:
    var a = studio.spawn_entity("DisabledShown")
    var b = studio.spawn_entity("DisabledHidden")
    var c = studio.spawn_entity("EnabledHidden")
    _ = studio.spawn_entity("Plain")
    var l = studio.spawn_card_entity(SPAWN_CARD_POINT_LIGHT)
    studio._sync_ecs_and_physics()
    var lu = studio.entities[l].copy()
    lu.name = "HiddenLight"
    assert_true(studio.commit_node_edit(l, lu^, "Rename"))
    studio._sync_ecs_and_physics()
    assert_false(studio.entities[l].has_mesh_renderer, "a point light has no mesh")
    _set_flags(studio, a, False, True)
    _set_flags(studio, b, False, False)
    _set_flags(studio, c, True, False)
    _set_flags(studio, l, True, False)


def _assert_flag_scene(studio: FlintStudioState, label: String) raises:
    _assert_flags(studio, "DisabledShown", False, True, label)
    _assert_flags(studio, "DisabledHidden", False, False, label)
    _assert_flags(studio, "EnabledHidden", True, False, label)
    _assert_flags(studio, "Plain", True, True, label)
    _assert_flags(studio, "HiddenLight", True, False, label)


def test_enabled_and_visible_persist_separately() raises:
    var d = _run_dir("enabled_visible")
    var studio = FlintStudioState()
    studio.project_root = d
    _flag_scene(studio)
    _assert_flag_scene(studio, "live")
    for ext in [String(".flscn"), String(".flscnb")]:
        var path = d + "/flags" + ext
        assert_true(studio.save_scene_file(path))
        var fresh = FlintStudioState()
        fresh.project_root = d
        assert_true(fresh.load_scene_file(path))
        fresh._sync_ecs_and_physics()
        _assert_flag_scene(fresh, "reload " + ext)
        # Undo / redo restore both flags exactly.
        _set_flags(fresh, _id_by_name(fresh, "DisabledShown"), True, True)
        _assert_flags(fresh, "DisabledShown", True, True, "edit " + ext)
        assert_true(fresh.undo())
        fresh._sync_ecs_and_physics()
        _assert_flags(fresh, "DisabledShown", False, True, "undo " + ext)
        assert_true(fresh.redo())
        fresh._sync_ecs_and_physics()
        _assert_flags(fresh, "DisabledShown", True, True, "redo " + ext)
        _set_flags(fresh, _id_by_name(fresh, "DisabledHidden"), False, True)
        _assert_flags(fresh, "DisabledHidden", False, True, "Visible on while disabled " + ext)
        assert_true(fresh.undo())
        fresh._sync_ecs_and_physics()
        _assert_flags(fresh, "DisabledHidden", False, False, "undo Visible on a disabled entity " + ext)
        assert_true(fresh.redo())
        fresh._sync_ecs_and_physics()
        _assert_flags(fresh, "DisabledHidden", False, True, "redo Visible on a disabled entity " + ext)


def test_world_snapshot_restores_visible_of_disabled_entity() raises:
    """Time travel (and agent revert) restore a world blob: the view must take Visible from it, not
    from whatever the view showed before."""
    var studio = FlintStudioState()
    _flag_scene(studio)
    var shot = studio._capture_world_blob()
    assert_true(len(shot) > 0)
    _set_flags(studio, _id_by_name(studio, "DisabledShown"), False, False)
    _set_flags(studio, _id_by_name(studio, "DisabledHidden"), False, True)
    _set_flags(studio, _id_by_name(studio, "HiddenLight"), True, True)
    assert_true(studio._restore_world_blob(shot))
    studio._sync_ecs_and_physics()
    _assert_flag_scene(studio, "snapshot restore")


def _smet_tag() -> List[UInt8]:
    var w = BlobWriter()
    w.begin(StudioEntityMetaComponent.SCHEMA_FOURCC, 1)
    w.end()
    var t = List[UInt8]()
    for i in range(4):
        t.append(w.data[i])
    return t^


def test_pre_v3_meta_still_loads() raises:
    """Scenes written before SMET v3 have no Visible flag: enabled meshes take it from the mesh renderer,
    and everything else comes back visible."""
    var studio = FlintStudioState()
    _flag_scene(studio)
    var packed = studio.scene.save_flscnb_bytes(studio.app.world, studio.registry)
    var raw = lz4_unpack(packed)
    var tag = _smet_tag()
    var patched = 0
    var i = 0
    while i + 8 <= len(raw):
        if (
            raw[i] == tag[0] and raw[i + 1] == tag[1] and raw[i + 2] == tag[2] and raw[i + 3] == tag[3]
            and raw[i + 4] == UInt8(3) and raw[i + 5] == 0 and raw[i + 6] == 0 and raw[i + 7] == 0
        ):
            raw[i + 4] = UInt8(2)  # a v2 reader skips the trailing Visible byte (`BlobReader.leave`)
            patched += 1
            i += 8
        else:
            i += 1
    assert_true(patched >= 5, "every meta chunk was rewritten as v2")
    var old = FlintStudioState()
    _ = old.scene.load_flscnb_bytes(old.app.world, old.registry, lz4_pack(raw), clear_existing=True)
    old._sync_ecs_and_physics()
    _assert_flags(old, "DisabledShown", False, True, "v2 flscnb")
    _assert_flags(old, "DisabledHidden", False, True, "v2 flscnb (a pre-v3 disabled mesh comes back visible)")
    _assert_flags(old, "EnabledHidden", True, False, "v2 flscnb")
    _assert_flags(old, "Plain", True, True, "v2 flscnb")
    _assert_flags(old, "HiddenLight", True, True, "v2 flscnb (a light's Visible was never stored)")

    # A pre-v3 .flscn (no "visible" key in the meta object) loads the same way.
    var text = studio.scene.save_flscn_string(studio.app.world, studio.registry)
    assert_true(text.find('"enabled": false, "visible": true}') >= 0, "the meta object ends with its Visible key")
    var stripped = text.replace(', "visible": true}', "}").replace(', "visible": false}', "}")
    var oldj = FlintStudioState()
    _ = oldj.scene.load_flscn_bytes(oldj.app.world, oldj.registry, _bytes(stripped), clear_existing=True)
    oldj._sync_ecs_and_physics()
    _assert_flags(oldj, "DisabledShown", False, True, "v2 flscn")
    _assert_flags(oldj, "EnabledHidden", True, False, "v2 flscn")
    _assert_flags(oldj, "HiddenLight", True, True, "v2 flscn")


def test_undo_byte_budget_evicts_oldest() raises:
    """Drives the byte cap with a small `undo_byte_budget` instead of 64 MB of history."""
    var studio = FlintStudioState()
    assert_equal(studio.undo_byte_budget, MAX_UNDO_BLOB_BYTES, "default budget unchanged")
    var eid = studio.spawn_entity("Mover")
    studio._sync_ecs_and_physics()
    studio.select_entity(eid)
    var first_id = studio._next_tx_id
    var total = 40
    for i in range(total):
        var upd = studio.entities[eid].copy()
        upd.translation = Vec3(Float32(i + 1), 0.0, 0.0)
        assert_true(studio.commit_node_edit(eid, upd^, "Move"))
    assert_equal(len(studio.undo_stack), total + 1, "nothing evicted under the default budget")
    var per_entry = studio.undo_stack[len(studio.undo_stack) - 1].approx_bytes()
    assert_true(per_entry > 0)
    # Undo 4 so the redo stack holds entries too, then shrink the budget to about 10 entries.
    for _ in range(4):
        assert_true(studio.undo())
    assert_equal(len(studio.redo_stack), 4)
    var budget = per_entry * 10 + per_entry // 2
    studio.set_undo_byte_budget(budget)
    assert_true(studio.undo_history_bytes() <= budget, "history fits the budget")
    var kept = len(studio.undo_stack)
    assert_true(kept >= 1 and kept < total - 4, "the oldest undo entries were evicted")
    var newest_undo = first_id + UInt64(total - 5)
    assert_equal(studio.undo_stack[kept - 1].tx_id, newest_undo, "the newest undo entry is kept")
    assert_equal(studio.undo_stack[0].tx_id, newest_undo - UInt64(kept - 1), "kept entries are the newest, in order")
    # Evicting old undo entries met the budget, so redo is intact and still redoes in order.
    assert_equal(len(studio.redo_stack), 4, "redo survives eviction of old undo entries")
    for k in range(4):
        assert_true(studio.redo())
        studio._sync_ecs_and_physics()
        assert_true(abs(studio.entities[eid].translation.x - Float32(total - 3 + k)) < Float32(1.0e-4), "redo order")
    assert_false(studio.redo())
    # Every retained entry still undoes, down to the bottom of the trimmed history.
    var undos = 0
    while studio.undo():
        undos += 1
    studio._sync_ecs_and_physics()
    assert_true(undos >= 1 and undos <= 12)
    assert_true(studio.entities[eid].translation.x > Float32(1.5), "evicted edits are no longer undoable")
    # New edits keep the cap.
    for i in range(20):
        var upd = studio.entities[eid].copy()
        upd.translation = Vec3(Float32(100 + i), 0.0, 0.0)
        assert_true(studio.commit_node_edit(eid, upd^, "Move"))
        assert_true(studio.undo_history_bytes() <= budget or len(studio.undo_stack) == 1)
    assert_equal(len(studio.redo_stack), 0)
    # A budget smaller than one entry still keeps the newest undo entry, and it undoes.
    studio.set_undo_byte_budget(1)
    assert_equal(len(studio.undo_stack), 1, "the newest entry is always kept")
    assert_true(studio.undo())
    studio._sync_ecs_and_physics()
    assert_true(abs(studio.entities[eid].translation.x - Float32(118.0)) < Float32(1.0e-4))
    assert_equal(len(studio.redo_stack), 1)
    assert_true(studio.redo())


def main() raises:
    test_interrupted_write_keeps_old_file()
    print("[PASS] test_interrupted_write_keeps_old_file")
    test_new_scene_is_untitled_and_undoable()
    print("[PASS] test_new_scene_is_untitled_and_undoable")
    test_time_travel_is_read_only()
    print("[PASS] test_time_travel_is_read_only")
    test_agent_revert_is_undoable()
    print("[PASS] test_agent_revert_is_undoable")
    test_clipboard_uses_editor_ids_and_load_clears_it()
    print("[PASS] test_clipboard_uses_editor_ids_and_load_clears_it")
    test_duplicate_copies_all_components_and_subtree()
    print("[PASS] test_duplicate_copies_all_components_and_subtree")
    test_autosave_is_keyed_by_full_path()
    print("[PASS] test_autosave_is_keyed_by_full_path")
    test_undo_history_cap_with_600_edits()
    print("[PASS] test_undo_history_cap_with_600_edits")
    test_capture_failure_refuses_edit_and_load()
    print("[PASS] test_capture_failure_refuses_edit_and_load")
    test_gizmo_cancel_keeps_redo()
    print("[PASS] test_gizmo_cancel_keeps_redo")
    test_flscnb_rejects_unknown_versions()
    print("[PASS] test_flscnb_rejects_unknown_versions")
    test_enabled_and_visible_persist_separately()
    print("[PASS] test_enabled_and_visible_persist_separately")
    test_world_snapshot_restores_visible_of_disabled_entity()
    print("[PASS] test_world_snapshot_restores_visible_of_disabled_entity")
    test_pre_v3_meta_still_loads()
    print("[PASS] test_pre_v3_meta_still_loads")
    test_undo_byte_budget_evicts_oldest()
    print("[PASS] test_undo_byte_budget_evicts_oldest")
    print("[PASS] test_studio_scene_integrity: scene data integrity, undo and save hold")
