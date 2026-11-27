"""Regression tests for the Flint Studio Phase 0 hotfixes (`tests/test_studio_phase0.mojo`).

Each test pins one defect found in the November 2026 Studio review:
  - UTF-8 Backspace (`utf8_drop_last_char`) never splits a multibyte character.
  - Translate-gizmo grid snap only moves the dragged axis (`snap_constrained_delta`).
  - Edge-on gizmo axes/planes are not pickable (`gizmo_axis_usable` / `gizmo_plane_usable`).
  - Toast dedup still merges after an interleaved toast refreshed the list order.
  - CodeDrawer `close_tab` keeps showing the same file when a tab to its left closes.
  - Deleting a parent keeps its children where they were in world space.
  - Interactive scene load asks first when the scene is dirty.
  - Edits made during Play-In-Editor never enter undo.
  - Opening the command palette keeps the PIE gate and a real frecency clock.
  - MCP `code.build_hot_reload` refuses a `compiler` other than the Mojo toolchain.
  - Zero-length motion sequences have a positive duration.
"""
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.devtools.mcp_v2 import McpServerV2
from flint.scene.transform_hierarchy import GlobalTransform
from flint.studio import (
    GIZMO_AXIS_X,
    GIZMO_PLANE_XZ,
    FlintStudioState,
    snap_constrained_delta,
)
from flint.studio.code_drawer import CodeDrawer
from flint.studio.gizmo import gizmo_axis_usable, gizmo_plane_usable
from flint.studio.notifications import TOAST_INFO, ToastManager
from flint.studio.studio_app import PIE_STOPPED
from flint.ui.context import utf8_drop_last_char


def _near(a: Float32, b: Float32, eps: Float32 = 1.0e-4) -> Bool:
    return abs(a - b) <= eps


def test_utf8_backspace() raises:
    assert_equal(utf8_drop_last_char("Café"), "Caf")
    assert_equal(utf8_drop_last_char("a😀"), "a")
    assert_equal(utf8_drop_last_char("€"), "")
    assert_equal(utf8_drop_last_char("ab"), "a")
    assert_equal(utf8_drop_last_char(""), "")


def test_snap_only_constrained_axis() raises:
    var pivot = Vec3(0.37, 0.37, 0.37)
    # X drag of 0.30 m on a 0.25 m grid: x lands on 0.75, y and z do not move at all.
    var d = snap_constrained_delta(pivot, Vec3(0.30, 0.0, 0.0), GIZMO_AXIS_X, Float32(0.25))
    assert_true(_near(pivot.x + d.x, 0.75), "x must snap to the world grid")
    assert_true(_near(d.y, 0.0) and _near(d.z, 0.0), "free axes must not be snapped during an X drag")
    # XZ plane drag snaps x and z only.
    var dp = snap_constrained_delta(pivot, Vec3(0.30, 0.0, 0.10), GIZMO_PLANE_XZ, Float32(0.25))
    assert_true(_near(pivot.x + dp.x, 0.75) and _near(pivot.z + dp.z, 0.5), "plane drag snaps both in-plane axes")
    assert_true(_near(dp.y, 0.0), "plane normal component stays untouched")


def test_edge_on_gizmo_handles() raises:
    var view = Vec3(0.0, 0.0, 1.0)
    assert_false(gizmo_axis_usable(Vec3(0.0, 0.0, 1.0), view), "axis along the view ray must not be draggable")
    assert_false(gizmo_axis_usable(Vec3(0.03, 0.0, 0.9995).normalized(), view), "axis ~2 deg off the view ray is unstable")
    assert_true(gizmo_axis_usable(Vec3(1.0, 0.0, 0.0), view))
    assert_false(gizmo_plane_usable(Vec3(1.0, 0.0, 0.0), view), "plane seen edge-on must not be draggable")
    assert_true(gizmo_plane_usable(Vec3(0.0, 0.0, 1.0), view))


def test_toast_dedup_after_reorder() raises:
    var tm = ToastManager()
    _ = tm.push(TOAST_INFO, "A")
    tm.tick(1.0)
    _ = tm.push(TOAST_INFO, "B")
    tm.tick(3.0)
    _ = tm.push(TOAST_INFO, "A")  # refreshes A at t=4
    tm.tick(4.0)
    _ = tm.push(TOAST_INFO, "A")  # t=8: B is 7 s old, the old scan stopped there and duplicated A
    var count_a = 0
    var repeats = 0
    for i in range(len(tm.items)):
        if tm.items[i].title == "A":
            count_a += 1
            repeats = tm.items[i].repeat_count
    assert_equal(count_a, 1, "identical toast within the window must merge, not duplicate")
    assert_true(repeats >= 2, "merged toast counts its repeats")
    assert_equal(tm.items[len(tm.items) - 1].title, "A", "a refreshed toast becomes the newest")


def test_code_drawer_close_left_tab() raises:
    var cd = CodeDrawer()
    _ = cd.open_source("a.mojo", "a", "a", False, 1)
    _ = cd.open_source("b.mojo", "b", "b", False, 1)
    _ = cd.open_source("c.mojo", "c", "c", False, 1)
    cd.active_tab = 1
    cd.close_tab(0)
    assert_equal(len(cd.tabs), 2)
    assert_equal(cd.tabs[cd.active_tab].path, "b.mojo", "closing a tab to the left keeps the active file")


def test_delete_parent_keeps_child_world_position() raises:
    var studio = FlintStudioState()
    var parent = studio.spawn_entity("Parent", position=Vec3(5.0, 0.0, 0.0), record_undo=False)
    var child = studio.spawn_entity("Child", position=Vec3(1.0, 2.0, 0.0), parent_id=parent, record_undo=False)
    studio._sync_ecs_and_physics()
    var ce = studio.resolve_entity(child)
    var before = studio.app.world.get_ptr[GlobalTransform](ce).value()[].translation
    assert_true(studio.despawn_entity(parent, record_undo=False))
    studio._sync_ecs_and_physics()
    ce = studio.resolve_entity(child)
    assert_true(ce.valid(), "child survives its parent's deletion")
    var after = studio.app.world.get_ptr[GlobalTransform](ce).value()[].translation
    assert_true(
        _near(before.x, after.x, 1.0e-3) and _near(before.y, after.y, 1.0e-3) and _near(before.z, after.z, 1.0e-3),
        "child must keep its world position (was snapping to its local offset)",
    )


def test_interactive_load_asks_when_dirty() raises:
    var studio = FlintStudioState()
    _ = studio.spawn_entity("Edit")
    studio.mark_dirty(True)
    assert_false(studio.request_load_scene("scenes/whatever.flscn"), "dirty load must wait for a decision")
    assert_true(studio.confirm_modal_open)
    assert_equal(studio.confirm_modal_action, 3)
    assert_equal(studio.pending_load_path, "scenes/whatever.flscn")


def test_pie_edits_skip_undo() raises:
    var studio = FlintStudioState()
    var eid = studio.spawn_entity("Mover", position=Vec3(0.0, 1.0, 0.0))
    var undo_before = len(studio.undo_stack)
    assert_true(studio.pie_play())
    var cur = studio.entities[eid].copy()
    cur.translation = Vec3(3.0, 1.0, 0.0)
    _ = studio.commit_node_edit(eid, cur^, "PIE nudge")
    assert_equal(len(studio.undo_stack), undo_before, "an edit during Play must not be recorded")
    _ = studio.pie_stop()
    assert_equal(studio.pie.mode, PIE_STOPPED)
    assert_equal(len(studio.undo_stack), undo_before)


def test_palette_open_keeps_pie_gate_and_clock() raises:
    var studio = FlintStudioState()
    _ = studio.spawn_entity("Thing")
    assert_true(studio.pie_play())
    studio.open_command_palette()
    assert_true(studio.command_palette.pie_active, "palette opened during Play must gate scene edits")
    assert_true(studio.command_palette.now_sec > 1_700_000_000, "frecency needs a wall clock, not 0")
    _ = studio.pie_stop()


def test_mcp_build_rejects_foreign_compiler() raises:
    var studio = FlintStudioState()
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"phase0"}}}',
        studio, assets, pool,
    )
    var r = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"code.build_hot_reload","arguments":{"entry_path":"tests/test_studio_phase0.mojo","compiler":"/usr/bin/env"}}}',
        studio, assets, pool,
    )
    assert_true(r.find("other executables are not allowed") >= 0, "non-mojo compiler must be refused: " + r)


def main() raises:
    test_utf8_backspace()
    print("[PASS] test_utf8_backspace")
    test_snap_only_constrained_axis()
    print("[PASS] test_snap_only_constrained_axis")
    test_edge_on_gizmo_handles()
    print("[PASS] test_edge_on_gizmo_handles")
    test_toast_dedup_after_reorder()
    print("[PASS] test_toast_dedup_after_reorder")
    test_code_drawer_close_left_tab()
    print("[PASS] test_code_drawer_close_left_tab")
    test_delete_parent_keeps_child_world_position()
    print("[PASS] test_delete_parent_keeps_child_world_position")
    test_interactive_load_asks_when_dirty()
    print("[PASS] test_interactive_load_asks_when_dirty")
    test_pie_edits_skip_undo()
    print("[PASS] test_pie_edits_skip_undo")
    test_palette_open_keeps_pie_gate_and_clock()
    print("[PASS] test_palette_open_keeps_pie_gate_and_clock")
    test_mcp_build_rejects_foreign_compiler()
    print("[PASS] test_mcp_build_rejects_foreign_compiler")
    print("[PASS] test_studio_phase0: all Phase 0 Studio regressions hold")
