"""Regression tests for Studio commands, palette, menus, shortcuts and selection UX
(`tests/test_studio_palette_ux.mojo`).

  - InputMap strict modifiers: in the editor layer, Ctrl/Alt+key never also fires the plain-key action
    (Ctrl+X, Ctrl+G); Shift stays tolerated; game contexts keep the lenient default.
  - Editor shortcuts: Esc clears the selection, Shift+F5 never starts Play, Ctrl+O opens the scene dialog.
  - Palette: Sphere / Plane / Sun spawn real primitives, Prefab opens a prefab picker, Select Tool sets
    the select gizmo, the Toggle commands really toggle, F8 opens the source editor, Save toasts once.
  - Alt+A chord: the next key resolves through `KeyChordHud.resolve_key`, is swallowed, and times out.
  - Menus: `SceneMenu.message` reaches the console; File/Edit items are disabled during Play; the pivot
    title follows the pivot mode.
  - Selection: Shift adds, Ctrl toggles (click and marquee); F2 with several selected never leaves a
    dangling focus request; Hierarchy arrow keys follow the search filter; the arrows have one owner.
  - Inspector: Enabled and Visible stay independent.
  - Command palette: blocks panel pointer input while open; frecency persists across sessions.
All state this test writes lives under a per-run directory in `build/`.
"""
from std.ffi import external_call
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.core.log import state_directory
from flint.input.input_map import ACTION_BOOL, MOD_CTRL, MOD_NONE, InputContext, InputMap
from flint.platform.input import InputState
from flint.platform.os import file_size, make_dirs
from flint.render.plugin import EnvironmentComponent, MeshRendererComponent, PointLightComponent
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD
from flint.studio.gizmo import GIZMO_MODE_SELECT, GIZMO_MODE_TRANSLATE
from flint.studio.palette import PALETTE_ASSETS
from flint.studio.palette_registry import (
    PAL_CMD_AGENT_TOGGLE_THEATRE_3D,
    PAL_CMD_OPEN_SOURCE_EDITOR,
    PAL_CMD_SAVE_SCENE,
    PAL_CMD_SPAWN_PLANE,
    PAL_CMD_SPAWN_PREFAB,
    PAL_CMD_SPAWN_SPHERE,
    PAL_CMD_SPAWN_SUN_LIGHT,
    PAL_CMD_TOGGLE_CODE_DRAWER,
    PAL_CMD_TOOL_SELECT,
    begin_studio_chord,
    studio_chord_key_label,
)
from flint.studio.studio_app import (
    COLLIDER_BOX,
    COLLIDER_SPHERE,
    PIE_STOPPED,
    FlintStudioState,
    StudioEntityMetaComponent,
)
from flint.studio.studio_workspace import PANEL_CODE_DRAWER


def _run_dir(tag: String) -> String:
    var pid = Int(external_call["getpid", Int32]())
    var d = "build/test_palette_ux_" + String(pid) + "/" + tag
    _ = make_dirs(d)
    return d


def _tap(mut studio: FlintStudioState, keys: List[Int]) raises:
    """One frame with `keys` freshly pressed (and held), then one frame with everything released."""
    var inp = InputState()
    inp.begin_frame()
    for i in range(len(keys)):
        inp.keys[keys[i]] = True
        inp.pressed[keys[i]] = True
    _ = studio.process_editor_input(inp, Float32(1.0 / 60.0))
    var up = InputState()
    up.begin_frame()
    _ = studio.process_editor_input(up, Float32(1.0 / 60.0))


def _keys(a: Int, b: Int = -1) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    if b >= 0:
        out.append(b)
    return out^


def test_strict_modifiers_input_map() raises:
    var imap = InputMap()
    _ = imap.register_action("cut", ACTION_BOOL)
    _ = imap.register_action("plain_x", ACTION_BOOL)
    _ = imap.register_action("crouch", ACTION_BOOL)
    _ = imap.register_action("walk", ACTION_BOOL)
    var editor = InputContext("Editor", priority=10, strict_modifiers=True)
    editor.bind_key("cut", 88, MOD_CTRL)
    editor.bind_key("plain_x", 88)
    editor.bind_key("crouch", 341)  # a modifier bound as an action is exempt
    imap.add_context(editor^)
    var game = InputContext("Game", priority=0)
    game.bind_key("walk", 87, MOD_NONE)
    imap.add_context(game^)
    imap.push_context("Game")
    imap.push_context("Editor")

    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(341, True)
    inp.set_logical_key(88, True)
    inp.set_logical_key(87, True)
    imap.update(inp, Float32(1.0 / 60.0))
    assert_true(imap.is_down("cut"), "Ctrl+X fires the chord")
    assert_false(imap.is_down("plain_x"), "Ctrl+X never also fires the plain X action in a strict context")
    assert_true(imap.is_down("crouch"), "the Ctrl key itself still drives an action bound to it")
    assert_true(imap.is_down("walk"), "lenient (game) contexts keep plain keys under Ctrl")

    inp.begin_frame()
    inp.set_logical_key(341, False)
    inp.set_logical_key(342, True)  # Alt
    imap.update(inp, Float32(1.0 / 60.0))
    assert_false(imap.is_down("plain_x"), "Alt+X never fires plain X either")

    inp.begin_frame()
    inp.set_logical_key(342, False)
    inp.set_logical_key(340, True)  # Shift stays tolerated
    imap.update(inp, Float32(1.0 / 60.0))
    assert_true(imap.is_down("plain_x"), "Shift+X still fires X")

    var round_trip = InputMap()
    assert_true(round_trip.load_json(imap.save_json()))
    var strict_found = False
    for ci in range(len(round_trip.contexts)):
        if round_trip.contexts[ci].name == "Editor":
            strict_found = round_trip.contexts[ci].strict_modifiers
    assert_true(strict_found, "strict_modifiers survives save_json/load_json")


def test_editor_shortcuts() raises:
    var studio = FlintStudioState()
    var a = studio.spawn_entity("ShortcutTarget")
    studio.select_entity(a)

    # Ctrl+X / Ctrl+G are not X (gizmo space) / G (grid snap).
    var local0 = studio.gizmo_local_space
    var snap0 = studio.snap_enabled
    _tap(studio, _keys(341, 88))
    _tap(studio, _keys(341, 71))
    assert_equal(studio.gizmo_local_space, local0, "Ctrl+X must not toggle the gizmo space")
    assert_equal(studio.snap_enabled, snap0, "Ctrl+G must not toggle grid snap")
    _tap(studio, _keys(88))
    assert_true(studio.gizmo_local_space != local0, "plain X still toggles the gizmo space")

    # Shift+F5 in Edit never starts Play.
    _tap(studio, _keys(340, 294))
    assert_equal(studio.pie.mode, PIE_STOPPED, "Shift+F5 (Stop) must not hit the F5 Play binding in Edit")

    # Ctrl+O opens the scene dialog (it used to silently revert to the saved file).
    studio.file_modal_open = False
    _tap(studio, _keys(341, 79))
    assert_true(studio.file_modal_open, "Ctrl+O opens the Open Scene dialog")
    studio.file_modal_open = False

    # Esc in Edit clears the selection.
    studio.select_entity(a)
    _tap(studio, _keys(256))
    assert_equal(studio.selected_count(), 0, "Esc clears the selection in Edit mode")


def test_palette_spawns_and_tools() raises:
    var studio = FlintStudioState()

    assert_true(studio.execute_palette_command(PAL_CMD_SPAWN_SPHERE))
    var sph = studio.selected_entity_id
    var sph_e = studio.resolve_entity(sph)
    assert_true(studio.entities[sph].name.startswith("Sphere_"))
    assert_equal(studio.app.world.get_ptr[StudioEntityMetaComponent](sph_e).value()[].collider_kind, COLLIDER_SPHERE)
    assert_true(Bool(studio.app.world.get_ptr[MeshRendererComponent](sph_e)), "a sphere has a mesh")

    assert_true(studio.execute_palette_command(PAL_CMD_SPAWN_PLANE))
    var pl = studio.selected_entity_id
    assert_true(pl != sph)
    assert_true(studio.entities[pl].name.startswith("Plane_"))
    assert_equal(studio.app.world.get_ptr[StudioEntityMetaComponent](studio.resolve_entity(pl)).value()[].collider_kind, COLLIDER_BOX)
    assert_true(abs(studio.entities[pl].scale.x - studio.entities[pl].scale.z) < Float32(1.0e-5))
    assert_true(studio.entities[pl].scale.x < Float32(1.0), "the 36 m grid primitive is scaled down to a floor plane")

    assert_true(studio.execute_palette_command(PAL_CMD_SPAWN_SUN_LIGHT))
    var sun_e = studio.resolve_entity(studio.selected_entity_id)
    assert_true(studio.entities[studio.selected_entity_id].name.startswith("Sun_"))
    assert_true(Bool(studio.app.world.get_ptr[EnvironmentComponent](sun_e)), "Sun is a directional EnvironmentComponent")
    assert_false(Bool(studio.app.world.get_ptr[PointLightComponent](sun_e)), "Sun is not a point light")

    # Prefab: no prefabs -> nothing spawned; with prefabs and none used yet -> a prefab picker opens.
    var n0 = studio.live_entity_count()
    studio.asset_browser_entries.clear()
    assert_false(studio.execute_palette_command(PAL_CMD_SPAWN_PREFAB))
    assert_equal(studio.live_entity_count(), n0, "Spawn Prefab never falls back to an Empty")
    studio.asset_browser_entries.append("prefabs/crate.flprefab")
    assert_true(studio.execute_palette_command(PAL_CMD_SPAWN_PREFAB))
    assert_true(studio.command_palette.is_open, "with no last-used prefab the palette opens as a prefab picker")
    assert_equal(studio.command_palette.mode, PALETTE_ASSETS)
    studio.command_palette.close()
    assert_equal(studio.live_entity_count(), n0)

    studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    assert_true(studio.execute_palette_command(PAL_CMD_TOOL_SELECT))
    assert_equal(studio.gizmo_mode, GIZMO_MODE_SELECT, "Select Tool switches to the select gizmo")

    var shown0 = studio.dock_panel_shown(PANEL_CODE_DRAWER)
    _ = studio.execute_palette_command(PAL_CMD_TOGGLE_CODE_DRAWER)
    assert_true(studio.dock_panel_shown(PANEL_CODE_DRAWER) != shown0, "Toggle Code Drawer flips it")
    _ = studio.execute_palette_command(PAL_CMD_TOGGLE_CODE_DRAWER)
    assert_equal(studio.dock_panel_shown(PANEL_CODE_DRAWER), shown0, "and toggling again flips it back")

    studio.source_editor.requested = False
    assert_true(studio.execute_palette_command(PAL_CMD_OPEN_SOURCE_EDITOR))
    assert_true(studio.source_editor.requested, "F8 / Open Source Editor requests the full-screen editor")
    studio.source_editor.requested = False


def test_save_toasts_once() raises:
    var studio = FlintStudioState()
    _ = studio.spawn_entity("Saved")
    studio.scene_file_path = _run_dir("save") + "/once.flscn"
    var before = len(studio.console_log)
    assert_true(studio.execute_palette_command(PAL_CMD_SAVE_SCENE))
    assert_equal(len(studio.console_log), before + 1, "palette Save shows exactly one toast")


def test_alt_a_chord() raises:
    assert_equal(studio_chord_key_label(InputState()), "")
    var studio = FlintStudioState()
    _tap(studio, _keys(342, 65))  # Alt+A
    assert_true(studio.key_chord_hud.active, "Alt+A arms the chord")
    # W is not in the Alt+A chord: the chord cancels and the key is swallowed (no Move tool switch).
    studio.gizmo_mode = GIZMO_MODE_SELECT
    _tap(studio, _keys(87))
    assert_false(studio.key_chord_hud.active)
    assert_equal(studio.gizmo_mode, GIZMO_MODE_SELECT, "the chord's next key never also fires its hotkey")

    _tap(studio, _keys(342, 65))
    assert_true(studio.key_chord_hud.active)
    var overlays0 = studio.agent_bus.theatre_overlays_enabled
    _tap(studio, _keys(79))  # O -> 3D Theatre Overlays
    assert_false(studio.key_chord_hud.active)
    assert_true(studio.agent_bus.theatre_overlays_enabled != overlays0, "Alt+A, O runs its palette command")
    _ = PAL_CMD_AGENT_TOGGLE_THEATRE_3D

    _ = begin_studio_chord(studio.key_chord_hud, "Alt+A")
    studio.tick_toasts(Float32(10.0))
    assert_false(studio.key_chord_hud.active, "an unanswered chord times out")


def test_menu_messages_and_play_state() raises:
    var studio = FlintStudioState()
    studio.menu.message = "Kimodo is open"
    studio.flush_menu_message()
    assert_equal(studio.menu.message, "")
    assert_equal(studio.console_log[len(studio.console_log) - 1], "Kimodo is open", "menu messages reach the console")

    var eid = studio.spawn_entity("Player")
    studio.select_entity(eid)
    _ = studio.handle_command_menu(InputState(), 1280, 720)
    assert_true(studio.menu.bar.groups[1].items[5].enabled, "Rename is enabled in Edit with one selected")
    assert_true(studio.pie_play())
    _ = studio.handle_command_menu(InputState(), 1280, 720)
    for i in range(len(studio.menu.bar.groups[0].items)):
        assert_false(studio.menu.bar.groups[0].items[i].enabled, "File items are disabled during Play")
    assert_false(studio.menu.bar.groups[1].items[4].enabled, "Select all is disabled during Play")
    assert_false(studio.menu.bar.groups[1].items[5].enabled, "Rename is disabled during Play")
    _ = studio.pie_stop()

    var t0 = studio.menu.bar.groups[2].items[6].title
    studio.toggle_pivot_mode()
    _ = studio.handle_command_menu(InputState(), 1280, 720)
    assert_true(studio.menu.bar.groups[2].items[6].title != t0, "the pivot item title follows the pivot mode")


def test_selection_rules() raises:
    var studio = FlintStudioState()
    var a = studio.spawn_entity("SelA")
    var b = studio.spawn_entity("SelB")
    studio.click_select(a, True, False, False)
    studio.click_select(b, True, True, False)
    assert_equal(studio.selected_count(), 2, "Shift+click adds")
    studio.click_select(b, True, True, False)
    assert_equal(studio.selected_count(), 2, "Shift+click on a selected entity keeps it (add-only)")
    studio.click_select(b, True, False, True)
    assert_equal(studio.selected_count(), 1, "Ctrl+click toggles it out")
    studio.click_select(-1, False, True, False)
    assert_equal(studio.selected_count(), 1, "a modified click on empty space keeps the selection")
    studio.click_select(-1, False, False, False)
    assert_equal(studio.selected_count(), 0, "a plain click on empty space clears")

    # F2 with several selected: no dangling focus request.
    studio.select_entity(a)
    studio.select_entity(b, multi_select=True)
    assert_false(studio.begin_rename_selected())
    assert_false(studio.rename_focus_requested, "rename cannot be honoured for a multi-selection")
    studio.select_entity(a)
    assert_true(studio.begin_rename_selected())
    assert_true(studio.rename_focus_requested)
    studio.rename_focus_requested = False

    # The arrows nudge only when the viewport is hovered and something is selected.
    studio.viewport_hovered = True
    assert_true(studio.arrows_drive_nudge())
    studio.clear_selection()
    assert_false(studio.arrows_drive_nudge(), "nothing selected: the arrows orbit")
    studio.select_entity(a)
    studio.viewport_hovered = False
    assert_false(studio.arrows_drive_nudge(), "viewport not hovered: the arrows orbit")


def test_hierarchy_nav_follows_search() raises:
    var studio = FlintStudioState()
    var a = studio.spawn_entity("AlphaMatch")
    var mid = studio.spawn_entity("BetaOther")
    var c = studio.spawn_entity("GammaMatch")
    studio._sync_ecs_and_physics()
    studio.outliner_state.search_query = "match"
    studio.select_entity(a)
    _ = studio.hierarchy_key_nav(False, True, False, False)
    assert_equal(studio.selected_entity_id, c, "Down skips rows the search filter hides")
    _ = mid
    studio.outliner_state.search_query = ""


def test_enabled_and_visible_are_independent() raises:
    var studio = FlintStudioState()
    var eid = studio.spawn_entity("Toggled")
    var e = studio.resolve_entity(eid)
    var v = studio.entities[eid].copy()
    v.enabled = False
    assert_true(studio.commit_node_edit(eid, v^, "Disable"))
    studio._sync_ecs_and_physics()
    assert_false(studio.entities[eid].enabled)
    assert_true(studio.entities[eid].visible, "disabling keeps the Visible switch on")
    assert_false(studio.app.world.get_ptr[MeshRendererComponent](e).value()[].visible, "a disabled mesh is hidden")

    var v2 = studio.entities[eid].copy()
    v2.enabled = True
    assert_true(studio.commit_node_edit(eid, v2^, "Enable"))
    assert_true(studio.app.world.get_ptr[MeshRendererComponent](e).value()[].visible)

    var v3 = studio.entities[eid].copy()
    v3.visible = False
    assert_true(studio.commit_node_edit(eid, v3^, "Hide"))
    studio._sync_ecs_and_physics()
    assert_true(studio.app.world.get_ptr[StudioEntityMetaComponent](e).value()[].enabled, "hiding keeps the entity enabled")
    assert_false(studio.entities[eid].visible)


def test_palette_blocks_panel_pointer() raises:
    var studio = FlintStudioState(
        headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=640, height=360, worker_threads=0
    )
    studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
    assert_false(studio.ui_ctx._blocked_by_modal())
    studio.open_command_palette()
    studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
    assert_true(studio.ui_ctx.pointer_blocked)
    assert_true(studio.ui_ctx._blocked_by_modal(), "panels ignore the pointer under the palette scrim")
    studio.command_palette.close()
    studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
    assert_false(studio.ui_ctx._blocked_by_modal())
    studio.shutdown()


def test_frecency_persists() raises:
    var path = state_directory("flint") + "/studio_palette_frecency.txt"
    var studio = FlintStudioState()
    studio.open_command_palette()
    studio.command_palette.set_query(">Toggle Reference Grid")
    var res = studio.command_palette.accept_index(0)
    assert_true(res.executed)
    var key = res.frecency_key
    _ = studio.execute_palette_action(res)
    assert_true(file_size(path) > 0, "running a palette command saves frecency next to the layout")
    studio.shutdown()
    var again = FlintStudioState()
    assert_true(again.command_palette.frecency.find(key) >= 0, "a new session loads the saved frecency")


def main() raises:
    # Keep every Studio state file (dock layout, frecency) inside this run's build directory.
    var state = _run_dir("state")
    var name = String("FLINT_STATE_DIR")
    _ = external_call["setenv", Int32](name.as_c_string_span(), state.as_c_string_span(), Int32(1))

    test_strict_modifiers_input_map()
    print("[PASS] test_strict_modifiers_input_map")
    test_editor_shortcuts()
    print("[PASS] test_editor_shortcuts")
    test_palette_spawns_and_tools()
    print("[PASS] test_palette_spawns_and_tools")
    test_save_toasts_once()
    print("[PASS] test_save_toasts_once")
    test_alt_a_chord()
    print("[PASS] test_alt_a_chord")
    test_menu_messages_and_play_state()
    print("[PASS] test_menu_messages_and_play_state")
    test_selection_rules()
    print("[PASS] test_selection_rules")
    test_hierarchy_nav_follows_search()
    print("[PASS] test_hierarchy_nav_follows_search")
    test_enabled_and_visible_are_independent()
    print("[PASS] test_enabled_and_visible_are_independent")
    test_palette_blocks_panel_pointer()
    print("[PASS] test_palette_blocks_panel_pointer")
    test_frecency_persists()
    print("[PASS] test_frecency_persists")
    print("[PASS] test_studio_palette_ux: palette, menu and shortcut fixes hold")
