"""Phase 2 Verification Gate Suite for Flint Studio Redesign (`tests/test_studio_redesign_ui.mojo`).

Pure Mojo 1.2 verification covering all Wave 1 + Wave 2 foundation modules:
  1. `size_of[GpuUiPrim]() == 64` ABI invariant.
  2. `UiAnimator` analytical springs (critically damped, underdamped, overdamped,
     NaN / `dt <= 0` safety, `UI_ANIM_REDUCED` / `UI_ANIM_INSTANT`, 10k idle rows -> 0 dict entries).
  3. CSS `:focus-visible` compatibility (`ui.focused_id != 0` on frame 0 while `show_focus_ring`
     stays False until keyboard/gamepad nav).
  4. Tier-0 CPU SDF rasterizer (`rasterize_ui_draw_list_cpu`) rendering `PRIM_ARC`,
     `UI_FLAG_OUTER_GLOW`, `UI_FLAG_GLASS_SCRIM`, `UI_FLAG_DASHED_FLOW`,
     `UI_FLAG_PER_CORNER_RADII`, and all 44 `draw_studio_icon` icons.
  5. `FrecencyStore` + `StudioCommandPalette` (fuzzy scoring, PIE safety, `UI_ANIM_INSTANT`),
     `ToastManager` + `JobProgressTracker`, `StudioHeaderBar` + `ViewportHud` (click passthrough,
     3D `OrientationGizmo` snap, 60-frame GPU sparkline).
  6. Harvested Devtools & MCP v2 hooks: `MojoHighlighter` (multi-line state), `DiffDocument`
     (Myers diff + hunk accept/reject), `ProblemStore` (`parse_compiler_output`), `AcpClient`
     deterministic `start_headless()` + `feed_bytes()` + `outbound_log`, `JobRunner` mock output
     -> diagnostics, `CodeDrawer` source/diff/hunk-click, `McpToolEvent` + `MCP_GATE_MUTATING`
     + 3 new MCP tools (`code.build_hot_reload`, `code.diagnostics`, `nav.bake`).
  7. `AgentTelemetryBus` + `agent_theatre_3d` (0-prim idle invariant + active `PRIM_ARC` /
     `UI_FLAG_DASHED_FLOW` / `UI_FLAG_GLASS_SCRIM`) + `AgentMissionControlPanel` (role select,
     prompt run, approval queue, time-travel scrub, revert plan).
"""
from std.collections import List
from std.math import abs, isfinite
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true
from std.time import monotonic

from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.devtools.acp_client import (
    ACP_READY,
    ACP_STARTING,
    AE_DONE,
    AE_PLAN,
    AE_READY,
    AE_THOUGHT,
    AcpClient,
)
from flint.devtools.diff_view import DiffDocument
from flint.devtools.job_runner import JOB_KIND_BUILD_SO as JR_JOB_KIND_BUILD_SO, JobRunner
from flint.devtools.mcp_v2 import (
    MCP_GATE_AUTO,
    MCP_GATE_MUTATING,
    McpServerV2,
)
from flint.devtools.mojo_highlight import (
    LANG_MOJO,
    STATE_NORMAL,
    STATE_TRIPLE_DOUBLE,
    TOK_KEYWORD,
    TOK_STRING,
    MojoHighlighter,
)
from flint.devtools.problems import (
    DIAG_SRC_BUILD,
    ProblemStore,
    parse_compiler_output,
)
from flint.platform.input import InputState
from flint.platform.os import run_process_redirected
from flint.render.font import SdfFont
from flint.render.render_world import (
    DebugLineBatch,
    RENDER_TIER_0_HEADLESS_SIMD,
    make_rh_look_at,
    make_rh_reverse_z_perspective,
)
from flint.render.simd_canvas import SimdCanvas
from flint.render.ui_cpu_raster import rasterize_ui_draw_list_cpu
from flint.studio.agent_bus import (
    APPROVAL_SRC_MCP_TOOL,
    APPROVAL_STEP,
    AgentGhostPreview,
    AgentTelemetryBus,
    GHOST_SPAWN,
    PendingApproval,
    ROLE_GAMEPLAY_CODER,
    ROLE_SCENE_BUILDER,
    STEP_COMPLETED,
    STEP_REVERTED,
)
from flint.studio.agent_panel import (
    AgentMissionControlPanel,
    PANEL_ACT_APPROVE,
    PANEL_ACT_RUN_PROMPT,
    PANEL_ACT_SELECT_OR_SPAWN_ROLE,
)
from flint.studio.agent_theatre_3d import (
    emit_agent_viewport_3d_overlays,
    emit_agent_viewport_hud_overlays,
)
from flint.studio.code_drawer import (
    CODE_ACT_ACCEPT_HUNK,
    CODE_MODE_DIFF_UNIFIED,
    CODE_MODE_SOURCE,
    CodeDrawer,
)
from flint.studio.frecency import FrecencyStore, fuzzy_score
from flint.studio.gizmo import GIZMO_AXIS_NONE, GIZMO_AXIS_X, GIZMO_MODE_SELECT, GIZMO_MODE_TRANSLATE
from flint.studio.studio_panels import OutlinerPanelState, OutlinerRowView, draw_outliner_panel_v2
from flint.studio.notifications import (
    JOB_KIND_BUILD_SO,
    JobProgressTracker,
    PROGRESS_STAGE_EXPANDED,
    PROGRESS_STAGE_PILL,
    PROGRESS_STAGE_QUIET,
    TOAST_ERROR,
    TOAST_INFO,
    TOAST_SUCCESS,
    ToastManager,
    draw_job_progress_pills,
    draw_toast_stack,
)
from flint.studio.palette import (
    PALETTE_ALL,
    PALETTE_COMMANDS,
    PALETTE_ITEM_COMMAND,
    PALETTE_ITEM_SPAWN,
    PaletteItem,
    StudioCommandPalette,
    draw_command_palette,
)
from flint.studio.studio_app import FlintStudioState, _resolve_flint_launcher_path, emit_studio_ui_draw_list
from flint.studio.viewport_hud import (
    HDR_ACT_COMMAND_PALETTE,
    HDR_ACT_WORKSPACE,
    HDR_ACT_WORKSPACE_PICKER,
    STUDIO_HEADER_BAR_H,
    StudioHeaderBar,
    VHUD_ACT_SNAP_VIEW,
    VHUD_ACT_CAM_MODE,
    VHUD_ACT_PIE_PLAY,
    VHUD_PIE_PLAYING,
    VHUD_PIE_STOPPED,
    VHUD_TOOL_MOVE,
    ViewportHud,
    WORKSPACE_AGENT_STUDIO,
    WORKSPACE_EDIT,
    WORKSPACE_LOOKDEV,
)
from flint.ui.animator import (
    SpringConfig,
    UI_ANIM_INSTANT,
    UI_ANIM_NORMAL,
    UI_ANIM_REDUCED,
    UiAnimator,
    UiSpring,
)
from flint.ui.context import (
    UI_KEY_ENTER,
    UI_KEY_TAB,
    UiContext,
)
from flint.ui.draw_list import (
    GpuUiPrim,
    PRIM_ARC,
    UI_FLAG_DASHED_FLOW,
    UI_FLAG_GLASS_SCRIM,
    UI_FLAG_OUTER_GLOW,
    UI_FLAG_PER_CORNER_RADII,
    UiDrawList,
)
from flint.ui.flexbox_dock import RectF
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    AGENT_STATE_EDITING_SCENE,
    AGENT_STATE_IDLE,
    ICON_COUNT,
    StudioTheme,
)


def _count_changed_pixels(canvas: SimdCanvas, bg: UInt32) -> Int:
    var changed = 0
    for i in range(len(canvas.pixels)):
        if canvas.pixels[i] != bg:
            changed += 1
    return changed


def _str_to_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        out.append(sb[i])
    return out^


# ---------------------------------------------------------------------------
# 1. 64-byte GpuUiPrim ABI & UiAnimator Springs
# ---------------------------------------------------------------------------

def test_gpu_prim_abi_and_springs() raises:
    print("  [1/7] GpuUiPrim 64B ABI & UiAnimator analytical springs...")
    assert_equal(size_of[GpuUiPrim](), 64)

    # Critically damped (zeta=1.0): monotonic convergence, zero overshoot
    var s_crit = UiSpring.create(0.0, SpringConfig(34.0, 1.0, 1.0e-4))
    s_crit.set_target(1.0)
    var max_crit = Float32(0.0)
    for _ in range(120):
        var v = s_crit.step(Float32(1.0 / 60.0), UI_ANIM_NORMAL)
        if v > max_crit:
            max_crit = v
    assert_true(s_crit.is_settled())
    assert_true(abs(s_crit.value - Float32(1.0)) < Float32(1.0e-3))
    assert_true(max_crit <= Float32(1.001))

    # Underdamped (zeta=0.45): overshoots target then converges cleanly
    var s_under = UiSpring.create(0.0, SpringConfig(28.0, 0.45, 1.0e-4))
    s_under.set_target(1.0)
    var max_under = Float32(0.0)
    for _ in range(180):
        var v = s_under.step(Float32(1.0 / 60.0), UI_ANIM_NORMAL)
        if v > max_under:
            max_under = v
    assert_true(max_under > Float32(1.05))
    assert_true(s_under.is_settled())
    assert_true(abs(s_under.value - Float32(1.0)) < Float32(1.0e-3))

    # Overdamped (zeta=1.8): monotonic slower convergence
    var s_over = UiSpring.create(0.0, SpringConfig(28.0, 1.8, 1.0e-4))
    s_over.set_target(1.0)
    var prev_over = Float32(0.0)
    for _ in range(240):
        var v = s_over.step(Float32(1.0 / 60.0), UI_ANIM_NORMAL)
        assert_true(v >= prev_over - Float32(1.0e-5))
        assert_true(v <= Float32(1.001))
        prev_over = v
    assert_true(s_over.is_settled())

    # NaN / dt<=0 safety
    var nan_f = Float32(0.0) / Float32(0.0)
    var s_nan = UiSpring.create(0.25, SpringConfig.snappy())
    s_nan.set_target(1.0)
    var v0 = s_nan.step(Float32(0.0), UI_ANIM_NORMAL)
    assert_true(abs(v0 - Float32(0.25)) < Float32(1.0e-6))
    var v_neg = s_nan.step(Float32(-0.5), UI_ANIM_NORMAL)
    assert_true(abs(v_neg - Float32(0.25)) < Float32(1.0e-6))
    var v_nan_dt = s_nan.step(nan_f, UI_ANIM_NORMAL)
    assert_true(isfinite(v_nan_dt))
    s_nan.set_target(nan_f)
    var v_nan_tgt = s_nan.step(Float32(1.0 / 60.0), UI_ANIM_NORMAL)
    assert_true(isfinite(v_nan_tgt))

    # UI_ANIM_REDUCED (zero overshoot even for underdamped spring) & UI_ANIM_INSTANT (immediate snap)
    var s_red = UiSpring.create(0.0, SpringConfig(24.0, 0.35, 1.0e-4))
    s_red.set_target(1.0)
    for _ in range(60):
        var vr = s_red.step(Float32(1.0 / 60.0), UI_ANIM_REDUCED)
        assert_true(vr <= Float32(1.001))
    assert_true(s_red.is_settled())

    var s_inst = UiSpring.create(0.0, SpringConfig.drawer())
    s_inst.set_target(1.0)
    var vi = s_inst.step(Float32(1.0 / 60.0), UI_ANIM_INSTANT)
    assert_equal(vi, Float32(1.0))
    assert_true(s_inst.is_settled())

    # UiAnimator 10k idle rows -> 0 dict entries
    var anim = UiAnimator(UI_ANIM_NORMAL)
    anim.begin_frame(Float32(1.0 / 60.0))
    for i in range(10000):
        var res = anim.drive_widget(UInt64(1000 + i), False, False, False)
        assert_equal(res[0], Float32(0.0))
        assert_equal(res[1], Float32(0.0))
        assert_equal(res[2], Float32(0.0))
    assert_equal(len(anim.widgets), 0)

    # Active widget allocates 1 entry, settles & evicts cleanly back to 0
    _ = anim.drive_widget(UInt64(42), True, False, False)
    assert_equal(len(anim.widgets), 1)
    for _ in range(120):
        anim.begin_frame(Float32(1.0 / 60.0))
        _ = anim.drive_widget(UInt64(42), False, False, False)
    assert_equal(len(anim.widgets), 0)


# ---------------------------------------------------------------------------
# 2. CSS `:focus-visible` vs. M0 `focused_id != 0` Compatibility
# ---------------------------------------------------------------------------

def test_outliner_keyboard_navigation() raises:
    var ui = UiContext()
    var inp = InputState()
    var theme = StudioTheme()
    var bus = AgentTelemetryBus()
    var state = OutlinerPanelState()
    var rows = List[OutlinerRowView]()
    for i in range(24):
        rows.append(OutlinerRowView(i, "Child " + String(i)))
    var rect = RectF(0, 0, 220, 160)
    inp.begin_frame()
    inp.move_mouse(400, 300)
    ui.begin_frame(inp, 640, 360)
    ui._set_focused(ui.widget_id("outliner_v2_world_root"))
    ui.key_end_pressed = True
    var last = draw_outliner_panel_v2(ui, rect, theme, rows, "Scene", 24, 0, False, bus, state)
    assert_equal(last.select_entity_id, 23)
    assert_true(state.scroll_y > 0)
    ui.end_frame()

    inp.begin_frame()
    ui.begin_frame(inp, 640, 360)
    ui._set_focused(ui.widget_id("outliner_v2_search"))
    state.search_query = "語é"
    ui.key_bs_pressed = True
    _ = draw_outliner_panel_v2(ui, rect, theme, rows, "Scene", 24, 0, False, bus, state)
    assert_equal(state.search_query, "語")
    assert_equal(state.scroll_y, Float32(0))
    ui.end_frame()

    inp.begin_frame()
    ui.begin_frame(inp, 640, 360)
    ui._set_focused(ui.widget_id("outliner_v2_plus"))
    ui.key_enter_pressed = True
    var add = draw_outliner_panel_v2(ui, rect, theme, rows, "Scene", 24, 0, False, bus, state)
    assert_true(add.quick_spawn_clicked)
    ui.end_frame()


def test_focus_visible_compatibility() raises:
    print("  [2/7] CSS :focus-visible vs M0 focused_id!=0 compatibility...")
    var ui = UiContext()
    var inp = InputState()
    ui.set_studio_theme(StudioTheme.paper_light())
    assert_equal(ui.theme.bg_window, ui.studio_theme.l0_canvas)
    assert_equal(ui.theme.bg_widget, ui.studio_theme.l3_raised)
    assert_equal(ui.theme.text_primary, ui.studio_theme.text_primary)
    assert_equal(ui.theme.focus_ring, ui.studio_theme.focus_ring)

    # Frame 0: mouse away from button -> M0 auto-focus sets focused_id != 0,
    # yet focus_visible & show_focus_ring MUST be False (no idle blue ring!).
    inp.begin_frame()
    inp.move_mouse(900.0, 600.0)
    ui.begin_frame(inp, 1280.0, 720.0, dt=Float32(1.0 / 60.0))
    _ = ui.button("btn_a", "Unparent", 100.0, 26.0)
    _ = ui.button("btn_b", "Inspect", 100.0, 26.0)
    var wid_a = ui.widget_id("btn_a")
    var wid_b = ui.widget_id("btn_b")
    assert_equal(ui.focused_id, wid_a)
    assert_false(ui.focus_visible)
    assert_false(ui.show_focus_ring(wid_a))
    ui.end_frame()

    # Frame 1: keyboard Tab activates :focus-visible and moves focus to btn_b on end_frame()
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_TAB, True)
    ui.begin_frame(inp, 1280.0, 720.0, dt=Float32(1.0 / 60.0))
    _ = ui.button("btn_a", "Unparent", 100.0, 26.0)
    _ = ui.button("btn_b", "Inspect", 100.0, 26.0)
    ui.end_frame()
    assert_equal(ui.focused_id, wid_b)
    assert_true(ui.focus_visible)
    assert_true(ui.show_focus_ring(wid_b))
    assert_false(ui.show_focus_ring(wid_a))
    inp.set_logical_key(UI_KEY_TAB, False)

    # Frame 2: pointer click clears focus_visible
    inp.begin_frame()
    inp.move_mouse(50.0, 14.0)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1280.0, 720.0, dt=Float32(1.0 / 60.0))
    _ = ui.button("btn_a", "Unparent", 100.0, 26.0)
    _ = ui.button("btn_b", "Inspect", 100.0, 26.0)
    assert_false(ui.focus_visible)
    assert_false(ui.show_focus_ring(ui.focused_id))
    ui.end_frame()


# ---------------------------------------------------------------------------
# 3. Tier-0 CPU SDF Rasterizer Parity & All 44 Studio Icons
# ---------------------------------------------------------------------------

def test_tier0_cpu_raster_and_all_44_icons() raises:
    print("  [3/7] Tier-0 CPU SDF rasterizer & all 44 Studio icons...")
    var bg: UInt32 = 0xFF101318
    var fonts = List[SdfFont]()
    fonts.append(SdfFont())

    # Verify each new SDF primitive/flag renders non-zero pixels on Tier-0 CPU rasterizer
    # A. PRIM_ARC
    var c_arc = SimdCanvas(64, 64)
    c_arc.clear(bg, bg)
    var dl_arc = UiDrawList()
    dl_arc.add_arc(32.0, 32.0, 18.0, 4.0, 0.0, 4.2, 0xFF38BDF8, track_color=0x44FFFFFF, glow_softness=5.0, flags=UI_FLAG_OUTER_GLOW)
    assert_equal(dl_arc.prims[0].header & UInt32(0xF), PRIM_ARC)
    rasterize_ui_draw_list_cpu(c_arc, dl_arc, fonts, ui_time_sec=0.5)
    assert_true(_count_changed_pixels(c_arc, bg) > 120)

    # B. UI_FLAG_OUTER_GLOW
    var c_glow = SimdCanvas(64, 64)
    c_glow.clear(bg, bg)
    var dl_glow = UiDrawList()
    dl_glow.add_glow_rect(16.0, 16.0, 32.0, 32.0, 0xFF222835, 0xFFA78BFA, corner_radius=6.0, glow_radius=8.0)
    assert_true(((dl_glow.prims[0].header >> 16) & UI_FLAG_OUTER_GLOW) != 0)
    rasterize_ui_draw_list_cpu(c_glow, dl_glow, fonts, ui_time_sec=0.0)
    assert_true(_count_changed_pixels(c_glow, bg) > 1000)

    # C. UI_FLAG_GLASS_SCRIM
    var c_glass = SimdCanvas(64, 64)
    c_glass.clear(0xFF204060, 0xFF603020)
    var snap_before = c_glass.pixels.copy()
    var dl_glass = UiDrawList()
    dl_glass.add_glass_panel(8.0, 8.0, 48.0, 48.0, 0xCC141821, corner_radius=8.0, blur_radius=3.0)
    assert_true(((dl_glass.prims[0].header >> 16) & UI_FLAG_GLASS_SCRIM) != 0)
    rasterize_ui_draw_list_cpu(c_glass, dl_glass, fonts, ui_time_sec=0.0)
    var glass_diff = 0
    for i in range(len(snap_before)):
        if c_glass.pixels[i] != snap_before[i]:
            glass_diff += 1
    assert_true(glass_diff > 1500)

    # D. UI_FLAG_DASHED_FLOW (animated Bezier flow changes pixels across time)
    var c_flow0 = SimdCanvas(96, 64)
    var c_flow1 = SimdCanvas(96, 64)
    c_flow0.clear(bg, bg)
    c_flow1.clear(bg, bg)
    var dl_flow = UiDrawList()
    dl_flow.add_bezier_flow(8.0, 52.0, 48.0, 8.0, 88.0, 52.0, 3.0, 0xFF38BDF8)
    assert_true(((dl_flow.prims[0].header >> 16) & UI_FLAG_DASHED_FLOW) != 0)
    rasterize_ui_draw_list_cpu(c_flow0, dl_flow, fonts, ui_time_sec=0.0)
    rasterize_ui_draw_list_cpu(c_flow1, dl_flow, fonts, ui_time_sec=0.35)
    assert_true(_count_changed_pixels(c_flow0, bg) > 40)
    var phase_diff = 0
    for i in range(len(c_flow0.pixels)):
        if c_flow0.pixels[i] != c_flow1.pixels[i]:
            phase_diff += 1
    assert_true(phase_diff > 10)

    # E. UI_FLAG_PER_CORNER_RADII
    var c_pc = SimdCanvas(64, 64)
    c_pc.clear(bg, bg)
    var dl_pc = UiDrawList()
    dl_pc.add_rounded_rect_ex(8.0, 8.0, 48.0, 48.0, 0xFF4C9BFF, 16.0, 0.0, 12.0, 2.0)
    assert_true(((dl_pc.prims[0].header >> 16) & UI_FLAG_PER_CORNER_RADII) != 0)
    rasterize_ui_draw_list_cpu(c_pc, dl_pc, fonts, ui_time_sec=0.0)
    assert_true(_count_changed_pixels(c_pc, bg) > 1500)
    # Top-left corner (9, 9) rounded off -> still bg; top-right corner (54, 9) sharp -> filled
    assert_equal(c_pc.pixels[9 * 64 + 9], bg)
    assert_true(c_pc.pixels[9 * 64 + 54] != bg)

    # F. All 45 Studio SDF icons render non-empty crisp pixels
    assert_equal(ICON_COUNT, 45)
    for icon_id in range(ICON_COUNT):
        var c_ic = SimdCanvas(32, 32)
        c_ic.clear(bg, bg)
        var dl_ic = UiDrawList()
        draw_studio_icon(dl_ic, icon_id, 4.0, 4.0, 24.0, 0xFFEDF1F7, Float32(1.75))
        assert_true(dl_ic.count() > 0)
        rasterize_ui_draw_list_cpu(c_ic, dl_ic, fonts, ui_time_sec=0.0)
        var lit = _count_changed_pixels(c_ic, bg)
        assert_true(lit >= 12)


# ---------------------------------------------------------------------------
# 4. Frecency, Command Palette, Notifications, HeaderBar & ViewportHud
# ---------------------------------------------------------------------------

def test_ux_palette_notifications_header_and_hud() raises:
    print("  [4/7] Frecency, CommandPalette, Notifications, HeaderBar & ViewportHud...")
    var theme = StudioTheme.flint_obsidian()
    var font = SdfFont()

    # Frecency + fuzzy scoring
    var frec = FrecencyStore()
    frec.note("cmd:spawn_light", 1000)
    frec.note("cmd:spawn_light", 1020)
    frec.note("cmd:save_scene", 1000)
    assert_true(frec.score("cmd:spawn_light", 1050) > frec.score("cmd:save_scene", 1050))
    var fm = fuzzy_score("spl", "Spawn Point Light")
    assert_true(fm.matched and fm.score > 0 and len(fm.byte_spans) > 0)

    # StudioCommandPalette: PIE safety & UI_ANIM_INSTANT + 0 prims when closed
    var pal = StudioCommandPalette()
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, 10, "Spawn Cube", "Mesh", "+C", 12, "spawn:cube", mutates_scene=True))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, 20, "Toggle Grid", "View", "G", 11, "cmd:grid", mutates_scene=False))

    var anim_inst = UiAnimator(UI_ANIM_INSTANT)
    anim_inst.begin_frame(Float32(1.0 / 60.0))
    var dl_closed = UiDrawList()
    _ = draw_command_palette(pal, dl_closed, 0, font, theme, anim_inst, 1280.0, 720.0, 0.0, 0.0, False, False)
    assert_equal(dl_closed.count(), 0)

    pal.open(PALETTE_ALL, "", pie_active=True, now_sec=1050)
    pal.set_query("Spawn")
    assert_true(len(pal.results) == 1)
    assert_false(pal.results[0].effective_enabled)  # blocked during PIE!
    var act_blocked = pal.handle_key(key_enter=True)
    assert_false(act_blocked.executed)

    pal.set_query(">Toggle")
    assert_equal(pal.mode, PALETTE_COMMANDS)
    assert_true(len(pal.results) == 1 and pal.results[0].effective_enabled)
    var dl_open = UiDrawList()
    _ = draw_command_palette(pal, dl_open, 0, font, theme, anim_inst, 1280.0, 720.0, 0.0, 0.0, False, False)
    assert_true(dl_open.count() > 10)
    var act_ok = pal.handle_key(key_enter=True)
    assert_true(act_ok.executed and act_ok.action_id == 20 and act_ok.closed)

    # ToastManager + JobProgressTracker
    var toasts = ToastManager()
    toasts.note_user_typing()
    _ = toasts.push(TOAST_INFO, "Background scan", "Indexing assets", own_job=False)
    assert_equal(toasts.held_count(), 1)
    _ = toasts.push(TOAST_ERROR, "Shader error", "Line 42", own_job=False)  # errors bypass typing hold
    assert_equal(len(toasts.visible_toasts()), 1)
    toasts.tick(Float32(3.0))  # releases typing hold
    assert_equal(toasts.held_count(), 0)
    assert_equal(len(toasts.visible_toasts()), 2)
    # Dedup increments repeat_count
    _ = toasts.push(TOAST_SUCCESS, "Saved", "main.flscn", own_job=True)
    _ = toasts.push(TOAST_SUCCESS, "Saved", "main.flscn", own_job=True)
    var found_dup = False
    for i in range(len(toasts.items)):
        if toasts.items[i].title == "Saved" and toasts.items[i].repeat_count == 2:
            found_dup = True
    assert_true(found_dup)

    var jobs = JobProgressTracker()
    var jid = jobs.start_job(JOB_KIND_BUILD_SO, "Building gameplay.so")
    assert_true(jid > 0)
    assert_equal(jobs.jobs[0].stage(), PROGRESS_STAGE_QUIET)
    jobs.tick(1500)
    assert_equal(jobs.jobs[0].stage(), PROGRESS_STAGE_PILL)
    jobs.tick(9500)
    assert_equal(jobs.jobs[0].stage(), PROGRESS_STAGE_EXPANDED)

    var dl_notif = UiDrawList()
    var vp = RectF(240.0, STUDIO_HEADER_BAR_H, 760.0, 540.0)
    _ = draw_toast_stack(toasts, dl_notif, 0, font, theme, anim_inst, vp, 0.0, 0.0, False)
    _ = draw_job_progress_pills(jobs, dl_notif, 0, font, theme, anim_inst, vp, 0.0, 0.0, False)
    assert_true(dl_notif.count() > 8)

    # StudioHeaderBar + ViewportHud (passthrough + 60f sparkline + OrientationGizmo snap)
    var hdr = StudioHeaderBar()
    var dl_hdr = UiDrawList()
    var h_act_idle = hdr.draw(
        dl_hdr, 0, font, theme, anim_inst, 1280.0, "main.flscn", True,
        WORKSPACE_EDIT, AGENT_STATE_EDITING_SCENE, 1, 0,
        640.0, 200.0, False, False,
    )
    assert_equal(h_act_idle.kind, 0)
    assert_true(hdr.contains_point(100.0, 16.0))
    assert_false(hdr.contains_point(100.0, 80.0))
    assert_true(dl_hdr.count() > 15)

    # Clicking Ctrl+K pill (around center-right of 1280 header) or workspace pill
    var dl_hdr2 = UiDrawList()
    _ = hdr.draw(
        dl_hdr2, 0, font, theme, anim_inst, 1280.0, "main.flscn", False,
        WORKSPACE_AGENT_STUDIO, AGENT_STATE_IDLE, 0, 0,
        0.0, 0.0, False, False,
    )
    assert_true(dl_hdr2.count() > 15)

    # Native window controls and a long title must leave a reachable workspace control.
    var workspace_reachable = False
    for x in range(4, 740, 8):
        var probe = UiDrawList()
        var hit = hdr.draw(
            probe, 0, font, theme, anim_inst, 740, "mountain_survey_lighting_preview.flscn", True,
            WORKSPACE_EDIT, AGENT_STATE_EDITING_SCENE, 1, 0,
            Float32(x), STUDIO_HEADER_BAR_H / 2, True, True, show_window_controls=True,
        )
        if hit.kind == HDR_ACT_WORKSPACE_PICKER or hit.kind == HDR_ACT_WORKSPACE:
            workspace_reachable = True
            break
    assert_true(workspace_reachable)

    var vhud = ViewportHud()
    for fi in range(60):
        vhud.push_gpu_ms(Float32(4.0 + Float32(fi % 5) * Float32(0.4)))
    assert_true(vhud.latest_gpu_ms() >= Float32(4.0))

    var dl_hud = UiDrawList()
    var cam_r = Vec3(1.0, 0.0, 0.0)
    var cam_u = Vec3(0.0, 1.0, 0.0)
    var cam_f = Vec3(0.0, 0.0, -1.0)
    var v_act_center = vhud.draw(
        dl_hud, 0, font, theme, anim_inst, vp,
        VHUD_TOOL_MOVE, False, True, Float32(0.5),
        VHUD_PIE_STOPPED, False, False, False,
        False, False, True,
        cam_r, cam_u, cam_f, 1, "",
        vp.x + vp.w * Float32(0.5), vp.y + vp.h * Float32(0.5), False, True,
    )
    # Center of viewport must pass straight through!
    assert_equal(v_act_center.kind, 0)
    assert_false(vhud.contains_point(vp.x + vp.w * Float32(0.5), vp.y + vp.h * Float32(0.5)))
    assert_true(len(vhud.hit_rects) >= 4)
    assert_true(dl_hud.count() > 20)

    # Click OrientationGizmo +X sphere in bottom-right cluster to trigger VHUD_ACT_SNAP_VIEW
    var giz_cx = vp.right() - Float32(48.0)
    var giz_cy = vp.bottom() - Float32(62.0)
    assert_true(vhud.contains_point(giz_cx, giz_cy))
    var dl_snap = UiDrawList()
    var act_s = vhud.draw(
        dl_snap, 0, font, theme, anim_inst, vp,
        VHUD_TOOL_MOVE, False, True, Float32(0.5),
        VHUD_PIE_STOPPED, False, False, False,
        False, False, True,
        cam_r, cam_u, cam_f, 1, "",
        giz_cx + Float32(27.0), giz_cy, True, True,
    )
    assert_equal(act_s.kind, VHUD_ACT_SNAP_VIEW)
    assert_true(act_s.snap_dir.x < Float32(-0.9))

    # In a narrow viewport, wrapped tools must use the same row for painting and input.
    var narrow = RectF(100, 50, 365, 350)
    for cluster in range(1, 3):
        var wrapped = UiDrawList()
        _ = vhud.draw(
            wrapped, 0, font, theme, anim_inst, narrow, VHUD_TOOL_MOVE, False, True, 0.5,
            VHUD_PIE_STOPPED, False, False, False, False, False, True,
            cam_r, cam_u, cam_f, 1, "", 0, 0, False, False,
        )
        var row = vhud.hit_rects[cluster]
        assert_true(row.y > vhud.hit_rects[cluster - 1].y)
        var pressed = vhud.draw(
            wrapped, 0, font, theme, anim_inst, narrow, VHUD_TOOL_MOVE, False, True, 0.5,
            VHUD_PIE_STOPPED, False, False, False, False, False, True,
            cam_r, cam_u, cam_f, 1, "", row.x + 15, row.center_y(), True, True,
        )
        assert_equal(pressed.kind, VHUD_ACT_PIE_PLAY if cluster == 1 else VHUD_ACT_CAM_MODE)

    # PIE active draws perimeter border
    var dl_pie = UiDrawList()
    _ = vhud.draw(
        dl_pie, 0, font, theme, anim_inst, vp,
        VHUD_TOOL_MOVE, False, True, Float32(0.5),
        VHUD_PIE_PLAYING, False, False, False,
        False, False, True,
        cam_r, cam_u, cam_f, 1, "",
        0.0, 0.0, False, False,
    )
    assert_true(dl_pie.count() > 20)
    _ = HDR_ACT_COMMAND_PALETTE


# ---------------------------------------------------------------------------
# 5. Devtools Harvest (Highlighter, Diff, Problems, ACP, JobRunner, CodeDrawer)
# ---------------------------------------------------------------------------

def test_devtools_harvest_and_code_drawer() raises:
    print("  [5/7] Devtools harvest (Highlighter, Diff, Problems, ACP, JobRunner, CodeDrawer)...")
    # A. MojoHighlighter multi-line triple-quoted string state
    var hl = MojoHighlighter()
    var r0 = hl.highlight_line('def hero(): """start docstring', STATE_NORMAL, LANG_MOJO)
    assert_equal(r0[1], STATE_TRIPLE_DOUBLE)
    var saw_kw = False
    for i in range(len(r0[0])):
        if r0[0][i].token_kind == TOK_KEYWORD:
            saw_kw = True
    assert_true(saw_kw)
    var r1 = hl.highlight_line('still inside docstring"""', r0[1], LANG_MOJO)
    assert_equal(r1[1], STATE_NORMAL)
    assert_true(len(r1[0]) > 0 and r1[0][0].token_kind == TOK_STRING)

    # B. DiffDocument Myers diff + hunk accept/reject
    var old_src = String("def speed() -> Float32:\n    return 5.0\n\ndef jump() -> Float32:\n    return 8.0\n")
    var new_src = String("def speed() -> Float32:\n    return 9.5\n\ndef jump() -> Float32:\n    return 12.0\n")
    var ddoc = DiffDocument("src/gameplay.mojo", old_src, new_src)
    assert_equal(ddoc.hunk_count(), 2)
    assert_true(ddoc.accept_hunk(0))
    assert_true(ddoc.reject_hunk(1))
    assert_equal(ddoc.pending_count(), 0)
    var mat = ddoc.materialized_text()
    assert_true(mat.find("9.5") >= 0 and mat.find("8.0") >= 0 and mat.find("12.0") < 0)

    # C. ProblemStore + parse_compiler_output
    var raw_err = String("src/gameplay.mojo:14:9: error: use of unknown declaration 'vel'\nsrc/gameplay.mojo:20:5: warning: unused variable 'tmp'\n")
    var diags = parse_compiler_output(raw_err, source=DIAG_SRC_BUILD)
    assert_equal(len(diags), 2)
    var pstore = ProblemStore()
    pstore.replace_source(DIAG_SRC_BUILD, diags)
    assert_equal(pstore.error_count(), 1)
    assert_equal(pstore.warning_count(), 1)

    # D. AcpClient deterministic start_headless() + feed_bytes() + outbound_log
    var acp = AcpClient()
    acp.start_headless(cwd=".", now_sec=1.0)
    assert_equal(acp.state, ACP_STARTING)
    var out0 = acp.take_outbound_log()
    assert_true(len(out0) >= 1 and out0[0].find("initialize") >= 0)

    # Respond to initialize (id=1), then session/new (id=2), then thought + plan + prompt done (id=3)
    acp.feed_bytes(_str_to_bytes('{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1}}\n'))
    acp.pump(now_sec=1.1)
    var out1 = acp.take_outbound_log()
    assert_true(len(out1) >= 1 and out1[0].find("session/new") >= 0)

    acp.feed_bytes(_str_to_bytes('{"jsonrpc":"2.0","id":2,"result":{"sessionId":"sess-det-1"}}\n'))
    acp.pump(now_sec=1.2)
    assert_equal(acp.state, ACP_READY)

    assert_true(acp.send_prompt("Spawn a torch", now_sec=1.3))
    var out2 = acp.take_outbound_log()
    assert_true(len(out2) >= 1 and out2[0].find("session/prompt") >= 0)

    acp.feed_bytes(_str_to_bytes(
        '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-det-1","update":{"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"Checking lighting"}}}}\n'
        + '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-det-1","update":{"sessionUpdate":"plan","entries":[{"content":"Spawn PointLight","status":"in_progress"}]}}}\n'
        + '{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}\n'
    ))
    acp.pump(now_sec=1.5)
    var saw_ready = False
    var saw_thought = False
    var saw_plan = False
    var saw_done = False
    for ei in range(len(acp.events)):
        var k = acp.events[ei].kind
        if k == AE_READY:
            saw_ready = True
        elif k == AE_THOUGHT:
            saw_thought = True
        elif k == AE_PLAN:
            saw_plan = True
        elif k == AE_DONE:
            saw_done = True
    assert_true(saw_ready and saw_thought and saw_plan and saw_done)
    acp.stop()

    # E. JobRunner mock output -> diagnostics
    var jr = JobRunner()
    var jid = jr.add_mock_job(JR_JOB_KIND_BUILD_SO, "Build gameplay.so", ".")
    var jidx = jr.find_index(jid)
    assert_true(jidx >= 0)
    jr.jobs[jidx].feed_mock_output("src/player.mojo:42:12: error: cannot assign Float32 to Int\n", finish_exit_code=1, now_sec=2.0)
    assert_equal(len(jr.jobs[jidx].diagnostics), 1)
    assert_equal(jr.jobs[jidx].diagnostics[0].line, 42)
    jr.last_completed_id = jid
    var j_diags = jr.latest_diagnostics()
    assert_equal(len(j_diags), 1)
    assert_equal(j_diags[0].line, 42)

    # F. CodeDrawer Source + Unified Diff + clickable hunk Accept
    var cd = CodeDrawer()
    var theme = StudioTheme.flint_obsidian()
    var font = SdfFont()
    var ti = cd.open_source("src/gameplay.mojo", old_src, old_src, False, 1)
    assert_equal(ti, 0)
    cd.view_mode = CODE_MODE_SOURCE
    var dl_cd = UiDrawList()
    var inp_cd = InputState()
    _ = cd.draw_and_handle(dl_cd, 0, font, theme, 0.0, 0.0, 640.0, 420.0, inp_cd, 0.0)
    assert_true(dl_cd.count() > 15)

    _ = cd.show_agent_proposal("src/gameplay.mojo", old_src, new_src, "Tune movement speed")
    cd.view_mode = CODE_MODE_DIFF_UNIFIED
    var dl_diff = UiDrawList()
    _ = cd.draw_and_handle(dl_diff, 0, font, theme, 0.0, 0.0, 640.0, 420.0, inp_cd, 0.1)
    assert_true(dl_diff.count() > 20)
    # Click hunk 0 [Accept] button directly (body_y = 30+24+28 = 82, abx = 640-58-8-58-6 = 510)
    var dl_ck = UiDrawList()
    var ck_in = InputState()
    ck_in.begin_frame()
    ck_in.move_mouse(Float32(535.0), Float32(91.0))
    ck_in.set_mouse(0, True)
    var act_hunk = cd.draw_and_handle(dl_ck, 0, font, theme, 0.0, 0.0, 640.0, 420.0, ck_in, 0.2)
    assert_equal(act_hunk.kind, CODE_ACT_ACCEPT_HUNK)
    assert_equal(act_hunk.hunk_index, 0)


# ---------------------------------------------------------------------------
# 6. MCP v2 Tool Events, Approval Gate & 3 New Tools
# ---------------------------------------------------------------------------

def test_mcp_v2_hooks_gate_and_new_tools() raises:
    print("  [6/7] MCP v2 McpToolEvent, MCP_GATE_MUTATING & 3 new tools...")
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    var assets = AssetServer()
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=".")

    # Verify tools/list advertises the 3 new tools
    var list_resp = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}', studio, assets, pool)
    assert_true(list_resp.find("code.build_hot_reload") >= 0)
    assert_true(list_resp.find("code.diagnostics") >= 0)
    assert_true(list_resp.find("nav.bake") >= 0)

    # Seed a diagnostic & query code.diagnostics (non-mutating)
    var seed_d = parse_compiler_output("src/hero.mojo:7:3: error: bad call\n", source=DIAG_SRC_BUILD)
    mcp.diagnostics_store.replace_source(DIAG_SRC_BUILD, seed_d)
    var diag_resp = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"code.diagnostics","arguments":{"max_items":10}}}',
        studio, assets, pool,
    )
    assert_true(diag_resp.find("hero.mojo") >= 0)

    # Enable MCP_GATE_MUTATING: nav.bake must be blocked until allow_tool_once
    mcp.set_gate_mode(MCP_GATE_MUTATING)
    _ = mcp.take_tool_events()
    var nav_blocked = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"nav.bake","arguments":{}}}',
        studio, assets, pool,
    )
    assert_true(nav_blocked.find("approval") >= 0 or nav_blocked.find("blocked") >= 0 or nav_blocked.find("error") >= 0)
    var evs_blocked = mcp.take_tool_events()
    assert_equal(len(evs_blocked), 1)
    assert_equal(evs_blocked[0].tool_name, "nav.bake")
    assert_true(evs_blocked[0].is_mutating and evs_blocked[0].blocked_by_gate)

    # Allow nav.bake once -> succeeds & emits ok McpToolEvent
    mcp.allow_tool_once("nav.bake")
    var nav_ok = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"nav.bake","arguments":{}}}',
        studio, assets, pool,
    )
    assert_true(nav_ok.find("polygon_count") >= 0)
    var evs_ok = mcp.take_tool_events()
    assert_equal(len(evs_ok), 1)
    assert_true(evs_ok[0].ok and (not evs_ok[0].blocked_by_gate))

    # Exercise code.build_hot_reload (reload_only fast path, no subprocess build needed)
    mcp.set_gate_mode(MCP_GATE_AUTO)
    var br_resp = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"code.build_hot_reload","arguments":{"reload_only":true}}}',
        studio, assets, pool,
    )
    assert_true(br_resp.byte_length() > 0)
    var evs_br = mcp.take_tool_events()
    assert_equal(len(evs_br), 1)
    assert_equal(evs_br[0].tool_name, "code.build_hot_reload")

    # Build the unmodified scaffold from another working directory, including
    # its real flint.app.hot_reload import and a project path with a space.
    var project = "/tmp/flint-studio-edit " + String(monotonic())
    var scaffold = List[String]()
    scaffold.append("/bin/bash")
    scaffold.append(_resolve_flint_launcher_path())
    scaffold.append("new")
    scaffold.append(project)
    assert_equal(run_process_redirected(scaffold), 0)
    var downstream = McpServerV2(workspace_root=project)
    var built = downstream.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"code.build_hot_reload","arguments":{}}}',
        studio, assets, pool,
    )
    var cleanup = List[String]()
    cleanup.append("/bin/rm")
    cleanup.append("-rf")
    cleanup.append("--")
    cleanup.append(project)
    assert_equal(run_process_redirected(cleanup), 0)
    # MCP v2 returns the tool result as JSON text inside `content`, so its quotes arrive escaped.
    var compiled = built.find('"compiled":true') >= 0 or built.find('\\"compiled\\":true') >= 0
    var reloaded = built.find('"reloaded":true') >= 0 or built.find('\\"reloaded\\":true') >= 0
    assert_true(compiled and reloaded, built)

    assets.close()
    studio.shutdown()


# ---------------------------------------------------------------------------
# 7. AgentTelemetryBus, agent_theatre_3d & AgentMissionControlPanel
# ---------------------------------------------------------------------------

def test_agent_bus_theatre_3d_and_panel() raises:
    print("  [7/7] AgentTelemetryBus, agent_theatre_3d & AgentMissionControlPanel...")
    var bus = AgentTelemetryBus()
    var lines_idle = DebugLineBatch()
    var dl_idle = UiDrawList()
    var vp_mat = make_rh_reverse_z_perspective(Float32(55.0 * 3.14159265 / 180.0), Float32(16.0 / 9.0), Float32(0.1), Float32(200.0)).multiply(
        make_rh_look_at(Vec3(0.0, 4.0, 8.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0))
    )

    # Strict 0-prim idle invariant
    assert_false(bus.has_active_visuals())
    _ = emit_agent_viewport_3d_overlays(bus, lines_idle, Float32(0.0))
    var n_hud_idle = emit_agent_viewport_hud_overlays(bus, dl_idle, vp_mat, 0.0, 0.0, 1280.0, 720.0, Float32(0.0))
    assert_equal(lines_idle.segment_count(), 0)
    assert_equal(n_hud_idle, 0)
    assert_equal(dl_idle.count(), 0)

    # Spawn session, begin turn, run MCP step + pending approval ghost
    var aid = bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_STEP)
    var pre_blob = List[UInt8](length=4, fill=UInt8(0x11))
    var post_blob = List[UInt8](length=4, fill=UInt8(0x22))
    var live_blob = List[UInt8](length=4, fill=UInt8(0x33))
    var tid = bus.begin_turn(aid, "Add warm spotlight", pre_blob, UInt64(101), now_sec=1.0)
    assert_true(tid > 0)

    var sid = bus.begin_mcp_tool_step(
        aid,
        "ecs.spawn_entity",
        "Spawned SpotLight Key_Warm",
        '{"name":"Key_Warm"}',
        entity_id=7,
        world_pos=Vec3(0.0, 1.2, 0.0),
        has_world_pos=True,
        pre_world_lz4=pre_blob,
        pre_hash=UInt64(101),
        now_sec=1.1,
    )
    assert_true(sid > 0)
    assert_true(bus.has_active_visuals())

    # Active 3D overlays + HUD overlays emit PRIM_ARC, UI_FLAG_DASHED_FLOW & UI_FLAG_GLASS_SCRIM
    var lines_act = DebugLineBatch()
    var dl_act = UiDrawList()
    _ = emit_agent_viewport_3d_overlays(bus, lines_act, Float32(0.5))
    var n_hud_act = emit_agent_viewport_hud_overlays(bus, dl_act, vp_mat, 0.0, 0.0, 1280.0, 720.0, Float32(0.5))
    assert_true(lines_act.segment_count() > 0)
    assert_true(n_hud_act > 0 and dl_act.count() > 0)

    var saw_arc = False
    var saw_flow = False
    var saw_glass = False
    for pi in range(dl_act.count()):
        var hdr = dl_act.prims[pi].header
        var kind = hdr & UInt32(0xF)
        var flags = hdr >> 16
        if kind == PRIM_ARC:
            saw_arc = True
        if (flags & UI_FLAG_DASHED_FLOW) != 0:
            saw_flow = True
        if (flags & UI_FLAG_GLASS_SCRIM) != 0:
            saw_glass = True
    assert_true(saw_arc and saw_flow and saw_glass)

    # Complete step, test time-travel scrub + revert plan
    assert_true(bus.complete_mcp_tool_step(
        aid, sid, True, '{"entity_id":7}', elapsed_ms=14.2, tx_id=55,
        entity_id=7, world_pos=Vec3(0.0, 1.2, 0.0), has_world_pos=True,
        post_world_lz4=post_blob, post_hash=UInt64(202),
    ))
    var tt_in = bus.enter_time_travel(aid, 0, live_blob, UInt64(303))
    assert_true(tt_in[0] and bus.time_travel_active and tt_in[2] == UInt64(202))
    var tt_out = bus.exit_time_travel()
    assert_true(tt_out[0] and (not bus.time_travel_active) and tt_out[2] == UInt64(303))

    var rev_plan = bus.plan_revert_step(aid, sid)
    assert_true(rev_plan.valid and rev_plan.expected_hash == UInt64(101))
    bus.commit_revert_plan(rev_plan)
    assert_equal(bus.sessions[0].steps[0].status, STEP_REVERTED)

    # Queue PendingApproval & verify AgentMissionControlPanel renders and emits actions
    var pa = PendingApproval(0, aid, sid, APPROVAL_SRC_MCP_TOOL, "Despawn Prop", "Despawn entity #7")
    pa.ghost = AgentGhostPreview(GHOST_SPAWN, aid, sid, 7, "Prop", Vec3(1.0, 1.0, 0.0), Quat.identity(), Vec3(0.5, 0.5, 0.5))
    var appr_id = bus.queue_approval(pa^)
    assert_equal(bus.pending_approval_count(), 1)

    var panel = AgentMissionControlPanel()
    var theme = StudioTheme.flint_obsidian()
    var font = SdfFont()
    panel.set_prompt_text("Light the courtyard")
    var dl_p = UiDrawList()
    var inp_p = InputState()
    _ = panel.render_and_update(dl_p, bus, theme, 0, font, 0.0, 0.0, 360.0, 680.0, inp_p, Float32(1.0 / 60.0), Float32(1.0))
    assert_true(dl_p.count() > 20)

    # Click through panel coordinates to verify role selection & Approve while approval is pending
    var saw_role = False
    var saw_approve = False
    for py in range(12, 460, 10):
        for px in range(12, 348, 18):
            var dl_c = UiDrawList()
            var cin = InputState()
            cin.begin_frame()
            cin.move_mouse(Float32(px), Float32(py))
            cin.set_mouse(0, True)
            var acts = panel.render_and_update(dl_c, bus, theme, 0, font, 0.0, 0.0, 360.0, 680.0, cin, Float32(1.0 / 60.0), Float32(1.0))
            for ai in range(len(acts)):
                var k = acts[ai].kind
                if k == PANEL_ACT_SELECT_OR_SPAWN_ROLE:
                    saw_role = True
                elif k == PANEL_ACT_APPROVE:
                    saw_approve = True
    assert_true(saw_role and saw_approve)
    var res_ap = bus.resolve_approval(appr_id, True)
    assert_true(res_ap[0] and bus.pending_approval_count() == 0)

    # End turn so session is idle, then verify Run prompt button emits PANEL_ACT_RUN_PROMPT
    bus.end_turn(aid, "end_turn", now_sec=1.5)
    var saw_run = False
    for py in range(100, 220, 8):
        for px in range(250, 355, 10):
            var dl_r = UiDrawList()
            var rin = InputState()
            rin.begin_frame()
            rin.move_mouse(Float32(px), Float32(py))
            rin.set_mouse(0, True)
            var acts_r = panel.render_and_update(dl_r, bus, theme, 0, font, 0.0, 0.0, 360.0, 680.0, rin, Float32(1.0 / 60.0), Float32(1.2))
            for ai in range(len(acts_r)):
                if acts_r[ai].kind == PANEL_ACT_RUN_PROMPT:
                    saw_run = True
                    break
        if saw_run:
            break
    assert_true(saw_run)
    _ = ROLE_GAMEPLAY_CODER
    _ = STEP_COMPLETED


def test_compact_workspace_and_select() raises:
    var studio = FlintStudioState(headless=True, width=640, height=480, worker_threads=0)
    studio.set_anim_mode(UI_ANIM_INSTANT)
    studio.set_ui_scale(2)
    var input = InputState()
    var dt = Float32(1.0 / 60.0)

    # A text field owns Q while editing. A fresh Q press in the viewport selects.
    studio.ui_ctx._focused_text_id = UInt64(123)
    input.begin_frame()
    input.set_logical_key(81, True)
    _ = studio.process_editor_input(input, dt)
    assert_equal(studio.gizmo_mode, GIZMO_MODE_TRANSLATE)
    studio.ui_ctx._focused_text_id = 0
    input.begin_frame()
    input.set_logical_key(81, False)
    _ = studio.process_editor_input(input, dt)
    studio.active_gizmo_handle = GIZMO_AXIS_X
    input.begin_frame()
    input.set_logical_key(81, True)
    _ = studio.process_editor_input(input, dt)
    assert_equal(studio.gizmo_mode, GIZMO_MODE_SELECT)
    assert_equal(studio.active_gizmo_handle, GIZMO_AXIS_NONE)
    input.begin_frame()
    input.set_logical_key(81, False)
    input.set_logical_key(87, True)
    _ = studio.process_editor_input(input, dt)
    assert_equal(studio.gizmo_mode, GIZMO_MODE_TRANSLATE)

    # Click the compact workspace switcher at physical coordinates on a 2x screen.
    var dl = UiDrawList()
    var font = SdfFont()
    input.begin_frame()
    input.set_logical_key(87, False)
    input.move_mouse(240, 40)
    input.set_mouse(0, True)
    emit_studio_ui_draw_list(dl, studio, 0, font, 640, 720, input=input)
    assert_true(studio.command_palette.is_open)
    assert_equal(studio.command_palette.raw_query, String("Workspace:"))

    # On a shorter window, wheel through results that no longer fit on screen.
    input.begin_frame()
    input.set_mouse(0, False)
    input.move_mouse(260, 260)
    input.wheel = -5
    emit_studio_ui_draw_list(dl, studio, 0, font, 640, 480, input=input)
    assert_equal(studio.ui_ctx.mouse_delta_x, Float32(10))
    assert_equal(studio.ui_ctx.mouse_delta_y, Float32(110))
    assert_equal(studio.command_palette.viewport_rows, 3)
    assert_true(studio.command_palette.scroll_top > 0)

    input.begin_frame()
    emit_studio_ui_draw_list(dl, studio, 0, font, 640, 960, input=input)
    assert_true(studio.command_palette.viewport_rows >= 5)
    studio.command_palette.set_query("Workspace: Lookdev")
    input.begin_frame()
    input.set_mouse(0, False)
    input.set_logical_key(UI_KEY_ENTER, True)
    _ = studio.process_editor_input(input, dt)
    assert_false(studio.command_palette.is_open)
    assert_equal(studio.workspace_mgr.active_preset, WORKSPACE_LOOKDEV)


def main() raises:
    print("============================================================")
    print(" FLINT STUDIO REDESIGN — PHASE 2 FOUNDATION GATE SUITE")
    print("============================================================")
    test_gpu_prim_abi_and_springs()
    test_focus_visible_compatibility()
    test_outliner_keyboard_navigation()
    test_tier0_cpu_raster_and_all_44_icons()
    test_ux_palette_notifications_header_and_hud()
    test_compact_workspace_and_select()
    test_devtools_harvest_and_code_drawer()
    test_mcp_v2_hooks_gate_and_new_tools()
    test_agent_bus_theatre_3d_and_panel()
    print("[PASS] All 7 Phase 2 Studio Redesign verification suites PASSED!")
