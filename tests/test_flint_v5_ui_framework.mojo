"""Flint v5 UI framework regressions: text shaping/measure caches, legacy theme sync and WCAG
contrast, drag thresholds, outliner hit-testing and windowing, dock collapse and splitter floors,
scroll views, popup click swallowing, shaped caret hit-testing, HiDPI clip rounding, row IDs,
focusable composite widgets and tree-row GC."""
from std.collections import List
from std.math import abs, isfinite
from std.testing import assert_equal, assert_false, assert_true
from std.time import monotonic
from flint.core.math import Vec3
from flint.platform.input import InputState
from flint.platform.os import read_bytes
from flint.render.font import SdfFont
from flint.text.shaper import ShapedGlyph, TextShaper
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.studio.agent_bus import AgentTelemetryBus
from flint.studio.studio_panels import (
    OutlinerPanelState,
    OutlinerRowView,
    draw_outliner_panel_v2,
    outliner_row_window,
    ascii_contains_ci,
)
from flint.ui.context import UiContext, hash_ui_id
from flint.ui.draw_list import UiDrawList, PRIM_GLYPH, clip_rect_i
from flint.ui.flexbox_dock import DockSpace, RectF, DOCK_SPLIT_H, DOCK_SPLIT_V
from flint.ui.theme import (
    StudioTheme,
    THEME_FLINT_OBSIDIAN,
    THEME_MIDNIGHT_STUDIO,
    THEME_OKABE_ITO,
    THEME_PAPER_LIGHT,
    contrast_ratio_rgba8,
)

comptime TTF_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _have_ttf() -> Bool:
    return len(read_bytes(TTF_PATH)) > 0


def _same_glyphs(a: List[ShapedGlyph], b: List[ShapedGlyph]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if (
            a[i].glyph_id != b[i].glyph_id
            or a[i].font_index != b[i].font_index
            or a[i].cluster != b[i].cluster
            or a[i].codepoint != b[i].codepoint
            or a[i].x_advance != b[i].x_advance
            or a[i].x_offset != b[i].x_offset
            or a[i].y_offset != b[i].y_offset
        ):
            return False
    return True


def _count_glyph_prims(dl: UiDrawList) -> Int:
    var n = 0
    for i in range(len(dl.prims)):
        if (dl.prims[i].header & 0xF) == PRIM_GLYPH:
            n += 1
    return n


def _frame(mut ui: UiContext, mut inp: InputState):
    ui.begin_frame(inp, 1000.0, 700.0)


# ---------------------------------------------------------------------------
# 1. Font shaper cache + per-frame text cache
# ---------------------------------------------------------------------------


def test_font_shaper_cache() raises:
    print("  [1] SdfFont cached shaper == fresh shaper; UiContext text cache...")
    if not _have_ttf():
        print("      (skipped: DejaVuSans.ttf not installed)")
        return
    var font = SdfFont()
    assert_true(font.load_ttf(TTF_PATH, bake_px=32, spread=4))
    assert_true(font.shaper_current())
    var samples = List[String]()
    samples.append("Hello, World")
    samples.append("AVAVA Ta Yo")
    samples.append("office affine")
    samples.append("Grüße — 東京")
    samples.append("")
    for i in range(len(samples)):
        var fresh = TextShaper(font.fallback_chain(), "", 0)
        assert_true(_same_glyphs(font.shape_text(samples[i], 15.0), fresh.shape_text(samples[i], 15.0)))
        var w_fresh: Float32 = 0.0
        for g in fresh.shape_text(samples[i], 15.0):
            w_fresh += g.x_advance
        assert_equal(font.measure_shaped(samples[i], 15.0), w_fresh)

    # Adding a fallback rebuilds the cache (still identical to a fresh chain).
    assert_true(font.add_fallback(TTF_PATH))
    assert_true(font.shaper_current())
    var fresh2 = TextShaper(font.fallback_chain(), "", 0)
    assert_true(_same_glyphs(font.shape_text("Mixed text", 13.0), fresh2.shape_text("Mixed text", 13.0)))
    # Direct edits are caught by the fingerprint: the stale cache is bypassed until resynced.
    font.invalidate_shaper()
    assert_false(font.shaper_current())
    assert_true(_same_glyphs(font.shape_text("Mixed text", 13.0), fresh2.shape_text("Mixed text", 13.0)))
    font.sync_shaper()
    assert_true(font.shaper_current())

    # Cached vs uncached shaping cost (the shaper used to be rebuilt per call).
    var n_iter = 200
    var t0 = monotonic()
    for _ in range(n_iter):
        _ = font.shape_text("Inspector  Transform", 13.0)
    var t1 = monotonic()
    for _ in range(n_iter):
        var tmp = TextShaper(font.fallback_chain(), "", 0)
        _ = tmp.shape_text("Inspector  Transform", 13.0)
    var t2 = monotonic()
    var cached_us = Float64(t1 - t0) / 1000.0 / Float64(n_iter)
    var rebuilt_us = Float64(t2 - t1) / 1000.0 / Float64(n_iter)
    print("      shape_text: cached", cached_us, "us/call vs rebuilt chain", rebuilt_us, "us/call")
    assert_true(cached_us < rebuilt_us)

    # UiContext: measure + draw the same strings reuse one shaped entry.
    var ui = UiContext()
    assert_true(ui.font.load_ttf(TTF_PATH, bake_px=32, spread=4))
    var inp = InputState()
    inp.begin_frame()
    _frame(ui, inp)
    var w1 = ui._measure_text("Directional Light", 13.0)
    var misses_after_first = ui.text_cache_misses
    var w2 = ui._measure_text("Directional Light", 13.0)
    assert_equal(w1, w2)
    assert_equal(w1, ui.font.measure_shaped("Directional Light", 13.0))
    assert_equal(ui.text_cache_misses, misses_after_first)
    assert_true(ui.text_cache_hits >= 1)
    ui.label("Directional Light")
    var glyphs_cached = _count_glyph_prims(ui.draw_list)
    ui.end_frame()
    assert_true(glyphs_cached >= 14)
    # The next frame draws the same label from the cache, with identical output.
    var hits_before = ui.text_cache_hits
    inp.begin_frame()
    _frame(ui, inp)
    ui.label("Directional Light")
    assert_equal(_count_glyph_prims(ui.draw_list), glyphs_cached)
    assert_true(ui.text_cache_hits > hits_before)
    ui.end_frame()
    # Idle entries are swept; live ones survive.
    for _ in range(40):
        inp.begin_frame()
        _frame(ui, inp)
        ui.label("Only this one")
        ui.end_frame()
    assert_true(len(ui._tc_entries) <= 4)

    # Frame cost of 300 labels (measure + draw each): first frame shapes, later frames hit the cache.
    var names = List[String]()
    for i in range(300):
        names.append("Entity_" + String(i) + " Mesh")
    var frame_us = List[Float64]()
    for _ in range(3):
        inp.begin_frame()
        var tf0 = monotonic()
        _frame(ui, inp)
        for i in range(len(names)):
            ui.label(names[i])
        ui.end_frame()
        frame_us.append(Float64(monotonic() - tf0) / 1000.0)
    print("      300 labels: cold frame", frame_us[0], "us, warm frames", frame_us[1], "/", frame_us[2], "us")
    assert_true(frame_us[2] < frame_us[0])

    # rich_label without a display font borrows the UI font (no copy) and still shapes.
    inp.begin_frame()
    _frame(ui, inp)
    ui.rich_label("Press [color=#FF8800]Play[/color] to start")
    assert_true(_count_glyph_prims(ui.draw_list) >= 18)
    ui.end_frame()
    print("      PASS")


# ---------------------------------------------------------------------------
# 2 + 12. Legacy UiTheme sync and WCAG contrast over every theme
# ---------------------------------------------------------------------------


def test_theme_sync_and_contrast() raises:
    print("  [2/12] UiTheme follows StudioTheme; WCAG 4.5:1 text contrast in all themes...")
    var ids = [THEME_FLINT_OBSIDIAN, THEME_MIDNIGHT_STUDIO, THEME_OKABE_ITO, THEME_PAPER_LIGHT]
    for ti in range(len(ids)):
        var t = StudioTheme.by_id(ids[ti])
        var surfaces = [t.l0_canvas, t.l1_shell, t.l2_card, t.l3_raised, t.l3_hover, t.l3_press]
        var texts = [t.text_primary, t.text_secondary, t.text_muted]
        for si in range(len(surfaces)):
            for xi in range(len(texts)):
                var cr = contrast_ratio_rgba8(texts[xi], surfaces[si])
                if cr < Float32(4.5):
                    print("      FAIL", t.name, "text", xi, "on surface", si, "=", cr)
                assert_true(cr >= Float32(4.5))

        var ui = UiContext()
        ui.set_studio_theme(t)
        assert_equal(ui.theme.text_primary, t.text_primary)
        assert_equal(ui.theme.text_muted, t.text_muted)
        assert_equal(ui.theme.bg_widget, t.l3_raised)
        assert_equal(ui.theme.bg_window, t.l0_canvas)
        var bgs = [ui.theme.bg_window, ui.theme.bg_panel, ui.theme.bg_header, ui.theme.bg_widget, ui.theme.bg_widget_hover]
        for bi in range(len(bgs)):
            assert_true(contrast_ratio_rgba8(ui.theme.text_primary, bgs[bi]) >= Float32(4.5))
            assert_true(contrast_ratio_rgba8(ui.theme.text_muted, bgs[bi]) >= Float32(4.5))
        # Selected-text highlight keeps primary text readable.
        assert_true(contrast_ratio_rgba8(ui.theme.text_primary, ui.theme.accent_dim) >= Float32(3.0))

    # Paper Light: a legacy label is drawn in the theme's dark text, not the dark preset's white.
    var ui_l = UiContext()
    ui_l.set_studio_theme(StudioTheme.paper_light())
    var inp = InputState()
    inp.begin_frame()
    _frame(ui_l, inp)
    ui_l.label("Visible?")
    var label_col = ui_l.draw_list.prims[len(ui_l.draw_list.prims) - 1].fill_rgba8 | UInt32(0xFF000000)
    ui_l.end_frame()
    assert_equal(label_col & UInt32(0x00FFFFFF), StudioTheme.paper_light().text_primary & UInt32(0x00FFFFFF))
    # Re-applying the same theme is a no-op; switching re-derives.
    ui_l.set_studio_theme(StudioTheme.paper_light())
    assert_equal(ui_l.theme.bg_widget, StudioTheme.paper_light().l3_raised)
    ui_l.set_studio_theme(StudioTheme.midnight_studio())
    assert_equal(ui_l.theme.bg_widget, StudioTheme.midnight_studio().l3_raised)
    assert_true(StudioTheme.okabe_ito().same_tokens(StudioTheme.okabe_ito()))
    assert_false(StudioTheme.okabe_ito().same_tokens(StudioTheme.paper_light()))
    print("      PASS")


# ---------------------------------------------------------------------------
# 3. Drag sources need a press on the item and 4 px of travel
# ---------------------------------------------------------------------------


def test_drag_threshold() raises:
    print("  [3] begin_drag_source: same item + 4 px threshold...")
    var ui = UiContext()
    var inp = InputState()
    inp.begin_frame()
    inp.move_mouse(30.0, 14.0)
    inp.set_mouse(0, True)
    _frame(ui, inp)
    _ = ui.button("item_a", "A", 120.0, 22.0)
    assert_false(ui.begin_drag_source("ITEM", 1, "a"))
    _ = ui.button("item_b", "B", 120.0, 22.0)
    _ = ui.begin_drag_source("ITEM", 2, "b")
    ui.end_frame()
    assert_false(ui.drag_drop.active)

    # 2 px of jitter: still a click, not a drag.
    inp.begin_frame()
    inp.move_mouse(32.0, 14.0)
    _frame(ui, inp)
    _ = ui.button("item_a", "A", 120.0, 22.0)
    assert_false(ui.begin_drag_source("ITEM", 1, "a"))
    ui.end_frame()
    assert_false(ui.drag_drop.active)

    # Past the threshold, onto item B: the drag carries A (where the press began).
    inp.begin_frame()
    inp.move_mouse(36.0, 40.0)
    _frame(ui, inp)
    _ = ui.button("item_a", "A", 120.0, 22.0)
    var a_drag = ui.begin_drag_source("ITEM", 1, "a")
    _ = ui.button("item_b", "B", 120.0, 22.0)
    var b_drag = ui.begin_drag_source("ITEM", 2, "b")
    ui.end_frame()
    assert_true(ui.drag_drop.active)
    assert_true(a_drag)
    assert_false(b_drag)
    assert_equal(ui.drag_drop.int_value, 1)
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    ui.end_frame()
    assert_false(ui.drag_drop.active)

    # A press that starts off every item and slides over one never drags.
    inp.begin_frame()
    inp.move_mouse(600.0, 600.0)
    inp.set_mouse(0, True)
    _frame(ui, inp)
    _ = ui.button("item_a", "A", 120.0, 22.0)
    _ = ui.begin_drag_source("ITEM", 1, "a")
    ui.end_frame()
    inp.begin_frame()
    inp.move_mouse(30.0, 14.0)
    _frame(ui, inp)
    _ = ui.button("item_a", "A", 120.0, 22.0)
    assert_false(ui.begin_drag_source("ITEM", 1, "a"))
    ui.end_frame()
    assert_false(ui.drag_drop.active)
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    ui.end_frame()
    print("      PASS")


# ---------------------------------------------------------------------------
# 4 + 5. Outliner rows under the search box; windowed rows == full rows
# ---------------------------------------------------------------------------


def _outliner_rows(n: Int) -> List[OutlinerRowView]:
    var rows = List[OutlinerRowView]()
    for i in range(n):
        var r = OutlinerRowView(i, "Entity_" + String(i))
        r.is_selected = i == 3
        rows.append(r^)
    return rows^


def test_outliner_hit_and_window() raises:
    print("  [4/5] Outliner: clipped rows ignore clicks over the search box; windowed rows...")
    var theme = StudioTheme()
    var bus = AgentTelemetryBus()
    var panel = RectF(0.0, 0.0, 260.0, 300.0)
    var rows = _outliner_rows(60)

    var ui = UiContext()
    var inp = InputState()
    var st = OutlinerPanelState()
    st.scroll_y = 42.0  # entity row 0 now sits 12 px above the list, under the search strip
    inp.begin_frame()
    inp.move_mouse(120.0, 20.0)  # inside the search box and inside row 0's rect
    inp.set_mouse(0, True)
    _frame(ui, inp)
    var act = draw_outliner_panel_v2(ui, panel, theme, rows, "World", 60, 1, False, bus, st)
    ui.end_frame()
    assert_equal(act.select_entity_id, -1)
    assert_equal(ui.focused_id, ui.widget_id("outliner_v2_search"))
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    ui.end_frame()

    # A click inside the list still selects.
    var st2 = OutlinerPanelState()
    inp.begin_frame()
    inp.move_mouse(120.0, 36.0 + 4.0 + 26.0 + 12.0)  # entity row 0 when unscrolled
    inp.set_mouse(0, True)
    _frame(ui, inp)
    var act2 = draw_outliner_panel_v2(ui, panel, theme, rows, "World", 60, 1, False, bus, st2)
    ui.end_frame()
    assert_equal(act2.select_entity_id, 0)
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    ui.end_frame()

    # Windowed mode draws exactly what the full list draws.
    var st_full = OutlinerPanelState()
    st_full.scroll_y = 300.0
    var st_win = OutlinerPanelState()
    st_win.scroll_y = 300.0
    var ui_full = UiContext()
    var ui_win = UiContext()
    inp.begin_frame()
    inp.move_mouse(900.0, 600.0)
    _frame(ui_full, inp)
    _ = draw_outliner_panel_v2(ui_full, panel, theme, rows, "World", 60, 1, False, bus, st_full)
    var n_full = len(ui_full.draw_list.prims)
    ui_full.end_frame()
    var win = outliner_row_window(panel, len(rows), st_win)
    assert_true(win[0] > 0 and win[1] < len(rows) and win[1] - win[0] < 30)
    var sub = List[OutlinerRowView]()
    for i in range(win[0], win[1]):
        sub.append(rows[i].copy())
    _frame(ui_win, inp)
    _ = draw_outliner_panel_v2(ui_win, panel, theme, sub, "World", 60, 1, False, bus, st_win, row_base=win[0], total_rows=len(rows))
    var n_win = len(ui_win.draw_list.prims)
    ui_win.end_frame()
    assert_equal(n_win, n_full)
    assert_equal(st_win.scroll_y, st_full.scroll_y)
    assert_true(ascii_contains_ci("Point_Light_02", "light_0"))
    assert_false(ascii_contains_ci("Camera", "light"))
    assert_true(ascii_contains_ci("abc", ""))
    print("      PASS")


# ---------------------------------------------------------------------------
# 6 + 8. Dock collapse reclaims space; splitter px floor
# ---------------------------------------------------------------------------


def test_dock_collapse_and_floor() raises:
    print("  [6/8] DockSpace: collapsed child gets header_height; splitter px floor...")
    var d = DockSpace()
    d.set_dock_style_v2(True)
    d.header_height = 26.0
    var vp = d.add_central_viewport("VP")
    var sp = d.split_node(vp, DOCK_SPLIT_V, 0.7, [String("Console")])
    var root = RectF(0.0, 0.0, 800.0, 600.0)
    d.solve_layout(root)
    assert_true(abs(d.nodes[sp[1]].rect.h - 180.0) < 0.01)
    _ = d.toggle_node_collapsed(sp[1])
    d.solve_layout(root)
    assert_true(abs(d.nodes[sp[1]].rect.h - 26.0) < 0.01)
    assert_true(abs(d.nodes[sp[0]].rect.h - 574.0) < 0.01)
    assert_true(abs(d.split_seam(vp) - 574.0) < 0.01)
    assert_true(abs(d.nodes[vp].ratio - 0.7) < 1.0e-5)  # the stored ratio survives the collapse
    var fp = d.find_panel("Console")
    assert_equal(fp[3].h, Float32(0.0))
    # No splitter to drag on a collapsed side.
    _ = d.interact_splitters(400.0, 574.0, True, True, root)
    assert_equal(d.dragging_node, -1)
    _ = d.toggle_node_collapsed(sp[1])
    d.solve_layout(root)
    assert_true(abs(d.nodes[sp[1]].rect.h - 180.0) < 0.01)

    # Horizontal split: collapsing the left panel leaves a header-wide strip.
    var dh = DockSpace()
    dh.set_dock_style_v2(True)
    var vh = dh.add_central_viewport("VP")
    var sh = dh.split_node(vh, DOCK_SPLIT_H, 0.25, [String("Hierarchy")], place_new_first=True)
    _ = dh.toggle_node_collapsed(sh[0])
    dh.solve_layout(root)
    assert_true(abs(dh.nodes[sh[0]].rect.w - dh.header_height) < 0.01)
    assert_true(abs(dh.nodes[sh[1]].rect.w - (800.0 - dh.header_height)) < 0.01)

    # Splitter floor: dragging to the edge keeps >= min_panel_px.
    var df = DockSpace()
    var vf = df.add_central_viewport("VP")
    var sf = df.split_node(vf, DOCK_SPLIT_H, 0.5, [String("Side")])
    var small = RectF(0.0, 0.0, 300.0, 200.0)
    df.solve_layout(small)
    _ = df.interact_splitters(150.0, 100.0, True, True, small)
    _ = df.interact_splitters(1.0, 100.0, True, False, small)
    assert_true(df.nodes[sf[0]].rect.w >= df.min_panel_px - 0.01)
    _ = df.interact_splitters(299.0, 100.0, True, False, small)
    assert_true(df.nodes[sf[1]].rect.w >= df.min_panel_px - 0.01)
    # A ratio that is fine in a big window still yields >= 32 px in a small one.
    df.nodes[vf].ratio = 0.08
    df.solve_layout(RectF(0.0, 0.0, 200.0, 100.0))
    assert_true(df.nodes[sf[0]].rect.w >= Float32(32.0) - 0.01)
    assert_true(abs(df.nodes[vf].ratio - 0.08) < 1.0e-5)
    print("      PASS")


# ---------------------------------------------------------------------------
# 7 + 14. Scroll views: content past the explicit height is reachable; innermost takes the wheel
# ---------------------------------------------------------------------------


def test_scroll_views() raises:
    print("  [7/14] Scroll views: max(explicit, auto) content height; nested wheel...")
    var ui = UiContext()
    var inp = InputState()
    var n = 40
    for _ in range(2):
        inp.begin_frame()
        _frame(ui, inp)
        var row_h = ui.theme.row_height
        ui.begin_scroll_view("console_scroll", 600.0, 280.0, row_h * Float32(n))
        for i in range(n):
            ui.label("line " + String(i))
        ui.end_scroll_view()
        ui.end_frame()
    var sid = ui.widget_id("console_scroll")
    var max_y = ui.state_map[sid].scroll_max_y
    var rows_h = Float32(n) * (ui.theme.row_height + Float32(2.0)) - Float32(2.0)
    assert_true(max_y >= rows_h - (Float32(280.0) - Float32(8.0)) - Float32(1.0))
    assert_true(max_y > ui.theme.row_height * Float32(n) - Float32(272.0))

    # Nested: only the innermost hovered view scrolls.
    var ui2 = UiContext()
    for f in range(3):
        inp.begin_frame()
        inp.move_mouse(60.0, 60.0)
        _frame(ui2, inp)
        if f == 2:
            ui2.mouse_wheel = -1.0
        ui2.begin_scroll_view("outer", 400.0, 300.0)
        ui2.begin_scroll_view("inner", 300.0, 200.0)
        for i in range(30):
            ui2.label("in " + String(i))
        ui2.end_scroll_view()
        for i in range(30):
            ui2.label("out " + String(i))
        ui2.end_scroll_view()
        ui2.end_frame()
    var outer_id = ui2.widget_id("outer")
    var inner_id = hash_ui_id(outer_id, "inner")
    assert_equal(ui2.state_map[outer_id].scroll_y, Float32(0.0))
    assert_true(ui2.state_map[inner_id].scroll_y > Float32(0.0))

    # The wheel lands in the same frame's layout (rects cached this frame are clickable next
    # frame), and also when it arrives on the pointer's first hovered frame.
    var ui3 = UiContext()
    var first_y = List[Float32]()
    for f in range(4):
        inp.begin_frame()
        # Frame 0 lays the views out with the pointer away; frame 1 brings it in with the wheel.
        if f > 0:
            inp.move_mouse(60.0, 60.0)
        else:
            inp.move_mouse(700.0, 500.0)
        _frame(ui3, inp)
        if f == 1 or f == 2:
            ui3.mouse_wheel = -1.0
        ui3.begin_scroll_view("outer", 400.0, 300.0)
        ui3.begin_scroll_view("inner", 300.0, 200.0)
        for i in range(30):
            ui3.label("in " + String(i))
            if i == 0:
                first_y.append(ui3.last_item_rect.y)
        ui3.end_scroll_view()
        ui3.end_scroll_view()
        ui3.end_frame()
    var o3 = ui3.widget_id("outer")
    var i3 = hash_ui_id(o3, "inner")
    assert_equal(ui3.state_map[o3].scroll_y, Float32(0.0))
    assert_equal(ui3.state_map[i3].scroll_y, Float32(72.0))
    # Each wheel frame moves its own rows (cold pointer on frame 1, warm on frame 2); frame 3
    # (no wheel) matches frame 2.
    assert_true(abs(first_y[1] - (first_y[0] - Float32(36.0))) < Float32(0.5))
    assert_true(abs(first_y[2] - (first_y[1] - Float32(36.0))) < Float32(0.5))
    assert_true(abs(first_y[3] - first_y[2]) < Float32(0.5))
    print("      PASS")


# ---------------------------------------------------------------------------
# 9. A click that closes a popup does not reach the widget under it
# ---------------------------------------------------------------------------


def test_popup_swallows_closing_click() raises:
    print("  [9] Closing a popup swallows its click...")
    var ui = UiContext()
    var inp = InputState()
    var items = List[String]()
    items.append("Alpha")
    items.append("Beta")
    var sel = 0
    var checked = False
    inp.begin_frame()
    inp.move_mouse(900.0, 650.0)
    _frame(ui, inp)
    _ = ui.checkbox("chk_far", "Far", checked)
    var chk_r = ui.last_item_rect
    _ = ui.combo_box("cmb", "Mode", sel, items)
    var box = ui.last_item_rect
    ui.end_frame()

    inp.begin_frame()
    inp.move_mouse(box.x + 20.0, box.center_y())
    inp.set_mouse(0, True)
    _frame(ui, inp)
    _ = ui.checkbox("chk_far", "Far", checked)
    _ = ui.combo_box("cmb", "Mode", sel, items)
    ui.end_frame()
    assert_true(ui._open_popup_id != 0)
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    _ = ui.checkbox("chk_far", "Far", checked)
    _ = ui.combo_box("cmb", "Mode", sel, items)
    ui.end_frame()
    assert_true(ui._open_popup_id != 0)

    # Press on the checkbox (drawn before the combo, outside the popup): the popup closes and the
    # checkbox stays put.
    var cx = chk_r.x + 8.0
    var cy = chk_r.center_y()
    assert_false(ui._popup_rect.contains(cx, cy))
    inp.begin_frame()
    inp.move_mouse(cx, cy)
    inp.set_mouse(0, True)
    _frame(ui, inp)
    assert_true(ui.wants_mouse())
    _ = ui.checkbox("chk_far", "Far", checked)
    _ = ui.combo_box("cmb", "Mode", sel, items)
    ui.end_frame()
    assert_equal(ui._open_popup_id, UInt64(0))
    assert_false(checked)
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    _ = ui.checkbox("chk_far", "Far", checked)
    _ = ui.combo_box("cmb", "Mode", sel, items)
    ui.end_frame()
    # With no popup open, the same press toggles it.
    inp.begin_frame()
    inp.set_mouse(0, True)
    _frame(ui, inp)
    _ = ui.checkbox("chk_far", "Far", checked)
    _ = ui.combo_box("cmb", "Mode", sel, items)
    ui.end_frame()
    assert_true(checked)
    inp.begin_frame()
    inp.set_mouse(0, False)
    _frame(ui, inp)
    ui.end_frame()
    print("      PASS")


# ---------------------------------------------------------------------------
# 10. Text hit-testing uses the same shaped widths as the caret
# ---------------------------------------------------------------------------


def test_text_hit_matches_caret() raises:
    print("  [10] text_input click lands where the caret is drawn...")
    if not _have_ttf():
        print("      (skipped: DejaVuSans.ttf not installed)")
        return
    var ui = UiContext()
    assert_true(ui.font.load_ttf(TTF_PATH, bake_px=32, spread=4))
    var inp = InputState()
    var text = String("AVATAR office WAVY")
    var codes = decode_utf8(text)
    var fs = ui.theme.font_size
    var r: RectF
    inp.begin_frame()
    inp.move_mouse(900.0, 650.0)
    _frame(ui, inp)
    _ = ui.text_input("ti", text, 600.0)
    r = ui.last_item_rect
    ui.end_frame()
    var pad_x = r.x + Float32(8.0)
    for k in range(1, len(codes)):
        # Just right of the caret position for prefix k (as drawn: `_measure_text(prefix)`).
        var caret = ui._measure_text(encode_utf8(codes, 0, k), fs)
        var nxt = ui._measure_text(encode_utf8(codes, 0, k + 1), fs)
        var x = pad_x + caret + (nxt - caret) * Float32(0.25)
        inp.begin_frame()
        inp.move_mouse(x, r.center_y())
        inp.set_mouse(0, True)
        _frame(ui, inp)
        _ = ui.text_input("ti", text, 600.0)
        ui.end_frame()
        inp.begin_frame()
        inp.set_mouse(0, False)
        _frame(ui, inp)
        _ = ui.text_input("ti", text, 600.0)
        ui.end_frame()
        var cur = ui.text_cursor_codepoint("ti")
        if cur != k:
            print("      FAIL at", k, "got", cur)
        assert_equal(cur, k)
        # Wait out the double-click window so the next press is a single click.
        ui.ui_time_sec += Float32(1.0)
    print("      PASS")


# ---------------------------------------------------------------------------
# 11. Clip rects cover fractional edges (floor min, ceil max)
# ---------------------------------------------------------------------------


def test_clip_rounding() raises:
    print("  [11] Clip rects ceil their max edge...")
    var c = clip_rect_i(10.2, 10.7, 100.4, 50.1)
    assert_equal(c[0], 10)
    assert_equal(c[1], 10)
    assert_equal(c[2], 101)
    assert_equal(c[3], 51)
    var e = clip_rect_i(10.0, 20.0, 30.0, 40.0)
    assert_equal(e[2], 30)
    assert_equal(e[3], 40)
    var dl = UiDrawList()
    dl.push_clip_f(0.5, 0.5, 99.5, 49.25)
    var cur = dl.current_clip()
    assert_equal(Int(cur[2]), 100)
    assert_equal(Int(cur[3]), 50)
    print("      PASS")


# ---------------------------------------------------------------------------
# 13. property_row_label IDs come from the caller's id; 14. focusable composites; tree GC
# ---------------------------------------------------------------------------


def test_row_ids_focus_and_tree_gc() raises:
    print("  [13/14] Row IDs from caller id; drag_float3/color_edit3 focusable; tree-row GC...")
    var ui = UiContext()
    var inp = InputState()
    var ra = RectF()
    var rb = RectF()
    for _ in range(3):
        inp.begin_frame()
        _frame(ui, inp)
        ra = ui.property_row_label("Position", 300.0, id="xf_a")[0]
        rb = ui.property_row_label("Position", 300.0, id="xf_b")[0]
        ui.end_frame()
    assert_true(rb.y >= ra.bottom() - 0.01)
    assert_true(ui.widget_id("prow:xf_a") != ui.widget_id("prow:xf_b"))

    # Tab walks into the drag_float3 / color_edit3 components.
    var v = Vec3(1.0, 2.0, 3.0)
    var col = Vec3(0.2, 0.4, 0.6)
    inp.begin_frame()
    _frame(ui, inp)
    _ = ui.drag_float3("pos", "Pos", v)
    _ = ui.color_edit3("tint", "Tint", col)
    var ring = ui.bg_focusables.copy()
    ui.end_frame()
    var want = [ui.widget_id("pos#0"), ui.widget_id("pos#1"), ui.widget_id("pos#2"), ui.widget_id("tint#c0"), ui.widget_id("tint#c2")]
    for wi in range(len(want)):
        var found = False
        for ri in range(len(ring)):
            if ring[ri] == want[wi]:
                found = True
        assert_true(found)
    # Keyboard nudge on a focused component.
    ui.focused_id = ui.widget_id("pos#1")
    inp.begin_frame()
    inp.set_logical_key(262, True)  # Right
    _frame(ui, inp)
    var ch = ui.drag_float3("pos", "Pos", v)
    _ = ui.color_edit3("tint", "Tint", col)
    ui.end_frame()
    inp.set_logical_key(262, False)
    assert_true(ch)
    assert_true(abs(v.y - Float32(2.05)) < Float32(1.0e-4))

    # open_tree_rows entries of rows that stopped being drawn are collected.
    var ut = UiContext()
    ut.gc_stale_frames = 4
    inp.begin_frame()
    inp.move_mouse(20.0, 12.0)
    inp.set_mouse(0, True)
    _frame(ut, inp)
    _ = ut.tree_node_indexed(7, "Row", 0)
    ut.end_frame()
    assert_true(7 in ut.open_tree_rows)
    inp.begin_frame()
    inp.set_mouse(0, False)
    for _ in range(12):
        inp.begin_frame()
        _frame(ut, inp)
        ut.end_frame()
    assert_false(7 in ut.open_tree_rows)
    ut.open_tree_rows[3] = True
    ut.clear_tree_rows()
    assert_equal(len(ut.open_tree_rows), 0)

    # The combo popup keeps the current item list (copied only when it changes).
    var items = List[String]()
    items.append("One")
    items.append("Two")
    var sel = 0
    var uc = UiContext()
    inp.begin_frame()
    inp.move_mouse(900.0, 650.0)
    _frame(uc, inp)
    _ = uc.combo_box("c", "C", sel, items)
    var box = uc.last_item_rect
    uc.end_frame()
    inp.begin_frame()
    inp.move_mouse(box.x + 10.0, box.center_y())
    inp.set_mouse(0, True)
    _frame(uc, inp)
    _ = uc.combo_box("c", "C", sel, items)
    uc.end_frame()
    inp.begin_frame()
    inp.set_mouse(0, False)
    items.append("Three")
    _frame(uc, inp)
    _ = uc.combo_box("c", "C", sel, items)
    uc.end_frame()
    assert_equal(len(uc._popup_combo_items), 3)
    assert_equal(uc._popup_combo_items[2], "Three")
    print("      PASS")


def main() raises:
    print("============================================================")
    print(" FLINT v5 UI FRAMEWORK REGRESSIONS")
    print("============================================================")
    test_font_shaper_cache()
    test_theme_sync_and_contrast()
    test_drag_threshold()
    test_outliner_hit_and_window()
    test_dock_collapse_and_floor()
    test_scroll_views()
    test_popup_swallows_closing_click()
    test_text_hit_matches_caret()
    test_clip_rounding()
    test_row_ids_focus_and_tree_gc()
    print("[PASS] All UI framework regression suites PASSED!")
