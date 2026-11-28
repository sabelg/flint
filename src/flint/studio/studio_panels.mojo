"""Redesigned Flint Studio Panels (`flint.studio.studio_panels`).

Pure Mojo 1.2 modular Studio UI panels with ZERO circular dependency on the main editor state:
  - `draw_outliner_panel_v2`: Slim search bar, `+` quick-spawn, contextual `[Unparent]`
    preserving legacy `(panel.x + 28, panel.y + 16)` hit compatibility, `24 px` rows with
    animated chevrons, semantic entity icons, UTF-8 ellipsis truncation, `2 px` cyan
    selection bar + tinted fill, right-aligned visibility/lock icons, and live pulsing
    `AgentTelemetryBus` activity badges (`PRIM_ARC`).
  - `draw_inspector_panel_v2`: Polished Entity Header Card, collapsible `L2_card`
    component cards (`Transform` with color-coded `drag_float3_v2`, `Material & Shading`
    with `slider_float_v2` & swatches, `Physics & Collider`, conditional `Light`/`Camera`/
    `Audio`/`Nav` cards), `+ Add Component` picker, and an Empty Selection state card.
  - `draw_asset_browser_panel_v2`: Breadcrumb path, type filter pills, left folder tree,
    quick-spawn strip, and responsive `L2_card` visual asset tiles with colored top accent
    strips, semantic `icon_for_path` glyphs, ellipsis truncation, extension badges, drag
    sources, and double-click `.mojo` -> `CodeDrawer` routing.
"""
from std.collections import List
from std.math import abs, isfinite, max, min
from flint.core.math import Vec3
from flint.studio.agent_bus import AgentTelemetryBus
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.ui.context import (
    UI_ALIGN_CENTER,
    UI_ALIGN_LEFT,
    UI_ALIGN_RIGHT,
    UI_FONT_BOLD,
    UI_FONT_MONO,
    UI_FONT_REGULAR,
    UiContext,
    format_float2,
    hash_ui_id,
    pack_rgba8,
    utf8_drop_last_char,
)
from flint.ui.draw_list import (
    UI_FLAG_BORDER,
    UI_FLAG_DASHED_FLOW,
    UI_FLAG_TOP_BEVEL,
)
from flint.ui.flexbox_dock import FLEX_COL, FlexStyle, RectF
from flint.ui.icons import draw_studio_chevron, draw_studio_icon
from flint.ui.theme import (
    ASSET_CARD_H,
    ASSET_CARD_W,
    ICON_AUDIO,
    ICON_CAMERA,
    ICON_CLOSE,
    ICON_COLLIDER,
    ICON_CUBE,
    ICON_EMPTY,
    ICON_EYE,
    ICON_EYE_OFF,
    ICON_FOLDER,
    ICON_FOLIAGE,
    ICON_GRID,
    ICON_LIGHT_POINT,
    ICON_LIGHT_SPOT,
    ICON_LOCK,
    ICON_MATERIAL,
    ICON_MOVE,
    ICON_PLUS,
    ICON_PREFAB,
    ICON_REFRESH,
    ICON_REVERT,
    ICON_SEARCH,
    ICON_SPARK,
    OUTLINER_ROW_H,
    RADIUS_MD,
    RADIUS_PILL,
    RADIUS_SM,
    RADIUS_XS,
    StudioTheme,
    icon_for_path,
    lerp_rgba8,
    rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)


# ============================================================================
# 1. Semantic Entity & Asset Filter Constants
# ============================================================================

comptime ENTITY_KIND_EMPTY: Int = 0
comptime ENTITY_KIND_MESH: Int = 1
comptime ENTITY_KIND_LIGHT_POINT: Int = 2
comptime ENTITY_KIND_LIGHT_SPOT: Int = 3
comptime ENTITY_KIND_CAMERA: Int = 4
comptime ENTITY_KIND_AUDIO: Int = 5
comptime ENTITY_KIND_PREFAB: Int = 6
comptime ENTITY_KIND_FOLIAGE: Int = 7
comptime ENTITY_KIND_COLLIDER: Int = 8

comptime ASSET_FILTER_ALL: Int = 0
comptime ASSET_FILTER_SCENES: Int = 1
comptime ASSET_FILTER_MESHES: Int = 2
comptime ASSET_FILTER_MATERIALS: Int = 3
comptime ASSET_FILTER_AUDIO: Int = 4
comptime ASSET_FILTER_SCRIPTS: Int = 5
comptime ASSET_FILTER_PREFABS: Int = 6


def entity_kind_icon(kind: Int) -> Int:
    """Maps an `ENTITY_KIND_*` constant to its canonical `ICON_*` glyph."""
    if kind == ENTITY_KIND_MESH:
        return ICON_CUBE
    if kind == ENTITY_KIND_LIGHT_POINT:
        return ICON_LIGHT_POINT
    if kind == ENTITY_KIND_LIGHT_SPOT:
        return ICON_LIGHT_SPOT
    if kind == ENTITY_KIND_CAMERA:
        return ICON_CAMERA
    if kind == ENTITY_KIND_AUDIO:
        return ICON_AUDIO
    if kind == ENTITY_KIND_PREFAB:
        return ICON_PREFAB
    if kind == ENTITY_KIND_FOLIAGE:
        return ICON_FOLIAGE
    if kind == ENTITY_KIND_COLLIDER:
        return ICON_COLLIDER
    return ICON_EMPTY


def entity_kind_color(kind: Int, theme: StudioTheme) -> UInt32:
    """Returns a subtle semantic accent tint for an entity icon."""
    if kind == ENTITY_KIND_MESH:
        return theme.accent_primary
    if kind == ENTITY_KIND_LIGHT_POINT or kind == ENTITY_KIND_LIGHT_SPOT:
        return theme.status_warn
    if kind == ENTITY_KIND_CAMERA:
        return theme.agent_flow
    if kind == ENTITY_KIND_AUDIO:
        return theme.play_mode
    if kind == ENTITY_KIND_PREFAB:
        return theme.agent_active
    if kind == ENTITY_KIND_FOLIAGE:
        return theme.status_ok
    if kind == ENTITY_KIND_COLLIDER:
        return theme.axis_y
    return theme.text_secondary


def _ascii_lower_panel(s: String) -> String:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        if b >= 65 and b <= 90:
            out.append(b + 32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def ascii_contains_ci(hay: String, needle_lower: String) -> Bool:
    """ASCII case-insensitive substring test against an already-lowercased needle, without
    allocating a lowered copy of `hay` (the Hierarchy filter runs it on every row)."""
    var n = needle_lower.byte_length()
    if n == 0:
        return True
    var h = hay.byte_length()
    if n > h:
        return False
    var hb = hay.as_bytes()
    var nb = needle_lower.as_bytes()
    for start in range(h - n + 1):
        var ok = True
        for j in range(n):
            var c = hb[start + j]
            if c >= 65 and c <= 90:
                c += 32
            if c != nb[j]:
                ok = False
                break
        if ok:
            return True
    return False


comptime OUTLINER_TOP_H: Float32 = 36.0
"""Height of the outliner's toolbar/search strip above the scrollable list."""
comptime OUTLINER_ROW_GAP: Float32 = 1.0
comptime OUTLINER_ROW_STRIDE: Float32 = OUTLINER_ROW_H + OUTLINER_ROW_GAP
"""Vertical pitch of outliner rows (`OUTLINER_ROW_H` from `flint.ui.theme` plus a 1 px gap)."""
comptime OUTLINER_ROOT_EXTRA_GAP: Float32 = 4.0
"""Extra space between the World root header band and the first entity row."""
comptime _OUTLINER_LIST_TOP_PAD: Float32 = 6.0


def outliner_list_rect(panel_rect: RectF) -> RectF:
    """The scrollable row area of `draw_outliner_panel_v2` (below the search/header strip)."""
    var list_y = panel_rect.y + OUTLINER_TOP_H
    return RectF(panel_rect.x, list_y + Float32(1.0), panel_rect.w, max(Float32(16.0), panel_rect.bottom() - list_y - Float32(2.0)))


def outliner_content_h(n_rows: Int) -> Float32:
    """Scrollable content height of an outliner showing the World root plus `n_rows` entity rows."""
    return Float32(1 + max(0, n_rows)) * OUTLINER_ROW_STRIDE + OUTLINER_ROOT_EXTRA_GAP + Float32(10.0)


def outliner_row_window(panel_rect: RectF, n_rows: Int, state: OutlinerPanelState, wheel: Float32 = 0.0) -> Tuple[Int, Int]:
    """`[first, last)` of the entity rows `draw_outliner_panel_v2` can draw this frame (with a margin
    for this frame's wheel scroll), so callers build row views only for those."""
    if n_rows <= 0:
        return (0, 0)
    var lr = outliner_list_rect(panel_rect)
    var content_h = outliner_content_h(n_rows)
    var max_scroll = max(Float32(0.0), content_h - lr.h)
    var sy = state.scroll_y if isfinite(state.scroll_y) else Float32(0.0)
    sy = max(Float32(0.0), min(max_scroll, sy))
    var slack = Int(abs(wheel) * Float32(36.0) / OUTLINER_ROW_STRIDE) + 4 if isfinite(wheel) else 4
    # Entity row vi sits at list.y + pad - scroll + stride + root_gap + vi * stride.
    var row0 = _OUTLINER_LIST_TOP_PAD + OUTLINER_ROW_STRIDE + OUTLINER_ROOT_EXTRA_GAP
    var first = Int((sy - row0 - OUTLINER_ROW_H) / OUTLINER_ROW_STRIDE) - 1 - slack
    var last = Int((sy + lr.h - row0) / OUTLINER_ROW_STRIDE) + 1 + slack
    first = max(0, min(n_rows, first))
    last = max(first, min(n_rows, last))
    return (first, last)


def _ascii_upper_panel(s: String) -> String:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        if b >= 97 and b <= 122:
            out.append(b - 32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def classify_asset_filter(rel_path: String) -> Int:
    """Classifies a workspace asset path into `ASSET_FILTER_*`."""
    var p = _ascii_lower_panel(String(rel_path.strip()))
    if p.endswith(".flscn") or p.endswith(".flscnb"):
        return ASSET_FILTER_SCENES
    if p.endswith(".glb") or p.endswith(".gltf") or p.endswith(".obj") or p.endswith(".flmesh"):
        return ASSET_FILTER_MESHES
    if p.endswith(".flmat") or p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".ktx2"):
        return ASSET_FILTER_MATERIALS
    if p.endswith(".wav") or p.endswith(".qoa") or p.endswith(".ogg") or p.endswith(".flac") or p.endswith(".flmix"):
        return ASSET_FILTER_AUDIO
    if p.endswith(".mojo") or p.endswith(".py") or p.endswith(".sh"):
        return ASSET_FILTER_SCRIPTS
    if p.endswith(".flprefab"):
        return ASSET_FILTER_PREFABS
    return ASSET_FILTER_ALL


def asset_filter_accent(filter_kind: Int, theme: StudioTheme) -> UInt32:
    if filter_kind == ASSET_FILTER_SCENES:
        return theme.status_ok
    if filter_kind == ASSET_FILTER_MESHES:
        return theme.accent_primary
    if filter_kind == ASSET_FILTER_MATERIALS:
        return theme.agent_active
    if filter_kind == ASSET_FILTER_AUDIO:
        return theme.status_warn
    if filter_kind == ASSET_FILTER_SCRIPTS:
        return theme.agent_flow
    if filter_kind == ASSET_FILTER_PREFABS:
        return theme.play_mode
    return theme.text_muted


def _path_basename(path: String) -> String:
    var slash = path.rfind("/")
    if slash >= 0 and slash + 1 < path.byte_length():
        return String(path[byte=slash + 1 :])
    return path


def _path_folder(path: String) -> String:
    var slash = path.rfind("/")
    if slash > 0:
        return String(path[byte=0:slash])
    return String("")


def _path_ext_badge(path: String) -> String:
    var base = _path_basename(path)
    var dot = base.rfind(".")
    if dot >= 0 and dot + 1 < base.byte_length():
        return _ascii_upper_panel(String(base[byte=dot + 1 :]))
    return String("FILE")


def _make_asset_tile(rel_path: String) -> AssetTileItem:
    var kind = classify_asset_filter(rel_path)
    return AssetTileItem(
        rel_path,
        _path_basename(rel_path),
        _path_folder(rel_path),
        _path_ext_badge(rel_path),
        kind,
        icon_for_path(rel_path, False),
        UInt32(0),
    )


def build_asset_tiles(rel_paths: List[String]) -> List[AssetTileItem]:
    """Converts raw relative workspace asset paths into structured `AssetTileItem`s."""
    var out = List[AssetTileItem](capacity=len(rel_paths))
    for i in range(len(rel_paths)):
        out.append(_make_asset_tile(rel_paths[i]))
    return out^


# ============================================================================
# 2. View-Model, Panel State & Action Structs
# ============================================================================


@fieldwise_init
struct OutlinerRowView(Copyable, Movable):
    var entity_id: Int
    var parent_id: Int
    var name: String
    var depth: Int
    var kind: Int
    var has_children: Bool
    var is_expanded: Bool
    var is_selected: Bool
    var is_primary: Bool
    var visible: Bool
    var locked: Bool
    var child_count: Int

    def __init__(out self, entity_id: Int = -1, name: String = "Entity"):
        self.entity_id = entity_id
        self.parent_id = -1
        self.name = name
        self.depth = 0
        self.kind = ENTITY_KIND_MESH
        self.has_children = False
        self.is_expanded = True
        self.is_selected = False
        self.is_primary = False
        self.visible = True
        self.locked = False
        self.child_count = 0


struct OutlinerPanelState(Copyable, Movable):
    var search_query: String
    var scroll_y: Float32
    var world_root_open: Bool
    var keyboard_entity_id: Int
    var hovered_entity_id: Int
    var last_click_entity_id: Int
    var last_click_time: Float32

    def __init__(out self):
        self.search_query = String("")
        self.scroll_y = Float32(0.0)
        self.world_root_open = True
        self.keyboard_entity_id = -1
        self.hovered_entity_id = -1
        self.last_click_entity_id = -1
        self.last_click_time = Float32(-10.0)


struct OutlinerAction(Copyable, Movable):
    var select_entity_id: Int
    var select_additive: Bool
    var select_range: Bool
    var toggle_expand_entity_id: Int
    var toggle_visible_entity_id: Int
    var toggle_lock_entity_id: Int
    var unparent_selected: Bool
    var parent_to_primary: Bool
    var quick_spawn_clicked: Bool
    var reparent_child_id: Int
    var reparent_target_id: Int
    var focus_entity_id: Int

    def __init__(out self):
        self.select_entity_id = -1
        self.select_additive = False
        self.select_range = False
        self.toggle_expand_entity_id = -1
        self.toggle_visible_entity_id = -1
        self.toggle_lock_entity_id = -1
        self.unparent_selected = False
        self.parent_to_primary = False
        self.quick_spawn_clicked = False
        self.reparent_child_id = -1
        self.reparent_target_id = -2
        self.focus_entity_id = -1


struct InspectorEntityView(Copyable, Movable):
    var has_selection: Bool
    var selected_count: Int
    var entity_id: Int
    var parent_id: Int
    var parent_name: String
    var name: String
    var archetype_label: String
    var entity_kind: Int
    var enabled: Bool
    var visible: Bool
    var locked: Bool
    # Transform
    var pos: Vec3
    var rot_deg: Vec3
    var scale: Vec3
    # Material / Render
    var has_material: Bool
    var shading_mode: Int
    var base_color: Vec3
    var emissive_color: Vec3
    var metallic: Float32
    var roughness: Float32
    var opacity: Float32
    var clearcoat: Float32
    var cast_shadows: Bool
    # Physics / Collider
    var has_physics: Bool
    var mass: Float32
    var collider_kind: Int
    var collider_radius: Float32
    var is_trigger: Bool
    # Optional Light
    var has_light: Bool
    var light_is_spot: Bool
    var light_color: Vec3
    var light_intensity: Float32
    var light_range: Float32
    var light_cone_deg: Float32
    # Optional Camera
    var has_camera: Bool
    var camera_fov_deg: Float32
    var camera_is_main: Bool
    # Optional Audio
    var has_audio: Bool
    var audio_gain: Float32
    var audio_loop: Bool
    var audio_spatial: Bool
    # Optional Nav
    var has_nav: Bool
    var nav_speed: Float32
    var nav_radius: Float32

    def __init__(out self):
        self.has_selection = False
        self.selected_count = 0
        self.entity_id = -1
        self.parent_id = -1
        self.parent_name = String("Root")
        self.name = String("")
        self.archetype_label = String("Entity")
        self.entity_kind = ENTITY_KIND_MESH
        self.enabled = True
        self.visible = True
        self.locked = False
        self.pos = Vec3(0.0, 0.0, 0.0)
        self.rot_deg = Vec3(0.0, 0.0, 0.0)
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.has_material = True
        self.shading_mode = 0
        self.base_color = Vec3(0.75, 0.75, 0.78)
        self.emissive_color = Vec3(0.0, 0.0, 0.0)
        self.metallic = Float32(0.1)
        self.roughness = Float32(0.45)
        self.opacity = Float32(1.0)
        self.clearcoat = Float32(0.0)
        self.cast_shadows = True
        self.has_physics = True
        self.mass = Float32(0.0)
        self.collider_kind = 1
        self.collider_radius = Float32(0.5)
        self.is_trigger = False
        self.has_light = False
        self.light_is_spot = False
        self.light_color = Vec3(1.0, 0.95, 0.85)
        self.light_intensity = Float32(10.0)
        self.light_range = Float32(16.0)
        self.light_cone_deg = Float32(35.0)
        self.has_camera = False
        self.camera_fov_deg = Float32(55.0)
        self.camera_is_main = False
        self.has_audio = False
        self.audio_gain = Float32(1.0)
        self.audio_loop = False
        self.audio_spatial = True
        self.has_nav = False
        self.nav_speed = Float32(3.5)
        self.nav_radius = Float32(0.4)


struct InspectorPanelState(Copyable, Movable):
    var scroll_y: Float32
    var add_component_open: Bool

    def __init__(out self):
        self.scroll_y = Float32(0.0)
        self.add_component_open = False


struct InspectorAction(Copyable, Movable):
    var changed_name: Bool
    var new_name: String
    var changed_enabled: Bool
    var new_enabled: Bool
    var changed_visible: Bool
    var new_visible: Bool
    var changed_transform: Bool
    var new_pos: Vec3
    var new_rot_deg: Vec3
    var new_scale: Vec3
    var reset_transform: Bool
    var changed_material: Bool
    var new_base_color: Vec3
    var new_emissive_color: Vec3
    var new_metallic: Float32
    var new_roughness: Float32
    var new_opacity: Float32
    var new_clearcoat: Float32
    var new_cast_shadows: Bool
    var new_shading_mode: Int
    var changed_physics: Bool
    var new_mass: Float32
    var new_collider_kind: Int
    var new_collider_radius: Float32
    var new_is_trigger: Bool
    var changed_light: Bool
    var new_light_color: Vec3
    var new_light_intensity: Float32
    var new_light_range: Float32
    var new_light_cone_deg: Float32
    var changed_camera: Bool
    var new_camera_fov_deg: Float32
    var new_camera_is_main: Bool
    var changed_audio: Bool
    var new_audio_gain: Float32
    var new_audio_loop: Bool
    var new_audio_spatial: Bool
    var changed_nav: Bool
    var new_nav_speed: Float32
    var new_nav_radius: Float32
    var add_component_kind: String
    var spawn_entity_clicked: Bool

    def __init__(out self):
        self.changed_name = False
        self.new_name = String("")
        self.changed_enabled = False
        self.new_enabled = True
        self.changed_visible = False
        self.new_visible = True
        self.changed_transform = False
        self.new_pos = Vec3(0.0, 0.0, 0.0)
        self.new_rot_deg = Vec3(0.0, 0.0, 0.0)
        self.new_scale = Vec3(1.0, 1.0, 1.0)
        self.reset_transform = False
        self.changed_material = False
        self.new_base_color = Vec3(0.75, 0.75, 0.78)
        self.new_emissive_color = Vec3(0.0, 0.0, 0.0)
        self.new_metallic = Float32(0.0)
        self.new_roughness = Float32(0.5)
        self.new_opacity = Float32(1.0)
        self.new_clearcoat = Float32(0.0)
        self.new_cast_shadows = True
        self.new_shading_mode = 0
        self.changed_physics = False
        self.new_mass = Float32(0.0)
        self.new_collider_kind = 0
        self.new_collider_radius = Float32(0.5)
        self.new_is_trigger = False
        self.changed_light = False
        self.new_light_color = Vec3(1.0, 1.0, 1.0)
        self.new_light_intensity = Float32(10.0)
        self.new_light_range = Float32(16.0)
        self.new_light_cone_deg = Float32(35.0)
        self.changed_camera = False
        self.new_camera_fov_deg = Float32(55.0)
        self.new_camera_is_main = False
        self.changed_audio = False
        self.new_audio_gain = Float32(1.0)
        self.new_audio_loop = False
        self.new_audio_spatial = True
        self.changed_nav = False
        self.new_nav_speed = Float32(3.5)
        self.new_nav_radius = Float32(0.4)
        self.add_component_kind = String("")
        self.spawn_entity_clicked = False


@fieldwise_init
struct AssetTileItem(Copyable, Movable):
    var rel_path: String
    var display_name: String
    var folder: String
    var ext: String
    var filter_kind: Int
    var icon_id: Int
    var swatch_rgba8: UInt32


struct AssetBrowserPanelState(Copyable, Movable):
    var active_filter: Int
    var active_folder: String
    var dialog: Int
    var import_folder: String
    var search_query: String
    var selected_path: String
    var scroll_y: Float32
    var last_click_path: String
    var last_click_time: Float32

    def __init__(out self):
        self.active_filter = ASSET_FILTER_ALL
        self.active_folder = String("")
        self.dialog = 0
        self.import_folder = String("")
        self.search_query = String("")
        self.selected_path = String("")
        self.scroll_y = Float32(0.0)
        self.last_click_path = String("")
        self.last_click_time = Float32(-10.0)


struct AssetBrowserAction(Copyable, Movable):
    var selected_path: String
    var activated_path: String
    var open_in_code_drawer_path: String
    var spawn_preset_kind: Int
    var rescan_clicked: Bool
    var import_clicked: Bool

    def __init__(out self):
        self.selected_path = String("")
        self.activated_path = String("")
        self.open_in_code_drawer_path = String("")
        self.spawn_preset_kind = -1
        self.rescan_clicked = False
        self.import_clicked = False


# ============================================================================
# 3. Internal Micro-Helpers (Scoped Direct-Rect Widgets & Swatches)
# ============================================================================


def _click_rect_button(
    mut ui: UiContext,
    wid: UInt64,
    r: RectF,
    label: String,
    theme: StudioTheme,
    icon_id: Int = -1,
    active: Bool = False,
    accent: UInt32 = 0,
    compact_font: Bool = True,
    tint_icon: Bool = False,
) -> Bool:
    """Direct-rect button with pointer, keyboard and gamepad activation and optical typography."""
    ui._register_focusable(wid, allow_auto_focus=False)
    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup(wid)
    var hov = (not ptr_blocked) and r.contains(ui.mouse_x, ui.mouse_y)
    if hov:
        ui.hovered_id = wid
        ui.hovered_rect = ui._clip_hovered_rect(r)
        if ui.mouse_pressed_l:
            ui.active_id = wid
            ui._set_focused(wid)
    var clicked = (not blocked) and ((hov and ui.mouse_released_l and ui.active_id == wid) or (ui.focused_id == wid and (ui.key_enter_pressed or ui.key_space_pressed or ui.pad_confirm)))
    var pressed = (not blocked) and ui.active_id == wid and ui.mouse_down_l
    var sp = ui.animator.drive_widget(wid, hov, pressed, False)
    var h_t = sp[0]
    var p_t = sp[1]

    var acc = accent if accent != 0 else theme.accent_primary
    var base_bg = rgba8_scale_alpha(acc, Float32(0.18)) if active else theme.l3_raised
    var hov_bg = rgba8_scale_alpha(acc, Float32(0.28)) if active else theme.l3_hover
    var bg = lerp_rgba8(lerp_rgba8(base_bg, hov_bg, h_t), theme.l3_press, p_t)
    var brd = rgba8_with_alpha(acc, 170) if active else lerp_rgba8(
        theme.border_control,
        rgba8_with_alpha(acc, 128) if tint_icon else theme.border_strong,
        h_t,
    )

    ui.draw_list.add_card_surface(
        r.x, r.y, r.w, r.h,
        bg, brd,
        RADIUS_SM, 1.0, True, False,
    )

    if ui.show_focus_ring(wid):
        ui.draw_list.add_rect(r.x, r.y, r.w, r.h, 0, RADIUS_SM, 2.0, theme.focus_ring)

    var font_sz = theme.font_caption if compact_font else theme.font_body
    var font_style = UI_FONT_BOLD if active else UI_FONT_REGULAR
    var has_lbl = label.byte_length() > 0
    var ico_fg = acc if (active or tint_icon) else lerp_rgba8(theme.text_secondary, theme.text_primary, max(h_t, Float32(0.55) if has_lbl else Float32(0.0)))
    var txt_fg = theme.text_primary if active else lerp_rgba8(theme.text_secondary, theme.text_primary, max(h_t, Float32(0.65)))
    if icon_id >= 0 and has_lbl:
        var isz = max(Float32(11.0), min(Float32(13.5), r.h - Float32(10.0)))
        var tw = ui._measure_text_styled(label, font_sz, font_style)
        var content_w = isz + Float32(5.0) + tw
        var sx = r.x + max(Float32(6.0), (r.w - content_w) * Float32(0.5))
        draw_studio_icon(ui.draw_list, icon_id, sx, r.center_y() - isz * Float32(0.5), isz, ico_fg, Float32(1.4))
        var tx = sx + isz + Float32(5.0)
        var text_r = RectF(tx, r.y, max(Float32(8.0), r.right() - tx - Float32(5.0)), r.h)
        ui.draw_text_in_rect(text_r, label, font_sz, txt_fg, UI_ALIGN_LEFT, font_style, Float32(0.0), True, False)
    elif icon_id >= 0:
        var isz = max(Float32(12.0), min(Float32(15.0), r.h - Float32(9.0)))
        draw_studio_icon(ui.draw_list, icon_id, r.center_x() - isz * Float32(0.5), r.center_y() - isz * Float32(0.5), isz, ico_fg, Float32(1.4))
    elif has_lbl:
        ui.draw_text_in_rect(r, label, font_sz, txt_fg, UI_ALIGN_CENTER, font_style, Float32(6.0), True, False)
    return clicked


def _draw_slim_scrollbar(
    mut ui: UiContext,
    wid: UInt64,
    area_rect: RectF,
    content_h: Float32,
    mut scroll_y: Float32,
    theme: StudioTheme,
    ptr_blocked: Bool,
):
    """Interactive 4px -> 6px rounded scrollbar with 10px hit band along `area_rect.right()`."""
    var max_scroll = max(Float32(0.0), content_h - area_rect.h)
    if max_scroll <= Float32(0.5):
        return
    var hit_w = Float32(10.0)
    var track_y = area_rect.y + Float32(4.0)
    var track_h = max(Float32(16.0), area_rect.h - Float32(8.0))
    var thumb_h = max(Float32(22.0), track_h * (area_rect.h / max(Float32(1.0), content_h)))
    var travel = max(Float32(1.0), track_h - thumb_h)
    var hit_r = RectF(area_rect.right() - hit_w, track_y, hit_w, track_h)
    var hov = (not ptr_blocked) and hit_r.contains(ui.mouse_x, ui.mouse_y)
    if hov and ui.mouse_pressed_l:
        ui.active_id = wid
    var drag = ui.active_id == wid and ui.mouse_down_l and (not ui._blocked_by_modal())
    if drag and isfinite(ui.mouse_y):
        var rel = max(Float32(0.0), min(Float32(1.0), (ui.mouse_y - track_y - thumb_h * Float32(0.5)) / travel))
        scroll_y = rel * max_scroll
    var sp = ui.animator.drive_widget(wid, hov, drag, False)
    var hot_t = max(sp[0], sp[1])
    var sb_w = Float32(4.0) + Float32(2.0) * hot_t
    var thumb_y = track_y + (scroll_y / max_scroll) * travel
    var thumb_x = area_rect.right() - sb_w - Float32(2.0)
    if hot_t > Float32(0.02):
        ui.draw_list.add_rect(
            thumb_x, track_y, sb_w, track_h,
            rgba8_scale_alpha(theme.l2_card, hot_t * Float32(0.55)),
            sb_w * Float32(0.5),
        )
    var thumb_col = theme.accent_primary if drag else lerp_rgba8(theme.border_strong, theme.text_secondary, hot_t)
    ui.draw_list.add_rect(thumb_x, thumb_y, sb_w, thumb_h, thumb_col, sb_w * Float32(0.5))


def _draw_compact_toggle(
    mut ui: UiContext,
    wid: UInt64,
    r: RectF,
    mut checked: Bool,
    theme: StudioTheme,
) -> Bool:
    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup(wid)
    var hov = (not ptr_blocked) and r.contains(ui.mouse_x, ui.mouse_y)
    var toggled = False
    if hov:
        ui.hovered_id = wid
        ui.hovered_rect = ui._clip_hovered_rect(r)
        if ui.mouse_pressed_l:
            checked = not checked
            toggled = True
    var slide_t = ui.animator.drive_aux(wid, Float32(1.0) if checked else Float32(0.0))
    var track_col = lerp_rgba8(theme.l3_raised, theme.play_mode, slide_t)
    ui.draw_list.add_rect_styled(
        r.x, r.y, r.w, r.h,
        track_col, theme.border_control,
        r.h * Float32(0.5), 1.0, 1.0, UI_FLAG_BORDER,
    )
    var tr = max(Float32(4.0), r.h * Float32(0.5) - Float32(2.5))
    var cx_lo = r.x + Float32(2.5) + tr
    var cx_hi = r.right() - Float32(2.5) - tr
    ui.draw_list.add_circle(cx_lo + (cx_hi - cx_lo) * slide_t, r.center_y(), tr, theme.text_primary)
    return toggled


def _draw_color_swatch_row_v2(
    mut ui: UiContext,
    id: String,
    label_text: String,
    mut color: Vec3,
    theme: StudioTheme,
    row_w: Float32,
) -> Bool:
    """Left-label color swatch + interactive RGB capsule scrubber row."""
    var row_pair = ui.property_row_label(label_text, row_w, id=id)
    var ctrl = row_pair[1]
    var r8 = Int(max(Float32(0.0), min(Float32(1.0), color.x)) * Float32(255.0) + Float32(0.5))
    var g8 = Int(max(Float32(0.0), min(Float32(1.0), color.y)) * Float32(255.0) + Float32(0.5))
    var b8 = Int(max(Float32(0.0), min(Float32(1.0), color.z)) * Float32(255.0) + Float32(0.5))
    var sw_col = pack_rgba8(r8, g8, b8, 255)
    var sw_w = Float32(26.0)
    var sw_r = RectF(ctrl.x, ctrl.y + Float32(2.0), sw_w, max(Float32(14.0), ctrl.h - Float32(4.0)))
    ui.draw_list.add_rect_styled(
        sw_r.x, sw_r.y, sw_r.w, sw_r.h,
        sw_col, theme.border_strong,
        RADIUS_XS, 1.0, 1.0, UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
    )

    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup()
    var ch_start = sw_r.right() + Float32(5.0)
    var ch_w = max(Float32(24.0), (ctrl.right() - ch_start - Float32(6.0)) / Float32(3.0))
    var vals = [color.x, color.y, color.z]
    var tints = [theme.axis_x, theme.axis_y, theme.axis_z]
    var changed = False
    for c in range(3):
        var cid = ui.widget_id(id + "#rgb_" + String(c))
        var cr = RectF(ch_start + Float32(c) * (ch_w + Float32(3.0)), ctrl.y + Float32(2.0), ch_w, max(Float32(14.0), ctrl.h - Float32(4.0)))
        var hov = (not ptr_blocked) and cr.contains(ui.mouse_x, ui.mouse_y)
        if hov:
            ui.hovered_id = cid
            ui.hovered_rect = ui._clip_hovered_rect(cr)
            if ui.mouse_pressed_l:
                ui.active_id = cid
        var drag = (not blocked) and ui.active_id == cid and ui.mouse_down_l
        if drag and (not ui.mouse_pressed_l) and isfinite(ui.mouse_delta_x) and ui.mouse_delta_x != Float32(0.0):
            var step = Float32(0.002) if ui.key_shift else Float32(0.006)
            var nv = max(Float32(0.0), min(Float32(1.0), vals[c] + ui.mouse_delta_x * step))
            if abs(nv - vals[c]) > Float32(1.0e-5):
                vals[c] = nv
                changed = True
        ui.draw_list.add_rect_styled(
            cr.x, cr.y, cr.w, cr.h,
            theme.l0_canvas,
            tints[c] if drag else theme.border_control,
            RADIUS_XS, 1.0, 1.0, UI_FLAG_BORDER,
        )
        ui.draw_list.add_rect(cr.x + Float32(1.0), cr.y + Float32(1.0), Float32(2.5), max(Float32(2.0), cr.h - Float32(2.0)), tints[c], Float32(1.0))
        var txt = ui._truncate_text_ellipsis(format_float2(vals[c]), max(Float32(10.0), cr.w - Float32(7.0)), theme.font_mono)
        ui._draw_text(txt, cr.x + Float32(5.5), cr.center_y() - theme.font_mono * Float32(0.5), theme.font_mono, theme.text_primary)
    if changed:
        color.x = vals[0]
        color.y = vals[1]
        color.z = vals[2]
    return changed


def _draw_segmented_row_v2(
    mut ui: UiContext,
    id: String,
    label_text: String,
    options: List[String],
    mut selected_idx: Int,
    theme: StudioTheme,
    row_w: Float32,
) -> Bool:
    """Left-label compact segmented pill selector for enums (shading mode, collider kind)."""
    var row_pair = ui.property_row_label(label_text, row_w, id=id)
    var ctrl = row_pair[1]
    var n = len(options)
    if n <= 0:
        return False
    ui.draw_list.add_rect_styled(
        ctrl.x, ctrl.y + Float32(1.0), ctrl.w, max(Float32(16.0), ctrl.h - Float32(2.0)),
        theme.l0_canvas, theme.border_subtle,
        RADIUS_SM, 1.0, 1.0, UI_FLAG_BORDER,
    )
    var gap = Float32(2.0)
    var seg_w = max(Float32(18.0), (ctrl.w - Float32(4.0) - gap * Float32(max(0, n - 1))) / Float32(n))
    var changed = False
    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup()
    for i in range(n):
        var sr = RectF(
            ctrl.x + Float32(2.0) + Float32(i) * (seg_w + gap),
            ctrl.y + Float32(3.0),
            seg_w,
            max(Float32(12.0), ctrl.h - Float32(6.0)),
        )
        var sid = ui.widget_id(id + "#seg_" + String(i))
        var is_sel = selected_idx == i
        var hov = (not ptr_blocked) and sr.contains(ui.mouse_x, ui.mouse_y)
        if hov:
            ui.hovered_id = sid
            ui.hovered_rect = ui._clip_hovered_rect(sr)
            if ui.mouse_pressed_l and selected_idx != i:
                selected_idx = i
                changed = True
        if is_sel:
            ui.draw_list.add_rect_styled(
                sr.x, sr.y, sr.w, sr.h,
                theme.l3_raised, theme.border_control,
                RADIUS_XS, 1.0, 1.0, UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
        elif hov:
            ui.draw_list.add_rect(sr.x, sr.y, sr.w, sr.h, theme.l2_card, RADIUS_XS)
        var disp = ui._truncate_text_ellipsis(options[i], max(Float32(8.0), sr.w - Float32(4.0)), theme.font_caption)
        var tw = ui._measure_text(disp, theme.font_caption)
        ui._draw_text(
            disp,
            sr.x + max(Float32(2.0), (sr.w - tw) * Float32(0.5)),
            sr.center_y() - theme.font_caption * Float32(0.5),
            theme.font_caption,
            theme.text_primary if is_sel else theme.text_secondary,
        )
    return changed


def _draw_card_body_bg(mut ui: UiContext, card_x: Float32, body_y: Float32, card_w: Float32, body_h: Float32, theme: StudioTheme):
    """Emits the joined lower `L2_card` container directly beneath a `collapsible_card` header."""
    if body_h <= Float32(1.0):
        return
    ui.draw_list.add_rounded_rect_ex(
        card_x, body_y, card_w, body_h,
        theme.l2_card,
        Float32(0.0), Float32(0.0), RADIUS_MD, RADIUS_MD,
        1.0, theme.border_subtle, 1.0, UI_FLAG_BORDER,
    )


def begin_inspector_card_body(
    mut ui: UiContext,
    card_x: Float32,
    card_w: Float32,
    body_top_y: Float32,
    joined_to_header: Bool = True,
) -> Int:
    """Reserves a dynamic-height `L2_card` background primitive behind subsequent widgets.
    Pair with `end_inspector_card_body(ui, idx, bottom_y)` once child widgets finish."""
    if card_w <= Float32(1.0):
        return -1
    ref theme = ui.studio_theme
    var prim_idx = len(ui.draw_list.prims)
    if joined_to_header:
        ui.draw_list.add_rounded_rect_ex(
            card_x, body_top_y, card_w, Float32(8.0),
            theme.l2_card,
            Float32(0.0), Float32(0.0), RADIUS_MD, RADIUS_MD,
            1.0, theme.border_subtle, 1.0, UI_FLAG_BORDER,
        )
    else:
        ui.draw_list.add_card_surface(
            card_x, body_top_y, card_w, Float32(8.0),
            theme.l2_card, theme.border_subtle,
            RADIUS_MD, 1.0, True, False,
        )
    return prim_idx


def end_inspector_card_body(
    mut ui: UiContext,
    prim_idx: Int,
    body_bottom_y: Float32,
    pad_bottom: Float32 = 6.0,
):
    """Finalizes a `begin_inspector_card_body` primitive to exact dynamic bottom height."""
    if prim_idx < 0 or prim_idx >= len(ui.draw_list.prims):
        return
    var y0 = ui.draw_list.prims[prim_idx].y0
    var target_y1 = body_bottom_y + max(Float32(0.0), pad_bottom)
    if isfinite(target_y1):
        ui.draw_list.prims[prim_idx].y1 = max(y0 + Float32(2.0), target_y1)


def draw_inspector_empty_state(mut ui: UiContext, width: Float32, avail_h: Float32) -> Bool:
    """Renders a polished `L2_card` empty-selection card inside the active Inspector box.
    Returns True if the primary `+ Spawn Entity` CTA was clicked."""
    var theme = ui.studio_theme.copy()
    var card_w = max(Float32(140.0), min(width, Float32(264.0)))
    var card_h = Float32(158.0)
    var alloc_h = max(card_h + Float32(12.0), min(avail_h, card_h + Float32(48.0)))
    var slot = ui._alloc_item_rect(ui.widget_id("insp_empty_card_slot"), width, alloc_h)
    var card_x = slot.x + (slot.w - card_w) * Float32(0.5)
    var card_y = slot.y + min(Float32(36.0), max(Float32(6.0), (slot.h - card_h) * Float32(0.32)))

    ui.draw_list.add_card_surface(
        card_x, card_y, card_w, card_h,
        theme.l2_card, theme.border_subtle,
        RADIUS_MD, 1.0, True, False,
    )
    var badge_sz = Float32(36.0)
    var bx = card_x + (card_w - badge_sz) * Float32(0.5)
    var by = card_y + Float32(18.0)
    ui.draw_list.add_card_surface(
        bx, by, badge_sz, badge_sz,
        theme.accent_soft, rgba8_with_alpha(theme.accent_primary, 96),
        RADIUS_MD, 1.0, True, False,
    )
    draw_studio_icon(ui.draw_list, ICON_CUBE, bx + Float32(9.0), by + Float32(9.0), Float32(18.0), theme.accent_primary, Float32(1.5))

    ui.draw_text_in_rect(
        RectF(card_x + Float32(10.0), by + badge_sz + Float32(8.0), card_w - Float32(20.0), Float32(20.0)),
        "No Entity Selected",
        theme.font_title,
        theme.text_primary,
        UI_ALIGN_CENTER,
        UI_FONT_BOLD,
        Float32(0.0),
        True,
        False,
    )
    ui.draw_text_in_rect(
        RectF(card_x + Float32(10.0), by + badge_sz + Float32(28.0), card_w - Float32(20.0), Float32(16.0)),
        "Select in Viewport or press Ctrl+K",
        theme.font_caption,
        theme.text_secondary,
        UI_ALIGN_CENTER,
        UI_FONT_REGULAR,
        Float32(0.0),
        True,
        False,
    )

    var cta_w = min(Float32(152.0), card_w - Float32(32.0))
    var cta_r = RectF(card_x + (card_w - cta_w) * Float32(0.5), card_y + card_h - Float32(38.0), cta_w, Float32(26.0))
    return _click_rect_button(ui, ui.widget_id("insp_empty_spawn"), cta_r, "Spawn Entity", theme, ICON_PLUS, True, theme.accent_primary, False, True)


def draw_studio_stats_panel(
    mut ui: UiContext,
    panel_rect: RectF,
    theme: StudioTheme,
    live_entities: Int,
    selected_count: Int,
    state_hash: UInt64,
    ui_prims: Int = 0,
    culled_items: Int = 0,
):
    """Renders cohesive `L2_card` telemetry tiles and metric rows for the Outliner `Stats` tab."""
    if panel_rect.w < Float32(60.0) or panel_rect.h < Float32(40.0):
        return
    ui.set_studio_theme(theme.copy())
    ui.draw_list.push_clip_f(panel_rect.x, panel_rect.y, panel_rect.right(), panel_rect.bottom())
    ui.draw_list.add_rect(panel_rect.x, panel_rect.y, panel_rect.w, panel_rect.h, theme.l1_shell)

    var pad = Float32(8.0)
    var cw = max(Float32(44.0), panel_rect.w - pad * Float32(2.0))
    var cx = panel_rect.x + pad
    var cy = panel_rect.y + pad

    # Section title
    ui.draw_text_in_rect(
        RectF(cx, cy, cw, Float32(18.0)),
        "Scene Telemetry",
        theme.font_section,
        theme.text_primary,
        UI_ALIGN_LEFT,
        UI_FONT_BOLD,
        Float32(2.0),
        True,
        False,
    )
    cy += Float32(24.0)

    # 2-column KPI tiles (`Entities` | `Selected`)
    var gap = Float32(6.0)
    var tile_w = max(Float32(40.0), (cw - gap) * Float32(0.5))
    var tile_h = Float32(48.0)
    if cy + tile_h <= panel_rect.bottom():
        var kpi_labels = [String("ENTITIES"), String("SELECTED")]
        var kpi_vals = [String(live_entities), String(selected_count)]
        var kpi_cols = [theme.accent_primary, theme.status_ok if selected_count > 0 else theme.text_secondary]
        for ki in range(2):
            var tx = cx + Float32(ki) * (tile_w + gap)
            ui.draw_list.add_card_surface(tx, cy, tile_w, tile_h, theme.l2_card, theme.border_subtle, RADIUS_MD, 1.0, True, False)
            ui.draw_list.add_accent_bar(tx + Float32(1.0), cy + Float32(10.0), Float32(2.5), tile_h - Float32(20.0), kpi_cols[ki], True)
            ui.draw_text_in_rect(
                RectF(tx + Float32(10.0), cy + Float32(6.0), tile_w - Float32(14.0), Float32(14.0)),
                kpi_labels[ki],
                Float32(10.0),
                theme.text_muted,
                UI_ALIGN_LEFT,
                UI_FONT_BOLD,
                Float32(0.0),
                True,
                False,
            )
            ui.draw_text_in_rect(
                RectF(tx + Float32(10.0), cy + Float32(22.0), tile_w - Float32(14.0), Float32(20.0)),
                kpi_vals[ki],
                theme.font_title,
                theme.text_primary,
                UI_ALIGN_LEFT,
                UI_FONT_MONO,
                Float32(0.0),
                True,
                False,
            )
        cy += tile_h + Float32(8.0)

    # Key-value detail card (`State Hash`, `UI Prims`, `Culled`)
    var row_h = Float32(24.0)
    var n_rows = 3 if (ui_prims > 0 or culled_items > 0) else 1
    var card_h = Float32(n_rows) * row_h + Float32(8.0)
    if cy + Float32(28.0) <= panel_rect.bottom():
        ui.draw_list.add_card_surface(cx, cy, cw, card_h, theme.l2_card, theme.border_subtle, RADIUS_MD, 1.0, True, False)
        var ry = cy + Float32(4.0)
        var keys = List[String](capacity=3)
        var vals = List[String](capacity=3)
        keys.append("State Hash")
        vals.append(String(state_hash))
        if n_rows == 3:
            keys.append("UI Prims")
            vals.append(String(ui_prims))
            keys.append("Culled")
            vals.append(String(culled_items))
        for ri in range(len(keys)):
            if ri > 0:
                ui.draw_list.add_line(cx + Float32(8.0), ry, cx + cw - Float32(8.0), ry, Float32(1.0), theme.border_subtle)
            var lbl_w = min(cw * Float32(0.46), Float32(88.0))
            ui.draw_text_in_rect(
                RectF(cx + Float32(8.0), ry, lbl_w, row_h),
                keys[ri],
                theme.font_caption,
                theme.text_secondary,
                UI_ALIGN_LEFT,
                UI_FONT_REGULAR,
                Float32(0.0),
                True,
                False,
            )
            var vx = cx + Float32(8.0) + lbl_w + Float32(4.0)
            ui.draw_text_in_rect(
                RectF(vx, ry, max(Float32(16.0), cx + cw - Float32(8.0) - vx), row_h),
                vals[ri],
                theme.font_mono,
                theme.text_primary,
                UI_ALIGN_RIGHT,
                UI_FONT_MONO,
                Float32(0.0),
                True,
                True,
            )
            ry += row_h
    ui.draw_list.pop_clip()


def draw_asset_empty_deck(
    mut ui: UiContext,
    rect: RectF,
    theme: StudioTheme,
    has_any_assets: Bool,
) -> Tuple[Bool, Int]:
    """Renders the crafted Asset Browser empty state directly on `l1_shell` (no hollow scroll box):
      - `rect.h < 84`: single-line Compact HiDPI Fallback (zero vertical clipping or phantom scrollbar)
      - `rect.h >= 84` & `rect.w >= 480` & empty project: Two-Zone Studio Deck (Left Hero Drop-Well +
        Right 3x2 Quick Create shelf preserving canonical `card_*` widget IDs).
    Returns `(import_clicked, spawn_preset_kind)` where `spawn_preset_kind == -1` if none clicked."""
    var import_clicked = False
    var spawn_kind = -1
    if rect.w < Float32(40.0) or rect.h < Float32(16.0):
        return (False, -1)

    var icon_id = ICON_SEARCH if has_any_assets else ICON_FOLDER
    var title_txt = "No matching assets" if has_any_assets else "No project assets yet"

    # ---- Compact HiDPI / Short-Pane Fallback (`rect.h < 84px`) ----
    if rect.h < Float32(84.0):
        var bar_h = min(rect.h, Float32(40.0))
        var bar_y = rect.y + (rect.h - bar_h) * Float32(0.5)
        ui.draw_list.add_recessed_well(
            rect.x, bar_y, rect.w, bar_h,
            rgba8_scale_alpha(theme.l0_canvas, Float32(0.75)), theme.border_subtle,
            RADIUS_MD, 1.0, False, 0,
        )
        var badge_sz = min(Float32(22.0), max(Float32(14.0), bar_h - Float32(8.0)))
        var bx = rect.x + Float32(10.0)
        var by = bar_y + (bar_h - badge_sz) * Float32(0.5)
        ui.draw_list.add_rect(bx, by, badge_sz, badge_sz, theme.accent_soft, RADIUS_XS)
        var isz = max(Float32(10.0), badge_sz - Float32(8.0))
        draw_studio_icon(ui.draw_list, icon_id, bx + (badge_sz - isz) * Float32(0.5), by + (badge_sz - isz) * Float32(0.5), isz, theme.accent_primary, Float32(1.4))
        var tx = bx + badge_sz + Float32(8.0)
        var hint_suffix = "  ·  Try a different filter" if has_any_assets else "  ·  Use Create or Import above"
        var full_line = title_txt + hint_suffix if rect.w >= Float32(340.0) else title_txt
        ui.draw_text_in_rect(
            RectF(tx, bar_y, max(Float32(16.0), rect.right() - tx - Float32(10.0)), bar_h),
            full_line,
            theme.font_caption,
            theme.text_primary if rect.w < Float32(340.0) else theme.text_secondary,
            UI_ALIGN_LEFT,
            UI_FONT_REGULAR,
            Float32(0.0),
            True,
            False,
        )
        return (False, -1)

    # ---- Two-Zone Studio Deck (Left Hero Drop-Well + Right 3x2 Quick Create Shelf) ----
    var show_quick_shelf = (not has_any_assets) and rect.w >= Float32(480.0)
    var gap = Float32(10.0)
    var hero_w = (rect.w - gap) * Float32(0.52) if show_quick_shelf else rect.w
    var hero_r = RectF(rect.x, rect.y, hero_w, rect.h)

    # Left Hero Drop-Well
    ui.draw_list.add_recessed_well(
        hero_r.x, hero_r.y, hero_r.w, hero_r.h,
        theme.l0_canvas, theme.border_control,
        RADIUS_MD, 1.0, False, 0,
    )
    ui.draw_list.add_accent_bar(
        hero_r.x + Float32(1.5),
        hero_r.y + Float32(14.0),
        Float32(2.5),
        max(Float32(16.0), hero_r.h - Float32(28.0)),
        rgba8_with_alpha(theme.accent_primary, 150),
        True,
    )

    var show_cta = (not has_any_assets) and hero_r.h >= Float32(98.0) and hero_r.w >= Float32(230.0)
    var block_h = Float32(66.0) if show_cta else Float32(42.0)
    var top_y = hero_r.y + max(Float32(10.0), (hero_r.h - block_h) * Float32(0.5))
    var badge_sz = Float32(36.0)
    var bx = hero_r.x + Float32(16.0)
    ui.draw_list.add_card_surface(
        bx, top_y, badge_sz, badge_sz,
        theme.accent_soft, rgba8_with_alpha(theme.accent_primary, 96),
        RADIUS_MD, 1.0, True, False,
    )
    draw_studio_icon(ui.draw_list, icon_id, bx + Float32(9.0), top_y + Float32(9.0), Float32(18.0), theme.accent_primary, Float32(1.5))

    var tx = bx + badge_sz + Float32(12.0)
    var text_w = max(Float32(24.0), hero_r.right() - tx - Float32(14.0))
    var hero_title = "No matching assets" if has_any_assets else "Project library is empty"
    var hero_sub = (
        "Try a different name, file extension, or clear your filter."
        if has_any_assets
        else "Import .glb, .flmat, .mojo, or .wav files to start."
    )
    ui.draw_text_in_rect(
        RectF(tx, top_y + Float32(1.0), text_w, Float32(18.0)),
        hero_title,
        theme.font_section,
        theme.text_primary,
        UI_ALIGN_LEFT,
        UI_FONT_BOLD,
        Float32(0.0),
        True,
        False,
    )
    ui.draw_text_in_rect(
        RectF(tx, top_y + Float32(20.0), text_w, Float32(16.0)),
        hero_sub,
        theme.font_caption,
        theme.text_secondary,
        UI_ALIGN_LEFT,
        UI_FONT_REGULAR,
        Float32(0.0),
        True,
        False,
    )
    if show_cta:
        var cta_r = RectF(tx, top_y + Float32(42.0), min(Float32(114.0), text_w), Float32(24.0))
        if _click_rect_button(ui, ui.widget_id("ab_empty_import_cta"), cta_r, "Import Asset", theme, ICON_PLUS, True, theme.accent_primary, True, True):
            import_clicked = True

    # Right 3x2 Quick Create Shelf
    if show_quick_shelf:
        var shelf_x = hero_r.right() + gap
        var shelf_w = max(Float32(60.0), rect.right() - shelf_x)
        var shelf_r = RectF(shelf_x, rect.y, shelf_w, rect.h)
        ui.draw_list.add_card_surface(
            shelf_r.x, shelf_r.y, shelf_r.w, shelf_r.h,
            rgba8_scale_alpha(theme.l2_card, Float32(0.55)), theme.border_subtle,
            RADIUS_MD, 1.0, True, False,
        )
        var pad_s = Float32(10.0)
        var hdr_r = RectF(shelf_r.x + pad_s, shelf_r.y + Float32(6.0), shelf_r.w - pad_s * Float32(2.0), Float32(16.0))
        ui.draw_text_in_rect(hdr_r, "QUICK CREATE", Float32(10.0), theme.text_muted, UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(0.0), True, False)
        ui.draw_text_in_rect(hdr_r, "6 primitives", Float32(10.0), theme.text_muted, UI_ALIGN_RIGHT, UI_FONT_MONO, Float32(0.0), True, False)

        var grid_x = shelf_r.x + pad_s
        var grid_y = hdr_r.bottom() + Float32(6.0)
        var grid_w = max(Float32(40.0), shelf_r.w - pad_s * Float32(2.0))
        var grid_h = max(Float32(24.0), shelf_r.bottom() - grid_y - Float32(10.0))
        var cols = 3
        var rows_n = 2
        var cell_gap = Float32(6.0)
        var cell_w = max(Float32(36.0), (grid_w - cell_gap * Float32(cols - 1)) / Float32(cols))
        var cell_h = min(Float32(32.0), max(Float32(22.0), (grid_h - cell_gap * Float32(rows_n - 1)) / Float32(rows_n)))
        var total_grid_h = cell_h * Float32(rows_n) + cell_gap * Float32(rows_n - 1)
        var start_gy = grid_y + max(Float32(0.0), (grid_h - total_grid_h) * Float32(0.5))

        var ids = [
            String("card_empty"),
            String("card_cube"),
            String("card_point_light"),
            String("card_spot_light"),
            String("card_camera"),
            String("card_audio_emitter"),
        ]
        var labels = [String("Empty"), String("Cube"), String("Point"), String("Spot"), String("Camera"), String("Audio")]
        var icons = [ICON_EMPTY, ICON_CUBE, ICON_LIGHT_POINT, ICON_LIGHT_SPOT, ICON_CAMERA, ICON_AUDIO]
        var tints = [
            theme.text_secondary,
            theme.accent_primary,
            theme.status_warn,
            theme.status_warn,
            theme.agent_flow,
            theme.play_mode,
        ]
        for si in range(6):
            var r_i = si // cols
            var c_i = si % cols
            var cr = RectF(
                grid_x + Float32(c_i) * (cell_w + cell_gap),
                start_gy + Float32(r_i) * (cell_h + cell_gap),
                cell_w,
                cell_h,
            )
            var swid = ui.widget_id(ids[si])
            if _click_rect_button(ui, swid, cr, labels[si], theme, icons[si], False, tints[si], True, True):
                spawn_kind = si
            if ui.hovered_id == swid:
                ui.queue_tooltip_rect(swid, cr, "Spawn " + labels[si] + " entity")

    return (import_clicked, spawn_kind)


def draw_asset_card_tile(
    mut ui: UiContext,
    wid: UInt64,
    r: RectF,
    item: AssetTileItem,
    is_selected: Bool,
    theme: StudioTheme,
) -> Tuple[Bool, Bool]:
    """Renders a polished visual `L2_card` asset tile with inset 2px semantic top bar, recessed
    preview well, mono extension pill, and optical centered filename. Returns `(clicked, hovered)`."""
    ui._register_focusable(wid, allow_auto_focus=False)
    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup(wid)
    var hov = (not ptr_blocked) and r.contains(ui.mouse_x, ui.mouse_y)
    if hov:
        ui.hovered_id = wid
        ui.hovered_rect = ui._clip_hovered_rect(r)
        if ui.mouse_pressed_l:
            ui.active_id = wid
            ui._set_focused(wid)
    var clicked = (not blocked) and (hov and ui.mouse_pressed_l)

    var accent = asset_filter_accent(item.filter_kind, theme)
    var sp = ui.animator.drive_widget(wid, hov, is_selected, False)
    var h_t = sp[0]
    var bg = theme.l3_raised if is_selected else lerp_rgba8(theme.l2_card, theme.l3_raised, h_t * Float32(0.75))
    var brd = accent if is_selected else lerp_rgba8(theme.border_subtle, theme.border_strong, h_t)

    ui.draw_list.add_card_surface(r.x, r.y, r.w, r.h, bg, brd, RADIUS_MD, 1.0, True, False)
    if ui.show_focus_ring(wid):
        ui.draw_list.add_rect(r.x, r.y, r.w, r.h, 0, RADIUS_MD, 2.0, theme.focus_ring)

    # Inset 2px top semantic accent pill bar
    ui.draw_list.add_accent_bar(r.x + Float32(10.0), r.y + Float32(2.0), max(Float32(12.0), r.w - Float32(20.0)), Float32(2.0), accent, True)

    # Recessed `l0_canvas` preview well
    var well_h = max(Float32(34.0), r.h - Float32(44.0))
    var well = RectF(r.x + Float32(6.0), r.y + Float32(8.0), max(Float32(24.0), r.w - Float32(12.0)), well_h)
    ui.draw_list.add_recessed_well(well.x, well.y, well.w, well.h, theme.l0_canvas, theme.border_subtle, RADIUS_SM, 1.0, False, 0)
    if (item.swatch_rgba8 >> 24) != 0:
        var sr = min(Float32(14.0), well.h * Float32(0.32))
        ui.draw_list.add_circle(well.center_x(), well.center_y(), sr, item.swatch_rgba8, Float32(1.2), theme.border_strong)
        ui.draw_list.add_circle(well.center_x() - sr * Float32(0.32), well.center_y() - sr * Float32(0.32), max(Float32(2.0), sr * Float32(0.24)), rgba8(255, 255, 255, 110))
    else:
        var isz = min(Float32(26.0), max(Float32(16.0), well.h - Float32(18.0)))
        draw_studio_icon(ui.draw_list, item.icon_id, well.center_x() - isz * Float32(0.5), well.center_y() - isz * Float32(0.5) - Float32(2.0), isz, accent, Float32(1.55))

    # Bottom-right translucent mono extension pill inside preview well
    var ext_sz = Float32(9.5)
    var ext_tw = ui._measure_text_styled(item.ext, ext_sz, UI_FONT_MONO)
    var ext_w = max(Float32(22.0), min(well.w - Float32(6.0), ext_tw + Float32(8.0)))
    var ext_r = RectF(well.right() - ext_w - Float32(3.0), well.bottom() - Float32(16.0), ext_w, Float32(13.0))
    ui.draw_list.add_rect(ext_r.x, ext_r.y, ext_r.w, ext_r.h, rgba8_scale_alpha(theme.l1_shell, Float32(0.88)), RADIUS_XS, Float32(1.0), theme.border_subtle)
    ui.draw_text_in_rect(ext_r, item.ext, ext_sz, accent, UI_ALIGN_CENTER, UI_FONT_MONO, Float32(2.0), True, False)

    # Centered truncated filename footer
    var foot_r = RectF(r.x + Float32(5.0), well.bottom() + Float32(4.0), max(Float32(16.0), r.w - Float32(10.0)), max(Float32(16.0), r.bottom() - well.bottom() - Float32(7.0)))
    ui.draw_text_in_rect(
        foot_r,
        item.display_name,
        theme.font_caption,
        theme.text_primary if (is_selected or hov) else theme.text_secondary,
        UI_ALIGN_CENTER,
        UI_FONT_BOLD if is_selected else UI_FONT_REGULAR,
        Float32(1.0),
        True,
        True,
    )
    if hov:
        ui.queue_tooltip_rect(wid, r, item.rel_path)

    ui.last_item_id = wid
    ui.last_item_rect = r
    return (clicked, hov)


def _push_panel_box(mut ui: UiContext, id: String, content_rect: RectF, gap: Float32 = 4.0, pad: Float32 = 8.0):
    """Pushes a clean local Flex column box over `content_rect` for v2 widgets."""
    var st = FlexStyle.col(gap=gap, pad=pad)
    st.fixed_w = content_rect.w
    st.fixed_h = content_rect.h
    var wid = ui.widget_id(id)
    var f_idx = ui.flex.add_node(-1, st, content_rect.w, content_rect.h)
    ui.dock_flex_root_idxs.append(f_idx)
    ui.dock_flex_root_rects.append(content_rect)
    from flint.ui.context import _BoxScope
    ui.box_stack.append(
        _BoxScope(
            wid,
            f_idx,
            content_rect,
            content_rect.x + pad,
            content_rect.y + pad,
            FLEX_COL,
            gap,
            pad,
            pad,
            content_rect.x + pad,
            content_rect.y + pad,
        )
    )
    ui.id_stack.append(wid)


def _pop_panel_box(mut ui: UiContext):
    if len(ui.box_stack) > 1:
        _ = ui.box_stack.pop()
        _ = ui.id_stack.pop()


# ============================================================================
# 4. Panel 1: Redesigned Outliner (`draw_outliner_panel_v2`)
# ============================================================================


def draw_outliner_panel_v2(
    mut ui: UiContext,
    panel_rect: RectF,
    theme: StudioTheme,
    rows: List[OutlinerRowView],
    world_name: String,
    total_entity_count: Int,
    selected_count: Int,
    selected_has_parent: Bool,
    bus: AgentTelemetryBus,
    mut state: OutlinerPanelState,
    row_base: Int = 0,
    total_rows: Int = -1,
) -> OutlinerAction:
    """Renders the Obsidian & Flint Ember Scene Outliner with symmetric toolbar, inline search
    clear, contextual `[Unparent]` at `(panel.x+28, panel.y+16)` compatibility rect, distinct
    world-root header band with mono count pill, `OUTLINER_ROW_H` rows with continuous tree
    guides + L-ticks, reclaimed leaf padding, bordered selection pill, `+N` collapsed badges,
    visibility/lock hover wells, draggable 4->6px scrollbar, and live Agent Activity Badges.

    Windowed mode (`total_rows >= 0`): `rows` holds only rows `[row_base, row_base + len(rows))`
    of an already-filtered list of `total_rows` (see `outliner_row_window`), so a big scene
    builds row views only for what is on screen. Otherwise `rows` is the whole list and the
    panel applies the search filter itself. Keyboard row traversal then covers the windowed rows
    (the window keeps a few rows of slack around the viewport, and follows the scroll)."""
    var act = OutlinerAction()
    if panel_rect.w < Float32(40.0) or panel_rect.h < Float32(40.0):
        return act^

    ui.set_studio_theme(theme)
    ui.draw_list.push_clip_f(panel_rect.x, panel_rect.y, panel_rect.right(), panel_rect.bottom())

    # Panel L1 backdrop
    ui.draw_list.add_rect(panel_rect.x, panel_rect.y, panel_rect.w, panel_rect.h, theme.l1_shell)

    var pad = Float32(8.0)
    var top_h = Float32(36.0)
    var ctrl_h = Float32(24.0)
    var top_y = panel_rect.y + Float32(6.0)
    var cur_x = panel_rect.x + pad

    # Compatibility contract (`test_flint_v5_assets_studio_mcp.mojo:3379-3391`):
    # When a selected entity has a parent, keep a clickable `[Unparent]` button covering
    # `(panel_rect.x + 28, panel_rect.y + 16)`!
    if selected_has_parent:
        var unp_r = RectF(cur_x, top_y, Float32(68.0), ctrl_h)
        var unp_wid = ui.widget_id("outliner_v2_unparent")
        if _click_rect_button(ui, unp_wid, unp_r, "Unparent", theme, -1, False, theme.accent_primary, True):
            act.unparent_selected = True
        cur_x = unp_r.right() + Float32(6.0)
    elif selected_count >= 2:
        var par_r = RectF(cur_x, top_y, Float32(72.0), ctrl_h)
        var par_wid = ui.widget_id("outliner_v2_parent_pri")
        if _click_rect_button(ui, par_wid, par_r, "Parent", theme, -1, False, theme.accent_primary, True):
            act.parent_to_primary = True
        cur_x = par_r.right() + Float32(6.0)

    var right = panel_rect.right() - pad
    var label_add = right - cur_x >= Float32(140.0)
    var plus_w = Float32(56.0) if label_add else Float32(26.0)
    var plus_r = RectF(right - plus_w, top_y, plus_w, ctrl_h)
    var search_w = max(Float32(0.0), plus_r.x - Float32(6.0) - cur_x)
    var search_r = RectF(cur_x, top_y, search_w, ctrl_h)
    var search_wid = ui.widget_id("outliner_v2_search")
    ui._register_focusable(search_wid, allow_auto_focus=False)
    ui._text_widget_ids.append(search_wid)
    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup(search_wid)
    var has_query_raw = state.search_query.byte_length() > 0
    var clear_r = RectF(search_r.right() - Float32(20.0), search_r.y + Float32(2.0), Float32(18.0), max(Float32(12.0), search_r.h - Float32(4.0)))
    var clear_hov = has_query_raw and search_w >= Float32(56.0) and (not ptr_blocked) and clear_r.contains(ui.mouse_x, ui.mouse_y)
    var search_hov = (not ptr_blocked) and search_r.contains(ui.mouse_x, ui.mouse_y)
    if clear_hov:
        ui.hovered_id = ui.widget_id("outliner_v2_search_clear")
        ui.hovered_rect = ui._clip_hovered_rect(clear_r)
        if ui.mouse_pressed_l:
            state.search_query = String("")
            state.scroll_y = Float32(0.0)
            has_query_raw = False
    elif search_hov:
        ui.hovered_id = search_wid
        ui.hovered_rect = ui._clip_hovered_rect(search_r)
        if ui.mouse_pressed_l:
            ui._set_focused(search_wid)
            ui.active_id = search_wid  # keeps the focus through end_frame (as text_input does)
    var search_foc = (not blocked) and ui.focused_id == search_wid
    if search_foc:
        ui._focused_text_id = search_wid
        ui._focused_text_seen = True
        if ui.key_esc_pressed:
            state.search_query = String("")
            ui.focused_id = 0
            ui._focused_text_id = 0
            has_query_raw = False
        elif ui.key_bs_pressed and state.search_query.byte_length() > 0:
            var next_q = utf8_drop_last_char(state.search_query)
            state.search_query = next_q
            has_query_raw = state.search_query.byte_length() > 0
        elif (ui.text_utf8.byte_length() > 0 or ui.ime_commit.byte_length() > 0) and not ui.key_ctrl:
            state.search_query += ui.text_utf8 + ui.ime_commit
            ui.ime_commit = ""
            has_query_raw = True

    ui.draw_list.push_clip_f(search_r.x, search_r.y, search_r.right(), search_r.bottom())
    ui.draw_list.add_recessed_well(
        search_r.x, search_r.y, search_r.w, search_r.h,
        theme.l0_canvas,
        theme.border_strong if search_hov else theme.border_subtle,
        RADIUS_SM, 1.0, search_foc, theme.focus_ring,
    )
    draw_studio_icon(
        ui.draw_list,
        ICON_SEARCH,
        search_r.x + Float32(7.0),
        search_r.center_y() - Float32(6.0),
        Float32(12.0),
        theme.accent_primary if search_foc else theme.text_muted,
        Float32(1.35),
    )
    var q_x = search_r.x + Float32(23.0)
    var clear_reserve = Float32(22.0) if (has_query_raw and search_w >= Float32(56.0)) else Float32(6.0)
    var q_avail = max(Float32(12.0), search_r.right() - q_x - clear_reserve)
    var q_rect = RectF(q_x, search_r.y, q_avail, search_r.h)
    var caret_x = q_x
    if has_query_raw:
        var q_disp = ui._truncate_text_ellipsis(state.search_query, q_avail, theme.font_caption)
        ui.draw_text_in_rect(q_rect, q_disp, theme.font_caption, theme.text_primary, UI_ALIGN_LEFT, UI_FONT_REGULAR, Float32(0.0), False, False)
        caret_x = min(q_rect.right() - Float32(1.0), q_x + ui._measure_text_styled(q_disp, theme.font_caption, UI_FONT_REGULAR) + Float32(1.0))
        if search_w >= Float32(56.0):
            if clear_hov:
                ui.draw_list.add_rect(clear_r.x, clear_r.y, clear_r.w, clear_r.h, theme.l3_raised, RADIUS_XS)
            draw_studio_icon(
                ui.draw_list,
                ICON_CLOSE,
                clear_r.center_x() - Float32(5.0),
                clear_r.center_y() - Float32(5.0),
                Float32(10.0),
                theme.text_primary if clear_hov else theme.text_muted,
                Float32(1.3),
            )
    else:
        ui.draw_text_in_rect(q_rect, "Find entity", theme.font_caption, theme.text_muted, UI_ALIGN_LEFT, UI_FONT_REGULAR, Float32(0.0), True, False)
    if search_foc:
        ui.draw_list.add_line(caret_x, top_y + Float32(5.0), caret_x, top_y + ctrl_h - Float32(5.0), Float32(1.2), theme.focus_ring)
        # IME candidate window anchor (platform text-input cursor rectangle)
        ui.text_caret_rect = RectF(caret_x, top_y + Float32(2.0), Float32(2.0), max(Float32(1.0), ctrl_h - Float32(4.0)))
    ui.draw_list.pop_clip()

    # `+ Add` quick-spawn button (accent-tinted icon + crisp hover border)
    var plus_wid = ui.widget_id("outliner_v2_plus")
    if _click_rect_button(ui, plus_wid, plus_r, "Add" if label_add else "", theme, ICON_PLUS, False, theme.accent_primary, True, True):
        act.quick_spawn_clicked = True
    if ui.hovered_id == plus_wid:
        ui.queue_tooltip_rect(plus_wid, plus_r, "Quick Spawn Entity", "Ctrl+K")

    # Subtle header divider hairline
    var list_y = panel_rect.y + top_h
    ui.draw_list.add_line(panel_rect.x, list_y, panel_rect.right(), list_y, Float32(1.0), theme.border_subtle)

    # Scrollable entity list region
    var list_rect = outliner_list_rect(panel_rect)
    var row_h = OUTLINER_ROW_H
    var row_gap = OUTLINER_ROW_GAP
    var stride = OUTLINER_ROW_STRIDE
    var root_extra_gap = OUTLINER_ROOT_EXTRA_GAP
    var windowed = total_rows >= 0
    var filter_l = _ascii_lower_panel(String(state.search_query.strip()))
    var has_filter = filter_l.byte_length() > 0 and not windowed

    # Collect filtered row indices (windowed callers have already filtered)
    var vis_indices = List[Int](capacity=len(rows))
    for i in range(len(rows)):
        if (not has_filter) or ascii_contains_ci(rows[i].name, filter_l):
            vis_indices.append(i)
    var n_list = total_rows if windowed else len(vis_indices)
    var first_vi = max(0, row_base) if windowed else 0

    var query_active = filter_l.byte_length() > 0
    var tree_open = state.world_root_open or query_active
    var n_shown = n_list if tree_open else 0
    var content_h = outliner_content_h(n_shown)
    var max_scroll = max(Float32(0.0), content_h - list_rect.h)
    if (not ptr_blocked) and list_rect.contains(ui.mouse_x, ui.mouse_y) and ui.mouse_wheel != Float32(0.0):
        state.scroll_y = max(Float32(0.0), min(max_scroll, state.scroll_y - ui.mouse_wheel * Float32(36.0)))
    var tree_id = ui.widget_id("outliner_v2_world_root")
    ui._register_focusable(tree_id, allow_auto_focus=False)
    var tree_focused = not blocked and ui.focused_id == tree_id
    if search_foc and ui.key_enter_pressed and query_active and len(vis_indices) > 0 and first_vi == 0:
        var first_entity_id = rows[vis_indices[0]].entity_id
        act.select_entity_id = first_entity_id
        state.keyboard_entity_id = first_entity_id
        state.scroll_y = Float32(0.0)
        ui._set_focused(tree_id)
        ui._focused_text_id = 0
    if tree_focused and (ui.key_up_pressed or ui.key_down_pressed or ui.key_home_pressed or ui.key_end_pressed):
        # One Tab stop for the tree; arrow keys traverse all rows, including offscreen rows
        # (windowed mode: every row in the window, which keeps slack rows beyond the viewport).
        var current = -1
        for vi in range(len(vis_indices)):
            if rows[vis_indices[vi]].entity_id == state.keyboard_entity_id:
                current = vi
                break
        var last = len(vis_indices) - 1 if tree_open else -1
        var target = min(last, current + 1) if ui.key_down_pressed else max(-1, current - 1)
        if ui.key_home_pressed or ui.key_end_pressed:
            target = last if ui.key_end_pressed else -1
        state.keyboard_entity_id = rows[vis_indices[target]].entity_id if target >= 0 else -1
        if target >= 0:
            act.select_entity_id = state.keyboard_entity_id
            act.select_range = ui.key_shift
        var g_target = first_vi + target if target >= 0 else -1
        var offset = Float32(g_target + 1) * stride + root_extra_gap + Float32(4.0)
        state.scroll_y = min(state.scroll_y, offset)
        state.scroll_y = max(state.scroll_y, offset + row_h - list_rect.h)
    state.scroll_y = max(Float32(0.0), min(max_scroll, state.scroll_y))
    # Rows scrolled partly under the header strip stay hit-testable only inside the list area,
    # so they never take clicks meant for the search box above.
    var in_list = list_rect.contains(ui.mouse_x, ui.mouse_y)

    ui.draw_list.push_clip_f(list_rect.x, list_rect.y, list_rect.right(), list_rect.bottom())

    var sb_reserve = Float32(9.0) if max_scroll > Float32(0.5) else Float32(0.0)
    var row_x = list_rect.x + Float32(6.0)
    var row_w = max(Float32(60.0), list_rect.w - Float32(12.0) - sb_reserve)
    var cur_y = list_rect.y + Float32(6.0) - state.scroll_y

    # ---- Distinct World Root Header Band ----
    var root_r = RectF(row_x, cur_y, row_w, row_h)
    if root_r.bottom() >= list_rect.y and root_r.y <= list_rect.bottom():
        var root_wid = tree_id
        var root_hov = (not ptr_blocked) and in_list and root_r.contains(ui.mouse_x, ui.mouse_y)
        var root_bg = theme.l2_card if root_hov else rgba8_scale_alpha(theme.l2_card, Float32(0.72))
        var root_brd = theme.border_control if root_hov else theme.border_subtle
        ui.draw_list.add_card_surface(root_r.x, root_r.y, root_r.w, root_r.h, root_bg, root_brd, RADIUS_SM, 1.0, True, False)
        if root_hov:
            ui.hovered_id = root_wid
            ui.hovered_rect = ui._clip_hovered_rect(root_r)
            if ui.mouse_pressed_l:
                ui._set_focused(tree_id)
                state.keyboard_entity_id = -1
                if not query_active:
                    state.world_root_open = not state.world_root_open
        if tree_focused and state.keyboard_entity_id < 0:
            if ui.show_focus_ring(tree_id):
                ui.draw_list.add_rect(root_r.x, root_r.y, root_r.w, root_r.h, 0, RADIUS_SM, 2.0, theme.focus_ring)
            if not query_active:
                if ui.key_enter_pressed or ui.key_space_pressed or ui.pad_confirm:
                    state.world_root_open = not state.world_root_open
                elif ui.key_left_pressed or ui.key_right_pressed:
                    state.world_root_open = ui.key_right_pressed
        var root_drop = ui.accept_drag_target_rect("STUDIO_ENTITY", root_r)
        if root_drop[0]:
            act.reparent_child_id = root_drop[1]
            act.reparent_target_id = -1
        var root_exp_t = ui.animator.drive_expand(root_wid, tree_open)
        draw_studio_chevron(ui.draw_list, root_r.x + Float32(6.0), root_r.center_y() - Float32(5.5), Float32(11.0), theme.text_secondary, root_exp_t, Float32(1.4))
        draw_studio_icon(ui.draw_list, ICON_GRID, root_r.x + Float32(21.0), root_r.center_y() - Float32(6.5), Float32(13.0), theme.accent_primary, Float32(1.35))

        # Mono count pill badge on right edge
        var count = String(n_list) + "/" + String(total_entity_count) if query_active else String(total_entity_count)
        var count_tw = ui._measure_text_styled(count, theme.font_mono, UI_FONT_MONO)
        var pill_w = max(Float32(20.0), count_tw + Float32(10.0))
        var pill_h = Float32(16.0)
        var pill_r = RectF(root_r.right() - pill_w - Float32(5.0), root_r.center_y() - pill_h * Float32(0.5), pill_w, pill_h)
        ui.draw_list.add_rect(pill_r.x, pill_r.y, pill_r.w, pill_r.h, theme.l0_canvas, RADIUS_PILL, Float32(1.0), theme.border_subtle)
        ui.draw_text_in_rect(pill_r, count, theme.font_mono, theme.text_secondary, UI_ALIGN_CENTER, UI_FONT_MONO, Float32(2.0), False, False)

        var lbl_r = RectF(root_r.x + Float32(39.0), root_r.y, max(Float32(12.0), pill_r.x - (root_r.x + Float32(43.0))), root_r.h)
        ui.draw_text_in_rect(lbl_r, world_name, theme.font_section, theme.text_primary, UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(0.0), True, False)
    cur_y += stride + root_extra_gap

    state.hovered_entity_id = -1

    # Entity rows (strict O(visible) culling; zero idle spring dict allocations!)
    if query_active and n_list == 0:
        ui.draw_text_in_rect(
            RectF(row_x + Float32(8.0), cur_y, max(Float32(40.0), row_w - Float32(16.0)), Float32(24.0)),
            "No matching entities",
            theme.font_caption,
            theme.text_muted,
            UI_ALIGN_LEFT,
            UI_FONT_REGULAR,
            Float32(0.0),
            True,
            False,
        )
    if tree_open:
        var indent_step = Float32(13.0)
        var n_vis = len(vis_indices)
        if windowed:
            ui.culled_items_this_frame += max(0, n_list - n_vis)
        for li in range(n_vis):
            var vi = first_vi + li
            ref row = rows[vis_indices[li]]
            var ry = cur_y + Float32(vi) * stride
            if ry + row_h < list_rect.y or ry > list_rect.bottom():
                ui.culled_items_this_frame += 1
                continue

            var rr = RectF(row_x, ry, row_w, row_h)
            var rwid = ui.widget_id("outliner_row_" + String(row.entity_id))
            var hov = (not ptr_blocked) and in_list and rr.contains(ui.mouse_x, ui.mouse_y)
            if hov:
                state.hovered_entity_id = row.entity_id
                ui.hovered_id = rwid
                ui.hovered_rect = ui._clip_hovered_rect(rr)

            # Row selection / hover surface (bordered selection pill + inset 3px pill bar)
            if row.is_selected:
                var sel_brd = rgba8_with_alpha(theme.accent_primary if row.is_primary else theme.border_strong, 84)
                ui.draw_list.add_rect_styled(rr.x, rr.y, rr.w, rr.h, theme.accent_soft, sel_brd, RADIUS_SM, 1.0, 1.0, UI_FLAG_BORDER)
                var bar_col = theme.accent_primary if row.is_primary else theme.text_secondary
                ui.draw_list.add_accent_bar(rr.x + Float32(3.0), rr.y + Float32(4.0), Float32(3.0), rr.h - Float32(8.0), bar_col, True)
            elif hov:
                ui.draw_list.add_rect_styled(rr.x, rr.y, rr.w, rr.h, theme.l2_card, theme.border_subtle, RADIUS_SM, 1.0, 1.0, UI_FLAG_BORDER)
            if tree_focused and state.keyboard_entity_id == row.entity_id:
                if ui.show_focus_ring(tree_id):
                    ui.draw_list.add_rect(rr.x, rr.y, rr.w, rr.h, 0, RADIUS_SM, 2.0, theme.focus_ring)
                if row.has_children and not query_active and ((ui.key_left_pressed and row.is_expanded) or (ui.key_right_pressed and not row.is_expanded)):
                    act.toggle_expand_entity_id = row.entity_id
                elif ui.key_enter_pressed or ui.key_space_pressed or ui.pad_confirm:
                    act.select_entity_id = row.entity_id
                    if ui.key_enter_pressed:
                        act.focus_entity_id = row.entity_id

            # Continuous depth >= 1 tree guide hairlines + crisp horizontal L-tick
            var base_indent_x = rr.x + Float32(8.0)
            var shown_depth = min(row.depth, max(0, Int((row_w - Float32(96.0)) / indent_step)))
            var next_depth = rows[vis_indices[li + 1]].depth if (li + 1 < n_vis) else -1
            for d in range(shown_depth):
                var gx = base_indent_x + Float32(d) * indent_step + Float32(7.0)
                var is_last_at_d = (d == shown_depth - 1) and (next_depth <= d)
                var gy_bot = rr.center_y() if is_last_at_d else rr.bottom()
                ui.draw_list.add_line(gx, rr.y - row_gap, gx, gy_bot, Float32(1.0), theme.border_control if (row.is_selected or hov) else theme.border_subtle)
                if d == shown_depth - 1:
                    ui.draw_list.add_line(gx, rr.center_y(), gx + Float32(6.0), rr.center_y(), Float32(1.0), theme.border_control if (row.is_selected or hov) else theme.border_subtle)

            var ix = base_indent_x + Float32(shown_depth) * indent_step
            var clicked_sub = False
            if row.has_children and not query_active:
                var chev_r = RectF(ix, rr.y, Float32(15.0), rr.h)
                var chev_hov = hov and chev_r.contains(ui.mouse_x, ui.mouse_y)
                if chev_hov and ui.mouse_pressed_l:
                    act.toggle_expand_entity_id = row.entity_id
                    clicked_sub = True
                draw_studio_chevron(
                    ui.draw_list,
                    ix + Float32(2.0),
                    rr.center_y() - Float32(5.5),
                    Float32(11.0),
                    theme.text_primary if chev_hov else theme.text_muted,
                    Float32(1.0) if row.is_expanded else Float32(0.0),
                    Float32(1.4),
                )
                ix += Float32(15.0)
            else:
                # Reclaim 8px of unused chevron slot on leaf rows so deep names fit untruncated!
                ix += Float32(7.0) if shown_depth > 0 else Float32(10.0)

            # Semantic entity icon
            var ico_col = entity_kind_color(row.kind, theme)
            if not row.visible:
                ico_col = rgba8_scale_alpha(ico_col, Float32(0.45))
            draw_studio_icon(ui.draw_list, entity_kind_icon(row.kind), ix, rr.center_y() - Float32(6.5), Float32(13.0), ico_col, Float32(1.35))
            ix += Float32(17.0)

            # Trailing right-aligned action slots: [Collapsed +N Badge] [Agent Badge] [Eye] [Lock]
            var badge = bus.entity_badge_info(row.entity_id)
            var right_edge = rr.right() - Float32(4.0)
            var badge_edge = right_edge
            var slot_w = Float32(22.0)
            var show_lock = hov or row.locked
            var show_eye = hov or (not row.visible) or (row.is_selected and row_w >= Float32(195.0) and (not badge[0]))
            var lock_r = RectF(right_edge - slot_w, rr.y, slot_w, rr.h)
            if show_lock:
                badge_edge = lock_r.x
            var eye_r = RectF(badge_edge - slot_w, rr.y, slot_w, rr.h)
            if show_eye:
                badge_edge = eye_r.x

            if show_lock:
                var lock_hov = hov and lock_r.contains(ui.mouse_x, ui.mouse_y)
                if lock_hov:
                    ui.draw_list.add_rect(lock_r.center_x() - Float32(10.0), lock_r.center_y() - Float32(10.0), Float32(20.0), Float32(20.0), theme.l3_raised, RADIUS_XS)
                    ui.queue_tooltip_rect(hash_ui_id(rwid, "lock"), lock_r, "Unlock entity" if row.locked else "Lock entity")
                    if ui.mouse_pressed_l:
                        act.toggle_lock_entity_id = row.entity_id
                        clicked_sub = True
                var lcol = theme.status_warn if row.locked else (theme.text_primary if lock_hov else theme.text_muted)
                draw_studio_icon(ui.draw_list, ICON_LOCK, lock_r.center_x() - Float32(6.0), lock_r.center_y() - Float32(6.0), Float32(12.0), lcol, Float32(1.3))

            if show_eye:
                var eye_hov = hov and eye_r.contains(ui.mouse_x, ui.mouse_y)
                if eye_hov:
                    ui.draw_list.add_rect(eye_r.center_x() - Float32(10.0), eye_r.center_y() - Float32(10.0), Float32(20.0), Float32(20.0), theme.l3_raised, RADIUS_XS)
                    ui.queue_tooltip_rect(hash_ui_id(rwid, "eye"), eye_r, "Show entity" if (not row.visible) else "Hide entity")
                    if ui.mouse_pressed_l:
                        act.toggle_visible_entity_id = row.entity_id
                        clicked_sub = True
                var ecol = (
                    theme.text_muted
                    if (not row.visible)
                    else (
                        theme.text_primary
                        if eye_hov
                        else (theme.text_secondary if hov else rgba8_scale_alpha(theme.text_secondary, Float32(0.36)))
                    )
                )
                draw_studio_icon(ui.draw_list, ICON_EYE if row.visible else ICON_EYE_OFF, eye_r.center_x() - Float32(6.0), eye_r.center_y() - Float32(6.0), Float32(12.0), ecol, Float32(1.3))

            # Live Agent Activity Badge (`bus.entity_badge_info`)
            if badge[0]:
                var bcx = badge_edge - Float32(8.0)
                var bcy = rr.center_y()
                var bcol = badge[1]
                if badge[2]:
                    ui.draw_list.add_arc(
                        bcx, bcy, Float32(6.0), Float32(1.5),
                        Float32(0.0), Float32(4.6),
                        bcol, rgba8_scale_alpha(bcol, Float32(0.25)),
                        Float32(3.5), UI_FLAG_DASHED_FLOW,
                    )
                    ui.draw_list.add_circle(bcx, bcy, Float32(2.4), bcol)
                else:
                    ui.draw_list.add_circle(bcx, bcy, Float32(3.0), rgba8_scale_alpha(bcol, Float32(0.75)))
                badge_edge = bcx - Float32(10.0)

            # Collapsed parent `+N` child-count pill badge
            if row.has_children and (not row.is_expanded) and (not query_active):
                var c_txt = ("+" + String(row.child_count)) if row.child_count > 0 else String("···")
                var c_tw = ui._measure_text_styled(c_txt, Float32(10.0), UI_FONT_MONO)
                var c_w = max(Float32(18.0), c_tw + Float32(8.0))
                if badge_edge - c_w > ix + Float32(32.0):
                    var c_r = RectF(badge_edge - c_w - Float32(3.0), rr.center_y() - Float32(7.5), c_w, Float32(15.0))
                    ui.draw_list.add_rect(c_r.x, c_r.y, c_r.w, c_r.h, theme.l0_canvas, RADIUS_PILL, Float32(1.0), theme.border_subtle)
                    ui.draw_text_in_rect(c_r, c_txt, Float32(10.0), theme.text_muted, UI_ALIGN_CENTER, UI_FONT_MONO, Float32(1.0), False, False)
                    badge_edge = c_r.x - Float32(3.0)

            # Truncated entity label (optical cap-height centered + full-name tooltip when truncated!)
            var text_max_w = max(Float32(16.0), badge_edge - ix - Float32(4.0))
            var name_style = UI_FONT_BOLD if row.is_primary else UI_FONT_REGULAR
            var name_disp = ui._truncate_text_ellipsis(row.name, text_max_w, theme.font_body)
            var name_col = theme.text_primary if (row.is_selected or hov) else (theme.text_secondary if row.visible else theme.text_muted)
            ui.draw_text_in_rect(RectF(ix, rr.y, text_max_w, rr.h), name_disp, theme.font_body, name_col, UI_ALIGN_LEFT, name_style, Float32(0.0), False, False)
            if hov and name_disp != row.name:
                ui.queue_tooltip_rect(rwid, rr, row.name)

            # Selection, double-click camera focus & drag-drop reparenting
            ui.last_item_id = rwid
            ui.last_item_rect = rr
            if hov and ui.mouse_pressed_l:
                ui._set_focused(tree_id)
                state.keyboard_entity_id = row.entity_id
            if hov and ui.mouse_pressed_l and not clicked_sub:
                act.select_entity_id = row.entity_id
                act.select_additive = ui.key_ctrl
                act.select_range = ui.key_shift
                if state.last_click_entity_id == row.entity_id and (ui.ui_time_sec - state.last_click_time) <= Float32(0.38):
                    act.focus_entity_id = row.entity_id
                    state.last_click_time = Float32(-10.0)
                else:
                    state.last_click_entity_id = row.entity_id
                    state.last_click_time = ui.ui_time_sec
            if not clicked_sub:
                _ = ui.begin_drag_source("STUDIO_ENTITY", row.entity_id, row.name)
            var drop_hit = ui.accept_drag_target_rect("STUDIO_ENTITY", rr)
            if drop_hit[0] and drop_hit[1] != row.entity_id:
                act.reparent_child_id = drop_hit[1]
                act.reparent_target_id = row.entity_id

    # Draggable sleek 4px -> 6px rounded scrollbar
    _draw_slim_scrollbar(ui, ui.widget_id("outliner_v2_scrollbar"), list_rect, content_h, state.scroll_y, theme, ptr_blocked)

    ui.draw_list.pop_clip()
    ui.draw_list.pop_clip()
    return act^


# ============================================================================
# 5. Panel 2: Redesigned Card-Based Inspector (`draw_inspector_panel_v2`)
# ============================================================================


def draw_inspector_identity(mut ui: UiContext, kind: Int, entity_id: Int, parent_name: String, width: Float32):
    """Show unified semantic icon badge, Bold archetype, mono `#id` chip, and muted parent context."""
    var types = ["Entity", "Mesh", "Point light", "Spot light", "Camera", "Audio emitter", "Prefab", "Foliage", "Collider"]
    ref theme = ui.studio_theme
    var row = ui._alloc_item_rect(ui.widget_id("insp_identity"), width, Float32(26.0))
    var accent = entity_kind_color(kind, theme)

    # 24x24 tinted semantic icon badge
    var badge_sz = Float32(24.0)
    var by = row.center_y() - badge_sz * Float32(0.5)
    ui.draw_list.add_card_surface(
        row.x, by, badge_sz, badge_sz,
        rgba8_scale_alpha(accent, Float32(0.16)),
        rgba8_with_alpha(accent, 92),
        RADIUS_SM, 1.0, True, False,
    )
    draw_studio_icon(ui.draw_list, entity_kind_icon(kind), row.x + Float32(5.0), by + Float32(5.0), Float32(14.0), accent, Float32(1.5))

    var type_lbl = String(types[max(0, min(kind, len(types) - 1))])
    var id_lbl = "#" + String(entity_id)
    var tx = row.x + badge_sz + Float32(8.0)
    var type_w = ui._measure_text_styled(type_lbl, theme.font_caption, UI_FONT_BOLD)
    var id_w = max(Float32(22.0), ui._measure_text_styled(id_lbl, theme.font_mono, UI_FONT_MONO) + Float32(10.0))

    # Right-aligned parent context if room permits
    var right_lim = row.right() - Float32(2.0)
    var parent_room = right_lim - (tx + type_w + Float32(6.0) + id_w + Float32(10.0))
    if parent_name.byte_length() > 0 and parent_room >= Float32(52.0):
        var p_w = min(parent_room, ui._measure_text_styled(parent_name, theme.font_caption, UI_FONT_REGULAR) + Float32(4.0))
        ui.draw_text_in_rect(
            RectF(right_lim - p_w, row.y, p_w, row.h),
            parent_name,
            theme.font_caption,
            theme.text_muted,
            UI_ALIGN_RIGHT,
            UI_FONT_REGULAR,
            Float32(0.0),
            True,
            False,
        )
        right_lim -= p_w + Float32(8.0)

    var avail_type = max(Float32(20.0), right_lim - tx - id_w - Float32(6.0))
    var actual_type_w = min(type_w, avail_type)
    ui.draw_text_in_rect(
        RectF(tx, row.y, actual_type_w, row.h),
        type_lbl,
        theme.font_caption,
        theme.text_primary,
        UI_ALIGN_LEFT,
        UI_FONT_BOLD,
        Float32(0.0),
        True,
        False,
    )
    var id_x = tx + actual_type_w + Float32(6.0)
    if id_x + Float32(18.0) <= right_lim:
        var cw = min(id_w, right_lim - id_x)
        var chip_r = RectF(id_x, row.center_y() - Float32(8.0), cw, Float32(16.0))
        ui.draw_list.add_rect(chip_r.x, chip_r.y, chip_r.w, chip_r.h, theme.l0_canvas, RADIUS_XS, Float32(1.0), theme.border_subtle)
        ui.draw_text_in_rect(chip_r, id_lbl, theme.font_mono, theme.text_secondary, UI_ALIGN_CENTER, UI_FONT_MONO, Float32(2.0), True, False)


def draw_inspector_panel_v2(
    mut ui: UiContext,
    panel_rect: RectF,
    theme: StudioTheme,
    entity: InspectorEntityView,
    mut state: InspectorPanelState,
) -> InspectorAction:
    """Renders the Obsidian v2 card-based Inspector with Entity Header Card, collapsible
    `L2_card` component sections, color-coded XYZ scrubbers, tactile sliders, toggles,
    `+ Add Component` picker, and an Empty Selection state card."""
    var act = InspectorAction()
    if panel_rect.w < Float32(80.0) or panel_rect.h < Float32(60.0):
        return act^

    ui.set_studio_theme(theme)
    ui.draw_list.push_clip_f(panel_rect.x, panel_rect.y, panel_rect.right(), panel_rect.bottom())
    ui.draw_list.add_rect(panel_rect.x, panel_rect.y, panel_rect.w, panel_rect.h, theme.l1_shell)

    var pad = Float32(10.0)
    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup()

    # ---- Empty Selection State Card ----
    if not entity.has_selection:
        var empty_box = RectF(panel_rect.x + pad, panel_rect.y + pad, max(Float32(120.0), panel_rect.w - pad * Float32(2.0)), max(Float32(40.0), panel_rect.h - pad * Float32(2.0)))
        _push_panel_box(ui, "insp_v2_empty", empty_box, gap=Float32(0.0), pad=Float32(0.0))
        if draw_inspector_empty_state(ui, empty_box.w, empty_box.h):
            act.spawn_entity_clicked = True
        _pop_panel_box(ui)
        ui.draw_list.pop_clip()
        return act^

    # Scroll container over Inspector cards
    var sb_w = Float32(6.0)
    var card_w = max(Float32(140.0), panel_rect.w - pad * Float32(2.0) - sb_w)
    var content_top = panel_rect.y + pad - state.scroll_y
    var inner_rect = RectF(panel_rect.x + pad, content_top, card_w, Float32(4000.0))
    _push_panel_box(ui, "insp_v2_body", inner_rect, gap=Float32(8.0), pad=Float32(0.0))

    # ---- 1. Unified Entity Header Card ----
    var hdr_y = ui.box_stack[len(ui.box_stack) - 1].cursor_y
    var hdr_h = Float32(68.0)
    var hdr_r = RectF(inner_rect.x, hdr_y, card_w, hdr_h)
    ui.draw_list.add_card_surface(hdr_r.x, hdr_r.y, hdr_r.w, hdr_r.h, theme.l2_card, theme.border_subtle, RADIUS_MD, 1.0, True, False)

    # 28x28 entity semantic icon badge
    var ico_col = entity_kind_color(entity.entity_kind, theme)
    var ib_r = RectF(hdr_r.x + Float32(10.0), hdr_r.y + Float32(10.0), Float32(28.0), Float32(28.0))
    ui.draw_list.add_card_surface(ib_r.x, ib_r.y, ib_r.w, ib_r.h, rgba8_scale_alpha(ico_col, Float32(0.18)), rgba8_with_alpha(ico_col, 96), RADIUS_SM, 1.0, True, False)
    draw_studio_icon(ui.draw_list, entity_kind_icon(entity.entity_kind), ib_r.x + Float32(6.0), ib_r.y + Float32(6.0), Float32(16.0), ico_col, Float32(1.5))

    # Active toggle switch at top-right of Entity Header Card
    var en_val = entity.enabled
    var tog_r = RectF(hdr_r.right() - Float32(42.0), hdr_r.y + Float32(15.0), Float32(32.0), Float32(17.0))
    if _draw_compact_toggle(ui, ui.widget_id("insp_v2_enabled"), tog_r, en_val, theme):
        act.changed_enabled = True
        act.new_enabled = en_val

    # Editable Bold name field (clean borderless look until hovered/focused, UTF-8 safe)
    var name_x = ib_r.right() + Float32(8.0)
    var name_w = max(Float32(60.0), tog_r.x - name_x - Float32(8.0))
    var name_r = RectF(name_x, hdr_r.y + Float32(11.0), name_w, Float32(24.0))
    var name_wid = ui.widget_id("insp_v2_name")
    var name_hov = (not ptr_blocked) and name_r.contains(ui.mouse_x, ui.mouse_y)
    if name_hov:
        ui.hovered_id = name_wid
        ui.hovered_rect = ui._clip_hovered_rect(name_r)
        if ui.mouse_pressed_l:
            ui._set_focused(name_wid)
    var name_foc = (not blocked) and ui.focused_id == name_wid
    var edit_name = entity.name
    if name_foc:
        ui.text_caret_rect = RectF(name_r.right() - Float32(4.0), name_r.y + Float32(2.0), Float32(2.0), Float32(18.0))
        ui._focused_text_id = name_wid
        ui._focused_text_seen = True
        ui._text_widget_ids.append(name_wid)
        if ui.key_esc_pressed or ui.key_enter_pressed:
            ui.focused_id = 0
            ui._focused_text_id = 0
        elif ui.key_bs_pressed and edit_name.byte_length() > 0:
            var next_n = utf8_drop_last_char(edit_name)
            edit_name = next_n
            act.changed_name = True
            act.new_name = edit_name
        elif (ui.text_utf8.byte_length() > 0 or ui.ime_commit.byte_length() > 0) and not ui.key_ctrl:
            edit_name += ui.text_utf8 + ui.ime_commit
            ui.ime_commit = ""
            act.changed_name = True
            act.new_name = edit_name
    if name_foc or name_hov:
        ui.draw_list.add_recessed_well(
            name_r.x, name_r.y, name_r.w, name_r.h,
            theme.l0_canvas if name_foc else theme.l3_raised,
            theme.border_control,
            RADIUS_XS, 1.0, name_foc, theme.focus_ring,
        )
    ui.draw_text_in_rect(name_r, edit_name, theme.font_title, theme.text_primary, UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(4.0), True, False)

    # Subtitle pills: `#id · Archetype · Parent` (or Multi-Select badge)
    var pill_y = hdr_r.y + Float32(43.0)
    var px_cur = hdr_r.x + Float32(10.0)
    if entity.selected_count > 1:
        px_cur += ui.badge_pill("Multi (" + String(entity.selected_count) + ")", theme.accent_soft, theme.accent_primary, px_cur, pill_y, Float32(17.0), False) + Float32(5.0)
    else:
        px_cur += ui.badge_pill("#" + String(entity.entity_id), theme.l0_canvas, theme.text_secondary, px_cur, pill_y, Float32(17.0), False) + Float32(5.0)
    if px_cur + Float32(54.0) < hdr_r.right() - Float32(8.0):
        px_cur += ui.badge_pill(entity.archetype_label, theme.l3_raised, ico_col, px_cur, pill_y, Float32(17.0), False) + Float32(6.0)
    if entity.parent_name.byte_length() > 0 and px_cur + Float32(40.0) < hdr_r.right() - Float32(8.0):
        ui.draw_text_in_rect(
            RectF(px_cur, pill_y, max(Float32(20.0), hdr_r.right() - px_cur - Float32(10.0)), Float32(17.0)),
            entity.parent_name,
            theme.font_caption,
            theme.text_muted,
            UI_ALIGN_LEFT,
            UI_FONT_REGULAR,
            Float32(0.0),
            True,
            False,
        )

    ui.box_stack[len(ui.box_stack) - 1].cursor_y = hdr_r.bottom() + Float32(8.0)

    # ---- 2. Transform Component Card ----
    var tr_state = ui.collapsible_card("insp_card_transform", "Transform", ICON_MOVE, True, card_w)
    var tr_hdr_r = ui.last_item_rect
    var reset_r = RectF(tr_hdr_r.right() - Float32(26.0), tr_hdr_r.y + Float32(4.0), Float32(20.0), Float32(20.0))
    if _click_rect_button(ui, ui.widget_id("insp_tr_reset"), reset_r, "", theme, ICON_REVERT, False, theme.accent_primary, True):
        act.changed_transform = True
        act.reset_transform = True
        act.new_pos = Vec3(0.0, 0.0, 0.0)
        act.new_rot_deg = Vec3(0.0, 0.0, 0.0)
        act.new_scale = Vec3(1.0, 1.0, 1.0)
    if tr_state[0]:
        var body_y = tr_hdr_r.bottom()
        var body_h = Float32(3) * Float32(28.0) + Float32(14.0)
        _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
        ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
        var inner_w = card_w - Float32(16.0)
        ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
        var pos_v = entity.pos
        var rot_v = entity.rot_deg
        var scl_v = entity.scale
        var tr_changed = False
        if ui.drag_float3_v2("insp_pos", "Position", pos_v, Float32(0.05), Float32(-1.0e5), Float32(1.0e5), inner_w, Float32(0.25)):
            tr_changed = True
        if ui.drag_float3_v2("insp_rot", "Rotation", rot_v, Float32(0.5), Float32(-360.0), Float32(360.0), inner_w, Float32(15.0)):
            tr_changed = True
        if ui.drag_float3_v2("insp_scl", "Scale", scl_v, Float32(0.02), Float32(0.01), Float32(1000.0), inner_w, Float32(0.25)):
            tr_changed = True
        if tr_changed and not act.reset_transform:
            act.changed_transform = True
            act.new_pos = pos_v
            act.new_rot_deg = rot_v
            act.new_scale = scl_v
        ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
        ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 3. Material & Render Component Card ----
    if entity.has_material:
        var mat_state = ui.collapsible_card("insp_card_material", "Material & Render", ICON_MATERIAL, True, card_w)
        var mat_hdr_r = ui.last_item_rect
        if mat_state[0]:
            var body_y = mat_hdr_r.bottom()
            var body_h = Float32(8) * Float32(28.0) + Float32(14.0)
            _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
            var inner_w = card_w - Float32(16.0)

            var sh_mode = entity.shading_mode
            var alb = entity.base_color
            var emi = entity.emissive_color
            var met = entity.metallic
            var rgh = entity.roughness
            var opa = entity.opacity
            var cc = entity.clearcoat
            var shd = entity.cast_shadows
            var m_ch = False
            var sh_opts = List[String](capacity=3)
            sh_opts.append(String("PBR"))
            sh_opts.append(String("Cel"))
            sh_opts.append(String("VFX"))
            if _draw_segmented_row_v2(ui, "insp_shading", "Shading", sh_opts, sh_mode, theme, inner_w):
                m_ch = True
            if _draw_color_swatch_row_v2(ui, "insp_albedo", "Albedo", alb, theme, inner_w):
                m_ch = True
            if _draw_color_swatch_row_v2(ui, "insp_emissive", "Emissive", emi, theme, inner_w):
                m_ch = True
            if ui.slider_float_v2("insp_metal", "Metallic", met, Float32(0.0), Float32(1.0), inner_w):
                m_ch = True
            if ui.slider_float_v2("insp_rough", "Roughness", rgh, Float32(0.04), Float32(1.0), inner_w):
                m_ch = True
            if ui.slider_float_v2("insp_opacity", "Opacity", opa, Float32(0.0), Float32(1.0), inner_w):
                m_ch = True
            if ui.slider_float_v2("insp_clearcoat", "Clearcoat", cc, Float32(0.0), Float32(1.0), inner_w):
                m_ch = True
            if ui.toggle_switch("insp_shadows", "Cast Shadows", shd, inner_w):
                m_ch = True

            if m_ch:
                act.changed_material = True
                act.new_shading_mode = sh_mode
                act.new_base_color = alb
                act.new_emissive_color = emi
                act.new_metallic = met
                act.new_roughness = rgh
                act.new_opacity = opa
                act.new_clearcoat = cc
                act.new_cast_shadows = shd

            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 4. Physics & Collider Card ----
    if entity.has_physics:
        var phy_state = ui.collapsible_card("insp_card_physics", "Physics & Collider", ICON_COLLIDER, True, card_w)
        var phy_hdr_r = ui.last_item_rect
        if phy_state[0]:
            var body_y = phy_hdr_r.bottom()
            var body_h = Float32(4) * Float32(28.0) + Float32(14.0)
            _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
            var inner_w = card_w - Float32(16.0)

            var c_kind = entity.collider_kind
            var c_rad = entity.collider_radius
            var c_mass = entity.mass
            var c_trig = entity.is_trigger
            var p_ch = False
            var col_opts = List[String](capacity=5)
            col_opts.append(String("None"))
            col_opts.append(String("Sph"))
            col_opts.append(String("Box"))
            col_opts.append(String("Cap"))
            col_opts.append(String("Mesh"))
            if _draw_segmented_row_v2(ui, "insp_col_kind", "Shape", col_opts, c_kind, theme, inner_w):
                p_ch = True
            if ui.slider_float_v2("insp_col_rad", "Radius", c_rad, Float32(0.05), Float32(25.0), inner_w):
                p_ch = True
            if ui.slider_float_v2("insp_mass", "Mass", c_mass, Float32(0.0), Float32(100.0), inner_w):
                p_ch = True
            if ui.toggle_switch("insp_trigger", "Is Trigger", c_trig, inner_w):
                p_ch = True

            if p_ch:
                act.changed_physics = True
                act.new_collider_kind = c_kind
                act.new_collider_radius = c_rad
                act.new_mass = c_mass
                act.new_is_trigger = c_trig

            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 5. Conditional Light Card ----
    if entity.has_light:
        var l_ico = ICON_LIGHT_SPOT if entity.light_is_spot else ICON_LIGHT_POINT
        var l_title = "Spot Light" if entity.light_is_spot else "Point Light"
        var l_state = ui.collapsible_card("insp_card_light", l_title, l_ico, True, card_w)
        var l_hdr_r = ui.last_item_rect
        if l_state[0]:
            var n_rows = 4 if entity.light_is_spot else 3
            var body_y = l_hdr_r.bottom()
            var body_h = Float32(n_rows) * Float32(28.0) + Float32(14.0)
            _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
            var inner_w = card_w - Float32(16.0)

            var lcol = entity.light_color
            var lint = entity.light_intensity
            var lrng = entity.light_range
            var lcone = entity.light_cone_deg
            var l_ch = False
            if _draw_color_swatch_row_v2(ui, "insp_l_col", "Color", lcol, theme, inner_w):
                l_ch = True
            if ui.slider_float_v2("insp_l_int", "Intensity", lint, Float32(0.0), Float32(100.0), inner_w):
                l_ch = True
            if ui.slider_float_v2("insp_l_rng", "Range", lrng, Float32(0.5), Float32(120.0), inner_w):
                l_ch = True
            if entity.light_is_spot:
                if ui.slider_float_v2("insp_l_cone", "Cone Deg", lcone, Float32(5.0), Float32(85.0), inner_w):
                    l_ch = True
            if l_ch:
                act.changed_light = True
                act.new_light_color = lcol
                act.new_light_intensity = lint
                act.new_light_range = lrng
                act.new_light_cone_deg = lcone
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 6. Conditional Camera Card ----
    if entity.has_camera:
        var cam_state = ui.collapsible_card("insp_card_camera", "Camera", ICON_CAMERA, True, card_w)
        var cam_hdr_r = ui.last_item_rect
        if cam_state[0]:
            var body_y = cam_hdr_r.bottom()
            var body_h = Float32(2) * Float32(28.0) + Float32(14.0)
            _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
            var inner_w = card_w - Float32(16.0)

            var cfov = entity.camera_fov_deg
            var cmain = entity.camera_is_main
            var c_ch = False
            if ui.slider_float_v2("insp_cam_fov", "FOV Deg", cfov, Float32(20.0), Float32(120.0), inner_w):
                c_ch = True
            if ui.toggle_switch("insp_cam_main", "Main Camera", cmain, inner_w):
                c_ch = True
            if c_ch:
                act.changed_camera = True
                act.new_camera_fov_deg = cfov
                act.new_camera_is_main = cmain
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 7. Conditional Audio Card ----
    if entity.has_audio:
        var aud_state = ui.collapsible_card("insp_card_audio", "Audio Emitter", ICON_AUDIO, True, card_w)
        var aud_hdr_r = ui.last_item_rect
        if aud_state[0]:
            var body_y = aud_hdr_r.bottom()
            var body_h = Float32(3) * Float32(28.0) + Float32(14.0)
            _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
            var inner_w = card_w - Float32(16.0)

            var again = entity.audio_gain
            var aloop = entity.audio_loop
            var aspat = entity.audio_spatial
            var a_ch = False
            if ui.slider_float_v2("insp_aud_gain", "Gain", again, Float32(0.0), Float32(2.0), inner_w):
                a_ch = True
            if ui.toggle_switch("insp_aud_loop", "Loop", aloop, inner_w):
                a_ch = True
            if ui.toggle_switch("insp_aud_spat", "Spatial 3D", aspat, inner_w):
                a_ch = True
            if a_ch:
                act.changed_audio = True
                act.new_audio_gain = again
                act.new_audio_loop = aloop
                act.new_audio_spatial = aspat
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 8. Conditional Nav Card ----
    if entity.has_nav:
        var nav_state = ui.collapsible_card("insp_card_nav", "Nav Agent", ICON_GRID, True, card_w)
        var nav_hdr_r = ui.last_item_rect
        if nav_state[0]:
            var body_y = nav_hdr_r.bottom()
            var body_h = Float32(2) * Float32(28.0) + Float32(14.0)
            _draw_card_body_bg(ui, inner_rect.x, body_y, card_w, body_h, theme)
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + Float32(6.0)
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x + Float32(8.0)
            var inner_w = card_w - Float32(16.0)

            var nspd = entity.nav_speed
            var nrad = entity.nav_radius
            var n_ch = False
            if ui.slider_float_v2("insp_nav_spd", "Move Speed", nspd, Float32(0.1), Float32(25.0), inner_w):
                n_ch = True
            if ui.slider_float_v2("insp_nav_rad", "Agent Radius", nrad, Float32(0.1), Float32(5.0), inner_w):
                n_ch = True
            if n_ch:
                act.changed_nav = True
                act.new_nav_speed = nspd
                act.new_nav_radius = nrad
            ui.box_stack[len(ui.box_stack) - 1].cursor_x = inner_rect.x
            ui.box_stack[len(ui.box_stack) - 1].cursor_y = body_y + body_h + Float32(8.0)

    # ---- 9. `+ Add Component` Button & Inline Picker ----
    var add_y = ui.box_stack[len(ui.box_stack) - 1].cursor_y
    var add_r = RectF(inner_rect.x, add_y, card_w, Float32(28.0))
    if _click_rect_button(ui, ui.widget_id("insp_add_comp_btn"), add_r, "Add Component", theme, ICON_PLUS, state.add_component_open, theme.accent_primary, False):
        state.add_component_open = not state.add_component_open
    ui.box_stack[len(ui.box_stack) - 1].cursor_y = add_r.bottom() + Float32(6.0)

    if state.add_component_open:
        var comp_names = [String("PointLight"), String("Collider"), String("AudioEmitter"), String("Camera"), String("NavAgent")]
        var comp_icons = [ICON_LIGHT_POINT, ICON_COLLIDER, ICON_AUDIO, ICON_CAMERA, ICON_GRID]
        var pick_y = ui.box_stack[len(ui.box_stack) - 1].cursor_y
        var pick_h = Float32(len(comp_names)) * Float32(26.0) + Float32(8.0)
        ui.draw_list.add_rect_styled(
            inner_rect.x, pick_y, card_w, pick_h,
            theme.l2_card, theme.border_control,
            RADIUS_MD, 1.0, 1.0, UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
        )
        for ci in range(len(comp_names)):
            var cr = RectF(inner_rect.x + Float32(4.0), pick_y + Float32(4.0) + Float32(ci) * Float32(26.0), card_w - Float32(8.0), Float32(24.0))
            if _click_rect_button(ui, ui.widget_id("insp_pick_" + comp_names[ci]), cr, comp_names[ci], theme, comp_icons[ci], False, theme.accent_primary, True):
                act.add_component_kind = comp_names[ci]
                state.add_component_open = False
        ui.box_stack[len(ui.box_stack) - 1].cursor_y = pick_y + pick_h + Float32(8.0)

    var total_content_h = ui.box_stack[len(ui.box_stack) - 1].cursor_y - content_top + Float32(12.0)
    _pop_panel_box(ui)

    # Wheel scroll & draggable sleek 4px -> 6px scrollbar
    var max_scroll = max(Float32(0.0), total_content_h - max(Float32(1.0), panel_rect.h - pad * Float32(2.0)))
    if (not ptr_blocked) and panel_rect.contains(ui.mouse_x, ui.mouse_y) and ui.mouse_wheel != Float32(0.0):
        state.scroll_y = max(Float32(0.0), min(max_scroll, state.scroll_y - ui.mouse_wheel * Float32(36.0)))
    state.scroll_y = max(Float32(0.0), min(max_scroll, state.scroll_y))
    _draw_slim_scrollbar(ui, ui.widget_id("insp_v2_scrollbar"), panel_rect, total_content_h + pad * Float32(2.0), state.scroll_y, theme, ptr_blocked)

    ui.draw_list.pop_clip()
    return act^


# ============================================================================
# 6. Panel 3: Redesigned Visual Asset Browser (`draw_asset_browser_panel_v2`)
# ============================================================================


def draw_asset_browser_panel_v2(
    mut ui: UiContext,
    panel_rect: RectF,
    theme: StudioTheme,
    tiles: List[AssetTileItem],
    mounted_packs: Int,
    mcp_clients: Int,
    mut state: AssetBrowserPanelState,
) -> AssetBrowserAction:
    """Renders the Obsidian & Flint Ember Asset & Content Drawer with unified 34px toolbar
    (interactive breadcrumb + count chip + type filter pills + Create/Refresh/+Import),
    conditional folder rail when subfolders exist, crafted Two-Zone Empty State / Compact HiDPI
    Fallback, and responsive 92x100 `L2_card` visual asset tiles (double-click `.mojo` routes
    to `CodeDrawer`)."""
    var act = AssetBrowserAction()
    if panel_rect.w < Float32(120.0) or panel_rect.h < Float32(52.0):
        return act^
    _ = mounted_packs
    _ = mcp_clients

    ui.set_studio_theme(theme)
    ui.draw_list.push_clip_f(panel_rect.x, panel_rect.y, panel_rect.right(), panel_rect.bottom())
    ui.draw_list.add_rect(panel_rect.x, panel_rect.y, panel_rect.w, panel_rect.h, theme.l1_shell)

    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup()

    # ---- Top Bar (34px): Breadcrumb + Count + Type Filter Pills + Actions ----
    var bar_h = Float32(34.0)
    var pill_h = Float32(24.0)
    var bar_y = panel_rect.y + Float32(5.0)
    var cx = panel_rect.x + Float32(8.0)

    # Interactive breadcrumb root pill (`Project`)
    var root_bc_w = Float32(72.0)
    if _click_rect_button(ui, ui.widget_id("ab_v2_bc_root"), RectF(cx, bar_y, root_bc_w, pill_h), "Project", theme, ICON_FOLDER, state.active_folder.byte_length() == 0, theme.accent_primary, True, True):
        state.active_folder = String("")
    cx += root_bc_w + Float32(5.0)

    if state.active_folder.byte_length() > 0:
        ui.draw_text_in_rect(RectF(cx, bar_y, Float32(8.0), pill_h), "/", theme.font_caption, theme.text_muted, UI_ALIGN_CENTER, UI_FONT_REGULAR, Float32(0.0), False, False)
        cx += Float32(9.0)
        var f_lbl = ui._truncate_text_ellipsis(state.active_folder, Float32(96.0), theme.font_caption)
        var fw = max(Float32(42.0), ui._measure_text_styled(f_lbl, theme.font_caption, UI_FONT_REGULAR) + Float32(14.0))
        _ = ui.badge_pill(f_lbl, theme.accent_soft, theme.accent_primary, cx, bar_y + Float32(2.0), Float32(20.0), False)
        cx += fw + Float32(6.0)
    elif len(tiles) > 0 and panel_rect.w >= Float32(360.0):
        var cnt_txt = String(len(tiles))
        var cnt_w = max(Float32(22.0), ui._measure_text_styled(cnt_txt, theme.font_mono, UI_FONT_MONO) + Float32(10.0))
        var cnt_r = RectF(cx, bar_y + Float32(3.0), cnt_w, Float32(18.0))
        ui.draw_list.add_rect(cnt_r.x, cnt_r.y, cnt_r.w, cnt_r.h, theme.l0_canvas, RADIUS_PILL, Float32(1.0), theme.border_subtle)
        ui.draw_text_in_rect(cnt_r, cnt_txt, theme.font_mono, theme.text_secondary, UI_ALIGN_CENTER, UI_FONT_MONO, Float32(2.0), False, False)
        cx += cnt_w + Float32(8.0)

    # Right-side action buttons (`+ Import`, `Refresh`)
    var right_x = panel_rect.right() - Float32(8.0)
    var imp_w = Float32(76.0) if panel_rect.w >= Float32(290.0) else Float32(26.0)
    var imp_r = RectF(right_x - imp_w, bar_y, imp_w, pill_h)
    if _click_rect_button(ui, ui.widget_id("ab_v2_import"), imp_r, "Import" if imp_w > Float32(30.0) else "", theme, ICON_PLUS, True, theme.accent_primary, True, True):
        act.import_clicked = True
    right_x = imp_r.x - Float32(5.0)

    var res_w = Float32(26.0)
    var res_r = RectF(right_x - res_w, bar_y, res_w, pill_h)
    var res_wid = ui.widget_id("ab_v2_rescan")
    if _click_rect_button(ui, res_wid, res_r, "", theme, ICON_REFRESH, False, theme.accent_primary, True):
        act.rescan_clicked = True
    if ui.hovered_id == res_wid:
        ui.queue_tooltip_rect(res_wid, res_r, "Rescan project assets")
    right_x = res_r.x - Float32(8.0)

    # Type filter pills (`All | Scenes | Meshes | Materials | Audio | Scripts | Prefabs`)
    if panel_rect.w >= Float32(460.0):
        var filter_names = [
            String("All"),
            String("Scenes"),
            String("Meshes"),
            String("Materials"),
            String("Audio"),
            String("Scripts"),
            String("Prefabs"),
        ]
        for fi in range(len(filter_names)):
            var fw = max(Float32(34.0), ui._measure_text_styled(filter_names[fi], theme.font_caption, UI_FONT_BOLD if state.active_filter == fi else UI_FONT_REGULAR) + Float32(14.0))
            if cx + fw > right_x - Float32(4.0):
                break
            var fr = RectF(cx, bar_y, fw, pill_h)
            var f_acc = asset_filter_accent(fi, theme) if fi > 0 else theme.accent_primary
            if _click_rect_button(ui, ui.widget_id("ab_v2_f_" + String(fi)), fr, filter_names[fi], theme, -1, state.active_filter == fi, f_acc, True):
                state.active_filter = fi
            cx += fw + Float32(4.0)

    var body_y = panel_rect.y + bar_h
    ui.draw_list.add_line(panel_rect.x, body_y, panel_rect.right(), body_y, Float32(1.0), theme.border_subtle)

    # Collect unique folders; show Left Folder Tree ONLY when subfolders actually exist!
    var folders = List[String]()
    folders.append(String(""))
    for ti in range(len(tiles)):
        var f = tiles[ti].folder
        if f.byte_length() == 0:
            continue
        var seen = False
        for fj in range(len(folders)):
            if folders[fj] == f:
                seen = True
                break
        if not seen:
            folders.append(f)

    var show_tree = len(folders) > 1 and panel_rect.w >= Float32(460.0)
    var tree_w = Float32(136.0) if show_tree else Float32(0.0)
    if show_tree:
        var tree_r = RectF(panel_rect.x, body_y, tree_w, max(Float32(10.0), panel_rect.bottom() - body_y))
        ui.draw_list.add_rect(tree_r.x, tree_r.y, tree_r.w, tree_r.h, theme.l0_canvas)
        ui.draw_list.add_line(tree_r.right(), tree_r.y, tree_r.right(), tree_r.bottom(), Float32(1.0), theme.border_subtle)

        var fy = tree_r.y + Float32(6.0)
        for fi in range(len(folders)):
            if fy + Float32(24.0) > tree_r.bottom():
                break
            var f_key = folders[fi]
            var f_lbl = String("All Assets") if f_key.byte_length() == 0 else f_key
            var fr = RectF(tree_r.x + Float32(5.0), fy, tree_r.w - Float32(10.0), Float32(22.0))
            var is_sel = state.active_folder == f_key
            var hov = (not ptr_blocked) and fr.contains(ui.mouse_x, ui.mouse_y)
            if hov:
                ui.hovered_id = ui.widget_id("ab_v2_dir_" + String(fi))
                ui.hovered_rect = ui._clip_hovered_rect(fr)
                if ui.mouse_pressed_l:
                    state.active_folder = f_key
            if is_sel:
                ui.draw_list.add_rect_styled(fr.x, fr.y, fr.w, fr.h, theme.accent_soft, rgba8_with_alpha(theme.accent_primary, 76), RADIUS_XS, 1.0, 1.0, UI_FLAG_BORDER)
            elif hov:
                ui.draw_list.add_rect(fr.x, fr.y, fr.w, fr.h, theme.l2_card, RADIUS_XS)
            draw_studio_icon(ui.draw_list, ICON_FOLDER, fr.x + Float32(6.0), fr.center_y() - Float32(6.0), Float32(12.0), theme.accent_primary if is_sel else theme.text_muted, Float32(1.3))
            ui.draw_text_in_rect(
                RectF(fr.x + Float32(22.0), fr.y, max(Float32(16.0), fr.w - Float32(26.0)), fr.h),
                f_lbl,
                theme.font_caption,
                theme.text_primary if is_sel else theme.text_secondary,
                UI_ALIGN_LEFT,
                UI_FONT_BOLD if is_sel else UI_FONT_REGULAR,
                Float32(0.0),
                True,
                False,
            )
            fy += Float32(24.0)

    # Filter matching tiles
    var filtered = List[Int](capacity=len(tiles))
    for ti in range(len(tiles)):
        ref t = tiles[ti]
        if state.active_filter != ASSET_FILTER_ALL and t.filter_kind != state.active_filter:
            continue
        if state.active_folder.byte_length() > 0 and t.folder != state.active_folder:
            continue
        filtered.append(ti)

    var content_area = RectF(
        panel_rect.x + tree_w + Float32(8.0),
        body_y + Float32(8.0),
        max(Float32(80.0), panel_rect.w - tree_w - Float32(16.0)),
        max(Float32(16.0), panel_rect.bottom() - body_y - Float32(16.0)),
    )

    if len(filtered) == 0:
        # Deliberately crafted Two-Zone Empty State (or Compact HiDPI Fallback when short!)
        var deck_res = draw_asset_empty_deck(ui, content_area, theme, len(tiles) > 0)
        if deck_res[0]:
            act.import_clicked = True
        if deck_res[1] >= 0:
            act.spawn_preset_kind = deck_res[1]
    else:
        # Responsive 92x100 Visual Asset Card Grid
        var tile_w = max(ASSET_CARD_W, Float32(92.0))
        var tile_h = max(ASSET_CARD_H, Float32(100.0))
        var gap = Float32(8.0)
        var sb_pad = Float32(10.0)
        var grid_w = max(tile_w, content_area.w - sb_pad)
        var cols = max(1, Int((grid_w + gap) / (tile_w + gap)))
        var n_rows = (len(filtered) + cols - 1) // cols
        var content_h = Float32(n_rows) * (tile_h + gap) + Float32(4.0)
        var max_scroll = max(Float32(0.0), content_h - content_area.h)
        if (not ptr_blocked) and content_area.contains(ui.mouse_x, ui.mouse_y) and ui.mouse_wheel != Float32(0.0):
            state.scroll_y = max(Float32(0.0), min(max_scroll, state.scroll_y - ui.mouse_wheel * Float32(36.0)))
        state.scroll_y = max(Float32(0.0), min(max_scroll, state.scroll_y))

        ui.draw_list.push_clip_f(content_area.x, content_area.y, content_area.right(), content_area.bottom())
        for idx in range(len(filtered)):
            var r_idx = idx // cols
            var c_idx = idx % cols
            var tx = content_area.x + Float32(c_idx) * (tile_w + gap)
            var ty = content_area.y + Float32(r_idx) * (tile_h + gap) - state.scroll_y
            if ty + tile_h < content_area.y or ty > content_area.bottom():
                ui.culled_items_this_frame += 1
                continue

            var item = tiles[filtered[idx]].copy()
            var tr = RectF(tx, ty, tile_w, tile_h)
            var twid = ui.widget_id("ab_tile_" + item.rel_path)
            var is_sel = state.selected_path == item.rel_path
            var tile_res = draw_asset_card_tile(ui, twid, tr, item, is_sel, theme)
            _ = ui.begin_drag_source("ASSET_ENTRY", filtered[idx], item.rel_path)
            if tile_res[0]:
                state.selected_path = item.rel_path
                act.selected_path = item.rel_path
                var is_dbl = state.last_click_path == item.rel_path and (ui.ui_time_sec - state.last_click_time) <= Float32(0.40)
                if is_dbl:
                    if _ascii_lower_panel(item.rel_path).endswith(".mojo"):
                        act.open_in_code_drawer_path = item.rel_path
                    else:
                        act.activated_path = item.rel_path
                    state.last_click_time = Float32(-10.0)
                else:
                    state.last_click_path = item.rel_path
                    state.last_click_time = ui.ui_time_sec

        _draw_slim_scrollbar(ui, ui.widget_id("ab_v2_scrollbar"), content_area, content_h, state.scroll_y, theme, ptr_blocked)
        ui.draw_list.pop_clip()

    ui.draw_list.pop_clip()
    return act^
