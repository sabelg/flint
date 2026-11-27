"""Flint Studio Design System Tokens, Themes, Icon IDs & TOML Theme Loader (`flint.ui.theme`).

Pure Mojo 1.2 design system defining:
  - Surface elevation ladder (`l0_canvas` .. `l4_overlay` + `l4_scrim`)
  - Alpha hairline borders & top inner bevel highlight
  - Color-vision-safe semantic accents (Primary, Play/PIE, Visual Agent, Status, XYZ axes)
  - Text hierarchy, typography scale, spacing rhythm (`4/8/12/16px`), and corner radii
  - 14-class syntax highlighting palette shared across Code Editor & Agent Diff views
  - 4 built-in themes (`Flint Obsidian`, `Midnight Studio`, `Okabe-Ito`, `Paper Light`)
  - Pure-Mojo TOML theme override loader (`[roles]`, `[colors]`, `[syntax]`)
  - Canonical `UiIconId` registry and file-path icon classifier (`icon_for_path`)
"""
from std.collections import List
from std.math import isfinite, pow


# ============================================================================
# 1. Color Packing, Interpolation, Hex Parsing & WCAG Contrast
# ============================================================================


@always_inline
def rgba8(r: Int, g: Int, b: Int, a: Int = 255) -> UInt32:
    """Packs `[0..255]` channels into Flint `0xAARRGGBB` (`B=[0:7], G=[8:15], R=[16:23], A=[24:31]`)."""
    var cr = UInt32(max(0, min(255, r)))
    var cg = UInt32(max(0, min(255, g)))
    var cb = UInt32(max(0, min(255, b)))
    var ca = UInt32(max(0, min(255, a)))
    return cb | (cg << 8) | (cr << 16) | (ca << 24)


@always_inline
def rgb_hex(rgb24: UInt32, a: Int = 255) -> UInt32:
    """Converts a `0xRRGGBB` literal plus `[0..255]` alpha into Flint `0xAARRGGBB`."""
    var ca = UInt32(max(0, min(255, a)))
    return (rgb24 & UInt32(0x00FFFFFF)) | (ca << 24)


@always_inline
def rgba8_r(col: UInt32) -> Int:
    return Int((col >> 16) & UInt32(0xFF))


@always_inline
def rgba8_g(col: UInt32) -> Int:
    return Int((col >> 8) & UInt32(0xFF))


@always_inline
def rgba8_b(col: UInt32) -> Int:
    return Int(col & UInt32(0xFF))


@always_inline
def rgba8_a(col: UInt32) -> Int:
    return Int((col >> 24) & UInt32(0xFF))


@always_inline
def rgba8_with_alpha(col: UInt32, alpha: Int) -> UInt32:
    """Returns `col` with its alpha channel replaced by `clamp(alpha, 0, 255)`."""
    var ca = UInt32(max(0, min(255, alpha)))
    return (col & UInt32(0x00FFFFFF)) | (ca << 24)


@always_inline
def rgba8_scale_alpha(col: UInt32, factor: Float32) -> UInt32:
    """Multiplies `col`'s alpha channel by `clamp(factor, 0.0, 1.0)` (NaN-safe)."""
    var f = max(Float32(0.0), min(Float32(1.0), factor)) if isfinite(factor) else Float32(0.0)
    var na = Int(Float32(rgba8_a(col)) * f + Float32(0.5))
    return rgba8_with_alpha(col, na)


@always_inline
def lerp_rgba8(a: UInt32, b: UInt32, t: Float32) -> UInt32:
    """Linearly interpolates two `0xAARRGGBB` colors with `t` clamped to `[0, 1]`."""
    var ct = max(Float32(0.0), min(Float32(1.0), t)) if isfinite(t) else Float32(0.0)
    var r = Int(Float32(rgba8_r(a)) + Float32(rgba8_r(b) - rgba8_r(a)) * ct + Float32(0.5))
    var g = Int(Float32(rgba8_g(a)) + Float32(rgba8_g(b) - rgba8_g(a)) * ct + Float32(0.5))
    var bl = Int(Float32(rgba8_b(a)) + Float32(rgba8_b(b) - rgba8_b(a)) * ct + Float32(0.5))
    var al = Int(Float32(rgba8_a(a)) + Float32(rgba8_a(b) - rgba8_a(a)) * ct + Float32(0.5))
    return rgba8(r, g, bl, al)


@always_inline
def mix_rgba8(a: UInt32, b: UInt32, percent_a: Int) -> UInt32:
    """Blends `percent_a`% of `a` into `(100 - percent_a)%` of `b`."""
    var pa = max(0, min(100, percent_a))
    var pb = 100 - pa
    var r = (rgba8_r(a) * pa + rgba8_r(b) * pb + 50) // 100
    var g = (rgba8_g(a) * pa + rgba8_g(b) * pb + 50) // 100
    var bl = (rgba8_b(a) * pa + rgba8_b(b) * pb + 50) // 100
    var al = (rgba8_a(a) * pa + rgba8_a(b) * pb + 50) // 100
    return rgba8(r, g, bl, al)


def _hex_nibble(b: UInt8) -> Int:
    var c = Int(b)
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return -1


def parse_hex_rgba8(text: String) -> Tuple[Bool, UInt32]:
    """Parses `#RGB`, `#RGBA`, `#RRGGBB`, or `#RRGGBBAA` into Flint `0xAARRGGBB`."""
    var s = String(text.strip())
    var n = s.byte_length()
    var bytes = s.as_bytes()
    if n < 4 or bytes[0] != UInt8(35):
        return (False, UInt32(0))
    var nibbles = List[Int](capacity=n - 1)
    for i in range(1, n):
        var v = _hex_nibble(bytes[i])
        if v < 0:
            return (False, UInt32(0))
        nibbles.append(v)
    var count = len(nibbles)
    if count == 3:
        return (True, rgba8(nibbles[0] * 17, nibbles[1] * 17, nibbles[2] * 17, 255))
    if count == 4:
        return (True, rgba8(nibbles[0] * 17, nibbles[1] * 17, nibbles[2] * 17, nibbles[3] * 17))
    if count == 6:
        var r = (nibbles[0] << 4) | nibbles[1]
        var g = (nibbles[2] << 4) | nibbles[3]
        var b = (nibbles[4] << 4) | nibbles[5]
        return (True, rgba8(r, g, b, 255))
    if count == 8:
        var r = (nibbles[0] << 4) | nibbles[1]
        var g = (nibbles[2] << 4) | nibbles[3]
        var b = (nibbles[4] << 4) | nibbles[5]
        var a = (nibbles[6] << 4) | nibbles[7]
        return (True, rgba8(r, g, b, a))
    return (False, UInt32(0))


def _srgb_channel_linear(c8: Int) -> Float32:
    var c = Float32(c8) / Float32(255.0)
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    return pow((c + Float32(0.055)) / Float32(1.055), Float32(2.4))


def relative_luminance_rgba8(col: UInt32) -> Float32:
    """WCAG 2.1 relative luminance in `[0.0, 1.0]`."""
    var r = _srgb_channel_linear(rgba8_r(col))
    var g = _srgb_channel_linear(rgba8_g(col))
    var b = _srgb_channel_linear(rgba8_b(col))
    return Float32(0.2126) * r + Float32(0.7152) * g + Float32(0.0722) * b


def contrast_ratio_rgba8(fg: UInt32, bg: UInt32) -> Float32:
    """WCAG 2.1 contrast ratio (`1.0` .. `21.0`)."""
    var l1 = relative_luminance_rgba8(fg)
    var l2 = relative_luminance_rgba8(bg)
    var hi = max(l1, l2)
    var lo = min(l1, l2)
    return (hi + Float32(0.05)) / (lo + Float32(0.05))


# ============================================================================
# 2. Spacing Rhythm, Corner Radii, Typography Scale & Layout Constants
# ============================================================================

comptime SPACE_XS: Float32 = 4.0
comptime SPACE_SM: Float32 = 8.0
comptime SPACE_MD: Float32 = 12.0
comptime SPACE_LG: Float32 = 16.0
comptime SPACE_XL: Float32 = 24.0

comptime RADIUS_XS: Float32 = 4.0
comptime RADIUS_SM: Float32 = 6.0
comptime RADIUS_MD: Float32 = 8.0
comptime RADIUS_LG: Float32 = 12.0
comptime RADIUS_PILL: Float32 = 999.0

comptime TYPE_TITLE_PX: Float32 = 15.0
comptime TYPE_BODY_PX: Float32 = 13.0
comptime TYPE_SECTION_PX: Float32 = 12.0
comptime TYPE_MONO_PX: Float32 = 12.0
comptime TYPE_CAPTION_PX: Float32 = 11.5

comptime LINE_TITLE_PX: Float32 = 20.0
comptime LINE_BODY_PX: Float32 = 18.0
comptime LINE_SECTION_PX: Float32 = 16.0
comptime LINE_CAPTION_PX: Float32 = 15.0

comptime HEADER_BAR_H: Float32 = 40.0
comptime OUTLINER_ROW_H: Float32 = 26.0
comptime CARD_HEADER_H: Float32 = 28.0
comptime INSPECTOR_ROW_H: Float32 = 24.0
comptime INSPECTOR_LABEL_RATIO: Float32 = 0.38
comptime ASSET_CARD_W: Float32 = 88.0
comptime ASSET_CARD_H: Float32 = 96.0
comptime SIDEBAR_RAIL_W: Float32 = 44.0

# 14-class syntax palette indices (matches syntax highlighter token classes)
comptime SYN_TEXT: Int = 0
comptime SYN_KEYWORD: Int = 1
comptime SYN_TYPE: Int = 2
comptime SYN_COMMENT: Int = 3
comptime SYN_STRING: Int = 4
comptime SYN_NUMBER: Int = 5
comptime SYN_DECORATOR: Int = 6
comptime SYN_SYMBOL: Int = 7
comptime SYN_BUILTIN: Int = 8
comptime SYN_SELF: Int = 9
comptime SYN_DEF: Int = 10
comptime SYN_PARAM: Int = 11
comptime SYN_PROPERTY: Int = 12
comptime SYN_NAMESPACE: Int = 13
comptime SYN_COUNT: Int = 14

# Built-in theme indices
comptime THEME_FLINT_OBSIDIAN: Int = 0
comptime THEME_MIDNIGHT_STUDIO: Int = 1
comptime THEME_OKABE_ITO: Int = 2
comptime THEME_PAPER_LIGHT: Int = 3
comptime STUDIO_THEME_COUNT: Int = 4

# Agent status semantic codes (aligned with AgentStatus in agent_bus)
comptime AGENT_STATE_IDLE: Int = 0
comptime AGENT_STATE_THINKING: Int = 1
comptime AGENT_STATE_PLANNING: Int = 2
comptime AGENT_STATE_EDITING_SCENE: Int = 3
comptime AGENT_STATE_WRITING_CODE: Int = 4
comptime AGENT_STATE_BUILDING: Int = 5
comptime AGENT_STATE_TESTING_PIE: Int = 6
comptime AGENT_STATE_WAITING_APPROVAL: Int = 7
comptime AGENT_STATE_ERROR: Int = 8


# ============================================================================
# 3. Canonical Vector/SDF Icon Registry (`UiIconId`) & Path Classifier
# ============================================================================

# Transport (0..3)
comptime ICON_PLAY: Int = 0
comptime ICON_PAUSE: Int = 1
comptime ICON_STEP: Int = 2
comptime ICON_STOP: Int = 3

# Gizmo & Viewport (4..11)
comptime ICON_MOVE: Int = 4
comptime ICON_ROTATE: Int = 5
comptime ICON_SCALE: Int = 6
comptime ICON_WORLD: Int = 7
comptime ICON_LOCAL: Int = 8
comptime ICON_SNAP: Int = 9
comptime ICON_CAMERA: Int = 10
comptime ICON_GRID: Int = 11

# Entities & Components (12..19)
comptime ICON_CUBE: Int = 12
comptime ICON_LIGHT_POINT: Int = 13
comptime ICON_LIGHT_SPOT: Int = 14
comptime ICON_AUDIO: Int = 15
comptime ICON_FOLIAGE: Int = 16
comptime ICON_EMPTY: Int = 17
comptime ICON_PREFAB: Int = 18
comptime ICON_COLLIDER: Int = 19

# Chrome & Actions (20..27)
comptime ICON_EYE: Int = 20
comptime ICON_EYE_OFF: Int = 21
comptime ICON_LOCK: Int = 22
comptime ICON_SEARCH: Int = 23
comptime ICON_PLUS: Int = 24
comptime ICON_CHEVRON: Int = 25
comptime ICON_MORE_V: Int = 26
comptime ICON_CLOSE: Int = 27

# Agent Theatre & Diff Controls (28..33)
comptime ICON_SPARK: Int = 28
comptime ICON_CHECK: Int = 29
comptime ICON_WARN: Int = 30
comptime ICON_REVERT: Int = 31
comptime ICON_DIFF: Int = 32
comptime ICON_FOCUS_CAM: Int = 33

# Asset & File Browser Icons (34..38)
comptime ICON_FOLDER: Int = 34
comptime ICON_CODE: Int = 35
comptime ICON_MATERIAL: Int = 36
comptime ICON_SCENE: Int = 37
comptime ICON_FILE: Int = 38

# Additional Studio HUD & Utility Curved Icons (39..43)
comptime ICON_REFRESH: Int = 39
comptime ICON_CLOCK: Int = 40
comptime ICON_BELL: Int = 41
comptime ICON_SUN: Int = 42
comptime ICON_MOON: Int = 43

# Semantic Icon Aliases
comptime ICON_AGENT_SPARK: Int = ICON_SPARK
comptime ICON_GLOBE: Int = ICON_WORLD

comptime ICON_COUNT: Int = 44


@fieldwise_init
struct UiIconId(Copyable, ImplicitlyCopyable, Movable):
    """Typed handle wrapping a canonical `ICON_*` identifier."""
    var value: Int

    def __init__(out self):
        self.value = ICON_FILE

    @always_inline
    def is_valid(self) -> Bool:
        return self.value >= 0 and self.value < ICON_COUNT


def _ascii_lower(s: String) -> String:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        if b >= 65 and b <= 90:
            out.append(b + 32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def icon_for_path(path: String, is_dir: Bool = False) -> Int:
    """Classifies a workspace file or asset path into its canonical `ICON_*` glyph."""
    if is_dir:
        return ICON_FOLDER
    var p = _ascii_lower(String(path.strip()))
    if p.endswith(".mojo") or p.endswith(".py") or p.endswith(".pyi") or p.endswith(".sh") or p.endswith(".bash"):
        return ICON_CODE
    if p.endswith(".flscn") or p.endswith(".flscnb"):
        return ICON_SCENE
    if p.endswith(".flmat"):
        return ICON_MATERIAL
    if p.endswith(".flprefab"):
        return ICON_PREFAB
    if p.endswith(".glb") or p.endswith(".gltf") or p.endswith(".obj") or p.endswith(".flmesh"):
        return ICON_CUBE
    if p.endswith(".flanim") or p.endswith(".flrig"):
        return ICON_PLAY
    if p.endswith(".wav") or p.endswith(".qoa") or p.endswith(".ogg") or p.endswith(".flac") or p.endswith(".flmix"):
        return ICON_AUDIO
    if p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".ktx2") or p.endswith(".hdr"):
        return ICON_CAMERA
    if p.endswith(".flnav") or p.endswith(".flter"):
        return ICON_GRID
    if p.endswith(".flpk") or p.endswith(".lock"):
        return ICON_LOCK
    if p.endswith(".diff") or p.endswith(".patch"):
        return ICON_DIFF
    return ICON_FILE


# ============================================================================
# 4. `StudioTheme` (`UiThemeV2`) Struct & Built-in Presets
# ============================================================================


def _make_syntax_list(
    c_text: UInt32,
    c_kw: UInt32,
    c_type: UInt32,
    c_comment: UInt32,
    c_string: UInt32,
    c_number: UInt32,
    c_deco: UInt32,
    c_sym: UInt32,
    c_builtin: UInt32,
    c_self: UInt32,
    c_def: UInt32,
    c_param: UInt32,
    c_prop: UInt32,
    c_ns: UInt32,
) -> List[UInt32]:
    var out = List[UInt32](capacity=SYN_COUNT)
    out.append(c_text)
    out.append(c_kw)
    out.append(c_type)
    out.append(c_comment)
    out.append(c_string)
    out.append(c_number)
    out.append(c_deco)
    out.append(c_sym)
    out.append(c_builtin)
    out.append(c_self)
    out.append(c_def)
    out.append(c_param)
    out.append(c_prop)
    out.append(c_ns)
    return out^


struct StudioTheme(Copyable, Movable):
    """Complete Flint Studio Design System token set (`UiThemeV2`)."""
    var name: String
    var id: String
    var is_dark: Bool
    var reduced_motion: Bool

    # Surface elevation ladder L0-L4
    var l0_canvas: UInt32
    var l1_shell: UInt32
    var l2_card: UInt32
    var l3_raised: UInt32
    var l3_hover: UInt32
    var l3_press: UInt32
    var l4_overlay: UInt32
    var l4_scrim: UInt32

    # Hairline borders & top inner bevel
    var border_subtle: UInt32
    var border_control: UInt32
    var border_strong: UInt32
    var bevel_top: UInt32

    # Semantic accents
    var accent_primary: UInt32
    var accent_soft: UInt32
    var focus_ring: UInt32
    var play_mode: UInt32
    var play_tint: UInt32
    var play_paused: UInt32
    var agent_active: UInt32
    var agent_flow: UInt32
    var agent_soft: UInt32
    var status_ok: UInt32
    var status_warn: UInt32
    var status_danger: UInt32
    var axis_x: UInt32
    var axis_y: UInt32
    var axis_z: UInt32

    # Text hierarchy
    var text_primary: UInt32
    var text_secondary: UInt32
    var text_muted: UInt32
    var text_on_accent: UInt32

    # Typography & layout metrics
    var font_title: Float32
    var font_section: Float32
    var font_body: Float32
    var font_caption: Float32
    var font_mono: Float32
    var row_height: Float32
    var header_height: Float32
    var scrollbar_width: Float32
    var radius_xs: Float32
    var radius_sm: Float32
    var radius_md: Float32
    var radius_lg: Float32
    var space_xs: Float32
    var space_sm: Float32
    var space_md: Float32
    var space_lg: Float32

    # 14-class syntax palette + direct named syntax/diff tokens
    var syntax: List[UInt32]
    var syntax_keyword: UInt32
    var syntax_type: UInt32
    var syntax_string: UInt32
    var syntax_number: UInt32
    var syntax_comment: UInt32
    var syntax_func: UInt32
    var syntax_macro: UInt32
    var syntax_operator: UInt32
    var diff_add_bg: UInt32
    var diff_del_bg: UInt32
    var diff_add_fg: UInt32
    var diff_del_fg: UInt32
    var diff_hunk_bg: UInt32

    def __init__(out self):
        self.name = "Flint Obsidian"
        self.id = "obsidian"
        self.is_dark = True
        self.reduced_motion = False

        # L0-L4 obsidian slate elevation ladder
        self.l0_canvas = rgba8(11, 13, 17, 255)        # #0B0D11
        self.l1_shell = rgba8(18, 21, 27, 255)         # #12151B
        self.l2_card = rgba8(25, 29, 38, 255)          # #191D26
        self.l3_raised = rgba8(34, 40, 53, 255)        # #222835
        self.l3_hover = rgba8(43, 51, 68, 255)         # #2B3344
        self.l3_press = rgba8(31, 37, 49, 255)         # #1F2531
        self.l4_overlay = rgba8(20, 24, 33, 224)       # #141821E0
        self.l4_scrim = rgba8(7, 9, 13, 184)           # #07090DB8

        # Hairlines & 1px inner top highlight
        self.border_subtle = rgba8(255, 255, 255, 18)  # #FFFFFF12
        self.border_control = rgba8(255, 255, 255, 31) # #FFFFFF1F
        self.border_strong = rgba8(255, 255, 255, 51)  # #FFFFFF33
        self.bevel_top = rgba8(255, 255, 255, 20)      # #FFFFFF14

        # Semantic accents
        self.accent_primary = rgba8(76, 155, 255, 255) # #4C9BFF
        self.accent_soft = rgba8(76, 155, 255, 36)     # #4C9BFF24
        self.focus_ring = rgba8(120, 183, 255, 204)    # #78B7FFCC
        self.play_mode = rgba8(45, 212, 191, 255)      # #2DD4BF
        self.play_tint = rgba8(45, 212, 191, 34)       # #2DD4BF22
        self.play_paused = rgba8(245, 183, 79, 255)    # #F5B74F
        self.agent_active = rgba8(167, 139, 250, 255)  # #A78BFA
        self.agent_flow = rgba8(56, 189, 248, 255)     # #38BDF8
        self.agent_soft = rgba8(167, 139, 250, 38)     # #A78BFA26
        self.status_ok = rgba8(52, 211, 153, 255)      # #34D399
        self.status_warn = rgba8(251, 191, 36, 255)    # #FBBF24
        self.status_danger = rgba8(248, 113, 113, 255) # #F87171
        self.axis_x = rgba8(229, 91, 100, 255)         # #E55B64
        self.axis_y = rgba8(82, 199, 122, 255)         # #52C77A
        self.axis_z = rgba8(78, 154, 241, 255)         # #4E9AF1

        # Text hierarchy
        self.text_primary = rgba8(237, 241, 247, 255)   # #EDF1F7
        self.text_secondary = rgba8(155, 165, 184, 255) # #9BA5B8
        self.text_muted = rgba8(147, 157, 177, 255)     # #939DB1 (>= 4.5:1 on every opaque surface, l3_hover included)
        self.text_on_accent = rgba8(11, 14, 20, 255)    # #0B0E14

        # Typography & geometry rhythm
        self.font_title = TYPE_TITLE_PX
        self.font_section = TYPE_SECTION_PX
        self.font_body = TYPE_BODY_PX
        self.font_caption = TYPE_CAPTION_PX
        self.font_mono = TYPE_MONO_PX
        self.row_height = INSPECTOR_ROW_H
        self.header_height = HEADER_BAR_H
        self.scrollbar_width = Float32(8.0)
        self.radius_xs = RADIUS_XS
        self.radius_sm = RADIUS_SM
        self.radius_md = RADIUS_MD
        self.radius_lg = RADIUS_LG
        self.space_xs = SPACE_XS
        self.space_sm = SPACE_SM
        self.space_md = SPACE_MD
        self.space_lg = SPACE_LG

        # CVD-safe 14-class dark syntax palette
        self.syntax = _make_syntax_list(
            rgb_hex(0xDCE4F2),  # text
            rgb_hex(0xA78BFA),  # keyword
            rgb_hex(0x38BDF8),  # type
            rgb_hex(0x738199),  # comment
            rgb_hex(0x6EE7B7),  # string
            rgb_hex(0xFB7185),  # number
            rgb_hex(0xFBBF24),  # decorator
            rgb_hex(0x94A3B8),  # symbol
            rgb_hex(0x67E8F9),  # builtin
            rgb_hex(0xF472B6),  # self
            rgb_hex(0xFDE047),  # def
            rgb_hex(0xE2E8F0),  # param
            rgb_hex(0x5EEAD4),  # property
            rgb_hex(0x38BDF8),  # namespace
        )
        self.syntax_keyword = self.syntax[SYN_KEYWORD]
        self.syntax_type = self.syntax[SYN_TYPE]
        self.syntax_string = self.syntax[SYN_STRING]
        self.syntax_number = self.syntax[SYN_NUMBER]
        self.syntax_comment = self.syntax[SYN_COMMENT]
        self.syntax_func = self.syntax[SYN_DEF]
        self.syntax_macro = self.syntax[SYN_DECORATOR]
        self.syntax_operator = self.syntax[SYN_SYMBOL]
        self.diff_add_bg = rgba8(52, 211, 153, 34)
        self.diff_del_bg = rgba8(248, 113, 113, 36)
        self.diff_add_fg = rgb_hex(0x6EE7B7)
        self.diff_del_fg = rgb_hex(0xFCA5A5)
        self.diff_hunk_bg = rgba8(76, 155, 255, 28)

    def same_tokens(self, other: Self) -> Bool:
        """True when every token equals `other`'s (lets `UiContext.set_studio_theme` skip a copy)."""
        if len(self.syntax) != len(other.syntax):
            return False
        for i in range(len(self.syntax)):
            if self.syntax[i] != other.syntax[i]:
                return False
        return (
            self.name == other.name
            and self.id == other.id
            and self.is_dark == other.is_dark
            and self.reduced_motion == other.reduced_motion
            and self.l0_canvas == other.l0_canvas
            and self.l1_shell == other.l1_shell
            and self.l2_card == other.l2_card
            and self.l3_raised == other.l3_raised
            and self.l3_hover == other.l3_hover
            and self.l3_press == other.l3_press
            and self.l4_overlay == other.l4_overlay
            and self.l4_scrim == other.l4_scrim
            and self.border_subtle == other.border_subtle
            and self.border_control == other.border_control
            and self.border_strong == other.border_strong
            and self.bevel_top == other.bevel_top
            and self.accent_primary == other.accent_primary
            and self.accent_soft == other.accent_soft
            and self.focus_ring == other.focus_ring
            and self.play_mode == other.play_mode
            and self.play_tint == other.play_tint
            and self.play_paused == other.play_paused
            and self.agent_active == other.agent_active
            and self.agent_flow == other.agent_flow
            and self.agent_soft == other.agent_soft
            and self.status_ok == other.status_ok
            and self.status_warn == other.status_warn
            and self.status_danger == other.status_danger
            and self.axis_x == other.axis_x
            and self.axis_y == other.axis_y
            and self.axis_z == other.axis_z
            and self.text_primary == other.text_primary
            and self.text_secondary == other.text_secondary
            and self.text_muted == other.text_muted
            and self.text_on_accent == other.text_on_accent
            and self.font_title == other.font_title
            and self.font_section == other.font_section
            and self.font_body == other.font_body
            and self.font_caption == other.font_caption
            and self.font_mono == other.font_mono
            and self.row_height == other.row_height
            and self.header_height == other.header_height
            and self.scrollbar_width == other.scrollbar_width
            and self.radius_xs == other.radius_xs
            and self.radius_sm == other.radius_sm
            and self.radius_md == other.radius_md
            and self.radius_lg == other.radius_lg
            and self.space_xs == other.space_xs
            and self.space_sm == other.space_sm
            and self.space_md == other.space_md
            and self.space_lg == other.space_lg
            and self.syntax_keyword == other.syntax_keyword
            and self.syntax_type == other.syntax_type
            and self.syntax_string == other.syntax_string
            and self.syntax_number == other.syntax_number
            and self.syntax_comment == other.syntax_comment
            and self.syntax_func == other.syntax_func
            and self.syntax_macro == other.syntax_macro
            and self.syntax_operator == other.syntax_operator
            and self.diff_add_bg == other.diff_add_bg
            and self.diff_del_bg == other.diff_del_bg
            and self.diff_add_fg == other.diff_add_fg
            and self.diff_del_fg == other.diff_del_fg
            and self.diff_hunk_bg == other.diff_hunk_bg
        )

    def _sync_syntax_fields(mut self):
        if len(self.syntax) >= SYN_COUNT:
            self.syntax_keyword = self.syntax[SYN_KEYWORD]
            self.syntax_type = self.syntax[SYN_TYPE]
            self.syntax_string = self.syntax[SYN_STRING]
            self.syntax_number = self.syntax[SYN_NUMBER]
            self.syntax_comment = self.syntax[SYN_COMMENT]
            self.syntax_func = self.syntax[SYN_DEF]
            self.syntax_macro = self.syntax[SYN_DECORATOR]
            self.syntax_operator = self.syntax[SYN_SYMBOL]

    @staticmethod
    def flint_obsidian() -> Self:
        return Self()

    @staticmethod
    def midnight_studio() -> Self:
        """Indigo-slate night preset with vibrant neon accents and CVD-safe syntax tones."""
        var t = Self()
        t.name = "Midnight Studio"
        t.id = "midnight"
        t.is_dark = True

        t.l0_canvas = rgb_hex(0x11121A)
        t.l1_shell = rgb_hex(0x16161E)
        t.l2_card = rgb_hex(0x1A1B26)
        t.l3_raised = rgb_hex(0x24283B)
        t.l3_hover = rgb_hex(0x2F354F)
        t.l3_press = rgb_hex(0x1F2335)
        t.l4_overlay = rgb_hex(0x1A1C2B, 232)
        t.l4_scrim = rgb_hex(0x0B0B10, 192)

        t.border_subtle = rgba8(192, 202, 245, 22)
        t.border_control = rgba8(192, 202, 245, 38)
        t.border_strong = rgba8(122, 162, 247, 88)
        t.bevel_top = rgba8(192, 202, 245, 22)

        t.accent_primary = rgb_hex(0x7AA2F7)
        t.accent_soft = rgb_hex(0x7AA2F7, 42)
        t.focus_ring = rgb_hex(0x7DCFFF, 210)
        t.play_mode = rgb_hex(0x73DACA)
        t.play_tint = rgb_hex(0x73DACA, 38)
        t.play_paused = rgb_hex(0xE0AF68)
        t.agent_active = rgb_hex(0xBB9AF7)
        t.agent_flow = rgb_hex(0x7DCFFF)
        t.agent_soft = rgb_hex(0xBB9AF7, 42)
        t.status_ok = rgb_hex(0x9ECE6A)
        t.status_warn = rgb_hex(0xE0AF68)
        t.status_danger = rgb_hex(0xF7768E)
        t.axis_x = rgb_hex(0xF7768E)
        t.axis_y = rgb_hex(0x9ECE6A)
        t.axis_z = rgb_hex(0x7AA2F7)

        t.text_primary = rgb_hex(0xD5DDF8)
        t.text_secondary = rgb_hex(0xA9B1D6)
        t.text_muted = rgb_hex(0x9A9FB8)  # >= 4.5:1 on every opaque surface (was #767D9F, 3.0:1 on l3_hover)
        t.text_on_accent = rgb_hex(0x16161E)

        t.syntax = _make_syntax_list(
            rgb_hex(0xC0CAF5),  # text
            rgb_hex(0xBB9AF7),  # keyword
            rgb_hex(0x2AC3DE),  # type
            rgb_hex(0x767D9F),  # comment
            rgb_hex(0x9ECE6A),  # string
            rgb_hex(0xFF7A93),  # number
            rgb_hex(0xE0AF68),  # decorator
            rgb_hex(0x89DDFF),  # symbol
            rgb_hex(0x7DCFFF),  # builtin
            rgb_hex(0xF7768E),  # self
            rgb_hex(0xF0D060),  # def
            rgb_hex(0xE0AF68),  # param
            rgb_hex(0x73DACA),  # property
            rgb_hex(0x2AC3DE),  # namespace
        )
        t._sync_syntax_fields()
        t.diff_add_bg = rgb_hex(0x9ECE6A, 34)
        t.diff_del_bg = rgb_hex(0xF7768E, 36)
        t.diff_add_fg = rgb_hex(0x9ECE6A)
        t.diff_del_fg = rgb_hex(0xF7768E)
        t.diff_hunk_bg = rgb_hex(0x7AA2F7, 30)
        return t^

    @staticmethod
    def okabe_ito() -> Self:
        """Color-vision-deficiency-safe dark theme built from the Okabe-Ito universal palette."""
        var t = Self()
        t.name = "Okabe-Ito High-Clarity"
        t.id = "okabe"
        t.is_dark = True

        t.l0_canvas = rgb_hex(0x0B0E11)
        t.l1_shell = rgb_hex(0x101418)
        t.l2_card = rgb_hex(0x171D23)
        t.l3_raised = rgb_hex(0x1F2730)
        t.l3_hover = rgb_hex(0x2A3542)
        t.l3_press = rgb_hex(0x1B222A)
        t.l4_overlay = rgb_hex(0x141A20, 232)
        t.l4_scrim = rgb_hex(0x06080A, 196)

        t.border_subtle = rgba8(230, 236, 242, 22)
        t.border_control = rgba8(230, 236, 242, 42)
        t.border_strong = rgb_hex(0x56B4E9, 110)
        t.bevel_top = rgba8(255, 255, 255, 22)

        t.accent_primary = rgb_hex(0x56B4E9)  # Sky blue
        t.accent_soft = rgb_hex(0x56B4E9, 44)
        t.focus_ring = rgb_hex(0xF0E442, 225) # High-vis yellow focus ring
        t.play_mode = rgb_hex(0x2EC08E)       # Bluish green
        t.play_tint = rgb_hex(0x2EC08E, 40)
        t.play_paused = rgb_hex(0xF0E442)     # Yellow
        t.agent_active = rgb_hex(0xE192C0)    # Reddish purple
        t.agent_flow = rgb_hex(0x56B4E9)      # Sky blue
        t.agent_soft = rgb_hex(0xE192C0, 44)
        t.status_ok = rgb_hex(0x2EC08E)
        t.status_warn = rgb_hex(0xF0E442)
        t.status_danger = rgb_hex(0xF07A45)   # Vermillion
        t.axis_x = rgb_hex(0xF07A45)          # Vermillion
        t.axis_y = rgb_hex(0x2EC08E)          # Bluish green
        t.axis_z = rgb_hex(0x56B4E9)          # Sky blue

        t.text_primary = rgb_hex(0xEEF2F6)
        t.text_secondary = rgb_hex(0xB2BCC6)
        t.text_muted = rgb_hex(0x979FA8)  # >= 4.5:1 on every opaque surface (was #7F8994, 3.5:1 on l3_hover)
        t.text_on_accent = rgb_hex(0x0B0E11)

        t.syntax = _make_syntax_list(
            rgb_hex(0xE6E6E6),  # text
            rgb_hex(0x56B4E9),  # keyword
            rgb_hex(0xF0E442),  # type
            rgb_hex(0x7F8994),  # comment
            rgb_hex(0x3CC79A),  # string
            rgb_hex(0xE69F00),  # number
            rgb_hex(0xE69F00),  # decorator
            rgb_hex(0xC8CDD2),  # symbol
            rgb_hex(0x56B4E9),  # builtin
            rgb_hex(0xE192C0),  # self
            rgb_hex(0xE192C0),  # def
            rgb_hex(0xE6E6E6),  # param
            rgb_hex(0xE6E6E6),  # property
            rgb_hex(0xF0E442),  # namespace
        )
        t._sync_syntax_fields()
        t.diff_add_bg = rgb_hex(0x2EC08E, 38)
        t.diff_del_bg = rgb_hex(0xF07A45, 40)
        t.diff_add_fg = rgb_hex(0x3CC79A)
        t.diff_del_fg = rgb_hex(0xF07A45)
        t.diff_hunk_bg = rgb_hex(0x56B4E9, 32)
        return t^

    @staticmethod
    def paper_light() -> Self:
        """Warm daylight studio theme for bright rooms."""
        var t = Self()
        t.name = "Paper Studio Light"
        t.id = "light"
        t.is_dark = False

        t.l0_canvas = rgb_hex(0xE2E4E9)
        t.l1_shell = rgb_hex(0xECEEF2)
        t.l2_card = rgb_hex(0xF8F9FB)
        t.l3_raised = rgb_hex(0xFFFFFF)
        t.l3_hover = rgb_hex(0xE7ECF5)
        t.l3_press = rgb_hex(0xDAE2F0)
        t.l4_overlay = rgb_hex(0xFFFFFF, 236)
        t.l4_scrim = rgb_hex(0x141821, 115)

        t.border_subtle = rgba8(15, 23, 42, 20)
        t.border_control = rgba8(15, 23, 42, 36)
        t.border_strong = rgba8(15, 23, 42, 62)
        t.bevel_top = rgba8(255, 255, 255, 210)

        t.accent_primary = rgb_hex(0x2563EB)
        t.accent_soft = rgb_hex(0x2563EB, 30)
        t.focus_ring = rgb_hex(0x2563EB, 185)
        t.play_mode = rgb_hex(0x0D9488)
        t.play_tint = rgb_hex(0x0D9488, 30)
        t.play_paused = rgb_hex(0xB45309)
        t.agent_active = rgb_hex(0x7C3AED)
        t.agent_flow = rgb_hex(0x0284C7)
        t.agent_soft = rgb_hex(0x7C3AED, 30)
        t.status_ok = rgb_hex(0x059669)
        t.status_warn = rgb_hex(0xB45309)
        t.status_danger = rgb_hex(0xC00060)
        t.axis_x = rgb_hex(0xD63B48)
        t.axis_y = rgb_hex(0x1E8F4E)
        t.axis_z = rgb_hex(0x2563EB)

        t.text_primary = rgb_hex(0x181B22)
        t.text_secondary = rgb_hex(0x4B5563)
        t.text_muted = rgb_hex(0x5B636F)  # >= 4.5:1 on every opaque surface (was #646D7A, 4.0:1 on l3_press)
        t.text_on_accent = rgb_hex(0xFFFFFF)

        t.syntax = _make_syntax_list(
            rgb_hex(0x1F1F1F),  # text
            rgb_hex(0x7A3E9D),  # keyword
            rgb_hex(0x00708A),  # type
            rgb_hex(0x646864),  # comment
            rgb_hex(0x3B7228),  # string
            rgb_hex(0xA74600),  # number
            rgb_hex(0x8A5700),  # decorator
            rgb_hex(0x4A4A44),  # symbol
            rgb_hex(0x00708A),  # builtin
            rgb_hex(0xB03060),  # self
            rgb_hex(0x1060C8),  # def
            rgb_hex(0x5A4FA0),  # param
            rgb_hex(0x355E8C),  # property
            rgb_hex(0x00708A),  # namespace
        )
        t._sync_syntax_fields()
        t.diff_add_bg = rgb_hex(0x059669, 26)
        t.diff_del_bg = rgb_hex(0xC00060, 24)
        t.diff_add_fg = rgb_hex(0x047857)
        t.diff_del_fg = rgb_hex(0xA60052)
        t.diff_hunk_bg = rgb_hex(0x2563EB, 22)
        return t^

    @staticmethod
    def by_id(theme_id: Int) -> Self:
        if theme_id == THEME_MIDNIGHT_STUDIO:
            return Self.midnight_studio()
        if theme_id == THEME_OKABE_ITO:
            return Self.okabe_ito()
        if theme_id == THEME_PAPER_LIGHT:
            return Self.paper_light()
        return Self.flint_obsidian()

    def syntax_color(self, token_kind: Int) -> UInt32:
        if token_kind >= 0 and token_kind < len(self.syntax):
            return self.syntax[token_kind]
        return self.text_primary

    def agent_status_color(self, status_code: Int) -> UInt32:
        """Maps `AgentStatus` codes to semantic Studio tokens."""
        if status_code == AGENT_STATE_THINKING or status_code == AGENT_STATE_PLANNING:
            return self.agent_active
        if status_code == AGENT_STATE_EDITING_SCENE:
            return self.accent_primary
        if status_code == AGENT_STATE_WRITING_CODE:
            return self.agent_flow
        if status_code == AGENT_STATE_BUILDING:
            return self.status_warn
        if status_code == AGENT_STATE_TESTING_PIE:
            return self.play_mode
        if status_code == AGENT_STATE_WAITING_APPROVAL:
            return self.play_paused
        if status_code == AGENT_STATE_ERROR:
            return self.status_danger
        return self.text_muted

    def set_role_color(mut self, role: String, col: UInt32) -> Bool:
        """Applies a high-level semantic `[roles]` override across related tokens."""
        var r = _ascii_lower(String(role.strip()))
        if r == "canvas" or r == "surface":
            self.l0_canvas = col
            return True
        if r == "shell" or r == "panel":
            self.l1_shell = col
            return True
        if r == "card":
            self.l2_card = col
            return True
        if r == "raised" or r == "control":
            self.l3_raised = col
            return True
        if r == "overlay":
            self.l4_overlay = col
            return True
        if r == "accent" or r == "cursor":
            self.accent_primary = col
            self.accent_soft = rgba8_with_alpha(col, 36)
            return True
        if r == "play" or r == "play_mode":
            self.play_mode = col
            self.play_tint = rgba8_with_alpha(col, 34)
            return True
        if r == "agent" or r == "agent_active":
            self.agent_active = col
            self.agent_soft = rgba8_with_alpha(col, 38)
            return True
        if r == "text":
            self.text_primary = col
            return True
        if r == "text_secondary":
            self.text_secondary = col
            return True
        if r == "text_muted":
            self.text_muted = col
            return True
        if r == "border":
            self.border_control = col
            return True
        if r == "diag_error" or r == "danger":
            self.status_danger = col
            return True
        if r == "diag_warn" or r == "warn":
            self.status_warn = col
            return True
        if r == "ok" or r == "success":
            self.status_ok = col
            return True
        return False

    def set_color_key(mut self, key: String, col: UInt32) -> Bool:
        """Sets an exact token field under `[colors]` (falls back to `[roles]`)."""
        var k = _ascii_lower(String(key.strip()))
        if k == "l0_canvas":
            self.l0_canvas = col
            return True
        if k == "l1_shell":
            self.l1_shell = col
            return True
        if k == "l2_card":
            self.l2_card = col
            return True
        if k == "l3_raised":
            self.l3_raised = col
            return True
        if k == "l3_hover":
            self.l3_hover = col
            return True
        if k == "l3_press":
            self.l3_press = col
            return True
        if k == "l4_overlay":
            self.l4_overlay = col
            return True
        if k == "l4_scrim":
            self.l4_scrim = col
            return True
        if k == "border_subtle":
            self.border_subtle = col
            return True
        if k == "border_control":
            self.border_control = col
            return True
        if k == "border_strong":
            self.border_strong = col
            return True
        if k == "bevel_top":
            self.bevel_top = col
            return True
        if k == "accent_primary":
            self.accent_primary = col
            return True
        if k == "accent_soft":
            self.accent_soft = col
            return True
        if k == "focus_ring":
            self.focus_ring = col
            return True
        if k == "play_mode":
            self.play_mode = col
            return True
        if k == "play_tint":
            self.play_tint = col
            return True
        if k == "play_paused":
            self.play_paused = col
            return True
        if k == "agent_active":
            self.agent_active = col
            return True
        if k == "agent_flow":
            self.agent_flow = col
            return True
        if k == "agent_soft":
            self.agent_soft = col
            return True
        if k == "status_ok":
            self.status_ok = col
            return True
        if k == "status_warn":
            self.status_warn = col
            return True
        if k == "status_danger":
            self.status_danger = col
            return True
        if k == "axis_x":
            self.axis_x = col
            return True
        if k == "axis_y":
            self.axis_y = col
            return True
        if k == "axis_z":
            self.axis_z = col
            return True
        if k == "text_primary":
            self.text_primary = col
            return True
        if k == "text_secondary":
            self.text_secondary = col
            return True
        if k == "text_muted":
            self.text_muted = col
            return True
        if k == "text_on_accent":
            self.text_on_accent = col
            return True
        if k == "diff_add_bg":
            self.diff_add_bg = col
            return True
        if k == "diff_del_bg":
            self.diff_del_bg = col
            return True
        if k == "diff_add_fg":
            self.diff_add_fg = col
            return True
        if k == "diff_del_fg":
            self.diff_del_fg = col
            return True
        if k == "diff_hunk_bg":
            self.diff_hunk_bg = col
            return True
        return self.set_role_color(k, col)

    def set_syntax_key(mut self, key: String, col: UInt32) -> Bool:
        """Sets a syntax token color under `[syntax]`."""
        var k = _ascii_lower(String(key.strip()))
        var idx = -1
        if k == "text":
            idx = SYN_TEXT
        elif k == "keyword":
            idx = SYN_KEYWORD
        elif k == "type":
            idx = SYN_TYPE
        elif k == "comment":
            idx = SYN_COMMENT
        elif k == "string":
            idx = SYN_STRING
        elif k == "number":
            idx = SYN_NUMBER
        elif k == "decorator" or k == "macro":
            idx = SYN_DECORATOR
        elif k == "symbol" or k == "operator":
            idx = SYN_SYMBOL
        elif k == "builtin":
            idx = SYN_BUILTIN
        elif k == "self":
            idx = SYN_SELF
        elif k == "def" or k == "func" or k == "function":
            idx = SYN_DEF
        elif k == "param":
            idx = SYN_PARAM
        elif k == "property":
            idx = SYN_PROPERTY
        elif k == "namespace":
            idx = SYN_NAMESPACE
        if idx < 0 or idx >= len(self.syntax):
            return False
        self.syntax[idx] = col
        self._sync_syntax_fields()
        return True

    def apply_toml_overrides(mut self, toml_text: String) -> List[String]:
        """Applies pure-Mojo TOML theme overrides (`[roles]`, `[colors]`, `[syntax]`).
        Returns a list of non-fatal diagnostic strings for unknown keys or invalid colors."""
        var problems = List[String]()
        var section = String("")
        var line_no = 0
        for raw_line in toml_text.split("\n"):
            line_no += 1
            var cleaned = _strip_toml_comment(String(raw_line))
            var line = String(cleaned.strip())
            if line.byte_length() == 0:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = _ascii_lower(String(line[byte=1 : line.byte_length() - 1].strip()))
                continue
            var eq = line.find("=")
            if eq < 0:
                problems.append("line " + String(line_no) + ": expected key = value")
                continue
            var key = String(line[byte=0:eq].strip())
            var val_raw = String(line[byte=eq + 1 :].strip())
            var val = _unquote_toml_str(val_raw)
            if section.byte_length() == 0:
                var lk = _ascii_lower(key)
                if lk == "name":
                    self.name = val
                elif lk == "id":
                    self.id = val
                elif lk == "base":
                    pass
                elif lk == "dark":
                    self.is_dark = _ascii_lower(val) == "true"
                elif lk == "reduced_motion":
                    self.reduced_motion = _ascii_lower(val) == "true"
                else:
                    problems.append("line " + String(line_no) + ": unknown header key \"" + key + "\"")
            elif section == "roles":
                var parsed = parse_hex_rgba8(val)
                if not parsed[0]:
                    problems.append("line " + String(line_no) + ": invalid hex color \"" + val + "\"")
                elif not self.set_role_color(key, parsed[1]):
                    problems.append("line " + String(line_no) + ": unknown role \"" + key + "\"")
            elif section == "colors":
                var parsed = parse_hex_rgba8(val)
                if not parsed[0]:
                    problems.append("line " + String(line_no) + ": invalid hex color \"" + val + "\"")
                elif not self.set_color_key(key, parsed[1]):
                    problems.append("line " + String(line_no) + ": unknown color \"" + key + "\"")
            elif section == "syntax":
                var parsed = parse_hex_rgba8(val)
                if not parsed[0]:
                    problems.append("line " + String(line_no) + ": invalid hex color \"" + val + "\"")
                elif not self.set_syntax_key(key, parsed[1]):
                    problems.append("line " + String(line_no) + ": unknown syntax key \"" + key + "\"")
            else:
                problems.append("line " + String(line_no) + ": unknown section [" + section + "]")
        return problems^

    @staticmethod
    def from_toml(toml_text: String, mut problems: List[String]) -> Self:
        """Builds a `StudioTheme` from TOML text, inheriting from `base = "..."` if present."""
        var base_name = String("obsidian")
        for raw_line in toml_text.split("\n"):
            var line = String(_strip_toml_comment(String(raw_line)).strip())
            if line.byte_length() == 0:
                continue
            if line.startswith("["):
                break
            var eq = line.find("=")
            if eq > 0:
                var key = _ascii_lower(String(line[byte=0:eq].strip()))
                if key == "base":
                    base_name = _unquote_toml_str(String(line[byte=eq + 1 :].strip()))
                    break
        var theme = theme_by_name(base_name)
        var errs = theme.apply_toml_overrides(toml_text)
        for e in errs:
            problems.append(e)
        return theme^


comptime UiThemeV2 = StudioTheme


@always_inline
def glass_tint(theme: StudioTheme) -> UInt32:
    """Returns the translucent frosted-glass scrim fill color for floating HUD & palette overlays."""
    return theme.l4_overlay


def theme_by_name(name: String) -> StudioTheme:
    """Resolves a built-in theme by id or display name (defaults to `Flint Obsidian`)."""
    var k = _ascii_lower(String(name.strip()))
    if k == "midnight" or k == "midnight studio" or k == "midnight_studio" or k == "neon":
        return StudioTheme.midnight_studio()
    if k == "okabe" or k == "okabe-ito" or k == "okabe_ito" or k == "okabe-ito high-clarity" or k == "cvd":
        return StudioTheme.okabe_ito()
    if k == "light" or k == "paper" or k == "paper light" or k == "paper_light" or k == "paper studio light":
        return StudioTheme.paper_light()
    return StudioTheme.flint_obsidian()


def builtin_theme_names() -> List[String]:
    var out = List[String](capacity=STUDIO_THEME_COUNT)
    out.append(String("Flint Obsidian"))
    out.append(String("Midnight Studio"))
    out.append(String("Okabe-Ito High-Clarity"))
    out.append(String("Paper Studio Light"))
    return out^


def _strip_toml_comment(line: String) -> String:
    var in_double = False
    var in_single = False
    var escaped = False
    var bytes = line.as_bytes()
    var n = len(bytes)
    for i in range(n):
        var c = Int(bytes[i])
        if escaped:
            escaped = False
            continue
        if in_double:
            if c == 92:  # '\'
                escaped = True
            elif c == 34:  # '"'
                in_double = False
        elif in_single:
            if c == 39:  # '\''
                in_single = False
        else:
            if c == 34:
                in_double = True
            elif c == 39:
                in_single = True
            elif c == 35:  # '#'
                return String(line[byte=0:i])
    return line


def _unquote_toml_str(val: String) -> String:
    var n = val.byte_length()
    if n >= 2:
        var bytes = val.as_bytes()
        var first = bytes[0]
        var last = bytes[n - 1]
        if (first == UInt8(34) and last == UInt8(34)) or (first == UInt8(39) and last == UInt8(39)):
            return String(val[byte=1 : n - 1])
    return val
