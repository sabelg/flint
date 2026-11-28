"""Feature 15: Tile-Binned GPU UI Draw List & 64-byte GpuUiPrim Layout.

Provides the CPU UI command accumulator `UiDrawList` streaming compact 64-byte
primitives (`GpuUiPrim`) to the GPU for sub-0.04 ms tile-binned compute compositing.
"""
from std.collections import List
from std.math import floor, isfinite
from std.memory import bitcast
from flint.render.font import SdfFont, decode_utf8

comptime PRIM_SHAPE: UInt32 = 0
comptime PRIM_GLYPH: UInt32 = 1
comptime PRIM_NINE_SLICE: UInt32 = 2
comptime PRIM_CONTROLLER_SYMBOL: UInt32 = 3
comptime PRIM_LINE: UInt32 = 4
comptime PRIM_IMAGE: UInt32 = 5
comptime PRIM_POLYGON: UInt32 = 6      # header: bbox, u0 = vertex count; its data prims follow it
comptime PRIM_POLY_DATA: UInt32 = 7    # 7 (x, y) vertices in words 0..13; never binned or drawn
comptime PRIM_ARC: UInt32 = 8          # analytical SDF ring / rounded-cap arc / progress ring / spinner
comptime POLY_VERTS_PER_PRIM: Int = 7
comptime POLY_BREAK: Float32 = 3.0e38  # vertex x that separates two outlines

comptime UI_FLAG_DROP_SHADOW: UInt32 = 1 << 0
comptime UI_FLAG_BORDER: UInt32 = 1 << 1
comptime UI_FLAG_CIRCLE: UInt32 = 1 << 2
comptime UI_FLAG_VERT_GRADIENT: UInt32 = 1 << 3
comptime UI_FLAG_QUAD_BEZIER: UInt32 = 1 << 4
comptime UI_FLAG_LINEAR_TEX: UInt32 = 1 << 5
comptime UI_FLAG_MSDF: UInt32 = 1 << 6
comptime UI_FLAG_MTSDF: UInt32 = 1 << 7
comptime UI_FLAG_PER_CORNER_RADII: UInt32 = 1 << 8
comptime UI_FLAG_TOP_BEVEL: UInt32 = 1 << 9
comptime UI_FLAG_HORIZ_GRADIENT: UInt32 = 1 << 10
comptime UI_FLAG_OUTER_GLOW: UInt32 = 1 << 11
comptime UI_FLAG_GLASS_SCRIM: UInt32 = 1 << 12
comptime UI_FLAG_DASHED_FLOW: UInt32 = 1 << 13


@always_inline
def pack_flow_params(dash_px: Int, gap_px: Int, speed_q4: Int, glow_alpha: Int) -> UInt32:
    """Packs `(dash_px, gap_px, speed_q4, glow_alpha)` into `secondary_rgba8` for `UI_FLAG_DASHED_FLOW`."""
    var d = UInt32(max(2, min(255, dash_px)))
    var g = UInt32(max(0, min(255, gap_px)))
    var sp = UInt32(max(0, min(255, speed_q4)))
    var gl = UInt32(max(0, min(255, glow_alpha)))
    return d | (g << 8) | (sp << 16) | (gl << 24)


@fieldwise_init
struct GpuUiPrim(Copyable, ImplicitlyCopyable, Movable):
    """64-byte GPU UI primitive record aligned to cache lines."""
    var x0: Float32
    var y0: Float32
    var x1: Float32
    var y1: Float32
    var u0: Float32
    var v0: Float32
    var u1: Float32
    var v1: Float32
    var clip_xy0: UInt32
    var clip_xy1: UInt32
    var fill_rgba8: UInt32
    var secondary_rgba8: UInt32
    var param0: Float32   # corner_radius_px OR sdf_to_screen_px
    var param1: Float32   # outline_px / border_thickness_px
    var param2: Float32   # softness_px / nine_slice_scale
    var header: UInt32   # (kind & 0xF) | ((slot & 0xFFF) << 4) | (flags << 16)

    def __init__(out self):
        self.x0 = 0.0
        self.y0 = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.u0 = 0.0
        self.v0 = 0.0
        self.u1 = 1.0
        self.v1 = 1.0
        self.clip_xy0 = 0
        self.clip_xy1 = 0xFFFF | (0xFFFF << 16)
        self.fill_rgba8 = 0xFFFFFFFF
        self.secondary_rgba8 = 0
        self.param0 = 0.0
        self.param1 = 0.0
        self.param2 = 1.0
        self.header = 0


@always_inline
def clip_rect_i(x0: Float32, y0: Float32, x1: Float32, y1: Float32) -> Tuple[Int, Int, Int, Int]:
    """Integer clip rect that covers `[x0, x1) x [y0, y1)`: floors the min edge and ceils the max
    edge, so fractional panel edges are never cut short (a lost pixel becomes 2 at 200% HiDPI)."""
    var lim = Float32(1000000.0)
    var fx0 = max(-lim, min(lim, x0)) if isfinite(x0) else Float32(0.0)
    var fy0 = max(-lim, min(lim, y0)) if isfinite(y0) else Float32(0.0)
    var fx1 = max(-lim, min(lim, x1)) if isfinite(x1) else Float32(0.0)
    var fy1 = max(-lim, min(lim, y1)) if isfinite(y1) else Float32(0.0)
    var ix0 = Int(floor(fx0))
    var iy0 = Int(floor(fy0))
    var ix1 = Int(floor(fx1))
    var iy1 = Int(floor(fy1))
    if Float32(ix1) < fx1:
        ix1 += 1
    if Float32(iy1) < fy1:
        iy1 += 1
    return (ix0, iy0, ix1, iy1)


struct UiDrawList(Movable):
    """CPU-side command list recording vector UI primitives to stream to the GPU."""
    var prims: List[GpuUiPrim]
    var clip_stack: List[Tuple[UInt16, UInt16, UInt16, UInt16]]

    def __init__(out self):
        self.prims = List[GpuUiPrim]()
        self.clip_stack = List[Tuple[UInt16, UInt16, UInt16, UInt16]]()

    def clear(mut self):
        self.prims.clear()
        self.clip_stack.clear()

    def count(self) -> Int:
        return len(self.prims)

    def current_clip(self) -> Tuple[UInt16, UInt16, UInt16, UInt16]:
        if len(self.clip_stack) > 0:
            return self.clip_stack[len(self.clip_stack) - 1]
        return (0, 0, 0xFFFF, 0xFFFF)

    def push_clip(mut self, x0: Int, y0: Int, x1: Int, y1: Int):
        var cur = self.current_clip()
        var ix0 = max(Int(cur[0]), min(65535, max(0, x0)))
        var iy0 = max(Int(cur[1]), min(65535, max(0, y0)))
        var ix1 = min(Int(cur[2]), min(65535, max(0, x1)))
        var iy1 = min(Int(cur[3]), min(65535, max(0, y1)))
        if ix1 < ix0:
            ix1 = ix0
        if iy1 < iy0:
            iy1 = iy0
        self.clip_stack.append((UInt16(ix0), UInt16(iy0), UInt16(ix1), UInt16(iy1)))

    def push_clip_f(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32):
        """`push_clip` for fractional logical edges: covers the whole span (floor min, ceil max)."""
        var c = clip_rect_i(x0, y0, x1, y1)
        self.push_clip(c[0], c[1], c[2], c[3])

    def pop_clip(mut self):
        if len(self.clip_stack) > 0:
            _ = self.clip_stack.pop()

    @always_inline
    def _clip_is_empty(self) -> Bool:
        if len(self.clip_stack) == 0:
            return False
        var cur = self.clip_stack[len(self.clip_stack) - 1]
        return cur[0] >= cur[2] or cur[1] >= cur[3]

    def add_rect(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        corner_radius: Float32 = 0.0,
        border_width: Float32 = 0.0,
        border_color: UInt32 = 0,
    ):
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var flags: UInt32 = (UI_FLAG_BORDER if border_width > 0.0 else 0)
        var hdr: UInt32 = (PRIM_SHAPE & 0xF) | (flags << 16)
        self.prims.append(GpuUiPrim(
            x, y, x + w, y + h,
            0.0, 0.0, 1.0, 1.0,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            fill_color, border_color,
            corner_radius, border_width, 1.0, hdr
        ))

    def add_circle(
        mut self,
        cx: Float32,
        cy: Float32,
        radius: Float32,
        fill_color: UInt32,
        border_width: Float32 = 0.0,
        border_color: UInt32 = 0,
    ):
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var flags: UInt32 = UI_FLAG_CIRCLE | (UI_FLAG_BORDER if border_width > 0.0 else 0)
        var hdr: UInt32 = (PRIM_SHAPE & 0xF) | (flags << 16)
        self.prims.append(GpuUiPrim(
            cx - radius, cy - radius, cx + radius, cy + radius,
            0.0, 0.0, 1.0, 1.0,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            fill_color, border_color,
            radius, border_width, 1.0, hdr
        ))

    def add_text(
        mut self,
        font_slot: Int,
        font: SdfFont,
        text: String,
        x: Float32,
        y: Float32,
        font_size: Float32,
        color: UInt32,
        outline: Float32 = 0.0,
        outline_color: UInt32 = 0,
        shadow: Float32 = 0.0,
    ):
        if self._clip_is_empty():
            return
        var codes = decode_utf8(text)
        var pen = x
        var c = self.current_clip()
        var scale = font_size / (font.bake_px if font.bake_px > 0.0 else 1.0)
        var sdf_to_screen = font.spread * scale / 127.0
        var aw = Float32(font.atlas_width if font.atlas_width > 0 else 1)
        var ah = Float32(font.atlas_height if font.atlas_height > 0 else 1)
        var g_flags: UInt32 = (
            (UI_FLAG_MSDF if font.channels >= 3 else UInt32(0))
            | (UI_FLAG_MTSDF if font.channels >= 4 else UInt32(0))
        )

        for i in range(len(codes)):
            var g_idx = font.glyph_for(codes[i])
            if g_idx < 0 or g_idx >= len(font.glyphs):
                continue
            ref g = font.glyphs[g_idx]
            if g.w > 0 and g.h > 0:
                var gx0 = pen + g.xoff * scale
                var gy0 = y + g.yoff * scale
                var gx1 = gx0 + Float32(g.w) * scale
                var gy1 = gy0 + Float32(g.h) * scale
                var u0 = Float32(g.x) / aw
                var v0 = Float32(g.y) / ah
                var u1 = Float32(g.x + g.w) / aw
                var v1 = Float32(g.y + g.h) / ah
                var hdr: UInt32 = (PRIM_GLYPH & 0xF) | (UInt32(font_slot & 0xFFF) << 4) | (g_flags << 16)
                self.prims.append(GpuUiPrim(
                    gx0, gy0, gx1, gy1,
                    u0, v0, u1, v1,
                    UInt32(c[0]) | (UInt32(c[1]) << 16),
                    UInt32(c[2]) | (UInt32(c[3]) << 16),
                    color, outline_color,
                    sdf_to_screen, outline, shadow, hdr
                ))
            pen += g.advance * scale
            if i + 1 < len(codes):
                pen += font.kern(codes[i], codes[i + 1]) * scale

    def add_nine_slice(
        mut self,
        tex_slot: Int,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        border_left: Float32,
        border_top: Float32,
        border_right: Float32,
        border_bottom: Float32,
        tint_color: UInt32 = 0xFFFFFFFF,
    ):
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var hdr: UInt32 = (PRIM_NINE_SLICE & 0xF) | (UInt32(tex_slot & 0xFFF) << 4)
        self.prims.append(GpuUiPrim(
            x, y, x + w, y + h,
            border_left, border_top, border_right, border_bottom,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            tint_color, 0,
            0.0, 0.0, 1.0, hdr
        ))

    def add_controller_symbol(
        mut self,
        symbol_id: Int,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        secondary_color: UInt32 = 0xFF1A1E24,
        corner_radius: Float32 = 4.0,
    ):
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var hdr: UInt32 = (PRIM_CONTROLLER_SYMBOL & 0xF) | (UInt32(symbol_id & 0xFFF) << 4)
        self.prims.append(GpuUiPrim(
            x, y, x + w, y + h,
            0.0, 0.0, 1.0, 1.0,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            fill_color, secondary_color,
            corner_radius, 1.0, 1.0, hdr
        ))

    def add_rect_styled(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        secondary_color: UInt32 = 0,
        corner_radius: Float32 = 0.0,
        border_width: Float32 = 0.0,
        softness: Float32 = 1.0,
        flags: UInt32 = 0,
    ):
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var eff_flags = flags | (UI_FLAG_BORDER if border_width > 0.0 else 0)
        var hdr: UInt32 = (PRIM_SHAPE & 0xF) | (eff_flags << 16)
        self.prims.append(GpuUiPrim(
            x, y, x + w, y + h,
            0.0, 0.0, 1.0, 1.0,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            fill_color, secondary_color,
            corner_radius, border_width, softness, hdr
        ))

    def add_rounded_rect_ex(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        r_tl: Float32,
        r_tr: Float32,
        r_br: Float32,
        r_bl: Float32,
        border_width: Float32 = 0.0,
        border_color: UInt32 = 0,
        softness: Float32 = 1.0,
        flags: UInt32 = 0,
    ):
        """Emits a rounded rectangle with independent per-corner radii `(r_tl, r_tr, r_br, r_bl)`."""
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var eff_flags = flags | UI_FLAG_PER_CORNER_RADII | (UI_FLAG_BORDER if border_width > 0.0 else UInt32(0))
        var max_r = max(max(r_tl, r_tr), max(r_br, r_bl))
        var hdr: UInt32 = (PRIM_SHAPE & 0xF) | (eff_flags << 16)
        self.prims.append(GpuUiPrim(
            x, y, x + w, y + h,
            max(Float32(0.0), r_tl),
            max(Float32(0.0), r_tr),
            max(Float32(0.0), r_br),
            max(Float32(0.0), r_bl),
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            fill_color, border_color,
            max(Float32(0.0), max_r), border_width, softness, hdr
        ))

    def add_glass_panel(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        tint_rgba8: UInt32,
        corner_radius: Float32 = 8.0,
        blur_radius: Float32 = 3.0,
        border_width: Float32 = 1.0,
        border_color: UInt32 = 0x33FFFFFF,
        top_bevel: Bool = True,
        drop_shadow: Bool = True,
    ):
        """Emits a translucent frosted-glass scrim panel with 5-tap LDS Kawase blur."""
        var flags: UInt32 = UI_FLAG_GLASS_SCRIM
        if top_bevel:
            flags |= UI_FLAG_TOP_BEVEL
        if drop_shadow:
            flags |= UI_FLAG_DROP_SHADOW
        self.add_rect_styled(
            x, y, w, h,
            tint_rgba8,
            border_color,
            corner_radius,
            border_width,
            max(Float32(1.0), blur_radius),
            flags,
        )

    def add_glow_rect(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        glow_color: UInt32,
        corner_radius: Float32 = 6.0,
        glow_radius: Float32 = 8.0,
        border_width: Float32 = 0.0,
        flags: UInt32 = 0,
    ):
        """Emits a rounded rectangle with an analytical zero-offset outer glow halo."""
        self.add_rect_styled(
            x, y, w, h,
            fill_color,
            glow_color,
            corner_radius,
            border_width,
            max(Float32(1.5), glow_radius),
            flags | UI_FLAG_OUTER_GLOW,
        )

    def add_ambient_shadow(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        corner_radius: Float32 = 8.0,
        spread_px: Float32 = 14.0,
        alpha: Int = 140,
        y_offset: Float32 = 2.0,
    ):
        """Emits an analytical Gaussian dark ambient shadow behind elevated glass overlays or modals."""
        var ca = UInt32(max(0, min(255, alpha)))
        if ca == 0:
            return
        var sh_col = ca << 24  # 0xAA000000 pure black shadow
        self.add_rect_styled(
            x,
            y + y_offset,
            w,
            h,
            UInt32(0),
            sh_col,
            corner_radius,
            Float32(0.0),
            max(Float32(1.5), min(Float32(28.0), spread_px)),
            UI_FLAG_OUTER_GLOW,
        )

    def add_recessed_well(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        border_color: UInt32,
        corner_radius: Float32 = 4.0,
        border_width: Float32 = 1.0,
        focused: Bool = False,
        focus_color: UInt32 = 0,
    ):
        """Emits a physically recessed input/track well with crisp hairline rim and optional focus halo."""
        var eff_border = border_color
        if focused and (focus_color >> 24) != 0:
            var halo_a = UInt32(min(96, Int((focus_color >> 24) // 3)))
            var halo_col = (focus_color & UInt32(0x00FFFFFF)) | (halo_a << 24)
            self.add_rect_styled(
                x, y, w, h,
                UInt32(0), halo_col,
                corner_radius, Float32(0.0), Float32(4.5),
                UI_FLAG_OUTER_GLOW,
            )
            eff_border = focus_color
        self.add_rect_styled(
            x, y, w, h,
            fill_color,
            eff_border,
            corner_radius,
            border_width,
            Float32(1.0),
            UI_FLAG_BORDER if border_width > Float32(0.0) else UInt32(0),
        )

    def add_card_surface(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        fill_color: UInt32,
        border_color: UInt32,
        corner_radius: Float32 = 8.0,
        border_width: Float32 = 1.0,
        top_bevel: Bool = True,
        drop_shadow: Bool = False,
    ):
        """Emits an elevated matte card/control surface with crisp hairline border and top specular bevel."""
        var flags: UInt32 = UI_FLAG_BORDER if border_width > Float32(0.0) else UInt32(0)
        if top_bevel:
            flags |= UI_FLAG_TOP_BEVEL
        if drop_shadow:
            flags |= UI_FLAG_DROP_SHADOW
        self.add_rect_styled(
            x, y, w, h,
            fill_color,
            border_color,
            corner_radius,
            border_width,
            Float32(6.0) if drop_shadow else Float32(1.0),
            flags,
        )

    def add_accent_bar(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        color: UInt32,
        pill: Bool = True,
    ):
        """Emits a crisp rounded vertical or horizontal semantic accent indicator bar."""
        var rad = Float32(999.0) if pill else min(w, h) * Float32(0.5)
        self.add_rect(x, y, w, h, color, rad)

    def add_arc(
        mut self,
        cx: Float32,
        cy: Float32,
        radius: Float32,
        stroke_width: Float32,
        start_rad: Float32,
        sweep_rad: Float32,
        color: UInt32,
        track_color: UInt32 = 0,
        glow_softness: Float32 = 0.0,
        flags: UInt32 = 0,
    ):
        """Emits a crisp analytical SDF ring, rounded-cap arc, or animated agent spinner (`PRIM_ARC`)."""
        if self._clip_is_empty() or radius <= Float32(0.0) or stroke_width <= Float32(0.0):
            return
        var c = self.current_clip()
        var half_st = max(Float32(0.5), stroke_width * Float32(0.5))
        var eff_flags = flags
        if (track_color >> 24) != 0:
            eff_flags |= UI_FLAG_BORDER
        if glow_softness > Float32(0.01):
            eff_flags |= UI_FLAG_OUTER_GLOW
        var pad = half_st + max(Float32(1.5), min(Float32(28.0), glow_softness)) + Float32(1.0)
        var hdr: UInt32 = (PRIM_ARC & 0xF) | (eff_flags << 16)
        self.prims.append(GpuUiPrim(
            cx - radius - pad,
            cy - radius - pad,
            cx + radius + pad,
            cy + radius + pad,
            cx, cy, start_rad, sweep_rad,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            color, track_color,
            radius, stroke_width, glow_softness, hdr
        ))

    def add_progress_ring(
        mut self,
        cx: Float32,
        cy: Float32,
        radius: Float32,
        stroke_width: Float32,
        progress_01: Float32,
        arc_color: UInt32,
        track_color: UInt32 = 0x33FFFFFF,
        glow_softness: Float32 = 0.0,
    ):
        """Emits a 12-o'clock clockwise circular progress ring (`progress_01 in [0, 1]`)."""
        var p = max(Float32(0.0), min(Float32(1.0), progress_01))
        var start_rad = Float32(-1.5707963)
        var sweep_rad = p * Float32(6.2831853)
        self.add_arc(cx, cy, radius, stroke_width, start_rad, sweep_rad, arc_color, track_color, glow_softness, 0)

    def add_line(
        mut self,
        x0: Float32,
        y0: Float32,
        x1: Float32,
        y1: Float32,
        thickness: Float32,
        color: UInt32,
        softness: Float32 = 1.0,
    ):
        if self._clip_is_empty():
            return
        var half_t = max(Float32(0.5), thickness * Float32(0.5))
        var pad = half_t + max(Float32(0.5), softness)
        var bx0 = min(x0, x1) - pad
        var by0 = min(y0, y1) - pad
        var bx1 = max(x0, x1) + pad
        var by1 = max(y0, y1) + pad
        var c = self.current_clip()
        var hdr: UInt32 = (PRIM_LINE & 0xF)
        self.prims.append(GpuUiPrim(
            bx0, by0, bx1, by1,
            x0, y0, x1, y1,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            color, 0,
            half_t, 0.0, softness, hdr
        ))

    def add_line_flow(
        mut self,
        x0: Float32,
        y0: Float32,
        x1: Float32,
        y1: Float32,
        thickness: Float32,
        color: UInt32,
        dash_px: Int = 10,
        gap_px: Int = 6,
        speed_q4: Int = 24,
        glow_alpha: Int = 120,
    ):
        """Emits a GPU-animated flowing dashed line segment driven by `push_ui[6] = ui_time_sec`."""
        if self._clip_is_empty():
            return
        var half_t = max(Float32(0.5), thickness * Float32(0.5))
        var pad = half_t + max(Float32(4.0), half_t * Float32(3.0))
        var c = self.current_clip()
        var hdr: UInt32 = (PRIM_LINE & 0xF) | (UI_FLAG_DASHED_FLOW << 16)
        self.prims.append(GpuUiPrim(
            min(x0, x1) - pad, min(y0, y1) - pad,
            max(x0, x1) + pad, max(y0, y1) + pad,
            x0, y0, x1, y1,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            color, pack_flow_params(dash_px, gap_px, speed_q4, glow_alpha),
            half_t, 0.0, 1.0, hdr
        ))

    def add_quad_bezier(
        mut self,
        x0: Float32,
        y0: Float32,
        cx: Float32,
        cy: Float32,
        x1: Float32,
        y1: Float32,
        thickness: Float32,
        color: UInt32,
        softness: Float32 = 1.0,
    ):
        if self._clip_is_empty():
            return
        var min_x = min(x0, x1)
        var max_x = max(x0, x1)
        var min_y = min(y0, y1)
        var max_y = max(y0, y1)

        # Exact quadratic Bezier interior extrema: B'(t) = 0 => t = (p0 - c) / (p0 - 2c + p1)
        var dx_den = x0 - Float32(2.0) * cx + x1
        if abs(dx_den) > Float32(1.0e-6):
            var tx = (x0 - cx) / dx_den
            if tx > Float32(0.0) and tx < Float32(1.0):
                var omt = Float32(1.0) - tx
                var ex = omt * omt * x0 + Float32(2.0) * omt * tx * cx + tx * tx * x1
                min_x = min(min_x, ex)
                max_x = max(max_x, ex)

        var dy_den = y0 - Float32(2.0) * cy + y1
        if abs(dy_den) > Float32(1.0e-6):
            var ty = (y0 - cy) / dy_den
            if ty > Float32(0.0) and ty < Float32(1.0):
                var omt = Float32(1.0) - ty
                var ey = omt * omt * y0 + Float32(2.0) * omt * ty * cy + ty * ty * y1
                min_y = min(min_y, ey)
                max_y = max(max_y, ey)

        var half_t = max(Float32(0.5), thickness * Float32(0.5))
        var pad = half_t + max(Float32(0.5), softness)
        var c = self.current_clip()
        var hdr: UInt32 = (PRIM_LINE & 0xF) | (UI_FLAG_QUAD_BEZIER << 16)
        self.prims.append(GpuUiPrim(
            min_x - pad, min_y - pad, max_x + pad, max_y + pad,
            x0, y0, x1, y1,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            color, 0,
            cx, cy, half_t, hdr
        ))

    def add_bezier_flow(
        mut self,
        x0: Float32,
        y0: Float32,
        cx: Float32,
        cy: Float32,
        x1: Float32,
        y1: Float32,
        thickness: Float32,
        color: UInt32,
        dash_px: Int = 12,
        gap_px: Int = 8,
        speed_q4: Int = 32,
        glow_alpha: Int = 140,
    ):
        """Emits a GPU-animated flowing quadratic Bézier tether beam (Agent Theatre & node graphs)."""
        if self._clip_is_empty():
            return
        var min_x = min(x0, x1)
        var max_x = max(x0, x1)
        var min_y = min(y0, y1)
        var max_y = max(y0, y1)
        var dx_den = x0 - Float32(2.0) * cx + x1
        if abs(dx_den) > Float32(1.0e-6):
            var tx = (x0 - cx) / dx_den
            if tx > Float32(0.0) and tx < Float32(1.0):
                var omt = Float32(1.0) - tx
                var ex = omt * omt * x0 + Float32(2.0) * omt * tx * cx + tx * tx * x1
                min_x = min(min_x, ex)
                max_x = max(max_x, ex)
        var dy_den = y0 - Float32(2.0) * cy + y1
        if abs(dy_den) > Float32(1.0e-6):
            var ty = (y0 - cy) / dy_den
            if ty > Float32(0.0) and ty < Float32(1.0):
                var omt = Float32(1.0) - ty
                var ey = omt * omt * y0 + Float32(2.0) * omt * ty * cy + ty * ty * y1
                min_y = min(min_y, ey)
                max_y = max(max_y, ey)

        var half_t = max(Float32(0.5), thickness * Float32(0.5))
        var pad = half_t + max(Float32(4.0), half_t * Float32(3.0))
        var c = self.current_clip()
        var hdr: UInt32 = (PRIM_LINE & 0xF) | ((UI_FLAG_QUAD_BEZIER | UI_FLAG_DASHED_FLOW) << 16)
        self.prims.append(GpuUiPrim(
            min_x - pad, min_y - pad, max_x + pad, max_y + pad,
            x0, y0, x1, y1,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            color, pack_flow_params(dash_px, gap_px, speed_q4, glow_alpha),
            cx, cy, half_t, hdr
        ))

    def add_image(
        mut self,
        tex_slot: Int,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        u0: Float32 = 0.0,
        v0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v1: Float32 = 1.0,
        tint_color: UInt32 = 0xFFFFFFFF,
        corner_radius: Float32 = 0.0,
        border_width: Float32 = 0.0,
        border_color: UInt32 = 0,
    ):
        if self._clip_is_empty():
            return
        var c = self.current_clip()
        var flags: UInt32 = (UI_FLAG_BORDER if border_width > 0.0 else 0)
        var hdr: UInt32 = (PRIM_IMAGE & 0xF) | (UInt32(tex_slot & 0xFFF) << 4) | (flags << 16)
        self.prims.append(GpuUiPrim(
            x, y, x + w, y + h,
            u0, v0, u1, v1,
            UInt32(c[0]) | (UInt32(c[1]) << 16),
            UInt32(c[2]) | (UInt32(c[3]) << 16),
            tint_color, border_color,
            corner_radius, border_width, 1.0, hdr
        ))

    def add_polygon(mut self, xy: List[Float32], counts: List[Int], color: UInt32) -> Int:
        """Anti-aliased filled polygon: one or more outlines (`counts[i]` points each, flat
        `xy` = x0, y0, x1, y1, ...), filled by the nonzero winding rule, so holes (opposite
        winding) and overlapping same-winding outlines (one union, blended once) both work.

        Records one `PRIM_POLYGON` header (the bounding box) followed by its `PRIM_POLY_DATA`
        prims: the outlines closed (first point repeated) and separated by a `POLY_BREAK`
        vertex. Returns the number of prims added (0 for nothing to draw)."""
        if self._clip_is_empty():
            return 0
        var verts = List[Float32]()
        var bx0 = Float32(3.0e38)
        var by0 = Float32(3.0e38)
        var bx1 = Float32(-3.0e38)
        var by1 = Float32(-3.0e38)
        var start = 0
        var n_pts = len(xy) // 2
        for ci in range(len(counts)):
            var n = counts[ci]
            if n < 0 or start + n > n_pts:
                break
            if n >= 3:
                if len(verts) > 0:
                    verts.append(POLY_BREAK)
                    verts.append(0.0)
                for k in range(n + 1):
                    var j = start + (k % n)
                    var x = xy[2 * j]
                    var y = xy[2 * j + 1]
                    verts.append(x)
                    verts.append(y)
                    bx0 = min(bx0, x)
                    by0 = min(by0, y)
                    bx1 = max(bx1, x)
                    by1 = max(by1, y)
            start += n
        var n_verts = len(verts) // 2
        if n_verts < 4 or not (bx1 > bx0 and by1 > by0):
            return 0
        var c = self.current_clip()
        var clip0 = UInt32(c[0]) | (UInt32(c[1]) << 16)
        var clip1 = UInt32(c[2]) | (UInt32(c[3]) << 16)
        self.prims.append(GpuUiPrim(
            bx0, by0, bx1, by1,
            Float32(n_verts), 0.0, 0.0, 0.0,
            clip0, clip1,
            color, 0,
            0.0, 0.0, 0.0, PRIM_POLYGON & 0xF
        ))
        var n_data = (n_verts + POLY_VERTS_PER_PRIM - 1) // POLY_VERTS_PER_PRIM
        for d in range(n_data):
            var w = List[Float32](length=14, fill=0.0)
            for k in range(POLY_VERTS_PER_PRIM):
                var v = d * POLY_VERTS_PER_PRIM + k
                if v < n_verts:
                    w[2 * k] = verts[2 * v]
                    w[2 * k + 1] = verts[2 * v + 1]
            # Words 8..11 are integer fields in the layout: they carry vertex 4 and 5 as raw float bits.
            self.prims.append(GpuUiPrim(
                w[0], w[1], w[2], w[3],
                w[4], w[5], w[6], w[7],
                bitcast[DType.uint32](w[8]), bitcast[DType.uint32](w[9]),
                bitcast[DType.uint32](w[10]), bitcast[DType.uint32](w[11]),
                w[12], w[13], 0.0, PRIM_POLY_DATA & 0xF
            ))
        return 1 + n_data

    def add_text_mut(
        mut self,
        font_slot: Int,
        mut font: SdfFont,
        text: String,
        x: Float32,
        y: Float32,
        font_size: Float32,
        color: UInt32,
        outline: Float32 = 0.0,
        outline_color: UInt32 = 0,
        shadow: Float32 = 0.0,
    ):
        font.ensure_text(text)
        self.add_text(
            font_slot,
            font,
            text,
            x,
            y,
            font_size,
            color,
            outline,
            outline_color,
            shadow,
        )

    @staticmethod
    @always_inline
    def _scale_packed_clip(packed: UInt32, s: Float32) -> UInt32:
        var x = Int(Float32(packed & UInt32(0xFFFF)) * s + Float32(0.5))
        var y = Int(Float32((packed >> 16) & UInt32(0xFFFF)) * s + Float32(0.5))
        var cx = max(0, min(65535, x))
        var cy = max(0, min(65535, y))
        return UInt32(cx) | (UInt32(cy) << 16)

    def scale_all(mut self, scale: Float32):
        """Uniformly scales every primitive in this draw list from logical UI coords to physical pixels.
        Preserves razor-sharp GPU SDF rendering at arbitrary HiDPI factors (e.g. 1.5x, 2.0x)."""
        if not isfinite(scale) or abs(scale - Float32(1.0)) < Float32(1.0e-4):
            return
        var s = max(Float32(0.5), min(Float32(4.0), scale))
        for i in range(len(self.prims)):
            ref p = self.prims[i]
            var kind = p.header & UInt32(0xF)
            var flags = (p.header >> 16) & UInt32(0xFFFF)
            if kind == PRIM_POLY_DATA:
                # 7 (x, y) vertices packed across words 0..13; preserve POLY_BREAK sentinels (< -1e20)
                if p.x0 > Float32(-1.0e20):
                    p.x0 *= s
                    p.y0 *= s
                if p.x1 > Float32(-1.0e20):
                    p.x1 *= s
                    p.y1 *= s
                if p.u0 > Float32(-1.0e20):
                    p.u0 *= s
                    p.v0 *= s
                if p.u1 > Float32(-1.0e20):
                    p.u1 *= s
                    p.v1 *= s
                var vx4 = bitcast[DType.float32](p.clip_xy0)
                var vy4 = bitcast[DType.float32](p.clip_xy1)
                if vx4 > Float32(-1.0e20):
                    p.clip_xy0 = bitcast[DType.uint32](vx4 * s)
                    p.clip_xy1 = bitcast[DType.uint32](vy4 * s)
                var vx5 = bitcast[DType.float32](p.fill_rgba8)
                var vy5 = bitcast[DType.float32](p.secondary_rgba8)
                if vx5 > Float32(-1.0e20):
                    p.fill_rgba8 = bitcast[DType.uint32](vx5 * s)
                    p.secondary_rgba8 = bitcast[DType.uint32](vy5 * s)
                if p.param0 > Float32(-1.0e20):
                    p.param0 *= s
                    p.param1 *= s
                continue

            p.x0 *= s
            p.y0 *= s
            p.x1 *= s
            p.y1 *= s
            p.clip_xy0 = Self._scale_packed_clip(p.clip_xy0, s)
            p.clip_xy1 = Self._scale_packed_clip(p.clip_xy1, s)

            if kind == PRIM_SHAPE:
                if (flags & UI_FLAG_PER_CORNER_RADII) != 0:
                    p.u0 *= s
                    p.v0 *= s
                    p.u1 *= s
                    p.v1 *= s
                p.param0 *= s
                p.param1 *= s
                if (flags & (UI_FLAG_OUTER_GLOW | UI_FLAG_DROP_SHADOW | UI_FLAG_GLASS_SCRIM)) != 0:
                    p.param2 *= s
            elif kind == PRIM_GLYPH:
                p.param0 *= s
                p.param1 *= s
                p.param2 *= s
            elif kind == PRIM_LINE:
                p.u0 *= s
                p.v0 *= s
                p.u1 *= s
                p.v1 *= s
                if (flags & UI_FLAG_QUAD_BEZIER) != 0:
                    p.param0 *= s
                    p.param1 *= s
                    p.param2 *= s
                else:
                    p.param0 *= s
            elif kind == PRIM_ARC:
                p.u0 *= s
                p.v0 *= s
                p.param0 *= s
                p.param1 *= s
                p.param2 *= s
            elif kind == PRIM_CONTROLLER_SYMBOL or kind == PRIM_IMAGE or kind == PRIM_NINE_SLICE:
                p.param0 *= s
