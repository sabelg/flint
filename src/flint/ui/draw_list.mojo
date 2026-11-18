"""Feature 15: Tile-Binned GPU UI Draw List & 64-byte GpuUiPrim Layout.

Provides the CPU UI command accumulator `UiDrawList` streaming compact 64-byte
primitives (`GpuUiPrim`) to the GPU for sub-0.04 ms tile-binned compute compositing.
"""
from std.collections import List
from flint.render.font import SdfFont, decode_utf8

comptime PRIM_SHAPE: UInt32 = 0
comptime PRIM_GLYPH: UInt32 = 1
comptime PRIM_NINE_SLICE: UInt32 = 2
comptime PRIM_CONTROLLER_SYMBOL: UInt32 = 3
comptime PRIM_LINE: UInt32 = 4
comptime PRIM_IMAGE: UInt32 = 5

comptime UI_FLAG_DROP_SHADOW: UInt32 = 1 << 0
comptime UI_FLAG_BORDER: UInt32 = 1 << 1
comptime UI_FLAG_CIRCLE: UInt32 = 1 << 2
comptime UI_FLAG_VERT_GRADIENT: UInt32 = 1 << 3
comptime UI_FLAG_QUAD_BEZIER: UInt32 = 1 << 4


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
        var nx0 = UInt16(max(Int(cur[0]), max(0, x0)))
        var ny0 = UInt16(max(Int(cur[1]), max(0, y0)))
        var nx1 = UInt16(min(Int(cur[2]), max(0, x1)))
        var ny1 = UInt16(min(Int(cur[3]), max(0, y1)))
        self.clip_stack.append((nx0, ny0, nx1, ny1))

    def pop_clip(mut self):
        if len(self.clip_stack) > 0:
            _ = self.clip_stack.pop()

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
        var codes = decode_utf8(text)
        var pen = x
        var c = self.current_clip()
        var scale = font_size / (font.bake_px if font.bake_px > 0.0 else 1.0)
        var sdf_to_screen = font.spread * scale / 127.0
        var aw = Float32(font.atlas_width if font.atlas_width > 0 else 1)
        var ah = Float32(font.atlas_height if font.atlas_height > 0 else 1)

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
                var hdr: UInt32 = (PRIM_GLYPH & 0xF) | (UInt32(font_slot & 0xFFF) << 4)
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
