"""Flint Unified 2D Depth-Sprite + 3D Mesh/Beam 8-Wide SIMD Software Canvas.

Provides perspective (`Camera3D`) and orthographic/isometric (`IsoCamera2D`)
projection into a unified float depth buffer (`SimdCanvas`), serving as:
  1. Ultra-low-footprint (`< 4 MiB`) deterministic CPU renderer & headless CI backend
  2. Direct HUD/overlay/sprite compositor companion to `flint.render.gpu_raster`
"""
from std.math import sin, cos, tan, sqrt, floor
from flint.core.math import Vec3
from flint.core.memory import MemCharge, MEM_RENDER
from flint.render.font import bitmap_glyph

comptime CLEAR_DEPTH: Float32 = -1.0e9


@always_inline
def pack_argb(r: Int, g: Int, b: Int, a: Int = 255) -> UInt32:
    var ca = UInt32(max(0, min(255, a)))
    var cr = UInt32(max(0, min(255, r)))
    var cg = UInt32(max(0, min(255, g)))
    var cb = UInt32(max(0, min(255, b)))
    return (ca << 24) | (cr << 16) | (cg << 8) | cb


@always_inline
def shade_rgb(col: UInt32, factor: Float32) -> UInt32:
    var s = max(Float32(0.0), min(Float32(2.0), factor))
    var r = Int(Float32((col >> 16) & 0xFF) * s)
    var g = Int(Float32((col >> 8) & 0xFF) * s)
    var b = Int(Float32(col & 0xFF) * s)
    return pack_argb(r, g, b, 255)


@always_inline
def _simd8_load_u32(addr: Int, offset: Int) -> SIMD[DType.uint32, 8]:
    return Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=addr + offset * 4).unsafe_load[width=8, alignment=4]()


@always_inline
def _simd8_store_u32(addr: Int, offset: Int, val: SIMD[DType.uint32, 8]):
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=addr + offset * 4).unsafe_store[alignment=4](val)


@always_inline
def _simd8_load_f32(addr: Int, offset: Int) -> SIMD[DType.float32, 8]:
    return Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=addr + offset * 4).unsafe_load[width=8, alignment=4]()


@always_inline
def _simd8_store_f32(addr: Int, offset: Int, val: SIMD[DType.float32, 8]):
    Pointer[Float32, MutAnyOrigin](unsafe_from_address=addr + offset * 4).unsafe_store[alignment=4](val)


struct Camera3D(Copyable, ImplicitlyCopyable, Movable):
    """Right-handed 3D perspective camera (Y-up) projecting to screen `(x, y, 1/z)`."""
    var eye: Vec3
    var target: Vec3
    var up: Vec3
    var fov_rad: Float32
    var near_z: Float32
    var screen_w: Int
    var screen_h: Int
    var basis_r: Vec3
    var basis_u: Vec3
    var basis_f: Vec3
    var focal_px: Float32

    def __init__(
        out self,
        screen_w: Int = 1280,
        screen_h: Int = 720,
        eye: Vec3 = Vec3(0.0, 4.5, 11.0),
        target: Vec3 = Vec3(0.0, 0.5, 0.0),
        fov_deg: Float32 = 55.0,
    ):
        self.eye = eye
        self.target = target
        self.up = Vec3(0.0, 1.0, 0.0)
        self.fov_rad = fov_deg * Float32(0.0174532925)
        self.near_z = 0.15
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.basis_r = Vec3(1.0, 0.0, 0.0)
        self.basis_u = Vec3(0.0, 1.0, 0.0)
        self.basis_f = Vec3(0.0, 0.0, -1.0)
        self.focal_px = 600.0
        self.update_basis()

    def update_basis(mut self):
        var fwd = (self.target - self.eye).normalized()
        var right = fwd.cross(self.up).normalized()
        var true_up = right.cross(fwd).normalized()
        self.basis_f = fwd
        self.basis_r = right
        self.basis_u = true_up
        var half_tan = tan(self.fov_rad * Float32(0.5))
        self.focal_px = (Float32(self.screen_h) * Float32(0.5)) / max(Float32(0.05), half_tan)

    @always_inline
    def project(self, p: Vec3) -> Tuple[Float32, Float32, Float32, Bool]:
        """Returns `(screen_x, screen_y, inv_z_depth, in_front)`."""
        var rel = p - self.eye
        var cz = rel.dot(self.basis_f)
        if cz < self.near_z:
            return (0.0, 0.0, CLEAR_DEPTH, False)
        var cx = rel.dot(self.basis_r)
        var cy = rel.dot(self.basis_u)
        var inv_z = Float32(1.0) / cz
        var sx = Float32(self.screen_w) * Float32(0.5) + cx * inv_z * self.focal_px
        var sy = Float32(self.screen_h) * Float32(0.5) - cy * inv_z * self.focal_px
        return (sx, sy, inv_z, True)


struct SimdCanvas(Movable):
    """8-wide SIMD depth-buffered ARGB8888 canvas supporting 3D geometry, 2D depth sprites & HUD."""
    var width: Int
    var height: Int
    var pixels: List[UInt32]
    var depth: List[Float32]
    var triangles_drawn: Int
    var sprites_drawn: Int
    var mem_charge: MemCharge

    def __init__(out self, width: Int = 1280, height: Int = 720):
        var w = max(64, width)
        var h = max(64, height)
        var total = w * h
        self.width = w
        self.height = h
        self.pixels = List[UInt32](length=total, fill=0xFF101622)
        self.depth = List[Float32](length=total, fill=CLEAR_DEPTH)
        self.triangles_drawn = 0
        self.sprites_drawn = 0
        self.mem_charge = MemCharge(MEM_RENDER)
        self.mem_charge.set(total * 8)

    def resize(mut self, width: Int, height: Int):
        var w = max(64, width)
        var h = max(64, height)
        if w == self.width and h == self.height:
            return
        var total = w * h
        self.width = w
        self.height = h
        self.pixels = List[UInt32](length=total, fill=0xFF101622)
        self.depth = List[Float32](length=total, fill=CLEAR_DEPTH)
        self.mem_charge.set(total * 8)

    def clear(mut self, top_rgb: UInt32 = 0x0D1726, bottom_rgb: UInt32 = 0x1B2B44):
        """Ergonomic alias for `clear_gradient_simd`."""
        self.clear_gradient_simd(top_rgb, bottom_rgb)

    def clear_gradient_simd(mut self, top_rgb: UInt32, bottom_rgb: UInt32):
        """8-wide SIMD vertical sky/backdrop gradient + depth reset."""
        self.triangles_drawn = 0
        self.sprites_drawn = 0
        var px_addr = Int(self.pixels.unsafe_ptr())
        var dp_addr = Int(self.depth.unsafe_ptr())
        var dep_vec = SIMD[DType.float32, 8](CLEAR_DEPTH)
        var r0 = Float32((top_rgb >> 16) & 0xFF)
        var g0 = Float32((top_rgb >> 8) & 0xFF)
        var b0 = Float32(top_rgb & 0xFF)
        var dr = Float32((bottom_rgb >> 16) & 0xFF) - r0
        var dg = Float32((bottom_rgb >> 8) & 0xFF) - g0
        var db = Float32(bottom_rgb & 0xFF) - b0
        var inv_h = Float32(1.0) / Float32(max(1, self.height - 1))

        for y in range(self.height):
            var t = Float32(y) * inv_h
            var col = pack_argb(Int(r0 + dr * t), Int(g0 + dg * t), Int(b0 + db * t), 255)
            var col_vec = SIMD[DType.uint32, 8](col)
            var row = y * self.width
            var vec_end = (self.width // 8) * 8
            var x = 0
            while x < vec_end:
                _simd8_store_u32(px_addr, row + x, col_vec)
                _simd8_store_f32(dp_addr, row + x, dep_vec)
                x += 8
            while x < self.width:
                self.pixels[row + x] = col
                self.depth[row + x] = CLEAR_DEPTH
                x += 1

    def draw_tri_screen(
        mut self,
        x0: Float32, y0: Float32, z0: Float32,
        x1: Float32, y1: Float32, z1: Float32,
        x2: Float32, y2: Float32, z2: Float32,
        color: UInt32,
    ):
        var min_x = max(0, Int(floor(min(x0, min(x1, x2)))))
        var max_x = min(self.width - 1, Int(floor(max(x0, max(x1, x2)))) + 1)
        var min_y = max(0, Int(floor(min(y0, min(y1, y2)))))
        var max_y = min(self.height - 1, Int(floor(max(y0, max(y1, y2)))) + 1)
        if min_x > max_x or min_y > max_y:
            return
        var area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < Float32(0.25):
            return
        self.triangles_drawn += 1
        var inv_area = Float32(1.0) / area
        var opaque = color | 0xFF000000
        for py in range(min_y, max_y + 1):
            var fy = Float32(py) + Float32(0.5)
            var row = py * self.width
            for px in range(min_x, max_x + 1):
                var fx = Float32(px) + Float32(0.5)
                var w0 = ((x1 - fx) * (y2 - fy) - (y1 - fy) * (x2 - fx)) * inv_area
                var w1 = ((x2 - fx) * (y0 - fy) - (y2 - fy) * (x0 - fx)) * inv_area
                var w2 = Float32(1.0) - w0 - w1
                if w0 >= Float32(-0.001) and w1 >= Float32(-0.001) and w2 >= Float32(-0.001):
                    var z = w0 * z0 + w1 * z1 + w2 * z2
                    var fi = row + px
                    if z >= self.depth[fi]:
                        self.pixels[fi] = opaque
                        self.depth[fi] = z

    def draw_tri_3d_lit(
        mut self,
        cam: Camera3D,
        p0: Vec3,
        p1: Vec3,
        p2: Vec3,
        base_color: UInt32,
        sun_dir: Vec3 = Vec3(0.45, 0.80, 0.38),
    ):
        var s0 = cam.project(p0)
        var s1 = cam.project(p1)
        var s2 = cam.project(p2)
        if not (s0[3] and s1[3] and s2[3]):
            return
        var nrm = (p1 - p0).cross(p2 - p0).normalized()
        var ndotl = max(Float32(0.0), abs(nrm.dot(sun_dir.normalized())))
        var lit = shade_rgb(base_color, Float32(0.28) + ndotl * Float32(0.85))
        self.draw_tri_screen(
            s0[0], s0[1], s0[2],
            s1[0], s1[1], s1[2],
            s2[0], s2[1], s2[2],
            lit,
        )

    def draw_line_3d(
        mut self,
        cam: Camera3D,
        a: Vec3,
        b: Vec3,
        color: UInt32,
        thickness_px: Int = 1,
    ):
        var sa = cam.project(a)
        var sb = cam.project(b)
        if not (sa[3] and sb[3]):
            return
        var dx = sb[0] - sa[0]
        var dy = sb[1] - sa[1]
        var steps = Int(max(abs(dx), abs(dy)))
        if steps <= 0:
            return
        var inv = Float32(1.0) / Float32(steps)
        var sx = dx * inv
        var sy = dy * inv
        var sz = (sb[2] - sa[2]) * inv
        var cx = sa[0]
        var cy = sa[1]
        var cz = sa[2]
        var rad = max(0, thickness_px - 1)
        var opaque = color | 0xFF000000
        for _ in range(steps + 1):
            var ix = Int(cx)
            var iy = Int(cy)
            for oy in range(-rad, rad + 1):
                var py = iy + oy
                if py < 0 or py >= self.height:
                    continue
                var row = py * self.width
                for ox in range(-rad, rad + 1):
                    var px = ix + ox
                    if px >= 0 and px < self.width:
                        var fi = row + px
                        if cz >= self.depth[fi]:
                            self.pixels[fi] = opaque
                            self.depth[fi] = cz
            cx += sx
            cy += sy
            cz += sz

    def blit_sprite_simd(
        mut self,
        sprite_addr: Int,
        sp_w: Int,
        sp_h: Int,
        dst_x: Int,
        dst_y: Int,
        sp_depth: Float32,
    ):
        """8-wide SIMD masked & depth-tested sprite blitter (0 = transparent)."""
        if sprite_addr == 0 or sp_w <= 0 or sp_h <= 0:
            return
        if dst_x + sp_w <= 0 or dst_x >= self.width or dst_y + sp_h <= 0 or dst_y >= self.height:
            return
        var x0 = max(0, -dst_x)
        var y0 = max(0, -dst_y)
        var x1 = min(sp_w, self.width - dst_x)
        var y1 = min(sp_h, self.height - dst_y)
        if x0 >= x1 or y0 >= y1:
            return
        self.sprites_drawn += 1
        var px_addr = Int(self.pixels.unsafe_ptr())
        var dp_addr = Int(self.depth.unsafe_ptr())
        var zero_u32 = SIMD[DType.uint32, 8](0)
        var alpha_or = SIMD[DType.uint32, 8](0xFF000000)
        var dep_vec = SIMD[DType.float32, 8](sp_depth)

        for sy in range(y0, y1):
            var fb_row = (dst_y + sy) * self.width + dst_x
            var sp_row = sy * sp_w
            var sx = x0
            var vec_end = x0 + ((x1 - x0) // 8) * 8
            while sx < vec_end:
                var src_col = _simd8_load_u32(sprite_addr, sp_row + sx)
                var non_zero = src_col.ne(zero_u32)
                if non_zero.reduce_or():
                    var fb_idx = fb_row + sx
                    var cur_dep = _simd8_load_f32(dp_addr, fb_idx)
                    var cur_col = _simd8_load_u32(px_addr, fb_idx)
                    var write_mask = non_zero & dep_vec.ge(cur_dep)
                    _simd8_store_u32(px_addr, fb_idx, write_mask.select(src_col | alpha_or, cur_col))
                    _simd8_store_f32(dp_addr, fb_idx, write_mask.select(dep_vec, cur_dep))
                sx += 8
            while sx < x1:
                var c = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=sprite_addr + (sp_row + sx) * 4)[unsafe_offset=0]
                if c != 0:
                    var fi = fb_row + sx
                    if sp_depth >= self.depth[fi]:
                        self.pixels[fi] = c | 0xFF000000
                        self.depth[fi] = sp_depth
                sx += 1

    # -----------------------------------------------------------------------
    # 2D HUD & Overlay Primitives
    # -----------------------------------------------------------------------

    def draw_rect(mut self, x: Int, y: Int, w: Int, h: Int, color: UInt32):
        var x0 = max(0, x)
        var y0 = max(0, y)
        var x1 = min(self.width, x + w)
        var y1 = min(self.height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        var px_addr = Int(self.pixels.unsafe_ptr())
        var opaque = color | 0xFF000000
        var col_vec = SIMD[DType.uint32, 8](opaque)
        for py in range(y0, y1):
            var row = py * self.width
            var px = x0
            var vec_end = x0 + ((x1 - x0) // 8) * 8
            while px < vec_end:
                _simd8_store_u32(px_addr, row + px, col_vec)
                px += 8
            while px < x1:
                self.pixels[row + px] = opaque
                px += 1

    def draw_rect_dim(mut self, x: Int, y: Int, w: Int, h: Int, tint: UInt32):
        var x0 = max(0, x)
        var y0 = max(0, y)
        var x1 = min(self.width, x + w)
        var y1 = min(self.height, y + h)
        var half_tint = (tint >> 1) & 0x007F7F7F
        for py in range(y0, y1):
            var row = py * self.width
            for px in range(x0, x1):
                var dst = (self.pixels[row + px] >> 1) & 0x007F7F7F
                self.pixels[row + px] = 0xFF000000 | (dst + half_tint)

    def draw_panel(mut self, x: Int, y: Int, w: Int, h: Int, body_col: UInt32 = 0x16202C, accent_col: UInt32 = 0xF49E38):
        self.draw_rect_dim(x, y, w, h, body_col)
        self.draw_rect(x, y, w, 2, accent_col)
        self.draw_rect(x, y + h - 2, w, 2, 0x0B1018)
        self.draw_rect(x, y, 2, h, 0x2E3F54)
        self.draw_rect(x + w - 2, y, 2, h, 0x0B1018)

    def draw_text(mut self, x: Int, y: Int, text: String, color: UInt32 = 0xF2F5F8, scale: Int = 2, shadow: Bool = True):
        if scale <= 0:
            return
        if shadow:
            self._draw_text_raw(x + 1, y + 1, text, 0xFF080B10, scale)
        self._draw_text_raw(x, y, text, color | 0xFF000000, scale)

    def _draw_text_raw(mut self, x: Int, y: Int, text: String, color: UInt32, scale: Int):
        var pen_x = x
        var bytes = text.as_bytes()
        for idx in range(text.byte_length()):
            var code = Int(bytes[idx])
            if code == 32:
                pen_x += 6 * scale
                continue
            var glyph = bitmap_glyph(code)
            for row in range(7):
                var row_bits = Int((glyph >> UInt64(row * 5)) & 0x1F)
                if row_bits == 0:
                    continue
                for col in range(5):
                    if ((row_bits >> col) & 1) != 0:
                        for sy in range(scale):
                            var py = y + row * scale + sy
                            if py < 0 or py >= self.height:
                                continue
                            var fb_row = py * self.width
                            for sx in range(scale):
                                var px = pen_x + col * scale + sx
                                if px >= 0 and px < self.width:
                                    self.pixels[fb_row + px] = color
            pen_x += 6 * scale
