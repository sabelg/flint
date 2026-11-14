"""Dependency-free CPU framebuffer, triangle rasterizer, and bitmap UI."""
from flint.core.log import LOG_GPU, log_error
from std.collections import Dict
from std.ffi import external_call
from std.math import sqrt, tan, sin, cos, floor
from flint.core.math import Vec3
from flint.core.profiler import ZONE_DEBUG_DRAW, ZONE_HUD_LAYER
from flint.render.debug_draw import DebugDraw, DebugView
from flint.render.grading import build_tone_table
from flint.render.font import SdfFont, bitmap_glyph, decode_utf8
from flint.text.rich_text import parse_rich, layout_rich, plain_text, SPAN_BUTTON, BUTTON_LABEL
from flint.ui.button_glyphs import (
    glyph_shape, glyph_label, pad_button, face_color, SHAPE_KEYCAP, SHAPE_FACE, SHAPE_PILL, SHAPE_DPAD,
    BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_DPAD_UP, BUTTON_DPAD_RIGHT, BUTTON_DPAD_DOWN,
)
from flint.ui.layout import UiFrame, ui_frame, MAX_SAFE_MARGIN
from flint.ui.nine_slice import NineSlice, slate_panel, focus_frame, slice_map
from flint.render.pause_map import PauseMap
from flint.render.minimap import MiniMap
from flint.render.lighting import LightingProfile
from flint.render.dither import DitherView, dither_view
from flint.render.gpu_layout import RECORD_FLOATS, TEXTURED_FLAG, PER_PIXEL_FLAG, DITHER_FLAG, CONSTANT_DITHER, EFFECT_BILINEAR, EFFECT_SHADOWS, CONSTANT_CAMERA, CONSTANT_KEY, CONSTANT_WORLD, CONSTANT_LIGHTS, CONSTANT_SPHERES, MAX_SHADOW_SPHERES
from flint.render.gpu_raster import BACKEND_AMDGPU, GpuRaster


comptime TEXT_SPRITE_LIMIT: Int = 512


@fieldwise_init
struct TextSprite(Copyable, Movable):
    """One finished line of HUD text: premultiplied pixels, shadow and outline included."""
    var width: Int
    var height: Int
    var pixels: List[UInt32]


def dot(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def clamp_byte(value: Float32) -> UInt8:
    return UInt8(max(0, min(255, Int(value))))


def rgb(red: UInt8, green: UInt8, blue: UInt8) -> UInt32:
    return UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)


struct RenderVertex(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var normal: Vec3
    var u: Float32
    var v: Float32
    var color: UInt32
    var light: Vec3
    var fog: Float32
    var emissive: Float32
    var occlusion: Float32

    def __init__(out self, position: Vec3 = Vec3(), normal: Vec3 = Vec3(0, 1, 0), u: Float32 = 0, v: Float32 = 0, color: UInt32 = 0xFFFFFFFF):
        self.position = position
        self.normal = normal
        self.u = u
        self.v = v
        self.color = color
        self.light = Vec3(0.90, 0.90, 0.90)
        self.fog = 0
        self.emissive = 0
        self.occlusion = -1


struct CameraVertex(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var normal: Vec3
    var u: Float32
    var v: Float32
    var color: UInt32
    var light: Vec3
    var fog: Float32
    var emissive: Float32
    var occlusion: Float32

    def __init__(out self, position: Vec3 = Vec3(), normal: Vec3 = Vec3(0, 1, 0), u: Float32 = 0, v: Float32 = 0, color: UInt32 = 0xFFFFFFFF):
        self.position = position
        self.normal = normal
        self.u = u
        self.v = v
        self.color = color
        self.light = Vec3(0.90, 0.90, 0.90)
        self.fog = 0
        self.emissive = 0
        self.occlusion = -1

    def interpolate(self, other: Self, amount: Float32) -> Self:
        var inverse = 1.0 - amount
        var result = Self(
            self.position * inverse + other.position * amount,
            self.normal * inverse + other.normal * amount,
            self.u * inverse + other.u * amount,
            self.v * inverse + other.v * amount,
            self.color,
        )
        result.light = self.light * inverse + other.light * amount
        result.fog = self.fog * inverse + other.fog * amount
        result.emissive = self.emissive * inverse + other.emissive * amount
        result.occlusion = self.occlusion * inverse + other.occlusion * amount
        return result^


struct ScreenVertex(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var inverse_z: Float32
    var normal: Vec3
    var u: Float32
    var v: Float32
    var color: UInt32
    var light: Vec3
    var fog: Float32
    var emissive: Float32
    var occlusion: Float32

    def __init__(out self):
        self.x = 0
        self.y = 0
        self.inverse_z = 1
        self.normal = Vec3(0, 1, 0)
        self.u = 0
        self.v = 0
        self.color = 0xFFFFFFFF
        self.light = Vec3(0.90, 0.90, 0.90)
        self.fog = 0
        self.emissive = 0
        self.occlusion = -1


struct SoftwareFramebuffer(Copyable, Movable):
    var width: Int
    var height: Int
    var pixels: List[UInt32]
    var depth: List[Float32]
    var camera: Vec3
    var forward: Vec3
    var right: Vec3
    var up: Vec3
    var focal: Float32
    var fov_y: Float32  # degrees, kept across resizes (package 3.2: a setting)
    var fog_color: UInt32
    var light: Vec3
    var drawn_triangles: Int
    var culled_triangles: Int
    var frame_seconds: Float32
    var fps: Int
    var last_present: Float64
    var pixels_tested: Int
    var pixels_depth_rejected: Int
    var pixels_shaded: Int
    var bloom: List[Float32]
    var bloom_temporary: List[Float32]
    var bloom_bytes: List[UInt8]
    var bloom_width: Int
    var bloom_height: Int
    var bloom_enabled: Bool
    var bloom_pixels: Int
    var tone_lut: List[UInt8]
    var post_saturation_steps: Int
    var gpu: GpuRaster
    var bloom_strength: Int
    var gpu_effects: UInt32
    var gpu_smoothing: Bool
    var allow_gpu_effects: Bool
    var drop_shadows: Bool
    var drop_smoothing: Bool
    var hud_pixels: List[UInt32]
    var hud_ops: List[Int32]
    var hud_texts: List[String]
    var hud_rectangles: List[Int32]
    var hud_hash: UInt64
    var hud_recording: Bool
    var hud_drawing: Bool
    var hud_redraws: Int
    var text_sprites: List[TextSprite]
    var text_sprite_slots: Dict[String, Int]
    var text_canvas: List[UInt32]
    var ui_scale: Float32
    var ui_x: Int
    var ui_y: Int
    var text_scale: Float32     # player text-size setting, 1.0 to 1.5
    var blur_cache: List[UInt32]  # the finished menu backdrop; the world is frozen behind menus
    var blur_cache_valid: Bool
    var pause_map: PauseMap
    var backdrop_generation: Int
    var debug: DebugDraw
    # Package 3.2: triangles drawn while `occluder_pass` is on carry
    # DITHER_FLAG and thin out inside this frame's `dither` disc.
    var occluder_pass: Bool
    var dither: DitherView
    # Package 3.12: nine-slice images (ui/nine_slice.mojo): 0 slate panel, 1 focus frame.
    var nine_slices: List[NineSlice]
    # The fonts (package 3.12: with their TrueType sources) and the UI settings.
    var display_font: SdfFont   # Lilita One: titles, banners, big numbers
    var ui_font: SdfFont        # Atkinson Hyperlegible Next SemiBold: everything else
    var safe_margin: Float32    # package 3.12: safe-area margin per side (0 to 0.1)
    var button_pad: Bool        # package 3.12: button glyphs show the controller
    var button_sony: Bool       # ... with PlayStation shapes
    var button_keys: List[String]  # package 3.15: keycap labels of remapped keys by glyph action ("" = default)
    var button_keys_generation: Int  # bumped when they change, so the recorded HUD redraws
    var minimap: MiniMap        # package 3.18: the HUD's top-down map (render/minimap.mojo); last, to keep the older fields where they were

    def __init__(out self, width: Int = 1280, height: Int = 720):
        self.width = width
        self.height = height
        self.pixels = List[UInt32](length=width * height, fill=0)
        self.depth = List[Float32](length=width * height, fill=0)
        self.camera = Vec3()
        self.forward = Vec3(0, 0, 1)
        self.right = Vec3(1, 0, 0)
        self.up = Vec3(0, 1, 0)
        self.focal = 1
        self.fov_y = 58
        self.fog_color = 0
        self.light = Vec3(-0.42, 0.82, -0.38).normalized()
        self.drawn_triangles = 0
        self.culled_triangles = 0
        self.frame_seconds = 1.0 / 60.0
        self.fps = 60
        self.last_present = 0
        self.pixels_tested = 0
        self.pixels_depth_rejected = 0
        self.pixels_shaded = 0
        self.bloom_width = 160
        self.bloom_height = 90
        self.bloom = List[Float32](length=self.bloom_width * self.bloom_height * 3, fill=0)
        self.bloom_temporary = List[Float32](length=self.bloom_width * self.bloom_height * 3, fill=0)
        self.bloom_bytes = List[UInt8](length=self.bloom_width * self.bloom_height * 3, fill=0)
        self.bloom_enabled = True
        self.bloom_pixels = 0
        self.tone_lut = List[UInt8](length=768, fill=0)
        self.post_saturation_steps = 0
        self.gpu = GpuRaster()
        self.bloom_strength = 5
        self.gpu_effects = 0
        self.gpu_smoothing = False
        self.allow_gpu_effects = True
        self.drop_shadows = False
        self.drop_smoothing = False
        self.occluder_pass = False
        self.dither = DitherView()
        self.nine_slices = List[NineSlice]()
        self.nine_slices.append(slate_panel())
        self.nine_slices.append(focus_frame())
        self.hud_pixels = List[UInt32]()
        self.hud_ops = List[Int32]()
        self.hud_texts = List[String]()
        self.hud_rectangles = List[Int32]()
        self.hud_hash = 0
        self.hud_recording = False
        self.hud_drawing = False
        self.hud_redraws = 0
        self.text_sprites = List[TextSprite]()
        self.text_sprite_slots = Dict[String, Int]()
        self.text_canvas = List[UInt32]()
        self.ui_scale = 1
        self.ui_x = 0
        self.ui_y = 0
        self.display_font = SdfFont()
        self.ui_font = SdfFont()
        self.text_scale = 1
        self.safe_margin = 0
        self.button_pad = False
        self.button_sony = False
        self.button_keys = List[String]()
        self.button_keys_generation = 0
        self.blur_cache = List[UInt32]()
        self.blur_cache_valid = False
        self.pause_map = PauseMap()
        self.backdrop_generation = 0
        self.debug = DebugDraw()
        self.minimap = MiniMap()
        _ = self.display_font.load("assets/deepsong/fonts/display.dsf")
        _ = self.ui_font.load("assets/deepsong/fonts/ui.dsf")
        # Package 3.12: the faces themselves, for the dynamic glyph cache
        # (same weights as tools/run.sh bake-fonts).
        _ = self.display_font.attach_source(DISPLAY_FONT_FILE)
        _ = self.ui_font.attach_source(UI_FONT_FILE, UI_FONT_WEIGHT)
        self.update_ui_transform()
        for index in range(256):
            self.tone_lut[index] = UInt8(index)
            self.tone_lut[index + 256] = UInt8(index)
            self.tone_lut[index + 512] = UInt8(index)
        self.set_camera(Vec3(0, 0, -5), Vec3(), 58)

    def ui_frame(self, player: Int = 0) -> UiFrame:
        """The visible canvas and its safe area (package 3.12, ui/layout.mojo)."""
        return ui_frame(self.width, self.height, self.ui_x, self.ui_y, self.ui_scale, self.safe_margin, player=player)

    def set_safe_margin(mut self, margin: Float32):
        self.safe_margin = max(Float32(0), min(MAX_SAFE_MARGIN, margin))

    def update_ui_transform(mut self):
        """Fit the 1280x720 UI canvas to the screen at any scale (1.5x at
        1080p), centred. It used to scale by whole numbers only, so 1080p
        drew the UI at 1x and the smallest text was 7 px tall."""
        self.ui_scale = max(Float32(0.25), min(Float32(self.width) / 1280.0, Float32(self.height) / 720.0))
        self.ui_x = max(0, Int((Float32(self.width) - 1280.0 * self.ui_scale) * 0.5))
        self.ui_y = max(0, Int((Float32(self.height) - 720.0 * self.ui_scale) * 0.5))

    def blur_scene(mut self, amount: Float32) -> Bool:
        """The menu backdrop. On the CPU path the frame in `pixels` is
        blurred in place. On the GPU path the frame that opens the menu also
        composes a copy without the HUD layer; the next frame reads that copy
        back once, blurs it on the CPU, and lays it into the HUD layer under
        the menu (the world is frozen behind menus, so once is enough).
        Returns False when no backdrop is ready; the caller dims instead."""
        if not self.gpu.active:
            self.blur_pixels(amount)
            return True
        if amount <= 0:
            self.blur_cache_valid = False
            self.gpu.forget_backdrop()
            return True
        if self.hud_recording:
            if not self.blur_cache_valid:
                if not self.gpu.backdrop_ready:
                    self.gpu.backdrop_wanted = True
                    return False
                var read = self.gpu.read_back_backdrop(self.pixels)
                self.stop_recording_if_gpu_lost()
                if not read:
                    return False
                self.blur_pixels(1)
                self.gpu.forget_backdrop()
            if not self.blur_cache_valid:
                return False
            self.record_hud(5, 0, 0, 0, 0, self.backdrop_generation, 0, 0, 0, 255)
        return True

    def draw_backdrop(mut self):
        """Replay of op 5: the cached backdrop, opaque, across the HUD layer."""
        if not self.blur_cache_valid or len(self.blur_cache) != len(self.hud_pixels):
            return
        self.add_hud_rectangle(0, 0, self.width, self.height)
        for index in range(len(self.blur_cache)):
            self.hud_pixels[index] = self.blur_cache[index] | UInt32(0xFF000000)

    def blur_pixels(mut self, amount: Float32):
        """Blur the finished 3D frame for a menu backdrop: average down by 4,
        blur that twice with a 5-tap box, then scale back up bilinearly,
        darkened and pulled toward a cool blue. `amount` 0..1 fades it in."""
        if amount <= 0:
            self.blur_cache_valid = False
            return
        if amount >= 1 and self.blur_cache_valid and len(self.blur_cache) == len(self.pixels):
            _ = external_call["memcpy", Int](Int(self.pixels.unsafe_ptr()), Int(self.blur_cache.unsafe_ptr()), len(self.pixels) * 4)
            return
        var factor = 4
        var small_width = max(1, self.width // factor)
        var small_height = max(1, self.height // factor)
        var small = List[Float32](length=small_width * small_height * 3, fill=0)
        var weight = 1.0 / Float32(factor * factor)
        for sy in range(small_height):
            for sx in range(small_width):
                var r: Float32 = 0
                var g: Float32 = 0
                var b: Float32 = 0
                for dy in range(factor):
                    var row = (sy * factor + dy) * self.width + sx * factor
                    for dx in range(factor):
                        var c = self.pixels[row + dx]
                        r += Float32((c >> 16) & 255)
                        g += Float32((c >> 8) & 255)
                        b += Float32(c & 255)
                var o = (sy * small_width + sx) * 3
                small[o] = r * weight
                small[o + 1] = g * weight
                small[o + 2] = b * weight
        var temp = List[Float32](length=len(small), fill=0)
        for _ in range(2):
            box_blur_pass(small, temp, small_width, small_height, True)
            box_blur_pass(temp, small, small_width, small_height, False)
        var keep = 1.0 - amount
        var dim = 1.0 - 0.38 * amount
        var tint_r = 14.0 * amount
        var tint_g = 24.0 * amount
        var tint_b = 38.0 * amount
        for y in range(self.height):
            var fy = (Float32(y) + 0.5) / Float32(factor) - 0.5
            var y0 = max(0, min(small_height - 1, Int(floor(fy))))
            var y1 = min(small_height - 1, y0 + 1)
            var ty = max(Float32(0), min(Float32(1), fy - Float32(y0)))
            for x in range(self.width):
                var fx = (Float32(x) + 0.5) / Float32(factor) - 0.5
                var x0 = max(0, min(small_width - 1, Int(floor(fx))))
                var x1 = min(small_width - 1, x0 + 1)
                var tx = max(Float32(0), min(Float32(1), fx - Float32(x0)))
                var index = y * self.width + x
                var c = self.pixels[index]
                var result = UInt32(0)
                for channel in range(3):
                    var a = small[(y0 * small_width + x0) * 3 + channel]
                    var b = small[(y0 * small_width + x1) * 3 + channel]
                    var cc = small[(y1 * small_width + x0) * 3 + channel]
                    var d = small[(y1 * small_width + x1) * 3 + channel]
                    var blurred = (a + (b - a) * tx) + ((cc + (d - cc) * tx) - (a + (b - a) * tx)) * ty
                    var shift = UInt32(16 - channel * 8)
                    var original = Float32((c >> shift) & 255)
                    var tint = tint_r if channel == 0 else (tint_g if channel == 1 else tint_b)
                    var value = (original * keep + blurred * amount) * dim + tint
                    result |= UInt32(max(0, min(255, Int(value)))) << shift
                self.pixels[index] = (c & 0xFF000000) | result
        if amount >= 1:
            self.blur_cache = self.pixels.copy()
            self.blur_cache_valid = True
            self.backdrop_generation += 1
        else:
            self.blur_cache_valid = False

    def draw_pause_map(mut self, ui_left: Int, ui_top: Int):
        """Blit the pause map with its top-left at a canvas point."""
        if self.hud_recording:
            self.record_hud(4, ui_left, ui_top, 0, 0, self.pause_map.generation, 0, 0, 0, 255)
            return
        if not self.pause_map.valid:
            return
        var x0 = self.ux(ui_left)
        var y0 = self.uy(ui_top)
        if self.hud_drawing:
            self.add_hud_rectangle(x0, y0, self.pause_map.width, self.pause_map.height)
        for y in range(self.pause_map.height):
            var sy = y0 + y
            if sy < 0 or sy >= self.height:
                continue
            for x in range(self.pause_map.width):
                var sx = x0 + x
                if sx < 0 or sx >= self.width:
                    continue
                var c = self.pause_map.pixels[y * self.pause_map.width + x]
                if (c >> 24) == 0:
                    continue
                self.blend_pixel(sy * self.width + sx, c & 0xFFFFFF, 255)

    def draw_minimap(mut self, ui_left: Int, ui_top: Int, ui_width: Int, ui_height: Int, alpha: UInt8 = 255):
        """Package 3.18: the minimap scaled (nearest texel) into a canvas
        rectangle. Recorded as HUD op 8 with the map's generation, so the
        HUD redraws when the picture changes. Columns with no floor leave
        what is under them; the rest are laid over with `alpha`, the
        premultiplied over inlined on the raw pointers (blend_pixel per
        pixel cost 3.12 1.5 ms of p99 at 3440x1440)."""
        if self.hud_recording:
            self.record_hud(8, ui_left, ui_top, ui_width, ui_height, self.minimap.generation, 0, 0, 0, alpha)
            return
        if not self.minimap.valid or alpha == 0 or self.minimap.width <= 0 or self.minimap.height <= 0:
            return
        var x0 = self.ux(ui_left)
        var y0 = self.uy(ui_top)
        var w = self.ux(ui_left + ui_width) - x0
        var h = self.uy(ui_top + ui_height) - y0
        if w <= 0 or h <= 0:
            return
        if self.hud_drawing:
            self.add_hud_rectangle(x0, y0, w, h)
        var left = max(0, x0)
        var right = min(self.width, x0 + w)
        var top = max(0, y0)
        var bottom = min(self.height, y0 + h)
        if right <= left or bottom <= top:
            return
        # Texel column of each screen column, once.
        var columns = List[Int](length=right - left, fill=0)
        for px in range(left, right):
            columns[px - left] = min(self.minimap.width - 1, (px - x0) * self.minimap.width // w)
        var amount = UInt32(alpha)
        var inverse = 255 - amount
        var texels = self.minimap.pixels.unsafe_ptr()
        var hud = self.hud_drawing
        var target = self.hud_pixels.unsafe_ptr() if hud else self.pixels.unsafe_ptr()
        for py in range(top, bottom):
            var tz = min(self.minimap.height - 1, (py - y0) * self.minimap.height // h)
            var source_row = tz * self.minimap.width
            var row = py * self.width
            for px in range(left, right):
                var c = texels[unsafe_offset=source_row + columns[px - left]]
                if (c >> 24) == 0:
                    continue
                var index = row + px
                var under = target[unsafe_offset=index]
                var r = (c >> 16) & 255
                var g = (c >> 8) & 255
                var b = c & 255
                if hud:
                    # Premultiplied over, as blend_hud_pixel does it.
                    var a = amount + ((under >> 24) * inverse) // 255
                    r = (r * amount + ((under >> 16) & 255) * inverse) // 255
                    g = (g * amount + ((under >> 8) & 255) * inverse) // 255
                    b = (b * amount + (under & 255) * inverse) // 255
                    target[unsafe_offset=index] = a << 24 | r << 16 | g << 8 | b
                else:
                    r = (r * amount + ((under >> 16) & 255) * inverse) // 255
                    g = (g * amount + ((under >> 8) & 255) * inverse) // 255
                    b = (b * amount + (under & 255) * inverse) // 255
                    target[unsafe_offset=index] = r << 16 | g << 8 | b

    def ux(self, x: Int) -> Int:
        """Screen x of a UI-canvas x."""
        return self.ui_x + Int(floor(Float32(x) * self.ui_scale + 0.5))

    def uy(self, y: Int) -> Int:
        return self.ui_y + Int(floor(Float32(y) * self.ui_scale + 0.5))

    def set_text_scale(mut self, scale: Float32):
        var clamped = max(Float32(1), min(Float32(1.5), scale))
        if clamped != self.text_scale:
            self.text_scale = clamped
            self.hud_hash = 0  # the recorded draw calls are the same; force a redraw

    def resize(mut self, width: Int, height: Int):
        if width <= 0 or height <= 0 or (width == self.width and height == self.height):
            return
        self.width = width
        self.height = height
        self.pixels = List[UInt32](length=width * height, fill=0)
        self.depth = List[Float32](length=width * height, fill=0)
        self.update_ui_transform()
        self.hud_pixels = List[UInt32]()
        self.hud_rectangles.clear()
        self.hud_hash = 0
        self.debug.written.clear()  # those pixels went with the old images
        # Both were made for the old size, and the map's pixels follow the UI scale.
        self.blur_cache_valid = False
        self.pause_map.valid = False
        var position = self.camera
        var target = self.camera + self.forward
        self.set_camera(position, target, self.fov_y)

    def clear(mut self, red: UInt8, green: UInt8, blue: UInt8):
        var color = rgb(red, green, blue)
        if self.gpu.active:
            self.hud_ops.clear()
            self.hud_texts.clear()
            self.hud_recording = True
        else:
            self.hud_recording = False
            for index in range(len(self.pixels)):
                self.pixels[index] = color
                self.depth[index] = 1000000000.0
        self.fog_color = color
        self.drawn_triangles = 0
        self.culled_triangles = 0
        self.pixels_tested = 0
        self.pixels_depth_rejected = 0
        self.pixels_shaded = 0
        self.bloom_pixels = 0
        if self.bloom_enabled:
            for index in range(len(self.bloom)):
                self.bloom[index] = 0

    def configure_post(mut self, profile: LightingProfile, quality: Int, reduced_effects: Bool):
        var desired_width = 0 if quality <= 0 or reduced_effects else (160 if quality == 1 else 240)
        var desired_height = 0 if desired_width == 0 else (90 if quality == 1 else 135)
        if desired_width > 0:
            desired_width = min(desired_width, self.width)
            desired_height = min(desired_height, self.height)
        self.bloom_enabled = desired_width > 0
        self.bloom_strength = 5 if quality == 1 else 4
        # Visual extras never reach gameplay, and low quality or reduced effects drop them.
        var extras = self.allow_gpu_effects and quality >= 1 and not reduced_effects
        self.gpu_effects = (EFFECT_BILINEAR | EFFECT_SHADOWS) if extras else UInt32(0)
        if self.drop_shadows:
            self.gpu_effects &= ~EFFECT_SHADOWS
        self.gpu_smoothing = extras and not self.drop_smoothing
        self.post_saturation_steps = Int((profile.saturation - 1.0) * 64.0) if quality > 0 else 0
        if desired_width > 0 and (desired_width != self.bloom_width or desired_height != self.bloom_height):
            self.bloom_width = desired_width
            self.bloom_height = desired_height
            self.bloom = List[Float32](length=desired_width * desired_height * 3, fill=0)
            self.bloom_temporary = List[Float32](length=desired_width * desired_height * 3, fill=0)
            self.bloom_bytes = List[UInt8](length=desired_width * desired_height * 3, fill=0)
        if not self.bloom_enabled:
            self.bloom_pixels = 0
        if profile.filmic <= 0 and profile.cdl_slope.x == 1 and profile.cdl_slope.y == 1 and profile.cdl_slope.z == 1 and profile.cdl_offset.length_squared() == 0 and profile.cdl_power.x == 1 and profile.cdl_power.y == 1 and profile.cdl_power.z == 1:
            # Ungraded profiles keep the original highlight shoulder exactly.
            for index in range(256):
                var value = Float32(index) / 255.0 * profile.exposure
                # Preserve the midtones and compress only the highlight shoulder.
                if value > 0.72:
                    value = 0.72 + (value - 0.72) / (1.0 + (value - 0.72) * 0.90)
                self.tone_lut[index] = clamp_byte(value * 255.0 * profile.grade_gain.x)
                self.tone_lut[index + 256] = clamp_byte(value * 255.0 * profile.grade_gain.y)
                self.tone_lut[index + 512] = clamp_byte(value * 255.0 * profile.grade_gain.z)
        else:
            build_tone_table(profile, self.tone_lut)

    def set_camera(mut self, position: Vec3, target: Vec3, fovy: Float32 = 58):
        self.camera = position
        self.forward = (target - position).normalized()
        if self.forward.length_squared() < 0.5:
            self.forward = Vec3(0, 0, 1)
        self.right = cross(Vec3(0, 1, 0), self.forward).normalized()
        if self.right.length_squared() < 0.5:
            self.right = Vec3(1, 0, 0)
        self.up = cross(self.forward, self.right).normalized()
        self.fov_y = fovy
        self.focal = Float32(self.height) * 0.5 / tan(fovy * 0.01745329252 * 0.5)
        self.dither = DitherView()

    def set_dither(mut self, target: Vec3, radius: Float32, margin: Float32, keep: Float32):
        """Thin occluders out around `target` this frame (render/dither.mojo);
        call after set_camera, which turns the disc off."""
        self.dither = dither_view(self.camera, self.right, self.up, self.forward, self.focal, self.width, self.height, target, radius, margin, keep)

    def upload_dither(self):
        self.gpu.write_float(CONSTANT_DITHER, self.dither.center_x)
        self.gpu.write_float(CONSTANT_DITHER + 4, self.dither.center_y)
        self.gpu.write_float(CONSTANT_DITHER + 8, self.dither.scale)
        self.gpu.write_float(CONSTANT_DITHER + 12, self.dither.offset)
        self.gpu.write_float(CONSTANT_DITHER + 16, self.dither.near_inverse_z)

    def sphere_visible(self, center: Vec3, radius: Float32) -> Bool:
        var relative = center - self.camera
        var depth = dot(relative, self.forward)
        if depth + radius < 0.06:
            return False
        var horizontal = dot(relative, self.right)
        var vertical = dot(relative, self.up)
        var horizontal_slope = Float32(self.width) * 0.5 / self.focal
        var vertical_slope = Float32(self.height) * 0.5 / self.focal
        var horizontal_margin = radius * sqrt(1.0 + horizontal_slope * horizontal_slope)
        var vertical_margin = radius * sqrt(1.0 + vertical_slope * vertical_slope)
        if horizontal - horizontal_slope * depth > horizontal_margin:
            return False
        if -horizontal - horizontal_slope * depth > horizontal_margin:
            return False
        if vertical - vertical_slope * depth > vertical_margin:
            return False
        if -vertical - vertical_slope * depth > vertical_margin:
            return False
        return True

    def camera_vertex(self, vertex: RenderVertex) -> CameraVertex:
        var relative = vertex.position - self.camera
        var result = CameraVertex(
            Vec3(dot(relative, self.right), dot(relative, self.up), dot(relative, self.forward)),
            vertex.normal,
            vertex.u,
            vertex.v,
            vertex.color,
        )
        result.light = vertex.light
        result.fog = vertex.fog
        result.emissive = vertex.emissive
        result.occlusion = vertex.occlusion
        return result^

    def project(self, vertex: CameraVertex) -> ScreenVertex:
        var result = ScreenVertex()
        var inverse_z = 1.0 / max(Float32(0.0001), vertex.position.z)
        result.x = Float32(self.width) * 0.5 + vertex.position.x * self.focal * inverse_z
        result.y = Float32(self.height) * 0.5 - vertex.position.y * self.focal * inverse_z
        result.inverse_z = inverse_z
        result.normal = vertex.normal
        result.u = vertex.u
        result.v = vertex.v
        result.color = vertex.color
        result.light = vertex.light
        result.fog = vertex.fog
        result.emissive = vertex.emissive
        result.occlusion = vertex.occlusion
        return result^

    def sample_texture(self, mut texture: List[UInt32], texture_width: Int, texture_height: Int, u: Float32, v: Float32) -> UInt32:
        if texture_width <= 0 or texture_height <= 0 or len(texture) < texture_width * texture_height:
            return 0xFFFFFFFF
        var wrapped_u = u - floor(u)
        var wrapped_v = v - floor(v)
        var x = max(0, min(texture_width - 1, Int(wrapped_u * Float32(texture_width))))
        var y = max(0, min(texture_height - 1, Int(wrapped_v * Float32(texture_height))))
        return texture[y * texture_width + x]

    def blend_hud_pixel(mut self, index: Int, color: UInt32, alpha: UInt8):
        """Premultiplied over, so the GPU can later place the layer on any scene."""
        var previous = self.hud_pixels[index]
        var inverse = 255 - Int(alpha)
        var a = Int(alpha) + (Int(previous >> 24) * inverse) // 255
        var red = (Int((color >> 16) & 255) * Int(alpha) + Int((previous >> 16) & 255) * inverse) // 255
        var green = (Int((color >> 8) & 255) * Int(alpha) + Int((previous >> 8) & 255) * inverse) // 255
        var blue = (Int(color & 255) * Int(alpha) + Int(previous & 255) * inverse) // 255
        self.hud_pixels[index] = UInt32(a) << 24 | UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)

    def blend_pixel(mut self, index: Int, color: UInt32, alpha: UInt8):
        if self.hud_drawing:
            self.blend_hud_pixel(index, color, alpha)
            return
        if alpha == 255:
            self.pixels[index] = color & 0x00FFFFFF
            return
        if alpha == 0:
            return
        var previous = self.pixels[index]
        var amount = Float32(alpha) / 255.0
        var inverse = 1.0 - amount
        var red = Float32((color >> 16) & 255) * amount + Float32((previous >> 16) & 255) * inverse
        var green = Float32((color >> 8) & 255) * amount + Float32((previous >> 8) & 255) * inverse
        var blue = Float32(color & 255) * amount + Float32(previous & 255) * inverse
        self.pixels[index] = rgb(clamp_byte(red), clamp_byte(green), clamp_byte(blue))

    def shade(self, base: UInt32, light: Vec3, fog: Float32, emissive: Float32) -> UInt32:
        var red = Float32((base >> 16) & 255) * (light.x + emissive * 1.25)
        var green = Float32((base >> 8) & 255) * (light.y + emissive * 1.25)
        var blue = Float32(base & 255) * (light.z + emissive * 1.25)
        red = red * (1.0 - fog) + Float32((self.fog_color >> 16) & 255) * fog
        green = green * (1.0 - fog) + Float32((self.fog_color >> 8) & 255) * fog
        blue = blue * (1.0 - fog) + Float32(self.fog_color & 255) * fog
        var mapped_red = Int(self.tone_lut[max(0, min(255, Int(red)))])
        var mapped_green = Int(self.tone_lut[max(0, min(255, Int(green))) + 256])
        var mapped_blue = Int(self.tone_lut[max(0, min(255, Int(blue))) + 512])
        if self.post_saturation_steps != 0:
            var luminance = (mapped_red * 54 + mapped_green * 183 + mapped_blue * 19) >> 8
            mapped_red = luminance + ((mapped_red - luminance) * (64 + self.post_saturation_steps) >> 6)
            mapped_green = luminance + ((mapped_green - luminance) * (64 + self.post_saturation_steps) >> 6)
            mapped_blue = luminance + ((mapped_blue - luminance) * (64 + self.post_saturation_steps) >> 6)
        return (base & 0xFF000000) | rgb(UInt8(max(0, min(255, mapped_red))), UInt8(max(0, min(255, mapped_green))), UInt8(max(0, min(255, mapped_blue))))

    def record_bloom(mut self, x: Int, y: Int, base: UInt32, emissive: Float32, alpha: Int):
        if not self.bloom_enabled or emissive < 0.16 or alpha < 24:
            return
        var bx = min(self.bloom_width - 1, x * self.bloom_width // self.width)
        var by = min(self.bloom_height - 1, y * self.bloom_height // self.height)
        var index = (by * self.bloom_width + bx) * 3
        var amount = emissive * (Float32(alpha) / 255.0)
        var red = Float32((base >> 16) & 255) * amount
        var green = Float32((base >> 8) & 255) * amount
        var blue = Float32(base & 255) * amount
        # Max-combine prevents overlapping emissive triangles from accumulating
        # unbounded energy in the low-resolution target.
        self.bloom[index] = max(self.bloom[index], red)
        self.bloom[index + 1] = max(self.bloom[index + 1], green)
        self.bloom[index + 2] = max(self.bloom[index + 2], blue)
        self.bloom_pixels += 1

    def blur_bloom(mut self):
        if not self.bloom_enabled or self.bloom_pixels == 0:
            return
        for y in range(self.bloom_height):
            for x in range(self.bloom_width):
                var index = (y * self.bloom_width + x) * 3
                for channel in range(3):
                    var left = max(0, x - 1)
                    var right = min(self.bloom_width - 1, x + 1)
                    self.bloom_temporary[index + channel] = self.bloom[(y * self.bloom_width + left) * 3 + channel] * 0.24 + self.bloom[index + channel] * 0.52 + self.bloom[(y * self.bloom_width + right) * 3 + channel] * 0.24
        for y in range(self.bloom_height):
            for x in range(self.bloom_width):
                var index = (y * self.bloom_width + x) * 3
                var above = max(0, y - 1)
                var below = min(self.bloom_height - 1, y + 1)
                for channel in range(3):
                    self.bloom[index + channel] = self.bloom_temporary[(above * self.bloom_width + x) * 3 + channel] * 0.24 + self.bloom_temporary[index + channel] * 0.52 + self.bloom_temporary[(below * self.bloom_width + x) * 3 + channel] * 0.24
        for index in range(len(self.bloom)):
            self.bloom_bytes[index] = UInt8(max(0, min(255, Int(self.bloom[index]))))

    def apply_post(mut self, profile: LightingProfile, quality: Int, reduced_effects: Bool):
        # Exposure, highlight compression, and grading already ran only on
        # covered scene pixels in shade(). Composite just nonempty bloom tiles;
        # HUD and overlay pixels have not been submitted at this point.
        if self.gpu.active:
            return
        self.blur_bloom()
        if not self.bloom_enabled or self.bloom_pixels == 0:
            return
        var bloom_divisor = self.bloom_strength
        for by in range(self.bloom_height):
            var minimum_y = by * self.height // self.bloom_height
            var maximum_y = min(self.height, (by + 1) * self.height // self.bloom_height)
            for bx in range(self.bloom_width):
                var bloom_index = (by * self.bloom_width + bx) * 3
                var bloom_red = Int(self.bloom_bytes[bloom_index]) // bloom_divisor
                var bloom_green = Int(self.bloom_bytes[bloom_index + 1]) // bloom_divisor
                var bloom_blue = Int(self.bloom_bytes[bloom_index + 2]) // bloom_divisor
                if bloom_red + bloom_green + bloom_blue < 2:
                    continue
                var minimum_x = bx * self.width // self.bloom_width
                var maximum_x = min(self.width, (bx + 1) * self.width // self.bloom_width)
                for y in range(minimum_y, maximum_y):
                    var index = y * self.width + minimum_x
                    for _ in range(minimum_x, maximum_x):
                        var color = self.pixels[index]
                        var red = min(255, Int((color >> 16) & 255) + bloom_red)
                        var green = min(255, Int((color >> 8) & 255) + bloom_green)
                        var blue = min(255, Int(color & 255) + bloom_blue)
                        self.pixels[index] = UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
                        index += 1

    def enable_gpu(mut self, backend: Int = BACKEND_AMDGPU) -> Bool:
        """Hand scene rasterization to the GPU. The CPU path stays in charge on failure."""
        return self.gpu.initialize(self.width, self.height, backend)

    def per_pixel_lighting(self) -> Bool:
        return self.gpu.active and self.gpu_effects != 0

    def mark_light_on_screen(mut self, light: Int, position: Vec3, radius: Float32):
        """Find the tiles a light's sphere can touch. Near the camera that is all of them."""
        var relative = position - self.camera
        var depth = dot(relative, self.forward)
        if depth < -radius:
            return
        if depth - radius < 0.25:
            self.gpu.mark_tile_lights(light, 0, 0, self.width - 1, self.height - 1)
            return
        var scale = self.focal / (depth - radius)
        var center_x = Float32(self.width) * 0.5 + dot(relative, self.right) * self.focal / depth
        var center_y = Float32(self.height) * 0.5 - dot(relative, self.up) * self.focal / depth
        var reach = radius * scale * 1.1
        self.gpu.mark_tile_lights(light, Int(max(Float32(-1), center_x - reach)), Int(max(Float32(-1), center_y - reach)), Int(min(Float32(self.width), center_x + reach)), Int(min(Float32(self.height), center_y + reach)))

    def upload_view(mut self):
        """Give the shader what it needs to rebuild each pixel's world position."""
        var base = CONSTANT_CAMERA
        var axes = [self.camera, self.right, self.up, self.forward]
        for index in range(4):
            self.gpu.write_float(base + index * 12, axes[index].x)
            self.gpu.write_float(base + index * 12 + 4, axes[index].y)
            self.gpu.write_float(base + index * 12 + 8, axes[index].z)
        self.gpu.write_float(base + 48, Float32(self.width) * 0.5)
        self.gpu.write_float(base + 52, Float32(self.height) * 0.5)
        self.gpu.write_float(base + 56, 1.0 / self.focal)

    def submit_gpu_frame(mut self) -> Bool:
        """Finish the HUD layer and queue the frame's GPU passes."""
        self.hud_recording = False
        with self.gpu.profiler.zone(ZONE_HUD_LAYER):
            self.update_hud_layer()
        self.paint_debug()
        self.upload_dither()
        if not self.gpu.submit(self.tone_lut, self.post_saturation_steps, self.fog_color, self.bloom_enabled, self.bloom_width, self.bloom_height, self.bloom_strength, self.gpu_effects if self.gpu.voxels_ready else self.gpu_effects & ~EFFECT_SHADOWS, self.gpu_smoothing):
            print("GPU raster stopped: " + self.gpu.error)
            log_error(LOG_GPU, "GPU raster stopped: " + self.gpu.error)
            return False
        return True

    def paint_debug(mut self):
        """Debug shapes go on last. The CPU's frame is redrawn from nothing each
        time; the GPU's HUD layer is not, so what the last frame painted there is
        first put back from the CPU's copy of that layer."""
        if not self.debug.has_work():
            return
        with self.gpu.profiler.zone(ZONE_DEBUG_DRAW):
            var view = DebugView(self.camera, self.right, self.up, self.forward, self.focal, self.width, self.height)
            if self.gpu.active:
                self.restore_hud_pixels()
                self.debug.paint(view, self.gpu.hud.cpu_address, self.gpu.row_pixels, 0xFF000000, True)
            else:
                self.debug.paint(view, Int(self.pixels.unsafe_ptr()), self.width, 0, False)

    def restore_hud_pixels(mut self):
        var layer = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.gpu.hud.cpu_address)
        var clean = self.hud_pixels.unsafe_ptr()
        for packed in self.debug.written:
            var x = Int(packed & 0xFFFF)
            var y = Int(packed >> 16)
            layer[unsafe_offset=y * self.gpu.row_pixels + x] = clean[unsafe_offset=y * self.width + x]
        self.debug.written.clear()

    def read_back_gpu(mut self) -> Bool:
        """Copy the newest finished frame into `pixels`. This waits for the GPU."""
        var read = self.gpu.active and self.gpu.read_back(self.pixels)
        self.stop_recording_if_gpu_lost()
        return read

    def stop_recording_if_gpu_lost(mut self):
        """With the GPU gone no frame will replay the HUD, so draw it directly."""
        if not self.gpu.active:
            self.hud_recording = False

    def record_hud(mut self, kind: Int, a: Int, b: Int, c: Int, d: Int, e: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8):
        self.hud_ops.append(Int32(kind))
        self.hud_ops.append(Int32(a))
        self.hud_ops.append(Int32(b))
        self.hud_ops.append(Int32(c))
        self.hud_ops.append(Int32(d))
        self.hud_ops.append(Int32(e))
        self.hud_ops.append(Int32(rgb(red, green, blue)))
        self.hud_ops.append(Int32(alpha))

    def recorded_hash(self) -> UInt64:
        var value = UInt64(0xCBF29CE484222325)
        for index in range(len(self.hud_ops)):
            value = (value ^ UInt64(UInt32(self.hud_ops[index]))) * UInt64(0x100000001B3)
        for text in self.hud_texts:
            for byte in text.as_bytes():
                value = (value ^ UInt64(byte)) * UInt64(0x100000001B3)
            # Above any byte, so "AB", "C" cannot hash like "A", "BC".
            value = (value ^ UInt64(0x100)) * UInt64(0x100000001B3)
        return value

    def add_hud_rectangle(mut self, x: Int, y: Int, width: Int, height: Int):
        var left = max(0, x)
        var top = max(0, y)
        var right = min(self.width, x + width)
        var bottom = min(self.height, y + height)
        if right <= left or bottom <= top:
            return
        self.hud_rectangles.append(Int32(left))
        self.hud_rectangles.append(Int32(top))
        self.hud_rectangles.append(Int32(right))
        self.hud_rectangles.append(Int32(bottom))

    def replay_hud(mut self):
        var ops = self.hud_ops.copy()
        var offset = 0
        while offset < len(ops):
            var kind = Int(ops[offset])
            var a = Int(ops[offset + 1])
            var b = Int(ops[offset + 2])
            var c = Int(ops[offset + 3])
            var d = Int(ops[offset + 4])
            var e = Int(ops[offset + 5])
            var color = UInt32(ops[offset + 6])
            var red = UInt8((color >> 16) & 255)
            var green = UInt8((color >> 8) & 255)
            var blue = UInt8(color & 255)
            var alpha = UInt8(Int(ops[offset + 7]))
            if kind == 0:
                self.draw_rectangle(a, b, c, d, red, green, blue, alpha)
            elif kind == 1:
                self.draw_rectangle_rounded(a, b, c, d, Float32(e) / 1000.0, red, green, blue, alpha)
            elif kind == 2:
                self.draw_circle(a, b, Float32(c), red, green, blue, alpha)
            elif kind == 4:
                self.draw_pause_map(a, b)
            elif kind == 8:
                self.draw_minimap(a, b, c, d, alpha)
            elif kind == 5:
                self.draw_backdrop()
            elif kind == 6:
                var rich = self.hud_texts[e]
                _ = self.draw_rich_text(rich, a, b, c, d // 4, d % 4, red, green, blue)
            elif kind == 7:
                self.draw_nine_slice(e % 64, a, b, c, d, alpha, Float32(e // 64) / 256.0)
            else:
                var text = self.hud_texts[e]
                self.draw_text(text, a, b, c, red, green, blue, d % 4, d // 4)
            offset += 8

    def update_hud_layer(mut self):
        """Redraw the premultiplied HUD only when this frame's draw calls changed."""
        var hash = self.recorded_hash()
        if hash == self.hud_hash:
            return
        self.hud_hash = hash
        self.hud_redraws += 1
        if len(self.hud_pixels) != self.width * self.height:
            self.hud_pixels = List[UInt32](length=self.width * self.height, fill=0)
        var previous = self.hud_rectangles.copy()
        self.hud_rectangles.clear()
        for index in range(0, len(previous), 4):
            for y in range(Int(previous[index + 1]), Int(previous[index + 3])):
                _ = external_call["memset", Int](Int(self.hud_pixels.unsafe_ptr()) + (y * self.width + Int(previous[index])) * 4, Int32(0), (Int(previous[index + 2]) - Int(previous[index])) * 4)
        self.hud_drawing = True
        self.replay_hud()
        self.hud_drawing = False
        var source = Int(self.hud_pixels.unsafe_ptr())
        var target = self.gpu.hud.cpu_address
        var changed = previous^
        if changed != self.hud_rectangles:
            changed.extend(self.hud_rectangles.copy())
        for index in range(0, len(changed), 4):
            var left = Int(changed[index])
            var span = (Int(changed[index + 2]) - left) * 4
            for y in range(Int(changed[index + 1]), Int(changed[index + 3])):
                _ = external_call["memcpy", Int](target + (y * self.gpu.row_pixels + left) * 4, source + (y * self.width + left) * 4, span)

    def fill_gpu_slot(self, slot: Int, a: ScreenVertex, b: ScreenVertex, c: ScreenVertex, offset: Int, texture_width: Int, texture_height: Int, alpha: UInt8, double_sided: Bool):
        """Turn one triangle into a GPU record. Touches only its own slot, so any thread may call it."""
        # The view keeps world +X on screen right, which mirrors winding: faces toward
        # the camera have a positive area here.
        var area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(area) <= 0.00001 or (not double_sided and area <= 0):
            return
        var minimum_x = max(0, Int(floor(min(a.x, min(b.x, c.x)))))
        var maximum_x = min(self.width - 1, Int(floor(max(a.x, max(b.x, c.x)))))
        var minimum_y = max(0, Int(floor(min(a.y, min(b.y, c.y)))))
        var maximum_y = min(self.height - 1, Int(floor(max(a.y, max(b.y, c.y)))))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            return
        var inverse_area = 1.0 / area
        var step_a_x = (b.y - c.y) * inverse_area
        var step_b_x = (c.y - a.y) * inverse_area
        var step_a_y = (c.x - b.x) * inverse_area
        var step_b_y = (a.x - c.x) * inverse_area
        self.gpu.store_bounds(slot, minimum_x, minimum_y, maximum_x, maximum_y)
        var address = self.gpu.record_address(slot)
        var record = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address)
        # Weights at the centre of pixel (0, 0), so shaders pass whole-number coordinates.
        var a0 = ((b.x - 0.5) * (c.y - 0.5) - (b.y - 0.5) * (c.x - 0.5)) * inverse_area
        var b0 = ((c.x - 0.5) * (a.y - 0.5) - (c.y - 0.5) * (a.x - 0.5)) * inverse_area
        record[unsafe_offset=0] = step_a_x
        record[unsafe_offset=1] = step_a_y
        record[unsafe_offset=2] = a0
        record[unsafe_offset=3] = step_b_x
        record[unsafe_offset=4] = step_b_y
        record[unsafe_offset=5] = b0

        def plane(index: Int, va: Float32, vb: Float32, vc: Float32) capturing:
            var da = va - vc
            var db = vb - vc
            record[unsafe_offset=index] = da * step_a_x + db * step_b_x
            record[unsafe_offset=index + 1] = da * step_a_y + db * step_b_y
            record[unsafe_offset=index + 2] = da * a0 + db * b0 + vc

        plane(6, a.inverse_z, b.inverse_z, c.inverse_z)
        plane(9, a.u * a.inverse_z, b.u * b.inverse_z, c.u * c.inverse_z)
        plane(12, a.v * a.inverse_z, b.v * b.inverse_z, c.v * c.inverse_z)
        plane(16, a.light.x, b.light.x, c.light.x)
        plane(19, a.light.y, b.light.y, c.light.y)
        plane(22, a.light.z, b.light.z, c.light.z)
        plane(25, a.fog, b.fog, c.fog)
        plane(28, a.emissive, b.emissive, c.emissive)
        var per_pixel = a.occlusion >= 0 and b.occlusion >= 0 and c.occlusion >= 0
        if per_pixel:
            plane(40, a.normal.x, b.normal.x, c.normal.x)
            plane(43, a.normal.y, b.normal.y, c.normal.y)
            plane(46, a.normal.z, b.normal.z, c.normal.z)
            plane(49, a.occlusion, b.occlusion, c.occlusion)
        var words = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=address)
        words[unsafe_offset=15] = a.color
        words[unsafe_offset=31] = UInt32(alpha) | (TEXTURED_FLAG if offset >= 0 else UInt32(0)) | (PER_PIXEL_FLAG if per_pixel else UInt32(0)) | (DITHER_FLAG if self.occluder_pass else UInt32(0))
        words[unsafe_offset=32] = UInt32(max(offset, 0))
        record[unsafe_offset=33] = Float32(texture_width)
        record[unsafe_offset=34] = Float32(texture_height)
        words[unsafe_offset=35] = UInt32(texture_width)
        words[unsafe_offset=36] = UInt32(max(texture_width - 1, 0))
        words[unsafe_offset=37] = UInt32(max(texture_height - 1, 0))

    def raster_triangle(mut self, a: ScreenVertex, b: ScreenVertex, c: ScreenVertex, mut texture: List[UInt32], texture_width: Int, texture_height: Int, alpha: UInt8, double_sided: Bool = False):
        if self.gpu.active:
            var slot = self.gpu.reserve_slots(1)
            if slot >= 0:
                var offset = -1
                if texture_width > 0 and texture_height > 0 and len(texture) >= texture_width * texture_height:
                    offset = self.gpu.texture_offset(Int(texture.unsafe_ptr()), texture_width, texture_height)
                self.fill_gpu_slot(slot, a, b, c, offset, texture_width, texture_height, alpha, double_sided)
            return
        # The view keeps world +X on screen right, which mirrors winding: faces toward
        # the camera have a positive area here.
        var area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(area) <= 0.00001 or (not double_sided and area <= 0):
            self.culled_triangles += 1
            return
        var minimum_x = max(0, Int(floor(min(a.x, min(b.x, c.x)))))
        var maximum_x = min(self.width - 1, Int(floor(max(a.x, max(b.x, c.x)))))
        var minimum_y = max(0, Int(floor(min(a.y, min(b.y, c.y)))))
        var maximum_y = min(self.height - 1, Int(floor(max(a.y, max(b.y, c.y)))))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            self.culled_triangles += 1
            return
        var inverse_area = 1.0 / area
        var textured = texture_width > 0 and texture_height > 0 and len(texture) >= texture_width * texture_height
        var start_x = Float32(minimum_x) + 0.5
        var start_y = Float32(minimum_y) + 0.5
        var row_a = ((b.x - start_x) * (c.y - start_y) - (b.y - start_y) * (c.x - start_x)) * inverse_area
        var row_b = ((c.x - start_x) * (a.y - start_y) - (c.y - start_y) * (a.x - start_x)) * inverse_area
        var step_a_x = (b.y - c.y) * inverse_area
        var step_b_x = (c.y - a.y) * inverse_area
        var step_a_y = (c.x - b.x) * inverse_area
        var step_b_y = (a.x - c.x) * inverse_area
        self.drawn_triangles += 1
        var dithered = self.occluder_pass and self.dither.active()
        for y in range(minimum_y, maximum_y + 1):
            var weight_a = row_a
            var weight_b = row_b
            var index = y * self.width + minimum_x
            for x in range(minimum_x, maximum_x + 1):
                var weight_c = 1.0 - weight_a - weight_b
                if weight_a >= -0.0001 and weight_b >= -0.0001 and weight_c >= -0.0001:
                    self.pixels_tested += 1
                    var inverse_z = weight_a * a.inverse_z + weight_b * b.inverse_z + weight_c * c.inverse_z
                    if inverse_z > 0:
                        var view_depth = 1.0 / inverse_z
                        if view_depth < self.depth[index] and not (dithered and self.dither.drops(x, y, inverse_z)):
                            var u = Float32(0)
                            var v = Float32(0)
                            if textured:
                                u = (weight_a * a.u * a.inverse_z + weight_b * b.u * b.inverse_z + weight_c * c.u * c.inverse_z) * view_depth
                                v = (weight_a * a.v * a.inverse_z + weight_b * b.v * b.inverse_z + weight_c * c.v * c.inverse_z) * view_depth
                            var light = a.light * weight_a + b.light * weight_b + c.light * weight_c
                            var fog = a.fog * weight_a + b.fog * weight_b + c.fog * weight_c
                            var emissive = a.emissive * weight_a + b.emissive * weight_b + c.emissive * weight_c
                            var sampled = self.sample_texture(texture, texture_width, texture_height, u, v) if textured else UInt32(0xFFFFFFFF)
                            var tint = a.color
                            var base = UInt32(255) << 24
                            var red = Int((sampled >> 16) & 255) * Int((tint >> 16) & 255) // 255
                            var green = Int((sampled >> 8) & 255) * Int((tint >> 8) & 255) // 255
                            var blue = Int(sampled & 255) * Int(tint & 255) // 255
                            base |= UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
                            var shaded = self.shade(base, light, fog, emissive)
                            var effective_alpha = Int(alpha) * Int((tint >> 24) & 255) // 255
                            effective_alpha = effective_alpha * Int((sampled >> 24) & 255) // 255
                            if effective_alpha > 0:
                                self.pixels_shaded += 1
                                self.blend_pixel(index, shaded, UInt8(effective_alpha))
                                self.record_bloom(index % self.width, index // self.width, base, emissive, effective_alpha)
                                if effective_alpha >= 128:
                                    self.depth[index] = view_depth
                        else:
                            self.pixels_depth_rejected += 1
                weight_a += step_a_x
                weight_b += step_b_x
                index += 1
            row_a += step_a_y
            row_b += step_b_y

    def draw_triangle(mut self, a: RenderVertex, b: RenderVertex, c: RenderVertex, mut texture: List[UInt32], texture_width: Int = 0, texture_height: Int = 0, alpha: UInt8 = 255, double_sided: Bool = False):
        var near = Float32(0.06)
        var camera_a = self.camera_vertex(a)
        var camera_b = self.camera_vertex(b)
        var camera_c = self.camera_vertex(c)
        if camera_a.position.z >= near and camera_b.position.z >= near and camera_c.position.z >= near:
            self.raster_triangle(self.project(camera_a), self.project(camera_b), self.project(camera_c), texture, texture_width, texture_height, alpha, double_sided)
            return
        var polygon = List[CameraVertex]()
        polygon.append(camera_a)
        polygon.append(camera_b)
        polygon.append(camera_c)
        var clipped = List[CameraVertex]()
        for index in range(len(polygon)):
            var current = polygon[index]
            var previous = polygon[(index + len(polygon) - 1) % len(polygon)]
            var current_inside = current.position.z >= near
            var previous_inside = previous.position.z >= near
            if current_inside != previous_inside:
                var amount = (near - previous.position.z) / (current.position.z - previous.position.z)
                clipped.append(previous.interpolate(current, amount))
            if current_inside:
                clipped.append(current)
        if len(clipped) < 3:
            self.culled_triangles += 1
            return
        for index in range(1, len(clipped) - 1):
            self.raster_triangle(self.project(clipped[0]), self.project(clipped[index]), self.project(clipped[index + 1]), texture, texture_width, texture_height, alpha, double_sided)

    def fill_gpu_clipped(self, first_slot: Int, a: RenderVertex, b: RenderVertex, c: RenderVertex, offset: Int, texture_width: Int, texture_height: Int, alpha: UInt8, double_sided: Bool):
        """Near-plane clip a triangle into its two reserved slots. Any thread may call it."""
        var near = Float32(0.06)
        var polygon = List[CameraVertex]()
        polygon.append(self.camera_vertex(a))
        polygon.append(self.camera_vertex(b))
        polygon.append(self.camera_vertex(c))
        var clipped = List[CameraVertex]()
        for index in range(3):
            var current = polygon[index]
            var previous = polygon[(index + 2) % 3]
            var current_inside = current.position.z >= near
            var previous_inside = previous.position.z >= near
            if current_inside != previous_inside:
                var amount = (near - previous.position.z) / (current.position.z - previous.position.z)
                clipped.append(previous.interpolate(current, amount))
            if current_inside:
                clipped.append(current)
        for index in range(1, len(clipped) - 1):
            self.fill_gpu_slot(first_slot + index - 1, self.project(clipped[0]), self.project(clipped[index]), self.project(clipped[index + 1]), offset, texture_width, texture_height, alpha, double_sided)

    def draw_line_3d(mut self, start: Vec3, end: Vec3, red: UInt8, green: UInt8, blue: UInt8):
        var a = self.project(self.camera_vertex(RenderVertex(start)))
        var b = self.project(self.camera_vertex(RenderVertex(end)))
        if self.camera_vertex(RenderVertex(start)).position.z < 0.06 or self.camera_vertex(RenderVertex(end)).position.z < 0.06:
            return
        if self.gpu.active:
            self.emit_gpu_line(a, b, rgb(red, green, blue) | UInt32(0xFF000000))
            return
        self.plot_line(a.x, a.y, b.x, b.y, rgb(red, green, blue))

    def emit_gpu_line(mut self, a: ScreenVertex, b: ScreenVertex, color: UInt32):
        """A depth-tested screen-space quad about one pixel wide per 720 lines."""
        var dx = b.x - a.x
        var dy = b.y - a.y
        var length = sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return
        var half_width = max(Float32(0.5), Float32(self.height) / 1440.0)
        var offset_x = -dy / length * half_width
        var offset_y = dx / length * half_width
        var corners = List[ScreenVertex]()
        for index in range(4):
            var corner = a.copy() if index < 2 else b.copy()
            var side = Float32(1) if index == 0 or index == 3 else Float32(-1)
            corner.x += offset_x * side
            corner.y += offset_y * side
            # A slight pull toward the camera keeps lines on surfaces from flickering.
            corner.inverse_z *= 1.002
            corner.color = color
            corner.light = Vec3(1, 1, 1)
            corner.fog = 0
            corner.emissive = 0
            corners.append(corner)
        var empty = List[UInt32]()
        self.raster_triangle(corners[0], corners[1], corners[2], empty, 0, 0, 255, True)
        self.raster_triangle(corners[0], corners[2], corners[3], empty, 0, 0, 255, True)

    def plot_line(mut self, start_x: Float32, start_y: Float32, end_x: Float32, end_y: Float32, color: UInt32):
        var steps = max(1, Int(max(abs(end_x - start_x), abs(end_y - start_y))))
        for index in range(steps + 1):
            var amount = Float32(index) / Float32(steps)
            var x = Int(start_x + (end_x - start_x) * amount)
            var y = Int(start_y + (end_y - start_y) * amount)
            if x >= 0 and y >= 0 and x < self.width and y < self.height:
                self.pixels[y * self.width + x] = color

    def draw_rectangle(mut self, x: Int, y: Int, width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        if self.hud_recording:
            self.record_hud(0, x, y, width, height, 0, red, green, blue, alpha)
            return
        if x == 0 and y == 0 and width == 1280 and height == 720:
            # Full-screen dimming overlays reach the real screen edges.
            if self.hud_drawing:
                self.add_hud_rectangle(0, 0, self.width, self.height)
            self.fill_rectangle(0, 0, self.width, self.height, red, green, blue, alpha)
            return
        var left = self.ux(x)
        var top = self.uy(y)
        var w = self.ux(x + width) - left
        var h = self.uy(y + height) - top
        if self.hud_drawing:
            self.add_hud_rectangle(left, top, w, h)
        self.fill_rectangle(left, top, w, h, red, green, blue, alpha)

    def fill_rectangle(mut self, x: Int, y: Int, width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        var color = rgb(red, green, blue)
        var left = max(0, x)
        var right = min(self.width, x + width)
        if self.hud_drawing:
            self.fill_hud_rectangle(left, max(0, y), right, min(self.height, y + height), color, alpha)
            return
        for py in range(max(0, y), min(self.height, y + height)):
            for px in range(left, right):
                self.blend_pixel(py * self.width + px, color, alpha)

    def fill_hud_rectangle(mut self, left: Int, top: Int, right: Int, bottom: Int, color: UInt32, alpha: UInt8):
        """Premultiplied over for a whole span, matching `blend_hud_pixel` exactly."""
        if alpha == 0 or right <= left:
            return
        var amount = UInt32(alpha)
        var inverse = 255 - amount
        var red = ((color >> 16) & 255) * amount
        var green = ((color >> 8) & 255) * amount
        var blue = (color & 255) * amount
        # Most of the layer is still empty when a panel lands on it.
        var over_nothing = amount << 24 | (red // 255) << 16 | (green // 255) << 8 | (blue // 255)
        var pixels = self.hud_pixels.unsafe_ptr()
        for y in range(top, bottom):
            var row = y * self.width
            for index in range(row + left, row + right):
                var under = pixels[unsafe_offset=index]
                if under == 0 or inverse == 0:
                    pixels[unsafe_offset=index] = over_nothing
                    continue
                var a = amount + ((under >> 24) * inverse) // 255
                var r = (red + ((under >> 16) & 255) * inverse) // 255
                var g = (green + ((under >> 8) & 255) * inverse) // 255
                var b = (blue + (under & 255) * inverse) // 255
                pixels[unsafe_offset=index] = a << 24 | r << 16 | g << 8 | b

    def draw_rectangle_rounded(mut self, ui_left: Int, ui_top: Int, ui_width: Int, ui_height: Int, roundness: Float32, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        if self.hud_recording:
            self.record_hud(1, ui_left, ui_top, ui_width, ui_height, Int(roundness * 1000.0 + 0.5), red, green, blue, alpha)
            return
        var x = self.ux(ui_left)
        var y = self.uy(ui_top)
        var width = self.ux(ui_left + ui_width) - x
        var height = self.uy(ui_top + ui_height) - y
        if self.hud_drawing:
            self.add_hud_rectangle(x, y, width, height)
        var radius = max(0, min(min(width, height) // 2, Int(roundness * Float32(min(width, height)))))
        self.fill_rectangle(x + radius, y, width - radius * 2, height, red, green, blue, alpha)
        self.fill_rectangle(x, y + radius, radius, height - radius * 2, red, green, blue, alpha)
        self.fill_rectangle(x + width - radius, y + radius, radius, height - radius * 2, red, green, blue, alpha)
        # Each corner is a quarter disc centred `radius` in from both edges.
        # (The right and bottom corners used to reuse the top-left distances,
        # which drew them inside out.)
        var r2 = Float32(radius) * Float32(radius)
        for py in range(radius):
            # Whole spans, not single pixels: each row of a corner is inside the disc
            # from `first` to the straight edge, and the far corners mirror it.
            var near_y = Float32(radius - py) - 0.5
            var far_y = Float32(py) + 0.5
            var first = radius
            var first_far = radius
            for px in range(radius):
                var near_x = Float32(radius - px) - 0.5
                if first == radius and near_x * near_x + near_y * near_y <= r2:
                    first = px
                if first_far == radius and near_x * near_x + far_y * far_y <= r2:
                    first_far = px
            var span = radius - first
            var span_far = radius - first_far
            self.fill_rectangle(x + first, y + py, span, 1, red, green, blue, alpha)
            self.fill_rectangle(x + width - radius, y + py, span, 1, red, green, blue, alpha)
            self.fill_rectangle(x + first_far, y + height - radius + py, span_far, 1, red, green, blue, alpha)
            self.fill_rectangle(x + width - radius, y + height - radius + py, span_far, 1, red, green, blue, alpha)

    def draw_rectangle_lines(mut self, x: Int, y: Int, width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8):
        self.draw_rectangle(x, y, width, 1, red, green, blue)
        self.draw_rectangle(x, y + height - 1, width, 1, red, green, blue)
        self.draw_rectangle(x, y, 1, height, red, green, blue)
        self.draw_rectangle(x + width - 1, y, 1, height, red, green, blue)

    def draw_circle(mut self, ui_x: Int, ui_y: Int, radius: Float32, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        if self.hud_recording:
            self.record_hud(2, ui_x, ui_y, Int(radius), 0, 0, red, green, blue, alpha)
            return
        var x = self.ux(ui_x)
        var y = self.uy(ui_y)
        var r = max(1, Int(radius * self.ui_scale + 0.5))
        if self.hud_drawing:
            self.add_hud_rectangle(x - r, y - r, r * 2 + 1, r * 2 + 1)
        var color = rgb(red, green, blue)
        for py in range(-r, r + 1):
            for px in range(-r, r + 1):
                if px * px + py * py > r * r:
                    continue
                var sx = x + px
                var sy = y + py
                if sx >= 0 and sy >= 0 and sx < self.width and sy < self.height:
                    self.blend_pixel(sy * self.width + sx, color, alpha)

    def draw_nine_slice(mut self, slice_index: Int, ui_left: Int, ui_top: Int, ui_width: Int, ui_height: Int, alpha: UInt8 = 255, corner_scale: Float32 = 1):
        """Package 3.12: a nine-slice image (ui/nine_slice.mojo) stretched over
        a canvas rectangle. Corners keep their size times the UI scale (and
        `corner_scale`), edges stretch along one axis, the centre along both."""
        if slice_index < 0 or slice_index >= len(self.nine_slices) or alpha == 0:
            return
        if self.hud_recording:
            self.record_hud(7, ui_left, ui_top, ui_width, ui_height, slice_index + 64 * Int(corner_scale * 256.0 + 0.5), 0, 0, 0, alpha)
            return
        var x = self.ux(ui_left)
        var y = self.uy(ui_top)
        var width = self.ux(ui_left + ui_width) - x
        var height = self.uy(ui_top + ui_height) - y
        if width <= 0 or height <= 0:
            return
        if self.hud_drawing:
            self.add_hud_rectangle(x, y, width, height)
        var scale = self.ui_scale * corner_scale
        var columns = slice_map(width, self.nine_slices[slice_index].width, self.nine_slices[slice_index].left, self.nine_slices[slice_index].right, scale)
        var rows = slice_map(height, self.nine_slices[slice_index].height, self.nine_slices[slice_index].top, self.nine_slices[slice_index].bottom, scale)
        var source_width = self.nine_slices[slice_index].width
        # A copy (48x48 at most): blend_pixel takes `mut self`.
        var image = self.nine_slices[slice_index].pixels.copy()
        var source = image.unsafe_ptr()
        var amount = UInt32(alpha)
        if self.hud_drawing:
            # The HUD layer: blend_hud_pixel's premultiplied over, inlined
            # on raw rows (the same integers, so the same pixels).
            var target = self.hud_pixels.unsafe_ptr()
            var column_ptr = columns.unsafe_ptr()
            for py in range(max(0, -y), min(height, self.height - y)):
                var row = rows[py] * source_width
                var target_row = (y + py) * self.width + x
                for px in range(max(0, -x), min(width, self.width - x)):
                    var texel = source[unsafe_offset=row + column_ptr[unsafe_offset=px]]
                    var coverage = Int((texel >> 24) * amount // 255)
                    if coverage == 0:
                        continue
                    var previous = target[unsafe_offset=target_row + px]
                    var inverse = 255 - coverage
                    var a = coverage + (Int(previous >> 24) * inverse) // 255
                    var red = (Int((texel >> 16) & 255) * coverage + Int((previous >> 16) & 255) * inverse) // 255
                    var green = (Int((texel >> 8) & 255) * coverage + Int((previous >> 8) & 255) * inverse) // 255
                    var blue = (Int(texel & 255) * coverage + Int(previous & 255) * inverse) // 255
                    target[unsafe_offset=target_row + px] = UInt32(a) << 24 | UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
            _ = len(image)
            _ = len(columns)
            return
        for py in range(max(0, -y), min(height, self.height - y)):
            var row = rows[py] * source_width
            var target_row = (y + py) * self.width
            for px in range(max(0, -x), min(width, self.width - x)):
                var texel = source[unsafe_offset=row + columns[px]]
                var coverage = (texel >> 24) * amount // 255
                if coverage == 0:
                    continue
                self.blend_pixel(target_row + x + px, texel & 0xFFFFFF, UInt8(coverage))
        _ = len(image)  # keep the copy alive past the raw pointer's last use

    def draw_text(mut self, text: String, ui_left: Int, ui_top: Int, size: Int, red: UInt8 = 240, green: UInt8 = 240, blue: UInt8 = 235, align: Int = 0, fit: Int = 0):
        """Text on the UI canvas. `ui_top` is the top of the capitals; `size`
        is pixels per em on the canvas. align: 0 left, 1 centre, 2 right
        (of `ui_left`). `fit` > 0 is the room on the canvas: wider text
        shrinks to fit, down to 80% of its base size. Sizes of 30 and up use
        the display face."""
        if self.hud_recording:
            self.hud_texts.append(text)
            self.record_hud(3, ui_left, ui_top, size, align + fit * 4, len(self.hud_texts) - 1, red, green, blue, 255)
            return
        var font_index = 1 if size >= 30 else 0
        if not self.font_ready(font_index):
            self.draw_bitmap_text(text, ui_left, ui_top, size, red, green, blue, align)
            return
        var px = text_em(size) * self.text_scale * self.ui_scale
        var width = self.measure_font(font_index, text, px)
        if fit > 0 and width > Float32(fit) * self.ui_scale:
            var floor_px = text_em(size) * self.ui_scale * 0.8
            px = max(floor_px, px * Float32(fit) * self.ui_scale / width)
            width = self.measure_font(font_index, text, px)
        var x = Float32(self.ux(ui_left))
        if align == 1:
            x -= width * 0.5
        elif align == 2:
            x -= width
        var cap = self.font_cap(font_index) * px / self.font_bake(font_index)
        var baseline = Float32(self.uy(ui_top)) + cap
        # Display text gets a dark outline; all text a soft shadow, so it
        # reads over bright scenery without a panel behind it.
        var outline = px * (Float32(0.07) if font_index == 1 else Float32(0))
        var shadow = max(Float32(1), px * 0.06)
        if self.hud_drawing:
            self.draw_text_sprite(font_index, text, x, baseline, px, width, outline, shadow, rgb(red, green, blue))
            return
        self.draw_sdf_ink(font_index, text, x, baseline, px, outline, shadow, rgb(red, green, blue))

    def draw_sdf_ink(mut self, font_index: Int, text: String, x: Float32, baseline: Float32, px: Float32, outline: Float32, shadow: Float32, color: UInt32):
        self.draw_sdf_run(font_index, text, x + shadow, baseline + shadow, px, rgb(0, 0, 0), 110, Float32(0), rgb(0, 0, 0), max(Float32(1.2), px * 0.05))
        self.draw_sdf_run(font_index, text, x, baseline, px, color, 255, outline, rgb(24, 18, 14), Float32(0))

    def text_sprite_slot(mut self, key: String, font_index: Int, text: String, fraction_x: Float32, fraction_y: Float32, px: Float32, width: Float32, outline: Float32, shadow: Float32, color: UInt32, reach: Int, above: Int) raises -> Int:
        """Find this line's finished pixels, drawing them once if they are new.

        Sampling a distance field costs about 8 ms for a full HUD at 3440x1440, and
        the HUD redraws whenever one number changes. Only the changed line pays now.
        """
        if key in self.text_sprite_slots:
            return self.text_sprite_slots[key]
        if len(self.text_sprites) >= TEXT_SPRITE_LIMIT:
            self.text_sprites.clear()
            self.text_sprite_slots.clear()
        if len(self.text_canvas) != len(self.hud_pixels):
            self.text_canvas = List[UInt32](length=len(self.hud_pixels), fill=0)
        var sprite_width = min(self.width, Int(width) + reach * 2 + 2)
        var sprite_height = min(self.height, Int(px * 1.5) + reach * 2 + 2)
        swap(self.hud_pixels, self.text_canvas)
        self.draw_sdf_ink(font_index, text, Float32(reach) + fraction_x, Float32(above) + fraction_y, px, outline, shadow, color)
        swap(self.hud_pixels, self.text_canvas)
        var pixels = List[UInt32](length=sprite_width * sprite_height, fill=0)
        for y in range(sprite_height):
            var row = Int(self.text_canvas.unsafe_ptr()) + y * self.width * 4
            _ = external_call["memcpy", Int](Int(pixels.unsafe_ptr()) + y * sprite_width * 4, row, sprite_width * 4)
            _ = external_call["memset", Int](row, Int32(0), sprite_width * 4)
        self.text_sprites.append(TextSprite(sprite_width, sprite_height, pixels^))
        self.text_sprite_slots[key] = len(self.text_sprites) - 1
        return len(self.text_sprites) - 1

    def draw_text_sprite(mut self, font_index: Int, text: String, x: Float32, baseline: Float32, px: Float32, width: Float32, outline: Float32, shadow: Float32, color: UInt32):
        """Lay one line of text into the HUD layer from its finished pixels."""
        var reach = Int(outline + shadow + px * 0.25) + 2
        var above = Int(px * 1.1) + reach
        var whole_x = Int(floor(x))
        var whole_y = Int(floor(baseline))
        var fraction_x = x - Float32(whole_x)
        var fraction_y = baseline - Float32(whole_y)
        var key = String(font_index) + ":" + String(Int(px * 256)) + ":" + String(Int(fraction_x * 256)) + ":" + String(Int(fraction_y * 256)) + ":" + String(color) + ":" + text
        var slot: Int
        try:
            slot = self.text_sprite_slot(key, font_index, text, fraction_x, fraction_y, px, width, outline, shadow, color, reach, above)
        except:
            self.draw_sdf_ink(font_index, text, x, baseline, px, outline, shadow, color)
            return
        var left = whole_x - reach
        var top = whole_y - above
        var sprite_width = self.text_sprites[slot].width
        var sprite_height = self.text_sprites[slot].height
        self.add_hud_rectangle(left, top, sprite_width, sprite_height)
        var source = self.text_sprites[slot].pixels.unsafe_ptr()
        var target = self.hud_pixels.unsafe_ptr()
        for y in range(max(0, -top), min(sprite_height, self.height - top)):
            var from_row = y * sprite_width
            var to_row = (top + y) * self.width + left
            for column in range(max(0, -left), min(sprite_width, self.width - left)):
                var ink = source[unsafe_offset=from_row + column]
                var coverage = ink >> 24
                if coverage == 0:
                    continue
                var under = target[unsafe_offset=to_row + column]
                if coverage == 255 or under == 0:
                    target[unsafe_offset=to_row + column] = ink
                    continue
                # Premultiplied over: the same result as blending the line straight in.
                var inverse = 255 - coverage
                var alpha = coverage + ((under >> 24) * inverse) // 255
                var red = ((ink >> 16) & 255) + (((under >> 16) & 255) * inverse) // 255
                var green = ((ink >> 8) & 255) + (((under >> 8) & 255) * inverse) // 255
                var blue = (ink & 255) + ((under & 255) * inverse) // 255
                target[unsafe_offset=to_row + column] = alpha << 24 | min(red, UInt32(255)) << 16 | min(green, UInt32(255)) << 8 | min(blue, UInt32(255))

    def text_width(self, text: String, size: Int) -> Int:
        """Width on the UI canvas of `draw_text` at this size."""
        var font_index = 1 if size >= 30 else 0
        if not self.font_ready(font_index):
            return text.byte_length() * 6 * max(1, size // 8)
        var em = text_em(size) * self.text_scale
        return Int(self.measure_font(font_index, text, em) + 0.5)

    def font_ready(self, index: Int) -> Bool:
        return self.display_font.valid if index == 1 else self.ui_font.valid

    def font_cap(self, index: Int) -> Float32:
        return self.display_font.cap_height if index == 1 else self.ui_font.cap_height

    def font_bake(self, index: Int) -> Float32:
        return self.display_font.bake_px if index == 1 else self.ui_font.bake_px

    def measure_font(self, index: Int, text: String, px: Float32) -> Float32:
        return self.display_font.measure(text, px) if index == 1 else self.ui_font.measure(text, px)

    def draw_sdf_run(mut self, font_index: Int, text: String, x: Float32, baseline: Float32, px: Float32, color: UInt32, opacity: Int, outline: Float32, outline_color: UInt32, softness: Float32):
        """One line of SDF glyphs. Coverage comes from the distance in screen
        pixels: 0.5 + d is a one-pixel anti-aliased edge; `outline` grows the
        shape by that many pixels in `outline_color`; `softness` widens the
        edge for shadows."""
        if font_index == 1:
            self.display_font.ensure_text(text)
        else:
            self.ui_font.ensure_text(text)
        var codes = decode_utf8(text)
        var pen = x
        for index in range(len(codes)):
            var glyph_index = self.ui_font.glyph_for(codes[index]) if font_index == 0 else self.display_font.glyph_for(codes[index])
            if glyph_index < 0:
                continue
            self.draw_sdf_glyph(font_index, glyph_index, pen, baseline, px, color, opacity, outline, outline_color, softness)
            var bake = self.font_bake(font_index)
            var advance = self.display_font.glyphs[glyph_index].advance if font_index == 1 else self.ui_font.glyphs[glyph_index].advance
            pen += advance * px / bake
            if index + 1 < len(codes):
                var kern = self.display_font.kern(codes[index], codes[index + 1]) if font_index == 1 else self.ui_font.kern(codes[index], codes[index + 1])
                pen += kern * px / bake

    def font_sample(self, font_index: Int, x: Float32, y: Float32) -> Float32:
        return self.display_font.sample(x, y) if font_index == 1 else self.ui_font.sample(x, y)

    def draw_sdf_glyph(mut self, font_index: Int, glyph_index: Int, pen: Float32, baseline: Float32, px: Float32, color: UInt32, opacity: Int, outline: Float32, outline_color: UInt32, softness: Float32):
        var glyph = self.display_font.glyphs[glyph_index] if font_index == 1 else self.ui_font.glyphs[glyph_index]
        if glyph.w == 0 or glyph.h == 0:
            return
        var spread = self.display_font.spread if font_index == 1 else self.ui_font.spread
        var scale = px / self.font_bake(font_index)  # screen px per bake px
        var left = pen + glyph.xoff * scale
        var top = baseline + glyph.yoff * scale
        var x0 = max(0, Int(floor(left)))
        var y0 = max(0, Int(floor(top)))
        var x1 = min(self.width, Int(floor(left + Float32(glyph.w) * scale)) + 1)
        var y1 = min(self.height, Int(floor(top + Float32(glyph.h) * scale)) + 1)
        # Distance units (0-255) to screen pixels.
        var to_screen = spread * scale / 127.0
        var edge = 0.5 + max(Float32(0), softness - 0.5)
        var width_edge = max(Float32(1), softness * 2)
        var fill_red = Int((color >> 16) & 255)
        var fill_green = Int((color >> 8) & 255)
        var fill_blue = Int(color & 255)
        var line_red = Int((outline_color >> 16) & 255)
        var line_green = Int((outline_color >> 8) & 255)
        var line_blue = Int(outline_color & 255)
        for sy in range(y0, y1):
            var v = (Float32(sy) + 0.5 - top) / scale
            if v < 0 or v > Float32(glyph.h):
                continue
            for sx in range(x0, x1):
                var u = (Float32(sx) + 0.5 - left) / scale
                if u < 0 or u > Float32(glyph.w):
                    continue
                var distance = (self.font_sample(font_index, Float32(glyph.x) + u, Float32(glyph.y) + v) - 128.0) * to_screen
                var fill = max(Float32(0), min(Float32(1), (distance + edge) / width_edge))
                var total = fill
                if outline > 0:
                    total = max(Float32(0), min(Float32(1), (distance + outline + 0.5)))
                if total <= 0.003:
                    continue
                var mixed = color
                if outline > 0 and total > 0:
                    var t = fill / total
                    var red = line_red + Int(Float32(fill_red - line_red) * t)
                    var green = line_green + Int(Float32(fill_green - line_green) * t)
                    var blue = line_blue + Int(Float32(fill_blue - line_blue) * t)
                    mixed = UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
                var alpha = UInt8(min(255, Int(total * Float32(opacity) + 0.5)))
                self.blend_pixel(sy * self.width + sx, mixed, alpha)

    def set_button_style(mut self, pad: Bool, sony: Bool):
        """Package 3.12: button glyphs in rich text follow the last device used."""
        self.button_pad = pad
        self.button_sony = sony

    def set_button_keys(mut self, keys: List[String]):
        """Package 3.15: keycaps show the keys the player bound."""
        var same = len(keys) == len(self.button_keys)
        if same:
            for index in range(len(keys)):
                if keys[index] != self.button_keys[index]:
                    same = False
        if not same:
            self.button_keys = keys.copy()
            self.button_keys_generation += 1

    def rich_text_lines(self, text: String, size: Int, max_width: Int) -> Int:
        """Package 3.22: how many lines `draw_rich_text` wraps `text` to at
        `max_width` canvas units (the caption box holds two)."""
        var base_face = 1 if size >= 30 else 0
        var px = text_em(size) * self.text_scale * self.ui_scale
        var layout = layout_rich(parse_rich(text), self.ui_font, self.display_font, px, base_face, Float32(max_width) * self.ui_scale, 1, self.button_pad, self.button_sony, self.button_keys)
        return max(1, layout.lines)

    def draw_rich_text(mut self, text: String, ui_left: Int, ui_top: Int, size: Int, max_width: Int = 0, align: Int = 0, red: UInt8 = 240, green: UInt8 = 240, blue: UInt8 = 235) -> Int:
        """Package 3.12: text with tags (text/rich_text.mojo: colour, display
        face, size, button glyphs), wrapped at `max_width` canvas units (0:
        one line unless '\\n'). `ui_top` is the top of the first line's
        capitals, as for `draw_text`; align 0 left, 1 centred on `ui_left`, 2
        right of it. Returns the height used on the canvas."""
        var base_face = 1 if size >= 30 else 0
        var px = text_em(size) * self.text_scale * self.ui_scale
        var spans = parse_rich(text)
        var layout = layout_rich(spans, self.ui_font, self.display_font, px, base_face, Float32(max_width) * self.ui_scale, align, self.button_pad, self.button_sony, self.button_keys)
        var height = Int(layout.height / self.ui_scale + 0.5)
        if self.hud_recording:
            self.hud_texts.append(text)
            # The button style rides in the alpha field, so a device change redraws.
            var style = UInt8((1 if self.button_pad else 0) + (2 if self.button_sony else 0) + 4 * (self.button_keys_generation % 64))
            self.record_hud(6, ui_left, ui_top, size, align + max_width * 4, len(self.hud_texts) - 1, red, green, blue, style)
            return height
        if not self.font_ready(base_face) or not self.font_ready(0):
            self.draw_bitmap_text(plain_text(spans), ui_left, ui_top, size, red, green, blue, align)
            return height
        var box = Float32(max_width) * self.ui_scale if max_width > 0 else layout.width
        var left = Float32(self.ux(ui_left))
        if align == 1:
            left -= box * 0.5
        elif align == 2:
            left -= box
        var top = Float32(self.uy(ui_top))
        var base = rgb(red, green, blue)
        for run in layout.runs:
            var x = left + run.x
            var baseline = top + run.baseline
            var color = UInt32(run.color) if run.color >= 0 else base
            if run.kind == SPAN_BUTTON:
                var cap = self.font_cap(0) * run.px / self.font_bake(0)
                var h = run.px * 1.15
                self.draw_button_glyph(run.action, x, baseline - cap - (h - cap) * 0.5, run.width, h)
                continue
            var outline = run.px * (Float32(0.07) if run.face == 1 else Float32(0))
            var shadow = max(Float32(1), run.px * 0.06)
            if self.hud_drawing:
                self.draw_text_sprite(run.face, run.text, x, baseline, run.px, run.width, outline, shadow, color)
            else:
                self.draw_sdf_ink(run.face, run.text, x, baseline, run.px, outline, shadow, color)
        return height

    def fill_round_screen(mut self, x: Float32, y: Float32, w: Float32, h: Float32, radius: Float32, color: UInt32, alpha: Int = 255):
        """An anti-aliased rounded box in screen pixels (a circle when the
        radius is half the size), for button glyphs."""
        var r = max(Float32(0), min(radius, min(w, h) * 0.5))
        var x0 = max(0, Int(floor(x)))
        var y0 = max(0, Int(floor(y)))
        var x1 = min(self.width, Int(floor(x + w)) + 1)
        var y1 = min(self.height, Int(floor(y + h)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        if self.hud_drawing:
            self.add_hud_rectangle(x0, y0, x1 - x0, y1 - y0)
        var cx = x + w * 0.5
        var cy = y + h * 0.5
        var hx = w * 0.5 - r
        var hy = h * 0.5 - r
        for py in range(y0, y1):
            var qy = abs(Float32(py) + 0.5 - cy) - hy
            var dy = max(Float32(0), qy)
            for px in range(x0, x1):
                var qx = abs(Float32(px) + 0.5 - cx) - hx
                var dx = max(Float32(0), qx)
                # The rounded box's signed distance (negative inside).
                var distance = sqrt(dx * dx + dy * dy) + min(max(qx, qy), Float32(0)) - r
                var coverage = max(Float32(0), min(Float32(1), 0.5 - distance))
                if coverage <= 0.003:
                    continue
                self.blend_pixel(py * self.width + px, color, UInt8(min(255, Int(coverage * Float32(alpha) + 0.5))))

    def fill_triangle_up_screen(mut self, cx: Float32, top: Float32, size: Float32, color: UInt32):
        """An upward triangle outline (the PlayStation triangle)."""
        var thickness = max(Float32(1.5), size * 0.16)
        var rows = Int(size + 0.5)
        for row in range(rows):
            var t = Float32(row) / max(Float32(1), size)
            var half = size * 0.58 * t
            var y = floor(top) + Float32(row)
            if Float32(row) >= size - thickness:
                self.fill_round_screen(floor(cx - half), y, floor(half * 2 + 0.5), 1, 0, color)
            else:
                var edge = floor(min(thickness * 1.15, half) + 0.5)
                self.fill_round_screen(floor(cx - half), y, edge, 1, 0, color)
                self.fill_round_screen(floor(cx + half) - edge, y, edge, 1, 0, color)

    def draw_button_glyph(mut self, action: Int, x: Float32, y: Float32, w: Float32, h: Float32):
        """One button glyph in screen pixels with its top-left at (x, y): a
        keycap for the keyboard, else the controller button's own shape. The
        colours match the menu prompts' glyphs (render/interface.mojo)."""
        var shape = glyph_shape(action, self.button_pad)
        var label = glyph_label(action, self.button_pad, self.button_sony, self.button_keys)
        var label_px = h / Float32(1.15) * BUTTON_LABEL
        var dark = rgb(32, 40, 52)
        if shape == SHAPE_KEYCAP:
            var cap_h = h * 0.9
            self.fill_round_screen(x, y + h * 0.1, w, cap_h, h * 0.2, rgb(120, 128, 120))
            self.fill_round_screen(x, y, w, cap_h, h * 0.2, rgb(226, 230, 218))
            self.button_label(label, x + w * 0.5, y + cap_h * 0.5, label_px, rgb(30, 36, 40))
            return
        var button = pad_button(action)
        if shape == SHAPE_FACE:
            if self.button_sony:
                var symbol = face_color(button, True)
                self.fill_round_screen(x, y, h, h, h * 0.5, dark)
                if button == BUTTON_A:
                    self.button_label("×", x + h * 0.5, y + h * 0.5, h * 0.95, symbol)
                elif button == BUTTON_B:
                    self.fill_round_screen(x + h * 0.22, y + h * 0.22, h * 0.56, h * 0.56, h * 0.28, symbol)
                    self.fill_round_screen(x + h * 0.33, y + h * 0.33, h * 0.34, h * 0.34, h * 0.17, dark)
                elif button == BUTTON_X:
                    self.fill_round_screen(x + h * 0.26, y + h * 0.26, h * 0.48, h * 0.48, h * 0.04, symbol)
                    self.fill_round_screen(x + h * 0.35, y + h * 0.35, h * 0.30, h * 0.30, 0, dark)
                else:
                    self.fill_triangle_up_screen(x + h * 0.5, y + h * 0.24, h * 0.5, symbol)
                return
            self.fill_round_screen(x, y, h, h, h * 0.5, face_color(button, False))
            self.button_label(label, x + h * 0.5, y + h * 0.5, label_px, rgb(250, 250, 245))
            return
        if shape == SHAPE_PILL:
            self.fill_round_screen(x, y + h * 0.08, w, h * 0.84, h * 0.3, rgb(40, 48, 60))
            self.button_label(label, x + w * 0.5, y + h * 0.5, label_px, rgb(230, 236, 228))
            return
        if shape == SHAPE_DPAD:
            var light = rgb(210, 214, 204)
            var arm = h * 0.32
            self.fill_round_screen(x + (h - arm) * 0.5, y, arm, h, arm * 0.3, light)
            self.fill_round_screen(x, y + (h - arm) * 0.5, h, arm, arm * 0.3, light)
            var gold = rgb(255, 211, 112)
            var tip = arm * 0.7
            var mid = (h - tip) * 0.5
            if button == BUTTON_DPAD_UP:
                self.fill_round_screen(x + mid, y + arm * 0.15, tip, tip, tip * 0.2, gold)
            elif button == BUTTON_DPAD_RIGHT:
                self.fill_round_screen(x + h - arm * 0.15 - tip, y + mid, tip, tip, tip * 0.2, gold)
            elif button == BUTTON_DPAD_DOWN:
                self.fill_round_screen(x + mid, y + h - arm * 0.15 - tip, tip, tip, tip * 0.2, gold)
            else:
                self.fill_round_screen(x + arm * 0.15, y + mid, tip, tip, tip * 0.2, gold)
            return
        # A stick: a light ring around a dark cap.
        self.fill_round_screen(x, y, h, h, h * 0.5, rgb(210, 214, 204))
        self.fill_round_screen(x + h * 0.14, y + h * 0.14, h * 0.72, h * 0.72, h * 0.36, rgb(40, 48, 60))
        self.button_label(label, x + h * 0.5, y + h * 0.5, label_px * 0.85, rgb(230, 236, 228))

    def button_label(mut self, label: String, cx: Float32, cy: Float32, px: Float32, color: UInt32):
        """A label centred on (cx, cy) in the UI face, no shadow."""
        if label.byte_length() == 0 or not self.font_ready(0):
            return
        var width = self.measure_font(0, label, px)
        var cap = self.font_cap(0) * px / self.font_bake(0)
        self.draw_sdf_run(0, label, cx - width * 0.5, cy + cap * 0.5, px, color, 255, Float32(0), color, Float32(0))

    def draw_bitmap_text(mut self, text: String, ui_left: Int, ui_top: Int, size: Int, red: UInt8, green: UInt8, blue: UInt8, align: Int):
        """The original 5x7 font: the fallback when the font assets are missing."""
        var scale = max(1, Int(Float32(max(1, size // 8)) * self.ui_scale + 0.5))
        var cursor = self.ux(ui_left)
        var width = text.byte_length() * 6 * scale
        if align == 1:
            cursor -= width // 2
        elif align == 2:
            cursor -= width
        var y = self.uy(ui_top)
        var color = rgb(red, green, blue)
        if self.hud_drawing:
            color |= UInt32(0xFF000000)
            self.add_hud_rectangle(cursor, y, width, 7 * scale)
        for index in range(text.byte_length()):
            var code = Int(text.as_bytes()[index])
            if code == 32:
                cursor += 6 * scale
                continue
            var glyph = bitmap_glyph(code)
            for row in range(7):
                for column in range(5):
                    if (glyph >> UInt64(row * 5 + column)) & 1:
                        for sy in range(scale):
                            var py = y + row * scale + sy
                            if py < 0 or py >= self.height:
                                continue
                            for sx in range(scale):
                                var px = cursor + column * scale + sx
                                if px >= 0 and px < self.width:
                                    if self.hud_drawing:
                                        self.hud_pixels[py * self.width + px] = color
                                    else:
                                        self.pixels[py * self.width + px] = color
            cursor += 6 * scale


def box_blur_pass(source: List[Float32], mut target: List[Float32], width: Int, height: Int, horizontal: Bool):
    """One 5-tap box blur along rows or columns of an RGB float image."""
    for y in range(height):
        for x in range(width):
            var r: Float32 = 0
            var g: Float32 = 0
            var b: Float32 = 0
            for k in range(-2, 3):
                var sx = max(0, min(width - 1, x + k)) if horizontal else x
                var sy = y if horizontal else max(0, min(height - 1, y + k))
                var o = (sy * width + sx) * 3
                r += source[o]
                g += source[o + 1]
                b += source[o + 2]
            var t = (y * width + x) * 3
            target[t] = r * 0.2
            target[t + 1] = g * 0.2
            target[t + 2] = b * 0.2


comptime UI_FONT_FILE = "assets/deepsong/fonts/AtkinsonHyperlegibleNext[wght].ttf"
comptime UI_FONT_WEIGHT: Float32 = 600      # SemiBold
comptime DISPLAY_FONT_FILE = "assets/deepsong/fonts/LilitaOne-Regular.ttf"


def text_em(size: Int) -> Float32:
    """Em size on the UI canvas for a `draw_text` size. Small sizes are raised
    to 19 so body text (ascender to descender) is at least 26 px tall at
    1080p (Xbox XAG 101)."""
    return Float32(max(size, 19))
