"""World-space debug shapes: lines, spheres, capsules, boxes and text.

Anything may ask for a shape from the main thread or from a tick:

    debug.sphere(position, 0.5, DEBUG_RED)
    debug.line(from_here, to_here, DEBUG_GREEN, seconds=2)

A shape with no `seconds` lasts one frame. With the `debug.draw` console
variable off every call returns at its first branch. Fan-out job bodies must not
draw: nothing here is locked.

Shapes are painted last, on top of everything, by one CPU painter into a 32-bit
image whose rows the caller describes. So every backend gets the same pixels:
the CPU rasterizer hands over its framebuffer, and a GPU backend hands over its
HUD layer, which it composites over the scene after all post-processing. A new
backend needs nothing here. Colours are exact on screen, because no tone mapping,
bloom or smoothing runs after the HUD, and nothing is depth tested, because the
CPU has no depth on a GPU frame.

The HUD layer outlives a frame, so `paint` can keep a list of what it wrote for
the caller to undo before the next frame.
"""
from std.math import cos, isfinite, pi, sin, sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.render.font import bitmap_glyph

comptime DEBUG_RED: UInt32 = 0xFF4040
comptime DEBUG_GREEN: UInt32 = 0x40FF70
comptime DEBUG_BLUE: UInt32 = 0x50A0FF
comptime DEBUG_YELLOW: UInt32 = 0xFFE040
comptime DEBUG_CYAN: UInt32 = 0x40F0F0
comptime DEBUG_MAGENTA: UInt32 = 0xF060F0
comptime DEBUG_ORANGE: UInt32 = 0xFF9030
comptime DEBUG_WHITE: UInt32 = 0xFFFFFF
comptime DEBUG_GREY: UInt32 = 0x9098A0

comptime SHAPE_LINE: Int = 0
comptime SHAPE_SPHERE: Int = 1
comptime SHAPE_CAPSULE: Int = 2
comptime SHAPE_BOX: Int = 3
comptime SHAPE_TEXT: Int = 4

comptime SHAPE_LIMIT: Int = 16384  # past this, shapes are counted and dropped
comptime RING_SEGMENTS: Int = 24
comptime NEAR: Float32 = 0.06
comptime GLYPH_ADVANCE: Int = 6
comptime GLYPH_ROWS: Int = 7
comptime GLYPH_COLUMNS: Int = 5
comptime NEWLINE: Int = 10
comptime SPACE: Int = 32


@fieldwise_init
struct DebugView(Copyable, ImplicitlyCopyable, Movable):
    """Where the frame looks from: the same projection the renderers use."""
    var camera: Vec3
    var right: Vec3
    var up: Vec3
    var forward: Vec3
    var focal: Float32
    var width: Int
    var height: Int

    def to_camera(self, point: Vec3) -> Vec3:
        var relative = point - self.camera
        return Vec3(
            relative.x * self.right.x + relative.y * self.right.y + relative.z * self.right.z,
            relative.x * self.up.x + relative.y * self.up.y + relative.z * self.up.z,
            relative.x * self.forward.x + relative.y * self.forward.y + relative.z * self.forward.z,
        )

    def screen_x(self, in_camera: Vec3) -> Float64:
        """Float64, because a point just past the near plane lands far off screen
        and a Float32 there cannot clip back to the right pixel."""
        return Float64(self.width) * 0.5 + Float64(in_camera.x) * Float64(self.focal) / Float64(in_camera.z)

    def screen_y(self, in_camera: Vec3) -> Float64:
        return Float64(self.height) * 0.5 - Float64(in_camera.y) * Float64(self.focal) / Float64(in_camera.z)


struct DebugShape(Copyable, Movable):
    var kind: Int
    var a: Vec3  # a line's or capsule's first end; the centre of anything else
    var b: Vec3  # a line's or capsule's second end; a box's half extents
    var radius: Float32
    var rotation: Quat
    var color: UInt32
    var remaining: Float32  # seconds left after this frame
    var text: String

    def __init__(out self, kind: Int, a: Vec3, b: Vec3, radius: Float32, rotation: Quat, color: UInt32, seconds: Float32, var text: String = String()):
        self.kind = kind
        self.a = a
        self.b = b
        self.radius = radius
        self.rotation = rotation
        self.color = color
        self.remaining = seconds
        self.text = text^

    def reach(self) -> Float32:
        """A radius around `a` that holds the whole shape."""
        if self.kind == SHAPE_SPHERE:
            return self.radius
        if self.kind == SHAPE_BOX:
            return self.b.length()
        return (self.b - self.a).length() + self.radius


def perpendicular(direction: Vec3) -> Vec3:
    """Some unit vector at right angles to a unit vector."""
    var other = Vec3(1, 0, 0) if abs(direction.x) < 0.7 else Vec3(0, 1, 0)
    return Vec3(
        direction.y * other.z - direction.z * other.y,
        direction.z * other.x - direction.x * other.z,
        direction.x * other.y - direction.y * other.x,
    ).normalized()


def side_of(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


struct DebugDraw(Copyable, Movable):
    var enabled: Bool
    var frame_seconds: Float32
    var shapes: List[DebugShape]
    var dropped: Int  # since the start
    var written: List[UInt32]  # pixels `paint` kept track of, as x | y << 16
    var ring_x: List[Float32]
    var ring_y: List[Float32]
    var view: DebugView
    var target: Int  # the image being painted
    var row_pixels: Int
    var view_w: Int
    var view_h: Int
    var high_bits: UInt32  # what a painted pixel holds above its colour
    var keep_written: Bool
    var thickness: Int
    var lines_painted: Int  # last paint
    var shapes_painted: Int

    def __init__(out self):
        self.enabled = False
        self.frame_seconds = 1.0 / 60.0
        self.shapes = List[DebugShape]()
        self.dropped = 0
        self.written = List[UInt32]()
        self.ring_x = List[Float32](capacity=RING_SEGMENTS + 1)
        self.ring_y = List[Float32](capacity=RING_SEGMENTS + 1)
        for index in range(RING_SEGMENTS + 1):
            var angle = Float32(index) / Float32(RING_SEGMENTS) * Float32(pi) * 2.0
            self.ring_x.append(cos(angle))
            self.ring_y.append(sin(angle))
        self.view = DebugView(Vec3(), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), 1, 0, 0)
        self.target = 0
        self.row_pixels = 0
        self.view_w = 0
        self.view_h = 0
        self.high_bits = 0
        self.keep_written = False
        self.thickness = 1
        self.lines_painted = 0
        self.shapes_painted = 0

    def set_enabled(mut self, on: Bool, frame_seconds: Float32):
        """Once a frame. Turning off forgets every shape still alive."""
        if self.enabled and not on:
            self.shapes.clear()
        self.enabled = on
        self.frame_seconds = frame_seconds

    def has_work(self) -> Bool:
        """True when a frame has something to paint or something to undo."""
        return len(self.shapes) > 0 or len(self.written) > 0

    def add(mut self, var shape: DebugShape):
        if len(self.shapes) >= SHAPE_LIMIT:
            self.dropped += 1
            return
        self.shapes.append(shape^)

    def line(mut self, a: Vec3, b: Vec3, color: UInt32, seconds: Float32 = 0):
        if self.enabled:
            self.add(DebugShape(SHAPE_LINE, a, b, 0, Quat(), color, seconds))

    def sphere(mut self, center: Vec3, radius: Float32, color: UInt32, seconds: Float32 = 0):
        if self.enabled:
            self.add(DebugShape(SHAPE_SPHERE, center, Vec3(), radius, Quat(), color, seconds))

    def capsule(mut self, a: Vec3, b: Vec3, radius: Float32, color: UInt32, seconds: Float32 = 0):
        """The two ends are the centres of its end caps."""
        if self.enabled:
            self.add(DebugShape(SHAPE_CAPSULE, a, b, radius, Quat(), color, seconds))

    def box(mut self, center: Vec3, half: Vec3, color: UInt32, rotation: Quat = Quat(), seconds: Float32 = 0):
        if self.enabled:
            self.add(DebugShape(SHAPE_BOX, center, half, 0, rotation, color, seconds))

    def text(mut self, position: Vec3, var text: String, color: UInt32 = DEBUG_WHITE, seconds: Float32 = 0):
        """Text centred above where `position` lands. '\\n' starts a new line."""
        if self.enabled:
            self.add(DebugShape(SHAPE_TEXT, position, Vec3(), 0, Quat(), color, seconds, text^))

    def text_scale(self) -> Int:
        """Screen pixels per font pixel: 14 tall up to 1080 lines, 21 at 1440."""
        return max(2, self.view.height // 480)

    def paint(mut self, view: DebugView, target: Int, row_pixels: Int, high_bits: UInt32, keep_written: Bool):
        """Paint every shape into the image, then age them."""
        self.lines_painted = 0
        self.shapes_painted = 0
        if target == 0 or row_pixels <= 0 or view.width <= 0 or view.height <= 0 or row_pixels < view.width:
            self.target = 0
            self.row_pixels = 0
            self.view_w = 0
            self.view_h = 0
            return
        self.view = view
        self.target = target
        self.row_pixels = row_pixels
        self.view_w = view.width
        self.view_h = view.height
        self.high_bits = high_bits
        self.keep_written = keep_written
        self.thickness = max(1, view.height // 720)
        self.shapes_painted = len(self.shapes)
        # Text goes over every wire, or labels are unreadable in a busy frame.
        for index in range(len(self.shapes)):
            if self.shapes[index].kind != SHAPE_TEXT:
                self.paint_shape(index)
        for index in range(len(self.shapes)):
            if self.shapes[index].kind == SHAPE_TEXT:
                self.paint_text(index)
        var keep = 0  # survivors slide down, staying in the order they were asked for
        for index in range(len(self.shapes)):
            self.shapes[index].remaining -= self.frame_seconds
            if self.shapes[index].remaining > 0:
                if keep != index:
                    self.shapes[keep] = self.shapes[index].copy()
                keep += 1
        while len(self.shapes) > keep:
            _ = self.shapes.pop()

    def paint_shape(mut self, index: Int):
        var kind = self.shapes[index].kind
        var a = self.shapes[index].a
        var b = self.shapes[index].b
        var radius = self.shapes[index].radius
        var color = self.shapes[index].color
        if not (isfinite(a.x) and isfinite(a.y) and isfinite(a.z)):
            return
        if kind != SHAPE_SPHERE and not (isfinite(b.x) and isfinite(b.y) and isfinite(b.z)):
            return
        if not isfinite(radius) or radius < Float32(0.0):
            return
        if kind != SHAPE_LINE and self.view.to_camera(a).z + self.shapes[index].reach() < NEAR:
            return
        if kind == SHAPE_LINE:
            self.world_line(a, b, color)
        elif kind == SHAPE_SPHERE:
            self.paint_sphere(a, radius, color)
        elif kind == SHAPE_CAPSULE:
            self.paint_capsule(a, b, radius, color)
        else:
            var rotation = self.shapes[index].rotation
            self.paint_box(a, b, rotation, color)

    def ring(mut self, center: Vec3, axis_x: Vec3, axis_y: Vec3, first: Int, last: Int, color: UInt32):
        """Segments `first` to `last` of a circle given by two scaled axes."""
        var previous = center + axis_x * self.ring_x[first] + axis_y * self.ring_y[first]
        for segment in range(first + 1, last + 1):
            var next = center + axis_x * self.ring_x[segment] + axis_y * self.ring_y[segment]
            self.world_line(previous, next, color)
            previous = next

    def paint_sphere(mut self, center: Vec3, radius: Float32, color: UInt32):
        """Three great circles, and the outline as the camera sees it."""
        var x = Vec3(radius, 0, 0)
        var y = Vec3(0, radius, 0)
        var z = Vec3(0, 0, radius)
        self.ring(center, x, y, 0, RING_SEGMENTS, color)
        self.ring(center, x, z, 0, RING_SEGMENTS, color)
        self.ring(center, y, z, 0, RING_SEGMENTS, color)
        var to_center = center - self.view.camera
        var distance = to_center.length()
        if distance > radius:
            # Sight lines touch the sphere on a circle nearer than its centre.
            var along = radius * radius / distance
            var size = sqrt(radius * radius - along * along)
            self.ring(center - to_center * (along / distance), self.view.right * size, self.view.up * size, 0, RING_SEGMENTS, color)

    def paint_capsule(mut self, a: Vec3, b: Vec3, radius: Float32, color: UInt32):
        var along = b - a
        if along.length_squared() < 0.000001:
            self.paint_sphere(a, radius, color)
            return
        var axis = along.normalized()
        var side = perpendicular(axis)
        var other = side_of(axis, side)
        var half = RING_SEGMENTS // 2
        var x = side * radius
        var y = other * radius
        var cap = axis * radius
        self.ring(a, x, y, 0, RING_SEGMENTS, color)
        self.ring(b, x, y, 0, RING_SEGMENTS, color)
        for corner in range(4):
            var offset = x * self.ring_x[corner * RING_SEGMENTS // 4] + y * self.ring_y[corner * RING_SEGMENTS // 4]
            self.world_line(a + offset, b + offset, color)
        # Half circles over each end, in two planes through the axis.
        self.ring(b, x, cap, 0, half, color)
        self.ring(b, y, cap, 0, half, color)
        self.ring(a, x, cap * -1, 0, half, color)
        self.ring(a, y, cap * -1, 0, half, color)

    def paint_box(mut self, center: Vec3, half: Vec3, rotation: Quat, color: UInt32):
        if not (isfinite(rotation.x) and isfinite(rotation.y) and isfinite(rotation.z) and isfinite(rotation.w)):
            return
        var corners = List[Vec3](capacity=8)
        for index in range(8):
            corners.append(center + rotation.rotate(Vec3(
                half.x if index & 1 else -half.x,
                half.y if index & 2 else -half.y,
                half.z if index & 4 else -half.z,
            )))
        for index in range(8):
            for axis in range(3):
                # Each edge joins corners one bit apart; draw it from the lower.
                var other = index | (1 << axis)
                if other != index:
                    self.world_line(corners[index], corners[other], color)

    def world_line(mut self, start: Vec3, end: Vec3, color: UInt32):
        if not (
            isfinite(start.x) and isfinite(start.y) and isfinite(start.z)
            and isfinite(end.x) and isfinite(end.y) and isfinite(end.z)
        ):
            return
        var a = self.view.to_camera(start)
        var b = self.view.to_camera(end)
        if not (
            isfinite(a.x) and isfinite(a.y) and isfinite(a.z)
            and isfinite(b.x) and isfinite(b.y) and isfinite(b.z)
        ):
            return
        if a.z < NEAR and b.z < NEAR:
            return
        if a.z < NEAR:
            if b.z == a.z:
                return
            a = a + (b - a) * ((NEAR - a.z) / (b.z - a.z))
        elif b.z < NEAR:
            if a.z == b.z:
                return
            b = b + (a - b) * ((NEAR - b.z) / (a.z - b.z))
        self.screen_line(self.view.screen_x(a), self.view.screen_y(a), self.view.screen_x(b), self.view.screen_y(b), color)

    def screen_line(mut self, start_x: Float64, start_y: Float64, end_x: Float64, end_y: Float64, color: UInt32):
        """Clip to the image (Liang-Barsky), then step along the longer axis."""
        if not (isfinite(start_x) and isfinite(start_y) and isfinite(end_x) and isfinite(end_y)):
            return
        var dx = end_x - start_x
        var dy = end_y - start_y
        if not (isfinite(dx) and isfinite(dy)):
            return
        var enter = Float64(0)
        var leave = Float64(1)
        var limit_x = Float64(self.view.width) - 0.001
        var limit_y = Float64(self.view.height) - 0.001
        var sides = [(-dx, start_x), (dx, limit_x - start_x), (-dy, start_y), (dy, limit_y - start_y)]
        for side in sides:
            var toward = side[0]
            var room = side[1]
            if toward == 0:
                if room < 0:
                    return
                continue
            var meet = room / toward
            if not isfinite(meet):
                return
            if toward < 0:
                enter = max(enter, meet)
            else:
                leave = min(leave, meet)
            if enter > leave:
                return
        var sx_clip = start_x + dx * enter
        var sy_clip = start_y + dy * enter
        dx *= leave - enter
        dy *= leave - enter
        if not (isfinite(sx_clip) and isfinite(sy_clip) and isfinite(dx) and isfinite(dy)):
            return
        var x = Float32(sx_clip)
        var y = Float32(sy_clip)
        var steps = Int(max(abs(dx), abs(dy))) + 1
        var max_steps = max(self.view.width, self.view.height) * 4 + 16
        if steps <= 0 or steps > max_steps:
            return
        var step_x = Float32(dx / Float64(steps))
        var step_y = Float32(dy / Float64(steps))
        if not (isfinite(x) and isfinite(y) and isfinite(step_x) and isfinite(step_y)):
            return
        self.lines_painted += 1
        var wide = abs(dx) >= abs(dy)
        var width = self.view.width
        var height = self.view.height
        var brush = self.thickness
        var back = (brush - 1) // 2
        var pixel = color | self.high_bits
        for _ in range(steps + 1):
            if not (isfinite(x) and isfinite(y)):
                break
            var px = Int(x)
            var py = Int(y)
            for offset in range(brush):
                var bx = px if wide else px + offset - back
                var by = py + offset - back if wide else py
                if bx >= 0 and by >= 0 and bx < width and by < height:
                    self.plot(bx, by, pixel)
            x += step_x
            y += step_y

    @always_inline
    def plot(mut self, x: Int, y: Int, pixel: UInt32):
        if (
            self.target == 0
            or self.row_pixels <= 0
            or self.view_w <= 0
            or self.view_h <= 0
            or x < 0
            or y < 0
            or x >= self.view_w
            or x >= self.row_pixels
            or y >= self.view_h
        ):
            return
        var image = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.target)
        image[unsafe_offset=y * self.row_pixels + x] = pixel
        if self.keep_written:
            self.written.append(UInt32(x) | UInt32(y) << 16)

    def paint_text(mut self, index: Int):
        """The bottom line ends at the anchor, so a label clears what it names."""
        var anchor = self.shapes[index].a
        if not (isfinite(anchor.x) and isfinite(anchor.y) and isfinite(anchor.z)):
            return
        var in_camera = self.view.to_camera(anchor)
        if not (isfinite(in_camera.x) and isfinite(in_camera.y) and isfinite(in_camera.z)) or in_camera.z < NEAR:
            return
        var sx = self.view.screen_x(in_camera)
        var sy = self.view.screen_y(in_camera)
        if not (isfinite(sx) and isfinite(sy)):
            return
        var center_x = Int(sx)
        var scale = self.text_scale()
        var text = self.shapes[index].text.copy()
        var count = text.byte_length()
        var lines = 1
        for position in range(count):
            if Int(text.as_bytes()[position]) == NEWLINE:
                lines += 1
        var line_height = (GLYPH_ROWS + 2) * scale
        var top = Int(sy) - lines * line_height
        var start = 0
        var color = self.shapes[index].color | self.high_bits
        var shadow = self.high_bits
        while start <= count:
            var stop = start
            while stop < count and Int(text.as_bytes()[stop]) != NEWLINE:
                stop += 1
            var left = center_x - (stop - start) * GLYPH_ADVANCE * scale // 2
            # The shadow goes down first, so it never covers the next glyph's ink.
            self.paint_run(text, start, stop, left + scale, top + scale, scale, shadow)
            self.paint_run(text, start, stop, left, top, scale, color)
            top += line_height
            start = stop + 1

    def paint_run(mut self, text: String, start: Int, stop: Int, left: Int, top: Int, scale: Int, pixel: UInt32):
        var width = self.view.width
        var height = self.view.height
        var cursor = left
        for position in range(start, stop):
            var code = Int(text.as_bytes()[position])
            if code != SPACE and cursor + GLYPH_COLUMNS * scale > 0 and cursor < width:
                var glyph = bitmap_glyph(code)
                for row in range(GLYPH_ROWS):
                    for column in range(GLYPH_COLUMNS):
                        if (glyph >> UInt64(row * GLYPH_COLUMNS + column)) & 1 == 0:
                            continue
                        for py in range(top + row * scale, top + (row + 1) * scale):
                            for px in range(cursor + column * scale, cursor + (column + 1) * scale):
                                if px >= 0 and py >= 0 and px < width and py < height:
                                    self.plot(px, py, pixel)
            cursor += GLYPH_ADVANCE * scale
