"""Menu widgets with focus navigation by pad, keyboard and mouse (package 3.12).

A `WidgetTree` holds buttons, toggles, sliders and labels laid out on the
1280x720 UI canvas. One widget has the focus. Directions move it to the
nearest enabled widget that way (centre to centre, preferring widgets lined
up with the current one; with nothing that way it wraps within the column or
row), the pointer focuses what it hovers and a press activates it. A held
stick or key repeats after a delay. Each widget eases its focus glow in and
out, and the whole tree slides and fades in when it opens.

All of this is presentation: menus never feed the simulation, so plain
floats and std.math are fine here.
"""
from std.math import exp, sqrt
from flint.platform.input import (
    InputState, PAD_A, PAD_B, PAD_UP, PAD_DOWN, PAD_LEFT, PAD_RIGHT,
    PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y, PAD_AXIS_DPAD_X, PAD_AXIS_DPAD_Y, PAD_DIRECTION,
)
from flint.render.rasterizer import SoftwareFramebuffer
from flint.ui.nine_slice import SLICE_SLATE, SLICE_FOCUS

comptime WIDGET_BUTTON: Int = 0
comptime WIDGET_TOGGLE: Int = 1
comptime WIDGET_SLIDER: Int = 2
comptime WIDGET_LABEL: Int = 3

comptime DIR_UP: Int = 0
comptime DIR_DOWN: Int = 1
comptime DIR_LEFT: Int = 2
comptime DIR_RIGHT: Int = 3
comptime DIR_NONE: Int = -1

comptime REPEAT_DELAY: Float32 = 0.35   # seconds a direction is held before it repeats
comptime REPEAT_EVERY: Float32 = 0.12
comptime FOCUS_TIME: Float32 = 0.12     # focus glow time constant
comptime OPEN_TIME: Float32 = 0.22      # open and close
comptime OPEN_SLIDE: Float32 = 18       # canvas units the widgets slide in from

comptime KEY_ESCAPE: Int = 256
comptime KEY_ENTER: Int = 257
comptime KEY_SPACE: Int = 32
comptime KEY_RIGHT: Int = 262
comptime KEY_LEFT: Int = 263
comptime KEY_DOWN: Int = 264
comptime KEY_UP: Int = 265
comptime KEY_W: Int = 87
comptime KEY_A: Int = 65
comptime KEY_S: Int = 83
comptime KEY_D: Int = 68


struct UiInput(Copyable, ImplicitlyCopyable, Movable):
    """What the menus need from this frame's input, from any device.
    Directions are held states; the tree turns them into steps with repeat."""
    var up: Bool
    var down: Bool
    var left: Bool
    var right: Bool
    var confirm: Bool      # pressed this frame
    var back: Bool         # pressed this frame
    var pointer_x: Float32 # canvas units
    var pointer_y: Float32
    var pointer_moved: Bool
    var pointer_pressed: Bool
    var from_pad: Bool     # a controller produced something this frame

    def __init__(out self):
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.confirm = False
        self.back = False
        self.pointer_x = -100000
        self.pointer_y = -100000
        self.pointer_moved = False
        self.pointer_pressed = False
        self.from_pad = False

    def direction(self) -> Int:
        if self.up:
            return DIR_UP
        if self.down:
            return DIR_DOWN
        if self.left:
            return DIR_LEFT
        if self.right:
            return DIR_RIGHT
        return DIR_NONE


def ui_input_from(input: InputState, pointer_x: Float32 = -100000, pointer_y: Float32 = -100000, pad: Int = 0) -> UiInput:
    """Keyboard (arrows, WASD, Enter, Space, Esc), the controller (d-pad, left
    stick, A, B) and the mouse (pointer on the canvas, left button) as one
    `UiInput`. `pointer_x`/`pointer_y` are canvas units
    (`NativeRuntime.mouse_canvas`)."""
    var ui = UiInput()
    var stick_x = input.gamepad_axis(pad, PAD_AXIS_LEFT_X)
    var stick_y = input.gamepad_axis(pad, PAD_AXIS_LEFT_Y)
    var dpad_x = input.gamepad_axis(pad, PAD_AXIS_DPAD_X)
    var dpad_y = input.gamepad_axis(pad, PAD_AXIS_DPAD_Y)
    var pad_up = input.gamepad_down(pad, PAD_UP) or stick_y < -PAD_DIRECTION or dpad_y < -0.5
    var pad_down = input.gamepad_down(pad, PAD_DOWN) or stick_y > PAD_DIRECTION or dpad_y > 0.5
    var pad_left = input.gamepad_down(pad, PAD_LEFT) or stick_x < -PAD_DIRECTION or dpad_x < -0.5
    var pad_right = input.gamepad_down(pad, PAD_RIGHT) or stick_x > PAD_DIRECTION or dpad_x > 0.5
    var pad_confirm = input.gamepad_pressed(pad, PAD_A)
    var pad_back = input.gamepad_pressed(pad, PAD_B)
    ui.up = input.key_down(KEY_UP) or input.key_down(KEY_W) or pad_up
    ui.down = input.key_down(KEY_DOWN) or input.key_down(KEY_S) or pad_down
    ui.left = input.key_down(KEY_LEFT) or input.key_down(KEY_A) or pad_left
    ui.right = input.key_down(KEY_RIGHT) or input.key_down(KEY_D) or pad_right
    ui.confirm = input.key_pressed(KEY_ENTER) or input.key_pressed(KEY_SPACE) or pad_confirm
    ui.back = input.key_pressed(KEY_ESCAPE) or pad_back
    ui.from_pad = pad_up or pad_down or pad_left or pad_right or pad_confirm or pad_back
    ui.pointer_x = pointer_x
    ui.pointer_y = pointer_y
    ui.pointer_moved = input.mouse_delta_x() != 0 or input.mouse_delta_y() != 0
    ui.pointer_pressed = input.mouse_pressed(0)
    return ui


struct Widget(Copyable, Movable):
    var id: Int
    var kind: Int
    var x: Int          # canvas units
    var y: Int
    var w: Int
    var h: Int
    var label: String
    var on: Bool        # toggles
    var value: Float32  # sliders, 0 to 1
    var step: Float32
    var enabled: Bool
    var focus_amount: Float32   # eased 0-1 glow
    var pulse: Float32          # seconds left of the activation flash

    def __init__(out self, id: Int, kind: Int, x: Int, y: Int, w: Int, h: Int, label: String):
        self.id = id
        self.kind = kind
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.on = False
        self.value = 0
        self.step = 0.1
        self.enabled = True
        self.focus_amount = 0
        self.pulse = 0

    def focusable(self) -> Bool:
        return self.enabled and self.kind != WIDGET_LABEL

    def centre_x(self) -> Float32:
        return Float32(self.x) + Float32(self.w) * 0.5

    def centre_y(self) -> Float32:
        return Float32(self.y) + Float32(self.h) * 0.5

    def contains(self, x: Float32, y: Float32) -> Bool:
        return x >= Float32(self.x) and y >= Float32(self.y) and x < Float32(self.x + self.w) and y < Float32(self.y + self.h)


struct WidgetTree(Copyable, Movable):
    var widgets: List[Widget]
    var focus: Int            # index into widgets, -1 for none
    var held: Int             # direction held last frame
    var held_time: Float32
    var repeats: Int          # steps taken while this direction is held
    var open_amount: Float32  # 0 closed, 1 open (eased by `open_progress`)
    var opening: Bool
    var activated: Int        # id activated this update, or -1
    var changed: Int          # id whose value changed this update, or -1
    var backed: Bool          # back was pressed this update

    def __init__(out self):
        self.widgets = List[Widget]()
        self.focus = -1
        self.held = DIR_NONE
        self.held_time = 0
        self.repeats = 0
        self.open_amount = 0
        self.opening = True
        self.activated = -1
        self.changed = -1
        self.backed = False

    def add(mut self, var widget: Widget) -> Int:
        self.widgets.append(widget^)
        var index = len(self.widgets) - 1
        if self.focus < 0 and self.widgets[index].focusable():
            self.focus = index
        return index

    def index_of(self, id: Int) -> Int:
        for index in range(len(self.widgets)):
            if self.widgets[index].id == id:
                return index
        return -1

    def focused_id(self) -> Int:
        if self.focus < 0 or self.focus >= len(self.widgets):
            return -1
        return self.widgets[self.focus].id

    def set_enabled(mut self, id: Int, enabled: Bool):
        var index = self.index_of(id)
        if index < 0:
            return
        self.widgets[index].enabled = enabled
        if not enabled and index == self.focus:
            # Hand the focus on to the next focusable widget.
            self.focus = -1
            for step in range(1, len(self.widgets) + 1):
                var other = (index + step) % len(self.widgets)
                if self.widgets[other].focusable():
                    self.focus = other
                    break

    def navigate(mut self, direction: Int) -> Bool:
        """Move the focus one step. Returns whether it moved."""
        if self.focus < 0:
            for index in range(len(self.widgets)):
                if self.widgets[index].focusable():
                    self.focus = index
                    return True
            return False
        var best = self.nearest(direction)
        if best < 0:
            best = self.wrap(direction)
        if best < 0 or best == self.focus:
            return False
        self.focus = best
        return True

    def nearest(self, direction: Int) -> Int:
        """The best widget strictly in `direction`: distance along the
        direction plus twice the sideways offset, so a widget straight
        ahead beats a nearer one off to the side."""
        var cx = self.widgets[self.focus].centre_x()
        var cy = self.widgets[self.focus].centre_y()
        var best = -1
        var best_score: Float32 = 1e30
        for index in range(len(self.widgets)):
            if index == self.focus or not self.widgets[index].focusable():
                continue
            var dx = self.widgets[index].centre_x() - cx
            var dy = self.widgets[index].centre_y() - cy
            var along: Float32
            var side: Float32
            if direction == DIR_UP:
                along = -dy
                side = abs(dx)
            elif direction == DIR_DOWN:
                along = dy
                side = abs(dx)
            elif direction == DIR_LEFT:
                along = -dx
                side = abs(dy)
            else:
                along = dx
                side = abs(dy)
            if along <= 0.5:
                continue
            # Within a 60-degree-wide cone only... unless nothing else is there.
            var score = along + side * 2.0
            if side > along * 2.0:
                score += 10000
            if score < best_score:
                best_score = score
                best = index
        return best

    def wrap(self, direction: Int) -> Int:
        """Nothing that way: the farthest widget the other way that lines up
        with this one (same column for up/down, same row for left/right)."""
        var cx = self.widgets[self.focus].centre_x()
        var cy = self.widgets[self.focus].centre_y()
        var half_w = Float32(self.widgets[self.focus].w) * 0.5
        var half_h = Float32(self.widgets[self.focus].h) * 0.5
        var best = -1
        var best_score: Float32 = -1e30
        for index in range(len(self.widgets)):
            if index == self.focus or not self.widgets[index].focusable():
                continue
            var dx = self.widgets[index].centre_x() - cx
            var dy = self.widgets[index].centre_y() - cy
            var score: Float32
            if direction == DIR_UP or direction == DIR_DOWN:
                if abs(dx) > half_w:
                    continue
                score = dy if direction == DIR_UP else -dy
            else:
                if abs(dy) > half_h:
                    continue
                score = dx if direction == DIR_LEFT else -dx
            if score > best_score:
                best_score = score
                best = index
        return best

    def pointer(mut self, x: Float32, y: Float32) -> Bool:
        """Focus the widget under the pointer. Returns whether one is there."""
        for index in range(len(self.widgets)):
            if self.widgets[index].focusable() and self.widgets[index].contains(x, y):
                self.focus = index
                return True
        return False

    def activate(mut self) -> Int:
        """Press the focused widget: toggles flip, buttons report their id."""
        if self.focus < 0 or not self.widgets[self.focus].focusable():
            return -1
        if self.widgets[self.focus].kind == WIDGET_TOGGLE:
            self.widgets[self.focus].on = not self.widgets[self.focus].on
            self.changed = self.widgets[self.focus].id
        self.widgets[self.focus].pulse = 0.18
        return self.widgets[self.focus].id

    def adjust(mut self, steps: Int) -> Bool:
        """Move the focused slider by whole steps, clamped to 0-1."""
        if self.focus < 0 or self.widgets[self.focus].kind != WIDGET_SLIDER:
            return False
        var before = self.widgets[self.focus].value
        var after = max(Float32(0), min(Float32(1), before + Float32(steps) * self.widgets[self.focus].step))
        # Snap to the step grid so repeated steps land exactly on 0 and 1.
        var grid = self.widgets[self.focus].step
        if grid > 0:
            after = Float32(Int(after / grid + 0.5)) * grid
            after = max(Float32(0), min(Float32(1), after))
        self.widgets[self.focus].value = after
        if after != before:
            self.changed = self.widgets[self.focus].id
            return True
        return False

    def step_direction(mut self, direction: Int):
        if self.focus >= 0 and self.widgets[self.focus].kind == WIDGET_SLIDER and (direction == DIR_LEFT or direction == DIR_RIGHT):
            _ = self.adjust(-1 if direction == DIR_LEFT else 1)
            return
        _ = self.navigate(direction)

    def update(mut self, input: UiInput, dt: Float32):
        """One frame of menu input and transitions. Afterwards `activated`,
        `changed` and `backed` say what happened."""
        self.activated = -1
        self.changed = -1
        self.backed = False
        var direction = input.direction()
        if direction != DIR_NONE:
            if direction != self.held:
                self.held = direction
                self.held_time = 0
                self.repeats = 0
                self.step_direction(direction)
            else:
                self.held_time += dt
                var due = 1 + Int((self.held_time - REPEAT_DELAY) / REPEAT_EVERY) if self.held_time >= REPEAT_DELAY else 0
                while self.repeats < due:
                    self.repeats += 1
                    self.step_direction(direction)
        else:
            self.held = DIR_NONE
            self.held_time = 0
            self.repeats = 0
        if input.pointer_moved or input.pointer_pressed:
            var over = self.pointer(input.pointer_x, input.pointer_y)
            if input.pointer_pressed and over:
                self.activated = self.activate()
        if input.confirm:
            self.activated = self.activate()
        if input.back:
            self.backed = True
        self.animate(dt)

    def animate(mut self, dt: Float32):
        var blend = 1.0 - exp(-dt / FOCUS_TIME)
        for index in range(len(self.widgets)):
            var target: Float32 = 1.0 if index == self.focus else 0.0
            self.widgets[index].focus_amount += (target - self.widgets[index].focus_amount) * blend
            if abs(target - self.widgets[index].focus_amount) < 0.001:
                self.widgets[index].focus_amount = target
            self.widgets[index].pulse = max(Float32(0), self.widgets[index].pulse - dt)
        var goal: Float32 = 1.0 if self.opening else 0.0
        var move = dt / OPEN_TIME
        if self.open_amount < goal:
            self.open_amount = min(goal, self.open_amount + move)
        else:
            self.open_amount = max(goal, self.open_amount - move)

    def open(mut self):
        self.opening = True

    def close(mut self):
        self.opening = False

    def open_progress(self) -> Float32:
        """The open amount through an ease-out cubic: fast start, soft landing."""
        var t = 1.0 - self.open_amount
        return 1.0 - t * t * t


def draw_widgets(mut fb: SoftwareFramebuffer, tree: WidgetTree):
    """Panels from the slate nine-slice, the gold focus frame faded by each
    widget's focus amount, labels, toggle lamps and slider tracks. While the
    tree opens or closes, everything slides up into place and fades."""
    var shown = tree.open_progress()
    if shown <= 0.001:
        return
    var lift = Int(OPEN_SLIDE * (1.0 - shown))
    var fade = UInt8(Int(255.0 * shown + 0.5))
    for index in range(len(tree.widgets)):
        ref widget = tree.widgets[index]
        var x = widget.x
        var y = widget.y + lift
        var dim = not widget.enabled
        if widget.kind != WIDGET_LABEL:
            fb.draw_nine_slice(SLICE_SLATE, x, y, widget.w, widget.h, UInt8(Int(fade) * (150 if dim else 235) // 255))
            if widget.pulse > 0:
                fb.draw_rectangle_rounded(x + 4, y + 4, widget.w - 8, widget.h - 8, 0.3, 120, 196, 186, UInt8(Int(fade) * Int(widget.pulse / 0.18 * 90.0) // 255))
            if widget.focus_amount > 0.01:
                fb.draw_nine_slice(SLICE_FOCUS, x - 3, y - 3, widget.w + 6, widget.h + 6, UInt8(Int(Float32(fade) * widget.focus_amount)))
        var lit = widget.focus_amount
        var red = UInt8(92) if dim else UInt8(214 + Int(41.0 * lit))
        var green = UInt8(105) if dim else UInt8(222 + Int(10.0 * lit))
        var blue = UInt8(109) if dim else UInt8(214 - Int(36.0 * lit))
        var text_y = y + max(0, widget.h // 2 - 9)
        if widget.kind == WIDGET_LABEL:
            fb.draw_text(widget.label, x, text_y, 16, 176, 214, 204, 0, widget.w)
        elif widget.kind == WIDGET_BUTTON:
            fb.draw_text(widget.label, x + widget.w // 2, text_y, 20, red, green, blue, 1, widget.w - 24)
        elif widget.kind == WIDGET_TOGGLE:
            fb.draw_text(widget.label, x + 18, text_y, 18, red, green, blue, 0, widget.w - 90)
            var lamp_x = x + widget.w - 50
            var lamp_y = y + widget.h // 2 - 9
            fb.draw_rectangle_rounded(lamp_x, lamp_y, 34, 18, 0.5, 30, 44, 54, fade)
            if widget.on:
                fb.draw_rectangle_rounded(lamp_x + 16, lamp_y + 2, 16, 14, 0.5, 98, 222, 200, fade)
            else:
                fb.draw_rectangle_rounded(lamp_x + 2, lamp_y + 2, 16, 14, 0.5, 92, 105, 109, fade)
        else:
            fb.draw_text(widget.label, x + 18, text_y, 18, red, green, blue, 0, widget.w // 2 - 24)
            var track_x = x + widget.w // 2
            var track_w = widget.w // 2 - 22
            var track_y = y + widget.h // 2 - 3
            fb.draw_rectangle_rounded(track_x, track_y, track_w, 6, 0.5, 37, 57, 66, fade)
            var filled = Int(Float32(track_w) * widget.value)
            fb.draw_rectangle_rounded(track_x, track_y, max(1, filled), 6, 0.5, 98, 222, 200, fade)
            fb.draw_circle(track_x + filled, track_y + 3, 8.0 + 2.0 * lit, 255, 211, 112, fade)
