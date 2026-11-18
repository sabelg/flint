"""Explicit keyboard, mouse, and Linux joystick state."""
from std.ffi import external_call
from std.math import sqrt
from flint.platform.os import close_fd

# Named pad controls. The first eleven match the Linux xpad joystick layout, and the
# rest are built from axes so menus and actions can treat them as plain buttons.
comptime PAD_A: Int = 0
comptime PAD_B: Int = 1
comptime PAD_X: Int = 2
comptime PAD_Y: Int = 3
comptime PAD_LB: Int = 4
comptime PAD_RB: Int = 5
comptime PAD_BACK: Int = 6
comptime PAD_START: Int = 7
comptime PAD_GUIDE: Int = 8
comptime PAD_LEFT_STICK: Int = 9
comptime PAD_RIGHT_STICK: Int = 10
comptime PAD_UP: Int = 16
comptime PAD_DOWN: Int = 17
comptime PAD_LEFT: Int = 18
comptime PAD_RIGHT: Int = 19
comptime PAD_LT: Int = 20
comptime PAD_RT: Int = 21
comptime PAD_STICK_UP: Int = 22
comptime PAD_STICK_DOWN: Int = 23
comptime PAD_STICK_LEFT: Int = 24
comptime PAD_STICK_RIGHT: Int = 25
comptime PAD_CONTROLS: Int = 26
comptime PAD_AXIS_LEFT_X: Int = 0
comptime PAD_AXIS_LEFT_Y: Int = 1
comptime PAD_AXIS_LT: Int = 2
comptime PAD_AXIS_RIGHT_X: Int = 3
comptime PAD_AXIS_RIGHT_Y: Int = 4
comptime PAD_AXIS_RT: Int = 5
comptime PAD_AXIS_DPAD_X: Int = 6
comptime PAD_AXIS_DPAD_Y: Int = 7
comptime PAD_DEADZONE: Float32 = 0.12
comptime PAD_DIRECTION: Float32 = 0.6
comptime PAD_RETRY_POLLS: Int = 120
comptime MAX_PADS: Int = 2
comptime ENODEV: Int32 = 19
comptime JSIOCGNAME_64: UInt64 = 0x80406A13
comptime JSIOCGAXES: UInt64 = 0x80016A11
comptime JSIOCGBUTTONS: UInt64 = 0x80016A12
comptime EV_FF: Int = 0x15
comptime FF_RUMBLE: Int = 0x50
comptime EVIOCSFF: UInt64 = 0x40304580
comptime EVIOCRMFF: UInt64 = 0x40044581
comptime PAD_MIN_AXES: Int = 2
comptime PAD_MIN_BUTTONS: Int = 4
comptime PAD_MAX_BUTTONS: Int = 32  # keyboards and mice posing as joysticks report far more
comptime KEY_SLOTS: Int = 512
# Package 3.15: holding a mash-to-hold key repeats its press every this many ticks.
comptime MASH_REPEAT_TICKS: Int = 8


def remappable_control(control: Int) -> Bool:
    """Pad controls the bindings may move (package 3.15): the face buttons,
    shoulders, triggers, Back and the left stick's click. Start, Guide, the
    d-pad and the stick directions always mean themselves (menus need them)."""
    return (control >= PAD_A and control <= PAD_BACK) or control == PAD_LEFT_STICK or control == PAD_LT or control == PAD_RT


def identity_map(count: Int) -> List[Int]:
    var out = List[Int](capacity=count)
    for index in range(count):
        out.append(index)
    return out^


def is_gamepad(name: String, axes: Int, buttons: Int) -> Bool:
    """Whether a joystick node is a controller. Keyboards, mice and a pad's
    motion sensors can also appear as /dev/input/jsN and must not take a slot."""
    if "Motion Sensors" in name or "Touchpad" in name or "Mouse" in name or "Keyboard" in name:
        return False
    return axes >= PAD_MIN_AXES and buttons >= PAD_MIN_BUTTONS and buttons <= PAD_MAX_BUTTONS


struct PadState(Copyable, Movable):
    var fd: Int
    var sony_layout: Bool
    var axes: List[Float32]
    var raw_axes: List[Float32]
    var down: List[Bool]
    var pressed: List[Bool]
    var physical: List[Bool]  # package 3.15: the physical controls held (before the pad map)
    var latched: List[Bool]   # logical controls a toggle holds down
    var mash_ticks: List[Int] # logical controls: ticks held since the last press
    var ff_effect_id: Int
    var lt_axis_min: Int
    var lt_axis_max: Int
    var lt_bipolar: Bool
    var rt_axis_min: Int
    var rt_axis_max: Int
    var rt_bipolar: Bool

    def __init__(out self):
        self.fd = -1
        self.ff_effect_id = -1
        self.sony_layout = False
        self.axes = List[Float32](length=8, fill=0)
        self.raw_axes = List[Float32](length=8, fill=0)
        self.down = List[Bool](length=PAD_CONTROLS, fill=False)
        self.pressed = List[Bool](length=PAD_CONTROLS, fill=False)
        self.physical = List[Bool](length=PAD_CONTROLS, fill=False)
        self.latched = List[Bool](length=PAD_CONTROLS, fill=False)
        self.mash_ticks = List[Int](length=PAD_CONTROLS, fill=0)
        self.lt_axis_min = 0
        self.lt_axis_max = 32767
        self.lt_bipolar = False
        self.rt_axis_min = 0
        self.rt_axis_max = 32767
        self.rt_bipolar = False

    def configure_trigger_axis(mut self, axis: Int, min_val: Int, max_val: Int):
        if axis == PAD_AXIS_LT:
            self.lt_axis_min = min_val
            self.lt_axis_max = max(min_val + 1, max_val)
            self.lt_bipolar = min_val < 0
        elif axis == PAD_AXIS_RT:
            self.rt_axis_min = min_val
            self.rt_axis_max = max(min_val + 1, max_val)
            self.rt_bipolar = min_val < 0

    def set_control(mut self, control: Int, down: Bool):
        if down and not self.down[control]:
            self.pressed[control] = True
        self.down[control] = down

    def mapped_button(self, number: Int) -> Int:
        """The named control for a raw button number, or -1 if it has none."""
        if not self.sony_layout:
            return number if number <= PAD_RIGHT_STICK else -1
        if number == 0:
            return PAD_A
        if number == 1:
            return PAD_B
        if number == 2:
            return PAD_Y
        if number == 3:
            return PAD_X
        if number == 4:
            return PAD_LB
        if number == 5:
            return PAD_RB
        if number == 8:
            return PAD_BACK
        if number == 9:
            return PAD_START
        if number == 10:
            return PAD_GUIDE
        if number == 11:
            return PAD_LEFT_STICK
        if number == 12:
            return PAD_RIGHT_STICK
        return -1  # 6 and 7 repeat the triggers, which also arrive as axes.

    def set_mapped(mut self, control: Int, down: Bool, pad_map: List[Int], toggle: List[Bool]):
        """A physical control through the pad map (package 3.15): a toggled
        logical control flips its latch on each physical press and ignores
        releases; controls that never remap set themselves."""
        if not remappable_control(control):
            self.set_control(control, down)
            return
        var logical = pad_map[control] if control < len(pad_map) else control
        var was = self.physical[control]
        self.physical[control] = down
        if logical < len(toggle) and toggle[logical]:
            if down and not was:
                self.latched[logical] = not self.latched[logical]
                self.set_control(logical, self.latched[logical])
            return
        self.set_control(logical, down)

    def apply_event(mut self, kind: Int, number: Int, raw_value: Int):
        """One Linux joystick event: kind 1 is a button and kind 2 is an axis.
        No remapping, no toggles, the default dead zone."""
        self.apply_event_with(kind, number, raw_value, identity_map(PAD_CONTROLS), List[Bool](length=PAD_CONTROLS, fill=False), PAD_DEADZONE, PAD_DEADZONE)

    def apply_event_with(mut self, kind: Int, number: Int, raw_value: Int, pad_map: List[Int], toggle: List[Bool], deadzone_left: Float32, deadzone_right: Float32):
        """One event through the bindings (package 3.15): the pad map and
        toggles for buttons and triggers, a dead zone per stick."""
        if kind == 1:
            var control = self.mapped_button(number)
            if control >= 0:
                self.set_mapped(control, raw_value != 0, pad_map, toggle)
        elif kind == 2 and number < len(self.axes):
            if number == PAD_AXIS_LT or number == PAD_AXIS_RT:
                var trig: Float32
                if number == PAD_AXIS_LT:
                    if raw_value < 0 and not self.lt_bipolar:
                        self.lt_bipolar = True
                        self.lt_axis_min = -32767
                        self.lt_axis_max = 32767
                    if self.lt_bipolar:
                        var span = Float32(max(1, self.lt_axis_max - self.lt_axis_min))
                        trig = max(Float32(0.0), min(Float32(1.0), Float32(raw_value - self.lt_axis_min) / span))
                    else:
                        var span = Float32(max(1, self.lt_axis_max - self.lt_axis_min))
                        trig = max(Float32(0.0), min(Float32(1.0), Float32(raw_value - self.lt_axis_min) / span))
                else:
                    if raw_value < 0 and not self.rt_bipolar:
                        self.rt_bipolar = True
                        self.rt_axis_min = -32767
                        self.rt_axis_max = 32767
                    if self.rt_bipolar:
                        var span = Float32(max(1, self.rt_axis_max - self.rt_axis_min))
                        trig = max(Float32(0.0), min(Float32(1.0), Float32(raw_value - self.rt_axis_min) / span))
                    else:
                        var span = Float32(max(1, self.rt_axis_max - self.rt_axis_min))
                        trig = max(Float32(0.0), min(Float32(1.0), Float32(raw_value - self.rt_axis_min) / span))
                self.axes[number] = trig
                self.raw_axes[number] = trig
                if number == PAD_AXIS_LT:
                    self.set_mapped(PAD_LT, trig > 0.1, pad_map, toggle)
                elif number == PAD_AXIS_RT:
                    self.set_mapped(PAD_RT, trig > 0.1, pad_map, toggle)
            elif number == PAD_AXIS_LEFT_X or number == PAD_AXIS_LEFT_Y:
                var val = max(Float32(-1.0), min(Float32(1.0), Float32(raw_value) / 32767.0))
                self.raw_axes[number] = val
                var rx = self.raw_axes[PAD_AXIS_LEFT_X]
                var ry = self.raw_axes[PAD_AXIS_LEFT_Y]
                var r = sqrt(rx * rx + ry * ry)
                if r < deadzone_left:
                    self.axes[PAD_AXIS_LEFT_X] = 0.0
                    self.axes[PAD_AXIS_LEFT_Y] = 0.0
                else:
                    var scale = min(Float32(1.0), (r - deadzone_left) / max(Float32(0.0001), Float32(1.0) - deadzone_left)) / max(r, Float32(1e-6))
                    self.axes[PAD_AXIS_LEFT_X] = rx * scale
                    self.axes[PAD_AXIS_LEFT_Y] = ry * scale
                self.set_control(PAD_STICK_LEFT, self.axes[PAD_AXIS_LEFT_X] < -PAD_DIRECTION)
                self.set_control(PAD_STICK_RIGHT, self.axes[PAD_AXIS_LEFT_X] > PAD_DIRECTION)
                self.set_control(PAD_STICK_UP, self.axes[PAD_AXIS_LEFT_Y] < -PAD_DIRECTION)
                self.set_control(PAD_STICK_DOWN, self.axes[PAD_AXIS_LEFT_Y] > PAD_DIRECTION)
            elif number == PAD_AXIS_RIGHT_X or number == PAD_AXIS_RIGHT_Y:
                var val = max(Float32(-1.0), min(Float32(1.0), Float32(raw_value) / 32767.0))
                self.raw_axes[number] = val
                var rx = self.raw_axes[PAD_AXIS_RIGHT_X]
                var ry = self.raw_axes[PAD_AXIS_RIGHT_Y]
                var r = sqrt(rx * rx + ry * ry)
                if r < deadzone_right:
                    self.axes[PAD_AXIS_RIGHT_X] = 0.0
                    self.axes[PAD_AXIS_RIGHT_Y] = 0.0
                else:
                    var scale = min(Float32(1.0), (r - deadzone_right) / max(Float32(0.0001), Float32(1.0) - deadzone_right)) / max(r, Float32(1e-6))
                    self.axes[PAD_AXIS_RIGHT_X] = rx * scale
                    self.axes[PAD_AXIS_RIGHT_Y] = ry * scale
            elif number == PAD_AXIS_DPAD_X:
                var value = Float32(raw_value) / 32767.0
                self.axes[number] = 0 if abs(value) < PAD_DEADZONE else value
                self.raw_axes[number] = value
                self.set_control(PAD_LEFT, value < -0.5)
                self.set_control(PAD_RIGHT, value > 0.5)
            elif number == PAD_AXIS_DPAD_Y:
                var value = Float32(raw_value) / 32767.0
                self.axes[number] = 0 if abs(value) < PAD_DEADZONE else value
                self.raw_axes[number] = value
                self.set_control(PAD_UP, value < -0.5)
                self.set_control(PAD_DOWN, value > 0.5)
            else:
                var value = Float32(raw_value) / 32767.0
                self.axes[number] = value
                self.raw_axes[number] = value

    def clear(mut self):
        self.ff_effect_id = -1
        self.lt_axis_min = 0
        self.lt_axis_max = 32767
        self.lt_bipolar = False
        self.rt_axis_min = 0
        self.rt_axis_max = 32767
        self.rt_bipolar = False
        for index in range(len(self.axes)):
            self.axes[index] = 0
            self.raw_axes[index] = 0
        for index in range(PAD_CONTROLS):
            self.down[index] = False
            self.physical[index] = False
            self.latched[index] = False
            self.mash_ticks[index] = 0
            self.pressed[index] = False

    def set_rumble(mut self, weak_motor: Float32, strong_motor: Float32, duration_ms: Int = 1000):
        if self.fd < 0:
            return
        var effect = List[UInt8](length=48, fill=0)
        # type = FF_RUMBLE (0x0050)
        effect[0] = 0x50
        effect[1] = 0x00
        # id
        var id_val = Int16(self.ff_effect_id)
        effect[2] = UInt8(id_val & 0xFF)
        effect[3] = UInt8((id_val >> 8) & 0xFF)
        # replay.length at offset 10 in Linux struct ff_effect
        var len_val = UInt16(min(65535, max(0, duration_ms)))
        effect[10] = UInt8(len_val & 0xFF)
        effect[11] = UInt8((len_val >> 8) & 0xFF)
        # strong magnitude
        var strong_mag = UInt16(min(65535, max(0, Int(strong_motor * 65535.0))))
        effect[16] = UInt8(strong_mag & 0xFF)
        effect[17] = UInt8((strong_mag >> 8) & 0xFF)
        # weak magnitude
        var weak_mag = UInt16(min(65535, max(0, Int(weak_motor * 65535.0))))
        effect[18] = UInt8(weak_mag & 0xFF)
        effect[19] = UInt8((weak_mag >> 8) & 0xFF)

        var ret = external_call["ioctl", Int32](Int32(self.fd), EVIOCSFF, effect.unsafe_ptr())
        if ret == 0:
            var new_id = Int(Int16(UInt16(effect[2]) | (UInt16(effect[3]) << 8)))
            self.ff_effect_id = new_id
            var play_ev = List[UInt8](length=24, fill=0)
            play_ev[16] = 0x15
            play_ev[18] = UInt8(new_id & 0xFF)
            play_ev[19] = UInt8((new_id >> 8) & 0xFF)
            play_ev[20] = 1
            _ = external_call["write", Int64](Int32(self.fd), play_ev.unsafe_ptr(), Int64(24))

    def stop_rumble(mut self):
        if self.fd >= 0 and self.ff_effect_id >= 0:
            var stop_ev = List[UInt8](length=24, fill=0)
            stop_ev[16] = 0x15
            stop_ev[18] = UInt8(self.ff_effect_id & 0xFF)
            stop_ev[19] = UInt8((self.ff_effect_id >> 8) & 0xFF)
            stop_ev[20] = 0
            _ = external_call["write", Int64](Int32(self.fd), stop_ev.unsafe_ptr(), Int64(24))


struct InputState(Copyable, Movable):
    var keys: List[Bool]
    var pressed: List[Bool]
    var released: List[Bool]
    var mouse: List[Bool]
    var mouse_pressed_state: List[Bool]
    var mouse_released_state: List[Bool]
    var delta_x: Float32
    var delta_y: Float32
    var wheel: Float32
    var last_mouse_x: Float32
    var last_mouse_y: Float32
    var has_mouse_position: Bool
    var pads: List[PadState]
    var pad_paths: List[String]
    var pad_retry: Int
    # Package 3.15: the bindings, applied before anything reads or records
    # the input, so the simulation and a recording only see logical keys
    # (the default layout) whatever the player bound.
    var key_map: List[Int]        # physical key -> logical key
    var pad_map: List[Int]        # physical pad control -> logical control
    var toggle_keys: List[Bool]   # by logical key: a press holds it, the next lets go
    var toggle_pads: List[Bool]   # by logical control
    var mash_keys: List[Bool]     # by logical key: held, it presses again every MASH_REPEAT_TICKS
    var mash_pads: List[Bool]     # by logical control
    var physical_keys: List[Bool] # physical keys held
    var key_latched: List[Bool]   # logical keys a toggle holds down
    var key_mash_ticks: List[Int] # logical keys: ticks held since the last press
    var deadzone_left: Float32
    var deadzone_right: Float32
    var mouse_extra_down: Array[Bool, 8]
    var text_input_utf8: String

    def __init__(out self):
        self.keys = List[Bool](length=512, fill=False)
        self.pressed = List[Bool](length=512, fill=False)
        self.released = List[Bool](length=512, fill=False)
        self.mouse = List[Bool](length=3, fill=False)
        self.mouse_pressed_state = List[Bool](length=3, fill=False)
        self.mouse_released_state = List[Bool](length=3, fill=False)
        self.delta_x = 0
        self.delta_y = 0
        self.wheel = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.has_mouse_position = False
        self.pads = List[PadState](length=MAX_PADS, fill=PadState())
        self.pad_paths = List[String](length=MAX_PADS, fill="")
        self.pad_retry = 0
        self.key_map = identity_map(KEY_SLOTS)
        self.pad_map = identity_map(PAD_CONTROLS)
        self.toggle_keys = List[Bool](length=KEY_SLOTS, fill=False)
        self.toggle_pads = List[Bool](length=PAD_CONTROLS, fill=False)
        self.mash_keys = List[Bool](length=KEY_SLOTS, fill=False)
        self.mash_pads = List[Bool](length=PAD_CONTROLS, fill=False)
        self.physical_keys = List[Bool](length=KEY_SLOTS, fill=False)
        self.key_latched = List[Bool](length=KEY_SLOTS, fill=False)
        self.key_mash_ticks = List[Int](length=KEY_SLOTS, fill=0)
        self.deadzone_left = PAD_DEADZONE
        self.deadzone_right = PAD_DEADZONE
        self.mouse_extra_down = Array[Bool, 8](fill=False)
        self.text_input_utf8 = String()

    def begin_frame(mut self):
        for index in range(512):
            self.pressed[index] = False
            self.released[index] = False
        for index in range(3):
            self.mouse_pressed_state[index] = False
            self.mouse_released_state[index] = False
        for pad in range(MAX_PADS):
            for index in range(PAD_CONTROLS):
                self.pads[pad].pressed[index] = False
        self.delta_x = 0
        self.delta_y = 0
        self.wheel = 0
        self.text_input_utf8 = String()

    def push_text_utf8(mut self, text: String):
        self.text_input_utf8 += text

    def mute(mut self):
        """Nothing is held and nothing happened. The console does this every frame it
        is open, before any tick, so the simulation and a recording both see silence."""
        self.begin_frame()
        for index in range(512):
            self.keys[index] = False
            self.physical_keys[index] = False
            self.key_latched[index] = False
            self.key_mash_ticks[index] = 0
        for index in range(3):
            self.mouse[index] = False
        for index in range(8):
            self.mouse_extra_down[index] = False
        for pad in range(MAX_PADS):
            self.pads[pad].clear()

    def set_mapping(mut self, var key_map: List[Int], var pad_map: List[Int], var toggle_keys: List[Bool], var toggle_pads: List[Bool], var mash_keys: List[Bool], var mash_pads: List[Bool], deadzone_left: Float32, deadzone_right: Float32):
        """New bindings (package 3.15, gameplay/bindings.mojo `Bindings.apply`).
        Everything held or latched is let go first (a release this frame), so
        no key stays stuck down under its old meaning; a key still held
        counts again from its next press."""
        if len(key_map) != KEY_SLOTS or len(pad_map) != PAD_CONTROLS or len(toggle_keys) != KEY_SLOTS or len(toggle_pads) != PAD_CONTROLS or len(mash_keys) != KEY_SLOTS or len(mash_pads) != PAD_CONTROLS:
            return
        for index in range(KEY_SLOTS):
            if self.keys[index]:
                self.released[index] = True
            self.keys[index] = False
            self.physical_keys[index] = False
            self.key_latched[index] = False
            self.key_mash_ticks[index] = 0
        for pad in range(MAX_PADS):
            for control in range(PAD_CONTROLS):
                if remappable_control(control):
                    self.pads[pad].down[control] = False
                self.pads[pad].physical[control] = False
                self.pads[pad].latched[control] = False
                self.pads[pad].mash_ticks[control] = 0
        self.key_map = key_map^
        self.pad_map = pad_map^
        self.toggle_keys = toggle_keys^
        self.toggle_pads = toggle_pads^
        self.mash_keys = mash_keys^
        self.mash_pads = mash_pads^
        self.deadzone_left = deadzone_left
        self.deadzone_right = deadzone_right

    def prepare_tick(mut self):
        """Once per simulation tick, before its input is recorded (package
        3.15): a mash-to-hold key or control held down presses again every
        MASH_REPEAT_TICKS ticks after its real press."""
        for key in range(KEY_SLOTS):
            if not self.mash_keys[key]:
                continue
            if self.pressed[key]:
                self.key_mash_ticks[key] = 0
            elif self.keys[key]:
                self.key_mash_ticks[key] += 1
                if self.key_mash_ticks[key] >= MASH_REPEAT_TICKS:
                    self.key_mash_ticks[key] = 0
                    self.pressed[key] = True
            else:
                self.key_mash_ticks[key] = 0
        for pad in range(MAX_PADS):
            for control in range(PAD_CONTROLS):
                if not self.mash_pads[control]:
                    continue
                if self.pads[pad].pressed[control]:
                    self.pads[pad].mash_ticks[control] = 0
                elif self.pads[pad].down[control]:
                    self.pads[pad].mash_ticks[control] += 1
                    if self.pads[pad].mash_ticks[control] >= MASH_REPEAT_TICKS:
                        self.pads[pad].mash_ticks[control] = 0
                        self.pads[pad].pressed[control] = True
                else:
                    self.pads[pad].mash_ticks[control] = 0

    def set_key(mut self, physical: Int, down: Bool):
        """A physical key through the key map; a toggled logical key flips its
        latch on each physical press and ignores releases (package 3.15)."""
        if physical < 0 or physical >= 512:
            return
        var key = self.key_map[physical]
        var was = self.physical_keys[physical]
        self.physical_keys[physical] = down
        if self.toggle_keys[key]:
            if down and not was:
                self.key_latched[key] = not self.key_latched[key]
                self.set_logical_key(key, self.key_latched[key])
            return
        self.set_logical_key(key, down)

    def release_held(mut self):
        """The window lost the keyboard: every logical key held or latched is
        let go (a release this frame), whatever physical key held it."""
        for key in range(KEY_SLOTS):
            self.physical_keys[key] = False
            self.key_latched[key] = False
            self.key_mash_ticks[key] = 0
            if self.keys[key]:
                self.set_logical_key(key, False)

    def set_logical_key(mut self, key: Int, down: Bool):
        if key < 0 or key >= 512:
            return
        if down and not self.keys[key]:
            self.pressed[key] = True
        if not down and self.keys[key]:
            self.released[key] = True
        self.keys[key] = down

    def set_mouse(mut self, button: Int, down: Bool):
        if button < 0 or button >= 8:
            return
        self.mouse_extra_down[button] = down
        if button < 3:
            if down and not self.mouse[button]:
                self.mouse_pressed_state[button] = True
            if not down and self.mouse[button]:
                self.mouse_released_state[button] = True
            self.mouse[button] = down

    def move_mouse(mut self, x: Float32, y: Float32):
        if self.has_mouse_position:
            self.delta_x += x - self.last_mouse_x
            self.delta_y += y - self.last_mouse_y
        self.last_mouse_x = x
        self.last_mouse_y = y
        self.has_mouse_position = True

    def add_mouse_delta(mut self, x: Float32, y: Float32):
        """Relative motion from a locked pointer."""
        self.delta_x += x
        self.delta_y += y

    def key_down(self, key: Int) -> Bool:
        return key >= 0 and key < 512 and self.keys[key]

    def key_pressed(self, key: Int) -> Bool:
        return key >= 0 and key < 512 and self.pressed[key]

    def key_released(self, key: Int) -> Bool:
        return key >= 0 and key < 512 and self.released[key]

    def mouse_down(self, button: Int) -> Bool:
        if button >= 0 and button < 3:
            return self.mouse[button]
        if button >= 3 and button < 8:
            return self.mouse_extra_down[button]
        return False

    def mouse_pressed(self, button: Int) -> Bool:
        return button >= 0 and button < 3 and self.mouse_pressed_state[button]

    def mouse_delta_x(self) -> Float32:
        return self.delta_x

    def mouse_delta_y(self) -> Float32:
        return self.delta_y

    def mouse_wheel(self) -> Float32:
        return self.wheel

    def open_pads(mut self):
        """Claim joystick nodes for empty pad slots, so a controller can arrive at any time."""
        for slot in range(MAX_PADS):
            if self.pads[slot].fd >= 0:
                continue
            for number in range(4):
                var path = "/dev/input/js" + String(number)
                var taken = False
                for other in range(MAX_PADS):
                    if self.pads[other].fd >= 0 and self.pad_paths[other] == path:
                        taken = True
                if taken:
                    continue
                var fd = Int(external_call["open", Int32](path.as_c_string_span(), Int32(0 | 2048), Int32(0)))
                if fd < 0:
                    continue
                var name = List[UInt8](length=64, fill=0)
                _ = external_call["ioctl", Int32](Int32(fd), JSIOCGNAME_64, name.unsafe_ptr())
                var text = String()
                for byte in name:
                    if byte == 0:
                        break
                    text += chr(Int(byte))
                var axes = List[UInt8](length=1, fill=0)
                var buttons = List[UInt8](length=1, fill=0)
                _ = external_call["ioctl", Int32](Int32(fd), JSIOCGAXES, axes.unsafe_ptr())
                _ = external_call["ioctl", Int32](Int32(fd), JSIOCGBUTTONS, buttons.unsafe_ptr())
                if not is_gamepad(text, Int(axes[0]), Int(buttons[0])):
                    close_fd(fd)
                    continue
                self.pads[slot].fd = fd
                self.pad_paths[slot] = path
                self.pads[slot].sony_layout = "Sony" in text or "DualSense" in text or "DualShock" in text or "Wireless Controller" in text
                break

    def poll_pad(mut self, slot: Int):
        var event = List[UInt8](length=8, fill=0)
        while True:
            var count = Int(external_call["read", Int64](Int32(self.pads[slot].fd), event.unsafe_ptr(), Int64(8)))
            if count != 8:
                # A controller that went away reports ENODEV instead of "try again".
                var error = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
                if count < 0 and error[unsafe_offset=0] == ENODEV:
                    close_fd(self.pads[slot].fd)
                    self.pads[slot].fd = -1
                    self.pads[slot].clear()
                return
            var raw_value = Int16(UInt16(event[4]) | (UInt16(event[5]) << 8))
            self.pads[slot].apply_event_with(Int(event[6]) & 0x7F, Int(event[7]), Int(raw_value), self.pad_map, self.toggle_pads, self.deadzone_left, self.deadzone_right)

    def poll_gamepad(mut self):
        self.pad_retry -= 1
        if self.pad_retry <= 0:
            self.pad_retry = PAD_RETRY_POLLS
            self.open_pads()
        for slot in range(MAX_PADS):
            if self.pads[slot].fd >= 0:
                self.poll_pad(slot)
        self.promote_pads()

    def set_pad_rumble(mut self, slot: Int, weak: Float32, strong: Float32, duration_ms: Int = 1000):
        if slot >= 0 and slot < MAX_PADS:
            self.pads[slot].set_rumble(weak, strong, duration_ms)

    def promote_pads(mut self):
        """Keep the first connected pad in slot 0, which is the one play reads,
        so unplugging the first controller hands play to the second."""
        if self.pads[0].fd >= 0 or self.pads[1].fd < 0:
            return
        self.pads[0] = self.pads[1].copy()
        self.pad_paths[0] = self.pad_paths[1]
        self.pads[1] = PadState()
        self.pad_paths[1] = ""

    def gamepad_available(self, pad: Int = 0) -> Bool:
        return pad >= 0 and pad < MAX_PADS and self.pads[pad].fd >= 0

    def gamepad_axis(self, pad: Int, axis: Int) -> Float32:
        if pad < 0 or pad >= MAX_PADS or axis < 0 or axis >= len(self.pads[pad].axes):
            return 0
        return self.pads[pad].axes[axis]

    def gamepad_down(self, pad: Int, button: Int) -> Bool:
        return pad >= 0 and pad < MAX_PADS and button >= 0 and button < PAD_CONTROLS and self.pads[pad].down[button]

    def gamepad_pressed(self, pad: Int, button: Int) -> Bool:
        return pad >= 0 and pad < MAX_PADS and button >= 0 and button < PAD_CONTROLS and self.pads[pad].pressed[button]

    def any_key_pressed(self) -> Bool:
        for key in range(512):
            if self.pressed[key]:
                return True
        return False

    def any_pad_pressed(self) -> Bool:
        for pad in range(MAX_PADS):
            for button in range(PAD_CONTROLS):
                if self.pads[pad].pressed[button]:
                    return True
        return False

    def pad_is_sony(self) -> Bool:
        for pad in range(MAX_PADS):
            if self.pads[pad].fd >= 0 and self.pads[pad].sony_layout:
                return True
        return False

    def close(mut self):
        for slot in range(MAX_PADS):
            close_fd(self.pads[slot].fd)
            self.pads[slot].fd = -1
