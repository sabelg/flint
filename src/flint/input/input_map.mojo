"""Context-Stacked Declarative `InputMap`, Deadzone Curves, Chords, Jump Buffer, Coyote Time & Lockstep Packing (`flint.input.input_map`).

Features:
  - Priority-sorted `InputContext` stack (`push_context`, `pop_context`, `blocks_lower_all`,
    per-binding physical input consumption).
  - Typed actions: `ACTION_BOOL`, `ACTION_AXIS1D`, `ACTION_AXIS2D`.
  - `AxisCurve`: scaled-radial & axial deadzones, outer saturation, response power curve, invert.
  - Modifier chords (`Ctrl+S`, `Shift+Space`, `Alt+Enter`) that preempt and consume unmodified keys.
  - Action-game timing primitives: `consume_buffered` (`120 ms` jump buffer) and
    `set_grounded` / `coyote_valid` / `consume_coyote` (`100 ms` ledge coyote time).
  - Automatic last-used device detection (`Keyboard/Mouse`, `Xbox`, `PlayStation`) &
    `prompt_glyph_for(action)` lookup.
  - `save_json()` / `load_json()` runtime rebinding persistence & `pack_deterministic_tick()`.
"""
from std.collections import List
from std.math import sqrt, abs, isfinite
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote
from flint.app.engine_app import Resource
from flint.ecs.world import comp_hash64
from flint.platform.input import (
    InputState,
    PAD_A,
    PAD_B,
    PAD_X,
    PAD_Y,
    PAD_LB,
    PAD_RB,
    PAD_BACK,
    PAD_START,
    PAD_GUIDE,
    PAD_LEFT_STICK,
    PAD_RIGHT_STICK,
    PAD_UP,
    PAD_DOWN,
    PAD_LEFT,
    PAD_RIGHT,
    PAD_LT,
    PAD_RT,
    PAD_CONTROLS,
    PAD_AXIS_LEFT_X,
    PAD_AXIS_LEFT_Y,
    PAD_AXIS_LT,
    PAD_AXIS_RIGHT_X,
    PAD_AXIS_RIGHT_Y,
    PAD_AXIS_RT,
    PAD_AXIS_DPAD_X,
    PAD_AXIS_DPAD_Y,
)

# Action Kinds
comptime ACTION_BOOL: Int = 0
comptime ACTION_AXIS1D: Int = 1
comptime ACTION_AXIS2D: Int = 2

# Modifier Chord Bitmask
comptime MOD_NONE: Int = 0
comptime MOD_CTRL: Int = 1 << 0
comptime MOD_SHIFT: Int = 1 << 1
comptime MOD_ALT: Int = 1 << 2

# Binding Source Kinds
comptime BIND_KEY: Int = 0
comptime BIND_MOUSE_BUTTON: Int = 1
comptime BIND_MOUSE_DELTA_2D: Int = 2
comptime BIND_MOUSE_WHEEL_1D: Int = 3
comptime BIND_PAD_BUTTON: Int = 4
comptime BIND_PAD_AXIS_1D: Int = 5
comptime BIND_PAD_STICK_2D: Int = 6

# Deadzone Modes
comptime DEADZONE_SCALED_RADIAL: Int = 0
comptime DEADZONE_AXIAL: Int = 1

# Active Device Prompt Kinds
comptime INPUT_DEV_KBM: Int = 0
comptime INPUT_DEV_XBOX: Int = 1
comptime INPUT_DEV_SONY: Int = 2

# Modifier keycodes in InputState
comptime _KEY_LSHIFT: Int = 340
comptime _KEY_LCTRL: Int = 341
comptime _KEY_LALT: Int = 342
comptime _KEY_RSHIFT: Int = 344
comptime _KEY_RCTRL: Int = 345
comptime _KEY_RALT: Int = 346


@always_inline
def _popcount3(m: Int) -> Int:
    return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1)


@always_inline
def _pow_curve(v01: Float32, exponent: Float32) -> Float32:
    if v01 <= Float32(0.0):
        return 0.0
    if v01 >= Float32(1.0) or abs(exponent - Float32(1.0)) < Float32(1.0e-4):
        return min(Float32(1.0), v01)
    return v01 ** exponent


struct AxisCurve(Copyable, ImplicitlyCopyable, Movable):
    """Analog stick / trigger response curve with scaled-radial or axial deadzone."""
    var mode: Int
    var inner_deadzone: Float32
    var outer_deadzone: Float32
    var exponent: Float32
    var sensitivity: Float32
    var invert_x: Bool
    var invert_y: Bool

    def __init__(
        out self,
        mode: Int = DEADZONE_SCALED_RADIAL,
        inner_deadzone: Float32 = 0.12,
        outer_deadzone: Float32 = 0.98,
        exponent: Float32 = 1.0,
        sensitivity: Float32 = 1.0,
        invert_x: Bool = False,
        invert_y: Bool = False,
    ):
        self.mode = mode
        var idz = inner_deadzone if isfinite(inner_deadzone) else Float32(0.12)
        var odz = outer_deadzone if isfinite(outer_deadzone) else Float32(0.98)
        var exp_v = exponent if isfinite(exponent) else Float32(1.0)
        var sens_v = sensitivity if isfinite(sensitivity) else Float32(1.0)
        self.inner_deadzone = max(Float32(0.0), min(Float32(0.85), idz))
        self.outer_deadzone = max(self.inner_deadzone + Float32(0.05), min(Float32(1.0), odz))
        self.exponent = max(Float32(0.25), min(Float32(4.0), exp_v))
        self.sensitivity = sens_v
        self.invert_x = invert_x
        self.invert_y = invert_y

    def eval_1d(self, raw: Float32) -> Float32:
        if not isfinite(raw):
            return Float32(0.0)
        var mag = abs(raw)
        if mag <= self.inner_deadzone:
            return 0.0
        var sign = Float32(-1.0) if raw < Float32(0.0) else Float32(1.0)
        if self.invert_x:
            sign = -sign
        var norm = min(Float32(1.0), (mag - self.inner_deadzone) / max(Float32(1.0e-5), self.outer_deadzone - self.inner_deadzone))
        return sign * _pow_curve(norm, self.exponent) * self.sensitivity

    def eval_2d(self, raw_x: Float32, raw_y: Float32) -> Tuple[Float32, Float32]:
        if not isfinite(raw_x) or not isfinite(raw_y):
            return (Float32(0.0), Float32(0.0))
        var sx = -raw_x if self.invert_x else raw_x
        var sy = -raw_y if self.invert_y else raw_y
        if self.mode == DEADZONE_AXIAL:
            var ax = self._eval_single_no_inv(sx)
            var ay = self._eval_single_no_inv(sy)
            return (ax * self.sensitivity, ay * self.sensitivity)

        # Scaled radial deadzone: continuous direction preservation with zero step discontinuity
        var r = sqrt(sx * sx + sy * sy)
        if not isfinite(r):
            var m = max(abs(sx), abs(sy))
            if m <= Float32(0.0) or not isfinite(m):
                return (Float32(0.0), Float32(0.0))
            var ux = sx / m
            var uy = sy / m
            var ur = sqrt(ux * ux + uy * uy)
            if ur <= Float32(1.0e-6) or not isfinite(ur):
                return (Float32(0.0), Float32(0.0))
            return ((ux / ur) * self.sensitivity, (uy / ur) * self.sensitivity)
        if r <= self.inner_deadzone or r <= Float32(1.0e-6):
            return (0.0, 0.0)
        var norm = min(Float32(1.0), (r - self.inner_deadzone) / max(Float32(1.0e-5), self.outer_deadzone - self.inner_deadzone))
        var curved = _pow_curve(norm, self.exponent) * self.sensitivity
        var inv_r = Float32(1.0) / r
        return (sx * inv_r * curved, sy * inv_r * curved)

    @always_inline
    def _eval_single_no_inv(self, v: Float32) -> Float32:
        if not isfinite(v):
            return Float32(0.0)
        var mag = abs(v)
        if mag <= self.inner_deadzone:
            return 0.0
        var sgn = Float32(-1.0) if v < Float32(0.0) else Float32(1.0)
        var norm = min(Float32(1.0), (mag - self.inner_deadzone) / max(Float32(1.0e-5), self.outer_deadzone - self.inner_deadzone))
        return sgn * _pow_curve(norm, self.exponent)


struct InputBinding(Copyable, ImplicitlyCopyable, Movable):
    """Maps a physical key/chord/button/axis to a named action within an `InputContext`."""
    var action_name: String
    var action_index: Int
    var source_kind: Int
    var code: Int          # keycode, mouse button, pad button, pad axis, or stick index (0=L, 1=R, 2=DPad)
    var modifiers: Int     # MOD_CTRL | MOD_SHIFT | MOD_ALT (for BIND_KEY)
    var scale_x: Float32
    var scale_y: Float32
    var consume_input: Bool
    var curve: AxisCurve

    def __init__(
        out self,
        action_name: String,
        source_kind: Int,
        code: Int,
        modifiers: Int = MOD_NONE,
        scale_x: Float32 = 1.0,
        scale_y: Float32 = 0.0,
        consume_input: Bool = True,
        curve: AxisCurve = AxisCurve(),
        action_index: Int = -1,
    ):
        self.action_name = action_name
        self.action_index = action_index
        self.source_kind = source_kind
        self.code = code
        self.modifiers = modifiers
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.consume_input = consume_input
        self.curve = curve


struct InputContext(Copyable, Movable):
    """Named input mapping layer with stack priority and optional lower-context blocking."""
    var name: String
    var priority: Int
    var blocks_lower_all: Bool
    var bindings: List[InputBinding]

    def __init__(out self, name: String, priority: Int = 0, blocks_lower_all: Bool = False):
        self.name = name
        self.priority = priority
        self.blocks_lower_all = blocks_lower_all
        self.bindings = List[InputBinding]()

    def bind_key(
        mut self,
        action: String,
        keycode: Int,
        modifiers: Int = MOD_NONE,
        scale_x: Float32 = 1.0,
        scale_y: Float32 = 0.0,
        consume: Bool = True,
    ):
        self.bindings.append(InputBinding(action, BIND_KEY, keycode, modifiers, scale_x, scale_y, consume))

    def bind_mouse_button(mut self, action: String, button: Int, consume: Bool = True):
        self.bindings.append(InputBinding(action, BIND_MOUSE_BUTTON, button, MOD_NONE, 1.0, 0.0, consume))

    def bind_mouse_look(mut self, action: String, sensitivity: Float32 = 1.0, invert_y: Bool = False):
        var c = AxisCurve(DEADZONE_AXIAL, 0.0, 1.0, 1.0, sensitivity, False, invert_y)
        self.bindings.append(InputBinding(action, BIND_MOUSE_DELTA_2D, 0, MOD_NONE, 1.0, 1.0, False, c))

    def bind_pad_button(
        mut self,
        action: String,
        pad_btn: Int,
        scale_x: Float32 = 1.0,
        scale_y: Float32 = 0.0,
        consume: Bool = True,
    ):
        self.bindings.append(InputBinding(action, BIND_PAD_BUTTON, pad_btn, MOD_NONE, scale_x, scale_y, consume))

    def bind_pad_axis_1d(mut self, action: String, pad_axis: Int, curve: AxisCurve = AxisCurve(DEADZONE_AXIAL, 0.05, 0.98)):
        self.bindings.append(InputBinding(action, BIND_PAD_AXIS_1D, pad_axis, MOD_NONE, 1.0, 0.0, True, curve))

    def bind_pad_stick_2d(mut self, action: String, stick_idx: Int = 0, curve: AxisCurve = AxisCurve()):
        self.bindings.append(InputBinding(action, BIND_PAD_STICK_2D, stick_idx, MOD_NONE, 1.0, 1.0, True, curve))


struct ActionState(Copyable, Movable):
    var name: String
    var kind: Int
    var down: Bool
    var pressed: Bool
    var released: Bool
    var value_x: Float32
    var value_y: Float32
    var last_pressed_time: Float64
    var has_unconsumed_press: Bool

    def __init__(out self, name: String, kind: Int = ACTION_BOOL):
        self.name = name
        self.kind = kind
        self.down = False
        self.pressed = False
        self.released = False
        self.value_x = 0.0
        self.value_y = 0.0
        self.last_pressed_time = -1000.0
        self.has_unconsumed_press = False


@fieldwise_init
struct InputPromptGlyph(Copyable, ImplicitlyCopyable, Movable):
    """Device-adaptive UI prompt glyph descriptor."""
    var device_kind: Int
    var label: String
    var symbol_id: Int
    var color_rgba8: UInt32


def _keycode_display_name(key: Int) -> String:
    if key >= 65 and key <= 90:
        return chr(key)
    if key >= 48 and key <= 57:
        return chr(key)
    if key == 32:
        return "Space"
    if key == 256:
        return "Esc"
    if key == 257:
        return "Enter"
    if key == 258:
        return "Tab"
    if key == 259:
        return "Backspace"
    if key == 262:
        return "Right"
    if key == 263:
        return "Left"
    if key == 264:
        return "Down"
    if key == 265:
        return "Up"
    return "Key" + String(key)


def _pad_control_display(ctrl: Int, sony: Bool) -> Tuple[String, UInt32]:
    if ctrl == PAD_A:
        return ("Cross" if sony else "A", UInt32(0xFFFFAA84) if sony else UInt32(0xFF54B056))
    if ctrl == PAD_B:
        return ("Circle" if sony else "B", UInt32(0xFF6E6EF0) if sony else UInt32(0xFF464ED6))
    if ctrl == PAD_X:
        return ("Square" if sony else "X", UInt32(0xFFD69BE7) if sony else UInt32(0xFFD87F3F))
    if ctrl == PAD_Y:
        return ("Triangle" if sony else "Y", UInt32(0xFFB5D35E) if sony else UInt32(0xFF3AB5E8))
    if ctrl == PAD_LB:
        return ("L1" if sony else "LB", UInt32(0xFFD0D6E0))
    if ctrl == PAD_RB:
        return ("R1" if sony else "RB", UInt32(0xFFD0D6E0))
    if ctrl == PAD_LT:
        return ("L2" if sony else "LT", UInt32(0xFFD0D6E0))
    if ctrl == PAD_RT:
        return ("R2" if sony else "RT", UInt32(0xFFD0D6E0))
    if ctrl == PAD_START:
        return ("Options" if sony else "Menu", UInt32(0xFFD0D6E0))
    if ctrl == PAD_BACK:
        return ("Create" if sony else "View", UInt32(0xFFD0D6E0))
    return ("Pad" + String(ctrl), UInt32(0xFFD0D6E0))


struct InputMap(Resource, Copyable, Movable):
    """Context-stacked declarative action & axis mapper."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.input.InputMap")
    var actions: List[ActionState]
    var contexts: List[InputContext]
    var active_stack: List[String]
    var elapsed_time: Float64
    var grounded: Bool
    var last_grounded_time: Float64
    var coyote_consumed: Bool
    var active_device: Int
    var pad_slot: Int

    def __init__(out self):
        self.actions = List[ActionState]()
        self.contexts = List[InputContext]()
        self.active_stack = List[String]()
        self.elapsed_time = 0.0
        self.grounded = False
        self.last_grounded_time = -1000.0
        self.coyote_consumed = False
        self.active_device = INPUT_DEV_KBM
        self.pad_slot = 0

    # ---- Action & Context Registration ----

    def register_action(mut self, name: String, kind: Int = ACTION_BOOL) -> Int:
        for i in range(len(self.actions)):
            if self.actions[i].name == name:
                self.actions[i].kind = kind
                return i
        var idx = len(self.actions)
        self.actions.append(ActionState(name, kind))
        for ci in range(len(self.contexts)):
            for bi in range(len(self.contexts[ci].bindings)):
                if self.contexts[ci].bindings[bi].action_name == name:
                    self.contexts[ci].bindings[bi].action_index = idx
        return idx

    def _action_index(self, name: String) -> Int:
        for i in range(len(self.actions)):
            if self.actions[i].name == name:
                return i
        return -1

    def _resolve_context_bindings(self, mut ctx: InputContext):
        for bi in range(len(ctx.bindings)):
            ctx.bindings[bi].action_index = self._action_index(ctx.bindings[bi].action_name)

    def add_context(mut self, var ctx: InputContext):
        self._resolve_context_bindings(ctx)
        for i in range(len(self.contexts)):
            if self.contexts[i].name == ctx.name:
                self.contexts[i] = ctx^
                return
        self.contexts.append(ctx^)

    def _context_index(self, name: String) -> Int:
        for i in range(len(self.contexts)):
            if self.contexts[i].name == name:
                return i
        return -1

    def push_context(mut self, name: String):
        for i in range(len(self.active_stack)):
            if self.active_stack[i] == name:
                _ = self.active_stack.pop(i)
                break
        self.active_stack.append(name)

    def pop_context(mut self, name: String = "") -> Bool:
        if len(self.active_stack) == 0:
            return False
        if name.byte_length() == 0:
            _ = self.active_stack.pop()
            return True
        var idx = len(self.active_stack) - 1
        while idx >= 0:
            if self.active_stack[idx] == name:
                _ = self.active_stack.pop(idx)
                return True
            idx -= 1
        return False

    def rebind_key(mut self, context_name: String, action_name: String, new_key: Int, new_modifiers: Int = MOD_NONE) -> Bool:
        var ci = self._context_index(context_name)
        if ci < 0:
            return False
        for bi in range(len(self.contexts[ci].bindings)):
            if self.contexts[ci].bindings[bi].action_name == action_name and self.contexts[ci].bindings[bi].source_kind == BIND_KEY:
                self.contexts[ci].bindings[bi].code = new_key
                self.contexts[ci].bindings[bi].modifiers = new_modifiers
                return True
        self.contexts[ci].bind_key(action_name, new_key, new_modifiers)
        var last_bi = len(self.contexts[ci].bindings) - 1
        self.contexts[ci].bindings[last_bi].action_index = self._action_index(action_name)
        return True

    def rebind_pad_button(mut self, context_name: String, action_name: String, new_pad_btn: Int) -> Bool:
        var ci = self._context_index(context_name)
        if ci < 0:
            return False
        for bi in range(len(self.contexts[ci].bindings)):
            if self.contexts[ci].bindings[bi].action_name == action_name and self.contexts[ci].bindings[bi].source_kind == BIND_PAD_BUTTON:
                self.contexts[ci].bindings[bi].code = new_pad_btn
                return True
        self.contexts[ci].bind_pad_button(action_name, new_pad_btn)
        var last_bi = len(self.contexts[ci].bindings) - 1
        self.contexts[ci].bindings[last_bi].action_index = self._action_index(action_name)
        return True

    # ---- Evaluation ----

    def update(mut self, input: InputState, dt: Float32):
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(10.0))
        self.elapsed_time += Float64(safe_dt)
        if self.grounded:
            self.last_grounded_time = self.elapsed_time

        # Detect active device for prompt glyphs
        var pad_act = input.any_pad_pressed()
        if not pad_act and self.pad_slot >= 0 and self.pad_slot < len(input.pads):
            for ax in range(6):
                var v = input.gamepad_axis(self.pad_slot, ax)
                if (ax == PAD_AXIS_LT or ax == PAD_AXIS_RT) and v > Float32(0.2):
                    pad_act = True
                elif ax != PAD_AXIS_LT and ax != PAD_AXIS_RT and abs(v) > Float32(0.25):
                    pad_act = True
        if pad_act:
            self.active_device = INPUT_DEV_SONY if input.pad_is_sony() else INPUT_DEV_XBOX
        elif input.any_key_pressed() or input.mouse_pressed(0) or input.mouse_pressed(1) or abs(input.mouse_delta_x()) + abs(input.mouse_delta_y()) > Float32(1.0):
            self.active_device = INPUT_DEV_KBM

        # Snapshot previous Bool states & clear frame accumulators
        var prev_down = List[Bool](length=len(self.actions), fill=False)
        var unbounded_axis = List[Bool](length=len(self.actions), fill=False)
        for i in range(len(self.actions)):
            prev_down[i] = self.actions[i].down
            self.actions[i].down = False
            self.actions[i].pressed = False
            self.actions[i].released = False
            self.actions[i].value_x = 0.0
            self.actions[i].value_y = 0.0

        # Active modifier state
        var cur_mods = MOD_NONE
        if input.key_down(_KEY_LCTRL) or input.key_down(_KEY_RCTRL):
            cur_mods |= MOD_CTRL
        if input.key_down(_KEY_LSHIFT) or input.key_down(_KEY_RSHIFT):
            cur_mods |= MOD_SHIFT
        if input.key_down(_KEY_LALT) or input.key_down(_KEY_RALT):
            cur_mods |= MOD_ALT

        # Consumed physical inputs across priority-sorted contexts
        var consumed_keys = Array[Bool, 512](fill=False)
        var consumed_mouse = Array[Bool, 8](fill=False)
        var consumed_pad_btns = Array[Bool, PAD_CONTROLS](fill=False)
        var consumed_pad_axes = Array[Bool, 8](fill=False)
        var consumed_sticks = Array[Bool, 3](fill=False)

        # Sort active contexts descending by (priority, stack_pos)
        var ordered_ctx = List[Int]()
        for s_pos in range(len(self.active_stack)):
            var ci = self._context_index(self.active_stack[s_pos])
            if ci >= 0:
                ordered_ctx.append(ci)

        for i in range(1, len(ordered_ctx)):
            var key_ci = ordered_ctx[i]
            var key_pri = self.contexts[key_ci].priority
            var j = i - 1
            # Higher priority first; for equal priority, later in active_stack (LIFO top) first!
            while j >= 0 and self.contexts[ordered_ctx[j]].priority <= key_pri:
                ordered_ctx[j + 1] = ordered_ctx[j]
                j -= 1
            ordered_ctx[j + 1] = key_ci

        for oci in range(len(ordered_ctx)):
            var ci = ordered_ctx[oci]
            ref ctx = self.contexts[ci]

            # Evaluate modifier chords in descending popcount specificity (3 -> 2 -> 1 -> 0)
            # so e.g. Ctrl+Shift+Z is never stolen by Ctrl+Z regardless of registration order.
            for pass_idx in range(4):
                var target_pop = 3 - pass_idx
                for bi in range(len(ctx.bindings)):
                    ref b = ctx.bindings[bi]
                    var mod_pop = 0
                    if b.source_kind == BIND_KEY and b.modifiers != MOD_NONE:
                        var m = b.modifiers
                        while m != 0:
                            mod_pop += m & 1
                            m >>= 1
                        if mod_pop > 3:
                            mod_pop = 3
                    if mod_pop != target_pop:
                        continue

                    var ai = b.action_index
                    if ai < 0 or ai >= len(self.actions):
                        ai = self._action_index(b.action_name)
                        if ai < 0:
                            continue

                    if b.source_kind == BIND_KEY:
                        var k = b.code
                        if k < 0 or k >= 512 or consumed_keys[k]:
                            continue
                        # Exact modifier match required when chord specifies modifiers;
                        # unmodified bindings still work unless the key itself was consumed by a chord!
                        if b.modifiers != MOD_NONE and (cur_mods & b.modifiers) != b.modifiers:
                            continue
                        if input.key_down(k):
                            self.actions[ai].down = True
                            self.actions[ai].value_x += b.scale_x
                            self.actions[ai].value_y += b.scale_y
                            if b.consume_input:
                                consumed_keys[k] = True

                    elif b.source_kind == BIND_MOUSE_BUTTON:
                        var mb = b.code
                        if mb >= 0 and mb < 8 and not consumed_mouse[mb] and input.mouse_down(mb):
                            self.actions[ai].down = True
                            self.actions[ai].value_x += b.scale_x
                            if b.consume_input:
                                consumed_mouse[mb] = True

                    elif b.source_kind == BIND_MOUSE_DELTA_2D:
                        var raw_dx = input.mouse_delta_x()
                        var raw_dy = input.mouse_delta_y()
                        if not isfinite(raw_dx):
                            raw_dx = Float32(0.0)
                        if not isfinite(raw_dy):
                            raw_dy = Float32(0.0)
                        var dx = -raw_dx if b.curve.invert_x else raw_dx
                        var dy = -raw_dy if b.curve.invert_y else raw_dy
                        var mx = dx * b.curve.sensitivity * b.scale_x
                        var my = dy * b.curve.sensitivity * b.scale_y
                        if isfinite(mx):
                            self.actions[ai].value_x += mx
                        if isfinite(my):
                            self.actions[ai].value_y += my
                        if (isfinite(mx) and abs(mx) > Float32(0.0)) or (isfinite(my) and abs(my) > Float32(0.0)):
                            unbounded_axis[ai] = True

                    elif b.source_kind == BIND_MOUSE_WHEEL_1D:
                        self.actions[ai].value_x += b.curve.eval_1d(input.mouse_wheel())

                    elif b.source_kind == BIND_PAD_BUTTON:
                        var pb = b.code
                        if pb >= 0 and pb < PAD_CONTROLS and not consumed_pad_btns[pb] and input.gamepad_down(self.pad_slot, pb):
                            self.actions[ai].down = True
                            self.actions[ai].value_x += b.scale_x
                            self.actions[ai].value_y += b.scale_y
                            if b.consume_input:
                                consumed_pad_btns[pb] = True

                    elif b.source_kind == BIND_PAD_AXIS_1D:
                        var ax = b.code
                        if ax >= 0 and ax < 8 and not consumed_pad_axes[ax]:
                            var raw = input.gamepad_axis(self.pad_slot, ax)
                            var ev = b.curve.eval_1d(raw) * b.scale_x
                            if abs(ev) > Float32(1.0e-5):
                                self.actions[ai].value_x += ev
                                if ev > Float32(0.25):
                                    self.actions[ai].down = True
                                if b.consume_input:
                                    consumed_pad_axes[ax] = True

                    elif b.source_kind == BIND_PAD_STICK_2D:
                        var st_idx = b.code
                        if st_idx >= 0 and st_idx < 3 and not consumed_sticks[st_idx]:
                            var rx: Float32
                            var ry: Float32
                            if st_idx == 0:
                                rx = input.gamepad_axis(self.pad_slot, PAD_AXIS_LEFT_X)
                                ry = input.gamepad_axis(self.pad_slot, PAD_AXIS_LEFT_Y)
                            elif st_idx == 1:
                                rx = input.gamepad_axis(self.pad_slot, PAD_AXIS_RIGHT_X)
                                ry = input.gamepad_axis(self.pad_slot, PAD_AXIS_RIGHT_Y)
                            else:
                                rx = input.gamepad_axis(self.pad_slot, PAD_AXIS_DPAD_X)
                                ry = input.gamepad_axis(self.pad_slot, PAD_AXIS_DPAD_Y)
                            var curved = b.curve.eval_2d(rx, ry)
                            if abs(curved[0]) + abs(curved[1]) > Float32(1.0e-5):
                                self.actions[ai].value_x += curved[0]
                                self.actions[ai].value_y += curved[1]
                                if b.consume_input:
                                    consumed_sticks[st_idx] = True

            if ctx.blocks_lower_all:
                break

        # Finalize edges, clamping, and jump-buffer timestamps
        for i in range(len(self.actions)):
            var k = self.actions[i].kind
            if k == ACTION_BOOL:
                var now_d = self.actions[i].down
                var was_d = prev_down[i]
                if now_d and not was_d:
                    self.actions[i].pressed = True
                    self.actions[i].last_pressed_time = self.elapsed_time
                    self.actions[i].has_unconsumed_press = True
                elif (not now_d) and was_d:
                    self.actions[i].released = True
                self.actions[i].value_x = Float32(1.0) if now_d else Float32(0.0)
            elif k == ACTION_AXIS1D:
                if not isfinite(self.actions[i].value_x):
                    self.actions[i].value_x = Float32(0.0)
                if not unbounded_axis[i]:
                    self.actions[i].value_x = max(Float32(-1.0), min(Float32(1.0), self.actions[i].value_x))
            elif k == ACTION_AXIS2D:
                if not isfinite(self.actions[i].value_x):
                    self.actions[i].value_x = Float32(0.0)
                if not isfinite(self.actions[i].value_y):
                    self.actions[i].value_y = Float32(0.0)
                if not unbounded_axis[i]:
                    var vx = self.actions[i].value_x
                    var vy = self.actions[i].value_y
                    var mag = sqrt(vx * vx + vy * vy)
                    if not isfinite(mag):
                        var m = max(abs(vx), abs(vy))
                        if m > Float32(0.0) and isfinite(m):
                            var ux = vx / m
                            var uy = vy / m
                            var ur = sqrt(ux * ux + uy * uy)
                            if ur > Float32(1.0e-6) and isfinite(ur):
                                self.actions[i].value_x = ux / ur
                                self.actions[i].value_y = uy / ur
                            else:
                                self.actions[i].value_x = Float32(0.0)
                                self.actions[i].value_y = Float32(0.0)
                    elif mag > Float32(1.0):
                        var inv = Float32(1.0) / mag
                        self.actions[i].value_x = vx * inv
                        self.actions[i].value_y = vy * inv

    # ---- Queries, Jump Buffer (`120 ms`) & Coyote Time (`100 ms`) ----

    def is_down(self, action_name: String) -> Bool:
        var i = self._action_index(action_name)
        return i >= 0 and self.actions[i].down

    def just_pressed(self, action_name: String) -> Bool:
        var i = self._action_index(action_name)
        return i >= 0 and self.actions[i].pressed

    def just_released(self, action_name: String) -> Bool:
        var i = self._action_index(action_name)
        return i >= 0 and self.actions[i].released

    def axis_1d(self, action_name: String) -> Float32:
        var i = self._action_index(action_name)
        return self.actions[i].value_x if i >= 0 else Float32(0.0)

    def axis_2d(self, action_name: String) -> Tuple[Float32, Float32]:
        var i = self._action_index(action_name)
        if i < 0:
            return (0.0, 0.0)
        return (self.actions[i].value_x, self.actions[i].value_y)

    def consume_buffered(mut self, action_name: String, window_sec: Float32 = 0.120) -> Bool:
        """Consumes a press buffered within the past `window_sec` (`120 ms` default jump buffer)."""
        var i = self._action_index(action_name)
        if i < 0 or not self.actions[i].has_unconsumed_press:
            return False
        var age = self.elapsed_time - self.actions[i].last_pressed_time
        if age <= Float64(window_sec) + 1.0e-6:
            self.actions[i].has_unconsumed_press = False
            return True
        return False

    def set_grounded(mut self, is_grounded: Bool):
        self.grounded = is_grounded
        if is_grounded:
            self.last_grounded_time = self.elapsed_time
            self.coyote_consumed = False

    def coyote_valid(self, window_sec: Float32 = 0.100) -> Bool:
        """Returns `True` while grounded OR within `window_sec` (`100 ms` default) after leaving ground."""
        if self.grounded:
            return True
        if self.coyote_consumed:
            return False
        return (self.elapsed_time - self.last_grounded_time) <= Float64(window_sec) + 1.0e-6

    def consume_coyote(mut self):
        self.coyote_consumed = True
        self.grounded = False

    # ---- Dynamic Prompt Glyph Lookup ----

    def prompt_glyph_for(self, action_name: String) -> InputPromptGlyph:
        var want_pad = self.active_device != INPUT_DEV_KBM
        var sony = self.active_device == INPUT_DEV_SONY

        var s_idx = len(self.active_stack) - 1
        while s_idx >= 0:
            var ci = self._context_index(self.active_stack[s_idx])
            if ci >= 0:
                for bi in range(len(self.contexts[ci].bindings)):
                    ref b = self.contexts[ci].bindings[bi]
                    if b.action_name != action_name:
                        continue
                    if want_pad and b.source_kind == BIND_PAD_BUTTON:
                        var pd = _pad_control_display(b.code, sony)
                        return InputPromptGlyph(self.active_device, pd[0], b.code, pd[1])
                    if (not want_pad) and b.source_kind == BIND_KEY:
                        var prefix = String()
                        if (b.modifiers & MOD_CTRL) != 0:
                            prefix += "Ctrl+"
                        if (b.modifiers & MOD_SHIFT) != 0:
                            prefix += "Shift+"
                        if (b.modifiers & MOD_ALT) != 0:
                            prefix += "Alt+"
                        return InputPromptGlyph(INPUT_DEV_KBM, prefix + _keycode_display_name(b.code), b.code, UInt32(0xFFE6EAF0))
            s_idx -= 1

        # Fallback: search any binding matching action_name
        for ci in range(len(self.contexts)):
            for bi in range(len(self.contexts[ci].bindings)):
                ref b = self.contexts[ci].bindings[bi]
                if b.action_name == action_name:
                    if b.source_kind == BIND_PAD_BUTTON:
                        var pd = _pad_control_display(b.code, sony)
                        return InputPromptGlyph(self.active_device, pd[0], b.code, pd[1])
                    if b.source_kind == BIND_KEY:
                        var prefix = String()
                        if (b.modifiers & MOD_CTRL) != 0:
                            prefix += "Ctrl+"
                        if (b.modifiers & MOD_SHIFT) != 0:
                            prefix += "Shift+"
                        if (b.modifiers & MOD_ALT) != 0:
                            prefix += "Alt+"
                        return InputPromptGlyph(INPUT_DEV_KBM, prefix + _keycode_display_name(b.code), b.code, UInt32(0xFFE6EAF0))
        return InputPromptGlyph(self.active_device, "?", -1, UInt32(0xFF8892A0))

    # ---- Deterministic Lockstep / Replay Tick Serialization ----

    def pack_deterministic_tick(self) -> List[UInt8]:
        """Packs all registered actions into a canonical deterministic byte frame:
        `[u16 action_count]` + per action:
          - `ACTION_BOOL`: `1 byte` (`bit0=down, bit1=pressed, bit2=released`)
          - `ACTION_AXIS1D`: `2 bytes` (`Int16` quantized `[-32767, +32767]`)
          - `ACTION_AXIS2D`: `4 bytes` (`2 x Int16` quantized `[-32767, +32767]`).
        """
        var out = List[UInt8]()
        var n = len(self.actions)
        out.append(UInt8(n & 0xFF))
        out.append(UInt8((n >> 8) & 0xFF))
        for i in range(n):
            ref a = self.actions[i]
            if a.kind == ACTION_BOOL:
                var bits = (1 if a.down else 0) | ((1 if a.pressed else 0) << 1) | ((1 if a.released else 0) << 2)
                out.append(UInt8(bits))
            elif a.kind == ACTION_AXIS1D:
                var q = Int16(max(-32767, min(32767, Int(a.value_x * Float32(32767.0)))))
                var u = UInt16(q)
                out.append(UInt8(u & 0xFF))
                out.append(UInt8((u >> 8) & 0xFF))
            else:
                var qx = Int16(max(-32767, min(32767, Int(a.value_x * Float32(32767.0)))))
                var qy = Int16(max(-32767, min(32767, Int(a.value_y * Float32(32767.0)))))
                var ux = UInt16(qx)
                var uy = UInt16(qy)
                out.append(UInt8(ux & 0xFF))
                out.append(UInt8((ux >> 8) & 0xFF))
                out.append(UInt8(uy & 0xFF))
                out.append(UInt8((uy >> 8) & 0xFF))
        return out^

    def unpack_deterministic_tick(mut self, pkt: List[UInt8]) -> Bool:
        if len(pkt) < 2:
            return False
        var n = Int(pkt[0]) | (Int(pkt[1]) << 8)
        if n != len(self.actions):
            return False
        var off = 2
        for i in range(n):
            var k = self.actions[i].kind
            if k == ACTION_BOOL:
                if off + 1 > len(pkt):
                    return False
                var b = Int(pkt[off])
                off += 1
                self.actions[i].down = (b & 1) != 0
                self.actions[i].pressed = (b & 2) != 0
                self.actions[i].released = (b & 4) != 0
                self.actions[i].value_x = Float32(1.0) if self.actions[i].down else Float32(0.0)
            elif k == ACTION_AXIS1D:
                if off + 2 > len(pkt):
                    return False
                var u = UInt16(pkt[off]) | (UInt16(pkt[off + 1]) << 8)
                off += 2
                self.actions[i].value_x = Float32(Int16(u)) / Float32(32767.0)
            else:
                if off + 4 > len(pkt):
                    return False
                var ux = UInt16(pkt[off]) | (UInt16(pkt[off + 1]) << 8)
                var uy = UInt16(pkt[off + 2]) | (UInt16(pkt[off + 3]) << 8)
                off += 4
                self.actions[i].value_x = Float32(Int16(ux)) / Float32(32767.0)
                self.actions[i].value_y = Float32(Int16(uy)) / Float32(32767.0)
        return off == len(pkt)

    # ---- JSON Persistence (Rebinding & Full Schema) ----

    @staticmethod
    def _fmt_f(v: Float32) -> String:
        if not isfinite(v):
            return "0.000"
        var neg = v < Float32(0.0)
        var av64 = abs(Float64(v))
        if av64 > 1.0e15:
            return ("-" if neg else "") + String(av64)
        var whole = Int(av64)
        var frac = Int((av64 - Float64(whole)) * 1000.0 + 0.5)
        if frac >= 1000:
            whole += 1
            frac = 0
        var fstr = String(frac)
        while fstr.byte_length() < 3:
            fstr = "0" + fstr
        return ("-" if neg else "") + String(whole) + "." + fstr

    def save_json(self) -> String:
        var out = String('{"version": 1, "actions": [')
        for i in range(len(self.actions)):
            if i > 0:
                out += ", "
            out += '{"name": ' + json_quote(self.actions[i].name) + ', "kind": ' + String(self.actions[i].kind) + "}"
        out += '], "stack": ['
        for i in range(len(self.active_stack)):
            if i > 0:
                out += ", "
            out += json_quote(self.active_stack[i])
        out += '], "contexts": ['
        for ci in range(len(self.contexts)):
            if ci > 0:
                out += ", "
            ref c = self.contexts[ci]
            out += '{"name": ' + json_quote(c.name)
            out += ', "priority": ' + String(c.priority)
            out += ', "blocks_lower_all": ' + ("true" if c.blocks_lower_all else "false")
            out += ', "bindings": ['
            for bi in range(len(c.bindings)):
                if bi > 0:
                    out += ", "
                ref b = c.bindings[bi]
                out += '{"action": ' + json_quote(b.action_name)
                out += ', "source_kind": ' + String(b.source_kind)
                out += ', "code": ' + String(b.code)
                out += ', "modifiers": ' + String(b.modifiers)
                out += ', "scale_x": ' + Self._fmt_f(b.scale_x)
                out += ', "scale_y": ' + Self._fmt_f(b.scale_y)
                out += ', "consume": ' + ("true" if b.consume_input else "false")
                out += ', "dz_mode": ' + String(b.curve.mode)
                out += ', "dz_in": ' + Self._fmt_f(b.curve.inner_deadzone)
                out += ', "dz_out": ' + Self._fmt_f(b.curve.outer_deadzone)
                out += ', "exp": ' + Self._fmt_f(b.curve.exponent)
                out += ', "sens": ' + Self._fmt_f(b.curve.sensitivity)
                out += "}"
            out += "]}"
        out += "]}"
        return out^

    def load_json(mut self, json_str: String) -> Bool:
        var bytes = List[UInt8]()
        for b in json_str.as_bytes():
            bytes.append(b)
        var doc = parse_json(bytes)
        if not doc.valid or doc.root < 0:
            return False

        var acts_arr = doc.object_get(doc.root, "actions")
        var ctxs_arr = doc.object_get(doc.root, "contexts")
        var stk_arr = doc.object_get(doc.root, "stack")
        if acts_arr < 0 or ctxs_arr < 0:
            return False

        self.actions.clear()
        self.contexts.clear()
        self.active_stack.clear()

        for i in range(doc.array_size(acts_arr)):
            var ao = doc.at(acts_arr, i)
            var nm = doc.text(doc.object_get(ao, "name"), "")
            var kd = doc.integer(doc.object_get(ao, "kind"), ACTION_BOOL)
            if nm.byte_length() > 0:
                _ = self.register_action(nm, kd)

        for ci in range(doc.array_size(ctxs_arr)):
            var co = doc.at(ctxs_arr, ci)
            var c_name = doc.text(doc.object_get(co, "name"), "")
            var c_pri = doc.integer(doc.object_get(co, "priority"), 0)
            var c_blk = doc.boolean(doc.object_get(co, "blocks_lower_all"), False)
            var ctx = InputContext(c_name, c_pri, c_blk)
            var b_arr = doc.object_get(co, "bindings")
            if b_arr >= 0:
                for bi in range(doc.array_size(b_arr)):
                    var bo = doc.at(b_arr, bi)
                    var act = doc.text(doc.object_get(bo, "action"), "")
                    var sk = doc.integer(doc.object_get(bo, "source_kind"), BIND_KEY)
                    var cd = doc.integer(doc.object_get(bo, "code"), 0)
                    var md = doc.integer(doc.object_get(bo, "modifiers"), MOD_NONE)
                    var sx = Float32(doc.number(doc.object_get(bo, "scale_x"), 1.0))
                    var sy = Float32(doc.number(doc.object_get(bo, "scale_y"), 0.0))
                    if not isfinite(sx):
                        sx = Float32(1.0)
                    if not isfinite(sy):
                        sy = Float32(0.0)
                    var cs = doc.boolean(doc.object_get(bo, "consume"), True)
                    var dz_in = Float32(doc.number(doc.object_get(bo, "dz_in"), 0.12))
                    var dz_out = Float32(doc.number(doc.object_get(bo, "dz_out"), 0.98))
                    var exp_v = Float32(doc.number(doc.object_get(bo, "exp"), 1.0))
                    var sens_v = Float32(doc.number(doc.object_get(bo, "sens"), 1.0))
                    var curve = AxisCurve(
                        doc.integer(doc.object_get(bo, "dz_mode"), DEADZONE_SCALED_RADIAL),
                        dz_in if isfinite(dz_in) else Float32(0.12),
                        dz_out if isfinite(dz_out) else Float32(0.98),
                        exp_v if isfinite(exp_v) else Float32(1.0),
                        sens_v if isfinite(sens_v) else Float32(1.0),
                    )
                    ctx.bindings.append(InputBinding(act, sk, cd, md, sx, sy, cs, curve))
            self.add_context(ctx^)

        if stk_arr >= 0:
            for si in range(doc.array_size(stk_arr)):
                var sn = doc.text(doc.at(stk_arr, si), "")
                if sn.byte_length() > 0:
                    self.push_context(sn)
        return True
