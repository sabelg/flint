"""Display-free checks for the platform, input and app-lifecycle layer:
  (a) `InputState` typematic key repeat: delay, rate, retyped text, release, modifiers, rate 0,
      platform repeat presses (X11 / evdev) and `key_pressed_or_repeat`.
  (b) Layout shortcuts: `layout_shortcut_code` (AZERTY letters, punctuation, non-Latin fallback)
      and `InputState.layout_key_map` routing a physical key to the code its label names.
  (c) Pointer enter/leave: leaving clears the position (no hover), entering gives no delta.
  (d) Pointer coordinates in framebuffer pixels after a resize (Wayland `map_pointer`, X11
      ConfigureNotify), at content scale 1 and 2.
  (e) X11: AltGr (Mod5) text from core-map columns 5/6, and server auto-repeat kept as a press.
  (f) One-shot SIGINT/SIGTERM quit flag (`install_quit_signal_handlers` / `quit_signal_received`).
"""
from std.ffi import external_call
from std.testing import assert_true, assert_equal, assert_false
from flint.platform.input import InputState, layout_shortcut_code, POINTER_OUTSIDE
from flint.platform.os import install_quit_signal_handlers, quit_signal_received
from flint.platform.wayland import WaylandWindow, wayland_layout_key_map
from flint.platform.x11 import X11Window, x11_put16, x11_put32
from flint.platform.xkb import XkbKeymap

comptime KEY_A: Int = 65
comptime KEY_Q: Int = 81
comptime KEY_BACKSPACE: Int = 259
comptime KEY_LEFT_SHIFT: Int = 340


def _feq(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < Float32(0.01)


def test_key_repeat_synthesizer() raises:
    var inp = InputState()
    inp.set_repeat_info(25, 500)
    var t0 = 100.0
    inp.begin_frame()
    inp.set_key(KEY_BACKSPACE, True)
    inp.set_repeat_text(KEY_BACKSPACE, "")
    inp.tick_key_repeat(t0)
    assert_true(inp.key_pressed(KEY_BACKSPACE))
    assert_true(inp.key_pressed_or_repeat(KEY_BACKSPACE))
    assert_false(inp.key_repeated(KEY_BACKSPACE))
    # Held, before the delay: nothing.
    inp.begin_frame()
    inp.tick_key_repeat(t0 + 0.49)
    assert_false(inp.key_pressed_or_repeat(KEY_BACKSPACE))
    # After the delay: one repeat, not a press.
    inp.begin_frame()
    inp.tick_key_repeat(t0 + 0.51)
    assert_true(inp.key_repeated(KEY_BACKSPACE))
    assert_true(inp.key_pressed_or_repeat(KEY_BACKSPACE))
    assert_false(inp.key_pressed(KEY_BACKSPACE))
    # 25 Hz: the next one 40 ms later, and a 120 ms frame catches up three.
    inp.begin_frame()
    inp.tick_key_repeat(t0 + 0.53)
    assert_false(inp.key_repeated(KEY_BACKSPACE))
    inp.begin_frame()
    inp.tick_key_repeat(t0 + 0.65)
    assert_true(inp.key_repeated(KEY_BACKSPACE))
    assert_equal(inp.repeat_fired_count, 3)
    # Released: no more repeats.
    inp.begin_frame()
    inp.set_key(KEY_BACKSPACE, False)
    inp.tick_key_repeat(t0 + 2.0)
    assert_false(inp.key_pressed_or_repeat(KEY_BACKSPACE))
    inp.begin_frame()
    inp.tick_key_repeat(t0 + 3.0)
    assert_false(inp.key_repeated(KEY_BACKSPACE))


def test_key_repeat_retypes_text() raises:
    var inp = InputState()
    inp.set_repeat_info(10, 300)
    inp.begin_frame()
    inp.set_key(KEY_A, True)
    inp.push_text_utf8("é")
    inp.set_repeat_text(KEY_A, "é")
    inp.tick_key_repeat(1.0)
    assert_equal(inp.text_input_utf8, "é")
    inp.begin_frame()
    inp.tick_key_repeat(1.31)
    assert_equal(inp.text_input_utf8, "é")
    inp.begin_frame()
    inp.tick_key_repeat(1.52)  # 1.3 + 0.1 + 0.1: two due this frame
    assert_equal(inp.text_input_utf8, "éé")
    # A long stall does not burst stale repeats: at most KEY_REPEAT_MAX_PER_FRAME, then rescheduled.
    inp.begin_frame()
    inp.tick_key_repeat(60.0)
    assert_true(inp.repeat_fired_count <= 8)
    inp.begin_frame()
    inp.tick_key_repeat(60.05)
    assert_false(inp.key_repeated(KEY_A))


def test_key_repeat_modifiers_and_rate_zero() raises:
    var inp = InputState()
    inp.set_repeat_info(25, 100)
    inp.begin_frame()
    inp.set_key(KEY_LEFT_SHIFT, True)
    inp.tick_key_repeat(0.0)
    inp.begin_frame()
    inp.tick_key_repeat(5.0)
    assert_false(inp.key_repeated(KEY_LEFT_SHIFT))
    # A held modifier does not stop the letter's repeat, and the letter is what repeats.
    inp.begin_frame()
    inp.set_key(KEY_A, True)
    inp.tick_key_repeat(5.0)
    inp.begin_frame()
    inp.tick_key_repeat(5.2)
    assert_true(inp.key_repeated(KEY_A))
    # Rate 0 (repeat off, or the platform repeats itself): no synthesis.
    var off = InputState()
    off.set_repeat_info(0, 0)
    off.begin_frame()
    off.set_key(KEY_A, True)
    off.tick_key_repeat(0.0)
    off.begin_frame()
    off.tick_key_repeat(10.0)
    assert_false(off.key_repeated(KEY_A))
    # A platform repeat (a press of a key already down) counts as a repeat, without retyping text.
    off.begin_frame()
    off.set_key(KEY_A, True)
    assert_true(off.key_repeated(KEY_A))
    assert_false(off.key_pressed(KEY_A))
    assert_equal(off.text_input_utf8, "")
    # Losing focus lets go of everything, the repeating key included.
    var lost = InputState()
    lost.begin_frame()
    lost.set_key(KEY_A, True)
    lost.tick_key_repeat(0.0)
    lost.release_held()
    lost.begin_frame()
    lost.tick_key_repeat(9.0)
    assert_false(lost.key_repeated(KEY_A))


def test_layout_shortcuts() raises:
    # AZERTY: the US Q position prints 'a', the US A position 'q', the US ';' position 'm'.
    assert_equal(layout_shortcut_code(KEY_Q, 97), KEY_A)
    assert_equal(layout_shortcut_code(KEY_A, 113), KEY_Q)
    assert_equal(layout_shortcut_code(59, 109), 77)
    # Punctuation and digits keep the GLFW code of the character; anything else falls back.
    assert_equal(layout_shortcut_code(77, 44), 44)
    assert_equal(layout_shortcut_code(49, 38), 49)    # AZERTY '&' on the 1 key: physical
    assert_equal(layout_shortcut_code(KEY_Q, 0x0439), KEY_Q)  # Cyrillic: physical
    assert_equal(layout_shortcut_code(KEY_BACKSPACE, 97), KEY_BACKSPACE)  # not a printable key
    # InputState routes a physical key through the table only when enabled.
    var table = List[Int](capacity=512)
    for i in range(512):
        table.append(i)
    table[KEY_Q] = KEY_A
    table[KEY_A] = KEY_Q
    var inp = InputState()
    inp.set_layout_key_map(table.copy())
    inp.begin_frame()
    inp.set_key(KEY_Q, True)
    assert_true(inp.key_pressed(KEY_Q))  # off: US positions
    inp.set_key(KEY_Q, False)
    inp.layout_shortcuts = True
    inp.begin_frame()
    inp.set_key(KEY_Q, True)
    assert_true(inp.key_pressed(KEY_A))
    assert_false(inp.key_down(KEY_Q))
    inp.set_key(KEY_Q, False)
    assert_false(inp.key_down(KEY_A))
    # The US fallback keymap maps every letter to itself.
    var us = XkbKeymap()
    us.reset_us_fallback()
    var us_table = wayland_layout_key_map(us)
    for code in range(65, 91):
        assert_equal(us_table[code], code)
    assert_equal(us_table[KEY_BACKSPACE], KEY_BACKSPACE)


def test_pointer_enter_leave() raises:
    var inp = InputState()
    inp.begin_frame()
    inp.move_mouse(100.0, 50.0)
    inp.begin_frame()
    inp.pointer_left()
    assert_false(inp.has_mouse_position)
    assert_true(inp.last_mouse_x <= POINTER_OUTSIDE)
    inp.begin_frame()
    inp.pointer_entered(400.0, 300.0)
    assert_true(inp.has_mouse_position)
    assert_true(_feq(inp.delta_x, 0.0) and _feq(inp.delta_y, 0.0))
    inp.move_mouse(410.0, 305.0)
    assert_true(_feq(inp.delta_x, 10.0) and _feq(inp.delta_y, 5.0))


def test_pointer_framebuffer_pixels_after_resize() raises:
    var win = WaylandWindow()
    win.logical_width = 1280  # the size the window was created at
    win.logical_height = 720
    win.width = 1000  # a configure resized it
    win.height = 500
    var p = win.map_pointer(250.0, 125.0)
    assert_true(_feq(p[0], 250.0) and _feq(p[1], 125.0))
    # Scale 2: surface-local logical pixels -> framebuffer pixels.
    win.preferred_buffer_scale = 2
    var fb = win.framebuffer_size()
    assert_equal(fb[0], 2000)
    assert_equal(fb[1], 1000)
    p = win.map_pointer(250.0, 125.0)
    assert_true(_feq(p[0], 500.0) and _feq(p[1], 250.0))
    # A zero-copy swapchain stretched over the window: its own pixels.
    win.gpu_width = 1280
    win.gpu_height = 720
    win.viewport_width = 1000
    win.viewport_height = 500
    p = win.map_pointer(500.0, 250.0)
    assert_true(_feq(p[0], 640.0) and _feq(p[1], 360.0))
    win.gpu_width = 0
    win.gpu_height = 0

    # X11: a ConfigureNotify moves the pointer space with the window.
    var x = X11Window()
    x.window_id = 0x200001
    x.logical_width = 1920
    x.logical_height = 1080
    x.width = 1920
    x.height = 1080
    var ev = List[UInt8](length=32, fill=0)
    ev[0] = 22  # ConfigureNotify
    x11_put32(ev, 8, x.window_id)
    x11_put16(ev, 20, 1200)
    x11_put16(ev, 22, 700)
    x.incoming = ev^
    x.process_events()
    assert_equal(x.width, 1200)
    assert_true(_feq(x.map_mouse_x(600.0), 600.0) and _feq(x.map_mouse_y(350.0), 350.0))
    assert_equal(x.framebuffer_size()[0], 1200)
    x.window_id = 0  # nothing to destroy


def test_x11_altgr_and_server_repeat() raises:
    var x = X11Window()
    x.min_keycode = 8
    x.max_keycode = 30
    x.keysyms_per_keycode = 6
    x.keymap = List[Int](length=(x.max_keycode - x.min_keycode + 1) * 6, fill=0)
    # Keycode 24 (evdev 16, the US Q position) on a German map: q Q q Q @ Ω.
    var base = (24 - x.min_keycode) * 6
    x.keymap[base + 0] = 0x71
    x.keymap[base + 1] = 0x51
    x.keymap[base + 2] = 0x71
    x.keymap[base + 3] = 0x51
    x.keymap[base + 4] = 0x40
    x.keymap[base + 5] = 0x07D9  # Greek OMEGA keysym
    x._append_key_utf8(24, 0)
    x._append_key_utf8(24, 1)
    x._append_key_utf8(24, 0x80)  # AltGr (Mod5)
    assert_equal(x.pending_text_utf8, "qQ@")
    x.pending_text_utf8 = ""
    # Layout table from the core map: the Q position prints q.
    var table = x.layout_key_map()
    assert_equal(table[KEY_Q], KEY_Q)
    # Server auto-repeat: KeyRelease + KeyPress at the same timestamp stays one press of the held key.
    var ev = List[UInt8](length=64, fill=0)
    ev[0] = 3
    ev[1] = 24
    x11_put32(ev, 4, 777)
    ev[32] = 2
    ev[33] = 24
    x11_put32(ev, 36, 777)
    x.incoming = ev^
    x.process_events()
    assert_equal(len(x.pending_keys), 1)
    assert_true(x.pending_key_down[0])
    assert_equal(x.pending_text_utf8, "q")
    var inp = InputState()
    inp.set_repeat_info(0, 0)
    inp.begin_frame()
    inp.set_key(KEY_Q, True)
    inp.begin_frame()
    inp.set_key(x.pending_keys[0], True)
    assert_true(inp.key_repeated(KEY_Q))
    assert_true(inp.key_pressed_or_repeat(KEY_Q))


def test_quit_signal_flag() raises:
    assert_true(install_quit_signal_handlers())
    assert_equal(quit_signal_received(), 0)
    _ = external_call["raise", Int32](Int32(15))  # SIGTERM: handled, not fatal
    assert_equal(quit_signal_received(), 15)
    assert_true(install_quit_signal_handlers())
    assert_equal(quit_signal_received(), 0)
    _ = external_call["raise", Int32](Int32(2))  # SIGINT
    assert_equal(quit_signal_received(), 2)
    # Leave the defaults behind for the rest of the process.


def main() raises:
    test_key_repeat_synthesizer()
    test_key_repeat_retypes_text()
    test_key_repeat_modifiers_and_rate_zero()
    test_layout_shortcuts()
    test_pointer_enter_leave()
    test_pointer_framebuffer_pixels_after_resize()
    test_x11_altgr_and_server_repeat()
    test_quit_signal_flag()
    print("test_flint_v5_platform_lifecycle: all checks passed")
