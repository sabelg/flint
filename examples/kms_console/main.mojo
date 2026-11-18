"""Bare-display demo for the DRM/KMS backend (`platform/kms.mojo`).

Runs fullscreen with no X11 or Wayland: switch to a text console (Ctrl+Alt+F3),
log in, and run

    FLINT_BACKEND=kms ./build/flint-kms

(on a text console FLINT_BACKEND can be left out: with neither WAYLAND_DISPLAY
nor DISPLAY set, Flint picks KMS by itself). Animated frames, the last keys
pressed, the mouse and the display mode are shown. Esc or Ctrl+C quits, and
the console comes back as it was.

    ./build/flint-kms --probe   list cards, connectors and modes (read-only, safe anywhere)
"""
from std.sys import argv
from std.math import sin, cos
from flint.core.math import Vec3
from flint.platform.os import get_time
from flint.platform.input import InputState
from flint.platform.wayland import WaylandWindow
from flint.platform.kms import kms_probe
from flint.render.simd_canvas import Camera3D, SimdCanvas


def key_label(key: Int) -> String:
    if key >= 65 and key <= 90:
        return String(chr(key))
    if key >= 48 and key <= 57:
        return String(chr(key))
    if key == 32:
        return "SPACE"
    if key == 257:
        return "ENTER"
    if key >= 262 and key <= 265:
        return ["RIGHT", "LEFT", "DOWN", "UP"][key - 262]
    if key >= 340 and key <= 346:
        return "MOD"
    if key >= 290 and key <= 301:
        return "F" + String(key - 289)
    return "#" + String(key)


def main():
    for i in range(1, len(argv())):
        if String(argv()[i]) == "--probe":
            print(kms_probe(), end="")
            return
    var w = 960
    var h = 540
    var canvas = SimdCanvas(w, h)
    var win = WaylandWindow()
    win.set_title("FLINT KMS CONSOLE")
    var input = InputState()
    if not win.initialize(w, h):
        print("[Flint] No display. On a text console run with FLINT_BACKEND=kms; try --probe.")
        return
    var start = get_time()
    var frames = 0
    var fps = 0
    var fps_at = start
    var keys = String("")
    while not win.close_requested:
        input.begin_frame()
        win.poll(input)
        if input.key_pressed(256):
            break
        for key in range(512):
            if input.key_pressed(key):
                keys = key_label(key) + " " + keys
        if keys.byte_length() > 48:
            var cut = String(keys[byte=0:48])
            keys = cut
        var now = get_time()
        var t = Float32(now - start)
        frames += 1
        if now - fps_at >= 1.0:
            fps = frames
            frames = 0
            fps_at = now

        var eye = Vec3(cos(t * 0.6) * 7.0, 3.0 + sin(t * 0.4), sin(t * 0.6) * 7.0)
        var cam = Camera3D(w, h, eye, Vec3(0.0, 0.0, 0.0), 55.0)
        canvas.clear(0x0D1726, 0x1B2B44)
        for i in range(-6, 7):
            canvas.draw_line_3d(cam, Vec3(Float32(i), 0.0, -6.0), Vec3(Float32(i), 0.0, 6.0), 0x2A3D5C, 1)
            canvas.draw_line_3d(cam, Vec3(-6.0, 0.0, Float32(i)), Vec3(6.0, 0.0, Float32(i)), 0x2A3D5C, 1)
        var top = Vec3(0.0, 2.2 + sin(t * 2.0) * 0.2, 0.0)
        for k in range(4):
            var a = t + Float32(k) * 1.5708
            var p = Vec3(cos(a) * 1.4, 0.6, sin(a) * 1.4)
            var q = Vec3(cos(a + 1.5708) * 1.4, 0.6, sin(a + 1.5708) * 1.4)
            canvas.draw_line_3d(cam, p, q, 0xF49E38, 2)
            canvas.draw_line_3d(cam, p, top, 0xFFE082, 2)
            canvas.draw_line_3d(cam, p, Vec3(0.0, -0.4, 0.0), 0x4FC3F7, 2)
        var bx = Int((sin(t * 1.3) * 0.5 + 0.5) * Float32(w - 40))
        canvas.draw_rect(bx, h - 28, 40, 12, 0x4FC3F7)
        canvas.draw_panel(14, 8, w - 28, 30, 0x101824, 0xF49E38)
        canvas.draw_text(24, 15, "FLINT " + win.backend_name() + "  " + String(win.width) + "X" + String(win.height) + "  " + String(fps) + " FPS  |  ESC OR CTRL+C QUITS", 0xFFE082, 2)
        canvas.draw_text(24, 52, "KEYS: " + keys, 0xF2F5F8, 2)
        canvas.draw_text(24, 76, "MOUSE: " + String(Int(input.last_mouse_x)) + "," + String(Int(input.last_mouse_y)) + ("  [LEFT]" if input.mouse_down(0) else "") + ("  [RIGHT]" if input.mouse_down(1) else ""), 0xF2F5F8, 2)
        win.present(canvas.pixels, w, h)
    win.close()
    print("[Flint] KMS demo closed after " + String(Int(get_time() - start)) + " s; console restored.")
