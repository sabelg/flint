"""Native Studio capture comparison and build-profile picker."""
from std.ffi import external_call
from std.sys import argv
from flint.assets.image import write_png
from flint.platform.input import InputState
from flint.platform.os import write_bytes
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import SimdCanvas
from flint.studio.profile_compare import Capture


def ms(value: Float64) -> String:
    if value < 0:
        return "--"
    var hundredths = Int(value * 100 + 0.5)
    var rest = hundredths % 100
    return String(hundredths // 100) + "." + ("0" if rest < 10 else "") + String(rest)


def delta(a: Float64, b: Float64) -> String:
    if a < 0 or b < 0:
        return "--"
    if abs(b - a) < 0.005:
        return "0.00"
    return ("+" if b >= a else "-") + ms(abs(b - a))


def main() raises:
    var args = argv()
    var picker = len(args) >= 4 and args[1] == "--pick"
    var text_only = len(args) == 4 and args[3] == "--text"
    var shot = len(args) == 5 and args[3] == "--shot"
    if not picker and len(args) != 3 and not shot and not text_only:
        raise Error("studio --compare BEFORE AFTER [--text | --shot FILE]")
    var win = WaylandWindow()
    var canvas = SimdCanvas(960, 640)
    var input = InputState()
    if picker:
        win.set_title("Flint Studio - Build Profiles")
        if not win.initialize(960, 640):
            raise Error("Build profile picker needs a display")
        var selected = 0
        var count = len(args) - 3
        while not win.close_requested:
            input.begin_frame()
            win.poll(input)
            if input.key_pressed(256):
                break
            selected = max(0, min(count - 1, selected + Int(input.key_pressed(264)) - Int(input.key_pressed(265))))
            if input.key_pressed(257):
                var value = List[UInt8]()
                for byte in String(args[3 + selected]).as_bytes():
                    value.append(byte)
                if not write_bytes(String(args[2]), value):
                    raise Error("Could not return selected profile")
                break
            canvas.clear()
            canvas.draw_text(30, 35, "BUILD PROFILES")
            var top = max(0, selected - 12)
            for i in range(top, min(count, top + 13)):
                var y = 95 + (i - top) * 36
                if i == selected:
                    canvas.draw_rect(18, y - 8, 920, 32, 0xFF31556A)
                canvas.draw_text(32, y, String(args[3 + i]))
            canvas.draw_text(30, 580, "UP/DOWN SELECT   ENTER BUILD   ESC CLOSE")
            win.present(canvas.pixels, 960, 640)
            _ = external_call["usleep", Int32](UInt32(20000))
        win.close()
        return

    var a = Capture(String(args[1]))
    var b = Capture(String(args[2]))
    var names = a.names.copy()
    for name in b.names:
        if name not in a.indexes:
            names.append(name)
    var means = List[String]()
    var tails = List[String]()
    var changes = List[String]()
    print("Capture frames:", len(a.starts), "before,", len(b.starts), "after")
    print("Zone | mean A/B ms | p95 A/B ms | delta p95 ms | sampled frames A/B")
    for name in names:
        var x = a.stats(name)
        var y = b.stats(name)
        means.append(ms(x[0]) + " / " + ms(y[0]))
        tails.append(ms(x[1]) + " / " + ms(y[1]))
        changes.append(delta(x[1], y[1]))
        print(name, "|", ms(x[0]), ms(y[0]), "|", ms(x[1]), ms(y[1]), "|", delta(x[1], y[1]), "|", x[2], y[2])
    if text_only:
        return
    win.set_title("Flint Studio - Profile Comparison")
    if not shot and not win.initialize(960, 640):
        raise Error("Display unavailable. Use --text or --shot FILE.")
    var row = 0
    var frame = 0
    var max_frames = max(len(a.starts), len(b.starts))
    var scale = Float64(20)
    var worst_a = 0
    var worst_b = 0
    for i in range(len(a.starts)):
        if a.at("frame", i) > a.at("frame", worst_a):
            worst_a = i
    for i in range(len(b.starts)):
        if b.at("frame", i) > b.at("frame", worst_b):
            worst_b = i
    scale = max(scale, max(a.at("frame", worst_a), b.at("frame", worst_b)) * 1.05)
    while True:
        if not shot:
            input.begin_frame()
            win.poll(input)
            if input.key_pressed(256) or win.close_requested:
                break
            row = max(0, min(len(names) - 1, row + Int(input.key_pressed(264)) - Int(input.key_pressed(265))))
            frame = max(0, min(max_frames - 1, frame + Int(input.key_pressed(262)) - Int(input.key_pressed(263))))
            if input.key_pressed(ord("A")):
                frame = worst_a
            if input.key_pressed(ord("B")):
                frame = worst_b
            if input.mouse_down(0) and input.last_mouse_y >= 55 and input.last_mouse_y <= 228:
                frame = max(0, min(max_frames-1, Int((input.last_mouse_x - 24) / 910 * Float32(max_frames - 1))))
        canvas.clear()
        canvas.draw_text(24, 18, "FRAME TIME (MS)")
        canvas.draw_text(252, 18, "A  BEFORE", 0xFF61C6BF)
        canvas.draw_text(430, 18, "B  AFTER", 0xFFF7B35F)
        canvas.draw_text(650, 18, "BUDGET 16.67 MS", 0xFFA0ABB7)
        # A vertical range per pixel preserves spikes when many frames share a column.
        for which in range(2):
            ref data = a if which == 0 else b
            var color = UInt32(0xFF61C6BF) if which == 0 else UInt32(0xFFF7B35F)
            for i in range(len(data.starts)):
                var px = 24 + i * 910 // max(1, max_frames - 1)
                var py = 225 - Int(min(Float64(1), data.at("frame", i) / scale) * 160)
                canvas.draw_rect(px, py, 2, 3, color)
        canvas.draw_rect(24, 225 - Int(16.667 / scale * 160), 912, 1, 0xFF6A7D87)
        canvas.draw_rect(24 + frame * 910 // max(1, max_frames - 1), 55, 1, 173, 0xFFFFFFFF)
        var selected = names[row]
        var av = a.at(selected, frame) if frame < len(a.starts) else -1
        var bv = b.at(selected, frame) if frame < len(b.starts) else -1
        canvas.draw_text(24, 243, "FRAME " + String(frame) + "   " + selected + "   A " + ms(av) + "   B " + ms(bv))
        canvas.draw_text(24, 291, "ZONE")
        canvas.draw_text(398, 291, "MEAN A/B")
        canvas.draw_text(600, 291, "P95 A/B")
        canvas.draw_text(819, 291, "CHANGE")
        var top = max(0, row - 8)
        for i in range(top, min(len(names), top + 9)):
            var y = 330 + (i - top) * 27
            if i == row:
                canvas.draw_rect(18, y - 4, 920, 24, 0xFF29475A)
            canvas.draw_text(24, y, names[i], 0xFFEDF3F7, 1 if names[i].byte_length() > 30 else 2)
            canvas.draw_text(398, y, means[i])
            canvas.draw_text(600, y, tails[i])
            canvas.draw_text(819, y, changes[i])
        canvas.draw_text(24, 600, "UP/DOWN ZONE   LEFT/RIGHT FRAME   CLICK TO SEEK   A/B SLOWEST FRAME")
        if shot:
            if not write_png(String(args[4]), canvas.pixels, 960, 640):
                raise Error("Could not write comparison image")
            break
        win.present(canvas.pixels, 960, 640)
        _ = external_call["usleep", Int32](UInt32(20000))
    if not shot:
        win.close()
