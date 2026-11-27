"""The in-window TurboMojo workspace. The editor owns its buffers while hidden."""
from std.ffi import external_call
from flint.platform.input import InputState
from flint.platform.os import env_value
from flint.platform.traits import WindowBackend
from flint.render.font import SdfFont
from flint.studio.ide_grid import IdeGrid
from flint.studio.ide_pty import IdePty
from flint.studio.ide_view import HEADER, CW, CH, BACK, raster_workspace, toolbar_hit

def special_key(key: Int, modifier: Int) -> String:
    var plain = ["\x1b", "\r", "\t", "\x7f"]
    if 256 <= key <= 259:
        if key == 258 and (modifier - 1) & 1:
            return "\x1b[Z"
        var prefix = String("\x1b") if key != 256 and (modifier - 1) & 2 else String()
        if key == 259 and (modifier - 1) & 4:
            return prefix + "\x08"
        return prefix + plain[key - 256]
    var finals = ["C", "D", "B", "A"]
    var suffix = String()
    if 262 <= key <= 265:
        suffix = finals[key - 262]
    elif key == 268 or key == 269:
        suffix = "H" if key == 268 else "F"
    elif 290 <= key <= 293:
        suffix = String(chr(80 + key - 290))
        if modifier == 1:
            return "\x1bO" + suffix
    if suffix:
        return "\x1b[" + (String() if modifier == 1 else "1;" + String(modifier)) + suffix
    var n = 0
    if key == 260 or key == 261:
        n = key - 258
    elif key == 266 or key == 267:
        n = key - 261
    elif 294 <= key <= 301:
        var f: Array[Int, 8] = [15, 17, 18, 19, 20, 21, 23, 24]
        n = f[key - 294]
    if n:
        return "\x1b[" + String(n) + (String() if modifier == 1 else ";" + String(modifier)) + "~"
    return String()


struct EmbeddedIde(Movable):
    var hover: Int
    var visible: Bool
    var requested: Bool
    var dirty: Bool
    var pty: IdePty
    var grid: IdeGrid
    var status: String
    var chunk: List[UInt8]
    var font: SdfFont
    var pixels: List[UInt32]
    var width: Int
    var height: Int

    def __init__(out self):
        self.hover = -1
        self.visible = False
        self.requested = False
        self.dirty = False
        self.pty = IdePty()
        self.grid = IdeGrid()
        self.status = String()
        self.chunk = List[UInt8]()
        self.font = SdfFont()
        self.pixels = List[UInt32]()
        self.width = 0
        self.height = 0

    def drain(mut self):
        if self.pty.fd < 0:
            return
        # Bound work per frame even when a build emits continuous output.
        for _ in range(16):
            var n = external_call["read", Int](
                self.pty.fd, self.chunk.unsafe_ptr(), len(self.chunk)
            )
            if n <= 0:
                if self.pty.reap():
                    self.status = "Session ended"
                    self.dirty = True
                break
            self.dirty = True
            for i in range(n):
                self.grid.feed(Int(self.chunk[i]))

    def forward(mut self, inp: InputState, mx: Int, my: Int):
        var ctrl = inp.key_down(341) or inp.key_down(345)
        var alt = inp.key_down(342) or inp.key_down(346)
        var shift = inp.key_down(340) or inp.key_down(344)
        var modifier = 1 + Int(shift) + 2 * Int(alt) + 4 * Int(ctrl)
        if ctrl or alt:
            if ctrl and (inp.key_pressed(32) or inp.key_pressed(50)):
                self.pty.pending.append(0)
            for key in range(65, 91):
                if inp.key_pressed(key):
                    if alt:
                        self.pty.pending.append(27)
                    self.pty.pending.append(
                        UInt8(key - 64 if ctrl else (key if shift else key + 32))
                    )
        elif inp.ime_commit:
            self.pty.send(inp.ime_commit)
        else:
            for b in inp.text_input_utf8.as_bytes():
                if b >= 32 and b != 127:
                    self.pty.pending.append(b)
        for key in range(256, 302):
            if inp.key_pressed(key):
                self.pty.send(special_key(key, modifier))
        if self.grid.mouse and my >= HEADER:
            var pos = (
                ";"
                + String(min(self.grid.cols, max(1, mx // CW + 1)))
                + ";"
                + String(min(self.grid.rows, (my - HEADER) // CH + 1))
            )
            var mods = (modifier - 1) * 4
            for button in range(3):
                var n = 0 if button == 0 else (2 if button == 1 else 1)
                if inp.mouse_pressed(button):
                    self.pty.send("\x1b[<" + String(n + mods) + pos + "M")
                if inp.mouse_released_state[button]:
                    self.pty.send("\x1b[<" + String(n + mods) + pos + "m")
                elif (
                    inp.mouse_down(button)
                    and not inp.mouse_pressed(button)
                    and (inp.delta_x != 0 or inp.delta_y != 0)
                ):
                    self.pty.send("\x1b[<" + String(32 + n + mods) + pos + "M")
            if inp.wheel != 0:
                self.pty.send("\x1b[<" + String((64 if inp.wheel > 0 else 65) + mods) + pos + "M")
        self.pty.flush()

    def tick(mut self, inp: InputState, w: Int, h: Int, mx: Int, my: Int, scene_hotkey: Bool = True) -> Bool:
        self.drain()
        var opening = not self.visible and (self.requested or (scene_hotkey and inp.key_pressed(297)))
        self.requested = False
        if not self.visible and not opening:
            return False
        var hit = toolbar_hit(mx, my, w, True)
        if hit != self.hover:
            self.hover = hit
            self.dirty = True
        if self.visible and (
            (inp.key_pressed(297) and (inp.key_down(341) or inp.key_down(345)))
            or (inp.mouse_pressed(0) and hit == BACK)
        ):
            self.visible = False
            return True
        var cols = min(512, max(1, w // CW))
        var rows = min(256, max(1, (h - HEADER) // CH))
        if w != self.width or h != self.height:
            self.dirty = True
        self.width = w
        self.height = h
        if opening:
            self.visible = True
            self.dirty = True
        if opening and self.pty.fd < 0:
            self.grid = IdeGrid()
            self.grid.resize(cols, rows)
            self.chunk = List[UInt8](length=8192, fill=0)
            self.status = String()
            if not env_value("FLINT_STUDIO_IDE_LAUNCHER"):
                self.status = "Launch with tools/flint.sh studio to attach PTY"
            elif not self.pty.start(cols, rows):
                self.status = "Could not open the editor terminal"
        elif cols != self.grid.cols or rows != self.grid.rows:
            self.grid.resize(cols, rows)
            if self.pty.fd >= 0:
                self.pty.resize(cols, rows)
        if not opening and self.pty.fd >= 0:
            if inp.mouse_pressed(0) and 0 <= hit < BACK:
                var keys: Array[Int, 4] = [292, 291, 298, 297]
                self.pty.send(special_key(keys[hit], 1))
                self.pty.flush()
            else:
                self.forward(inp, mx, my)
        return True

    def present[W: WindowBackend](mut self, mut win: W, inp: InputState, scene_hotkey: Bool = True) -> Bool:
        var fb = win.framebuffer_size()
        var w = max(1, fb[0])
        var h = max(1, fb[1])
        var scale = win.content_scale()
        if not (scale > Float32(0.0)):
            scale = Float32(1.0)
        var mx = Int(inp.last_mouse_x * scale)
        var my = Int(inp.last_mouse_y * scale)
        if not self.tick(inp, w, h, mx, my, scene_hotkey):
            return False
        if self.visible:
            if self.dirty:
                raster_workspace(self.pixels, self.font, self.grid, w, h, True, self.hover, self.status)
                self.dirty = False
            win.present_argb(self.pixels, w, h)
            _ = external_call["usleep", Int32](UInt32(12000))
        return True

    def close(mut self):
        self.pty.stop()

    def __deinit__(deinit self):
        self.close()
