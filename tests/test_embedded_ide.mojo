"""Regression checks for fragmented PTY frames, bounds safety, RAII and Studio's focus shortcut."""
from std.ffi import external_call
from std.testing import assert_equal, assert_true, assert_false
from flint.platform.input import InputState
from flint.platform.os import list_directory
from flint.platform.traits import WindowBackend
from flint.render.font import SdfFont
from flint.studio.ide_grid import IdeGrid
from flint.studio.ide_pty import IdePty
from flint.studio.ide_view import BACK, draw_workspace, raster_ui, raster_workspace, toolbar_hit, toolbar_start
from flint.studio.source_editor import EmbeddedIde
from flint.ui.draw_list import UiDrawList


struct MockWindowBackend(WindowBackend):
    var w: Int
    var h: Int
    var scale: Float32
    var presents: Int
    var last_px_count: Int
    var closed: Bool
    var clip: String

    def __init__(out self, w: Int = 640, h: Int = 360, scale: Float32 = 1.0):
        self.w = w
        self.h = h
        self.scale = scale
        self.presents = 0
        self.last_px_count = 0
        self.closed = False
        self.clip = ""

    def __deinit__(deinit self):
        self.shutdown()

    def initialize(mut self, width: Int, height: Int) -> Bool:
        self.w = width
        self.h = height
        return True

    def poll_events(mut self, mut input: InputState):
        pass

    def present_argb(mut self, pixels: List[UInt32], width: Int, height: Int):
        self.presents += 1
        self.last_px_count = len(pixels)

    def toggle_fullscreen(mut self):
        pass

    def should_close(self) -> Bool:
        return self.closed

    def framebuffer_size(self) -> Tuple[Int, Int]:
        return (self.w, self.h)

    def content_scale(self) -> Float32:
        return self.scale

    def clipboard_get(mut self) -> String:
        return self.clip

    def clipboard_set(mut self, text: String):
        self.clip = text

    def shutdown(mut self):
        self.closed = True


def _fd_count() -> Int:
    return len(list_directory("/proc/self/fd"))


def _drop_pty_with_pipe():
    var fds: Array[Int32, 2] = [-1, -1]
    if external_call["pipe", Int32](Pointer(to=fds)) == 0:
        _ = external_call["close", Int32](fds[1])
        var p = IdePty()
        p.fd = fds[0]
        p.pid = 0


def _drop_ide_with_pipe():
    var fds: Array[Int32, 2] = [-1, -1]
    if external_call["pipe", Int32](Pointer(to=fds)) == 0:
        _ = external_call["close", Int32](fds[1])
        var ed = EmbeddedIde()
        ed.pty.fd = fds[0]
        ed.pty.pid = 0


def main() raises:
    comptime assert conforms_to(MockWindowBackend, WindowBackend)

    var start = toolbar_start(True)
    assert_equal(toolbar_hit(20, 18, 800, True), BACK)
    assert_equal(toolbar_hit(start + 68, 18, 800, True), 1)
    assert_equal(toolbar_hit(start + 68, 41, 800, True), -1)
    assert_equal(toolbar_hit(start + 64, 18, 800, True), -1)
    assert_equal(toolbar_hit(start + 220, 18, start + 225, True), -1)
    # Tier 5 [HIGH-2]: zero/narrow width toolbar_hit & 0-dim IdeGrid / IdeView bounds safety
    assert_equal(toolbar_hit(0, 0, 0, True), -1)
    assert_equal(toolbar_hit(10, 10, 100, True), -1)
    assert_equal(toolbar_hit(-1, 10, 800, True), -1)

    var grid0 = IdeGrid()
    grid0.resize(0, 0)
    assert_equal(grid0.cols, 0)
    assert_equal(grid0.rows, 0)
    assert_equal(len(grid0.cells), 0)
    for byte in "Hello\n\x1b[100;100H\x1b[2JX".as_bytes():
        grid0.feed(Int(byte))
    assert_true(grid0.x >= 0 and grid0.y >= 0)
    grid0.resize(-4, -2)
    assert_equal(grid0.cols, 0)
    assert_equal(grid0.rows, 0)

    var ui0 = UiDrawList()
    var font0 = SdfFont()
    draw_workspace(ui0, grid0, 400, 300, font0, True, -1, "", [], -1, [], False)
    for i in range(len(ui0.prims)):
        assert_true(ui0.prims[i].x0 >= Float32(0.0))
        assert_true(ui0.prims[i].y0 >= Float32(0.0))
    var ui_neg = UiDrawList()
    draw_workspace(ui_neg, grid0, 0, 0, font0, True, -1, "", [], -1, [], False)
    assert_equal(len(ui_neg.prims), 0)
    var px0 = List[UInt32]()
    raster_workspace(px0, font0, grid0, 0, 10, True, -1, "")
    assert_equal(len(px0), 0)
    raster_ui(px0, font0, ui0, -5, 10)
    assert_equal(len(px0), 0)

    var grid = IdeGrid()
    grid.resize(20, 4)
    for byte in "\x1b[2;3H\x1b[38;2;34;120;211m界\x1b]8;;https://local\x1b\\\x1b[3;1HA".as_bytes():
        grid.feed(Int(byte))
    assert_equal(grid.cells[22].code, 0x754C)
    assert_equal(grid.cells[22].fg, UInt32(0x2278D3))
    assert_equal(grid.cells[40].code, 65)
    for byte in "\x1b[2J\x1b[?25l".as_bytes():
        grid.feed(Int(byte))
    assert_equal(grid.cells[22].code, 32)
    assert_false(grid.cursor)

    # Tier 5 [HIGH-3]: IdePty & EmbeddedIde RAII drop, stop(pid=0), resize(-1), move semantics & generic WindowBackend
    var fd_before = _fd_count()
    _drop_pty_with_pipe()
    _drop_ide_with_pipe()
    var fd_after = _fd_count()
    assert_equal(fd_after, fd_before)

    var pty_stop_check = IdePty()
    var pipe_fds: Array[Int32, 2] = [-1, -1]
    assert_equal(external_call["pipe", Int32](Pointer(to=pipe_fds)), 0)
    _ = external_call["close", Int32](pipe_fds[1])
    pty_stop_check.fd = pipe_fds[0]
    pty_stop_check.pid = 0
    pty_stop_check.resize(-1, -1)  # Must not crash on negative dims
    pty_stop_check.stop()
    assert_equal(pty_stop_check.fd, -1)
    assert_equal(_fd_count(), fd_before)

    var editor = EmbeddedIde()
    var inp = InputState()
    editor.grid.resize(80, 24)
    editor.visible = True
    inp.set_key(341, True)
    inp.set_key(297, True)
    assert_true(editor.tick(inp, 720, 464, 0, 0))
    assert_false(editor.visible)
    inp.set_key(341, False)
    inp.set_key(297, False)
    inp.begin_frame()
    assert_false(editor.tick(inp, 720, 464, 0, 0))
    inp.push_text_utf8("hé")
    editor.forward(inp, 0, 0)
    inp.begin_frame()
    inp.set_key(291, True)
    editor.forward(inp, 0, 0)
    var expected = "hé\x1bOQ"
    assert_equal(len(editor.pty.pending), expected.byte_length())
    for i in range(len(editor.pty.pending)):
        assert_equal(editor.pty.pending[i], expected.as_bytes()[i])

    # Generic WindowBackend present() test
    var mock_win = MockWindowBackend(320, 180, 1.0)
    var ed2 = EmbeddedIde()
    ed2.visible = True
    ed2.dirty = True
    ed2.grid.resize(30, 6)
    var inp2 = InputState()
    assert_true(ed2.present(mock_win, inp2, scene_hotkey=False))
    assert_equal(mock_win.presents, 1)
    assert_equal(mock_win.last_px_count, 320 * 180)
    print("Embedded IDE grid, UTF-8 input, F2, bounds safety, RAII and generic WindowBackend passed.")
