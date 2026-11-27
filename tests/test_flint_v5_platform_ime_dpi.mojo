"""Display-free checks for platform IME, X11 HiDPI and KMS pointer space:
  (a) `zwp_text_input_v3` event decoding (wire packets) into double-buffered preedit / commit /
      delete-surrounding state, applied on `done` and flushed into `InputState`.
  (b) `UiContext.text_input` receives the commit and the delete-surrounding range, and reports
      `wants_text_input` with a caret rectangle while focused.
  (c) The caret rectangle mapped from framebuffer pixels to surface-local coordinates.
  (d) X11 `Xft.dpi` parsing from RESOURCE_MANAGER and the dpi -> content scale rounding / clamping,
      with the physical-size fallback.
  (e) KMS pointer in framebuffer pixels: relative motion accumulation, clamping to the mode, and
      rescaling when the framebuffer changes size.
"""
from std.testing import assert_true, assert_equal, assert_false
from flint.platform.input import InputState
from flint.platform.text_input_v3 import TextInputV3State, ti3_cursor_rect_from_framebuffer, utf8_codepoints_in
from flint.platform.wayland import WaylandWindow
from flint.platform.x11 import X11Window, parse_xft_dpi, x11_scale_from_dpi, x11_dpi_from_physical
from flint.platform.kms import KmsWindow, kms_pointer_axis, kms_pointer_rescale
from flint.ui.context import UiContext


def _feq(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < Float32(0.01)


def _put32(mut data: List[UInt8], value: Int):
    var v = value & 0xFFFFFFFF
    data.append(UInt8(v & 0xFF))
    data.append(UInt8((v >> 8) & 0xFF))
    data.append(UInt8((v >> 16) & 0xFF))
    data.append(UInt8((v >> 24) & 0xFF))


def _put_string(mut data: List[UInt8], text: String, null: Bool = False):
    if null:
        _put32(data, 0)
        return
    var n = text.byte_length() + 1
    _put32(data, n)
    for b in text.as_bytes():
        data.append(b)
    data.append(0)
    while (len(data) & 3) != 0:
        data.append(0)


def _packet(object: Int, opcode: Int, body: List[UInt8]) -> List[UInt8]:
    var pkt = List[UInt8]()
    _put32(pkt, object)
    _put32(pkt, ((8 + len(body)) << 16) | opcode)
    for b in body:
        pkt.append(b)
    return pkt^


def _feed(mut st: TextInputV3State, opcode: Int, body: List[UInt8]):
    var pkt = _packet(9, opcode, body)
    st.handle_event(pkt, opcode, len(pkt))


def test_text_input_v3_decoding() raises:
    var st = TextInputV3State()
    var b = List[UInt8]()
    _put32(b, 3)  # enter(surface 3)
    _feed(st, 0, b)
    assert_equal(st.entered_surface, 3)

    # preedit "にほ" (6 bytes), caret after the first char (byte 3), nothing applied before done.
    b = List[UInt8]()
    _put_string(b, "にほ")
    _put32(b, 3)
    _put32(b, 3)
    _feed(st, 2, b)
    assert_false(st.has_output())
    b = List[UInt8]()
    _put32(b, 1)
    _feed(st, 5, b)  # done(1)
    assert_true(st.has_output())
    assert_equal(st.preedit, "にほ")
    assert_equal(st.preedit_cursor_cp, 1)
    assert_equal(st.preedit_sel_len, 0)
    var inp = InputState()
    inp.begin_frame()
    st.flush_into(inp)
    assert_true(inp.ime_active)
    assert_equal(inp.ime_preedit, "にほ")
    assert_equal(inp.ime_cursor_cp, 1)
    assert_false(st.has_output())

    # A clause highlighted (bytes 0..6 = 2 codepoints).
    b = List[UInt8]()
    _put_string(b, "日本")
    _put32(b, 0)
    _put32(b, 6)
    _feed(st, 2, b)
    b = List[UInt8]()
    _put32(b, 2)
    _feed(st, 5, b)
    assert_equal(st.preedit_cursor_cp, 0)
    assert_equal(st.preedit_sel_len, 2)

    # Commit "日本" with no new preedit: the preedit resets to empty on done.
    b = List[UInt8]()
    _put_string(b, "日本")
    _feed(st, 3, b)
    b = List[UInt8]()
    _put32(b, 3)
    _feed(st, 5, b)
    assert_equal(st.preedit, "")
    assert_equal(st.commit_text, "日本")
    inp.begin_frame()
    st.flush_into(inp)
    assert_equal(inp.ime_commit, "日本")
    assert_false(inp.ime_active)
    assert_equal(inp.ime_preedit, "")

    # delete_surrounding_text(3, 0) + commit "é" + new preedit "x" in one batch; hidden cursor (-1).
    b = List[UInt8]()
    _put32(b, 3)
    _put32(b, 0)
    _feed(st, 4, b)
    b = List[UInt8]()
    _put_string(b, "é")
    _feed(st, 3, b)
    b = List[UInt8]()
    _put_string(b, "x")
    _put32(b, -1)
    _put32(b, -1)
    _feed(st, 2, b)
    b = List[UInt8]()
    _put32(b, 99)  # a stale serial is still applied
    _feed(st, 5, b)
    assert_equal(st.last_done_serial, 99)
    inp.begin_frame()
    st.flush_into(inp)
    assert_equal(inp.ime_delete_before, 3)
    assert_equal(inp.ime_delete_after, 0)
    assert_equal(inp.ime_commit, "é")
    assert_true(inp.ime_active)
    assert_equal(inp.ime_preedit, "x")
    assert_equal(inp.ime_cursor_cp, 1)  # hidden cursor: after the preedit
    inp.begin_frame()
    assert_equal(inp.ime_delete_before, 0)
    assert_equal(inp.ime_commit, "")

    # A null commit string and leave: the preedit is dropped.
    b = List[UInt8]()
    _put_string(b, "", null=True)
    _feed(st, 3, b)
    _feed(st, 1, _surface_body(3))
    assert_equal(st.entered_surface, 0)
    assert_false(st.enabled)
    st.flush_into(inp)
    assert_false(inp.ime_active)
    assert_equal(inp.ime_preedit, "")

    # Truncated packets are ignored without reading past the end.
    var short_pkt = _packet(9, 2, List[UInt8]())
    st.handle_event(short_pkt, 2, len(short_pkt))
    assert_equal(utf8_codepoints_in("aé日", 0, 6), 3)
    assert_equal(utf8_codepoints_in("aé日", 1, 3), 1)


def _surface_body(surface: Int) -> List[UInt8]:
    var b = List[UInt8]()
    _put32(b, surface)
    return b^


def test_wayland_text_input_without_protocol() raises:
    # No compositor, no zwp_text_input_manager_v3: requests are no-ops, nothing breaks.
    var win = WaylandWindow()
    assert_false(win.supports_text_input_protocol())
    win.set_text_input(True, 100.0, 40.0, 2.0, 18.0)
    assert_true(win.text_input_wanted)
    assert_false(win.ti3.enabled)
    win.set_text_input(False)
    assert_false(win.text_input_wanted)


def test_cursor_rect_to_surface() raises:
    # Scale 2: framebuffer 2000x1000 over a 1000x500 surface.
    var r = ti3_cursor_rect_from_framebuffer(400.0, 200.0, 4.0, 36.0, 2000, 1000, 1000, 500)
    assert_equal(r[0], 200)
    assert_equal(r[1], 100)
    assert_equal(r[2], 2)
    assert_equal(r[3], 18)
    # Scale 1, clamped inside the surface, minimum 1x1.
    r = ti3_cursor_rect_from_framebuffer(5000.0, -3.0, 0.0, 0.0, 800, 600, 800, 600)
    assert_equal(r[0], 799)
    assert_equal(r[1], 0)
    assert_equal(r[2], 1)
    assert_equal(r[3], 1)


def test_ui_text_input_ime_commit_and_delete() raises:
    var ui = UiContext()
    var text = String("abc")
    var inp = InputState()
    # Frame 1: focus the field and press End (caret after "abc").
    inp.begin_frame()
    inp.set_key(269, True)
    ui.begin_frame(inp, 800.0, 600.0, Float32(1.0 / 60.0))
    ui.focused_id = ui.widget_id("field")
    _ = ui.text_input("field", text, 300.0)
    ui.end_frame()
    assert_true(ui.wants_text_input())
    assert_true(ui.text_caret_rect.h > Float32(0.0))
    inp.begin_frame()
    inp.set_key(269, False)
    # Frame 3: the IME deletes one byte before the caret and commits "日".
    var st = TextInputV3State()
    st.delete_before = 1
    st.commit_text = "日"
    inp.begin_frame()
    st.flush_into(inp)
    ui.begin_frame(inp, 800.0, 600.0, Float32(1.0 / 60.0))
    var changed = ui.text_input("field", text, 300.0)
    ui.end_frame()
    assert_true(changed)
    assert_equal(text, "ab日")


def test_xft_dpi_parsing_and_scale() raises:
    assert_true(_feq(parse_xft_dpi("Xft.antialias:\t1\nXft.dpi:\t144\nXft.hinting:\t1\n"), 144.0))
    assert_true(_feq(parse_xft_dpi("  Xft.dpi :  192.5\n"), 192.5))
    assert_true(_feq(parse_xft_dpi("Xft.dpi:\t96"), 96.0))  # no trailing newline
    assert_true(_feq(parse_xft_dpi("*dpi: 120\nXft.dpix: 200\n"), 0.0))
    assert_true(_feq(parse_xft_dpi("Xft.dpi: abc\n"), 0.0))
    assert_true(_feq(parse_xft_dpi(""), 0.0))
    assert_true(_feq(x11_scale_from_dpi(96.0), 1.0))
    assert_true(_feq(x11_scale_from_dpi(120.0), 1.25))
    assert_true(_feq(x11_scale_from_dpi(144.0), 1.5))
    assert_true(_feq(x11_scale_from_dpi(192.0), 2.0))
    assert_true(_feq(x11_scale_from_dpi(168.0), 1.75))
    assert_true(_feq(x11_scale_from_dpi(100.0), 1.0))   # rounds to 1.0
    assert_true(_feq(x11_scale_from_dpi(72.0), 1.0))    # clamped up
    assert_true(_feq(x11_scale_from_dpi(960.0), 4.0))   # clamped down
    assert_true(_feq(x11_scale_from_dpi(0.0), 1.0))
    assert_true(_feq(x11_dpi_from_physical(3840, 600), 162.56))
    assert_true(_feq(x11_dpi_from_physical(1920, 0), 0.0))

    var x = X11Window()
    x.screen_width = 3840
    x.screen_width_mm = 508  # 192 dpi
    x.apply_dpi(144.0)  # Xft.dpi wins
    assert_true(_feq(x.content_scale(), 1.5))
    assert_equal(x.dpi_source, "Xft.dpi")
    x.apply_dpi(0.0)  # physical fallback
    assert_true(_feq(x.content_scale(), 2.0))
    assert_equal(x.dpi_source, "physical")
    x.screen_width_mm = 0  # nothing known
    x.apply_dpi(0.0)
    assert_true(_feq(x.content_scale(), 1.0))
    # Through WaylandWindow's X11 fallback, the same API Studio's UI scale tracking reads.
    var w = WaylandWindow()
    w.using_x11 = True
    w.x11.screen_width = 2560
    w.x11.apply_dpi(192.0)
    assert_true(_feq(w.content_scale(), 2.0))
    w.using_x11 = False


def test_kms_pointer_framebuffer_space() raises:
    assert_true(_feq(kms_pointer_axis(10.0, -25.0, 1920), 0.0))
    assert_true(_feq(kms_pointer_axis(1900.0, 50.0, 1920), 1919.0))
    assert_true(_feq(kms_pointer_axis(100.0, 7.0, 1920), 107.0))
    assert_true(_feq(kms_pointer_rescale(960.0, 1920, 3840), 1920.0))
    assert_true(_feq(kms_pointer_rescale(1919.0, 1920, 1280), 1279.0))  # clamped to the last pixel
    assert_true(_feq(kms_pointer_rescale(5.0, 0, 1280), 640.0))

    var k = KmsWindow()
    k.logical_width = 1280  # the size the app asked for
    k.logical_height = 720
    k.width = 3840  # the mode: framebuffer_size()
    k.height = 2160
    k.cursor_x = 1920.0
    k.cursor_y = 1080.0
    assert_equal(k.framebuffer_size()[0], 3840)
    # Relative counts accumulate and move the cursor 1:1 in framebuffer pixels.
    k.apply_relative_motion(10.0, 0.0)
    k.apply_relative_motion(5.0, -4.0)
    assert_true(_feq(k.pending_rel_x, 15.0) and _feq(k.pending_rel_y, -4.0))
    assert_true(_feq(k.cursor_x, 1935.0) and _feq(k.cursor_y, 1076.0))
    # Clamped to the mode, not to the logical size.
    k.apply_relative_motion(5000.0, 5000.0)
    assert_true(_feq(k.cursor_x, 3839.0) and _feq(k.cursor_y, 2159.0))
    k.apply_relative_motion(-9000.0, -9000.0)
    assert_true(_feq(k.cursor_x, 0.0) and _feq(k.cursor_y, 0.0))
    # A swapchain at another size: same place on screen, reported without a jump delta.
    k.cursor_x = 1920.0
    k.cursor_y = 1080.0
    k.resize_pointer_space(1920, 1080)
    assert_true(_feq(k.cursor_x, 960.0) and _feq(k.cursor_y, 540.0))
    assert_equal(k.framebuffer_size()[0], 1920)
    assert_true(k.pointer_rebased)


def main() raises:
    test_text_input_v3_decoding()
    test_wayland_text_input_without_protocol()
    test_cursor_rect_to_surface()
    test_ui_text_input_ime_commit_and_delete()
    test_xft_dpi_parsing_and_scale()
    test_kms_pointer_framebuffer_space()
