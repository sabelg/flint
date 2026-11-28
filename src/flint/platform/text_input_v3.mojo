"""`zwp_text_input_v3` (text-input-unstable-v3) wire codec and state, for IME composition on Wayland.

The window backend owns the protocol objects (manager global, one text_input per seat) and sends the
requests built here; this module decodes the events into the double-buffered state the protocol
describes and hands the applied result to `InputState` (`set_ime_preedit` / `commit_ime_preedit` /
`delete_ime_surrounding`), which `UiContext` already turns into preedit display and committed text.

Events (zwp_text_input_v3): 0 enter(surface), 1 leave(surface),
2 preedit_string(text?, cursor_begin, cursor_end), 3 commit_string(text?),
4 delete_surrounding_text(before_length, after_length), 5 done(serial).
Requests: 0 destroy, 1 enable, 2 disable, 3 set_surrounding_text, 4 set_text_change_cause,
5 set_content_type(hint, purpose), 6 set_cursor_rectangle(x, y, w, h), 7 commit.
Manager (zwp_text_input_manager_v3): 0 destroy, 1 get_text_input(new_id, seat).
"""
from flint.platform.input import InputState

comptime TI3_EV_ENTER: Int = 0
comptime TI3_EV_LEAVE: Int = 1
comptime TI3_EV_PREEDIT_STRING: Int = 2
comptime TI3_EV_COMMIT_STRING: Int = 3
comptime TI3_EV_DELETE_SURROUNDING: Int = 4
comptime TI3_EV_DONE: Int = 5

comptime TI3_REQ_DESTROY: Int = 0
comptime TI3_REQ_ENABLE: Int = 1
comptime TI3_REQ_DISABLE: Int = 2
comptime TI3_REQ_SET_CONTENT_TYPE: Int = 5
comptime TI3_REQ_SET_CURSOR_RECTANGLE: Int = 6
comptime TI3_REQ_COMMIT: Int = 7

comptime TI3_MANAGER_GET_TEXT_INPUT: Int = 1


def _ti_get32(data: List[UInt8], offset: Int) -> Int:
    return Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)


def _ti_gets32(data: List[UInt8], offset: Int) -> Int:
    var value = _ti_get32(data, offset)
    return value - 4294967296 if value >= 2147483648 else value


def _ti_string(data: List[UInt8], offset: Int, size: Int) -> Tuple[String, Int]:
    """A (nullable) string argument at `offset`: its text ("" for null) and the offset past it."""
    var end = min(size, len(data))
    if offset + 4 > end:
        return (String(""), end)
    var length = _ti_get32(data, offset)
    var next = offset + 4 + ((length + 3) & ~3)
    if length <= 1 or offset + 4 + length > end:
        return (String(""), min(next, end))
    var bytes = List[UInt8](capacity=length - 1)
    for i in range(length - 1):
        bytes.append(data[offset + 4 + i])
    return (String(unsafe_from_utf8=bytes), next)


def utf8_codepoints_in(text: String, byte_begin: Int, byte_end: Int) -> Int:
    """Codepoints that start inside `text[byte_begin:byte_end]` (byte offsets clamped to the text)."""
    var bytes = text.as_bytes()
    var n = len(bytes)
    var lo = max(0, min(n, byte_begin))
    var hi = max(lo, min(n, byte_end))
    var count = 0
    for i in range(lo, hi):
        if (Int(bytes[i]) & 0xC0) != 0x80:
            count += 1
    return count


struct TextInputV3State(Copyable, Movable):
    """Pending (double-buffered) and applied IME state of one `zwp_text_input_v3`."""
    # Pending, applied and reset on `done`.
    var pending_preedit: String
    var pending_cursor_begin: Int
    var pending_cursor_end: Int
    var pending_commit: String
    var pending_delete_before: Int
    var pending_delete_after: Int
    # Applied, handed to `InputState` by `flush_into`.
    var preedit: String
    var preedit_cursor_cp: Int   # caret inside the preedit, in codepoints (end of the preedit if hidden)
    var preedit_sel_len: Int     # highlighted clause length in codepoints
    var preedit_dirty: Bool
    var commit_text: String      # committed since the last flush
    var delete_before: Int       # bytes since the last flush
    var delete_after: Int
    # Focus and enable state.
    var entered_surface: Int     # surface the text input focus is on (0 = none)
    var enabled: Bool            # enable sent and not yet disabled
    var commit_count: Int        # commit requests sent (the serial `done` echoes back)
    var last_done_serial: Int
    var done_count: Int
    var rect_x: Int              # last cursor rectangle sent (surface-local)
    var rect_y: Int
    var rect_w: Int
    var rect_h: Int

    def __init__(out self):
        self.pending_preedit = String()
        self.pending_cursor_begin = 0
        self.pending_cursor_end = 0
        self.pending_commit = String()
        self.pending_delete_before = 0
        self.pending_delete_after = 0
        self.preedit = String()
        self.preedit_cursor_cp = 0
        self.preedit_sel_len = 0
        self.preedit_dirty = False
        self.commit_text = String()
        self.delete_before = 0
        self.delete_after = 0
        self.entered_surface = 0
        self.enabled = False
        self.commit_count = 0
        self.last_done_serial = 0
        self.done_count = 0
        self.rect_x = -1
        self.rect_y = -1
        self.rect_w = -1
        self.rect_h = -1

    def _reset_pending(mut self):
        self.pending_preedit = String()
        self.pending_cursor_begin = 0
        self.pending_cursor_end = 0
        self.pending_commit = String()
        self.pending_delete_before = 0
        self.pending_delete_after = 0

    def handle_event(mut self, pkt: List[UInt8], opcode: Int, size: Int):
        """Decodes one event (full packet with its 8-byte header) of the text_input object."""
        if opcode == TI3_EV_ENTER and size >= 12:
            self.entered_surface = _ti_get32(pkt, 8)
            self.enabled = False  # enable must follow enter; the backend sends it when a field wants text
        elif opcode == TI3_EV_LEAVE:
            self.entered_surface = 0
            self.enabled = False
            self._reset_pending()
            if self.preedit.byte_length() > 0:
                self.preedit = String()
                self.preedit_cursor_cp = 0
                self.preedit_sel_len = 0
                self.preedit_dirty = True
        elif opcode == TI3_EV_PREEDIT_STRING and size >= 20:
            var s = _ti_string(pkt, 8, size)
            var off = s[1]
            self.pending_preedit = s[0]
            if off + 8 <= size:
                self.pending_cursor_begin = _ti_gets32(pkt, off)
                self.pending_cursor_end = _ti_gets32(pkt, off + 4)
            else:
                self.pending_cursor_begin = self.pending_preedit.byte_length()
                self.pending_cursor_end = self.pending_cursor_begin
        elif opcode == TI3_EV_COMMIT_STRING and size >= 12:
            self.pending_commit = _ti_string(pkt, 8, size)[0]
        elif opcode == TI3_EV_DELETE_SURROUNDING and size >= 16:
            self.pending_delete_before = _ti_get32(pkt, 8)
            self.pending_delete_after = _ti_get32(pkt, 12)
        elif opcode == TI3_EV_DONE and size >= 12:
            self.apply_done(_ti_get32(pkt, 8))

    def apply_done(mut self, serial: Int):
        """`done`: applies the pending state in protocol order (preedit replaced, surrounding text
        deleted, commit inserted, new preedit placed) and resets it. A serial that does not match the
        commit count is still applied, as the protocol asks."""
        self.last_done_serial = serial
        self.done_count += 1
        if self.pending_delete_before > 0 or self.pending_delete_after > 0:
            self.delete_before += self.pending_delete_before
            self.delete_after += self.pending_delete_after
        if self.pending_commit.byte_length() > 0:
            self.commit_text += self.pending_commit
        var text = self.pending_preedit
        var cursor_cp: Int
        var sel_len = 0
        var n = text.byte_length()
        if self.pending_cursor_begin < 0 or self.pending_cursor_end < 0:
            cursor_cp = utf8_codepoints_in(text, 0, n)  # cursor hidden: keep the caret after the preedit
        else:
            var lo = min(self.pending_cursor_begin, self.pending_cursor_end)
            var hi = max(self.pending_cursor_begin, self.pending_cursor_end)
            cursor_cp = utf8_codepoints_in(text, 0, lo)
            sel_len = utf8_codepoints_in(text, lo, hi)
        if text != self.preedit or cursor_cp != self.preedit_cursor_cp or sel_len != self.preedit_sel_len:
            self.preedit = text
            self.preedit_cursor_cp = cursor_cp
            self.preedit_sel_len = sel_len
            self.preedit_dirty = True
        self._reset_pending()

    def has_output(self) -> Bool:
        return self.preedit_dirty or self.commit_text.byte_length() > 0 or self.delete_before > 0 or self.delete_after > 0

    def flush_into(mut self, mut input: InputState):
        """Hands what `done` applied since the last flush to the frame's input."""
        if self.delete_before > 0 or self.delete_after > 0:
            input.delete_ime_surrounding(self.delete_before, self.delete_after)
            self.delete_before = 0
            self.delete_after = 0
        if self.commit_text.byte_length() > 0:
            input.commit_ime_preedit(self.commit_text)
            self.commit_text = String()
            self.preedit_dirty = True  # the commit cleared the input's preedit: restate the current one
        if self.preedit_dirty:
            if self.preedit.byte_length() > 0:
                input.set_ime_preedit(self.preedit, self.preedit_cursor_cp, self.preedit_sel_len)
            elif input.ime_active:
                input.set_ime_preedit(String(""))
            self.preedit_dirty = False

    def rect_changed(self, x: Int, y: Int, w: Int, h: Int) -> Bool:
        return x != self.rect_x or y != self.rect_y or w != self.rect_w or h != self.rect_h

    def note_rect(mut self, x: Int, y: Int, w: Int, h: Int):
        self.rect_x = x
        self.rect_y = y
        self.rect_w = w
        self.rect_h = h

    def forget_rect(mut self):
        self.rect_x = -1
        self.rect_y = -1
        self.rect_w = -1
        self.rect_h = -1


def ti3_cursor_rect_from_framebuffer(
    fb_x: Float32, fb_y: Float32, fb_w: Float32, fb_h: Float32,
    target_w: Int, target_h: Int, surface_w: Int, surface_h: Int,
) -> Tuple[Int, Int, Int, Int]:
    """A caret rectangle in framebuffer pixels (what the app draws) as surface-local coordinates for
    `set_cursor_rectangle`: the inverse of the backend's pointer mapping."""
    var sx = Float32(1.0)
    var sy = Float32(1.0)
    if target_w > 0 and surface_w > 0:
        sx = Float32(surface_w) / Float32(target_w)
    if target_h > 0 and surface_h > 0:
        sy = Float32(surface_h) / Float32(target_h)
    var x = Int(fb_x * sx + Float32(0.5)) if fb_x >= Float32(0.0) else 0
    var y = Int(fb_y * sy + Float32(0.5)) if fb_y >= Float32(0.0) else 0
    var w = max(1, Int(fb_w * sx + Float32(0.5)))
    var h = max(1, Int(fb_h * sy + Float32(0.5)))
    if surface_w > 0:
        x = min(x, surface_w - 1)
    if surface_h > 0:
        y = min(y, surface_h - 1)
    return (x, y, w, h)
