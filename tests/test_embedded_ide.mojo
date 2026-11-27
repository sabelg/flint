"""Regression checks for fragmented PTY frames, bounds safety, RAII and Studio's focus shortcut."""
from std.ffi import external_call
from std.math import abs
from std.testing import assert_equal, assert_true, assert_false
from flint.platform.input import InputState
from flint.devtools.diff_view import DIFF_ROW_NOTE, NO_EOL_NOTE, DiffDocument, DiffSide
from flint.devtools.mojo_highlight import (
    LANG_JS,
    LANG_MARKDOWN,
    LANG_PYTHON,
    LANG_RUST,
    MojoHighlighter,
    STATE_MD_FENCE,
    STATE_NORMAL,
    TOK_COMMENT,
    TOK_DEF,
    TOK_KEYWORD,
    TOK_NUMBER,
    TOK_STRING,
    TOK_SYMBOL,
    TOK_TEXT,
    HighlightSpan,
    js_in_template,
    js_template_frames,
    lang_for_path,
    md_fence_lang,
    md_in_fence,
    next_line_state,
)
from flint.devtools.problems import (
    DIAG_ERROR,
    DIAG_SRC_BUILD,
    DIAG_WARNING,
    Diagnostic,
    ProblemStore,
    parse_compiler_output,
)
from flint.platform.os import env_value, file_mode, make_dirs, read_bytes, remove_file, write_bytes, write_file_durable
from flint.devtools.safe_fs import resolve_confined_path, write_confined_durable
from flint.studio.code_drawer import (
    CODE_TAB_WIDTH,
    CodeDrawer,
    advance_text,
    byte_offset_for_col,
    diag_squiggle_span,
)
from flint.ui.theme import StudioTheme
from flint.platform.os import list_directory
from flint.platform.traits import WindowBackend
from flint.render.font import SdfFont
from flint.studio.ide_grid import IdeGrid, WIDE_CONTINUATION, char_width, palette
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


def _feed(mut g: IdeGrid, text: String):
    for byte in text.as_bytes():
        g.feed(Int(byte))


def test_grid_escapes_width_and_scroll() raises:
    # ESC intermediates: `ESC ( B` (designate G0) must not print the final byte.
    var g = IdeGrid()
    g.resize(16, 3)
    _feed(g, "\x1b(BA\x1b)0C\x1b#8D")
    assert_equal(g.cell(0, 0).code, 65)
    assert_equal(g.cell(0, 1).code, 67)
    assert_equal(g.cell(0, 2).code, 68)
    assert_equal(g.cell(0, 3).code, 32)
    # CSI prefixes: `CSI > 4;2 m` is modifyOtherKeys, not SGR underline / colour.
    _feed(g, "\x1b[31m\x1b[>4;2m\x1b[=5m\x1b[<4mX")
    assert_equal(g.cell(0, 3).code, 88)
    assert_equal(g.cell(0, 3).fg, palette(1))
    assert_false(g.cell(0, 3).underline)
    _feed(g, "\x1b[4mU\x1b[0m")
    assert_true(g.cell(0, 4).underline)

    # Display widths: CJK and emoji take two cells, combining marks / U+FE0F attach.
    assert_equal(char_width(0x41), 1)
    assert_equal(char_width(0x754C), 2)
    assert_equal(char_width(0x1F600), 2)
    assert_equal(char_width(0x0301), 0)
    assert_equal(char_width(0xFE0F), 0)
    assert_equal(char_width(0x00E9), 1)
    var w = IdeGrid()
    w.resize(10, 3)
    _feed(w, "界a😀e\u0301b")
    assert_equal(w.cell(0, 0).code, 0x754C)
    assert_equal(w.cell(0, 1).code, WIDE_CONTINUATION)
    assert_equal(w.cell(0, 2).code, 97)
    assert_equal(w.cell(0, 3).code, 0x1F600)
    assert_equal(w.cell(0, 4).code, WIDE_CONTINUATION)
    assert_equal(w.cell(0, 5).code, 101)
    assert_equal(w.mark(0, 5), "\u0301")
    assert_equal(w.cell(0, 6).code, 98)
    assert_equal(w.x, 7)
    _feed(w, "\r\n\u2764\ufe0fZ")
    assert_equal(w.cell(1, 0).code, 0x2764)
    assert_equal(w.mark(1, 0), "\ufe0f")
    assert_equal(w.cell(1, 1).code, 90)
    # A mark after a wide glyph attaches to its lead cell.
    _feed(w, "\r\n界\u0301")
    assert_equal(w.mark(2, 0), "\u0301")
    # A wide glyph that does not fit in the last column wraps whole.
    var e = IdeGrid()
    e.resize(3, 3)
    _feed(e, "ab界")
    assert_equal(e.cell(0, 2).code, 32)
    assert_equal(e.cell(1, 0).code, 0x754C)
    assert_equal(e.cell(1, 1).code, WIDE_CONTINUATION)
    # Overwriting half of a wide glyph blanks the other half.
    _feed(e, "\x1b[2;2HQ")
    assert_equal(e.cell(1, 0).code, 32)
    assert_equal(e.cell(1, 1).code, 81)

    # TAB moves to the next 8-column stop (and clamps to the last column).
    var t = IdeGrid()
    t.resize(20, 2)
    _feed(t, "a\tb\tc\td")
    assert_equal(t.cell(0, 8).code, 98)
    assert_equal(t.cell(0, 16).code, 99)
    assert_equal(t.cell(0, 19).code, 100)

    # Ring-buffer scroll: a bottom line feed rotates `top` and clears one row.
    var r = IdeGrid()
    r.resize(4, 3)
    _feed(r, "1\r\n2\r\n3\r\n4")
    assert_equal(r.top, 1)
    assert_equal(r.cell(0, 0).code, 50)
    assert_equal(r.cell(1, 0).code, 51)
    assert_equal(r.cell(2, 0).code, 52)
    _feed(r, "\r\n5\r\n6\r\n7\r\n8")
    assert_equal(r.row_text(0), "6   ")
    assert_equal(r.row_text(2), "8   ")
    # Cursor addressing and erase stay logical after scrolling.
    _feed(r, "\x1b[1;1HZ\x1b[3;1H\x1b[2K")
    assert_equal(r.row_text(0), "Z   ")
    assert_equal(r.row_text(2), "    ")
    var ui = UiDrawList()
    var f = SdfFont()
    draw_workspace(ui, r, 200, 120, f, False, -1, "", [], -1, [], False)
    print("IdeGrid ESC intermediates, CSI prefixes, wcwidth, combining marks, TAB and ring scroll passed.")


def _write_text(path: String, text: String) raises:
    var data = List[UInt8](text.as_bytes())
    assert_true(write_bytes(path, data))


def _read_text(path: String) -> String:
    var raw = read_bytes(path)
    return String(unsafe_from_utf8=raw) if len(raw) > 0 else String("")


def _scratch_dir() raises -> String:
    var base = env_value("TMPDIR")
    if base.byte_length() == 0:
        base = "/tmp"
    var d = base + "/flint-code-drawer-test-" + String(Int(external_call["getpid", Int32]()))
    assert_true(make_dirs(d))
    return d


def test_diff_line_endings() raises:
    # A final-newline change gets its own hunk: Accept All yields the new EOF state.
    var d0 = DiffDocument("a.txt", "x\ny", "z\ny\n")
    assert_equal(d0.hunk_count(), 2)
    d0.accept_all()
    assert_equal(d0.materialized_text(), "z\ny\n")
    var d1 = DiffDocument("a.txt", "x\ny", "z\ny\n")
    d1.reject_all()
    assert_equal(d1.materialized_text(), "x\ny")
    var d2 = DiffDocument("a.txt", "x\ny", "z\ny\n")
    _ = d2.accept_hunk(0)
    _ = d2.reject_hunk(1)
    assert_equal(d2.materialized_text(), "z\ny")
    var d3 = DiffDocument("a.txt", "a\nb\n", "a\nb")
    assert_equal(d3.hunk_count(), 1)
    d3.accept_all()
    assert_equal(d3.materialized_text(), "a\nb")
    var saw_note = False
    var rows = d3.unified_rows()
    for i in range(len(rows)):
        if rows[i].kind == DIFF_ROW_NOTE and rows[i].text == NO_EOL_NOTE:
            saw_note = True
    assert_true(saw_note)

    # CRLF files keep CRLF, even when the proposal was written with LF.
    var c0 = DiffDocument("w.txt", "a\r\nb\r\nc\r\n", "a\r\nB\r\nc\r\n")
    assert_equal(c0.hunk_count(), 1)
    c0.accept_all()
    assert_equal(c0.materialized_text(), "a\r\nB\r\nc\r\n")
    var c1 = DiffDocument("w.txt", "a\r\nb\r\nc\r\n", "a\nB\nc\n")
    assert_equal(c1.hunk_count(), 1)
    c1.accept_all()
    assert_equal(c1.materialized_text(), "a\r\nB\r\nc\r\n")
    c1.reject_all()
    assert_equal(c1.materialized_text(), "a\r\nb\r\nc\r\n")
    # A stray CR inside an LF file survives byte-for-byte.
    var c2 = DiffDocument("w.txt", "a\nb\r\nc\n", "a\nb\r\nC\n")
    c2.accept_all()
    assert_equal(c2.materialized_text(), "a\nb\r\nC\n")
    assert_equal(DiffSide.from_text("p\r\nq").to_text(), "p\r\nq")
    print("Diff CRLF preservation and final-newline hunks passed.")


def test_highlight_fences_and_dialects() raises:
    var hl = MojoHighlighter()
    var md = List[String]()
    md.append("```python")
    md.append('x = """doc')
    md.append('still doc"""')
    md.append("```")
    md.append("# Heading")
    var spans = hl.highlight_lines(md, LANG_MARKDOWN)
    assert_equal(spans[2][0].token_kind, TOK_STRING)  # inside a multi-line string in the fence
    assert_equal(spans[4][0].token_kind, TOK_DEF)     # fence closed: Markdown heading again
    var st = STATE_NORMAL
    for i in range(3):
        st = next_line_state(md[i], st, LANG_MARKDOWN)
    assert_true(md_in_fence(st))
    assert_equal(md_fence_lang(st), LANG_PYTHON)
    assert_equal(next_line_state("```", STATE_NORMAL, LANG_MARKDOWN), STATE_MD_FENCE)
    # Rust lifetimes no longer swallow the rest of the line as a char literal.
    assert_equal(lang_for_path("src/lib.rs"), LANG_RUST)
    assert_equal(lang_for_path("web/app.ts"), LANG_JS)
    var rs = hl.highlight_line("fn f<'a>(x: &'a str) -> u32 { 1 }", STATE_NORMAL, LANG_RUST)
    var saw_num = False
    for i in range(len(rs[0])):
        if rs[0][i].token_kind == TOK_NUMBER:
            saw_num = True
        assert_true(rs[0][i].token_kind != TOK_STRING)
    assert_true(saw_num)
    var js = hl.highlight_line("const s = `a ${b}`; // c", STATE_NORMAL, LANG_JS)
    assert_equal(js[0][0].token_kind, TOK_KEYWORD)
    assert_equal(js[0][len(js[0]) - 1].token_kind, TOK_COMMENT)

    # Diff rows take their spans from whole-file highlighting (multi-line string state).
    var cd = CodeDrawer()
    var old_src = String('def f():\n    """start\n    middle\n    end"""\n    return 1\n')
    var new_src = String('def f():\n    """start\n    middle changed\n    end"""\n    return 1\n')
    _ = cd.show_agent_proposal("doc.mojo", old_src, new_src)
    cd.tabs[0].ensure_diff_spans(cd.highlighter)
    assert_equal(cd.tabs[0].diff_new_spans[2][0].token_kind, TOK_STRING)
    assert_equal(cd.tabs[0].diff_old_spans[2][0].token_kind, TOK_STRING)
    print("Markdown fence state, Rust/JS dialects and diff-row highlighting passed.")


def test_code_drawer_files() raises:
    var dir = _scratch_dir()
    var path = dir + "/hero.mojo"
    _write_text(path, "def a():\n    pass\n")
    var cd = CodeDrawer()

    # A failed read reports instead of opening an empty tab.
    assert_equal(cd.open_file(dir + "/missing.mojo", 3), -1)
    assert_true(cd.last_error.byte_length() > 0)
    assert_equal(len(cd.tabs), 0)

    var ti = cd.open_file(path, 2)
    assert_equal(ti, 0)
    assert_equal(cd.tabs[0].cursor_line, 1)
    assert_equal(cd.tabs[0].disk_path, path)

    # Unsaved edits survive re-opening (from disk or from memory).
    cd.tabs[0].text = "def a():\n    return 2\n"
    cd.tabs[0].dirty = True
    cd.tabs[0].rebuild_caches(cd.highlighter)
    assert_equal(cd.open_file(path, 1), 0)
    assert_equal(cd.tabs[0].text, "def a():\n    return 2\n")
    _ = cd.open_source(path, "def a():\n    pass\n")
    assert_equal(cd.tabs[0].text, "def a():\n    return 2\n")
    assert_true(cd.tabs[0].disk_conflict)
    cd.keep_tab_buffer(0)
    assert_false(cd.tabs[0].disk_conflict)

    # Save is durable and clears dirty.
    assert_true(cd.save_tab(0))
    assert_false(cd.tabs[0].dirty)
    assert_equal(_read_text(path), "def a():\n    return 2\n")

    # External change on a clean tab reloads it.
    _write_text(path, "def a():\n    return 33\n")
    assert_equal(cd.poll_disk_changes(force=True), 1)
    assert_equal(cd.tabs[0].text, "def a():\n    return 33\n")
    assert_false(cd.tabs[0].dirty)

    # External change under unsaved edits flags a conflict and keeps the buffer.
    cd.tabs[0].text = "def a():\n    return 4\n"
    cd.tabs[0].dirty = True
    _write_text(path, "def a():\n    return 555\n")
    assert_equal(cd.poll_disk_changes(force=True), 1)
    assert_true(cd.tabs[0].disk_conflict)
    assert_equal(cd.tabs[0].text, "def a():\n    return 4\n")
    assert_false(cd.save_tab(0))
    assert_equal(_read_text(path), "def a():\n    return 555\n")
    assert_true(cd.reload_tab_from_disk(0))
    assert_equal(cd.tabs[0].text, "def a():\n    return 555\n")
    assert_false(cd.tabs[0].dirty or cd.tabs[0].disk_conflict)

    # Closing a dirty tab asks first; force discards.
    cd.tabs[0].text = "changed\n"
    cd.tabs[0].dirty = True
    cd.close_tab(0)
    assert_equal(len(cd.tabs), 1)
    assert_equal(cd.pending_close_tab, 0)
    cd.close_tab(0, force=True)
    assert_equal(len(cd.tabs), 0)
    assert_equal(cd.pending_close_tab, -1)

    # Agent-facing API: materialized text of a pending proposal, save by path.
    var pi = cd.open_file(path)
    var cur_pi = cd.tab_text(path)
    _ = cd.show_agent_proposal(path, cur_pi, "def a():\n    return 6\n")
    assert_equal(cd.materialized_text(pi, accept_pending=True), "def a():\n    return 6\n")
    assert_equal(cd.materialized_text(pi), "def a():\n    return 555\n")
    cd.apply_all_diff(True)
    assert_true(cd.tabs[pi].dirty)
    assert_true(cd.save_path(path))
    assert_equal(_read_text(path), "def a():\n    return 6\n")
    assert_equal(cd.tab_text(path), "def a():\n    return 6\n")

    # CRLF file: accepting a proposal and saving keeps CRLF on disk.
    var crlf_path = dir + "/win.toml"
    _write_text(crlf_path, "a = 1\r\nb = 2\r\n")
    var wi = cd.open_file(crlf_path)
    var cur_wi = cd.tab_text(crlf_path)
    _ = cd.show_agent_proposal(crlf_path, cur_wi, "a = 1\nb = 3\n")
    cd.apply_all_diff(True)
    assert_true(cd.save_tab(wi))
    assert_equal(_read_text(crlf_path), "a = 1\r\nb = 3\r\n")
    _ = remove_file(path)
    _ = remove_file(crlf_path)
    print("CodeDrawer open/reopen guards, durable save, disk watch & conflicts passed.")


def _kind_at(spans: List[HighlightSpan], byte: Int) -> Int:
    for s in spans:
        if s.start_byte <= byte and byte < s.end_byte:
            return s.token_kind
    return -1


def test_js_multiline_template() raises:
    var hl = MojoHighlighter()
    var src = List[String]()
    src.append("const s = `line one")                          # 0 opens a template
    src.append("still ${user.name + f({a: 1})} text }")        # 1 interpolation, nested braces
    src.append("end` + 42; // tail")                           # 2 closes it
    src.append("let t = `a ${ `inner")                         # 3 nested template in ${}
    src.append("${x}` } b`;")                                  # 4 inner, outer ${} and outer close
    src.append("const z = 1; /* open")                         # 5 block comment still works
    src.append("`not a template` */ var q = 2;")               # 6
    var spans = hl.highlight_lines(src, LANG_JS)
    var l1 = src[1]
    assert_equal(_kind_at(spans[1], 0), TOK_STRING)                       # "still" stays string
    assert_equal(_kind_at(spans[1], l1.find("${")), TOK_SYMBOL)
    assert_equal(_kind_at(spans[1], l1.find("user")), TOK_TEXT)           # code, not string
    assert_equal(_kind_at(spans[1], l1.find("f(")), TOK_DEF)
    assert_equal(_kind_at(spans[1], l1.find("1")), TOK_NUMBER)
    assert_equal(_kind_at(spans[1], l1.find("})}") + 2), TOK_SYMBOL)      # closing `}` of ${}
    assert_equal(_kind_at(spans[1], l1.find("text")), TOK_STRING)         # back in the template
    assert_equal(_kind_at(spans[1], l1.byte_length() - 1), TOK_STRING)    # stray `}` is text
    assert_equal(_kind_at(spans[2], 0), TOK_STRING)
    assert_equal(_kind_at(spans[2], 3), TOK_STRING)                       # the closing backtick
    assert_equal(_kind_at(spans[2], src[2].find("42")), TOK_NUMBER)
    assert_equal(_kind_at(spans[2], src[2].find("//")), TOK_COMMENT)
    assert_equal(_kind_at(spans[3], 0), TOK_KEYWORD)
    assert_equal(_kind_at(spans[3], src[3].find("inner")), TOK_STRING)
    var l4 = src[4]
    assert_equal(_kind_at(spans[4], 0), TOK_SYMBOL)                       # inner `${`
    assert_equal(_kind_at(spans[4], 2), TOK_TEXT)                         # x
    assert_equal(_kind_at(spans[4], 4), TOK_STRING)                       # inner closing backtick
    assert_equal(_kind_at(spans[4], l4.find("} b")), TOK_SYMBOL)          # outer ${} closes
    assert_equal(_kind_at(spans[4], l4.find(" b") + 1), TOK_STRING)
    assert_equal(_kind_at(spans[4], l4.byte_length() - 1), TOK_SYMBOL)   # `;` back in code
    assert_equal(_kind_at(spans[5], 0), TOK_KEYWORD)
    assert_equal(_kind_at(spans[6], 1), TOK_COMMENT)                      # backtick in a comment
    assert_equal(_kind_at(spans[6], src[6].find("var")), TOK_KEYWORD)

    # Line states (the drawer's incremental path) agree with the highlighter's.
    var st = STATE_NORMAL
    var expect_in = [True, True, False, True, False, False, False]
    for i in range(len(src)):
        st = next_line_state(src[i], st, LANG_JS)
        assert_equal(js_in_template(st), expect_in[i])
        if i == 3:
            var fr = js_template_frames(st)
            assert_equal(len(fr), 3)                                      # text, ${ depth 0, text
            assert_equal(fr[0], -1)
            assert_equal(fr[1], 0)
            assert_equal(fr[2], -1)
    assert_equal(st, STATE_NORMAL)
    # Braces inside an interpolation are balanced across lines.
    var b = List[String]()
    b.append("x = `${ {")
    b.append("  a: `}`, b: '}' }.a } tail")
    b.append("` + 1")
    var bs = hl.highlight_lines(b, LANG_JS)
    var s1 = next_line_state(b[0], STATE_NORMAL, LANG_JS)
    assert_equal(js_template_frames(s1)[1], 1)
    assert_equal(_kind_at(bs[1], b[1].find("tail")), TOK_STRING)
    assert_equal(_kind_at(bs[1], b[1].find("a:")), TOK_TEXT)
    assert_equal(_kind_at(bs[2], b[2].find("1")), TOK_NUMBER)
    assert_equal(next_line_state(b[2], next_line_state(b[1], s1, LANG_JS), LANG_JS), STATE_NORMAL)
    # A Markdown ```js fence carries the template state inside its packed state.
    var md = List[String]()
    md.append("```js")
    md.append("const m = `one")
    md.append("two`;")
    md.append("```")
    var ms = hl.highlight_lines(md, LANG_MARKDOWN)
    assert_equal(_kind_at(ms[2], 0), TOK_STRING)
    assert_equal(_kind_at(ms[2], 4), TOK_SYMBOL)
    print("JS/TS multi-line template literals, ${} interpolation and nesting passed.")


def test_wcwidth_tables() raises:
    # CJK / fullwidth / Hangul.
    assert_equal(char_width(0x4E00), 2)
    assert_equal(char_width(0x3000), 2)    # ideographic space (F)
    assert_equal(char_width(0xFF21), 2)    # fullwidth A
    assert_equal(char_width(0xAC00), 2)    # Hangul syllable
    assert_equal(char_width(0x1100), 2)    # leading jamo
    assert_equal(char_width(0x1160), 0)    # medial jamo
    assert_equal(char_width(0x11A8), 0)    # final jamo
    assert_equal(char_width(0xD7B0), 0)
    assert_equal(char_width(0x20000), 2)
    assert_equal(char_width(0x3FFFD), 2)
    assert_equal(char_width(0xFF61), 1)    # halfwidth katakana
    # Emoji presentation.
    assert_equal(char_width(0x1F600), 2)
    assert_equal(char_width(0x1F9D1), 2)
    assert_equal(char_width(0x1FAF8), 2)
    assert_equal(char_width(0x231A), 2)
    assert_equal(char_width(0x2764), 1)    # text presentation by default
    assert_equal(char_width(0x1F3FB), 0)   # skin-tone modifier attaches
    # Combining / format / selectors.
    assert_equal(char_width(0x0301), 0)
    assert_equal(char_width(0x20DD), 0)    # Me
    assert_equal(char_width(0x0E31), 0)
    assert_equal(char_width(0x0F71), 0)    # Tibetan (missing from the old table)
    assert_equal(char_width(0x200D), 0)    # ZWJ
    assert_equal(char_width(0x200B), 0)
    assert_equal(char_width(0xFE0F), 0)
    assert_equal(char_width(0xE0100), 0)
    assert_equal(char_width(0xE0041), 0)   # tag
    assert_equal(char_width(0x00AD), 1)    # soft hyphen prints
    assert_equal(char_width(0x0915), 1)
    # Controls.
    assert_equal(char_width(0x00), -1)
    assert_equal(char_width(0x07), -1)
    assert_equal(char_width(0x1B), -1)
    assert_equal(char_width(0x7F), -1)
    assert_equal(char_width(0x85), -1)
    assert_equal(char_width(0x9F), -1)
    assert_equal(char_width(0xA0), 1)
    var g = IdeGrid()
    g.resize(12, 2)
    _feed(g, "👩\u200d💻a\u0085b")
    assert_equal(g.cell(0, 0).code, 0x1F469)
    assert_equal(g.mark(0, 0), "\u200d")
    assert_equal(g.cell(0, 2).code, 0x1F4BB)
    assert_equal(g.cell(0, 4).code, 97)
    assert_equal(g.cell(0, 5).code, 98)    # UTF-8 encoded C1 control takes no cell
    assert_equal(g.x, 6)
    print("wcwidth tables: CJK, emoji, combining, ZWJ, jamo and controls passed.")


def test_durable_save_keeps_mode() raises:
    var dir = _scratch_dir()
    var script = dir + "/run.sh"
    _write_text(script, "#!/bin/sh\necho 1\n")
    var owned = script
    assert_equal(external_call["chmod", Int32](owned.as_c_string_span(), UInt32(0o755)), 0)
    var data = List[UInt8]("#!/bin/sh\necho 2\n".as_bytes())
    assert_true(write_file_durable(script, data))
    assert_equal(file_mode(script), 0o755)
    assert_equal(_read_text(script), "#!/bin/sh\necho 2\n")
    # Code Drawer save keeps the exec bit too (and a private 0600 file stays private).
    var cd = CodeDrawer()
    var ti = cd.open_file(script)
    cd.tabs[ti].text = "#!/bin/sh\necho 3\n"
    cd.tabs[ti].dirty = True
    assert_true(cd.save_tab(ti))
    assert_equal(file_mode(script), 0o755)
    var secret = dir + "/secret.toml"
    _write_text(secret, "k = 1\n")
    var owned_secret = secret
    assert_equal(external_call["chmod", Int32](owned_secret.as_c_string_span(), UInt32(0o600)), 0)
    var sd = List[UInt8]("k = 2\n".as_bytes())
    assert_true(write_file_durable(secret, sd))
    assert_equal(file_mode(secret), 0o600)
    # Confined agent writes (safe_fs) go through the same path.
    var cp = resolve_confined_path(dir, "run.sh")
    var w = write_confined_durable(cp, "#!/bin/sh\necho 4\n")
    assert_true(w[0])
    assert_equal(file_mode(script), 0o755)
    # A new file gets 0666 & ~umask.
    var fresh = dir + "/fresh.txt"
    _ = remove_file(fresh)
    var um = Int(external_call["umask", UInt32](UInt32(0o022)))
    _ = external_call["umask", UInt32](UInt32(um))
    var fd = List[UInt8]("new\n".as_bytes())
    assert_true(write_file_durable(fresh, fd))
    assert_equal(file_mode(fresh), 0o666 & ~um)
    _ = remove_file(script)
    _ = remove_file(secret)
    _ = remove_file(fresh)
    print("Durable save keeps file mode bits (exec, 0600), new files honour umask passed.")


def test_code_drawer_view() raises:
    var cd = CodeDrawer()
    var body = String("")
    for i in range(100):
        body += "line " + String(i) + "\n"
    _ = cd.open_source("long.mojo", body)
    var theme = StudioTheme.flint_obsidian()
    var font = SdfFont()
    var inp = InputState()
    var dl = UiDrawList()
    _ = cd.draw_and_handle(dl, 0, font, theme, 0.0, 0.0, 640.0, 720.0, inp, 0.0)
    var rows = cd.visible_rows
    assert_true(rows > 12)
    # jump_to_line uses the real visible row count: an on-screen line does not scroll.
    cd.jump_to_line(rows - 1)
    assert_equal(cd.tabs[0].scroll_line, 0)
    cd.jump_to_line(80)
    assert_true(cd.tabs[0].scroll_line <= 79 and cd.tabs[0].scroll_line + rows > 79)
    # Down past the last visible row scrolls the cursor into view.
    cd.tabs[0].scroll_line = 0
    cd.tabs[0].cursor_line = rows - 1
    cd.has_keyboard_focus = True
    var down = InputState()
    down.begin_frame()
    down.set_key(264, True)
    var dl2 = UiDrawList()
    _ = cd.draw_and_handle(dl2, 0, font, theme, 0.0, 0.0, 640.0, 720.0, down, 0.0)
    assert_equal(cd.tabs[0].cursor_line, rows)
    assert_equal(cd.tabs[0].scroll_line, 1)

    # Squiggles: byte column -> tab-expanded prefix width.
    assert_equal(byte_offset_for_col("a界b", 3), 1)  # mid-codepoint column snaps back
    var mono = SdfFont()
    if mono.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32, 5):
        var px = Float32(13.0)
        var sp_w = mono.measure(" ", px)
        var sq = diag_squiggle_span(mono, px, "\tfoo(bar)", 2)
        assert_true(abs(sq[0] - sp_w * Float32(CODE_TAB_WIDTH)) < Float32(0.01))
        assert_true(abs(sq[1] - mono.measure("foo", px)) < Float32(0.01))
        var sq2 = diag_squiggle_span(mono, px, "\tfoo(bar)", 6)
        assert_true(abs(sq2[0] - advance_text(mono, px, 0.0, 0.0, "\tfoo(")) < Float32(0.01))
        assert_true(abs(advance_text(mono, px, 0.0, 0.0, "ab\tc") - sp_w * Float32(CODE_TAB_WIDTH + 1)) < Float32(0.01))

    # MCP diagnostics mirror: only sources that changed there are replaced.
    var ext = ProblemStore()
    ext.replace_source(DIAG_SRC_BUILD, parse_compiler_output("/p/a.mojo:3:4: error: bad\n"))
    cd.mirror_problem_store(ext)
    assert_equal(cd.problems.error_count(), 1)
    var own = List[Diagnostic]()
    own.append(Diagnostic(DIAG_SRC_BUILD, DIAG_WARNING, "/p/b.mojo", 1, 1, "own build"))
    own.append(Diagnostic(DIAG_SRC_BUILD, DIAG_ERROR, "/p/b.mojo", 2, 1, "own build"))
    cd.problems.replace_source(DIAG_SRC_BUILD, own)
    cd.mirror_problem_store(ext)
    assert_equal(cd.problems.warning_count(), 1)
    ext.replace_source(DIAG_SRC_BUILD, List[Diagnostic]())
    cd.mirror_problem_store(ext)
    assert_equal(cd.problems.error_count() + cd.problems.warning_count(), 0)
    print("CodeDrawer visible rows, cursor scroll, tab-aware squiggles and problem mirror passed.")


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
    test_grid_escapes_width_and_scroll()
    test_diff_line_endings()
    test_highlight_fences_and_dialects()
    test_code_drawer_files()
    test_js_multiline_template()
    test_wcwidth_tables()
    test_durable_save_keeps_mode()
    test_code_drawer_view()
    print("Embedded IDE grid, UTF-8 input, F2, bounds safety, RAII and generic WindowBackend passed.")
