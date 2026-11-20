"""Comprehensive Verification Suite for Flint v5 Milestone M4c (F27 Multi-OS Win32 & macOS
Backends + Integrated F23–F28 Platform, Input, Text, UI & Socket Parity):
  (a) F27 Win32Window & CocoaWindow: WindowBackend trait conformance, non-Copyable RAII
      move/deinit, safe non-native initialize()->False, content_scale, clipboard round-trip,
      WNDCLASSEXW (80B) / MSG (48B) / RECT (16B) / top-down BITMAPINFOHEADER (40B, biHeight < 0),
      UTF-8 <-> UTF-16LE surrogate pairs, Win32 WM_* & Cocoa NSEvent translation + Y-flip,
      and VulkanGpu.attach_win32_surface(0, 0) / attach_macos_surface(0) null-handle safety.
  (b) F27 Multi-OS UdpSocket / NetTransport: CompilationTarget dispatch, Linux / Win32 / Darwin
      socket constant & sockaddr_in6 / sockaddr_in binary wire layout round-trips (including
      Darwin sin6_len=28 + sin6_family=30), UdpSocket is_valid()/RAII move, and live loopback.
  (c) F27 Win32Audio & MacOsAudio: AudioBackend trait conformance, non-Copyable RAII, safe
      non-native initialize()->False, WAVEFORMATEX (18B) / WAVEFORMATEXTENSIBLE (40B) /
      WAVEHDR (48B) / CoreAudio ASBD (40B 'lpcm') / AudioQueueBuffer (56B) layouts, and
      deterministic 4-slot PCM ring mixing via trigger/set_ambient_drone/pump.
  (d) Integrated F23–F28: Generic WindowBackend & AudioBackend harnesses across all 5 window
      backends (WaylandWindow, X11Window, KmsWindow, Win32Window, CocoaWindow), XkbKeymap +
      EvdevHub.poll_into + not uses_legacy_jsdev + TrueTypeFont/MSDF/BiDi/wrap + UiContext
      text_input + /proc/self/fd zero-FD-leak check.
"""
from std.collections import List
from std.memory import bitcast
from std.sys.info import CompilationTarget, size_of
from std.testing import assert_equal, assert_false, assert_true
from flint.input import EvdevHub
from flint.net.socket_udp import (
    CURRENT_SOCK_OS,
    DARWIN_AF_INET6,
    DARWIN_EAGAIN,
    DARWIN_ENOBUFS,
    DARWIN_SOCKADDR_IN6_LEN,
    IS_WINDOWS,
    LINUX_AF_INET6,
    LINUX_EAGAIN,
    SOCK_OS_LINUX,
    SOCK_OS_MACOS,
    SOCK_OS_WINDOWS,
    WIN32_AF_INET6,
    WIN32_WSAEWOULDBLOCK,
    NetEndpoint,
    NetTransport,
    UdpSocket,
    DatagramView,
)
from flint.platform.input import InputState
from flint.platform.kms import KmsWindow
from flint.platform.macos_audio import (
    K_AUDIO_FORMAT_LINEAR_PCM,
    MACOS_AQ_BUFFER_BYTES,
    MACOS_ASBD_BYTES,
    MACOS_AUDIO_RING_SLOTS,
    MacOsAudio,
    macos_pack_asbd,
    macos_pack_audio_queue_buffer,
)
from flint.platform.macos_window import (
    COCOA_RECT_BYTES,
    NS_EVENT_FLAGS_CHANGED,
    NS_EVENT_KEY_DOWN,
    NS_EVENT_KEY_UP,
    NS_EVENT_LEFT_MOUSE_DOWN,
    NS_EVENT_LEFT_MOUSE_UP,
    NS_EVENT_MOUSE_MOVED,
    NS_EVENT_SCROLL_WHEEL,
    NS_MOD_COMMAND,
    NS_MOD_SHIFT,
    CocoaRect,
    CocoaWindow,
    _filter_cocoa_text_utf8,
    cocoa_flip_y,
    cocoa_keycode_to_flint_key,
    cocoa_pack_rect,
    cocoa_unpack_rect,
)
from flint.platform.os import list_directory
from flint.platform.traits import AudioBackend, WindowBackend
from flint.platform.vulkan import VulkanGpu
from flint.platform.wayland import WaylandWindow
from flint.platform.win32_audio import (
    WAVE_FORMAT_EXTENSIBLE,
    WAVE_FORMAT_PCM,
    WIN32_AUDIO_RING_SLOTS,
    WIN32_WAVEFORMATEX_BYTES,
    WIN32_WAVEFORMATEXTENSIBLE_BYTES,
    WIN32_WAVEHDR_BYTES,
    Win32Audio,
    win32_guid_clsid_mm_device_enumerator,
    win32_guid_iid_iaudio_client,
    win32_guid_iid_iaudio_render_client,
    win32_guid_iid_imm_device_enumerator,
    win32_pack_wavehdr,
    win32_pack_waveformatex,
    win32_pack_waveformatextensible,
)
from flint.platform.win32_window import (
    WM_CHAR,
    WM_CLOSE,
    WM_DPICHANGED,
    WM_KEYDOWN,
    WM_KEYUP,
    WM_LBUTTONDOWN,
    WM_LBUTTONUP,
    WM_MOUSEMOVE,
    WM_MOUSEWHEEL,
    WM_RBUTTONDOWN,
    WM_SIZE,
    WM_UNICHAR,
    WIN32_BITMAPINFOHEADER_BYTES,
    WIN32_MSG_BYTES,
    WIN32_RECT_BYTES,
    WIN32_WNDCLASSEXW_BYTES,
    Win32Window,
    utf16le_to_utf8,
    utf8_to_utf16le,
    win32_pack_bitmapinfoheader,
    win32_pack_msg,
    win32_pack_rect,
    win32_pack_wndclassexw,
    win32_resolve_vk,
    win32_vk_to_flint_key,
)
from flint.platform.x11 import X11Window
from flint.platform.xkb import XkbKeymap, x11_keysym_to_codepoint
from flint.render.font import SdfFont
from flint.text.bidi import BIDI_AUTO, resolve_bidi
from flint.text.line_break import wrap
from flint.text.rich_text import parse_rich
from flint.text.sdf_bake import bake_glyph_msdf
from flint.text.ttf import TrueTypeFont
from flint.text.utf8 import decode_utf8
from flint.ui.context import UiContext


def _u16_le(b: List[UInt8], off: Int) -> UInt16:
    return UInt16(b[off]) | (UInt16(b[off + 1]) << 8)


def _u32_le(b: List[UInt8], off: Int) -> UInt32:
    return (
        UInt32(b[off])
        | (UInt32(b[off + 1]) << 8)
        | (UInt32(b[off + 2]) << 16)
        | (UInt32(b[off + 3]) << 24)
    )


def _u64_le(b: List[UInt8], off: Int) -> UInt64:
    var out = UInt64(0)
    for i in range(8):
        out |= UInt64(b[off + i]) << UInt64(i * 8)
    return out


def _put_u16_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _put_u32_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _build_minimal_otto_otf() -> List[UInt8]:
    """Builds a minimal valid CFF/OTF ('OTTO') SFNT with glyph 1 ('A') for MSDF checks."""
    var head = List[UInt8](length=54, fill=0)
    head[1] = 1
    head[12] = 0x5F; head[13] = 0x0F; head[14] = 0x3C; head[15] = 0xF5
    head[18] = 0x03; head[19] = 0xE8  # unitsPerEm = 1000
    head[40] = 0x03; head[41] = 0xE8
    head[42] = 0x03; head[43] = 0xE8

    var hhea = List[UInt8](length=36, fill=0)
    hhea[1] = 1
    hhea[4] = 0x03; hhea[5] = 0x20   # ascender = 800
    hhea[6] = 0xFF; hhea[7] = 0x38   # descender = -200
    hhea[34] = 0; hhea[35] = 2       # numberOfHMetrics = 2

    var maxp = List[UInt8](length=6, fill=0)
    maxp[1] = 0x50
    maxp[4] = 0; maxp[5] = 2         # numGlyphs = 2

    var hmtx = List[UInt8]()
    _put_u16_be(hmtx, 500); _put_u16_be(hmtx, 0)
    _put_u16_be(hmtx, 600); _put_u16_be(hmtx, 50)

    # cmap format 4 mapping 'A' (65) -> gid 1
    var cmap = List[UInt8]()
    _put_u16_be(cmap, 0); _put_u16_be(cmap, 1)
    _put_u16_be(cmap, 3); _put_u16_be(cmap, 1); _put_u32_be(cmap, 12)
    _put_u16_be(cmap, 4); _put_u16_be(cmap, 32); _put_u16_be(cmap, 0)
    _put_u16_be(cmap, 4); _put_u16_be(cmap, 4); _put_u16_be(cmap, 1); _put_u16_be(cmap, 0)
    _put_u16_be(cmap, 65); _put_u16_be(cmap, 0xFFFF); _put_u16_be(cmap, 0)
    _put_u16_be(cmap, 65); _put_u16_be(cmap, 0xFFFF)
    _put_u16_be(cmap, (-64) & 0xFFFF); _put_u16_be(cmap, 1)
    _put_u16_be(cmap, 0); _put_u16_be(cmap, 0)

    # CFF v1.0 with 2 Type2 charstrings (gid 0 empty, gid 1 square 100..500 x 0..700)
    var cs0 = List[UInt8]()
    cs0.append(14)  # endchar
    var cs1 = List[UInt8]()
    # 100 0 rmoveto, 400 hlineto, 700 vlineto, -400 hlineto, endchar
    cs1.append(247); cs1.append(0)
    cs1.append(139)
    cs1.append(21)  # rmoveto
    cs1.append(248); cs1.append(44)
    cs1.append(6)   # hlineto
    cs1.append(249); cs1.append(88)
    cs1.append(7)   # vlineto
    cs1.append(252); cs1.append(44)
    cs1.append(6)   # hlineto
    cs1.append(14)  # endchar

    var cs_index = List[UInt8]()
    _put_u16_be(cs_index, 2)
    cs_index.append(UInt8(1))
    cs_index.append(UInt8(1))
    cs_index.append(UInt8(1 + len(cs0)))
    cs_index.append(UInt8(1 + len(cs0) + len(cs1)))
    for b in cs0:
        cs_index.append(b)
    for b in cs1:
        cs_index.append(b)

    var hdr_len = 4
    var name_idx_len = 7
    var top_idx_len = 9
    var str_idx_len = 2
    var gsubr_idx_len = 2
    var cs_off = hdr_len + name_idx_len + top_idx_len + str_idx_len + gsubr_idx_len
    var dict_bytes = List[UInt8]()
    dict_bytes.append(29)
    _put_u32_be(dict_bytes, cs_off)
    dict_bytes.append(17)
    var top_idx_real_len = 5 + len(dict_bytes)
    cs_off = hdr_len + name_idx_len + top_idx_real_len + str_idx_len + gsubr_idx_len
    dict_bytes.clear()
    dict_bytes.append(29)
    _put_u32_be(dict_bytes, cs_off)
    dict_bytes.append(17)

    var cff = List[UInt8]()
    cff.append(1); cff.append(0); cff.append(4); cff.append(1)
    _put_u16_be(cff, 1); cff.append(1); cff.append(1); cff.append(3)
    cff.append(77); cff.append(65)
    _put_u16_be(cff, 1); cff.append(1); cff.append(1); cff.append(UInt8(1 + len(dict_bytes)))
    for b in dict_bytes:
        cff.append(b)
    _put_u16_be(cff, 0)
    _put_u16_be(cff, 0)
    for b in cs_index:
        cff.append(b)

    var tags: List[UInt32] = [0x43464620, 0x636D6170, 0x68656164, 0x68686561, 0x686D7478, 0x6D617870]
    var tbls: List[List[UInt8]] = [cff^, cmap^, head^, hhea^, hmtx^, maxp^]
    var n = len(tags)
    var out = List[UInt8]()
    _put_u32_be(out, 0x4F54544F)
    _put_u16_be(out, n); _put_u16_be(out, 64); _put_u16_be(out, 2); _put_u16_be(out, 32)
    var offset = 12 + n * 16
    var padded = List[List[UInt8]]()
    for i in range(n):
        var t = tbls[i].copy()
        var orig_len = len(t)
        while (len(t) & 3) != 0:
            t.append(0)
        _put_u32_be(out, Int(tags[i]))
        _put_u32_be(out, 0)
        _put_u32_be(out, offset)
        _put_u32_be(out, orig_len)
        offset += len(t)
        padded.append(t^)
    for i in range(n):
        out.extend(padded[i].copy())
    return out^


struct _GenericWindowApp[W: WindowBackend](Movable):
    """Verifies `WindowBackend` (including `Deinitable`) supports owned struct field storage."""
    var win: Self.W

    def __init__(out self, var win: Self.W):
        self.win = win^


struct _GenericAudioApp[A: AudioBackend](Movable):
    """Verifies `AudioBackend` (including `Deinitable`) supports owned struct field storage."""
    var aud: Self.A

    def __init__(out self, var aud: Self.A):
        self.aud = aud^


def _take_owned_window[W: WindowBackend](var w: W):
    var w2 = w^
    w2.shutdown()


def _take_owned_audio[A: AudioBackend](var a: A):
    var a2 = a^
    a2.shutdown()


def _generic_window_probe[W: WindowBackend](mut win: W, expect_init_ok: Bool = False) raises:
    """Exercises a generic `WindowBackend` polymorphically."""
    var ok = win.initialize(640, 360)
    if not expect_init_ok:
        assert_false(ok)
    assert_true(win.content_scale() >= Float32(1.0))
    var fb = win.framebuffer_size()
    assert_true(fb[0] >= 1 and fb[1] >= 1)
    win.clipboard_set("generic-backend-clip")
    assert_equal(win.clipboard_get(), "generic-backend-clip")
    var inp = InputState()
    win.poll_events(inp)
    var px = List[UInt32](length=4 * 4, fill=0xFF112233)
    win.present_argb(px, 4, 4)
    win.toggle_fullscreen()
    _ = win.should_close()
    win.shutdown()
    win.shutdown()


def _generic_audio_probe[A: AudioBackend](mut aud: A) raises:
    """Exercises a generic `AudioBackend` polymorphically."""
    assert_false(aud.initialize())
    aud.trigger(3, 1.25, 0.8)
    aud.set_ambient_drone(0.35)
    aud.pump(0.9)
    aud.shutdown()
    aud.shutdown()


def _run_scoped_multios_raii():
    var w32 = Win32Window()
    var w32_moved = w32^
    _ = w32_moved^
    var mac = CocoaWindow()
    var mac_moved = mac^
    _ = mac_moved^
    var wa = Win32Audio()
    var wa_moved = wa^
    _ = wa_moved^
    var ma = MacOsAudio()
    var ma_moved = ma^
    _ = ma_moved^
    var sock = UdpSocket.open_dual_stack(0)
    var sock_moved = sock^
    _ = sock_moved^


def test_f27_win32_and_macos_window_backends() raises:
    # 1. Trait conformance & non-Copyable RAII checks
    comptime assert conforms_to(Win32Window, WindowBackend)
    comptime assert conforms_to(CocoaWindow, WindowBackend)
    comptime assert conforms_to(WaylandWindow, WindowBackend)
    comptime assert conforms_to(X11Window, WindowBackend)
    comptime assert conforms_to(KmsWindow, WindowBackend)
    comptime assert not conforms_to(Win32Window, Copyable)
    comptime assert not conforms_to(CocoaWindow, Copyable)

    var w32 = Win32Window()
    assert_false(w32.initialize(800, 600))
    assert_true(w32.content_scale() >= Float32(1.0))
    assert_equal(w32.framebuffer_size()[0], 800)
    assert_equal(w32.framebuffer_size()[1], 600)
    w32.clipboard_set("win32-clip-¡Hola!")
    assert_equal(w32.clipboard_get(), "win32-clip-¡Hola!")
    var w32_m = w32^
    w32_m.shutdown()
    w32_m.shutdown()

    # 2. Win64 ABI binary layout checks
    var wc = win32_pack_wndclassexw(0x23, 0x1122334455667788, 0x1000, 0x2000, 0x3000)
    assert_equal(len(wc), WIN32_WNDCLASSEXW_BYTES)
    assert_equal(Int(_u32_le(wc, 0)), 80)
    assert_equal(Int(_u32_le(wc, 4)), 0x23)
    assert_equal(_u64_le(wc, 8), UInt64(0x1122334455667788))
    assert_equal(_u64_le(wc, 24), UInt64(0x1000))
    assert_equal(_u64_le(wc, 40), UInt64(0x2000))
    assert_equal(_u64_le(wc, 64), UInt64(0x3000))

    var bmi = win32_pack_bitmapinfoheader(1920, 1080)
    assert_equal(len(bmi), WIN32_BITMAPINFOHEADER_BYTES)
    assert_equal(Int(_u32_le(bmi, 0)), 40)
    assert_equal(Int(Int32(_u32_le(bmi, 4))), 1920)
    # Critical: negative biHeight for top-down ARGB DIB
    assert_equal(Int(Int32(_u32_le(bmi, 8))), -1080)
    assert_equal(Int(_u16_le(bmi, 12)), 1)
    assert_equal(Int(_u16_le(bmi, 14)), 32)

    var rect = win32_pack_rect(10, 20, 1290, 740)
    assert_equal(len(rect), WIN32_RECT_BYTES)
    var wmsg = win32_pack_msg(0xABCD, UInt32(WM_KEYDOWN), 0x41, 1)
    assert_equal(len(wmsg), WIN32_MSG_BYTES)
    assert_equal(_u64_le(wmsg, 0), UInt64(0xABCD))
    assert_equal(Int(_u32_le(wmsg, 8)), WM_KEYDOWN)

    # 3. UTF-8 <-> UTF-16LE codec with BMP, CJK & surrogate pairs + lone surrogate guard
    var sample = String("Flint ¡Hola! エンジン 🔥🎮")
    var u16 = utf8_to_utf16le(sample, True)
    assert_equal(Int(u16[len(u16) - 1]), 0)
    assert_equal(utf16le_to_utf8(u16), sample)
    # Verify 🔥 (U+1F525) encoded as D83D DD25
    var fire_u16 = utf8_to_utf16le("🔥", False)
    assert_equal(len(fire_u16), 2)
    assert_equal(Int(fire_u16[0]), 0xD83D)
    assert_equal(Int(fire_u16[1]), 0xDD25)
    var lone = List[UInt16]()
    lone.append(UInt16(0xD800))
    lone.append(UInt16(65))
    lone.append(UInt16(0))
    assert_equal(utf16le_to_utf8(lone), "\uFFFDA")

    # 4. Win32 VK & WM_* message dispatch (including surrogate WM_CHAR & signed Int16 coords)
    assert_equal(win32_vk_to_flint_key(0x41), 65)
    assert_equal(win32_vk_to_flint_key(0x1B), 256)
    assert_equal(win32_vk_to_flint_key(0x0D), 257)
    assert_equal(win32_vk_to_flint_key(0x25), 263)
    assert_equal(win32_vk_to_flint_key(0x70), 290)
    assert_equal(win32_resolve_vk(0x11, Int64(1 << 24)), 345)  # extended RightCtrl

    var w_ev = Win32Window()
    var inp = InputState()
    inp.begin_frame()
    w_ev.process_win32_message(WM_SIZE, 0, Int64((720 << 16) | 1280), inp)
    assert_equal(w_ev.framebuffer_size()[0], 1280)
    assert_equal(w_ev.framebuffer_size()[1], 720)
    w_ev.process_win32_message(WM_DPICHANGED, UInt64((144 << 16) | 144), 0, inp)
    assert_true(abs(w_ev.content_scale() - Float32(1.5)) < Float32(0.01))
    w_ev.process_win32_message(WM_KEYDOWN, 0x41, 0, inp)
    assert_true(inp.key_down(65))
    w_ev.process_win32_message(WM_KEYUP, 0x41, 0, inp)
    assert_false(inp.key_down(65))
    # Surrogate pair over 2 WM_CHAR messages -> 🔥
    w_ev.process_win32_message(WM_CHAR, 0xD83D, 0, inp)
    w_ev.process_win32_message(WM_CHAR, 0xDD25, 0, inp)
    w_ev.process_win32_message(WM_UNICHAR, UInt64(0x0041), 0, inp)
    assert_equal(inp.text_input_utf8, "🔥A")
    # Negative client mouse coords encoded as signed Int16 in lParam
    var neg_x = UInt64(UInt16(Int16(-25)))
    var neg_y = UInt64(UInt16(Int16(-10))) << 16
    w_ev.process_win32_message(WM_MOUSEMOVE, 0, Int64(neg_x | neg_y), inp)
    assert_true(abs(inp.last_mouse_x - Float32(-25.0)) < Float32(0.01))
    assert_true(abs(inp.last_mouse_y - Float32(-10.0)) < Float32(0.01))
    w_ev.process_win32_message(WM_LBUTTONDOWN, 0, 0, inp)
    w_ev.process_win32_message(WM_RBUTTONDOWN, 0, 0, inp)
    assert_true(inp.mouse_down(0) and inp.mouse_down(1))
    w_ev.process_win32_message(WM_LBUTTONUP, 0, 0, inp)
    assert_false(inp.mouse_down(0))
    w_ev.process_win32_message(WM_MOUSEWHEEL, UInt64(UInt16(Int16(240))) << 16, 0, inp)
    assert_true(abs(inp.mouse_wheel() - Float32(2.0)) < Float32(0.01))
    assert_false(w_ev.should_close())
    w_ev.process_win32_message(WM_CHAR, 0xD83D, 0, inp)  # buffer high surrogate
    w_ev.process_win32_message(WM_CLOSE, 0, 0, inp)
    assert_true(w_ev.should_close())
    # Regression: shutdown() + initialize() must reset close_requested and high_surrogate
    w_ev.shutdown()
    assert_false(w_ev.initialize(800, 600))
    assert_false(w_ev.should_close())
    var inp_reinit = InputState()
    inp_reinit.begin_frame()
    w_ev.process_win32_message(WM_CHAR, 0xDD25, 0, inp_reinit)
    assert_equal(inp_reinit.text_input_utf8, "")

    # 5. CocoaWindow ABI, keycode, AppKit Y-flip & NSEvent translation
    assert_equal(size_of[CocoaRect](), COCOA_RECT_BYTES)
    var packed_rect = cocoa_pack_rect(10.0, 20.0, 1440.0, 900.0)
    assert_equal(len(packed_rect), COCOA_RECT_BYTES)
    var unpacked_rect = cocoa_unpack_rect(packed_rect)
    assert_true(abs(unpacked_rect.width - 1440.0) < 1e-6)
    assert_true(abs(unpacked_rect.height - 900.0) < 1e-6)
    assert_true(abs(cocoa_flip_y(100.0, 720) - Float32(620.0)) < Float32(0.01))

    assert_equal(cocoa_keycode_to_flint_key(0x00), 65)   # A
    assert_equal(cocoa_keycode_to_flint_key(0x24), 257)  # Return
    assert_equal(cocoa_keycode_to_flint_key(0x35), 256)  # Escape
    assert_equal(cocoa_keycode_to_flint_key(0x7B), 263)  # Left
    assert_equal(cocoa_keycode_to_flint_key(0x7E), 265)  # Up

    var mac = CocoaWindow()
    assert_false(mac.initialize(1024, 768))
    assert_true(mac.content_scale() >= Float32(1.0))
    mac.clipboard_set("macos-clip")
    assert_equal(mac.clipboard_get(), "macos-clip")
    var minp = InputState()
    minp.begin_frame()
    mac.process_cocoa_event(NS_EVENT_MOUSE_MOVED, 0, 0, 320.0, 100.0, 4.0, -2.0, 0.0, "", minp)
    assert_true(abs(minp.last_mouse_x - Float32(320.0)) < Float32(0.01))
    assert_true(abs(minp.last_mouse_y - Float32(668.0)) < Float32(0.01))
    mac.process_cocoa_event(NS_EVENT_LEFT_MOUSE_DOWN, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "", minp)
    assert_true(minp.mouse_down(0))
    mac.process_cocoa_event(NS_EVENT_LEFT_MOUSE_UP, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "", minp)
    assert_false(minp.mouse_down(0))
    mac.process_cocoa_event(NS_EVENT_SCROLL_WHEEL, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.5, "", minp)
    assert_true(abs(minp.mouse_wheel() - Float32(1.5)) < Float32(0.01))
    mac.process_cocoa_event(NS_EVENT_KEY_DOWN, 0x00, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "a", minp)
    assert_true(minp.key_down(65))
    # AppKit private-use arrow char U+F702 must not leak into text_input_utf8 (single or mixed)
    assert_equal(_filter_cocoa_text_utf8("a\uF702b"), "ab")
    assert_equal(_filter_cocoa_text_utf8("a\uF702"), "a")
    assert_equal(_filter_cocoa_text_utf8("\uF702"), "")
    assert_equal(_filter_cocoa_text_utf8("\x08hi\x7F!"), "hi!")
    assert_equal(_filter_cocoa_text_utf8("🔥\uF703"), "🔥")
    mac.process_cocoa_event(NS_EVENT_KEY_DOWN, 0x7B, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "\uF702", minp)
    assert_equal(minp.text_input_utf8, "a")
    mac.process_cocoa_event(NS_EVENT_KEY_DOWN, 0x00, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "a\uF702b", minp)
    assert_equal(minp.text_input_utf8, "aab")
    mac.process_cocoa_event(NS_EVENT_KEY_UP, 0x00, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "", minp)
    assert_false(minp.key_down(65))
    mac.process_cocoa_event(
        NS_EVENT_FLAGS_CHANGED, 0, NS_MOD_SHIFT | NS_MOD_COMMAND, 0.0, 0.0, 0.0, 0.0, 0.0, "", minp
    )
    assert_true(minp.key_down(340) and minp.key_down(343))
    var mac_m = mac^
    mac_m.shutdown()
    mac_m.shutdown()

    # 6. Vulkan WSI null-handle safety
    var vk = VulkanGpu()
    assert_equal(vk.attach_win32_surface(UInt64(0), UInt64(0)).surface, 0)
    assert_equal(vk.attach_macos_surface(UInt64(0)).surface, 0)
    vk.close()


def test_f27_multios_udp_socket_and_wire_layouts() raises:
    comptime assert conforms_to(UdpSocket, NetTransport)
    comptime assert not conforms_to(UdpSocket, Copyable)

    assert_true(CompilationTarget.is_linux())
    assert_false(CompilationTarget.is_macos())
    assert_false(IS_WINDOWS)
    assert_equal(CURRENT_SOCK_OS, SOCK_OS_LINUX)
    assert_equal(Int(LINUX_AF_INET6), 10)
    assert_equal(Int(WIN32_AF_INET6), 23)
    assert_equal(Int(WIN32_WSAEWOULDBLOCK), 10035)
    assert_equal(Int(DARWIN_AF_INET6), 30)
    assert_equal(Int(DARWIN_EAGAIN), 35)
    assert_equal(Int(DARWIN_ENOBUFS), 55)
    assert_equal(Int(LINUX_EAGAIN), 11)

    # Wire layout round-trips across all 3 OSes
    var ep6 = NetEndpoint.ipv6_loopback(7777)
    var ep4 = NetEndpoint.ipv4(192, 168, 42, 7, 8888)
    var buf = List[UInt8](length=32, fill=0)
    var ptr = Int(buf.unsafe_ptr())

    # Linux
    ep6.write_sockaddr_in6_for_os(ptr, SOCK_OS_LINUX)
    assert_equal(Int(_u16_le(buf, 0)), 10)
    assert_equal((Int(buf[2]) << 8) | Int(buf[3]), 7777)
    assert_true(NetEndpoint.from_sockaddr_for_os(ptr, SOCK_OS_LINUX) == ep6)

    # Win32 (ws2_32.dll)
    ep6.write_sockaddr_in6_for_os(ptr, SOCK_OS_WINDOWS)
    assert_equal(Int(_u16_le(buf, 0)), 23)
    assert_equal((Int(buf[2]) << 8) | Int(buf[3]), 7777)
    assert_true(NetEndpoint.from_sockaddr_for_os(ptr, SOCK_OS_WINDOWS) == ep6)

    ep4.write_sockaddr_in6_for_os(ptr, SOCK_OS_WINDOWS)
    assert_true(NetEndpoint.from_sockaddr_for_os(ptr, SOCK_OS_WINDOWS) == ep4)

    # macOS (Darwin sin6_len = 28 at +0, sin6_family = 30 at +1)
    ep6.write_sockaddr_in6_for_os(ptr, SOCK_OS_MACOS)
    assert_equal(buf[0], DARWIN_SOCKADDR_IN6_LEN)
    assert_equal(Int(buf[1]), 30)
    assert_equal((Int(buf[2]) << 8) | Int(buf[3]), 7777)
    assert_true(NetEndpoint.from_sockaddr_for_os(ptr, SOCK_OS_MACOS) == ep6)

    # Darwin IPv4 sockaddr_in (sin_len=16, sin_family=2)
    for i in range(32):
        buf[i] = 0
    buf[0] = 16
    buf[1] = 2
    buf[2] = UInt8((9090 >> 8) & 0xFF)
    buf[3] = UInt8(9090 & 0xFF)
    buf[4] = 127; buf[5] = 0; buf[6] = 0; buf[7] = 1
    assert_true(NetEndpoint.from_sockaddr_for_os(ptr, SOCK_OS_MACOS) == NetEndpoint.ipv4_loopback(9090))
    _ = len(buf)

    # UdpSocket RAII move + is_valid() + live Linux loopback exchange
    var s_uninit = UdpSocket()
    assert_false(s_uninit.is_valid())
    assert_false(s_uninit.is_open())

    var s1 = UdpSocket.open_dual_stack(0)
    assert_true(s1.is_valid())
    var port1 = s1.local_port()
    assert_true(port1 > 0)
    var s1_moved = s1^
    assert_true(s1_moved.is_valid())
    assert_equal(s1_moved.local_port(), port1)

    var s2 = UdpSocket.open_dual_stack(0)
    assert_true(s2.is_valid())
    var dst = NetEndpoint.ipv4_loopback(s1_moved.local_port())
    var msg: List[UInt8] = [0x46, 0x4C, 0x4E, 0x54, 0x27]
    assert_true(s2.send_to(dst, msg))
    _ = len(msg)
    var views = List[DatagramView]()
    var n = s1_moved.recv_batch(views)
    if n > 0:
        var rx = views[0].to_bytes()
        assert_equal(len(rx), 5)
        assert_equal(rx[4], UInt8(0x27))
    assert_true(s1_moved.is_valid())


def test_f27_win32_and_macos_audio_backends() raises:
    comptime assert conforms_to(Win32Audio, AudioBackend)
    comptime assert conforms_to(MacOsAudio, AudioBackend)
    comptime assert not conforms_to(Win32Audio, Copyable)
    comptime assert not conforms_to(MacOsAudio, Copyable)

    # 1. Win32Audio ABI layouts + COM GUIDs + deterministic 4-slot PCM ring mixing
    var wfx = win32_pack_waveformatex(48000, 2, 16, WAVE_FORMAT_PCM)
    assert_equal(len(wfx), WIN32_WAVEFORMATEX_BYTES)
    assert_equal(_u16_le(wfx, 0), WAVE_FORMAT_PCM)
    assert_equal(Int(_u16_le(wfx, 2)), 2)
    assert_equal(Int(_u32_le(wfx, 4)), 48000)
    assert_equal(Int(_u32_le(wfx, 8)), 192000)
    assert_equal(Int(_u16_le(wfx, 12)), 4)
    assert_equal(Int(_u16_le(wfx, 14)), 16)

    var wfxe = win32_pack_waveformatextensible(48000, 2, 16, 3, False)
    assert_equal(len(wfxe), WIN32_WAVEFORMATEXTENSIBLE_BYTES)
    assert_equal(_u16_le(wfxe, 0), WAVE_FORMAT_EXTENSIBLE)
    assert_equal(Int(_u16_le(wfxe, 16)), 22)
    assert_equal(Int(_u32_le(wfxe, 20)), 3)

    var hdr = win32_pack_wavehdr(0x12345678, 4096, 2, 1)
    assert_equal(len(hdr), WIN32_WAVEHDR_BYTES)
    assert_equal(_u64_le(hdr, 0), UInt64(0x12345678))
    assert_equal(Int(_u32_le(hdr, 8)), 4096)

    assert_equal(len(win32_guid_clsid_mm_device_enumerator()), 16)
    assert_equal(len(win32_guid_iid_imm_device_enumerator()), 16)
    assert_equal(len(win32_guid_iid_iaudio_client()), 16)
    assert_equal(len(win32_guid_iid_iaudio_render_client()), 16)

    var wa = Win32Audio()
    assert_false(wa.initialize())
    wa.trigger(2, 1.0, 0.9)
    wa.set_ambient_drone(0.4)
    for _ in range(WIN32_AUDIO_RING_SLOTS + 1):
        wa.pump(1.0)
    assert_equal(wa.mixed_blocks, WIN32_AUDIO_RING_SLOTS + 1)
    var w_pcm = wa.last_mixed_slot()
    assert_equal(len(w_pcm), wa.period_frames * 2)
    var w_nonzero = False
    for s in w_pcm:
        assert_true(Int(s) >= -32767 and Int(s) <= 32767)
        if s != 0:
            w_nonzero = True
    assert_true(w_nonzero)
    var wa_moved = wa^
    wa_moved.shutdown()
    wa_moved.shutdown()

    # 2. MacOsAudio CoreAudio ASBD + AudioQueueBuffer + 4-slot PCM ring mixing
    var asbd = macos_pack_asbd(48000.0, 2, 16, False)
    assert_equal(len(asbd), MACOS_ASBD_BYTES)
    assert_true(abs(bitcast[DType.float64](_u64_le(asbd, 0)) - 48000.0) < 1e-6)
    assert_equal(_u32_le(asbd, 8), K_AUDIO_FORMAT_LINEAR_PCM)
    assert_equal(Int(_u32_le(asbd, 12)), 12)  # signed integer + packed
    assert_equal(Int(_u32_le(asbd, 16)), 4)   # mBytesPerPacket
    assert_equal(Int(_u32_le(asbd, 20)), 1)   # mFramesPerPacket
    assert_equal(Int(_u32_le(asbd, 24)), 4)   # mBytesPerFrame
    assert_equal(Int(_u32_le(asbd, 28)), 2)   # mChannelsPerFrame
    assert_equal(Int(_u32_le(asbd, 32)), 16)  # mBitsPerChannel

    var aqb = macos_pack_audio_queue_buffer(4096, 0xCAFEBABE, 2048, 1)
    assert_equal(len(aqb), MACOS_AQ_BUFFER_BYTES)
    assert_equal(Int(_u32_le(aqb, 0)), 4096)
    assert_equal(_u64_le(aqb, 8), UInt64(0xCAFEBABE))
    assert_equal(Int(_u32_le(aqb, 16)), 2048)

    var ma = MacOsAudio()
    assert_false(ma.initialize())
    ma.trigger(5, 1.5, 0.85)
    ma.set_ambient_drone(0.5)
    for _ in range(MACOS_AUDIO_RING_SLOTS + 1):
        ma.pump(1.0)
    assert_equal(ma.mixed_blocks, MACOS_AUDIO_RING_SLOTS + 1)
    var m_pcm = ma.last_mixed_slot()
    assert_equal(len(m_pcm), ma.period_frames * 2)
    var m_nonzero = False
    for s in m_pcm:
        assert_true(Int(s) >= -32767 and Int(s) <= 32767)
        if s != 0:
            m_nonzero = True
    assert_true(m_nonzero)
    var ma_moved = ma^
    ma_moved.shutdown()
    ma_moved.shutdown()


def test_integrated_f23_to_f28_pipeline_and_raii() raises:
    # 1. Generic polymorphic backend harnesses across all 5 window + 2 multi-OS audio backends
    var wl = WaylandWindow()
    assert_true(wl.supports_fractional_scale_protocol())
    assert_true(wl.supports_xdg_decoration_protocol())
    assert_true(wl.supports_data_device_clipboard())
    wl.clipboard_set("wl-generic")
    assert_equal(wl.clipboard_get(), "wl-generic")

    var x11 = X11Window()
    assert_true(x11.supports_mit_shm())
    assert_true(x11.supports_xi_raw_motion())
    _generic_window_probe(x11, True)

    var kms = KmsWindow()
    kms.clipboard_set("kms-clip")
    assert_equal(kms.clipboard_get(), "kms-clip")
    assert_true(kms.content_scale() >= Float32(1.0))

    var w32 = Win32Window()
    _generic_window_probe(w32)
    var mac = CocoaWindow()
    _generic_window_probe(mac)

    var wa = Win32Audio()
    _generic_audio_probe(wa)
    var ma = MacOsAudio()
    _generic_audio_probe(ma)

    # Generic Deinitable owned struct field storage & by-value transfer across WindowBackend & AudioBackend
    var app_w32 = _GenericWindowApp(Win32Window())
    assert_true(app_w32.win.content_scale() >= Float32(1.0))
    var app_mac = _GenericWindowApp(CocoaWindow())
    assert_true(app_mac.win.content_scale() >= Float32(1.0))
    var app_wl = _GenericWindowApp(WaylandWindow())
    assert_true(app_wl.win.content_scale() >= Float32(1.0))
    var app_x11 = _GenericWindowApp(X11Window())
    assert_true(app_x11.win.content_scale() >= Float32(1.0))
    var app_kms = _GenericWindowApp(KmsWindow())
    assert_true(app_kms.win.content_scale() >= Float32(1.0))
    _take_owned_window(Win32Window())
    _take_owned_window(CocoaWindow())
    _take_owned_window(WaylandWindow())
    _take_owned_window(X11Window())
    _take_owned_window(KmsWindow())

    var app_wa = _GenericAudioApp(Win32Audio())
    assert_false(app_wa.aud.initialize())
    var app_ma = _GenericAudioApp(MacOsAudio())
    assert_false(app_ma.aud.initialize())
    _take_owned_audio(Win32Audio())
    _take_owned_audio(MacOsAudio())

    # 2. Integrated F23+F24+F25+F26 pipeline:
    #    XKB/keysym -> InputState + EvdevHub -> CFF/OTF MSDF + BiDi/RTL -> UiContext text_input
    var xkb = XkbKeymap()
    assert_true(
        xkb.parse(
            'xkb_keymap {\nxkb_keycodes "evdev" { <AC01> = 38; };\nxkb_symbols "us" { key <AC01> { [ a, A ] }; };\n};'
        )
    )
    assert_equal(xkb.translate_key_to_utf8(30), "a")
    assert_equal(x11_keysym_to_codepoint(0x010005E9), 0x05E9)

    var hub = EvdevHub(enable_hotplug=False)
    var inp = InputState()
    inp.begin_frame()
    hub.poll_into(inp)
    assert_false(inp.uses_legacy_jsdev())
    inp.push_text_utf8("Hi שלום")

    var font = TrueTypeFont()
    var otto = _build_minimal_otto_otf()
    assert_true(font.load_bytes(otto^))
    assert_true(font.valid)
    var msdf = bake_glyph_msdf(font, 65, 24, 3)
    assert_true(msdf.channels >= 3)

    var bidi = resolve_bidi(decode_utf8("שלום world"), BIDI_AUTO)
    assert_equal(bidi.base_level, 1)
    assert_true(len(parse_rich("שלום")) >= 1)
    assert_true(len(wrap(SdfFont(), decode_utf8("שלום"), 16.0, 200.0)) >= 1)

    var ui = UiContext()
    var txt = String("")
    ui.begin_frame(inp, 800.0, 600.0)
    ui.focused_id = ui.widget_id("f23_28_box")
    assert_true(ui.text_input("f23_28_box", txt, 260.0))
    ui.end_frame()
    assert_equal(txt, "Hi שלום")
    assert_true(ui.draw_list.count() > 0)

    # 3. /proc/self/fd zero-FD-leak check across RAII moves/drops
    var fds_before = len(list_directory("/proc/self/fd"))
    _run_scoped_multios_raii()
    var fds_after = len(list_directory("/proc/self/fd"))
    assert_equal(fds_after, fds_before)


def main() raises:
    test_f27_win32_and_macos_window_backends()
    print("[PASS] (a) F27 Win32Window & CocoaWindow (WindowBackend, ABI layouts, UTF-16LE, WM_*/NSEvent, WSI)")
    test_f27_multios_udp_socket_and_wire_layouts()
    print("[PASS] (b) F27 Multi-OS UdpSocket / NetTransport (Linux, Win32 ws2_32, Darwin sockaddr_in6 + RAII)")
    test_f27_win32_and_macos_audio_backends()
    print("[PASS] (c) F27 Win32Audio & MacOsAudio (AudioBackend, WAVEFORMATEX/WAVEHDR, CoreAudio ASBD/AQ, PCM mix)")
    test_integrated_f23_to_f28_pipeline_and_raii()
    print("[PASS] (d) Integrated F23–F28 (All 5 WindowBackends, XKB, EvdevHub, OTF/MSDF/BiDi, UiContext, Zero-FD Leak)")
    print("ALL FLINT V5 MULTIOS & INTEGRATED F23-F28 TESTS PASSED")
