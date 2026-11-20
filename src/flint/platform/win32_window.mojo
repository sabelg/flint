"""Pure-Mojo Win32 window, input, clipboard, GDI present & Vulkan WSI backend (`flint.platform.win32_window`).

Zero C/C++ wrapper libraries: dynamically loads `user32.dll`, `gdi32.dll`, `kernel32.dll`,
and `shcore.dll` via `LoadLibraryW`/`GetProcAddress` when compiled for Windows, while
keeping pure-Mojo Win64 ABI struct packers, UTF-8 <-> UTF-16LE codecs, virtual-key
translators, and Win32 message dispatchers testable across all host platforms.
"""
from std.atomic import Atomic, Ordering
from std.collections import List
from std.ffi import external_call
from std.sys.info import CompilationTarget
from flint.platform.input import InputState
from flint.platform.traits import WindowBackend
from flint.platform.vulkan import VulkanGpu, SwapchainHandle
from flint.text.utf8 import REPLACEMENT, decode_utf8, encode_codepoint

comptime IS_WINDOWS: Bool = (not CompilationTarget.is_linux()) and (not CompilationTarget.is_macos())

# Win32 ABI struct sizes (64-bit Windows)
comptime WIN32_WNDCLASSEXW_BYTES: Int = 80
comptime WIN32_MSG_BYTES: Int = 48
comptime WIN32_RECT_BYTES: Int = 16
comptime WIN32_PAINTSTRUCT_BYTES: Int = 72
comptime WIN32_BITMAPINFOHEADER_BYTES: Int = 40

# Win32 window messages
comptime WM_DESTROY: Int = 0x0002
comptime WM_SIZE: Int = 0x0005
comptime WM_SETFOCUS: Int = 0x0007
comptime WM_KILLFOCUS: Int = 0x0008
comptime WM_PAINT: Int = 0x000F
comptime WM_CLOSE: Int = 0x0010
comptime WM_QUIT: Int = 0x0012
comptime WM_KEYDOWN: Int = 0x0100
comptime WM_KEYUP: Int = 0x0101
comptime WM_CHAR: Int = 0x0102
comptime WM_SYSKEYDOWN: Int = 0x0104
comptime WM_SYSKEYUP: Int = 0x0105
comptime WM_UNICHAR: Int = 0x0109
comptime WM_IME_COMPOSITION: Int = 0x010F
comptime WM_MOUSEMOVE: Int = 0x0200
comptime WM_LBUTTONDOWN: Int = 0x0201
comptime WM_LBUTTONUP: Int = 0x0202
comptime WM_RBUTTONDOWN: Int = 0x0204
comptime WM_RBUTTONUP: Int = 0x0205
comptime WM_MBUTTONDOWN: Int = 0x0207
comptime WM_MBUTTONUP: Int = 0x0208
comptime WM_MOUSEWHEEL: Int = 0x020A
comptime WM_DROPFILES: Int = 0x0233
comptime WM_DPICHANGED: Int = 0x02E0

# Window styles, GDI, Clipboard & DPI constants
comptime WS_OVERLAPPEDWINDOW: UInt32 = 0x00CF0000
comptime WS_VISIBLE: UInt32 = 0x10000000
comptime WS_POPUP: UInt32 = 0x80000000
comptime WS_CLIPSIBLINGS: UInt32 = 0x04000000
comptime WS_CLIPCHILDREN: UInt32 = 0x02000000
comptime CS_HREDRAW: UInt32 = 0x0002
comptime CS_VREDRAW: UInt32 = 0x0001
comptime CS_OWNDC: UInt32 = 0x0020
comptime CW_USEDEFAULT: Int32 = -2147483648
comptime PM_REMOVE: UInt32 = 0x0001
comptime SW_SHOW: Int32 = 5
comptime GWL_STYLE: Int32 = -16
comptime GWLP_USERDATA: Int32 = -21
comptime SWP_NOOWNERZORDER: UInt32 = 0x0200
comptime SWP_FRAMECHANGED: UInt32 = 0x0020
comptime CF_UNICODETEXT: UInt32 = 13
comptime GMEM_MOVEABLE: UInt32 = 0x0002
comptime BI_RGB: UInt32 = 0
comptime DIB_RGB_COLORS: UInt32 = 0
comptime SRCCOPY: UInt32 = 0x00CC0020
comptime DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: Int = -4
comptime UNICODE_NOCHAR: UInt64 = 0xFFFF
comptime IDC_ARROW: Int = 32512


@always_inline
def _w32_put16(mut buf: List[UInt8], off: Int, val: UInt16):
    buf[off] = UInt8(val & 0xFF)
    buf[off + 1] = UInt8((val >> 8) & 0xFF)


@always_inline
def _w32_put32(mut buf: List[UInt8], off: Int, val: UInt32):
    buf[off] = UInt8(val & 0xFF)
    buf[off + 1] = UInt8((val >> 8) & 0xFF)
    buf[off + 2] = UInt8((val >> 16) & 0xFF)
    buf[off + 3] = UInt8((val >> 24) & 0xFF)


@always_inline
def _w32_put64(mut buf: List[UInt8], off: Int, val: UInt64):
    for i in range(8):
        buf[off + i] = UInt8((val >> UInt64(i * 8)) & 0xFF)


@always_inline
def _w32_get32(buf: List[UInt8], off: Int) -> UInt32:
    return (
        UInt32(buf[off])
        | (UInt32(buf[off + 1]) << 8)
        | (UInt32(buf[off + 2]) << 16)
        | (UInt32(buf[off + 3]) << 24)
    )


@always_inline
def _w32_get16(buf: List[UInt8], off: Int) -> UInt16:
    return UInt16(buf[off]) | (UInt16(buf[off + 1]) << 8)


@always_inline
def _w32_get64(buf: List[UInt8], off: Int) -> UInt64:
    var out = UInt64(0)
    for i in range(8):
        out |= UInt64(buf[off + i]) << UInt64(i * 8)
    return out


def win32_pack_wndclassexw(
    style: UInt32,
    wndproc_addr: UInt64,
    hinstance: UInt64,
    hcursor: UInt64,
    class_name_ptr: UInt64,
    hicon: UInt64 = 0,
    hbr_background: UInt64 = 0,
) -> List[UInt8]:
    """Packs an 80-byte Win64 `WNDCLASSEXW` structure."""
    var buf = List[UInt8](length=WIN32_WNDCLASSEXW_BYTES, fill=0)
    _w32_put32(buf, 0, UInt32(WIN32_WNDCLASSEXW_BYTES))  # cbSize
    _w32_put32(buf, 4, style)                             # style
    _w32_put64(buf, 8, wndproc_addr)                      # lpfnWndProc
    _w32_put32(buf, 16, 0)                                # cbClsExtra
    _w32_put32(buf, 20, 0)                                # cbWndExtra
    _w32_put64(buf, 24, hinstance)                        # hInstance
    _w32_put64(buf, 32, hicon)                            # hIcon
    _w32_put64(buf, 40, hcursor)                          # hCursor
    _w32_put64(buf, 48, hbr_background)                   # hbrBackground
    _w32_put64(buf, 56, 0)                                # lpszMenuName
    _w32_put64(buf, 64, class_name_ptr)                   # lpszClassName
    _w32_put64(buf, 72, hicon)                            # hIconSm
    return buf^


def win32_pack_bitmapinfoheader(width: Int, height: Int) -> List[UInt8]:
    """Packs a 40-byte top-down 32-bpp ARGB `BITMAPINFOHEADER` (`biHeight = -height`)."""
    var w = max(1, width)
    var h = max(1, height)
    var buf = List[UInt8](length=WIN32_BITMAPINFOHEADER_BYTES, fill=0)
    _w32_put32(buf, 0, UInt32(WIN32_BITMAPINFOHEADER_BYTES))  # biSize
    _w32_put32(buf, 4, UInt32(Int32(w)))                      # biWidth
    _w32_put32(buf, 8, UInt32(Int32(-h)))                     # biHeight (negative = top-down)
    _w32_put16(buf, 12, 1)                                    # biPlanes
    _w32_put16(buf, 14, 32)                                   # biBitCount (ARGB 32-bit)
    _w32_put32(buf, 16, BI_RGB)                               # biCompression
    _w32_put32(buf, 20, UInt32(w * h * 4))                    # biSizeImage
    return buf^


def win32_pack_rect(left: Int32, top: Int32, right: Int32, bottom: Int32) -> List[UInt8]:
    """Packs a 16-byte Win32 `RECT` structure."""
    var buf = List[UInt8](length=WIN32_RECT_BYTES, fill=0)
    _w32_put32(buf, 0, UInt32(left))
    _w32_put32(buf, 4, UInt32(top))
    _w32_put32(buf, 8, UInt32(right))
    _w32_put32(buf, 12, UInt32(bottom))
    return buf^


def win32_pack_msg(hwnd: UInt64, message: UInt32, wparam: UInt64, lparam: Int64) -> List[UInt8]:
    """Packs a 48-byte Win64 `MSG` structure."""
    var buf = List[UInt8](length=WIN32_MSG_BYTES, fill=0)
    _w32_put64(buf, 0, hwnd)
    _w32_put32(buf, 8, message)
    _w32_put64(buf, 16, wparam)
    _w32_put64(buf, 24, UInt64(lparam))
    return buf^


def utf8_to_utf16le(s: String, null_terminate: Bool = True) -> List[UInt16]:
    """Encodes UTF-8 `String` into UTF-16LE code units with full supplementary surrogate pairs."""
    var codes = decode_utf8(s)
    var out = List[UInt16](capacity=len(codes) + 1)
    for i in range(len(codes)):
        var cp = codes[i]
        if cp < 0 or cp > 0x10FFFF or (cp >= 0xD800 and cp <= 0xDFFF):
            cp = REPLACEMENT
        if cp < 0x10000:
            out.append(UInt16(cp))
        else:
            var u = cp - 0x10000
            out.append(UInt16(0xD800 + (u >> 10)))
            out.append(UInt16(0xDC00 + (u & 0x3FF)))
    if null_terminate:
        out.append(UInt16(0))
    return out^


def utf16le_to_utf8(units: List[UInt16]) -> String:
    """Decodes UTF-16LE code units (up to NUL if present) into strict UTF-8 `String`."""
    var bytes = List[UInt8](capacity=len(units) * 2)
    var i = 0
    var n = len(units)
    while i < n:
        var u0 = Int(units[i])
        if u0 == 0:
            break
        if u0 >= 0xD800 and u0 <= 0xDBFF:
            if i + 1 < n:
                var u1 = Int(units[i + 1])
                if u1 >= 0xDC00 and u1 <= 0xDFFF:
                    var cp = 0x10000 + ((u0 - 0xD800) << 10) + (u1 - 0xDC00)
                    encode_codepoint(bytes, cp)
                    i += 2
                    continue
            encode_codepoint(bytes, REPLACEMENT)
            i += 1
        elif u0 >= 0xDC00 and u0 <= 0xDFFF:
            encode_codepoint(bytes, REPLACEMENT)
            i += 1
        else:
            encode_codepoint(bytes, u0)
            i += 1
    return String(unsafe_from_utf8=bytes) if len(bytes) > 0 else String()


def utf16le_ptr_to_utf8(ptr: Int, max_units: Int = 65536) -> String:
    """Reads a NUL-terminated UTF-16LE buffer from `ptr` and converts it to UTF-8."""
    if ptr == 0 or max_units <= 0:
        return String()
    var units = List[UInt16]()
    var p = Pointer[UInt16, ImmutAnyOrigin](unsafe_from_address=ptr)
    for i in range(max_units):
        var u = p[unsafe_offset=i]
        if u == 0:
            break
        units.append(u)
    return utf16le_to_utf8(units)


def win32_vk_to_flint_key(vk: Int) -> Int:
    """Maps Win32 Virtual-Key (`VK_*`) codes to Flint `InputState` physical key slots."""
    if vk >= 0x41 and vk <= 0x5A:
        return vk  # 'A'..'Z' == 65..90
    if vk >= 0x30 and vk <= 0x39:
        return vk  # '0'..'9' == 48..57
    if vk == 0x20:
        return 32   # VK_SPACE
    if vk == 0x1B:
        return 256  # VK_ESCAPE
    if vk == 0x0D:
        return 257  # VK_RETURN
    if vk == 0x09:
        return 258  # VK_TAB
    if vk == 0x08:
        return 259  # VK_BACK
    if vk == 0x2D:
        return 260  # VK_INSERT
    if vk == 0x2E:
        return 261  # VK_DELETE
    if vk == 0x27:
        return 262  # VK_RIGHT
    if vk == 0x25:
        return 263  # VK_LEFT
    if vk == 0x28:
        return 264  # VK_DOWN
    if vk == 0x26:
        return 265  # VK_UP
    if vk == 0x21:
        return 266  # VK_PRIOR (PageUp)
    if vk == 0x22:
        return 267  # VK_NEXT (PageDown)
    if vk == 0x24:
        return 268  # VK_HOME
    if vk == 0x23:
        return 269  # VK_END
    if vk == 0x14:
        return 280  # VK_CAPITAL (CapsLock)
    if vk >= 0x70 and vk <= 0x7B:
        return 290 + (vk - 0x70)  # VK_F1..VK_F12
    if vk == 0xA0 or vk == 0x10:
        return 340  # VK_LSHIFT / VK_SHIFT
    if vk == 0xA2 or vk == 0x11:
        return 341  # VK_LCONTROL / VK_CONTROL
    if vk == 0xA4 or vk == 0x12:
        return 342  # VK_LMENU / VK_MENU
    if vk == 0x5B:
        return 343  # VK_LWIN
    if vk == 0xA1:
        return 344  # VK_RSHIFT
    if vk == 0xA3:
        return 345  # VK_RCONTROL
    if vk == 0xA5:
        return 346  # VK_RMENU
    if vk == 0x5C:
        return 347  # VK_RWIN
    if vk == 0xBA:
        return 59   # VK_OEM_1 (;)
    if vk == 0xBB:
        return 61   # VK_OEM_PLUS (=)
    if vk == 0xBC:
        return 44   # VK_OEM_COMMA (,)
    if vk == 0xBD:
        return 45   # VK_OEM_MINUS (-)
    if vk == 0xBE:
        return 46   # VK_OEM_PERIOD (.)
    if vk == 0xBF:
        return 47   # VK_OEM_2 (/)
    if vk == 0xC0:
        return 96   # VK_OEM_3 (`)
    if vk == 0xDB:
        return 91   # VK_OEM_4 ([)
    if vk == 0xDC:
        return 92   # VK_OEM_5 (\)
    if vk == 0xDD:
        return 93   # VK_OEM_6 (])
    if vk == 0xDE:
        return 39   # VK_OEM_7 (')
    return -1


def win32_resolve_vk(vk: Int, lparam: Int64) -> Int:
    """Resolves generic Win32 `VK_SHIFT`/`VK_CONTROL`/`VK_MENU` using extended-key & scan bits."""
    var extended = (Int(lparam) & (1 << 24)) != 0
    var scancode = (Int(lparam) >> 16) & 0xFF
    if vk == 0x10:
        return 344 if scancode == 0x36 else 340
    if vk == 0x11:
        return 345 if extended else 341
    if vk == 0x12:
        return 346 if extended else 342
    return win32_vk_to_flint_key(vk)


def _win32_load_dll(name: String) -> Int:
    comptime if IS_WINDOWS:
        var wide = utf8_to_utf16le(name, True)
        var h = external_call["LoadLibraryW", Int](wide.unsafe_ptr())
        _ = len(wide)
        return h
    else:
        return 0


def _win32_get_proc(lib: Int, sym: String) -> Int:
    comptime if IS_WINDOWS:
        if lib == 0:
            return 0
        var owned = sym
        var addr = external_call["GetProcAddress", Int](lib, owned.as_c_string_span().ptr())
        _ = owned.byte_length()
        return addr
    else:
        return 0


def _win32_free_dll(lib: Int):
    comptime if IS_WINDOWS:
        if lib != 0:
            _ = external_call["FreeLibrary", Int32](lib)


def _win32_get_module_handle(name: String) -> Int:
    comptime if IS_WINDOWS:
        var wide = utf8_to_utf16le(name, True)
        var h = external_call["GetModuleHandleW", Int](wide.unsafe_ptr())
        _ = len(wide)
        return h
    else:
        return 0


def _win32_wndproc_trampoline(
    hwnd: Int, msg: UInt32, wparam: UInt64, lparam: Int64
) abi("C") -> Int64:
    """Win32 `WNDPROC` callback: records `WM_CLOSE`/`WM_DESTROY` via `GWLP_USERDATA` without leaking DLL refs."""
    comptime if IS_WINDOWS:
        var h_user32 = _win32_get_module_handle("user32.dll")
        if h_user32 == 0:
            return 0
        if msg == UInt32(WM_CLOSE) or msg == UInt32(WM_DESTROY):
            var state_addr = 0
            var fn_get_long = _win32_get_proc(h_user32, "GetWindowLongPtrW")
            if fn_get_long != 0:
                comptime GetLongFn = def(Int, Int32) thin abi("C") -> Int64
                state_addr = Int(
                    Pointer(to=fn_get_long).unsafe_bitcast[GetLongFn]()[unsafe_offset=0](
                        hwnd, GWLP_USERDATA
                    )
                )
            if state_addr != 0:
                Pointer[Atomic[Int32], origin=MutAnyOrigin](
                    unsafe_from_address=state_addr
                )[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int32(1))
                if msg == UInt32(WM_DESTROY):
                    var fn_post_quit = _win32_get_proc(h_user32, "PostQuitMessage")
                    if fn_post_quit != 0:
                        comptime PostQuitFn = def(Int32) thin abi("C") -> None
                        Pointer(to=fn_post_quit).unsafe_bitcast[PostQuitFn]()[unsafe_offset=0](0)
            return 0
        var def_proc = _win32_get_proc(h_user32, "DefWindowProcW")
        if def_proc != 0:
            comptime DefFn = def(Int, UInt32, UInt64, Int64) thin abi("C") -> Int64
            return Pointer(to=def_proc).unsafe_bitcast[DefFn]()[unsafe_offset=0](
                hwnd, msg, wparam, lparam
            )
    return 0


struct Win32Window(WindowBackend):
    """Pure-Mojo Win32 `WindowBackend` (`user32.dll` + `gdi32.dll` + `VK_KHR_win32_surface`)."""
    var hwnd: UInt64
    var hinstance: UInt64
    var hdc: UInt64
    var width: Int
    var height: Int
    var dpi: Int
    var close_requested: Bool
    var focused: Bool
    var fullscreen: Bool
    var pointer_locked: Bool
    var window_title: String
    var clipboard_text: String
    var clipboard_owned: Bool
    var high_surrogate: UInt16
    var saved_style: UInt32
    var saved_rect: List[Int32]
    var gpu_swapchain: UInt64
    var user32_lib: Int
    var gdi32_lib: Int
    var kernel32_lib: Int
    var shcore_lib: Int
    var pending_text_utf8: String
    var pending_dropped_files: List[String]
    var dib_header: List[UInt8]
    var wndproc_state_ptr: Int
    var active_flag: Bool

    def __init__(out self):
        self.hwnd = 0
        self.hinstance = 0
        self.hdc = 0
        self.width = 1280
        self.height = 720
        self.dpi = 96
        self.close_requested = False
        self.focused = True
        self.fullscreen = False
        self.pointer_locked = False
        self.window_title = "FLINT ENGINE"
        self.clipboard_text = ""
        self.clipboard_owned = False
        self.high_surrogate = 0
        self.saved_style = WS_OVERLAPPEDWINDOW
        self.saved_rect = List[Int32](length=4, fill=0)
        self.gpu_swapchain = 0
        self.user32_lib = 0
        self.gdi32_lib = 0
        self.kernel32_lib = 0
        self.shcore_lib = 0
        self.pending_text_utf8 = ""
        self.pending_dropped_files = List[String]()
        self.dib_header = win32_pack_bitmapinfoheader(1280, 720)
        self.wndproc_state_ptr = 0
        self.active_flag = False

    def __moveinit__(out self, deinit existing: Self):
        self.hwnd = existing.hwnd
        self.hinstance = existing.hinstance
        self.hdc = existing.hdc
        self.width = existing.width
        self.height = existing.height
        self.dpi = existing.dpi
        self.close_requested = existing.close_requested
        self.focused = existing.focused
        self.fullscreen = existing.fullscreen
        self.pointer_locked = existing.pointer_locked
        self.window_title = existing.window_title^
        self.clipboard_text = existing.clipboard_text^
        self.clipboard_owned = existing.clipboard_owned
        self.high_surrogate = existing.high_surrogate
        self.saved_style = existing.saved_style
        self.saved_rect = existing.saved_rect^
        self.gpu_swapchain = existing.gpu_swapchain
        self.user32_lib = existing.user32_lib
        self.gdi32_lib = existing.gdi32_lib
        self.kernel32_lib = existing.kernel32_lib
        self.shcore_lib = existing.shcore_lib
        self.pending_text_utf8 = existing.pending_text_utf8^
        self.pending_dropped_files = existing.pending_dropped_files^
        self.dib_header = existing.dib_header^
        self.wndproc_state_ptr = existing.wndproc_state_ptr
        self.active_flag = existing.active_flag
        existing.hwnd = 0
        existing.hinstance = 0
        existing.hdc = 0
        existing.user32_lib = 0
        existing.gdi32_lib = 0
        existing.kernel32_lib = 0
        existing.shcore_lib = 0
        existing.gpu_swapchain = 0
        existing.wndproc_state_ptr = 0
        existing.active_flag = False

    def __deinit__(deinit self):
        self.shutdown()

    def active(self) -> Bool:
        return self.active_flag and self.hwnd != 0

    def initialize(mut self, width: Int, height: Int) -> Bool:
        var title = self.window_title
        return self.initialize(width, height, title)

    def initialize(mut self, width: Int, height: Int, title: String) -> Bool:
        self.shutdown()
        self.width = max(1, width)
        self.height = max(1, height)
        self.window_title = title
        self.dib_header = win32_pack_bitmapinfoheader(self.width, self.height)
        comptime if IS_WINDOWS:
            self.kernel32_lib = _win32_load_dll("kernel32.dll")
            self.user32_lib = _win32_load_dll("user32.dll")
            self.gdi32_lib = _win32_load_dll("gdi32.dll")
            self.shcore_lib = _win32_load_dll("shcore.dll")
            if self.user32_lib == 0 or self.gdi32_lib == 0 or self.kernel32_lib == 0:
                self.shutdown()
                return False

            var fn_dpi_ctx = _win32_get_proc(self.user32_lib, "SetProcessDpiAwarenessContext")
            if fn_dpi_ctx != 0:
                comptime SetDpiCtxFn = def(Int) thin abi("C") -> Int32
                _ = Pointer(to=fn_dpi_ctx).unsafe_bitcast[SetDpiCtxFn]()[unsafe_offset=0](
                    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                )

            var fn_mod = _win32_get_proc(self.kernel32_lib, "GetModuleHandleW")
            if fn_mod != 0:
                comptime GetModFn = def(Int) thin abi("C") -> UInt64
                self.hinstance = Pointer(to=fn_mod).unsafe_bitcast[GetModFn]()[unsafe_offset=0](0)

            var class_utf16 = utf8_to_utf16le("FlintEngineWin32Window", True)
            var title_utf16 = utf8_to_utf16le(title, True)

            var fn_load_cursor = _win32_get_proc(self.user32_lib, "LoadCursorW")
            var hcursor = UInt64(0)
            if fn_load_cursor != 0:
                comptime LoadCursorFn = def(UInt64, Int) thin abi("C") -> UInt64
                hcursor = Pointer(to=fn_load_cursor).unsafe_bitcast[LoadCursorFn]()[unsafe_offset=0](
                    0, IDC_ARROW
                )

            comptime WndProcFn = def(Int, UInt32, UInt64, Int64) thin abi("C") -> Int64
            var cb: WndProcFn = _win32_wndproc_trampoline
            var cb_addr = UInt64(Pointer(to=cb).unsafe_bitcast[Int]()[unsafe_offset=0])
            var wc = win32_pack_wndclassexw(
                CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
                cb_addr,
                self.hinstance,
                hcursor,
                UInt64(Int(class_utf16.unsafe_ptr())),
            )
            var fn_reg = _win32_get_proc(self.user32_lib, "RegisterClassExW")
            if fn_reg != 0:
                comptime RegFn = def(Int) thin abi("C") -> UInt16
                _ = Pointer(to=fn_reg).unsafe_bitcast[RegFn]()[unsafe_offset=0](Int(wc.unsafe_ptr()))
            _ = len(wc)

            var rect = win32_pack_rect(0, 0, Int32(self.width), Int32(self.height))
            var fn_adj = _win32_get_proc(self.user32_lib, "AdjustWindowRectEx")
            var style = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN
            if fn_adj != 0:
                comptime AdjFn = def(Int, UInt32, Int32, UInt32) thin abi("C") -> Int32
                _ = Pointer(to=fn_adj).unsafe_bitcast[AdjFn]()[unsafe_offset=0](
                    Int(rect.unsafe_ptr()), style, 0, 0
                )
            var win_w = Int32(_w32_get32(rect, 8) - _w32_get32(rect, 0))
            var win_h = Int32(_w32_get32(rect, 12) - _w32_get32(rect, 4))

            var fn_create = _win32_get_proc(self.user32_lib, "CreateWindowExW")
            if fn_create == 0:
                self.shutdown()
                return False
            comptime CreateWinFn = def(
                UInt32, Int, Int, UInt32, Int32, Int32, Int32, Int32, UInt64, UInt64, UInt64, Int
            ) thin abi("C") -> UInt64
            self.hwnd = Pointer(to=fn_create).unsafe_bitcast[CreateWinFn]()[unsafe_offset=0](
                0,
                Int(class_utf16.unsafe_ptr()),
                Int(title_utf16.unsafe_ptr()),
                style | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                win_w,
                win_h,
                0,
                0,
                self.hinstance,
                0,
            )
            _ = len(class_utf16)
            _ = len(title_utf16)
            if self.hwnd == 0:
                self.shutdown()
                return False

            self.wndproc_state_ptr = external_call["calloc", Int](Int(1), Int(8))
            if self.wndproc_state_ptr != 0:
                var fn_set_long = _win32_get_proc(self.user32_lib, "SetWindowLongPtrW")
                if fn_set_long != 0:
                    comptime SetLongPtrFn = def(UInt64, Int32, Int64) thin abi("C") -> Int64
                    _ = Pointer(to=fn_set_long).unsafe_bitcast[SetLongPtrFn]()[unsafe_offset=0](
                        self.hwnd, GWLP_USERDATA, Int64(self.wndproc_state_ptr)
                    )

            var fn_get_dc = _win32_get_proc(self.user32_lib, "GetDC")
            if fn_get_dc != 0:
                comptime GetDcFn = def(UInt64) thin abi("C") -> UInt64
                self.hdc = Pointer(to=fn_get_dc).unsafe_bitcast[GetDcFn]()[unsafe_offset=0](self.hwnd)

            var fn_dpi_win = _win32_get_proc(self.user32_lib, "GetDpiForWindow")
            if fn_dpi_win != 0:
                comptime DpiWinFn = def(UInt64) thin abi("C") -> UInt32
                var d = Int(Pointer(to=fn_dpi_win).unsafe_bitcast[DpiWinFn]()[unsafe_offset=0](self.hwnd))
                if d >= 96:
                    self.dpi = d

            var fn_show = _win32_get_proc(self.user32_lib, "ShowWindow")
            if fn_show != 0:
                comptime ShowFn = def(UInt64, Int32) thin abi("C") -> Int32
                _ = Pointer(to=fn_show).unsafe_bitcast[ShowFn]()[unsafe_offset=0](self.hwnd, SW_SHOW)

            self.active_flag = True
            return True
        else:
            return False

    def process_win32_message(
        mut self, msg: Int, wparam: UInt64, lparam: Int64, mut input: InputState
    ):
        """Translates a Win32 message (`WM_*`) into `Win32Window` and `InputState` updates."""
        if msg == WM_CLOSE or msg == WM_DESTROY or msg == WM_QUIT:
            self.close_requested = True
            return
        if msg == WM_SIZE:
            var w = Int(UInt64(lparam) & 0xFFFF)
            var h = Int((UInt64(lparam) >> 16) & 0xFFFF)
            self.width = max(1, w)
            self.height = max(1, h)
            self.dib_header = win32_pack_bitmapinfoheader(self.width, self.height)
            return
        if msg == WM_DPICHANGED:
            var new_dpi = Int(wparam & 0xFFFF)
            if new_dpi > 0:
                self.dpi = max(96, new_dpi)
            return
        if msg == WM_SETFOCUS:
            self.focused = True
            return
        if msg == WM_KILLFOCUS:
            self.focused = False
            self.high_surrogate = 0
            return
        if msg == WM_KEYDOWN or msg == WM_SYSKEYDOWN:
            var fk = win32_resolve_vk(Int(wparam & 0xFFFF), lparam)
            if fk >= 0:
                input.set_key(fk, True)
            return
        if msg == WM_KEYUP or msg == WM_SYSKEYUP:
            var fk = win32_resolve_vk(Int(wparam & 0xFFFF), lparam)
            if fk >= 0:
                input.set_key(fk, False)
            return
        if msg == WM_CHAR:
            var unit = Int(wparam & 0xFFFF)
            if unit >= 0xD800 and unit <= 0xDBFF:
                self.high_surrogate = UInt16(unit)
                return
            if unit >= 0xDC00 and unit <= 0xDFFF:
                if self.high_surrogate != 0:
                    var hi = Int(self.high_surrogate)
                    self.high_surrogate = 0
                    var cp = 0x10000 + ((hi - 0xD800) << 10) + (unit - 0xDC00)
                    var utf8 = List[UInt8](capacity=4)
                    encode_codepoint(utf8, cp)
                    input.push_text_utf8(String(unsafe_from_utf8=utf8))
                return
            self.high_surrogate = 0
            if unit >= 0x20 and unit != 0x7F:
                var utf8 = List[UInt8](capacity=4)
                encode_codepoint(utf8, unit)
                input.push_text_utf8(String(unsafe_from_utf8=utf8))
            return
        if msg == WM_UNICHAR:
            if wparam == UNICODE_NOCHAR:
                return
            var cp = Int(wparam & 0x1FFFFF)
            if cp >= 0x20 and cp != 0x7F:
                var utf8 = List[UInt8](capacity=4)
                encode_codepoint(utf8, cp)
                input.push_text_utf8(String(unsafe_from_utf8=utf8))
            return
        if msg == WM_MOUSEMOVE:
            # GET_X_LPARAM / GET_Y_LPARAM: signed 16-bit client coordinates
            var raw_u = UInt64(lparam)
            var sx = Int(Int16(UInt16(raw_u & 0xFFFF)))
            var sy = Int(Int16(UInt16((raw_u >> 16) & 0xFFFF)))
            input.move_mouse(Float32(sx), Float32(sy))
            return
        if msg == WM_LBUTTONDOWN:
            input.set_mouse(0, True)
            return
        if msg == WM_LBUTTONUP:
            input.set_mouse(0, False)
            return
        if msg == WM_RBUTTONDOWN:
            input.set_mouse(1, True)
            return
        if msg == WM_RBUTTONUP:
            input.set_mouse(1, False)
            return
        if msg == WM_MBUTTONDOWN:
            input.set_mouse(2, True)
            return
        if msg == WM_MBUTTONUP:
            input.set_mouse(2, False)
            return
        if msg == WM_MOUSEWHEEL:
            var delta = Int(Int16(UInt16((wparam >> 16) & 0xFFFF)))
            input.wheel += Float32(delta) / Float32(120.0)
            return

    def poll_events(mut self, mut input: InputState):
        comptime if IS_WINDOWS:
            if not self.active() or self.user32_lib == 0:
                return
            if self.wndproc_state_ptr != 0:
                var flag = Pointer[Atomic[Int32], origin=MutAnyOrigin](
                    unsafe_from_address=self.wndproc_state_ptr
                )[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
                if flag != 0:
                    self.close_requested = True
            var fn_peek = _win32_get_proc(self.user32_lib, "PeekMessageW")
            var fn_trans = _win32_get_proc(self.user32_lib, "TranslateMessage")
            var fn_disp = _win32_get_proc(self.user32_lib, "DispatchMessageW")
            if fn_peek == 0:
                return
            comptime PeekFn = def(Int, UInt64, UInt32, UInt32, UInt32) thin abi("C") -> Int32
            comptime TransFn = def(Int) thin abi("C") -> Int32
            comptime DispFn = def(Int) thin abi("C") -> Int64
            var msg_buf = List[UInt8](length=WIN32_MSG_BYTES, fill=0)
            var msg_ptr = Int(msg_buf.unsafe_ptr())
            for _ in range(256):
                var has_msg = Pointer(to=fn_peek).unsafe_bitcast[PeekFn]()[unsafe_offset=0](
                    msg_ptr, UInt64(0), 0, 0, PM_REMOVE
                )
                if has_msg == 0:
                    break
                var m_id = Int(_w32_get32(msg_buf, 8))
                var wp = _w32_get64(msg_buf, 16)
                var lp = Int64(_w32_get64(msg_buf, 24))
                self.process_win32_message(m_id, wp, lp, input)
                if fn_trans != 0:
                    _ = Pointer(to=fn_trans).unsafe_bitcast[TransFn]()[unsafe_offset=0](msg_ptr)
                if fn_disp != 0:
                    _ = Pointer(to=fn_disp).unsafe_bitcast[DispFn]()[unsafe_offset=0](msg_ptr)
            _ = len(msg_buf)
            if self.wndproc_state_ptr != 0:
                var flag = Pointer[Atomic[Int32], origin=MutAnyOrigin](
                    unsafe_from_address=self.wndproc_state_ptr
                )[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
                if flag != 0:
                    self.close_requested = True
        else:
            _ = input.uses_legacy_jsdev()

    def present_argb(mut self, pixels: List[UInt32], width: Int, height: Int):
        comptime if IS_WINDOWS:
            if (
                not self.active()
                or self.hdc == 0
                or self.gdi32_lib == 0
                or width <= 0
                or height <= 0
                or len(pixels) < width * height
            ):
                return
            var fn_stretch = _win32_get_proc(self.gdi32_lib, "StretchDIBits")
            if fn_stretch == 0:
                return
            var bmi = win32_pack_bitmapinfoheader(width, height)
            comptime StretchFn = def(
                UInt64, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int, Int, UInt32, UInt32
            ) thin abi("C") -> Int32
            _ = Pointer(to=fn_stretch).unsafe_bitcast[StretchFn]()[unsafe_offset=0](
                self.hdc,
                0,
                0,
                Int32(self.width),
                Int32(self.height),
                0,
                0,
                Int32(width),
                Int32(height),
                Int(pixels.unsafe_ptr()),
                Int(bmi.unsafe_ptr()),
                DIB_RGB_COLORS,
                SRCCOPY,
            )
            _ = len(bmi)
        else:
            pass

    def toggle_fullscreen(mut self):
        self.fullscreen = not self.fullscreen
        comptime if IS_WINDOWS:
            if not self.active() or self.user32_lib == 0:
                return
            var fn_get_long = _win32_get_proc(self.user32_lib, "GetWindowLongPtrW")
            var fn_set_long = _win32_get_proc(self.user32_lib, "SetWindowLongPtrW")
            var fn_set_pos = _win32_get_proc(self.user32_lib, "SetWindowPos")
            if fn_get_long == 0 or fn_set_long == 0 or fn_set_pos == 0:
                return
            comptime GetLongFn = def(UInt64, Int32) thin abi("C") -> Int64
            comptime SetLongFn = def(UInt64, Int32, Int64) thin abi("C") -> Int64
            comptime SetPosFn = def(UInt64, UInt64, Int32, Int32, Int32, Int32, UInt32) thin abi("C") -> Int32
            var style = UInt32(Pointer(to=fn_get_long).unsafe_bitcast[GetLongFn]()[unsafe_offset=0](self.hwnd, GWL_STYLE))
            if self.fullscreen:
                self.saved_style = style
                _ = Pointer(to=fn_set_long).unsafe_bitcast[SetLongFn]()[unsafe_offset=0](
                    self.hwnd, GWL_STYLE, Int64((style & ~WS_OVERLAPPEDWINDOW) | WS_POPUP)
                )
            else:
                _ = Pointer(to=fn_set_long).unsafe_bitcast[SetLongFn]()[unsafe_offset=0](
                    self.hwnd, GWL_STYLE, Int64(self.saved_style | WS_OVERLAPPEDWINDOW)
                )
            _ = Pointer(to=fn_set_pos).unsafe_bitcast[SetPosFn]()[unsafe_offset=0](
                self.hwnd, 0, 0, 0, Int32(self.width), Int32(self.height), SWP_NOOWNERZORDER | SWP_FRAMECHANGED
            )

    def should_close(self) -> Bool:
        return self.close_requested

    def framebuffer_size(self) -> Tuple[Int, Int]:
        return (max(1, self.width), max(1, self.height))

    def content_scale(self) -> Float32:
        return max(Float32(1.0), Float32(self.dpi) / Float32(96.0))

    def clipboard_set(mut self, text: String):
        self.clipboard_text = text
        self.clipboard_owned = True
        comptime if IS_WINDOWS:
            if self.user32_lib == 0 or self.kernel32_lib == 0:
                return
            var fn_open = _win32_get_proc(self.user32_lib, "OpenClipboard")
            var fn_empty = _win32_get_proc(self.user32_lib, "EmptyClipboard")
            var fn_set = _win32_get_proc(self.user32_lib, "SetClipboardData")
            var fn_close = _win32_get_proc(self.user32_lib, "CloseClipboard")
            var fn_alloc = _win32_get_proc(self.kernel32_lib, "GlobalAlloc")
            var fn_lock = _win32_get_proc(self.kernel32_lib, "GlobalLock")
            var fn_unlock = _win32_get_proc(self.kernel32_lib, "GlobalUnlock")
            if (
                fn_open == 0
                or fn_empty == 0
                or fn_set == 0
                or fn_close == 0
                or fn_alloc == 0
                or fn_lock == 0
                or fn_unlock == 0
            ):
                return
            var units = utf8_to_utf16le(text, True)
            var byte_len = len(units) * 2
            comptime OpenFn = def(UInt64) thin abi("C") -> Int32
            if Pointer(to=fn_open).unsafe_bitcast[OpenFn]()[unsafe_offset=0](self.hwnd) == 0:
                return
            comptime EmptyFn = def() thin abi("C") -> Int32
            _ = Pointer(to=fn_empty).unsafe_bitcast[EmptyFn]()[unsafe_offset=0]()
            comptime AllocFn = def(UInt32, Int) thin abi("C") -> UInt64
            var hmem = Pointer(to=fn_alloc).unsafe_bitcast[AllocFn]()[unsafe_offset=0](
                GMEM_MOVEABLE, byte_len
            )
            if hmem != 0:
                comptime LockFn = def(UInt64) thin abi("C") -> Int
                comptime UnlockFn = def(UInt64) thin abi("C") -> Int32
                comptime SetFn = def(UInt32, UInt64) thin abi("C") -> UInt64
                var dst = Pointer(to=fn_lock).unsafe_bitcast[LockFn]()[unsafe_offset=0](hmem)
                if dst != 0:
                    _ = external_call["memcpy", Int](dst, Int(units.unsafe_ptr()), Int64(byte_len))
                    _ = Pointer(to=fn_unlock).unsafe_bitcast[UnlockFn]()[unsafe_offset=0](hmem)
                    _ = Pointer(to=fn_set).unsafe_bitcast[SetFn]()[unsafe_offset=0](
                        CF_UNICODETEXT, hmem
                    )
            _ = len(units)
            comptime CloseFn = def() thin abi("C") -> Int32
            _ = Pointer(to=fn_close).unsafe_bitcast[CloseFn]()[unsafe_offset=0]()

    def clipboard_get(mut self) -> String:
        comptime if IS_WINDOWS:
            if self.user32_lib != 0 and self.kernel32_lib != 0:
                var fn_open = _win32_get_proc(self.user32_lib, "OpenClipboard")
                var fn_get = _win32_get_proc(self.user32_lib, "GetClipboardData")
                var fn_close = _win32_get_proc(self.user32_lib, "CloseClipboard")
                var fn_lock = _win32_get_proc(self.kernel32_lib, "GlobalLock")
                var fn_unlock = _win32_get_proc(self.kernel32_lib, "GlobalUnlock")
                if fn_open != 0 and fn_get != 0 and fn_close != 0 and fn_lock != 0 and fn_unlock != 0:
                    comptime OpenFn = def(UInt64) thin abi("C") -> Int32
                    if Pointer(to=fn_open).unsafe_bitcast[OpenFn]()[unsafe_offset=0](self.hwnd) != 0:
                        comptime GetFn = def(UInt32) thin abi("C") -> UInt64
                        var hmem = Pointer(to=fn_get).unsafe_bitcast[GetFn]()[unsafe_offset=0](CF_UNICODETEXT)
                        var out = String()
                        if hmem != 0:
                            comptime LockFn = def(UInt64) thin abi("C") -> Int
                            comptime UnlockFn = def(UInt64) thin abi("C") -> Int32
                            var ptr = Pointer(to=fn_lock).unsafe_bitcast[LockFn]()[unsafe_offset=0](hmem)
                            if ptr != 0:
                                out = utf16le_ptr_to_utf8(ptr)
                                _ = Pointer(to=fn_unlock).unsafe_bitcast[UnlockFn]()[unsafe_offset=0](hmem)
                        comptime CloseFn = def() thin abi("C") -> Int32
                        _ = Pointer(to=fn_close).unsafe_bitcast[CloseFn]()[unsafe_offset=0]()
                        return out
        return self.clipboard_text

    def attach_vulkan(
        mut self,
        mut vk: VulkanGpu,
        width: Int = 0,
        height: Int = 0,
        prefer_mailbox: Bool = True,
        prefer_hdr10: Bool = False,
    ) -> SwapchainHandle:
        var sw = width if width > 0 else self.width
        var sh = height if height > 0 else self.height
        var sc = vk.attach_win32_surface(
            self.hwnd,
            self.hinstance,
            width=sw,
            height=sh,
            prefer_mailbox=prefer_mailbox,
            prefer_hdr10=prefer_hdr10,
        )
        self.gpu_swapchain = sc.vk_swapchain
        return sc^

    def shutdown(mut self):
        comptime if IS_WINDOWS:
            if self.hwnd != 0 and self.hdc != 0 and self.user32_lib != 0:
                var fn_rel = _win32_get_proc(self.user32_lib, "ReleaseDC")
                if fn_rel != 0:
                    comptime RelFn = def(UInt64, UInt64) thin abi("C") -> Int32
                    _ = Pointer(to=fn_rel).unsafe_bitcast[RelFn]()[unsafe_offset=0](self.hwnd, self.hdc)
            if self.hwnd != 0 and self.user32_lib != 0:
                var fn_set_long = _win32_get_proc(self.user32_lib, "SetWindowLongPtrW")
                if fn_set_long != 0:
                    comptime SetLongPtrFn = def(UInt64, Int32, Int64) thin abi("C") -> Int64
                    _ = Pointer(to=fn_set_long).unsafe_bitcast[SetLongPtrFn]()[unsafe_offset=0](
                        self.hwnd, GWLP_USERDATA, Int64(0)
                    )
                var fn_dest = _win32_get_proc(self.user32_lib, "DestroyWindow")
                if fn_dest != 0:
                    comptime DestFn = def(UInt64) thin abi("C") -> Int32
                    _ = Pointer(to=fn_dest).unsafe_bitcast[DestFn]()[unsafe_offset=0](self.hwnd)
            if self.wndproc_state_ptr != 0:
                external_call["free", NoneType](self.wndproc_state_ptr)
            _win32_free_dll(self.shcore_lib)
            _win32_free_dll(self.gdi32_lib)
            _win32_free_dll(self.user32_lib)
            _win32_free_dll(self.kernel32_lib)
        self.hwnd = 0
        self.hinstance = 0
        self.hdc = 0
        self.user32_lib = 0
        self.gdi32_lib = 0
        self.kernel32_lib = 0
        self.shcore_lib = 0
        self.gpu_swapchain = 0
        self.wndproc_state_ptr = 0
        self.active_flag = False
        self.close_requested = False
        self.high_surrogate = 0
        self.pending_text_utf8 = ""
        self.pending_dropped_files = List[String]()
        self.dpi = 96
        self.fullscreen = False
        self.focused = True
        self.pointer_locked = False
