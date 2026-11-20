"""Pure-Mojo macOS Cocoa / AppKit / CAMetalLayer window & Vulkan WSI backend (`flint.platform.macos_window`).

Zero C/C++ wrapper libraries: talks directly to `libobjc.A.dylib`, `AppKit.framework`,
`QuartzCore.framework`, and `CoreGraphics.framework` via `objc_msgSend` when compiled
for macOS (respecting `objc_msgSend_stret` on x86_64 for `NSRect`-returning selectors),
while keeping Cocoa geometry, `NSEvent` decoding, and AppKit Y-flip helpers testable
on Linux.
"""
from std.collections import List
from std.memory import bitcast
from std.sys.info import CompilationTarget
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.input import InputState
from flint.platform.traits import WindowBackend
from flint.platform.vulkan import VulkanGpu, SwapchainHandle
from flint.text.utf8 import decode_utf8, encode_codepoint

# NSEventType constants (AppKit NSEvent.h)
comptime NS_EVENT_LEFT_MOUSE_DOWN: Int = 1
comptime NS_EVENT_LEFT_MOUSE_UP: Int = 2
comptime NS_EVENT_RIGHT_MOUSE_DOWN: Int = 3
comptime NS_EVENT_RIGHT_MOUSE_UP: Int = 4
comptime NS_EVENT_MOUSE_MOVED: Int = 5
comptime NS_EVENT_LEFT_MOUSE_DRAGGED: Int = 6
comptime NS_EVENT_RIGHT_MOUSE_DRAGGED: Int = 7
comptime NS_EVENT_KEY_DOWN: Int = 10
comptime NS_EVENT_KEY_UP: Int = 11
comptime NS_EVENT_FLAGS_CHANGED: Int = 12
comptime NS_EVENT_SCROLL_WHEEL: Int = 22
comptime NS_EVENT_OTHER_MOUSE_DOWN: Int = 25
comptime NS_EVENT_OTHER_MOUSE_UP: Int = 26
comptime NS_EVENT_OTHER_MOUSE_DRAGGED: Int = 27

# NSEventModifierFlags
comptime NS_MOD_CAPS_LOCK: UInt64 = 1 << 16
comptime NS_MOD_SHIFT: UInt64 = 1 << 17
comptime NS_MOD_CONTROL: UInt64 = 1 << 18
comptime NS_MOD_OPTION: UInt64 = 1 << 19
comptime NS_MOD_COMMAND: UInt64 = 1 << 20

# NSWindowStyleMask & backing
comptime NS_WINDOW_STYLE_TITLED: UInt64 = 1
comptime NS_WINDOW_STYLE_CLOSABLE: UInt64 = 2
comptime NS_WINDOW_STYLE_MINIATURIZABLE: UInt64 = 4
comptime NS_WINDOW_STYLE_RESIZABLE: UInt64 = 8
comptime NS_WINDOW_STYLE_DEFAULT: UInt64 = 15
comptime NS_BACKING_STORE_BUFFERED: UInt64 = 2

comptime COCOA_RECT_BYTES: Int = 32
comptime COCOA_POINT_BYTES: Int = 16
comptime CG_BITMAP_ARGB32_LE: UInt32 = 0x2002  # kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst


@fieldwise_init
struct CocoaRect(Copyable, ImplicitlyCopyable, Movable):
    """32-byte ABI-compatible `NSRect` / `CGRect` (`CGPoint origin; CGSize size;`)."""
    var x: Float64
    var y: Float64
    var width: Float64
    var height: Float64

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.width = 0.0
        self.height = 0.0


@fieldwise_init
struct CocoaPoint(Copyable, ImplicitlyCopyable, Movable):
    """16-byte ABI-compatible `NSPoint` / `CGPoint`."""
    var x: Float64
    var y: Float64

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0


@always_inline
def _mac_put64(mut buf: List[UInt8], off: Int, val: UInt64):
    for i in range(8):
        buf[off + i] = UInt8((val >> UInt64(i * 8)) & 0xFF)


@always_inline
def _mac_get64(buf: List[UInt8], off: Int) -> UInt64:
    var out = UInt64(0)
    for i in range(8):
        out |= UInt64(buf[off + i]) << UInt64(i * 8)
    return out


def cocoa_pack_rect(x: Float64, y: Float64, width: Float64, height: Float64) -> List[UInt8]:
    """Packs a 32-byte little-endian `NSRect` / `CGRect`."""
    var out = List[UInt8](length=COCOA_RECT_BYTES, fill=0)
    _mac_put64(out, 0, bitcast[DType.uint64](x))
    _mac_put64(out, 8, bitcast[DType.uint64](y))
    _mac_put64(out, 16, bitcast[DType.uint64](width))
    _mac_put64(out, 24, bitcast[DType.uint64](height))
    return out^


def cocoa_unpack_rect(buf: List[UInt8]) -> CocoaRect:
    """Unpacks a 32-byte little-endian `NSRect` / `CGRect`."""
    if len(buf) < COCOA_RECT_BYTES:
        return CocoaRect()
    return CocoaRect(
        bitcast[DType.float64](_mac_get64(buf, 0)),
        bitcast[DType.float64](_mac_get64(buf, 8)),
        bitcast[DType.float64](_mac_get64(buf, 16)),
        bitcast[DType.float64](_mac_get64(buf, 24)),
    )


def cocoa_flip_y(appkit_y: Float64, window_height: Int) -> Float32:
    """Converts AppKit bottom-left origin Y coordinate to Flint top-left origin Y."""
    return Float32(Float64(max(1, window_height)) - appkit_y)


def cocoa_keycode_to_flint_key(keycode: Int) -> Int:
    """Maps macOS Carbon/HIToolbox virtual keycodes (`kVK_*`) to Flint `InputState` slots."""
    if keycode == 0x00:
        return 65   # A
    if keycode == 0x0B:
        return 66   # B
    if keycode == 0x08:
        return 67   # C
    if keycode == 0x02:
        return 68   # D
    if keycode == 0x0E:
        return 69   # E
    if keycode == 0x03:
        return 70   # F
    if keycode == 0x05:
        return 71   # G
    if keycode == 0x04:
        return 72   # H
    if keycode == 0x22:
        return 73   # I
    if keycode == 0x26:
        return 74   # J
    if keycode == 0x28:
        return 75   # K
    if keycode == 0x25:
        return 76   # L
    if keycode == 0x2E:
        return 77   # M
    if keycode == 0x2D:
        return 78   # N
    if keycode == 0x1F:
        return 79   # O
    if keycode == 0x23:
        return 80   # P
    if keycode == 0x0C:
        return 81   # Q
    if keycode == 0x0F:
        return 82   # R
    if keycode == 0x01:
        return 83   # S
    if keycode == 0x11:
        return 84   # T
    if keycode == 0x20:
        return 85   # U
    if keycode == 0x09:
        return 86   # V
    if keycode == 0x0D:
        return 87   # W
    if keycode == 0x07:
        return 88   # X
    if keycode == 0x10:
        return 89   # Y
    if keycode == 0x06:
        return 90   # Z
    if keycode == 0x1D:
        return 48   # 0
    if keycode == 0x12:
        return 49   # 1
    if keycode == 0x13:
        return 50   # 2
    if keycode == 0x14:
        return 51   # 3
    if keycode == 0x15:
        return 52   # 4
    if keycode == 0x17:
        return 53   # 5
    if keycode == 0x16:
        return 54   # 6
    if keycode == 0x1A:
        return 55   # 7
    if keycode == 0x1C:
        return 56   # 8
    if keycode == 0x19:
        return 57   # 9
    if keycode == 0x31:
        return 32   # Space
    if keycode == 0x27:
        return 39   # Apostrophe
    if keycode == 0x2B:
        return 44   # Comma
    if keycode == 0x1B:
        return 45   # Minus
    if keycode == 0x2F:
        return 46   # Period
    if keycode == 0x2C:
        return 47   # Slash
    if keycode == 0x29:
        return 59   # Semicolon
    if keycode == 0x18:
        return 61   # Equal
    if keycode == 0x21:
        return 91   # LeftBracket
    if keycode == 0x2A:
        return 92   # Backslash
    if keycode == 0x1E:
        return 93   # RightBracket
    if keycode == 0x32:
        return 96   # Grave
    if keycode == 0x35:
        return 256  # Escape
    if keycode == 0x24 or keycode == 0x4C:
        return 257  # Return / KeypadEnter
    if keycode == 0x30:
        return 258  # Tab
    if keycode == 0x33:
        return 259  # Delete (Backspace)
    if keycode == 0x72:
        return 260  # Help/Insert
    if keycode == 0x75:
        return 261  # ForwardDelete
    if keycode == 0x7C:
        return 262  # RightArrow
    if keycode == 0x7B:
        return 263  # LeftArrow
    if keycode == 0x7D:
        return 264  # DownArrow
    if keycode == 0x7E:
        return 265  # UpArrow
    if keycode == 0x74:
        return 266  # PageUp
    if keycode == 0x79:
        return 267  # PageDown
    if keycode == 0x73:
        return 268  # Home
    if keycode == 0x77:
        return 269  # End
    if keycode == 0x39:
        return 280  # CapsLock
    if keycode == 0x7A:
        return 290  # F1
    if keycode == 0x78:
        return 291  # F2
    if keycode == 0x63:
        return 292  # F3
    if keycode == 0x76:
        return 293  # F4
    if keycode == 0x60:
        return 294  # F5
    if keycode == 0x61:
        return 295  # F6
    if keycode == 0x62:
        return 296  # F7
    if keycode == 0x64:
        return 297  # F8
    if keycode == 0x65:
        return 298  # F9
    if keycode == 0x6D:
        return 299  # F10
    if keycode == 0x67:
        return 300  # F11
    if keycode == 0x6F:
        return 301  # F12
    if keycode == 0x38:
        return 340  # LeftShift
    if keycode == 0x3B:
        return 341  # LeftCtrl
    if keycode == 0x3A:
        return 342  # LeftOption (Alt)
    if keycode == 0x37:
        return 343  # LeftCommand (Super)
    if keycode == 0x3C:
        return 344  # RightShift
    if keycode == 0x3E:
        return 345  # RightCtrl
    if keycode == 0x3D:
        return 346  # RightOption
    if keycode == 0x36:
        return 347  # RightCommand
    return -1


def _filter_cocoa_text_utf8(chars_utf8: String) -> String:
    """Filters ASCII control characters and AppKit private-use function keys (U+F700..U+F8FF) per codepoint."""
    if chars_utf8.byte_length() == 0:
        return String()
    var cps = decode_utf8(chars_utf8)
    var out_bytes = List[UInt8](capacity=chars_utf8.byte_length())
    for i in range(len(cps)):
        var cp = cps[i]
        if cp < 0x20 or cp == 0x7F or (cp >= 0xF700 and cp <= 0xF8FF):
            continue
        encode_codepoint(out_bytes, cp)
    return String(unsafe_from_utf8=out_bytes) if len(out_bytes) > 0 else String()


struct CocoaWindow(WindowBackend):
    """Pure-Mojo macOS Cocoa `WindowBackend` (`libobjc.A.dylib` + AppKit + `CAMetalLayer`)."""
    var ns_app: UInt64
    var ns_window: UInt64
    var ns_view: UInt64
    var ca_metal_layer: UInt64
    var width: Int
    var height: Int
    var backing_scale: Float32
    var close_requested: Bool
    var focused: Bool
    var fullscreen: Bool
    var pointer_locked: Bool
    var window_title: String
    var clipboard_text: String
    var clipboard_owned: Bool
    var gpu_swapchain: UInt64
    var objc_lib: Int
    var appkit_lib: Int
    var quartz_lib: Int
    var coregraphics_lib: Int
    var corefoundation_lib: Int
    var fn_get_class: Int
    var fn_sel_reg: Int
    var fn_msg_send: Int
    var fn_msg_send_stret: Int
    var pending_text_utf8: String
    var pending_dropped_files: List[String]
    var active_flag: Bool

    def __init__(out self):
        self.ns_app = 0
        self.ns_window = 0
        self.ns_view = 0
        self.ca_metal_layer = 0
        self.width = 1280
        self.height = 720
        self.backing_scale = 1.0
        self.close_requested = False
        self.focused = True
        self.fullscreen = False
        self.pointer_locked = False
        self.window_title = "FLINT ENGINE"
        self.clipboard_text = ""
        self.clipboard_owned = False
        self.gpu_swapchain = 0
        self.objc_lib = 0
        self.appkit_lib = 0
        self.quartz_lib = 0
        self.coregraphics_lib = 0
        self.corefoundation_lib = 0
        self.fn_get_class = 0
        self.fn_sel_reg = 0
        self.fn_msg_send = 0
        self.fn_msg_send_stret = 0
        self.pending_text_utf8 = ""
        self.pending_dropped_files = List[String]()
        self.active_flag = False

    def __moveinit__(out self, deinit existing: Self):
        self.ns_app = existing.ns_app
        self.ns_window = existing.ns_window
        self.ns_view = existing.ns_view
        self.ca_metal_layer = existing.ca_metal_layer
        self.width = existing.width
        self.height = existing.height
        self.backing_scale = existing.backing_scale
        self.close_requested = existing.close_requested
        self.focused = existing.focused
        self.fullscreen = existing.fullscreen
        self.pointer_locked = existing.pointer_locked
        self.window_title = existing.window_title^
        self.clipboard_text = existing.clipboard_text^
        self.clipboard_owned = existing.clipboard_owned
        self.gpu_swapchain = existing.gpu_swapchain
        self.objc_lib = existing.objc_lib
        self.appkit_lib = existing.appkit_lib
        self.quartz_lib = existing.quartz_lib
        self.coregraphics_lib = existing.coregraphics_lib
        self.corefoundation_lib = existing.corefoundation_lib
        self.fn_get_class = existing.fn_get_class
        self.fn_sel_reg = existing.fn_sel_reg
        self.fn_msg_send = existing.fn_msg_send
        self.fn_msg_send_stret = existing.fn_msg_send_stret
        self.pending_text_utf8 = existing.pending_text_utf8^
        self.pending_dropped_files = existing.pending_dropped_files^
        self.active_flag = existing.active_flag
        existing.ns_app = 0
        existing.ns_window = 0
        existing.ns_view = 0
        existing.ca_metal_layer = 0
        existing.gpu_swapchain = 0
        existing.objc_lib = 0
        existing.appkit_lib = 0
        existing.quartz_lib = 0
        existing.coregraphics_lib = 0
        existing.corefoundation_lib = 0
        existing.fn_get_class = 0
        existing.fn_sel_reg = 0
        existing.fn_msg_send = 0
        existing.fn_msg_send_stret = 0
        existing.active_flag = False

    def __deinit__(deinit self):
        self.shutdown()

    def active(self) -> Bool:
        return self.active_flag and self.ns_window != 0

    def _objc_cls(self, name: String) -> UInt64:
        if self.fn_get_class == 0:
            return 0
        var fn_p = self.fn_get_class
        var owned = name
        comptime GetClsFn = def(Int) thin abi("C") -> UInt64
        var c = Pointer(to=fn_p).unsafe_bitcast[GetClsFn]()[unsafe_offset=0](
            Int(owned.as_c_string_span().ptr())
        )
        _ = owned.byte_length()
        return c

    def _objc_sel(self, name: String) -> UInt64:
        if self.fn_sel_reg == 0:
            return 0
        var fn_p = self.fn_sel_reg
        var owned = name
        comptime SelFn = def(Int) thin abi("C") -> UInt64
        var s = Pointer(to=fn_p).unsafe_bitcast[SelFn]()[unsafe_offset=0](
            Int(owned.as_c_string_span().ptr())
        )
        _ = owned.byte_length()
        return s

    def _msg0(self, obj: UInt64, sel_name: String) -> UInt64:
        if obj == 0 or self.fn_msg_send == 0:
            return 0
        var sel = self._objc_sel(sel_name)
        var fn_p = self.fn_msg_send
        comptime Msg0 = def(UInt64, UInt64) thin abi("C") -> UInt64
        return Pointer(to=fn_p).unsafe_bitcast[Msg0]()[unsafe_offset=0](obj, sel)

    def _msg1(self, obj: UInt64, sel_name: String, arg0: UInt64) -> UInt64:
        if obj == 0 or self.fn_msg_send == 0:
            return 0
        var sel = self._objc_sel(sel_name)
        var fn_p = self.fn_msg_send
        comptime Msg1 = def(UInt64, UInt64, UInt64) thin abi("C") -> UInt64
        return Pointer(to=fn_p).unsafe_bitcast[Msg1]()[unsafe_offset=0](obj, sel, arg0)

    def _msg0_f64(self, obj: UInt64, sel_name: String) -> Float64:
        if obj == 0 or self.fn_msg_send == 0:
            return 0.0
        var sel = self._objc_sel(sel_name)
        var fn_p = self.fn_msg_send
        comptime MsgF64 = def(UInt64, UInt64) thin abi("C") -> Float64
        return Pointer(to=fn_p).unsafe_bitcast[MsgF64]()[unsafe_offset=0](obj, sel)

    def _msg0_rect(self, obj: UInt64, sel_name: String) -> CocoaRect:
        if obj == 0 or self.fn_msg_send == 0:
            return CocoaRect()
        var sel = self._objc_sel(sel_name)
        comptime if CompilationTarget.is_x86():
            if self.fn_msg_send_stret != 0:
                var fn_s = self.fn_msg_send_stret
                var out_rect = CocoaRect()
                comptime MsgStret = def(Int, UInt64, UInt64) thin abi("C") -> None
                Pointer(to=fn_s).unsafe_bitcast[MsgStret]()[unsafe_offset=0](
                    Int(Pointer(to=out_rect)), obj, sel
                )
                return out_rect
        var fn_p = self.fn_msg_send
        comptime MsgRect = def(UInt64, UInt64) thin abi("C") -> CocoaRect
        return Pointer(to=fn_p).unsafe_bitcast[MsgRect]()[unsafe_offset=0](obj, sel)

    def _nsstring(self, text: String) -> UInt64:
        var cls = self._objc_cls("NSString")
        if cls == 0 or self.fn_msg_send == 0:
            return 0
        var sel = self._objc_sel("stringWithUTF8String:")
        var owned = text
        var fn_p = self.fn_msg_send
        comptime MsgPtr = def(UInt64, UInt64, Int) thin abi("C") -> UInt64
        var ns = Pointer(to=fn_p).unsafe_bitcast[MsgPtr]()[unsafe_offset=0](
            cls, sel, Int(owned.as_c_string_span().ptr())
        )
        _ = owned.byte_length()
        return ns

    def _nsstring_to_utf8(self, ns_str: UInt64) -> String:
        if ns_str == 0:
            return String()
        var ptr = Int(self._msg0(ns_str, "UTF8String"))
        if ptr == 0:
            return String()
        var bytes = List[UInt8]()
        var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=ptr)
        var i = 0
        while True:
            var b = p[unsafe_offset=i]
            if b == 0:
                break
            bytes.append(b)
            i += 1
        return String(unsafe_from_utf8=bytes) if len(bytes) > 0 else String()

    def initialize(mut self, width: Int, height: Int) -> Bool:
        var title = self.window_title
        return self.initialize(width, height, title)

    def initialize(mut self, width: Int, height: Int, title: String) -> Bool:
        self.shutdown()
        self.width = max(1, width)
        self.height = max(1, height)
        self.window_title = title
        comptime if CompilationTarget.is_macos():
            self.objc_lib = dl_open("/usr/lib/libobjc.A.dylib")
            self.appkit_lib = dl_open("/System/Library/Frameworks/AppKit.framework/AppKit")
            self.quartz_lib = dl_open("/System/Library/Frameworks/QuartzCore.framework/QuartzCore")
            self.coregraphics_lib = dl_open(
                "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            )
            self.corefoundation_lib = dl_open(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            if self.objc_lib == 0 or self.appkit_lib == 0 or self.quartz_lib == 0:
                self.shutdown()
                return False
            self.fn_get_class = dl_sym(self.objc_lib, "objc_getClass")
            self.fn_sel_reg = dl_sym(self.objc_lib, "sel_registerName")
            self.fn_msg_send = dl_sym(self.objc_lib, "objc_msgSend")
            self.fn_msg_send_stret = dl_sym(self.objc_lib, "objc_msgSend_stret")
            if self.fn_get_class == 0 or self.fn_sel_reg == 0 or self.fn_msg_send == 0:
                self.shutdown()
                return False

            var cls_app = self._objc_cls("NSApplication")
            self.ns_app = self._msg0(cls_app, "sharedApplication")
            if self.ns_app == 0:
                self.shutdown()
                return False
            _ = self._msg1(self.ns_app, "setActivationPolicy:", 0)
            _ = self._msg0(self.ns_app, "finishLaunching")

            var cls_win = self._objc_cls("NSWindow")
            var win_alloc = self._msg0(cls_win, "alloc")
            var rect = CocoaRect(0.0, 0.0, Float64(self.width), Float64(self.height))
            var sel_init = self._objc_sel("initWithContentRect:styleMask:backing:defer:")
            var fn_p = self.fn_msg_send
            comptime InitWinFn = def(
                UInt64, UInt64, CocoaRect, UInt64, UInt64, UInt8
            ) thin abi("C") -> UInt64
            self.ns_window = Pointer(to=fn_p).unsafe_bitcast[InitWinFn]()[unsafe_offset=0](
                win_alloc,
                sel_init,
                rect,
                NS_WINDOW_STYLE_DEFAULT,
                NS_BACKING_STORE_BUFFERED,
                0,
            )
            if self.ns_window == 0:
                self.shutdown()
                return False

            var ns_title = self._nsstring(title)
            if ns_title != 0:
                _ = self._msg1(self.ns_window, "setTitle:", ns_title)
            _ = self._msg1(self.ns_window, "setReleasedWhenClosed:", 0)

            self.ns_view = self._msg0(self.ns_window, "contentView")
            var cls_layer = self._objc_cls("CAMetalLayer")
            self.ca_metal_layer = self._msg0(cls_layer, "layer")
            if self.ns_view != 0 and self.ca_metal_layer != 0:
                _ = self._msg1(self.ns_view, "setWantsLayer:", 1)
                _ = self._msg1(self.ns_view, "setLayer:", self.ca_metal_layer)

            var scale = self._msg0_f64(self.ns_window, "backingScaleFactor")
            if scale >= 1.0:
                self.backing_scale = Float32(scale)
            _ = self._msg1(self.ns_window, "makeKeyAndOrderFront:", 0)
            _ = self._msg1(self.ns_app, "activateIgnoringOtherApps:", 1)
            self.active_flag = True
            return True
        else:
            return False

    def process_cocoa_event(
        mut self,
        event_type: Int,
        keycode: Int,
        modifiers: UInt64,
        x: Float64,
        y: Float64,
        dx: Float64,
        dy: Float64,
        wheel_y: Float64,
        chars_utf8: String,
        mut input: InputState,
    ):
        """Translates an AppKit `NSEvent` into `CocoaWindow` and `InputState` updates."""
        if event_type == NS_EVENT_LEFT_MOUSE_DOWN:
            input.set_mouse(0, True)
        elif event_type == NS_EVENT_LEFT_MOUSE_UP:
            input.set_mouse(0, False)
        elif event_type == NS_EVENT_RIGHT_MOUSE_DOWN:
            input.set_mouse(1, True)
        elif event_type == NS_EVENT_RIGHT_MOUSE_UP:
            input.set_mouse(1, False)
        elif event_type == NS_EVENT_OTHER_MOUSE_DOWN:
            input.set_mouse(2, True)
        elif event_type == NS_EVENT_OTHER_MOUSE_UP:
            input.set_mouse(2, False)
        elif (
            event_type == NS_EVENT_MOUSE_MOVED
            or event_type == NS_EVENT_LEFT_MOUSE_DRAGGED
            or event_type == NS_EVENT_RIGHT_MOUSE_DRAGGED
            or event_type == NS_EVENT_OTHER_MOUSE_DRAGGED
        ):
            if self.pointer_locked:
                input.add_mouse_delta(Float32(dx), Float32(dy))
            else:
                input.move_mouse(Float32(x), cocoa_flip_y(y, self.height))
        elif event_type == NS_EVENT_SCROLL_WHEEL:
            input.wheel += Float32(wheel_y)
        elif event_type == NS_EVENT_KEY_DOWN:
            var fk = cocoa_keycode_to_flint_key(keycode)
            if fk >= 0:
                input.set_key(fk, True)
            var filtered = _filter_cocoa_text_utf8(chars_utf8)
            if filtered.byte_length() > 0:
                input.push_text_utf8(filtered)
        elif event_type == NS_EVENT_KEY_UP:
            var fk = cocoa_keycode_to_flint_key(keycode)
            if fk >= 0:
                input.set_key(fk, False)
        elif event_type == NS_EVENT_FLAGS_CHANGED:
            input.set_key(340, (modifiers & NS_MOD_SHIFT) != 0)
            input.set_key(341, (modifiers & NS_MOD_CONTROL) != 0)
            input.set_key(342, (modifiers & NS_MOD_OPTION) != 0)
            input.set_key(343, (modifiers & NS_MOD_COMMAND) != 0)
            input.set_key(280, (modifiers & NS_MOD_CAPS_LOCK) != 0)

    def poll_events(mut self, mut input: InputState):
        comptime if CompilationTarget.is_macos():
            if not self.active() or self.ns_app == 0 or self.fn_msg_send == 0:
                return
            var cls_pool = self._objc_cls("NSAutoreleasePool")
            var pool = self._msg0(self._msg0(cls_pool, "alloc"), "init")
            var mode = self._nsstring("kCFRunLoopDefaultMode")
            var sel_next = self._objc_sel("nextEventMatchingMask:untilDate:inMode:dequeue:")
            var fn_p = self.fn_msg_send
            comptime NextEventFn = def(
                UInt64, UInt64, UInt64, UInt64, UInt64, UInt8
            ) thin abi("C") -> UInt64
            comptime LocFn = def(UInt64, UInt64) thin abi("C") -> CocoaPoint

            # Query current contentView bounds & backingScaleFactor
            if self.ns_view != 0:
                var frame = self._msg0_rect(self.ns_view, "bounds")
                if frame.width >= 1.0 and frame.height >= 1.0:
                    self.width = Int(frame.width)
                    self.height = Int(frame.height)
            var scale = self._msg0_f64(self.ns_window, "backingScaleFactor")
            if scale >= 1.0:
                self.backing_scale = Float32(scale)
            if self._msg0(self.ns_window, "isVisible") == 0:
                self.close_requested = True

            for _ in range(256):
                var ev = Pointer(to=fn_p).unsafe_bitcast[NextEventFn]()[unsafe_offset=0](
                    self.ns_app, sel_next, 0xFFFFFFFFFFFFFFFF, 0, mode, 1
                )
                if ev == 0:
                    break
                var ev_type = Int(self._msg0(ev, "type"))
                var keycode = Int(self._msg0(ev, "keyCode") & 0xFFFF)
                var mods = self._msg0(ev, "modifierFlags")
                var sel_loc = self._objc_sel("locationInWindow")
                var pt = Pointer(to=fn_p).unsafe_bitcast[LocFn]()[unsafe_offset=0](ev, sel_loc)
                var dx = Float64(0.0)
                var dy = Float64(0.0)
                var wheel_y = Float64(0.0)
                var chars = String()
                if (
                    ev_type == NS_EVENT_MOUSE_MOVED
                    or ev_type == NS_EVENT_LEFT_MOUSE_DRAGGED
                    or ev_type == NS_EVENT_RIGHT_MOUSE_DRAGGED
                    or ev_type == NS_EVENT_OTHER_MOUSE_DRAGGED
                ):
                    dx = self._msg0_f64(ev, "deltaX")
                    dy = self._msg0_f64(ev, "deltaY")
                elif ev_type == NS_EVENT_SCROLL_WHEEL:
                    wheel_y = self._msg0_f64(ev, "scrollingDeltaY")
                elif ev_type == NS_EVENT_KEY_DOWN:
                    chars = self._nsstring_to_utf8(self._msg0(ev, "characters"))
                self.process_cocoa_event(
                    ev_type, keycode, mods, pt.x, pt.y, dx, dy, wheel_y, chars, input
                )
                _ = self._msg1(self.ns_app, "sendEvent:", ev)
            _ = self._msg0(self.ns_app, "updateWindows")
            if pool != 0:
                _ = self._msg0(pool, "drain")
        else:
            _ = input.uses_legacy_jsdev()

    def present_argb(mut self, pixels: List[UInt32], width: Int, height: Int):
        comptime if CompilationTarget.is_macos():
            if (
                not self.active()
                or self.coregraphics_lib == 0
                or self.corefoundation_lib == 0
                or self.ca_metal_layer == 0
                or width <= 0
                or height <= 0
                or len(pixels) < width * height
            ):
                return
            var fn_cf_data = dl_sym(self.corefoundation_lib, "CFDataCreate")
            var fn_cf_rel = dl_sym(self.corefoundation_lib, "CFRelease")
            var fn_prov = dl_sym(self.coregraphics_lib, "CGDataProviderCreateWithCFData")
            var fn_cs = dl_sym(self.coregraphics_lib, "CGColorSpaceCreateDeviceRGB")
            var fn_img = dl_sym(self.coregraphics_lib, "CGImageCreate")
            var fn_rel_prov = dl_sym(self.coregraphics_lib, "CGDataProviderRelease")
            var fn_rel_cs = dl_sym(self.coregraphics_lib, "CGColorSpaceRelease")
            var fn_rel_img = dl_sym(self.coregraphics_lib, "CGImageRelease")
            if fn_cf_data == 0 or fn_cf_rel == 0 or fn_prov == 0 or fn_cs == 0 or fn_img == 0:
                return
            comptime CfDataFn = def(Int, Int, Int) thin abi("C") -> UInt64
            comptime ProvFn = def(UInt64) thin abi("C") -> UInt64
            comptime CsFn = def() thin abi("C") -> UInt64
            comptime ImgFn = def(
                Int, Int, Int, Int, Int, UInt64, UInt32, UInt64, Int, UInt8, Int32
            ) thin abi("C") -> UInt64
            comptime RelFn = def(UInt64) thin abi("C") -> None

            var byte_size = width * height * 4
            var cf_data = Pointer(to=fn_cf_data).unsafe_bitcast[CfDataFn]()[unsafe_offset=0](
                0, Int(pixels.unsafe_ptr()), byte_size
            )
            _ = len(pixels)
            if cf_data == 0:
                return
            var prov = Pointer(to=fn_prov).unsafe_bitcast[ProvFn]()[unsafe_offset=0](cf_data)
            Pointer(to=fn_cf_rel).unsafe_bitcast[RelFn]()[unsafe_offset=0](cf_data)
            if prov == 0:
                return
            var cs = Pointer(to=fn_cs).unsafe_bitcast[CsFn]()[unsafe_offset=0]()
            var img = Pointer(to=fn_img).unsafe_bitcast[ImgFn]()[unsafe_offset=0](
                width, height, 8, 32, width * 4, cs, CG_BITMAP_ARGB32_LE, prov, 0, 0, 0
            )
            if img != 0:
                _ = self._msg1(self.ca_metal_layer, "setContents:", img)
                if fn_rel_img != 0:
                    Pointer(to=fn_rel_img).unsafe_bitcast[RelFn]()[unsafe_offset=0](img)
            if fn_rel_prov != 0:
                Pointer(to=fn_rel_prov).unsafe_bitcast[RelFn]()[unsafe_offset=0](prov)
            if cs != 0 and fn_rel_cs != 0:
                Pointer(to=fn_rel_cs).unsafe_bitcast[RelFn]()[unsafe_offset=0](cs)
        else:
            pass

    def toggle_fullscreen(mut self):
        self.fullscreen = not self.fullscreen
        comptime if CompilationTarget.is_macos():
            if self.active():
                _ = self._msg1(self.ns_window, "toggleFullScreen:", 0)

    def should_close(self) -> Bool:
        return self.close_requested

    def framebuffer_size(self) -> Tuple[Int, Int]:
        var s = self.content_scale()
        return (
            max(1, Int(Float32(self.width) * s + Float32(0.5))),
            max(1, Int(Float32(self.height) * s + Float32(0.5))),
        )

    def content_scale(self) -> Float32:
        return max(Float32(1.0), self.backing_scale)

    def clipboard_set(mut self, text: String):
        self.clipboard_text = text
        self.clipboard_owned = True
        comptime if CompilationTarget.is_macos():
            if self.fn_msg_send == 0:
                return
            var cls_pb = self._objc_cls("NSPasteboard")
            var pb = self._msg0(cls_pb, "generalPasteboard")
            if pb == 0:
                return
            _ = self._msg0(pb, "clearContents")
            var ns_txt = self._nsstring(text)
            var ns_type = self._nsstring("public.utf8-plain-text")
            if ns_txt != 0 and ns_type != 0:
                var sel = self._objc_sel("setString:forType:")
                var fn_p = self.fn_msg_send
                comptime SetStrFn = def(UInt64, UInt64, UInt64, UInt64) thin abi("C") -> UInt8
                _ = Pointer(to=fn_p).unsafe_bitcast[SetStrFn]()[unsafe_offset=0](
                    pb, sel, ns_txt, ns_type
                )

    def clipboard_get(mut self) -> String:
        comptime if CompilationTarget.is_macos():
            if self.fn_msg_send != 0:
                var cls_pool = self._objc_cls("NSAutoreleasePool")
                var pool = self._msg0(self._msg0(cls_pool, "alloc"), "init")
                var cls_pb = self._objc_cls("NSPasteboard")
                var pb = self._msg0(cls_pb, "generalPasteboard")
                var out = String()
                var read_ok = False
                if pb != 0:
                    var ns_type = self._nsstring("public.utf8-plain-text")
                    if ns_type != 0:
                        var ns_str = self._msg1(pb, "stringForType:", ns_type)
                        out = self._nsstring_to_utf8(ns_str)
                        read_ok = True
                if pool != 0:
                    _ = self._msg0(pool, "drain")
                if read_ok:
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
        var fb = self.framebuffer_size()
        var sw = width if width > 0 else fb[0]
        var sh = height if height > 0 else fb[1]
        var sc = vk.attach_macos_surface(
            self.ca_metal_layer,
            width=sw,
            height=sh,
            prefer_mailbox=prefer_mailbox,
            prefer_hdr10=prefer_hdr10,
        )
        self.gpu_swapchain = sc.vk_swapchain
        return sc^

    def shutdown(mut self):
        comptime if CompilationTarget.is_macos():
            if self.ns_window != 0 and self.fn_msg_send != 0:
                _ = self._msg0(self.ns_window, "close")
            _ = dl_close(self.corefoundation_lib)
            _ = dl_close(self.coregraphics_lib)
            _ = dl_close(self.quartz_lib)
            _ = dl_close(self.appkit_lib)
            _ = dl_close(self.objc_lib)
        self.ns_app = 0
        self.ns_window = 0
        self.ns_view = 0
        self.ca_metal_layer = 0
        self.gpu_swapchain = 0
        self.objc_lib = 0
        self.appkit_lib = 0
        self.quartz_lib = 0
        self.coregraphics_lib = 0
        self.corefoundation_lib = 0
        self.fn_get_class = 0
        self.fn_sel_reg = 0
        self.fn_msg_send = 0
        self.fn_msg_send_stret = 0
        self.active_flag = False
        self.close_requested = False
        self.backing_scale = 1.0
        self.fullscreen = False
        self.focused = True
        self.pointer_locked = False
        self.pending_text_utf8 = ""
        self.pending_dropped_files = List[String]()
