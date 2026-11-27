"""Direct X11 core wire-protocol window & input backend over Linux AF_UNIX sockets.

Speaks the binary X11R6 protocol directly to `/tmp/.X11-unix/X<N>` using standard
`.Xauthority` (`MIT-MAGIC-COOKIE-1`) authentication — zero `libX11`, `libxcb`, XKB,
GLX, or C/C++ wrapper libraries are linked.
"""
from std.ffi import external_call
from flint.platform.input import InputState, layout_shortcut_code
from flint.platform.traits import WindowBackend
from flint.input.evdev import EvdevHub
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.vulkan import VulkanGpu, SwapchainHandle
from flint.platform.xkb import x11_keysym_to_codepoint
from flint.text.utf8 import encode_codepoint
from flint.platform.os import env_value, env_now, env_flag, read_bytes, close_fd, get_time

comptime X11_RELEASE_HELD: Int = -1
comptime X11_ATOM_RESOURCE_MANAGER: Int = 23  # predefined XA_RESOURCE_MANAGER
comptime X11_PROPERTY_CHANGE_MASK: Int = 0x400000
comptime X11_MOUSE_BUTTONS: Int = 3


def x11_put16(mut data: List[UInt8], offset: Int, value: Int):
    data[offset] = UInt8(value & 255)
    data[offset + 1] = UInt8((value >> 8) & 255)


def x11_put32(mut data: List[UInt8], offset: Int, value: Int):
    data[offset] = UInt8(value & 255)
    data[offset + 1] = UInt8((value >> 8) & 255)
    data[offset + 2] = UInt8((value >> 16) & 255)
    data[offset + 3] = UInt8((value >> 24) & 255)


def x11_get16(data: List[UInt8], offset: Int) -> Int:
    return Int(data[offset]) | (Int(data[offset + 1]) << 8)


def x11_gets16(data: List[UInt8], offset: Int) -> Int:
    var v = x11_get16(data, offset)
    return v - 65536 if v >= 32768 else v


def x11_get32(data: List[UInt8], offset: Int) -> Int:
    return (
        Int(data[offset])
        | (Int(data[offset + 1]) << 8)
        | (Int(data[offset + 2]) << 16)
        | (Int(data[offset + 3]) << 24)
    )


def x11_gets32(data: List[UInt8], offset: Int) -> Int:
    var v = x11_get32(data, offset)
    return v - 4294967296 if v >= 2147483648 else v


def x11_fp3232(data: List[UInt8], offset: Int) -> Float32:
    var integral = x11_gets32(data, offset)
    var frac = UInt32(x11_get32(data, offset + 4))
    return Float32(Float64(integral) + Float64(frac) / 4294967296.0)


def x11_be16(data: List[UInt8], offset: Int) -> Int:
    return (Int(data[offset]) << 8) | Int(data[offset + 1])


def x11_evdev_key(key: Int) -> Int:
    """Maps Linux evdev keycodes (X11 keycode - 8) to Flint physical key slots."""
    var letters = [30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44]
    for index in range(26):
        if key == letters[index]:
            return 65 + index
    # Keypad (GLFW slots): KP_0..KP_9 = 320..329, KP_DECIMAL 330, KP_DIVIDE 331, KP_MULTIPLY 332,
    # KP_SUBTRACT 333, KP_ADD 334 (KP_ENTER stays Enter, 257).
    var keypad = [82, 79, 80, 81, 75, 76, 77, 71, 72, 73]
    for index in range(10):
        if key == keypad[index]:
            return 320 + index
    if key == 83:
        return 330
    if key == 98:
        return 331
    if key == 55:
        return 332
    if key == 74:
        return 333
    if key == 78:
        return 334
    var digits = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for index in range(10):
        if key == digits[index]:
            return 48 + index
    if key == 1:
        return 256
    if key == 28 or key == 96:
        return 257
    if key == 57:
        return 32
    if key == 105:
        return 263
    if key == 106:
        return 262
    if key == 103:
        return 265
    if key == 108:
        return 264
    if key == 15:
        return 258
    if key == 14:
        return 259
    if key == 41:
        return 96
    if key == 12:
        return 45
    if key == 52:
        return 46
    if key == 42:
        return 340
    if key == 29:
        return 341
    if key == 56:
        return 342
    if key == 54:
        return 344
    if key == 97:
        return 345
    if key == 100:
        return 346
    if key >= 59 and key <= 68:
        return 290 + key - 59
    if key == 87:
        return 300
    if key == 88:
        return 301
    if key == 13:
        return 61
    if key == 26:
        return 91
    if key == 27:
        return 93
    if key == 39:
        return 59
    if key == 40:
        return 39
    if key == 43:
        return 92
    if key == 51:
        return 44
    if key == 53:
        return 47
    if key == 102:
        return 268  # Home
    if key == 107:
        return 269  # End
    if key == 110:
        return 260  # Insert
    if key == 111:
        return 261  # Delete
    if key == 104:
        return 266  # PageUp
    if key == 109:
        return 267  # PageDown
    if key == 58:
        return 280  # CapsLock
    if key == 125:
        return 343  # LeftSuper
    if key == 126:
        return 347  # RightSuper
    return -1


def parse_xft_dpi(resources: String) -> Float32:
    """`Xft.dpi` from the root window's RESOURCE_MANAGER string (`Xft.dpi:\t144` lines, as written by
    xrdb or the desktop's settings daemon). 0 when absent or not a positive number."""
    var bytes = resources.as_bytes()
    var n = len(bytes)
    var key = String("Xft.dpi")
    var kb = key.as_bytes()
    var kn = len(kb)
    var line_start = 0
    while line_start < n:
        var line_end = line_start
        while line_end < n and bytes[line_end] != 10:
            line_end += 1
        var i = line_start
        while i < line_end and (bytes[i] == 32 or bytes[i] == 9):
            i += 1
        var matched = i + kn <= line_end
        if matched:
            for k in range(kn):
                if bytes[i + k] != kb[k]:
                    matched = False
                    break
        if matched:
            i += kn
            while i < line_end and (bytes[i] == 32 or bytes[i] == 9):
                i += 1
            if i < line_end and bytes[i] == 58:  # ':'
                i += 1
                while i < line_end and (bytes[i] == 32 or bytes[i] == 9):
                    i += 1
                var whole = 0.0
                var frac = 0.0
                var frac_div = 1.0
                var digits = 0
                var in_frac = False
                while i < line_end:
                    var c = Int(bytes[i])
                    if c >= 48 and c <= 57:
                        if in_frac:
                            frac = frac * 10.0 + Float64(c - 48)
                            frac_div *= 10.0
                        else:
                            whole = whole * 10.0 + Float64(c - 48)
                        digits += 1
                        if digits > 9:
                            break
                    elif c == 46 and not in_frac:
                        in_frac = True
                    else:
                        break
                    i += 1
                var value = whole + frac / frac_div
                if digits > 0 and value > 0.0:
                    return Float32(value)
                return 0.0
        line_start = line_end + 1
    return 0.0


def x11_dpi_from_physical(pixels: Int, millimeters: Int) -> Float32:
    """DPI from the screen's size in pixels and millimeters (core setup / RandR); 0 when unknown."""
    if pixels <= 0 or millimeters <= 0:
        return 0.0
    return Float32(Float64(pixels) * 25.4 / Float64(millimeters))


def x11_scale_from_dpi(dpi: Float32) -> Float32:
    """Content scale for a DPI: dpi / 96 rounded to the nearest 0.25 and clamped to 1.0 .. 4.0
    (1.0 for an unknown or implausible DPI)."""
    if not (dpi > Float32(0.0)) or dpi > Float32(2000.0):
        return 1.0
    var steps = Int(dpi / Float32(96.0) * Float32(4.0) + Float32(0.5))
    var scale = Float32(steps) / Float32(4.0)
    return max(Float32(1.0), min(Float32(4.0), scale))


def parse_x11_display_number(display: String) -> String:
    """Extracts `<N>` from `:<N>`, `:<N>.<S>`, or `unix:<N>.<S>` (returns empty string if remote TCP or unset)."""
    if display.byte_length() == 0:
        return ""
    var bytes = display.as_bytes()
    var colon = -1
    for i in range(len(bytes)):
        if bytes[i] == 58:  # ':'
            colon = i
            break
    if colon < 0:
        return ""
    if colon > 0:
        var host = String(display[byte=0:colon])
        if host != "unix" and host != "localhost":
            return ""
    var num_bytes = List[UInt8]()
    for i in range(colon + 1, len(bytes)):
        var b = bytes[i]
        if b == 46:  # '.'
            break
        if b < 48 or b > 57:
            return ""
        num_bytes.append(b)
    if len(num_bytes) == 0:
        return ""
    return String(unsafe_from_utf8=num_bytes)


def read_xauthority_cookie(display_num: String) -> Tuple[List[UInt8], List[UInt8]]:
    """Reads `MIT-MAGIC-COOKIE-1` from `$XAUTHORITY` or `~/.Xauthority` for local display `display_num`."""
    var auth_path = env_now("XAUTHORITY")
    if auth_path.byte_length() == 0:
        auth_path = env_value("XAUTHORITY")
    if auth_path.byte_length() == 0:
        var home = env_value("HOME")
        if home.byte_length() > 0:
            auth_path = home + "/.Xauthority"
    var empty_name = List[UInt8]()
    var empty_data = List[UInt8]()
    if auth_path.byte_length() == 0:
        return (empty_name^, empty_data^)

    var raw = read_bytes(auth_path)
    if len(raw) < 12:
        return (empty_name^, empty_data^)

    var best_name = List[UInt8]()
    var best_data = List[UInt8]()
    var best_score = 0
    var pos = 0
    while pos + 4 <= len(raw):
        var family = x11_be16(raw, pos)
        var addr_len = x11_be16(raw, pos + 2)
        pos += 4
        if pos + addr_len + 2 > len(raw):
            break
        pos += addr_len

        var num_len = x11_be16(raw, pos)
        pos += 2
        if pos + num_len + 2 > len(raw):
            break
        var rec_num_bytes = List[UInt8]()
        for i in range(num_len):
            rec_num_bytes.append(raw[pos + i])
        var rec_num = String(unsafe_from_utf8=rec_num_bytes)
        pos += num_len

        var name_len = x11_be16(raw, pos)
        pos += 2
        if pos + name_len + 2 > len(raw):
            break
        var rec_name = List[UInt8]()
        for i in range(name_len):
            rec_name.append(raw[pos + i])
        pos += name_len

        var data_len = x11_be16(raw, pos)
        pos += 2
        if pos + data_len > len(raw):
            break
        var rec_data = List[UInt8]()
        for i in range(data_len):
            rec_data.append(raw[pos + i])
        pos += data_len

        if num_len > 0 and rec_num != display_num:
            continue

        var score = 1
        if rec_num == display_num:
            score += 4
        if family == 256 or family == 65535:  # FamilyLocal or FamilyWild
            score += 2
        if score > best_score and len(rec_name) > 0 and len(rec_data) > 0:
            best_score = score
            best_name = rec_name^
            best_data = rec_data^

    return (best_name^, best_data^)


struct X11Window(WindowBackend):
    var fd: Int
    var width: Int
    var height: Int
    var logical_width: Int
    var logical_height: Int
    var close_requested: Bool
    var focused: Bool
    var fullscreen: Bool
    var pointer_locked: Bool
    var rid_base: Int
    var rid_mask: Int
    var next_rid: Int
    var req_seq: Int
    var root_window: Int
    var root_visual: Int
    var root_depth: Int
    var black_pixel: Int
    var screen_width: Int
    var screen_height: Int
    var max_req_words: Int
    var window_id: Int
    var gc_id: Int
    var wm_protocols: Int
    var wm_delete_window: Int
    var net_wm_state: Int
    var net_wm_state_fullscreen: Int
    var incoming: List[UInt8]
    var band_buf: List[UInt8]
    var scaled_buf: List[UInt32]
    var x_map: List[Int]
    var x_map_src: Int
    var x_map_dst: Int
    var last_present_at: Float64
    var uncapped: Bool
    var last_error: String
    var pending_mouse_x: Float32
    var pending_mouse_y: Float32
    var has_pending_mouse: Bool
    var last_mouse_x: Float32
    var last_mouse_y: Float32
    var has_last_mouse: Bool
    var pending_rel_x: Float32
    var pending_rel_y: Float32
    var pending_buttons: List[Int]
    var pending_button_down: List[Bool]
    var pending_wheel: Float32
    var pending_keys: List[Int]
    var pending_key_down: List[Bool]
    # MIT-SHM extension state
    var shm_available: Bool
    var shm_disabled: Bool
    var shm_opcode: Int
    var shm_first_event: Int
    var shm_id: Int
    var shm_addr: Int
    var shm_size: Int
    var shm_seg: Int
    var shm_w: Int
    var shm_h: Int
    var shm_slot: Int
    var shm_busy0: Bool
    var shm_busy1: Bool
    # XInput2 XI_RawMotion + blank cursor
    var xi_opcode: Int
    var xi_raw_active: Bool
    var blank_pixmap: Int
    var blank_cursor: Int
    # Keyboard mapping + UTF-8 text_input_utf8
    var min_keycode: Int
    var max_keycode: Int
    var keysyms_per_keycode: Int
    var keymap: List[Int]
    var pending_text_utf8: String
    # CLIPBOARD selection
    var atom_clipboard: Int
    var atom_utf8_string: Int
    var atom_targets: Int
    var atom_flint_clip: Int
    var atom_incr: Int
    var clipboard_text: String
    var clipboard_owned: Bool
    var selection_notify_arrived: Bool
    var selection_notify_prop: Int
    # Vulkan XCB + EvdevHub
    var xcb_lib: Int
    var xcb_conn: Int
    var xcb_disconnect: Int
    var gpu_swapchain: Int
    var evdev: EvdevHub
    # Pointer enter (1) / leave (2) since the last poll, 0 for neither.
    var pointer_presence: Int
    # The core keyboard mapping changed: rebuild the input's layout shortcut table.
    var layout_dirty: Bool
    # HiDPI: content scale from `Xft.dpi` (else the screen's physical size), and the root
    # RESOURCE_MANAGER changed since it was read (the desktop changed its scale).
    var screen_width_mm: Int
    var screen_height_mm: Int
    var dpi: Float32
    var dpi_scale: Float32
    var dpi_source: String
    var resources_dirty: Bool

    def __init__(out self):
        self.fd = -1
        self.width = 1280
        self.height = 720
        self.logical_width = 1280
        self.logical_height = 720
        self.close_requested = False
        self.focused = True
        self.fullscreen = False
        self.pointer_locked = False
        self.rid_base = 0
        self.rid_mask = 0
        self.next_rid = 1
        self.req_seq = 0
        self.root_window = 0
        self.root_visual = 0
        self.root_depth = 24
        self.black_pixel = 0
        self.screen_width = 1920
        self.screen_height = 1080
        self.max_req_words = 65535
        self.window_id = 0
        self.gc_id = 0
        self.wm_protocols = 0
        self.wm_delete_window = 0
        self.net_wm_state = 0
        self.net_wm_state_fullscreen = 0
        self.incoming = List[UInt8]()
        self.band_buf = List[UInt8]()
        self.scaled_buf = List[UInt32]()
        self.x_map = List[Int]()
        self.x_map_src = 0
        self.x_map_dst = 0
        self.last_present_at = 0.0
        self.uncapped = False
        self.last_error = ""
        self.pending_mouse_x = 0
        self.pending_mouse_y = 0
        self.has_pending_mouse = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.has_last_mouse = False
        self.pending_rel_x = 0
        self.pending_rel_y = 0
        self.pending_buttons = List[Int]()
        self.pending_button_down = List[Bool]()
        self.pending_wheel = 0
        self.pending_keys = List[Int]()
        self.pending_key_down = List[Bool]()
        self.shm_available = False
        self.shm_disabled = False
        self.shm_opcode = 0
        self.shm_first_event = 0
        self.shm_id = -1
        self.shm_addr = 0
        self.shm_size = 0
        self.shm_seg = 0
        self.shm_w = 0
        self.shm_h = 0
        self.shm_slot = 0
        self.shm_busy0 = False
        self.shm_busy1 = False
        self.xi_opcode = 0
        self.xi_raw_active = False
        self.blank_pixmap = 0
        self.blank_cursor = 0
        self.min_keycode = 8
        self.max_keycode = 255
        self.keysyms_per_keycode = 0
        self.keymap = List[Int]()
        self.pending_text_utf8 = ""
        self.atom_clipboard = 0
        self.atom_utf8_string = 0
        self.atom_targets = 0
        self.atom_flint_clip = 0
        self.atom_incr = 0
        self.clipboard_text = ""
        self.clipboard_owned = False
        self.selection_notify_arrived = False
        self.selection_notify_prop = 0
        self.xcb_lib = 0
        self.xcb_conn = 0
        self.xcb_disconnect = 0
        self.gpu_swapchain = 0
        self.evdev = EvdevHub(enable_hotplug=False)
        self.pointer_presence = 0
        self.layout_dirty = False
        self.screen_width_mm = 0
        self.screen_height_mm = 0
        self.dpi = 0.0
        self.dpi_scale = 1.0
        self.dpi_source = String("default")
        self.resources_dirty = False

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.width = existing.width
        self.height = existing.height
        self.logical_width = existing.logical_width
        self.logical_height = existing.logical_height
        self.close_requested = existing.close_requested
        self.focused = existing.focused
        self.fullscreen = existing.fullscreen
        self.pointer_locked = existing.pointer_locked
        self.rid_base = existing.rid_base
        self.rid_mask = existing.rid_mask
        self.next_rid = existing.next_rid
        self.req_seq = existing.req_seq
        self.root_window = existing.root_window
        self.root_visual = existing.root_visual
        self.root_depth = existing.root_depth
        self.black_pixel = existing.black_pixel
        self.screen_width = existing.screen_width
        self.screen_height = existing.screen_height
        self.max_req_words = existing.max_req_words
        self.window_id = existing.window_id
        self.gc_id = existing.gc_id
        self.wm_protocols = existing.wm_protocols
        self.wm_delete_window = existing.wm_delete_window
        self.net_wm_state = existing.net_wm_state
        self.net_wm_state_fullscreen = existing.net_wm_state_fullscreen
        self.incoming = existing.incoming^
        self.band_buf = existing.band_buf^
        self.scaled_buf = existing.scaled_buf^
        self.x_map = existing.x_map^
        self.x_map_src = existing.x_map_src
        self.x_map_dst = existing.x_map_dst
        self.last_present_at = existing.last_present_at
        self.uncapped = existing.uncapped
        self.last_error = existing.last_error^
        self.pending_mouse_x = existing.pending_mouse_x
        self.pending_mouse_y = existing.pending_mouse_y
        self.has_pending_mouse = existing.has_pending_mouse
        self.last_mouse_x = existing.last_mouse_x
        self.last_mouse_y = existing.last_mouse_y
        self.has_last_mouse = existing.has_last_mouse
        self.pending_rel_x = existing.pending_rel_x
        self.pending_rel_y = existing.pending_rel_y
        self.pending_buttons = existing.pending_buttons^
        self.pending_button_down = existing.pending_button_down^
        self.pending_wheel = existing.pending_wheel
        self.pending_keys = existing.pending_keys^
        self.pending_key_down = existing.pending_key_down^
        self.shm_available = existing.shm_available
        self.shm_disabled = existing.shm_disabled
        self.shm_opcode = existing.shm_opcode
        self.shm_first_event = existing.shm_first_event
        self.shm_id = existing.shm_id
        self.shm_addr = existing.shm_addr
        self.shm_size = existing.shm_size
        self.shm_seg = existing.shm_seg
        self.shm_w = existing.shm_w
        self.shm_h = existing.shm_h
        self.shm_slot = existing.shm_slot
        self.shm_busy0 = existing.shm_busy0
        self.shm_busy1 = existing.shm_busy1
        self.xi_opcode = existing.xi_opcode
        self.xi_raw_active = existing.xi_raw_active
        self.blank_pixmap = existing.blank_pixmap
        self.blank_cursor = existing.blank_cursor
        self.min_keycode = existing.min_keycode
        self.max_keycode = existing.max_keycode
        self.keysyms_per_keycode = existing.keysyms_per_keycode
        self.keymap = existing.keymap^
        self.pending_text_utf8 = existing.pending_text_utf8^
        self.atom_clipboard = existing.atom_clipboard
        self.atom_utf8_string = existing.atom_utf8_string
        self.atom_targets = existing.atom_targets
        self.atom_flint_clip = existing.atom_flint_clip
        self.atom_incr = existing.atom_incr
        self.clipboard_text = existing.clipboard_text^
        self.clipboard_owned = existing.clipboard_owned
        self.selection_notify_arrived = existing.selection_notify_arrived
        self.selection_notify_prop = existing.selection_notify_prop
        self.xcb_lib = existing.xcb_lib
        self.xcb_conn = existing.xcb_conn
        self.xcb_disconnect = existing.xcb_disconnect
        self.gpu_swapchain = existing.gpu_swapchain
        self.evdev = existing.evdev^
        self.pointer_presence = existing.pointer_presence
        self.layout_dirty = existing.layout_dirty
        self.screen_width_mm = existing.screen_width_mm
        self.screen_height_mm = existing.screen_height_mm
        self.dpi = existing.dpi
        self.dpi_scale = existing.dpi_scale
        self.dpi_source = existing.dpi_source^
        self.resources_dirty = existing.resources_dirty
        existing.fd = -1
        existing.window_id = 0
        existing.gc_id = 0
        existing.shm_id = -1
        existing.shm_addr = 0
        existing.shm_size = 0
        existing.shm_seg = 0
        existing.blank_pixmap = 0
        existing.blank_cursor = 0
        existing.xcb_lib = 0
        existing.xcb_conn = 0
        existing.xcb_disconnect = 0
        existing.gpu_swapchain = 0

    def __deinit__(deinit self):
        self.close()

    def active(self) -> Bool:
        return self.fd >= 0 and self.window_id != 0

    def allocate_xid(mut self) -> Int:
        var step = self.rid_mask & (-self.rid_mask)
        if step <= 0:
            step = 1
        var xid = self.rid_base | ((self.next_rid * step) & self.rid_mask)
        self.next_rid += 1
        return xid

    def wait_io(self, events: Int, timeout_ms: Int) -> Bool:
        var descriptor = List[UInt8](length=8, fill=0)
        x11_put32(descriptor, 0, self.fd)
        x11_put16(descriptor, 4, events)
        return external_call["poll", Int32](descriptor.unsafe_ptr(), UInt64(1), Int32(timeout_ms)) > 0

    def write_ptr(mut self, base_addr: Int, length: Int) -> Bool:
        if self.fd < 0:
            return False
        var offset = 0
        while offset < length:
            var ptr = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=base_addr + offset)
            var count = Int(
                external_call["send", Int64](
                    Int32(self.fd),
                    ptr,
                    Int64(length - offset),
                    Int32(0x4000),  # MSG_NOSIGNAL
                )
            )
            if count > 0:
                offset += count
                continue
            if count < 0:
                var err_ptr = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
                var err = Int(err_ptr[unsafe_offset=0])
                if err == 4:  # EINTR
                    continue
                if err == 11:  # EAGAIN / EWOULDBLOCK
                    _ = self.wait_io(4, 50)
                    continue
            self.close_requested = True
            return False
        return True

    def write_all(mut self, data: List[UInt8], length: Int) -> Bool:
        return self.write_ptr(Int(data.unsafe_ptr()), length)

    def _send_req(mut self, data: List[UInt8]) -> Int:
        if not self.write_all(data, len(data)):
            return -1
        self.req_seq = (self.req_seq + 1) & 0xFFFF
        return self.req_seq

    def _send_req_ptr(mut self, base_addr: Int, length: Int) -> Int:
        if not self.write_ptr(base_addr, length):
            return -1
        self.req_seq = (self.req_seq + 1) & 0xFFFF
        return self.req_seq

    def recv_exact(mut self, count: Int) -> List[UInt8]:
        if count <= 0:
            return List[UInt8]()
        var retries = 0
        var tmp = List[UInt8]()
        while len(self.incoming) < count:
            if self.fd < 0:
                return List[UInt8]()
            if not self.wait_io(1, 2000):
                retries += 1
                if retries > 2:
                    return List[UInt8]()
                continue
            var need = min(count - len(self.incoming), 65536)
            if len(tmp) < need:
                tmp.resize(need, 0)
            var n = Int(
                external_call["recv", Int64](
                    Int32(self.fd),
                    tmp.unsafe_ptr(),
                    Int64(need),
                    Int32(0),
                )
            )
            if n > 0:
                retries = 0
                for i in range(n):
                    self.incoming.append(tmp[i])
                continue
            if n < 0:
                var err_ptr = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
                var err = Int(err_ptr[unsafe_offset=0])
                if err == 4 or err == 11:
                    continue
            if n == 0:
                self.close_requested = True
            return List[UInt8]()
        var out = List[UInt8](capacity=count)
        for i in range(count):
            out.append(self.incoming[i])
        var rem = len(self.incoming) - count
        for i in range(rem):
            self.incoming[i] = self.incoming[count + i]
        self.incoming.resize(rem, 0)
        return out^

    def _wait_reply(
        mut self, expected_seq: Int, check_prior_seq: Int = -1
    ) -> Tuple[Bool, Int, List[UInt8], List[UInt8]]:
        """Reads X11 packets until `expected_seq` reply/error arrives, preserving `GenericEvent` (35) framing.
        Returns `(ok, error_code_or_prior_err, pkt32, extra_bytes)`. If `check_prior_seq >= 0` and an
        error arrives for `check_prior_seq` before `expected_seq`, `ok` becomes False with `error_code`."""
        var target = expected_seq & 0xFFFF
        var prior = check_prior_seq & 0xFFFF if check_prior_seq >= 0 else -1
        var prior_err = 0
        var deferred = List[UInt8]()
        var res_ok = False
        var res_err = -1
        var res_pkt = List[UInt8]()
        var res_extra = List[UInt8]()
        for _ in range(256):
            var pkt = self.recv_exact(32)
            if len(pkt) < 32:
                break
            var kind = Int(pkt[0])
            if kind == 1:  # Reply
                var extra_words = x11_get32(pkt, 4)
                var extra = self.recv_exact(extra_words * 4) if extra_words > 0 else List[UInt8]()
                if extra_words > 0 and len(extra) < extra_words * 4:
                    break
                var seq = x11_get16(pkt, 2)
                if seq == target:
                    res_ok = prior_err == 0
                    res_err = prior_err if prior_err != 0 else 0
                    res_pkt = pkt^
                    res_extra = extra^
                    break
                continue
            if kind == 0:  # Error
                var seq = x11_get16(pkt, 2)
                var err = Int(pkt[1])
                if prior >= 0 and seq == prior:
                    prior_err = err if err != 0 else 1
                    continue
                if seq == target:
                    res_ok = False
                    res_err = err if err != 0 else 1
                    res_pkt = pkt^
                    break
                continue
            if (kind & 0x7F) == 35:  # GenericEvent (32 + extra_words*4)
                var extra_words = x11_get32(pkt, 4)
                for b in pkt:
                    deferred.append(b)
                if extra_words > 0:
                    var extra = self.recv_exact(extra_words * 4)
                    if len(extra) < extra_words * 4:
                        break
                    for b in extra:
                        deferred.append(b)
                continue
            for b in pkt:
                deferred.append(b)
        if len(deferred) > 0:
            for b in self.incoming:
                deferred.append(b)
            self.incoming = deferred^
        return (res_ok, res_err, res_pkt^, res_extra^)

    def intern_atom(mut self, name: String) -> Int:
        var n = name.byte_length()
        var padded = (n + 3) & ~3
        var req = List[UInt8](length=8 + padded, fill=0)
        req[0] = 16  # InternAtom
        req[1] = 0   # only-if-exists = False
        x11_put16(req, 2, (8 + padded) // 4)
        x11_put16(req, 4, n)
        for i in range(n):
            req[8 + i] = name.as_bytes()[i]
        var seq = self._send_req(req)
        if seq < 0:
            return 0
        var rep = self._wait_reply(seq)
        if not rep[0] or len(rep[2]) < 12:
            return 0
        return x11_get32(rep[2], 8)

    def _query_extension(mut self, name: String) -> Tuple[Bool, Int, Int, Int]:
        var n = name.byte_length()
        var padded = (n + 3) & ~3
        var req = List[UInt8](length=8 + padded, fill=0)
        req[0] = 98  # QueryExtension
        x11_put16(req, 2, (8 + padded) // 4)
        x11_put16(req, 4, n)
        for i in range(n):
            req[8 + i] = name.as_bytes()[i]
        var seq = self._send_req(req)
        if seq < 0:
            return (False, 0, 0, 0)
        var rep = self._wait_reply(seq)
        if not rep[0] or len(rep[2]) < 12:
            return (False, 0, 0, 0)
        ref pkt = rep[2]
        return (pkt[8] != 0, Int(pkt[9]), Int(pkt[10]), Int(pkt[11]))

    def refresh_keyboard_mapping(mut self):
        if self.fd < 0 or self.max_keycode < self.min_keycode:
            return
        var count = self.max_keycode - self.min_keycode + 1
        var req = List[UInt8](length=8, fill=0)
        req[0] = 101  # GetKeyboardMapping
        x11_put16(req, 2, 2)
        req[4] = UInt8(self.min_keycode & 0xFF)
        req[5] = UInt8(count & 0xFF)
        var seq = self._send_req(req)
        if seq < 0:
            return
        var rep = self._wait_reply(seq)
        if not rep[0] or len(rep[2]) < 32:
            return
        var per_kc = Int(rep[2][1])
        ref extra = rep[3]
        if per_kc <= 0 or len(extra) < count * per_kc * 4:
            return
        self.keysyms_per_keycode = per_kc
        self.keymap.clear()
        self.keymap.reserve(count * per_kc)
        for i in range(count * per_kc):
            self.keymap.append(x11_get32(extra, i * 4))
        self.layout_dirty = True

    def layout_key_map(self) -> List[Int]:
        """For `InputState.set_layout_key_map`: each physical GLFW code -> the code its unshifted
        group-1 keysym names on the server's keymap (`layout_shortcut_code`)."""
        var out = List[Int](capacity=512)
        for code in range(512):
            out.append(code)
        if self.keysyms_per_keycode <= 0:
            return out^
        for keycode in range(self.min_keycode, self.max_keycode + 1):
            var physical = x11_evdev_key(keycode - 8)
            if physical < 0 or physical >= 512:
                continue
            var base = (keycode - self.min_keycode) * self.keysyms_per_keycode
            if base >= len(self.keymap):
                break
            out[physical] = layout_shortcut_code(physical, x11_keysym_to_codepoint(self.keymap[base]))
        return out^

    def _append_key_utf8(mut self, detail: Int, state: Int):
        # Skip UTF-8 text generation when Ctrl (4) or Alt/Mod1 (8) is active
        if (state & 0x0C) != 0:
            return
        if self.keysyms_per_keycode <= 0 or detail < self.min_keycode or detail > self.max_keycode:
            return
        var idx = detail - self.min_keycode
        var base = idx * self.keysyms_per_keycode
        if base + self.keysyms_per_keycode > len(self.keymap):
            return
        var group = (state >> 13) & 3
        var g_off = group * 2
        if g_off >= self.keysyms_per_keycode:
            g_off = 0
        var ks0 = self.keymap[base + g_off]
        var ks1 = self.keymap[base + g_off + 1] if (g_off + 1 < self.keysyms_per_keycode) else 0
        if ks0 == 0 and g_off != 0:
            ks0 = self.keymap[base]
            ks1 = self.keymap[base + 1] if self.keysyms_per_keycode >= 2 else 0
        # AltGr (ISO_Level3_Shift, Mod5): XKB's core map keeps levels 3/4 in columns 5/6 (group 2: 7/8).
        if (state & 0x80) != 0:
            var l3 = 4 + (2 if group == 1 else 0)
            if l3 < self.keysyms_per_keycode and self.keymap[base + l3] != 0:
                ks0 = self.keymap[base + l3]
                ks1 = self.keymap[base + l3 + 1] if (l3 + 1 < self.keysyms_per_keycode) else 0
        if ks1 == 0:
            if (ks0 >= 97 and ks0 <= 122) or (ks0 >= 0xE0 and ks0 <= 0xF6) or (ks0 >= 0xF8 and ks0 <= 0xFE):
                ks1 = ks0 - 32
            elif ks0 >= 0x01000000:
                var cp0 = x11_keysym_to_codepoint(ks0)
                if (cp0 >= 97 and cp0 <= 122) or (cp0 >= 0xE0 and cp0 <= 0xF6) or (cp0 >= 0xF8 and cp0 <= 0xFE):
                    ks1 = 0x01000000 | (cp0 - 32)
        var shift = (state & 1) != 0
        var caps = (state & 2) != 0
        var numlock = (state & 16) != 0
        var cp0 = x11_keysym_to_codepoint(ks0)
        var level: Int
        if (ks0 >= 0xFF80 and ks0 <= 0xFFBD) or (ks1 >= 0xFF80 and ks1 <= 0xFFBD):
            level = 1 if (numlock != shift) else 0
        elif ks0 != ks1 and (
            (cp0 >= 97 and cp0 <= 122)
            or (cp0 >= 65 and cp0 <= 90)
            or (cp0 >= 0xC0 and cp0 <= 0xFF and cp0 != 0xD7 and cp0 != 0xF7)
        ):
            level = 1 if (shift != caps) else 0
        else:
            level = 1 if shift else 0
        var ks = ks1 if (level == 1 and ks1 != 0) else ks0
        var cp = x11_keysym_to_codepoint(ks)
        if cp > 0:
            var bytes = List[UInt8](capacity=4)
            encode_codepoint(bytes, cp)
            self.pending_text_utf8 += String(unsafe_from_utf8=bytes)

    def set_title(mut self, title: String):
        if not self.active():
            return
        var n = title.byte_length()
        var padded = (n + 3) & ~3
        var req = List[UInt8](length=24 + padded, fill=0)
        req[0] = 18  # ChangeProperty
        req[1] = 0   # PropModeReplace
        x11_put16(req, 2, (24 + padded) // 4)
        x11_put32(req, 4, self.window_id)
        x11_put32(req, 8, 39)  # XA_WM_NAME
        x11_put32(req, 12, 31)  # XA_STRING
        req[16] = 8
        x11_put32(req, 20, n)
        for i in range(n):
            req[24 + i] = title.as_bytes()[i]
        _ = self._send_req(req)

    def initialize(mut self, width: Int, height: Int) -> Bool:
        return self.initialize(width, height, "FLINT ENGINE")

    def initialize(mut self, width: Int, height: Int, title: String) -> Bool:
        if env_flag("FLINT_HEADLESS"):
            return False
        var disp = env_now("DISPLAY")
        if disp.byte_length() == 0:
            disp = env_value("DISPLAY")
        var disp_num = parse_x11_display_number(disp)
        if disp_num.byte_length() == 0:
            self.last_error = "DISPLAY unset or non-local"
            return False

        var sock_path = "/tmp/.X11-unix/X" + disp_num
        if sock_path.byte_length() >= 106:
            return False

        _ = external_call["sigignore", Int32](Int32(13))  # SIGPIPE
        self.fd = Int(external_call["socket", Int32](Int32(1), Int32(1), Int32(0)))
        if self.fd < 0:
            return False

        var address = List[UInt8](length=110, fill=0)
        address[0] = 1  # AF_UNIX
        for i in range(sock_path.byte_length()):
            address[2 + i] = sock_path.as_bytes()[i]
        if external_call["connect", Int32](Int32(self.fd), address.unsafe_ptr(), Int32(110)) != 0:
            self.close()
            self.last_error = "connect failed for " + sock_path
            return False

        self.width = width
        self.height = height
        self.logical_width = max(1, width)
        self.logical_height = max(1, height)
        self.uncapped = env_flag("FLINT_UNCAPPED")
        self.pending_mouse_x = 0
        self.pending_mouse_y = 0
        self.has_pending_mouse = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.has_last_mouse = False
        self.pending_rel_x = 0
        self.pending_rel_y = 0
        self.pending_buttons.clear()
        self.pending_button_down.clear()
        self.pending_wheel = 0
        self.pending_keys.clear()
        self.pending_key_down.clear()

        # 1. X11 Connection Setup Handshake (`'l'`, X11.0 + Xauthority MIT-MAGIC-COOKIE-1)
        var auth = read_xauthority_cookie(disp_num)
        ref auth_name = auth[0]
        ref auth_data = auth[1]
        var name_pad = (len(auth_name) + 3) & ~3
        var data_pad = (len(auth_data) + 3) & ~3
        var setup_req = List[UInt8](length=12 + name_pad + data_pad, fill=0)
        setup_req[0] = 0x6C  # 'l' little-endian
        x11_put16(setup_req, 2, 11)
        x11_put16(setup_req, 4, 0)
        x11_put16(setup_req, 6, len(auth_name))
        x11_put16(setup_req, 8, len(auth_data))
        for i in range(len(auth_name)):
            setup_req[12 + i] = auth_name[i]
        for i in range(len(auth_data)):
            setup_req[12 + name_pad + i] = auth_data[i]

        if not self.write_all(setup_req, len(setup_req)):
            self.close()
            return False

        var hdr = self.recv_exact(8)
        if len(hdr) < 8 or hdr[0] != 1:
            self.last_error = "X11 connection setup rejected"
            self.close()
            return False

        var add_bytes = x11_get16(hdr, 6) * 4
        var body = self.recv_exact(add_bytes)
        if len(body) < 32:
            self.close()
            return False

        # Offsets within `body` (which starts at byte 8 of the full setup reply):
        self.rid_base = x11_get32(body, 4)
        self.rid_mask = x11_get32(body, 8)
        self.req_seq = 0
        var vendor_len = x11_get16(body, 16)
        self.max_req_words = max(4096, x11_get16(body, 18))
        var num_screens = Int(body[20])
        var num_formats = Int(body[21])
        self.min_keycode = Int(body[26])
        self.max_keycode = Int(body[27])
        var vendor_pad = (vendor_len + 3) & ~3
        var screen_off = 32 + vendor_pad + num_formats * 8
        if num_screens < 1 or screen_off + 40 > len(body):
            self.last_error = "X11 screen descriptor truncated"
            self.close()
            return False

        self.root_window = x11_get32(body, screen_off + 0)
        self.black_pixel = x11_get32(body, screen_off + 12)
        self.screen_width = x11_get16(body, screen_off + 20)
        self.screen_height = x11_get16(body, screen_off + 22)
        self.screen_width_mm = x11_get16(body, screen_off + 24)
        self.screen_height_mm = x11_get16(body, screen_off + 26)
        self.root_visual = x11_get32(body, screen_off + 32)
        self.root_depth = Int(body[screen_off + 38])

        # 2. Create Window & Graphics Context
        self.window_id = self.allocate_xid()
        self.gc_id = self.allocate_xid()
        var win_x = max(0, (self.screen_width - width) // 2)
        var win_y = max(0, (self.screen_height - height) // 2)

        # CreateWindow (opcode 1): CWBackPixel (0x2) | CWEventMask (0x800)
        var cw = List[UInt8](length=40, fill=0)
        cw[0] = 1
        cw[1] = UInt8(self.root_depth)
        x11_put16(cw, 2, 10)
        x11_put32(cw, 4, self.window_id)
        x11_put32(cw, 8, self.root_window)
        x11_put16(cw, 12, win_x)
        x11_put16(cw, 14, win_y)
        x11_put16(cw, 16, width)
        x11_put16(cw, 18, height)
        x11_put16(cw, 20, 0)
        x11_put16(cw, 22, 1)  # InputOutput
        x11_put32(cw, 24, self.root_visual)
        x11_put32(cw, 28, 0x00000802)  # CWBackPixel | CWEventMask
        x11_put32(cw, 32, self.black_pixel)
        # KeyPress|KeyRelease|ButtonPress|ButtonRelease|EnterWindow|LeaveWindow|PointerMotion|Exposure|
        # StructureNotify|FocusChange
        x11_put32(cw, 36, 0x0022807F)
        if self._send_req(cw) < 0:
            self.close()
            return False

        # CreateGC (opcode 55): CWGraphicsExposures (0x10000) = 0
        var gc = List[UInt8](length=20, fill=0)
        gc[0] = 55
        x11_put16(gc, 2, 5)
        x11_put32(gc, 4, self.gc_id)
        x11_put32(gc, 8, self.window_id)
        x11_put32(gc, 12, 0x00010000)
        x11_put32(gc, 16, 0)
        if self._send_req(gc) < 0:
            self.close()
            return False

        # Create 1x1 blank pixmap + cursor for pointer lock
        self.blank_pixmap = self.allocate_xid()
        var pm = List[UInt8](length=16, fill=0)
        pm[0] = 53  # CreatePixmap
        pm[1] = 1   # depth 1
        x11_put16(pm, 2, 4)
        x11_put32(pm, 4, self.blank_pixmap)
        x11_put32(pm, 8, self.window_id)
        x11_put16(pm, 12, 1)
        x11_put16(pm, 14, 1)
        if self._send_req(pm) >= 0:
            self.blank_cursor = self.allocate_xid()
            var cr = List[UInt8](length=32, fill=0)
            cr[0] = 93  # CreateCursor
            x11_put16(cr, 2, 8)
            x11_put32(cr, 4, self.blank_cursor)
            x11_put32(cr, 8, self.blank_pixmap)
            x11_put32(cr, 12, self.blank_pixmap)
            _ = self._send_req(cr)

        # 3. Intern Window Manager & Clipboard Atoms
        self.wm_protocols = self.intern_atom("WM_PROTOCOLS")
        self.wm_delete_window = self.intern_atom("WM_DELETE_WINDOW")
        self.net_wm_state = self.intern_atom("_NET_WM_STATE")
        self.net_wm_state_fullscreen = self.intern_atom("_NET_WM_STATE_FULLSCREEN")
        self.atom_clipboard = self.intern_atom("CLIPBOARD")
        self.atom_utf8_string = self.intern_atom("UTF8_STRING")
        self.atom_targets = self.intern_atom("TARGETS")
        self.atom_flint_clip = self.intern_atom("FLINT_CLIPBOARD")
        self.atom_incr = self.intern_atom("INCR")

        # 3b. Query MIT-SHM & XInput2 extensions + keyboard mapping
        var shm_ext = self._query_extension("MIT-SHM")
        if shm_ext[0]:
            var shm_ver_req = List[UInt8](length=4, fill=0)
            shm_ver_req[0] = UInt8(shm_ext[1] & 0xFF)
            shm_ver_req[1] = 0  # X_ShmQueryVersion
            x11_put16(shm_ver_req, 2, 1)
            var s_seq = self._send_req(shm_ver_req)
            if s_seq >= 0:
                var s_rep = self._wait_reply(s_seq)
                if s_rep[0]:
                    self.shm_available = True
                    self.shm_opcode = shm_ext[1]
                    self.shm_first_event = shm_ext[2]

        var xi_ext = self._query_extension("XInputExtension")
        if xi_ext[0]:
            var xi_op = xi_ext[1]
            var xi_ver_req = List[UInt8](length=8, fill=0)
            xi_ver_req[0] = UInt8(xi_op & 0xFF)
            xi_ver_req[1] = 47  # X_XIQueryVersion
            x11_put16(xi_ver_req, 2, 2)
            x11_put16(xi_ver_req, 4, 2)
            x11_put16(xi_ver_req, 6, 0)
            var x_seq = self._send_req(xi_ver_req)
            if x_seq >= 0:
                var x_rep = self._wait_reply(x_seq)
                if x_rep[0] and len(x_rep[2]) >= 12 and x11_get16(x_rep[2], 8) >= 2:
                    var sel = List[UInt8](length=20, fill=0)
                    sel[0] = UInt8(xi_op & 0xFF)
                    sel[1] = 46  # X_XISelectEvents
                    x11_put16(sel, 2, 5)
                    x11_put32(sel, 4, self.root_window)
                    x11_put16(sel, 8, 1)  # num_masks = 1
                    x11_put16(sel, 12, 1)  # deviceid = XIAllMasterDevices
                    x11_put16(sel, 14, 1)  # mask_len = 1 (4-byte word)
                    x11_put32(sel, 16, 1 << 17)  # XI_RawMotionMask
                    if self._send_req(sel) >= 0:
                        self.xi_opcode = xi_op
                        self.xi_raw_active = True

        self.refresh_keyboard_mapping()
        self.set_title(title)
        # Follow the desktop's scale: PropertyNotify on the root for RESOURCE_MANAGER (Xft.dpi).
        var cwa = List[UInt8](length=12, fill=0)
        cwa[0] = 2  # ChangeWindowAttributes
        x11_put16(cwa, 2, 3)
        x11_put32(cwa, 4, self.root_window)
        x11_put32(cwa, 8, 0x800)  # CWEventMask
        var cwa_ev = List[UInt8](length=4, fill=0)
        x11_put32(cwa_ev, 0, X11_PROPERTY_CHANGE_MASK)
        for b in cwa_ev:
            cwa.append(b)
        _ = self._send_req(cwa)
        self.refresh_dpi()

        if self.wm_protocols != 0 and self.wm_delete_window != 0:
            var prop = List[UInt8](length=28, fill=0)
            prop[0] = 18  # ChangeProperty
            prop[1] = 0   # PropModeReplace
            x11_put16(prop, 2, 7)
            x11_put32(prop, 4, self.window_id)
            x11_put32(prop, 8, self.wm_protocols)
            x11_put32(prop, 12, 4)  # XA_ATOM
            prop[16] = 32
            x11_put32(prop, 20, 1)
            x11_put32(prop, 24, self.wm_delete_window)
            _ = self._send_req(prop)

        # 4. MapWindow (opcode 8)
        var mw = List[UInt8](length=8, fill=0)
        mw[0] = 8
        x11_put16(mw, 2, 2)
        x11_put32(mw, 4, self.window_id)
        if self._send_req(mw) < 0:
            self.close()
            return False

        _ = self.evdev.enable_hotplug()
        _ = self.evdev.discover_initial()
        self.last_present_at = get_time()
        return True

    def receive(mut self):
        if self.fd < 0:
            return
        var buffer = List[UInt8](length=8192, fill=0)
        while True:
            var count = Int(
                external_call["recv", Int64](
                    Int32(self.fd),
                    buffer.unsafe_ptr(),
                    Int64(len(buffer)),
                    Int32(64),  # MSG_DONTWAIT
                )
            )
            if count > 0:
                for i in range(count):
                    self.incoming.append(buffer[i])
                if count < len(buffer):
                    break
                continue
            if count == 0:
                self.close_requested = True
            break
        self.process_events()

    def _handle_selection_request(mut self, pos: Int):
        if pos + 28 > len(self.incoming):
            return
        var time = x11_get32(self.incoming, pos + 4)
        var requestor = x11_get32(self.incoming, pos + 12)
        var selection = x11_get32(self.incoming, pos + 16)
        var target = x11_get32(self.incoming, pos + 20)
        var prop = x11_get32(self.incoming, pos + 24)
        if prop == 0:
            prop = target
        var out_prop = 0
        if selection == self.atom_clipboard and self.clipboard_owned and self.fd >= 0:
            if target == self.atom_targets and self.atom_targets != 0:
                var req = List[UInt8](length=36, fill=0)
                req[0] = 18  # ChangeProperty
                req[1] = 0
                x11_put16(req, 2, 9)
                x11_put32(req, 4, requestor)
                x11_put32(req, 8, prop)
                x11_put32(req, 12, 4)  # XA_ATOM
                req[16] = 32
                x11_put32(req, 20, 3)
                x11_put32(req, 24, self.atom_targets)
                x11_put32(req, 28, self.atom_utf8_string if self.atom_utf8_string != 0 else 31)
                x11_put32(req, 32, 31)  # XA_STRING
                _ = self._send_req(req)
                out_prop = prop
            elif target == self.atom_utf8_string or target == 31:
                var bytes = self.clipboard_text.as_bytes()
                var n = len(bytes)
                var padded = (n + 3) & ~3
                var req = List[UInt8](length=24 + padded, fill=0)
                req[0] = 18  # ChangeProperty
                req[1] = 0
                x11_put16(req, 2, (24 + padded) // 4)
                x11_put32(req, 4, requestor)
                x11_put32(req, 8, prop)
                x11_put32(req, 12, target)
                req[16] = 8
                x11_put32(req, 20, n)
                for i in range(n):
                    req[24 + i] = bytes[i]
                _ = self._send_req(req)
                out_prop = prop
        if self.fd >= 0:
            var ev = List[UInt8](length=44, fill=0)
            ev[0] = 25  # SendEvent
            ev[1] = 0
            x11_put16(ev, 2, 11)
            x11_put32(ev, 4, requestor)
            x11_put32(ev, 8, 0)
            ev[12] = 31  # SelectionNotify
            x11_put32(ev, 16, time)
            x11_put32(ev, 20, requestor)
            x11_put32(ev, 24, selection)
            x11_put32(ev, 28, target)
            x11_put32(ev, 32, out_prop)
            _ = self._send_req(ev)

    def process_events(mut self):
        var pos = 0
        while pos + 32 <= len(self.incoming):
            var raw_type = Int(self.incoming[pos])
            if raw_type == 1:
                var extra = x11_get32(self.incoming, pos + 4) * 4
                if pos + 32 + extra > len(self.incoming):
                    break
                pos += 32 + extra
                continue
            if (raw_type & 0x7F) == 35:  # GenericEvent (XInput2 XI_RawMotion)
                var extra = x11_get32(self.incoming, pos + 4) * 4
                if pos + 32 + extra > len(self.incoming):
                    break
                var ext_id = Int(self.incoming[pos + 1])
                var evtype = x11_get16(self.incoming, pos + 8)
                if self.xi_opcode != 0 and ext_id == self.xi_opcode and evtype == 17:
                    if self.pointer_locked and self.focused:
                        var vlen = x11_get16(self.incoming, pos + 22)
                        var mask_bytes = vlen * 4
                        var mask_off = pos + 32
                        if mask_off + mask_bytes <= pos + 32 + extra and mask_bytes > 0:
                            var n_set = 0
                            var has_x = False
                            var has_y = False
                            var idx_x = 0
                            var idx_y = 0
                            for b in range(mask_bytes):
                                var byte_v = Int(self.incoming[mask_off + b])
                                for bit in range(8):
                                    if (byte_v & (1 << bit)) != 0:
                                        var v_idx = b * 8 + bit
                                        if v_idx == 0:
                                            has_x = True
                                            idx_x = n_set
                                        elif v_idx == 1:
                                            has_y = True
                                            idx_y = n_set
                                        n_set += 1
                            var vals_off = mask_off + mask_bytes
                            var raw_off = vals_off + n_set * 8
                            var base_off = raw_off if (raw_off + n_set * 8 <= pos + 32 + extra) else vals_off
                            if base_off + n_set * 8 <= pos + 32 + extra:
                                if has_x:
                                    self.pending_rel_x += x11_fp3232(self.incoming, base_off + idx_x * 8)
                                if has_y:
                                    self.pending_rel_y += x11_fp3232(self.incoming, base_off + idx_y * 8)
                pos += 32 + extra
                continue

            var code = raw_type & 0x7F
            var detail = Int(self.incoming[pos + 1])

            if self.shm_first_event != 0 and code == self.shm_first_event:
                var shm_off = x11_get32(self.incoming, pos + 16)
                if shm_off == 0:
                    self.shm_busy0 = False
                else:
                    self.shm_busy1 = False
            elif code == 2 or code == 3:
                # Server auto-repeat: KeyRelease immediately followed by KeyPress at the identical timestamp.
                # Keep it as one press of the held key, which `InputState` reports as a repeat
                # (`key_pressed_or_repeat`), with its text, at the user's `xset r rate`.
                if code == 3 and pos + 64 <= len(self.incoming):
                    var next_code = Int(self.incoming[pos + 32]) & 0x7F
                    var next_detail = Int(self.incoming[pos + 33])
                    if next_code == 2 and next_detail == detail:
                        if x11_get32(self.incoming, pos + 4) == x11_get32(self.incoming, pos + 36):
                            var held = x11_evdev_key(detail - 8)
                            if held >= 0:
                                self.pending_keys.append(held)
                                self.pending_key_down.append(True)
                            self._append_key_utf8(detail, x11_get16(self.incoming, pos + 32 + 28))
                            pos += 64
                            continue
                var key = x11_evdev_key(detail - 8)
                if key >= 0:
                    self.pending_keys.append(key)
                    self.pending_key_down.append(code == 2)
                if code == 2:
                    self._append_key_utf8(detail, x11_get16(self.incoming, pos + 28))
            elif code == 4 or code == 5:
                self.pending_mouse_x = Float32(x11_gets16(self.incoming, pos + 24))
                self.pending_mouse_y = Float32(x11_gets16(self.incoming, pos + 26))
                self.has_pending_mouse = True
                var down = code == 4
                if detail == 1:
                    self.pending_buttons.append(0)  # Left
                    self.pending_button_down.append(down)
                elif detail == 3:
                    self.pending_buttons.append(1)  # Right
                    self.pending_button_down.append(down)
                elif detail == 2:
                    self.pending_buttons.append(2)  # Middle
                    self.pending_button_down.append(down)
                elif detail == 4 and down:
                    self.pending_wheel += 1.0
                elif detail == 5 and down:
                    self.pending_wheel -= 1.0
            elif code == 6:  # MotionNotify
                var mx = Float32(x11_gets16(self.incoming, pos + 24))
                var my = Float32(x11_gets16(self.incoming, pos + 26))
                if self.pointer_locked:
                    if not self.xi_raw_active and self.has_last_mouse:
                        self.pending_rel_x += mx - self.last_mouse_x
                        self.pending_rel_y += my - self.last_mouse_y
                    self.last_mouse_x = mx
                    self.last_mouse_y = my
                    self.has_last_mouse = True
                else:
                    self.pending_mouse_x = mx
                    self.pending_mouse_y = my
                    self.has_pending_mouse = True
            elif code == 7 or code == 8:  # EnterNotify / LeaveNotify (event-x:24, event-y:26, state:28, mode:30)
                var mode = Int(self.incoming[pos + 30])
                var buttons_held = (x11_get16(self.incoming, pos + 28) & 0x1F00) != 0
                if mode == 0 and code == 7:
                    self.pending_mouse_x = Float32(x11_gets16(self.incoming, pos + 24))
                    self.pending_mouse_y = Float32(x11_gets16(self.incoming, pos + 26))
                    self.has_pending_mouse = True
                    self.pointer_presence = 1
                elif mode == 0 and code == 8 and not buttons_held and not self.pointer_locked:
                    # A drag that leaves the window keeps its implicit grab and its motion events.
                    self.has_pending_mouse = False
                    self.pointer_presence = 2
            elif code == 9:  # FocusIn
                self.focused = True
            elif code == 10:  # FocusOut
                self.focused = False
                self.pending_keys.append(X11_RELEASE_HELD)
                self.pending_key_down.append(False)
                self.pending_buttons.append(X11_RELEASE_HELD)
                self.pending_button_down.append(False)
            elif code == 22:  # ConfigureNotify (x:16, y:18, width:20, height:22)
                var cfg_win = x11_get32(self.incoming, pos + 8)
                var nw = x11_get16(self.incoming, pos + 20)
                var nh = x11_get16(self.incoming, pos + 22)
                if (cfg_win == 0 or cfg_win == self.window_id) and nw > 0 and nh > 0:
                    self.width = nw
                    self.height = nh
                    # Pointer positions follow the window, in the pixels `framebuffer_size()` reports
                    # (every renderer resizes to it), not the size the window was created at.
                    self.logical_width = nw
                    self.logical_height = nh
            elif code == 29:  # SelectionClear (offset 8 = window, offset 12 = selection atom)
                if x11_get32(self.incoming, pos + 12) == self.atom_clipboard:
                    self.clipboard_owned = False
            elif code == 30:  # SelectionRequest
                self._handle_selection_request(pos)
            elif code == 31:  # SelectionNotify
                self.selection_notify_arrived = True
                self.selection_notify_prop = x11_get32(self.incoming, pos + 20)
            elif code == 34:  # MappingNotify (offset 1 = pad1, offset 4 = request)
                if Int(self.incoming[pos + 4]) == 1:  # MappingKeyboard
                    pos += 32
                    var remaining = len(self.incoming) - pos
                    for i in range(remaining):
                        self.incoming[i] = self.incoming[pos + i]
                    self.incoming.resize(remaining, 0)
                    pos = 0
                    self.refresh_keyboard_mapping()
                    continue
            elif code == 33:  # ClientMessage
                if detail == 32 and self.wm_delete_window != 0:
                    if x11_get32(self.incoming, pos + 12) == self.wm_delete_window:
                        self.close_requested = True
            elif code == 17:  # DestroyNotify
                self.close_requested = True
            elif code == 28:  # PropertyNotify (window:4, atom:8)
                if x11_get32(self.incoming, pos + 4) == self.root_window and x11_get32(self.incoming, pos + 8) == X11_ATOM_RESOURCE_MANAGER:
                    self.resources_dirty = True

            pos += 32

        if pos > 0:
            var remaining = len(self.incoming) - pos
            for i in range(remaining):
                self.incoming[i] = self.incoming[pos + i]
            self.incoming.resize(remaining, 0)

    def map_mouse_x(self, raw_x: Float32) -> Float32:
        if self.width <= 0 or self.width == self.logical_width:
            return raw_x
        return raw_x * Float32(self.logical_width) / Float32(self.width)

    def map_mouse_y(self, raw_y: Float32) -> Float32:
        if self.height <= 0 or self.height == self.logical_height:
            return raw_y
        return raw_y * Float32(self.logical_height) / Float32(self.height)

    def scale_pixels_into_buffer(mut self, mut pixels: List[UInt32], source_width: Int, source_height: Int, dst_w: Int, dst_h: Int):
        if len(self.scaled_buf) != dst_w * dst_h:
            self.scaled_buf.resize(dst_w * dst_h, 0)
        var src_ptr = pixels.unsafe_ptr()
        var dst_ptr = self.scaled_buf.unsafe_ptr()
        if source_width == dst_w:
            var row_bytes = dst_w * 4
            for y in range(dst_h):
                var sy = min(source_height - 1, (y * source_height) // dst_h)
                _ = external_call["memcpy", Int](
                    Int(dst_ptr.unsafe_offset(y * dst_w)),
                    Int(src_ptr.unsafe_offset(sy * source_width)),
                    row_bytes,
                )
            return

        if self.x_map_src != source_width or self.x_map_dst != dst_w or len(self.x_map) != dst_w:
            self.x_map.resize(dst_w, 0)
            for x in range(dst_w):
                self.x_map[x] = min(source_width - 1, (x * source_width) // dst_w)
            self.x_map_src = source_width
            self.x_map_dst = dst_w

        var xmap_ptr = self.x_map.unsafe_ptr()
        for y in range(dst_h):
            var sy = min(source_height - 1, (y * source_height) // dst_h)
            var srow = sy * source_width
            var drow = y * dst_w
            for x in range(dst_w):
                dst_ptr[unsafe_offset=drow + x] = src_ptr[unsafe_offset=srow + xmap_ptr[unsafe_offset=x]] | UInt32(0xFF000000)

    def read_resource_manager(mut self) -> String:
        """The root window's RESOURCE_MANAGER string (X resources), "" when unset."""
        if self.fd < 0 or self.root_window == 0:
            return ""
        var gp = List[UInt8](length=24, fill=0)
        gp[0] = 20  # GetProperty
        gp[1] = 0   # delete = False
        x11_put16(gp, 2, 6)
        x11_put32(gp, 4, self.root_window)
        x11_put32(gp, 8, X11_ATOM_RESOURCE_MANAGER)
        x11_put32(gp, 12, 31)  # XA_STRING
        x11_put32(gp, 16, 0)
        x11_put32(gp, 20, 262144)  # long-length, in 4-byte units: 1 MiB is plenty
        var seq = self._send_req(gp)
        if seq < 0:
            return ""
        var rep = self._wait_reply(seq)
        if not rep[0] or len(rep[2]) < 24 or Int(rep[2][1]) != 8:
            return ""
        var nbytes = x11_get32(rep[2], 16)
        ref extra = rep[3]
        nbytes = min(nbytes, len(extra))
        if nbytes <= 0:
            return ""
        var out = List[UInt8](capacity=nbytes)
        for i in range(nbytes):
            out.append(extra[i])
        return String(unsafe_from_utf8=out)

    def apply_dpi(mut self, xft_dpi: Float32):
        """Picks the DPI (`Xft.dpi`, else the screen's physical size) and the content scale from it."""
        if xft_dpi > Float32(0.0):
            self.dpi = xft_dpi
            self.dpi_source = String("Xft.dpi")
        else:
            var phys = x11_dpi_from_physical(self.screen_width, self.screen_width_mm)
            # Servers without real EDID data report 96 dpi or nonsense; only trust a plausible size.
            if phys >= Float32(48.0) and phys <= Float32(600.0):
                self.dpi = phys
                self.dpi_source = String("physical")
            else:
                self.dpi = 96.0
                self.dpi_source = String("default")
        self.dpi_scale = x11_scale_from_dpi(self.dpi)

    def refresh_dpi(mut self):
        self.resources_dirty = False
        self.apply_dpi(parse_xft_dpi(self.read_resource_manager()))

    def poll(mut self, mut input: InputState):
        self.evdev.poll(input)
        if self.fd < 0:
            return
        self.receive()
        if self.resources_dirty:
            self.refresh_dpi()
        # The server repeats held keys itself; InputState must not synthesize a second stream.
        if input.repeat_rate_hz != 0:
            input.set_repeat_info(0, 0)
        if self.layout_dirty:
            self.layout_dirty = False
            input.set_layout_key_map(self.layout_key_map())
        if self.pointer_presence == 2 and not self.has_pending_mouse:
            input.pointer_left()
        if self.has_pending_mouse:
            var px = self.map_mouse_x(self.pending_mouse_x)
            var py = self.map_mouse_y(self.pending_mouse_y)
            if self.pointer_presence == 1:
                input.pointer_entered(px, py)
            else:
                input.move_mouse(px, py)
            self.has_pending_mouse = False
        self.pointer_presence = 0
        if self.pointer_locked and (self.pending_rel_x != 0.0 or self.pending_rel_y != 0.0):
            input.add_mouse_delta(self.pending_rel_x, self.pending_rel_y)
            self.pending_rel_x = 0
            self.pending_rel_y = 0
        for i in range(len(self.pending_buttons)):
            if self.pending_buttons[i] == X11_RELEASE_HELD:
                for b in range(X11_MOUSE_BUTTONS):
                    if input.mouse_down(b):
                        input.set_mouse(b, False)
            else:
                input.set_mouse(self.pending_buttons[i], self.pending_button_down[i])
        self.pending_buttons.clear()
        self.pending_button_down.clear()
        input.wheel += self.pending_wheel
        self.pending_wheel = 0
        for i in range(len(self.pending_keys)):
            if self.pending_keys[i] == X11_RELEASE_HELD:
                input.release_held()
            else:
                input.set_key(self.pending_keys[i], self.pending_key_down[i])
        self.pending_keys.clear()
        self.pending_key_down.clear()
        if self.pending_text_utf8.byte_length() > 0:
            input.push_text_utf8(self.pending_text_utf8)
            self.pending_text_utf8 = ""

    def _destroy_shm(mut self):
        if self.fd >= 0 and self.shm_seg != 0 and self.shm_opcode != 0:
            var det = List[UInt8](length=8, fill=0)
            det[0] = UInt8(self.shm_opcode & 0xFF)
            det[1] = 2  # X_ShmDetach
            x11_put16(det, 2, 2)
            x11_put32(det, 4, self.shm_seg)
            _ = self._send_req(det)
        self.shm_seg = 0
        if self.shm_addr != 0:
            _ = external_call["shmdt", Int32](self.shm_addr)
            self.shm_addr = 0
        if self.shm_id >= 0:
            _ = external_call["shmctl", Int32](Int32(self.shm_id), Int32(0), Int(0))  # IPC_RMID
            self.shm_id = -1
        self.shm_size = 0
        self.shm_w = 0
        self.shm_h = 0
        self.shm_slot = 0
        self.shm_busy0 = False
        self.shm_busy1 = False

    def _ensure_shm(mut self, dst_w: Int, dst_h: Int) -> Bool:
        if not self.shm_available or self.shm_disabled or self.shm_opcode == 0:
            return False
        if self.shm_seg != 0 and self.shm_w == dst_w and self.shm_h == dst_h:
            return True
        self._destroy_shm()
        var total = dst_w * dst_h * 4 * 2
        if total <= 0:
            return False
        var shmid = Int(external_call["shmget", Int32](Int32(0), Int64(total), Int32(0o1600)))
        if shmid < 0:
            self.shm_disabled = True
            return False
        var addr = external_call["shmat", Int](Int32(shmid), Int(0), Int32(0))
        if addr == -1 or addr == 0:
            _ = external_call["shmctl", Int32](Int32(shmid), Int32(0), Int(0))
            self.shm_disabled = True
            return False
        var seg = self.allocate_xid()
        var att = List[UInt8](length=16, fill=0)
        att[0] = UInt8(self.shm_opcode & 0xFF)
        att[1] = 1  # X_ShmAttach
        x11_put16(att, 2, 4)
        x11_put32(att, 4, seg)
        x11_put32(att, 8, shmid)
        att[12] = 1  # readOnly = True
        var att_seq = self._send_req(att)
        if att_seq < 0:
            _ = external_call["shmdt", Int32](addr)
            _ = external_call["shmctl", Int32](Int32(shmid), Int32(0), Int(0))
            self.shm_disabled = True
            return False
        var focus_req = List[UInt8](length=4, fill=0)
        focus_req[0] = 43  # GetInputFocus sync fence
        x11_put16(focus_req, 2, 1)
        var focus_seq = self._send_req(focus_req)
        var sync_ok = False
        if focus_seq >= 0:
            var rep = self._wait_reply(focus_seq, att_seq)
            sync_ok = rep[0]
        # Immediately mark IPC_RMID so kernel reclaims the SHM segment on process/server exit
        _ = external_call["shmctl", Int32](Int32(shmid), Int32(0), Int(0))
        if not sync_ok:
            _ = external_call["shmdt", Int32](addr)
            self.shm_disabled = True
            return False
        self.shm_id = -1
        self.shm_addr = addr
        self.shm_size = total
        self.shm_seg = seg
        self.shm_w = dst_w
        self.shm_h = dst_h
        self.shm_slot = 0
        self.shm_busy0 = False
        self.shm_busy1 = False
        return True

    def present(mut self, mut pixels: List[UInt32], source_width: Int, source_height: Int):
        if self.fd < 0 or self.window_id == 0 or source_width <= 0 or source_height <= 0:
            return
        self.receive()
        if self.close_requested:
            return

        var dst_w = max(1, self.width)
        var dst_h = max(1, self.height)
        var use_scaled = source_width != dst_w or source_height != dst_h
        var shm_ready = self._ensure_shm(dst_w, dst_h)
        if shm_ready and self.shm_busy0 and self.shm_busy1:
            # Both slots are still being read by the server: never overwrite one mid-read. Give the
            # ShmCompletion a moment, then drop this frame (the next one is fresher anyway).
            if self.wait_io(1, 4):
                self.receive()
            if self.shm_busy0 and self.shm_busy1:
                if get_time() - self.last_present_at < 0.25:
                    return
                # A completion that never came (server reset, lost event): do not stall forever.
                self.shm_busy0 = False
                self.shm_busy1 = False
        if use_scaled:
            self.scale_pixels_into_buffer(pixels, source_width, source_height, dst_w, dst_h)

        if shm_ready:
            var slot: Int
            if self.shm_busy0 and not self.shm_busy1:
                slot = 1
            elif self.shm_busy1 and not self.shm_busy0:
                slot = 0
            else:
                slot = self.shm_slot ^ 1
            self.shm_slot = slot
            var frame_px = dst_w * dst_h
            var frame_bytes = frame_px * 4
            var shm_off = slot * frame_bytes
            var src_ptr = self.scaled_buf.unsafe_ptr() if use_scaled else pixels.unsafe_ptr()
            var dst_ptr = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.shm_addr + shm_off)
            var alpha_v = SIMD[DType.uint32, 8](0xFF000000)
            var idx = 0
            while idx + 8 <= frame_px:
                var v = src_ptr.unsafe_load[width=8, alignment=4](idx)
                dst_ptr.unsafe_store[alignment=4](idx, v | alpha_v)
                idx += 8
            while idx < frame_px:
                dst_ptr[unsafe_offset=idx] = src_ptr[unsafe_offset=idx] | UInt32(0xFF000000)
                idx += 1

            if slot == 0:
                self.shm_busy0 = True
            else:
                self.shm_busy1 = True

            var put_req = List[UInt8](length=40, fill=0)
            put_req[0] = UInt8(self.shm_opcode & 0xFF)
            put_req[1] = 3  # X_ShmPutImage
            x11_put16(put_req, 2, 10)
            x11_put32(put_req, 4, self.window_id)
            x11_put32(put_req, 8, self.gc_id)
            x11_put16(put_req, 12, dst_w)
            x11_put16(put_req, 14, dst_h)
            x11_put16(put_req, 16, 0)
            x11_put16(put_req, 18, 0)
            x11_put16(put_req, 20, dst_w)
            x11_put16(put_req, 22, dst_h)
            x11_put16(put_req, 24, 0)
            x11_put16(put_req, 26, 0)
            put_req[28] = UInt8(self.root_depth)
            put_req[29] = 2  # ZPixmap
            put_req[30] = 1  # sendEvent = True
            x11_put32(put_req, 32, self.shm_seg)
            x11_put32(put_req, 36, shm_off)
            if self._send_req(put_req) < 0:
                return
        else:
            var max_band_words = min(self.max_req_words - 8, 32768)
            var rows_per_band = max(1, max_band_words // dst_w)
            var needed_bytes = 24 + dst_w * rows_per_band * 4
            if len(self.band_buf) < needed_bytes:
                self.band_buf.resize(needed_bytes, 0)

            var y = 0
            while y < dst_h:
                var bh = min(rows_per_band, dst_h - y)
                var byte_pixels = dst_w * bh * 4
                var pkt_len = 24 + byte_pixels
                self.band_buf[0] = 72  # PutImage
                self.band_buf[1] = 2   # ZPixmap
                x11_put16(self.band_buf, 2, pkt_len // 4)
                x11_put32(self.band_buf, 4, self.window_id)
                x11_put32(self.band_buf, 8, self.gc_id)
                x11_put16(self.band_buf, 12, dst_w)
                x11_put16(self.band_buf, 14, bh)
                x11_put16(self.band_buf, 16, 0)
                x11_put16(self.band_buf, 18, y)
                self.band_buf[20] = 0
                self.band_buf[21] = UInt8(self.root_depth)
                self.band_buf[22] = 0
                self.band_buf[23] = 0

                var band_px = dst_w * bh
                var band_ptr = Int(self.band_buf.unsafe_ptr())
                var dst_u32 = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=band_ptr + 24)
                var src_u32 = (
                    self.scaled_buf.unsafe_ptr().unsafe_offset(y * dst_w)
                    if use_scaled
                    else pixels.unsafe_ptr().unsafe_offset(y * dst_w)
                )
                var alpha_v = SIMD[DType.uint32, 8](0xFF000000)
                var idx = 0
                while idx + 8 <= band_px:
                    var v = src_u32.unsafe_load[width=8, alignment=4](idx)
                    dst_u32.unsafe_store[alignment=4](idx, v | alpha_v)
                    idx += 8
                while idx < band_px:
                    dst_u32[unsafe_offset=idx] = src_u32[unsafe_offset=idx] | UInt32(0xFF000000)
                    idx += 1

                if self._send_req_ptr(band_ptr, pkt_len) < 0:
                    return
                y += bh

        # Smooth ~60 Hz pacing on asynchronous X11 sockets unless FLINT_UNCAPPED=1
        if not self.uncapped:
            var now = get_time()
            var target_dt = 1.0 / 62.0
            var remain_ms = Int((target_dt - (now - self.last_present_at)) * 1000.0)
            if remain_ms > 1 and remain_ms <= 16:
                _ = self.wait_io(1, remain_ms)
            self.last_present_at = get_time()
        else:
            self.last_present_at = get_time()

    def toggle_fullscreen(mut self):
        if not self.active() or self.net_wm_state == 0 or self.net_wm_state_fullscreen == 0:
            return
        self.fullscreen = not self.fullscreen
        var ev = List[UInt8](length=44, fill=0)
        ev[0] = 25  # SendEvent
        ev[1] = 0
        x11_put16(ev, 2, 11)
        x11_put32(ev, 4, self.root_window)
        x11_put32(ev, 8, 0x00180000)  # SubstructureRedirectMask | SubstructureNotifyMask
        ev[12] = 33  # ClientMessage
        ev[13] = 32  # format = 32
        x11_put32(ev, 16, self.window_id)
        x11_put32(ev, 20, self.net_wm_state)
        x11_put32(ev, 24, 1 if self.fullscreen else 0)  # _NET_WM_STATE_ADD / REMOVE
        x11_put32(ev, 28, self.net_wm_state_fullscreen)
        x11_put32(ev, 32, 0)
        x11_put32(ev, 36, 1)  # normal application source indication
        _ = self._send_req(ev)

    def set_pointer_lock(mut self, active: Bool):
        """Toggles pointer lock / raw relative motion tracking with Wayland symmetry."""
        self.pointer_locked = active
        self.has_last_mouse = False
        self.pending_rel_x = 0
        self.pending_rel_y = 0
        if not self.active():
            return
        if active:
            # GrabPointer (opcode 26, len 6 words = 24 bytes)
            var req = List[UInt8](length=24, fill=0)
            req[0] = 26
            req[1] = 1  # owner_events = True
            x11_put16(req, 2, 6)
            x11_put32(req, 4, self.window_id)
            x11_put16(req, 8, 0x004C)  # ButtonPress | ButtonRelease | PointerMotion
            req[10] = 1  # pointer_mode = Asynchronous
            req[11] = 1  # keyboard_mode = Asynchronous
            x11_put32(req, 12, self.window_id)  # confine_to
            x11_put32(req, 16, self.blank_cursor)
            x11_put32(req, 20, 0)  # time = CurrentTime
            _ = self._send_req(req)
        else:
            # UngrabPointer (opcode 27, len 2 words = 8 bytes)
            var req = List[UInt8](length=8, fill=0)
            req[0] = 27
            x11_put16(req, 2, 2)
            x11_put32(req, 4, 0)  # time = CurrentTime
            _ = self._send_req(req)

    def clipboard_set(mut self, text: String):
        self.clipboard_text = text
        self.clipboard_owned = True
        if not self.active() or self.atom_clipboard == 0:
            return
        var req = List[UInt8](length=16, fill=0)
        req[0] = 22  # SetSelectionOwner
        x11_put16(req, 2, 4)
        x11_put32(req, 4, self.window_id)
        x11_put32(req, 8, self.atom_clipboard)
        x11_put32(req, 12, 0)  # CurrentTime
        _ = self._send_req(req)

    def clipboard_get(mut self) -> String:
        if self.clipboard_owned:
            return self.clipboard_text
        if not self.active() or self.atom_clipboard == 0:
            return ""
        var owner_req = List[UInt8](length=8, fill=0)
        owner_req[0] = 23  # GetSelectionOwner
        x11_put16(owner_req, 2, 2)
        x11_put32(owner_req, 4, self.atom_clipboard)
        var o_seq = self._send_req(owner_req)
        if o_seq < 0:
            return ""
        var o_rep = self._wait_reply(o_seq)
        if not o_rep[0] or len(o_rep[2]) < 12:
            return ""
        var owner = x11_get32(o_rep[2], 8)
        if owner == self.window_id:
            return self.clipboard_text
        if owner == 0:
            return ""
        var targets = [self.atom_utf8_string if self.atom_utf8_string != 0 else 31, 31]
        for t_idx in range(2):
            var target = targets[t_idx]
            if t_idx == 1 and target == targets[0]:
                continue
            self.selection_notify_arrived = False
            self.selection_notify_prop = 0
            var conv = List[UInt8](length=24, fill=0)
            conv[0] = 24  # ConvertSelection
            x11_put16(conv, 2, 6)
            x11_put32(conv, 4, self.window_id)
            x11_put32(conv, 8, self.atom_clipboard)
            x11_put32(conv, 12, target)
            x11_put32(conv, 16, self.atom_flint_clip)
            x11_put32(conv, 20, 0)
            if self._send_req(conv) < 0:
                return ""
            var deadline = get_time() + 0.25
            while not self.selection_notify_arrived and get_time() < deadline and not self.close_requested:
                _ = self.wait_io(1, 15)
                self.receive()
            if not self.selection_notify_arrived or self.selection_notify_prop == 0:
                continue
            var gp = List[UInt8](length=24, fill=0)
            gp[0] = 20  # GetProperty
            gp[1] = 1   # delete = True
            x11_put16(gp, 2, 6)
            x11_put32(gp, 4, self.window_id)
            x11_put32(gp, 8, self.selection_notify_prop)
            x11_put32(gp, 12, 0)  # AnyPropertyType
            x11_put32(gp, 16, 0)
            x11_put32(gp, 20, 65536)
            var g_seq = self._send_req(gp)
            if g_seq < 0:
                return ""
            var g_rep = self._wait_reply(g_seq)
            if not g_rep[0] or len(g_rep[2]) < 24:
                continue
            var ptype = x11_get32(g_rep[2], 8)
            if ptype == self.atom_incr and self.atom_incr != 0:
                continue
            var fmt = Int(g_rep[2][1])
            var nitems = x11_get32(g_rep[2], 16)
            var nbytes = nitems * max(1, fmt // 8)
            ref extra = g_rep[3]
            if nbytes <= 0 or nbytes > len(extra):
                return ""
            var out = List[UInt8](capacity=nbytes)
            for i in range(nbytes):
                out.append(extra[i])
            return String(unsafe_from_utf8=out)
        return ""

    def _ensure_xcb_connection(mut self) -> Bool:
        if self.xcb_conn != 0:
            return True
        if self.xcb_lib == 0:
            self.xcb_lib = dl_open("libxcb.so.1")
        if self.xcb_lib == 0:
            return False
        var fn_connect = dl_sym(self.xcb_lib, "xcb_connect")
        var fn_has_err = dl_sym(self.xcb_lib, "xcb_connection_has_error")
        self.xcb_disconnect = dl_sym(self.xcb_lib, "xcb_disconnect")
        if fn_connect == 0 or fn_has_err == 0 or self.xcb_disconnect == 0:
            _ = dl_close(self.xcb_lib)
            self.xcb_lib = 0
            self.xcb_disconnect = 0
            return False
        comptime XcbConnect = def(Int, Int) thin abi("C") -> Int
        comptime XcbHasError = def(Int) thin abi("C") -> Int32
        var conn = Pointer(to=fn_connect).unsafe_bitcast[XcbConnect]()[unsafe_offset=0](0, 0)
        if conn == 0:
            return False
        var err = Pointer(to=fn_has_err).unsafe_bitcast[XcbHasError]()[unsafe_offset=0](conn)
        if err != 0:
            comptime XcbDisconnect = def(Int) thin abi("C") -> None
            var fn_disc = self.xcb_disconnect
            Pointer(to=fn_disc).unsafe_bitcast[XcbDisconnect]()[unsafe_offset=0](conn)
            return False
        self.xcb_conn = conn
        return True

    def _close_xcb_connection(mut self):
        if self.xcb_conn != 0 and self.xcb_disconnect != 0:
            comptime XcbDisconnect = def(Int) thin abi("C") -> None
            var fn_disc = self.xcb_disconnect
            Pointer(to=fn_disc).unsafe_bitcast[XcbDisconnect]()[unsafe_offset=0](self.xcb_conn)
        self.xcb_conn = 0
        self.xcb_disconnect = 0
        if self.xcb_lib != 0:
            _ = dl_close(self.xcb_lib)
            self.xcb_lib = 0

    def attach_vulkan(
        mut self,
        mut vk: VulkanGpu,
        width: Int = 0,
        height: Int = 0,
        prefer_mailbox: Bool = True,
        prefer_hdr10: Bool = False,
    ) -> SwapchainHandle:
        if not self.active() or not self._ensure_xcb_connection():
            return SwapchainHandle()
        var w = width if width > 0 else self.width
        var h = height if height > 0 else self.height
        var sc = vk.attach_surface(
            xcb_conn=self.xcb_conn,
            xcb_win=self.window_id,
            width=w,
            height=h,
            prefer_mailbox=prefer_mailbox,
            prefer_hdr10=prefer_hdr10,
        )
        self.gpu_swapchain = sc.vk_swapchain
        return sc^

    def detach_vulkan(mut self, mut vk: VulkanGpu) -> Bool:
        vk.close_swapchain()
        self.gpu_swapchain = 0
        self._close_xcb_connection()
        return self.active()

    def supports_mit_shm(self) -> Bool:
        return not self.shm_disabled

    def supports_xi_raw_motion(self) -> Bool:
        return True

    def poll_events(mut self, mut input: InputState):
        self.poll(input)

    def present_argb(mut self, pixels: List[UInt32], width: Int, height: Int):
        var copy_px = pixels.copy()
        self.present(copy_px, width, height)

    def framebuffer_size(self) -> Tuple[Int, Int]:
        return (max(1, self.width), max(1, self.height))

    def content_scale(self) -> Float32:
        """`Xft.dpi` / 96 (physical DPI when unset), in 0.25 steps from 1.0 to 4.0. X11 windows are
        sized in device pixels, so this only scales the UI; the framebuffer is the window size."""
        return self.dpi_scale

    def should_close(self) -> Bool:
        return self.close_requested

    def shutdown(mut self):
        self.close()

    def close(mut self):
        self.evdev.close()
        self._close_xcb_connection()
        self._destroy_shm()
        if self.fd >= 0:
            if self.blank_cursor != 0:
                var fc = List[UInt8](length=8, fill=0)
                fc[0] = 95  # FreeCursor
                x11_put16(fc, 2, 2)
                x11_put32(fc, 4, self.blank_cursor)
                _ = self._send_req(fc)
            if self.blank_pixmap != 0:
                var fp = List[UInt8](length=8, fill=0)
                fp[0] = 54  # FreePixmap
                x11_put16(fp, 2, 2)
                x11_put32(fp, 4, self.blank_pixmap)
                _ = self._send_req(fp)
            if self.window_id != 0:
                var req = List[UInt8](length=8, fill=0)
                req[0] = 4  # DestroyWindow
                x11_put16(req, 2, 2)
                x11_put32(req, 4, self.window_id)
                _ = self._send_req(req)
        close_fd(self.fd)
        self.fd = -1
        self.window_id = 0
        self.gc_id = 0
        self.blank_cursor = 0
        self.blank_pixmap = 0
        self.gpu_swapchain = 0
        self.clipboard_owned = False
