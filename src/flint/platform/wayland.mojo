"""Direct Wayland/XDG-shell client using Linux sockets and shared memory.

No libwayland, XKB, EGL, OpenGL, or native compatibility shim is used. The one
exception is SD_PRESENT=swapchain, where the Vulkan driver needs libwayland objects
for this window: see platform/wayland_bridge.mojo. All this code runs the same then.
"""
from std.ffi import external_call
from flint.platform.input import InputState
from flint.platform.traits import WindowBackend
from flint.input.evdev import EvdevHub
from flint.platform.wayland_bridge import WaylandBridge
from flint.platform.x11 import X11Window
from flint.platform.xkb import XkbKeymap
from flint.platform.kms import KmsWindow, on_virtual_console, fit_rect
from flint.platform.vulkan import VulkanGpu, SwapchainHandle
from flint.platform.os import env_value, env_flag, close_fd, get_time, last_errno, EINTR

# Held in the pending key and button queues: everything held is released here.
comptime RELEASE_HELD: Int = -1
comptime KEY_CODES: Int = 512
comptime MOUSE_BUTTONS: Int = 3


def wl_put16(mut data: List[UInt8], offset: Int, value: Int):
    data[offset] = UInt8(value & 255)
    data[offset + 1] = UInt8((value >> 8) & 255)


def wl_put32(mut data: List[UInt8], offset: Int, value: Int):
    data[offset] = UInt8(value & 255)
    data[offset + 1] = UInt8((value >> 8) & 255)
    data[offset + 2] = UInt8((value >> 16) & 255)
    data[offset + 3] = UInt8((value >> 24) & 255)


def wl_put64(mut data: List[UInt8], offset: Int, value: UInt64):
    for index in range(8):
        data[offset + index] = UInt8((value >> UInt64(index * 8)) & 255)


def wl_get16(data: List[UInt8], offset: Int) -> Int:
    return Int(data[offset]) | (Int(data[offset + 1]) << 8)


def wl_get32(data: List[UInt8], offset: Int) -> Int:
    return Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)


def wl_gets32(data: List[UInt8], offset: Int) -> Int:
    var value = wl_get32(data, offset)
    return value - 4294967296 if value >= 2147483648 else value


def wl_append32(mut data: List[UInt8], value: Int):
    var offset = len(data)
    data.resize(offset + 4, 0)
    wl_put32(data, offset, value)


def wl_append_string(mut data: List[UInt8], value: String):
    var offset = len(data)
    var padded = (value.byte_length() + 1 + 3) & ~3
    data.resize(offset + 4 + padded, 0)
    wl_put32(data, offset, value.byte_length() + 1)
    for index in range(value.byte_length()):
        data[offset + 4 + index] = value.as_bytes()[index]


def wl_fixed(value: Int) -> Float32:
    return Float32(value) / 256.0


def wl_get_string(data: List[UInt8], offset: Int, max_end: Int) -> String:
    if offset + 4 > max_end or offset + 4 > len(data):
        return ""
    var length = wl_get32(data, offset)
    if length <= 1 or offset + 4 + length > max_end or offset + 4 + length > len(data):
        return ""
    var bytes = List[UInt8](capacity=length - 1)
    for i in range(length - 1):
        bytes.append(data[offset + 4 + i])
    return String(unsafe_from_utf8=bytes)


def wl_append_nullable_string(mut data: List[UInt8], value: String):
    if value.byte_length() == 0:
        wl_append32(data, 0)
    else:
        wl_append_string(data, value)


def _hex_digit_val(b: UInt8) -> Int:
    if b >= 48 and b <= 57:
        return Int(b - 48)
    if b >= 65 and b <= 70:
        return Int(b - 55)
    if b >= 97 and b <= 102:
        return Int(b - 87)
    return -1


def parse_uri_list(text: String) -> List[String]:
    """Parses RFC 2483 `text/uri-list` payload into decoded local filesystem paths."""
    var out = List[String]()
    var bytes = text.as_bytes()
    var n = len(bytes)
    var i = 0
    while i < n:
        var start = i
        while i < n and bytes[i] != 10:
            i += 1
        var end = i
        if end > start and bytes[end - 1] == 13:
            end -= 1
        while start < end and (bytes[start] == 32 or bytes[start] == 9):
            start += 1
        while end > start and (bytes[end - 1] == 32 or bytes[end - 1] == 9):
            end -= 1
        if i < n and bytes[i] == 10:
            i += 1
        if end <= start or bytes[start] == 35:  # empty or '#' comment
            continue
        var line = String(text[byte=start:end])
        var path_raw = line
        if line.startswith("file://localhost/"):
            path_raw = String(line[byte=16:])
        elif line.startswith("file:///"):
            path_raw = String(line[byte=7:])
        var pbytes = path_raw.as_bytes()
        var decoded = List[UInt8](capacity=len(pbytes))
        var j = 0
        while j < len(pbytes):
            if pbytes[j] == 37 and j + 2 < len(pbytes):
                var hi = _hex_digit_val(pbytes[j + 1])
                var lo = _hex_digit_val(pbytes[j + 2])
                if hi >= 0 and lo >= 0:
                    decoded.append(UInt8((hi << 4) | lo))
                    j += 3
                    continue
            decoded.append(pbytes[j])
            j += 1
        out.append(String(unsafe_from_utf8=decoded))
    return out^


def evdev_key(key: Int) -> Int:
    # Wayland carries Linux input-event key codes. This is the game's fixed
    # physical-key mapping; it remains usable without an XKB client library.
    var letters = [30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44]
    for index in range(26):
        if key == letters[index]:
            return 65 + index
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


struct WaylandWindow(WindowBackend):
    var fd: Int
    var bridge: WaylandBridge
    var wants_bridge: Bool
    var width: Int
    var height: Int
    var close_requested: Bool
    var configured: Bool
    var focused: Bool
    var fullscreen: Bool
    var next_id: Int
    var registry: Int
    var compositor: Int
    var compositor_name: Int
    var shm: Int
    var shm_name: Int
    var dmabuf: Int
    var buffer_releases: Int
    var commits: Int
    var viewporter: Int
    var viewport: Int
    var viewport_width: Int
    var viewport_height: Int
    var pointer_constraints: Int
    var relative_manager: Int
    var relative_pointer: Int
    var locked_pointer: Int
    var pointer_locked: Bool
    var pending_relative_x: Float32
    var pending_relative_y: Float32
    var wm_base: Int
    var wm_name: Int
    var seat: Int
    var seat_name: Int
    var seat_version: Int
    var seat_capabilities: Int
    var surface: Int
    var xdg_surface: Int
    var toplevel: Int
    var pointer: Int
    var pointer_enter_serial: Int
    var keyboard: Int
    var incoming: List[UInt8]
    var pending_fds: List[Int]
    var awaiting_syncs: List[Int]
    var done_callbacks: List[Int]
    var protocol_errors: Int
    var dmabuf_objects: List[Int]
    var buffer_rejected: Bool
    var last_error: String
    var shared_fd: Int
    var shared_size: Int
    var pool: Int
    var buffers: List[Int]
    var buffer_busy: List[Bool]
    var buffer_width: Int
    var buffer_height: Int
    var frame_callback: Int
    var frame_requested_at: Float64
    var pending_mouse_x: Float32
    var pending_mouse_y: Float32
    var pending_buttons: List[Int]
    var pending_button_down: List[Bool]
    var pending_wheel: Float32
    var pending_keys: List[Int]
    var pending_key_down: List[Bool]
    var x11: X11Window
    var using_x11: Bool
    var kms: KmsWindow
    var using_kms: Bool
    var wants_kms_vulkan: Bool  # on KMS, leave the display to a Vulkan swapchain (KmsWindow GPU mode)
    var wants_gpu_fullscreen: Bool  # a GPU game: Vulkan bridge + fullscreen, swapchain scaled by the compositor
    var swapchain: SwapchainHandle
    var gpu_width: Int  # size of the swapchain attach_vulkan made on Wayland, 0 when none
    var gpu_height: Int
    var window_title: String
    var logical_width: Int
    var logical_height: Int
    # F23: Fractional scaling, Server-Side Decorations, Clipboard & DnD, XKB, EvdevHub
    var fractional_scale_manager: Int
    var fractional_scale: Int
    var fractional_scale_num: Int
    var preferred_buffer_scale: Int
    var applied_buffer_scale: Int
    var scale_changed: Bool
    var decoration_manager: Int
    var toplevel_decoration: Int
    var decoration_mode: Int
    var data_device_manager: Int
    var data_device_version: Int
    var data_device: Int
    var last_input_serial: Int
    var clipboard_source: Int
    var clipboard_text: String
    var clipboard_owned: Bool
    var selection_offer_id: Int
    var offer_ids: List[Int]
    var offer_mimes: List[List[String]]
    var dnd_offer_id: Int
    var dnd_serial: Int
    var dnd_x: Float32
    var dnd_y: Float32
    var pending_dropped_files: List[String]
    var xkb_keymap: XkbKeymap
    var pending_text_utf8: String
    var evdev: EvdevHub

    def __init__(out self):
        self.fd = -1
        self.bridge = WaylandBridge()
        self.wants_bridge = False
        self.width = 1280
        self.height = 720
        self.close_requested = False
        self.configured = False
        self.focused = False
        self.fullscreen = False
        self.next_id = 2
        self.registry = 0
        self.compositor = 0
        self.compositor_name = 0
        self.shm = 0
        self.shm_name = 0
        self.dmabuf = 0
        self.buffer_releases = 0
        self.commits = 0
        self.viewporter = 0
        self.viewport = 0
        self.viewport_width = 0
        self.viewport_height = 0
        self.pointer_constraints = 0
        self.relative_manager = 0
        self.relative_pointer = 0
        self.locked_pointer = 0
        self.pointer_locked = False
        self.pending_relative_x = 0
        self.pending_relative_y = 0
        self.wm_base = 0
        self.wm_name = 0
        self.seat = 0
        self.seat_name = 0
        self.seat_version = 0
        self.seat_capabilities = 0
        self.surface = 0
        self.xdg_surface = 0
        self.toplevel = 0
        self.pointer = 0
        self.pointer_enter_serial = 0
        self.keyboard = 0
        self.incoming = List[UInt8]()
        self.pending_fds = List[Int]()
        self.awaiting_syncs = List[Int]()
        self.done_callbacks = List[Int]()
        self.protocol_errors = 0
        self.dmabuf_objects = List[Int]()
        self.buffer_rejected = False
        self.last_error = ""
        self.shared_fd = -1
        self.shared_size = 0
        self.pool = 0
        self.buffers = List[Int]()
        self.buffer_busy = List[Bool]()
        self.buffer_width = 0
        self.buffer_height = 0
        self.frame_callback = 0
        self.frame_requested_at = 0
        self.pending_mouse_x = -1
        self.pending_mouse_y = -1
        self.pending_buttons = List[Int]()
        self.pending_button_down = List[Bool]()
        self.pending_wheel = 0
        self.pending_keys = List[Int]()
        self.pending_key_down = List[Bool]()
        self.x11 = X11Window()
        self.using_x11 = False
        self.kms = KmsWindow()
        self.using_kms = False
        self.wants_kms_vulkan = False
        self.wants_gpu_fullscreen = False
        self.swapchain = SwapchainHandle()
        self.gpu_width = 0
        self.gpu_height = 0
        self.window_title = "FLINT ENGINE"
        self.logical_width = 1280
        self.logical_height = 720
        self.fractional_scale_manager = 0
        self.fractional_scale = 0
        self.fractional_scale_num = 120
        self.preferred_buffer_scale = 1
        self.applied_buffer_scale = 1
        self.scale_changed = False
        self.decoration_manager = 0
        self.toplevel_decoration = 0
        self.decoration_mode = 0
        self.data_device_manager = 0
        self.data_device_version = 0
        self.data_device = 0
        self.last_input_serial = 0
        self.clipboard_source = 0
        self.clipboard_text = ""
        self.clipboard_owned = False
        self.selection_offer_id = 0
        self.offer_ids = List[Int]()
        self.offer_mimes = List[List[String]]()
        self.dnd_offer_id = 0
        self.dnd_serial = 0
        self.dnd_x = 0
        self.dnd_y = 0
        self.pending_dropped_files = List[String]()
        self.xkb_keymap = XkbKeymap()
        self.pending_text_utf8 = ""
        self.evdev = EvdevHub(enable_hotplug=False)

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.bridge = existing.bridge^
        self.wants_bridge = existing.wants_bridge
        self.width = existing.width
        self.height = existing.height
        self.close_requested = existing.close_requested
        self.configured = existing.configured
        self.focused = existing.focused
        self.fullscreen = existing.fullscreen
        self.next_id = existing.next_id
        self.registry = existing.registry
        self.compositor = existing.compositor
        self.compositor_name = existing.compositor_name
        self.shm = existing.shm
        self.shm_name = existing.shm_name
        self.dmabuf = existing.dmabuf
        self.buffer_releases = existing.buffer_releases
        self.commits = existing.commits
        self.viewporter = existing.viewporter
        self.viewport = existing.viewport
        self.viewport_width = existing.viewport_width
        self.viewport_height = existing.viewport_height
        self.pointer_constraints = existing.pointer_constraints
        self.relative_manager = existing.relative_manager
        self.relative_pointer = existing.relative_pointer
        self.locked_pointer = existing.locked_pointer
        self.pointer_locked = existing.pointer_locked
        self.pending_relative_x = existing.pending_relative_x
        self.pending_relative_y = existing.pending_relative_y
        self.wm_base = existing.wm_base
        self.wm_name = existing.wm_name
        self.seat = existing.seat
        self.seat_name = existing.seat_name
        self.seat_version = existing.seat_version
        self.seat_capabilities = existing.seat_capabilities
        self.surface = existing.surface
        self.xdg_surface = existing.xdg_surface
        self.toplevel = existing.toplevel
        self.pointer = existing.pointer
        self.pointer_enter_serial = existing.pointer_enter_serial
        self.keyboard = existing.keyboard
        self.incoming = existing.incoming^
        self.pending_fds = existing.pending_fds^
        self.awaiting_syncs = existing.awaiting_syncs^
        self.done_callbacks = existing.done_callbacks^
        self.protocol_errors = existing.protocol_errors
        self.dmabuf_objects = existing.dmabuf_objects^
        self.buffer_rejected = existing.buffer_rejected
        self.last_error = existing.last_error^
        self.shared_fd = existing.shared_fd
        self.shared_size = existing.shared_size
        self.pool = existing.pool
        self.buffers = existing.buffers^
        self.buffer_busy = existing.buffer_busy^
        self.buffer_width = existing.buffer_width
        self.buffer_height = existing.buffer_height
        self.frame_callback = existing.frame_callback
        self.frame_requested_at = existing.frame_requested_at
        self.pending_mouse_x = existing.pending_mouse_x
        self.pending_mouse_y = existing.pending_mouse_y
        self.pending_buttons = existing.pending_buttons^
        self.pending_button_down = existing.pending_button_down^
        self.pending_wheel = existing.pending_wheel
        self.pending_keys = existing.pending_keys^
        self.pending_key_down = existing.pending_key_down^
        self.x11 = existing.x11^
        self.using_x11 = existing.using_x11
        self.kms = existing.kms^
        self.using_kms = existing.using_kms
        self.wants_kms_vulkan = existing.wants_kms_vulkan
        self.wants_gpu_fullscreen = existing.wants_gpu_fullscreen
        self.swapchain = existing.swapchain^
        self.gpu_width = existing.gpu_width
        self.gpu_height = existing.gpu_height
        self.window_title = existing.window_title^
        self.logical_width = existing.logical_width
        self.logical_height = existing.logical_height
        self.fractional_scale_manager = existing.fractional_scale_manager
        self.fractional_scale = existing.fractional_scale
        self.fractional_scale_num = existing.fractional_scale_num
        self.preferred_buffer_scale = existing.preferred_buffer_scale
        self.applied_buffer_scale = existing.applied_buffer_scale
        self.scale_changed = existing.scale_changed
        self.decoration_manager = existing.decoration_manager
        self.toplevel_decoration = existing.toplevel_decoration
        self.decoration_mode = existing.decoration_mode
        self.data_device_manager = existing.data_device_manager
        self.data_device_version = existing.data_device_version
        self.data_device = existing.data_device
        self.last_input_serial = existing.last_input_serial
        self.clipboard_source = existing.clipboard_source
        self.clipboard_text = existing.clipboard_text^
        self.clipboard_owned = existing.clipboard_owned
        self.selection_offer_id = existing.selection_offer_id
        self.offer_ids = existing.offer_ids^
        self.offer_mimes = existing.offer_mimes^
        self.dnd_offer_id = existing.dnd_offer_id
        self.dnd_serial = existing.dnd_serial
        self.dnd_x = existing.dnd_x
        self.dnd_y = existing.dnd_y
        self.pending_dropped_files = existing.pending_dropped_files^
        self.xkb_keymap = existing.xkb_keymap^
        self.pending_text_utf8 = existing.pending_text_utf8^
        self.evdev = existing.evdev^
        existing.fd = -1
        existing.shared_fd = -1
        existing.shared_size = 0
        existing.using_x11 = False
        existing.using_kms = False
        existing.gpu_width = 0
        existing.gpu_height = 0
        existing.toplevel = 0
        existing.xdg_surface = 0
        existing.surface = 0
        existing.pointer = 0
        existing.keyboard = 0
        existing.seat = 0
        existing.wm_base = 0
        existing.pool = 0
        existing.viewport = 0
        existing.fractional_scale = 0
        existing.toplevel_decoration = 0
        existing.data_device = 0
        existing.clipboard_source = 0
        existing.selection_offer_id = 0
        existing.dnd_offer_id = 0

    def __deinit__(deinit self):
        self.close()

    def set_title(mut self, title: String):
        self.window_title = title
        if self.using_x11:
            self.x11.set_title(title)
        elif self.fd >= 0 and self.toplevel > 0:
            var body = List[UInt8]()
            wl_append_string(body, title)
            _ = self.request(self.toplevel, 2, body)

    def backend_name(self) -> String:
        if self.using_kms:
            return self.kms.backend_name()
        if self.using_x11:
            return "X11"
        if self.fd >= 0:
            return "Wayland"
        return "Headless"

    def allocate_id(mut self) -> Int:
        if self.bridge.active():
            return self.bridge.allocate_id()
        var result = self.next_id
        self.next_id += 1
        return result

    def write_all(mut self, mut data: List[UInt8]) -> Bool:
        if self.bridge.active():
            var sent = self.bridge.write_all(Int(data.unsafe_ptr()), len(data))
            if not sent:
                self.close_requested = True
            return sent
        var offset = 0
        while offset < len(data):
            var count = Int(external_call["write", Int64](Int32(self.fd), data.unsafe_ptr().unsafe_offset(offset), Int64(len(data) - offset)))
            if count <= 0:
                self.close_requested = True
                return False
            offset += count
        return True

    def request(mut self, object: Int, opcode: Int, mut body: List[UInt8]) -> Bool:
        if self.fd < 0:
            return False
        var data = List[UInt8](length=8 + len(body), fill=0)
        wl_put32(data, 0, object)
        wl_put16(data, 4, opcode)
        wl_put16(data, 6, len(data))
        for index in range(len(body)):
            data[8 + index] = body[index]
        return self.write_all(data)

    def empty_request(mut self, object: Int, opcode: Int) -> Bool:
        var body = List[UInt8]()
        return self.request(object, opcode, body)

    def send_with_fd(mut self, mut data: List[UInt8], fd: Int) -> Bool:
        if self.bridge.active():
            var sent = self.bridge.write_with_fd(Int(data.unsafe_ptr()), len(data), fd)
            if not sent:
                self.close_requested = True
            return sent
        var iov = List[UInt8](length=16, fill=0)
        wl_put64(iov, 0, UInt64(data.unsafe_ptr()))
        wl_put64(iov, 8, UInt64(len(data)))
        var control = List[UInt8](length=24, fill=0)
        wl_put64(control, 0, 20)
        wl_put32(control, 8, 1)
        wl_put32(control, 12, 1)
        wl_put32(control, 16, fd)
        var message = List[UInt8](length=56, fill=0)
        wl_put64(message, 16, UInt64(iov.unsafe_ptr()))
        wl_put64(message, 24, 1)
        wl_put64(message, 32, UInt64(control.unsafe_ptr()))
        wl_put64(message, 40, 24)
        var count = Int(external_call["sendmsg", Int64](Int32(self.fd), message.unsafe_ptr(), Int32(0)))
        # The message holds these two by address, which does not keep them alive.
        _ = iov^
        _ = control^
        if count != len(data):
            var error_number = 0
            if count < 0:
                var errno_pointer = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
                error_number = Int(errno_pointer[unsafe_offset=0])
            self.last_error = "sendmsg returned " + String(count) + " errno " + String(error_number) + " for " + String(len(data)) + " bytes"
            self.close_requested = True
            return False
        return True

    def wait_socket(self, timeout_ms: Int) -> Bool:
        if self.bridge.active():
            return self.bridge.wait(timeout_ms)
        var descriptor = List[UInt8](length=8, fill=0)
        wl_put32(descriptor, 0, self.fd)
        wl_put16(descriptor, 4, 1)
        return external_call["poll", Int32](descriptor.unsafe_ptr(), UInt64(1), Int32(timeout_ms)) > 0

    def connect_socket(mut self) -> Bool:
        # A compositor disconnect must surface as a write error, not kill the game.
        _ = external_call["signal", Int](Int32(13), Int(1))  # SIGPIPE, SIG_IGN
        var runtime = env_value("XDG_RUNTIME_DIR")
        var display = env_value("WAYLAND_DISPLAY")
        if display.byte_length() == 0:
            display = "wayland-0"
        var path = display
        if display.byte_length() > 0 and display.as_bytes()[0] != 47:
            if runtime.byte_length() == 0:
                return False
            path = runtime + "/" + display
        if path.byte_length() >= 106:
            return False
        self.fd = Int(external_call["socket", Int32](Int32(1), Int32(1), Int32(0)))
        if self.fd < 0:
            return False
        var address = List[UInt8](length=110, fill=0)
        address[0] = 1
        for index in range(path.byte_length()):
            address[2 + index] = path.as_bytes()[index]
        if external_call["connect", Int32](Int32(self.fd), address.unsafe_ptr(), Int32(110)) != 0:
            self.close()
            return False
        return True

    def bind_global(mut self, name: Int, interface: String, version: Int) -> Int:
        var object = self.allocate_id()
        var body = List[UInt8]()
        wl_append32(body, name)
        wl_append_string(body, interface)
        wl_append32(body, version)
        wl_append32(body, object)
        if not self.request(self.registry, 0, body):
            return 0
        return object

    def bind_compositor(mut self, name: Int, version: Int) -> Int:
        if not self.bridge.active():
            return self.bind_global(name, "wl_compositor", 1)
        var object = self.bridge.reserve_compositor()
        var body = List[UInt8]()
        wl_append32(body, name)
        wl_append_string(body, "wl_compositor")
        wl_append32(body, min(version, 4))
        wl_append32(body, object)
        if not self.request(self.registry, 0, body):
            return 0
        return object

    def sync(mut self) -> Int:
        var callback = self.allocate_id()
        self.awaiting_syncs.append(callback)
        var body = List[UInt8]()
        wl_append32(body, callback)
        _ = self.request(1, 0, body)
        return callback

    def pop_incoming_fd(mut self) -> Int:
        if len(self.pending_fds) == 0:
            return -1
        return self.pending_fds.pop(0)

    def _register_offer(mut self, offer_id: Int):
        if offer_id == 0:
            return
        for i in range(len(self.offer_ids)):
            if self.offer_ids[i] == offer_id:
                self.offer_mimes[i].clear()
                return
        self.offer_ids.append(offer_id)
        self.offer_mimes.append(List[String]())

    def _has_offer(self, offer_id: Int) -> Bool:
        if offer_id == 0:
            return False
        for i in range(len(self.offer_ids)):
            if self.offer_ids[i] == offer_id:
                return True
        return False

    def _add_offer_mime(mut self, offer_id: Int, mime: String):
        for i in range(len(self.offer_ids)):
            if self.offer_ids[i] == offer_id:
                self.offer_mimes[i].append(mime)
                return

    def _offer_has_mime(self, offer_id: Int, mime: String) -> Bool:
        for i in range(len(self.offer_ids)):
            if self.offer_ids[i] == offer_id:
                for m in self.offer_mimes[i]:
                    if m == mime:
                        return True
                return False
        return False

    def _destroy_offer(mut self, offer_id: Int):
        if offer_id == 0:
            return
        if self.fd >= 0:
            _ = self.empty_request(offer_id, 2)  # wl_data_offer.destroy
        for i in range(len(self.offer_ids)):
            if self.offer_ids[i] == offer_id:
                _ = self.offer_ids.pop(i)
                _ = self.offer_mimes.pop(i)
                return

    def _pick_text_mime(self, offer_id: Int) -> String:
        if self._offer_has_mime(offer_id, "text/plain;charset=utf-8"):
            return "text/plain;charset=utf-8"
        if self._offer_has_mime(offer_id, "UTF8_STRING"):
            return "UTF8_STRING"
        if self._offer_has_mime(offer_id, "text/plain"):
            return "text/plain"
        return ""

    def _pick_dnd_mime(self, offer_id: Int) -> String:
        if self._offer_has_mime(offer_id, "text/uri-list"):
            return "text/uri-list"
        if self._offer_has_mime(offer_id, "text/plain;charset=utf-8"):
            return "text/plain;charset=utf-8"
        if self._offer_has_mime(offer_id, "text/plain"):
            return "text/plain"
        return ""

    def _read_offer_string(mut self, offer_id: Int, mime: String) -> String:
        if self.fd < 0 or offer_id == 0 or mime.byte_length() == 0:
            return ""
        var pipe_fds = List[Int32](length=2, fill=-1)
        if external_call["pipe2", Int32](pipe_fds.unsafe_ptr(), Int32(0x80800)) != 0:  # O_NONBLOCK | O_CLOEXEC
            if external_call["pipe", Int32](pipe_fds.unsafe_ptr()) != 0:
                return ""
            _ = external_call["fcntl", Int32](pipe_fds[0], Int32(4), Int32(0x800))
        var read_fd = Int(pipe_fds[0])
        var write_fd = Int(pipe_fds[1])

        var body = List[UInt8]()
        wl_append_string(body, mime)
        var req = List[UInt8](length=8 + len(body), fill=0)
        wl_put32(req, 0, offer_id)
        wl_put16(req, 4, 1)  # wl_data_offer.receive
        wl_put16(req, 6, len(req))
        for i in range(len(body)):
            req[8 + i] = body[i]
        var ok = self.send_with_fd(req, write_fd)
        close_fd(write_fd)
        if not ok:
            close_fd(read_fd)
            return ""

        var out = List[UInt8]()
        var chunk = List[UInt8](length=4096, fill=0)
        var pfd = List[UInt8](length=8, fill=0)
        var deadline = get_time() + 0.25
        while get_time() < deadline:
            var n = Int(external_call["read", Int64](Int32(read_fd), chunk.unsafe_ptr(), Int64(len(chunk))))
            if n > 0:
                for i in range(n):
                    out.append(chunk[i])
                continue
            if n == 0:
                break
            var err = last_errno()
            if err == EINTR:
                continue
            if err != Int32(11):  # not EAGAIN
                break
            wl_put32(pfd, 0, read_fd)
            wl_put16(pfd, 4, 1)  # POLLIN
            _ = external_call["poll", Int32](pfd.unsafe_ptr(), UInt64(1), Int32(10))
            self.receive()
        _ = len(chunk)
        _ = len(pfd)
        close_fd(read_fd)
        return String(unsafe_from_utf8=out)

    def ensure_data_device(mut self):
        if self.data_device != 0 or self.data_device_manager == 0 or self.seat == 0 or self.fd < 0:
            return
        self.data_device = self.allocate_id()
        if self.bridge.active():
            self.bridge.note_data_device(self.data_device)
        var body = List[UInt8]()
        wl_append32(body, self.data_device)
        wl_append32(body, self.seat)
        _ = self.request(self.data_device_manager, 1, body)  # get_data_device

    def receive(mut self):
        if self.bridge.active():
            self.bridge.read_into(self.incoming, self.pending_fds)
            if self.bridge.failed():
                self.close_requested = True
            self.handle_incoming()
            return
        if self.fd < 0:
            return
        var buffer = List[UInt8](length=16384, fill=0)
        var iov = List[UInt8](length=16, fill=0)
        var control = List[UInt8](length=272, fill=0)
        var msg = List[UInt8](length=56, fill=0)
        for _ in range(128):
            for k in range(len(control)):
                control[k] = 0
            for k in range(len(msg)):
                msg[k] = 0
            wl_put64(iov, 0, UInt64(buffer.unsafe_ptr()))
            wl_put64(iov, 8, UInt64(len(buffer)))
            wl_put64(msg, 16, UInt64(iov.unsafe_ptr()))
            wl_put64(msg, 24, 1)
            wl_put64(msg, 32, UInt64(control.unsafe_ptr()))
            wl_put64(msg, 40, UInt64(len(control)))
            var count = Int(external_call["recvmsg", Int64](Int32(self.fd), msg.unsafe_ptr(), Int32(64)))  # MSG_DONTWAIT
            if count == 0:
                self.close_requested = True
                break
            if count < 0:
                if last_errno() == EINTR:
                    continue
                break
            for index in range(count):
                self.incoming.append(buffer[index])
            var ctrl_len = 0
            for idx in range(8):
                ctrl_len |= Int(msg[40 + idx]) << (idx * 8)
            var off = 0
            while off + 16 <= ctrl_len and off + 16 <= len(control):
                var cmsg_len = 0
                for idx in range(8):
                    cmsg_len |= Int(control[off + idx]) << (idx * 8)
                if cmsg_len < 16 or off + cmsg_len > len(control):
                    break
                if wl_get32(control, off + 8) == 1 and wl_get32(control, off + 12) == 1:  # SOL_SOCKET, SCM_RIGHTS
                    var n_fds = (cmsg_len - 16) // 4
                    for idx in range(n_fds):
                        self.pending_fds.append(wl_get32(control, off + 16 + idx * 4))
                off += (cmsg_len + 7) & ~7
            if count < len(buffer) and ctrl_len == 0:
                break
        _ = len(buffer)
        _ = len(iov)
        _ = len(control)
        _ = len(msg)
        self.handle_incoming()

    def handle_incoming(mut self):
        while len(self.incoming) >= 8:
            var object = wl_get32(self.incoming, 0)
            var header = wl_get32(self.incoming, 4)
            var opcode = header & 65535
            var size = header >> 16
            if size < 8 or (size & 3) != 0:
                self.protocol_errors += 1
                self.last_error = "invalid event header"
                self.close_requested = True
                self.incoming.clear()
                return
            if len(self.incoming) < size:
                break
            var pkt = List[UInt8](length=size, fill=0)
            for i in range(size):
                pkt[i] = self.incoming[i]
            var rem = len(self.incoming) - size
            if rem == 0:
                self.incoming.clear()
            else:
                for i in range(rem):
                    self.incoming[i] = self.incoming[size + i]
                self.incoming.resize(rem, 0)
            self.handle_event(pkt, object, opcode, size)

    def handle_event(mut self, pkt: List[UInt8], object: Int, opcode: Int, size: Int):
        if object == 1 and opcode == 0:
            self.protocol_errors += 1
            var culprit = wl_get32(pkt, 8) if size >= 12 else 0
            var message = "wl_display error"
            # wl_display.error carries object id, error code, then the message.
            if size >= 24:
                var length = wl_get32(pkt, 16)
                if length > 0 and 20 + length <= size:
                    var bytes = List[UInt8]()
                    for index in range(length - 1):
                        bytes.append(pkt[20 + index])
                    message = "object " + String(wl_get32(pkt, 8)) + " code " + String(wl_get32(pkt, 12)) + ": " + String(unsafe_from_utf8=bytes)
            self.last_error = message
            # The connection is lost either way, but only this error has a way forward.
            if culprit != 0 and (culprit == self.dmabuf or culprit in self.dmabuf_objects):
                self.buffer_rejected = True
            else:
                self.close_requested = True
            return
        if object == 1 and opcode == 1:
            return
        if object == self.registry and opcode == 0 and size >= 20:
            var name = wl_get32(pkt, 8)
            var length = wl_get32(pkt, 12)
            if length <= 0 or 16 + length > size:
                return
            var bytes = List[UInt8]()
            for index in range(length - 1):
                bytes.append(pkt[16 + index])
            var interface = String(unsafe_from_utf8=bytes)
            var version_offset = 16 + ((length + 3) & ~3)
            if version_offset + 4 > size:
                return
            var version = wl_get32(pkt, version_offset)
            if interface == "wl_compositor" and self.compositor == 0:
                self.compositor_name = name
                self.compositor = self.bind_compositor(name, version)
            elif interface == "wl_shm" and self.shm == 0:
                self.shm_name = name
                self.shm = self.bind_global(name, interface, 1)
            elif interface == "zwp_linux_dmabuf_v1" and self.dmabuf == 0 and version >= 2:
                self.dmabuf = self.bind_global(name, interface, min(version, 3))
            elif interface == "wp_viewporter" and self.viewporter == 0:
                self.viewporter = self.bind_global(name, interface, 1)
            elif interface == "zwp_pointer_constraints_v1" and self.pointer_constraints == 0:
                self.pointer_constraints = self.bind_global(name, interface, 1)
            elif interface == "zwp_relative_pointer_manager_v1" and self.relative_manager == 0:
                self.relative_manager = self.bind_global(name, interface, 1)
            elif interface == "xdg_wm_base" and self.wm_base == 0:
                self.wm_name = name
                self.wm_base = self.bind_global(name, interface, 1)
            elif interface == "wl_seat" and self.seat == 0:
                self.seat_name = name
                self.seat_version = min(version, 5)
                self.seat = self.bind_global(name, interface, self.seat_version)
                self.ensure_data_device()
            elif interface == "wp_fractional_scale_manager_v1" and self.fractional_scale_manager == 0:
                self.fractional_scale_manager = self.bind_global(name, interface, 1)
            elif interface == "zxdg_decoration_manager_v1" and self.decoration_manager == 0:
                self.decoration_manager = self.bind_global(name, interface, 1)
            elif interface == "wl_data_device_manager" and self.data_device_manager == 0:
                self.data_device_version = min(version, 3)
                self.data_device_manager = self.bind_global(name, interface, self.data_device_version)
                self.ensure_data_device()
            return
        if object == self.surface and self.surface != 0:
            if opcode == 2 and size >= 12:  # preferred_buffer_scale
                self.preferred_buffer_scale = max(1, wl_get32(pkt, 8))
            return
        if object == self.fractional_scale and self.fractional_scale != 0:
            if opcode == 0 and size >= 12:  # preferred_scale
                var num = wl_get32(pkt, 8)
                if num > 0:
                    self.fractional_scale_num = num
                    self.scale_changed = True
            return
        if object == self.toplevel_decoration and self.toplevel_decoration != 0:
            if opcode == 0 and size >= 12:  # configure(mode)
                self.decoration_mode = wl_get32(pkt, 8)
            return
        if object == self.wm_base and opcode == 0:
            var body = List[UInt8]()
            wl_append32(body, wl_get32(pkt, 8))
            _ = self.request(self.wm_base, 3, body)
            return
        if object == self.xdg_surface and opcode == 0:
            var body = List[UInt8]()
            wl_append32(body, wl_get32(pkt, 8))
            _ = self.request(self.xdg_surface, 4, body)
            self.configured = True
            return
        if object == self.toplevel and opcode == 0:
            var width = wl_get32(pkt, 8)
            var height = wl_get32(pkt, 12)
            if width > 0 and height > 0 and (width != self.width or height != self.height):
                self.width = width
                self.height = height
            self.fullscreen = False
            self.focused = False
            if size >= 20:
                var states_length = wl_get32(pkt, 16)
                var count = min((size - 20) // 4, states_length // 4)
                for index in range(count):
                    var state = wl_get32(pkt, 20 + index * 4)
                    if state == 2:
                        self.fullscreen = True
                    elif state == 4:
                        self.focused = True
            return
        if object == self.toplevel and opcode == 1:
            self.close_requested = True
            return
        if object == self.seat and opcode == 0:
            self.seat_capabilities = wl_get32(pkt, 8)
            self.create_input_devices()
            return
        if object == self.locked_pointer and self.locked_pointer != 0:
            self.pointer_locked = opcode == 0
            if self.pointer_locked:
                self.hide_cursor()
            return
        if object == self.relative_pointer and self.relative_pointer != 0:
            # Fields: time high, time low, dx, dy, then unaccelerated dx and dy at 24 and 28.
            if opcode == 0 and size >= 32 and self.pointer_locked:
                self.pending_relative_x += wl_fixed(wl_gets32(pkt, 24))
                self.pending_relative_y += wl_fixed(wl_gets32(pkt, 28))
            return
        if object == self.pointer and self.pointer != 0:
            if opcode == 0 and size >= 16:
                self.pointer_enter_serial = wl_get32(pkt, 8)
                self.last_input_serial = self.pointer_enter_serial
                if self.pointer_locked:
                    self.hide_cursor()
            elif opcode == 1:
                self.pending_buttons.append(RELEASE_HELD)
                self.pending_button_down.append(False)
            elif opcode == 2 and size >= 20:
                self.pending_mouse_x = wl_fixed(wl_gets32(pkt, 12))
                self.pending_mouse_y = wl_fixed(wl_gets32(pkt, 16))
            elif opcode == 3 and size >= 24:
                self.last_input_serial = wl_get32(pkt, 8)
                var button = wl_get32(pkt, 16) - 272  # BTN_LEFT, BTN_RIGHT, BTN_MIDDLE
                if button >= 0 and button < MOUSE_BUTTONS:
                    self.pending_buttons.append(button)
                    self.pending_button_down.append(wl_get32(pkt, 20) == 1)
            elif opcode == 4 and size >= 20:
                var axis = wl_get32(pkt, 12)
                if axis == 0:
                    self.pending_wheel -= wl_fixed(wl_gets32(pkt, 16))
            return
        if object == self.keyboard and self.keyboard != 0:
            if opcode == 0 and size >= 16:  # keymap(format, fd, size)
                var fmt = wl_get32(pkt, 8)
                var ksz = wl_get32(pkt, 12)
                var kfd = self.pop_incoming_fd()
                if fmt == 1 and kfd >= 0 and ksz > 0:
                    var mapped = external_call["mmap", Int](
                        Int(0), Int64(ksz), Int32(1), Int32(2), Int32(kfd), Int64(0)
                    )  # PROT_READ=1, MAP_PRIVATE=2
                    if mapped != 0 and mapped != -1:
                        var ptr = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=mapped)
                        var bytes = List[UInt8](capacity=ksz)
                        for i in range(ksz):
                            var b = ptr[unsafe_offset=i]
                            if b == 0:
                                break
                            bytes.append(b)
                        _ = external_call["munmap", Int32](mapped, Int64(ksz))
                        _ = self.xkb_keymap.parse(String(unsafe_from_utf8=bytes))
                elif fmt == 0:
                    self.xkb_keymap.reset_us_fallback()
                if kfd >= 0:
                    close_fd(kfd)
            elif opcode == 1:
                if size >= 12:
                    self.last_input_serial = wl_get32(pkt, 8)
                self.focused = True
            elif opcode == 2:
                self.focused = False
                self.pending_keys.append(RELEASE_HELD)
                self.pending_key_down.append(False)
            elif opcode == 3 and size >= 24:
                self.last_input_serial = wl_get32(pkt, 8)
                var evdev_code = wl_get32(pkt, 16)
                var is_pressed = wl_get32(pkt, 20) == 1
                var key = evdev_key(evdev_code)
                if key >= 0:
                    self.pending_keys.append(key)
                    self.pending_key_down.append(is_pressed)
                if is_pressed:
                    var utf8 = self.xkb_keymap.translate_key_to_utf8(evdev_code)
                    if utf8.byte_length() > 0:
                        self.pending_text_utf8 += utf8
            elif opcode == 4 and size >= 28:  # modifiers(serial, depressed, latched, locked, group)
                self.xkb_keymap.set_modifiers(
                    wl_get32(pkt, 12),
                    wl_get32(pkt, 16),
                    wl_get32(pkt, 20),
                    wl_get32(pkt, 24),
                )
            return
        if object == self.data_device and self.data_device != 0:
            if opcode == 0 and size >= 12:  # data_offer(new_id)
                self._register_offer(wl_get32(pkt, 8))
            elif opcode == 1 and size >= 28:  # enter(serial, surface, x, y, id)
                var serial = wl_get32(pkt, 8)
                var oid = wl_get32(pkt, 24)
                if self.dnd_offer_id != 0 and self.dnd_offer_id != oid and self.dnd_offer_id != self.selection_offer_id:
                    self._destroy_offer(self.dnd_offer_id)
                self.dnd_serial = serial
                self.dnd_x = wl_fixed(wl_gets32(pkt, 16))
                self.dnd_y = wl_fixed(wl_gets32(pkt, 20))
                self.dnd_offer_id = oid
                if oid != 0:
                    var dnd_mime = self._pick_dnd_mime(oid)
                    var acc_body = List[UInt8]()
                    wl_append32(acc_body, serial)
                    wl_append_nullable_string(acc_body, dnd_mime)
                    _ = self.request(oid, 0, acc_body)  # accept
                    if self.data_device_version >= 3 and dnd_mime.byte_length() > 0:
                        var act_body = List[UInt8]()
                        wl_append32(act_body, 1)  # copy
                        wl_append32(act_body, 1)  # preferred = copy
                        _ = self.request(oid, 4, act_body)  # set_actions
            elif opcode == 2:  # leave
                if self.dnd_offer_id != 0 and self.dnd_offer_id != self.selection_offer_id:
                    self._destroy_offer(self.dnd_offer_id)
                self.dnd_offer_id = 0
            elif opcode == 3 and size >= 20:  # motion(time, x, y)
                self.dnd_x = wl_fixed(wl_gets32(pkt, 12))
                self.dnd_y = wl_fixed(wl_gets32(pkt, 16))
            elif opcode == 4:  # drop
                if self.dnd_offer_id != 0:
                    var drop_oid = self.dnd_offer_id
                    self.dnd_offer_id = 0
                    var dnd_mime = self._pick_dnd_mime(drop_oid)
                    if dnd_mime.byte_length() > 0:
                        var payload = self._read_offer_string(drop_oid, dnd_mime)
                        if dnd_mime == "text/uri-list":
                            var paths = parse_uri_list(payload)
                            for p in paths:
                                self.pending_dropped_files.append(p)
                        elif payload.byte_length() > 0:
                            self.pending_dropped_files.append(payload)
                    if self.data_device_version >= 3:
                        _ = self.empty_request(drop_oid, 3)  # finish
                    if drop_oid != self.selection_offer_id:
                        self._destroy_offer(drop_oid)
            elif opcode == 5 and size >= 12:  # selection(id)
                var new_sel = wl_get32(pkt, 8)
                if self.selection_offer_id != 0 and self.selection_offer_id != new_sel and self.selection_offer_id != self.dnd_offer_id:
                    self._destroy_offer(self.selection_offer_id)
                self.selection_offer_id = new_sel
            return
        if self._has_offer(object):
            if opcode == 0 and size >= 12:  # wl_data_offer.offer(mime_type)
                var mime = wl_get_string(pkt, 8, size)
                if mime.byte_length() > 0:
                    self._add_offer_mime(object, mime)
            return
        if object == self.clipboard_source and self.clipboard_source != 0:
            if opcode == 1:  # wl_data_source.send(mime_type, fd)
                var sfd = self.pop_incoming_fd()
                if sfd >= 0:
                    var bytes = self.clipboard_text.as_bytes()
                    var off = 0
                    while off < len(bytes):
                        var w = Int(
                            external_call["write", Int64](
                                Int32(sfd),
                                bytes.unsafe_ptr().unsafe_offset(off),
                                Int64(len(bytes) - off),
                            )
                        )
                        if w > 0:
                            off += w
                            continue
                        if w < 0 and (last_errno() == EINTR or last_errno() == Int32(11)):
                            continue
                        break
                    close_fd(sfd)
            elif opcode == 2:  # wl_data_source.cancelled
                _ = self.empty_request(self.clipboard_source, 1)  # destroy
                self.clipboard_source = 0
                self.clipboard_owned = False
            return
        if object in self.buffers and opcode == 0:
            for index in range(len(self.buffers)):
                if self.buffers[index] == object:
                    self.buffer_busy[index] = False
                    self.buffer_releases += 1
            return
        if object == self.frame_callback and opcode == 0:
            self.frame_callback = 0
            return
        # Only awaited syncs are kept. Frame callbacks replaced before they fired, which
        # is one per frame while a surface is hidden, end here and are forgotten.
        if opcode == 0:
            for index in range(len(self.awaiting_syncs)):
                if self.awaiting_syncs[index] == object:
                    _ = self.awaiting_syncs.pop(index)
                    self.done_callbacks.append(object)
                    return

    def create_input_devices(mut self):
        if self.seat == 0:
            return
        if (self.seat_capabilities & 1) != 0 and self.pointer == 0:
            self.pointer = self.allocate_id()
            var body = List[UInt8]()
            wl_append32(body, self.pointer)
            _ = self.request(self.seat, 0, body)
        if (self.seat_capabilities & 2) != 0 and self.keyboard == 0:
            self.keyboard = self.allocate_id()
            if self.bridge.active():
                self.bridge.note_keyboard(self.keyboard)
            var body = List[UInt8]()
            wl_append32(body, self.keyboard)
            _ = self.request(self.seat, 1, body)

    def wait_callback(mut self, callback: Int, require_configure: Bool = False) -> Bool:
        var finished = False
        for _ in range(200):
            for done in self.done_callbacks:
                if done == callback and (self.configured or not require_configure):
                    finished = True
            if finished or self.close_requested or self.buffer_rejected:
                break
            _ = self.wait_socket(10)
            self.receive()
        self.forget_callback(callback)
        return finished and self.protocol_errors == 0

    def forget_callback(mut self, callback: Int):
        for index in range(len(self.done_callbacks)):
            if self.done_callbacks[index] == callback:
                _ = self.done_callbacks.pop(index)
                break
        for index in range(len(self.awaiting_syncs)):
            if self.awaiting_syncs[index] == callback:
                _ = self.awaiting_syncs.pop(index)
                break

    def initialize(mut self, width: Int, height: Int) -> Bool:
        if env_flag("FLINT_HEADLESS"):
            return False
        self.width = width
        self.height = height
        self.logical_width = max(1, width)
        self.logical_height = max(1, height)
        self.pending_mouse_x = 0
        self.pending_mouse_y = 0
        self.pending_buttons = List[Int]()
        self.pending_button_down = List[Bool]()
        self.pending_wheel = 0
        self.pending_keys = List[Int]()
        self.pending_key_down = List[Bool]()
        # FLINT_BACKEND=kms|x11|wayland picks one; otherwise Wayland, then X11, and the bare
        # display (DRM/KMS) when neither is advertised and we run on a text console.
        var choice = env_value("FLINT_BACKEND")
        var auto_kms = (
            choice.byte_length() == 0
            and env_value("WAYLAND_DISPLAY").byte_length() == 0
            and env_value("DISPLAY").byte_length() == 0
            and on_virtual_console()
        )
        if choice == "kms" or choice == "KMS" or auto_kms:
            if self.start_kms(width, height):
                return True
            print("FLINT_KMS_UNAVAILABLE " + self.kms.last_error)
            if not auto_kms:
                return False
        if choice == "x11" or choice == "X11":
            if self.x11.initialize(width, height, self.window_title):
                self.using_x11 = True
                self.configured = True
                self.focused = True
                return True
            return False
        if not self.connect_socket():
            if choice == "wayland" or choice == "WAYLAND":
                return False
            if self.x11.initialize(width, height, self.window_title):
                self.using_x11 = True
                self.configured = True
                self.focused = True
                return True
            return False
        if self.wants_gpu_fullscreen:
            self.wants_bridge = True
        if self.wants_bridge and not self.bridge.start(self.fd):
            # The window works without it, on the dmabuf and shared-memory paths.
            print("WAYLAND_BRIDGE_UNAVAILABLE " + self.bridge.error)
        self.registry = self.allocate_id()
        var body = List[UInt8]()
        wl_append32(body, self.registry)
        if not self.request(1, 1, body):
            self.close()
            return False
        if not self.wait_callback(self.sync()):
            self.close()
            return False
        if self.compositor == 0 or self.shm == 0 or self.wm_base == 0:
            self.last_error = "required Wayland globals unavailable"
            self.close()
            return False
        if self.bridge.active():
            self.surface = self.bridge.create_surface()
        else:
            self.surface = self.allocate_id()
            body = List[UInt8]()
            wl_append32(body, self.surface)
            _ = self.request(self.compositor, 0, body)
        self.xdg_surface = self.allocate_id()
        body = List[UInt8]()
        wl_append32(body, self.xdg_surface)
        wl_append32(body, self.surface)
        _ = self.request(self.wm_base, 2, body)
        self.toplevel = self.allocate_id()
        body = List[UInt8]()
        wl_append32(body, self.toplevel)
        _ = self.request(self.xdg_surface, 1, body)
        body = List[UInt8]()
        wl_append_string(body, self.window_title)
        _ = self.request(self.toplevel, 2, body)
        body = List[UInt8]()
        wl_append_string(body, "flint")
        _ = self.request(self.toplevel, 3, body)
        if self.wants_gpu_fullscreen and not env_flag("FLINT_WINDOWED"):
            # Before the first commit, so the first configure is already the fullscreen size.
            body = List[UInt8]()
            wl_append32(body, 0)  # xdg_toplevel.set_fullscreen on any output
            _ = self.request(self.toplevel, 11, body)
        if self.fractional_scale_manager != 0 and self.fractional_scale == 0:
            self.fractional_scale = self.allocate_id()
            body = List[UInt8]()
            wl_append32(body, self.fractional_scale)
            wl_append32(body, self.surface)
            _ = self.request(self.fractional_scale_manager, 1, body)
        if self.decoration_manager != 0 and self.toplevel_decoration == 0:
            self.toplevel_decoration = self.allocate_id()
            body = List[UInt8]()
            wl_append32(body, self.toplevel_decoration)
            wl_append32(body, self.toplevel)
            _ = self.request(self.decoration_manager, 1, body)
            body = List[UInt8]()
            wl_append32(body, 2)  # server-side decoration mode
            _ = self.request(self.toplevel_decoration, 1, body)
        _ = self.empty_request(self.surface, 6)
        body = List[UInt8]()
        wl_append32(body, 0)
        wl_append32(body, 0)
        wl_append32(body, width)
        wl_append32(body, height)
        _ = self.request(self.surface, 2, body)
        if not self.wait_callback(self.sync(), True):
            self.close()
            return False
        _ = self.evdev.enable_hotplug()
        _ = self.evdev.discover_initial()
        return True

    def enable_vulkan_bridge(mut self):
        """Enables `WaylandBridge` before `initialize()` so windowed Wayland sessions can attach zero-copy Vulkan swapchains."""
        self.wants_bridge = True

    def attach_vulkan(mut self, mut vk: VulkanGpu, width: Int, height: Int, prefer_mailbox: Bool = True, prefer_hdr10: Bool = False) -> SwapchainHandle:
        """A swapchain on whatever this window is, so a game presents zero-copy without
        knowing the backend. width x height is the renderer's output size.
        - Wayland: needs `wants_gpu_fullscreen` (or `wants_bridge` / `enable_vulkan_bridge()`) set before initialize(),
          for the Vulkan driver's own connection to the compositor. The swapchain keeps its
          size; the compositor scales it to the window (wp_viewporter), aspect-fit when
          fullscreen, so a fullscreen game can be scanned out with no copy at all.
        - KMS: Vulkan takes the display at a mode of exactly that size (KmsWindow.attach_vulkan).
        - X11: delegates to `X11Window.attach_vulkan()` (`VK_KHR_xcb_surface`).
        An empty handle (`swapchain == 0`) means keep presenting CPU frames with present(),
        which still works. When it is not empty present() does nothing, so draw HUDs
        on the GPU. Call `vk.close_swapchain()` before close()."""
        if self.using_kms:
            self.swapchain = self.kms.attach_vulkan(vk, width, height, prefer_hdr10)
            return SwapchainHandle(
                self.swapchain.surface,
                self.swapchain.swapchain,
                self.swapchain.width,
                self.swapchain.height,
                self.swapchain.format,
                self.swapchain.color_space,
                self.swapchain.present_mode,
                self.swapchain.image_count,
                self.swapchain.is_hdr,
            )
        if self.using_x11:
            self.swapchain = self.x11.attach_vulkan(vk, width, height, prefer_mailbox, prefer_hdr10)
            if self.swapchain.swapchain != 0:
                self.gpu_width = self.swapchain.width
                self.gpu_height = self.swapchain.height
            return SwapchainHandle(
                self.swapchain.surface,
                self.swapchain.swapchain,
                self.swapchain.width,
                self.swapchain.height,
                self.swapchain.format,
                self.swapchain.color_space,
                self.swapchain.present_mode,
                self.swapchain.image_count,
                self.swapchain.is_hdr,
            )
        if not self.bridge.active() or self.bridge.surface_proxy == 0:
            self.swapchain = SwapchainHandle()
            return SwapchainHandle()
        self.swapchain = vk.attach_surface(
            wl_display=self.bridge.display,
            wl_surface=self.bridge.surface_proxy,
            width=width,
            height=height,
            prefer_mailbox=prefer_mailbox,
            prefer_hdr10=prefer_hdr10,
        )
        if self.swapchain.swapchain != 0:
            self.gpu_width = self.swapchain.width
            self.gpu_height = self.swapchain.height
            self.fit_swapchain_to_window()
        return SwapchainHandle(
            self.swapchain.surface,
            self.swapchain.swapchain,
            self.swapchain.width,
            self.swapchain.height,
            self.swapchain.format,
            self.swapchain.color_space,
            self.swapchain.present_mode,
            self.swapchain.image_count,
            self.swapchain.is_hdr,
        )

    def detach_vulkan(mut self, mut vk: VulkanGpu) -> Bool:
        """Undo attach_vulkan: close the swapchain and go back to present() with CPU frames
        (on KMS this sets the window's own mode again). True when present() works again."""
        self.swapchain = SwapchainHandle()
        self.gpu_width = 0
        self.gpu_height = 0
        if self.using_kms:
            return self.kms.detach_vulkan(vk)
        if self.using_x11:
            return self.x11.detach_vulkan(vk)
        vk.close_swapchain()
        return self.fd >= 0

    def presents_on_gpu(self) -> Bool:
        """True once a Vulkan swapchain owns presentation on KMS, Wayland, or X11 (present() is then a no-op)."""
        return (self.using_kms and self.kms.for_vulkan and len(self.kms.fb_ids) == 0) or self.swapchain.vk_swapchain != 0

    def start_kms(mut self, width: Int, height: Int) -> Bool:
        """Draw straight to the display through DRM/KMS (platform/kms.mojo)."""
        self.kms.for_vulkan = self.wants_kms_vulkan
        if not self.kms.initialize(width, height):
            return False
        self.using_kms = True
        self.configured = True
        self.focused = True
        self.fullscreen = True
        self.width = self.kms.width
        self.height = self.kms.height
        return True

    def wait_for_frame(mut self, max_wait_ms: Int = 20):
        """Paces the render loop against a pending `wl_surface.frame` callback via `poll()`
        so fast GPU renderers never busy-spin or discard frames between compositor VSyncs."""
        if self.using_kms:
            self.kms.wait_for_frame(max_wait_ms)
            return
        if self.using_x11 or self.fd < 0 or self.frame_callback == 0 or max_wait_ms <= 0:
            return
        self.receive()
        if self.frame_callback == 0 or self.close_requested:
            return
        var deadline = get_time() + Float64(max_wait_ms) * 0.001
        while self.frame_callback != 0 and not self.close_requested:
            var rem_s = deadline - get_time()
            if rem_s <= 0.0002:
                break
            var rem_ms = max(1, Int(rem_s * 1000.0))
            if not self.wait_socket(rem_ms):
                break
            self.receive()

    def poll(mut self, mut input: InputState):
        if self.using_kms:
            self.kms.poll(input)
            self.close_requested = self.kms.close_requested
            return
        if self.using_x11:
            self.x11.poll(input)
            self.width = self.x11.width
            self.height = self.x11.height
            self.close_requested = self.x11.close_requested
            self.focused = self.x11.focused
            self.fullscreen = self.x11.fullscreen
            return
        self.evdev.poll(input)
        if self.fd < 0:
            return
        self.receive()
        if self.gpu_width > 0:
            self.fit_swapchain_to_window()
        if self.pending_mouse_x >= 0 and self.pending_mouse_y >= 0:
            var mx = self.pending_mouse_x
            var my = self.pending_mouse_y
            # Pointer positions are surface-local: the viewport's size once one is set.
            var surface_w = self.viewport_width if self.viewport_width > 0 else self.width
            var surface_h = self.viewport_height if self.viewport_height > 0 else self.height
            if surface_w > 0 and surface_w != self.logical_width:
                mx = mx * Float32(self.logical_width) / Float32(surface_w)
            if surface_h > 0 and surface_h != self.logical_height:
                my = my * Float32(self.logical_height) / Float32(surface_h)
            if not self.pointer_locked:
                input.move_mouse(mx, my)
            else:
                input.last_mouse_x = mx
                input.last_mouse_y = my
                input.has_mouse_position = True
            self.pending_mouse_x = -1
            self.pending_mouse_y = -1
        for index in range(len(self.pending_buttons)):
            if self.pending_buttons[index] == RELEASE_HELD:
                for button in range(MOUSE_BUTTONS):
                    if input.mouse_down(button):
                        input.set_mouse(button, False)
            else:
                input.set_mouse(self.pending_buttons[index], self.pending_button_down[index])
        self.pending_buttons.clear()
        self.pending_button_down.clear()
        input.add_mouse_delta(self.pending_relative_x, self.pending_relative_y)
        self.pending_relative_x = 0
        self.pending_relative_y = 0
        input.wheel += self.pending_wheel
        self.pending_wheel = 0
        for index in range(len(self.pending_keys)):
            if self.pending_keys[index] == RELEASE_HELD:
                input.release_held()  # package 3.15: past the key map and any toggle latch
            else:
                input.set_key(self.pending_keys[index], self.pending_key_down[index])
        self.pending_keys.clear()
        self.pending_key_down.clear()
        if self.pending_text_utf8.byte_length() > 0:
            input.push_text_utf8(self.pending_text_utf8)
            self.pending_text_utf8 = ""
        for p in self.pending_dropped_files:
            input.push_dropped_file(p)
        self.pending_dropped_files.clear()

    def destroy_shared(mut self):
        for buffer in self.buffers:
            _ = self.empty_request(buffer, 0)
        self.buffers.clear()
        self.buffer_busy.clear()
        if self.pool > 0:
            _ = self.empty_request(self.pool, 1)
            self.pool = 0
        close_fd(self.shared_fd)
        self.shared_fd = -1
        self.shared_size = 0
        self.buffer_width = 0
        self.buffer_height = 0

    def create_shared(mut self, width: Int, height: Int) -> Bool:
        self.destroy_shared()
        var name = "flint-framebuffer"
        self.shared_fd = Int(external_call["memfd_create", Int32](name.as_c_string_span(), UInt32(1)))
        if self.shared_fd < 0:
            self.last_error = "memfd_create failed"
            return False
        self.shared_size = width * height * 4 * 2
        if external_call["ftruncate", Int32](Int32(self.shared_fd), Int64(self.shared_size)) != 0:
            self.last_error = "shared-memory ftruncate failed"
            return False
        self.pool = self.allocate_id()
        var body = List[UInt8]()
        wl_append32(body, self.pool)
        wl_append32(body, self.shared_size)
        var data = List[UInt8](length=8 + len(body), fill=0)
        wl_put32(data, 0, self.shm)
        wl_put16(data, 4, 0)
        wl_put16(data, 6, len(data))
        for index in range(len(body)):
            data[8 + index] = body[index]
        if not self.send_with_fd(data, self.shared_fd):
            self.last_error = "shared-memory pool request failed: " + self.last_error
            return False
        var frame_size = width * height * 4
        for index in range(2):
            var buffer = self.allocate_id()
            body = List[UInt8]()
            wl_append32(body, buffer)
            wl_append32(body, index * frame_size)
            wl_append32(body, width)
            wl_append32(body, height)
            wl_append32(body, width * 4)
            wl_append32(body, 1)
            if not self.request(self.pool, 0, body):
                self.last_error = "shared-memory buffer request failed"
                return False
            self.buffers.append(buffer)
            self.buffer_busy.append(False)
        _ = self.empty_request(self.pool, 1)
        self.pool = 0
        self.buffer_width = width
        self.buffer_height = height
        return True

    def shared_write(
        mut self,
        mut pixels: List[UInt32],
        offset: Int,
        opaque_already_set: Bool = False,
        pixel_count: Int = -1,
    ) -> Bool:
        var n_px = len(pixels) if pixel_count < 0 else min(len(pixels), pixel_count)
        if not opaque_already_set:
            # Keep the unused XRGB byte opaque (8-wide SIMD!).
            var base_addr = Int(pixels.unsafe_ptr())
            var alpha_v = SIMD[DType.uint32, 8](0xff000000)
            var idx = 0
            while idx + 8 <= n_px:
                var p_addr = base_addr + idx * 4
                var v = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=p_addr).unsafe_load[width=8, alignment=4]()
                Pointer[UInt32, MutAnyOrigin](unsafe_from_address=p_addr).unsafe_store[alignment=4](v | alpha_v)
                idx += 8
            while idx < n_px:
                pixels[idx] = pixels[idx] | UInt32(0xff000000)
                idx += 1
        var written = 0
        var size = n_px * 4
        while written < size:
            var count = Int(external_call["pwrite", Int64](Int32(self.shared_fd), pixels.unsafe_ptr().unsafe_offset(written // 4), Int64(size - written), Int64(offset + written)))
            if count <= 0:
                return False
            written += count
        return True

    def present(
        mut self,
        mut pixels: List[UInt32],
        source_width: Int,
        source_height: Int,
        opaque_already_set: Bool = False,
    ):
        if self.swapchain.vk_swapchain != 0:
            return
        if self.using_kms:
            self.kms.present(pixels, source_width, source_height)
            self.close_requested = self.kms.close_requested
            return
        if self.using_x11:
            self.x11.present(pixels, source_width, source_height)
            self.width = self.x11.width
            self.height = self.x11.height
            self.close_requested = self.x11.close_requested
            return
        if self.fd < 0 or not self.configured or source_width <= 0 or source_height <= 0:
            return
        # A callback that arrived during this frame must count, or every second frame is dropped.
        self.receive()
        if self.frame_callback != 0:
            if get_time() - self.frame_requested_at < 0.25:
                return
            # A compositor may withhold callbacks for an occluded surface.
            # Permit a fresh frame rather than stalling presentation forever.
            self.frame_callback = 0
        # When `wp_viewporter` is available, keep the shm pool at `source_width x source_height`
        # so the Wayland compositor stretches it on the GPU in 0 ms instead of scaling on CPU!
        var tgt_w = source_width
        var tgt_h = source_height
        if self.viewporter == 0:
            var fb = self.framebuffer_size()
            tgt_w = fb[0]
            tgt_h = fb[1]
            var target_bscale = max(1, self.preferred_buffer_scale)
            if self.applied_buffer_scale != target_bscale:
                var bs_body = List[UInt8]()
                wl_append32(bs_body, target_bscale)
                _ = self.request(self.surface, 8, bs_body)  # wl_surface.set_buffer_scale
                self.applied_buffer_scale = target_bscale
        if self.shared_fd < 0 or self.buffer_width != tgt_w or self.buffer_height != tgt_h:
            if not self.create_shared(tgt_w, tgt_h):
                self.protocol_errors += 1
                self.last_error = "shared-memory allocation failed: " + self.last_error
                return
        var selected = -1
        for index in range(len(self.buffer_busy)):
            if not self.buffer_busy[index]:
                selected = index
                break
        if selected < 0:
            return
        # Populate the shared buffer before the commit that makes it visible.
        var frame_size = tgt_w * tgt_h * 4
        var write_ok: Bool
        if source_width == tgt_w and source_height == tgt_h:
            write_ok = self.shared_write(pixels, selected * frame_size, opaque_already_set, tgt_w * tgt_h)
        else:
            var scaled = List[UInt32](length=tgt_w * tgt_h, fill=0)
            for y in range(tgt_h):
                var source_y = min(source_height - 1, y * source_height // tgt_h)
                for x in range(tgt_w):
                    var source_x = min(source_width - 1, x * source_width // tgt_w)
                    scaled[y * tgt_w + x] = pixels[source_y * source_width + source_x]
            write_ok = self.shared_write(scaled, selected * frame_size, opaque_already_set, tgt_w * tgt_h)
        if not write_ok:
            self.protocol_errors += 1
            self.last_error = "shared-memory pixel write failed"
            return
        self.commit_buffer(selected)

    def scale_to_window(mut self):
        """Let the compositor stretch buffers of any size over the whole window."""
        self.set_viewport_destination(self.width, self.height)

    def set_viewport_destination(mut self, width: Int, height: Int):
        """wp_viewport.set_destination: the surface's size on screen, whatever its buffers'
        size (the compositor scales them on the GPU). Applies at the surface's next commit."""
        if self.viewporter == 0 or width <= 0 or height <= 0:
            return
        if self.viewport_width == width and self.viewport_height == height:
            return
        var body = List[UInt8]()
        if self.viewport == 0:
            self.viewport = self.allocate_id()
            wl_append32(body, self.viewport)
            wl_append32(body, self.surface)
            _ = self.request(self.viewporter, 1, body)
            body = List[UInt8]()
        wl_append32(body, width)
        wl_append32(body, height)
        _ = self.request(self.viewport, 2, body)
        self.viewport_width = width
        self.viewport_height = height

    def fit_swapchain_to_window(mut self):
        """Show the attach_vulkan swapchain over the window: stretched to it when windowed,
        aspect-fit when fullscreen (the compositor centres it on black). Vulkan's next
        present commits it; no frame is copied or redrawn."""
        if self.gpu_width <= 0 or self.gpu_height <= 0 or self.width <= 0 or self.height <= 0:
            return
        var before_w = self.viewport_width
        var before_h = self.viewport_height
        if self.fullscreen:
            var rect = fit_rect(self.gpu_width, self.gpu_height, self.width, self.height)
            self.set_viewport_destination(rect[2], rect[3])
        else:
            self.set_viewport_destination(self.width, self.height)
        if env_flag("FLINT_DEBUG_PRESENT") and (before_w != self.viewport_width or before_h != self.viewport_height):
            print(
                "FLINT_WAYLAND_GPU_VIEWPORT swapchain " + String(self.gpu_width) + "x" + String(self.gpu_height)
                + " -> " + String(self.viewport_width) + "x" + String(self.viewport_height)
                + " in a " + String(self.width) + "x" + String(self.height)
                + (" fullscreen" if self.fullscreen else " window")
                + (" (no wp_viewporter)" if self.viewporter == 0 else "")
            )

    def commit_buffer(mut self, selected: Int):
        """Attach a filled buffer, request the next frame callback, and commit."""
        self.scale_to_window()
        var body = List[UInt8]()
        wl_append32(body, self.buffers[selected])
        wl_append32(body, 0)
        wl_append32(body, 0)
        _ = self.request(self.surface, 1, body)
        body = List[UInt8]()
        wl_append32(body, 0)
        wl_append32(body, 0)
        wl_append32(body, self.width)
        wl_append32(body, self.height)
        _ = self.request(self.surface, 2, body)
        self.frame_callback = self.allocate_id()
        self.frame_requested_at = get_time()
        body = List[UInt8]()
        wl_append32(body, self.frame_callback)
        _ = self.request(self.surface, 3, body)
        _ = self.empty_request(self.surface, 6)
        self.buffer_busy[selected] = True
        self.commits += 1

    def create_dmabuf_buffer(mut self, fd: Int, width: Int, height: Int, stride: Int) -> Int:
        """Wrap a linear XRGB8888 GPU buffer in a wl_buffer. Returns its index, or -1."""
        if self.dmabuf == 0:
            self.last_error = "compositor has no zwp_linux_dmabuf_v1"
            return -1
        var params = self.allocate_id()
        self.dmabuf_objects.append(params)
        var body = List[UInt8]()
        wl_append32(body, params)
        if not self.request(self.dmabuf, 1, body):
            return -1
        # zwp_linux_buffer_params_v1.add: the fd itself travels as SCM_RIGHTS.
        var add = List[UInt8](length=28, fill=0)
        wl_put32(add, 0, params)
        wl_put16(add, 4, 1)
        wl_put16(add, 6, len(add))
        wl_put32(add, 16, stride)
        if not self.send_with_fd(add, fd):
            return -1
        var buffer = self.allocate_id()
        self.dmabuf_objects.append(buffer)
        body = List[UInt8]()
        wl_append32(body, buffer)
        wl_append32(body, width)
        wl_append32(body, height)
        wl_append32(body, 0x34325258)  # DRM_FORMAT_XRGB8888
        wl_append32(body, 0)
        if not self.request(params, 3, body):  # create_immed
            return -1
        _ = self.empty_request(params, 0)
        self.buffers.append(buffer)
        self.buffer_busy.append(False)
        return len(self.buffers) - 1

    def set_pointer_lock(mut self, active: Bool):
        """Hold the cursor still and read relative motion while active.

        Without compositor support, or while a lock is refused, ordinary pointer
        motion keeps driving the camera.
        """
        if self.using_kms:
            self.kms.set_pointer_lock(active)
            self.pointer_locked = active
            return
        if self.fd < 0 or self.pointer == 0 or self.surface == 0 or self.pointer_constraints == 0 or self.relative_manager == 0:
            return
        if active and self.locked_pointer == 0:
            if self.relative_pointer == 0:
                self.relative_pointer = self.allocate_id()
                var relative_body = List[UInt8]()
                wl_append32(relative_body, self.relative_pointer)
                wl_append32(relative_body, self.pointer)
                _ = self.request(self.relative_manager, 1, relative_body)
            self.locked_pointer = self.allocate_id()
            var body = List[UInt8]()
            wl_append32(body, self.locked_pointer)
            wl_append32(body, self.surface)
            wl_append32(body, self.pointer)
            wl_append32(body, 0)  # no region: lock anywhere on the surface
            wl_append32(body, 2)  # persistent lifetime
            _ = self.request(self.pointer_constraints, 1, body)
            self.pointer_locked = True
            self.hide_cursor()
        elif not active and self.locked_pointer != 0:
            _ = self.empty_request(self.locked_pointer, 0)
            self.locked_pointer = 0
            self.pointer_locked = False

    def hide_cursor(mut self):
        if self.using_kms:
            self.kms.hide_cursor()
            return
        if self.pointer == 0 or self.pointer_enter_serial == 0:
            return
        var body = List[UInt8]()
        wl_append32(body, self.pointer_enter_serial)
        wl_append32(body, 0)
        wl_append32(body, 0)
        wl_append32(body, 0)
        _ = self.request(self.pointer, 0, body)

    def toggle_fullscreen(mut self):
        if self.using_kms:
            return  # a bare display is always fullscreen
        if self.using_x11:
            self.x11.toggle_fullscreen()
            self.fullscreen = self.x11.fullscreen
            return
        if self.toplevel == 0:
            return
        if self.fullscreen:
            # xdg_toplevel.unset_fullscreen
            _ = self.empty_request(self.toplevel, 12)
        else:
            # xdg_toplevel.set_fullscreen, with a null output (any output).
            var body = List[UInt8]()
            wl_append32(body, 0)
            _ = self.request(self.toplevel, 11, body)

    def content_scale(self) -> Float32:
        if self.using_kms:
            return 1.0
        if self.using_x11:
            return self.x11.content_scale()
        if self.fractional_scale != 0 and self.fractional_scale_num > 0:
            return Float32(self.fractional_scale_num) / Float32(120.0)
        return Float32(max(1, self.preferred_buffer_scale))

    def framebuffer_size(self) -> Tuple[Int, Int]:
        if self.using_kms:
            return (max(1, self.kms.width), max(1, self.kms.height))
        if self.using_x11:
            return self.x11.framebuffer_size()
        var s = self.content_scale()
        return (
            max(1, Int(Float32(self.width) * s + Float32(0.5))),
            max(1, Int(Float32(self.height) * s + Float32(0.5))),
        )

    def supports_fractional_scale_protocol(self) -> Bool:
        return True

    def supports_xdg_decoration_protocol(self) -> Bool:
        return True

    def supports_data_device_clipboard(self) -> Bool:
        return True

    def clipboard_set(mut self, text: String):
        if self.using_x11:
            self.x11.clipboard_set(text)
            return
        if self.using_kms:
            self.kms.clipboard_set(text)
            return
        self.clipboard_text = text
        self.clipboard_owned = True
        if self.fd < 0 or self.data_device_manager == 0 or self.data_device == 0:
            return
        if self.clipboard_source != 0:
            _ = self.empty_request(self.clipboard_source, 1)  # destroy old source
            self.clipboard_source = 0
        self.clipboard_source = self.allocate_id()
        if self.bridge.active():
            self.bridge.note_data_source(self.clipboard_source)
        var cbody = List[UInt8]()
        wl_append32(cbody, self.clipboard_source)
        _ = self.request(self.data_device_manager, 0, cbody)  # create_data_source
        for mime in ["text/plain;charset=utf-8", "UTF8_STRING", "text/plain"]:
            var obody = List[UInt8]()
            wl_append_string(obody, String(mime))
            _ = self.request(self.clipboard_source, 0, obody)  # wl_data_source.offer
        var sbody = List[UInt8]()
        wl_append32(sbody, self.clipboard_source)
        wl_append32(sbody, self.last_input_serial)
        _ = self.request(self.data_device, 1, sbody)  # set_selection

    def clipboard_get(mut self) -> String:
        if self.using_x11:
            return self.x11.clipboard_get()
        if self.using_kms:
            return self.kms.clipboard_get()
        if self.clipboard_owned:
            return self.clipboard_text
        if self.fd < 0 or self.selection_offer_id == 0:
            return ""
        var mime = self._pick_text_mime(self.selection_offer_id)
        if mime.byte_length() == 0:
            return ""
        return self._read_offer_string(self.selection_offer_id, mime)

    def take_dropped_files(mut self) -> List[String]:
        var out = self.pending_dropped_files.copy()
        self.pending_dropped_files.clear()
        return out^

    def poll_events(mut self, mut input: InputState):
        self.poll(input)

    def present_argb(mut self, pixels: List[UInt32], width: Int, height: Int):
        var copy_px = pixels.copy()
        self.present(copy_px, width, height)

    def should_close(self) -> Bool:
        return self.close_requested

    def shutdown(mut self):
        self.close()

    def reconnect(mut self) -> Bool:
        """Open the window again on a new connection, after a fatal error about buffers.

        A compositor hangs up on any protocol error, and with the connection go the
        surface and every object on it. Keys held at that moment will never be released.
        """
        var width = self.width
        var height = self.height
        var fullscreen = self.fullscreen
        self.close()
        self = WaylandWindow()
        if not self.initialize(width, height):
            return False
        if fullscreen:
            self.toggle_fullscreen()
        self.pending_keys.append(RELEASE_HELD)
        self.pending_key_down.append(False)
        self.pending_buttons.append(RELEASE_HELD)
        self.pending_button_down.append(False)
        return True

    def close(mut self):
        self.swapchain = SwapchainHandle()
        self.gpu_width = 0
        self.gpu_height = 0
        self.evdev.close()
        self.kms.close()
        self.using_kms = False
        self.x11.close()
        self.using_x11 = False
        self.destroy_shared()
        if self.fd >= 0:
            if self.toplevel_decoration > 0:
                _ = self.empty_request(self.toplevel_decoration, 0)
            if self.fractional_scale > 0:
                _ = self.empty_request(self.fractional_scale, 0)
            if self.clipboard_source > 0:
                _ = self.empty_request(self.clipboard_source, 1)
            if self.selection_offer_id > 0:
                _ = self.empty_request(self.selection_offer_id, 2)
            if self.dnd_offer_id > 0 and self.dnd_offer_id != self.selection_offer_id:
                _ = self.empty_request(self.dnd_offer_id, 2)
            if self.data_device > 0 and self.data_device_version >= 2:
                _ = self.empty_request(self.data_device, 2)
            if self.toplevel > 0:
                _ = self.empty_request(self.toplevel, 0)
            if self.xdg_surface > 0:
                _ = self.empty_request(self.xdg_surface, 0)
            if self.surface > 0:
                _ = self.empty_request(self.surface, 0)
            if self.pointer > 0:
                _ = self.empty_request(self.pointer, 1)
            if self.keyboard > 0:
                _ = self.empty_request(self.keyboard, 0)
            if self.seat > 0 and self.seat_version >= 5:
                _ = self.empty_request(self.seat, 3)
            if self.wm_base > 0:
                _ = self.empty_request(self.wm_base, 0)
        self.bridge.stop()
        for pfd in self.pending_fds:
            close_fd(pfd)
        self.pending_fds.clear()
        close_fd(self.fd)
        self.fd = -1
        self.frame_callback = 0
        self.frame_requested_at = 0
        self.viewport = 0
        self.viewport_width = 0
        self.viewport_height = 0
        self.fractional_scale = 0
        self.toplevel_decoration = 0
        self.data_device = 0
        self.clipboard_source = 0
        self.clipboard_owned = False
        self.selection_offer_id = 0
        self.dnd_offer_id = 0
        self.offer_ids.clear()
        self.offer_mimes.clear()
        self.toplevel = 0
        self.xdg_surface = 0
        self.surface = 0
        self.pointer = 0
        self.keyboard = 0
        self.seat = 0
        self.wm_base = 0
