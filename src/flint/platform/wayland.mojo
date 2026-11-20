"""Direct Wayland/XDG-shell client using Linux sockets and shared memory.

No libwayland, XKB, EGL, OpenGL, or native compatibility shim is used. The one
exception is SD_PRESENT=swapchain, where the Vulkan driver needs libwayland objects
for this window: see platform/wayland_bridge.mojo. All this code runs the same then.
"""
from std.ffi import external_call
from flint.platform.input import InputState
from flint.platform.wayland_bridge import WaylandBridge
from flint.platform.x11 import X11Window
from flint.platform.kms import KmsWindow, on_virtual_console, fit_rect
from flint.platform.vulkan import VulkanGpu, SwapchainHandle
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.os import env_value, env_flag, close_fd, get_time

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
    return -1


struct WaylandWindow(Copyable, Movable):
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
    var x11_xcb_lib: Int
    var x11_xcb_conn: Int
    var x11_xcb_disconnect: Int

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
        self.x11_xcb_lib = 0
        self.x11_xcb_conn = 0
        self.x11_xcb_disconnect = 0

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

    def receive(mut self):
        if self.bridge.active():
            self.bridge.read_into(self.incoming)
            if self.bridge.failed():
                self.close_requested = True
            self.handle_incoming()
            return
        var buffer = List[UInt8](length=16384, fill=0)
        while True:
            var count = Int(external_call["recv", Int64](Int32(self.fd), buffer.unsafe_ptr(), Int64(len(buffer)), Int32(64)))
            if count == 0:
                self.close_requested = True
                break
            if count < 0:
                break
            for index in range(count):
                self.incoming.append(buffer[index])
            if count < len(buffer):
                break
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
                return
            if len(self.incoming) < size:
                break
            self.handle_event(object, opcode, size)
            var rem = len(self.incoming) - size
            if rem == 0:
                self.incoming.clear()
            else:
                for i in range(rem):
                    self.incoming[i] = self.incoming[size + i]
                self.incoming.resize(rem, 0)

    def handle_event(mut self, object: Int, opcode: Int, size: Int):
        if object == 1 and opcode == 0:
            self.protocol_errors += 1
            var culprit = wl_get32(self.incoming, 8) if size >= 12 else 0
            var message = "wl_display error"
            # wl_display.error carries object id, error code, then the message.
            if size >= 24:
                var length = wl_get32(self.incoming, 16)
                if length > 0 and 20 + length <= size:
                    var bytes = List[UInt8]()
                    for index in range(length - 1):
                        bytes.append(self.incoming[20 + index])
                    message = "object " + String(wl_get32(self.incoming, 8)) + " code " + String(wl_get32(self.incoming, 12)) + ": " + String(unsafe_from_utf8=bytes)
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
            var name = wl_get32(self.incoming, 8)
            var length = wl_get32(self.incoming, 12)
            if length <= 0 or 16 + length > size:
                return
            var bytes = List[UInt8]()
            for index in range(length - 1):
                bytes.append(self.incoming[16 + index])
            var interface = String(unsafe_from_utf8=bytes)
            var version_offset = 16 + ((length + 3) & ~3)
            if version_offset + 4 > size:
                return
            var version = wl_get32(self.incoming, version_offset)
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
            return
        if object == self.wm_base and opcode == 0:
            var body = List[UInt8]()
            wl_append32(body, wl_get32(self.incoming, 8))
            _ = self.request(self.wm_base, 3, body)
            return
        if object == self.xdg_surface and opcode == 0:
            var body = List[UInt8]()
            wl_append32(body, wl_get32(self.incoming, 8))
            _ = self.request(self.xdg_surface, 4, body)
            self.configured = True
            return
        if object == self.toplevel and opcode == 0:
            var width = wl_get32(self.incoming, 8)
            var height = wl_get32(self.incoming, 12)
            if width > 0 and height > 0 and (width != self.width or height != self.height):
                self.width = width
                self.height = height
            self.fullscreen = False
            self.focused = False
            if size >= 20:
                var states_length = wl_get32(self.incoming, 16)
                var count = min((size - 20) // 4, states_length // 4)
                for index in range(count):
                    var state = wl_get32(self.incoming, 20 + index * 4)
                    if state == 2:
                        self.fullscreen = True
                    elif state == 4:
                        self.focused = True
            return
        if object == self.toplevel and opcode == 1:
            self.close_requested = True
            return
        if object == self.seat and opcode == 0:
            self.seat_capabilities = wl_get32(self.incoming, 8)
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
                self.pending_relative_x += wl_fixed(wl_gets32(self.incoming, 24))
                self.pending_relative_y += wl_fixed(wl_gets32(self.incoming, 28))
            return
        if object == self.pointer:
            if opcode == 0 and size >= 16:
                self.pointer_enter_serial = wl_get32(self.incoming, 8)
                if self.pointer_locked:
                    self.hide_cursor()
            elif opcode == 1:
                self.pending_buttons.append(RELEASE_HELD)
                self.pending_button_down.append(False)
            elif opcode == 2 and size >= 20:
                self.pending_mouse_x = wl_fixed(wl_gets32(self.incoming, 12))
                self.pending_mouse_y = wl_fixed(wl_gets32(self.incoming, 16))
            elif opcode == 3 and size >= 24:
                var button = wl_get32(self.incoming, 16) - 272  # BTN_LEFT, BTN_RIGHT, BTN_MIDDLE
                if button >= 0 and button < MOUSE_BUTTONS:
                    self.pending_buttons.append(button)
                    self.pending_button_down.append(wl_get32(self.incoming, 20) == 1)
            elif opcode == 4 and size >= 20:
                var axis = wl_get32(self.incoming, 12)
                if axis == 0:
                    self.pending_wheel -= wl_fixed(wl_gets32(self.incoming, 16))
            return
        if object == self.keyboard:
            if opcode == 1:
                self.focused = True
            elif opcode == 2:
                self.focused = False
                self.pending_keys.append(RELEASE_HELD)
                self.pending_key_down.append(False)
            elif opcode == 3 and size >= 24:
                var key = evdev_key(wl_get32(self.incoming, 16))
                if key >= 0:
                    self.pending_keys.append(key)
                    self.pending_key_down.append(wl_get32(self.incoming, 20) == 1)
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
        return True

    def enable_vulkan_bridge(mut self):
        """Enables `WaylandBridge` before `initialize()` so windowed Wayland sessions can attach zero-copy Vulkan swapchains."""
        self.wants_bridge = True

    def _ensure_xcb_connection(mut self) -> Bool:
        """Lazily opens `libxcb.so.1` and connects an `xcb_connection_t*` for `VK_KHR_xcb_surface`."""
        if self.x11_xcb_conn != 0:
            return True
        if self.x11_xcb_lib == 0:
            self.x11_xcb_lib = dl_open("libxcb.so.1")
        if self.x11_xcb_lib == 0:
            return False
        var fn_connect = dl_sym(self.x11_xcb_lib, "xcb_connect")
        var fn_has_err = dl_sym(self.x11_xcb_lib, "xcb_connection_has_error")
        self.x11_xcb_disconnect = dl_sym(self.x11_xcb_lib, "xcb_disconnect")
        if fn_connect == 0 or fn_has_err == 0 or self.x11_xcb_disconnect == 0:
            _ = dl_close(self.x11_xcb_lib)
            self.x11_xcb_lib = 0
            self.x11_xcb_disconnect = 0
            return False
        comptime XcbConnect = def(Int, Int) thin abi("C") -> Int
        comptime XcbHasError = def(Int) thin abi("C") -> Int32
        var conn = Pointer(to=fn_connect).unsafe_bitcast[XcbConnect]()[unsafe_offset=0](0, 0)
        if conn == 0:
            return False
        var err = Pointer(to=fn_has_err).unsafe_bitcast[XcbHasError]()[unsafe_offset=0](conn)
        if err != 0:
            comptime XcbDisconnect = def(Int) thin abi("C") -> None
            var fn_disc = self.x11_xcb_disconnect
            Pointer(to=fn_disc).unsafe_bitcast[XcbDisconnect]()[unsafe_offset=0](conn)
            return False
        self.x11_xcb_conn = conn
        return True

    def _close_xcb_connection(mut self):
        if self.x11_xcb_conn != 0 and self.x11_xcb_disconnect != 0:
            comptime XcbDisconnect = def(Int) thin abi("C") -> None
            var fn_disc = self.x11_xcb_disconnect
            Pointer(to=fn_disc).unsafe_bitcast[XcbDisconnect]()[unsafe_offset=0](self.x11_xcb_conn)
        self.x11_xcb_conn = 0
        self.x11_xcb_disconnect = 0
        if self.x11_xcb_lib != 0:
            _ = dl_close(self.x11_xcb_lib)
            self.x11_xcb_lib = 0

    def attach_vulkan(mut self, mut vk: VulkanGpu, width: Int, height: Int, prefer_mailbox: Bool = True, prefer_hdr10: Bool = False) -> SwapchainHandle:
        """A swapchain on whatever this window is, so a game presents zero-copy without
        knowing the backend. width x height is the renderer's output size.
        - Wayland: needs `wants_gpu_fullscreen` (or `wants_bridge` / `enable_vulkan_bridge()`) set before initialize(),
          for the Vulkan driver's own connection to the compositor. The swapchain keeps its
          size; the compositor scales it to the window (wp_viewporter), aspect-fit when
          fullscreen, so a fullscreen game can be scanned out with no copy at all.
        - KMS: Vulkan takes the display at a mode of exactly that size (KmsWindow.attach_vulkan).
        - X11: opens a `libxcb.so.1` connection and attaches `VK_KHR_xcb_surface` against `self.x11.window_id`.
        An empty handle (`swapchain == 0`) means keep presenting CPU frames with present(),
        which still works. When it is not empty present() does nothing, so draw HUDs
        on the GPU. Call `vk.close_swapchain()` before close()."""
        if self.using_kms:
            var sc_kms = self.kms.attach_vulkan(vk, width, height, prefer_hdr10)
            self.swapchain = sc_kms.copy()
            return sc_kms^
        if self.using_x11:
            if not self.x11.active() or not self._ensure_xcb_connection():
                self.swapchain = SwapchainHandle()
                return SwapchainHandle()
            var sc_x11 = vk.attach_surface(
                xcb_conn=self.x11_xcb_conn,
                xcb_win=self.x11.window_id,
                width=width,
                height=height,
                prefer_mailbox=prefer_mailbox,
                prefer_hdr10=prefer_hdr10,
            )
            self.swapchain = sc_x11.copy()
            if sc_x11.swapchain != 0:
                self.gpu_width = sc_x11.width
                self.gpu_height = sc_x11.height
            return sc_x11^
        if not self.bridge.active() or self.bridge.surface_proxy == 0:
            self.swapchain = SwapchainHandle()
            return SwapchainHandle()
        var sc = vk.attach_surface(
            wl_display=self.bridge.display,
            wl_surface=self.bridge.surface_proxy,
            width=width,
            height=height,
            prefer_mailbox=prefer_mailbox,
            prefer_hdr10=prefer_hdr10,
        )
        self.swapchain = sc.copy()
        if sc.swapchain != 0:
            self.gpu_width = sc.width
            self.gpu_height = sc.height
            self.fit_swapchain_to_window()
        return sc^

    def detach_vulkan(mut self, mut vk: VulkanGpu) -> Bool:
        """Undo attach_vulkan: close the swapchain and go back to present() with CPU frames
        (on KMS this sets the window's own mode again). True when present() works again."""
        self.swapchain = SwapchainHandle()
        if self.using_kms:
            return self.kms.detach_vulkan(vk)
        vk.close_swapchain()
        if self.using_x11:
            self._close_xcb_connection()
        self.gpu_width = 0
        self.gpu_height = 0
        return self.fd >= 0 or self.using_x11

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
        input.poll_gamepad()
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
        var tgt_w = source_width if self.viewporter != 0 else self.width
        var tgt_h = source_height if self.viewporter != 0 else self.height
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

    def reconnect(mut self) -> Bool:
        """Open the window again on a new connection, after a fatal error about buffers.

        A compositor hangs up on any protocol error, and with the connection go the
        surface and every object on it. Keys held at that moment will never be released.
        """
        var width = self.width
        var height = self.height
        var fullscreen = self.fullscreen
        self.bridge.stop()
        close_fd(self.fd)
        close_fd(self.shared_fd)
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
        if self.using_kms:
            self.kms.close()
            self.using_kms = False
            return
        if self.using_x11:
            self._close_xcb_connection()
            self.x11.close()
            self.using_x11 = False
            return
        if self.fd >= 0:
            self.destroy_shared()
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
        close_fd(self.fd)
        self.fd = -1
        self.frame_callback = 0
        self.frame_requested_at = 0
        self.viewport = 0
        self.viewport_width = 0
        self.viewport_height = 0
        self.toplevel = 0
        self.xdg_surface = 0
        self.surface = 0
        self.pointer = 0
        self.keyboard = 0
