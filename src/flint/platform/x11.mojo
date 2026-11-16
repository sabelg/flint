"""Direct X11 core wire-protocol window & input backend over Linux AF_UNIX sockets.

Speaks the binary X11R6 protocol directly to `/tmp/.X11-unix/X<N>` using standard
`.Xauthority` (`MIT-MAGIC-COOKIE-1`) authentication — zero `libX11`, `libxcb`, XKB,
GLX, or C/C++ wrapper libraries are linked.
"""
from std.ffi import external_call
from flint.platform.input import InputState
from flint.platform.os import env_value, env_now, env_flag, read_bytes, close_fd, get_time

comptime X11_RELEASE_HELD: Int = -1
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


def x11_be16(data: List[UInt8], offset: Int) -> Int:
    return (Int(data[offset]) << 8) | Int(data[offset + 1])


def x11_evdev_key(key: Int) -> Int:
    """Maps Linux evdev keycodes (X11 keycode - 8) to Flint physical key slots."""
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


struct X11Window(Copyable, Movable):
    var fd: Int
    var width: Int
    var height: Int
    var logical_width: Int
    var logical_height: Int
    var close_requested: Bool
    var focused: Bool
    var fullscreen: Bool
    var rid_base: Int
    var rid_mask: Int
    var next_rid: Int
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
    var pending_buttons: List[Int]
    var pending_button_down: List[Bool]
    var pending_wheel: Float32
    var pending_keys: List[Int]
    var pending_key_down: List[Bool]

    def __init__(out self):
        self.fd = -1
        self.width = 1280
        self.height = 720
        self.logical_width = 1280
        self.logical_height = 720
        self.close_requested = False
        self.focused = True
        self.fullscreen = False
        self.rid_base = 0
        self.rid_mask = 0
        self.next_rid = 1
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
        self.pending_buttons = List[Int]()
        self.pending_button_down = List[Bool]()
        self.pending_wheel = 0
        self.pending_keys = List[Int]()
        self.pending_key_down = List[Bool]()

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
                external_call["write", Int64](
                    Int32(self.fd),
                    ptr,
                    Int64(length - offset),
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

    def recv_exact(mut self, count: Int) -> List[UInt8]:
        var out = List[UInt8](length=count, fill=0)
        var got = 0
        var retries = 0
        while got < count:
            if not self.wait_io(1, 2000):
                retries += 1
                if retries > 2:
                    return List[UInt8]()
                continue
            var n = Int(
                external_call["recv", Int64](
                    Int32(self.fd),
                    out.unsafe_ptr().unsafe_offset(got),
                    Int64(count - got),
                    Int32(0),
                )
            )
            if n > 0:
                got += n
                continue
            if n < 0:
                var err_ptr = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()
                var err = Int(err_ptr[unsafe_offset=0])
                if err == 4 or err == 11:
                    continue
            return List[UInt8]()
        return out^

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
        if not self.write_all(req, len(req)):
            return 0

        for _ in range(64):
            var pkt = self.recv_exact(32)
            if len(pkt) < 32:
                return 0
            if pkt[0] == 1:  # Reply
                var extra_words = x11_get32(pkt, 4)
                if extra_words > 0:
                    _ = self.recv_exact(extra_words * 4)
                return x11_get32(pkt, 8)
            if pkt[0] == 0:  # Error
                return 0
            # Queue any async event that arrived during startup
            for b in pkt:
                self.incoming.append(b)
        return 0

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
        _ = self.write_all(req, len(req))

    def initialize(mut self, width: Int, height: Int, title: String = "FLINT ENGINE") -> Bool:
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

        _ = external_call["signal", Int](Int32(13), Int(1))  # SIGPIPE, SIG_IGN
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
        var vendor_len = x11_get16(body, 16)
        self.max_req_words = max(4096, x11_get16(body, 18))
        var num_screens = Int(body[20])
        var num_formats = Int(body[21])
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
        # KeyPress|KeyRelease|ButtonPress|ButtonRelease|PointerMotion|Exposure|StructureNotify|FocusChange
        x11_put32(cw, 36, 0x0022804F)
        if not self.write_all(cw, len(cw)):
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
        if not self.write_all(gc, len(gc)):
            self.close()
            return False

        # 3. Intern Window Manager Atoms (`WM_DELETE_WINDOW` + `_NET_WM_STATE_FULLSCREEN`)
        self.wm_protocols = self.intern_atom("WM_PROTOCOLS")
        self.wm_delete_window = self.intern_atom("WM_DELETE_WINDOW")
        self.net_wm_state = self.intern_atom("_NET_WM_STATE")
        self.net_wm_state_fullscreen = self.intern_atom("_NET_WM_STATE_FULLSCREEN")

        self.set_title(title)

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
            _ = self.write_all(prop, len(prop))

        # 4. MapWindow (opcode 8)
        var mw = List[UInt8](length=8, fill=0)
        mw[0] = 8
        x11_put16(mw, 2, 2)
        x11_put32(mw, 4, self.window_id)
        if not self.write_all(mw, len(mw)):
            self.close()
            return False

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

    def process_events(mut self):
        var pos = 0
        while pos + 32 <= len(self.incoming):
            var raw_type = Int(self.incoming[pos])
            if raw_type == 1 or (raw_type & 0x7F) == 35:
                # Unexpected async reply or GenericEvent: consume extra words if any
                var extra = x11_get32(self.incoming, pos + 4) * 4
                if pos + 32 + extra > len(self.incoming):
                    break
                pos += 32 + extra
                continue

            var code = raw_type & 0x7F
            var detail = Int(self.incoming[pos + 1])

            if code == 2 or code == 3:
                # Filter X11 synthetic key auto-repeat: KeyRelease immediately followed by KeyPress at identical timestamp
                if code == 3 and pos + 64 <= len(self.incoming):
                    var next_code = Int(self.incoming[pos + 32]) & 0x7F
                    var next_detail = Int(self.incoming[pos + 33])
                    if next_code == 2 and next_detail == detail:
                        if x11_get32(self.incoming, pos + 4) == x11_get32(self.incoming, pos + 36):
                            pos += 64
                            continue
                var key = x11_evdev_key(detail - 8)
                if key >= 0:
                    self.pending_keys.append(key)
                    self.pending_key_down.append(code == 2)
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
                self.pending_mouse_x = Float32(x11_gets16(self.incoming, pos + 24))
                self.pending_mouse_y = Float32(x11_gets16(self.incoming, pos + 26))
                self.has_pending_mouse = True
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
            elif code == 33:  # ClientMessage
                if detail == 32 and self.wm_delete_window != 0:
                    if x11_get32(self.incoming, pos + 12) == self.wm_delete_window:
                        self.close_requested = True
            elif code == 17:  # DestroyNotify
                self.close_requested = True

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

    def poll(mut self, mut input: InputState):
        input.poll_gamepad()
        if self.fd < 0:
            return
        self.receive()
        if self.has_pending_mouse:
            input.move_mouse(self.map_mouse_x(self.pending_mouse_x), self.map_mouse_y(self.pending_mouse_y))
            self.has_pending_mouse = False
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

    def present(mut self, mut pixels: List[UInt32], source_width: Int, source_height: Int):
        if self.fd < 0 or self.window_id == 0 or source_width <= 0 or source_height <= 0:
            return
        self.receive()
        if self.close_requested:
            return

        var dst_w = max(1, self.width)
        var dst_h = max(1, self.height)
        var use_scaled = source_width != dst_w or source_height != dst_h
        if use_scaled:
            self.scale_pixels_into_buffer(pixels, source_width, source_height, dst_w, dst_h)

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

            var band_ptr = Int(self.band_buf.unsafe_ptr())
            var dst_ptr = band_ptr + 24
            var src_ptr = (
                Int(self.scaled_buf.unsafe_ptr().unsafe_offset(y * dst_w))
                if use_scaled
                else Int(pixels.unsafe_ptr().unsafe_offset(y * dst_w))
            )
            _ = external_call["memcpy", Int](dst_ptr, src_ptr, byte_pixels)

            if not self.write_ptr(band_ptr, pkt_len):
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
        _ = self.write_all(ev, len(ev))

    def close(mut self):
        if self.fd >= 0 and self.window_id != 0:
            var req = List[UInt8](length=8, fill=0)
            req[0] = 4  # DestroyWindow
            x11_put16(req, 2, 2)
            x11_put32(req, 4, self.window_id)
            _ = self.write_all(req, len(req))
        close_fd(self.fd)
        self.fd = -1
        self.window_id = 0
        self.gc_id = 0
