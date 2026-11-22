"""Input devices from systemd-logind, over a pure-Mojo D-Bus client.

`/dev/input/event*` is normally readable only by root and the `input` group, and
logind grants no ACL on it, because any reader is a keylogger. Display servers
get the devices the way this module does: they become the session's controller
(`Session.TakeControl`) and ask for each device (`Session.TakeDevice`). logind
opens it and passes the descriptor back over the bus socket (SCM_RIGHTS). This
only works for the session in the foreground, which is the one a game on a text
console runs in.

While a process controls the session, logind keeps the VT in KD_GRAPHICS with its
keyboard off, so keys come from evdev. If the process dies, logind sees the bus
connection close and puts the VT back as it was. That is a second safety net
under kms.mojo's own.

Only what those calls need is here: the EXTERNAL auth handshake with fd passing,
method calls with `b` and `uu` arguments, and replies carrying `h` (a descriptor
index). No libdbus or libsystemd.
"""
from std.ffi import external_call
from flint.core.anchor import sys, IS_X86, SYS_READ, SYS_WRITE
from flint.platform.os import open_path_fd, close_fd, get_time

comptime SYS_SOCKET: Int = 41 if IS_X86 else 198
comptime SYS_CONNECT: Int = 42 if IS_X86 else 203
comptime SYS_RECVMSG: Int = 47 if IS_X86 else 212
comptime SYS_GETUID: Int = 102 if IS_X86 else 174
comptime DBUS_METHOD_CALL: Int = 1
comptime DBUS_METHOD_RETURN: Int = 2
comptime DBUS_ERROR: Int = 3
comptime DBUS_SIGNAL: Int = 4
comptime CALL_TIMEOUT_S: Float64 = 3.0


def dbus_align(mut data: List[UInt8], boundary: Int):
    while len(data) % boundary != 0:
        data.append(0)


def dbus_u32(mut data: List[UInt8], value: Int):
    dbus_align(data, 4)
    for index in range(4):
        data.append(UInt8((value >> (index * 8)) & 0xFF))


def dbus_string(mut data: List[UInt8], text: String):
    """An `s` or `o`: u32 length, the bytes, a nul."""
    dbus_u32(data, text.byte_length())
    for byte in text.as_bytes():
        data.append(byte)
    data.append(0)


def dbus_signature(mut data: List[UInt8], text: String):
    """A `g`: one length byte, the bytes, a nul."""
    data.append(UInt8(text.byte_length()))
    for byte in text.as_bytes():
        data.append(byte)
    data.append(0)


def dbus_put64(mut data: List[UInt8], offset: Int, value: Int):
    for index in range(8):
        data[offset + index] = UInt8((value >> (index * 8)) & 0xFF)


def dbus_get32(data: List[UInt8], offset: Int) -> Int:
    var value = 0
    for index in range(4):
        value |= Int(data[offset + index]) << (index * 8)
    return value


def dbus_header_field(mut data: List[UInt8], code: Int, type_code: String, value: String):
    dbus_align(data, 8)
    data.append(UInt8(code))
    dbus_signature(data, type_code)
    if type_code == "g":
        dbus_signature(data, value)
    else:
        dbus_string(data, value)


def dbus_method_call(
    serial: Int, destination: String, path: String, interface: String, member: String,
    signature: String, body: List[UInt8],
) -> List[UInt8]:
    """A whole little-endian METHOD_CALL message."""
    var data = List[UInt8]()
    data.append(108)  # 'l'
    data.append(UInt8(DBUS_METHOD_CALL))
    data.append(0)  # flags
    data.append(1)  # protocol version
    dbus_u32(data, len(body))
    dbus_u32(data, serial)
    dbus_u32(data, 0)  # header fields length, patched below
    dbus_header_field(data, 1, "o", path)
    if interface.byte_length() > 0:
        dbus_header_field(data, 2, "s", interface)
    dbus_header_field(data, 3, "s", member)
    dbus_header_field(data, 6, "s", destination)
    if signature.byte_length() > 0:
        dbus_header_field(data, 8, "g", signature)
    var fields = len(data) - 16
    for index in range(4):
        data[12 + index] = UInt8((fields >> (index * 8)) & 0xFF)
    dbus_align(data, 8)
    for byte in body:
        data.append(byte)
    return data^


struct DbusMessage(Copyable, Movable):
    var kind: Int
    var serial: Int
    var reply_serial: Int
    var unix_fds: Int
    var error_name: String
    var body: List[UInt8]
    var fds: List[Int]

    def __init__(out self):
        self.kind = 0
        self.serial = 0
        self.reply_serial = 0
        self.unix_fds = 0
        self.error_name = ""
        self.body = List[UInt8]()
        self.fds = List[Int]()


def dbus_message_length(data: List[UInt8]) -> Int:
    """Bytes in the first whole message of `data`, or 0 while it is incomplete."""
    if len(data) < 16:
        return 0
    var header_end = 16 + dbus_get32(data, 12)
    while header_end % 8 != 0:
        header_end += 1
    var total = header_end + dbus_get32(data, 4)
    return total if len(data) >= total else 0


def dbus_parse(data: List[UInt8], length: Int) -> DbusMessage:
    """Decode one little-endian message's header fields and body."""
    var message = DbusMessage()
    var limit = min(length, len(data))
    if limit < 16:
        return message^
    message.kind = Int(data[1])
    message.serial = dbus_get32(data, 8)
    var header_len = dbus_get32(data, 12)
    if header_len < 0 or header_len > limit - 16:
        return message^
    var fields_end = 16 + header_len
    var pos = 16
    while pos < fields_end:
        while pos % 8 != 0:
            pos += 1
        if pos + 2 > fields_end:
            break
        var code = Int(data[pos])
        var sig_length = Int(data[pos + 1])
        if pos + 2 + sig_length + 1 > fields_end:
            break
        var type_code = Int(data[pos + 2]) if sig_length > 0 else 0
        pos += 2 + sig_length + 1
        if type_code == 117:  # 'u'
            while pos % 4 != 0:
                pos += 1
            if pos + 4 > fields_end:
                break
            var value = dbus_get32(data, pos)
            pos += 4
            if code == 5:
                message.reply_serial = value
            elif code == 9:
                message.unix_fds = value
        elif type_code == 115 or type_code == 111:  # 's', 'o'
            while pos % 4 != 0:
                pos += 1
            if pos + 4 > fields_end:
                break
            var text_length = dbus_get32(data, pos)
            if text_length < 0 or text_length >= fields_end - (pos + 4):
                break
            if code == 4:
                var name = List[UInt8](capacity=text_length)
                for index in range(text_length):
                    name.append(data[pos + 4 + index])
                message.error_name = String(unsafe_from_utf8=name)
            pos += 4 + text_length + 1
        elif type_code == 103:  # 'g'
            if pos + 1 > fields_end:
                break
            var g_len = Int(data[pos])
            if pos + 1 + g_len + 1 > fields_end:
                break
            pos += 1 + g_len + 1
        else:
            break  # a type no bus daemon puts in a header; stop rather than misread
    var body_start = fields_end
    while body_start % 8 != 0:
        body_start += 1
    if body_start < limit:
        for index in range(body_start, limit):
            message.body.append(data[index])
    return message^


def hex_ascii(text: String) -> String:
    var digits = "0123456789abcdef"
    var out = String("")
    for byte in text.as_bytes():
        out += String(digits[byte=Int(byte) >> 4])
        out += String(digits[byte=Int(byte) & 15])
    return out


struct LogindSession(Movable):
    """A system-bus connection that controls this process's logind session."""
    var fd: Int
    var serial: Int
    var controlled: Bool
    var incoming: List[UInt8]
    var pending_fds: List[Int]
    var last_error: String

    def __init__(out self):
        self.fd = -1
        self.serial = 0
        self.controlled = False
        self.incoming = List[UInt8]()
        self.pending_fds = List[Int]()
        self.last_error = ""

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.serial = existing.serial
        self.controlled = existing.controlled
        self.incoming = existing.incoming^
        self.pending_fds = existing.pending_fds^
        self.last_error = existing.last_error^
        existing.fd = -1
        existing.controlled = False

    def __deinit__(deinit self):
        self.close()

    def write_all(mut self, data: List[UInt8]) -> Bool:
        var done = 0
        while done < len(data):
            var count = sys(SYS_WRITE, self.fd, Int(data.unsafe_ptr()) + done, len(data) - done)
            if count == -4 or count == -11:
                _ = self.wait_readable(50)
                continue
            if count <= 0:
                return False
            done += count
        return True

    def write_text(mut self, text: String) -> Bool:
        var data = List[UInt8]()
        for byte in text.as_bytes():
            data.append(byte)
        return self.write_all(data)

    def wait_readable(self, timeout_ms: Int) -> Bool:
        var descriptor = List[UInt8](length=8, fill=0)
        for index in range(4):
            descriptor[index] = UInt8((self.fd >> (index * 8)) & 0xFF)
        descriptor[4] = 1  # POLLIN
        return external_call["poll", Int32](descriptor.unsafe_ptr(), UInt64(1), Int32(timeout_ms)) > 0

    def read_line(mut self) -> String:
        """One auth-protocol line, without its CRLF. Empty on timeout or hangup."""
        var line = List[UInt8]()
        var byte = List[UInt8](length=1, fill=0)
        var deadline = get_time() + CALL_TIMEOUT_S
        while get_time() < deadline:
            var count = sys(SYS_READ, self.fd, Int(byte.unsafe_ptr()), 1)
            if count == 1:
                if byte[0] == 10:
                    if len(line) > 0 and line[len(line) - 1] == 13:
                        _ = line.pop()
                    return String(unsafe_from_utf8=line)
                line.append(byte[0])
            elif count == -11 or count == -4:
                _ = self.wait_readable(100)
            else:
                break
        return String(unsafe_from_utf8=line)

    def receive(mut self) -> Bool:
        """recvmsg once: bytes join `incoming`, passed descriptors join `pending_fds`."""
        var buffer = List[UInt8](length=4096, fill=0)
        var iov = List[UInt8](length=16, fill=0)
        var control = List[UInt8](length=256, fill=0)
        var header = List[UInt8](length=56, fill=0)
        dbus_put64(iov, 0, Int(buffer.unsafe_ptr()))
        dbus_put64(iov, 8, len(buffer))
        dbus_put64(header, 16, Int(iov.unsafe_ptr()))
        dbus_put64(header, 24, 1)
        dbus_put64(header, 32, Int(control.unsafe_ptr()))
        dbus_put64(header, 40, len(control))
        var count = sys(SYS_RECVMSG, self.fd, Int(header.unsafe_ptr()), 0x40 | 0x40000000)  # DONTWAIT | CMSG_CLOEXEC
        if count <= 0:
            _ = buffer^
            _ = iov^
            _ = control^
            return False
        for index in range(count):
            self.incoming.append(buffer[index])
        var control_length = 0
        for index in range(8):
            control_length |= Int(header[40 + index]) << (index * 8)
        var offset = 0
        while offset + 16 <= control_length:
            var cmsg_length = 0
            for index in range(8):
                cmsg_length |= Int(control[offset + index]) << (index * 8)
            if cmsg_length < 16:
                break
            if dbus_get32(control, offset + 8) == 1 and dbus_get32(control, offset + 12) == 1:  # SOL_SOCKET, SCM_RIGHTS
                for index in range((cmsg_length - 16) // 4):
                    self.pending_fds.append(dbus_get32(control, offset + 16 + index * 4))
            offset += (cmsg_length + 7) // 8 * 8
        _ = buffer^
        _ = iov^
        _ = control^
        return True

    def next_message(mut self) -> DbusMessage:
        """The next whole message, or one with kind 0 when none has arrived."""
        var length = dbus_message_length(self.incoming)
        if length == 0:
            return DbusMessage()
        var message = dbus_parse(self.incoming, length)
        for _ in range(min(message.unix_fds, len(self.pending_fds))):
            message.fds.append(self.pending_fds.pop(0))
        var rest = List[UInt8]()
        for index in range(length, len(self.incoming)):
            rest.append(self.incoming[index])
        self.incoming = rest^
        return message^

    def call(mut self, destination: String, path: String, interface: String, member: String, signature: String, body: List[UInt8]) -> DbusMessage:
        """A method call and its reply (kind 2, or 3 for an error; 0 on timeout)."""
        self.serial += 1
        var serial = self.serial
        if not self.write_all(dbus_method_call(serial, destination, path, interface, member, signature, body)):
            self.last_error = member + ": write to the system bus failed"
            return DbusMessage()
        var deadline = get_time() + CALL_TIMEOUT_S
        while get_time() < deadline:
            var message = self.next_message()
            if message.kind == 0:
                if not self.receive():
                    _ = self.wait_readable(100)
                continue
            if (message.kind == DBUS_METHOD_RETURN or message.kind == DBUS_ERROR) and message.reply_serial == serial:
                if message.kind == DBUS_ERROR:
                    self.last_error = member + ": " + message.error_name
                return message^
            for fd in message.fds:  # signals and stray replies carry none we want
                close_fd(fd)
        self.last_error = member + ": no reply from the system bus"
        return DbusMessage()

    def connect(mut self) -> Bool:
        var path = String("/run/dbus/system_bus_socket")
        self.fd = sys(SYS_SOCKET, 1, 1 | 0x80000 | 0x800, 0)  # AF_UNIX, SOCK_STREAM | CLOEXEC | NONBLOCK
        if self.fd < 0:
            self.last_error = "socket failed"
            return False
        var address = List[UInt8](length=110, fill=0)
        address[0] = 1
        for index in range(path.byte_length()):
            address[2 + index] = path.as_bytes()[index]
        var result = sys(SYS_CONNECT, self.fd, Int(address.unsafe_ptr()), 110)
        if result == -115:  # EINPROGRESS on a non-blocking socket
            _ = self.wait_readable(500)
            result = 0
        _ = address^
        if result != 0:
            self.last_error = "cannot reach the system bus at " + path
            self.close()
            return False
        var uid = String(sys(SYS_GETUID))
        var nul = List[UInt8](length=1, fill=0)
        if not self.write_all(nul) or not self.write_text("AUTH EXTERNAL " + hex_ascii(uid) + "\r\n"):
            self.close()
            return False
        if not self.read_line().startswith("OK "):
            self.last_error = "system bus refused EXTERNAL auth"
            self.close()
            return False
        _ = self.write_text("NEGOTIATE_UNIX_FD\r\n")
        if not self.read_line().startswith("AGREE_UNIX_FD"):
            self.last_error = "system bus will not pass file descriptors"
            self.close()
            return False
        _ = self.write_text("BEGIN\r\n")
        var hello = self.call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "Hello", "", List[UInt8]())
        if hello.kind != DBUS_METHOD_RETURN:
            self.close()
            return False
        return True

    def take_control(mut self) -> Bool:
        """Become the controller of the session this process runs in (no force)."""
        if self.fd < 0 and not self.connect():
            return False
        var body = List[UInt8]()
        dbus_u32(body, 0)
        var reply = self.call("org.freedesktop.login1", "/org/freedesktop/login1/session/auto", "org.freedesktop.login1.Session", "TakeControl", "b", body)
        self.controlled = reply.kind == DBUS_METHOD_RETURN
        return self.controlled

    def take_device(mut self, major: Int, minor: Int) -> Int:
        """An open descriptor for a device node, from logind. -1 on refusal."""
        if not self.controlled:
            return -1
        var body = List[UInt8]()
        dbus_u32(body, major)
        dbus_u32(body, minor)
        var reply = self.call("org.freedesktop.login1", "/org/freedesktop/login1/session/auto", "org.freedesktop.login1.Session", "TakeDevice", "uu", body)
        if reply.kind != DBUS_METHOD_RETURN or len(reply.body) < 4:
            for fd in reply.fds:
                close_fd(fd)
            return -1
        var index = dbus_get32(reply.body, 0)
        var result = -1
        for slot in range(len(reply.fds)):
            if slot == index:
                result = reply.fds[slot]
            else:
                close_fd(reply.fds[slot])
        return result

    def drain(mut self):
        """Read and drop queued signals (PauseDevice, PropertiesChanged) so the socket never fills."""
        if self.fd < 0:
            return
        while self.receive():
            pass
        while True:
            var message = self.next_message()
            if message.kind == 0:
                break
            for fd in message.fds:
                close_fd(fd)

    def close(mut self):
        if self.fd >= 0 and self.controlled:
            _ = self.call("org.freedesktop.login1", "/org/freedesktop/login1/session/auto", "org.freedesktop.login1.Session", "ReleaseControl", "", List[UInt8]())
        self.controlled = False
        for fd in self.pending_fds:
            close_fd(fd)
        self.pending_fds.clear()
        close_fd(self.fd)
        self.fd = -1


def device_numbers(event_name: String) -> Tuple[Int, Int]:
    """Major and minor of /dev/input/<name>, read from /sys/class/input/<name>/dev ("13:67")."""
    var path = "/sys/class/input/" + event_name + "/dev"
    var fd = open_path_fd(path, Int32(0x80000))
    if fd < 0:
        return (-1, -1)
    var buf = List[UInt8](length=32, fill=0)
    var count = sys(SYS_READ, fd, Int(buf.unsafe_ptr()), 32)
    close_fd(fd)
    var major = 0
    var minor = 0
    var seen_colon = False
    for index in range(max(0, count)):
        var byte = Int(buf[index])
        if byte == 58:
            seen_colon = True
        elif byte >= 48 and byte <= 57:
            if seen_colon:
                minor = minor * 10 + byte - 48
            else:
                major = major * 10 + byte - 48
        else:
            break
    if not seen_colon:
        return (-1, -1)
    return (major, minor)
