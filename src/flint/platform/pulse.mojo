"""Direct client for the PulseAudio native protocol, served by PipeWire and PulseAudio.

Mojo speaks the Unix socket itself, as it does for Wayland, so sound follows the
desktop's chosen output and shares it with other programs. libpulse is not used.
All protocol integers are big-endian. Version 13 fixes the stream request layout.
"""
from std.ffi import external_call
from flint.platform.os import close_fd, env_value, read_bytes

comptime PROTOCOL_VERSION: UInt32 = 13
comptime COMMAND_CHANNEL: UInt32 = 0xFFFFFFFF
comptime DESCRIPTOR_BYTES: Int = 20
comptime COOKIE_BYTES: Int = 256

comptime COMMAND_ERROR: UInt32 = 0
comptime COMMAND_REPLY: UInt32 = 2
comptime COMMAND_CREATE_PLAYBACK_STREAM: UInt32 = 3
comptime COMMAND_AUTH: UInt32 = 8
comptime COMMAND_SET_CLIENT_NAME: UInt32 = 9
comptime COMMAND_GET_SINK_INFO_LIST: UInt32 = 22
comptime COMMAND_REQUEST: UInt32 = 61
comptime COMMAND_UNDERFLOW: UInt32 = 63
comptime COMMAND_STREAM_KILLED: UInt32 = 64

comptime SAMPLE_S16LE: UInt8 = 3
comptime CHANNEL_LEFT: UInt8 = 1
comptime CHANNEL_RIGHT: UInt8 = 2
comptime VOLUME_NORMAL: UInt32 = 0x10000
comptime NO_INDEX: UInt32 = 0xFFFFFFFF

comptime MSG_DONTWAIT: Int32 = 0x40
comptime MSG_NOSIGNAL: Int32 = 0x4000
comptime EAGAIN: Int32 = 11


def put_be32(mut data: List[UInt8], value: UInt32):
    data.append(UInt8((value >> 24) & 255))
    data.append(UInt8((value >> 16) & 255))
    data.append(UInt8((value >> 8) & 255))
    data.append(UInt8(value & 255))


def get_be32(data: List[UInt8], offset: Int) -> UInt32:
    return (UInt32(data[offset]) << 24) | (UInt32(data[offset + 1]) << 16) | (UInt32(data[offset + 2]) << 8) | UInt32(data[offset + 3])


def tag_u32(mut data: List[UInt8], value: UInt32):
    data.append(UInt8(ord("L")))
    put_be32(data, value)


def tag_bool(mut data: List[UInt8], value: Bool):
    data.append(UInt8(ord("1") if value else ord("0")))


def tag_string(mut data: List[UInt8], value: String):
    data.append(UInt8(ord("t")))
    for index in range(value.byte_length()):
        data.append(value.as_bytes()[index])
    data.append(0)


def tag_null_string(mut data: List[UInt8]):
    data.append(UInt8(ord("N")))


def tag_arbitrary(mut data: List[UInt8], bytes: List[UInt8]):
    data.append(UInt8(ord("x")))
    put_be32(data, UInt32(len(bytes)))
    for index in range(len(bytes)):
        data.append(bytes[index])


def tag_proplist(mut data: List[UInt8], key: String, value: String):
    """A single-entry property list. Values keep their terminating zero."""
    data.append(UInt8(ord("P")))
    tag_string(data, key)
    var bytes = List[UInt8]()
    for index in range(value.byte_length()):
        bytes.append(value.as_bytes()[index])
    bytes.append(0)
    tag_u32(data, UInt32(len(bytes)))
    tag_arbitrary(data, bytes)
    tag_null_string(data)


def tag_sample_spec(mut data: List[UInt8], rate: UInt32):
    data.append(UInt8(ord("a")))
    data.append(SAMPLE_S16LE)
    data.append(2)
    put_be32(data, rate)


def tag_stereo_map(mut data: List[UInt8]):
    data.append(UInt8(ord("m")))
    data.append(2)
    data.append(CHANNEL_LEFT)
    data.append(CHANNEL_RIGHT)


def tag_stereo_volume(mut data: List[UInt8]):
    data.append(UInt8(ord("v")))
    data.append(2)
    put_be32(data, VOLUME_NORMAL)
    put_be32(data, VOLUME_NORMAL)


def frame_descriptor(length: Int, channel: UInt32) -> List[UInt8]:
    """Length, channel, offset high, offset low, and flags."""
    var data = List[UInt8]()
    put_be32(data, UInt32(length))
    put_be32(data, channel)
    put_be32(data, 0)
    put_be32(data, 0)
    put_be32(data, 0)
    return data^


def create_stream_body(tag: UInt32, rate: UInt32, target_bytes: UInt32, prebuffer_bytes: UInt32, minimum_request: UInt32, sink: String = "") -> List[UInt8]:
    """The playback stream request. An empty `sink` asks for the default
    sink (the bytes are the same as before package 3.15); otherwise the
    stream goes to the sink of that name (package 3.15: output choice)."""
    var body = List[UInt8]()
    tag_u32(body, COMMAND_CREATE_PLAYBACK_STREAM)
    tag_u32(body, tag)
    tag_sample_spec(body, rate)
    tag_stereo_map(body)
    tag_u32(body, NO_INDEX)  # any sink index: chosen by name, or the default
    if sink.byte_length() > 0:
        tag_string(body, sink)
    else:
        tag_null_string(body)    # the default sink
    tag_u32(body, 0xFFFFFFFF)  # maximum length: server default
    tag_bool(body, False)    # not corked
    tag_u32(body, target_bytes)
    tag_u32(body, prebuffer_bytes)
    tag_u32(body, minimum_request)
    tag_u32(body, 0)         # sync group
    tag_stereo_volume(body)
    for _ in range(7):       # no_remap through variable_rate
        tag_bool(body, False)
    tag_bool(body, False)    # not muted
    tag_bool(body, True)     # latency follows target_bytes
    tag_proplist(body, "media.name", "SHARDDEPTH")
    return body^


struct SinkInfo(Copyable, Movable):
    """One output the sound server offers (package 3.15)."""
    var index: UInt32
    var name: String
    var description: String

    def __init__(out self, index: UInt32, name: String, description: String):
        self.index = index
        self.name = name
        self.description = description


struct TagReader:
    """Reads tagged values of a protocol message. Any malformed or unknown
    tag sets `failed`; nothing reads past the end."""
    var data: List[UInt8]
    var at: Int
    var failed: Bool

    def __init__(out self, var data: List[UInt8], at: Int):
        self.data = data^
        self.at = at
        self.failed = False

    def done(self) -> Bool:
        return self.failed or self.at >= len(self.data)

    def need(mut self, count: Int) -> Bool:
        if self.failed or self.at + count > len(self.data):
            self.failed = True
            return False
        return True

    def u32(mut self) -> UInt32:
        if not self.need(5) or self.data[self.at] != UInt8(ord("L")):
            self.failed = True
            return 0
        var value = get_be32(self.data, self.at + 1)
        self.at += 5
        return value

    def string(mut self) -> String:
        """A string, or empty for a null string."""
        if not self.need(1):
            return String()
        var kind = self.data[self.at]
        if kind == UInt8(ord("N")):
            self.at += 1
            return String()
        if kind != UInt8(ord("t")):
            self.failed = True
            return String()
        var end = self.at + 1
        while end < len(self.data) and self.data[end] != 0:
            end += 1
        if end >= len(self.data):
            self.failed = True
            return String()
        var bytes = List[UInt8]()
        for index in range(self.at + 1, end):
            bytes.append(self.data[index])
        self.at = end + 1
        return String(unsafe_from_utf8=bytes)

    def skip(mut self):
        """Skips one tagged value of any kind this client knows."""
        if not self.need(1):
            return
        var kind = Int(self.data[self.at])
        if kind == ord("L") or kind == ord("V"):
            _ = self.need(5)
            self.at += 5
        elif kind == ord("B"):
            _ = self.need(2)
            self.at += 2
        elif kind == ord("R") or kind == ord("r") or kind == ord("U") or kind == ord("T"):
            _ = self.need(9)
            self.at += 9
        elif kind == ord("1") or kind == ord("0") or kind == ord("N"):
            self.at += 1
        elif kind == ord("t"):
            _ = self.string()
        elif kind == ord("a"):
            _ = self.need(7)
            self.at += 7
        elif kind == ord("m"):
            if self.need(2):
                var channels = Int(self.data[self.at + 1])
                _ = self.need(2 + channels)
                self.at += 2 + channels
        elif kind == ord("v"):
            if self.need(2):
                var channels = Int(self.data[self.at + 1])
                _ = self.need(2 + channels * 4)
                self.at += 2 + channels * 4
        elif kind == ord("x"):
            if self.need(5):
                var length = Int(get_be32(self.data, self.at + 1))
                _ = self.need(5 + length)
                self.at += 5 + length
        elif kind == ord("P"):
            # key string, u32 length, arbitrary value ... until a null string
            self.at += 1
            for _ in range(256):
                if not self.need(1):
                    return
                if self.data[self.at] == UInt8(ord("N")):
                    self.at += 1
                    return
                _ = self.string()
                _ = self.u32()
                if self.failed or not self.need(1) or self.data[self.at] != UInt8(ord("x")):
                    self.failed = True
                    return
                self.skip()
            self.failed = True
        else:
            self.failed = True


# What follows a sink's description in a protocol-13 sink info record:
# sample spec, channel map, owner module, volume, mute, monitor source,
# its name, latency, driver, flags, property list, requested latency.
comptime SINK_FIELDS_AFTER_DESCRIPTION: Int = 12


def parse_sink_list(reply: List[UInt8]) -> List[SinkInfo]:
    """The sinks of a GET_SINK_INFO_LIST reply (the whole message: command
    and tag first). Parsing stops at the first record it cannot read and
    keeps the ones before it."""
    var sinks = List[SinkInfo]()
    var reader = TagReader(reply.copy(), 10)
    for _ in range(64):
        if reader.done():
            break
        var index = reader.u32()
        var name = reader.string()
        var description = reader.string()
        for _ in range(SINK_FIELDS_AFTER_DESCRIPTION):
            reader.skip()
        if reader.failed or name.byte_length() == 0:
            break
        var shown = description if description.byte_length() > 0 else name.copy()
        sinks.append(SinkInfo(index, name, shown))
    return sinks^


struct PulseClient(Movable):
    var fd: Int
    var channel: UInt32
    var requested_bytes: Int
    var next_tag: UInt32
    var incoming: List[UInt8]
    var underflows: Int
    var error: String

    def __init__(out self):
        self.fd = -1
        self.channel = 0
        self.requested_bytes = 0
        self.next_tag = 0
        self.incoming = List[UInt8]()
        self.underflows = 0
        self.error = ""

    def fail(mut self, message: String) -> Bool:
        self.error = message
        self.close()
        return False

    def close(mut self):
        close_fd(self.fd)
        self.fd = -1

    def connected(self) -> Bool:
        return self.fd >= 0

    def open_socket(mut self) -> Bool:
        var runtime = env_value("XDG_RUNTIME_DIR")
        if runtime.byte_length() == 0:
            self.error = "XDG_RUNTIME_DIR is not set"
            return False
        var path = runtime + "/pulse/native"
        if path.byte_length() > 107:
            self.error = "sound server path is too long"
            return False
        self.fd = Int(external_call["socket", Int32](Int32(1), Int32(1 | 0x80000), Int32(0)))
        if self.fd < 0:
            self.error = "socket failed"
            return False
        var address = List[UInt8](length=110, fill=0)
        address[0] = 1  # AF_UNIX
        for index in range(path.byte_length()):
            address[2 + index] = path.as_bytes()[index]
        if external_call["connect", Int32](Int32(self.fd), address.unsafe_ptr(), Int32(110)) != 0:
            return self.fail("no sound server at " + path)
        return True

    def send_all(mut self, data: List[UInt8], first_flags: Int32) -> Int:
        """Returns bytes sent, 0 if the socket is full, or -1 on failure."""
        var sent = 0
        var flags = first_flags
        while sent < len(data):
            var count = Int(external_call["send", Int64](Int32(self.fd), data.unsafe_ptr().unsafe_offset(sent), Int64(len(data) - sent), flags))
            if count <= 0:
                var error = external_call["__errno_location", Pointer[Int32, origin=MutAnyOrigin]]()[unsafe_offset=0]
                if error == EAGAIN and sent == 0:
                    return 0
                if error != EAGAIN:
                    return -1
            else:
                sent += count
            # Finish a started frame, or the stream would lose its framing.
            flags = MSG_NOSIGNAL
        return sent

    def send_command(mut self, body: List[UInt8]) -> Bool:
        var frame = frame_descriptor(len(body), COMMAND_CHANNEL)
        frame.extend(body.copy())
        return self.send_all(frame, MSG_NOSIGNAL) == len(frame)

    def receive(mut self):
        var buffer = List[UInt8](length=4096, fill=0)
        while self.fd >= 0:
            var count = Int(external_call["recv", Int64](Int32(self.fd), buffer.unsafe_ptr(), Int64(len(buffer)), MSG_DONTWAIT))
            if count == 0:
                _ = self.fail("sound server disconnected")
                return
            if count < 0:
                return
            for index in range(count):
                self.incoming.append(buffer[index])

    def next_frame(mut self, mut channel: UInt32, mut payload: List[UInt8]) -> Bool:
        """Remove one complete frame from the received bytes."""
        if len(self.incoming) < DESCRIPTOR_BYTES:
            return False
        var length = Int(get_be32(self.incoming, 0))
        if len(self.incoming) < DESCRIPTOR_BYTES + length:
            return False
        channel = get_be32(self.incoming, 4)
        payload = List[UInt8](length=length, fill=0)
        for index in range(length):
            payload[index] = self.incoming[DESCRIPTOR_BYTES + index]
        var remaining = List[UInt8]()
        for index in range(DESCRIPTOR_BYTES + length, len(self.incoming)):
            remaining.append(self.incoming[index])
        self.incoming = remaining^
        return True

    def handle_event(mut self, payload: List[UInt8]):
        if len(payload) < 10:
            return
        var command = get_be32(payload, 1)
        if command == COMMAND_REQUEST and len(payload) >= 20 and get_be32(payload, 11) == self.channel:
            self.requested_bytes += Int(get_be32(payload, 16))
        elif command == COMMAND_UNDERFLOW:
            self.underflows += 1
        elif command == COMMAND_STREAM_KILLED:
            _ = self.fail("sound server closed the stream")

    def wait_reply(mut self, tag: UInt32, mut reply: List[UInt8]) -> Bool:
        """Wait up to about two seconds for the reply to one command."""
        var descriptor = List[UInt8](length=8, fill=0)
        for _ in range(200):
            descriptor[0] = UInt8(self.fd & 255)
            descriptor[1] = UInt8((self.fd >> 8) & 255)
            descriptor[4] = 1  # POLLIN
            _ = external_call["poll", Int32](descriptor.unsafe_ptr(), UInt64(1), Int32(10))
            self.receive()
            if self.fd < 0:
                return False
            var channel = UInt32(0)
            var payload = List[UInt8]()
            while self.next_frame(channel, payload):
                if channel != COMMAND_CHANNEL or len(payload) < 10:
                    continue
                if get_be32(payload, 6) == tag and get_be32(payload, 1) == COMMAND_REPLY:
                    reply = payload^
                    return True
                if get_be32(payload, 6) == tag and get_be32(payload, 1) == COMMAND_ERROR:
                    var code = get_be32(payload, 11) if len(payload) >= 15 else UInt32(0)
                    return self.fail("sound server refused a request, error " + String(code))
                self.handle_event(payload)
        return self.fail("sound server did not reply")

    def request(mut self, var body: List[UInt8], tag: UInt32, mut reply: List[UInt8]) -> Bool:
        if not self.send_command(body):
            return self.fail("sound server write failed")
        return self.wait_reply(tag, reply)

    def new_tag(mut self) -> UInt32:
        self.next_tag += 1
        return self.next_tag

    def connect(mut self, rate: Int, block_bytes: Int, sink: String = "") -> Bool:
        """Opens the stream, on the sink named `sink` or the default one."""
        if not self.handshake():
            return False
        var reply = List[UInt8]()
        var tag = self.new_tag()
        if not self.request(create_stream_body(tag, UInt32(rate), UInt32(block_bytes * 4), UInt32(block_bytes * 2), UInt32(block_bytes), sink), tag, reply):
            return False
        if len(reply) < 25:
            return self.fail("sound server sent a short stream reply")
        self.channel = get_be32(reply, 11)
        self.requested_bytes = Int(get_be32(reply, 21))
        return True

    def list_sinks(mut self) -> List[SinkInfo]:
        """The server's outputs, on this connection (made for the question
        when there is none: then it is closed again). Empty without a server."""
        var opened = False
        if self.fd < 0:
            if not self.handshake():
                return List[SinkInfo]()
            opened = True
        var reply = List[UInt8]()
        var tag = self.new_tag()
        var body = List[UInt8]()
        tag_u32(body, COMMAND_GET_SINK_INFO_LIST)
        tag_u32(body, tag)
        var sinks = List[SinkInfo]()
        if self.request(body^, tag, reply):
            sinks = parse_sink_list(reply)
        if opened:
            self.close()
        return sinks^

    def handshake(mut self) -> Bool:
        """Socket, authentication and the client's name."""
        if not self.open_socket():
            return False
        var cookie = read_bytes(env_value("HOME") + "/.config/pulse/cookie")
        if len(cookie) != COOKIE_BYTES:
            cookie = List[UInt8](length=COOKIE_BYTES, fill=0)
        var reply = List[UInt8]()
        var tag = self.new_tag()
        var body = List[UInt8]()
        tag_u32(body, COMMAND_AUTH)
        tag_u32(body, tag)
        tag_u32(body, PROTOCOL_VERSION)
        tag_arbitrary(body, cookie)
        if not self.request(body^, tag, reply):
            return False

        tag = self.new_tag()
        body = List[UInt8]()
        tag_u32(body, COMMAND_SET_CLIENT_NAME)
        tag_u32(body, tag)
        tag_proplist(body, "application.name", "SHARDDEPTH")
        return self.request(body^, tag, reply)

    def pump(mut self):
        """Collect buffer requests from the sound server."""
        self.receive()
        var channel = UInt32(0)
        var payload = List[UInt8]()
        while self.fd >= 0 and self.next_frame(channel, payload):
            if channel == COMMAND_CHANNEL:
                self.handle_event(payload)

    def write_samples(mut self, samples: List[Int16]) -> Bool:
        """Send one block. False means the socket is full and nothing was sent."""
        var bytes = len(samples) * 2
        var frame = frame_descriptor(bytes, self.channel)
        frame.resize(DESCRIPTOR_BYTES + bytes, 0)
        _ = external_call["memcpy", Int](Int(frame.unsafe_ptr()) + DESCRIPTOR_BYTES, Int(samples.unsafe_ptr()), bytes)
        var sent = self.send_all(frame, MSG_DONTWAIT | MSG_NOSIGNAL)
        if sent < 0:
            _ = self.fail("sound server write failed")
            return False
        if sent == 0:
            return False
        self.requested_bytes -= bytes
        return True
