"""Live link: the running game listens on a local Unix socket for what Blender
is editing (the current pose, the edited clip, the active object's mesh) and
shows it within a frame, with no export.

Presentation only. Nothing received here reaches the simulation, saves or replay
checksums: the messages become render/live_view.mojo state, which only moves
model nodes and swaps meshes at draw time. The link is off unless SD_LIVELINK=1,
and never on in scripted runs, recordings, replays or tests of the game
(livelink_allowed).

Wire format, little endian. Every message is a 12-byte header and a payload:

    u32 magic "SDLL"   u16 version (1)   u16 kind   u32 payload bytes

    str  = u16 byte count, UTF-8 bytes
    TRS  = 10 f32: translation xyz, rotation xyzw, scale xyz, relative to the
           bone's rest (Blender's pose bone matrix_basis), glTF axes

    HELLO  str client
    POSE   str target, u16 bones, bones x (str name, TRS)
    CLIP   str target, str clip, f32 fps, u32 frames, u8 loop, u16 bones,
           bones x str name, frames x bones x TRS
    MESH   str target, str object, u32 vertices, vertices x 3 f32 position,
           vertices x 3 f32 normal, vertices x 4 u8 colour, u32 base colour
           (0xAARRGGBB), u32 indices, indices x u32
    CLEAR  str target ("layer" clears every PLACE)
    ACK    (game to client) u16 kind acknowledged, str what the game did
    PLACE  str what (brush, area or entity), u32 bytes, that many bytes of one
           layer-file record as JSON (package 1.2: a light, fog, camera,
           audio or trigger area, or a brush, previewed where it would be)

A target is the asset name the add-on edits: Maroon (any form) poses the player,
anything else is shown as a preview beside the player.
"""
from std.ffi import external_call
from flint.core.replay import bits_float, float_bits
from flint.platform.os import close_fd, env_value, last_errno

comptime LIVELINK_MAGIC: UInt32 = 0x4C4C4453  # "SDLL"
comptime LIVELINK_VERSION: Int = 1
comptime LINK_HELLO: Int = 0
comptime LINK_POSE: Int = 1
comptime LINK_CLIP: Int = 2
comptime LINK_MESH: Int = 3
comptime LINK_CLEAR: Int = 4
comptime LINK_ACK: Int = 5
comptime LINK_PLACE: Int = 6
comptime MAX_RECORD_BYTES: Int = 1 << 20
comptime HEADER_BYTES: Int = 12
comptime MAX_PAYLOAD: Int = 64 * 1024 * 1024
comptime MAX_CLIENTS: Int = 4
comptime TRS_FLOATS: Int = 10
comptime AF_UNIX: Int32 = 1
comptime SOCK_STREAM_NONBLOCK: Int32 = 1 | 2048 | 524288  # SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC
comptime MSG_DONTWAIT: Int32 = 64
comptime MSG_NOSIGNAL: Int32 = 0x4000
comptime EAGAIN: Int32 = 11


def livelink_allowed(requested: Bool, automated: Bool, recording: Bool, replaying: Bool) -> Bool:
    """Only an interactive session that asked for it. A scripted route, a
    recording or a replay must see exactly what it would without Blender."""
    return requested and not automated and not recording and not replaying


def livelink_socket_path() -> String:
    var path = env_value("SD_LIVELINK_SOCKET")
    if path.byte_length() > 0:
        return path
    var runtime = env_value("XDG_RUNTIME_DIR")
    if runtime.byte_length() > 0:
        return runtime + "/deepsong-livelink.sock"
    return "build/deepsong-livelink.sock"


# ---------------------------------------------------------------- bytes

def put_u16(mut out: List[UInt8], value: Int):
    out.append(UInt8(value & 255))
    out.append(UInt8((value >> 8) & 255))


def put_u32(mut out: List[UInt8], value: UInt32):
    for shift in [0, 8, 16, 24]:
        out.append(UInt8((value >> UInt32(shift)) & 255))


def put_f32(mut out: List[UInt8], value: Float32):
    put_u32(out, float_bits(value))


def put_str(mut out: List[UInt8], text: String):
    var bytes = text.as_bytes()
    put_u16(out, len(bytes))
    for byte in bytes:
        out.append(byte)


def framed(kind: Int, payload: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    put_u32(out, LIVELINK_MAGIC)
    put_u16(out, LIVELINK_VERSION)
    put_u16(out, kind)
    put_u32(out, UInt32(len(payload)))
    out.extend(payload.copy())
    return out^


struct Reader(Movable):
    var data: List[UInt8]
    var offset: Int

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.offset = 0

    def need(self, count: Int) raises:
        if count < 0 or self.offset + count > len(self.data):
            raise Error("message ends early at byte " + String(self.offset))

    def u8(mut self) raises -> Int:
        self.need(1)
        self.offset += 1
        return Int(self.data[self.offset - 1])

    def u16(mut self) raises -> Int:
        self.need(2)
        var value = Int(self.data[self.offset]) | (Int(self.data[self.offset + 1]) << 8)
        self.offset += 2
        return value

    def u32(mut self) raises -> UInt32:
        self.need(4)
        var value = UInt32(self.data[self.offset]) | (UInt32(self.data[self.offset + 1]) << 8) | (UInt32(self.data[self.offset + 2]) << 16) | (UInt32(self.data[self.offset + 3]) << 24)
        self.offset += 4
        return value

    def f32(mut self) raises -> Float32:
        var value = bits_float(self.u32())
        if value != value or abs(value) > 1e7:
            raise Error("a number at byte " + String(self.offset - 4) + " is not finite")
        return value

    def text(mut self) raises -> String:
        var count = self.u16()
        self.need(count)
        var bytes = List[UInt8]()
        for index in range(count):
            bytes.append(self.data[self.offset + index])
        self.offset += count
        return String(unsafe_from_utf8=bytes)

    def done(self) raises:
        if self.offset != len(self.data):
            raise Error(String(len(self.data) - self.offset) + " bytes left over")


struct LiveMessage(Copyable, Movable):
    var kind: Int
    var target: String
    var name: String          # clip name, mesh object name, or client name
    var fps: Float32
    var frames: Int
    var loop: Bool
    var bones: List[String]
    var values: List[Float32]  # POSE: bones x TRS; CLIP: frames x bones x TRS
    var positions: List[Float32]
    var normals: List[Float32]
    var colors: List[UInt8]
    var base_color: UInt32
    var indices: List[Int]
    var client: Int            # which connection it came from
    var record: String         # PLACE: the layer record, as JSON

    def __init__(out self, kind: Int):
        self.kind = kind
        self.target = ""
        self.name = ""
        self.fps = 0
        self.frames = 0
        self.loop = False
        self.bones = List[String]()
        self.values = List[Float32]()
        self.positions = List[Float32]()
        self.normals = List[Float32]()
        self.colors = List[UInt8]()
        self.base_color = 0xFFFFFFFF
        self.indices = List[Int]()
        self.client = -1
        self.record = ""


def decode_message(kind: Int, var payload: List[UInt8]) raises -> LiveMessage:
    var reader = Reader(payload^)
    var message = LiveMessage(kind)
    if kind == LINK_HELLO:
        message.name = reader.text()
    elif kind == LINK_POSE:
        message.target = reader.text()
        var count = reader.u16()
        for _ in range(count):
            message.bones.append(reader.text())
            for _ in range(TRS_FLOATS):
                message.values.append(reader.f32())
    elif kind == LINK_CLIP:
        message.target = reader.text()
        message.name = reader.text()
        message.fps = reader.f32()
        message.frames = Int(reader.u32())
        message.loop = reader.u8() != 0
        var count = reader.u16()
        if message.fps <= 0 or message.frames <= 0 or message.frames > 100000:
            raise Error("clip " + message.name + " has " + String(message.frames) + " frames at " + String(message.fps) + " fps")
        for _ in range(count):
            message.bones.append(reader.text())
        reader.need(message.frames * count * TRS_FLOATS * 4)
        for _ in range(message.frames * count * TRS_FLOATS):
            message.values.append(reader.f32())
    elif kind == LINK_MESH:
        message.target = reader.text()
        message.name = reader.text()
        var vertices = Int(reader.u32())
        reader.need(vertices * 28)
        for _ in range(vertices * 3):
            message.positions.append(reader.f32())
        for _ in range(vertices * 3):
            message.normals.append(reader.f32())
        for _ in range(vertices * 4):
            message.colors.append(UInt8(reader.u8()))
        message.base_color = reader.u32()
        var count = Int(reader.u32())
        if count % 3 != 0:
            raise Error("mesh " + message.name + " has " + String(count) + " indices, not whole triangles")
        reader.need(count * 4)
        for _ in range(count):
            var index = Int(reader.u32())
            if index >= vertices:
                raise Error("mesh " + message.name + " points at vertex " + String(index) + " of " + String(vertices))
            message.indices.append(index)
    elif kind == LINK_CLEAR:
        message.target = reader.text()
    elif kind == LINK_PLACE:
        message.target = reader.text()
        var count = Int(reader.u32())
        if count > MAX_RECORD_BYTES:
            raise Error("a placed record of " + String(count) + " bytes")
        reader.need(count)
        var bytes = List[UInt8]()
        for _ in range(count):
            bytes.append(UInt8(reader.u8()))
        message.record = String(unsafe_from_utf8=bytes)
    else:
        raise Error("unknown message kind " + String(kind))
    reader.done()
    return message^


# ---------------------------------------------------------------- what a client sends

def encode_hello(client: String) -> List[UInt8]:
    var payload = List[UInt8]()
    put_str(payload, client)
    return framed(LINK_HELLO, payload)


def encode_pose(target: String, bones: List[String], values: List[Float32]) -> List[UInt8]:
    var payload = List[UInt8]()
    put_str(payload, target)
    put_u16(payload, len(bones))
    for index in range(len(bones)):
        put_str(payload, bones[index])
        for item in range(TRS_FLOATS):
            put_f32(payload, values[index * TRS_FLOATS + item])
    return framed(LINK_POSE, payload)


def encode_clip(target: String, clip: String, fps: Float32, frames: Int, loop: Bool, bones: List[String], values: List[Float32]) -> List[UInt8]:
    var payload = List[UInt8]()
    put_str(payload, target)
    put_str(payload, clip)
    put_f32(payload, fps)
    put_u32(payload, UInt32(frames))
    payload.append(UInt8(1) if loop else UInt8(0))
    put_u16(payload, len(bones))
    for bone in bones:
        put_str(payload, bone)
    for value in values:
        put_f32(payload, value)
    return framed(LINK_CLIP, payload)


def encode_mesh(target: String, object: String, positions: List[Float32], normals: List[Float32], colors: List[UInt8], base_color: UInt32, indices: List[Int]) -> List[UInt8]:
    var payload = List[UInt8]()
    put_str(payload, target)
    put_str(payload, object)
    put_u32(payload, UInt32(len(positions) // 3))
    for value in positions:
        put_f32(payload, value)
    for value in normals:
        put_f32(payload, value)
    for value in colors:
        payload.append(value)
    put_u32(payload, base_color)
    put_u32(payload, UInt32(len(indices)))
    for index in indices:
        put_u32(payload, UInt32(index))
    return framed(LINK_MESH, payload)


def encode_clear(target: String) -> List[UInt8]:
    var payload = List[UInt8]()
    put_str(payload, target)
    return framed(LINK_CLEAR, payload)


def encode_place(what: String, record: String) -> List[UInt8]:
    var payload = List[UInt8]()
    put_str(payload, what)
    var bytes = record.as_bytes()
    put_u32(payload, UInt32(len(bytes)))
    for byte in bytes:
        payload.append(byte)
    return framed(LINK_PLACE, payload)


def encode_ack(kind: Int, text: String) -> List[UInt8]:
    var payload = List[UInt8]()
    put_u16(payload, kind)
    put_str(payload, text)
    return framed(LINK_ACK, payload)


# ---------------------------------------------------------------- sockets

def unix_address(path: String) -> List[UInt8]:
    var address = List[UInt8](length=110, fill=0)
    address[0] = UInt8(AF_UNIX)
    var bytes = path.as_bytes()
    for index in range(min(len(bytes), 107)):
        address[2 + index] = bytes[index]
    return address^


def send_all(fd: Int, data: List[UInt8]) -> Bool:
    var offset = 0
    var attempts = 0
    while offset < len(data) and attempts < 100000:
        var count = Int(external_call["send", Int64](Int32(fd), data.unsafe_ptr().unsafe_offset(offset), Int64(len(data) - offset), MSG_NOSIGNAL))
        if count > 0:
            offset += count
            continue
        if count < 0 and last_errno() == EAGAIN:
            attempts += 1
            _ = external_call["usleep", Int32](UInt32(100))
            continue
        return False
    return offset == len(data)


def receive_available(fd: Int, mut into: List[UInt8]) -> Bool:
    """Appends whatever has arrived; False once the other side has hung up."""
    var buffer = List[UInt8](length=262144, fill=0)
    while True:
        var count = Int(external_call["recv", Int64](Int32(fd), buffer.unsafe_ptr(), Int64(len(buffer)), MSG_DONTWAIT))
        if count > 0:
            for index in range(count):
                into.append(buffer[index])
            continue
        if count == 0:
            return False
        return last_errno() == EAGAIN


def connect_livelink(path: String) -> Int:
    """A client connection (the scripted test client; Blender has its own)."""
    if path.byte_length() >= 107:
        return -1
    var fd = Int(external_call["socket", Int32](AF_UNIX, Int32(1 | 524288), Int32(0)))
    if fd < 0:
        return -1
    var address = unix_address(path)
    if external_call["connect", Int32](Int32(fd), address.unsafe_ptr(), Int32(110)) != 0:
        close_fd(fd)
        return -1
    return fd


def take_frame(mut buffer: List[UInt8], mut kind: Int, mut payload: List[UInt8]) raises -> Bool:
    """Removes one whole message from the front of buffer. Raises on a header
    that cannot be ours, so the connection is dropped instead of misread."""
    if len(buffer) < HEADER_BYTES:
        return False
    var magic = UInt32(buffer[0]) | (UInt32(buffer[1]) << 8) | (UInt32(buffer[2]) << 16) | (UInt32(buffer[3]) << 24)
    var version = Int(buffer[4]) | (Int(buffer[5]) << 8)
    kind = Int(buffer[6]) | (Int(buffer[7]) << 8)
    var length = Int(buffer[8]) | (Int(buffer[9]) << 8) | (Int(buffer[10]) << 16) | (Int(buffer[11]) << 24)
    if magic != LIVELINK_MAGIC:
        raise Error("not a live link message (bad magic)")
    if version != LIVELINK_VERSION:
        raise Error("live link version " + String(version) + ", the game speaks " + String(LIVELINK_VERSION))
    if length > MAX_PAYLOAD:
        raise Error("a message of " + String(length) + " bytes is over the " + String(MAX_PAYLOAD) + " limit")
    if len(buffer) < HEADER_BYTES + length:
        return False
    payload = List[UInt8]()
    for index in range(HEADER_BYTES, HEADER_BYTES + length):
        payload.append(buffer[index])
    var rest = List[UInt8]()
    for index in range(HEADER_BYTES + length, len(buffer)):
        rest.append(buffer[index])
    buffer = rest^
    return True


struct LiveLinkServer(Movable):
    var listen_fd: Int
    var path: String
    var clients: List[Int]
    var buffers: List[List[UInt8]]
    var received: Int
    var refused: Int
    var last_error: String

    def __init__(out self):
        self.listen_fd = -1
        self.path = ""
        self.clients = List[Int]()
        self.buffers = List[List[UInt8]]()
        self.received = 0
        self.refused = 0
        self.last_error = ""

    def __deinit__(deinit self):
        for fd in self.clients:
            close_fd(fd)
        if self.listen_fd >= 0:
            close_fd(self.listen_fd)
            var owned = self.path
            _ = external_call["unlink", Int32](owned.as_c_string_span())

    def active(self) -> Bool:
        return self.listen_fd >= 0

    def start(mut self, path: String) -> Bool:
        """Listens on path. A socket file left by a game that crashed is
        replaced; a live one (another game listening) is not."""
        if path.byte_length() >= 107:
            self.last_error = "socket path too long: " + path
            return False
        _ = external_call["signal", Int](Int32(13), Int(1))  # SIGPIPE ignored: a closed client is an error code
        var probe = connect_livelink(path)
        if probe >= 0:
            close_fd(probe)
            self.last_error = "another game is already listening on " + path
            return False
        var owned = path
        _ = external_call["unlink", Int32](owned.as_c_string_span())
        var fd = Int(external_call["socket", Int32](AF_UNIX, SOCK_STREAM_NONBLOCK, Int32(0)))
        if fd < 0:
            self.last_error = "socket() failed"
            return False
        var address = unix_address(path)
        if external_call["bind", Int32](Int32(fd), address.unsafe_ptr(), Int32(110)) != 0 or external_call["listen", Int32](Int32(fd), Int32(MAX_CLIENTS)) != 0:
            close_fd(fd)
            self.last_error = "cannot listen on " + path + " (errno " + String(last_errno()) + ")"
            return False
        self.listen_fd = fd
        self.path = path
        return True

    def drop(mut self, index: Int, reason: String):
        self.last_error = reason
        close_fd(self.clients[index])
        _ = self.clients.pop(index)
        _ = self.buffers.pop(index)

    def poll(mut self) -> List[LiveMessage]:
        """Accepts and reads without blocking; returns every whole message."""
        var messages = List[LiveMessage]()
        if self.listen_fd < 0:
            return messages^
        while len(self.clients) < MAX_CLIENTS:
            var fd = Int(external_call["accept4", Int32](Int32(self.listen_fd), Int(0), Int(0), Int32(2048 | 524288)))
            if fd < 0:
                break
            self.clients.append(fd)
            self.buffers.append(List[UInt8]())
        var index = 0
        while index < len(self.clients):
            var alive = receive_available(self.clients[index], self.buffers[index])
            var bad = False
            while True:
                var kind = 0
                var payload = List[UInt8]()
                try:
                    if not take_frame(self.buffers[index], kind, payload):
                        break
                    var message = decode_message(kind, payload^)
                    message.client = self.clients[index]
                    messages.append(message^)
                    self.received += 1
                except error:
                    self.refused += 1
                    _ = send_all(self.clients[index], encode_ack(kind, "refused: " + String(error)))
                    self.drop(index, String(error))
                    bad = True
                    break
            if bad:
                continue
            if not alive:
                self.drop(index, "client hung up")
                continue
            index += 1
        return messages^

    def acknowledge(mut self, message: LiveMessage, text: String):
        for fd in self.clients:
            if fd == message.client:
                _ = send_all(fd, encode_ack(message.kind, text))
