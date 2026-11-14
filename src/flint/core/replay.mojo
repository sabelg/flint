"""Input recording, headless replay and per-tick state checksums.

A recording is the exact input the simulation saw on every fixed tick, plus a
checksum of the authoritative state after that tick. Replaying feeds the same
input into a fresh Game and compares checksums tick by tick, so the first
divergence is reported with the tick number and the part of the state that
differs. Presentation (rendering, audio, animation) is not part of the state.

File layout, little-endian:
    "SDRP"  u32 version  i32 start_layer  u32 flags
    (version 2 on) u32 stamp_length, stamp bytes: the build that recorded it
    u32 blob_count
    blob_count x (u32 name_length, name bytes, u32 data_length, data bytes)
    then fixed-size tick records until the end of the file (TICK_BYTES each).
Blobs carry the save files and settings that existed when recording started,
so a replay never reads or writes the player's real saves. Version 1 files
(no stamp) still load.

The checksums: from version 3 (package 0.10) each of the CHECK_SECTIONS is
the FNV-1a hash of that part of the serialised simulation state, the same
versioned blobs a snapshot holds (Game.serialize_section; SECTION_* below).
Versions 1 and 2 hashed chosen fields; replaying one checks it with
Game.legacy_state_checksums, the same fields, so an older recording or crash
report still replays where the gameplay it covers has not changed since (0.10
made spawns wait for the end of the tick, which moves thrown chunks and shots
by one tick).

A crash report's replay ends with the input of the tick that crashed and all
eight checksums zero (core/crash.mojo); `is_crash_tick` recognises it.

Version 4 (package 3.0) adds dev records: what the tuning workbench did
between two ticks (a live tunables change, a teleport, a marked moment), each
one record of TICK_BYTES in the tick stream, so the stream keeps its fixed
step. A dev record's checksum area starts with DEV_MAGIC and the record's
kind; its input area holds a u16 payload length and the payload. A replay
applies one before the tick that follows it, as play did. `count_ticks` and
`skip_dev_records` let a reader step over them; version 3 files have none.
"""
from flint.core.math import Vec3
from flint.platform.input import InputState, MAX_PADS, PAD_CONTROLS
from flint.core.build_info import build_stamp

comptime REPLAY_VERSION: Int = 4  # 3: checksums hash the serialised tables (package 0.10); 4: dev records (3.0)
comptime REPLAY_OLDEST_VERSION: Int = 1
comptime REPLAY_PAD_FD: Int = 1073741824  # marks a replayed pad as present; never read
comptime KEY_COUNT: Int = 512
comptime PAD_AXES: Int = 8
comptime CHECK_SECTIONS: Int = 8
comptime SECTION_PLAYER: Int = 0
comptime SECTION_CAMERA: Int = 1
comptime SECTION_ENEMIES: Int = 2      # enemies and projectiles
comptime SECTION_BODIES: Int = 3
comptime SECTION_TERRAIN: Int = 4      # the voxel edit journal
comptime SECTION_OBJECTS: Int = 5      # collectibles, props, triggers, lights, the boss
comptime SECTION_PROGRESSION: Int = 6
comptime SECTION_FLOW: Int = 7         # layer, mode, clocks, the tick's sim events
comptime PAD_BYTES: Int = 1 + PAD_AXES * 4 + 4 + 4
comptime INPUT_BYTES: Int = KEY_COUNT // 8 * 3 + 2 + 12 + MAX_PADS * PAD_BYTES
comptime TICK_BYTES: Int = INPUT_BYTES + CHECK_SECTIONS * 8
comptime DEV_MAGIC: UInt64 = 0x44524F4345525644  # "DVRECORD" in the first checksum word
comptime DEV_PAYLOAD_BYTES: Int = INPUT_BYTES - 2
comptime DEV_TUNES: Int = 1     # live tunables changed (gameplay/live_tunes.mojo)
comptime DEV_TELEPORT: Int = 2  # Maroon moved to a bookmark (gameplay/dev_events.mojo)
comptime DEV_MARK: Int = 3      # a playtester marked this moment; changes nothing
comptime DEV_DEMO: Int = 4      # a demo's test pieces were built (package 2.14: mechanics.demo)


def section_name(index: Int) -> String:
    var names = ["player", "camera", "enemies", "bodies", "terrain", "objects", "progression", "flow"]
    return String(names[index]) if index >= 0 and index < CHECK_SECTIONS else "unknown"


def float_bits(value: Float32) -> UInt32:
    var copy = value
    return Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]


def bits_float(bits: UInt32) -> Float32:
    var copy = bits
    return Pointer(to=copy).unsafe_bitcast[Float32]()[unsafe_offset=0]


struct Fnv64(Copyable, Movable):
    """FNV-1a, 64-bit. Floats are hashed by their exact bit pattern."""
    var value: UInt64

    def __init__(out self):
        self.value = 0xCBF29CE484222325

    def add_byte(mut self, byte: UInt8):
        self.value = (self.value ^ UInt64(byte)) * 0x100000001B3

    def add_u32(mut self, word: UInt32):
        for shift in range(4):
            self.add_byte(UInt8((word >> UInt32(shift * 8)) & 0xFF))

    def add_int(mut self, number: Int):
        var word = UInt64(number)
        for shift in range(8):
            self.add_byte(UInt8((word >> UInt64(shift * 8)) & 0xFF))

    def add_bool(mut self, flag: Bool):
        self.add_byte(UInt8(1) if flag else UInt8(0))

    def add_f32(mut self, number: Float32):
        self.add_u32(float_bits(number))

    def add_vec3(mut self, v: Vec3):
        self.add_f32(v.x)
        self.add_f32(v.y)
        self.add_f32(v.z)


# ---------------------------------------------------------------------------
# Byte writer and reader
# ---------------------------------------------------------------------------

def put_u32(mut out: List[UInt8], word: UInt32):
    for shift in range(4):
        out.append(UInt8((word >> UInt32(shift * 8)) & 0xFF))


def put_u64(mut out: List[UInt8], word: UInt64):
    for shift in range(8):
        out.append(UInt8((word >> UInt64(shift * 8)) & 0xFF))


def put_bits(mut out: List[UInt8], flags: List[Bool], count: Int):
    var byte = UInt8(0)
    for index in range(count):
        if index < len(flags) and flags[index]:
            byte |= UInt8(1) << UInt8(index % 8)
        if index % 8 == 7:
            out.append(byte)
            byte = 0
    if count % 8 != 0:
        out.append(byte)


struct ByteReader(Movable):
    var data: List[UInt8]
    var offset: Int
    var failed: Bool

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.offset = 0
        self.failed = False

    def remaining(self) -> Int:
        return len(self.data) - self.offset

    def u8(mut self) -> UInt8:
        if self.offset >= len(self.data):
            self.failed = True
            return 0
        var value = self.data[self.offset]
        self.offset += 1
        return value

    def u32(mut self) -> UInt32:
        var value = UInt32(0)
        for shift in range(4):
            value |= UInt32(self.u8()) << UInt32(shift * 8)
        return value

    def u64(mut self) -> UInt64:
        var value = UInt64(0)
        for shift in range(8):
            value |= UInt64(self.u8()) << UInt64(shift * 8)
        return value

    def bits(mut self, count: Int) -> List[Bool]:
        var flags = List[Bool](length=count, fill=False)
        var byte = UInt8(0)
        for index in range(count):
            if index % 8 == 0:
                byte = self.u8()
            flags[index] = (byte >> UInt8(index % 8)) & 1 == 1
        return flags^

    def bytes(mut self, count: Int) -> List[UInt8]:
        var out = List[UInt8]()
        if count < 0 or self.offset + count > len(self.data):
            self.failed = True
            return out^
        for index in range(count):
            out.append(self.data[self.offset + index])
        self.offset += count
        return out^


# ---------------------------------------------------------------------------
# Input snapshots
# ---------------------------------------------------------------------------

def write_input(mut out: List[UInt8], input: InputState):
    """Everything about the input that the simulation can read this tick."""
    put_bits(out, input.keys, KEY_COUNT)
    put_bits(out, input.pressed, KEY_COUNT)
    put_bits(out, input.released, KEY_COUNT)
    var mouse = UInt16(0)
    for button in range(3):
        if input.mouse[button]:
            mouse |= UInt16(1) << UInt16(button)
        if input.mouse_pressed_state[button]:
            mouse |= UInt16(1) << UInt16(button + 3)
        if input.mouse_released_state[button]:
            mouse |= UInt16(1) << UInt16(button + 6)
    out.append(UInt8(mouse & 0xFF))
    out.append(UInt8(mouse >> 8))
    put_u32(out, float_bits(input.delta_x))
    put_u32(out, float_bits(input.delta_y))
    put_u32(out, float_bits(input.wheel))
    for pad in range(MAX_PADS):
        var flags = UInt8(0)
        if input.pads[pad].fd >= 0:
            flags |= 1
        if input.pads[pad].sony_layout:
            flags |= 2
        out.append(flags)
        for axis in range(PAD_AXES):
            put_u32(out, float_bits(input.pads[pad].axes[axis]))
        put_bits(out, input.pads[pad].down, 32)
        put_bits(out, input.pads[pad].pressed, 32)


def read_input(mut reader: ByteReader, mut input: InputState):
    """Overwrites the input state with one recorded tick. Replayed pads get a
    marker descriptor so the game sees them as connected; nothing reads it."""
    input.keys = reader.bits(KEY_COUNT)
    input.pressed = reader.bits(KEY_COUNT)
    input.released = reader.bits(KEY_COUNT)
    var mouse = UInt16(reader.u8()) | (UInt16(reader.u8()) << 8)
    for button in range(3):
        input.mouse[button] = (mouse >> UInt16(button)) & 1 == 1
        input.mouse_pressed_state[button] = (mouse >> UInt16(button + 3)) & 1 == 1
        input.mouse_released_state[button] = (mouse >> UInt16(button + 6)) & 1 == 1
    input.delta_x = bits_float(reader.u32())
    input.delta_y = bits_float(reader.u32())
    input.wheel = bits_float(reader.u32())
    for pad in range(MAX_PADS):
        var flags = reader.u8()
        input.pads[pad].fd = REPLAY_PAD_FD if flags & 1 == 1 else -1
        input.pads[pad].sony_layout = flags & 2 == 2
        for axis in range(PAD_AXES):
            input.pads[pad].axes[axis] = bits_float(reader.u32())
        var down = reader.bits(32)
        var pressed = reader.bits(32)
        for control in range(PAD_CONTROLS):
            input.pads[pad].down[control] = down[control]
            input.pads[pad].pressed[control] = pressed[control]


def release_replay_pads(mut input: InputState):
    """Clears the marker descriptors before the input state is closed."""
    for pad in range(MAX_PADS):
        if input.pads[pad].fd == REPLAY_PAD_FD:
            input.pads[pad].fd = -1


# ---------------------------------------------------------------------------
# Recording file
# ---------------------------------------------------------------------------

struct ReplayBlob(Copyable, Movable):
    var name: String
    var data: List[UInt8]

    def __init__(out self, name: String, var data: List[UInt8]):
        self.name = name
        self.data = data^


def encode_header(start_layer: Int, flags: Int, blobs: List[ReplayBlob], stamp: String = build_stamp()) -> List[UInt8]:
    var out = List[UInt8]()
    for ch in "SDRP".as_bytes():
        out.append(ch)
    put_u32(out, UInt32(REPLAY_VERSION))
    put_u32(out, UInt32(start_layer))
    put_u32(out, UInt32(flags))
    var stamp_bytes = stamp.as_bytes()
    put_u32(out, UInt32(len(stamp_bytes)))
    for ch in stamp_bytes:
        out.append(ch)
    put_u32(out, UInt32(len(blobs)))
    for blob in blobs:
        var name = blob.name.as_bytes()
        put_u32(out, UInt32(len(name)))
        for ch in name:
            out.append(ch)
        put_u32(out, UInt32(len(blob.data)))
        for byte in blob.data:
            out.append(byte)
    return out^


def encode_checks(mut out: List[UInt8], checks: List[UInt64]):
    for index in range(CHECK_SECTIONS):
        put_u64(out, checks[index] if index < len(checks) else UInt64(0))


struct ReplayHeader(Movable):
    var valid: Bool
    var version: Int
    var start_layer: Int
    var flags: Int
    var stamp: String  # empty for version 1
    var blobs: List[ReplayBlob]

    def __init__(out self):
        self.valid = False
        self.version = 0
        self.start_layer = 0
        self.flags = 0
        self.stamp = String()
        self.blobs = List[ReplayBlob]()


def decode_header(mut reader: ByteReader) -> ReplayHeader:
    var header = ReplayHeader()
    var magic = reader.bytes(4)
    if len(magic) != 4 or Int(magic[0]) != 83 or Int(magic[1]) != 68 or Int(magic[2]) != 82 or Int(magic[3]) != 80:
        return header^
    header.version = Int(reader.u32())
    header.start_layer = Int(reader.u32())
    header.flags = Int(reader.u32())
    if header.version < REPLAY_OLDEST_VERSION or header.version > REPLAY_VERSION:
        return header^
    if header.version >= 2:
        var length = Int(reader.u32())
        if length > 256:
            return header^
        var stamp_bytes = reader.bytes(length)
        header.stamp = String(unsafe_from_utf8=stamp_bytes) if len(stamp_bytes) > 0 else String()
    var count = Int(reader.u32())
    if count > 64:
        return header^
    for _ in range(count):
        var name_bytes = reader.bytes(Int(reader.u32()))
        var name = String(unsafe_from_utf8=name_bytes) if len(name_bytes) > 0 else String()
        var data = reader.bytes(Int(reader.u32()))
        header.blobs.append(ReplayBlob(name, data^))
    header.valid = not reader.failed
    return header^


# ---------------------------------------------------------------------------
# Dev records (version 4)
# ---------------------------------------------------------------------------

def encode_dev_record(mut out: List[UInt8], kind: Int, payload: List[UInt8]):
    """One dev record of TICK_BYTES. Payloads longer than DEV_PAYLOAD_BYTES are
    the caller's to split (live_tunes.mojo sends several records)."""
    var count = min(len(payload), DEV_PAYLOAD_BYTES)
    out.append(UInt8(count & 0xFF))
    out.append(UInt8(count >> 8))
    for index in range(count):
        out.append(payload[index])
    for _ in range(DEV_PAYLOAD_BYTES - count):
        out.append(0)
    put_u64(out, DEV_MAGIC)
    put_u64(out, UInt64(kind))
    for _ in range(CHECK_SECTIONS - 2):
        put_u64(out, 0)


def word_at(data: List[UInt8], at: Int) -> UInt64:
    var value = UInt64(0)
    for byte in range(8):
        value |= UInt64(data[at + byte]) << UInt64(byte * 8)
    return value


def is_dev_record_at(data: List[UInt8], offset: Int) -> Bool:
    """Whether the TICK_BYTES record at `offset` is a dev record."""
    if offset + TICK_BYTES > len(data):
        return False
    return word_at(data, offset + INPUT_BYTES) == DEV_MAGIC


struct DevRecord(Copyable, Movable):
    var kind: Int
    var payload: List[UInt8]

    def __init__(out self, kind: Int, var payload: List[UInt8]):
        self.kind = kind
        self.payload = payload^


def read_dev_record(mut reader: ByteReader) -> DevRecord:
    """Reads the dev record at the reader (check `is_dev_record_at` first)."""
    var start = reader.offset
    var count = Int(reader.u8()) | (Int(reader.u8()) << 8)
    var payload = reader.bytes(min(count, DEV_PAYLOAD_BYTES))
    reader.offset = start + INPUT_BYTES
    _ = reader.u64()
    var kind = Int(reader.u64())
    reader.offset = start + TICK_BYTES
    return DevRecord(kind, payload^)


def take_dev_records(mut reader: ByteReader) -> List[DevRecord]:
    """Every dev record from the reader up to the next tick record."""
    var records = List[DevRecord]()
    while is_dev_record_at(reader.data, reader.offset):
        records.append(read_dev_record(reader))
    return records^


def skip_dev_records(mut reader: ByteReader) -> Int:
    var skipped = 0
    while is_dev_record_at(reader.data, reader.offset):
        reader.offset += TICK_BYTES
        skipped += 1
    return skipped


def count_ticks(reader: ByteReader) -> Int:
    """Tick records from the reader to the end, dev records not counted."""
    var ticks = 0
    var at = reader.offset
    while at + TICK_BYTES <= len(reader.data):
        if not is_dev_record_at(reader.data, at):
            ticks += 1
        at += TICK_BYTES
    return ticks


def tick_offsets(reader: ByteReader) -> List[Int]:
    """The byte offset of every tick record from the reader on (dev records
    before a tick belong to it: its offset is the first of them)."""
    var offsets = List[Int]()
    var at = reader.offset
    var group = at
    while at + TICK_BYTES <= len(reader.data):
        if not is_dev_record_at(reader.data, at):
            offsets.append(group)
            group = at + TICK_BYTES
        at += TICK_BYTES
    return offsets^


def is_crash_tick(checks: List[UInt64]) -> Bool:
    """All checksums zero: the tick a crash report's replay ends with, which crashed
    before it could be checked. FNV-1a never hashes to zero for real state."""
    for value in checks:
        if value != 0:
            return False
    return len(checks) > 0
