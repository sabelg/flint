"""Microphone capture over the PulseAudio native protocol (`flint.platform.pulse_record`).

Opens a record stream (protocol 13 `CREATE_RECORD_STREAM`) on the same hand-written client
`flint.platform.pulse` uses for playback, so it works on PulseAudio and PipeWire's
pipewire-pulse without libpulse. The server converts to the rate and channel count asked
for (16 kHz mono S16LE for speech models), so no resampling is needed here.
"""

from std.ffi import external_call
from flint.platform.pulse import (
    PulseClient, COMMAND_CHANNEL, SAMPLE_S16LE, NO_INDEX, put_be32, tag_u32, tag_bool, tag_string, tag_null_string,
    tag_proplist, get_be32,
)

comptime COMMAND_CREATE_RECORD_STREAM: UInt32 = 5
comptime CHANNEL_MONO: UInt8 = 0


def create_record_body(tag: UInt32, rate: UInt32, fragment_bytes: UInt32, source: String = "") -> List[UInt8]:
    var body = List[UInt8]()
    tag_u32(body, COMMAND_CREATE_RECORD_STREAM)
    tag_u32(body, tag)
    body.append(UInt8(ord("a")))   # sample spec: S16LE, mono, rate
    body.append(SAMPLE_S16LE)
    body.append(1)
    put_be32(body, rate)
    body.append(UInt8(ord("m")))   # channel map: mono
    body.append(1)
    body.append(CHANNEL_MONO)
    tag_u32(body, NO_INDEX)
    if source.byte_length() > 0:
        tag_string(body, source)
    else:
        tag_null_string(body)       # the default source
    tag_u32(body, 0xFFFFFFFF)       # maximum length: server default
    tag_bool(body, False)           # not corked
    tag_u32(body, fragment_bytes)
    for _ in range(7):              # no_remap through variable_rate
        tag_bool(body, False)
    tag_bool(body, False)           # no peak detection
    tag_bool(body, True)            # latency follows fragment_bytes
    tag_proplist(body, "media.name", "Flint voice input")
    tag_u32(body, NO_INDEX)         # direct_on_input: none (read from protocol 13 on)
    return body^


struct PulseRecorder(Movable):
    """`open()` then call `read()` every frame: it returns the samples (float32, [-1, 1]) that
    arrived since the last call."""
    var client: PulseClient
    var channel: UInt32
    var rate: Int
    var error: String
    var frames_read: Int

    def __init__(out self):
        self.client = PulseClient()
        self.channel = 0
        self.rate = 16000
        self.error = ""
        self.frames_read = 0

    def open(mut self, rate: Int = 16000, fragment_ms: Int = 20, source: String = "") -> Bool:
        self.rate = rate
        if not self.client.handshake():
            self.error = self.client.error
            return False
        var reply = List[UInt8]()
        var tag = self.client.new_tag()
        var frag = UInt32(rate * fragment_ms // 1000 * 2)
        if not self.client.request(create_record_body(tag, UInt32(rate), frag, source), tag, reply):
            self.error = self.client.error
            return False
        if len(reply) < 15:
            self.error = "sound server sent a short record stream reply"
            self.client.close()
            return False
        self.channel = get_be32(reply, 11)
        return True

    def is_open(self) -> Bool:
        return self.client.connected()

    def read(mut self, wait_ms: Int = 0) -> List[Float32]:
        var out = List[Float32]()
        if not self.client.connected():
            return out^
        if wait_ms > 0:
            var descriptor = List[UInt8](length=8, fill=0)
            descriptor[0] = UInt8(self.client.fd & 255)
            descriptor[1] = UInt8((self.client.fd >> 8) & 255)
            descriptor[4] = 1  # POLLIN
            _ = external_call["poll", Int32](descriptor.unsafe_ptr(), UInt64(1), Int32(wait_ms))
        self.client.receive()
        var channel = UInt32(0)
        var payload = List[UInt8]()
        while self.client.connected() and self.client.next_frame(channel, payload):
            if channel == COMMAND_CHANNEL:
                self.client.handle_event(payload)
                continue
            if channel != self.channel:
                continue
            var n = len(payload) // 2
            for i in range(n):
                var s = Int(payload[2 * i]) | (Int(payload[2 * i + 1]) << 8)
                if s >= 32768:
                    s -= 65536
                out.append(Float32(s) / 32768.0)
            self.frames_read += n
        if not self.client.connected():
            self.error = self.client.error
        return out^

    def close(mut self):
        self.client.close()
