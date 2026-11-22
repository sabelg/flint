"""Frame-authored combat windows. End and close are exclusive."""
from std.ffi import external_call
from flint.assets.json import JsonDocument, parse_json
from flint.platform.os import read_bytes, write_synced, durable_rename


def frame_number(doc: JsonDocument, key: String) raises -> Int:
    var node = doc.object_get(doc.root, key)
    var n = doc.number(node, -1)
    if not (n >= 0 and n <= 10000) or n != Float64(Int(n)):
        raise Error("Invalid integer in combat timing: " + key)
    return Int(n)


struct CombatTiming(Copyable, Movable):
    var fps: Int
    var end: Int
    var impact: Int
    var close: Int
    var chain: Int
    var reach: Float32

    def __init__(out self, path: String) raises:
        var doc = parse_json(read_bytes(path))
        if not doc.valid or frame_number(doc, "version") != 1:
            raise Error("Expected a Flint combat timing preset: " + path)
        self.fps = frame_number(doc, "fps")
        self.end = frame_number(doc, "end")
        self.impact = frame_number(doc, "impact")
        self.close = frame_number(doc, "close")
        self.chain = frame_number(doc, "chain")
        self.reach = Float32(doc.number(doc.object_get(doc.root, "reach")))
        if not (
            1 <= self.fps <= 120
            and 0 <= self.impact < self.close <= self.chain < self.end
            and self.reach > 0
            and self.reach <= 100
        ):
            raise Error("Invalid combat frame order, rate, or reach")

    def active(self, frame: Float32) -> Bool:
        return Float32(self.impact) <= frame < Float32(self.close)

    def can_chain(self, frame: Float32) -> Bool:
        return Float32(self.chain) <= frame < Float32(self.end)

    def marker(self, index: Int) -> Int:
        if index == 0:
            return self.impact
        if index == 1:
            return self.close
        if index == 2:
            return self.chain
        return -1

    def move_marker(mut self, index: Int, frame: Int):
        if index == 0:
            self.impact = min(self.close - 1, max(0, frame))
        elif index == 1:
            self.close = min(self.chain, max(self.impact + 1, frame))
        elif index == 2:
            self.chain = min(self.end - 1, max(self.close, frame))

    def save(self, path: String) raises:
        var result = String('{\n  "version": 1,\n  "fps": ') + String(self.fps)
        result += (
            ',\n  "end": '
            + String(self.end)
            + ',\n  "impact": '
            + String(self.impact)
        )
        result += (
            ',\n  "close": '
            + String(self.close)
            + ',\n  "chain": '
            + String(self.chain)
        )
        result += ',\n  "reach": ' + String(self.reach) + "\n}\n"
        var data = List[UInt8]()
        for byte in result.as_bytes():
            data.append(byte)
        var temporary = (
            path + ".tmp." + String(external_call["getpid", Int32]())
        )
        if not write_synced(temporary, data) or not durable_rename(
            temporary, path
        ):
            raise Error("Could not save combat timing: " + path)
