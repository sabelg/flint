"""Compare complete Flint Trace Event captures, grouping repeated scopes by frame."""
from std.collections import Dict
from std.math import ceil, isfinite
from flint.assets.json import JsonDocument, parse_json
from flint.platform.os import read_bytes


struct Samples(Copyable, Movable):
    var values: List[Float64]

    def __init__(out self, count: Int):
        self.values = List[Float64](length=count, fill=-1)


struct Capture(Movable):
    var starts: List[Float64]
    var ids: Dict[Int, Int]
    var names: List[String]
    var columns: List[Samples]
    var indexes: Dict[String, Int]

    def __init__(out self, path: String) raises:
        self.starts = List[Float64]()
        self.ids = Dict[Int, Int]()
        self.names = List[String]()
        self.columns = List[Samples]()
        self.indexes = Dict[String, Int]()
        var doc = parse_json(read_bytes(path))
        if not doc.valid:
            raise Error("Unreadable or incomplete trace: " + path)
        var events = doc.root
        if doc.nodes[events].kind == 5:
            events = doc.object_get(events, "traceEvents")
        var frames = List[Float64]()
        for i in range(doc.array_size(events)):
            var e = doc.at(events, i)
            if doc.text(doc.object_get(e, "ph")) != "X" or doc.text(doc.object_get(e, "name")) != "frame":
                continue
            var start = doc.number(doc.object_get(e, "ts"), -1)
            var dur = doc.number(doc.object_get(e, "dur"), -1)
            var id = doc.integer(doc.object_get(doc.object_get(e, "args"), "frame"), len(frames))
            if (
                not isfinite(start)
                or not isfinite(dur)
                or start < 0
                or dur < 0
                or id < 0
                or id in self.ids
                or (len(frames) > 0 and start < self.starts[len(frames)-1])
            ):
                raise Error("Invalid or unordered Flint frame in " + path)
            self.ids[id] = len(frames)
            self.starts.append(start)
            frames.append(dur * 0.001)
        if len(frames) == 0:
            raise Error("No completed Flint frames in " + path)
        var total = self.column("frame")
        self.columns[total].values = frames^
        for i in range(doc.array_size(events)):
            var e = doc.at(events, i)
            var ph = doc.text(doc.object_get(e, "ph"))
            if ph != "X" and ph != "C":
                continue
            var name = doc.text(doc.object_get(e, "name"))
            if not name or name == "frame":
                continue
            var args = doc.object_get(e, "args")
            var id = doc.integer(doc.object_get(args, "frame"))
            var frame = self.ids.get(id, -1)
            if ph == "C":
                if name == "gpu_ms" and frame >= 0 and args >= 0 and args < len(doc.nodes) and doc.nodes[args].kind == 5:
                    for k in range(len(doc.nodes[args].keys)):
                        var key = doc.nodes[args].keys[k]
                        var ms = doc.number(doc.nodes[args].children[k], -1)
                        if key != "frame" and isfinite(ms) and ms >= 0:
                            self.add(key, frame, ms)
                continue
            var ts = doc.number(doc.object_get(e, "ts"), -1)
            var dur = doc.number(doc.object_get(e, "dur"), -1)
            if not isfinite(ts) or not isfinite(dur) or dur < 0:
                continue
            if id < 0:
                # Legacy CPU events carry a timestamp, while delayed GPU spans carry a frame ID.
                var lo = 0
                var hi = len(self.starts)
                while lo < hi:
                    var mid = (lo + hi) // 2
                    if self.starts[mid] <= ts:
                        lo = mid + 1
                    else:
                        hi = mid
                frame = lo - 1
                if frame >= 0 and ts + dur > self.starts[frame] + self.columns[total].values[frame] * 1000 + 2:
                    continue
            if frame >= 0:
                self.add(name, frame, dur * 0.001)
        if "pace" in self.indexes:
            var work = self.column("frame less pace")
            var pace = self.indexes["pace"]
            for i in range(len(self.starts)):
                self.columns[work].values[i] = max(Float64(0), self.columns[total].values[i] - max(Float64(0), self.columns[pace].values[i]))

    def column(mut self, name: String) -> Int:
        var existing = self.indexes.get(name, -1)
        if existing >= 0:
            return existing
        var idx = len(self.names)
        self.indexes[name] = idx
        self.names.append(name)
        self.columns.append(Samples(len(self.starts)))
        return idx

    def add(mut self, name: String, frame: Int, ms: Float64):
        if frame < 0 or frame >= len(self.starts):
            return
        var idx = self.column(name)
        ref values = self.columns[idx].values
        values[frame] = max(Float64(0), values[frame]) + ms

    def at(self, name: String, frame: Int) -> Float64:
        var idx = self.indexes.get(name, -1)
        if idx < 0 or frame < 0 or frame >= len(self.columns[idx].values):
            return -1
        return self.columns[idx].values[frame]

    def stats(self, name: String) -> Tuple[Float64, Float64, Int]:
        var idx = self.indexes.get(name, -1)
        var measured = List[Float64]()
        var sum = Float64(0)
        if idx >= 0:
            for value in self.columns[idx].values:
                if value >= 0:
                    measured.append(value)
                    sum += value
        if not measured:
            return (-1, -1, 0)
        sort(measured)
        var n = len(measured)
        return (sum / Float64(n), measured[Int(ceil(Float64(n) * 0.95)) - 1], n)
