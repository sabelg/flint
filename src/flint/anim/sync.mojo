"""Clip facts the runtime reads, sync markers and animation events (package 3.3).

Sync markers are a clip's foot events (`footstep_L`, `footstep_R`, from the
Blender markers `event:footstep_L` and `event:footstep_R`). A sync phase
counts markers: phase 1.25 is a quarter of the way from the second marker
to the third. Clips that list the same feet in the same order play at the
same phase, so in a blend between them the feet land together whatever the
clips' lengths and the spacing of their steps (Unreal's sync markers; Bungie
and Ubisoft call them foot-sync markers). A clip without markers has one
segment, the whole clip, starting at time 0.
"""
from flint.core.math import Vec3
from flint.assets.manifest import ManifestClip


struct ClipEventInfo(Copyable, ImplicitlyCopyable, Movable):
    var kind: String
    var arg: String
    var time: Float32

    def __init__(out self, kind: String, arg: String, time: Float32):
        self.kind = kind
        self.arg = arg
        self.time = time


struct ClipInfo(Copyable, Movable):
    """One clip, in the game's clip order: what its manifest says."""
    var name: String
    var duration: Float32
    var loop: Bool
    var speed: Float32       # ds_speed: ground speed at 1x playback (m/s)
    var root_motion: Vec3    # the root bone's travel over the clip (game axes)
    var events: List[ClipEventInfo]

    def __init__(out self):
        self.name = ""
        self.duration = 0
        self.loop = False
        self.speed = 0
        self.root_motion = Vec3()
        self.events = List[ClipEventInfo]()

    def __init__(out self, name: String, duration: Float32, loop: Bool, speed: Float32):
        self.name = name
        self.duration = duration
        self.loop = loop
        self.speed = speed
        self.root_motion = Vec3()
        self.events = List[ClipEventInfo]()

    @staticmethod
    def from_manifest(clip: ManifestClip) -> ClipInfo:
        var info = ClipInfo(clip.name, clip.length, clip.loop, clip.speed)
        info.root_motion = clip.root_motion
        for event in clip.events:
            info.events.append(ClipEventInfo(event.kind, event.arg, event.time))
        return info^

    def event_time(self, kind: String, fallback: Float32 = -1) -> Float32:
        for event in self.events:
            if event.kind == kind:
                return event.time
        return fallback


def is_foot_event(kind: String) -> Bool:
    return kind == "footstep_L" or kind == "footstep_R"


struct SyncTrack(Copyable, Movable):
    """A clip's sync markers: times in [0, duration), in order, starting
    at the first left foot."""
    var duration: Float32
    var times: List[Float32]
    var names: List[String]

    def __init__(out self):
        self.duration = 0
        self.times = List[Float32]()
        self.names = List[String]()

    def __init__(out self, clip: ClipInfo):
        self.duration = clip.duration
        self.times = List[Float32]()
        self.names = List[String]()
        var times = List[Float32]()
        var names = List[String]()
        for event in clip.events:
            if is_foot_event(event.kind) and event.time >= 0 and (event.time < clip.duration or clip.duration <= 0):
                # Insertion keeps them sorted by time (a handful per clip).
                var at = len(times)
                while at > 0 and times[at - 1] > event.time:
                    at -= 1
                times.insert(at, event.time)
                names.insert(at, event.kind)
        var first = 0
        for index in range(len(names)):
            if names[index] == "footstep_L":
                first = index
                break
        for step in range(len(times)):
            var index = (first + step) % len(times)
            self.times.append(times[index])
            self.names.append(names[index])

    def markers(self) -> Int:
        return max(1, len(self.times))

    def matches(self, other: SyncTrack) -> Bool:
        """Same feet in the same order: the two can share a phase."""
        if len(self.names) != len(other.names) or len(self.names) == 0:
            return False
        for index in range(len(self.names)):
            if self.names[index] != other.names[index]:
                return False
        return True

    def start(self, marker: Int) -> Float32:
        if len(self.times) == 0:
            return 0
        return self.times[marker % len(self.times)]

    def segment_length(self, marker: Int) -> Float32:
        """Seconds from this marker to the next (round the loop)."""
        if len(self.times) <= 1 or self.duration <= 0:
            return max(Float32(0.0001), self.duration)
        var index = marker % len(self.times)
        var a = self.times[index]
        var b = self.times[(index + 1) % len(self.times)]
        if b <= a:
            b += self.duration
        return max(Float32(0.0001), b - a)

    def phase_at(self, time: Float32) -> Float32:
        """The sync phase in [0, markers) at a clip time."""
        if self.duration <= 0:
            return 0
        var t = wrap_time(time, self.duration)
        if len(self.times) == 0:
            return t / self.duration
        var n = len(self.times)
        for step in range(n):
            var a = self.times[step]
            var length = self.segment_length(step)
            var into = t - a
            if into < 0:
                into += self.duration
            if into < length:
                return Float32(step) + into / length
        return 0

    def time_at(self, phase: Float32) -> Float32:
        """The clip time at a sync phase."""
        if self.duration <= 0:
            return 0
        var n = self.markers()
        var p = wrap_time(phase, Float32(n))
        var marker = min(n - 1, Int(p))
        var fraction = p - Float32(marker)
        if len(self.times) == 0:
            return fraction * self.duration
        return wrap_time(self.start(marker) + fraction * self.segment_length(marker), self.duration)


def wrap_time(time: Float32, length: Float32) -> Float32:
    if length <= 0:
        return 0
    var t = time
    if t >= length or t < 0:
        t = t - length * Float32(Int(t / length))
        if t < 0:
            t += length
        if t >= length:
            t -= length
    return max(Float32(0), t)


struct FiredEvent(Copyable, ImplicitlyCopyable, Movable):
    """An event a clip passed this frame."""
    var kind: String
    var arg: String
    var clip: Int
    var weight: Float32

    def __init__(out self, kind: String, arg: String, clip: Int, weight: Float32):
        self.kind = kind
        self.arg = arg
        self.clip = clip
        self.weight = weight


def passed_events(clip: ClipInfo, clip_index: Int, before: Float32, after: Float32, looped: Int, weight: Float32, mut out: List[FiredEvent]):
    """Events in (before, after] of one play, plus whole loops: looped is
    how many times playback wrapped past the end this frame. An event at
    time 0 fires when the clip starts (before < 0)."""
    if clip.duration <= 0:
        return
    for event in clip.events:
        if event.kind == "cancel_from" or event.kind == "cancel_until":
            continue  # window markers, read by the state machine
        var times = 0
        if looped == 0:
            if event.time > before and event.time <= after:
                times = 1
        else:
            if event.time > before:
                times += 1
            times += looped - 1
            if event.time <= after:
                times += 1
        for _ in range(min(times, 4)):
            out.append(FiredEvent(event.kind, event.arg, clip_index, weight))
