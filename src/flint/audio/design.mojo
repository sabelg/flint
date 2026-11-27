"""Data-Driven Sound-Design Layer: Events, Containers, Game Parameters & Mixer Snapshots
(`flint.audio.design`).

Runs entirely on the game/main thread on top of `AudioServer`:
  - `SoundEvent` containers: `single`, `random` (weighted, avoids the last `avoid_repeat`
    picks), `sequence` (round-robin), and `switch` (child chosen by a game parameter's
    value range). Children are clips or nested events. Per-event volume/pitch with random
    variation, bus, looping, spatial, streaming and an attenuation preset.
  - `GameParameter` (RTPC): named floats with optional slew; `RtpcBinding` maps a
    parameter through a piecewise-linear `ParamCurve` onto voice `volume` (linear),
    `volume_db`, `pitch` (ratio) or `lowpass` (Hz). Live instances are re-evaluated in
    `SoundBank.update`.
  - `MixerSnapshot`: named bus-gain sets; `activate_snapshot(name, seconds)` blends every
    known bus from its current gain to the snapshot (unlisted buses -> 1.0) over time.
  - `SoundBank.load_json(text, server)`: the whole bank from a small JSON description.
"""
from std.collections import Dict, List
from std.math import exp, abs, isfinite
from flint.core.math import Vec3
from flint.assets.json import JsonDocument, parse_json
from flint.audio.attenuation import AttenuationPreset, attenuation_curve_from_name
from flint.audio.server import (
    AudioServer,
    SoundHandle,
    BUS_MASTER,
    BUS_SFX,
    BUS_MUSIC,
    BUS_VOICE,
    BUS_AMBIENCE,
    BUS_UI,
)

comptime CONTAINER_SINGLE: Int = 0
comptime CONTAINER_RANDOM: Int = 1
comptime CONTAINER_SEQUENCE: Int = 2
comptime CONTAINER_SWITCH: Int = 3

comptime RTPC_VOLUME: Int = 0
comptime RTPC_VOLUME_DB: Int = 1
comptime RTPC_PITCH: Int = 2
comptime RTPC_LOWPASS: Int = 3

comptime _MAX_EVENT_DEPTH: Int = 8


def container_from_name(name: String) -> Int:
    if name == "random":
        return CONTAINER_RANDOM
    if name == "sequence":
        return CONTAINER_SEQUENCE
    if name == "switch":
        return CONTAINER_SWITCH
    return CONTAINER_SINGLE


def rtpc_target_from_name(name: String) -> Int:
    if name == "volume_db":
        return RTPC_VOLUME_DB
    if name == "pitch":
        return RTPC_PITCH
    if name == "lowpass" or name == "filter" or name == "cutoff":
        return RTPC_LOWPASS
    if name == "volume":
        return RTPC_VOLUME
    return -1


struct ParamCurve(Copyable, Movable):
    """Piecewise-linear curve (clamped at both ends). Empty curve = identity."""
    var xs: List[Float32]
    var ys: List[Float32]

    def __init__(out self):
        self.xs = List[Float32]()
        self.ys = List[Float32]()

    def add(mut self, x: Float32, y: Float32):
        # keep sorted by x (insertion sort; curves are tiny)
        var i = len(self.xs)
        self.xs.append(x)
        self.ys.append(y)
        while i > 0 and self.xs[i - 1] > self.xs[i]:
            var tx = self.xs[i - 1]
            var ty = self.ys[i - 1]
            self.xs[i - 1] = self.xs[i]
            self.ys[i - 1] = self.ys[i]
            self.xs[i] = tx
            self.ys[i] = ty
            i -= 1

    def evaluate(self, x: Float32) -> Float32:
        var n = len(self.xs)
        if n == 0:
            return x
        if x <= self.xs[0]:
            return self.ys[0]
        for i in range(1, n):
            if x <= self.xs[i]:
                var span = self.xs[i] - self.xs[i - 1]
                var w = (x - self.xs[i - 1]) / span if span > Float32(1.0e-9) else Float32(1.0)
                return self.ys[i - 1] + (self.ys[i] - self.ys[i - 1]) * w
        return self.ys[n - 1]


struct GameParameter(Copyable, Movable):
    var name: String
    var value: Float32
    var target: Float32
    var min_value: Float32
    var max_value: Float32
    var slew_per_sec: Float32  # 0 = changes apply instantly

    def __init__(out self, name: String, default: Float32 = 0.0, min_value: Float32 = 0.0, max_value: Float32 = 1.0, slew_per_sec: Float32 = 0.0):
        self.name = name
        self.min_value = min(min_value, max_value)
        self.max_value = max(min_value, max_value)
        self.value = max(self.min_value, min(self.max_value, default))
        self.target = self.value
        self.slew_per_sec = max(Float32(0.0), slew_per_sec)


struct RtpcBinding(Copyable, Movable):
    var param: Int
    var target: Int
    var curve: ParamCurve

    def __init__(out self, param: Int, target: Int, var curve: ParamCurve):
        self.param = param
        self.target = target
        self.curve = curve^


struct SoundEntry(Copyable, Movable):
    """Container child: a clip (`asset_id`, resolved from `clip`) or a nested event (`event`)."""
    var clip: String
    var asset_id: Int
    var event_name: String
    var weight: Float32
    var switch_min: Float32
    var switch_max: Float32

    def __init__(out self, clip: String = "", asset_id: Int = -1, event_name: String = "", weight: Float32 = 1.0):
        self.clip = clip
        self.asset_id = asset_id
        self.event_name = event_name
        self.weight = max(Float32(0.0), weight)
        self.switch_min = Float32(-3.0e38)
        self.switch_max = Float32(3.0e38)


struct SoundEvent(Copyable, Movable):
    var name: String
    var container: Int
    var children: List[SoundEntry]
    var switch_param: Int
    var avoid_repeat: Int
    var history: List[Int]
    var seq_cursor: Int
    var volume: Float32
    var volume_random: Float32  # +/- fraction
    var pitch: Float32
    var pitch_random: Float32   # +/- fraction
    var bus_id: Int
    var looping: Bool
    var spatial: Bool
    var stream: Bool
    var priority: Int
    var ref_dist: Float32
    var max_dist: Float32
    var attenuation_id: Int
    var rtpcs: List[RtpcBinding]

    def __init__(out self, name: String, container: Int = CONTAINER_SINGLE):
        self.name = name
        self.container = container
        self.children = List[SoundEntry]()
        self.switch_param = -1
        self.avoid_repeat = 1
        self.history = List[Int]()
        self.seq_cursor = 0
        self.volume = 1.0
        self.volume_random = 0.0
        self.pitch = 1.0
        self.pitch_random = 0.0
        self.bus_id = BUS_SFX
        self.looping = False
        self.spatial = False
        self.stream = False
        self.priority = 1
        self.ref_dist = 15.0
        self.max_dist = 350.0
        self.attenuation_id = -1
        self.rtpcs = List[RtpcBinding]()


struct MixerSnapshot(Copyable, Movable):
    var name: String
    var bus_ids: List[Int]
    var gains: List[Float32]

    def __init__(out self, name: String):
        self.name = name
        self.bus_ids = List[Int]()
        self.gains = List[Float32]()

    def set_gain(mut self, bus_id: Int, gain: Float32):
        for i in range(len(self.bus_ids)):
            if self.bus_ids[i] == bus_id:
                self.gains[i] = max(Float32(0.0), gain)
                return
        self.bus_ids.append(bus_id)
        self.gains.append(max(Float32(0.0), gain))


struct _ActiveInstance(Copyable, Movable):
    var handle: SoundHandle
    var event_index: Int
    var last_gain: Float32
    var last_pitch: Float32
    var last_lowpass: Float32
    var age: Float32
    var seen_playing: Bool

    def __init__(out self, handle: SoundHandle, event_index: Int):
        self.handle = handle
        self.event_index = event_index
        self.age = 0.0
        self.seen_playing = False
        self.last_gain = 1.0
        self.last_pitch = 1.0
        self.last_lowpass = -1.0


@fieldwise_init
struct RtpcValues(Copyable, ImplicitlyCopyable, Movable):
    var gain: Float32
    var pitch: Float32
    var lowpass_hz: Float32  # -1 = no filter binding


struct SoundBank(Movable):
    """Main-thread sound-design runtime (events, game parameters, mixer snapshots)."""
    var params: List[GameParameter]
    var events: List[SoundEvent]
    var snapshots: List[MixerSnapshot]
    var bus_names: Dict[String, Int]
    var clip_ids: Dict[String, Int]
    var attenuation_names: Dict[String, Int]
    var instances: List[_ActiveInstance]
    var rng_state: UInt64
    # Snapshot blending state, indexed by bus id
    var bus_current: List[Float32]
    var bus_from: List[Float32]
    var bus_target: List[Float32]
    var bus_sent: List[Float32]
    var blend_elapsed: Float32
    var blend_duration: Float32
    var blending: Bool
    var active_snapshot: String
    var last_asset_id: Int  # clip chosen by the most recent `post_event` (debug/telemetry)

    def __init__(out self, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.params = List[GameParameter]()
        self.events = List[SoundEvent]()
        self.snapshots = List[MixerSnapshot]()
        self.bus_names = Dict[String, Int]()
        self.clip_ids = Dict[String, Int]()
        self.attenuation_names = Dict[String, Int]()
        self.instances = List[_ActiveInstance]()
        self.rng_state = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)
        self.bus_current = List[Float32]()
        self.bus_from = List[Float32]()
        self.bus_target = List[Float32]()
        self.bus_sent = List[Float32]()
        self.blend_elapsed = 0.0
        self.blend_duration = 0.0
        self.blending = False
        self.active_snapshot = ""
        self.last_asset_id = -1
        self.register_bus("Master", BUS_MASTER)
        self.register_bus("SFX", BUS_SFX)
        self.register_bus("Music", BUS_MUSIC)
        self.register_bus("Voice", BUS_VOICE)
        self.register_bus("Ambience", BUS_AMBIENCE)
        self.register_bus("UI", BUS_UI)

    # ---- registry ----

    def register_bus(mut self, name: String, bus_id: Int):
        if bus_id < 0:
            return
        self.bus_names[name] = bus_id
        while len(self.bus_current) <= bus_id:
            self.bus_current.append(1.0)
            self.bus_from.append(1.0)
            self.bus_target.append(1.0)
            self.bus_sent.append(1.0)

    def bus_id(self, name: String, fallback: Int = BUS_SFX) -> Int:
        var v = self.bus_names.get(name)
        return v.value() if v else fallback

    def bind_clip(mut self, name: String, asset_id: Int):
        """Names an already-registered asset so events can reference it as `"clip": name`."""
        self.clip_ids[name] = asset_id

    def add_parameter(mut self, var param: GameParameter) -> Int:
        var existing = self.find_parameter(param.name)
        if existing >= 0:
            self.params[existing] = param^
            return existing
        self.params.append(param^)
        return len(self.params) - 1

    def find_parameter(self, name: String) -> Int:
        for i in range(len(self.params)):
            if self.params[i].name == name:
                return i
        return -1

    def set_parameter(mut self, name: String, value: Float32, immediate: Bool = False) -> Bool:
        var i = self.find_parameter(name)
        if i < 0 or not isfinite(value):
            return False
        var v = max(self.params[i].min_value, min(self.params[i].max_value, value))
        self.params[i].target = v
        if immediate or self.params[i].slew_per_sec <= Float32(0.0):
            self.params[i].value = v
        return True

    def get_parameter(self, name: String, fallback: Float32 = 0.0) -> Float32:
        var i = self.find_parameter(name)
        return self.params[i].value if i >= 0 else fallback

    def add_event(mut self, var ev: SoundEvent) -> Int:
        var existing = self.find_event(ev.name)
        if existing >= 0:
            self.events[existing] = ev^
            return existing
        self.events.append(ev^)
        return len(self.events) - 1

    def find_event(self, name: String) -> Int:
        for i in range(len(self.events)):
            if self.events[i].name == name:
                return i
        return -1

    def add_snapshot(mut self, var snap: MixerSnapshot) -> Int:
        for i in range(len(snap.bus_ids)):
            if snap.bus_ids[i] >= len(self.bus_current):
                self.register_bus("bus" + String(snap.bus_ids[i]), snap.bus_ids[i])
        for i in range(len(self.snapshots)):
            if self.snapshots[i].name == snap.name:
                self.snapshots[i] = snap^
                return i
        self.snapshots.append(snap^)
        return len(self.snapshots) - 1

    # ---- deterministic RNG ----

    def _next_u64(mut self) -> UInt64:
        var x = self.rng_state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.rng_state = x
        return x * UInt64(0x2545F4914F6CDD1D)

    def _rand01(mut self) -> Float32:
        return Float32(Float64(self._next_u64() >> 11) * (1.0 / 9007199254740992.0))

    # ---- containers ----

    def _choose_child(mut self, ei: Int) -> Int:
        var n = len(self.events[ei].children)
        if n == 0:
            return -1
        var kind = self.events[ei].container
        if kind == CONTAINER_SEQUENCE:
            var c = self.events[ei].seq_cursor % n
            self.events[ei].seq_cursor = (c + 1) % n
            return c
        if kind == CONTAINER_SWITCH:
            var pi = self.events[ei].switch_param
            var v = self.params[pi].value if (pi >= 0 and pi < len(self.params)) else Float32(0.0)
            for i in range(n):
                if v >= self.events[ei].children[i].switch_min and v <= self.events[ei].children[i].switch_max:
                    return i
            return -1
        if kind == CONTAINER_RANDOM and n > 1:
            var avoid = max(0, min(self.events[ei].avoid_repeat, n - 1))
            var total_w = Float32(0.0)
            for i in range(n):
                if not self._recent(ei, i, avoid):
                    total_w += self.events[ei].children[i].weight
            var pick = n - 1
            if total_w > Float32(0.0):
                var r = self._rand01() * total_w
                var acc = Float32(0.0)
                for i in range(n):
                    if self._recent(ei, i, avoid):
                        continue
                    acc += self.events[ei].children[i].weight
                    pick = i
                    if r < acc:
                        break
            self.events[ei].history.append(pick)
            while len(self.events[ei].history) > 16:
                _ = self.events[ei].history.pop(0)
            return pick
        return 0

    def _recent(self, ei: Int, child: Int, avoid: Int) -> Bool:
        var h = len(self.events[ei].history)
        for k in range(min(avoid, h)):
            if self.events[ei].history[h - 1 - k] == child:
                return True
        return False

    def _resolve_leaf(mut self, ei: Int, depth: Int, mut vol: Float32, mut pitch: Float32) -> Int:
        """Walks nested containers down to a clip asset id (`-1` if none)."""
        if depth > _MAX_EVENT_DEPTH or ei < 0 or ei >= len(self.events):
            return -1
        var vr = self.events[ei].volume_random
        var pr = self.events[ei].pitch_random
        var ev_vol = self.events[ei].volume
        var ev_pitch = self.events[ei].pitch
        var rv = self._rand01()
        var rp = self._rand01()
        vol *= ev_vol * (Float32(1.0) + vr * (rv * Float32(2.0) - Float32(1.0)))
        pitch *= ev_pitch * (Float32(1.0) + pr * (rp * Float32(2.0) - Float32(1.0)))
        var c = self._choose_child(ei)
        if c < 0:
            return -1
        var nested = self.events[ei].children[c].event_name
        if nested.byte_length() > 0:
            return self._resolve_leaf(self.find_event(nested), depth + 1, vol, pitch)
        return self.events[ei].children[c].asset_id

    # ---- RTPC ----

    def evaluate_rtpc(self, ei: Int) -> RtpcValues:
        var out = RtpcValues(1.0, 1.0, -1.0)
        if ei < 0 or ei >= len(self.events):
            return out
        for i in range(len(self.events[ei].rtpcs)):
            ref b = self.events[ei].rtpcs[i]
            if b.param < 0 or b.param >= len(self.params):
                continue
            var y = b.curve.evaluate(self.params[b.param].value)
            if not isfinite(y):
                continue
            if b.target == RTPC_VOLUME:
                out.gain *= max(Float32(0.0), y)
            elif b.target == RTPC_VOLUME_DB:
                out.gain *= exp(y * Float32(0.11512925465))
            elif b.target == RTPC_PITCH:
                out.pitch *= max(Float32(0.05), y)
            elif b.target == RTPC_LOWPASS:
                out.lowpass_hz = y if out.lowpass_hz < Float32(0.0) else min(out.lowpass_hz, y)
        return out

    # ---- playback ----

    def post_event(mut self, mut server: AudioServer, name: String, pos: Vec3 = Vec3(), vel: Vec3 = Vec3()) -> SoundHandle:
        """Plays an event (spatial events at `pos`). Returns an invalid handle if nothing plays."""
        var ei = self.find_event(name)
        if ei < 0:
            return SoundHandle()
        var vol = Float32(1.0)
        var pitch = Float32(1.0)
        var aid = self._resolve_leaf(ei, 0, vol, pitch)
        self.last_asset_id = aid
        if aid < 0:
            return SoundHandle()
        var bus = self.events[ei].bus_id
        var looping = self.events[ei].looping
        var h: SoundHandle
        if self.events[ei].spatial:
            h = server.play_3d(
                aid, pos, vel, vol, pitch, bus, looping,
                self.events[ei].priority, self.events[ei].ref_dist, self.events[ei].max_dist,
                self.events[ei].attenuation_id,
            )
        else:
            h = server.play_2d(aid, vol, pitch, 0.0, bus, looping)
        if not h.is_valid():
            return h
        var inst = _ActiveInstance(h, ei)
        if len(self.events[ei].rtpcs) > 0:
            var rv = self.evaluate_rtpc(ei)
            _ = server.set_voice_params(h, rv.gain, rv.pitch, rv.lowpass_hz)
            inst.last_gain = rv.gain
            inst.last_pitch = rv.pitch
            inst.last_lowpass = rv.lowpass_hz
        self.instances.append(inst^)
        return h

    def stop_event(mut self, mut server: AudioServer, name: String, fade_seconds: Float32 = 0.0) -> Int:
        """Stops every live instance of `name`; returns how many were stopped."""
        var ei = self.find_event(name)
        var n = 0
        for i in range(len(self.instances)):
            if self.instances[i].event_index == ei:
                if server.stop_sound(self.instances[i].handle, fade_seconds):
                    n += 1
        return n

    def live_instance_count(self) -> Int:
        return len(self.instances)

    # ---- mixer snapshots ----

    def activate_snapshot(mut self, name: String, blend_seconds: Float32 = 0.0) -> Bool:
        var si = -1
        for i in range(len(self.snapshots)):
            if self.snapshots[i].name == name:
                si = i
                break
        if si < 0:
            return False
        for b in range(len(self.bus_current)):
            self.bus_from[b] = self.bus_current[b]
            self.bus_target[b] = 1.0
        for k in range(len(self.snapshots[si].bus_ids)):
            var b = self.snapshots[si].bus_ids[k]
            if b >= 0 and b < len(self.bus_target):
                self.bus_target[b] = self.snapshots[si].gains[k]
        self._start_blend(blend_seconds)
        self.active_snapshot = name
        return True

    def deactivate_snapshot(mut self, blend_seconds: Float32 = 0.0):
        for b in range(len(self.bus_current)):
            self.bus_from[b] = self.bus_current[b]
            self.bus_target[b] = 1.0
        self._start_blend(blend_seconds)
        self.active_snapshot = ""

    def _start_blend(mut self, blend_seconds: Float32):
        self.blend_elapsed = 0.0
        self.blend_duration = max(Float32(0.0), blend_seconds) if isfinite(blend_seconds) else Float32(0.0)
        self.blending = True

    def bus_gain(self, bus_id: Int) -> Float32:
        if bus_id < 0 or bus_id >= len(self.bus_current):
            return 1.0
        return self.bus_current[bus_id]

    # ---- per-frame update ----

    def update(mut self, mut server: AudioServer, dt: Float32):
        """Slews game parameters, re-applies RTPC curves to live instances, prunes finished
        instances, and advances mixer-snapshot blends (pushing changed bus gains)."""
        var step = max(Float32(0.0), dt) if isfinite(dt) else Float32(0.0)
        for i in range(len(self.params)):
            var tgt = self.params[i].target
            var cur = self.params[i].value
            if cur != tgt:
                var rate = self.params[i].slew_per_sec
                if rate <= Float32(0.0):
                    self.params[i].value = tgt
                elif cur < tgt:
                    self.params[i].value = min(tgt, cur + rate * step)
                else:
                    self.params[i].value = max(tgt, cur - rate * step)

        var keep = List[_ActiveInstance]()
        for i in range(len(self.instances)):
            var inst = self.instances[i].copy()
            inst.age += step
            if server.is_playing(inst.handle):
                inst.seen_playing = True
            elif inst.seen_playing or inst.age > Float32(0.5):
                continue  # finished (the grace period covers a PLAY not yet drained by the mixer)
            var ei = inst.event_index
            if ei >= 0 and ei < len(self.events) and len(self.events[ei].rtpcs) > 0:
                var rv = self.evaluate_rtpc(ei)
                if (
                    abs(rv.gain - inst.last_gain) > Float32(1.0e-4)
                    or abs(rv.pitch - inst.last_pitch) > Float32(1.0e-4)
                    or abs(rv.lowpass_hz - inst.last_lowpass) > Float32(0.5)
                ):
                    _ = server.set_voice_params(inst.handle, rv.gain, rv.pitch, rv.lowpass_hz)
                    inst.last_gain = rv.gain
                    inst.last_pitch = rv.pitch
                    inst.last_lowpass = rv.lowpass_hz
            keep.append(inst^)
        self.instances = keep^

        if self.blending:
            self.blend_elapsed += step
            var w = Float32(1.0)
            if self.blend_duration > Float32(1.0e-6):
                w = min(Float32(1.0), self.blend_elapsed / self.blend_duration)
            for b in range(len(self.bus_current)):
                var g = self.bus_from[b] + (self.bus_target[b] - self.bus_from[b]) * w
                self.bus_current[b] = g
                if abs(g - self.bus_sent[b]) > Float32(1.0e-5):
                    _ = server.set_bus_gain(b, g)
                    self.bus_sent[b] = g
            if w >= Float32(1.0):
                self.blending = False

    # ---- JSON ----

    def _resolve_clip(mut self, mut server: AudioServer, clip: String, stream: Bool) -> Int:
        var named = self.clip_ids.get(clip)
        if named:
            return named.value()
        var aid = server.load_audio_cached(clip, stream)
        if aid >= 0:
            self.clip_ids[clip] = aid
        return aid

    def load_json(mut self, text: String, mut server: AudioServer) raises -> Int:
        """Loads parameters, attenuation presets, buses, events and snapshots. Returns the
        number of events loaded; raises on malformed JSON or unknown references."""
        var doc = parse_json(List[UInt8](text.as_bytes()))
        if not doc.valid or doc.root < 0:
            raise Error("SoundBank.load_json: malformed JSON")
        var root = doc.root

        var buses = doc.object_get(root, "buses")
        if buses >= 0 and doc.nodes[buses].kind == 5:
            for k in range(len(doc.nodes[buses].keys)):
                var bid = doc.integer(doc.nodes[buses].children[k], -1)
                if bid >= 0:
                    self.register_bus(doc.nodes[buses].keys[k], bid)

        var params = doc.object_get(root, "parameters")
        for i in range(doc.array_size(params)):
            var pn = doc.at(params, i)
            var name = doc.text(doc.object_get(pn, "name"), "")
            if name.byte_length() == 0:
                raise Error("SoundBank.load_json: parameter without a name")
            _ = self.add_parameter(GameParameter(
                name,
                Float32(doc.number(doc.object_get(pn, "default"), 0.0)),
                Float32(doc.number(doc.object_get(pn, "min"), 0.0)),
                Float32(doc.number(doc.object_get(pn, "max"), 1.0)),
                Float32(doc.number(doc.object_get(pn, "slew"), 0.0)),
            ))

        var attens = doc.object_get(root, "attenuations")
        for i in range(doc.array_size(attens)):
            var an = doc.at(attens, i)
            var name = doc.text(doc.object_get(an, "name"), "")
            var curve = attenuation_curve_from_name(doc.text(doc.object_get(an, "curve"), "inverse"))
            if name.byte_length() == 0 or curve < 0:
                raise Error("SoundBank.load_json: bad attenuation preset '" + name + "'")
            var preset = AttenuationPreset(
                curve,
                Float32(doc.number(doc.object_get(an, "min"), 1.0)),
                Float32(doc.number(doc.object_get(an, "max"), 100.0)),
                Float32(doc.number(doc.object_get(an, "rolloff"), 1.0)),
            )
            var pts = doc.object_get(an, "points")
            for k in range(doc.array_size(pts)):
                var pt = doc.at(pts, k)
                _ = preset.add_point(
                    Float32(doc.number(doc.at(pt, 0), 0.0)), Float32(doc.number(doc.at(pt, 1), 1.0))
                )
            var cone = doc.object_get(an, "cone")
            if cone >= 0:
                preset.set_cone(
                    Float32(doc.number(doc.object_get(cone, "inner"), 360.0)),
                    Float32(doc.number(doc.object_get(cone, "outer"), 360.0)),
                    Float32(doc.number(doc.object_get(cone, "outer_gain"), 1.0)),
                )
            self.attenuation_names[name] = server.register_attenuation(preset)

        var events = doc.object_get(root, "events")
        var loaded = 0
        for i in range(doc.array_size(events)):
            var en = doc.at(events, i)
            var name = doc.text(doc.object_get(en, "name"), "")
            if name.byte_length() == 0:
                raise Error("SoundBank.load_json: event without a name")
            var ev = SoundEvent(name, container_from_name(doc.text(doc.object_get(en, "type"), "single")))
            ev.volume = Float32(doc.number(doc.object_get(en, "volume"), 1.0))
            ev.volume_random = Float32(doc.number(doc.object_get(en, "volume_random"), 0.0))
            ev.pitch = Float32(doc.number(doc.object_get(en, "pitch"), 1.0))
            ev.pitch_random = Float32(doc.number(doc.object_get(en, "pitch_random"), 0.0))
            ev.bus_id = self.bus_id(doc.text(doc.object_get(en, "bus"), "SFX"), BUS_SFX)
            ev.looping = doc.boolean(doc.object_get(en, "looping"), False)
            ev.spatial = doc.boolean(doc.object_get(en, "spatial"), False)
            ev.stream = doc.boolean(doc.object_get(en, "stream"), False)
            ev.priority = doc.integer(doc.object_get(en, "priority"), 1)
            ev.ref_dist = Float32(doc.number(doc.object_get(en, "ref_dist"), 15.0))
            ev.max_dist = Float32(doc.number(doc.object_get(en, "max_dist"), 350.0))
            ev.avoid_repeat = doc.integer(doc.object_get(en, "avoid_repeat"), 1)
            var att_name = doc.text(doc.object_get(en, "attenuation"), "")
            if att_name.byte_length() > 0:
                var att = self.attenuation_names.get(att_name)
                if not att:
                    raise Error("SoundBank.load_json: unknown attenuation '" + att_name + "'")
                ev.attenuation_id = att.value()
            var sw = doc.text(doc.object_get(en, "switch"), "")
            if sw.byte_length() > 0:
                ev.switch_param = self.find_parameter(sw)
                if ev.switch_param < 0:
                    raise Error("SoundBank.load_json: unknown switch parameter '" + sw + "'")
            var kids = doc.object_get(en, "children")
            for k in range(doc.array_size(kids)):
                var cn = doc.at(kids, k)
                var entry = SoundEntry(
                    doc.text(doc.object_get(cn, "clip"), ""),
                    doc.integer(doc.object_get(cn, "asset_id"), -1),
                    doc.text(doc.object_get(cn, "event"), ""),
                    Float32(doc.number(doc.object_get(cn, "weight"), 1.0)),
                )
                entry.switch_min = Float32(doc.number(doc.object_get(cn, "min"), -3.0e38))
                entry.switch_max = Float32(doc.number(doc.object_get(cn, "max"), 3.0e38))
                if entry.asset_id < 0 and entry.clip.byte_length() > 0:
                    entry.asset_id = self._resolve_clip(server, entry.clip, ev.stream)
                    if entry.asset_id < 0:
                        raise Error("SoundBank.load_json: cannot load clip '" + entry.clip + "'")
                ev.children.append(entry^)
            var rtpcs = doc.object_get(en, "rtpc")
            for k in range(doc.array_size(rtpcs)):
                var rn = doc.at(rtpcs, k)
                var pname = doc.text(doc.object_get(rn, "param"), "")
                var pidx = self.find_parameter(pname)
                var tgt = rtpc_target_from_name(doc.text(doc.object_get(rn, "target"), ""))
                if pidx < 0 or tgt < 0:
                    raise Error("SoundBank.load_json: bad rtpc binding on '" + name + "'")
                var curve = ParamCurve()
                var cpts = doc.object_get(rn, "curve")
                for c in range(doc.array_size(cpts)):
                    var cp = doc.at(cpts, c)
                    curve.add(Float32(doc.number(doc.at(cp, 0), 0.0)), Float32(doc.number(doc.at(cp, 1), 1.0)))
                ev.rtpcs.append(RtpcBinding(pidx, tgt, curve^))
            _ = self.add_event(ev^)
            loaded += 1

        # Nested event references must exist (checked after all events are known).
        for e in range(len(self.events)):
            for c in range(len(self.events[e].children)):
                var nested = self.events[e].children[c].event_name
                if nested.byte_length() > 0 and self.find_event(nested) < 0:
                    raise Error("SoundBank.load_json: unknown nested event '" + nested + "'")

        var snaps = doc.object_get(root, "snapshots")
        for i in range(doc.array_size(snaps)):
            var sn = doc.at(snaps, i)
            var snap = MixerSnapshot(doc.text(doc.object_get(sn, "name"), ""))
            var bmap = doc.object_get(sn, "buses")
            if bmap >= 0 and doc.nodes[bmap].kind == 5:
                for k in range(len(doc.nodes[bmap].keys)):
                    var bname = doc.nodes[bmap].keys[k]
                    var bid = self.bus_id(bname, -1)
                    if bid < 0:
                        raise Error("SoundBank.load_json: unknown bus '" + bname + "'")
                    snap.set_gain(bid, Float32(doc.number(doc.nodes[bmap].children[k], 1.0)))
            _ = self.add_snapshot(snap^)
        return loaded
