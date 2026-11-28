"""Perception System: Sight Cones with Physics LOS, Hearing from Noise Events & Memory
(`flint.ai.perception`, v5).

- `PerceptionSystem.register_stimulus(entity, pos, team, body_id)`: things that can be seen.
- `PerceptionSystem.report_noise(pos, radius, source, team)`: one-shot (or `ttl`) noise events.
- `Perceiver`: eye pose, field of view, sight range, hearing scale, awareness gain/decay and
  `memory_duration`; owns `PerceptionMemory` records (last-known position, last-sensed time,
  awareness, which senses currently report it).
- `PerceptionSystem.update(world, dt)`: sight = range + cone + `PhysicsWorld.raycast` line of sight
  (the stimulus' own body and the perceiver's body never occlude); hearing = noise within
  `radius * hearing_scale`; memories not refreshed decay and are forgotten after `memory_duration`.
  Emits `PerceptionEvent`s (`PERCEPT_EVT_SIGHTED`, `PERCEPT_EVT_LOST_SIGHT`, `PERCEPT_EVT_HEARD`,
  `PERCEPT_EVT_FORGOT`).
"""
from std.collections import List
from std.math import sqrt, min, max, isfinite
from flint.core.math import Vec3
from flint.core.detmath import det_cos
from flint.physics.filter import CollisionFilter, CAT_ALL
from flint.physics.world import PhysicsWorld

comptime SENSE_SIGHT: UInt8 = 1
comptime SENSE_HEARING: UInt8 = 2

comptime PERCEPT_EVT_SIGHTED: Int = 0
comptime PERCEPT_EVT_LOST_SIGHT: Int = 1
comptime PERCEPT_EVT_HEARD: Int = 2
comptime PERCEPT_EVT_FORGOT: Int = 3

comptime _P_DEG_TO_RAD: Float32 = 0.01745329252


@fieldwise_init
struct PerceptionStimulus(Copyable, ImplicitlyCopyable, Movable):
    var entity: UInt64
    var pos: Vec3
    var team: Int
    var body_id: Int  # physics body of the stimulus (never occludes itself), -1 if none
    var active: Bool


@fieldwise_init
struct NoiseEvent(Copyable, ImplicitlyCopyable, Movable):
    var pos: Vec3
    var radius: Float32  # audible distance at hearing_scale 1
    var source: UInt64
    var team: Int
    var ttl: Float32  # seconds remaining (0 = this update only)


@fieldwise_init
struct PerceptionMemory(Copyable, ImplicitlyCopyable, Movable):
    var entity: UInt64
    var team: Int
    var last_known_pos: Vec3
    var first_sensed_time: Float32
    var last_sensed_time: Float32
    var awareness: Float32  # [0, 1]
    var visible: Bool
    var heard: Bool
    var senses: UInt8  # SENSE_* mask of the last refresh


@fieldwise_init
struct PerceptionEvent(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var perceiver: Int
    var entity: UInt64
    var pos: Vec3


struct Perceiver(Copyable, Movable):
    var entity: UInt64
    var team: Int
    var body_id: Int
    var eye_pos: Vec3
    var forward: Vec3
    var fov_deg: Float32
    var sight_range: Float32
    var near_sense_radius: Float32  # 360-degree proximity sense (still needs LOS)
    var hearing_scale: Float32
    var memory_duration: Float32
    var sight_gain: Float32  # awareness per second at point-blank range
    var awareness_decay: Float32  # awareness lost per second when unsensed
    var ignore_same_team: Bool
    var enabled: Bool
    var memories: List[PerceptionMemory]

    def __init__(
        out self,
        entity: UInt64 = 0,
        team: Int = 0,
        fov_deg: Float32 = 110.0,
        sight_range: Float32 = 25.0,
        memory_duration: Float32 = 5.0,
    ):
        self.entity = entity
        self.team = team
        self.body_id = -1
        self.eye_pos = Vec3()
        self.forward = Vec3(0.0, 0.0, 1.0)
        self.fov_deg = fov_deg
        self.sight_range = sight_range
        self.near_sense_radius = 1.0
        self.hearing_scale = 1.0
        self.memory_duration = memory_duration
        self.sight_gain = 4.0
        self.awareness_decay = 0.25
        self.ignore_same_team = True
        self.enabled = True
        self.memories = List[PerceptionMemory]()

    def find_memory(self, entity: UInt64) -> Int:
        for i in range(len(self.memories)):
            if self.memories[i].entity == entity:
                return i
        return -1

    def knows(self, entity: UInt64) -> Bool:
        return self.find_memory(entity) >= 0

    def can_see(self, entity: UInt64) -> Bool:
        var i = self.find_memory(entity)
        return i >= 0 and self.memories[i].visible

    def best_target(self) -> Int:
        """Memory index with the highest awareness (visible wins ties), or -1."""
        var best = -1
        var best_s = Float32(-1.0)
        for i in range(len(self.memories)):
            var s = self.memories[i].awareness + (Float32(0.5) if self.memories[i].visible else Float32(0.0))
            if s > best_s:
                best_s = s
                best = i
        return best


struct PerceptionSystem(Copyable, Movable):
    var stimuli: List[PerceptionStimulus]
    var noises: List[NoiseEvent]
    var perceivers: List[Perceiver]
    var events: List[PerceptionEvent]
    var time: Float32
    var occlude_filter: CollisionFilter
    var los_tolerance: Float32

    def __init__(out self):
        self.stimuli = List[PerceptionStimulus]()
        self.noises = List[NoiseEvent]()
        self.perceivers = List[Perceiver]()
        self.events = List[PerceptionEvent]()
        self.time = 0.0
        self.occlude_filter = CollisionFilter(CAT_ALL, CAT_ALL, 0)
        self.los_tolerance = 0.05

    # ---- stimuli ----
    def register_stimulus(mut self, entity: UInt64, pos: Vec3, team: Int = 0, body_id: Int = -1) -> Int:
        for i in range(len(self.stimuli)):
            if self.stimuli[i].entity == entity:
                self.stimuli[i] = PerceptionStimulus(entity, pos, team, body_id, True)
                return i
        self.stimuli.append(PerceptionStimulus(entity, pos, team, body_id, True))
        return len(self.stimuli) - 1

    def update_stimulus(mut self, idx: Int, pos: Vec3):
        if idx >= 0 and idx < len(self.stimuli):
            self.stimuli[idx].pos = pos

    def set_stimulus_active(mut self, idx: Int, active: Bool):
        if idx >= 0 and idx < len(self.stimuli):
            self.stimuli[idx].active = active

    def unregister_stimulus(mut self, entity: UInt64):
        var i = 0
        while i < len(self.stimuli):
            if self.stimuli[i].entity == entity:
                _ = self.stimuli.pop(i)
            else:
                i += 1

    def report_noise(mut self, pos: Vec3, radius: Float32, source: UInt64 = 0, team: Int = -1, ttl: Float32 = 0.0):
        if not isfinite(radius) or radius <= Float32(0.0):
            return
        self.noises.append(NoiseEvent(pos, radius, source, team, max(Float32(0.0), ttl)))

    # ---- perceivers ----
    def add_perceiver(mut self, var p: Perceiver) -> Int:
        self.perceivers.append(p^)
        return len(self.perceivers) - 1

    def set_perceiver_pose(mut self, idx: Int, eye_pos: Vec3, forward: Vec3):
        if idx < 0 or idx >= len(self.perceivers):
            return
        self.perceivers[idx].eye_pos = eye_pos
        var f = forward.normalized()
        self.perceivers[idx].forward = f if f.length_squared() > Float32(0.5) else Vec3(0.0, 0.0, 1.0)

    def take_events(mut self) -> List[PerceptionEvent]:
        var out = self.events^
        self.events = List[PerceptionEvent]()
        return out^

    def _has_los(self, world: PhysicsWorld, eye: Vec3, target: Vec3, ignore_body: Int, target_body: Int) -> Bool:
        var d = target - eye
        var dist = d.length()
        if dist <= Float32(1.0e-4):
            return True
        var dir = d * (Float32(1.0) / dist)
        var hit = world.raycast(eye, dir, dist, self.occlude_filter, ignore_body)
        if not hit.hit:
            return True
        if target_body >= 0 and hit.body_id == target_body:
            return True
        return hit.distance >= dist - self.los_tolerance

    def _touch(mut self, pi: Int, entity: UInt64, team: Int, pos: Vec3, sense: UInt8, gain: Float32):
        var mi = self.perceivers[pi].find_memory(entity)
        if mi < 0:
            self.perceivers[pi].memories.append(PerceptionMemory(entity, team, pos, self.time, self.time, Float32(0.0), False, False, 0))
            mi = len(self.perceivers[pi].memories) - 1
        ref m = self.perceivers[pi].memories[mi]
        m.last_known_pos = pos
        m.last_sensed_time = self.time
        m.awareness = min(Float32(1.0), m.awareness + gain)
        m.senses = m.senses | sense
        if sense == SENSE_SIGHT:
            m.visible = True
        else:
            m.heard = True

    def update(mut self, world: PhysicsWorld, dt: Float32):
        var safe_dt = dt if (isfinite(dt) and dt > Float32(0.0)) else Float32(0.0)
        self.time += safe_dt
        for pi in range(len(self.perceivers)):
            if not self.perceivers[pi].enabled:
                continue
            # Snapshot previous visibility for edge events, then clear per-frame sense flags
            var was_visible = List[Bool](capacity=len(self.perceivers[pi].memories))
            for mi in range(len(self.perceivers[pi].memories)):
                was_visible.append(self.perceivers[pi].memories[mi].visible)
                self.perceivers[pi].memories[mi].visible = False
                self.perceivers[pi].memories[mi].heard = False
                self.perceivers[pi].memories[mi].senses = 0
            var n_prev = len(was_visible)

            var eye = self.perceivers[pi].eye_pos
            var fwd = self.perceivers[pi].forward
            var cos_half = det_cos(self.perceivers[pi].fov_deg * Float32(0.5) * _P_DEG_TO_RAD)
            var rng = self.perceivers[pi].sight_range
            var near_r = self.perceivers[pi].near_sense_radius
            var my_ent = self.perceivers[pi].entity
            var my_team = self.perceivers[pi].team
            var my_body = self.perceivers[pi].body_id
            var same_team_ignored = self.perceivers[pi].ignore_same_team
            var gain_rate = self.perceivers[pi].sight_gain

            # 1. Sight
            for si in range(len(self.stimuli)):
                var st = self.stimuli[si]
                if not st.active or st.entity == my_ent:
                    continue
                if same_team_ignored and st.team == my_team:
                    continue
                var d = st.pos - eye
                var dist = d.length()
                if dist > rng:
                    continue
                var in_cone = dist <= near_r
                if not in_cone and dist > Float32(1.0e-5):
                    in_cone = d.dot(fwd) / dist >= cos_half
                if not in_cone:
                    continue
                if not self._has_los(world, eye, st.pos, my_body, st.body_id):
                    continue
                var closeness = Float32(1.0) - dist / max(Float32(0.01), rng)
                self._touch(pi, st.entity, st.team, st.pos, SENSE_SIGHT, gain_rate * safe_dt * (Float32(0.25) + Float32(0.75) * closeness) + Float32(1.0e-3))

            # 2. Hearing
            var hs = self.perceivers[pi].hearing_scale
            for ni in range(len(self.noises)):
                var nz = self.noises[ni]
                if nz.source == my_ent and nz.source != 0:
                    continue
                if same_team_ignored and nz.team == my_team:
                    continue
                var r = nz.radius * hs
                var dist = (nz.pos - eye).length()
                if dist > r:
                    continue
                self._touch(pi, nz.source, nz.team, nz.pos, SENSE_HEARING, Float32(0.5) * (Float32(1.0) - dist / max(Float32(0.01), r)) + Float32(0.05))
                self.events.append(PerceptionEvent(PERCEPT_EVT_HEARD, pi, nz.source, nz.pos))

            # 3. Edge events, decay & forgetting
            var mem_dur = self.perceivers[pi].memory_duration
            var decay = self.perceivers[pi].awareness_decay
            var mi = 0
            var orig_i = 0
            while mi < len(self.perceivers[pi].memories):
                var m = self.perceivers[pi].memories[mi]
                var prev_vis = was_visible[orig_i] if orig_i < n_prev else False
                if m.visible and not prev_vis:
                    self.events.append(PerceptionEvent(PERCEPT_EVT_SIGHTED, pi, m.entity, m.last_known_pos))
                elif prev_vis and not m.visible:
                    self.events.append(PerceptionEvent(PERCEPT_EVT_LOST_SIGHT, pi, m.entity, m.last_known_pos))
                orig_i += 1
                if m.senses == 0:
                    self.perceivers[pi].memories[mi].awareness = max(Float32(0.0), m.awareness - decay * safe_dt)
                    if self.time - m.last_sensed_time > mem_dur:
                        self.events.append(PerceptionEvent(PERCEPT_EVT_FORGOT, pi, m.entity, m.last_known_pos))
                        _ = self.perceivers[pi].memories.pop(mi)
                        continue
                mi += 1

        # 4. Expire noise events
        var k = 0
        while k < len(self.noises):
            if self.noises[k].ttl <= safe_dt:
                _ = self.noises.pop(k)
            else:
                self.noises[k].ttl -= safe_dt
                k += 1
