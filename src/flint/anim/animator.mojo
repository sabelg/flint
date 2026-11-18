"""The animation runtime (package 3.3 / v4): runs an `AnimGraphDef` on a model.

Upgrades in Flint v4:
- Wires 2D Johansen Gradient-Band `blend_spaces_2d` (`st.blend_2d >= 0`) alongside
  1D `blend_spaces` and single-clip states.
- Caches additive layer reference poses at `bind` time (`self.layer_refs`) instead
  of re-sampling `layer.reference_time` every frame.
- Blends `SyncTrack` segment lengths across active sample weights so phase speed
  varies smoothly across locomotion crossfades.
- Exact loop-wrap `extract_root_motion` (`(end - prev) + (cur - start)` when
  `time < prev_time` or `looped > 0`, with zero `0.5m` cutoff or dropped frames)
  plus `root_yaw_delta` extraction.
"""
from std.math import floor
from flint.core.detmath import det_atan2, det_sin, det_cos, PI
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.model import NativeModel
from flint.anim.pose import (
    Pose, SkeletonInstance, Inertializer, sample_pose, blend_into, blend_masked,
    add_additive, full_mask, bone_mask, write_pose,
)
from flint.anim.sync import ClipInfo, SyncTrack, FiredEvent, passed_events, wrap_time
from flint.anim.graph import (
    AnimGraphDef, all_hold, ROOT_STRIP, ROOT_EXTRACT,
)


@always_inline
def _wrap_rad(a: Float32) -> Float32:
    comptime TAU_F32: Float32 = 6.28318530717958647692
    return a - TAU_F32 * floor((a + PI) / TAU_F32)


@always_inline
def _quat_yaw(q: Quat) -> Float32:
    """Extracts horizontal heading yaw (radians about +Y, +Z forward) from `q`."""
    var fwd = q.rotate(Vec3(0.0, 0.0, 1.0))
    if fwd.x * fwd.x + fwd.z * fwd.z < Float32(1.0e-8):
        return Float32(0.0)
    return det_atan2(fwd.x, fwd.z)


def extract_root_motion(
    model: NativeModel,
    clip: Int,
    prev_time: Float32,
    cur_time: Float32,
    looped: Int,
    root_node: Int,
    mut scratch_a: Pose,
    mut scratch_b: Pose,
) -> Tuple[Vec3, Float32]:
    """Exact clip root motion `(delta_pos_xyz, delta_yaw_rad)` across `[prev_time, cur_time]`.

    Handles forward and backward loop wraps cleanly via `(end - prev) + (cur - start)`
    without dropping wrap frames or clamping large displacements at `0.5m`.
    """
    if clip < 0 or clip >= len(model.animations) or root_node < 0 or root_node >= len(model.nodes):
        return (Vec3(), Float32(0.0))
    var dur = model.animations[clip].duration
    if dur <= Float32(0.0):
        return (Vec3(), Float32(0.0))

    sample_pose(model, clip, prev_time, scratch_a)
    sample_pose(model, clip, cur_time, scratch_b)
    var p_prev = scratch_a.t[root_node]
    var p_cur = scratch_b.t[root_node]
    var y_prev = _quat_yaw(scratch_a.r[root_node])
    var y_cur = _quat_yaw(scratch_b.r[root_node])

    var wrapped_loops = looped
    if wrapped_loops == 0 and cur_time < prev_time - Float32(1.0e-5):
        wrapped_loops = 1

    if wrapped_loops <= 0:
        return (p_cur - p_prev, _wrap_rad(y_cur - y_prev))

    sample_pose(model, clip, Float32(0.0), scratch_a)
    sample_pose(model, clip, dur, scratch_b)
    var p_start = scratch_a.t[root_node]
    var p_end = scratch_b.t[root_node]
    var y_start = _quat_yaw(scratch_a.r[root_node])
    var y_end = _quat_yaw(scratch_b.r[root_node])

    var cycle_dp = p_end - p_start
    var cycle_dy = _wrap_rad(y_end - y_start)
    var extra_cycles = Float32(max(0, wrapped_loops - 1))
    var dp = (p_end - p_prev) + cycle_dp * extra_cycles + (p_cur - p_start)
    var dy = _wrap_rad(y_end - y_prev) + cycle_dy * extra_cycles + _wrap_rad(y_cur - y_start)
    return (dp, dy)


def extract_root_motion(
    model: NativeModel,
    clip: Int,
    prev_time: Float32,
    cur_time: Float32,
    root_node: Int,
) -> Tuple[Vec3, Float32]:
    """Convenience overload automatically detecting loop wrap when `cur_time < prev_time`."""
    var sa = Pose()
    var sb = Pose()
    var looped = 1 if cur_time < prev_time - Float32(1.0e-5) else 0
    return extract_root_motion(model, clip, prev_time, cur_time, looped, root_node, sa, sb)


struct LayerState(Copyable, ImplicitlyCopyable, Movable):
    var weight: Float32
    var time: Float32
    var on: Bool

    def __init__(out self):
        self.weight = 0
        self.time = 0
        self.on = False


struct Animator(Copyable, Movable):
    var graph: AnimGraphDef
    var clips: List[ClipInfo]
    var tracks: List[SyncTrack]
    var params: List[Float32]
    # Bound to one model (one level of detail of a rig).
    var handle: Int
    var signature: UInt64
    var masks: List[List[Float32]]
    var layer_refs: List[Pose]
    var all_nodes: List[Float32]
    var root_node: Int
    # The state machine.
    var state: Int
    var previous: Int
    var state_time: Float32
    var clip_time: Float32
    var prev_clip_time: Float32
    var last_looped: Int
    var phase: Float32
    var prev_phase: Float32
    var switched: Bool
    var switches: Int
    var halflife: Float32
    var layers: List[LayerState]
    # This frame's results.
    var pose: Pose
    var ahead: Pose
    var scratch: Pose
    var reference: Pose
    var root_scratch_a: Pose
    var root_scratch_b: Pose
    var inertial: Inertializer
    var events: List[FiredEvent]
    var weights: List[Float32]
    var leader: Int
    var leader_time: Float32
    var rate: Float32
    var stride_scale: Float32
    var root_delta: Vec3
    var root_yaw_delta: Float32
    var extract_root_yaw: Bool
    var root_travel: Vec3
    var last_root: Vec3
    var last_root_yaw: Float32
    # Inputs besides the params: ground speed for ground-rate blend spaces,
    # and distance left to a stop (0: not stopping).
    var ground_speed: Float32
    var stop_distance: Float32
    var stop_active: Bool
    var stop_from: Float32
    var stop_target: Float32
    var stop_total: Float32

    def __init__(out self):
        self.graph = AnimGraphDef()
        self.clips = List[ClipInfo]()
        self.tracks = List[SyncTrack]()
        self.params = List[Float32]()
        self.handle = -1
        self.signature = 0
        self.masks = List[List[Float32]]()
        self.layer_refs = List[Pose]()
        self.all_nodes = List[Float32]()
        self.root_node = -1
        self.state = -1
        self.previous = -1
        self.state_time = 0
        self.clip_time = 0
        self.prev_clip_time = 0
        self.last_looped = 0
        self.phase = 0
        self.prev_phase = 0
        self.switched = False
        self.switches = 0
        self.halflife = 0.08
        self.layers = List[LayerState]()
        self.pose = Pose()
        self.ahead = Pose()
        self.scratch = Pose()
        self.reference = Pose()
        self.root_scratch_a = Pose()
        self.root_scratch_b = Pose()
        self.inertial = Inertializer()
        self.events = List[FiredEvent]()
        self.weights = List[Float32]()
        self.leader = -1
        self.leader_time = 0
        self.rate = 1
        self.stride_scale = 1
        self.root_delta = Vec3()
        self.root_yaw_delta = 0
        self.extract_root_yaw = False
        self.root_travel = Vec3()
        self.last_root = Vec3()
        self.last_root_yaw = 0
        self.ground_speed = 0
        self.stop_distance = 0
        self.stop_active = False
        self.stop_from = 0
        self.stop_target = 0
        self.stop_total = 0

    def configure(mut self, graph: AnimGraphDef, clips: List[ClipInfo]):
        """A new graph or clip table; the animation starts over."""
        self.graph = graph.copy()
        self.clips = clips.copy()
        self.tracks = List[SyncTrack]()
        for clip in self.clips:
            self.tracks.append(SyncTrack(clip))
        if len(self.params) != len(self.graph.params):
            self.params = List[Float32](length=len(self.graph.params), fill=0)
        self.layers = List[LayerState](length=len(self.graph.layers), fill=LayerState())
        self.layer_refs.clear()
        self.handle = -1
        self.reset()

    def configured(self) -> Bool:
        return self.graph.valid() and len(self.clips) > 0

    def reset(mut self):
        """Start over without blending (a teleport, a seek, a new form)."""
        self.state = -1
        self.previous = -1
        self.state_time = 0
        self.clip_time = 0
        self.prev_clip_time = 0
        self.last_looped = 0
        self.phase = 0
        self.prev_phase = 0
        self.inertial.reset()
        for index in range(len(self.layers)):
            self.layers[index] = LayerState()
        self.root_travel = Vec3()
        self.root_yaw_delta = 0

    def param(self, name: String) -> Int:
        return self.graph.param(name)

    def set(mut self, index: Int, value: Float32):
        if index >= 0 and index < len(self.params):
            self.params[index] = value

    def set_bool(mut self, index: Int, value: Bool):
        self.set(index, Float32(1) if value else Float32(0))

    def state_name(self) -> String:
        if self.state < 0 or self.state >= len(self.graph.states):
            return ""
        return self.graph.states[self.state].name

    def layer_weight(self, name: String) -> Float32:
        var index = self.graph.layer(name)
        return self.layers[index].weight if index >= 0 else Float32(0)

    def clip_of_state(self) -> Int:
        """The clip the state plays (a blend space's leader)."""
        if self.state < 0:
            return -1
        if self.graph.states[self.state].clip >= 0:
            return self.graph.states[self.state].clip
        return self.leader

    def bind(mut self, model: NativeModel, handle: Int):
        """Masks, cached additive layer reference poses, and root node for this model."""
        var hash = UInt64(14695981039346656037)
        for node in model.nodes:
            for byte in node.name.as_bytes():
                hash = (hash ^ UInt64(byte)) * 1099511628211
            hash = (hash ^ 0x2F) * 1099511628211
        if (
            handle == self.handle
            and hash == self.signature
            and len(self.all_nodes) == len(model.nodes)
            and len(self.layer_refs) == len(self.graph.layers)
        ):
            return
        if hash != self.signature:
            self.inertial.reset()
        self.handle = handle
        self.signature = hash
        self.masks = List[List[Float32]]()
        for mask in self.graph.masks:
            self.masks.append(bone_mask(model, mask.bones, mask.weights, mask.children))
        self.all_nodes = full_mask(len(model.nodes))
        self.root_node = model.find_node("Root")
        self.layer_refs = List[Pose]()
        for index in range(len(self.graph.layers)):
            var ref_pose = Pose()
            ref layer = self.graph.layers[index]
            if layer.additive and layer.clip >= 0 and layer.clip < len(model.animations):
                sample_pose(model, layer.clip, layer.reference_time, ref_pose)
            self.layer_refs.append(ref_pose^)

    def hold_time(self, state: Int) -> Float32:
        ref st = self.graph.states[state]
        if st.clip < 0 or st.clip >= len(self.clips):
            return 0
        if st.hold_event.byte_length() == 0:
            return self.clips[st.clip].duration
        return self.clips[st.clip].event_time(st.hold_event, self.clips[st.clip].duration)

    def holding(self) -> Bool:
        if self.state < 0:
            return False
        ref st = self.graph.states[self.state]
        return st.holds and all_hold(st.hold_while, self.params) and self.clip_time < self.hold_time(self.state)

    def select(self) -> Int:
        var candidate = len(self.graph.states) - 1
        for index in range(len(self.graph.states)):
            if all_hold(self.graph.states[index].when, self.params):
                candidate = index
                break
        if self.state >= 0 and candidate > self.state and self.holding():
            return self.state
        return candidate

    def _is_blend_state(self, state_idx: Int) -> Bool:
        if state_idx < 0 or state_idx >= len(self.graph.states):
            return False
        ref st = self.graph.states[state_idx]
        return st.blend >= 0 or st.blend_2d >= 0

    def update(mut self, model: NativeModel, handle: Int, dt: Float32):
        """One frame: state, clocks, pose (in self.pose), events."""
        self.events.clear()
        self.switched = False
        self.root_delta = Vec3()
        self.root_yaw_delta = Float32(0.0)
        if not self.configured() or len(model.nodes) == 0:
            return
        self.bind(model, handle)
        var chosen = self.select()
        var restart_clock = False
        if chosen != self.state:
            self.previous = self.state
            var was_blend = self._is_blend_state(self.state)
            self.halflife = self.graph.blend_time(self.state, chosen)
            self.state = chosen
            self.switched = True
            self.switches += 1
            self.state_time = 0
            ref st = self.graph.states[chosen]
            if st.reset_time or st.clip >= 0:
                self.clip_time = 0
                self.prev_clip_time = 0
                restart_clock = True
            if (st.blend >= 0 or st.blend_2d >= 0) and not (was_blend and not st.reset_time):
                self.phase = 0
                self.prev_phase = 0
                restart_clock = True
        else:
            self.state_time += dt
        self.advance(dt, restart_clock)
        var out = Pose()
        swap(out, self.pose)
        self.evaluate_base(model, out, 0)
        swap(out, self.pose)
        self.root_motion(model, restart_clock)
        self.update_layers(dt)
        swap(out, self.pose)
        self.apply_layers(model, out)
        if self.switched:
            var ahead = Pose()
            swap(ahead, self.ahead)
            self.evaluate_base(model, ahead, dt)
            self.apply_layers(model, ahead)
            if self.root_node >= 0 and self.root_node < ahead.count():
                var mode = self.graph.states[self.state].root_motion
                if mode == ROOT_STRIP or mode == ROOT_EXTRACT:
                    var base = model.nodes[self.root_node].base_translation
                    ahead.t[self.root_node] = Vec3(base.x, ahead.t[self.root_node].y, base.z)
                    if self.extract_root_yaw:
                        var base_yaw = _quat_yaw(model.nodes[self.root_node].base_rotation)
                        var cur_yaw = _quat_yaw(ahead.r[self.root_node])
                        var strip_q = Quat(0.0, det_sin((base_yaw - cur_yaw) * Float32(0.5)), 0.0, det_cos((base_yaw - cur_yaw) * Float32(0.5)))
                        ahead.r[self.root_node] = strip_q.multiply(ahead.r[self.root_node]).normalized()
            swap(ahead, self.ahead)
        self.inertial.apply(out, self.ahead, self.switched, self.halflife, dt)
        swap(out, self.pose)

    def _blended_segment_length_1d(self, blend_idx: Int, leader_clip: Int, segment: Int) -> Float32:
        ref space = self.graph.blend_spaces[blend_idx]
        ref lead_track = self.tracks[leader_clip]
        if not space.sync:
            return lead_track.segment_length(segment)
        var weighted_len = Float32(0.0)
        var total_w = Float32(0.0)
        for index in range(len(space.samples)):
            var w = self.weights[index] if index < len(self.weights) else Float32(0.0)
            if w <= Float32(0.0001):
                continue
            var clip = space.samples[index].clip
            if clip == leader_clip or self.tracks[clip].matches(lead_track):
                weighted_len += w * self.tracks[clip].segment_length(segment)
                total_w += w
        if total_w > Float32(0.0001):
            return max(Float32(0.0001), weighted_len / total_w)
        return lead_track.segment_length(segment)

    def _blended_segment_length_2d(self, blend2d_idx: Int, leader_clip: Int, segment: Int) -> Float32:
        ref space = self.graph.blend_spaces_2d[blend2d_idx]
        ref lead_track = self.tracks[leader_clip]
        if not space.sync:
            return lead_track.segment_length(segment)
        var weighted_len = Float32(0.0)
        var total_w = Float32(0.0)
        for index in range(len(space.samples)):
            var w = self.weights[index] if index < len(self.weights) else Float32(0.0)
            if w <= Float32(0.0001):
                continue
            var clip = space.samples[index].clip
            if clip == leader_clip or self.tracks[clip].matches(lead_track):
                weighted_len += w * self.tracks[clip].segment_length(segment)
                total_w += w
        if total_w > Float32(0.0001):
            return max(Float32(0.0001), weighted_len / total_w)
        return lead_track.segment_length(segment)

    def _advance_blend_phase(
        mut self,
        dt: Float32,
        restarted: Bool,
        is_2d: Bool,
        space_idx: Int,
        leader_clip: Int,
        lead_weight: Float32,
        rate_in: Float32,
    ):
        ref track = self.tracks[leader_clip]
        var rate = rate_in
        var matched = False
        self.prev_phase = self.phase
        if self.stop_distance > 0 and not restarted:
            if not self.stop_active or self.stop_distance > self.stop_total:
                self.stop_active = True
                self.stop_from = self.phase
                self.stop_target = Float32(Int(self.phase) + 1)
                self.stop_total = self.stop_distance
            var covered = max(Float32(0), min(Float32(1), 1.0 - self.stop_distance / max(Float32(0.0001), self.stop_total)))
            var wanted_phase = self.stop_from + (self.stop_target - self.stop_from) * covered
            var step = max(Float32(0), wanted_phase - self.phase)
            var seg_len = self._blended_segment_length_2d(space_idx, leader_clip, Int(self.phase)) if is_2d else self._blended_segment_length_1d(space_idx, leader_clip, Int(self.phase))
            rate = step * seg_len / max(Float32(0.0001), dt)
            self.stride_scale = 1
            matched = True
        else:
            self.stop_active = False
        self.rate = rate

        var before = track.time_at(self.phase)
        self.prev_clip_time = before
        var remaining_time = Float32(0) if restarted else dt * rate
        var phase_loops = 0
        if matched:
            remaining_time = 0
            var target_phase = self.stop_from + (self.stop_target - self.stop_from) * max(Float32(0), min(Float32(1), 1.0 - self.stop_distance / max(Float32(0.0001), self.stop_total)))
            if target_phase > self.phase:
                self.phase = target_phase
            if self.phase >= Float32(track.markers()):
                self.phase -= Float32(track.markers())
                self.stop_from -= Float32(track.markers())
                self.stop_target -= Float32(track.markers())
                phase_loops += 1
        var markers = track.markers()
        var guard = 0
        while remaining_time > 0 and guard < 64:
            var segment = min(markers - 1, Int(self.phase))
            var length = self._blended_segment_length_2d(space_idx, leader_clip, segment) if is_2d else self._blended_segment_length_1d(space_idx, leader_clip, segment)
            var left = (Float32(segment + 1) - self.phase) * length
            if remaining_time < left:
                self.phase += remaining_time / length
                break
            remaining_time -= left
            self.phase = Float32(segment + 1)
            if self.phase >= Float32(markers):
                self.phase -= Float32(markers)
                phase_loops += 1
            guard += 1
        var after = track.time_at(self.phase)
        var looped = max(phase_loops, 1 if (before >= 0 and after < before) else 0)
        self.last_looped = looped
        passed_events(self.clips[leader_clip], leader_clip, before, after, looped, lead_weight, self.events)
        self.leader = leader_clip
        self.leader_time = after
        self.clip_time = after

    def advance(mut self, dt: Float32, restarted: Bool):
        ref st = self.graph.states[self.state]
        self.stride_scale = 1
        self.last_looped = 0
        self.rate = st.rate * (self.params[st.rate_param] if st.rate_param >= 0 else Float32(1))
        if st.clip >= 0:
            self.weights = List[Float32]()
            self.leader = st.clip
            ref info = self.clips[st.clip]
            self.prev_clip_time = Float32(0.0) if restarted else self.clip_time
            var before = Float32(-1) if restarted else self.clip_time
            var looped = 0
            if st.time_param >= 0:
                var target = max(Float32(0), min(Float32(1), self.params[st.time_param])) * info.duration
                if target > self.clip_time or restarted:
                    passed_events(info, st.clip, before, target, 0, 1, self.events)
                self.clip_time = target
            else:
                var t = self.clip_time + (0 if restarted else dt * self.rate)
                if info.loop and info.duration > 0:
                    while t < 0:
                        t += info.duration
                        looped -= 1
                    while t >= info.duration:
                        t -= info.duration
                        looped += 1
                else:
                    t = max(Float32(0), min(t, info.duration))
                passed_events(info, st.clip, before, t, max(0, looped), 1, self.events)
                self.clip_time = t
            self.last_looped = max(0, looped)
            self.leader_time = self.clip_time
            return

        if st.blend_2d >= 0:
            ref space2d = self.graph.blend_spaces_2d[st.blend_2d]
            var px = self.params[space2d.param_x] if space2d.param_x >= 0 and space2d.param_x < len(self.params) else Float32(0.0)
            var py = self.params[space2d.param_y] if space2d.param_y >= 0 and space2d.param_y < len(self.params) else Float32(0.0)
            self.weights = space2d.weights(px, py)
            var lead2d = 0
            for index in range(len(self.weights)):
                if self.weights[index] > self.weights[lead2d]:
                    lead2d = index
            var leader_clip2d = space2d.samples[lead2d].clip
            var natural2d = Float32(0.0)
            for index in range(len(self.weights)):
                natural2d += self.weights[index] * self.clips[space2d.samples[index].clip].speed
            var rate2d = self.rate
            if space2d.ground_rate and natural2d > 0:
                var wanted2d = self.ground_speed / natural2d
                rate2d = max(space2d.min_rate, min(space2d.max_rate, wanted2d))
                self.stride_scale = max(Float32(0.7), min(Float32(1.4), wanted2d / max(Float32(0.01), rate2d)))
            self._advance_blend_phase(dt, restarted, True, st.blend_2d, leader_clip2d, self.weights[lead2d], rate2d)
            return

        ref space = self.graph.blend_spaces[st.blend]
        self.weights = space.weights(self.params[space.param])
        var lead = 0
        for index in range(len(self.weights)):
            if self.weights[index] > self.weights[lead]:
                lead = index
        var leader_clip = space.samples[lead].clip
        var natural = Float32(0)
        for index in range(len(self.weights)):
            natural += self.weights[index] * self.clips[space.samples[index].clip].speed
        var rate = self.rate
        if space.ground_rate and natural > 0:
            var wanted = self.ground_speed / natural
            rate = max(space.min_rate, min(space.max_rate, wanted))
            self.stride_scale = max(Float32(0.7), min(Float32(1.4), wanted / max(Float32(0.01), rate)))
        self._advance_blend_phase(dt, restarted, False, st.blend, leader_clip, self.weights[lead], rate)

    def sample_time(self, sample_clip: Int, leader_clip: Int, phase: Float32, synced: Bool) -> Float32:
        ref track = self.tracks[sample_clip]
        ref lead = self.tracks[leader_clip]
        if synced and (sample_clip == leader_clip or track.matches(lead)):
            return track.time_at(phase)
        var share = lead.time_at(phase) / max(Float32(0.0001), lead.duration)
        return share * self.clips[sample_clip].duration

    def evaluate_base(mut self, model: NativeModel, mut out: Pose, extra: Float32):
        """The state's pose now, or `extra` seconds ahead (for inertializer velocity)."""
        ref st = self.graph.states[self.state]
        if st.clip >= 0:
            ref info = self.clips[st.clip]
            var t = self.clip_time
            if extra > 0 and st.time_param < 0:
                t += extra * self.rate
                t = wrap_time(t, info.duration) if info.loop else max(Float32(0), min(t, info.duration))
            sample_pose(model, st.clip, t, out)
            return

        if st.blend_2d >= 0:
            ref space2d = self.graph.blend_spaces_2d[st.blend_2d]
            var lead2d = self.leader if self.leader >= 0 else space2d.samples[0].clip
            var phase2d = self.phase
            if extra > 0:
                ref track2d = self.tracks[lead2d]
                var seg_len2d = self._blended_segment_length_2d(st.blend_2d, lead2d, Int(phase2d))
                phase2d = wrap_time(phase2d + extra * self.rate / seg_len2d, Float32(track2d.markers()))
            var total2d = Float32(0)
            for index in range(len(space2d.samples)):
                var w = self.weights[index] if index < len(self.weights) else Float32(0)
                if w <= 0.0001:
                    continue
                var clip = space2d.samples[index].clip
                var t = self.sample_time(clip, lead2d, phase2d, space2d.sync)
                if total2d <= 0:
                    sample_pose(model, clip, t, out)
                else:
                    sample_pose(model, clip, t, self.scratch)
                    blend_into(out, self.scratch, w / (total2d + w))
                total2d += w
            if total2d <= 0:
                sample_pose(model, lead2d, 0, out)
            return

        ref space = self.graph.blend_spaces[st.blend]
        var lead = self.leader if self.leader >= 0 else space.samples[0].clip
        var phase = self.phase
        if extra > 0:
            ref track = self.tracks[lead]
            var seg_len = self._blended_segment_length_1d(st.blend, lead, Int(phase))
            phase = wrap_time(phase + extra * self.rate / seg_len, Float32(track.markers()))
        var total = Float32(0)
        for index in range(len(space.samples)):
            var w = self.weights[index] if index < len(self.weights) else Float32(0)
            if w <= 0.0001:
                continue
            var clip = space.samples[index].clip
            var t = self.sample_time(clip, lead, phase, space.sync)
            if total <= 0:
                sample_pose(model, clip, t, out)
            else:
                sample_pose(model, clip, t, self.scratch)
                blend_into(out, self.scratch, w / (total + w))
            total += w
        if total <= 0:
            sample_pose(model, lead, 0, out)

    def root_motion(mut self, model: NativeModel, restarted: Bool):
        if self.root_node < 0 or self.root_node >= self.pose.count():
            return
        var mode = self.graph.states[self.state].root_motion
        var root = self.pose.t[self.root_node]
        var root_yaw = _quat_yaw(self.pose.r[self.root_node])
        if mode == ROOT_EXTRACT and not restarted:
            var active_clip = self.clip_of_state()
            if active_clip >= 0 and active_clip < len(model.animations) and self.graph.states[self.state].clip >= 0:
                var extracted = extract_root_motion(
                    model,
                    active_clip,
                    self.prev_clip_time,
                    self.clip_time,
                    self.last_looped,
                    self.root_node,
                    self.root_scratch_a,
                    self.root_scratch_b,
                )
                self.root_delta = Vec3(extracted[0].x, Float32(0.0), extracted[0].z)
                self.root_yaw_delta = extracted[1]
            elif self.last_looped > 0 and active_clip >= 0 and active_clip < len(model.animations):
                var extracted_blend = extract_root_motion(
                    model,
                    active_clip,
                    self.prev_clip_time,
                    self.clip_time,
                    self.last_looped,
                    self.root_node,
                    self.root_scratch_a,
                    self.root_scratch_b,
                )
                self.root_delta = Vec3(extracted_blend[0].x, Float32(0.0), extracted_blend[0].z)
                self.root_yaw_delta = extracted_blend[1]
            else:
                var delta = root - self.last_root
                self.root_delta = Vec3(delta.x, Float32(0.0), delta.z)
                self.root_yaw_delta = _wrap_rad(root_yaw - self.last_root_yaw)
            self.root_travel = self.root_travel + self.root_delta
        if restarted:
            self.root_travel = Vec3()
            self.root_yaw_delta = Float32(0.0)
        self.last_root = root
        self.last_root_yaw = root_yaw
        if mode == ROOT_STRIP or mode == ROOT_EXTRACT:
            var base = model.nodes[self.root_node].base_translation
            self.pose.t[self.root_node] = Vec3(base.x, root.y, base.z)
            if self.extract_root_yaw:
                var base_yaw = _quat_yaw(model.nodes[self.root_node].base_rotation)
                var strip_q = Quat(0.0, det_sin((base_yaw - root_yaw) * Float32(0.5)), 0.0, det_cos((base_yaw - root_yaw) * Float32(0.5)))
                self.pose.r[self.root_node] = strip_q.multiply(self.pose.r[self.root_node]).normalized()

    def update_layers(mut self, dt: Float32):
        for index in range(len(self.graph.layers)):
            ref layer = self.graph.layers[index]
            var on = all_hold(layer.when, self.params)
            if on and len(layer.over) > 0:
                on = False
                for state in layer.over:
                    if state == self.state:
                        on = True
            var target = Float32(0)
            if on:
                target = layer.weight
                if layer.weight_param >= 0:
                    target *= max(Float32(0), min(Float32(1), self.params[layer.weight_param]))
            var step = dt / max(Float32(0.001), layer.fade)
            var w = self.layers[index].weight
            if w < target:
                w = min(target, w + step)
            else:
                w = max(target, w - step)
            if w > 0 and not self.layers[index].on:
                self.layers[index].time = 0
                self.layers[index].on = True
                if layer.clip >= 0:
                    passed_events(self.clips[layer.clip], layer.clip, -1, 0, 0, w, self.events)
            elif w > 0:
                ref info = self.clips[layer.clip]
                var before = self.layers[index].time
                var t = before + dt * layer.rate
                var looped = 0
                if layer.loop and info.duration > 0:
                    while t < 0:
                        t += info.duration
                    while t >= info.duration:
                        t -= info.duration
                        looped += 1
                else:
                    t = max(Float32(0), min(t, info.duration))
                if w >= 0.5:
                    passed_events(info, layer.clip, before, t, looped, w, self.events)
                self.layers[index].time = t
            if w <= 0:
                self.layers[index].on = False
            self.layers[index].weight = w

    def apply_layers(mut self, model: NativeModel, mut out: Pose):
        for index in range(len(self.graph.layers)):
            var w = self.layers[index].weight
            if w <= 0.001:
                continue
            ref layer = self.graph.layers[index]
            sample_pose(model, layer.clip, self.layers[index].time, self.scratch)
            if layer.additive:
                if index < len(self.layer_refs) and self.layer_refs[index].count() == len(model.nodes):
                    if layer.mask >= 0:
                        add_additive(out, self.scratch, self.layer_refs[index], self.masks[layer.mask], w)
                    else:
                        add_additive(out, self.scratch, self.layer_refs[index], self.all_nodes, w)
                else:
                    sample_pose(model, layer.clip, layer.reference_time, self.reference)
                    if layer.mask >= 0:
                        add_additive(out, self.scratch, self.reference, self.masks[layer.mask], w)
                    else:
                        add_additive(out, self.scratch, self.reference, self.all_nodes, w)
            elif layer.mask >= 0:
                blend_masked(out, self.scratch, self.masks[layer.mask], w)
            else:
                blend_masked(out, self.scratch, self.all_nodes, w)

    def layer_cover(self, mut cover: List[Float32], count: Int, skip: Int = -1):
        """For each node, the share of the pose the override layers that play
        now have written (0: the state's own pose, 1: all a layer's)."""
        if len(cover) != count:
            cover = List[Float32](length=count, fill=0)
        else:
            for index in range(count):
                cover[index] = 0
        for index in range(len(self.graph.layers)):
            var w = self.layers[index].weight
            ref layer = self.graph.layers[index]
            if w <= 0.001 or layer.additive or index == skip:
                continue
            if layer.mask >= 0:
                for node in range(min(count, len(self.masks[layer.mask]))):
                    var m = min(Float32(1), self.masks[layer.mask][node] * w)
                    if m > 0:
                        cover[node] = 1.0 - (1.0 - cover[node]) * (1.0 - m)
            else:
                for node in range(count):
                    cover[node] = 1.0 - (1.0 - cover[node]) * (1.0 - min(Float32(1), w))

    def add_additive_again(mut self, model: NativeModel, share: Float32):
        """Applies active additive layers onto `self.pose` again at `share` of their weight."""
        if share <= 0.001:
            return
        for index in range(len(self.graph.layers)):
            var w = self.layers[index].weight * share
            ref layer = self.graph.layers[index]
            if w <= 0.001 or not layer.additive:
                continue
            sample_pose(model, layer.clip, self.layers[index].time, self.scratch)
            if index < len(self.layer_refs) and self.layer_refs[index].count() == len(model.nodes):
                if layer.mask >= 0:
                    add_additive(self.pose, self.scratch, self.layer_refs[index], self.masks[layer.mask], w)
                else:
                    add_additive(self.pose, self.scratch, self.layer_refs[index], self.all_nodes, w)
            else:
                sample_pose(model, layer.clip, layer.reference_time, self.reference)
                if layer.mask >= 0:
                    add_additive(self.pose, self.scratch, self.reference, self.masks[layer.mask], w)
                else:
                    add_additive(self.pose, self.scratch, self.reference, self.all_nodes, w)

    def write(self, mut model: NativeModel):
        write_pose(model, self.pose)

    def apply_to_skeleton(self, mut skel: SkeletonInstance, model: NativeModel):
        """Applies `self.pose` into `skel` without mutating the shared `NativeModel`."""
        skel.apply_pose(self.pose, model)
