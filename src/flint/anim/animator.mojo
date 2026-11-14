"""The animation runtime (package 3.3): runs an `AnimGraphDef` on a model.

Each frame the driver sets the graph's parameters and calls `update`; the
animator picks the state, advances its clock (or its sync phase, for a
blend space), samples and blends the clips into a pose, removes root
motion where the state says so, adds the override and additive layers,
inertializes a state change and collects the events the clips passed.
`write_pose` then puts the pose in the model's nodes, and IK and the
procedural passes (anim/procedural.mojo) work from there.

Blend spaces play at one sync phase (anim/sync.mojo): the sample with the
most weight leads, its segment length sets how fast the phase moves, and
every sample is read at its own time for that phase. With a ground rate,
playback runs at ground speed over the blended ds_speed, inside the
space's rate limits; what the limits leave over is `stride_scale`, for the
stride warp. Distance matching for a stop: with `stop_distance` set, the
phase runs so the next foot marker lands where the character comes to
rest. Presentation only.
"""
from flint.core.math import Vec3
from flint.assets.model import NativeModel
from flint.anim.pose import (
    Pose, Inertializer, sample_pose, blend_into, blend_masked, add_additive,
    full_mask, bone_mask, write_pose,
)
from flint.anim.sync import ClipInfo, SyncTrack, FiredEvent, passed_events, wrap_time
from flint.anim.graph import (
    AnimGraphDef, StateDef, all_hold, ROOT_KEEP, ROOT_STRIP, ROOT_EXTRACT,
)


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
    var all_nodes: List[Float32]
    var root_node: Int
    # The state machine.
    var state: Int
    var previous: Int
    var state_time: Float32
    var clip_time: Float32
    var phase: Float32
    var switched: Bool
    var switches: Int
    var halflife: Float32
    var layers: List[LayerState]
    # This frame's results.
    var pose: Pose
    var ahead: Pose
    var scratch: Pose
    var reference: Pose
    var inertial: Inertializer
    var events: List[FiredEvent]
    var weights: List[Float32]
    var leader: Int
    var leader_time: Float32
    var rate: Float32
    var stride_scale: Float32
    var root_delta: Vec3
    var root_travel: Vec3
    var last_root: Vec3
    # Inputs besides the params: the ground speed for ground-rate blend
    # spaces, and the distance left to a stop (0: not stopping).
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
        self.all_nodes = List[Float32]()
        self.root_node = -1
        self.state = -1
        self.previous = -1
        self.state_time = 0
        self.clip_time = 0
        self.phase = 0
        self.switched = False
        self.switches = 0
        self.halflife = 0.08
        self.layers = List[LayerState]()
        self.pose = Pose()
        self.ahead = Pose()
        self.scratch = Pose()
        self.reference = Pose()
        self.inertial = Inertializer()
        self.events = List[FiredEvent]()
        self.weights = List[Float32]()
        self.leader = -1
        self.leader_time = 0
        self.rate = 1
        self.stride_scale = 1
        self.root_delta = Vec3()
        self.root_travel = Vec3()
        self.last_root = Vec3()
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
        self.phase = 0
        self.inertial.reset()
        for index in range(len(self.layers)):
            self.layers[index] = LayerState()
        self.root_travel = Vec3()

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
        """Masks and the root node for this model. A different model with
        the same nodes (another level of detail) keeps the animation going."""
        var hash = UInt64(14695981039346656037)
        for node in model.nodes:
            for byte in node.name.as_bytes():
                hash = (hash ^ UInt64(byte)) * 1099511628211
            hash = (hash ^ 0x2F) * 1099511628211
        if handle == self.handle and hash == self.signature and len(self.all_nodes) == len(model.nodes):
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

    def update(mut self, model: NativeModel, handle: Int, dt: Float32):
        """One frame: state, clocks, pose (in self.pose), events."""
        self.events.clear()
        self.switched = False
        self.root_delta = Vec3()
        if not self.configured() or len(model.nodes) == 0:
            return
        self.bind(model, handle)
        var chosen = self.select()
        var restart_clock = False
        if chosen != self.state:
            self.previous = self.state
            var was_blend = self.state >= 0 and self.graph.states[self.state].blend >= 0
            self.halflife = self.graph.blend_time(self.state, chosen)
            self.state = chosen
            self.switched = True
            self.switches += 1
            self.state_time = 0
            ref st = self.graph.states[chosen]
            if st.reset_time or st.clip >= 0:
                self.clip_time = 0
                restart_clock = True
            if st.blend >= 0 and not (was_blend and not st.reset_time):
                self.phase = 0
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
            swap(ahead, self.ahead)
        self.inertial.apply(out, self.ahead, self.switched, self.halflife, dt)
        swap(out, self.pose)

    def advance(mut self, dt: Float32, restarted: Bool):
        ref st = self.graph.states[self.state]
        self.stride_scale = 1
        self.rate = st.rate * (self.params[st.rate_param] if st.rate_param >= 0 else Float32(1))
        if st.clip >= 0:
            self.weights = List[Float32]()
            self.leader = st.clip
            ref info = self.clips[st.clip]
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
                        t += info.duration  # a rate param below 0 runs the loop backwards
                    while t >= info.duration:
                        t -= info.duration
                        looped += 1
                else:
                    t = max(Float32(0), min(t, info.duration))
                passed_events(info, st.clip, before, t, looped, 1, self.events)
                self.clip_time = t
            self.leader_time = self.clip_time
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
        var rate = st.rate
        if space.ground_rate and natural > 0:
            var wanted = self.ground_speed / natural
            rate = max(space.min_rate, min(space.max_rate, wanted))
            self.stride_scale = max(Float32(0.7), min(Float32(1.4), wanted / max(Float32(0.01), rate)))
        ref track = self.tracks[leader_clip]
        var matched = False
        if self.stop_distance > 0 and not restarted:
            # Distance matching for a stop: from the frame the stop starts,
            # the phase is a function of how much of the stopping distance is
            # covered, reaching the next foot marker as he comes to rest.
            if not self.stop_active or self.stop_distance > self.stop_total:
                self.stop_active = True
                self.stop_from = self.phase
                self.stop_target = Float32(Int(self.phase) + 1)
                self.stop_total = self.stop_distance
            var covered = max(Float32(0), min(Float32(1), 1.0 - self.stop_distance / max(Float32(0.0001), self.stop_total)))
            var wanted_phase = self.stop_from + (self.stop_target - self.stop_from) * covered
            var step = max(Float32(0), wanted_phase - self.phase)
            rate = step * track.segment_length(Int(self.phase)) / max(Float32(0.0001), dt)
            self.stride_scale = 1
            matched = True
        else:
            self.stop_active = False
        self.rate = rate
        # The leader's time before this frame (a new leader reads its own
        # time at the phase, so its events continue from there). A cycle
        # entered afresh starts on its first marker with that foot already
        # down: it is not a new step, so it does not fire.
        var before = track.time_at(self.phase)
        var remaining_time = Float32(0) if restarted else dt * rate
        if matched:
            remaining_time = 0
            var target_phase = self.stop_from + (self.stop_target - self.stop_from) * max(Float32(0), min(Float32(1), 1.0 - self.stop_distance / max(Float32(0.0001), self.stop_total)))
            if target_phase > self.phase:
                self.phase = target_phase
            if self.phase >= Float32(track.markers()):
                self.phase -= Float32(track.markers())
                self.stop_from -= Float32(track.markers())
                self.stop_target -= Float32(track.markers())
        var markers = track.markers()
        var guard = 0
        while remaining_time > 0 and guard < 64:
            var segment = min(markers - 1, Int(self.phase))
            var length = track.segment_length(segment)
            var left = (Float32(segment + 1) - self.phase) * length
            if remaining_time < left:
                self.phase += remaining_time / length
                break
            remaining_time -= left
            self.phase = Float32(segment + 1)
            if self.phase >= Float32(markers):
                self.phase -= Float32(markers)
            guard += 1
        var after = track.time_at(self.phase)
        var looped = 1 if (before >= 0 and after < before) else 0
        passed_events(self.clips[leader_clip], leader_clip, before, after, looped, self.weights[lead], self.events)
        self.leader = leader_clip
        self.leader_time = after
        self.clip_time = after

    def sample_time(self, sample_clip: Int, leader_clip: Int, phase: Float32, synced: Bool) -> Float32:
        ref track = self.tracks[sample_clip]
        ref lead = self.tracks[leader_clip]
        if synced and (sample_clip == leader_clip or track.matches(lead)):
            return track.time_at(phase)
        # Not synced: the same share of each clip.
        var share = lead.time_at(phase) / max(Float32(0.0001), lead.duration)
        return share * self.clips[sample_clip].duration

    def evaluate_base(mut self, model: NativeModel, mut out: Pose, extra: Float32):
        """The state's pose now, or extra seconds on (the inertializer's look ahead)."""
        ref st = self.graph.states[self.state]
        if st.clip >= 0:
            ref info = self.clips[st.clip]
            var t = self.clip_time
            if extra > 0 and st.time_param < 0:
                t += extra * self.rate
                t = wrap_time(t, info.duration) if info.loop else min(t, info.duration)
            sample_pose(model, st.clip, t, out)
            return
        ref space = self.graph.blend_spaces[st.blend]
        var lead = self.leader if self.leader >= 0 else space.samples[0].clip
        var phase = self.phase
        if extra > 0:
            ref track = self.tracks[lead]
            phase = wrap_time(phase + extra * self.rate / track.segment_length(Int(phase)), Float32(track.markers()))
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
        if mode == ROOT_EXTRACT and not restarted:
            var delta = root - self.last_root
            # A loop's wrap jumps back; that frame's delta is dropped.
            if delta.length() < 0.5:
                self.root_delta = Vec3(delta.x, 0, delta.z)
                self.root_travel = self.root_travel + self.root_delta
        if restarted:
            self.root_travel = Vec3()
        self.last_root = root
        if mode == ROOT_STRIP or mode == ROOT_EXTRACT:
            var base = model.nodes[self.root_node].base_translation
            self.pose.t[self.root_node] = Vec3(base.x, root.y, base.z)

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
                        t += info.duration  # a rate param below 0 runs the loop backwards
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
        now have written (0: the state's own pose, 1: all a layer's): what
        an action on the move is using of him. Package 3.23: a gait takes
        the walk's place under them, never theirs. `skip`: a layer left out
        (a gait asked for by name is how he carries, in the carry's place)."""
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
        """The additive layers that play now, onto the pose again at `share`
        of their weight: something took the place of `share` of the pose
        they were added to (under a gait the landing's thump stays)."""
        if share <= 0.001:
            return
        for index in range(len(self.graph.layers)):
            var w = self.layers[index].weight * share
            ref layer = self.graph.layers[index]
            if w <= 0.001 or not layer.additive:
                continue
            sample_pose(model, layer.clip, self.layers[index].time, self.scratch)
            sample_pose(model, layer.clip, layer.reference_time, self.reference)
            if layer.mask >= 0:
                add_additive(self.pose, self.scratch, self.reference, self.masks[layer.mask], w)
            else:
                add_additive(self.pose, self.scratch, self.reference, self.all_nodes, w)

    def write(self, mut model: NativeModel):
        write_pose(model, self.pose)
