"""The animation graph as data (package 3.3): states, 1D blend spaces, bone
masks, override and additive layers, transition blend times and the
runtime's settings, read from `assets/anim/<name>.json`.

The state machine picks, each frame, the first state in file order whose
`when` conditions all hold (the last state, with no conditions, is the
fallback). A state may `hold`: once in it, it stays while its `while`
conditions hold and its clip has not reached the named event (Punch's
`cancel_from` marker), unless a state listed before it takes over.
`transitions` give the blend time for a pair of states (`*` for any); the
first matching entry wins, then `blend`. A state plays one clip or a blend
space. A clip state's time may come from a parameter in [0, 1] instead of
the clock (`time_param`): that is how distance matching drives the jump
and the fall from the height above the ground.

Conditions are `[param, op, value]` with op one of < <= > >= == !=; a value
may be a clip name, which stands for that clip's number in the game's clip
order. Parameter names must be among `params`, and the code that drives
the graph says which names it sets (`AnimGraphDef.missing_params`).
"""
from std.math import isfinite
from flint.core.detmath import det_sin, det_cos
from flint.assets.json import JsonDocument, parse_json
from flint.platform.os import read_bytes

comptime OP_LT: Int = 0
comptime OP_LE: Int = 1
comptime OP_GT: Int = 2
comptime OP_GE: Int = 3
comptime OP_EQ: Int = 4
comptime OP_NE: Int = 5

comptime ROOT_KEEP: Int = 0     # the clip's root motion plays as authored
comptime ROOT_STRIP: Int = 1    # the root bone's horizontal travel is removed
comptime ROOT_EXTRACT: Int = 2  # removed and reported (Animator.root_delta)

comptime WARP_NONE: Int = 0
comptime WARP_GRAB: Int = 1     # hands to the piece being torn up
comptime WARP_MANTLE: Int = 2   # hands to the ledge's lip
comptime WARP_HUG: Int = 3      # arms round a target (3.8's bear hug)

comptime ANIM_GRAPH_PATH = "assets/anim/default.json"


struct Condition(Copyable, ImplicitlyCopyable, Movable):
    var param: Int
    var op: Int
    var value: Float32

    def __init__(out self, param: Int, op: Int, value: Float32):
        self.param = param
        self.op = op
        self.value = value

    def holds(self, params: List[Float32]) -> Bool:
        if self.param < 0 or self.param >= len(params):
            return False
        var v = params[self.param]
        if self.op == OP_LT:
            return v < self.value
        if self.op == OP_LE:
            return v <= self.value
        if self.op == OP_GT:
            return v > self.value
        if self.op == OP_GE:
            return v >= self.value
        if self.op == OP_EQ:
            return abs(v - self.value) < 0.0001
        return abs(v - self.value) >= 0.0001


def all_hold(conditions: List[Condition], params: List[Float32]) -> Bool:
    for condition in conditions:
        if not condition.holds(params):
            return False
    return True


struct MaskDef(Copyable, Movable):
    var name: String
    var bones: List[String]
    var weights: List[Float32]
    var children: Bool

    def __init__(out self):
        self.name = ""
        self.bones = List[String]()
        self.weights = List[Float32]()
        self.children = True


struct BlendSampleDef(Copyable, ImplicitlyCopyable, Movable):
    var clip: Int
    var at: Float32

    def __init__(out self, clip: Int, at: Float32):
        self.clip = clip
        self.at = at


struct BlendSample2DDef(Copyable, ImplicitlyCopyable, Movable):
    """Sample in a 2D blend space (polar or Cartesian coordinates)."""
    var clip: Int
    var x: Float32
    var y: Float32

    def __init__(out self, clip: Int, x: Float32, y: Float32):
        self.clip = clip
        self.x = x
        self.y = y


struct BlendSpace2DDef(Copyable, Movable):
    """A 2D blend space evaluated via Johansen (2009) Gradient-Band Interpolation
    in either Freeform Cartesian (`is_polar = False`) or Freeform Polar/Directional
    (`is_polar = True`, where `x` is angle in radians and `y` is speed/magnitude)."""
    var name: String
    var param_x: Int
    var param_y: Int
    var samples: List[BlendSample2DDef]
    var is_polar: Bool
    var sync: Bool
    var ground_rate: Bool
    var min_rate: Float32
    var max_rate: Float32

    def __init__(out self):
        self.name = ""
        self.param_x = -1
        self.param_y = -1
        self.samples = List[BlendSample2DDef]()
        self.is_polar = False
        self.sync = True
        self.ground_rate = False
        self.min_rate = 0.5
        self.max_rate = 2.0

    def weights(self, x: Float32, y: Float32) -> List[Float32]:
        """Johansen (2009) Gradient-Band Interpolation weights summing to 1.0."""
        var n = len(self.samples)
        var out = List[Float32](length=n, fill=0)
        if n == 0:
            return out^
        if n == 1:
            out[0] = Float32(1.0)
            return out^
        if not (isfinite(x) and isfinite(y)):
            out[0] = Float32(1.0)
            return out^

        var total_w = Float32(0.0)
        if not self.is_polar:
            for i in range(n):
                var xi = self.samples[i].x
                var yi = self.samples[i].y
                var vip_x = x - xi
                var vip_y = y - yi
                var wi = Float32(1.0)
                for j in range(n):
                    if i == j:
                        continue
                    var vij_x = self.samples[j].x - xi
                    var vij_y = self.samples[j].y - yi
                    var denom = vij_x * vij_x + vij_y * vij_y
                    if denom > Float32(1.0e-10):
                        var hij_raw = Float32(1.0) - (vip_x * vij_x + vip_y * vij_y) / denom
                        if isfinite(hij_raw):
                            var hij = max(Float32(0.0), min(Float32(1.0), hij_raw))
                            if hij < wi:
                                wi = hij
                out[i] = wi
                total_w += wi
        else:
            # Branch-cut-free polar gradient bands on S^1 x R (`x` = angle rad, `y` = magnitude)
            var cx = det_cos(x)
            var sx = det_sin(x)
            for i in range(n):
                var cxi = det_cos(self.samples[i].x)
                var sxi = det_sin(self.samples[i].x)
                var ri = self.samples[i].y
                var wi = Float32(1.0)
                for j in range(n):
                    if i == j:
                        continue
                    var cxj = det_cos(self.samples[j].x)
                    var sxj = det_sin(self.samples[j].x)
                    var rj = self.samples[j].y
                    var r_scale = max(Float32(0.5), Float32(0.5) * (abs(ri) + abs(rj)))
                    var vij_0 = (cxj - cxi) * r_scale
                    var vij_1 = (sxj - sxi) * r_scale
                    var vij_2 = rj - ri
                    var denom = vij_0 * vij_0 + vij_1 * vij_1 + vij_2 * vij_2
                    if denom > Float32(1.0e-10):
                        var vip_0 = (cx - cxi) * r_scale
                        var vip_1 = (sx - sxi) * r_scale
                        var vip_2 = y - ri
                        var hij_raw = Float32(1.0) - (vip_0 * vij_0 + vip_1 * vij_1 + vip_2 * vij_2) / denom
                        if isfinite(hij_raw):
                            var hij = max(Float32(0.0), min(Float32(1.0), hij_raw))
                            if hij < wi:
                                wi = hij
                out[i] = wi
                total_w += wi

        if total_w > Float32(1.0e-6):
            var inv_tot = Float32(1.0) / total_w
            for i in range(n):
                out[i] *= inv_tot
        else:
            # Degenerate exterior fallback: pick nearest sample
            var best = 0
            var best_d2 = Float32(1.0e30)
            for i in range(n):
                var d2: Float32
                if self.is_polar:
                    var dc = det_cos(x) - det_cos(self.samples[i].x)
                    var ds = det_sin(x) - det_sin(self.samples[i].x)
                    var dr = y - self.samples[i].y
                    d2 = dc * dc + ds * ds + dr * dr
                else:
                    var dx = x - self.samples[i].x
                    var dy = y - self.samples[i].y
                    d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = i
            out[best] = Float32(1.0)
        return out^


struct BlendSpaceDef(Copyable, Movable):
    """A 1D blend space: samples along one parameter, played at one sync
    phase. With `ground_rate`, playback runs at the ground speed over the
    blended clips' ds_speed, clamped to [min_rate, max_rate]; the stride
    warp makes up the rest."""
    var name: String
    var param: Int
    var samples: List[BlendSampleDef]
    var sync: Bool
    var ground_rate: Bool
    var min_rate: Float32
    var max_rate: Float32

    def __init__(out self):
        self.name = ""
        self.param = -1
        self.samples = List[BlendSampleDef]()
        self.sync = True
        self.ground_rate = False
        self.min_rate = 0.5
        self.max_rate = 2.0

    def weights(self, value: Float32) -> List[Float32]:
        """Piecewise linear between neighbouring samples, clamped at the ends."""
        var n = len(self.samples)
        var out = List[Float32](length=n, fill=0)
        if n == 0:
            return out^
        if n == 1 or value <= self.samples[0].at:
            out[0] = 1
            return out^
        if value >= self.samples[n - 1].at:
            out[n - 1] = 1
            return out^
        for index in range(n - 1):
            var a = self.samples[index].at
            var b = self.samples[index + 1].at
            if value >= a and value <= b:
                var w = (value - a) / max(Float32(0.0001), b - a)
                out[index] = 1 - w
                out[index + 1] = w
                return out^
        out[n - 1] = 1
        return out^


struct StateDef(Copyable, Movable):
    var name: String
    var clip: Int          # -1 for a blend-space state
    var blend: Int         # 1D blend space index, or -1
    var blend_2d: Int      # 2D blend space index, or -1
    var rate: Float32
    var time_param: Int    # -1: clock; else a param in [0, 1] is the clip's normalised time
    var rate_param: Int    # -1: none; else the clock runs at `rate` times this param (0 holds the pose, below 0 runs it backwards: package 3.7)
    var when: List[Condition]
    var hold_while: List[Condition]
    var hold_event: String
    var holds: Bool
    var root_motion: Int
    var warp: Int
    var reset_time: Bool   # re-entering starts the clip over (default); False keeps the phase

    def __init__(out self):
        self.name = ""
        self.clip = -1
        self.blend = -1
        self.blend_2d = -1
        self.rate = 1
        self.time_param = -1
        self.rate_param = -1
        self.when = List[Condition]()
        self.hold_while = List[Condition]()
        self.hold_event = ""
        self.holds = False
        self.root_motion = ROOT_KEEP
        self.warp = WARP_NONE
        self.reset_time = True


struct LayerDef(Copyable, Movable):
    """An override (masked) or additive layer over the base state."""
    var name: String
    var clip: Int
    var mask: Int          # mask index, -1: every node
    var additive: Bool
    var reference_time: Float32
    var when: List[Condition]
    var over: List[Int]    # base states it plays over; empty: any
    var fade: Float32      # seconds to fade fully in or out
    var weight: Float32
    var weight_param: Int  # -1, or a param in [0, 1] that scales the weight
    var rate: Float32
    var loop: Bool

    def __init__(out self):
        self.name = ""
        self.clip = -1
        self.mask = -1
        self.additive = False
        self.reference_time = 0
        self.when = List[Condition]()
        self.over = List[Int]()
        self.fade = 0.12
        self.weight = 1
        self.weight_param = -1
        self.rate = 1
        self.loop = True


struct TransitionDef(Copyable, ImplicitlyCopyable, Movable):
    var source: Int  # -1: any
    var target: Int  # -1: any
    var blend: Float32

    def __init__(out self, source: Int, target: Int, blend: Float32):
        self.source = source
        self.target = target
        self.blend = blend


struct FormOffsetDef(Copyable, Movable):
    """One form's retarget offsets (package 3.9): the forms share the base
    rig and clips, and each draws them on its own proportions. `scale`
    multiplies the draw scale, `bones` scale joints (and what hangs from
    them) on top of the clips, `stride` is how much farther a step carries
    the form at the same clip speed (root motion and the locomotion rate
    are divided by it), and the foot settings scale four-leg foot placement's
    paw height and half body length."""
    var name: String
    var scale: Float32
    var stride: Float32
    var hand_plant: Float32
    var half_length: Float32
    var bones: List[String]
    var bone_scales: List[Float32]

    def __init__(out self):
        self.name = ""
        self.scale = 1
        self.stride = 1
        self.hand_plant = 1
        self.half_length = 1
        self.bones = List[String]()
        self.bone_scales = List[Float32]()

    def bone_scale(self, bone: String) -> Float32:
        for index in range(len(self.bones)):
            if self.bones[index] == bone:
                return self.bone_scales[index]
        return 1


struct AnimGraphDef(Copyable, Movable):
    var name: String
    var params: List[String]
    var masks: List[MaskDef]
    var blend_spaces: List[BlendSpaceDef]
    var blend_spaces_2d: List[BlendSpace2DDef]
    var states: List[StateDef]
    var layers: List[LayerDef]
    var transitions: List[TransitionDef]
    var default_blend: Float32
    var setting_names: List[String]
    var setting_values: List[Float32]
    var forms: List[FormOffsetDef]

    def __init__(out self):
        self.name = ""
        self.params = List[String]()
        self.masks = List[MaskDef]()
        self.blend_spaces = List[BlendSpaceDef]()
        self.blend_spaces_2d = List[BlendSpace2DDef]()
        self.states = List[StateDef]()
        self.layers = List[LayerDef]()
        self.transitions = List[TransitionDef]()
        self.default_blend = 0.08
        self.setting_names = List[String]()
        self.setting_values = List[Float32]()
        self.forms = List[FormOffsetDef]()

    def valid(self) -> Bool:
        return len(self.states) > 0

    def blend_space_2d(self, name: String) -> Int:
        for index in range(len(self.blend_spaces_2d)):
            if self.blend_spaces_2d[index].name == name:
                return index
        return -1

    def param(self, name: String) -> Int:
        for index in range(len(self.params)):
            if self.params[index] == name:
                return index
        return -1

    def state(self, name: String) -> Int:
        for index in range(len(self.states)):
            if self.states[index].name == name:
                return index
        return -1

    def layer(self, name: String) -> Int:
        for index in range(len(self.layers)):
            if self.layers[index].name == name:
                return index
        return -1

    def form(self, name: String) -> Int:
        for index in range(len(self.forms)):
            if self.forms[index].name == name:
                return index
        return -1

    def setting(self, name: String, fallback: Float32) -> Float32:
        for index in range(len(self.setting_names)):
            if self.setting_names[index] == name:
                return self.setting_values[index]
        return fallback

    def blend_time(self, source: Int, target: Int) -> Float32:
        for transition in self.transitions:
            if (transition.source < 0 or transition.source == source) and (transition.target < 0 or transition.target == target):
                return transition.blend
        return self.default_blend

    def missing_params(self, provided: List[String]) -> List[String]:
        """Params the file declares that the driving code never sets."""
        var missing = List[String]()
        for name in self.params:
            var found = False
            for known in provided:
                if known == name:
                    found = True
                    break
            if not found:
                missing.append(name)
        return missing^


def op_code(op: String) raises -> Int:
    if op == "<":
        return OP_LT
    if op == "<=":
        return OP_LE
    if op == ">":
        return OP_GT
    if op == ">=":
        return OP_GE
    if op == "==":
        return OP_EQ
    if op == "!=":
        return OP_NE
    raise Error("unknown operator " + op)


def clip_number(clips: List[String], name: String) -> Int:
    for index in range(len(clips)):
        if clips[index] == name:
            return index
    return -1


def read_conditions(document: JsonDocument, node: Int, graph: AnimGraphDef, clips: List[String], where: String) raises -> List[Condition]:
    var out = List[Condition]()
    for index in range(document.array_size(node)):
        var item = document.at(node, index)
        if document.array_size(item) != 3:
            raise Error(where + ": a condition is [param, op, value]")
        var name = document.text(document.at(item, 0))
        var param = graph.param(name)
        if param < 0:
            raise Error(where + ": unknown param " + name)
        var op = op_code(document.text(document.at(item, 1)))
        var value_node = document.at(item, 2)
        var value: Float32
        if document.nodes[value_node].kind == 3:
            var clip = clip_number(clips, document.text(value_node))
            if clip < 0:
                raise Error(where + ": unknown clip " + document.text(value_node))
            value = Float32(clip)
        else:
            value = Float32(document.number(value_node))
        out.append(Condition(param, op, value))
    return out^


def need_clip(clips: List[String], name: String, where: String) raises -> Int:
    var clip = clip_number(clips, name)
    if clip < 0:
        raise Error(where + ": unknown clip " + name)
    return clip


def parse_anim_graph(document: JsonDocument, clips: List[String]) raises -> AnimGraphDef:
    """The graph in a parsed file; clips are the game's clip names in order.
    Raises with what is wrong: the caller keeps its old graph."""
    if not document.valid:
        raise Error("animation graph: not valid JSON")
    var root = document.root
    var graph = AnimGraphDef()
    graph.name = document.text(document.object_get(root, "name"), "graph")
    graph.default_blend = Float32(document.number(document.object_get(root, "blend"), 0.08))
    var params = document.object_get(root, "params")
    for index in range(document.array_size(params)):
        var name = document.text(document.at(params, index))
        if graph.param(name) >= 0:
            raise Error("params: " + name + " twice")
        graph.params.append(name)
    var settings = document.object_get(root, "settings")
    if settings >= 0:
        ref node = document.nodes[settings]
        for index in range(len(node.keys)):
            graph.setting_names.append(node.keys[index])
            graph.setting_values.append(Float32(document.number(node.children[index])))
    var masks = document.object_get(root, "masks")
    if masks >= 0:
        ref node = document.nodes[masks]
        for index in range(len(node.keys)):
            var mask = MaskDef()
            mask.name = node.keys[index]
            var body = node.children[index]
            mask.children = document.boolean(document.object_get(body, "children"), True)
            var bones = document.object_get(body, "bones")
            if bones < 0:
                raise Error("mask " + mask.name + ": no bones")
            ref bone_node = document.nodes[bones]
            for b in range(len(bone_node.keys)):
                mask.bones.append(bone_node.keys[b])
                mask.weights.append(Float32(document.number(bone_node.children[b], 1)))
            graph.masks.append(mask^)
    var forms = document.object_get(root, "forms")
    if forms >= 0:
        ref form_node = document.nodes[forms]
        for index in range(len(form_node.keys)):
            var offset = FormOffsetDef()
            offset.name = form_node.keys[index]
            var body = form_node.children[index]
            offset.scale = Float32(document.number(document.object_get(body, "scale"), 1))
            offset.stride = Float32(document.number(document.object_get(body, "stride"), 1))
            offset.hand_plant = Float32(document.number(document.object_get(body, "hand_plant"), 1))
            offset.half_length = Float32(document.number(document.object_get(body, "half_length"), 1))
            if offset.scale <= 0.2 or offset.scale > 3 or offset.stride <= 0.2 or offset.stride > 3:
                raise Error("form " + offset.name + ": scale and stride must be 0.2 to 3")
            var bones = document.object_get(body, "bones")
            if bones >= 0:
                ref bone_node = document.nodes[bones]
                for b in range(len(bone_node.keys)):
                    var value = Float32(document.number(bone_node.children[b], 1))
                    if value <= 0.2 or value > 3:
                        raise Error("form " + offset.name + ": bone " + bone_node.keys[b] + " scale must be 0.2 to 3")
                    offset.bones.append(bone_node.keys[b])
                    offset.bone_scales.append(value)
            graph.forms.append(offset^)
    var spaces = document.object_get(root, "blend_spaces")
    if spaces >= 0:
        ref node = document.nodes[spaces]
        for index in range(len(node.keys)):
            var space = BlendSpaceDef()
            space.name = node.keys[index]
            var body = node.children[index]
            var where = "blend space " + space.name
            space.param = graph.param(document.text(document.object_get(body, "param")))
            if space.param < 0:
                raise Error(where + ": unknown param " + document.text(document.object_get(body, "param")))
            space.sync = document.boolean(document.object_get(body, "sync"), True)
            space.ground_rate = document.text(document.object_get(body, "rate")) == "ground"
            space.min_rate = Float32(document.number(document.object_get(body, "min_rate"), 0.5))
            space.max_rate = Float32(document.number(document.object_get(body, "max_rate"), 2.0))
            var samples = document.object_get(body, "samples")
            var last = Float32(-1.0e9)
            for s in range(document.array_size(samples)):
                var sample = document.at(samples, s)
                var clip = need_clip(clips, document.text(document.object_get(sample, "clip")), where)
                var at = Float32(document.number(document.object_get(sample, "at")))
                if at <= last:
                    raise Error(where + ": samples must rise along " + document.text(document.object_get(body, "param")))
                last = at
                space.samples.append(BlendSampleDef(clip, at))
            if len(space.samples) == 0:
                raise Error(where + ": no samples")
            graph.blend_spaces.append(space^)
    var spaces_2d = document.object_get(root, "blend_spaces_2d")
    if spaces_2d >= 0:
        ref node2d = document.nodes[spaces_2d]
        for index in range(len(node2d.keys)):
            var space2d = BlendSpace2DDef()
            space2d.name = node2d.keys[index]
            var body2d = node2d.children[index]
            var where2d = "blend space 2d " + space2d.name
            var px_name = document.text(document.object_get(body2d, "param_x"))
            var py_name = document.text(document.object_get(body2d, "param_y"))
            space2d.param_x = graph.param(px_name)
            space2d.param_y = graph.param(py_name)
            if space2d.param_x < 0 or space2d.param_y < 0:
                raise Error(where2d + ": unknown param_x or param_y")
            space2d.is_polar = document.boolean(document.object_get(body2d, "polar"), False)
            space2d.sync = document.boolean(document.object_get(body2d, "sync"), True)
            space2d.ground_rate = document.text(document.object_get(body2d, "rate")) == "ground"
            space2d.min_rate = Float32(document.number(document.object_get(body2d, "min_rate"), 0.5))
            space2d.max_rate = Float32(document.number(document.object_get(body2d, "max_rate"), 2.0))
            var samples2d = document.object_get(body2d, "samples")
            for s in range(document.array_size(samples2d)):
                var sample2d = document.at(samples2d, s)
                var clip2d = need_clip(clips, document.text(document.object_get(sample2d, "clip")), where2d)
                var sx = Float32(document.number(document.object_get(sample2d, "x")))
                var sy = Float32(document.number(document.object_get(sample2d, "y")))
                space2d.samples.append(BlendSample2DDef(clip2d, sx, sy))
            if len(space2d.samples) == 0:
                raise Error(where2d + ": no samples")
            graph.blend_spaces_2d.append(space2d^)
    var states = document.object_get(root, "states")
    # Names first, so layers and transitions can name any state.
    for index in range(document.array_size(states)):
        var state = StateDef()
        state.name = document.text(document.object_get(document.at(states, index), "name"))
        if state.name.byte_length() == 0 or graph.state(state.name) >= 0:
            raise Error("states: a state needs a unique name (" + state.name + ")")
        graph.states.append(state^)
    for index in range(document.array_size(states)):
        var body = document.at(states, index)
        var where = "state " + graph.states[index].name
        var clip_name = document.text(document.object_get(body, "clip"))
        var space_name = document.text(document.object_get(body, "blend"))
        var space2d_name = document.text(document.object_get(body, "blend_2d"))
        if clip_name.byte_length() > 0:
            graph.states[index].clip = need_clip(clips, clip_name, where)
        elif space2d_name.byte_length() > 0:
            graph.states[index].blend_2d = graph.blend_space_2d(space2d_name)
            if graph.states[index].blend_2d < 0:
                raise Error(where + ": unknown 2d blend space " + space2d_name)
        elif space_name.byte_length() > 0:
            for s in range(len(graph.blend_spaces)):
                if graph.blend_spaces[s].name == space_name:
                    graph.states[index].blend = s
            if graph.states[index].blend < 0:
                graph.states[index].blend_2d = graph.blend_space_2d(space_name)
            if graph.states[index].blend < 0 and graph.states[index].blend_2d < 0:
                raise Error(where + ": unknown blend space " + space_name)
        else:
            raise Error(where + ": needs a clip or a blend space")
        graph.states[index].rate = Float32(document.number(document.object_get(body, "rate"), 1))
        var time_param = document.text(document.object_get(body, "time_param"))
        if time_param.byte_length() > 0:
            graph.states[index].time_param = graph.param(time_param)
            if graph.states[index].time_param < 0:
                raise Error(where + ": unknown param " + time_param)
        var rate_param = document.text(document.object_get(body, "rate_param"))
        if rate_param.byte_length() > 0:
            graph.states[index].rate_param = graph.param(rate_param)
            if graph.states[index].rate_param < 0:
                raise Error(where + ": unknown param " + rate_param)
        graph.states[index].when = read_conditions(document, document.object_get(body, "when"), graph, clips, where)
        var hold = document.object_get(body, "hold")
        if hold >= 0:
            graph.states[index].holds = True
            graph.states[index].hold_event = document.text(document.object_get(hold, "until"))
            graph.states[index].hold_while = read_conditions(document, document.object_get(hold, "while"), graph, clips, where)
        var root_mode = document.text(document.object_get(body, "root_motion"), "keep")
        if root_mode == "keep":
            graph.states[index].root_motion = ROOT_KEEP
        elif root_mode == "strip":
            graph.states[index].root_motion = ROOT_STRIP
        elif root_mode == "extract":
            graph.states[index].root_motion = ROOT_EXTRACT
        else:
            raise Error(where + ": root_motion is keep, strip or extract")
        var warp = document.text(document.object_get(body, "warp"), "")
        if warp == "grab":
            graph.states[index].warp = WARP_GRAB
        elif warp == "mantle":
            graph.states[index].warp = WARP_MANTLE
        elif warp == "hug":
            graph.states[index].warp = WARP_HUG
        elif warp.byte_length() > 0:
            raise Error(where + ": warp is grab, mantle or hug")
        graph.states[index].reset_time = document.boolean(document.object_get(body, "restart"), True)
    if len(graph.states) == 0:
        raise Error("animation graph: no states")
    if len(graph.states[len(graph.states) - 1].when) > 0:
        raise Error("animation graph: the last state is the fallback and has no conditions")
    var layers = document.object_get(root, "layers")
    for index in range(document.array_size(layers)):
        var body = document.at(layers, index)
        var layer = LayerDef()
        layer.name = document.text(document.object_get(body, "name"))
        var where = "layer " + layer.name
        layer.clip = need_clip(clips, document.text(document.object_get(body, "clip")), where)
        var mask_name = document.text(document.object_get(body, "mask"))
        if mask_name.byte_length() > 0:
            for m in range(len(graph.masks)):
                if graph.masks[m].name == mask_name:
                    layer.mask = m
            if layer.mask < 0:
                raise Error(where + ": unknown mask " + mask_name)
        layer.additive = document.boolean(document.object_get(body, "additive"), False)
        layer.reference_time = Float32(document.number(document.object_get(body, "reference_time"), 0))
        layer.when = read_conditions(document, document.object_get(body, "when"), graph, clips, where)
        var over = document.object_get(body, "over")
        for o in range(document.array_size(over)):
            var state = graph.state(document.text(document.at(over, o)))
            if state < 0:
                raise Error(where + ": unknown state " + document.text(document.at(over, o)))
            layer.over.append(state)
        layer.fade = Float32(document.number(document.object_get(body, "fade"), 0.12))
        layer.weight = Float32(document.number(document.object_get(body, "weight"), 1))
        var weight_param = document.text(document.object_get(body, "weight_param"))
        if weight_param.byte_length() > 0:
            layer.weight_param = graph.param(weight_param)
            if layer.weight_param < 0:
                raise Error(where + ": unknown param " + weight_param)
        layer.rate = Float32(document.number(document.object_get(body, "rate"), 1))
        layer.loop = document.boolean(document.object_get(body, "loop"), True)
        graph.layers.append(layer^)
    var transitions = document.object_get(root, "transitions")
    for index in range(document.array_size(transitions)):
        var body = document.at(transitions, index)
        var source_name = document.text(document.object_get(body, "from"), "*")
        var target_name = document.text(document.object_get(body, "to"), "*")
        var source = -1 if source_name == "*" else graph.state(source_name)
        var target = -1 if target_name == "*" else graph.state(target_name)
        if (source_name != "*" and source < 0) or (target_name != "*" and target < 0):
            raise Error("transition " + source_name + " -> " + target_name + ": unknown state")
        graph.transitions.append(TransitionDef(source, target, Float32(document.number(document.object_get(body, "blend"), Float64(graph.default_blend)))))
    return graph^


def read_anim_graph(path: String, clips: List[String]) raises -> AnimGraphDef:
    var bytes = read_bytes(path)
    if len(bytes) == 0:
        raise Error("animation graph: " + path + " is missing or empty")
    return parse_anim_graph(parse_json(bytes), clips)
