"""A seekable two-clip move, with a ledge-relative contact, camera and cue track."""
from std.ffi import external_call
from flint.anim.pose import Pose, SkeletonInstance, blend_into, sample_pose
from flint.anim.procedural import Placement, smooth01
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.assets.model import NativeModel
from flint.core.math import Vec3
from flint.platform.os import read_bytes, write_synced, durable_rename


def _scalar(doc: JsonDocument, node: Int) raises -> Float32:
    if node < 0 or doc.nodes[node].kind != 2:
        raise Error("Expected a number in motion preset")
    var x = doc.number(node)
    if not (x >= -1e8 and x <= 1e8):
        raise Error("Motion preset number out of range")
    return Float32(x)


def _vec(doc: JsonDocument, key: String) raises -> Vec3:
    var a = doc.object_get(doc.root, key)
    if doc.array_size(a) != 3:
        raise Error("Expected three coordinates for " + key)
    return Vec3(_scalar(doc, doc.at(a, 0)), _scalar(doc, doc.at(a, 1)), _scalar(doc, doc.at(a, 2)))


def _vec_json(v: Vec3) -> String:
    return "[" + String(v.x) + ", " + String(v.y) + ", " + String(v.z) + "]"


def _clip(model: NativeModel, name: String) raises -> Int:
    for i in range(len(model.animations)):
        if model.animations[i].name == name and model.animations[i].duration > 0:
            return i
    raise Error("Missing or empty animation: " + name)


comptime TRACK_FPS = 30


def cue_kind(index: Int) -> String:
    var names = [String("Footstep"), String("Cloth"), String("Dust"), String("Shake")]
    if index < 0 or index >= len(names):
        return String("Unknown")
    return names[index]


@fieldwise_init
struct MotionCue(Copyable, ImplicitlyCopyable, Movable):
    var frame: Int
    var kind: Int

    def crossed(self, previous: Float32, current: Float32) -> Bool:
        var seconds = Float32(self.frame) / TRACK_FPS
        return previous < seconds and seconds <= current


struct MotionSequence(Copyable, Movable):
    var model_path: String
    var first: String
    var second: String
    var contact: String
    var support: String  # Optional second wrist, constrained to the ledge line.
    var markers: List[String]
    var overlap: Float32  # Fraction of the shorter clip.
    var start: Vec3     # Actor origin relative to the ledge in metres.
    var scale: Float32
    var eye_a: Vec3
    var eye_b: Vec3
    var aim: Vec3
    var cue: Float32     # Normalized sequence time.
    var sound: String   # Optional WAV file, relative to the preset.
    var ledge: Vec3     # Default instance for Studio; games can supply another.
    var yaw: Float32
    var events: List[MotionCue]

    def __init__(out self, path: String) raises:
        var doc = parse_json(read_bytes(path))
        if not doc.valid or doc.integer(doc.object_get(doc.root, "version")) != 1:
            raise Error("Expected a version 1 Flint motion preset: " + path)
        self.model_path = doc.text(doc.object_get(doc.root, "model"))
        self.first = doc.text(doc.object_get(doc.root, "first"))
        self.second = doc.text(doc.object_get(doc.root, "second"))
        self.contact = doc.text(doc.object_get(doc.root, "contact"))
        self.support = doc.text(doc.object_get(doc.root, "support"))
        self.markers = List[String]()
        var names = doc.object_get(doc.root, "markers")
        for i in range(doc.array_size(names)):
            self.markers.append(doc.text(doc.at(names, i)))
        self.overlap = _scalar(doc, doc.object_get(doc.root, "overlap"))
        self.start = _vec(doc, "start")
        self.scale = _scalar(doc, doc.object_get(doc.root, "scale"))
        self.eye_a = _vec(doc, "cameraStart")
        self.eye_b = _vec(doc, "cameraEnd")
        self.aim = _vec(doc, "cameraAim")
        self.cue = _scalar(doc, doc.object_get(doc.root, "cue"))
        self.sound = doc.text(doc.object_get(doc.root, "sound"))
        self.ledge = _vec(doc, "ledge")
        self.yaw = _scalar(doc, doc.object_get(doc.root, "yaw"))
        if not (self.overlap > 0 and self.overlap <= 1 and self.cue >= 0 and self.cue <= 1 and self.scale > 0):
            raise Error("Invalid overlap, cue time or model scale")

        self.events = []
        var events = doc.object_get(doc.root, "events")
        if events >= 0 and doc.nodes[events].kind != 4:
            raise Error("Expected an array of motion events")
        for i in range(doc.array_size(events)):
            var event = doc.at(events, i)
            var frame = _scalar(doc, doc.object_get(event, "frame"))
            var name = doc.text(doc.object_get(event, "kind"))
            var kind = -1
            for k in range(4):
                if cue_kind(k) == name:
                    kind = k
            if kind < 0 or frame < 0 or Float32(Int(frame)) != frame:
                raise Error("Invalid motion event")
            self.set_event(Int(frame), kind)

    def set_event(mut self, frame: Int, kind: Int):
        if frame < 0 or kind < 0 or kind >= 4:
            return
        for i in range(len(self.events)):
            if self.events[i].frame == frame and self.events[i].kind == kind:
                return
            if self.events[i].frame > frame:
                self.events.insert(i, MotionCue(frame, kind))
                return
        self.events.append(MotionCue(frame, kind))

    def save(self, path: String) raises:
        var marks = String("[")
        for i in range(len(self.markers)):
            if i:
                marks += ", "
            marks += json_quote(self.markers[i])
        var lines: List[String] = [
            String('"version": 1'), '"model": ' + json_quote(self.model_path),
            '"first": ' + json_quote(self.first), '"second": ' + json_quote(self.second),
            '"contact": ' + json_quote(self.contact), '"markers": ' + marks + "]",
            '"support": ' + json_quote(self.support),
            '"overlap": ' + String(self.overlap), '"scale": ' + String(self.scale),
            '"start": ' + _vec_json(self.start), '"ledge": ' + _vec_json(self.ledge),
            '"yaw": ' + String(self.yaw), '"cameraStart": ' + _vec_json(self.eye_a),
            '"cameraEnd": ' + _vec_json(self.eye_b), '"cameraAim": ' + _vec_json(self.aim),
            '"cue": ' + String(self.cue), '"sound": ' + json_quote(self.sound),
        ]
        var events = String("[")
        for i in range(len(self.events)):
            var event = self.events[i]
            events += (", " if i else "") + String('{"frame": ') + String(event.frame) + ', "kind": ' + json_quote(cue_kind(event.kind)) + "}"
        lines.append('"events": ' + events + "]")
        var result = String("{\n")
        for i in range(len(lines)):
            result += "  " + lines[i] + (",\n" if i + 1 < len(lines) else "\n}\n")
        var data = List[UInt8]()
        for byte in result.as_bytes():
            data.append(byte)
        var temporary = path + ".tmp." + String(external_call["getpid", Int32]())
        if not write_synced(temporary, data) or not durable_rename(temporary, path):
            raise Error("Could not save motion preset: " + path)

    def camera(self, phase: Float32, wall: Placement) -> Tuple[Vec3, Vec3]:
        return (wall.to_world(self.eye_a + (self.eye_b - self.eye_a) * smooth01(phase)), wall.to_world(self.aim))

    def cue_crossed(self, previous: Float32, current: Float32) -> Bool:
        """Forward playback only. Start at a negative phase to include a cue at zero."""
        return previous < self.cue and self.cue <= current


struct MotionPlayer(Movable):
    var first: Int
    var second: Int
    var contact: Int
    var wrist: Int
    var elbow: Int
    var shoulder: Int
    var markers: List[Int]
    var bones: SkeletonInstance
    var pose: Pose
    var incoming: Pose

    def __init__(out self, model: NativeModel, sequence: MotionSequence) raises:
        self.first = _clip(model, sequence.first)
        self.second = _clip(model, sequence.second)
        self.contact = model.find_node(sequence.contact) if sequence.contact else -1
        if sequence.contact and self.contact < 0:
            raise Error("Missing contact joint: " + sequence.contact)
        self.markers = List[Int]()
        for name in sequence.markers:
            var joint = model.find_node(name)
            if joint < 0:
                raise Error("Missing preview joint: " + name)
            self.markers.append(joint)
        self.bones = SkeletonInstance(len(model.nodes))
        self.bones.reset_to_rest(model)
        self.wrist = model.find_node(sequence.support) if sequence.support else -1
        self.elbow = self.bones.parents[self.wrist] if self.wrist >= 0 else -1
        self.shoulder = self.bones.parents[self.elbow] if self.elbow >= 0 else -1
        if sequence.support and (self.contact < 0 or self.wrist == self.contact or self.shoulder < 0):
            raise Error("Support contact needs a separate three-joint limb and primary contact")
        self.pose = Pose(len(model.nodes))
        self.incoming = Pose(len(model.nodes))

    def duration(self, model: NativeModel, sequence: MotionSequence) -> Float32:
        var a = model.animations[self.first].duration
        var b = model.animations[self.second].duration
        return a + b - min(a, b) * sequence.overlap

    def can_catch(self, wall: Placement, hand: Vec3, velocity: Vec3, reach: Float32) -> Bool:
        """Test a descending hand on the approach side. The wall faces local -Z."""
        if self.contact < 0 or reach <= 0 or velocity.y > 0:
            return False
        var local = wall.to_model(hand)
        return local.z <= 0 and wall.vector_to_model(velocity).z > 0 and (hand - wall.position).length() <= reach

    def sample(mut self, model: NativeModel, sequence: MotionSequence, wall: Placement, seconds: Float32, entry: Optional[Vec3] = None) -> Placement:
        """Evaluate from time alone. Draw bones.pose at the returned placement."""
        var a = model.animations[self.first].duration
        var b = model.animations[self.second].duration
        var overlap = min(a, b) * sequence.overlap
        var t = max(Float32(0), min(a + b - overlap, seconds))
        var incoming_t = max(Float32(0), t - (a - overlap))
        var weight: Float32
        if overlap <= Float32(1.0e-12):
            weight = Float32(1.0) if incoming_t > Float32(0.0) else Float32(0.0)
        else:
            weight = smooth01(incoming_t / overlap)
        sample_pose(model, self.first, min(a, t), self.pose)
        if incoming_t > 0:
            sample_pose(model, self.second, min(b, incoming_t), self.incoming)
            blend_into(self.pose, self.incoming, weight)
        self.bones.apply_pose(self.pose, model)
        var start = entry.value() if entry else wall.to_world(sequence.start)
        var actor = Placement(start, wall.scale * sequence.scale, wall.yaw)
        if self.contact >= 0:
            var hand = actor.to_world(self.bones.global_position(self.contact))
            actor.position = actor.position + (wall.position - hand) * weight
        if self.wrist >= 0 and weight > 0:
            var wrist = actor.to_world(self.bones.global_position(self.wrist))
            var offset = wall.to_model(wrist)
            var target = actor.to_model(wall.to_world(Vec3(offset.x, 0, 0)))
            var pole = self.bones.global_position(self.elbow)
            self.bones.solve_two_bone(self.shoulder, self.elbow, self.wrist, target, pole, model, weight)
        return actor
