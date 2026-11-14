"""The JSON file the pipeline writes next to every export.

tools/blender/pipeline.py writes it and tools/asset_pipeline.mojo checks it. It
names every clip with its length, loop flag, root motion and events, plus the
sockets, combat volumes, collision shapes and climb route placed in Blender.
Read on the main thread, between ticks; nothing here reaches the simulation.
"""
from flint.core.math import Vec3
from flint.core.replay import Fnv64
from flint.assets.json import JsonDocument, parse_json
from flint.platform.os import file_size, read_bytes

# Bumped when an export or cooked layout stops being readable by older engines.
comptime ENGINE_FORMAT: Int = 1
# Package 3.4: what a skeleton entry's `helper` makes of the bone (anim/helper_joints.mojo).
comptime HELPER_HALF: Int = 1
comptime HELPER_TWIST: Int = 2


struct ClipEvent(Copyable, Movable):
    var kind: String
    var arg: String
    var time: Float32

    def __init__(out self, kind: String, arg: String, time: Float32):
        self.kind = kind
        self.arg = arg
        self.time = time


struct VisemeCurve(Copyable, Movable):
    var name: String
    var times: List[Float32]
    var values: List[Float32]

    def __init__(out self, name: String):
        self.name = name
        self.times = List[Float32]()
        self.values = List[Float32]()

    def sample(self, time: Float32) -> Float32:
        """Linear between keys, held outside them."""
        if len(self.times) == 0:
            return 0
        if time <= self.times[0]:
            return self.values[0]
        for index in range(1, len(self.times)):
            if time <= self.times[index]:
                var span = self.times[index] - self.times[index - 1]
                var amount = (time - self.times[index - 1]) / span if span > 0 else Float32(1)
                return self.values[index - 1] + (self.values[index] - self.values[index - 1]) * amount
        return self.values[len(self.values) - 1]


struct ManifestClip(Copyable, Movable):
    var name: String
    var length: Float32
    var frames: Int
    var loop: Bool
    var speed: Float32
    var root_motion: Vec3
    var events: List[ClipEvent]
    var visemes: List[VisemeCurve]

    def __init__(out self):
        self.name = ""
        self.length = 0
        self.frames = 0
        self.loop = False
        self.speed = 0
        self.root_motion = Vec3()
        self.events = List[ClipEvent]()
        self.visemes = List[VisemeCurve]()

    def event_time(self, kind: String, fallback: Float32 = -1) -> Float32:
        for event in self.events:
            if event.kind == kind:
                return event.time
        return fallback

    def events_between(self, start: Float32, end: Float32) -> List[ClipEvent]:
        """Events with start < time <= end, as played across one frame."""
        var found = List[ClipEvent]()
        for event in self.events:
            if event.time > start and event.time <= end:
                found.append(event.copy())
        return found^


struct ManifestBone(Copyable, Movable):
    var name: String
    var parent: String
    var head: Vec3
    # Package 3.3: a spring bone (the pose bone's ds_spring property in
    # Blender): its tail at rest, and how it swings (anim/spring_bones.mojo).
    var spring: Bool
    var tail: Vec3
    var stiffness: Float32
    var drag: Float32
    var gravity: Float32
    var radius: Float32
    var wind: Float32
    # Package 3.4: a helper joint (the pose bone's ds_helper property): the
    # engine poses it from `helper_source` (anim/helper_joints.mojo).
    var helper_kind: Int  # 0 none, HELPER_HALF, HELPER_TWIST
    var helper_source: String
    var helper_amount: Float32

    def __init__(out self, name: String, parent: String, head: Vec3):
        self.name = name
        self.parent = parent
        self.head = head
        self.spring = False
        self.tail = head
        self.stiffness = 0
        self.drag = 0
        self.gravity = 0
        self.radius = 0
        self.wind = 0
        self.helper_kind = 0
        self.helper_source = String()
        self.helper_amount = 0


struct ManifestCorrective(Copyable, Movable):
    """Package 3.5: a morph target driven by a bone's swing (anim/correctives.mojo):
    weight = clamp(x * swing_x + z * swing_z, 0, 1)."""
    var morph: String
    var bone: String
    var x: Float32
    var z: Float32

    def __init__(out self, morph: String, bone: String, x: Float32, z: Float32):
        self.morph = morph
        self.bone = bone
        self.x = x
        self.z = z


struct ManifestSocket(Copyable, Movable):
    var name: String
    var bone: String
    var matrix: List[Float32]  # column major, in the bone's frame

    def __init__(out self, name: String, bone: String, var matrix: List[Float32]):
        self.name = name
        self.bone = bone
        self.matrix = matrix^


struct ManifestVolume(Copyable, Movable):
    var role: String  # hit or hurt
    var bone: String
    var shape: String  # box, capsule or sphere
    var size: Vec3
    var matrix: List[Float32]

    def __init__(out self, role: String, bone: String, shape: String, size: Vec3, var matrix: List[Float32]):
        self.role = role
        self.bone = bone
        self.shape = shape
        self.size = size
        self.matrix = matrix^


struct ManifestShape(Copyable, Movable):
    var kind: String  # box, sphere, capsule or convex
    var center: Vec3
    var size: Vec3
    var points: List[Vec3]

    def __init__(out self, kind: String, center: Vec3, size: Vec3, var points: List[Vec3]):
        self.kind = kind
        self.center = center
        self.size = size
        self.points = points^


struct ManifestLod(Copyable, Movable):
    var file: String
    var triangles: Int

    def __init__(out self, file: String, triangles: Int):
        self.file = file
        self.triangles = triangles


struct AssetManifest(Copyable, Movable):
    var valid: Bool
    var error: String
    var asset: String
    var asset_id: String
    var kind: String  # mesh or animation
    var folder: String
    var blender_version: String
    var engine_format: Int
    var skinning: Int  # package 3.4: 0 linear blend, 1 dual quaternions, 2 centers of rotation; -1 not said
    var source_hash: String
    var content_hash: String
    var scripts_hash: String
    var stable_print: UInt64  # the file's bytes without the source hash, which a mere .blend save rewrites
    var file: String
    var animation: String
    var clip_prefix: String
    var height: Float32
    var influences: Int
    var morphs: List[String]
    var morphs_per_vertex: Int
    var correctives: List[ManifestCorrective]
    var lods: List[ManifestLod]
    var skeleton: List[ManifestBone]
    var clips: List[ManifestClip]
    var sockets: List[ManifestSocket]
    var volumes: List[ManifestVolume]
    var collision: List[ManifestShape]
    var climb_path: List[Vec3]

    def __init__(out self):
        self.valid = False
        self.error = ""
        self.asset = ""
        self.asset_id = ""
        self.kind = ""
        self.folder = ""
        self.blender_version = ""
        self.engine_format = 0
        self.skinning = -1
        self.source_hash = ""
        self.content_hash = ""
        self.scripts_hash = ""
        self.stable_print = 0
        self.file = ""
        self.animation = ""
        self.clip_prefix = ""
        self.height = 0
        self.influences = 0
        self.morphs = List[String]()
        self.morphs_per_vertex = 0
        self.correctives = List[ManifestCorrective]()
        self.lods = List[ManifestLod]()
        self.skeleton = List[ManifestBone]()
        self.clips = List[ManifestClip]()
        self.sockets = List[ManifestSocket]()
        self.volumes = List[ManifestVolume]()
        self.collision = List[ManifestShape]()
        self.climb_path = List[Vec3]()

    def find_clip(self, name: String) -> Int:
        for index in range(len(self.clips)):
            if self.clips[index].name == name:
                return index
        return -1

    def find_socket(self, name: String) -> Int:
        for index in range(len(self.sockets)):
            if self.sockets[index].name == name:
                return index
        return -1


def read_json_vec3(document: JsonDocument, node: Int) -> Vec3:
    return Vec3(Float32(document.number(document.at(node, 0))), Float32(document.number(document.at(node, 1))), Float32(document.number(document.at(node, 2))))


def read_matrix(document: JsonDocument, node: Int) -> List[Float32]:
    var matrix = List[Float32]()
    for index in range(document.array_size(node)):
        matrix.append(Float32(document.number(document.at(node, index))))
    return matrix^


def read_clip(document: JsonDocument, node: Int) -> ManifestClip:
    var clip = ManifestClip()
    clip.name = document.text(document.object_get(node, "name"))
    clip.length = Float32(document.number(document.object_get(node, "length")))
    clip.frames = document.integer(document.object_get(node, "frames"), 0)
    clip.loop = document.boolean(document.object_get(node, "loop"), False)
    clip.speed = Float32(document.number(document.object_get(node, "speed")))
    clip.root_motion = read_json_vec3(document, document.object_get(node, "root_motion"))
    var events = document.object_get(node, "events")
    for index in range(document.array_size(events)):
        var event = document.at(events, index)
        clip.events.append(ClipEvent(document.text(document.object_get(event, "kind")), document.text(document.object_get(event, "arg")), Float32(document.number(document.object_get(event, "time")))))
    var visemes = document.object_get(node, "visemes")
    if visemes >= 0 and document.nodes[visemes].kind == 5:
        for index in range(len(document.nodes[visemes].keys)):
            var curve = VisemeCurve(document.nodes[visemes].keys[index])
            var keys = document.nodes[visemes].children[index]
            for key in range(document.array_size(keys)):
                curve.times.append(Float32(document.number(document.at(document.at(keys, key), 0))))
                curve.values.append(Float32(document.number(document.at(document.at(keys, key), 1))))
            clip.visemes.append(curve^)
    return clip^


def parse_manifest(data: List[UInt8]) -> AssetManifest:
    var manifest = AssetManifest()
    var document = parse_json(data)
    if not document.valid or document.nodes[document.root].kind != 5:
        manifest.error = "the manifest is not valid JSON"
        return manifest^
    var root = document.root
    manifest.asset = document.text(document.object_get(root, "asset"))
    manifest.asset_id = document.text(document.object_get(root, "asset_id"))
    manifest.kind = document.text(document.object_get(root, "kind"))
    manifest.folder = document.text(document.object_get(root, "folder"))
    manifest.blender_version = document.text(document.object_get(root, "blender_version"))
    manifest.engine_format = document.integer(document.object_get(root, "engine_format"), 0)
    var skinning = document.text(document.object_get(root, "skinning"))
    manifest.skinning = 0 if skinning == "lbs" else (1 if skinning == "dqs" else (2 if skinning == "cor" else -1))
    var inputs = document.object_get(root, "inputs")
    manifest.source_hash = document.text(document.object_get(inputs, "source"))
    manifest.content_hash = document.text(document.object_get(inputs, "content"))
    manifest.scripts_hash = document.text(document.object_get(inputs, "scripts"))
    var stable = Fnv64()
    var text = String(unsafe_from_utf8=data.copy())
    if manifest.source_hash.byte_length() > 0:
        text = text.replace(manifest.source_hash, "")
    for byte in text.as_bytes():
        stable.add_byte(byte)
    manifest.stable_print = stable.value
    manifest.file = document.text(document.object_get(root, "file"))
    manifest.animation = document.text(document.object_get(root, "animation"))
    manifest.clip_prefix = document.text(document.object_get(root, "clip_prefix"))
    manifest.height = Float32(document.number(document.object_get(root, "height")))
    manifest.influences = document.integer(document.object_get(root, "influences"), 0)
    manifest.morphs_per_vertex = document.integer(document.object_get(root, "morphs_per_vertex"), 0)
    var morphs = document.object_get(root, "morphs")
    for index in range(document.array_size(morphs)):
        manifest.morphs.append(document.text(document.at(morphs, index)))
    var correctives = document.object_get(root, "correctives")
    for index in range(document.array_size(correctives)):
        var row = document.at(correctives, index)
        manifest.correctives.append(ManifestCorrective(
            document.text(document.object_get(row, "morph")), document.text(document.object_get(row, "bone")),
            Float32(document.number(document.object_get(row, "x"))), Float32(document.number(document.object_get(row, "z"))),
        ))
    var lods = document.object_get(root, "lods")
    for index in range(document.array_size(lods)):
        var lod = document.at(lods, index)
        manifest.lods.append(ManifestLod(document.text(document.object_get(lod, "file")), document.integer(document.object_get(lod, "triangles"), 0)))
    var bones = document.object_get(root, "skeleton")
    for index in range(document.array_size(bones)):
        var bone = document.at(bones, index)
        var entry = ManifestBone(document.text(document.object_get(bone, "name")), document.text(document.object_get(bone, "parent")), read_json_vec3(document, document.object_get(bone, "head")))
        var spring = document.object_get(bone, "spring")
        if spring >= 0:
            entry.spring = True
            entry.tail = read_json_vec3(document, document.object_get(spring, "tail"))
            entry.stiffness = Float32(document.number(document.object_get(spring, "stiffness"), 0.6))
            entry.drag = Float32(document.number(document.object_get(spring, "drag"), 0.4))
            entry.gravity = Float32(document.number(document.object_get(spring, "gravity"), 0))
            entry.radius = Float32(document.number(document.object_get(spring, "radius"), 0.05))
            entry.wind = Float32(document.number(document.object_get(spring, "wind"), 1))
        var helper = document.object_get(bone, "helper")
        if helper >= 0:
            var kind = document.text(document.object_get(helper, "kind"))
            entry.helper_kind = HELPER_TWIST if kind == "twist" else (HELPER_HALF if kind == "half" else 0)
            entry.helper_source = document.text(document.object_get(helper, "source"))
            entry.helper_amount = Float32(document.number(document.object_get(helper, "amount"), 0.5))
        manifest.skeleton.append(entry^)
    var clips = document.object_get(root, "clips")
    for index in range(document.array_size(clips)):
        manifest.clips.append(read_clip(document, document.at(clips, index)))
    var sockets = document.object_get(root, "sockets")
    for index in range(document.array_size(sockets)):
        var socket = document.at(sockets, index)
        manifest.sockets.append(ManifestSocket(document.text(document.object_get(socket, "name")), document.text(document.object_get(socket, "bone")), read_matrix(document, document.object_get(socket, "matrix"))))
    var volumes = document.object_get(root, "volumes")
    for index in range(document.array_size(volumes)):
        var volume = document.at(volumes, index)
        manifest.volumes.append(ManifestVolume(document.text(document.object_get(volume, "role")), document.text(document.object_get(volume, "bone")), document.text(document.object_get(volume, "shape")), read_json_vec3(document, document.object_get(volume, "size")), read_matrix(document, document.object_get(volume, "matrix"))))
    var collision = document.object_get(root, "collision")
    for index in range(document.array_size(collision)):
        var shape = document.at(collision, index)
        var points = List[Vec3]()
        var listed = document.object_get(shape, "points")
        for point in range(document.array_size(listed)):
            points.append(read_json_vec3(document, document.at(listed, point)))
        manifest.collision.append(ManifestShape(document.text(document.object_get(shape, "kind")), read_json_vec3(document, document.object_get(shape, "center")), read_json_vec3(document, document.object_get(shape, "size")), points^))
    var climb = document.object_get(root, "climb_path")
    for index in range(document.array_size(climb)):
        manifest.climb_path.append(read_json_vec3(document, document.at(climb, index)))
    manifest.valid = True
    return manifest^


def load_manifest(path: String) -> AssetManifest:
    if file_size(path) <= 0:
        var manifest = AssetManifest()
        manifest.error = "no manifest at " + path + "; run bash tools/run.sh assets"
        return manifest^
    return parse_manifest(read_bytes(path))
