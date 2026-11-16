"""Import checks: the engine refuses a bad asset and says what is wrong with it.

Each check returns one message per problem, each naming the asset. An empty
list means the asset may load. The game, the hot reload and the pipeline tool
all go through these, so a file refused in one is refused in all of them.
"""
from flint.core.math import Vec3
from flint.assets.manifest import AssetManifest, ENGINE_FORMAT
from flint.assets.model import NativeModel
from flint.assets.json import JsonDocument
from flint.assets.clip_names import known_event_kind

comptime MAX_BONE_INFLUENCES: Int = 4
comptime MAX_VERTEX_MORPHS: Int = 8
# Rest bones of a mesh and of the shared clips must agree this closely (metres).
comptime REST_TOLERANCE: Float32 = 0.01
# Measured height against the registry's. LOD0 is what was authored; decimation
# moves the extremes of the others, up to 28% on the rail track.
comptime HEIGHT_TOLERANCE: Float32 = 0.05
comptime LOD_HEIGHT_TOLERANCE: Float32 = 0.35
comptime CLIP_LENGTH_TOLERANCE: Float32 = 0.05


def refusal(asset: String, problem: String) -> String:
    return "asset " + asset + ": " + problem


def bind_extent(mut model: NativeModel) -> List[Float32]:
    """Bind-pose minimum and maximum on each axis, game axes (Y up): x0 y0 z0 x1 y1 z1.
    Call before the model animates: node transforms count, and skins use rest."""
    var transforms = model.global_transforms()
    var result = List[Float32](length=6, fill=0)
    for axis in range(3):
        result[axis] = 1e9
        result[axis + 3] = -1e9
    for node_index in range(len(model.nodes)):
        for mesh_index in model.nodes[node_index].mesh_indices:
            if mesh_index < 0 or mesh_index >= len(model.meshes):
                continue
            var skinned = model.meshes[mesh_index].skin >= 0
            for vertex in range(model.meshes[mesh_index].vertex_count()):
                var offset = vertex * 3
                var point = Vec3(model.meshes[mesh_index].positions[offset], model.meshes[mesh_index].positions[offset + 1], model.meshes[mesh_index].positions[offset + 2])
                if not skinned:
                    point = transforms[node_index].transform_point(point)
                result[0] = min(result[0], point.x)
                result[1] = min(result[1], point.y)
                result[2] = min(result[2], point.z)
                result[3] = max(result[3], point.x)
                result[4] = max(result[4], point.y)
                result[5] = max(result[5], point.z)
    return result^


def parent_name(model: NativeModel, node: Int) -> String:
    for index in range(len(model.nodes)):
        for child in model.nodes[index].children:
            if child == node:
                return model.nodes[index].name
    return ""


def check_format(manifest: AssetManifest, name: String) -> List[String]:
    var problems = List[String]()
    if not manifest.valid:
        problems.append(refusal(name, manifest.error))
        return problems^
    if manifest.engine_format != ENGINE_FORMAT:
        problems.append(refusal(name, "engine format " + String(manifest.engine_format) + ", this engine reads " + String(ENGINE_FORMAT) + "; re-export asset " + name))
    if manifest.asset != name:
        problems.append(refusal(name, "the manifest belongs to " + manifest.asset))
    if manifest.asset_id.byte_length() == 0:
        problems.append(refusal(name, "the manifest has no asset id"))
    return problems^


def check_placed_objects(manifest: AssetManifest, model: NativeModel, name: String, mut problems: List[String]):
    """Sockets and volumes must sit on bones the model really has, and every
    shape placed in Blender must be one the engine can use."""
    for socket in manifest.sockets:
        if model.find_node(socket.bone) < 0:
            problems.append(refusal(name, "socket " + socket.name + " is on bone " + socket.bone + ", which the model does not have"))
        if len(socket.matrix) != 16:
            problems.append(refusal(name, "socket " + socket.name + " has no transform"))
    for volume in manifest.volumes:
        if model.find_node(volume.bone) < 0:
            problems.append(refusal(name, volume.role + " volume is on bone " + volume.bone + ", which the model does not have"))
        if (volume.role != "hit" and volume.role != "hurt") or (volume.shape != "box" and volume.shape != "capsule" and volume.shape != "sphere"):
            problems.append(refusal(name, "volume on " + volume.bone + " is a " + volume.role + " " + volume.shape + ", not a hit or hurt box, capsule or sphere"))
        if volume.size.x <= 0 or volume.size.y <= 0 or volume.size.z <= 0:
            problems.append(refusal(name, volume.role + " volume on " + volume.bone + " has no size"))
    for shape in manifest.collision:
        if shape.kind != "box" and shape.kind != "sphere" and shape.kind != "capsule" and shape.kind != "convex":
            problems.append(refusal(name, "collision shape " + shape.kind + " is not box, sphere, capsule or convex"))
        if shape.kind == "convex" and len(shape.points) < 4:
            problems.append(refusal(name, "a convex collision shape needs at least 4 points"))
    if len(manifest.climb_path) == 1:
        problems.append(refusal(name, "climb_path has one point; it needs a start and an end"))


def check_mesh(manifest: AssetManifest, mut model: NativeModel, name: String, lod: Int) -> List[String]:
    """One level of detail against its manifest."""
    var problems = check_format(manifest, name)
    if not manifest.valid:
        return problems^
    var label = "LOD" + String(lod)
    if len(model.meshes) == 0 or model.triangle_count() == 0:
        problems.append(refusal(name, label + " has no triangles"))
        return problems^
    if lod < len(manifest.lods) and model.triangle_count() != manifest.lods[lod].triangles:
        problems.append(refusal(name, label + " has " + String(model.triangle_count()) + " triangles, the manifest says " + String(manifest.lods[lod].triangles)))
    if model.max_influences > MAX_BONE_INFLUENCES:
        problems.append(refusal(name, label + " has " + String(model.max_influences) + " bone influences on a vertex, the limit is " + String(MAX_BONE_INFLUENCES)))
    var morphs = model.max_vertex_morphs()
    if morphs > MAX_VERTEX_MORPHS:
        problems.append(refusal(name, label + " has " + String(morphs) + " morph targets on a vertex, the limit is " + String(MAX_VERTEX_MORPHS)))
    var extent = bind_extent(model)
    var size = Vec3(extent[3] - extent[0], extent[4] - extent[1], extent[5] - extent[2])
    var tolerance = manifest.height * (HEIGHT_TOLERANCE if lod == 0 else LOD_HEIGHT_TOLERANCE)
    if manifest.height > 0 and abs(size.y - manifest.height) > tolerance:
        if abs(size.z - manifest.height) <= tolerance or abs(size.x - manifest.height) <= tolerance:
            problems.append(refusal(name, "wrong axis: " + label + " is " + String(manifest.height) + " tall along a horizontal axis, up must be Blender's Z"))
        else:
            problems.append(refusal(name, "wrong scale: " + label + " is " + String(size.y) + " tall, it should be " + String(manifest.height)))
    check_placed_objects(manifest, model, name, problems)
    if lod == 0:
        # Package 3.5: a corrective needs its bone and its morph target (LOD0 carries the morphs).
        for corrective in manifest.correctives:
            if model.find_node(corrective.bone) < 0:
                problems.append(refusal(name, "corrective " + corrective.morph + " reads bone " + corrective.bone + ", which the model does not have"))
            var found = False
            for morph in model.morph_names:
                if morph == corrective.morph:
                    found = True
            if not found:
                problems.append(refusal(name, "corrective " + corrective.morph + " has no morph target of that name in " + label))
    return problems^


def check_clips(manifest: AssetManifest, model: NativeModel, name: String, needed: List[String]) -> List[String]:
    """A clip file (or a mesh that owns its clips) against the game's clip list."""
    var problems = check_format(manifest, name)
    if not manifest.valid:
        return problems^
    for wanted in needed:
        var index = manifest.find_clip(wanted)
        var in_file = False
        for animation in model.animations:
            if animation.name == manifest.clip_prefix + wanted:
                in_file = True
                if index >= 0 and abs(animation.duration - manifest.clips[index].length) > CLIP_LENGTH_TOLERANCE:
                    problems.append(refusal(name, "clip " + wanted + " is " + String(animation.duration) + " s in the file, the manifest says " + String(manifest.clips[index].length)))
        if index < 0 or not in_file:
            problems.append(refusal(name, "clip " + wanted + " the game needs is missing"))
    for clip in manifest.clips:
        for event in clip.events:
            if not known_event_kind(event.kind):
                problems.append(refusal(name, "clip " + clip.name + " has event kind " + event.kind + ", which the game does not know"))
            if event.time < 0 or event.time > clip.length + CLIP_LENGTH_TOLERANCE:
                problems.append(refusal(name, "clip " + clip.name + " has event " + event.kind + " outside the clip"))
    return problems^


def check_skeleton_match(clips: NativeModel, model: NativeModel, name: String, clip_asset: String) -> List[String]:
    """A mesh may bind to shared clips only if it is the same rig: same bones,
    same parents, same rest offsets."""
    var problems = List[String]()
    for channel_node in clips.animated_nodes():
        ref bone = clips.nodes[channel_node].name
        var found = model.find_node(bone)
        if found < 0:
            problems.append(refusal(name, "skeleton does not match the rig: bone " + bone + " of " + clip_asset + " is missing"))
            continue
        if parent_name(clips, channel_node) != parent_name(model, found):
            problems.append(refusal(name, "skeleton does not match the rig: bone " + bone + " has parent " + parent_name(model, found) + ", " + clip_asset + " has " + parent_name(clips, channel_node)))
        var offset = clips.nodes[channel_node].base_translation - model.nodes[found].base_translation
        if offset.length() > REST_TOLERANCE:
            problems.append(refusal(name, "skeleton does not match the rig: bone " + bone + " rests " + String(offset.length()) + " from where " + clip_asset + " has it"))
    return problems^


def check_skeleton_template(model: NativeModel, template: JsonDocument, template_name: String, name: String) -> List[String]:
    """Every bone of the skeleton template (tools/blender/skeletons/<name>.json)
    is in the exported file under the same parent. Root bones hang off the
    armature node, which no template names."""
    var problems = List[String]()
    if not template.valid:
        problems.append(refusal(name, "skeleton template " + template_name + " is missing or not JSON"))
        return problems^
    var bones = template.object_get(template.root, "bones")
    for index in range(template.array_size(bones)):
        var bone = template.at(bones, index)
        var bone_name = template.text(template.object_get(bone, "name"))
        var parent = template.text(template.object_get(bone, "parent"))
        var node = model.find_node(bone_name)
        if node < 0:
            problems.append(refusal(name, "bone " + bone_name + " of the " + template_name + " skeleton template is missing"))
            continue
        if parent.byte_length() > 0 and parent_name(model, node) != parent:
            problems.append(refusal(name, "bone " + bone_name + " has parent " + parent_name(model, node) + ", the " + template_name + " skeleton template says " + parent))
    return problems^
