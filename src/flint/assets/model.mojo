"""Mojo-owned geometry, material, scene, skin, and animation data."""
from std.math import sqrt
from std.sys import size_of
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4, trs
from flint.core.springs import decay_spring
from flint.core.ik import two_bone_ik, aim_rotation


struct NativeTexture(Copyable, Movable):
    var width: Int
    var height: Int
    var pixels: List[UInt32]

    def __init__(out self):
        self.width = 0
        self.height = 0
        self.pixels = List[UInt32]()


struct NativeNode(Copyable, Movable):
    var name: String
    var mesh: Int
    var mesh_indices: List[Int]
    var children: List[Int]
    var translation: Vec3
    var rotation: Quat
    var scale: Vec3
    var base_translation: Vec3
    var base_rotation: Quat
    var base_scale: Vec3
    var matrix: Mat4
    var has_matrix: Bool

    def __init__(out self):
        self.name = ""
        self.mesh = -1
        self.mesh_indices = List[Int]()
        self.children = List[Int]()
        self.translation = Vec3()
        self.rotation = Quat()
        self.scale = Vec3(1, 1, 1)
        self.base_translation = Vec3()
        self.base_rotation = Quat()
        self.base_scale = Vec3(1, 1, 1)
        self.matrix = Mat4()
        self.has_matrix = False

    def local_transform(self) -> Mat4:
        if self.has_matrix:
            return self.matrix.copy()
        return trs(self.translation, self.rotation, self.scale)


struct NativeSkin(Copyable, Movable):
    var joints: List[Int]
    var inverse_bind: List[Float32]

    def __init__(out self):
        self.joints = List[Int]()
        self.inverse_bind = List[Float32]()


struct NativeMesh(Copyable, Movable):
    var positions: List[Float32]
    var normals: List[Float32]
    var texcoords: List[Float32]
    var colors: List[UInt8]
    var ao: List[UInt8]
    var indices: List[Int]
    var joints: List[UInt16]
    var weights: List[Float32]
    var texture: Int
    var base_color: UInt32
    var emissive_color: Vec3
    var emissive_strength: Float32
    var skin: Int
    var alpha_mode: Int
    var double_sided: Bool
    # Morph targets: morph_count blocks of vertex_count() * 3 deltas each.
    var morph_count: Int
    var morph_positions: List[Float32]
    var morph_normals: List[Float32]
    # Package 3.4: each vertex's center of rotation in bind space (three
    # floats; anim/skinning.mojo `compute_cor`). Empty on unskinned meshes.
    var centers: List[Float32]

    def __init__(out self):
        self.positions = List[Float32]()
        self.normals = List[Float32]()
        self.texcoords = List[Float32]()
        self.colors = List[UInt8]()
        self.ao = List[UInt8]()
        self.indices = List[Int]()
        self.joints = List[UInt16]()
        self.weights = List[Float32]()
        self.texture = -1
        self.base_color = 0xFFFFFFFF
        self.emissive_color = Vec3(1, 1, 1)
        self.emissive_strength = 0
        self.skin = -1
        self.alpha_mode = 0
        self.double_sided = False
        self.morph_count = 0
        self.morph_positions = List[Float32]()
        self.morph_normals = List[Float32]()
        self.centers = List[Float32]()

    def vertex_count(self) -> Int:
        return len(self.positions) // 3

    def triangle_count(self) -> Int:
        if len(self.indices) > 0:
            return len(self.indices) // 3
        return self.vertex_count() // 3


struct NativeChannel(Copyable, Movable):
    var node: Int
    var path: Int
    var times: List[Float32]
    var values: List[Float32]
    var step: Bool

    def __init__(out self):
        self.node = -1
        self.path = 0
        self.times = List[Float32]()
        self.values = List[Float32]()
        self.step = False


struct NativeAnimation(Copyable, Movable):
    var name: String
    var channels: List[NativeChannel]
    var duration: Float32

    def __init__(out self):
        self.name = ""
        self.channels = List[NativeChannel]()
        self.duration = 0

    def frame_count(self) -> Int:
        return max(1, Int(self.duration * 30.0) + 1)


comptime DEFAULT_BLEND_HALFLIFE: Float32 = 0.08


struct InertialState(Copyable, Movable):
    """Per-model pose memory used to blend away clip changes."""
    var clip: Int
    var has_pose: Bool
    var out_t: List[Vec3]
    var out_r: List[Quat]
    var out_vel_t: List[Vec3]
    var out_vel_r: List[Vec3]
    var off_t: List[Vec3]
    var vel_t: List[Vec3]
    var off_r: List[Vec3]
    var vel_r: List[Vec3]
    var next_t: List[Vec3]
    var next_r: List[Quat]

    def __init__(out self):
        self.clip = -1
        self.has_pose = False
        self.out_t = List[Vec3]()
        self.out_r = List[Quat]()
        self.out_vel_t = List[Vec3]()
        self.out_vel_r = List[Vec3]()
        self.off_t = List[Vec3]()
        self.vel_t = List[Vec3]()
        self.off_r = List[Vec3]()
        self.vel_r = List[Vec3]()
        self.next_t = List[Vec3]()
        self.next_r = List[Quat]()

    def resize(mut self, count: Int):
        self.out_t = List[Vec3](length=count, fill=Vec3())
        self.out_r = List[Quat](length=count, fill=Quat())
        self.out_vel_t = List[Vec3](length=count, fill=Vec3())
        self.out_vel_r = List[Vec3](length=count, fill=Vec3())
        self.off_t = List[Vec3](length=count, fill=Vec3())
        self.vel_t = List[Vec3](length=count, fill=Vec3())
        self.off_r = List[Vec3](length=count, fill=Vec3())
        self.vel_r = List[Vec3](length=count, fill=Vec3())
        self.next_t = List[Vec3](length=count, fill=Vec3())
        self.next_r = List[Quat](length=count, fill=Quat())
        self.has_pose = False
        self.clip = -1


struct NativeModel(Copyable, Movable):
    var meshes: List[NativeMesh]
    var textures: List[NativeTexture]
    var nodes: List[NativeNode]
    var roots: List[Int]
    var skins: List[NativeSkin]
    var animations: List[NativeAnimation]
    var active_animation: Int
    var active_time: Float32
    var bounds_center: Vec3
    var bounds_radius: Float32
    var bounds_valid: Bool
    var cached_transforms: List[Mat4]
    var transforms_revision: Int
    var cached_revision: Int
    var anim: InertialState
    var parents: List[Int]
    var max_influences: Int
    var morph_names: List[String]
    # Package 3.4: how far each morph target is applied, by morph_names
    # (empty: none). Presentation sets it; skinning applies it.
    var morph_weights: List[Float32]
    # Package 3.4: how its skinned meshes deform (anim/skinning.mojo SKIN_*), from
    # the asset's manifest; -1 when nothing said, which is linear blend.
    var skin_method: Int

    def __init__(out self):
        self.meshes = List[NativeMesh]()
        self.textures = List[NativeTexture]()
        self.nodes = List[NativeNode]()
        self.roots = List[Int]()
        self.skins = List[NativeSkin]()
        self.animations = List[NativeAnimation]()
        self.active_animation = -1
        self.active_time = 0
        self.bounds_center = Vec3()
        self.bounds_radius = 0
        self.bounds_valid = False
        self.cached_transforms = List[Mat4]()
        self.transforms_revision = 0
        self.cached_revision = -1
        self.anim = InertialState()
        self.parents = List[Int]()
        self.max_influences = 0
        self.morph_names = List[String]()
        self.morph_weights = List[Float32]()
        self.skin_method = -1

    def set_morph_weight(mut self, morph: Int, weight: Float32):
        """How far morph target `morph` (an index into morph_names) is applied."""
        if morph < 0 or morph >= len(self.morph_names):
            return
        if len(self.morph_weights) != len(self.morph_names):
            self.morph_weights = List[Float32](length=len(self.morph_names), fill=0)
        self.morph_weights[morph] = weight

    def bytes(self) -> Int:
        """Heap bytes this model holds (list capacities), for its memory tag."""
        var total = self.meshes.capacity() * size_of[NativeMesh]() + self.textures.capacity() * size_of[NativeTexture]()
        for mesh in self.meshes:
            total += (mesh.positions.capacity() + mesh.normals.capacity() + mesh.texcoords.capacity() + mesh.weights.capacity()) * 4
            total += mesh.colors.capacity() + mesh.ao.capacity() + mesh.indices.capacity() * 8 + mesh.joints.capacity() * 2
            total += (mesh.morph_positions.capacity() + mesh.morph_normals.capacity() + mesh.centers.capacity()) * 4
        for texture in self.textures:
            total += texture.pixels.capacity() * 4
        total += self.nodes.capacity() * size_of[NativeNode]()
        for node in self.nodes:
            total += (node.mesh_indices.capacity() + node.children.capacity()) * 8
        total += self.skins.capacity() * size_of[NativeSkin]()
        for skin in self.skins:
            total += skin.joints.capacity() * 8 + skin.inverse_bind.capacity() * 4
        total += self.animations.capacity() * size_of[NativeAnimation]()
        for animation in self.animations:
            total += animation.channels.capacity() * size_of[NativeChannel]()
            for channel in animation.channels:
                total += (channel.times.capacity() + channel.values.capacity()) * 4
        total += self.cached_transforms.capacity() * size_of[Mat4]() + (self.roots.capacity() + self.parents.capacity()) * 8
        return total

    def loaded(self) -> Bool:
        """A clip file has no meshes and is still a good file."""
        return len(self.meshes) > 0 or len(self.animations) > 0

    def max_vertex_morphs(self) -> Int:
        """The most morph targets moving any one vertex."""
        var most = 0
        for mesh in self.meshes:
            var vertices = mesh.vertex_count()
            if mesh.morph_count == 0 or len(mesh.morph_positions) != mesh.morph_count * vertices * 3:
                continue
            for vertex in range(vertices):
                var count = 0
                for morph in range(mesh.morph_count):
                    var offset = (morph * vertices + vertex) * 3
                    if mesh.morph_positions[offset] != 0 or mesh.morph_positions[offset + 1] != 0 or mesh.morph_positions[offset + 2] != 0:
                        count += 1
                most = max(most, count)
        return most

    def animated_nodes(self) -> List[Int]:
        """Every node some clip moves, each once, in node order."""
        var moved = List[Bool](length=len(self.nodes), fill=False)
        for animation in self.animations:
            for channel in animation.channels:
                if channel.node >= 0 and channel.node < len(moved):
                    moved[channel.node] = True
        var nodes = List[Int]()
        for index in range(len(moved)):
            if moved[index]:
                nodes.append(index)
        return nodes^

    def channel_rests(self, channel: NativeChannel, node: Int) -> Bool:
        """True when every key of the channel is the node's rest value."""
        comptime CLOSE: Float32 = 0.000001
        var components = 4 if channel.path == 1 else 3
        var keys = len(channel.times)
        if keys == 0 or len(channel.values) < keys * components:
            return False
        ref rest = self.nodes[node]
        for key in range(keys):
            var at = key * components
            if channel.path == 0:
                if abs(channel.values[at] - rest.base_translation.x) > CLOSE or abs(channel.values[at + 1] - rest.base_translation.y) > CLOSE or abs(channel.values[at + 2] - rest.base_translation.z) > CLOSE:
                    return False
            elif channel.path == 1:
                # q and -q are one rotation.
                var sign = Float32(1) if channel.values[at + 3] * rest.base_rotation.w + channel.values[at] * rest.base_rotation.x + channel.values[at + 1] * rest.base_rotation.y + channel.values[at + 2] * rest.base_rotation.z >= 0 else Float32(-1)
                if abs(channel.values[at] * sign - rest.base_rotation.x) > CLOSE or abs(channel.values[at + 1] * sign - rest.base_rotation.y) > CLOSE or abs(channel.values[at + 2] * sign - rest.base_rotation.z) > CLOSE or abs(channel.values[at + 3] * sign - rest.base_rotation.w) > CLOSE:
                    return False
            else:
                if abs(channel.values[at] - rest.base_scale.x) > CLOSE or abs(channel.values[at + 1] - rest.base_scale.y) > CLOSE or abs(channel.values[at + 2] - rest.base_scale.z) > CLOSE:
                    return False
        return True

    def bind_clips(mut self, source: NativeModel) -> Bool:
        """Takes source's clips, matched to this model's nodes by bone name, so
        one clip file drives any mesh on the same rig. Leaves the model as it
        was and returns False if a bone a clip moves is missing here."""
        var remap = List[Int](length=len(source.nodes), fill=-1)
        for node in source.animated_nodes():
            remap[node] = self.find_node(source.nodes[node].name)
            if remap[node] < 0:
                return False
        # A channel that holds this model's own rest value on every key is
        # left out: sampling starts from the rest pose, so the pose is the
        # same, and each bound mesh carries only what moves (package 3.6: a
        # glTF export keys translation, rotation and scale on every bone, and
        # the clip file is bound to every level of detail of every form).
        var bound = List[NativeAnimation]()
        for clip in range(len(source.animations)):
            var animation = NativeAnimation()
            animation.name = source.animations[clip].name
            animation.duration = source.animations[clip].duration
            for channel in source.animations[clip].channels:
                var node = remap[channel.node] if channel.node >= 0 and channel.node < len(remap) else -1
                if node >= 0 and self.channel_rests(channel, node):
                    continue
                var kept = channel.copy()
                kept.node = node
                animation.channels.append(kept^)
            bound.append(animation^)
        self.animations = bound^
        self.anim = InertialState()
        return True

    def keep_clips(mut self, names: List[String]) -> Int:
        """Keeps only the named clips, in their order here, and of those only
        the channels that leave the rest pose (as `bind_clips` does for a
        mesh). For a clip file that is sampled, never bound (package 3.23:
        Maroon_Moves holds 107 moves, the acting plays about half, and a
        glTF export keys every bone's translation, rotation and scale).
        Returns how many clips are left."""
        var kept = List[NativeAnimation]()
        for clip in range(len(self.animations)):
            var wanted = False
            for name in names:
                if self.animations[clip].name == name:
                    wanted = True
                    break
            if not wanted:
                continue
            var animation = NativeAnimation()
            animation.name = self.animations[clip].name
            animation.duration = self.animations[clip].duration
            for channel in self.animations[clip].channels:
                if channel.node >= 0 and channel.node < len(self.nodes) and self.channel_rests(channel, channel.node):
                    continue
                animation.channels.append(channel.copy())
            kept.append(animation^)
        self.animations = kept^
        self.anim = InertialState()
        return len(self.animations)

    def attachment(mut self, bone: Int, matrix: List[Float32]) -> Vec3:
        """Model-space position of a socket or volume on a bone, in the current
        pose. matrix is the manifest's: 16 floats, column major, in the bone's frame."""
        if bone < 0 or bone >= len(self.nodes) or len(matrix) != 16:
            return Vec3()
        var world = self.global_transforms()[bone].copy()
        return world.transform_point(Vec3(matrix[12], matrix[13], matrix[14]))

    def compute_bounds(mut self):
        var minimum = Vec3(1000000000, 1000000000, 1000000000)
        var maximum = Vec3(-1000000000, -1000000000, -1000000000)
        var found = False
        for mesh_index in range(len(self.meshes)):
            for vertex in range(self.meshes[mesh_index].vertex_count()):
                var offset = vertex * 3
                var point = Vec3(self.meshes[mesh_index].positions[offset], self.meshes[mesh_index].positions[offset + 1], self.meshes[mesh_index].positions[offset + 2])
                minimum.x = min(minimum.x, point.x)
                minimum.y = min(minimum.y, point.y)
                minimum.z = min(minimum.z, point.z)
                maximum.x = max(maximum.x, point.x)
                maximum.y = max(maximum.y, point.y)
                maximum.z = max(maximum.z, point.z)
                found = True
        if not found:
            self.bounds_center = Vec3()
            self.bounds_radius = 0
            self.bounds_valid = False
            return
        self.bounds_center = (minimum + maximum) * 0.5
        var radius_squared = Float32(0)
        for mesh_index in range(len(self.meshes)):
            for vertex in range(self.meshes[mesh_index].vertex_count()):
                var offset = vertex * 3
                var point = Vec3(self.meshes[mesh_index].positions[offset], self.meshes[mesh_index].positions[offset + 1], self.meshes[mesh_index].positions[offset + 2])
                radius_squared = max(radius_squared, (point - self.bounds_center).length_squared())
        self.bounds_radius = sqrt(radius_squared) + 1.0
        self.bounds_valid = True

    def triangle_count(self) -> Int:
        var count = 0
        for mesh in self.meshes:
            count += mesh.triangle_count()
        return count

    def animation_frame_count(self, animation: Int) -> Int:
        if animation < 0 or animation >= len(self.animations):
            return 0
        return self.animations[animation].frame_count()

    def reorder_animations(mut self, names: List[String]) -> Bool:
        """Puts clips in the given order, matched by name, so the game's clip
        numbers hold whatever order an exporter wrote them in. Leaves the
        model untouched and returns False if any name is missing."""
        var order = List[Int]()
        for name in names:
            var found = -1
            for index in range(len(self.animations)):
                if self.animations[index].name == name:
                    found = index
                    break
            if found < 0:
                return False
            order.append(found)
        var reordered = List[NativeAnimation]()
        for index in order:
            reordered.append(self.animations[index].copy())
        for index in range(len(self.animations)):
            var listed = False
            for used in order:
                if used == index:
                    listed = True
            if not listed:
                reordered.append(self.animations[index].copy())
        self.animations = reordered^
        self.anim = InertialState()
        return True

    def animation_name(self, animation: Int) -> String:
        if animation < 0 or animation >= len(self.animations):
            return ""
        return self.animations[animation].name

    def animation_duration(self, animation: Int) -> Float32:
        if animation < 0 or animation >= len(self.animations):
            return 0
        return self.animations[animation].duration

    def update_animation(mut self, animation: Int, frame: Int):
        """Legacy whole-frame playback; kept for tools that step by frame."""
        if animation < 0 or animation >= len(self.animations):
            return
        self.sample_clip(animation, Float32(frame) / 30.0)
        self.anim.has_pose = False

    def animate(mut self, animation: Int, time: Float32, dt: Float32, halflife: Float32 = DEFAULT_BLEND_HALFLIFE):
        """Samples a clip at a continuous time, and hides clip changes with
        inertialization: the jump between the old pose and the new clip is
        kept as an offset that a critically damped spring decays to zero
        (Bollo, GDC 2018; Holden's spring formulation)."""
        if animation < 0 or animation >= len(self.animations):
            return
        var count = len(self.nodes)
        if len(self.anim.out_t) != count:
            self.anim.resize(count)
        var switching = self.anim.has_pose and animation != self.anim.clip and halflife > 0 and dt > 0
        if switching:
            # Sample one step ahead to know how fast the new clip moves.
            self.sample_clip(animation, time + dt)
            for index in range(count):
                self.anim.next_t[index] = self.nodes[index].translation
                self.anim.next_r[index] = self.nodes[index].rotation
        self.sample_clip(animation, time)
        if switching:
            var inverse_dt = 1.0 / dt
            for index in range(count):
                var src_t = self.nodes[index].translation
                var src_r = self.nodes[index].rotation
                var src_vel_t = (self.anim.next_t[index] - src_t) * inverse_dt
                var src_vel_r = self.anim.next_r[index].multiply(src_r.conjugate()).to_scaled_angle() * inverse_dt
                self.anim.off_t[index] = self.anim.out_t[index] - src_t
                self.anim.vel_t[index] = self.anim.out_vel_t[index] - src_vel_t
                self.anim.off_r[index] = self.anim.out_r[index].multiply(src_r.conjugate()).to_scaled_angle()
                self.anim.vel_r[index] = self.anim.out_vel_r[index] - src_vel_r
        elif not self.anim.has_pose:
            for index in range(count):
                self.anim.off_t[index] = Vec3()
                self.anim.vel_t[index] = Vec3()
                self.anim.off_r[index] = Vec3()
                self.anim.vel_r[index] = Vec3()
        for index in range(count):
            if dt > 0:
                decay_spring(self.anim.off_t[index], self.anim.vel_t[index], halflife, dt)
                decay_spring(self.anim.off_r[index], self.anim.vel_r[index], halflife, dt)
            var final_t = self.nodes[index].translation + self.anim.off_t[index]
            var final_r = Quat.from_scaled_angle(self.anim.off_r[index]).multiply(self.nodes[index].rotation).normalized()
            if self.anim.has_pose and dt > 0:
                self.anim.out_vel_t[index] = (final_t - self.anim.out_t[index]) * (1.0 / dt)
                self.anim.out_vel_r[index] = final_r.multiply(self.anim.out_r[index].conjugate()).to_scaled_angle() * (1.0 / dt)
            else:
                self.anim.out_vel_t[index] = Vec3()
                self.anim.out_vel_r[index] = Vec3()
            self.anim.out_t[index] = final_t
            self.anim.out_r[index] = final_r
            self.nodes[index].translation = final_t
            self.nodes[index].rotation = final_r
        self.anim.clip = animation
        self.anim.has_pose = True

    def find_node(self, name: String) -> Int:
        for index in range(len(self.nodes)):
            if self.nodes[index].name == name:
                return index
        return -1

    def ensure_parents(mut self):
        if len(self.parents) == len(self.nodes):
            return
        self.parents = List[Int](length=len(self.nodes), fill=-1)
        for index in range(len(self.nodes)):
            for child in self.nodes[index].children:
                if child >= 0 and child < len(self.nodes):
                    self.parents[child] = index

    def global_rotation(mut self, node: Int) -> Quat:
        self.ensure_parents()
        var rotation = Quat()
        var current = node
        var guard = 0
        while current >= 0 and guard < 128:
            rotation = self.nodes[current].rotation.multiply(rotation)
            current = self.parents[current]
            guard += 1
        return rotation.normalized()

    def global_position(mut self, node: Int) -> Vec3:
        """Model-space joint position (bones are assumed unscaled)."""
        self.ensure_parents()
        var position = Vec3()
        var current = node
        var guard = 0
        while current >= 0 and guard < 128:
            position = self.nodes[current].translation + self.nodes[current].rotation.rotate(position)
            current = self.parents[current]
            guard += 1
        return position

    def solve_two_bone(mut self, upper: Int, middle: Int, end: Int, target: Vec3, pole: Vec3, weight: Float32 = 1.0):
        """Bends upper and middle so end reaches target (model space)."""
        if upper < 0 or middle < 0 or end < 0 or weight <= 0:
            return
        var a = self.global_position(upper)
        var b = self.global_position(middle)
        var c = self.global_position(end)
        var goal = c + (target - c) * min(Float32(1), weight)
        var result = two_bone_ik(a, b, c, goal, self.global_rotation(upper), self.global_rotation(middle), pole)
        self.nodes[upper].rotation = self.nodes[upper].rotation.multiply(result.upper).normalized()
        self.nodes[middle].rotation = self.nodes[middle].rotation.multiply(result.middle).normalized()
        var swing = aim_rotation(a, self.global_position(end), goal, self.global_rotation(upper))
        self.nodes[upper].rotation = self.nodes[upper].rotation.multiply(swing).normalized()
        self.transforms_revision += 1

    def offset_node(mut self, node: Int, model_delta: Vec3):
        """Moves a joint by a model-space offset (e.g. lowering the pelvis)."""
        if node < 0:
            return
        self.ensure_parents()
        var parent = self.parents[node]
        var local = model_delta
        if parent >= 0:
            local = self.global_rotation(parent).conjugate().rotate(model_delta)
        self.nodes[node].translation = self.nodes[node].translation + local
        self.transforms_revision += 1

    def rotate_node_model(mut self, node: Int, rotation: Quat):
        """Turns a joint about its own pivot by a model-space rotation (its
        children follow): global' = rotation * global (package 3.3's
        look-at, lean, slope and spring bones)."""
        if node < 0 or node >= len(self.nodes):
            return
        self.ensure_parents()
        var parent = self.parents[node]
        var parent_rotation = self.global_rotation(parent) if parent >= 0 else Quat()
        var local_delta = parent_rotation.conjugate().multiply(rotation).multiply(parent_rotation)
        self.nodes[node].rotation = local_delta.multiply(self.nodes[node].rotation).normalized()
        self.transforms_revision += 1

    def set_node_basis(mut self, node: Int, translation: Vec3, rotation: Quat, scale: Vec3):
        """Poses a joint relative to its rest, as Blender's pose bone
        matrix_basis does: local = rest * basis. The live link uses this."""
        if node < 0 or node >= len(self.nodes):
            return
        var base_t = self.nodes[node].base_translation
        var base_r = self.nodes[node].base_rotation
        var base_s = self.nodes[node].base_scale
        self.nodes[node].translation = base_t + base_r.rotate(Vec3(base_s.x * translation.x, base_s.y * translation.y, base_s.z * translation.z))
        self.nodes[node].rotation = base_r.multiply(rotation).normalized()
        self.nodes[node].scale = Vec3(base_s.x * scale.x, base_s.y * scale.y, base_s.z * scale.z)
        self.transforms_revision += 1

    def blend_offset(self) -> Float32:
        """Largest remaining translation offset, for tests and debugging."""
        var largest = Float32(0)
        for index in range(len(self.anim.off_t)):
            largest = max(largest, self.anim.off_t[index].length())
        return largest

    def sample_clip(mut self, animation: Int, time_seconds: Float32):
        self.active_animation = animation
        self.active_time = time_seconds
        self.transforms_revision += 1
        for index in range(len(self.nodes)):
            self.nodes[index].translation = self.nodes[index].base_translation
            self.nodes[index].rotation = self.nodes[index].base_rotation
            self.nodes[index].scale = self.nodes[index].base_scale
        ref clip = self.animations[animation]
        var time = max(Float32(0), min(self.active_time, clip.duration))
        for channel in clip.channels:
            if channel.node < 0 or channel.node >= len(self.nodes) or len(channel.times) == 0:
                continue
            var lower_bound = 0
            var upper_bound = len(channel.times) - 1
            while lower_bound + 1 < upper_bound:
                var middle = (lower_bound + upper_bound) // 2
                if channel.times[middle] < time:
                    lower_bound = middle
                else:
                    upper_bound = middle
            var upper = lower_bound
            var lower = max(0, min(len(channel.times) - 1, upper))
            var next = min(len(channel.times) - 1, lower + 1)
            var amount = Float32(0)
            if not channel.step and next != lower and channel.times[next] > channel.times[lower]:
                amount = max(Float32(0), min(Float32(1), (time - channel.times[lower]) / (channel.times[next] - channel.times[lower])))
            var components = 4 if channel.path == 1 else 3
            var first = lower * components
            var second = next * components
            if first + components > len(channel.values) or second + components > len(channel.values):
                continue
            if channel.path == 0:
                self.nodes[channel.node].translation = Vec3(
                    channel.values[first] + (channel.values[second] - channel.values[first]) * amount,
                    channel.values[first + 1] + (channel.values[second + 1] - channel.values[first + 1]) * amount,
                    channel.values[first + 2] + (channel.values[second + 2] - channel.values[first + 2]) * amount,
                )
            elif channel.path == 1:
                var a = Quat(channel.values[first], channel.values[first + 1], channel.values[first + 2], channel.values[first + 3]).normalized()
                var b = Quat(channel.values[second], channel.values[second + 1], channel.values[second + 2], channel.values[second + 3]).normalized()
                self.nodes[channel.node].rotation = a.slerp(b, amount)
            else:
                self.nodes[channel.node].scale = Vec3(
                    channel.values[first] + (channel.values[second] - channel.values[first]) * amount,
                    channel.values[first + 1] + (channel.values[second + 1] - channel.values[first + 1]) * amount,
                    channel.values[first + 2] + (channel.values[second + 2] - channel.values[first + 2]) * amount,
                )

    def compute_node(mut self, node: Int, parent: Mat4, mut output: List[Mat4], mut visited: List[Bool]):
        if node < 0 or node >= len(self.nodes) or visited[node]:
            return
        visited[node] = True
        var transform = parent.multiply(self.nodes[node].local_transform())
        output[node] = transform.copy()
        var children = self.nodes[node].children.copy()
        for child in children:
            self.compute_node(child, transform, output, visited)

    def global_transforms(mut self) -> List[Mat4]:
        if self.cached_revision == self.transforms_revision and len(self.cached_transforms) == len(self.nodes):
            return self.cached_transforms.copy()
        var output = List[Mat4](length=len(self.nodes), fill=Mat4())
        var visited = List[Bool](length=len(self.nodes), fill=False)
        var identity = Mat4()
        for root in self.roots:
            self.compute_node(root, identity, output, visited)
        for index in range(len(self.nodes)):
            if not visited[index]:
                self.compute_node(index, identity, output, visited)
        self.cached_transforms = output.copy()
        self.cached_revision = self.transforms_revision
        return output^

    def mesh_bounds(self, mesh: Int) -> List[Float32]:
        if mesh < 0 or mesh >= len(self.meshes) or self.meshes[mesh].vertex_count() == 0:
            return List[Float32]()
        var result = List[Float32](length=6, fill=0)
        result[0] = 1000000000.0
        result[1] = 1000000000.0
        result[2] = 1000000000.0
        result[3] = -1000000000.0
        result[4] = -1000000000.0
        result[5] = -1000000000.0
        for index in range(self.meshes[mesh].vertex_count()):
            for axis in range(3):
                var value = self.meshes[mesh].positions[index * 3 + axis]
                result[axis] = min(result[axis], value)
                result[axis + 3] = max(result[axis + 3], value)
        return result^
