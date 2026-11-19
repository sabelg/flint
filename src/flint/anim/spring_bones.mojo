"""Spring bones (package 3.3 / v4): secondary motion chains with root-space
velocity/displacement inertia clamping and intra-substep parent-to-child propagation.

Upgrades in Flint v4:
- `max_root_step: Float32 = 0.35`: clamps world root motion per frame by shifting
  `tails` and `previous` together by any excess root displacement above
  `max_root_step * scale`, so high-speed `120 m/s` mecha flight or `100 m`
  teleports produce natural bounded sweep instead of `10 m` Verlet explosions!
- Intra-substep parent-to-child propagation: inside `step()`, each child spring bone
  updates its `head` and animated `goal` from its parent spring bone's live
  simulated `tail` swing before integrating Verlet and enforcing bone length.
- Non-mutating `update_skeleton` overload for `SkeletonInstance`.
"""
from std.math import abs, max, min, sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.model import NativeModel
from flint.assets.manifest import ManifestBone, ManifestVolume
from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement, rotation_between

comptime SPRING_STEP: Float32 = 1.0 / 120.0
comptime SPRING_MAX_STEPS: Int = 12


struct SpringBone(Copyable, ImplicitlyCopyable, Movable):
    var node: Int
    var parent: Int
    var parent_spring: Int
    var depth: Int
    var tail_local: Vec3
    var length: Float32
    var stiffness: Float32
    var drag: Float32
    var gravity: Float32
    var radius: Float32
    var wind: Float32

    def __init__(out self):
        self.node = -1
        self.parent = -1
        self.parent_spring = -1
        self.depth = 0
        self.tail_local = Vec3()
        self.length = 0
        self.stiffness = 0
        self.drag = 0
        self.gravity = 0
        self.radius = 0
        self.wind = 0


struct SpringCollider(Copyable, ImplicitlyCopyable, Movable):
    var node: Int
    var center: Vec3
    var axis: Vec3
    var half: Float32
    var radius: Float32

    def __init__(out self):
        self.node = -1
        self.center = Vec3()
        self.axis = Vec3(0, 1, 0)
        self.half = 0
        self.radius = 0


def rest_rotation(model: NativeModel, parents: List[Int], node: Int) -> Quat:
    var rotation = Quat()
    var current = node
    var guard = 0
    while current >= 0 and guard < 128:
        rotation = model.nodes[current].base_rotation.multiply(rotation)
        current = parents[current]
        guard += 1
    return rotation.normalized()


def closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3:
    var ab = b - a
    var length_squared = ab.length_squared()
    if length_squared < Float32(1.0e-10):
        return a
    var t = ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y + (p.z - a.z) * ab.z) / length_squared
    t = max(Float32(0), min(Float32(1), t))
    return a + ab * t


struct SpringBones(Copyable, Movable):
    var handle: Int
    var bones: List[SpringBone]
    var colliders: List[SpringCollider]
    var tails: List[Vec3]
    var previous: List[Vec3]
    var accumulator: Float32
    var max_root_step: Float32
    var last_root_world: Vec3
    var ready: Bool
    var steps: Int
    var resets: Int
    # Scratch, per bone, this frame (world space).
    var heads: List[Vec3]
    var goals: List[Vec3]
    var anim_heads: List[Vec3]
    var anim_goals: List[Vec3]
    var capsule_a: List[Vec3]
    var capsule_b: List[Vec3]

    def __init__(out self):
        self.handle = -1
        self.bones = List[SpringBone]()
        self.colliders = List[SpringCollider]()
        self.tails = List[Vec3]()
        self.previous = List[Vec3]()
        self.accumulator = 0
        self.max_root_step = 0.35
        self.last_root_world = Vec3()
        self.ready = False
        self.steps = 0
        self.resets = 0
        self.heads = List[Vec3]()
        self.goals = List[Vec3]()
        self.anim_heads = List[Vec3]()
        self.anim_goals = List[Vec3]()
        self.capsule_a = List[Vec3]()
        self.capsule_b = List[Vec3]()

    def count(self) -> Int:
        return len(self.bones)

    def bind(mut self, model: NativeModel, handle: Int, skeleton: List[ManifestBone], volumes: List[ManifestVolume]):
        """Spring bones and colliders for this model, from its manifest."""
        if handle == self.handle:
            return
        var same_rig = self.handle > 0
        self.handle = handle
        var parents = List[Int](length=len(model.nodes), fill=-1)
        for index in range(len(model.nodes)):
            for child in model.nodes[index].children:
                if child >= 0 and child < len(parents):
                    parents[child] = index
        var bones = List[SpringBone]()
        for entry in skeleton:
            if not entry.spring:
                continue
            var node = model.find_node(entry.name)
            if node < 0:
                continue
            var bone = SpringBone()
            bone.node = node
            bone.parent = parents[node]
            var depth = 0
            var current = parents[node]
            while current >= 0 and depth < 128:
                depth += 1
                current = parents[current]
            bone.depth = depth
            var rest = rest_rotation(model, parents, node)
            bone.tail_local = rest.conjugate().rotate(entry.tail - entry.head)
            bone.length = bone.tail_local.length()
            bone.stiffness = entry.stiffness
            bone.drag = max(Float32(0), min(Float32(1), entry.drag))
            bone.gravity = entry.gravity
            bone.radius = entry.radius
            bone.wind = entry.wind
            if bone.length > Float32(0.0001):
                var at = len(bones)
                while at > 0 and bones[at - 1].depth > depth:
                    at -= 1
                bones.insert(at, bone)

        # Resolve `parent_spring` index (`< i` since `bones` is sorted by ascending `depth`)
        for i in range(len(bones)):
            var p_node = bones[i].parent
            var found_ps = -1
            for j in range(i):
                if bones[j].node == p_node:
                    found_ps = j
                    break
            bones[i].parent_spring = found_ps

        var colliders = List[SpringCollider]()
        for volume in volumes:
            if volume.shape != "capsule" and volume.shape != "sphere":
                continue
            var node = model.find_node(volume.bone)
            if node < 0 or len(volume.matrix) != 16:
                continue
            var collider = SpringCollider()
            collider.node = node
            collider.center = Vec3(volume.matrix[12], volume.matrix[13], volume.matrix[14])
            var axis = Vec3(volume.matrix[4], volume.matrix[5], volume.matrix[6])
            collider.axis = axis.normalized() if axis.length() > Float32(0.0001) else Vec3(0, 1, 0)
            collider.radius = volume.size.x * Float32(0.5)
            collider.half = max(Float32(0), volume.size.y * Float32(0.5) - collider.radius) if volume.shape == "capsule" else Float32(0)
            colliders.append(collider)
        var changed = len(bones) != len(self.bones)
        self.bones = bones^
        self.colliders = colliders^
        if changed or not same_rig:
            self.reset()

    def reset(mut self):
        self.ready = False
        self.accumulator = 0
        self.resets += 1

    def _clamp_root_inertia(mut self, cur_root_world: Vec3, scale: Float32):
        var delta = cur_root_world - self.last_root_world
        var dist = delta.length()
        var limit = max(Float32(0.001), self.max_root_step * scale)
        if dist > limit and dist > Float32(1.0e-6):
            var excess = delta * ((dist - limit) / dist)
            for i in range(len(self.tails)):
                self.tails[i] = self.tails[i] + excess
                self.previous[i] = self.previous[i] + excess
        self.last_root_world = cur_root_world

    def update(mut self, mut model: NativeModel, placement: Placement, wind: Vec3, dt: Float32):
        var n = len(self.bones)
        if n == 0:
            return
        if len(self.heads) != n:
            self.heads = List[Vec3](length=n, fill=Vec3())
            self.goals = List[Vec3](length=n, fill=Vec3())
            self.anim_heads = List[Vec3](length=n, fill=Vec3())
            self.anim_goals = List[Vec3](length=n, fill=Vec3())
        var scale = (abs(placement.scale.x) + abs(placement.scale.y) + abs(placement.scale.z)) / Float32(3.0)
        for index in range(n):
            var bone = self.bones[index]
            var rotation = model.global_rotation(bone.node)
            var head = model.global_position(bone.node)
            var hw = placement.to_world(head)
            var gw = placement.to_world(head + rotation.rotate(bone.tail_local))
            self.anim_heads[index] = hw
            self.anim_goals[index] = gw
            self.heads[index] = hw
            self.goals[index] = gw

        var root_node = model.roots[0] if len(model.roots) > 0 else 0
        var cur_root_world = placement.to_world(model.global_position(root_node)) if len(model.nodes) > 0 else placement.position

        self.capsule_a.clear()
        self.capsule_b.clear()
        for collider in self.colliders:
            var rotation = model.global_rotation(collider.node)
            var center = model.global_position(collider.node) + rotation.rotate(collider.center)
            var axis = rotation.rotate(collider.axis) * collider.half
            self.capsule_a.append(placement.to_world(center - axis))
            self.capsule_b.append(placement.to_world(center + axis))

        if not self.ready or len(self.tails) != n:
            self.tails = self.goals.copy()
            self.previous = self.goals.copy()
            self.last_root_world = cur_root_world
            self.ready = True
        else:
            self._clamp_root_inertia(cur_root_world, scale)

        self.accumulator += max(Float32(0), dt)
        var steps = 0
        while self.accumulator >= SPRING_STEP and steps < SPRING_MAX_STEPS:
            self.step(wind, scale)
            self.accumulator -= SPRING_STEP
            steps += 1
        if steps == SPRING_MAX_STEPS:
            self.accumulator = 0

        for index in range(n):
            var bone = self.bones[index]
            var rotation = model.global_rotation(bone.node)
            var head = model.global_position(bone.node)
            var current = rotation.rotate(bone.tail_local)
            var wanted = placement.vector_to_model(self.tails[index] - placement.to_world(head))
            if wanted.length() > Float32(0.00001):
                model.rotate_node_model(bone.node, rotation_between(current, wanted))

    def update_skeleton(mut self, mut skel: SkeletonInstance, model: NativeModel, placement: Placement, wind: Vec3, dt: Float32):
        """Non-mutating `SkeletonInstance` overload for `SpringBones`."""
        var n = len(self.bones)
        if n == 0:
            return
        skel.ensure_topology(model)
        skel.update_matrices(model)
        if len(self.heads) != n:
            self.heads = List[Vec3](length=n, fill=Vec3())
            self.goals = List[Vec3](length=n, fill=Vec3())
            self.anim_heads = List[Vec3](length=n, fill=Vec3())
            self.anim_goals = List[Vec3](length=n, fill=Vec3())
        var scale = (abs(placement.scale.x) + abs(placement.scale.y) + abs(placement.scale.z)) / Float32(3.0)
        for index in range(n):
            var bone = self.bones[index]
            var rotation = skel.global_rotation(bone.node)
            var head = skel.global_position(bone.node)
            var hw = placement.to_world(head)
            var gw = placement.to_world(head + rotation.rotate(bone.tail_local))
            self.anim_heads[index] = hw
            self.anim_goals[index] = gw
            self.heads[index] = hw
            self.goals[index] = gw

        var root_node = model.roots[0] if len(model.roots) > 0 else 0
        var cur_root_world = placement.to_world(skel.global_position(root_node)) if skel.node_count > 0 else placement.position

        self.capsule_a.clear()
        self.capsule_b.clear()
        for collider in self.colliders:
            var rotation = skel.global_rotation(collider.node)
            var center = skel.global_position(collider.node) + rotation.rotate(collider.center)
            var axis = rotation.rotate(collider.axis) * collider.half
            self.capsule_a.append(placement.to_world(center - axis))
            self.capsule_b.append(placement.to_world(center + axis))

        if not self.ready or len(self.tails) != n:
            self.tails = self.goals.copy()
            self.previous = self.goals.copy()
            self.last_root_world = cur_root_world
            self.ready = True
        else:
            self._clamp_root_inertia(cur_root_world, scale)

        self.accumulator += max(Float32(0), dt)
        var steps = 0
        while self.accumulator >= SPRING_STEP and steps < SPRING_MAX_STEPS:
            self.step(wind, scale)
            self.accumulator -= SPRING_STEP
            steps += 1
        if steps == SPRING_MAX_STEPS:
            self.accumulator = 0

        for index in range(n):
            var bone = self.bones[index]
            var rotation = skel.global_rotation(bone.node)
            var head = skel.global_position(bone.node)
            var current = rotation.rotate(bone.tail_local)
            var wanted = placement.vector_to_model(self.tails[index] - placement.to_world(head))
            if wanted.length() > Float32(0.00001):
                skel.rotate_node_model(bone.node, rotation_between(current, wanted), model)

    def step(mut self, wind: Vec3, scale: Float32):
        var h = SPRING_STEP
        for index in range(len(self.bones)):
            var bone = self.bones[index]

            # Propagate parent simulated `tail` into child `head` & `goal` inside each substep!
            var head = self.anim_heads[index]
            var goal = self.anim_goals[index]
            var ps = bone.parent_spring
            if ps >= 0 and ps < index:
                var p_anim_arm = self.anim_goals[ps] - self.anim_heads[ps]
                var p_sim_arm = self.tails[ps] - self.heads[ps]
                var p_swing = rotation_between(p_anim_arm, p_sim_arm)
                head = self.heads[ps] + p_swing.rotate(self.anim_heads[index] - self.anim_heads[ps])
                goal = head + p_swing.rotate(self.anim_goals[index] - self.anim_heads[index])
            self.heads[index] = head
            self.goals[index] = goal

            var tail = self.tails[index]
            var inertia = (tail - self.previous[index]) * (Float32(1.0) - bone.drag)
            var pull = (goal - tail) * min(Float32(1.0), bone.stiffness * h)
            var accel = Vec3(0, -bone.gravity, 0) + wind * bone.wind
            var next = tail + inertia + pull + accel * (h * h)

            # Enforce bone length from live parent-propagated `head`
            var arm = next - head
            var length = bone.length * scale
            if arm.length() > Float32(0.00001):
                next = head + arm * (length / arm.length())

            # Push out of capsule/sphere colliders
            for c in range(len(self.colliders)):
                var collider = self.colliders[c]
                if collider.node == bone.node or collider.node == bone.parent:
                    continue
                var closest = closest_on_segment(next, self.capsule_a[c], self.capsule_b[c])
                var away = next - closest
                var reach = (collider.radius + bone.radius) * scale
                var distance = away.length()
                if distance < reach:
                    var push = away * (Float32(1.0) / distance) if distance > Float32(0.00001) else Vec3(0, 1, 0)
                    next = closest + push * reach
                    var again = next - head
                    if again.length() > Float32(0.00001):
                        next = head + again * (length / again.length())
            self.previous[index] = tail
            self.tails[index] = next
        self.steps += 1


@fieldwise_init
struct DirectSpringSphereCollider(Copyable, ImplicitlyCopyable, Movable):
    """Sphere collider attached to a `DirectSpringChain` root frame (`local_center`, `radius`)."""
    var local_center: Vec3
    var radius: Float32

    def __init__(out self):
        self.local_center = Vec3()
        self.radius = 0.0


struct DirectSpringChain(Copyable, Movable):
    """Standalone fixed-`120 Hz` Verlet secondary-motion chain solver for hair, capes, ribbons & tails."""
    var rest_local_offsets: List[Vec3]
    var rest_lengths: List[Float32]
    var heads: List[Vec3]
    var tails: List[Vec3]
    var previous: List[Vec3]
    var colliders: List[DirectSpringSphereCollider]
    var stiffness: Float32
    var drag: Float32
    var gravity: Float32
    var wind_scale: Float32
    var link_radius: Float32
    var max_root_step: Float32
    var accumulator: Float32
    var last_root_pos: Vec3
    var last_root_rot: Quat
    var ready: Bool

    def __init__(
        out self,
        rest_local_offsets: List[Vec3],
        stiffness: Float32 = 18.0,
        drag: Float32 = 0.14,
        gravity: Float32 = -6.5,
        wind_scale: Float32 = 1.0,
        link_radius: Float32 = 0.025,
        max_root_step: Float32 = 0.35,
    ):
        var n = len(rest_local_offsets)
        self.rest_local_offsets = List[Vec3](capacity=max(1, n))
        self.rest_lengths = List[Float32](capacity=max(1, n))
        for i in range(n):
            var off = rest_local_offsets[i]
            var l = off.length()
            if l < Float32(1e-5):
                off = Vec3(0.0, -0.001, 0.0)
                l = Float32(0.001)
            self.rest_local_offsets.append(off)
            self.rest_lengths.append(l)

        self.heads = List[Vec3](length=n, fill=Vec3())
        self.tails = List[Vec3](length=n, fill=Vec3())
        self.previous = List[Vec3](length=n, fill=Vec3())
        self.colliders = List[DirectSpringSphereCollider]()
        self.stiffness = max(Float32(0.0), stiffness)
        self.drag = max(Float32(0.0), min(Float32(1.0), drag))
        self.gravity = gravity
        self.wind_scale = wind_scale
        self.link_radius = max(Float32(0.0), link_radius)
        self.max_root_step = max(Float32(1e-4), max_root_step)
        self.accumulator = 0.0
        self.last_root_pos = Vec3()
        self.last_root_rot = Quat()
        self.ready = False
        self.reset(Vec3(), Quat())
        self.ready = False

    def reset(mut self, root_world_pos: Vec3, root_world_rot: Quat):
        var q = root_world_rot.normalized()
        var n = len(self.rest_local_offsets)
        if len(self.heads) != n:
            self.heads = List[Vec3](length=n, fill=Vec3())
            self.tails = List[Vec3](length=n, fill=Vec3())
            self.previous = List[Vec3](length=n, fill=Vec3())
        var cur_head = root_world_pos
        for i in range(n):
            var cur_tail = cur_head + q.rotate(self.rest_local_offsets[i])
            self.heads[i] = cur_head
            self.tails[i] = cur_tail
            self.previous[i] = cur_tail
            cur_head = cur_tail
        self.accumulator = 0.0
        self.last_root_pos = root_world_pos
        self.last_root_rot = q
        self.ready = True

    def add_sphere_collider(mut self, local_center: Vec3, radius: Float32):
        self.colliders.append(DirectSpringSphereCollider(local_center, max(Float32(0.0), radius)))

    def _project_link_constraints(
        self,
        head: Vec3,
        goal: Vec3,
        rest_len: Float32,
        candidate: Vec3,
        root_world_pos: Vec3,
        q_root: Quat,
    ) -> Vec3:
        var next = candidate
        var arm0 = next - head
        var arm0_len = arm0.length()
        if arm0_len > Float32(1e-6):
            next = head + arm0 * (rest_len / arm0_len)
        else:
            next = goal

        if len(self.colliders) == 0:
            return next

        for _ in range(2):
            for c_idx in range(len(self.colliders)):
                var col = self.colliders[c_idx]
                var c_world = root_world_pos + q_root.rotate(col.local_center)
                var reach = col.radius + self.link_radius
                var away = next - c_world
                var dist_c = away.length()
                if dist_c < reach:
                    var push_dir: Vec3
                    if dist_c > Float32(1e-5):
                        push_dir = away * (Float32(1.0) / dist_c)
                    else:
                        var fb = goal - c_world
                        push_dir = fb.normalized() if fb.length() > Float32(1e-5) else Vec3(0.0, 1.0, 0.0)
                    next = c_world + push_dir * reach

                    # Exact circle-of-intersection projection preserving |next - head| == rest_len AND |next - c_world| >= reach
                    var hc = c_world - head
                    var d_hc = hc.length()
                    if d_hc > Float32(1e-5):
                        var u_hc = hc * (Float32(1.0) / d_hc)
                        var rel = next - head
                        var x_max = (d_hc * d_hc + rest_len * rest_len - reach * reach) / (Float32(2.0) * d_hc)
                        var x_clamped = max(-rest_len, min(rest_len, x_max))
                        var x_cur = rel.dot(u_hc)
                        if x_cur > x_clamped:
                            var perp = rel - u_hc * x_cur
                            var v_perp: Vec3
                            if perp.length() > Float32(1e-5):
                                v_perp = perp.normalized()
                            else:
                                var alt = Vec3(0.0, 0.0, 1.0) if abs(u_hc.z) < Float32(0.9) else Vec3(1.0, 0.0, 0.0)
                                v_perp = u_hc.cross(alt).normalized()
                            var y_req = sqrt(max(Float32(0.0), rest_len * rest_len - x_clamped * x_clamped))
                            next = head + u_hc * x_clamped + v_perp * y_req

            var arm_f = next - head
            var arm_f_len = arm_f.length()
            if arm_f_len > Float32(1e-6):
                next = head + arm_f * (rest_len / arm_f_len)
            else:
                next = goal
        return next

    def step(
        mut self,
        dt: Float32,
        root_world_pos: Vec3,
        root_world_rot: Quat,
        wind_world: Vec3 = Vec3(0.0, 0.0, 0.0),
    ):
        var n = len(self.rest_local_offsets)
        if n == 0:
            self.last_root_pos = root_world_pos
            self.last_root_rot = root_world_rot.normalized()
            return

        var q_root = root_world_rot.normalized()
        if not self.ready or len(self.tails) != n:
            self.reset(root_world_pos, q_root)
        else:
            var root_delta = root_world_pos - self.last_root_pos
            var root_dist = root_delta.length()
            var limit = max(Float32(1e-4), self.max_root_step)
            if root_dist > limit:
                var excess = root_delta * ((root_dist - limit) / root_dist)
                for i in range(n):
                    self.tails[i] = self.tails[i] + excess
                    self.previous[i] = self.previous[i] + excess
            self.last_root_pos = root_world_pos
            self.last_root_rot = q_root

        self.accumulator += max(Float32(0.0), dt)
        var substeps = 0
        var h = SPRING_STEP
        var pull_k = min(Float32(1.0), self.stiffness * h)
        var accel_dt2 = (Vec3(0.0, self.gravity, 0.0) + wind_world * self.wind_scale) * (h * h)

        while self.accumulator >= SPRING_STEP and substeps < SPRING_MAX_STEPS:
            for i in range(n):
                var head = root_world_pos if i == 0 else self.tails[i - 1]
                self.heads[i] = head
                var goal = head + q_root.rotate(self.rest_local_offsets[i])
                var rest_len = self.rest_lengths[i]
                var tail = self.tails[i]
                var inertia = (tail - self.previous[i]) * (Float32(1.0) - self.drag)
                var pull = (goal - tail) * pull_k
                var raw_next = tail + inertia + pull + accel_dt2
                var next = self._project_link_constraints(head, goal, rest_len, raw_next, root_world_pos, q_root)
                self.previous[i] = tail
                self.tails[i] = next
            self.accumulator -= SPRING_STEP
            substeps += 1

        if substeps == SPRING_MAX_STEPS:
            self.accumulator = 0.0
        elif substeps == 0:
            for i in range(n):
                var head = root_world_pos if i == 0 else self.tails[i - 1]
                self.heads[i] = head
                var goal = head + q_root.rotate(self.rest_local_offsets[i])
                var old_tail = self.tails[i]
                var new_tail = self._project_link_constraints(
                    head, goal, self.rest_lengths[i], old_tail, root_world_pos, q_root
                )
                self.previous[i] = self.previous[i] + (new_tail - old_tail)
                self.tails[i] = new_tail

    def link_count(self) -> Int:
        return len(self.rest_local_offsets)

    def world_head_pos(self, idx: Int) -> Vec3:
        var n = len(self.heads)
        if n == 0:
            return self.last_root_pos
        var c_idx = max(0, min(n - 1, idx))
        return self.heads[c_idx]

    def world_tail_pos(self, idx: Int) -> Vec3:
        var n = len(self.tails)
        if n == 0:
            return self.last_root_pos
        var c_idx = max(0, min(n - 1, idx))
        return self.tails[c_idx]

    def world_link_rot(self, idx: Int) -> Quat:
        var n = len(self.rest_local_offsets)
        if n == 0:
            return self.last_root_rot
        var c_idx = max(0, min(n - 1, idx))
        var rest_world_dir = self.last_root_rot.rotate(self.rest_local_offsets[c_idx]).normalized()
        var live_world_dir = (self.tails[c_idx] - self.heads[c_idx]).normalized()
        return rotation_between(rest_world_dir, live_world_dir).multiply(self.last_root_rot).normalized()
