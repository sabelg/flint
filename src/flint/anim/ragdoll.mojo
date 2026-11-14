"""Bodies that react (package 3.11): a powered ragdoll over the animated pose.

Nineteen particles stand on the joints of a bear-shaped skeleton (pelvis,
spine, chest, neck, head and nose, the arms and the legs), held together by
the 2.12 XPBD core (physics/xpbd.mojo `ParticleMesh`): stiff links along the
bones, braces across the hips, shoulders and torso so the trunk keeps its
shape, and `min` links that stop a knee or an elbow folding through itself.
Each particle stands on the terrain under it (one ray a frame from the
caller) with friction. It runs at a fixed 120 Hz presentation step.

Four modes:

- ANIMATED: the particles sit on the animated joints (and carry their
  velocity, so a body that goes limp mid-run keeps going).
- REACTING (a hit): the particles near the hit get its push and a weight; a
  power (per second) pulls every particle back toward its animated joint and
  the weights fall back to zero, so the reaction is blended per bone and
  dies away: a chest hit snaps the chest and head back, the feet barely move.
- LIMP (a knockdown or a defeat): no power, gravity, every weight 1; the
  body falls and settles on the ground.
- GETUP: when the knockdown ends, the settled ragdoll is matched against
  frames of the recovery clips (a heading-free direction for each bone, the
  pelvis and chest counted three times) and the best clip plays from the
  best frame, the ragdoll fading into it over GETUP_IN seconds and the clip
  into the live graph over the last GETUP_OUT seconds.

The bones follow the particles parents first: the pelvis moves to its
particle and turns to the frame of its spine and hip particles, the chest
to its neck and shoulders, every other bone swings to point at its child
particle, each by its weight. Presentation only: the game is read, never
written, and the same frames give the same pictures (fixed order, no
randomness). The sim's stand-ins are elsewhere: Maroon's capsule, and a
cosmetic body for each defeated enemy (gameplay/corpses.mojo).
"""
from std.math import sqrt, exp, atan2, sin, cos
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.ik import cross, dot, safe_normalized, angle_axis
from flint.assets.model import NativeModel
from flint.anim.pose import Pose, sample_pose, blend_into, write_pose
from flint.anim.procedural import Placement, rotation_between
from flint.physics.xpbd import ParticleMesh, ChainCollider, LINK_EQUAL, LINK_MIN

comptime RAGDOLL_STEP: Float32 = 1.0 / 120.0
comptime RAGDOLL_MAX_STEPS: Int = 12
comptime RAGDOLL_SUBSTEPS: Int = 2

comptime RAGDOLL_ANIMATED: Int = 0
comptime RAGDOLL_REACTING: Int = 1
comptime RAGDOLL_LIMP: Int = 2
comptime RAGDOLL_GETUP: Int = 3

comptime REACT_POWER: Float32 = 16.0       # per second, back toward the pose
comptime REACT_HALFLIFE: Float32 = 0.16    # seconds for a hit's weights to halve
comptime REACT_MAX: Float32 = 0.9          # the most a hit hands a bone to the ragdoll
comptime REACT_REACH: Float32 = 0.9        # metres (at scale 1) a hit's share falls by 1/e
comptime SETTLE_SPEED: Float32 = 0.35      # m/s mean particle speed counted as at rest
comptime SETTLE_SECONDS: Float32 = 0.35
comptime GETUP_IN: Float32 = 0.25          # seconds from the ragdoll to the matched frame
comptime GETUP_OUT: Float32 = 0.35         # seconds from the clip to the live graph
comptime GETUP_MAX: Float32 = 1.2          # the most of a clip a get-up plays
comptime GETUP_SAMPLES: Int = 12           # frames tried per clip


def ragdoll_joints() -> List[String]:
    return [
        String("Pelvis"), String("Spine1"), String("Spine2"), String("Chest"), String("Neck"), String("Head"),
        String("UpperArm.L"), String("Forearm.L"), String("Hand.L"),
        String("UpperArm.R"), String("Forearm.R"), String("Hand.R"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
        String("Nose"),
    ]


def ragdoll_parents() -> List[Int]:
    """Each particle's parent particle (-1: the pelvis)."""
    return [-1, 0, 1, 2, 3, 4, 3, 6, 7, 3, 9, 10, 0, 12, 13, 0, 15, 16, 5]


def ragdoll_masses() -> List[Float32]:
    return [
        Float32(6), Float32(4), Float32(4), Float32(6), Float32(1.5), Float32(3),
        Float32(1.2), Float32(1.0), Float32(0.5), Float32(1.2), Float32(1.0), Float32(0.5),
        Float32(2.0), Float32(1.4), Float32(0.6), Float32(2.0), Float32(1.4), Float32(0.6),
        Float32(0.4),
    ]


def ragdoll_radii() -> List[Float32]:
    """Metres at scale 1 (Maroon's model is about 3 m long)."""
    return [
        Float32(0.32), Float32(0.32), Float32(0.34), Float32(0.34), Float32(0.22), Float32(0.24),
        Float32(0.13), Float32(0.11), Float32(0.10), Float32(0.13), Float32(0.11), Float32(0.10),
        Float32(0.15), Float32(0.12), Float32(0.10), Float32(0.15), Float32(0.12), Float32(0.10),
        Float32(0.08),
    ]


def getup_clips() -> List[String]:
    """Clips whose frames a get-up may start from, best first on a tie."""
    return [String("Recovery"), String("Defeat"), String("Landing"), String("Belly_Slide"), String("Hurt"), String("Roll"), String("Idle")]


def quat_from_basis(x: Vec3, y: Vec3, z: Vec3) -> Quat:
    """The rotation whose columns are the orthonormal x, y, z (Shepperd)."""
    var trace = x.x + y.y + z.z
    if trace > 0:
        var s = sqrt(trace + 1.0) * 2.0
        return Quat((y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, 0.25 * s).normalized()
    if x.x > y.y and x.x > z.z:
        var s = sqrt(1.0 + x.x - y.y - z.z) * 2.0
        return Quat(0.25 * s, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s).normalized()
    if y.y > z.z:
        var s = sqrt(1.0 + y.y - x.x - z.z) * 2.0
        return Quat((y.x + x.y) / s, 0.25 * s, (z.y + y.z) / s, (z.x - x.z) / s).normalized()
    var s = sqrt(1.0 + z.z - x.x - y.y) * 2.0
    return Quat((z.x + x.z) / s, (z.y + y.z) / s, 0.25 * s, (x.y - y.x) / s).normalized()


def frame_of(up: Vec3, side: Vec3) -> Quat:
    """A frame whose y is `up` and whose x is `side` made square to it."""
    var y = safe_normalized(up, Vec3(0, 1, 0))
    var x = side - y * dot(side, y)
    x = safe_normalized(x, safe_normalized(cross(y, Vec3(0, 0, 1)), Vec3(1, 0, 0)))
    var z = cross(x, y)
    return quat_from_basis(x, y, z)


def heading_free(dirs: List[Vec3], side: Vec3, fallback: Vec3) -> List[Vec3]:
    """The directions turned about +Y so the body's side axis (or, lying on
    its side, the fallback) points along +X."""
    var flat = Vec3(side.x, 0, side.z)
    if flat.length() < 0.3 * max(Float32(1.0e-6), side.length()):
        flat = Vec3(fallback.x, 0, fallback.z)
    var angle = atan2(flat.z, flat.x) if flat.length() > 1.0e-6 else Float32(0)
    var c = cos(angle)
    var s = sin(angle)
    var out = List[Vec3]()
    for d in dirs:
        # Turn by -angle about +Y: (x, z) -> (x c + z s, -x s + z c).
        out.append(Vec3(d.x * c + d.z * s, d.y, -d.x * s + d.z * c))
    return out^


def pose_cost(a: List[Vec3], b: List[Vec3]) -> Float32:
    """Squared distance between two feature lists (the first two counted thrice)."""
    var total = Float32(0)
    for i in range(min(len(a), len(b))):
        var w = Float32(3) if i < 2 else Float32(1)
        total += (a[i] - b[i]).length_squared() * w
    return total


struct GetupChoice(Copyable, ImplicitlyCopyable, Movable):
    var clip: Int
    var time: Float32
    var cost: Float32

    def __init__(out self, clip: Int = -1, time: Float32 = 0, cost: Float32 = 1.0e9):
        self.clip = clip
        self.time = time
        self.cost = cost


struct Ragdoll(Copyable, Movable):
    var handle: Int
    var nodes: List[Int]       # the model node of each particle (-1: missing)
    var mesh: ParticleMesh
    var particle: List[Int]    # each joint's particle in the mesh (-1: missing)
    var weight: List[Float32]  # per joint: how far its bone follows the ragdoll
    var animated: List[Vec3]   # this frame's animated joints (world)
    var floors: List[Float32]  # ground under each joint (world y), from the caller
    var mode: Int
    var ready: Bool
    var accumulator: Float32
    var scale: Float32
    var still: Float32         # seconds at rest in LIMP
    var limp_time: Float32
    var getup: GetupChoice
    var getup_time: Float32
    var getup_length: Float32
    var root: Int
    var scratch: Pose
    var clip_pose: Pose
    # Counters (tests, the inspector).
    var hits: Int
    var steps: Int
    var getups: Int
    var last_match: String

    def __init__(out self):
        self.handle = -1
        self.nodes = List[Int]()
        self.mesh = ParticleMesh()
        self.particle = List[Int]()
        self.weight = List[Float32]()
        self.animated = List[Vec3]()
        self.floors = List[Float32]()
        self.mode = RAGDOLL_ANIMATED
        self.ready = False
        self.accumulator = 0
        self.scale = 1
        self.still = 0
        self.limp_time = 0
        self.getup = GetupChoice()
        self.getup_time = 0
        self.getup_length = 0
        self.root = -1
        self.scratch = Pose()
        self.clip_pose = Pose()
        self.hits = 0
        self.steps = 0
        self.getups = 0
        self.last_match = String("")

    def bound(self) -> Bool:
        return len(self.nodes) > 0 and self.nodes[0] >= 0

    def bind(mut self, mut model: NativeModel, handle: Int):
        """The joints of this model (another model of the same rig keeps the state)."""
        if handle == self.handle:
            return
        self.handle = handle
        var names = ragdoll_joints()
        var nodes = List[Int]()
        for name in names:
            nodes.append(model.find_node(name))
        var same = len(nodes) == len(self.nodes)
        if same:
            for i in range(len(nodes)):
                if nodes[i] != self.nodes[i]:
                    same = False
        self.nodes = nodes^
        self.root = model.find_node("Root")
        model.ensure_parents()
        if not same:
            self.ready = False

    def reset(mut self):
        """Back on the pose at once (a teleport, a new layer, a seek)."""
        self.ready = False
        self.mode = RAGDOLL_ANIMATED
        self.accumulator = 0
        for i in range(len(self.weight)):
            self.weight[i] = 0

    def active(self) -> Bool:
        return self.mode != RAGDOLL_ANIMATED

    def largest_weight(self) -> Float32:
        var w = Float32(0)
        for value in self.weight:
            w = max(w, value)
        return w

    def build(mut self, placement: Placement):
        """The particle mesh on this frame's animated joints."""
        var parents = ragdoll_parents()
        var masses = ragdoll_masses()
        var radii = ragdoll_radii()
        self.scale = max(Float32(0.05), (abs(placement.scale.x) + abs(placement.scale.y) + abs(placement.scale.z)) / 3.0)
        self.mesh = ParticleMesh()
        self.mesh.damping = 0.8
        self.mesh.friction = 0.45
        self.particle = List[Int](length=len(self.nodes), fill=-1)
        for i in range(len(self.nodes)):
            if self.nodes[i] < 0:
                continue
            self.particle[i] = self.mesh.add_particle(self.animated[i], masses[i], radii[i] * self.scale)
        # Bones (stiff), then the braces that keep the trunk's shape.
        for i in range(len(self.nodes)):
            var p = parents[i]
            if p >= 0 and self.particle[i] >= 0 and self.particle[p] >= 0:
                self.mesh.add_link(self.particle[p], self.particle[i], 0)
        var braces = [(6, 9), (12, 15), (6, 12), (9, 15), (6, 15), (9, 12), (0, 3), (1, 3), (0, 2), (3, 5), (6, 4), (9, 4), (2, 6), (2, 9), (0, 6), (0, 9)]
        for pair in braces:
            var a = self.particle[pair[0]]
            var b = self.particle[pair[1]]
            if a >= 0 and b >= 0:
                self.mesh.add_link(a, b, 1.0e-5)
        # Elbows and knees: the hand never closer than 60% of the arm to the
        # shoulder, the foot than 60% of the leg to the hip.
        var folds = [(6, 8), (9, 11), (12, 14), (15, 17), (3, 0)]
        for pair in folds:
            var a = self.particle[pair[0]]
            var b = self.particle[pair[1]]
            if a >= 0 and b >= 0:
                self.mesh.add_link(a, b, 0, LINK_MIN, 0.6)
        self.weight = List[Float32](length=len(self.nodes), fill=0)
        self.floors = List[Float32](length=len(self.nodes), fill=-1.0e9)
        self.ready = True

    def read_animated(mut self, mut model: NativeModel, placement: Placement):
        if len(self.animated) != len(self.nodes):
            self.animated = List[Vec3](length=len(self.nodes), fill=Vec3())
        for i in range(len(self.nodes)):
            if self.nodes[i] >= 0:
                self.animated[i] = placement.to_world(model.global_position(self.nodes[i]))

    def joint_world(self, joint: Int) -> Vec3:
        """Where a joint's particle is (tests, shots, the caller's ground rays)."""
        if joint < 0 or joint >= len(self.particle) or self.particle[joint] < 0:
            return Vec3()
        return self.mesh.position[self.particle[joint]]

    def set_floor(mut self, joint: Int, height: Float32):
        if joint >= 0 and joint < len(self.floors):
            self.floors[joint] = height

    def hit(mut self, point: Vec3, push: Vec3, strength: Float32):
        """A blow at `point` (world) with a velocity `push` (m/s): the
        particles near it take the push and hand their bones to the ragdoll
        by how near they are."""
        if not self.ready:
            return
        self.hits += 1
        var reach = REACT_REACH * self.scale
        for i in range(len(self.nodes)):
            var p = self.particle[i]
            if p < 0:
                continue
            var d = (self.mesh.position[p] - point).length()
            var share = exp(-d / max(Float32(0.01), reach))
            self.mesh.push(p, push * share)
            if self.mode != RAGDOLL_LIMP:
                self.weight[i] = max(self.weight[i], min(Float32(1), strength) * share * REACT_MAX)
        if self.mode == RAGDOLL_ANIMATED:
            self.mode = RAGDOLL_REACTING

    def go_limp(mut self):
        if self.mode == RAGDOLL_LIMP:
            return
        self.mode = RAGDOLL_LIMP
        self.still = 0
        self.limp_time = 0
        for i in range(len(self.weight)):
            self.weight[i] = 1

    def settled(self) -> Bool:
        return self.mode == RAGDOLL_LIMP and self.still >= SETTLE_SECONDS

    def simulate(mut self, dt: Float32):
        var limp = self.mode == RAGDOLL_LIMP
        self.mesh.gravity_scale = 1 if limp else 0
        for i in range(len(self.nodes)):
            var p = self.particle[i]
            if p < 0:
                continue
            self.mesh.target[p] = self.animated[i]
            self.mesh.power[p] = 0 if limp else REACT_POWER
            self.mesh.floor[p] = self.floors[i]
        self.accumulator += max(Float32(0), dt)
        var steps = 0
        var none = List[Vec3]()
        var colliders = List[ChainCollider]()
        while self.accumulator >= RAGDOLL_STEP and steps < RAGDOLL_MAX_STEPS:
            self.mesh.step(RAGDOLL_STEP, RAGDOLL_SUBSTEPS, 9.81, none, colliders)
            self.accumulator -= RAGDOLL_STEP
            steps += 1
            self.steps += 1
        if steps == RAGDOLL_MAX_STEPS:
            self.accumulator = 0

    def follow_animation(mut self, dt: Float32):
        """ANIMATED: on the joints, with their velocity."""
        for i in range(len(self.nodes)):
            var p = self.particle[i]
            if p < 0:
                continue
            var before = self.mesh.position[p]
            var now = self.animated[i]
            self.mesh.position[p] = now
            self.mesh.previous[p] = before
            self.mesh.velocity[p] = (now - before) * (1.0 / max(Float32(1.0e-3), dt))

    def update(mut self, mut model: NativeModel, handle: Int, placement: Placement, limp_wanted: Bool, dt: Float32):
        """One frame, after the graph and the procedural passes have posed the
        model and before the helper joints, correctives and spring bones."""
        self.bind(model, handle)
        if not self.bound():
            return
        self.read_animated(model, placement)
        if not self.ready:
            self.build(placement)
        if limp_wanted:
            self.go_limp()
        elif self.mode == RAGDOLL_LIMP:
            self.start_getup(model)
        if self.mode == RAGDOLL_ANIMATED:
            self.follow_animation(dt)
            return
        if self.mode == RAGDOLL_GETUP:
            self.play_getup(model, placement, dt)
            return
        self.simulate(dt)
        if self.mode == RAGDOLL_LIMP:
            self.limp_time += dt
            if self.mesh.mean_speed() < SETTLE_SPEED * self.scale:
                self.still += dt
            else:
                self.still = 0
        else:
            var fade = exp(-0.69314718 * dt / REACT_HALFLIFE)
            var top = Float32(0)
            for i in range(len(self.weight)):
                self.weight[i] *= fade
                if self.weight[i] < 0.01:
                    self.weight[i] = 0
                top = max(top, self.weight[i])
            if top == 0:
                self.mode = RAGDOLL_ANIMATED
                return
        self.drive(model, placement)

    def drive(mut self, mut model: NativeModel, placement: Placement):
        """The bones toward the particles, parents first, each by its weight."""
        var parents = ragdoll_parents()
        # Pelvis: its place, then its frame from the spine and the hips.
        var w0 = self.weight[0]
        if w0 > 0 and self.particle[0] >= 0:
            var pelvis = self.nodes[0]
            var now = placement.to_world(model.global_position(pelvis))
            model.offset_node(pelvis, placement.vector_to_model(self.joint_world(0) - now) * w0)
            self.turn_frame(model, placement, 0, 1, 12, 15, w0)
        for i in range(1, len(self.nodes)):
            if self.nodes[i] < 0 or self.weight[i] <= 0:
                continue
            if i == 3:
                self.turn_frame(model, placement, 3, 4, 6, 9, self.weight[3])
                continue
            # The bone at joint i points at its first child joint.
            var child = -1
            for j in range(len(parents)):
                if parents[j] == i and self.nodes[j] >= 0:
                    child = j
                    break
            if child < 0:
                continue
            var head = model.global_position(self.nodes[i])
            var tail = model.global_position(self.nodes[child])
            var want = placement.vector_to_model(self.joint_world(child) - self.joint_world(i))
            var turn = rotation_between(tail - head, want)
            model.rotate_node_model(self.nodes[i], Quat().slerp(turn, self.weight[i]))

    def turn_frame(mut self, mut model: NativeModel, placement: Placement, joint: Int, up: Int, left: Int, right: Int, weight: Float32):
        """A bone with two children across it (the pelvis, the chest): turned
        so its up child and its left-right pair match the particles."""
        if self.nodes[joint] < 0 or self.nodes[up] < 0:
            return
        var head = model.global_position(self.nodes[joint])
        var up_now = model.global_position(self.nodes[up]) - head
        var up_want = placement.vector_to_model(self.joint_world(up) - self.joint_world(joint))
        if self.nodes[left] >= 0 and self.nodes[right] >= 0:
            var side_now = model.global_position(self.nodes[left]) - model.global_position(self.nodes[right])
            var side_want = placement.vector_to_model(self.joint_world(left) - self.joint_world(right))
            var turn = frame_of(up_want, side_want).multiply(frame_of(up_now, side_now).conjugate()).normalized()
            model.rotate_node_model(self.nodes[joint], Quat().slerp(turn, weight))
        else:
            model.rotate_node_model(self.nodes[joint], Quat().slerp(rotation_between(up_now, up_want), weight))

    # ------------------------------------------------------------------
    # Get-up by pose match
    # ------------------------------------------------------------------

    def features(self, points: List[Vec3]) -> List[Vec3]:
        """Heading-free bone directions of a set of joint points: pelvis to
        spine, chest to neck first, then every other bone."""
        var parents = ragdoll_parents()
        var dirs = List[Vec3]()
        var pairs = List[Tuple[Int, Int]]()
        pairs.append((0, 1))
        pairs.append((3, 4))
        for i in range(len(parents)):
            if parents[i] >= 0 and i != 1 and i != 4:
                pairs.append((parents[i], i))
        for pair in pairs:
            var a = pair[0]
            var b = pair[1]
            if a < len(points) and b < len(points) and self.nodes[a] >= 0 and self.nodes[b] >= 0:
                dirs.append(safe_normalized(points[b] - points[a], Vec3(0, 1, 0)))
            else:
                dirs.append(Vec3())
        var side = points[12] - points[15] if len(points) > 15 else Vec3(1, 0, 0)
        var fallback = points[3] - points[0] if len(points) > 3 else Vec3(0, 0, 1)
        return heading_free(dirs, side, fallback)

    def ragdoll_points(self) -> List[Vec3]:
        var points = List[Vec3]()
        for i in range(len(self.nodes)):
            points.append(self.joint_world(i))
        return points^

    def pose_points(self, model: NativeModel, pose: Pose) -> List[Vec3]:
        """Model-space joint points of a pose (forward kinematics over the
        model's parents)."""
        var points = List[Vec3]()
        for i in range(len(self.nodes)):
            var node = self.nodes[i]
            var position = Vec3()
            var current = node
            var guard = 0
            while current >= 0 and guard < 128 and current < pose.count():
                position = pose.t[current] + pose.r[current].rotate(position)
                current = model.parents[current] if current < len(model.parents) else -1
                guard += 1
            points.append(position)
        return points^

    def match(mut self, mut model: NativeModel) -> GetupChoice:
        """The clip frame closest to the ragdoll's pose."""
        model.ensure_parents()
        var want = self.features(self.ragdoll_points())
        var best = GetupChoice()
        var names = getup_clips()
        for name in names:
            var clip = -1
            for index in range(len(model.animations)):
                if model.animations[index].name == name or model.animations[index].name.endswith("_" + name):
                    clip = index
                    break
            if clip < 0:
                continue
            var duration = model.animations[clip].duration
            for k in range(GETUP_SAMPLES):
                var t = duration * 0.7 * Float32(k) / Float32(max(1, GETUP_SAMPLES - 1))
                sample_pose(model, clip, t, self.clip_pose)
                var cost = pose_cost(want, self.features(self.pose_points(model, self.clip_pose)))
                if cost < best.cost - 1.0e-4:
                    best = GetupChoice(clip, t, cost)
        return best

    def start_getup(mut self, mut model: NativeModel):
        self.getup = self.match(model)
        self.getup_time = 0
        if self.getup.clip < 0:
            # Nothing to match: fade straight back to the graph.
            self.getup_length = GETUP_IN + GETUP_OUT
            self.last_match = String("")
        else:
            var left = model.animations[self.getup.clip].duration - self.getup.time
            self.getup_length = max(GETUP_IN + GETUP_OUT, min(GETUP_MAX, left))
            self.last_match = model.animations[self.getup.clip].name
        self.getups += 1
        self.mode = RAGDOLL_GETUP

    def getup_weights(self) -> Tuple[Float32, Float32]:
        """(the clip over the graph, the ragdoll over the clip) now."""
        var t = self.getup_time
        var r = max(Float32(0), 1 - t / GETUP_IN)
        r = r * r * (3 - 2 * r)
        var out_start = self.getup_length - GETUP_OUT
        var g = Float32(1) if t <= out_start else max(Float32(0), 1 - (t - out_start) / GETUP_OUT)
        if self.getup.clip < 0:
            g = 0
        return (g, r)

    def play_getup(mut self, mut model: NativeModel, placement: Placement, dt: Float32):
        self.getup_time += max(Float32(0), dt)
        if self.getup_time >= self.getup_length:
            self.mode = RAGDOLL_ANIMATED
            for i in range(len(self.weight)):
                self.weight[i] = 0
            self.follow_animation(dt)
            return
        var weights = self.getup_weights()
        var g = weights[0]
        var r = weights[1]
        if g > 0 and self.getup.clip >= 0:
            self.scratch.read(model)
            sample_pose(model, self.getup.clip, self.getup.time + self.getup_time, self.clip_pose)
            if self.root >= 0 and self.root < self.clip_pose.count():
                # He stays where the game has him: the clip's root never moves him.
                self.clip_pose.t[self.root] = self.scratch.t[self.root]
                self.clip_pose.r[self.root] = self.scratch.r[self.root]
            for i in range(min(self.clip_pose.count(), self.scratch.count())):
                self.clip_pose.s[i] = self.scratch.s[i]  # the form's joint scales stay
            blend_into(self.scratch, self.clip_pose, g)
            write_pose(model, self.scratch)
        for i in range(len(self.weight)):
            self.weight[i] = r
        if r > 0:
            self.drive(model, placement)
