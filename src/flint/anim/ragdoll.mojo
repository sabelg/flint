"""Bodies that react (package 3.11 / v4): XPBD particle ragdoll (`Ragdoll`) +
data-driven 6-DOF active-PD rigid-capsule ragdoll (`ArticulatedRagdoll`).

Upgrades in Flint v4:
1. 100% deterministic `flint.core.detmath` (`det_exp`, `det_atan2`, `det_sin`,
   `det_cos`, `det_acos`, `PI`) across all ragdoll & recovery calculations.
2. `Ragdoll.settled_pelvis_yaw()` + `Ragdoll.settled_root_offset()` and root
   alignment in `Ragdoll.play_getup` so recovery never snaps heading 120 degrees
   or drags across the floor.
3. Data-driven 6-DOF `ArticulatedRagdoll` (`ArticulatedLimbDef`,
   `ArticulatedLimbState`, `RagdollObbCollider`, `ArticulatedRagdoll`) usable on
   any humanoid, Zentraedi `Regult`/`Glaug` reverse-walker, or `VF-1 Battloid` rig:
   - Arbitrary hierarchical skeleton bone mappings (zero hardcoded bone names).
   - Exact solid-capsule axial & transverse moments of inertia (`compute_capsule_inertia`).
   - Exact 6-DOF linear & angular velocity handoff (`v_i`, `omega_i`) from
     consecutive `SkeletonInstance` poses or `prev_world_matrices`.
   - Implicitly-stabilized Active PD Joint Torque Motors (`active_stiffness`,
     `active_damping`, `max_torque`) with equal-and-opposite action-reaction
     torques and ballistic/melee impulse response (`apply_impulse_at_bone`).
   - Anatomical swing-cone & 1-DOF hinge/twist angle limits (knees and elbows
     never hyperextend backward).
   - Terrain/deck surface-plane + OBB 6-DOF lever-arm collision push-out & friction,
     plus Zero-G orbital mode (`gravity = Vec3(0, 0, 0)`).
   - `begin_recovery(mut root_pos, mut root_yaw, mut inertial, target_pose)`
     re-rooting world `(root_pos, root_yaw)` to the settled pelvis with `0.0 mm`
     snap and seamlessly seeding `Inertializer`.
"""
from std.math import sqrt, floor
from flint.core.detmath import det_exp, det_atan2, det_sin, det_cos, PI
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.ik import cross, dot, safe_normalized, angle_axis
from flint.assets.model import NativeModel
from flint.anim.pose import (
    Pose,
    SkeletonInstance,
    Inertializer,
    sample_pose,
    blend_into,
    write_pose,
    det_quat_to_scaled_angle,
    det_quat_from_scaled_angle,
    decompose_trs,
)
from flint.anim.procedural import Placement, DEG, rotation_between, pose_position
from flint.anim.helper_joints import twist_part
from flint.physics.xpbd import ParticleMesh, ChainCollider, LINK_MIN

comptime RAGDOLL_STEP: Float32 = 1.0 / 120.0
comptime RAGDOLL_MAX_STEPS: Int = 12
comptime RAGDOLL_SUBSTEPS: Int = 2
comptime RAGDOLL_BLOB_TAG: UInt32 = fourcc("RAGD")
comptime RAGDOLL_BLOB_VERSION: Int = 1

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


@always_inline
def _wrap_pi(angle: Float32) -> Float32:
    comptime TAU_F32: Float32 = 6.28318530717958647692
    return angle - TAU_F32 * floor((angle + PI) / TAU_F32)


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


def _heading_azimuth(side: Vec3, fallback: Vec3) -> Float32:
    """Azimuth angle `atan2(flat.z, flat.x)` of the horizontal body axis."""
    var flat = Vec3(side.x, 0, side.z)
    if flat.length() < Float32(0.3) * max(Float32(1.0e-6), side.length()):
        flat = Vec3(fallback.x, 0, fallback.z)
    return det_atan2(flat.z, flat.x) if flat.length() > Float32(1.0e-6) else Float32(0.0)


def _points_heading_azimuth(points: List[Vec3], nodes: List[Int]) -> Float32:
    """Extracts horizontal heading azimuth from hip pair `(12, 15)` or spine `(0, 3)`."""
    var has_hips = len(points) > 15 and len(nodes) > 15 and nodes[12] >= 0 and nodes[15] >= 0
    var has_chest = len(points) > 3 and len(nodes) > 3 and nodes[0] >= 0 and nodes[3] >= 0
    var has_spine = len(points) > 1 and len(nodes) > 1 and nodes[0] >= 0 and nodes[1] >= 0
    var side = (points[12] - points[15]) if has_hips else Vec3()
    var fallback = Vec3(0, 0, 1)
    if has_chest:
        fallback = points[3] - points[0]
    elif has_spine:
        fallback = points[1] - points[0]
    if not has_hips:
        side = Vec3(fallback.z, 0.0, -fallback.x)
    return _heading_azimuth(side, fallback)


def heading_free(dirs: List[Vec3], side: Vec3, fallback: Vec3) -> List[Vec3]:
    """The directions turned about +Y so the body's side axis (or, lying on
    its side, the fallback) points along +X."""
    var angle = _heading_azimuth(side, fallback)
    var c = det_cos(angle)
    var s = det_sin(angle)
    var out = List[Vec3]()
    for d in dirs:
        # Turn by +angle about +Y: (x, z) -> (x c + z s, -x s + z c).
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
    var getup_clip_azimuth: Float32
    var getup_clip_pelvis: Vec3
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
        self.getup_clip_azimuth = 0
        self.getup_clip_pelvis = Vec3()
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
            var share = det_exp(-d / max(Float32(0.01), reach))
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

    def settled_pelvis_yaw(self) -> Float32:
        """World-space horizontal yaw (radians about +Y, +Z forward) of the settled ragdoll pelvis."""
        if not self.ready or len(self.nodes) == 0:
            return Float32(0.0)
        return _wrap_pi(-_points_heading_azimuth(self.ragdoll_points(), self.nodes))

    def settled_root_offset(self, placement: Placement = Placement()) -> Vec3:
        """Model-space horizontal offset `Vec3(dx, 0, dz)` from `placement` to the settled ragdoll pelvis."""
        if not self.ready or len(self.particle) == 0 or self.particle[0] < 0:
            return Vec3()
        var p_m = placement.to_model(self.joint_world(0))
        return Vec3(p_m.x, Float32(0.0), p_m.z)

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
            var fade = det_exp(Float32(-0.69314718) * dt / REACT_HALFLIFE)
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
        self.getup_clip_azimuth = 0
        self.getup_clip_pelvis = Vec3()
        if self.getup.clip < 0:
            # Nothing to match: fade straight back to the graph.
            self.getup_length = GETUP_IN + GETUP_OUT
            self.last_match = String("")
        else:
            var left = model.animations[self.getup.clip].duration - self.getup.time
            self.getup_length = max(GETUP_IN + GETUP_OUT, min(GETUP_MAX, left))
            self.last_match = model.animations[self.getup.clip].name
            sample_pose(model, self.getup.clip, self.getup.time, self.clip_pose)
            var matched_pts = self.pose_points(model, self.clip_pose)
            self.getup_clip_azimuth = _points_heading_azimuth(matched_pts, self.nodes)
            if len(matched_pts) > 0:
                self.getup_clip_pelvis = matched_pts[0]
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

            # Align clip root heading & horizontal offset to settled ragdoll pelvis
            # so recovery never snaps 120 degrees or drags across the floor!
            var rag_model_pts = List[Vec3](capacity=len(self.nodes))
            for idx in range(len(self.nodes)):
                rag_model_pts.append(placement.to_model(self.joint_world(idx)))
            var rag_azimuth = _points_heading_azimuth(rag_model_pts, self.nodes)
            var yaw_align = _wrap_pi(self.getup_clip_azimuth - rag_azimuth)
            var q_align = angle_axis(yaw_align, Vec3(0, 1, 0)).normalized()
            var root_offset = self.settled_root_offset(placement)
            var rot_clip_pelvis = q_align.rotate(Vec3(self.getup_clip_pelvis.x, 0.0, self.getup_clip_pelvis.z))

            var align_node = self.root if (self.root >= 0 and self.root < self.clip_pose.count()) else (
                self.nodes[0] if (len(self.nodes) > 0 and self.nodes[0] >= 0 and self.nodes[0] < self.clip_pose.count()) else -1
            )
            if align_node >= 0:
                if align_node == self.root:
                    self.clip_pose.t[align_node] = Vec3(
                        root_offset.x - rot_clip_pelvis.x,
                        self.scratch.t[align_node].y,
                        root_offset.z - rot_clip_pelvis.z,
                    )
                else:
                    self.clip_pose.t[align_node] = Vec3(
                        root_offset.x,
                        self.clip_pose.t[align_node].y,
                        root_offset.z,
                    )
                self.clip_pose.r[align_node] = q_align.multiply(self.clip_pose.r[align_node]).normalized()

            for i in range(min(self.clip_pose.count(), self.scratch.count())):
                self.clip_pose.s[i] = self.scratch.s[i]  # the form's joint scales stay
            blend_into(self.scratch, self.clip_pose, g)
            write_pose(model, self.scratch)
        for i in range(len(self.weight)):
            self.weight[i] = r
        if r > 0:
            self.drive(model, placement)


# ===========================================================================
# Data-Driven 6-DOF Articulated Rigid-Capsule & Active-PD Ragdoll (Flint v4)
# ===========================================================================

def compute_capsule_inertia(mass: Float32, radius: Float32, length: Float32) -> Tuple[Float32, Float32]:
    """Exact principal moments of inertia `(I_axial, I_transverse)` about CoM for a
    uniform-density solid capsule of total `mass`, hemisphere `radius`, and cylinder
    segment `length`."""
    var m = max(Float32(1.0e-4), mass)
    var r = max(Float32(1.0e-3), radius)
    var l = max(Float32(0.0), length)
    var r2 = r * r
    var l2 = l * l
    var v_cyl = PI * r2 * l
    var v_sph = (Float32(4.0) / Float32(3.0)) * PI * r2 * r
    var v_tot = max(Float32(1.0e-9), v_cyl + v_sph)
    var m_cyl = m * (v_cyl / v_tot)
    var m_sph = m * (v_sph / v_tot)

    var i_axial = Float32(0.5) * m_cyl * r2 + Float32(0.4) * m_sph * r2
    var i_trans = (
        m_cyl * (l2 / Float32(12.0) + Float32(0.25) * r2)
        + m_sph * (Float32(0.4) * r2 + Float32(0.25) * l2 + Float32(0.375) * r * l)
    )
    return (max(Float32(1.0e-5), i_axial), max(Float32(1.0e-5), i_trans))


struct RagdollObbCollider(Copyable, ImplicitlyCopyable, Movable):
    """World-space oriented bounding box collider for carrier decks, bulkheads, & cover."""
    var center: Vec3
    var rotation: Quat
    var half_extents: Vec3
    var friction: Float32
    var restitution: Float32

    def __init__(
        out self,
        center: Vec3 = Vec3(),
        rotation: Quat = Quat(),
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        friction: Float32 = 0.45,
        restitution: Float32 = 0.05,
    ):
        self.center = center
        self.rotation = rotation.normalized()
        self.half_extents = half_extents
        self.friction = friction
        self.restitution = restitution


struct ArticulatedLimbDef(Copyable, ImplicitlyCopyable, Movable):
    """Data-driven rigid-capsule definition & anatomical joint limits for one skeleton bone."""
    var bone_node: Int
    var parent_limb: Int
    var mass: Float32
    var radius: Float32
    var length: Float32
    var local_axis: Vec3
    var local_com_offset: Vec3
    var parent_anchor_local: Vec3
    var rest_local_rot: Quat
    # Anatomical joint limits
    var is_hinge: Bool
    var hinge_axis_local: Vec3
    var hinge_min_rad: Float32
    var hinge_max_rad: Float32
    var swing_cone_rad: Float32
    var twist_min_rad: Float32
    var twist_max_rad: Float32
    # Active PD joint torque motor parameters
    var active_stiffness: Float32
    var active_damping: Float32
    var max_torque: Float32

    def __init__(
        out self,
        bone_node: Int = -1,
        parent_limb: Int = -1,
        mass: Float32 = 3.0,
        radius: Float32 = 0.08,
        length: Float32 = 0.35,
        local_axis: Vec3 = Vec3(0, -1, 0),
        parent_anchor_local: Vec3 = Vec3(),
        rest_local_rot: Quat = Quat(),
        is_hinge: Bool = False,
        hinge_axis_local: Vec3 = Vec3(1, 0, 0),
        hinge_min_rad: Float32 = 0.0,
        hinge_max_rad: Float32 = 2.35,
        swing_cone_rad: Float32 = 0.85,
        twist_min_rad: Float32 = -0.55,
        twist_max_rad: Float32 = 0.55,
        active_stiffness: Float32 = 220.0,
        active_damping: Float32 = 24.0,
        max_torque: Float32 = 350.0,
    ):
        var ax = safe_normalized(local_axis, Vec3(0, -1, 0))
        self.bone_node = bone_node
        self.parent_limb = parent_limb
        self.mass = max(Float32(1.0e-3), mass)
        self.radius = max(Float32(0.01), radius)
        self.length = max(Float32(0.02), length)
        self.local_axis = ax
        self.local_com_offset = ax * (Float32(0.5) * self.length)
        self.parent_anchor_local = parent_anchor_local
        self.rest_local_rot = rest_local_rot.normalized()
        self.is_hinge = is_hinge
        self.hinge_axis_local = safe_normalized(hinge_axis_local, Vec3(1, 0, 0))
        self.hinge_min_rad = min(hinge_min_rad, hinge_max_rad)
        self.hinge_max_rad = max(hinge_min_rad, hinge_max_rad)
        self.swing_cone_rad = max(Float32(0.0), swing_cone_rad)
        self.twist_min_rad = min(twist_min_rad, twist_max_rad)
        self.twist_max_rad = max(twist_min_rad, twist_max_rad)
        self.active_stiffness = max(Float32(0.0), active_stiffness)
        self.active_damping = max(Float32(0.0), active_damping)
        self.max_torque = max(Float32(0.0), max_torque)


struct ArticulatedLimbState(Copyable, ImplicitlyCopyable, Movable):
    """Live 6-DOF world-space rigid-body state `(com_pos, rotation, linear_vel, angular_vel)`."""
    var com_pos: Vec3
    var prev_com_pos: Vec3
    var rotation: Quat
    var prev_rotation: Quat
    var linear_vel: Vec3
    var angular_vel: Vec3
    var inv_mass: Float32
    var inv_inertia_axial: Float32
    var inv_inertia_trans: Float32
    var target_local_rot: Quat
    var on_surface: Bool

    def __init__(out self):
        self.com_pos = Vec3()
        self.prev_com_pos = Vec3()
        self.rotation = Quat()
        self.prev_rotation = Quat()
        self.linear_vel = Vec3()
        self.angular_vel = Vec3()
        self.inv_mass = Float32(1.0)
        self.inv_inertia_axial = Float32(1.0)
        self.inv_inertia_trans = Float32(1.0)
        self.target_local_rot = Quat()
        self.on_surface = False

    @always_inline
    def apply_world_inv_inertia(self, vec_world: Vec3, local_axis: Vec3) -> Vec3:
        """Closed-form symmetric positive-definite `I_world^-1 * vec_world` for an
        axially symmetric capsule aligned along `rotation * local_axis`."""
        var axis_w = self.rotation.rotate(local_axis)
        var proj = dot(vec_world, axis_w)
        var v_par = axis_w * proj
        var v_perp = vec_world - v_par
        return v_par * self.inv_inertia_axial + v_perp * self.inv_inertia_trans

    @always_inline
    def joint_pivot_world(self, local_com_offset: Vec3, scale: Float32) -> Vec3:
        return self.com_pos - self.rotation.rotate(local_com_offset * scale)


def clamp_joint_relative_rotation(def_i: ArticulatedLimbDef, q_rel: Quat) -> Tuple[Quat, Float32]:
    """Clamps relative joint rotation `q_rel = q_parent^* * q_child` against
    `def_i`'s anatomical hinge (`[hinge_min_rad, hinge_max_rad]`) or swing-cone
    + twist limits. Returns `(q_rel_clamped, measured_angle_rad)`."""
    var q_delta = def_i.rest_local_rot.conjugate().multiply(q_rel).normalized().shortest()
    if def_i.is_hinge:
        var h_ax = def_i.hinge_axis_local
        var q_hinge = twist_part(q_delta, h_ax).shortest()
        var q_swing = q_delta.multiply(q_hinge.conjugate()).normalized().shortest()
        var sin_h = dot(Vec3(q_hinge.x, q_hinge.y, q_hinge.z), h_ax)
        var raw_angle = Float32(2.0) * det_atan2(sin_h, max(Float32(0.0), q_hinge.w))
        var clamped_angle = max(def_i.hinge_min_rad, min(def_i.hinge_max_rad, raw_angle))

        var swing_vec = det_quat_to_scaled_angle(q_swing)
        var swing_len = swing_vec.length()
        var max_lat = min(def_i.swing_cone_rad, Float32(0.02))
        var q_swing_clamped = Quat()
        if max_lat > Float32(1.0e-5):
            if swing_len > max_lat and swing_len > Float32(1.0e-6):
                swing_vec = swing_vec * (max_lat / swing_len)
            q_swing_clamped = det_quat_from_scaled_angle(swing_vec)
        var q_delta_clamped = q_swing_clamped.multiply(angle_axis(clamped_angle, h_ax)).normalized()
        return (def_i.rest_local_rot.multiply(q_delta_clamped).normalized(), clamped_angle)

    var shaft = def_i.local_axis
    var q_twist = twist_part(q_delta, shaft).shortest()
    var q_swing = q_delta.multiply(q_twist.conjugate()).normalized().shortest()
    var sin_t = dot(Vec3(q_twist.x, q_twist.y, q_twist.z), shaft)
    var twist_angle = Float32(2.0) * det_atan2(sin_t, max(Float32(0.0), q_twist.w))
    var clamped_twist = max(def_i.twist_min_rad, min(def_i.twist_max_rad, twist_angle))

    var swing_vec = det_quat_to_scaled_angle(q_swing)
    var swing_mag = swing_vec.length()
    if swing_mag > def_i.swing_cone_rad and swing_mag > Float32(1.0e-6):
        swing_vec = swing_vec * (def_i.swing_cone_rad / swing_mag)
        swing_mag = def_i.swing_cone_rad
    var q_delta_clamped = det_quat_from_scaled_angle(swing_vec).multiply(angle_axis(clamped_twist, shaft)).normalized()
    return (def_i.rest_local_rot.multiply(q_delta_clamped).normalized(), swing_mag)


def _pose_global_rot(pose: Pose, parents: List[Int], node: Int) -> Quat:
    if node < 0 or node >= pose.count():
        return Quat()
    var rot = Quat()
    var cur = node
    var guard = 0
    while cur >= 0 and cur < pose.count() and guard < 128:
        rot = pose.r[cur].multiply(rot)
        cur = parents[cur] if cur < len(parents) else -1
        guard += 1
    return rot.normalized()


struct ArticulatedRagdoll(Copyable, Movable):
    """Data-driven 6-DOF active-PD & limp ragdoll for humanoids, reverse-joint walkers,
    and variable-geometry mechas (`VF-1 Battloid`, `Regult`, `Glaug`), supporting both
    1G terrain/deck surfaces and Zero-G orbital combat (`gravity = Vec3(0, 0, 0)`)."""
    var limbs: List[ArticulatedLimbDef]
    var states: List[ArticulatedLimbState]
    var obb_colliders: List[RagdollObbCollider]
    var skel_parents: List[Int]
    var skel_topo_order: List[Int]
    var rest_pose: Pose
    var pelvis_rest_global_rot: Quat
    # Environment & solver settings
    var gravity: Vec3
    var has_floor: Bool
    var floor_normal: Vec3
    var floor_offset: Float32
    var floor_friction: Float32
    var linear_damping: Float32
    var angular_damping: Float32
    var scale: Float32
    var solver_iterations: Int
    var joint_compliance: Float32
    # Active-PD motor & root attitude controls
    var motor_scale: Float32
    var root_attitude_stiffness: Float32
    var root_attitude_damping: Float32
    var root_attitude_max_torque: Float32
    var target_root_world_rot: Quat
    var is_limp: Bool
    var is_active: Bool
    var still_seconds: Float32
    var steps_taken: Int

    def __init__(out self):
        self.limbs = List[ArticulatedLimbDef]()
        self.states = List[ArticulatedLimbState]()
        self.obb_colliders = List[RagdollObbCollider]()
        self.skel_parents = List[Int]()
        self.skel_topo_order = List[Int]()
        self.rest_pose = Pose()
        self.pelvis_rest_global_rot = Quat()
        self.gravity = Vec3(0.0, -9.81, 0.0)
        self.has_floor = True
        self.floor_normal = Vec3(0.0, 1.0, 0.0)
        self.floor_offset = Float32(0.0)
        self.floor_friction = Float32(0.55)
        self.linear_damping = Float32(0.45)
        self.angular_damping = Float32(0.85)
        self.scale = Float32(1.0)
        self.solver_iterations = 8
        self.joint_compliance = Float32(0.0)
        self.motor_scale = Float32(1.0)
        self.root_attitude_stiffness = Float32(260.0)
        self.root_attitude_damping = Float32(32.0)
        self.root_attitude_max_torque = Float32(550.0)
        self.target_root_world_rot = Quat()
        self.is_limp = False
        self.is_active = False
        self.still_seconds = Float32(0.0)
        self.steps_taken = 0

    def limb_count(self) -> Int:
        return len(self.limbs)

    def bind_skeleton(mut self, model: NativeModel):
        """Caches the skeleton's topological hierarchy and rest pose for re-rooting & IK."""
        var skel = SkeletonInstance(len(model.nodes))
        skel.reset_to_rest(model)
        self.skel_parents = skel.parents.copy()
        self.skel_topo_order = skel.topo_order.copy()
        self.rest_pose.rest(model)

    def add_limb(mut self, limb_def: ArticulatedLimbDef) -> Int:
        """Appends a configured `ArticulatedLimbDef` (root limb first, parents before children)."""
        var idx = len(self.limbs)
        self.limbs.append(limb_def)
        var st = ArticulatedLimbState()
        st.inv_mass = Float32(1.0) / max(Float32(1.0e-3), limb_def.mass)
        var inertia = compute_capsule_inertia(limb_def.mass, limb_def.radius * self.scale, limb_def.length * self.scale)
        st.inv_inertia_axial = Float32(1.0) / inertia[0]
        st.inv_inertia_trans = Float32(1.0) / inertia[1]
        st.target_local_rot = limb_def.rest_local_rot
        self.states.append(st)
        if idx == 0 and limb_def.bone_node >= 0 and limb_def.bone_node < self.rest_pose.count():
            self.pelvis_rest_global_rot = _pose_global_rot(self.rest_pose, self.skel_parents, limb_def.bone_node)
        return idx

    def add_limb_from_model(
        mut self,
        model: NativeModel,
        bone_node: Int,
        parent_limb: Int,
        mass: Float32,
        radius: Float32,
        length: Float32 = 0.0,
        local_axis: Vec3 = Vec3(),
        is_hinge: Bool = False,
        hinge_axis_local: Vec3 = Vec3(1, 0, 0),
        hinge_min_rad: Float32 = 0.0,
        hinge_max_rad: Float32 = 2.35,
        swing_cone_rad: Float32 = 0.85,
        twist_min_rad: Float32 = -0.55,
        twist_max_rad: Float32 = 0.55,
        active_stiffness: Float32 = 220.0,
        active_damping: Float32 = 24.0,
        max_torque: Float32 = 350.0,
    ) -> Int:
        """Automatically derives `parent_anchor_local`, `rest_local_rot`, and (when omitted)
        `local_axis` + `length` from `model`'s rest hierarchy."""
        if len(self.skel_parents) != len(model.nodes):
            self.bind_skeleton(model)

        var q_u_rest = _pose_global_rot(self.rest_pose, self.skel_parents, bone_node)
        var p_u_rest = pose_position(self.rest_pose, self.skel_parents, bone_node)

        var anchor_local = Vec3()
        var rest_rel = q_u_rest
        if parent_limb >= 0 and parent_limb < len(self.limbs):
            var p_node = self.limbs[parent_limb].bone_node
            var q_p_rest = _pose_global_rot(self.rest_pose, self.skel_parents, p_node)
            var p_p_rest = pose_position(self.rest_pose, self.skel_parents, p_node)
            anchor_local = q_p_rest.conjugate().rotate(p_u_rest - p_p_rest)
            rest_rel = q_p_rest.conjugate().multiply(q_u_rest).normalized()

        var ax = local_axis
        var seg_len = length
        if (ax.length_squared() < Float32(1.0e-6) or seg_len <= Float32(0.0)) and bone_node >= 0 and bone_node < len(model.nodes):
            if len(model.nodes[bone_node].children) > 0:
                var ch = model.nodes[bone_node].children[0]
                if ch >= 0 and ch < len(model.nodes):
                    var child_vec = model.nodes[ch].base_translation
                    if child_vec.length() > Float32(1.0e-4):
                        if ax.length_squared() < Float32(1.0e-6):
                            ax = child_vec.normalized()
                        if seg_len <= Float32(0.0):
                            seg_len = child_vec.length()
        if ax.length_squared() < Float32(1.0e-6):
            ax = Vec3(0, 1, 0) if parent_limb < 0 else Vec3(0, -1, 0)
        if seg_len <= Float32(0.0):
            seg_len = Float32(0.30)

        var def_in = ArticulatedLimbDef(
            bone_node=bone_node,
            parent_limb=parent_limb,
            mass=mass,
            radius=radius,
            length=seg_len,
            local_axis=ax,
            parent_anchor_local=anchor_local,
            rest_local_rot=rest_rel,
            is_hinge=is_hinge,
            hinge_axis_local=hinge_axis_local,
            hinge_min_rad=hinge_min_rad,
            hinge_max_rad=hinge_max_rad,
            swing_cone_rad=swing_cone_rad,
            twist_min_rad=twist_min_rad,
            twist_max_rad=twist_max_rad,
            active_stiffness=active_stiffness,
            active_damping=active_damping,
            max_torque=max_torque,
        )
        return self.add_limb(def_in)

    def set_zero_g(mut self, enabled: Bool):
        """Switches between Zero-G orbital space (`gravity = Vec3(0,0,0)`, no floor)
        and standard 1G surface physics."""
        if enabled:
            self.gravity = Vec3(0.0, 0.0, 0.0)
            self.has_floor = False
        else:
            self.gravity = Vec3(0.0, -9.81, 0.0)
            self.has_floor = True

    def set_surface_plane(mut self, normal: Vec3, offset: Float32, friction: Float32 = 0.55):
        self.has_floor = True
        self.floor_normal = safe_normalized(normal, Vec3(0.0, 1.0, 0.0))
        self.floor_offset = offset
        self.floor_friction = max(Float32(0.0), min(Float32(1.0), friction))

    def add_obb_collider(mut self, collider: RagdollObbCollider):
        self.obb_colliders.append(collider)

    def clear_obb_colliders(mut self):
        self.obb_colliders.clear()

    def go_limp(mut self):
        """Disables Active-PD motors so the ragdoll collapses freely under gravity/inertia."""
        self.is_limp = True
        self.is_active = True
        self.motor_scale = Float32(0.0)
        self.still_seconds = Float32(0.0)

    def set_active_pd(mut self, motor_scale: Float32 = 1.0):
        """Enables Active-PD joint torque tracking (`motor_scale` scales stiffness & max_torque)."""
        self.is_limp = False
        self.is_active = True
        self.motor_scale = max(Float32(0.0), motor_scale)
        self.still_seconds = Float32(0.0)

    def limb_for_bone(self, bone_node: Int) -> Int:
        """Finds the limb index mapped to `bone_node` (or nearest mapped ancestor bone)."""
        var cur = bone_node
        var guard = 0
        while cur >= 0 and guard < 128:
            for i in range(len(self.limbs)):
                if self.limbs[i].bone_node == cur:
                    return i
            if cur < len(self.skel_parents):
                cur = self.skel_parents[cur]
            else:
                break
            guard += 1
        return -1

    def _refresh_inertias(mut self):
        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            self.states[i].inv_mass = Float32(1.0) / max(Float32(1.0e-3), def_i.mass)
            var inertia = compute_capsule_inertia(def_i.mass, def_i.radius * self.scale, def_i.length * self.scale)
            self.states[i].inv_inertia_axial = Float32(1.0) / inertia[0]
            self.states[i].inv_inertia_trans = Float32(1.0) / inertia[1]

    def set_target_pose(mut self, anim_pose: Pose, root_world_yaw_rad: Float32 = 0.0):
        """Updates every limb's desired Active-PD target joint rotation from `anim_pose`."""
        var q_world_yaw = angle_axis(root_world_yaw_rad, Vec3(0.0, 1.0, 0.0))
        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var q_u = _pose_global_rot(anim_pose, self.skel_parents, def_i.bone_node)
            if def_i.parent_limb < 0:
                self.target_root_world_rot = q_world_yaw.multiply(q_u).normalized()
                self.states[i].target_local_rot = q_u
            else:
                var p_node = self.limbs[def_i.parent_limb].bone_node
                var q_p = _pose_global_rot(anim_pose, self.skel_parents, p_node)
                self.states[i].target_local_rot = q_p.conjugate().multiply(q_u).normalized()

    def handoff_from_poses(
        mut self,
        skel_prev: SkeletonInstance,
        placement_prev: Placement,
        skel_curr: SkeletonInstance,
        placement_curr: Placement,
        model: NativeModel,
        dt: Float32,
    ):
        """Exact 6-DOF linear & angular velocity handoff `(v_i, omega_i)` from two
        consecutive `SkeletonInstance` poses `(n-1)` and `(n)` separated by `dt`."""
        if len(self.skel_parents) != len(model.nodes):
            self.bind_skeleton(model)
        self.scale = max(
            Float32(0.05),
            (abs(placement_curr.scale.x) + abs(placement_curr.scale.y) + abs(placement_curr.scale.z)) / Float32(3.0),
        )
        var scale_prev = max(
            Float32(0.05),
            (abs(placement_prev.scale.x) + abs(placement_prev.scale.y) + abs(placement_prev.scale.z)) / Float32(3.0),
        )
        self._refresh_inertias()

        var q_place_prev = angle_axis(placement_prev.yaw * DEG, Vec3(0.0, 1.0, 0.0))
        var q_place_curr = angle_axis(placement_curr.yaw * DEG, Vec3(0.0, 1.0, 0.0))
        var inv_dt = Float32(1.0) / dt if dt > Float32(1.0e-6) else Float32(0.0)

        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var node = def_i.bone_node
            var q_prev = q_place_prev.multiply(skel_prev.global_rotation(node)).normalized()
            var q_curr = q_place_curr.multiply(skel_curr.global_rotation(node)).normalized()

            var piv_prev = placement_prev.to_world(skel_prev.global_position(node))
            var piv_curr = placement_curr.to_world(skel_curr.global_position(node))
            var com_prev = piv_prev + q_prev.rotate(def_i.local_com_offset * scale_prev)
            var com_curr = piv_curr + q_curr.rotate(def_i.local_com_offset * self.scale)

            self.states[i].com_pos = com_curr
            self.states[i].prev_com_pos = com_prev
            self.states[i].rotation = q_curr
            self.states[i].prev_rotation = q_prev
            if inv_dt > Float32(0.0):
                self.states[i].linear_vel = (com_curr - com_prev) * inv_dt
                # Quat.to_scaled_angle / det_quat_to_scaled_angle already returns 2 * log(dq)
                var dq = q_curr.multiply(q_prev.conjugate()).normalized()
                self.states[i].angular_vel = det_quat_to_scaled_angle(dq) * inv_dt
            else:
                self.states[i].linear_vel = Vec3()
                self.states[i].angular_vel = Vec3()

        self.set_target_pose(skel_curr.pose, placement_curr.yaw * DEG)
        self.is_active = True
        self.still_seconds = Float32(0.0)

    def handoff_from_skeleton(
        mut self,
        skel_curr: SkeletonInstance,
        placement_curr: Placement,
        model: NativeModel,
        dt: Float32,
        placement_prev: Placement = Placement(Vec3(1.0e30, 0, 0)),
    ):
        """6-DOF velocity handoff directly from a single `SkeletonInstance` that called
        `snapshot_prev_matrices()` prior to advancing the current frame."""
        if len(self.skel_parents) != len(model.nodes):
            self.bind_skeleton(model)
        var p_prev = placement_curr if placement_prev.position.x > Float32(1.0e29) else placement_prev
        self.scale = max(
            Float32(0.05),
            (abs(placement_curr.scale.x) + abs(placement_curr.scale.y) + abs(placement_curr.scale.z)) / Float32(3.0),
        )
        var scale_prev = max(
            Float32(0.05),
            (abs(p_prev.scale.x) + abs(p_prev.scale.y) + abs(p_prev.scale.z)) / Float32(3.0),
        )
        self._refresh_inertias()

        var q_place_prev = angle_axis(p_prev.yaw * DEG, Vec3(0.0, 1.0, 0.0))
        var q_place_curr = angle_axis(placement_curr.yaw * DEG, Vec3(0.0, 1.0, 0.0))
        var inv_dt = Float32(1.0) / dt if (dt > Float32(1.0e-6) and skel_curr.has_prev) else Float32(0.0)

        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var node = def_i.bone_node
            var q_curr = q_place_curr.multiply(skel_curr.global_rotation(node)).normalized()
            var piv_curr = placement_curr.to_world(skel_curr.global_position(node))
            var com_curr = piv_curr + q_curr.rotate(def_i.local_com_offset * self.scale)

            var q_prev = q_curr
            var com_prev = com_curr
            if skel_curr.has_prev and node >= 0 and node < len(skel_curr.prev_world_matrices):
                var prev_trs = decompose_trs(skel_curr.prev_world_matrices[node])
                q_prev = q_place_prev.multiply(prev_trs[1]).normalized()
                var piv_prev = p_prev.to_world(prev_trs[0])
                com_prev = piv_prev + q_prev.rotate(def_i.local_com_offset * scale_prev)

            self.states[i].com_pos = com_curr
            self.states[i].prev_com_pos = com_prev
            self.states[i].rotation = q_curr
            self.states[i].prev_rotation = q_prev
            if inv_dt > Float32(0.0):
                self.states[i].linear_vel = (com_curr - com_prev) * inv_dt
                var dq = q_curr.multiply(q_prev.conjugate()).normalized()
                self.states[i].angular_vel = det_quat_to_scaled_angle(dq) * inv_dt
            else:
                self.states[i].linear_vel = Vec3()
                self.states[i].angular_vel = Vec3()

        self.set_target_pose(skel_curr.pose, placement_curr.yaw * DEG)
        self.is_active = True
        self.still_seconds = Float32(0.0)

    def apply_impulse_at_bone(mut self, bone_node: Int, world_impulse: Vec3, world_point: Vec3):
        """Applies a 6-DOF linear + torque impulse `world_impulse` (N*s) at `world_point`
        to the rigid capsule mapped to `bone_node` (or nearest ancestor limb)."""
        var idx = self.limb_for_bone(bone_node)
        if idx < 0:
            return
        var r = world_point - self.states[idx].com_pos
        var torque_impulse = r.cross(world_impulse)
        self.states[idx].linear_vel = self.states[idx].linear_vel + world_impulse * self.states[idx].inv_mass
        self.states[idx].angular_vel = self.states[idx].angular_vel + self.states[idx].apply_world_inv_inertia(
            torque_impulse, self.limbs[idx].local_axis
        )
        self.still_seconds = Float32(0.0)

    def apply_impulse_at_bone(mut self, bone_node: Int, world_impulse: Vec3):
        """Applies `world_impulse` slightly distal of the limb's CoM for realistic linear + spin stagger."""
        var idx = self.limb_for_bone(bone_node)
        if idx < 0:
            return
        var hit_pt = self.states[idx].com_pos + self.states[idx].rotation.rotate(
            self.limbs[idx].local_axis * (Float32(0.25) * self.limbs[idx].length * self.scale)
        )
        self.apply_impulse_at_bone(bone_node, world_impulse, hit_pt)

    def _apply_active_pd_motors(mut self, h: Float32):
        if self.is_limp or self.motor_scale <= Float32(1.0e-5):
            return

        # Optional root attitude gyro/balance PD controller when standing/active
        if len(self.limbs) > 0 and self.root_attitude_stiffness > Float32(0.0):
            var kp_r = self.root_attitude_stiffness * self.motor_scale
            var kd_r = self.root_attitude_damping * self.motor_scale
            var max_tau_r = self.root_attitude_max_torque * self.motor_scale
            var err_r = det_quat_to_scaled_angle(
                self.target_root_world_rot.multiply(self.states[0].rotation.conjugate())
            )
            var w_r = self.states[0].angular_vel
            var denom_r = Float32(1.0) + h * self.states[0].inv_inertia_trans * (kd_r + h * kp_r)
            var tau_r = (err_r * kp_r - w_r * (kd_r + h * kp_r)) * (Float32(1.0) / denom_r)
            var tau_r_len = tau_r.length()
            if tau_r_len > max_tau_r and tau_r_len > Float32(1.0e-6):
                tau_r = tau_r * (max_tau_r / tau_r_len)
            self.states[0].angular_vel = self.states[0].angular_vel + self.states[0].apply_world_inv_inertia(
                tau_r * h, self.limbs[0].local_axis
            )

        # Internal joint PD torque motors (strictly equal-and-opposite action-reaction!)
        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var p = def_i.parent_limb
            if p < 0 or p >= len(self.limbs):
                continue
            var kp = def_i.active_stiffness * self.motor_scale
            var kd = def_i.active_damping * self.motor_scale
            var max_tau = def_i.max_torque * self.motor_scale
            if (kp <= Float32(0.0) and kd <= Float32(0.0)) or max_tau <= Float32(0.0):
                continue

            var q_goal_w = self.states[p].rotation.multiply(self.states[i].target_local_rot).normalized()
            var err_w = det_quat_to_scaled_angle(q_goal_w.multiply(self.states[i].rotation.conjugate()))
            var w_rel = self.states[i].angular_vel - self.states[p].angular_vel
            var inv_i_eff = self.states[i].inv_inertia_trans + self.states[p].inv_inertia_trans
            var denom = Float32(1.0) + h * inv_i_eff * (kd + h * kp)
            var tau_w = (err_w * kp - w_rel * (kd + h * kp)) * (Float32(1.0) / denom)
            var tau_len = tau_w.length()
            if tau_len > max_tau and tau_len > Float32(1.0e-6):
                tau_w = tau_w * (max_tau / tau_len)

            var j_ang = tau_w * h
            self.states[i].angular_vel = self.states[i].angular_vel + self.states[i].apply_world_inv_inertia(
                j_ang, def_i.local_axis
            )
            self.states[p].angular_vel = self.states[p].angular_vel - self.states[p].apply_world_inv_inertia(
                j_ang, self.limbs[p].local_axis
            )

    def _push_capsule_point_contact(mut self, idx: Int, contact_pt: Vec3, normal: Vec3, penetration: Float32):
        """6-DOF rigid-body positional push-out along `normal` by `penetration > 0`."""
        if penetration <= Float32(0.0):
            return
        var r_c = contact_pt - self.states[idx].com_pos
        var rn = r_c.cross(normal)
        var i_inv_rn = self.states[idx].apply_world_inv_inertia(rn, self.limbs[idx].local_axis)
        var w_eff = self.states[idx].inv_mass + dot(rn, i_inv_rn)
        if w_eff <= Float32(1.0e-9):
            return
        var j_mag = penetration / w_eff
        var impulse_p = normal * j_mag
        self.states[idx].com_pos = self.states[idx].com_pos + impulse_p * self.states[idx].inv_mass
        var d_rot = self.states[idx].apply_world_inv_inertia(r_c.cross(impulse_p), self.limbs[idx].local_axis)
        self.states[idx].rotation = det_quat_from_scaled_angle(d_rot).multiply(self.states[idx].rotation).normalized()
        self.states[idx].on_surface = True

    def _solve_collisions(mut self):
        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var half_seg = def_i.local_axis * (Float32(0.5) * def_i.length * self.scale)
            var cap_r = def_i.radius * self.scale
            var world_half = self.states[i].rotation.rotate(half_seg)
            var end_a = self.states[i].com_pos - world_half
            var end_b = self.states[i].com_pos + world_half

            # 1. Surface plane (terrain / warship deck)
            if self.has_floor:
                var n = self.floor_normal
                var dist_a = (dot(n, end_a) - self.floor_offset) - cap_r
                if dist_a < Float32(0.0):
                    self._push_capsule_point_contact(i, end_a - n * cap_r, n, -dist_a)
                    world_half = self.states[i].rotation.rotate(half_seg)
                    end_b = self.states[i].com_pos + world_half
                var dist_b = (dot(n, end_b) - self.floor_offset) - cap_r
                if dist_b < Float32(0.0):
                    self._push_capsule_point_contact(i, end_b - n * cap_r, n, -dist_b)

            # 2. World OBB colliders
            for c_idx in range(len(self.obb_colliders)):
                var obb = self.obb_colliders[c_idx]
                var inv_q = obb.rotation.conjugate()
                for s_pt in range(3):
                    world_half = self.states[i].rotation.rotate(half_seg)
                    var sample_w = self.states[i].com_pos
                    if s_pt == 0:
                        sample_w = self.states[i].com_pos - world_half
                    elif s_pt == 2:
                        sample_w = self.states[i].com_pos + world_half
                    var p_loc = inv_q.rotate(sample_w - obb.center)
                    var clamped = Vec3(
                        max(-obb.half_extents.x, min(obb.half_extents.x, p_loc.x)),
                        max(-obb.half_extents.y, min(obb.half_extents.y, p_loc.y)),
                        max(-obb.half_extents.z, min(obb.half_extents.z, p_loc.z)),
                    )
                    var diff_loc = p_loc - clamped
                    var d_len = diff_loc.length()
                    if d_len > Float32(1.0e-6):
                        if d_len < cap_r:
                            var n_w = obb.rotation.rotate(diff_loc * (Float32(1.0) / d_len))
                            var pen = cap_r - d_len
                            self._push_capsule_point_contact(i, sample_w - n_w * d_len, n_w, pen)
                    else:
                        # Sphere center inside OBB: push out nearest face
                        var dx = obb.half_extents.x - abs(p_loc.x)
                        var dy = obb.half_extents.y - abs(p_loc.y)
                        var dz = obb.half_extents.z - abs(p_loc.z)
                        var n_loc: Vec3
                        var face_pen: Float32
                        if dx <= dy and dx <= dz:
                            n_loc = Vec3(Float32(1.0) if p_loc.x >= Float32(0.0) else Float32(-1.0), 0, 0)
                            face_pen = dx + cap_r
                        elif dy <= dz:
                            n_loc = Vec3(0, Float32(1.0) if p_loc.y >= Float32(0.0) else Float32(-1.0), 0)
                            face_pen = dy + cap_r
                        else:
                            n_loc = Vec3(0, 0, Float32(1.0) if p_loc.z >= Float32(0.0) else Float32(-1.0))
                            face_pen = dz + cap_r
                        var n_w = obb.rotation.rotate(n_loc)
                        self._push_capsule_point_contact(i, sample_w, n_w, face_pen)

    def _solve_joint_constraints(mut self, h: Float32):
        var alpha_h = self.joint_compliance / max(Float32(1.0e-8), h * h)
        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var p = def_i.parent_limb
            if p < 0 or p >= len(self.limbs):
                continue
            var def_p = self.limbs[p]

            # 1. Anatomical hinge / swing-cone / twist angular limit projection
            var q_rel = self.states[p].rotation.conjugate().multiply(self.states[i].rotation).normalized()
            var clamped_res = clamp_joint_relative_rotation(def_i, q_rel)
            var q_goal_w = self.states[p].rotation.multiply(clamped_res[0]).normalized()
            var ang_err = det_quat_to_scaled_angle(q_goal_w.multiply(self.states[i].rotation.conjugate()))
            var err_mag = ang_err.length()
            if err_mag > Float32(1.0e-5):
                var u = ang_err * (Float32(1.0) / err_mag)
                var wp = dot(u, self.states[p].apply_world_inv_inertia(u, def_p.local_axis))
                var wi = dot(u, self.states[i].apply_world_inv_inertia(u, def_i.local_axis))
                var w_sum = wp + wi
                if w_sum > Float32(1.0e-9):
                    var dp = ang_err * (-wp / w_sum)
                    var di = ang_err * (wi / w_sum)
                    self.states[p].rotation = det_quat_from_scaled_angle(dp).multiply(self.states[p].rotation).normalized()
                    self.states[i].rotation = det_quat_from_scaled_angle(di).multiply(self.states[i].rotation).normalized()

            # 2. 6-DOF spherical ball-socket positional attachment projection
            var r_p_local = (def_i.parent_anchor_local - def_p.local_com_offset) * self.scale
            var r_i_local = (-def_i.local_com_offset) * self.scale
            var r_p = self.states[p].rotation.rotate(r_p_local)
            var r_i = self.states[i].rotation.rotate(r_i_local)
            var anchor_p = self.states[p].com_pos + r_p
            var anchor_i = self.states[i].com_pos + r_i
            var c_err = anchor_i - anchor_p
            var c_len = c_err.length()
            if c_len > Float32(1.0e-6):
                var n_dir = c_err * (Float32(1.0) / c_len)
                var rpxn = r_p.cross(n_dir)
                var rixn = r_i.cross(n_dir)
                var wp_pos = self.states[p].inv_mass + dot(rpxn, self.states[p].apply_world_inv_inertia(rpxn, def_p.local_axis))
                var wi_pos = self.states[i].inv_mass + dot(rixn, self.states[i].apply_world_inv_inertia(rixn, def_i.local_axis))
                var denom = wp_pos + wi_pos + alpha_h
                if denom > Float32(1.0e-9):
                    var lam = -c_len / denom
                    var j_vec = n_dir * lam
                    self.states[p].com_pos = self.states[p].com_pos - j_vec * self.states[p].inv_mass
                    self.states[i].com_pos = self.states[i].com_pos + j_vec * self.states[i].inv_mass
                    var dp_rot = self.states[p].apply_world_inv_inertia(-(r_p.cross(j_vec)), def_p.local_axis)
                    var di_rot = self.states[i].apply_world_inv_inertia(r_i.cross(j_vec), def_i.local_axis)
                    self.states[p].rotation = det_quat_from_scaled_angle(dp_rot).multiply(self.states[p].rotation).normalized()
                    self.states[i].rotation = det_quat_from_scaled_angle(di_rot).multiply(self.states[i].rotation).normalized()

    def _enforce_exact_hierarchy_limits(mut self):
        """Final top-down pass ensuring zero residual knee/elbow hyperextension or pivot gap
        while strictly preserving the articulated system's total Center of Mass."""
        var com_before = self.total_center_of_mass()
        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var p = def_i.parent_limb
            if p < 0 or p >= len(self.limbs):
                continue
            var def_p = self.limbs[p]
            var q_rel = self.states[p].rotation.conjugate().multiply(self.states[i].rotation).normalized()
            var clamped_res = clamp_joint_relative_rotation(def_i, q_rel)
            self.states[i].rotation = self.states[p].rotation.multiply(clamped_res[0]).normalized()
            var anchor_p = self.states[p].com_pos + self.states[p].rotation.rotate(
                (def_i.parent_anchor_local - def_p.local_com_offset) * self.scale
            )
            self.states[i].com_pos = anchor_p + self.states[i].rotation.rotate(def_i.local_com_offset * self.scale)
        var com_shift = com_before - self.total_center_of_mass()
        for i in range(len(self.limbs)):
            self.states[i].com_pos = self.states[i].com_pos + com_shift

    def step(mut self, dt: Float32, substeps: Int = 4):
        """Advances 6-DOF articulated rigid-capsule physics by `dt` in `substeps` steps."""
        if dt <= Float32(0.0) or len(self.limbs) == 0:
            return
        var n_sub = max(1, substeps)
        var h = dt / Float32(n_sub)
        var inv_h = Float32(1.0) / h

        for _ in range(n_sub):
            self._apply_active_pd_motors(h)

            var lin_keep = max(Float32(0.0), Float32(1.0) - self.linear_damping * h)
            var ang_keep = max(Float32(0.0), Float32(1.0) - self.angular_damping * h)
            for i in range(len(self.limbs)):
                self.states[i].prev_com_pos = self.states[i].com_pos
                self.states[i].prev_rotation = self.states[i].rotation
                self.states[i].on_surface = False

                var v = (self.states[i].linear_vel + self.gravity * h) * lin_keep
                var w = self.states[i].angular_vel * ang_keep
                self.states[i].linear_vel = v
                self.states[i].angular_vel = w
                self.states[i].com_pos = self.states[i].com_pos + v * h
                self.states[i].rotation.integrate(w, h)

            for _ in range(self.solver_iterations):
                self._solve_collisions()
                self._solve_joint_constraints(h)
            self._enforce_exact_hierarchy_limits()
            if self.has_floor:
                var max_pen = Float32(0.0)
                var n_f = self.floor_normal
                for i in range(len(self.limbs)):
                    var def_i = self.limbs[i]
                    var half_seg = self.states[i].rotation.rotate(
                        def_i.local_axis * (Float32(0.5) * def_i.length * self.scale)
                    )
                    var cap_r = def_i.radius * self.scale
                    var d_a = (dot(n_f, self.states[i].com_pos - half_seg) - self.floor_offset) - cap_r
                    var d_b = (dot(n_f, self.states[i].com_pos + half_seg) - self.floor_offset) - cap_r
                    if d_a < Float32(0.0):
                        max_pen = max(max_pen, -d_a)
                        self.states[i].on_surface = True
                    if d_b < Float32(0.0):
                        max_pen = max(max_pen, -d_b)
                        self.states[i].on_surface = True
                if max_pen > Float32(0.0):
                    var lift = n_f * max_pen
                    for i in range(len(self.limbs)):
                        self.states[i].com_pos = self.states[i].com_pos + lift

            # Derive constrained 6-DOF velocities & apply surface contact friction
            for i in range(len(self.limbs)):
                var v_new = (self.states[i].com_pos - self.states[i].prev_com_pos) * inv_h
                var dq = self.states[i].rotation.multiply(self.states[i].prev_rotation.conjugate()).normalized()
                var w_new = det_quat_to_scaled_angle(dq) * inv_h
                if self.states[i].on_surface:
                    var n = self.floor_normal
                    var vn = dot(v_new, n)
                    var vt = v_new - n * vn
                    var keep_t = max(Float32(0.0), Float32(1.0) - self.floor_friction * Float32(0.25))
                    v_new = vt * keep_t + n * max(Float32(0.0), vn)
                    w_new = w_new * max(Float32(0.0), Float32(1.0) - self.floor_friction * Float32(0.15))
                self.states[i].linear_vel = v_new
                self.states[i].angular_vel = w_new
            self.steps_taken += 1

        if self.mean_speed() < SETTLE_SPEED * self.scale:
            self.still_seconds += dt
        else:
            self.still_seconds = Float32(0.0)

    def joint_hinge_angle(self, limb_idx: Int) -> Float32:
        """Signed hinge angle (radians) of `limb_idx` relative to its parent limb's rest frame."""
        if limb_idx < 0 or limb_idx >= len(self.limbs):
            return Float32(0.0)
        var def_i = self.limbs[limb_idx]
        var p = def_i.parent_limb
        if p < 0 or p >= len(self.limbs):
            return Float32(0.0)
        var q_rel = self.states[p].rotation.conjugate().multiply(self.states[limb_idx].rotation).normalized()
        var q_delta = def_i.rest_local_rot.conjugate().multiply(q_rel).normalized().shortest()
        var q_hinge = twist_part(q_delta, def_i.hinge_axis_local).shortest()
        var sin_h = dot(Vec3(q_hinge.x, q_hinge.y, q_hinge.z), def_i.hinge_axis_local)
        return Float32(2.0) * det_atan2(sin_h, max(Float32(0.0), q_hinge.w))

    def mean_speed(self) -> Float32:
        var n = len(self.states)
        if n == 0:
            return Float32(0.0)
        var total = Float32(0.0)
        for i in range(n):
            total += self.states[i].linear_vel.length()
        return total / Float32(n)

    def settled(self) -> Bool:
        return self.still_seconds >= SETTLE_SECONDS

    def total_mass(self) -> Float32:
        var m = Float32(0.0)
        for i in range(len(self.limbs)):
            m += self.limbs[i].mass
        return m

    def total_center_of_mass(self) -> Vec3:
        var m_sum = Float32(0.0)
        var weighted = Vec3()
        for i in range(len(self.limbs)):
            var m = self.limbs[i].mass
            weighted = weighted + self.states[i].com_pos * m
            m_sum += m
        if m_sum <= Float32(1.0e-6):
            return Vec3()
        return weighted * (Float32(1.0) / m_sum)

    def total_linear_momentum(self) -> Vec3:
        var p = Vec3()
        for i in range(len(self.limbs)):
            p = p + self.states[i].linear_vel * self.limbs[i].mass
        return p

    def pelvis_world_position(self) -> Vec3:
        """World-space joint pivot of the root/pelvis limb (`limbs[0]`)."""
        if len(self.limbs) == 0:
            return Vec3()
        return self.states[0].joint_pivot_world(self.limbs[0].local_com_offset, self.scale)

    def pelvis_world_yaw(self) -> Float32:
        """Singularity-free world horizontal yaw (radians about +Y, +Z forward) of the
        settled or live root/pelvis limb across upright, prone, supine, and side poses."""
        if len(self.limbs) == 0:
            return Float32(0.0)
        var rest_inv = self.pelvis_rest_global_rot.conjugate()
        var local_fwd = rest_inv.rotate(Vec3(0.0, 0.0, 1.0))
        var local_left = rest_inv.rotate(Vec3(1.0, 0.0, 0.0))
        var q_w = self.states[0].rotation
        var w_fwd = q_w.rotate(local_fwd)
        var w_left = q_w.rotate(local_left)
        var h_fwd = Vec3(w_fwd.x, 0.0, w_fwd.z)
        if h_fwd.length_squared() >= Float32(0.16):
            return det_atan2(h_fwd.x, h_fwd.z)
        var h_from_left = Vec3(-w_left.z, 0.0, w_left.x)
        if h_from_left.length_squared() > Float32(1.0e-6):
            return det_atan2(h_from_left.x, h_from_left.z)
        return Float32(0.0)

    def write_to_skeleton(
        self,
        mut skel: SkeletonInstance,
        model: NativeModel,
        placement: Placement,
    ):
        """Writes all simulated 6-DOF limb transforms back into `skel.pose` & `world_matrices`."""
        skel.ensure_topology(model)
        if len(self.limbs) == 0:
            return
        var q_place = angle_axis(placement.yaw * DEG, Vec3(0.0, 1.0, 0.0))
        var q_place_inv = q_place.conjugate()

        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var u = def_i.bone_node
            if u < 0 or u >= skel.node_count:
                continue
            var pivot_w = self.states[i].joint_pivot_world(def_i.local_com_offset, self.scale)
            var pivot_m = placement.to_model(pivot_w)
            var rot_m = q_place_inv.multiply(self.states[i].rotation).normalized()

            var p_node = skel.parents[u]
            if p_node < 0:
                if def_i.parent_limb < 0:
                    skel.pose.t[u] = pivot_m
                skel.pose.r[u] = rot_m
            else:
                var gp_rot = skel.global_rotation(p_node)
                var gp_pos = skel.global_position(p_node)
                if def_i.parent_limb < 0:
                    skel.pose.t[u] = gp_rot.conjugate().rotate(pivot_m - gp_pos)
                skel.pose.r[u] = gp_rot.conjugate().multiply(rot_m).normalized()
            skel.update_matrices(model)

    def pose_in_frame(self, root_pos: Vec3, root_yaw: Float32, base_pose: Pose) -> Pose:
        """Reconstructs a full `Pose` matching the current world ragdoll state when
        rendered under `(root_pos, root_yaw)` at `self.scale`."""
        var out = Pose()
        out.assign(base_pose)
        var count = out.count()
        if count == 0 or len(self.limbs) == 0:
            return out^

        var q_root = angle_axis(root_yaw, Vec3(0.0, 1.0, 0.0))
        var q_root_inv = q_root.conjugate()
        var inv_scale = Float32(1.0) / max(Float32(0.01), self.scale)

        for i in range(len(self.limbs)):
            var def_i = self.limbs[i]
            var u = def_i.bone_node
            if u < 0 or u >= count:
                continue
            var pivot_w = self.states[i].joint_pivot_world(def_i.local_com_offset, self.scale)
            var pivot_m = q_root_inv.rotate(pivot_w - root_pos) * inv_scale
            var rot_m = q_root_inv.multiply(self.states[i].rotation).normalized()

            var p_node = self.skel_parents[u] if u < len(self.skel_parents) else -1
            if p_node < 0:
                if def_i.parent_limb < 0:
                    out.t[u] = pivot_m
                out.r[u] = rot_m
            else:
                var gp_rot = _pose_global_rot(out, self.skel_parents, p_node)
                var gp_pos = pose_position(out, self.skel_parents, p_node)
                if def_i.parent_limb < 0:
                    out.t[u] = gp_rot.conjugate().rotate(pivot_m - gp_pos)
                out.r[u] = gp_rot.conjugate().multiply(rot_m).normalized()
        return out^

    def begin_recovery(
        mut self,
        mut root_pos: Vec3,
        mut root_yaw: Float32,
        mut inertial: Inertializer,
        target_pose: Pose,
    ):
        """World Pelvis Re-Rooting + Inertialized Getup Recovery:
        1. Updates the entity's world `(root_pos, root_yaw)` to the settled ragdoll
           pelvis (`0.0 mm` snap/drag!).
        2. Expresses the live ragdoll skeleton pose in the new `(root_pos, root_yaw)`
           coordinate frame and seeds `inertial` (`out_t/r/s`, `off_t/r/s`, Bollo
           scalar-axis `rot_axis/rot_ang/rot_vel`) so subsequent `inertial.apply(...)`
           calls blend seamlessly from the exact settled ragdoll pose onto `target_pose`.
        """
        if len(self.limbs) == 0 or target_pose.count() == 0:
            return

        var pelvis_w = self.pelvis_world_position()
        var new_yaw = self.pelvis_world_yaw()
        var q_yaw = angle_axis(new_yaw, Vec3(0.0, 1.0, 0.0))

        var pelvis_node = self.limbs[0].bone_node
        var tgt_pelvis_m = pose_position(target_pose, self.skel_parents, pelvis_node) * self.scale
        var tgt_pelvis_horiz_w = q_yaw.rotate(Vec3(tgt_pelvis_m.x, Float32(0.0), tgt_pelvis_m.z))

        var new_root_y: Float32
        if self.has_floor and abs(self.floor_normal.y) > Float32(0.25):
            var nx = self.floor_normal.x
            var ny = self.floor_normal.y
            var nz = self.floor_normal.z
            var rx = pelvis_w.x - tgt_pelvis_horiz_w.x
            var rz = pelvis_w.z - tgt_pelvis_horiz_w.z
            new_root_y = (self.floor_offset - nx * rx - nz * rz) / ny
        else:
            new_root_y = pelvis_w.y - tgt_pelvis_m.y

        root_pos = Vec3(
            pelvis_w.x - tgt_pelvis_horiz_w.x,
            new_root_y,
            pelvis_w.z - tgt_pelvis_horiz_w.z,
        )
        root_yaw = new_yaw

        # Express exact ragdoll pose in the newly re-rooted `(root_pos, root_yaw)` frame
        var rag_pose = self.pose_in_frame(root_pos, root_yaw, target_pose)
        var count = target_pose.count()
        inertial.resize(count)

        var q_yaw_inv = q_yaw.conjugate()
        var inv_scale = Float32(1.0) / max(Float32(0.01), self.scale)

        for idx in range(count):
            var cont_t = rag_pose.t[idx]
            var cont_r = rag_pose.r[idx]
            var cont_s = rag_pose.s[idx]
            var tgt_t = target_pose.t[idx]
            var tgt_r = target_pose.r[idx]
            var tgt_s = target_pose.s[idx]

            inertial.out_t[idx] = cont_t
            inertial.out_r[idx] = cont_r
            inertial.out_s[idx] = cont_s
            inertial.prev_t[idx] = cont_t
            inertial.prev_r[idx] = cont_r
            inertial.prev_s[idx] = cont_s
            inertial.out_vel_t[idx] = Vec3()
            inertial.out_vel_r[idx] = Vec3()
            inertial.out_vel_s[idx] = Vec3()

            inertial.off_t[idx] = cont_t - tgt_t
            inertial.vel_t[idx] = Vec3()
            inertial.off_s[idx] = cont_s - tgt_s
            inertial.vel_s[idx] = Vec3()

            var theta_vec = det_quat_to_scaled_angle(cont_r.multiply(tgt_r.conjugate()))
            var theta_0 = theta_vec.length()
            if theta_0 > Float32(1.0e-6):
                var u = theta_vec * (Float32(1.0) / theta_0)
                inertial.rot_axis[idx] = u
                inertial.rot_ang[idx] = theta_0
            else:
                inertial.rot_axis[idx] = Vec3(0, 1, 0)
                inertial.rot_ang[idx] = Float32(0.0)
            inertial.rot_vel[idx] = Float32(0.0)
            inertial.off_r[idx] = theta_vec
            inertial.vel_r[idx] = Vec3()

        # Carry residual root linear velocity into pelvis translation spring if non-zero
        if pelvis_node >= 0 and pelvis_node < count:
            var root_v_m = q_yaw_inv.rotate(self.states[0].linear_vel) * inv_scale
            var p_node = self.skel_parents[pelvis_node] if pelvis_node < len(self.skel_parents) else -1
            if p_node >= 0:
                var gp_rot = _pose_global_rot(rag_pose, self.skel_parents, p_node)
                root_v_m = gp_rot.conjugate().rotate(root_v_m)
            inertial.out_vel_t[pelvis_node] = root_v_m
            inertial.vel_t[pelvis_node] = root_v_m

        inertial.has_pose = True
        inertial.is_settled = False
        inertial.has_prev_settled = False
        self.is_limp = False
        self.is_active = False

    def serialize(self, mut w: BlobWriter):
        """Writes deterministic binary snapshot of `ArticulatedRagdoll`."""
        w.begin(RAGDOLL_BLOB_TAG, RAGDOLL_BLOB_VERSION)
        w.integer(len(self.limbs))
        for i in range(len(self.limbs)):
            ref ld = self.limbs[i]
            w.integer(ld.bone_node)
            w.integer(ld.parent_limb)
            w.f32(ld.mass)
            w.f32(ld.radius)
            w.f32(ld.length)
            w.vec3(ld.local_axis)
            w.vec3(ld.local_com_offset)
            w.vec3(ld.parent_anchor_local)
            w.quat(ld.rest_local_rot)
            w.boolean(ld.is_hinge)
            w.vec3(ld.hinge_axis_local)
            w.f32(ld.hinge_min_rad)
            w.f32(ld.hinge_max_rad)
            w.f32(ld.swing_cone_rad)
            w.f32(ld.twist_min_rad)
            w.f32(ld.twist_max_rad)
            w.f32(ld.active_stiffness)
            w.f32(ld.active_damping)
            w.f32(ld.max_torque)

        w.integer(len(self.states))
        for i in range(len(self.states)):
            ref st = self.states[i]
            w.vec3(st.com_pos)
            w.vec3(st.prev_com_pos)
            w.quat(st.rotation)
            w.quat(st.prev_rotation)
            w.vec3(st.linear_vel)
            w.vec3(st.angular_vel)
            w.f32(st.inv_mass)
            w.f32(st.inv_inertia_axial)
            w.f32(st.inv_inertia_trans)
            w.quat(st.target_local_rot)
            w.boolean(st.on_surface)

        w.integer(len(self.obb_colliders))
        for i in range(len(self.obb_colliders)):
            ref ob = self.obb_colliders[i]
            w.vec3(ob.center)
            w.quat(ob.rotation)
            w.vec3(ob.half_extents)
            w.f32(ob.friction)
            w.f32(ob.restitution)

        w.integer(len(self.skel_parents))
        for i in range(len(self.skel_parents)):
            w.integer(self.skel_parents[i])

        w.integer(len(self.skel_topo_order))
        for i in range(len(self.skel_topo_order)):
            w.integer(self.skel_topo_order[i])

        var rp_cnt = self.rest_pose.count()
        w.integer(rp_cnt)
        for i in range(rp_cnt):
            w.vec3(self.rest_pose.t[i])
            w.quat(self.rest_pose.r[i])
            w.vec3(self.rest_pose.s[i])

        w.quat(self.pelvis_rest_global_rot)
        w.vec3(self.gravity)
        w.boolean(self.has_floor)
        w.vec3(self.floor_normal)
        w.f32(self.floor_offset)
        w.f32(self.floor_friction)
        w.f32(self.linear_damping)
        w.f32(self.angular_damping)
        w.f32(self.scale)
        w.integer(self.solver_iterations)
        w.f32(self.joint_compliance)
        w.f32(self.motor_scale)
        w.f32(self.root_attitude_stiffness)
        w.f32(self.root_attitude_damping)
        w.f32(self.root_attitude_max_torque)
        w.quat(self.target_root_world_rot)
        w.boolean(self.is_limp)
        w.boolean(self.is_active)
        w.f32(self.still_seconds)
        w.integer(self.steps_taken)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        """Restores `ArticulatedRagdoll` from a binary snapshot."""
        _ = r.enter(RAGDOLL_BLOB_TAG, RAGDOLL_BLOB_VERSION)
        var n_limbs = r.integer()
        if r.failed or n_limbs < 0 or n_limbs > 4096 or n_limbs > max(0, r.limit() - r.offset) // 125:
            raise Error("ragdoll blob: invalid limb count")
        self.limbs.clear()
        for _ in range(n_limbs):
            var ld = ArticulatedLimbDef()
            ld.bone_node = r.integer()
            ld.parent_limb = r.integer()
            ld.mass = r.f32()
            ld.radius = r.f32()
            ld.length = r.f32()
            ld.local_axis = r.vec3()
            ld.local_com_offset = r.vec3()
            ld.parent_anchor_local = r.vec3()
            ld.rest_local_rot = r.quat()
            ld.is_hinge = r.boolean()
            ld.hinge_axis_local = r.vec3()
            ld.hinge_min_rad = r.f32()
            ld.hinge_max_rad = r.f32()
            ld.swing_cone_rad = r.f32()
            ld.twist_min_rad = r.f32()
            ld.twist_max_rad = r.f32()
            ld.active_stiffness = r.f32()
            ld.active_damping = r.f32()
            ld.max_torque = r.f32()
            self.limbs.append(ld)

        var n_states = r.integer()
        if (
            r.failed
            or n_states < 0
            or n_states > 4096
            or n_states != n_limbs
            or n_states > max(0, r.limit() - r.offset) // 109
        ):
            raise Error("ragdoll blob: invalid state count")
        self.states.clear()
        for _ in range(n_states):
            var st = ArticulatedLimbState()
            st.com_pos = r.vec3()
            st.prev_com_pos = r.vec3()
            st.rotation = r.quat()
            st.prev_rotation = r.quat()
            st.linear_vel = r.vec3()
            st.angular_vel = r.vec3()
            st.inv_mass = r.f32()
            st.inv_inertia_axial = r.f32()
            st.inv_inertia_trans = r.f32()
            st.target_local_rot = r.quat()
            st.on_surface = r.boolean()
            self.states.append(st)

        var n_obbs = r.integer()
        if r.failed or n_obbs < 0 or n_obbs > 4096 or n_obbs > max(0, r.limit() - r.offset) // 48:
            raise Error("ragdoll blob: invalid obb count")
        self.obb_colliders.clear()
        for _ in range(n_obbs):
            var c = r.vec3()
            var q = r.quat()
            var he = r.vec3()
            var fr = r.f32()
            var re = r.f32()
            self.obb_colliders.append(RagdollObbCollider(c, q, he, fr, re))

        var n_par = r.integer()
        if r.failed or n_par < 0 or n_par > 65536 or n_par > max(0, r.limit() - r.offset) // 8:
            raise Error("ragdoll blob: invalid parent count")
        self.skel_parents.clear()
        for _ in range(n_par):
            self.skel_parents.append(r.integer())

        var n_topo = r.integer()
        if r.failed or n_topo < 0 or n_topo > 65536 or n_topo > max(0, r.limit() - r.offset) // 8:
            raise Error("ragdoll blob: invalid topo count")
        self.skel_topo_order.clear()
        for _ in range(n_topo):
            self.skel_topo_order.append(r.integer())

        var rp_cnt = r.integer()
        if r.failed or rp_cnt < 0 or rp_cnt > 65536 or rp_cnt > max(0, r.limit() - r.offset) // 40:
            raise Error("ragdoll blob: invalid rest pose count")
        self.rest_pose = Pose(rp_cnt)
        for i in range(rp_cnt):
            self.rest_pose.t[i] = r.vec3()
            self.rest_pose.r[i] = r.quat()
            self.rest_pose.s[i] = r.vec3()

        self.pelvis_rest_global_rot = r.quat()
        self.gravity = r.vec3()
        self.has_floor = r.boolean()
        self.floor_normal = r.vec3()
        self.floor_offset = r.f32()
        self.floor_friction = r.f32()
        self.linear_damping = r.f32()
        self.angular_damping = r.f32()
        self.scale = r.f32()
        self.solver_iterations = r.integer()
        self.joint_compliance = r.f32()
        self.motor_scale = r.f32()
        self.root_attitude_stiffness = r.f32()
        self.root_attitude_damping = r.f32()
        self.root_attitude_max_torque = r.f32()
        self.target_root_world_rot = r.quat()
        self.is_limp = r.boolean()
        self.is_active = r.boolean()
        self.still_seconds = r.f32()
        self.steps_taken = r.integer()
        r.leave()

        for i in range(n_par):
            var sp = self.skel_parents[i]
            if sp < -1 or sp >= n_par or sp == i:
                raise Error("ragdoll blob: invalid skel_parent index")
        for i in range(n_topo):
            var st_idx = self.skel_topo_order[i]
            if st_idx < 0 or st_idx >= n_par:
                raise Error("ragdoll blob: invalid skel_topo_order index")
        for i in range(n_limbs):
            var p = self.limbs[i].parent_limb
            if p < -1 or p >= n_limbs or p == i:
                raise Error("ragdoll blob: invalid parent_limb index")
            var cur = p
            var steps = 0
            while cur >= 0 and steps < n_limbs:
                if cur == i:
                    raise Error("ragdoll blob: cycle in parent_limb")
                cur = self.limbs[cur].parent_limb
                steps += 1
            if cur >= 0:
                raise Error("ragdoll blob: cycle in parent_limb")
            var bn = self.limbs[i].bone_node
            if n_par > 0:
                if bn < -1 or bn >= n_par:
                    raise Error("ragdoll blob: bone_node index out of range")
            elif bn < -1:
                raise Error("ragdoll blob: negative bone_node index")
