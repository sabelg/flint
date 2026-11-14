"""Cloth proxies (package 3.11): the Glider's membranes as XPBD cloth.

Each membrane is a small sheet of particles (MEMBRANE_COLS along the arm by
MEMBRANE_ROWS from the arm to the flank) on the 2.12 XPBD core
(physics/xpbd.mojo `ParticleMesh`): structural links along both directions,
shear links across each cell and bend links two apart, all a little
compliant. The arm edge is pinned to the shoulder, elbow and wrist of the
posed skeleton, the flank edge to the chest, spine, pelvis and thigh pushed
out to the side, every frame; the rows between are free. The air pushes
each free particle along the sheet's normal by the air's speed across the
sheet (the relative velocity of the wind mirror's air and the particle,
dotted with the normal): falling or gliding fills the sail from below, a
gust from the side ruffles it, and gravity and the links pull it back. The sheets are cut MEMBRANE_SLACK longer from
arm to flank than the glide's span, so they have room to belly.

The membranes on the Glider mesh are still two shape keys (`form_furl`,
`form_billow`, package 3.9), so the proxy drives them: the billow is the
sheet's mean bulge off the plane of its pinned edges, either way (both
sides averaged),
and when the arms go overhead the sheet's links are stretched past
MEMBRANE_STRETCH, the membrane furls by as much instead of stretching the
skin (the 3.9 note "membranes stretch with arms overhead"). Per-vertex
cloth skinning of the membranes would need the mesh's membrane vertices
bound to the proxy; that is left for a later asset pass.

Presentation only (fixed order, no randomness): the same frames give the
same pictures.
"""
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.ik import cross, dot, safe_normalized
from flint.assets.model import NativeModel
from flint.anim.procedural import Placement
from flint.physics.xpbd import ParticleMesh, ChainCollider
from flint.anim.pose import Pose, sample_pose


def pose_point(model: NativeModel, pose: Pose, node: Int) -> Vec3:
    """A node's model-space position in a pose (forward kinematics over the
    model's parents; call `ensure_parents` first)."""
    var position = Vec3()
    var current = node
    var guard = 0
    while current >= 0 and current < pose.count() and guard < 128:
        position = pose.t[current] + pose.r[current].rotate(position)
        current = model.parents[current] if current < len(model.parents) else -1
        guard += 1
    return position


def find_clip(model: NativeModel, name: String) -> Int:
    for index in range(len(model.animations)):
        if model.animations[index].name == name or model.animations[index].name.endswith("_" + name):
            return index
    return -1

comptime MEMBRANE_COLS: Int = 4
comptime MEMBRANE_ROWS: Int = 4
comptime MEMBRANE_STEP: Float32 = 1.0 / 120.0
comptime MEMBRANE_MAX_STEPS: Int = 12
comptime MEMBRANE_AIR: Float32 = 7.0        # 1/s: push along the normal per m/s of air across the sheet
comptime MEMBRANE_DEPTH: Float32 = 0.22     # metres of bulge (scale 1) that read as a full billow
comptime MEMBRANE_STRETCH: Float32 = 1.12   # the cross links' stretch the membrane takes before it furls
comptime MEMBRANE_FURL_RANGE: Float32 = 0.4
comptime MEMBRANE_SLACK: Float32 = 1.15     # cloth from arm to flank against the glide's span: room to belly


def membrane_arm(side: Int) -> List[String]:
    var s = String(".L") if side == 0 else String(".R")
    return [String("UpperArm") + s, String("Forearm") + s, String("Hand") + s]


def membrane_flank(side: Int) -> List[String]:
    var s = String(".L") if side == 0 else String(".R")
    return [String("Chest"), String("Spine2"), String("Pelvis"), String("Thigh") + s]


struct MembraneSheet(Copyable, Movable):
    var arm_nodes: List[Int]
    var flank_nodes: List[Int]
    var mesh: ParticleMesh
    var arm: List[Vec3]      # this frame's pinned arm edge (world), MEMBRANE_COLS
    var flank: List[Vec3]    # this frame's pinned flank edge (world)
    var cross_links: List[Int]
    var built: Bool
    var edge_center: Vec3     # mean of the pinned points this frame
    var edge_velocity: Vec3   # how the pinned edges moved this frame (world, m/s)

    def __init__(out self):
        self.arm_nodes = List[Int]()
        self.flank_nodes = List[Int]()
        self.mesh = ParticleMesh()
        self.arm = List[Vec3]()
        self.flank = List[Vec3]()
        self.cross_links = List[Int]()
        self.built = False
        self.edge_center = Vec3()
        self.edge_velocity = Vec3()

    def ok(self) -> Bool:
        if len(self.arm_nodes) != 3 or len(self.flank_nodes) != 4:
            return False
        for n in self.arm_nodes:
            if n < 0:
                return False
        for n in self.flank_nodes:
            if n < 0:
                return False
        return True

    def index(self, row: Int, col: Int) -> Int:
        return row * MEMBRANE_COLS + col

    def track(mut self, dt: Float32, fresh: Bool):
        """The pinned edges' own motion this frame (the body as drawn)."""
        var center = Vec3()
        for c in range(MEMBRANE_COLS):
            center = center + self.arm[c] + self.flank[c]
        center = center * (1.0 / Float32(2 * MEMBRANE_COLS))
        if fresh or dt <= 0:
            self.edge_velocity = Vec3()
        else:
            self.edge_velocity = (center - self.edge_center) * (1.0 / dt)
        self.edge_center = center

    def edges(mut self, mut model: NativeModel, placement: Placement):
        """The pinned edges from the pose: the arm's shoulder, elbow, mid
        forearm and wrist; the flank from the chest down to the thigh, pushed
        out toward the arm by 60% of the shoulder's reach."""
        var shoulder = placement.to_world(model.global_position(self.arm_nodes[0]))
        var elbow = placement.to_world(model.global_position(self.arm_nodes[1]))
        var wrist = placement.to_world(model.global_position(self.arm_nodes[2]))
        self.arm = [shoulder, elbow, elbow + (wrist - elbow) * 0.5, wrist]
        var chest = placement.to_world(model.global_position(self.flank_nodes[0]))
        var out = (shoulder - chest) * 0.6
        out.y = 0
        self.flank = List[Vec3]()
        for k in range(4):
            var at = placement.to_world(model.global_position(self.flank_nodes[k]))
            self.flank.append(at + out * (Float32(1.0) if k < 3 else Float32(0.3)))

    def edges_of_pose(mut self, model: NativeModel, pose: Pose, placement: Placement):
        """The edges of a sampled pose (the Glide clip's full spread)."""
        var shoulder = placement.to_world(pose_point(model, pose, self.arm_nodes[0]))
        var elbow = placement.to_world(pose_point(model, pose, self.arm_nodes[1]))
        var wrist = placement.to_world(pose_point(model, pose, self.arm_nodes[2]))
        self.arm = [shoulder, elbow, elbow + (wrist - elbow) * 0.5, wrist]
        var chest = placement.to_world(pose_point(model, pose, self.flank_nodes[0]))
        var out = (shoulder - chest) * 0.6
        out.y = 0
        self.flank = List[Vec3]()
        for k in range(4):
            var at = placement.to_world(pose_point(model, pose, self.flank_nodes[k]))
            self.flank.append(at + out * (Float32(1.0) if k < 3 else Float32(0.3)))

    def build(mut self):
        self.mesh = ParticleMesh()
        self.mesh.damping = 1.2
        self.mesh.wind_drag = 0
        self.mesh.friction = 0
        for r in range(MEMBRANE_ROWS):
            var share = Float32(r) / Float32(MEMBRANE_ROWS - 1)
            for c in range(MEMBRANE_COLS):
                var at = self.arm[c] + (self.flank[c] - self.arm[c]) * share
                var pinned = r == 0 or r == MEMBRANE_ROWS - 1
                _ = self.mesh.add_particle(at, 0 if pinned else Float32(0.08), 0.02)
        self.cross_links = List[Int]()
        for r in range(MEMBRANE_ROWS):
            for c in range(MEMBRANE_COLS):
                if c + 1 < MEMBRANE_COLS:
                    self.mesh.add_link(self.index(r, c), self.index(r, c + 1), 2.0e-4)
                if r + 1 < MEMBRANE_ROWS:
                    self.cross_links.append(self.mesh.links())
                    self.mesh.add_link(self.index(r, c), self.index(r + 1, c), 2.0e-4, rest_share=MEMBRANE_SLACK)
                if r + 1 < MEMBRANE_ROWS and c + 1 < MEMBRANE_COLS:
                    self.mesh.add_link(self.index(r, c), self.index(r + 1, c + 1), 1.0e-3, rest_share=MEMBRANE_SLACK)
                    self.mesh.add_link(self.index(r, c + 1), self.index(r + 1, c), 1.0e-3, rest_share=MEMBRANE_SLACK)
                if r + 2 < MEMBRANE_ROWS:
                    self.mesh.add_link(self.index(r, c), self.index(r + 2, c), 5.0e-3, rest_share=MEMBRANE_SLACK)
                if c + 2 < MEMBRANE_COLS:
                    self.mesh.add_link(self.index(r, c), self.index(r, c + 2), 5.0e-3)
        self.built = True

    def pin_edges(mut self):
        for c in range(MEMBRANE_COLS):
            self.mesh.pin(self.index(0, c), self.arm[c])
            self.mesh.pin(self.index(MEMBRANE_ROWS - 1, c), self.flank[c])

    def normal(self) -> Vec3:
        """The sheet's plane normal from its pinned edges, toward the sky."""
        var along = self.arm[MEMBRANE_COLS - 1] - self.arm[0]
        var across = self.flank[0] - self.arm[0]
        var n = safe_normalized(cross(along, across), Vec3(0, 1, 0))
        if n.y < 0:
            n = n * -1.0
        return n

    def air(mut self, wind: Vec3, body_velocity: Vec3, strength: Float32):
        """The air on each free particle, along the normal, by the air across
        it: the wind against the body's own speed through the air (the
        simulation's velocity), less the particle's flutter relative to the
        edges it hangs from."""
        var n = self.normal()
        for i in range(self.mesh.count()):
            if self.mesh.inv_mass[i] == 0:
                self.mesh.accel[i] = Vec3()
                continue
            var flutter = self.mesh.velocity[i] - self.edge_velocity
            var across = dot(wind - body_velocity - flutter, n)
            self.mesh.accel[i] = n * (across * MEMBRANE_AIR * strength)

    def bulge(self) -> Float32:
        """Mean height of the free rows off the plane of the pinned edges (metres)."""
        var n = self.normal()
        var total = Float32(0)
        var count = 0
        for r in range(1, MEMBRANE_ROWS - 1):
            var share = Float32(r) / Float32(MEMBRANE_ROWS - 1)
            for c in range(MEMBRANE_COLS):
                var flat = self.arm[c] + (self.flank[c] - self.arm[c]) * share
                total += dot(self.mesh.position[self.index(r, c)] - flat, n)
                count += 1
        return total / Float32(count) if count > 0 else Float32(0)

    def stretch(self) -> Float32:
        """Mean stretch of the links from the arm to the flank (1: at rest)."""
        var total = Float32(0)
        for k in self.cross_links:
            var d = (self.mesh.position[self.mesh.link_b[k]] - self.mesh.position[self.mesh.link_a[k]]).length()
            total += d / max(Float32(1.0e-4), self.mesh.rest[k])
        return total / Float32(len(self.cross_links)) if len(self.cross_links) > 0 else Float32(1)

    def span_stretch(self) -> Float32:
        """How far apart the pinned edges are against the sheet's rest, the
        worst column (1: as at full spread)."""
        var worst = Float32(0)
        for c in range(MEMBRANE_COLS):
            var rest = Float32(0)
            for k in self.cross_links:
                for r in range(MEMBRANE_ROWS - 1):
                    if self.mesh.link_a[k] == self.index(r, c):
                        rest += self.mesh.rest[k]
            worst = max(worst, (self.flank[c] - self.arm[c]).length() * MEMBRANE_SLACK / max(Float32(1.0e-4), rest))
        return worst

    def lay_on_edges(mut self):
        """Every particle back on the flat sheet between this frame's edges, at rest."""
        for r in range(MEMBRANE_ROWS):
            var share = Float32(r) / Float32(MEMBRANE_ROWS - 1)
            for c in range(MEMBRANE_COLS):
                self.mesh.place(self.index(r, c), self.arm[c] + (self.flank[c] - self.arm[c]) * share)


struct MembraneCloth(Copyable, Movable):
    var handle: Int
    var sheets: List[MembraneSheet]
    var accumulator: Float32
    var scale: Float32
    var billow: Float32       # 0..1.3 for the shape key
    var furl: Float32         # 0..0.6 extra furl from overstretch
    var steps: Int
    var wind_seen: Vec3

    def __init__(out self):
        self.handle = -1
        self.sheets = List[MembraneSheet]()
        self.accumulator = 0
        self.scale = 1
        self.billow = 0
        self.furl = 0
        self.steps = 0
        self.wind_seen = Vec3()

    def reset(mut self):
        for i in range(len(self.sheets)):
            self.sheets[i].built = False
        self.accumulator = 0

    def bind(mut self, model: NativeModel, handle: Int):
        if handle == self.handle:
            return
        self.handle = handle
        self.sheets = List[MembraneSheet]()
        for side in range(2):
            var sheet = MembraneSheet()
            for name in membrane_arm(side):
                sheet.arm_nodes.append(model.find_node(name))
            for name in membrane_flank(side):
                sheet.flank_nodes.append(model.find_node(name))
            self.sheets.append(sheet^)

    def ready(self) -> Bool:
        return len(self.sheets) == 2 and self.sheets[0].ok() and self.sheets[1].ok()

    def update(mut self, mut model: NativeModel, handle: Int, placement: Placement, wind: Vec3, body_velocity: Vec3, spread: Float32, dt: Float32):
        """One frame: the edges from the pose, the sheets stepped at 120 Hz,
        then the billow and furl read off them. `spread` (0..1) is how open
        the membranes are (3.9's spread spring): a furled sail takes no air."""
        self.bind(model, handle)
        if not self.ready():
            return
        self.scale = max(Float32(0.05), (abs(placement.scale.x) + abs(placement.scale.y) + abs(placement.scale.z)) / 3.0)
        self.wind_seen = wind
        if not self.sheets[0].built or not self.sheets[1].built:
            # The sheet's rest is the Glide clip's first frame: full spread.
            model.ensure_parents()
            var pose = Pose()
            var glide = find_clip(model, "Glide")
            if glide >= 0:
                sample_pose(model, glide, 0, pose)
            else:
                pose.read(model)
            for s in range(2):
                self.sheets[s].edges_of_pose(model, pose, placement)
                self.sheets[s].build()
        for s in range(2):
            self.sheets[s].edges(model, placement)
            var fresh = self.sheets[s].mesh.steps_run == 0
            self.sheets[s].track(dt, fresh)
            if fresh:
                self.sheets[s].lay_on_edges()
                self.sheets[s].mesh.steps_run = 1
        self.accumulator += max(Float32(0), dt)
        var steps = 0
        var none = List[Vec3]()
        var colliders = List[ChainCollider]()
        while self.accumulator >= MEMBRANE_STEP and steps < MEMBRANE_MAX_STEPS:
            for s in range(2):
                self.sheets[s].pin_edges()
                self.sheets[s].air(wind, body_velocity, spread)
                self.sheets[s].mesh.step(MEMBRANE_STEP, 2, 9.81 * 0.35, none, colliders)
            self.accumulator -= MEMBRANE_STEP
            steps += 1
            self.steps += 1
        if steps == MEMBRANE_MAX_STEPS:
            self.accumulator = 0
        # The sail's fullness, whichever way the air bellies it (the shape key
        # has one bellied shape).
        var bulge = (abs(self.sheets[0].bulge()) + abs(self.sheets[1].bulge())) * 0.5
        self.billow = max(Float32(0), min(Float32(1.3), bulge / (MEMBRANE_DEPTH * self.scale))) * spread
        var stretch = max(self.sheets[0].span_stretch(), self.sheets[1].span_stretch())
        self.furl = max(Float32(0), min(Float32(0.6), (stretch - MEMBRANE_STRETCH) / MEMBRANE_FURL_RANGE))
