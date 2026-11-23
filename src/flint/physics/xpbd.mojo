"""The XPBD particle-chain & cloth/ragdoll mesh core (v4): multi-iteration Gauss-Seidel
compliance projection, capsule push-out, and two-way RigidAttachment impulse coupling.

Upgrades in v4:
1. Persists Lagrange multipliers `lambdas: List[Float32]` (zeroed at the start of each
   substep) across `iterations: Int = 2` Gauss-Seidel passes using true XPBD projection:
   `dl = (-C - alpha_tilde * lam) / (w_sum + alpha_tilde)`, `lam += dl`.
2. Supports both sphere and 3D capsule push-out (`ChainCollider.capsule(a, b, radius)`).
3. Adds `RigidAttachment` two-way coupling between XPBD chains/meshes and dynamic rigid
   bodies in `SolverBodies`, feeding exact generalized impulses `J = -(dl / h) * n`
   and torques `r_b x J` back into rigid body linear and angular velocities.
"""
from std.collections import List
from std.math import sqrt, max, min, abs, isfinite
from flint.core.math import Vec3, Quat
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.physics.solver_bodies import SolverBodies, rotate_vec


comptime LINK_EQUAL: Int = 0
comptime LINK_MIN: Int = 1
comptime XPBD_CHAIN_TAG: UInt32 = fourcc("XPBC")
comptime XPBD_MESH_TAG: UInt32 = fourcc("XPBM")
comptime XPBD_ATT_TAG: UInt32 = fourcc("XPBA")
comptime XPBD_BLOB_VERSION: Int = 1
comptime XPBD_ATT_BLOB_VERSION: Int = 2

comptime COLLIDER_SPHERE: Int = 0
comptime COLLIDER_CAPSULE: Int = 1
comptime COLLIDER_BOX: Int = 2


struct ChainCollider(Copyable, ImplicitlyCopyable, Movable):
    var center: Vec3
    var end: Vec3
    var radius: Float32
    var is_capsule: Bool
    var kind: Int
    var rot: Quat
    var half_extents: Vec3
    var solver_body: Int
    var body_id: Int

    def __init__(out self, center: Vec3, radius: Float32, solver_body: Int = -1, body_id: Int = -1):
        self.center = center
        self.end = center
        self.radius = radius
        self.is_capsule = False
        self.kind = COLLIDER_SPHERE
        self.rot = Quat()
        self.half_extents = Vec3()
        self.solver_body = solver_body
        self.body_id = body_id

    def __init__(out self, a: Vec3, b: Vec3, radius: Float32, solver_body: Int = -1, body_id: Int = -1):
        self.center = a
        self.end = b
        self.radius = radius
        self.is_capsule = True
        self.kind = COLLIDER_CAPSULE
        self.rot = Quat()
        self.half_extents = Vec3()
        self.solver_body = solver_body
        self.body_id = body_id

    @staticmethod
    def sphere(center: Vec3, radius: Float32, solver_body: Int = -1, body_id: Int = -1) -> Self:
        return Self(center, radius, solver_body, body_id)

    @staticmethod
    def capsule(a: Vec3, b: Vec3, radius: Float32, solver_body: Int = -1, body_id: Int = -1) -> Self:
        return Self(a, b, radius, solver_body, body_id)

    @staticmethod
    def box(center: Vec3, rot: Quat, half_extents: Vec3, solver_body: Int = -1, body_id: Int = -1) -> Self:
        var out = Self(center, Float32(0.0), solver_body, body_id)
        out.kind = COLLIDER_BOX
        out.is_capsule = False
        out.rot = rot.normalized()
        out.half_extents = half_extents
        return out

    @always_inline
    def closest_axis_point(self, p: Vec3) -> Vec3:
        if not self.is_capsule:
            return self.center
        var ab = self.end - self.center
        var ab2 = ab.length_squared()
        if ab2 <= Float32(1.0e-10):
            return self.center
        var t = max(Float32(0.0), min(Float32(1.0), (p - self.center).dot(ab) / ab2))
        return self.center + ab * t


struct RigidAttachment(Copyable, ImplicitlyCopyable, Movable):
    """Two-way XPBD attachment linking particle `particle` to `SolverBodies` row `solver_body`
    (or persistent `PhysicsWorld` body `body_id`) at body-local offset `local_anchor` with compliance `compliance` (m/N)."""
    var particle: Int
    var solver_body: Int
    var body_id: Int
    var local_anchor: Vec3
    var rest_length: Float32
    var compliance: Float32
    var lam: Float32

    def __init__(
        out self,
        particle: Int,
        solver_body: Int,
        local_anchor: Vec3,
        compliance: Float32 = 0.0,
        rest_length: Float32 = 0.0,
        body_id: Int = -1,
    ):
        self.particle = particle
        self.solver_body = solver_body
        self.body_id = body_id
        self.local_anchor = local_anchor
        self.rest_length = rest_length
        self.compliance = compliance
        self.lam = 0.0

    @staticmethod
    def to_body_id(
        particle: Int,
        body_id: Int,
        local_anchor: Vec3,
        compliance: Float32 = 0.0,
        rest_length: Float32 = 0.0,
    ) -> Self:
        return Self(particle, -1, local_anchor, compliance, rest_length, body_id)

    def serialize(self, mut w: BlobWriter):
        w.begin(XPBD_ATT_TAG, XPBD_ATT_BLOB_VERSION)
        w.integer(self.particle)
        w.integer(self.solver_body)
        w.vec3(self.local_anchor)
        w.f32(self.rest_length)
        w.f32(self.compliance)
        w.f32(self.lam)
        w.integer(self.body_id)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        var ver = r.enter(XPBD_ATT_TAG, XPBD_ATT_BLOB_VERSION)
        self.particle = r.integer()
        self.solver_body = r.integer()
        self.local_anchor = r.vec3()
        self.rest_length = r.f32()
        self.compliance = r.f32()
        self.lam = r.f32()
        self.body_id = r.integer() if ver >= 2 else -1
        r.leave()


@always_inline
def _particle_attached_to_collider(
    particle: Int,
    c: ChainCollider,
    attachments: List[RigidAttachment],
) -> Bool:
    for a_i in range(len(attachments)):
        ref att = attachments[a_i]
        if att.particle != particle:
            continue
        if c.body_id >= 0 and att.body_id == c.body_id:
            return True
        if c.solver_body >= 0 and att.solver_body == c.solver_body:
            return True
    return False


@always_inline
def _collide_particle_with_collider(
    mut p: Vec3,
    p_radius: Float32,
    wp: Float32,
    c: ChainCollider,
    mut bodies: SolverBodies,
    h: Float32,
    sub_t: Float32,
) -> Bool:
    """Resolves particle vs Sphere/Capsule/OBB `ChainCollider` with two-way impulse exchange
    onto `bodies[c.solver_body]` when `solver_body` refers to a dynamic `SolverBodies` row.
    Returns `True` if contact occurred."""
    var b = c.solver_body
    var has_solver = b >= 0 and b < bodies.count
    var shift = bodies.velocity(b) * sub_t if has_solver else Vec3()
    var c_center = c.center + shift
    var c_end = c.end + shift

    var n_w: Vec3
    var pen: Float32
    var contact_pt: Vec3

    if c.kind == COLLIDER_BOX:
        var p_loc = c.rot.conjugate().rotate(p - c_center)
        var he = c.half_extents
        var clamped = Vec3(
            max(-he.x, min(he.x, p_loc.x)),
            max(-he.y, min(he.y, p_loc.y)),
            max(-he.z, min(he.z, p_loc.z)),
        )
        var diff = p_loc - clamped
        var d = diff.length()
        if d > Float32(1.0e-6):
            if d >= p_radius:
                return False
            var n_loc = diff * (Float32(1.0) / d)
            pen = p_radius - d
            n_w = c.rot.rotate(n_loc)
            contact_pt = c_center + c.rot.rotate(clamped)
        else:
            var dx = he.x - abs(p_loc.x)
            var dy = he.y - abs(p_loc.y)
            var dz = he.z - abs(p_loc.z)
            var n_loc: Vec3
            var face_d: Float32
            if dx <= dy and dx <= dz:
                n_loc = Vec3(Float32(1.0) if p_loc.x >= Float32(0.0) else Float32(-1.0), 0.0, 0.0)
                face_d = dx
            elif dy <= dz:
                n_loc = Vec3(0.0, Float32(1.0) if p_loc.y >= Float32(0.0) else Float32(-1.0), 0.0)
                face_d = dy
            else:
                n_loc = Vec3(0.0, 0.0, Float32(1.0) if p_loc.z >= Float32(0.0) else Float32(-1.0))
                face_d = dz
            pen = face_d + p_radius
            n_w = c.rot.rotate(n_loc)
            contact_pt = c_center + c.rot.rotate(p_loc + n_loc * face_d)
    else:
        var cp = c_center
        if c.is_capsule:
            var ab = c_end - c_center
            var ab2 = ab.length_squared()
            if ab2 > Float32(1.0e-10):
                var t = max(Float32(0.0), min(Float32(1.0), (p - c_center).dot(ab) / ab2))
                cp = c_center + ab * t
        var d_vec = p - cp
        var dist = d_vec.length()
        var reach = c.radius + p_radius
        if dist >= reach:
            return False
        if dist > Float32(1.0e-6):
            n_w = d_vec * (Float32(1.0) / dist)
        else:
            n_w = Vec3(0.0, 1.0, 0.0)
        pen = reach - dist
        contact_pt = cp + n_w * c.radius

    if pen <= Float32(0.0):
        return False

    var mb = bodies.inv_mass[b] if has_solver else Float32(0.0)
    if not has_solver or mb <= Float32(0.0):
        p = p + n_w * pen
        return True

    var rb = contact_pt - (bodies.center(b) + shift)
    var ib = bodies.inertia(b)
    var rn = rb.cross(n_w)
    var w_body = mb + rn.dot(ib.apply(rn))
    var denom = wp + w_body
    if denom <= Float32(1.0e-12):
        return False

    var dl = pen / denom
    if wp > Float32(0.0):
        p = p + n_w * (wp * dl)

    var j_body = -n_w * (dl / h)
    var vb = bodies.velocity(b) + j_body * mb
    var wb = bodies.angular(b) + ib.apply(rb.cross(j_body))
    bodies.set_velocity(b, vb, wb)
    return True


@always_inline
def _project_rigid_attachment(
    mut pos: List[Vec3],
    inv_mass: List[Float32],
    mut att: RigidAttachment,
    mut bodies: SolverBodies,
    h: Float32,
    substep_time: Float32,
):
    var p = att.particle
    if p < 0 or p >= len(pos):
        return
    var wp = inv_mass[p]
    var b = att.solver_body
    var body_valid = b >= 0 and b < bodies.count
    var mb = bodies.inv_mass[b] if body_valid else Float32(0.0)
    var rb = Vec3()
    var anchor_w = att.local_anchor
    if body_valid:
        rb = rotate_vec(bodies.orientation(b), att.local_anchor)
        var vb0 = bodies.velocity(b)
        var wb0 = bodies.angular(b)
        anchor_w = bodies.center(b) + rb + (vb0 + wb0.cross(rb)) * substep_time

    var d = pos[p] - anchor_w
    var dist = d.length()
    var c = dist - att.rest_length
    if abs(c) <= Float32(1.0e-7):
        return

    var n_dir = d * (Float32(1.0) / dist) if dist > Float32(1.0e-6) else Vec3(0, 1, 0)
    var w_body = Float32(0.0)
    if body_valid and mb > Float32(0.0):
        var ib = bodies.inertia(b)
        var rn = rb.cross(n_dir)
        w_body = mb + rn.dot(ib.apply(rn))

    var alpha_tilde = att.compliance / (h * h)
    var denom = wp + w_body + alpha_tilde
    if denom <= Float32(1.0e-12):
        return

    var dl = (-c - alpha_tilde * att.lam) / denom
    att.lam += dl

    if wp > Float32(0.0):
        pos[p] = pos[p] + n_dir * (wp * dl)

    if body_valid and w_body > Float32(0.0):
        # XPBD impulse J_p = +n * (dl / h) on particle => Newton's 3rd Law J_b = -n * (dl / h) on body
        var j_body = -n_dir * (dl / h)
        var ib = bodies.inertia(b)
        var vb = bodies.velocity(b) + j_body * mb
        var wb = bodies.angular(b) + ib.apply(rb.cross(j_body))
        bodies.set_velocity(b, vb, wb)


struct ParticleChain(Copyable, Movable):
    var position: List[Vec3]
    var previous: List[Vec3]
    var velocity: List[Vec3]
    var inv_mass: List[Float32]
    var rest: List[Float32]      # rest[i]: between particle i and i + 1
    var lambdas: List[Float32]   # XPBD Lagrange multipliers persisted across iterations within substep
    var compliance: Float32
    var damping: Float32         # per second
    var wind_drag: Float32       # per second, toward the air's velocity
    var floor: Float32           # no particle below this height (-1e9: none)

    def __init__(out self):
        self.position = List[Vec3]()
        self.previous = List[Vec3]()
        self.velocity = List[Vec3]()
        self.inv_mass = List[Float32]()
        self.rest = List[Float32]()
        self.lambdas = List[Float32]()
        self.compliance = 0.0
        self.damping = 0.4
        self.wind_drag = 1.5
        self.floor = -1.0e9

    @staticmethod
    def line(start: Vec3, end: Vec3, particles: Int, mass: Float32 = 0.05) -> Self:
        """A straight chain of `particles` from start to end, both ends free."""
        var chain = Self()
        var count = max(2, particles)
        var step = (end - start) * (1.0 / Float32(count - 1))
        for i in range(count):
            var p = start + step * Float32(i)
            chain.position.append(p)
            chain.previous.append(p)
            chain.velocity.append(Vec3())
            chain.inv_mass.append(1.0 / max(Float32(1.0e-4), mass))
        for _ in range(count - 1):
            chain.rest.append(step.length())
            chain.lambdas.append(Float32(0.0))
        return chain^

    def count(self) -> Int:
        return len(self.position)

    def pin(mut self, index: Int, at: Vec3):
        """Holds a particle at a point (inverse mass 0); call each frame."""
        if index < 0 or index >= len(self.position):
            return
        self.inv_mass[index] = 0
        self.position[index] = at
        self.velocity[index] = Vec3()

    def length(self) -> Float32:
        var total = Float32(0)
        for i in range(len(self.position) - 1):
            total += (self.position[i + 1] - self.position[i]).length()
        return total

    def rest_length(self) -> Float32:
        var total = Float32(0)
        for r in self.rest:
            total += r
        return total

    def _ensure_lambdas(mut self):
        var n_links = len(self.rest)
        if len(self.lambdas) < n_links:
            self.lambdas.resize(n_links, Float32(0.0))
        for k in range(n_links):
            self.lambdas[k] = Float32(0.0)

    def step(
        mut self,
        dt: Float32,
        substeps: Int,
        gravity: Float32,
        wind: List[Vec3],
        colliders: List[ChainCollider],
        iterations: Int = 2,
    ):
        """Advances the chain by `dt` with `iterations` Gauss-Seidel XPBD passes per substep."""
        var dummy_atts = List[RigidAttachment]()
        var dummy_bodies = SolverBodies()
        self.step_coupled(dt, substeps, gravity, wind, colliders, dummy_atts, dummy_bodies, iterations)

    def step_substep_coupled(
        mut self,
        h: Float32,
        sub_t: Float32,
        gravity: Float32,
        wind: List[Vec3],
        colliders: List[ChainCollider],
        mut attachments: List[RigidAttachment],
        mut bodies: SolverBodies,
        iterations: Int = 2,
    ):
        """Advances one XPBD substep of size `h` with two-way `RigidAttachment` impulse feedback."""
        if not isfinite(h) or h <= Float32(1.0e-8):
            return
        var safe_h = min(h, Float32(0.1))
        var safe_sub_t = sub_t if isfinite(sub_t) else Float32(0.0)
        var safe_gravity = gravity if isfinite(gravity) else Float32(9.81)
        var iters = max(1, min(64, iterations))
        var alpha_tilde = self.compliance / (safe_h * safe_h)

        self._ensure_lambdas()
        for a_i in range(len(attachments)):
            attachments[a_i].lam = Float32(0.0)

        for i in range(len(self.position)):
            if not (isfinite(self.position[i].x) and isfinite(self.position[i].y) and isfinite(self.position[i].z)):
                self.position[i] = Vec3()
            if not (isfinite(self.velocity[i].x) and isfinite(self.velocity[i].y) and isfinite(self.velocity[i].z)):
                self.velocity[i] = Vec3()
            self.previous[i] = self.position[i]
            if self.inv_mass[i] == 0:
                continue
            var v = self.velocity[i]
            v.y -= safe_gravity * safe_h
            if i < len(wind):
                var wv = wind[i]
                if isfinite(wv.x) and isfinite(wv.y) and isfinite(wv.z):
                    v = v + (wv - v) * min(Float32(1), self.wind_drag * safe_h)
            v = v * max(Float32(0), 1 - self.damping * safe_h)
            self.velocity[i] = v
            self.position[i] = self.position[i] + v * safe_h

        for _ in range(iters):
            for a_i in range(len(attachments)):
                _project_rigid_attachment(self.position, self.inv_mass, attachments[a_i], bodies, safe_h, safe_sub_t)

            for i in range(len(self.rest)):
                var wa = self.inv_mass[i]
                var wb = self.inv_mass[i + 1]
                var w_sum = wa + wb
                var denom = w_sum + alpha_tilde
                if denom <= Float32(0.0) or w_sum <= Float32(0.0):
                    continue
                var d = self.position[i + 1] - self.position[i]
                var length = d.length()
                if length < Float32(1.0e-6):
                    continue
                var c = length - self.rest[i]
                var lam = self.lambdas[i]
                var dl = (-c - alpha_tilde * lam) / denom
                self.lambdas[i] = lam + dl
                var n_dir = d * (Float32(1.0) / length)
                self.position[i] = self.position[i] - n_dir * (dl * wa)
                self.position[i + 1] = self.position[i + 1] + n_dir * (dl * wb)

        for i in range(len(self.position)):
            if self.inv_mass[i] == 0:
                continue
            for c in colliders:
                if _particle_attached_to_collider(i, c, attachments):
                    continue
                var p_i = self.position[i]
                if _collide_particle_with_collider(p_i, Float32(0.0), self.inv_mass[i], c, bodies, safe_h, safe_sub_t):
                    self.position[i] = p_i
            if self.position[i].y < self.floor:
                self.position[i].y = self.floor
            self.velocity[i] = (self.position[i] - self.previous[i]) * (Float32(1.0) / safe_h)

    def step_coupled(
        mut self,
        dt: Float32,
        substeps: Int,
        gravity: Float32,
        wind: List[Vec3],
        colliders: List[ChainCollider],
        mut attachments: List[RigidAttachment],
        mut bodies: SolverBodies,
        iterations: Int = 2,
    ):
        """Advances the chain with two-way `RigidAttachment` impulse feedback into `SolverBodies`."""
        if not isfinite(dt) or dt <= Float32(1.0e-8):
            return
        var n = max(1, min(64, substeps))
        var h = min(dt, Float32(1.0)) / Float32(n)
        for sub in range(n):
            self.step_substep_coupled(
                h, Float32(sub) * h, gravity, wind, colliders, attachments, bodies, iterations
            )

    def serialize(self, mut w: BlobWriter):
        """Bit-exact snapshot of XPBD `ParticleChain` including `lambdas`."""
        w.begin(XPBD_CHAIN_TAG, XPBD_BLOB_VERSION)
        var n_p = len(self.position)
        w.integer(n_p)
        for i in range(n_p):
            w.vec3(self.position[i])
            w.vec3(self.previous[i])
            w.vec3(self.velocity[i])
            w.f32(self.inv_mass[i])
        var n_r = len(self.rest)
        w.integer(n_r)
        for i in range(n_r):
            w.f32(self.rest[i])
        var n_lam = len(self.lambdas)
        w.integer(n_lam)
        for i in range(n_lam):
            w.f32(self.lambdas[i])
        w.f32(self.compliance)
        w.f32(self.damping)
        w.f32(self.wind_drag)
        w.f32(self.floor)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        """Restores bit-exact `ParticleChain` state from `BlobReader`."""
        _ = r.enter(XPBD_CHAIN_TAG, XPBD_BLOB_VERSION)
        var n_p = r.integer()
        if r.failed or n_p < 0 or n_p > 1 << 20 or n_p > max(0, r.limit() - r.offset) // 40:
            raise Error("xpbd chain blob: invalid particle count")
        self.position.clear()
        self.previous.clear()
        self.velocity.clear()
        self.inv_mass.clear()
        for _ in range(n_p):
            self.position.append(r.vec3())
            self.previous.append(r.vec3())
            self.velocity.append(r.vec3())
            self.inv_mass.append(r.f32())
        var n_r = r.integer()
        if r.failed or n_r < 0 or n_r > 1 << 20 or n_r > max(0, r.limit() - r.offset) // 4:
            raise Error("xpbd chain blob: invalid link count")
        if n_r > 0 and n_p < n_r + 1:
            raise Error("xpbd chain blob: link count exceeds particle count")
        self.rest.clear()
        for _ in range(n_r):
            self.rest.append(r.f32())
        var n_lam = r.integer()
        if r.failed or n_lam < 0 or n_lam > 1 << 20 or n_lam > max(0, r.limit() - r.offset) // 4:
            raise Error("xpbd chain blob: invalid lambda count")
        self.lambdas.clear()
        for _ in range(n_lam):
            self.lambdas.append(r.f32())
        self.compliance = r.f32()
        self.damping = r.f32()
        self.wind_drag = r.f32()
        self.floor = r.f32()
        r.leave()


struct ParticleMesh(Copyable, Movable):
    var position: List[Vec3]
    var previous: List[Vec3]
    var velocity: List[Vec3]
    var inv_mass: List[Float32]
    var radius: List[Float32]
    var floor: List[Float32]     # per particle (-1e9: none)
    var target: List[Vec3]
    var power: List[Float32]     # per second toward the target (0: free)
    var accel: List[Vec3]        # extra acceleration through the next step (air on a cloth)
    var link_a: List[Int]
    var link_b: List[Int]
    var rest: List[Float32]
    var link_compliance: List[Float32]
    var link_kind: List[Int]
    var lambdas: List[Float32]   # XPBD Lagrange multipliers persisted across iterations within substep
    var damping: Float32         # per second
    var wind_drag: Float32       # per second, toward the air's velocity
    var friction: Float32        # share of sliding speed lost per substep
    var gravity_scale: Float32
    var touching: Int            # particles on their floor after the last step
    var steps_run: Int           # steps taken (0: just built)

    def __init__(out self):
        self.position = List[Vec3]()
        self.previous = List[Vec3]()
        self.velocity = List[Vec3]()
        self.inv_mass = List[Float32]()
        self.radius = List[Float32]()
        self.floor = List[Float32]()
        self.target = List[Vec3]()
        self.power = List[Float32]()
        self.accel = List[Vec3]()
        self.link_a = List[Int]()
        self.link_b = List[Int]()
        self.rest = List[Float32]()
        self.link_compliance = List[Float32]()
        self.link_kind = List[Int]()
        self.lambdas = List[Float32]()
        self.damping = 0.4
        self.wind_drag = 0.0
        self.friction = 0.35
        self.gravity_scale = 1.0
        self.touching = 0
        self.steps_run = 0

    def count(self) -> Int:
        return len(self.position)

    def links(self) -> Int:
        return len(self.link_a)

    def add_particle(mut self, at: Vec3, mass: Float32, radius: Float32 = 0.05) -> Int:
        self.position.append(at)
        self.previous.append(at)
        self.velocity.append(Vec3())
        self.inv_mass.append(1.0 / max(Float32(1.0e-4), mass) if mass > 0 else Float32(0))
        self.radius.append(radius)
        self.floor.append(-1.0e9)
        self.target.append(at)
        self.power.append(0)
        self.accel.append(Vec3())
        return len(self.position) - 1

    def add_link(mut self, a: Int, b: Int, compliance: Float32 = 0, kind: Int = LINK_EQUAL, rest_share: Float32 = 1):
        """A link at the particles' present distance (times `rest_share`)."""
        if a < 0 or b < 0 or a >= len(self.position) or b >= len(self.position) or a == b:
            return
        self.link_a.append(a)
        self.link_b.append(b)
        self.rest.append((self.position[b] - self.position[a]).length() * rest_share)
        self.link_compliance.append(compliance)
        self.link_kind.append(kind)
        self.lambdas.append(Float32(0.0))

    def place(mut self, index: Int, at: Vec3):
        """Puts a particle somewhere at rest (a reset)."""
        if index < 0 or index >= len(self.position):
            return
        self.position[index] = at
        self.previous[index] = at
        self.velocity[index] = Vec3()
        self.target[index] = at

    def pin(mut self, index: Int, at: Vec3):
        """Holds a particle at a point (inverse mass 0); call each frame."""
        if index < 0 or index >= len(self.position):
            return
        self.inv_mass[index] = 0
        self.position[index] = at
        self.velocity[index] = Vec3()

    def push(mut self, index: Int, impulse_velocity: Vec3):
        if index >= 0 and index < len(self.position) and self.inv_mass[index] > 0:
            self.velocity[index] = self.velocity[index] + impulse_velocity

    def mean_speed(self) -> Float32:
        var total = Float32(0)
        var n = 0
        for i in range(len(self.velocity)):
            if self.inv_mass[i] > 0:
                total += self.velocity[i].length()
                n += 1
        return total / Float32(n) if n > 0 else Float32(0)

    def link_error(self) -> Float32:
        """The largest relative stretch of an equal link."""
        var worst = Float32(0)
        for k in range(len(self.link_a)):
            if self.link_kind[k] != LINK_EQUAL or self.rest[k] < 1.0e-5:
                continue
            var d = (self.position[self.link_b[k]] - self.position[self.link_a[k]]).length()
            worst = max(worst, abs(d - self.rest[k]) / self.rest[k])
        return worst

    def _ensure_lambdas(mut self):
        var n_links = len(self.link_a)
        if len(self.lambdas) < n_links:
            self.lambdas.resize(n_links, Float32(0.0))
        for k in range(n_links):
            self.lambdas[k] = Float32(0.0)

    def step(
        mut self,
        dt: Float32,
        substeps: Int,
        gravity: Float32,
        wind: List[Vec3],
        colliders: List[ChainCollider],
        iterations: Int = 2,
    ):
        """Advances by `dt` in `substeps` small steps with `iterations` XPBD Gauss-Seidel passes."""
        var dummy_atts = List[RigidAttachment]()
        var dummy_bodies = SolverBodies()
        self.step_coupled(dt, substeps, gravity, wind, colliders, dummy_atts, dummy_bodies, iterations)

    def step_substep_coupled(
        mut self,
        h: Float32,
        sub_t: Float32,
        gravity: Float32,
        wind: List[Vec3],
        colliders: List[ChainCollider],
        mut attachments: List[RigidAttachment],
        mut bodies: SolverBodies,
        iterations: Int = 2,
    ):
        """Advances one XPBD mesh substep of size `h` with two-way `RigidAttachment` impulse feedback."""
        if not isfinite(h) or h <= Float32(1.0e-8):
            return
        var safe_h = min(h, Float32(0.1))
        var safe_sub_t = sub_t if isfinite(sub_t) else Float32(0.0)
        var safe_gravity = gravity if isfinite(gravity) else Float32(9.81)
        var iters = max(1, min(64, iterations))
        var inv_h2 = Float32(1.0) / (safe_h * safe_h)
        var g = safe_gravity * self.gravity_scale

        self._ensure_lambdas()
        for a_i in range(len(attachments)):
            attachments[a_i].lam = Float32(0.0)

        for i in range(len(self.position)):
            if not (isfinite(self.position[i].x) and isfinite(self.position[i].y) and isfinite(self.position[i].z)):
                self.position[i] = Vec3()
            if not (isfinite(self.velocity[i].x) and isfinite(self.velocity[i].y) and isfinite(self.velocity[i].z)):
                self.velocity[i] = Vec3()
            self.previous[i] = self.position[i]
            if self.inv_mass[i] == 0:
                continue
            var ac = self.accel[i]
            var safe_ac = ac if (isfinite(ac.x) and isfinite(ac.y) and isfinite(ac.z)) else Vec3()
            var v = self.velocity[i] + safe_ac * safe_h
            v.y -= g * safe_h
            if i < len(wind) and self.wind_drag > 0:
                var wv = wind[i]
                if isfinite(wv.x) and isfinite(wv.y) and isfinite(wv.z):
                    v = v + (wv - v) * min(Float32(1), self.wind_drag * safe_h)
            v = v * max(Float32(0), 1 - self.damping * safe_h)
            self.velocity[i] = v
            var p = self.position[i] + v * safe_h
            if self.power[i] > 0:
                p = p + (self.target[i] - p) * min(Float32(1), self.power[i] * safe_h)
            self.position[i] = p

        for _ in range(iters):
            for a_i in range(len(attachments)):
                _project_rigid_attachment(self.position, self.inv_mass, attachments[a_i], bodies, safe_h, safe_sub_t)

            for k in range(len(self.link_a)):
                var a = self.link_a[k]
                var b = self.link_b[k]
                var wa = self.inv_mass[a]
                var wb = self.inv_mass[b]
                var w_sum = wa + wb
                var alpha_tilde = self.link_compliance[k] * inv_h2
                var denom = w_sum + alpha_tilde
                if denom <= Float32(0.0) or w_sum <= Float32(0.0):
                    continue
                var d = self.position[b] - self.position[a]
                var length = d.length()
                if length < Float32(1.0e-6):
                    continue
                var c = length - self.rest[k]
                var lam = self.lambdas[k]
                if self.link_kind[k] == LINK_MIN and c >= Float32(0.0) and lam <= Float32(0.0):
                    continue
                var dl = (-c - alpha_tilde * lam) / denom
                if self.link_kind[k] == LINK_MIN:
                    var new_lam = max(Float32(0.0), lam + dl)
                    dl = new_lam - lam
                    self.lambdas[k] = new_lam
                else:
                    self.lambdas[k] = lam + dl
                var n_dir = d * (Float32(1.0) / length)
                self.position[a] = self.position[a] - n_dir * (dl * wa)
                self.position[b] = self.position[b] + n_dir * (dl * wb)

        self.touching = 0
        for i in range(len(self.position)):
            if self.inv_mass[i] == 0:
                continue
            for c in colliders:
                if _particle_attached_to_collider(i, c, attachments):
                    continue
                var p_i = self.position[i]
                if _collide_particle_with_collider(p_i, self.radius[i], self.inv_mass[i], c, bodies, safe_h, safe_sub_t):
                    self.position[i] = p_i
            var ground = self.floor[i] + self.radius[i]
            var on_floor = False
            if self.position[i].y < ground:
                self.position[i].y = ground
                on_floor = True
                self.touching += 1
            var v = (self.position[i] - self.previous[i]) * (Float32(1.0) / safe_h)
            if on_floor:
                var keep = max(Float32(0), 1 - self.friction)
                v = Vec3(v.x * keep, max(Float32(0), v.y), v.z * keep)
                self.position[i] = self.previous[i] + v * safe_h
                if self.position[i].y < ground:
                    self.position[i].y = ground
            self.velocity[i] = v

    def step_coupled(
        mut self,
        dt: Float32,
        substeps: Int,
        gravity: Float32,
        wind: List[Vec3],
        colliders: List[ChainCollider],
        mut attachments: List[RigidAttachment],
        mut bodies: SolverBodies,
        iterations: Int = 2,
    ):
        """Advances the mesh with two-way `RigidAttachment` impulse feedback into `SolverBodies`."""
        if not isfinite(dt) or dt <= Float32(1.0e-8):
            return
        var n = max(1, min(64, substeps))
        var h = min(dt, Float32(1.0)) / Float32(n)
        self.steps_run += 1

        for sub in range(n):
            self.step_substep_coupled(
                h, Float32(sub) * h, gravity, wind, colliders, attachments, bodies, iterations
            )

    def serialize(self, mut w: BlobWriter):
        """Bit-exact snapshot of XPBD `ParticleMesh` including `lambdas`."""
        w.begin(XPBD_MESH_TAG, XPBD_BLOB_VERSION)
        var n_p = len(self.position)
        w.integer(n_p)
        for i in range(n_p):
            w.vec3(self.position[i])
            w.vec3(self.previous[i])
            w.vec3(self.velocity[i])
            w.f32(self.inv_mass[i])
            w.f32(self.radius[i])
            w.f32(self.floor[i])
            w.vec3(self.target[i])
            w.f32(self.power[i])
            w.vec3(self.accel[i])
        var n_l = len(self.link_a)
        w.integer(n_l)
        for k in range(n_l):
            w.integer(self.link_a[k])
            w.integer(self.link_b[k])
            w.f32(self.rest[k])
            w.f32(self.link_compliance[k])
            w.integer(self.link_kind[k])
        var n_lam = len(self.lambdas)
        w.integer(n_lam)
        for k in range(n_lam):
            w.f32(self.lambdas[k])
        w.f32(self.damping)
        w.f32(self.wind_drag)
        w.f32(self.friction)
        w.f32(self.gravity_scale)
        w.integer(self.touching)
        w.integer(self.steps_run)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        """Restores bit-exact `ParticleMesh` state from `BlobReader`."""
        _ = r.enter(XPBD_MESH_TAG, XPBD_BLOB_VERSION)
        var n_p = r.integer()
        if r.failed or n_p < 0 or n_p > 1 << 20 or n_p > max(0, r.limit() - r.offset) // 76:
            raise Error("xpbd mesh blob: invalid particle count")
        self.position.clear()
        self.previous.clear()
        self.velocity.clear()
        self.inv_mass.clear()
        self.radius.clear()
        self.floor.clear()
        self.target.clear()
        self.power.clear()
        self.accel.clear()
        for _ in range(n_p):
            self.position.append(r.vec3())
            self.previous.append(r.vec3())
            self.velocity.append(r.vec3())
            self.inv_mass.append(r.f32())
            self.radius.append(r.f32())
            self.floor.append(r.f32())
            self.target.append(r.vec3())
            self.power.append(r.f32())
            self.accel.append(r.vec3())
        var n_l = r.integer()
        if r.failed or n_l < 0 or n_l > 1 << 21 or n_l > max(0, r.limit() - r.offset) // 32:
            raise Error("xpbd mesh blob: invalid link count")
        self.link_a.clear()
        self.link_b.clear()
        self.rest.clear()
        self.link_compliance.clear()
        self.link_kind.clear()
        for _ in range(n_l):
            var la = r.integer()
            var lb = r.integer()
            if la < 0 or la >= n_p or lb < 0 or lb >= n_p:
                raise Error("xpbd mesh blob: link particle index out of range")
            self.link_a.append(la)
            self.link_b.append(lb)
            self.rest.append(r.f32())
            self.link_compliance.append(r.f32())
            self.link_kind.append(r.integer())
        var n_lam = r.integer()
        if r.failed or n_lam < 0 or n_lam > 1 << 21 or n_lam > max(0, r.limit() - r.offset) // 4:
            raise Error("xpbd mesh blob: invalid lambda count")
        self.lambdas.clear()
        for _ in range(n_lam):
            self.lambdas.append(r.f32())
        self.damping = r.f32()
        self.wind_drag = r.f32()
        self.friction = r.f32()
        self.gravity_scale = r.f32()
        self.touching = r.integer()
        self.steps_run = r.integer()
        r.leave()
