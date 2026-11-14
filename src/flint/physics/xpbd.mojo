"""The XPBD particle-chain core (package 2.12): cosmetic ropes now, cloth
proxies and ragdoll strands in 3.11.

Extended Position-Based Dynamics (Macklin, Müller and Chentanez 2016) with
small steps (Macklin et al. 2019): each substep predicts positions from
velocity, gravity and the wind's drag, then projects every distance
constraint once, Gauss-Seidel in chain order, with its compliance (alpha:
metres per newton, 0 is rigid) as alpha / h^2 and the multiplier started at
zero each substep, then derives velocity from the move. Pinned particles
(inverse mass 0) are set from outside each frame, as a rope's ends follow
the sim links. Collision is against spheres and a floor height: a particle
inside is moved to the surface and keeps its tangential motion.

Presentation only: it reads the wind mirror and the sim's poses, and never
writes the simulation, saves or checksums. Its own results are still
repeatable (fixed order, no randomness), so screenshots are stable.
"""
from std.math import sqrt
from flint.core.math import Vec3


struct ChainCollider(Copyable, ImplicitlyCopyable, Movable):
    var center: Vec3
    var radius: Float32

    def __init__(out self, center: Vec3, radius: Float32):
        self.center = center
        self.radius = radius


struct ParticleChain(Copyable, Movable):
    var position: List[Vec3]
    var previous: List[Vec3]
    var velocity: List[Vec3]
    var inv_mass: List[Float32]
    var rest: List[Float32]      # rest[i]: between particle i and i + 1
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

    def step(mut self, dt: Float32, substeps: Int, gravity: Float32, wind: List[Vec3], colliders: List[ChainCollider]):
        """Advances the chain by dt. `wind` is the air's velocity at each
        particle (empty: still air)."""
        var n = max(1, substeps)
        var h = dt / Float32(n)
        var alpha = self.compliance / (h * h)
        for _ in range(n):
            for i in range(len(self.position)):
                self.previous[i] = self.position[i]
                if self.inv_mass[i] == 0:
                    continue
                var v = self.velocity[i]
                v.y -= gravity * h
                if i < len(wind):
                    v = v + (wind[i] - v) * min(Float32(1), self.wind_drag * h)
                v = v * max(Float32(0), 1 - self.damping * h)
                self.velocity[i] = v
                self.position[i] = self.position[i] + v * h
            for i in range(len(self.rest)):
                var wa = self.inv_mass[i]
                var wb = self.inv_mass[i + 1]
                var sum = wa + wb + alpha
                if sum <= 0:
                    continue
                var d = self.position[i + 1] - self.position[i]
                var length = d.length()
                if length < 1.0e-6:
                    continue
                var c = length - self.rest[i]
                var lam = -c / sum
                var n_dir = d * (1.0 / length)
                self.position[i] = self.position[i] - n_dir * (lam * wa)
                self.position[i + 1] = self.position[i + 1] + n_dir * (lam * wb)
            for i in range(len(self.position)):
                if self.inv_mass[i] == 0:
                    continue
                for c in colliders:
                    var d = self.position[i] - c.center
                    var dist = d.length()
                    if dist < c.radius and dist > 1.0e-6:
                        self.position[i] = c.center + d * (c.radius / dist)
                if self.position[i].y < self.floor:
                    self.position[i].y = self.floor
                self.velocity[i] = (self.position[i] - self.previous[i]) * (1.0 / h)


# ---------------------------------------------------------------------------
# Package 3.11: the same core for any set of particles and links (ragdolls
# and cloth proxies). A link holds two particles at a rest distance with its
# own compliance; a `min` link only pushes apart (a limb that must not fold
# through itself). Each particle may be driven toward a target (a powered
# ragdoll follows the animation with `power` per second, 0 is limp), stands
# on its own floor height (the terrain under it, read once a frame) with
# friction, and is kept out of spheres. Same order every step: repeatable.
# ---------------------------------------------------------------------------

comptime LINK_EQUAL: Int = 0
comptime LINK_MIN: Int = 1


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
    var damping: Float32         # per second
    var wind_drag: Float32       # per second, toward the air's velocity
    var friction: Float32        # share of sliding speed a floor contact keeps per substep is (1 - friction)
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

    def place(mut self, index: Int, at: Vec3):
        """Puts a particle somewhere at rest (a reset)."""
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
        """The largest relative stretch of an equal link (tests)."""
        var worst = Float32(0)
        for k in range(len(self.link_a)):
            if self.link_kind[k] != LINK_EQUAL or self.rest[k] < 1.0e-5:
                continue
            var d = (self.position[self.link_b[k]] - self.position[self.link_a[k]]).length()
            worst = max(worst, abs(d - self.rest[k]) / self.rest[k])
        return worst

    def step(mut self, dt: Float32, substeps: Int, gravity: Float32, wind: List[Vec3], colliders: List[ChainCollider]):
        """Advances by dt in `substeps` small steps. `wind` is the air's
        velocity at each particle (empty: still air)."""
        var n = max(1, substeps)
        var h = dt / Float32(n)
        var g = gravity * self.gravity_scale
        self.steps_run += 1
        for _ in range(n):
            for i in range(len(self.position)):
                self.previous[i] = self.position[i]
                if self.inv_mass[i] == 0:
                    continue
                var v = self.velocity[i] + self.accel[i] * h
                v.y -= g * h
                if i < len(wind) and self.wind_drag > 0:
                    v = v + (wind[i] - v) * min(Float32(1), self.wind_drag * h)
                v = v * max(Float32(0), 1 - self.damping * h)
                self.velocity[i] = v
                var p = self.position[i] + v * h
                if self.power[i] > 0:
                    p = p + (self.target[i] - p) * min(Float32(1), self.power[i] * h)
                self.position[i] = p
            for k in range(len(self.link_a)):
                var a = self.link_a[k]
                var b = self.link_b[k]
                var wa = self.inv_mass[a]
                var wb = self.inv_mass[b]
                var alpha = self.link_compliance[k] / (h * h)
                var sum = wa + wb + alpha
                if sum <= 0:
                    continue
                var d = self.position[b] - self.position[a]
                var length = d.length()
                if length < 1.0e-6:
                    continue
                var c = length - self.rest[k]
                if self.link_kind[k] == LINK_MIN and c >= 0:
                    continue
                var lam = -c / sum
                var n_dir = d * (1.0 / length)
                self.position[a] = self.position[a] - n_dir * (lam * wa)
                self.position[b] = self.position[b] + n_dir * (lam * wb)
            self.touching = 0
            for i in range(len(self.position)):
                if self.inv_mass[i] == 0:
                    continue
                for c in colliders:
                    var d = self.position[i] - c.center
                    var dist = d.length()
                    var reach = c.radius + self.radius[i]
                    if dist < reach and dist > 1.0e-6:
                        self.position[i] = c.center + d * (reach / dist)
                var ground = self.floor[i] + self.radius[i]
                var on_floor = False
                if self.position[i].y < ground:
                    self.position[i].y = ground
                    on_floor = True
                    self.touching += 1
                var v = (self.position[i] - self.previous[i]) * (1.0 / h)
                if on_floor:
                    var keep = max(Float32(0), 1 - self.friction)
                    v = Vec3(v.x * keep, max(Float32(0), v.y), v.z * keep)
                    # The slide it lost is not kept in the position either.
                    self.position[i] = self.previous[i] + v * h
                    if self.position[i].y < ground:
                        self.position[i].y = ground
                self.velocity[i] = v
