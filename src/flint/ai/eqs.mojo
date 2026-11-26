"""Environment Queries (EQS-style point queries) & NavMesh Cover Points (`flint.ai.eqs`, v5).

`EnvQuery` generates candidate points (grid, rings, or navmesh-sampled), optionally projects them
onto the navmesh, then runs weighted tests:
- `EQS_TEST_DISTANCE`: distance to a reference point (prefer near or far),
- `EQS_TEST_LOS`: physics line of sight from a reference eye (filter or score visible / hidden),
- `EQS_TEST_PATH_LENGTH`: navmesh path length from a reference point (unreachable points dropped),
- `EQS_TEST_DOT`: alignment of (point - ref) with a direction,
- `EQS_TEST_CUSTOM`: per-point scores supplied by the game (`set_custom_scores`).
Each scoring test min-max normalises its raw values over surviving items; the item score is the
weighted sum. `generate_cover_points` samples navmesh boundary edges backed by physics walls.
"""
from std.collections import List
from std.math import sqrt, min, max, isfinite
from flint.core.math import Vec3
from flint.core.detmath import det_sin, det_cos, PI
from flint.physics.filter import CollisionFilter, CAT_ALL
from flint.physics.world import PhysicsWorld
from flint.nav.navmesh import NavMesh

comptime EQS_GEN_GRID: Int = 0
comptime EQS_GEN_RING: Int = 1
comptime EQS_GEN_NAVMESH: Int = 2

comptime EQS_TEST_DISTANCE: Int = 0
comptime EQS_TEST_LOS: Int = 1
comptime EQS_TEST_PATH_LENGTH: Int = 2
comptime EQS_TEST_DOT: Int = 3
comptime EQS_TEST_CUSTOM: Int = 4


struct EqsTest(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var ref_point: Vec3
    var direction: Vec3
    var weight: Float32
    var prefer_high: Bool  # score high raw values (far / long / visible / aligned)
    var filter_only: Bool  # discard failing items instead of scoring (LOS)
    var want_visible: Bool
    var eye_height: Float32
    var custom_slot: Int

    def __init__(out self, kind: Int, ref_point: Vec3 = Vec3(), weight: Float32 = 1.0, prefer_high: Bool = False):
        self.kind = kind
        self.ref_point = ref_point
        self.direction = Vec3(0.0, 0.0, 1.0)
        self.weight = weight
        self.prefer_high = prefer_high
        self.filter_only = False
        self.want_visible = True
        self.eye_height = 1.6
        self.custom_slot = -1


@fieldwise_init
struct EqsItem(Copyable, ImplicitlyCopyable, Movable):
    var pos: Vec3
    var score: Float32
    var valid: Bool
    var poly: Int


struct EqsResult(Copyable, Movable):
    var items: List[EqsItem]
    var best_index: Int

    def __init__(out self):
        self.items = List[EqsItem]()
        self.best_index = -1

    def found(self) -> Bool:
        return self.best_index >= 0

    def best(self) -> Vec3:
        if self.best_index < 0:
            return Vec3()
        return self.items[self.best_index].pos

    def valid_count(self) -> Int:
        var n = 0
        for i in range(len(self.items)):
            if self.items[i].valid:
                n += 1
        return n


struct EnvQuery(Copyable, Movable):
    var gen_kind: Int
    var center: Vec3
    var radius: Float32  # grid half-extent / outer ring radius / navmesh sample radius
    var inner_radius: Float32
    var spacing: Float32
    var rings: Int
    var points_per_ring: Int
    var project_to_navmesh: Bool
    var tests: List[EqsTest]
    var custom_scores: List[List[Float32]]
    var los_filter: CollisionFilter
    var target_height: Float32

    def __init__(out self, gen_kind: Int = EQS_GEN_GRID, center: Vec3 = Vec3(), radius: Float32 = 5.0, spacing: Float32 = 1.0):
        self.gen_kind = gen_kind
        self.center = center
        self.radius = radius
        self.inner_radius = 0.0
        self.spacing = max(Float32(0.05), spacing)
        self.rings = 2
        self.points_per_ring = 12
        self.project_to_navmesh = True
        self.tests = List[EqsTest]()
        self.custom_scores = List[List[Float32]]()
        self.los_filter = CollisionFilter(CAT_ALL, CAT_ALL, 0)
        self.target_height = 1.0

    @staticmethod
    def grid(center: Vec3, half_extent: Float32, spacing: Float32) -> EnvQuery:
        return EnvQuery(EQS_GEN_GRID, center, half_extent, spacing)

    @staticmethod
    def ring(center: Vec3, inner_radius: Float32, outer_radius: Float32, rings: Int = 2, points_per_ring: Int = 12) -> EnvQuery:
        var q = EnvQuery(EQS_GEN_RING, center, outer_radius, 1.0)
        q.inner_radius = inner_radius
        q.rings = max(1, rings)
        q.points_per_ring = max(3, points_per_ring)
        return q^

    @staticmethod
    def navmesh_points(center: Vec3, radius: Float32, spacing: Float32) -> EnvQuery:
        return EnvQuery(EQS_GEN_NAVMESH, center, radius, spacing)

    def add_distance_test(mut self, point: Vec3, weight: Float32 = 1.0, prefer_far: Bool = False) -> Int:
        self.tests.append(EqsTest(EQS_TEST_DISTANCE, point, weight, prefer_far))
        return len(self.tests) - 1

    def add_los_test(mut self, eye: Vec3, want_visible: Bool = True, filter_only: Bool = True, weight: Float32 = 1.0) -> Int:
        var t = EqsTest(EQS_TEST_LOS, eye, weight, True)
        t.want_visible = want_visible
        t.filter_only = filter_only
        self.tests.append(t)
        return len(self.tests) - 1

    def add_path_length_test(mut self, from_point: Vec3, weight: Float32 = 1.0, prefer_short: Bool = True) -> Int:
        self.tests.append(EqsTest(EQS_TEST_PATH_LENGTH, from_point, weight, not prefer_short))
        return len(self.tests) - 1

    def add_dot_test(mut self, origin: Vec3, direction: Vec3, weight: Float32 = 1.0, prefer_aligned: Bool = True) -> Int:
        var t = EqsTest(EQS_TEST_DOT, origin, weight, prefer_aligned)
        t.direction = direction.normalized()
        self.tests.append(t)
        return len(self.tests) - 1

    def add_custom_test(mut self, weight: Float32 = 1.0, prefer_high: Bool = True) -> Int:
        var t = EqsTest(EQS_TEST_CUSTOM, Vec3(), weight, prefer_high)
        t.custom_slot = len(self.custom_scores)
        self.custom_scores.append(List[Float32]())
        self.tests.append(t)
        return t.custom_slot

    def set_custom_scores(mut self, slot: Int, scores: List[Float32]):
        """Raw per-item values for a custom test, aligned with `generate(nav)` output order."""
        if slot >= 0 and slot < len(self.custom_scores):
            self.custom_scores[slot] = scores.copy()

    def generate(self, nav: NavMesh) -> List[EqsItem]:
        var raw = List[Vec3]()
        if self.gen_kind == EQS_GEN_RING:
            for r in range(self.rings):
                var rad = self.radius if self.rings == 1 else self.inner_radius + (self.radius - self.inner_radius) * Float32(r) / Float32(self.rings - 1)
                for k in range(self.points_per_ring):
                    var a = Float32(2.0) * PI * Float32(k) / Float32(self.points_per_ring)
                    raw.append(self.center + Vec3(det_sin(a) * rad, 0.0, det_cos(a) * rad))
        else:
            var n = Int(self.radius / self.spacing)
            for iz in range(-n, n + 1):
                for ix in range(-n, n + 1):
                    var off = Vec3(Float32(ix) * self.spacing, 0.0, Float32(iz) * self.spacing)
                    if self.gen_kind == EQS_GEN_NAVMESH and off.length_squared() > self.radius * self.radius:
                        continue
                    raw.append(self.center + off)
        var items = List[EqsItem](capacity=len(raw))
        var project = self.project_to_navmesh or self.gen_kind == EQS_GEN_NAVMESH
        for i in range(len(raw)):
            if project and nav.poly_count() > 0:
                var near = nav.find_nearest_poly(raw[i], self.spacing * Float32(0.5) + Float32(0.05))
                if near[0] < 0:
                    continue
                items.append(EqsItem(near[1], 0.0, True, near[0]))
            else:
                items.append(EqsItem(raw[i], 0.0, True, -1))
        return items^

    def score(self, mut items: List[EqsItem], nav: NavMesh, world: PhysicsWorld) -> Int:
        """Runs all tests over `items`; returns the best valid index (first wins ties) or -1."""
        var n = len(items)
        var raw = List[Float32](length=n, fill=Float32(0.0))
        for ti in range(len(self.tests)):
            var t = self.tests[ti]
            for i in range(n):
                if not items[i].valid:
                    continue
                var p = items[i].pos
                if t.kind == EQS_TEST_DISTANCE:
                    raw[i] = (p - t.ref_point).length()
                elif t.kind == EQS_TEST_DOT:
                    var d = p - t.ref_point
                    d.y = 0.0
                    var l = d.length()
                    raw[i] = d.dot(t.direction) / l if l > Float32(1.0e-5) else Float32(0.0)
                elif t.kind == EQS_TEST_PATH_LENGTH:
                    var path = nav.find_path(t.ref_point, p)
                    if not path.found:
                        items[i].valid = False
                        continue
                    raw[i] = path.total_length
                elif t.kind == EQS_TEST_LOS:
                    var tgt = p + Vec3(0.0, self.target_height, 0.0)
                    var dvec = tgt - t.ref_point
                    var dist = dvec.length()
                    var visible = True
                    if dist > Float32(1.0e-4):
                        var hit = world.raycast(t.ref_point, dvec * (Float32(1.0) / dist), dist, self.los_filter)
                        visible = (not hit.hit) or hit.distance >= dist - Float32(0.05)
                    var good = visible == t.want_visible
                    if t.filter_only:
                        if not good:
                            items[i].valid = False
                        continue
                    raw[i] = Float32(1.0) if good else Float32(0.0)
                else:
                    var slot = t.custom_slot
                    if slot >= 0 and slot < len(self.custom_scores) and i < len(self.custom_scores[slot]):
                        raw[i] = self.custom_scores[slot][i]
                    else:
                        raw[i] = 0.0
            if t.kind == EQS_TEST_LOS and t.filter_only:
                continue
            # Min-max normalise over valid items and accumulate weighted score
            var lo = Float32(1.0e30)
            var hi = Float32(-1.0e30)
            for i in range(n):
                if items[i].valid:
                    lo = min(lo, raw[i])
                    hi = max(hi, raw[i])
            var span = hi - lo
            for i in range(n):
                if not items[i].valid:
                    continue
                var norm = (raw[i] - lo) / span if span > Float32(1.0e-6) else Float32(1.0)
                if not t.prefer_high:
                    norm = Float32(1.0) - norm if span > Float32(1.0e-6) else Float32(1.0)
                items[i].score += t.weight * norm
        var best = -1
        var best_s = Float32(-1.0e30)
        for i in range(n):
            if items[i].valid and items[i].score > best_s:
                best_s = items[i].score
                best = i
        return best

    def run(self, nav: NavMesh, world: PhysicsWorld) -> EqsResult:
        var res = EqsResult()
        res.items = self.generate(nav)
        res.best_index = self.score(res.items, nav, world)
        return res^


# =========================================================================
# Cover points from navmesh boundary edges
# =========================================================================

@fieldwise_init
struct CoverPoint(Copyable, ImplicitlyCopyable, Movable):
    var pos: Vec3  # standing position (pulled `agent_radius` inside the walkable edge)
    var wall_normal: Vec3  # unit XZ direction from the point toward the protecting wall
    var poly: Int
    var full_height: Bool  # wall still blocks at standing eye height (vs. low/crouch cover)


def generate_cover_points(
    nav: NavMesh,
    world: PhysicsWorld,
    spacing: Float32 = 1.0,
    agent_radius: Float32 = 0.4,
    crouch_height: Float32 = 0.8,
    stand_height: Float32 = 1.6,
    probe_dist: Float32 = 1.0,
) -> List[CoverPoint]:
    """Samples navmesh boundary edges (edges with no walkable polygon just outside) every `spacing`
    metres and keeps samples whose outward ray at `crouch_height` hits physics geometry."""
    var out = List[CoverPoint]()
    var filt = CollisionFilter(CAT_ALL, CAT_ALL, 0)
    var step = max(Float32(0.1), spacing)
    for pi in range(nav.poly_count()):
        var nv = len(nav.polys[pi].verts)
        for e in range(nv):
            var a = nav.polys[pi].verts[e]
            var b = nav.polys[pi].verts[(e + 1) % nv]
            var ex = b.x - a.x
            var ez = b.z - a.z
            var el = sqrt(ex * ex + ez * ez)
            if el < Float32(1.0e-3):
                continue
            # Outward normal for the CCW (+shoelace) polygon winding used by `NavMesh`
            var n_out = Vec3(ez / el, 0.0, -ex / el)
            var mid = (a + b) * Float32(0.5)
            var probe = mid + n_out * Float32(0.08)
            var outside = nav.find_nearest_poly(probe, Float32(0.03))
            if outside[0] >= 0:
                continue  # portal / interior edge
            var cnt = max(1, Int(el / step))
            for k in range(cnt):
                var t = (Float32(k) + Float32(0.5)) / Float32(cnt)
                var ep = a + (b - a) * t
                var stand = ep - n_out * agent_radius
                var origin_c = Vec3(stand.x, stand.y + crouch_height, stand.z)
                var hit = world.raycast(origin_c, n_out, agent_radius + probe_dist, filt)
                if not hit.hit:
                    continue
                var origin_s = Vec3(stand.x, stand.y + stand_height, stand.z)
                var hit_s = world.raycast(origin_s, n_out, agent_radius + probe_dist, filt)
                out.append(CoverPoint(stand, n_out, pi, hit_s.hit))
    return out^


def cover_protects(cp: CoverPoint, threat_pos: Vec3, min_dot: Float32 = 0.3) -> Bool:
    var d = threat_pos - cp.pos
    d.y = 0.0
    var l = d.length()
    if l < Float32(1.0e-4):
        return False
    return d.dot(cp.wall_normal) / l >= min_dot


def find_best_cover(
    points: List[CoverPoint], agent_pos: Vec3, threat_pos: Vec3, world: PhysicsWorld, crouch_height: Float32 = 0.8
) -> Int:
    """Nearest cover point (to `agent_pos`) whose wall faces the threat and blocks its line of sight."""
    var best = -1
    var best_d2 = Float32(1.0e30)
    var filt = CollisionFilter(CAT_ALL, CAT_ALL, 0)
    for i in range(len(points)):
        if not cover_protects(points[i], threat_pos):
            continue
        var c = points[i].pos + Vec3(0.0, crouch_height, 0.0)
        var dv = c - threat_pos
        var dist = dv.length()
        if dist < Float32(1.0e-3):
            continue
        var hit = world.raycast(threat_pos, dv * (Float32(1.0) / dist), dist, filt)
        if not hit.hit or hit.distance >= dist - Float32(0.05):
            continue
        var d2 = (points[i].pos - agent_pos).length_squared()
        if d2 < best_d2:
            best_d2 = d2
            best = i
    return best
